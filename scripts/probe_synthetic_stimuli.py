"""Synthetic per-column radiance stimuli on the radiance path (FlyBrain.vision -> OpticLobe.step_frame), fly pinned,
no ray tracing: the retinal input IS the recorded input (retina.mode = 'synthetic_direct').

The matched visual assay Neurome asked for after the size-ladder intake (docs/NEUROME_INTERFACE.md 3b, item 1):
object centre elevation, angular trajectory, angular speed, contrast and background held constant across sizes,
separate height and width ladders matching the published rectangles (Keles & Frye 2017: preferred vertical extent
8.8 deg at width ~4.4 deg), a fixed-centre square ladder, bars / gratings / flicker / ON-OFF flashes as the
specificity battery, and a per-body receptive-field LOCALIZER (a 4.5-deg dark square flashed at every node of a
10-deg azimuth x elevation grid) that fixes each body's measurement region before the size runs.

Every stimulus is a function of the retina's column directions (`Retina.col_az_el`, the way scripts/probe_motion.py
synthesises its grating) returning (n_frames, n_col, 4) radiance in the room's units and background, with a matched
blank (the constant background). Per-column coverage of an object uses the SAME ommatidial sampling the room uses
(Retina.ray_directions: 7 rays, acceptance 4.5 deg, Gaussian weights), so a sub-column object dims a column by the
fraction of its acceptance kernel it covers, as it does in the scene.

    # geometry check, CPU, no connectome needed beyond the retina
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py preview --stimulus rect --width 4.4 --height 8.8
    # one arm pair (stimulus + matched blank) on the GPU
    python scripts/probe_synthetic_stimuli.py record --stimulus rect --width 4.4 --height 8.8 --contrast -0.995 --seed 0 --out out/synth/rect_dark_s0
    python scripts/probe_synthetic_stimuli.py record --stimulus localizer --optic gain_fb=0 --seed 0 --out out/synth/loc_fb0_s0
    # the batch (one cluster_run.py call; fetch a NAMED directory)
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py plan --out out/synth --name synth && sh out/synth/batch.sh
    # CPU: the RF map from the localizer runs, then the per-family summary read through the map
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py rfmap --runs "out/synth/loc_shipped_s*_nodes.npz" --csv out/synth/rfmap_shipped.csv --json out/interp/synth/rfmap_shipped.json
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py analyse --dir out/synth --rf-map out/synth/rfmap_shipped.csv --json out/interp/synth/families.json

Units and background. The background is the per-channel [UV, B, G, R] mean of the room's blank arm at the pinned
pose of the object sweep (BACKGROUND, from out/apply_object/ladder/d114_null_s0_retina.npz: mean over 1200 frames x
1466 columns); a dark object is `contrast` = -0.995 (the black ball's radiance is 0.0043-0.0052 of the wall's in
out/apply_object/ladder/d300_stim_s0_retina.npz), a bright one +0.995 (1.995 x background, the wall's own maximum is
2.0 x). Object radiance = background x (1 + contrast x coverage). Azimuth is + left, elevation + up (retina.py).

Nothing in flyverse/ is edited: the model is read through FlyBrain with the LIFParams / OpticParams overrides of the
common CLI (`--optic gain_fb=0` is the deterministic lobe).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyverse.interp import common                       # noqa: E402
from flyverse.interp.common import Result, to_jsonable   # noqa: E402

# ---------------------------------------------------------------------------------------------------- constants
#: per-channel [UV, B, G, R] mean radiance of the room's blank arm at the object sweep's pinned pose
#: (out/apply_object/ladder/d114_null_s0_retina.npz, mean over 1200 frames x 1466 columns; column SD 0.014 / 0.038 /
#: 0.038 / 0.038). The wall's per-column maximum there is 0.196 (2.0 x this mean), its minimum 0.005-0.03.
BACKGROUND = (0.0364, 0.0930, 0.0949, 0.0985)
#: the black ball's radiance / the wall's at the darkest column of out/apply_object/ladder/d300_stim_s0_retina.npz is
#: 0.0043-0.0052: Weber contrast -0.995. A bright object of +0.995 is the mirror image (1.995 x background).
DARK_CONTRAST = -0.995
FRAME_S = common.FRAME_MS / 1000.0
ACCEPTANCE_DEG, RAYS = 4.5, 7                            # retina.EyeGeometry defaults (the room's own sampling)
SELECTION = ["LC11", "LC10a", "T2", "T3", "Tm5Y", "TmY21", "Mi1"]   # Mi1: hex-annotated medulla reference for the RF map
QUANTITIES = ("drive_mv", "spike_count", "optic_dr")
SPIKING_Q, RATE_Q = "drive_mv", "optic_dr"
# ladders (deg): Keles & Frye 2017 style -- one dimension varied, the other fixed
HEIGHT_LADDER = {"width": 4.4, "heights": [2.2, 4.4, 8.8, 15.0, 22.0, 30.0]}
WIDTH_LADDER = {"height": 8.8, "widths": [2.2, 4.4, 8.8, 15.0, 22.0, 30.0]}
SQUARE_LADDER = [4.5, 8.8, 11.0, 15.0, 20.0, 30.0]
RETINA_MODE = "synthetic_direct"


# ---------------------------------------------------------------------------------------------------- sampling kernel
def ray_offsets(acceptance_deg: float = ACCEPTANCE_DEG, k: int = RAYS) -> tuple[np.ndarray, np.ndarray]:
    """(k, 2) angular offsets (deg: d_az, d_el) and (k,) weights of the per-column sample rays -- Retina.ray_directions'
    kernel (centre + a hexagon at acceptance / 2, weights exp(-(d / r)^2), normalised) in degrees."""
    k = int(k)
    if k <= 1:
        return np.zeros((1, 2)), np.ones(1)
    r = acceptance_deg / 2.0
    ang = np.linspace(0, 2 * np.pi, k - 1, endpoint=False)
    offs = np.concatenate([[[0.0, 0.0]], np.c_[r * np.cos(ang), r * np.sin(ang)]])
    w = np.exp(-0.5 * (np.hypot(offs[:, 0], offs[:, 1]) / r) ** 2 * 2.0)
    return offs, w / w.sum()


def sample_points(col_az_el: np.ndarray, acceptance_deg: float = ACCEPTANCE_DEG, k: int = RAYS):
    """(n_col, k) azimuths, (n_col, k) elevations (deg) and (k,) weights of every column's sample rays."""
    offs, w = ray_offsets(acceptance_deg, k)
    az = np.asarray(col_az_el)[:, 0][:, None] + offs[None, :, 0]
    el = np.asarray(col_az_el)[:, 1][:, None] + offs[None, :, 1]
    return az, el, w


def rect_coverage(az, el, w, centre_az, centre_el, width_deg, height_deg) -> np.ndarray:
    """(n_col,) fraction of each column's acceptance kernel inside an axis-aligned rectangle of the (az, el) plane
    (equirectangular: exact at elevation 0, the ladders' elevation)."""
    inside = (np.abs(az - centre_az) <= width_deg / 2.0) & (np.abs(el - centre_el) <= height_deg / 2.0)
    return (inside * w[None, :]).sum(1)


def radiance_from_coverage(cov: np.ndarray, contrast: float, background=BACKGROUND) -> np.ndarray:
    """(n_col, 4) = background x (1 + contrast x coverage)."""
    bg = np.asarray(background, np.float32)
    return (bg[None, :] * (1.0 + contrast * np.asarray(cov, np.float32)[:, None])).astype(np.float32)


def blank_frame(n_col: int, background=BACKGROUND) -> np.ndarray:
    return np.tile(np.asarray(background, np.float32)[None, :], (int(n_col), 1))


# ---------------------------------------------------------------------------------------------------- stimuli
@dataclass
class Stimulus:
    """A presented radiance sequence. Either `frames` (T, n_col, 4) or `pattern` (P, n_col, 4) + `index` (T,) (a
    piecewise-constant sequence: frame t = pattern[index[t]]); `blank` (n_col, 4) is the matched blank arm's constant
    frame; `track` holds per-frame arrays (object centre, size, visibility, level, node ...)."""
    name: str
    params: dict
    blank: np.ndarray
    dt_s: float = FRAME_S
    frames: np.ndarray | None = None
    pattern: np.ndarray | None = None
    index: np.ndarray | None = None
    track: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(len(self.frames) if self.frames is not None else len(self.index))

    @property
    def n_col(self) -> int:
        return int(self.blank.shape[0])

    def frame(self, t: int) -> np.ndarray:
        return self.frames[t] if self.frames is not None else self.pattern[int(self.index[t])]

    def presented(self) -> np.ndarray:
        """The full (T, n_col, 4) sequence (materialised from the pattern when stored that way)."""
        return self.frames if self.frames is not None else self.pattern[self.index]

    def arrays(self) -> dict:
        """What the run writes as <out>_radiance.npz: the radiance presented, frame by frame or as pattern + index."""
        out = {"blank": self.blank, "dt_s": np.float64(self.dt_s), "name": np.str_(self.name),
               "params": np.str_(json.dumps(to_jsonable(self.params)))}
        if self.frames is not None:
            out["radiance"] = self.frames
        else:
            out["pattern"] = self.pattern; out["index"] = self.index
        out.update({f"track__{k}": np.asarray(v) for k, v in self.track.items()})
        return out


def triangle_azimuth(t_s: np.ndarray, half_span_deg: float, speed_deg_s: float, start_deg: float | None = None) -> np.ndarray:
    """Azimuth (deg) of an object translating at constant |angular speed| between -half_span and +half_span, left -> right
    first (starting at +half_span, azimuth + = left), reversing at the ends: a triangle wave in ANGLE, so the angular
    speed is constant (the object sweep's lateral triangle wave in metres was 45.8 deg/s at the centre and 18.8 at
    the ends)."""
    span = 2.0 * half_span_deg
    if span <= 0 or speed_deg_s <= 0:
        return np.full(len(t_s), 0.0 if start_deg is None else start_deg)
    period = 2.0 * span / speed_deg_s
    phase = (np.asarray(t_s, np.float64) / period) % 1.0
    frac = np.where(phase < 0.5, phase * 2.0, 2.0 - phase * 2.0)            # 0 -> 1 -> 0
    return half_span_deg - span * frac


def rectangle(col_az_el, seconds: float = 3.0, width_deg: float = 4.4, height_deg: float = 8.8, contrast: float = DARK_CONTRAST,
              speed_deg_s: float = 40.0, elevation_deg: float = 0.0, half_span_deg: float = 30.0, background=BACKGROUND,
              acceptance_deg: float = ACCEPTANCE_DEG, rays: int = RAYS, dt_s: float = FRAME_S, name: str = "rect") -> Stimulus:
    """A width x height rectangle (deg) of Weber contrast `contrast` translating in azimuth at `speed_deg_s` at a fixed
    elevation, between -half_span and +half_span (triangle wave in angle), over `seconds`."""
    n = int(round(seconds / dt_s))
    t = np.arange(n) * dt_s
    az_c = triangle_azimuth(t, half_span_deg, speed_deg_s)
    az, el, w = sample_points(col_az_el, acceptance_deg, rays)
    frames = np.empty((n, len(col_az_el), 4), np.float32)
    for k in range(n):
        frames[k] = radiance_from_coverage(rect_coverage(az, el, w, az_c[k], elevation_deg, width_deg, height_deg), contrast, background)
    params = {"seconds": seconds, "width_deg": width_deg, "height_deg": height_deg, "contrast": contrast, "speed_deg_s": speed_deg_s,
              "elevation_deg": elevation_deg, "half_span_deg": half_span_deg, "background": list(background),
              "acceptance_deg": acceptance_deg, "rays": rays, "trajectory": "triangle wave in azimuth, constant |angular speed|, starting at +half_span (left)"}
    track = {"t_s": t, "az_deg": az_c, "el_deg": np.full(n, elevation_deg), "width_deg": np.full(n, width_deg),
             "height_deg": np.full(n, height_deg), "visible": np.ones(n, bool)}
    return Stimulus(name, params, blank_frame(len(col_az_el), background), dt_s, frames=frames, track=track)


def bar(col_az_el, seconds: float = 3.0, width_deg: float = 7.0, contrast: float = DARK_CONTRAST, speed_deg_s: float = 40.0,
        half_span_deg: float = 30.0, background=BACKGROUND, acceptance_deg=ACCEPTANCE_DEG, rays=RAYS, dt_s=FRAME_S) -> Stimulus:
    """A full-height vertical bar (height 360 deg: every elevation) translating like the rectangles."""
    s = rectangle(col_az_el, seconds, width_deg, 360.0, contrast, speed_deg_s, 0.0, half_span_deg, background, acceptance_deg, rays, dt_s, name="bar")
    s.params["height_deg"] = "full"
    return s


def grating(col_az_el, seconds: float = 3.0, period_deg: float = 30.0, contrast: float = 0.5, speed_deg_s: float = 40.0,
            direction: str = "front->back", background=BACKGROUND, dt_s: float = FRAME_S) -> Stimulus:
    """scripts/probe_motion.py's sine grating in the room's units: radiance = background x (1 + contrast sin(2 pi (x - s v t)
    / period)), x = |azimuth| for front->back / back->front (increasing |az| on each eye), elevation for up / down."""
    n = int(round(seconds / dt_s)); t = np.arange(n) * dt_s
    az, el = np.asarray(col_az_el)[:, 0], np.asarray(col_az_el)[:, 1]
    if direction in ("front->back", "back->front"):
        x = np.abs(az); sgn = 1 if direction == "front->back" else -1
    elif direction in ("up", "down"):
        x = el; sgn = 1 if direction == "up" else -1
    else:
        raise ValueError(f"unknown grating direction {direction!r}")
    bg = np.asarray(background, np.float32)
    frames = np.empty((n, len(az), 4), np.float32)
    for k in range(n):
        phase = 2 * np.pi * (x - sgn * speed_deg_s * t[k]) / period_deg
        frames[k] = bg[None, :] * (1.0 + contrast * np.sin(phase))[:, None]
    params = {"seconds": seconds, "period_deg": period_deg, "contrast": contrast, "speed_deg_s": speed_deg_s, "direction": direction,
              "background": list(background), "formula": "probe_motion.grating in the room's units"}
    return Stimulus("grating", params, blank_frame(len(az), background), dt_s, frames=frames,
                    track={"t_s": t, "phase_shift_deg": sgn * speed_deg_s * t})


def flicker(col_az_el, seconds: float = 3.0, hz: float = 2.0, contrast: float = 0.5, background=BACKGROUND, dt_s: float = FRAME_S) -> Stimulus:
    """Full-field square-wave flicker: background x (1 + contrast) for the first half of each period, x (1 - contrast) for
    the second (time-mean = the background, the matched blank)."""
    n = int(round(seconds / dt_s)); t = np.arange(n) * dt_s
    level = np.where(((t * hz) % 1.0) < 0.5, 1.0 + contrast, 1.0 - contrast).astype(np.float32)
    bg = np.asarray(background, np.float32)
    frames = (level[:, None, None] * bg[None, None, :]).repeat(len(col_az_el), axis=1).astype(np.float32)
    params = {"seconds": seconds, "hz": hz, "contrast": contrast, "background": list(background), "waveform": "square, bright half first"}
    return Stimulus("flicker", params, blank_frame(len(col_az_el), background), dt_s, frames=frames, track={"t_s": t, "level": level})


def flash(col_az_el, seconds: float = 3.0, size_deg: float = 8.8, contrast: float = DARK_CONTRAST, az_deg: float = 0.0, el_deg: float = 0.0,
          on_s: float = 0.5, period_s: float = 1.5, first_s: float = 0.5, background=BACKGROUND, acceptance_deg=ACCEPTANCE_DEG,
          rays=RAYS, dt_s: float = FRAME_S) -> Stimulus:
    """A stationary square of `size_deg` at (az, el) that appears for `on_s` every `period_s` from `first_s`: each
    appearance is an isolated transition (bright square: ON at onset, OFF at offset; dark square: the reverse)."""
    n = int(round(seconds / dt_s)); t = np.arange(n) * dt_s
    az, el, w = sample_points(col_az_el, acceptance_deg, rays)
    obj = radiance_from_coverage(rect_coverage(az, el, w, az_deg, el_deg, size_deg, size_deg), contrast, background)
    blank = blank_frame(len(col_az_el), background)
    since = (t - first_s) % period_s
    visible = (t >= first_s - 1e-9) & (since < on_s - 1e-9)
    index = visible.astype(np.int64)
    params = {"seconds": seconds, "size_deg": size_deg, "contrast": contrast, "az_deg": az_deg, "el_deg": el_deg, "on_s": on_s,
              "period_s": period_s, "first_s": first_s, "background": list(background), "acceptance_deg": acceptance_deg, "rays": rays,
              "transition": "ON at onset / OFF at offset" if contrast > 0 else "OFF at onset / ON at offset"}
    return Stimulus("flash_on" if contrast > 0 else "flash_off", params, blank, dt_s, pattern=np.stack([blank, obj]), index=index,
                    track={"t_s": t, "visible": visible, "az_deg": np.full(n, az_deg), "el_deg": np.full(n, el_deg),
                           "width_deg": np.full(n, size_deg), "height_deg": np.full(n, size_deg)})


def localizer_grid(col_az_el, spacing_deg: float = 10.0, size_deg: float = 4.5, az_range=(-120.0, 120.0), el_range=(-70.0, 70.0),
                   acceptance_deg: float = ACCEPTANCE_DEG) -> np.ndarray:
    """(n_nodes, 2) azimuth / elevation of the grid nodes that reach at least one column (a column centre within
    size / 2 + acceptance / 2 of the node; the others would present nothing)."""
    azs = np.arange(az_range[0], az_range[1] + 1e-9, spacing_deg); els = np.arange(el_range[0], el_range[1] + 1e-9, spacing_deg)
    c = np.asarray(col_az_el)
    reach = size_deg / 2.0 + acceptance_deg / 2.0
    nodes = []
    for e in els:
        for a in azs:
            d = np.hypot(c[:, 0] - a, c[:, 1] - e)
            if d.min() <= reach:
                nodes.append((float(a), float(e)))
    return np.asarray(nodes, np.float64).reshape(-1, 2)


def localizer(col_az_el, spacing_deg: float = 10.0, size_deg: float = 4.5, contrast: float = DARK_CONTRAST, flash_s: float = 0.2,
              blank_s: float = 0.3, order_seed: int = 0, az_range=(-120.0, 120.0), el_range=(-70.0, 70.0), background=BACKGROUND,
              acceptance_deg=ACCEPTANCE_DEG, rays=RAYS, dt_s: float = FRAME_S) -> Stimulus:
    """The receptive-field localizer: a `size_deg` square of `contrast` flashed for `flash_s` at every node of an
    azimuth x elevation grid (spacing `spacing_deg`), `blank_s` of background between flashes and before the first,
    in a fixed shuffled order (`order_seed`; the same for every run). Stored as pattern (node frames + the blank) and
    a per-frame index; `track['node']` is the node presented (-1 = blank), `track['on']` the flash frames."""
    nodes = localizer_grid(col_az_el, spacing_deg, size_deg, az_range, el_range, acceptance_deg)
    az, el, w = sample_points(col_az_el, acceptance_deg, rays)
    blank = blank_frame(len(col_az_el), background)
    pattern = np.empty((len(nodes) + 1, len(col_az_el), 4), np.float32)
    pattern[0] = blank
    for i, (a, e) in enumerate(nodes):
        pattern[i + 1] = radiance_from_coverage(rect_coverage(az, el, w, a, e, size_deg, size_deg), contrast, background)
    order = np.random.RandomState(order_seed).permutation(len(nodes))
    n_on, n_off = int(round(flash_s / dt_s)), int(round(blank_s / dt_s))
    index, node, on = [], [], []
    for i in order:
        index += [0] * n_off + [i + 1] * n_on
        node += [-1] * n_off + [int(i)] * n_on
        on += [False] * n_off + [True] * n_on
    index += [0] * n_off; node += [-1] * n_off; on += [False] * n_off       # a closing blank: the last node's offset window
    n = len(index)
    params = {"spacing_deg": spacing_deg, "size_deg": size_deg, "contrast": contrast, "flash_s": flash_s, "blank_s": blank_s,
              "order_seed": order_seed, "az_range": list(az_range), "el_range": list(el_range), "n_nodes": int(len(nodes)),
              "seconds": n * dt_s, "background": list(background), "acceptance_deg": acceptance_deg, "rays": rays}
    return Stimulus("localizer", params, blank, dt_s, pattern=pattern, index=np.asarray(index, np.int64),
                    track={"t_s": np.arange(n) * dt_s, "node": np.asarray(node, np.int64), "on": np.asarray(on, bool),
                           "node_az_deg": nodes[:, 0], "node_el_deg": nodes[:, 1], "order": order.astype(np.int64)})


FAMILIES = {"rect": rectangle, "bar": bar, "grating": grating, "flicker": flicker, "flash": flash, "localizer": localizer}


def make_stimulus(kind: str, col_az_el, args) -> Stimulus:
    """The stimulus of a `record` / `preview` command line."""
    bg = tuple(args.background) if args.background else BACKGROUND
    if kind == "rect":
        return rectangle(col_az_el, args.seconds, args.width, args.height, args.contrast, args.speed, args.elevation, args.half_span, bg, rays=args.rays)
    if kind == "bar":
        return bar(col_az_el, args.seconds, args.width, args.contrast, args.speed, args.half_span, bg, rays=args.rays)
    if kind == "grating":
        return grating(col_az_el, args.seconds, args.period, abs(args.contrast) if args.contrast_set else 0.5, args.speed, args.direction, bg)
    if kind == "flicker":
        return flicker(col_az_el, args.seconds, args.hz, abs(args.contrast) if args.contrast_set else 0.5, bg)
    if kind == "flash":
        return flash(col_az_el, args.seconds, args.width, args.contrast, args.azimuth, args.elevation, args.on_s, args.period_s, bg, rays=args.rays)
    if kind == "localizer":
        return localizer(col_az_el, args.grid_spacing, args.width if args.width_set else 4.5, args.contrast, args.flash_s, args.blank_s,
                         args.order_seed, background=bg, rays=args.rays)
    raise ValueError(f"unknown stimulus {kind!r}; choose from {list(FAMILIES)}")


def stimulus_args(ap):
    ap.add_argument("--stimulus", required=True, choices=list(FAMILIES))
    ap.add_argument("--seconds", type=float, default=3.0, help="stimulus window (s); the localizer sets its own")
    ap.add_argument("--width", type=float, default=4.4, help="rect / bar width, flash and localizer square size (deg)")
    ap.add_argument("--height", type=float, default=8.8, help="rect height (deg)")
    ap.add_argument("--contrast", type=float, default=DARK_CONTRAST, help="Weber contrast of the object (dark < 0 < bright); grating / flicker use |contrast|, default 0.5")
    ap.add_argument("--speed", type=float, default=40.0, help="angular speed (deg/s) of rect / bar / grating")
    ap.add_argument("--elevation", type=float, default=0.0, help="centre elevation (deg) of rect / flash")
    ap.add_argument("--azimuth", type=float, default=0.0, help="flash centre azimuth (deg)")
    ap.add_argument("--half-span", type=float, default=30.0, help="rect / bar sweep: azimuth -half_span .. +half_span (deg)")
    ap.add_argument("--period", type=float, default=30.0, help="grating period (deg)")
    ap.add_argument("--direction", default="front->back", choices=["front->back", "back->front", "up", "down"])
    ap.add_argument("--hz", type=float, default=2.0, help="flicker frequency")
    ap.add_argument("--on-s", type=float, default=0.5, help="flash duration (s)")
    ap.add_argument("--period-s", type=float, default=1.5, help="flash period (s)")
    ap.add_argument("--grid-spacing", type=float, default=10.0, help="localizer grid spacing (deg)")
    ap.add_argument("--flash-s", type=float, default=0.2); ap.add_argument("--blank-s", type=float, default=0.3)
    ap.add_argument("--order-seed", type=int, default=0, help="localizer node order (fixed across runs)")
    ap.add_argument("--rays", type=int, default=RAYS, help="sample rays per column (7 = the room's; more = finer sub-column coverage)")
    ap.add_argument("--background", type=float, nargs=4, default=None, metavar="R", help="[UV B G R] background radiance (default BACKGROUND)")


def _flag_set(argv_or_ns, flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in (sys.argv[1:] if argv_or_ns is None else argv_or_ns))


# ---------------------------------------------------------------------------------------------------- GPU record
def build_fb(c, lif, op, seed: int, device=None):
    """A pinned FlyBrain (no room, no body): the room's LIF settings (dt 0.5, prune_frozen, native CUDA kernels and warp
    CSR on CUDA as scripts/probe_object_sweep.py's Sim; event_driven on CUDA as there)."""
    import torch
    from flyverse.fly import FlyBrain
    from flyverse.device import resolve
    dev = resolve(device)
    cuda = dev.type == "cuda" and torch.cuda.is_available()
    if cuda and lif.event_driven is None:
        lif.event_driven = True
    return FlyBrain(c, seed=seed, device=dev, lif_params=lif, optic_params=op, cuda_kernels=True if cuda else None,
                    cuda_sparse="warp" if cuda else "torch", cuda_graphs=False)


class NodeAccumulator:
    """Online per-node reductions of the localizer (the full time course of 15k frames x 6k cells is not kept): per node
    the mean over the flash frames (`on`), the first `off_frames` frames after the offset (`off`) and the last
    `base_frames` frames of the blank before the onset (`base`), of every recorded quantity, (n_nodes, n_cells)."""

    def __init__(self, stim: Stimulus, n_cells: int, quantities, base_frames: int = 10, off_frames: int = 10):
        node, on = stim.track["node"], stim.track["on"]
        n_nodes = int(stim.params["n_nodes"]); T = len(node)
        self.role = np.full(T, -1, np.int64); self.role_node = np.full(T, -1, np.int64)      # 0 on, 1 off, 2 base
        onset = np.flatnonzero(on & ~np.r_[False, on[:-1]]); offset = np.flatnonzero(~on & np.r_[False, on[:-1]])
        for s in onset:
            e = s
            while e < T and on[e]:
                e += 1
            self.role[s:e] = 0; self.role_node[s:e] = node[s]
            b0 = max(0, s - base_frames); self.role[b0:s] = 2; self.role_node[b0:s] = node[s]
        for s in offset:
            self.role[s:s + off_frames] = np.where(self.role[s:s + off_frames] == -1, 1, self.role[s:s + off_frames])
            self.role_node[s:s + off_frames] = np.where(self.role_node[s:s + off_frames] == -1, node[s - 1], self.role_node[s:s + off_frames])
        self.q = tuple(quantities)
        self.sum = {q: np.zeros((3, n_nodes, n_cells), np.float64) for q in self.q}
        self.cnt = np.zeros((3, n_nodes), np.int64)

    def add(self, t: int, values: dict) -> None:
        r, nd = self.role[t], self.role_node[t]
        if r < 0:
            return
        for q in self.q:
            self.sum[q][r, nd] += values[q]
        self.cnt[r, nd] += 1

    def finish(self) -> dict:
        out = {"n_on": self.cnt[0], "n_off": self.cnt[1], "n_base": self.cnt[2]}
        for q in self.q:
            for r, lab in enumerate(("on", "off", "base")):
                out[f"{lab}__{q}"] = (self.sum[q][r] / np.maximum(self.cnt[r], 1)[:, None]).astype(np.float32)
        return out


def run_arm(fb, stim: Stimulus, present_stimulus: bool, settle_s: float, selection, quantities, label: str, quiet: bool = False):
    """Settle on the blank for `settle_s`, then present the stimulus (or the blank for the same frames) frame by frame,
    recording `quantities` of `selection` after every frame. Returns (Recording | None, node reductions | None,
    per-frame radiance checksum (T,)). The radiance handed to FlyBrain.vision each frame is the stimulus' own array:
    the retinal input is the recorded input (no ray tracing, no replay)."""
    import torch
    c = fb.c
    rec = common.Recorder(c, selection, quantities=quantities)
    n_settle = int(round(settle_s / stim.dt_s))
    blank_t = torch.as_tensor(stim.blank, dtype=torch.float32, device=fb.device)
    for _ in range(n_settle):
        fb.vision(blank_t); fb.step(common.FRAME_MS)
    is_loc = stim.name == "localizer"
    nodes = NodeAccumulator(stim, len(rec.idx), quantities) if (is_loc and present_stimulus) else None
    keep_series = not is_loc
    series = {q: [] for q in quantities} if is_loc else None      # localizer: per-type population means only
    tr = None
    checksum = np.zeros(stim.n_frames, np.float64)
    t0 = time.time(); T = stim.n_frames
    frame_t = None
    for t in range(T):
        f = stim.frame(t) if present_stimulus else stim.blank
        checksum[t] = float(np.asarray(f, np.float64).sum())
        if frame_t is None or frame_t.shape != f.shape:
            frame_t = torch.as_tensor(f, dtype=torch.float32, device=fb.device)
        else:
            frame_t.copy_(torch.as_tensor(f, dtype=torch.float32))
        fb.vision(frame_t); fb.step(common.FRAME_MS)
        if keep_series:
            rec.capture(fb)
        else:
            rec.capture(fb)
            vals = {q: rec._frames[q].pop() for q in quantities}; rec._t.pop()
            if nodes is not None:
                nodes.add(t, vals)
            if tr is None:
                tr = common.Recording(np.zeros(0), rec.idx, rec.body_ids, rec.types).recorder()
            for q in quantities:
                series[q].append(tr.snapshot(vals[q]).astype(np.float32))
        if not quiet and (t % 500 == 0 or t == T - 1):
            print(f"  [{label}] frame {t + 1}/{T} ({time.time() - t0:.0f} s wall; {(t + 1) / max(time.time() - t0, 1e-9):.1f} fps)", flush=True)
    if keep_series:
        recording = rec.finish(meta={"arm": label, "settle_s": settle_s, "stimulus": stim.name, "presented": present_stimulus})
    else:
        recording = common.Recording(np.arange(T) * stim.dt_s * 1000.0, rec.idx, rec.body_ids, rec.types,
                                     {f"pooled_{q}": np.stack(series[q]) for q in quantities}, {},
                                     {"arm": label, "settle_s": settle_s, "stimulus": stim.name, "presented": present_stimulus,
                                      "pooled_keys": list(tr.keys), "note": "localizer: per-type population means per frame; per-cell node reductions in <out>_nodes.npz"})
    return recording, (nodes.finish() if nodes is not None else None), checksum


def cmd_record(args) -> int:
    import torch
    from flyverse import connectome
    from flyverse.interp import trace as tr
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    lif, op = common.params_from_args(args)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    sel = [s for s in args.select.split(",") if s]
    fb = build_fb(c, lif, op, args.seed, args.device)
    dev = fb.brain.device
    print(f"FlyBrain ready: {c.n} neurons, {fb.retina.n_columns} columns, device {dev}, optic overrides {common.parse_kv(args.optic)}, "
          f"lif overrides {common.parse_kv(args.lif)}, receptor {lif.receptor_model}/{lif.receptor_net_rule}", flush=True)
    if dev.type != "cuda" and not args.allow_cpu:
        sys.exit(f"device {dev}: not CUDA (node race; resubmit) -- pass --allow-cpu for a CPU smoke")
    col_az_el = np.asarray(fb.retina.col_az_el)
    stim = make_stimulus(args.stimulus, col_az_el, args)
    print(f"stimulus {stim.name}: {stim.n_frames} frames ({stim.n_frames * stim.dt_s:.1f} s), {json.dumps(to_jsonable(stim.params))}", flush=True)
    arms = [("blank_a", False), ("blank_b", False)] if args.null else [("stim", True), ("blank", False)]
    recs, nodes, sums = {}, {}, {}
    for i, (label, present) in enumerate(arms):
        if i:
            fb = build_fb(c, lif, op, args.seed, args.device)          # a fresh brain per arm, same seed (probe_object_sweep's pattern)
        recs[label], nodes[label], sums[label] = run_arm(fb, stim, present, args.settle, sel, QUANTITIES, label, args.quiet)
    retina_rec = dict(tr.retina_record(fb.retina, c), mode=RETINA_MODE, file=str(out) + "_radiance.npz",
                      sampling=f"synthetic per-column radiance handed to FlyBrain.vision every frame ({stim.dt_s * 1000:.0f} ms); "
                               "no ray tracing; the stored array IS the presented input", pinned_pose=None, geometry="fly pinned, no body")
    prov = common.provenance(c, lif, op, fb=fb, device=args.device, seeds=[args.seed], batch=1,
                             stimulus={"protocol": f"synthetic:{stim.name}", "params": stim.params, "arms": [a for a, _ in arms],
                                       "settle_s": args.settle, "control": "the constant background for the same frames (matched blank)",
                                       "selection": sel, "quantities": list(QUANTITIES), "null": bool(args.null)},
                             retina=retina_rec, cache_dir=args.cache_dir)
    prov["execution"]["device"] = str(dev)
    for label, r in recs.items():
        r.meta.update({"provenance": {k: prov[k] for k in ("flyverse_commit", "source_fingerprint", "compiled_connectome", "model", "execution", "stimulus")},
                       "retina_mode": RETINA_MODE, "seed": args.seed, "generator": "scripts/probe_synthetic_stimuli.py record " + " ".join(map(shlex.quote, sys.argv[2:]))})
        tr.save_recording(r, str(out) + f"_{label}")
    rad = stim.arrays(); rad.update({"col_az_el": col_az_el, "col_dir": np.asarray(fb.retina.col_dir), "col_side": np.asarray(fb.retina.col_side).astype(str),
                                     "col_hex": np.asarray(fb.retina.col_hex), "retina_mode": np.str_(RETINA_MODE),
                                     **{f"checksum__{k}": v for k, v in sums.items()}})
    np.savez_compressed(str(out) + "_radiance.npz", **rad)
    for label, nd in nodes.items():
        if nd is not None:
            np.savez_compressed(str(out) + "_nodes.npz", node_az_deg=stim.track["node_az_deg"], node_el_deg=stim.track["node_el_deg"],
                                idx=recs[label].idx, body_ids=recs[label].body_ids, types=recs[label].types.astype(str), **nd)
    with open(str(out) + "_prov.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(prov), f, indent=1)
    summary = {"stimulus": stim.name, "params": stim.params, "arms": [a for a, _ in arms], "device": str(dev), "seed": args.seed,
               "optic_overrides": common.parse_kv(args.optic), "n_frames": stim.n_frames, "retina_mode": RETINA_MODE,
               "checksum_equal_stim_vs_presented": bool(np.allclose(sums[arms[0][0]], np.asarray(stim.presented(), np.float64).sum((1, 2)))) if not args.null else None}
    if not args.null and stim.name != "localizer":
        summary["per_type"] = family_stats(recs["stim"], recs["blank"])
        print_family(summary["per_type"])
    with open(str(out) + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=1)
    print(f"written {out}_{{{','.join(recs)}}}.npz, {out}_radiance.npz, {out}_prov.json, {out}_summary.json; device {dev}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- statistics
def family_stats(stim: common.Recording, blank: common.Recording, window=None) -> dict:
    """The object sweep's per-type statistics (docs/audits/object_sweep.md 8.1, kept distinct as Neurome asked): for the
    spiking types the max over cells of the per-cell time-mean (stim - blank) drive (`diff_max_over_cells_mean_mv`),
    its mean over cells, and the spike-rate differences; for the rate units `diff_signed_best_cell` (max over cells of
    |mean dr_stim - mean dr_blank|), `diff_abs_best_cell_mean` (max over cells of mean|dr_stim| - mean|dr_blank|) and the
    population `diff_signed_mean`. One run: magnitudes, no verdicts."""
    from flyverse.interp import trace as tr
    out = {}
    types = stim.types.astype(str)
    ds, db = tr.cell_means(stim, SPIKING_Q, window), tr.cell_means(blank, SPIKING_Q, window)
    rs, rb = tr.cell_means(stim, RATE_Q, window), tr.cell_means(blank, RATE_Q, window)
    ras, rab = tr.cell_means(stim, RATE_Q + "_abs", window), tr.cell_means(blank, RATE_Q + "_abs", window)
    win_s = (stim.t_ms[-1] - stim.t_ms[0] + common.FRAME_MS) / 1000.0 if stim.n_frames > 1 else common.FRAME_MS / 1000.0
    hz_s = (stim.quantities["spike_count"][-1] - stim.quantities["spike_count"][0]) / win_s if "spike_count" in stim.quantities else None
    hz_b = (blank.quantities["spike_count"][-1] - blank.quantities["spike_count"][0]) / win_s if "spike_count" in blank.quantities else None
    for t in sorted(set(types)):
        m = types == t
        if np.isfinite(rs[m]).all():
            d = rs[m] - rb[m]; da = ras[m] - rab[m]
            out[t] = {"kind": "rate", "n_cells": int(m.sum()), "dev_mean_stim": float(rs[m].mean()), "dev_mean_blank": float(rb[m].mean()),
                      "dev_abs_mean_stim": float(ras[m].mean()), "diff_signed_mean": float(d.mean()), "diff_signed_best_cell": float(np.abs(d).max()),
                      "diff_abs_mean": float(da.mean()), "diff_abs_best_cell_mean": float(da.max())}
        else:
            d = ds[m] - db[m]
            out[t] = {"kind": "spiking", "n_cells": int(m.sum()), "drive_mean_stim_mv": float(ds[m].mean()), "drive_mean_blank_mv": float(db[m].mean()),
                      "diff_max_over_cells_mean_mv": float(d.max()), "diff_mean_over_cells_mean_mv": float(d.mean()), "diff_min_over_cells_mean_mv": float(d.min())}
            if hz_s is not None:
                out[t].update({"rate_hz_mean_stim": float(hz_s[m].mean()), "rate_hz_mean_blank": float(hz_b[m].mean()),
                               "diff_rate_hz_mean": float((hz_s[m] - hz_b[m]).mean()), "diff_rate_hz_max_cell": float((hz_s[m] - hz_b[m]).max()),
                               "cells_firing_stim": int((hz_s[m] > 0).sum()), "cells_firing_blank": int((hz_b[m] > 0).sum())})
    return out


def print_family(per_type: dict) -> None:
    rows = []
    for t, d in per_type.items():
        if d["kind"] == "spiking":
            rows.append({"type": t, "n": d["n_cells"], "stat": "diff_max_over_cells_mean_mv", "value": d["diff_max_over_cells_mean_mv"],
                         "pop_mean": d["diff_mean_over_cells_mean_mv"], "rate_diff_hz": d.get("diff_rate_hz_mean", np.nan)})
        else:
            rows.append({"type": t, "n": d["n_cells"], "stat": "diff_signed_best_cell", "value": d["diff_signed_best_cell"],
                         "pop_mean": d["diff_signed_mean"], "rate_diff_hz": d["diff_abs_best_cell_mean"]})
    common.print_table(pd.DataFrame(rows).rename(columns={"rate_diff_hz": "rate_diff_hz | diff_abs_best_cell_mean"}), floatfmt="{:+.5f}")


# ---------------------------------------------------------------------------------------------------- RF map
def angular_distance_deg(az1, el1, az2, el2) -> np.ndarray:
    """Great-circle angle between two (az, el) directions in degrees."""
    a1, e1, a2, e2 = map(np.deg2rad, (az1, el1, az2, el2))
    cosd = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


def fit_rf(R: np.ndarray, node_az: np.ndarray, node_el: np.ndarray, z_min: float = 5.0, half_max: float = 0.5,
           spacing_deg: float = 10.0, min_peak: float = 1e-9) -> pd.DataFrame:
    """Per cell (column of R, (n_nodes, n_cells)): the centre and width of its receptive field from the per-node
    responses. Rule: the peak node is argmax |R|; `sign` its sign; the noise is the MAD over nodes (x 1.4826) of R;
    a centre is FITTED when |peak| >= z_min x noise (and > min_peak); the centre is the (R - half_max x peak)-weighted
    centroid of the nodes at or above half-maximum (same sign), the width the diameter of the disc with the area of
    those nodes (2 sqrt(n spacing^2 / pi): an equivalent-disc FWHM; at least `spacing_deg`);
    n_nodes_above_threshold counts those nodes."""
    R = np.asarray(R, np.float64)
    n_nodes, n_cells = R.shape
    med = np.nanmedian(R, axis=0); mad = 1.4826 * np.nanmedian(np.abs(R - med[None, :]), axis=0)
    ip = np.nanargmax(np.abs(np.nan_to_num(R)), axis=0)
    peak = R[ip, np.arange(n_cells)]
    sign = np.sign(peak); sign[sign == 0] = 1.0
    rows = []
    for j in range(n_cells):
        pk = float(peak[j]); s = float(sign[j]); nz = float(mad[j])
        fitted = bool(abs(pk) > min_peak and (nz == 0 and abs(pk) > 0 or abs(pk) >= z_min * nz))
        thr = half_max * abs(pk)
        r = s * R[:, j]
        above = np.flatnonzero(r >= thr) if fitted else np.zeros(0, np.int64)
        if fitted and len(above):
            w = r[above] - thr + 1e-12 * abs(pk)
            az0 = float((w * node_az[above]).sum() / w.sum()); el0 = float((w * node_el[above]).sum() / w.sum())
            width = float(max(2.0 * np.sqrt(len(above) * spacing_deg ** 2 / np.pi), spacing_deg))
        else:
            az0 = el0 = width = float("nan")
        rows.append({"az_deg": az0, "el_deg": el0, "width_deg": width, "peak": pk, "n_nodes_above_threshold": int(len(above)),
                     "sign": int(s), "noise_mad": nz, "z_peak": float(abs(pk) / nz) if nz > 0 else float("inf") if abs(pk) > 0 else 0.0,
                     "fitted": fitted, "peak_node_az_deg": float(node_az[ip[j]]), "peak_node_el_deg": float(node_el[ip[j]])})
    return pd.DataFrame(rows)


def rf_map_from_nodes(nodes_npz, quantity_of_type=None, z_min: float = 5.0, window: str = "on") -> pd.DataFrame:
    """The RF map of one localizer run (<out>_nodes.npz): per recorded body the fit of `window` - base ('on' = the flash
    window, 'off' = the 100 ms after the offset, 'onoff' = the larger |.| of the two) on drive_mv (spiking) / optic_dr
    (rate units). Columns: bodyId, type, az_deg, el_deg, width_deg, peak, n_nodes_above_threshold, then the rest."""
    z = np.load(nodes_npz, allow_pickle=False)
    types = z["types"].astype(str); body = z["body_ids"]
    az, el = z["node_az_deg"], z["node_el_deg"]
    is_rate = np.isfinite(z[f"on__{RATE_Q}"]).all(0)
    q = np.where(is_rate, RATE_Q, SPIKING_Q)

    def resp(lab):
        R = np.where(is_rate[None, :], z[f"{lab}__{RATE_Q}"] - z[f"base__{RATE_Q}"], z[f"{lab}__{SPIKING_Q}"] - z[f"base__{SPIKING_Q}"])
        return np.asarray(R, np.float64)
    if window == "onoff":
        Ron, Roff = resp("on"), resp("off")
        R = np.where(np.abs(Ron).max(0)[None, :] >= np.abs(Roff).max(0)[None, :], Ron, Roff)
    else:
        R = resp(window)
    spacing = float(np.min(np.diff(np.unique(az)))) if len(np.unique(az)) > 1 else 10.0
    df = fit_rf(R, az, el, z_min=z_min, spacing_deg=spacing)
    df.insert(0, "type", types); df.insert(0, "bodyId", [str(int(b)) for b in body])
    df["quantity"] = q; df["window"] = window; df["model_index"] = z["idx"].astype(int)
    cols = ["bodyId", "type", "az_deg", "el_deg", "width_deg", "peak", "n_nodes_above_threshold"]
    return df[cols + [c for c in df.columns if c not in cols]]


def anatomical_centres(c, retina, idx) -> pd.DataFrame:
    """The anatomical column of each cell (flyverse.interp.trace.column_of_cells) as an (az, el) centre; NaN without one."""
    from flyverse.interp import trace as tr
    rate_idx = np.flatnonzero((c.neurons.superclass == "ol_intrinsic").to_numpy() & ~c.neurons.type.isin(__import__("flyverse.connectome", fromlist=["PHOTORECEPTOR_TYPES"]).PHOTORECEPTOR_TYPES).to_numpy())
    col, n_ann = tr.column_of_cells(c, retina, rate_idx)
    cc = col[np.asarray(idx)]
    az = np.where(cc >= 0, retina.col_az_el[np.maximum(cc, 0), 0], np.nan); el = np.where(cc >= 0, retina.col_az_el[np.maximum(cc, 0), 1], np.nan)
    hex_ann = c.neurons.hex1.notna().to_numpy()[np.asarray(idx)]
    return pd.DataFrame({"anat_column": cc, "anat_az_deg": az, "anat_el_deg": el, "hex_annotated": hex_ann})


def cmd_rfmap(args) -> int:
    runs = sorted(p for g in args.runs for p in glob.glob(g))
    if not runs:
        sys.exit(f"no localizer node files match {args.runs}")
    maps = [rf_map_from_nodes(p, z_min=args.z_min, window=args.window) for p in runs]
    m0 = maps[0].copy()
    # reproducibility across runs: per body the centre scatter over the runs in which it was fitted
    if len(maps) > 1:
        A = np.stack([m["az_deg"].to_numpy() for m in maps]); E = np.stack([m["el_deg"].to_numpy() for m in maps]); F = np.stack([m["fitted"].to_numpy() for m in maps])
        m0["n_runs_fitted"] = F.sum(0); m0["az_sd_runs"] = np.nanstd(A, axis=0, ddof=0); m0["el_sd_runs"] = np.nanstd(E, axis=0, ddof=0)
        with np.errstate(invalid="ignore"):
            m0["centre_spread_deg"] = np.nanmax([angular_distance_deg(A[i], E[i], A[k], E[k]) for i in range(len(maps)) for k in range(i + 1, len(maps))], axis=0)
        m0["az_deg"] = np.nanmean(A, axis=0); m0["el_deg"] = np.nanmean(E, axis=0)
        m0["width_deg"] = np.nanmean(np.stack([m["width_deg"].to_numpy() for m in maps]), axis=0)
        m0["peak"] = np.nanmean(np.stack([m["peak"].to_numpy() for m in maps]), axis=0)
        m0["fitted"] = F.all(0)
        m0.loc[~m0["fitted"], ["az_deg", "el_deg", "width_deg"]] = np.nan
    else:
        m0["n_runs_fitted"] = m0["fitted"].astype(int)
    from flyverse import connectome, retina as retina_mod
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    r = retina_mod.build_retina(c)
    an = anatomical_centres(c, r, m0["model_index"].to_numpy())
    m0 = pd.concat([m0.reset_index(drop=True), an.reset_index(drop=True)], axis=1)
    with np.errstate(invalid="ignore"):
        m0["anat_distance_deg"] = angular_distance_deg(m0["az_deg"], m0["el_deg"], m0["anat_az_deg"], m0["anat_el_deg"])
    per_type = []
    for t, g in m0.groupby("type"):
        ok = g["fitted"].to_numpy(); d = g["anat_distance_deg"].to_numpy(); dd = d[np.isfinite(d)]
        per_type.append({"type": t, "n_bodies": int(len(g)), "n_fitted": int(ok.sum()), "coverage": float(ok.mean()),
                         "n_fitted_all_runs": int((g["n_runs_fitted"] == len(maps)).sum()), "n_runs": len(maps),
                         "width_median_deg": float(np.nanmedian(g.loc[ok, "width_deg"])) if ok.any() else np.nan,
                         "peak_median": float(np.nanmedian(np.abs(g.loc[ok, "peak"]))) if ok.any() else np.nan,
                         "n_with_anatomy": int(len(dd)), "anat_distance_median_deg": float(np.median(dd)) if len(dd) else np.nan,
                         "anat_within_10deg": float((dd <= 10).mean()) if len(dd) else np.nan,
                         "anat_within_15deg": float((dd <= 15).mean()) if len(dd) else np.nan,
                         "centre_spread_median_deg": float(np.nanmedian(g.loc[ok, "centre_spread_deg"])) if (len(maps) > 1 and ok.any()) else np.nan,
                         "az_median": float(np.nanmedian(g.loc[ok, "az_deg"])) if ok.any() else np.nan, "el_median": float(np.nanmedian(g.loc[ok, "el_deg"])) if ok.any() else np.nan})
    pt = pd.DataFrame(per_type)
    common.print_table(pt, floatfmt="{:+.3f}")
    csv = Path(args.csv); csv.parent.mkdir(parents=True, exist_ok=True); m0.to_csv(csv, index=False)
    prov0 = json.load(open(str(runs[0]).replace("_nodes.npz", "_prov.json"), encoding="utf-8")) if os.path.exists(str(runs[0]).replace("_nodes.npz", "_prov.json")) else common.provenance(c)
    res = Result.new("trace", prov0)
    res.replicates = {"n": len(runs), "unit": "runs", "runs": [{"run_index": i, "file": str(p)} for i, p in enumerate(runs)], "null": None}
    res.add_table("rf_per_type", pt); res.add_table("rf_map", m0.drop(columns=["model_index"]))
    res.summary = {"rf_map_csv": str(csv), "n_runs": len(runs), "window": args.window, "z_min": args.z_min, "fit_rule": fit_rf.__doc__,
                   "schema": "flyverse.interp.rfmap/1", "coverage": {row["type"]: row["coverage"] for row in per_type}}
    res.validation = {"name": "RF-map coverage and anatomical agreement (Mi1 hex-annotated cells are the ground truth for the localizer itself)",
                      "measured": {row["type"]: {"coverage": row["coverage"], "anat_distance_median_deg": row["anat_distance_median_deg"]} for row in per_type},
                      "status": "measured", "reference": None, "source": "scripts/probe_synthetic_stimuli.py rfmap"}
    res.files = {"generator": "scripts/probe_synthetic_stimuli.py rfmap " + " ".join(map(shlex.quote, sys.argv[2:])), "runs": [str(p) for p in runs], "csv": str(csv)}
    p = res.save(args.json or common.default_json_path("trace", res.run_id))
    print(f"problems: {res.check() or 'none'}; written {csv} and {p}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- family analysis
def rf_window_stats(stim: common.Recording, blank: common.Recording, rad: dict, rfmap: pd.DataFrame) -> pd.DataFrame:
    """Per body with a fitted RF centre: the (stim - blank) response averaged over the frames in which the object's
    centre lies within width / 2 + object half-size of the body's RF centre (the body's own measurement region), next
    to its whole-window mean."""
    if "track__az_deg" not in rad:
        return pd.DataFrame()
    az, el = rad["track__az_deg"], rad["track__el_deg"]; vis = rad.get("track__visible", np.ones(len(az), bool))
    half = 0.5 * np.maximum(rad["track__width_deg"], np.minimum(rad["track__height_deg"], 60.0))
    m = rfmap.set_index("bodyId")
    body = [str(int(b)) for b in stim.body_ids]
    rows = []
    for j, b in enumerate(body):
        if b not in m.index or not bool(m.loc[b, "fitted"]):
            continue
        q = RATE_Q if np.isfinite(stim.quantities[RATE_Q][:, j]).all() else SPIKING_Q
        d = stim.quantities[q][:, j].astype(np.float64) - blank.quantities[q][:, j].astype(np.float64)
        n = min(len(d), len(az))
        inside = vis[:n] & (angular_distance_deg(az[:n], el[:n], m.loc[b, "az_deg"], m.loc[b, "el_deg"]) <= m.loc[b, "width_deg"] / 2.0 + half[:n])
        rows.append({"bodyId": b, "type": str(stim.types[j]), "quantity": q, "rf_az_deg": float(m.loc[b, "az_deg"]), "rf_el_deg": float(m.loc[b, "el_deg"]),
                     "n_frames_in_rf": int(inside.sum()), "diff_in_rf": float(d[:n][inside].mean()) if inside.any() else np.nan,
                     "diff_whole_window": float(d.mean()), "peak_abs_frame": float(np.abs(d).max())})
    return pd.DataFrame(rows)


def cmd_analyse(args) -> int:
    from flyverse.interp import trace as tr
    from flyverse import connectome
    d = Path(args.dir)
    stim_files = sorted(d.glob("*_stim.npz")) + sorted(d.glob("*_blank_a.npz"))
    if not stim_files:
        sys.exit(f"no *_stim.npz / *_blank_a.npz under {d}")
    rfmap = pd.read_csv(args.rf_map, dtype={"bodyId": str}) if args.rf_map else None
    per_type_rows, per_body_rows, families = [], [], {}
    prov0 = None
    for sf in stim_files:
        stem = str(sf)[:-len("_stim.npz")] if sf.name.endswith("_stim.npz") else str(sf)[:-len("_blank_a.npz")]
        bf = Path(stem + ("_blank.npz" if sf.name.endswith("_stim.npz") else "_blank_b.npz"))
        if not bf.exists():
            print(f"  skip {sf.name}: no matched blank"); continue
        stim, blank = common.Recording.load(sf), common.Recording.load(bf)
        rad = dict(np.load(stem + "_radiance.npz", allow_pickle=False))
        name = Path(stem).name
        fam = family_stats(stim, blank)
        families[name] = {"stimulus": str(rad["name"]), "params": json.loads(str(rad["params"])), "arms": [sf.name, bf.name], "per_type": fam,
                          "device": stim.meta.get("provenance", {}).get("execution", {}).get("device"), "seed": stim.meta.get("seed")}
        if prov0 is None and os.path.exists(stem + "_prov.json"):
            prov0 = json.load(open(stem + "_prov.json", encoding="utf-8"))
        print(f"\n== {name}: {rad['name']} (device {families[name]['device']})"); print_family(fam)
        for t, v in fam.items():
            per_type_rows.append({"run": name, "stimulus": str(rad["name"]), "type": t, **v})
        if rfmap is not None:
            rw = rf_window_stats(stim, blank, rad, rfmap)
            if len(rw):
                rw.insert(0, "run", name); per_body_rows.append(rw)
                g = rw.groupby("type").agg(n_bodies_with_rf=("bodyId", "size"), n_bodies_object_in_rf=("n_frames_in_rf", lambda x: int((x > 0).sum())),
                                           diff_in_rf_mean=("diff_in_rf", "mean"), diff_in_rf_best_body=("diff_in_rf", lambda x: float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else np.nan),
                                           diff_whole_window_mean=("diff_whole_window", "mean")).reset_index()
                families[name]["rf_window"] = g.to_dict("records")
                print("  read through the RF map (per body, object inside its fitted RF):"); common.print_table(g, floatfmt="{:+.5f}")
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    res = Result.new("trace", prov0 or common.provenance(c))
    res.add_table("per_type", pd.DataFrame(per_type_rows))
    if per_body_rows:
        res.add_table("per_body_rf_window", pd.concat(per_body_rows, ignore_index=True))
    res.summary = {"families": families, "n_runs_per_family": 1, "verdicts": "none: one run per family -- magnitudes only (docs/INTERP.md 2.4: >= 4-5 runs per arm in one submission before a difference is called)",
                   "rf_map": args.rf_map}
    res.validation = {"name": "synthetic families on the shipped lobe (one instance each)", "measured": {k: v["per_type"] for k, v in families.items()},
                      "status": "measured", "reference": None, "source": "scripts/probe_synthetic_stimuli.py analyse"}
    res.files = {"generator": "scripts/probe_synthetic_stimuli.py analyse " + " ".join(map(shlex.quote, sys.argv[2:])), "dir": str(d)}
    p = res.save(args.json or common.default_json_path("trace", res.run_id))
    print(f"\nproblems: {res.check() or 'none'}; written {p}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- preview / plan
def preview_report(stim: Stimulus, col_az_el: np.ndarray) -> dict:
    """Geometry check of a stimulus on the eye: contrast range, columns touched, the track's angular speed."""
    P = stim.presented(); bg = stim.blank[0]
    rel = (P.sum(2) - stim.blank.sum(1)[None, :]) / stim.blank.sum(1)[None, :]
    out = {"name": stim.name, "n_frames": stim.n_frames, "n_col": stim.n_col, "min_rel": float(rel.min()), "max_rel": float(rel.max()),
           "columns_changed_5pct_mean_per_frame": float((np.abs(rel) > 0.05).sum(1).mean()), "columns_changed_50pct_mean_per_frame": float((np.abs(rel) > 0.5).sum(1).mean()),
           "background": bg.tolist()}
    if "az_deg" in stim.track and stim.n_frames > 1:
        v = np.abs(np.diff(stim.track["az_deg"])) / stim.dt_s
        out.update({"angular_speed_deg_s_median": float(np.median(v)), "angular_speed_deg_s_max": float(v.max()), "elevation_deg": float(stim.track["el_deg"][0])})
    if stim.name == "localizer":
        out.update({"n_nodes": stim.params["n_nodes"], "seconds": stim.params["seconds"]})
    return out


def cmd_preview(args) -> int:
    from flyverse import connectome, retina as retina_mod
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    r = retina_mod.build_retina(c)
    stim = make_stimulus(args.stimulus, np.asarray(r.col_az_el), args)
    rep = preview_report(stim, np.asarray(r.col_az_el))
    print(json.dumps(to_jsonable(dict(rep, params=stim.params)), indent=1))
    return 0


def batch_jobs(out: str, seconds: float = 3.0, settle: float = 2.0, seed: int = 0, loc_runs: int = 3) -> list[tuple[str, str]]:
    """(job name, record arguments) of the validation batch: the localizer on the deterministic lobe (x1) and the shipped
    lobe (x loc_runs), one instance of each stimulus family on the shipped lobe, and a blank/blank null."""
    common_ = f"--seconds {seconds} --settle {settle}"
    jobs = [("loc_fb0_s0", f"--stimulus localizer --optic gain_fb=0 --seed {seed}")]
    jobs += [(f"loc_shipped_s{k}", f"--stimulus localizer --seed {seed + k}") for k in range(loc_runs)]
    jobs += [("rect_dark_h088_w044", f"--stimulus rect --width 4.4 --height 8.8 --contrast {DARK_CONTRAST} {common_} --seed {seed}"),
             ("rect_bright_h088_w044", f"--stimulus rect --width 4.4 --height 8.8 --contrast {-DARK_CONTRAST} {common_} --seed {seed}"),
             ("square_dark_088", f"--stimulus rect --width 8.8 --height 8.8 --contrast {DARK_CONTRAST} {common_} --seed {seed}"),
             ("square_dark_300", f"--stimulus rect --width 30 --height 30 --contrast {DARK_CONTRAST} {common_} --seed {seed}"),
             ("bar_dark_w070", f"--stimulus bar --width 7 --contrast {DARK_CONTRAST} {common_} --seed {seed}"),
             ("grating_p30_c05", f"--stimulus grating --period 30 --contrast 0.5 {common_} --seed {seed}"),
             ("flicker_2hz_c05", f"--stimulus flicker --hz 2 --contrast 0.5 {common_} --seed {seed}"),
             ("flash_on_088", f"--stimulus flash --width 8.8 --contrast {-DARK_CONTRAST} {common_} --seed {seed}"),
             ("flash_off_088", f"--stimulus flash --width 8.8 --contrast {DARK_CONTRAST} {common_} --seed {seed}"),
             ("null_blank_blank", f"--stimulus rect --width 4.4 --height 8.8 --contrast {DARK_CONTRAST} {common_} --seed {seed} --null")]
    return jobs


def cmd_plan(args) -> int:
    out = args.out.rstrip("/")
    jobs = batch_jobs(out, args.seconds, args.settle, args.seed, args.loc_runs)
    lines = []
    for name, a in jobs:
        lines.append(f"mkdir -p {out} && python -c 'import torch; assert torch.cuda.is_available()' && "
                     f"python scripts/probe_synthetic_stimuli.py record {a} --out {out}/{name} > {out}/{name}.txt 2>&1; tail -3 {out}/{name}.txt")
    Path(out).mkdir(parents=True, exist_ok=True)
    log = f"out/{args.name}_cluster.log"
    sh = ["#!/bin/sh", f"# generated by scripts/probe_synthetic_stimuli.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs "
          f"(localizer fb0 x1 + shipped x{args.loc_runs}; one instance per stimulus family; one blank/blank null)", "set -e", f"mkdir -p {out} out",
          f'if [ -f {log} ]; then mv {log} "out/{args.name}_cluster.$(date +%Y%m%dT%H%M%S).log"; fi',
          f"python scripts/cluster_run.py --name {args.name} --minutes {args.minutes} \\"]
    sh += [f"  {shlex.quote(l)} \\" for l in lines]
    sh += [f"  --fetch {out}/ 2>&1 | tee {log}"]
    p = Path(out) / "batch.sh"
    p.write_text("\n".join(sh) + "\n", encoding="utf-8")
    print(f"written {p}: {len(jobs)} jobs; run: sh {p}")
    return 0


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="one stimulus + its matched blank on the GPU"); stimulus_args(r); common.add_common_args(r)
    r.add_argument("--out", required=True, help="output stem: <out>_stim.npz, <out>_blank.npz, <out>_radiance.npz, <out>_nodes.npz, <out>_prov.json, <out>_summary.json")
    r.add_argument("--settle", type=float, default=2.0, help="settling time on the blank before the window (s)")
    r.add_argument("--select", default=",".join(SELECTION), help="recorded populations (common grammar, comma-separated)")
    r.add_argument("--null", action="store_true", help="blank vs blank (arms blank_a / blank_b)")
    r.add_argument("--allow-cpu", action="store_true", help="CPU smoke (a cluster job must run on CUDA)")
    pv = sub.add_parser("preview", help="geometry check on the shipped retina (CPU)"); stimulus_args(pv); pv.add_argument("--cache-dir", default=None)
    m = sub.add_parser("rfmap", help="the RF map from localizer runs (CPU)")
    m.add_argument("--runs", nargs="+", required=True, help="glob(s) of <out>_nodes.npz files; several runs = the map is their mean and their scatter is on file")
    m.add_argument("--csv", required=True); m.add_argument("--json", default=None); m.add_argument("--z-min", type=float, default=5.0)
    m.add_argument("--window", default="on", choices=["on", "off", "onoff"]); m.add_argument("--cache-dir", default=None)
    a = sub.add_parser("analyse", help="per-family summary of a directory of runs, read through an RF map (CPU)")
    a.add_argument("--dir", required=True); a.add_argument("--rf-map", default=None); a.add_argument("--json", default=None); a.add_argument("--cache-dir", default=None)
    pl = sub.add_parser("plan", help="write the cluster batch")
    pl.add_argument("--out", default="out/synth"); pl.add_argument("--name", default="synth"); pl.add_argument("--minutes", type=int, default=45)
    pl.add_argument("--seconds", type=float, default=3.0); pl.add_argument("--settle", type=float, default=2.0); pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--loc-runs", type=int, default=3)
    args = ap.parse_args(argv)
    argv_ = sys.argv[1:] if argv is None else list(argv)
    args.contrast_set = _flag_set(argv_, "--contrast"); args.width_set = _flag_set(argv_, "--width")
    return {"record": cmd_record, "preview": cmd_preview, "rfmap": cmd_rfmap, "analyse": cmd_analyse, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
