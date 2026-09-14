"""Synthetic per-column radiance stimuli on the radiance path (FlyBrain.vision -> OpticLobe.step_frame), fly pinned,
no ray tracing: the retinal input IS the recorded input (retina.mode = 'synthetic_direct').

The matched visual assay Neurome asked for after the size-ladder intake (docs/NEUROME_INTERFACE.md 3b, item 1):
object centre elevation, angular trajectory, angular speed, contrast and background held constant across sizes,
separate height and width ladders matching the published rectangles (Keles & Frye 2017: preferred vertical extent
8.8 deg at width ~4.4 deg), a fixed-centre square ladder, bars / gratings / flicker / ON-OFF flashes as the
specificity battery, and a per-body receptive-field LOCALIZER (a 4.5-deg dark square flashed for 200 ms at every node
of a 10-deg azimuth x elevation grid, blank between) that fixes each body's measurement region before the size runs.

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
    # the validation batch (one cluster_run.py call; fetch a NAMED directory) and the ladder batch for the size runs
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py plan --out out/synth --name synth && sh out/synth/batch.sh
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py plan --ladders --runs 5 --out out/synth_ladder --name synthlad
    # CPU: verify the batch, the RF map from the localizer runs, then the per-family summary read through the map
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py verify --dir out/synth
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py rfmap --runs "out/synth/loc_shipped_s*_nodes.npz" --csv out/synth/rfmap_shipped.csv --json out/interp/synth/rfmap_shipped.json
    PYTHONIOENCODING=utf-8 python scripts/probe_synthetic_stimuli.py analyse --dir out/synth --rf-map out/synth/rfmap_shipped.csv --json out/interp/synth/families.json

ON / OFF transitions. The `flash` stimulus is a PERIODIC square (`on_s` every `period_s`), so every run contains both
transition polarities -- a bright square's onset is the ON transition and its offset the OFF transition, a dark
square's the reverse. The whole-window statistic therefore pools them: a `flash_on` run against a `flash_off` run
separates a BRIGHT flash from a DARK one, not ON from OFF. The ON / OFF measurement is an ANALYSIS-side split of each
run (`transition_frames` / `transition_stats`, the same on / off role split the localizer's `NodeAccumulator` uses):
`family_stats` over the frames of the `--transition-window-s` (0.3 s) after each rising edge and after each falling
edge, reported as rows `transition = on | off` beside the pooled row. The stimulus itself is unchanged.

Units and background. The background is the per-channel [UV, B, G, R] mean of the room's blank arm at the pinned
pose of the object sweep (BACKGROUND, from out/apply_object/ladder/d114_null_s0_retina.npz: mean over 1200 frames x
1466 columns); a dark object is `contrast` = -0.995 (the black ball's radiance is 0.0043-0.0052 of the wall's in
out/apply_object/ladder/d300_stim_s0_retina.npz), a bright one +0.995 (1.995 x background, the wall's own maximum is
2.0 x). Object radiance = background x (1 + contrast x coverage). Azimuth is + left, elevation + up (retina.py).

Nothing in flyverse/ is edited: the model is read through FlyBrain with the LIFParams / OpticParams overrides of the
common CLI (`--optic gain_fb=0` is the deterministic lobe; the opt-in stream hooks of docs/audits/optic_stream_hooks.md
pass through the same flag, so this battery is the specificity control of the model comparison).
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
LC_TYPES = ("LC11", "LC10a")
# ladders (deg): Keles & Frye 2017 style -- one dimension varied, the other fixed; and the fixed-centre square ladder
HEIGHT_LADDER = {"width": 4.4, "heights": [2.2, 4.4, 8.8, 15.0, 22.0, 30.0]}
WIDTH_LADDER = {"height": 8.8, "widths": [2.2, 4.4, 8.8, 15.0, 22.0, 30.0]}
SQUARE_LADDER = [4.5, 8.8, 11.0, 15.0, 20.0, 30.0]
RETINA_MODE = "synthetic_direct"
#: analysis-side ON / OFF transition window (s): the frames after each luminance edge that the split statistic averages
#: (the flash stimulus is a PERIODIC square, so a run holds both polarities -- see transition_stats)
TRANSITION_WINDOW_S = 0.3
RFMAP_SCHEMA = "flyverse.interp.rfmap/1"
RF_CSV_COLUMNS = ["bodyId", "type", "az_deg", "el_deg", "width_deg", "peak", "n_nodes_above_threshold"]


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
    level = np.where(((t * hz) % 1.0) < 0.5 - 1e-9, 1.0 + contrast, 1.0 - contrast).astype(np.float32)
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
              acceptance_deg=ACCEPTANCE_DEG, rays=RAYS, dt_s: float = FRAME_S, passes: int = 1) -> Stimulus:
    """The receptive-field localizer: a `size_deg` square of `contrast` flashed for `flash_s` at every node of an
    azimuth x elevation grid (spacing `spacing_deg`), `blank_s` of background between flashes and before the first,
    in a fixed shuffled order (`order_seed`; the same for every run), the whole grid `passes` times (a new order per
    pass, seeds order_seed + pass; the per-node reductions average over the passes). Stored as pattern (node frames +
    the blank) and a per-frame index; `track['node']` is the node presented (-1 = blank), `track['on']` the flash frames."""
    nodes = localizer_grid(col_az_el, spacing_deg, size_deg, az_range, el_range, acceptance_deg)
    az, el, w = sample_points(col_az_el, acceptance_deg, rays)
    blank = blank_frame(len(col_az_el), background)
    pattern = np.empty((len(nodes) + 1, len(col_az_el), 4), np.float32)
    pattern[0] = blank
    for i, (a, e) in enumerate(nodes):
        pattern[i + 1] = radiance_from_coverage(rect_coverage(az, el, w, a, e, size_deg, size_deg), contrast, background)
    passes = max(1, int(passes))
    order = np.concatenate([np.random.RandomState(order_seed + p).permutation(len(nodes)) for p in range(passes)])
    n_on, n_off = int(round(flash_s / dt_s)), int(round(blank_s / dt_s))
    index, node, on = [], [], []
    for i in order:
        index += [0] * n_off + [i + 1] * n_on
        node += [-1] * n_off + [int(i)] * n_on
        on += [False] * n_off + [True] * n_on
    index += [0] * n_off; node += [-1] * n_off; on += [False] * n_off       # a closing blank: the last node's offset window
    n = len(index)
    params = {"spacing_deg": spacing_deg, "size_deg": size_deg, "contrast": contrast, "flash_s": flash_s, "blank_s": blank_s,
              "order_seed": order_seed, "az_range": list(az_range), "el_range": list(el_range), "n_nodes": int(len(nodes)), "passes": passes,
              "seconds": n * dt_s, "background": list(background), "acceptance_deg": acceptance_deg, "rays": rays}
    return Stimulus("localizer", params, blank, dt_s, pattern=pattern, index=np.asarray(index, np.int64),
                    track={"t_s": np.arange(n) * dt_s, "node": np.asarray(node, np.int64), "on": np.asarray(on, bool),
                           "node_az_deg": nodes[:, 0], "node_el_deg": nodes[:, 1], "order": order.astype(np.int64)})


def blank(col_az_el, seconds: float = 3.0, background=BACKGROUND, dt_s: float = FRAME_S) -> Stimulus:
    """The constant background for `seconds` (the stimulus arm equals the blank arm): the stationarity control -- what a
    recording does on its own, per cell, with nothing presented."""
    n = int(round(seconds / dt_s))
    b = blank_frame(len(col_az_el), background)
    return Stimulus("blank", {"seconds": seconds, "background": list(background)}, b, dt_s, pattern=b[None], index=np.zeros(n, np.int64),
                    track={"t_s": np.arange(n) * dt_s})


FAMILIES = {"rect": rectangle, "bar": bar, "grating": grating, "flicker": flicker, "flash": flash, "localizer": localizer, "blank": blank}


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
        return flash(col_az_el, args.seconds, args.width, args.contrast, args.azimuth, args.elevation, args.on_s, args.period_s, background=bg, rays=args.rays)
    if kind == "localizer":
        return localizer(col_az_el, args.grid_spacing, args.width if args.width_set else 4.5, args.contrast, args.flash_s, args.blank_s,
                         args.order_seed, background=bg, rays=args.rays, passes=args.passes)
    if kind == "blank":
        return blank(col_az_el, args.seconds, bg)
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
    ap.add_argument("--passes", type=int, default=1, help="localizer: present the whole grid this many times (new order per pass; reductions average)")
    ap.add_argument("--rays", type=int, default=RAYS, help="sample rays per column (7 = the room's; more = finer sub-column coverage)")
    ap.add_argument("--background", type=float, nargs=4, default=None, metavar="R", help="[UV B G R] background radiance (default BACKGROUND)")


def _flag_set(argv, flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


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


class GatherRecorder(common.Recorder):
    """common.Recorder with the per-frame gather done on the device: one small transfer per quantity per frame instead
    of the whole (N,) brain tensors (the localizer captures ~15k frames per arm). Same quantities, same output."""

    def capture(self, fb, t_ms=None, motor=None) -> None:
        import torch
        brain = getattr(fb, "brain", fb)
        if int(getattr(brain, "B", 1)) != 1 or motor is not None:
            return super().capture(fb, t_ms, motor)
        if getattr(self, "_idx_t", None) is None:
            dev = brain.drive.device
            self._idx_t = torch.as_tensor(self.idx, device=dev)
            optic = getattr(fb, "optic", None)
            pos = self._optic_positions(optic) if optic is not None else -np.ones(len(self.idx), np.int64)
            self._ok = pos >= 0; self._pos_t = torch.as_tensor(pos[self._ok], device=dev)
        self._t.append(float(brain.t if t_ms is None else t_ms))
        for q in self.quantities:
            if q == "rate_hz":
                x = np.asarray(brain.rate_np())[self.idx]
            elif q in ("drive_mv", "v_mv", "adapt_mv", "refrac", "spike_count"):
                name = {"drive_mv": "drive", "v_mv": "v", "adapt_mv": "adapt", "refrac": "refrac", "spike_count": "spike_counts"}[q]
                src = getattr(brain, name)
                x = (src[0] if src.ndim == 2 else src).index_select(0, self._idx_t).cpu().numpy()
                if q == "refrac":
                    x = (x > 0).astype(np.float32)
            else:
                x = np.full(len(self.idx), np.nan, np.float32)
                optic = getattr(fb, "optic", None)
                if optic is not None and self._ok.any():
                    src = optic.delta_rate if q == "optic_dr" else optic.rates()
                    x[self._ok] = (src[0] if src.ndim == 2 else src).index_select(0, self._pos_t).cpu().numpy()
            self._frames[q].append(np.asarray(x, dtype=np.float32))


class NodeAccumulator:
    """Online per-node reductions of the localizer (the full time course of 15k frames x 7k cells is not kept): per node
    the mean over the flash frames (`on`), the first `off_frames` frames after the offset (`off`) and the last
    `base_frames` frames of the blank before the onset (`base`), of every recorded quantity, (n_nodes, n_cells).
    `spike_count` (cumulative in the Recording convention) is accumulated as spikes per frame."""

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
        self._prev_sc = None

    def add(self, t: int, values: dict) -> None:
        vals = dict(values)
        if "spike_count" in vals:
            sc = np.asarray(vals["spike_count"], np.float64)
            vals["spike_count"] = sc - (self._prev_sc if self._prev_sc is not None else sc)
            self._prev_sc = sc
        r, nd = self.role[t], self.role_node[t]
        if r < 0:
            return
        for q in self.q:
            self.sum[q][r, nd] += vals[q]
        self.cnt[r, nd] += 1

    def finish(self) -> dict:
        out = {"n_on": self.cnt[0], "n_off": self.cnt[1], "n_base": self.cnt[2]}
        for q in self.q:
            lab_q = "spikes_per_frame" if q == "spike_count" else q
            for r, lab in enumerate(("on", "off", "base")):
                out[f"{lab}__{lab_q}"] = (self.sum[q][r] / np.maximum(self.cnt[r], 1)[:, None]).astype(np.float32)
        return out


def play(fb, stim: Stimulus, present_stimulus: bool, on_frame=None) -> np.ndarray:
    """Hand the stimulus (or its blank, for the same frames) to FlyBrain.vision frame by frame, stepping one frame each;
    returns the per-frame checksum (sum of the radiance presented). The device tensor is allocated ONCE and filled with
    copy_ -- never `torch.as_tensor(frame)`, which on the CPU aliases the numpy view and let the next copy_ overwrite the
    stimulus' own stored radiance (the smoke's checksum mismatch)."""
    import torch
    frame_t = torch.empty((stim.n_col, 4), dtype=torch.float32, device=fb.device)
    checksum = np.zeros(stim.n_frames, np.float64)
    for t in range(stim.n_frames):
        f = stim.frame(t) if present_stimulus else stim.blank
        checksum[t] = float(np.asarray(f, np.float64).sum())
        frame_t.copy_(torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)))
        fb.vision(frame_t); fb.step(common.FRAME_MS)
        if on_frame is not None:
            on_frame(t)
    return checksum


def run_arm(fb, stim: Stimulus, present_stimulus: bool, settle_s: float, selection, quantities, label: str, quiet: bool = False):
    """Settle on the blank for `settle_s`, then present the stimulus (or the blank for the same frames) frame by frame,
    recording `quantities` of `selection` after every frame. Returns (Recording, node reductions | None, per-frame
    radiance checksum (T,), wall seconds). The radiance handed to FlyBrain.vision each frame is the stimulus' own
    array: the retinal input is the recorded input (no ray tracing, no replay). A localizer arm (stimulus OR blank)
    keeps per-node reductions of every cell and per-type population means per frame instead of the full series."""
    import torch
    rec = GatherRecorder(fb.c, selection, quantities=quantities)
    n_settle = int(round(settle_s / stim.dt_s))
    blank_t = torch.as_tensor(np.ascontiguousarray(stim.blank), dtype=torch.float32, device=fb.device).clone()
    for _ in range(n_settle):
        fb.vision(blank_t); fb.step(common.FRAME_MS)
    is_loc = stim.name == "localizer"
    nodes = NodeAccumulator(stim, len(rec.idx), quantities) if is_loc else None
    series = {q: [] for q in quantities} if is_loc else None      # localizer: per-type population means only
    trec = common.Recording(np.zeros(0), rec.idx, rec.body_ids, rec.types).recorder() if is_loc else None
    T = stim.n_frames; t0 = time.time()

    def on_frame(t):
        rec.capture(fb)
        if is_loc:
            vals = {q: rec._frames[q].pop() for q in quantities}; rec._t.pop()
            nodes.add(t, vals)
            for q in quantities:
                series[q].append(trec.snapshot(vals[q]).astype(np.float32))
        if not quiet and (t % 500 == 0 or t == T - 1):
            print(f"  [{label}] frame {t + 1}/{T} ({time.time() - t0:.0f} s wall; {(t + 1) / max(time.time() - t0, 1e-9):.1f} fps)", flush=True)

    checksum = play(fb, stim, present_stimulus, on_frame)
    wall = time.time() - t0
    meta = {"arm": label, "settle_s": settle_s, "stimulus": stim.name, "presented": present_stimulus, "wall_s": wall, "fps": T / max(wall, 1e-9)}
    if not is_loc:
        recording = rec.finish(meta=meta)
    else:
        recording = common.Recording(np.arange(T) * stim.dt_s * 1000.0, rec.idx, rec.body_ids, rec.types,
                                     {f"pooled_{q}": np.stack(series[q]) for q in quantities}, {},
                                     dict(meta, pooled_keys=list(trec.keys), note="localizer: per-type population means per frame "
                                          "(spike_count cumulative as in the Recording convention); per-cell node reductions in <out>_nodes*.npz"))
    return recording, (nodes.finish() if nodes is not None else None), checksum, wall


def cmd_record(args) -> int:
    from flyverse import connectome
    from flyverse.interp import trace as tr
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    lif, op = common.params_from_args(args)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    sel = [s for s in args.select.split(",") if s]
    fb = build_fb(c, lif, op, args.seed, args.device)
    dev = fb.brain.device
    hook_info = getattr(fb.optic, "hook_info", None)
    print(f"FlyBrain ready: {c.n} neurons, {fb.retina.n_columns} columns, device {dev}, optic overrides {common.parse_kv(args.optic)}, "
          f"lif overrides {common.parse_kv(args.lif)}, receptor {lif.receptor_model}/{lif.receptor_net_rule}, hooks {hook_info}", flush=True)
    if dev.type != "cuda" and not args.allow_cpu:
        sys.exit(f"device {dev}: not CUDA (node race; resubmit) -- pass --allow-cpu for a CPU smoke")
    col_az_el = np.asarray(fb.retina.col_az_el)
    stim = make_stimulus(args.stimulus, col_az_el, args)
    print(f"stimulus {stim.name}: {stim.n_frames} frames ({stim.n_frames * stim.dt_s:.1f} s), {json.dumps(to_jsonable(stim.params))}", flush=True)
    arms = [("blank_a", False), ("blank_b", False)] if args.null else [("stim", True), ("blank", False)]
    recs, nodes, sums, walls = {}, {}, {}, {}
    for i, (label, present) in enumerate(arms):
        if i:
            fb = build_fb(c, lif, op, args.seed, args.device)          # a fresh brain per arm, same seed (probe_object_sweep's pattern)
        recs[label], nodes[label], sums[label], walls[label] = run_arm(fb, stim, present, args.settle, sel, QUANTITIES, label, args.quiet)
    retina_rec = dict(tr.retina_record(fb.retina, c), mode=RETINA_MODE, file=str(out) + "_radiance.npz",
                      sampling=f"synthetic per-column radiance handed to FlyBrain.vision every frame ({stim.dt_s * 1000:.0f} ms); "
                               "no ray tracing; the stored array IS the presented input", pinned_pose=None, geometry="fly pinned, no body")
    prov = common.provenance(c, lif, op, fb=fb, device=args.device, seeds=[args.seed], batch=1,
                             stimulus={"protocol": f"synthetic:{stim.name}", "params": stim.params, "arms": [a for a, _ in arms],
                                       "settle_s": args.settle, "control": "the constant background for the same frames (matched blank)",
                                       "selection": sel, "quantities": list(QUANTITIES), "null": bool(args.null)},
                             retina=retina_rec, cache_dir=args.cache_dir)
    prov["execution"]["device"] = str(dev)
    if hook_info is not None:
        prov["model"]["optic_hook_info"] = to_jsonable(hook_info)
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
            suffix = "_nodes.npz" if label == "stim" else f"_nodes_{label}.npz"
            np.savez_compressed(str(out) + suffix, node_az_deg=stim.track["node_az_deg"], node_el_deg=stim.track["node_el_deg"], arm=np.str_(label),
                                presented=np.bool_(label == "stim"), idx=recs[label].idx, body_ids=recs[label].body_ids, types=recs[label].types.astype(str), **nd)
    with open(str(out) + "_prov.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(prov), f, indent=1)
    presented_sum = np.asarray(stim.presented(), np.float64).sum((1, 2)); blank_sum = float(np.asarray(stim.blank, np.float64).sum())
    checks = {lab: bool(np.allclose(sums[lab], presented_sum if present else blank_sum)) for lab, present in arms}
    summary = {"stimulus": stim.name, "params": stim.params, "arms": [a for a, _ in arms], "device": str(dev), "seed": args.seed,
               "optic_overrides": common.parse_kv(args.optic), "lif_overrides": common.parse_kv(args.lif), "n_frames": stim.n_frames, "retina_mode": RETINA_MODE,
               "checksum_equal_stim_vs_presented": checks.get("stim"), "checksum_per_arm": checks, "wall_s": walls,
               "fps": {k: stim.n_frames / max(v, 1e-9) for k, v in walls.items()}, "hook_info": to_jsonable(hook_info) if hook_info is not None else None}
    if not args.null and stim.name != "localizer":
        summary["per_type"] = family_stats(recs["stim"], recs["blank"])
        print_family(summary["per_type"])
        split = transition_stats(recs["stim"], recs["blank"], stim.name, stim.params, stim.track,
                                 getattr(args, "transition_window_s", TRANSITION_WINDOW_S), stim.dt_s)
        if split is not None:                          # flash / flicker: the ON and OFF transitions of THIS run, apart
            summary["per_type_transitions"] = split
            print_transitions(split)
    with open(str(out) + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=1)
    print(f"written {out}_{{{','.join(recs)}}}.npz, {out}_radiance.npz, {out}_prov.json, {out}_summary.json; device {dev}; "
          f"checksums {checks}; wall {', '.join(f'{k} {v:.0f} s' for k, v in walls.items())}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- statistics
def time_course(stim: common.Recording, blank: common.Recording, every: int = 5) -> pd.DataFrame:
    """Per type, the population-mean (stim - blank) per frame (drive_mv for spiking types, optic_dr for rate units),
    every `every`-th frame, plus the spikes per frame summed over the type: the time course reported before any
    population maximum (docs/NEUROME_INTERFACE.md 3b item 1)."""
    rows = []
    types = stim.types.astype(str)
    t_s = stim.t_ms / 1000.0
    for t in sorted(set(types)):
        m = types == t
        q = RATE_Q if np.isfinite(stim.quantities[RATE_Q][:, m]).all() else SPIKING_Q
        d = stim.quantities[q][:, m].astype(np.float64).mean(1) - blank.quantities[q][:, m].astype(np.float64).mean(1)
        sp_s = np.diff(stim.quantities["spike_count"][:, m].astype(np.float64).sum(1), prepend=0.0) if "spike_count" in stim.quantities else None
        sp_b = np.diff(blank.quantities["spike_count"][:, m].astype(np.float64).sum(1), prepend=0.0) if "spike_count" in blank.quantities else None
        for k in range(0, len(t_s), every):
            rows.append({"type": t, "quantity": q, "t_s": float(t_s[k] - t_s[0]), "diff_pop_mean": float(d[k]),
                         "spikes_stim": float(sp_s[k]) if sp_s is not None else np.nan, "spikes_blank": float(sp_b[k]) if sp_b is not None else np.nan})
    return pd.DataFrame(rows)


def frame_subset(rec: common.Recording, mask: np.ndarray) -> common.Recording:
    """The recording restricted to the frames of `mask` -- NOT necessarily contiguous, which is what separates the ON
    from the OFF transitions of a periodic flash. `spike_count` (cumulative in the Recording convention) is
    re-accumulated over the kept frames, so `last - first` counts the spikes of the kept windows and not of the gaps
    between them; every other quantity is a plain row selection."""
    mask = np.asarray(mask, bool)
    q = {}
    for k, v in rec.quantities.items():
        v = np.asarray(v)
        if k == "spike_count" and v.ndim == 2 and len(v):
            per = np.diff(v.astype(np.float64), axis=0, prepend=v[:1].astype(np.float64))
            q[k] = np.cumsum(per[mask], axis=0).astype(np.float32)
        else:
            q[k] = v[mask]
    return common.Recording(rec.t_ms[mask], rec.idx, rec.body_ids, rec.types, q,
                            {k: np.asarray(v)[mask] for k, v in rec.motor.items()},
                            dict(rec.meta, frame_mask=True, frames_kept=int(mask.sum())))


def luminance_state(name: str, params: dict, track) -> np.ndarray | None:
    """Per-frame bool of the two-level families: True on the frames at the BRIGHTER level, False at the darker one --
    the signal whose RISING edges are ON transitions and whose FALLING edges are OFF transitions. `flash` is a PERIODIC
    square (`on_s` every `period_s`), so every run carries BOTH polarities: a bright square's onset is the ON
    transition and its offset the OFF transition, a dark square's the reverse (the stimulus' own
    `params['transition']`). `flicker`'s stored level carries the same information. None for the families with no level
    step to split on (rect / bar / grating / localizer / blank), whose statistic is the whole window."""
    def get(k):
        for key in (f"track__{k}", k):
            if key in track:
                return track[key]
        return None
    if str(name) in ("flash_on", "flash_off", "flash"):
        vis = get("visible")
        return None if vis is None else (np.asarray(vis, bool) if float(params.get("contrast", -1.0)) > 0 else ~np.asarray(vis, bool))
    if str(name) == "flicker":
        lev = get("level")
        return None if lev is None else np.asarray(lev, np.float64) > 1.0
    return None


def transition_frames(hi, dt_s: float = FRAME_S, window_s: float = TRANSITION_WINDOW_S) -> dict:
    """The analysis-side ON / OFF split (the stimulus is NOT touched). `hi` is the per-frame brighter-level flag of
    `luminance_state`; the returned per-frame bool masks `on` / `off` are the first `window_s` after each rising /
    falling edge, truncated at the next edge so that a frame belongs to exactly one transition. The frames before the
    first edge (the initial level, and the settle) belong to neither. Same role split as the localizer's
    `NodeAccumulator` (on = the frames after the onset, off = the frames after the offset), applied to a family run."""
    hi = np.asarray(hi, bool); T = int(len(hi))
    n_win = max(1, int(round(window_s / dt_s)))
    edges = (np.flatnonzero(hi[1:] != hi[:-1]) + 1) if T > 1 else np.zeros(0, np.int64)
    masks = {"on": np.zeros(T, bool), "off": np.zeros(T, bool)}
    idx: dict = {"on": [], "off": []}
    for i, s in enumerate(edges):
        s = int(s); nxt = int(edges[i + 1]) if i + 1 < len(edges) else T
        kind = "on" if bool(hi[s]) else "off"
        masks[kind][s:min(s + n_win, nxt, T)] = True
        idx[kind].append(s)
    return {"on": masks["on"], "off": masks["off"], "edges_on": np.asarray(idx["on"], np.int64),
            "edges_off": np.asarray(idx["off"], np.int64), "window_frames": int(n_win), "window_s": float(n_win * dt_s)}


def transition_masks(name: str, params: dict, track, window_s: float = TRANSITION_WINDOW_S, dt_s: float = FRAME_S,
                     n_frames: int | None = None) -> dict | None:
    """`transition_frames` of a stored run (`luminance_state` + the run's track), clipped / zero-padded to `n_frames`
    (the recording's length). None when the family has no level step to split on."""
    hi = luminance_state(name, params, track)
    if hi is None:
        return None
    hi = np.asarray(hi, bool)
    n = int(len(hi) if n_frames is None else min(len(hi), int(n_frames)))
    tf = transition_frames(hi[:n], dt_s, window_s)
    if n_frames is not None and int(n_frames) != n:
        for k in ("on", "off"):
            m = np.zeros(int(n_frames), bool); m[:n] = tf[k]; tf[k] = m
    return tf


def clip_mask(mask: np.ndarray, n_frames: int) -> np.ndarray:
    """A frame mask re-cut to `n_frames` (a null run scored on another run's edge schedule: same protocol, same dt)."""
    out = np.zeros(int(n_frames), bool); k = min(len(mask), int(n_frames)); out[:k] = np.asarray(mask, bool)[:k]
    return out


def transition_stats(stim: common.Recording, blank: common.Recording, name: str, params: dict, track,
                     window_s: float = TRANSITION_WINDOW_S, dt_s: float = FRAME_S) -> dict | None:
    """`family_stats` computed separately over the frames after each ON edge and after each OFF edge of a run.

    Why it exists: `family_stats(..., window=None)` is the WHOLE-window time mean, and the flash stimulus is periodic,
    so the pooled row of a `flash_on` run against a `flash_off` run separates a BRIGHT square from a DARK one -- not an
    ON transition from an OFF one (both are inside each run). These rows are the ON / OFF measurement Neurome asked
    for: `transition = on | off`, reported beside the pooled row, never instead of it. None when the family has no
    level step (`luminance_state`)."""
    n = int(min(stim.n_frames, blank.n_frames))
    tf = transition_masks(name, params, track, window_s, dt_s, n_frames=n)
    if tf is None:
        return None
    out = {"window_s": tf["window_s"], "window_frames": tf["window_frames"],
           "n_edges": {k: int(len(tf[f"edges_{k}"])) for k in ("on", "off")},
           "edge_t_s": {k: [float(i * dt_s) for i in tf[f"edges_{k}"]] for k in ("on", "off")},
           "n_frames": {}, "per_type": {},
           "definition": "family_stats over the frames of the window after each edge of that polarity; the pooled row "
                         "is the whole window (both polarities) and separates bright from dark, not ON from OFF"}
    for k in ("on", "off"):
        m = clip_mask(tf[k], stim.n_frames)
        out["n_frames"][k] = int(m.sum())
        if int(m.sum()) >= 2:
            out["per_type"][k] = family_stats(stim, blank, frames=m)
    return out


def print_transitions(tr_split: dict) -> None:
    """The ON / OFF split beside the pooled table (`transition_stats`)."""
    for k, pt in tr_split.get("per_type", {}).items():
        print(f"  -- transition {k.upper()}: {tr_split['n_frames'][k]} frames = the {tr_split['window_s'] * 1000:.0f} ms after each of "
              f"{tr_split['n_edges'][k]} {k.upper()} edges (the pooled table above is the whole window: BOTH polarities)")
        print_family(pt)


def family_stats(stim: common.Recording, blank: common.Recording, window=None, frames=None) -> dict:
    """The object sweep's per-type statistics (docs/audits/object_sweep.md 8.1, kept distinct as Neurome asked): for the
    spiking types the max over cells of the per-cell time-mean (stim - blank) drive (`diff_max_over_cells_mean_mv`),
    its mean over cells, and the spike-rate differences; for the rate units `diff_signed_best_cell` (max over cells of
    |mean dr_stim - mean dr_blank|), `diff_abs_best_cell_mean` (max over cells of mean|dr_stim| - mean|dr_blank|) and the
    population `diff_signed_mean`. Plus the time course's peak: max over frames of |population-mean (stim - blank)| and
    its time. One run: magnitudes, no verdicts.

    `window` = a contiguous (start_s, end_s); `frames` = a per-frame bool mask, which may be discontiguous (the ON /
    OFF transition split of `transition_frames`). With neither, every statistic is the WHOLE-window mean -- for the
    periodic flash that pools both transition polarities (see `transition_stats`). Under `frames` the spike rates use
    the kept frames' own duration and `tc_peak_t_s` is measured from the FIRST KEPT frame (the kept windows are
    concatenated), so read it as an offset within the split, not as a time in the run."""
    from flyverse.interp import trace as tr
    out = {}
    if frames is not None:
        stim, blank = frame_subset(stim, np.asarray(frames, bool)), frame_subset(blank, np.asarray(frames, bool))
        window = None
    types = stim.types.astype(str)
    ds, db = tr.cell_means(stim, SPIKING_Q, window), tr.cell_means(blank, SPIKING_Q, window)
    rs, rb = tr.cell_means(stim, RATE_Q, window), tr.cell_means(blank, RATE_Q, window)
    ras, rab = tr.cell_means(stim, RATE_Q + "_abs", window), tr.cell_means(blank, RATE_Q + "_abs", window)
    win_s = ((stim.n_frames * common.FRAME_MS) / 1000.0 if frames is not None else
             (stim.t_ms[-1] - stim.t_ms[0] + common.FRAME_MS) / 1000.0 if stim.n_frames > 1 else common.FRAME_MS / 1000.0)
    hz_s = (stim.quantities["spike_count"][-1] - stim.quantities["spike_count"][0]) / win_s if "spike_count" in stim.quantities else None
    hz_b = (blank.quantities["spike_count"][-1] - blank.quantities["spike_count"][0]) / win_s if "spike_count" in blank.quantities else None
    t_s = (stim.t_ms - stim.t_ms[0]) / 1000.0
    for t in sorted(set(types)):
        m = types == t
        if np.isfinite(rs[m]).all():
            d = rs[m] - rb[m]; da = ras[m] - rab[m]
            tc = stim.quantities[RATE_Q][:, m].astype(np.float64).mean(1) - blank.quantities[RATE_Q][:, m].astype(np.float64).mean(1)
            out[t] = {"kind": "rate", "n_cells": int(m.sum()), "dev_mean_stim": float(rs[m].mean()), "dev_mean_blank": float(rb[m].mean()),
                      "dev_abs_mean_stim": float(ras[m].mean()), "diff_signed_mean": float(d.mean()), "diff_signed_best_cell": float(np.abs(d).max()),
                      "diff_abs_mean": float(da.mean()), "diff_abs_best_cell_mean": float(da.max())}
        else:
            d = ds[m] - db[m]
            tc = stim.quantities[SPIKING_Q][:, m].astype(np.float64).mean(1) - blank.quantities[SPIKING_Q][:, m].astype(np.float64).mean(1)
            out[t] = {"kind": "spiking", "n_cells": int(m.sum()), "drive_mean_stim_mv": float(ds[m].mean()), "drive_mean_blank_mv": float(db[m].mean()),
                      "diff_max_over_cells_mean_mv": float(d.max()), "diff_mean_over_cells_mean_mv": float(d.mean()), "diff_min_over_cells_mean_mv": float(d.min())}
            if hz_s is not None:
                out[t].update({"rate_hz_mean_stim": float(hz_s[m].mean()), "rate_hz_mean_blank": float(hz_b[m].mean()),
                               "diff_rate_hz_mean": float((hz_s[m] - hz_b[m]).mean()), "diff_rate_hz_max_cell": float((hz_s[m] - hz_b[m]).max()),
                               "cells_firing_stim": int((hz_s[m] > 0).sum()), "cells_firing_blank": int((hz_b[m] > 0).sum())})
        k = int(np.argmax(np.abs(tc)))
        out[t].update({"tc_peak_pop_mean": float(tc[k]), "tc_peak_t_s": float(t_s[k]), "tc_abs_mean": float(np.abs(tc).mean())})
    return out


def per_body_rows(stim: common.Recording, blank: common.Recording, types=LC_TYPES) -> pd.DataFrame:
    """Per body of `types`: the whole-window time-mean (stim - blank), the frame of largest |difference| and its value,
    and the spike rates of both arms -- the per-body view that precedes any population maximum."""
    rows = []
    ts = stim.types.astype(str); t_s = (stim.t_ms - stim.t_ms[0]) / 1000.0
    win_s = (stim.t_ms[-1] - stim.t_ms[0] + common.FRAME_MS) / 1000.0 if stim.n_frames > 1 else common.FRAME_MS / 1000.0
    for t in types:
        for j in np.flatnonzero(ts == t):
            q = RATE_Q if np.isfinite(stim.quantities[RATE_Q][:, j]).all() else SPIKING_Q
            d = stim.quantities[q][:, j].astype(np.float64) - blank.quantities[q][:, j].astype(np.float64)
            k = int(np.argmax(np.abs(d)))
            row = {"bodyId": str(int(stim.body_ids[j])), "type": t, "quantity": q, "diff_time_mean": float(d.mean()),
                   "peak_abs_frame_diff": float(d[k]), "t_peak_s": float(t_s[k]), "n_frames": int(len(d))}
            if "spike_count" in stim.quantities:
                row["rate_hz_stim"] = float((stim.quantities["spike_count"][-1, j] - stim.quantities["spike_count"][0, j]) / win_s)
                row["rate_hz_blank"] = float((blank.quantities["spike_count"][-1, j] - blank.quantities["spike_count"][0, j]) / win_s)
            rows.append(row)
    return pd.DataFrame(rows)


def print_family(per_type: dict) -> None:
    rows = []
    for t, d in per_type.items():
        if d["kind"] == "spiking":
            rows.append({"type": t, "n": d["n_cells"], "stat": "diff_max_over_cells_mean_mv", "value": d["diff_max_over_cells_mean_mv"],
                         "pop_mean": d["diff_mean_over_cells_mean_mv"], "third": d.get("diff_rate_hz_max_cell", np.nan), "tc_peak": d["tc_peak_pop_mean"], "tc_t": d["tc_peak_t_s"]})
        else:
            rows.append({"type": t, "n": d["n_cells"], "stat": "diff_signed_best_cell", "value": d["diff_signed_best_cell"],
                         "pop_mean": d["diff_signed_mean"], "third": d["diff_abs_best_cell_mean"], "tc_peak": d["tc_peak_pop_mean"], "tc_t": d["tc_peak_t_s"]})
    common.print_table(pd.DataFrame(rows).rename(columns={"third": "diff_rate_hz_max_cell | diff_abs_best_cell_mean"}), floatfmt="{:+.5f}")


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


def node_responses(nodes_npz, window: str = "on"):
    """(R (n_nodes, n_cells), node_az, node_el, types, body_ids, idx, quantity per cell) of one localizer arm: the per-node
    `window` - base response ('on' = the flash window, 'off' = the 100 ms after the offset, 'onoff' = the larger |.| of
    the two), drive_mv for spiking cells / optic_dr for rate units."""
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
    return R, az, el, types, body, z["idx"].astype(int), q


def rf_map_from_nodes(nodes_npz, z_min: float = 5.0, window: str = "on") -> pd.DataFrame:
    """The RF map of one localizer arm (<out>_nodes.npz, or _nodes_blank.npz for the false-fit rate): per recorded body
    the fit of `window` - base on drive_mv (spiking) / optic_dr (rate units) (`node_responses`). Columns: bodyId, type,
    az_deg, el_deg, width_deg, peak, n_nodes_above_threshold, then the rest."""
    z = np.load(nodes_npz, allow_pickle=False)
    R, az, el, types, body, idx, q = node_responses(nodes_npz, window)
    spacing = float(np.min(np.diff(np.unique(az)))) if len(np.unique(az)) > 1 else 10.0
    df = fit_rf(R, az, el, z_min=z_min, spacing_deg=spacing)
    df.insert(0, "type", types); df.insert(0, "bodyId", [str(int(b)) for b in body])
    df["quantity"] = q; df["window"] = window; df["model_index"] = idx
    if "on__spikes_per_frame" in z.files:
        df["spikes_on_minus_base_hz_peak"] = np.abs(z["on__spikes_per_frame"] - z["base__spikes_per_frame"]).max(0) * (1000.0 / common.FRAME_MS)
    return df[RF_CSV_COLUMNS + [c for c in df.columns if c not in RF_CSV_COLUMNS]]


def anatomical_centres(c, retina, idx) -> pd.DataFrame:
    """The anatomical column of each cell (flyverse.interp.trace.column_of_cells over the optic lobe's rate units, the
    same rule as OpticLobe.rate_idx) as an (az, el) centre; NaN without one."""
    from flyverse.connectome import PHOTORECEPTOR_TYPES
    from flyverse.interp import trace as tr
    nrn = c.neurons
    rate_idx = np.flatnonzero((nrn.superclass == "ol_intrinsic").to_numpy() & ~nrn.type.isin(PHOTORECEPTOR_TYPES).to_numpy())
    col, n_ann = tr.column_of_cells(c, retina, rate_idx)
    cc = col[np.asarray(idx)]
    az = np.where(cc >= 0, retina.col_az_el[np.maximum(cc, 0), 0], np.nan); el = np.where(cc >= 0, retina.col_az_el[np.maximum(cc, 0), 1], np.nan)
    hex_ann = nrn.hex1.notna().to_numpy()[np.asarray(idx)]
    return pd.DataFrame({"anat_column": cc, "anat_az_deg": az, "anat_el_deg": el, "hex_annotated": hex_ann})


def merge_maps(maps: list[pd.DataFrame]) -> pd.DataFrame:
    """Several runs of the localizer -> one map: per body the mean centre / width / peak over the runs in which it was
    fitted, `fitted` = fitted in EVERY run, `n_runs_fitted`, the per-axis SD of the centre over runs and the largest
    pairwise great-circle distance between the runs' centres (`centre_spread_deg`)."""
    m0 = maps[0].copy()
    if len(maps) == 1:
        m0["n_runs_fitted"] = m0["fitted"].astype(int); m0["az_sd_runs"] = np.nan; m0["el_sd_runs"] = np.nan; m0["centre_spread_deg"] = np.nan
        return m0
    A = np.stack([m["az_deg"].to_numpy(float) for m in maps]); E = np.stack([m["el_deg"].to_numpy(float) for m in maps]); F = np.stack([m["fitted"].to_numpy(bool) for m in maps])
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        m0["n_runs_fitted"] = F.sum(0); m0["az_sd_runs"] = np.nanstd(A, axis=0, ddof=0); m0["el_sd_runs"] = np.nanstd(E, axis=0, ddof=0)
        m0["centre_spread_deg"] = np.nanmax([angular_distance_deg(A[i], E[i], A[k], E[k]) for i in range(len(maps)) for k in range(i + 1, len(maps))], axis=0)
        m0["az_deg"] = np.nanmean(A, axis=0); m0["el_deg"] = np.nanmean(E, axis=0)
        m0["width_deg"] = np.nanmean(np.stack([m["width_deg"].to_numpy(float) for m in maps]), axis=0)
        m0["peak"] = np.nanmean(np.stack([m["peak"].to_numpy(float) for m in maps]), axis=0)
        m0["n_nodes_above_threshold"] = np.rint(np.mean(np.stack([m["n_nodes_above_threshold"].to_numpy(float) for m in maps]), axis=0)).astype(int)
    m0["fitted"] = F.all(0)
    m0.loc[~m0["fitted"], ["az_deg", "el_deg", "width_deg"]] = np.nan
    return m0


def rf_per_type(m0: pd.DataFrame, n_runs: int, blank_maps: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """Per type: coverage (fraction of bodies with a fitted centre), widths, peaks, the agreement with the anatomical
    column (median great-circle distance; fractions within 10 / 15 deg), the centre scatter over runs, the
    criterion-free retinotopy check (`peak_within_15deg_of_anat`: the fraction of bodies whose PEAK node -- fitted or
    not -- lies within 15 deg of the anatomical column, against `chance_within_15deg`, the fraction of grid nodes
    within 15 deg of that column), the blank-arm noise (`z_blank_median` = |peak| over the SD of the same cell's
    blank-arm per-node responses) and -- when the blank arms were fitted with the same rule -- the false-fit rate
    (bodies 'fitted' on a blank arm)."""
    rows = []
    for t, g in m0.groupby("type", sort=True):
        ok = g["fitted"].to_numpy(bool); d = g["anat_distance_deg"].to_numpy(float) if "anat_distance_deg" in g else np.full(len(g), np.nan)
        dd = d[np.isfinite(d)]
        row = {"type": t, "n_bodies": int(len(g)), "n_fitted": int(ok.sum()), "coverage": float(ok.mean()),
               "n_fitted_all_runs": int((g["n_runs_fitted"] == n_runs).sum()), "n_fitted_any_run": int((g["n_runs_fitted"] > 0).sum()), "n_runs": n_runs,
               "width_median_deg": float(np.nanmedian(g.loc[ok, "width_deg"])) if ok.any() else np.nan,
               "peak_median": float(np.nanmedian(np.abs(g.loc[ok, "peak"]))) if ok.any() else np.nan,
               "peak_median_all": float(np.nanmedian(np.abs(g["peak"]))),
               "z_peak_median_all": float(np.nanmedian(g["z_peak"].replace(np.inf, np.nan))),
               "n_with_anatomy": int(len(dd)), "anat_distance_median_deg": float(np.median(dd)) if len(dd) else np.nan,
               "anat_within_10deg": float((dd <= 10).mean()) if len(dd) else np.nan, "anat_within_15deg": float((dd <= 15).mean()) if len(dd) else np.nan,
               "centre_spread_median_deg": float(np.nanmedian(g.loc[ok, "centre_spread_deg"])) if (n_runs > 1 and ok.any()) else np.nan,
               "az_median": float(np.nanmedian(g.loc[ok, "az_deg"])) if ok.any() else np.nan, "el_median": float(np.nanmedian(g.loc[ok, "el_deg"])) if ok.any() else np.nan}
        if "peak_anat_distance_deg" in g:
            pa = g["peak_anat_distance_deg"].to_numpy(float); pa = pa[np.isfinite(pa)]
            row.update({"peak_within_15deg_of_anat": float((pa <= 15).mean()) if len(pa) else np.nan,
                        "chance_within_15deg": float(np.nanmean(g["chance_within_15deg"])) if "chance_within_15deg" in g else np.nan,
                        "peak_within_30deg_of_anat": float((pa <= 30).mean()) if len(pa) else np.nan})
        if "z_blank" in g:
            zb = g["z_blank"].replace(np.inf, np.nan)
            row.update({"z_blank_median": float(np.nanmedian(zb)), "frac_z_blank_ge_3": float(np.nanmean(zb >= 3)), "frac_z_blank_ge_5": float(np.nanmean(zb >= 5)),
                        "blank_node_sd_median": float(np.nanmedian(g["blank_node_sd"]))})
        if blank_maps:
            fr = [float(b.loc[b["type"] == t, "fitted"].to_numpy(bool).mean()) for b in blank_maps]
            row.update({"false_fit_rate_blank_arm_mean": float(np.mean(fr)), "false_fit_rate_blank_arm_per_run": fr})
        rows.append(row)
    return pd.DataFrame(rows)


def rf_sensitivity(Rs: list, Rbs: list, az, el, types, z_mins=(3.0, 4.0, 5.0, 8.0), spacing_deg: float = 10.0) -> pd.DataFrame:
    """Per type and z_min: the coverage (fitted in every run) on the stimulus arms and the false-fit rate on the blank
    arms (fitted in any run), so the criterion's trade-off is on file."""
    rows = []
    for zm in z_mins:
        fs = np.stack([fit_rf(R, az, el, z_min=zm, spacing_deg=spacing_deg)["fitted"].to_numpy(bool) for R in Rs]).all(0)
        fb = np.stack([fit_rf(R, az, el, z_min=zm, spacing_deg=spacing_deg)["fitted"].to_numpy(bool) for R in Rbs]).any(0) if Rbs else None
        for t in sorted(set(types)):
            m = types == t
            rows.append({"type": t, "z_min": zm, "coverage_all_runs": float(fs[m].mean()), "n_fitted": int(fs[m].sum()),
                         "false_fit_rate_blank_any_run": float(fb[m].mean()) if fb is not None else np.nan})
    return pd.DataFrame(rows)


def cmd_rfmap(args) -> int:
    runs = sorted(p for g in args.runs for p in glob.glob(g))
    if not runs:
        sys.exit(f"no localizer node files match {args.runs}")
    if not args.no_verify:
        bad = verify_runs([Path(p).parent for p in runs], require_cuda=not args.allow_cpu)
        if bad:
            sys.exit(f"batch verification failed: {bad}")
    maps = [rf_map_from_nodes(p, z_min=args.z_min, window=args.window) for p in runs]
    blank_files = [p.replace("_nodes.npz", "_nodes_blank.npz") for p in runs]
    blank_maps = [rf_map_from_nodes(p, z_min=args.z_min, window=args.window) for p in blank_files if os.path.exists(p)]
    m0 = merge_maps(maps)
    from flyverse import connectome, retina as retina_mod
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    r = retina_mod.build_retina(c)
    an = anatomical_centres(c, r, m0["model_index"].to_numpy())
    m0 = pd.concat([m0.reset_index(drop=True), an.reset_index(drop=True)], axis=1)
    Rs = [node_responses(p, args.window)[0] for p in runs]
    Rbs = [node_responses(p, args.window)[0] for p in blank_files if os.path.exists(p)]
    _, node_az, node_el, types_, _, _, _ = node_responses(runs[0], args.window)
    with np.errstate(invalid="ignore"):
        m0["anat_distance_deg"] = angular_distance_deg(m0["az_deg"], m0["el_deg"], m0["anat_az_deg"], m0["anat_el_deg"])
        # criterion-free retinotopy: the peak node (of the first run) against the anatomical column, and its chance level
        m0["peak_anat_distance_deg"] = angular_distance_deg(maps[0]["peak_node_az_deg"], maps[0]["peak_node_el_deg"], m0["anat_az_deg"], m0["anat_el_deg"])
        aa, ee = m0["anat_az_deg"].to_numpy(float), m0["anat_el_deg"].to_numpy(float)
        near = np.zeros(len(m0)); okc = np.isfinite(aa)
        cache = {}
        for i in np.flatnonzero(okc):
            key = (round(aa[i], 3), round(ee[i], 3))
            if key not in cache:
                cache[key] = float((angular_distance_deg(node_az, node_el, aa[i], ee[i]) <= 15).mean())
            near[i] = cache[key]
        m0["chance_within_15deg"] = np.where(okc, near, np.nan)
        if Rbs:
            bsd = np.mean([R.std(0) for R in Rbs], axis=0)
            m0["blank_node_sd"] = bsd; m0["z_blank"] = np.abs(m0["peak"].to_numpy(float)) / np.where(bsd > 0, bsd, np.nan)
    pt = rf_per_type(m0, len(maps), blank_maps)
    common.print_table(pt.drop(columns=[c for c in pt.columns if c.endswith("_per_run")]), floatfmt="{:+.3f}")
    spacing = float(np.min(np.diff(np.unique(node_az)))) if len(np.unique(node_az)) > 1 else 10.0
    sens = rf_sensitivity(Rs, Rbs, node_az, node_el, types_, spacing_deg=spacing)
    print("coverage (fitted in every run) / false-fit rate (blank arm, any run) by z_min:")
    common.print_table(sens.pivot(index="type", columns="z_min", values=["coverage_all_runs", "false_fit_rate_blank_any_run"]).reset_index(), floatfmt="{:.3f}")
    csv = Path(args.csv); csv.parent.mkdir(parents=True, exist_ok=True)
    m0 = m0[RF_CSV_COLUMNS + [c for c in m0.columns if c not in RF_CSV_COLUMNS]]
    m0.to_csv(csv, index=False)
    prov_path = str(runs[0]).replace("_nodes.npz", "_prov.json")
    prov0 = json.load(open(prov_path, encoding="utf-8")) if os.path.exists(prov_path) else common.provenance(c)
    res = Result.new("trace", prov0)
    res.replicates = {"n": len(runs), "unit": "runs", "runs": [{"run_index": i, "file": str(p), "seed": _seed_of(p), "device": _device_of(p)} for i, p in enumerate(runs)],
                      "null": {"blank_arm_files": [p for p in blank_files if os.path.exists(p)], "reading": "the same fit on the blank arm: its 'fitted' fraction is the false-fit rate"}}
    res.add_table("rf_per_type", pt); res.add_table("rf_map", m0.drop(columns=["model_index"])); res.add_table("rf_sensitivity", sens)
    loc_params = _summary_of(runs[0]).get("params", {})
    res.summary = {"schema": RFMAP_SCHEMA, "rf_map_csv": str(csv), "n_runs": len(runs), "window": args.window, "z_min": args.z_min, "fit_rule": fit_rf.__doc__,
                   "localizer_params": loc_params,
                   "coverage": {row["type"]: row["coverage"] for row in pt.to_dict("records")},
                   "false_fit_rate_blank_arm": {row["type"]: row.get("false_fit_rate_blank_arm_mean") for row in pt.to_dict("records")},
                   "anat_distance_median_deg": {row["type"]: row["anat_distance_median_deg"] for row in pt.to_dict("records")},
                   "peak_within_15deg_of_anat": {row["type"]: (row.get("peak_within_15deg_of_anat"), row.get("chance_within_15deg")) for row in pt.to_dict("records")},
                   "optic_overrides": _optic_overrides_of(runs[0]), "columns": RF_CSV_COLUMNS,
                   "reading": "coverage and agreement are one map's magnitudes; the centre scatter over runs (centre_spread_deg) is the reproducibility on file; "
                              "peak_within_15deg_of_anat vs chance_within_15deg is the criterion-free retinotopy check"}
    res.validation = {"name": "RF-map coverage, false-fit rate on the blank arm, and agreement with the anatomical column (trace.column_of_cells)",
                      "measured": {row["type"]: {k: row[k] for k in ("coverage", "anat_distance_median_deg", "anat_within_10deg", "false_fit_rate_blank_arm_mean",
                                                                     "peak_within_15deg_of_anat", "chance_within_15deg", "z_blank_median") if k in row}
                                   for row in pt.to_dict("records")},
                      "status": "measured", "reference": {"Mi1": "hex-annotated medulla cells: the localizer's own ground truth for a column-sized RF"},
                      "source": "scripts/probe_synthetic_stimuli.py rfmap"}
    res.files = {"generator": "scripts/probe_synthetic_stimuli.py rfmap " + " ".join(map(shlex.quote, sys.argv[2:])), "runs": [str(p) for p in runs], "csv": str(csv)}
    p = res.save(args.json or common.default_json_path("trace", res.run_id))
    print(f"problems: {res.check() or 'none'}; written {csv} and {p}", flush=True)
    return 0


def _stem_of(nodes_path: str) -> str:
    s = str(nodes_path)
    for suf in ("_nodes_blank.npz", "_nodes.npz"):
        if s.endswith(suf):
            return s[:-len(suf)]
    return s


def _summary_of(nodes_path: str) -> dict:
    p = _stem_of(nodes_path) + "_summary.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def _seed_of(p):
    return _summary_of(p).get("seed")


def _device_of(p):
    return _summary_of(p).get("device")


def _optic_overrides_of(p):
    return _summary_of(p).get("optic_overrides")


# ---------------------------------------------------------------------------------------------------- verify
def verify_dir(d) -> pd.DataFrame:
    """Every run under `d` (by its <stem>_summary.json): the realised device in the summary, the provenance and the
    console (the 'device cuda' line), the checksum of the presented radiance against the stored stimulus, and that
    both arm recordings exist. A row is `ok` when all agree and the device is CUDA."""
    rows = []
    for s in sorted(Path(d).glob("*_summary.json")):
        stem = str(s)[:-len("_summary.json")]; name = Path(stem).name
        j = json.load(open(s, encoding="utf-8"))
        console = Path(stem + ".txt"); text = console.read_text(encoding="utf-8", errors="replace") if console.exists() else ""
        pp = Path(stem + "_prov.json"); prov = json.load(open(pp, encoding="utf-8")) if pp.exists() else {}
        dev_prov = prov.get("execution", {}).get("device")
        arms = j.get("arms", []); arms_ok = all(Path(stem + f"_{a}.npz").exists() for a in arms)
        checks = j.get("checksum_per_arm") or {"stim": j.get("checksum_equal_stim_vs_presented")}
        chk_ok = all(v in (True, None) for v in checks.values())
        rows.append({"run": name, "stimulus": j.get("stimulus"), "seed": j.get("seed"), "optic": json.dumps(j.get("optic_overrides", {})),
                     "device_summary": j.get("device"), "device_prov": dev_prov, "console_exists": console.exists(),
                     "console_cuda": "device cuda" in text, "console_written": "written " in text, "checksums_ok": chk_ok, "arms_ok": arms_ok,
                     "n_frames": j.get("n_frames"), "wall_s": j.get("wall_s"), "prov_exists": pp.exists()})
    have = {r["run"] for r in rows}
    for t in sorted(Path(d).glob("*.txt")):                       # a console with no summary: the job died before writing (a job line ending in
        name = t.name[:-4]                                         # `tail -3` reports exit 0 whatever python did, so absence is the only trace)
        if name in have or name.endswith("_console") or not any(tok in t.read_text(encoding="utf-8", errors="replace") for tok in ("FlyBrain", "Traceback", "stimulus")):
            continue
        rows.append({"run": name, "stimulus": None, "seed": None, "optic": None, "device_summary": None, "device_prov": None, "console_exists": True,
                     "console_cuda": "device cuda" in t.read_text(encoding="utf-8", errors="replace"), "console_written": False, "checksums_ok": False,
                     "arms_ok": False, "n_frames": None, "wall_s": None, "prov_exists": False})
    df = pd.DataFrame(rows)
    if len(df):
        df["cuda"] = df["device_summary"].astype(str).str.startswith("cuda") & (df["device_prov"].astype(str).str.startswith("cuda"))
        df["ok"] = df["cuda"] & df["checksums_ok"] & df["arms_ok"] & df["prov_exists"] & (~df["console_exists"] | (df["console_cuda"] & df["console_written"]))
    return df


def verify_runs(dirs, require_cuda: bool = True) -> list[str]:
    """The problems of the runs under `dirs` (empty = verified)."""
    bad = []
    for d in sorted(set(map(str, dirs))):
        df = verify_dir(d)
        if not len(df):
            bad.append(f"{d}: no *_summary.json"); continue
        for r in df.to_dict("records"):
            ok = r["ok"] if require_cuda else (r["checksums_ok"] and r["arms_ok"] and r["prov_exists"])
            if not ok:
                bad.append(f"{d}/{r['run']}: device {r['device_summary']}/{r['device_prov']} console_cuda {r['console_cuda']} checksums {r['checksums_ok']} arms {r['arms_ok']}")
    return bad


def cmd_verify(args) -> int:
    df = verify_dir(args.dir)
    if not len(df):
        sys.exit(f"no *_summary.json under {args.dir}")
    common.print_table(df[["run", "stimulus", "seed", "optic", "device_summary", "device_prov", "console_cuda", "checksums_ok", "arms_ok", "n_frames", "ok"]], max_rows=200)
    log = Path(args.log) if args.log else None
    if log and log.exists():
        txt = log.read_text(encoding="utf-8", errors="replace")
        line = [l for l in txt.splitlines() if "job(s)" in l]
        print(f"cluster log {log}: {line[-1] if line else 'no job(s) line'}")
    n_bad = int((~df["ok"]).sum())
    print(f"{len(df)} run(s), {n_bad} not verified")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"dir": str(args.dir), "runs": to_jsonable(df.to_dict("records")), "n_runs": int(len(df)), "n_bad": n_bad}, f, indent=1)
    return 1 if n_bad else 0


# ---------------------------------------------------------------------------------------------------- stationarity
def stationarity_stats(rec: common.Recording, skip_s: float = 1.0, win_frames: int = 10, gap_frames: int = 20) -> pd.DataFrame:
    """Per type, how much a recording moves on its own after `skip_s`: for a per-cell recording the median over cells of
    the SD over frames, of the frame-to-frame |difference|, and of the |difference| between two `win_frames`-frame
    window means `gap_frames` apart (the localizer's on-vs-base comparison), plus the SD and the strongest spectral
    peak of the population mean; for a pooled (localizer) recording the same on the population means. A blank arm
    should read ~0 on a settled deterministic lobe; what it reads instead is the floor every per-node response is
    compared against."""
    rows = []
    t0 = int(round(skip_s * 1000.0 / common.FRAME_MS))
    pooled = any(k.startswith("pooled_") for k in rec.quantities)
    keys = list(rec.meta.get("pooled_keys", [])) if pooled else sorted(set(rec.types.astype(str)))
    for q in ([k for k in rec.quantities if k.startswith("pooled_")] if pooled else [k for k in rec.quantities if k != "spike_count"]):
        X = rec.quantities[q].astype(np.float64)[t0:]
        if len(X) < win_frames + gap_frames + 2:
            continue
        for t in keys:
            x = X[:, [keys.index(t)]] if pooled else X[:, rec.types.astype(str) == t]
            if x.shape[1] == 0 or not np.isfinite(x).all():
                continue
            pop = x.mean(1); f = np.fft.rfftfreq(len(pop), common.FRAME_MS / 1000.0); P = np.abs(np.fft.rfft(pop - pop.mean())) ** 2
            k = int(np.argmax(P[1:]) + 1) if len(P) > 1 else 0
            wm = np.abs(x[:win_frames].mean(0) - x[gap_frames:gap_frames + win_frames].mean(0))
            rows.append({"quantity": q.replace("pooled_", ""), "type": t, "pooled": pooled, "n_cells": int(x.shape[1]), "n_frames": int(len(x)),
                         "cell_sd_over_frames_median": float(np.median(x.std(0))), "frame_diff_abs_median": float(np.median(np.abs(np.diff(x, axis=0)))),
                         "window_mean_diff_abs_median": float(np.median(wm)), "cell_abs_mean_median": float(np.median(np.abs(x).mean(0))),
                         "pop_mean_sd": float(pop.std()), "pop_first_third_sd": float(pop[:len(pop) // 3].std()), "pop_last_third_sd": float(pop[-(len(pop) // 3):].std()),
                         "pop_spectral_peak_hz": float(f[k]) if k else np.nan, "pop_spectral_peak_share": float(P[k] / P[1:].sum()) if k and P[1:].sum() > 0 else np.nan})
    return pd.DataFrame(rows)


def cmd_stationarity(args) -> int:
    files = sorted(p for g in args.runs for p in glob.glob(g))
    if not files:
        sys.exit(f"no recordings match {args.runs}")
    out = []
    for f in files:
        rec = common.Recording.load(f)
        df = stationarity_stats(rec, args.skip_s)
        df.insert(0, "arm", rec.meta.get("arm")); df.insert(0, "run", Path(f).name.replace(".npz", ""))
        df["optic_overrides"] = json.dumps(rec.meta.get("provenance", {}).get("model", {}).get("optic", {}).get("gain_fb"))
        df["device"] = rec.meta.get("provenance", {}).get("execution", {}).get("device"); df["seed"] = rec.meta.get("seed")
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    show = df[df["type"].isin(args.types.split(","))] if args.types else df
    common.print_table(show[["run", "arm", "quantity", "type", "n_cells", "cell_sd_over_frames_median", "frame_diff_abs_median", "window_mean_diff_abs_median",
                             "pop_mean_sd", "pop_first_third_sd", "pop_last_third_sd", "pop_spectral_peak_hz", "pop_spectral_peak_share"]], floatfmt="{:.5f}", max_rows=200)
    if args.json:
        rec0 = common.Recording.load(files[0]); prov = rec0.meta.get("provenance")
        if prov is None or any(k not in prov for k in common.REQUIRED_PROVENANCE):
            from flyverse import connectome
            c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
            prov = dict(common.provenance(c), **(prov or {}))
        res = Result.new("trace", prov)
        res.add_table("stationarity", df)
        res.summary = {"files": files, "skip_s": args.skip_s, "definition": stationarity_stats.__doc__}
        res.validation = {"name": "stationarity of a blank arm (the floor of every within-run window comparison)", "measured": None, "status": "measured",
                          "reference": "a settled deterministic lobe reads ~0 (optic_measures.md 6: two blank runs agree to 1e-9, which does not test this)",
                          "source": "scripts/probe_synthetic_stimuli.py stationarity"}
        res.files = {"generator": "scripts/probe_synthetic_stimuli.py stationarity " + " ".join(map(shlex.quote, sys.argv[2:]))}
        p = res.save(args.json)
        print(f"problems: {res.check() or 'none'}; written {p}")
    return 0


def divergence_stats(stim: common.Recording, blank: common.Recording, on: np.ndarray) -> pd.DataFrame:
    """Per type (pooled localizer recordings): |stim - blank| of the population mean before the first flash (the two
    arms share the seed and the settle, so a deterministic lobe reads 0 there), during the first flash, over the 30
    frames after it, over the last third of the run, and the correlation of the two arms over that last third. A
    stable fixed point re-converges after a flash; a sustained oscillation that a flash perturbs stays decorrelated."""
    rows = []
    keys = list(stim.meta.get("pooled_keys", []))
    first = int(np.flatnonzero(on)[0]) if on.any() else 0
    end = first
    while end < len(on) and on[end]:
        end += 1
    T = min(stim.n_frames, blank.n_frames); third = T // 3
    for q in [k for k in stim.quantities if k.startswith("pooled_") and "spike" not in k]:
        S, B = stim.quantities[q].astype(np.float64)[:T], blank.quantities[q].astype(np.float64)[:T]
        for j, t in enumerate(keys):
            s, b = S[:, j], B[:, j]
            if not (np.isfinite(s).all() and np.isfinite(b).all()) or s.std() == 0:
                continue
            d = np.abs(s - b)
            rows.append({"quantity": q.replace("pooled_", ""), "type": t, "first_flash_frame": first,
                         "before_first_flash_max": float(d[:first].max()) if first else np.nan, "during_first_flash_max": float(d[first:end].max()),
                         "after_first_flash_30_max": float(d[end:end + 30].max()), "last_third_mean": float(d[-third:].mean()),
                         "blank_last_third_sd": float(b[-third:].std()), "corr_last_third": float(np.corrcoef(s[-third:], b[-third:])[0, 1])})
    return pd.DataFrame(rows)


def cmd_divergence(args) -> int:
    out = []
    for stem in args.stems:
        stim, blank = common.Recording.load(stem + "_stim.npz"), common.Recording.load(stem + "_blank.npz")
        rad = np.load(stem + "_radiance.npz", allow_pickle=False)
        if "track__on" not in rad.files:
            print(f"  skip {stem}: not a localizer"); continue
        df = divergence_stats(stim, blank, rad["track__on"]); df.insert(0, "run", Path(stem).name)
        df["optic_gain_fb"] = stim.meta.get("provenance", {}).get("model", {}).get("optic", {}).get("gain_fb"); df["device"] = stim.meta.get("provenance", {}).get("execution", {}).get("device")
        out.append(df)
    if not out:
        sys.exit("nothing to compare")
    df = pd.concat(out, ignore_index=True)
    common.print_table(df[df["type"].isin(args.types.split(","))] if args.types else df, floatfmt="{:.4g}", max_rows=200)
    if args.json:
        rec0 = common.Recording.load(args.stems[0] + "_stim.npz"); prov = rec0.meta.get("provenance")
        if prov is None or any(k not in prov for k in common.REQUIRED_PROVENANCE):
            from flyverse import connectome
            c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
            prov = dict(common.provenance(c), **(prov or {}))
        res = Result.new("trace", prov); res.add_table("divergence", df)
        res.summary = {"stems": args.stems, "definition": divergence_stats.__doc__}
        res.validation = {"name": "stimulus-vs-blank trajectory divergence of a localizer pair (same seed, same settle)", "measured": None, "status": "measured",
                          "reference": "a deterministic lobe: 0 before the first flash", "source": "scripts/probe_synthetic_stimuli.py divergence"}
        res.files = {"generator": "scripts/probe_synthetic_stimuli.py divergence " + " ".join(map(shlex.quote, sys.argv[2:]))}
        p = res.save(args.json); print(f"problems: {res.check() or 'none'}; written {p}")
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
    from flyverse import connectome
    dirs = [Path(x) for x in (args.dir if isinstance(args.dir, list) else [args.dir])]; d = dirs[0]
    if not args.no_verify:
        bad = verify_runs(dirs, require_cuda=not args.allow_cpu)
        if bad:
            sys.exit("batch verification failed:\n  " + "\n  ".join(bad))
    stim_files = sorted(p for dd in dirs for p in dd.glob("*_stim.npz")) + sorted(p for dd in dirs for p in dd.glob("*_blank_a.npz"))
    if not stim_files:
        sys.exit(f"no *_stim.npz / *_blank_a.npz under {dirs}")
    rfmap = pd.read_csv(args.rf_map, dtype={"bodyId": str}) if args.rf_map else None
    per_type_rows, per_body, tcs, families = [], [], [], {}
    win_s = float(getattr(args, "transition_window_s", TRANSITION_WINDOW_S))
    trans_masks: dict = {}                             # run -> its ON / OFF frame masks (the null is scored on them too)
    prov0 = None
    for sf in stim_files:
        stem = str(sf)[:-len("_stim.npz")] if sf.name.endswith("_stim.npz") else str(sf)[:-len("_blank_a.npz")]
        bf = Path(stem + ("_blank.npz" if sf.name.endswith("_stim.npz") else "_blank_b.npz"))
        if not bf.exists():
            print(f"  skip {sf.name}: no matched blank"); continue
        stim, blank = common.Recording.load(sf), common.Recording.load(bf)
        rad = dict(np.load(stem + "_radiance.npz", allow_pickle=False))
        name = Path(stem).name
        if str(rad["name"]) == "localizer":
            continue                                                    # the localizer's map is rfmap's; its arms are not a family
        fam = family_stats(stim, blank)
        params = json.loads(str(rad["params"]))
        split = transition_stats(stim, blank, str(rad["name"]), params, rad, win_s, float(rad["dt_s"]))
        families[name] = {"stimulus": str(rad["name"]), "params": json.loads(str(rad["params"])), "arms": [sf.name, bf.name], "per_type": fam,
                          "device": stim.meta.get("provenance", {}).get("execution", {}).get("device"), "seed": stim.meta.get("seed"),
                          "null": sf.name.endswith("_blank_a.npz"), "optic_overrides": stim.meta.get("provenance", {}).get("stimulus", {}).get("optic_overrides")}
        if prov0 is None and os.path.exists(stem + "_prov.json"):
            prov0 = json.load(open(stem + "_prov.json", encoding="utf-8"))
        print(f"\n== {name}: {rad['name']} (device {families[name]['device']}{', blank vs blank' if families[name]['null'] else ''})"); print_family(fam)
        is_null = families[name]["null"]
        for t, v in fam.items():
            per_type_rows.append({"run": name, "stimulus": str(rad["name"]), "null": is_null, "type": t,
                                  "transition": "pooled", "edge_schedule": name, "n_frames_used": int(stim.n_frames), **v})
        if split is not None:                          # the ON / OFF rows beside the pooled one (flash / flicker)
            families[name]["transitions"] = split
            print_transitions(split)
            for k, pt in split["per_type"].items():
                for t, v in pt.items():
                    per_type_rows.append({"run": name, "stimulus": str(rad["name"]), "null": is_null, "type": t,
                                          "transition": k, "edge_schedule": name, "n_frames_used": int(split["n_frames"][k]), **v})
            if not is_null:
                trans_masks[name] = transition_masks(str(rad["name"]), params, rad, win_s, float(rad["dt_s"]), stim.n_frames)
        if is_null and trans_masks:                    # the split's own floor: blank/blank scored on each family's edge schedule
            floor = {}
            for src, tf in trans_masks.items():
                for k in ("on", "off"):
                    mm = clip_mask(tf[k], stim.n_frames)
                    if int(mm.sum()) < 2:
                        continue
                    pt = family_stats(stim, blank, frames=mm)
                    floor[f"{src}:{k}"] = {"n_frames": int(mm.sum()), "per_type": pt}
                    for t, v in pt.items():
                        per_type_rows.append({"run": name, "stimulus": str(rad["name"]), "null": True, "type": t,
                                              "transition": k, "edge_schedule": src, "n_frames_used": int(mm.sum()), **v})
            if floor:
                families[name]["transition_floor"] = floor
                print(f"  -- the ON / OFF split's own floor: these blank/blank arms scored on the edge schedule of {sorted(trans_masks)}")
                for lab, f in floor.items():
                    print(f"  -- floor {lab} ({f['n_frames']} frames)"); print_family(f["per_type"])
        tc = time_course(stim, blank); tc.insert(0, "run", name); tcs.append(tc)
        pb = per_body_rows(stim, blank)
        if rfmap is not None:
            rw = rf_window_stats(stim, blank, rad, rfmap)
            if len(rw):
                pb = pb.merge(rw[["bodyId", "rf_az_deg", "rf_el_deg", "n_frames_in_rf", "diff_in_rf"]], on="bodyId", how="left")
                g = rw.groupby("type").agg(n_bodies_with_rf=("bodyId", "size"), n_bodies_object_in_rf=("n_frames_in_rf", lambda x: int((x > 0).sum())),
                                           diff_in_rf_mean=("diff_in_rf", "mean"), diff_in_rf_best_body=("diff_in_rf", lambda x: float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else np.nan),
                                           diff_whole_window_mean=("diff_whole_window", "mean")).reset_index()
                families[name]["rf_window"] = g.to_dict("records")
                print("  read through the RF map (per body, object inside its fitted RF):"); common.print_table(g, floatfmt="{:+.5f}")
        pb.insert(0, "run", name); per_body.append(pb)
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    res = Result.new("trace", prov0 or common.provenance(c))
    res.replicates = {"n": 1, "unit": "runs", "runs": [{"run_index": i, "file": k, "seed": v["seed"], "device": v["device"]} for i, (k, v) in enumerate(families.items())],
                      "null": [k for k, v in families.items() if v["null"]] or None}
    res.add_table("per_type", pd.DataFrame(per_type_rows))
    if per_body:
        res.add_table("per_body", pd.concat(per_body, ignore_index=True))
    if tcs:
        res.add_table("time_course", pd.concat(tcs, ignore_index=True))
    res.summary = {"families": families, "n_runs_per_family": 1,
                   "verdicts": "none: one run per family -- magnitudes only (docs/INTERP.md 2.4: >= 4-5 runs per arm in one submission before a difference is called)",
                   "rf_map": args.rf_map, "statistics": {"spiking": ["diff_max_over_cells_mean_mv", "diff_mean_over_cells_mean_mv", "diff_rate_hz_max_cell"],
                                                           "rate": ["diff_signed_best_cell", "diff_abs_best_cell_mean", "diff_signed_mean"],
                                                           "time_course": ["tc_peak_pop_mean", "tc_peak_t_s", "tc_abs_mean"]},
                   "transition_split": {"column": "transition", "values": ["pooled", "on", "off"], "schedule_column": "edge_schedule",
                                        "null_rows": "a blank/blank run carries on / off rows too, scored on each family's edge schedule "
                                                     "(edge_schedule names it): the floor of the window statistic, since a blank arm has no edge",
                                        "window_s": win_s,
                                        "reading": "`pooled` is the whole-window mean; for the PERIODIC flash it holds both edges, so a "
                                                   "flash_on run against a flash_off run separates BRIGHT from DARK. `on` / `off` are the "
                                                   "isolated transition measurements (the window after each rising / falling luminance edge "
                                                   "of that run), the ON / OFF split Neurome asked for."}}
    res.validation = {"name": "synthetic families on the shipped lobe (one instance each)", "measured": {k: v["per_type"] for k, v in families.items()},
                      "status": "measured", "reference": None, "source": "scripts/probe_synthetic_stimuli.py analyse"}
    res.files = {"generator": "scripts/probe_synthetic_stimuli.py analyse " + " ".join(map(shlex.quote, sys.argv[2:])), "dir": [str(x) for x in dirs]}
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


def batch_jobs(seconds: float = 3.0, settle: float = 2.0, seed: int = 0, loc_runs: int = 3) -> list[tuple[str, str]]:
    """(job name, record arguments) of the validation batch: the localizer on the deterministic lobe (x1) and the shipped
    lobe (x loc_runs), one instance of each stimulus family on the shipped lobe, and a blank/blank null."""
    common_ = f"--seconds {seconds} --settle {settle}"
    jobs = [("loc_fb0_s0", f"--stimulus localizer --settle {settle} --optic gain_fb=0 --seed {seed}")]
    jobs += [(f"loc_shipped_s{k}", f"--stimulus localizer --settle {settle} --seed {seed + k}") for k in range(loc_runs)]
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


def ladder_jobs(seconds: float = 3.0, settle: float = 2.0, seed: int = 0, runs: int = 5, contrasts=("dark", "bright"), extra_optic: str = "") -> list[tuple[str, str]]:
    """(job name, record arguments) of the size ladders for the matched assay proper: the height ladder (width 4.4), the
    width ladder (height 8.8) and the square ladder, dark and bright, `runs` runs per arm plus `runs` blank/blank nulls
    -- all arms of one family in ONE submission (docs/INTERP.md 2.4). Split by `--family` when the box count is small."""
    common_ = f"--seconds {seconds} --settle {settle}" + (f" --optic {shlex.quote(extra_optic)}" if extra_optic else "")
    con = {"dark": DARK_CONTRAST, "bright": -DARK_CONTRAST}
    arms = []
    for h in HEIGHT_LADDER["heights"]:
        arms.append((f"hlad_h{h * 10:03.0f}", f"--stimulus rect --width {HEIGHT_LADDER['width']} --height {h}"))
    for w in WIDTH_LADDER["widths"]:
        arms.append((f"wlad_w{w * 10:03.0f}", f"--stimulus rect --width {w} --height {WIDTH_LADDER['height']}"))
    for s in SQUARE_LADDER:
        arms.append((f"sqlad_s{s * 10:03.0f}", f"--stimulus rect --width {s} --height {s}"))
    jobs = []
    for name, a in arms:
        for cname in contrasts:
            for k in range(runs):
                jobs.append((f"{name}_{cname}_r{k}", f"{a} --contrast {con[cname]} {common_} --seed {seed + k}"))
    for k in range(runs):
        jobs.append((f"null_r{k}", f"--stimulus rect --width 4.4 --height 8.8 --contrast {DARK_CONTRAST} {common_} --seed {seed + k} --null"))
    return jobs


def write_batch(out: str, name: str, minutes: int, jobs: list[tuple[str, str]], note: str, script: str = "batch.sh") -> Path:
    lines = [f"mkdir -p {out} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
             f"python scripts/probe_synthetic_stimuli.py record {a} --out {out}/{jn} > {out}/{jn}.txt 2>&1; tail -3 {out}/{jn}.txt" for jn, a in jobs]
    Path(out).mkdir(parents=True, exist_ok=True)
    log = f"out/{name}_cluster.log"
    sh = ["#!/bin/sh", f"# generated by scripts/probe_synthetic_stimuli.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs ({note})", "set -e",
          f"mkdir -p {out} out", f'if [ -f {log} ]; then mv {log} "out/{name}_cluster.$(date +%Y%m%dT%H%M%S).log"; fi',
          f"python scripts/cluster_run.py --name {name} --minutes {minutes} \\"]
    sh += [f"  {shlex.quote(l)} \\" for l in lines]
    sh += [f"  --fetch {out}/ 2>&1 | tee {log}"]
    p = Path(out) / script
    p.write_text("\n".join(sh) + "\n", encoding="utf-8")
    with open(Path(out) / (script.replace(".sh", "") + "_jobs.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "minutes": minutes, "jobs": [{"job": jn, "args": a} for jn, a in jobs], "note": note}, f, indent=1)
    return p


def cmd_plan(args) -> int:
    out = args.out.rstrip("/")
    if args.ladders:
        fam = args.family
        jobs = ladder_jobs(args.seconds, args.settle, args.seed, args.runs, extra_optic=args.extra_optic)
        if fam:
            jobs = [j for j in jobs if j[0].startswith(fam) or j[0].startswith("null_")]
        note = f"size ladders x dark/bright x {args.runs} runs + {args.runs} nulls" + (f", family {fam}" if fam else "")
    else:
        jobs = batch_jobs(args.seconds, args.settle, args.seed, args.loc_runs)
        note = f"localizer fb0 x1 + shipped x{args.loc_runs}; one instance per stimulus family; one blank/blank null"
    p = write_batch(out, args.name, args.minutes, jobs, note)
    print(f"written {p}: {len(jobs)} jobs ({note}); run: sh {p}")
    return 0


# ---------------------------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="one stimulus + its matched blank on the GPU"); stimulus_args(r); common.add_common_args(r)
    r.add_argument("--out", required=True, help="output stem: <out>_stim.npz, <out>_blank.npz, <out>_radiance.npz, <out>_nodes[.|_blank.]npz, <out>_prov.json, <out>_summary.json")
    r.add_argument("--settle", type=float, default=2.0, help="settling time on the blank before the window (s)")
    r.add_argument("--select", default=",".join(SELECTION), help="recorded populations (common grammar, comma-separated)")
    r.add_argument("--null", action="store_true", help="blank vs blank (arms blank_a / blank_b)")
    r.add_argument("--allow-cpu", action="store_true", help="CPU smoke (a cluster job must run on CUDA)")
    r.add_argument("--transition-window-s", type=float, default=TRANSITION_WINDOW_S,
                   help="analysis-side ON / OFF split: the window after each luminance edge summarised separately (flash / flicker)")
    pv = sub.add_parser("preview", help="geometry check on the shipped retina (CPU)"); stimulus_args(pv); pv.add_argument("--cache-dir", default=None)
    m = sub.add_parser("rfmap", help="the RF map from localizer runs (CPU)")
    m.add_argument("--runs", nargs="+", required=True, help="glob(s) of <out>_nodes.npz files; several runs = the map is their mean and their scatter is on file")
    m.add_argument("--csv", required=True); m.add_argument("--json", default=None); m.add_argument("--z-min", type=float, default=5.0)
    m.add_argument("--window", default="on", choices=["on", "off", "onoff"]); m.add_argument("--cache-dir", default=None)
    m.add_argument("--no-verify", action="store_true", help="skip the batch verification"); m.add_argument("--allow-cpu", action="store_true", help="verify without requiring CUDA (smoke)")
    a = sub.add_parser("analyse", help="per-family summary of a directory of runs, read through an RF map (CPU)")
    a.add_argument("--dir", required=True, nargs="+", help="run directories (several = fetched by separate clients, e.g. a resubmitted job)")
    a.add_argument("--rf-map", default=None); a.add_argument("--json", default=None); a.add_argument("--cache-dir", default=None)
    a.add_argument("--no-verify", action="store_true"); a.add_argument("--allow-cpu", action="store_true")
    a.add_argument("--transition-window-s", type=float, default=TRANSITION_WINDOW_S,
                   help="analysis-side ON / OFF split: the window after each luminance edge summarised separately, as rows "
                        "transition = on | off beside the pooled whole-window row (flash / flicker)")
    st = sub.add_parser("stationarity", help="how much a recording (a blank arm) moves on its own: the floor of every within-run window comparison (CPU)")
    st.add_argument("--runs", nargs="+", required=True, help="glob(s) of recordings (<out>_blank.npz, <out>_blank_a.npz, a localizer's pooled arm ...)")
    st.add_argument("--skip-s", type=float, default=1.0); st.add_argument("--types", default="LC11,LC10a,T2,T3,Tm5Y,TmY21,Mi1"); st.add_argument("--json", default=None)
    st.add_argument("--cache-dir", default=None)
    dv = sub.add_parser("divergence", help="stimulus-vs-blank trajectory divergence of localizer pairs (pooled arms; CPU)")
    dv.add_argument("--stems", nargs="+", required=True, help="run stems (<stem>_stim.npz / _blank.npz / _radiance.npz)")
    dv.add_argument("--types", default="LC11,LC10a,T2,T3,Tm5Y,TmY21,Mi1"); dv.add_argument("--json", default=None); dv.add_argument("--cache-dir", default=None)
    v = sub.add_parser("verify", help="console-vs-summary-vs-provenance check of a fetched batch (CPU)")
    v.add_argument("--dir", required=True); v.add_argument("--log", default=None, help="the cluster console log (its 'job(s)' line is echoed)"); v.add_argument("--json", default=None)
    pl = sub.add_parser("plan", help="write the cluster batch (the validation batch, or --ladders for the size runs)")
    pl.add_argument("--out", default="out/synth"); pl.add_argument("--name", default="synth"); pl.add_argument("--minutes", type=int, default=45)
    pl.add_argument("--seconds", type=float, default=3.0); pl.add_argument("--settle", type=float, default=2.0); pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--loc-runs", type=int, default=3)
    pl.add_argument("--ladders", action="store_true", help="the height / width / square ladders x dark / bright x --runs, plus --runs nulls")
    pl.add_argument("--runs", type=int, default=5); pl.add_argument("--family", default=None, help="hlad | wlad | sqlad: one ladder only")
    pl.add_argument("--extra-optic", default="", help="an --optic override for every ladder job (a hook arm)")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    argv_ = sys.argv[1:] if argv is None else list(argv)
    args.contrast_set = _flag_set(argv_, "--contrast"); args.width_set = _flag_set(argv_, "--width")
    return {"record": cmd_record, "preview": cmd_preview, "rfmap": cmd_rfmap, "analyse": cmd_analyse, "plan": cmd_plan, "verify": cmd_verify,
            "stationarity": cmd_stationarity, "divergence": cmd_divergence}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
