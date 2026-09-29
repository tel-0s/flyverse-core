"""tron -- light cycles (games/PLAN.md section 9).

A TRON arena: a black floor with a glowing grid, a square arena wall, and two light cycles at a constant speed that
leave jet walls behind them and turn only in 90 degree steps. The fly's eyes ride a cycle, 1.0 m above the floor and
looking along its heading; its 1,466 ommatidial columns see the arena ray traced on the GPU (emissive walls and a
glossy floor), the same scene the chase camera shows.

The fly's only control is one declared decoder on its escape pathway (`games.common.ReadDecoder`, read-only):

    turn trigger (DECODER)   the cycle turns 90 degrees on each upward crossing of the giant fibres' mean rate
                             (DNp01 L/R) through 33 Hz -- the body model's takeoff line (flyverse/body.py,
                             Flight.gf_hz) -- no sooner than REARM_S after the previous turn;
    turn side    (DECODER)   s = (mean LPLC2 + mean LC4, left) - (same, right); s' = mean of s over the 100 ms before
                             the crossing minus the brain's resting offset SIDE_OFFSET_HZ (-2.1 Hz);
                             s' > 0 -> turn right, else turn left. (Without the offset nearly every turn goes left.)

Everything else is GAME: the arena, the constant speed, the 90-degree turn itself, collisions, the round schedule
(the same start positions for every seed), the opponent's rules when it is not a fly, and the camera.

After a connected fly run, the run log's `summary.baselines` holds brain-free GAME replays on the same tick timeline:
the fly's own turns (a determinism check), a never-turning cyan cycle, and BASELINE_DRAWS random cyan players at the
fly run's own turn rate, all against an open-loop replay of the orange cycle's recorded turns; the metric is cyan
crashes per minute of riding, and `rank_of_fly` says where the fly sits among the draws.

    python games/tron.py --probe out/games/tron/probe_malecns.json --seed 100          # open-loop measurements
    python games/tron.py --seed 100 --seconds 40 --record out/games/tron/dev.mp4        # a dev clip
    python games/tron.py --seed 100 --seconds 40 --control --record out/games/tron/dev_control.mp4
    python games/tron.py --seed 100 --seconds 40 --player random --record out/games/tron/dev_random.mp4
    python games/tron.py --baselines-from out/games/tron/dev.json                      # recompute baselines, CPU
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (puts the repo on sys.path)
import torch  # noqa: E402

# ======================================================================================================== constants
# World frame: x, y on the floor, z up; metres and seconds. Colours are linear RGB; the fly gets rgb_to_radiance
# (UV = 0.5 B, logged by RunLog as `uv_from_rgb`).
ARENA_HALF = 30.0              # GAME: the arena is [-30, 30] m square
ARENA_WALL_H = 6.0             # GAME: the arena's wall height, m
TRAIL_H = 3.5                  # GAME: jet-wall height, m
TRAIL_HALF_W = 0.10            # GAME: jet-wall half thickness, m
EYE_H = 1.0                    # GAME: the fly's eyes ride 1.0 m above the floor, on the cycle's centre line
EYE_FWD = 1.3                  # ... at the cycle's nose (1.3 m ahead of its centre): it meets a wall when the cycle does
CYCLE_LEN, CYCLE_W, CYCLE_H = 2.6, 0.7, 1.15    # GAME: the cycle's body box (m)
FRONT = CYCLE_LEN / 2          # collision point: the cycle's nose, this far ahead of its centre
SPEED = 10.0                   # GAME: constant cycle speed, m/s (chosen on dev seed 100, games/captions/tron.md)
GRID = 5.0                     # floor grid spacing, m
GRID_LINE = 0.07               # floor grid line half width, m
GF_THRESHOLD_HZ = 33.0         # flyverse.body.Flight.gf_hz (checked at start-up)
REARM_S = 0.30                 # DECODER: minimum time between two turns
SIDE_WINDOW_S = 0.10           # DECODER: the side signal is averaged over this window before the crossing
TICK_S = common.TICK_MS / 1000.0
STYLE = "dark"                 # GAME: the scene look (see STYLES), chosen on dev seeds for what the fly's eye responds to

HEADINGS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])   # 0 +x, 1 +y, 2 -x, 3 -y (CCW)

CYAN = np.array([0.10, 0.85, 1.00])
ORANGE = np.array([1.00, 0.42, 0.06])
GRID_RGB = np.array([0.10, 0.30, 0.85])
WALL_RGB = np.array([0.12, 0.30, 1.00])

KIND_ARENA, KIND_TRAIL, KIND_CYCLE = 0, 1, 2

# Scene looks (GAME). 'glow': emissive jet walls on a black arena. 'dark_*': walls of dark glass with bright edges
# (and 1 m vertical light bands with '_stripes') in front of a brighter floor grid and horizon, so an approaching wall
# covers bright background (an OFF loom).
STYLES = {
    "glow": {"cam_exposure": 1.0, "sky": [0.0, 0.002, 0.006], "horizon": [0.015, 0.035, 0.09], "floor": [0.004, 0.006, 0.012],
             "grid_gain": 0.9, "trail_body": (0.22, 0.50), "trail_base": False, "stripes": 0.0, "stripe_gain": 0.0},
    "glow_stripes": {"cam_exposure": 1.0, "sky": [0.0, 0.002, 0.006], "horizon": [0.015, 0.035, 0.09], "floor": [0.004, 0.006, 0.012],
                     "grid_gain": 0.9, "trail_body": (0.10, 0.25), "trail_base": True, "stripes": 1.0,
                     "stripe_gain": 0.9},
    "dark": {"cam_exposure": 0.32, "sky": [0.02, 0.04, 0.09], "horizon": [0.14, 0.28, 0.55], "floor": [0.02, 0.035, 0.07],
             "grid_gain": 1.8, "trail_body": (0.02, 0.03), "trail_base": True, "stripes": 0.0, "stripe_gain": 0.0},
    "dark_stripes": {"cam_exposure": 0.32, "sky": [0.02, 0.04, 0.09], "horizon": [0.14, 0.28, 0.55], "floor": [0.02, 0.035, 0.07],
                     "grid_gain": 1.8, "trail_body": (0.02, 0.03), "trail_base": True, "stripes": 1.0,
                     "stripe_gain": 0.7},
}


# ======================================================================================================== geometry
def turn_heading(h: int, side: str) -> int:
    """90-degree turn: 'L' is counter-clockwise (toward +y when heading +x), 'R' clockwise."""
    return (int(h) + (1 if side == "L" else -1)) % 4


def heading_vec(h: int) -> np.ndarray:
    return HEADINGS[int(h) % 4]


def left_vec(h: int) -> np.ndarray:
    return HEADINGS[(int(h) + 1) % 4]


def segment_box(a, b, half_w=TRAIL_HALF_W, height=TRAIL_H):
    """An axis-aligned jet-wall box (lo, hi) along the floor segment a -> b, thickened by half_w on every side."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo = np.minimum(a, b) - half_w
    hi = np.maximum(a, b) + half_w
    return np.array([lo[0], lo[1], 0.0]), np.array([hi[0], hi[1], height])


def arena_boxes(half=ARENA_HALF, h=ARENA_WALL_H, t=1.0):
    """The four arena walls as boxes whose inner faces sit at +-half."""
    out = []
    for lo, hi in (((-half - t, -half - t), (half + t, -half)), ((-half - t, half), (half + t, half + t)),
                   ((-half - t, -half), (-half, half)), ((half, -half), (half + t, half))):
        out.append((np.array([lo[0], lo[1], 0.0]), np.array([hi[0], hi[1], h])))
    return out


def cycle_box(pos, h):
    """The cycle's body as an axis-aligned box (headings are axis-aligned)."""
    f, lft = heading_vec(h), left_vec(h)
    c = np.asarray(pos, float)
    ext = np.abs(f) * CYCLE_LEN / 2 + np.abs(lft) * CYCLE_W / 2
    return np.array([c[0] - ext[0], c[1] - ext[1], 0.05]), np.array([c[0] + ext[0], c[1] + ext[1], CYCLE_H])


def sweep_hit(p0, p1, boxes, pad):
    """Earliest fraction s in [0, 1] at which the floor point moving p0 -> p1 enters any box (lo, hi) grown by `pad`
    in x and y, or None. Returns (s, index)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    best = (None, None)
    for k, (lo, hi) in enumerate(boxes):
        lo2, hi2 = lo[:2] - pad, hi[:2] + pad
        t0, t1 = 0.0, 1.0
        ok = True
        for ax in range(2):
            if abs(d[ax]) < 1e-12:
                if p0[ax] < lo2[ax] or p0[ax] > hi2[ax]:
                    ok = False
                    break
            else:
                a, b = (lo2[ax] - p0[ax]) / d[ax], (hi2[ax] - p0[ax]) / d[ax]
                if a > b:
                    a, b = b, a
                t0, t1 = max(t0, a), min(t1, b)
                if t0 > t1:
                    ok = False
                    break
        if ok and (best[0] is None or t0 < best[0]):
            best = (t0, k)
    return best


# ======================================================================================================== decoder law
class TurnLaw:
    """The fly's one control, as pure logic. `update(t, gf, side)` once per 10 ms tick with the giant fibres' mean
    rate and the loom side signal s = (LPLC2 + LC4)_L - (LPLC2 + LC4)_R (population means, Hz); returns 'L', 'R' or
    None. A turn fires on an upward crossing of `threshold` (the previous tick below it, this one at or above), and
    no sooner than `rearm_s` after the last turn; its side is away from the more-looming eye: the mean s over the
    last `window_s`, minus `side_offset` (the brain's resting L - R while riding open floor, measured on a dev seed),
    > 0: the left eye looms more -> turn right; <= 0 -> turn left. `live=False` (a round's
    GET READY, a crashed cycle) keeps tracking the rate but never fires."""

    def __init__(self, threshold=GF_THRESHOLD_HZ, rearm_s=REARM_S, window_s=SIDE_WINDOW_S, side_offset=0.0):
        self.threshold, self.rearm_s, self.window_s = float(threshold), float(rearm_s), float(window_s)
        self.side_offset = float(side_offset)
        self.last_cross = None
        self.prev = 0.0
        self.last_turn = -1e9
        self.hist = []                          # (t, s) over the side window

    def parameters(self):
        return {"threshold_Hz": self.threshold, "rearm_s": self.rearm_s, "side_window_s": self.window_s,
                "side_offset_Hz": self.side_offset, "side_rule": "mean(s) - side_offset > 0 -> R, else L",
                "side_cells": ["LPLC2", "LC4"], "trigger_cells": ["DNp01"]}

    def side_signal(self):
        """Mean s over the window, minus the brain's resting offset (its mean s while riding open floor)."""
        return float(np.mean([s for _, s in self.hist])) - self.side_offset if self.hist else 0.0

    def update(self, t, gf, side, live=True):
        self.hist.append((t, float(side)))
        while self.hist and self.hist[0][0] <= t - self.window_s + 1e-9:
            self.hist.pop(0)
        crossed = self.prev < self.threshold <= gf
        self.prev = float(gf)
        # every upward crossing, and what became of it: 'turn', 'blocked_rearm' or 'not_live' (for the run log)
        self.last_cross = None
        if crossed:
            since = t - self.last_turn
            status = ("not_live" if not live else "blocked_rearm" if since < self.rearm_s - 1e-9 else "turn")
            self.last_cross = {"status": status, "since_last_turn_s": round(since, 3) if since < 1e8 else None}
        if not (live and crossed and t - self.last_turn >= self.rearm_s - 1e-9):
            return None
        self.last_turn = t
        return "R" if self.side_signal() > 0 else "L"


class RandomTurns:
    """GAME baseline: turns at Poisson times at `rate_hz` (the fly's own turn rate, measured on dev seeds), to a side
    drawn 50 / 50, with the same re-arm interval as the fly's decoder."""

    def __init__(self, rate_hz, rng, rearm_s=REARM_S):
        self.p = 1.0 - math.exp(-float(rate_hz) * TICK_S)
        self.rng, self.rearm_s, self.last_turn = rng, float(rearm_s), -1e9

    def update(self, t, live=True):
        u, side = self.rng.random(), self.rng.random()          # always drawn: the stream does not depend on `live`
        if not live or t - self.last_turn < self.rearm_s - 1e-9 or u >= self.p:
            return None
        self.last_turn = t
        return "L" if side < 0.5 else "R"


# ======================================================================================================== the tracer
def _dev_t(x, device):
    return torch.as_tensor(np.asarray(x, np.float32), device=device)


class Renderer:
    """A small emissive ray tracer on torch: the floor (glossy black, footprint-filtered grid), the sky, and
    axis-aligned boxes (arena walls, jet walls, cycle bodies). `trace(o, d, boxes, pix)` -> (M, 3) linear RGB; `pix`
    is each ray's angular footprint (rad), used to filter the grid lines so neither the camera nor the fly's 7 rays
    per column alias them."""

    def __init__(self, device, style="glow"):
        self.device = torch.device(device)
        self.style = style

    @staticmethod
    def pack(boxes):
        """[(lo, hi, kind, rgb)] -> tensors."""
        lo = np.array([b[0] for b in boxes], np.float32)
        hi = np.array([b[1] for b in boxes], np.float32)
        kind = np.array([b[2] for b in boxes], np.int64)
        rgb = np.array([b[3] for b in boxes], np.float32)
        return lo, hi, kind, rgb

    def _boxes(self, o, d, lo, hi):
        """Nearest box hit per ray: (t, index, axis) with t = inf for a miss."""
        dsafe = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        inv = 1.0 / dsafe
        t1 = (lo[None] - o[:, None]) * inv[:, None]
        t2 = (hi[None] - o[:, None]) * inv[:, None]
        tmn, tmx = torch.minimum(t1, t2), torch.maximum(t1, t2)
        tn, ax = tmn.max(-1)
        tf = tmx.min(-1).values
        hit = (tf >= tn) & (tf > 1e-4)
        t = torch.where(hit, tn.clamp_min(1e-4), torch.full_like(tn, float("inf")))
        tb, kb = t.min(1)
        axb = ax.gather(1, kb[:, None])[:, 0]
        return tb, kb, axb

    @staticmethod
    def _line(u, spacing, half_w, foot):
        """Footprint-filtered brightness of lines at multiples of `spacing` (coordinate u), each 2 half_w wide."""
        dist = torch.abs(torch.remainder(u / spacing + 0.5, 1.0) - 0.5) * spacing
        we = torch.maximum(torch.full_like(foot, half_w), 0.5 * foot)
        return (half_w / we) * torch.exp(-0.5 * (dist / we) ** 2)

    def _sky(self, d):
        z = d[:, 2:3]
        glow = torch.exp(-torch.abs(z) * 9.0)
        st = STYLES[self.style]
        return _dev_t(st["sky"], self.device) + glow * _dev_t(st["horizon"], self.device)

    def _shade_box(self, p, n_ax, kind, rgb, lo, hi, foot):
        """Emission of the box surfaces hit at p (M, 3)."""
        z = p[:, 2]
        out = torch.zeros_like(p)
        # the horizontal coordinate along the face: x on a y-face, y on an x-face (top faces use x + y)
        u = torch.where(n_ax == 0, p[:, 1], p[:, 0])
        # arena walls: dark blue panels, a grid and a bright top edge
        wall = kind == KIND_ARENA
        if wall.any():
            g = torch.maximum(self._line(u, GRID, 0.06, foot), self._line(z, 1.5, 0.05, foot))
            top = torch.clamp(1.0 - (ARENA_WALL_H - z) / 0.25, 0, 1)
            e = (0.018 + 0.30 * g[:, None]) * _dev_t(WALL_RGB, self.device) + top[:, None] * _dev_t([0.5, 0.8, 1.6], self.device)
            out = torch.where(wall[:, None], e, out)
        trail = kind == KIND_TRAIL
        if trail.any():
            st = STYLES[self.style]
            th = hi[:, 2]
            h = torch.clamp(z / th, 0, 1)
            body = st["trail_body"][0] + st["trail_body"][1] * h ** 2
            top = torch.clamp(1.0 - (th - z) / 0.10, 0, 1)
            if st["trail_base"]:
                top = torch.maximum(top, torch.clamp(1.0 - z / 0.08, 0, 1))
            hot = rgb * 0.7 + 0.3 * rgb.max(-1, keepdim=True).values   # = rgb * 0.7 + 0.3 at full colour
            e = rgb * body[:, None] + top[:, None] * hot * 1.1
            if st["stripes"]:
                # vertical bands along the wall, 1 m apart: texture that expands sideways as the wall nears
                along = torch.where(n_ax == 0, p[:, 1], torch.where(n_ax == 1, p[:, 0], p[:, 0] + p[:, 1]))
                band = self._line(along, st["stripes"], 0.06, foot)
                e = e + (st["stripe_gain"] * band)[:, None] * rgb
            out = torch.where(trail[:, None], e, out)
        cyc = kind == KIND_CYCLE
        if cyc.any():
            # distance from the hit point to the nearest edge of its face, in the face's two in-plane coordinates
            dlo, dhi = p - lo, hi - p
            dmin = torch.minimum(dlo, dhi)
            big = torch.full_like(dmin, 1e3)
            axes = torch.arange(3, device=p.device)[None]
            dmin = torch.where(axes == n_ax[:, None], big, dmin)
            edge = torch.clamp(1.0 - dmin.min(1).values / 0.09, 0, 1)
            e = 0.02 + rgb * (0.05 + 2.2 * edge[:, None])
            out = torch.where(cyc[:, None], e, out)
        return out

    def trace(self, o, d, boxes, pix, *, reflect=True, fog=90.0):
        """o (3,) or (M, 3), d (M, 3) unit directions (torch, on device); boxes from `pack` (numpy); pix: ray
        footprint in rad (float or (M,))."""
        dev = self.device
        d = torch.as_tensor(d, dtype=torch.float32, device=dev)
        M = d.shape[0]
        o = torch.as_tensor(np.asarray(o, np.float32), device=dev)
        o = o.expand(M, 3) if o.ndim == 1 else o
        pix = torch.as_tensor(pix, dtype=torch.float32, device=dev).expand(M)
        lo, hi, kind, rgb = (_dev_t(boxes[0], dev), _dev_t(boxes[1], dev), torch.as_tensor(boxes[2], device=dev),
                             _dev_t(boxes[3], dev))
        tb, kb, axb = self._boxes(o, d, lo, hi)
        # floor
        tf = torch.where(d[:, 2] < -1e-6, -o[:, 2] / torch.where(d[:, 2] < -1e-6, d[:, 2], -torch.ones_like(d[:, 2])),
                         torch.full_like(tb, float("inf")))
        floor = tf < tb
        col = self._sky(d).expand(M, 3).clone()
        hitb = torch.isfinite(tb) & ~floor
        if hitb.any():
            p = o[hitb] + d[hitb] * tb[hitb, None]
            k = kb[hitb]
            e = self._shade_box(p, axb[hitb], kind[k], rgb[k], lo[k], hi[k], pix[hitb] * tb[hitb])
            att = torch.exp(-tb[hitb] / (fog * 3))[:, None]
            col[hitb] = e * att + (1 - att) * col[hitb]
        if floor.any():
            t = tf[floor]
            p = o[floor] + d[floor] * t[:, None]
            dz = d[floor, 2].abs().clamp_min(0.02)
            foot = pix[floor] * t / dz.sqrt()                        # the footprint stretches at grazing angles
            g = torch.maximum(self._line(p[:, 0], GRID, GRID_LINE, foot), self._line(p[:, 1], GRID, GRID_LINE, foot))
            fade = torch.exp(-t / fog)[:, None]
            st = STYLES[self.style]
            base = _dev_t(st["floor"], dev) + g[:, None] * _dev_t(GRID_RGB, dev) * st["grid_gain"]
            c = base * fade + (1 - fade) * self._sky(d[floor])
            if reflect:
                # one glossy bounce off the black floor: the jet walls mirrored, Schlick-weighted
                rd = d[floor].clone()
                rd[:, 2] = -rd[:, 2]
                ro = p + rd * 1e-3
                tr, kr, axr = self._boxes(ro, rd, lo, hi)
                hitr = torch.isfinite(tr)
                if hitr.any():
                    fres = 0.10 + 0.55 * (1 - dz) ** 5
                    pr = ro[hitr] + rd[hitr] * tr[hitr, None]
                    kk = kr[hitr]
                    er = self._shade_box(pr, axr[hitr], kind[kk], rgb[kk], lo[kk], hi[kk], pix[floor][hitr] * (t[hitr] + tr[hitr]))
                    blur = torch.exp(-tr[hitr] / 25.0)[:, None]      # a rough mirror: far reflections fade
                    c[hitr] = c[hitr] + (fres[hitr, None] * blur * er) * fade[hitr]
            col[floor] = c
        return col


_RAY_GRIDS = {}


def camera_grid(width, height, fov_deg, device):
    """Unit pinhole directions in camera space (forward, left, up) as a (H*W, 3) device tensor, row-major from the
    top-left pixel, built once per (W, H, fov, device); and the per-ray footprint (rad)."""
    key = (int(width), int(height), round(float(fov_deg), 6), str(device))
    tan = math.tan(math.radians(fov_deg) / 2)
    if key not in _RAY_GRIDS:
        xs = (2 * (np.arange(width) + 0.5) / width - 1) * tan
        ys = (1 - 2 * (np.arange(height) + 0.5) / height) * tan * height / width
        X, Y = np.meshgrid(xs, ys)
        g = np.stack([np.ones_like(X), -X, Y], -1).reshape(-1, 3)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        _RAY_GRIDS.clear()                                     # one camera size at a time
        _RAY_GRIDS[key] = torch.as_tensor(g.astype(np.float32), device=device)
    return _RAY_GRIDS[key], 2 * tan / width


def camera_basis(forward, up):
    """Rows: forward, left, up of a camera looking along `forward` with `up` roughly up."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    left = np.cross(np.asarray(up, float), f)
    left /= np.linalg.norm(left)
    return np.stack([f, left, np.cross(f, left)])


def camera_rays(pos, forward, up, width, height, fov_deg, device):
    """Pinhole rays (H*W, 3), row-major from the top-left pixel, and the per-ray footprint (rad): the cached
    camera-space grid rotated into the world."""
    g, pix = camera_grid(width, height, fov_deg, device)
    R = torch.as_tensor(camera_basis(forward, up).astype(np.float32), device=g.device)
    return g @ R, pix


def render_camera(renderer, boxes, pos, forward, up, width, height, fov_deg, chunk=1 << 18):
    """(H, W, 3) linear RGB tensor."""
    d, pix = camera_rays(pos, forward, up, width, height, fov_deg, renderer.device)
    out = torch.empty_like(d)
    for i in range(0, d.shape[0], chunk):
        out[i:i + chunk] = renderer.trace(pos, d[i:i + chunk], boxes, pix)
    return out.reshape(height, width, 3)


def bloom_tonemap(img, strength=0.85, threshold=0.75):
    """(H, W, 3) linear RGB tensor -> (H, W, 3) uint8 sRGB numpy: a glow from a blurred pyramid of what is above
    `threshold`, then a filmic curve. Display only (the fly's radiance is never tone mapped)."""
    x = img.permute(2, 0, 1)[None]
    bright = torch.clamp(x - threshold, min=0)
    acc = torch.zeros_like(x)
    lvl = bright
    H, W = x.shape[-2:]
    for i, w in enumerate((0.9, 0.8, 0.7, 0.6, 0.5)):
        lvl = torch.nn.functional.avg_pool2d(lvl, 2, ceil_mode=True)
        blur = torch.nn.functional.avg_pool2d(lvl, 3, 1, 1, count_include_pad=False)
        acc = acc + w * torch.nn.functional.interpolate(blur, size=(H, W), mode="bilinear", align_corners=False)
    y = x + strength * acc
    y = 1.0 - torch.exp(-1.35 * y)                                   # soft shoulder
    y = torch.where(y <= 0.0031308, 12.92 * y, 1.055 * torch.clamp(y, min=1e-8) ** (1 / 2.4) - 0.055)
    return (torch.clamp(y, 0, 1)[0].permute(1, 2, 0) * 255 + 0.5).to(torch.uint8).cpu().numpy()


# ======================================================================================================== world state
@dataclass
class Cycle:
    name: str
    color: np.ndarray
    pos: np.ndarray
    heading: int
    alive: bool = True
    segments: list = field(default_factory=list)      # finished trail segments: (a, b)
    seg_start: np.ndarray = None                      # the live segment starts here
    turns: int = 0
    crash_t: float = None

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float).copy()
        if self.seg_start is None:
            self.seg_start = self.pos.copy()

    def turn(self, side):
        self.segments.append((self.seg_start.copy(), self.pos.copy()))
        self.seg_start = self.pos.copy()
        self.heading = turn_heading(self.heading, side)
        self.turns += 1

    def trail_boxes(self, include_live=True):
        segs = list(self.segments)
        if include_live and np.linalg.norm(self.pos - self.seg_start) > 1e-6:
            segs.append((self.seg_start, self.pos))
        return [segment_box(a, b) for a, b in segs]

    @property
    def nose(self):
        return self.pos + heading_vec(self.heading) * FRONT

    def eye(self):
        f = heading_vec(self.heading)
        return np.array([self.pos[0] + f[0] * EYE_FWD, self.pos[1] + f[1] * EYE_FWD, EYE_H])

    def basis(self):
        f, lft = heading_vec(self.heading), left_vec(self.heading)
        return (f[0], f[1], 0.0), (lft[0], lft[1], 0.0), (0.0, 0.0, 1.0)


def obstacles_for(me: Cycle, others):
    """Boxes `me`'s nose can hit: the arena, every other cycle's body and whole trail, and its own finished
    segments (its live segment is behind it)."""
    boxes = list(arena_boxes())
    for c in others:
        boxes.append(cycle_box(c.pos, c.heading))
        boxes += c.trail_boxes(include_live=True)
    boxes += me.trail_boxes(include_live=False)
    return boxes


def move_cycle(me: Cycle, others, dist):
    """Advance `me` by `dist` along its heading, stopping at the first obstacle its nose meets (the cycle stops
    TRAIL_HALF_W short of it). Returns the index kind of what it hit ('arena' / 'trail' / 'cycle' / 'own') or None."""
    f = heading_vec(me.heading)
    boxes = obstacles_for(me, others)
    n0 = me.nose
    s, k = sweep_hit(n0, n0 + f * dist, boxes, pad=CYCLE_W / 2 * 0.0)
    if s is None:
        me.pos = me.pos + f * dist
        return None
    me.pos = me.pos + f * dist * s
    me.alive = False
    n_arena = 4
    n_other = sum(1 + len(c.trail_boxes(True)) for c in others)
    if k < n_arena:
        return "arena"
    if k < n_arena + n_other:
        return "opponent"
    return "own trail"


def scene_boxes(cycles, *, hide=None):
    """[(lo, hi, kind, rgb)] for the renderer: arena, every trail, every cycle body except `hide` (the fly's own
    body is not in its own view)."""
    out = [(lo, hi, KIND_ARENA, WALL_RGB) for lo, hi in arena_boxes()]
    for c in cycles:
        for lo, hi in c.trail_boxes(True):
            out.append((lo, hi, KIND_TRAIL, c.color))
        if c is not hide:
            lo, hi = cycle_box(c.pos, c.heading)
            out.append((lo, hi, KIND_CYCLE, c.color))
    return out


# ======================================================================================================== brain readout
LOOM = ("LPLC2", "LC4")


def cell_index(fb, t, side=None):
    crit = {"type": t}
    if side is not None:
        crit["somaSide"] = side
    return np.asarray(fb.c.select(**crit))


class Readout:
    """Population means of the cells the decoder reads, straight from `fb.brain.rate` (declared)."""

    def __init__(self, fb):
        self.fb = fb
        self.idx = {}
        for t in ("DNp01",) + LOOM:
            for s in "LR":
                ix = cell_index(fb, t, s)
                self.idx[f"{t}_{s}"] = torch.as_tensor(ix, device=fb.device, dtype=torch.long)
        self.counts = {k: int(v.numel()) for k, v in self.idx.items()}

    def read(self):
        r = self.fb.brain.rate
        r = r[0] if r.ndim == 2 else r
        out = {k: (float(r[v].mean()) if v.numel() else 0.0) for k, v in self.idx.items()}
        out["gf"] = 0.5 * (out["DNp01_L"] + out["DNp01_R"])
        out["loom_L"] = out["LPLC2_L"] + out["LC4_L"]
        out["loom_R"] = out["LPLC2_R"] + out["LC4_R"]
        out["side"] = out["loom_L"] - out["loom_R"]
        return out


# ======================================================================================================== probe
def _probe_trials(h):
    # name, speed m/s, wall kind, wall y-range (lo, hi) or None, run seconds, wall height
    return [("wall_v10", 10.0, "trail", (-60, 60), 3.0, h),
            ("wall_v15", 15.0, "trail", (-60, 60), 3.0, h),
            ("wall_v20", 20.0, "trail", (-60, 60), 3.0, h),
            ("wall_v10_h35", 10.0, "trail", (-60, 60), 3.0, 3.5),
            ("wall_v15_h35", 15.0, "trail", (-60, 60), 3.0, 3.5),
            ("arena_v15", 15.0, "arena", None, 3.0, h),
            ("left_half_v12", 12.0, "trail", (-0.6, 60), 3.0, h),
            ("right_half_v12", 12.0, "trail", (-60, 0.6), 3.0, h),
            ("pass_left_v15", 15.0, "pass", (3.0, 3.0), 3.0, h),
            ("open_v15", 15.0, "open", None, 3.0, h)]


# 'probe2' reproduces the trial list behind out/games/tron/probe2_*.json (written when the jet walls were 2.2 m);
# 'game' is the same list with the game's own wall height (TRAIL_H = 3.5 m). The first probe (probe_*_s100.json, glow
# look, 8-25 m/s) came from an earlier trial list that is not in this file (captions/tron.md says so).
PROBE_SETS = {"probe2": _probe_trials(2.2), "game": _probe_trials(TRAIL_H)}
PROBE_TRIALS = PROBE_SETS["game"]


def probe(args):
    """Open-loop: the cycle runs straight at a wall (or past one) at a fixed speed; log the decoder's inputs per
    tick. Each trial: HOLD_S of the first frame held still, then the run; the eye stops 0.3 m before the wall."""
    fb = common.build_brain(args, **({"dataset": args.dataset} if args.dataset else {}))
    eyes = common.Eyes(fb)
    ro = Readout(fb)
    rend = Renderer(fb.device, args.style)
    dirs = eyes.world_dirs()                                  # body frame == world frame here: heading +x
    pix = math.radians(1.5)
    hold_s = 1.2
    res = {"dataset": fb.c.dataset, "seed": args.seed, "style": args.style, "probe_set": args.probe_set,
           "counts": ro.counts, "argv": sys.argv, "sources": common.source_hashes([__file__, common.__file__]),
           "device_name": torch.cuda.get_device_name(0) if str(fb.device).startswith("cuda") else "cpu",
           "trials": []}
    print("cells:", ro.counts, flush=True)
    t_all = time.time()
    for name, v, kind, yr, run_s, th in PROBE_SETS[args.probe_set]:
        D = v * run_s                                          # the wall's face is D ahead of the eye at the start
        eye0 = np.array([-D, 0.0, EYE_H])
        boxes = [(lo, hi, KIND_ARENA, WALL_RGB) for lo, hi in arena_boxes(half=200.0)]
        if kind == "trail":
            boxes.append((np.array([0.0, yr[0], 0.0]), np.array([0.2, yr[1], th]), KIND_TRAIL, ORANGE))
        elif kind == "arena":
            boxes.append((np.array([0.0, -200, 0.0]), np.array([1.0, 200, ARENA_WALL_H]), KIND_ARENA, WALL_RGB))
        elif kind == "pass":
            boxes.append((np.array([-D - 30, yr[0] - 0.1, 0.0]), np.array([30.0, yr[0] + 0.1, th]), KIND_TRAIL, ORANGE))
        packed = Renderer.pack(boxes)
        rows = []
        n_hold, n_run = int(round(hold_s / TICK_S)), int(round(run_s / TICK_S))
        for i in range(n_hold + n_run + 10):
            moving = i >= n_hold
            x = eye0[0] + (v * TICK_S * (i - n_hold) if moving else 0.0)
            if kind in ("trail", "arena"):
                x = min(x, -0.3)
            rgb = rend.trace((x, 0.0, EYE_H), dirs, packed, pix)
            rad = eyes.pool(common.rgb_to_radiance(rgb))
            fb.vision(rad)
            fb.step(common.TICK_MS)
            r = ro.read()
            ttc = (-x) / v if kind in ("trail", "arena") else None
            rows.append({"t": round((i - n_hold) * TICK_S, 3), "d": round(-x, 3), "ttc": ttc,
                         **{k: round(val, 2) for k, val in r.items()}})
        run = [r for r in rows if r["t"] >= 0]
        gf = np.array([r["gf"] for r in run])
        cross = [j for j in range(1, len(run)) if gf[j - 1] < GF_THRESHOLD_HZ <= gf[j]]
        first = run[cross[0]] if cross else None
        side_at = None
        if first is not None:
            j = cross[0]
            side_at = float(np.mean([run[q]["side"] for q in range(max(0, j - 10), j)]))
        hold = [r for r in rows if r["t"] < 0]
        summ = {"name": name, "speed": v, "kind": kind, "yr": yr, "height": th,
                "gf_hold_max": max(r["gf"] for r in hold), "gf_run_max": float(gf.max()),
                "t_gf_max": run[int(gf.argmax())]["t"], "d_gf_max": run[int(gf.argmax())]["d"],
                "crossings": [run[j]["t"] for j in cross],
                "first_cross_d": None if first is None else first["d"],
                "first_cross_ttc_ms": None if first is None or first["ttc"] is None else round(first["ttc"] * 1000, 1),
                "side_at_cross": side_at,
                "side_run_mean": float(np.mean([r["side"] for r in run])),
                "loomL_max": max(r["loom_L"] for r in run), "loomR_max": max(r["loom_R"] for r in run),
                "lplc2_max": max(max(r["LPLC2_L"], r["LPLC2_R"]) for r in run),
                "lc4_max": max(max(r["LC4_L"], r["LC4_R"]) for r in run)}
        print(json.dumps(summ), flush=True)
        res["trials"].append({"summary": summ, "rows": rows})
    res["wall_s"] = round(time.time() - t_all, 1)
    Path(args.probe).parent.mkdir(parents=True, exist_ok=True)
    Path(args.probe).write_text(json.dumps(res), encoding="utf-8")
    print("wrote", args.probe, flush=True)




# ======================================================================================================== the game
READY_S = 1.2                  # GAME: each round opens with the cycles standing still this long (trigger not live)
END_S = 1.8                    # GAME: after a crash the arena freezes this long (the wreck's walls fade), then a new round
ROUND_MAX_S = 30.0             # GAME: a round still running after this long is a draw
EYE_PIX = math.radians(1.5)    # angular footprint of each of the fly's 7 rays per column (grid-line filtering)
RANDOM_RATE_HZ = 0.38          # GAME baseline default: the MaleCNS fly's turn rate on dev seed 102 (4 turns in 10.4 s
                               # of racing, out/games/tron/dev3/dev102.json). Recordings pass --random-rate with the
                               # matching fly run's own summary.turn_rate_A_hz; the in-log draws always use that rate.
BASELINE_DRAWS = 50            # GAME: brain-free random-player draws per connected fly run (summary.baselines)
BOT_LOOK_S = 0.45              # GAME bot: turns when the free run ahead is shorter than this many seconds of travel
OPP_SEED_OFFSET = 1000         # a second MaleCNS brain runs on seed + this
# DECODER: each brain's resting L - R of (LPLC2 + LC4), subtracted before the side is read. MaleCNS: its open-floor
# and wall-pass runs in the probe, dark look, dev seed 100 (-2.10, -2.14 Hz). FAFB: its wall-pass run in the
# dark + light-bands look, seed 100 (-2.05 Hz); FAFB was not probed in the game's own dark look before this was set.
SIDE_OFFSET_HZ = {"malecns": -2.1, "fafb": -2.1}
# display: the chase camera sits behind, to the left and well above the cycle, off the line of its own jet wall, and
# is kept inside the arena (a start 6-8 m from a wall would otherwise put it behind that wall)
CAM_BACK, CAM_SIDE, CAM_UP = 10.0, 3.0, 8.0
CAM_CLAMP = ARENA_HALF - 1.0
CAM_TAU_S = 0.22               # display: the camera's yaw follows the cycle with this time constant (brain time)

# The round schedule (GAME): the same start positions and headings for every seed. (A = the MaleCNS fly's cycle.)
SCHEDULE = [
    ((-22.0, -5.0, 0), (22.0, 5.0, 2)),        # head to head, offset: they pass and ride on to the far walls
    ((-18.0, 18.0, 3), (18.0, -18.0, 1)),      # from opposite corners
    ((0.0, -24.0, 1), (-8.0, 24.0, 3)),
    ((18.0, -12.0, 2), (-18.0, 12.0, 0)),
    ((-24.0, 0.0, 0), (0.0, 24.0, 3)),         # crossing paths
    ((12.0, 23.0, 3), (-12.0, -23.0, 1)),
]

DESIGN = (1920, 1080)
TOP, FOOT_Y = 64, 1032
MAIN = (0, TOP, 1440, FOOT_Y - TOP)            # the chase camera (67 % of the canvas)
COL_X, COL_W = 1456, 448
SUB_PX = 34                    # banner subline size (legible at phone width)


def population_means(x) -> dict:
    """{label: (B, n) rate tensor} -> {label: mean Hz of row 0}, with one device -> host copy."""
    keys = list(x)
    v = torch.stack([x[k][0].float().mean() if x[k].numel() else torch.zeros((), device=x[k].device)
                     for k in keys]).cpu().numpy()
    return dict(zip(keys, (float(a) for a in v)))


def turn_inputs(m: dict):
    """(giant-fibre mean, loom side signal, loom L, loom R) from population means keyed 'DNp01_L' etc."""
    gf = 0.5 * (m["DNp01_L"] + m["DNp01_R"])
    L = m["LPLC2_L"] + m["LC4_L"]
    R = m["LPLC2_R"] + m["LC4_R"]
    return gf, L - R, L, R


def free_run(me: Cycle, others, heading, limit=250.0):
    """Metres the nose could travel along `heading` (from where it would sit on that heading) before meeting
    anything, with `me`'s live segment counted as finished when `heading` is a turn."""
    probe_c = Cycle(me.name, me.color, me.pos, heading, segments=list(me.segments), seg_start=me.seg_start)
    boxes = obstacles_for(probe_c, others)
    if heading != me.heading and np.linalg.norm(me.pos - me.seg_start) > 1e-6:
        boxes.append(segment_box(me.seg_start, me.pos))
    n0 = probe_c.nose
    s, _ = sweep_hit(n0, n0 + heading_vec(heading) * limit, boxes, pad=0.0)
    return limit if s is None else s * limit


class FlyPilot:
    """A cycle ridden by a brain: its eyes see the arena from the cycle, its decoder (read-only) turns it."""
    kind = "fly"

    def __init__(self, fb, label, connected=True):
        self.fb, self.label, self.connected = fb, label, bool(connected)
        self.eyes = common.Eyes(fb)
        self.law = TurnLaw(side_offset=SIDE_OFFSET_HZ[fb.c.dataset])
        reads = {f"{t}_{s}": {"type": t, "somaSide": s} for t in ("DNp01",) + LOOM for s in "LR"}
        self.dec = fb.attach(common.ReadDecoder(
            f"turn_{label}", reads, lambda dt, x: population_means(x),
            law=("turn 90 deg on each upward crossing of mean(DNp01 L, R) through 33 Hz, >= 0.30 s after the last turn; "
                 "side: s = (mean LPLC2 + mean LC4, left) - (same, right); s' = mean of s over the last 100 ms minus "
                 f"the resting offset {self.law.side_offset:+.1f} Hz; s' > 0 -> right, else left"),
            parameters=self.law.parameters()))
        self.counts = {k: int(len(cell_index(fb, *k.split("_")))) for k in reads}
        self.m = {k: 0.0 for k in reads}
        self.gf = self.side = self.loomL = self.loomR = 0.0
        self.rad = None
        self.would_turn = 0

    def step(self, game, me, t, live):
        f, lft, up = me.basis()
        d = self.eyes.world_dirs(f, lft, up)
        rgb = game.render.trace(me.eye(), d, game.packed_hide(me), EYE_PIX)
        self.rad = self.eyes.pool(common.rgb_to_radiance(rgb))
        self.fb.vision(self.rad)
        self.fb.step(common.TICK_MS)
        self.m = self.dec.value or self.m
        self.gf, self.side, self.loomL, self.loomR = turn_inputs(self.m)
        cmd = self.law.update(t, self.gf, self.side, live)
        if cmd is not None and not self.connected:
            self.would_turn += 1
            return None, cmd
        return cmd, cmd


class RandomPilot:
    kind = "random"

    def __init__(self, rate_hz, seed, draw=None):
        key = [0x7209, int(seed)] if draw is None else [0x7209, int(seed), 1 + int(draw)]
        self.rate_hz = float(rate_hz)
        self.gen = RandomTurns(rate_hz, np.random.default_rng(key))

    def step(self, game, me, t, live):
        c = self.gen.update(t, live)
        return c, c


class ReplayPilot:
    """GAME: repeats a recorded run's applied turns for one cycle at the same tick of the same round (the tick counted
    from that round's GO), open loop: it does not see anything. After the recorded turns of a round run out, the cycle
    rides straight."""
    kind = "replay"

    def __init__(self, schedule):
        self.schedule = {(int(r), int(k)): s for (r, k), s in dict(schedule).items()}

    def step(self, game, me, t, live):
        if not live:
            return None, None
        c = self.schedule.get((game.round + 1, game.race_tick(t)))
        return c, c


class NeverPilot:
    kind = "never"

    def step(self, game, me, t, live):
        return None, None


def bot_choice(me: Cycle, others, look_m):
    """GAME bot: None while the free run ahead is at least `look_m`; otherwise the side with more free run (L on a
    tie)."""
    if free_run(me, others, me.heading) >= look_m:
        return None
    fl = free_run(me, others, turn_heading(me.heading, "L"))
    fr = free_run(me, others, turn_heading(me.heading, "R"))
    return "L" if fl >= fr else "R"


class BotPilot:
    """GAME opponent: turns when the free run ahead is under BOT_LOOK_S of travel, toward the side with more room."""
    kind = "bot"

    def step(self, game, me, t, live):
        if not live:
            return None, None
        c = bot_choice(me, [o for o in game.cycles if o is not me], SPEED * BOT_LOOK_S)
        return c, c


class Banner:
    """A cut-on banner: a headline with its provenance chip as a tab (DECODER for a decoder turn, GAME otherwise) and
    an optional subline with its own chip."""

    def __init__(self, t, text, sub="", color=common.WHITE, hold=0.6, fade=0.8, size=58, chip="game", sub_chip="game"):
        self.t, self.text, self.sub, self.color, self.hold, self.fade, self.size = t, text, sub, color, hold, fade, size
        self.chip, self.sub_chip = chip, sub_chip

    def alpha(self, now):
        age = now - self.t
        if age < 0.08:
            return age / 0.08
        if age < self.hold:
            return 1.0
        return max(0.0, 1.0 - (age - self.hold) / self.fade)


def banner_text(pg, font, text, color):
    """Render a banner line; a '→' is drawn as an arrow shape (display fonts often lack the glyph)."""
    parts = text.split("→")
    imgs = [font.render(p.strip(), True, color) for p in parts]
    if len(imgs) == 1:
        return imgs[0]
    hgt = max(i.get_height() for i in imgs)
    aw = int(hgt * 1.1)
    out = pg.Surface((sum(i.get_width() for i in imgs) + aw * (len(imgs) - 1), hgt), pg.SRCALPHA)
    x = 0
    for k, img in enumerate(imgs):
        out.blit(img, (x, (hgt - img.get_height()) // 2))
        x += img.get_width()
        if k < len(imgs) - 1:
            cy, th = hgt // 2, max(3, hgt // 11)
            pg.draw.rect(out, color, (x + aw * 0.22, cy - th // 2, aw * 0.42, th))
            pg.draw.polygon(out, color, [(x + aw * 0.56, cy - hgt * 0.2), (x + aw * 0.82, cy), (x + aw * 0.56, cy + hgt * 0.2)])
            x += aw
    return out


def rgb8(c, k=1.0):
    return tuple(int(255 * min(1.0, float(v) * k)) for v in c)


class Tron(common.Game):
    title = "tron"
    subtitle = "light cycles on the escape pathway"

    def __init__(self, args, pilots=None):
        super().__init__(args)
        from flyverse.body import Flight
        if abs(Flight().gf_hz - GF_THRESHOLD_HZ) > 1e-9:
            raise SystemExit(f"Flight.gf_hz is {Flight().gf_hz}, the decoder assumes {GF_THRESHOLD_HZ}")
        self.fbs = {}
        if pilots is not None:                      # brain-free GAME replays (baselines): the pilots are given
            self.pa, self.pb = pilots
            self.player_kind, self.opp = self.pa.kind, self.pb.kind
        else:
            self.player_kind = "fly" if args.control else args.player
            # player A (cyan): the MaleCNS fly, or a baseline in its place
            if self.player_kind == "fly":
                fb = common.build_brain(args)
                self.fbs["malecns"] = fb
                self.pa = FlyPilot(fb, "A", connected=not args.control)
            elif self.player_kind == "random":
                self.pa = RandomPilot(args.random_rate, args.seed)
            else:
                self.pa = NeverPilot()
            # player B (orange)
            self.opp = args.opponent
            if self.opp in ("fafb", "malecns"):
                from flyverse import FlyBrain
                fb = FlyBrain(device=common.pick_device(args.device), dataset=self.opp,
                              seed=args.seed + (OPP_SEED_OFFSET if self.opp == "malecns" else 0))
                self.fbs["opponent_" + self.opp] = fb
                self.pb = FlyPilot(fb, "B")
            else:
                self.pb = BotPilot()
        dev = next(iter(self.fbs.values())).device if self.fbs else common.pick_device(args.device)
        self.render = Renderer(dev, args.style)
        self.names = {"A": {"fly": "MaleCNS", "random": "RANDOM", "never": "NEVER TURNS", "replay": "REPLAY"}[self.player_kind],
                      "B": {"fafb": "FAFB", "malecns": "MaleCNS #2", "bot": "BOT", "replay": "REPLAY",
                            "never": "NEVER TURNS"}[self.opp]}
        self.ticks = 0
        self.ride_time = {"A": 0.0, "B": 0.0}
        self.cam_t = 0.0
        self.round = -1
        self.phase = "ready"
        self.phase_t = 0.0
        self.round_t0 = 0.0
        self.score = {"A": 0, "B": 0, "draw": 0}
        self.cycles = []
        self.fade = {"A": 1.0, "B": 1.0}
        self.banners = []
        self.fx = []
        self.fx_rng = np.random.default_rng([0x7209, int(args.seed), 2])      # drawing only
        self.cam_yaw = None
        self.cam_scale = float(args.cam_scale) if args.cam_scale else (1.0 if args.record else 0.5)
        self.hist = {k: [] for k in ("t", "gfA", "gfB")}
        self.turn_log = {"A": [], "B": []}
        self.race_time = 0.0
        self.rounds = []
        self.gf_trace = {"A": [], "B": []}                     # (t, gf) per tick, for the run log's crossing count
        self._packed_cache = {}
        self._canvas = None
        self._declare()
        self._new_round(0.0)

    # -------------------------------------------------------------------------------------------- bookkeeping
    def _declare(self):
        L = self.log
        common.declare(L, "arena", "game", "a 60 x 60 m square arena (walls 6 m) with a 5 m floor grid; two light "
                       "cycles at a constant speed turn only in 90-degree steps; each leaves a jet wall (3.5 m tall, "
                       "0.2 m thick) along its path; a cycle whose nose meets an arena wall, a jet wall or the other "
                       "cycle is out; the last cycle riding wins the round; both out on the same tick is a draw",
                       speed_m_s=SPEED, arena_half_m=ARENA_HALF, trail_h_m=TRAIL_H, wall_h_m=ARENA_WALL_H,
                       ready_s=READY_S, end_s=END_S, round_max_s=ROUND_MAX_S)
        common.declare(L, "schedule", "game", "the rounds' start positions and headings, the same for every seed",
                       rounds=[[list(a), list(b)] for a, b in SCHEDULE])
        cols = {self.names[k]: p.eyes.n_col for k, p in (("A", self.pa), ("B", self.pb)) if isinstance(p, FlyPilot)}
        common.declare(L, "eye_mount", "game", "the fly's eyes ride its cycle 1.0 m above the floor at its nose (1.3 m "
                       "ahead of its centre, the point that collides), looking along the heading (x forward, y left, z up); its own cycle body is not "
                       "drawn in its view. Each brain's own ommatidial columns ("
                       + ", ".join(f"{k} {v:,}" for k, v in cols.items()) + ") x 7 rays traced through the same scene "
                       "as the camera: emissive edges, a glossy floor with a footprint-filtered grid (one mirror bounce), "
                       "UV = 0.5 B", eye_h_m=EYE_H, eye_fwd_m=EYE_FWD, ray_footprint_deg=math.degrees(EYE_PIX),
                       columns=cols)
        common.declare(L, "players", "game", f"cyan: {self.player_kind}"
                       + (" (decoder attached and read, its turns NOT applied: the cycle never turns)" if self.args.control else "")
                       + f"; orange: {self.opp}", player=self.player_kind, control=bool(self.args.control),
                       opponent=self.opp)
        if self.player_kind == "random":
            common.declare(L, "random_player", "game", "turns at Poisson times at a given rate (recordings: the matching "
                           "fly run's own turn rate), side 50/50, 0.30 s re-arm (numpy default_rng([0x7209, seed]))",
                           rate_hz=self.pa.rate_hz)
        if "replay" in (self.player_kind, self.opp):
            common.declare(L, "replay_player", "game", "repeats a recorded run's turns at the same tick of the same "
                           "round, open loop; rides straight once they run out")
        if self.player_kind == "never":
            common.declare(L, "never_player", "game", "never turns")
        if self.opp == "bot":
            common.declare(L, "bot_player", "game", "turns when the free run ahead is under BOT_LOOK_S of travel, to "
                           "the side with more free run", look_s=BOT_LOOK_S)
        if self.opp == "malecns":
            common.declare(L, "opponent_seed", "game", "the second MaleCNS brain runs on seed + 1000",
                           offset=OPP_SEED_OFFSET)
        common.declare(L, "hud_display", "game", "chase camera (behind, left of and above the cyan cycle, kept inside "
                       "the arena), minimap, bloom, sparks and banners are display only; the fly's radiance is never "
                       "tone mapped. A crashed cycle's walls fade out over the END phase, in the fly's view too",
                       cam_back_m=CAM_BACK, cam_side_m=CAM_SIDE, cam_up_m=CAM_UP, cam_clamp_m=CAM_CLAMP)

    def brains(self):
        self._summary()                                   # common.run saves the log right after this call
        if (self.fbs and self.player_kind == "fly" and not self.args.control
                and getattr(self.args, "baseline_draws", BASELINE_DRAWS) > 0):
            t0 = time.time()
            b = run_baselines(self.args.seed, self.ticks, self.turn_log, self.log.summary,
                              draws=getattr(self.args, "baseline_draws", BASELINE_DRAWS))
            self.log.summary["baselines"] = b
            print(f"tron: baselines ({time.time() - t0:.0f} s): replay matches fly {b['replay_matches_fly']}; "
                  f"never {b['never']['crashes_per_min_A']} crashes/min; rank {json.dumps(b['rank_of_fly'])}", flush=True)
        return dict(self.fbs)

    def banner(self, text, sub="", color=common.WHITE, **kw):
        self.banners = [b for b in self.banners if b.alpha(self.t_s) > 0][-2:]
        self.banners.append(Banner(self.t_s, text, sub, color, **kw))

    def race_tick(self, t):
        """Ticks since this round's GO (0 on the first live tick): the key a ReplayPilot's turns are stored under."""
        return int(round((t - self.round_t0) / TICK_S))

    def _new_round(self, t):
        self.round += 1
        a, b = SCHEDULE[self.round % len(SCHEDULE)]
        self.cycles = [Cycle("A", CYAN, a[:2], a[2]), Cycle("B", ORANGE, b[:2], b[2])]
        self.phase, self.phase_t = "ready", t
        self.round_t0 = t + READY_S
        self.fade = {"A": 1.0, "B": 1.0}
        self.log.event(t, "round_start", round=self.round + 1, a=list(a), b=list(b))
        self.banner(f"ROUND {self.round + 1}", "GET READY", common.WHITE, hold=READY_S - 0.3, size=64, sub_chip=None)

    def packed_hide(self, me):
        key = me.name
        if key not in self._packed_cache:
            self._packed_cache[key] = Renderer.pack(self.scene(hide=me))
        return self._packed_cache[key]

    def scene(self, hide=None):
        out = [(lo, hi, KIND_ARENA, WALL_RGB) for lo, hi in arena_boxes()]
        for c in self.cycles:
            k = self.fade[c.name]
            if k <= 0.01:
                continue
            for lo, hi in c.trail_boxes(True):
                out.append((lo, hi, KIND_TRAIL, c.color * k))
            if c is not hide and c.alive:
                lo, hi = cycle_box(c.pos, c.heading)
                out.append((lo, hi, KIND_CYCLE, c.color))
        return out

    # -------------------------------------------------------------------------------------------- tick
    def tick(self):
        t = self.t_s
        self._packed_cache = {}
        live = self.phase == "race"
        cmds = {}
        for key, pilot, cyc in (("A", self.pa, self.cycles[0]), ("B", self.pb, self.cycles[1])):
            cmd, raw = pilot.step(self, cyc, t, live and cyc.alive)
            cmds[key] = cmd
            if isinstance(pilot, FlyPilot) and pilot.law.last_cross is not None:
                lc = dict(pilot.law.last_cross)
                if lc["status"] == "turn" and cmd is None:
                    lc["status"] = "turn_not_applied"
                self.log.event(t, "gf_crossing", player=key, phase=self.phase, gf_hz=round(pilot.gf, 1),
                               side_signal_hz=round(pilot.law.side_signal(), 2), **lc)
            if raw is not None and live and cyc.alive:
                self._on_turn(key, pilot, cyc, t, raw, applied=cmd is not None)
        self.ticks += 1
        self.t_s = t = t + TICK_S
        for key, p in (("A", self.pa), ("B", self.pb)):
            g = getattr(p, "gf", 0.0)
            self.hist["gf" + key].append(g)
            self.gf_trace[key].append((round(t, 3), round(g, 2), self.phase))
        self.hist["t"].append(t)
        for k in self.hist:
            if len(self.hist[k]) > 600:
                del self.hist[k][:len(self.hist[k]) - 600]
        if self.phase == "ready":
            if t >= self.round_t0 - 1e-9:
                self.phase = "race"
                self.banner("GO", "", common.WHITE, hold=0.25, fade=0.4, size=72)
            return
        if self.phase == "end":
            for c in self.cycles:
                if not c.alive:
                    self.fade[c.name] = max(0.0, 1.0 - (t - self.phase_t) / (END_S * 0.8))
            if t >= self.phase_t + END_S - 1e-9:
                self._new_round(t)
            return
        # race: apply the turns, then move both cycles against the state at the start of the tick
        self.race_time += TICK_S
        for c in self.cycles:
            if c.alive:
                self.ride_time[c.name] += TICK_S
        for key, c in zip("AB", self.cycles):
            if cmds[key] is not None and c.alive:
                c.turn(cmds[key])
        snap = [Cycle(c.name, c.color, c.pos, c.heading, segments=list(c.segments), seg_start=c.seg_start.copy())
                for c in self.cycles]
        hits = {}
        for i, c in enumerate(self.cycles):
            if c.alive:
                hits[c.name] = move_cycle(c, [snap[1 - i]], SPEED * TICK_S)
        a, b = self.cycles
        if a.alive and b.alive:                                  # nose to nose: the bodies met this tick
            la, ha = cycle_box(a.pos, a.heading)
            lb, hb = cycle_box(b.pos, b.heading)
            if np.all(la[:2] < hb[:2]) and np.all(lb[:2] < ha[:2]):
                a.alive = b.alive = False
                hits["A"] = hits["B"] = "opponent"
        out = [c for c in self.cycles if not c.alive and c.crash_t is None]
        for c in out:
            c.crash_t = t
            self._on_crash(c, hits.get(c.name) or "opponent", t)
        if out:
            alive = [c for c in self.cycles if c.alive]
            self._end_round(t, alive[0].name if len(alive) == 1 else None)
        elif t - self.round_t0 >= ROUND_MAX_S - 1e-9:
            self._end_round(t, None, timeout=True)

    def _on_turn(self, key, pilot, cyc, t, side, applied):
        others = [c for c in self.cycles if c is not cyc]
        ahead = free_run(cyc, others, cyc.heading)
        after = free_run(cyc, others, turn_heading(cyc.heading, side))
        ev = {"player": key, "side": side, "applied": applied, "free_ahead_m": round(ahead, 2),
              "ttc_ahead_ms": round(ahead / SPEED * 1000, 1), "free_after_turn_m": round(after, 2),
              "round": self.round + 1, "race_tick": self.race_tick(t)}
        if isinstance(pilot, FlyPilot):
            ev.update(gf_hz=round(pilot.gf, 1), side_signal_hz=round(pilot.law.side_signal(), 2),
                      loom_L=round(pilot.loomL, 2), loom_R=round(pilot.loomR, 2))
        self.log.event(t, "turn" if applied else "turn_not_applied", **ev)
        self.turn_log[key].append(ev)
        col = rgb8(CYAN if key == "A" else ORANGE, 1.1)
        who = self.names[key]
        word = {"L": "LEFT", "R": "RIGHT"}[side]
        what = (f"{ahead:.1f} m" if ahead < 10 else f"{ahead:.0f} m") + " to the wall ahead"
        if isinstance(pilot, FlyPilot):
            if applied:
                self.banner(f"GIANT FIBRE {pilot.gf:.0f} Hz  →  TURN {word}", f"{who} · {what}", col, chip="decoder")
            else:
                self.banner(f"GIANT FIBRE {pilot.gf:.0f} Hz", f"{who} · decoder disconnected: no turn", col,
                            chip="decoder")
        else:
            self.banner(f"{who} TURNS {word}", what, col, size=48)

    def _on_crash(self, c, into, t):
        p = self.pa if c.name == "A" else self.pb
        brain = isinstance(p, FlyPilot)
        gmax = max(self.hist["gf" + c.name][-100:] or [0.0])
        self.log.event(t, "crash", player=c.name, into=into, round=self.round + 1,
                       survived_s=round(t - self.round_t0, 2), turns=c.turns,
                       gf_hz=round(p.gf, 1) if brain else None, gf_max_last_1s=round(gmax, 1) if brain else None,
                       pos=np.round(c.pos, 2).tolist())
        self.fx.append({"t": t, "pos": np.array([c.nose[0], c.nose[1], 0.7]), "color": c.color,
                        "dirs": self.fx_rng.normal(size=(56, 3)) * [1, 1, 0.6] + [0, 0, 0.7]})

    def _end_round(self, t, winner, timeout=False):
        self.phase, self.phase_t = "end", t
        self.score["draw" if winner is None else winner] += 1
        rec = {"round": self.round + 1, "winner": winner, "duration_s": round(t - self.round_t0, 2),
               "timeout": timeout, "turns": {c.name: c.turns for c in self.cycles},
               "crashed": [c.name for c in self.cycles if not c.alive]}
        self.rounds.append(rec)
        self.log.event(t, "round_end", **rec)
        if winner is None:
            self.banner("TIME" if timeout else "DRAW", f"{ROUND_MAX_S:.0f} s" if timeout else "both derezzed",
                        common.WHITE, hold=1.0, size=70, sub_chip=None)
        else:
            loser = "B" if winner == "A" else "A"
            self.banner(f"{self.names[loser]} DEREZZED", f"{self.names[winner]} takes round {self.round + 1}",
                        rgb8(CYAN if loser == "A" else ORANGE, 1.1), hold=1.0, size=70, sub_chip=None)
        self._summary()

    def _summary(self):
        s = {"rounds": len(self.rounds), "score": dict(self.score), "names": self.names,
             "race_time_s": round(self.race_time, 2), "round_records": self.rounds,
             "unfinished_round": None if self.phase == "end" else {
                 "round": self.round + 1, "phase": self.phase, "riding_s": round(max(0.0, self.t_s - self.round_t0), 2),
                 "turns": {c.name: c.turns for c in self.cycles}},
             "mean_round_s": round(float(np.mean([r["duration_s"] for r in self.rounds])), 2) if self.rounds else None}
        s["ticks"] = self.ticks
        for key, pilot in (("A", self.pa), ("B", self.pb)):
            tl = [e for e in self.turn_log[key] if e["applied"]]
            s[f"turns_{key}"] = len(tl)
            s[f"turn_rate_{key}_hz"] = round(len(tl) / max(self.race_time, 1e-9), 3)
            s[f"riding_s_{key}"] = round(self.ride_time[key], 2)
            s[f"turns_{key}_wall_within_1s"] = sum(1 for e in tl if e["ttc_ahead_ms"] <= 1000)
            s[f"turns_{key}_into_wall_within_1s"] = sum(1 for e in tl if e["free_after_turn_m"] <= SPEED)
            s[f"crashes_{key}"] = sum(1 for r in self.rounds if key in r["crashed"])
            s[f"crashes_per_min_{key}"] = round(60.0 * s[f"crashes_{key}"] / max(self.ride_time[key], 1e-9), 3)
            if isinstance(pilot, FlyPilot):
                cr = [e for e in self.log.events if e["kind"] == "gf_crossing" and e["player"] == key]
                s[f"gf_crossings_{key}_by_status"] = {st: sum(1 for e in cr if e["status"] == st) for st in
                                                      ("turn", "turn_not_applied", "blocked_rearm", "not_live")}
                s[f"decoder_{key}_not_applied"] = pilot.would_turn
                s[f"cells_{key}"] = pilot.counts
                g = self.gf_trace[key]
                s[f"gf_crossings_{key}_by_phase"] = {ph: sum(1 for i in range(1, len(g)) if g[i][2] == ph
                                                             and g[i - 1][1] < GF_THRESHOLD_HZ <= g[i][1])
                                                     for ph in ("ready", "race", "end")}
        self.log.summary = s

    # -------------------------------------------------------------------------------------------- camera
    def camera_pose(self):
        """Display only. Behind, left of and above the cyan cycle, yaw smoothed over brain time, the camera's floor
        position kept inside the arena; it looks at a point ahead of the cycle, nearer when the camera is nearer, so
        the cycle stays in the frame."""
        a = self.cycles[0]
        hv = heading_vec(a.heading)
        yaw = math.atan2(hv[1], hv[0])
        if self.cam_yaw is None or self.phase == "ready":
            self.cam_yaw, self.cam_t = yaw, self.t_s
        dt = max(0.0, self.t_s - self.cam_t)
        self.cam_t = self.t_s
        dyaw = (yaw - self.cam_yaw + math.pi) % (2 * math.pi) - math.pi
        self.cam_yaw += dyaw * (1 - math.exp(-dt / CAM_TAU_S))
        cy = self.cam_yaw
        if self.phase == "end" and not a.alive:
            cy += 0.35 * (self.t_s - self.phase_t)                         # a slow orbit round the wreck
        f = np.array([math.cos(cy), math.sin(cy), 0.0])
        lft = np.array([-f[1], f[0], 0.0])
        p = np.array([a.pos[0], a.pos[1], 0.0])
        cam = p - f * CAM_BACK + lft * CAM_SIDE + np.array([0, 0, CAM_UP])
        cam[:2] = np.clip(cam[:2], -CAM_CLAMP, CAM_CLAMP)
        dh = float(np.hypot(*(p - cam)[:2]))
        tgt = p + f * float(np.clip(0.8 * dh, 2.5, 9.0)) + np.array([0, 0, 0.6])
        return cam, tgt - cam

    @staticmethod
    def project(pt, cam, fwd, W, H, fov):
        f = np.asarray(fwd, float) / np.linalg.norm(fwd)
        lft = np.cross([0, 0, 1.0], f)
        lft /= np.linalg.norm(lft)
        u = np.cross(f, lft)
        v = np.asarray(pt, float) - cam
        z = v @ f
        if z <= 0.1:
            return None
        tan = math.tan(math.radians(fov) / 2)
        x = -(v @ lft) / z / tan
        y = (v @ u) / z / (tan * H / W)
        return (W * (x + 1) / 2, H * (1 - y) / 2, z)

    # -------------------------------------------------------------------------------------------- drawing
    def draw(self, surface):
        pg = self.hud.pg
        if self._canvas is None:
            self._canvas = pg.Surface(DESIGN)
        c = self._canvas
        self._draw(c)
        if surface.get_size() == DESIGN:
            surface.blit(c, (0, 0))
        else:
            surface.blit(pg.transform.smoothscale(c, surface.get_size()), (0, 0))

    def _draw(self, s):
        pg, hud = self.hud.pg, self.hud
        s.fill(common.BG)
        mx, my, mw, mh = MAIN
        W, H = int(mw * self.cam_scale), int(mh * self.cam_scale)
        cam, fwd = self.camera_pose()
        fov = 72.0
        img = render_camera(self.render, Renderer.pack(self.scene()), cam, fwd, (0, 0, 1), W, H, fov)
        ex = STYLES[self.args.style]["cam_exposure"]
        u8 = bloom_tonemap(img * ex, strength=1.1, threshold=0.75 * ex)
        surf = pg.image.frombuffer(np.ascontiguousarray(u8).tobytes(), (W, H), "RGB")
        if (W, H) != (mw, mh):
            surf = pg.transform.smoothscale(surf, (mw, mh))
        s.blit(surf, (mx, my))
        self._draw_fx(s, cam, fwd, fov)
        self._draw_minimap(s)
        self._draw_score(s)
        self._draw_banners(s)
        self._draw_column(s)
        hud.header(s, "TRON", "light cycles on the escape pathway", self._model_line())
        hud.footer(s, self.footer_laws())

    def footer_laws(self):
        """The footer's legend, which says exactly what the decoder does in this arm, and what the GAME does."""
        laws = []
        flies = [p for p in (self.pa, self.pb) if isinstance(p, FlyPilot)]
        if flies:
            offs = sorted({p.law.side_offset for p in flies})
            off = " / ".join(f"{o:+.1f}" for o in offs)
            laws.append("turn = mean DNp01 crosses 33 Hz upward, ≥ 0.3 s apart")
            laws.append(f"side = R if 100 ms mean L-R (LPLC2+LC4) > {off} Hz, else L")
        cyan = {"fly": "", "random": f"cyan: Poisson turns {getattr(self.pa, 'rate_hz', 0.0):.2f}/s",
                "never": "cyan: never turns", "replay": "cyan: replayed turns"}[self.player_kind]
        if self.args.control:
            cyan = "cyan: turns NOT applied"
        orange = {"fafb": "orange: FAFB, same law", "malecns": "orange: MaleCNS #2, same law",
                  "bot": "orange: GAME bot", "replay": "orange: replayed turns", "never": "orange: never turns"}[self.opp]
        laws.append(" · ".join(x for x in (cyan, orange, f"{SPEED:.0f} m/s" + ("" if cyan else ", 90° turns")) if x))
        return laws

    def _model_line(self):
        parts = []
        for k, fb in self.fbs.items():
            ds = {"malecns": "MaleCNS", "fafb": "FlyWire FAFB"}.get(fb.c.dataset, fb.c.dataset)
            parts.append(f"{ds} {fb.c.release} · {fb.c.n:,}")
        return "  vs  ".join(parts) + " neurons · raw" if parts else "GAME baselines"

    def _draw_fx(self, s, cam, fwd, fov):
        pg = self.hud.pg
        mx, my, mw, mh = MAIN
        keep = []
        for fx in self.fx:
            age = self.t_s - fx["t"]
            if age > 1.4:
                continue
            keep.append(fx)
            col = rgb8(fx["color"], 1.2)
            a = max(0.0, 1 - age / 1.4)
            layer = pg.Surface((mw, mh), pg.SRCALPHA)
            for d in fx["dirs"]:
                p0 = fx["pos"] + d * (age * 9.0)
                p1 = fx["pos"] + d * (age * 9.0 + 0.9)
                q0, q1 = self.project(p0, cam, fwd, mw, mh, fov), self.project(p1, cam, fwd, mw, mh, fov)
                if q0 and q1:
                    pg.draw.line(layer, col + (int(230 * a),), q0[:2], q1[:2], 3)
            q = self.project(fx["pos"], cam, fwd, mw, mh, fov)
            if q:
                r = int(10 + 420 * age / max(q[2], 1.0))
                pg.draw.circle(layer, (255, 255, 255, int(160 * a)), (int(q[0]), int(q[1])), r, 4)
            s.blit(layer, (mx, my))
        self.fx = keep

    def _draw_minimap(self, s):
        pg, hud = self.hud.pg, self.hud
        x0, y0, size = MAIN[0] + 20, MAIN[1] + 20, 270
        panel = pg.Surface((size, size + 4), pg.SRCALPHA)
        panel.fill((6, 10, 16, 190))
        s.blit(panel, (x0, y0))
        hud.text(s, "ARENA FROM ABOVE", (x0 + 10, y0 + 7), 13, common.MUTED, bold=True)
        hud.chip(s, "game", (x0 + size - 62, y0 + 4), 11)
        ax, ay, asz = x0 + 15, y0 + 32, size - 30

        def P(p):
            return (ax + (p[0] + ARENA_HALF) / (2 * ARENA_HALF) * asz, ay + (ARENA_HALF - p[1]) / (2 * ARENA_HALF) * asz)
        pg.draw.rect(s, (40, 90, 200), (ax, ay, asz, asz), 2)
        for c in self.cycles:
            k = self.fade[c.name]
            col = rgb8(c.color, k)
            pts = [P(a) for a, _ in c.segments] + [P(c.seg_start), P(c.pos)]
            if k > 0.02:
                pg.draw.lines(s, tuple(v // 3 for v in col), False, pts, 7)
                pg.draw.lines(s, col, False, pts, 3)
            q = [int(v) for v in P(c.pos)]
            if c.alive:
                pg.draw.circle(s, (255, 255, 255), q, 5)
                pg.draw.circle(s, col, q, 5, 2)
            else:
                pg.draw.line(s, (255, 255, 255), (q[0] - 6, q[1] - 6), (q[0] + 6, q[1] + 6), 3)
                pg.draw.line(s, (255, 255, 255), (q[0] - 6, q[1] + 6), (q[0] + 6, q[1] - 6), 3)

    def _draw_score(self, s):
        hud = self.hud
        cx = MAIN[0] + MAIN[2] // 2
        y = MAIN[1] + 16
        ca, cb = rgb8(CYAN), rgb8(ORANGE)
        r = hud.text(s, f"{self.score['A']}", (cx - 30, y), 56, ca, bold=True, display=True, anchor="midtop")
        hud.text(s, ":", (cx, y), 56, common.MUTED, bold=True, display=True, anchor="midtop")
        hud.text(s, f"{self.score['B']}", (cx + 30, y), 56, cb, bold=True, display=True, anchor="midtop")
        hud.text(s, self.names["A"], (cx - 64, y + 16), 30, ca, bold=True, display=True, anchor="topright")
        hud.text(s, self.names["B"], (cx + 64, y + 16), 30, cb, bold=True, display=True, anchor="topleft")
        if self.phase == "ready":
            t_round = 0.0
        elif self.phase == "end":
            t_round = self.phase_t - self.round_t0
        else:
            t_round = self.t_s - self.round_t0
        sub = f"ROUND {self.round + 1}  ·  {t_round:5.2f} s  ·  draws {self.score['draw']}"
        rr = hud.text(s, sub, (cx, r.bottom + 2), 22, common.TEXT, anchor="midtop")
        hud.chip(s, "game", (rr.right + 12, rr.y + 2), 11)

    def _chip_surface(self, kind, size):
        pg, hud = self.hud.pg, self.hud
        label = common.PROVENANCE[kind][0]
        f = hud.font(size, True)
        cs = pg.Surface((f.size(label)[0] + 12, f.get_height() + 6), pg.SRCALPHA)
        hud.chip(cs, kind, (0, 0), size)
        return cs

    def banner_surface(self, b):
        """A banner at full opacity: its headline's chip as a tab above the box's left edge, the subline's chip
        inline before it."""
        pg, hud = self.hud.pg, self.hud
        img = banner_text(pg, hud.font(b.size, True, True), b.text, b.color)
        sub = hud.font(SUB_PX, False, False).render(b.sub, True, common.TEXT) if b.sub else None
        tab = self._chip_surface(b.chip, 13) if b.chip else None
        sc = self._chip_surface(b.sub_chip, 12) if (sub is not None and b.sub_chip) else None
        sub_w = (sub.get_width() if sub else 0) + (sc.get_width() + 12 if sc else 0)
        w = max(img.get_width(), sub_w, tab.get_width() if tab else 0) + 60
        th = tab.get_height() if tab else 0
        h = img.get_height() + (sub.get_height() + 8 if sub else 0) + 24
        out = pg.Surface((w, h + th), pg.SRCALPHA)
        pg.draw.rect(out, (4, 8, 12, 178), (0, th, w, h))
        if tab:
            out.blit(tab, (0, 0))
        out.blit(img, img.get_rect(midtop=(w // 2, th + 10)))
        if sub:
            x = (w - sub_w) // 2
            y = th + 14 + img.get_height()
            if sc:
                out.blit(sc, (x, y + (sub.get_height() - sc.get_height()) // 2))
                x += sc.get_width() + 12
            out.blit(sub, (x, y))
        return out

    def _draw_banners(self, s):
        cx = MAIN[0] + MAIN[2] // 2
        y = MAIN[1] + int(MAIN[3] * 0.70)            # low enough that two stacked banners clear the minimap
        for b in reversed([b for b in self.banners if b.alpha(self.t_s) > 0][-2:]):
            box = self.banner_surface(b)
            box.set_alpha(int(255 * b.alpha(self.t_s)))
            s.blit(box, box.get_rect(midbottom=(cx, y)))
            y -= box.get_height() + 10

    def _draw_column(self, s):
        pg, hud = self.hud.pg, self.hud
        x, w = COL_X, COL_W
        y = TOP + 12
        ca, cb = rgb8(CYAN), rgb8(ORANGE)
        for key, pilot, col, h in (("A", self.pa, ca, 262), ("B", self.pb, cb, 206)):
            if isinstance(pilot, FlyPilot):
                r = hud.panel(s, (x, y, w, h), f"what the {self.names[key]} fly sees")
                hud.text(s, "radiance → fb.vision", (x + w - 12, y + 10), 13, common.DIM, anchor="topright")
                pg.draw.line(s, col, (x + 1, y + 32), (x + w - 2, y + 32), 2)
                if pilot.rad is not None:
                    hud.mosaic(s, r, pilot.eyes, common.eye_colors(pilot.rad, "human", 2.0))
            else:
                r = hud.panel(s, (x, y, w, h), f"{self.names[key]} player", "game")
                note = {"random": [f"Poisson turns, {getattr(pilot, 'rate_hz', 0.0):.2f} per s", "side 50 / 50"],
                        "never": ["rides straight", "never turns"],
                        "bot": [f"turns {BOT_LOOK_S:.2f} s before a wall", "toward the side with more room"],
                        "replay": ["replays recorded turns", "open loop"]}[pilot.kind]
                hud.text(s, "no brain on this cycle (GAME baseline):", (r.x + 4, r.y + 6), 17, common.MUTED)
                for i, line in enumerate(note):
                    hud.text(s, line, (r.x + 4, r.y + 44 + 40 * i), 30, col, bold=True, display=True)
            y += h + 10
        # giant fibre traces
        h = 206
        r = hud.panel(s, (x, y, w, h), "giant fibre DNp01", "connectome")
        tr = pg.Rect(r.x, r.y + 32, r.w, r.h - 32)
        vmax = 80.0
        n = 200
        ga, gb = self.hist["gfA"][-n:], self.hist["gfB"][-n:]
        fa, fo = isinstance(self.pa, FlyPilot), isinstance(self.pb, FlyPilot)
        if fa:
            hud.trace(s, tr, ga, 0, vmax, ca, threshold=GF_THRESHOLD_HZ)
        elif fo:
            hud.trace(s, tr, gb, 0, vmax, cb, threshold=GF_THRESHOLD_HZ)
        else:
            hud.trace(s, tr, [], 0, vmax, ca, threshold=GF_THRESHOLD_HZ)
        if fa and fo:                                  # both on top of the first fill, cyan last
            hud.trace(s, tr, gb, 0, vmax, cb, fill=False)
            hud.trace(s, tr, ga, 0, vmax, ca, fill=False)
        hud.text(s, "33 Hz", (tr.x + 4, tr.bottom - (GF_THRESHOLD_HZ / vmax) * tr.h - 18), 14, common.RED)
        xx = r.x
        for key, pilot, col in (("A", self.pa, ca), ("B", self.pb, cb)):
            if isinstance(pilot, FlyPilot):
                xx = hud.text(s, f"{self.names[key]} {pilot.gf:5.1f} Hz", (xx, r.y - 2), 24, col, bold=True).right + 20
        y += h + 10
        # loom sides and the decoded side, for each fly
        h = 150
        r = hud.panel(s, (x, y, w, h), "LPLC2 + LC4, L vs R", "connectome")
        row = 0
        for key, pilot, col in (("A", self.pa, ca), ("B", self.pb, cb)):
            if not isinstance(pilot, FlyPilot):
                continue
            yy = r.y + row * 52
            vmax = 30.0
            hud.bar(s, (r.x, yy, 124, 40), pilot.loomL, vmax, "L", col, fmt="{:.0f}")
            hud.bar(s, (r.x + 136, yy, 124, 40), pilot.loomR, vmax, "R", col, fmt="{:.0f}")
            # the decoder's side, exactly as it would be read now: s' = 100 ms mean of L - R minus the resting offset
            sd = pilot.law.side_signal()
            ar = hud.text(s, "→ R" if sd > 0 else "→ L", (r.x + 276, yy - 4), 26, common.TEAL, bold=True)
            hud.chip(s, "decoder", (ar.right + 10, yy + 2), 10)
            hud.text(s, f"s' {sd:+.2f} Hz", (r.x + 276, yy + 25), 15, common.TEAL)
            if not pilot.connected:
                hud.text(s, "DISCONNECTED", (x + w - 12, y + 10), 13, common.RED, bold=True, anchor="topright")
            row += 1
        if row == 0:
            hud.text(s, "no fly in this arm", (r.x, r.y + 8), 17, common.MUTED)
        y += h + 10
        # turns tally
        h = FOOT_Y - 12 - y
        r = hud.panel(s, (x, y, w, h), "turns this run")
        for i, (key, pilot, col) in enumerate((("A", self.pa, ca), ("B", self.pb, cb))):
            tl = [e for e in self.turn_log[key] if e["applied"]]
            name = {"NEVER TURNS": "NEVER"}.get(self.names[key], self.names[key])
            tx = hud.text(s, f"{name} {len(tl)}", (r.x + i * (r.w // 2), r.y), 24, col, bold=True)
            hud.chip(s, "decoder" if isinstance(pilot, FlyPilot) else "game", (tx.right + 8, tx.y + 5), 10)


# ======================================================================================================== baselines
def outcome_of(summary: dict) -> dict:
    """The cyan cycle's outcome in one run, from its summary. The ranking metric is `crashes_per_min_A`: cyan's
    crashes per minute of riding (lower is better); riding = race-phase time with the cyan cycle alive."""
    sc = summary["score"]
    return {"crashes_A": summary["crashes_A"], "riding_s_A": summary["riding_s_A"],
            "crashes_per_min_A": summary["crashes_per_min_A"], "rounds": summary["rounds"], "won_A": sc["A"],
            "won_B": sc["B"], "draws": sc["draw"], "turns_A": summary["turns_A"], "crashes_B": summary["crashes_B"]}


def play_game_only(seed, n_ticks, pa, pb) -> "Tron":
    """GAME: the same arena, schedule and tick timeline as a recorded run, with no brain: two given pilots."""
    args = argparse.Namespace(seed=int(seed), device="cpu", record=None, control=False, player=pa.kind,
                              opponent=pb.kind, random_rate=getattr(pa, "rate_hz", 0.0), cam_scale=0.2, style=STYLE,
                              seconds=n_ticks * TICK_S, baseline_draws=0)
    g = Tron(args, pilots=(pa, pb))
    for _ in range(int(n_ticks)):
        g.tick()
    g._summary()
    return g


def turn_schedule(turns) -> dict:
    """{(round, race_tick): side} of a run's applied turns (turn events or turn_log entries)."""
    return {(int(e["round"]), int(e["race_tick"])): e["side"] for e in turns if e.get("applied", True)}


def fly_rank(fly: dict, runs: list) -> dict:
    """Where the fly run's cyan outcome sits among the random draws."""
    x = fly["crashes_per_min_A"]
    v = sorted(r["crashes_per_min_A"] for r in runs)
    w = sorted(r["won_A"] for r in runs)
    return {"metric": "cyan crashes per minute of riding (lower is better)", "fly": x, "draws": len(runs),
            "draws_fewer_crashes_per_min": sum(a < x for a in v), "draws_equal": sum(a == x for a in v),
            "draws_more_crashes_per_min": sum(a > x for a in v),
            "crashes_per_min_min_median_max": [v[0], float(np.median(v)), v[-1]] if v else None,
            "fly_rounds_won": fly["won_A"], "draws_winning_more_rounds": sum(a > fly["won_A"] for a in w),
            "draws_winning_fewer_rounds": sum(a < fly["won_A"] for a in w),
            "rounds_won_min_median_max": [w[0], float(np.median(w)), w[-1]] if w else None}


def run_baselines(seed, n_ticks, turn_log, fly_summary, *, draws=BASELINE_DRAWS) -> dict:
    """GAME baselines for a connected fly run, brain-free, on its seed and tick timeline. The orange cycle repeats its
    recorded turns open loop (ReplayPilot) in every arm. Arms: the fly's own turns replayed (a determinism check: it
    must reproduce the fly run's rounds), a never-turning cyan cycle, and `draws` random cyan players at the fly run's
    own turn rate (turns per second of riding), numpy default_rng([0x7209, seed, 1 + draw])."""
    sa, sb = turn_schedule(turn_log["A"]), turn_schedule(turn_log["B"])
    fly_o = outcome_of(fly_summary)
    rep = play_game_only(seed, n_ticks, ReplayPilot(sa), ReplayPilot(sb))
    rate = fly_summary["turns_A"] / max(fly_summary["riding_s_A"], 1e-9)
    runs = [outcome_of(play_game_only(seed, n_ticks, RandomPilot(rate, seed, draw=d), ReplayPilot(sb)).log.summary)
            for d in range(int(draws))]
    return {"what": "GAME only, no brain: same seed, arena, schedule and tick timeline as the fly run; the orange "
                    "cycle repeats its recorded turns open loop; scored by the same code",
            "ticks": int(n_ticks), "fly": fly_o, "replay": outcome_of(rep.log.summary),
            "replay_matches_fly": (outcome_of(rep.log.summary) == fly_o
                                   and rep.log.summary["round_records"] == fly_summary["round_records"]),
            "never": outcome_of(play_game_only(seed, n_ticks, NeverPilot(), ReplayPilot(sb)).log.summary),
            "random": {"rate_hz": round(rate, 4), "rng": "numpy default_rng([0x7209, seed, 1 + draw])",
                       "runs": runs},
            "rank_of_fly": fly_rank(fly_o, runs)}


def baselines_from_log(path, draws=BASELINE_DRAWS) -> dict:
    """Recompute `summary.baselines` from a connected fly run's log (CPU, no brain)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    s = d["summary"]
    turns = {k: [e for e in d["events"] if e["kind"] == "turn" and e["player"] == k] for k in "AB"}
    return run_baselines(d["meta"]["seed"], s["ticks"], turns, s, draws=draws)


# ======================================================================================================== main
def main(argv=None):
    ap = common.standard_args(__doc__.splitlines()[0], seconds=40.0)
    ap.add_argument("--probe", metavar="JSON", default=None, help="open-loop wall-approach measurements, no video")
    ap.add_argument("--dataset", default=None, help="probe: FlyBrain dataset (default malecns)")
    ap.add_argument("--style", choices=sorted(STYLES), default=STYLE, help="scene look (GAME)")
    ap.add_argument("--player", choices=("fly", "random", "never"), default="fly",
                    help="the cyan cycle: the MaleCNS fly (default) or a GAME baseline in its place")
    ap.add_argument("--control", action="store_true", help="the fly's decoder attached and read, its turns not applied")
    ap.add_argument("--opponent", choices=("fafb", "malecns", "bot"), default="fafb", help="the orange cycle")
    ap.add_argument("--random-rate", type=float, default=RANDOM_RATE_HZ, help="random player's turns per second")
    ap.add_argument("--cam-scale", type=float, default=None, help="chase-camera render scale (default 1 recording)")
    ap.add_argument("--probe-set", choices=sorted(PROBE_SETS), default="game",
                    help="probe: 'game' (the game's 3.5 m jet walls) or 'probe2' (the 2.2 m set of probe2_*.json)")
    ap.add_argument("--baseline-draws", type=int, default=BASELINE_DRAWS,
                    help="connected fly runs: brain-free random draws written to summary.baselines (0: none)")
    ap.add_argument("--baselines-from", metavar="JSON", default=None,
                    help="recompute a fly run log's baselines on CPU (no brain) and write <log>_baselines.json")
    args = ap.parse_args(argv)
    if args.baselines_from:
        b = baselines_from_log(args.baselines_from, args.baseline_draws)
        out = Path(args.baselines_from).with_name(Path(args.baselines_from).stem + "_baselines.json")
        out.write_text(json.dumps(b, indent=1), encoding="utf-8")
        print(f"replay matches fly: {b['replay_matches_fly']}; rank {json.dumps(b['rank_of_fly'])}; wrote {out}")
        return b
    if args.probe:
        return probe(args)
    if args.control and args.player != "fly":
        raise SystemExit("--control applies to --player fly")
    game = Tron(args)
    common.run(game, args)


if __name__ == "__main__":
    main()
