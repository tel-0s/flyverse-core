"""swarm -- 64 flies, one GPU.

One `flyverse.BatchSim(64)`: 64 copies of the MaleCNS v1.0 brain (preset 'raw', 167,106 neurons each) stepped together
as one batched FlyBrain. Every fly has its own eyes (1,466 columns x 7 rays, the 64 scenes ray-traced as one batch)
and its own body (the shipped `Locomotion` / `Flight`, fenced to a tray on the room's table). A giant swatter (GAME)
comes down on the crowd in five rounds, faster each round. Nothing decides for a fly but its own giant fibre: the
shipped body model launches the escape hop when DNp01 >= 33 Hz (flyverse/body.py, `Flight.gf_hz`). **This game has
no decoders**; the only neural -> motion mappings are the body model's own readouts.

Each round starts a fresh crowd: all 64 brains and bodies are reset (`BatchSim.reset`) and re-laid on the grid, so
every swat meets flies at rest. That is a GAME choice, made because the model does not settle after an escape: each
landing drives the giant fibre again (the ground looms as the fly falls) and many flies hop again every 1.5-2 s, past
the body's 1 s landing refractory. The hops after each swat stay in the clip and are counted in the run log.

    python games/swarm.py --batch 16                        # interactive: click the tray to swat, 1-5 speed, R new crowd
    python games/swarm.py --seed 100 --record out/games/swarm/dev.mp4 --screenshot out/games/swarm/dev.png
    python games/swarm.py --seed 100 --control blind --record out/games/swarm/dev_blind.mp4
    python games/swarm.py --seed 101 --measure --log out/games/swarm/measure_seed101.json   # the rounds, no drawing

GAME (declared in the run log): the tray and its fence, the room light, the crowd's layout and headings, the rounds,
the swatter (shape, path, speeds), its aim (the point that covers the most live flies), and the outcome rule. Two
separate questions are scored. Did the fly's own giant fibre launch the hop in time (the connectome's answer)? And did
the hop carry it clear of the swatter (the answer of the body model's forward hop)?

    at risk          under the head's footprint when the swat begins or at contact, or met by the head
    JUMPED IN TIME   at risk, and its giant fibre launched the escape hop in a step that ended before contact
    SPLAT            on the tray under the footprint at contact
    CAUGHT           the head (or handle) met its path while it was in the air, swept tick by tick, during the
                     descent or the head's 0.2 s rest on the tray (the body model has no swatter surface, so a fly
                     coming down on the resting head is counted caught, not landed on it)
    CLEAR            at risk, neither splat nor caught: the head never touched it (hopped, walked or took off clear)
    STARTLED         not at risk, but hopped during the descent

The hop is the body model's escape launch (0.6 m/s at 45 deg, along the fly's heading); in the air the power MNs add
some thrust and lift, and the fly comes down within about half a second (the raw model has no sustained flight).
The flies do not see each other: each fly's scene holds the room, the tray and the swatter. `--control blind` (same
seed) leaves the swatter out of the flies' scenes; the main view still draws it, and the same rule scores the run.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import torch  # noqa: E402

# ---------------------------------------------------------------------------------------------------- the scene (GAME)
TRAY_HALF = 0.17            # a 34 x 34 cm white tray on the room's table (world.make_room)
TRAY_H = 0.004
RIM_W, RIM_H = 0.008, 0.012
FENCE_HALF = 0.158          # the body model's fence (BatchBody.fence): the flies walk inside the rim
HEAD_ALONG, HEAD_ACROSS, HEAD_THICK = 0.100, 0.085, 0.006   # swatter head semi-axes (m): 20 x 17 cm, 12 mm thick
HANDLE_HALF, HANDLE_R = 0.13, 0.0055                        # a 26 cm rod, flat, in the head's plane
APPROACH_DEG = 65.0         # the head comes in along a straight path 65 deg above horizontal, from the handle side
H0 = 0.45                   # height of the head above the tray when a swat begins (m)
DWELL_S = 0.20              # the head rests on the tray this long, then lifts back along its path
LIFT_MPS = 1.2
FLY_R = 0.0015              # a fly counts as under the swatter within this margin of its footprint, and is caught
                            # in the air when its path comes this close to the head's (or handle's) surface
SPEEDS = (0.5, 1.0, 1.5, 2.0, 3.0)
ROUNDS = ((0.5, "-x"), (1.0, "+y"), (1.5, "-x"), (2.0, "+y"), (3.0, "-x"))   # (swat speed m/s, handle side) per round
ROUND_S = 4.5               # rounds 2.. begin (fresh crowd) every 4.5 s of brain time
FIRST_SWAT_S = 2.0          # round 1's swat: the clip opens on the crowd walking
SETTLE_S = 1.5              # later rounds: walking time between the reset and the swat
CLIP_SECONDS = 24.0
HANDLE_DIRS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
PARK = (9.0, 9.0, 9.0)      # outside the 4 x 4 m room: no ray reaches it
FLY_DRAW_SCALE = 2.5        # flies are drawn 2.5x life size in the main view (a 2.5 mm fly is ~6 px there)
HISTORY_S = 3.0
CAM_DETAIL = 0.5            # texture-noise amplitude of the human camera's render (the flies' scenes keep the room's 1.4)
MOSAIC_EXPOSURE = 1.2       # display exposure of the fly's-eye mosaic (the lamp lights the tray directly)
CAM_PREROLL_S = 0.45        # presentation: the camera pulls back this long before a swat begins
CAM_BOX = (40, 24, 1304, 800)   # the swat framing keeps the tray and the head inside these main-view pixels (of
                                # 1344 x 967); the tray stays above the banners' strip
SHAKE_S, SHAKE_M = 0.28, 0.003  # impact shake: the camera itself jitters (up to ~3 mm), so the frame never shows gaps
DOT = "  ·  "               # segment separator; in the display font it is drawn as a filled circle (see _line_surface)

OUTCOME_COLOR = {"clear": C.SAGE, "caught": C.LILAC, "splat": C.RED, "startled": C.AMBER, "safe": C.DIM}
OUTCOME_WORD = {"clear": "got clear", "caught": "caught in the air", "splat": "splat on the tray"}


# ---------------------------------------------------------------------------------------------------- pure geometry
def head_radii(handle: str) -> tuple:
    """(rx, ry, rz) of the head (an axis-aligned flattened ellipsoid) for a handle along x or y."""
    return (HEAD_ALONG, HEAD_ACROSS, HEAD_THICK) if handle[1] == "x" else (HEAD_ACROSS, HEAD_ALONG, HEAD_THICK)


def handle_radii(handle: str) -> tuple:
    return (HANDLE_HALF, HANDLE_R, HANDLE_R) if handle[1] == "x" else (HANDLE_R, HANDLE_HALF, HANDLE_R)


def handle_offset(handle: str) -> np.ndarray:
    """Handle centre minus head centre: the rod starts 3 cm inside the head's rim and rides 13 mm above the tray when
    the head rests on it, clear of the rim and of the flies (only the head can splat a fly)."""
    d = np.array(HANDLE_DIRS[handle]) * (HEAD_ALONG + HANDLE_HALF - 0.03)
    return np.array([d[0], d[1], RIM_H + 0.001 + HANDLE_R - HEAD_THICK])


def in_ellipse(xy, centre, rx, ry, margin=0.0) -> np.ndarray:
    xy = np.atleast_2d(np.asarray(xy, float))
    return ((xy[:, 0] - centre[0]) / (rx + margin)) ** 2 + ((xy[:, 1] - centre[1]) / (ry + margin)) ** 2 <= 1.0


def footprint(xy, aim, handle, margin=FLY_R) -> np.ndarray:
    """True where a point on the tray lies under the swatter's head resting at `aim` (within `margin`)."""
    rx, ry, _ = head_radii(handle)
    return in_ellipse(xy, aim, rx, ry, margin)


def aim_limits(handle: str) -> tuple:
    """How far the head's centre may go from the tray's centre (x, y) so that the head lands inside the rim."""
    rx, ry, _ = head_radii(handle)
    inner = TRAY_HALF - RIM_W - 0.002
    return inner - rx, inner - ry


def rho(xy, aim, handle) -> np.ndarray:
    """Distance from the head's centre in units of the head: 0 at the centre, 1 on its rim."""
    rx, ry, _ = head_radii(handle)
    xy = np.atleast_2d(np.asarray(xy, float))
    return np.sqrt(((xy[:, 0] - aim[0]) / rx) ** 2 + ((xy[:, 1] - aim[1]) / ry) ** 2)


def bearing_deg(xy, heading, target) -> np.ndarray:
    """Direction of `target` in each fly's body frame, degrees: 0 ahead, +90 to the left, +-180 behind."""
    xy = np.atleast_2d(np.asarray(xy, float))
    a = np.degrees(np.arctan2(target[1] - xy[:, 1], target[0] - xy[:, 0]) - np.asarray(heading, float))
    return (a + 180.0) % 360.0 - 180.0


def aim_point(xy, eligible, handle, lim=None, step=0.01):
    """GAME targeting: the head centre, on a `step` grid within +-lim (x, y; default: inside the rim), that covers the
    most eligible flies (live and on the tray); ties go to the candidate nearest the tray's centre. Returns
    ((x, y), flies covered)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    pts = xy[np.asarray(eligible, bool)]
    if not len(pts):
        return (0.0, 0.0), 0
    lx, ly = aim_limits(handle) if lim is None else np.broadcast_to(np.asarray(lim, float), (2,))
    gx = np.arange(-np.floor(lx / step + 1e-9), np.floor(lx / step + 1e-9) + 1) * step
    gy = np.arange(-np.floor(ly / step + 1e-9), np.floor(ly / step + 1e-9) + 1) * step
    cx, cy = np.meshgrid(gx, gy, indexing="ij")
    cand = np.c_[cx.ravel(), cy.ravel()]
    rx, ry, _ = head_radii(handle)
    inside = (((pts[None, :, 0] - cand[:, None, 0]) / rx) ** 2 + ((pts[None, :, 1] - cand[:, None, 1]) / ry) ** 2) <= 1.0
    count = inside.sum(1)
    best = np.flatnonzero(count == count.max())
    j = best[np.argmin(np.hypot(cand[best, 0], cand[best, 1]))]
    return (float(cand[j, 0]), float(cand[j, 1])), int(count[j])


def classify(inside0, inside_c, splat, caught, launched) -> np.ndarray:
    """The GAME outcome rule, per fly (see the module docstring): 'splat', 'caught', 'clear', 'startled' or 'safe'.
    `splat` = on the tray under the footprint at contact; `caught` = the head met the fly's path in the air;
    `launched` = its giant fibre launched the escape hop during the descent. Whether an at-risk fly jumped in time is
    `launched`, scored separately from the outcome (`jumped_in_time`)."""
    inside0, inside_c, splat, caught, launched = (np.asarray(a, bool) for a in (inside0, inside_c, splat, caught, launched))
    risk = inside0 | inside_c | splat | caught
    out = np.full(len(risk), "safe", dtype=object)
    out[~risk & launched] = "startled"
    out[risk] = "clear"
    out[risk & caught] = "caught"
    out[splat] = "splat"
    return out


def jumped_in_time(outcome, launched) -> np.ndarray:
    """At risk (splat, caught or clear) and the giant fibre launched the hop before contact."""
    return np.isin(np.asarray(outcome, object), ("splat", "caught", "clear")) & np.asarray(launched, bool)


def path_meets_ellipsoid(p0, p1, c0, c1, radii) -> np.ndarray:
    """(N,) True where a point moving in a straight line p0 -> p1 comes within an axis-aligned ellipsoid (semi-axes
    `radii`) whose centre moves c0 -> c1 over the same interval. Exact for straight motion within the step: in the
    ellipsoid's frame the point moves on a segment, and the test is that segment's distance from the unit ball."""
    r = np.asarray(radii, float)
    a = (np.atleast_2d(np.asarray(p0, float)) - np.asarray(c0, float)) / r
    b = (np.atleast_2d(np.asarray(p1, float)) - np.asarray(c1, float)) / r
    d = b - a
    s = np.clip(-(a * d).sum(1) / np.maximum((d * d).sum(1), 1e-18), 0.0, 1.0)
    m = a + s[:, None] * d
    return (m * m).sum(1) <= 1.0


GF_STOPS = ((0.0, (22, 30, 34)), (10.0, (34, 86, 94)), (22.0, (107, 201, 208)), (33.0, (232, 179, 104)),
            (50.0, (240, 104, 96)), (80.0, (255, 240, 225)))


def gf_color(hz) -> np.ndarray:
    """Giant-fibre rate (Hz) -> RGB uint8, piecewise linear: dark -> teal below the 33 Hz threshold, amber at it,
    red -> white above."""
    hz = np.asarray(hz, float)
    xs = np.array([s[0] for s in GF_STOPS])
    cs = np.array([s[1] for s in GF_STOPS], float)
    out = np.stack([np.interp(hz, xs, cs[:, k]) for k in range(3)], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


MAIN_GF_TOP = 50.0          # in the main view the ramp stops at red: its white top would vanish on the white tray


def gf_color_main(hz) -> np.ndarray:
    """`gf_color` clamped at red (MAIN_GF_TOP Hz), for marks drawn over the white tray."""
    return gf_color(np.minimum(np.asarray(hz, float), MAIN_GF_TOP))


def segment_hits_ellipsoid(origin, points, centre, radii) -> np.ndarray:
    """(N,) True where the segment origin -> point passes through the ellipsoid before reaching the point."""
    o = np.asarray(origin, float)
    p = np.atleast_2d(np.asarray(points, float))
    c, r = np.asarray(centre, float), np.asarray(radii, float)
    oc, dd = (o - c) / r, (p - o) / r
    a = (dd * dd).sum(1)
    b = 2 * dd @ oc
    cc = oc @ oc - 1
    disc = b * b - 4 * a * cc
    sq = np.sqrt(np.maximum(disc, 0))
    t0, t1 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    return (disc > 0) & (((t0 > 0) & (t0 < 0.999)) | ((t1 > 0) & (t1 < 0.999)))


def round_plan(rounds=ROUNDS, round_s=ROUND_S, first=FIRST_SWAT_S, settle=SETTLE_S, start=0.0):
    """GAME schedule: [(round index, reset time or None, swat start time, speed, handle)]. The first round needs no
    reset (the run starts fresh) when it starts at t = 0."""
    out = []
    for k, (v, h) in enumerate(rounds):
        t_round = start + k * round_s
        reset = None if (k == 0 and start == 0.0) else t_round
        out.append((k, reset, t_round + (first if k == 0 and start == 0.0 else settle), float(v), h))
    return out


def crowd_layout(n, spacing, jitter, rng):
    """GAME: n flies on a jittered square grid around the tray's centre, with uniformly random headings (rad)."""
    k = int(np.ceil(np.sqrt(n)))
    g = (np.arange(k) - (k - 1) / 2) * spacing
    gx, gy = np.meshgrid(g, g, indexing="ij")
    xy = np.c_[gx.ravel(), gy.ravel()][:n] + rng.normal(0, jitter, (n, 2))
    return np.clip(xy, -FENCE_HALF, FENCE_HALF), rng.uniform(-np.pi, np.pi, n)


@dataclass
class Swat:
    """One swat (GAME): a straight approach at speed v from the handle side, a rest on the tray, a lift."""
    t0: float
    v: float
    handle: str
    index: int = 0
    aim: tuple = (0.0, 0.0)
    aimed_by: str = "targeting"
    done: bool = False
    rec: dict = field(default_factory=dict)

    @property
    def path_len(self) -> float:
        return H0 / np.sin(np.deg2rad(APPROACH_DEG))

    @property
    def tc(self) -> float:
        """Brain time at which the head reaches the tray (contact)."""
        return self.t0 + self.path_len / self.v

    @property
    def t_end(self) -> float:
        return self.tc + DWELL_S + self.path_len / LIFT_MPS

    def offset(self, s: float) -> np.ndarray:
        d = HANDLE_DIRS[self.handle]
        th = np.deg2rad(APPROACH_DEG)
        return np.array([d[0] * np.cos(th) * s, d[1] * np.cos(th) * s, np.sin(th) * s])

    def path_s(self, t: float):
        """Distance of the head from its resting point along the path at brain time t (None = parked)."""
        if t < self.t0 or t >= self.t_end:
            return None
        if t < self.tc:
            return self.path_len - self.v * (t - self.t0)
        if t < self.tc + DWELL_S:
            return 0.0
        return LIFT_MPS * (t - self.tc - DWELL_S)

    def head_centre(self, t: float, surface_z: float):
        s = self.path_s(t)
        if s is None:
            return None
        return np.array([self.aim[0], self.aim[1], surface_z + HEAD_THICK]) + self.offset(s)

    def height(self, t: float) -> float:
        s = self.path_s(t)
        return np.inf if s is None else s * np.sin(np.deg2rad(APPROACH_DEG))


# ---------------------------------------------------------------------------------------------------- camera
class Camera:
    """A pinhole camera for the main view. Pixel x grows to the scene's right (unlike room_demo's `render_camera`,
    whose image is mirrored), and `project` is its exact inverse."""

    def __init__(self, pos, target, width, height, vfov_deg):
        self.pos = np.asarray(pos, float)
        f = np.asarray(target, float) - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, [0.0, 0.0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.W, self.H = int(width), int(height)
        self.tv = np.tan(np.deg2rad(vfov_deg) / 2)
        self.th = self.tv * self.W / self.H

    def rays(self, device):
        dev = torch.device(device)
        x = ((torch.arange(self.W, device=dev, dtype=torch.float32) + 0.5) / self.W * 2 - 1) * self.th
        y = (1 - (torch.arange(self.H, device=dev, dtype=torch.float32) + 0.5) / self.H * 2) * self.tv
        f, r, u = (torch.tensor(v, dtype=torch.float32, device=dev) for v in (self.f, self.r, self.u))
        d = f + x[None, :, None] * r + y[:, None, None] * u
        d = (d / d.norm(dim=-1, keepdim=True)).reshape(-1, 3)
        o = torch.tensor(self.pos, dtype=torch.float32, device=dev).expand_as(d)
        return o, d

    def project(self, pts):
        """(N, 3) world points -> (N, 2) pixel coordinates and (N,) depth along the view axis."""
        v = np.atleast_2d(np.asarray(pts, float)) - self.pos
        z = v @ self.f
        zs = np.where(z > 1e-6, z, 1e-6)
        px = (1 + (v @ self.r) / zs / self.th) * self.W / 2
        py = (1 - (v @ self.u) / zs / self.tv) * self.H / 2
        return np.c_[px, py], z

    def unproject_to_plane(self, px, py, z_plane):
        x = ((px / self.W) * 2 - 1) * self.th
        y = (1 - (py / self.H) * 2) * self.tv
        d = self.f + x * self.r + y * self.u
        if abs(d[2]) < 1e-9:
            return None
        t = (z_plane - self.pos[2]) / d[2]
        return None if t <= 0 else self.pos + t * d


CAM_VFOV = 37.0


def orbit_pose(t_s, seconds):
    """Presentation: the slow orbit's (azimuth, elevation) in radians: azimuth -52 -> -36 deg over the clip, 43 deg up."""
    frac = float(np.clip(t_s / max(seconds, 1e-6), 0, 1))
    return np.deg2rad(-52 + 16 * frac), np.deg2rad(43.0)


def orbit_camera(t_s, seconds, width, height, centre):
    """Presentation: the orbit camera between swats, 0.67 m out, aimed 4.5 cm in front of the tray's centre so that the
    whole tray sits in the upper part of the frame."""
    az, el = orbit_pose(t_s, seconds)
    toward = np.array([np.cos(az), np.sin(az), 0.0])
    target = np.asarray(centre, float) + 0.045 * toward
    pos = target + 0.67 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return Camera(pos, target, width, height, CAM_VFOV)


def fit_camera(az, el, width, height, pts, box, vfov=CAM_VFOV):
    """A camera looking along the orbit direction (az, el), placed as close as it can be while every point in `pts`
    projects inside `box` = (x0, y0, x1, y1) pixels. Closed form: each box edge is a half-space on the camera's
    position; the position is (a r + b u + c f) in the view basis, c as far forward as both pairs of edges allow."""
    back = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    f = -back
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    tv = np.tan(np.deg2rad(vfov) / 2); th = tv * width / height
    x0, y0, x1, y1 = box
    xl, xh = (2 * x0 / width - 1) * th, (2 * x1 / width - 1) * th
    yh, yl = (1 - 2 * y0 / height) * tv, (1 - 2 * y1 / height) * tv
    P = np.atleast_2d(np.asarray(pts, float))
    c1, c2 = (P @ (xh * f - r)).min(), (P @ (r - xl * f)).min()
    c3, c4 = (P @ (yh * f - u)).min(), (P @ (u - yl * f)).min()
    c = min((c1 + c2) / (xh - xl), (c3 + c4) / (yh - yl))
    a = 0.5 * ((xh * c - c1) + (c2 + xl * c))
    b = 0.5 * ((yh * c - c3) + (c4 + yl * c))
    pos = a * r + b * u + c * f
    return Camera(pos, pos + f, width, height, vfov)


def swat_frame_points(sw, surface_z, s=None) -> np.ndarray:
    """What the swat framing must hold: the tray's corners, and the head's rim at distance `s` along its path (default:
    where the swat begins), for every aim the rule could pick (the corners of `aim_limits`) or the player's aim.
    Independent of the targeting's choice, so the camera can pull back before the aim is known."""
    a = TRAY_HALF
    tray = [(sx * a, sy * a, surface_z) for sx in (-1, 1) for sy in (-1, 1)]
    rx, ry, _ = head_radii(sw.handle)
    if sw.aimed_by == "targeting":
        lx, ly = aim_limits(sw.handle)
        aims = [(sx * lx, sy * ly) for sx in (-1, 1) for sy in (-1, 1)]
    else:
        aims = [sw.aim]
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    start = sw.offset(sw.path_len if s is None else s)
    rims = [np.c_[ax + start[0] + rx * np.cos(ang), ay + start[1] + ry * np.sin(ang),
                  np.full(len(ang), surface_z + HEAD_THICK + start[2])] for ax, ay in aims]
    return np.vstack([np.array(tray)] + rims)


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------------------------------- the game
CAM_PUSH_S = 1.0            # presentation: after a swat begins the camera may push back in over at most this long


class Swarm(C.Game):
    title = "swarm"
    subtitle = "64 flies, one GPU"

    def __init__(self, args):
        super().__init__(args)
        from flyverse import BatchSim, world
        self.world = world
        self.B = int(args.batch)
        self.dt = C.TICK_MS / 1000.0
        self.headless = args.record is not None or bool(getattr(args, "measure", False))
        self.seconds = float(args.seconds)
        self.view_scale = float(args.view_scale) if args.view_scale else (1.0 if self.headless else 0.5)
        self.control = args.control
        dev = C.pick_device(args.device)
        self.device = dev
        xy, headings = crowd_layout(self.B, args.spacing, args.jitter, np.random.default_rng([args.seed, 0]))
        cam_world, info = world.make_room(args.seed, "apple")
        self.top = float(info["table_top_z"])
        self.z_s = self.top + TRAY_H
        starts = np.c_[xy, np.full(self.B, self.z_s)]
        fast = dev.startswith("cuda") and not args.no_fast
        t_build = time.time()
        self.sim = self._build(BatchSim, starts, dev, fast)
        self.build_s = time.time() - t_build
        sim = self.sim
        self.fb = sim.fb
        sim.body.fence = (-FENCE_HALF, FENCE_HALF, -FENCE_HALF, FENCE_HALF, self.z_s)
        for fly, h in zip(sim.flies, headings):
            fly.heading = float(h)
        # the same additions in all 64 fly scenes and in the camera's: tray, rim, swatter head, handle
        boxes = self._tray_boxes()
        for w in sim.world.worlds:
            w.boxes += [world.Box(lo, hi, m) for lo, hi, m in boxes]
            w.spheres.append(world.Sphere(PARK, handle_radii("-x"), "black"))
        self.head_idx = sim.loom_idx                                 # BatchSim's loom sphere is the swatter head
        self.handle_idx = len(sim.world.worlds[0].spheres) - 1
        sim.world.move_sphere(self.head_idx, PARK, head_radii("-x"))
        self.light_pos = None
        if args.room_light == "globe":                               # GAME: the lamp's light 1 cm below its globe
            lamp = next(q for q in cam_world.spheres if q.material == "lamp")
            self.light_pos = (lamp.center[0], lamp.center[1], lamp.center[2] - lamp.radii[2] - 0.01)
            for w in sim.world.worlds:
                w.light_pos = self.light_pos
            cam_world.light_pos = self.light_pos
        sim.world.invalidate()
        cam_world.device = torch.device(dev)
        cam_world.detail = CAM_DETAIL
        cam_world.boxes += [world.Box(lo, hi, m) for lo, hi, m in boxes]
        cam_world.spheres.append(world.Sphere(PARK, head_radii("-x"), "black"))
        cam_world.spheres.append(world.Sphere(PARK, handle_radii("-x"), "black"))
        self.cam_world = cam_world
        self.cam_head_idx, self.cam_handle_idx = len(cam_world.spheres) - 2, len(cam_world.spheres) - 1
        self._placed = None
        # read-only access to the rates the HUD shows (LC4 / LPLC2 per fly); the GF is fb.motor().gf
        c = self.fb.c
        self.idx = {k: torch.as_tensor(c.select(type=k), device=self.fb.device) for k in ("LC4", "LPLC2")}
        self.gf_thr = float(sim.flights[0].gf_hz)
        self.eyes = C.Eyes(self.fb)
        # state
        self.speed = SPEEDS[1]
        self.handles = ["-x", "+y"]
        plan = round_plan() if (self.headless or args.auto) else []
        self.plan = plan
        self.queue = [Swat(t0, v, h, index=k) for k, _, t0, v, h in plan]
        self.resets = [(tr, k) for k, tr, _, _, _ in plan if tr is not None]
        self.n_rounds = len(plan)
        self.round, self.round_t0, self.reset_times = 0, 0.0, []
        self.round_base = 0                                          # round index the current schedule counts from
        self.want_reset = False
        self.fade_from = -9.0
        self.active: Swat | None = None
        self.swats: list[Swat] = []
        self.dead = np.zeros(self.B, bool)
        self.death_xy = np.zeros((self.B, 2))
        self.death_t = np.full(self.B, np.nan)
        self.death_kind = np.full(self.B, "", dtype=object)       # 'splat' or 'caught'
        self.last_outcome = np.full(self.B, "", dtype=object)
        self.n_hist = int(round(HISTORY_S / self.dt))
        self.gf_hist = deque(maxlen=self.n_hist)
        self.lc_hist = deque(maxlen=self.n_hist)
        self.gf_all = []                                             # (T, B) float32, for the run summary
        self.launch_all = []                                         # (t, fly) of every escape-route launch
        self.voluntary_all = []                                      # (t, fly) of every voluntary (wing-power) take-off
        self.gf = np.zeros(self.B)
        self.lc = np.zeros((self.B, 2))
        self.order = np.argsort(np.hypot(xy[:, 0], xy[:, 1]))
        self.focus = int(self.order[0])
        self.focus_frozen = None                                     # (colors, until_t, 'SPLAT' / 'CAUGHT') once it is hit
        self.sim_wall = deque(maxlen=200)
        self.frame_wall = deque(maxlen=50)
        self._t_frame = None
        self.fx = []                                                 # transient effects
        self.banners = []
        self.shake_until = -1.0
        self.fx_rng = np.random.default_rng(1234)                    # presentation only (shake, splat shapes)
        self._sprites = None
        self._splat_sprites = {}
        self.final_at = (self.queue[-1].t_end + 0.8) if self.queue else None
        self.final_until = None                                      # interactive: the card goes after a few seconds
        self._declare(args, fast, xy, headings)
        self.log.event(0.0, "round", round=1, reset=False, start_xy=np.round(xy, 4),
                       start_heading_deg=np.round(np.degrees(headings), 1))
        self.banner(f"{self.B} FLIES{DOT}{self.B} BRAINS{DOT}ONE GPU", C.WHITE,
                    f"{self.B} x {self.fb.c.n:,} neurons, stepped together", hold=1.2)

    # -------------------------------------------------------------------------------------------- setup
    def _build(self, BatchSim, starts, dev, fast):
        kw = dict(cuda_graphs=True, cuda_kernels=True, event_driven=True) if fast else {}
        for attempt in range(4):
            try:
                return BatchSim(self.B, seed=self.args.seed, fruit_set="apple", fence=True, start=starts, device=dev, **kw)
            except torch.OutOfMemoryError:
                if attempt == 3:
                    raise
                print("CUDA out of memory building the batch; waiting 60 s (the GPU is shared)", flush=True)
                torch.cuda.empty_cache()
                time.sleep(60)

    def _tray_boxes(self):
        z0, z1 = self.top, self.top + TRAY_H
        a, w, h = TRAY_HALF, RIM_W, RIM_H
        return [((-a, -a, z0), (a, a, z1), "plate"),
                ((-a, -a, z1), (a, -a + w, z1 + h), "table"), ((-a, a - w, z1), (a, a, z1 + h), "table"),
                ((-a, -a + w, z1), (-a + w, a - w, z1 + h), "table"), ((a - w, -a + w, z1), (a, a - w, z1 + h), "table")]

    def _declare(self, args, fast, xy, headings):
        log = self.log
        log.meta.update(batch=self.B, control=self.control, build_s=round(self.build_s, 1),
                        backend=dict(cuda_graphs=fast, cuda_kernels=fast, event_driven=fast))
        log.meta["shipped_readouts"] = [
            {"name": "escape takeoff", "law": "body model (flyverse/body.py, Flight via BatchBody): the fly's own giant "
             "fibre DNp01 >= gf_hz launches the escape hop (0.6 m/s at 45 deg along its heading; in the air the "
             "power MNs add thrust and lift), no new escape within 1 s of landing",
             "gf_hz": self.gf_thr, "note": "the shipped readout, not a game decoder"},
            {"name": "walking", "law": "body model Locomotion.readout: speed from the forward DNs and leg MNs, yaw from "
             "DNa02 L-R (the raw model walks straight)", "note": "the shipped readout, not a game decoder"}]
        log.meta["shipped_readouts"].append(
            {"name": "voluntary take-off", "law": "body model: power MNs >= Flight.takeoff_power_hz (50 Hz) held "
             "Flight.takeoff_hold_s (0.3 s) launches a 0.2 m/s take-off; logged as 'takeoff_voluntary' events",
             "note": "the shipped readout, not a game decoder"})
        log.meta["hud_reads"] = "HUD only, controls nothing: per-fly mean rate of LC4 (126 cells) and LPLC2 (185)"
        log.meta["presentation"] = ("main view: the same geometry and light ray-traced for a human camera with texture "
                                    f"noise {CAM_DETAIL} (the flies' scenes: the room's 1.4), flies drawn as sprites at "
                                    f"{FLY_DRAW_SCALE}x life size; the camera orbits at 43 deg elevation, pulls back "
                                    f"{CAM_PREROLL_S} s before each swat so that the whole head is in frame, then pushes "
                                    f"back in (at most over {CAM_PUSH_S} s) as the head comes down; the impact shake moves "
                                    f"the camera by up to {SHAKE_M * 1000:.0f} mm; the fly's-eye mosaic is the radiance "
                                    f"handed to fb.vision for the focus fly, shown with exposure {MOSAIC_EXPOSURE}; main "
                                    f"view rendered at {self.view_scale:g}x and scaled to the frame")
        C.declare(log, "room air", "game", "the shipped BatchSim room defaults, unchanged: 0.3 m/s wind blowing from +x "
                  "to -x with a 20 deg, 8 s meander, and the apple's odour plume (the apple sits outside the tray, so no "
                  "fly tastes it); every fly smells and feels it through fb.smell / fb.wind; the same in the control",
                  wind_speed_mps=0.3, wind_towards_deg=180.0, meander_deg=20.0)
        C.declare(log, "tray", "game", "a 34 x 34 cm white tray with a 12 mm dark rim on the room's table; the body "
                  "model's fence keeps the flies inside it (at the fence a fly turns towards the centre at 90 deg/s)",
                  tray_half_m=TRAY_HALF, fence_half_m=FENCE_HALF, rim_h_m=RIM_H)
        if self.light_pos is not None:
            C.declare(log, "room light", "game", "the room's point light moved 1 cm below the lamp's globe; in "
                      "world.make_room it sits at the globe's centre, so every shadow ray hits the globe and the room is "
                      "lit by ambient light alone (no shadows); here the lamp lights the tray and the swatter casts a "
                      "shadow", light_pos_m=tuple(round(v, 3) for v in self.light_pos))
        C.declare(log, "crowd", "game", "each round the flies start on a jittered square grid around the tray's "
                  "centre with uniformly random headings (numpy seed = [--seed, round]); they do not see each other",
                  spacing_m=args.spacing, jitter_m=args.jitter)
        C.declare(log, "rounds", "game", "one swat per round; every round after the first begins with BatchSim.reset "
                  "of all rows (brains: FlyBrain.reset per row, keeping the shared clock; bodies: back to the start "
                  "grid), so every swat meets a crowd at rest; splat flies return in the next round",
                  rounds=[dict(round=k + 1, reset_s=tr, swat_s=t0, speed_mps=v, handle=h) for k, tr, t0, v, h in self.plan]
                  if self.plan else "player (R: new crowd)", round_s=ROUND_S, settle_s=SETTLE_S)
        C.declare(log, "swatter", "game", "a black flattened ellipsoid head with a flat rod handle; it comes in "
                  "along a straight path from the handle side at the swat's speed, rests on the tray, lifts back",
                  head_semi_axes_m=(HEAD_ALONG, HEAD_ACROSS, HEAD_THICK), handle_half_m=HANDLE_HALF,
                  approach_deg=APPROACH_DEG, start_height_m=H0, dwell_s=DWELL_S, lift_mps=LIFT_MPS)
        C.declare(log, "aim", "game", "each swat aims at the head position (1 cm grid) that covers the most live flies "
                  "on the tray at the moment it begins; ties go nearest the tray's centre", grid_m=0.01)
        C.declare(log, "outcome rule", "game", "two questions, scored separately. JUMPED IN TIME = at risk and the "
                  "fly's giant fibre launched the escape hop in a step that ended before contact. The outcome: SPLAT = "
                  "on the tray under the head's footprint (+1.5 mm) at contact (scored on the last state before "
                  "contact); CAUGHT = the head or handle (+1.5 mm) met the fly's path while it was in the air "
                  "(airborne at either end of a step), swept step by step with both moving in straight lines, "
                  "during the descent or the head's rest on the tray (the body model has no "
                  "swatter surface, so a fly coming down on the resting head counts as caught, not as landed on it); "
                  "CLEAR = at risk and neither (the head never touched it); at risk = under the footprint when the "
                  "swat begins or at contact, or caught; STARTLED = not at risk, hopped in the descent; LATE = an "
                  "escape hop launched in the contact step or during the rest; a splat or caught fly leaves the "
                  "round", margin_m=FLY_R, rest_s=DWELL_S)
        if self.control == "blind":
            C.declare(log, "blind control", "game", "the swatter is left out of all 64 flies' scenes (no head, no "
                      "handle, no shadow); the main view still draws it and the same rule scores the run")

    # -------------------------------------------------------------------------------------------- world update
    def brains(self):
        return {"swarm": self.fb}

    def _positions(self):
        f = self.sim.flies
        return (np.array([[q.x, q.y, q.z] for q in f]), np.array([q.heading for q in f]),
                np.array([q.airborne for q in f], bool))

    def _place_swatter(self, t):
        sw = self.active
        centre = None if sw is None else sw.head_centre(t, self.z_s)
        if centre is None:
            key = None
            head = handle = PARK
            hr, hdr = head_radii("-x"), handle_radii("-x")
        else:
            key = (tuple(np.round(centre, 6)), sw.handle)
            head = tuple(centre)
            handle = tuple(centre + handle_offset(sw.handle))
            hr, hdr = head_radii(sw.handle), handle_radii(sw.handle)
        if key == self._placed:
            return
        self._placed = key
        if self.control != "blind":
            self.sim.world.move_sphere(self.head_idx, head, hr)
            self.sim.world.move_sphere(self.handle_idx, handle, hdr)
        self.cam_world.move_sphere(self.cam_head_idx, head, hr)
        self.cam_world.move_sphere(self.cam_handle_idx, handle, hdr)

    def _new_round(self, k: int, t: float):
        """GAME: a fresh crowd. Every row's brain and body is reset (BatchSim.reset keeps the shared neural clock) and
        the flies are re-laid on the start grid with new headings."""
        xy, headings = crowd_layout(self.B, self.args.spacing, self.args.jitter,
                                    np.random.default_rng([self.args.seed, k]))
        sim = self.sim
        sim.starts[:] = np.c_[xy, np.full(self.B, self.z_s)]
        sim.reset(rows=np.arange(self.B))
        for fly, h in zip(sim.flies, headings):
            fly.heading = float(h)
        self._placed = None                                          # reset parked the loom sphere (our head)
        self.dead[:] = False
        self.death_t[:] = np.nan
        self.death_kind[:] = ""
        self.last_outcome[:] = ""
        self.round, self.round_t0 = k, t
        self.reset_times.append(t)
        self.order = np.argsort(np.hypot(xy[:, 0], xy[:, 1]))
        self.focus, self.focus_frozen = int(self.order[0]), None
        self.fade_from = t
        self.fx = []
        self.log.event(t, "round", round=k + 1, reset=True, start_xy=np.round(xy, 4),
                       start_heading_deg=np.round(np.degrees(headings), 1))
        self.banners = []
        nxt = self.queue[0] if self.queue else None
        speed = f"{DOT}{nxt.v:.1f} m/s" if nxt is not None else ""
        self.banner(f"{self._round_label(k)}{speed}{DOT}{self.B} FRESH FLIES", C.SAGE,
                    "brains and bodies reset (GAME): the swat meets a crowd at rest", hold=0.7, size=50)

    def _round_label(self, k=None) -> str:
        k = self.round if k is None else k
        if self.n_rounds and k >= self.round_base:                   # (before a new schedule's first reset: free play)
            return f"ROUND {k - self.round_base + 1} / {self.n_rounds}"
        return f"ROUND {k + 1}"

    def _start_swat(self, sw: Swat, t: float):
        pos, head, air = self._positions()
        xy = pos[:, :2]
        alive = ~self.dead
        sw.t0 = t
        if sw.aimed_by == "targeting":
            sw.aim, _ = aim_point(xy, alive & ~air, sw.handle)
        inside0 = footprint(xy, sw.aim, sw.handle) & alive
        n = self.B
        sw.rec = dict(inside0=inside0, rho0=rho(xy, sw.aim, sw.handle), bearing0=bearing_deg(xy, head, sw.aim),
                      launch_t=np.full(n, np.nan), vol_t=np.full(n, np.nan), late_t=np.full(n, np.nan),
                      caught_t=np.full(n, np.nan), caught_phase=np.full(n, "", dtype=object),
                      caught_z=np.full(n, np.nan), gf_peak=np.zeros(n), alive0=alive.copy(),
                      round=self.round, round_t0=self.round_t0, scored=False)
        self.active = sw
        self.swats.append(sw)
        at_risk = np.flatnonzero(inside0)
        if len(at_risk):
            self.focus = int(at_risk[np.argmin(sw.rec["rho0"][at_risk])])
        elif alive.any():
            live = np.flatnonzero(alive)
            self.focus = int(live[np.argmin(sw.rec["rho0"][live])])
        order = np.argsort(np.where(alive, sw.rec["rho0"], 1e9))
        self.order = order
        self.log.event(t, "swat", swat=sw.index + 1, round=self.round + 1, speed_mps=sw.v, handle=sw.handle,
                       aim_m=np.round(sw.aim, 4),
                       aimed_by=sw.aimed_by, contact_s=round(sw.tc, 4), under_at_start=int(inside0.sum()),
                       alive=int(alive.sum()), focus_fly=self.focus)

    def _kill(self, i: int, xy, t: float, kind: str):
        self.dead[i] = True
        self.death_xy[i] = xy
        self.death_t[i] = t
        self.death_kind[i] = kind
        if i == self.focus:
            self.focus_frozen = (self._mosaic_colors(i), t + 1.4, kind.upper())

    def _contact(self, sw: Swat, t: float):
        """The head reaches the tray within this step: a fly on the tray under its footprint in the last state before
        contact (at most 10 ms earlier) is splat."""
        pos, head, air = self._positions()
        xy = pos[:, :2]
        r = sw.rec
        alive = r["alive0"] & ~self.dead                             # flies caught in the air earlier are out
        inside_c = footprint(xy, sw.aim, sw.handle) & alive
        splat = inside_c & ~air
        r.update(inside_c=inside_c, splat=splat, air_c=air & alive, z_c=pos[:, 2] - self.z_s)
        sw.done = True
        for i in np.flatnonzero(splat):
            i = int(i)
            self._kill(i, xy[i], sw.tc, "splat")
            self.log.event(sw.tc, "splat", swat=sw.index + 1, fly=i, xy_m=np.round(xy[i], 4),
                           rho0=round(float(r["rho0"][i]), 3), bearing0_deg=round(float(r["bearing0"][i]), 1),
                           gf_peak_before_impact_hz=round(float(r["gf_peak"][i]), 1),
                           hopped_before_contact=bool(not np.isnan(r["launch_t"][i])))
        self.fx.append(dict(kind="impact", t=sw.tc, aim=sw.aim, handle=sw.handle))
        self.shake_until = sw.tc + SHAKE_S

    def _sweep(self, sw: Swat, t: float, t_now: float, pa, pb, airborne):
        """GAME, the mid-air test for one step: a live fly in the air at either end of the step (`airborne`) whose
        straight path pa -> pb meets the head or the handle (each moving in a straight line over the step; radii +
        FLY_R) is caught. A fly walking on the tray is scored by the footprint at contact instead (SPLAT). The test
        stops when the head's rest on the tray ends."""
        r = sw.rec
        cand = r["alive0"] & ~self.dead & np.asarray(airborne, bool)
        if not cand.any():
            return
        te = min(t_now, sw.tc + DWELL_S)
        pe = pa + (te - t) / (t_now - t) * (pb - pa)
        c0, c1 = sw.head_centre(t, self.z_s), sw.head_centre(te, self.z_s)
        off = handle_offset(sw.handle)
        hit = path_meets_ellipsoid(pa, pe, c0, c1, np.asarray(head_radii(sw.handle)) + FLY_R)
        hit |= path_meets_ellipsoid(pa, pe, c0 + off, c1 + off, np.asarray(handle_radii(sw.handle)) + FLY_R)
        phase = "descent" if t < sw.tc else "rest"
        for i in np.flatnonzero(hit & cand):
            i = int(i)
            z = float(pe[i, 2] - self.z_s) * 1000
            r["caught_t"][i], r["caught_phase"][i], r["caught_z"][i] = t_now, phase, z
            lt = r["launch_t"][i]
            self.log.event(t_now, "caught", swat=sw.index + 1, fly=i, phase=phase, height_mm=round(z, 1),
                           xy_m=np.round(pe[i, :2], 4), gf_peak_before_impact_hz=round(float(r["gf_peak"][i]), 1),
                           hop_lead_ms=None if np.isnan(lt) else round(float(sw.tc - lt) * 1000, 1))
            self.fx.append(dict(kind="caught", t=t_now, pos=pe[i].copy()))
            self._kill(i, pe[i, :2], t_now, "caught")

    def _score(self, sw: Swat):
        """The head's rest is over: classify every fly that was live when the swat began (see `classify`)."""
        r = sw.rec
        r["scored"] = True
        caught = ~np.isnan(r["caught_t"])
        launched = ~np.isnan(r["launch_t"])
        out = classify(r["inside0"], r["inside_c"], r["splat"], caught, launched)
        out[~r["alive0"]] = "gone"
        jumped = jumped_in_time(out, launched)
        risk = np.isin(out, ("splat", "caught", "clear"))
        clear = out == "clear"
        vol = ~np.isnan(r["vol_t"])
        how = np.full(self.B, "", dtype=object)
        how[clear & launched] = "hop"
        how[clear & ~launched & vol] = "voluntary take-off"
        how[clear & ~launched & ~vol & r["air_c"]] = "airborne before the swat"
        how[clear & (how == "")] = "walked"
        lead = (sw.tc - r["launch_t"]) * 1000
        rows = []
        for i in np.flatnonzero(risk | (out == "startled")):
            rows.append(dict(fly=int(i), outcome=out[i], jumped_in_time=bool(jumped[i]), clear_by=how[i] or None,
                             rho0=round(float(r["rho0"][i]), 3), bearing0_deg=round(float(r["bearing0"][i]), 1),
                             hop_lead_ms=None if np.isnan(lead[i]) else round(float(lead[i]), 1),
                             gf_peak_before_impact_hz=round(float(r["gf_peak"][i]), 1),
                             airborne_at_contact=bool(r["air_c"][i]),
                             height_at_contact_mm=round(float(r["z_c"][i]) * 1000, 1),
                             caught_phase=r["caught_phase"][i] or None,
                             caught_height_mm=None if np.isnan(r["caught_z"][i]) else round(float(r["caught_z"][i]), 1)))
        counts = dict(at_risk=int(risk.sum()), jumped_in_time=int(jumped.sum()), clear=int(clear.sum()),
                      caught=int(np.sum(out == "caught")), splat=int(np.sum(out == "splat")),
                      startled=int(np.sum(out == "startled")),
                      jumped_and_clear=int(np.sum(jumped & clear)), jumped_and_caught=int(np.sum(jumped & (out == "caught"))),
                      jumped_and_splat=int(np.sum(jumped & (out == "splat"))),
                      caught_descending=int(np.sum((out == "caught") & (r["caught_phase"] == "descent"))),
                      caught_on_resting_head=int(np.sum((out == "caught") & (r["caught_phase"] == "rest"))),
                      clear_by_hop=int(np.sum(how == "hop")), clear_walked=int(np.sum(how == "walked")),
                      clear_by_voluntary_takeoff=int(np.sum(how == "voluntary take-off")),
                      clear_airborne_before=int(np.sum(how == "airborne before the swat")),
                      late_hops=int(np.sum(~np.isnan(r["late_t"]) & r["alive0"])),
                      voluntary_takeoffs_in_descent=int(np.sum(vol & r["alive0"])))
        r.update(outcome=out, jumped=jumped, counts=counts, flies=rows, clear_how=how)
        for i in range(self.B):
            if out[i] != "gone":
                self.last_outcome[i] = out[i]
        jl = lead[jumped]
        self.log.event(sw.tc + DWELL_S, "result", swat=sw.index + 1, speed_mps=sw.v, contact_s=round(sw.tc, 4), **counts,
                       hop_lead_ms_median=None if not len(jl) else round(float(np.median(jl)), 1),
                       hop_lead_ms_min=None if not len(jl) else round(float(np.min(jl)), 1), flies=rows)
        c = counts
        self.banner(f"{c['jumped_in_time']} JUMPED IN TIME{DOT}{c['clear']} GOT CLEAR", C.WHITE,
                    f"{c['at_risk']} under it at {sw.v:.1f} m/s{DOT}{c['caught']} caught in the air{DOT}"
                    f"{c['splat']} splat on the tray", hold=1.6, size=60, kind="result")
        self._summarize()

    def tick(self):
        t = self.t_s
        if self.active is not None and t >= self.active.t_end - 1e-9:
            self.active = None
        if self.active is None and self.resets and t >= self.resets[0][0] - 1e-9:
            self._new_round(self.resets.pop(0)[1], t)
        elif self.active is None and self.want_reset:
            self.want_reset = False
            self._new_round(self.round + 1, t)
        if self.active is None and self.queue and t >= self.queue[0].t0 - 1e-9:
            self._start_swat(self.queue.pop(0), t)
        sw = self.active
        self._place_swatter(t)
        if sw is not None and not sw.done and t < sw.tc <= t + self.dt + 1e-9:
            self._contact(sw, t)
        window = sw is not None and not sw.rec["scored"] and t < sw.tc + DWELL_S - 1e-9
        pos_a, _, air_a = self._positions() if window else (None, None, None)
        tw = time.perf_counter()
        motor = self.sim.step()                                      # eyes, senses, fb.step(10 ms), bodies: all 64
        self.gf = np.asarray(motor.gf, float).reshape(-1).copy()
        r = self.fb.brain.rate
        self.lc = torch.stack([r[:, self.idx["LC4"]].mean(1), r[:, self.idx["LPLC2"]].mean(1)], 1).float().cpu().numpy()
        self.sim_wall.append(time.perf_counter() - tw)
        self.t_s = t_now = t + self.dt
        self.gf_hist.append(self.gf)
        self.lc_hist.append(self.lc)
        self.gf_all.append(np.where(self.dead, np.nan, self.gf).astype(np.float32))
        esc = self.sim.body.launched_escape & ~self.dead             # a splat or caught fly has left the game
        vol = self.sim.body.launched_voluntary & ~self.dead
        phase = None
        if sw is not None and not sw.rec["scored"]:
            rec = sw.rec
            if not sw.done:                                          # the step ended before contact: the descent
                phase = "descent"
                rec["gf_peak"] = np.maximum(rec["gf_peak"], self.gf)
                rec["launch_t"][esc & np.isnan(rec["launch_t"])] = t_now
                rec["vol_t"][vol & np.isnan(rec["vol_t"])] = t_now
            else:                                                    # the contact step or the head's rest: late
                phase = "late"
                rec["late_t"][esc & np.isnan(rec["late_t"])] = t_now
        pos_b, _, air_b = self._positions()
        for i in np.flatnonzero(esc):
            i = int(i)
            self.launch_all.append((round(t_now, 3), i))
            self.log.event(t_now, "launch", fly=i, gf_hz=round(float(self.gf[i]), 1),
                           swat=(sw.index + 1) if phase else None, phase=phase,
                           before_contact_ms=round((sw.tc - t_now) * 1000, 1) if phase else None)
            self.fx.append(dict(kind="launch", t=t_now, fly=i, pos=pos_b[i].copy()))
            if i == self.focus and phase == "descent":
                self.banner(f"GIANT FIBRE {self.gf[i]:.0f} Hz  ->  JUMP", C.AMBER,
                            f"fly {i}{DOT}{int(round((sw.tc - t_now) * 1000))} ms before impact", hold=0.8, size=52)
        for i in np.flatnonzero(vol):
            i = int(i)
            self.voluntary_all.append((round(t_now, 3), i))
            self.log.event(t_now, "takeoff_voluntary", fly=i, swat=(sw.index + 1) if phase else None, phase=phase)
        if window:
            self._sweep(sw, t, t_now, pos_a, pos_b, air_a | air_b)
        if sw is not None and sw.done and not sw.rec["scored"] and t_now >= sw.tc + DWELL_S - 1e-9:
            self._score(sw)
        if len(self.gf_all) % 100 == 0:
            self._summarize()

    # -------------------------------------------------------------------------------------------- summary
    def _round_end(self, s: Swat) -> float:
        later = [r for r in self.reset_times if r > s.t0]
        return min(later) if later else self.t_s

    def _summarize(self):
        done = [s for s in self.swats if s.rec.get("scored")]
        per = []
        pooled = {"rho": [], "bearing": [], "outcome": [], "jumped": []}
        for s in done:
            r = s.rec
            out, jumped = r["outcome"], r["jumped"]
            risk = np.isin(out, ("splat", "caught", "clear"))
            jl = ((s.tc - r["launch_t"]) * 1000)[jumped]
            end = self._round_end(s)
            after = [(t, i) for t, i in self.launch_all if s.tc < t <= end]
            before = [(t, i) for t, i in self.launch_all if r["round_t0"] + 0.5 <= t < s.t0]
            nr = int(risk.sum())
            zc = r["z_c"][jumped] * 1000
            per.append(dict(
                swat=s.index + 1, round=r["round"] + 1, speed_mps=s.v, aim_m=np.round(s.aim, 3), **r["counts"],
                jumped_in_time_fraction=None if not nr else round(float(jumped.sum()) / nr, 3),
                clear_fraction=None if not nr else round(float(np.sum(out == "clear")) / nr, 3),
                hop_lead_ms_median=None if not len(jl) else round(float(np.median(jl)), 1),
                hop_lead_ms_min=None if not len(jl) else round(float(np.min(jl)), 1),
                gf_peak_before_impact_hz_median_at_risk=None if not nr else round(float(np.median(r["gf_peak"][risk])), 1),
                descent_s=round(float(s.tc - s.t0), 3),
                jumped_height_at_contact_mm_median=None if not len(zc) else round(float(np.median(zc)), 1),
                hops_after_contact=len(after), hopping_flies_after_contact=len({i for _, i in after}),
                after_window_s=round(float(end - s.tc), 2), hops_before_swat=len(before)))
            pooled["rho"] += list(r["rho0"][risk]); pooled["bearing"] += list(r["bearing0"][risk])
            pooled["outcome"] += list(out[risk]); pooled["jumped"] += list(jumped[risk])
        rho_a, bear = np.array(pooled["rho"]), np.abs(np.array(pooled["bearing"]))
        outc, jmp = np.array(pooled["outcome"], dtype=object), np.array(pooled["jumped"], bool)

        def tally(mask):
            return dict(n=int(mask.sum()), jumped_in_time=int(np.sum(jmp[mask])),
                        **{k: int(np.sum(outc[mask] == k)) for k in ("clear", "caught", "splat")})
        by_rho, by_bearing = {}, {}
        if len(outc):
            for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 9.0)):
                by_rho[f"{lo}-{hi}"] = tally((rho_a >= lo) & (rho_a < hi))
            bins = ((0, 60, "facing its landing point (|b|<60)"), (60, 120, "side-on"), (120, 181, "facing away (|b|>=120)"))
            for lo, hi, name in bins:
                by_bearing[name] = tally((bear >= lo) & (bear < hi))
            rim = rho_a >= 0.8
            for (lo, hi, _), name in zip(bins, ("rim facing", "rim side-on", "rim facing away")):
                by_bearing[name] = tally(rim & (bear >= lo) & (bear < hi))
        gf = np.array(self.gf_all) if self.gf_all else np.zeros((0, self.B))
        quiet = np.zeros(len(gf), bool)                              # before each swat: 0.5 s after the round began -> t0
        for s in self.swats:
            a, b = int(round((s.rec["round_t0"] + 0.5) / self.dt)), int(round(s.t0 / self.dt))
            quiet[a:b] = True
        qgf = gf[quiet]
        qgf = qgf[np.isfinite(qgf)]
        keys = ("at_risk", "jumped_in_time", "clear", "caught", "splat", "startled", "jumped_and_clear",
                "jumped_and_caught", "jumped_and_splat", "caught_descending", "caught_on_resting_head", "clear_by_hop",
                "clear_walked", "clear_by_voluntary_takeoff", "clear_airborne_before", "late_hops",
                "voluntary_takeoffs_in_descent")
        tot = {k: int(sum(p[k] for p in per)) for k in keys} if per else {}
        wall = np.mean(self.sim_wall) / self.dt if self.sim_wall else None
        self.log.summary = dict(
            rounds=per, totals=tot, flies=self.B, by_rho=by_rho, by_bearing=by_bearing, control=self.control,
            gf_before_swats=dict(ticks=int(quiet.sum()), p99_hz=None if not qgf.size else round(float(np.percentile(qgf, 99)), 1),
                                 max_hz=None if not qgf.size else round(float(qgf.max()), 1)),
            launches_total=len(self.launch_all), voluntary_takeoffs_total=len(self.voluntary_all),
            wall_s_per_brain_s_sim=None if wall is None else round(float(wall), 2),
            wall_s_per_brain_s_frame=None if not self.frame_wall else
            round(float(np.mean(self.frame_wall) * self.args.fps), 2))

    def finished(self):
        if self.headless and self.t_s >= self.seconds - 1e-9:        # the last frame: bring the run log's summary up to date
            self._summarize()
        return False

    # -------------------------------------------------------------------------------------------- interaction
    def _clear_final(self):
        self.final_at = self.final_until = None

    def handle(self, event, canvas_pos=None):
        import pygame as pg
        if event.type == pg.KEYDOWN:
            if event.unicode and event.unicode in "12345":
                self.speed = SPEEDS[int(event.unicode) - 1]
            elif event.key == pg.K_r:
                self.want_reset = True
                self._clear_final()
                if not self.queue:                                   # free play again: rounds count without a total
                    self.plan, self.n_rounds = [], 0
            elif event.key == pg.K_a and not self.queue:
                self.want_reset = False                              # the schedule's first round resets anyway
                self._clear_final()
                plan = round_plan(start=self.t_s + 0.5)
                self.plan, self.n_rounds = plan, len(plan)
                self.round_base = self.round + 1
                self.queue = [Swat(t0, v, h, index=len(self.swats) + k) for k, _, t0, v, h in plan]
                self.resets = [(tr, self.round_base + k) for k, tr, _, _, _ in plan]
                self.final_at = self.queue[-1].t_end + 0.8
                self.final_until = self.final_at + 6.0
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and canvas_pos is not None and self.active is None:
            if self.queue and self.queue[0].aimed_by == "player":
                return                                               # one pending swat at a time
            rect = self._main_rect(self._canvas_size)
            x, y = canvas_pos
            if rect.collidepoint(x, y):
                cam = self._camera(rect)
                p = cam.unproject_to_plane(x - rect.x, y - rect.y, self.z_s)
                if p is not None:
                    self._clear_final()
                    h = self.handles[len(self.swats) % 2]
                    lim = np.array(aim_limits(h))
                    aim = tuple(np.clip(p[:2], -lim, lim))
                    self.queue.insert(0, Swat(self.t_s + CAM_PREROLL_S, self.speed, h, index=len(self.swats), aim=aim,
                                              aimed_by="player"))

    # -------------------------------------------------------------------------------------------- camera
    _canvas_size = C.DEFAULT_SIZE

    def _main_rect(self, size):
        import pygame as pg
        w, h = size
        return pg.Rect(0, 65, int(w * 0.70), h - 65 - 48)

    @staticmethod
    def _cam_box(W, H):
        """CAM_BOX (drawn for the 1344 x 967 main view) scaled to this view."""
        return (CAM_BOX[0] / 1344 * W, CAM_BOX[1] / 967 * H, CAM_BOX[2] / 1344 * W, CAM_BOX[3] / 967 * H)

    @staticmethod
    def _lead_s(sw) -> float:
        """The swat framing holds the head from this long after the swat begins (it slides in from the top)."""
        return min(0.1, 0.2 * (sw.tc - sw.t0))

    def _wide_camera(self, sw, az, el, W, H):
        s = sw.path_len - sw.v * self._lead_s(sw)
        return fit_camera(az, el, W, H, swat_frame_points(sw, self.z_s, s), self._cam_box(W, H))

    def _cam_weight(self, t, sw, base, az, el, W, H) -> float:
        """How far toward the swat framing the camera is (0 = orbit, 1 = wide): the pre-roll, then the least pull-back
        that keeps the head in the box, never pushing back in faster than over CAM_PUSH_S."""
        if not sw.rec:                                               # queued, not begun: the pre-roll
            return smoothstep((t - (sw.t0 - CAM_PREROLL_S)) / CAM_PREROLL_S)
        hold = 1.0 - smoothstep((t - sw.t0) / CAM_PUSH_S)
        if t >= sw.tc or hold >= 1.0:
            return hold
        rx, ry, _ = head_radii(sw.handle)
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        c = sw.head_centre(max(t, sw.t0 + self._lead_s(sw)), self.z_s)
        rim = np.c_[c[0] + rx * np.cos(ang), c[1] + ry * np.sin(ang), np.full(len(ang), c[2])]
        wide = self._wide_camera(sw, az, el, W, H)
        x0, y0, x1, y1 = self._cam_box(W, H)

        def fits(k):
            pos = (1 - k) * base.pos + k * wide.pos
            p, z = Camera(pos, pos + base.f, W, H, CAM_VFOV).project(rim)
            return bool((z > 0).all() and (p[:, 0] >= x0 - 1).all() and (p[:, 0] <= x1 + 1).all()
                        and (p[:, 1] >= y0 - 1).all() and (p[:, 1] <= y1 + 1).all())
        if fits(hold):
            return hold
        lo, hi = hold, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if fits(mid) else (mid, hi)
        return hi

    def _camera(self, rect, shake=False):
        """Presentation: the orbit camera, pulled back toward the swat framing around each swat (`_cam_weight`), with
        the impact shake applied to the camera itself."""
        t, W, H = self.t_s, rect.w, rect.h
        base = orbit_camera(t, self.seconds, W, H, (0.0, 0.0, self.z_s))
        az, el = orbit_pose(t, self.seconds)
        pos = base.pos
        best, k = None, 0.0
        for sw in ([self.swats[-1]] if self.swats else []) + ([self.queue[0]] if self.queue else []):
            if not sw.rec and t < sw.t0 - CAM_PREROLL_S:
                continue
            w = self._cam_weight(t, sw, base, az, el, W, H)
            if w > k:
                best, k = sw, w
        if best is not None:
            pos = (1 - k) * base.pos + k * self._wide_camera(best, az, el, W, H).pos
        if shake and t < self.shake_until:
            j = self.fx_rng.normal(0, SHAKE_M * (self.shake_until - t) / SHAKE_S / 2, 2)
            pos = pos + j[0] * base.r + j[1] * base.u
        return Camera(pos, pos + base.f, W, H, CAM_VFOV)

    # -------------------------------------------------------------------------------------------- drawing
    def banner(self, text, color, sub="", hold=1.2, size=58, kind=""):
        self.banners.append(dict(t=self.t_s, text=text, color=color, sub=sub, hold=hold, size=size, kind=kind))

    def _mosaic_colors(self, i):
        rad = getattr(self.sim, "col_rad", None)
        if rad is None:
            return np.zeros((self.eyes.n_col, 3), np.uint8)
        return C.eye_colors(rad[i], exposure=MOSAIC_EXPOSURE)

    def _line_surface(self, text, size, color):
        """One line in the bold display font. Its '·' glyph looks like a hyphen, so each DOT separator is drawn as a
        filled circle between the rendered parts."""
        import pygame as pg
        f = self.hud.font(size, True, True)
        imgs = [f.render(p, True, color) for p in str(text).split(DOT)]
        gap = max(8, int(size * 0.42))
        h = max(i.get_height() for i in imgs)
        s = pg.Surface((sum(i.get_width() for i in imgs) + 2 * gap * (len(imgs) - 1), h), pg.SRCALPHA)
        x = 0
        for k, img in enumerate(imgs):
            s.blit(img, (x, 0))
            x += img.get_width()
            if k < len(imgs) - 1:
                pg.draw.circle(s, color, (x + gap, int(h * 0.53)), max(3, size // 12))
                x += 2 * gap
        return s

    def draw(self, surface):
        import pygame as pg
        now = time.perf_counter()
        if self._t_frame is not None:
            self.frame_wall.append(now - self._t_frame)
        self._t_frame = now
        self._canvas_size = surface.get_size()
        W, H = self._canvas_size
        surface.fill(C.BG)
        main = self._main_rect((W, H))
        self._draw_main(surface, main)
        self.hud.header(surface, "SWARM", f"{self.B} flies, {self.B} brains, one GPU: a giant swatter comes down on the crowd",
                        right=C.model_line(self.fb) + ("   ·   CONTROL: swatter invisible to the flies"
                                                        if self.control == "blind" else ""))
        x0 = main.right + 12
        cw = W - x0 - 12
        y = main.y + 10
        y = self._draw_eye(surface, pg.Rect(x0, y, cw, 262)) + 8
        y = self._draw_raster(surface, pg.Rect(x0, y, cw, 236)) + 8
        y = self._draw_traces(surface, pg.Rect(x0, y, cw, 168)) + 8
        self._draw_outcomes(surface, pg.Rect(x0, y, cw, main.bottom - y - 2))
        self.hud.footer(surface, [f"DNp01 ≥ {self.gf_thr:.0f} Hz → escape hop: the shipped body model, no game decoder",
                                  "GAME: swatter, aim, tray, light, rounds (reset), splat / caught rule",
                                  f"flies drawn {FLY_DRAW_SCALE:g}x"])

    # ---- main view
    def _sprites_init(self):
        import pygame as pg
        s = pg.Surface((48, 48), pg.SRCALPHA)
        pg.draw.ellipse(s, (205, 215, 225, 150), (6, 17, 17, 26))            # wings (left / right)
        pg.draw.ellipse(s, (205, 215, 225, 150), (25, 17, 17, 26))
        pg.draw.ellipse(s, (30, 26, 24, 255), (18, 10, 12, 30))               # thorax + abdomen
        pg.draw.circle(s, (20, 18, 18, 255), (24, 10), 6)                     # head
        pg.draw.circle(s, (150, 30, 25, 255), (20, 8), 3)                     # red eyes
        pg.draw.circle(s, (150, 30, 25, 255), (28, 8), 3)
        self._sprites = {"fly": s}

    def _splat_sprite(self, i, size):
        import pygame as pg
        key = (i, size)
        if key not in self._splat_sprites:
            rng = np.random.default_rng(1000 + i)
            s = pg.Surface((size * 2, size * 2), pg.SRCALPHA)
            c = size
            for _ in range(9):
                r = rng.uniform(0.18, 0.42) * size
                off = rng.normal(0, 0.25 * size, 2)
                pg.draw.circle(s, (120, 18, 16, 235), (int(c + off[0]), int(c + off[1])), int(r))
            for _ in range(10):
                a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.55, 0.95) * size
                pg.draw.circle(s, (140, 24, 20, 230), (int(c + d * np.cos(a)), int(c + d * np.sin(a))),
                               max(1, int(rng.uniform(0.04, 0.1) * size)))
            pg.draw.circle(s, (40, 10, 10, 255), (c, c), max(2, size // 5))
            self._splat_sprites[key] = s
        return self._splat_sprites[key]

    def _render(self, cam):
        """Ray-trace the main view (at `view_scale` of its size, for a faster interactive window) -> (H, W, 3) uint8."""
        if self.view_scale != 1.0:
            cam = Camera(cam.pos, cam.pos + cam.f, max(8, round(cam.W * self.view_scale)),
                         max(8, round(cam.H * self.view_scale)), CAM_VFOV)
        o, d = cam.rays(self.device)
        out = [self.cam_world.trace(o[k:k + 400_000], d[k:k + 400_000]) for k in range(0, len(o), 400_000)]
        rad = torch.cat(out).reshape(cam.H, cam.W, 4)
        return np.ascontiguousarray(self.world.to_rgb8(rad, exposure=2.1))

    @staticmethod
    def _ring(pg, overlay, color, alpha, centre, radius, width):
        """A ring with a dark under-stroke, so it reads on the white tray and on the dark shadow alike."""
        if radius < 2 or alpha <= 0:
            return
        pg.draw.circle(overlay, (12, 14, 16, int(alpha * 0.75)), centre, int(radius), width + 3)
        pg.draw.circle(overlay, (*color, int(alpha)), centre, int(radius) - 1, width)

    def _draw_main(self, surface, rect):
        import pygame as pg
        if self._sprites is None:
            self._sprites_init()
        cam = self._camera(rect, shake=True)
        img = self._render(cam)
        view = pg.image.frombuffer(img.tobytes(), (img.shape[1], img.shape[0]), "RGB")
        if view.get_size() != rect.size:
            view = pg.transform.smoothscale(view, rect.size)
        t = self.t_s
        surface.set_clip(rect)
        surface.blit(view, rect.topleft)
        ox, oy = rect.x, rect.y
        pos, heading, air = self._positions()
        sw = self.active
        centre = None if sw is None else sw.head_centre(t, self.z_s)
        occluders = []
        if centre is not None:
            occluders = [(centre, head_radii(sw.handle)),
                         (centre + handle_offset(sw.handle), handle_radii(sw.handle))]

        def hidden(pts):
            h = np.zeros(len(pts), bool)
            for c, r in occluders:
                h |= segment_hits_ellipsoid(cam.pos, pts, c, r)
            return h
        # splats (on the tray): flies splat under the head, or caught in the air and swatted down
        dead = np.flatnonzero(self.dead)
        if len(dead):
            dp = np.c_[self.death_xy[dead], np.full(len(dead), self.z_s + 0.0002)]
            p2, _ = cam.project(dp)
            hid = hidden(dp)
            squash = max(0.35, float(-cam.f[2]))
            for j, i in enumerate(dead):
                if hid[j]:
                    continue
                size = 16
                age = t - self.death_t[i]
                spr = self._splat_sprite(int(i), size)
                grow = min(1.0, 0.4 + age / 0.15)
                spr = pg.transform.smoothscale(spr, (max(2, int(4 * size * grow)), max(2, int(4 * size * grow * squash))))
                surface.blit(spr, spr.get_rect(center=(ox + p2[j, 0], oy + p2[j, 1])))
        # live flies
        live = np.flatnonzero(~self.dead)
        body = pos + np.array([0, 0, 0.0008])
        p2, depth = cam.project(body)
        fwd = np.c_[np.cos(heading), np.sin(heading), np.zeros(self.B)]
        p_head, _ = cam.project(body + 0.0025 * fwd)
        hid = hidden(body)
        shadow_p, _ = cam.project(np.c_[pos[:, :2], np.full(self.B, self.z_s)])
        scale_px = cam.H / 2 / (np.maximum(depth, 1e-3) * cam.tv) * 0.0026 * FLY_DRAW_SCALE   # a 2.6 mm fly, drawn larger
        base = self._sprites["fly"]
        overlay = pg.Surface(rect.size, pg.SRCALPHA)
        for i in live[np.argsort(-depth[live])]:
            if hid[i]:                                               # behind or under the swatter, seen from the camera
                continue
            x, y = p2[i, 0], p2[i, 1]
            gf = float(self.gf[i])
            col = tuple(int(v) for v in gf_color_main(gf))
            size = float(np.clip(scale_px[i], 12, 60))
            if air[i]:
                pg.draw.ellipse(overlay, (0, 0, 0, 70), pg.Rect(shadow_p[i, 0] - size * 0.35, shadow_p[i, 1] - size * 0.15,
                                                                 size * 0.7, size * 0.3))
            if gf > 8:
                rr = size * (0.45 + 0.55 * min(gf / 60.0, 1.4))
                a = int(np.clip(30 + 2.6 * gf, 30, 190))
                pg.draw.circle(overlay, (*col, a // 2), (int(x), int(y)), int(rr))
                pg.draw.circle(overlay, (20, 22, 24, min(160, a)), (int(x), int(y)), int(rr), 1)
                if gf >= self.gf_thr:
                    self._ring(pg, overlay, col, 235, (int(x), int(y)), rr, 2)
            v = p_head[i] - p2[i]
            ang = np.degrees(np.arctan2(-v[1], v[0])) - 90
            spr = pg.transform.rotozoom(base, ang, size / 40.0)
            overlay.blit(spr, spr.get_rect(center=(int(x), int(y))))
        # effects
        keep = []
        for e in self.fx:
            age = t - e["t"]
            if e["kind"] in ("launch", "caught") and age < 0.45:
                keep.append(e)
                at = e["pos"][None] + np.array([0, 0, 0.0008])
                if hidden(at)[0]:
                    continue
                pp, _ = cam.project(at)
                k = age / 0.45
                col = C.AMBER if e["kind"] == "launch" else C.RED
                self._ring(pg, overlay, col, 230 * (1 - k), (int(pp[0, 0]), int(pp[0, 1])), 8 + 26 * k, 3)
            elif e["kind"] == "impact" and age < 0.5:
                keep.append(e)
                k = age / 0.5
                rx, ry, _ = head_radii(e["handle"])
                a = np.linspace(0, 2 * np.pi, 64)
                grow = 1.0 + 0.35 * k
                ring = np.c_[e["aim"][0] + grow * rx * np.cos(a), e["aim"][1] + grow * ry * np.sin(a), np.full(64, self.z_s)]
                pr, _ = cam.project(ring)
                pts = pr.astype(int).tolist()
                pg.draw.polygon(overlay, (12, 14, 16, int(170 * (1 - k))), pts, 9)
                pg.draw.polygon(overlay, (*C.AMBER, int(235 * (1 - k))), pts, 4)
        self.fx = keep
        # the focus fly's reticle and label
        f = self.focus
        if not self.dead[f] and not hid[f]:
            x, y = p2[f]
            s = 26
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts = [(x + sx * s, y + sy * (s - 9)), (x + sx * s, y + sy * s), (x + sx * (s - 9), y + sy * s)]
                    pg.draw.lines(overlay, (12, 14, 16, 200), False, pts, 7)
                    pg.draw.lines(overlay, (*C.WHITE, 240), False, pts, 3)
            lab = self.hud.font(18, True).render(f"FLY {f}", True, C.WHITE)
            pill = lab.get_rect(topleft=(int(x + s + 10), int(y - s - 2))).inflate(14, 6)
            pg.draw.rect(overlay, (*C.BG, 215), pill, border_radius=6)
            overlay.blit(lab, lab.get_rect(center=pill.center))
        surface.blit(overlay, (ox, oy))
        if 0 <= t - self.fade_from < 0.35:                         # a fresh crowd: fade in from dark
            veil = pg.Surface(rect.size, pg.SRCALPHA)
            veil.fill((*C.BG, int(255 * (1 - (t - self.fade_from) / 0.35))))
            surface.blit(veil, rect.topleft)
        self._draw_main_text(surface, rect)
        self._draw_banners(surface, rect)
        if self.final_at is not None and t >= self.final_at and (self.final_until is None or t < self.final_until):
            self._draw_final(surface, rect, min(1.0, (t - self.final_at) / 0.5))
        surface.set_clip(None)

    def _backdrop(self, surface, r, alpha=175):
        import pygame as pg
        s = pg.Surface((r.w, r.h), pg.SRCALPHA)
        pg.draw.rect(s, (*C.BG, alpha), s.get_rect(), border_radius=8)
        surface.blit(s, r.topleft)

    def _text_box(self, surface, x, y, lines, anchor="left", min_w=0):
        """A translucent box sized to its lines. `lines` = [(kind, text, size, colour, chip)], kind 'display' or
        'mono'; returns the box."""
        import pygame as pg
        imgs = []
        for kind, text, size, color, chip in lines:
            img = (self._line_surface(text, size, color) if kind == "display"
                   else self.hud.font(size).render(text, True, color))
            imgs.append((img, chip))
        w = max([min_w] + [img.get_width() + (100 if chip else 0) for img, chip in imgs]) + 32
        gaps = [10] + [6] * (len(imgs) - 1)
        h = sum(img.get_height() for img, _ in imgs) + sum(gaps) + 12
        box = pg.Rect(x - w if anchor == "right" else x, y, w, h)
        self._backdrop(surface, box)
        yy = box.y
        for (img, chip), g in zip(imgs, gaps):
            yy += g
            r = img.get_rect(topright=(box.right - 16, yy)) if anchor == "right" else img.get_rect(topleft=(box.x + 16, yy))
            surface.blit(img, r)
            if chip:
                self.hud.chip(surface, chip, (r.right + 14, r.y + (r.h - 22) // 2), 13)
            yy += img.get_height()
        return box

    def _draw_main_text(self, surface, rect):
        import pygame as pg
        hud, t = self.hud, self.t_s
        # top-left: the round and the swat
        sw = self.active
        title = self._round_label()
        if sw is not None:
            phase = ("coming down" if t < sw.tc else "on the tray" if t < sw.tc + DWELL_S else "lifting")
            lines = [("display", title, 36, C.AMBER, "game"), ("mono", f"{sw.v:.1f} m/s{DOT}{phase}", 24, C.TEXT, None)]
            if t < sw.tc:
                lines.append(("mono", f"height {sw.height(t) * 100:4.1f} cm{DOT}impact in "
                              f"{int(round((sw.tc - t) * 1000))} ms", 20, C.MUTED, None))
                lines.append(("mono", f"{int(sw.rec['inside0'].sum())} flies under it", 20, C.MUTED, None))
            elif sw.rec.get("scored"):
                c = sw.rec["counts"]
                lines.append(("mono", f"jumped in time {c['jumped_in_time']}{DOT}clear {c['clear']}{DOT}"
                              f"hit {c['caught'] + c['splat']}", 20, C.MUTED, None))
                lines.append(("mono", f"{self._hops_since(sw.tc)} hops since impact", 20, C.MUTED, None))
            else:
                lines.append(("mono", "impact", 20, C.MUTED, None))
        else:
            nxt = self.queue[0] if self.queue else None
            last = self.swats[-1] if self.swats else None
            this_round = last is not None and last.rec.get("scored") and last.rec["round"] == self.round
            lines = [("display", title, 36, C.SAGE, "game")]
            if nxt is not None and not this_round:
                lines.append(("mono", f"walking{DOT}swat in {max(0.0, nxt.t0 - t):.1f} s{DOT}{nxt.v:.1f} m/s", 22,
                              C.TEXT, None))
            elif this_round:
                c = last.rec["counts"]
                lines.append(("mono", f"{last.v:.1f} m/s: jumped in time {c['jumped_in_time']}{DOT}clear {c['clear']}",
                              22, C.TEXT, None))
            if nxt is None and not self.headless:
                lines.append(("mono", f"click the tray to swat{DOT}keys 1-5: speed ({self.speed:.1f} m/s)", 18,
                              C.TEXT, None))
                lines.append(("mono", f"R new crowd{DOT}A the {len(ROUNDS)} rounds", 18, C.TEXT, None))
            hops = f"{DOT}{self._hops_since(last.tc)} hops since impact" if this_round else ""
            lines.append(("mono", f"{int((~self.dead).sum())} of {self.B} alive{hops}", 20, C.MUTED, None))
        self._text_box(surface, rect.x + 18, rect.y + 16, lines, min_w=300)
        # top-right: the machine
        lines = [("display", f"{self.B} BRAINS × {self.fb.c.n:,}", 30, C.WHITE, None),
                 ("mono", f"= {self.B * self.fb.c.n / 1e6:.1f} M simulated neurons, one batch", 20, C.TEXT, None)]
        if self.sim_wall:
            k = np.mean(self.sim_wall) / self.dt
            name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "") if self.device.startswith("cuda") else "CPU"
            lines.append(("mono", f"sim step: 1 s of brain = {k:.1f} s wall{DOT}{name}, shared", 18, C.MUTED, None))
        self._text_box(surface, rect.right - 18, rect.y + 16, lines, anchor="right")
        # bottom-left: legend
        box3 = pg.Rect(rect.x + 18, rect.bottom - 88, 620, 76)
        self._backdrop(surface, box3)
        tr = hud.text(surface, "disc = each fly's giant fibre DNp01", (box3.x + 14, box3.y + 8), 17, C.TEXT)
        hud.chip(surface, "connectome", (tr.right + 10, box3.y + 6), 12)
        bar = pg.Rect(box3.x + 14, box3.y + 36, 300, 12)
        ramp = gf_color_main(np.linspace(0, MAIN_GF_TOP, bar.w))
        rs = pg.surfarray.make_surface(np.repeat(ramp[:, None, :], bar.h, 1))
        surface.blit(rs, bar.topleft)
        xt = bar.x + int(bar.w * self.gf_thr / MAIN_GF_TOP)
        pg.draw.line(surface, C.BG, (xt, bar.y - 4), (xt, bar.bottom + 3), 4)
        pg.draw.line(surface, C.WHITE, (xt, bar.y - 4), (xt, bar.bottom + 3), 2)
        hud.text(surface, f"{self.gf_thr:.0f} Hz = hop", (xt, bar.bottom + 3), 15, C.WHITE, anchor="midtop")
        hud.text(surface, f"{MAIN_GF_TOP:.0f}+", (bar.right, bar.bottom + 3), 15, C.MUTED, anchor="topright")
        hud.text(surface, "splat, caught:", (box3.x + 340, box3.y + 34), 16, C.RED)
        hud.text(surface, "GAME rule", (box3.x + 340, box3.y + 52), 16, C.RED)

    def _hops_since(self, t0):
        return sum(1 for t, _ in self.launch_all if t > t0)

    def _draw_banners(self, surface, rect):
        import pygame as pg
        t = self.t_s
        live = [b for b in self.banners if t - b["t"] < b["hold"] + 0.8]
        self.banners = live
        y = rect.bottom - 96
        for b in reversed(live[-2:]):
            age = t - b["t"]
            a = 1.0 if age < b["hold"] else max(0.0, 1 - (age - b["hold"]) / 0.8)
            img = self._line_surface(b["text"], b["size"], b["color"])
            sub = self.hud.font(22).render(b["sub"], True, C.TEXT) if b["sub"] else None
            h = img.get_height() + (sub.get_height() + 6 if sub else 0) + 20
            w = max(img.get_width(), sub.get_width() if sub else 0) + 60
            band = pg.Surface((w, h), pg.SRCALPHA)
            pg.draw.rect(band, (*C.BG, int(200 * a)), band.get_rect(), border_radius=12)
            pg.draw.rect(band, (*b["color"], int(255 * a)), band.get_rect(), 2, border_radius=12)
            img.set_alpha(int(255 * a))
            band.blit(img, img.get_rect(midtop=(w // 2, 10)))
            if sub:
                sub.set_alpha(int(255 * a))
                band.blit(sub, sub.get_rect(midtop=(w // 2, 12 + img.get_height())))
            r = band.get_rect(midbottom=(rect.centerx, y))
            surface.blit(band, r)
            y = r.top - 10

    def _draw_final(self, surface, rect, a):
        import pygame as pg
        hud = self.hud
        done = [s for s in self.swats if s.rec.get("scored")]
        if not done:
            return
        tot = {k: sum(s.rec["counts"][k] for s in done) for k in ("at_risk", "jumped_in_time", "clear", "caught", "splat",
                                                                   "jumped_and_clear")}
        w, h = 940, 290 + 54 * len(done)
        card = pg.Surface((w, h), pg.SRCALPHA)
        pg.draw.rect(card, (*C.BG, int(232 * a)), card.get_rect(), border_radius=14)
        pg.draw.rect(card, (*C.LINE, int(255 * a)), card.get_rect(), 2, border_radius=14)
        card.blit(self._line_surface(f"{len(done)} SWAT{'S' if len(done) != 1 else ''}{DOT}{self.B} FLIES EACH", 44,
                                     C.WHITE), (28, 18))
        hud.chip(card, "game", (w - 110, 30), 13)
        hud.text(card, f"giant fibre fired in time: {tot['jumped_in_time']} of {tot['at_risk']} flies under the swatter",
                 (28, 76), 22, C.AMBER)
        hud.text(card, f"got clear: {tot['clear']} ({tot['jumped_and_clear']} by the hop){DOT}caught in the air: "
                 f"{tot['caught']}{DOT}splat: {tot['splat']}", (28, 106), 22, C.TEXT)
        for k, s in enumerate(done):
            y = 148 + 54 * k
            c = s.rec["counts"]
            risk = max(1, c["at_risk"])
            hud.text(card, f"{s.v:.1f} m/s", (28, y + 6), 26, C.AMBER, bold=True)
            bar = pg.Rect(170, y + 4, 400, 20)
            pg.draw.rect(card, C.INSET, bar, border_radius=4)
            pg.draw.rect(card, C.AMBER, (bar.x, bar.y, int(bar.w * c["jumped_in_time"] / risk), bar.h), border_radius=4)
            bar2 = pg.Rect(170, y + 28, 400, 9)
            pg.draw.rect(card, C.INSET, bar2, border_radius=3)
            pg.draw.rect(card, C.SAGE, (bar2.x, bar2.y, int(bar2.w * c["clear"] / risk), bar2.h), border_radius=3)
            hud.text(card, f"{c['jumped_in_time']}/{c['at_risk']} jumped{DOT}{c['clear']} clear", (bar.right + 18, y + 6),
                     24, C.TEXT)
        y = 148 + 54 * len(done) + 4
        pg.draw.rect(card, C.AMBER, (28, y + 5, 14, 12))
        hud.text(card, "jumped in time (DNp01 ≥ 33 Hz before impact)", (50, y), 17, C.MUTED)
        pg.draw.rect(card, C.SAGE, (520, y + 7, 14, 7))
        hud.text(card, "got clear (never touched)", (542, y), 17, C.MUTED)
        rim = self.log.summary.get("by_bearing", {})
        ra, rb = rim.get("rim facing"), rim.get("rim facing away")
        if ra and rb and ra["n"] and rb["n"]:
            hud.text(card, f"at the rim, facing its landing point: GF in time {ra['jumped_in_time']}/{ra['n']}, "
                     f"clear {ra['clear']}", (28, y + 32), 20, C.TEXT)
            hud.text(card, f"at the rim, facing away:              GF in time {rb['jumped_in_time']}/{rb['n']}, "
                     f"clear {rb['clear']}", (28, y + 58), 20, C.TEXT)
        jc = sum(s.rec["counts"]["jumped_and_caught"] for s in done)
        hud.text(card, f"the shipped hop goes forward: {jc} of the {tot['jumped_in_time']} that jumped in time were "
                 f"caught in the air", (28, y + 90), 17, C.MUTED)
        card.set_alpha(int(255 * a))
        surface.blit(card, card.get_rect(center=(rect.centerx, rect.centery - 30)))

    # ---- right column
    def _draw_eye(self, surface, rect):
        import pygame as pg
        hud = self.hud
        inner = hud.panel(surface, rect, "What the fly sees")
        f = self.focus
        frozen = self.focus_frozen is not None and self.t_s < self.focus_frozen[1]
        if frozen:
            colors = self.focus_frozen[0]
        else:
            if self.focus_frozen is not None:
                self.focus_frozen = None
            if self.dead[f]:
                live = np.flatnonzero(~self.dead)
                if len(live):
                    self.focus = f = int(live[np.argmin(np.hypot(*self._positions()[0][live, :2].T))])
            colors = self._mosaic_colors(f)
        mrect = pg.Rect(inner.x, inner.y + 2, inner.w, inner.h - 26)
        hud.mosaic(surface, mrect, self.eyes, colors)
        hud.text(surface, f"FLY {f}", (rect.right - 14, rect.y + 8), 20, C.WHITE, bold=True, anchor="topright")
        hud.text(surface, "1,466 ommatidia: the radiance handed to fb.vision", (inner.x, inner.bottom - 20), 15, C.DIM)
        if frozen:
            s = pg.Surface(mrect.size, pg.SRCALPHA)
            s.fill((120, 10, 10, 120))
            surface.blit(s, mrect.topleft)
            img = self._line_surface(self.focus_frozen[2], 64, C.WHITE)
            surface.blit(img, img.get_rect(center=mrect.center))
        return rect.bottom

    def _draw_raster(self, surface, rect):
        import pygame as pg
        hud = self.hud
        inner = hud.panel(surface, rect, f"{self.B} giant fibres", "connectome")
        hud.text(surface, "last 3 s · nearest the aim on top", (rect.right - 14, rect.y + 10), 14, C.MUTED,
                 anchor="topright")
        hist = np.array(self.gf_hist) if self.gf_hist else np.zeros((1, self.B))
        T = self.n_hist
        if len(hist) < T:
            hist = np.r_[np.zeros((T - len(hist), self.B)), hist]
        hist = hist[: T // 2 * 2].reshape(T // 2, 2, self.B).max(1)             # one column per video frame (20 ms)
        rows = self.order
        img = gf_color(hist[:, rows].T).astype(np.float32)                      # (B, cols, 3)
        dead_rows = self.dead[rows]
        img[dead_rows] = img[dead_rows] * 0.25 + np.array([60, 8, 8]) * 0.75
        img = img.astype(np.uint8)
        lab_w = 16
        area = pg.Rect(inner.x + lab_w, inner.y, inner.w - lab_w - 44, inner.h - 18)
        rs = pg.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
        surface.blit(pg.transform.scale(rs, area.size), area.topleft)
        rh = area.h / self.B
        for k, i in enumerate(rows):                                            # outcome of each fly's last swat
            oc = self.death_kind[i] if self.dead[i] else self.last_outcome[i]
            if oc in OUTCOME_COLOR and oc != "safe":
                pg.draw.rect(surface, OUTCOME_COLOR[oc], (inner.x + 2, area.y + k * rh, 10, max(1, rh - 0.5)))
        t = self.t_s
        for tt in self.reset_times:
            if t - HISTORY_S <= tt <= t:
                x = area.right - (t - tt) / HISTORY_S * area.w
                pg.draw.line(surface, C.SAGE, (x, area.y - 2), (x, area.bottom + 2), 1)
                hud.text(surface, "reset", (x + 4, area.bottom + 1), 13, C.SAGE)
        for s in self.swats:
            for tt, col in ((s.t0, C.AMBER), (s.tc, C.WHITE)):
                if t - HISTORY_S <= tt <= t:
                    x = area.right - (t - tt) / HISTORY_S * area.w
                    pg.draw.line(surface, col, (x, area.y - 2), (x, area.bottom + 2), 2 if col == C.WHITE else 1)
                    if col == C.WHITE:
                        hud.text(surface, "impact", (x - 4, area.bottom + 1), 13, C.WHITE, anchor="topright")
        now_n = int(np.sum(self.gf[~self.dead] >= self.gf_thr))
        hud.text(surface, f"{now_n}", (inner.right, area.y + area.h / 2 - 18), 28, C.AMBER if now_n else C.DIM, bold=True,
                 anchor="topright")
        hud.text(surface, f"≥{self.gf_thr:.0f}", (inner.right, area.y + area.h / 2 + 12), 13, C.MUTED, anchor="topright")
        return rect.bottom

    def _draw_traces(self, surface, rect):
        import pygame as pg
        hud = self.hud
        f = self.focus
        inner = hud.panel(surface, rect, f"Fly {f}", "connectome")
        gf = np.array([g[f] for g in self.gf_hist]) if self.gf_hist else np.zeros(1)
        lc = np.array([v[f] for v in self.lc_hist]) if self.lc_hist else np.zeros((1, 2))
        half = (inner.w - 14) // 2
        r1 = pg.Rect(inner.x, inner.y + 26, half, inner.h - 30)
        r2 = pg.Rect(inner.x + half + 14, inner.y + 26, half, inner.h - 30)
        hud.text(surface, "giant fibre DNp01", (r1.x, inner.y - 2), 15, C.TEXT)
        cur = float(gf[-1])
        hud.text(surface, f"{cur:.0f} Hz", (r1.right, inner.y - 8), 24, C.AMBER if cur >= self.gf_thr else C.SAGE, bold=True,
                 anchor="topright")
        hud.trace(surface, r1, gf, 0, 90, C.AMBER, threshold=self.gf_thr)
        hud.text(surface, "LPLC2 · LC4 mean", (r2.x, inner.y - 2), 15, C.TEXT)
        vmax = max(10.0, float(np.max(lc)) * 1.15)
        hud.text(surface, f"{lc[-1, 1]:.0f} · {lc[-1, 0]:.0f} Hz", (r2.right, inner.y - 8), 24, C.TEAL, bold=True,
                 anchor="topright")
        hud.trace(surface, r2, lc[:, 1], 0, vmax, C.TEAL)
        _polyline(surface, r2, lc[:, 0], 0, vmax, C.LILAC)
        return rect.bottom

    def _draw_outcomes(self, surface, rect):
        import pygame as pg
        hud = self.hud
        inner = hud.panel(surface, rect, "Outcomes", "game")
        hud.text(surface, "map: its landing point, in each fly's frame", (rect.right - 14, rect.y + 10), 14, C.MUTED,
                 anchor="topright")
        done = [s for s in self.swats if s.rec.get("scored")]
        tot = {k: sum(s.rec["counts"][k] for s in done) for k in ("at_risk", "jumped_in_time", "clear", "caught", "splat")}
        x = inner.x
        for k, (val, col, lab) in enumerate(((tot["jumped_in_time"], C.AMBER, "jumped in time"),
                                             (tot["clear"], C.SAGE, "got clear"),
                                             (tot["caught"] + tot["splat"], C.RED, "hit"))):
            xx = x + (0, 134, 240)[k]
            hud.text(surface, f"{val}", (xx, inner.y - 6), 40, col, bold=True, display=True)
            hud.text(surface, lab, (xx + 2, inner.y + 38), 13, C.MUTED)
        hud.text(surface, f"of {tot['at_risk']} flies under the swatter", (x, inner.y + 58), 14, C.MUTED)
        y = inner.y + 84
        cols = ((x + 180, "jumped", C.AMBER), (x + 230, "clear", C.SAGE), (x + 268, "hit", C.RED))
        for cx_, lab, col in cols:
            hud.text(surface, lab, (cx_, y), 13, col, anchor="topright")
        y += 18
        for s in done[-5:]:
            c = s.rec["counts"]
            hud.text(surface, f"R{s.rec['round'] - self.round_base + 1 if self.n_rounds else s.rec['round'] + 1} "
                     f"{s.v:.1f} m/s", (x, y), 16, C.AMBER)
            for (cx_, _, _), val in zip(cols, (f"{c['jumped_in_time']}/{c['at_risk']}", f"{c['clear']}",
                                                f"{c['caught'] + c['splat']}")):
                hud.text(surface, val, (cx_, y), 16, C.TEXT, anchor="topright")
            y += 20
        # polar map: the swatter's landing point in each at-risk fly's own frame when the swat began, and what happened
        R = int(min((inner.w - 300) // 2, (inner.h - 44) // 2))
        cx, cy = inner.right - R - 8, inner.y + R + 18
        for k in (1.0, 0.5):
            pg.draw.circle(surface, C.LINE, (cx, cy), int(R * k / 1.4), 1)
        pg.draw.circle(surface, C.LINE, (cx, cy), R, 1)
        pg.draw.line(surface, C.LINE, (cx, cy - R), (cx, cy + R), 1)
        pg.draw.line(surface, C.LINE, (cx - R, cy), (cx + R, cy), 1)
        hud.text(surface, "ahead", (cx, cy - R - 17), 13, C.MUTED, anchor="midtop")
        hud.text(surface, "behind", (cx, cy + R + 1), 13, C.MUTED, anchor="midtop")
        hud.text(surface, "L", (cx - R - 4, cy), 13, C.MUTED, anchor="midright")
        hud.text(surface, "R", (cx + R + 4, cy), 13, C.MUTED, anchor="midleft")
        hud.text(surface, "rim", (cx + int(R / 1.4 * 0.7), cy - int(R / 1.4 * 0.7) - 14), 12, C.DIM)
        for j, s in enumerate(done):
            fade = 0.45 + 0.55 * (j + 1) / len(done)
            out, jumped = s.rec["outcome"], s.rec["jumped"]
            for i in np.flatnonzero(np.isin(out, ("clear", "caught", "splat"))):
                rr = min(float(s.rec["rho0"][i]), 1.4) / 1.4 * R
                b = np.deg2rad(s.rec["bearing0"][i])
                px, py = cx - np.sin(b) * rr, cy - np.cos(b) * rr
                col = tuple(int(c * fade + C.PANEL[k] * (1 - fade)) for k, c in enumerate(OUTCOME_COLOR[out[i]]))
                pg.draw.circle(surface, col, (int(px), int(py)), 4, 0 if jumped[i] else 1)
        # legend
        ly = inner.bottom - 16
        pg.draw.circle(surface, C.MUTED, (x + 5, ly + 8), 4)
        t1 = hud.text(surface, "jumped", (x + 14, ly), 13, C.MUTED)
        pg.draw.circle(surface, C.MUTED, (t1.right + 13, ly + 8), 4, 1)
        t2 = hud.text(surface, "did not", (t1.right + 22, ly), 13, C.MUTED)
        xx = t2.right + 14
        for oc in ("clear", "caught", "splat"):
            xx = hud.text(surface, oc, (xx, ly), 13, OUTCOME_COLOR[oc]).right + 10


def _polyline(surface, rect, values, vmin, vmax, color, width=2):
    """A second line over a `Hud.trace` (which repaints its background)."""
    import pygame as pg
    v = np.asarray(values, float)
    if len(v) < 2:
        return
    x = rect.x + np.linspace(0, rect.w - 1, len(v))
    y = rect.bottom - 1 - np.clip((v - vmin) / (vmax - vmin), 0, 1) * (rect.h - 2)
    pg.draw.lines(surface, color, False, np.c_[x, y].round().astype(int).tolist(), width)


def measure(args):
    """--measure: run the rounds (tick only, no drawing, no pygame) and save the run log. The same game code and rules
    as a clip; the log's argv and source hash name this file."""
    game = Swarm(args)
    n = int(round(args.seconds / (C.TICK_MS / 1000.0)))
    t_wall = time.time()
    for k in range(n):
        game.tick()
        if k % 100 == 0:
            print(f"  swarm --measure: {game.t_s:5.2f} / {args.seconds:.1f} s brain, {time.time() - t_wall:6.0f} s wall",
                  flush=True)
    game._summarize()
    game.log.meta.update(mode="measure (tick only, no drawing)", wall_s=round(time.time() - t_wall, 1),
                         brain_s=round(game.t_s, 3))
    path = args.log or f"out/games/swarm/measure_seed{args.seed}{'_blind' if args.control == 'blind' else ''}.json"
    p = game.log.save(path, game.brains())
    s = game.log.summary
    for r in s["rounds"]:
        print(f"  round {r['round']} {r['speed_mps']:.1f} m/s: at risk {r['at_risk']}, jumped in time "
              f"{r['jumped_in_time']}, clear {r['clear']} (by the hop {r['clear_by_hop']}), caught {r['caught']}, "
              f"splat {r['splat']}, late {r['late_hops']}, lead {r['hop_lead_ms_median']} ms, "
              f"GF peak {r['gf_peak_before_impact_hz_median_at_risk']} Hz")
    print(f"  totals {s['totals']}")
    print(f"run log {p}")
    return p


def main(argv=None):
    ap = C.standard_args(__doc__.splitlines()[0], seconds=CLIP_SECONDS)
    ap.add_argument("--batch", type=int, default=64, help="flies (batch rows); 64 is the game, fewer for a faster window")
    ap.add_argument("--control", choices=("none", "blind"), default="none",
                    help="blind: the swatter is not in the flies' scenes (same seed, same rule)")
    ap.add_argument("--spacing", type=float, default=0.024, help="GAME: start-grid spacing (m)")
    ap.add_argument("--jitter", type=float, default=0.004, help="GAME: start-position jitter SD (m)")
    ap.add_argument("--auto", action="store_true", help="interactive: run the recorded swat schedule")
    ap.add_argument("--room-light", choices=("globe", "shipped"), default="globe",
                    help="GAME: 'globe' moves the room's point light just below the lamp's globe so it lights the "
                         "scene; 'shipped' keeps world.make_room's (inside the globe: ambient light only)")
    ap.add_argument("--no-fast", action="store_true", help="CUDA: plain backend (no graphs / native kernels / events)")
    ap.add_argument("--measure", action="store_true",
                    help="run the rounds for --seconds without drawing and save the run log (--log, default "
                         "out/games/swarm/measure_seed<seed>[_blind].json)")
    ap.add_argument("--view-scale", type=float, default=None,
                    help="presentation: ray-trace the main view at this fraction of its size (default 1 when recording, "
                         "0.5 in the window)")
    args = ap.parse_args(argv)
    if args.measure:
        return measure(args)
    game = Swarm(args)
    C.run(game, args)


if __name__ == "__main__":
    main()
