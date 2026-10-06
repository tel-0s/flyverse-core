"""swat -- a fly swatter against the giant fibre. The raw MaleCNS v1.0 fly walks on the room table (scripts/room_demo.py's
scene with its fence) and a fly swatter (GAME) swings at it. Nothing is decoded: the only path from the brain to the
game is the shipped body model (flyverse/body.py). Locomotion walks the fly, and `Flight.maybe_takeoff` launches the
escape jump when the giant fibre (DNp01, mean rate) reaches 33 Hz.

The rule (PLAN.md 1, where it is called 'escaped'): the fly is IN TIME when the body takes off on a 10 ms tick before
the tick on which the paddle reaches the spot where it stands; otherwise SPLAT. Both are read on one clock
(`referee_step`): each tick the brain steps, the paddle moves to that tick's time and is tested against the fly where
it stands, and only then does the body act, so it cannot also take off on the tick the paddle arrives (a tie goes to
the paddle). The margin is the number of whole ticks from the takeoff tick to the arrival tick. The hop is not scored:
it is the body model's fixed 45 deg forward jump, the swing ends where the paddle meets it in the air, and the screen
and the log say when that happened.

    python games/swat.py                                              # interactive
    python games/swat.py --record out/games/swat/dev.mp4 --seed 100 --screenshot out/games/swat/dev.png
    python games/swat.py --record out/games/swat/control.mp4 --seed 100 --control   # the fly cannot see the swatter

Interactive: the swatter hovers where the mouse points (on the mouse ray, 0.45 m from the fly, at least 20 deg up);
hold the left button to wind up (0.3 -> 2.5 m/s over 1.2 s of wall-clock time), release to swing (queued until the fly
has settled). Right-drag or the arrow keys orbit the camera, the wheel zooms, HOME resets it, A starts / stops the
scripted sequence, ESC quits.

Record: the scripted SCHEDULE of swats (GAME), the same for every seed. A swing starts once the swatter has aimed for
0.7 s and the fly has stood on the table for 1.2 s (the body's own landing refractory is 1.0 s). Photo finishes
(takeoff <= 30 ms before the arrival tick, or a giant fibre that crossed <= 30 ms after contact) get a 0.25x replay of
-250 .. +120 ms around the arrival tick, rendered while the run went on and shown 0.9 s after it. The clip ends 3 s
after the end card appears (or at --seconds).
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (puts the repo on sys.path)
from common import AMBER, BG, DIM, LILAC, MUTED, RED, SAGE, TEAL, TEXT, WHITE  # noqa: E402
import torch  # noqa: E402
from flyverse import world as W  # noqa: E402

# ------------------------------------------------------------------------------------------------ the game's numbers
TICK_S = common.TICK_MS / 1000.0
PADDLE_RADII = (0.06, 0.045, 0.004)      # m: the swatter head, a flattened ellipsoid lying face down (PLAN.md 1)
HANDLE_RADII = (0.19, 0.006, 0.004)      # m: the handle, a thin ellipsoid rising 20 deg from the paddle's rim
HANDLE_TILT_DEG = 20.0
CONTACT_GAP = 0.004                      # m: contact = the eye comes within 4 mm of the paddle (PLAN.md 1)
START_DIST = 0.45                        # m: the paddle centre hovers this far from the fly's eye while aiming
AIM_S = 0.7                              # s of aiming before a swing
WARMUP_S = 1.5                           # no swing in the first 1.5 s of brain time (the optic lobe starts from rest)
SETTLE_S = 1.2                           # s the fly must have stood on the table before a swing (landing refractory 1.0)
PIN_S = 0.4                              # s a splatted fly is held under the paddle (GAME)
LIFT_SPEED = 1.0                         # m/s the swatter lifts back to START_DIST after contact
MOVE_S = 1.0                             # s to swing round to the next direction, at START_DIST (smoothstep)
ANCHOR_TAU_S = 0.25                      # s: between swings the aim follows the fly's eye with this lag
REPLAY_PRE_S, REPLAY_POST_S = 0.25, 0.12  # the replay window around the paddle's arrival tick
REPLAY_DELAY_S = 0.9                     # the replay starts this long after the arrival tick
REPLAY_SLOW = 4                          # 0.25x
REPLAY_MARGIN_MS = 30.0                  # replay a swat when its margin, or a splat's late GF crossing, is <= this
REPLAY_SCALE, REPLAY_WIDE_SCALE = 0.7, 0.5  # render scale of the replay's stored close-up and WIDE inset
CUT_DIST, CUT_TTC_S = 0.06, 0.08         # the main view cuts to the close-up when the paddle is this near (m or s)
END_CARD_S = 3.0                         # a recording ends this long after the end card appears
WINDUP_S, V_MIN, V_MAX = 1.2, 0.3, 2.5   # interactive wind-up (wall-clock seconds held)

# The display cameras (GAME; nothing here feeds the brain). (d_min, d_max) m, fov deg, where the fly sits in frame.
HERO_CAM = dict(d=(0.024, 0.030), fov=30.0, look=0.10, h_frac=0.12, h=(0.0020, 0.006), follow=1.0, backoff=(1.0, 0.012))
REPLAY_CAM = dict(d=(0.040, 0.052), fov=18.0, look=0.45, h_frac=0.12, h=(0.0022, 0.008), follow=0.85, backoff=(1.5, 0.015))
FLYCAM_D, FLYCAM_FOV, FLYCAM_EL = 0.0115, 30.0, 14.0   # the picture-in-picture fly-cam while the main view is WIDE

# The scripted sequence (GAME): (label, azimuth deg in the fly's frame (0 ahead, + left), elevation deg, speed m/s).
# Rising speed, varied direction; the same for every seed.
SCHEDULE = [
    ("from above", 0.0, 90.0, 0.5),
    ("from the front", 0.0, 45.0, 0.6),
    ("from the left", 90.0, 35.0, 0.7),
    ("from above", 0.0, 90.0, 0.8),
    ("from the right", -90.0, 35.0, 0.9),
    ("from behind", 180.0, 45.0, 1.0),
    ("from the front", 0.0, 45.0, 1.2),
    ("from above", 0.0, 90.0, 1.5),
    ("from the left", 90.0, 35.0, 2.0),
    ("from above", 0.0, 90.0, 2.5),
]

# extra materials [UV, B, G, R]: the fly model, seen only by the human camera (the fly never sees its own body)
MATERIALS = {
    "fly_body": (0.05, 0.13, 0.22, 0.38),
    "fly_abdomen": (0.05, 0.16, 0.27, 0.42),
    "fly_eye": (0.03, 0.04, 0.06, 0.85),
    "fly_wing": (0.35, 0.62, 0.65, 0.68),
}

Z = np.array([0.0, 0.0, 1.0])
LINE_HI = (120, 138, 142)
GF_VMAX, LPLC2_VMAX, LC4_VMAX = 120.0, 15.0, 4.0     # HUD trace scales, Hz


# ------------------------------------------------------------------------------------------------ pure geometry
def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def approach_dir(heading, az_deg, el_deg):
    """World unit vector from the fly towards the swatter: azimuth in the fly's frame (0 ahead, + left), elevation up."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    fh = np.array([math.cos(heading), math.sin(heading), 0.0])
    lh = np.array([-math.sin(heading), math.cos(heading), 0.0])
    return unit(math.cos(e) * math.cos(a) * fh + math.cos(e) * math.sin(a) * lh + math.sin(e) * Z)


def clamp_elevation(u, lo_deg=20.0, hi_deg=90.0):
    """u with its elevation clamped to [lo, hi] deg and its azimuth kept."""
    u = unit(u)
    h = math.hypot(u[0], u[1])
    e = math.radians(min(max(math.degrees(math.atan2(u[2], h)), lo_deg), hi_deg))
    az = math.atan2(u[1], u[0]) if h > 1e-9 else 0.0
    return np.array([math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)])


def paddle_frame(handle_az):
    """Rotation (columns = local x, y, z in world) of the face-down paddle whose long axis points along `handle_az`."""
    return rot_z(handle_az)


def handle_pose(center, handle_az):
    """(centre, rotation) of the handle: from the paddle's rim along `handle_az`, rising HANDLE_TILT_DEG."""
    h = np.array([math.cos(handle_az), math.sin(handle_az), 0.0])
    t = math.radians(HANDLE_TILT_DEG)
    x = math.cos(t) * h + math.sin(t) * Z
    y = unit(np.cross(Z, h))
    c = np.asarray(center, float) + h * PADDLE_RADII[0] + x * (HANDLE_RADII[0] + 0.01)
    return c, np.stack([x, y, np.cross(x, y)], axis=1)


def inflated_metric(point, center, rot, radii, gap):
    """|R^T (p - c) / (radii + gap)|: <= 1 when `point` is inside the ellipsoid grown by `gap` on every semi-axis."""
    q = (np.asarray(rot).T @ (np.asarray(point, float) - np.asarray(center, float))) / (np.asarray(radii) + gap)
    return float(np.sqrt((q * q).sum()))


def contact_distance(u, rot, radii=PADDLE_RADII, gap=CONTACT_GAP):
    """Distance along u from the eye to the paddle centre at contact: the eye on the paddle grown by `gap`
    (8 mm for a swing from straight above: a 4 mm half-thickness plus the 4 mm gap)."""
    q = (np.asarray(rot).T @ np.asarray(u, float)) / (np.asarray(radii) + gap)
    return float(1.0 / np.sqrt((q * q).sum()))


def slerp(u0, u1, s):
    u0, u1 = unit(u0), unit(u1)
    d = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    if d > 0.9995:
        return unit(u0 + s * (u1 - u0))
    if d < -0.9995:                                   # opposite: go over the top
        mid = Z.copy() if abs(u0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        return slerp(u0, mid, 2 * s) if s < 0.5 else slerp(mid, u1, 2 * s - 1)
    th = math.acos(d)
    return unit((math.sin((1 - s) * th) * u0 + math.sin(s * th) * u1) / math.sin(th))


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ema(prev, target, dt, tau):
    a = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
    return prev + a * (np.asarray(target, float) - prev)


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def windup_speed(held_s):
    """Interactive wind-up: 0.3 m/s at the press, 2.5 m/s after 1.2 s held (smoothstep)."""
    return V_MIN + (V_MAX - V_MIN) * smoothstep(held_s / WINDUP_S)


def fly_parts(pos, fwd, left, up, gait=None):
    """The fly model for the human camera: [(centre, radii, rotation, material)] in metres. Body length ~2.6 mm, eyes
    ~1 mm above the feet (the body model's eye height is 1.2 mm). Wings stay folded: the raw model has no flight
    state, and its escape is a ballistic hop. `gait` (cycles, or None = standing) swings the legs in a tripod
    (L1 R2 L3 against R1 L2 R3) for the camera only; nothing reads it back."""
    mm = 1e-3
    Rb = np.stack([unit(fwd), unit(left), unit(up)], axis=1)
    parts = [((0.10, 0.0, 0.78), (0.55, 0.42, 0.40), 0.0, "fly_body"),
             ((0.78, 0.0, 0.92), (0.28, 0.40, 0.33), 0.0, "fly_body"),
             ((0.80, 0.27, 0.94), (0.24, 0.17, 0.30), 0.0, "fly_eye"),
             ((0.80, -0.27, 0.94), (0.24, 0.17, 0.30), 0.0, "fly_eye"),
             ((-0.85, 0.0, 0.72), (0.74, 0.42, 0.38), 0.0, "fly_abdomen"),
             ((-0.78, 0.30, 1.14), (1.05, 0.36, 0.025), math.radians(-12), "fly_wing"),
             ((-0.78, -0.30, 1.14), (1.05, 0.36, 0.025), math.radians(12), "fly_wing")]
    out = [(np.asarray(pos, float) + Rb @ (np.asarray(c) * mm), tuple(r * mm for r in rad), Rb @ rot_z(yaw), mat)
           for c, rad, yaw, mat in parts]
    for pair, (hx, fx) in enumerate(((0.35, 0.75), (0.10, 0.12), (-0.15, -0.60))):   # three leg pairs, hip -> foot
        for side in (1.0, -1.0):
            hip = np.array([hx, 0.22 * side, 0.50]); foot = np.array([fx, 0.80 * side, 0.0])
            if gait is not None:
                ph = 2 * math.pi * (gait + (0.5 if (pair % 2 == 0) == (side < 0) else 0.0))
                foot = foot + np.array([0.22 * math.sin(ph), 0.0, 0.12 * max(math.cos(ph), 0.0)])
            ax = unit(foot - hip); length = float(np.linalg.norm(foot - hip))
            y = unit(np.cross(Z, ax))
            R = np.stack([ax, y, np.cross(ax, y)], axis=1)
            out.append((np.asarray(pos, float) + Rb @ ((hip + foot) / 2 * mm), (length / 2 * mm, 0.045 * mm, 0.045 * mm),
                        Rb @ R, "fly_body"))
    return out


N_FLY_PARTS = 13
STRIDE_M = 0.00093                       # display gait: one step cycle per 0.93 mm walked (body.LegCycle.step_ref_m)


# ------------------------------------------------------------------------------------------------ camera
class Camera:
    """A pinhole camera with the usual image convention: column 0 is the camera's left, row 0 is up."""

    def __init__(self, pos, look_at, width, height, fov_deg, up=(0.0, 0.0, 1.0)):
        self.pos = np.asarray(pos, float)
        self.f = unit(np.asarray(look_at, float) - self.pos)
        r = np.cross(self.f, up)
        self.r = unit(r if np.linalg.norm(r) > 1e-6 else np.array([0.0, -1.0, 0.0]))   # right
        self.u = np.cross(self.r, self.f)                                               # up
        self.w, self.h = int(width), int(height)
        self.tan = math.tan(math.radians(fov_deg) / 2)

    def rays(self, device):
        asp = self.w / self.h
        xs = torch.linspace(-self.tan * asp, self.tan * asp, self.w, device=device)
        ys = torch.linspace(self.tan, -self.tan, self.h, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        f, r, u = (torch.tensor(v, dtype=torch.float32, device=device) for v in (self.f, self.r, self.u))
        d = f[None, None] + gx[..., None] * r[None, None] + gy[..., None] * u[None, None]
        return (d / d.norm(dim=-1, keepdim=True)).reshape(-1, 3)

    def project(self, p):
        """World point -> (x, y) pixel, or None behind the camera."""
        v = np.asarray(p, float) - self.pos
        z = float(v @ self.f)
        if z <= 1e-6:
            return None
        x = float(v @ self.r) / z / (self.tan * self.w / self.h)
        y = float(v @ self.u) / z / self.tan
        return (x * 0.5 + 0.5) * (self.w - 1), (0.5 - y * 0.5) * (self.h - 1)

    def pixels_per_metre(self, p):
        z = max(float((np.asarray(p, float) - self.pos) @ self.f), 1e-6)
        return self.h / (2 * self.tan * z)

    def ray_through(self, x, y):
        gx = (x / max(self.w - 1, 1) * 2 - 1) * self.tan * self.w / self.h
        gy = (1 - y / max(self.h - 1, 1) * 2) * self.tan
        return unit(self.f + gx * self.r + gy * self.u)


@dataclass
class CameraRig:
    """Orbit camera: look-at point, distance, azimuth and elevation (deg), each eased towards its goal."""
    target: np.ndarray
    dist: float = 0.25
    az: float = 0.0
    el: float = 14.0
    fov: float = 50.0

    def update(self, target, dist, az, el, dt, tau_target=0.08, tau_dist=0.12, tau_angle=0.6, tau_el=None):
        self.target = ema(self.target, target, dt, tau_target)
        self.dist = float(math.exp(ema(math.log(self.dist), math.log(dist), dt, tau_dist)))
        a = 1.0 - math.exp(-dt / tau_angle)
        self.az = self.az + a * math.degrees(wrap_angle(math.radians(az - self.az)))
        b = a if tau_el is None else 1.0 - math.exp(-dt / tau_el)
        self.el = self.el + b * (el - self.el)

    def camera(self, width, height):
        az, el = math.radians(self.az), math.radians(self.el)
        pos = self.target + self.dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return Camera(pos, self.target, width, height, self.fov)


def aim_point(origin, d, center, radius):
    """Where a mouse ray meets the aiming sphere: the first crossing in front of the camera (the far one when the camera
    is inside the sphere); when the ray misses, the sphere point nearest to the ray."""
    o, d, c = np.asarray(origin, float), unit(d), np.asarray(center, float)
    oc = o - c
    b = float(oc @ d); cc = float(oc @ oc) - radius * radius
    disc = b * b - cc
    if disc >= 0:
        for t in (-b - math.sqrt(disc), -b + math.sqrt(disc)):
            if t > 0:
                return o + t * d
    closest = o + max(-b, 0.0) * d
    return c + radius * unit(closest - c)


def hero_goal(eye, airborne, ground_z, paddle_underside_z, cam):
    """A low close-up that looks in under the paddle (GAME, display only): (look-at point, distance, elevation deg).
    The distance pushes in from d_max to d_min as the paddle's underside comes down to the eye (frame half-height
    0.6 x (underside height + 6 mm)); the camera sits h_frac x that height above the eye (clipped to `h`), always
    below the paddle, and looks `look` x the frame's half-height above the eye, so the fly sits a little below centre.
    In the air the look-at point follows `follow` of the hop's rise and the camera backs off by backoff[0] x the rise
    (at most backoff[1] m)."""
    eye = np.asarray(eye, float)
    rise = max(float(eye[2]) - ground_z, 0.0) if airborne else 0.0
    base = np.array([eye[0], eye[1], ground_z + cam["follow"] * rise])
    ph = max(float(paddle_underside_z) - float(eye[2]), 0.0)          # the paddle's underside above the eye
    tan = math.tan(math.radians(cam["fov"]) / 2)
    D = float(np.clip(0.6 * (ph + 0.006) / tan, *cam["d"]))
    look = min(0.4 * ph, cam["look"] * D * tan)
    D += min(cam["backoff"][0] * rise, cam["backoff"][1])
    h = float(np.clip(cam["h_frac"] * ph, *cam["h"])) + (float(eye[2]) - base[2])
    return base + np.array([0.0, 0.0, look]), math.hypot(D, h - look), math.degrees(math.atan2(h - look, D))


# ------------------------------------------------------------------------------------------------ the ray tracer
class SwatWorld(W.World):
    """`flyverse.world.World` with oriented ellipsoids (a rotation per sphere, `rots`) and the extra MATERIALS. The
    shading, shadows and textures are world.py's own; only the ellipsoid intersection is generalised (rays are rotated
    into each ellipsoid's frame), and cameras are traced in chunks to bound GPU memory on a shared card."""

    def _pack(self):
        if len(getattr(self, "rots", [])) != len(self.spheres):
            self.rots = list(getattr(self, "rots", [])) + [None] * (len(self.spheres) - len(getattr(self, "rots", [])))

        def mat(name):
            return W.Material(name, MATERIALS[name]) if name in MATERIALS else W.MATERIALS[name]
        d = self.device
        ms = [mat(s.material) for s in self.spheres] + [mat(b.material) for b in self.boxes] + [mat(p.material) for p in self.planes]
        self._refl = torch.tensor([m.refl for m in ms] + [(0, 0, 0, 0)], dtype=torch.float32, device=d)
        self._refl2 = torch.tensor([m.refl2 for m in ms] + [(0, 0, 0, 0)], dtype=torch.float32, device=d)
        self._pattern = torch.tensor([m.pattern for m in ms] + [0], dtype=torch.long, device=d)
        self._pscale = torch.tensor([m.scale for m in ms] + [1.0], dtype=torch.float32, device=d)
        self._emit = torch.tensor([m.emit for m in ms] + [(0, 0, 0, 0)], dtype=torch.float32, device=d)
        self._sc = torch.tensor([s.center for s in self.spheres] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._sr = torch.tensor([s.radii for s in self.spheres] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        rots = np.asarray([np.eye(3) if r is None else r for r in self.rots], np.float32).reshape(-1, 3, 3)
        self._srot = torch.tensor(rots, dtype=torch.float32, device=d)
        self._blo = torch.tensor([b.lo for b in self.boxes] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._bhi = torch.tensor([b.hi for b in self.boxes] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._pp = torch.tensor([p.point for p in self.planes] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        pn = torch.tensor([p.normal for p in self.planes] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._pn = pn / pn.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self._lp = torch.tensor(self.light_pos, dtype=torch.float32, device=d)
        self._lc = torch.tensor(self.light_color, dtype=torch.float32, device=d)
        self._amb = torch.tensor(self.ambient, dtype=torch.float32, device=d)
        self._noise_corners = torch.tensor(W._NOISE_CORNERS, device=d)
        self._metal = False                        # the oriented intersection below is plain torch
        self._scene_shape = (len(self.spheres), len(self.boxes), len(self.planes))
        self._trace_graphs = {}
        self._packed = True

    def move(self, idx, center, rot=None):
        """Move (and turn) sphere `idx` in place."""
        s = self.spheres[idx]
        s.center = tuple(float(v) for v in center)
        if rot is not None:
            self.rots[idx] = np.asarray(rot, float)
        if getattr(self, "_packed", False) and self._scene_shape == (len(self.spheres), len(self.boxes), len(self.planes)):
            self._sc[idx].copy_(torch.tensor(s.center, dtype=torch.float32))
            if rot is not None:
                self._srot[idx].copy_(torch.tensor(np.asarray(rot, np.float32)))

    def _intersect(self, o, d):
        M = o.shape[0]
        best_t = torch.full((M,), W.INF, device=o.device)
        best_n = torch.zeros(M, 3, device=o.device)
        best_id = torch.full((M,), self._refl.shape[0] - 1, dtype=torch.long, device=o.device)
        eps = 1e-4
        if len(self.spheres):                      # oriented ellipsoids: rotate into each frame, scale to a unit sphere
            oc = torch.einsum("msj,sjk->msk", o[:, None, :] - self._sc[None], self._srot) / self._sr[None]
            dd = torch.einsum("mj,sjk->msk", d, self._srot) / self._sr[None]
            a = (dd * dd).sum(-1); b = 2 * (oc * dd).sum(-1); c = (oc * oc).sum(-1) - 1
            disc = b * b - 4 * a * c
            ok = disc > 0
            sq = torch.sqrt(disc.clamp_min(0))
            t0 = (-b - sq) / (2 * a); t1 = (-b + sq) / (2 * a)
            t = torch.where(t0 > eps, t0, t1)
            t = torch.where(ok & (t > eps), t, torch.full_like(t, W.INF))
            tmin, si = t.min(dim=1)
            hit = tmin < best_t
            g = si[:, None, None].expand(M, 1, 3)
            pl = oc.gather(1, g)[:, 0] + dd.gather(1, g)[:, 0] * torch.where(hit, tmin, torch.zeros_like(tmin))[:, None]
            n = torch.einsum("mk,mjk->mj", pl / self._sr[si], self._srot[si])
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            best_n = torch.where(hit[:, None], n, best_n); best_t = torch.where(hit, tmin, best_t)
            best_id = torch.where(hit, si, best_id)
        if len(self.boxes):                        # (world.py's own box and plane code)
            inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
            t_lo = (self._blo[None] - o[:, None, :]) * inv[:, None, :]
            t_hi = (self._bhi[None] - o[:, None, :]) * inv[:, None, :]
            tn = torch.minimum(t_lo, t_hi); tf = torch.maximum(t_lo, t_hi)
            t_enter, ax = tn.max(dim=-1); t_exit = tf.min(dim=-1).values
            t = torch.where((t_exit > t_enter) & (t_enter > eps), t_enter, torch.full_like(t_enter, W.INF))
            tmin, bi = t.min(dim=1)
            hit = tmin < best_t
            axis = ax.gather(1, bi[:, None])[:, 0]
            sign = -torch.sign(d.gather(1, axis[:, None])[:, 0])
            n = torch.zeros(M, 3, device=o.device); n.scatter_(1, axis[:, None], sign[:, None])
            best_n = torch.where(hit[:, None], n, best_n); best_t = torch.where(hit, tmin, best_t)
            best_id = torch.where(hit, bi + len(self.spheres), best_id)
        if len(self.planes):
            denom = (d[:, None, :] * self._pn[None]).sum(-1)
            t = ((self._pp[None] - o[:, None, :]) * self._pn[None]).sum(-1) / torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
            t = torch.where((t > eps) & (denom.abs() > 1e-9), t, torch.full_like(t, W.INF))
            tmin, pi = t.min(dim=1)
            hit = tmin < best_t
            n = self._pn[pi]
            n = torch.where((n * d).sum(-1, keepdim=True) > 0, -n, n)
            best_n = torch.where(hit[:, None], n, best_n); best_t = torch.where(hit, tmin, best_t)
            best_id = torch.where(hit, pi + len(self.spheres) + len(self.boxes), best_id)
        return best_t, best_n, best_id

    def render(self, cam: Camera, chunk=262144):
        """(H, W, 4) radiance through `cam`."""
        d = cam.rays(self.device)
        o = torch.tensor(cam.pos, dtype=torch.float32, device=self.device)[None]
        out = [self.trace(o.expand(min(chunk, d.shape[0] - s), 3), d[s:s + chunk]) for s in range(0, d.shape[0], chunk)]
        return torch.cat(out).reshape(cam.h, cam.w, 4)


def make_world(base, extra_spheres=(), device=None) -> SwatWorld:
    """A SwatWorld with copies of `base`'s objects plus `extra_spheres` [(centre, radii, rotation, material)]."""
    w = SwatWorld(spheres=[W.Sphere(tuple(s.center), tuple(s.radii), s.material) for s in base.spheres],
                  boxes=[W.Box(tuple(b.lo), tuple(b.hi), b.material) for b in base.boxes],
                  planes=[W.Plane(tuple(p.point), tuple(p.normal), p.material) for p in base.planes],
                  light_pos=base.light_pos, light_color=base.light_color, ambient=base.ambient, detail=base.detail,
                  device=device if device is not None else base.device)
    w.rots = [None] * len(w.spheres)
    for c, r, rot, mat in extra_spheres:
        w.spheres.append(W.Sphere(tuple(float(v) for v in c), tuple(float(v) for v in r), mat))
        w.rots.append(None if rot is None else np.asarray(rot, float))
    return w


# ------------------------------------------------------------------------------------------------ the swatter
class Swatter:
    """The swatter's state machine (GAME): aim -> swing -> [pin] -> lift -> hold -> move -> aim. Kinematics in world
    metres; it reads the fly's eye position and airborne flag, nothing from the brain."""

    def __init__(self, u, anchor, handle_az):
        self.u = unit(u)
        self.anchor = np.asarray(anchor, float).copy()
        self.handle_az = float(handle_az)
        self.dist = START_DIST
        self.phase, self.t_phase = "aim", 0.0
        self.v = 0.0
        self.u_from, self.u_to = self.u.copy(), self.u.copy()
        self.took_off = False                      # the fly took off during this swing: its spot is frozen

    @property
    def rot(self):
        return paddle_frame(self.handle_az)

    @property
    def center(self):
        return self.anchor + self.u * self.dist

    @property
    def d_contact(self):
        return contact_distance(self.u, self.rot)

    def time_to_contact(self):
        """Seconds until the swinging paddle reaches the spot it is aimed at (None outside a swing)."""
        return (self.dist - self.d_contact) / self.v if self.phase == "swing" and self.v > 0 else None

    def touches(self, eye):
        """The eye is within CONTACT_GAP of the paddle (inside the paddle grown by 4 mm on every semi-axis)."""
        return inflated_metric(eye, self.center, self.rot, PADDLE_RADII, CONTACT_GAP) <= 1.0

    def set_phase(self, name):
        self.phase, self.t_phase = name, 0.0

    def follow_handle(self, dt, tau=0.3):
        """The handle points to where the swing comes from (kept while the swing is near vertical)."""
        if math.hypot(self.u[0], self.u[1]) > 0.3:
            a = 1.0 - math.exp(-dt / tau)
            self.handle_az = wrap_angle(self.handle_az + a * wrap_angle(math.atan2(self.u[1], self.u[0]) - self.handle_az))

    def start_swing(self, v, eye):
        self.v = float(v)
        self.anchor = np.asarray(eye, float).copy()
        self.dist = START_DIST
        self.took_off = False
        self.set_phase("swing")

    def start_move(self, u_to):
        self.u_from, self.u_to = self.u.copy(), unit(u_to)
        self.set_phase("move")

    def step(self, dt, eye, airborne):
        """Advance one tick. In a swing, returns 'contact' on the tick the paddle reaches the spot it is aimed at (the
        standing fly's eye, tracked while it stands, frozen where it took off), or a standing fly's eye comes within
        4 mm of it. A fly in the air is not tested here: `referee_step` tests it with `touches` after its flight step,
        at the same instant."""
        self.t_phase += dt
        eye = np.asarray(eye, float)
        if self.phase == "swing":
            if not airborne and not self.took_off:
                self.anchor = eye.copy()            # the swing tracks the fly while it stands on the table
            self.dist = max(self.dist - self.v * dt, self.d_contact)
            if self.dist <= self.d_contact + 1e-9 or (not airborne and self.touches(eye)):
                return "contact"
            return None
        if self.phase == "pin":
            if self.t_phase >= PIN_S:
                self.set_phase("lift")
            return None
        if self.phase == "lift":
            self.dist = min(self.dist + LIFT_SPEED * dt, START_DIST)
            if self.dist >= START_DIST:
                self.set_phase("hold")
            return None
        self.anchor = ema(self.anchor, eye, dt, ANCHOR_TAU_S)      # hold / move / aim: follow the fly with a lag
        if self.phase == "move":
            self.u = slerp(self.u_from, self.u_to, smoothstep(self.t_phase / MOVE_S))
            if self.t_phase >= MOVE_S:
                self.u = self.u_to.copy()
                self.set_phase("aim")
        self.follow_handle(dt)
        return None


# ------------------------------------------------------------------------------------------------ the referee
def ticks_until_arrival(remaining_m, v, dt, tol=1e-9):
    """Whole ticks until a paddle `remaining_m` short of the spot it is aimed at, moving v*dt per tick, gets there:
    the tick on which Swatter.step reports it (same tolerance). 0 = it is there now."""
    if remaining_m <= tol:
        return 0
    return max(1, int(math.ceil((remaining_m - tol) / (v * dt))))


def swat_outcome(ticks_to_arrival):
    """PLAN.md 1 ('escaped' there), on one clock: IN TIME when the body took off on a tick strictly before the tick on
    which the paddle reaches the fly's spot, i.e. `ticks_to_arrival` >= 1 at the takeoff tick; the margin is that many
    whole 10 ms ticks. No takeoff (None), or one on the arrival tick (0), is a SPLAT. Returns (outcome, margin_ms)."""
    if ticks_to_arrival is None or ticks_to_arrival < 1:
        return "splat", None
    return "in_time", int(round(ticks_to_arrival * common.TICK_MS))


def swept_touch(rel0, rel1, rot, radii=PADDLE_RADII, gap=CONTACT_GAP):
    """True if an eye moving, relative to the paddle, in a straight line from `rel0` to `rel1` (eye minus paddle centre
    at the start and the end of a tick) comes within `gap` of the paddle at any instant of the tick. A paddle at
    1.5-2.5 m/s and a hop closing on each other cover 20-30 mm per tick, more than the paddle's 16 mm shell, so a test
    at the tick's end alone would let the hop pass through it."""
    k = np.asarray(radii) + gap
    p0 = (np.asarray(rot).T @ np.asarray(rel0, float)) / k
    p1 = (np.asarray(rot).T @ np.asarray(rel1, float)) / k
    if p0 @ p0 <= 1.0 or p1 @ p1 <= 1.0:
        return True
    d = p1 - p0
    a, b, c = float(d @ d), float(2 * p0 @ d), float(p0 @ p0 - 1.0)
    disc = b * b - 4 * a * c
    if a < 1e-18 or disc < 0:
        return False
    t = (-b - math.sqrt(disc)) / (2 * a)
    return 0.0 <= t <= 1.0


def touch_dist(e, u, rot, radii=PADDLE_RADII, gap=CONTACT_GAP):
    """The paddle's distance along `u` from the anchor at which the point `e` (relative to the anchor) lies on the
    paddle's 4 mm shell on its anchor side: where a swing that met the fly in the air is shown to stop. None if no
    position along `u` touches it."""
    k = np.asarray(radii) + gap
    qe = (np.asarray(rot).T @ np.asarray(e, float)) / k
    qu = (np.asarray(rot).T @ np.asarray(u, float)) / k
    a, b, c = float(qu @ qu), float(-2 * qe @ qu), float(qe @ qe - 1.0)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    return (-b + math.sqrt(disc)) / (2 * a)


def referee_step(s, dt, eye, airborne, body):
    """One tick of the swatter and the body on one clock, after the brain has stepped to the tick's time. The order
    is the rule, and SwatGame.tick uses exactly this function:
      1. the paddle moves to the tick's time and is tested against the fly where it stands (for a fly already in the
         air: only whether the paddle has reached the spot it took off from);
      2. `body(blocked)` runs the shipped body for the tick and returns (took_off, eye, airborne) after it. `blocked`
         is True when the paddle reached the standing fly in step 1: the body cannot also take off on that tick (a
         tie goes to the paddle);
      3. a takeoff during the swing reads the paddle at this same tick: the whole ticks it still needs to reach the
         spot (the margin) and the exact remaining time (distance / speed);
      4. a fly in the air is tested against the paddle over this same tick (swept: eye and paddle both move). If they
         met, that is the contact (even on the tick the paddle also reached the spot), and the paddle is shown where
         it touches the eye's 4 mm shell rather than where it would have gone.
    Returns {'arrived': None | 'table' | 'takeoff spot' | 'landed', 'took': bool, 'ticks_to_arrival': int | None,
    'ttc_ms': float | None, 'air': bool}."""
    out = {"arrived": None, "took": False, "ticks_to_arrival": None, "ttc_ms": None, "air": False}
    swinging = s.phase == "swing"
    eye0, c0 = np.asarray(eye, float).copy(), s.center.copy()
    if s.step(dt, eye, airborne) == "contact":
        if not s.took_off:
            out["arrived"] = "table"
        else:                                          # it jumped in time; did it land where the paddle came down?
            out["arrived"] = "landed" if (not airborne and s.touches(eye)) else "takeoff spot"
    took, eye, airborne = body(out["arrived"] == "table")
    out["took"] = bool(took)
    if swinging and out["arrived"] != "table":
        if took:
            rem = max(s.dist - s.d_contact, 0.0)
            out["ticks_to_arrival"] = ticks_until_arrival(rem, s.v, dt)
            out["ttc_ms"] = rem / s.v * 1000.0
            s.took_off = True
        eye = np.asarray(eye, float)
        if airborne and swept_touch(eye0 - c0, eye - s.center, s.rot):
            out["air"], out["arrived"] = True, None
            back = touch_dist(eye - s.anchor, s.u, s.rot)
            if back is not None:
                s.dist = max(s.dist, back)
    return out


# ------------------------------------------------------------------------------------------------ the game
class Banner:
    def __init__(self, t, text, subs, color, hold=0.8, fade=0.8):
        self.t, self.text, self.color, self.hold, self.fade = t, text, color, hold, fade
        self.subs = [subs] if isinstance(subs, str) else list(subs)

    def alpha(self, now):
        age = now - self.t
        if age < 0:
            return 0.0
        return 1.0 if age < self.hold else max(0.0, 1.0 - (age - self.hold) / self.fade)


class SwatGame(common.Game):
    title = "swat"
    subtitle = "a fly swatter vs the giant fibre  ·  raw model, zero decoders"
    MAIN = (0, 64, 1384, 968)             # x, y, w, h of the main view on the 1920 x 1080 canvas (65 %)
    PIP = (464, 290)                      # the picture-in-picture, bottom right of the main view
    SIDE_X = 1392

    def __init__(self, args):
        super().__init__(args)
        from flyverse import air, body
        self.fb = common.build_brain(args)
        self.eyes = common.Eyes(self.fb)
        self.device = self.fb.device
        room, self.info = W.make_room(args.seed, args.fruit)
        self.top = self.info["table_top_z"]
        x0, x1, y0, y1 = self.info["table_extent"]
        self.walk_bounds = (x0 + 0.02, x1 - 0.02, y0 + 0.02, y1 - 0.02)          # room_demo --fence
        self.fly_bounds = (x0, x1, y0, y1, 2.6)
        self.loco, self.flight, self.metabolism = body.Locomotion(), body.Flight(), body.Metabolism()
        sx, sy, sh = args.start
        self.fly = body.FlyState(x=sx, y=sy, z=self.top, heading=math.radians(sh))
        self.air = air.Air([(n, c, r / 0.02, r) for n, c, r in self.info["fruit"]],
                           air.WindParams(speed=0.3, direction_deg=180.0), seed=args.seed)
        self.script = list(SCHEDULE)
        self.auto = args.record is not None or args.auto
        self.k = 0                                            # ticks of brain time; t_s = k * 10 ms
        self.n_swing = 0                                      # every swing (scheduled or the player's)
        self.n_sched = self.first = max(0, min(args.from_swat, len(SCHEDULE)) - 1)   # the schedule's next row
        self.n_player = 0
        eye0 = self.fly.eye_pos
        self.swatter = Swatter(approach_dir(self.fly.heading, *SCHEDULE[self.first][1:3]), eye0,
                               self.fly.heading + math.radians(150))
        hc, hr = handle_pose(self.swatter.center, self.swatter.handle_az)
        sw = [(self.swatter.center, PADDLE_RADII, self.swatter.rot, "black"), (hc, HANDLE_RADII, hr, "black")]
        self.n_room = len(room.spheres)
        self.eye_world = make_world(room, [] if args.control else sw, device=self.device)
        self.cam_world = make_world(room, sw + fly_parts(self.fly.pos, self.fly.forward, self.fly.left, self.fly.up),
                                    device=self.device)
        # read-only views of named cells for the HUD (CONNECTOME): population means of the loom detectors
        self.idx = {k: torch.as_tensor(self.fb.c.select(type=k), device=self.device) for k in ("LC4", "LPLC2")}
        self.n_cells = {k: int(len(v)) for k, v in self.idx.items()}
        self.hist_len = 300                                   # 3 s at 100 samples / s
        self.hist = {k: [] for k in ("gf", "lc4", "lplc2", "ttm")}
        self.marks, self.banners, self.results = [], [], []
        az0 = math.degrees(self.fly.heading) + 60
        self.cam = CameraRig(target=eye0 + 0.2 * Z, dist=0.8, az=az0, el=16.0, fov=50.0)          # WIDE
        self.mcam = CameraRig(target=eye0.copy(), dist=HERO_CAM["d"][1], az=az0, el=10.0, fov=HERO_CAM["fov"])  # CLOSE-UP
        self.pcam = CameraRig(target=eye0.copy(), dist=FLYCAM_D, az=az0, el=FLYCAM_EL, fov=FLYCAM_FOV)  # PIP fly-cam
        self.rcam = None
        self.main_close = False
        self.user_az = self.user_el = None
        self.user_zoom = 1.0
        self.mouse = self.mouse_aim = self.press_wall = self.queued_v = self.dragging = None
        self.score = {"swats": 0, "in_time": 0, "splat": 0}
        self.cur = None                                       # the current swat's record
        self.pinned = False
        self.splats = []                                      # (t, table point, n): the display-only splat marks
        self.gait = 0.0
        self.min_fruit_clearance, self._in_fruit = float("inf"), False
        self.gf_hz = self.ttm_hz = self.lc4_hz = self.lplc2_hz = 0.0
        self.last_rad = None
        self.free_gf = []                                     # GF on the ground between swings (the baseline)
        self.replay_buf, self.replay, self.replay_at = [], None, None
        self.end_card_t = None
        self.render_scale = args.render_scale or (1.0 if args.record else 0.5)
        self._declare()
        save = self.log.save

        def save_with_summary(path, fbs=None):
            self.log.summary = self.summary()
            return save(path, fbs)
        self.log.save = save_with_summary

    # -------------------------------------------------------------------------------------------- provenance
    def _declare(self):
        log, a = self.log, self.args
        log.meta["decoders"] = "none: no ReadDecoder is attached and no brain quantity is mapped to a game control"
        log.meta["body"] = ("flyverse/body.py as shipped: Locomotion.readout / step walks the fly (fenced table, as "
                            "room_demo --fence); Flight.maybe_takeoff launches the escape jump when the giant fibre "
                            "(DNp01 mean rate, 100 ms rate estimate) >= Flight.gf_hz = %.0f Hz and the fly has stood "
                            ">= %.1f s since its last landing (0.6 m/s at 45 deg, forward), or a voluntary takeoff (0.2 m/s "
                            "at 30 deg) when the wing power MNs hold >= %.0f Hz for %.1f s; Flight.step flies the hop (no "
                            "sustained wing power in the raw model, so it is ballistic). Each takeoff is logged with the "
                            "rule that fired" % (self.flight.gf_hz, self.flight.landing_refractory_s,
                                                 self.flight.takeoff_power_hz, self.flight.takeoff_hold_s))
        log.meta["scene"] = ("scripts/room_demo.py's room: world.make_room(seed, fruit_set=%r), wind 0.3 m/s from 180 "
                             "deg, fruit odour plumes (air.Air), sugar taste on fruit contact; start (%.2f m, %.2f m, "
                             "heading %.0f deg)" % ((a.fruit,) + tuple(a.start)))
        common.declare(log, "swatter", "game",
                       "a black paddle (ellipsoid, face down, long axis towards the handle) plus a handle hovers %.2f m "
                       "from the fly's eye, then swings in a straight line at v towards the eye, tracking it while the "
                       "fly stands and frozen where it took off. The swing ends on the tick the paddle reaches that "
                       "spot (the eye there %.0f mm from the paddle), or when the eye of a fly in the air comes within "
                       "%.0f mm of the paddle; the swatter then lifts" % (START_DIST, CONTACT_GAP * 1000, CONTACT_GAP * 1000),
                       "the fly's eye position and airborne flag (body state)",
                       paddle_radii_m=PADDLE_RADII, handle_radii_m=HANDLE_RADII, handle_tilt_deg=HANDLE_TILT_DEG,
                       start_dist_m=START_DIST, contact_gap_m=CONTACT_GAP, material="black",
                       visible_to_fly=not a.control)
        common.declare(log, "outcome", "game",
                       "IN TIME (PLAN.md's 'escaped') if the body model took off (Flight.maybe_takeoff: GF >= 33 Hz, or "
                       "its voluntary wing-power rule) on a 10 ms tick before the tick on which the paddle reached the "
                       "fly's spot; SPLAT otherwise. One clock (referee_step): each tick the brain steps, the paddle "
                       "moves to that tick's time and is tested against the fly where it stands, then the body acts, so "
                       "a takeoff on the arrival tick is a SPLAT (a tie goes to the paddle). Margin = whole ticks from "
                       "the takeoff tick to the arrival tick; a splat's lateness = whole ticks from contact to the first "
                       "tick with GF >= 33 Hz (0 = the same tick). The hop is not scored: where the paddle meets it in "
                       "the air is logged (contact_where)", "the body's airborne flag and takeoff tick",
                       tick_ms=common.TICK_MS)
        common.declare(log, "pin and respawn", "game",
                       "after a SPLAT the body is held where it is (no walking, no takeoff) for %.1f s under the paddle, "
                       "then the swatter lifts at %.1f m/s and the fly carries on from the same spot (respawn in place; "
                       "the brain is never reset)" % (PIN_S, LIFT_SPEED), "", pin_s=PIN_S, lift_m_s=LIFT_SPEED)
        common.declare(log, "timing", "game",
                       "no swing in the first %.1f s; a swing starts after >= %.1f s of aiming and with the fly on the "
                       "table for >= %.1f s (the body's own landing refractory is %.1f s) and no replay pending; between "
                       "swings the swatter swings round to the next direction at %.2f m in %.1f s, following the fly "
                       "with a %.2f s lag" % (WARMUP_S, AIM_S, SETTLE_S, self.flight.landing_refractory_s, START_DIST,
                                              MOVE_S, ANCHOR_TAU_S),
                       "the fly's eye position, airborne flag and time since landing", warmup_s=WARMUP_S, aim_s=AIM_S,
                       settle_s=SETTLE_S, move_s=MOVE_S, anchor_tau_s=ANCHOR_TAU_S)
        if self.auto:
            common.declare(log, "schedule", "game", "the scripted sequence of swats, the same for every seed", "",
                           swats=[{"label": s[0], "az_deg": s[1], "el_deg": s[2], "v_m_s": s[3]} for s in SCHEDULE],
                           starts_at_swat=self.first + 1)
        if not a.record:
            common.declare(log, "player", "game", "a human aims (mouse ray, >= 20 deg up) and winds up the speed "
                           "(%.1f -> %.1f m/s over %.1f s of wall-clock time held); a release before the fly has "
                           "settled is queued until it has" % (V_MIN, V_MAX, WINDUP_S), "")
        common.declare(log, "camera, fly model, replay", "game",
                       "the main view is a human camera; the fly's body is drawn there only (ellipsoids, wings folded, "
                       "legs swung in a display-only tripod gait, one cycle per 0.93 mm walked; a splat mark drawn on "
                       "the table after a SPLAT): the fly's eyes see the room and the swatter, never their own body. "
                       "Close calls (margin <= %.0f ms, or a splat whose GF crossed 33 Hz <= %.0f ms after contact) are "
                       "replayed at 0.25x (%d to +%d ms around the arrival tick, from close-ups stored during the run at "
                       "%.1fx resolution, with the side column as it was) %.1f s after the arrival tick; the brain runs on "
                       "meanwhile. A recording ends %.1f s after the end card appears"
                       % (REPLAY_MARGIN_MS, REPLAY_MARGIN_MS, -REPLAY_PRE_S * 1000, REPLAY_POST_S * 1000, REPLAY_SCALE,
                          REPLAY_DELAY_S, END_CARD_S), "", render_scale=self.render_scale)
        if a.control:
            common.declare(log, "control", "game", "CONTROL RUN: the swatter is removed from the fly's world (the "
                           "camera still draws it); everything else is identical", "")

    def brains(self):
        return {"brain": self.fb}

    # -------------------------------------------------------------------------------------------- the world
    def _shade(self, dirs):
        d = torch.as_tensor(dirs, dtype=torch.float32, device=self.device)
        o = torch.as_tensor(self.fly.eye_pos, dtype=torch.float32, device=self.device)[None].expand_as(d)
        return self.eye_world.trace(o, d)

    def _place_swatter(self, worlds):
        s = self.swatter
        hc, hr = handle_pose(s.center, s.handle_az)
        for w in worlds:
            w.move(self.n_room, s.center, s.rot); w.move(self.n_room + 1, hc, hr)

    def _sync_camera_world(self):
        self._place_swatter([self.cam_world])
        f = self.fly
        gait = None if f.airborne or f.speed < 5e-4 else self.gait
        for j, (c, _, rot, _) in enumerate(fly_parts(f.pos, f.forward, f.left, f.up, gait)):
            self.cam_world.move(self.n_room + 2 + j, c, rot)

    def _next_direction(self):
        if self.auto and self.n_sched < len(self.script):
            _, az, el, _ = self.script[self.n_sched]
            return approach_dir(self.fly.heading, az, el)
        return self.swatter.u

    def _settle_left(self):
        """Seconds until a swing may start (0 = now), or None while the fly is in the air or a replay is pending."""
        if self.fly.airborne or self.replay is not None or self.replay_at is not None:
            return None
        return max(WARMUP_S - self.t_s, SETTLE_S - self.fly.ground_time, 0.0)

    def _settled(self):
        return self._settle_left() == 0.0

    def _swing(self, v, label):
        s = self.swatter
        self.n_swing += 1
        s.start_swing(v, self.fly.eye_pos)
        el = math.degrees(math.asin(float(np.clip(s.u[2], -1, 1))))
        rel_az = math.degrees(wrap_angle(math.atan2(s.u[1], s.u[0]) - self.fly.heading)) if el < 89.5 else 0.0
        rem = START_DIST - s.d_contact
        k_arr = self.k + ticks_until_arrival(rem, v, TICK_S)   # fixed now: the paddle's speed and path length are
        self.cur = {"n": self.n_swing, "label": label, "v_m_s": round(v, 3), "el_deg": round(el, 1),
                    "az_deg": round(rel_az, 1), "k_start": self.k, "t_start": round(self.t_s, 3), "k_arrival": k_arr,
                    "t_arrival": round(k_arr * TICK_S, 3), "t_nominal_arrival": round(self.t_s + rem / v, 4),
                    "gf_peak": 0.0, "lc4_peak": 0.0, "lplc2_peak": 0.0, "ttm_peak": 0.0, "t_gf_cross": None,
                    "k_takeoff": None, "t_takeoff": None, "takeoff_rule": None, "margin_ms": None,
                    "paddle_ttc_at_takeoff_ms": None, "outcome": None, "k_contact": None, "t_contact": None,
                    "contact_where": None, "air_contact_after_takeoff_ms": None, "k_late_gf": None, "t_late_gf": None,
                    "late_gf_ms": None, "replay": None}
        self.marks.append((self.t_s, "swing"))
        self.replay_buf = []
        # the replay's close-up camera: side-on to the fly, on the side away from the swing's lateral component
        right = -np.array([-math.sin(self.fly.heading), math.cos(self.fly.heading), 0.0])
        side = 90.0 if float(s.u @ right) > 0.3 else -90.0
        self.rcam = CameraRig(target=self.fly.eye_pos.copy(), dist=REPLAY_CAM["d"][1], az=math.degrees(self.fly.heading) + side,
                              el=9.0, fov=REPLAY_CAM["fov"])
        self._replay_director(0.0, snap=True)
        self.log.event(self.t_s, "swing", n=self.n_swing, label=label, v_m_s=round(v, 3), el_deg=round(el, 1),
                       az_deg=round(rel_az, 1), fly_pos=np.round(self.fly.pos, 4),
                       heading_deg=round(math.degrees(self.fly.heading), 1), arrival_t=self.cur["t_arrival"])

    def _no_replay(self):
        self.cur["replay"] = False
        self.replay_buf = []

    def _takeoff_in_swing(self, ref, kind):
        cur = self.cur
        n = ref["ticks_to_arrival"]
        outcome, margin = swat_outcome(n)
        cur.update(k_takeoff=self.k, t_takeoff=round(self.t_s, 3), takeoff_rule=kind, outcome=outcome, margin_ms=margin,
                   paddle_ttc_at_takeoff_ms=round(ref["ttc_ms"], 1))
        if self.k + n != cur["k_arrival"]:                  # the two ways of counting must agree (see the tests)
            self.log.event(self.t_s, "clock_mismatch", n=cur["n"], k=self.k, ticks=n, k_arrival=cur["k_arrival"])
        if margin is not None and margin <= REPLAY_MARGIN_MS:
            cur["replay"] = True
            self.replay_at = cur["t_arrival"] + REPLAY_DELAY_S
        else:
            self._no_replay()
        return margin

    def _contact(self, where):
        s, cur = self.swatter, self.cur
        if cur["outcome"] is None:                          # no takeoff before the paddle arrived
            cur["outcome"] = "splat"
        outcome = cur["outcome"]
        cur.update(k_contact=self.k, t_contact=round(self.t_s, 3), contact_where=where)
        self.score["swats"] += 1
        self.score[outcome] += 1
        self.marks.append((self.t_s, "contact"))
        if outcome == "in_time":
            m = cur["margin_ms"]
            if where == "air":
                cur["air_contact_after_takeoff_ms"] = int(round((self.k - cur["k_takeoff"]) * common.TICK_MS))
                what = "then its fixed 45° hop met the paddle in the air"
            elif where == "landed":
                what = "then it landed where the paddle came down"
            else:
                what = "and the paddle reached the empty spot"
            self.banners.append(Banner(self.t_s, "JUMPED IN TIME  +%d ms" % m,
                                       ["took off %d ms before the paddle reached its spot" % m, what], SAGE))
            s.set_phase("lift")
        else:
            if cur["t_gf_cross"] is not None:
                sub = ["the giant fibre crossed 33 Hz on this same 10 ms tick", "a tie goes to the paddle"]
            else:
                sub = ["the giant fibre had reached only %.0f Hz by contact" % cur["gf_peak"]]
            self.banners.append(Banner(self.t_s, "SPLAT", sub, RED))
            s.set_phase("pin")
            self.pinned = True
            self.splats.append((self.t_s, np.array([self.fly.x, self.fly.y, self.top]), cur["n"]))
        self.log.event(self.t_s, "contact", n=cur["n"], outcome=outcome, where=where, margin_ms=cur["margin_ms"],
                       air_contact_after_takeoff_ms=cur["air_contact_after_takeoff_ms"],
                       gf_peak_hz=round(cur["gf_peak"], 1), gf_cross_t=cur["t_gf_cross"], takeoff_t=cur["t_takeoff"],
                       lc4_peak_hz=round(cur["lc4_peak"], 2), lplc2_peak_hz=round(cur["lplc2_peak"], 2),
                       ttm_peak_hz=round(cur["ttm_peak"], 1), eye=np.round(self.fly.eye_pos, 4))
        self.results.append(cur)

    # -------------------------------------------------------------------------------------------- tick
    def _body(self, dt, w, cmd, tasting, frozen):
        """The body as shipped for one tick -- unless `frozen` (pinned under the paddle: GAME). Returns (took, rule)."""
        f = self.fly
        if frozen:
            f.ground_time += dt
            return False, None
        if f.airborne:
            self.flight.step(f, w, dt, lambda x, y: self.top, self.fly_bounds)
            if not f.airborne:
                self.log.event(self.t_s, "landing", pos=np.round(f.pos, 4))
            return False, None
        if self.flight.maybe_takeoff(f, w, dt):             # which of the body's two rules fired (checked in its order)
            return True, ("escape" if w["gf"] >= w["gf_threshold"] and f.ground_time >= self.flight.landing_refractory_s
                          else "voluntary")
        if self.metabolism.update(bool(tasting), f.speed, dt):   # feeding: a fly that feeds stops walking
            cmd = dict(cmd, speed=0.0, yaw=0.0)
        self.loco.step(f, cmd, dt, self.walk_bounds)
        self.gait = (self.gait + abs(f.speed) * dt / STRIDE_M) % 1.0      # display only: the model's legs
        return False, None

    def tick(self):
        dt = TICK_S
        f, fb, s = self.fly, self.fb, self.swatter
        # senses, as room_demo's Sim.step, from the scene as it stands: the mosaic shows exactly this radiance
        rad = self.eyes.radiance(self._shade, f.forward, f.left, f.up)
        fb.vision(rad)
        self.last_rad = rad
        dist = min(float(np.linalg.norm(f.pos - np.asarray(cc))) - r for _, cc, r in self.info["fruit"])
        tasting = 1.0 if (dist < 0.015 and not f.airborne) else 0.0
        clear = min(float(np.linalg.norm(f.eye_pos - np.asarray(cc))) - r for _, cc, r in self.info["fruit"])
        self.min_fruit_clearance = min(self.min_fruit_clearance, clear)
        if clear < 0 and not self._in_fruit:                  # a scene artefact: the walk does not avoid fruit
            self.log.event(self.t_s, "eye_inside_fruit", pos=np.round(f.pos, 4))
        self._in_fruit = clear < 0
        self.air.step(dt)
        fb.smell(*self.air.antennae(f.eye_pos, f.left, f.forward))
        fb.wind(*self.air.deflections(f.forward, f.left))
        fb.taste(tasting)
        fb.step(common.TICK_MS)
        self.k += 1
        self.t_s = self.k * dt
        motor = fb.motor()
        cmd = self.loco.readout(motor, dt_s=dt)
        w = self.flight.readout(motor)
        # connectome readings for the HUD, at this tick
        r = fb.brain.rate[0]
        self.gf_hz, self.ttm_hz = float(motor.gf), float(motor.ttm)
        self.lc4_hz = float(r[self.idx["LC4"]].mean()); self.lplc2_hz = float(r[self.idx["LPLC2"]].mean())
        for key, v in (("gf", self.gf_hz), ("lc4", self.lc4_hz), ("lplc2", self.lplc2_hz), ("ttm", self.ttm_hz)):
            self.hist[key].append(v)
            if len(self.hist[key]) > self.hist_len:
                del self.hist[key][0]
        cur = self.cur
        in_swing = s.phase == "swing"                        # the swing was under way during this tick
        if in_swing:
            for key, v in (("gf_peak", self.gf_hz), ("ttm_peak", self.ttm_hz), ("lc4_peak", self.lc4_hz),
                           ("lplc2_peak", self.lplc2_hz)):
                cur[key] = max(cur[key], v)
            if cur["t_gf_cross"] is None and self.gf_hz >= self.flight.gf_hz:
                cur["t_gf_cross"] = round(self.t_s, 3)
        elif s.phase in ("aim", "hold", "move") and not f.airborne and f.ground_time > 0.5:
            self.free_gf.append(self.gf_hz)
        # the swatter (GAME) and the body (shipped) on one clock: see referee_step
        if not self.auto and s.phase == "aim" and self.mouse_aim is not None:
            s.u = slerp(s.u, self.mouse_aim, 0.15)
        rule = [None]

        def body(blocked):
            if self.pinned and s.phase not in ("pin", "swing"):
                self.pinned = False                          # the pin is over: the body is free on this tick
            took, rule[0] = self._body(dt, w, cmd, tasting, blocked or self.pinned)
            return took, f.eye_pos, f.airborne
        ref = referee_step(s, dt, f.eye_pos, f.airborne, body)
        if ref["arrived"]:
            self._contact(ref["arrived"])
        if ref["took"]:
            self._takeoff(ref, rule[0], w, in_swing)
        if ref["air"]:
            self._contact("air")
        # a splatted fly's giant fibre, after contact
        if self.pinned and cur and cur["k_late_gf"] is None and self.gf_hz >= self.flight.gf_hz:
            late = int(round((self.k - cur["k_contact"]) * common.TICK_MS))
            cur.update(k_late_gf=self.k, t_late_gf=round(self.t_s, 3), late_gf_ms=late)
            self.log.event(self.t_s, "late_gf", n=cur["n"], gf_hz=round(self.gf_hz, 1), after_contact_ms=late)
            if late > 0:
                self.banners.append(Banner(self.t_s, "GF %.0f Hz · %d ms too late" % (self.gf_hz, late),
                                           "the giant fibre crossed 33 Hz after contact", AMBER, hold=0.6, fade=0.6))
            if late <= REPLAY_MARGIN_MS and cur["replay"] is None:
                cur["replay"] = True
                self.replay_at = cur["t_arrival"] + REPLAY_DELAY_S
        if (cur and cur["outcome"] == "splat" and cur["replay"] is None
                and self.t_s > cur["t_arrival"] + REPLAY_MARGIN_MS / 1000 + 1e-6):
            self._no_replay()                                # no late crossing within the replay margin
        # the schedule (GAME), and a player's queued swing
        if s.phase == "hold" and not (self.auto and self.n_sched >= len(self.script)):
            s.start_move(self._next_direction())
        if self.auto and s.phase == "aim" and s.t_phase >= AIM_S and self._settled() and self.n_sched < len(self.script):
            label, _, _, v = self.script[self.n_sched]
            self.n_sched += 1
            self._swing(v, label)
        elif not self.auto and self.queued_v is not None and s.phase == "aim" and self._settled():
            self.n_player += 1
            self._swing(self.queued_v, "player")
            self.queued_v = None
        self._place_swatter([] if self.args.control else [self.eye_world])
        self._director(dt)
        # replay: store every tick's close-up around the arrival tick; start the replay once the window has passed
        if self._capturing():
            self._capture()
        if self.replay_at is not None and self.t_s >= self.replay_at - 1e-9:
            self._start_replay()
        elif self.replay is not None:
            self.replay["i"] += 1.0 / REPLAY_SLOW
            if self.replay["i"] >= len(self.replay["frames"]):
                self.replay = None
        if (self.end_card_t is None and self.auto and self.n_sched >= len(self.script) and s.phase in ("hold", "aim", "move")
                and self.replay is None and self.replay_at is None and self.results):
            self.end_card_t = self.t_s
            self.log.event(self.t_s, "end_card", in_time=self.score["in_time"], splat=self.score["splat"])
        self.marks = [(t, kind) for t, kind in self.marks if t > self.t_s - 3.2]

    def _takeoff(self, ref, kind, w, in_swing):
        s = self.swatter
        self.marks.append((self.t_s, "takeoff"))
        info = dict(rule=kind, gf_hz=round(self.gf_hz, 1), power_hz=round(float(w["power"]), 1))
        if kind == "escape":
            sub = ("body model: GF ≥ 33 Hz → escape jump" if in_swing else
                   "after the splat: the body jumps once the pin is released" if s.phase == "lift" else
                   "no swing in progress")
            self.banners.append(Banner(self.t_s, "GIANT FIBRE %.0f Hz → TAKEOFF" % self.gf_hz, sub, TEAL,
                                       hold=0.5, fade=0.6))
        else:
            self.banners.append(Banner(self.t_s, "WING POWER %.0f Hz → TAKEOFF" % w["power"],
                                       "body model: power MNs ≥ 50 Hz for 0.3 s (voluntary takeoff)", LILAC,
                                       hold=0.5, fade=0.6))
        if ref["ticks_to_arrival"] is not None:
            margin = self._takeoff_in_swing(ref, kind)
            self.log.event(self.t_s, "takeoff", n=self.cur["n"], during="swing", margin_ms=margin,
                           ticks_before_arrival=ref["ticks_to_arrival"], paddle_ttc_ms=round(ref["ttc_ms"], 1), **info)
        else:
            self.log.event(self.t_s, "takeoff", during=s.phase, **info)

    # -------------------------------------------------------------------------------------------- replay
    def _capturing(self):
        cur = self.cur
        if cur is None or self.replay is not None or cur["replay"] is False:
            return False
        t_a = cur["t_arrival"]
        return t_a - REPLAY_PRE_S - 0.02 <= self.t_s <= t_a + REPLAY_POST_S + 1e-6

    def _capture(self):
        """One tick of the replay window: the profile close-up, the WIDE view of the same instant, and the side column."""
        f = self.fly
        self._replay_director(TICK_S)
        pw, ph = self.PIP
        self.replay_buf.append({"t": round(self.t_s, 3), "img": self._render(self.rcam, scale=REPLAY_SCALE),
                                "wide": self._render(self.cam, self.PIP, scale=REPLAY_WIDE_SCALE),
                                "ring": self.cam.camera(pw, ph).project(f.eye_pos), "airborne": bool(f.airborne),
                                "side": self._side_state(colors=True)})

    def _start_replay(self):
        cur = self.cur
        t_a = cur["t_arrival"]
        frames = [fr for fr in self.replay_buf if t_a - REPLAY_PRE_S - 1e-6 <= fr["t"] <= t_a + REPLAY_POST_S + 1e-6]
        self.replay_at, self.replay_buf = None, []
        if frames:
            self.replay = {"frames": frames, "i": 0.0, "n": cur["n"], "t_arrival": t_a, "t_takeoff": cur["t_takeoff"],
                           "t_contact": cur["t_contact"], "where": cur["contact_where"], "outcome": cur["outcome"],
                           "margin": cur["margin_ms"], "late": cur["late_gf_ms"]}
            self.log.event(self.t_s, "replay", n=cur["n"], frames=len(frames), speed=1.0 / REPLAY_SLOW,
                           window_ms=[round(-REPLAY_PRE_S * 1000), round(REPLAY_POST_S * 1000)])

    # -------------------------------------------------------------------------------------------- camera
    def _side_az(self, avoid_az_deg, prefer_deg):
        """Of the two azimuths square to `avoid_az_deg` (the paddle's long axis / the handle), the one nearer `prefer`."""
        return min((avoid_az_deg + 90.0, avoid_az_deg - 90.0),
                   key=lambda a: abs(math.degrees(wrap_angle(math.radians(a - prefer_deg)))))

    def _hero(self, cam):
        s, f = self.swatter, self.fly
        return hero_goal(f.eye_pos, f.airborne, self.top + f.eye_height, s.center[2] - PADDLE_RADII[2], cam)

    def _director(self, dt):
        """Three live cameras, all square to the paddle's long axis on the fly's front side. WIDE frames the swatter and
        the fly (pushing in during the swing); CLOSE-UP is a low 30 deg lens 2-3 cm from the fly that ducks under the
        paddle; the fly-cam is a tight shot of the fly for the picture-in-picture. The main view cuts to the close-up
        when the swinging paddle is within CUT_DIST or CUT_TTC_S of the fly or the fly takes off, and back to wide once
        the swatter has lifted away and the fly is on the table; the picture-in-picture shows the fly-cam under WIDE,
        and WIDE under the close-up."""
        s, f = self.swatter, self.fly
        eye, P = f.eye_pos, s.center
        sep = float(np.linalg.norm(P - eye))
        if self.auto:
            az = self._side_az(math.degrees(s.handle_az), math.degrees(f.heading) + 60.0)
        else:                                  # a player aims through the camera: it must not follow the swatter
            az = self.cam.az if self.user_az is None else self.user_az
        ttc = s.time_to_contact()
        if s.phase == "swing" and (sep <= CUT_DIST or s.took_off or (ttc is not None and ttc <= CUT_TTC_S)):
            self.main_close = True
        elif s.phase in ("hold", "move", "aim") and not f.airborne:
            self.main_close = False
        tgt, dist, el = self._hero(HERO_CAM)
        self.mcam.update(tgt, dist, az, el, dt, tau_target=0.01 if f.airborne else 0.04, tau_dist=0.03, tau_angle=0.6,
                         tau_el=0.01)                      # it must duck as fast as the paddle comes down
        self.pcam.update(eye + np.array([0.0, 0.0, 0.0003]), FLYCAM_D, az, FLYCAM_EL, dt, tau_target=0.02,
                         tau_dist=0.1, tau_angle=0.6)
        w_tgt = eye + 0.45 * (P - eye)
        w_tgt[2] = max(w_tgt[2], self.top + 0.01)
        w_dist = float(np.clip(1.9 * sep + 0.05, 0.12, 1.0))
        self.cam.update(w_tgt, w_dist * self.user_zoom, az if self.user_az is None else self.user_az,
                        16.0 if self.user_el is None else self.user_el, dt,
                        tau_target=0.06 if s.phase == "swing" else 0.2, tau_dist=0.06 if s.phase == "swing" else 0.3)

    def _replay_director(self, dt, snap=False):
        """The replay's close-up (rendered every tick of the replay window): the fly in profile, telephoto."""
        tgt, dist, el = self._hero(REPLAY_CAM)
        r = self.rcam
        if snap:
            r.target, r.dist, r.el = tgt, dist, el
        else:
            r.update(tgt, dist, r.az, el, dt, tau_target=0.02, tau_dist=0.02, tau_angle=0.05, tau_el=0.01)

    # -------------------------------------------------------------------------------------------- input
    def handle(self, event, canvas_pos=None):
        import pygame
        if event.type == pygame.MOUSEMOTION and canvas_pos is not None:
            self.mouse = canvas_pos
            if self.dragging is not None:
                (x0, y0), az0, el0 = self.dragging
                self.user_az = az0 - (canvas_pos[0] - x0) * 0.3
                self.user_el = float(np.clip(el0 + (canvas_pos[1] - y0) * 0.2, 2.0, 85.0))
            self._aim_from_mouse()
        elif event.type == pygame.MOUSEBUTTONDOWN and canvas_pos is not None:
            if event.button == 1 and not self.auto:
                self.press_wall = time.monotonic()
            elif event.button == 3:
                self.dragging = (canvas_pos, self.cam.az, self.cam.el)
            elif event.button in (4, 5):
                self.user_zoom = float(np.clip(self.user_zoom * (0.9 if event.button == 4 else 1.1), 0.3, 3.0))
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.press_wall is not None:
                self.queued_v = windup_speed(time.monotonic() - self.press_wall)
                self.press_wall = None
                self.log.event(self.t_s, "player_release", v_m_s=round(self.queued_v, 3))
            elif event.button == 3:
                self.dragging = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                self.auto = not self.auto
                self.queued_v = self.press_wall = None
                if self.auto and self.swatter.phase == "aim":           # swing round to the scheduled direction first
                    self.swatter.start_move(self._next_direction())
                self.log.event(self.t_s, "auto", on=self.auto, next_scheduled_swat=self.n_sched + 1)
            elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.user_az = (self.cam.az if self.user_az is None else self.user_az) + (15 if event.key == pygame.K_LEFT else -15)
            elif event.key in (pygame.K_UP, pygame.K_DOWN):
                base = self.cam.el if self.user_el is None else self.user_el
                self.user_el = float(np.clip(base + (5 if event.key == pygame.K_UP else -5), 2, 85))
            elif event.key == pygame.K_HOME:
                self.user_az = self.user_el = None
                self.user_zoom = 1.0

    def _aim_from_mouse(self):
        if self.mouse is None or self.auto:
            return
        mx, my, mw, mh = self.MAIN
        x, y = self.mouse
        if not (mx <= x < mx + mw and my <= y < my + mh):
            return
        cam = self.cam.camera(mw, mh)
        p = aim_point(cam.pos, cam.ray_through(x - mx, y - my), self.fly.eye_pos, START_DIST)
        self.mouse_aim = clamp_elevation(p - self.fly.eye_pos, 20.0, 90.0)

    def finished(self):
        return (self.args.record is not None and self.end_card_t is not None
                and self.t_s - self.end_card_t >= END_CARD_S - 1e-9)

    # -------------------------------------------------------------------------------------------- drawing
    def _render(self, rig, size=None, scale=None):
        """The scene through `rig` -> a pygame surface of `size` (default: the main view's)."""
        import pygame
        self._sync_camera_world()
        mw, mh = size or self.MAIN[2:]
        k = self.render_scale * (1.0 if scale is None else scale)
        cam = rig.camera(max(8, int(mw * k)), max(8, int(mh * k)))
        rgb = W.to_rgb8(self.cam_world.render(cam), exposure=2.2)
        surf = pygame.image.frombuffer(np.ascontiguousarray(rgb).tobytes(), (cam.w, cam.h), "RGB")
        return pygame.transform.smoothscale(surf, (mw, mh)) if (cam.w, cam.h) != (mw, mh) else surf.copy()

    def draw(self, surface):
        hud = self.hud
        surface.fill(BG)
        mx, my, mw, mh = self.MAIN
        if self.replay is not None:
            fr = self._draw_replay(surface)
            side = fr["side"]
        else:
            main, other = (self.mcam, self.cam) if self.main_close else (self.cam, self.pcam)
            surface.blit(self._render(main), (mx, my))
            self._overlays(surface, main.camera(mw, mh), other)
            side = None
        hud.header(surface, self.title, self.subtitle, common.model_line(self.fb))
        self._side(surface, side)
        hud.footer(surface, ["decoders: none",
                             "takeoff = body model: GF (DNp01) ≥ 33 Hz → Flight.maybe_takeoff",
                             "IN TIME = takeoff tick before the paddle's arrival tick",
                             "swatter, schedule, pin: GAME"])

    def _box(self, surface, rect, alpha=170):
        import pygame
        box = pygame.Surface(rect[2:], pygame.SRCALPHA)
        box.fill((9, 12, 14, alpha))
        surface.blit(box, rect[:2])

    def _draw_replay(self, surface):
        import pygame
        hud = self.hud
        mx, my, mw, mh = self.MAIN
        rp = self.replay
        fr = rp["frames"][min(int(rp["i"]), len(rp["frames"]) - 1)]
        t = fr["t"]
        surface.blit(fr["img"], (mx, my))
        pw, ph = self.PIP                                    # the wide view of the same moment
        px, py = mx + mw - pw - 16, my + mh - 92 - ph
        surface.blit(fr["wide"], (px, py))
        pygame.draw.rect(surface, LINE_HI, (px - 2, py - 2, pw + 4, ph + 4), 2)
        ring = fr["ring"]
        if ring is not None and 0 <= ring[0] < pw and 0 <= ring[1] < ph:
            pygame.draw.circle(surface, WHITE, (int(px + ring[0]), int(py + ring[1])), 11, 2)
        self._box(surface, (px, py, 64, 26), 170)
        hud.text(surface, "WIDE", (px + 8, py + 4), 16, WHITE, bold=True)
        self._box(surface, (mx, my + mh - 76, mw, 76), 205)
        hud.text(surface, "REPLAY  0.25×", (mx + 24, my + mh - 64), 44, AMBER, bold=True, display=True)
        rel = int(round((t - rp["t_arrival"]) * 1000))
        when = ("paddle arrives in %d ms" % -rel) if rel < 0 else ("paddle arrived %d ms ago" % rel if rel > 0
                                                                   else "paddle arrives")
        x = hud.text(surface, "swat %d  ·  %s" % (rp["n"], when), (mx + 330, my + mh - 54), 26,
                     WHITE if rel < 0 else MUTED).right
        x = hud.chip(surface, "game", (x + 10, my + mh - 50), 12).right + 26
        gf = fr["side"]["gf"]
        x = hud.text(surface, "GF %2.0f Hz" % gf, (x, my + mh - 54), 26, RED if gf >= self.flight.gf_hz else SAGE,
                     bold=True).right
        hud.chip(surface, "connectome", (x + 10, my + mh - 50), 12)
        if fr["airborne"]:
            hud.text(surface, "AIRBORNE", (mx + mw - 28, my + mh - 62), 40, SAGE, bold=True, display=True, anchor="topright")
        if rp["outcome"] == "in_time" and rp["t_takeoff"] is not None and t >= rp["t_takeoff"] - 1e-6:
            self._box(surface, (mx + 20, my + 20, 760, 128), 190)
            hud.text(surface, "JUMPED IN TIME  +%d ms" % rp["margin"], (mx + 40, my + 28), 60, SAGE, bold=True,
                     display=True)
            if rp["where"] == "air" and t >= rp["t_contact"] - 1e-6:
                hud.text(surface, "the hop met the paddle in the air (not scored)", (mx + 40, my + 106), 24, TEXT)
            else:
                hud.text(surface, "took off %d ms before the paddle reached its spot" % rp["margin"], (mx + 40, my + 106),
                         24, TEXT)
        elif rp["outcome"] == "splat" and t >= rp["t_arrival"] - 1e-6:
            self._box(surface, (mx + 20, my + 20, 760, 128), 190)
            hud.text(surface, "SPLAT", (mx + 40, my + 28), 60, RED, bold=True, display=True)
            late = rp["late"]
            if late is not None:
                hud.text(surface, "GF crossed 33 Hz on the contact tick" if late == 0 else
                         "GF crossed 33 Hz %d ms after contact" % late, (mx + 40, my + 106), 24, TEXT)
        pygame.draw.rect(surface, AMBER, (mx, my, mw, mh), 4)
        return fr

    def _draw_splats(self, surface, cam):
        """A splat mark lying on the table where a fly was swatted (display only): solid while pinned, then fading."""
        import pygame
        mx, my, mw, mh = self.MAIN
        self.splats = [sp for sp in self.splats if self.t_s - sp[0] < PIN_S + 1.8]
        for t0, p, n in self.splats:
            age = self.t_s - t0
            a = 1.0 if age < PIN_S else max(0.0, 1.0 - (age - PIN_S) / 1.8)
            q = cam.project(p)
            if q is None or a <= 0:
                continue
            r = cam.pixels_per_metre(p) * 0.0042
            if r < 4:
                continue
            r = min(r, 110.0)                                    # a pool round the feet, not over the close-up
            squash = max(abs(float((cam.pos - p) @ Z)) / max(float(np.linalg.norm(cam.pos - p)), 1e-9), 0.22)
            rng = np.random.default_rng(1000 + n)
            size = int(4 * r) + 8
            layer = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size / 2
            ang = np.linspace(0, 2 * np.pi, 22, endpoint=False)
            rad = r * (0.55 + 0.45 * rng.random(22))
            pygame.draw.polygon(layer, (118, 24, 20, 215), [(c + rr * math.cos(t), c + rr * math.sin(t)) for rr, t in zip(rad, ang)])
            for _ in range(9):                                   # droplets
                t, d = rng.random() * 2 * np.pi, r * (1.05 + 0.8 * rng.random())
                pygame.draw.circle(layer, (118, 24, 20, 215), (c + d * math.cos(t), c + d * math.sin(t)), max(2, r * 0.09 * (1 + rng.random())))
            layer = pygame.transform.smoothscale(layer, (size, max(4, int(size * squash))))
            layer.set_alpha(int(170 * a))
            surface.blit(layer, layer.get_rect(center=(mx + q[0], my + q[1])))

    def _swat_box(self, surface):
        """Top left: the current swat (GAME)."""
        hud = self.hud
        mx, my, mw, mh = self.MAIN
        s, cur = self.swatter, self.cur
        self._box(surface, (mx + 16, my + 16, 640, 96))
        sub2, col2 = None, TEXT
        if cur is not None and s.phase in ("swing", "pin", "lift"):
            head = "SWAT %d  ·  %s" % (cur["n"], cur["label"].upper())
            sub = "%.1f m/s  ·  %.0f° up" % (cur["v_m_s"], cur["el_deg"])
            if s.phase == "swing" and not s.took_off:
                sub2 = "arrives in %3.0f ms" % max(s.time_to_contact() * 1000, 0)
            elif cur["margin_ms"] is not None:
                sub2, col2 = "took off %d ms early" % cur["margin_ms"], SAGE
        elif self.auto:
            nxt = self.script[self.n_sched] if self.n_sched < len(self.script) else None
            if nxt:
                head, sub = "AIMING  ·  %s" % nxt[0].upper(), "next swing %.1f m/s" % nxt[3]
            else:
                head, sub = "SEQUENCE OVER", ""
        else:
            head = "AIMING"
            if self.press_wall is not None:
                sub = "winding up: %.2f m/s" % windup_speed(time.monotonic() - self.press_wall)
            elif self.queued_v is not None:
                left = self._settle_left()
                sub = "%.1f m/s queued" % self.queued_v
                sub2 = ("fly in the air" if left is None and self.fly.airborne else "replay first" if left is None
                        else "fly settling %.1f s" % left if left > 0 else "swatter swinging round")
            else:
                sub = "hold the left button, release to swing"
        t = hud.text(surface, head, (mx + 32, my + 26), 32, TEXT, bold=True, display=True)
        hud.chip(surface, "game", (t.right + 14, my + 34), 13)
        hud.text(surface, sub, (mx + 32, my + 70), 24, AMBER)
        if sub2:
            hud.text(surface, sub2, (mx + 380, my + 70), 24, col2)

    def _overlays(self, surface, cam, pip_rig):
        import pygame
        hud = self.hud
        mx, my, mw, mh = self.MAIN
        f = self.fly
        self._draw_splats(surface, cam)
        p = cam.project(f.eye_pos)                                   # mark the fly when it is small on screen
        fly_y = None if p is None else my + p[1]
        if p is not None and cam.pixels_per_metre(f.eye_pos) * 0.0026 < 30:
            x, y = mx + p[0], my + p[1]
            if mx + 30 <= x < mx + mw - 30 and my + 30 <= y < my + mh - 30:
                pygame.draw.circle(surface, WHITE, (int(x), int(y)), 24, 2)
                hud.text(surface, "fly", (x + 26, y - 30), 20, WHITE, bold=True)
        self._swat_box(surface)
        # top right: the score (GAME)
        sc = self.score
        self._box(surface, (mx + mw - 416, my + 16, 400, 96))
        swats = ("%d/%d" % (sc["swats"] + self.first, len(self.script)) if self.auto and self.n_player == 0
                 else str(sc["swats"]))
        for x, label, val, col in ((mx + mw - 396, "IN TIME", str(sc["in_time"]), SAGE),
                                   (mx + mw - 262, "SPLAT", str(sc["splat"]), RED), (mx + mw - 140, "SWATS", swats, TEXT)):
            hud.text(surface, label, (x, my + 24), 18, MUTED, bold=True)
            hud.text(surface, val, (x, my + 44), 46, col, bold=True, display=True)
        hud.chip(surface, "game", (mx + mw - 72, my + 22), 12)
        # bottom: what decides, and the clock
        self._box(surface, (mx + 16, my + mh - 62, 700, 46))
        hud.text(surface, "takeoff = body model (flyverse/body.py): GF ≥ 33 Hz", (mx + 32, my + mh - 52), 22, TEXT)
        self._box(surface, (mx + mw - 300, my + mh - 62, 284, 46))
        hud.text(surface, "brain t = %6.2f s" % self.t_s, (mx + mw - 32, my + mh - 52), 22, MUTED, anchor="topright")
        # the picture in picture: the fly-cam under WIDE, WIDE (with the fly ringed) under the close-up
        pw, ph = self.PIP
        px, py = mx + mw - pw - 16, my + mh - 74 - ph
        surface.blit(self._render(pip_rig, (pw, ph)), (px, py))
        pygame.draw.rect(surface, LINE_HI, (px - 2, py - 2, pw + 4, ph + 4), 2)
        q = pip_rig.camera(pw, ph).project(f.eye_pos)
        if pip_rig is self.cam and q is not None and 0 <= q[0] < pw and 0 <= q[1] < ph:
            pygame.draw.circle(surface, WHITE, (int(px + q[0]), int(py + q[1])), 11, 2)
        close = pip_rig is not self.cam
        self._box(surface, (px, py, 118 if close else 64, 26), 170)
        hud.text(surface, "CLOSE-UP" if close else "WIDE", (px + 8, py + 4), 16, WHITE, bold=True)
        if self.args.control:
            hud.text(surface, "CONTROL RUN: THE FLY CANNOT SEE THE SWATTER", (mx + mw // 2, my + 124), 28, AMBER,
                     bold=True, anchor="midtop")
        self._draw_banners(surface, fly_y)
        if (self.end_card_t is not None and self.swatter.phase in ("hold", "aim", "move") and self.replay is None
                and self.replay_at is None):
            self._end_card(surface)

    def _end_card(self, surface):
        """After the last swat: the tally, from the run's own results."""
        import pygame
        hud = self.hud
        mx, my, mw, mh = self.MAIN
        res = self.results
        it = [r for r in res if r["outcome"] == "in_time"]
        sp = [r for r in res if r["outcome"] == "splat"]
        w, h = 960, 300
        x, y = mx + 16, my + 170
        self._box(surface, (x, y, w, h), 225)
        pygame.draw.rect(surface, LINE_HI, (x, y, w, h), 2, border_radius=6)
        hud.text(surface, "%d JUMPED IN TIME" % len(it), (x + 40, y + 22), 60, SAGE, bold=True, display=True)
        hud.text(surface, "%d SPLAT" % len(sp), (x + w - 40, y + 22), 60, RED, bold=True, display=True, anchor="topright")
        lines = []
        if it:
            fast = max(it, key=lambda r: (r["v_m_s"], r["margin_ms"]))
            lines.append("fastest in time: %.1f m/s %s, %d ms to spare" % (fast["v_m_s"], fast["label"], fast["margin_ms"]))
        late = [r["late_gf_ms"] for r in sp if r["late_gf_ms"] is not None]
        if late:
            span = "%d" % late[0] if min(late) == max(late) else "%d-%d" % (min(late), max(late))
            lines.append("splats: GF crossed 33 Hz %s ms after contact (%d of %d)" % (span, len(late), len(sp)))
        if it:
            air = sum(r["contact_where"] == "air" for r in it)
            lines.append("in time, then the hop met the paddle in the air: %d of %d" % (air, len(it)))
        yy = y + 110
        for line in lines:
            hud.text(surface, line, (x + 40, yy), 22, TEXT)
            yy += 36
        hud.text(surface, "no decoder · body model: GF ≥ 33 Hz → takeoff · the hop is not scored", (x + 40, y + h - 38),
                 20, MUTED)
        hud.chip(surface, "game", (x + w - 70, y + h - 36), 12)

    def _banner_line(self, text, color, size=56):
        """Render a banner line in the display font; '→' and '·' (absent from it) are drawn as vector glyphs."""
        import pygame
        font = self.hud.font(size, True, True)
        parts, glyphs, cur = [], [], ""
        for ch in text:
            if ch in "→·":
                parts.append(cur.strip()); glyphs.append(ch); cur = ""
            else:
                cur += ch
        parts.append(cur.strip())
        imgs = [font.render(p, True, color) if p else None for p in parts]
        gaps = [int(size * (1.1 if g == "→" else 0.6)) for g in glyphs]
        w = sum(i.get_width() for i in imgs if i is not None) + sum(gaps)
        h = max(i.get_height() for i in imgs if i is not None)
        out = pygame.Surface((max(w, 1), h), pygame.SRCALPHA)
        x = 0
        for i, img in enumerate(imgs):
            if img is not None:
                out.blit(img, (x, 0))
                x += img.get_width()
            if i < len(glyphs):
                cy, gap = h * 0.52, gaps[i]
                if glyphs[i] == "→":
                    a0, a1, t = x + gap * 0.18, x + gap * 0.82, size * 0.09
                    pygame.draw.polygon(out, color, [(a0, cy - t), (a1 - size * 0.28, cy - t), (a1 - size * 0.28, cy - size * 0.24),
                                                     (a1, cy), (a1 - size * 0.28, cy + size * 0.24), (a1 - size * 0.28, cy + t),
                                                     (a0, cy + t)])
                else:
                    pygame.draw.circle(out, color, (x + gap / 2, cy), max(3, size * 0.075))
                x += gap
        return out

    def _draw_banners(self, surface, fly_y=None):
        """The event banners: at the bottom of the main view (left of the picture-in-picture), stacking up -- or, when
        the fly is in the lower part of the view, at the top, below the info boxes, stacking down."""
        import pygame
        mx, my, mw, mh = self.MAIN
        self.banners = [b for b in self.banners if self.t_s - b.t < b.hold + b.fade + 0.1]
        live = [b for b in self.banners if b.alpha(self.t_s) > 0][-2:][::-1]
        top = fly_y is not None and fly_y > my + 0.62 * mh
        y = my + (170 if self.args.control else 128) if top else my + mh - 76
        cx = mx + mw // 2 if top else mx + (mw - self.PIP[0] - 16) // 2
        for b in live:
            big = self._banner_line(b.text, b.color)
            subs = [self.hud.font(24).render(s, True, TEXT) for s in b.subs if s]
            w = max([big.get_width()] + [s.get_width() for s in subs]) + 64
            h = big.get_height() + sum(s.get_height() + 2 for s in subs) + 22
            panel = pygame.Surface((w, h), pygame.SRCALPHA)
            panel.fill((9, 12, 14, 205))
            pygame.draw.rect(panel, b.color + (255,), (0, 0, w, h), 3, border_radius=6)
            panel.blit(big, big.get_rect(midtop=(w // 2, 6)))
            yy = big.get_height() + 10
            for s in subs:
                panel.blit(s, s.get_rect(midtop=(w // 2, yy)))
                yy += s.get_height() + 2
            room = mw - 64 if top else mw - self.PIP[0] - 48
            if w > room:                                     # never wider than the space it has
                panel = pygame.transform.smoothscale(panel, (room, max(1, int(h * room / w))))
                h = panel.get_height()
            panel.set_alpha(int(255 * b.alpha(self.t_s)))
            if top:
                surface.blit(panel, panel.get_rect(midtop=(cx, y)))
                y += h + 10
            else:
                surface.blit(panel, panel.get_rect(midbottom=(cx, y)))
                y -= h + 10

    def _side_state(self, colors=False):
        """What the side column shows at this tick; `colors` also stores the fly's-eye mosaic (for a replay)."""
        return {"t": self.t_s, "gf": self.gf_hz, "lc4": self.lc4_hz, "lplc2": self.lplc2_hz, "ttm": self.ttm_hz,
                "hist": {k: list(v) for k, v in self.hist.items()}, "marks": list(self.marks),
                "colors": (common.eye_colors(self.last_rad, "human", 2.5) if colors and self.last_rad is not None else None)}

    def _side(self, surface, snap=None):
        """The side column: live, or (in a replay) as it was at the replayed instant."""
        import pygame
        hud = self.hud
        replay = snap is not None
        if snap is None:
            snap = {"t": self.t_s, "gf": self.gf_hz, "lc4": self.lc4_hz, "lplc2": self.lplc2_hz, "ttm": self.ttm_hz,
                    "hist": self.hist, "marks": self.marks,
                    "colors": common.eye_colors(self.last_rad, "human", 2.5) if self.last_rad is not None else None}
        x, w = self.SIDE_X, 1920 - self.SIDE_X - 8
        r = hud.panel(surface, (x, 72, w, 318), "what the fly sees")
        if replay:
            hud.text(surface, "REPLAY: t = %.2f s" % snap["t"], (x + w - 12, 80), 16, AMBER, bold=True, anchor="topright")
        else:
            hud.text(surface, "exact radiance -> fb.vision", (x + w - 12, 82), 14, DIM, anchor="topright")
        if snap["colors"] is not None:
            hud.mosaic(surface, (r.x, r.y, r.w, r.h), self.eyes, snap["colors"])
        y = 398
        r = hud.panel(surface, (x, y, w, 282), "giant fibre DNp01", "connectome")
        col = RED if snap["gf"] >= self.flight.gf_hz else SAGE
        hud.text(surface, "%.0f" % snap["gf"], (r.x + 2, r.y - 4), 58, col, bold=True, display=True)
        hud.text(surface, "Hz    33 Hz = takeoff (body model)", (r.x + 112, r.y + 28), 17, MUTED)
        tr = pygame.Rect(r.x, r.y + 70, r.w, r.h - 94)
        hud.trace(surface, tr, self._padded(snap["hist"]["gf"]), 0, GF_VMAX, SAGE, threshold=self.flight.gf_hz)
        self._marks(surface, tr, snap["marks"], snap["t"])
        hud.text(surface, "33", (tr.x + 4, tr.bottom - self.flight.gf_hz / GF_VMAX * tr.h - 18), 14, RED)
        lx = r.x
        for label, colr in (("last 3 s:", DIM), ("| swing", AMBER), ("| takeoff", SAGE), ("| contact", WHITE)):
            lx = hud.text(surface, label, (lx, tr.bottom + 5), 14, colr).right + 10
        y += 290
        r = hud.panel(surface, (x, y, w, 226), "loom detectors  (mean rate)", "connectome")
        half = (r.h - 8) // 2
        for j, (key, name, vmax, colr) in enumerate((("lplc2", "LPLC2", LPLC2_VMAX, TEAL), ("lc4", "LC4", LC4_VMAX, LILAC))):
            ry = r.y + j * (half + 8)
            hud.text(surface, "%s  (%d cells)" % (name, self.n_cells[name]), (r.x, ry), 16, TEXT)
            hud.text(surface, "%.1f Hz" % snap[key], (r.right, ry - 5), 24, colr, bold=True, anchor="topright")
            hud.trace(surface, (r.x, ry + 24, r.w, half - 26), self._padded(snap["hist"][key]), 0, vmax, colr)
        y += 234
        r = hud.panel(surface, (x, y, w, 1032 - 8 - y), "TTMn  jump-muscle motor neurons", "connectome")
        t = hud.text(surface, "%.0f" % snap["ttm"], (r.x + 2, r.y - 10), 40, AMBER, bold=True, display=True)
        hud.text(surface, "Hz mean", (t.right + 8, r.y + 8), 17, MUTED)
        track = pygame.Rect(r.x + 190, r.y + 8, r.w - 190, 18)
        pygame.draw.rect(surface, common.INSET, track, border_radius=3)
        frac = float(np.clip(snap["ttm"] / 60.0, 0, 1))
        if frac > 0:
            pygame.draw.rect(surface, AMBER, (track.x, track.y, max(2, round(track.w * frac)), track.h), border_radius=3)
        if replay:                                           # the whole column is the replayed instant
            pygame.draw.rect(surface, AMBER, (x - 4, 68, w + 8, 1032 - 4 - 68), 3, border_radius=8)

    def _padded(self, h):
        return [0.0] * (self.hist_len - len(h)) + list(h)

    def _marks(self, surface, rect, marks, t_now):
        import pygame
        span = self.hist_len * common.TICK_MS / 1000
        colors = {"swing": AMBER, "takeoff": SAGE, "contact": WHITE}
        for t, kind in marks:
            fx = 1 - (t_now - t) / span
            if 0 <= fx <= 1:
                xx = rect.x + fx * (rect.w - 1)
                pygame.draw.line(surface, colors[kind], (xx, rect.y + 2), (xx, rect.bottom - 2), 2)

    # -------------------------------------------------------------------------------------------- the log
    def summary(self):
        res = self.results
        free = np.asarray(self.free_gf) if self.free_gf else np.zeros(1)
        keys = ("n", "label", "v_m_s", "el_deg", "az_deg", "outcome", "margin_ms", "paddle_ttc_at_takeoff_ms",
                "t_start", "t_arrival", "t_nominal_arrival", "t_gf_cross", "t_takeoff", "takeoff_rule", "t_contact",
                "contact_where", "air_contact_after_takeoff_ms", "t_late_gf", "late_gf_ms", "gf_peak", "lc4_peak",
                "lplc2_peak", "ttm_peak", "replay")
        it = [r for r in res if r["outcome"] == "in_time"]
        sp = [r for r in res if r["outcome"] == "splat"]
        return {"rule": "IN TIME (PLAN.md's 'escaped') = the body took off on a 10 ms tick before the tick on which the "
                        "paddle reached the fly's spot; margins and lateness in whole ticks; the hop is not scored",
                "swats": len(res), "in_time": len(it), "splat": len(sp),
                "in_time_margins_ms": [r["margin_ms"] for r in it],
                "in_time_hop_met_paddle_in_air": sum(r["contact_where"] == "air" for r in it),
                "in_time_paddle_reached_empty_spot": sum(r["contact_where"] == "takeoff spot" for r in it),
                "splat_late_gf_ms": [r["late_gf_ms"] for r in sp],
                "per_swat": [{k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k)) for k in keys} for r in res],
                "player_swings": self.n_player,
                "gf_on_ground_between_swings_hz": {"p50": round(float(np.percentile(free, 50)), 2),
                                                   "p99": round(float(np.percentile(free, 99)), 2),
                                                   "max": round(float(free.max()), 2), "n_ticks": len(self.free_gf)},
                "takeoffs_after_splat_pin": sum(1 for e in self.log.events if e["kind"] == "takeoff" and e.get("during") in ("pin", "lift")),
                "takeoffs_unprompted": {rule: sum(1 for e in self.log.events if e["kind"] == "takeoff"
                                                  and e.get("during") in ("aim", "hold", "move") and e.get("rule") == rule)
                                        for rule in ("escape", "voluntary")},
                "end_card_t_s": None if self.end_card_t is None else round(self.end_card_t, 2),
                "final_fly_pos": np.round(self.fly.pos, 4),
                "min_eye_fruit_clearance_m": round(self.min_fruit_clearance, 4),
                "eye_inside_fruit_events": sum(1 for e in self.log.events if e["kind"] == "eye_inside_fruit")}


def parse_args(argv=None):
    ap = common.standard_args(__doc__.splitlines()[0], seconds=40.0)
    ap.add_argument("--control", action="store_true", help="control run: the swatter is invisible to the fly")
    ap.add_argument("--auto", action="store_true", help="interactive window running the scripted sequence")
    ap.add_argument("--start", type=float, nargs=3, default=(0.5, 0.0, 180.0), metavar=("X", "Y", "HEADING_DEG"),
                    help="the fly's start on the table (m, m, deg); default (0.5, 0, 180): walking away from the apple, "
                         "which the hops' leftward heading drift then keeps behind it")
    ap.add_argument("--fruit", choices=("apple", "all"), default="apple",
                    help="room_demo's fruit set: 'apple' (one apple; default) or 'all'. The fenced walk does not treat "
                         "fruit as obstacles, so with 'all' the hopping fly can end up inside one")
    ap.add_argument("--from-swat", type=int, default=1, metavar="N",
                    help="dev: start the scripted sequence at swat N (logged; recordings use 1)")
    ap.add_argument("--render-scale", type=float, default=None,
                    help="camera render resolution factor (default 1.0 when recording, 0.5 interactive); the camera "
                         "never feeds the brain")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return common.run(SwatGame(args), args)


if __name__ == "__main__":
    main()
