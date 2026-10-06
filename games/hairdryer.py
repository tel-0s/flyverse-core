"""hairdryer -- steer the fly to the apple with a fan (games/PLAN.md section 6).

The room from scripts/room_demo.py: the table with its checked cloth, one apple, and an invisible fence 2 cm inside
the table edge (room_demo --fence). A hairdryer, held by a scripted holder (record mode) or by the mouse
(interactive), blows at the fly. The hairdryer is in the ray-traced scene, so the fly sees it too.

    wind at the fly     GAME: blows from the nozzle to the fly, speed = 0.55 m/s x 0.20 m / d (a round jet's 1/d
                        centreline decay), capped at 1.0 m/s; the apple's odour plume follows it (flyverse.air.Air)
    Johnston's organ    shipped sense: antennal deflection per side -> JO-C / JO-E -> the wind descending neurons
    wind DNs            CONNECTOME: for lateral winds DNp18 fires on the side the wind comes from, DNp33 on the other
                        side; the frontal code is left-biased (the index's null is ~15 deg right of dead ahead)
    turn                DECODER (declared, attached read-only): yaw = k * EMA(u), u = 1/2 [(DNp18 L - R) - (DNp33 L - R)]
                        of the brain's rate_hz (its 100 ms running rate estimate), capped at 60 deg/s (faster
                        self-turns next to the apple fire the giant fibre), turning the fly towards the side the index
                        reports: upwind for lateral winds. The raw model does not turn on its own (DNa02 is held below
                        threshold); this decoder is the game's, added to the shipped body readout's yaw.
    walk                shipped body readout (flyverse/body.py Locomotion): speed from the forward DNs, ~0.9 cm/s
    contact -> taste    GAME (room_demo's rule): within 1.5 cm of the apple's surface -> fb.taste(1): the sweet GRNs
                        fire at 120 Hz (shipped sense) -> MN9, the proboscis motor neuron (CONNECTOME)
    feeding             GAME: while it touches the apple the fly stands still (room_demo holds a feeding fly)

Why upwind: on dev seeds the wind-DN index crosses zero with a steep slope for wind from ~15 deg right of dead ahead,
but only weakly from behind (DNp33-L fires for wind from anywhere behind). A turn-into-the-wind decoder therefore has
a clean fixed point; a turn-away one would settle ~40 deg off. So the hairdryer works as a lure: hold it beyond the
apple and the fly walks up the jet. Measurements and limits: games/captions/hairdryer.md.

    python games/hairdryer.py                                   # interactive: the mouse holds the hairdryer
    python games/hairdryer.py --record out/games/hairdryer/dev.mp4 --seed 100 --seconds 32
    python games/hairdryer.py --record ... --decoder off        # control: same seed, the turn decoder disconnected

Interactive keys: move the mouse to place the hairdryer (it always points at the fly), hold the left button to blow
(SPACE latches it on / off), mouse wheel = power (x0.3-2.4, the wind still capped at 1.0 m/s), A = scripted holder
on / off, D = decoder on / off, R = restart, ESC quit.

On screen: the scripted holder (record mode) is labelled on the hairdryer and in the footer as GAME -- it sees the
fly's pose and the apple's position, which the brain does not. Every banner carries the provenance chip of what it
reports. The wind streaks are a picture of the jet (they flow round the apple; the wind model has no apple in it).
"""
from __future__ import annotations

import copy
import math
import sys
from collections import deque
from pathlib import Path
from typing import ClassVar

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # puts the repo on sys.path (flyverse imports below rely on it)
import torch

DEG = math.pi / 180.0
DT_S = common.TICK_MS / 1000.0

# ------------------------------------------------------------------------------------------------ GAME constants
START = (0.105, 0.035, -100.0)       # fly x, y (m) and heading (deg) on the table: the apple is behind it, to its left
FAN_START = (0.30, 150.0)            # hairdryer's first place: range (m) and bearing from the fly relative to its heading
FAN_ON_S = 1.5                       # record mode: the holder switches the hairdryer on at this brain time
NOZZLE_HEIGHT = 0.075                # m above the table top
JET_V_REF, JET_D_REF, JET_V_MAX = 0.55, 0.20, 1.0   # round-jet law: v = V_REF * D_REF / d, capped
CONTACT_REACH = 0.015                # m from the apple's surface: room_demo's taste rule
FLY_DRAW_SCALE = 6.0                 # the fly is drawn at 6x life size in the main view

# display only (never reaches the brain)
CAM_EL_DEG, CAM_FOV_DEG = 50.0, 40.0  # main camera elevation and vertical field of view
WIDE_SHOT_S = 3.5                    # record mode: the hairdryer stays in the framing this long after it switches on
BANNER_MIN_S = 1.2                   # a banner holds full opacity at least this long before a newer one replaces it
BANNER_HOLD_S, BANNER_FADE_S = 1.6, 0.8
HIST_LEN = 800                       # HUD history: 8 s of 10 ms ticks


def wrap(a):
    """Angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------------ pure logic
def jet_speed(d, v_ref=JET_V_REF, d_ref=JET_D_REF, v_max=JET_V_MAX):
    """GAME: wind speed (m/s) at horizontal distance d (m) from the nozzle on the jet's axis: the far-field round-jet
    law, centreline speed ~ 1/d, normalised to v_ref at d_ref and capped at v_max."""
    return float(min(v_max, v_ref * d_ref / max(float(d), 1e-3)))


def wind_towards_deg(nozzle_xy, fly_xy):
    """Direction the wind blows TOWARDS at the fly (deg, world frame, 0 = +x), i.e. from the nozzle to the fly --
    the convention of flyverse.air.WindParams.direction_deg."""
    d = np.asarray(fly_xy, float) - np.asarray(nozzle_xy, float)
    return float(np.rad2deg(np.arctan2(d[1], d[0])))


def wind_from_relative(towards_deg, heading):
    """Where the wind comes FROM relative to the fly's heading (rad, + = from the left, 0 = dead ahead)."""
    return wrap(np.deg2rad(towards_deg) + math.pi - heading)


def in_contact(fly_pos, apple_center, apple_radius, airborne, reach=CONTACT_REACH):
    """GAME (room_demo's rule): the fly touches the fruit when it is on its feet within `reach` of the surface."""
    gap = float(np.linalg.norm(np.asarray(fly_pos, float) - np.asarray(apple_center, float))) - apple_radius
    return (not airborne) and gap < reach


def turn_towards_wind(yaw_deg_s, alpha_rad, deadband_deg=3.0):
    """True when a turn (+ = left) goes towards the side the wind comes from (alpha + = from the left), False when it
    goes the other way, None when the wind is (nearly) dead ahead or dead astern and the question has no answer."""
    a = abs(alpha_rad) / DEG
    if a < deadband_deg or a > 180.0 - deadband_deg or yaw_deg_s == 0:
        return None
    return bool((yaw_deg_s > 0) == (alpha_rad > 0))            # a plain bool (numpy's would log as a string)


def turn_banner(u_hz, yaw_deg_s, alpha_rad):
    """The DECODER banner: headline, sub-line and whether the turn goes towards the wind. It names the decoder as the
    one that turns, quotes its actual input (the wind-DN index u) and says 'towards the wind' only when that is so."""
    towards = turn_towards_wind(yaw_deg_s, alpha_rad)
    head = f"DECODER: TURN {'LEFT' if yaw_deg_s > 0 else 'RIGHT'}"
    a = alpha_rad / DEG
    where = "ahead" if abs(a) < 3 else ("behind" if abs(a) > 177 else f"{abs(a):.0f} deg {'left' if a > 0 else 'right'}")
    sub = f"wind DNs L-R {u_hz:+.0f} Hz -> {abs(yaw_deg_s):.0f} deg/s" + (
        ", towards the wind" if towards else f"; wind from {where}")
    return head, sub, towards


def deflect_round_sphere(p, center, radius, margin=0.002, keep=0.35, age_after_hit=None):
    """Visual only (the wind streaks): particles (rows x, y, z, vx, vy, vz, age, ...) that have entered the sphere are
    put back on its surface and slide round it: the inward part of the velocity is removed (the tangential part is
    kept as it is, so a glancing streak flows on and a head-on one stalls). Particles left with less than `keep` of
    their speed -- the head-on ones, at the stagnation point -- are retired (age set to 1e9); the others are aged to
    at least `age_after_hit` (if given), so they fade out as they wrap round instead of splashing. In place; returns p."""
    if not len(p):
        return p
    rel = p[:, 0:3] - np.asarray(center, float)[None]
    dist = np.linalg.norm(rel, axis=1)
    hit = dist < radius + margin
    if not hit.any():
        return p
    n = rel[hit] / np.maximum(dist[hit], 1e-9)[:, None]
    p[hit, 0:3] = np.asarray(center, float)[None] + n * (radius + margin)
    v = p[hit, 3:6]
    speed = np.linalg.norm(v, axis=1)
    vt = v - np.minimum((v * n).sum(1), 0.0)[:, None] * n
    p[hit, 3:6] = vt
    slow = np.linalg.norm(vt, axis=1) < keep * speed
    idx = np.flatnonzero(hit)
    if age_after_hit is not None:
        p[idx[~slow], 6] = np.maximum(p[idx[~slow], 6], age_after_hit)
    p[idx[slow], 6] = 1e9
    return p


def trace_x(n, maxlen, x0, width):
    """x positions of the last n samples of a history of at most maxlen, on a fixed time axis: the newest sample at
    the right edge, one sample every width / (maxlen - 1) px, so the axis does not stretch while the history fills."""
    step = (width - 1) / max(maxlen - 1, 1)
    return x0 + (width - 1) - (n - 1 - np.arange(n)) * step


class BannerQueue:
    """Event banners in brain time. A new banner replaces the one on screen only once that one has been up for
    `min_s` at full opacity; until then it waits in a short queue (oldest first). 'Urgent' kinds (an escape jump)
    replace at once. Low-priority kinds (the decoder's turn banners) are dropped rather than queued behind others."""

    def __init__(self, min_s=BANNER_MIN_S, urgent=("jump",), low=("turn",), depth=3):
        self.min_s, self.urgent, self.low, self.depth = min_s, tuple(urgent), tuple(low), depth
        self.current, self.pending = None, []

    def show(self, t, banner):
        b = dict(banner, t0=t, t_event=t)
        free = self.current is None or t - self.current["t0"] >= self.min_s
        if b.get("kind") in self.urgent or (free and not self.pending):
            self.current, self.pending = b, [q for q in self.pending if q.get("kind") not in self.low]
        elif b.get("kind") in self.low and self.pending:
            return
        else:
            self.pending = (self.pending + [b])[-self.depth:]
            self.update(t)

    def update(self, t):
        if self.pending and (self.current is None or t - self.current["t0"] >= self.min_s):
            self.current = dict(self.pending.pop(0), t0=t)
        return self.current


class WindTurnDecoder:
    """DECODER: the wind descending neurons' left-right index -> a turn INTO the wind.

        u   = 1/2 [(DNp18_L - DNp18_R) - (DNp33_L - DNp33_R)]      Hz; > 0 when the wind comes from the left
        u_s = EMA(u, tau)
        yaw = clip(k * u_s, +-yaw_max)                              rad/s, + = a left turn (towards a wind from the left)

    Used as the `fn` of a games.common.ReadDecoder (attached, read-only): it receives the previous frame's rates of
    the four cells and keeps what it read, for the HUD."""

    READS: ClassVar[dict] = {"DNp18_L": {"type": "DNp18", "somaSide": "L"}, "DNp18_R": {"type": "DNp18", "somaSide": "R"},
             "DNp33_L": {"type": "DNp33", "somaSide": "L"}, "DNp33_R": {"type": "DNp33", "somaSide": "R"}}

    def __init__(self, gain_deg_s_per_hz=3.0, tau_s=0.15, yaw_max_deg_s=60.0):
        self.gain = float(gain_deg_s_per_hz)
        self.tau_s = float(tau_s)
        self.yaw_max = float(yaw_max_deg_s)
        self.reset()

    def reset(self):
        self.u = self.u_s = self.yaw = 0.0
        self.rates = dict.fromkeys(self.READS, 0.0)

    @staticmethod
    def index(p18L, p18R, p33L, p33R):
        return 0.5 * ((p18L - p18R) - (p33L - p33R))

    def update(self, dt_s, rates):
        self.rates = {k: float(rates[k]) for k in self.READS}
        r = self.rates
        self.u = self.index(r["DNp18_L"], r["DNp18_R"], r["DNp33_L"], r["DNp33_R"])
        a = math.exp(-dt_s / self.tau_s)
        self.u_s = a * self.u_s + (1 - a) * self.u
        self.yaw = float(np.clip(self.gain * DEG * self.u_s, -self.yaw_max * DEG, self.yaw_max * DEG))
        return self.yaw

    def __call__(self, dt_ms, x):
        keys = list(self.READS)
        vals = torch.stack([x[k].float().mean() for k in keys]).tolist()
        return self.update(dt_ms / 1000.0, dict(zip(keys, vals)))

    @property
    def law(self):
        return (f"yaw = {self.gain:g} deg/s per Hz x EMA{self.tau_s:g}s of 1/2[(DNp18 L-R) - (DNp33 L-R)] "
                f"(rate_hz: the brain's 100 ms running rate estimate), |yaw| <= {self.yaw_max:g} deg/s; + = a left "
                "turn, towards the side the index reports (upwind for lateral winds; its null is ~15 deg right of "
                "dead ahead)")

    def parameters(self):
        return {"gain_deg_s_per_hz": self.gain, "tau_s": self.tau_s, "yaw_max_deg_s": self.yaw_max,
                "direction": "towards the side the index reports (upwind for lateral winds)",
                "input": "rate_hz = flyverse/brain.py's running rate estimate (rate_tau 100 ms), previous 10 ms frame",
                "chain": "100 ms rate estimate -> EMA tau_s -> + Locomotion.readout yaw -> Locomotion.step (80 ms)",
                "added_to": "flyverse.body.Locomotion.readout yaw"}


class FanHolder:
    """GAME: the scripted hairdryer-holder of record mode, a hand-written heuristic (not the fly).

    It holds the nozzle beyond the apple on the fly's line to it, `lead` past the apple's centre and never closer
    than `min_range` to the fly, and pointed at the fly. Three corrections, all tuned on dev seeds >= 100:
      * wind from behind barely turns the decoder (the wind DNs' rear dead zone), so the nozzle is kept within
        `max_rel_deg` of the fly's heading: a fly facing away gets the wind from the side first;
      * the decoder settles with the wind ~15 deg right of dead ahead, so the holder integrates the fly's heading
        error to the apple (k_i, only while |error| < i_gate) into an offset of the nozzle's bearing;
      * within `settle` of the apple's centre it stops re-aiming (bearing and range held, the hand just follows the
        fly), so the fan does not swing about in the fly's view during the last few centimetres.
    The hand moves in polar coordinates around the fly (it never swings the hairdryer over it), at most `max_speed`."""

    def __init__(self, range_m, bearing_rel, heading, *, lead=0.10, min_range=0.15, max_rel_deg=100.0, k_i=0.6,
                 i_max_deg=40.0, i_gate_deg=45.0, max_speed=0.20, radial_speed=0.10, settle=0.075):
        self.range = float(range_m)
        self.bearing = wrap(heading + bearing_rel)           # world bearing fly -> nozzle
        self.lead, self.min_range, self.settle = lead, min_range, settle
        self.max_rel, self.k_i = max_rel_deg * DEG, k_i
        self.i_max, self.i_gate = i_max_deg * DEG, i_gate_deg * DEG
        self.max_speed, self.radial_speed = max_speed, radial_speed
        self.offset = 0.0
        self.side = 1.0
        self.err = 0.0
        self.clamped = False
        self.settled = False

    def parameters(self):
        return {"lead_m": self.lead, "min_range_m": self.min_range, "max_rel_deg": self.max_rel / DEG,
                "k_i_per_s": self.k_i, "i_max_deg": self.i_max / DEG, "i_gate_deg": self.i_gate / DEG,
                "max_speed_m_s": self.max_speed, "radial_speed_m_s": self.radial_speed, "settle_m": self.settle}

    def target(self, fly_xy, heading, apple_xy):
        """(bearing, range) the holder wants, and the fly's heading error to the apple (rad)."""
        d = np.asarray(apple_xy, float) - np.asarray(fly_xy, float)
        brg = math.atan2(d[1], d[0])
        dist = float(np.hypot(*d))
        err = wrap(heading - brg)
        rel = wrap(brg + self.offset - heading)
        clamped = abs(rel) > self.max_rel
        if clamped:
            if abs(rel) < 160 * DEG:                           # hysteresis: near dead astern keep the side in use
                self.side = math.copysign(1.0, rel)
            rel = self.side * self.max_rel
        rng = self.min_range if clamped else max(self.min_range, dist + self.lead)
        return wrap(heading + rel), rng, err, clamped

    def update(self, dt_s, fly_xy, heading, apple_xy):
        tb, tr, err, clamped = self.target(fly_xy, heading, apple_xy)
        self.err, self.clamped = err, clamped
        self.settled = float(np.hypot(*(np.asarray(apple_xy, float) - np.asarray(fly_xy, float)))) < self.settle
        if self.settled:
            return self.nozzle(fly_xy)
        if abs(err) < self.i_gate:
            self.offset = float(np.clip(self.offset - self.k_i * err * dt_s, -self.i_max, self.i_max))
        db = wrap(tb - self.bearing)
        step = self.max_speed * dt_s / max(self.range, 0.05)
        self.bearing = wrap(self.bearing + float(np.clip(db, -step, step)))
        self.range += float(np.clip(tr - self.range, -self.radial_speed * dt_s, self.radial_speed * dt_s))
        return self.nozzle(fly_xy)

    def nozzle(self, fly_xy):
        return np.asarray(fly_xy, float) + self.range * np.array([math.cos(self.bearing), math.sin(self.bearing)])


def hairdryer_parts(nozzle, aim):
    """The hairdryer as spheres (world.Sphere is axis-aligned, so a rotated body is built from balls): (centre,
    radius, material) for the nozzle ring, the barrel, the intake and the handle. `aim` points from the nozzle to the
    fly (3-D; it is pitched down at the table)."""
    nozzle = np.asarray(nozzle, float)
    a = np.asarray(aim, float)
    a = a / max(np.linalg.norm(a), 1e-9)
    back = -a
    parts = [(nozzle, 0.021, "black")]
    for k in range(1, 7):
        parts.append((nozzle + back * 0.019 * k, 0.029, "blueberry"))
    parts.append((nozzle + back * 0.135, 0.032, "black"))
    base = nozzle + back * 0.085
    down = np.array([0.0, 0.0, -1.0]) + 0.35 * np.array([back[0], back[1], 0.0])
    down /= np.linalg.norm(down)
    for k in range(1, 5):
        parts.append((base + down * 0.019 * k, 0.015, "blueberry"))
    return parts


# ------------------------------------------------------------------------------------------------ camera
class Cam:
    """A pinhole camera with a correct (not mirrored) projection. flyverse.world.render_camera returns a left-right
    mirrored image (column 0 looks to the camera's right; room_demo's Camera.project mirrors to match), so `render`
    flips it back."""

    def __init__(self, pos, target, w, h, fov_v_deg):
        self.pos = np.asarray(pos, float)
        f = np.asarray(target, float) - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, [0.0, 0.0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.w, self.h, self.fov = w, h, fov_v_deg
        self.tan = math.tan(fov_v_deg * DEG / 2)
        self.asp = w / h

    def project(self, p):
        v = np.asarray(p, float) - self.pos
        z = float(v @ self.f)
        if z <= 1e-4:
            return None
        x = float(v @ self.r) / z / (self.tan * self.asp)
        y = float(v @ self.u) / z / self.tan
        return ((x * 0.5 + 0.5) * self.w, (-y * 0.5 + 0.5) * self.h)

    def depth(self, pts):
        """Distance of each point in front of the camera along its axis (<= 0: behind it)."""
        return (np.asarray(pts, float) - self.pos) @ self.f

    def project_many(self, pts):
        v = np.asarray(pts, float) - self.pos
        z = np.maximum(v @ self.f, 1e-4)
        x = (v @ self.r) / z / (self.tan * self.asp)
        y = (v @ self.u) / z / self.tan
        return np.stack([(x * 0.5 + 0.5) * self.w, (-y * 0.5 + 0.5) * self.h], -1)

    def unproject_to_z(self, px, py, z0):
        """The point on the plane z = z0 under canvas pixel (px, py), or None."""
        x = (px / self.w * 2 - 1) * self.tan * self.asp
        y = -(py / self.h * 2 - 1) * self.tan
        d = self.f + x * self.r + y * self.u
        if abs(d[2]) < 1e-9:
            return None
        t = (z0 - self.pos[2]) / d[2]
        return None if t <= 0 else self.pos + t * d

    def render(self, wd, scale=1, detail=None):
        keep = wd.detail
        if detail is not None:
            wd.detail = detail
        try:
            img = wd.render_camera(self.pos, self.f, self.u, max(1, self.w // scale), max(1, self.h // scale), self.fov)
        finally:
            wd.detail = keep
        return torch.flip(img, dims=[1])


# ------------------------------------------------------------------------------------------------ the game
class HairdryerGame(common.Game):
    title = "hairdryer"
    subtitle = "steer the fly to the apple with a fan"

    # layout (canvas px, 1920 x 1080)
    MAIN = (12, 74, 1340, 950)
    COL_X, COL_W = 1364, 544

    def __init__(self, args):
        super().__init__(args)
        from flyverse import air, body, world
        self.air_mod, self.body_mod, self.world_mod = air, body, world
        self.interactive = args.record is None
        self.render_scale = args.render_scale or (1 if not self.interactive else 2)
        self.fb = common.build_brain(args)
        self.eyes = common.Eyes(self.fb)
        self.dec = WindTurnDecoder(args.gain, args.tau, args.yaw_max)
        self.reader = common.ReadDecoder("wind_turn", WindTurnDecoder.READS, self.dec, law=self.dec.law,
                                         parameters=self.dec.parameters())
        self.fb.attach(self.reader)
        self.decoder_on = args.decoder == "on"
        self.wd, self.info = world.make_room(args.seed, "apple")
        self.top = self.info["table_top_z"]
        _, cen, rad = self.info["fruit"][0]
        self.apple, self.apple_r = np.asarray(cen, float), float(rad)
        self.contact_r_xy = math.sqrt(max((self.apple_r + CONTACT_REACH) ** 2 - (self.apple[2] - self.top) ** 2, 0.0))
        self.n_hd = len(hairdryer_parts((0, 0, 0), (1, 0, 0)))
        self.hd_idx = list(range(len(self.wd.spheres), len(self.wd.spheres) + self.n_hd))
        self.wd.spheres += [world.Sphere((9.0, 9.0, 9.0), (r, r, r), m) for _, r, m in hairdryer_parts((9, 9, 9), (1, 0, 0))]
        x0, x1, y0, y1 = self.info["table_extent"]
        self.fence = (x0 + 0.02, x1 - 0.02, y0 + 0.02, y1 - 0.02)
        self.flight_room = (x0, x1, y0, y1, 2.6)
        self.particle_rng = np.random.default_rng(12345)          # visual only (wind streaks); never touches the brain
        self.start = tuple(float(v) for v in args.start.split(",")) if args.start else START
        if len(self.start) != 3:
            raise SystemExit("--start takes x,y,heading_deg")
        self._declare()
        self.reset_round()

    # ------------------------------------------------------------------------------------------ setup
    def _declare(self):
        a = self.args
        log = self.log
        log.meta["body"] = ("flyverse.body.Locomotion.readout / .step and Flight.readout / .maybe_takeoff, shipped and "
                            "unmodified; the decoder's yaw is added to the readout's yaw before Locomotion.step")
        log.meta["decoder_connected"] = self.decoder_on
        common.declare(log, "wind_turn", "decoder", self.dec.law,
                       "DNp18 L/R, DNp33 L/R (rate_hz = the brain's 100 ms running rate estimate, 1 cell each)",
                       connected=self.decoder_on, **self.dec.parameters())
        common.declare(log, "proboscis_drawing", "decoder", "display only, controls nothing: the fly sprite's "
                       "proboscis is drawn out while MN9 > 1 Hz, length 0.3 + 0.9 x min(1, MN9 / 20 Hz) mm (x the "
                       "sprite's 6x)", "MN9 (fb.motor().proboscis, the two MN9 cells' mean rate)", display_only=True)
        common.declare(log, "turn_banner", "game", "display only: a DECODER banner when |decoder yaw| > 0.8 x its cap "
                       "(at most every 3.5 s, before contact) and the turn goes towards the side the wind comes from; "
                       "every such crossing is logged as a 'turn' event with towards_wind true / false", "")
        common.declare(log, "fan_jet", "game",
                       f"wind at the fly blows from the nozzle to the fly (uniform horizontal field, flyverse.air.Air, "
                       f"meander off), speed = {JET_V_REF:g} m/s x {JET_D_REF:g} m / d, capped at {JET_V_MAX:g} m/s; "
                       "the apple's odour plume follows it", "",
                       v_ref_m_s=JET_V_REF, d_ref_m=JET_D_REF, v_max_m_s=JET_V_MAX, nozzle_height_m=NOZZLE_HEIGHT)
        if self.interactive:
            common.declare(log, "fan_mouse", "game", "the player places the hairdryer with the mouse (6-60 cm from the "
                           "fly, the hand at <= 0.6 m/s); it always points at the fly; hold the button (or latch "
                           "SPACE) to blow; the wheel scales the jet law by 0.3-2.4, the wind still capped at "
                           f"{JET_V_MAX:g} m/s; A hands over to the scripted holder", "",
                           power_min=0.3, power_max=2.4, v_max_m_s=JET_V_MAX)
        hp = FanHolder(FAN_START[0], FAN_START[1] * DEG, 0.0).parameters()
        common.declare(log, "fan_holder", "game",
                       f"scripted holder: nozzle beyond the apple on the fly's line to it ({hp['lead_m'] * 100:g} cm past "
                       f"its centre, >= {hp['min_range_m'] * 100:g} cm from the fly), kept within "
                       f"{hp['max_rel_deg']:g} deg of the fly's heading, bearing offset by the integral of the fly's "
                       f"heading error to the apple; stops re-aiming within {hp['settle_m'] * 100:g} cm of the apple's "
                       f"centre; the hand moves round the fly at <= {hp['max_speed_m_s']:g} m/s; on at {a.fan_on:g} s, "
                       "off at first contact", "fly pose, apple position (not the brain)", fan_on_s=a.fan_on, **hp)
        common.declare(log, "contact_taste", "game", "on its feet within 1.5 cm of the apple's surface -> "
                       "fb.taste(1.0): the 165 sweet GRNs fire at 120 Hz (shipped sense); else 0", "",
                       reach_m=CONTACT_REACH)
        common.declare(log, "feeding_hold", "game", "while the fly touches the apple its walk is held (speed 0, yaw "
                       "0), as room_demo holds a feeding fly", "")
        common.declare(log, "fence", "game", "invisible fence 2 cm inside the table edge (room_demo --fence); "
                       "body.Locomotion's edge rule turns the fly inward there", "")
        x, y, h = self.start
        common.declare(log, "start", "game", f"fly at ({x:g}, {y:g}) m, heading {h:g} deg"
                       + (", the apple behind it to its left; the same for every seed" if self.start == START else ""),
                       "", x=x, y=y, heading_deg=h)

    def reset_round(self):
        body = self.body_mod
        if getattr(self, "samples", None):                     # interactive restart: keep the round that just ended
            self._summarize()
            done = {k: v for k, v in self.log.summary.items() if k not in ("samples_10hz", "sample_columns",
                                                                           "earlier_rounds", "note")}
            self.rounds.append(dict(done, round_start_t_s=round(self.round_t0, 3), round_end_t_s=round(self.t_s, 3)))
        elif not hasattr(self, "rounds"):
            self.rounds = []
        x, y, h = self.start
        self.fly = body.FlyState(x=x, y=y, z=self.top, heading=h * DEG)
        self.loco, self.flight = body.Locomotion(), body.Flight()
        self.air = self.air_mod.Air([(n, c, r / 0.02, r) for n, c, r in self.info["fruit"]],
                                    self.air_mod.WindParams(speed=0.0, direction_deg=0.0, meander_deg=0.0),
                                    seed=self.args.seed)
        self.holder = FanHolder(FAN_START[0], FAN_START[1] * DEG, self.fly.heading)
        # display only: the main camera sits on the side of the fly's line to the apple away from where the holder
        # will bring the hairdryer, so the hairdryer never fills the foreground (a copy: the holder is not touched)
        tb = copy.copy(self.holder).target(self.fly.pos[:2], self.fly.heading, self.apple[:2])[0]
        to_apple = self.apple[:2] - self.fly.pos[:2]
        self.cam_side = 1.0 if to_apple[0] * math.sin(tb) - to_apple[1] * math.cos(tb) < 0 else -1.0
        self.auto = not self.interactive
        self.fan_xy = self.holder.nozzle(self.fly.pos[:2])
        self.mouse_xy = None
        self.fan_on = False
        self.fan_on_t = None                 # brain time the hairdryer first switched on this round
        self.mouse_down = False              # interactive: the left button blows while held ...
        self.space_latch = False             # ... and SPACE latches it on / off
        self.power = 1.0                     # interactive: multiplies the jet law (the cap still applies)
        self.wind_v = 0.0
        self.alpha = 0.0                     # wind-from angle relative to the heading (rad)
        self.tasting = 0.0
        self.contact_t = None
        self.fed_announced = False
        self.mn9_first = False
        self.feeding = False
        self.round_t0 = self.t_s
        self.holder_t0 = self.t_s            # the scripted holder's own clock (its switch-on time counts from here)
        self.rad = None
        self.motor = None
        self.yaw_applied = 0.0
        self.trail = [self.fly.pos.copy()]
        self.hist = {k: deque(maxlen=HIST_LEN) for k in ("p18L", "p18R", "p33L", "p33R", "yaw", "mn9", "gf")}
        self.banners = BannerQueue()
        self.last_turn_banner = -9.0
        self._hd_pose = None
        self.particles = np.zeros((0, 7))    # x, y, z, vx, vy, vz, age
        self.cam_state = None
        self.path_len = 0.0
        self.min_dist = self._dist_xy()
        self.start_dist = self.min_dist
        self.takeoffs = {"giant fibre": 0, "wing power MNs": 0}
        self.samples = []
        self.stats = {"u_abs": [], "err_abs": [], "mn9_contact": [], "gf_max": 0.0, "gf_max_pre": 0.0, "wind": []}
        self.dec.reset()
        self._place_hairdryer()
        self.show_banner("HAIRDRYER OFF", "the raw model walks straight; every turn here is the decoder's",
                         common.MUTED, "fan", ("game",))

    # ------------------------------------------------------------------------------------------ helpers
    def _dist_xy(self):
        return float(np.hypot(self.fly.x - self.apple[0], self.fly.y - self.apple[1]))

    def _jet(self, d):
        """GAME: wind speed at the fly for a nozzle d m away; the interactive power scales the law, the cap holds."""
        return min(JET_V_MAX, jet_speed(d) * self.power)

    def _place_hairdryer(self):
        nz = np.array([self.fan_xy[0], self.fan_xy[1], self.top + NOZZLE_HEIGHT])
        aim = self.fly.pos + np.array([0, 0, 0.004]) - nz
        self.nozzle = nz
        self.aim = aim / max(np.linalg.norm(aim), 1e-9)
        pose = np.concatenate([nz, aim])
        if self._hd_pose is not None and np.array_equal(pose, self._hd_pose):
            return                                             # nothing moved (e.g. after contact): skip 12 copies
        self._hd_pose = pose
        for i, (c, r, _) in zip(self.hd_idx, hairdryer_parts(nz, aim)):
            self.wd.move_sphere(i, c, (r, r, r))

    def show_banner(self, text, sub="", color=common.WHITE, kind=None, chips=("game",)):
        """Queue an event banner; `chips` are the provenance classes of what it reports (drawn on it)."""
        self.banners.show(self.t_s, {"text": text, "sub": sub, "color": color, "kind": kind, "chips": tuple(chips)})

    def eye_radiance(self):
        d = self.fly.body_to_world(self.eyes.dirs_body.reshape(-1, 3))
        o = np.broadcast_to(self.fly.eye_pos, d.shape)
        samples = self.wd.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        return self.eyes.pool(samples)

    # ------------------------------------------------------------------------------------------ the tick
    def tick(self):
        fly, t = self.fly, self.t_s
        # 1. the hand (GAME)
        if self.auto:
            if self.contact_t is None:
                self.fan_xy = self.holder.update(DT_S, fly.pos[:2], fly.heading, self.apple[:2])
            want_on = (t - self.holder_t0) >= self.args.fan_on and self.contact_t is None
        else:
            if self.mouse_xy is not None:
                d = self.mouse_xy - fly.pos[:2]
                r = float(np.hypot(*d))
                tgt = fly.pos[:2] + d / max(r, 1e-9) * float(np.clip(r, 0.06, 0.60))
                step = tgt - self.fan_xy
                n = float(np.hypot(*step))
                self.fan_xy = self.fan_xy + step * min(1.0, 0.6 * DT_S / max(n, 1e-9))
            want_on = self.mouse_down or self.space_latch
        d_fan = float(np.hypot(*(self.fan_xy - fly.pos[:2])))
        if want_on != self.fan_on:
            self.fan_on = want_on
            self.log.event(t, "fan_on" if want_on else "fan_off", nozzle_xy=[round(v, 4) for v in self.fan_xy],
                           dist_to_fly_m=round(d_fan, 4))
            if want_on:
                if self.fan_on_t is None:
                    self.fan_on_t = t
                v = self._jet(d_fan)
                self.show_banner("HAIRDRYER ON", f"{v:.2f} m/s at the fly  ·  Johnston's organ -> wind DNs",
                                 common.AMBER, "fan", ("game",))
        self._place_hairdryer()
        # 2. the air (GAME wind, shipped plume)
        self.wind_v = self._jet(d_fan) if self.fan_on else 0.0
        towards = wind_towards_deg(self.fan_xy, fly.pos[:2])
        self.air.wind.speed, self.air.wind.direction_deg = self.wind_v, towards
        self.air.step(DT_S)
        self.alpha = wind_from_relative(towards, fly.heading)
        # 3. senses (shipped transduction)
        self.rad = self.eye_radiance()
        self.fb.vision(self.rad)
        self.fb.smell(*self.air.antennae(fly.eye_pos, fly.left, fly.forward))
        self.fb.wind(*self.air.deflections(fly.forward, fly.left))
        touching = in_contact(fly.pos, self.apple, self.apple_r, fly.airborne)
        self.tasting = 1.0 if touching else 0.0
        self.fb.taste(self.tasting)
        # 4. the brain
        self.fb.step(common.TICK_MS)
        self.t_s += DT_S
        t = self.t_s
        m = self.fb.motor()
        self.motor = m
        yaw_dec = float(self.reader.value or 0.0)
        # 5. the body (shipped readout + the declared decoder's yaw)
        cmd = self.loco.readout(m, dt_s=DT_S)
        wcmd = self.flight.readout(m)
        self.yaw_applied = yaw_dec if self.decoder_on else 0.0
        cmd = dict(cmd, yaw=cmd["yaw"] + self.yaw_applied)
        if touching and self.contact_t is None:
            self.contact_t = t
            self.log.event(t, "contact", time_from_fan_on_s=None if self.fan_on_t is None else round(t - self.fan_on_t, 3),
                           x=round(fly.x, 4), y=round(fly.y, 4), path_m=round(self.path_len, 4),
                           rule="on its feet within 1.5 cm of the apple's surface")
            self.show_banner("CONTACT", f"within 1.5 cm of the apple at t = {t - self.round_t0:.1f} s -> sugar "
                             "GRNs 120 Hz", common.SAGE, "contact", ("game",))
        self.feeding = touching
        if self.feeding:
            cmd = dict(cmd, speed=0.0, yaw=0.0)
        p0 = fly.pos.copy()
        dist_before = self._dist_xy()
        if fly.airborne:
            self.flight.step(fly, wcmd, DT_S, lambda x, y: self.top, self.flight_room)
            if not fly.airborne:
                self.log.event(t, "landing", x=round(fly.x, 4), y=round(fly.y, 4))
        else:
            if self.flight.maybe_takeoff(fly, wcmd):
                escape = float(wcmd["gf"]) >= float(wcmd.get("gf_threshold", self.flight.gf_hz))
                self.takeoffs["giant fibre" if escape else "wing power MNs"] += 1
                self.log.event(t, "takeoff", cause="giant fibre" if escape else "wing power MNs",
                               gf_hz=round(float(m.gf), 2), power_hz=round(float(m.power), 2),
                               dist_to_apple_centre_cm=round(dist_before * 100, 2))
                if escape:
                    self.show_banner(f"GIANT FIBRE {m.gf:.0f} Hz -> JUMP", "body model: GF >= 33 Hz -> escape "
                                     "takeoff", common.RED, "jump", ("connectome",))
                else:
                    self.show_banner(f"WING MNs {m.power:.0f} Hz -> HOP", "body model: wing power MNs >= 50 Hz for "
                                     "0.3 s -> takeoff", common.LILAC, "hop", ("connectome",))
            else:
                self.loco.step(fly, cmd, DT_S, self.fence)
        self.path_len += float(np.linalg.norm(fly.pos[:2] - p0[:2]))
        dist = self._dist_xy()
        self.min_dist = min(self.min_dist, dist)
        # 6. HUD history, banners, log
        r = self.dec.rates
        for k, v in (("p18L", r["DNp18_L"]), ("p18R", r["DNp18_R"]), ("p33L", r["DNp33_L"]), ("p33R", r["DNp33_R"]),
                     ("yaw", self.dec.yaw / DEG), ("mn9", float(m.proboscis)), ("gf", float(m.gf))):
            self.hist[k].append(v)
        self.stats["gf_max"] = max(self.stats["gf_max"], float(m.gf))
        if self.contact_t is None:
            self.stats["gf_max_pre"] = max(self.stats["gf_max_pre"], float(m.gf))
        if self.fan_on:
            self.stats["wind"].append(self.wind_v)
            self.stats["u_abs"].append(abs(self.dec.u_s))
        if self.feeding:
            self.stats["mn9_contact"].append(float(m.proboscis))
        if self.contact_t is not None and not self.mn9_first and self.hist["mn9"] and self.hist["mn9"][-1] > 2.0:
            self.mn9_first = True
            self.log.event(t, "mn9_fires", mn9_hz=round(float(m.proboscis), 2),
                           after_contact_s=round(t - self.contact_t, 3))
        touch = self.stats["mn9_contact"]
        if self.contact_t is not None and not self.fed_announced and len(touch) >= round(1.0 / DT_S):
            self.fed_announced = True                          # one second of contact: announce its mean MN9 rate
            mean = float(np.mean(touch))
            self.log.event(t, "feeding", mn9_mean_first_s_hz=round(mean, 2))
            if mean > 0.5:
                self.show_banner(f"FEEDING: MN9 {mean:.1f} Hz", "1 s mean · sugar GRNs -> proboscis motor neuron",
                                 common.SAGE, "feeding", ("connectome",))
            else:
                self.show_banner(f"MN9 SILENT: {mean:.1f} Hz", "sugar GRNs fire, the proboscis motor neuron does not",
                                 common.MUTED, "feeding", ("connectome",))
        yd = self.dec.yaw / DEG
        if (self.decoder_on and self.fan_on and abs(yd) > 0.8 * self.dec.yaw_max and t - self.last_turn_banner > 3.5
                and self.contact_t is None):
            self.last_turn_banner = t
            head, sub, towards = turn_banner(self.dec.u_s, yd, self.alpha)
            if towards:                                        # the banner only when the turn is towards the wind
                self.show_banner(head, sub, common.TEAL, "turn", ("connectome", "decoder"))
            self.log.event(t, "turn", side="L" if yd > 0 else "R", yaw_deg_s=round(yd, 1),
                           index_u_smoothed_hz=round(self.dec.u_s, 1), dnp18_L_hz=round(r["DNp18_L"], 1),
                           dnp18_R_hz=round(r["DNp18_R"], 1), dnp33_L_hz=round(r["DNp33_L"], 1),
                           dnp33_R_hz=round(r["DNp33_R"], 1), wind_from_deg=round(self.alpha / DEG, 1),
                           towards_wind=towards, bannered=bool(towards))
        step_i = round((t - self.round_t0) / DT_S)
        if step_i % 5 == 0:
            self.trail.append(fly.pos.copy())
        if step_i % 10 == 0:
            brg = math.atan2(self.apple[1] - fly.y, self.apple[0] - fly.x)
            err = wrap(fly.heading - brg)
            if self.fan_on and self.contact_t is None:
                self.stats["err_abs"].append(abs(err) / DEG)
            self.samples.append([round(t, 2), round(fly.x, 4), round(fly.y, 4), round(fly.heading / DEG, 1),
                                 round(dist * 100, 2), round(self.wind_v, 3), round(self.alpha / DEG, 1),
                                 round(self.dec.u, 1), round(self.dec.u_s, 1), round(yd, 1),
                                 round(r["DNp18_L"], 1), round(r["DNp18_R"], 1), round(r["DNp33_L"], 1),
                                 round(r["DNp33_R"], 1), round(float(m.proboscis), 2), round(float(m.gf), 1),
                                 int(self.tasting), round(err / DEG, 1)])
            self._summarize()
        self._advance_particles()
        self.banners.update(self.t_s)

    def _summarize(self):
        s = self.stats
        on = self.fan_on_t if self.fan_on_t is not None else self.round_t0 + self.args.fan_on
        self.log.summary = {
            "decoder_connected": self.decoder_on,
            "reached_apple": self.contact_t is not None,
            "contact_t_s": None if self.contact_t is None else round(self.contact_t - self.round_t0, 3),
            "contact_after_fan_on_s": None if self.contact_t is None else round(self.contact_t - on, 3),
            "start_dist_to_apple_centre_cm": round(self.start_dist * 100, 2),
            "contact_radius_xy_cm": round(self.contact_r_xy * 100, 2),
            "min_dist_to_apple_centre_cm": round(self.min_dist * 100, 2),
            "final_dist_to_apple_centre_cm": round(self._dist_xy() * 100, 2),
            "path_length_cm": round(self.path_len * 100, 2),
            "mean_abs_heading_error_to_apple_deg_fan_on": round(float(np.mean(s["err_abs"])), 1) if s["err_abs"] else None,
            "mean_abs_u_smoothed_hz_fan_on": round(float(np.mean(s["u_abs"])), 2) if s["u_abs"] else None,
            "mean_wind_m_s_fan_on": round(float(np.mean(s["wind"])), 3) if s["wind"] else None,
            "mn9_mean_hz_while_touching": round(float(np.mean(s["mn9_contact"])), 2) if s["mn9_contact"] else None,
            "mn9_max_hz_while_touching": round(float(np.max(s["mn9_contact"])), 2) if s["mn9_contact"] else None,
            "seconds_touching": round(len(s["mn9_contact"]) * DT_S, 2),
            "gf_max_hz": round(s["gf_max"], 2),
            "gf_max_hz_before_contact": round(s["gf_max_pre"], 2),
            "takeoffs_giant_fibre": self.takeoffs["giant fibre"],
            "takeoffs_wing_power_mns": self.takeoffs["wing power MNs"],
            "sample_columns": ["t_s", "x_m", "y_m", "heading_deg", "dist_cm", "wind_m_s", "wind_from_deg_rel",
                               "u_hz", "u_smoothed_hz", "decoder_yaw_deg_s", "DNp18_L", "DNp18_R", "DNp33_L",
                               "DNp33_R", "MN9_hz", "GF_hz", "tasting", "heading_error_to_apple_deg"],
            "samples_10hz": self.samples,
        }
        if self.rounds:
            self.log.summary["note"] = ("interactive restarts: this summary covers the last round (since "
                                        f"t = {self.round_t0:.2f} s); earlier_rounds holds the ones before")
            self.log.summary["earlier_rounds"] = self.rounds

    # ------------------------------------------------------------------------------------------ particles (visual)
    def _advance_particles(self):
        p = self.particles
        if len(p):
            p[:, 0:3] += p[:, 3:6] * DT_S
            p[:, 6] += DT_S
            low = p[:, 2] < self.top + 0.003                     # the jet spreads along the cloth
            if low.any():
                p[low, 2] = self.top + 0.003
                p[low, 5] = 0.0
            deflect_round_sphere(p, self.apple, self.apple_r, age_after_hit=0.62)   # wraps round the apple and fades
                                                                # out over ~0.3 s (a picture of the jet, not the model)
            p = p[p[:, 6] < 0.9]
        if self.fan_on and self.wind_v > 0:
            n = int(self.particle_rng.poisson(3.0 + 6.0 * self.wind_v))
            rng = self.particle_rng
            a = self.aim
            side = np.cross(a, [0.0, 0.0, 1.0])
            side /= max(np.linalg.norm(side), 1e-9)
            up = np.cross(side, a)
            ang = rng.uniform(0, 2 * np.pi, n)
            rr = rng.uniform(0, 0.016, n)
            pos = self.nozzle[None] + (np.cos(ang) * rr)[:, None] * side[None] + (np.sin(ang) * rr)[:, None] * up[None]
            spread = rng.normal(0, 0.10, (n, 2))
            dirs = a[None] + spread[:, :1] * side[None] + spread[:, 1:] * up[None] * 0.5
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            vis = 0.55 * max(self.wind_v, 0.25) * rng.uniform(0.7, 1.3, (n, 1))
            new = np.concatenate([pos, dirs * vis, np.zeros((n, 1))], 1)
            p = np.concatenate([p, new], 0)[-600:]
        self.particles = p

    # ------------------------------------------------------------------------------------------ input
    def handle(self, event, canvas_pos=None):
        import pygame
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.space_latch = not self.space_latch
            elif event.key == pygame.K_a:
                self.auto = not self.auto
                if self.auto:
                    rel = wrap(math.atan2(*(self.fan_xy - self.fly.pos[:2])[::-1]) - self.fly.heading)
                    self.holder = FanHolder(float(np.hypot(*(self.fan_xy - self.fly.pos[:2]))), rel, self.fly.heading)
                    self.holder_t0 = self.t_s - self.args.fan_on   # blows at once; the brain-time clock is untouched
                self.log.event(self.t_s, "autopilot", on=self.auto)
            elif event.key == pygame.K_d:
                self.decoder_on = not self.decoder_on
                self.log.event(self.t_s, "decoder", connected=self.decoder_on)
                self.show_banner("DECODER " + ("CONNECTED" if self.decoder_on else "DISCONNECTED"),
                                 "the wind DNs " + ("steer the walk" if self.decoder_on else "fire, but nothing turns"),
                                 common.TEAL, "decoder", ("decoder",))
            elif event.key == pygame.K_r:
                self.log.event(self.t_s, "restart")
                self.reset_round()
        elif event.type == pygame.MOUSEWHEEL:
            self.power = float(np.clip(self.power * (1.15 ** event.y), 0.3, 2.4))
        elif event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            self.mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP and getattr(event, "button", 0) == 1:
            self.mouse_down = False
        if canvas_pos is not None and event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN) and self.cam_state:
            mx, my, _, _ = self.MAIN
            cam = self._camera()
            p = cam.unproject_to_z(canvas_pos[0] - mx, canvas_pos[1] - my, self.top)
            if p is not None:
                self.mouse_xy = np.asarray(p[:2], float)

    # ------------------------------------------------------------------------------------------ drawing
    def _camera(self, state=None):
        _, _, mw, mh = self.MAIN
        tgt, dist, az = state or self.cam_state
        el = CAM_EL_DEG * DEG
        pos = tgt + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return Cam(pos, tgt, mw, mh, CAM_FOV_DEG)

    def _wide_shot(self):
        """Keep the hairdryer in the framing: always while a player holds it; in record mode until WIDE_SHOT_S after
        it switches on (the set-up beat). Then the camera closes in on the fly and the apple."""
        if self.interactive:
            return True
        return self.fan_on_t is None or self.t_s - self.fan_on_t < WIDE_SHOT_S

    def _cam_goal(self):
        """(target, distance, azimuth) the main camera eases towards: side-on to the fly's line to the apple (on the
        side away from the hairdryer, see reset_round), at the nearest distance that keeps the key points inside a box of the
        view -- the fly and the apple (+ the hairdryer in the wide shot); after contact the fly and the near side of
        the apple, close up. Display only."""
        _, _, mw, mh = self.MAIN
        fly = self.fly
        up = np.array([0.0, 0.0, 1.0])
        d = self.apple[:2] - fly.pos[:2]
        az = math.atan2(d[1], d[0]) + self.cam_side * math.pi / 2
        if self.contact_t is not None:                          # the fly (+ its proboscis) and the apple's near side
            n = fly.pos - self.apple
            near = self.apple + n / max(float(np.linalg.norm(n)), 1e-9) * self.apple_r
            pts = np.array([fly.pos - 0.010 * fly.forward, fly.pos + 0.016 * fly.forward, fly.pos + 0.008 * up, near,
                            self.apple + 0.25 * self.apple_r * up])
            box, ds = (0.10, 0.90, 0.24, 0.90), np.arange(0.05, 0.8, 0.005)
        else:                                                   # the fly and most of the apple (its top may crop)
            pts = [fly.pos, fly.pos + 0.01 * up, self.apple + 0.75 * self.apple_r * up, self.apple - self.apple_r * up]
            if self._wide_shot():
                pts += [self.nozzle, self.nozzle - self.aim * 0.05]
            pts = np.array(pts)
            box, ds = (0.08, 0.92, 0.26, 0.88), np.arange(0.12, 1.4, 0.01)
        c = (pts.min(0) + pts.max(0)) / 2                       # aim at the middle of their 3-D box
        for dd in ds:
            cam = self._camera((c, float(dd), az))
            if cam.depth(pts).min() < 0.01:
                continue
            uv = cam.project_many(pts) / np.array([mw, mh])
            if (box[0] < uv[:, 0].min() and uv[:, 0].max() < box[1] and box[2] < uv[:, 1].min()
                    and uv[:, 1].max() < box[3]):
                return c, float(dd), az
        return c, float(ds[-1]), az

    def _update_camera(self):
        c, d, az = self._cam_goal()
        if self.cam_state is None:
            self.cam_state = (c, d, az)
            return
        c0, d0, az0 = self.cam_state
        tau = 0.9 if self.contact_t is not None else 1.2          # s; the push-in after contact takes about 2 s
        a = 1 - math.exp(-(1.0 / self.args.fps) / tau)
        self.cam_state = (c0 + a * (c - c0), float(math.exp(math.log(d0) + a * (math.log(d) - math.log(d0)))),
                          az0 + a * wrap(az - az0))

    def _footer_laws(self):
        dec = (f"DECODER yaw = {self.dec.gain:g} deg/s/Hz x EMA[(DNp18 L-R)-(DNp33 L-R)]/2, <= {self.dec.yaw_max:g} "
               "deg/s" + ("" if self.decoder_on else " (OFF)"))
        hand = "GAME scripted holder aims it: sees fly pose + apple" if self.auto else "GAME the player aims it (mouse)"
        rule = f"GAME jet {JET_V_REF:g} m/s x {JET_D_REF:g} m/d; within 1.5 cm of apple -> taste 1"
        return [dec, hand, rule]

    def draw(self, surface):
        import pygame
        hud = self.hud
        surface.fill(common.BG)
        hud.header(surface, self.title, self.subtitle, common.model_line(self.fb))
        self._update_camera()
        cam = self._camera()
        self._draw_main(surface, cam)
        x, w = self.COL_X, self.COL_W
        self._draw_eye_panel(surface, pygame.Rect(x, 74, w, 292))
        self._draw_dn_panel(surface, pygame.Rect(x, 374, w, 222))
        self._draw_decoder_panel(surface, pygame.Rect(x, 604, w, 238))
        self._draw_mn9_panel(surface, pygame.Rect(x, 850, w, 174))
        hud.footer(surface, self._footer_laws())

    def _backed_text(self, surf, txt, pos, size, color, *, anchor="topleft", bold=False, display=False, alpha=185,
                     pad=(8, 3)):
        """Text on a translucent dark backing (legible over the checked cloth); returns the backing rect."""
        pg = self.hud.pg
        img = self.hud.font(size, bold, display).render(txt, True, color)
        r = img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        br = r.inflate(2 * pad[0], 2 * pad[1])
        back = pg.Surface(br.size, pg.SRCALPHA)
        back.fill((10, 12, 14, alpha))
        surf.blit(back, br.topleft)
        surf.blit(img, r)
        return br

    def _draw_main(self, s, cam):
        import pygame
        hud = self.hud
        mx, my, mw, mh = self.MAIN
        img = cam.render(self.wd, self.render_scale, detail=0.55)
        rgb = np.ascontiguousarray(self.world_mod.to_rgb8(img, exposure=2.1))
        surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        if surf.get_size() != (mw, mh):
            surf = pygame.transform.smoothscale(surf, (mw, mh))
        view = pygame.Surface((mw, mh))
        view.blit(surf, (0, 0))
        ov = pygame.Surface((mw, mh), pygame.SRCALPHA)
        self._draw_trail(ov, cam)
        self._draw_apple_target(ov, cam)
        self._draw_particles(ov, cam)
        self._draw_wind_arrow(ov, cam)
        self._draw_fly(ov, cam)
        view.blit(ov, (0, 0))
        view.blit(self._vignette(mw, mh), (0, 0))
        self._draw_labels(view, cam)
        s.blit(view, (mx, my))
        pygame.draw.rect(s, common.LINE, (mx, my, mw, mh), 1, border_radius=4)
        self._draw_status(s, mx + 16, my + 14)
        self._draw_banner(s, pygame.Rect(mx, my, mw, mh))
        hud.text(s, f"t = {self.t_s - self.round_t0:5.2f} s", (mx + mw - 18, my + 14), 26, common.TEXT,
                 anchor="topright")
        hud.text(s, "brain time", (mx + mw - 18, my + 46), 14, common.DIM, anchor="topright")
        self._backed_text(s, f"fly drawn {FLY_DRAW_SCALE:g}x life size  ·  wind streaks: a picture of the jet, not "
                          "its data", (mx + 16, my + mh - 12), 15, common.MUTED, anchor="bottomleft")
        if self.contact_t is not None:
            box = pygame.Rect(mx + 16, my + 182, 356, 64)
            pygame.draw.rect(s, (22, 40, 26), box, border_radius=6)
            pygame.draw.rect(s, common.SAGE, box, 2, border_radius=6)
            hud.text(s, "APPLE WITHIN REACH", (box.x + 16, box.y + 8), 20, common.SAGE, bold=True, display=True)
            hud.text(s, f"at t = {self.contact_t - self.round_t0:.1f} s", (box.x + 16, box.y + 32), 24, common.WHITE,
                     bold=True, display=True)
            hud.chip(s, "game", (box.right - 64, box.y + 10), 11)
            hud.text(s, "1.5 cm rule", (box.right - 12, box.y + 38), 13, common.MUTED, anchor="topright")

    def _vignette(self, w, h):
        key = (w, h)
        if getattr(self, "_vig_key", None) != key:
            import pygame
            yy, xx = np.mgrid[0:h, 0:w]
            r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
            a = (np.clip((r - 0.75) / 0.6, 0, 1) ** 1.6 * 150).astype(np.uint8)
            arr = np.zeros((h, w, 4), np.uint8)
            arr[..., 3] = a
            arr[..., :3] = (6, 8, 9)
            self._vig = pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA").copy()
            self._vig_key = key
        return self._vig

    def _draw_trail(self, ov, cam):
        import pygame
        if len(self.trail) < 2:
            return
        pts3 = np.array(self.trail + [self.fly.pos])
        pts = cam.project_many(pts3)
        front = cam.depth(pts3) > 0.01
        n = len(pts)
        seg = max(1, n // 60)
        for i in range(0, n - 1, seg):
            j = min(n - 1, i + seg)
            if front[i] and front[j]:
                a = int(60 + 170 * (i / max(n - 1, 1)))
                pygame.draw.line(ov, (*common.TEAL, a), pts[i], pts[j], 4)

    def _hidden_by_apple(self, cam, pts):
        """(n,) True where the sight line from the camera to a point meets the apple first."""
        o = cam.pos
        d = np.asarray(pts, float) - o
        L = np.linalg.norm(d, axis=1)
        d = d / L[:, None]
        oc = o - self.apple
        b = d @ oc
        c = oc @ oc - self.apple_r ** 2
        disc = b * b - c
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        return (disc > 0) & (t > 0) & (t < L - 1e-4)

    def _draw_apple_target(self, ov, cam):
        import pygame
        ang = np.linspace(0, 2 * np.pi, 64)
        pulse = 0.5 + 0.5 * math.sin(self.t_s * 4.0)
        col = common.SAGE if self.contact_t is not None else common.AMBER
        pts3 = np.stack([self.apple[0] + self.contact_r_xy * np.cos(ang), self.apple[1] + self.contact_r_xy * np.sin(ang),
                         np.full_like(ang, self.top + 0.0005)], 1)
        ring = cam.project_many(pts3)
        seen = ~self._hidden_by_apple(cam, pts3) & (cam.depth(pts3) > 0.01)   # the ring passes behind the apple
        a = int(150 + 80 * pulse)
        for i in range(len(ang) - 1):
            if seen[i] and seen[i + 1]:
                pygame.draw.line(ov, (*col, a), ring[i], ring[i + 1], 3)
        if self.contact_t is None:                            # dashed line fly -> apple contact edge
            d = self.apple[:2] - self.fly.pos[:2]
            dist = float(np.hypot(*d))
            if dist > self.contact_r_xy + 0.005:
                u = d / dist
                a = self.fly.pos[:2] + u * 0.012
                b = self.apple[:2] - u * self.contact_r_xy
                n = max(2, int(np.hypot(*(b - a)) / 0.008))
                pts = cam.project_many(np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n),
                                                 np.full(n, self.top + 0.0005)], 1))
                for i in range(0, n - 1, 2):
                    pygame.draw.line(ov, (*common.AMBER, 170), pts[i], pts[i + 1], 2)

    def _draw_particles(self, ov, cam):
        """The wind streaks (a picture of the jet): depth-tested against the apple, so none crosses its face."""
        import pygame
        p = self.particles
        if not len(p):
            return
        h3 = p[:, :3]
        t3 = p[:, :3] - p[:, 3:6] * 0.06
        m3 = 0.5 * (h3 + t3)
        seen = ~(self._hidden_by_apple(cam, h3) | self._hidden_by_apple(cam, t3) | self._hidden_by_apple(cam, m3))
        seen &= (cam.depth(h3) > 0.01) & (cam.depth(t3) > 0.01)
        head, tail = cam.project_many(h3), cam.project_many(t3)
        age = p[:, 6]
        alpha = np.clip(np.minimum(age / 0.08, (0.9 - age) / 0.35), 0, 1) * 200
        for h, tl, a, ok in zip(head, tail, alpha, seen):
            if ok and a > 4:
                pygame.draw.line(ov, (215, 238, 255, int(a)), tl, h, 2)

    def _draw_wind_arrow(self, ov, cam):
        import pygame
        if not self.fan_on or self.wind_v <= 0:
            return
        f = self.fly.pos
        u = f[:2] - self.fan_xy
        u = u / max(np.hypot(*u), 1e-9)
        n = np.array([-u[1], u[0]])
        L = 0.045
        w = 0.004 + 0.006 * min(self.wind_v, 1.0)
        tail = f[:2] - u * (0.018 + L)
        tip = f[:2] - u * 0.016
        neck = tip - u * 0.012
        poly = [tail + n * w * 0.45, neck + n * w * 0.45, neck + n * w * 1.2, tip, neck - n * w * 1.2,
                neck - n * w * 0.45, tail - n * w * 0.45]
        pts3 = np.array([[q[0], q[1], self.top + 0.001] for q in poly])
        if cam.depth(pts3).min() < 0.01:
            return
        pts = cam.project_many(pts3)
        pygame.draw.polygon(ov, (150, 215, 255, 215), pts.tolist())
        pygame.draw.lines(ov, (235, 248, 255, 255), True, pts.tolist(), 2)

    def _fly_polys(self):
        """Fly sprite in the body frame (mm, x forward, y left), drawn at FLY_DRAW_SCALE: wings, abdomen, thorax, head."""
        def ell(cx, cy, rx, ry, rot=0.0, n=18):
            t = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x, y = rx * np.cos(t), ry * np.sin(t)
            c, s_ = math.cos(rot), math.sin(rot)
            return np.stack([cx + c * x - s_ * y, cy + s_ * x + c * y], 1)
        return [("wing", ell(-0.75, 0.55, 1.15, 0.42, 2.75)), ("wing", ell(-0.75, -0.55, 1.15, 0.42, -2.75)),
                ("abdomen", ell(-1.05, 0.0, 0.95, 0.55)), ("thorax", ell(0.15, 0.0, 0.55, 0.48)),
                ("head", ell(0.85, 0.0, 0.3, 0.42))]

    def _draw_fly(self, ov, cam):
        import pygame
        fly = self.fly
        fwd, left = fly.forward, fly.left
        base = fly.pos + np.array([0, 0, 0.0006])
        sc = FLY_DRAW_SCALE / 1000.0
        # halo
        ang = np.linspace(0, 2 * np.pi, 40)
        pulse = 0.5 + 0.5 * math.sin(self.t_s * 5.0)
        rr = 0.013 + 0.002 * pulse
        halo = cam.project_many(np.stack([fly.x + rr * np.cos(ang), fly.y + rr * np.sin(ang),
                                          np.full_like(ang, self.top + 0.0004)], 1))
        pygame.draw.polygon(ov, (*common.SAGE, 38), halo.tolist())
        pygame.draw.lines(ov, (*common.SAGE, 200), True, halo.tolist(), 2)
        colors = {"wing": (205, 222, 235, 150), "abdomen": (58, 46, 34, 255), "thorax": (84, 66, 44, 255),
                  "head": (178, 52, 38, 255)}
        for part, poly in self._fly_polys():
            world_pts = base[None] + sc * (poly[:, :1] * fwd[None] + poly[:, 1:] * left[None])
            pts = cam.project_many(world_pts).tolist()
            pygame.draw.polygon(ov, colors[part], pts)
            if part != "wing":
                pygame.draw.lines(ov, (20, 16, 12, 255), True, pts, 1)
            else:
                pygame.draw.lines(ov, (240, 248, 255, 200), True, pts, 1)
        if self.motor is not None and self.motor.proboscis > 1.0:            # proboscis out while MN9 fires
            ln = min(1.0, float(self.motor.proboscis) / 20.0) * 0.9 + 0.3
            a = base + sc * 1.1 * fwd
            b = base + sc * (1.1 + ln) * fwd
            pa, pb = cam.project(a), cam.project(b)
            if pa and pb:
                px_per_m = math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / max(sc * ln, 1e-9)
                width = int(np.clip(0.00012 * FLY_DRAW_SCALE * px_per_m, 4, 14))   # ~0.12 mm thick, drawn 6x
                pygame.draw.line(ov, (240, 196, 120, 255), pa, pb, width)
                pygame.draw.circle(ov, (240, 196, 120, 255), (int(pb[0]), int(pb[1])), max(2, width // 2 + 1))
        # decoder turn arc
        yd = self.yaw_applied / DEG
        if abs(yd) > 8 and not self.feeding:
            sweep = float(np.clip(yd / self.dec.yaw_max, -1, 1)) * 100 * DEG
            h = fly.heading
            th = np.linspace(h, h + sweep, 16)
            R = 0.019
            arc = cam.project_many(np.stack([fly.x + R * np.cos(th), fly.y + R * np.sin(th),
                                             np.full_like(th, self.top + 0.0006)], 1))
            pygame.draw.lines(ov, (*common.TEAL, 235), False, arc.tolist(), 5)
            end = th[-1]
            tip = np.array([fly.x + R * math.cos(end), fly.y + R * math.sin(end)])
            tang = np.sign(sweep) * np.array([-math.sin(end), math.cos(end)])
            rad_ = np.array([math.cos(end), math.sin(end)])
            tri = [tip + tang * 0.006, tip - rad_ * 0.004, tip + rad_ * 0.004]
            tp = cam.project_many(np.array([[q[0], q[1], self.top + 0.0006] for q in tri]))
            pygame.draw.polygon(ov, (*common.TEAL, 245), tp.tolist())

    def _draw_labels(self, view, cam):
        hud = self.hud
        mw, mh = view.get_size()
        hd_box = self._draw_hairdryer_label(view, cam)
        p = cam.project(self.apple + np.array([0, 0, self.apple_r + 0.012]))
        if p and 60 < p[0] < mw - 60 and 200 < p[1] < mh:        # not under the status box / banner band
            r = hud.font(22, True, True).render("APPLE", True, common.WHITE).get_rect(midbottom=(int(p[0]), int(p[1] - 4)))
            if hd_box is None or not r.colliderect(hd_box):
                hud.text(view, "APPLE", (p[0], p[1] - 4), 22, common.WHITE, bold=True, display=True, anchor="midbottom")
        d = self._dist_xy() - self.contact_r_xy
        if self.contact_t is None and d > 0.02:                # distance at the middle of the dashed line (GAME)
            u = (self.apple[:2] - self.fly.pos[:2]) / max(self._dist_xy(), 1e-9)
            mid = self.fly.pos[:2] + u * (0.012 + d) / 2
            m = cam.project([mid[0], mid[1], self.top])
            if m:
                br = self._backed_text(view, f"{d * 100:.1f} cm", m, 22, common.AMBER, anchor="center", bold=True,
                                       alpha=170, pad=(6, 2))
                hud.chip(view, "game", (br.right + 4, br.y + 3), 10)

    def _draw_hairdryer_label(self, view, cam):
        """'HAIRDRYER ON' + who holds it (the scripted holder is GAME and sees what the brain does not). Kept inside
        the view: when the hairdryer is out of shot the label sits at the edge with a pointer towards it. Returns the
        label's rect (None when not drawn)."""
        import pygame
        hud = self.hud
        mw, mh = view.get_size()
        anchor = self.nozzle - self.aim * 0.07 + np.array([0.0, 0.0, 0.045])
        if cam.depth([anchor])[0] < 0.01:
            return None
        q = cam.project_many([anchor])[0]
        inside = 0 <= q[0] <= mw and 0 <= q[1] <= mh
        if self.contact_t is not None and not inside:
            return None                                        # its part is over once the fly is at the apple
        col = common.AMBER if self.fan_on else common.MUTED
        l1 = f"HAIRDRYER {'ON' if self.fan_on else 'OFF'}"
        l2 = ("scripted holder: sees the fly's pose + the apple" if self.auto else "held by the player (mouse)")
        i1 = hud.font(20, True, True).render(l1, True, col)
        i2 = hud.font(14).render(l2, True, common.TEXT)
        chip_w = hud.font(11, True).size("GAME")[0] + 12
        w = max(i1.get_width() + 10 + chip_w, i2.get_width()) + 20
        h = i1.get_height() + i2.get_height() + 12
        x = int(np.clip(q[0] - w / 2, 30, mw - w - 30))       # room for the pointer at either edge
        y_min = 262 if self.contact_t is not None else 176     # below the status box (and the 'within reach' box)
        y = int(np.clip(q[1] - h - 6, y_min, mh - h - 48))
        back = pygame.Surface((w, h), pygame.SRCALPHA)
        back.fill((10, 12, 14, 175))
        view.blit(back, (x, y))
        pygame.draw.rect(view, col, (x, y, 4, h))
        view.blit(i1, (x + 12, y + 4))
        hud.chip(view, "game", (x + 12 + i1.get_width() + 10, y + 7), 11)
        view.blit(i2, (x + 12, y + 6 + i1.get_height()))
        box = pygame.Rect(x, y, w, h)
        if not box.inflate(40, 40).collidepoint(q[0], q[1]):   # clamped: point at the hairdryer from the box edge
            c = np.array(box.center, float)
            dvec = np.array(q, float) - c
            dvec /= max(np.linalg.norm(dvec), 1e-9)
            t_edge = min(w / 2 / max(abs(dvec[0]), 1e-9), h / 2 / max(abs(dvec[1]), 1e-9))
            base = c + dvec * (t_edge + 3)
            tip = c + dvec * (t_edge + 21)
            nrm = np.array([-dvec[1], dvec[0]])
            pygame.draw.polygon(view, col, [tip.tolist(), (base + nrm * 8).tolist(), (base - nrm * 8).tolist()])
        return box

    def _draw_status(self, s, x, y):
        import pygame
        hud = self.hud
        box = pygame.Rect(x, y, 356, 154)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((13, 16, 18, 200))
        s.blit(panel, box.topleft)
        pygame.draw.rect(s, common.LINE, box, 1, border_radius=6)
        on = self.fan_on
        a = self.alpha / DEG
        side = "left" if a > 3 else ("right" if a < -3 else "ahead")
        d = max(self._dist_xy() - self.contact_r_xy, 0.0)
        rows = [("game", "WIND AT THE FLY", f"{self.wind_v:.2f} m/s", common.WHITE if on else common.DIM),
                ("game", "FROM", "--" if not on else (f"{abs(a):.0f} deg {side}" if abs(a) < 150 else "behind"),
                 common.WHITE if on else common.DIM),
                ("game", "TO THE APPLE", "in reach" if self.feeding else f"{d * 100:.1f} cm",
                 common.SAGE if self.feeding else common.WHITE),
                ("decoder", "STEERING", "connected" if self.decoder_on else "OFF: control",
                 common.TEAL if self.decoder_on else common.RED)]
        yy = y + 10
        for kind, label, value, col in rows:
            hud.chip(s, kind, (x + 12, yy + 3), 11)
            hud.text(s, label, (x + 88, yy + 2), 15, common.MUTED, bold=True)
            hud.text(s, value, (x + 344, yy - 2), 24, col, bold=True, anchor="topright")
            yy += 36

    def _draw_banner(self, s, rect):
        import pygame
        b = self.banners.current
        if b is None:
            return
        age = self.t_s - b["t0"]
        alpha = 1.0 if age < BANNER_HOLD_S else max(0.0, 1.0 - (age - BANNER_HOLD_S) / BANNER_FADE_S)
        if alpha <= 0:
            return
        hud = self.hud
        color = b["color"]
        t1 = hud.font(50, True, True).render(b["text"], True, color)
        t2 = hud.font(20).render(b["sub"], True, common.TEXT) if b["sub"] else None
        chips = pygame.Surface((420, 24), pygame.SRCALPHA)
        cx = 0
        for kind in b["chips"]:
            cx = hud.chip(chips, kind, (cx, 0), 12).right + 8
        w = max(t1.get_width(), t2.get_width() if t2 else 0, cx) + 70
        h = 26 + t1.get_height() + (t2.get_height() + 6 if t2 else 0) + 22
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((10, 12, 14, int(205 * alpha)))
        pygame.draw.rect(box, (*color, int(255 * alpha)), (0, 0, 8, h))
        chips.set_alpha(int(255 * alpha))
        box.blit(chips, (40, 12))
        t1.set_alpha(int(255 * alpha))
        box.blit(t1, (38, 34))
        if t2:
            t2.set_alpha(int(255 * alpha))
            box.blit(t2, (40, 36 + t1.get_height()))
        dx = int(40 * max(0.0, 1 - age / 0.15))                  # a short slide-in
        lo, hi = rect.x + 388, rect.right - 196                # between the status box and the clock
        x = max(lo, min((lo + hi) // 2 - w // 2, hi - w))
        s.blit(box, (x + dx, rect.y + 16))

    # ------------------------------------------------------------------------------------------ right column
    def _trace(self, s, rect, values, vmin, vmax, color, *, fill=True, bg=True):
        """A line plot on a fixed 8 s axis (HIST_LEN ticks): the newest sample at the right edge, so the axis does
        not stretch while the history fills. `bg=False` overlays a second line on the same rect."""
        pg = self.hud.pg
        rect = pg.Rect(rect)
        if bg:
            pg.draw.rect(s, common.INSET, rect, border_radius=3)
        v = np.asarray(values, float)
        if len(v) < 2:
            return
        x = trace_x(len(v), HIST_LEN, rect.x, rect.w)
        y = rect.bottom - 1 - np.clip((v - vmin) / (vmax - vmin), 0, 1) * (rect.h - 2)
        pts = np.c_[x, y].round().astype(int).tolist()
        if fill:
            poly = [(pts[0][0], rect.bottom - 1)] + pts + [(pts[-1][0], rect.bottom - 1)]
            shade = tuple(int(c * 0.28 + b * 0.72) for c, b in zip(color, common.INSET))
            pg.draw.polygon(s, shade, poly)
        pg.draw.lines(s, color, False, pts, 2)

    def _draw_eye_panel(self, s, rect):
        hud = self.hud
        inner = hud.panel(s, rect, "What the fly sees")
        hud.text(s, "1,466 ommatidia · the radiance handed to fb.vision", (rect.right - 12, rect.y + 11), 13,
                 common.DIM, anchor="topright")
        if self.rad is not None:
            hud.mosaic(s, inner, self.eyes, common.eye_colors(self.rad, "human", exposure=2.5))

    def _draw_dn_panel(self, s, rect):
        hud = self.hud
        inner = hud.panel(s, rect, "Wind descending neurons", "connectome")
        r = self.dec.rates
        L_col, R_col = common.SAGE, common.LILAC
        hud.text(s, "DNp18", (inner.x, inner.y + 2), 16, common.MUTED, bold=True)
        hud.text(s, f"L {r['DNp18_L']:3.0f}", (inner.x + 70, inner.y - 4), 28, L_col, bold=True)
        hud.text(s, f"R {r['DNp18_R']:3.0f}", (inner.x + 190, inner.y - 4), 28, R_col, bold=True)
        hud.text(s, "Hz", (inner.x + 300, inner.y + 4), 16, common.DIM)
        hud.text(s, f"DNp33  L {r['DNp33_L']:.0f}  R {r['DNp33_R']:.0f} Hz", (inner.right, inner.y + 4), 16,
                 common.MUTED, anchor="topright")
        tr = self.hud.pg.Rect(inner.x, inner.y + 38, inner.w, inner.h - 56)
        self._trace(s, tr, self.hist["p18L"], 0, 100, L_col)
        self._trace(s, tr, self.hist["p18R"], 0, 100, R_col, fill=False, bg=False)
        hud.text(s, "100 Hz", (tr.x + 4, tr.y + 2), 12, common.DIM)
        hud.text(s, "lateral wind: DNp18 up on its side · frontal null ~15 deg R · 8 s", (inner.x, tr.bottom + 4), 13,
                 common.DIM)

    def _draw_decoder_panel(self, s, rect):
        import pygame
        hud = self.hud
        inner = hud.panel(s, rect, "Decoder: turn to the side the wind DNs report", "decoder")
        # dial: the fly points up; the arrow shows where the wind comes from (GAME geometry, not the brain)
        R = 62
        c = (inner.x + R + 8, inner.y + R + 16)
        pygame.draw.circle(s, common.INSET, c, R + 8)
        pygame.draw.circle(s, common.LINE, c, R + 8, 1)
        null = -15 * DEG                                       # measured null of the index (dev seeds)
        pygame.draw.line(s, common.MUTED, (c[0] - (R - 4) * math.sin(null), c[1] - (R - 4) * math.cos(null)),
                         (c[0] - (R + 8) * math.sin(null), c[1] - (R + 8) * math.cos(null)), 3)
        hud.text(s, "null", (c[0] - (R + 10) * math.sin(null) + 4, c[1] - (R + 10) * math.cos(null)), 12,
                 common.MUTED, anchor="bottomleft")
        body = [(c[0], c[1] - 16), (c[0] + 7, c[1] + 2), (c[0] + 5, c[1] + 14), (c[0] - 5, c[1] + 14), (c[0] - 7, c[1] + 2)]
        pygame.draw.polygon(s, common.TEXT, body)
        if self.fan_on and self.wind_v > 0:
            a = self.alpha
            src = (c[0] - R * math.sin(a), c[1] - R * math.cos(a))
            end = (c[0] - 24 * math.sin(a), c[1] - 24 * math.cos(a))
            pygame.draw.line(s, (150, 215, 255), src, end, 5)
            ux, uy = (end[0] - src[0]), (end[1] - src[1])
            n = math.hypot(ux, uy) or 1
            ux, uy = ux / n, uy / n
            tri = [(end[0] + ux * 10, end[1] + uy * 10), (end[0] - uy * 8, end[1] + ux * 8), (end[0] + uy * 8, end[1] - ux * 8)]
            pygame.draw.polygon(s, (150, 215, 255), tri)
            pygame.draw.circle(s, common.AMBER, (int(src[0]), int(src[1])), 7)
        ch = hud.chip(s, "game", (c[0] - 58, c[1] + R + 13), 10)
        hud.text(s, "wind from", (ch.right + 6, c[1] + R + 13), 13, common.DIM)
        # value and trace
        x0 = inner.x + 2 * R + 38
        yd = self.dec.yaw / DEG
        on = self.decoder_on
        if abs(yd) < 15:
            word, col = "STEADY", common.TEXT if on else common.MUTED
        else:
            word, col = ("TURN LEFT" if yd > 0 else "TURN RIGHT"), common.TEAL if on else common.MUTED
        hud.text(s, word, (x0, inner.y - 2), 26, col, bold=True, display=True)
        hud.text(s, f"{yd:+4.0f} deg/s", (inner.right, inner.y), 24, common.TEAL if on else common.MUTED, bold=True,
                 anchor="topright")
        hud.text(s, f"index u {self.dec.u_s:+5.1f} Hz", (x0, inner.y + 32), 16, common.MUTED)
        towards = turn_towards_wind(yd, self.alpha) if (self.fan_on and abs(yd) >= 15) else None
        if not on:
            hud.text(s, "not applied (control)", (inner.right, inner.y + 32), 16, common.RED, anchor="topright")
        elif towards is not None:
            hud.text(s, "towards the wind" if towards else "wind on the other side", (inner.right, inner.y + 32), 16,
                     common.TEAL if towards else common.AMBER, anchor="topright")
        tr = pygame.Rect(x0, inner.y + 58, inner.right - x0, inner.h - 76)
        pygame.draw.rect(s, common.INSET, tr, border_radius=3)
        pygame.draw.line(s, common.LINE, (tr.x, tr.centery), (tr.right, tr.centery), 1)
        for sgn in (-1, 1):                                    # the decoder's cap
            yc = tr.centery - sgn * (tr.h / 2 - 2) / 1.25
            for xx in range(tr.x, tr.right, 12):
                pygame.draw.line(s, common.DIM, (xx, yc), (min(xx + 5, tr.right), yc), 1)
        v = np.asarray(self.hist["yaw"], float)
        if len(v) > 1:
            xs = trace_x(len(v), HIST_LEN, tr.x, tr.w)
            ys = tr.centery - np.clip(v / (1.25 * self.dec.yaw_max), -1, 1) * (tr.h / 2 - 2)
            pygame.draw.lines(s, common.TEAL if on else common.MUTED, False,
                              np.c_[xs, ys].round().astype(int).tolist(), 2)
        for txt, pos, anchor in (("left", (tr.x + 4, tr.y + 2), "topleft"), ("right", (tr.x + 4, tr.bottom - 16), "topleft"),
                                 (f"dashes: cap +-{self.dec.yaw_max:g} deg/s · 8 s", (tr.right - 4, tr.bottom - 16),
                                  "topright")):
            img = hud.font(12).render(txt, True, common.DIM)
            r = img.get_rect(**{anchor: pos})
            pygame.draw.rect(s, common.INSET, r.inflate(4, 0))
            s.blit(img, r)

    def _draw_mn9_panel(self, s, rect):
        import pygame
        hud = self.hud
        inner = hud.panel(s, rect, "Proboscis motor neuron MN9", "connectome")
        mn9 = float(self.hist["mn9"][-1]) if self.hist["mn9"] else 0.0
        gf = float(self.hist["gf"][-1]) if self.hist["gf"] else 0.0
        touching = self.stats["mn9_contact"]
        if self.feeding and touching:                         # the mean since contact: a steadier number than 2 cells
            mean = float(np.mean(touching))
            hud.text(s, f"{mean:4.1f} Hz", (inner.x, inner.y - 4), 30, common.SAGE, bold=True)
        else:
            hud.text(s, f"{mn9:4.1f} Hz", (inner.x, inner.y - 4), 30, common.SAGE if mn9 > 1 else common.TEXT,
                     bold=True)
        sugar = "SUGAR GRNs 120 Hz" if self.tasting else "no sugar contact"
        hud.chip(s, "game", (inner.x + 150, inner.y + 3), 11)
        hud.text(s, sugar, (inner.x + 212, inner.y + 2), 16, common.SAGE if self.tasting else common.DIM, bold=True)
        tr = pygame.Rect(inner.x, inner.y + 38, inner.w - 166, inner.h - 40)
        self._trace(s, tr, self.hist["mn9"], 0, 40, common.SAGE)
        if self.feeding and touching:
            hud.text(s, f"mean over {len(touching) * DT_S:.1f} s of contact · now {mn9:.0f} Hz", (tr.x + 6, tr.y + 4),
                     13, common.MUTED)
        else:
            hud.text(s, "0-40 Hz · 8 s", (tr.x + 6, tr.y + 4), 12, common.DIM)
        # the giant fibre (DNp01), with the body model's 33 Hz escape threshold
        g = pygame.Rect(tr.right + 14, tr.y - 4, inner.right - tr.right - 14, tr.h + 4)
        hot = gf >= 33
        hud.text(s, "GIANT FIBRE", (g.x, g.y), 13, common.MUTED, bold=True)
        hud.text(s, f"{gf:.0f} Hz", (g.right, g.y + 16), 26, common.RED if hot else common.TEXT, bold=True,
                 anchor="topright")
        track = pygame.Rect(g.x, g.y + 50, g.w, 10)
        pygame.draw.rect(s, common.INSET, track, border_radius=3)
        frac = float(np.clip(gf / 60.0, 0, 1))
        if frac > 0:
            pygame.draw.rect(s, common.RED if hot else common.MUTED, (track.x, track.y, max(2, round(track.w * frac)),
                                                                        track.h), border_radius=3)
        xt = track.x + round(track.w * 33 / 60)
        pygame.draw.line(s, common.RED, (xt, track.y - 4), (xt, track.bottom + 3), 2)
        hud.text(s, "33 Hz -> jump (body)", (g.x, track.bottom + 6), 12, common.DIM)

    # ------------------------------------------------------------------------------------------ bookkeeping
    def brains(self):
        return {"brain": self.fb}

    def finished(self):
        return False


def main(argv=None):
    ap = common.standard_args(__doc__.split("\n")[0], seconds=32.0)
    ap.add_argument("--decoder", choices=("on", "off"), default="on",
                    help="off = control: the decoder is attached and logged but its yaw is not applied")
    ap.add_argument("--gain", type=float, default=3.0, help="decoder gain, deg/s of yaw per Hz of the wind index")
    ap.add_argument("--tau", type=float, default=0.15, help="decoder smoothing time constant, s")
    ap.add_argument("--yaw-max", type=float, default=60.0, help="decoder yaw cap, deg/s (see the caption: faster self-turns near the apple fire the GF)")
    ap.add_argument("--fan-on", type=float, default=FAN_ON_S, help="record mode: when the holder switches the fan on, s")
    ap.add_argument("--render-scale", type=int, default=0, help="main view rendered at 1/N resolution (0 = auto)")
    ap.add_argument("--start", default=None, metavar="X,Y,HEADING",
                    help=f"fly start on the table, m and deg (default {START[0]},{START[1]},{START[2]:g})")
    args = ap.parse_args(argv)
    game = HairdryerGame(args)
    return common.run(game, args)


if __name__ == "__main__":
    main()
