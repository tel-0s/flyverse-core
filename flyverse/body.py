"""Fly body: pose on the table + a transparent motor readout from named neuron groups.

The readout is deliberately simple and *named* (no hidden decoder): walking speed and turning are read
from the classic locomotor descending neurons and from the leg motor neurons in the VNC; proboscis
extension from MN9. All rates are exponentially-smoothed firing rates from the brain (Hz).

    forward  = baseline + k_fwd * (DNp09 + DNa01 + DNa03 + DNb01 + DNa04) + k_leg * leg MNs - k_back * MDN
    turn     = -k_opto * (opto_L - opto_R) - k_turn * (DNa02_R - DNa02_L) + k_leg * (legMN_L - legMN_R)
    proboscis = MN9 rate                                            (positive turn = left)

`opto` = DNp04 + LPT27 + LPT30, the descending / lobula-plate-tangential types whose left-right
asymmetry *flips with rotation direction* in this model (scripts/benchmark.py rotation test): during a
leftward rotation of the fly the left cells fire, so the optomotor (course-stabilising) response turns
the fly right -- hence the minus sign. DNa02 gets 80% of its input from the central-complex steering
output (PFL3/LAL/PS), so it is read as a goal-steering command (ipsilateral turn, Rayshubskiy 2020).
These are model-derived readouts, not established mappings (doomfly used DNp20/DNpe017, arbitrary).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .motor import MotorGroups, WingGroups, MotorRates, motor_groups, wing_groups, read_motor, LH_ODOUR_TYPES


@dataclass
class FlyState:
    """Pose. On a surface the source of truth is (fwd, normal): forward vector in the face's plane and the
    face's outward normal (`surfaces.py`); heading / pitch / roll are derived for display and for the
    flight integrator, which owns them while airborne. `heading = h` on a surface rotates fwd about the
    normal to azimuth h where that is defined (horizontal faces), and is a no-op on vertical faces."""
    x: float = 0.0            # m, world frame
    y: float = 0.0
    z: float = 0.75           # height of the walking surface
    pitch: float = 0.0        # rad, positive = nose up (free in flight)
    roll: float = 0.0         # rad, positive = right wing down (banking)
    speed: float = 0.0        # m/s
    yaw_rate: float = 0.0     # rad/s
    proboscis: float = 0.0    # 0..1
    eye_height: float = 0.0012
    airborne: bool = False
    vx: float = 0.0           # m/s, world frame (used when airborne)
    vy: float = 0.0
    vz: float = 0.0
    air_time: float = 0.0
    ground_time: float = 1e9  # seconds since the last landing (escape refractory)
    fwd: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    face: object = None       # surfaces.Face the fly stands on (None = legacy flat plane)
    _heading: float = 0.0     # flight azimuth (rad); on a surface, the azimuth of fwd
    # walking over an edge: the face changes at once, the body rotates over `edge_len` of travel (a real
    # fly takes a couple of body lengths to bend over an edge; an instantaneous 90 deg swing of the scene
    # was firing the giant fibre at every edge -- NOTES, session 8)
    _edge_axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _edge_theta: float = 0.0  # signed rotation from the old frame to the new (rad), 0 = no blend
    _edge_left: float = 0.0   # metres of travel left in the blend
    edge_len: float = 0.004

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def set_pos(self, p) -> None:
        self.x, self.y, self.z = (float(v) for v in p)

    @property
    def heading(self) -> float:
        if self.airborne or abs(self.fwd[2]) > 0.999:
            return self._heading
        return float(np.arctan2(self.fwd[1], self.fwd[0]))

    @heading.setter
    def heading(self, h: float) -> None:
        self._heading = float(h)
        if not self.airborne and abs(self.normal[2]) > 0.5:          # horizontal face: rotate fwd to azimuth h
            self.fwd = np.array([np.cos(h), np.sin(h), 0.0])

    def place(self, x: float, y: float, z: float, heading: float = 0.0, normal=(0.0, 0.0, 1.0), face=None) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.normal = np.array(normal, float); self.face = face; self.airborne = False
        self.pitch = self.roll = 0.0; self.vx = self.vy = self.vz = 0.0
        self._heading = float(heading); self._edge_left = 0.0; self._edge_theta = 0.0
        if abs(self.normal[2]) > 0.5:
            self.fwd = np.array([np.cos(heading), np.sin(heading), 0.0])
        else:                                                        # a wall: forward = up the wall
            self.fwd = np.array([0.0, 0.0, 1.0])

    def _sensed(self, v: np.ndarray) -> np.ndarray:
        """A body-frame vector as the fly currently holds it: mid-rotation over an edge if a blend is active."""
        if self._edge_left <= 0.0 or self._edge_theta == 0.0:
            return v
        ang = -self._edge_theta * (self._edge_left / self.edge_len)     # remaining rotation, applied backwards
        k = self._edge_axis; c, s_ = np.cos(ang), np.sin(ang)
        return v * c + np.cross(k, v) * s_ + k * np.dot(k, v) * (1 - c)   # Rodrigues

    def begin_edge(self, n_from: np.ndarray, n_to: np.ndarray) -> None:
        axis = np.cross(n_from, n_to); norm = np.linalg.norm(axis)
        if norm < 1e-9:
            self._edge_theta = 0.0; self._edge_left = 0.0; return
        self._edge_axis = axis / norm
        self._edge_theta = float(np.arctan2(norm, np.dot(n_from, n_to)))
        self._edge_left = self.edge_len

    @property
    def forward(self) -> np.ndarray:
        if not self.airborne:
            return self._sensed(self.fwd)
        cp = np.cos(self.pitch)
        return np.array([cp * np.cos(self._heading), cp * np.sin(self._heading), np.sin(self.pitch)])

    @property
    def left(self) -> np.ndarray:
        if not self.airborne:
            return np.cross(self.up, self.forward)
        f = self.forward
        l0 = np.array([-np.sin(self._heading), np.cos(self._heading), 0.0])
        u0 = np.cross(f, l0)
        return np.cos(self.roll) * l0 + np.sin(self.roll) * u0

    @property
    def up(self) -> np.ndarray:
        if not self.airborne:
            return self._sensed(self.normal)
        return np.cross(self.forward, self.left)

    def body_to_world(self, dirs_body: np.ndarray) -> np.ndarray:
        """(…, 3) body-frame directions (x fwd, y left, z up) -> world frame."""
        R = np.stack([self.forward, self.left, self.up], axis=1)   # columns = body axes (full 3-D orientation)
        return dirs_body @ R.T

    @property
    def eye_pos(self) -> np.ndarray:
        return self.pos + self.eye_height * self.up

    def flight_frame_from_surface(self) -> None:
        """Entering the air: express the surface pose as heading / pitch / roll for the flight integrator."""
        f, n = self.fwd, self.normal
        self._heading = float(np.arctan2(f[1], f[0])) if abs(f[2]) < 0.999 else self._heading
        self.pitch = float(np.arcsin(np.clip(f[2], -1, 1)))
        l0 = np.array([-np.sin(self._heading), np.cos(self._heading), 0.0]); u0 = np.cross(self.forward_air(), l0)
        left = np.cross(n, f)
        self.roll = float(np.arctan2(np.dot(left, u0), np.dot(left, l0)))

    def forward_air(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        return np.array([cp * np.cos(self._heading), cp * np.sin(self._heading), np.sin(self.pitch)])

    def __post_init__(self):
        if isinstance(self.face, dict):                              # loaded from a saved state: re-resolved on the next step
            self.face = None
        self.fwd = np.asarray(self.fwd, float); self.normal = np.asarray(self.normal, float)


_flystate_init = FlyState.__init__


def _flystate_init_with_heading(self, *args, heading=None, **kw):   # FlyState(heading=...) keeps working
    _flystate_init(self, *args, **kw)
    if heading is not None:
        self.heading = heading


FlyState.__init__ = _flystate_init_with_heading


@dataclass
class Metabolism:
    """Energy state that closes the foraging loop. Energy drains with time (faster while walking), refills
    while feeding; hunger = 1 - energy scales the odour-gated upwind drive (starved flies are more
    responsive to food odours) and satiety ends a meal (a sated fly leaves the fruit and wanders).
    Time constants are demo-scale: a full tank lasts ~4 min of walking, a meal takes ~15 s."""
    energy: float = 0.6
    drain_per_s: float = 1.0 / 300.0       # resting drain
    walk_drain_per_m: float = 0.15         # extra drain per metre walked
    feed_per_s: float = 1.0 / 15.0
    satiety: float = 0.95                  # stop feeding above this
    resume_below: float = 0.7              # ... and only start a new meal below this (hysteresis)
    sated: bool = False
    meals: int = 0
    _feeding_prev: bool = False

    @property
    def hunger(self) -> float:
        return float(np.clip(1.0 - self.energy, 0.0, 1.0))

    def update(self, tasting: bool, speed: float, dt_s: float) -> bool:
        """Advance; returns True if the fly is feeding this step."""
        self.energy -= self.drain_per_s * dt_s + self.walk_drain_per_m * abs(speed) * dt_s
        if self.sated and self.energy < self.resume_below:
            self.sated = False
        feeding = bool(tasting) and not self.sated
        if feeding:
            self.energy += self.feed_per_s * dt_s
            if not self._feeding_prev:
                self.meals += 1
            if self.energy >= self.satiety:
                self.sated = True; feeding = False
        self._feeding_prev = feeding
        self.energy = float(np.clip(self.energy, 0.0, 1.0))
        return feeding

    @property
    def state(self) -> str:
        return "sated" if self.sated else ("starving" if self.energy < 0.15 else "hungry" if self.energy < 0.5 else "ok")


@dataclass
class Locomotion:
    """Walking: a physical readout of the motor signals in `MotorRates`. No behaviour lives here -- the
    odour-gated steering, casting, hunger and search programs are in `programs.py` and are off by
    default. The two filters below are readout corrections, not behaviour: the optomotor term is
    high-passed (DNp04's right side carries a chronic bias from the model's asymmetric eye), and MDN
    only counts above a threshold (it fires a few Hz from self-motion optic flow)."""
    k_fwd: float = 0.02 / 40.0     # m/s per Hz of forward-DN mean rate (40 Hz -> 2 cm/s, a brisk fly walk)
    k_leg: float = 0.02 / 60.0     # m/s per Hz of mean leg-MN rate
    k_back: float = 0.02 / 40.0
    k_turn: float = np.deg2rad(200) / 40.0   # rad/s per Hz of DNa02 asymmetry (ipsilateral turning, Rayshubskiy 2020)
    k_opto: float = np.deg2rad(150) / 15.0   # rad/s per Hz of DNp04/LPT asymmetry (optomotor, stabilising)
    opto_hp_tau_s: float = 2.0
    k_leg_turn: float = np.deg2rad(100) / 30.0
    max_speed: float = 0.03
    max_yaw: float = np.deg2rad(400)
    edge_turn: float = np.deg2rad(90)   # turn rate towards the interior while the fly is against an edge / wall
    tau_ms: float = 80.0           # motor smoothing
    baseline_speed: float = 0.008  # m/s intrinsic walking drive (flies walk spontaneously; keeps optic flow alive)
    mdn_threshold: float = 15.0    # Hz

    def readout(self, motor, g: MotorGroups | None = None, dt_s: float = 0.01) -> dict:
        # Legacy (Brain, MotorGroups) probes use the same raw-rate adapter.
        motor = read_motor(motor, groups=g) if g is not None else motor
        fwd, back = motor.fwd_dn, motor.back_dn
        tL, tR = motor.turn_L, motor.turn_R
        oL, oR = motor.opto_L, motor.opto_R
        lL, lR = motor.leg_L, motor.leg_R
        asym = oL - oR
        a_hp = np.exp(-dt_s / self.opto_hp_tau_s)
        self._opto_bias = a_hp * getattr(self, "_opto_bias", asym) + (1 - a_hp) * asym
        opto = asym - self._opto_bias
        back_eff = max(back - self.mdn_threshold, 0.0) if np.isscalar(back) else np.maximum(back - self.mdn_threshold, 0.0)
        speed = self.baseline_speed + self.k_fwd * fwd + self.k_leg * 0.5 * (lL + lR) - self.k_back * back_eff
        yaw = -self.k_opto * opto - self.k_turn * (tR - tL) + self.k_leg_turn * (lL - lR)
        return {"speed": float(np.clip(speed, -self.max_speed, self.max_speed)),
                "yaw": float(np.clip(yaw, -self.max_yaw, self.max_yaw)),
                "proboscis": float(np.clip(motor.proboscis / 30.0, 0, 1)),
                "rates": {"fwdDN": fwd, "MDN": back, "opto_L": oL, "opto_R": oR, "wind_L": motor.wind_ipsi_L, "wind_R": motor.wind_ipsi_R,
                          "LH odour": motor.lh_odour, "DNa02_L": tL, "DNa02_R": tR, "legMN_L": lL, "legMN_R": lR, "MN9": motor.proboscis}}

    def step(self, fly: FlyState, cmd: dict, dt_s: float, bounds) -> None:
        """Walk. `bounds` is a surfaces.Surfaces (the fly sticks to faces, walks over edges and up walls) or,
        for legacy callers, an (x0, x1, y0, y1) plane with contact-mediated edges."""
        if isinstance(cmd, MotorRates):
            cmd = self.readout(cmd, dt_s=dt_s)
        a = np.exp(-dt_s * 1000 / self.tau_ms)
        fly.speed = a * fly.speed + (1 - a) * cmd["speed"]
        fly.yaw_rate = a * fly.yaw_rate + (1 - a) * cmd["yaw"]
        fly.proboscis = cmd["proboscis"]
        if hasattr(bounds, "walk"):
            n_before = fly.normal.copy()
            p, f, n, face = bounds.walk(fly.pos, fly.fwd, fly.normal, fly.face, fly.speed * dt_s, fly.yaw_rate * dt_s)
            fly.set_pos(p); fly.fwd, fly.normal, fly.face = f, n, face
            if np.dot(n, n_before) < 0.99:
                fly.begin_edge(n_before, n)
            elif fly._edge_left > 0.0:
                fly._edge_left = max(0.0, fly._edge_left - abs(fly.speed) * dt_s)
            if abs(f[2]) < 0.999:
                fly._heading = float(np.arctan2(f[1], f[0]))
            return
        fly.heading = fly.heading + fly.yaw_rate * dt_s
        nx = fly.x + fly.speed * dt_s * np.cos(fly.heading)
        ny = fly.y + fly.speed * dt_s * np.sin(fly.heading)
        x0, x1, y0, y1 = bounds
        in_x, in_y = x0 <= nx <= x1, y0 <= ny <= y1
        if in_x and in_y:
            fly.x, fly.y = nx, ny
        else:
            if in_x: fly.x = nx
            if in_y: fly.y = ny
            fly.x = float(np.clip(fly.x, x0, x1)); fly.y = float(np.clip(fly.y, y0, y1))
            to_centre = np.arctan2(0.5 * (y0 + y1) - fly.y, 0.5 * (x0 + x1) - fly.x)
            err = (to_centre - fly.heading + np.pi) % (2 * np.pi) - np.pi
            fly.heading = fly.heading + np.sign(err) * min(abs(err), self.edge_turn * dt_s)
        fly.heading = (fly.heading + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------- flight
@dataclass
class Flight:
    """Airborne dynamics. Takeoff = a giant-fibre spike (escape jump: ballistic hop, wings may or may not
    engage) or sustained wing-power motor neuron activity (voluntary takeoff). In the air, thrust and lift
    come from the power MNs, yaw from steering-MN asymmetry; without wingbeat the fly falls and lands."""
    # In the air the body has all three rotational degrees of freedom: yaw from the steering-MN
    # asymmetry, pitch from the flight path (climb/dive), roll = banking into the turn. On flat
    # surfaces pitch and roll are 0.
    jump_speed: float = 0.6            # m/s initial escape-jump velocity (Drosophila jumps ~0.5-1 m/s)
    jump_pitch_deg: float = 45.0
    k_thrust: float = 0.8 / 60.0       # m/s of airspeed per Hz of mean power-MN rate (60 Hz -> 0.8 m/s)
    hover_hz: float = 20.0             # power-MN rate needed to hold altitude
    k_lift: float = 1.5 / 40.0         # m/s^2 of vertical acceleration per Hz above hover
    gravity: float = 3.0               # m/s^2, reduced: a fly's terminal velocity is low (drag)
    drag: float = 2.5                  # 1/s
    k_yaw: float = np.deg2rad(600) / 30.0
    max_yaw: float = np.deg2rad(900)
    bank_per_yaw: float = 0.12         # rad of roll per rad/s of yaw rate (banked turns), clipped to 60 deg
    k_roll: float = np.deg2rad(40) / 30.0   # rad of commanded roll per Hz of steering-MN asymmetry (L - R)
    pitch_follow: float = 0.5          # how much of the flight-path angle the nose follows (haltere reflex levels the rest)
    max_pitch: float = np.deg2rad(40)
    orient_tau_ms: float = 120.0       # pitch/roll settle with this time constant (haltere-mediated stabilisation)
    takeoff_power_hz: float = 50.0     # sustained power-MN rate that launches a voluntary takeoff
    takeoff_hold_s: float = 0.3        # ... sustained for this long (a wingbeat command, not a flicker)
    # smoothed GF rate that counts as an escape. Calibrated from data (NOTES, session 8): per-second maxima of the
    # walking GF have median 20 Hz, 90th percentile 28, 99th 32; real loom bursts are 41-67 Hz. 38 sits above the
    # 99th percentile and below the weakest loom (was 30, set when the walking GF peaked at 26). Habituation
    # lives in programs.EscapeGating.
    gf_hz: float = 38.0
    tau_ms: float = 40.0

    def readout(self, motor, wg: WingGroups | None = None, dt_s: float = 0.01) -> dict:
        motor = read_motor(motor, wings=wg) if wg is not None else motor
        out = {name: float(getattr(motor, name)) for name in ("gf", "ttm", "power", "steer_L", "steer_R", "haltere")}
        out["gf_threshold"] = self.gf_hz
        return out

    landing_refractory_s: float = 1.0  # no new escape within this time of landing (a jump-land-jump chain is not fly behaviour)

    def maybe_takeoff(self, fly: FlyState, w: dict, dt_s: float = 0.01) -> bool:
        if isinstance(w, MotorRates):
            w = self.readout(w)
        fly.ground_time += dt_s
        if w["gf"] >= w.get("gf_threshold", self.gf_hz) and fly.ground_time >= self.landing_refractory_s:
            # the giant fibre spike is the escape trigger (TTMn follows it 1:1 in the animal)
            self.launch(fly, escape=True); return True
        # voluntary takeoff needs sustained wingbeat drive (0.1 s), not a transient
        self._power_hold = getattr(self, "_power_hold", 0.0) + dt_s if w["power"] >= self.takeoff_power_hz else 0.0
        if self._power_hold >= self.takeoff_hold_s:
            self._power_hold = 0.0
            self.launch(fly, escape=False); return True
        return False

    def launch(self, fly: FlyState, escape: bool) -> None:
        v = self.jump_speed if escape else 0.2
        pitch = np.deg2rad(self.jump_pitch_deg if escape else 30)
        d = np.cos(pitch) * fly.forward + np.sin(pitch) * fly.up      # in the surface frame: "up" is the face normal
        fly.set_pos(fly.pos + 0.003 * fly.up)
        fly.flight_frame_from_surface()
        fly.airborne = True
        fly.vx, fly.vy, fly.vz = (float(c) for c in v * d)
        fly.air_time = 0.0

    def step(self, fly: FlyState, w: dict, dt_s: float, surface_z, room: tuple) -> None:
        """Integrate one airborne step. `surface_z` is a surfaces.Surfaces (land on the first face the path
        crosses) or, for legacy callers, a callable (x, y) -> z of what is below; room = (x0,x1,y0,y1,z1)."""
        if isinstance(w, MotorRates):
            w = self.readout(w)
        a = np.exp(-dt_s * 1000 / self.tau_ms)
        yaw = np.clip(-self.k_yaw * (w["steer_R"] - w["steer_L"]), -self.max_yaw, self.max_yaw)
        fly.yaw_rate = a * fly.yaw_rate + (1 - a) * yaw
        fly._heading += fly.yaw_rate * dt_s
        thrust = self.k_thrust * w["power"]
        f = fly.forward
        ax = (thrust * f[0] - self.drag * fly.vx)
        ay = (thrust * f[1] - self.drag * fly.vy)
        az = self.k_lift * (w["power"] - self.hover_hz) - self.gravity if w["power"] > 1 else -self.gravity
        az -= self.drag * fly.vz
        fly.vx += ax * dt_s; fly.vy += ay * dt_s; fly.vz += az * dt_s
        p_prev = fly.pos
        fly.x += fly.vx * dt_s; fly.y += fly.vy * dt_s; fly.z += fly.vz * dt_s
        fly.air_time += dt_s
        if hasattr(surface_z, "land"):
            hit = surface_z.land(p_prev, fly.pos)
            if hit is None and surface_z.in_solid(fly.pos):          # safety: never inside a solid or outside the room
                hit = surface_z.snap(fly.pos)
            if hit is not None:
                point, face = hit
                v = np.array([fly.vx, fly.vy, fly.vz]); n = face.normal
                fwd = v - n * np.dot(v, n)
                if np.linalg.norm(fwd) < 1e-3:
                    fwd = fly.forward - n * np.dot(fly.forward, n)
                if np.linalg.norm(fwd) < 1e-3:
                    fwd = np.cross(n, np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0]))
                fly.set_pos(point); fly.fwd = fwd / np.linalg.norm(fwd); fly.normal = n; fly.face = face
                fly.airborne = False; fly.pitch = fly.roll = 0.0; fly.vx = fly.vy = fly.vz = 0.0
                fly.ground_time = 0.0; fly._edge_left = 0.0; fly._edge_theta = 0.0
                if abs(fly.fwd[2]) < 0.999:
                    fly._heading = float(np.arctan2(fly.fwd[1], fly.fwd[0]))
            else:
                ao = np.exp(-dt_s * 1000 / self.orient_tau_ms)
                horiz = float(np.hypot(fly.vx, fly.vy))
                pitch_t = float(np.clip(self.pitch_follow * np.arctan2(fly.vz, max(horiz, 0.05)), -self.max_pitch, self.max_pitch))
                roll_t = float(np.clip(-self.bank_per_yaw * fly.yaw_rate + self.k_roll * (w["steer_L"] - w["steer_R"]), -np.deg2rad(60), np.deg2rad(60)))
                fly.pitch = ao * fly.pitch + (1 - ao) * pitch_t
                fly.roll = ao * fly.roll + (1 - ao) * roll_t
            return
        # orientation in the air. Roll: commanded by the wing steering-muscle asymmetry (a banked turn is
        # what an asymmetric stroke produces) plus the bank that goes with the yaw rate; pitch: the nose
        # follows part of the flight-path angle. Both relax towards level -- the haltere-mediated
        # stabilising reflexes that keep a real fly upright -- with `orient_tau_ms`.
        ao = np.exp(-dt_s * 1000 / self.orient_tau_ms)
        horiz = float(np.hypot(fly.vx, fly.vy))
        pitch_t = float(np.clip(self.pitch_follow * np.arctan2(fly.vz, max(horiz, 0.05)), -self.max_pitch, self.max_pitch))
        roll_t = float(np.clip(-self.bank_per_yaw * fly.yaw_rate + self.k_roll * (w["steer_L"] - w["steer_R"]), -np.deg2rad(60), np.deg2rad(60)))
        fly.pitch = ao * fly.pitch + (1 - ao) * pitch_t
        fly.roll = ao * fly.roll + (1 - ao) * roll_t
        x0, x1, y0, y1, z1 = room
        if not (x0 < fly.x < x1):   # walls: land on the wall = stop, drop to the floor below
            fly.x = np.clip(fly.x, x0 + 0.01, x1 - 0.01); fly.vx = 0.0
        if not (y0 < fly.y < y1):
            fly.y = np.clip(fly.y, y0 + 0.01, y1 - 0.01); fly.vy = 0.0
        if fly.z > z1 - 0.01:
            fly.z = z1 - 0.01; fly.vz = min(fly.vz, 0.0)
        ground = surface_z(fly.x, fly.y)
        if fly.z <= ground and fly.vz <= 0 and fly.air_time > 0.05:
            fly.z = ground; fly.airborne = False; fly.ground_time = 0.0
            fly.normal = np.array([0.0, 0.0, 1.0]); fly.fwd = np.array([np.cos(fly._heading), np.sin(fly._heading), 0.0])
            fly.pitch = 0.0; fly.roll = 0.0                            # landed on a flat surface
            fly.ground_time = 0.0
            fly.speed = float(np.hypot(fly.vx, fly.vy)) * 0.2   # landing: most momentum lost
            fly.vx = fly.vy = fly.vz = 0.0
