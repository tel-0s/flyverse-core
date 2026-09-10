"""Fly body: pose on the table + a transparent motor readout from named neuron groups.

The readout is deliberately simple and *named* (no hidden decoder): walking speed and turning are read
from the classic locomotor descending neurons and from the leg motor neurons in the VNC; proboscis
extension from MN9. All rates are exponentially-smoothed firing rates from the brain (Hz).

    forward  = k_fwd  * (DNp09 + DNa01 + DNa03 + DNb01 + leg MN total)/... - k_back * MDN
    turn     = k_turn * (DNa02_R - DNa02_L) + k_leg * (legMN_L - legMN_R)   (positive = turn left)
    proboscis = MN9 rate

These are hypotheses about what the connectome's outputs mean, not established mappings (doomfly used
DNp20/DNpe017, which they themselves call arbitrary).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectome import Connectome


@dataclass
class MotorGroups:
    fwd_dn: np.ndarray
    back_dn: np.ndarray
    turn_L: np.ndarray
    turn_R: np.ndarray
    leg_L: np.ndarray
    leg_R: np.ndarray
    proboscis: np.ndarray
    names: dict = field(default_factory=dict)


def motor_groups(c: Connectome) -> MotorGroups:
    fwd_types = ["DNp09", "DNa01", "DNa03", "DNb01", "DNa04"]
    g = MotorGroups(
        fwd_dn=c.select(type=fwd_types),
        back_dn=c.select(type="MDN"),
        turn_L=c.select(type="DNa02", somaSide="L"),
        turn_R=c.select(type="DNa02", somaSide="R"),
        leg_L=c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"], somaSide="L"),
        leg_R=c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"], somaSide="R"),
        proboscis=c.select(type="MN9"),
    )
    g.names = {"fwd_dn": fwd_types, "back_dn": ["MDN"], "turn": ["DNa02 L/R"], "leg": ["leg MNs L/R"], "proboscis": ["MN9"]}
    return g


@dataclass
class FlyState:
    x: float = 0.0            # m, world frame
    y: float = 0.0
    z: float = 0.75           # height of the walking surface
    heading: float = 0.0      # rad, 0 = +x, positive = counter-clockwise (left)
    speed: float = 0.0        # m/s
    yaw_rate: float = 0.0     # rad/s
    proboscis: float = 0.0    # 0..1
    eye_height: float = 0.0012
    airborne: bool = False
    vx: float = 0.0           # m/s, world frame (used when airborne)
    vy: float = 0.0
    vz: float = 0.0
    air_time: float = 0.0

    @property
    def forward(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading), 0.0])

    @property
    def left(self) -> np.ndarray:
        return np.array([-np.sin(self.heading), np.cos(self.heading), 0.0])

    def body_to_world(self, dirs_body: np.ndarray) -> np.ndarray:
        """(…, 3) body-frame directions (x fwd, y left, z up) -> world frame."""
        R = np.stack([self.forward, self.left, np.array([0, 0, 1.0])], axis=1)   # columns = body axes
        return dirs_body @ R.T

    @property
    def eye_pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z + self.eye_height])


@dataclass
class Locomotion:
    k_fwd: float = 0.02 / 40.0     # m/s per Hz of forward-DN mean rate (40 Hz -> 2 cm/s, a brisk fly walk)
    k_leg: float = 0.02 / 60.0     # m/s per Hz of mean leg-MN rate
    k_back: float = 0.02 / 40.0
    k_turn: float = np.deg2rad(200) / 40.0   # rad/s per Hz of DNa02 asymmetry
    k_leg_turn: float = np.deg2rad(100) / 30.0
    max_speed: float = 0.03
    max_yaw: float = np.deg2rad(400)
    tau_ms: float = 80.0           # motor smoothing
    baseline_speed: float = 0.004  # m/s intrinsic walking drive (flies walk spontaneously; keeps optic flow alive)

    def readout(self, brain, g: MotorGroups) -> dict:
        r = brain.mean_rate
        fwd = r(g.fwd_dn); back = r(g.back_dn)
        tL, tR = r(g.turn_L), r(g.turn_R)
        lL, lR = r(g.leg_L), r(g.leg_R)
        prob = r(g.proboscis)
        speed = self.baseline_speed + self.k_fwd * fwd + self.k_leg * 0.5 * (lL + lR) - self.k_back * back
        # convention: DNa02 drives ipsilateral turning (Rayshubskiy et al. 2020): right DNa02 -> turn right
        yaw = -self.k_turn * (tR - tL) + self.k_leg_turn * (lL - lR)
        return {"speed": float(np.clip(speed, -self.max_speed, self.max_speed)),
                "yaw": float(np.clip(yaw, -self.max_yaw, self.max_yaw)),
                "proboscis": float(np.clip(prob / 30.0, 0, 1)),
                "rates": {"fwdDN": fwd, "MDN": back, "DNa02_L": tL, "DNa02_R": tR, "legMN_L": lL, "legMN_R": lR, "MN9": prob}}

    def step(self, fly: FlyState, cmd: dict, dt_s: float, bounds: tuple) -> None:
        a = np.exp(-dt_s * 1000 / self.tau_ms)
        fly.speed = a * fly.speed + (1 - a) * cmd["speed"]
        fly.yaw_rate = a * fly.yaw_rate + (1 - a) * cmd["yaw"]
        fly.proboscis = cmd["proboscis"]
        fly.heading += fly.yaw_rate * dt_s
        nx = fly.x + fly.speed * dt_s * np.cos(fly.heading)
        ny = fly.y + fly.speed * dt_s * np.sin(fly.heading)
        x0, x1, y0, y1 = bounds
        if x0 <= nx <= x1 and y0 <= ny <= y1:   # stay on the table (flies rarely walk off edges)
            fly.x, fly.y = nx, ny
        else:
            fly.heading += np.pi / 2 * dt_s * 8   # nudge away from the edge


# ---------------------------------------------------------------------------- flight
@dataclass
class WingGroups:
    gf: np.ndarray          # giant fibre DNp01 (escape takeoff)
    ttm: np.ndarray         # TTMn: tergotrochanteral "jump" muscle motor neurons
    power: np.ndarray       # DLMn + DVMn: indirect flight power muscles (wingbeat)
    steer_L: np.ndarray     # direct steering muscle MNs (b1-3, i1-2, iii1/3, hg1-4, ps1-2, tp1-2, tpn) left
    steer_R: np.ndarray
    haltere: np.ndarray


def wing_groups(c: Connectome) -> WingGroups:
    steer = c.select(superclass="vnc_motor", subclass="wm", type="~^(b[123]|i[12]|iii[13]|hg[1-4]|ps[12]|tp[12]|tpn) MN$")
    side = c.neurons.somaSide.to_numpy()
    return WingGroups(
        gf=c.select(type="DNp01"),
        ttm=c.select(type="TTMn"),
        power=c.select(type="~^(DLMn|DVMn)"),
        steer_L=steer[side[steer] == "L"], steer_R=steer[side[steer] == "R"],
        haltere=c.select(superclass="vnc_motor", subclass="hm"),
    )


@dataclass
class Flight:
    """Airborne dynamics. Takeoff = a giant-fibre spike (escape jump: ballistic hop, wings may or may not
    engage) or sustained wing-power motor neuron activity (voluntary takeoff). In the air, thrust and lift
    come from the power MNs, yaw from steering-MN asymmetry; without wingbeat the fly falls and lands."""
    jump_speed: float = 0.6            # m/s initial escape-jump velocity (Drosophila jumps ~0.5-1 m/s)
    jump_pitch_deg: float = 45.0
    k_thrust: float = 0.8 / 60.0       # m/s of airspeed per Hz of mean power-MN rate (60 Hz -> 0.8 m/s)
    hover_hz: float = 20.0             # power-MN rate needed to hold altitude
    k_lift: float = 1.5 / 40.0         # m/s^2 of vertical acceleration per Hz above hover
    gravity: float = 3.0               # m/s^2, reduced: a fly's terminal velocity is low (drag)
    drag: float = 2.5                  # 1/s
    k_yaw: float = np.deg2rad(600) / 30.0
    max_yaw: float = np.deg2rad(900)
    takeoff_power_hz: float = 30.0     # sustained power-MN rate that launches a voluntary takeoff
    gf_hz: float = 20.0                # smoothed GF rate that counts as an escape spike
    tau_ms: float = 40.0

    def readout(self, brain, wg: WingGroups) -> dict:
        r = brain.mean_rate
        return {"gf": r(wg.gf), "ttm": r(wg.ttm), "power": r(wg.power), "steer_L": r(wg.steer_L), "steer_R": r(wg.steer_R),
                "haltere": r(wg.haltere)}

    def maybe_takeoff(self, fly: FlyState, w: dict, dt_s: float = 0.01) -> bool:
        if w["gf"] >= self.gf_hz or w["ttm"] >= self.gf_hz:
            self.launch(fly, escape=True); return True
        # voluntary takeoff needs sustained wingbeat drive (0.1 s), not a transient
        self._power_hold = getattr(self, "_power_hold", 0.0) + dt_s if w["power"] >= self.takeoff_power_hz else 0.0
        if self._power_hold >= 0.1:
            self._power_hold = 0.0
            self.launch(fly, escape=False); return True
        return False

    def launch(self, fly: FlyState, escape: bool) -> None:
        fly.airborne = True
        v = self.jump_speed if escape else 0.2
        pitch = np.deg2rad(self.jump_pitch_deg if escape else 30)
        fly.vx = v * np.cos(pitch) * np.cos(fly.heading)
        fly.vy = v * np.cos(pitch) * np.sin(fly.heading)
        fly.vz = v * np.sin(pitch)
        fly.air_time = 0.0

    def step(self, fly: FlyState, w: dict, dt_s: float, surface_z, room: tuple) -> None:
        """Integrate one airborne step. surface_z(x, y) -> z of what is below; room = (x0,x1,y0,y1,z1)."""
        a = np.exp(-dt_s * 1000 / self.tau_ms)
        yaw = np.clip(-self.k_yaw * (w["steer_R"] - w["steer_L"]), -self.max_yaw, self.max_yaw)
        fly.yaw_rate = a * fly.yaw_rate + (1 - a) * yaw
        fly.heading += fly.yaw_rate * dt_s
        thrust = self.k_thrust * w["power"]
        f = fly.forward
        ax = (thrust * f[0] - self.drag * fly.vx)
        ay = (thrust * f[1] - self.drag * fly.vy)
        az = self.k_lift * (w["power"] - self.hover_hz) - self.gravity if w["power"] > 1 else -self.gravity
        az -= self.drag * fly.vz
        fly.vx += ax * dt_s; fly.vy += ay * dt_s; fly.vz += az * dt_s
        fly.x += fly.vx * dt_s; fly.y += fly.vy * dt_s; fly.z += fly.vz * dt_s
        fly.air_time += dt_s
        x0, x1, y0, y1, z1 = room
        if not (x0 < fly.x < x1):   # walls: land on the wall = stop, drop to the floor below
            fly.x = np.clip(fly.x, x0 + 0.01, x1 - 0.01); fly.vx = 0.0
        if not (y0 < fly.y < y1):
            fly.y = np.clip(fly.y, y0 + 0.01, y1 - 0.01); fly.vy = 0.0
        if fly.z > z1 - 0.01:
            fly.z = z1 - 0.01; fly.vz = min(fly.vz, 0.0)
        ground = surface_z(fly.x, fly.y)
        if fly.z <= ground and fly.vz <= 0 and fly.air_time > 0.05:
            fly.z = ground; fly.airborne = False
            fly.speed = float(np.hypot(fly.vx, fly.vy)) * 0.2   # landing: most momentum lost
            fly.vx = fly.vy = fly.vz = 0.0
