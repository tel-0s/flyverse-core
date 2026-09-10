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
