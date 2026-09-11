"""A simulated central-complex steering stage that plugs INTO the brain.

The full model's central complex is silent (NOTES, session 8): no compass (EPG / PEN at 0 Hz), no goal
comparison (PFL 0 Hz), while its inputs (the wind-direction DNs, the lateral-horn odour population) and
its output (PFL3 -> DNa02 -> leg motor neurons; PFL3-left at 80 Hz drives DNa02-right 22.6 Hz) both
work. `CompassSteering` fills exactly that hole: it reads the brain's own wind and odour signals from
`MotorRates`, decides a steering error the way the LAL / FB circuit would (upwind when the odour says
so, crosswind when the plume is lost, downwind with local search when hungry and without odour), and
expresses it by *stimulating PFL3 left / right* (and DNp09 for forward drive). The connectome does
the rest down to the legs; the body reads DNa02 as it always did. Unlike `programs.AnemotaxisProgram`
this injects nothing into the body -- swap it out when the real compass works.

Use:  cx = CompassSteering(); each frame, after `motor = fb.motor()`: `cx.apply(fb, motor, fly, metabolism, dt_s)`.
The demo runs it with `--program cx`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .programs import AnemotaxisProgram


@dataclass
class CompassSteering:
    pfl_hz: float = 80.0            # PFL3 drive at full steering error (80 Hz on one side -> DNa02 contralateral ~23 Hz)
    fwd_hz: float = 60.0            # DNp09 drive at full forward command (the surge)
    k_error: float = 1.0 / 40.0     # steering error per Hz of wind-DN asymmetry (40 Hz asymmetry = full error)
    gate: AnemotaxisProgram = field(default_factory=AnemotaxisProgram)   # odour gate, casting and search logic, reused
    pfl_L: dict = field(default_factory=lambda: {"type": "PFL3", "somaSide": "L"})
    pfl_R: dict = field(default_factory=lambda: {"type": "PFL3", "somaSide": "R"})
    fwd_sel: dict = field(default_factory=lambda: {"type": "DNp09"})

    def apply(self, fb, motor, fly, metabolism, dt_s: float) -> dict:
        g = self.gate
        # run the program's decision logic on a zero command to get its steering / speed *intent*
        intent = g.apply(motor, {"speed": 0.0, "yaw": 0.0, "rates": {}}, fly, metabolism, dt_s)
        # the program's yaw is a rate in rad/s built from k_wind * gate * hunger * upwind (+ cast + search);
        # normalise it to a steering error in [-1, 1]: 120 deg/s (its full-gate value at 40 Hz asymmetry) = 1
        err = float(np.clip(intent["yaw"] / (g.k_wind * 40.0), -1.0, 1.0))
        fwd = float(np.clip(intent["speed"] / (g.wind_speed_bonus + g.search_speed), 0.0, 1.0))
        # + error = turn left. PFL3 projects contralaterally (PFL3-L -> DNa02-R -> turn right), so drive the RIGHT PFL3 to turn left
        ms = dt_s * 1000 * 1.5
        fb.stimulate(self.pfl_R, hz=self.pfl_hz * max(err, 0.0), ms=ms)
        fb.stimulate(self.pfl_L, hz=self.pfl_hz * max(-err, 0.0), ms=ms)
        fb.stimulate(self.fwd_sel, hz=self.fwd_hz * fwd, ms=ms)
        self.error, self.forward = err, fwd
        return {"mode": intent["mode"], "error": err, "forward": fwd, "gate": g.gate, "odour_hz": g.odour_hz}

    def state(self) -> dict:
        return {"gate": self.gate.state(), "error": getattr(self, "error", 0.0), "forward": getattr(self, "forward", 0.0)}

    def load_state(self, d: dict) -> None:
        self.gate.load_state(d.get("gate", {}))
