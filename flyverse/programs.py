"""Hand-designed behaviour programs: swappable approximations of brain functions the full model does not
(yet) produce on its own. They sit between `MotorRates` and the body and are OFF by default -- the plain
body (`body.Locomotion`, `body.Flight`) is a physical readout of the connectome model, nothing more.

Each program is an explicit, documented stand-in for a circuit:

* `AnemotaxisProgram` -- odour-gated upwind steering, casting on plume loss, hunger scaling and a search
  program. In the animal this is the lateral accessory lobe / central complex steering system acting on
  the wind-direction and odour signals the model *does* carry (DNp18 / DNp33; the lateral-horn odour
  population `motor.LH_ODOUR_TYPES`). The program reads exactly those brain signals and supplies the
  integration.
* `EscapeGating` -- habituation of the giant-fibre escape. In the animal the loom detectors (LPLC2 /
  LC4) are suppressed by wide-field motion; the rate optic lobe here is not, so a wall or a berry
  sweeping past the eye drives the GF like a loom. (An efference-copy discount of the fly's own turning
  was tried and rejected: a loom makes the fly turn, so it suppressed real escapes -- NOTES, session 8.)

Use: `cmd = program.apply(motor, cmd, fly, metabolism, dt_s)` after `Locomotion.readout`, and
`wcmd = gating.apply(motor, wcmd, fly, dt_s)` after `Flight.readout`. `scripts/room_demo.py --program
anemotaxis --escape-gating` runs them; `scripts/probe_sustain.py` measures the fly with and without.
The point of keeping them separate: a screen (`flyverse/screen.py`) can look for the circuit that makes
a program unnecessary, and a program can be replaced by a brain module when one is found.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AnemotaxisProgram:
    """Odour-gated anemotaxis (van Breugel & Dickinson 2014; Alvarez-Salvado et al. 2018)."""
    # the odour gate: the lateral-horn population's mean rate (1 s smoothed) against a threshold, or the
    # PN level statistic (max - median over glomeruli with >= pn_min_cells PNs); NOTES session 8
    odour_source: str = "lh"                 # "lh" | "pn"
    # per-channel (baseline Hz, gate range Hz) for motor.LH_ODOUR_CHANNELS: berry from the mixed-fruit screen,
    # apple from the single-apple screen (3.4-4.7 Hz plume-free, 12.5 at 40 cm, 20 at 8 cm)
    lh_channels: dict = field(default_factory=lambda: {"berry": (10.0, 12.0), "apple": (5.0, 8.0)})
    pn_base_hz: float = 20.0
    pn_gate_hz: float = 40.0
    pn_min_cells: int = 3
    smooth_s: float = 1.0
    gate_tau_s: float = 1.5                  # the surge outlasts the odour by ~1.5 s
    # upwind steering from the wind-direction DNs, gated by odour and scaled by hunger
    k_wind: float = np.deg2rad(120) / 40.0   # rad/s per Hz of wind-DN asymmetry at full gate
    wind_speed_bonus: float = 0.006          # m/s extra forward drive at full gate (the surge)
    hunger_gain_min: float = 0.1             # gain at zero hunger, relative to full hunger
    # casting: crosswind zigzag on plume loss
    # casting is the flight literature (van Breugel & Dickinson 2014); a walking fly that loses odour makes a
    # brief crosswind turn and then the offset response below dominates (Alvarez-Salvado et al. 2018)
    cast_yaw: float = np.deg2rad(90)
    cast_half_period_s: float = 1.5
    cast_duration_s: float = 3.0
    lost_gate: float = 0.25                  # gate below this (after > 0.6 within 30 s) = plume lost
    # search: hungry and no surge-level odour for offset_delay_s -> downwind drift (offset response,
    # fading with odour) + Ornstein-Uhlenbeck turning, with klinokinesis (less turning while the odour rises)
    search_hunger: float = 0.3
    offset_delay_s: float = 1.5              # walking flies turn downwind 1-2 s after odour offset (Alvarez-Salvado 2018);
                                             # 10 s never triggered next to a flickering source and the fly sat upwind of it
    offset_duration_s: float = 20.0          # the downwind drift is an episode after plume loss, then pure local search
    k_downwind: float = np.deg2rad(60) / 40.0
    search_yaw_sd: float = np.deg2rad(60)
    search_yaw_tau_s: float = 1.0
    search_rising_turn: float = 0.25
    search_trend_tau_s: float = 4.0
    search_speed: float = 0.006
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def odour(self, motor, dt_s: float) -> float:
        a = np.exp(-dt_s / self.smooth_s)
        if self.odour_source == "lh":
            lh = {k: float(v) for k, v in motor.lh_odour.items()}
            sm = getattr(self, "_lh_smooth", lh)
            self._lh_smooth = {k: a * sm.get(k, v) + (1 - a) * v for k, v in lh.items()}
            gates = {k: float(np.clip((self._lh_smooth[k] - base) / rng, 0.0, 1.0)) for k, (base, rng) in self.lh_channels.items() if k in self._lh_smooth}
            best = max(gates, key=gates.get) if gates else None
            self.odour_hz = self._lh_smooth.get(best, 0.0); self.odour_channel = best
            return gates.get(best, 0.0)
        names = list(motor.pn_glomeruli)
        glom = np.array([float(motor.pn_glomeruli[n]) for n in names], float)
        n_pn = np.array([motor.pn_glom_cells.get(n, 1) for n in names])
        self._glom_smooth = a * getattr(self, "_glom_smooth", glom) + (1 - a) * glom
        x = self._glom_smooth[n_pn >= self.pn_min_cells] if (n_pn >= self.pn_min_cells).any() else self._glom_smooth
        self.odour_hz = float(x.max() - np.median(x)) if len(x) else 0.0
        return float(np.clip((self.odour_hz - self.pn_base_hz) / self.pn_gate_hz, 0.0, 1.0))

    def apply(self, motor, cmd: dict, fly, metabolism, dt_s: float) -> dict:
        upwind = 0.5 * ((motor.wind_ipsi_L - motor.wind_ipsi_R) - (motor.wind_contra_L - motor.wind_contra_R))
        odour = self.odour(motor, dt_s)
        self.gate = max(odour, getattr(self, "gate", 0.0) * np.exp(-dt_s / self.gate_tau_s))
        t = getattr(self, "_t", 0.0) + dt_s; self._t = t
        # plume-loss detection and casting
        if self.gate > 0.6:
            self._last_hit = t; self._cast_from = None
        elif self.gate < self.lost_gate and getattr(self, "_last_hit", -1e9) > t - 30.0 and getattr(self, "_cast_from", None) is None:
            self._cast_from = t
        cast = 0.0
        if getattr(self, "_cast_from", None) is not None:
            age = t - self._cast_from
            if age < self.cast_duration_s:
                cast = self.cast_yaw * (1.0 if int(age / self.cast_half_period_s) % 2 == 0 else -1.0)
            else:
                self._cast_from = None; self._last_hit = -1e9
        hunger = metabolism.hunger if metabolism is not None else 1.0
        hunger_scale = self.hunger_gain_min + (1 - self.hunger_gain_min) * hunger
        # search program
        since_hit = t - getattr(self, "_last_hit", -1e9)
        searching = hunger > self.search_hunger and self.gate < 0.6 and cast == 0.0 and since_hit > self.offset_delay_s
        a_ou = np.exp(-dt_s / self.search_yaw_tau_s)
        self._ou = a_ou * getattr(self, "_ou", 0.0) + np.sqrt(1 - a_ou ** 2) * self.search_yaw_sd * self.rng.standard_normal()
        in_offset = since_hit < self.offset_delay_s + self.offset_duration_s          # only for a while after a real hit
        downwind = self.k_downwind * hunger * max(1.0 - self.gate / 0.6, 0.0) * (1.0 if in_offset else 0.0)
        a_tr = np.exp(-dt_s / self.search_trend_tau_s)
        self._odour_slow = a_tr * getattr(self, "_odour_slow", self.odour_hz) + (1 - a_tr) * self.odour_hz
        rising = self.odour_hz > self._odour_slow + 0.5
        search_yaw = ((self.search_rising_turn if rising else 1.0) * self._ou - downwind * upwind) if searching else 0.0
        out = dict(cmd)
        out["speed"] = cmd["speed"] + self.wind_speed_bonus * self.gate * hunger_scale + (self.search_speed if searching else 0.0)
        out["yaw"] = cmd["yaw"] + self.k_wind * self.gate * hunger_scale * upwind + cast + search_yaw
        out["mode"] = "casting" if cast else ("surging" if self.gate > 0.6 else ("exploring" if searching else "searching"))
        out["rates"] = dict(cmd.get("rates", {}), **{"odour Hz": self.odour_hz, "gate x10": self.gate * 10})
        return out

    def state(self) -> dict:
        return {k: v for k, v in vars(self).items() if k != "rng"}

    def load_state(self, d: dict) -> None:
        self.__dict__.update({k: v for k, v in d.items() if k != "rng"})


@dataclass
class EscapeGating:
    """Escape threshold = gf_hz + habituation x (10 s running mean of the GF rate); warm-started at gf_hz."""
    habituation: float = 1.0
    tau_s: float = 10.0

    def apply(self, motor, wcmd: dict, fly, dt_s: float) -> dict:
        gf = float(wcmd["gf"]); base = float(wcmd.get("gf_threshold", 30.0))
        a = np.exp(-dt_s / self.tau_s)
        self._gf_mean = a * getattr(self, "_gf_mean", base) + (1 - a) * gf     # warm start: the post-reset transient
        out = dict(wcmd)
        out["gf_threshold"] = base + self.habituation * self._gf_mean
        return out

    def state(self) -> dict:
        return dict(vars(self))

    def load_state(self, d: dict) -> None:
        self.__dict__.update(d)


def make_program(name: str | None):
    """'none' | 'anemotaxis' -> program instance or None."""
    if name in (None, "none", ""):
        return None
    if name == "anemotaxis":
        return AnemotaxisProgram()
    if name == "cx":
        from .cx import CompassSteering        # steers through the brain (PFL3 stimulation), not the body
        return CompassSteering()
    raise ValueError(f"unknown program {name!r}")
