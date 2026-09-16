"""Explicit navigation experiments; no changes to parent synapses or body commands.

Sources, exact equations versus engineering choices, and acceptance gates are in
docs/audits/navigation_instruments.md. These modules are NOT adopted physiology.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from .compass import epg_columns
from .senses import batch_values

TAU = 2 * math.pi
PFL_SOURCE = "https://doi.org/10.1038/s41586-023-07006-3"
WANG_SOURCE = "https://github.com/pwang724/fly-circuit-exploration/tree/80b94e6608cf927ca2c9f2bbce0577c5a998684c"


def _frame_seconds(dt_ms):
    if not math.isfinite(dt_ms) or not 0 <= dt_ms <= 10.000001:
        raise ValueError("expected a finite module frame of 0-10 ms")
    return dt_ms / 1000


def pfl3_rates(heading, goal):
    """Mussells Pires 2024, Methods 'Full PFL3 model', PDF pp.17-18.

    Returns model rates in Hz, (B,2,12), left then right LAL. These idealized
    populations are not a cell-by-cell mapping onto MaleCNS soma annotations.
    """
    g = -np.array([15, 45, 75, 105, 135, 165, -165, -135, -105, -75, -45, -15])
    h = -np.array(
        [
            [
                -67.5,
                -22.5,
                22.5,
                22.5,
                67.5,
                112.5,
                112.5,
                157.5,
                -157.5,
                -157.5,
                -112.5,
                -67.5,
            ],
            [
                67.5,
                112.5,
                157.5,
                157.5,
                -157.5,
                -112.5,
                -112.5,
                -67.5,
                -22.5,
                -22.5,
                22.5,
                67.5,
            ],
        ]
    )
    return _pfl3(
        heading,
        goal,
        torch.as_tensor(np.deg2rad(h), device=heading.device, dtype=heading.dtype),
        torch.as_tensor(np.deg2rad(g), device=heading.device, dtype=heading.dtype),
    )


def _pfl3(heading, goal, hpref, gpref):
    x = torch.cos(heading[:, :, None] - hpref[None]) + 0.63 * torch.cos(
        goal[:, :, None] - gpref[None, None]
    )
    return 29.23 * F.softplus(2.17 * (x - 0.7))


class NeuralInstrument:
    kind = "stop-gap"
    required_preset = "instrumented"
    quantity_in = "rate_hz"
    channel_out = "poisson_hz"
    cuda_async_validation = True
    cuda_graph_safe = True
    poisson_always_on = False
    requires_any = ()
    incompatible = ()
    state_bounds: ClassVar[dict] = {}
    state_flags = ()

    def __init__(self, c):
        self.source_body_ids = c.neurons.bodyId.to_numpy().copy()
        self.reads, self.writes = {}, {}
        self.parameters = {}

    def validate_parent(self, fb):
        if fb.preset != "instrumented":
            raise ValueError(f"{self.name} requires preset=instrumented")
        if not np.array_equal(self.source_body_ids, fb.c.neurons.bodyId.to_numpy()):
            raise ValueError(f"{self.name}: different connectome row order")
        if fb.brain.p.dt * getattr(self, "max_hz", 250) > 1000:
            raise ValueError(
                f"{self.name}: Poisson ceiling exceeds probability bound at this dt"
            )

    def install(self, fb):
        self.validate_parent(fb)
        if fb.attached_modules.get(self.name) is not self:
            fb.attach(self)

    def reset(self, B, device):
        self.B, self.device = B, torch.device(device)
        self._state_names = []
        self.initialize()

    def state(self, name, value):
        value = np.asarray(value, dtype=np.float32)
        t = torch.as_tensor(value, device=self.device).clone()
        if t.ndim == 0:
            t = t.expand(self.B, 1).clone()
        setattr(self, name, t)
        self._state_names.append(name)

    def upload(self, name, value):
        v = batch_values(value, self.B, name).astype(np.float32)
        if not np.isfinite(v).all():
            raise ValueError(f"{name} overflows float32")
        getattr(self, name).copy_(
            torch.from_numpy(v[:, None]), non_blocking=self.device.type == "cuda"
        )

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in self._state_names}

    def load_state_dict(self, state):
        if set(state) != set(self._state_names):
            raise ValueError(f"{self.name}: checkpoint fields differ")
        checked = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in state.items()
        }
        for k, v in checked.items():
            if v.shape != getattr(self, k).shape or not bool(torch.isfinite(v).all()):
                raise ValueError(f"{self.name}: invalid checkpoint {k}")
            if k in self.state_bounds:
                lo, hi = self.state_bounds[k]
                if not bool(((v >= lo) & (v <= hi)).all()):
                    raise ValueError(f"{self.name}: checkpoint {k} outside [{lo},{hi}]")
            if k in self.state_flags and not bool(((v == 0) | (v == 1)).all()):
                raise ValueError(f"{self.name}: checkpoint {k} must be boolean")
        for k, v in checked.items():
            getattr(self, k).copy_(v)

    def reset_rows(self, rows):
        # Initialize a small peer to obtain the same defaults without replacing captured buffers.
        import copy

        fresh = copy.copy(self)
        fresh.reset(self.B, self.device)
        for k in self._state_names:
            getattr(self, k)[rows] = getattr(fresh, k)[rows]

    def describe(self):
        return dict(
            name=self.name,
            kind=self.kind,
            law="unverified",
            **{"class": f"flyverse.navigation:{type(self).__name__}"},
            gap=self.gap,
            removal=self.removal,
            parameters=self.parameters,
            sources=self.sources,
            audits=["docs/audits/navigation_instruments.md"],
            requires_any=list(self.requires_any),
            incompatible=list(self.incompatible),
            reads={k: self.source_body_ids[v].tolist() for k, v in self.reads.items()},
            writes={
                k: self.source_body_ids[v].tolist() for k, v in self.writes.items()
            },
        )


class RecurrentCompass(NeuralInstrument):
    """Wang's reduced EPG/PEN loop, independently expressed as batched matrix dynamics.

    Memory resides in recurrent rates, not an integrated phase scalar. The default EB/PB=0
    is the declared textbook counterfactual. EB/PB=2.7 is a separate brake diagnostic, not
    a measured effective gain. No velocity calibration or physiological adoption is implied.
    """

    name = "compass_ring"
    incompatible = ("compass",)
    gap = "a recurrent angular-memory experiment beside the kinematic compass stand-in"
    removal = "a validated native heading circuit or a quantitatively validated connectome-fitted model"
    sources: ClassVar[list] = [WANG_SOURCE, "https://elifesciences.org/articles/23496"]
    max_hz = 50.0001
    poisson_always_on = (
        True  # output has an explicit positive 0.0001 Hz engineering floor
    )
    state_bounds: ClassVar[dict] = {"e": (0, 5), "p": (0, 60)}

    def __init__(self, c, *, eb_ratio=0.0, velocity_gain=0.3 / (math.pi / 2)):
        super().__init__(c)
        if not math.isfinite(eb_ratio) or not 0 <= eb_ratio <= 5.4:
            raise ValueError("eb_ratio must be in [0,5.4]")
        if not math.isfinite(velocity_gain) or not 0 < velocity_gain <= 1:
            raise ValueError("invalid velocity_gain")
        self.idx, self.columns = epg_columns(c)
        self.writes = {"epg": self.idx}
        self.parameters = {
            "eb_ratio": eb_ratio,
            "velocity_gain": velocity_gain,
            "tau_ms": 50.0,
            "substep_ms": 2.0,
            "local_gain": 0.6,
            "return_gain": 1.6,
            "inhibition": 1.2,
            "rate_cap": 5.0,
            "peak_hz": 50.0,
            "floor_hz": 0.0001,
            "noise": 0.0,
            "calibrated": False,
        }
        e2p = np.zeros((16, 16), np.float32)
        p2e = np.zeros_like(e2p)
        for side in range(2):
            for g in range(8):
                j = 8 * side + g
                read = 2 * g if side == 0 else 2 * (7 - g) + 1
                write = (read + (-2 if side == 0 else 2)) % 16
                tile = [write, (write + (1 if side == 0 else -1)) % 16]
                e2p[j, read] = 1.0
                e2p[j, tile] += eb_ratio / 2
                p2e[tile, j] = 0.8
        delta = (np.arange(16)[:, None] - np.arange(16)[None] + 8) % 16 - 8
        self._matrices = (e2p, p2e, 0.6 * np.exp(-0.5 * delta**2).astype(np.float32))

    def initialize(self):
        self.e2p, self.p2e, self.local = [
            torch.as_tensor(v, device=self.device) for v in self._matrices
        ]
        self.columns_t = torch.as_tensor(self.columns, device=self.device)
        self.state(
            "e",
            np.broadcast_to(
                np.maximum(np.cos(np.arange(16) * TAU / 16), 0), (self.B, 16)
            ),
        )
        self.state("p", np.zeros((self.B, 16)))
        self.state("yaw", 0.0)

    def observe_turn(self, yaw_rate):
        self.upload("yaw", yaw_rate)

    def step(self, dt_ms, inputs):
        if not math.isfinite(dt_ms) or not 0 <= dt_ms <= 10.000001:
            raise ValueError("invalid module frame")
        steps = max(1, math.ceil(dt_ms / 2))
        a = dt_ms / (50 * steps)
        # Positive body yaw is left; increasing right-shifter gain advances the declared EPG coordinate.
        vel = (self.yaw * self.parameters["velocity_gain"]).clamp(-0.8, 0.8)
        gain = torch.cat(((1 - vel).expand(-1, 8), (1 + vel).expand(-1, 8)), dim=1)
        for _ in range(steps):
            self.p.add_(a * ((self.e @ self.e2p.T * gain).relu() - self.p))
            target = (
                self.e @ self.local.T
                + self.p @ self.p2e.T
                - 4.8 * self.e.mean(1, keepdim=True)
            ).relu()
            self.e.add_(a * (target - self.e)).clamp_(0, 5)
        return {"epg": 0.0001 + 10 * self.e[:, self.columns_t]}


class HungerGain(NeuralInstrument):
    """Energy -> explicit navigation gain; no fabricated hunger-neuron or receptor mapping."""

    name = "hunger"
    requires_any = ("plume", "flight")
    gap = "the environment metabolism is not reported to navigation circuits"
    removal = "a sourced metabolic transducer with verified target cells and dynamics"
    sources: ClassVar[list] = ["https://doi.org/10.1016/j.cell.2011.02.008"]
    state_bounds: ClassVar[dict] = {"level": (0, 1)}

    def __init__(self, c):
        super().__init__(c)
        self.parameters = {
            "law": "gain=(1-energy)*(not sated)",
            "default_without_input": 1.0,
            "kinetics": "instantaneous engineering control, not insulin/sNPF kinetics",
        }

    def initialize(self):
        self.state("level", 1.0)

    def observe_internal(self, energy, sated, **kwargs):
        self.upload("level", (1 - energy) * (1 - sated))

    def step(self, dt_ms, inputs):
        _frame_seconds(dt_ms)
        return {}


class PlumeNavigation(NeuralInstrument):
    """Odor-gated upwind goal, learned entry bearing, and the published PFL3 comparator.

    Reads biological EPG and LH rates plus antennal deflection. The goal memory and
    translation of model LAL rate difference into Poisson input on biological PFL3 sides
    are explicitly synthetic. No world wind angle, source position or body heading input.
    """

    name = "plume"
    requires_any = ("compass", "compass_ring")
    gap = "heading, odor and wind signals lack a functional goal-memory/steering bridge"
    removal = (
        "native odor/wind goal memory and PFL3 steering pass the same sensory controls"
    )
    sources: ClassVar[list] = [
        PFL_SOURCE,
        "https://doi.org/10.1038/s41586-026-10827-7",
        "https://doi.org/10.1038/s41467-022-32247-7",
        "https://doi.org/10.1038/s41467-024-46225-8",
        "https://doi.org/10.1016/j.cub.2013.12.023",
    ]
    max_hz = 80.0
    state_bounds: ClassVar[dict] = {
        "odor": (0, 1),
        "lost_s": (0, 3600),
        "entry_x": (-1, 1),
        "entry_y": (-1, 1),
        "heading": (-math.pi, math.pi),
        "turn": (-1, 1),
        "strength": (0, 1.00001),
    }
    state_flags = ("inside", "seen", "airborne", "feeding")

    def __init__(self, c):
        super().__init__(c)
        from .motor import LH_ODOUR_CHANNELS

        idx, w = epg_columns(c)
        self.reads = {"epg": idx}
        self._epg_weights = (
            np.stack([np.cos(w * TAU / 16), np.sin(w * TAU / 16)], 1)
            / np.bincount(w, minlength=16)[w, None]
        )
        for name, types in LH_ODOUR_CHANNELS.items():
            ix = np.flatnonzero(c.neurons.type.isin(types).to_numpy())
            if not len(ix):
                raise ValueError(f"plume requires LH odor channel {name}")
            self.reads["odor_" + name] = ix
        for side in ("L", "R"):
            ix = np.flatnonzero(
                (c.neurons.type.eq("PFL3") & c.neurons.somaSide.eq(side)).to_numpy()
            )
            if not len(ix):
                raise ValueError(f"plume requires PFL3 on side {side}")
            self.writes["pfl_" + side] = ix
        ix = np.flatnonzero(c.neurons.type.eq("DNp09").to_numpy())
        if not len(ix):
            raise ValueError("plume requires DNp09")
        self.writes["forward"] = ix
        self.parameters = {
            "pfl_model": {
                "a_hz": 29.23,
                "b": 2.17,
                "c": -0.7,
                "d": 0.63,
                "columns": 12,
            },
            "odor_baseline_hz": {"berry": 10.0, "apple": 5.0},
            "odor_range_hz": {"berry": 12.0, "apple": 8.0},
            "odor_tau_s": 1.0,
            "entry_learning": 0.5,
            "return_memory_s": 30.0,
            "return_delay_s": 0.45,
            "pfl_input_peak_hz": 80.0,
            "forward_peak_hz": 60.0,
            "lal_difference_scale_hz": 40.0,
            "flight_cast_half_period_s": 1.5,
            "minimum_heading_strength": 0.6,
        }
        self.hunger = None

    def bind_instruments(self, instruments):
        self.hunger = instruments.get("hunger")

    def initialize(self):
        self.epg_weights = torch.as_tensor(
            self._epg_weights, dtype=torch.float32, device=self.device
        )
        self.epg_mass = torch.linalg.vector_norm(self.epg_weights, dim=1, keepdim=True)
        g = -np.array([15, 45, 75, 105, 135, 165, -165, -135, -105, -75, -45, -15])
        h = -np.array(
            [
                [
                    -67.5,
                    -22.5,
                    22.5,
                    22.5,
                    67.5,
                    112.5,
                    112.5,
                    157.5,
                    -157.5,
                    -157.5,
                    -112.5,
                    -67.5,
                ],
                [
                    67.5,
                    112.5,
                    157.5,
                    157.5,
                    -157.5,
                    -112.5,
                    -112.5,
                    -67.5,
                    -22.5,
                    -22.5,
                    22.5,
                    67.5,
                ],
            ]
        )
        self.hpref = torch.as_tensor(
            np.deg2rad(h), dtype=torch.float32, device=self.device
        )
        self.gpref = torch.as_tensor(
            np.deg2rad(g), dtype=torch.float32, device=self.device
        )
        for k, v in {
            "odor": 0.0,
            "inside": 0.0,
            "seen": 0.0,
            "lost_s": 0.0,
            "wind_x": 0.0,
            "wind_y": 0.0,
            "entry_x": 0.0,
            "entry_y": 0.0,
            "airborne": 0.0,
            "feeding": 0.0,
            "goal": 0.0,
            "heading": 0.0,
            "turn": 0.0,
            "strength": 0.0,
        }.items():
            self.state(k, v)

    def observe_wind(self, dL, dR):
        l = batch_values(dL, self.B, "dL")
        r = batch_values(dR, self.B, "dR")
        self.upload("wind_x", l + r)
        self.upload("wind_y", l - r)

    def observe_internal(self, airborne, feeding, **kwargs):
        self.upload("airborne", airborne)
        self.upload("feeding", feeding)

    def step(self, dt_ms, inputs):
        dt = _frame_seconds(dt_ms)
        vector = inputs["epg"] @ self.epg_weights
        # Each EPG contributes one inverse-occupancy weight to circular mass.
        mass = inputs["epg"] @ self.epg_mass
        self.strength.copy_(
            torch.linalg.vector_norm(vector, dim=1, keepdim=True) / mass.clamp_min(1e-8)
        )
        self.heading.copy_(torch.atan2(vector[:, 1:2], vector[:, :1]))
        gates = [
            (inputs["odor_" + k].mean(1, keepdim=True) - base)
            / self.parameters["odor_range_hz"][k]
            for k, base in self.parameters["odor_baseline_hz"].items()
        ]
        odor = torch.stack(gates).amax(0).clamp(0, 1)
        self.odor.lerp_(odor, 1 - math.exp(-dt))
        inside = torch.where(
            self.odor > 0.6,
            torch.ones_like(self.inside),
            torch.where(self.odor < 0.25, torch.zeros_like(self.inside), self.inside),
        )
        valid = self.strength >= 0.6
        entered = (inside > 0) & ((self.inside == 0) | (self.seen == 0)) & valid
        hx, hy = torch.cos(self.heading), torch.sin(self.heading)
        self.entry_x.copy_(
            torch.where(entered, 0.5 * self.entry_x + 0.5 * hx, self.entry_x)
        )
        self.entry_y.copy_(
            torch.where(entered, 0.5 * self.entry_y + 0.5 * hy, self.entry_y)
        )
        self.seen.copy_(torch.maximum(self.seen, entered.float()))
        self.lost_s.copy_(
            torch.where(
                inside > 0,
                torch.zeros_like(self.lost_s),
                (self.lost_s + dt).clamp_max(3600),
            )
        )
        self.inside.copy_(inside)
        upwind = self.heading + torch.atan2(self.wind_y, self.wind_x)
        entry = torch.atan2(self.entry_y, self.entry_x)
        cast_sign = 1 - 2 * (torch.floor(self.lost_s / 1.5).remainder(2))
        returning = torch.where(
            self.airborne > 0, upwind + cast_sign * math.pi / 2, entry
        )
        self.goal.copy_(
            torch.where((inside > 0) | (self.lost_s < 0.45), upwind, returning)
        )
        rates = _pfl3(self.heading, self.goal, self.hpref, self.gpref)
        # With the paper's negated preferred-angle arrays, R-L follows positive goal error.
        # This frame convention is tested over heading and goal angles, before any behavioral run.
        turn = (
            rates[:, 1].mean(1, keepdim=True) - rates[:, 0].mean(1, keepdim=True)
        ) / 40.0
        gain = (self.odor + self.seen * 0.35 * torch.exp(-self.lost_s / 30)).clamp(0, 1)
        gain = gain * valid * (1 - self.feeding)
        if self.hunger is not None:
            gain = gain * self.hunger.level
        self.turn.copy_((turn * gain).clamp(-1, 1))
        # Existing graph: PFL3-R -> DNa02-L -> positive/left yaw; opposite for PFL3-L.
        return {
            "pfl_R": (80 * self.turn.relu()).expand(-1, len(self.writes["pfl_R"])),
            "pfl_L": (80 * (-self.turn).relu()).expand(-1, len(self.writes["pfl_L"])),
            "forward": (60 * gain).expand(-1, len(self.writes["forward"])),
        }


class FlightDrive(NeuralInstrument):
    """Engineering flight-power servo and PFL3-to-wing bridge on the neural boundary.

    This deliberately bypasses unvalidated DNg02/VNC control by driving the biological wing MNs.
    The paper supports investigating descending power control, NOT these gains or a motor-neuron
    rate of 100 Hz. The latter is derived from the shipped body's lift/gravity equation.
    """

    name = "flight"
    gap = "wing power cannot sustain level flight and the walking steering readout does not control wings"
    removal = "validated descending/VNC flight-state and steering mechanisms supply the same control"
    sources: ClassVar[list] = ["https://pmc.ncbi.nlm.nih.gov/articles/PMC9206711/"]
    max_hz = 250.0
    state_bounds: ClassVar[dict] = {
        "energy": (0, 1),
        "integral": (-100, 150),
        "power_input": (0, 250),
        "ground_s": (0, 20),
        "bout_s": (0, 8),
        "odor": (0, 1),
    }
    state_flags = (
        "sated",
        "airborne",
        "feeding",
        "observed",
        "reserve_low",
        "odor_near",
        "landing",
        "active",
        "was_airborne",
    )

    def __init__(self, c):
        super().__init__(c)
        from .motor import LH_ODOUR_CHANNELS, wing_groups

        wings = wing_groups(c)
        for k in ("power", "steer_L", "steer_R"):
            idx = getattr(wings, k)
            if not len(idx):
                raise ValueError(f"flight requires nonempty {k} wing group")
            self.writes[k] = idx
        self.reads = {"power": wings.power}
        for side in ("L", "R"):
            idx = np.flatnonzero(
                (c.neurons.type.eq("PFL3") & c.neurons.somaSide.eq(side)).to_numpy()
            )
            if not len(idx):
                raise ValueError(f"flight requires PFL3 {side}")
            self.reads["pfl_" + side] = idx
        # Read the frame's neural inputs, not PlumeNavigation's mutable state:
        # the latter would make flight depend on instrument attachment order.
        odor_channels = {}
        for key, (base, span) in {"berry": (10.0, 12.0), "apple": (5.0, 8.0)}.items():
            idx = np.flatnonzero(c.neurons.type.isin(LH_ODOUR_CHANNELS[key]).to_numpy())
            if len(idx):
                self.reads["odor_" + key] = idx
                odor_channels[key] = {"baseline_hz": base, "range_hz": span}
        self.parameters = {
            "target_power_hz": 100.0,
            "power_kp": 0.25,
            "power_ki_per_s": 0.5,
            "max_input_hz": 250.0,
            "steering_baseline_hz": 10.0,
            "steering_gain": 0.1,
            "reserve_stop": 0.35,
            "reserve_resume": 0.5,
            "ground_search_s": 20.0,
            "max_power_bout_s": 8.0,
            "odor_channels": odor_channels,
            "odor_tau_s": 1.0,
            "odor_land": 0.6,
            "odor_clear": 0.25,
            "landing_policy": "withdraw artificial wing drive until touchdown",
            "policy_source": "unverified engineering foraging priority; no physiological fit",
            "source": "body lift equilibrium: 20 + 3/(1.5/40) = 100 Hz",
            "altitude_control": False,
            "flight_energy_cost": False,
        }
        self.hunger = None

    def bind_instruments(self, instruments):
        self.hunger = instruments.get("hunger")

    def describe(self):
        record = super().describe()
        record["audits"].append("docs/audits/flight_foraging_priority.md")
        return record

    def initialize(self):
        for k, v in {
            "energy": 0.0,
            "sated": 0.0,
            "airborne": 0.0,
            "feeding": 0.0,
            "observed": 0.0,
            "integral": 0.0,
            "power_input": 0.0,
            "ground_s": 0.0,
            "bout_s": 0.0,
            "odor": 0.0,
            "reserve_low": 0.0,
            "odor_near": 0.0,
            "landing": 0.0,
            "active": 0.0,
            "was_airborne": 0.0,
        }.items():
            self.state(k, v)

    def observe_internal(self, energy, sated, airborne, feeding):
        for k, v in {
            "energy": energy,
            "sated": sated,
            "airborne": airborne,
            "feeding": feeding,
            "observed": 1.0,
        }.items():
            self.upload(k, v)

    def step(self, dt_ms, inputs):
        dt = _frame_seconds(dt_ms)
        p = self.parameters
        self.reserve_low.copy_(
            torch.where(
                self.energy <= p["reserve_stop"],
                1.0,
                torch.where(self.energy >= p["reserve_resume"], 0.0, self.reserve_low),
            )
        )
        gates = [
            (inputs["odor_" + key].mean(1, keepdim=True) - channel["baseline_hz"])
            / channel["range_hz"]
            for key, channel in p["odor_channels"].items()
        ]
        odor = (
            torch.stack(gates).amax(0).clamp(0, 1)
            if gates
            else torch.zeros_like(self.odor)
        )
        self.odor.lerp_(odor, 1 - math.exp(-dt / p["odor_tau_s"]))
        self.odor_near.copy_(
            torch.where(
                self.odor >= p["odor_land"],
                1.0,
                torch.where(self.odor <= p["odor_clear"], 0.0, self.odor_near),
            )
        )
        inhibited = (
            (self.reserve_low > 0)
            | (self.odor_near > 0)
            | (self.sated > 0)
            | (self.feeding > 0)
            | (self.observed == 0)
        )
        if self.hunger is not None:
            inhibited = inhibited | ((self.hunger.level == 0) & (self.airborne == 0))
        airborne = self.airborne > 0
        landed = (self.was_airborne > 0) & ~airborne
        expired = self.bout_s >= p["max_power_bout_s"]
        # Once descent is requested, intermittent odor/energy cannot restart lift midair.
        self.landing.copy_(airborne & ((self.landing > 0) | inhibited | expired))
        self.ground_s.copy_(
            torch.where(
                ~airborne & ~inhibited & ~landed & (self.active == 0),
                (self.ground_s + dt).clamp_max(p["ground_search_s"]),
                0.0,
            )
        )
        request = (
            ~inhibited
            & (self.landing == 0)
            & ~landed
            & ~expired
            & ((self.active > 0) | airborne | (self.ground_s >= p["ground_search_s"]))
        )
        # Count commanded flight, including a failed takeoff, so no episode can run forever.
        self.bout_s.copy_(
            torch.where(
                request,
                (self.bout_s + dt).clamp_max(p["max_power_bout_s"]),
                0.0,
            )
        )
        self.active.copy_(request)
        self.was_airborne.copy_(self.airborne)
        error = 100 - inputs["power"].mean(1, keepdim=True)
        self.integral.copy_(
            torch.where(
                request,
                (self.integral + 0.5 * dt * error).clamp(-100, 150),
                torch.zeros_like(error),
            )
        )
        self.power_input.copy_(
            torch.where(
                request,
                (100 + 0.25 * error + self.integral).clamp(0, 250),
                torch.zeros_like(error),
            )
        )
        # PFL3-R signals left walking turns; wing MN-L signals left flight turns in the shipped body.
        turn = 0.1 * (
            inputs["pfl_R"].mean(1, keepdim=True)
            - inputs["pfl_L"].mean(1, keepdim=True)
        )
        turn = turn.clamp(-10, 10)
        return {
            "power": self.power_input.expand(-1, len(self.writes["power"])),
            "steer_L": (request * (10 + turn)).expand(-1, len(self.writes["steer_L"])),
            "steer_R": (request * (10 - turn)).expand(-1, len(self.writes["steer_R"])),
        }
