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

from .connectome import Connectome


@dataclass
class MotorGroups:
    fwd_dn: np.ndarray
    back_dn: np.ndarray
    turn_L: np.ndarray
    turn_R: np.ndarray
    opto_L: np.ndarray
    opto_R: np.ndarray
    wind_ipsi_L: np.ndarray     # DNp18 (+ DNge016, DNge175, DNg05_a, DNp19): fire on the side the wind comes from
    wind_ipsi_R: np.ndarray
    wind_contra_L: np.ndarray   # DNp33, DNg99: fire on the side away from the wind
    wind_contra_R: np.ndarray
    pn: np.ndarray              # antennal-lobe projection neurons: the odour signal that gates upwind turning
    leg_L: np.ndarray
    leg_R: np.ndarray
    proboscis: np.ndarray
    names: dict = field(default_factory=dict)
    pn_glom: np.ndarray = None  # glomerulus id of each PN (the gate reads the most active glomerulus)
    lh_odour: np.ndarray = None # lateral-horn types whose rate is the food-odour signal (see LH_ODOUR_TYPES)


# Lateral-horn cell types that report fruit odour in the model, found by recording every LH / MB-output / DN
# type at fruit and plume-free sites with heading-matched controls (NOTES, session 8): ~23 Hz next to fruit
# (blueberry or apple, facing into or away from the wind) vs ~5-10 Hz on a plume-free table spot and ~2-5 on
# the floor. The LH is the innate-valence output of the olfactory system; these 14 cells are the gate.
LH_ODOUR_TYPES = ["LHPD4d2_b", "LHPD4a2", "LHAV3k1", "LHAV3h1", "LHPD5c1", "LHAD1f2"]


def motor_groups(c: Connectome) -> MotorGroups:
    fwd_types = ["DNp09", "DNa01", "DNa03", "DNb01", "DNa04"]
    g = MotorGroups(
        fwd_dn=c.select(type=fwd_types),
        back_dn=c.select(type="MDN"),
        turn_L=c.select(type="DNa02", somaSide="L"),
        turn_R=c.select(type="DNa02", somaSide="R"),
        opto_L=c.select(type=["DNp04", "LPT27", "LPT30"], somaSide="L"),
        opto_R=c.select(type=["DNp04", "LPT27", "LPT30"], somaSide="R"),
        wind_ipsi_L=c.select(type=["DNp18", "DNge016", "DNge175", "DNg05_a", "DNp19"], somaSide="L"),
        wind_ipsi_R=c.select(type=["DNp18", "DNge016", "DNge175", "DNg05_a", "DNp19"], somaSide="R"),
        wind_contra_L=c.select(type=["DNp33", "DNg99"], somaSide="L"),
        wind_contra_R=c.select(type=["DNp33", "DNg99"], somaSide="R"),
        # uniglomerular PNs only: the multiglomerular M_ types (275 cells, many GABAergic) are not a glomerulus
        # and burst to 150 Hz from antennal-lobe activity alone
        pn=c.select(type="~^(?!M_)[^_]+_(l2PN|adPN|lPN|lvPN|ilPN|ivPN|vPN)"),
        leg_L=c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"], somaSide="L"),
        leg_R=c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"], somaSide="R"),
        proboscis=c.select(type="MN9"),
    )
    gl = np.array([t.split("_")[0] for t in c.neurons.type.to_numpy()[g.pn]])
    g.pn_glom = np.unique(gl, return_inverse=True)[1]
    g.lh_odour = c.select(type=LH_ODOUR_TYPES)
    g.names = {"fwd_dn": fwd_types, "back_dn": ["MDN"], "turn": ["DNa02 L/R"], "opto": ["DNp04, LPT27, LPT30 L/R"], "leg": ["leg MNs L/R"], "proboscis": ["MN9"]}
    return g


@dataclass
class FlyState:
    x: float = 0.0            # m, world frame
    y: float = 0.0
    z: float = 0.75           # height of the walking surface
    heading: float = 0.0      # rad, 0 = +x, positive = counter-clockwise (left)
    pitch: float = 0.0        # rad, positive = nose up (free in flight, 0 on flat surfaces)
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

    @property
    def forward(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        return np.array([cp * np.cos(self.heading), cp * np.sin(self.heading), np.sin(self.pitch)])

    @property
    def left(self) -> np.ndarray:
        f = self.forward
        l0 = np.array([-np.sin(self.heading), np.cos(self.heading), 0.0])
        u0 = np.cross(f, l0)
        return np.cos(self.roll) * l0 + np.sin(self.roll) * u0

    @property
    def up(self) -> np.ndarray:
        return np.cross(self.forward, self.left)

    def body_to_world(self, dirs_body: np.ndarray) -> np.ndarray:
        """(…, 3) body-frame directions (x fwd, y left, z up) -> world frame."""
        R = np.stack([self.forward, self.left, self.up], axis=1)   # columns = body axes (full 3-D orientation)
        return dirs_body @ R.T

    @property
    def eye_pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z + self.eye_height])


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
    k_fwd: float = 0.02 / 40.0     # m/s per Hz of forward-DN mean rate (40 Hz -> 2 cm/s, a brisk fly walk)
    k_leg: float = 0.02 / 60.0     # m/s per Hz of mean leg-MN rate
    k_back: float = 0.02 / 40.0
    k_turn: float = np.deg2rad(200) / 40.0   # rad/s per Hz of DNa02 asymmetry
    k_opto: float = np.deg2rad(150) / 15.0   # rad/s per Hz of DNp04/LPT asymmetry (optomotor, stabilising)
    opto_hp_tau_s: float = 2.0               # the optomotor term uses the asymmetry minus its 2 s running mean:
                                             # DNp04's right side carries a chronic bias (asymmetric eye) that
                                             # otherwise turns the stabilising term into a constant spin
    # odour-gated anemotaxis (van Breugel & Dickinson 2014): the wind-direction DNs (DNp18 ipsilateral,
    # DNp33 contralateral to the wind; scripts/probe_wind.py) steer the fly upwind, with a gain gated by
    # the antennal lobe's odour signal; the gate decays over ~1.5 s after the plume is lost (the surge)
    k_wind: float = np.deg2rad(120) / 40.0   # rad/s per Hz of wind-DN asymmetry at full gate
    # gate = clip((max over glomeruli of the 1 s-smoothed mean PN rate - median over glomeruli - pn_base_hz) / pn_gate_hz).
    # Chosen offline on recorded PN activity (NOTES, session 8): fruit odour is a sustained, glomerulus-specific
    # elevation (blueberry: DM2 80 Hz; apple: DM1 / VA2 / DM2), the model's antennal-lobe noise is 1 s bursts of
    # 2-cell glomeruli; smoothing + a minimum glomerulus size + max-minus-median separates them (0% false gate on
    # a plume-free spot, 100% next to fruit). An adaptive baseline on the max was tried first and rejected.
    pn_base_hz: float = 20.0
    pn_gate_hz: float = 40.0
    pn_smooth_s: float = 1.0
    pn_min_cells: int = 3
    # ... or, by default, the brain's own odour signal: the mean rate of the lateral-horn population
    # LH_ODOUR_TYPES (1 s smoothed), gate = clip((rate - lh_base_hz) / lh_gate_hz)
    odour_source: str = "lh"                 # "lh" | "pn"
    lh_base_hz: float = 10.0
    lh_gate_hz: float = 12.0
    gate_tau_s: float = 1.5
    wind_speed_bonus: float = 0.006          # m/s of extra forward drive at full gate (surge upwind)
    # casting: when the plume is lost after a hit, search crosswind -- a zigzag whose sign alternates
    # every `cast_half_period_s`, for up to `cast_duration_s` (the fly's search program, body-level)
    cast_yaw: float = np.deg2rad(90)
    cast_half_period_s: float = 1.5
    cast_duration_s: float = 8.0
    lost_gate: float = 0.25                  # gate below this (after having been above 0.6) = plume lost
    hunger_gain_min: float = 0.1             # upwind drive at zero hunger, relative to full hunger (0.3 walked sated
                                             # flies past the fruit to the upwind edge, where no plume reaches)
    # search program when hungry and without odour (Alvarez-Salvado et al. 2018: walking flies that lose an
    # odour turn and walk downwind for a while, then wander): after `offset_delay_s` without odour, a
    # hunger-scaled downwind drift (the wind DNs with reversed sign; facing downwind is its stable point)
    # plus Ornstein-Uhlenbeck yaw noise for local search
    search_hunger: float = 0.3               # hunger above which the search program runs
    offset_delay_s: float = 10.0
    k_downwind: float = np.deg2rad(60) / 40.0
    search_yaw_sd: float = np.deg2rad(60)    # rad/s standard deviation of the OU yaw noise
    search_yaw_tau_s: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    metabolism: Metabolism = field(default_factory=Metabolism)
    k_leg_turn: float = np.deg2rad(100) / 30.0
    max_speed: float = 0.03
    max_yaw: float = np.deg2rad(400)
    edge_turn: float = np.deg2rad(90)   # turn rate towards the interior while the fly is against an edge / wall
    tau_ms: float = 80.0           # motor smoothing
    baseline_speed: float = 0.008  # m/s intrinsic walking drive (flies walk spontaneously; keeps optic flow alive)
    mdn_threshold: float = 15.0    # Hz; MDN fires a few Hz from self-motion optic flow, only a real burst means "back up"

    def readout(self, brain, g: MotorGroups) -> dict:
        r = brain.mean_rate
        fwd = r(g.fwd_dn); back = r(g.back_dn)
        tL, tR = r(g.turn_L), r(g.turn_R)
        oL, oR = r(g.opto_L), r(g.opto_R)
        lL, lR = r(g.leg_L), r(g.leg_R)
        prob = r(g.proboscis)
        wiL, wiR, wcL, wcR = r(g.wind_ipsi_L), r(g.wind_ipsi_R), r(g.wind_contra_L), r(g.wind_contra_R)
        upwind = 0.5 * ((wiL - wiR) - (wcL - wcR))          # + = wind from the left = turn left to face it
        asym = oL - oR
        a_hp = np.exp(-0.01 / self.opto_hp_tau_s)
        self._opto_bias = a_hp * getattr(self, "_opto_bias", asym) + (1 - a_hp) * asym
        opto = asym - self._opto_bias
        a_sm = np.exp(-0.01 / self.pn_smooth_s)
        if self.odour_source == "lh" and g.lh_odour is not None and len(g.lh_odour):
            lh = r(g.lh_odour)
            self._lh_smooth = a_sm * getattr(self, "_lh_smooth", lh) + (1 - a_sm) * lh
            odour = float(np.clip((self._lh_smooth - self.lh_base_hz) / self.lh_gate_hz, 0.0, 1.0))
            self._odour_hz = self._lh_smooth
        else:
            pn_rates = brain.rates(g.pn)
            n_pn = np.bincount(g.pn_glom)
            glom_mean = np.bincount(g.pn_glom, weights=pn_rates) / np.maximum(n_pn, 1)
            self._glom_smooth = a_sm * getattr(self, "_glom_smooth", glom_mean) + (1 - a_sm) * glom_mean
            x = self._glom_smooth[n_pn >= self.pn_min_cells]
            odour = float(np.clip((x.max() - np.median(x) - self.pn_base_hz) / self.pn_gate_hz, 0.0, 1.0))
            self._odour_hz = float(x.max() - np.median(x))
        self._gate = max(odour, getattr(self, "_gate", 0.0) * np.exp(-0.01 / self.gate_tau_s))
        # plume-loss detection and casting
        t = getattr(self, "_t", 0.0) + 0.01; self._t = t
        if self._gate > 0.6:
            self._last_hit = t; self._cast_from = None
        elif self._gate < self.lost_gate and getattr(self, "_last_hit", -1e9) > t - 30.0 and getattr(self, "_cast_from", None) is None:
            self._cast_from = t                                              # plume lost: start casting
        cast = 0.0
        if getattr(self, "_cast_from", None) is not None:
            age = t - self._cast_from
            if age < self.cast_duration_s:
                cast = self.cast_yaw * (1.0 if int(age / self.cast_half_period_s) % 2 == 0 else -1.0)
            else:
                self._cast_from = None; self._last_hit = -1e9
        self._cast = cast
        hunger_scale = self.hunger_gain_min + (1 - self.hunger_gain_min) * self.metabolism.hunger
        # search program: hungry, no odour for a while, not casting
        since_hit = t - getattr(self, "_last_hit", -1e9)
        # (runs below the surge level too: in the isotropic near-field of a fruit the gate sits at 0.3-0.5, an
        # upwind surge goes nowhere, and the way into the plume proper is downwind of the source)
        searching = (self.metabolism.hunger > self.search_hunger and self._gate < 0.6 and cast == 0.0
                     and since_hit > self.offset_delay_s)
        a_ou = np.exp(-0.01 / self.search_yaw_tau_s)
        self._ou = a_ou * getattr(self, "_ou", 0.0) + np.sqrt(1 - a_ou ** 2) * self.search_yaw_sd * self.rng.standard_normal()
        downwind = self.k_downwind * self.metabolism.hunger * max(1.0 - self._gate / 0.6, 0.0)   # fades as the odour grows
        search_yaw = (self._ou - downwind * upwind) if searching else 0.0
        self._searching = bool(searching)
        back_eff = max(back - self.mdn_threshold, 0.0) if np.isscalar(back) else np.maximum(back - self.mdn_threshold, 0.0)
        speed = self.baseline_speed + self.k_fwd * fwd + self.k_leg * 0.5 * (lL + lR) - self.k_back * back_eff + self.wind_speed_bonus * self._gate * hunger_scale
        # convention: DNa02 drives ipsilateral turning (Rayshubskiy et al. 2020): right DNa02 -> turn right
        yaw = -self.k_opto * opto - self.k_turn * (tR - tL) + self.k_leg_turn * (lL - lR) + self.k_wind * self._gate * hunger_scale * upwind + cast + search_yaw
        return {"speed": float(np.clip(speed, -self.max_speed, self.max_speed)),
                "yaw": float(np.clip(yaw, -self.max_yaw, self.max_yaw)),
                "proboscis": float(np.clip(prob / 30.0, 0, 1)),
                "mode": "casting" if cast else ("surging" if self._gate > 0.6 else ("exploring" if searching else "searching")),
                "rates": {"fwdDN": fwd, "MDN": back, "opto_L": oL, "opto_R": oR, "wind_L": wiL, "wind_R": wiR, "odour Hz": self._odour_hz, "gate x10": self._gate * 10, "DNa02_L": tL, "DNa02_R": tR, "legMN_L": lL, "legMN_R": lR, "MN9": prob}}

    def step(self, fly: FlyState, cmd: dict, dt_s: float, bounds: tuple) -> None:
        a = np.exp(-dt_s * 1000 / self.tau_ms)
        fly.speed = a * fly.speed + (1 - a) * cmd["speed"]
        fly.yaw_rate = a * fly.yaw_rate + (1 - a) * cmd["yaw"]
        fly.proboscis = cmd["proboscis"]
        fly.heading += fly.yaw_rate * dt_s
        nx = fly.x + fly.speed * dt_s * np.cos(fly.heading)
        ny = fly.y + fly.speed * dt_s * np.sin(fly.heading)
        x0, x1, y0, y1 = bounds
        in_x, in_y = x0 <= nx <= x1, y0 <= ny <= y1
        if in_x and in_y:
            fly.x, fly.y = nx, ny
        else:
            # against an edge (table edge: the front legs find no substrate; wall: body contact). Flies
            # rarely walk off edges: slide along it and turn towards the interior at a walking-turn rate.
            # (An earlier version spun the fly at 720 deg/s here, which drove the loom detectors from the
            # rotating scene and made it hop itself off the table -- NOTES, session 8.)
            if in_x: fly.x = nx
            if in_y: fly.y = ny
            fly.x = float(np.clip(fly.x, x0, x1)); fly.y = float(np.clip(fly.y, y0, y1))
            to_centre = np.arctan2(0.5 * (y0 + y1) - fly.y, 0.5 * (x0 + x1) - fly.x)
            err = (to_centre - fly.heading + np.pi) % (2 * np.pi) - np.pi
            fly.heading += np.sign(err) * min(abs(err), self.edge_turn * dt_s)
        fly.heading = (fly.heading + np.pi) % (2 * np.pi) - np.pi


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
    gf_hz: float = 30.0                # smoothed GF rate that counts as an escape (a burst, not one crosstalk spike)
    # escape habituation: the threshold rises by `gf_habituation` x the GF's running mean over `gf_hab_tau_s`.
    # A loom is a burst out of silence (45 Hz from ~3) and fires; the sustained 60-80 Hz the model's
    # LPLC2 / LC4 produce next to a wall (10 cm away, filling the eye, expanding with every turn) does
    # not keep the fly hopping. Looming-evoked escapes habituate in the animal too.
    gf_habituation: float = 1.0
    gf_hab_tau_s: float = 10.0
    tau_ms: float = 40.0

    def readout(self, brain, wg: WingGroups) -> dict:
        r = brain.mean_rate
        gf = r(wg.gf)
        a = np.exp(-0.01 / self.gf_hab_tau_s)
        self._gf_mean = a * getattr(self, "_gf_mean", 0.0) + (1 - a) * gf
        return {"gf": gf, "gf_threshold": self.gf_hz + self.gf_habituation * self._gf_mean,
                "ttm": r(wg.ttm), "power": r(wg.power), "steer_L": r(wg.steer_L), "steer_R": r(wg.steer_R),
                "haltere": r(wg.haltere)}

    landing_refractory_s: float = 1.0  # no new escape within this time of landing (a jump-land-jump chain is not fly behaviour)

    def maybe_takeoff(self, fly: FlyState, w: dict, dt_s: float = 0.01) -> bool:
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
            fly.z = ground; fly.airborne = False
            fly.pitch = 0.0; fly.roll = 0.0                            # landed on a flat surface
            fly.ground_time = 0.0
            fly.speed = float(np.hypot(fly.vx, fly.vy)) * 0.2   # landing: most momentum lost
            fly.vx = fly.vy = fly.vz = 0.0
