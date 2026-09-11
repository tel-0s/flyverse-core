"""Named neural readouts. Rates are Hz; no behavioral gains or world geometry live here."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
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
    pn_glom: np.ndarray = None  # glomerulus id of each PN
    pn_names: np.ndarray = None
    lh_odour: dict = None       # channel -> lateral-horn cell indices (LH_ODOUR_CHANNELS)


# Lateral-horn populations that report fruit odours in the model, found by screening every LH / MB-output / DN
# type at fruit and plume-free sites with heading-matched controls (NOTES, session 8). The LH is odour-tuned:
# the mixed-fruit table (blueberries dominant) selected one set (~23 Hz next to fruit, ~5 plume-free), a lone
# apple another (LHPD4d1 20.6 Hz at 8 cm, 12.5 at 40 cm, 3.4 plume-free -- graded with distance). Each is a
# channel with its own baseline; a gate reads the strongest. LHPD5c1 appears in both.
LH_ODOUR_CHANNELS = {
    "berry": ["LHPD4d2_b", "LHPD4a2", "LHAV3k1", "LHAV3h1", "LHPD5c1", "LHAD1f2"],
    "apple": ["LHPD4d1", "LHAV4a1_a", "LHAV4a1_b", "LHCENT12_a", "LHPD2a1"],
}
LH_ODOUR_TYPES = LH_ODOUR_CHANNELS["berry"]      # backwards compatibility


def motor_groups(c: Connectome) -> MotorGroups:
    fwd_types = ["DNp09", "DNa01", "DNa03", "DNb01", "DNa04"]
    g = MotorGroups(
        fwd_dn=c.select(type=fwd_types),
        back_dn=c.select(type="MDN"),
        turn_L=c.select(type="DNa02", somaSide="L"),
        turn_R=c.select(type="DNa02", somaSide="R"),
        # optomotor: the populations whose L - R flips sign under sustained imposed yaw (scripts/screen_rotation.py):
        # HSN / HSE (the horizontal-system tangential cells) and DNp20. DNp04 + LPT27/30, used until session 8,
        # do not flip (d' 0.1-0.5) and are driven by walking instead.
        opto_L=c.select(type=["DNp20", "HSN", "HSE"], somaSide="L"),
        opto_R=c.select(type=["DNp20", "HSN", "HSE"], somaSide="R"),
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
    g.pn_names, g.pn_glom = np.unique(gl, return_inverse=True)
    g.lh_odour = {name: c.select(type=types) for name, types in LH_ODOUR_CHANNELS.items()}
    g.names = {"fwd_dn": fwd_types, "back_dn": ["MDN"], "turn": ["DNa02 L/R"], "opto": ["DNp20, HSN, HSE L/R"], "leg": ["leg MNs L/R"], "proboscis": ["MN9"]}
    return g


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


@dataclass(frozen=True)
class MotorRates:
    """Independent CPU snapshot: scalars for B=1, arrays (B,) otherwise. Missing groups read zero."""
    fwd_dn: float | np.ndarray = 0.0
    back_dn: float | np.ndarray = 0.0
    turn_L: float | np.ndarray = 0.0
    turn_R: float | np.ndarray = 0.0
    opto_L: float | np.ndarray = 0.0
    opto_R: float | np.ndarray = 0.0
    wind_ipsi_L: float | np.ndarray = 0.0
    wind_ipsi_R: float | np.ndarray = 0.0
    wind_contra_L: float | np.ndarray = 0.0
    wind_contra_R: float | np.ndarray = 0.0
    leg_L: float | np.ndarray = 0.0
    leg_R: float | np.ndarray = 0.0
    proboscis: float | np.ndarray = 0.0
    gf: float | np.ndarray = 0.0
    ttm: float | np.ndarray = 0.0
    power: float | np.ndarray = 0.0
    steer_L: float | np.ndarray = 0.0
    steer_R: float | np.ndarray = 0.0
    haltere: float | np.ndarray = 0.0
    lh_odour: dict = field(default_factory=dict)         # channel -> mean rate (Hz) of its LH population
    pn_glomeruli: dict = field(default_factory=dict)
    pn_glom_cells: dict = field(default_factory=dict)   # PNs per glomerulus (the PN gate ignores tiny ones)
    time_ms: float = 0.0

    def row(self, index: int = 0) -> MotorRates:
        arrays = [getattr(self, f.name) for f in fields(self) if isinstance(getattr(self, f.name), np.ndarray)]
        arrays += [v for v in self.pn_glomeruli.values() if isinstance(v, np.ndarray)]
        batch = len(arrays[0]) if arrays else 1
        if index < 0 or index >= batch:
            raise IndexError(index)
        def at(value):
            if np.isscalar(value):
                return value
            return float(value[index])
        return MotorRates(**{f.name: ({k: at(v) for k, v in self.pn_glomeruli.items()}
                            if f.name == "pn_glomeruli" else self.time_ms if f.name == "time_ms"
                            else dict(self.pn_glom_cells) if f.name == "pn_glom_cells"
                            else {k: at(v) for k, v in self.lh_odour.items()} if f.name == "lh_odour"
                            else at(getattr(self, f.name))) for f in fields(self)})


def read_motor(brain, groups: MotorGroups | None = None, wings: WingGroups | None = None) -> MotorRates:
    """Gather every readout from one rate snapshot, including batched brains."""
    groups = groups if groups is not None else motor_groups(brain.c)
    wings = wings if wings is not None else wing_groups(brain.c)
    if getattr(brain, "cuda", False):
        return _read_cuda_motor(brain, groups, wings)
    rates = brain.rate_np()[None] if brain.B == 1 else brain.rate.detach().cpu().numpy()

    def scalar_or_batch(value):
        return float(value[0]) if brain.B == 1 else value.copy()

    def mean(idx):
        return scalar_or_batch(rates[:, idx].mean(axis=1) if len(idx) else np.zeros(brain.B))

    values = {name: mean(getattr(groups, name)) for name in (
        "fwd_dn", "back_dn", "turn_L", "turn_R", "opto_L", "opto_R", "wind_ipsi_L", "wind_ipsi_R",
        "wind_contra_L", "wind_contra_R", "leg_L", "leg_R", "proboscis")}
    values.update({f.name: mean(getattr(wings, f.name)) for f in fields(wings)})
    values["lh_odour"] = {name: mean(idx) for name, idx in (groups.lh_odour or {}).items()}
    gloms = groups.pn_names
    pn = rates[:, groups.pn]
    if len(gloms):
        counts = np.bincount(groups.pn_glom)
        means = np.stack([np.bincount(groups.pn_glom, weights=row) / np.maximum(counts, 1) for row in pn])
        values["pn_glomeruli"] = {name: scalar_or_batch(means[:, i]) for i, name in enumerate(gloms)}
        values["pn_glom_cells"] = {name: int(counts[i]) for i, name in enumerate(gloms)}
    return MotorRates(**values, time_ms=brain.t)


def _read_cuda_motor(brain, groups, wings):
    """One small GPU reduction and one transfer; full rates stay at diagnostic cadence."""
    import torch
    from . import cuda
    names = ["fwd_dn", "back_dn", "turn_L", "turn_R", "opto_L", "opto_R", "wind_ipsi_L", "wind_ipsi_R",
             "wind_contra_L", "wind_contra_R", "leg_L", "leg_R", "proboscis"]
    indices = [getattr(groups, name) for name in names]
    for f in fields(wings):
        names.append(f.name); indices.append(getattr(wings, f.name))
    channels = list((groups.lh_odour or {}).items())                 # per-channel LH populations (LH_ODOUR_CHANNELS)
    for ch, idx in channels:
        names.append(f"lh_odour:{ch}"); indices.append(idx)
    gloms = groups.pn_names
    indices += [groups.pn[groups.pn_glom == i] for i in range(len(gloms))]
    # Selectors are public and may be edited by experiments. Cache the packing,
    # not the values, and notice selector edits without transferring rates.
    indices = [np.asarray(idx) for idx in indices]
    if any(idx.ndim != 1 or (idx.size and (not np.issubdtype(idx.dtype,np.integer) or
            (idx < 0).any() or (idx >= brain.n).any())) for idx in indices):
        raise ValueError("motor groups must be vectors of connectome indices")
    indices = [idx.astype(np.int32,copy=False) for idx in indices]
    key = tuple(idx.tobytes() for idx in indices)
    cached = getattr(brain, "_cuda_motor_groups", None)
    if cached is None or cached[0] != key:
        flat = np.concatenate(indices)
        ptr = torch.tensor(np.r_[0,np.cumsum([len(idx) for idx in indices])], device=brain.device, dtype=torch.int32)
        idx = torch.tensor(flat, device=brain.device, dtype=torch.int32)
        brain._cuda_motor_groups = key, ptr, idx
    _, ptr, idx = brain._cuda_motor_groups
    means = cuda.group_means(ptr,idx,brain.rate,brain.spikes).cpu().numpy()
    brain._cuda_spike_total = ((brain.step_count,brain.t,brain.spikes._version),means[:,-1].copy())
    def value(i):
        return float(means[0,i]) if brain.B == 1 else means[:,i].copy()
    values = {name:value(i) for i,name in enumerate(names) if not name.startswith("lh_odour:")}
    values["lh_odour"] = {name.split(":", 1)[1]: value(i) for i, name in enumerate(names) if name.startswith("lh_odour:")}
    values["pn_glomeruli"] = {name:value(len(names)+i) for i,name in enumerate(gloms)}
    values["pn_glom_cells"] = {name:len(indices[len(names)+i]) for i,name in enumerate(gloms)}
    return MotorRates(**values,time_ms=brain.t)
