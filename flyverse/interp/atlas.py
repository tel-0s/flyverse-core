"""atlas -- stimulate each population briefly and read every motor readout, with a null (docs/INTERP.md 4.5).

`scripts/screen_dns.py` generalised into the toolkit's contract. One `FlyBrain(batch=B)`; inside a batch, row `r`
carries the pulse for population `r` (`FlyBrain.stimulate`, `hz` for `ms`, after `settle_ms` of quiet) and the
remaining rows carry no pulse at all -- those are the null rows, the matched control of every number here.
`replicates` independent runs (separate processes / seeds, the project's replicate unit) give the scatter, and every
(population, readout) difference is reported through `common.compare`, never as a bare delta.

What is measured per (population, readout): the mean of the readout over the pulse window (`mean`), over its second
half (`half`, the steady state the 30 s screens report), the last frame (`final`, what `scripts/benchmark.py`'s dn
section reads after 400 ms) and the pre-pulse baseline (`pre`). Readouts are the `MotorRates` cell sets themselves
(`flyverse.motor.motor_groups` / `wing_groups`, pooled exactly as `read_motor` pools them), their left-right
differences, and -- when `pattern` is given -- every type it matches through `screen.TypeRecorder`, by soma side.

The tool reads the model and changes nothing: a pulse is `FlyBrain.stimulate`, a context is the FlyBrain's own
sensory entry points (`fb.wind` / `fb.smell` / `fb.taste`), returning to rest between chunks is `FlyBrain.reset()`,
and the connectome, `LIFParams` and `OpticParams` are whatever the caller resolved.

Two columns keep a row honest, because a stimulation screen has two ways of lying. `self_drive` marks the rows where
the stimulated cells are themselves part of the readout -- `turn_L` IS DNa02_L, `gf` IS DNp01, `wind_ipsi_L` contains
DNp18_L -- so the readout moves by construction and not through a circuit; the summary reports the strongest mover
that is not inside the readout beside the strongest overall. `out_to_pruned_share` says how much of a population's
outgoing weight lands on cells the LIF does not run: with `prune_frozen` the optic rate units are frozen and pruned
out of the matrix, so 87 % of the photoreceptors' output goes nowhere and their atlas row reads null as an artefact
of the compiled model, not as a claim about the circuit.

GPU / CPU split: `scripts/interp_atlas.py run` executes one run (one seed) on the cluster and writes an `AtlasRun`
(npz + json, with the provenance block and the REALISED device); `analyse` reads a set of runs on the desktop and
writes the `Result`. `atlas(...)` does both in process (what the CPU test and `--device cpu` use).
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import connectome as cn
from .. import motor as motor_mod
from ..screen import TypeRecorder
from . import common
from .common import MIN_REPLICATES, Z_RESULT, Result, body_str, compare, population, provenance, resolve, to_jsonable

ATLAS_RUN_SCHEMA = "flyverse.interp.atlas_run/1"

# Every MotorRates field that is a pooled cell set (time_ms / pn_glomeruli / pn_glom_cells are not readouts of a
# population; lh_odour is expanded per channel and the PN glomeruli only when asked).
MOTOR_FIELDS = ("fwd_dn", "back_dn", "turn_L", "turn_R", "opto_L", "opto_R", "wind_ipsi_L", "wind_ipsi_R",
                "wind_contra_L", "wind_contra_R", "leg_L", "leg_R", "proboscis")
MOTOR_PAIRS = (("turn_L", "turn_R"), ("opto_L", "opto_R"), ("wind_ipsi_L", "wind_ipsi_R"),
               ("wind_contra_L", "wind_contra_R"), ("leg_L", "leg_R"), ("steer_L", "steer_R"))
SUMMARIES = ("mean", "half", "final", "pre")

# The floor on the null arm's SD used for the atlas' own z (`z_floor`): a readout whose null draws are bit-identical
# (an all-silent motor group, say) has SD 0, and `common.compare`'s z is then NaN whatever the effect size. The
# floor is declared, not fitted: 0.05 Hz is well below the 0.5 Hz 'never firing' bound of the toolkit.
SD_FLOOR_HZ = 0.05


# ------------------------------------------------------------------------------------------------ sensory contexts
def wind_deflections(heading_deg: float = -90.0, speed: float = 0.3, direction_deg: float = 180.0,
                     full_speed: float = 0.5) -> tuple[float, float]:
    """The normalised backward antennal deflections (dL, dR) of a fly pinned at `heading_deg` in a horizontal wind.

    Reproduces `flyverse.air.Air.deflections` for a pinned pose without building a room: wind blows towards
    `direction_deg` (the defaults are `air.WindParams`: 0.3 m/s towards 180 deg, i.e. from +x), heading -90 faces -y
    so the wind arrives on the fly's LEFT (`scripts/screen_steering.py`'s `windL` site), +90 on its right.
    """
    d = math.radians(direction_deg)
    vec = np.array([speed * math.cos(d), speed * math.sin(d), 0.0])
    h = math.radians(heading_deg)
    forward = np.array([math.cos(h), math.sin(h), 0.0])
    left = np.array([-math.sin(h), math.cos(h), 0.0])
    wx, wy = float(forward @ vec), float(left @ vec)
    c45 = math.cos(math.pi / 4)
    return -(wx * c45 + wy * c45) / full_speed, -(wx * c45 - wy * c45) / full_speed


CONTEXTS = {
    "wind_left": {"wind": wind_deflections(-90.0)},
    "wind_right": {"wind": wind_deflections(+90.0)},
    "wind_head_on": {"wind": wind_deflections(0.0)},
    "wind_off": {"wind": (0.0, 0.0)},
    "sugar": {"taste": 1.0},
}


def apply_context(fb, context) -> dict:
    """Drive a FlyBrain's senses for a context: None, a name in CONTEXTS, a dict {sense: args}, or a callable(fb)."""
    if context is None:
        return {"context": None}
    if callable(context):
        context(fb)
        return {"context": getattr(context, "__name__", "callable")}
    spec = CONTEXTS[context] if isinstance(context, str) else dict(context)
    for sense, arg in spec.items():
        if sense == "wind":
            fb.wind(*arg)
        elif sense == "smell":
            fb.smell(*arg)
        elif sense == "taste":
            fb.taste(arg)
        else:
            raise ValueError(f"unknown sense {sense!r} in context {context!r}; use wind / smell / taste or a callable")
    return {"context": context if isinstance(context, str) else "dict", "spec": to_jsonable(spec)}


def jo_wind_rates(c: cn.Connectome, dL: float, dR: float) -> dict[str, np.ndarray]:
    """{'idx', 'hz'} -- the Johnston's-organ C / E drive `senses.Wind` produces for one pinned pose.

    The same per-cell rates `fb.wind(dL, dR)` would install, returned as an explicit stimulation vector so the atlas
    can present them as a *population* (the wind arm of VALIDATION['atlas']).
    """
    from .. import senses
    w = senses.Wind(c)
    rE, rC = w.rates(dL, dR, 1)
    idx = np.concatenate([w.joE, w.joC]).astype(np.int64)
    hz = np.concatenate([rE[0], rC[0]]).astype(np.float32)
    order = np.argsort(idx, kind="stable")
    idx, hz = idx[order], hz[order]
    keep = np.ones(len(idx), bool)
    keep[1:] = idx[1:] != idx[:-1]                       # a JO cell in both sets keeps its first (E) rate
    return {"idx": idx[keep], "hz": hz[keep]}


# ------------------------------------------------------------------------------------------------ populations
@dataclass
class PopSpec:
    """One stimulated population: the resolved cells and the per-cell pulse rate / duration it gets."""
    label: str
    spec: object
    idx: np.ndarray
    hz: np.ndarray
    ms: float

    def record(self, c: cn.Connectome, frozen: np.ndarray | None = None, keep_ids_max: int = 10000) -> dict:
        rec = {"label": self.label, "spec": common.spec_repr(self.spec), "n_cells": int(len(self.idx)),
               "hz_max": float(self.hz.max()) if len(self.hz) else 0.0,
               "hz_mean": float(self.hz.mean()) if len(self.hz) else 0.0, "ms": float(self.ms)}
        if len(self.idx) <= keep_ids_max:
            rec["body_ids"] = body_str(c.neurons.bodyId.to_numpy()[self.idx])
        if frozen is not None and len(self.idx):
            rec["frozen_frac"] = float(np.isin(self.idx, frozen).mean())
        return rec


def make_populations(c: cn.Connectome, populations, *, by_side: bool = True, split: str | None = "type",
                     hz: float = 150.0, ms: float = 400.0, min_cells: int = 1) -> list[PopSpec]:
    """Turn the tool's `populations` argument into the stimulation list.

    `populations` is a spec of `common.resolve`'s grammar, a list of them, a dict {label: spec}, or a list of dicts
    {'label', 'spec', 'hz' (scalar or per-cell), 'ms'}. With `split='type'` (the default, and what
    `--populations superclass=descending_neuron` means) each spec is split into one population per cell type, and
    with `by_side` further into `TYPE_L` / `TYPE_R` wherever both sides exist; `split=None` keeps one population per
    spec. Populations with fewer than `min_cells` cells are dropped.
    """
    entries: list[dict] = []
    if isinstance(populations, dict):
        entries = [{"label": k, "spec": v} for k, v in populations.items()]
    elif isinstance(populations, (list, tuple)) and populations and isinstance(populations[0], dict) and \
            ("spec" in populations[0] or "idx" in populations[0]):
        entries = [dict(e) for e in populations]
    elif isinstance(populations, (list, tuple)) and not isinstance(populations, str):
        entries = [{"label": None, "spec": s} for s in populations]
    else:
        entries = [{"label": None, "spec": populations}]

    out: list[PopSpec] = []
    types = c.neurons.type.fillna("").to_numpy()
    sides = c.neurons.somaSide.fillna("?").to_numpy() if "somaSide" in c.neurons else np.full(c.n, "?")
    for e in entries:
        base_hz, base_ms = e.get("hz", hz), e.get("ms", ms)
        idx = np.asarray(e["idx"], np.int64) if "idx" in e else resolve(c, e["spec"])
        spec = e.get("spec", "index:" + "|".join(str(int(i)) for i in idx[:8]) + ("..." if len(idx) > 8 else ""))
        if split == "type" and "idx" not in e:
            groups: list[tuple[str, str, np.ndarray]] = []
            bodies = c.neurons.bodyId.to_numpy()
            for t in sorted(set(types[idx])):
                sel = idx[types[idx] == t]
                label = t or "(untyped)"
                # An untyped group is not a type: name its cells, so the recorded spec resolves back to exactly them.
                base = (lambda part: f"type={t}") if t else (lambda part: "body:" + "|".join(body_str(bodies[part])))
                if by_side and (sides[sel] == "L").any() and (sides[sel] == "R").any():
                    for s in ("L", "R"):
                        part = sel[sides[sel] == s]
                        groups.append((f"{label}_{s}", f"{base(part)}&somaSide={s}" if t else base(part), part))
                    rest = sel[~np.isin(sides[sel], ["L", "R"])]
                    if len(rest):
                        groups.append((f"{label}_?", f"{base(rest)}&{_side_clause(sides[rest])}" if t else base(rest), rest))
                else:
                    groups.append((label, base(sel), sel))
            for label, spec_s, sel in groups:
                out.append(PopSpec(label, spec_s, sel, _hz_vector(base_hz, len(sel)), float(base_ms)))
        else:
            label = e.get("label") or common.spec_repr(spec)
            out.append(PopSpec(label, spec, idx, _hz_vector(base_hz, len(idx)), float(base_ms)))
    out = [p for p in out if len(p.idx) >= max(1, int(min_cells))]
    seen: dict[str, int] = {}
    for p in out:                                        # labels are the join key between runs: make them unique
        seen[p.label] = seen.get(p.label, 0) + 1
        if seen[p.label] > 1:
            p.label = f"{p.label}#{seen[p.label]}"
    return out


def remap(pops: list[PopSpec], c_from: cn.Connectome, c_to: cn.Connectome) -> list[PopSpec]:
    """Re-address a stimulation list from one connectome to another by bodyId.

    `FlyBrain(c, modules=...)` runs `regions.subset(c, modules)`, which drops cells and renumbers the rest, and
    `FlyBrain.stimulate` takes indices of the brain it is running. Populations built against the full connectome are
    therefore translated here; cells absent from `c_to` are dropped (with their per-cell rates) and an empty
    population is kept in the list so the run's population order is the one the caller asked for.
    """
    a, b = c_from.neurons.bodyId.to_numpy(), c_to.neurons.bodyId.to_numpy()
    if c_from is c_to or (len(a) == len(b) and np.array_equal(a, b)):
        return pops
    out = []
    for p in pops:
        j = c_to.body_to_index.reindex(a[p.idx]).to_numpy()
        keep = np.isfinite(j.astype(np.float64))
        out.append(PopSpec(p.label, p.spec, j[keep].astype(np.int64), p.hz[keep], p.ms))
    return out


def _side_clause(side_labels) -> str:
    """The `somaSide:` clause of `common.resolve` that selects exactly these soma-side labels ('?' is the toolkit's
    display name for a missing side, and `resolve` matches a missing side as the empty string)."""
    vals = sorted({("" if s == "?" else str(s)) for s in np.asarray(side_labels).ravel()})
    return "somaSide:" + "|".join(vals)


def _hz_vector(hz, n: int) -> np.ndarray:
    v = np.asarray(hz, np.float32)
    if v.ndim == 0:
        return np.full(n, float(v), np.float32)
    if len(v) != n:
        raise ValueError(f"per-cell hz has {len(v)} entries for {n} cells")
    return v.astype(np.float32)


def preset_populations(c: cn.Connectome, name: str, *, by_side: bool = True, hz: float = 150.0, ms: float = 400.0,
                       min_cells: int = 1) -> list[PopSpec]:
    """The population lists this round runs: 'dn' (every descending-neuron type), 'sensory' (every sensory class plus
    the named sensory channels the model actually reads), 'dn+sensory', and 'validation' (the VALIDATION['atlas']
    arms: the JO wind drive left / right, DNa02 L / R, PFL3 L / R and the reference DNs of `benchmark.py`'s dn)."""
    if name in ("dn", "dn+sensory", "all"):
        dn = make_populations(c, "superclass=descending_neuron", by_side=by_side, split="type", hz=hz, ms=ms,
                              min_cells=min_cells)
    else:
        dn = []
    if name in ("sensory", "dn+sensory", "all"):
        sens = _sensory_populations(c, by_side=by_side, hz=hz, ms=ms, min_cells=min_cells)
    else:
        sens = []
    if name == "validation":
        return validation_populations(c, hz=hz, ms=ms)
    if not dn and not sens:
        raise ValueError(f"unknown preset {name!r}; choose from dn / sensory / dn+sensory / validation")
    return dn + sens


def _sensory_populations(c: cn.Connectome, *, by_side: bool, hz: float, ms: float, min_cells: int) -> list[PopSpec]:
    """Every sensory class (the `class` column of the cells whose superclass contains 'sensory'), every sensory
    superclass (which separates the VNC's mechanosensors from the brain's), and the named channels the model's own
    front ends read (photoreceptors, ORNs, sweet GRNs, the JO wind drive)."""
    n = c.neurons
    sensory = n.superclass.fillna("").str.contains("sensory").to_numpy()
    cls_col, sup_col = n["class"].fillna("").to_numpy(), n.superclass.fillna("").to_numpy()
    entries: list[dict] = []
    sides = n.somaSide.fillna("?").to_numpy()

    def add(label: str, spec: str, idx: np.ndarray) -> None:
        if by_side and (sides[idx] == "L").any() and (sides[idx] == "R").any():
            for s in ("L", "R"):
                entries.append({"label": f"{label}_{s}", "spec": f"{spec}&somaSide={s}", "idx": idx[sides[idx] == s]})
            rest = idx[~np.isin(sides[idx], ["L", "R"])]
            if len(rest):
                entries.append({"label": f"{label}_?", "spec": f"{spec}&{_side_clause(sides[rest])}", "idx": rest})
        else:
            entries.append({"label": label, "spec": spec, "idx": idx})

    for cl in sorted({x for x in cls_col[sensory] if x}):
        add(f"class.{cl}", f"superclass~sensory&class={cl}", np.flatnonzero(sensory & (cls_col == cl)))
    for sup in sorted({x for x in sup_col[sensory] if x}):
        add(f"superclass.{sup}", f"superclass={sup}", np.flatnonzero(sup_col == sup))
    # the named channels of the model's own sensory front ends (senses.py / motor.py), which no `class` row isolates
    entries.append({"label": "channel.photoreceptors", "spec": "type:" + "|".join(cn.PHOTORECEPTOR_TYPES),
                    "idx": resolve(c, "type:" + "|".join(cn.PHOTORECEPTOR_TYPES))})
    try:
        from .. import senses
        sm = senses.Smell(c)
        entries.append({"label": "channel.ORN_all", "spec": "senses.Smell.orn_idx", "idx": np.asarray(sm.orn_idx)})
    except Exception:                                                              # noqa: BLE001 -- subset without ORNs
        pass
    try:
        from .. import senses
        ta = senses.Taste(c)
        if len(ta.sweet):
            entries.append({"label": "channel.sweet_GRN", "spec": "senses.Taste.sweet", "idx": np.asarray(ta.sweet)})
    except Exception:                                                              # noqa: BLE001
        pass
    for label, (dL, dR) in (("channel.JO_wind_left", CONTEXTS["wind_left"]["wind"]),
                            ("channel.JO_wind_right", CONTEXTS["wind_right"]["wind"])):
        try:
            w = jo_wind_rates(c, dL, dR)
        except Exception:                                                          # noqa: BLE001 -- no JO cells
            continue
        entries.append({"label": label, "spec": f"senses.Wind.rates(dL={dL:+.4f}, dR={dR:+.4f})",
                        "idx": w["idx"], "hz": w["hz"]})
    return make_populations(c, entries, by_side=by_side, split=None, hz=hz, ms=ms, min_cells=min_cells)


def validation_populations(c: cn.Connectome, *, hz: float = 150.0, ms: float = 400.0) -> list[PopSpec]:
    """The VALIDATION['atlas'] arms as stimulated populations (docs/INTERP.md 6).

    `JO_wind_left` / `JO_wind_right` present the exact per-cell Johnston's-organ rates the room's plume-free site
    produces at heading -90 / +90 (`scripts/screen_steering.py`'s windL / windR), so the DNp18 / DNp33 left-right
    flip is measured on the same input as the reference; DNa02_L at 150 Hz for 400 ms is `benchmark.py`'s dn
    protocol; PFL3_L at 80 Hz is NOTES session 8's steering test.
    """
    entries: list[dict] = []
    for label, (dL, dR) in (("JO_wind_left", CONTEXTS["wind_left"]["wind"]),
                            ("JO_wind_right", CONTEXTS["wind_right"]["wind"]),
                            ("JO_wind_head_on", CONTEXTS["wind_head_on"]["wind"])):
        w = jo_wind_rates(c, dL, dR)
        # 3 s, not 400 ms: `scripts/benchmark.py`'s wind section averages 10 s of pinned room with the first 3 s
        # skipped, so the atlas arm has to hold the drive long enough for the same steady state. Head-on wind is the
        # internal control -- the same JO cells, no left-right difference in the drive.
        entries.append({"label": label, "spec": f"senses.Wind.rates(dL={dL:+.4f}, dR={dR:+.4f})",
                        "idx": w["idx"], "hz": w["hz"], "ms": max(ms, 3000.0)})
    for label, spec, rate in (("DNa02_L", "type=DNa02&somaSide=L", 150.0), ("DNa02_R", "type=DNa02&somaSide=R", 150.0),
                              ("PFL3_L", "type=PFL3&somaSide=L", 80.0), ("PFL3_R", "type=PFL3&somaSide=R", 80.0),
                              ("DNp09", "DNp09", 150.0), ("MDN", "MDN", 150.0), ("DNp01", "DNp01", 150.0)):
        idx = resolve(c, spec)
        if len(idx):
            entries.append({"label": label, "spec": spec, "idx": idx, "hz": rate, "ms": ms})
    return make_populations(c, entries, by_side=False, split=None, hz=hz, ms=ms)


VALIDATION_PATTERN = r"^(DNp18|DNp33|DNge016|DNg99|DNg05_a|DNp19|DNp73|WED080|DNa02|PFL3|MDN|DNp09|DNp01)$"


# ------------------------------------------------------------------------------------------------ readouts
def readout_groups(c: cn.Connectome, readouts=("motor",), pattern: str | None = None, by_side: bool = True,
                   include_pn: bool = False) -> tuple[dict[str, np.ndarray], list[tuple[str, str, str]]]:
    """({readout name: cell indices}, [(derived name, plus, minus)]).

    'motor' in `readouts` gives every pooled `MotorRates` field (`flyverse.motor.motor_groups` / `wing_groups` and
    the `lh_odour` channels, the same cell sets `read_motor` pools), 'pn' adds one readout per PN glomerulus, and
    `pattern` adds every type it matches through `screen.TypeRecorder` (keys prefixed `type.`, by soma side when
    `by_side`). The derived list holds the left-right differences (`leg_LR`, `turn_LR`, `type.DNp18_LR`, ...).
    """
    groups: dict[str, np.ndarray] = {}
    derived: list[tuple[str, str, str]] = []
    if "motor" in readouts:
        g, wg = motor_mod.motor_groups(c), motor_mod.wing_groups(c)
        for name in MOTOR_FIELDS:
            groups[name] = np.asarray(getattr(g, name), np.int64)
        for f in dataclasses.fields(wg):
            groups[f.name] = np.asarray(getattr(wg, f.name), np.int64)
        for ch, idx in (g.lh_odour or {}).items():
            groups[f"lh_odour.{ch}"] = np.asarray(idx, np.int64)
        if include_pn or "pn" in readouts:
            for i, gl in enumerate(g.pn_names):
                groups[f"pn.{gl}"] = np.asarray(g.pn[g.pn_glom == i], np.int64)
        for a, b in MOTOR_PAIRS:
            if a in groups and b in groups:
                derived.append((f"{a[:-2]}_LR", a, b))
    if pattern:
        tr = TypeRecorder.build(c, pattern=pattern, by_side=by_side)
        keys = set(tr.keys)
        for k, key in enumerate(tr.keys):
            groups[f"type.{key}"] = tr.idx[tr.inv == k]
        if by_side:
            for key in tr.keys:
                if key.endswith("_L") and key[:-2] + "_R" in keys:
                    derived.append((f"type.{key[:-2]}_LR", f"type.{key}", f"type.{key[:-2]}_R"))
    return groups, derived


class _Pooler:
    """Population means of one (B, N) rate tensor, as one gather plus one segment sum (what TypeRecorder.snapshot
    does per frame on the CPU, kept on the device so a 64-fly batch costs a single small transfer)."""

    def __init__(self, groups: dict[str, np.ndarray], device):
        import torch
        self.names = list(groups)
        parts = [np.asarray(groups[k], np.int64).ravel() for k in self.names]
        idx = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        seg = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(parts)]) if parts else np.zeros(0, np.int64)
        self.counts = np.array([max(len(p), 1) for p in parts], np.float32)
        self.n_cells = np.array([len(p) for p in parts], np.int64)
        self.idx_t = torch.as_tensor(idx, device=device)
        self.seg_t = torch.as_tensor(seg, device=device)
        self.counts_t = torch.as_tensor(self.counts, device=device)

    def __call__(self, rate) -> np.ndarray:
        import torch
        out = torch.zeros(rate.shape[0], len(self.names), device=rate.device, dtype=rate.dtype)
        if len(self.idx_t):
            out.index_add_(1, self.seg_t, rate[:, self.idx_t])
        return (out / self.counts_t).detach().cpu().numpy()


# ------------------------------------------------------------------------------------------------ one run
@dataclass
class AtlasRun:
    """One independent run (one seed, one process): the pulse-window summaries of every population and null row."""
    meta: dict
    pops: list
    readouts: list
    values: dict = field(default_factory=dict)           # summary name -> (P, R)
    null_ids: list = field(default_factory=list)
    null_values: dict = field(default_factory=dict)      # summary name -> (Q, R)
    body_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    body_values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    body_null: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))

    @property
    def labels(self) -> list:
        return [p["label"] for p in self.pops]

    def frame(self, summary: str = "mean") -> pd.DataFrame:
        return pd.DataFrame(self.values[summary], index=self.labels, columns=self.readouts)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"body_idx": self.body_idx, "body_values": self.body_values, "body_null": self.body_null}
        arrays.update({f"v__{k}": v for k, v in self.values.items()})
        arrays.update({f"n__{k}": v for k, v in self.null_values.items()})
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"schema": ATLAS_RUN_SCHEMA, "meta": to_jsonable(self.meta), "pops": to_jsonable(self.pops),
                       "readouts": list(self.readouts), "null_ids": list(self.null_ids),
                       "summaries": list(self.values)}, f, indent=1)
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path) -> "AtlasRun":
        path = Path(path)
        if path.suffix in (".npz", ".json"):
            path = path.with_suffix("")
        with open(path.with_suffix(".json"), encoding="utf-8") as f:
            side = json.load(f)
        if side.get("schema") != ATLAS_RUN_SCHEMA:
            raise ValueError(f"{path}: schema {side.get('schema')!r} is not {ATLAS_RUN_SCHEMA}")
        z = np.load(path.with_suffix(".npz"), allow_pickle=False)
        return cls(meta=side["meta"], pops=side["pops"], readouts=side["readouts"],
                   values={k[3:]: z[k] for k in z.files if k.startswith("v__")},
                   null_ids=side["null_ids"], null_values={k[3:]: z[k] for k in z.files if k.startswith("n__")},
                   body_idx=z["body_idx"], body_values=z["body_values"], body_null=z["body_null"])


def _reset(fb, context) -> dict:
    """Back to rest with the context's sensory drive on and no pulse pending.

    `FlyBrain.reset()` is the public entry point and does the whole job (LIF and optic state, the base Poisson drive,
    the pending pulses, the held radiance, the clock and any captured CUDA graph); the context is then re-applied
    through `fb.wind` / `fb.smell` / `fb.taste`. The toolkit touches no private state of the model.
    """
    fb.reset()
    return apply_context(fb, context)


def _pulse_matrix(rows: list[tuple[int, PopSpec]], batch: int) -> tuple[np.ndarray, np.ndarray]:
    order: dict[int, int] = {}
    cols: list[int] = []
    for _, ps in rows:
        for i in ps.idx:
            k = int(i)
            if k not in order:
                order[k] = len(cols)
                cols.append(k)
    idx = np.asarray(cols, np.int64)
    mat = np.zeros((batch, len(idx)), np.float32)
    for row, ps in rows:
        pos = np.fromiter((order[int(i)] for i in ps.idx), np.int64, len(ps.idx))
        np.maximum.at(mat[row], pos, ps.hz)
    return idx, mat


def run_once(c: cn.Connectome, pops: list[PopSpec], *, hz: float = 150.0, ms: float = 400.0, settle_ms: float = 200.0,
             batch: int = 64, readouts=("motor",), pattern: str | None = None, by_side: bool = True,
             n_null: int = 4, params=None, optic_params=None, device=None, context=None, seed: int = 0,
             modules=None, frame_ms: float = common.FRAME_MS, per_body: bool = True, include_pn: bool = False,
             fb=None, quiet: bool = True, progress=None) -> AtlasRun:
    """One independent run: every population in `pops` pulsed once, `n_null` unstimulated rows per batch.

    Builds one `FlyBrain(batch=batch, seed=seed)` (or reuses `fb`) and walks `pops` in chunks of `batch - n_null`.
    Returns the `AtlasRun`; the caller saves it. This is the GPU half -- `scripts/interp_atlas.py run`.
    """
    from ..fly import FlyBrain
    if not pops:
        raise ValueError("run_once needs at least one population to stimulate")
    own = fb is None
    if own:
        fb = FlyBrain(c, modules=modules, batch=int(batch), device=device, seed=int(seed),
                      lif_params=params, optic_params=optic_params)
    c_run = fb.c
    pops = remap(pops, c, c_run)                         # a module subset renumbers the cells the pulses name
    batch = int(fb.B)
    n_null = int(max(0, min(n_null, batch - 1)))
    per_chunk = max(1, batch - n_null)
    groups, derived = readout_groups(c_run, readouts, pattern, by_side, include_pn)
    pooler = _Pooler(groups, fb.brain.device)
    names = list(groups) + [d[0] for d in derived]
    d_plus = [list(groups).index(a) for _, a, _ in derived]
    d_minus = [list(groups).index(b) for _, _, b in derived]
    frozen = common.frozen_indices(fb)

    body_idx = np.zeros(0, np.int64)
    if per_body:
        keep = [v for k, v in groups.items() if not k.startswith("pn.") and not k.startswith("type.")]
        body_idx = np.unique(np.concatenate(keep)) if keep else np.zeros(0, np.int64)
    import torch
    body_t = torch.as_tensor(body_idx, device=fb.brain.device)

    n_settle = int(round(settle_ms / frame_ms))
    P, R = len(pops), len(names)
    vals = {s: np.full((P, R), np.nan, np.float32) for s in SUMMARIES}
    body_vals = np.full((P, len(body_idx)), np.nan, np.float32)
    null_rows: list[dict] = []
    null_vals = {s: [] for s in SUMMARIES}
    null_body: list[np.ndarray] = []
    t0 = time.time()

    for start in range(0, P, per_chunk):
        chunk = pops[start:start + per_chunk]
        ctx_rec = _reset(fb, context)
        pre = np.zeros((max(n_settle, 1), batch, R), np.float32)
        for f in range(max(n_settle, 1)):
            fb.step(frame_ms)
            pre[f] = _with_derived(pooler(fb.brain.rate), d_plus, d_minus)
        rows = [(k, ps) for k, ps in enumerate(chunk) if len(ps.idx)]
        for pulse_ms in sorted({ps.ms for _, ps in rows}):
            sel = [(k, ps) for k, ps in rows if ps.ms == pulse_ms]
            idx, mat = _pulse_matrix(sel, batch)
            if len(idx):
                fb.stimulate(idx, mat, float(pulse_ms))
        ms_row = np.full(batch, float(max([ps.ms for _, ps in rows], default=ms)), np.float64)
        for k, ps in rows:
            ms_row[k] = float(ps.ms)
        n_pulse = int(round(ms_row.max() / frame_ms))
        trace = np.zeros((n_pulse, batch, R), np.float32)
        body_acc = np.zeros((batch, len(body_idx)), np.float64)
        body_n = np.zeros(batch, np.float64)
        for f in range(n_pulse):
            fb.step(frame_ms)
            trace[f] = _with_derived(pooler(fb.brain.rate), d_plus, d_minus)
            if len(body_idx):
                inside = (f + 1) * frame_ms <= ms_row + 1e-9
                body_acc[inside] += fb.brain.rate[:, body_t].detach().cpu().numpy()[inside]
                body_n += inside
        pre_mean = pre[max(1, len(pre) // 2):].mean(axis=0) if len(pre) else np.full((batch, R), np.nan, np.float32)
        for k, ps in enumerate(chunk):
            row = k
            w = max(1, int(round(ps.ms / frame_ms)))
            vals["mean"][start + k] = trace[:w, row].mean(axis=0)
            vals["half"][start + k] = trace[max(1, w // 2):w, row].mean(axis=0)
            vals["final"][start + k] = trace[w - 1, row]
            vals["pre"][start + k] = pre_mean[row]
            if len(body_idx):
                body_vals[start + k] = body_acc[row] / max(body_n[row], 1.0)
        for row in range(len(chunk), batch):
            w = n_pulse
            null_rows.append({"id": f"null:seed{seed}:chunk{start // per_chunk}:row{row}", "seed": int(seed),
                              "chunk": start // per_chunk, "row": row})
            null_vals["mean"].append(trace[:w, row].mean(axis=0))
            null_vals["half"].append(trace[max(1, w // 2):w, row].mean(axis=0))
            null_vals["final"].append(trace[w - 1, row])
            null_vals["pre"].append(pre_mean[row])
            if len(body_idx):
                null_body.append(body_acc[row] / max(body_n[row], 1.0))
        if progress is not None:
            progress(min(start + per_chunk, P), P, time.time() - t0)
        elif not quiet:
            print(f"  atlas seed {seed}: {min(start + per_chunk, P)}/{P} populations, {time.time() - t0:.0f}s", flush=True)

    meta = {"schema": ATLAS_RUN_SCHEMA, "seed": int(seed), "hz": float(hz), "ms": float(ms),
            "settle_ms": float(settle_ms), "batch": batch, "n_null": n_null, "per_chunk": per_chunk,
            "frame_ms": float(frame_ms), "by_side": bool(by_side), "pattern": pattern,
            "readouts_requested": list(readouts), "include_pn": bool(include_pn), "context": ctx_rec,
            "n_cells": {k: int(v) for k, v in zip(names, list(pooler.n_cells) + [0] * len(derived))},
            "derived": [list(d) for d in derived], "wall_s": round(time.time() - t0, 1),
            "provenance": provenance(c_run, params, optic_params, fb=fb, device=device, seeds=[int(seed)],
                                     batch=batch, stimulus=_stimulus_record(pops, hz, ms, settle_ms, ctx_rec, n_null)),
            "n_populations": P, "modules": list(modules) if modules else None}
    run = AtlasRun(meta=meta, pops=[p.record(c_run, frozen) for p in pops], readouts=names, values=vals,
                   null_ids=[r["id"] for r in null_rows],
                   null_values={s: (np.stack(v) if v else np.zeros((0, R), np.float32)) for s, v in null_vals.items()},
                   body_idx=body_idx, body_values=body_vals,
                   body_null=(np.stack(null_body) if null_body else np.zeros((0, len(body_idx)), np.float32)))
    if own:
        del fb
    return run


def _with_derived(x: np.ndarray, plus, minus) -> np.ndarray:
    if not plus:
        return x
    return np.concatenate([x, x[:, plus] - x[:, minus]], axis=1)


def _stimulus_record(pops, hz, ms, settle_ms, ctx_rec, n_null) -> dict:
    return {"protocol": "atlas", "params": {"hz": float(hz), "ms": float(ms), "settle_ms": float(settle_ms),
                                            "n_populations": len(pops), "context": ctx_rec, "n_null_rows": int(n_null),
                                            "stimulus": "FlyBrain.stimulate, Poisson, both sides unless the spec "
                                                        "names one; rates per cell where the population supplies them"},
            "control": f"{n_null} unstimulated rows of the same FlyBrain batch, same seed, same context"}


# ------------------------------------------------------------------------------------------------ analysis
def readout_membership(c: cn.Connectome, meta: dict) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Rebuild a run's readout cell sets on the CPU: ({readout: boolean mask over c}, {readout: n cells}).

    A left-right readout's mask is the union of its two sides. Used to mark the rows where the stimulated population
    is itself part of the readout -- `turn_L` is DNa02_L, `gf` is DNp01, `wind_ipsi_L` contains DNp18_L, so
    stimulating those cells moves those readouts by construction and not through a circuit.
    """
    groups, derived = readout_groups(c, tuple(meta.get("readouts_requested", ("motor",))), meta.get("pattern"),
                                     bool(meta.get("by_side", True)), bool(meta.get("include_pn", False)))
    masks = {}
    for k, v in groups.items():
        m = np.zeros(c.n, bool)
        m[np.asarray(v, np.int64)] = True
        masks[k] = m
    for name, a, b in [tuple(d) for d in meta.get("derived", [])]:
        if a in masks and b in masks:
            masks[name] = masks[a] | masks[b]
    n_cells = {k: int(len(v)) for k, v in groups.items()}
    for name, a, b in [tuple(d) for d in meta.get("derived", [])]:
        n_cells.setdefault(name, n_cells.get(a, 0) + n_cells.get(b, 0))
    return masks, n_cells


def pruned_out_shares(c: cn.Connectome, pop_idx: dict[str, np.ndarray]) -> dict[str, float]:
    """{population: the share of its outgoing |W| that lands on cells the LIF does not run}.

    With `LIFParams.prune_frozen` (the shipped default) `FlyBrain` freezes and prunes the optic rate units
    (`OpticLobe.rate_idx` = superclass ol_intrinsic that are not photoreceptors), so a pulse into a population whose
    output goes there reaches nothing in the LIF -- the path exists only inside the rate-model optic lobe. A share
    near 1 is the reason an atlas row reads null, and the tool says so instead of leaving it unexplained.
    """
    import scipy.sparse as sp
    W = c.W
    A = sp.csr_matrix((np.abs(W.data), W.indices, W.indptr), shape=W.shape)
    nrn = c.neurons
    is_pr = nrn.type.fillna("").isin(cn.PHOTORECEPTOR_TYPES).to_numpy()
    pruned = np.flatnonzero((nrn.superclass.fillna("") == "ol_intrinsic").to_numpy() & ~is_pr)
    total = np.asarray(A.sum(axis=0)).ravel()
    onto_pruned = np.asarray(A[pruned].sum(axis=0)).ravel() if len(pruned) else np.zeros(c.n)
    out = {}
    for lab, idx in pop_idx.items():
        t = float(total[idx].sum()) if len(idx) else 0.0
        out[lab] = float(onto_pruned[idx].sum() / t) if t > 0 else float("nan")
    return out


def _pop_indices(c: cn.Connectome, pop_rec: dict) -> dict[str, np.ndarray]:
    """{population label: its indices in `c`}, from the recorded bodyIds (or the recorded spec when ids were dropped)."""
    out: dict[str, np.ndarray] = {}
    for lab, p in pop_rec.items():
        ids = p.get("body_ids")
        if ids:
            j = c.body_to_index.reindex(np.array([int(b) for b in ids], np.int64)).to_numpy()
            out[lab] = j[np.isfinite(j.astype(np.float64))].astype(np.int64)
        else:
            try:
                out[lab] = resolve(c, p.get("spec"))
            except Exception:                            # noqa: BLE001 -- a spec this connectome cannot resolve
                out[lab] = np.zeros(0, np.int64)
    return out


def _overlaps(c: cn.Connectome, pop_rec: dict, masks: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """{population label: {readout: how many of its cells are in that readout's cell set}} (only non-zero entries)."""
    out: dict[str, dict[str, int]] = {}
    for lab, idx in _pop_indices(c, pop_rec).items():
        out[lab] = {k: int(m[idx].sum()) for k, m in masks.items() if len(idx) and m[idx].any()}
    return out


def analyse_runs(runs, *, c: cn.Connectome | None = None, top: int = 20, z_min: float = Z_RESULT,
                 min_n: int = MIN_REPLICATES, sd_floor: float = SD_FLOOR_HZ, summary: str = "mean",
                 per_body_for=None, validation: bool = True, generator: str | None = None) -> Result:
    """Combine independent runs into the `Result`: per (population, readout) the arm comparison against the null rows.

    `runs`: AtlasRun objects or paths. The stimulus arm of a (population, readout) is that population's value in each
    run (the replicate unit is the run); the null arm is every unstimulated row of every run for the same readout.
    `common.compare` gives z / Welch / U / p / verdict; `z_floor` and `verdict_atlas` repeat it with the null SD
    floored at `sd_floor` Hz so a readout with a bit-identical null (SD 0, z NaN) is still judged instead of being
    reported as null. `summary` picks which pulse-window statistic is the headline ('mean', 'half', 'final', 'pre').
    """
    runs = [r if isinstance(r, AtlasRun) else AtlasRun.load(r) for r in runs]
    if not runs:
        raise ValueError("analyse_runs needs at least one run")
    labels = [p["label"] for p in runs[0].pops]
    common_labels = [l for l in labels if all(l in set(r.labels) for r in runs)]
    readouts = [r0 for r0 in runs[0].readouts if all(r0 in set(r.readouts) for r in runs)]
    pos = [{l: i for i, l in enumerate(r.labels)} for r in runs]
    rpos = [{k: i for i, k in enumerate(r.readouts)} for r in runs]
    pop_rec = {p["label"]: p for p in runs[0].pops}
    masks, ro_cells = readout_membership(c, runs[0].meta) if c is not None else ({}, {})
    overlap = _overlaps(c, pop_rec, masks) if c is not None else {}
    pruned_share = pruned_out_shares(c, _pop_indices(c, pop_rec)) if c is not None else {}

    rows = []
    for j, ro in enumerate(readouts):
        null = np.concatenate([r.null_values[summary][:, rpos[k][ro]] for k, r in enumerate(runs)
                               if len(r.null_values[summary])]) if any(len(r.null_values[summary]) for r in runs) else np.zeros(0)
        null_stats = common.ArmStats.of(null)
        for lab in common_labels:
            stim = np.array([runs[k].values[summary][pos[k][lab], rpos[k][ro]] for k in range(len(runs))])
            cmp = compare(stim, null, z_min=z_min, min_n=min_n)
            diff = cmp["diff"]
            sd = null_stats.sd if np.isfinite(null_stats.sd) else 0.0
            z_floor = diff / max(sd, sd_floor)
            if min(cmp["stim"]["n"], cmp["null"]["n"]) < min_n:
                verdict = "underpowered"
            elif abs(z_floor) >= z_min and (not np.isfinite(cmp["p"]) or cmp["p"] <= 0.05):
                verdict = "result"
            else:
                verdict = "null"
            p = pop_rec[lab]
            shared = int(overlap.get(lab, {}).get(ro, 0))
            rows.append({"self_drive": bool(shared), "readout_shared_cells": shared,
                         "readout_n_cells": int(ro_cells.get(ro, 0)),
                         "out_to_pruned_share": pruned_share.get(lab),
                         "population": lab, "spec": p.get("spec"), "n_cells": p.get("n_cells"),
                         "hz": p.get("hz_max"), "ms": p.get("ms"), "frozen_frac": p.get("frozen_frac"),
                         "readout": ro, "n_runs": int(cmp["stim"]["n"]), "stim_mean": cmp["stim"]["mean"],
                         "stim_sd": cmp["stim"]["sd"], "null_mean": cmp["null"]["mean"], "null_sd": cmp["null"]["sd"],
                         "null_n": int(cmp["null"]["n"]), "diff": diff, "z": cmp["z"], "z_floor": float(z_floor),
                         "welch": cmp["welch"], "U": cmp["U"], "p": cmp["p"], "verdict_compare": cmp["verdict"],
                         "verdict": verdict, "stim_values": [float(x) for x in stim],
                         "final_mean": float(np.mean([runs[k].values["final"][pos[k][lab], rpos[k][ro]]
                                                      for k in range(len(runs))])),
                         "pre_mean": float(np.mean([runs[k].values["pre"][pos[k][lab], rpos[k][ro]]
                                                    for k in range(len(runs))]))})
    df = pd.DataFrame(rows)

    prov = dict(runs[0].meta["provenance"])
    # The run jobs execute inside the cluster's per-run copy of the checkout, which carries no .git, so their
    # `flyverse_commit` reads 'unknown'; the desktop that analyses them is the checkout that produced the code, and
    # its state is recorded alongside rather than overwriting what the run actually reported.
    prov["flyverse_commit_analysis"] = common.git_state()
    prov["execution"] = dict(prov["execution"])
    prov["execution"]["devices"] = [r.meta["provenance"]["execution"].get("device") for r in runs]
    prov["execution"]["seeds"] = {"brain": [r.meta["seed"] for r in runs], "env": None}
    prov["execution"]["replicate_unit"] = "runs"
    res = Result.new("atlas", prov)
    res.replicates = {"n": len(runs), "unit": "runs",
                      "runs": [{"run_index": k, "seed": r.meta["seed"], "file": r.meta.get("file"),
                                "device": r.meta["provenance"]["execution"].get("device"),
                                "wall_s": r.meta.get("wall_s")} for k, r in enumerate(runs)],
                      "null": {"rows_per_run": runs[0].meta["n_null"], "ids": runs[0].null_ids[:64],
                               "n_draws": int(sum(len(r.null_ids) for r in runs))}}
    res.add_table("atlas", df)
    movers = _movers(df, top)
    res.add_table("movers", movers)
    if c is not None:
        res.add_table("readout_per_body", _per_body_table(c, runs, df, per_body_for, summary))
        for lab in common_labels[:2000]:
            p = pop_rec[lab]
            res.populations.append({"label": lab, "spec": p.get("spec"), "n_cells": p.get("n_cells"),
                                    "body_ids": p.get("body_ids", []), "unit_kind": None,
                                    "frozen_frac": p.get("frozen_frac"),
                                    "out_to_pruned_share": pruned_share.get(lab)})
    res.summary = _summary(df, movers, runs, summary, sd_floor)
    if validation:
        res.validation = dict(res.validation, **validate(df, runs))
    res.files = {"recordings": [r.meta.get("file") for r in runs],
                 "generator": generator or "flyverse/interp/atlas.py::analyse_runs"}
    return res


def _movers(df: pd.DataFrame, top: int) -> pd.DataFrame:
    """The `top` populations per readout by |difference from the null|, self-drive marked.

    A population whose difference is exactly zero did not move the readout at all (the readout is silent in both
    arms) and is not listed -- a readout with fewer than `top` movers gets a shorter table rather than a padded one.
    """
    out = []
    for ro, g in df.groupby("readout", sort=False):
        g = g[g["diff"].abs() > 0]
        g = g.reindex(g["diff"].abs().sort_values(ascending=False).index).head(top)
        for rank, (_, r) in enumerate(g.iterrows(), 1):
            out.append({"readout": ro, "rank": rank, "population": r.population, "n_cells": r.n_cells,
                        "hz": r.hz, "stim_mean": r.stim_mean, "null_mean": r.null_mean, "diff": r["diff"],
                        "z": r.z, "z_floor": r.z_floor, "p": r.p, "verdict": r.verdict,
                        "self_drive": bool(r.get("self_drive", False)),
                        "readout_shared_cells": int(r.get("readout_shared_cells", 0) or 0),
                        "stim_sd": r.stim_sd, "n_runs": r.n_runs})
    return pd.DataFrame(out)


def _per_body_table(c: cn.Connectome, runs, df: pd.DataFrame, per_body_for, summary: str) -> pd.DataFrame:
    """Neurome `readout_per_body` rows for the readout cells: one row per (population, readout body)."""
    r0 = runs[0]
    if not len(r0.body_idx):
        return pd.DataFrame(columns=common.EXPORT_TABLES["readout_per_body"])
    if per_body_for is None:
        per_body_for = list(dict.fromkeys(df[df.verdict == "result"].sort_values("diff", key=abs, ascending=False)
                                          .population.tolist()[:8]))
    per_body_for = [l for l in per_body_for if l in set(r0.labels)]
    if not per_body_for:
        return pd.DataFrame(columns=common.EXPORT_TABLES["readout_per_body"])
    kinds = common.unit_kinds(c)
    types = c.neurons.type.fillna("").to_numpy()
    bodies = c.neurons.bodyId.to_numpy()
    pos = [{l: i for i, l in enumerate(r.labels)} for r in runs]
    ctrl = np.concatenate([r.body_null for r in runs if len(r.body_null)]) if any(len(r.body_null) for r in runs) else None
    ctrl_mean = ctrl.mean(axis=0) if ctrl is not None and len(ctrl) else np.zeros(len(r0.body_idx))
    ctrl_ids = [i for r in runs for i in r.null_ids][:16]
    settle_s = r0.meta["settle_ms"] / 1000.0
    rows = []
    for lab in per_body_for:
        vals = np.stack([runs[k].body_values[pos[k][lab]] for k in range(len(runs))])
        m, sd = vals.mean(axis=0), (vals.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(vals.shape[1]))
        w_end = settle_s + float(next((p["ms"] for p in r0.pops if p["label"] == lab), r0.meta["ms"])) / 1000.0
        for j, i in enumerate(r0.body_idx):
            rows.append({"bodyId": str(int(bodies[i])), "model_index": int(i), "type": types[i],
                         "unit_kind": kinds[i], "quantity": "output_Hz", "window_start_s": settle_s,
                         "window_end_s": w_end, "stimulus_value": float(m[j]), "control_value": float(ctrl_mean[j]),
                         "stimulus_minus_control": float(m[j] - ctrl_mean[j]), "unit": "Hz", "n_trials": len(runs),
                         "trial_sd": float(sd[j]), "control_ids": ctrl_ids, "population": lab, "summary": summary})
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame, movers: pd.DataFrame, runs, summary: str, sd_floor: float) -> dict:
    res = df[df.verdict == "result"]
    top_pop = {}
    for ro, g in movers.groupby("readout", sort=False):
        g = g[g.verdict == "result"]
        if not len(g):
            continue
        top_pop[ro] = {"population": g.iloc[0].population, "diff": float(g.iloc[0]["diff"]),
                       "z_floor": float(g.iloc[0].z_floor), "self_drive": bool(g.iloc[0].get("self_drive", False))}
        ext = g[~g["self_drive"].astype(bool)] if "self_drive" in g else g
        if len(ext):                                     # the strongest population that is not part of the readout
            top_pop[ro]["top_upstream"] = {"population": ext.iloc[0].population, "diff": float(ext.iloc[0]["diff"])}
    top_ro = {}
    for pop, g in res.groupby("population", sort=False):
        g = g.reindex(g["diff"].abs().sort_values(ascending=False).index)
        top_ro[pop] = {"readout": g.iloc[0].readout, "diff": float(g.iloc[0]["diff"])}
    frozen = df[(df.frozen_frac.fillna(0) > 0.5)].population.unique().tolist()
    pruned = (df[df.out_to_pruned_share.fillna(0) > 0.5].drop_duplicates("population")
              .set_index("population").out_to_pruned_share.round(3).to_dict()) if "out_to_pruned_share" in df else {}
    silent = sorted(set(df.population) - set(df[df["diff"].abs() > 0].population))
    return {"n_populations": int(df.population.nunique()), "n_readouts": int(df.readout.nunique()),
            "n_runs": len(runs), "summary_statistic": summary, "sd_floor_hz": sd_floor,
            "n_results": int(len(res)), "n_populations_with_a_result": int(res.population.nunique()),
            "top_population_per_readout": top_pop,
            "top_readout_per_population": dict(list(top_ro.items())[:400]),
            "populations_mostly_frozen": frozen[:200],
            "populations_whose_output_is_pruned": dict(list(pruned.items())[:200]),
            "n_populations_that_moved_nothing": len(silent), "populations_that_moved_nothing": silent[:200],
            "null_rows_per_run": runs[0].meta["n_null"], "batch": runs[0].meta["batch"]}


def _pick(df: pd.DataFrame, pop: str, ro: str, col: str = "stim_mean"):
    m = df[(df.population == pop) & (df.readout == ro)]
    return float(m.iloc[0][col]) if len(m) and col in m.columns else float("nan")


def _per_run(df: pd.DataFrame, pop: str, ro: str) -> list:
    """The population's value of one readout in each independent run (the replicate unit), for the scatter."""
    m = df[(df.population == pop) & (df.readout == ro)]
    if not len(m) or "stim_values" not in m.columns:
        return []
    return [float(x) for x in m.iloc[0]["stim_values"]]


def _flip_per_run(df: pd.DataFrame, a: str, b: str, ro: str) -> list:
    """(L-R in arm `a`) - (L-R in arm `b`) per run: the wind flip with its run-to-run scatter."""
    va, vb = _per_run(df, a, ro), _per_run(df, b, ro)
    return [float(x - y) for x, y in zip(va, vb)] if len(va) == len(vb) and va else []


def validate(df: pd.DataFrame, runs) -> dict:
    """Fill VALIDATION['atlas'] from the measured table: the wind flips, PFL3 -> DNa02 and DNa02 -> legs."""
    ref = common.VALIDATION["atlas"]["reference"]
    measured: dict = {}
    ok: list[bool] = []
    have = set(df.population.unique())
    if {"JO_wind_left", "JO_wind_right"} <= have:
        for t, key, bound in (("DNp18", "wind.DNp18_flip_hz", 15.0), ("DNp33", "wind.DNp33_flip_hz", -15.0)):
            ro = f"type.{t}_LR"
            L, R = _pick(df, "JO_wind_left", ro), _pick(df, "JO_wind_right", ro)
            flip = L - R
            per_run = _flip_per_run(df, "JO_wind_left", "JO_wind_right", ro)
            measured[key] = {"flip_hz": flip, "wind_left_LR_hz": L, "wind_right_LR_hz": R,
                             "wind_head_on_LR_hz": _pick(df, "JO_wind_head_on", ro) if "JO_wind_head_on" in have else None,
                             "flip_per_run_hz": per_run, "flip_sd_hz": float(np.std(per_run, ddof=1)) if len(per_run) > 1 else None,
                             "reference": ref.get(key), "criterion": f"{'>=' if bound > 0 else '<='} {bound}"}
            ok.append(bool(flip >= bound) if bound > 0 else bool(flip <= bound))
        for t in ("DNge016", "DNg99", "DNg05_a", "DNp19", "DNp73", "WED080"):
            ro = f"type.{t}_LR"
            if ((df.readout == ro).any()):
                per_run = _flip_per_run(df, "JO_wind_left", "JO_wind_right", ro)
                measured[f"wind.{t}_flip_hz"] = {
                    "flip_hz": _pick(df, "JO_wind_left", ro) - _pick(df, "JO_wind_right", ro),
                    "wind_head_on_LR_hz": _pick(df, "JO_wind_head_on", ro) if "JO_wind_head_on" in have else None,
                    "flip_per_run_hz": per_run,
                    "flip_sd_hz": float(np.std(per_run, ddof=1)) if len(per_run) > 1 else None}
    if "DNa02_L" in have:
        measured["DNa02_L_150Hz"] = {"leg_L_hz": _pick(df, "DNa02_L", "leg_L"), "leg_R_hz": _pick(df, "DNa02_L", "leg_R"),
                                     "leg_asym_hz": _pick(df, "DNa02_L", "leg_LR"),
                                     "leg_asym_hz_final_frame": _pick(df, "DNa02_L", "leg_LR", "final_mean"),
                                     "leg_asym_per_run_hz": _per_run(df, "DNa02_L", "leg_LR"),
                                     "leg_asym_sd_hz": _pick(df, "DNa02_L", "leg_LR", "stim_sd"),
                                     "reference": ref.get("DNa02_L_150Hz"),
                                     "criterion": "left minus right leg-MN mean > 0.3 Hz (benchmark dn.DNa02_L_leg_asym_hz)"}
        ok.append(bool(measured["DNa02_L_150Hz"]["leg_asym_hz"] > 0.3))
    if "PFL3_L" in have and (df.readout == "type.DNa02_R").any():
        measured["PFL3_L_80Hz"] = {"DNa02_R_hz": _pick(df, "PFL3_L", "type.DNa02_R"),
                                   "DNa02_L_hz": _pick(df, "PFL3_L", "type.DNa02_L"),
                                   "DNa02_R_per_run_hz": _per_run(df, "PFL3_L", "type.DNa02_R"),
                                   "DNa02_R_sd_hz": _pick(df, "PFL3_L", "type.DNa02_R", "stim_sd"),
                                   "reference": ref.get("PFL3_L_80Hz"),
                                   "criterion": "contralateral DNa02 above ipsilateral"}
        ok.append(bool(measured["PFL3_L_80Hz"]["DNa02_R_hz"] > measured["PFL3_L_80Hz"]["DNa02_L_hz"]))
    if not measured:
        return {"measured": None, "status": "not run"}
    status = "reproduced" if all(ok) else ("not reproduced" if not any(ok) else "partly reproduced")
    measured["n_runs"] = len(runs)
    measured["protocol"] = {"settle_ms": runs[0].meta["settle_ms"], "summary": "pulse-window mean",
                            "null": f"{runs[0].meta['n_null']} unstimulated rows per batch"}
    return {"measured": to_jsonable(measured), "status": status}


# ------------------------------------------------------------------------------------------------ the tool
def atlas(c: cn.Connectome, populations, *, hz: float = 150.0, ms: float = 400.0, settle_ms: float = 200.0,
          batch: int = 64, readouts=("motor",), pattern=None, by_side: bool = True, null: bool = True,
          replicates: int = MIN_REPLICATES, params=None, optic_params=None, device=None, context=None, seed: int = 0,
          split: str | None = "type", n_null: int = 4, min_cells: int = 1, modules=None, include_pn: bool = False,
          per_body: bool = True, top: int = 20, summary: str = "mean", out_dir=None, quiet: bool = True,
          sd_floor: float = SD_FLOOR_HZ, per_body_for=None) -> Result:
    """Stimulate each population briefly and record every motor readout, with a null (docs/INTERP.md 4.5).

    One `FlyBrain(batch=batch)`; row r stimulates population r (`FlyBrain.stimulate` at `hz` for `ms`, both sides
    unless the spec names one) after `settle_ms` of quiet, and `n_null` rows of the same batch receive no pulse --
    those are the control. `replicates` independent runs (seeds `seed`, `seed+1`, ...) give the scatter; per
    (population, readout) the pulse-window means are compared with `common.compare` against every null row of every
    run. Readouts: 'motor' = every pooled `MotorRates` field plus the left-right differences, and every type
    `pattern` matches through `screen.TypeRecorder` (by soma side when `by_side`). `context` names a sensory context
    (`CONTEXTS`: wind_left / wind_right / wind_head_on / wind_off / sugar, a {sense: args} dict, or a callable
    driving the FlyBrain's senses) held on for the whole run, default none -- the `screen_dns.py` protocol.

    `populations` follows `common.resolve`'s grammar; with `split='type'` a spec is split into one population per
    type (and side), and a list of dicts {'label', 'spec'/'idx', 'hz', 'ms'} names populations explicitly with their
    own rates (that is how the JO wind arms present the exact per-cell rates `senses.Wind` produces). Validation:
    `common.VALIDATION['atlas']` -- the DNp18 / DNp33 wind flips, PFL3 -> DNa02, DNa02 -> leg motor neurons.

    Tables: 'atlas' (population x readout with the arm comparison, `self_drive` / `readout_shared_cells` /
    `out_to_pruned_share`), 'movers' (the top `top` populations per readout that moved it at all),
    'readout_per_body' (Neurome fields for the readout cells of the strongest populations).
    """
    pops = populations if (populations and isinstance(populations[0], PopSpec)) else \
        make_populations(c, populations, by_side=by_side, split=split, hz=hz, ms=ms, min_cells=min_cells)
    runs = []
    for k, s in enumerate(common.replicate_seeds(replicates, seed)):
        run = run_once(c, pops, hz=hz, ms=ms, settle_ms=settle_ms, batch=batch, readouts=readouts, pattern=pattern,
                       by_side=by_side, n_null=(n_null if null else 0), params=params, optic_params=optic_params,
                       device=device, context=context, seed=s, modules=modules, per_body=per_body,
                       include_pn=include_pn, quiet=quiet)
        if out_dir is not None:
            path = run.save(Path(out_dir) / f"run_r{k}")
            run.meta["file"] = str(path)
        runs.append(run)
    return analyse_runs(runs, c=c, top=top, summary=summary, sd_floor=sd_floor, per_body_for=per_body_for,
                        generator="flyverse/interp/atlas.py::atlas")
