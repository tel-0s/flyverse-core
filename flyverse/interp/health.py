"""flyverse.interp.health -- per-type operating-point statistics of a rollout, and the observatory readout source.

The question this tool answers is not "what does the fly do" but "what state is each population in while it does
it": silent, at threshold, refractory-limited, adapted, excitation- or inhibition-dominated, fan-in-scaled, or cut
off by sign-0 / frozen inputs. It reads a `common.Recording` (rate_hz and, when recorded, v_mv / adapt_mv / refrac /
spike_count of a cell set) plus, for the structural columns, the Connectome and the shaped weights
(`common.effective_weights`); it changes nothing.

    from flyverse.interp import health, HealthReadout
    res = health(recording, c=c, params=lif, window=(5.5, 8.0), presyn=presyn_recording)   # Result
    res.table("per_type")                                                                   # one row per type
    fb.nt_source = HealthReadout(fb)                                                        # live channels in the map

Per group (a type by default; `by` = 'type' | 'type_side' | 'module' | 'superclass' | 'class' | 'nt' | {label: spec}):

    rate_mean / rate_max        mean over cells and frames of the running rate; the largest per-cell time-max (Hz)
    spike_rate_hz               spikes over the window / duration (from spike_count; an estimate independent of rate_tau)
    silent_frac                 fraction of cells whose max rate over the window is below thresholds['never_firing_hz']
    at_threshold_frac           fraction of cells whose mean v over non-refractory frames is within
                                thresholds['at_threshold_mv'] of v_th   (needs v_mv; refrac excludes reset frames)
    refractory_load             mean over cells of rate x t_ref (the fraction of time spent refractory);
                                refractory_load_max the largest cell; refractory_limited_frac the cells above
                                thresholds['refractory_limited'] (0.4: the 200-260 Hz bump at t_ref 2.2 ms is 0.44-0.57)
    refrac_measured             mean of the recorded refrac flag (the same quantity measured, not inferred)
    adapt_load                  mean adapt / (v_th - v_rest)     (needs adapt_mv)
    e_in_mv_s / i_in_mv_s       rate-weighted synaptic input from the recorded presynaptic cells, split by sign:
                                sum_j max(A_ij, 0) r_j and sum_j min(A_ij, 0) r_j in mV/s (decompose's 'current');
                                ei_balance = (E - |I|) / (E + |I|) in [-1, +1]; input_coverage = the share of |A| input
                                weight whose presynaptic cell is in `presyn` (default: the recording itself)
    fanin_scale_min/med/max     EffectiveWeights.scale (the input-normalisation factor per cell)
    sign0_in_share              raw synapses onto the group from sign-0 presynaptic cells / all raw input synapses
    sign0_out_share             raw output synapses of the group that carry no sign / all its output synapses
    frozen_in_share / frozen_out_share    the same for optic rate units (never spike in the LIF)
    state_flags                 'silent' (no cell above never_firing_hz), 'mostly_silent' (silent_frac >= 0.5),
                                'refractory_limited' (refractory_load_max >= 0.4), 'high_rate' (rate_mean >= 40 Hz),
                                'at_threshold' (at_threshold_frac >= 0.5), 'adapted' (adapt_load >= 0.5),
                                'sign0_output' (sign0_out_share >= 0.5), 'sign0_input' (sign0_in_share >= 0.15,
                                the NT audit's 15 % bound); every bound in DEFAULT_THRESHOLDS / `thresholds`

With `recording=None` and a Connectome the structural columns alone are computed for every cell of `c` (the NT
audit's group tables: docs/audits/nt_audit.md 1a / 1b) and the summary carries the global sign-0 counts.

Validation target (common.VALIDATION['health']): the 200 Hz refractory-limited bump (EPG 180-260 Hz -> load
0.40-0.57 at t_ref 2.2; PEN 40-65, Delta7 90-112 Hz) and the sign-0 populations of the NT audit. The generator of
those numbers is scripts/interp_health.py (record on the cluster, analyse / structure / validate on the CPU); the
reproduction is docs/audits/interp_health.md.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import common
from .common import Recording, Result, EffectiveWeights, Population, NEVER_FIRING_HZ
from .. import connectome as cn
from .. import regions
from ..nt_readout import NTChannel, NTSnapshot

DEFAULT_THRESHOLDS = {
    "never_firing_hz": NEVER_FIRING_HZ,   # a cell whose max rate over the window stays below this is silent
    "at_threshold_mv": 1.0,               # mean non-refractory v within this of v_th
    "refractory_limited": 0.4,            # rate x t_ref above this: the refractory period bounds the rate
    "high_rate_hz": 40.0,                 # a population mean above this is 'high_rate' (E-PG in the animal: < 20 Hz)
    "adapt_high": 0.5,                    # mean adapt / (v_th - v_rest)
    "mostly_silent_frac": 0.5,
    "at_threshold_frac": 0.5,
    "sign0_output_share": 0.5,
    "sign0_input_share": 0.15,            # docs/audits/nt_audit.md section 3's bound
}
GROUPINGS = ("type", "type_side", "module", "superclass", "class", "nt")
LIF_QUANTITIES = ("rate_hz", "v_mv", "adapt_mv", "refrac", "spike_count")
# the per-cell columns the Neurome readout_per_body table carries, with their units
BODY_QUANTITIES = {"rate_hz": "Hz", "spike_rate_hz": "Hz", "v_margin_mv": "mV", "refractory_load": "fraction",
                   "adapt_load": "fraction", "ei_balance": "index", "input_coverage": "fraction", "fanin_scale": "factor",
                   "sign0_in_share": "fraction"}


# ---------------------------------------------------------------------------------------------- parameters
def params_from_meta(meta: dict):
    """A LIFParams rebuilt from a recording's provenance block (model.lif, defaults resolved), or None."""
    from .. import brain
    lif = (meta or {}).get("provenance", {}).get("model", {}).get("lif")
    if not lif:
        return None
    names = {f.name for f in dataclasses.fields(brain.LIFParams)}
    kw = {}
    for k, v in lif.items():
        if k not in names:
            continue
        if k in ("path_gain", "type_path_gain") and isinstance(v, list):
            v = [tuple(x) for x in v]
        if k == "slow_gain_clip" and isinstance(v, list):
            v = tuple(v)
        kw[k] = v
    try:
        return brain.LIFParams(**kw)
    except TypeError:
        return None


def resolve_params(params=None, recording: Recording | None = None):
    """The LIFParams the statistics are judged against: `params`, else the recording's provenance, else the defaults."""
    if params is not None:
        return params
    p = params_from_meta(recording.meta) if recording is not None else None
    if p is not None:
        return p
    from .. import brain
    return brain.LIFParams()


def full_counts(c: cn.Connectome) -> tuple[sp.csr_matrix, bool]:
    """Raw, unsigned, uncapped synapse counts on every stored entry of c.W: |W| where the entry is signed (c.W holds the
    raw signed counts before the connection cap) and, where it is an explicit zero (sign 0), the count from
    cache/sign0_counts.npz (connectome.sign0_counts). Returns (counts csr, sign0_counts_available). This is the matrix
    the NT audit's tables are built from (docs/audits/nt_audit.md: 124,161,873 synapses, 2,701,289 of them sign 0)."""
    C = c.W.tocsr().copy()
    C.data = np.abs(C.data).astype(np.float64)               # float64: the 124,161,873-synapse total is exact
    avail = False
    try:
        cnt = cn.sign0_counts(c)
        if cnt is not None and len(np.asarray(cnt)) == C.nnz:
            zero = C.data == 0
            C.data[zero] = np.asarray(cnt, dtype=np.float64)[zero]
            avail = True
    except Exception:  # noqa: BLE001 -- the raw weights table is not on every machine
        pass
    return C, avail


def presynaptic_indices(c: cn.Connectome, idx, exclude=None) -> np.ndarray:
    """Every cell with a stored entry onto `idx` (sign-0 explicit zeros included), minus `exclude`."""
    idx = np.asarray(idx)
    pre = np.unique(c.W.tocsr()[idx].indices) if len(idx) else np.zeros(0, np.int64)
    if exclude is not None and len(pre):
        pre = np.setdiff1d(pre, np.asarray(exclude))
    return pre


# ---------------------------------------------------------------------------------------------- grouping
def group_labels(c, idx, by, recording: Recording | None = None) -> np.ndarray:
    """One group label per cell of `idx` (object array; None = in no group). `by`: a GROUPINGS name, any neurons
    column, or {label: spec} (specs in the common selection grammar or index arrays; first match wins)."""
    idx = np.asarray(idx)
    if isinstance(by, dict):
        if c is None:
            raise ValueError("a {label: spec} grouping needs the Connectome")
        out = np.full(len(idx), None, dtype=object)
        pos = -np.ones(c.n, np.int64); pos[idx] = np.arange(len(idx))
        for label, spec in by.items():
            sel = common.resolve(c, spec)
            p = pos[sel]; p = p[p >= 0]
            fresh = p[[out[q] is None for q in p]] if len(p) else p
            out[fresh] = label
        return out
    if by == "type":
        t = recording.types.astype(str) if (c is None and recording is not None) else c.neurons.type.fillna("").to_numpy()[idx]
        return np.asarray(t, dtype=object)
    if by == "type_side":
        if c is None:
            if recording is None:
                raise ValueError("type_side needs a Connectome or a recording")
            t = recording.types.astype(str); s = recording.meta.get("sides", ["?"] * len(idx))
        else:
            t = c.neurons.type.fillna("").to_numpy()[idx]; s = c.neurons.somaSide.fillna("?").to_numpy()[idx]
        return np.array([f"{a}_{b}" for a, b in zip(t, s)], dtype=object)
    if c is None:
        raise ValueError(f"grouping by {by!r} needs the Connectome")
    if by == "module":
        return regions.labels(c)[idx].astype(object)
    if by in c.neurons.columns:
        return c.neurons[by].fillna("").astype(str).to_numpy()[idx].astype(object)
    raise ValueError(f"unknown grouping {by!r}; choose from {GROUPINGS}, a neurons column, or a dict of specs")


# ---------------------------------------------------------------------------------------------- per-cell statistics
def _nanmean0(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=0)


def cell_dynamics(R: Recording, lif, thr: dict) -> pd.DataFrame:
    """Per recorded cell: rate_mean, rate_max, silent, refractory_load and, when the quantity was recorded,
    spike_rate_hz, refrac_measured, v_mean / v_margin_mv / at_threshold, adapt_mean / adapt_load."""
    if R.batched:
        raise ValueError("health works on one fly at a time: pass Recording.row(b) for a batch member")
    if R.n_frames == 0:
        raise ValueError("the recording (or the window) holds no frames")
    rate = np.asarray(R.quantities["rate_hz"], dtype=np.float64)
    n = rate.shape[1]
    d = pd.DataFrame({"rate_mean": rate.mean(axis=0), "rate_max": rate.max(axis=0)})
    d["silent"] = d.rate_max < thr["never_firing_hz"]
    d["refractory_load"] = d.rate_mean * float(lif.t_ref) / 1000.0
    T = R.n_frames
    dt_ms = float(np.median(np.diff(R.t_ms))) if T > 1 else common.FRAME_MS
    if "spike_count" in R.quantities and T > 1:
        sc = np.asarray(R.quantities["spike_count"], dtype=np.float64)
        d["spike_rate_hz"] = (sc[-1] - sc[0]) / ((T - 1) * dt_ms / 1000.0)
    else:
        d["spike_rate_hz"] = np.nan
    notref = np.ones((T, n), bool)
    if "refrac" in R.quantities:
        rf = np.asarray(R.quantities["refrac"], dtype=np.float64)
        d["refrac_measured"] = rf.mean(axis=0)
        notref = rf <= 0
    else:
        d["refrac_measured"] = np.nan
    if "v_mv" in R.quantities:
        v = np.asarray(R.quantities["v_mv"], dtype=np.float64)
        vm = _nanmean0(np.where(notref, v, np.nan))
        d["v_mean_mv"] = vm
        d["v_margin_mv"] = float(lif.v_th) - vm
        d["at_threshold"] = np.abs(d.v_margin_mv) <= thr["at_threshold_mv"]
    else:
        d["v_mean_mv"] = np.nan; d["v_margin_mv"] = np.nan; d["at_threshold"] = np.nan
    if "adapt_mv" in R.quantities:
        a = np.asarray(R.quantities["adapt_mv"], dtype=np.float64).mean(axis=0)
        d["adapt_mean_mv"] = a
        d["adapt_load"] = a / max(float(lif.v_th - lif.v_rest), 1e-9)
    else:
        d["adapt_mean_mv"] = np.nan; d["adapt_load"] = np.nan
    return d


def cell_structure(c: cn.Connectome, idx, ew: EffectiveWeights, counts: sp.csr_matrix, frozen_mask: np.ndarray,
                   sign0_mask: np.ndarray) -> pd.DataFrame:
    """Per cell of `idx`: fanin_scale, in_syn / in_sign0 / in_frozen (raw synapses onto the cell), out_syn and the
    sign0 / frozen flags of the cell itself (its whole output is silenced when set)."""
    idx = np.asarray(idx)
    Cc = counts.tocsr()
    rows = Cc[idx]
    out_all = np.asarray(counts.sum(axis=0)).ravel()
    d = pd.DataFrame({"fanin_scale": ew.scale[idx].astype(np.float64),
                      "in_syn": np.asarray(rows.sum(axis=1)).ravel(),
                      "in_sign0": np.asarray(rows[:, np.flatnonzero(sign0_mask)].sum(axis=1)).ravel(),
                      "in_frozen": np.asarray(rows[:, np.flatnonzero(frozen_mask)].sum(axis=1)).ravel(),
                      "out_syn": out_all[idx], "sign0": sign0_mask[idx], "frozen": frozen_mask[idx]})
    d["out_sign0"] = np.where(d.sign0, d.out_syn, 0.0)
    d["out_frozen"] = np.where(d.frozen, d.out_syn, 0.0)
    return d


def cell_input_balance(ew: EffectiveWeights, idx, presyn: Recording) -> pd.DataFrame:
    """Per cell of `idx`: e_in_mv_s / i_in_mv_s = the rate-weighted positive / negative input from the cells of
    `presyn` (their mean rate_hz over its frames), ei_balance and input_coverage (share of |A| input covered)."""
    if presyn.batched:
        raise ValueError("presyn must be a single-fly recording (Recording.row)")
    idx = np.asarray(idx)
    r = np.asarray(presyn.quantities["rate_hz"], dtype=np.float64).mean(axis=0)
    A = ew.A[idx].tocsr()
    blk = A[:, presyn.idx].tocsr()
    pos = blk.copy(); pos.data = np.maximum(pos.data, 0.0)
    neg = blk.copy(); neg.data = np.minimum(neg.data, 0.0)
    e = np.asarray(pos @ r).ravel(); i = np.asarray(neg @ r).ravel()
    tot = np.asarray(abs(A).sum(axis=1)).ravel(); cov = np.asarray(abs(blk).sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        bal = np.where(e - i > 0, (e + i) / (e - i), np.nan)
        coverage = np.where(tot > 0, cov / tot, np.nan)
    return pd.DataFrame({"e_in_mv_s": e, "i_in_mv_s": i, "ei_balance": bal, "input_coverage": coverage})


# ---------------------------------------------------------------------------------------------- aggregation
def _mode(s: pd.Series):
    v = s.dropna()
    return v.mode().iloc[0] if len(v) else None


def _share(num: pd.Series, den: pd.Series) -> float:
    d = float(den.sum())
    return float(num.sum()) / d if d > 0 else float("nan")


def flags_for(row: pd.Series, thr: dict) -> str:
    """The flag string of one aggregated group row (see the module docstring)."""
    f = []
    if np.isfinite(row.get("rate_max", np.nan)):
        if row["rate_max"] < thr["never_firing_hz"]:
            f.append("silent")
        elif row.get("silent_frac", 0) >= thr["mostly_silent_frac"]:
            f.append("mostly_silent")
        if row.get("refractory_load_max", 0) >= thr["refractory_limited"]:
            f.append("refractory_limited")
        if row.get("rate_mean", 0) >= thr["high_rate_hz"]:
            f.append("high_rate")
        if np.isfinite(row.get("at_threshold_frac", np.nan)) and row["at_threshold_frac"] >= thr["at_threshold_frac"]:
            f.append("at_threshold")
        if np.isfinite(row.get("adapt_load", np.nan)) and row["adapt_load"] >= thr["adapt_high"]:
            f.append("adapted")
    if np.isfinite(row.get("sign0_out_share", np.nan)) and row["sign0_out_share"] >= thr["sign0_output_share"]:
        f.append("sign0_output")
    if np.isfinite(row.get("sign0_in_share", np.nan)) and row["sign0_in_share"] >= thr["sign0_input_share"]:
        f.append("sign0_input")
    return "|".join(f)


def aggregate(cells: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """The per-group table from the per-cell frame (cells with group None are dropped)."""
    d = cells[cells.group.notna()]
    if len(d) == 0:
        return pd.DataFrame()
    g = d.groupby("group", sort=True)
    out = pd.DataFrame({"n_cells": g.size()})
    out["unit_kind"] = g["unit_kind"].agg(_mode) if "unit_kind" in d else None
    dyn = "rate_mean" in d
    if dyn:
        out["rate_mean"] = g.rate_mean.mean(); out["rate_max"] = g.rate_max.max()
        out["spike_rate_hz"] = g.spike_rate_hz.mean()
        out["silent_frac"] = g.silent.mean()
        out["at_threshold_frac"] = g.at_threshold.apply(lambda s: float(np.nanmean(s.astype(float))) if s.notna().any() else np.nan)
        out["v_margin_mv"] = g.v_margin_mv.mean()
        out["refractory_load"] = g.refractory_load.mean(); out["refractory_load_max"] = g.refractory_load.max()
        out["refractory_limited_frac"] = g.refractory_load.apply(lambda s: float((s >= thr["refractory_limited"]).mean()))
        out["refrac_measured"] = g.refrac_measured.mean()
        out["adapt_load"] = g.adapt_load.mean()
        if "ei_balance" in d:
            out["e_in_mv_s"] = g.e_in_mv_s.mean(); out["i_in_mv_s"] = g.i_in_mv_s.mean()
            out["ei_balance"] = g.ei_balance.mean(); out["input_coverage"] = g.input_coverage.mean()
    if "fanin_scale" in d:
        out["fanin_scale_min"] = g.fanin_scale.min(); out["fanin_scale_med"] = g.fanin_scale.median(); out["fanin_scale_max"] = g.fanin_scale.max()
        out["in_syn"] = g.in_syn.sum(); out["out_syn"] = g.out_syn.sum()
        out["n_sign0_pre"] = g.apply(lambda x: int((x.sign0 & (x.out_syn > 0)).sum()))
        out["sign0_in_share"] = g.apply(lambda x: _share(x.in_sign0, x.in_syn))
        out["sign0_out_share"] = g.apply(lambda x: _share(x.out_sign0, x.out_syn))
        out["frozen_in_share"] = g.apply(lambda x: _share(x.in_frozen, x.in_syn))
        out["frozen_out_share"] = g.apply(lambda x: _share(x.out_frozen, x.out_syn))
    out["state_flags"] = [flags_for(r, thr) for _, r in out.iterrows()]
    out.index.name = "group"
    return out.reset_index()


def sign0_summary(c: cn.Connectome, counts: sp.csr_matrix, sign0_mask: np.ndarray, sign0_counts_available: bool) -> dict:
    """The NT audit's global counts on this graph: sign-0 cells, those with output, their synapses and share."""
    out_all = np.asarray(counts.sum(axis=0)).ravel()
    has_entry = np.diff(c.W.tocsc().indptr) > 0
    total = float(out_all.sum())
    return {"n_neurons": int(c.n), "n_sign0_cells": int(sign0_mask.sum()),
            "n_sign0_with_output_entries": int((sign0_mask & has_entry).sum()),
            "n_sign0_presynaptic": int((sign0_mask & (out_all > 0)).sum()),
            "sign0_synapses": float(out_all[sign0_mask].sum()), "synapses_total": total,
            "sign0_synapse_share": float(out_all[sign0_mask].sum() / total) if total > 0 else float("nan"),
            "sign0_counts_available": bool(sign0_counts_available),
            "note": "raw counts of sign-0 entries need cache/sign0_counts.npz; without it they read 0 and the shares are lower bounds"}


# ---------------------------------------------------------------------------------------------- the tool
def health(recording, *, c=None, params=None, window=None, by="type", ew=None, thresholds=None, fb=None,
           presyn=None, counts=None, flags=None, readout_rows=True, max_readout_cells=20000) -> Result:
    """Per-type operating-point statistics of a rollout (docs/INTERP.md 4.6; the module docstring lists every column).

    recording: a single-fly common.Recording with rate_hz (+ v_mv / adapt_mv / refrac / spike_count when recorded), or
    None for the structural columns of every cell of `c`. c: the Connectome (None = dynamic columns only, grouping
    by 'type' / 'type_side'). params: LIFParams (None = the recording's provenance, else the defaults). window:
    (start_s, end_s). by: the grouping (GROUPINGS, a neurons column, or {label: spec}). ew: a precomputed
    EffectiveWeights under `params`. thresholds: overrides of DEFAULT_THRESHOLDS. fb: a FlyBrain (its optic rate
    units are the frozen set; else the static ol_intrinsic rule). presyn: a Recording with rate_hz of the target's
    presynaptic cells for the E/I columns (default: the recording itself). counts: a raw-count matrix (default
    full_counts: |W| on signed entries plus cache/sign0_counts.npz on the sign-0 zeros). flags: a silent_flags frame is accepted for API symmetry (unused: the shares are computed
    from `counts`). readout_rows: write the Neurome readout_per_body rows (skipped above max_readout_cells).
    """
    thr = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    lif = resolve_params(params, recording)
    if recording is None and c is None:
        raise ValueError("health needs a recording, a Connectome, or both")
    R = None
    if recording is not None:
        R = recording.window(*window) if window is not None else recording
        idx = np.asarray(R.idx); types = np.asarray(R.types).astype(str); bodies = np.asarray(R.body_ids)
        if c is not None and (idx.max(initial=-1) >= c.n or not np.array_equal(c.neurons.bodyId.to_numpy()[idx], bodies)):
            raise ValueError("the recording's cells are not the Connectome's rows: it was made on another graph / subset")
    else:
        idx = np.arange(c.n); types = c.neurons.type.fillna("").to_numpy().astype(str); bodies = c.neurons.bodyId.to_numpy()
    cells = pd.DataFrame({"index": idx, "bodyId": bodies, "type": types})
    cells["group"] = group_labels(c, idx, by, R)
    cells["unit_kind"] = common.unit_kinds(c, fb)[idx] if c is not None else None
    if R is not None:
        cells = pd.concat([cells, cell_dynamics(R, lif, thr)], axis=1)
    struct_summary = {}
    if c is not None:
        ew = ew or common.effective_weights(c, lif)
        sign0_avail = True
        if counts is None:
            counts, sign0_avail = full_counts(c)
        sign = c.neurons["sign"].to_numpy() if "sign" in c.neurons else np.ones(c.n)
        sign0_mask = sign == 0
        frozen_mask = np.zeros(c.n, bool)
        fr = common.frozen_indices(fb)
        frozen_mask[fr if fr is not None else np.flatnonzero(common.unit_kinds(c) == "graded")] = True
        cells = pd.concat([cells, cell_structure(c, idx, ew, counts, frozen_mask, sign0_mask)], axis=1)
        struct_summary = sign0_summary(c, counts, sign0_mask, sign0_avail)
        if R is not None:
            P = presyn if presyn is not None else recording
            if "rate_hz" not in P.quantities:
                raise ValueError("presyn needs rate_hz")
            Pw = P.window(*window) if (window is not None and presyn is not None) else (P if presyn is not None else R)
            cells = pd.concat([cells, cell_input_balance(ew, idx, Pw)], axis=1)
    per = aggregate(cells, thr)
    per.insert(1, "by", by if isinstance(by, str) else "custom")

    # ---- provenance / result
    meta = dict(recording.meta) if recording is not None else {}
    prov = meta.get("provenance")
    if not prov:
        if c is not None:
            prov = common.provenance(c, lif, fb=fb, seeds=[meta.get("seed")] if meta.get("seed") is not None else None,
                                     stimulus=meta.get("stimulus") or {"protocol": meta.get("protocol"), "params": {}, "control": None})
        else:
            prov = {"flyverse_commit": common.git_state(), "dataset_release": common.dataset_release(),
                    "compiled_connectome": {}, "model": common.model_record(lif), "execution": common.execution_record(fb),
                    "stimulus": meta.get("stimulus") or {"protocol": meta.get("protocol"), "params": {}, "control": None},
                    "retina": {"file": None, "n_columns": None, "column_to_bodies": None}, "units": common.UNITS}
    if R is None:                                   # structural-only: the numbers were computed here, on the CPU, without a simulation
        ex = dict(prov.get("execution", {}))
        ex.update({"device": "cpu", "device_requested": "cpu", "note": "structural computation on the cached connectome; no rollout"})
        prov = dict(prov, execution=ex)
    res = Result.new("health", prov)
    res.add_table("per_type", per)
    if R is not None:
        res.replicates = {"n": 1, "unit": "runs", "runs": [{"run_index": 0, "seed": meta.get("seed"), "file": meta.get("file"),
                                                            "device": prov.get("execution", {}).get("device")}], "null": None}
        label = meta.get("label", "recorded")
        res.add_population(Population(label, meta.get("selection", f"<{len(idx)} recorded cells>"), idx, bodies),
                           unit_kind=_mode(cells.unit_kind) if "unit_kind" in cells else None, keep_ids=len(idx) <= 10000)
        if readout_rows and len(idx) <= max_readout_cells:
            res.add_table("readout_per_body", body_rows(cells, R))
    ws = [float(R.t_ms[0]) / 1000.0, float(R.t_ms[-1]) / 1000.0] if R is not None else None
    res.summary = {"n_cells": int(len(idx)), "n_groups": int(len(per)), "by": by if isinstance(by, str) else list(by),
                   "window_s": list(window) if window is not None else ws, "n_frames": int(R.n_frames) if R is not None else 0,
                   "quantities": list(R.quantities) if R is not None else [], "thresholds": thr,
                   "lif": {"v_th": float(lif.v_th), "v_rest": float(lif.v_rest), "v_reset": float(lif.v_reset), "t_ref_ms": float(lif.t_ref)},
                   "flag_counts": {f: int(per["state_flags"].str.contains(f).sum()) for f in
                                   ("silent", "mostly_silent", "refractory_limited", "high_rate", "at_threshold", "adapted", "sign0_output", "sign0_input")} if len(per) else {},
                   "flagged": {g: f for g, f in zip(per["group"], per["state_flags"]) if f} if len(per) else {},
                   "presyn": None if (R is None or c is None) else {"n_cells": int(len((presyn or recording).idx)),
                                                                     "coverage_mean": float(np.nanmean(cells.input_coverage)) if cells.input_coverage.notna().any() else None,
                                                                     "source": "presyn recording" if presyn is not None else "the recording itself"},
                   "sign0": struct_summary,
                   "ew_md5": ew.md5 if c is not None else None}
    res.files = {"recording": meta.get("file"), "presyn": getattr(presyn, "meta", {}).get("file") if presyn is not None else None,
                 "generator": "flyverse.interp.health.health"}
    res._cells = cells      # the per-cell frame for callers that want it (not serialised)
    return res


def body_rows(cells: pd.DataFrame, R: Recording) -> pd.DataFrame:
    """The Neurome readout_per_body rows (common.EXPORT_TABLES) of the per-cell frame: one row per (body, quantity)."""
    ws, we = float(R.t_ms[0]) / 1000.0, float(R.t_ms[-1]) / 1000.0
    rows = []
    src = {"rate_hz": "rate_mean", "spike_rate_hz": "spike_rate_hz", "v_margin_mv": "v_margin_mv", "refractory_load": "refractory_load",
           "adapt_load": "adapt_load", "ei_balance": "ei_balance", "input_coverage": "input_coverage", "fanin_scale": "fanin_scale",
           "sign0_in_share": None}
    have = [q for q, col in src.items() if (col in cells.columns and cells[col].notna().any()) or (q == "sign0_in_share" and "in_syn" in cells)]
    for r in cells.itertuples():
        for q in have:
            if q == "sign0_in_share":
                v = r.in_sign0 / r.in_syn if r.in_syn > 0 else np.nan
            else:
                v = getattr(r, src[q])
            if not np.isfinite(v):
                continue
            rows.append({"bodyId": str(int(r.bodyId)), "model_index": int(r.index), "type": r.type, "unit_kind": r.unit_kind,
                         "quantity": q, "window_start_s": ws, "window_end_s": we, "stimulus_value": float(v), "control_value": None,
                         "stimulus_minus_control": None, "unit": BODY_QUANTITIES[q], "n_trials": 1, "trial_sd": None, "control_ids": []})
    return pd.DataFrame(rows, columns=common.EXPORT_TABLES["readout_per_body"])


# ---------------------------------------------------------------------------------------------- replicates
STATS = ("rate_mean", "rate_max", "spike_rate_hz", "silent_frac", "at_threshold_frac", "v_margin_mv", "refractory_load",
         "refractory_load_max", "refractory_limited_frac", "refrac_measured", "adapt_load", "ei_balance", "e_in_mv_s", "i_in_mv_s",
         "input_coverage", "fanin_scale_med", "sign0_in_share", "sign0_out_share")


def replicate_table(results: list[Result], table: str = "per_type") -> pd.DataFrame:
    """Per (group, statistic) the mean / sd / n / values over independent runs' `table` rows."""
    frames = [r.table(table).assign(run_index=k) for k, r in enumerate(results)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows = []
    if len(df) == 0:
        return pd.DataFrame(rows)
    for grp, sub in df.groupby("group", sort=True):
        for s in STATS:
            if s not in sub:
                continue
            a = common.ArmStats.of(sub[s].to_numpy(dtype=np.float64))
            rows.append({"group": grp, "stat": s, "mean": a.mean, "sd": a.sd, "n": a.n, "values": a.values,
                         "n_cells": int(sub.n_cells.iloc[0]), "state_flags": "|".join(sorted({f for x in sub["state_flags"].fillna("") for f in x.split("|") if f}))})
    return pd.DataFrame(rows)


def compare_arms(stim: list[Result], null: list[Result], table: str = "per_type", stats=("rate_mean", "silent_frac", "at_threshold_frac", "refractory_load")) -> pd.DataFrame:
    """common.compare per (group, stat) between two replicate arms (e.g. receptor default vs off)."""
    a = replicate_table(stim, table); b = replicate_table(null, table)
    rows = []
    if len(a) == 0 or len(b) == 0:
        return pd.DataFrame(rows)
    bi = b.set_index(["group", "stat"])
    for r in a.itertuples():
        if r.stat not in stats or (r.group, r.stat) not in bi.index:
            continue
        q = bi.loc[(r.group, r.stat)]
        cmp = common.compare(r.values, q["values"])
        rows.append({"group": r.group, "stat": r.stat, "stim_mean": cmp["stim"]["mean"], "stim_sd": cmp["stim"]["sd"], "stim_n": cmp["stim"]["n"],
                     "null_mean": cmp["null"]["mean"], "null_sd": cmp["null"]["sd"], "null_n": cmp["null"]["n"], "diff": cmp["diff"],
                     "z": cmp["z"], "welch": cmp["welch"], "U": cmp["U"], "p": cmp["p"], "verdict": cmp["verdict"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- the readout source
class HealthReadout:
    """The observatory readout source (docs/NT_READOUT.md NTSource): per-neuron operating-point channels sampled live
    from a FlyBrain, as an NTSnapshot keyed by bodyId. Channels: rate_hz (Brain.rate), v_margin_mv (v_th - v),
    adapt_mv, refractory (0 / 1), fanin_scale (Brain.input_scale), drive_mv (the injected current) and optic_rate
    (the OpticLobe's rate units, [0-1]; NaN elsewhere). The LIF channels are NaN on the optic rate units (frozen:
    they never spike). Owned CPU copies; sampled only when the map asks (never from the stepping loop). Attach with
    `fb.nt_source = HealthReadout(fb)`; no UI file changes."""

    CHANNELS = {"rate_hz": ("Hz", 0.0, 200.0), "v_margin_mv": ("mV", 0.0, 7.0), "adapt_mv": ("mV", 0.0, 10.0),
                "refractory": ("0/1", 0.0, 1.0), "fanin_scale": ("factor", 0.0, 1.0), "drive_mv": ("mV", 0.0, 30.0),
                "optic_rate": ("rate units", 0.0, 1.0)}
    LIF_CHANNELS = ("rate_hz", "v_margin_mv", "adapt_mv", "refractory", "drive_mv")

    def __init__(self, fb, channels=("rate_hz", "v_margin_mv", "adapt_mv", "refractory", "fanin_scale")):
        unknown = [ch for ch in channels if ch not in self.CHANNELS]
        if unknown:
            raise ValueError(f"unknown health channels {unknown}; choose from {list(self.CHANNELS)}")
        self.fb = fb
        b = getattr(fb, "brain", fb)
        p = getattr(b, "p", None)
        self.v_th = float(getattr(p, "v_th", -45.0)); self.v_rest = float(getattr(p, "v_rest", -52.0))
        spec = dict(self.CHANNELS); spec["v_margin_mv"] = ("mV", 0.0, max(self.v_th - self.v_rest, 1e-3))
        self.channels = tuple(NTChannel(ch, *spec[ch]) for ch in channels)
        self.names = tuple(channels)
        self.body_ids = np.asarray(fb.c.neurons.bodyId.to_numpy(), dtype=np.int64)
        self.n = len(self.body_ids)
        fr = common.frozen_indices(fb)
        self.frozen = np.zeros(self.n, bool)
        if fr is not None:
            self.frozen[np.asarray(fr)] = True

    def _row(self, x, batch_index: int) -> np.ndarray:
        a = np.asarray(common._np(x), dtype=np.float32)
        return a[batch_index] if a.ndim == 2 else a

    def readout(self, *, batch_index: int = 0) -> NTSnapshot | None:
        b = getattr(self.fb, "brain", self.fb)
        B = int(getattr(b, "B", 1))
        if not 0 <= int(batch_index) < B:
            raise IndexError(batch_index)
        levels = np.full((self.n, len(self.names)), np.nan, dtype=np.float32)
        for k, ch in enumerate(self.names):
            if ch == "rate_hz":
                v = self._row(b.rate, batch_index)
            elif ch == "v_margin_mv":
                v = self.v_th - self._row(b.v, batch_index)
            elif ch == "adapt_mv":
                v = self._row(b.adapt, batch_index)
            elif ch == "refractory":
                v = (self._row(b.refrac, batch_index) > 0).astype(np.float32)
            elif ch == "drive_mv":
                v = self._row(b.drive, batch_index)
            elif ch == "fanin_scale":
                s = getattr(b, "input_scale", None)
                v = np.ones(self.n, np.float32) if s is None else np.asarray(s, dtype=np.float32)
            else:                                                   # optic_rate
                v = np.full(self.n, np.nan, np.float32)
                optic = getattr(self.fb, "optic", None)
                if optic is not None:
                    v[np.asarray(optic.rate_idx)] = self._row(optic.rates(), batch_index)
            v = np.array(v, dtype=np.float32, copy=True)
            if ch in self.LIF_CHANNELS:
                v[self.frozen] = np.nan
            levels[:, k] = v
        levels[~np.isfinite(levels) & ~np.isnan(levels)] = np.nan       # +-inf never reaches the map
        return NTSnapshot(max(float(getattr(b, "t", 0.0)), 0.0), self.body_ids, self.channels, levels)
