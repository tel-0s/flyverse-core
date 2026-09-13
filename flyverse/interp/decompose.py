"""decompose -- what drives a cell set, per presynaptic type / transmitter / receptor tier / sign / module / side.

The first tool of the interpretability toolkit (docs/INTERP.md 4.1). It reads the model and never changes it: every
number comes from `common.effective_weights` (= the matrix Brain installs, brain._shaped_weights x fan-in scale x
w_syn) and from Recordings written by `scripts/interp_decompose.py record`.

Three questions, one function each:

* `decompose(c, target, recording=None, ...)` -- **static** (no recording): per (target type, presynaptic group) the
  effective input one volley of the group would deliver, in mV per post cell per presynaptic volley (the
  `EffectiveWeights.type_matrix` quantity, docs/audits/cx_glno.md section 1 as a function), with the raw uncapped
  count, the share of the target's raw input, the sign rule (nt_sign / receptor:<tier>) and the silence flags.
  **Dynamic** (a Recording, or several runs, or several arms of runs): per frame the rate-weighted input
      I_i(t) = sum_j A[i, j] r_j(t)        [mV/s of synaptic input per post cell; kind 'current']
  split by the presynaptic grouping `by`, averaged over `window` and the target cells of each type, per run; with
  `null_recording` every arm is compared with the matched control through `common.compare` (z against the null
  scatter, Welch, U, p, verdict; 'underpowered' below three runs). `g_mv` = value x tau_syn / 1000 is the mean
  synaptic conductance the same input holds (mV), directly comparable with the recorded optic drive (drive_mv),
  which appears as the pseudo-group 'optic_drive' when the recording carries it. Entries whose presynaptic cell is
  a frozen rate unit are pruned from the LIF (their effect arrives through the drive) and are reported as such.
* `contrast(c, target, params_a, params_b, ...)` -- the structural difference between two parameter sets (two
  receptor tables, a gain, a hold): every stored entry onto the target whose SHAPED weight differs (a sign or gain
  decision: grouped by post type, pre type, pre transmitter, tier, sign a -> sign b, with entry and raw synapse
  counts) and, separately, every entry that is merely RESCALED because its postsynaptic cell's fan-in scale moved
  (the second route of receptor_integration.md E.3) -- the tool that shows the 282-synapse taste dependence (E.4) and
  the seven rescaled cells as tables, over the whole brain in seconds (array-level, CSR order of c.W).
  Arms of a dynamic decomposition recorded under different parameters (meta['arm']) are weighted by their own
  effective weights, so a hold-table comparison sees the weight change and the rate change together.
* `arm_params(arm, base)` -- the round-5 arm convention (off / default / hold<Group> / a receptors_by_type.csv path)
  as a LIFParams, shared by the CLI's record and analyse so a recording and its analysis name the model the same way.

Units. Static values are mV per post cell per presynaptic volley (every cell of the group spiking once). Dynamic
values are mV/s per post cell (A in mV per presynaptic spike x rate in Hz). `raw_count` is the uncapped, unsigned
synapse count (cache/sign0_counts.npz for sign-0 entries). Per-type rows are means over the named bodies; nothing
is merged (docs/NEUROME_INTERFACE.md section 2).

Graded targets (ol_intrinsic rate units): the static decomposition through A describes the anatomical shaped weights
onto them, but the LIF never integrates that row -- the optic lobe drives them through its own normalised matrices
(OpticLobe.W_rr / W_rp / W_rs). The dynamic decomposition of a graded target is therefore refused unless
`optic_params` is given, and then it decomposes the recurrent (W_rr x optic_dr) and spiking-feedback (W_rs x rate/100)
inputs from a recording that carries optic_dr; the photoreceptor term (W_rp x contrast activity) is not a Recorder
quantity and is reported as 'not recorded'. That branch is kind 'optic_input' -- an input decomposition, not an output
attribution (docs/INTERP.md section 9).

CPU only; torch is touched only through brain._receptor / _shaped_weights (via common.effective_weights) and, for
graded targets, optic.OpticLobe on the CPU device.
"""
from __future__ import annotations

import dataclasses
import glob as globmod
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import connectome as cn
from .. import regions
from . import common
from .common import (Recording, Result, effective_weights, links, raw_counts, silent_flags, resolve, population, body_str,
                     compare, unit_kinds, to_jsonable, ROOT)

GROUPINGS = ("type", "transmitter", "tier", "sign", "module", "side", "cell")
_GROUP_COL = {"type": "pre_type", "transmitter": "pre_nt", "tier": "tier", "sign": "sign_label", "module": "pre_module",
              "side": "pre_side", "cell": "body_pre"}
ARMS = ("off", "default", "holdBrain", "holdOptic", "holdBrainGlu", "holdBrainHis", "holdKC", "holdDN1")
STATIC_KIND, DYNAMIC_KIND, DRIVE_KIND, OPTIC_KIND = "effective_weight_mV", "current", "drive_mV", "optic_input"
STATIC_UNIT = "mV per post cell per presynaptic volley"
DYNAMIC_UNIT = "mV/s per post cell"
MAX_LINK_ROWS = 5000            # the 'links' table kept in the JSON (the full edge table can be written as CSV by the CLI)
TIMESERIES_GROUPS = 12          # per target type, the groups whose per-frame series the JSON keeps


# ---------------------------------------------------------------------------------------------- arms
def arm_params(arm: str, base=None, out_dir=None):
    """The round-5 arm convention as a LIFParams (dataclasses.replace on `base`, default LIFParams()):
    'off' = receptor_model None (the presynaptic-sign rule); 'default' = receptor 'sign' / net rule 'abs' on the
    shipped table (= LIFParams()'s own default, made explicit); 'hold<Group>' = the same on
    out/receptors_hold<Group>.csv, built by scripts/build_hold_tables.py when the file is missing (the groups KC,
    DN1, Brain, Optic, BrainGlu, BrainHis); a path ending in .csv = that receptor table under 'sign' / 'abs'."""
    from .. import brain
    p = base if base is not None else brain.LIFParams()
    if arm in (None, "", "as-given"):
        return p
    if arm == "off":
        return dataclasses.replace(p, receptor_model=None)
    if arm == "default":
        return dataclasses.replace(p, receptor_model="sign", receptor_net_rule="abs", receptor_table=None)
    if arm.endswith(".csv"):
        if not os.path.isfile(arm):
            raise FileNotFoundError(arm)
        return dataclasses.replace(p, receptor_model="sign", receptor_net_rule="abs", receptor_table=arm)
    if arm.startswith("hold"):
        group = arm[len("hold"):]
        out_dir = str(out_dir or (ROOT / "out"))
        path = os.path.join(out_dir, f"receptors_hold{group}.csv")
        if not os.path.isfile(path):
            if str(ROOT / "scripts") not in sys.path:
                sys.path.insert(0, str(ROOT / "scripts"))
            import build_hold_tables as bht          # noqa: E402 -- the audited hold-table writer, not a re-implementation
            os.makedirs(out_dir, exist_ok=True)
            header, t, _ = bht.hold_table(group)
            bht.write_atomic(path, header + t.to_csv(index=False, lineterminator="\n"))
        return dataclasses.replace(p, receptor_model="sign", receptor_net_rule="abs", receptor_table=path)
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}, a receptors_by_type.csv path, or 'as-given'")


def arm_of(params) -> str:
    """The arm label a LIFParams corresponds to (for the record)."""
    if params.receptor_model is None:
        return "off"
    t = params.receptor_table
    if t is None:
        return "default" if (params.receptor_model, params.receptor_net_rule) == ("sign", "abs") else f"{params.receptor_model}/{params.receptor_net_rule}"
    name = os.path.basename(str(t))
    if name.startswith("receptors_hold") and name.endswith(".csv"):
        return "hold" + name[len("receptors_hold"):-4]
    return str(t)


# ---------------------------------------------------------------------------------------------- edges
def static_frozen(c: cn.Connectome) -> np.ndarray:
    """The optic lobe's rate units by the static rule OpticLobe uses (ol_intrinsic and not a photoreceptor)."""
    n = c.neurons
    is_pr = n.type.isin(cn.PHOTORECEPTOR_TYPES).to_numpy()
    return np.flatnonzero((n.superclass.fillna("") == "ol_intrinsic").to_numpy() & ~is_pr)


def _receptor_for(c, params, receptor=None):
    from .. import brain
    return brain._receptor(c, params, receptor)


def counts_matrix(c: cn.Connectome) -> tuple[sp.csr_matrix, bool]:
    """Raw, unsigned, uncapped synapse counts per stored entry of c.W (post x pre) -- `common.raw_counts`.

    |W.data| merged with cache/sign0_counts.npz on the explicit-zero (sign-0) entries. The private merge this tool
    carried while the shared accessor substituted the sign-0 array for the whole vector is gone (docs/INTERP.md 11,
    defect 1, closed). Returns (counts, sign0_available)."""
    return raw_counts(c)


def edges(c: cn.Connectome, ew, target_idx, receptor=None, counts=None, frozen_idx=None, rates=None) -> tuple[pd.DataFrame, np.ndarray]:
    """The edge table of every stored entry onto `target_idx` (common.links, Neurome names) plus the grouping columns
    pre_module / pre_side / post_side / tier / sign_label. Returns (table, presynaptic indices)."""
    target_idx = np.asarray(target_idx)
    Wc = c.W.tocsr()
    pre_idx = np.unique(Wc[target_idx].tocoo().col)
    if counts is None:
        counts, _ = counts_matrix(c)
    flags = silent_flags(c, pre_idx, frozen_idx=static_frozen(c) if frozen_idx is None else frozen_idx, rates=rates)
    df = links(c, ew, pre=pre_idx, post=target_idx, receptor=receptor, counts=counts, flags=flags)
    n = c.neurons
    lab = regions.labels(c)
    side = n.somaSide.fillna("?").to_numpy() if "somaSide" in n else np.array(["?"] * c.n)
    df["pre_module"] = lab[df.pre_index.to_numpy()]
    df["pre_side"] = side[df.pre_index.to_numpy()]
    df["post_side"] = side[df.post_index.to_numpy()]
    df["tier"] = df.sign_rule
    df["sign_label"] = np.where(df.sign > 0, "E", np.where(df.sign < 0, "I", "0"))
    return df, pre_idx


def group_labels(df: pd.DataFrame, by) -> pd.Series:
    by = (by,) if isinstance(by, str) else tuple(by)
    unknown = set(by) - set(GROUPINGS)
    if unknown:
        raise ValueError(f"unknown grouping {sorted(unknown)}; choose from {GROUPINGS}")
    parts = [df[_GROUP_COL[b]].astype(str) for b in by]
    return parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep="/")


# ---------------------------------------------------------------------------------------------- static
def static_table(df: pd.DataFrame, n_post: dict, by) -> pd.DataFrame:
    """Per (post type, presynaptic group): value = summed effective input / n post cells of the type (mV per post cell
    per presynaptic volley), its E and I parts, entries, presynaptic cells, raw count, share of the type's raw input,
    sign (+1 / -1 / 0 / 'mixed'), silent entries."""
    g = df.assign(pre_group=group_labels(df, by).to_numpy(), pos=df.effective_mv.clip(lower=0), neg=df.effective_mv.clip(upper=0),
                  is_silent=(df.silent != "").astype(int))
    agg = g.groupby(["post_type", "pre_group"], sort=False).agg(
        value=("effective_mv", "sum"), value_E=("pos", "sum"), value_I=("neg", "sum"), n_entries=("effective_mv", "size"),
        n_pre=("pre_index", "nunique"), raw_count=("synaptic_pair_count", "sum"), silent_entries=("is_silent", "sum"),
        sign_rule=("sign_rule", lambda s: "|".join(sorted(set(s))))).reset_index()
    npost = agg.post_type.map(n_post).astype(float)
    for col in ("value", "value_E", "value_I"):
        agg[col] = agg[col] / npost
    agg["n_post"] = npost.astype(int)
    tot = g.groupby("post_type").synaptic_pair_count.sum()
    agg["share"] = agg.raw_count / agg.post_type.map(tot).replace(0, np.nan)
    agg["sign"] = np.where((agg.value_E > 0) & (agg.value_I < 0), "mixed", np.sign(agg.value).astype(int).astype(str))
    agg["kind"] = STATIC_KIND
    agg["unit"] = STATIC_UNIT
    agg["abs_value"] = agg.value.abs()
    agg = agg.sort_values(["post_type", "abs_value"], ascending=[True, False]).drop(columns="abs_value").reset_index(drop=True)
    return agg


def cancelling_pair(per_type: pd.DataFrame, value_col: str = "value") -> dict:
    """Per post type: the largest opposite-signed pair of groups (max over pairs of min(|E|, |I|)), and the E / I totals."""
    out = {}
    for t, sub in per_type.groupby("post_type", sort=False):
        v = sub.set_index("pre_group")[value_col].astype(float)
        pos, neg = v[v > 0].sort_values(ascending=False), v[v < 0].sort_values()
        rec = {"E_total": float(v[v > 0].sum()), "I_total": float(v[v < 0].sum()), "net": float(v.sum()),
               "n_groups": int(len(v)), "n_E": int((v > 0).sum()), "n_I": int((v < 0).sum())}
        if len(pos) and len(neg):
            rec["cancelling_pair"] = {"E": pos.index[0], "E_value": float(pos.iloc[0]), "I": neg.index[0],
                                      "I_value": float(neg.iloc[0]), "cancellation": float(min(pos.iloc[0], -neg.iloc[0]))}
        else:
            rec["cancelling_pair"] = None
        out[t] = rec
    return out


# ---------------------------------------------------------------------------------------------- dynamic
def _as_arms(recording, null_recording) -> tuple[dict, str | None]:
    """{arm label: [Recording, ...]} from a Recording, a list, a dict or a glob; the null arm is labelled 'null'
    unless it is itself a dict with one key. Batched recordings are refused (rows are not replicates)."""
    def load_list(x, label):
        if x is None:
            return []
        if isinstance(x, (str, Path)):
            files = sorted(globmod.glob(str(x))) if any(ch in str(x) for ch in "*?[") else [str(x)]
            files = [f for f in files if f.endswith(".npz")] or files
            if not files:
                raise FileNotFoundError(f"{label}: no recording matches {x}")
            return [Recording.load(f) for f in files]
        if isinstance(x, Recording):
            return [x]
        return [r if isinstance(r, Recording) else Recording.load(r) for r in x]
    arms = {}
    if isinstance(recording, dict):
        for k, v in recording.items():
            arms[str(k)] = load_list(v, k)
    else:
        arms["stim"] = load_list(recording, "recording")
    null_label = None
    if null_recording is not None:
        if isinstance(null_recording, dict):
            (k, v), = null_recording.items()
            null_label = str(k); arms[null_label] = load_list(v, k)
        else:
            null_label = "null"; arms[null_label] = load_list(null_recording, "null")
    for k, runs in arms.items():
        if not runs:
            raise ValueError(f"arm {k!r} has no recordings")
        for r in runs:
            if r.batched:
                raise ValueError(f"arm {k!r}: a batched recording is not a set of replicates; pass Recording.row(b) per row")
            if "rate_hz" not in r.quantities:
                raise ValueError(f"arm {k!r}: the recording carries no rate_hz")
    return arms, null_label


def _run_values(df: pd.DataFrame, rec: Recording, groups: pd.Series, n_post: dict, window, tau_syn: float,
                keep_series: bool = True, eff=None):
    """One run: per (post type, group) the window-mean rate-weighted input (mV/s per post cell), the mean rate of
    the group's presynaptic cells, the per-frame series, the per-target-cell input, the optic-drive pseudo-rows
    and the coverage of the recording over the edge table. `eff` overrides the per-entry effective weight (the
    run's own arm's A entries, in df order); default df.effective_mv."""
    R = rec.window(*window) if window else rec
    if R.n_frames == 0:
        raise ValueError(f"the window {window} holds no frames of a recording spanning {rec.t_ms.min() / 1000:.3f}-{rec.t_ms.max() / 1000:.3f} s")
    pos = pd.Series(np.arange(len(R.idx)), index=R.idx.astype(np.int64))
    pre_pos = df.pre_index.map(pos)
    have = pre_pos.notna().to_numpy()
    r = R.quantities["rate_hz"].astype(np.float64)                          # (T, n_rec)
    eff = df.effective_mv.to_numpy(np.float64) if eff is None else np.asarray(eff, np.float64)
    keys = pd.MultiIndex.from_arrays([df.post_type.to_numpy(), groups.to_numpy()], names=["post_type", "pre_group"])
    codes, uniq = pd.factorize(keys, sort=False)
    npost = np.array([n_post[t] for t, _ in uniq], dtype=np.float64)
    E = int(have.sum())
    pp = pre_pos.to_numpy()[have].astype(np.int64)
    # per-frame group series: S (n_groups x n_rec) with eff / n_post, series = r @ S.T
    S = sp.csr_matrix((eff[have] / npost[codes[have]], (codes[have], pp)), shape=(len(uniq), len(R.idx)))
    series = np.asarray(S @ r.T).T if E else np.zeros((len(r), len(uniq)))   # (T, n_groups)
    mean_rate_pre = r.mean(axis=0)                                            # (n_rec,)
    entry_value = np.full(len(df), np.nan)
    entry_value[have] = eff[have] * mean_rate_pre[pp]
    value = series.mean(axis=0)
    value_E = np.bincount(codes[have], weights=np.clip(entry_value[have], 0, None), minlength=len(uniq)) / npost
    value_I = np.bincount(codes[have], weights=np.clip(entry_value[have], None, 0), minlength=len(uniq)) / npost
    # mean rate of the group's presynaptic cells (unique cells)
    grp_rate = pd.DataFrame({"code": codes[have], "pre": df.pre_index.to_numpy()[have], "rate": mean_rate_pre[pp]}).drop_duplicates(["code", "pre"]) \
        .groupby("code").rate.mean().reindex(range(len(uniq))).to_numpy()
    n_missing = np.bincount(codes[~have], minlength=len(uniq)) if (~have).any() else np.zeros(len(uniq), int)
    per_group = pd.DataFrame({"post_type": [t for t, _ in uniq], "pre_group": [g for _, g in uniq], "value": value,
                              "value_E": value_E, "value_I": value_I, "rate_hz": grp_rate, "g_mv": value * tau_syn / 1000.0,
                              "n_entries_missing": n_missing,
                              "weight_mv_per_volley": np.bincount(codes, weights=eff, minlength=len(uniq)) / npost})
    # per target cell
    post_codes, post_uniq = pd.factorize(df.post_index.to_numpy(), sort=True)
    cell_input = np.bincount(post_codes[have], weights=entry_value[have], minlength=len(post_uniq))
    per_cell = pd.DataFrame({"post_index": post_uniq, "input_current_mV_per_s": cell_input})
    tpos = pos.reindex(post_uniq)
    tin = tpos.notna().to_numpy()
    # the peak view: per post type the frame at which the type's mean output rate peaks in the window (the frame a
    # 'max over frames' check such as walk.GF_max reads), and every group's input at that frame
    per_group["value_at_peak"] = np.nan; per_group["peak_t_ms"] = np.nan; per_group["output_peak_hz"] = np.nan
    if tin.any():
        post_ty = df.drop_duplicates("post_index").set_index("post_index").post_type.reindex(post_uniq).to_numpy()
        for t in np.unique(post_ty):
            cells_t = tpos.to_numpy()[tin & (post_ty == t)].astype(int)
            if len(cells_t) == 0:
                continue
            k = int(np.argmax(r[:, cells_t].mean(axis=1)))
            mt = (per_group.post_type == t).to_numpy()
            per_group.loc[mt, "value_at_peak"] = series[k, mt]
            per_group.loc[mt, "peak_t_ms"] = float(R.t_ms[k]); per_group.loc[mt, "output_peak_hz"] = float(r[k, cells_t].mean())
    if tin.any():
        per_cell.loc[tin, "output_Hz"] = r[:, tpos.to_numpy()[tin].astype(int)].mean(axis=0)
        if "drive_mv" in R.quantities:
            per_cell.loc[tin, "upstream_drive_mV"] = R.quantities["drive_mv"].astype(np.float64)[:, tpos.to_numpy()[tin].astype(int)].mean(axis=0)
    drive_rows = []
    if "drive_mv" in R.quantities and tin.any():
        pt = df.drop_duplicates("post_index").set_index("post_index").post_type
        d = per_cell[tin].assign(post_type=pt.reindex(per_cell.post_index[tin]).to_numpy())
        for t, sub in d.groupby("post_type"):
            drive_rows.append({"post_type": t, "pre_group": "optic_drive", "value": float(sub.upstream_drive_mV.mean() * 1000.0 / tau_syn),
                               "value_E": np.nan, "value_I": np.nan, "rate_hz": np.nan, "g_mv": float(sub.upstream_drive_mV.mean()),
                               "n_entries_missing": 0, "kind": DRIVE_KIND})
    coverage = {"entries": int(len(df)), "entries_recorded": E, "abs_weight_recorded_share": float(np.abs(eff[have]).sum() / max(np.abs(eff).sum(), 1e-12)),
                "presynaptic_cells_missing": int(df.pre_index[~have].nunique()), "frames": int(R.n_frames),
                "window_s": [float(R.t_ms.min() / 1000.0), float(R.t_ms.max() / 1000.0)]}
    return per_group, drive_rows, per_cell, (series if keep_series else None), R.t_ms, entry_value, coverage


def _arm_stats(values_by_run: list) -> dict:
    a = common.ArmStats.of(values_by_run)
    return {"mean": a.mean, "sd": a.sd, "n": a.n, "values": a.values}


# ---------------------------------------------------------------------------------------------- graded (optic) targets
def optic_matrices(c: cn.Connectome, optic_params=None, receptor=None, receptor_gain=None):
    """The OpticLobe's own normalised matrices on the CPU device (the exact objects optic.py builds: W_rr, W_rs, W_rp
    as scipy csr over rate_idx / spk_idx / pr_idx, and the params) -- built through optic.OpticLobe, not re-derived."""
    from .. import optic, retina as retina_mod
    r = retina_mod.build_retina(c)
    ol = optic.OpticLobe(c, r, optic_params, device="cpu", receptor=receptor, receptor_gain=receptor_gain)

    def to_scipy(M):
        M = M.to_sparse_coo().coalesce()
        i = M.indices().numpy(); v = M.values().numpy()
        return sp.csr_matrix((v, (i[0], i[1])), shape=tuple(M.shape))
    return {"W_rr": to_scipy(ol.W_rr), "W_rs": to_scipy(ol.W_rs), "W_rp": to_scipy(ol.W_rp), "rate_idx": ol.rate_idx,
            "spk_idx": ol.spk_idx, "pr_idx": ol.pr_idx, "params": ol.p}


def _graded_dynamic(c, target_idx, arms, null_label, optic_params, receptor, receptor_gain, by, window, top):
    """kind 'optic_input': per (target type, group) the recurrent (gain_rr x W_rr x optic_dr) and spiking-feedback
    (gain_fb x W_rs x clip(rate / 100, 0, 3)) inputs to the graded target cells over the window, per run and arm."""
    om = optic_matrices(c, optic_params, receptor, receptor_gain)
    p = om["params"]
    n = c.neurons
    ty = n.type.fillna("").to_numpy(); nt = n["nt"].fillna("unknown").to_numpy() if "nt" in n else np.array(["unknown"] * c.n)
    lab = regions.labels(c); side = n.somaSide.fillna("?").to_numpy()
    rpos = pd.Series(np.arange(len(om["rate_idx"])), index=om["rate_idx"])
    tr = rpos.reindex(target_idx).dropna().astype(int).to_numpy()
    rows = []
    for name, M, src_idx, q, gain, xform in (("recurrent", om["W_rr"], om["rate_idx"], "optic_dr", p.gain_rr, lambda x: x),
                                            ("spiking_feedback", om["W_rs"], om["spk_idx"], "rate_hz", p.gain_fb, lambda x: np.clip(x / 100.0, 0, 3))):
        sub = M[tr].tocoo()
        pre = src_idx[sub.col]; post = target_idx[rpos.reindex(target_idx).notna().to_numpy()][sub.row]
        e = pd.DataFrame({"post_index": post, "pre_index": pre, "w": sub.data * gain, "post_type": ty[post], "pre_type": ty[pre], "pre_nt": nt[pre],
                          "pre_module": lab[pre], "pre_side": side[pre], "tier": "optic_norm", "sign_label": np.where(sub.data > 0, "E", np.where(sub.data < 0, "I", "0")),
                          "body_pre": body_str(n.bodyId.to_numpy()[pre]), "term": name, "quantity": q, "xform": xform})
        rows.append(e)
    E_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    n_post = {t: int((ty[target_idx] == t).sum()) for t in np.unique(ty[target_idx])}
    out = []
    for arm, runs in arms.items():
        for k, rec in enumerate(runs):
            R = rec.window(*window) if window else rec
            pos = pd.Series(np.arange(len(R.idx)), index=R.idx.astype(np.int64))
            for term, e in E_all.groupby("term"):
                q = e.quantity.iloc[0]
                if q not in R.quantities:
                    continue
                x = R.quantities[q].astype(np.float64)
                pp = e.pre_index.map(pos)
                have = pp.notna().to_numpy()
                mean_x = np.nanmean(e.xform.iloc[0](x), axis=0)
                val = np.full(len(e), np.nan); val[have] = e.w.to_numpy()[have] * mean_x[pp.to_numpy()[have].astype(int)]
                g = e.assign(pre_group=group_labels(e, by).to_numpy(), val=val)
                agg = g.groupby(["post_type", "pre_group"]).val.sum().reset_index()
                agg["value"] = agg.val / agg.post_type.map(n_post); agg["term"] = term; agg["arm"] = arm; agg["run"] = k
                out.append(agg.drop(columns="val"))
    long = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["post_type", "pre_group", "value", "term", "arm", "run"])
    long["kind"] = OPTIC_KIND; long["unit"] = "rate units per optic step (input to the unit's leaky integrator)"
    return long, {"photoreceptor_term": "not recorded (W_rp x contrast activity is not a Recorder quantity)", "gain_rr": p.gain_rr, "gain_fb": p.gain_fb, "norm": p.norm}


# ---------------------------------------------------------------------------------------------- the tool
def decompose(c, target, *, recording=None, params=None, optic_params=None, receptor=None, by=("type",), tiers=True,
              window=None, kind="current", null_recording=None, fb=None, top=40, counts=None, ew=None, frozen=None,
              arm_label=None, keep_links=True, arm_weights_override=None) -> Result:
    """Signed input to the cell set `target` decomposed by presynaptic population (docs/INTERP.md 4.1; the contract is
    flyverse.interp.stubs.decompose).

    Static (recording None): per (target type, pre group) the effective input a volley of the group would deliver
    (mV per post cell per presynaptic volley) with raw counts, share, sign rule and silence flags. Dynamic (recording
    = a Recording, a list of runs, a {arm: [runs]} dict, or a glob): per run the window-mean rate-weighted input
    I_i(t) = sum_j A[i, j] r_j(t) (mV/s per post cell, kind 'current') split by `by` (a subset of GROUPINGS); with
    `null_recording` every arm is compared with the matched control through common.compare. `tiers` adds the
    'per_tier' table (the grouping plus the receptor tier). `window` = (start_s, end_s). `top` bounds the printed /
    body-level tables. Extra keywords: `counts` / `ew` reuse a raw-count matrix / EffectiveWeights; `frozen` names the
    rate units (default: the static ol_intrinsic rule); `arm_label` labels a static result; `keep_links` keeps the
    edge table in the JSON (capped at MAX_LINK_ROWS). Runs recorded under another arm (meta['arm'] = off / hold<Group>
    / a table path, as `scripts/interp_decompose.py record --arm` writes it) are weighted by that arm's own effective
    weights (`arm_params`), so an arm comparison sees both the rate change and the weight change; `arm_weights_override`
    = {arm label: LIFParams} names them explicitly. The static reference columns (weight_mv_per_volley, raw_count,
    sign_rule) are `params`'; each arm's own volley weight is `<arm>_weight_mv_per_volley`.

    Tables: 'per_type', 'per_tier', 'contributions' (Neurome fields; body-level rows for the top groups),
    'readout_per_body', 'links', 'timeseries' (dynamic). Summary: per target type the E / I totals and the largest
    cancelling pair; the arms and their comparison; the recording coverage.
    """
    from .. import brain
    if kind not in ("current",):
        raise ValueError("kind: only 'current' (rate-weighted synaptic input) is defined; graded targets switch to 'optic_input' automatically")
    p = params if params is not None else brain.LIFParams()
    by = (by,) if isinstance(by, str) else tuple(by)
    pop = population(c, target, "target")
    target_idx = pop.idx
    if len(target_idx) == 0:
        raise ValueError(f"target {target!r} selects no cell")
    rec_signs = _receptor_for(c, p, receptor)
    if ew is None:
        ew = effective_weights(c, p, rec_signs)
    if counts is None:
        counts, sign0_ok = counts_matrix(c)
    else:
        sign0_ok = None
    kinds = unit_kinds(c, fb)
    graded_target = bool((kinds[target_idx] == "graded").all()) and len(target_idx) > 0
    ty = c.neurons.type.fillna("").to_numpy()
    n_post = {t: int((ty[target_idx] == t).sum()) for t in np.unique(ty[target_idx])}
    frozen_idx = common.frozen_indices(fb) if fb is not None else (np.asarray(frozen) if frozen is not None else None)

    arms, null_label = (_as_arms(recording, null_recording) if recording is not None else ({}, None))
    rates_for_flags = None
    if arms:
        first = next(iter(arms.values()))[0]
        rates_for_flags = {int(i): float(v) for i, v in zip(first.idx, first.quantities["rate_hz"].max(axis=0))}
    df, pre_idx = edges(c, ew, target_idx, receptor=rec_signs, counts=counts, frozen_idx=frozen_idx, rates=rates_for_flags)
    groups = group_labels(df, by)
    reference_graph = common.connectome_fingerprint(c)["md5"]
    normalisation = f"input_norm alpha {p.input_norm_alpha} ref {p.input_norm_ref}"
    window_rec = list(window) if window else None

    # ---- provenance
    if arms:
        meta0 = next(iter(arms.values()))[0].meta
        prov = dict(meta0.get("provenance") or common.provenance(c, p, optic_params, fb=fb))
        seeds = [r.meta.get("seed") for runs in arms.values() for r in runs]
        prov["execution"] = dict(prov.get("execution", {}), seeds={"brain": seeds, "env": prov.get("execution", {}).get("seeds", {}).get("env")},
                                 replicate_unit="runs")
        if prov["execution"].get("device") is None:
            prov["execution"]["device"] = str(getattr(getattr(fb, "brain", None), "device", None)) if fb is not None else None
        prov["stimulus"] = meta0.get("stimulus") or prov.get("stimulus")
    else:
        prov = common.provenance(c, p, optic_params, fb=fb, stimulus={"protocol": "static", "params": {}, "control": None})
        if prov["execution"].get("device") is None:
            prov["execution"]["device"] = "cpu"
            prov["execution"]["backend"] = dict(prov["execution"].get("backend", {}), structural=True, simulation=False)
    res = Result.new("decompose", prov)
    res.add_population(pop, unit_kind=("graded" if graded_target else ("mixed" if len(set(kinds[target_idx])) > 1 else str(kinds[target_idx][0]))))
    res.add_population(population(c, pre_idx, "presynaptic"), unit_kind=None, keep_ids=len(pre_idx) <= 10000)

    # ---- static tables (always: the structural decomposition is the reference of the dynamic one)
    per_static = static_table(df, n_post, by)
    per_tier = static_table(df, n_post, tuple(by) + (("tier",) if "tier" not in by else ())) if tiers else None
    summary = {"target": common.spec_repr(target), "n_target": int(len(target_idx)), "n_post_by_type": n_post, "by": list(by),
               "arm": arm_label or arm_of(p), "receptor_model": p.receptor_model, "receptor_table": p.receptor_table,
               "n_presynaptic": int(len(pre_idx)), "n_entries": int(len(df)), "sign0_counts_available": sign0_ok,
               "silent_entries": {k: int((df.silent.str.contains(k)).sum()) for k in ("sign0", "frozen", "pruned", "never_firing")},
               "static": cancelling_pair(per_static), "graded_target": graded_target, "window_s": window_rec,
               "reference_graph": reference_graph, "effective_weights_md5": ew.md5, "shaped_md5": ew.shaped_md5,
               "fanin_scale_target": {t: {"min": float(ew.scale[target_idx[ty[target_idx] == t]].min()), "max": float(ew.scale[target_idx[ty[target_idx] == t]].max())} for t in n_post}}

    if not arms:
        res.add_table("per_type", per_static)
        if per_tier is not None:
            res.add_table("per_tier", per_tier)
        res.add_table("contributions", _contributions(df, groups, per_static, top, STATIC_KIND, normalisation, reference_graph, None, None))
        res.add_table("readout_per_body", _static_readout(c, df, target_idx, kinds))
        if keep_links:
            res.add_table("links", _links_table(df, groups, per_static, top))
        summary["kind"] = STATIC_KIND
        res.summary = summary
        res.files = {"generator": "flyverse.interp.decompose.decompose (static)"}
        return res

    # ---- dynamic
    if graded_target:
        if optic_params is None and fb is None:
            raise ValueError("a graded (optic rate-unit) target needs optic_params (or a FlyBrain) for the optic decomposition")
        long, note = _graded_dynamic(c, target_idx, arms, null_label, optic_params, rec_signs, brain._receptor_gain(p), by, window, top)
        wide = _wide(long.assign(pre_group=long.term + ":" + long.pre_group), arms, null_label, ["post_type", "pre_group"])
        res.add_table("per_type", wide)
        res.add_table("readout_per_body", _static_readout(c, df, target_idx, kinds))
        res.add_table("contributions", _contributions(df, groups, per_static, top, STATIC_KIND, normalisation, reference_graph, None, None))
        summary.update({"kind": OPTIC_KIND, "optic": note, "arms": {k: len(v) for k, v in arms.items()}, "null_arm": null_label})
        res.summary = summary
        res.replicates = _replicates(arms, null_label)
        res.files = {"generator": "flyverse.interp.decompose.decompose (graded target, optic input)"}
        return res

    tau_syn = float(p.tau_syn)
    # each arm's runs are weighted by THAT arm's effective weights: a run recorded under a hold table / off carries its
    # arm label in meta['arm'] (scripts/interp_decompose.py record); the static reference columns stay `params`'
    eff_by_arm, arm_weights = {}, {}
    for arm, runs in arms.items():
        lab = runs[0].meta.get("arm")
        if arm_weights_override is not None and arm in arm_weights_override:
            p_arm = arm_weights_override[arm]
        elif lab in (None, "", "as-given") or lab == arm_of(p):
            p_arm = None
        else:
            p_arm = arm_params(lab, p)
        if p_arm is None:
            eff_by_arm[arm] = None; arm_weights[arm] = {"arm": arm_of(p), "effective_md5": ew.md5, "source": "params"}
        else:
            ew_arm = effective_weights(c, p_arm, _receptor_for(c, p_arm))
            eff_by_arm[arm] = np.asarray(ew_arm.A[df.post_index.to_numpy(), df.pre_index.to_numpy()]).ravel().astype(np.float64)
            arm_weights[arm] = {"arm": arm_of(p_arm), "effective_md5": ew_arm.md5, "source": "meta.arm" if arm not in (arm_weights_override or {}) else "override",
                                "entries_differing_from_params": int((eff_by_arm[arm] != df.effective_mv.to_numpy()).sum())}
    rows, series_first, coverage, cells = [], None, {}, []
    for arm, runs in arms.items():
        for k, rec in enumerate(runs):
            pg, drive_rows, per_cell, series, t_ms, entry_value, cov = _run_values(df, rec, groups, n_post, window, tau_syn, keep_series=(arm == next(iter(arms)) and k == 0),
                                                                                  eff=eff_by_arm[arm])
            pg["arm"] = arm; pg["run"] = k; pg["kind"] = DYNAMIC_KIND
            rows.append(pg)
            if drive_rows:
                rows.append(pd.DataFrame(drive_rows).assign(arm=arm, run=k))
            per_cell["arm"] = arm; per_cell["run"] = k
            cells.append(per_cell)
            coverage[f"{arm}[{k}]"] = cov
            if series is not None:
                series_first = (t_ms, series, pg[["post_type", "pre_group"]].copy(), arm, entry_value)
    long = pd.concat(rows, ignore_index=True)
    wide = _wide(long, arms, null_label, ["post_type", "pre_group"])
    first_arm = next(iter(arms))
    # the static weight per group alongside (mV per post cell per volley)
    st = per_static.set_index(["post_type", "pre_group"])
    wide["weight_mv_per_volley"] = [float(st.value.get((t, g), np.nan)) for t, g in zip(wide.post_type, wide.pre_group)]
    wide["raw_count"] = [float(st.raw_count.get((t, g), np.nan)) for t, g in zip(wide.post_type, wide.pre_group)]
    wide["sign_rule"] = [str(st.sign_rule.get((t, g), "")) for t, g in zip(wide.post_type, wide.pre_group)]
    wide["unit"] = DYNAMIC_UNIT
    wide["abs_value"] = wide[f"{first_arm}_mean"].abs()
    wide = wide.sort_values(["post_type", "abs_value"], ascending=[True, False]).drop(columns="abs_value").reset_index(drop=True)
    res.add_table("per_type", wide)
    if tiers:
        groups_t = group_labels(df, tuple(by) + (("tier",) if "tier" not in by else ()))
        rows_t = []
        for arm, runs in arms.items():
            for k, rec in enumerate(runs):
                pg, _, _, _, _, _, _ = _run_values(df, rec, groups_t, n_post, window, tau_syn, keep_series=False, eff=eff_by_arm[arm])
                pg["arm"] = arm; pg["run"] = k; pg["kind"] = DYNAMIC_KIND
                rows_t.append(pg)
        res.add_table("per_tier", _wide(pd.concat(rows_t, ignore_index=True), arms, null_label, ["post_type", "pre_group"]))
    # per-frame series of the top groups (first arm, run 0)
    if series_first is not None:
        t_ms, series, keys, arm0, entry_value = series_first
        ts = []
        for t in n_post:
            m = (keys.post_type == t).to_numpy()
            order = np.argsort(-np.abs(series[:, m].mean(axis=0)))[:TIMESERIES_GROUPS]
            for j in np.flatnonzero(m)[order]:
                for ti, v in zip(t_ms, series[:, j]):
                    ts.append({"arm": arm0, "run": 0, "post_type": t, "pre_group": keys.pre_group.iloc[j], "t_ms": float(ti), "value": float(v)})
        res.add_table("timeseries", ts)
        dyn_vals = pd.Series(entry_value, index=df.index)
    else:
        dyn_vals = None
    # body-level tables
    per_first = wide.rename(columns={f"{first_arm}_mean": "value"})[["post_type", "pre_group", "value"]]
    res.add_table("contributions", _contributions(df, groups, per_first, top, DYNAMIC_KIND, normalisation, reference_graph, window_rec, dyn_vals))
    res.add_table("readout_per_body", _dynamic_readout(c, pd.concat(cells, ignore_index=True), target_idx, kinds, arms, null_label, window_rec, coverage))
    if keep_links:
        res.add_table("links", _links_table(df, groups, per_first, top, dyn_vals))
    summary.update({"kind": DYNAMIC_KIND, "tau_syn_ms": tau_syn, "arms": {k: len(v) for k, v in arms.items()}, "null_arm": null_label,
                    "arm_weights": arm_weights, "coverage": coverage,
                    "dynamic": {arm: cancelling_pair(wide.rename(columns={f"{arm}_mean": "value"}), "value") for arm in arms}})
    res.summary = summary
    res.replicates = _replicates(arms, null_label)
    res.files = {"generator": "flyverse.interp.decompose.decompose (dynamic)",
                 "recordings": [r.meta.get("file") for runs in arms.values() for r in runs]}
    return res


def _wide(long: pd.DataFrame, arms: dict, null_label, keys: list) -> pd.DataFrame:
    """Long (arm, run, keys, value) -> one row per keys with <arm>_mean / _sd / _n / _values and, against the null arm,
    <arm>_z / _welch / _U / _p / _verdict (common.compare)."""
    out = long.drop_duplicates(keys)[keys].reset_index(drop=True)
    extra = [c for c in ("rate_hz", "g_mv", "value_E", "value_I", "value_at_peak", "peak_t_ms", "output_peak_hz", "weight_mv_per_volley", "kind",
                         "n_entries_missing", "term") if c in long.columns]
    piv = {}
    for arm in arms:
        sub = long[long.arm == arm]
        g = sub.groupby(keys, sort=False)
        vals = g.value.apply(list)
        piv[arm] = vals
        stats = vals.apply(_arm_stats)
        idx = pd.MultiIndex.from_frame(out[keys]) if len(keys) > 1 else pd.Index(out[keys[0]])
        st = stats.reindex(idx)
        out[f"{arm}_mean"] = [s["mean"] if isinstance(s, dict) else np.nan for s in st]
        out[f"{arm}_sd"] = [s["sd"] if isinstance(s, dict) else np.nan for s in st]
        out[f"{arm}_n"] = [s["n"] if isinstance(s, dict) else 0 for s in st]
        out[f"{arm}_values"] = [s["values"] if isinstance(s, dict) else [] for s in st]
        for col in extra:
            if col in ("kind", "term"):
                out[col] = g[col].first().reindex(idx).to_numpy()
            else:
                out[f"{arm}_{col}"] = g[col].mean().reindex(idx).to_numpy() if col != "n_entries_missing" else g[col].max().reindex(idx).to_numpy()
    if null_label is not None:
        for arm in arms:
            if arm == null_label:
                continue
            comp = [compare(a, b) if (len(a) and len(b)) else None for a, b in zip(out[f"{arm}_values"], out[f"{null_label}_values"])]
            for f in ("z", "welch", "U", "p", "verdict", "diff"):
                out[f"{arm}_{f}"] = [c_[f] if c_ else (np.nan if f != "verdict" else "n/a") for c_ in comp]
    return out


def _replicates(arms: dict, null_label) -> dict:
    runs = []
    for arm, rs in arms.items():
        for k, r in enumerate(rs):
            ex = (r.meta.get("provenance") or {}).get("execution", {})
            runs.append({"arm": arm, "run_index": k, "seed": r.meta.get("seed"), "file": r.meta.get("file"), "device": ex.get("device"),
                         "protocol": r.meta.get("protocol"), "n_frames": int(r.n_frames)})
    n = min(len(v) for v in arms.values())
    return {"n": int(n), "unit": "runs", "runs": runs, "null": null_label}


def _contributions(df, groups, per_group, top, kind, normalisation, reference_graph, window, dyn_vals) -> pd.DataFrame:
    """Neurome 'contributions' rows (body level) for the `top` groups per post type."""
    keep = per_group.assign(a=per_group.value.abs()).sort_values(["post_type", "a"], ascending=[True, False]).groupby("post_type").head(top)
    sel = set(zip(keep.post_type, keep.pre_group))
    m = np.array([(t, g) in sel for t, g in zip(df.post_type, groups)])
    sub = df[m]
    value = sub.effective_mv.to_numpy() if dyn_vals is None else dyn_vals[m].to_numpy()
    out = pd.DataFrame({"body_pre": sub.body_pre.to_numpy(), "body_post": sub.body_post.to_numpy(), "pre_type": sub.pre_type.to_numpy(),
                        "post_type": sub.post_type.to_numpy(), "value": value, "kind": kind, "sign_rule": sub.sign_rule.to_numpy(),
                        "gain_rule": sub.gain_rule.to_numpy(), "normalisation": normalisation, "reference_graph": reference_graph,
                        "window": [window] * len(sub), "synaptic_pair_count": sub.synaptic_pair_count.to_numpy(),
                        "pre_group": groups[m].to_numpy(), "silent": sub.silent.to_numpy(), "unit": (STATIC_UNIT if dyn_vals is None else DYNAMIC_UNIT)})
    if len(out) > MAX_LINK_ROWS:                   # the JSON keeps the strongest MAX_LINK_ROWS body-level rows (the full edge table is `links`' CSV)
        out = out.reindex(np.abs(np.nan_to_num(out.value.to_numpy(np.float64))).argsort()[::-1][:MAX_LINK_ROWS]).reset_index(drop=True)
    return out


def _links_table(df, groups, per_group, top, dyn_vals=None) -> pd.DataFrame:
    keep = per_group.assign(a=per_group.value.abs()).sort_values(["post_type", "a"], ascending=[True, False]).groupby("post_type").head(top)
    sel = set(zip(keep.post_type, keep.pre_group))
    m = np.array([(t, g) in sel for t, g in zip(df.post_type, groups)])
    out = df[m].assign(pre_group=groups[m].to_numpy())
    if dyn_vals is not None:
        out = out.assign(value_mv_per_s=dyn_vals[m].to_numpy())
    out = out.reindex(out.effective_mv.abs().sort_values(ascending=False).index)
    return out.head(MAX_LINK_ROWS).drop(columns=["gain_rule"]).reset_index(drop=True)


def _static_readout(c, df, target_idx, kinds) -> pd.DataFrame:
    """readout_per_body for the static case: per target body the summed effective input (mV per volley of everything)."""
    tot = df.groupby("post_index").effective_mv.sum()
    bid = c.neurons.bodyId.to_numpy(); ty = c.neurons.type.fillna("").to_numpy()
    return pd.DataFrame({"bodyId": body_str(bid[target_idx]), "model_index": target_idx.astype(int), "type": ty[target_idx],
                         "unit_kind": kinds[target_idx], "quantity": "total_effective_input_mV_per_volley", "window_start_s": None,
                         "window_end_s": None, "stimulus_value": tot.reindex(target_idx).fillna(0.0).to_numpy(), "control_value": None,
                         "stimulus_minus_control": None, "unit": "mV", "n_trials": 0, "trial_sd": None, "control_ids": [[]] * len(target_idx)})


def _dynamic_readout(c, cells: pd.DataFrame, target_idx, kinds, arms: dict, null_label, window, coverage) -> pd.DataFrame:
    """readout_per_body: per target body and quantity (input_current_mV_per_s, output_Hz, upstream_drive_mV -- LC11 /
    LC10a and every other type get one row per quantity, never pooled), per arm: the mean over runs, the null arm's
    mean as control_value, the run sd and n, the null run ids as control_ids."""
    bid = c.neurons.bodyId.to_numpy(); ty = c.neurons.type.fillna("").to_numpy()
    rows = []
    quantities = [q for q in ("input_current_mV_per_s", "output_Hz", "upstream_drive_mV") if q in cells.columns]
    units = {"input_current_mV_per_s": "mV/s", "output_Hz": "Hz", "upstream_drive_mV": "mV"}
    null = cells[cells.arm == null_label] if null_label else None
    null_ids = [f"{null_label}[{k}]" for k in range(int(null.run.max()) + 1)] if (null is not None and len(null)) else []
    for arm in arms:
        sub = cells[cells.arm == arm]
        for q in quantities:
            g = sub.groupby("post_index")[q]
            mean, sd, n = g.mean(), g.std(ddof=1), g.count()
            ctrl = null.groupby("post_index")[q].mean() if null is not None else None
            for i in target_idx:
                if i not in mean.index or not np.isfinite(mean[i]):
                    continue
                cv = float(ctrl[i]) if (ctrl is not None and i in ctrl.index and np.isfinite(ctrl[i])) else None
                rows.append({"bodyId": str(int(bid[i])), "model_index": int(i), "type": ty[i], "unit_kind": kinds[i], "quantity": q,
                             "window_start_s": window[0] if window else None, "window_end_s": window[1] if window else None,
                             "stimulus_value": float(mean[i]), "control_value": cv,
                             "stimulus_minus_control": (float(mean[i]) - cv) if cv is not None else None, "unit": units[q],
                             "n_trials": int(n[i]), "trial_sd": float(sd[i]) if np.isfinite(sd[i]) else None, "control_ids": null_ids, "arm": arm})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- contrast
def _tier_labels(receptor, n_entries: int) -> np.ndarray:
    """Per stored entry (CSR order of c.W): 'nt_sign' where the presynaptic prior decided, 'receptor:<tier>' where the
    table did (receptor.tier >= nt_class), as common.links labels them."""
    lab = np.array(["nt_sign"] * n_entries, dtype=object)
    if receptor is not None:
        t = np.asarray(receptor.tier)
        matched = t >= cn.RECEPTOR_TIERS.index("nt_class")
        lab[matched] = np.array(["receptor:" + cn.RECEPTOR_TIERS[i] for i in t[matched]], dtype=object)
    return lab


def contrast(c, target, params_a, params_b, *, labels=("a", "b"), receptor_a=None, receptor_b=None, counts=None, by=("type",),
             optic_params=None, top=40) -> Result:
    """The structural difference between two parameter sets on the input of `target` -- what a hold table, a gain or a
    receptor arm changes, entry by entry, and through which of the two routes:

    * **sign / gain changes**: entries whose SHAPED weight (brain._shaped_weights: receptor sign x gain x cap x path
      gains, before the fan-in scale) differs -- the entries the two tables decide differently ('delta_links', all of
      them; 'delta_per_type' by (post type, pre group, transmitter, tier, sign a -> sign b); 'delta_by_transmitter';
      'delta_by_post_type');
    * **rescaled entries**: entries whose shaped weight is identical but whose effective weight moved because the
      postsynaptic cell's fan-in scale moved (a silencing changes the cell's row sum `tot`, hence
      `clip((ref / tot) ** alpha, 0.02, 1)`) -- receptor_integration.md E.3's second route ('rescaled_cells': every
      target cell whose scale moved, with tot / scale under both arms and its rescaled entry count).

    Both arms are compared at the array level over the stored entries of c.W (same sparsity, CSR order), so the whole
    brain (target = every cell) takes seconds. 'contributions' carries the sign-changed entries (value = effective b -
    a, Neurome fields) followed by the rescaled ones, capped at MAX_LINK_ROWS. The summary counts entries and raw
    synapses of each route and names the changed presynaptic types."""
    from .. import brain
    pop = population(c, target, "target")
    target_idx = pop.idx
    if len(target_idx) == 0:
        raise ValueError(f"target {target!r} selects no cell")
    if counts is None:
        counts, _ = counts_matrix(c)
    ra, rb = _receptor_for(c, params_a, receptor_a), _receptor_for(c, params_b, receptor_b)
    Wc = c.W.tocsr()
    if not Wc.has_sorted_indices:
        Wc = Wc.copy(); Wc.sort_indices()
    Wa, Wb = brain._shaped_weights(c, params_a, ra).tocsr(), brain._shaped_weights(c, params_b, rb).tocsr()
    for W in (Wa, Wb):
        W.sort_indices()
        if not (np.array_equal(W.indptr, Wc.indptr) and np.array_equal(W.indices, Wc.indices)):
            raise RuntimeError("the shaped matrix does not share c.W's sparsity pattern; the array-level contrast needs it")
    ewa, ewb = effective_weights(c, params_a, ra), effective_weights(c, params_b, rb)
    row = np.repeat(np.arange(c.n), np.diff(Wc.indptr)); col = Wc.indices
    Aa = (ewa.scale[row] * Wa.data).astype(np.float32) * np.float32(params_a.w_syn)
    Ab = (ewb.scale[row] * Wb.data).astype(np.float32) * np.float32(params_b.w_syn)
    in_t = np.zeros(c.n, bool); in_t[target_idx] = True
    m = in_t[row]
    shaped_changed = m & (Wa.data != Wb.data)
    rescaled_only = m & ~shaped_changed & (Aa != Ab)
    cnt = counts.tocsr(); cnt.sort_indices()
    cnt_data = cnt.data if cnt.nnz == Wc.nnz else np.abs(Wc.data)
    n = c.neurons
    ty = n.type.fillna("").to_numpy(); nt = n["nt"].fillna("unknown").to_numpy() if "nt" in n else np.array(["unknown"] * c.n)
    bid = n.bodyId.to_numpy(); lab = regions.labels(c); side = n.somaSide.fillna("?").to_numpy() if "somaSide" in n else np.array(["?"] * c.n)
    tier_a, tier_b = _tier_labels(ra, Wc.nnz), _tier_labels(rb, Wc.nnz)
    la, lb = labels

    def table(mask, kind):
        i = np.flatnonzero(mask)
        r_, q_ = row[i], col[i]
        df = pd.DataFrame({"pre_index": q_, "post_index": r_, "body_pre": body_str(bid[q_]), "body_post": body_str(bid[r_]),
                           "pre_type": ty[q_], "post_type": ty[r_], "pre_nt": nt[q_], "pre_module": lab[q_], "pre_side": side[q_],
                           "synaptic_pair_count": cnt_data[i].astype(np.float64), "shaped_a": Wa.data[i].astype(np.float64),
                           "shaped_b": Wb.data[i].astype(np.float64), "effective_mv_a": Aa[i].astype(np.float64), "effective_mv_b": Ab[i].astype(np.float64),
                           "sign_a": np.sign(Wa.data[i]).astype(int), "sign_b": np.sign(Wb.data[i]).astype(int), "tier_a": tier_a[i], "tier_b": tier_b[i],
                           "fanin_scale_a": ewa.scale[r_].astype(np.float64), "fanin_scale_b": ewb.scale[r_].astype(np.float64), "change": kind})
        df["delta_mv"] = df.effective_mv_b - df.effective_mv_a
        if kind == "sign_or_gain":
            df["change"] = np.where(df.sign_a != df.sign_b, "sign", "gain")
        df["tier"] = df.tier_a
        df["sign_label"] = np.where(df.sign_a > 0, "E", np.where(df.sign_a < 0, "I", "0"))
        df["pre_group"] = group_labels(df, by).to_numpy() if len(df) else np.array([], dtype=object)
        return df

    changed = table(shaped_changed, "sign_or_gain")
    rescaled = table(rescaled_only, "rescaled")
    by_type = changed.groupby(["post_type", "pre_group", "pre_nt", "tier_a", "tier_b", "sign_a", "sign_b"]).agg(
        entries=("delta_mv", "size"), synapses=("synaptic_pair_count", "sum"), n_pre=("pre_index", "nunique"), n_post=("post_index", "nunique"),
        effective_mv_a=("effective_mv_a", "sum"), effective_mv_b=("effective_mv_b", "sum"), delta_mv=("delta_mv", "sum")).reset_index() \
        .sort_values("entries", ascending=False).reset_index(drop=True)
    by_nt = changed.groupby(["pre_nt", "sign_a", "sign_b"]).agg(entries=("delta_mv", "size"), synapses=("synaptic_pair_count", "sum")).reset_index()
    by_post = changed.groupby("post_type").agg(entries=("delta_mv", "size"), synapses=("synaptic_pair_count", "sum"), n_post=("post_index", "nunique")).reset_index() \
        .sort_values("entries", ascending=False).reset_index(drop=True)
    moved = target_idx[ewa.scale[target_idx] != ewb.scale[target_idx]]
    n_resc = pd.Series(rescaled.post_index).value_counts() if len(rescaled) else pd.Series(dtype=int)
    n_sil = pd.Series(changed.post_index).value_counts() if len(changed) else pd.Series(dtype=int)
    cells = pd.DataFrame({"bodyId": body_str(bid[moved]), "model_index": moved.astype(int), "type": ty[moved],
                          "tot_a": ewa.tot[moved].astype(np.float64), "tot_b": ewb.tot[moved].astype(np.float64),
                          "fanin_scale_a": ewa.scale[moved].astype(np.float64), "fanin_scale_b": ewb.scale[moved].astype(np.float64),
                          "sign_changed_entries": n_sil.reindex(moved).fillna(0).astype(int).to_numpy(),
                          "rescaled_entries": n_resc.reindex(moved).fillna(0).astype(int).to_numpy()})
    cells["scale_delta"] = cells.fanin_scale_b - cells.fanin_scale_a
    cells = cells.reindex(cells.scale_delta.abs().sort_values(ascending=False).index).reset_index(drop=True)
    prov = common.provenance(c, params_b, optic_params, stimulus={"protocol": "static contrast", "params": {"a": _params_short(params_a), "b": _params_short(params_b)}, "control": la})
    prov["execution"]["device"] = "cpu"; prov["execution"]["backend"] = dict(prov["execution"].get("backend", {}), structural=True, simulation=False)
    prov["model_a"] = common.model_record(params_a, optic_params, body=False)
    res = Result.new("decompose", prov)
    res.add_population(pop, unit_kind=None, keep_ids=len(target_idx) <= 10000)
    link_cols = ["pre_index", "post_index", "body_pre", "body_post", "pre_type", "post_type", "pre_nt", "synaptic_pair_count", "shaped_a", "shaped_b",
                 "effective_mv_a", "effective_mv_b", "delta_mv", "sign_a", "sign_b", "tier_a", "tier_b", "fanin_scale_a", "fanin_scale_b", "change", "pre_group"]
    res.add_table("delta_links", changed[link_cols].head(MAX_LINK_ROWS))
    res.add_table("delta_per_type", by_type)
    res.add_table("delta_by_transmitter", by_nt)
    res.add_table("delta_by_post_type", by_post)
    res.add_table("rescaled_cells", cells)
    res.add_table("rescaled_by_post_type", rescaled.groupby("post_type").agg(entries=("delta_mv", "size"), n_post=("post_index", "nunique"),
                                                                           abs_delta_mv=("delta_mv", lambda s: float(np.abs(s).sum()))).reset_index()
                  .sort_values("entries", ascending=False).reset_index(drop=True) if len(rescaled) else [])
    rg = common.connectome_fingerprint(c)["md5"]
    both = pd.concat([changed, rescaled], ignore_index=True).head(MAX_LINK_ROWS)
    p = params_b
    gain_rule = (f"conn_cap {p.conn_cap}; same_type_gain {p.same_type_gain}; input_norm ({p.input_norm_ref}, alpha {p.input_norm_alpha}); "
                 f"w_syn {p.w_syn}; path_gain + type_path_gain as resolved in provenance.model")
    res.add_table("contributions", pd.DataFrame({
        "body_pre": both.body_pre.to_numpy(), "body_post": both.body_post.to_numpy(), "pre_type": both.pre_type.to_numpy(),
        "post_type": both.post_type.to_numpy(), "value": both.delta_mv.to_numpy(), "kind": STATIC_KIND,
        "sign_rule": (both.tier_a.astype(str) + "->" + both.tier_b.astype(str)).to_numpy(), "gain_rule": gain_rule,
        "normalisation": f"input_norm alpha {p.input_norm_alpha} ref {p.input_norm_ref}", "reference_graph": rg, "window": [None] * len(both),
        "synaptic_pair_count": both.synaptic_pair_count.to_numpy(), "change": both.change.to_numpy()}))
    res.summary = {"target": common.spec_repr(target), "n_target": int(len(target_idx)), "labels": [la, lb],
                   "arm_a": arm_of(params_a), "arm_b": arm_of(params_b), "entries_total": int(m.sum()),
                   "entries_sign_changed": int(len(changed)), "synapses_sign_changed": float(changed.synaptic_pair_count.sum()),
                   "entries_rescaled_only": int(len(rescaled)), "abs_delta_rescaled_mv": float(np.abs(rescaled.delta_mv).sum()) if len(rescaled) else 0.0,
                   "entries_changed": int(len(changed) + len(rescaled)),
                   "by_transmitter": {f"{r.pre_nt} {int(r.sign_a):+d}->{int(r.sign_b):+d}": {"entries": int(r.entries), "synapses": float(r.synapses)} for r in by_nt.itertuples()},
                   "by_post_type": {r.post_type: {"entries": int(r.entries), "synapses": float(r.synapses)} for r in by_post.itertuples()},
                   "pre_types_changed": sorted(changed.pre_type.unique().tolist()),
                   "target_cells_with_moved_fanin_scale": int(len(moved)),
                   "rescaled_cells": cells.head(top).to_dict("records"),
                   "shaped_md5": {la: ewa.shaped_md5, lb: ewb.shaped_md5}, "effective_md5": {la: ewa.md5, lb: ewb.md5}}
    res.files = {"generator": "flyverse.interp.decompose.contrast"}
    return res


def _params_short(p) -> dict:
    return {"receptor_model": p.receptor_model, "receptor_net_rule": p.receptor_net_rule, "receptor_table": p.receptor_table,
            "type_path_gain": p.type_path_gain, "path_gain": p.path_gain, "conn_cap": p.conn_cap, "same_type_gain": p.same_type_gain,
            "input_norm_alpha": p.input_norm_alpha, "input_norm_ref": p.input_norm_ref, "w_syn": p.w_syn}


# ---------------------------------------------------------------------------------------------- printing
def print_per_type(res: Result, top: int = 40, arms=None) -> str:
    """The printed table of a decompose Result: the top groups per target type with the arm means / z where present."""
    df = res.table("per_type")
    if df.empty:
        return common.print_table(df)
    cols = ["post_type", "pre_group"]
    if "value" in df.columns:
        cols += [c for c in ("value", "value_E", "value_I", "n_entries", "n_pre", "raw_count", "share", "sign", "sign_rule", "silent_entries") if c in df.columns]
        show = df.groupby("post_type").head(top)[cols]
    else:
        arms = arms or [c[:-5] for c in df.columns if c.endswith("_mean")]
        for a in arms:
            cols += [f"{a}_mean", f"{a}_sd"]
            if f"{a}_z" in df.columns:
                cols += [f"{a}_z", f"{a}_verdict"]
        cols += [c for c in ("weight_mv_per_volley", "raw_count", f"{arms[0]}_rate_hz", "sign_rule") if c in df.columns]
        show = df.groupby("post_type").head(top)[[c for c in cols if c in df.columns]]
    return common.print_table(show, max_rows=top * max(1, df.post_type.nunique()))
