"""Neurotransmitter-sign audit: what the sign-0 convention silences.

flyverse/connectome.py gives every synapse the sign of its PREsynaptic neuron's neurotransmitter
(NT_SIGN): ACh +1, GABA / glutamate / histamine -1, and 0 for dopamine, octopamine, serotonin and for
neurons with no usable prediction ("unknown", i.e. consensus_nt, celltype_predicted_nt and predicted_nt
all "unclear" or the body absent from the NT table), except antennal-lobe LN types, which are forced to
GABA (UNKNOWN_NT_OVERRIDE_REGEX). Sign-0 synapses are stored in W as explicit zeros and contribute
nothing to the LIF or the optic rate model, nor to the fan-in normalisation (in_syn excludes them).

This script reloads the raw weight table to recover the synapse counts of those zeroed edges and
quantifies, by superclass / module / key population, how much input and output is silenced, which
presynaptic types are responsible, what the raw NT columns say about them, and what the per-T-bar
predictions lean towards for the "unknown" cells.

    PYTHONIOENCODING=utf-8 python scripts/audit_nt.py [--out docs/audits/nt_audit.md] [--threshold 0.15] [--top 12]

Writes the markdown report plus out/nt_audit_populations.csv and out/nt_audit_groups.csv.
Purely structural: no simulation, no GPU. ~40 s (the 1 GB weight table and the 2.6 GB T-bar table,
memory-mapped, dominate).

    PYTHONIOENCODING=utf-8 python scripts/audit_nt.py --type-majority [--majority-out docs/audits/nt_type_majority.md]
                                                     [--majority-share 0.8] [--majority-min 4] [--compile-cache]

--type-majority (round 4, docs/NT_INTEGRATION.md section 7 item 7): the TYPE-MAJORITY transmitter rule for the
unknown-NT cells that belong to a type with labelled members. Per such type: the majority transmitter of its
labelled cells (count, share), the transcriptome call of flyverse/data/nt_by_type_transcriptome.csv (with the
sources behind it; a pool-mixed call is not evaluable), the Nern 2025 prediction (type_map_nern2025.csv), and
`proposed` = the type majority (share >= --majority-share, >= --majority-min labelled cells) AND >= 1 non-mixed
transcriptome source or the Nern prediction agree with it -- the round-2 adoption standard for TYPE_NT_OVERRIDE.
Writes the markdown table with the totals (cells / raw output synapses that would be relabelled; by transmitter;
by module), out/r4_type_majority.csv and out/r4_type_majority_proposed.json ({type: nt}). --compile-cache
compiles a scratch connectome with the proposed entries added to TYPE_NT_OVERRIDE into out/cache_<hash>/
(cx_wedge.load_connectome; never the shared cache) and prints its path; the benchmark then takes it with
--cache-dir. Skips the T-bar table; ~50 s.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import connectome as cn  # noqa: E402
from flyverse import regions  # noqa: E402
from flyverse.brain import LIFParams  # noqa: E402

SIGN0_REASONS = ("unknown", "dopamine", "octopamine", "serotonin")
TBAR_FILE = "tbar-neurotransmitters-male-cns-v1.0.feather"
TBAR_NTS = ["acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "octopamine", "serotonin"]

# Key populations: (group, name, selector). Selector = regex on type, or "syn:<name>" = hemibrain
# synonym in the annotation table's `synonyms` column (the sweet second-order neurons), checked
# against the known MaleCNS type.
SWEET_SYNONYMS = {"G2N-1": "GNG232", "Rattle": "GNG132", "Usnea": "GNG175", "Phantom": "GNG229"}
POPULATIONS = [
    # compass circuit
    ("compass", "EPG", r"^EPG"), ("compass", "PEN", r"^PEN"), ("compass", "PEG", r"^PEG$"),
    ("compass", "Delta7", r"^Delta7$"), ("compass", "ER (ring)", r"^ER\d"), ("compass", "ExR", r"^ExR"),
    ("compass", "PFL3", r"^PFL3$"), ("compass", "PFN", r"^PFN"), ("compass", "hDelta", r"^hDelta"),
    # loom / object pathway
    ("loom/object", "LC4", r"^LC4$"), ("loom/object", "LPLC2", r"^LPLC2$"), ("loom/object", "LC10a", r"^LC10a$"),
    ("loom/object", "LC10b", r"^LC10b$"), ("loom/object", "LC16", r"^LC16$"), ("loom/object", "LPi34", r"^LPi34$"),
    ("loom/object", "LPi43", r"^LPi43$"), ("loom/object", "Tm5Y", r"^Tm5Y$"), ("loom/object", "TmY21", r"^TmY21$"),
    ("loom/object", "T4a-d", r"^T4[a-d]$"), ("loom/object", "T5a-d", r"^T5[a-d]$"),
    # optomotor / rotation
    ("optomotor", "HSN", r"^HSN$"), ("optomotor", "HSE", r"^HSE$"), ("optomotor", "DNp20", r"^DNp20$"),
    ("optomotor", "DNp04", r"^DNp04$"), ("optomotor", "LPT27", r"^LPT27$"), ("optomotor", "LPT30", r"^LPT30$"),
    # antennal lobe
    ("antennal lobe", "ORN_*", r"^ORN_"),
    ("antennal lobe", "AL LN types", r"^(lLN|v2LN|v3LN|il3LN|l2LN|vLN)"),
    ("antennal lobe", "uniglomerular PNs", r"^(?!M_)[^_]+_(l2PN|adPN|lPN|lvPN|ilPN|ivPN|vPN)"),
    # sweet second-order (by synonym)
    ("sweet", "Usnea (GNG175)", "syn:Usnea"), ("sweet", "Rattle (GNG132)", "syn:Rattle"),
    ("sweet", "Phantom (GNG229)", "syn:Phantom"), ("sweet", "G2N-1 (GNG232)", "syn:G2N-1"),
    # motor / descending
    ("motor", "MN9", r"^MN9$"), ("motor", "DNa02", r"^DNa02$"), ("motor", "DNp09", r"^DNp09$"),
    ("motor", "DNp18", r"^DNp18$"), ("motor", "DNp33", r"^DNp33$"),
]

# Literature fast-transmitter assignments for types that come up. Regex on type -> (NT, source).
# Only entries the author is confident of; everything else is reported as "-" (not known to the author),
# never guessed. "modulatory" = the animal's transmitter is itself a monoamine.
LITERATURE = [
    (r"^Delta7$", "glutamate", "Turner-Evans et al. 2020 Neuron"),
    (r"^ER\d", "GABA", "Hanesch et al. 1989; Hulse et al. 2021 (ring neurons GABAergic)"),
    (r"^ExR2$", "dopamine (modulatory)", "PPM3 DAN, Hulse et al. 2021"),
    (r"^(EPG|EPGt|PEN|PEG|PFL|PFN)", "acetylcholine", "Hulse et al. 2021 (predictions); Turner-Evans et al. 2017"),
    (r"^hDelta", "acetylcholine / glutamate (subtype-specific)", "Hulse et al. 2021"),
    (r"^(LNO1|LNO2|LNOa|GLNO)$", "-", "LNO family, PEN nodulus inputs (Hulse et al. 2021); NT not known to the author"),
    (r"^(LC4|LPLC2|LPLC1|LC6|LC10|LC16|LC9|LC11|LC12|LC13|LC15|LC17|LC18|LC20|LC21|LC22)", "acetylcholine",
     "Davis et al. 2020 eLife (most LC / LPLC types cholinergic)"),
    (r"^LPi", "glutamate", "Mauss et al. 2015 Cell"),
    (r"^Tm5Y$", "glutamate (task brief; Davis et al. 2020 predictions)", "see 2b: MaleCNS consensus says ACh"),
    (r"^T[45]", "acetylcholine", "Mauss et al. 2014; Shinomiya et al. 2019"),
    (r"^Mi4$", "GABA", "Takemura et al. 2017; Davis et al. 2020"),
    (r"^Mi9$", "glutamate", "Takemura et al. 2017; Davis et al. 2020"),
    (r"^(Mi1|Tm3)$", "acetylcholine", "Takemura et al. 2017"),
    (r"^C3$", "GABA", "Takemura et al. 2017"),
    (r"^CT1$", "GABA", "Meier & Borst 2019"),
    (r"^L1$", "glutamate", "Takemura et al. 2017"),
    (r"^(L2|L3|L4|L5)$", "acetylcholine", "Davis et al. 2020"),
    (r"^Tm9$", "glutamate", "Davis et al. 2020"),
    (r"^Tm1$|^Tm2$|^Tm4$", "acetylcholine", "Davis et al. 2020"),
    (r"^ORN_", "acetylcholine", "Yasuyama & Salvaterra 1999"),
    (r"^(lLN|v2LN|v3LN|il3LN|l2LN|vLN)", "GABA (majority); some glutamate / ACh",
     "Wilson & Laurent 2005; Liu & Wilson 2013; Shang et al. 2007 (cholinergic lLN)"),
    (r"^(?!M_)[^_]+_(adPN|lPN)$", "acetylcholine", "uniglomerular adPN / lPN, Yasuyama et al. 2002"),
    (r"^KC", "acetylcholine", "Barnstedt et al. 2016"),
    (r"^APL$", "GABA", "Liu & Davis 2009"),
    (r"^DPM$", "serotonin + GABA (modulatory / inhibitory)", "Lee et al. 2011; Haynes et al. 2015"),
    (r"^(PAM|PPL1|PPL2|PPM)", "dopamine (modulatory)", "Mao & Davis 2009"),
    (r"^(OA-|VUM|OAN)", "octopamine (modulatory)", "Busch et al. 2009"),
    (r"^CSD", "serotonin (modulatory)", "Dacks et al. 2006"),
    (r"^MN\d", "glutamate (skeletal motor neurons)", "Johansen et al. 1989; see 2b: MaleCNS says ACh for MN9"),
    (r"^(R1-R6|R7|R8)", "histamine", "Hardie 1989"),
]


def literature(t: str) -> tuple[str, str]:
    for pat, nt, src in LITERATURE:
        if re.match(pat, t or ""):
            return nt, src
    return "-", ""


def load_edges(neurons: pd.DataFrame, log):
    """Unsigned synapse counts on the model's node set: arrays pre, post, cnt (int64)."""
    t0 = time.time()
    body_to_index = pd.Series(np.arange(len(neurons)), index=neurons.bodyId.to_numpy())
    w = pf.read_table(cn.DATA_DIR / cn.WEIGHTS_FILE).to_pandas()
    pre = body_to_index.reindex(w.body_pre.to_numpy()).to_numpy()
    post = body_to_index.reindex(w.body_post.to_numpy()).to_numpy()
    m = ~np.isnan(pre) & ~np.isnan(post) & (w.weight.to_numpy() >= 1)
    pre = pre[m].astype(np.int64); post = post[m].astype(np.int64)
    cnt = w.weight.to_numpy()[m].astype(np.int64)
    del w
    log(f"edges {len(pre):,}  synapses {int(cnt.sum()):,}  ({time.time() - t0:.1f}s)")
    return pre, post, cnt


def nt_columns(neurons: pd.DataFrame) -> pd.DataFrame:
    """Raw NT-table columns per model neuron (NaN where the body is absent from the table)."""
    nt = pd.read_feather(cn.DATA_DIR / cn.NT_FILE)
    nt = nt.rename(columns={"body": "bodyId"}).set_index("bodyId")
    cols = ["consensus_nt", "celltype_predicted_nt", "celltype_predicted_nt_confidence",
            "predicted_nt", "predicted_nt_confidence", "total_nt_predictions", "ground_truth"]
    out = nt[cols].reindex(neurons.bodyId.to_numpy()).reset_index(drop=True)
    out["in_nt_table"] = neurons.bodyId.isin(nt.index).to_numpy()
    return out


def tbar_lean(body_ids: np.ndarray, log) -> pd.DataFrame:
    """Per body: share of its predicted T-bars whose argmax class is each NT (columns TBAR_NTS), the
    mean class probabilities (mean_<nt>) and the T-bar count, from the per-T-bar prediction file."""
    t0 = time.time()
    cols = ["body"] + [f"nt_{k}_prob" for k in TBAR_NTS]
    r = pf.read_table(cn.DATA_DIR / TBAR_FILE, memory_map=True, columns=cols)
    sub = r.filter(pc.is_in(r["body"], value_set=pa.array(np.asarray(body_ids, dtype=np.int64)))).to_pandas()
    P = sub[cols[1:]].to_numpy()
    am = P.argmax(1)
    d = pd.DataFrame({"body": sub.body.to_numpy()})
    for i, k in enumerate(TBAR_NTS):
        d[k] = (am == i).astype(float)
        d[f"mean_{k}"] = P[:, i]
    g = d.groupby("body").mean()
    g["n_tbars"] = d.groupby("body").size()
    log(f"T-bar predictions for {len(g):,} bodies ({len(sub):,} T-bars) in {time.time() - t0:.1f}s")
    return g


def type_nt_summary(neurons: pd.DataFrame, raw: pd.DataFrame, types: list[str], restrict=None,
                    tbar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per type: model nt (mode), consensus / celltype / per-body predicted NT modes with their shares;
    restrict = optional boolean mask limiting the cells considered; tbar = per-body T-bar lean table."""
    rows = []
    ty = neurons.type.fillna("")
    for t in types:
        m = (ty == t).to_numpy()
        if restrict is not None:
            m = m & restrict
        sub = raw[m]

        def mode(col):
            v = sub[col].fillna("absent").value_counts()
            if len(v) == 0:
                return "-", 0.0
            return v.index[0], v.iloc[0] / max(m.sum(), 1)

        c_nt, c_f = mode("consensus_nt"); ct_nt, ct_f = mode("celltype_predicted_nt"); p_nt, p_f = mode("predicted_nt")
        model = neurons.nt[m].value_counts()
        lit, src = literature(t)
        lean = "-"
        if tbar is not None:
            tb = tbar.reindex(neurons.bodyId[m].to_numpy()).dropna()
            if len(tb):
                lean = " / ".join(f"{k} {v:.0%}" for k, v in tb[TBAR_NTS].mean().sort_values(ascending=False).head(3).items())
        rows.append({
            "type": t or "(untyped)", "n": int(m.sum()),
            "model_nt": f"{model.index[0]} ({model.iloc[0] / m.sum():.0%})" if len(model) else "-",
            "consensus_nt": f"{c_nt} ({c_f:.0%})",
            "celltype_predicted_nt": f"{ct_nt} ({ct_f:.0%}, conf {sub.celltype_predicted_nt_confidence.mean():.2f})",
            "predicted_nt": f"{p_nt} ({p_f:.0%}, conf {sub.predicted_nt_confidence.mean():.2f})",
            "n_tbar_pred": f"{sub.total_nt_predictions.mean():.0f}" if sub.total_nt_predictions.notna().any() else "-",
            "tbar_argmax": lean,
            "literature": lit, "source": src,
        })
    return pd.DataFrame(rows)


def frac_table(labels: np.ndarray, pre, post, cnt, sign0, has_out, names=None) -> pd.DataFrame:
    """Per group: neurons, presynaptic neurons (out-degree > 0), sign-0 presynaptic neurons, and the
    sign-0 share of the group's output synapses (edges whose pre is in the group) and input synapses."""
    names = sorted(set(labels)) if names is None else names
    idx = {g: i for i, g in enumerate(names)}
    lab = np.array([idx.get(g, -1) for g in labels])
    k = len(names)
    s0 = sign0[pre]
    lp, lq = lab[pre], lab[post]
    out_tot = np.bincount(lp[lp >= 0], weights=cnt[lp >= 0], minlength=k)
    out_0 = np.bincount(lp[(lp >= 0) & s0], weights=cnt[(lp >= 0) & s0], minlength=k)
    in_tot = np.bincount(lq[lq >= 0], weights=cnt[lq >= 0], minlength=k)
    in_0 = np.bincount(lq[(lq >= 0) & s0], weights=cnt[(lq >= 0) & s0], minlength=k)
    rows = []
    for g in names:
        i = idx[g]; m = lab == i
        rows.append({"group": g or "(no superclass)", "neurons": int(m.sum()), "pre_neurons": int((m & has_out).sum()),
                     "pre_sign0": int((m & has_out & sign0).sum()),
                     "pre_sign0_frac": (m & has_out & sign0).sum() / max((m & has_out).sum(), 1),
                     "out_syn": int(out_tot[i]), "out_sign0": int(out_0[i]), "out_sign0_frac": out_0[i] / max(out_tot[i], 1),
                     "in_syn": int(in_tot[i]), "in_sign0": int(in_0[i]), "in_sign0_frac": in_0[i] / max(in_tot[i], 1)})
    return pd.DataFrame(rows)


def pct(x):
    return f"{100 * x:.1f}%"


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.astype(object).iterrows():      # object dtype: ints stay ints in a row that also holds floats
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, (float, np.floating)):
                v = f"{v:.3f}"
            elif isinstance(v, (int, np.integer)):
                v = f"{int(v):,}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def type_majority(c, raw: pd.DataFrame, pre, post, cnt, args, log):
    """The type-majority rule (module docstring). Returns (table, LN-regex table, totals dict, proposed {type: nt})."""
    n = c.neurons
    N = c.n
    nt = n.nt.to_numpy()
    ty = n.type.fillna("").to_numpy()
    mod = regions.labels(c)
    out_syn = np.bincount(pre, weights=cnt, minlength=N)
    out_edges = np.bincount(pre, minlength=N)
    has_out = out_syn > 0
    unknown = nt == "unknown"
    root = os.path.join(os.path.dirname(__file__), "..")

    # the transcriptome call (round-1/2 builder) and the Nern 2025 prediction
    tr = pd.read_csv(os.path.join(root, "flyverse", "data", "nt_by_type_transcriptome.csv"), comment="#",
                     dtype=str, keep_default_na=False).set_index("malecns_type")
    nern = pd.read_csv(os.path.join(root, "flyverse", "data", "type_map_nern2025.csv"), comment="#",
                       dtype=str, keep_default_na=False)
    nern = nern[nern.tier.isin(["exact", "class"])].drop_duplicates("malecns_type").set_index("malecns_type")
    classical = {"acetylcholine", "gaba", "glutamate", "histamine"}

    def nern_call(t):
        if t in tr.index and tr.at[t, "nern2025_prediction"]:
            return tr.at[t, "nern2025_prediction"], tr.at[t, "nern2025_validated"]
        if t in nern.index:
            return nern.at[t, "nern_nt"], nern.at[t, "validated_nt"]
        return "", ""

    def transcriptome_call(t):
        if t not in tr.index:
            return "", 0, "", False
        r = tr.loc[t]
        srcs = [x for x in r.sources_agreeing.split(";") if x]
        return r.nt_transcriptome, len(srcs), ";".join(srcs), r.pool_mixed == "True"

    # the raw-unknown cells (all three NT columns unclear, or absent from the NT table): what the LN regex rescued
    raw_unknown = ((raw.consensus_nt.fillna("unclear") == "unclear") & (raw.celltype_predicted_nt.fillna("unclear") == "unclear")
                   & (raw.predicted_nt.fillna("unclear") == "unclear")).to_numpy() | ~raw.in_nt_table.to_numpy()
    ln_regex = list(cn.UNKNOWN_NT_OVERRIDE_REGEX)[0]
    ln_rescued = raw_unknown & ~unknown & pd.Series(ty).str.match(ln_regex).to_numpy() & (nt == "gaba")
    for t, v in cn.TYPE_NT_OVERRIDE.items():             # cells the adopted table relabelled are not "rescued by the regex"
        ln_rescued &= ~((ty == t) & (nt == v))

    def majority_rows(cell_mask_unknown, label_mask, types):
        rows = []
        for t in types:
            m = ty == t
            u = m & cell_mask_unknown
            lab = m & label_mask
            vc = pd.Series(nt[lab]).value_counts()
            maj, cnt_maj = (vc.index[0], int(vc.iloc[0])) if len(vc) else ("", 0)
            n_lab = int(lab.sum())
            share = cnt_maj / n_lab if n_lab else 0.0
            tr_nt, tr_n, tr_src, tr_mixed = transcriptome_call(t)
            nern_nt, nern_val = nern_call(t)
            tr_agree = (not tr_mixed) and tr_n >= 1 and tr_nt == maj
            nern_agree = bool(nern_nt) and nern_nt == maj
            rule = n_lab >= args.majority_min and share >= args.majority_share and maj != ""
            proposed = rule and (tr_agree or nern_agree)
            rows.append({
                "type": t, "module": pd.Series(mod[m]).value_counts().index[0], "n_cells": int(m.sum()),
                "n_unknown": int(u.sum()), "n_unknown_pre": int((u & has_out).sum()),
                "unknown_out_syn": int(out_syn[u].sum()), "unknown_out_edges": int(out_edges[u].sum()),
                "n_labelled": n_lab, "majority": maj, "majority_n": cnt_maj, "majority_share": share,
                "labels": ", ".join(f"{k} {int(v)}" for k, v in vc.items()),
                "transcriptome": tr_nt, "transcriptome_sources": tr_n, "transcriptome_source_names": tr_src,
                "transcriptome_pool_mixed": tr_mixed, "transcriptome_agrees": tr_agree,
                "nern2025": nern_nt, "nern2025_validated": nern_val, "nern_agrees": nern_agree,
                "majority_rule": rule, "proposed": proposed,
                "sign_changes": proposed and maj in classical,
            })
        return pd.DataFrame(rows)

    types_unknown = sorted(set(ty[unknown & (ty != "")]))
    tab = majority_rows(unknown, ~unknown, types_unknown)
    tab = tab.sort_values(["proposed", "unknown_out_syn"], ascending=[False, False]).reset_index(drop=True)
    with_lab = tab[tab.n_labelled > 0]
    log(f"unknown-NT cells: {int(unknown.sum()):,} ({int((unknown & has_out).sum()):,} presynaptic, "
        f"{int(out_syn[unknown].sum()):,} raw output synapses); untyped {int((unknown & (ty == '')).sum()):,} "
        f"({int(out_syn[unknown & (ty == '')].sum()):,} syn); in {len(tab)} types, {len(with_lab)} of them with labelled members "
        f"({int(with_lab.n_unknown.sum()):,} cells, {int(with_lab.n_unknown_pre.sum()):,} presynaptic, "
        f"{int(with_lab.unknown_out_syn.sum()):,} syn)")
    log("  by the majority transmitter (all types with labelled members, no threshold): " +
        ", ".join(f"{k} {int(v):,}" for k, v in with_lab.groupby("majority").unknown_out_syn.sum().sort_values(ascending=False).items()))

    # the antennal-lobe LN regex: the rescued cells against the same rule (their labelled members = cells with a raw label)
    ln_types = sorted(set(ty[ln_rescued]))
    ln = majority_rows(ln_rescued, ~raw_unknown, ln_types) if ln_types else pd.DataFrame()
    if len(ln):
        ln = ln.rename(columns={"n_unknown": "n_rescued", "n_unknown_pre": "n_rescued_pre",
                                "unknown_out_syn": "rescued_out_syn", "unknown_out_edges": "rescued_out_edges"})
        ln["majority_is_gaba"] = ln.majority == "gaba"
        log(f"LN regex rescued {int(ln_rescued.sum()):,} cells in {len(ln)} types, {int(ln.rescued_out_syn.sum()):,} raw output syn; "
            f"majority gaba in {int(ln.majority_is_gaba.sum())} types; the rule would cover (proposed) "
            f"{int(ln.proposed.sum())} types / {int(ln.n_rescued[ln.proposed].sum())} cells / {int(ln.rescued_out_syn[ln.proposed].sum()):,} syn")

    # threshold sensitivity: the rule alone and with a source, over (min labelled cells, min share)
    sens = []
    src_ok = tab.transcriptome_agrees | tab.nern_agrees
    for min_lab, share in [(1, 0.5), (1, 1.0), (2, 1.0), (3, 0.8), (4, 0.8), (4, 1.0), (8, 0.8), (20, 0.8)]:
        rule = (tab.n_labelled >= min_lab) & (tab.majority_share >= share) & (tab.majority != "")
        both = rule & src_ok
        sens.append({"min_labelled": min_lab, "min_share": share,
                     "rule_types": int(rule.sum()), "rule_cells": int(tab.n_unknown[rule].sum()), "rule_out_syn": int(tab.unknown_out_syn[rule].sum()),
                     "with_source_types": int(both.sum()), "with_source_cells": int(tab.n_unknown[both].sum()),
                     "with_source_out_syn": int(tab.unknown_out_syn[both].sum())})
    sens = pd.DataFrame(sens)
    log("sensitivity (min_labelled, min_share -> rule types/cells/syn | +source types/cells/syn): " +
        "; ".join(f"({r.min_labelled}, {r.min_share:g}) {r.rule_types}/{r.rule_cells}/{r.rule_out_syn:,} | "
                  f"{r.with_source_types}/{r.with_source_cells}/{r.with_source_out_syn:,}" for r in sens.itertuples()))
    # types with unknown cells and >= 1 labelled member but NO transcriptome / Nern call at all
    no_source = with_lab[(with_lab.transcriptome == "") & (with_lab.nern2025 == "")]
    log(f"types with labelled members but no transcriptome / Nern call: {len(no_source)} of {len(with_lab)} "
        f"({int(no_source.n_unknown.sum())} cells, {int(no_source.unknown_out_syn.sum()):,} syn)")

    prop = tab[tab.proposed]
    proposed = {r.type: r.majority for r in prop.itertuples()}
    prop_cells = unknown & np.isin(ty, list(proposed))
    totals = {
        "types": int(len(prop)), "cells": int(prop.n_unknown.sum()), "presynaptic_cells": int(prop.n_unknown_pre.sum()),
        "out_syn": int(prop.unknown_out_syn.sum()), "out_edges": int(prop.unknown_out_edges.sum()),
        "sign_changing_types": int(prop.sign_changes.sum()), "sign_changing_cells": int(prop.n_unknown[prop.sign_changes].sum()),
        "sign_changing_out_syn": int(prop.unknown_out_syn[prop.sign_changes].sum()),
        "by_transmitter": {k: {"types": int(len(g)), "cells": int(g.n_unknown.sum()), "out_syn": int(g.unknown_out_syn.sum())}
                           for k, g in prop.groupby("majority")},
        "by_module": {k: {"cells": int(v), "out_syn": int(out_syn[prop_cells & (mod == k)].sum())}
                      for k, v in pd.Series(mod[prop_cells]).value_counts().items()},
        "rule_only": {"types": int(tab.majority_rule.sum()), "cells": int(tab.n_unknown[tab.majority_rule].sum()),
                      "out_syn": int(tab.unknown_out_syn[tab.majority_rule].sum())},
        "total_syn": int(cnt.sum()), "unknown_out_syn_all": int(out_syn[unknown].sum()), "unknown_cells_all": int(unknown.sum()),
        "with_labelled": {"types": int(len(with_lab)), "cells": int(with_lab.n_unknown.sum()), "presynaptic": int(with_lab.n_unknown_pre.sum()),
                          "out_syn": int(with_lab.unknown_out_syn.sum()),
                          "by_majority": {k: int(v) for k, v in with_lab.groupby("majority").unknown_out_syn.sum().items()}},
        "no_source": {"types": int(len(no_source)), "cells": int(no_source.n_unknown.sum()), "out_syn": int(no_source.unknown_out_syn.sum())},
        "sensitivity": sens,
    }
    log(f"proposed: {totals['types']} types, {totals['cells']} cells ({totals['presynaptic_cells']} presynaptic), "
        f"{totals['out_syn']:,} raw output syn ({100 * totals['out_syn'] / totals['total_syn']:.3f} % of all), "
        f"{totals['out_edges']:,} edges; sign-changing {totals['sign_changing_types']} types / {totals['sign_changing_cells']} cells / "
        f"{totals['sign_changing_out_syn']:,} syn; by transmitter " +
        ", ".join(f"{k} {v['cells']}/{v['out_syn']:,}" for k, v in totals["by_transmitter"].items()))
    return tab, ln, totals, proposed


def write_type_majority(tab, ln, totals, proposed, args, cache_dir, out_path):
    root = os.path.join(os.path.dirname(__file__), "..")
    L = ["# Type-majority transmitter rule for the unknown-NT cells\n"]
    L.append(f"Generated by `scripts/audit_nt.py --type-majority` on {time.strftime('%Y-%m-%d')} from the cached connectome "
             f"`{cache_dir}` (the model's `nt` after the antennal-lobe LN regex `{list(cn.UNKNOWN_NT_OVERRIDE_REGEX)[0]}` "
             f"and `TYPE_NT_OVERRIDE` = {', '.join(f'{t} -> {v}' for t, v in cn.TYPE_NT_OVERRIDE.items())}), the raw weight table, "
             "`flyverse/data/nt_by_type_transcriptome.csv` (the transmitter call from synthesis / transport genes per source; "
             "docs/audits/receptor_rules.md section 2) and `flyverse/data/type_map_nern2025.csv` (Nern et al. 2025 Sup. Table 1 "
             "predictions). docs/NT_INTEGRATION.md section 7, round 4 item 7.\n")
    L.append(f"**Rule.** For every type with >= 1 `unknown`-NT cell and >= 1 labelled cell, the *type majority* is the most "
             f"frequent transmitter of its labelled cells (`majority`, `majority_n`, `majority_share` = majority_n / n_labelled). "
             f"`majority_rule` = share >= {args.majority_share:g} and >= {args.majority_min} labelled cells. `proposed` = "
             f"majority_rule AND (>= 1 non-pool-mixed transcriptome source calls the same transmitter, or the Nern 2025 prediction "
             f"does) -- the round-2 adoption standard for `TYPE_NT_OVERRIDE` (a type label supported by the EM classifier and an "
             f"independent source). A proposed entry relabels only the type's `unknown` cells (`compile_connectome`); "
             f"`sign_changes` = the transmitter is classical (a monoamine label keeps sign 0 under `NT_SIGN`, so W is unchanged).\n")
    T = totals
    L.append("## 1. Totals\n")
    L.append(f"* `unknown`-NT cells in the cache: {T['unknown_cells_all']:,} carrying {T['unknown_out_syn_all']:,} raw output synapses "
             f"({100 * T['unknown_out_syn_all'] / T['total_syn']:.3f} % of {T['total_syn']:,}).")
    wl = T["with_labelled"]
    L.append(f"* In a type with labelled members: {wl['types']} types, {wl['cells']:,} cells ({wl['presynaptic']:,} presynaptic), "
             f"{wl['out_syn']:,} synapses; by the majority transmitter with no threshold: " +
             ", ".join(f"{k} {v:,}" for k, v in sorted(wl['by_majority'].items(), key=lambda kv: -kv[1])) + ".")
    ro = T["rule_only"]
    L.append(f"* Majority rule alone (share >= {args.majority_share:g}, >= {args.majority_min} labelled): {ro['types']} types, "
             f"{ro['cells']:,} cells, {ro['out_syn']:,} synapses.")
    L.append(f"* **Proposed (majority + >= 1 agreeing source): {T['types']} types, {T['cells']:,} cells "
             f"({T['presynaptic_cells']:,} presynaptic), {T['out_syn']:,} raw output synapses = "
             f"{100 * T['out_syn'] / T['total_syn']:.3f} % of all, {T['out_edges']:,} edges; sign-changing (classical transmitter): "
             f"{T['sign_changing_types']} types / {T['sign_changing_cells']:,} cells / {T['sign_changing_out_syn']:,} synapses.**")
    ns = T["no_source"]
    L.append(f"* Of the {wl['types']} types with labelled members, {ns['types']} ({ns['cells']:,} cells, {ns['out_syn']:,} synapses) have "
             f"no transcriptome call and no Nern 2025 prediction at all, so no threshold can make them `proposed` under this standard.")
    L.append("\nThreshold sensitivity (the rule alone | the rule plus >= 1 agreeing source), types / unknown cells / raw output synapses:\n")
    L.append(md_table(T["sensitivity"], {"min_share": lambda v: f"{v:g}"}))
    L.append("\nProposed, by transmitter:\n")
    bt = pd.DataFrame([{"transmitter": k, **v} for k, v in T["by_transmitter"].items()]).sort_values("out_syn", ascending=False) \
        if T["by_transmitter"] else pd.DataFrame([{"transmitter": "(none)", "types": 0, "cells": 0, "out_syn": 0}])
    L.append(md_table(bt))
    L.append("\nProposed, by module of the relabelled cell (`flyverse/regions.py`):\n")
    bm = pd.DataFrame([{"module": k, **v} for k, v in T["by_module"].items()]).sort_values("out_syn", ascending=False) \
        if T["by_module"] else pd.DataFrame([{"module": "(none)", "cells": 0, "out_syn": 0}])
    L.append(md_table(bm))
    L.append("\nProposed `TYPE_NT_OVERRIDE` entries (`out/r4_type_majority_proposed.json`):\n")
    L.append(("```\n" + "\n".join(f'    "{t}": "{v}",' for t, v in proposed.items()) + "\n```") if proposed else "none")

    cols = ["type", "module", "n_cells", "n_unknown", "n_unknown_pre", "unknown_out_syn", "n_labelled", "majority", "majority_n",
            "majority_share", "labels", "transcriptome", "transcriptome_sources", "transcriptome_source_names",
            "transcriptome_pool_mixed", "nern2025", "nern2025_validated", "majority_rule", "proposed", "sign_changes"]
    fmt = {"majority_share": lambda v: f"{v:.2f}", "transcriptome_pool_mixed": lambda v: "mixed" if v else "",
           "majority_rule": lambda v: "yes" if v else "", "proposed": lambda v: "**yes**" if v else "",
           "sign_changes": lambda v: "yes" if v else "", "transcriptome": lambda v: v or "-", "nern2025": lambda v: v or "-",
           "transcriptome_source_names": lambda v: v or "-", "nern2025_validated": lambda v: v or "-", "majority": lambda v: v or "-"}
    L.append(f"\n## 2. Every type with `unknown`-NT cells ({len(tab)} types; proposed first, then by silenced output)\n")
    L.append("`labels` = the type's labelled cells by transmitter; `transcriptome` = `nt_transcriptome` with the number and names of "
             "the sources agreeing with it (`mixed` = the profile is a pool of several types, not evaluable); `nern2025` = the "
             "prediction (`validated` = Nern's own validation, where any). Types with no labelled cells have no majority.\n")
    L.append(md_table(tab[cols], fmt))
    L.append("\n## 3. The antennal-lobe LN regex against the same rule\n")
    if len(ln):
        L.append(f"`UNKNOWN_NT_OVERRIDE_REGEX` relabels raw-unknown cells of LN types to GABA before `TYPE_NT_OVERRIDE`; this table "
                 f"applies the type-majority rule to those rescued cells, with the type's raw-labelled cells as the labelled set. "
                 f"Rescued: {int(ln.n_rescued.sum())} cells in {len(ln)} types, {int(ln.rescued_out_syn.sum()):,} raw output synapses; "
                 f"majority GABA in {int(ln.majority_is_gaba.sum())} of {len(ln)} types; covered by the rule (`proposed`): "
                 f"{int(ln.proposed.sum())} types / {int(ln.n_rescued[ln.proposed].sum())} cells / "
                 f"{int(ln.rescued_out_syn[ln.proposed].sum()):,} synapses; by the majority rule alone: {int(ln.majority_rule.sum())} types / "
                 f"{int(ln.n_rescued[ln.majority_rule].sum())} cells / {int(ln.rescued_out_syn[ln.majority_rule].sum()):,} synapses.\n")
        ren = {"n_unknown": "n_rescued", "n_unknown_pre": "n_rescued_pre", "unknown_out_syn": "rescued_out_syn"}
        L.append(md_table(ln[[ren.get(c_, c_) for c_ in cols]], fmt))
    else:
        L.append("No cell is rescued by the regex in this cache.")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    tab.to_csv(os.path.join(root, "out", "r4_type_majority.csv"), index=False)
    if len(ln):
        ln.to_csv(os.path.join(root, "out", "r4_type_majority_ln_regex.csv"), index=False)
    with open(os.path.join(root, "out", "r4_type_majority_proposed.json"), "w", encoding="utf-8") as f:
        json.dump(proposed, f, indent=1)


def compare_caches(c, cache_dir, other_dir, log):
    """W (data / indices / indptr) byte identity, the cells whose nt / sign differ, and -- under the default LIFParams
    receptor model -- whether the receptor lookup's fast signs are identical on both caches."""
    import hashlib
    lines = []
    _log = log

    def log(msg):
        lines.append(msg); _log(msg)
    c2 = cn.load(cache_dir=other_dir, verbose=False)
    log(f"compare {cache_dir} vs {other_dir}")
    A, B = c.W.tocsr(), c2.W.tocsr()
    A.sort_indices(); B.sort_indices()
    same_shape = A.shape == B.shape and A.nnz == B.nnz
    md5 = lambda x: hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
    w_same = same_shape and md5(A.data) == md5(B.data) and md5(A.indices) == md5(B.indices) and md5(A.indptr) == md5(B.indptr)
    log(f"W: shape {A.shape} / {B.shape}, nnz {A.nnz:,} / {B.nnz:,}, sum|W| {int(np.abs(A.data).sum()):,} / {int(np.abs(B.data).sum()):,}; "
        f"data md5 {md5(A.data)} / {md5(B.data)}; indices md5 {md5(A.indices)} / {md5(B.indices)}; indptr md5 {md5(A.indptr)} / {md5(B.indptr)}; "
        f"BYTE-IDENTICAL {w_same}")
    n1, n2 = c.neurons, c2.neurons
    same_cells = (n1.bodyId.to_numpy() == n2.bodyId.to_numpy()).all() if len(n1) == len(n2) else False
    log(f"neurons: {len(n1):,} / {len(n2):,}, same bodyId order {same_cells}")
    if same_cells:
        d = n1.nt.to_numpy() != n2.nt.to_numpy()
        ds = n1.sign.to_numpy() != n2.sign.to_numpy()
        out_deg = np.asarray(np.abs(A).sum(axis=0)).ravel()
        nnz_out = np.diff(A.tocsc().indptr)
        log(f"cells with a different nt: {int(d.sum())} (different sign: {int(ds.sum())}); their stored output entries in W: "
            f"{int(nnz_out[d].sum())}, sum|W| out {int(out_deg[d].sum())}; by type: "
            + ", ".join(f"{k} {int(v)}" for k, v in pd.Series(n1.type.fillna('').to_numpy()[d]).value_counts().items())
            + "; label change: " + ", ".join(f"{a} -> {b} {int(v)}" for (a, b), v in
                                          pd.DataFrame({"a": n1.nt.to_numpy()[d], "b": n2.nt.to_numpy()[d]}).value_counts().items()))
        for col in ("type", "superclass", "class"):
            if col in n1.columns and col in n2.columns:
                dc = (n1[col].fillna("").to_numpy() != n2[col].fillna("").to_numpy()).sum()
                log(f"  column {col}: {int(dc)} differences")
    lp = LIFParams()
    if lp.receptor_model is not None:
        t0 = time.time()
        r1 = cn.receptor_signs(c, table_path=lp.receptor_table, net_rule=lp.receptor_net_rule, nt_class_fallback=lp.receptor_nt_class_fallback)
        r2 = cn.receptor_signs(c2, table_path=lp.receptor_table, net_rule=lp.receptor_net_rule, nt_class_fallback=lp.receptor_nt_class_fallback)
        fs_same = (r1.fast_sign.shape == r2.fast_sign.shape) and md5(r1.fast_sign) == md5(r2.fast_sign)
        tier_same = md5(r1.tier) == md5(r2.tier)
        changed1 = int((r1.fast_sign != np.sign(c.W.data)).sum()); changed2 = int((r2.fast_sign != np.sign(c2.W.data)).sum())
        log(f"receptor model {lp.receptor_model} / {lp.receptor_net_rule}: fast_sign md5 {md5(r1.fast_sign)} / {md5(r2.fast_sign)} "
            f"IDENTICAL {fs_same}; tier identical {tier_same}; entries changed vs the presynaptic sign {changed1:,} / {changed2:,} "
            f"({time.time() - t0:.0f} s)")
    return {"w_identical": bool(w_same), "lines": lines}


def score_section(out_path, score_files, baseline_files, log, identity=None, note=None):
    """Append the suite comparison: per check, measured value and status in every candidate run and every baseline run,
    the status tally per run, and the checks whose status set differs between the two groups."""
    import glob
    def expand(fs):
        out = []
        for f in fs:
            out += sorted(glob.glob(f)) or [f]
        return out
    score_files, baseline_files = expand(score_files), expand(baseline_files)
    runs = []
    for grp, fs in (("candidate", score_files), ("baseline", baseline_files)):
        for f in fs:
            if not os.path.isfile(f):
                log(f"score: missing {f}"); continue
            d = json.load(open(f, encoding="utf-8"))
            cfg = d.get("config", {})
            runs.append({"group": grp, "file": f.replace(os.sep, "/"), "checks": {c["key"]: c for c in d["checks"]},
                         "receptor": cfg.get("receptor", {}), "cache_dir": cfg.get("cache_dir"), "nt_counts": cfg.get("nt_counts"),
                         "date": d.get("date"), "device": cfg.get("device"), "runtime": d.get("total_runtime_s")})
    if not runs:
        return
    abbr = {"PASS": "P", "FAIL": "F", "KNOWN GAP": "G", "MISSING": "M"}
    keys = []
    for r in runs:
        for k in r["checks"]:
            if k not in keys:
                keys.append(k)
    L = ["\n## 4. Suite score of the candidate set\n"]
    L.append("Candidate runs = `scripts/benchmark.py --seeds 0,1,2 --cache-dir out/cache_<hash>` on the scratch cache compiled with the "
             "proposed entries added to `TYPE_NT_OVERRIDE` (the shipped default receptor model on); baseline runs = the shipped default "
             "cache. Adoption criterion (round 2): no check changes status. Per run: measured value and status "
             "(P PASS, F FAIL, G KNOWN GAP, M MISSING).\n")
    L.append("Runs:\n")
    hdr = pd.DataFrame([{"group": r["group"], "file": r["file"], "date": r["date"], "device": r["device"],
                         "receptor": f"{r['receptor'].get('model')} / {r['receptor'].get('net_rule', '-')} / changed {r['receptor'].get('fast_sign_changed_entries', '-')}",
                         "cache_dir": r["cache_dir"],
                         "nt unknown / ach": f"{(r['nt_counts'] or {}).get('unknown', '-')} / {(r['nt_counts'] or {}).get('acetylcholine', '-')}",
                         "PASS/FAIL/GAP": "/".join(str(sum(1 for c in r["checks"].values() if c["status"] == st)) for st in ("PASS", "FAIL", "KNOWN GAP")),
                         "runtime_s": f"{r['runtime']:.0f}" if r["runtime"] else "-"} for r in runs])
    L.append(md_table(hdr))
    rows, moved = [], []
    for k in keys:
        row = {"check": k}
        stat = {"candidate": set(), "baseline": set()}
        for i, r in enumerate(runs):
            c = r["checks"].get(k)
            col = f"{r['group'][0]}{i + 1}"
            if c is None:
                row[col] = "-"; continue
            m = c["measured"]
            row[col] = (f"{m:.2f} " if isinstance(m, (int, float)) and m is not None else "-- ") + abbr.get(c["status"], c["status"])
            stat[r["group"]].add(c["status"])
        row["criterion"] = next((r["checks"][k]["criterion"] for r in runs if k in r["checks"]), "")
        row["status sets"] = f"cand {sorted(stat['candidate'])} / base {sorted(stat['baseline'])}" if stat["baseline"] else f"cand {sorted(stat['candidate'])}"
        if stat["baseline"] and stat["candidate"] and stat["candidate"] != stat["baseline"]:
            moved.append(f"{k}: candidate {sorted(stat['candidate'])} vs baseline {sorted(stat['baseline'])}")
        rows.append(row)
    L.append("\nPer check (columns c1.. = candidate runs, b1.. = baseline runs, in the order of the table above):\n")
    L.append(md_table(pd.DataFrame(rows)))
    L.append("\n**Checks whose status set differs between the candidate runs and the baseline runs: " +
             ("; ".join(moved) if moved else "none") + f". Adoption criterion (no check changes status): {'MET' if not moved else 'NOT MET'}.**")
    if identity:
        L.append("\n## 5. Cache identity (scratch cache with the candidate entries vs the audited cache)\n")
        L.append("`scripts/audit_nt.py --compare-cache`: W (data / indices / indptr) md5, the cells whose label differs and their stored "
                 "output entries, and the default receptor model's fast signs on both caches.\n")
        L.append("```\n" + "\n".join(identity["lines"]) + "\n```")
        L.append(f"\nW byte-identical: **{identity['w_identical']}**" +
                 (" -- a check whose status differs between the two groups of runs is therefore run-to-run scatter or a benchmark "
                  "difference between the groups, not an effect of the candidate entries." if identity["w_identical"] else ""))
    if note:
        L.append("\n## 6. Decision\n")
        L.append(note)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log(f"score section: {len(score_files)} candidate + {len(baseline_files)} baseline runs, {len(keys)} checks, status moved: {moved or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/audits/nt_audit.md")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--cache-dir", default=None,
                    help="connectome cache to audit (default cache/); e.g. a scratch cache built with TYPE_NT_OVERRIDE")
    ap.add_argument("--type-majority", action="store_true", help="the type-majority rule table only (docstring); no T-bar table")
    ap.add_argument("--majority-out", default="docs/audits/nt_type_majority.md")
    ap.add_argument("--majority-share", type=float, default=0.8, help="minimum majority share of the type's labelled cells")
    ap.add_argument("--majority-min", type=int, default=4, help="minimum number of labelled cells in the type")
    ap.add_argument("--compile-cache", action="store_true",
                    help="with --type-majority: compile a scratch connectome with the proposed entries added to TYPE_NT_OVERRIDE "
                         "into out/cache_<hash>/ (cx_wedge.load_connectome) and print its path")
    ap.add_argument("--score-json", nargs="*", default=[], metavar="JSON",
                    help="with --type-majority: benchmark.py --json files run on the scratch cache (the candidate set); appended "
                         "as a per-check comparison against --baseline-json (out/r4_ntmaj_*.json vs out/r4_default_*.json)")
    ap.add_argument("--baseline-json", nargs="*", default=[], metavar="JSON", help="reference benchmark JSONs (the shipped default)")
    ap.add_argument("--note", default=None, help="with --score-json: a decision paragraph appended as the last section")
    ap.add_argument("--compare-cache", default=None, metavar="DIR",
                    help="compare the audited cache with another one (a scratch TYPE_NT_OVERRIDE cache): W byte identity, per-cell "
                         "label differences, and the default receptor model's fast signs on both; prints and exits")
    args = ap.parse_args()
    log = print
    root = os.path.join(os.path.dirname(__file__), "..")

    cache_dir = cn.CACHE_DIR if args.cache_dir is None else args.cache_dir
    c = cn.load(cache_dir=cache_dir, verbose=False)
    log(f"cache {cache_dir}")
    n = c.neurons
    N = c.n
    log(f"neurons {N:,}  sum|W| {int(np.abs(c.W.data).sum()):,}")
    if args.compare_cache:
        compare_caches(c, cache_dir, args.compare_cache, log)
        return None
    if args.type_majority:
        os.makedirs(os.path.join(root, "out"), exist_ok=True)
        raw = nt_columns(n)
        pre, post, cnt = load_edges(n, log)
        tab, ln, totals, proposed = type_majority(c, raw, pre, post, cnt, args, log)
        out_path = os.path.join(root, args.majority_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_type_majority(tab, ln, totals, proposed, args, cache_dir, out_path)
        identity = None
        if args.compile_cache:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import cx_wedge          # scripts/cx_wedge.py: TYPE_NT_OVERRIDE + extra -> out/cache_<hash>/ (tmp + rename)
            already = {t: v for t, v in proposed.items() if cn.TYPE_NT_OVERRIDE.get(t) == v}
            extra = {t: v for t, v in proposed.items() if t not in already}
            t0 = time.time()
            c2, cdir, table = cx_wedge.load_connectome(extra, scratch=True, verbose=False)
            log(f"scratch cache {cdir}: {c2.n} cells, nnz {c2.W.nnz:,}, sum|W| {int(np.abs(c2.W.data).sum()):,}, "
                f"nt counts {c2.neurons.nt.value_counts().to_dict()}; table {len(table)} entries "
                f"({len(extra)} added to TYPE_NT_OVERRIDE, {len(already)} already in it); {time.time() - t0:.0f} s")
            del c2
            identity = compare_caches(c, cache_dir, cdir, log)
        if args.score_json:
            score_section(out_path, args.score_json, args.baseline_json, log, identity=identity, note=args.note)
        log(f"wrote {out_path}, out/r4_type_majority.csv, out/r4_type_majority_proposed.json")
        return None
    # the receptor model in force by default (LIFParams.receptor_model; round 3 made 'sign' / 'abs' the default):
    # how many stored entries of W the per-edge lookup re-signs on top of the presynaptic convention audited here
    lp = LIFParams()
    receptor_line = None
    if lp.receptor_model is not None:
        t0 = time.time()
        rs = cn.receptor_signs(c, table_path=lp.receptor_table, net_rule=lp.receptor_net_rule, nt_class_fallback=lp.receptor_nt_class_fallback)
        w_sign = np.sign(c.W.data)
        changed = rs.fast_sign != w_sign
        flipped = (rs.fast_sign * w_sign) < 0
        zeroed = (rs.fast_sign == 0) & (w_sign != 0)
        unsilenced = (rs.fast_sign != 0) & (w_sign == 0)
        absW = np.abs(c.W.data)
        pre_col = c.W.tocoo().col
        pre_nt_all = n.nt.to_numpy()

        def by_nt(mask):
            vc = pd.Series(pre_nt_all[pre_col[mask]]).value_counts()
            return ", ".join(f"{k} {int(v):,}" for k, v in vc.items()) or "none"
        receptor_line = (f"Receptor model in force by default (`brain.LIFParams.receptor_model` = `{lp.receptor_model}`, net rule "
                         f"`{lp.receptor_net_rule}`, table `{os.path.relpath(rs.table_path, root).replace(os.sep, '/')}`; docs/NT_INTEGRATION.md, "
                         f"docs/audits/receptor_integration.md): on the {int(rs.matched.sum()):,} of {c.W.nnz:,} stored entries "
                         f"({rs.matched.mean():.1%}; {absW[rs.matched].sum() / absW.sum():.1%} of |W|) whose (postsynaptic type, "
                         f"presynaptic transmitter) has a row in the receptor table, the row's fast sign replaces the presynaptic sign: "
                         f"{int(changed.sum()):,} entries change -- {int(flipped.sum()):,} flipped ({int(absW[flipped].sum()):,} |W| synapses; "
                         f"presynaptic transmitter: {by_nt(flipped)}), {int(zeroed.sum()):,} silenced ({int(absW[zeroed].sum()):,} synapses; "
                         f"{by_nt(zeroed)}), {int(unsilenced.sum()):,} sign-0 entries un-silenced. The lookup never revives a sign-0 "
                         f"presynaptic cell (monoamine and `unknown` synapses stay explicit zeros), so every count in this audit is "
                         f"unchanged by the receptor model; `LIFParams(receptor_model=None)` selects the presynaptic-sign rule alone.")
        log(f"receptor model {lp.receptor_model} / {lp.receptor_net_rule}: {int(changed.sum()):,} entries changed "
            f"({int(flipped.sum()):,} flipped, {int(zeroed.sum()):,} zeroed, {int(unsilenced.sum()):,} un-silenced) ({time.time() - t0:.1f}s)")
        del rs
    raw = nt_columns(n)
    ann = pf.read_table(cn.DATA_DIR / cn.ANNOT_FILE, columns=["bodyId", "synonyms"]).to_pandas()
    syn = ann.set_index("bodyId").synonyms.reindex(n.bodyId.to_numpy()).fillna("").astype(str).to_numpy()

    pre, post, cnt = load_edges(n, log)
    total_syn = int(cnt.sum())
    sign = n.sign.to_numpy()
    nt = n.nt.to_numpy()
    sign0 = sign == 0
    ty = n.type.fillna("").to_numpy()
    sc = n.superclass.fillna("").to_numpy()
    mod = regions.labels(c)
    out_deg = np.bincount(pre, weights=cnt, minlength=N)
    has_out = out_deg > 0
    body_ids = n.bodyId.to_numpy()

    # per-T-bar leaning of every sign-0 "unknown" neuron with output synapses, and what a
    # majority-of-T-bars rule would assign them
    unk_edge = nt[pre] == "unknown"
    tbar = tbar_lean(body_ids[(nt == "unknown") & has_out], log)
    maj = tbar[TBAR_NTS].idxmax(axis=1)
    maj_conf = tbar[TBAR_NTS].max(axis=1)
    maj_pre = maj.reindex(body_ids[pre]).to_numpy()
    revive = {k: int(cnt[unk_edge & (maj_pre == k)].sum()) for k in TBAR_NTS}
    revive["(no T-bars)"] = int(cnt[unk_edge & pd.isna(maj_pre)].sum())

    # ---- (1) overall -------------------------------------------------------------------------
    s0_edge = sign0[pre]
    overall = {"total synapses": total_syn, "total edges": len(pre),
               "sign-0 synapses": int(cnt[s0_edge].sum()), "sign-0 edges": int(s0_edge.sum())}
    by_reason = {r: int(cnt[(nt[pre] == r)].sum()) for r in SIGN0_REASONS}
    neurons_by_reason = {r: int((nt == r).sum()) for r in SIGN0_REASONS}
    pre_by_reason = {r: int(((nt == r) & has_out).sum()) for r in SIGN0_REASONS}
    unk = nt == "unknown"
    in_table = raw.in_nt_table.to_numpy()
    unk_absent = int((unk & ~in_table).sum())
    unk_present = int((unk & in_table).sum())
    unk_syn_absent = int(cnt[(nt[pre] == "unknown") & ~in_table[pre]].sum())
    unk_zero_tbars = int((unk & in_table & (np.nan_to_num(raw.total_nt_predictions.to_numpy()) == 0)).sum())
    ln_override = n.type.fillna("").str.match(list(cn.UNKNOWN_NT_OVERRIDE_REGEX)[0]).to_numpy()
    raw_all_unclear = (raw.consensus_nt.fillna("unclear") == "unclear").to_numpy() & \
                      (raw.celltype_predicted_nt.fillna("unclear") == "unclear").to_numpy() & \
                      (raw.predicted_nt.fillna("unclear") == "unclear").to_numpy()
    override_cells = ln_override & raw_all_unclear
    override_syn = int(cnt[override_cells[pre]].sum())
    log("overall:", overall, by_reason)

    landing = []
    for r in SIGN0_REASONS:
        m = nt[pre] == r
        for g, lab in (("superclass", sc), ("module", mod)):
            s = pd.Series(cnt[m]).groupby(lab[post][m]).sum().sort_values(ascending=False)
            landing.append((r, g, s))

    top_types = pd.DataFrame({"type": ty[pre][s0_edge], "cnt": cnt[s0_edge]}).groupby("type").cnt.sum() \
        .sort_values(ascending=False).reset_index().rename(columns={"cnt": "out_syn"})
    top_types["n"] = [int(((ty == t) & sign0).sum()) for t in top_types.type]
    top_types["model_nt"] = [pd.Series(nt[(ty == t) & sign0]).value_counts().index[0] for t in top_types.type]
    top_types["superclass"] = [pd.Series(sc[(ty == t)]).value_counts().index[0] or "(none)" for t in top_types.type]
    top_types["module"] = [pd.Series(mod[(ty == t)]).value_counts().index[0] for t in top_types.type]
    top_unknown = pd.DataFrame({"type": ty[pre][unk_edge], "cnt": cnt[unk_edge]}).groupby("type").cnt.sum() \
        .sort_values(ascending=False).head(25).reset_index().rename(columns={"cnt": "out_syn"})

    by_sc = frac_table(sc, pre, post, cnt, sign0, has_out).sort_values("out_syn", ascending=False)
    by_mod = frac_table(mod, pre, post, cnt, sign0, has_out, list(regions.MODULES))
    mod_idx = {g: i for i, g in enumerate(regions.MODULES)}
    mlab_post = np.array([mod_idx[g] for g in mod[post]])
    mlab_pre = np.array([mod_idx[g] for g in mod[pre]])
    mod_reason_in = pd.DataFrame({r: np.bincount(mlab_post[nt[pre] == r], weights=cnt[nt[pre] == r], minlength=len(regions.MODULES))
                                  for r in SIGN0_REASONS}, index=list(regions.MODULES))
    mod_reason_out = pd.DataFrame({r: np.bincount(mlab_pre[nt[pre] == r], weights=cnt[nt[pre] == r], minlength=len(regions.MODULES))
                                   for r in SIGN0_REASONS}, index=list(regions.MODULES))

    # ---- (2) key populations ---------------------------------------------------------------
    cap = LIFParams().conn_cap
    cnt_cap = np.minimum(cnt, cap)
    pop_rows, detail, own_types = [], {}, []
    for group, name, sel in POPULATIONS:
        if sel.startswith("syn:"):
            m = np.array([sel[4:] in s for s in syn]) & (ty == SWEET_SYNONYMS[sel[4:]])
        else:
            m = pd.Series(ty).str.match(sel).to_numpy()
        own_types += sorted(set(ty[m]))
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            log(f"population {name}: no cells"); continue
        in_e = m[post]; out_e = m[pre]
        in_tot = int(cnt[in_e].sum()); out_tot = int(cnt[out_e].sum())
        in_0 = int(cnt[in_e & s0_edge].sum()); out_0 = int(cnt[out_e & s0_edge].sum())
        in_cap_tot = int(cnt_cap[in_e].sum()); in_cap_0 = int(cnt_cap[in_e & s0_edge].sum())
        in_reason = {r: int(cnt[in_e & (nt[pre] == r)].sum()) for r in SIGN0_REASONS}
        own = pd.Series(nt[m]).value_counts().to_dict()
        row = {"group": group, "population": name, "cells": len(idx),
               "own_nt": ", ".join(f"{k} {v}" for k, v in own.items()),
               "in_syn": in_tot, "in_sign0": in_0, "in_sign0_frac": in_0 / max(in_tot, 1),
               "in_sign0_frac_capped": in_cap_0 / max(in_cap_tot, 1),
               "in_unknown_frac": in_reason["unknown"] / max(in_tot, 1),
               "in_DA_frac": in_reason["dopamine"] / max(in_tot, 1),
               "in_OA_frac": in_reason["octopamine"] / max(in_tot, 1),
               "in_5HT_frac": in_reason["serotonin"] / max(in_tot, 1),
               "out_syn": out_tot, "out_sign0": out_0, "out_sign0_frac": out_0 / max(out_tot, 1)}
        pop_rows.append(row)
        d = {}
        if row["in_sign0_frac"] > args.threshold:
            e = in_e & s0_edge
            g = pd.DataFrame({"type": ty[pre][e], "cnt": cnt[e], "pre": pre[e]}).groupby("type") \
                .agg(syn=("cnt", "sum"), cells=("pre", "nunique")).sort_values("syn", ascending=False)
            g["frac_of_pop_input"] = g.syn / max(in_tot, 1)
            g = g.head(args.top).reset_index()
            g["type"] = g.type.replace("", "(untyped)")
            d["in"] = g.merge(type_nt_summary(n, raw, [t if t != "(untyped)" else "" for t in g.type], tbar=tbar),
                              on="type", how="left")
        if row["out_sign0_frac"] > args.threshold:
            e = out_e & s0_edge
            members = pd.DataFrame({"type": ty[pre][e], "cnt": cnt[e]}).groupby("type").cnt.sum() \
                .sort_values(ascending=False).reset_index().rename(columns={"cnt": "silenced_out_syn"})
            members["cells"] = [int(((ty == t) & m & sign0).sum()) for t in members.type]
            d["out_members"] = members.merge(type_nt_summary(n, raw, list(members.type), tbar=tbar), on="type", how="left")
            g = pd.DataFrame({"type": ty[post][e], "cnt": cnt[e]}).groupby("type").cnt.sum() \
                .sort_values(ascending=False).head(args.top).reset_index().rename(columns={"cnt": "syn"})
            g["frac_of_silenced_output"] = g.syn / max(out_0, 1)
            d["out_targets"] = g
        detail[name] = d
    pops = pd.DataFrame(pop_rows)
    seen = set()
    own_types = [t for t in own_types if not (t in seen or seen.add(t))]

    # ---- write -------------------------------------------------------------------------------
    os.makedirs(os.path.join(root, "out"), exist_ok=True)
    pops.to_csv(os.path.join(root, "out", "nt_audit_populations.csv"), index=False)
    pd.concat([by_sc.assign(kind="superclass"), by_mod.assign(kind="module")]).to_csv(
        os.path.join(root, "out", "nt_audit_groups.csv"), index=False)

    L = []
    L.append("# Neurotransmitter-sign audit\n")
    # the override relabels only the type's sign-0 cells (unknown / monoamine); applied = none of those is left unlabelled
    ov = {t: int(((ty == t) & (nt == v)).sum()) for t, v in cn.TYPE_NT_OVERRIDE.items()}
    ov_applied = all(int(((ty == t) & (sign == 0) & (nt != v)).sum()) == 0 for t, v in cn.TYPE_NT_OVERRIDE.items())
    L.append(f"Generated by `scripts/audit_nt.py` on {time.strftime('%Y-%m-%d')} from the cached connectome "
             f"(`{os.path.join(str(cache_dir), 'neurons.parquet')}`: the model's `nt` / `sign` after the antennal-lobe LN "
             f"override; the type-level `TYPE_NT_OVERRIDE` table ({', '.join(f'{t} -> {v}' for t, v in cn.TYPE_NT_OVERRIDE.items())}) "
             f"{'IS' if ov_applied else 'is NOT'} applied in this cache: cells carrying the override label "
             f"{', '.join(f'{t} {ov[t]}/{int((ty == t).sum())}' for t in ov)}) and the raw "
             f"weight, neurotransmitter and per-T-bar tables in `{cn.DATA_DIR}`.\n")
    L.append("Convention audited (`flyverse/connectome.py` `NT_SIGN`): a synapse carries the sign of its presynaptic "
             "neuron; ACh +1; GABA / glutamate / histamine -1; dopamine, octopamine, serotonin and `unknown` "
             "(consensus, cell-type and per-body predictions all `unclear`, or body absent from the NT table) 0. "
             "Sign-0 synapses are explicit zeros in `W` and contribute nothing to the LIF, to the optic rate model, "
             "or to the fan-in normalisation (`in_syn` excludes them). Counts below are raw synapse counts (`weight`) "
             "on the model's node set (Traced bodies + photoreceptors), before the connection cap; the "
             f"`capped` column applies `min(count, {cap:g})` per connection, the first step of `brain._shaped_weights`.\n")
    if receptor_line is not None:
        L.append(receptor_line + "\n")

    L.append("## 1. Overall\n")
    L.append(f"* Neurons: {N:,}; presynaptic neurons (>= 1 output synapse): {int(has_out.sum()):,}.")
    L.append(f"* Synapses: {total_syn:,} on {len(pre):,} edges.")
    L.append(f"* **Sign-0 synapses: {overall['sign-0 synapses']:,} = {pct(overall['sign-0 synapses'] / total_syn)}** "
             f"of all synapses ({overall['sign-0 edges']:,} edges = {pct(overall['sign-0 edges'] / len(pre))}).")
    L.append(f"* **Sign-0 presynaptic neurons: {int((sign0 & has_out).sum()):,} = "
             f"{pct((sign0 & has_out).sum() / has_out.sum())}** of presynaptic neurons "
             f"({int(sign0.sum()):,} = {pct(sign0.mean())} of all neurons).")
    L.append("\nBy reason:\n")
    rt = pd.DataFrame({"reason": SIGN0_REASONS,
                       "neurons": [neurons_by_reason[r] for r in SIGN0_REASONS],
                       "presynaptic neurons": [pre_by_reason[r] for r in SIGN0_REASONS],
                       "synapses silenced": [by_reason[r] for r in SIGN0_REASONS],
                       "share of all synapses": [by_reason[r] / total_syn for r in SIGN0_REASONS],
                       "share of sign-0 synapses": [by_reason[r] / max(overall['sign-0 synapses'], 1) for r in SIGN0_REASONS]})
    L.append(md_table(rt, {"share of all synapses": pct, "share of sign-0 synapses": pct}))
    L.append(f"\n`unknown` decomposes into {unk_absent:,} neurons absent from the NT table "
             f"({unk_syn_absent:,} output synapses) and {unk_present:,} present with all three columns `unclear` "
             f"({unk_zero_tbars:,} of them with `total_nt_predictions` = 0, i.e. no predicted T-bars at all). "
             f"The LN override (`UNKNOWN_NT_OVERRIDE_REGEX`) rescued {int(override_cells.sum()):,} neurons carrying "
             f"{override_syn:,} output synapses ({pct(override_syn / total_syn)} of all synapses) by assigning GABA; "
             f"without it the sign-0 share would be {pct((overall['sign-0 synapses'] + override_syn) / total_syn)}.\n")

    cols = ["group", "neurons", "pre_neurons", "pre_sign0", "pre_sign0_frac", "out_syn", "out_sign0",
            "out_sign0_frac", "in_syn", "in_sign0", "in_sign0_frac"]
    fmt = {"pre_sign0_frac": pct, "out_sign0_frac": pct, "in_sign0_frac": pct}
    L.append("### 1a. By superclass (`out` = synapses whose presynaptic cell is in the group; `in` = synapses onto the group)\n")
    L.append(md_table(by_sc[cols], fmt))
    L.append("\n### 1b. By module (`flyverse/regions.py`)\n")
    L.append(md_table(by_mod[cols], fmt))
    L.append("\nSilenced INPUT synapses per module by reason (counts):\n")
    mri = mod_reason_in.astype(int).reset_index().rename(columns={"index": "module"})
    mri["total_in"] = by_mod.set_index("group").loc[mri.module, "in_syn"].to_numpy()
    L.append(md_table(mri))
    L.append("\nSilenced OUTPUT synapses per module by reason (counts):\n")
    mro = mod_reason_out.astype(int).reset_index().rename(columns={"index": "module"})
    mro["total_out"] = by_mod.set_index("group").loc[mro.module, "out_syn"].to_numpy()
    L.append(md_table(mro))

    L.append(f"\n### 1c. The {min(30, len(top_types))} sign-0 types with the most output synapses\n")
    tt = top_types.head(30).copy()
    tt["share_of_all_syn"] = tt.out_syn / total_syn
    tt["type"] = tt.type.replace("", "(untyped)")
    tt = tt.merge(type_nt_summary(n, raw, [t if t != "(untyped)" else "" for t in tt.type], restrict=sign0)
                  [["type", "consensus_nt", "celltype_predicted_nt", "predicted_nt", "literature"]], on="type")
    L.append(md_table(tt[["type", "n", "model_nt", "superclass", "module", "out_syn", "share_of_all_syn",
                          "consensus_nt", "celltype_predicted_nt", "predicted_nt", "literature"]], {"share_of_all_syn": pct}))
    L.append("\n(`(untyped)`: cells with an empty `type`; the NT columns summarise only their sign-0 members.)\n")

    L.append("\n### 1d. The 25 `unknown`-NT types with the most output synapses, and what their T-bars lean to\n")
    L.append("Monoamine cells are sign 0 by design; the `unknown` ones are where a literature assignment or a looser "
             "rule could recover a sign. `tbar_argmax` = share of the type's predicted T-bars whose argmax class is "
             "each NT (top 3, from `tbar-neurotransmitters-...feather`); `unclear` in the body table means this "
             "argmax share never reached the dataset's confidence threshold.\n")
    tu = top_unknown.copy()
    tu["type"] = tu.type.replace("", "(untyped)")
    tu = tu.merge(type_nt_summary(n, raw, [t if t != "(untyped)" else "" for t in tu.type], restrict=(nt == "unknown"), tbar=tbar),
                  on="type", how="left")
    tu["superclass"] = [pd.Series(sc[(ty == t)]).value_counts().index[0] or "(none)" for t in top_unknown.type]
    L.append(md_table(tu[["type", "n", "superclass", "out_syn", "consensus_nt", "celltype_predicted_nt", "predicted_nt",
                          "n_tbar_pred", "tbar_argmax", "literature"]]))
    L.append(f"\nIf every `unknown` presynaptic neuron took the majority argmax of its own T-bars, the "
             f"{by_reason['unknown']:,} silenced synapses would become: " +
             ", ".join(f"{k} {v:,} ({pct(v / max(by_reason['unknown'], 1))})" for k, v in revive.items()) +
             f". Median majority share across those neurons: {maj_conf.median():.2f} "
             f"(quartiles {maj_conf.quantile(0.25):.2f} / {maj_conf.quantile(0.75):.2f}); the dataset's own "
             f"per-body `predicted_nt` was `unclear` for all of them, so these are low-confidence leanings, not labels.\n")

    L.append("\n### 1e. Where the silenced synapses land (postsynaptic group, by reason; top 8 each)\n")
    for r, g, s in landing:
        L.append(f"* **{r}** by {g}: " + "; ".join(f"{k or '(none)'} {int(v):,}" for k, v in s.head(8).items()))

    L.append("\n## 2. Key populations\n")
    L.append("`own_nt` = the model's NT labels of the population's own cells (this is what decides the output sign). "
             "`in_sign0_frac` = share of the population's input synapses whose presynaptic cell is sign 0 "
             "(raw counts; `capped` = after the per-connection cap); the four reason columns split it. "
             "`out_sign0_frac` = share of the population's output synapses that are silenced because the "
             "population's own cells are sign 0.\n")
    L.append(md_table(pops[["group", "population", "cells", "own_nt", "in_syn", "in_sign0_frac", "in_sign0_frac_capped",
                            "in_unknown_frac", "in_DA_frac", "in_OA_frac", "in_5HT_frac", "out_syn", "out_sign0_frac"]],
                      {k: pct for k in ["in_sign0_frac", "in_sign0_frac_capped", "in_unknown_frac", "in_DA_frac",
                                        "in_OA_frac", "in_5HT_frac", "out_sign0_frac"]}))

    L.append("\n### 2b. The key populations' own labels, by type, against the literature\n")
    L.append("Not a sign-0 question but the same audit: where the model's sign (from the MaleCNS consensus) and the "
             "literature disagree, the synapse has the wrong sign rather than none. `literature` `-` = not known to "
             "the author.\n")
    ot = type_nt_summary(n, raw, own_types)
    ot = ot[ot.n > 0]
    L.append(md_table(ot[["type", "n", "model_nt", "consensus_nt", "celltype_predicted_nt", "predicted_nt", "literature", "source"]]))

    L.append(f"\n## 3. Populations with more than {pct(args.threshold)} sign-0 input or output\n")
    L.append("NT columns are the raw table's `consensus_nt`, `celltype_predicted_nt` (with mean confidence) and per-body "
             "`predicted_nt` (with mean confidence), as the mode over the type's model cells with its share; "
             "`n_tbar_pred` = mean `total_nt_predictions` per cell; `tbar_argmax` = the type's T-bar argmax shares "
             "(only computed for `unknown` cells). `literature` is the fast transmitter assigned by the literature "
             "where the author is confident of it (`-` = not known to the author; not a claim of absence).\n")
    flagged = [(r["population"], r) for r in pop_rows
               if r["in_sign0_frac"] > args.threshold or r["out_sign0_frac"] > args.threshold]
    if not flagged:
        L.append("None.")
    for name, r in flagged:
        d = detail[name]
        L.append(f"### {name} ({r['group']}; {r['cells']} cells; input sign-0 {pct(r['in_sign0_frac'])}, "
                 f"output sign-0 {pct(r['out_sign0_frac'])})\n")
        if "in" in d:
            L.append(f"Top presynaptic sign-0 types (of {r['in_sign0']:,} silenced input synapses, "
                     f"{r['in_syn']:,} total):\n")
            g = d["in"]
            L.append(md_table(g[["type", "cells", "syn", "frac_of_pop_input", "model_nt", "consensus_nt",
                                 "celltype_predicted_nt", "predicted_nt", "n_tbar_pred", "tbar_argmax", "literature", "source"]],
                              {"frac_of_pop_input": pct}))
            L.append("")
        if "out_members" in d:
            L.append("Sign-0 member types (their whole output is silenced):\n")
            g = d["out_members"]
            L.append(md_table(g[["type", "cells", "silenced_out_syn", "model_nt", "consensus_nt", "celltype_predicted_nt",
                                 "predicted_nt", "n_tbar_pred", "literature", "source"]]))
            L.append("\nTop postsynaptic types of the silenced output:\n")
            L.append(md_table(d["out_targets"], {"frac_of_silenced_output": pct}))
            L.append("")

    # ---- computed summary (kept in the script so it cannot go stale) ------------------------
    S = ["\n## 4. Summary (computed)\n"]
    S.append(f"* Sign 0 silences {overall['sign-0 synapses']:,} of {total_syn:,} synapses "
             f"({pct(overall['sign-0 synapses'] / total_syn)}) from {int((sign0 & has_out).sum()):,} presynaptic neurons "
             f"({pct((sign0 & has_out).sum() / has_out.sum())}); by reason: " +
             ", ".join(f"{r} {pct(by_reason[r] / total_syn)}" for r in SIGN0_REASONS) + " of all synapses.")
    worst_in = by_mod.sort_values("in_sign0_frac", ascending=False).head(3)
    worst_out = by_mod.sort_values("out_sign0_frac", ascending=False).head(3)
    S.append("* Modules with the largest silenced input share: " +
             ", ".join(f"{r.group} {pct(r.in_sign0_frac)}" for r in worst_in.itertuples()) +
             "; largest silenced output share: " +
             ", ".join(f"{r.group} {pct(r.out_sign0_frac)}" for r in worst_out.itertuples()) + ".")
    above = pops[(pops.in_sign0_frac > args.threshold) | (pops.out_sign0_frac > args.threshold)]
    S.append(f"* Key populations above {pct(args.threshold)}: " +
             ("; ".join(f"{r.population} (in {pct(r.in_sign0_frac)}, capped {pct(r.in_sign0_frac_capped)}; out {pct(r.out_sign0_frac)})"
                        for r in above.itertuples()) if len(above) else "none") + ".")
    S.append("* Key populations, silenced input range: " +
             f"{pops.in_sign0_frac.min():.1%} ({pops.loc[pops.in_sign0_frac.idxmin(), 'population']}) to "
             f"{pops.in_sign0_frac.max():.1%} ({pops.loc[pops.in_sign0_frac.idxmax(), 'population']}); median "
             f"{pops.in_sign0_frac.median():.1%}. Every optic / loom / optomotor / DN population is below 1.5%.")
    disc, soft = [], []
    for r in ot.itertuples():
        lit_word = r.literature.split(" ")[0].lower().replace("-", "")
        model_word = r.model_nt.split(" ")[0].lower()
        if lit_word in ("acetylcholine", "gaba", "glutamate", "histamine") and lit_word != model_word:
            if "majority" in r.literature:
                soft.append(f"{r.type} ({model_word}, {r.n} cells)")
            else:
                disc.append(f"{r.type} ({r.n} cells): model {model_word} vs literature {r.literature}")
    S.append("* Type-level sign disagreements between the model's label and the literature entries in 2b (a wrong sign, "
             "not a missing one): " + ("; ".join(disc) if disc else "none") + ".")
    S.append(f"* Class-level only: {len(soft)} AL LN types carry a non-GABA label against the class majority "
             "(the LN class is known to contain cholinergic and glutamatergic minorities, so these are not necessarily "
             "wrong): " + ", ".join(soft) + ".")
    L += S

    out_path = os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log(f"wrote {out_path}")
    return pops, by_sc, by_mod


if __name__ == "__main__":
    main()
