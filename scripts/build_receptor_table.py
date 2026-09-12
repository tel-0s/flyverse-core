#!/usr/bin/env python
"""Per-type transmitter calls, receptor response classes and the per-edge lookup (docs/NT_INTEGRATION.md steps 3-5).

Everything here is derived deterministically from the per-source tables that the acquire / map tasks of the
NT-integration workflow wrote into flyverse/data/ (no downloads, no simulation, CPU only):

  type_map_ozel2021.csv, expression_ozel2021.csv, expression_ozel2021_mm.csv
      Özel et al. 2021 (Nature 589:88; GEO GSE142787): adult optic-lobe scRNA-seq clusters; per-cluster mean
      log-normalised expression and the authors' mixture-model P(on).
  type_map_davis2020.csv, expression_davis2020.csv
      Davis et al. 2020 (eLife 9:e50901; GEO GSE116969): driver-sorted bulk nuclear RNA-seq of visual / MB / CX
      types; TPM and the authors' per-gene mixture-model P(on).
  type_map_central.csv, expression_central.csv
      Davie et al. 2018 (Cell 174:982; GEO GSE107451) and Fly Cell Atlas 2022 (Science 375:eabk2432) head 10x:
      per-cluster mean log1p(cp10k) and fraction of cells expressing.
  type_map_kurmangaliyev2020.csv, expression_kurmangaliyev2020.csv
      Kurmangaliyev et al. 2020 (Neuron 108:1045; GEO GSE156455): PUPAL (24-96 h APF) optic-lobe scRNA-seq; the
      96 h APF (pharate adult) rows are used, 'late' (72-96 h pooled) for clusters with < 20 cells at 96 h.
  type_map_nern2025.csv
      Nern et al. 2025 (Nature 641:1225): per-type transmitter prediction (no expression data).
  type_aliases.csv
      read for the record only; every type map already carries MaleCNS v1.0 names.
  cache/ (flyverse.connectome.load): neurons (type, superclass, nt, sign) and W[post, pre] (signed counts; sign-0
      presynaptic cells are explicit zeros). Optional: the raw MaleCNS weights feather for the uncapped synapse
      counts of those zeroed edges (FLYVERSE_DATA_DIR; falls back to c.W only).

Outputs:
  flyverse/data/nt_by_type_transcriptome.csv   one row per MaleCNS type with an expression profile
  flyverse/data/receptors_by_type.csv          one row per (MaleCNS type, transmitter)
  docs/audits/receptor_nt_disagreements.md     transmitter cross-check (transcriptome vs MaleCNS consensus vs Nern 2025)
  docs/audits/receptor_rules.md                the rules, thresholds and the coverage of the edge lookup
  stdout: headline numbers

Importable:
  edge_lookup(c, receptors=None, net_rule="class", nt_class_fallback=False) -> DataFrame aligned with c.W's stored
      entries (c.W.tocoo() order, explicit zeros included), and edge_stats(c, edges, raw=None) -> coverage /
      flip / monoamine summaries. load_raw_counts(c) -> uncapped synapse counts aligned the same way (or None).

Profile selection per (type, transmitter) -- round-2 rule (docs/audits/receptor_verification.md, verify:implement):
  the best-ranked profile (tier, QC, source priority) decides the sign, EXCEPT that a profile with no receptor
  group for the transmitter ('none' = the edge would be silenced) only stands if every other source profiling the
  type agrees. A single-nucleus 'none' (fca2022 / davie2018, dropout-prone) never outranks a whole-cell profile
  (davis2020 / ozel2021 / kurmangaliyev2020) that has the group on: that profile is used instead
  ('group_on_override'); any other disagreement about 'none' falls back to NT_SIGN ('none_contested'), and so does a
  'none' carried by a single source ('none_single_source'): a silencing needs >= 2 agreeing sources.
Round-3 rule (the symmetric case; docs/audits/receptor_verification.md, round-2 critic follow-up 1): a FLIP -- the
  selected profile's fast net sign is the opposite of the NT_SIGN prior (a +1 on glutamate, a -1 on a +1 transmitter)
  -- that is contradicted by >= 1 other profiled source whose net for the same variant is the prior's sign falls back
  to NT_SIGN ('flip_contested'; --flip-rule any, the default). --flip-rule majority falls back only when the
  contradicting sources outnumber the other sources agreeing with the flip ('flip_contested_majority');
  --flip-rule off reproduces the round-2 table. The rule is applied per variant (class / abs / nonmda) with each
  source's net under that variant; 'mixed' and 'none' sources neither contradict nor agree.

Run:  PYTHONIOENCODING=utf-8 python scripts/build_receptor_table.py [--previous <old receptors_by_type.csv>] [--flip-rule any|majority|off]
      (about 2-3 min with the raw weights; --previous adds a change section to receptor_rules.md)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyverse import connectome as cn  # noqa: E402
from flyverse import regions  # noqa: E402

DATA = ROOT / "flyverse/data"
AUDITS = ROOT / "docs/audits"
OUT_NT = DATA / "nt_by_type_transcriptome.csv"
OUT_RECEPTORS = DATA / "receptors_by_type.csv"
OUT_DISAGREE = AUDITS / "receptor_nt_disagreements.md"
OUT_RULES = AUDITS / "receptor_rules.md"
RAW_WEIGHTS = Path(os.environ.get("FLYVERSE_DATA_DIR", r"D:\Datasets\male-cns-connectome-v1.0\flat-connectome")) / \
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# --------------------------------------------------------------------------------------------------------------
# Rules (also written to docs/audits/receptor_rules.md and the CSV headers)
# --------------------------------------------------------------------------------------------------------------
TRANSMITTERS = ["acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "octopamine", "serotonin"]
CLASSICAL = ["acetylcholine", "gaba", "glutamate", "histamine"]
MONOAMINES = ["dopamine", "octopamine", "serotonin"]

# Transmitter identity from synthesis / transport genes: (genes, rule). "any": one gene suffices (ChAT is poorly
# captured in 10x data, VAChT is the reliable 10x marker; both are bimodal in bulk data); "all": both genes must be
# on (Tbh and DAT are expressed at low level in many non-aminergic clusters, so the enzyme alone is not enough).
# GABA uses Gad1 only: VGAT is 1.7-586 TPM over the neuronal QC-pass Davis 2020 rows (photoreceptors 1.7-14.5 with
# P(on) = 0; 60-586 TPM with P(on) = 1 in every other neuronal row, T1 172.6 / L1 304.6 / Mi1 236.1), so it cannot
# discriminate among non-photoreceptor neurons; it is recorded in the expression tables but not used for the call.
NT_MARKERS = {
    "acetylcholine": (["ChAT", "VAChT"], "any"),
    "gaba": (["Gad1"], "any"),
    "glutamate": (["VGlut"], "any"),
    "histamine": (["Hdc"], "any"),
    "dopamine": (["ple", "DAT"], "all"),
    "octopamine": (["Tdc2", "Tbh"], "all"),
    "serotonin": (["Trh", "SerT"], "all"),
}
NT_SECONDARY_RATIO = 0.5   # another present classical marker within 0.5x of the primary -> co-expression flag

# Receptor groups: transmitter -> {"fast": [(sign, name, genes)], "slow": [...]}. The sign is the direction of the
# postsynaptic effect on membrane potential / spiking that the receptor class produces in Drosophila neurons.
RECEPTOR_GROUPS = {
    "acetylcholine": {
        "fast": [(+1, "nAChR", ["nAChRalpha1", "nAChRalpha2", "nAChRalpha3", "nAChRalpha4", "nAChRalpha5",
                                "nAChRalpha6", "nAChRalpha7", "nAChRbeta1", "nAChRbeta2", "nAChRbeta3"])],
        "slow": [(+1, "mAChR-A", ["mAChR-A"]), (-1, "mAChR-B", ["mAChR-B"])],
    },
    "gaba": {
        "fast": [(-1, "GABA-A", ["Rdl", "Lcch3", "Grd"])],
        "slow": [(-1, "GABA-B", ["GABA-B-R1", "GABA-B-R2", "GABA-B-R3"])],
    },
    "glutamate": {
        "fast": [(-1, "GluCl", ["GluClalpha"]),
                 (+1, "iGluR", ["KaiR1D", "GluRIA", "GluRIB", "Nmdar1", "Nmdar2"])],
        "slow": [(-1, "mGluR", ["mGluR"])],
    },
    "histamine": {
        "fast": [(-1, "HisCl", ["HisCl1", "ort"])],
        "slow": [],
    },
    "dopamine": {
        "fast": [],
        "slow": [(+1, "Dop1R/DopEcR", ["Dop1R1", "Dop1R2", "DopEcR"]), (-1, "Dop2R", ["Dop2R"])],
    },
    "octopamine": {
        "fast": [],
        "slow": [(+1, "Oamb/Octbeta", ["Oamb", "Octbeta1R", "Octbeta2R", "Octbeta3R"]), (-1, "Octalpha2R", ["Octalpha2R"])],
    },
    "serotonin": {
        "fast": [],
        "slow": [(+1, "5-HT2/7", ["5-HT2A", "5-HT2B", "5-HT7"]), (-1, "5-HT1", ["5-HT1A", "5-HT1B"])],
    },
}
# The NMDA receptor is an obligate Nmdar1 + Nmdar2 heteromer: the iGluR group counts as present only if one of the
# non-NMDA members is on or both Nmdar genes are on. Because Nmdar2 is 5.3-1,073 TPM over the neuronal QC-pass
# Davis 2020 rows -- off (P(on) = 0) only in the photoreceptors R1-6 / R7 / R8, 102-1,073 TPM with P(on) = 1 in every
# other neuronal row (a ubiquitous, voltage-gated coincidence detector rather than a fast transmitter receptor) --
# the glutamate rows also carry a `_nonmda` variant in which the fast + group is KaiR1D / GluRIA / GluRIB only.
NMDA_PAIR = ("Nmdar1", "Nmdar2")
IGLUR_NONMDA = (+1, "iGluR_nonNMDA", ["KaiR1D", "GluRIA", "GluRIB"])
GENES = sorted({g for t in NT_MARKERS.values() for g in t[0]} |
               {g for r in RECEPTOR_GROUPS.values() for kind in r.values() for _, _, gs in kind for g in gs})

# "on" thresholds per source (see receptor_rules.md).
OZEL_ON = 0.5            # mixture-model P(on)
DAVIS_P_ON = 0.5         # Davis mixture-model P(on) ...
DAVIS_TPM_FLOOR = 10.0   # ... and at least the authors' modelled-gene floor (10 TPM); TPM >= 10 alone when P(on) is NaN
CENTRAL_FRAC_ON = 0.2    # fraction of cells / nuclei with >= 1 UMI
ABS_RATIO = 2.0          # net_abs: the larger group must exceed the other by this factor, else 'mixed'
POOL_PURE_SHARE = 0.9    # a pooled source row is 'mixed' when its member types' majority model label holds < 90 % of cells
WEAK_MARKER_RANK = 0.25  # primary marker below this quantile of the source's on-profiles -> confidence 'low'
LOG_SOURCES = {"ozel2021", "kurmangaliyev2020", "fca2022", "davie2018"}   # levels are means of log1p: linearised with expm1 before summing
KURM_TIME, KURM_TIME_FALLBACK, KURM_MIN_CELLS = "96h", "late", 20   # Kurmangaliyev: pharate-adult rows, pooled 72-96 h when < 20 cells

GAIN_CLASSES = ["none", "low", "mid", "high"]
TIER_RANK = {"exact": 0, "alias": 1, "fuzzy": 2, "class": 3}
# adult bulk / whole-cell first; the pupal whole-cell atlas after the adult ones; single-nucleus last
SOURCE_PRIORITY = {"davis2020": 0, "ozel2021": 1, "kurmangaliyev2020": 2, "fca2022": 3, "davie2018": 4}
WHOLE_CELL_SOURCES = {"davis2020", "ozel2021", "kurmangaliyev2020"}   # bulk nuclear / whole-cell: a 'none' is evidence
SINGLE_NUCLEUS_SOURCES = {"fca2022", "davie2018"}                      # 10x single nucleus: a 'none' may be dropout
QC_RANK = {"pass": 0, "suboptimal_only": 1}
NON_NEURONAL_KEYWORDS = ("glia", "glial", "muscle", "hemocyte", "plasmatocyte", "cone cell", "epithelial", "pigment",
                         "fat body", "fat mass", "artefact", "unannotated", "dissected", "not neurons",
                         "non-neuronal", "heat-shock", "photoreceptor-like", "stress signature")
OL_SUPERCLASSES = ["ol_intrinsic", "visual_projection", "ol_sensory"]
CENTRAL_SUPERCLASSES = ["cb_intrinsic", "cb_sensory", "cb_endocrine", "cb_motor", "cb_efferent", "visual_centrifugal"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def is_non_neuronal(*texts) -> bool:
    s = " ".join(str(t) for t in texts if isinstance(t, str)).lower()
    return any(k in s for k in NON_NEURONAL_KEYWORDS)


# --------------------------------------------------------------------------------------------------------------
# 1. Source loaders -> (profiles, maps)
#    profiles: one row per (source, source_key): val_<gene> (level, source units), on_<gene> (bool), neuronal (bool)
#    maps: one row per (source, source_key, malecns_type): tier, qc, n_targets, circular, source_name
# --------------------------------------------------------------------------------------------------------------
def _profile(base: dict, vals: dict, ons: dict, neuronal) -> pd.DataFrame:
    """Assemble a profile frame in one concat: val_<gene> (level, source units), lin_<gene> (linearised level),
    on_<gene> (bool), neuronal."""
    cols = dict(base)
    src = base["source"]
    for g in GENES:
        v = np.asarray(vals.get(g, np.nan), dtype=float)
        v = np.broadcast_to(v, (len(base["source_key"]),)).copy()
        cols["val_" + g] = v
        cols["lin_" + g] = np.nan_to_num(np.expm1(v) if src in LOG_SOURCES else v)
        o = np.asarray(ons.get(g, False))
        cols["on_" + g] = np.broadcast_to(o, (len(base["source_key"]),)).copy()
    cols["neuronal"] = np.asarray(neuronal, dtype=bool)
    return pd.DataFrame(cols)


def load_ozel():
    ex = read_csv(DATA / "expression_ozel2021.csv")
    mm = read_csv(DATA / "expression_ozel2021_mm.csv")
    tm = read_csv(DATA / "type_map_ozel2021.csv")
    ex["source_key"] = ex.source_cluster.astype(str)
    mm["source_key"] = mm.source_cluster.astype(str)
    mm = mm.set_index("source_key")
    unm = tm[tm.tier == "unmatched"]
    bad = {str(k) for k, e in zip(unm.source_cluster, unm.evidence) if is_non_neuronal(e)}
    prof = _profile({"source": "ozel2021", "source_key": ex.source_key.to_numpy(), "source_name": ex.source_name.astype(str).to_numpy()},
                    {g: ex[g].to_numpy() for g in GENES if g in ex},
                    {g: (mm.loc[ex.source_key, g].to_numpy() >= OZEL_ON) for g in GENES if g in mm},
                    ~ex.source_key.isin(bad))
    m = tm[tm.tier != "unmatched"].copy()
    m["source"] = "ozel2021"
    m["source_key"] = m.source_cluster.astype(str)
    m["qc"] = "pass"
    m["circular"] = False
    m["n_targets"] = m.groupby("source_key").malecns_type.transform("size")
    return prof, m[["source", "source_key", "source_name", "malecns_type", "tier", "qc", "n_targets", "circular", "evidence"]]


def load_davis():
    ex = read_csv(DATA / "expression_davis2020.csv")
    tm = read_csv(DATA / "type_map_davis2020.csv")
    vals, ons = {}, {}
    for g in GENES:
        tpm = ex[g + "_tpm"].to_numpy(dtype=float)
        pon = ex[g + "_p_on"].to_numpy(dtype=float)
        vals[g] = tpm
        ons[g] = np.where(np.isnan(pon), tpm >= DAVIS_TPM_FLOOR, (pon >= DAVIS_P_ON) & (tpm >= DAVIS_TPM_FLOOR))
    unm = tm[tm.tier == "unmatched"]
    bad = {n for n, e in zip(unm.source_name, unm.evidence) if is_non_neuronal(e, n)}
    population = {"ChAT", "Gad1", "VGlut", "opticlobe", "lamina"}   # whole-class / tissue references, not types
    prof = _profile({"source": "davis2020", "source_key": ex.source_name.to_numpy(), "source_name": ex.source_name.to_numpy(),
                     "qc": ex.qc.to_numpy()}, vals, ons, ~ex.source_name.isin(bad | population))
    m = tm[tm.tier != "unmatched"].copy()
    m["source"] = "davis2020"
    m["source_key"] = m.source_name
    m["circular"] = m.malecns_type.str.startswith("<nt=")
    m["n_targets"] = m.groupby("source_key").malecns_type.transform("size")
    return prof, m[["source", "source_key", "source_name", "malecns_type", "tier", "qc", "n_targets", "circular", "evidence"]]


def load_central():
    ex = read_csv(DATA / "expression_central.csv")
    tm = read_csv(DATA / "type_map_central.csv")
    ex = ex[ex.sex_subset == "all"]
    val = ex[ex.metric == "mean_log1p_cp10k"].set_index(["source", "source_name"])
    frac = ex[ex.metric == "frac_expr"].set_index(["source", "source_name"])
    frac = frac.loc[val.index]
    unm = tm[tm.tier == "unmatched"]
    bad = {(s, str(n)) for s, n, e in zip(unm.source, unm.source_name, unm.evidence) if is_non_neuronal(e, n)}
    profs = []
    for src in ("davie2018", "fca2022"):
        m = np.array([i[0] == src for i in val.index])
        if not m.any():
            continue
        v, f = val[m], frac[m]
        keys = [str(i[1]) for i in v.index]
        profs.append(_profile({"source": src, "source_key": np.array(keys), "source_name": np.array(keys)},
                              {g: v[g].to_numpy(dtype=float) for g in GENES if g in v},
                              {g: (f[g].to_numpy(dtype=float) >= CENTRAL_FRAC_ON) for g in GENES if g in f},
                              [(src, k) not in bad for k in keys]))
    prof = pd.concat(profs, ignore_index=True)
    m = tm[tm.tier != "unmatched"].copy()
    m["source_key"] = m.source_name.astype(str)
    m["qc"] = "pass"
    m["circular"] = m.rule.fillna("").str.startswith("query:nt")
    m["n_targets"] = m.groupby(["source", "source_key"]).malecns_type.transform("size")
    return prof, m[["source", "source_key", "source_name", "malecns_type", "tier", "qc", "n_targets", "circular", "evidence"]]


def load_kurmangaliyev():
    """Kurmangaliyev 2020 (pupal optic lobe): per cluster the 96 h APF rows (pharate adult, the closest to the adult
    sources: median Pearson r 0.90 with the Özel adult profile of the same type over the 54 genes, vs 0.89 for the
    72-96 h pool and 0.80 for all timepoints), or the 72-96 h pool when the cluster has < KURM_MIN_CELLS cells at 96 h.
    'on' = fraction of cells with >= 1 UMI >= CENTRAL_FRAC_ON (no mixture model is published for this atlas)."""
    ex = read_csv(DATA / "expression_kurmangaliyev2020.csv")
    tm = read_csv(DATA / "type_map_kurmangaliyev2020.csv")
    ex["source_cluster"] = ex.source_cluster.astype(str)
    val = ex[ex.metric == "mean_log1p_cp10k"].set_index(["source_cluster", "time"])
    frac = ex[ex.metric == "frac_expr"].set_index(["source_cluster", "time"])
    keys, times = [], []
    for k in sorted(val.index.get_level_values(0).unique()):
        t = KURM_TIME
        if (k, t) not in val.index or val.loc[(k, t), "n_cells"] < KURM_MIN_CELLS:
            t = KURM_TIME_FALLBACK if (k, KURM_TIME_FALLBACK) in val.index else "all"
        keys.append(k); times.append(t)
    idx = list(zip(keys, times))
    v, f = val.loc[idx], frac.loc[idx]
    unm = tm[tm.tier == "unmatched"]
    bad = {str(k) for k, n, e in zip(unm.source_cluster, unm.source_name, unm.evidence) if is_non_neuronal(e, n)}
    names = np.array([f"{k}@{t}" for k, t in idx])
    prof = _profile({"source": "kurmangaliyev2020", "source_key": np.array(keys), "source_name": names,
                     "time": np.array(times), "n_cells_source": v.n_cells.to_numpy()},
                    {g: v[g].to_numpy(dtype=float) for g in GENES if g in v},
                    {g: (f[g].to_numpy(dtype=float) >= CENTRAL_FRAC_ON) for g in GENES if g in f},
                    [k not in bad for k in keys])
    m = tm[tm.tier != "unmatched"].copy()
    m["source"] = "kurmangaliyev2020"
    m["source_key"] = m.source_cluster.astype(str)
    tmap = dict(zip(keys, times))
    m["source_name"] = [f"{k}@{tmap.get(k, '?')}" for k in m.source_key]
    m["qc"] = "pass"
    m["circular"] = False
    m["n_targets"] = m.groupby("source_key").malecns_type.transform("size")
    return prof, m[["source", "source_key", "source_name", "malecns_type", "tier", "qc", "n_targets", "circular", "evidence"]]


# --------------------------------------------------------------------------------------------------------------
# 2. Per-profile derivations
# --------------------------------------------------------------------------------------------------------------
def marker_call(prof: pd.DataFrame) -> pd.DataFrame:
    """Transmitter identity per profile row from the synthesis / transport genes (linearised levels)."""
    pres, score = {}, {}
    for nt, (genes, rule) in NT_MARKERS.items():
        on = np.column_stack([prof["on_" + g].to_numpy() for g in genes])
        v = np.column_stack([prof["lin_" + g].to_numpy(dtype=float) for g in genes])
        pres[nt] = on.all(axis=1) if rule == "all" else on.any(axis=1)
        score[nt] = v.min(axis=1) if rule == "all" else v.max(axis=1)
    # quantile rank of a marker level among the source's neuronal profiles where that marker is present
    rank = {nt: np.full(len(prof), np.nan) for nt in TRANSMITTERS}
    src_arr, neu = prof.source.to_numpy(), prof.neuronal.to_numpy()
    for nt in TRANSMITTERS:
        for src in np.unique(src_arr):
            ref = (src_arr == src) & neu & pres[nt]
            rows = (src_arr == src) & pres[nt]
            if ref.sum() >= 2:
                ref_vals = np.sort(score[nt][ref])
                rank[nt][rows] = np.searchsorted(ref_vals, score[nt][rows], side="right") / len(ref_vals)
            else:
                rank[nt][rows] = 1.0
    out = []
    for i in range(len(prof)):
        cl = [(score[nt][i], nt) for nt in CLASSICAL if pres[nt][i]]
        mono = [(score[nt][i], nt) for nt in MONOAMINES if pres[nt][i]]
        cl.sort(reverse=True); mono.sort(reverse=True)
        if cl:
            primary = cl[0][1]
            secondary = [nt for s, nt in cl[1:] if s >= NT_SECONDARY_RATIO * cl[0][0]]
            co = [nt for _, nt in mono]
        elif mono:
            primary = mono[0][1]
            secondary = [nt for s, nt in mono[1:] if s >= NT_SECONDARY_RATIO * mono[0][0]]
            co = []
        else:
            primary, secondary, co = "none", [], []
        out.append({"nt_call": primary, "nt_secondary": "+".join(secondary), "co_monoamine": "+".join(co),
                    "marker_rank": (rank[primary][i] if primary != "none" else np.nan),
                    "marker_scores": ";".join(f"{nt}={score[nt][i]:.3g}{'*' if pres[nt][i] else ''}" for nt in TRANSMITTERS)})
    return pd.DataFrame(out, index=prof.index)


def all_groups():
    for nt, kinds in RECEPTOR_GROUPS.items():
        for kind, groups in kinds.items():
            for sign, name, genes in groups:
                yield nt, kind, sign, name, genes
    yield "glutamate", "fast", IGLUR_NONMDA[0], IGLUR_NONMDA[1], IGLUR_NONMDA[2]


def receptor_groups(prof: pd.DataFrame) -> pd.DataFrame:
    """Per profile row and receptor group: on (bool), val (sum of the linearised levels of the members that are on),
    lead (the on-member with the largest level), cls (0-3): tertiles within the source over neuronal rows where the
    group is on."""
    cols = {}
    src_arr, neu = prof.source.to_numpy(), prof.neuronal.to_numpy()
    for nt, kind, sign, name, genes in all_groups():
        on = np.column_stack([prof["on_" + g].to_numpy() for g in genes])
        v = np.column_stack([prof["lin_" + g].to_numpy(dtype=float) for g in genes])
        if name == "iGluR":
            non_nmda = [j for j, g in enumerate(genes) if g not in NMDA_PAIR]
            nmda = [j for j, g in enumerate(genes) if g in NMDA_PAIR]
            present = on[:, non_nmda].any(axis=1) | on[:, nmda].all(axis=1)
            on = on.copy()
            on[:, nmda] &= on[:, nmda].all(axis=1)[:, None]      # NMDA counts only as a pair
        else:
            present = on.any(axis=1)
        von = np.where(on, v, 0.0)
        val = von.sum(axis=1)
        lead_idx = von.argmax(axis=1)
        lead = np.where(present, np.array(genes)[lead_idx], "")
        key = f"{nt}|{kind}|{name}"
        cols[key + "|on"] = present
        cols[key + "|val"] = val
        cols[key + "|lead"] = lead
        cls = np.zeros(len(prof), dtype=int)
        for src in np.unique(src_arr):
            ref = (src_arr == src) & neu & present
            rows = (src_arr == src) & present
            if ref.sum() >= 3:
                q1, q2 = np.quantile(val[ref], [1 / 3, 2 / 3])
                cls[rows] = 1 + (val[rows] > q1).astype(int) + (val[rows] > q2).astype(int)
            else:
                cls[rows] = 2
        cols[key + "|cls"] = cls
    return pd.DataFrame(cols, index=prof.index)


def combine(sign_groups: list[tuple[int, str]], rec: pd.Series, prior: int):
    """Combine the receptor groups of one (transmitter, kind) for one profile row.
    Returns dict: pos_cls, neg_cls, pos_val, neg_val, pos_lead, neg_lead, net ('+1'/'-1'/'mixed'/'none'), sign,
    gain (class of the winner), net_abs, sign_abs, gain_abs, groups (string)."""
    pos_cls = neg_cls = 0
    pos_val = neg_val = 0.0
    pos_lead = neg_lead = ""
    pos_lead_val = neg_lead_val = -1.0
    names = []
    for sign, key in sign_groups:
        on, val, cls, lead = rec[key + "|on"], rec[key + "|val"], int(rec[key + "|cls"]), rec[key + "|lead"]
        names.append(f"{key.split('|')[-1]}:{GAIN_CLASSES[cls] if on else 'off'}")
        if not on:
            continue
        if sign > 0:
            pos_cls, pos_val = max(pos_cls, cls), pos_val + val
            if val > pos_lead_val:
                pos_lead, pos_lead_val = lead, val
        else:
            neg_cls, neg_val = max(neg_cls, cls), neg_val + val
            if val > neg_lead_val:
                neg_lead, neg_lead_val = lead, val
    # class rule (the plan's): sign of the larger gain class
    if pos_cls == 0 and neg_cls == 0:
        net, sign, gain = "none", 0, 0
    elif pos_cls > neg_cls:
        net, sign, gain = "+1", +1, pos_cls
    elif neg_cls > pos_cls:
        net, sign, gain = "-1", -1, neg_cls
    else:
        net, sign, gain = "mixed", prior, pos_cls
    # absolute rule: sum of member levels in source units, larger side must exceed the other ABS_RATIO-fold
    if pos_cls == 0 and neg_cls == 0:
        net_a, sign_a, gain_a = "none", 0, 0
    elif neg_cls == 0 or pos_val >= ABS_RATIO * neg_val:
        net_a, sign_a, gain_a = "+1", +1, pos_cls
    elif pos_cls == 0 or neg_val >= ABS_RATIO * pos_val:
        net_a, sign_a, gain_a = "-1", -1, neg_cls
    else:
        net_a, sign_a, gain_a = "mixed", prior, max(pos_cls, neg_cls)
    return dict(pos_cls=pos_cls, neg_cls=neg_cls, pos_val=pos_val, neg_val=neg_val, pos_lead=pos_lead, neg_lead=neg_lead,
                net=net, sign=sign, gain=gain, net_abs=net_a, sign_abs=sign_a, gain_abs=gain_a, groups=";".join(names))


NONE_CONTESTED = "none_contested"
NONE_SINGLE = "none_single_source"
MIN_SOURCES_TO_SILENCE = 2   # a silencing (no fast receptor) needs at least this many agreeing sources
NONE_FALLBACK = (NONE_CONTESTED, NONE_SINGLE)   # net labels whose fast sign is the NT_SIGN prior


def select_profile(cands: list[dict], min_sources: int = MIN_SOURCES_TO_SILENCE) -> tuple[int, str]:
    """Round-2 profile selection for one (type, transmitter, fast variant).

    cands: the best row of every source that profiles the type, in rank order (tier, QC, source priority, ...),
    each {'source': str, 'net': '+1' | '-1' | 'mixed' | 'none'} where net == 'none' means the profile has no fast
    receptor group for the transmitter (the edge would be silenced).
    Returns (index of the candidate to use, selection label):
      'primary'            -- the best-ranked profile decides (it has the group on, or every source agrees on 'none'
                              and there are at least `min_sources` of them);
      'group_on_override'  -- the best-ranked profile is a single-nucleus 'none' and a whole-cell source has the group
                              on: the best-ranked whole-cell profile with the group on is used instead;
      'none_contested'     -- the best-ranked profile is 'none' but another source has the group on and the override
                              does not apply (whole-cell 'none' vs any 'on', or single-nucleus 'none' vs single-nucleus
                              'on'): the caller keeps the primary profile but falls back to NT_SIGN for the fast sign;
      'none_single_source' -- every source agrees on 'none' but fewer than `min_sources` profile the type (one
                              profile's dropout / threshold call is not enough to silence an anatomical synapse):
                              the caller keeps the primary profile and falls back to NT_SIGN.
    A silencing therefore needs every source that profiles the type to agree, and at least two of them."""
    if not cands:
        raise ValueError("no candidate profiles")
    if cands[0]["net"] != "none":
        return 0, "primary"
    on = [i for i, cnd in enumerate(cands) if i > 0 and cnd["net"] != "none"]
    if not on:
        return 0, ("primary" if len(cands) >= min_sources else NONE_SINGLE)
    if cands[0]["source"] in SINGLE_NUCLEUS_SOURCES:
        for i in on:
            if cands[i]["source"] in WHOLE_CELL_SOURCES:
                return i, "group_on_override"
    return 0, NONE_CONTESTED


FLIP_CONTESTED = "flip_contested"
FLIP_CONTESTED_MAJORITY = "flip_contested_majority"
FLIP_RULES = ("any", "majority", "off")
FLIP_RULE_DEFAULT = "any"
FLIP_FALLBACK = (FLIP_CONTESTED, FLIP_CONTESTED_MAJORITY)   # net labels of a contested flip (fast sign = the NT_SIGN prior)
FAST_FALLBACK = NONE_FALLBACK + FLIP_FALLBACK              # every net label whose fast sign is the prior with gain class none


def contest_flip(cands: list[dict], i: int, prior: int, rule: str = FLIP_RULE_DEFAULT) -> tuple[str | None, list[str]]:
    """Round-3 symmetric rule for one (type, transmitter, variant): is the selected profile's FLIP contested?

    cands: one entry per source profiling the type, {'source': str, 'net': '+1' | '-1' | 'mixed' | 'none'} with the
    net computed under the SAME variant (class, abs or nonmda) as the decision; i: the candidate select_profile chose;
    prior: cn.NT_SIGN of the transmitter (+1 / -1; 0 for the monoamines, which have no fast group).
    A flip is a net whose sign is the opposite of the prior (a '+1' on glutamate; a '-1' on acetylcholine). Sources
    whose net is the prior's own sign contradict it; sources at the flip's sign agree with it; 'mixed' and 'none'
    do neither.
    Returns (label, contradicting sources): label FLIP_CONTESTED when rule == 'any' and >= 1 source contradicts,
    FLIP_CONTESTED_MAJORITY when rule == 'majority' and the contradicting sources outnumber the agreeing ones, else
    None (the flip stands; the list still names any contradicting sources). rule == 'off' never contests."""
    if rule not in FLIP_RULES:
        raise ValueError(f"flip rule must be one of {FLIP_RULES}, got {rule!r}")
    if rule == "off" or prior == 0 or not cands:
        return None, []
    net = cands[i]["net"]
    if net not in ("+1", "-1") or int(net) == prior:
        return None, []
    opp = f"{prior:+d}"
    contra = [cnd["source"] for j, cnd in enumerate(cands) if j != i and cnd["net"] == opp]
    agree = [cnd["source"] for j, cnd in enumerate(cands) if j != i and cnd["net"] == net]
    if not contra:
        return None, []
    if rule == "any":
        return FLIP_CONTESTED, contra
    return (FLIP_CONTESTED_MAJORITY if len(contra) > len(agree) else None), contra


# --------------------------------------------------------------------------------------------------------------
# 3. Connectome-side per-type numbers
# --------------------------------------------------------------------------------------------------------------
def load_raw_counts(c: cn.Connectome, log=print) -> np.ndarray | None:
    """Uncapped synapse counts aligned with c.W's stored entries (CSR order), or None if the feather is absent."""
    if not RAW_WEIGHTS.exists():
        log(f"[warn] raw weights not found at {RAW_WEIGHTS}; monoamine / unknown synapse counts unavailable (edges only)")
        return None
    t0 = time.time()
    w = pd.read_feather(RAW_WEIGHTS, columns=["body_pre", "body_post", "weight"])
    idx = pd.Series(np.arange(c.n), index=c.neurons.bodyId.to_numpy())
    pre = idx.reindex(w.body_pre.to_numpy()).to_numpy()
    post = idx.reindex(w.body_post.to_numpy()).to_numpy()
    m = ~np.isnan(pre) & ~np.isnan(post) & (w.weight.to_numpy() >= 1)
    R = sp.csr_matrix((w.weight.to_numpy()[m].astype(np.float32), (post[m].astype(np.int64), pre[m].astype(np.int64))),
                      shape=c.W.shape)
    R.sum_duplicates()
    W = c.W.tocsr()
    if R.nnz != W.nnz or not np.array_equal(R.indptr, W.indptr) or not np.array_equal(R.indices, W.indices):
        log("[warn] raw weights do not align with c.W (different node set or build); using |W| only")
        return None
    log(f"raw counts aligned: {R.nnz:,} edges, {int(R.data.sum()):,} synapses ({time.time() - t0:.1f}s)")
    return R.data.astype(np.float64)


def per_type_numbers(c: cn.Connectome, raw: np.ndarray | None) -> pd.DataFrame:
    n = c.neurons
    W = c.W.tocsr()
    A = abs(W)
    out_w = np.asarray(A.sum(axis=0)).ravel()
    in_w = np.asarray(A.sum(axis=1)).ravel()
    if raw is not None:
        R = sp.csr_matrix((raw, W.indices, W.indptr), shape=W.shape)
        out_r = np.asarray(R.sum(axis=0)).ravel()
        in_r = np.asarray(R.sum(axis=1)).ravel()
    else:
        out_r = np.full(c.n, np.nan); in_r = np.full(c.n, np.nan)
    df = pd.DataFrame({"type": n.type.fillna(""), "superclass": n.superclass.fillna(""), "nt": n.nt,
                       "out_W": out_w, "in_W": in_w, "out_raw": out_r, "in_raw": in_r})
    df = df[df.type != ""]
    g = df.groupby("type")
    t = g.agg(n_cells=("nt", "size"), out_syn_W=("out_W", "sum"), in_syn_W=("in_W", "sum"),
              out_syn_raw=("out_raw", "sum"), in_syn_raw=("in_raw", "sum"))
    t["superclass"] = g.superclass.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")

    def mode_share(s):
        vc = s.value_counts()
        return pd.Series({"malecns_consensus": vc.index[0], "malecns_consensus_share": vc.iloc[0] / len(s),
                          "n_unknown_cells": int((s == "unknown").sum())})
    t = t.join(g.nt.apply(mode_share).unstack())
    t["n_unknown_cells"] = t.n_unknown_cells.astype(int)
    return t


# --------------------------------------------------------------------------------------------------------------
# 4. Build the per-type tables
# --------------------------------------------------------------------------------------------------------------
def build_tables(c: cn.Connectome, raw: np.ndarray | None, log=print, flip_rule: str = FLIP_RULE_DEFAULT):
    if flip_rule not in FLIP_RULES:
        raise ValueError(f"flip_rule must be one of {FLIP_RULES}, got {flip_rule!r}")
    profs, maps = [], []
    for loader in (load_ozel, load_davis, load_central, load_kurmangaliyev):
        p, m = loader()
        profs.append(p); maps.append(m)
    prof = pd.concat(profs, ignore_index=True)
    maps = pd.concat(maps, ignore_index=True)
    prof = prof.join(marker_call(prof)).join(receptor_groups(prof)).assign(pid=np.arange(len(prof)))
    key = prof.set_index(["source", "source_key"]).pid
    maps["pid"] = key.reindex(list(zip(maps.source, maps.source_key))).to_numpy()
    missing = maps.pid.isna()
    if missing.any():
        log(f"[warn] {int(missing.sum())} map rows without an expression profile dropped: "
            f"{maps[missing][['source', 'source_key']].drop_duplicates().to_dict('records')[:5]}")
        maps = maps[~missing]
    maps["pid"] = maps.pid.astype(int)
    maps["tier_rank"] = maps.tier.map(TIER_RANK)
    maps["qc_rank"] = maps.qc.map(QC_RANK).fillna(1)
    maps["src_rank"] = maps.source.map(SOURCE_PRIORITY)
    maps = maps.sort_values(["malecns_type", "tier_rank", "qc_rank", "src_rank", "n_targets", "source_name"],
                            kind="mergesort").reset_index(drop=True)
    types = per_type_numbers(c, raw)
    nern = read_csv(DATA / "type_map_nern2025.csv")
    nern = nern[nern.tier.isin(["exact", "class"])].drop_duplicates("malecns_type").set_index("malecns_type")

    # pooled source rows: is the pool homogeneous in the model's own labels? (cell-weighted majority label share)
    lab = types.malecns_consensus.reindex(maps.malecns_type).to_numpy()
    ncell = types.n_cells.reindex(maps.malecns_type).fillna(0).to_numpy()
    maps["_lab"], maps["_ncell"] = lab, ncell
    pool = {}
    for (s, k), g in maps.groupby(["source", "source_key"]):
        gg = g[(g._lab != "unknown") & g._lab.notna()]
        if len(g) <= 1 or gg.empty:
            pool[(s, k)] = (False, "")
            continue
        w = gg.groupby("_lab")._ncell.sum().sort_values(ascending=False)
        share = w.iloc[0] / max(w.sum(), 1)
        pool[(s, k)] = (share < POOL_PURE_SHARE, ";".join(f"{l}:{int(n)}" for l, n in w.items()))
    maps["pool_mixed"] = [pool[(s, k)][0] for s, k in zip(maps.source, maps.source_key)]
    maps["pool_labels"] = [pool[(s, k)][1] for s, k in zip(maps.source, maps.source_key)]
    maps = maps.drop(columns=["_lab", "_ncell"])

    selector_maps = maps[maps.malecns_type.str.startswith("<")]
    type_maps = maps[~maps.malecns_type.str.startswith("<")]

    # ---- (a) transmitter table ------------------------------------------------------------------------------
    nt_rows = []
    for t, g in type_maps.groupby("malecns_type", sort=True):
        gnc = g[~g.circular]
        if gnc.empty:
            continue
        # per source: the best-ranked row; a pure pool beats a mixed pool of the same source
        gnc = gnc.assign(_mixed=gnc.pool_mixed.astype(int)).sort_values(
            ["tier_rank", "_mixed", "qc_rank", "src_rank", "n_targets", "source_name"], kind="mergesort")
        best_per_source = gnc.drop_duplicates("source")
        calls = {}
        for _, r in best_per_source.iterrows():
            p = prof.iloc[r.pid]
            calls[r.source] = (p.nt_call, r.tier, r.qc, bool(r.pool_mixed))
        called = {s: v for s, v in calls.items() if v[0] != "none"}
        best = best_per_source.iloc[0]
        pbest = prof.iloc[best.pid]
        pool_mixed, pool_labels = bool(best.pool_mixed), best.pool_labels
        if called:
            # majority over sources with a call (pure pools outrank mixed pools), tie -> the best-ranked source
            pure = {s: v for s, v in called.items() if not v[3]}
            use = pure if pure else called
            counts = pd.Series([v[0] for v in use.values()]).value_counts()
            top = counts[counts == counts.max()].index.tolist()
            if len(top) == 1:
                call = top[0]
            else:
                first = next(s for s in best_per_source.source if s in use and use[s][0] in top)
                call = use[first][0]
            agree = sorted(s for s, v in called.items() if v[0] == call)
            disagree = sorted(f"{s}:{v[0]}" for s, v in called.items() if v[0] != call)
            best_called = next(r for _, r in best_per_source.iterrows() if r.source in agree)
            tier = best_called.tier
            pool_mixed, pool_labels = bool(best_called.pool_mixed), best_called.pool_labels
            n_agree = len(agree)
            pcall = prof.iloc[best_called.pid]
            if disagree or pool_mixed or (pcall.marker_rank < WEAK_MARKER_RANK):
                confidence = "low"
            elif tier in ("exact", "alias") and n_agree >= 2:
                confidence = "high"
            elif tier in ("exact", "alias") or n_agree >= 2:
                confidence = "medium"
            else:
                confidence = "low"
            if best_called.qc != "pass" and n_agree == 1:
                confidence = "low"
            secondary, co = pcall.nt_secondary, pcall.co_monoamine
            marker_rank = round(float(pcall.marker_rank), 3)
            src_best, name_best, qc_best = best_called.source, best_called.source_name, best_called.qc
        else:
            call, agree, disagree, tier, confidence = "none", [], [], best.tier, "none"
            secondary, co, marker_rank = "", pbest.co_monoamine, np.nan
            src_best, name_best, qc_best = best.source, best.source_name, best.qc
        tn = types.loc[t] if t in types.index else None
        model = tn.malecns_consensus if tn is not None else ""
        nern_nt = nern.nern_nt[t] if t in nern.index else ""
        nern_val = nern.validated_nt[t] if t in nern.index and isinstance(nern.validated_nt[t], str) else ""
        agree_model = (call == model) if (call not in ("none",) and model not in ("unknown", "")) else None
        agree_nern = (call == nern_nt) if (call != "none" and nern_nt not in ("", "unclear")) else None
        model_nern = (model == nern_nt) if (model not in ("unknown", "") and nern_nt not in ("", "unclear")) else None
        if call == "none":
            flag = "transcriptome_none"
        elif pool_mixed and agree_model is False:
            flag = "pool_mixed_not_evaluable"
        elif model == "unknown":
            flag = "model_unknown_transcriptome_nern_agree" if agree_nern else \
                   ("model_unknown_transcriptome_vs_nern" if agree_nern is False else "model_unknown_transcriptome_only")
        elif agree_model and (agree_nern or agree_nern is None):
            flag = "all_agree" if agree_nern else "transcriptome_model_agree_nern_unclear"
        elif agree_model and agree_nern is False:
            flag = "transcriptome_model_agree_vs_nern"
        elif not agree_model and agree_nern:
            flag = "transcriptome_nern_agree_vs_model"
        elif not agree_model and model_nern:
            flag = "model_nern_agree_vs_transcriptome"
        elif not agree_model and agree_nern is None:
            flag = "transcriptome_vs_model_nern_unclear"
        else:
            flag = "all_differ"
        nt_rows.append({
            "malecns_type": t,
            "n_cells": int(tn.n_cells) if tn is not None else 0,
            "superclass": tn.superclass if tn is not None else "",
            "out_syn_W": int(tn.out_syn_W) if tn is not None else 0,
            "out_syn_raw": (int(tn.out_syn_raw) if tn is not None and not np.isnan(tn.out_syn_raw) else -1),
            "nt_transcriptome": call, "nt_secondary": secondary, "co_monoamine": co,
            "confidence": confidence, "tier": tier, "source": src_best, "source_name": name_best, "qc": qc_best,
            "marker_rank": marker_rank, "pool_mixed": pool_mixed, "pool_labels": pool_labels,
            "sources_agreeing": ";".join(agree), "sources_disagreeing": ";".join(disagree),
            "n_sources": len(calls),
            "malecns_consensus": model,
            "malecns_consensus_share": round(float(tn.malecns_consensus_share), 3) if tn is not None else np.nan,
            "n_unknown_cells": int(tn.n_unknown_cells) if tn is not None else 0,
            "nern2025_prediction": nern_nt, "nern2025_validated": nern_val,
            "agree_model": "" if agree_model is None else str(agree_model),
            "agree_nern": "" if agree_nern is None else str(agree_nern),
            "agreement": flag,
            "model_sign": int(cn.NT_SIGN.get(model, 0)), "transcriptome_sign": int(cn.NT_SIGN.get(call, 0)),
            "marker_scores": (pcall.marker_scores if called else pbest.marker_scores),
        })
    nt_table = pd.DataFrame(nt_rows)

    # ---- (b) receptor table ---------------------------------------------------------------------------------
    rec_rows = []

    def fast_groups(nt, nonmda=False):
        gs = [(s, f"{nt}|fast|{n}") for s, n, _ in RECEPTOR_GROUPS[nt]["fast"]]
        if nonmda and nt == "glutamate":
            gs = [(s, k) for s, k in gs if not k.endswith("|iGluR")] + [(IGLUR_NONMDA[0], f"glutamate|fast|{IGLUR_NONMDA[1]}")]
        return gs

    def slow_groups(nt):
        return [(s, f"{nt}|slow|{n}") for s, n, _ in RECEPTOR_GROUPS[nt]["slow"]]

    def receptor_rows_for(t, g, tn):
        cands = g.drop_duplicates("source")          # best row per source, in rank order (g is sorted)
        primary = cands.iloc[0]
        for nt in TRANSMITTERS:
            prior = int(cn.NT_SIGN[nt])
            per = []                                  # per candidate source: (map row, fast, fast_nonmda, slow)
            for _, a in cands.iterrows():
                pa = prof.iloc[a.pid]
                per.append((a, combine(fast_groups(nt), pa, prior), combine(fast_groups(nt, nonmda=True), pa, prior),
                            combine(slow_groups(nt), pa, 0)))
            i_c, how_c = select_profile([{"source": a.source, "net": f["net"]} for a, f, _, _ in per])
            i_n, how_n = select_profile([{"source": a.source, "net": fn["net"]} for a, _, fn, _ in per])
            best, fast, _, slow = per[i_c]
            fast_nn = per[i_n][2]
            # the class / abs variants use the profile chosen for the class variant (same group presence); the
            # nonmda variant may pick a different one (NMDA-only profiles are 'none' there). A contested 'none'
            # keeps the primary profile and the NT_SIGN prior as the fast sign (gain class none = factor 1).
            fast = dict(fast); fast_nn = dict(fast_nn)
            if how_c in NONE_FALLBACK:
                fast.update(net=how_c, sign=prior, gain=0, net_abs=how_c, sign_abs=prior, gain_abs=0)
            if how_n in NONE_FALLBACK:
                fast_nn.update(net=how_n, sign=prior, gain=0)
            on_sources = [a.source for a, f, _, _ in per if f["net"] != "none"]
            sel_c = how_c if how_c in ("primary", NONE_SINGLE) else f"{how_c}:{best.source}" if how_c == "group_on_override" \
                else f"{how_c}:{'+'.join(on_sources)}"
            sel_n = how_n if how_n in ("primary", NONE_SINGLE) else f"{how_n}:{per[i_n][0].source}" if how_n == "group_on_override" \
                else f"{how_n}:{'+'.join(a.source for a, _, fn, _ in per if fn['net'] != 'none')}"
            # round-3 symmetric rule: a flip contradicted by other profiled sources (per variant, each source's net
            # under that variant) falls back to NT_SIGN with gain class none, like a contested 'none'.
            contested = {}
            if how_c not in NONE_FALLBACK:
                lab, contra = contest_flip([{"source": a.source, "net": f["net"]} for a, f, _, _ in per], i_c, prior, flip_rule)
                if lab:
                    fast.update(net=lab, sign=prior, gain=0); contested["class"] = contra; sel_c = f"{lab}:{'+'.join(contra)}"
                lab, contra = contest_flip([{"source": a.source, "net": f["net_abs"]} for a, f, _, _ in per], i_c, prior, flip_rule)
                if lab:
                    fast.update(net_abs=lab, sign_abs=prior, gain_abs=0); contested["abs"] = contra
            if how_n not in NONE_FALLBACK:
                lab, contra = contest_flip([{"source": a.source, "net": fn["net"]} for a, _, fn, _ in per], i_n, prior, flip_rule)
                if lab:
                    fast_nn.update(net=lab, sign=prior, gain=0); contested["nonmda"] = contra; sel_n = f"{lab}:{'+'.join(contra)}"
            alt = [f"{a.source}({a.tier}):fast={fa['net']},abs={fa['net_abs']},nonmda={fn['net']},slow={sa['net']}"
                   for j, (a, fa, fn, sa) in enumerate(per) if j != i_c]
            rec_rows.append({
                "malecns_type": t, "transmitter": nt,
                "fast_sign": fast["sign"], "fast_gain_class": GAIN_CLASSES[fast["gain"]],
                "fast_pos_class": GAIN_CLASSES[fast["pos_cls"]], "fast_neg_class": GAIN_CLASSES[fast["neg_cls"]],
                "fast_net": fast["net"], "fast_pos_lead": fast["pos_lead"], "fast_neg_lead": fast["neg_lead"],
                "fast_sign_abs": fast["sign_abs"], "fast_gain_class_abs": GAIN_CLASSES[fast["gain_abs"]],
                "fast_net_abs": fast["net_abs"],
                "fast_sign_nonmda": fast_nn["sign"], "fast_gain_class_nonmda": GAIN_CLASSES[fast_nn["gain"]],
                "fast_pos_class_nonmda": GAIN_CLASSES[fast_nn["pos_cls"]], "fast_net_nonmda": fast_nn["net"],
                "fast_pos_lead_nonmda": fast_nn["pos_lead"],
                "slow_sign": slow["sign"], "slow_gain_class": GAIN_CLASSES[slow["gain"]],
                "slow_pos_class": GAIN_CLASSES[slow["pos_cls"]], "slow_neg_class": GAIN_CLASSES[slow["neg_cls"]],
                "slow_net": slow["net"], "slow_pos_lead": slow["pos_lead"], "slow_neg_lead": slow["neg_lead"],
                "slow_sign_abs": slow["sign_abs"], "slow_gain_class_abs": GAIN_CLASSES[slow["gain_abs"]],
                "slow_net_abs": slow["net_abs"],
                "tier": best.tier, "source": best.source, "source_name": best.source_name, "qc": best.qc,
                "pool_mixed": bool(best.pool_mixed),
                "fast_selection": sel_c, "fast_selection_nonmda": sel_n, "source_nonmda": per[i_n][0].source,
                "primary_source": primary.source, "primary_tier": primary.tier, "n_sources": len(per),
                "n_cells": int(tn.n_cells) if tn is not None else 0,
                "superclass": tn.superclass if tn is not None else "",
                "in_syn_W": int(tn.in_syn_W) if tn is not None else 0,
                "receptor_groups": fast["groups"] + (";" if fast["groups"] and slow["groups"] else "") + slow["groups"],
                "fast_pos_val": round(fast["pos_val"], 3), "fast_neg_val": round(fast["neg_val"], 3),
                "slow_pos_val": round(slow["pos_val"], 3), "slow_neg_val": round(slow["neg_val"], 3),
                "alt_sources": ";".join(alt),
                "flip_contested": ";".join(f"{v}:{'+'.join(s)}" for v, s in contested.items()),
            })

    for t, g in type_maps.groupby("malecns_type", sort=True):
        receptor_rows_for(t, g, types.loc[t] if t in types.index else None)
    for t, g in selector_maps.groupby("malecns_type", sort=True):
        receptor_rows_for(t, g, None)
    rec_table = pd.DataFrame(rec_rows)
    return nt_table, rec_table, types, prof, maps


# --------------------------------------------------------------------------------------------------------------
# 5. Edge lookup
# --------------------------------------------------------------------------------------------------------------
GAIN_CODE = {g: i for i, g in enumerate(GAIN_CLASSES)}
TIER_CODES = ["fallback", "pre_unknown", "nt_class", "class", "fuzzy", "alias", "exact"]
SOURCE_CODES = ["none", "davis2020", "ozel2021", "fca2022", "davie2018", "kurmangaliyev2020"]


def edge_lookup(c: cn.Connectome, receptors: pd.DataFrame | None = None, net_rule: str = "class",
                nt_class_fallback: bool = False) -> pd.DataFrame:
    """Per stored entry of c.W (order of c.W.tocsr().tocoo(), explicit zeros included): the presynaptic model
    transmitter, the postsynaptic type, and the fast / slow sign and gain class from receptors_by_type.csv.

    net_rule: 'class' = sign of the larger gain class (the plan's rule; ties -> NT_SIGN prior), 'abs' = sign of the
    larger summed expression in source units (ABS_RATIO-fold margin; ties -> prior), 'nonmda' = the class rule with
    the glutamate fast + group restricted to KaiR1D / GluRIA / GluRIB (NMDA excluded; slow columns as 'class').
    nt_class_fallback: for postsynaptic types without a profile, use the Davis 2020 ChAT / Gad1 / VGlut whole-class
    receptor baseline of the postsynaptic cell's own transmitter class (tier 'nt_class'); off by default.
    Fallback for everything else: fast sign = NT_SIGN of the presynaptic cell (c.neurons.sign), slow sign 0, tier
    'fallback' ('pre_unknown' when the presynaptic transmitter is unknown, i.e. nothing to look up).
    Columns: post, pre (int32 node indices), pre_nt, post_type (categorical), fast_sign, slow_sign (int8),
    fast_gain, slow_gain (int8: 0 none, 1 low, 2 mid, 3 high), tier, source (categorical)."""
    if receptors is None:
        receptors = read_csv(OUT_RECEPTORS)
    if net_rule not in ("class", "abs", "nonmda"):
        raise ValueError("net_rule must be 'class', 'abs' or 'nonmda'")
    sfx = {"class": "", "abs": "_abs", "nonmda": "_nonmda"}[net_rule]
    ssfx = "_abs" if net_rule == "abs" else ""
    W = c.W.tocsr().tocoo()
    post, pre = W.row.astype(np.int32), W.col.astype(np.int32)
    n = c.neurons
    nt_cats = TRANSMITTERS + ["unknown"]
    nt_code = pd.Categorical(n.nt, categories=nt_cats).codes.astype(np.int16)
    nt_code[nt_code < 0] = len(TRANSMITTERS)
    type_cat = pd.Categorical(n.type.fillna(""))
    type_code = type_cat.codes.astype(np.int32)
    type_index = {t: i for i, t in enumerate(type_cat.categories)}

    rt = receptors[~receptors.malecns_type.str.startswith("<")]
    L = np.full((len(type_cat.categories), len(nt_cats)), -1, dtype=np.int64)
    rt_idx = np.arange(len(rt))
    ti = np.array([type_index.get(t, -1) for t in rt.malecns_type])
    ni = np.array([TRANSMITTERS.index(x) for x in rt.transmitter])
    ok = ti >= 0
    L[ti[ok], ni[ok]] = rt_idx[ok]
    fs = rt["fast_sign" + sfx].to_numpy(np.int8); ss = rt["slow_sign" + ssfx].to_numpy(np.int8)
    fg = rt["fast_gain_class" + sfx].map(GAIN_CODE).to_numpy(np.int8)
    sg = rt["slow_gain_class" + ssfx].map(GAIN_CODE).to_numpy(np.int8)
    tier = rt.tier.map({t: i for i, t in enumerate(TIER_CODES)}).to_numpy(np.int8)
    src = rt.source.map({s: i for i, s in enumerate(SOURCE_CODES)}).fillna(0).to_numpy(np.int8)
    pm = rt.pool_mixed.to_numpy(bool) if "pool_mixed" in rt else np.zeros(len(rt), bool)

    pre_nt = nt_code[pre]
    idx = L[type_code[post], pre_nt]
    matched = idx >= 0
    i2 = np.where(matched, idx, 0)
    fast_sign = np.where(matched, fs[i2], n.sign.to_numpy().astype(np.int8)[pre]).astype(np.int8)
    slow_sign = np.where(matched, ss[i2], 0).astype(np.int8)
    fast_gain = np.where(matched, fg[i2], 0).astype(np.int8)
    slow_gain = np.where(matched, sg[i2], 0).astype(np.int8)
    tier_e = np.where(matched, tier[i2], np.where(pre_nt == len(TRANSMITTERS), 1, 0)).astype(np.int8)
    src_e = np.where(matched, src[i2], 0).astype(np.int8)
    pool_e = matched & pm[i2]

    if nt_class_fallback:
        sel = receptors[receptors.malecns_type.str.startswith("<nt=")]
        L2 = np.full((len(nt_cats), len(nt_cats)), -1, dtype=np.int64)
        for j, (_, r) in enumerate(sel.iterrows()):
            post_nt = r.malecns_type[len("<nt="):-1]
            if post_nt in TRANSMITTERS:
                L2[TRANSMITTERS.index(post_nt), TRANSMITTERS.index(r.transmitter)] = j
        idx2 = L2[nt_code[post], pre_nt]
        use = (~matched) & (idx2 >= 0)
        j2 = np.where(use, idx2, 0)
        fast_sign[use] = sel["fast_sign" + sfx].to_numpy(np.int8)[j2[use]]
        slow_sign[use] = sel["slow_sign" + ssfx].to_numpy(np.int8)[j2[use]]
        fast_gain[use] = sel["fast_gain_class" + sfx].map(GAIN_CODE).to_numpy(np.int8)[j2[use]]
        slow_gain[use] = sel["slow_gain_class" + ssfx].map(GAIN_CODE).to_numpy(np.int8)[j2[use]]
        tier_e[use] = TIER_CODES.index("nt_class")
        src_e[use] = SOURCE_CODES.index("davis2020")

    return pd.DataFrame({
        "post": post, "pre": pre,
        "pre_nt": pd.Categorical.from_codes(pre_nt, categories=nt_cats),
        "post_type": pd.Categorical.from_codes(type_code[post], categories=type_cat.categories),
        "fast_sign": fast_sign, "slow_sign": slow_sign, "fast_gain": fast_gain, "slow_gain": slow_gain,
        "tier": pd.Categorical.from_codes(tier_e, categories=TIER_CODES),
        "source": pd.Categorical.from_codes(src_e, categories=SOURCE_CODES),
        "pool_mixed": pool_e,
    })


def _tier_table(mask, tiers, wabs, raw, pool_mixed=None):
    rows = []
    tot_e, tot_w = int(mask.sum()), float(wabs[mask].sum())
    tot_r = float(raw[mask].sum()) if raw is not None else np.nan
    for t in ["exact", "alias", "fuzzy", "class", "nt_class", "fallback", "pre_unknown"]:
        m = mask & (tiers == t)
        e, w = int(m.sum()), float(wabs[m].sum())
        r = float(raw[m].sum()) if raw is not None else np.nan
        if e == 0 and t == "nt_class":
            continue
        rows.append({"tier": t, "edges": e, "edges_frac": e / max(tot_e, 1), "syn_W": w, "syn_W_frac": w / max(tot_w, 1),
                     "syn_raw": r, "syn_raw_frac": (r / max(tot_r, 1)) if raw is not None else np.nan})
    rows.append({"tier": "total", "edges": tot_e, "edges_frac": 1.0, "syn_W": tot_w, "syn_W_frac": 1.0,
                 "syn_raw": tot_r, "syn_raw_frac": 1.0 if raw is not None else np.nan})
    if pool_mixed is not None:
        m = mask & pool_mixed
        e, w = int(m.sum()), float(wabs[m].sum())
        r = float(raw[m].sum()) if raw is not None else np.nan
        rows.append({"tier": "(of which mixed-pool class prior)", "edges": e, "edges_frac": e / max(tot_e, 1), "syn_W": w,
                     "syn_W_frac": w / max(tot_w, 1), "syn_raw": r, "syn_raw_frac": (r / max(tot_r, 1)) if raw is not None else np.nan})
    return pd.DataFrame(rows)


def edge_stats(c: cn.Connectome, edges: pd.DataFrame, raw: np.ndarray | None = None, top: int = 25,
               receptors: pd.DataFrame | None = None, lead_col: str = "fast_pos_lead") -> dict:
    """Coverage by tier (overall, per post module, per pre module), glutamate flips (split by the lead receptor
    gene of the postsynaptic type when `receptors` is given) and monoamine slow signs. Works on integer codes."""
    W = c.W.tocsr()
    wabs = np.abs(W.data).astype(np.float64)
    tier_codes = edges.tier.cat.codes.to_numpy()
    tier_names = np.array(edges.tier.cat.categories)
    tiers = tier_names[tier_codes]                       # small vocabulary -> cheap fancy indexing
    mod = regions.labels(c)
    post_i, pre_i = edges.post.to_numpy(), edges.pre.to_numpy()
    mod_cat = pd.Categorical(mod, categories=list(regions.MODULES))
    mod_codes = mod_cat.codes
    post_mod, pre_mod = mod_codes[post_i], mod_codes[pre_i]
    all_mask = np.ones(len(edges), dtype=bool)
    pool_e = edges.pool_mixed.to_numpy() if "pool_mixed" in edges else None
    out = {"overall": _tier_table(all_mask, tiers, wabs, raw, pool_e)}
    for side, arr in (("post", post_mod), ("pre", pre_mod)):
        tabs = []
        for mi, m in enumerate(regions.MODULES):
            t = _tier_table(arr == mi, tiers, wabs, raw, pool_e); t.insert(0, "module", m); tabs.append(t)
        out[f"by_{side}_module"] = pd.concat(tabs, ignore_index=True)
    nt_codes = edges.pre_nt.cat.codes.to_numpy()
    nt_cats = list(edges.pre_nt.cat.categories)
    fast_sign, slow_sign = edges.fast_sign.to_numpy(), edges.slow_sign.to_numpy()
    matched_tier = np.isin(tiers, ["exact", "alias", "fuzzy", "class", "nt_class"])
    # glutamate flips
    glu = nt_codes == nt_cats.index("glutamate")
    flip = glu & (fast_sign == 1)
    zero = glu & (fast_sign == 0)
    matched = glu & matched_tier
    syn_r = raw if raw is not None else wabs
    out["glutamate"] = {
        "edges": int(glu.sum()), "syn_W": float(wabs[glu].sum()),
        "matched_edges": int(matched.sum()), "matched_syn_W": float(wabs[matched].sum()),
        "flip_edges": int(flip.sum()), "flip_syn_W": float(wabs[flip].sum()),
        "silenced_edges": int(zero.sum()), "silenced_syn_W": float(wabs[zero].sum()),
    }
    pt_codes = edges.post_type.cat.codes.to_numpy()
    pt_cats = np.array(edges.post_type.cat.categories)
    df = pd.DataFrame({"post_type": pt_cats[pt_codes[flip]], "syn": wabs[flip], "module": np.array(regions.MODULES)[post_mod[flip]]})
    top_types = df.groupby("post_type").syn.agg(["sum", "size"]).sort_values("sum", ascending=False).head(top)
    top_types["frac_of_flipped_syn"] = top_types["sum"] / max(float(wabs[flip].sum()), 1)
    top_types = top_types.rename(columns={"sum": "syn_W", "size": "edges"})
    if receptors is not None:
        rg = receptors[receptors.transmitter == "glutamate"].set_index("malecns_type")
        top_types["lead_gene"] = rg[lead_col].reindex(top_types.index).fillna("").to_numpy()
        top_types["GluCl_class"] = rg["fast_neg_class"].reindex(top_types.index).fillna("").to_numpy()
        top_types["source"] = rg["source"].reindex(top_types.index).fillna("").to_numpy()
        lead_all = rg[lead_col].reindex(df.post_type).fillna("").to_numpy()
        by_lead = pd.DataFrame({"lead_gene": lead_all, "syn": df.syn.to_numpy()}).groupby("lead_gene").syn.agg(["sum", "size"])
        by_lead["frac_of_flipped_syn"] = by_lead["sum"] / max(float(wabs[flip].sum()), 1)
        out["glutamate_flip_by_lead_gene"] = by_lead.rename(columns={"sum": "syn_W", "size": "edges"}).sort_values("syn_W", ascending=False)
    out["glutamate_flip_top_post_types"] = top_types
    out["glutamate_flip_by_post_module"] = df.groupby("module").syn.agg(["sum", "size"]).rename(columns={"sum": "syn_W", "size": "edges"})
    # monoamines: per (transmitter, post module) the slow-sign outcome of every edge
    slow_gain = edges.slow_gain.to_numpy()
    rows = []
    for nt in MONOAMINES:
        m = nt_codes == nt_cats.index(nt)
        for mi, mname in list(enumerate(regions.MODULES)) + [(-1, "ALL")]:
            sel = (post_mod == mi) if mi >= 0 else all_mask
            tot = m & sel
            mt = tot & matched_tier
            pos, neg = mt & (slow_sign == 1), mt & (slow_sign == -1)
            mixed = mt & (slow_sign == 0) & (slow_gain > 0)
            none = mt & (slow_sign == 0) & (slow_gain == 0)
            rows.append({"transmitter": nt, "post_module": mname, "edges": int(tot.sum()), "syn_raw": float(syn_r[tot].sum()),
                         "matched_syn": float(syn_r[mt].sum()), "pos_syn": float(syn_r[pos].sum()), "neg_syn": float(syn_r[neg].sum()),
                         "mixed_syn": float(syn_r[mixed].sum()), "no_receptor_syn": float(syn_r[none].sum()),
                         "unmatched_syn": float(syn_r[tot & ~matched_tier].sum()),
                         "pos_edges": int(pos.sum()), "neg_edges": int(neg.sum()), "mixed_edges": int(mixed.sum())})
    out["monoamine_slow"] = pd.DataFrame(rows)
    # classical transmitters onto matched targets that end up with no fast receptor
    rows = []
    for nt in CLASSICAL:
        m = (nt_codes == nt_cats.index(nt)) & matched_tier
        z = m & (fast_sign == 0)
        rows.append({"transmitter": nt, "matched_edges": int(m.sum()), "matched_syn_W": float(wabs[m].sum()),
                     "no_fast_receptor_edges": int(z.sum()), "no_fast_receptor_syn_W": float(wabs[z].sum())})
    out["classical_no_receptor"] = pd.DataFrame(rows)
    # silenced classical edges (profiled target, fast sign 0): by (pre transmitter, pre type, post type), by post
    # type, and the named pairs of the round-2 selection rule
    cls_codes = [nt_cats.index(x) for x in CLASSICAL]
    z = np.isin(nt_codes, cls_codes) & matched_tier & (fast_sign == 0)
    types_all = c.neurons.type.fillna("").to_numpy()
    zp = pd.DataFrame({"pre_nt": np.array(nt_cats)[nt_codes[z]], "pre_type": types_all[pre_i[z]],
                       "post_type": pt_cats[pt_codes[z]], "syn": wabs[z]})
    out["silenced_total"] = {"edges": int(z.sum()), "syn_W": float(wabs[z].sum())}
    out["silenced_pairs"] = (zp.groupby(["pre_nt", "pre_type", "post_type"]).syn.agg(["sum", "size"])
                             .rename(columns={"sum": "syn_W", "size": "edges"}).sort_values("syn_W", ascending=False).head(top))
    out["silenced_by_post_type"] = (zp.groupby(["pre_nt", "post_type"]).syn.agg(["sum", "size"])
                                    .rename(columns={"sum": "syn_W", "size": "edges"}).sort_values("syn_W", ascending=False).head(top))
    pre_t, post_t = zp.pre_type.to_numpy().astype(str), zp.post_type.to_numpy().astype(str)
    named = {
        "R7* -> Tm5a / Tm5b (histamine)": np.char.startswith(pre_t, "R7") & np.isin(post_t, ["Tm5a", "Tm5b"]),
        "R8* -> Mi1 (histamine)": np.char.startswith(pre_t, "R8") & (post_t == "Mi1"),
        "glutamate / GABA -> R7* / R8* (photoreceptor targets)": zp.pre_nt.isin(["glutamate", "gaba"]).to_numpy() & np.isin(post_t, cn.PHOTORECEPTOR_TYPES),
        "histamine -> any": (zp.pre_nt == "histamine").to_numpy(),
        "glutamate -> any": (zp.pre_nt == "glutamate").to_numpy(),
        "GABA -> any": (zp.pre_nt == "gaba").to_numpy(),
        "acetylcholine -> any": (zp.pre_nt == "acetylcholine").to_numpy(),
    }
    out["silenced_named"] = pd.DataFrame([{"pair": k, "edges": int(m.sum()), "syn_W": float(zp.syn.to_numpy()[m].sum())}
                                          for k, m in named.items()])
    return out


def compare_tables(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """Per (type, transmitter) differences between two receptors_by_type tables: net-call / sign / profile changes."""
    key = ["malecns_type", "transmitter"]
    cols = ["fast_net", "fast_net_abs", "fast_net_nonmda", "fast_sign", "fast_sign_abs", "fast_sign_nonmda",
            "slow_net", "tier", "source", "source_name", "fast_pos_lead", "fast_pos_lead_nonmda"]
    cols = [c for c in cols if c in new and c in old]
    m = new[key + cols].merge(old[key + cols], on=key, how="outer", suffixes=("_new", "_old"), indicator=True)
    both = m[m["_merge"] == "both"].copy()
    out = {"types_new": sorted(set(new.malecns_type) - set(old.malecns_type)),
           "types_removed": sorted(set(old.malecns_type) - set(new.malecns_type)),
           "rows_both": int(len(both))}
    changed = np.zeros(len(both), dtype=bool)
    trans = {}
    for c in cols:
        a, b = both[c + "_new"], both[c + "_old"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            d = ~np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
            d = pd.Series(d, index=both.index)
        else:
            d = a.fillna("").astype(str) != b.fillna("").astype(str)   # an empty lead is NaN when read back from CSV
        out["changed_" + c] = int(d.sum())
        if c.startswith("fast_net") or c == "slow_net":
            changed |= d.to_numpy()
            tr = both[d].groupby([c + "_old", c + "_new"]).size().sort_values(ascending=False)
            trans[c] = tr
    out["transitions"] = trans
    out["rows_changed"] = both[changed | (both.source_new.astype(str) != both.source_old.astype(str))].sort_values(key)
    return out


# --------------------------------------------------------------------------------------------------------------
# 6. Writers
# --------------------------------------------------------------------------------------------------------------
def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for k in cols:
            v = r[k]
            if k in fmt:
                cells.append(fmt[k](v))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(x):
    return "" if (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def write_csv_with_header(df: pd.DataFrame, path: Path, header: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write("# " + line + "\n")
        df.to_csv(f, index=False, lineterminator="\n")


def coverage_by_group(rec: pd.DataFrame, types: pd.DataFrame) -> pd.DataFrame:
    """Per-type best-tier coverage of the receptor table (types / cells / input synapses), by cell group."""
    best = rec[~rec.malecns_type.str.startswith("<")].drop_duplicates("malecns_type").set_index("malecns_type").tier
    t = types.copy()
    t["tier"] = best.reindex(t.index).fillna("unmatched")
    groups = {"optic lobe (ol_intrinsic + visual_projection + ol_sensory)": t.superclass.isin(OL_SUPERCLASSES),
              "visual_centrifugal": t.superclass == "visual_centrifugal",
              "central brain (cb_* + visual_centrifugal)": t.superclass.isin(CENTRAL_SUPERCLASSES),
              "descending + ascending": t.superclass.isin(["descending_neuron", "ascending_neuron"]),
              "VNC (vnc_* + sensory_ascending)": t.superclass.str.startswith("vnc_") | t.superclass.str.contains("ascending"),
              "all typed cells": np.ones(len(t), dtype=bool)}
    rows = []
    for gname, m in groups.items():
        sub = t[m]
        tot_c, tot_in, tot_out = sub.n_cells.sum(), sub.in_syn_W.sum(), sub.out_syn_W.sum()
        for tier in ["exact", "alias", "fuzzy", "class", "unmatched"]:
            s = sub[sub.tier == tier]
            rows.append({"group": gname, "tier": tier, "types": len(s), "cells": int(s.n_cells.sum()),
                         "cells_frac": s.n_cells.sum() / max(tot_c, 1), "in_syn_W": float(s.in_syn_W.sum()),
                         "in_syn_frac": s.in_syn_W.sum() / max(tot_in, 1), "out_syn_W": float(s.out_syn_W.sum()),
                         "out_syn_frac": s.out_syn_W.sum() / max(tot_out, 1)})
        rows.append({"group": gname, "tier": "total", "types": len(sub), "cells": int(tot_c), "cells_frac": 1.0,
                     "in_syn_W": float(tot_in), "in_syn_frac": 1.0, "out_syn_W": float(tot_out), "out_syn_frac": 1.0})
    return pd.DataFrame(rows)


def write_disagreements(nt_table: pd.DataFrame, types: pd.DataFrame, nern: pd.DataFrame, raw_available: bool):
    t = nt_table.copy()
    syn_col = "out_syn_raw" if raw_available else "out_syn_W"
    L = ["# Transmitter cross-check: transcriptome vs MaleCNS consensus vs Nern 2025",
         "",
         f"Generated by `scripts/build_receptor_table.py` from `flyverse/data/nt_by_type_transcriptome.csv` "
         f"({len(t):,} MaleCNS types with an expression profile) and `type_map_nern2025.csv`. `model sign` = "
         "`NT_SIGN` of the model's label (mode of `neurons.nt` over the type's cells, after the AL-LN override); "
         "`synapses` = raw (uncapped) output synapses of the type on the model's node set"
         + ("" if raw_available else " (raw weights unavailable: |W| sums, sign-0 types show 0)") +
         ". Rule of the plan (step 3): a sign / label change is proposed only where two of the three sources agree "
         "against the model, i.e. the transcriptome call and the Nern 2025 prediction agree and differ from the "
         "MaleCNS consensus. The transcriptome call rule and its thresholds are in `docs/audits/receptor_rules.md`.",
         ""]
    # summary of agreement flags
    vc = t.agreement.value_counts()
    L += ["## 1. Agreement summary", "",
          md_table(pd.DataFrame({"agreement": vc.index, "types": vc.values,
                                 "cells": [int(t[t.agreement == a].n_cells.sum()) for a in vc.index],
                                 "synapses": [int(t[t.agreement == a][syn_col].sum()) for a in vc.index]})), ""]
    called = t[(t.nt_transcriptome != "none")]
    both = called[called.agree_model != ""]
    pure = both[~both.pool_mixed.astype(bool)]
    nern_any = t.nern2025_prediction.fillna("").astype(str) != ""
    nern_call = nern_any & (t.nern2025_prediction.fillna("").astype(str) != "unclear")
    comp = t[t.agree_nern != ""]
    L += [f"* Types with a transcriptome call: {len(called):,} of {len(t):,}; comparable with the model (model label not "
          f"unknown), all rows: {len(both):,} / agree {int((both.agree_model == 'True').sum()):,} "
          f"({100 * (both.agree_model == 'True').mean():.1f}%) / disagree {int((both.agree_model == 'False').sum()):,} "
          f"(this count includes {int(both.pool_mixed.astype(bool).sum()):,} mixed-pool rows, "
          f"{int(((both.agree_model == 'True') & both.pool_mixed.astype(bool)).sum()):,} of which happen to agree).",
          f"* **Outside mixed pools (symmetric denominators): comparable {len(pure):,} / agree "
          f"{int((pure.agree_model == 'True').sum()):,} ({100 * (pure.agree_model == 'True').mean():.1f}%) / disagree "
          f"{int((pure.agree_model == 'False').sum()):,}.**",
          f"* Nern 2025 label present for {int(nern_any.sum()):,} of {len(t):,} types ({int(nern_call.sum()):,} not "
          f"'unclear'); transcriptome call comparable with Nern: {len(comp):,} / agree {int((comp.agree_nern == 'True').sum()):,} "
          f"/ disagree {int((comp.agree_nern == 'False').sum()):,}.",
          f"* Confidence of the calls: {called.confidence.value_counts().to_dict()}.",
          f"* Sources: {t.source.value_counts().to_dict()} (best-ranked agreeing source per type); "
          f"{int((t.n_sources >= 2).sum()):,} types have >= 2 sources.", ""]

    def verdict(r):
        if r.agreement == "pool_mixed_not_evaluable":
            return f"not evaluable: the source cluster pools types with mixed model labels ({r.pool_labels})"
        if r.agreement == "transcriptome_nern_agree_vs_model":
            return "PROPOSE: relabel to " + r.nt_transcriptome + (" (sign flip)" if r.model_sign != r.transcriptome_sign else " (label only; sign unchanged under NT_SIGN, matters for the receptor model)")
        if r.agreement == "model_nern_agree_vs_transcriptome":
            return "keep model label (Nern agrees with the model)"
        if r.agreement == "transcriptome_vs_model_nern_unclear":
            return "unresolved (1 vs 1; Nern unclear or outside its scope)"
        if r.agreement == "transcriptome_model_agree_vs_nern":
            return "keep model label (transcriptome agrees with the model)"
        if r.agreement == "all_differ":
            return "unresolved (three different labels)"
        return r.agreement

    dis = t[t.agreement.isin(["transcriptome_nern_agree_vs_model", "model_nern_agree_vs_transcriptome",
                              "transcriptome_vs_model_nern_unclear", "transcriptome_model_agree_vs_nern", "all_differ"])].copy()
    dis["verdict"] = dis.apply(verdict, axis=1)
    dis = dis.sort_values(syn_col, ascending=False)
    cols = ["malecns_type", "n_cells", syn_col, "model_sign", "malecns_consensus", "malecns_consensus_share",
            "nt_transcriptome", "nt_secondary", "confidence", "tier", "source", "sources_agreeing", "sources_disagreeing",
            "nern2025_prediction", "nern2025_validated", "verdict", "marker_scores"]
    L += ["## 2. Disagreements among the three sources (transcriptome call present, per-type or pure-pool profile)", "",
          f"{len(dis):,} types; proposals (two of three against the model): "
          f"{int((dis.agreement == 'transcriptome_nern_agree_vs_model').sum())}; of those with a sign flip under "
          f"`NT_SIGN`: {int(((dis.agreement == 'transcriptome_nern_agree_vs_model') & (dis.model_sign != dis.transcriptome_sign)).sum())}. "
          "`marker_scores` = the marker levels of the deciding profile in the source's units (TPM for davis2020; expm1 of the "
          "log-mean for the 10x sources; `*` = on by the source's rule), so a 'below the floor' value is visible (e.g. l-LNv "
          "Davis SerT 5.4 TPM < 10, not 0). Sources vote equally whatever their tier (a class-pool profile counts like a "
          "per-type one; only mixed pools are demoted), so R7 / R8 types are called acetylcholine where the pooled "
          "Kurmangaliyev 2020 'R7.8' and Davie 2018 'Photoreceptors' clusters (ChAT / VAChT above Hdc in expm1 units, "
          "both markers on) outvote the per-type Davis 2020 R7 / R8 drivers (histamine): a co-expression that Nern 2025 "
          "validated as histamine+acetylcholine for R8y / R8p, not a sign question -- the model keeps histamine.",
          "", md_table(dis[cols].rename(columns={syn_col: "synapses"})), ""]
    pm = t[t.agreement == "pool_mixed_not_evaluable"].copy()
    pm["verdict"] = pm.apply(verdict, axis=1)
    pm = pm.sort_values(syn_col, ascending=False)
    L += ["## 2b. Not evaluable: the transcriptome profile is a pooled cluster whose member types carry mixed model labels", "",
          f"{len(pm):,} types (Davie 2018 'MBON' / 'Clock' / 'Photoreceptors', FCA 'antennal lobe projection neuron', ...). "
          "The pool's marker call describes the pool, not the type; these rows are kept in the receptor table as class-tier "
          "priors and are excluded from the disagreement count.", "",
          md_table(pm[["malecns_type", "n_cells", syn_col, "malecns_consensus", "nt_transcriptome", "nt_secondary", "tier",
                       "source", "source_name", "pool_labels", "nern2025_prediction"]].rename(columns={syn_col: "synapses"})), ""]

    # Nern vs model disagreements without a transcriptome profile
    nn = nern[nern.tier.isin(["exact", "class"])].copy()
    nn = nn[(nn.nern_nt.notna()) & (nn.nern_nt != "unclear") & (nn.malecns_nt != "unknown") & (nn.nern_nt != nn.malecns_nt)]
    nn = nn[~nn.malecns_type.isin(set(t.malecns_type))]
    L += ["## 3. Nern 2025 vs MaleCNS consensus, types without a transcriptome profile (1 vs 1, unresolved)", "",
          md_table(nn[["malecns_type", "n_cells_malecns", "out_syn_raw", "malecns_nt", "nern_nt", "fw_nt", "validated_nt"]]
                   .sort_values("out_syn_raw", ascending=False).rename(columns={"n_cells_malecns": "cells", "out_syn_raw": "synapses"})), ""]

    # unknown-NT types that get a label
    unk = t[(t.malecns_consensus == "unknown") | (t.n_unknown_cells > 0)].copy()
    unk = unk[(unk.nt_transcriptome != "none") | (unk.nern2025_prediction.fillna("").isin(TRANSMITTERS))]
    unk["usable"] = np.where(unk.pool_mixed, "no (mixed pool)",
                             np.where(unk.confidence == "low", "weak (low confidence)", "yes"))
    unk = unk.sort_values(syn_col, ascending=False)
    L += ["## 4. Unknown-NT types (model label `unknown`, or types with unknown-NT cells) that get a label", "",
          "`n_unknown_cells` = cells of the type whose model label is `unknown` (sign 0). A label from the transcriptome "
          "alone is a single-source rescue; where Nern 2025 agrees it is two-source. `usable` = no for mixed pools, weak "
          "for low-confidence calls (suboptimal driver, weak marker, disagreeing sources). Nern-only rescues of the audit's "
          "watch list (Mi19 serotonin, TmY14 glutamate) are in `docs/audits/receptor_sources_nern2025.md`.", "",
          md_table(unk[["malecns_type", "n_cells", "n_unknown_cells", syn_col, "malecns_consensus", "nt_transcriptome",
                        "confidence", "marker_rank", "sources_agreeing", "nern2025_prediction", "agreement", "usable"]]
                   .rename(columns={syn_col: "synapses"})), ""]
    nern_unknown = nern[nern.tier.isin(["exact", "class"]) & (nern.malecns_nt == "unknown") & nern.nern_nt.isin(TRANSMITTERS)]
    nern_unknown = nern_unknown[~nern_unknown.malecns_type.isin(set(unk.malecns_type))]
    if len(nern_unknown):
        L += ["Unknown-NT types labelled by Nern 2025 only (no transcriptome profile):", "",
              md_table(nern_unknown[["malecns_type", "n_cells_malecns", "out_syn_raw", "nern_nt", "fw_nt", "validated_nt"]]
                       .rename(columns={"n_cells_malecns": "cells", "out_syn_raw": "synapses"})), ""]
    # marker-silent types
    none = t[t.nt_transcriptome == "none"].sort_values(syn_col, ascending=False)
    L += ["## 5. Types whose expression profile carries no transmitter marker (`transcriptome_none`)", "",
          "No proposal: photoreceptors and T1 are known marker-silent in scRNA-seq / nuclear RNA-seq (the histamine "
          "pathway of R1-R8 is not captured by Hdc in these data; T1 expresses none of the markers, Davis 2020). "
          "The model keeps its label.", "",
          md_table(none[["malecns_type", "n_cells", syn_col, "malecns_consensus", "tier", "source", "nern2025_prediction",
                         "co_monoamine"]].rename(columns={syn_col: "synapses"})), ""]
    OUT_DISAGREE.write_text("\n".join(L), encoding="utf-8")


def write_rules(stats_class: dict, stats_abs: dict, cov: pd.DataFrame, rec: pd.DataFrame, nt_table: pd.DataFrame,
                raw_available: bool, n_edges: int, stats_ntclass: dict | None, stats_nonmda: dict | None = None,
                prev: tuple | None = None, flip_rule: str = FLIP_RULE_DEFAULT):
    rec_t = rec[~rec.malecns_type.str.startswith("<")]
    L = ["# Receptor rules: transmitter -> postsynaptic response class, and the coverage of the edge lookup",
         "",
         "Generated by `scripts/build_receptor_table.py` (docs/NT_INTEGRATION.md steps 3-5). The rules below are the "
         "ones written into the headers of `flyverse/data/receptors_by_type.csv` and "
         "`flyverse/data/nt_by_type_transcriptome.csv`; the numbers are what `edge_lookup(c)` and `edge_stats` return "
         "on the cached MaleCNS v1.0 model.",
         "",
         "## 1. Sources and units",
         "",
         "| source | data | unit used for the level | 'on' criterion | mapped types (from the per-source type maps) |",
         "|---|---|---|---|---|",
         "| davis2020 | Davis et al. 2020 eLife, driver-sorted bulk nuclear RNA-seq (TAPIN / INTACT), adult heads, mixed sex | TPM (cell-type mean over QC-pass samples; QC-fail drivers flagged `suboptimal_only`) | authors' mixture-model P(on) >= 0.5 AND TPM >= 10 (TPM >= 10 alone where P(on) is undefined) | exact / alias / fuzzy / class per `type_map_davis2020.csv` |",
         "| ozel2021 | Özel et al. 2021 Nature, adult optic-lobe scRNA-seq, female | per-cluster mean of ln(1 + UMI/total x 1e4) | authors' mixture-model P(on) >= 0.5 | exact / fuzzy / class per `type_map_ozel2021.csv` |",
         "| fca2022 | Fly Cell Atlas 2022 head 10x (single nucleus), 5 d, mixed sex (pooled rows) | per-cluster mean log1p(cp10k) | fraction of nuclei with >= 1 UMI >= 0.2 | exact / alias / fuzzy / class per `type_map_central.csv` |",
         "| davie2018 | Davie et al. 2018 Cell whole-brain 10x, 0-50 d pooled, mixed sex | per-cluster mean log1p(cp10k) | fraction of cells with >= 1 UMI >= 0.2 | exact / fuzzy / class per `type_map_central.csv` |",
         f"| kurmangaliyev2020 | Kurmangaliyev et al. 2020 Neuron, PUPAL optic-lobe scRNA-seq (24-96 h APF; female by roX check), whole cell 10x v3 | per-cluster mean log1p(cp10k) of the {KURM_TIME} APF (pharate adult) rows, `{KURM_TIME_FALLBACK}` (72-96 h pooled) when < {KURM_MIN_CELLS} cells at {KURM_TIME} | fraction of cells with >= 1 UMI >= {CENTRAL_FRAC_ON} | exact / alias / fuzzy / class per `type_map_kurmangaliyev2020.csv` |",
         "| nern2025 | Nern et al. 2025 Nature, optic-lobe EM transmitter classifier (same male volume) | transmitter label only | - | exact / class per `type_map_nern2025.csv` |",
         "",
         "Expression is not conductance: levels are used only (i) to rank marker genes within a profile and (ii) to bin "
         "receptor groups into quantile classes within a source. Values are never compared across sources. "
         "Source classes for the selection rule (section 3): whole-cell = davis2020 (bulk nuclear), ozel2021, "
         "kurmangaliyev2020 (whole-cell 10x); single-nucleus = fca2022, davie2018 (10x nuclei; low-abundance receptors "
         "drop out, so a 'none' is weaker evidence). The Kurmangaliyev atlas is pupal: its 96 h APF profiles correlate "
         "with the Özel adult profile of the same type at median Pearson r 0.90 over the 54 genes (72-96 h pool 0.89, "
         "all timepoints 0.80; 42 types in both, computed by the round-2 rebuild), and it ranks after the two adult "
         "whole-cell sources at equal tier.",
         "",
         "## 2. Transmitter identity from synthesis / transport genes (`nt_by_type_transcriptome.csv`)",
         "",
         "* Marker groups: acetylcholine = ChAT or VAChT on (score = max of the two levels); GABA = Gad1 on (VGAT is "
         "not used: over the 69 neuronal QC-pass Davis 2020 rows it is 1.7-586 TPM, off only in the photoreceptors "
         "(1.7-14.5 TPM, P(on) = 0) and 60-586 TPM with P(on) = 1 in every other row, T1 172.6 / L1 304.6 / Mi1 236.1, "
         "so it does not discriminate among non-photoreceptor neurons); glutamate = "
         "VGlut; histamine = Hdc; dopamine = ple AND DAT on (score = min); octopamine = Tdc2 AND Tbh on (min); "
         "serotonin = Trh AND SerT on (min). The 'AND' for the monoamines is deliberate: Tbh, Tdc2 and DAT are "
         "detected at low level in many non-aminergic clusters (Tbh in T4 / T5, Dm11, Poxn; Tdc2 0.6-35 TPM over the "
         "neuronal QC-pass Davis rows with mean P(on) 0.88 -- 'on' by this script's own rule, P(on) >= 0.5 AND "
         "TPM >= 10, in 32 % of them), the enzyme alone is not evidence. Levels of the log-mean sources are linearised "
         "(expm1) before comparison.",
         "* Primary call = the present classical marker group (ACh / GABA / Glu / His) with the highest level in the "
         "source's own units; other present classical groups within 0.5x of it are listed in `nt_secondary` "
         "(co-expression or cluster impurity); present monoamine groups are listed in `co_monoamine` (Mi15 ACh + DA). "
         "If no classical group is present the primary is the highest monoamine group; otherwise `none`. "
         "`marker_rank` = quantile of the primary marker's level among the source's neuronal profiles in which that "
         "marker is present (a weak marker, rank < 0.25, gives confidence `low`).",
         "* Pooled source rows (a cluster or driver mapped to several MaleCNS types): `pool_mixed` = the member "
         "types' cell-weighted majority model label holds < 90 % of the pool's cells (Davie 'MBON', 'Clock', FCA "
         "'antennal lobe projection neuron', Davis 'Lat', ...). A mixed pool's call describes the pool, not the type: "
         "such rows never outvote a per-type or pure-pool source, get confidence `low`, and a disagreement with the "
         "model is flagged `pool_mixed_not_evaluable` instead of counted.",
         "* Per MaleCNS type: one call per source (the best-ranked mapping row of that source; rows selected by the "
         "model's own transmitter class -- Davis ChAT / Gad1 / VGlut drivers, Davie / FCA 'Dopaminergic', "
         "'Serotonergic', 'Octopaminergic' classes -- are circular and excluded here). `nt_transcriptome` = majority "
         "over the sources with a call (tie -> the best-ranked source). `confidence`: high = mapping tier exact / "
         "alias and >= 2 sources agree with none disagreeing; medium = exact / alias with one source, or >= 2 sources "
         "agreeing at fuzzy / class tier; low = single fuzzy / class source, a `suboptimal_only` Davis driver as the "
         "only source, a mixed pool, a weak marker, or any disagreement between sources.",
         "* `malecns_consensus` = mode of the model's `nt` over the type's cells (cache); `nern2025_prediction` = "
         "Nern 2025 Sup. Table 1. `agreement` combines the three; a change is proposed only for "
         "`transcriptome_nern_agree_vs_model` (docs/audits/receptor_nt_disagreements.md).",
         "",
         "## 3. Receptor groups -> response classes (`receptors_by_type.csv`)",
         "",
         "| transmitter | fast + | fast - | slow + | slow - |",
         "|---|---|---|---|---|",
         "| acetylcholine | nAChR (alpha1-7, beta1-3) | - | mAChR-A (Gq) | mAChR-B (Gi) |",
         "| GABA | - | Rdl, Lcch3, Grd (GABA-A) | - | GABA-B-R1/2/3 |",
         "| glutamate | KaiR1D, GluRIA, GluRIB, Nmdar1 + Nmdar2 (iGluR) | GluClalpha (GluCl) | - | mGluR |",
         "| histamine | - | HisCl1, ort | - | - |",
         "| dopamine | - | - | Dop1R1, Dop1R2, DopEcR | Dop2R |",
         "| octopamine | - | - | Oamb, Octbeta1R, Octbeta2R, Octbeta3R | Octalpha2R |",
         "| serotonin | - | - | 5-HT2A, 5-HT2B, 5-HT7 | 5-HT1A, 5-HT1B |",
         "",
         "mAChR-C (unknown coupling; absent from two sources) is not used. NMDA receptors are obligate Nmdar1 + Nmdar2 "
         "heteromers, so the iGluR group is present only if a non-NMDA member is on or both Nmdar genes are on.",
         "",
         "* Group presence = any member gene on (with the NMDA exception). Group level = sum of the levels of the "
         "members that are on, in the source's units linearised (TPM as is; expm1 of the log-mean for the 10x "
         "sources, so that summing over members is a linear operation). `*_pos_lead` / `*_neg_lead` name the member "
         "gene with the largest level.",
         "* Because Nmdar2 is 5.3-1,073 TPM over the neuronal QC-pass Davis 2020 rows -- off (P(on) = 0) only in "
         "R1-6 / R7 / R8, 102-1,073 TPM with P(on) = 1 in every other neuronal row (Nmdar1 on in about half) -- the "
         "glutamate rows carry a third variant `fast_*_nonmda`: the class rule with the fast + group restricted to "
         "KaiR1D / GluRIA / GluRIB. `edge_lookup(..., net_rule='nonmda')` uses it. KaiR1D is CG3822 (FBgn0038837) in "
         "every source since the round-2 Davis rebuild (round 1 had CG8916 under that name).",
         "* Gain class (none / low / mid / high): `none` if the group is absent; otherwise the tertile of the group "
         "level among the neuronal profiles of the same source in which the group is present (tertiles are computed "
         "per source and per group, so a class says 'this target expresses the group at a low / mid / high level "
         "for that source', not an absolute conductance). Sources with fewer than three such profiles give `mid`.",
         "* Per (type, transmitter, fast|slow): `*_pos_class` / `*_neg_class` are the classes of the + and - groups; "
         "`*_net` = sign of the larger class (`mixed` if equal and both present, `none` if both absent); `*_sign` is "
         "the numeric sign (+1 / -1 / 0 = none; for `mixed` the prior of `NT_SIGN` is kept for the fast class -- "
         "glutamate -1 -- and 0 for the slow class), `*_gain_class` the class of the winning group.",
         f"* The same with the suffix `_abs` under the absolute rule: the larger side must exceed the other "
         f"{ABS_RATIO:g}-fold in summed source-unit level, else `mixed`. This rule is provided because the class rule "
         "compares ranks, not amounts: GluCl is expressed at ~10x the iGluR level in every profiled optic type "
         "(Davis 2020), which the tertile classes erase.",
         "* One candidate profile per source and type: the mapping row with the best tier (exact > alias > fuzzy > "
         "class), then QC pass before suboptimal, then the row pooling the fewest MaleCNS types, then the source name "
         "(alphabetical); the sources are ranked davis2020 > ozel2021 > kurmangaliyev2020 > fca2022 > davie2018 at "
         "equal tier / QC. The best-ranked candidate is the `primary` profile.",
         "* **Profile selection per (type, transmitter) -- round-2 rule** (`select_profile`; "
         "docs/audits/receptor_verification.md, verify:implement and the critic): the primary profile decides the fast "
         "sign, EXCEPT when it has no fast receptor group for the transmitter (`none`, i.e. the edge would be silenced). "
         "A silencing stands only if every source that profiles the type agrees. (i) If the primary is a single-nucleus "
         "profile (fca2022 / davie2018) and a whole-cell source (davis2020 / ozel2021 / kurmangaliyev2020) has the group "
         "on, that whole-cell profile is used for the row (`fast_selection` = `group_on_override:<source>`; `tier` / "
         "`source` / `source_name` then name it). (ii) Any other disagreement about `none` (a whole-cell `none` against "
         "any `on`, or a single-nucleus `none` against a single-nucleus `on`) keeps the primary profile but falls back "
         "to `NT_SIGN` for the fast sign: `fast_net` = `none_contested`, `fast_sign` = the prior, `fast_gain_class` = "
         "`none` (gain factor 1 under `sign+gain`). (iii) A `none` carried by fewer than "
         f"{MIN_SOURCES_TO_SILENCE} sources (every source agrees, but only one profiles the type) also falls back to "
         "`NT_SIGN`: `fast_net` = `none_single_source` (an anatomical synapse is silenced only on >= 2 concurring "
         "profiles; one profile's dropout or threshold call -- e.g. Özel cluster 163 Pm1/Pm5/Pm6 with Rdl P(on) 0.37 at "
         "mean log 2.4 -- is not enough). (iv) The `_nonmda` variant is selected separately (a profile whose "
         "only iGluR members are Nmdar1 + Nmdar2 is `none` there; `fast_selection_nonmda`, `source_nonmda`). The slow "
         "columns come from the profile selected for the class variant. `primary_source` / `primary_tier` record what "
         "the round-1 rule (primary always) would have used; `alt_sources` lists the other candidates' net calls "
         "(`fast=` class variant, `abs=`, `nonmda=`, `slow=`).",
         "* **Contested flips -- round-3 rule** (`contest_flip`; docs/audits/receptor_verification.md, round-2 critic "
         "follow-up 1), the symmetric case of (ii): a FLIP -- the selected profile's fast net is the opposite sign of "
         "the `NT_SIGN` prior (a `+1` on glutamate; a `-1` on a +1 transmitter, which no fast group produces today) -- "
         "stands only if no other source profiling the type contradicts it. The rule is applied per variant with each "
         "source's net under that variant (class nets for the class columns, `_abs` nets for the abs columns, `_nonmda` "
         "nets for the nonmda columns); a source at the prior's own sign contradicts the flip, a source at the flip's "
         "sign agrees with it, `mixed` and `none` do neither. Two variants of the rule exist (`--flip-rule`): `any` "
         "(**the default and the rule of this table**: >= 1 contradicting source, `fast_net*` = `flip_contested`) and "
         "`majority` (fall back only when the contradicting sources outnumber the agreeing ones, `fast_net*` = "
         "`flip_contested_majority`); `off` reproduces the round-2 table. A contested flip keeps the selected profile "
         "(`tier` / `source` / `alt_sources` unchanged) but takes `fast_sign*` = the prior and `fast_gain_class*` = "
         "`none` (factor 1), exactly like `none_contested`; the column `flip_contested` names the contradicting sources "
         "per variant (`class:...;abs:...;nonmda:...`, empty when no variant is contested), `fast_selection` / "
         "`fast_selection_nonmda` carry `flip_contested*:<sources>` for the class / nonmda variants. "
         f"This table was built with `--flip-rule {flip_rule}`.",
         "* Rows `<nt=acetylcholine|gaba|glutamate>` are the Davis 2020 ChAT / Gad1 / VGlut protein-trap drivers: the "
         "receptor baseline of a whole transmitter class, usable as a fallback for unprofiled targets "
         "(`edge_lookup(..., nt_class_fallback=True)`), off by default.",
         "",
         "## 4. Edge lookup (`edge_lookup(c)`)",
         "",
         "For every stored entry of `c.W` (post, pre; explicit zeros of sign-0 presynaptic cells included): pre "
         "transmitter = `c.neurons.nt` of the presynaptic cell; post type = `c.neurons.type`; if (post type, pre "
         "transmitter) has a row in `receptors_by_type.csv` the fast / slow sign and gain classes come from it (tier "
         "= the row's mapping tier); otherwise the fast sign is the present rule (`NT_SIGN` of the presynaptic cell), "
         "the slow sign 0 and the tier `fallback` (`pre_unknown` when the presynaptic transmitter is unknown -- there "
         "is nothing to look up).",
         "",
         f"Denominators: {n_edges:,} stored edges; synapses as |W| (sign-0 edges count 0) and, where the raw weights "
         f"were available, uncapped raw counts (sign-0 edges count their real synapses). Raw counts "
         f"{'were' if raw_available else 'were NOT'} available for this build.",
         "",
         "### 4a. Coverage by tier, all edges (post-type match; class rule; no NT-class fallback)",
         "",
         md_table(stats_class["overall"], {"edges_frac": pct, "syn_W_frac": pct, "syn_raw_frac": pct}), ""]
    if stats_ntclass is not None:
        L += ["With `nt_class_fallback=True` (Davis ChAT / Gad1 / VGlut class baselines for unprofiled cholinergic / "
              "GABAergic / glutamatergic targets):", "",
              md_table(stats_ntclass["overall"], {"edges_frac": pct, "syn_W_frac": pct, "syn_raw_frac": pct}), ""]
    L += ["### 4b. Coverage by tier per postsynaptic module (`flyverse/regions.py`), optic first", "",
          md_table(stats_class["by_post_module"], {"edges_frac": pct, "syn_W_frac": pct, "syn_raw_frac": pct}), "",
          "### 4c. Coverage by tier per presynaptic module", "",
          md_table(stats_class["by_pre_module"], {"edges_frac": pct, "syn_W_frac": pct, "syn_raw_frac": pct}), "",
          "### 4d. Per-type coverage of the receptor table (best tier per type; input synapses are the receptor-relevant side)", "",
          md_table(cov, {"cells_frac": pct, "in_syn_frac": pct, "out_syn_frac": pct}), ""]
    variants = [("class rule (`net`)", stats_class), ("absolute rule (`net_abs`)", stats_abs)]
    if stats_nonmda is not None:
        variants.append(("class rule without NMDA (`net_nonmda`)", stats_nonmda))
    for name, st in variants:
        g = st["glutamate"]
        L += [f"### 4e. Glutamatergic edges under the {name}", "",
              f"* glutamatergic edges {g['edges']:,} ({g['syn_W']:,.0f} synapses |W|); onto a profiled target "
              f"{g['matched_edges']:,} ({g['matched_syn_W']:,.0f}); **flip -1 -> +1 (iGluR targets): {g['flip_edges']:,} "
              f"edges, {g['flip_syn_W']:,.0f} synapses = {100 * g['flip_syn_W'] / max(g['syn_W'], 1):.1f}% of all "
              f"glutamatergic synapses, {100 * g['flip_syn_W'] / max(g['matched_syn_W'], 1):.1f}% of those onto profiled "
              f"targets**; silenced (no fast glutamate receptor on the target): {g['silenced_edges']:,} edges, "
              f"{g['silenced_syn_W']:,.0f} synapses.", "",
              "By postsynaptic module:", "", md_table(st["glutamate_flip_by_post_module"].reset_index()), ""]
        if "glutamate_flip_by_lead_gene" in st:
            L += ["By the lead (highest-level) fast + receptor gene of the postsynaptic type:", "",
                  md_table(st["glutamate_flip_by_lead_gene"].reset_index(), {"frac_of_flipped_syn": pct}), ""]
        L += ["Top postsynaptic types:", "", md_table(st["glutamate_flip_top_post_types"].reset_index(), {"frac_of_flipped_syn": pct}), ""]
    rec_t = rec[~rec.malecns_type.str.startswith("<")]
    da = rec_t[(rec_t.transmitter == "dopamine") & (rec_t.slow_pos_class != "none")]
    L += ["### 4f. Monoamine synapses that acquire a slow sign (raw synapses where available, else |W| = 0 by construction)", "",
          "`matched_syn` = synapses onto a profiled target; `pos` / `neg` = slow sign +1 / -1; `mixed` = both receptor "
          "groups present at the same class (class rule: sign 0) or within the 2-fold margin (absolute rule); "
          "`no_receptor` = profiled target without any receptor for the transmitter; `unmatched` = target without a profile "
          "(slow sign 0, tier fallback).", "",
          "Class rule (`slow_net`):", "", md_table(stats_class["monoamine_slow"]), "",
          "Absolute rule (`slow_net_abs`):", "", md_table(stats_abs["monoamine_slow"]), "",
          f"Lead gene of the dopamine slow + group over the {len(da):,} profiled (type) rows where it is present: "
          f"{da.slow_pos_lead.value_counts().to_dict()}. DopEcR is an ecdysone-responsive GPCR with dopamine affinity in "
          "the micromolar range and is on in 67 of the 79 QC-pass Davis 2020 rows including glia and muscle (93 % of the "
          "neuronal QC-pass rows; lowest L3 97.8 TPM with P(on) 0); if it is dropped from the group the dopamine "
          "slow sign rests on Dop1R1 / Dop1R2 vs Dop2R only (rerun with `RECEPTOR_GROUPS` edited; not done here).", "",
          "### 4g. Classical transmitters onto profiled targets that express no fast receptor for them (edge silenced by the lookup)", "",
          "Under the round-2 selection rule a silencing needs every source profiling the target to agree (section 3).", "",
          md_table(stats_class["classical_no_receptor"]), "",
          f"Silenced classical edges in total: {stats_class['silenced_total']['edges']:,} entries / "
          f"{stats_class['silenced_total']['syn_W']:,.0f} |W| synapses. The named pairs of the verification record "
          "(R7* / R8* = every presynaptic type whose name starts with R7 / R8, R7R8_unclear counted under R7):", "",
          md_table(stats_class["silenced_named"]), "",
          "Top silenced (presynaptic transmitter, presynaptic type, postsynaptic type) triples:", "",
          md_table(stats_class["silenced_pairs"].reset_index()), "",
          "Top silenced postsynaptic types per presynaptic transmitter:", "",
          md_table(stats_class["silenced_by_post_type"].reset_index()), ""]
    # net-call tallies per transmitter over the type rows (the 'claim 14' recount)
    L += ["### 4h. Net-call tallies per transmitter over the type rows of `receptors_by_type.csv`", "",
          f"{len(rec_t):,} rows = {rec_t.malecns_type.nunique():,} types x {len(TRANSMITTERS)} transmitters "
          "(the `<nt=...>` selector rows excluded). Counts of rows per net call:", ""]
    tally = []
    for nt in TRANSMITTERS:
        r = rec_t[rec_t.transmitter == nt]
        for col in ["fast_net", "fast_net_abs", "fast_net_nonmda", "slow_net", "slow_net_abs"]:
            if nt in MONOAMINES and col.startswith("fast"):
                continue
            if nt != "glutamate" and col == "fast_net_nonmda":
                continue
            vc = r[col].value_counts()
            tally.append({"transmitter": nt, "column": col, "+1": int(vc.get("+1", 0)), "-1": int(vc.get("-1", 0)),
                          "mixed": int(vc.get("mixed", 0)), "none": int(vc.get("none", 0)),
                          "none_contested": int(vc.get(NONE_CONTESTED, 0)), "none_single_source": int(vc.get(NONE_SINGLE, 0)),
                          "flip_contested": int(vc.get(FLIP_CONTESTED, 0)) + int(vc.get(FLIP_CONTESTED_MAJORITY, 0))})
    L += [md_table(pd.DataFrame(tally)), "",
          "Source x tier of the profile deciding the fast sign (class variant); one row per (type, transmitter):", "",
          md_table(rec_t.groupby(["source", "tier"]).size().rename("rows").reset_index()), "",
          "Selection outcomes over all type rows: " +
          ", ".join(f"{k} {v:,}" for k, v in rec_t.fast_selection.str.split(":").str[0].value_counts().items()) +
          "; nonmda: " + ", ".join(f"{k} {v:,}" for k, v in rec_t.fast_selection_nonmda.str.split(":").str[0].value_counts().items()) + ".", ""]
    L += ["## 5. Caveats", "",
          "* Sex and age: davis2020 / davie2018 / fca2022 are mixed-sex, ozel2021 female; MaleCNS is male. FCA male-only "
          "rows exist in `expression_central.csv` (differences < 0.15 log units for the receptor genes at cluster level) "
          "and were not used.",
          "* The 'on' thresholds (P(on) 0.5; TPM 10; fraction 0.2) are choices, stated here so they can be swept; the "
          "single-nucleus sources (fca2022, davie2018) under-detect low-abundance receptors, so their `none` classes "
          "are less trustworthy than davis2020 / ozel2021 ones.",
          "* Class-tier profiles (pooled clusters such as Özel 'PR', 'T4-5a/b', Davis 'T4', FCA 'Kenyon cell') give "
          "every member type the same profile.",
          "* The class rule can flip glutamatergic edges wherever the target's iGluR tertile is above its GluCl "
          "tertile, regardless of the absolute levels; score both rules (step 6 of the plan) before adopting either.",
          "* No receptor table covers the VNC, the descending neurons, the lateral horn, the CX columnar system "
          "beyond Davis's Delta7 / EPG / PEN / PFN drivers, or the gustatory second-order cells: those edges stay "
          "under `NT_SIGN` (tier `fallback`).",
          "* kurmangaliyev2020 is a pupal atlas (no post-eclosion timepoint); it decides a row only where no adult "
          "source offers a better tier (T4a-d / T5a-d exact where Davis has the pooled T4 / T5 drivers at fuzzy tier). "
          "Its T1 / R1-R6 clusters are marker-silent like the adult ones."]
    if prev is not None:
        d, st_prev = prev
        L += ["", "## 6. Changes against the previous table (`--previous`)", "",
              f"Compared with the previous `receptors_by_type.csv` ({d['rows_both']:,} (type, transmitter) rows in both; "
              f"types added {len(d['types_new'])}: {d['types_new'][:20]}; removed {len(d['types_removed'])}: "
              f"{d['types_removed'][:20]}). Rows whose value changed: " +
              ", ".join(f"{k[8:]} {v:,}" for k, v in d.items() if k.startswith("changed_")) + ".", ""]
        for col, tr in d["transitions"].items():
            if len(tr):
                L += [f"`{col}` transitions (old -> new: rows):", "",
                      md_table(tr.rename("rows").reset_index()), ""]
        gp, gc = st_prev["glutamate"], stats_class["glutamate"]
        L += ["Edge-level effect (class rule, |W| synapses): glutamate flips -1 -> +1 "
              f"{gp['flip_edges']:,} / {gp['flip_syn_W']:,.0f} -> {gc['flip_edges']:,} / {gc['flip_syn_W']:,.0f}; "
              f"silenced classical edges {st_prev['silenced_total']['edges']:,} / {st_prev['silenced_total']['syn_W']:,.0f} -> "
              f"{stats_class['silenced_total']['edges']:,} / {stats_class['silenced_total']['syn_W']:,.0f}.", "",
              "Named silenced pairs, previous table:", "", md_table(st_prev["silenced_named"]), "",
              "Named silenced pairs, this table:", "", md_table(stats_class["silenced_named"]), "",
              "Silenced pairs of the previous table (top):", "", md_table(st_prev["silenced_pairs"].reset_index()), ""]
        rows = d["rows_changed"]
        show = [c for c in ["malecns_type", "transmitter", "fast_net_old", "fast_net_new", "fast_net_abs_old", "fast_net_abs_new",
                            "fast_net_nonmda_old", "fast_net_nonmda_new", "slow_net_old", "slow_net_new", "source_old", "source_new",
                            "tier_old", "tier_new", "fast_pos_lead_old", "fast_pos_lead_new"] if c in rows]
        L += [f"All {len(rows):,} rows whose net call, slow call or deciding source changed:", "", md_table(rows[show]), ""]
    OUT_RULES.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="build nt_by_type_transcriptome.csv / receptors_by_type.csv and the two audit docs")
    ap.add_argument("--previous", default=None, help="a previous receptors_by_type.csv: adds a change section to receptor_rules.md")
    ap.add_argument("--flip-rule", default=FLIP_RULE_DEFAULT, choices=list(FLIP_RULES),
                    help="round-3 contested-flip rule: 'any' (>= 1 contradicting source, default), 'majority' (contradicting > agreeing), "
                         "'off' (round-2 behaviour)")
    args = ap.parse_args()
    t0 = time.time()
    log = print
    c = cn.load(verbose=False)
    log(f"connectome: {c.n:,} neurons, {c.W.nnz:,} edges ({time.time() - t0:.1f}s)")
    raw = load_raw_counts(c, log)
    nt_table, rec_table, types, prof, maps = build_tables(c, raw, log, flip_rule=args.flip_rule)
    fc = rec_table[~rec_table.malecns_type.astype(str).str.startswith("<")]
    for col in ("fast_net", "fast_net_abs", "fast_net_nonmda"):
        n = int(fc[col].isin(FLIP_FALLBACK).sum())
        log(f"flip rule {args.flip_rule}: {col} contested rows {n}: "
            + ", ".join(f"{t}/{nt}" for t, nt in fc.loc[fc[col].isin(FLIP_FALLBACK), ['malecns_type', 'transmitter']].itertuples(index=False)))
    log(f"tables built: {len(nt_table):,} types in the NT table, {len(rec_table):,} receptor rows "
        f"({rec_table.malecns_type.nunique():,} types) ({time.time() - t0:.1f}s)")

    nt_header = [
        "nt_by_type_transcriptome.csv -- transmitter identity per MaleCNS v1.0 type from synthesis / transport genes, built by",
        "scripts/build_receptor_table.py from the per-source tables in flyverse/data/ (Özel 2021 GSE142787; Davis 2020 GSE116969;",
        "Davie 2018 GSE107451; Fly Cell Atlas 2022 head; Nern 2025 Sup. Table 1) and cache/neurons.parquet. Rules and thresholds:",
        "docs/audits/receptor_rules.md section 2; disagreements: docs/audits/receptor_nt_disagreements.md.",
        "nt_transcriptome: acetylcholine = ChAT|VAChT on; gaba = Gad1 on (VGAT unused: 60-586 TPM with P(on) = 1 in every non-photoreceptor",
        "neuronal QC-pass Davis row); glutamate = VGlut;",
        "histamine = Hdc; dopamine = ple&DAT; octopamine = Tdc2&Tbh; serotonin = Trh&SerT ('on' = source mixture-model P(on) >= 0.5",
        "[ozel2021], P(on) >= 0.5 & TPM >= 10 [davis2020], fraction of cells >= 0.2 [davie2018 / fca2022 / kurmangaliyev2020 96 h APF]);",
        "primary = highest classical",
        "marker level (expm1-linearised for the 10x sources), monoamines only when no classical marker is on; majority over sources",
        "(circular NT-class selections excluded; mixed pools never outvote per-type rows); none = no marker on.",
        "confidence: high = exact/alias tier and >= 2 sources agree; medium = exact/alias single source or >= 2 fuzzy/class; low = single",
        "fuzzy/class, suboptimal driver, mixed pool (pool_mixed: member types' majority model label < 90 % of pool cells; pool_labels",
        "lists them), weak marker (marker_rank < 0.25 = quantile of the marker level among the source's on-profiles) or disagreement.",
        "malecns_consensus = mode of the model nt over the type's cells (share given); nern2025_prediction = Nern 2025; agreement = joint",
        "flag; out_syn_raw = uncapped raw output synapses on the node set (-1 if the raw weights were unavailable); out_syn_W = |W|",
        "(sign-0 cells contribute 0).",
    ]
    write_csv_with_header(nt_table, OUT_NT, nt_header)
    rec_header = [
        "receptors_by_type.csv -- postsynaptic response class per (MaleCNS v1.0 type, transmitter), built by",
        "scripts/build_receptor_table.py from flyverse/data/expression_{ozel2021,davis2020,central,kurmangaliyev2020}.csv and the",
        "matching type maps (Özel 2021 GSE142787; Davis 2020 GSE116969 CC BY 4.0; Davie 2018 GSE107451; Fly Cell Atlas 2022 head;",
        "Kurmangaliyev 2020 GSE156455, pupal 96 h APF). Full rules and coverage: docs/audits/receptor_rules.md.",
        "Receptor groups: acetylcholine fast + nAChR alpha1-7/beta1-3, slow + mAChR-A (Gq) / slow - mAChR-B (Gi); gaba fast - Rdl/Lcch3/Grd,",
        "slow - GABA-B-R1/2/3; glutamate fast - GluClalpha, fast + KaiR1D/GluRIA/GluRIB/Nmdar1+Nmdar2, slow - mGluR; histamine fast -",
        "HisCl1/ort; dopamine slow + Dop1R1/Dop1R2/DopEcR, slow - Dop2R; octopamine slow + Oamb/Octbeta1-3R, slow - Octalpha2R;",
        "serotonin slow + 5-HT2A/2B/7, slow - 5-HT1A/1B. Group present = any member gene on (NMDA needs both Nmdar1 and Nmdar2);",
        "group level = sum of the on-members' levels (TPM; expm1 of the log-mean for the 10x sources); gain class = none if absent,",
        "else tertile (low/mid/high) of the group level among the source's neuronal profiles where the group is present (quantiles",
        "within source, NOT linear expression). *_pos_lead / *_neg_lead = the on-member with the largest level.",
        "fast_sign / slow_sign: +1 / -1 / 0 = none. *_pos_class / *_neg_class: classes of the + and - groups; *_net: sign of the",
        "larger class, 'mixed' if equal (then *_sign keeps the NT_SIGN prior for fast, 0 for slow), 'none' if both absent;",
        "*_gain_class: class of the winning group. *_abs columns: the same under the absolute rule (larger summed level, 2-fold",
        "margin, else mixed). fast_*_nonmda: the class rule with the glutamate fast + group restricted to KaiR1D/GluRIA/GluRIB",
        "(Nmdar2 is 102-1,073 TPM with P(on) = 1 in every non-photoreceptor neuronal QC-pass Davis row); identical to the plain",
        "columns for the other transmitters. KaiR1D = CG3822 in every source.",
        "tier / source / source_name / qc: the profile deciding the row. Candidates: the best row per source (best tier, QC pass",
        "first, then the row pooling the fewest types, then name), ranked davis2020 > ozel2021 > kurmangaliyev2020 > fca2022 >",
        "davie2018 at equal tier / QC; the best-ranked is the primary (primary_source / primary_tier). Selection rule (round 2):",
        "the primary decides unless its fast group for the transmitter is 'none' and another source has it on -- a single-",
        "nucleus (fca2022 / davie2018) 'none' is replaced by the best whole-cell (davis2020 / ozel2021 / kurmangaliyev2020)",
        "profile with the group on (fast_selection = group_on_override:<source>); any other such disagreement keeps the primary",
        "but falls back to NT_SIGN: fast_net = none_contested, fast_sign = the prior, fast_gain_class = none (factor 1); a",
        "'none' carried by a single source falls back the same way (fast_net = none_single_source). A silencing (fast_sign 0)",
        "therefore needs every source profiling the type to agree and at least two of them. fast_selection_nonmda /",
        "source_nonmda: the same for the nonmda variant (selected separately). n_sources: candidate sources for the type.",
        f"Contested flips (round 3, --flip-rule {args.flip_rule}): a flip (the selected profile's fast net at the opposite sign of the",
        "NT_SIGN prior, e.g. glutamate +1) contradicted by >= 1 other source whose net under the same variant (class / abs / nonmda)",
        "is the prior's sign falls back to NT_SIGN the same way: fast_net* = flip_contested (rule 'any', the default) or",
        "flip_contested_majority (rule 'majority': only when the contradicting sources outnumber those agreeing with the flip);",
        "'mixed' / 'none' sources neither contradict nor agree; flip_contested lists the contradicting sources per variant.",
        "pool_mixed: the profile is a pooled cluster whose member types carry mixed model transmitter labels (a class prior, not",
        "a per-type measurement); alt_sources: the other candidates' net calls (fast= class, abs=, nonmda=, slow=). Rows '<nt=...>'",
        "are the Davis 2020 ChAT / Gad1 / VGlut whole-class baselines (tier class), used only with edge_lookup(nt_class_fallback=True).",
    ]
    write_csv_with_header(rec_table, OUT_RECEPTORS, rec_header)
    log(f"wrote {OUT_NT} and {OUT_RECEPTORS}")

    nern = read_csv(DATA / "type_map_nern2025.csv")
    write_disagreements(nt_table, types, nern, raw is not None)
    log(f"wrote {OUT_DISAGREE}")

    edges_class = edge_lookup(c, rec_table, net_rule="class")
    st_class = edge_stats(c, edges_class, raw, receptors=rec_table)
    edges_abs = edge_lookup(c, rec_table, net_rule="abs")
    st_abs = edge_stats(c, edges_abs, raw, receptors=rec_table)
    edges_nn = edge_lookup(c, rec_table, net_rule="nonmda")
    st_nn = edge_stats(c, edges_nn, raw, receptors=rec_table, lead_col="fast_pos_lead_nonmda")
    edges_nt = edge_lookup(c, rec_table, net_rule="class", nt_class_fallback=True)
    st_nt = edge_stats(c, edges_nt, raw)
    cov = coverage_by_group(rec_table, types)
    prev = None
    if args.previous:
        old = read_csv(Path(args.previous))
        st_prev = edge_stats(c, edge_lookup(c, old, net_rule="class"), raw, receptors=old)
        prev = (compare_tables(rec_table, old), st_prev)
        log(f"previous table {args.previous}: {len(old):,} rows; changed net calls: "
            + ", ".join(f"{k[8:]} {v}" for k, v in prev[0].items() if k.startswith("changed_")))
    write_rules(st_class, st_abs, cov, rec_table, nt_table, raw is not None, len(edges_class), st_nt, st_nn, prev,
                flip_rule=args.flip_rule)
    log(f"wrote {OUT_RULES} ({time.time() - t0:.1f}s)")

    # ---- headline numbers -----------------------------------------------------------------------------------
    print("\n=== HEADLINE ===")
    ov = st_class["overall"].set_index("tier")
    matched_tiers = ["exact", "alias", "fuzzy", "class"]
    print("whole CNS edges by tier (edges / |W| syn / raw syn fractions):")
    for t in matched_tiers + ["fallback", "pre_unknown"]:
        if t in ov.index:
            r = ov.loc[t]
            print(f"  {t:12s} {r.edges:>12,.0f} {100 * r.edges_frac:5.1f}%  {r.syn_W:>14,.0f} {100 * r.syn_W_frac:5.1f}%  "
                  f"{r.syn_raw:>14,.0f} {100 * r.syn_raw_frac:5.1f}%")
    bp = st_class["by_post_module"]
    for m in ("optic", "visual_projection", "central", "mushroom_body"):
        sub = bp[bp.module == m].set_index("tier")
        mt = sub.loc[[t for t in matched_tiers if t in sub.index]]
        print(f"post module {m}: matched any tier = {100 * mt.edges_frac.sum():.1f}% of edges, "
              f"{100 * mt.syn_W_frac.sum():.1f}% of |W| synapses, {100 * mt.syn_raw_frac.sum():.1f}% raw "
              f"(exact {100 * (sub.loc['exact'].syn_W_frac if 'exact' in sub.index else 0):.1f}%)")
    print("per-type coverage of the receptor table (input synapses):")
    for g in cov.group.unique():
        s = cov[(cov.group == g) & cov.tier.isin(matched_tiers)]
        print(f"  {g}: types {int(s.types.sum())}, cells {100 * s.cells_frac.sum():.1f}%, in-syn {100 * s.in_syn_frac.sum():.1f}%, "
              f"out-syn {100 * s.out_syn_frac.sum():.1f}%")
    for name, st in (("class", st_class), ("abs", st_abs), ("nonmda", st_nn)):
        g = st["glutamate"]
        print(f"glutamate flips ({name} rule): {g['flip_edges']:,} edges / {g['flip_syn_W']:,.0f} syn = "
              f"{100 * g['flip_syn_W'] / g['syn_W']:.1f}% of glutamatergic synapses "
              f"({100 * g['flip_syn_W'] / max(g['matched_syn_W'], 1):.1f}% of those onto profiled targets); "
              f"silenced {g['silenced_edges']:,} / {g['silenced_syn_W']:,.0f}")
        print("  top post types: " + ", ".join(f"{t} {r.syn_W:,.0f} [{r.lead_gene}]" for t, r in st["glutamate_flip_top_post_types"].head(10).iterrows()))
        if "glutamate_flip_by_lead_gene" in st:
            print("  by lead gene: " + ", ".join(f"{g_} {r.syn_W:,.0f}" for g_, r in st["glutamate_flip_by_lead_gene"].iterrows()))
    for name, st in (("class", st_class), ("nonmda", st_nn)) + ((("previous", prev[1]),) if prev else ()):
        z = st["silenced_total"]
        print(f"silenced classical edges ({name}): {z['edges']:,} entries / {z['syn_W']:,.0f} syn; "
              + "; ".join(f"{r.pair} {r.edges:,} / {r.syn_W:,.0f}" for _, r in st["silenced_named"].iterrows()))
    rt_ = rec_table[~rec_table.malecns_type.str.startswith("<")]
    print("selection outcomes: " + ", ".join(f"{k} {v:,}" for k, v in rt_.fast_selection.str.split(":").str[0].value_counts().items())
          + "; nonmda: " + ", ".join(f"{k} {v:,}" for k, v in rt_.fast_selection_nonmda.str.split(":").str[0].value_counts().items()))
    pm = int((nt_table.agreement == "pool_mixed_not_evaluable").sum())
    print(f"pool-mixed (not evaluable) rows: {pm}")
    for name, st in (("class", st_class), ("abs", st_abs)):
        ms = st["monoamine_slow"]
        for _, r in ms[ms.post_module == "ALL"].iterrows():
            print(f"monoamine {r.transmitter} ({name} rule): {r.syn_raw:,.0f} synapses; onto profiled targets {r.matched_syn:,.0f} "
                  f"({100 * r.matched_syn / max(r.syn_raw, 1):.1f}%): slow +1 {r.pos_syn:,.0f}, -1 {r.neg_syn:,.0f}, "
                  f"mixed {r.mixed_syn:,.0f}, no receptor {r.no_receptor_syn:,.0f}")
        bym = ms[ms.post_module != "ALL"].groupby("post_module").agg(pos=("pos_syn", "sum"), neg=("neg_syn", "sum"),
                                                                     mixed=("mixed_syn", "sum"), all_syn=("syn_raw", "sum"))
        print(f"  by post module ({name}): " + ", ".join(f"{m} +{r.pos:,.0f}/-{r.neg:,.0f}/~{r.mixed:,.0f} of {r.all_syn:,.0f}"
                                                         for m, r in bym.iterrows()))
    dis = nt_table[nt_table.agreement.isin(["transcriptome_nern_agree_vs_model", "model_nern_agree_vs_transcriptome",
                                            "transcriptome_vs_model_nern_unclear", "transcriptome_model_agree_vs_nern", "all_differ"])]
    prop = nt_table[nt_table.agreement == "transcriptome_nern_agree_vs_model"]
    print(f"disagreements: {len(dis)} types ({nt_table.agreement.value_counts().to_dict()}); proposals: {len(prop)} "
          f"{prop.malecns_type.tolist()}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
