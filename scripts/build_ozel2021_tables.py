"""Build the Özel et al. 2021 (GEO GSE142787, adult optic lobe scRNA-seq) -> MaleCNS tables.

Inputs (git-ignored, downloaded by hand; see docs/audits/receptor_sources_ozel2021.md):
    data/external/ozel2021/41586_2020_2879_MOESM4_ESM.xlsx          Supplementary Table 1: cluster annotations
    data/external/ozel2021/GSE142787_Log_normalized_average_expression.xlsx  sheet Adult_average_expression
    data/external/ozel2021/GSE142787_Mixture_modeling.xlsx           sheet Adult_MM_final
    cache/neurons.parquet, cache/W_post_pre.npz                      the model's connectome cache

Outputs (redistributable derived tables):
    flyverse/data/type_map_ozel2021.csv       source cluster -> MaleCNS type, with tier + evidence
    flyverse/data/expression_ozel2021.csv     per adult cluster, log-normalised mean expression of 54 NT genes
    flyverse/data/expression_ozel2021_mm.csv  per adult cluster, mixture-model P(expressed) of the same genes
    docs/audits/receptor_sources_ozel2021.md  provenance (URL, size, SHA-256), licence, coverage tables

    PYTHONIOENCODING=utf-8 python scripts/build_ozel2021_tables.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "ozel2021"
DATA = ROOT / "flyverse" / "data"
AUDIT = ROOT / "docs" / "audits" / "receptor_sources_ozel2021.md"

# ----------------------------------------------------------------------------------------------
# Gene list (model names) -> symbol in the source's gene build (BDGP6.88, FlyBase 2019 symbols).
# ----------------------------------------------------------------------------------------------
GENE_GROUPS = {
    "synthesis_transport": ["ChAT", "VAChT", "Gad1", "VGAT", "VGlut", "Hdc", "ple", "DAT", "Tdc2", "Tbh", "Trh", "SerT"],
    "fast_receptor": ["nAChRalpha1", "nAChRalpha2", "nAChRalpha3", "nAChRalpha4", "nAChRalpha5", "nAChRalpha6",
                      "nAChRalpha7", "nAChRbeta1", "nAChRbeta2", "nAChRbeta3", "Rdl", "Lcch3", "Grd", "GluClalpha",
                      "KaiR1D", "GluRIA", "GluRIB", "Nmdar1", "Nmdar2", "HisCl1", "ort"],
    "slow_receptor": ["mAChR-A", "mAChR-B", "mAChR-C", "GABA-B-R1", "GABA-B-R2", "GABA-B-R3", "mGluR",
                      "Dop1R1", "Dop1R2", "Dop2R", "DopEcR", "Oamb", "Octbeta1R", "Octbeta2R", "Octbeta3R",
                      "Octalpha2R", "5-HT1A", "5-HT1B", "5-HT2A", "5-HT2B", "5-HT7"],
}
GENES = [g for gs in GENE_GROUPS.values() for g in gs]
# FlyBase symbol changes between the source build (BDGP6.88, 2019) and the names used in the plan.
SOURCE_SYMBOL = {"ChAT": "Cha", "KaiR1D": "CG3822", "Octalpha2R": "CG18208", "mAChR-C": "CG7918"}

# ----------------------------------------------------------------------------------------------
# Cluster -> MaleCNS type map. One entry per adult cluster that we can evidence; everything else is
# reported as unmatched. Tiers: exact (author's annotation is a MaleCNS type name, unstarred),
# alias (name differs but a documented renaming links them; none needed for this source),
# fuzzy (author-flagged less confident '*' annotation, or a 'likely' assignment, or a subtype letter
# whose correspondence to the MaleCNS subtype letter is not established), class (the source cluster
# is a union of several MaleCNS types, or a subdivision of one MaleCNS type).
# ----------------------------------------------------------------------------------------------
ST1 = "Supplementary Table 1 (41586_2020_2879_MOESM4_ESM.xlsx)"
CLEAR = "ST1: 'Clear match to {t} transcriptome (Extended Data Fig. 3)' (Pearson correlation with the driver-line bulk transcriptome of Konstantinides 2018 / Davis 2020, best-vs-second gap >= 0.05)"

PHOTORECEPTORS = ["R1-R6", "R7y", "R7p", "R7d", "R7_unclear", "R8y", "R8p", "R8d", "R8_unclear", "R7R8_unclear"]
T4T5_AB = ["T4a", "T4b", "T5a", "T5b"]
T4T5_CD = ["T4c", "T4d", "T5c", "T5d"]

MAP: list[tuple[int, str, list[str], str, str]] = [
    # (cluster, source_name, [malecns types], tier, evidence)
    (1, "T1", ["T1"], "exact", CLEAR.format(t="T1") + "; clusters 1+2 merged (OOBE 0.14)"),
    (3, "PR", PHOTORECEPTORS, "class", "ST1: 'Clear match to photoreceptors transcriptome; contains photoreceptor cells R1 to R8' -> all MaleCNS ol_sensory photoreceptor types; not resolved to R1-6 / R7 / R8 / pale / yellow"),
    (6, "T2", ["T2"], "exact", CLEAR.format(t="T2")),
    (7, "T3", ["T3"], "exact", CLEAR.format(t="T3")),
    (8, "Lawf2", ["Lawf2"], "exact", CLEAR.format(t="Lawf2")),
    (9, "Dm4", ["Dm4"], "exact", CLEAR.format(t="Dm4")),
    (12, "Dm12", ["Dm12"], "exact", CLEAR.format(t="Dm12")),
    (14, "Dm1", ["Dm1"], "exact", CLEAR.format(t="Dm1")),
    (15, "Dm10", ["Dm10"], "exact", CLEAR.format(t="Dm10")),
    (19, "Dm8", ["Dm8a", "Dm8b"], "class", CLEAR.format(t="Dm8") + "; MaleCNS splits Dm8 into Dm8a (FlyWire yDm8) and Dm8b (pDm8) by connectivity (Nern 2025 ED Fig. 5); the cluster contains both"),
    (27, "LC14", ["LC14a-1", "LC14a-2", "LC14b"], "class", "ST1: only cluster expressing atonal (mixture modelling), the marker of LC14 / LC14b; 'the cluster might be impure, maybe because of subtypes LC14/LC14b' -> MaleCNS LC14a-1, LC14a-2, LC14b"),
    (31, "Mi15", ["Mi15"], "exact", CLEAR.format(t="Mi15")),
    (33, "Mi9", ["Mi9"], "exact", CLEAR.format(t="Mi9")),
    (34, "Tm5c", ["Tm5c"], "exact", CLEAR.format(t="Tm5c")),
    (42, "TmY5a", ["TmY5a"], "exact", CLEAR.format(t="TmY5a")),
    (55, "Tm29", ["Tm29"], "exact", CLEAR.format(t="Tm29") + "; MaleCNS Tm29 = FlyWire Tm5d (flywireType column), same name in Özel"),
    (60, "T2a", ["T2a"], "exact", "ST1: only unidentified cluster expressing the T2a markers acj6 and ap (mixture modelling) with a unicolumnar-sized cluster; main text lists T2a among marker-identified clusters"),
    (61, "Tm20", ["Tm20"], "exact", CLEAR.format(t="Tm20")),
    (64, "LC12*", ["LC12"], "fuzzy", "ST1 (author-starred = less confident): LC12 is cut+, Acj6+, toy+, beat-1c+, kn- (Extended Data Fig. 4); only cluster matching by mixture modelling"),
    (66, "LLPC1", ["LLPC1"], "exact", CLEAR.format(t="LLPC1")),
    (72, "LPC1", ["LPC1"], "exact", CLEAR.format(t="LPC1")),
    (75, "TmY4*", ["TmY4"], "fuzzy", "ST1 (author-starred): TmY4 is D+/vvl-/toy- (Extended Data Fig. 4) and DIP-gamma+; only clusters 75 and 164 (=Mi4) match by mixture modelling; 'very likely to correspond to TmY4'"),
    (77, "LC10a", ["LC10a"], "exact", CLEAR.format(t="LC10a")),
    (78, "LC10d", ["LC10d"], "exact", CLEAR.format(t="LC10d") + "; the LC10d bulk transcriptome was of lower quality in Davis 2020"),
    (79, "LC10b", ["LC10b"], "exact", "Methods: 'LC10b transcriptome correlated best with cluster 79, but also well though more weakly with clusters 93, 78 (LC10d) and 77 (LC10a)'; assigned LC10b by best correlation"),
    (93, "93 (likely LC10c)", ["LC10c-1", "LC10c-2"], "fuzzy", "ST1 / Methods: 'cluster 93 likely being LC10c' (second-best correlate of the LC10b transcriptome); MaleCNS splits LC10c into LC10c-1 / LC10c-2 by connectivity (Nern 2025)"),
    (100, "LC16", ["LC16"], "exact", "ST1: 'Clear match to LC16 transcriptome'; Methods: best match with a correlation gap of only 0.03 (below the 0.05 criterion used elsewhere)"),
    (104, "Tm5ab", ["Tm5a", "Tm5b"], "class", CLEAR.format(t="Tm5ab") + "; the Tm5a/b driver-line transcriptome pools both types (Davis 2020), MaleCNS separates Tm5a and Tm5b by connectivity"),
    (107, "L4", ["L4"], "exact", CLEAR.format(t="L4")),
    (108, "L2", ["L2"], "exact", CLEAR.format(t="L2")),
    (109, "L1", ["L1"], "exact", CLEAR.format(t="L1")),
    (110, "L3", ["L3"], "exact", CLEAR.format(t="L3")),
    (111, "LC17*", ["LC17"], "fuzzy", "ST1 (author-starred): LC17 is Acj6+, toy+, kn+ (Extended Data Fig. 4) and beat-VI+; only cluster matching by mixture modelling"),
    (113, "Lawf1", ["Lawf1"], "exact", CLEAR.format(t="Lawf1")),
    (117, "L5", ["L5"], "exact", CLEAR.format(t="L5")),
    (118, "Tm2", ["Tm2"], "exact", CLEAR.format(t="Tm2")),
    (119, "TmY3", ["TmY3"], "exact", CLEAR.format(t="TmY3")),
    (125, "Tm4", ["Tm4"], "exact", CLEAR.format(t="Tm4")),
    (126, "Tm3", ["Tm3"], "exact", CLEAR.format(t="Tm3") + "; clusters 126+127 merged (OOBE 0.23)"),
    (128, "Dm2", ["Dm2"], "exact", CLEAR.format(t="Dm2")),
    (136, "Dm11", ["Dm11"], "exact", CLEAR.format(t="Dm11")),
    (137, "TmY14*", ["TmY14"], "fuzzy", "ST1 (author-starred): TmY14 is kn+, DIP-gamma+, toy+, D-, dac- (Extended Data Fig. 4); only cluster matching by mixture modelling; VGlut+ as expected for TmY14 (Raghu & Borst 2011); DIP-beta discrepancy noted by the authors"),
    (140, "Tm1", ["Tm1"], "exact", CLEAR.format(t="Tm1") + "; clusters 140+141 merged (OOBE 0.14)"),
    (142, "Mi1", ["Mi1"], "exact", CLEAR.format(t="Mi1") + "; clusters 142+143 merged (OOBE 0.14)"),
    (144, "Dm9", ["Dm9"], "exact", CLEAR.format(t="Dm9")),
    (145, "Lai", ["Lai"], "exact", CLEAR.format(t="Lai")),
    (147, "LC4", ["LC4"], "exact", CLEAR.format(t="LC4")),
    (148, "LPLC1", ["LPLC1"], "exact", CLEAR.format(t="LPLC1")),
    (150, "LPLC2", ["LPLC2"], "exact", CLEAR.format(t="LPLC2")),
    (151, "Pm3*", ["Pm3"], "fuzzy", "ST1 (author-starred) / Methods: the Pm3 bulk transcriptome 'correlated best, but weakly, with cluster 151', verified by previously described Pm3 markers (Extended Data Fig. 3c, mixture modelling); hemibrain name Pm3 retained in MaleCNS"),
    (152, "Pm4", ["Pm4"], "exact", CLEAR.format(t="Pm4") + "; hemibrain name Pm4 retained in MaleCNS"),
    (163, "Pm1", ["Pm1"], "exact", "ST1: only cluster hth+, svp+, tsh+, Lim3+ by mixture modelling, the Pm1 markers of Erclik et al. 2017; hemibrain name Pm1 retained in MaleCNS (hemibrainType column)"),
    (164, "Mi4", ["Mi4"], "exact", CLEAR.format(t="Mi4")),
    (165, "C3", ["C3"], "exact", CLEAR.format(t="C3")),
    (175, "Pm2", ["Pm2a", "Pm2b"], "class", CLEAR.format(t="Pm2") + "; MaleCNS splits hemibrain Pm2 into Pm2a / Pm2b by connectivity (Nern 2025 ED Fig. 2e-h; the Pm2 driver labels the group '[Pm2]')"),
    (182, "C2", ["C2"], "exact", CLEAR.format(t="C2")),
    (222, "LC6", ["LC6"], "exact", "ST1: backprojection from P70; 'Clear match to LC6 transcriptome' (cluster 97 split into 221 / 222 at P70, 222 = LC6)"),
    (225, "Dm3a", ["Dm3a", "Dm3b", "Dm3c"], "class", "ST1: Dm3 subtype from P70 backprojection. Özel could not assign the two orthogonal Dm3 arbor orientations to anatomical directions (ED Fig. 7d, 'Dm3x / Dm3y'); MaleCNS has three Dm3 types (Dm3a = FlyWire Dm3p, Dm3b = Dm3q, Dm3c = Dm3v; Nern 2025). Subtype letters are NOT assumed to correspond; mapped to the Dm3 family"),
    (226, "Dm3b", ["Dm3a", "Dm3b", "Dm3c"], "class", "as 225: Dm3 subtype letter not transferable; mapped to the Dm3 family"),
    (227, "Tm9v", ["Tm9"], "class", "ST1: ventral Tm9 (Wnt4+) from P70 backprojection; MaleCNS has a single Tm9 type (dorsal / ventral halves could be split by hex column, not done here)"),
    (228, "Tm9d", ["Tm9"], "class", "ST1: dorsal Tm9 (Wnt10+) from P70 backprojection; MaleCNS Tm9 is one type"),
    (234, "T4-5a/b", T4T5_AB, "class", "ST1: T4/T5 subdivision a/b projected from P50 (OOBE 7 % vs class 235 in adults); pools T4a, T4b, T5a, T5b"),
    (235, "T4-5c/d", T4T5_CD, "class", "ST1: T4/T5 subdivision c/d projected from P50; pools T4c, T4d, T5c, T5d"),
]

# Adult clusters with a named annotation that cannot be mapped to a MaleCNS type, with the reason.
UNMATCHED_NAMED = {
    70: ("TmY8*", "no TmY8 type in MaleCNS / Nern 2025 (FlyWire and hemibrain columns contain no TmY8 either); author-starred annotation by CG42458+ / vvl+"),
    98: ("98", "ST1: 'very likely to be a LC cell: second best match for LC16' -- no type"),
    221: ("221", "ST1: split of cluster 97 at P70, 'likely another LC neuron' -- no type"),
    229: ("88a", "unannotated subdivision of cluster 88"), 230: ("88b", "unannotated subdivision of cluster 88"),
    231: ("169a", "unannotated subdivision of cluster 169"), 232: ("169b", "unannotated subdivision of cluster 169"),
    112: ("112", "ST1: best match for Kenyon cells (Davis 2020) -- central-brain contamination, not an optic-lobe type; not mapped to MaleCNS KC types because the cluster is described as possibly mixed"),
    102: ("102", "ST1: possible multiplets including central-brain neurons"),
    191: ("191", "ST1: neither glia nor optic-lobe neuron"),
    192: ("LQ", "ST1: ambient RNA / low quality"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_annotation() -> pd.DataFrame:
    wb = openpyxl.load_workbook(EXT / "41586_2020_2879_MOESM4_ESM.xlsx", read_only=True)
    rows = list(wb["Annotation"].iter_rows(values_only=True))[1:]
    rec = [(int(r[0]), str(r[1]), (r[2] or "").replace("\n", " "), (r[3] or "").replace("\n", " "))
           for r in rows if r[0] is not None]
    return pd.DataFrame(rec, columns=["cluster", "annotation", "comment_adult", "comment_pupal"]).set_index("cluster")


def read_sheet(fname: str, sheet: str) -> tuple[pd.DataFrame, list[str]]:
    """Return the requested genes x clusters (numeric columns only) and the full gene list."""
    wb = openpyxl.load_workbook(EXT / fname, read_only=True)
    ws = wb[sheet]
    it = ws.iter_rows(values_only=True)
    header = [str(c) for c in next(it)]
    ncol = [i for i, c in enumerate(header) if i > 0 and c.isdigit()]
    want = {SOURCE_SYMBOL.get(g, g): g for g in GENES}
    genes_all, hits = [], {}
    for r in it:
        if r[0] is None:
            continue
        g = str(r[0]); genes_all.append(g)
        if g in want:
            hits[want[g]] = [r[i] for i in ncol]
    df = pd.DataFrame({g: hits.get(g, [np.nan] * len(ncol)) for g in GENES},
                      index=pd.Index([int(header[i]) for i in ncol], name="cluster")).astype(float)
    return df, genes_all


def write_csv_with_header(df: pd.DataFrame, path: Path, header_lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False, lineterminator="\n")


def main() -> None:
    ann = read_annotation()
    avg, genes_avg = read_sheet("GSE142787_Log_normalized_average_expression.xlsx", "Adult_average_expression")
    mm, genes_mm = read_sheet("GSE142787_Mixture_modeling.xlsx", "Adult_MM_final")
    assert genes_avg == genes_mm, "gene lists differ between the two GEO tables"
    adult_clusters = list(avg.index)
    mm = mm.reindex(adult_clusters)
    absent = [g for g in GENES if avg[g].isna().all()]
    print(f"adult clusters {len(adult_clusters)}, genes in source build {len(genes_avg)}, "
          f"requested genes present {len(GENES) - len(absent)}/{len(GENES)}, absent {absent}")

    def source_name(c: int) -> str:
        a = ann.loc[c, "annotation"] if c in ann.index else str(c)
        return a if a != str(c) else str(c)

    # ------------------------------------------------------------------ connectome
    neurons = pd.read_parquet(ROOT / "cache" / "neurons.parquet")
    W = sp.load_npz(ROOT / "cache" / "W_post_pre.npz").tocsr()
    absW = abs(W)
    neurons["out_syn"] = np.asarray(absW.sum(axis=0)).ravel()      # column j = presynaptic neuron j
    neurons["in_syn_raw"] = np.asarray(absW.sum(axis=1)).ravel()
    neurons["type"] = neurons["type"].fillna("")
    OL = ["ol_intrinsic", "visual_projection", "ol_sensory"]
    is_ol = neurons.superclass.isin(OL)
    ncell = neurons.groupby("type").size()
    known_types = set(ncell.index) - {""}
    for _, _, ts, _, _ in MAP:
        for t in ts:
            assert t in known_types, f"{t} is not a MaleCNS type"

    # ------------------------------------------------------------------ type map
    rows = []
    for c, sname, ts, tier, ev in MAP:
        assert c in adult_clusters, f"cluster {c} has no adult expression column"
        for t in ts:
            rows.append(dict(source_name=sname, malecns_type=t, tier=tier, evidence=ev,
                             n_cells_malecns=int(ncell[t]), source_cluster=c,
                             malecns_superclass=neurons.loc[neurons.type == t, "superclass"].mode().iat[0],
                             malecns_nt=neurons.loc[neurons.type == t, "nt"].mode().iat[0]))
    tmap = pd.DataFrame(rows)
    mapped_clusters = {c for c, *_ in MAP}
    for c in adult_clusters:
        if c in mapped_clusters:
            continue
        sname, why = UNMATCHED_NAMED.get(c, (source_name(c), None))
        a = ann.loc[c, "annotation"] if c in ann.index else str(c)
        if why is None:
            if re.match(r"^G", a) or a in ("LQ",):
                why = f"glia / low-quality cluster ('{a}')"
            else:
                why = "unannotated neuronal cluster (no type assigned by the authors)"
        tmap.loc[len(tmap)] = dict(source_name=sname, malecns_type="", tier="unmatched", evidence=why,
                                   n_cells_malecns=0, source_cluster=c, malecns_superclass="", malecns_nt="")
    tmap = tmap.sort_values(["source_cluster", "malecns_type"]).reset_index(drop=True)
    tier_rank = {"exact": 0, "alias": 1, "fuzzy": 2, "class": 3}
    write_csv_with_header(tmap, DATA / "type_map_ozel2021.csv", [
        "Özel et al. 2021 Nature 589:88-95 (doi:10.1038/s41586-020-2879-3), adult optic-lobe scRNA-seq clusters",
        "(GEO GSE142787, Supplementary Table 1 annotations) -> MaleCNS v1.0 cell types (cache/neurons.parquet).",
        "Built by scripts/build_ozel2021_tables.py; provenance and coverage in docs/audits/receptor_sources_ozel2021.md.",
        "tier: exact = author annotation is the MaleCNS type name (unstarred); alias = documented renaming (none needed);",
        "fuzzy = author-starred (less confident) annotation, a 'likely' assignment, or a subtype letter not shown to correspond;",
        "class = source cluster pools several MaleCNS types or is a subdivision of one; unmatched = no evidenced MaleCNS type.",
        "source_name = the author's annotation (cluster number when unannotated); source_cluster = ST1 / GEO column id;",
        "n_cells_malecns = cells of malecns_type in MaleCNS; malecns_nt = the model's transmitter label (mode).",
    ])

    # ------------------------------------------------------------------ expression tables
    def expr_table(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.insert(0, "source_cluster", out.index)
        out.insert(0, "source_name", [source_name(c) for c in out.index])
        best = tmap[tmap.tier != "unmatched"].groupby("source_cluster").tier.agg(lambda s: min(s, key=tier_rank.get))
        out.insert(2, "map_tier", [best.get(c, "unmatched") for c in out.index])
        return out.reset_index(drop=True)

    common = [
        "Özel et al. 2021 Nature 589:88-95 (doi:10.1038/s41586-020-2879-3); GEO GSE142787 (SuperSeries GSE142789).",
        "Adult timepoint only: 109,743 cells, female Canton-S 1-3 days old (male libraries removed by the authors), 10x v2.",
        "One row per adult cluster (199 columns of the GEO sheet: 193 final clusters incl. P70/P50 backprojections 221-235).",
        "Gene symbols are the plan's names; the source build (BDGP6.88) names differ for ChAT (=Cha), KaiR1D (=CG3822),",
        "Octalpha2R (=CG18208); mAChR-C (CG7918) is absent from the source gene set -> NaN. Built by scripts/build_ozel2021_tables.py.",
    ]
    write_csv_with_header(expr_table(avg), DATA / "expression_ozel2021.csv", common + [
        "UNIT: per-cluster arithmetic mean of Seurat LogNormalize values, ln(1 + UMI / total_UMI * 1e4) per cell",
        "(sheet Adult_average_expression of GSE142787_Log_normalized_average_expression.xlsx, non-integrated data).",
    ])
    write_csv_with_header(expr_table(mm), DATA / "expression_ozel2021_mm.csv", common + [
        "UNIT: mixture-model probability that the gene is expressed in the cluster (0-1; ~fraction of cells ON),",
        "sheet Adult_MM_final of GSE142787_Mixture_modeling.xlsx (Özel 2021 Methods 'mixture modelling').",
    ])

    # ------------------------------------------------------------------ coverage
    best_tier = (tmap[tmap.tier != "unmatched"].groupby("malecns_type").tier
                 .agg(lambda s: min(s, key=tier_rank.get)))
    neurons["tier"] = neurons.type.map(best_tier).fillna("unmatched")
    tiers = ["exact", "alias", "fuzzy", "class", "unmatched"]

    def coverage(mask: pd.Series, label: str) -> pd.DataFrame:
        sub = neurons[mask]
        g = sub.groupby("tier")
        d = pd.DataFrame({"types": g.type.agg(lambda t: t[t != ""].nunique()), "cells": g.size(), "out_syn": g.out_syn.sum(),
                          "in_syn": g.in_syn_raw.sum()}).reindex(tiers).fillna(0)
        for c in ("types", "cells", "out_syn", "in_syn"):
            d[c] = d[c].astype(int)
            d[c + "_frac"] = d[c] / max(d[c].sum(), 1)
        d.loc["total"] = d.sum(numeric_only=True)
        d.index.name = label
        return d

    cov_ol = coverage(is_ol, "optic lobe (ol_intrinsic + visual_projection + ol_sensory)")
    cov_sc = {sc: coverage(neurons.superclass == sc, sc) for sc in OL + ["visual_centrifugal"]}
    cov_cb = coverage(~is_ol & (neurons.superclass != "visual_centrifugal"), "everything else (central brain + VNC)")
    # both-ends coverage of the synapses whose presynaptic cell is in the optic lobe
    idx_ol = np.flatnonzero(is_ol.to_numpy())
    coo = absW[:, idx_ol].tocoo()
    pre_tier = neurons.tier.to_numpy()[idx_ol][coo.col]
    post_tier = neurons.tier.to_numpy()[coo.row]
    rank = np.array([tier_rank.get(t, 4) for t in tiers])
    tr = {t: i for i, t in enumerate(tiers)}
    pr = np.array([tr[t] for t in pre_tier]); po = np.array([tr[t] for t in post_tier])
    both = pd.DataFrame(index=tiers, columns=tiers, dtype=float)
    for i, a in enumerate(tiers):
        for j, b in enumerate(tiers):
            both.loc[a, b] = coo.data[(pr == i) & (po == j)].sum()
    both.index.name = "pre tier \\ post tier (synapses, pre in optic lobe)"
    tot_ol_syn = coo.data.sum()

    # transmitter cross-check on mapped clusters: mixture-model P(on) of the synthesis genes
    syn = ["ChAT", "VAChT", "Gad1", "VGAT", "VGlut", "Hdc", "ple", "DAT", "Tdc2", "Tbh", "Trh", "SerT"]
    xc = []
    for c, sname, ts, tier, _ in MAP:
        r = mm.loc[c, syn]
        calls = []
        if max(r["ChAT"], r["VAChT"]) >= 0.5: calls.append("ACh")
        if max(r["Gad1"], r["VGAT"]) >= 0.5: calls.append("GABA")
        if r["VGlut"] >= 0.5: calls.append("Glu")
        if r["Hdc"] >= 0.5: calls.append("His")
        if min(r["ple"], r["DAT"]) >= 0.5: calls.append("DA")
        if min(r["Tdc2"], r["Tbh"]) >= 0.5: calls.append("OA")
        if min(r["Trh"], r["SerT"]) >= 0.5: calls.append("5HT")
        for t in ts:
            nt = neurons.loc[neurons.type == t, "nt"].value_counts(normalize=True)
            xc.append(dict(source_cluster=c, source_name=sname, malecns_type=t, tier=tier,
                           malecns_nt=f"{nt.index[0]} ({nt.iat[0]:.0%})", transcript_call="+".join(calls) or "none",
                           **{g: round(float(r[g]), 2) for g in syn}))
    xc = pd.DataFrame(xc)
    short = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu", "histamine": "His", "dopamine": "DA",
             "octopamine": "OA", "serotonin": "5HT", "unknown": "unknown"}
    xc["agree"] = [short[m.split(" ")[0]] in t.split("+") for m, t in zip(xc.malecns_nt, xc.transcript_call)]

    # ------------------------------------------------------------------ audit document
    files = [
        ("GSE142787_Log_normalized_average_expression.xlsx", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE142nnn/GSE142787/suppl/GSE142787_Log_normalized_average_expression.xlsx", "per-cluster mean log-normalised expression, 6 sheets (P15, P30, P40, P50, P70, Adult); Adult sheet 12,028 genes x 199 clusters"),
        ("GSE142787_Mixture_modeling.xlsx", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE142nnn/GSE142787/suppl/GSE142787_Mixture_modeling.xlsx", "per-cluster binarised expression (mixture-model P(on)), 6 sheets; Adult sheet 12,028 genes x 199 clusters + model columns"),
        ("GSE142787_Cluster_markers.xlsx", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE142nnn/GSE142787/suppl/GSE142787_Cluster_markers.xlsx", "Seurat FindAllMarkers per stage (raw + curated sheets); Cluster_markers_Adult 51,393 rows"),
        ("GSE142787_family.soft.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE142nnn/GSE142787/soft/GSE142787_family.soft.gz", "GEO series metadata (38 10x libraries; adult samples: Canton S, 1-3 d, female or pooled, optic lobe)"),
        ("41586_2020_2879_MOESM1_ESM.pdf", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-2879-3/MediaObjects/41586_2020_2879_MOESM1_ESM.pdf", "Supplementary Information (discussion; describes the three GEO summary tables)"),
        ("41586_2020_2879_MOESM2_ESM.pdf", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-2879-3/MediaObjects/41586_2020_2879_MOESM2_ESM.pdf", "Reporting Summary"),
        ("41586_2020_2879_MOESM3_ESM.pdf", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-2879-3/MediaObjects/41586_2020_2879_MOESM3_ESM.pdf", "FACS gating strategy"),
        ("41586_2020_2879_MOESM4_ESM.xlsx", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-2879-3/MediaObjects/41586_2020_2879_MOESM4_ESM.xlsx", "Supplementary Table 1: cluster -> cell-type annotation with adult / pupal comments (435 rows, clusters 1-268)"),
        ("41586_2020_2879_MOESM5_ESM.xlsx", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-2879-3/MediaObjects/41586_2020_2879_MOESM5_ESM.xlsx", "Supplementary Table 2: unique transcription-factor combinations per neuronal cluster (1,408 rows)"),
        ("41586_2020_2879_MOESM6_ESM.docx", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-2879-3/MediaObjects/41586_2020_2879_MOESM6_ESM.docx", "Supplementary Table 3: fly strains and antibodies"),
        ("41586_2020_2879_MOESM7_ESM.zip", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-020-2879-3/MediaObjects/41586_2020_2879_MOESM7_ESM.zip", "Appendix 1: the Keras neural-network classifier (model*.h5, Means*.npy, marker lists, Readme)"),
    ]
    not_downloaded = [
        ("GSE142787_Adult.rds.gz", "3.5 GB", "full adult Seurat object (cell-level counts); over the 2 GB limit and needs R -> not downloaded; per-cluster summaries used instead"),
        ("GSE142787_Adult_male.rds.gz", "58 MB", "the discarded adult male cells as a Seurat object; needs R (no R on this machine) -> not downloaded"),
        ("GSE142787_P15/P30/P40/P50/P70.rds.gz", "0.6-1.1 GB each", "pupal stages, not used (adult timepoint only)"),
    ]

    def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
        d = df.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
        cols = [d.index.name or ""] + list(map(str, d.columns))
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for i, r in d.iterrows():
            lines.append("| " + " | ".join([str(i)] + [str(v) for v in r]) + " |")
        return "\n".join(lines)

    def cov_md(d: pd.DataFrame) -> str:
        d = d.copy()
        for c in ("types", "cells", "out_syn", "in_syn"):
            d[c] = d[c].astype(int).map("{:,}".format)
            d[c + "_frac"] = d[c + "_frac"].map("{:.1%}".format)
        return md(d)

    n_tiers = tmap[tmap.tier != "unmatched"].groupby("tier").agg(clusters=("source_cluster", "nunique"),
                                                                 malecns_types=("malecns_type", "nunique")).reindex(["exact", "alias", "fuzzy", "class"]).fillna(0).astype(int)
    unm = tmap[tmap.tier == "unmatched"]
    ann_named = unm[~unm.source_name.str.fullmatch(r"\d+")]
    out = []
    out.append("# Receptor / transmitter expression sources: Özel et al. 2021 (adult optic lobe scRNA-seq)\n")
    out.append(f"Generated by `scripts/build_ozel2021_tables.py` on {date.today().isoformat()}. Source key `ozel2021`. "
               "Part of the plan in `docs/NT_INTEGRATION.md` (step 1 acquire-and-pin, step 2 type map).\n")
    out.append("## 1. Citation and licence\n")
    out.append("Özel, M. N., Simon, F., Jafari, S., Holguera, I., Chen, Y.-C., Benhra, N., El-Danaf, R. N., Kapuralin, K., Malin, J. A., "
               "Konstantinides, N. & Desplan, C. *Neuronal diversity and convergence in a visual system developmental atlas.* "
               "Nature 589, 88-95 (2021). https://doi.org/10.1038/s41586-020-2879-3 (PMID 33149298, PMC7790857).\n")
    out.append("* Data: GEO **GSE142787** (scRNA-seq sub-series; the paper's Data availability names the SuperSeries **GSE142789**). "
               "38 10x Genomics Chromium v2 libraries (~7,000 cells each) from adult and five pupal stages; adult = 109,743 cells after "
               "QC, **female Canton-S, 1-3 days old** (two libraries also contained males; the authors removed those cells; MaleCNS is male).\n"
               "* Licence: the Nature article is under Springer Nature's standard terms (not CC BY); the GEO records carry no licence "
               "statement (NCBI GEO data are publicly accessible for reuse with citation). The raw files stay in `data/external/ozel2021/` "
               "(git-ignored). The files under `flyverse/data/` are derived summary statistics (54 genes x 199 clusters of the authors' "
               "own per-cluster summary tables, plus our mapping) with attribution, which is the reuse the Supplementary Discussion "
               "invites ('These tables can be used to identify molecular effectors ... without additional bioinformatic analysis').\n"
               "* Nomenclature: the authors named clusters with the EM / Fischbach type names of 2020 and matched them to driver-line bulk "
               "transcriptomes (Konstantinides et al. 2018 Cell; Davis et al. 2020 eLife). MaleCNS uses the Nern et al. 2025 names; the "
               "renamings that matter here were read off the `hemibrainType` / `flywireType` columns of `cache/neurons.parquet` "
               "(Tm29 = FlyWire Tm5d; Dm3a/b/c = FlyWire Dm3p/q/v; Dm8a/b = yDm8/pDm8; Pm1-4 keep their hemibrain names; Pm2 -> Pm2a/Pm2b, "
               "LC10c -> LC10c-1/-2, LC14a -> LC14a-1/-2 are MaleCNS connectivity splits).\n")
    out.append("## 2. Files\n")
    out.append("| file (`data/external/ozel2021/`) | URL | bytes | SHA-256 | content |\n|---|---|---|---|---|")
    for fname, url, desc in files:
        p = EXT / fname
        out.append(f"| {fname} | {url} | {p.stat().st_size:,} | `{sha256(p)}` | {desc} |")
    out.append("\nNot downloaded:\n")
    out.append("| file | size | reason |\n|---|---|---|")
    for fname, size, why in not_downloaded:
        out.append(f"| {fname} | {size} | {why} |")
    out.append("\nAll downloads: `curl -sL` from the URLs above on " + date.today().isoformat() + "; GEO FTP listing dated 2019-12-18 .. 2019-12-31 for the supplementary files.\n")
    out.append("## 3. Derived tables\n")
    out.append(f"* `flyverse/data/type_map_ozel2021.csv`: {len(tmap)} rows = {tmap.source_cluster.nunique()} adult clusters; "
               f"{len(tmap[tmap.tier != 'unmatched'])} (cluster, MaleCNS type) pairs over {tmap[tmap.tier != 'unmatched'].malecns_type.nunique()} MaleCNS types; "
               f"{len(unm)} clusters unmatched ({len(ann_named)} of them carry an author annotation that has no MaleCNS type: "
               + ", ".join(f"{r.source_name} [{r.source_cluster}]" for r in ann_named.itertuples()) + ").\n")
    out.append(md(n_tiers.rename_axis("tier")) + "\n")
    out.append(f"* `flyverse/data/expression_ozel2021.csv`: {len(avg)} clusters x {len(GENES)} genes, per-cluster mean of Seurat "
               f"LogNormalize values ln(1 + UMI/total x 1e4). {len(GENES) - len(absent)}/{len(GENES)} genes present in the source gene set "
               f"({len(genes_avg):,} genes); absent -> NaN: {absent} (mAChR-C = CG7918 is not in the BDGP6.88 set the authors used). "
               "Symbol translations: ChAT = `Cha`, KaiR1D = `CG3822`, Octalpha2R = `CG18208`.\n"
               f"* `flyverse/data/expression_ozel2021_mm.csv`: the same clusters x genes as the mixture-model probability of expression "
               "(0-1; the authors' binarised 'on/off' call per cluster, Methods 'mixture modelling'). This is the binned quantity the plan's "
               "step 4 asks for (expression is not conductance).\n")
    out.append("Tier rules: **exact** = the author's annotation (unstarred) is the MaleCNS type name; **alias** = documented renaming "
               "(no row needed for this source: every name that differs between 2020 and MaleCNS is a split, i.e. class); **fuzzy** = "
               "author-starred (`*` = less confident, marker-combination evidence only), or a 'likely' assignment (cluster 93 -> LC10c), "
               "or a subtype letter not shown to correspond; **class** = the cluster pools several MaleCNS types (PR, Dm8, LC14, Tm5ab, Pm2, "
               "T4-5a/b, T4-5c/d, Dm3a/b) or subdivides one (Tm9v / Tm9d). Every row keeps the source cluster id and the ST1 sentence it rests on.\n")
    out.append("## 4. Coverage of MaleCNS (best tier per type; cells and synapses from `cache/`, `|W|` column sums = output synapses, "
               "row sums = input synapses; sign-0 synapses (2.2 % overall, 0.1 % of optic-lobe output) are zeros in `W` and therefore excluded)\n")
    out.append(cov_md(cov_ol) + "\n")
    for sc in OL + ["visual_centrifugal"]:
        out.append(cov_md(cov_sc[sc]) + "\n")
    out.append(cov_md(cov_cb) + "\n")
    out.append("Central-brain / VNC coverage is 0 by construction: the atlas is optic lobe only (cluster 112 = Kenyon-cell contamination is not mapped).\n")
    out.append("### 4a. Both-ends coverage of optic-lobe output synapses (pre-tier x post-tier; counts of synapses whose presynaptic cell is in the optic lobe)\n")
    b = both.copy()
    b["row_total"] = b.sum(axis=1)
    b.loc["col_total"] = b.sum(axis=0)
    out.append(md(b.map(lambda v: f"{int(v):,}")) + "\n")
    named = ["exact", "fuzzy", "class"]
    both_named = both.loc[named, named].to_numpy().sum()
    both_ef = both.loc[["exact", "fuzzy"], ["exact", "fuzzy"]].to_numpy().sum()
    out.append(f"* Optic-lobe output synapses (total {tot_ol_syn:,.0f}): pre AND post matched at any tier {both_named:,.0f} "
               f"({both_named / tot_ol_syn:.1%}); at exact or fuzzy on both ends {both_ef:,.0f} ({both_ef / tot_ol_syn:.1%}); "
               f"pre matched at any tier {both.loc[named].to_numpy().sum():,.0f} ({both.loc[named].to_numpy().sum() / tot_ol_syn:.1%}).\n")
    # largest unmatched optic types
    um = neurons[is_ol & (neurons.tier == "unmatched")].groupby("type").agg(cells=("bodyId", "size"), out_syn=("out_syn", "sum")).sort_values("out_syn", ascending=False)
    out.append("### 4b. The 40 unmatched optic-lobe types with the most output synapses\n")
    um40 = um.head(40).copy(); um40["cells"] = um40.cells.map("{:,}".format); um40["out_syn"] = um40.out_syn.map("{:,.0f}".format)
    out.append(md(um40.rename_axis("type")) + "\n")
    out.append("## 5. Transmitter cross-check on the mapped clusters (mixture-model P(on) of the synthesis / transport genes; "
               "`transcript_call` = ACh if max(ChAT, VAChT) >= 0.5, GABA if max(Gad1, VGAT) >= 0.5, Glu if VGlut >= 0.5, His if Hdc >= 0.5, "
               "DA / OA / 5HT if both genes of the pair >= 0.5; `malecns_nt` = the model's label with its share of the type's cells)\n")
    out.append(md(xc.set_index("source_cluster"), "{:.2f}") + "\n")
    dis = xc[~xc.agree]
    out.append(f"* {int(xc.agree.sum())} of {len(xc)} (cluster, type) pairs agree with the model's label; disagreements ({len(dis)}): "
               + "; ".join(f"{r.malecns_type} [{r.source_name}] model {r.malecns_nt} vs transcript {r.transcript_call}" for r in dis.itertuples()) + ".\n")
    out.append("## 6. Caveats\n")
    out.append("* Sex: female atlas vs male connectome (noted in `docs/NT_INTEGRATION.md` section 6).\n"
               "* Age: adult 1-3 days; receptor expression in older flies may differ.\n"
               "* Cluster purity: the authors flag LC14 (possible LC14/LC14b mix), Pm3 (weak correlation), LC16 (gap 0.03), LC10b (LC10a/c/d correlate too).\n"
               "* T4/T5: only the a/b vs c/d split exists in the adult tables (clusters 234 / 235); the per-subtype clusters 261-268 are P50-only.\n"
               "* Dm3: the Özel a/b letters and the MaleCNS a/b/c letters are independent labellings; both Dm3 rows are mapped to the family.\n"
               "* 111 neuronal clusters (of 172) carry no type; the largest unmatched MaleCNS optic types are listed in 4b. Expect these to "
               "be resolvable only with a later atlas (e.g. the 2024-2025 optic-lobe / FlyWire-matched atlases) or by marker-gene matching, "
               "which this task did not attempt (no mapping without evidence).\n")
    AUDIT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {AUDIT}")
    print(cov_ol)
    print(both)
    print(xc[["source_name", "malecns_type", "tier", "malecns_nt", "transcript_call", "agree"]].to_string())


if __name__ == "__main__":
    main()
