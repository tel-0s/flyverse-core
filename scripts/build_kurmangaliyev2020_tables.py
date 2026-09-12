"""Build the Kurmangaliyev et al. 2020 (GEO GSE156455, pupal optic-lobe scRNA-seq, 24-96 h APF) -> MaleCNS tables.

Round 2 of the receptor-expression integration (docs/NT_INTEGRATION.md section 7): the source was searched for a
receptor profile of the small-object pathway types (Tm5Y, TmY21, TmY13, LC11, Y3, Li19, Tm32) that no round-1
source covers. The GEO record ships only the raw count matrix (10x MEX, 331 M non-zeros) plus per-cell metadata
with the authors' cluster / cell-type annotation, so the per-cluster summary is computed here on CPU (one streaming
pass over the matrix, ~5-10 min; the per-cell counts of the 54 genes are cached next to the raw files).

Inputs (git-ignored; `python scripts/fetch_data.py --external kurmangaliyev2020`):
    data/external/kurmangaliyev2020/GSE156455_README.txt
    data/external/kurmangaliyev2020/GSE156455_features_main.tsv.gz      17,561 genes (FlyBase r6.29 symbols)
    data/external/kurmangaliyev2020/GSE156455_barcodes_main.tsv.gz      193,727 cells
    data/external/kurmangaliyev2020/GSE156455_matrix_main.mtx.gz        raw UMI counts, MatrixMarket, 1.0 GB gz
    data/external/kurmangaliyev2020/GSE156455_metadata_main.tsv.gz      set / rep / trep / genotype / time / class / type / subtype
    data/external/kurmangaliyev2020/GSE156455_family.soft.gz            GEO series / sample metadata
    cache/neurons.parquet, cache/W_post_pre.npz                          the model's connectome cache
    flyverse/data/receptors_by_type.csv                                  round-1 receptor table (which types already have a profile)

Outputs (redistributable derived tables):
    flyverse/data/type_map_kurmangaliyev2020.csv      source cluster -> MaleCNS type, with tier + evidence
    flyverse/data/expression_kurmangaliyev2020.csv    per cluster x timepoint x metric, 54 NT genes
    docs/audits/receptor_sources_profiles.md          provenance (URL, size, SHA-256), licence, coverage, target-type outcome

    PYTHONIOENCODING=utf-8 python scripts/build_kurmangaliyev2020_tables.py [--no-cache]
"""
from __future__ import annotations

import gzip
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "kurmangaliyev2020"
DATA = ROOT / "flyverse" / "data"
AUDIT = ROOT / "docs" / "audits" / "receptor_sources_profiles.md"
CACHE = EXT / "_percell_54genes.npz"

# ----------------------------------------------------------------------------------------------
# Gene list (model names), identical to scripts/build_ozel2021_tables.py. The source build is FlyBase r6.29
# (Cell Ranger reference, GEO SOFT 'Genome_build'), which already carries the modern symbols: ChAT, KaiR1D,
# mAChR-C, Octalpha2R are present as such (no translation needed; all 54 found).
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
SOURCE_SYMBOL: dict[str, str] = {}
# extra genes carried through the streaming pass for the sex check (male-specific lncRNAs) and QC
EXTRA = {"roX1": "lncRNA:roX1", "roX2": "lncRNA:roX2", "Yp1": "Yp1", "Yp2": "Yp2", "elav": "elav", "repo": "repo"}
TIMES = ["24h", "36h", "48h", "60h", "72h", "84h", "96h"]
LATE = ["72h", "84h", "96h"]
FRAC_ON = 0.2   # 'on' call used in the audit document's transmitter cross-check (fraction of cells with >= 1 UMI)

# The seven small-object-pathway types the round-2 task asked this source for.
TARGETS = ["Tm5Y", "TmY21", "TmY13", "LC11", "Y3", "Li19", "Tm32"]

# ----------------------------------------------------------------------------------------------
# Cluster -> MaleCNS type map. Cluster unit = the metadata `subtype` column (196 labels: 187 `type` labels with
# T4.T5 -> T4a-d/T5a-d, Dm3 -> Dm3a/b, Tm9 -> Tm9a/b), plus the three parent pools as extra rows. Tiers as in
# build_ozel2021_tables.py: exact (author annotation is the MaleCNS type name), alias (documented renaming),
# fuzzy (annotation not verifiable against MaleCNS or a known name collision), class (pool of several MaleCNS
# types, or a subdivision of one). Evidence = GEO metadata column + the bridge used.
# ----------------------------------------------------------------------------------------------
META = "GSE156455_metadata_main.tsv.gz `type` column (authors' cluster annotation; Kurmangaliyev 2020 STAR Methods, annotation by known marker genes / driver-line transcriptomes of Konstantinides 2018 and Davis 2020 -- the article is paywalled, the method statement is not verified here)"
PHOTORECEPTORS_78 = ["R7y", "R7p", "R7d", "R7_unclear", "R8y", "R8p", "R8d", "R8_unclear", "R7R8_unclear"]
T4T5 = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
DM3 = ["Dm3a", "Dm3b", "Dm3c"]

EXACT = ["T1", "T2", "T2a", "T3", "C2", "C3", "L1", "L2", "L3", "L4", "L5", "Mi1", "Mi4", "Mi9", "Mi15",
         "Tm1", "Tm2", "Tm3", "Tm4", "Tm20", "Tm5c", "TmY3", "TmY5a", "Dm1", "Dm2", "Dm4", "Dm9", "Dm10",
         "Dm11", "Dm12", "Lawf1", "Lawf2", "Lai", "LLPC1", "LPC1", "LC4", "LC6", "LC10a", "LC10b", "LPLC1",
         "LPLC2", "Pm4", "Tm9"]
EXACT_NOTES = {
    "Dm11": "Nern 2025 Sup_Table_7: Schlegel Dm11 = MaleCNS Dm11 (158 cells) + Dm-DRA2 (33 cells); minor split not resolved here",
    "Lai": "Nern 2025 Sup_Table_7: many OL Lai fragmented (48 counted there; 84 typed in MaleCNS)",
    "Pm4": "name bridge: Nern 2025 Sup_Table_7 Pm4 = Schlegel Pm4 / Matsliah Pm05 (hemibrain_type None)",
    "Tm9": "parent of the Tm9a / Tm9b subtypes (both mapped to the single MaleCNS Tm9 at class tier)",
    "LC10b": "Davis 2020 driver-defined type; MaleCNS LC10b 95 cells",
}
MAP: list[tuple[str, str, list[str], str, str]] = []
for t in EXACT:
    MAP.append((t, t, [t], "exact", META + (f"; {EXACT_NOTES[t]}" if t in EXACT_NOTES else "")))
MAP += [
    ("R1.6", "R1.6", ["R1-R6"], "alias", META + "; 'R1.6' is the outer photoreceptors R1-R6 (metadata class R); MaleCNS type 'R1-R6' (flywireType R1-6)"),
    ("R7.8", "R7.8", PHOTORECEPTORS_78, "class", META + "; inner photoreceptors R7 + R8 pooled (class R); MaleCNS splits them into pale / yellow / DRA / unclear types"),
    ("Lpi3.4", "Lpi3.4", ["LPi34"], "alias", META + "; 'Lpi3.4' = LPi3-4 of Mauss et al. 2015 (lobula-plate intrinsic, layer 3 -> 4); Nern 2025 dropped the hyphen (LPi34 = Matsliah LPi09 / Schlegel CB3857, Sup_Table_7); same bridge as the Davis 2020 'LPi-34' alias row"),
    ("Tm29", "Tm29", ["Tm29"], "fuzzy", META + "; NAME COLLISION (receptor_verification.md, ozel2021 correction): MaleCNS Tm29 = FlyWire Tm5d (Matsliah) / CB3851 (Schlegel; also lumps Tm39 / Tm40), no hemibrain name; whether the 2018-2020 'Tm29' driver / marker definition labels Nern-2025 Tm29 is not evidenced; transmitter Glu agrees on both sides (weak support)"),
    ("Pm3", "Pm3", ["Pm3"], "fuzzy", META + "; name bridge unverified: Nern 2025 Sup_Table_7 Pm3 = Schlegel CB3856 / Matsliah Pm09 (no FlyWire 'Pm3', hemibrain_type None); the Erclik 2017 / Konstantinides 2018 'Pm3' marker definition is assumed to be the type Nern 2025 kept the name for; Ozel 2021 matched the same bulk transcriptome only weakly (cluster 151)"),
    ("Dm8", "Dm8", ["Dm8a", "Dm8b"], "class", META + "; MaleCNS splits Dm8 into Dm8a (FlyWire yDm8) and Dm8b (pDm8) by connectivity (Nern 2025); the cluster contains both"),
    ("Dm3", "Dm3", DM3, "class", META + "; parent pool of the Dm3a / Dm3b subtypes; MaleCNS has three Dm3 types (Dm3a = FlyWire Dm3p, Dm3b = Dm3q, Dm3c = Dm3v; Nern 2025)"),
    ("Dm3a", "Dm3a", DM3, "class", META + " `subtype`; the authors' a / b letters (two arbor orientations) and the MaleCNS a / b / c letters are independent labellings, not shown to correspond -> mapped to the Dm3 family (same treatment as Ozel 2021 clusters 225 / 226)"),
    ("Dm3b", "Dm3b", DM3, "class", META + " `subtype`; as Dm3a: letter not transferable, mapped to the Dm3 family"),
    ("Tm9a", "Tm9a", ["Tm9"], "class", META + " `subtype`; subdivision of Tm9 (MaleCNS has one Tm9 type; the a / b split is not assigned to dorsal / ventral here)"),
    ("Tm9b", "Tm9b", ["Tm9"], "class", META + " `subtype`; subdivision of Tm9"),
    ("T4.T5", "T4.T5", T4T5, "class", META + "; parent pool of the eight T4 / T5 subtypes"),
]
for t in T4T5:
    MAP.append((t, t, [t], "exact", META + " `subtype`; T4/T5 subtype clusters; README 11/02/20: 'Labels for T4c and T4d subtype clusters were swapped (T4c <-> T4d)' -- the corrected metadata_main.tsv.gz (2020-11-04) is used"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_inputs():
    feat = pd.read_csv(EXT / "GSE156455_features_main.tsv.gz", sep="\t", header=None, names=["fbgn", "symbol", "kind"])
    bc = pd.read_csv(EXT / "GSE156455_barcodes_main.tsv.gz", sep="\t", header=None, names=["barcode"])
    meta = pd.read_csv(EXT / "GSE156455_metadata_main.tsv.gz", sep="\t")
    meta = meta.set_index("barcode").reindex(bc.barcode).reset_index()
    assert meta.type.notna().all(), "metadata / barcode mismatch"
    meta["library"] = meta.barcode.str.rsplit("_", n=1).str[0]
    return feat, meta


def stream_counts(feat: pd.DataFrame, ncell: int, symbols: list[str], use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """One pass over the MEX matrix: per-cell total UMI and per-cell counts of `symbols`. Cached in CACHE."""
    mtx = EXT / "GSE156455_matrix_main.mtx.gz"
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        if list(z["symbols"]) == symbols and str(z["mtx_sha"]) == sha256(mtx):
            return z["totals"], z["counts"], int(z["nnz"])
    sym_to_row = {s: i for i, s in enumerate(feat.symbol)}
    sel = np.full(len(feat), -1, dtype=np.int64)
    for j, s in enumerate(symbols):
        sel[sym_to_row[s]] = j
    nsel = len(symbols)
    totals = np.zeros(ncell, dtype=np.float64)
    counts = np.zeros(ncell * nsel, dtype=np.float64)
    with gzip.open(mtx, "rt") as f:
        header = f.readline(); dims = f.readline().split()
        assert header.startswith("%%MatrixMarket"), header
        ngene, nc, nnz = int(dims[0]), int(dims[1]), int(dims[2])
        assert ngene == len(feat) and nc == ncell, dims
        seen = 0
        for chunk in pd.read_csv(f, sep=" ", header=None, names=["g", "c", "v"], dtype=np.int64, chunksize=20_000_000, engine="c"):
            g = chunk.g.to_numpy() - 1; c = chunk.c.to_numpy() - 1; v = chunk.v.to_numpy().astype(np.float64)
            totals += np.bincount(c, weights=v, minlength=ncell)
            s = sel[g]; m = s >= 0
            counts += np.bincount(c[m] * nsel + s[m], weights=v[m], minlength=ncell * nsel)
            seen += len(chunk)
            print(f"  streamed {seen / 1e6:,.0f} M / {nnz / 1e6:,.0f} M non-zeros", flush=True)
        assert seen == nnz, (seen, nnz)
    counts = counts.reshape(ncell, nsel).astype(np.float32)
    np.savez_compressed(CACHE, totals=totals, counts=counts, symbols=np.array(symbols), mtx_sha=sha256(mtx), nnz=nnz)
    return totals, counts, nnz


def write_csv_with_header(df: pd.DataFrame, path: Path, header_lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False, lineterminator="\n")


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


def main() -> None:
    use_cache = "--no-cache" not in sys.argv
    feat, meta = read_inputs()
    ncell = len(meta)
    present = set(feat.symbol)
    missing = [g for g in GENES if SOURCE_SYMBOL.get(g, g) not in present]
    assert not missing, f"genes absent from the r6.29 feature list: {missing}"
    symbols = [SOURCE_SYMBOL.get(g, g) for g in GENES] + list(EXTRA.values())
    print(f"cells {ncell:,}, genes {len(feat):,}, requested genes present {len(GENES)}/{len(GENES)}")
    totals, counts, nnz = stream_counts(feat, ncell, symbols, use_cache)
    print(f"non-zeros {nnz:,}; total UMI median per cell {np.median(totals):,.0f}")
    col = {g: j for j, g in enumerate(GENES)}
    col.update({k: len(GENES) + j for j, k in enumerate(EXTRA)})
    lognorm = np.log1p(counts / totals[:, None] * 1e4).astype(np.float32)
    on = counts >= 1

    # ------------------------------------------------------------------ cluster unit
    meta["cluster"] = meta.subtype.astype(str)
    parents = meta.loc[meta.type != meta.subtype, ["type", "subtype"]].drop_duplicates()
    parent_of = dict(zip(parents.subtype, parents.type))
    cls_of = meta.groupby("cluster")["class"].agg(lambda s: s.mode().iat[0]).to_dict()
    cls_of.update({p: meta.loc[meta.type == p, "class"].mode().iat[0] for p in parent_of.values()})

    def unannotated(c: str) -> bool:
        # N## / G## / C## / R## numbered clusters; the neuron types C2 / C3 (class N) are annotated names
        return re.match(r"^[NGCR]\d+$", c) is not None and c[0] == cls_of[c]

    clusters = sorted(meta.cluster.unique(), key=lambda s: (unannotated(s), s)) + sorted(set(parent_of.values()))
    n_by_cluster = meta.groupby("cluster").size()

    # ------------------------------------------------------------------ connectome
    neurons = pd.read_parquet(ROOT / "cache" / "neurons.parquet")
    W = sp.load_npz(ROOT / "cache" / "W_post_pre.npz").tocsr()
    absW = abs(W)
    neurons["out_syn"] = np.asarray(absW.sum(axis=0)).ravel()
    neurons["in_syn_raw"] = np.asarray(absW.sum(axis=1)).ravel()
    neurons["type"] = neurons["type"].fillna("")
    OL = ["ol_intrinsic", "visual_projection", "ol_sensory"]
    is_ol = neurons.superclass.isin(OL)
    ncell_t = neurons.groupby("type").size()
    known_types = set(ncell_t.index) - {""}
    for _, _, ts, _, _ in MAP:
        for t in ts:
            assert t in known_types, f"{t} is not a MaleCNS type"
    mapped_names = {c for c, *_ in MAP}
    for c in mapped_names:
        assert c in set(clusters), f"mapped cluster {c} not in the metadata"
    annotated = [c for c in clusters if not unannotated(c)]
    assert set(annotated) == mapped_names, set(annotated) ^ mapped_names

    # ------------------------------------------------------------------ type map
    rows = []
    for c, sname, ts, tier, ev in MAP:
        for t in ts:
            rows.append(dict(source_name=sname, malecns_type=t, tier=tier, evidence=ev, n_cells_malecns=int(ncell_t[t]),
                             source_cluster=c, n_cells_source=int(n_by_cluster.get(c, meta[meta.type == c].shape[0])),
                             malecns_superclass=neurons.loc[neurons.type == t, "superclass"].mode().iat[0],
                             malecns_nt=neurons.loc[neurons.type == t, "nt"].mode().iat[0]))
    tmap = pd.DataFrame(rows)
    for c in clusters:
        if c in mapped_names:
            continue
        cls = c[0]
        why = {"N": "unannotated neuronal cluster (metadata class N; no type assigned by the authors)",
               "G": "glia cluster (metadata class G)",
               "C": "central-brain cluster (metadata class C; not an optic-lobe type)",
               "R": "unannotated photoreceptor-class cluster (metadata class R)"}[cls]
        tmap.loc[len(tmap)] = dict(source_name=c, malecns_type="", tier="unmatched", evidence=why, n_cells_malecns=0,
                                   source_cluster=c, n_cells_source=int(n_by_cluster[c]), malecns_superclass="", malecns_nt="")
    tier_rank = {"exact": 0, "alias": 1, "fuzzy": 2, "class": 3}
    tmap["_k"] = tmap.source_cluster.map({c: i for i, c in enumerate(clusters)})
    tmap = tmap.sort_values(["_k", "malecns_type"]).drop(columns="_k").reset_index(drop=True)
    write_csv_with_header(tmap, DATA / "type_map_kurmangaliyev2020.csv", [
        "Kurmangaliyev, Yoo, Valdes-Aleman, Sanfilippo & Zipursky 2020 Neuron 108:1045-1057 (doi:10.1016/j.neuron.2020.10.006),",
        "pupal optic-lobe scRNA-seq clusters (GEO GSE156455, metadata_main `type` / `subtype` columns) -> MaleCNS v1.0 cell types",
        "(cache/neurons.parquet). Built by scripts/build_kurmangaliyev2020_tables.py; provenance and coverage in",
        "docs/audits/receptor_sources_profiles.md. Cluster unit = `subtype` (196 labels) + the parent pools T4.T5 / Dm3 / Tm9.",
        "tier: exact = author annotation is the MaleCNS type name; alias = documented renaming (R1.6 -> R1-R6, Lpi3.4 -> LPi34);",
        "fuzzy = name collision / bridge not verifiable (Tm29, Pm3); class = cluster pools several MaleCNS types or subdivides one;",
        "unmatched = no evidenced MaleCNS type (N## unannotated neurons, G## glia, C## central brain, R184).",
        "n_cells_source = cells of the cluster in the main dataset (all timepoints 24-96 h APF); n_cells_malecns = cells of the type in MaleCNS.",
    ])

    # ------------------------------------------------------------------ expression table
    best = tmap[tmap.tier != "unmatched"].groupby("source_cluster").tier.agg(lambda s: min(s, key=tier_rank.get))
    cl_idx = {c: np.flatnonzero(meta.cluster.to_numpy() == c) for c in clusters if c not in parent_of.values()}
    for p in set(parent_of.values()):
        cl_idx[p] = np.flatnonzero(meta.type.to_numpy() == p)
    time_arr = meta.time.to_numpy()
    recs = []
    for c in clusters:
        idx = cl_idx[c]
        for tlabel, tset in [(t, [t]) for t in TIMES] + [("late", LATE), ("all", TIMES)]:
            ii = idx[np.isin(time_arr[idx], tset)]
            if len(ii) == 0:
                continue
            base = dict(source_name=c, source_cluster=c, map_tier=best.get(c, "unmatched"), time=tlabel, n_cells=int(len(ii)))
            recs.append(dict(base, metric="mean_log1p_cp10k", **{g: float(lognorm[ii, col[g]].mean()) for g in GENES}))
            recs.append(dict(base, metric="frac_expr", **{g: float(on[ii, col[g]].mean()) for g in GENES}))
    expr = pd.DataFrame(recs)
    write_csv_with_header(expr, DATA / "expression_kurmangaliyev2020.csv", [
        "Kurmangaliyev et al. 2020 Neuron 108:1045-1057 (doi:10.1016/j.neuron.2020.10.006); GEO GSE156455, main dataset",
        "(193,727 cells, 24-96 h APF in 12 h steps, w1118 + 40 DGRP lines multiplexed, 10x v3, FlyBase r6.29; sex not annotated --",
        "see docs/audits/receptor_sources_profiles.md for the roX1 / roX2 check). PUPAL atlas: the '96h' rows are pharate adults",
        "(the GEO sample titles call them 'Adult'); there is no post-eclosion timepoint. Built by scripts/build_kurmangaliyev2020_tables.py",
        "from the raw count matrix (one streaming pass; no author-provided per-cluster table exists).",
        "One row per (cluster, time, metric): time = 24h..96h, 'late' = 72h+84h+96h pooled, 'all' = every timepoint; n_cells = cells averaged.",
        "metric mean_log1p_cp10k = per-cluster arithmetic mean of ln(1 + UMI / total_UMI x 1e4) per cell (Seurat LogNormalize, as in",
        "expression_ozel2021.csv); metric frac_expr = fraction of the cluster's cells with >= 1 UMI of the gene (as in expression_central.csv).",
        "Gene symbols are the plan's names; all 54 are present in the r6.29 feature list under these symbols (no translation needed).",
    ])

    # ------------------------------------------------------------------ sex check (per library)
    rox = counts[:, col["roX1"]] + counts[:, col["roX2"]]
    meta["rox_cp10k"] = rox / totals * 1e4
    meta["male_like"] = meta.rox_cp10k >= 5.0
    sexchk = meta.groupby("library").agg(cells=("barcode", "size"), median_rox_cp10k=("rox_cp10k", "median"),
                                          frac_male_like=("male_like", "mean")).sort_index()
    sex_by_time = meta.groupby("time").agg(cells=("barcode", "size"), frac_male_like=("male_like", "mean")).reindex(TIMES)
    male_frac = float(meta.male_like.mean())
    rox_zero = float(((counts[:, col["roX1"]] + counts[:, col["roX2"]]) == 0).mean())
    sex_verdict = ("female (roX1 / roX2 essentially absent)" if male_frac < 0.05 else "male" if male_frac > 0.95 else "mixed-sex")

    # ------------------------------------------------------------------ coverage
    best_tier = (tmap[tmap.tier != "unmatched"].groupby("malecns_type").tier.agg(lambda s: min(s, key=tier_rank.get)))
    neurons["tier"] = neurons.type.map(best_tier).fillna("unmatched")
    tiers = ["exact", "alias", "fuzzy", "class", "unmatched"]

    def coverage(mask: pd.Series, label: str, tier_col: str = "tier") -> pd.DataFrame:
        sub = neurons[mask]
        g = sub.groupby(tier_col)
        d = pd.DataFrame({"types": g.type.agg(lambda t: t[t != ""].nunique()), "cells": g.size(), "out_syn": g.out_syn.sum(),
                          "in_syn": g.in_syn_raw.sum()}).reindex(tiers).fillna(0)
        for c in ("types", "cells", "out_syn", "in_syn"):
            d[c] = d[c].astype(int)
            d[c + "_frac"] = d[c] / max(d[c].sum(), 1)
        d.loc["total"] = d.sum(numeric_only=True)
        d.index.name = label
        return d

    def cov_md(d: pd.DataFrame) -> str:
        d = d.copy()
        for c in ("types", "cells", "out_syn", "in_syn"):
            d[c] = d[c].astype(int).map("{:,}".format)
            d[c + "_frac"] = d[c + "_frac"].map("{:.1%}".format)
        return md(d)

    cov_ol = coverage(is_ol, "optic lobe (ol_intrinsic + visual_projection + ol_sensory)")
    cov_sc = {sc: coverage(neurons.superclass == sc, sc) for sc in OL}
    # union with the round-1 receptor table: which MaleCNS types get their FIRST profile from this source
    rec = pd.read_csv(DATA / "receptors_by_type.csv", comment="#")
    r1_types = set(rec.malecns_type[~rec.malecns_type.astype(str).str.startswith("<")])
    r1_best = rec[~rec.malecns_type.astype(str).str.startswith("<")].drop_duplicates("malecns_type").set_index("malecns_type").tier
    neurons["tier_r1"] = neurons.type.map(r1_best).fillna("unmatched")
    neurons["tier_union"] = [a if tier_rank.get(a, 9) <= tier_rank.get(b, 9) else b for a, b in zip(neurons.tier_r1, neurons.tier)]
    cov_r1 = coverage(is_ol, "optic lobe, round-1 receptor table alone (receptors_by_type.csv best tier per type)", "tier_r1")
    cov_union = coverage(is_ol, "optic lobe, round-1 table + kurmangaliyev2020 (best tier per type)", "tier_union")
    new_types = sorted(set(best_tier.index) - r1_types)
    newt = neurons[neurons.type.isin(new_types)].groupby("type").agg(cells=("bodyId", "size"), out_syn=("out_syn", "sum"), in_syn=("in_syn_raw", "sum"))
    newt["tier"] = best_tier.reindex(newt.index)
    newt["source_cluster"] = [", ".join(tmap[tmap.malecns_type == t].source_cluster) for t in newt.index]
    # targets
    tg = neurons[neurons.type.isin(TARGETS)].groupby("type").agg(cells=("bodyId", "size"), in_syn=("in_syn_raw", "sum"), out_syn=("out_syn", "sum")).reindex(TARGETS)
    tg["nt_malecns"] = [neurons.loc[neurons.type == t, "nt"].mode().iat[0] for t in TARGETS]
    tg["in_metadata_type"] = [int((meta.type == t).sum()) for t in TARGETS]
    tg["in_metadata_subtype"] = [int((meta.subtype == t).sum()) for t in TARGETS]
    tg["tier_here"] = best_tier.reindex(TARGETS).fillna("none")
    tg["tier_round1"] = r1_best.reindex(TARGETS).fillna("none")

    # ------------------------------------------------------------------ transmitter cross-check (late pool, frac_expr >= FRAC_ON)
    syn = ["ChAT", "VAChT", "Gad1", "VGAT", "VGlut", "Hdc", "ple", "DAT", "Tdc2", "Tbh", "Trh", "SerT"]
    fl = expr[(expr.metric == "frac_expr") & (expr.time == "late")].set_index("source_cluster")
    xc = []
    for c, sname, ts, tier, _ in MAP:
        r = fl.loc[c, syn]
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
            xc.append(dict(source_cluster=c, malecns_type=t, tier=tier, n_late=int(fl.loc[c, "n_cells"]),
                           malecns_nt=f"{nt.index[0]} ({nt.iat[0]:.0%})", transcript_call="+".join(calls) or "none",
                           **{g: round(float(r[g]), 2) for g in syn}))
    xc = pd.DataFrame(xc)
    short = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu", "histamine": "His", "dopamine": "DA",
             "octopamine": "OA", "serotonin": "5HT", "unknown": "unknown"}
    xc["agree"] = [short[m.split(" ")[0]] in t.split("+") for m, t in zip(xc.malecns_nt, xc.transcript_call)]
    # the same call for the unannotated neuronal clusters (used by the audit's target-type section)
    un = []
    for c in clusters:
        if c in mapped_names or not (c.startswith("N") and unannotated(c)):
            continue
        if c not in fl.index:
            un.append(dict(cluster=c, n_all=int(n_by_cluster[c]), n_late=0, n_96h=0, transcript_call="no cells at 72-96h"))
            continue
        r = fl.loc[c, syn]
        calls = []
        if max(r["ChAT"], r["VAChT"]) >= 0.5: calls.append("ACh")
        if max(r["Gad1"], r["VGAT"]) >= 0.5: calls.append("GABA")
        if r["VGlut"] >= 0.5: calls.append("Glu")
        if r["Hdc"] >= 0.5: calls.append("His")
        un.append(dict(cluster=c, n_all=int(n_by_cluster[c]), n_late=int(fl.loc[c, "n_cells"]), n_96h=int(expr[(expr.metric == "frac_expr") & (expr.time == "96h") & (expr.source_cluster == c)].n_cells.sum()),
                       transcript_call="+".join(calls) or "none", **{g: round(float(r[g]), 2) for g in ["ChAT", "VAChT", "Gad1", "VGAT", "VGlut", "Hdc"]}))
    un = pd.DataFrame(un)
    # sizes: cells per 96h cluster for the annotated types vs MaleCNS cell counts (for the scaling argument)
    n96 = expr[(expr.metric == "frac_expr") & (expr.time == "96h")].set_index("source_cluster").n_cells
    n_all = expr[(expr.metric == "frac_expr") & (expr.time == "all")].set_index("source_cluster").n_cells
    exact_uni = [(c, ts[0]) for c, _, ts, tier, _ in MAP if tier == "exact" and len(ts) == 1]
    ratio = pd.DataFrame({"cluster": [c for c, _ in exact_uni], "malecns_type": [t for _, t in exact_uni],
                          "cells_all": [int(n_all[c]) for c, _ in exact_uni], "cells_96h": [int(n96.get(c, 0)) for c, _ in exact_uni],
                          "malecns_cells": [int(ncell_t[t]) for _, t in exact_uni]})
    ratio["all_per_malecns"] = ratio.cells_all / ratio.malecns_cells
    ratio["p96_per_malecns"] = ratio.cells_96h / ratio.malecns_cells

    # ------------------------------------------------------------------ audit document
    files = [
        ("GSE156455_README.txt", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_README.txt", "authors' description of the files, cellID / sampleID format, 11/02/20 T4c<->T4d label correction"),
        ("GSE156455_features_main.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_features_main.tsv.gz", "17,561 genes (FBgn, symbol, 'Gene Expression'), FlyBase r6.29"),
        ("GSE156455_barcodes_main.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_barcodes_main.tsv.gz", "193,727 cell ids (sampleID_barcode)"),
        ("GSE156455_matrix_main.mtx.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_matrix_main.mtx.gz", f"raw UMI counts, MatrixMarket coordinate integer, 17,561 x 193,727, {nnz:,} non-zeros"),
        ("GSE156455_metadata_main.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_metadata_main.tsv.gz", "per cell: set (W1118 / DGRP), rep, trep, genotype, time (24h..96h), class (N / G / C / R), type, subtype (2020-11-04 corrected version)"),
        ("GSE156455_tsne_main.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_tsne_main.tsv.gz", "tSNE embedding of the main dataset (not used)"),
        ("GSE156455_features_early.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_features_early.tsv.gz", "gene list of the early (0-24 h APF) dataset (identical bytes to features_main)"),
        ("GSE156455_barcodes_early.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_barcodes_early.tsv.gz", "28,633 cell ids of the early dataset (not used: no synapses at 0-24 h APF)"),
        ("GSE156455_metadata_early.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/suppl/GSE156455_metadata_early.tsv.gz", "early-dataset metadata (0h / 12h / 24h APF; 15,510 cells in unannotated 'x' clusters)"),
        ("GSE156455_family.soft.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE156nnn/GSE156455/soft/GSE156455_family.soft.gz", "GEO series / sample records (36 10x v3 libraries: 16 DGRP_All, 20 W1118 at 24h / 48h / 72h / 'Adult' = 96h APF)"),
    ]
    not_downloaded = [
        ("GSE156455_matrix_early.mtx.gz", "152 MB", "0-24 h APF counts; no synapses / receptors of interest at that stage; not used"),
        ("Neuron supplementary tables (mmc*.xlsx)", "n/a", "the article (doi:10.1016/j.neuron.2020.10.006) and its Cell Press supplement are paywalled (HTTP 403 from cell.com / sciencedirect on 2026-09-11; no PMC deposit: Europe PMC lists PMID 33125872 with no PMCID); the GEO metadata carries the cluster annotation used here"),
    ]
    n_annot_types = int(sum(1 for t in meta.type.unique() if not (re.match(r"^[NGCR]\d+$", t) and t[0] == cls_of.get(t, meta.loc[meta.type == t, "class"].iat[0]))))
    n_unannot_N = int(sum(1 for c in clusters if c.startswith("N") and unannotated(c)))
    n_ol_unmatched_types = int(neurons[is_ol & (neurons.tier == "unmatched") & (neurons.type != "")].type.nunique())
    tg_md = tg.copy()
    for c in ("cells", "in_syn", "out_syn"):
        tg_md[c] = tg_md[c].map("{:,.0f}".format)
    n_tiers = tmap[tmap.tier != "unmatched"].groupby("tier").agg(clusters=("source_cluster", "nunique"), malecns_types=("malecns_type", "nunique")).reindex(["exact", "alias", "fuzzy", "class"]).fillna(0).astype(int)
    unm = tmap[tmap.tier == "unmatched"]
    ach_un = un[un.transcript_call == "ACh"]
    out = []
    out.append("# Receptor / transmitter expression sources, round 2: Kurmangaliyev et al. 2020 (pupal optic lobe scRNA-seq) and the small-object-pathway profile search\n")
    out.append(f"Generated by `scripts/build_kurmangaliyev2020_tables.py` on {date.today().isoformat()}. Source key `kurmangaliyev2020`. "
               "Task `fix:profiles` of round 2 (`docs/NT_INTEGRATION.md` section 7): find a receptor profile for Tm5Y, TmY21, TmY13, LC11, Y3, Li19, Tm32, "
               "which have 0 % profiled input in all four round-1 sources. Part (B) of that task (Özel 2021's unannotated clusters) is "
               "appended to `docs/audits/receptor_sources_ozel2021.md`.\n")
    out.append("## 0. Outcome for the seven target types\n")
    out.append("**None of the seven types is annotated in this source** (`type` and `subtype` columns of the GEO metadata contain no Tm5Y / TmY21 / TmY13 / LC11 / Y3 / Li19 / Tm32 label), "
               f"so no profile at any tier. The source annotates {n_annot_types} `type` labels ({n_annot_types - 2} neuronal + R1.6 / R7.8; T4.T5, Dm3, Tm9 carry subtypes) and leaves "
               f"{n_unannot_N} neuronal clusters (N##, {int(meta.type.str.match(r'^N\d+$').sum()):,} cells) unannotated; the seven types "
               f"are among the {n_ol_unmatched_types} MaleCNS optic types this source leaves unmatched. Together with part (B) this closes hypothesis (a): the small-object pathway's receptor "
               "profile is **outside the receptor route** with the atlases available (Özel 2021, Davis 2020, Kurmangaliyev 2020, Davie 2018, FCA 2022, Nern 2025).\n")
    out.append(md(tg_md.rename_axis("target type")) + "\n")
    out.append("`in_metadata_type` / `in_metadata_subtype` = cells carrying that label in GSE156455_metadata_main (0 = not annotated); `tier_here` = tier in this source's type map; "
               "`tier_round1` = tier in `receptors_by_type.csv`. Cells / synapses from `cache/` (|W| row sums = input synapses).\n")
    out.append("## 1. Citation and licence\n")
    out.append("Kurmangaliyev, Y. Z., Yoo, J., Valdes-Aleman, J., Sanfilippo, P. & Zipursky, S. L. *Transcriptional programs of circuit assembly in the Drosophila visual system.* "
               "Neuron 108, 1045-1057.e6 (2020). https://doi.org/10.1016/j.neuron.2020.10.006 (PMID 33125872; no PMC deposit).\n")
    out.append("* Data: GEO **GSE156455** (public 2020-10-26, last update 2021-01-25). 36 10x Genomics Chromium v3 libraries, FACS-sorted optic-lobe cells; Cell Ranger 3.1.0 on "
               "FlyBase r6.29; Seurat 3.1.2; DGRP samples demultiplexed by parental genotype (demuxlet). Main dataset 193,727 cells at **24, 36, 48, 60, 72, 84, 96 h APF** "
               "(w1118: 24h / 48h / 72h / 96h libraries; DGRP: all timepoints multiplexed in one pool, 40 lines); early dataset 28,633 cells at 0-24 h APF. "
               "**There is no post-eclosion adult timepoint**: the four libraries titled `W1118_Adult_*` in the GEO SOFT record carry `age: 96h APF` and are the `96h` cells of the metadata (pharate adults).\n"
               f"* Sex: not stated in the GEO record. Checked here from the male-specific lncRNAs roX1 + roX2 (section 3): **{sex_verdict}**; {male_frac:.2%} of cells are male-like (>= 5 roX cp10k), {rox_zero:.1%} have zero roX UMIs.\n"
               "* Licence: NCBI GEO data are publicly accessible for reuse with citation (no licence statement on the record); the Neuron article is Elsevier / Cell Press (paywalled). "
               "Raw files stay in `data/external/kurmangaliyev2020/` (git-ignored); the files under `flyverse/data/` are our per-cluster summary statistics (54 genes) with attribution.\n"
               "* Nomenclature: the authors used the 2020 EM / Fischbach names; 'T4.T5', 'R1.6', 'R7.8', 'Lpi3.4' use '.' for '/' or '-'. Bridges to MaleCNS names are in the evidence column of the type map (Nern 2025 Sup_Table_7 for Tm29, Pm3, Pm4, LPi34, Dm3, Dm8).\n")
    out.append("## 2. Files\n")
    out.append("| file (`data/external/kurmangaliyev2020/`) | URL | bytes | SHA-256 | content |\n|---|---|---|---|---|")
    for fname, url, desc in files:
        p = EXT / fname
        out.append(f"| {fname} | {url} | {p.stat().st_size:,} | `{sha256(p)}` | {desc} |")
    out.append("\nNot downloaded:\n")
    out.append("| file | size | reason |\n|---|---|---|")
    for fname, size, why in not_downloaded:
        out.append(f"| {fname} | {size} | {why} |")
    out.append("\nAll downloads: `curl -sL` from the GEO FTP on 2026-09-11 (FTP listing dated 2020-10-23 .. 2020-11-04). Manifest entry: `flyverse/data/manifest.json` -> `external.kurmangaliyev2020`. "
               "The matrix is 1.0 GB compressed (under the task's 2 GB limit), so it was downloaded and aggregated per annotated cluster on CPU (`stream_counts`, one pass, per-cell counts of the 54 + 6 genes cached in `_percell_54genes.npz`).\n")
    out.append("## 3. What the main dataset contains\n")
    cls = meta.groupby("class").size()
    out.append(f"* {ncell:,} cells; classes N (neurons) {cls['N']:,}, G (glia) {cls['G']:,}, C (central brain) {cls['C']:,}, R (photoreceptors) {cls['R']:,}. "
               f"{meta.type.nunique()} `type` labels, {meta.subtype.nunique()} `subtype` labels; {len(annotated)} annotated cluster names (incl. the 8 T4/T5 and the Dm3a/b, Tm9a/b subtypes and their 3 parent pools), "
               f"{len(unm)} unmatched ({(unm.source_cluster.str.match(r'^N')).sum()} N##, {(unm.source_cluster.str.match(r'^G')).sum()} G##, {(unm.source_cluster.str.match(r'^C')).sum()} C##, {(unm.source_cluster.str.match(r'^R')).sum()} R##).\n"
               f"* Cells per timepoint: " + ", ".join(f"{t} {int((meta.time == t).sum()):,}" for t in TIMES) + f"; median total UMI per cell {np.median(totals):,.0f}.\n")
    out.append("* Sex check: roX1 + roX2 UMI per 10k, per cell; a cell is 'male-like' at >= 5 cp10k (roX1 is a male-specific dosage-compensation lncRNA at hundreds of UMIs per male cell, ~0 in females). Per library:\n")
    sx = sexchk.copy(); sx["median_rox_cp10k"] = sx.median_rox_cp10k.map("{:.1f}".format); sx["frac_male_like"] = sx.frac_male_like.map("{:.2f}".format)
    out.append(md(sx.rename_axis("library")) + "\n")
    sb = sex_by_time.copy(); sb["frac_male_like"] = sb.frac_male_like.map("{:.2f}".format)
    out.append(md(sb.rename_axis("time")) + "\n")
    out.append(f"* Overall {male_frac:.2%} of cells ({int(meta.male_like.sum()):,}) are male-like, {rox_zero:.1%} have zero roX1 + roX2 UMIs, and no library exceeds "
               f"{sexchk.frac_male_like.max():.1%} male-like cells (highest: the W1118 24h libraries): the atlas is **{sex_verdict}** -- like Özel 2021 and unlike MaleCNS (male). "
               "For comparison, male cells in the FCA head / Davie 2018 brain data carry roX1 at tens to hundreds of UMIs per cell, so an unsexed pool would show ~50 % male-like cells.\n")
    out.append("* Cluster sizes vs MaleCNS cell counts (exact, single-type rows): cells in the source per MaleCNS cell of the type, over all timepoints and at 96 h. "
               "This is the scaling one would use to judge whether an unannotated cluster is the right size for a target type (Tm5Y 898 MaleCNS cells, TmY21 372, TmY13 432, LC11 143, Y3 627, Li19 51, Tm32 125):\n")
    rt = ratio.set_index("cluster").copy(); rt["all_per_malecns"] = rt.all_per_malecns.map("{:.2f}".format); rt["p96_per_malecns"] = rt.p96_per_malecns.map("{:.2f}".format)
    out.append(md(rt) + "\n")
    out.append(f"* Median cells per MaleCNS cell: all timepoints {ratio.all_per_malecns.median():.2f} (IQR {ratio.all_per_malecns.quantile(.25):.2f}-{ratio.all_per_malecns.quantile(.75):.2f}), 96 h {ratio.p96_per_malecns.median():.2f}. "
               f"Expected 96 h cluster size for Tm5Y ~{ratio.p96_per_malecns.median() * 898:.0f} cells, TmY21 ~{ratio.p96_per_malecns.median() * 372:.0f}, TmY13 ~{ratio.p96_per_malecns.median() * 432:.0f}, Y3 ~{ratio.p96_per_malecns.median() * 627:.0f}, LC11 ~{ratio.p96_per_malecns.median() * 143:.0f}, Tm32 ~{ratio.p96_per_malecns.median() * 125:.0f}, Li19 ~{ratio.p96_per_malecns.median() * 51:.0f}.\n")
    out.append("## 4. Derived tables\n")
    out.append(f"* `flyverse/data/type_map_kurmangaliyev2020.csv`: {len(tmap)} rows = {tmap.source_cluster.nunique()} clusters; "
               f"{len(tmap[tmap.tier != 'unmatched'])} (cluster, MaleCNS type) pairs over {tmap[tmap.tier != 'unmatched'].malecns_type.nunique()} MaleCNS types; {len(unm)} clusters unmatched.\n")
    out.append(md(n_tiers.rename_axis("tier")) + "\n")
    out.append(f"* `flyverse/data/expression_kurmangaliyev2020.csv`: {len(expr)} rows = {expr.source_cluster.nunique()} clusters x up to 9 time labels (24h..96h, late = 72-96h, all) x 2 metrics "
               f"(`mean_log1p_cp10k` as in expression_ozel2021.csv; `frac_expr` = fraction of cells with >= 1 UMI as in expression_central.csv), 54 genes, all present in r6.29 under the plan's symbols. "
               "Recommended rows for an adult-proxy receptor profile: `time = 96h` when `n_cells >= 30`, else `late`.\n")
    out.append("Tier rules: **exact** = the author annotation is the MaleCNS type name (43 type labels + 8 T4/T5 subtypes); **alias** = documented renaming (R1.6 -> R1-R6; Lpi3.4 -> LPi34, the Davis 2020 'LPi-34' bridge); "
               "**fuzzy** = Tm29 (MaleCNS Tm29 = FlyWire Tm5d / CB3851, the 2020 name's correspondence is not evidenced -- the round-1 skeptic's correction) and Pm3 (Pm3 = CB3856 / Pm09, no FlyWire 'Pm3'); "
               "**class** = R7.8, Dm8, Dm3 / Dm3a / Dm3b (subtype letters not transferable), Tm9a / Tm9b (subdivision), T4.T5 (parent pool). Every row keeps the source label and the bridge it rests on. "
               "The annotation method itself (which markers / reference transcriptomes the authors used per cluster) could not be read: the article is paywalled and GEO ships no annotation notes; the labels are taken as the authors' calls.\n")
    out.append("## 5. Coverage of MaleCNS (best tier per type; cells and synapses from `cache/`, |W| column sums = output synapses, row sums = input synapses)\n")
    out.append(cov_md(cov_ol) + "\n")
    for sc in OL:
        out.append(cov_md(cov_sc[sc]) + "\n")
    out.append("### 5a. What this source adds to the round-1 receptor table (optic lobe)\n")
    out.append(cov_md(cov_r1) + "\n")
    out.append(cov_md(cov_union) + "\n")
    if len(newt):
        nt2 = newt.copy()
        for c in ("cells", "out_syn", "in_syn"):
            nt2[c] = nt2[c].map("{:,.0f}".format)
        out.append(f"* MaleCNS types whose first profile comes from this source ({len(newt)}): \n\n" + md(nt2.rename_axis("type")) + "\n")
    else:
        out.append("* No MaleCNS type gets its first profile from this source: every mapped type already has a round-1 profile (Özel 2021 / Davis 2020).\n")
    d_c = int(cov_union.loc["unmatched", "cells"] - cov_r1.loc["unmatched", "cells"]); d_s = int(cov_union.loc["unmatched", "in_syn"] - cov_r1.loc["unmatched", "in_syn"])
    out.append(f"* Optic-lobe cells left unprofiled: round 1 {int(cov_r1.loc['unmatched', 'cells']):,} ({cov_r1.loc['unmatched', 'cells_frac']:.1%}) -> with this source {int(cov_union.loc['unmatched', 'cells']):,} ({cov_union.loc['unmatched', 'cells_frac']:.1%}), "
               f"change {d_c:+,}; unprofiled input synapses {int(cov_r1.loc['unmatched', 'in_syn']):,} ({cov_r1.loc['unmatched', 'in_syn_frac']:.1%}) -> {int(cov_union.loc['unmatched', 'in_syn']):,} ({cov_union.loc['unmatched', 'in_syn_frac']:.1%}), change {d_s:+,}. "
               f"The value of the source is therefore a second, independent (pupal, {sex_verdict.split(' (')[0]}, 10x v3) measurement of already-profiled types, plus the per-subtype T4a-d / T5a-d resolution that Özel's adult tables lack.\n")
    improved = neurons[is_ol & (neurons.tier_union != neurons.tier_r1) & (neurons.type != "")].groupby("type").agg(
        cells=("bodyId", "size"), in_syn=("in_syn_raw", "sum"), tier_round1=("tier_r1", "first"), tier_here=("tier", "first"))
    if len(improved):
        im = improved.copy(); im["cells"] = im.cells.map("{:,}".format); im["in_syn"] = im.in_syn.map("{:,.0f}".format)
        out.append(f"* Types whose best tier improves with this source ({len(improved)} types, {int(improved.cells.sum()):,} cells, {int(improved.in_syn.sum()):,} input synapses):\n\n" + md(im.rename_axis("type")) + "\n")
    out.append("## 6. Transmitter cross-check on the mapped clusters (late pool 72-96 h; `frac_expr` = fraction of cells with >= 1 UMI; `transcript_call` = ACh if max(ChAT, VAChT) >= 0.5, GABA if max(Gad1, VGAT) >= 0.5, Glu if VGlut >= 0.5, His if Hdc >= 0.5, DA / OA / 5HT if both genes of the pair >= 0.5)\n")
    out.append(md(xc.set_index("source_cluster"), "{:.2f}") + "\n")
    dis = xc[~xc.agree]
    out.append(f"* {int(xc.agree.sum())} of {len(xc)} (cluster, type) pairs agree with the model's label; disagreements ({len(dis)}): "
               + "; ".join(f"{r.malecns_type} [{r.source_cluster}] model {r.malecns_nt} vs transcript {r.transcript_call}" for r in dis.itertuples()) + ".\n")
    out.append("## 7. The unannotated neuronal clusters and the target types\n")
    out.append(f"* {len(un)} N## clusters, {int(un.n_all.sum()):,} cells. Transmitter call (late pool, same rule as section 6): "
               + ", ".join(f"{k} {v}" for k, v in un.transcript_call.value_counts().items()) + ".\n")
    out.append(f"* The five cholinergic targets (Tm5Y, TmY21, TmY13, LC11, Y3; Nern 2025 and MaleCNS: ACh) could only be among the {len(ach_un)} ACh-calling N## clusters; Li19 (GABA) among the "
               f"{int((un.transcript_call == 'GABA').sum())} GABA clusters; Tm32 (Glu) among the {int((un.transcript_call == 'Glu').sum())} Glu clusters. No gene-level marker for any of the seven exists in the sources at hand "
               "(Nern 2025 SI: anatomy / connectivity / split-GAL4 lines only, 0 gene mentions in the 70-page catalogue; Davis 2020 supp. 1 has no driver for them; the Özel cluster-marker table describes Özel clusters, not types), "
               "so no N## cluster can be assigned to a target type -- listed here for a future marker-based match only:\n")
    out.append(md(un.set_index("cluster"), "{:.2f}") + "\n")
    out.append("## 8. Caveats\n")
    out.append("* Developmental stage: pupal (24-96 h APF). Receptor genes are still being switched on through 60-96 h APF (Kurmangaliyev 2020's central finding is the late, coordinated pan-neuronal synaptic program); "
               "the 96 h rows are pharate adults, ~1 day before the 1-3 day adults of Özel 2021 and the 5-day-plus heads of Davis 2020. Treat 96 h profiles as adult proxies, not adult measurements.\n"
               f"* Sex: {sex_verdict} (section 3); MaleCNS is male.\n"
               "* Cluster purity and the authors' annotation basis are not verifiable here (article paywalled); the T4c / T4d swap in the original release was corrected by the authors (README 11/02/20) and the corrected metadata is used.\n"
               "* Small 96 h clusters (Dm4, Dm1, LC10b, LPC1, LPLC1, Pm3, Pm4 ...) have < 30 cells at that timepoint; use the `late` rows for them.\n"
               "* No author-provided per-cluster table: the per-cluster numbers are ours (one streaming pass, `stream_counts`), cached per cell in `data/external/kurmangaliyev2020/_percell_54genes.npz` (keyed on the matrix SHA-256).\n")
    AUDIT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {AUDIT}")
    print(tg)
    print(cov_ol)
    print(cov_union)
    print(newt)
    print(xc[["source_cluster", "malecns_type", "tier", "n_late", "malecns_nt", "transcript_call", "agree"]].to_string())
    print(sex_by_time)


if __name__ == "__main__":
    main()
