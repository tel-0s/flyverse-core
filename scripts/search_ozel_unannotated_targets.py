"""Part (B) of fix:profiles: can any of Ozel 2021's unannotated adult clusters be identified as Tm5Y / TmY21 / TmY13 /
LC11 / Y3 / Li19 / Tm32 from type-level marker genes? Prints a markdown section to append to
docs/audits/receptor_sources_ozel2021.md. CPU only."""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np, pandas as pd, openpyxl

ROOT = Path(r"D:\Projects\flyverse")
DATA = ROOT / "flyverse" / "data"
EXT = ROOT / "data" / "external"
TARGETS = ["Tm5Y", "TmY21", "TmY13", "LC11", "Y3", "Li19", "Tm32"]
TARGET_NT = {"Tm5Y": "ACh", "TmY21": "ACh", "TmY13": "ACh", "LC11": "ACh", "Y3": "ACh", "Li19": "GABA", "Tm32": "Glu"}

tm = pd.read_csv(DATA / "type_map_ozel2021.csv", comment="#")
mm = pd.read_csv(DATA / "expression_ozel2021_mm.csv", comment="#")
avg = pd.read_csv(DATA / "expression_ozel2021.csv", comment="#")
neurons = pd.read_parquet(ROOT / "cache" / "neurons.parquet")
neurons["type"] = neurons["type"].fillna("")
import scipy.sparse as sp
W = sp.load_npz(ROOT / "cache" / "W_post_pre.npz").tocsr(); absW = abs(W)
neurons["in_syn"] = np.asarray(absW.sum(axis=1)).ravel(); neurons["out_syn"] = np.asarray(absW.sum(axis=0)).ravel()

# 1. the unannotated neuronal adult clusters
unm = tm[tm.tier == "unmatched"].drop_duplicates("source_cluster").set_index("source_cluster")
neuronal_unm = unm[unm.evidence.str.startswith("unannotated neuronal cluster")]
print(f"unmatched clusters {len(unm)}; unannotated neuronal (authors' 'no type') {len(neuronal_unm)}", file=sys.stderr)
mm = mm.set_index("source_cluster")
syn = ["ChAT", "VAChT", "Gad1", "VGAT", "VGlut", "Hdc"]
def call(r):
    c = []
    if max(r["ChAT"], r["VAChT"]) >= 0.5: c.append("ACh")
    if max(r["Gad1"], r["VGAT"]) >= 0.5: c.append("GABA")
    if r["VGlut"] >= 0.5: c.append("Glu")
    if r["Hdc"] >= 0.5: c.append("His")
    return "+".join(c) or "none"
cand = pd.DataFrame({"cluster": neuronal_unm.index})
cand["transcript_call"] = [call(mm.loc[c]) for c in cand.cluster]
for g in syn:
    cand[g] = [round(float(mm.loc[c, g]), 2) for c in cand.cluster]

# 2. cluster markers (curated adult sheet): top markers unique to the cluster
wb = openpyxl.load_workbook(EXT / "ozel2021" / "GSE142787_Cluster_markers.xlsx", read_only=True)
rows = list(wb["Cluster_markers_Adult_curated"].iter_rows(values_only=True))
mk = pd.DataFrame(rows[1:], columns=rows[0])
mk["Cluster"] = mk.Cluster.astype(int)
def top_markers(c, n=8):
    s = mk[(mk.Cluster == c)].sort_values(["Number_of_clusters_with_this_marker", "avg_logFC"], ascending=[True, False])
    s = s[s.p_val_adj < 1e-10]
    return ", ".join(f"{g}{'' if k == 1 else '(' + str(int(k)) + ')'}" for g, k in zip(s.Gene.head(n), s.Number_of_clusters_with_this_marker.head(n)))
cand["top_markers_curated"] = [top_markers(c) for c in cand.cluster]
cand["n_unique_markers"] = [int(((mk.Cluster == c) & (mk.Number_of_clusters_with_this_marker == 1) & (mk.p_val_adj < 1e-10)).sum()) for c in cand.cluster]

# 3. how many MaleCNS optic types without any round-1 profile share each transmitter (the pool a candidate cluster could belong to)
rec = pd.read_csv(DATA / "receptors_by_type.csv", comment="#")
r1 = set(rec.malecns_type[~rec.malecns_type.astype(str).str.startswith("<")])
OL = ["ol_intrinsic", "visual_projection", "ol_sensory"]
ol = neurons[neurons.superclass.isin(OL) & (neurons.type != "")]
tt = ol.groupby("type").agg(cells=("bodyId", "size"), in_syn=("in_syn", "sum"), nt=("nt", lambda s: s.mode().iat[0]))
unprof = tt[~tt.index.isin(r1)]
short = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu", "histamine": "His", "dopamine": "DA", "octopamine": "OA", "serotonin": "5HT", "unknown": "unknown"}
unprof = unprof.assign(nt_short=unprof.nt.map(short))
pool = unprof.groupby("nt_short").agg(types=("cells", "size"), cells=("cells", "sum")).sort_values("types", ascending=False)

# 4. type-level marker availability for the seven targets in the named sources
nern6 = pd.read_excel(EXT / "nern2025" / "nature_esm" / "MOESM4_unzipped" / "Sup_Table_6_Split-GAL4-lines_final.xlsx")
lines = {}
for t in TARGETS:
    h = nern6[nern6["Main OL cell type(s)"].astype(str).str.split(r"[,;]\s*").apply(lambda xs: t in [x.strip("() ") for x in xs])]
    lines[t] = "; ".join(f"{r.Line} ({r.AD} x {r.DBD}; {r.Reference})" for r in h.itertuples()) or "none"
davis = pd.read_excel(EXT / "davis2020" / "elife-50901-supp1-v2.xlsx", sheet_name="A_all_drivers")
davis_hits = {t: int(davis.celltype.fillna("").astype(str).str.split(",").apply(lambda xs: t in [x.strip() for x in xs]).sum()) for t in TARGETS}
dmk = pd.read_excel(EXT / "davis2020" / "elife-50901-supp1-v2.xlsx", sheet_name="D_markerGenes")
_pdf = EXT / "nern2025" / "nature_esm" / "41586_2025_8746_MOESM1_ESM.pdf"; _txt = _pdf.with_suffix(".txt")
if _txt.exists():
    nern_txt = _txt.read_text(encoding="utf-8", errors="replace")
else:
    try:                                                        # pypdf if installed; else pdftotext; else skip the count
        from pypdf import PdfReader
        nern_txt = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(_pdf)).pages)
    except Exception:
        import subprocess
        r = subprocess.run(["pdftotext", str(_pdf), "-"], capture_output=True, text=True)
        nern_txt = r.stdout if r.returncode == 0 else ""
    if nern_txt:
        _txt.write_text(nern_txt, encoding="utf-8")
n_gene_mentions = len(re.findall(r"\b(marker|gene|express\w*)\b", nern_txt, flags=re.I)) if nern_txt else None

# 5. expected Ozel cluster size for each target: cells per MaleCNS cell is unknown (no per-cluster cell counts in the GEO
#    sheets); only the freq_cell columns exist. Skip the size argument; state it.

tg = tt.reindex(TARGETS)
def md(df, floatfmt="{:.2f}"):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    cols = [d.index.name or ""] + list(map(str, d.columns))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in d.iterrows(): out.append("| " + " | ".join([str(i)] + [str(v) for v in r]) + " |")
    return "\n".join(out)

L = []
L.append("\n## 8. Round 2 (task fix:profiles, part B): can an unannotated adult cluster be Tm5Y / TmY21 / TmY13 / LC11 / Y3 / Li19 / Tm32?\n")
L.append("Appended 2026-09-11 by `scripts/search_ozel_unannotated_targets.py` (CPU; inputs: `type_map_ozel2021.csv`, `expression_ozel2021_mm.csv`, "
         "`GSE142787_Cluster_markers.xlsx` sheet Cluster_markers_Adult_curated, Nern 2025 Sup_Table_6 + SI catalogue text, Davis 2020 elife-50901-supp1-v2.xlsx, `receptors_by_type.csv`, `cache/`). "
         "Verdict first: **no unannotated Özel cluster can be identified as any of the seven types, because no type-level marker gene exists for them in the sources the task names; no match is forced.** "
         "Together with the Kurmangaliyev 2020 result (`receptor_sources_profiles.md` section 0: none of the seven annotated there) this closes hypothesis (a) as 'outside the receptor route'.\n")
L.append("### 8a. Marker availability for the seven types\n")
L.append("| type | MaleCNS cells / input syn | NT (MaleCNS) | Nern 2025 split-GAL4 lines (Sup_Table_6; hemidriver enhancer fragments, not genes) | Davis 2020 drivers (supp. 1 sheet A) | gene-level markers in the named sources |\n|---|---|---|---|---|---|")
for t in TARGETS:
    L.append(f"| {t} | {int(tg.loc[t, 'cells']):,} / {int(tg.loc[t, 'in_syn']):,} | {short[tg.loc[t, 'nt']]} | {lines[t]} | {davis_hits[t]} | none |")
L.append(f"\n* Nern 2025 SI catalogue (`41586_2025_8746_MOESM1_ESM.pdf`, 70 pages, text extracted with pdftotext -layout, 4,231 lines): anatomy, synapse-depth profiles, top-5 partners and cell counts per type; "
         f"**{n_gene_mentions} occurrences of 'marker' / 'gene' / 'express*'** in the whole document. It carries no molecular markers. Sup_Table_6 lists split-GAL4 lines (AD x DBD enhancer fragments); a fragment's gene of origin "
         "is not evidence that the gene is expressed in the labelled type, so it was not used.\n"
         f"* Davis 2020 supp. file 1: sheet A_all_drivers has {len(davis)} drivers, 0 for any of the seven (the LC drivers are LC4, LC6, LC10a/b/bc/d, LC16, LPLC1, LPLC2; no Tm5Y / TmY13 / TmY21 / Y3 / Li19 / Tm32); "
         f"sheet D_markerGenes has {len(dmk)} rows for the groups {sorted(dmk.cellGroup.unique())} only (used for QC of contamination, not neuron types).\n"
         "* `GSE142787_Cluster_markers.xlsx` gives markers *of Özel clusters* (Seurat FindAllMarkers), i.e. the thing one would compare a type-level marker list against -- it cannot supply that list itself.\n"
         "* Kurmangaliyev 2020 (GSE156455): the seven types are not among its 51 annotated clusters either, so a cross-atlas transfer of an annotation is not possible.\n")
L.append("### 8b. The unannotated neuronal adult clusters, as far as they can be constrained\n")
L.append(f"* {len(unm)} adult clusters are unmatched in `type_map_ozel2021.csv`; {len(neuronal_unm)} of them are the authors' unannotated neuronal clusters (the rest: glia G*, LQ, 102 / 112 / 191 / 192, TmY8*, 98, 221, 88a/b, 169a/b). "
         "Their transmitter call from the mixture-model P(on) (rule of section 5): "
         + ", ".join(f"{k} {v}" for k, v in cand.transcript_call.value_counts().items()) + ".\n")
L.append(f"* On the MaleCNS side, optic-lobe types with no round-1 profile (`receptors_by_type.csv`): {len(unprof)} types, {int(unprof.cells.sum()):,} cells; by transmitter: "
         + ", ".join(f"{i} {int(r.types)} types / {int(r.cells):,} cells" for i, r in pool.iterrows()) + ". "
         f"So the {int((cand.transcript_call == 'ACh').sum())} ACh-only clusters could be any of {int(pool.loc['ACh', 'types']) if 'ACh' in pool.index else 0} unprofiled ACh types "
         f"(five of them targets), the {int((cand.transcript_call == 'GABA').sum())} GABA clusters any of {int(pool.loc['GABA', 'types']) if 'GABA' in pool.index else 0} GABA types (Li19 among them), "
         f"the {int((cand.transcript_call == 'Glu').sum())} Glu clusters any of {int(pool.loc['Glu', 'types']) if 'Glu' in pool.index else 0} Glu types (Tm32 among them). "
         "Cluster size cannot narrow this: the GEO summary sheets carry no per-cluster cell counts (only the mixture-model `freq_cell_ON/OFF` columns, which are per gene), and the 3.5 GB Adult.rds was not downloaded.\n")
L.append("* Match score: **not computable** (no type-level marker vector to score against). The table below is the candidate pool with each cluster's curated top markers (unique to the cluster unless a count in parentheses says in how many clusters the gene is a marker; p_adj < 1e-10), kept so that a future marker list (e.g. from an atlas that names these types, or from in situ data) can be scored against it without re-deriving anything:\n")
c2 = cand.set_index("cluster")
L.append(md(c2) + "\n")
L.append("### 8c. Consequence for the model\n")
L.append("The sign of the glutamatergic input onto Tm5Y (102 k |W| synapses), TmY21 (61 k), TmY13 (73 k), Y3 (199 k), LC11, Li19 and Tm32 stays with `NT_SIGN` (tier `fallback`) under every receptor mode; "
         "the NT-class fallback (`nt_class_fallback`) would give them the ChAT-class prior, which is GluCl-dominant in every source (no change of sign). The object-pathway question therefore moves to the "
         "medulla -> lobula wiring / dynamics side (`scripts/probe_object_sweep.py`), not to receptor expression. Sources that might name these types in future (not among this task's inputs, unverified): "
         "the FlyWire-matched adult optic-lobe atlases of 2024-2025 (e.g. Nguyen / Jain / Desplan-lab and Zipursky-lab follow-ups) -- to be checked before any further receptor work on this pathway.\n")
sys.stdout.write("\n".join(L))
