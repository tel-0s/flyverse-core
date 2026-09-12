# Receptor / transmitter sources: central brain (Davie 2018, Fly Cell Atlas 2022)

Generated 2026-09-11 (NT-integration workflow, source key `central`; `docs/NT_INTEGRATION.md` step 1-2, item 7).
Round 2 (fix:central) applied the verify:tables:central corrections of `docs/audits/receptor_verification.md`; section 7
lists every number that changed. Deliverables: `flyverse/data/type_map_central.csv` (source cluster -> MaleCNS type,
with tier and evidence), `flyverse/data/expression_central.csv` (per-cluster expression of the 55 transmitter / receptor
genes), this file.

**Build (all CPU, from the repo root, in this order; the raw inputs are git-ignored under `data/external/central/` and
are fetched with `python scripts/fetch_data.py --external central`):**

```
PYTHONIOENCODING=utf-8 python scripts/build_central_agg_davie.py   # Davie MEX -> data/external/central/derived/davie_cluster_expr.parquet (5 s)
PYTHONIOENCODING=utf-8 python scripts/build_central_agg_fca.py     # FCA loom  -> data/external/central/derived/fca_cluster_expr.parquet (~20 s)
PYTHONIOENCODING=utf-8 python scripts/build_central_map.py         # -> flyverse/data/type_map_central.csv, expression_central.csv,
                                                                   #    out/coverage_central.md, out/nt_crosscheck_central.md, out/central_summary.json
```

`build_central_map.py` needs `cache/neurons.parquet` + `cache/W_post_pre.npz` (built by `flyverse.connectome.load()`) and, for
the sign-0-inclusive `raw` synapse columns, the MaleCNS weights feather at `flyverse.connectome.DATA_DIR` (env `FLYVERSE_DATA`).
Sections 3-5 below are the script's `out/` reports verbatim; the round-2 rebuild reproduced the round-1 expression table
(560 rows x 54 genes, byte-identical data rows) and the round-1 map rows (1,150 rows, identical in every column except
`evidence`, which changed on 279 rows of 15 labels).

Everything below is CPU-only (downloads, pandas, structural connectome counts); no simulation was run.

## 1. Files acquired (`data/external/central/`, git-ignored)

| file | source / URL | accession | bytes | SHA-256 |
|---|---|---|---|---|
| `s_fca_biohub_head_10x.loom` | flycellatlas.org "Head, 10x, Stringent, Loom": https://cloud.flycellatlas.org/index.php/s/jtH5fGwLqtiSRSD/download | FCA head 10x stringent (raw reads: ArrayExpress E-MTAB-10519) | 742,982,593 | `95de36ec70fe399c40356820cd1d1bbe0df64aefe51e760c50fde0783d8c654e` |
| `GSE107451_DGRP-551_w1118_WholeBrain_57k_0d_1d_3d_6d_9d_15d_30d_50d_10X_DGEM_MEX.mtx.tsv.tar.gz` | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE107nnn/GSE107451/suppl/ | GEO GSE107451 | 219,256,938 | `5beaf1378736a41b8668cf8ae1816900b491aeba44a0aeeca71480d80d5d7ca6` |
| `davie2018_57k_mex/matrix.mtx` (extracted) | (from the tarball) 17,473 genes x 56,902 cells, 70,776,454 nonzeros, integer UMI | GEO GSE107451 | 941,835,490 | `dc64ae2701c9f6f03719bff00d6678191aac308f8d9c2a7d327153e49b0688ea` |
| `davie2018_57k_mex/genes.tsv` | (from the tarball) FBgn + symbol | GEO GSE107451 | 341,655 | `24e7dd761c6f78cb24efaebe5bc32d0d3aaa08799d32923cbc4a382d23261d47` |
| `davie2018_57k_mex/barcodes.tsv` | (from the tarball) | GEO GSE107451 | 1,750,942 | `d5bef617a9bfcd98cda8da3216b78843491c758e596dc381c4a27fe2ba5ecb46` |
| `davie2018_57k_mex/annotation.tsv` | (from the tarball) sample / technology / replicate / line / age per cell | GEO GSE107451 | 3,103,614 | `45c3f2e6862b34d6c1819f5fd13c31fe37fd240e17b14fd49a7851348eb51542` |
| `GSE107451_DGRP-551_w1118_WholeBrain_57k_Metadata.tsv.gz` | same GEO directory; per-cell `annotation` (the paper's final cluster labels), age, sex, genotype, Seurat resolutions | GEO GSE107451 | 4,642,833 | `bb0e13e263d182b819df3e4bebfde77e267c070a30ec5d68ba19bd9a19c46cc9` |
| `GSE107451_DGRP-551_w1118_WholeBrain_57k_Metadata_README.txt` | same | GEO GSE107451 | 569 | `847d1f0bea6a527b20f0e4406a5da672b775aed62e7fdf9ca36f206693157723` |
| `davie2018_mmc2.xlsx` (Table S2: cluster annotation, method, references; res. 2 / res. 8 / sub-clustering) | https://ars.els-cdn.com/content/image/1-s2.0-S0092867418307207-mmc2.xlsx | Cell supplement | 140,827 | `0d4743d8c6e3637fdf5529e5f021e75e4cf5db6a3f144b1c8ac7b9d2413fdcc7` |
| `davie2018_mmc3.xlsx` (Table S3: Seurat marker genes per res.2 cluster) | ...-mmc3.xlsx | Cell supplement | 959,066 | `72cf6a8f974d8e998c11c37537cd0cef7abf39382b93ce221a75ebb9faacf5cc` |
| `davie2018_mmc4.xlsx` (Table S4: SCENIC regulons) | ...-mmc4.xlsx | Cell supplement | 141,916 | `f8d784445c98f3c7a13d2fc0f20bfd3700a78998d1c13473dab3a5fb5ed80d92` |
| `davie2018_mmc5.xlsx` (Table S5: primers) | ...-mmc5.xlsx | Cell supplement | 9,832 | `dfd68a86d636b96000cd72a2b55dec2fda11a9e5dd6a971c974b3d26aa67edfd` |

Derived intermediates (git-ignored, rebuilt by the aggregators): `data/external/central/derived/davie_cluster_expr.parquet`
(348 rows = 116 labels x 3 metrics, 17,473 gene columns) and `fca_cluster_expr.parquet` (13,056 gene columns; 82 labels x 3
metrics x {all, male}).

Not downloaded: the FCA head 10x stringent **h5ad** (the loom carries the same matrix and annotations; no URL was recorded
for it and the FCA share serves no Content-Length, so its size is not claimed here), the FCA cross-tissue neuron h5ad and
loom, the FCA raw h5ad, the Davie 157k-cell matrix (the 57k set is the paper's filtered atlas), Davie Table S1
(`mmc1.xlsx` returns 404 on the CDN; not needed). Publisher pages (science.org, cell.com) return 403 to non-browser
clients; statements below come from the PMC copies (PMC8944923, PMC6086935).

**Citations and licences.**
Davie K, Janssens J, Koldere D, De Waegeneer M, Pech U, Kreft L, Aibar S, Makhzami S, Christiaens V, Bravo Gonzalez-Blas C,
Poovathingal S, Hulselmans G, Spanier KI, Moerman T, Vanspauwen B, Geurs S, Voet T, Lammertyn J, Thienpont B, Liu S,
Konstantinides N, Fiers M, Verstreken P, Aerts S. *A Single-Cell Transcriptome Atlas of the Aging Drosophila Brain.*
Cell 2018;174(4):982-998.e20, doi:10.1016/j.cell.2018.05.057. Open access, **CC BY-NC-ND 4.0** (paper and supplements);
GEO GSE107451 data carry no licence restriction. Whole brains (not heads) of DGRP-551 and w1118, 8 ages 0-50 d, both sexes.
Li H, Janssens J, De Waegeneer M, Kolluru SS, Davie K, Gardeux V, Saelens W, David FPA, Brbic M, Spanier K, Leskovec J,
McLaughlin CN, Xie Q, Jones RC, Brueckner K, Shim J, Tattikota SG, Schnorrer F, Rust K, Nystul TG, Carvalho-Santos Z,
Ribeiro C, Pal S, Mahadevaraju S, Przytycka TM, Allen AM, Goodwin SF, Berry CW, Fuller MT, White-Cooper H, Matunis EL,
DiNardo S, Galenza A, O'Brien LE, Dow JAT, FCA Consortium, Jasper H, Oliver B, Perrimon N, Deplancke B, Quake SR,
Luo L, Aerts S. *Fly Cell Atlas: A single-nucleus transcriptomic atlas of the adult fruit fly.* Science
2022;375(6584):eabk2432, doi:10.1126/science.abk2432. The FCA download page states no licence for the processed loom
(checked 2026-09-11; data are "freely available" via SCope/ASAP; raw reads E-MTAB-10519 / E-MTAB-10628 are open). We
redistribute only per-cluster summary statistics (means / fractions), not the matrix. Heads (brain + retina + optic lobe +
antennae + fat + muscle + epithelia), 5-day-old flies, both sexes (male 47,409 / female 49,105 / mix 4,013 nuclei).

## 2. What the sources contain and how the tables were derived

*Davie 2018*: 56,902 cells x 17,473 genes (10x, whole brain, filtered atlas). Cluster key = the GEO metadata `annotation`
column = the paper's final labels: 75 named labels (Table S2 res. 2 clusters annotated by literature markers and/or mapping
to Konstantinides 2018 optic-lobe transcriptomes, res. 8 clusters IPC / DCN / Capa / Gr43a / CCAP, and the sub-clusterings
of the lamina, PN, dopaminergic, OA/TA and neuropeptide clusters) plus 41 numeric labels = res. 2 clusters left
"Unannotated" in Table S2 (35,818 cells, 63 % of the atlas). All 55 requested genes are present (KaiR1D as `CG3822`,
Octalpha2R as `CG18208`).

*FCA 2022 head*: 100,527 nuclei x 13,056 genes (the stringent loom is gene-filtered; raw UMI counts in `/matrix`).
Cluster key = `col_attrs/annotation`: 82 labels (FBbt terms), of which "unannotated" is 44,906 nuclei (44.7 %). 54 of the 55
genes present (`mAChR-C` is absent from the retained gene set -> NaN). The head atlas is dominated by optic-lobe and
sensory types; the only central-brain labels are the three KC classes + "Kenyon cell", "dopaminergic PAM neuron",
"dopaminergic neuron", "octopaminergic/tyraminergic neuron", "antennal lobe projection neuron", "Poxn neuron",
ORNs, JO neurons. There are **no** FCA head labels for ring neurons, clock neurons, MBONs, LH or CX columnar types.

*Expression units* (`expression_central.csv`): per cell, UMI counts / cell total x 1e4 (cp10k; Davie total over all
17,473 genes, FCA over the 13,056 retained genes -- so FCA values are inflated relative to full-transcriptome cp10k and
the two sources are comparable **within** source only), then per cluster `mean_log1p_cp10k` = mean of log1p(cp10k)
and `frac_expr` = fraction of cells with >= 1 UMI. Davie: all ages and both sexes pooled (the paper reports that RNA
content, not identity, changes with age). FCA: `sex_subset` = `all` and `male` (male-only means differ from pooled by
< 0.15 log units for every receptor gene in the gamma-KC row; sex is not a large effect at this resolution).

*Mapping tiers* (`type_map_central.csv`, one row per (source_name, malecns_type)): `exact` -- the source label (or its
FBbt type token) is a MaleCNS type name; `alias` -- a documented synonym of one type (Lawf1/2, Tm3a -> Tm3, outer
photoreceptor -> R1-R6, Gr21a/63a ORN -> ORN_V); `fuzzy` -- the label names a family that MaleCNS splits into several
types, selected by the name rule stored in `rule` (e.g. `G-KC` -> `^KCg`, `PAM` -> `^PAM\d\d`, `T4/T5` -> `^T[45][a-d]`,
`adPN` -> `_adPN$`, `Pm1/Pm2` -> Pm1|Pm2a|Pm2b, `DCN` -> `^LC14`); `class` -- the label is a cell class, the MaleCNS
side is a `class` / `subclass` / consensus-`nt` selection or a curated list (MBON, ALPN, Kenyon_Cell, ORN_*, JO-*,
visual; dopaminergic non-PAM = nt dopamine & not PAM & non-VNC; serotonergic / octopaminergic = nt serotonin /
octopamine & non-VNC -- NT-defined classes, circular for a transmitter cross-check, usable for receptor profiles;
clock = the 10 named clock types s-LNv, l-LNv, 5thsLNv_LNd6, LNd_b, LNd_c, LPN_a, LPN_b, DN1a, DN1pA, DN1pB;
Poxn = ER1-ER4). `unmatched` rows keep the reason in `evidence`. Marker-gene values quoted in `evidence` are Davie
`mean_log1p_cp10k`. `evidence` carries `CAUTION` where an exact-name tier is suspect (FCA Tm29) and `SUBSET` where an
alias covers only part of the type (FCA Tm3a -> Tm3).

Mapping decisions that need a reader's attention:

* **Ring neurons**: the only route is the `Poxn` cluster (Davie 99 cells, FCA 102 nuclei). Poxn+ protocerebral
  dorsal-cluster neurons are the EB ring neurons R1-R4 (Omoto et al. 2019 Biol Open, ppd5/DALv2 lineage; Minocha et
  al. 2017 PLoS One), and the cluster is Gad1-high (Davie 3.20, FCA 2.12; ring neurons are GABAergic, MaleCNS ER1-4
  GABA 100 %), but Poxn is also expressed in ~100 deutocerebral (antennal-lobe-associated) neurons that are not ring
  neurons, so the cluster is mixed: tier `class`, ER1-ER4 (24 types, 257 cells), ER5 / ER6 excluded.
* **PNs**: Davie's adPN / lPN sub-clusters (C15+, kn+, unpg+, CG31676+; Li et al. 2017) are mapped to the whole
  `_adPN$` / `_lPN$` lineage; the glomerulus-level identity of each marker subset is not resolved here (Li 2017's
  glomerular assignments were not acquired).
* **DANs**: PAM -> the 15 PAM types (316 cells; raw output 222,136 node-restricted / 271,752 unrestricted, |W| 0 because
  dopamine is sign 0); non-PAM dopaminergic (Davie `Dopaminergic` = Fer2(-) cells, FCA `dopaminergic neuron`) -> nt-defined
  class (30 types, 78 cells: PPL1/PPL2/PPM/PAL/FB/ExR2 dopaminergic types).
* **NT-defined class rows reach DN / CX / gustatory types** (the non-VNC filter includes `descending_neuron`): the
  Serotonergic and Octopaminergic selections pull in 13 DN types (23 cells: DNg26, DNg30, DNp29, DNp32, DNpe048 via
  serotonin; DNg34, DNg66, DNg104, DNge138, DNge149-152 via octopamine), 12 non-ring CX types (94 cells: ExR2, FB1C,
  FB1H, FB2A, FB4L, FB4M, FB5H via dopamine; ExR3, FB4Y, PFGs, PFR_a via serotonin; EL via octopamine) and the gustatory
  LB2b (3 cells, octopamine). Table 3e lists them per row. These rows are circular for a transmitter cross-check and
  usable only as class-prior receptor profiles.
* **Davie `Pm3`** was an exact name hit but the cluster is glutamatergic (VGlut 3.97, Gad1 0.20) while MaleCNS Pm3 is
  GABA 100 % -> **unmatched** (the Pm numbering changed in the Nern 2025 nomenclature). `Tm1/TmY8` and FCA `TmY8`:
  MaleCNS v1.0 has no `TmY8` (-> Tm1 only / unmatched).
* **FCA `Tm29`** (115 nuclei) keeps tier `exact` but is flagged `CAUTION` (suspect): its profile is cholinergic (VAChT 1.24,
  56 % of nuclei; ChAT 0.47; VGlut 0.18, 9 %) whereas MaleCNS Tm29 (= FlyWire Tm5d, 544 cells) is glutamate 100 %;
  `flyverse/data/type_aliases.csv` already records "NT pred OL=glutamate FW=acetylcholine (MISMATCH)" for this type. Either
  the Oezel/FCA `Tm29` is not the Nern 2025 `Tm29` (name collision) or it is a fourth disagreement; it is listed as a
  disagreement in section 4 and should not be used as a Tm29 receptor profile until the correspondence is resolved.
* **FCA `Tm3a` -> Tm3** is tier `alias` but is a **subset**, not a synonym: Oezel 2021's Tm3a is about half of Tm3 (the
  Tm3a/Tm3b split), so the profile describes part of the 2,054-cell MaleCNS type.
* **FCA `Johnston organ neuron` / `auditory sensory neuron`** have inverted specificity: the narrower FBbt term
  "Johnston organ neuron" (541 nuclei) is mapped to all 34 JO-* types (672 cells) while the broader "auditory sensory
  neuron" (706 nuclei) is mapped to the MaleCNS subclass `auditory` only (10 JO-A/JO-B/JO-CA types, 111 cells). Harmless for
  coverage (the 10 types are inside the 34), but the two profiles should be read as one JO population.
* **FCA `Mi15`** (282 nuclei): DA-leaning (DAT 1.97, ple 0.75 pooled / 0.63 male, 114 male nuclei) with VAChT 1.25
  against MaleCNS ACh 100 % (1,151 cells) -- a disagreement (section 4), consistent with Davis 2020 calling Mi15 dopaminergic.
* **Clock superclasses**: of the 10 named clock types, s-LNv (8 cells) and 5thsLNv_LNd6 (4) are superclass
  `visual_projection` and l-LNv (8) is `ol_intrinsic`, so only 7 types / 32 cells count toward the central-brain coverage
  tables (which are computed by superclass and were already so); the Davie `LNv` row (s-LNv + l-LNv) contributes nothing
  to the central-brain rows.
* **Davie `ITP`** (25 cells) expresses ITP in only 24 % of its cells (mean log 0.41) -> not mapped to the MaleCNS `ITP`
  type; `dorsal_Fan-shaped_Body` (R23E10-mapped, VGlut 4.48) -> unmatched, candidate FB6-FB8 tangential types but the
  R23E10 -> EM-type correspondence was not established; `Tyraminergic` -> unmatched (no tyramine label in MaleCNS);
  the neuropeptide clusters (Capa, CCAP, FMRFa, AstA, CCHa1, Proc, Mip, Gr43a, Hsp) -> unmatched (see the table at
  the end).

## 3. Coverage of MaleCNS v1.0

Base: 164,501 typed cells (`type` non-empty) of the 167,106 model nodes; output synapses counted two ways: `|W|` =
signed synapses in `c.W` (sign-0 presynaptic cells -- DA / OA / 5-HT / unknown -- contribute 0, so a DAN type has 0
here), `raw` = synapse counts from the MaleCNS weights table. **Every `raw` number in this document is on the
node-restricted base** (edges with both endpoints in the node set; typed total 123,200,528 = the NT audit's 124,161,873
minus the 2,605 untyped cells' 961,345) unless labelled "unrestricted" (= the same presynaptic bodies' weight sum over all
postsynaptic bodies of the feather; the two bases are shown side by side in table 3d, e.g. PAM 222,136 vs 271,752).
"best tier per type" = the highest tier any source label reaches for that type (exact > alias > fuzzy > class), computed
by sorting on tier order before de-duplicating types. Optic lobe = ol_intrinsic + visual_projection + ol_sensory; central
= cb_intrinsic + cb_sensory + cb_endocrine + cb_motor + cb_efferent + visual_centrifugal.

Map totals: 1,150 rows = 1,063 matched + 87 unmatched; 576 distinct MaleCNS types matched, best tier **exact 36 / alias 3 /
fuzzy 128 / class 409**.

Headline: **optic lobe 79 / 628 types, 70,557 / 104,654 cells (67.4 %), 55.6 % of its output synapses** (exact 35 types
39.9 % of cells / 37.9 % of synapses; alias 2, 5.2 / 3.5 %; fuzzy 39, 22.3 / 14.2 %; class 3, 0.1 / 0.0 %).
**Central brain 484 / 6,925 types, 9,173 / 36,773 cells (24.9 %), 15.6 % (|W|) / 18.5 % (raw) of its output synapses**
(exact 1 type = IPC, 16 cells; alias 1 = ORN_V, 55 cells; fuzzy 89 types, 4,986 cells 13.6 % / 8.6 % |W|; class 393
types, 4,116 cells 11.2 % / 6.9 % |W| / 9.7 % raw). Central coverage is carried by the KCs (4,064 cells, 2.25 M |W|
output), the uniglomerular PNs (219 cells, 1.15 M), ORNs (2,635 cells, 1.35 M), MBONs (97, 0.26 M), ring neurons
ER1-4 (257, 0.43 M), JO neurons (672, 0.26 M), DANs / OA / 5-HT (sign-0 in |W|; 364,450 + 606,809 + 524,146 = 1.50 M raw
node-restricted for the three class rows, 222,136 for PAM). At class tier the NT-defined Serotonergic / Octopaminergic /
Dopaminergic selections **do** reach 13 DN types (23 cells), the FB tangential / ExR / PFGs / PFR_a / EL CX types (12 types,
94 non-ring cells; plus ER1-4 via Poxn) and the gustatory LB2b (3 cells) -- see table 3e -- but those rows are circular for
a transmitter cross-check and give only class-prior receptor profiles. Nothing in the LH, the CX columnar system
(EPG / PEN / PFN / hDelta), the non-aminergic DNs, the gustatory second-order cells or the GNG is covered by either
source: 75 % of central cells and 84 % of central |W| output synapses stay under the present `NT_SIGN` rule.

### 3a. Both sources (best tier per type)

#### Optic lobe (ol_intrinsic + visual_projection + ol_sensory): 628 types, 104,654 cells, 51,922,856 signed output synapses (|W|), 51,989,581 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 35 | 41,720 | 39.9% | 19,670,246 | 37.9% | 19,696,452 | 37.9% |
| alias | 2 | 5,431 | 5.2% | 1,819,373 | 3.5% | 1,819,373 | 3.5% |
| fuzzy | 39 | 23,309 | 22.3% | 7,370,295 | 14.2% | 7,377,855 | 14.2% |
| class | 3 | 97 | 0.1% | 6,138 | 0.0% | 6,737 | 0.0% |
| any tier | 79 | 70,557 | 67.4% | 28,866,052 | 55.6% | 28,900,417 | 55.6% |
| unmatched | 549 | 34,097 | 32.6% | 23,056,804 | 44.4% | 23,089,164 | 44.4% |

#### Central brain (cb_intrinsic + cb_sensory + cb_endocrine + cb_motor + cb_efferent + visual_centrifugal): 6925 types, 36,773 cells, 40,871,360 signed output synapses (|W|), 42,943,403 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 1 | 16 | 0.0% | 0 | 0.0% | 218 | 0.0% |
| alias | 1 | 55 | 0.1% | 23,866 | 0.1% | 23,866 | 0.1% |
| fuzzy | 89 | 4,986 | 13.6% | 3,520,654 | 8.6% | 3,748,142 | 8.7% |
| class | 393 | 4,116 | 11.2% | 2,830,171 | 6.9% | 4,174,512 | 9.7% |
| any tier | 484 | 9,173 | 24.9% | 6,374,691 | 15.6% | 7,946,738 | 18.5% |
| unmatched | 6441 | 27,600 | 75.1% | 34,496,669 | 84.4% | 34,996,665 | 81.5% |

#### All brain (optic + central + descending_neuron): 8033 types, 142,737 cells, 96,687,728 signed output synapses (|W|), 99,012,817 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 36 | 41,736 | 29.2% | 19,670,246 | 20.3% | 19,696,670 | 19.9% |
| alias | 3 | 5,486 | 3.8% | 1,843,239 | 1.9% | 1,843,239 | 1.9% |
| fuzzy | 128 | 28,295 | 19.8% | 10,890,949 | 11.3% | 11,125,997 | 11.2% |
| class | 409 | 4,236 | 3.0% | 2,836,309 | 2.9% | 4,309,735 | 4.4% |
| any tier | 576 | 79,753 | 55.9% | 35,240,743 | 36.4% | 36,975,641 | 37.3% |
| unmatched | 7457 | 62,984 | 44.1% | 61,446,985 | 63.6% | 62,037,176 | 62.7% |

#### All MaleCNS typed cells: 11751 types, 164,501 cells, 120,527,064 signed output synapses (|W|), 123,200,528 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 36 | 41,736 | 25.4% | 19,670,246 | 16.3% | 19,696,670 | 16.0% |
| alias | 3 | 5,486 | 3.3% | 1,843,239 | 1.5% | 1,843,239 | 1.5% |
| fuzzy | 128 | 28,295 | 17.2% | 10,890,949 | 9.0% | 11,125,997 | 9.0% |
| class | 409 | 4,236 | 2.6% | 2,836,309 | 2.4% | 4,309,735 | 3.5% |
| any tier | 576 | 79,753 | 48.5% | 35,240,743 | 29.2% | 36,975,641 | 30.0% |
| unmatched | 11175 | 84,748 | 51.5% | 85,286,321 | 70.8% | 86,224,887 | 70.0% |

### 3b. davie2018 only

#### Optic lobe (ol_intrinsic + visual_projection + ol_sensory): 628 types, 104,654 cells, 51,922,856 signed output synapses (|W|), 51,989,581 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 16 | 20,154 | 19.3% | 8,397,904 | 16.2% | 8,424,112 | 16.2% |
| alias | 0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| fuzzy | 25 | 22,083 | 21.1% | 9,033,253 | 17.4% | 9,040,813 | 17.4% |
| class | 12 | 6,103 | 5.8% | 652,997 | 1.3% | 653,596 | 1.3% |
| any tier | 53 | 48,340 | 46.2% | 18,084,154 | 34.8% | 18,118,521 | 34.9% |
| unmatched | 575 | 56,314 | 53.8% | 33,838,702 | 65.2% | 33,871,060 | 65.1% |

#### Central brain (cb_intrinsic + cb_sensory + cb_endocrine + cb_motor + cb_efferent + visual_centrifugal): 6925 types, 36,773 cells, 40,871,360 signed output synapses (|W|), 42,943,403 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 1 | 16 | 0.0% | 0 | 0.0% | 218 | 0.0% |
| alias | 0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| fuzzy | 86 | 4,621 | 12.6% | 3,413,819 | 8.4% | 3,641,307 | 8.5% |
| class | 180 | 764 | 2.1% | 724,530 | 1.8% | 2,068,850 | 4.8% |
| any tier | 267 | 5,401 | 14.7% | 4,138,349 | 10.1% | 5,710,375 | 13.3% |
| unmatched | 6658 | 31,372 | 85.3% | 36,733,011 | 89.9% | 37,233,028 | 86.7% |

#### All brain (optic + central + descending_neuron): 8033 types, 142,737 cells, 96,687,728 signed output synapses (|W|), 99,012,817 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 17 | 20,170 | 14.1% | 8,397,904 | 8.7% | 8,424,330 | 8.5% |
| alias | 0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| fuzzy | 111 | 26,704 | 18.7% | 12,447,072 | 12.9% | 12,682,120 | 12.8% |
| class | 205 | 6,890 | 4.8% | 1,377,527 | 1.4% | 2,850,932 | 2.9% |
| any tier | 333 | 53,764 | 37.7% | 22,222,503 | 23.0% | 23,957,382 | 24.2% |
| unmatched | 7700 | 88,973 | 62.3% | 74,465,225 | 77.0% | 75,055,435 | 75.8% |

#### All MaleCNS typed cells: 11751 types, 164,501 cells, 120,527,064 signed output synapses (|W|), 123,200,528 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 17 | 20,170 | 12.3% | 8,397,904 | 7.0% | 8,424,330 | 6.8% |
| alias | 0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| fuzzy | 111 | 26,704 | 16.2% | 12,447,072 | 10.3% | 12,682,120 | 10.3% |
| class | 205 | 6,890 | 4.2% | 1,377,527 | 1.1% | 2,850,932 | 2.3% |
| any tier | 333 | 53,764 | 32.7% | 22,222,503 | 18.4% | 23,957,382 | 19.4% |
| unmatched | 11418 | 110,737 | 67.3% | 98,304,561 | 81.6% | 99,243,146 | 80.6% |

### 3c. fca2022 only

#### Optic lobe (ol_intrinsic + visual_projection + ol_sensory): 628 types, 104,654 cells, 51,922,856 signed output synapses (|W|), 51,989,581 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 33 | 40,976 | 39.2% | 19,619,970 | 37.8% | 19,646,179 | 37.8% |
| alias | 4 | 6,175 | 5.9% | 1,869,646 | 3.6% | 1,869,646 | 3.6% |
| fuzzy | 30 | 21,744 | 20.8% | 6,399,641 | 12.3% | 6,399,641 | 12.3% |
| class | 1 | 8 | 0.0% | 0 | 0.0% | 599 | 0.0% |
| any tier | 68 | 68,903 | 65.8% | 27,889,257 | 53.7% | 27,916,065 | 53.7% |
| unmatched | 560 | 35,751 | 34.2% | 24,033,599 | 46.3% | 24,073,516 | 46.3% |

#### Central brain (cb_intrinsic + cb_sensory + cb_endocrine + cb_motor + cb_efferent + visual_centrifugal): 6925 types, 36,773 cells, 40,871,360 signed output synapses (|W|), 42,943,403 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| alias | 1 | 55 | 0.1% | 23,866 | 0.1% | 23,866 | 0.1% |
| fuzzy | 32 | 4,743 | 12.9% | 2,353,457 | 5.8% | 2,575,593 | 6.0% |
| class | 336 | 3,964 | 10.8% | 3,692,749 | 9.0% | 4,574,890 | 10.7% |
| any tier | 369 | 8,762 | 23.8% | 6,070,072 | 14.9% | 7,174,349 | 16.7% |
| unmatched | 6556 | 28,011 | 76.2% | 34,801,288 | 85.1% | 35,769,054 | 83.3% |

#### All brain (optic + central + descending_neuron): 8033 types, 142,737 cells, 96,687,728 signed output synapses (|W|), 99,012,817 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 33 | 40,976 | 28.7% | 19,619,970 | 20.3% | 19,646,179 | 19.8% |
| alias | 5 | 6,230 | 4.4% | 1,893,512 | 2.0% | 1,893,512 | 1.9% |
| fuzzy | 62 | 26,487 | 18.6% | 8,753,098 | 9.1% | 8,975,234 | 9.1% |
| class | 345 | 3,983 | 2.8% | 3,692,749 | 3.8% | 4,663,237 | 4.7% |
| any tier | 445 | 77,676 | 54.4% | 33,959,329 | 35.1% | 35,178,162 | 35.5% |
| unmatched | 7588 | 65,061 | 45.6% | 62,728,399 | 64.9% | 63,834,655 | 64.5% |

#### All MaleCNS typed cells: 11751 types, 164,501 cells, 120,527,064 signed output synapses (|W|), 123,200,528 raw output synapses (node-restricted)

| tier | types | cells | cells % | out syn (|W|) | % | out syn (raw) | % |
|---|---|---|---|---|---|---|---|
| exact | 33 | 40,976 | 24.9% | 19,619,970 | 16.3% | 19,646,179 | 15.9% |
| alias | 5 | 6,230 | 3.8% | 1,893,512 | 1.6% | 1,893,512 | 1.5% |
| fuzzy | 62 | 26,487 | 16.1% | 8,753,098 | 7.3% | 8,975,234 | 7.3% |
| class | 345 | 3,983 | 2.4% | 3,692,749 | 3.1% | 4,663,237 | 3.8% |
| any tier | 445 | 77,676 | 47.2% | 33,959,329 | 28.2% | 35,178,162 | 28.6% |
| unmatched | 11306 | 86,825 | 52.8% | 86,567,735 | 71.8% | 88,022,366 | 71.4% |

### 3d. Central-brain classes matched (both sources; MaleCNS cells / signed out / raw out on both bases)

| source | source_name | tier | n MaleCNS types | MaleCNS cells | out syn (|W|) | out syn (raw, node-restricted) | out syn (raw, unrestricted) | source cells |
|---|---|---|---|---|---|---|---|---|
| davie2018 | G-KC | fuzzy | 7 | 1,557 | 1,010,440 | 1,010,440 | 1,335,123 | 1,642 |
| davie2018 | A/B-KC | fuzzy | 4 | 1,810 | 776,480 | 776,480 | 1,020,388 | 688 |
| davie2018 | A/B*-KC | fuzzy | 3 | 695 | 459,702 | 459,702 | 648,794 | 518 |
| davie2018 | PAM | fuzzy | 15 | 316 | 0 | 222,136 | 271,752 | 222 |
| davie2018 | Dopaminergic | class | 30 | 78 | 0 | 364,450 | 596,183 | 125 |
| davie2018 | Serotonergic | class | 75 | 282 | 8,354 | 524,146 | 1,192,524 | 391 |
| davie2018 | Octopaminergic | class | 27 | 77 | 792 | 606,809 | 1,105,504 | 76 |
| davie2018 | Clock | class | 10 | 52 | 39,490 | 47,050 | 145,338 | 389 |
| davie2018 | DN1 | fuzzy | 3 | 16 | 13,426 | 13,426 | 46,989 | 29 |
| davie2018 | MBON | class | 37 | 97 | 263,422 | 263,422 | 870,384 | 257 |
| davie2018 | Olfactory_projection_neurons | fuzzy | 51 | 219 | 1,153,771 | 1,153,771 | 2,858,462 | 71 |
| davie2018 | adPN | fuzzy | 39 | 152 | 854,785 | 854,785 | 2,125,579 | 107 |
| davie2018 | adPN/C15 | fuzzy | 39 | 152 | 854,785 | 854,785 | 2,125,579 | 46 |
| davie2018 | adPN/kn | fuzzy | 39 | 152 | 854,785 | 854,785 | 2,125,579 | 70 |
| davie2018 | adPN/C15&kn | fuzzy | 39 | 152 | 854,785 | 854,785 | 2,125,579 | 131 |
| davie2018 | adPN/kn&CG31676 | fuzzy | 39 | 152 | 854,785 | 854,785 | 2,125,579 | 11 |
| davie2018 | lPN | fuzzy | 12 | 67 | 298,986 | 298,986 | 732,883 | 14 |
| davie2018 | lPN/unpg | fuzzy | 12 | 67 | 298,986 | 298,986 | 732,883 | 41 |
| davie2018 | lPN/CG31676 | fuzzy | 12 | 67 | 298,986 | 298,986 | 732,883 | 52 |
| davie2018 | Poxn | class | 24 | 257 | 432,545 | 432,545 | 508,067 | 99 |
| davie2018 | IPC | exact | 1 | 16 | 0 | 218 | 542 | 17 |
| davie2018 | Crz | fuzzy | 2 | 4 | 0 | 5,294 | 12,460 | 9 |
| davie2018 | Hug | fuzzy | 1 | 4 | 0 | 58 | 155 | 42 |
| fca2022 | gamma Kenyon cell | fuzzy | 7 | 1,557 | 1,010,440 | 1,010,440 | 1,335,123 | 1,139 |
| fca2022 | alpha/beta Kenyon cell | fuzzy | 4 | 1,810 | 776,480 | 776,480 | 1,020,388 | 525 |
| fca2022 | alpha'/beta' Kenyon cell | fuzzy | 3 | 695 | 459,702 | 459,702 | 648,794 | 431 |
| fca2022 | Kenyon cell | class | 15 | 4,064 | 2,246,664 | 2,246,664 | 3,004,381 | 116 |
| fca2022 | dopaminergic PAM neuron | fuzzy | 15 | 316 | 0 | 222,136 | 271,752 | 288 |
| fca2022 | dopaminergic neuron | class | 30 | 78 | 0 | 364,450 | 596,183 | 25 |
| fca2022 | octopaminergic/tyraminergic neuron | class | 27 | 77 | 792 | 606,809 | 1,105,504 | 120 |
| fca2022 | antennal lobe projection neuron | class | 180 | 682 | 1,775,824 | 1,775,824 | 4,912,176 | 23 |
| fca2022 | Poxn neuron | class | 24 | 257 | 432,545 | 432,545 | 508,067 | 102 |
| fca2022 | olfactory receptor neuron | class | 53 | 2,635 | 1,349,270 | 1,349,270 | 2,192,327 | 1,364 |
| fca2022 | adult olfactory receptor neuron Gr21a/63a | alias | 1 | 55 | 23,866 | 23,866 | 43,285 | 6 |
| fca2022 | antennal trichoid sensillum at4 | fuzzy | 3 | 365 | 106,835 | 106,835 | 162,360 | 7 |
| fca2022 | Johnston organ neuron | class | 34 | 672 | 264,977 | 264,998 | 644,418 | 541 |
| fca2022 | auditory sensory neuron | class | 10 | 111 | 30,635 | 30,656 | 68,338 | 706 |

### 3e. NT-defined class rows: DN, CX and gustatory types they reach

These rows select the MaleCNS side by consensus `nt`, so they cannot cross-check a transmitter; they are usable only as class-prior receptor profiles.

| source | source_name | MaleCNS types (cells) | of which DN types (cells) | of which CX types (cells) | of which gustatory (cells) |
|---|---|---|---|---|---|
| davie2018 | Dopaminergic | 30 (78) | 0 (0) | 7 (24): ExR2 4, FB1C 4, FB1H 2, FB2A 4, FB4L 4, FB4M 4, FB5H 2 | 0 (0) |
| davie2018 | Serotonergic | 75 (282) | 5 (12): DNg26 4, DNg30 2, DNp29 2, DNp32 2, DNpe048 2 | 4 (52): ExR3 2, FB4Y 4, PFGs 18, PFR_a 28 | 0 (0) |
| davie2018 | Octopaminergic | 27 (77) | 8 (11): DNg104 2, DNg34 2, DNg66 1, DNge138 2, DNge149 1, DNge150 1, DNge151 1, DNge152 1 | 1 (18): EL 18 | 1 (3): LB2b 3 |
| fca2022 | dopaminergic neuron | 30 (78) | 0 (0) | 7 (24): ExR2 4, FB1C 4, FB1H 2, FB2A 4, FB4L 4, FB4M 4, FB5H 2 | 0 (0) |
| fca2022 | octopaminergic/tyraminergic neuron | 27 (77) | 8 (11): DNg104 2, DNg34 2, DNg66 1, DNge138 2, DNge149 1, DNge150 1, DNge151 1, DNge152 1 | 1 (18): EL 18 | 1 (3): LB2b 3 |

Clock list: 10 types / 52 cells; superclass per type: s-LNv visual_projection (8), l-LNv ol_intrinsic (8), 5thsLNv_LNd6 visual_projection (4), LNd_b cb_intrinsic (4), LNd_c cb_intrinsic (6), LPN_a cb_intrinsic (4), LPN_b cb_intrinsic (2), DN1a cb_intrinsic (4), DN1pA cb_intrinsic (8), DN1pB cb_intrinsic (4). Only 7 types / 32 cells are in the central-brain superclasses (s-LNv and 5thsLNv_LNd6 are visual_projection, l-LNv is ol_intrinsic).

## 4. Transmitter cross-check (marker genes vs MaleCNS consensus)

Per matched label: marker score per transmitter = mean of its synthesis/transport genes (`mean_log1p_cp10k`; ACh = VAChT,
ChAT; GABA = Gad1, VGAT; Glu = VGlut; DA = ple, DAT; OA = Tdc2, Tbh; 5-HT = SerT, Trh; His = Hdc). **Marker-call rule
(round 2, per source):** call = top transmitter if its score >= **1.0 (davie2018) / 0.5 (fca2022)** and the `frac_expr`
of its best marker gene >= 0.3; `X?` = above threshold but < 2x the runner-up; `none` = marker-silent. The FCA threshold
is lower because FCA is nuclear RNA with cp10k over 13,056 retained genes: the FCA labels of confirmed cholinergic types
(L2-L5, Tm4, Tm20, TmY4, LC10/12/17, T2-T5, the three KC classes, ORNs, JO; MaleCNS ACh 100 % or majority) score
0.62-0.98 on mean(VAChT, ChAT), while the marker-silent FCA labels (T1, at4, Gr21a/63a, R7 / R8 / outer photoreceptors)
score <= 0.39; Davie whole-cell clusters of confirmed cholinergic types score 1.28-2.5 and its marker-silent clusters
(T1 0.33, IPC 0.35, Hug 0.22, Crz 0.54, LNv 0.57, Photoreceptors 0.98) stay below 1.0, so Davie keeps 1.0 (a uniform 0.5
would call Davie Photoreceptors "Glu?" and LNv "OA?" spuriously). Round 1 used 1.0 for both sources and filed 40 labels as
marker-silent, 28 of them FCA labels whose best marker sits at 0.62-0.98 (cholinergic: the three KC classes, Kenyon cell,
ORN, JO, auditory, T2, T3, T4/T5 x3, L2-L5, Lawf1, Tm3a, Tm4, Tm20, Tm29, TmY4, LC10/12/17; GABAergic: C2, Dm10;
octopaminergic: OA/TA) -- 27 of them now agree and Tm29 now disagrees. MaleCNS consensus = cell-weighted
`nt` of the mapped types. Result over the 111 matched labels: **agree 95, marker-silent 12, disagree 4** (round 1:
68 / 40 / 3). The five NT-defined class rows (Davie Dopaminergic / Serotonergic / Octopaminergic, FCA dopaminergic neuron /
octopaminergic-tyraminergic neuron) are marked `(circular)`: they select the MaleCNS side by consensus `nt`, so their
agreement is not evidence.

Findings: (i) every KC, PN, ORN, JO, PAM, MBON, ring-neuron (Poxn), lamina, T2/T3/T4/T5, Mi1/Mi4/Mi9, Tm1/2/4/9/20/5c,
TmY4/5a/14, C2/C3, Dm3/8/9/10/11/12, Pm2/Pm4, LC10/12/17, LC14, Lai and Lawf label agrees with the MaleCNS consensus (the
DA / OA / 5-HT class rows agree circularly); (ii) **TmY14** is VGlut 4.04 (Davie, 1,399 cells) / 1.90 (FCA, 153 nuclei)
with no ACh markers -> glutamate, which settles the 91 all-`unclear` TmY14 cells of the NT audit (table 1d: T-bars
glutamate 49 % / ACh 48 %) as glutamate; (iii) **T1** expresses no transmitter marker at all (Hdc 0.00 in both sources,
VAChT 0.34 / 0.25, VGlut 0.33 / 0.10; Davie ort 0.97 mean log1p cp10k, 55 % of cells, mean cp10k 3.04), so the MaleCNS
`histamine` consensus for T1 (1,777 cells, sign -1 in the model) has no transcriptomic support -- T1 is known as
marker-negative (Davis 2020); (iv) **Mi15** (FCA, 282 nuclei -> 1,151 cells) is DAT 1.97 / ple 0.75 pooled (0.63 male)
with VAChT 1.25 -> `DA?`, a dopaminergic-leaning profile against the MaleCNS ACh 100 % call (Mi15 was reported dopaminergic
by Davis 2020); (v) **Tm29** (FCA, 115 nuclei -> 544 cells): VAChT 1.24 (56 %) / ChAT 0.47 / VGlut 0.18 (9 %) -> `ACh`
against MaleCNS glutamate 100 % -- either a name collision (Oezel/FCA Tm29 != Nern 2025 Tm29 = FlyWire Tm5d) or a real
disagreement; the map row carries `CAUTION`; (vi) the Davie `LNv` cluster (Pdf 6.29) has SerT 0.00 / Trh 0.00, so the
MaleCNS `serotonin` consensus for l-LNv (8 cells) is unsupported (the 8 s-LNv are ACh); (vii) `Clock` disagrees only
because the 389-cell cluster is far broader than the 10 named clock types; `dorsal rim area` is 2 nuclei. The 12
marker-silent labels are the peptidergic / endocrine clusters (IPC, Crz, Hug), LNv, T1 (both sources) and the
photoreceptor labels (Davie Photoreceptors; FCA outer / R7 / R8) plus two tiny FCA ORN labels (Gr21a/63a 6 nuclei, at4 7).

| source | source_name | tier | types | cells | marker call | score | frac | VAChT | Gad1 | VGlut | DAT | Tbh | SerT | Hdc | MaleCNS consensus (cell-weighted) | agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| davie2018 | G-KC | fuzzy | 7 | 1,557 | ACh | 1.48 | 0.9 | 2.31 | 0.08 | 0.23 | 0.02 | 0.01 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | A/B-KC | fuzzy | 4 | 1,810 | ACh | 1.28 | 0.84 | 2.09 | 0.12 | 0.24 | 0.03 | 0.0 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | A/B*-KC | fuzzy | 3 | 695 | ACh | 1.55 | 0.89 | 2.26 | 0.09 | 0.22 | 1.52 | 0.01 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | PAM | fuzzy | 15 | 316 | DA | 3.64 | 0.97 | 0.23 | 0.07 | 0.34 | 4.12 | 0.14 | 0.0 | 0.0 | DA 100% | yes |
| davie2018 | Dopaminergic | class | 30 | 78 | DA | 3.34 | 0.93 | 0.71 | 0.15 | 0.87 | 3.33 | 0.11 | 0.04 | 0.0 | DA 99%, unk 1% | yes (circular) |
| davie2018 | Serotonergic | class | 75 | 282 | 5-HT? | 1.53 | 0.64 | 0.9 | 0.35 | 1.36 | 0.02 | 0.13 | 1.26 | 0.0 | 5-HT 96%, ACh 2%, unk 2% | yes (circular) |
| davie2018 | Octopaminergic | class | 27 | 77 | OA | 2.5 | 0.91 | 0.47 | 0.07 | 0.71 | 0.06 | 2.56 | 0.02 | 0.0 | OA 99%, Glu 1% | yes (circular) |
| davie2018 | Clock | class | 10 | 52 | Glu | 2.03 | 0.63 | 1.1 | 0.11 | 2.03 | 0.05 | 0.35 | 0.0 | 0.0 | ACh 54%, Glu 31%, 5-HT 15% | NO |
| davie2018 | LNv | fuzzy | 2 | 16 | none | 0.57 | 0.88 | 0.44 | 0.05 | 0.21 | 0.13 | 1.07 | 0.0 | 0.0 | 5-HT 50%, ACh 50% | - |
| davie2018 | DN1 | fuzzy | 3 | 16 | Glu | 5.01 | 1.0 | 0.04 | 0.0 | 5.01 | 0.0 | 1.43 | 0.0 | 0.0 | Glu 100% | yes |
| davie2018 | MBON | class | 37 | 97 | ACh? | 1.13 | 0.55 | 1.41 | 0.35 | 1.05 | 0.02 | 0.08 | 0.01 | 0.0 | ACh 52%, Glu 27%, GABA 22% | yes |
| davie2018 | Olfactory_projection_neurons | fuzzy | 51 | 219 | ACh | 1.91 | 0.93 | 2.22 | 0.05 | 0.38 | 0.04 | 0.08 | 0.02 | 0.0 | ACh 100% | yes |
| davie2018 | adPN | fuzzy | 39 | 152 | ACh | 1.98 | 0.94 | 2.36 | 0.1 | 0.13 | 0.0 | 0.16 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | adPN/C15 | fuzzy | 39 | 152 | ACh | 2.18 | 0.96 | 2.59 | 0.11 | 0.23 | 0.03 | 0.11 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | adPN/kn | fuzzy | 39 | 152 | ACh | 2.29 | 0.91 | 2.66 | 0.06 | 0.13 | 0.03 | 0.28 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | adPN/C15&kn | fuzzy | 39 | 152 | ACh | 2.2 | 0.98 | 2.41 | 0.05 | 0.16 | 0.01 | 0.08 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | adPN/kn&CG31676 | fuzzy | 39 | 152 | ACh | 2.06 | 1.0 | 2.89 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | lPN | fuzzy | 12 | 67 | ACh | 2.49 | 1.0 | 2.73 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | lPN/unpg | fuzzy | 12 | 67 | ACh | 2.04 | 0.93 | 2.33 | 0.09 | 0.19 | 0.0 | 0.23 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | lPN/CG31676 | fuzzy | 12 | 67 | ACh | 1.94 | 0.9 | 2.31 | 0.05 | 0.07 | 0.0 | 0.15 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | Poxn | class | 24 | 257 | GABA | 2.09 | 0.99 | 0.27 | 3.2 | 0.24 | 0.01 | 1.09 | 0.0 | 0.0 | GABA 100% | yes |
| davie2018 | IPC | exact | 1 | 16 | none | 0.35 | 0.18 | 0.3 | 0.27 | 0.35 | 0.0 | 0.07 | 0.02 | 0.0 | unk 100% | - |
| davie2018 | Crz | fuzzy | 2 | 4 | none | 0.54 | 0.22 | 0.55 | 0.5 | 0.54 | 0.0 | 0.15 | 0.0 | 0.0 | 5-HT 75%, unk 25% | - |
| davie2018 | Hug | fuzzy | 1 | 4 | none | 0.22 | 0.19 | 0.12 | 0.07 | 0.22 | 0.0 | 0.13 | 0.0 | 0.0 | unk 100% | - |
| davie2018 | DCN | fuzzy | 3 | 88 | ACh | 1.88 | 0.96 | 2.26 | 0.11 | 0.39 | 0.05 | 0.1 | 0.01 | 0.0 | ACh 100% | yes |
| davie2018 | TmY14 | exact | 1 | 477 | Glu | 4.04 | 0.98 | 0.23 | 0.14 | 4.04 | 0.03 | 0.06 | 0.0 | 0.0 | Glu 44%, ACh 37%, unk 19% | yes |
| davie2018 | Mi1 | exact | 1 | 1,773 | ACh | 2.5 | 0.96 | 2.91 | 0.1 | 0.21 | 0.0 | 0.01 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | T1 | exact | 1 | 1,777 | none | 0.33 | 0.15 | 0.34 | 0.15 | 0.33 | 0.01 | 0.03 | 0.0 | 0.0 | His 100% | - |
| davie2018 | Tm9 | exact | 1 | 1,771 | ACh | 2.18 | 0.85 | 2.41 | 0.11 | 0.48 | 0.01 | 0.03 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | Tm5c | exact | 1 | 750 | Glu | 3.91 | 1.0 | 0.11 | 0.08 | 3.91 | 0.0 | 0.07 | 0.0 | 0.0 | Glu 100% | yes |
| davie2018 | Tm5ab | fuzzy | 2 | 1,146 | ACh | 1.66 | 0.85 | 2.31 | 0.09 | 0.26 | 0.01 | 0.27 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | Dm8/Dm11 | fuzzy | 3 | 1,262 | Glu | 4.24 | 0.97 | 0.25 | 0.09 | 4.24 | 0.01 | 0.3 | 0.0 | 0.0 | Glu 100% | yes |
| davie2018 | Dm9 | exact | 1 | 273 | Glu | 5.19 | 1.0 | 0.14 | 0.11 | 5.19 | 0.02 | 0.0 | 0.01 | 0.01 | Glu 100% | yes |
| davie2018 | Pm1/Pm2 | fuzzy | 3 | 589 | GABA | 2.44 | 0.99 | 0.38 | 3.7 | 0.23 | 0.01 | 0.04 | 0.0 | 0.0 | GABA 100% | yes |
| davie2018 | Pm1/Pm2/Pm3 | fuzzy | 4 | 658 | GABA | 2.32 | 0.93 | 0.23 | 3.44 | 0.23 | 0.01 | 0.03 | 0.0 | 0.0 | GABA 100% | yes |
| davie2018 | Pm4 | exact | 1 | 168 | GABA | 2.53 | 1.0 | 0.24 | 3.87 | 0.32 | 0.01 | 0.24 | 0.0 | 0.01 | GABA 100% | yes |
| davie2018 | T2 | exact | 1 | 1,630 | ACh | 1.43 | 0.87 | 2.31 | 0.16 | 0.38 | 0.02 | 0.29 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | T3 | exact | 1 | 1,940 | ACh | 1.83 | 0.91 | 2.62 | 0.09 | 0.32 | 0.02 | 0.34 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | T4/T5 | fuzzy | 8 | 13,580 | ACh | 1.66 | 0.87 | 2.35 | 0.15 | 0.51 | 0.03 | 0.1 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | C2 | exact | 1 | 1,745 | GABA | 1.72 | 0.89 | 0.26 | 2.58 | 0.38 | 0.04 | 0.01 | 0.01 | 0.0 | GABA 100% | yes |
| davie2018 | C3 | exact | 1 | 1,779 | GABA | 2.16 | 0.93 | 0.21 | 3.39 | 0.34 | 0.03 | 0.02 | 0.0 | 0.01 | GABA 100% | yes |
| davie2018 | Tm1/TmY8 | fuzzy | 1 | 1,777 | ACh | 2.06 | 0.9 | 2.6 | 0.15 | 0.44 | 0.03 | 0.03 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | Lawf1 | exact | 1 | 361 | ACh | 1.68 | 0.84 | 1.98 | 0.06 | 0.25 | 0.01 | 0.0 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | Lawf2 | exact | 1 | 383 | ACh | 1.79 | 0.84 | 2.05 | 0.18 | 0.6 | 0.02 | 0.05 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | L1 | exact | 1 | 1,776 | Glu | 2.34 | 0.65 | 1.15 | 0.01 | 2.34 | 0.0 | 0.07 | 0.0 | 0.0 | Glu 100% | yes |
| davie2018 | L2 | exact | 1 | 1,779 | ACh | 2.42 | 0.95 | 3.17 | 0.04 | 0.22 | 0.02 | 0.0 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | L3 | exact | 1 | 1,772 | ACh | 1.64 | 0.81 | 2.05 | 0.08 | 0.37 | 0.02 | 0.0 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | L4/L5 | fuzzy | 2 | 3,556 | ACh | 2.48 | 0.93 | 2.86 | 0.07 | 0.4 | 0.01 | 0.0 | 0.0 | 0.0 | ACh 100% | yes |
| davie2018 | Lamina_monopolar | fuzzy | 5 | 8,883 | ACh | 2.16 | 0.93 | 2.72 | 0.09 | 0.3 | 0.01 | 0.02 | 0.0 | 0.0 | ACh 80%, Glu 20% | yes |
| davie2018 | Photoreceptors | class | 10 | 6,091 | none | 0.98 | 0.36 | 1.13 | 0.37 | 0.98 | 0.01 | 0.06 | 0.0 | 0.06 | His 100% | - |
| fca2022 | gamma Kenyon cell | fuzzy | 7 | 1,557 | ACh | 0.9 | 0.56 | 1.4 | 0.09 | 0.05 | 0.06 | 0.06 | 0.01 | 0.01 | ACh 100% | yes |
| fca2022 | alpha/beta Kenyon cell | fuzzy | 4 | 1,810 | ACh | 0.67 | 0.44 | 1.11 | 0.07 | 0.02 | 0.03 | 0.01 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | alpha'/beta' Kenyon cell | fuzzy | 3 | 695 | ACh | 0.83 | 0.47 | 1.24 | 0.09 | 0.06 | 0.68 | 0.08 | 0.01 | 0.01 | ACh 100% | yes |
| fca2022 | Kenyon cell | class | 15 | 4,064 | ACh? | 0.81 | 0.5 | 1.05 | 0.48 | 0.49 | 0.09 | 0.26 | 0.05 | 0.0 | ACh 100% | yes |
| fca2022 | dopaminergic PAM neuron | fuzzy | 15 | 316 | DA | 2.53 | 0.99 | 0.29 | 0.1 | 0.19 | 3.62 | 0.28 | 0.01 | 0.0 | DA 100% | yes |
| fca2022 | dopaminergic neuron | class | 30 | 78 | DA | 1.55 | 0.92 | 0.85 | 0.14 | 0.63 | 2.43 | 0.21 | 0.17 | 0.0 | DA 99%, unk 1% | yes (circular) |
| fca2022 | octopaminergic/tyraminergic neuron | class | 27 | 77 | OA | 0.97 | 0.38 | 0.45 | 0.13 | 0.18 | 0.55 | 0.92 | 0.35 | 0.02 | OA 99%, Glu 1% | yes (circular) |
| fca2022 | antennal lobe projection neuron | class | 180 | 682 | ACh | 1.26 | 0.74 | 1.67 | 0.08 | 0.0 | 0.0 | 0.45 | 0.0 | 0.0 | ACh 68%, GABA 31%, Glu 1% | yes |
| fca2022 | Poxn neuron | class | 24 | 257 | GABA? | 1.33 | 0.79 | 0.29 | 2.12 | 0.01 | 0.0 | 2.01 | 0.0 | 0.0 | GABA 100% | yes |
| fca2022 | olfactory receptor neuron | class | 53 | 2,635 | ACh | 0.65 | 0.36 | 0.89 | 0.05 | 0.03 | 0.02 | 0.02 | 0.01 | 0.01 | ACh 100% | yes |
| fca2022 | adult olfactory receptor neuron Gr21a/63a | alias | 1 | 55 | none | 0.28 | 0.17 | 0.29 | 0.37 | 0.0 | 0.0 | 0.36 | 0.0 | 0.0 | ACh 100% | - |
| fca2022 | antennal trichoid sensillum at4 | fuzzy | 3 | 365 | none | 0.25 | 0.14 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | ACh 100% | - |
| fca2022 | Johnston organ neuron | class | 34 | 672 | ACh | 0.79 | 0.43 | 1.12 | 0.06 | 0.04 | 0.0 | 0.04 | 0.01 | 0.01 | ACh 89%, unk 11% | yes |
| fca2022 | auditory sensory neuron | class | 10 | 111 | ACh | 0.89 | 0.51 | 1.21 | 0.05 | 0.07 | 0.02 | 0.06 | 0.01 | 0.01 | ACh 67%, unk 33% | yes |
| fca2022 | columnar neuron T1 | exact | 1 | 1,777 | none | 0.18 | 0.13 | 0.25 | 0.12 | 0.1 | 0.02 | 0.07 | 0.0 | 0.0 | His 100% | - |
| fca2022 | T neuron T4/T5a-b | fuzzy | 4 | 6,753 | ACh | 0.69 | 0.38 | 1.07 | 0.13 | 0.07 | 0.0 | 0.25 | 0.01 | 0.0 | ACh 100% | yes |
| fca2022 | T neuron T4/T5c-d | fuzzy | 4 | 6,827 | ACh | 0.68 | 0.34 | 1.01 | 0.07 | 0.05 | 0.01 | 0.33 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | T neuron T4/T5 | fuzzy | 8 | 13,580 | ACh? | 0.66 | 0.44 | 1.04 | 0.38 | 0.41 | 0.02 | 0.32 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | T neuron T2 | exact | 1 | 1,630 | ACh | 0.94 | 0.5 | 1.4 | 0.09 | 0.06 | 0.03 | 0.88 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | T neuron T2a | exact | 1 | 1,872 | ACh? | 1.02 | 0.55 | 1.62 | 0.09 | 0.05 | 0.0 | 1.17 | 0.01 | 0.01 | ACh 100%, unk 0% | yes |
| fca2022 | T neuron T3 | exact | 1 | 1,940 | ACh? | 0.92 | 0.48 | 1.35 | 0.1 | 0.08 | 0.02 | 1.27 | 0.02 | 0.0 | ACh 100% | yes |
| fca2022 | lamina monopolar neuron L1 | exact | 1 | 1,776 | Glu | 1.5 | 0.55 | 0.12 | 0.06 | 1.5 | 0.01 | 0.02 | 0.01 | 0.01 | Glu 100% | yes |
| fca2022 | lamina monopolar neuron L2 | exact | 1 | 1,779 | ACh | 0.98 | 0.53 | 1.31 | 0.06 | 0.07 | 0.01 | 0.01 | 0.0 | 0.01 | ACh 100% | yes |
| fca2022 | lamina monopolar neuron L3 | exact | 1 | 1,772 | ACh | 0.62 | 0.32 | 0.83 | 0.07 | 0.08 | 0.02 | 0.02 | 0.01 | 0.0 | ACh 100% | yes |
| fca2022 | lamina monopolar neuron L4 | exact | 1 | 1,769 | ACh | 0.92 | 0.42 | 1.17 | 0.11 | 0.1 | 0.05 | 0.02 | 0.0 | 0.01 | ACh 100% | yes |
| fca2022 | lamina monopolar neuron L5 | exact | 1 | 1,787 | ACh | 0.88 | 0.43 | 1.2 | 0.08 | 0.06 | 0.01 | 0.04 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | lamina intrinsic amacrine neuron Lai | exact | 1 | 84 | Glu | 1.16 | 0.47 | 0.1 | 0.09 | 1.16 | 0.02 | 0.15 | 0.01 | 0.0 | Glu 100% | yes |
| fca2022 | lamina wide-field 1 neuron | alias | 1 | 361 | ACh | 0.79 | 0.4 | 1.04 | 0.04 | 0.1 | 0.03 | 0.04 | 0.0 | 0.01 | ACh 100% | yes |
| fca2022 | lamina wide-field 2 neuron | alias | 1 | 383 | ACh | 1.07 | 0.54 | 1.38 | 0.16 | 0.13 | 0.05 | 0.09 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | medullary intrinsic neuron Mi1 | exact | 1 | 1,773 | ACh | 1.38 | 0.69 | 1.91 | 0.07 | 0.08 | 0.0 | 0.03 | 0.0 | 0.02 | ACh 100% | yes |
| fca2022 | medullary intrinsic neuron Mi4 | exact | 1 | 1,772 | GABA | 1.42 | 0.74 | 0.14 | 2.25 | 0.05 | 0.01 | 0.72 | 0.01 | 0.0 | GABA 100% | yes |
| fca2022 | medullary intrinsic neuron Mi9 | exact | 1 | 1,775 | Glu | 2.59 | 0.84 | 0.19 | 0.06 | 2.59 | 0.0 | 0.1 | 0.0 | 0.01 | Glu 100% | yes |
| fca2022 | medullary intrinsic neuron Mi15 | exact | 1 | 1,151 | DA? | 1.36 | 0.67 | 1.25 | 0.18 | 0.11 | 1.97 | 0.12 | 0.0 | 0.0 | ACh 100% | NO |
| fca2022 | transmedullary neuron Tm1 | exact | 1 | 1,777 | ACh | 1.04 | 0.57 | 1.53 | 0.07 | 0.07 | 0.02 | 0.06 | 0.01 | 0.01 | ACh 100% | yes |
| fca2022 | transmedullary neuron Tm2 | exact | 1 | 1,766 | ACh | 1.1 | 0.56 | 1.58 | 0.07 | 0.07 | 0.01 | 0.05 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | transmedullary neuron Tm3a | alias | 1 | 2,054 | ACh | 0.89 | 0.47 | 1.34 | 0.09 | 0.09 | 0.0 | 0.75 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | transmedullary neuron Tm4 | exact | 1 | 1,670 | ACh | 0.95 | 0.53 | 1.47 | 0.13 | 0.04 | 0.03 | 0.16 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | transmedullary neuron Tm9 | exact | 1 | 1,771 | ACh | 1.17 | 0.5 | 1.48 | 0.07 | 0.07 | 0.04 | 0.08 | 0.0 | 0.02 | ACh 100% | yes |
| fca2022 | transmedullary neuron Tm20 | exact | 1 | 1,762 | ACh | 0.72 | 0.42 | 1.23 | 0.05 | 0.12 | 0.01 | 0.27 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | transmedullary neuron Tm5c | exact | 1 | 750 | Glu | 1.33 | 0.46 | 0.15 | 0.13 | 1.33 | 0.03 | 0.03 | 0.02 | 0.0 | Glu 100% | yes |
| fca2022 | transmedullary neuron Tm29 | exact | 1 | 544 | ACh | 0.86 | 0.56 | 1.24 | 0.07 | 0.18 | 0.05 | 0.09 | 0.0 | 0.0 | Glu 100% | NO |
| fca2022 | transmedullary Y neuron TmY4 | exact | 1 | 562 | ACh | 0.94 | 0.5 | 1.45 | 0.04 | 0.14 | 0.0 | 0.1 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | transmedullary Y neuron TmY5a | exact | 1 | 1,364 | Glu | 1.8 | 0.62 | 0.21 | 0.1 | 1.8 | 0.01 | 0.09 | 0.01 | 0.0 | Glu 100% | yes |
| fca2022 | transmedullary Y neuron TmY14 | exact | 1 | 477 | Glu | 1.9 | 0.68 | 0.16 | 0.03 | 1.9 | 0.0 | 0.04 | 0.0 | 0.0 | Glu 44%, ACh 37%, unk 19% | yes |
| fca2022 | centrifugal neuron C2 | exact | 1 | 1,745 | GABA | 0.8 | 0.42 | 0.13 | 1.18 | 0.03 | 0.0 | 0.04 | 0.02 | 0.01 | GABA 100% | yes |
| fca2022 | centrifugal neuron C3 | exact | 1 | 1,779 | GABA | 1.19 | 0.65 | 0.13 | 1.92 | 0.04 | 0.01 | 0.11 | 0.0 | 0.01 | GABA 100% | yes |
| fca2022 | distal medullary amacrine neuron Dm3 | fuzzy | 3 | 3,128 | Glu | 1.25 | 0.42 | 0.27 | 0.06 | 1.25 | 0.0 | 0.1 | 0.02 | 0.0 | Glu 100% | yes |
| fca2022 | distal medullary amacrine neuron Dm8 | fuzzy | 2 | 1,104 | Glu | 1.84 | 0.61 | 0.24 | 0.12 | 1.84 | 0.04 | 0.59 | 0.01 | 0.0 | Glu 100% | yes |
| fca2022 | distal medullary amacrine neuron Dm9 | exact | 1 | 273 | Glu | 1.89 | 0.71 | 0.17 | 0.04 | 1.89 | 0.0 | 0.02 | 0.0 | 0.0 | Glu 100% | yes |
| fca2022 | distal medullary amacrine neuron Dm10 | exact | 1 | 626 | GABA | 0.95 | 0.55 | 0.21 | 1.59 | 0.1 | 0.0 | 0.17 | 0.0 | 0.01 | GABA 100% | yes |
| fca2022 | distal medullary amacrine neuron Dm11 | exact | 1 | 158 | Glu? | 1.67 | 0.5 | 0.58 | 0.74 | 1.67 | 0.0 | 2.25 | 0.0 | 0.0 | Glu 100% | yes |
| fca2022 | distal medullary amacrine neuron Dm12 | exact | 1 | 276 | Glu | 1.25 | 0.44 | 0.11 | 0.03 | 1.25 | 0.0 | 0.07 | 0.0 | 0.0 | Glu 100% | yes |
| fca2022 | proximal medullary amacrine neuron Pm2 | fuzzy | 2 | 343 | GABA | 1.74 | 0.94 | 0.24 | 2.87 | 0.03 | 0.0 | 0.08 | 0.0 | 0.0 | GABA 100% | yes |
| fca2022 | proximal medullary amacrine neuron Pm4 | exact | 1 | 168 | GABA | 1.5 | 0.82 | 0.22 | 2.22 | 0.2 | 0.08 | 1.12 | 0.0 | 0.0 | GABA 100% | yes |
| fca2022 | lobula columnar neuron LC10 | fuzzy | 7 | 960 | ACh | 0.83 | 0.47 | 1.31 | 0.11 | 0.06 | 0.0 | 0.28 | 0.0 | 0.0 | ACh 100% | yes |
| fca2022 | lobula columnar neuron LC12 | exact | 1 | 498 | ACh | 0.85 | 0.48 | 1.28 | 0.19 | 0.2 | 0.0 | 0.28 | 0.03 | 0.0 | ACh 100% | yes |
| fca2022 | lobula columnar neuron LC17 | exact | 1 | 353 | ACh | 0.77 | 0.47 | 1.26 | 0.11 | 0.06 | 0.01 | 0.08 | 0.02 | 0.0 | ACh 100% | yes |
| fca2022 | outer photoreceptor cell | alias | 1 | 3,377 | none | 0.36 | 0.19 | 0.08 | 0.06 | 0.04 | 0.02 | 0.01 | 0.01 | 0.36 | His 100% | - |
| fca2022 | photoreceptor cell R7 | fuzzy | 4 | 1,300 | none | 0.39 | 0.18 | 0.13 | 0.08 | 0.05 | 0.03 | 0.01 | 0.02 | 0.39 | His 100% | - |
| fca2022 | photoreceptor cell R8 | fuzzy | 4 | 1,329 | none | 0.34 | 0.19 | 0.42 | 0.06 | 0.04 | 0.02 | 0.07 | 0.01 | 0.31 | His 100% | - |
| fca2022 | dorsal rim area | fuzzy | 2 | 158 | ACh | 1.03 | 0.5 | 1.03 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | His 100% | NO |

## 5. Unmatched source labels

| source | source_name | source cells | note |
|---|---|---|---|
| davie2018 | Tyraminergic | 122 | Table S2 subcluster of 64 'Tyraminergic' (Tdc2 2.59, Tbh 0.11); MaleCNS has no tyramine NT label and no tyraminergic type names |
| davie2018 | dorsal_Fan-shaped_Body | 232 | Table S2 res2 cluster 61 'dFB' (mapped to R23E10 sorted cells generated in the study); VGlut 4.48. Candidate = FB tangential neurons of dorsal layers (FB6-FB8 types) but the R23E10 -> EM-type correspondence is not established here; not mapped |
| davie2018 | ITP | 25 | label 'ITP' (25 cells) but ITP mean log1p cp10k only 0.41 (vs 2.63 in 'Mip/ITP', 1.76 in 'LNv'); the MaleCNS type ITP (7 cells) is not evidenced by this cluster's expression -> not mapped |
| davie2018 | Mip/ITP | 9 | 9 cells; Mip 1.64, ITP 2.63, Dh31 4.54; no confident MaleCNS type (candidates ITP, DMS not distinguishable) |
| davie2018 | Capa | 73 | Table S2 res8 cluster 119 'Capa neurons' (Diesner 2018); Capa 4.28 but Ms 6.65 too (73 cells); MaleCNS CAPA (2 cells) / DMS (6) are far smaller endocrine types -> not mapped |
| davie2018 | CCAP | 9 | Table S2 res8 cluster 151 'CCAP' (Luan 2006); no CCAP-named type in MaleCNS v1.0 |
| davie2018 | FMRFa | 44 | FMRFa 2.72; MaleCNS FMRFa_Tv is a VNC endocrine type (Davie is brain-only) -> not mapped |
| davie2018 | AstA/NPF | 20 | NPF 3.96, AstA 4.80, ChAT-high; no confident MaleCNS type (NPFL1-I is 2 serotonergic cells) |
| davie2018 | AstA/Nplp1 | 83 | AstA 6.33, Gad1 3.30; MaleCNS AstA1 is a 2-cell pair; cluster (83 cells) is broader -> not mapped |
| davie2018 | CCHa1 | 22 | no CCHa1-named MaleCNS type |
| davie2018 | Proc/Ms | 71 | Proc 1.46, Ms 0.55, Mip 1.07; candidate DMS (6 cells) not evidenced |
| davie2018 | Proc/Gpb5 | 52 | Proc 5.25; no Proc-named MaleCNS type |
| davie2018 | Mip | 814 | Table S2 res2 cluster 17 'Mip' (Carlsson 2010; Min 2016); 814 cells, Mip 0.70 -> a broad cluster, no MaleCNS Mip type |
| davie2018 | Mip/OCT | 26 | 26 cells; Tdc2 2.83, Mip 4.25, Dh31 4.81 (tyraminergic/octopaminergic Mip+); no MaleCNS correspondence established |
| davie2018 | Peptidergic | 155 | Table S2 res2 cluster 46 'Peptidergic'; heterogeneous (Proc 2.32); no MaleCNS type |
| davie2018 | Gr43a | 79 | Table S2 res8 cluster 129 'Gr43a neurons' (Miyamoto & Amrein 2014); Gr43a 1.17; no Gr43a-named brain type in MaleCNS |
| davie2018 | Hsp | 668 | heat-shock / stress signature cluster, not a cell type |
| davie2018 | Pm3 | 82 | Table S2 res2 cluster 73 'Pm3' (Erclik 2017 markers) but the cluster is glutamatergic (VGlut 3.97, Gad1 0.20) whereas MaleCNS Pm3 is GABA 100% (Pm numbering was reorganised in the Nern 2025 optic-lobe nomenclature) -> name match not trusted |
| davie2018 | Astrocyte-like | 1,421 | glia |
| davie2018 | Ensheathing_glia | 1,517 | glia |
| davie2018 | Cortex_glia | 239 | glia |
| davie2018 | Perineurial_glia | 259 | glia |
| davie2018 | Subperineurial_glia | 217 | glia |
| davie2018 | Chiasm_glia | 86 | glia |
| davie2018 | Plasmatocytes | 329 | hemocytes |
| fca2022 | transmedullary Y neuron TmY8 | 390 | MaleCNS v1.0 has no type named TmY8 (renamed in the Nern 2025 optic-lobe nomenclature; correspondence not established here) |
| fca2022 | ocellus retinula cell | 61 | ocellar photoreceptors are not in the MaleCNS node set (OCG types are ocellar-ganglion interneurons) |
| fca2022 | photoreceptor-like | 117 | no type |
| fca2022 | unannotated | 44,906 | unannotated nuclei |
| fca2022 | artefact | 216 | artefact |
| fca2022 | cone cell | 5,397 | non-neuronal |
| fca2022 | epithelial cell | 4,804 | non-neuronal |
| fca2022 | ensheathing glial cell | 1,855 | glia |
| fca2022 | adult brain perineurial glial cell | 1,640 | glia |
| fca2022 | skeletal muscle of head | 1,579 | non-neuronal |
| fca2022 | pigment cell | 1,364 | non-neuronal |
| fca2022 | adult lamina epithelial/marginal glial cell | 984 | glia |
| fca2022 | adult reticular neuropil associated glial cell | 954 | glia |
| fca2022 | optic-lobe-associated cortex glial cell | 570 | glia |
| fca2022 | hemocyte | 352 | non-neuronal |
| fca2022 | adult optic chiasma glial cell | 322 | glia |
| fca2022 | subperineurial glial cell | 279 | glia |
| fca2022 | pericerebral adult fat mass | 198 | non-neuronal |
| fca2022 | adult fat body | 103 | non-neuronal |
| fca2022 | adult brain cell body glial cell | 67 | glia |
| fca2022 | perineurial glial sheath | 6 | glia |
| davie2018 | 41 unannotated numeric res.2 clusters | 35,818 | 'Unannotated' in Table S2 |

## 6. Caveats

* Sex: both sources are mixed-sex; MaleCNS is male. FCA male-only rows are in `expression_central.csv` (`sex_subset = male`).
* Age: FCA 5 d; Davie 0-50 d pooled. Davie's optic-lobe clusters come from whole-brain dissections that include the optic lobes.
* Cluster purity: `class`-tier labels (Poxn, Clock, Serotonergic, Octopaminergic, MBON, ALPN) pool many EM types with different receptor profiles; receptor values for them are population means, usable as a class prior only. The NT-defined rows (rule `query:nt ...`) additionally reach DN / CX / LB2b types (table 3e) and are circular for any transmitter check.
* Tier caveats: FCA `Tm29` exact is suspect (`CAUTION`, section 2); FCA `Tm3a` -> Tm3 alias is a subset; the FCA JO / auditory pair has inverted specificity; `exact` for the Davie / FCA optic-lobe names means the same name in MaleCNS v1.0, not verified same cells.
* FCA cp10k is normalised over 13,056 retained genes (inflated vs Davie); compare within a source, or use `frac_expr`. The marker-call thresholds are per source for the same reason.
* Expression is not conductance: bin before use (`NT_INTEGRATION.md` section 6).
* The optic-lobe rows here duplicate what the Oezel 2021 / Davis 2020 / Nern 2025 sources cover at higher resolution; they are kept because they are an independent adult (FCA) measurement and because the Davie / FCA exact-name hits gave the Pm3, TmY8 and Tm29 nomenclature warnings above.

## 7. Round-2 change log (fix:central; every changed number)

Applied from `docs/audits/receptor_verification.md`, verify:tables:central. Data rows of both CSVs are unchanged by the
rebuild (expression 560 x 54 genes byte-identical; map 1,150 rows identical except `evidence`); the coverage tables in
section 3 are identical to round 1 (verified by diff of the regenerated `out/coverage_central.md` against the round-1 text).

| item | round 1 | round 2 | where |
|---|---|---|---|
| builders | scratchpad `agg_davie.py` / `agg_fca.py` / `build_central_map.py` with absolute paths | `scripts/build_central_agg_davie.py`, `build_central_agg_fca.py`, `build_central_map.py`; relative paths, run order in each docstring; CSV headers cite them | scripts/, CSV headers |
| T1 evidence | `ort 2.32` | `ort 0.97 mean log1p cp10k, 55 % of cells (mean cp10k 3.04)` (Davie, 428 cells) | type_map_central.csv line 559 (data row 546), section 4 (iii) |
| best-tier counts | exact 37 / alias 3 / fuzzy 127 / class 409; 86 unmatched rows | exact 36 / alias 3 / fuzzy 128 / class 409 (576 types); 1,063 matched + 87 unmatched rows (sort by tier order before `drop_duplicates`) | section 3, `out/central_summary.json` |
| PAM raw output | 271,752 (unrestricted feather total, quoted as if node-restricted) | 222,136 node-restricted / 271,752 unrestricted; every raw number now states its base; table 3d carries both bases | sections 2, 3, 3d |
| FCA Mi15 ple | 0.66 | 0.75 pooled (282 nuclei) / 0.63 male (114); DAT 1.97, VAChT 1.25 unchanged | section 2, 4 (iv), map evidence |
| coverage prose | "nothing in the CX columnar system, the DNs, the gustatory second-order cells is covered" | NT-defined class rows reach 13 DN types (23 cells), 12 non-ring CX types (94 cells: ExR2/3, FB1C/1H/2A/4L/4M/4Y/5H, PFGs, PFR_a, EL) and LB2b (3 cells), circular for a transmitter check; EPG/PEN/PFN/hDelta still uncovered | section 3 headline, table 3e, map evidence |
| Tm29 | exact, cross-check `-` (marker-silent) | exact + `CAUTION` (suspect), cross-check `NO` (ACh 0.86 vs Glu 100 %) | map evidence, sections 2 and 4 |
| marker-call rule | score >= 1.0 both sources | score >= 1.0 (davie2018) / 0.5 (fca2022) and best-marker frac_expr >= 0.3 | script docstring, section 4 |
| cross-check tally | agree 68 / marker-silent 40 / disagree 3 | agree 95 / marker-silent 12 / disagree 4 (Clock, Mi15, Tm29, dorsal rim area); the nine optic FCA labels with MaleCNS ACh 100 % (L2-L5, Tm3a, Tm4, Tm20, TmY4, LC10/12/17, T2-T5) now read `ACh` / `ACh?` | section 4 |
| h5ad size | "2,644,193,366 B (>2 GB)" | dropped (no URL recorded, size unverifiable) | section 1 |
| Clock superclasses | not noted | s-LNv + 5thsLNv_LNd6 visual_projection, l-LNv ol_intrinsic: 7 of 10 types / 32 of 52 cells are central-brain (the verification record's "44 of 52" does not reproduce: 52 - 8 - 8 - 4 = 32) | section 2, map evidence, `out/coverage_central.md` |
| Tm3a alias | "Tm3a -> Tm3 (single type)" | marked SUBSET (about half of Tm3) | map evidence, section 2 |
| JO / auditory | not noted | inverted specificity noted (541 nuclei -> 34 types / 672 cells; 706 nuclei -> 10 types / 111 cells) | map evidence, section 2 |
| non-PAM DA class raw | 364,450 (base unstated) | 364,450 node-restricted / 596,183 unrestricted (the record's "341,252 / 468,434" could not be reproduced from the shipped selection of 30 types / 78 cells) | table 3d |

Consumers after the rebuild: `scripts/build_receptor_table.py` `load_central()` reads the new files (198 profiles, 1,063
map rows, 189 circular rows) and `tests/test_receptor_model.py` passes on CPU (8 passed, 9.8 s); `receptors_by_type.csv`
was not regenerated here (it belongs to the receptor-table task).
