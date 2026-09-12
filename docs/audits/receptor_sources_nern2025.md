# Source record: Nern et al. 2025 (optic-lobe cell-type inventory, transmitter predictions, cross-dataset names)

Source key `nern2025`. Written 2026-09-11 by the acquire / map task of the NT-integration workflow
(`docs/NT_INTEGRATION.md`, step 1-2). Everything below is reproducible with
`PYTHONIOENCODING=utf-8 python scripts/build_type_map_nern2025.py` (CPU, ~30 s; needs the raw MaleCNS weights
feather for the `out_syn_raw` column, otherwise it falls back to `c.W`).

## 1. Citation and licence

Nern A, Loesche F, Takemura S-y, Burnett LE, Dreher M, Gruntman E, Hoeller J, Huang GB, Januszewski M,
Klapoetke NC, Koskela S, Longden KD, Lu Z, Preibisch S, Qiu W, Rogers EM, Seenivasan P, Zhao A, Bogovic J,
Canino BS, Clements J, Cook M, Finley-May S, Flynn MA, Fragniere AMC, Hameed I, Hayworth KJ, Hopkins GP,
Hubbard PM, Katz WT, Kovalyak J, Lauchie SA, Leonard M, Lohff A, Maldonado CA, Mooney C, Okeoma N, Olbris DJ,
Ordish C, Paterson T, Phillips EM, Pietzsch T, Salinas JR, Rivlin PK, Schlegel P, Scott AL, Scuderi LA, Takemura S,
Talebi I, Thomson A, Trautman ET, Umayam L, Walsh C, Walsh JJ, Xu CS, Yakal EA, Yang T, Zhao T, Funke J, George R,
Hess HF, Jefferis GSXE, Knecht C, Korff W, Plaza SM, Romani S, Saalfeld S, Scheffer LK, Berg S, Rubin GM, Reiser MB.
**Connectome-driven neural inventory of a complete visual system.** *Nature* 641, 1225-1237 (2025).
doi:[10.1038/s41586-025-08746-0](https://doi.org/10.1038/s41586-025-08746-0). PMC12119369. Preprint:
bioRxiv 10.1101/2024.04.16.589741.

Licence: the article and its supplementary information are **CC BY 4.0** (article licence statement). The
Cell Type Explorer archive (Zenodo 10.5281/zenodo.15015112) is CC BY 4.0. The supplementary-code repository
(`reiserlab/male-drosophila-visual-system-connectome-code`) is GPL-3.0; only the `params/` copies of the
Nature supplementary tables were taken from it (they are byte-identical to the Nature files, see hashes).
The MaleCNS v1.0 annotation columns used for the `malecns_*` alias systems are CC BY (male-cns.janelia.org).

Data availability (paper): the connectome is served by neuPrint as `optic-lobe:v1.0`
(https://neuprint.janelia.org/?dataset=optic-lobe:v1.0); the Cell Type Explorer at
https://reiserlab.github.io/male-drosophila-visual-system-connectome/ ; code at
https://github.com/reiserlab/male-drosophila-visual-system-connectome-code. No GEO accession (no transcriptome).

**Same tissue as MaleCNS.** The optic-lobe dataset is the right optic lobe of the same male CNS EM volume that
MaleCNS v1.0 is built from ("The complete Central Nervous System ... of a male Drosophila was dissected ...
Proofreading of the visual regions on one side has been completed", Methods). The transmitter predictions are
therefore *not* from an independent animal: they are a separate classifier (trained on optic-lobe ground truth,
Supplementary Table 5) applied to the same presynapses. The independent labels in this source are (a) the
experimental validation set (Sup_Table_5: FISH / EASI-FISH / bulk RNA-seq / antibody, 148 rows) and (b) the
FlyWire predictions carried in Sup_Table_7 (female FAFB volume, Eckstein et al. 2024 classifier).

## 2. Files acquired (`data/external/nern2025/`, git-ignored)

All downloaded 2026-09-11 with curl. Nature's `MediaObjects` host serves the ESM files without login (the
article HTML itself redirects to an IdP; only the article page, not the files, is gated).

| file | URL | size (bytes) | SHA-256 |
|---|---|---|---|
| `nature_esm/41586_2025_8746_MOESM1_ESM.pdf` (Supplementary Information: Supplementary Fig. 1, the 70-page per-type catalogue) | https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-08746-0/MediaObjects/41586_2025_8746_MOESM1_ESM.pdf | 64,469,458 | `a142a82b327a48092d064d245c2bb9d995b0b7a4ee1231af5fbbb01628665d6f` |
| `nature_esm/41586_2025_8746_MOESM2_ESM.pdf` (Reporting Summary, 3 pp.) | .../41586_2025_8746_MOESM2_ESM.pdf | 2,633,126 | `709af476b254271f70fe3ae4cc29921a9fc07e05faa1aa781930368e151205cd` |
| `nature_esm/41586_2025_8746_MOESM3_ESM.pdf` (Peer Review File, 39 pp.) | .../41586_2025_8746_MOESM3_ESM.pdf | 602,033 | `7d960c1c6bdf0f244aae2d4fc71aac9dc5d58339121b649c72c59029a21e73d3` |
| `nature_esm/41586_2025_8746_MOESM4_ESM.zip` (Supplementary Tables 1-7 + guide) | .../41586_2025_8746_MOESM4_ESM.zip | 687,758 | `ef34f64c74e157fdec06312e3a1d6e567ae48a383944dd99a3f854e7ba593c93` |
| `nature_esm/MOESM4_unzipped/Sup_Table_1_Cell-types_and_counts_final.xlsx` | (in MOESM4 zip) | 29,923 | `095a80142923b09e17e5025f6d0e8b8aa0faed0adf5fbad17aff793e30b284a8` |
| `nature_esm/MOESM4_unzipped/Sup_Table_2_nonprimary_connectivity_final.xlsx` | (in MOESM4 zip) | 13,984 | `e6d0d66e00236515cc040b8cd10dc8a9c6fd6d2c8c62905ee96d51db9b536a45` |
| `nature_esm/MOESM4_unzipped/Sup_Table_3_Columnar_cell_types_locations_with_py_final.xlsx` | (in MOESM4 zip) | 123,972 | `f18662c6972effa5c035cd8b31f59ed34c870faeda614bc208dd249181cc38f9` |
| `nature_esm/MOESM4_unzipped/Sup_Table_4_Mi1_T4_alignment_final.xlsx` | (in MOESM4 zip) | 64,250 | `0b82f07c4b8173c9620a290d199c3e63f7bd74b6ddff19845e9cbbc9689e116d` |
| `nature_esm/MOESM4_unzipped/Sup_Table_5_Neurotransmitter_validation_final.xlsx` | (in MOESM4 zip) | 17,321 | `958284e7a46ba7e638eca75408532c644bf791476b0feba9bba6abecdfc6d5ee` |
| `nature_esm/MOESM4_unzipped/Sup_Table_6_Split-GAL4-lines_final.xlsx` | (in MOESM4 zip) | 59,360 | `eb7977e05cf6dd2066af9871f8fbc39330371b8a18a5ee6ff8405e7e2854f01a` |
| `nature_esm/MOESM4_unzipped/Sup_Table_7_MatchingCellTypes_final.xlsx` | (in MOESM4 zip) | 655,805 | `d9e1b35282abb7da6cb5ec6ffa84bed4e9231bcde5994d7e41711e2751b61c8c` |
| `nature_esm/MOESM4_unzipped/Supplementary Tables Guide.pdf` | (in MOESM4 zip) | 85,891 | `5a333b1a21d328f4336679f8282e189f44b328db2fd2ac90cea67e8c714b9614` |
| `github_code_params/Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx` (identical to Sup_Table_1) | https://raw.githubusercontent.com/reiserlab/male-drosophila-visual-system-connectome-code/main/params/Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx (commit dbafc73, 2025-06-17) | 29,923 | `095a80142923b09e17e5025f6d0e8b8aa0faed0adf5fbad17aff793e30b284a8` |
| `github_code_params/Nern-et-al_SuppTable05_Neurotransmitter_validation.xlsx` (1 byte differs from Sup_Table_5: same content) | .../params/Nern-et-al_SuppTable05_Neurotransmitter_validation.xlsx | 17,322 | `38baa04d0d8b13eccbfb28193d7d38451679ccb09674341ec21ff897598b0602` |
| `github_code_params/Primary_cell_type_table.xlsx` (734 types x main group / figure group) | .../params/Primary_cell_type_table.xlsx | 28,048 | `5ed863fbe1ccb6e2f2c8e9f2271fdc5861394df6bb47b25b30174d3451c7b501` |
| `github_code_params/confusion_df.csv` (Fig. 4b synapse-level NT confusion matrix, 7 x 7) | .../params/confusion_df.csv | 956 | `d466b63c27c82229bab072a1e751ab16b9a494c7672cede68ea2127003da9a48` |
| `github_code_params/gt_count.csv` (ground-truth counts per NT: types / bodies / train / test synapses) | .../params/gt_count.csv | 246 | `6ff7a70437cdf98e559dc11e6a9361bb428c25b0d6b67c1a8c1cd79693b1dc26` |
| `github_code_params/Edge_cell_types.xlsx` | .../params/Edge_cell_types.xlsx | 10,358 | `c675d894a9ad2a6d7fcb4a37c7ae0c004f1a9fea4fa95ae3b1792d1b15fb0246` |
| `github_code_params/uncurated_vpns_vcns.xlsx` (old_type -> type renames for 103 VPN / VCN) | .../params/uncurated_vpns_vcns.xlsx | 10,714 | `9cb02a3ba0005dac736ef9c79db63809754db05648cfa199afd1bc288e331b72` |
| `github_code_params/LICENSE` (GPL-3.0, code repo) | .../LICENSE | 35,149 | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |

Not downloaded: the Zenodo Cell Type Explorer archive (3.0 GB of static HTML, MD5 7eda88b7eceb09fed99ef221fd60304e;
adds nothing beyond the tables), and the neuPrint `optic-lobe:v1.0` per-synapse probabilities (needs a token;
the per-type consensus in Sup_Table_1 is what the plan asks for).

## 3. What the tables contain and how the labels were made

**Sup_Table_1** (778 rows = 732 right-side types `*_R` + 97 left-side instances `*_L` of 46 bilateral types; 51
types exist only as `_L` rows): `cell type`, `instance`, `no. of cells`, `main groups` (ONIN 149, ONCN 95, VPN
385, VCN 110, other 39 rows), `bodyId in figures`, `predicted neurotransmitter` in {ACh 273, Glu 175, GABA 130,
His 8, OA 8, 5HT 3, Dop 2, unclear 179} (row counts; per type: ACh 257, Glu 158, GABA 127, His 8, OA 6, 5HT 2,
Dop 2, unclear 172). Left and right instances of the same type never disagree (0 contradictions).

Prediction method (Methods, "Neurotransmitter prediction"): a 3-D CNN on 640 nm EM cubes classifies each of
7,014,581 presynapses into 7 transmitters, trained on 59 ground-truth types (Sup_Table_5 `Part_of_training_data`
= yes; 60 rows), 70/10/20 split by neuron. Per-cell-type label = most frequent presynaptic class if the type has
>= 100 presynapses and confidence >= 0.5, else `unclear`. **Consensus rule:** types predicted Dop / OA / 5HT
without high-confidence experimental support in Sup_Table_5 are set to `unclear` (the monoamines were
under-represented in training and over-predicted). The `predicted neurotransmitter` column of Sup_Table_1 is
this consensus (its counts equal the `OL_transmitter_pred` counts of Sup_Table_7). Synapse-level accuracy on
held-out data (confusion_df.csv diagonal): ACh 0.933, Glu 0.879, GABA 0.893, His 0.961, Dop 0.971, OA 0.907,
5HT 0.893. There is **no per-type confidence column** in the supplement; the only confidence tiers available
per type are (i) called vs `unclear`, (ii) in the ground-truth set or not, (iii) experimentally validated or not.

**Sup_Table_5** (148 rows, 147 distinct names; 146 are Sup_Table_1 types plus the bracket groups `[Cm11]` =
one or more of Cm11a-d and `[Pm2]` = Pm2a/b): `Cell Type`, `Driver line used`, `Observed signal (new FISH data
only)`, `Method` (EASI-FISH 62, bulk TAPIN RNA-seq 57, antibody 15, FISH 9, FACS RNA-seq 2, ...), `Inferred
transmitter` (ACh 60, Glu 37, GABA 29, OA 6, His 3, His+ACh 3, 5HT 2, Dop 2, ACh+Dop 1, unclear 5),
`Reference(s)`, `Part_of_training_data` (yes 60 / no 88), `Notes`. Rows with `yes` are the classifier's
ground truth and do not validate `nern_nt` independently; the 88 `no` rows do.

**Sup_Table_7** (733 rows; 732 OL types + one FlyWire-only row for LTe12 = LoVP109): `group`, `OL_type`,
`Matsliah_type` (FlyWire optic-lobe names, Matsliah et al. 2024), `Schlegel_type` (FlyWire whole-brain names,
Schlegel et al. 2024), `hemibrain_type`, cell counts in each dataset, `matched as` (1-to-1 644, 2-to-1 32,
3-to-1 19, unmatched 13, 6-to-6 6, 6-to-1 6, 1-to-2 4, 4-to-1 4, 1-to-many 2, 1-to-3 1, 1-to-8 1),
`URL` (neuroglancer side-by-side), `OL_transmitter_pred`, `FW_transmitter_pred`, `Notes`. The paper calls these
matches "preliminary"; >98 % of types (99.8 % of cells) matched.

## 4. Derived tables (redistributable, `flyverse/data/`)

### `flyverse/data/type_map_nern2025.csv` (784 rows)

Columns `source_name, malecns_type, tier, evidence, n_cells_malecns` as specified, plus `nern_main_group,
nern_n_cells_R, nern_n_cells_L, nern_nt, fw_nt, t7_matched_as, validated_nt, validated_method,
validated_reference, validated_in_training, malecns_superclass, malecns_nt, malecns_nt_share,
malecns_n_unknown_nt, out_syn_W, out_syn_raw`. NT vocabulary is the model's (`acetylcholine gaba glutamate
histamine dopamine octopamine serotonin unclear`, `a+b` for a source that lists two).

Tiers used:

- **exact** (732 rows): identical type name in Sup_Table_1 and in `cache/neurons.parquet` `type`. All 732 Nern
  names are MaleCNS names (MaleCNS adopted the Nern nomenclature), so no `alias` or `fuzzy` mapping was needed
  and none was made. 684 of the 732 are MaleCNS optic-lobe-superclass types; 48 are central types that Nern
  lists because they have optic-lobe arbors (VCN / `other`: OA-AL2i1-3, OA-ASM1, 5-HTPMPV03, DNp11, PLP*,
  KCg-s1, ...); 157 cells.
- **class** (13 rows, 957 cells): MaleCNS `<prefix>_unclear` bins (cells whose subtype could not be resolved)
  mapped to the set of Nern subtypes of that prefix when every *called* subtype carries the same prediction and
  at least half of the subtypes are called: `LC10_unclear` (LC10a/b/c-1/c-2/d/e, 5 ACh + 1 unclear),
  `LLPC_unclear`, `LPC_unclear`, `LPLC_unclear`, `MeTu4_unclear` (all ACh), `MeVP6_unclear` -> MeVP6 (Glu),
  `PVLP046_unclear` -> PVLP046 (GABA), `R7_unclear` (R7d/p/y His, 404 cells), `R8_unclear` (R8d/p/y His, 442),
  `R7R8_unclear` (His, 85), `T4_unclear` (T4a-d ACh), `T5a_unclear` -> T5a (ACh), `Tm3_unclear` -> Tm3 (ACh).
  Membership rule is in `class_members()` (prefix ending in a digit: prefix or prefix + lowercase subtype
  letters; prefix ending in a letter: the prefix itself or prefix + digits + optional letters, so `Tm` does not
  capture `TmY3`).
- **unmatched** (39 rows, all MaleCNS optic-lobe-superclass types; 146 cells, 0.14 % of OL-superclass cells):
  13 ocellar types / 46 cells (OCG*/OCC*: the ocelli are outside the inventory), 13 region bins / 64 cells with
  no type prefix (`ME_unclear`, `LO_unclear`, `LOP_*`, `LA_ME_unclear`, `LopVC_unclear`, `LpMe_unclear`,
  `OLVp_unclear`, ...), 11 class bins / 33 cells whose subtypes disagree (`Cm_unclear` GABA 20 / Glu 12 / ACh 6,
  `Li_unclear`, `LPi_unclear` Glu 11 / GABA 4, `Pm_unclear`, `Tm_unclear`, `TmY_unclear`, `Y_unclear`,
  `aMe_unclear`, `MeVC_unclear`, `MeVP_unclear`, `LoVP_unclear`), and 2 / 3 cells: `LoVP109` (LM-defined, absent
  from the OL dataset per Sup_Table_7 note) and `Pm7_Li28` (a fused cell). No mapping was invented for these.

### `flyverse/data/type_aliases_nern2025.csv` (14,180 rows)

`malecns_type, alias, system, tier, evidence`. Systems: `flywire_matsliah2024` (230 rows), `flywire_schlegel2024`
(649), `hemibrain` (333) from Sup_Table_7; `malecns_flywireType` (8,137) and `malecns_hemibrainType` (4,831)
from the MaleCNS v1.0 per-cell annotation columns (majority value per type, all superclasses). Tier: `exact`
= alias identical to the MaleCNS name (Sup_Table_7 1-to-1, or >= 90 % of annotated cells); `alias` = different
name, 1-to-1 / >= 90 %; `fuzzy` = n-to-1 or 1-to-n match, composite alias string (`LTe49a,b,d,e,f`,
`AOTU055,AOTU056,...`), or majority < 90 %. Counts: Matsliah exact 73 / alias 147 / fuzzy 10; Schlegel 250 /
331 / 68; hemibrain 133 / 109 / 91; malecns_flywireType 4,550 / 2,706 / 881; malecns_hemibrainType 3,805 /
784 / 242. (The plan's unsuffixed `type_aliases.csv` is written by `scripts/build_type_map_typing.py`, the
typing-source task; this file is the Nern-specific contribution and should be merged there.)

No `expression_nern2025.csv`: the source has no expression data (EM connectome + literature / FISH transmitter
calls only).

## 5. Coverage of MaleCNS (cells and output synapses)

`out_syn_W` = sum of |`c.W`| over each cell's output edges (the model's matrix: sign-0 monoamine / unknown
edges are explicit zeros); `out_syn_raw` = uncapped MaleCNS synapse counts on the model's 167,106-node set
(124.2 M total). `central_or_vnc` = every superclass outside the four optic-lobe ones.

| group | tier | types | cells | cells frac | out_syn_W | frac | out_syn_raw | frac |
|---|---|---|---|---|---|---|---|---|
| ol_intrinsic | exact | 248 | 89,265 | 99.86 % | 43,909,936 | 99.91 % | 43,943,110 | 99.91 % |
| ol_intrinsic | class | 3 | 6 | 0.01 % | 786 | 0.00 % | 893 | 0.00 % |
| ol_intrinsic | unmatched | 20 | 82 | 0.09 % | 38,796 | 0.09 % | 38,796 | 0.09 % |
| ol_intrinsic | untyped | - | 37 | 0.04 % | 599 | 0.00 % | 1,443 | 0.00 % |
| visual_projection | exact | 327 | 9,148 | 99.42 % | 7,263,482 | 99.21 % | 7,296,928 | 99.21 % |
| visual_projection | class | 6 | 18 | 0.20 % | 7,005 | 0.10 % | 7,005 | 0.10 % |
| visual_projection | unmatched | 13 | 35 | 0.38 % | 51,187 | 0.70 % | 51,187 | 0.70 % |
| ol_sensory | exact | 8 | 5,167 | 84.73 % | 536,678 | 82.38 % | 536,678 | 82.38 % |
| ol_sensory | class | 3 | 931 | 15.27 % | 114,798 | 17.62 % | 114,798 | 17.62 % |
| visual_centrifugal | exact | 101 | 533 | 94.67 % | 1,757,998 | 99.75 % | 2,135,218 | 99.79 % |
| visual_centrifugal | class | 1 | 1 | 0.18 % | 536 | 0.03 % | 536 | 0.03 % |
| visual_centrifugal | unmatched | 6 | 28 | 4.97 % | 3,333 | 0.19 % | 3,422 | 0.16 % |
| visual_centrifugal | untyped | - | 1 | 0.18 % | 561 | 0.03 % | 561 | 0.03 % |
| central_or_vnc | exact | 48 | 157 | 0.25 % | 310,134 | 0.46 % | 364,630 | 0.52 % |
| central_or_vnc | class | 1 | 1 | 0.00 % | 151 | 0.00 % | 151 | 0.00 % |
| central_or_vnc | unmatched | 10,968 | 59,129 | 95.59 % | 66,532,248 | 98.21 % | 68,707,176 | 98.11 % |
| central_or_vnc | untyped | - | 2,567 | 4.15 % | 898,911 | 1.33 % | 959,341 | 1.37 % |
| ALL | exact | 732 | 104,270 | 62.40 % | 53,778,228 | 44.29 % | 54,276,564 | 43.71 % |
| ALL | class | 13 | 957 | 0.57 % | 123,276 | 0.10 % | 123,383 | 0.10 % |
| ALL | unmatched | 11,006 | 59,274 | 35.47 % | 66,625,564 | 54.87 % | 68,800,581 | 55.41 % |
| ALL | untyped | - | 2,605 | 1.56 % | 900,071 | 0.74 % | 961,345 | 0.77 % |

Optic-lobe superclasses together (105,252 cells): exact 104,113 (98.92 %), class 956 (0.91 %), unmatched 145
(0.14 %), untyped 38 (0.04 %); by raw output synapses (54,130,575): exact + class 54,035,166 = 99.82 %.

Edge-level (both presynaptic and postsynaptic type at tier exact or class): 91.9 % of the synapses whose
presynaptic cell is in an optic-lobe superclass (`c.W`: 91.99 %, raw: 91.93 %; the remaining 8 % go to
central-brain targets outside the inventory), 99.82 % of those synapses have the presynaptic type matched;
over the whole CNS 40.7 % (`c.W`) / 40.2 % (raw) of synapses have both ends matched and 43.8 % have the
presynaptic end matched. Central-brain coverage is by construction near zero (0.25 % of cells): this source
covers the optic lobe only.

By confidence tier of the *transmitter label* (745 exact + class rows): called (not `unclear`) 573 types,
`unclear` 172. Among the 697 optic-lobe-superclass exact + class types, the 131 `unclear` ones hold 3,452
cells and 1,888,216 raw output synapses (3.49 % of the 54.13 M OL-superclass output synapses; the other 41
`unclear` types are central). Experimentally validated (Sup_Table_5, any method): 145 matched types, 60 of
them classifier ground truth and 85 outside the training set (the independent check).

## 6. Transmitter cross-check (per type, exact + class rows; a type counts when both labels are called)

| comparison | types | agree | disagree | disagreeing cells / raw output synapses |
|---|---|---|---|---|
| Nern 2025 vs MaleCNS model label (`neurons.nt`, mode per type) | 570 | 564 (98.9 %) | 6 | 428 / 253,093 |
| Nern 2025 vs FlyWire prediction (Sup_Table_7) | 544 | 503 (92.5 %) | 41 | 8,525 / 3,075,098 |
| Nern 2025 vs experimental validation (Sup_Table_5) | 140 | 136 | 4 (all co-transmission: Mi15 ACh+Dop, R8p / R8y / HBeyelet His+ACh) | 1,969 / 472,563 |
| MaleCNS model label vs experimental validation | 139 | 135 | 4 (the same four) | 1,969 / 472,563 |
| FlyWire prediction vs experimental validation | 139 | 118 (84.9 %) | 21 | 7,823 / 2,158,233 |

The six Nern-vs-MaleCNS disagreements, with the raw MaleCNS NT columns (`body-neurotransmitters-male-cns-v1.0`):

| type | superclass | cells | raw out syn | Nern | MaleCNS model | MaleCNS `celltype_predicted_nt` (conf) | MaleCNS `ground_truth` | validated (Sup_Table_5) |
|---|---|---|---|---|---|---|---|---|
| Tm31 | ol_intrinsic | 120 | 83,719 | glutamate | gaba | glutamate (0.82) | gaba | - |
| Pm12 | ol_intrinsic | 4 | 79,859 | glutamate | gaba | glutamate (0.60) | gaba | - |
| Li22 | ol_intrinsic | 232 | 49,365 | glutamate | gaba | glutamate (0.77) | gaba | - |
| aMe17a | visual_centrifugal | 2 | 20,900 | glutamate | serotonin (from celltype prediction; consensus unclear) | serotonin (0.56) | - | - |
| LC30 | visual_projection | 57 | 9,819 | acetylcholine | glutamate | acetylcholine (0.87) | glutamate | - |
| LoVP92 | visual_projection | 13 | 9,431 | gaba | acetylcholine | acetylcholine (0.91) | - | - |

In four of the six the two EM classifiers agree (glutamate / ACh) and the MaleCNS `consensus_nt` was overridden by
its `ground_truth` column, which covers 3,318 types / 85,480 cells across the CNS -- far more than any
experimental set, so it is a propagated type-level label (source unstated in the flat files), not a per-type
measurement. Neither Tm31, Pm12, Li22 nor LC30 is in Sup_Table_5. These four (240,762 output synapses) are
sign flips (glutamate -1 vs GABA -1 is sign-neutral under `NT_SIGN`, but not under a receptor model; LC30 ACh +1
vs Glu -1 is a sign flip today).

Nern-vs-FlyWire disagreements are dominated by FlyWire calling GABA where Nern calls glutamate (Dm1, Dm6, Dm9
[FW: ACh], Dm12, Dm16, Dm19, Dm20, Cm7, Cm9, Cm25, Cm34, Mi13, TmY16, LPi34, LPi43, LPi3412, Lai, Pm12, Pm13,
Li36, LoVC16, LoVC26, LT88, LT68, MeVPMe10/11, aMe17c) and by the photoreceptors (FlyWire has no histamine
class: R1-R6 / R8 -> ACh, R7 -> Glu, HBeyelet -> Glu). Where Sup_Table_5 arbitrates (Dm1, Dm9, Dm12, Dm19,
Tm29, TmY16, LPi34, LPi43, Lai, R1-R6, R7p/y, Mi19, OA-ASM1, 5-HTPMPV03, MeVC21, 5thsLNv_LNd6) Nern and MaleCNS
are right and FlyWire wrong in every case. The FlyWire label is therefore the weakest of the three for optic
types and should not out-vote the two male-volume labels.

Rescue of model `unknown` cells: 141 unknown-NT cells sit in 16 matched types; Nern calls 6 of them / 109 cells:
**TmY14** (91 unknown of 477 cells; Nern glutamate, FlyWire glutamate; MaleCNS per-body votes Glu 209 / ACh 177 /
unclear 91, type-level unclear at 0.49) -> glutamate is supported by two labels; **Mi19** (12 / 12; Nern
serotonin, validated serotonin by EASI-FISH, in training set; FlyWire dopamine) -> serotonin (sign 0 under
`NT_SIGN`, but a label for the slow term); aMe8 (3; ACh), MeTu1 (1), T2a (1), T5a_unclear (1; class ACh).

Watch-list from the NT audit (`docs/audits/nt_audit.md`):

| type | cells | raw out syn | MaleCNS model | Nern | FlyWire | validated (method, in training) |
|---|---|---|---|---|---|---|
| Tm5Y | 898 | 438,245 | acetylcholine | acetylcholine | acetylcholine | - |
| TmY14 | 477 | 132,799 | glutamate (mode, 44 %) / unknown 91 | glutamate | glutamate | - |
| Mi19 | 12 | 24,996 | unknown | serotonin | dopamine | serotonin (EASI-FISH, yes) |
| LoVCLo2 | 2 | 14,808 | unknown | unclear | serotonin | - |
| LoVCLo3 | 2 | 40,059 | octopamine | octopamine | octopamine | octopamine (antibody, no) |
| LoVC18 / LoVC22 | 4 / 4 | 29,667 / 27,115 | dopamine | dopamine | dopamine | dopamine (antibody; no / yes) |
| OA-AL2i1 / OA-AL2i2 / OA-ASM1 | 2 / 4 / 4 | 49,845 / 56,142 / 40,967 | octopamine | octopamine | OA / OA / dopamine | octopamine (antibody) |
| T1 | 1,777 | 7,372 | histamine (celltype conf 0.51; per-body `unclear` for 1,776 / 1,777) | unclear | acetylcholine | unclear: no ChAT / VGlut / GAD1 / ple / SerT / Tbh (TAPIN, no) |
| Lat3 / Lat4 | 8 / 2 | 599 / 18 | octopamine / unknown | unclear | serotonin (5-HT-IR Tan) | - |
| Tm29 | 544 | 159,931 | glutamate | glutamate | acetylcholine | glutamate (TAPIN, no) |
| Dm9 | 273 | 247,828 | glutamate | glutamate | acetylcholine | glutamate (TAPIN, yes) |
| Mi13 | 910 | 248,247 | glutamate | glutamate | gaba | - |
| TmY16 | 168 | 144,398 | glutamate | glutamate | gaba | glutamate (EASI-FISH, no) |

Tm5Y: three labels say acetylcholine; the audit's "glutamate" literature entry stays retracted. T1: the model
gives 1,777 cells a histamine (-1) sign from a 0.51-confidence type-level call that the per-body predictions
and the experimental data do not support (T1 expresses none of the standard markers) -- the honest label is
`unknown` / sign 0 (7,372 output synapses, 0.006 % of all).

## 7. Caveats and open points

- Not an independent animal: Nern predictions come from the same presynapses as MaleCNS v1.0 (a differently
  trained classifier); 98.9 % agreement is expected. Independence is provided only by Sup_Table_5 (88 types
  outside the training set) and the FlyWire column (which is the least reliable of the three).
- No per-type confidence number exists in the supplement; `unclear` is the only confidence flag. neuPrint
  `optic-lobe:v1.0` carries per-body confidence (`predictedNt`, `celltypePredictedNt`, `consensusNt`) if a
  numeric tier is needed later (token required).
- Nern `unclear` types: 172 of 732 (3,258 MaleCNS cells, 2.6 % of OL-superclass output synapses), mostly small
  VPN / VCN types and the monoamine-suspect types set to unclear by the consensus rule.
- Sup_Table_7 matches are "preliminary" (paper's word); n-to-1 rows are `fuzzy` in the alias table. The
  MaleCNS annotation columns (`flywireType` / `hemibrainType`) are a second, larger alias source and agree with
  Sup_Table_7 where both exist (e.g. Tm5Y = FlyWire Tm5f).
- The MaleCNS `ground_truth` column overrides the classifier for Tm31 / Pm12 / Li22 (GABA vs Glu) and LC30
  (Glu vs ACh); its provenance should be traced before either side is adopted.
- The bracket groups `[Cm11]` (ACh) and `[Pm2]` (GABA, "probably Pm2a") in Sup_Table_5 were not expanded into
  per-subtype validation rows.
- Sex: same male volume as MaleCNS, so no sex caveat for this source (the FlyWire column is female).
