# Receptor / transmitter expression source: Davis et al. 2020 (key `davis2020`)

Part of the NT-integration workflow (`docs/NT_INTEGRATION.md`, step 1 "acquire and pin", step 2 "type map").
Generated 2026-09-11 by `scratchpad/build_davis2020.py` (session-10 scratchpad; copy of the script is not in the repo,
its logic is described in section 4). Raw files live in `data/external/davis2020/` (git-ignored); derived tables in
`flyverse/data/`.

## 1. Citation and licence

Davis FP, Nern A, Picard S, Reiser MB, Rubin GM, Eddy SR, Henry GL (2020). *A genetic, genomic, and computational
resource for exploring neural circuit function.* eLife 9:e50901. https://doi.org/10.7554/eLife.50901
(bioRxiv 2018: https://doi.org/10.1101/385476; code: https://github.com/fredpdavis/opticlobe; resource site
http://www.opticlobe.com, unreachable on 2026-09-11, ECONNREFUSED).

Method: split-GAL4 / GAL4 driver lines x INTACT or TAPIN-seq nuclear RNA-seq of adult heads (mixed sex except one
male / female pair of T4.T5 samples), 266 samples, 100 drivers, 67 cell types + 3 NT-class "broad" drivers + dissected
tissue + controls. Quantification: TPM (kallisto-style, ENSEMBL r91 / FlyBase 2017_04 gene models); per gene a
bimodal / unimodal mixture model on log TPM gives P(expressed) per sample / driver / cell type.

Licence: the eLife article and its supplementary files are CC BY 4.0 (eLife's standard licence; the article page
states "Open access"). GEO series GSE116969 processed tables are public NCBI data with no separate licence. Our
derived CSVs (`flyverse/data/expression_davis2020.csv`, `flyverse/data/type_map_davis2020.csv`) are per-type
extracts of a small gene panel and carry the citation in their header comment.

## 2. Files acquired (`data/external/davis2020/`)

GEO accession **GSE116969** (Davis & Nern et al.; released 2018-07-11), supplementary directory
`https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116969/suppl/`. Raw FASTQ (SRA) were not downloaded (not
needed; the processed tables are the paper's own quantification).

| file | bytes | SHA-256 | content |
|---|---|---|---|
| GSE116969_README_dataTables.txt | 3,322 | fbd0e7cbf91f42ad6ef44c46c2f98f73a09bb3049804159d56ac8d9812ceb1f9 | README for tables 1-7 |
| GSE116969_dataTable1.transcripts_x_samples_TPM.all_genes.txt.gz | 12,442,683 | b96dd43df9a92ad8a1bb2d12aeb5f4ae34e642d707e2d9ed1ccab20627f7b4a2 | transcript x sample TPM, all samples |
| GSE116969_dataTable2.genes_x_samples_TPM.all_genes.txt.gz | 7,203,098 | 115497e3d60fe1a051e910e84ea29814543c6348b0a32b85a3ada507aed04686 | gene x sample TPM, all genes (coding, noncoding, reporter, ERCC) |
| GSE116969_dataTable3.genes_x_samples_TPM.coding_genes.txt.gz | 6,513,528 | 5c035c18b531995e2fe1b56109c1c8550628b678d305b6942b9e073bf37a3181 | gene x sample TPM, coding genes, **all 266 samples incl. QC-fail** (columns gene_id, gene_name, samples) |
| GSE116969_dataTable4.genes_x_cells_TPM.coding_genes_QCpass.txt.gz | 2,165,840 | 90f3ae41bf501ffb8d4fd7609c76062bf2bf1aea2ad28063f9e955fbe2fcdada | **gene x cell-type mean TPM, QC-pass samples** (13,931 genes x 79 cells) -- the TPM source used |
| GSE116969_dataTable5a.genes_x_samples_TPM.modeled_genes.txt.gz | 4,749,834 | 70333616bfeb6233501b572dc99907b1de63a79304d796f3474dc849c751355a | modeled genes (>= 10 TPM somewhere) x sample TPM |
| GSE116969_dataTable5b.genes_x_samples_p_expression.modeled_genes.txt.gz | 1,396,892 | fe17a2a54bfe2534892e46fdd09f717b73eabe7f4124c19c8cf62fbb1e8008ff | modeled genes x sample P(on) + model columns |
| GSE116969_dataTable6a.genes_x_drivers_TPM.modeled_genes.txt.gz | 2,456,829 | 5e233a9a709268f85c84d1715b48fc511f5b0f1c9f55a2c31ad67467942beb7f | modeled genes x driver TPM (96 drivers, incl. T4.T5_d1_male / _female) |
| GSE116969_dataTable6b.genes_x_drivers_p_expression.modeled_genes.txt.gz | 702,030 | 69fbfc2a50186b8c44f9ee4b5549d4a83555fba9a20c5e601c9954a7d2af63e5 | modeled genes x driver P(on) |
| GSE116969_dataTable7a.genes_x_cells_TPM.modeled_genes.txt.gz | 1,997,850 | 88f3b2168214fa7ff5b9c950f307331eeb484e9d91de52f3e24ed75768fa8d5b | modeled genes x cell TPM (12,377 x 77) |
| GSE116969_dataTable7b.genes_x_cells_p_expression.modeled_genes.txt.gz | 600,566 | 8f827bdd4c64b208281260e5c7643c3887e9bc380b06a24a84cef83f6b14c462 | **modeled genes x cell P(on)** (12,377 x 77 + 9 model columns) -- the p_on source used |
| elife-50901-supp1-v2.xlsx | 57,861 | 8f7890472d7b07e4a04a0fd38716dd5f5e46d94995467a5ceba54d01ab15d361 | eLife Supplementary file 1: A_all_drivers (129 drivers: driverID, type, external ID, celltype, celltype.details, additional expression, how identified, split halves, reference), B_RNAseq_samples (266 samples, QC pass / suboptimal), C_benchmark_entries, D_markerGenes, E_anatomy_details, F_TAPINseq_buffers |

Supp. file 1 URL: `https://elifesciences.org/download/aHR0cHM6Ly9jZG4uZWxpZmVzY2llbmNlcy5vcmcvYXJ0aWNsZXMvNTA5MDEvZWxpZmUtNTA5MDEtc3VwcDEtdjIueGxzeA--/elife-50901-supp1-v2.xlsx`
(Supplementary file 2 is the key-resource table, docx; not downloaded.) Nothing was gated or above 2 GB.

Gene-symbol notes: the tables use FlyBase 2017_04 symbols; `KaiR1D` is `CG8916`, `Octalpha2R` is `CG18208`
(both present); `Oct-TyrR` is present but not in the task's panel. All 57 panel genes were found.

## 3. Source type names (exact strings)

Cell-level columns of tables 4 / 7a / 7b (77-79 names): C2, C2.C3, C3, ChAT, Crz, Dm1, Dm10, Dm11, Dm12, Dm3, Dm4, Dm8,
Dm9, Gad1, Glia_Eg, Glia_Mg, Glia_Psg, KC_ab_c, KC_ab_c.p.s, KC_ab_p, KC_ab_s, KC_gd, Kdm2, L1, L1.L2, L2, L3, L4, L5,
Lai, lamina (table 4 only), Lawf1, Lawf2, LC10a, LC10b, LC16, LC4, LC6, LLPC1, LPC1, LPi-34, LPLC1, LPLC2, Mi1, Mi15,
Mi4, Mi9, Muscles_App, Muscles_Head, NPF, opticlobe (table 4 only), PAM_1, PAM_3, PAM_4, PB_1, PB_2, PB_3, PB_4, PB_5,
Pm3, Pm4, R1-6, R7, R8_Rh5, R8_Rh6, T1, T4, T4.T5, T5, Tm1, Tm2, Tm20, Tm29, Tm3, Tm4, Tm9, TmY3, TmY5a, VGlut.

Drivers whose every sample Davis marked `suboptimal` (absent from tables 4-7; only in tables 1-3, 21 names): CCAP,
Dsk, Ilp2, KC_apbp_ap, KC_apbp_m, Lat, LC10bc, LC10d, lLNv, LPTC_HS.VS, MBON_bp1, MBON_g1pedc, PAM_2, PAM_5, PAM_6,
PAM_7, PAM_8, PAM_9, PB_6, Pdf, R7_Rh3. These are carried in our tables with `qc = suboptimal_only` (TPM = mean of
their table-3 samples, no P(on)); use with caution, and never as the sole evidence for a sign.

The central-complex drivers use Wolff et al. 2015 long names in supp. 1A: PB_1 = "PB18.s_Gx_7Gy.b__PB18.s-9i1i8c.b",
PB_2 = "PBG1-8.b-EBw.s-D_Vgall.b", PB_3 = "PBG2-9.s-EBt.b-NO1.b", PB_4 = "PBG2-9.s-FBl1.b-NO3P.b",
PB_5 = "PBG2-9.s-FBl2.b-NO3A.b", PB_6 = "PBG2-9.s-FBl3.b-NO2D.b". The PAM drivers are Aso 2014 compartments (PAM_1 =
PAM-b'2a; PAM_3 = PAM-b1, PAM-b2; PAM_4 = PAM-g4, PAM-g4<g1g2; ...).

**Tm5Y is not in Davis 2020.** No driver for Tm5Y (or Tm5a/b/c, TmY14, Mi19) was profiled, so the "Davis et al. 2020
glutamate prediction for Tm5Y" quoted in `docs/NT_INTEGRATION.md` section 2 cannot be sourced from this dataset's
tables. The nearest profiled type is Tm29 (first named in this paper, "similar to Tm5b"), glutamatergic in both Davis
(VGlut 1,130 TPM) and MaleCNS.

## 4. Derived tables

### 4a. `flyverse/data/type_map_davis2020.csv` (150 rows, 100 source names)

Columns: `source_name, malecns_type, tier, evidence, n_cells_malecns, qc, malecns_superclass, malecns_nt,
out_syn_malecns, davis_nt_call, davis_nt_secondary`. One row per (source, MaleCNS type); a pooled source has several
rows. Tiers:

* **exact** (42 sources -> 42 types): identical string in supp. 1A and MaleCNS `type`; the MaleCNS `flywireType`
  column is recorded as a check. Pm3 / Pm4 and Tm29 are exact by name but note that FlyWire renumbered Pm types
  (MaleCNS Pm3 = FlyWire Pm09, Pm4 = Pm05) -- the Davis driver was identified from Nern et al. 2015 and Nern et al.
  2025 kept those names, so the join holds on the MaleCNS side, not on FlyWire's.
* **alias** (20 sources -> 20 types): documented one-to-one rename. LPi-34 -> LPi34; R1-6 -> R1-R6; R8_Rh5 -> R8p;
  R8_Rh6 -> R8y; KC_ab_c/p/s -> KCab-c/p/s; KC_gd -> KCg-d; KC_apbp_m -> KCa'b'-m; MBON_bp1 -> MBON10 (MaleCNS
  instance `MBON10(B'1)`); MBON_g1pedc -> MBON11 (`MBON11(y1pedc>a/B)`); PAM_1 -> PAM02, PAM_5 -> PAM01, PAM_7 ->
  PAM11, PAM_8 -> PAM12 (MaleCNS instance names carry the Aso compartment, e.g. `PAM02(B'2a)`); lLNv -> l-LNv;
  PB_1 -> Delta7, PB_2 -> EPG, PB_5 -> PFNa, PB_6 -> PFNd (Wolff & Rubin 2018, J Comp Neurol 526:2585, Table 5:
  PB18.s-GxD7Gy.b = Delta7, PBG1-8.b-EBw.s-D/V GA.b = E-PG, PBG2-9.s-FBl2.b-NO3A.b = P-FNA, PBG2-9.s-FBl3.b-NO2D.b =
  P-FND; Table 2 lists SS00090 and SS00078 for those types).
* **fuzzy** (21 sources -> 42 types): one Davis driver pools several MaleCNS subtypes that were split later, or the
  driver is a combination. Rule: the source name (or the driver's `celltype.details`) is the prefix / the pooled set of
  the MaleCNS subtype names, and supp. 1A says the driver covers the whole population. T4 -> T4a-d; T5 -> T5a-d;
  T4.T5 -> all eight; Dm3 -> Dm3a/b/c; Dm8 -> Dm8a/b; R7 -> R7p/R7y/R7d (Rh3 + Rh4 driver; R7_unclear excluded);
  R7_Rh3 -> R7p/R7d; C2.C3, L1.L2, KC_ab_c.p.s (combo drivers); LC10bc -> LC10b, LC10c-1, LC10c-2; Lat -> Lat1-5;
  LPTC_HS.VS -> HSN, HSE, HSS, VS; KC_apbp_ap -> KCa'b'-ap1/ap2; PAM_3 -> PAM04 + PAM09 + PAM10; PAM_4 -> PAM07 +
  PAM08; PAM_2, PAM_6, PAM_9 (multi-compartment drivers); PB_3 -> PEN_a + PEN_b (Wolff & Rubin 2018 Table 5:
  PBG2-9.s-EBt.b-NO1.b = P-EN1, P-EN2; **supp. 1A warns this driver labels many optic-lobe cells incl. Mi1 and the GF,
  outnumbering the PB cells -- treat PB_3 as contaminated**); PB_4 -> PFNp_a-e (Table 5: P-FNP).
* **class** (5 sources): ChAT / Gad1 / VGlut protein-trap drivers -> every MaleCNS cell with that predicted NT
  (104,040 / 22,135 / 29,616 cells; useful as whole-class receptor baselines, not as type profiles); opticlobe ->
  all optic superclasses; Pdf -> l-LNv + s-LNv.
* **unmatched** (12 sources): Kdm2, NPF, Crz, CCAP, Dsk, Ilp2 (no MaleCNS type label for a neuropeptide / enhancer-trap
  population), lamina (tissue), Glia_Eg / Glia_Mg / Glia_Psg (MaleCNS has no glia), Muscles_App / Muscles_Head.

### 4b. `flyverse/data/expression_davis2020.csv` (100 rows x 114 columns)

One row per source name; `<gene>_tpm` (57 genes, cell-type mean TPM from table 4, or table-3 sample means for
`suboptimal_only`), `<gene>_p_on` (table 7b P(expressed); NaN for suboptimal-only rows), `n_samples_pass`,
`n_samples_suboptimal`, and the transmitter summary `davis_nt_call` (class with the largest pair-minimum TPM over
ChAT+VAChT / Gad1+VGAT / VGlut / Hdc / ple+DAT / Tdc2+Tbh / Trh+SerT, if >= 10 TPM), `davis_nt_secondary` (other
classes >= 50 TPM), `davis_nt_scores_top3`. Unit is TPM as published; nothing is re-normalised.

## 5. Coverage of MaleCNS (cells and |W| synapses; sign-0 presynaptic cells count 0 output synapses)

Optic group = superclass ol_intrinsic + visual_projection + ol_sensory: 104,689 cells, 629 types, 51.92 M output
synapses (42.8 % of the CNS's 121.4 M signed synapses).

| tier | optic types | optic cells | optic cells % | optic out-syn % | optic in-syn % |
|---|---|---|---|---|---|
| exact | 42 | 38,608 | 36.9 | 38.2 | 27.1 |
| alias | 5 | 4,315 | 4.1 | 0.9 | 0.6 |
| fuzzy | 24 | 18,997 | 18.1 | 9.5 | 9.2 |
| class (s-LNv via Pdf) | 1 | 8 | 0.0 | 0.0 | 0.0 |
| **matched, any** | **72** | **61,928** | **59.2** | **48.7** | **36.9** |
| unmatched | 557 | 42,761 | 40.8 | 51.3 | 63.1 |

QC-pass sources only (drop the 21 suboptimal-only drivers): 60 optic types, 61,077 cells (58.3 %), 47.9 % of optic
output synapses, 35.5 % of input synapses. Optic-internal synapses (pre and post both in the optic group,
47.11 M) with **both ends matched: 11.04 M = 23.4 %** (22.5 % QC-pass only). The big unmatched optic types are the
ones the object pathway needs: Tm5Y, Tm5a/b/c, TmY14, T2/T3, Mi2/Mi13, Dm2/Dm15/Dm16, most Pm, LPi43, all LC types
other than LC4/6/10a/10b/10d/16, all Li / Lo / MeTu / LT / DN-projecting types.

visual_centrifugal (563 cells): 14 cells (Lat1/2/5 via the suboptimal Lat driver) matched, 0 % of output synapses.

Central brain (cb_intrinsic, 32,160 cells, 6,606 types, 37.47 M out-syn): alias 15 types / 1,980 cells / 3.0 % of
output synapses; fuzzy 15 types / 971 cells / 1.3 %; unmatched 90.8 % of cells, 95.7 % of output synapses. QC-pass
only: 19 types, 2,046 cells (6.4 %), 2.9 % of output synapses (KCab-c/p/s, KCg-d, PAM02/04/07/08/09/10, Delta7, EPG,
PEN_a/b, PFNa, PFNp_a-e).

Whole CNS (167,106 cells): exact 23.1 % of cells / 16.3 % of output synapses; alias 3.8 % / 1.3 %; fuzzy 12.0 % /
4.5 %; matched total 38.9 % of cells, 22.1 % of output synapses (QC-pass only: 37.8 % / 21.4 %); synapses with both
ends matched 11.85 M = 9.8 % (9.2 % QC-pass).

## 6. Transmitter cross-check (Davis synthesis genes vs MaleCNS majority label)

120 of 134 matched (source, type) pairs agree on the primary class. Disagreements: T1 (MaleCNS histamine; Davis:
no synthesis gene above 5 TPM -- Hdc 0.6, so T1's MaleCNS label is unsupported by its transcriptome), L2 via the
L1.L2 pool (pool is glutamate-dominated by L1; the pure L2 row agrees: ACh), and the suboptimal-only rows Lat1-5
(MaleCNS unknown / Lat3 octopamine; Davis mixed glutamate 193 / GABA 121 / histamine 76 -- a five-type pool),
LPTC HS/VS (MaleCNS ACh; Davis VGlut 140, VAChT 893 but ChAT only 65 -- inconclusive), l-LNv (MaleCNS serotonin;
Davis Trh / SerT 0 TPM, GABA / glutamate / ACh markers ~90 each -- the serotonin label looks wrong, but the sample is
QC-fail) and the Pdf pool. Secondary-class flags worth keeping: Mi15 dopamine (ple 683, DAT 195 TPM next to ChAT
235) -- the paper's own Mi15 co-transmission finding; Mi9 Gad1 276 / VGAT 251 beside VGlut 1,773; R8p / R8y ChAT
135-190 and VAChT 345-467 beside Hdc ~600 (cholinergic markers in R8, also reported by Davis); PFNp glutamate 66; KCab-p
Gad1 101. None of these is a sign change on its own: they are the inputs to step 3 of the plan (two-of-three rule with
the MaleCNS consensus and the Nern 2025 prediction).

Receptor side, examples from the table (TPM): GluClalpha is high in every optic type profiled (T4 2,193, T5 2,782,
Mi1 1,235, Tm3 1,445, LC4 3,237, LPi34 3,235, C2 1,630) while KaiR1D is 0.6-190 (Dm10 191, Mi15 156, Tm4 146, L1 121,
T4 114, Mi4 115, T5 51, Mi1 1.6, LC4 1.1): on this panel glutamate is GluCl-dominated (inhibitory) in almost every
matched optic target, with the iGluR share highest in Dm10 / Tm4 / L1 / T4 / Mi4 / Mi15. Rdl is 12-24 TPM in L1 / L2
(GABA-A nearly absent from L1 / L2) versus 1,000-19,500 elsewhere (LC4 19,516). ort (HisCl) is where expected: L1
647, L2 412, L3 514, Lai 212, Dm9 425, Tm20 329, Dm11 210, Dm8 203, T1 217; elsewhere < 80.

## 7. Open points

* Nern et al. 2025 / MaleCNS renamed or split several Nern-2015 types; the exact-tier rows rely on the annotator
  continuity (same author on both papers) and on the `flywireType` column where FlyWire kept the name. A one-line
  check per exact row against the Nern 2025 supplementary synonym table (typing-source task) would upgrade the
  evidence.
* PB_3 (P-EN) is contaminated per Davis's own note; PB_1 (Delta7, VGlut 4,113 TPM) and PB_2 (EPG) are clean.
* Davis's samples are mixed-sex heads; MaleCNS is male. Not addressed here.
* The `class` rows (ChAT / Gad1 / VGlut) are the only central-brain-wide receptor baselines in this source; the
  per-type central coverage is 3-4 % of cb_intrinsic output synapses and confined to MB / CX.
