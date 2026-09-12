# Receptor / transmitter expression source: Davis et al. 2020 (key `davis2020`)

Part of the NT-integration workflow (`docs/NT_INTEGRATION.md`, step 1 "acquire and pin", step 2 "type map").
Round 1 generated 2026-09-11 from a session scratchpad; **round 2 (2026-09-11) rebuilt both tables with
`scripts/build_davis2020_tables.py`** (repo-relative paths; `PYTHONIOENCODING=utf-8 python
scripts/build_davis2020_tables.py --report <path>.json` reproduces every number below, CPU only, ~1 min). Raw files live in
`data/external/davis2020/` (git-ignored; `python scripts/fetch_data.py --external davis2020`); derived tables in
`flyverse/data/`. The round-1 errors and their corrections are listed in section 8 (skeptics' record:
`docs/audits/receptor_verification.md`, verify:tables:davis2020 and verify:derive).

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
needed; the processed tables are the paper's own quantification). All 12 files present, 39,371,853 bytes; every SHA-256
re-verified by the round-1 skeptic (tables 4 and 7b re-downloaded and hashed identically).

| file | bytes | SHA-256 | content |
|---|---|---|---|
| GSE116969_README_dataTables.txt | 3,322 | fbd0e7cbf91f42ad6ef44c46c2f98f73a09bb3049804159d56ac8d9812ceb1f9 | README for tables 1-7 |
| GSE116969_dataTable1.transcripts_x_samples_TPM.all_genes.txt.gz | 12,442,683 | b96dd43df9a92ad8a1bb2d12aeb5f4ae34e642d707e2d9ed1ccab20627f7b4a2 | transcript x sample TPM, all samples |
| GSE116969_dataTable2.genes_x_samples_TPM.all_genes.txt.gz | 7,203,098 | 115497e3d60fe1a051e910e84ea29814543c6348b0a32b85a3ada507aed04686 | gene x sample TPM, all genes (coding, noncoding, reporter, ERCC) |
| GSE116969_dataTable3.genes_x_samples_TPM.coding_genes.txt.gz | 6,513,528 | 5c035c18b531995e2fe1b56109c1c8550628b678d305b6942b9e073bf37a3181 | gene x sample TPM, coding genes, **all 266 samples incl. QC-fail** (columns gene_id, gene_name, samples) -- the source for suboptimal-only drivers and for the gene_id check |
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

**Gene-symbol notes (corrected in round 2).** The tables use FlyBase 2017_04 symbols. The panel has **54 genes**
(12 synthesis / transport, 10 nAChR, Rdl / Lcch3 / Grd, GluClalpha, KaiR1D / GluRIA / GluRIB / Nmdar1 / Nmdar2,
HisCl1 / ort, 3 mAChR, 3 GABA-B, mGluR, 4 Dop, 5 Oct, 5 5-HT); all 54 are present. Two panel names differ from the
2017 symbols: **`KaiR1D` = `CG3822` (FBgn0038837)** and `Octalpha2R` = `CG18208` (FBgn0038653); the builder asserts both
gene_ids against dataTable3. Round 1 wrongly used `CG8916` (FBgn0030707, an uncharacterised Cys-loop channel, alias
Lcch-14A) for KaiR1D, so the round-1 `KaiR1D_tpm` / `KaiR1D_p_on` columns carried a different gene (section 8). The
gene `nan` (nanchung) in table 4 is read as NaN by pandas; harmless for the panel.

## 3. Source type names (exact strings)

Cell-level columns of tables 4 / 7a / 7b (77-79 names): C2, C2.C3, C3, ChAT, Crz, Dm1, Dm10, Dm11, Dm12, Dm3, Dm4, Dm8,
Dm9, Gad1, Glia_Eg, Glia_Mg, Glia_Psg, KC_ab_c, KC_ab_c.p.s, KC_ab_p, KC_ab_s, KC_gd, Kdm2, L1, L1.L2, L2, L3, L4, L5,
Lai, lamina (table 4 only), Lawf1, Lawf2, LC10a, LC10b, LC16, LC4, LC6, LLPC1, LPC1, LPi-34, LPLC1, LPLC2, Mi1, Mi15,
Mi4, Mi9, Muscles_App, Muscles_Head, NPF, opticlobe (table 4 only), PAM_1, PAM_3, PAM_4, PB_1, PB_2, PB_3, PB_4, PB_5,
Pm3, Pm4, R1-6, R7, R8_Rh5, R8_Rh6, T1, T4, T4.T5, T5, Tm1, Tm2, Tm20, Tm29, Tm3, Tm4, Tm9, TmY3, TmY5a, VGlut.

Drivers whose every sample Davis marked `suboptimal` (absent from tables 4-7; only in tables 1-3, 21 names): CCAP,
Dsk, Ilp2, KC_apbp_ap, KC_apbp_m, Lat, LC10bc, LC10d, lLNv, LPTC_HS.VS, MBON_bp1, MBON_g1pedc, PAM_2, PAM_5, PAM_6,
PAM_7, PAM_8, PAM_9, PB_6, Pdf, R7_Rh3. These are carried in our tables with `qc = suboptimal_only` (TPM = mean of
their table-3 samples, no P(on)); use with caution, and never as the sole evidence for a sign. Two drivers (Dm12_d1,
NPF_d1) mix pass and suboptimal samples; their cell rows use the QC-pass samples only (table 4).

The central-complex drivers use Wolff et al. 2015 long names in supp. 1A: PB_1 = "PB18.s_Gx_7Gy.b__PB18.s-9i1i8c.b",
PB_2 = "PBG1-8.b-EBw.s-D_Vgall.b", PB_3 = "PBG2-9.s-EBt.b-NO1.b", PB_4 = "PBG2-9.s-FBl1.b-NO3P.b",
PB_5 = "PBG2-9.s-FBl2.b-NO3A.b", PB_6 = "PBG2-9.s-FBl3.b-NO2D.b". The PAM drivers are Aso 2014 compartments (PAM_1 =
PAM-b'2a; PAM_3 = PAM-b1, PAM-b2; PAM_4 = PAM-g4, PAM-g4<g1g2; ...).

**Tm5Y is not in Davis 2020.** No driver for Tm5Y (or Tm5a/b/c, TmY14, Mi19, T2, T3, LPi43, any LC other than
LC4/6/10a/10b/10d/16) was profiled, so the "Davis et al. 2020 glutamate prediction for Tm5Y" quoted in
`docs/NT_INTEGRATION.md` section 2 cannot be sourced from this dataset's tables. The nearest profiled type is Tm29
(first named in this paper, "similar to Tm5b"), glutamatergic in both Davis (VGlut 1,130 TPM) and MaleCNS.

## 4. Derived tables

### 4a. `flyverse/data/type_map_davis2020.csv` (151 rows, 100 source names)

Columns: `source_name, malecns_type, tier, evidence, n_cells_malecns, qc, malecns_superclass, malecns_nt,
out_syn_malecns, davis_nt_call, davis_nt_secondary`. One row per (source, MaleCNS type); a pooled source has several
rows. Sources per tier: exact 42 / alias 20 / fuzzy 21 / class 9 / unmatched 8. MaleCNS types per best tier: exact 42 /
alias 20 / fuzzy 42 / class 6 (s-LNv, IPC, CRZ01, CRZ02, DSKMP3, NPFL1-I). Tiers:

* **exact** (42 sources -> 42 types): identical string in supp. 1A and MaleCNS `type`; the MaleCNS `flywireType`
  column is recorded as a check. Pm3 / Pm4 and Tm29 are exact by name but note that FlyWire renumbered Pm types
  (MaleCNS Pm3 = FlyWire Pm09, Pm4 = Pm05) -- the Davis driver was identified from Nern et al. 2015 and Nern et al.
  2025 kept those names, so the join holds on the MaleCNS side, not on FlyWire's. Tm29 rests on the name only
  (FlyWire calls the same cells Tm5d, no hemibrain name); the central-source skeptic (`receptor_verification.md`,
  verify:tables:central) records that FCA "transmedullary neuron Tm29" is cholinergic (VAChT fraction 0.56, VGlut 0.09)
  while MaleCNS Tm29 is glutamate -- Davis Tm29 VGlut 1,130 TPM agrees with MaleCNS, not with FCA.
* **alias** (20 sources -> 20 types): documented one-to-one rename. LPi-34 -> LPi34; R1-6 -> R1-R6; R8_Rh5 -> R8p;
  R8_Rh6 -> R8y; KC_ab_c/p/s -> KCab-c/p/s; KC_gd -> KCg-d; KC_apbp_m -> KCa'b'-m; MBON_bp1 -> MBON10 (MaleCNS
  instance `MBON10(B'1)`); MBON_g1pedc -> MBON11 (`MBON11(y1pedc>a/B)`); PAM_1 -> PAM02, PAM_5 -> PAM01, PAM_7 ->
  PAM11, PAM_8 -> PAM12 (MaleCNS instance names carry the Aso compartment, e.g. `PAM02(B'2a)`); lLNv -> l-LNv (supp. 1A:
  SS00645 = R61G12-AD x R10H10-DBD, "Cell type description: Helfrich-Forster et al 2007", driver reference "a gift from
  Heather Dionne, Rubin lab" -- round 1 cited the description as the driver's reference); PB_1 -> Delta7, PB_2 -> EPG,
  PB_5 -> PFNa, PB_6 -> PFNd (Wolff & Rubin 2018, J Comp Neurol 526:2585, Table 5: PB18.s-GxD7Gy.b = Delta7,
  PBG1-8.b-EBw.s-D/V GA.b = E-PG, PBG2-9.s-FBl2.b-NO3A.b = P-FNA, PBG2-9.s-FBl3.b-NO2D.b = P-FND; Table 2 lists SS00090
  and SS00078 for those types).
* **fuzzy** (21 sources -> 42 types): one Davis driver pools several MaleCNS subtypes that were split later, or the
  driver is a combination. Rule: the source name (or the driver's `celltype.details`) is the prefix / the pooled set of
  the MaleCNS subtype names, and supp. 1A says the driver covers the whole population. **T4 -> T4a-d and T5 -> T5a-d are
  flagged SUBTYPE-BIASED in the evidence column**: both T4 drivers (SS02344, SS23866) are recorded as "T4: strongest in
  T4b,T4c", and the T5 cell mean pools SS25175 ("strongest in T5c and T5d", "some weakly labeled T4 cells") with
  SS23757 ("mainly T5a and T5b", same T4 note) -- a per-subtype consumer must treat these profiles as a biased pool, not
  as four independent profiles. T4.T5 -> all eight; Dm3 -> Dm3a/b/c; Dm8 -> Dm8a/b; R7 -> R7p/R7y/R7d (Rh3 + Rh4
  driver; R7_unclear excluded); R7_Rh3 -> R7p/R7d; C2.C3, L1.L2, KC_ab_c.p.s (combo drivers); LC10bc -> LC10b, LC10c-1,
  LC10c-2; Lat -> Lat1-5 (supp. 1A: SS00657 = R23E12-AD x R55B04-DBD, "Cell type description: Tuthill et al 2013,
  Fischbach and Dittrich 1989", driver reference "Dionne et al in preparation" -- round 1 wrote "Tuthill 2013 lamina
  tangential" as if it were the driver reference); LPTC_HS.VS -> HSN, HSE, HSS, VS; KC_apbp_ap -> KCa'b'-ap1/ap2; PAM_3
  -> PAM04 + PAM09 + PAM10; PAM_4 -> PAM07 + PAM08; PAM_2, PAM_6, PAM_9 (multi-compartment drivers); PB_3 -> PEN_a +
  PEN_b (Wolff & Rubin 2018 Table 5: PBG2-9.s-EBt.b-NO1.b = P-EN1, P-EN2; **supp. 1A warns this driver labels many
  optic-lobe cells incl. Mi1 and the GF, outnumbering the PB cells -- treat PB_3 as contaminated**); PB_4 -> PFNp_a-e
  (Table 5: P-FNP).
* **class** (9 sources): ChAT / Gad1 / VGlut protein-trap drivers -> every MaleCNS cell with that predicted NT
  (104,040 / 22,135 / 29,616 cells; 73.6 M / 25.9 M / 21.2 M output synapses; useful as whole-class receptor baselines,
  not as type profiles); opticlobe -> all optic superclasses (105,252 cells); Pdf -> l-LNv + s-LNv (8 + 8 cells); and,
  new in round 2, the neuropeptide promoter fusions matched to the MaleCNS peptide types they define: **Ilp2 -> IPC**
  (BL-37516, Rulifson et al 2002; MaleCNS IPC 16 cells, superclass cb_endocrine, hemibrainType "PI1,PI2,PI3", nt
  unknown; suboptimal-only), **Crz -> CRZ01 + CRZ02** (BL-51977, Tayler et al 2012; 2 + 2 cells, nt serotonin; QC-pass;
  the head driver may label more Crz cells than these four), **Dsk -> DSKMP3** (BL-51981, Asahina & Anderson 2013; 4
  cells, the only Dsk-named MaleCNS type, no DSKMP1; suboptimal-only; partial), **NPF -> NPFL1-I** (BL-25681, Wu et al
  2003; 2 cells = the large lateral NPF neuron; the driver labels ~20 brain NPF cells, so partial; QC-pass). Round 1's
  note that NPFL1-I is "a different, lateral-horn type" is withdrawn. All five peptide types have sign 0 in the model
  (nt serotonin / unknown), so `out_syn_malecns` = 0 for them by the |W| convention and they add nothing to synapse
  coverage; the rows exist so that the receptor table can carry a Davis profile for these populations.
* **unmatched** (8 sources): Kdm2 (enhancer trap, no cell-type identity) and CCAP (no CCAP-named type in MaleCNS
  v1.0) -- the only two peptide / enhancer-trap drivers without a MaleCNS counterpart; lamina (tissue); Glia_Eg /
  Glia_Mg / Glia_Psg (MaleCNS has no glia); Muscles_App / Muscles_Head.

### 4b. `flyverse/data/expression_davis2020.csv` (100 rows x 115 columns)

One row per source name; 4 meta columns (`source_name, qc, n_samples_pass, n_samples_suboptimal`), `<gene>_tpm` (54
genes, cell-type mean TPM from table 4, or table-3 sample means for `suboptimal_only`), `<gene>_p_on` (54 columns, table
7b P(expressed); NaN for suboptimal-only rows and for the lamina / opticlobe tissue rows, which table 7b lacks), and the
transmitter summary `davis_nt_call` (class with the largest pair-minimum TPM over ChAT+VAChT / Gad1+VGAT / VGlut / Hdc /
ple+DAT / Tdc2+Tbh / Trh+SerT, if >= 10 TPM), `davis_nt_secondary` (other classes >= 50 TPM), `davis_nt_scores_top3`.
Unit is TPM as published; nothing is re-normalised. The round-1 skeptic confirmed every QC-pass value equals table 4
and every P(on) equals table 7b (max abs diff 0.0); round 2 changes only the two KaiR1D columns and the header.

## 5. Coverage of MaleCNS (cells and |W| synapses; sign-0 presynaptic cells count 0 output synapses)

Type counts are named types only (the untyped pseudo-type is excluded: 37 untyped cells in the optic group, 880 in
cb_intrinsic, 2,605 CNS-wide). Optic group = superclass ol_intrinsic + visual_projection + ol_sensory: 104,689 cells,
628 named types, 51.92 M output synapses (42.8 % of the CNS's 121.4 M signed synapses), 49.41 M input synapses.

| tier | optic types | optic cells | optic cells % | optic out-syn % | optic in-syn % |
|---|---|---|---|---|---|
| exact | 42 | 38,608 | 36.9 | 38.2 | 27.1 |
| alias | 5 | 4,315 | 4.1 | 0.9 | 0.6 |
| fuzzy | 24 | 18,997 | 18.1 | 9.5 | 9.2 |
| class (s-LNv via Pdf) | 1 | 8 | 0.0 | 0.0 | 0.0 |
| **matched, any** | **72** | **61,928** | **59.2** | **48.7** | **36.9** |
| unmatched | 556 | 42,761 | 40.8 | 51.3 | 63.1 |

QC-pass sources only (drop the 21 suboptimal-only drivers; a type reached by both a QC-pass and a suboptimal-only
driver -- R7p, R7d, LC10b, PAM02, PAM07, PAM08, PAM09, PAM10 -- keeps its QC-pass driver, which round 1 got wrong for
R7p / PAM08): **61 optic types, 61,409 cells (58.7 %), 48.0 % of optic output synapses, 35.6 % of input synapses**
(exact 41 / alias 4 / fuzzy 16). Optic-internal synapses (pre and post both in the optic group, 47.11 M) with **both
ends matched: 11.04 M = 23.4 %** (10.66 M = 22.6 % QC-pass only). The big unmatched optic types are the ones the object
pathway needs: Tm5Y, Tm5a/b/c, TmY14, T2/T3, Mi2/Mi13, Dm2/Dm15/Dm16, most Pm, LPi43, all LC types other than
LC4/6/10a/10b/10d/16, all Li / Lo / MeTu / LT / DN-projecting types.

visual_centrifugal (563 cells, 108 types): 14 cells (Lat1/2/5 via the suboptimal Lat driver) matched, 0 % of output
synapses, 0.6 % of input synapses.

Central brain (cb_intrinsic, 32,160 cells, 6,605 named types, 37.47 M out-syn): alias 15 types / 1,980 cells / 3.0 % of
output synapses; fuzzy 15 types / 971 cells / 1.3 %; class 4 types / 10 cells (CRZ01, CRZ02, DSKMP3, NPFL1-I; 0 out-syn,
sign 0); matched 34 types / 2,961 cells (9.2 %) / 4.3 % out / 4.0 % in; unmatched 90.8 % of cells, 95.7 % of output
synapses. QC-pass only: exact / alias / fuzzy 20 types, 2,096 cells (6.5 %), 2.9 % of output synapses (KCab-c/p/s,
KCg-d, PAM02/04/07/08/09/10, Delta7, EPG, PEN_a/b, PFNa, PFNp_a-e) plus the 3 QC-pass class types (6 cells) = 23 types /
2,102 cells. cb_endocrine (72 cells): IPC 16 cells (22 %) via Ilp2, 0 output synapses (sign 0), 42 % of the
superclass's input synapses.

Whole CNS (167,106 cells): exact 23.1 % of cells / 16.3 % of output synapses; alias 3.8 % / 1.3 %; fuzzy 12.0 % /
4.5 %; class 34 cells / 0.0 %; **matched total 64,919 cells = 38.8 %, 22.1 % of output synapses, 16.4 % of input
synapses** (QC-pass only: 63,511 cells = 38.0 % / 21.4 % / 15.4 %); synapses with both ends matched 11.85 M = 9.8 %
(11.19 M = 9.2 % QC-pass).

## 6. Transmitter cross-check (Davis synthesis genes vs MaleCNS majority label)

120 of 139 matched (source, type) pairs agree on the primary class (120 of the 134 round-1 pairs; the 5 new peptide rows
all disagree, as expected: MaleCNS labels IPC unknown and CRZ01/02, DSKMP3, NPFL1-I serotonin, while the Davis marker
pairs give low-level GABA / ACh calls of 46-204 TPM -- these are peptidergic populations and the marker-pair call is not
meaningful for them. Serotonin markers: Crz Trh 8.6 / SerT 0.6 and NPF Trh 5.6 / SerT 7.3 TPM (QC-pass; nothing
serotonergic); Ilp2 Trh 17.8 / SerT 117.3 and Dsk Trh 15.8 / SerT 106.7 TPM (suboptimal-only sample means; SerT without
Trh, pair-minimum 16-18 so below the 50 TPM secondary flag)). Disagreements among the round-1 pairs: **T1 (MaleCNS
histamine; Davis: no synthesis enzyme > 5 TPM -- Hdc 0.61, ChAT 3.04, Gad1 1.59, ple 1.34, Trh 4.51 TPM; Tdc2 5.49 and
Tbh 5.75 sit just above it; VGAT is 172.6 TPM but that is non-diagnostic -- L1 304.6, L2 216.7, Mi1 236.1 -- so T1's
MaleCNS label is unsupported by its transcriptome)**, L2 via the L1.L2 pool (pool is glutamate-dominated by L1; the pure
L2 row agrees: ACh), and the suboptimal-only rows Lat1-5 (MaleCNS unknown / Lat3 octopamine; Davis mixed glutamate 193
/ GABA 121 / histamine 76 -- a five-type pool), LPTC HS/VS (MaleCNS ACh; Davis VGlut 140, VAChT 893 but ChAT only 65 --
inconclusive), l-LNv (MaleCNS serotonin; Davis Trh 0.0 and SerT 5.4 TPM -- below the 10 TPM floor, not zero -- with
GABA / glutamate / ACh markers ~90 each: the serotonin label looks wrong, but the sample is QC-fail) and the Pdf pool.
Secondary-class flags worth keeping: Mi15 dopamine (ple 683, DAT 195 TPM next to ChAT 235) -- the paper's own Mi15
co-transmission finding; Mi9 Gad1 276 / VGAT 251 beside VGlut 1,773; R8p / R8y ChAT 135-190 and VAChT 345-467 beside
Hdc ~600 (cholinergic markers in R8, also reported by Davis); PFNp glutamate 66; KCab-p Gad1 101. None of these is a
sign change on its own: they are the inputs to step 3 of the plan (two-of-three rule with the MaleCNS consensus and the
Nern 2025 prediction).

Receptor side, examples from the rebuilt table (TPM, QC-pass cell means): GluClalpha is high in every optic type
profiled (T4 2,193, T5 2,782, Mi1 1,235, Tm3 1,445, LC4 3,237, LPi34 3,235, C2 1,630; lowest T1 13.6, Tm9 160, Dm9 184,
L1 52). **KaiR1D (CG3822) is broad and flat**: over the 59 optic-matched sources (exact / alias / fuzzy rows whose
MaleCNS type is in the optic group) it is 0.8-2,507 TPM with Lai 2,507 the
sole outlier, then Tm2 115, LPLC2 112, LLPC1 108, Tm9 107, Lawf2 99, T1 94, TmY5a 90, Tm4 90, Dm11 88, TmY3 87, T5 85;
Mi4 76, Dm10 75, Mi15 71, LC4 66, Mi1 46, L1 46; photoreceptors 0.8-8.5 with P(on) 0. It is "on" (P(on) >= 0.5 and TPM
>= 10) in 89 of the 100 rows (round 1's CG8916 column: 26). The round-1 statements "KaiR1D 0.6-190, highest Dm10 191 /
Mi15 156 / Tm4 146 / L1 121 / T4 114 / Mi4 115, Mi1 1.6, LC4 1.1" and "iGluR share highest in Dm10 / Tm4 / L1 / T4 / Mi4
/ Mi15" described CG8916 and are withdrawn. With the real gene the iGluR share (KaiR1D + GluRIA + GluRIB + Nmdar1 +
Nmdar2 over that sum plus GluClalpha) is highest in T1 0.95, Lai 0.90, Dm9 0.86, L1 0.78, Tm9 0.71, L4 0.68, L3 0.65,
Dm10 0.55, Dm11 0.49, Mi4 0.48, and glutamate remains GluCl-dominated (inhibitory) in T4 / T5 / LC4 / LPi34 / Mi1 / Tm3
and most other matched optic targets. Rdl is 12-24 TPM in L1 / L2 (GABA-A nearly absent from L1 / L2) versus
1,000-19,500 elsewhere (LC4 19,516). ort (HisCl) is where expected: L1 647, L2 412, L3 514, Lai 212, Dm9 425, Tm20
329, Dm11 210, Dm8 203, T1 217; elsewhere < 80.

### 6b. Gene ranges quoted downstream (with denominators)

The derive step (`receptor_rules.md`, `receptors_by_type.csv`) quoted ranges for four genes with the wrong denominator or
range in round 1. The correct statements, from `scripts/build_davis2020_tables.py --report` (`gene_ranges`), over
**neuronal QC-pass rows = the 79 table-4 columns minus Glia_Eg / Glia_Mg / Glia_Psg / Muscles_App / Muscles_Head /
lamina / opticlobe (72 rows) minus the ChAT / Gad1 / VGlut class drivers = 69 rows**; "on" = P(on) >= 0.5 AND TPM >= 10,
the rule `build_receptor_table.py` applies to this source:

| gene | TPM range, 69 neuronal rows (min .. max) | on / 69 | on / 72 | on / 79 (glia + muscle incl.) | notes |
|---|---|---|---|---|---|
| Nmdar2 | 5.32 (R8_Rh5) .. 1,073 (TmY3) | 65 (94 %) | 68 | 69 (87 %) | off only in the 4 photoreceptor rows (R1-6 35.4, R7 30.6, R8_Rh5 5.3, R8_Rh6 8.0 TPM, P(on) = 0); excluding them 102 (L2) .. 1,073 with P(on) = 1 in 65 / 65 |
| VGAT | 1.74 (R8_Rh5) .. 586 (Pm4) | 65 (94 %) | 68 | 69 (87 %) | photoreceptors 1.7-14.5, P(on) = 0; excluding them 60.0 (PAM_4) .. 586 with P(on) = 1 in 65 / 65 -- VGAT does not discriminate among non-photoreceptor neurons (T1 172.6, L1 304.6, Mi1 236.1) |
| Tdc2 | 0.64 (R8_Rh5) .. 35.1 (Kdm2) | 22 (32 %) | 24 | 25 (32 %) | P(on) mean 0.88, >= 0.5 in 63 / 69 (91 %), = 1 in 27 / 69 (39 %), but the TPM >= 10 half of the rule fails in most rows; top: Kdm2 35, LPi-34 25, LC10b 21, LPLC2 19, NPF 17, TmY3 17 |
| DopEcR | 97.8 (L3) .. 12,716 (Dm3) | 64 (93 %) | 67 | 67 (85 %) | off rows (69): L3 (97.8 TPM, P(on) 0), Crz, PAM_3, PAM_4, PB_1; photoreceptors 2,085-3,313 with P(on) = 1; top: Dm3 12,716, Dm8 10,007, T4.T5 9,353, KC_ab_s 9,180, T4 9,001 |
| KaiR1D (CG3822) | 0.76 (R8_Rh6) .. 2,507 (Lai) | 65 (94 %) | 68 | 69 (87 %) | off only in photoreceptors; excluding them 12.8 (PAM_4) .. 2,507, P(on) = 1 in 64 / 65 |
| GluRIA | 0.18 (PAM_1) .. 487 (Lai) | 39 (56 %) | 42 | 42 (53 %) | off in L1 / L2 / Mi9 / T4 / T4.T5 / Tm2 / Tm9 / LC4 / LPi-34 and the photoreceptors, among others |
| GluRIB | 1.09 (Dm8) .. 550 (Lai) | 41 (59 %) | 44 | 44 (56 %) | off in L1 / L2 / T1 / Tm2 / Tm4 / Tm9 / Tm20 / LPi-34 and the photoreceptors, among others |
| Nmdar1 | 2.54 (L2) .. 216 (Kdm2) | 40 (58 %) | 43 | 43 (54 %) | off in L1-L5, Mi1, T1, T5, Tm2 / Tm4 / Tm9 and the photoreceptors, among others |
| GluClalpha | 13.6 (T1) .. 10,283 (PB_2) | 60 (87 %) | 63 | 67 (85 %) | off rows (69): Crz, Dm9, L1, T1, Tm9 and the 4 photoreceptors (R1-6 224, R7 121 TPM with P(on) 0) |

So: "Nmdar2 95-727 TPM in every Davis type, P(on) >= 0.5 in 87 %" -> 5.3-1,073 over neuronal rows, off in
photoreceptors, on in 94 % of neuronal rows (the 87 % is 69 / 79 with glia and muscle in the denominator); "VGAT 51-311
with P(on) = 1 in every type" -> 1.7-586, photoreceptors off; "Tdc2 4-15 TPM with P(on) 0.89" -> 0.6-35 TPM, P(on) mean
0.88 but on by the script's own rule in only 32 %; "DopEcR on in 85 %" -> 67 / 79 including glia and muscle, 93 % (64 /
69) of neuronal rows.

## 7. Open points

* Nern et al. 2025 / MaleCNS renamed or split several Nern-2015 types; the exact-tier rows rely on the annotator
  continuity (same author on both papers) and on the `flywireType` column where FlyWire kept the name. A one-line
  check per exact row against the Nern 2025 supplementary synonym table (typing-source task) would upgrade the
  evidence.
* PB_3 (P-EN) is contaminated per Davis's own note; PB_1 (Delta7, VGlut 4,113 TPM) and PB_2 (EPG) are clean.
* Davis's samples are mixed-sex heads; MaleCNS is male. Not addressed here.
* The `class` rows (ChAT / Gad1 / VGlut) are the only central-brain-wide receptor baselines in this source; the
  per-type central coverage is 3-4 % of cb_intrinsic output synapses and confined to MB / CX plus the five
  sign-0 peptide types.
* **Downstream rebuild still owed**: `flyverse/data/receptors_by_type.csv` was built from the round-1 KaiR1D column
  (davis2020 is `SOURCE_PRIORITY` 0). `scripts/build_receptor_table.py` must be rerun on the corrected table; the
  in-memory rebuild in `receptor_verification.md` (verify:derive) predicts class-rule glutamate flips 902,325 ->
  877,917 synapses, nonmda 1,613,925 -> 1,703,230, abs unchanged at 127,309 but KaiR1D-led, and net-call changes for
  9 (class) / 22 (nonmda) / 3 (abs) Davis-profiled types; the five new class rows will give IPC, CRZ01/02, DSKMP3 and
  NPFL1-I a Davis profile at tier `class` (all sign 0 presynaptically, so no synapse-sign effect from their own outputs).

## 8. Round-2 change log (every number that changed)

Rebuilt 2026-09-11 by `scripts/build_davis2020_tables.py`; both CSVs reproduce byte-for-byte across two runs
(sha256 expression `180d2de83d01...`, type map `2031fb8d68e3...`). Old vs new diff computed on the shipped files.

1. **Script**: copied from the session scratchpad; now repo-relative (`ROOT = Path(__file__).resolve().parents[1]`,
   inputs `data/external/davis2020/`, cache `cache/`, outputs `flyverse/data/`), no scratchpad constants; the JSON
   report is optional (`--report`). Asserts the two aliased gene_ids against dataTable3.
2. **KaiR1D := CG3822** (FBgn0038837; TPM from dataTable4, P(on) from dataTable7b, table-3 sample means for the 21
   suboptimal-only drivers). Changed cells: `KaiR1D_tpm` in all 100 rows, `KaiR1D_p_on` in 50 rows (48 of them to
   1.00, Muscles_Head 0.00 -> 0.69, PAM_4 0.00 -> 0.89; the other 50 are unchanged: 23 NaN rows = 21 suboptimal-only +
   lamina + opticlobe, 19 rows already 1.00, 8 photoreceptor / glia / muscle rows 0.00 in both). Old (CG8916) -> new
   (CG3822) TPM, selected: Lai 3.27 -> 2,507.41; Tm2 38.32 -> 115.11; LPLC2 4.21 -> 111.61; LLPC1 3.17 -> 108.31; Tm9
   19.40 -> 107.32; Lawf2 2.77 -> 98.67; T1 2.88 -> 94.39; TmY5a 10.37 -> 90.44; Tm4 145.90 -> 89.91; Dm11 8.99 -> 88.05;
   T5 51.19 -> 84.51; T4 114.26 -> 78.55; Mi4 115.00 -> 76.16; Dm10 190.50 -> 75.34; Mi15 155.76 -> 70.66; LC4 1.14 ->
   66.12; Mi9 37.80 -> 63.78; Mi1 1.63 -> 45.90; L1 120.99 -> 45.62; L2 65.67 -> 47.10; Dm4 50.08 -> 36.10; R1-6 3.18 ->
   8.45; R7 2.63 -> 5.67; R8_Rh5 0.32 -> 1.57; R8_Rh6 0.64 -> 0.76. "On" (P(on) >= 0.5 & TPM >= 10; TPM >= 10 alone where
   P(on) is undefined): 26 -> 89 of 100 rows, 63 rows off -> on, none on -> off. Header sentence now reads "KaiR1D
   (= CG3822, FBgn0038837; round 1 wrongly used CG8916 = FBgn0030707 / Lcch-14A)". No other expression column changed.
3. **Panel size**: 54 genes / 115 columns (4 meta + 54 tpm + 54 p_on + 3 call columns), not "57 genes" / "114 columns".
4. **QC-pass-only coverage bug**: a QC-pass filter was applied after an unstable per-type de-duplication, dropping
   types reached by both a QC-pass and a suboptimal-only driver. Fixed by a stable sort that prefers the QC-pass driver
   at equal tier (affected: R7p, R7d, LC10b, PAM02, PAM07, PAM08, PAM09, PAM10). Requoted: optic QC-pass 60 types /
   61,077 cells / 58.3 % / 47.9 % out / 35.5 % in -> **61 / 61,409 / 58.7 % / 48.0 % / 35.6 %**; optic both-ends QC-pass
   22.5 % -> **22.6 %**; cb_intrinsic QC-pass 19 types / 2,046 cells / 6.4 % -> **20 / 2,096 / 6.5 %** (exact / alias /
   fuzzy; 23 / 2,102 with the 3 QC-pass class types), 2.9 % out unchanged; whole-CNS QC-pass 37.8 % cells -> **38.0 %**,
   21.4 % out unchanged, both-ends 9.2 % unchanged.
5. **Class rows added**: Ilp2 -> IPC (16 cells), Crz -> CRZ01 (2) + CRZ02 (2), Dsk -> DSKMP3 (4), NPF -> NPFL1-I (2);
   4 unmatched rows removed. Type map 150 -> 151 rows; sources per tier class 5 -> 9, unmatched 12 -> 8; MaleCNS types at
   best tier class 1 -> 6. All-tier coverage: cb_intrinsic class 0 -> 4 types / 10 cells, matched 30 -> 34 types, 2,951
   -> 2,961 cells (9.2 %), out-syn unchanged (4.3 %; sign-0 cells); cb_endocrine matched 0 -> 16 cells; whole-CNS matched
   64,893 -> 64,919 cells, 38.8 % either way (round 1 printed 38.9 % from summed rounded parts). Transmitter cross-check
   pairs 134 -> 139, agreeing 120 -> 120. CCAP and Kdm2 stay unmatched (MaleCNS v1.0 has no CCAP-named type; Kdm2 is
   an enhancer trap with no cell-type identity).
6. **T1 wording**: "nothing > 5 TPM" -> "no synthesis enzyme > 5 TPM" (VGAT 172.6 non-diagnostic, cf. L1 304.6 / L2
   216.7 / Mi1 236.1; Tbh 5.75, Tdc2 5.49). l-LNv: SerT 5.4 TPM (below the 10 TPM floor), not 0; Trh 0.0.
7. **Evidence strings** (18 map rows): Lat1-5 cite the supp. 1A halves, the cell-type description (Tuthill et al 2013,
   Fischbach and Dittrich 1989) and the driver reference "Dionne et al in preparation"; l-LNv cites the description
   (Helfrich-Forster et al 2007) and the driver reference "a gift from Heather Dionne, Rubin lab"; T4a-d / T5a-d flagged
   "SUBTYPE-BIASED" with the verbatim `celltype.details`; Pdf, Kdm2, CCAP carry the genotype and reference. Numbers in
   those rows (n_cells, out_syn, nt) are unchanged.
8. **Gene ranges** rewritten with the actual neuronal QC-pass ranges and denominators (section 6b).
9. **Type counts** exclude the untyped pseudo-type: optic 629 -> 628 types (unmatched 557 -> 556), cb_intrinsic 6,606 ->
   6,605, visual_centrifugal 108 (new). Cell and synapse figures are unaffected.
10. Unchanged and re-confirmed: all-tier optic coverage (72 types / 61,928 cells / 59.2 % / 48.7 % / 36.9 %; both ends
    23.4 %), exact / alias / fuzzy counts (42 / 20 / 21 sources), class-selector sizes, every non-KaiR1D expression
    value, `tests/test_receptor_model.py` 8 passed (CPU, 10 s).
