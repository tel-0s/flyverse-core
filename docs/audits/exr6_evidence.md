# ExR6 and ER6: transmitter evidence and the remaining DC-brake question

2026-09-15. Read-only investigation from `8854691`, branch `docs/exr6-evidence`.
Scope: the data question in `HANDOFF_EXR6_ASTRA.md` and
[`compass_dc_balance.md`, section 5](compass_dc_balance.md#5-what-an-adoption-would-require-nothing-is-adopted).
No model, label, receptor table, weight, gain, or benchmark status is changed.

## 1. Finding

**ExR6's glutamate identity and ER6's GABA identity have type-specific experimental
support. They are not supported only by EM predictions.** The missing evidence is
the strength, kinetics, and receptor localization of their particular connections
onto EPG and the two PEN types. In particular, transmitter identity alone does not
validate the model's large, fast, negative DC contribution.

[Wolff et al., 2025, *Cell type-specific driver lines targeting the Drosophila
central complex and their use to investigate neuropeptide expression and sleep
regulation*](https://doi.org/10.7554/eLife.104764.3), Figure 9, reports ExR6 vGlut
as **strong** and ER6 Gad1 as **weak** expression. These are qualitative EASI-FISH
scores, not numerical probabilities. The paper was a 2024 preprint, explaining
the older year in FlyWire's annotation strings. Adult females were assayed.
Its peptide dash means untested, whereas "none detected" means tested without a
detected signal. Both focus types have a dash, and both used only probe set 1,
which tests classical transmitters, not monoamines.

The independent source rows are in
[Figure 9 source data 1](https://cdn.elifesciences.org/articles/104764/elife-104764-fig9-data1-v1.xlsx):

| Type | Driver assayed | Marker call | Probe sets | Peptide result |
|---|---|---|---|---|
| ExR6 | SS53617 | vGlut | 1 | blank, not a negative assay |
| ER6 | SS58833 | Gad1 | 1 | blank, not a negative assay |
| ER4m | SS78687 | Gad1, ple | 1,2,4 | Dh31 |

The first two marker cells are worksheet D158 (bold) and D166 (regular), consistent
with Figure 9's strength legend. Figure 2's driver catalogue also lists SS53617
for ExR6, with an extra-expression caveat for one brain, and SS58833 for ER6, with
minimal VNC expression. Its ER4m anatomical line is SS41209; the assayed line
above is different. These mappings are anatomical assignments to the Hulse types,
not new transcriptome-cluster guesses.
[Figure 2 source data 1](https://cdn.elifesciences.org/articles/104764/elife-104764-fig2-data1-v1.xlsx)

**Recommendation:** retain the current two transmitter identities. No relabel or
new production receptor row is justified by this audit. The PEN glutamate response
deserves a separate compartment-specific investigation: section 5 includes a
recent primary preprint that conflicts with a uniformly inhibitory PEN assignment.

## 2. Reproduction and source boundaries

Run from the worktree with existing release caches:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python scripts/exr6_evidence.py --external D:/Projects/flyverse/data/external
python -m pytest tests/test_bit_identity.py -q --basetemp=out/pytest-exr6-evidence
```

The generator reads `cache/`, `cache/fafb/`, and `cache/banc/`; it cannot compile
or save a connectome. `--cache`, `--male-raw`, `--female-raw`, `--external`, and
`--out` make the paths explicit. Downloaded Wolff worksheets are hash-pinned and
kept under the output directory. No GPU, cluster submission, or timing benchmark
is involved.

`out/exr6_evidence/report.json` records input SHA-256s, output CSV SHA-256s, cache
manifests, script/commit identity, and cache MD5s before and after extraction.
The companion CSVs preserve every selected cell and all raw NT columns. In
particular, `male_cells.csv` retains all per-body and per-type predictions and
confidences; `male_tbars_by_body.csv` and `male_tbars_by_type.csv` retain all seven
vote fractions and mean probabilities. No classifier confidence is presented as
a confidence interval on biological transmitter identity.

Raw sources:

- MaleCNS v1.0: `D:/Datasets/male-cns-connectome-v1.0/flat-connectome/`, body and
  T-bar neurotransmitter feathers.
- FAFB v783: `D:/Datasets/flywire/Female Adult Fly Brain v783/neurons.csv.gz` for
  the backend's scores, plus the separately pinned, maintained
  [Schlegel/FlyWire annotation TSV](https://github.com/flyconnectome/flywire_annotations/blob/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv)
  for `top_nt`, `top_nt_conf`, `known_nt`, and `known_nt_source`. This TSV is not
  the static publisher ESM table and does not have identical prediction scores
  to the Codex CSV. Its local filename starts `schlegel2024_`.
- BANC v888: `D:/Datasets/flywire/BANC v888/neurons.csv.gz`, retaining predictions,
  verified strings, and verified neuropeptides separately.
- Existing expression and type-map CSVs under `flyverse/data/`; the source
  notebooks/audits remain the authority for their derivation.

Selected source SHA-256s (full values, not cache-compatibility fingerprints):

```text
body-neurotransmitters-male-cns-v1.0.feather
95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621
tbar-neurotransmitters-male-cns-v1.0.feather
bade84c9eab431dd537ff644aaf3d203d639a819c739ecedb338e7d109064f4d
schlegel2024_Supplemental_file1_neuron_annotations.tsv
9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be
FAFB v783 neurons.csv.gz
6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719
BANC v888 neurons.csv.gz
40a2201554a8c34d2c4b07c8322543a07c1b3faf5acafbac363fd1a3d0fa617f
elife-104764-fig2-data1-v1.xlsx
9ab5996b4f87abfe625f2e2789ddfa900c13489f4231b4b2be12c9e18209e2d1
elife-104764-fig9-data1-v1.xlsx
d6a81bd862dc2de8fdc34cd38d886423a1b36bb004374545b339e314a7d5fb40
expression_davis2020.csv
93bfb815e7c32c6c7dba31fe87d8f92908dec3ee4a74e55a7b57ea45a50dbd75
receptors_by_type.csv
7ed52d9c000e187402d60e28486e877af534f87c4d382314c6875d67898727b9
```

## 3. Cross-connectome labels

### 3.1. MaleCNS family and T-bar votes

In every row below, all bodies agree across `predicted_nt`,
`celltype_predicted_nt`, `consensus_nt`, and the compiled `nt`. This agreement
does not make those four columns independent evidence. Ground-truth annotations
are reported separately; the body file has no source-citation column.

Vote share means the fraction of the type's T-bars with the indicated class as
argmax across all seven predictors. It differs from both the release's confidence
and the mean softmax probability. All selected counts match the body table's
`total_nt_predictions`; there are no tied argmaxes. Source: `male_cells.csv` and
`male_tbars_by_type.csv`.

| Type | Cells | Shared label | Type confidence | Body confidence range | T-bars | Vote share for label | `ground_truth` |
|---|---|---|---|---|---|---|---|
| ExR1 | 4 | acetylcholine | 0.758879 | 0.728122-0.784826 | 7,438 | 77.75% | acetylcholine |
| ExR2 | 4 | dopamine | 0.639645 | 0.607751-0.670089 | 3,014 | 67.98% | dopamine |
| ExR3 | 2 | serotonin | 0.755810 | 0.751267-0.760268 | 6,905 | 86.81% | serotonin |
| ExR4 | 2 | glutamate | 0.627670 | 0.624997-0.630370 | 3,413 | 69.50% | blank |
| ExR5 | 4 | glutamate | 0.700834 | 0.690377-0.716079 | 7,286 | 78.04% | glutamate |
| ExR6 | 2 | glutamate | 0.634250 | 0.627669-0.641323 | 5,216 | 69.56% | glutamate |
| ExR7 | 4 | acetylcholine | 0.805858 | 0.798531-0.816190 | 5,974 | 82.57% | acetylcholine |
| ExR8 | 4 | acetylcholine | 0.928565 | 0.915266-0.934342 | 2,750 | 95.24% | blank |
| ER6 | 4 | gaba | 0.644028 | 0.630015-0.668604 | 2,890 | 69.34% | gaba |
| ER4m | 11 | gaba | 0.891531 | 0.883015-0.898581 | 12,813 | 97.85% | blank |

ExR6's next vote classes are ACh 23.06% and GABA 2.47%; ER6's are glutamate
19.17% and ACh 8.17%. These are classifier disagreements, not measurements of
co-transmission. Focus-cell detail follows; all ER4m bodies are preserved in the
same output CSVs rather than replaced by a representative cell.

| Male body | Type | Body confidence | T-bars | Glutamate votes | GABA votes |
|---|---|---|---|---|---|
| 10099 | ExR6 | 0.641323 | 2,514 | 70.41% | 2.59% |
| 12506 | ExR6 | 0.627669 | 2,702 | 68.76% | 2.37% |
| 14464 | ER6 | 0.637038 | 731 | 20.25% | 68.54% |
| 17419 | ER6 | 0.630015 | 720 | 19.17% | 67.78% |
| 512319 | ER6 | 0.640557 | 719 | 18.22% | 68.98% |
| 516110 | ER6 | 0.668604 | 720 | 19.03% | 72.08% |

### 3.2. FAFB probabilities versus known annotations

The compiled labels use the Codex CSV and threshold 0.5. They do not consume the
maintained TSV's `known_nt` column. `unknown` here describes the backend decision,
not the state of the literature. Source: `fafb_cells.csv` and `family_summary.csv`.

| Family | Compiled labels | Codex label-score range | Positive `known_nt` identities | Named annotation source |
|---|---|---|---|---|
| ExR1 | GABA 2, unknown 2 | 0.00-0.56 | ACh | Wolff EASI-FISH |
| ExR2 | dopamine 3, unknown 1 | 0.36-0.64 | dopamine | Nassel and Elekes 1992 immuno |
| ExR3 | unknown 2 | 0.00-0.41 | serotonin | Pooryasin 2015 immuno; Wolff EASI-FISH |
| ExR4 | ACh 1, unknown 1 | 0.40-0.58 | blank | blank |
| ExR5 | glutamate 4 | 0.53-0.55 | glutamate | Wolff EASI-FISH |
| ExR6 | unknown 2 | 0.00-0.49 | glutamate | Wolff EASI-FISH |
| ExR7 | ACh 3, unknown 1 | 0.00-0.57 | ACh | Wolff EASI-FISH |
| ExR8 | ACh 4 | 0.77-0.81 | blank | blank |
| ER6 | GABA 4 | 0.75-0.79 | GABA | Wolff EASI-FISH; Xie 2017 immuno |
| ER4m | GABA 11 | 0.87-0.98 | GABA, dopamine | Wolff EASI-FISH; Xie 2017 immuno |

For this family comparison only, ExR2 includes the released `ExR2_1` and
`ExR2_2` names. FAFB has two of each; BANC has one `ExR2`, one `ExR2_1`, and two
`ExR2_2`. The FAFB annotation TSV places the split names in hemibrain type ExR2.
This grouping changes no alias table and assumes no cross-animal cell pairing.
ExR2's monoamine identity cannot be generalized to ExR6.

The six Codex probabilities for the two ExR6 cells, in their recorded precision:

| FAFB root ID | ACh | GABA | Glu | DA | 5HT | Oct | CSV label / score | TSV `top_nt` / confidence |
|---|---|---|---|---|---|---|---|---|
| 720575940620027515 | 0.18 | 0.29 | 0.49 | 0.03 | 0.01 | 0.00 | GLUT / 0.49 | glutamate / 0.503016 |
| 720575940633145903 | 0.27 | 0.37 | 0.32 | 0.03 | 0.01 | 0.01 | blank / 0.00 | gaba / 0.356875 |

Both TSV rows carry glutamate-positive, ACh-negative, GABA-negative annotations
sourced to Wolff EASI-FISH. The zero label score in the second CSV row does not
mean all six probabilities are zero. Rounding also means these probabilities
need not sum exactly to one. `fafb_cells.csv` preserves the same fields for all
family cells, including each ER6 and ER4m root.

### 3.3. BANC predictions, verification, and peptides

Source: `banc_cells.csv`. Verified strings take precedence under the shipped
backend rule; their first classical transmitter determines the single fast `nt`.

| Family | Cells | Predicted NT / confidence range | Verified NT string | Compiled NT | Verified peptide |
|---|---|---|---|---|---|
| ExR1 | 4 | ACH / 0.44-0.60 | acetylcholine | acetylcholine | blank |
| ExR2 | 4 | DA / 0.78-0.94 | dopamine | dopamine | blank |
| ExR3 | 2 | SER / 0.66-0.78 | serotonin | serotonin | blank |
| ExR4 | 2 | GLUT / 0.68-0.70 | blank | glutamate | blank |
| ExR5 | 4 | GLUT / 0.69-0.76 | glutamate | glutamate | blank |
| ExR6 | 2 | GLUT / 0.72-0.75 | glutamate | glutamate | blank |
| ExR7 | 4 | DA / 0.54-0.62 | acetylcholine | acetylcholine | blank |
| ExR8 | 4 | ACH / 0.32-0.43 | blank | acetylcholine | blank |
| ER6 | 4 | GABA / 0.79-0.83 | gaba | gaba | blank |
| ER4m | 4 | GABA / 0.95-0.96 | dopamine,gaba | gaba | Dh31 |

ExR6 roots `720575941564468487` and `720575941643294920` have GLUT confidences
0.75 and 0.72 respectively, both verified glutamate. The ExR7 prediction/verified
conflict is particularly clear. ExR8's low-confidence ACh labels survive because
BANC does not apply FAFB's 0.5 rule. ER4m's full verified string is retained in
`nt_verified`, but a compiled GABA label does not simulate concurrent dopamine
or Dh31 release.
The cited ER4m assay establishes marker co-expression, not direct simultaneous
release at every ER4m synapse.

**Do not count three connectomes as three independent wet-lab confirmations.**
FAFB explicitly names the Wolff assay. BANC supplies verified strings without a
per-row citation; MaleCNS supplies `ground_truth` without one. Their agreement
with the assay is useful, but their independence is not established. Count and
classifier differences are not evidence of sex-specific transmitter changes.

## 4. Anatomy, function, and mapping limits

[Hulse et al., 2021, *A connectome of the Drosophila central complex reveals
network motifs suitable for flexible navigation and context-dependent action
selection*](https://elifesciences.org/articles/66039), Figures 14 and 56, places
ExR6 in recurrent EB/GA circuitry: it contacts EPG in the gall and reciprocally
connects with EPG in the EB, with selective PEN_a and EPGt contacts in the broader
ExR comparison. This is anatomical evidence, not a type-specific NT assay or a
measurement of tonic gain. The paper explicitly cites ER6-evoked inhibition of
EPG and allows direct gall connections or an indirect columnar route. Its
dense-core-vesicle observation concerns EL/PFGs, not evidence of peptide release
by ExR6 or ER6. ExR family membership therefore supplies neither a shared
transmitter nor a shared slow-timescale function.

The underlying experiment is
[Franconville, Beron and Jayaraman, 2018, *Building a functional connectome of the
Drosophila central complex*](https://elifesciences.org/articles/37017), Inputs
and Figure 4 supplement 3. The historical GB-Eo type, linked to ER6 by Hulse,
inhibits E-PG in an ex vivo optogenetic/calcium assay. Picrotoxin blocks the
inhibition. The authors explicitly allow GABA-A or inhibitory glutamate
transmission; picrotoxin does not distinguish those possibilities here. Population
stimulation/calcium imaging also does not establish monosynaptic latency or
exclude an indirect route. This supports an ionotropic inhibitory component,
not an ER6-specific GABA-B mechanism or a calibrated unitary LIF weight.

The older ring literature cannot all be treated as ER6 ground truth:

- [Omoto et al., 2017, *Visual input to the Drosophila central complex by
  developmentally and functionally distinct neuronal populations*](https://doi.org/10.1016/j.cub.2017.02.063)
  concerns the anterior visual pathway. ER6's anatomical predecessor R6 appears
  explicitly in [Omoto et al., 2018, *Neuronal constituents and putative
  interactions within the Drosophila ellipsoid body neuropil*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6278638/),
  Figure 3D, driver VT011965-Gal4. Morphological naming is not a transmitter assay.
- [Xie et al., 2017, *The laminar organization of the Drosophila ellipsoid body
  is semaphorin-dependent and prevents the formation of ectopic synaptic
  connections*](https://elifesciences.org/articles/25328), Figure 6, specifically
  tests Gad1/Rdl in R2/R4m and discusses R3. Its broad ring-neuron result should
  not be promoted to an independent ER6 assay merely because FlyWire includes
  that citation in an ER6 `known_nt_source` string.
- [Fisher et al., 2019, *Sensorimotor experience remaps visual input to a
  heading-direction network*](https://pubmed.ncbi.nlm.nih.gov/31748749/) and
  [Kim et al., 2019, *Generation of stable heading representations in diverse
  visual scenes*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8115876/) establish
  visual ring-to-compass mechanisms. Neither supplies the ExR6/ER6 marker assay
  used here; that assignment comes from the named Wolff drivers.

Existing expression coverage, reproduced in `type_maps.csv` and
`nern_family_coverage.csv`:

| Source | Relevant coverage | ExR6 / ER6 consequence |
|---|---|---|
| Davie 2018 / FCA 2022, `type_map_central.csv` | Class mappings for ExR2 (dopaminergic), ExR3 (Davie serotonergic), ER4m (Poxn) | No exact or alias expression map for either focus type |
| Ozel 2021 / Kurmangaliyev 2020 | No mapped focus-family profile | No justified cluster assignment |
| Nern 2025 optic-lobe SI Tables 1 and 5 | No exact focus-family name in either worksheet | No additional ExR6/ER6 EASI-FISH result from these optic-lobe tables |
| Davis 2020 | PB_2 -> EPG alias; PB_3 -> both PEN types, fuzzy | Target evidence only, detailed below |

Poxn is a mixed class, including non-ring cells; the committed map explicitly
excludes ER5/ER6. `type_map_typing.csv` is anatomical naming evidence, not an
expression profile. Absence from these selected-gene panels does not establish
absence of a transcript or peptide. The newer
[Epiney et al., 2025 CX nuclei atlas](https://doi.org/10.7554/eLife.105896.3),
Figure 8, offers E-PG/P-EN cluster assignments with marker validation, but no
named ExR6/ER6 profile. Subtype-specific receptor values have not been extracted
here; it is a follow-up source, not an invented mapping in this audit.

## 5. Receptors on EPG and PEN: a supported hypothesis, not a measured edge rule

The relevant table is **postsynaptic**. Missing rows for ExR6 and ER6 concern
inputs *to* those cells, not the sign of their outputs. EPG and both PEN types
already have glutamate and GABA rows. `receptor_rows.csv` reproduces their signs
and source tiers; the existing ER4m rows remain class-level receptor evidence
even though ER4m now has better presynaptic transmitter evidence.

The source is [Davis et al., 2020, *A genetic, genomic, and computational resource
for exploring neural circuit function*](https://elifesciences.org/articles/50901).
Below are the committed `expression_davis2020.csv` TPM and P(on) values, reproduced
in `target_expression.csv`. They are nuclear-RNA abundance and a fitted expression
probability, not receptor conductance or compartment-specific synaptic density.

| Gene | EPG source PB_2 TPM / P(on) | Pooled PEN source PB_3 TPM / P(on) |
|---|---|---|
| GluClalpha | 10282.57 / 1.00 | 4517.25 / 1.00 |
| mGluR | 17.65 / 0.00 | 795.61 / 1.00 |
| Rdl | 14889.51 / 1.00 | 7956.24 / 1.00 |
| GABA-B-R1 | 294.24 / 1.00 | 442.54 / 1.00 |
| GABA-B-R2 | 58.34 / 1.00 | 24.50 / 1.00 |
| GABA-B-R3 | 333.63 / 1.00 | 115.83 / 0.98 |

PB_2/SS00090 is the **alias** E-PG mapping; EPGt is excluded. PB_3/SS02268 is
**fuzzy**, pools PEN_a/PEN_b, and also labels Mi1/other optic-lobe neurons and GF;
the source's driver annotation says optic-lobe cells outnumber PB cells. These
limitations are already in `type_map_davis2020.csv` and
[`receptor_sources_davis2020.md`](receptor_sources_davis2020.md).

The current table selects fast -1 for both transmitters on all three targets.
For EPG it selects slow -1 for GABA and slow 0 for glutamate; for both PEN types,
slow -1 for both. Those are derived table decisions. EPG's GluClalpha and Rdl
expression make inhibitory ionotropic responses plausible, while the GABA-B
transcripts also permit a slow component. Pooled PB_3 cannot tell which PEN
subtype, cell, or compartment expresses each receptor, nor whether that receptor
mediates ExR6/ER6 input. Gene abundance cannot choose an exclusive fast versus
slow pathway, a receptor assembly, or the gain of either pathway.

### 5.1. A newer result against treating the two PEN rows as settled

[Eddy et al., 2026, *Divergent excitatory and inhibitory signaling in a head
direction circuit*](https://doi.org/10.64898/2026.01.18.700161), v1 posted January
21, is a **preprint**, not a peer-reviewed result. Its Figure 2 reports that local
PB glutamate puffs suppress EPG calcium but increase PENb calcium; PENa increases
less consistently. Responses persist under mecamylamine. Figure 3 reports strong
tagged GluCl in EPG (7 cells, 2 brains), weak signal in PENb (6 cells, 2 brains).
Excitatory kainate/NMDA receptor labeling was inconclusive. The
[author-paper full text mirror](https://www.researchgate.net/publication/400001621_Divergent_excitatory_and_inhibitory_signaling_in_a_head_direction_circuit)
was inspected; bioRxiv's API confirms the version and date.

This directly challenges the pooled transcriptome's uniformly inhibitory PEN
interpretation **in the PB**. It does not isolate ExR6, measure its EB/GA synapses,
identify the excitatory receptor, or establish a fast unitary conductance. Thus
neither a global PEN `fast_sign=+1` row nor an ExR6-specific sign flip follows.
Compartment-specific receptor tagging and selective stimulation are the decisive
next measurements; a type-only receptor row would generalize beyond the evidence.

## 6. Modulation and the reviewable recommendation

For ExR6 and ER6, peptide/monoamine co-transmission and a dominant slow effect
remain **unknown**, not negative. Wolff's focus rows do not test them. The blank
BANC peptide fields add no negative evidence. The ER4m dopamine/GABA/Dh31 result
is a useful comparator, not permission to transfer a modulatory identity to other
ring neurons. Nor does EPG/PEN expression of a monoamine receptor establish that
ExR6 or ER6 releases its ligand.

| Type | Transmitter decision | Outgoing response decision | What would settle the unknown part |
|---|---|---|---|
| ExR6 | Retain glutamate; direct vGlut assay supports existing label | EPG inhibition plausible; PEN sign/kinetics and EB/GA receptor placement unresolved | SS53617-specific stimulation with recordings from EPG, PEN_a and PEN_b; receptor localization and cell-specific loss/rescue in the actual contact compartments |
| ER6 | Retain GABA; weaker Gad1 assay plus inhibitory physiology support existing label | Ionotropic inhibitory component supported for EPG; exact receptor, directness, gain, slow contribution and PEN response unresolved | SS58833-specific stimulation with directness controls; Rdl versus GABA-B perturbation and recordings resolving early and sustained responses |

For co-transmission, test the same anatomically verified cells with the monoamine
marker sets and a named peptide panel, distinguishing untested from tested-negative
genes. Direct release measurements and loss/rescue would establish function beyond
transcript detection. Check driver off-target expression, both sexes, and the
receiving compartment before generalizing the result to the male simulation.

No replacement `TYPE_NT_OVERRIDE` entry is proposed. Its current contract applies
only to unknown/monoamine MaleCNS labels; classical glutamate and GABA cells keep
their consensus. Adding these two existing identities to that dictionary would
be a no-op, not a fix. Their literature evidence meets the non-classifier part
of the standard, but supplies no reason to relabel them. No numerical receptor
row is proposed: missing receptor identity, compartment, and kinetic measurements
must not be filled with invented gains or an unjustified tier promotion.

The owner can therefore close **"type-level transmitter wholly unknown"**, while
keeping **"what makes this input a large DC brake at the shipped gains?"** open.
The data do not license deleting, weakening, or slowing these synapses. Any later
model proposal still needs a scratch recompile, the suite across at least three
draws, and the owner's no-status-change gate. None was run or adopted here.

## 7. Validation

- The extraction preserves all selected cells when joining release annotations;
  all selected MaleCNS T-bar counts match the body release and no argmax ties occur.
- Each cached file has equal before/after MD5 in `report.json`. MaleCNS:
  `neurons.parquet` = `c50c598a708b5b373cbaffca7d6a9d82`;
  `W_post_pre.npz` = `ac131529cebf98decde58d0c227b7954`;
  `sign0_counts.npz` = `bf01d724acf2a1fec8fdb60ef8a9e066`.
- CPU bit-identity check: **3 passed, 5 subtests passed**, with the existing 14
  warnings (`out/exr6_evidence/bit_identity.log`). No golden was regenerated.
  This is a default-path regression check, not a biological test.

## Report

```yaml
summary: >-
  ExR6 glutamate and ER6 GABA have type-specific EASI-FISH support.
  Their output gain, compartment-specific receptors and modulatory status remain
  unresolved. Nothing is adopted.
key_claims:
  - Wolff SS53617 is vGlut-positive; SS58833 is Gad1-positive.
  - MaleCNS ground_truth and BANC verified labels agree with those identities.
  - FAFB compiled unknown ExR6 labels reflect its score rule, not absent literature.
  - ExR6 and ER6 peptide and monoamine tests are missing, not negative.
  - Existing PEN receptor rows pool a contaminated driver across both subtypes.
  - A 2026 preprint challenges inhibitory PEN glutamate responses in the PB only.
files_written:
  - docs/audits/exr6_evidence.md
  - docs/NT_INTEGRATION.md
  - scripts/exr6_evidence.py
  - docs/HANDOFF_EXR6_REPLY.md (ignored)
  - out/exr6_evidence/ (ignored reproducible extracts and source worksheets)
api: unchanged
validation:
  device: CPU only
  cache_bytes: unchanged
  tbar_counts: match body table
  bit_identity: 3 passed, 5 subtests passed
  biological_or_gain_benchmarks: not run
recommendations:
  - Retain both existing transmitter labels; no TYPE_NT_OVERRIDE edit.
  - Do not promote PB_3 to exact PEN subtype or synapse-specific receptor evidence.
  - Resolve receptor location, response kinetics and co-transmission experimentally.
  - Any later adoption remains an owner decision after the multi-draw suite gate.
open_questions:
  - Which receptors mediate ExR6 and ER6 contacts in EB and GA, separately from PB?
  - What are the direct unitary response and sustained response at physiological rates?
  - Do the focus cells co-release a monoamine or peptide?
```
