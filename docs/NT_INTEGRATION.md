# Neurotransmitter and receptor data integration

Status: **rounds 1-4 done; the receptor plan is closing** (section 7): the receptor-derived sign stage is the
default for the share of weight the data decide; the data are exhausted; a short closing round (evidence rewrite,
the take-off check on the right instrument, the GF-damping adoption run) and then the thread hands over to the
dynamics questions. This page records the question that prompted it, what data
exist, what the model does today, what we need, the task outline, and what each round found. Numbers in
sections 1-6 come from `docs/audits/nt_audit.md`; round-1 numbers from `docs/audits/receptor_*.md`.

## 1. The question

"Do we have data on not just neurotransmitter classifications, but concentrations of them in the
brain of Drosophila?" — and, following that, whether there is a newer connectome-to-transcriptome
alignment for MaleCNS specifically.

Short answer: nobody has per-synapse transmitter *concentrations* for the fly brain, and for a LIF
model that is not the quantity we want anyway. What the model needs is the **sign and relative
gain of each (presynaptic type, postsynaptic type) pair**, which is set by the transmitter the
presynaptic cell releases *and the receptors the postsynaptic cell expresses*. The second half is
what we do not use yet.

## 2. What the model does today

`flyverse/connectome.py`:

- One transmitter label per neuron from the MaleCNS `body-neurotransmitters` table: `consensus_nt`,
  falling back to `celltype_predicted_nt`, then `predicted_nt`, skipping `unclear`.
- `NT_SIGN`: acetylcholine +1; GABA, glutamate, histamine −1; dopamine, octopamine, serotonin and
  `unknown` 0. The sign is the presynaptic cell's; the postsynaptic cell plays no part.
- Sign-0 synapses are explicit zeros in `W` — they carry nothing and are excluded from fan-in
  normalisation.
- One override: `UNKNOWN_NT_OVERRIDE_REGEX` assigns GABA to 27 unknown antennal-lobe LNs.

Audit numbers (`docs/audits/nt_audit.md`): sign 0 silences **2.2 % of synapses** (2.73 M of
124.2 M) from 1.7 % of presynaptic neurons; by reason: unknown 0.7 %, dopamine 0.5 %, octopamine
0.5 %, serotonin 0.5 %. It is unevenly placed: mushroom body 9.8 % of input / 12.5 % of output,
gustatory 5.5 / 8.4 %, visual centrifugal cells 17.6 % of their output (the OA-AL2i, OA-ASM1,
LoVC octopamine / dopamine / serotonin types), PEN 21.9 % of input (unknown-NT partners). Every
optic / loom / optomotor / DN population is below 1.5 % silenced input, so the *missing* signs are
not the optic lobe's problem; *wrong* signs might be:

- Tm5Y (898 cells): model acetylcholine, literature glutamate (Davis et al. 2020 predictions).
- MN9: model acetylcholine, literature glutamate (skeletal motor neuron; harmless as an output).
- hDeltaK: consensus acetylcholine, cell-type prediction serotonin.
- Unknown optic types with many outputs and split T-bar votes: TmY14 (glutamate 49 % / ACh 48 %),
  Mi19 (serotonin 51 %), LoVCLo2 (serotonin 50 %).
- Glutamate is uniformly −1, but glutamate is inhibitory only where the target expresses GluCl;
  on iGluR / KaiR1D targets it is excitatory. This is a per-*postsynaptic*-type fact that the
  present scheme cannot express, and the optic lobe is full of glutamatergic types (Mi9, Tm9, Dm,
  Pm, LPi34/43, Δ7 in the CX).
- Monoamines are 0 rather than a signed slow term, which is why benchmark assay 7
  (hunger-modulated thresholds) has no route into the model.

## 3. What exists (sources)

**Per-neuron transmitter identity from EM (what we use).** Eckstein et al. 2024, *Cell* — synapse-
morphology classifiers, six transmitters, per-synapse then per-neuron labels with confidence.
MaleCNS ships the same style of prediction (`body-neurotransmitters-male-cns-v1.0.feather`,
`tbar-neurotransmitters-male-cns-v1.0.feather`). No receptors, no concentrations.

**No MaleCNS transcriptome alignment yet.** The MaleCNS paper's only expression alignment is
light-microscopy co-registration of *fruitless* and *doublesex* neurons (4,858 and 412 cells) —
sex-circuit markers, not a transcriptome. [Cell](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6),
[downloads](https://male-cns.janelia.org/download/).

**The bridge is cross-connectome cell typing.** FlyWire's whole-brain typing matched 3,643
hemibrain types and defined 8,453 in total; MaleCNS uses the same nomenclature, so anything keyed
to FlyWire / hemibrain type names transfers by name. Caveat from the same paper: about a third of
hemibrain types could not be reliably re-identified across brains, so a name join needs a
confidence column. [Schlegel et al. 2024, Nature](https://www.nature.com/articles/s41586-024-07686-5).

**Receptor-level data by cell type exists for the optic lobe** — where the model's open questions
(small-object pathway, loom mode, optomotor) sit:

- [Özel et al. 2021, Nature](https://www.nature.com/articles/s41586-020-2879-3): developmental
  scRNA-seq atlas of the visual system; clusters mapped to named EM types (Mi1 by *hth/bsh*, T1
  by *Eaat1*, GABAergic Pm1–3, photoreceptor targets by the histamine receptor *ort*, …).
- [Davis et al. 2020, eLife](https://elifesciences.org/articles/50901): per-type transcriptomes of
  visual neurons (driver-line sorted; the source of the "most LC / LPLC types are cholinergic"
  statement and the Tm5Y glutamate call), with receptor and transmitter-synthesis genes per type.
- [Nern et al. 2025, Nature](https://www.nature.com/articles/s41586-025-08746-0): complete
  optic-lobe connectome inventory with transmitter predictions per type in the same names — a
  second, independent NT label to cross-check the MaleCNS consensus against.

**Central brain:** [Fly Cell Atlas 2022, Science](https://www.science.org/doi/10.1126/science.abk2432)
(whole-fly single-nucleus atlas) and [Davie et al. 2018, Cell](https://www.sciencedirect.com/science/article/pii/S0092867418307207)
(aging brain atlas); clusters are coarser than EM types, so per-type receptor profiles are only
available for well-marked classes (Kenyon cells, DANs, ring neurons, PNs). Spatial transcriptomics
is being positioned as the bridge ([eLife reviewed preprint](https://elifesciences.org/reviewed-preprints/92618)).

**Cautions.** A 2026 *Neuroinformatics* note documents transmitter *and alias* ambiguity across
platforms for specific neurons ([Springer](https://link.springer.com/article/10.1007/s12021-026-09783-4)) —
exactly what a name-keyed join hits. There is receptor-mapping work for serotonergic modulation of
visual neurons ([bioRxiv](https://www.biorxiv.org/content/10.1101/619759.full.pdf)), which is the
kind of table we want for the monoamines.

## 4. What we need

Per cell type (MaleCNS name), from the transcriptomic sources:

| quantity | genes | what it decides |
|---|---|---|
| transmitter synthesis / transport | *ChAT, VAChT*; *Gad1, VGAT*; *VGlut*; *Hdc*; *ple, DAT*; *Tdc2, Tbh*; *Trh, SerT* | an independent NT label to compare with the MaleCNS consensus (resolve Tm5Y, TmY14, Mi19, hDeltaK, LoVC*, the 1,965 all-`unclear` cells where a type match exists) |
| fast receptors | nAChR subunits (*nAChRα1–7, β1–3*); *Rdl, Lcch3, Grd* (GABA-A); *GluClα*; *KaiR1D, GluRIA/IB, Nmdar1/2* (iGluR); *HisCl1, ort* | sign of the fast synapse for each transmitter on this target: GABA → −; glutamate → − (GluCl) or + (iGluR); histamine → −; ACh → + |
| slow receptors | *mAChR-A/B/C*; *GABA-B-R1/2/3*; *mGluR*; *Dop1R1, Dop1R2, Dop2R, DopEcR*; *Oamb, Octβ1R–3R, Octα2R*; *5-HT1A, 1B, 2A, 2B, 7* | a signed slow term: cAMP-raising (+): Dop1R1, Oamb / Octβ, 5-HT7; cAMP-lowering (−): Dop2R, Octα2R, 5-HT1A / 1B; Gq (+, calcium): Dop1R2, 5-HT2A / 2B, mAChR-A, Oamb |
| cluster → EM-type mapping with confidence | the papers' supplementary tables | the join key, with a flag for ambiguous or many-to-one matches |

And on our side:

- A weight-shaping stage that can set a synapse's sign from (pre transmitter, post receptor
  profile) rather than from the pre cell alone (since round 3 the DEFAULT, `receptor_model='sign'` / `abs`;
  `None` restores the presynaptic rule byte for byte), implemented as a swappable step in
  `brain._shaped_weights` (`LIFParams.receptor_model`), default off until it scores better.
- A slow-current term in the LIF for the monoamine / metabotropic class (a second, low-pass
  conductance with its own time constant per receptor class), so that dopamine, octopamine and
  serotonin become signed rather than zero. This is a model-level change (not a behaviour program):
  it adds a mechanism the connectome data already implies.
- Validation targets that already exist: `probe_figure_ground.py` (small-object signal), `probe_loom.py`
  (GF threshold and loom-rate coding), `screen_rotation.py` (optomotor), `probe_bitter.py` (sugar / bitter
  must keep replicating), `scripts/benchmark.py` (the extended suite), and the sustain runs.

## 5. Integration outline (optic lobe first)

Start in the optic lobe because the type matching is cleanest (Özel / Davis / Nern all use EM type
names), the rate model makes sign flips cheap to score, and both open model deficits live there.

1. **Acquire and pin.** Download the supplementary tables (Özel 2021 cluster annotations and
   per-cluster expression; Davis 2020 per-type expression; Nern 2025 per-type transmitter table;
   FCA brain per-cluster expression for step 6). Record accession numbers and file hashes in
   `docs/audits/receptor_sources.md`. Keep the raw files out of the repo (they are large and not
   ours); put derived per-type tables in `flyverse/data/`.
2. **Type map.** `scripts/build_type_map.py`: source cluster / driver name → MaleCNS `type`, with a
   confidence tier (exact name; alias table from the Neuroinformatics note and Schlegel 2024
   supplements; unmatched). Report coverage: fraction of optic-lobe cells and synapses whose pre
   and post types are both matched.
3. **Transmitter cross-check.** Per matched type, synthesis-gene calls vs MaleCNS consensus vs Nern
   2025 prediction. Output a disagreement table; propose sign changes only where two of three
   agree against the model (Tm5Y is the first candidate). The 1,965 all-`unclear` cells get a label
   where their type is matched.
4. **Receptor profiles → response classes.** Per matched type, a row per transmitter:
   fast sign (or "none"), slow sign (or "none"), and a relative gain class from expression level
   (binned, not linear — expression is not conductance). `flyverse/data/receptors_by_type.csv`.
5. **Edge lookup and model hook.** `connectome.receptor_signs(c, table)` returns a per-edge sign /
   gain vector; `brain._shaped_weights` applies it when `LIFParams.receptor_model` is set; the rate
   optic model (`optic.py`) takes the same vector. Add the slow-current term for the metabotropic
   class with per-class time constants (the numbers are parameters to be swept, documented as such).
6. **Score.** Rerun the optic probes and the benchmark suite with and without the receptor model;
   the specific hypotheses to test are (a) glutamatergic optic types signed per target change the
   figure-ground signal for Tm5Y / TmY21 / LC10 and the LPi → LPLC2 balance (the ×4 pair gain was
   a stop-gap for exactly this); (b) the loom-rate coding of the GF; (c) the optomotor DNp04 / LPT
   readout that never flipped. Accept only changes that improve the suite without breaking the
   sugar / bitter replication.
7. **Central brain.** Same pipeline with FCA / Davie 2018 for the confidently matched classes:
   PEN (21.9 % unknown input), ExR, ring neurons, Kenyon cells and DANs (assay 7, hunger), the
   octopaminergic visual-centrifugal cells (17.6 % silenced output; arousal / flight modulation of
   the optic lobe). Lower confidence, flagged as such.
8. **Retire stop-gaps that the data replaces.** Each of `DEFAULT_PAIR_GAIN`, the GF-input
   damping, and the AL LN override is re-tested with the receptor model on; anything the data
   reproduces is removed from the defaults (the anti-runaway retirement audit gives the procedure).

Deliverables: `flyverse/data/receptors_by_type.csv`, `flyverse/data/nt_by_type_transcriptome.csv`,
`flyverse/data/type_map.csv`, `scripts/build_type_map.py`, `scripts/build_receptor_table.py`,
`LIFParams.receptor_model`, `docs/audits/receptor_sources.md`, `docs/audits/receptor_integration.md`
(coverage, disagreements, scores), and a NOTES entry.

Execution plan: an ultracode workflow with phases acquire → map (parallel per source) → derive →
implement → score (parallel per probe) → skeptics (refute the sign changes) → critic (what is
uncovered), launched after the session-9 audit workflow reports, so that it can take the NT audit's
sign-0 and disagreement tables as its target lists.

## 6. Risks and open points

- Coverage: Davis 2020 covers driver-line-defined types (tens), Özel 2021 ~200 clusters; the
  optic lobe has ~700 types in Nern 2025. Expect partial coverage; unmatched types keep the
  present NT_SIGN rule, and the report must say what share of synapses is under which rule.
- Expression is not conductance; gain classes are coarse. The receptor model should first be
  tested as *sign only* (fast class), then with binned gains, so that each step is scoreable.
- Developmental atlas timepoints (Özel 2021) vs adult expression: use the latest (adult) timepoint.
- Sex: transcriptomic sources are mixed-sex or female; MaleCNS is male. Receptor differences
  between sexes are documented mainly in the fruitless / doublesex circuits, not the optic lobe;
  note it and move on.
- Alias ambiguity across platforms (the 2026 note): every join row keeps its source name so a
  disagreement can be traced.

## 7. Round 1 (session 10): what was built, what it found, what was wrong

Executed as a 17-agent workflow (five acquire/map agents, derive, implement, score, eight skeptics, one
critic; GPU work on the cluster). Deliverables: `docs/audits/receptor_sources_{ozel2021,davis2020,nern2025,
central,typing}.md`, `receptor_rules.md`, `receptor_nt_disagreements.md`, `receptor_integration.md` (scores),
`receptor_verification.md` (the skeptics' verdicts verbatim), the tables in `flyverse/data/` (`type_map_*.csv`,
`expression_*.csv`, `type_aliases*.csv`, `nt_by_type_transcriptome.csv`, `receptors_by_type.csv`), the builders
`scripts/build_*_tables.py` / `build_type_map_*.py` / `build_receptor_table.py`, `flyverse/data/manifest.json` +
`scripts/fetch_data.py` (55 external files, hashed), and the model stage `LIFParams.receptor_model`
(`connectome.receptor_signs`, `brain._shaped_weights`, `optic.OpticLobe(receptor=)`, `tests/test_receptor_model.py`).

**Coverage.** The receptor lookup decides the sign on 29.7 % of edges / 25.7 % of |W| synapses (exact 14.8 %,
fuzzy 7.0 %, class 2.7 %, alias 1.2 %); 74.3 % keep NT_SIGN. Optic module 52 %, visual projection 38 %, mushroom
body 93 % (class priors), central brain 5 %, VNC 0. Of the populations behind the open model questions only
LC10a, LPLC2, LPi34, LC4, T4/T5, HSN/HSE and the CX ring (EPG via the E-PG driver, PEN via a contaminated PB
driver) have a profile; **Tm5Y, TmY21, TmY13, LC11, Y3, Li19, DNp04, LPT27/30, DNp20, DNa02, MN9, PFL3, hDelta
and the sweet interneurons have none in any of the four expression sources.**

**Transmitter cross-check.** 296 types comparable outside mixed pools, 287 agree (97 %), 9 disagree, 0 proposals
under the two-of-three rule. Labels for unknown-NT cells that the data support: TmY14 glutamate (91 cells, three
transcriptome sources + Nern 2025), Mi19 serotonin, aMe8 ACh; T1's histamine label (conf 0.51) is marker-silent in
all four sources. Not yet applied to the model.

**Scores** (`receptor_integration.md`; net rule `class`): no mode beats the current model. `sign` flips 4.3 % of
glutamatergic synapses (902,325; Mi4 36 % of its input, L3 59 %, Mi9 22 %, T2a 20 %) and silences 286,600 (histamine
onto targets whose single-nucleus profile has no ort/HisCl call); it moves the figure-ground signal onto Mi4 (z 0 ->
3.1) but not onto Tm5Y / LC10a (best cell 0.14-0.29 mV vs a 7 mV threshold), raises the pinned loom GF peak 21-24 ->
32 Hz, and breaks the Shiu-rules sugar -> MN9 check (123.5 -> 5.6 Hz, a 657 spikes/step storm; the calibrated
replication survives, 4.6 -> 6.9 / 0.0). `sign+gain` and `full` storm (walking GF p99 68 Hz; KC 37 Hz, MBON 330 Hz,
31 spontaneous hops). DNp04 / LPT27 / LPT30 never flip in any mode. Compass: the receptor data confirm the
EPG <- ExR4/ExR6/ExR5/Delta7 glutamate is fast GluCl inhibition (the `cx_wedge` "untuned loop"), and change 0
EPG / PEN / Delta7 entries -- the compass is a gain / dynamics question, not a receptor one.

**What the skeptics found** (all "mostly sound"; full text in `receptor_verification.md`):
1. Davis 2020's `KaiR1D` column is gene CG8916 (a Cys-loop channel), not KaiR1D = CG3822; 15 nonmda / 9 class
   net calls rest on it. Class-rule flips 902,325 -> 877,917 after the fix (computed in memory, not yet shipped).
2. 59 % of the class-rule glutamate flips (531,322 syn) are ones the `abs` rule does not make (Nmdar2-led tertile
   calls where GluCl TPM exceeds the whole iGluR sum: Mi4, Mi9, L3); Mi9 / L3 / Tm9 flips are contradicted by every
   alternative source. `abs` and `nonmda` were never scored -- the "sign is worth keeping" verdict is untested.
3. Tier ordering lets a single-nucleus profile whose receptor group is "none" (dropout) outrank a Davis/Ozel
   profile with the group on: ~150 k Glu/GABA synapses silenced onto R7/R8, 14,953 R7 -> Tm5a/b and 13,013
   R8 -> Mi1 histamine synapses silenced -- artefacts on the object pathway's input.
4. Tier inflation: Ozel Pm1 "exact" is a 3-way MaleCNS split (Pm1/Pm5/Pm6); Tm29 exact rests on a name that
   FCA calls cholinergic vs MaleCNS glutamate; typing "none" rows omit 2,605 untyped cells; the FlyWire annotation
   release is v3.1.0 not v3.0.0; 903 of 4,589 FlyWire alias rows name types absent from the release; 47 Cm->Sm
   conflicts unflagged; Ilp2 -> IPC, Crz, Dsk, NPF left unmatched though MaleCNS has those types.
5. The slow term is not the mechanism the plan asked for: 98.4 % of its entries are classical metabotropic (mAChR-B,
   GABA-B, mGluR), the monoamines get a slow sign on 12 % of their synapses, the dopamine "+" rests on DopEcR in 560
   of 607 types, one tau for all classes, 4x the integrated charge of a fast synapse, none of it in the rate optic
   lobe, and it forces the Torch path.
6. Reproducibility: the Davis and central builders lived only in a session scratchpad (now copied to
   `scripts/build_davis2020_tables.py`, `build_central_agg_davie.py`, `build_central_agg_fca.py`,
   `build_central_map.py`; paths to be fixed); several quoted numbers were misread (see the record).

**Round 2 (done; `docs/audits/receptor_verification.md` round-2 section):** every round-1 table correction was
applied and the tables rebuilt (KaiR1D = CG3822; Pm1 / Tm29 / typing tiers; a silence needs >= 2 agreeing sources and
a single-nucleus "none" never outranks a whole-cell profile: silenced classical edges 286,600 -> 83,473 synapses, all
histamine; Kurmangaliyev 2020 added as a sixth source). Findings:

1. **Hypothesis (a) is closed on the receptor route.** Tm5Y, TmY21, TmY13, LC11, Y3, Li19 and Tm32 have no profile in
   any of the six sources (Kurmangaliyev's 199 clusters carry none of them; no marker exists to key Ozel's 106
   unannotated clusters); the lookup decides 0 input synapses of LC11 / Tm5Y / TmY21 / TmY13 and changes 0 exact-tier
   inputs of LC10a/b, LC16, LPLC2, LC4, T3 under any rule; and the moving-ball assay (`scripts/probe_object_sweep.py`,
   `docs/audits/object_sweep.md`) shows no object signal at LC11 / LC10a under off, sign-class, sign-abs or with T2/T3
   rectified ((ball - none) best-cell drive 0.03-0.48 mV vs a 7 mV threshold; the per-run pass/fail is a coin flip at
   the 1 Hz criterion). The object item becomes a medulla -> lobula wiring / dynamics question with that script and
   `probe_figure_ground.py` as benchmarks.
2. **The first suite-neutral receptor mode:** `receptor_model='sign'` with the absolute-level net rule (`abs`; 127,462
   glutamate flips + 83,473 histamine silencings = 0.17 % of |W|) scores 27 PASS / 0 FAIL / 2 known gaps in four of four
   suite runs against off's 26/1/2 and 24/3/2; it raises the pinned loom GF 19 -> 37 Hz and demo escapes to 12/12 seeds
   (off 3/12), with one measured cost (legacy loom.GF_peak -10 Hz, still PASS). It changes 0 entries of the object
   pathway, the optomotor readout or the compass core. **Not yet adopted**: 26 of its 40 +1 rows are contradicted by
   another source (Tm9: Davis alone vs three sources at -1; L1; 24 ER ring rows), no run holds Tm9 at -1, and the
   builder has no "contested flip" rule -- round 3. `class` (Shiu sugar storm, contradicted Mi9 / L3 / Tm9 flips),
   `nonmda` (ON pathway inverted) and the class-baseline fallback (21/6/2) are not candidates.
3. **Transmitter labels adopted (data-driven, default model changed):** `TYPE_NT_OVERRIDE` in `connectome.py` --
   TmY14 glutamate (three transcriptome sources + Nern 2025), Mi19 serotonin (EASI-FISH validated), aMe8 acetylcholine
   -- applied to those types' sign-0 cells: 107 cells, 95 output signs, 33,446 raw synapses (0.027 %); sign-0 share
   2.203 -> 2.176 %. Adopted because a three-seed suite run shows no check changing status; the local `cache/` was
   rebuilt (backup `out/cache_pre_override/`), the cluster's shared cache is rebuilt with the same code. T1's histamine
   label (conf 0.51) is unsupported by five sources but is not changed (1,777 cells; a fifth source, FlyWire top_nt,
   says ACh 873 / GABA 439 / His 0). The type-majority rule for the other ~496 unknown presynaptic cells is undecided.
4. **Compass:** the receptor model is dynamically inert on the ring (sign-class bit-identical to base in 6/6 runs; the
   ExR4/ExR5/ExR6/Delta7 -> EPG glutamate, 9.1 % of EPG's input, is fast GluCl inhibition by the E-PG driver data). The
   silent GLNO <-> PEN loop matters: GLNO glutamatergic abolishes the bump at gE 1.75 in 6/6 seeds, cholinergic holds it
   in 6/6; both EM predictions (MaleCNS T-bars 51 % Glu; FlyWire GABA 3/4) favour inhibitory, so the cx_wedge gains
   were tuned against a silent GLNO -- a GLNO=gaba gain scan is the next compass experiment (`docs/audits/cx_glno.md`).
5. **Slow term redesigned** (`docs/audits/slow_term.md`): per-class scale and tau (classical metabotropic default 0;
   monoamine only), additive / gain / threshold modes, an optic-lobe term; the round-1 runaway is gone (the class split,
   not the scale). Four unit-gain settings pass the four runaway-sensitive checks -- on the class fast weights, whose
   control fails Shiu; never run on abs weights; 'gain' acts on the net input (disinhibits net-inhibited targets); the
   KC-direction story did not survive verification. It stays an experiment flag for assay 7, not a default.

**Round 3 (done; `docs/audits/receptor_verification.md` round-3 section):**

1. **Default model changed, data-driven.** `LIFParams.receptor_model = 'sign'`, `receptor_net_rule = 'abs'` on the
   contested-flip table (a flip stands only when no other profiled source contradicts it; 26 abs rows removed: Tm9,
   L1, 24 ER ring rows): 48,295 entries = 179,944 |W| synapses = 0.148 % (30,916 glutamate flips onto iGluR targets,
   17,379 two-source histamine silencings); 95 % of the changed weight lands on optic rate units (T1, Dm9, Mi4, Mi1),
   8,833 synapses on spiking cells (KCg-m, DN1 clock, KCa'b', OA silencings). `None` reproduces the previous weights
   byte for byte (md5-pinned test). Evidence as corrected in round 4 (the round-3 '27/0/2 in 10 of 10' was measured
   with the benchmark's walk / motion sections half applied): 26/1/2 in 11 of 11 fully-applied suite runs, equal to
   off's best tally, no check worse in status than any off run; demo loom escapes 12/12 vs 3/12; bitter 3 seeds 139.9 / 138.9 / 131.5 vs 123.5 / 122.0
   / 114.7; figure-ground and object sweep within scatter. Costs on record: legacy loom.GF_peak 27-32 vs 37-44 Hz
   (PASS), KC_active 816 vs 1426, spontaneous take-offs 24 vs 3 in 16 flies x 5 min (U 226, p 7.7e-5; one batch),
   pinned-loom escape at the 33 Hz threshold in 2 of 6 runs (round-2 table 5/5 at 37 Hz; off 0/5 at 19). The
   justification the adoption rests on is the data-side argument for `abs` in `receptor_rules.md` section 3 (absolute
   GluCl vs iGluR level; contested flips removed), not the suite count.
   **Caveat found by the critic:** benchmark.py's legacy `walk` and `motion` sections built the optic lobe without the
   receptor lookup, so `walk.power_max` (the FAIL -> PASS) and the legacy `loom.GF_peak` (the cost) were measured
   with the model half applied; every room section and probe ran it as shipped. Fixed; re-score is round-4 item 1,
   and until then those two numbers are not properties of the shipped model.
2. **Step 8 under the new default:** none of LPi x4, GF x0.3, AL LN override can be retired -- each alone fails
   walk.power_max in 3/3 (51.56 / 69.77 / 72.8-79.3 vs 46.10); the session-9 "GF x0.3 goes" recommendation is retracted
   (without it the walking GF max rises 8.5 -> 9.2 Hz instead of collapsing); LPi x1 undoes the -10 Hz loom cost. All
   three verdicts rest on the half-applied `sec_walk` and are re-derived in round 4.
3. **Compass, GLNO=gaba scan** (+ the skeptic's seed-matched silent control): with the loop closed inhibitory the bump
   persists at gE 2 / gD 15 (180-184 vs 201-204 Hz silent), 2/8, 2.25/15-25, 2.5/25; totals equal (21/36 persisting
   runs per condition), the window moves down one gD step, is not narrower; the session-9 window was seed 0 only. The
   150-250 Hz rate problem is untouched; GLNO stays unlabelled; no compass default moves.
4. **Slow term on abs weights:** the assay-7 precondition is not met (walk.power_max fails in 11 of 12 active runs;
   gain 0.02 breaks Shiu at 5.7 Hz); the walk / motion sections are not bit-reproducible on the Torch path; every
   active arm used dop1r1 with no DopEcR control. Hunger stays an experiment flag.
5. **Object sweep with a null:** the none-vs-none null is +0.07-0.19 mV (a max over 95-275 cells); LPLC2 is the only
   type above it (z +5.4 abs; Welch +4.4 off; +0.10 / +0.21 mV vs the 7 mV criterion; invisible-ball control
   confirms the null); LC11 / LC10a / LC16 / LC4 at the null, LC10b marginal; the medulla carries the ball (Mi4 z
   +22 to +29). Hypothesis (a) stays closed; the object item is a separate plan entry (medulla -> lobula small-field
   wiring / dynamics; `probe_object_sweep.py --null` and `probe_figure_ground.py` are its benchmarks).

**Round 4 (done; `docs/audits/receptor_verification.md` round-4 section):**

1. **Re-score with the fixed benchmark (the round-3 caveat resolved AGAINST the round-3 headline).** Fully applied,
   the default is 26 PASS / 1 FAIL / 2 known gaps in 11 of 11 suite runs (the FAIL is walk.power_max 79.5 Hz against
   a hand-set 50 Hz bound), equal to off's best tally (off 26/1/2 x3, 24/3/2 x1; walk.power_max 73.2). "No check worse
   in status than any off run" still holds; the default stays, justified in kind (expression-derived sign, contested
   flips removed, reversible) and as "not worse", not as "better on the suite". The three round-3 numbers were
   half-applied artefacts: walk.power_max 46.10 PASS -> 79.47 FAIL (6.3 Hz worse than off); the "-10 Hz loom cost"
   is a +10-14 Hz gain; motion.min_dsi 0.17 -> 0.23. Off is again producible from the shipped scripts and reproduces
   the round-2 off values bit for bit.
2. **Attribution:** neither the KC nor the DN1-clock flips carry the taste rise (10.93 under both holds); DN1 carries
   65 % of the Shiu rise; walk.power_max is non-monotone in the applied flips; the taste rise is most plausibly fan-in
   normalisation from the optic-side synapses (untested).
3. **Take-offs replicate** (74 vs 11 hops over 48 flies x 300 s in 3/3 brain RNGs, p 1.4e-9; 95 vs 16 over four) and
   are NOT all spontaneous: at seed 0, 10 GF-escape + 14 voluntary vs off 2 + 0; the default's walking-GF tail is
   higher (median 33.0 vs 28.5 Hz, 8/16 vs 2/16 rows at the 33 Hz escape threshold). Feeding at --energy 0.9 produces
   meals but cannot rank the models. The scored hop check is owed, with a voluntary/escape split and a voluntary-only
   reference.
4. **Step 8 under the corrected benchmark:** all three round-3 verdicts withdrawn. The GF x0.3 damping CAN be retired
   (no_gf_damping 27/0/2 with walk.power_max 48.48 in 10/10 draws across two batches; damping DNp70 alone is
   inert -- the four inhibitory inputs are the whole effect); not adopted yet -- to be adopted alone with its own suite
   run and hop batches, since restoring 2,899 |W| of inhibition onto DNp01 is the one experiment that could move the
   take-off cost. LPi x4 cannot be retired (x1 fails; x2 is the only passing point of a non-monotone scan); the AL LN
   override cannot be retired.
5. **Compass:** three operating points robust to the GLNO sign (gE 2 / gD 15, 2.25 / 25, 2.5 / 25; 6/6 confined in both
   conditions); the shipped receptor default changes 0 ring-core entries and is byte-identical to off on the ring.
   The compass thread is now dynamics work.
6. **Type-majority rule:** proposes 3 types / 36 cells / 0 synapses; not adopted; cannot replace the 27-cell AL LN regex
   (the majority alone would contradict it in 6 of 8 types) and cannot reach the ~496 unknown cells (95 of 99 types
   without any independent call). Closes the session-9 open item negatively.
7. **Slow term:** `--deterministic` is not the tool -- the CSR x dense product is non-deterministic under the flag
   (per-op probe); the walk.power_max cost is the term (-27 Hz), not the dop1r1 signs (+3 Hz); assay 7 stays blocked
   because the term-off reference itself fails walk.power_max. Parked; no native slow kernel.

**Round 5 (closing; launched):** (a) the take-off check on the right instrument -- an escape / voluntary split in
`batch_sustain`'s JSON, 3 + 3 batches live and 3 + 3 at gf_hz = 1e9, a voluntary-only reference, and a scored
`hops` section for benchmark.py; (b) the GF-damping adoption run -- no-flag suite x3 on the edited default plus the
same hop batches under it (the experiment that links the walk.power_max FAIL to the behavioural cost); (c) the
Brain-side vs optic-side hold to settle the taste attribution. Then the receptor thread hands over to the dynamics
questions (compass with senses, PEN L/R shift, PFN -> hDelta -> PFL3; medulla -> lobula small-field wiring; a
metabolism that lets feeding rank models). The data are exhausted: six sources, 25.7 % of the weight decided, the
object-pathway types unprofiled everywhere, the ring inert, 95 of 99 unknown-NT types unreachable -- a round 6 of
receptor work would have nothing to measure.
