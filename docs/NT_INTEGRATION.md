# Neurotransmitter and receptor data integration

Status: **rounds 1-5 done; the receptor plan is CLOSED** (section 7): the receptor-derived sign stage is the
default for the share of weight the data decide (26 % of |W|), reversible by one flag; the GF x0.3 input damping is
retired; the data are exhausted; the thread hands over to the dynamics questions listed at the end of section 7. This page records the question that prompted it, what data
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
  `brain._shaped_weights` (`LIFParams.receptor_model`); the default since round 3.
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
8. **Retire stop-gaps that the data replaces.** Each of `DEFAULT_PAIR_GAIN` and the AL LN override
   is re-tested with the receptor model on; anything the data reproduces is removed from the
   defaults (the anti-runaway retirement audit gives the procedure). The GF-input damping was the
   third item here and is **done**: retired in round 5 (`DEFAULT_TYPE_PATH_GAIN` keeps only
   LC4 / LPLC2 -> DNp01 x3; `GF_DAMPED_TYPE_PATH_GAIN` restores it), so it is no longer a stop-gap
   to re-test. Of what remains, dynamics round 1 found `DEFAULT_PAIR_GAIN[4]` a literal no-op
   (retirable now), `DEFAULT_PAIR_GAIN[1]` data-contradicted but load-bearing (re-labelled, not
   retired), and both the LPi x4 and drive-clip decisions blocked on the `walk.power_max` bound.

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
- **Re-examine the SILENCING rule's premise on photoreceptor -> medulla edges** (opened by dynamics round 2,
  2026-09-13; `docs/audits/receptor_integration.md` G.6.5-G.6.7). The rule at issue is the *silencing* rule --
  "a silencing (`fast_sign` 0) needs every source profiling the type to agree and at least two of them", i.e.
  that absent `HisCl1` / `ort` in a medulla type's transcriptomes its photoreceptor synapses transmit nothing
  -- and **not** the contested-flip rule (R3.1). The 45 histamine rows that carry the room's take-off cost are
  `fast_net_abs = none`, `fast_sign_abs = 0`, `fast_selection primary` and `flip_contested` empty on 45/45 with
  unanimous sources, while the class the flip rule *did* police (the T1 / Dm9 / Lai glutamate flips) carries
  nothing in the room, so re-examining `contest_flip` would touch the wrong class. Two sub-items, in order:
  1. **First, and it is not a table question.** These are the R7 / R8 (and R1-R6) -> Mi / Dm / C / L / Tm
     synapses -- 15,509 entries, 80,432 |W|, 90 % of the class's entries and 97 % of its |W| -- whose
     postsynaptic types are among the best-profiled cells in every optic-lobe atlas, so the `none` calls are
     the data's **strongest**, not their weakest. If they are right, the room cost is a **model** consequence
     of a *correct* silencing, and the question moves to `optic.py`'s use of the receptor factor on
     photoreceptor -> rate edges (`optic.py:165-225`: every optic-lobe edge, photoreceptor -> rate included,
     takes |count| x the row's fast sign) -- a **swappable-module** question, not a table question. Under the
     shipped default these direct photoreceptor inputs are zero in the medulla rate model.
  2. **Second, and only from expression data.** If the integration wants a **bound** rather than a silencing
     for this class (a tier-`silenced` prior kept at the presynaptic sign with a small gain), it must be
     justified from the expression levels (the `fast_neg_val` of these rows, all 0 under the group rule) --
     **never from the room take-off rate**. Deciding it on behaviour is hand-tuning toward behaviour.
  QC caveat the item carries: tier is `exact` on 34/45, `fuzzy` on 9 (Dm3a/b/c, LC14a-1, LC14a-2, LC14b, Pm1,
  Pm2a/b) and `alias` on 2 (LPi34, l-LNv); QC `pass` on 44/45 (l-LNv `suboptimal_only`). Immaterial by weight
  -- tier-`exact` rows carry 17,049 of 17,256 entries (98.8 %) and QC-`pass` rows 17,189 (99.6 %) -- but the
  earlier "all 45 tier exact, QC pass" was wrong and must not be re-quoted.

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
   (PASS), KC_active 816 vs 1426, take-offs (escape or voluntary) 24 vs 3 in 16 flies x 5 min (U 226, p 7.7e-5; one batch),
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

**Round 5 (closing; done; `docs/audits/receptor_verification.md` round-5 section):**

1. **The take-off cost is now an instrument.** Every hop in the batched room records its route (GF escape at 33 Hz
   vs voluntary wing power >= 50 Hz held 0.3 s; `batch_body` / `batch_sim` / `batch_sustain`, verified against the
   scalar model at 2e-12), and `benchmark.py --sections hops` (opt-in, 2,400 fly-s) scores voluntary and escape
   rates per 1,000 fly-s and the walking-GF median with references from the presynaptic-sign model. Measured, live
   route, 3 brain RNGs x 16 flies x 300 s per arm: pre-retirement default 75 = 31 escape + 44 voluntary vs off 9 = 9 + 0
   (voluntary p 4e-11); with the escape route disabled 30 voluntary vs 0 -- off never takes off voluntarily in 28,800
   fly-s. The shipped default's voluntary entry is a KNOWN GAP in substance (P(pass per draw) ~ 0.1); one 150 s draw
   measures the rate to a factor ~2.7, so quote >= 3.
2. **GF x0.3 input damping retired (default model changed).** `DEFAULT_TYPE_PATH_GAIN` keeps only LC4/LPLC2 -> DNp01
   x3; `GF_DAMPED_TYPE_PATH_GAIN` restores the old list (md5-pinned). Suite 27/0/2 in 4 of 4 draws (walk.power_max
   48.48 in 14/14), no check worse in status than the pre-retirement default; room take-offs not worse in either
   route (shipped 56 = 24 + 32 vs pre-retirement 75 = 31 + 44; p(shipped > pre) 0.77 / 0.95), the excess over off
   survives (6.2x; voluntary 32 vs 0) with an unmoved walking-GF tail (31.9 vs 32.25 Hz). The tally gain is the
   retirement's, not the receptor model's -- and the two interact: off with the damping retired fails walk.power_max
   at 95.5-97.1 Hz, so under the shipped gains the receptor model is ~48 Hz better than off on that check where under
   the damped gains it was 6.3 Hz worse. Skeptic verdict: sound.
3. **Attribution.** Taste, both smell checks and the three sugar checks are 100 % Brain-side (bit-exact in three run
   dirs); motion.min_dsi 100 % optic; loom.GF_peak mostly optic; walk.power_max both sides non-additively; walk.GF_max
   neither -- either half of the receptor signs alone raises the walking GF 2.5-2.9x while both together cancel.
   Fan-in normalisation is refuted (0 optic-side input_scale moves; MN9's fan-in bit-identical; the CPU dissociation
   survives with the normalisation off). The CPU double dissociation names the 123 Brain-side histamine silencings
   (282 synapses onto OA-AL2i3 / TmY14 / DNge14x) as what taste depends on and the 3,709 KC + DN1 glutamate flips as
   what smell depends on -- dependence, not magnitude (CPU +3.5 / 0.0 / +0.1 Hz over seeds).

**The default model at the close.** `LIFParams.receptor_model = 'sign'`, `receptor_net_rule = 'abs'` on the
contested-flip table (48,295 entries = 0.148 % of |W|; `None` restores the presynaptic rule byte for byte);
`DEFAULT_TYPE_PATH_GAIN` = LC4/LPLC2 -> DNp01 x3 only (`GF_DAMPED_TYPE_PATH_GAIN` restores the damping);
`TYPE_NT_OVERRIDE` = TmY14 glutamate, Mi19 serotonin, aMe8 ACh. Weight md5s: shipped 0e30e4a8, damped gains f0d145d1
(the round-3/4 default), receptor None on the shipped gains fcb5bec2, None on the damped gains 2e276b30
(pre-round-3) -- all pinned in `tests/test_receptor_model.py`. Suite 27/0/2 (walk.power_max 48.5 vs a hand-set 50 Hz
bound); costs: KC_active 816 vs 1426, and the room take-off excess (voluntary 2.2 + escape 1.7 vs 0 + 0.6 per 1,000
fly-s), which neither the KC / DN1 holds nor the DNp01 inhibition accounts for. **Dynamics round 1 (2026-09-12)
changed no default**: `git diff -- flyverse/` is empty at 305f507, and every gain it used -- the compass gE / gD
operating points, the GLNO=gaba relabel, the feeding drain scales -- was an experiment override, stated as such in
its own audit. The weight md5s above therefore still describe the shipped model.

**Handover, updated after dynamics round 1 (2026-09-12)** (the receptor data are exhausted: six sources, 26 % of
|W| decided; PEN / GLNO / PFL3 / hDelta / Tm5Y / TmY21 / LC11 / DNp04 / LPT unprofiled everywhere; the ring inert;
95 of 99 unknown-NT types unreachable). The round-5 list, each item with its first command, is in
`docs/audits/receptor_verification.md` (round 5, "Handover"); five threads ran it, and none of them moved a
default. Where each item now stands:

1. **Compass -- measured, and negative.** The bump survives full senses at all three GLNO-sign-robust operating
   points (gE 2 / gD 15, 2.25 / 25, 2.5 / 25): 38.0 s in 192/192 flies over 12 gain runs at 219-261 Hz, against a
   shipped control that dies within 0.1 s. It does **not** track heading (|circ corr(centre, heading)| <= 0.11),
   does **not** steer (yaw first harmonic 0.3-1.2 deg/s), and leaves PFN at 0.6-0.8 Hz and hDelta at 1.5-1.9 Hz,
   so **PFN -> hDelta -> PFL3 is answered NO**; the bump is pinned to 5-7 attractor sites out of 16 wedges. PEN
   has no signed rotation input in MaleCNS (GLNO is silent, fully contralateral and efference-fed; LNO / SpsP
   make 0-13 synapses onto PEN), and relabelling GLNO GABA gives an **elastic** +0.30 against +0.04 wedge L-only
   deflection (6/6 seeds, p 0.031). Visual rotation at 90 deg/s moves the bump 0.00 +- 0.01 w/s against a 4.0 w/s
   ideal. `out/cxroom_orig/`, `out/cxvfy/`,
   `out/verify_cx_shift_shift.md`, `out/vcx_*_s345.json`; `docs/audits/compass_room.md`, `cx_shift.md` (both
   still stubs; R2-0 fills them from data that already landed).
   **[Dynamics round 2] The efferent route is run, and the chain is live and unsigned.** With the
   proprioceptive / haltere transducer on, PS196_b goes 0.193/0.519 -> 1.437/1.505 Hz and GLNO 0.034/0.038 ->
   0.664/0.657 Hz in the 60 s room (5 runs/arm, `result` p 0.0079) -- the first time PEN's nodulus input
   carries anything in the plain fly -- and the bump still does not move: drift -0.0047..+0.0036 wedges/s on
   the phase means of all three arms against 4.0 ideal, every one of 36 arm-phase-runs inside
   -0.0097..+0.0060, GLNO L-R +26.93..+28.07 Hz in every phase. Under the labelled Coriolis stop-gap the
   chain is driven 13x harder (PS196_b 11.05/7.52 Hz in the turn) and **its L-R moves the same way in both
   turn directions** (+3.52 ccw, +3.95 cw), so the failure is a **sign**, not a magnitude. Structural
   reason: the only afferent class two steps from PS196_b is the haltere SApp, and `MotorRates.haltere` is
   one bilateral number; the sided leg channels reach it only at k = 3. **Signing GLNO remains moot** -- 12
   mV/s of symmetric PS196_b against a ring at 25-145 Hz. The compass item is now a **body-model** item
   (a side-split haltere MN readout in `motor.py`; a leg cycle in `body.py`), not a receptor or
   transmitter item. `docs/audits/vnc_drive.md` 6; `out/vncd/analysis/compass_*.csv`.
   **[Behaviour round 3] The body-model items are built and the answer is unchanged.** `body.LegCycle` + `motor.read_haltere_sides`
   (opt-in) make the leg afferents fire at 88 Hz and the haltere channel sided (L 15.3 / R 18.1 Hz); the signed self-turn report
   now exists at depth 1 (AN04B003 flip -9.8 to -12.2 Hz vs nulls ~1 Hz), reaches PS196_b at 1-3 Hz (-3.00 +- 0.29 under the
   transducer + unitary-high combination) and is gone at GLNO (|flip| <= 0.5 Hz) and PEN / EPG; drift |mean| <= 0.005 w/s vs 4.0
   in every arm. At the shipped gains no bump forms under any bracket of the per-transmitter unitary (48 / 48 survival 0.00 s): a
   transmitter scale cannot set the Delta7 : ring ratio. The compass item is now (a) a type-level ring mechanism and (b) the GLNO
   fan-in / sign-0 link, both untouched. `docs/audits/body_sided_state.md` 6, `unitary_strength.md` 4, `round3_integration.md` 5-6.
2. **Object -- the stage is named.** The figure is lost by ON/OFF cancellation at T2 / T3 / Tm5Y / TmY21 and then
   by l1 pooling at LC10 / LC11 (LC11 +0.046 mV, LC10a +0.080 against LPLC2's +0.54 and a 7 mV criterion). No
   hand-crafted optic measure sits on those edges and none of 16 ablations moves it; the LPLC2 object null under
   the shipped gains is now z +1.2 (rank-sum p 0.07) against round 3's +5.4, because the null rose, not the ball
   arm. `gain_fb = 0` is the deterministic null (small-field single-cell signal T2 0.066 against Mi4 0.047,
   reproduced to 0.000e+00 on an independent cluster run), and the stage pipeline is **not** reproducible at
   fixed seed, so every per-type z is +-1.5. The levers left are physiology parameters (slow GABA-B on T2 / T3,
   a per-type output normalisation at the LC cells), not data. `out/optic_audit/`, `out/optic_verify/`;
   `docs/audits/optic_measures.md`.
3. **Feeding -- cannot rank models, and the reason is not the metabolism.** Neither horizon nor drain makes meals
   countable: `meals` is identically equal to `contacts` in 12/12 runs, 0.13-0.50 per fly, p 0.4-0.8 everywhere.
   Starvation is not the limit. The closest-approach separation the audit proposed as a replacement **did not
   replicate** (5 of 6 runs, sign test p 0.22) and its negative tail is airborne; the drain's direction on
   food-finding is unresolved. The mechanism is the **terminal approach of the `cx` program** (plume Gaussian
   downwind only, no concentration-change rule, 81 % of flies ending upwind of the apple, closest approach
   2.3-7.2 cm against a 3.8 cm capture radius). `body.py` was untouched and stays so. `out/feedh_*.json`,
   `out/skfeed_*.json`; `docs/audits/feeding_horizon.md`.
4. **`walk.power_max` -- DECIDED (dynamics round 2, `docs/audits/anti_runaway.md` round 6): de-score it.**
   48.4805 Hz in 12 of 12 shipped-default GPU draws, margin 1.5195 Hz -- **inside the worst single-arm
   scatter on record** (11.86 Hz on `holdOptic`; the `off` suite arm alone spans 1.56 Hz). Non-monotone in
   the LPi scan (51.51 / 48.00 / 48.48) while `walk.GF_max` is monotone; the two path gains move it in
   opposite directions (32.41 at DN -> VNC x1, 34.24 with both off); **Spearman -0.600** against the room
   take-off rate over the four hold arms, the two FAILing arms being the two quietest rooms. The 50 is
   `body.Flight.takeoff_power_hz` (50 Hz held 0.3 s) applied to a per-frame maximum; the sustained form
   `walk.power_sustained_hz` keeps that referent and stays scored. **Done:** `scripts/benchmark.py:85` now
   carries `Ref(22, "notnone", 0, "4", note="REPORTED, NOT SCORED since session 10 ...")`, the form
   `loom.escape_cm` uses. Note `benchmark.py:135` makes `notnone` pass whenever the value is not None, so the
   row stays in the pass tally as a report -- "de-scored", not "unscored". **Not** to be re-derived from a
   gain scan. The `retire_measures.py` import defect that blocked this item's first command is fixed.
   De-scoring does **not** license retiring LPi x4, `drive_clip_mv` or the AL LN override.
   `out/pm_bound/`; `docs/audits/anti_runaway.md` round 6.
5. **Take-off -- optic-side.** Over 4 matched batches (19,200 fly-s per arm) the optic side of the receptor signs
   (44,463 medulla entries) carries **most** of the cost -- 75 % of the hop excess, 83 % voluntary, 95 % of the
   GF-median shift, having read 92-106 % on three batches -- and the Brain side (3,832 entries) **none** of it.
   The two halves do not cancel in the room as they do in `walk.GF_max`, so the pinned walk section is not a
   proxy; off with the damping retired is the same denominator as off with it (0.486 vs 0.625 per 1,000 fly-s;
   0 voluntary take-offs in 52,800 fly-s over 11 batches); and 0 of the 48,295 changed entries land on the
   take-off pathway.
   **[Dynamics round 2] Split and replicated: the histamine silencings carry it, the glutamate flips carry
   none.** Five arms in one submission x 4 runs, plus a fresh-seed replication at brain seeds 4/5/6
   (7 runs/arm pooled, 33,600 fly-s each). `holdOpticHis` (17,256 entries, 83,191 |W|, 90 % of entries
   photoreceptor -> medulla) is `compare`-**null** against the shipped default on hops, escape, voluntary and
   the GF median (pooled hops 0.840x [0.669, 1.054]) and `result` against off on all four (z +9.4 / +3.4 /
   +29.2 / +4.0). `holdOpticGlu` (27,207 entries, 87,920 |W| -- the **larger** perturbation) reproduces off
   on all four, with 0 voluntary take-offs in 19,200 fly-s. A 3,832-entry random draw from the carrying
   class, matched to the Brain side's entry dose and |W|, also reproduces off, so **dose in entries or |W|
   does not order the arms**. Two limits: the **escape** channel stays unattributable (the reported batch's
   103.6 % does not replicate -- fresh seeds 0.633x [0.353, 1.116]; at 7 v 7 even `off` vs shipped is
   `compare`-null on escape, |z| 1.92 < 3), and **dose in postsynaptic cells touched is not excluded and is
   structurally unclosable by this design** (off 0 / Glu 2,091 / Random 2,505 / His 10,411 / shipped 14,161
   orders monotonically with the outcome; an entry-matched subset can never match its class's cell count).
   Off's voluntary rate is 1 in 86,400 fly-s over 18 batches, not 0. Next split: per-row within the
   histamine class (Dm2 / Mi15 / Mi4 / Mi1 / L4 / C3 alone and their complement), same exposure.
   `out/d2_hold/`, `out/sk_d2_hold/`; `docs/audits/receptor_integration.md` G.6. `out/d1_*.json`,
   `out/sk_d1_*_4.json`; `docs/audits/receptor_integration.md` G.5.
6. **Optic stop-gaps -- one re-labelled, one candidate for retirement.** `DEFAULT_PAIR_GAIN[1]`
   (Tm4 / Tm9 / CT1 / TmY15 -> T5 x5) is data-contradicted: 78,877 of its 101,619 edges are Tm4 / Tm9 -> T5
   **acetylcholine** against 22,742 GABA, so it is a drive gain on T5's main excitation, not delayed inhibition.
   It is load-bearing (loom 29-36 Hz, T5 DSI 0.15 without it) and stays, **re-labelled and documented as a
   stop-gap**. `OpticParams.drive_clip_mv = 35` is a retirement candidate (11/11 checks without it) pending 3
   independent draws, and `DEFAULT_PAIR_GAIN[4]` is a literal no-op that can go now. `out/optic_audit/`;
   `docs/audits/optic_measures.md`.
7. **Proprioceptive / haltere transducer -- built, measured, adopted as a MODULE, not a default** (dynamics
   round 2; `docs/audits/proprioception_transducer.md`, `vnc_drive.md`). `senses.Proprioception` drives the
   941 wired-but-never-driven `vnc_sensory` / `sensory_ascending` proprioceptors from the body state the VNC
   motor neurons produce -- a `Wind` -> JO-shaped transducer, six literature-bracket ledger rows all
   `op report`, no number chosen against behaviour, default **OFF** and the shipped path bit-identical **on
   CPU** with it absent, attached-but-unfed, or `BatchSim(proprioception=None)`. It drives the wiring as the
   connectome said it would (AN04B003 0.589/0.314 -> 3.713/3.690 Hz; PS196_b -> 1.44/1.51; GLNO -> 0.66/0.66)
   and closes neither deficit (DNa02 moves a fifth of the way to threshold; PS196_b is symmetric).
   **What blocks a default, and what an adoption would need, in order:** (a) `rest.spikes_per_step` reads
   **6.0 against `< 5`, FAIL**, because the check is defined as "no input" and 740 tonic afferents are input
   -- a redefinition needs a ledger row for what a resting VNC reads, and no measured Drosophila FeCO /
   hair-plate / campaniform rate exists in any of the six rows; (b) four bench measures move 37-81 % while
   passing loose bounds (`taste.MN9_hz` -62 %, `smell.PN_hz` -64 %, `smell.KC_active` -37 %,
   `walk.GF_max_hz` +81 %) -- a cross-modal effect, unmeasured as to cause, on a body-less `brain.Brain`
   harness, with **one effective replicate per arm**; (c) the full 29-check suite x >= 3 draws with the sense
   on through `BatchSim`, plus `--sections hops` and the 4-batch room take-off protocol; (d) the loop gain
   per channel at the operating point (haltere: d(MN)/d(afferent Hz) = **0.163** on the ground, ~0.26 in a
   turn); (e) the two documented mismatches (35 SNpp19 prosternal/neck hair plates driven by the leg law;
   the leg variable is the MN rate, not a joint angle). **Closed:** the opt-in checkpoint/partial-reset gap
   (`BatchSim.state_dict` carries the motor snapshot; `CheckpointAndResetTests`). **Never a default:** the
   haltere Coriolis term -- a loop through `body.Locomotion`'s hand-written yaw scalar. If a minimal default
   is ever costed, cost **arm C** (the three leg channels), which reproduces the whole measured effect while
   the haltere channel alone reproduces none of it.
   **[Behaviour round 3] Extended with a leg cycle and a side-split haltere readout, still a MODULE.** Arm D
   (`'all+leg_cycle+haltere_sided'`): DNa02 0.54 / 0.38 Hz, clean yaw SD 7.9 deg/s, straightness 0.83, 7 / 16 flies off the table,
   a fixed +1.2 deg/s left drift from the connectome's own asymmetry; no clean frame above 100 deg/s; the sided term on DNa02 is
   tripod-locked and does not lead the yaw. **The C-vs-B effect is a LEVEL effect** (chordotonal 23 -> 88 Hz) not separated from the
   phase structure: the level-matched control (`'all'`, `mn_ref_hz` ~3.5) is the first thing to run. Owed for a default: that
   control, the ledger rows `lit.walk.*`, the hops section + room protocol at >= 6 runs per arm with the sided spec, `half_width_m`
   measured, `MotorRates.haltere_L/_R` (owner decision -- **deferred**, session 11). The pinned suite cannot carry the sense (only
   `sec_hops` builds a `BatchSim`).
8. **Monoamine slow class -- measured at data-anchored magnitudes, and parked again (behaviour round 3;
   `docs/audits/monoamine_slow_term.md`).** Coverage 27.2 % of the 1.88 M monoamine synapses carried, 0 onto the VNC / ascending
   neurons, none onto DNa02 / DNp09 / MDN / DNp01 / MN9. `add_low` 0.02: behaviour `null` (5 v 5 H200, 4 v 4 B200), +3 % spikes,
   `taste.MN9_hz` 10.93 -> 2.48 on CUDA and **1.97 < 2 FAIL on the CPU reference path**; 0.2 / 1.0 additive: runaway in 5 / 5 (the
   KC <-> DAN loop, causal on CPU); 0.2 gain: x2.55 spikes, MBONs zeroed, hopping. The anchors (Cohn 2015 ~0 mV on KCs; Longden
   2010 / Maimon 2010 x1.5-2 OA gain) are 50-100x apart on one scalar. **Do not scale the monoamine class with one scalar again.**
   What is needed: `SLOW_CLASSES` split per transmitter (`receptor_signs` slow_class per presynaptic transmitter; `slow_gain_by_class`
   / `slow_tau_by_class` with three keys, default None, CPU bit-identity test); a KC>MBON plasticity module gated by the DAN rate
   (`lit.MBON11.kc_mbon_depression`); separate E / I accumulators in gain mode; VNC receptor rows (no VNC expression source is in
   the builder's five). The `full` model stays opt-in; `sign` stays the default.
9. **Per-transmitter unitary strength -- an instrument, not a mechanism (behaviour round 3; `docs/audits/unitary_strength.md`).**
   `LIFParams.w_syn_by_nt` (default None, byte-identical) scales |W| per presynaptic transmitter before the cap. Data: ORN -> PN
   x0.79 of 0.275 (Kazama & Wilson 2008 over Tobin 2017's counts; the primary may read 7 mV = x1.11 -- pin it), PN -> KC x0.12-0.36,
   PN -> LHN x0.45-1.2; no Drosophila central unitary IPSP; the one insect unitary I/E is 0.28 (Periplaneta, J Neurosci 34:13039,
   to be added as a ledger row); `unitary.IoverE.chloride_driving_force`'s E_Cl endpoints are uncited. Every I/E < 1 bracket fails
   `walk.power_sustained_hz` (89-194 vs < 50) and hops the fly off the table; no bracket forms a bump. Next: an ACh-only family
   ({acetylcholine: 0.8} / 0.5, inhibition x1) through the suite, compass and room (transducer on and off), one block, with
   `taste.MN9_hz` re-read as the re-calibration it is. The field itself is **kept** as an opt-in instrument (owner decision,
   session 11); no bracket of it is adoptable.
10. **Optic stop-gaps -- the two adopt-alone suites were run and both candidates are NOT adoptable (`anti_runaway.md` round 7,
    `guard_suites_r3.md`).** `drive_clip_mv`: suite clean, room voluntary take-offs x2 (3.96 vs 1.94 per 1,000 fly-s; B200 4 v 4
    `result` p 0.029); the clip binds on the wing-power route. LPi x1: 34.6 per 1,000 fly-s, 48 / 48 flies with a walking GF above
    the 33 Hz escape threshold. Replacement statements unchanged (a bound on the optic -> spiking injected current; a sign-correct
    LPi -> LPLC2 strength from data), each to be scored in the room at >= 6 runs per arm. Item 6's 'pending 3 independent draws'
    for the clip is closed: the draws were run and the room refuted it.
11. **NT sources 4 and 5 -- the two public female connectomes (session 11; `docs/audits/flywire_banc_survey.md`,
    `docs/CONNECTOME_BACKENDS_SPEC.md`).** The receptor data are exhausted; the *transmitter* data are not. **FlyWire FAFB v783**
    carries six-class per-cell NT probabilities (`da/ser/gaba/glut/ach/oct_avg`) plus an `nt_type` call for 86 % of its 139,255
    cells; **BANC v888** carries a predicted nine-class label and, for **65,369 cells, a *verified* transmitter from the
    literature**, co-transmitters included (`glutamate,serotonin`). Read by type name against MaleCNS's 3,312 sign-0 bodies:
    dopamine is **solid** (395 MaleCNS -> FAFB DA 372; BANC verified dopamine 365); octopamine is mixed (OCT 37, **GABA 18**,
    absent 78; BANC octopamine 82, `?` 24); **serotonin is the least corroborated** (SER 72 but **DA 68**, ACh 20, Glu 12, absent
    239; BANC serotonin 56, **tyramine 46**, glycine 8, `?` 140) -- MaleCNS's `serotonin` class splits SER / DA / tyramine across
    sources, which is the same class item 8's slow-class split has to name per transmitter. And **about 400 of the 2,361
    `unknown` cells MaleCNS silences carry a classical-transmitter prediction in BANC** (ACh 150 / GABA 148 / Glu 111), so part of
    the silenced set is reachable without new receptor data. Conflict rows to add to the NT table: **PFL3** (ACh 24/24 in MaleCNS
    and FAFB, BANC *predicts* TYR 24/25 -- and BANC has PFL2 *verified* tyramine 12/12); **Delta7** (BANC verified
    `glutamate,serotonin` co-transmission against MaleCNS glutamate); **LAL074**, a PS059 input (MaleCNS / FAFB glutamate, BANC
    predicts SER on 2 of 4). Two rules come with the data: BANC's predictor **over-calls dopamine** relative to FAFB (8,072 vs
    584 DA cells brain-wide; 4.4 % vs 0.5 % of synapses), so BANC *predicted* monoamine labels are used only where its *verified*
    column agrees; and counts never cross releases unscaled (synapse yield MaleCNS : FAFB : BANC ~ 1 : 0.6 : 0.3). Nothing here
    is adopted -- it is a survey of two sources, and any use of them goes through the same audit-and-verify procedure.

Do not start a round 6 of receptor work.


## 8. Independent female connectomes: NT sources 4 and 5 (2026-09-14)

The backend integration adds **FAFB v783 per-cell probabilities** (source 4) and **BANC v888 verified
transmitters** (source 5). The public [FlyWire Codex releases](https://codex.flywire.ai/faq) are pinned by
URL and SHA-256 in `flyverse/data/manifest.json`. Reproduce the table below with
`python scripts/cross_connectome.py --out out/connectome_backends/anatomy`; `report.json` contains
per-release provenance, conflicts, and the complete type-level overlap with MaleCNS's sign-zero cells.
These sources annotate presynaptic transmitter identity; they do not measure live concentrations or
establish a postsynaptic receptor response.

FAFB's `neurons.csv.gz` contains `nt_type`, `nt_type_score`, and six probability columns
(`da_avg`, `ser_avg`, `gaba_avg`, `glut_avg`, `ach_avg`, `oct_avg`). The backend accepts its NT label at
score >= 0.5, otherwise `unknown`; the score threshold is recorded in the cache manifest. Of 139,255
cells, 28,991 are below threshold or have no score before the photoreceptor histamine rule is applied.
The unmodified scores are retained in `cache/fafb/nt_scores.parquet`. Its six-class predictor does not
provide a histamine probability. Photoreceptor identity supplies that label under the existing rule.

BANC's `Verified NT type` is present on 65,369 source rows before the non-neuron exclusion. When
present, it takes precedence over `Predicted NT type`. A verified co-transmitter string selects its
first classical transmitter (ACh/GABA/Glu/His), otherwise its first monoamine; the complete string
remains in `nt_verified`. A verified label containing only unsupported transmitters, such as nitric
oxide or glycine, becomes `unknown` rather than falling back to an incompatible prediction. This
includes 14 glycine-only and 7 nitric-oxide-only source rows. Tyramine is a new canonical identity
with **fast sign 0**. The receptor table has no tyramine column: it contributes neither a fabricated
fast sign nor a fabricated slow effect. Raw counts remain available at its explicit-zero CSR entries.

| Type | MaleCNS v1.0 compiled label | FAFB v783 (score >= 0.5) | BANC v888 verified-first label |
|---|---|---|---|
| PFL3 | ACh 24/24 | ACh 24/24 | TYR 24/25, ACh 1/25; prediction only |
| PFL2 | ACh 12/12 | ACh 12/12 | TYR 12/12, **verified** |
| Delta7 | Glu 42/42 | Glu 33/42, unknown 9/42 | Glu 40/40; verified `glutamate,serotonin` retained |
| LAL074 | Glu 2/2 | Glu 4/4 | Glu 2/4, serotonin 2/4; prediction only |
| PS059 | GABA 4/4 | unknown 4/4; `nt_type` missing, score 0 | GABA 4/4 |

These conflicts are observations, not reasons to change MaleCNS or add a female override. In particular,
FAFB's PS059 -> DNa02 anatomical contacts remain in the graph (311 L->L and 351 R->R synapses), but have
fast sign zero under the specified per-cell NT rule. Its GABA probabilities are only 0.37-0.44; the backend
does not silently substitute the source edge table's transmitter labels or MaleCNS's GABA assignment.
PFL3's predicted BANC tyramine is not verified by the PFL2 result. The Delta7 string documents possible
co-transmission that the single-`nt` fast-sign contract does not simulate as two simultaneous channels.
No MaleCNS unknown-type regex or type-level NT override is applied to either female graph.

Alias normalization changes the population behind an overlap count. Among MaleCNS's 2,361 `unknown`
cells, 795 have a normalized type represented in BANC. BANC cells belonging to those types include
1,841 ACh, 263 GABA and 199 glutamate labels. **These are not 2,303 individually recovered MaleCNS
labels**: the types can also contain cells that MaleCNS already labels, and cell counts differ across
releases. The preliminary survey's approximate 400-cell prediction overlap used a different name join.
The machine-readable report keeps both the MaleCNS denominator and the female label histogram.
Likewise, the 415 MaleCNS serotonin cells have 346 cells whose types appear in BANC; BANC cells of those
types include serotonin 160, dopamine 82, tyramine 42 and classical-transmitter labels. No one-to-one
cross-animal match is inferred.

The default compiler retains unclassified BANC rows: only the explicitly named glia/non-neuron/trachea
classes are excluded. All NT totals and input budgets therefore use the resulting 157,789-cell graph.
See `docs/audits/connectome_backends.md` for counts, alias decisions, and the distinction between missing
type matches and independently established sex-specific circuits.
