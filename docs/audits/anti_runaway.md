# The anti-runaway measures: what each stands in for, what breaks without it, what could replace it

Script: `scripts/retire_measures.py` (one process per configuration, each running the full `scripts/benchmark.py` suite:
29 checks, seeds 0 and 1, native backend unless noted; per-type adaptation cannot use the native LIF kernel, so those
runs and their baseline use the eager Torch path). Two complete replicates of every configuration exist:

* `out/retire_cluster/` -- 26 configurations run concurrently on the cluster (B200, seven jobs per GPU, one batch, guarded
  script). **Primary dataset**; the tables below are from it unless marked.
* `out/retire/` -- 19 configurations run sequentially on the desktop 4090 by the first attempt. The `no_cap` run of that
  set is **invalid**: it ran before the connectome guard was added to the script (file 15:15, run 15:06), and the
  `conn_cap = 0` branch of `brain._shaped_weights` aliases the shared `c.W` and multiplies the pathway gains into it in
  place on every Brain built (abs sum 121.4M -> 125.6M per build, compounding: DN -> VNC x3 per Brain), which is why
  that run's `bitter.shiu_*` checks moved (124 -> 49 Hz) although that section builds its own cap-free parameters. Every
  other local run either copies (cap > 0) or is guarded by its hook; they agree with the cluster replicate to the
  chaos-level noise documented in NOTES session 3 ("repeat a benchmark before trusting a 20% difference").

Per-check x per-configuration matrices: `out/retire_cluster/comparison.md` / `.json` and `out/retire/comparison.md` /
`.json` (`python scripts/retire_measures.py --report --out out/retire_cluster`). Connectome statistics behind each
measure: `out/retire/measure_stats.json`.

The seven measures are the model's hand-crafted layer on top of Shiu et al. 2024's uniform 0.275 mV synapse
(`flyverse/brain.py`, `LIFParams` and `_shaped_weights`; docs/NOTES.md session 3 records why each was added, sessions 4,
6 and 8 the pathway and type gains). Session 3's finding was that the point model with sensory input has two regimes,
silent or epileptic, and each measure moves one population from the second to the first. The question here is whether
each measure is a stand-in for a documented property of a *specific* population, in which case it should apply to that
population and not to the whole brain.

## 2. Results: the suite with each measure removed or replaced

Three replicates: local RTX 4090 (`out/retire/`, 19 configurations; its `no_cap` is contaminated -- see 3.1 -- and its
`adapt_by_type` row is a no-op duplicate of `baseline_eager` because of a script bug since fixed), the cluster batch
(`out/retire_cluster/`, 26 configurations on B200s, `comparison.md` there is the full table), and the skeptic's third
replicate (`out/retire_skeptic/`: the corrected `adapt_by_type`, plus `baseline_eager`, `no_same_type`, `soft_cap_120`,
`no_gf_damping`). Run-to-run spread of the native suite is large for some checks (walking GF p99 18-28 Hz in the baseline
across five runs; `walk.power_max_hz` bistable 61 / 79; `loom_escape` bistable at the threshold), so **a check counts as
broken only when it fails in every replicate that ran the configuration.**

| measure | stands in for | populations that need it (literature) | suite without it (breaks in all replicates) | proposed replacement | suite with the replacement |
|---|---|---|---|---|---|
| spike-frequency adaptation 1.5 mV / 200 ms, everywhere | Ca-activated K currents, spike-rate adaptation | documented in many central and VNC types; absent or weak in persistent-activity cells (EPG, ring), motor neurons, GF | **6 breaks**: taste MN9 5.85 -> 0.52, DNa02 leg asymmetry 2.58 -> -0.19, MDN top clique 152 -> 267 Hz (AVLP/CL storm), wing power sustained 37 -> 138, walking GF p99 20 -> 50-98, LH clean 4.4 -> 12.7 | `adapt_by_type`: 0 mV in CX columnar + ring neurons + motor neurons + GF, 1.5 mV elsewhere | **26 pass / 1 fail (bistable power_max), breaks none**: taste 8.97, p99 27.2, power sustained 39.2, loom 48.5 Hz 2/2 escapes, odour clean 4.46, rotation flip -6.9 |
| connection cap 60 synapses | saturation of unitary EPSP with synapse count; gap-junction and dendritic normalisation | 154,008 connections (0.6%) exceed 60; the cap keeps 94.2% of weight | breaks DNa02 asymmetry (-0.13), MDN storm (265 Hz), odour clean (8.1); taste MN9 49.6 | soft cap `120 (1 - exp(-n/120))` (knee at the old cap) | breaks none in both replicates; taste 8.5-10.4, KC active 1599 -> 763; the 'fixes walk.power_max' of the first draft was a bistable draw |
| same-type damping x0.1 | absence of strong recurrent excitation within a type (KC-KC, ORN-ORN axo-axonic contacts are largely not synaptic) | KCs (1.15 M KC->KC synapses), AL LNs, ORNs, a few cliques; NOT the visual projection types (LC4 146 / LPLC2 250 within-type synapses per cell, all excitatory) | **no break survives replication** (p99 40.0 / 33.4 / 29.0); but KC rate 2.9 -> 12.2 Hz, taste MN9 24 Hz, loom GF 31 -> 57-93 Hz | `same_type_by_type`: x0.1 only on AL LNs, KCs, ORNs, FR1, DNg33, AVLP/CL giants | passes in both replicates with no margin (power sustained 49.5 vs bound 50, p99 28-36, legacy escape at 17 cm, loom GF 75-81); adding the VP types breaks power sustained (50.1) and silences smell (PN 3.0) and walking GF (0) |
| fan-in scaling (5000 / total)^1 | input resistance and dendritic attenuation of giant neurons | GF (41k shaped inputs, x0.12), DNa02 (x0.31), HSN (x0.33); 2,435 neurons above 5,000 shaped inputs | breaks walking GF p99 (58 Hz, 2-4 voluntary takeoffs) in both | `fan_in_sqrt`: (5000 / total)^0.5 | passes in both (p99 32); `ref 10000` breaks odour clean (7.4); giants-only breaks odour clean (8.3) |
| antennal-lobe depression u 0.2 on ORN + LN terminals | ORN -> PN short-term depression (Kazama & Wilson 2008), LN gain control | ORNs, AL LNs | breaks odour clean (92 Hz; PN mean 85, max 337, LN 176) in both; so does ORN-only | none that passes; keep as is (`std_orn_ln` is the current default, see 1.5) | -- |
| DN -> VNC x3, VP -> DN x2 | descending-neuron output efficacy, electrical coupling in the escape pathway | DN -> leg / wing premotor networks, GF | breaks nothing, but the DN motor maps collapse: DNa02 legL 3.08 -> 0.64, DNp09 premotor 45 -> 24, MDN IN06B020 152 -> 62, wing power sustained 37 -> 15; removing only VP -> DN x2 breaks nothing (bitter calibrated 4.6 -> 8.1) | `path_gain_typed`: LC4 / LPLC2 -> GF x6, GF -> TTMn / PSI x10, no superclass gains | same DN-map collapse as dropping DN -> VNC x3 |
| GF input damping x0.3 (five inputs) | none documented: it was tuned against walking-induced GF bursts | four of the five damped inputs are inhibitory under NT_SIGN (SAD073, GNG300, CL367 GABA; PVLP010 glutamate; only DNp70 cholinergic) | breaks nothing in either replicate; walking GF max 8.5 -> 1.1 Hz | **retire it** (or damp DNp70 alone: identical result) | -- |
| type gains LC4 / LPLC2 -> GF x3 | loom pathway efficacy (electrical GF inputs) | GF | breaks nothing (loom_escape bistable) | keep, or fold into the typed set above | -- |

Combinations: `all_replacements` (adapt_by_type + soft_cap_120 + same_type_by_type_vp + fan_in_sqrt + typed gains, eager)
breaks walking GF p99 (44.1) with taste MN9 45.9 and loom GF 84 Hz; `all_replacements_native` breaks DNa02 asymmetry
(0.23) with taste 12.6. The replacements interact (taste MN9 rises under both the soft cap and per-type same-type
damping), so they must be adopted one at a time and re-scored.

Recommendations that the numbers support: (1) drop the GF x0.3 damping; (2) switch adaptation to `adapt_by_type`
(compass, ring, motor and GF unadapted) -- this is also the adaptation half of the working compass found in
`cx_wedge.md`; (3) `soft_cap_120` and `fan_in_sqrt` are each safe alone and worth adopting one at a time; (4) keep the
AL depression and DN -> VNC x3; (5) same-type damping stays global until the visual-projection question (LC4 / LPLC2
within-type synapses) is settled by the figure-ground assay, since the per-type list passes only without margin.

Every number in this section and in section 1 was measured with `LIFParams.receptor_model = None`. Under the round-3
default (`'sign'` / `'abs'`) recommendation (1) no longer holds -- dropping the GF x0.3 damping breaks
`walk.power_max` in 3 of 3 replicates -- and neither do the GF-damping rows of the table above; see "Round 3: with
the data-driven signs" at the end. The other rows have not been re-measured under the new default.

## 1. What each measure stands in for

### 1.1 Spike-frequency adaptation (`adapt_jump` 1.5 mV / spike, `adapt_tau` 200 ms, every neuron)

Model: every spike adds 1.5 mV of hyperpolarising current that decays with 200 ms; at rate R the steady adaptation is
0.3 mV x R, i.e. 15 mV at 50 Hz against a 7 mV threshold. Added in session 3 (at 1.0, then 1.5) as the one brake that
does not block a feed-forward command: depression is presynaptic-rate dependent and had silenced the VNC; adaptation is
per neuron, so a 150 Hz descending command still gets through while a *recurrent* clique throttles itself.

Physiology it stands in for: slow negative feedback on firing (Ca-activated and M-type K currents, Na-channel slow
inactivation). Documented in the fly for the olfactory receptor neurons and their transduction (Nagel & Wilson 2011),
for Kenyon cells (which fire few spikes per odour; Turner et al. 2008), and it is the generic property of most
cholinergic central neurons recorded in whole-cell mode. It is **absent, or overridden, in the populations that hold
persistent activity**: the compass (EPG heading bump persisting in darkness for tens of seconds, Seelig & Jayaraman 2015;
PEN / PEG / Delta7 of the same attractor, Turner-Evans et al. 2017, Green et al. 2017; the fan-shaped-body goal and
PFL cells, Hulse et al. 2021), tonic ring neurons (Omoto et al. 2017; Sun et al. 2017), and the motor neurons: the flight
motor neurons (DLMn, DVMn) fire one spike per muscle potential for the whole of a flight bout, TTMn and the GF fire a
single spike per escape (Allen et al. 2006), and leg motor neurons fire tonically at rates set by posture (Azevedo et
al. 2020). The 1.5 mV / spike figure itself is not from any measurement; it is the smallest value at which the session-3
cross-type AVLP / CL clique (AVLP154 / 157 / 488 -> AVLP520 / AVLP428 / CL212 / CL002, 280 Hz) stayed quiet.

Where it bites in the model (in-synapse counts do not matter here; it is about who sustains rate): the AVLP / CL clique,
the antennal-lobe LNs (lLN1 at 140 Hz with it, 280 without), the VNC premotor cliques behind the wing-power motor
neurons, the SEZ interneurons (GNG141 / 038 / 042 at 240 Hz during tasting).

### 1.2 Per-connection cap (`conn_cap` 60 synapse-equivalents)

Model: a connection of n synapses contributes min(n, 60) x 0.275 mV = at most 16.5 mV per presynaptic spike. 154,008 of
25.58M connections (0.60%) are above 60 synapses; they hold 13.4% of all synapses, and the cap keeps 94.2% of the
connectome's synaptic weight. The largest are within the central brain (51,539 over-cap cb -> cb connections), the VNC
(21,374), the optic lobe (18,552), VNC -> motor neurons (8,327) and cb -> DN (8,190); the AVLP giants are 435 synapses
per cell (120 mV per spike under linearity), DNg49 / DNge125 -> MNnm13 500+.

Physiology it stands in for: the unitary PSP does not grow linearly with synapse number. Each synapse is a conductance
whose driving force shrinks as the dendrite depolarises, synchronous release from one axon onto one dendrite saturates
the local membrane, and release probability is below one; in the antennal lobe, Tobin et al. 2017 found PN unitary
EPSPs about equal across glomeruli whose ORN -> PN synapse counts differ several-fold, the dendrite size compensating.
This is a property of *every* connection, not of a population, so a global rule is the right shape; what is not
physiological is the hard knee. A conductance-like saturation A (1 - exp(-n / A)) is the natural form (linear for
small n, the same asymptote): with A = 60 a 60-synapse connection drops to 38 equivalents and a 20-synapse one to 17;
with A = 120 (the knee near the old cap) 60 -> 47, 435 -> 117.

### 1.3 Same-type damping (`same_type_gain` 0.1 on every within-type synapse)

Model: synapses between two cells of the same type are scaled x0.1. Within-type synapses are 2.8% of the connectome
(2.5% excitatory), concentrated in a few populations (excitatory within-type synapses per cell): lLN1_bc 2,895 (34% of
its input), GNG117 944, DNg33 748, INXXX149 632, FR1 483 (32%), PFNv 452 (48%), KCg-m 309 (60%), LC17 284, PEN 278,
LPLC2 250, EPG 231, LC9 186, LPLC1 155, LC12 152, LC4 146, ORN 67. Added in session 3 as the cure for the lLN1_bc / FR1 /
DNg33 cliques and for the "DLMn clique" (which is not within-type wiring: DLMn + DVMn have 44 within-type synapses in
total, the GF 2, TTMn 0; the DLMn storm ran through the VNC premotor network and was fixed by adaptation, not by this).

Physiology it stands in for: two different things. (a) Populations whose members are **electrically coupled** and fire in
synchrony rather than exciting each other through chemical recurrence -- documented for the antennal-lobe local neurons
(LN-LN and LN-PN gap junctions; Yaksi & Wilson 2010; Huang et al. 2010), for the giant fibre's output synapses (GF ->
TTMn and GF -> PSI are shakB gap junctions with a chemical component; Phelan et al. 2008; Allen & Murphey 2007), and
for motor-neuron pools in other insects; the connectome's chemical within-type counts for these are small anyway
except for the LNs. (b) **Axo-axonic within-type synapses** whose function is presynaptic / modulatory rather than
somatic excitation: KC -> KC (1,153,845 synapses = 63% of all KC input; KCg-m alone 415k = 57% of its input, with no measured recurrent KC excitation --
KC output is sparse and APL-gated, Lin et al. 2014; Takemura et al. 2017; Zheng et al. 2018 find the same counts in
hemibrain / FAFB), ORN -> ORN inside a glomerulus (Tobin et al. 2017; Horne et al. 2018), and the LC / LPLC / MeTu
axon bundles in the optic glomeruli (Wu et al. 2016) -- LC4 and LPLC2 carry 146 / 250 within-type synapses per cell,
all excitatory in the sign table, with no documented recurrent excitation among LC neurons. The compass columnar
cells (EPG 231, PEN 278, PFN 98-452 per cell) *are* recurrently wired in the attractor models, but through EPG <-> PEN,
not EPG -> EPG; session 8's compass audit found their within-type synapses irrelevant to the (failed) bump.

Populations for which the damping is not supported: the descending neurons (DNg33 excepted), VNC interneurons and
motor neurons, the lateral horn, the central complex. The literature-supported replacement is therefore a per-type
list: AL LNs, KCs, ORNs, the visual projection types, and the measured session-3 cliques (FR1, DNg33, AVLP / CL giants).

### 1.4 Fan-in scaling (`input_norm_alpha` 1, `input_norm_ref` 5000)

Model: a neuron whose total (shaped) input exceeds 5,000 synapse-equivalents gets every input scaled by 5000 / total.
2,717 neurons (raw counts) are above 5,000 inputs, 667 above 10,000, 136 above 20,000 (on the shaped totals the model actually scales: 2,435 and 467); they receive 21% of all synapses.
On the shaped totals the GF's factor is x0.12 (41k equivalents per cell after the VP -> DN x2), DNa02 x0.31, HSN x0.33,
DNp09 x0.56, DNp18 x0.61, DM1_lPN x0.55, lLN1_bc x0.96 (factors on the shaped totals, as `Brain` computes them; the first draft quoted raw-count factors); TTMn, MN9, MDN, EPG, KCs and LC4 / LPLC2 are untouched (x1). Added in session
2-3 for the GF (~40k inputs), which otherwise fired from a few hundred active walking-related synapses.

Physiology it stands in for: input resistance falls with cell size, so one synapse depolarises a giant neuron less. The
GF is the textbook case (a giant axon with a low input resistance, hard to fire except by the loom volley or a
mechanical shock; von Reyn et al. 2014; Ache et al. 2019); the lobula-plate tangential cells (HS / VS) are large and
graded; and Gouwens & Wilson 2009 / Tobin et al. 2017 show the PN's dendritic size compensating its synapse number.
So the property is real and size-dependent, but it is a *dendritic* property (synapses on a large dendrite are
attenuated and the cell integrates more of them), whose strength in the animal is far weaker than 1 / total: a cell
with four times the inputs does not need four times the active synapses to fire. The two literature-consistent
shapes are a sublinear exponent ((ref / total)^0.5: the GF x0.35, DNa02 x0.46, HSN x0.55) or a threshold rule that
touches only the giants (only cells above 10k shaped inputs, by 5000 / total or 10000 / total).

### 1.5 Antennal-lobe depression (`DEFAULT_STD_U_BY_TYPE`: u 0.2, tau 300 ms on ORNs and AL LNs)

Model: Tsodyks-Markram depression on the terminals of the 2,635 ORNs and 317 AL local neurons (a PN pattern in the default map never matched a cell because the patterns are applied with `re.match`; it has been removed, so the documented and the effective model now agree: ORN + LN only, PN terminals undepressed); steady resource
1 / (1 + u R tau): 0.45 at 20 Hz, 0.14 at 100 Hz. Session 3 switched depression off everywhere else because it blocked
descending commands, and put it back here because the PN <-> cholinergic-LN loop ran at 300 Hz from 1 Hz of spontaneous
ORN input.

Physiology: ORN -> PN is the classic strongly depressing synapse of the fly (Kazama & Wilson 2008: paired-pulse and
train depression to roughly half within a few spikes, release probability high), and the antennal lobe's gain control
is presynaptic GABA-B inhibition of ORN terminals by the GABAergic LNs, scaling with total ORN activity (Olsen & Wilson
2008; Root et al. 2008) -- this is the population-level normalisation that lets PNs report odour identity across
concentrations. Depression of LN -> PN, PN -> LN or PN -> PN transmission is not characterised (the excitatory LN
network of Olsen et al. 2007 / Shang et al. 2007 is what the model's PN <-> LN loop is). So the literature supports
u ~ 0.2-0.5 on ORN terminals; the LN / PN depression is the stand-in for the presynaptic inhibition the model does not
have (its LNs are GABA-A-like fast synapses onto PN somata / dendrites, not GABA-B onto ORN terminals).

### 1.6 Pathway gains (DN -> VNC x3, visual projection -> DN x2)

Model: superclass-level multipliers on the shaped weights (session 4). DN -> VNC x3 restored descending drive after the
cap and adaptation had silenced it (DNa02 -> ipsilateral leg MNs, MDN -> backward-walking set, DNp09 -> its premotor set,
each specific, no storm). VP -> DN x2 compensates the GF's fan-in factor on the LC4 / LPLC2 volley (DN -> VNC x3 covers
271,675 connections, x2 28,999).

Physiology: neither superclass is a population with a documented uniform synaptic gain. What is documented is
(a) the GF's loom input: LC4 and LPLC2 make **mixed chemical + electrical (shakB) synapses** onto the GF dendrite (Ache
et al. 2019; von Reyn et al. 2017; Klapoetke et al. 2017), the electrical component being what the cap, the fan-in
scaling and the uniform chemical synapse cannot represent; (b) the GF's output: GF -> TTMn and GF -> PSI (-> DLMn) are
large mixed electrical synapses that fire the jump and flight motor neurons one-to-one (Phelan et al. 2008; Allen &
Murphey 2007), which the model represents as 90 chemical synapses; (c) that single DN types drive behaviour when
activated alone (MDN, Bidaye et al. 2014; DNp09, Zacarias et al. 2018; DNa02, Rayshubskiy et al. 2020), so the DN ->
VNC synapses are strong, but there is no measurement that they are three times stronger than a central-brain
synapse, and the connectome's DN -> VNC connections are already large (4,056 above the cap, DNg49 / DNge125 -> MNnm13
at 464-533 synapses). The typed replacement supported by the literature is the GF pathway itself: LC4 / LPLC2 -> GF
x6 (the current effective x2 x x3) and GF -> TTMn / PSI x10 as the electrical-synapse stand-in, with no superclass
gains. Whether the DN -> VNC x3 can be dropped is an empirical question the suite answers below.

### 1.7 Type gains (LC4 / LPLC2 -> GF x3; SAD073 / GNG300 / DNp70 / CL367 / PVLP010 -> GF x0.3)

Model: `DEFAULT_TYPE_PATH_GAIN`. The x3 is session 6 (with the x2 pathway gain, LC4 / LPLC2 -> GF is x6 in total); the
x0.3 is session 8's damping of "the five central-brain inputs that fire the GF during ordinary walking". The GF's input
by type: LC4 6,362 synapses, LPLC2 4,862 (31% of its input together), DNp70 1,416, PVLP122 1,216, SAD064 1,215, SAD073
1,177, PVLP010 711, PVLP123 620, PVLP151 603, CL038 549, JO-B1_a 545, CL367 537.

Physiology: the x3 is the electrical LC4 / LPLC2 -> GF synapse (above). The x0.3 has no physiological referent, and
under the current neurotransmitter table four of the five damped types are **inhibitory** (SAD073, GNG300, CL367
GABA; PVLP010 glutamate; only DNp70 is cholinergic): damping them removes 70% of 2,899 inhibitory and 1,416 excitatory synapse-equivalents from the GF (2,029 and 991), i.e. and
425 of excitation. It was added when the GF's walking-time drivers were ranked by input weight without sign; with the
sign, the measure works against its purpose (the ablation below: walking GF max 8.5 -> 1.1 Hz without it).

## Round 3: with the data-driven signs

Everything above was measured with the presynaptic-transmitter sign rule (`LIFParams.receptor_model = None`). The
default is now the receptor model: `receptor_model = 'sign'`, `receptor_net_rule = 'abs'` (flyverse/brain.py; adopted
in docs/NT_INTEGRATION.md section 7 / docs/audits/receptor_verification.md round 2 on 27 PASS / 0 FAIL / 2 KNOWN GAP in
3 of 3 no-flag suite runs, `out/r3_default_{1,2,3}.json`). Three stop-gaps that had never been scored were run under
it, one at a time, together with a baseline: `scripts/retire_measures.py` gained `--receptor-model` /
`--receptor-net-rule` (pass-through to benchmark.py like `--sections`; `off` = "leave LIFParams alone", which under the
round-3 default means sign / abs -- every run records the LIFParams it actually used in `config.lif`), `--check` (print
a configuration's effective parameters without simulating) and `--report-replicates`, plus the three configurations
`pair_gain_lpi_x1`, `no_gf_damping` (already there) and `no_al_ln_override`. Its weight hooks also had to be fixed:
`brain._shaped_weights` takes a third argument (the per-edge ReceptorSigns) since the receptor block was added, so
every hook in the script raised `TypeError: _protected() takes 2 positional arguments but 3 were given` and the first
attempt at this batch produced 12 runs with all 29 checks MISSING.

**The batch.** One cluster batch, 12 jobs (4 configurations x 3 replicates), `--seeds 0,1,2`, full suite, native
backend, the shared override cache (sum|W| 121,460,584), run dir `$CLUSTER_RUNS/r3-retire-33dbb1`, 10.3 min
wall, 0 failed; 5.7-9.3 min per run. Results `out/retire_r3/<config>_r<N>/<config>.json` (+ `.log`), the full
configuration x replicate x check matrix in `out/retire_r3/replicates.md` / `.json`
(`python scripts/retire_measures.py --report-replicates out/retire_r3`). Rule, as above: **a check counts as broken
only when it fails in every replicate of the configuration while passing in every baseline replicate**; no check
differs in status between the three baseline replicates, and no configuration differs in status between its own
replicates, so nothing here rests on a single run.

### What each configuration is, structurally (local CPU, the same cache; `out/r3_retire_structure{,2}.json`)

* `pair_gain_lpi_x1` -- `optic.DEFAULT_PAIR_GAIN`'s `LPi34|LPi43 -> LPLC2` factor from 4.0 back to 1.0, every other
  pair gain kept. The edges: 1,231 (180 LPi cells -> 185 LPLC2 cells), 7,543 synapse-equivalents, presynaptic
  transmitter glutamate on all 1,231, 2.16 % of LPLC2's 349,724 |W| input; against T4/T5 -> LPLC2 112,941 |W| the
  ratio is 0.067 at x1 and 0.134 with the gains in force (LPi x4 vs T4/T5 out x2). The receptor table decides
  **100 %** of LPLC2's input (102,618 of 102,690 edges; 349,724 of 349,724 |W|) and changes **0** signs there: the
  data confirm the LPi inhibition (glutamate -> GluCl) and every other LPLC2 input sign, and say nothing about its
  strength. Retiring the x4 is therefore a magnitude question the receptor data cannot settle.
* `no_gf_damping` -- `SAD073 / GNG300 / DNp70 / CL367 / PVLP010 -> DNp01 x0.3` removed, `LC4 / LPLC2 -> DNp01 x3`
  kept. Those five are 21 edges / 4,315 |W| of the GF's 36,589 |W| input (SAD073 1,177 GABA, DNp70 1,416 ACh,
  PVLP010 711 glutamate, CL367 537 GABA, GNG300 474 GABA; LC4 6,362 / 126 edges, LPLC2 4,862 / 185 edges).
  DNp01 has **no receptor profile**: 0 of its 1,455 input edges are matched under any rule, so the round-3 default
  changes nothing at the GF's own input -- only the drive that arrives there.
* `no_al_ln_override` -- `connectome.UNKNOWN_NT_OVERRIDE_REGEX` emptied for the run (a documented monkeypatch in
  `retire_measures.no_al_ln_override_connectome()`, which compiles the connectome in-process (30-69 s in the shipped logs);
  `connectome.load(rebuild=True)` is not used because it would save over the shared cache, and connectome.py is not
  edited). 27 cells in 13 types (l2LN22 3, lLN2T_d 3, v2LN40_2 3, v2LN41 3, lLN2F_a 2, v2LN30 2, v2LN31 2, v2LN3A 2,
  v2LNX01 2, vLN27 2, lLN10 1, vLN26 1, vLN29 1) lose the GABA label and go back to nt 'unknown' = sign 0: 13,287
  output edges, 136,223 |W| (0.112 % of the connectome; per cell 380 min / 1,472 median / 26,314 max), onto
  cb_intrinsic 106,111, cb_sensory 29,213 (ORNs 26,097; PN-named types 37,000-49,000 depending on the name rule (the 42,005 first quoted could not be reproduced from the cache; the generator was not shipped)), descending neurons 858.
  sum|W| 121,460,584 -> 121,324,368, unknown-NT cells 2,361 -> 2,388, every other entry identical; the compile gives
  the same numbers locally and on the cluster. With the label on, 10,093 of those edges (76,105 |W|) are
  receptor-matched and the abs rule changes 0 of their signs -- the postsynaptic profiles are consistent with GABA.

### The suite, configuration x replicate

Twelve checks are identical in all 12 runs: `rest.spikes_per_step` 0.00, `taste.MN9_hz` 10.93, `dn.DNa02_L_leg_asym_hz`
2.58, `dn.DNp09_top_hz` 152, `loom.escape_cm` 3.50, `motion.correct_directions` 8, `loom_escape.escapes` 3,
`bitter.calibrated_sugar_MN9_hz` 5.52, `bitter.calibrated_sugar_bitter_MN9_hz` 0.00, `bitter.shiu_sugar_MN9_hz` 139.90,
`bitter.shiu_sugar_bitter_MN9_hz` 0.82, `compass.wedge_cells_persisting` 0 (KNOWN GAP). The rest:

| check | base r1 | base r2 | base r3 | lpi_x1 r1 | lpi_x1 r2 | lpi_x1 r3 | no_gf r1 | no_gf r2 | no_gf r3 | no_ln r1 | no_ln r2 | no_ln r3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smell.PN_hz (< 100)             | 7.86  | 7.86  | 7.86  | 7.86  | 7.86  | 7.86  | 7.86  | 7.86  | 7.86  | 5.04  | 5.04  | 5.04  |
| smell.KC_active (> 0)           | 816   | 816   | 816   | 816   | 816   | 816   | 816   | 816   | 816   | 1805  | 1805  | 1805  |
| dn.MDN_top_hz (< 250)           | 153   | 153   | 153   | 153   | 153   | 153   | 153   | 153   | 153   | 150   | 150   | 150   |
| walk.GF_max_hz (< 38)           | 8.52  | 8.52  | 8.52  | 12.26 | 12.26 | 12.26 | 9.22  | 9.22  | 9.22  | 5.51  | 5.51  | 5.51  |
| **walk.power_max_hz (< 50)**    | 46.10 | 46.10 | 46.10 | **51.56 FAIL** | **51.56 FAIL** | **51.56 FAIL** | **69.77 FAIL** | **69.77 FAIL** | **69.77 FAIL** | **72.78 FAIL** | **79.29 FAIL** | **79.29 FAIL** |
| walk.power_sustained_hz (< 50)  | 21.36 | 21.36 | 21.36 | 24.82 | 24.82 | 24.82 | 37.49 | 37.05 | 37.49 | 28.63 | 30.37 | 30.37 |
| loom.GF_peak_hz (>= 20)         | 27.05 | 28.04 | 27.05 | 38.30 | 38.50 | 38.30 | 32.47 | 33.74 | 29.93 | 34.77 | 33.75 | 35.22 |
| rotate.DNp20_flip_hz (< -2)     | -39.5 | -26.2 | -30.5 | -22.0 | -30.4 | -26.1 | -32.0 | -24.6 | -30.9 | -18.2 | -27.6 | -33.1 |
| motion.min_dsi (>= 0.1)         | 0.168 | 0.171 | 0.168 | 0.186 | 0.186 | 0.186 | 0.186 | 0.186 | 0.186 | 0.170 | 0.168 | 0.168 |
| loom_escape.GF_peak_hz (>= 33)  | 59.46 | 47.40 | 53.87 | 60.92 | 62.59 | 59.21 | 53.94 | 54.08 | 49.96 | 50.84 | 48.91 | 51.08 |
| walk_gf.p99_hz (< 38)           | 27.50 | 19.96 | 23.86 | 24.93 | 30.36 | 29.18 | 20.11 | 23.05 | 22.04 | 19.72 | 18.79 | 21.77 |
| rotation.group_flip_hz (<= -3)  | -10.2 | -9.55 | -9.67 | -9.46 | -9.06 | -9.12 | -9.84 | -10.9 | -10.1 | -10.1 | -8.64 | -9.20 |
| object.LC10a_flip_hz (gap)      | 0.00  | -0.00 | -0.01 | -0.00 | 0.01  | 0.01  | 0.01  | 0.00  | -0.00 | -0.01 | -0.00 | 0.01  |
| wind.DNp18_flip_hz (>= 15)      | 46.53 | 45.06 | 46.65 | 45.53 | 45.08 | 45.00 | 45.49 | 45.38 | 44.66 | 44.83 | 45.57 | 46.39 |
| wind.DNp33_flip_hz (<= -15)     | -49.7 | -49.9 | -49.9 | -50.0 | -49.9 | -50.2 | -50.3 | -49.3 | -49.4 | -49.7 | -50.5 | -49.3 |
| odour.apple_8cm_hz (>= 10)      | 17.27 | 17.50 | 17.41 | 17.46 | 17.33 | 17.42 | 17.53 | 17.43 | 17.43 | 18.70 | 18.70 | 18.66 |
| odour.apple_clean_hz (<= 6)     | 4.56  | 4.24  | 4.45  | 4.35  | 4.39  | 4.56  | 4.54  | 4.34  | 4.50  | 5.54  | 5.66  | 5.19  |
| **pass / fail / known gap**     | 27/0/2 | 27/0/2 | 27/0/2 | 26/1/2 | 26/1/2 | 26/1/2 | 26/1/2 | 26/1/2 | 26/1/2 | 26/1/2 | 26/1/2 | 26/1/2 |

Breaks (in all three replicates, against a baseline that passes in all three): **`walk.power_max_hz` for all three
configurations, and nothing else**. No fixes, no check that differs between a configuration's own replicates, and no
check whose status differs between the baseline replicates. Baseline scatter, pooling these three replicates with the
four suite runs of the same settings from the adoption task (`out/r3_default_{1,2,3}.json`, no flags, and
`out/r3_default_flagged.json`, the same settings passed explicitly; n = 7): `walk.power_max` 46.0955 in 7/7, `walk.GF_max` 8.5189 in 7/7, `smell.PN` 7.8612,
`smell.KC_active` 816, `taste.MN9` 10.93, `bitter.shiu` 139.90 in 7/7 (deterministic sections), `loom.GF_peak`
27.05-31.90, `loom_escape.GF_peak` 46.87-59.46 with 3 escapes in 7/7, `walk_gf.p99` 16.40-27.50, `rotate.DNp20`
-22.0 to -39.5, `rotation.group_flip` -8.82 to -10.19, `motion.min_dsi` 0.1679-0.1726, `odour.apple_clean` 4.24-4.72.

`walk.power_max_hz` is the per-frame maximum of the wing-power motor-neuron mean, bounded by `Flight.takeoff_power_hz`
(50). It is the check the round-3 default itself repaired (with `receptor_model = None`: 73.18 FAIL in
all four round-2 off runs `out/rm2_off{,_r2,_r3}.json` and `out/skeptic2/rm_off_r4.json`, and 60.88 / 79.06 FAIL in
the two older pre-receptor baselines `out/retire/baseline.json` / `out/retire_cluster/baseline.json`; sign + abs:
46.0955), and its margin is 3.9 Hz. The values it takes are bit-stable for baseline / pair_gain_lpi_x1 / no_gf_damping and vary 72.78-79.29 (78.89 in a fourth run) for no_al_ln_override (46.10 / 51.56 / 69.77 / 72.78 / 79.29),
i.e. it reads as two regimes -- a ~46 Hz one and a 60-80 Hz one -- and each of the three ablations alone knocks the
model above the 50 Hz bound -- no_gf_damping and no_al_ln_override into the 60-80 Hz regime the receptor default had just taken it out of, pair_gain_lpi_x1 only to 51.56 (a 1.56 Hz margin on the check section 2 called bistable). The takeoff criterion the body actually
uses, `walk.power_sustained_hz` (50 Hz held for 0.3 s), stays well below its bound in every run (21.4 baseline;
24.8 / 37.0-37.5 / 28.6-30.4), so no run turns walking into flight; what breaks is the margin, not the behaviour.

### Per measure: can it be retired now?

**No, none of the three, on this evidence.** Each fails the one-at-a-time criterion by itself, so no combination was
tried; what each run does establish:

* **LPi34 / LPi43 -> LPLC2 x4** (`pair_gain_lpi_x1`) -- **cannot be retired**: breaks `walk.power_max` (51.56 in 3/3,
  deterministic) and confirms the mechanism it was added for in session 9: without it the walking giant-fibre maximum
  rises 8.52 -> 12.26 Hz (deterministic, still under the 38 Hz bound) and the 99th percentile of walking GF maxima
  moves from 16.4-27.5 to 24.9-30.4, i.e. the fly's own turning drives the GF harder. Its cost is also visible: the
  pinned-loom GF peak 27.05-28.04 -> 38.30-38.50 Hz, which is the whole of the -10 Hz `loom.GF_peak` cost the round-3
  default is on record for (off 36.74-43.88 in the four round-2 runs) -- weakening the LPi inhibition undoes that
  cost, which locates it in the T4/T5 <-> LPi <-> LPLC2 balance and not at the giant fibre. The receptor data cover
  this connection completely (100 % of LPLC2's input matched) and confirm its inhibitory sign without licensing any strength, so the honest replacement is not
  "delete the gain" but a sign-correct LPi -> LPLC2 strength derived from something other than a hand-set factor; the
  cheapest next test is a scan of that factor (1, 2, 3, 4) on `walk.power_max` / `walk.GF_max` / `loom.GF_peak`.
* **GF input damping x0.3** (`no_gf_damping`) -- **cannot be retired**, and the argument that recommended retiring it
  above no longer holds. Before the receptor model, removing it drove the walking GF maximum 8.51 -> 1.11 Hz in both
  pre-receptor replicates (`out/retire_cluster/no_gf_damping.json`, `out/retire/no_gf_damping.json`, session-8 code
  and cache): with four of the five damped inputs inhibitory, damping them was removing inhibition and working
  against its own purpose. Under the data-driven signs the same ablation moves the
  walking GF maximum the other way, 8.52 -> 9.22 Hz (deterministic in 3/3), lifts `loom.GF_peak` 27.05-28.04 ->
  29.93-33.74 and leaves `loom_escape` inside the baseline spread (49.96-54.08 vs 46.87-59.46, 3 escapes everywhere) --
  but breaks `walk.power_max` at 69.77 (3/3, deterministic) and nearly doubles `walk.power_sustained` 21.4 -> 37.0-37.5.
  The five edges themselves are untouched by the receptor table (0 of DNp01's 1,455 input edges matched), so what
  changed is the drive arriving at the GF, not the damped synapses (the receptor default alone already moves this
  check: `walk.GF_max` 4.63 under off in all four round-2 runs vs 8.5189 under sign + abs in 7/7). The measure still
  has no physiological referent; the next thing to run is the replacement already in the script, `gf_damping_dnp70` (damp only the one cholinergic
  input of the five), which was suite-neutral before the receptor default and was not part of this batch.
* **Unknown-NT antennal-lobe LN -> GABA** (`no_al_ln_override`) -- **cannot be retired**: breaks `walk.power_max`
  (72.78 / 79.29 / 79.29) and moves the olfactory numbers in exactly the runaway direction it was added against,
  though not yet past their bounds: Kenyon cells above 1 Hz 816 -> 1805 (deterministic; the off default gives 1426),
  the LH apple channel at the plume-free spot 4.24-4.56 -> 5.19-5.66 against a bound of 6 (margin 1.5 -> 0.4 Hz) and
  at 8 cm 17.3-17.5 -> 18.66-18.70, while the PN mean falls 7.86 -> 5.04 Hz and MDN's top rate 153 -> 150. It is the
  cheapest of the three to justify keeping: 27 cells and 0.112 % of |W|, whose GABA label the receptor table is
  consistent with (10,093 of their 13,287 edges matched, 0 sign changes) even though it cannot supply the label
  itself. The data-driven replacement is a transmitter prediction for those 27 cells (the type-majority rule for the
  ~496 remaining unknown presynaptic cells, still undecided in docs/audits/nt_audit.md), not the removal of the rule.

Method note for whoever repeats this: `python scripts/retire_measures.py --check --configs <name>` prints the
effective LIFParams, path / type / pair gains and connectome statistics of a configuration without running anything
(CPU), which is how the three configurations above were verified to change what they claim and nothing else.

### Corrections (round-3 verification, `verify:exp:retire`)

* The baseline spreads quoted for walk_gf.p99 (16.4-27.5) and loom_escape (46.9-59.5) are pooled over seven runs (three retire baselines + the four adopt-task runs); within retire_r3 alone they are 20.0-27.5 and 47.4-59.5. The skeptic's rerun extends no_al_ln_override's loom.GF_peak to 32.2-35.2 (PASS).
* The three "replicates" do not vary the seed for the deciding check: `sec_walk` draws no RNG (--seeds 4,5,6 reproduces --seeds 0,1,2 bit for bit), so they replicate GPU nondeterminism only.
* Every retire_r3 JSON carries `config.receptor_flags = {model: off, net_rule: class}` (the CLI values) next to the authoritative `config.lif` = sign / abs; and `out/r3_default_*.json` recorded `receptor.model None` although sign/abs was active (benchmark.py defect, fixed: the header now records the LIFParams used).
* **All three round-3 verdicts rest on `walk.power_max_hz` measured in the half-applied `sec_walk` (optic lobe without the receptor lookup, see receptor_integration.md round-3 corrections); they are to be re-derived from the round-4 baseline after the fix.** `report()` no longer prints "fixes" when no baseline is present.

## Round 4: with the corrected benchmark (all three round-3 verdicts retracted)

Round 3's `walk`, `motion` and legacy-`loom` numbers were produced with the optic lobe built *without* the receptor
lookup (`scripts/benchmark.py` `sec_walk` / `sec_motion`; docs/audits/receptor_verification.md round-4 follow-up 1).
The fix is in, `--receptor-model` now distinguishes `default` / `off` / model name, and the whole study was re-run.
**The deciding check moved so far that every round-3 verdict in section "Per measure: can it be retired now?" is
withdrawn: the baseline itself now fails `walk.power_max_hz`** (79.4650 against a bound of 50, bit-identical in 3 of 3
replicates), and the three ablations that round 3 said "break" that check are the configurations that repair it.

**The batch.** One cluster batch, 21 jobs (7 configurations x 3 replicates), `--seeds 0,1,2`, full suite, native
backend, NVIDIA B200, shared override cache (sum|W| 121,460,584), run dir `$CLUSTER_RUNS/r4-retire-4620b3`,
12.3 min wall, 0 failed, 6.4-10.9 min per run. Each job:

    python scripts/retire_measures.py --configs <cfg> --seeds 0,1,2 --receptor-model sign --receptor-net-rule abs \
        --timeout 45 --out out/retire_r4/<cfg>_r<N>

`--receptor-model sign --receptor-net-rule abs` is passed **explicitly** because the flag's meaning changed: round 3's
`off` meant "leave LIFParams alone", it now means `receptor_model = None`. The explicit pair is byte-equal to the
shipped defaults -- every field of `brain.LIFParams()` is identical after `benchmark.Context._apply_receptor`
(checked in-process; `config.lif` in all 21 JSONs reads `sign` / `abs` / table `None` / fallback `False`, and
`config.device` reads NVIDIA B200 in all 21).

Results `out/retire_r4/<config>_r<N>/<config>.json` (+ `.log`, + the job's `.txt`), the configuration x replicate x
check matrix in `out/retire_r4/replicates.md` / `.json`
(`python scripts/retire_measures.py --report-replicates out/retire_r4`), a key-value digest in
`out/r4_retire_summary.json`, the effective parameters of all seven configurations in `out/r4_retire_structure.txt`
(`python scripts/retire_measures.py --check --configs ... --receptor-model sign --receptor-net-rule abs`, CPU, no
simulation). Two configurations are new: `pair_gain_lpi_x2` / `pair_gain_lpi_x3` (the LPi -> LPLC2 factor at 2.0 /
3.0; `retire_measures.pair_gain_lpi(factor)` is now a factory and `pair_gain_lpi_x1` is `pair_gain_lpi(1.0)`).

**The baseline is the shipped default and it is independently confirmed.** The three `baseline` runs here reproduce
the re-score task's three no-flag suite runs exactly (`out/r4_rescore_fetch/r4_default_{1,2,3}.json`, run dir
`$CLUSTER_RUNS/r4-rescore-680005`, `config.receptor` = sign / abs / flag `default` / 48,295 entries):
`walk.power_max` 79.4650, `walk.GF_max` 4.6061, `walk.power_sustained` 37.8895, `loom.GF_peak` 50.3826 / 50.8586,
`motion.min_dsi` 0.2334, `taste.MN9` 10.9342, `smell.KC_active` 816, `bitter.shiu_sugar` 139.8985,
**26 PASS / 1 FAIL / 2 KNOWN GAP** in 6 of 6 runs. The three round-3 numbers the caveat named are therefore not what
round 3 recorded:

| number | round 3 (half-applied `sec_walk` / `sec_motion`) | round 4 (corrected) | `off` |
|---|---|---|---|
| `walk.power_max_hz` (< 50)  | 46.0955 PASS in 7/7 | **79.4650 FAIL in 6/6** | 73.1827 FAIL in 7/7 |
| `walk.GF_max_hz` (< 38)     | 8.5189 in 7/7       | 4.6061 in 6/6       | 4.6287 in 3/3 |
| `loom.GF_peak_hz` (>= 20)   | 27.05-31.90         | 50.38-50.86         | 38.16-40.83 (r4), 36.74-43.88 (r2) |
| `motion.min_dsi` (>= 0.1)   | 0.1679-0.1726       | 0.2334              | 0.1679-0.1717 |

(`off` = `out/r4_rescore_fetch/r4_off_{1,2,3}.json`, `--receptor-model off`, which records `model: null` and
reproduces the round-2 off values: taste 5.8455, KC 1426, Shiu 123.5394; 26/1/2 twice and 24/3/2 once, the two
extra FAILs of `r4_off_1` being `loom_escape.GF_peak` 32.2553 against a bound of 33 and `loom_escape.escapes` 0
against >= 1 -- the demo fly does not escape in that run, where the default escapes in 3 of 3 in all six runs.) So the receptor default does not repair
`walk.power_max` (it is 6.3 Hz *worse* than `off` there) and does not cost 10 Hz of legacy `loom.GF_peak` (it gains
10). Those are the re-score task's conclusions to draw; what matters here is that **`walk.power_max` is the model's
one open FAIL under every fully-applied configuration measured so far** -- 79.47 (default), 73.18 (off), 60.88 /
79.06 (the two pre-receptor baselines `out/retire/baseline.json`, `out/retire_cluster/baseline.json`) -- and that the
check's own note says its reference is 22 Hz at DN -> VNC x3 and 50 at x6, i.e. the bound is itself a hand-set number.

### The suite, configuration x replicate (full table: `out/retire_r4/replicates.md`)

Bit-identical in all 21 runs: `rest.spikes_per_step` 0.00, `taste.MN9_hz` 10.9342, `dn.DNa02_L_leg_asym_hz` 2.58,
`dn.DNp09_top_hz` 152, `loom.escape_cm` 3.50, `motion.correct_directions` 8, `loom_escape.escapes` 3,
`bitter.calibrated_sugar_MN9_hz` 5.52, `bitter.calibrated_sugar_bitter_MN9_hz` 0.00, `bitter.shiu_sugar_MN9_hz`
139.8985, `bitter.shiu_sugar_bitter_MN9_hz` 0.82, `compass.wedge_cells_persisting` 0 (KNOWN GAP). Every `walk.*` and
`smell.*` value is bit-identical across a configuration's own three replicates (`sec_walk` and `sec_motion` draw no
RNG), so the three replicates test GPU nondeterminism only, as in round 3.

| check | baseline x3 | no_gf_damping x3 | gf_damping_dnp70 x3 | lpi_x1 x3 | lpi_x2 x3 | lpi_x3 x3 | no_al_ln_override x3 |
|---|---|---|---|---|---|---|---|
| **walk.power_max_hz (< 50)** | **79.4650 FAIL** | 48.4805 | 48.4805 | **51.5078 FAIL** | 48.1205 | **60.4216 FAIL** | **60.8826 FAIL** |
| walk.GF_max_hz (< 38)        | 4.6061 | 4.6292 | 4.6292 | 9.7681 | 9.0290 | 8.5064 | 8.7516 |
| walk.power_sustained_hz (< 50) | 37.8895 | 20.1091 | 20.1091 | 20.2129 | 27.9224 | 25.1316 | 29.4649 |
| loom.GF_peak_hz (>= 20)      | 50.38 / 50.86 / 50.38 | 47.2162 x3 | 43.82 / 44.04 / 47.26 | 64.84 / 60.19 / 58.06 | 50.2233 x3 | 46.59 / 46.26 / 46.26 | 55.50 / 50.78 / 50.78 |
| loom_escape.GF_peak_hz (>= 33) | 55.82-58.43 | 47.39-49.61 | 46.85-52.62 | 58.68-64.35 | 52.68-65.27 | 50.98-60.34 | 47.81-56.34 |
| walk_gf.p99_hz (< 38)        | 19.53-23.49 | 20.22-25.02 | 15.67-27.02 | 26.98-31.14 | 23.63-31.09 | 20.20-23.92 | 19.67-24.58 |
| motion.min_dsi (>= 0.1)      | 0.2334 | 0.2411 | 0.2395-0.2460 | 0.2481-0.2494 | 0.2437 | 0.2398-0.2400 | 0.2397 |
| smell.PN_hz (< 100)          | 7.86 | 7.86 | 7.86 | 7.86 | 7.86 | 7.86 | 5.04 |
| smell.KC_active (> 0)        | 816 | 816 | 816 | 816 | 816 | 816 | 1805 |
| dn.MDN_top_hz (< 250)        | 153 | 153 | 153 | 153 | 153 | 153 | 150 |
| odour.apple_clean_hz (<= 6)  | 4.34-4.47 | 4.33-4.50 | 4.56-4.63 | 4.45-4.55 | 4.48-4.56 | 4.33-4.59 | 5.18-5.49 |
| odour.apple_8cm_hz (>= 10)   | 17.35-17.55 | 17.35-17.55 | 17.36-17.50 | 17.48-17.57 | 17.31-17.43 | 17.41-17.45 | 18.57-18.78 |
| rotate.DNp20_flip_hz (< -2)  | -30.2 to -40.0 | -32.5 to -35.4 | -29.1 to -36.4 | -22.9 to -40.5 | -30.0 to -38.0 | -32.3 to -40.9 | -34.7 to -40.3 |
| **pass / fail / known gap**  | 26/1/2 x3 | **27/0/2 x3** | **27/0/2 x3** | 26/1/2 x3 | **27/0/2 x3** | 26/1/2 x3 | 26/1/2 x3 |

Applying the rule unchanged -- *a check counts as broken only when it fails in every replicate of the configuration
while passing in every baseline replicate* -- **nothing breaks anything**: `breaks_in_all` is empty for all six
non-baseline configurations, no check differs in status between the three baseline replicates, and no configuration
differs in status between its own replicates. `walk.power_max_hz` cannot count as broken by anyone now, because the
baseline fails it; three configurations (`no_gf_damping`, `gf_damping_dnp70`, `pair_gain_lpi_x2`) *fix* it in 3/3.

### The LPi34 / LPi43 -> LPLC2 factor scan (x1 / x2 / x3 / x4)

Structure (`out/r4_lpi_scan_structure.json`, local CPU on the same cache; `c.W` is `W[post, pre]`): 1,231 edges,
7,543 |W| from 180 LPi34/LPi43 cells onto 185 LPLC2 cells; T4/T5 -> LPLC2 is 33,409 edges / 112,941 |W|; LPLC2's
whole input is 102,690 edges / 349,724 |W|. All round-3 structural numbers reproduce. With the gains in force the LPi
share of LPLC2's shaped input is 2.16 / 4.22 / 6.20 / 8.10 % at x1 / x2 / x3 / x4 and the LPi : (T4/T5 x2) ratio is
0.033 / 0.067 / 0.100 / 0.134.

The scan, all values bit-identical across each configuration's three replicates:

| factor | walk.power_max (< 50) | walk.GF_max (< 38) | walk_gf.p99 (< 38) | loom.GF_peak (>= 20) | pass/fail/gap |
|---|---|---|---|---|---|
| x1 | 51.5078 FAIL | 9.7681 | 26.98-31.14 | 58.06-64.84 | 26/1/2 |
| x2 | **48.1205 PASS** | 9.0290 | 23.63-31.09 | 50.2233 | **27/0/2** |
| x3 | 60.4216 FAIL | 8.5064 | 20.20-23.92 | 46.26-46.59 | 26/1/2 |
| x4 (shipped) | 79.4650 FAIL | 4.6061 | 19.53-23.49 | 50.38-50.86 | 26/1/2 |

**Where the factor crosses the bound: twice, and not monotonically.** `walk.power_max` is 51.51 / 48.12 / 60.42 /
79.47 -- it crosses 50 downward between x1 and x2 and upward again between x2 and x3, and the shipped x4 is the worst
point of the scan by 19 Hz. The only monotone quantities are the ones the gain was added for in session 9:
`walk.GF_max` falls 9.77 -> 9.03 -> 8.51 -> 4.61 and `walk_gf.p99` falls with it, i.e. a stronger LPi -> LPLC2
inhibition does suppress the giant-fibre drive the fly's own turning produces -- but every point of the scan is far
under that check's 38 Hz bound, so the mechanism is real and the *size* of the hand-set factor buys nothing that is
scored. `loom.GF_peak` is also non-monotone (58-65 / 50.22 / 46.3-46.6 / 50.4-50.9) and passes everywhere (bound 20),
so round 3's "the x4 costs the loom peak" no longer holds either: the fully-applied default's legacy loom peak is
50 Hz at x4, higher than at x2 or x3.

### The GF input damping: DNp70 alone reproduces the ablation, not the baseline

Structure (`out/r4_gf_damping_structure.json`, reproducing round 3): the five damped inputs are 21 edges / 4,315 |W|
of DNp01's 36,589 |W| input -- SAD073 1,177 GABA, DNp70 1,416 ACh, PVLP010 711 glutamate, CL367 537 GABA, GNG300 474
GABA; the one cholinergic input is 32.8 % of the damped weight. `gf_damping_dnp70` therefore restores full strength to
2,899 |W| of *inhibition* onto the giant fibre and keeps the x0.3 on the only excitation.

The answer to "does damping only DNp70 reproduce the baseline?" is **no -- it reproduces the ablation**:
`gf_damping_dnp70` and `no_gf_damping` give bit-identical `walk.power_max` 48.4805, `walk.GF_max` 4.6292 and
`walk.power_sustained` 20.1091, and both score 27/0/2 in 3/3. Damping the four inhibitory inputs is the whole of the
baseline's `walk.power_max` 79.47; damping the cholinergic one is dynamically inert in `sec_walk`. The two differ only
inside the scatter elsewhere (`loom.GF_peak` 47.2162 x3 vs 43.82 / 44.04 / 47.26; `loom_escape.GF_peak` 47.4-49.6 vs
46.9-52.6; `odour.apple_clean` 4.33-4.50 vs 4.56-4.63).

### Per measure: can it be retired now?

The rule is unchanged: **retire nothing unless a replacement passes in 3/3 with a status margin.** Three
configurations clear 3/3 at 27/0/2; their tightest margin is `walk.power_max` itself, +1.52 Hz (`no_gf_damping`,
`gf_damping_dnp70`) and +1.88 Hz (`pair_gain_lpi_x2`) on a 50 Hz bound -- 3.0 % and 3.8 %, on a check whose value is
bit-identical across replicates (zero scatter) but which section 2 above has already described as two-regime.

* **GF input damping x0.3 (five inputs, `LIFParams.type_path_gain`)** -- **retire it.** Its plain ablation
  `no_gf_damping` scores 27/0/2 in 3/3, breaks no check, and turns the default's only FAIL into a PASS
  (`walk.power_max` 79.4650 -> 48.4805, `walk.power_sustained` 37.89 -> 20.11). Round 3's verdict ("cannot be retired,
  breaks walk.power_max at 69.77") was produced by the half-applied `sec_walk` and is withdrawn; the session-9
  recommendation "GF x0.3 goes", retracted in round 3, is reinstated on better evidence. The measure still has no
  physiological referent, and four of the five inputs it damps are inhibitory under both NT_SIGN and the receptor
  table (the receptor lookup matches 0 of DNp01's 1,455 input edges, so it has nothing to say about them). No
  replacement is needed: `gf_damping_dnp70` gives the identical walk numbers, so keeping a damping term on the one
  cholinergic input buys nothing. Costs on record: `loom.GF_peak` 50.4-50.9 -> 47.2 and `loom_escape.GF_peak`
  55.8-58.4 -> 47.4-49.6, both far inside their bounds (20 / 33) with 3 escapes in 3/3. Adopt it alone, with a suite
  run, as section 2 prescribes -- and not in the same change as anything below.
* **LPi34 / LPi43 -> LPLC2 x4 (`optic.DEFAULT_PAIR_GAIN`)** -- **cannot be retired, but x4 is not the value the suite
  supports.** Removing it (`pair_gain_lpi_x1`) is 26/1/2 in 3/3 with `walk.power_max` 51.5078, so the ablation does
  not pass and the measure stays. What the scan adds is that the shipped x4 is the worst of the four factors on that
  check (79.47) and **x2 is the only point that passes the whole suite (27/0/2 in 3/3, margin 1.88 Hz)**, at the price
  of the walking giant-fibre drive the gain exists to suppress (`walk.GF_max` 4.61 -> 9.03, `walk_gf.p99` 19.5-23.5 ->
  23.6-31.1, both still under 38). That is a re-parameterisation of a hand-set factor, not a retirement, and the
  receptor data cannot license it: they match 100 % of LPLC2's input and confirm the LPi glutamate as GluCl -1 without
  saying anything about strength (round-3 section above). It should not be adopted on `walk.power_max` alone while
  that check fails in the default -- a non-monotone check with a 19 Hz swing across the scan is describing a regime
  boundary, not a dose-response.
* **Unknown-NT antennal-lobe LN -> GABA (`connectome.UNKNOWN_NT_OVERRIDE_REGEX`)** -- **cannot be retired**, verdict
  unchanged but for a different reason. It is 26/1/2 in 3/3, so it neither breaks nor fixes anything by the rule
  (its `walk.power_max` 60.8826 is better than the baseline's 79.47 and still over the bound), and it moves the
  olfactory numbers in the runaway direction it was added against: Kenyon cells above 1 Hz 816 -> 1805, PN mean
  7.86 -> 5.04, the LH apple channel at the plume-free spot 4.34-4.47 -> 5.18-5.49 against a bound of 6 (margin
  1.6 -> 0.5 Hz) and at 8 cm 17.35-17.55 -> 18.57-18.78, MDN top 153 -> 150. The structure reproduces exactly
  (recomputed locally with `connectome.UNKNOWN_NT_OVERRIDE_REGEX = {}` and `compile_connectome`): 27 cells in the same
  13 types, 13,287 output edges, 136,223 |W|, by target superclass cb_intrinsic 106,111 / cb_sensory 29,213 /
  descending 858 / ascending 16 / visual_projection 13, ORN_ targets 7,124 edges 26,097 |W|, sum|W| 121,460,584 ->
  121,324,368, unknown-NT cells 2,361 -> 2,388. The round-3 "PN-named 42,005" that could not be reproduced is
  45,222 |W| over 2,662 edges under a prefix/suffix name rule, inside the 37,000-49,000 range that correction quoted.
  The data-driven replacement is still a transmitter prediction for those 27 cells, not the removal of the rule.

**What this does not settle.** (1) Every "fix" above is a fix of a check the shipped default fails, so adopting one
changes a default while the receptor default is itself under review for the same check; the order matters, and
`walk.power_max`'s 50 Hz bound (its own note: 22 Hz at DN -> VNC x3, 50 at x6) deserves a look before anything is
tuned to sit 1.5 Hz under it. (2) Combinations were not run -- the rule is one at a time, and `no_gf_damping` +
`pair_gain_lpi_x2` is untested. (3) The three replicates still do not vary the seed for the deciding check: `sec_walk`
draws no RNG, so 3/3 here means "three GPU runs", not three independent samples; the scatter quoted for
`walk.power_max` is exactly zero and must not be read as a confidence interval. (4) `pair_gain_lpi_x4` is the shipped
baseline and was not run as a separate configuration; the x4 column of the scan table is the `baseline` column.

### Corrections (round-4 verification, `verify:exp:retire`)

* The baseline's walk section is not bit-reproducible: an 8th full-suite baseline with a byte-identical config gave
  walk.power_max 80.0928 (+0.63 Hz) and power_sustained 39.9078 (+2.02); motion.min_dsi takes three values across
  the six baseline draws and differs between replicates in all seven configurations; loom.GF_peak spans 46.80-54.27
  over 8 + 6 draws (not 0.48 Hz). The GF-damping retirement still clears its criterion -- no_gf_damping 48.4805 in
  10/10 draws across two batches -- with a +1.52 Hz margin against a largest observed baseline excursion of 0.63 Hz.
* The tightest PASS margins among the passing configurations are odour.apple_channel_clean (1.37-1.67 Hz) and the
  calibrated sugar+bitter check (1.00 Hz), not only walk.power_max; walk.power_max is the tightest for LPi x2 only.
* "LPi share of LPLC2's shaped input 2.16 / 4.22 / 6.20 / 8.10 %" applies the LPi gain without the T4/T5 x2 pair gain
  in force in the same lobe; with every gain applied the shares are 1.63 / 3.21 / 4.74 / 6.22 %.
* "PN-named 45,222 |W| over 2,662 edges" is not reproducible under any of twelve name rules (type endswith PN 37,349;
  substring 48,982; class ALPN 49,483; ...) -- drop it; the generator now shipped is
  `scripts/skeptic_retire_r4_structure.py` (all other structural numbers reproduce from it).
* walk_gf.p99 is not monotone across the LPi scan (replicate spread 4-11 Hz); only walk.GF_max is. loom.GF_peak
  separation default vs off is 3-17 Hz over the pooled draws, not "~11 Hz".
* `retire_measures.py`'s docstring described the old "off leaves LIFParams alone" semantics and its argparse default
  `off` silently scored the previous model after the round-4 benchmark fix: now `default` (LIFParams' own) with `off`
  = None explicitly.
* The three round-4 "replicates" replicate GPU nondeterminism only for walk.* (`sec_walk` draws no RNG).

## Round 5: GF damping adoption

Round 4's verdict on the GF input damping ("retire it ... adopt it alone, with a suite run, as section 2 prescribes -- and
not in the same change as anything below") is executed here: the `(SAD073|GNG300|DNp70|CL367|PVLP010 -> DNp01, 0.3)` entry
is removed from `brain.DEFAULT_TYPE_PATH_GAIN`, and the edited default is scored on the suite (x3) and on the room
take-off protocol (3 brain RNGs x 16 flies x 5 min, live escape route) against the pre-retirement default and off.
**Decision: the retirement is kept.** Both halves of the criterion hold -- no check worse in status than the
pre-retirement default in 3/3 and `walk.power_max` PASS in 3/3 (27 / 0 / 2 x3, 48.48 Hz against the 50 Hz bound) -- and
the take-off rates are not worse in either route (escape 1.67 vs 2.15, voluntary 2.22 vs 3.06 per 1,000 fly-s; the
one-sided p that the edited default hops *more* is 0.77 / 0.95). The excess over off (measured under the damped gains) shrinks by a point estimate inside rerun scatter, 27-32 % but does not go
away: 56 vs 75 hops in 14,400 fly-s (two-sided p 0.087), still 6.2x off's 9 (p 1.7e-07). Nothing else in `flyverse/`
changed; `pair_gain_lpi_x2` and the AL LN override were not touched (the rule is one measure at a time).

### What changed in the code (`flyverse/brain.py` lines 221-233; nothing else in `flyverse/`)

`DEFAULT_TYPE_PATH_GAIN = [(r"^(LC4|LPLC2)$", r"^DNp01$", 3.0)]` -- the loom gain alone. The previous two-entry list is kept
as `brain.GF_DAMPED_TYPE_PATH_GAIN`, so `LIFParams(type_path_gain=brain.GF_DAMPED_TYPE_PATH_GAIN)` is the round-3 / round-4
default, and `scripts/retire_measures.py`'s configurations keep a meaning relative to the new default: `baseline` and
`no_gf_damping` (`[LOOM_GF]`, line 244) are now the same weights as the shipped default, `gf_damping_dnp70` (line 245) is
the default plus a x0.3 on DNp70 -> DNp01 alone, and the pre-retirement default is reachable only through
`GF_DAMPED_TYPE_PATH_GAIN` (`retire_measures.py` was not edited; its `GF_DAMP` constant, line 61, still names the entry, and
a "restore the damping" configuration would be `{"type_path_gain": brain.GF_DAMPED_TYPE_PATH_GAIN}`). `scripts/cx_wedge.py`,
`cx_wedge_verify.py`, `benchmark.py` (config header) and `retire_measures.py` read `brain.DEFAULT_TYPE_PATH_GAIN` by name
and follow the change; the compass scripts extend the list, so their ring gains are unaffected.

Structure on the local adopted cache (`scripts/r5_adopt_structure.py` -> `out/r5_adopt_structure.json`, `.log`; CPU):

| quantity | value |
|---|---|
| entries that differ, new default vs previous default | 21, all onto DNp01, previous / new ratio 0.3 exactly (float32 0.30000001) |
| by presynaptic type (edges; shaped \|W\| new / previous, i.e. after the 60-synapse cap) | SAD073 gaba 8; 480 / 144 -- CL367 gaba 4; 240 / 72 -- DNp70 acetylcholine 4; 240 / 72 -- GNG300 gaba 3; 123 / 36.9 -- PVLP010 glutamate 2; 120 / 36 |
| shaped \|W\| restored onto DNp01 | 1,203 - 361 = 842, of which inhibitory 963 - 289 = 674 (the round-4 "4,315 \|W\| / 2,899 inhibitory" are the raw pre-cap synapse counts; most of the 21 edges sit at the 60-synapse cap, so the shaped change is smaller than the raw one) |
| DNp01 shaped input, new / previous | 1,455 edges both; \|W\| 83,046 / 82,204; inhibitory \|W\| 9,289 (11.19 %) / 8,615 (10.48 %) |
| md5 of `_shaped_weights` (sorted CSR data + indices + indptr) | new default `0e30e4a80cb607d4a168d1b08ebd6a40`; `type_path_gain=GF_DAMPED_TYPE_PATH_GAIN` `f0d145d1bb81b446ebc51f89ded7bd4b` = the round-3 pinned hash of the previous default, byte for byte; `receptor_model=None` `fcb5bec2a6c492196a622e31cdb24fc6` (new gains) / `2e276b30b6117c1f62688b01775eda6b` (previous gains = the pre-round-3 pin) |
| the receptor lookup's part | none: it matches 0 of DNp01's 1,455 input edges (round 4), so `sign` vs `None` differs by the same 48,295 entries under either list, and the 21 entries differ identically under `None` |

Tests (`tests/test_receptor_model.py`, class `CachedConnectomeTests`): `test_default_type_gains_have_no_gf_damping` (no
entry onto DNp01 with a factor below 1, none of the five types matches any entry, the two lists are exactly the ones above)
and `test_gf_damped_type_gains_reproduce_the_previous_weights` (the two defaults differ on exactly 21 entries, all onto
DNp01 from those five types, by the factor 0.3; on the adopted cache the previous list reproduces `f0d145d1...` and, with
`receptor_model=None`, `2e276b30...`); the round-3 hash test now pins the new default's pair (`0e30e4a8...` /
`fcb5bec2...`) and the same 48,295 / 30,916 / 17,379 sign-vs-None counts. `python -m pytest tests/ -q` on the CPU
(`CUDA_VISIBLE_DEVICES=""`): 80 passed, 29 skipped, 200 subtests (40.9 s).

### The batch (`scripts/r5_adopt_batch.sh`; run dir `$CLUSTER_RUNS/r5-adopt-fb3608`)

One cluster call, 7 jobs, 0 failed, 49.0 min wall (`out/r5_adopt_cluster.log`), submitted 2026-09-12 10:33 UTC while the
instrument task's 14-job batch `r5-hops-218d81` (submitted 10:25) was running on the same node. `cluster_run.py` ships the
working-tree diff, so the jobs ran the edited `brain.py` (line 221 of the run directory's copy read over ssh before the jobs
started) with the round-5 instrument files. Headers read back from every job: the suite jobs `receptor model sign (abs;
flag --receptor-model default); fast sign changed on 48,295 of 25,578,600 entries`, NVIDIA B200; the room jobs
`BatchSim B=16 neurons=167,106 device=cuda`, `escape at GF >= 33 Hz`. Jobs 0-2: `benchmark.py --seeds 0,1,2` (no flags) ->
`out/r5_adopt_default_{1,2,3}.json` (418 / 418 / 430 s; `config.type_path_gain` = the one-entry list). Jobs 3-5:
`batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels
--event-driven --cuda-sparse torch --seed k --seeds <16k..16k+15>`, k = 0, 1, 2 -> `out/r5_adopt_sustain_live_{1,2,3}.json`
(2,890 / 2,901 / 2,898 s). Job 6: `benchmark.py --sections hops` -> `out/r5_adopt_hops.json` (1,759 s). The comparison arms
were not re-run: the pre-retirement room arm is the instrument batch's `out/r5_sustain_default_live_{1,2,3}.json` (the
identical command and seeds under the previous list) and its off arm `out/r5_sustain_off_live_{1,2,3}.json`; the
pre-retirement suite runs are round 4's `out/r4_default_{1,2,3}.json`; the `hops` section of the previous default / off is
`out/r5_hops_default.json` / `out/r5_hops_off.json`. Tables: `scripts/r5_adopt_report.py` -> `out/r5_adopt_report.log`,
`.json` (CPU; the suite half alone in `out/r5_adopt_report_suite.log`).

### The suite: edited default x3 vs the pre-retirement default x3 vs off x3

| check | criterion | edited default x3 (`r5_adopt_default`) | pre-retirement default x3 (`r4_default`) | off x3 (`r4_off`) | r4 `no_gf_damping` x3 |
|---|---|---|---|---|---|
| **walk.power_max_hz** | < 50 | **48.48 P / 48.48 P / 48.48 P** | 79.47 F x3 | 73.18 F x3 | 48.48 P x3 |
| walk.power_sustained_hz | < 50 | 20.11 x3 | 37.89 x3 | 31.44 / 31.58 / 31.44 | 20.11 x3 |
| walk.GF_max_hz | < 38 | 4.63 x3 | 4.61 x3 | 4.63 x3 | 4.63 x3 |
| walk_gf.p99_hz | < 38 | 15.81 / 20.50 / 19.76 | 22.28 / 18.31 / 20.35 | 22.52 / 23.59 / 17.81 | 24.79 / 20.22 / 25.02 |
| loom.GF_peak_hz | >= 20 | 47.22 x3 | 50.38 / 50.38 / 50.86 | 40.13 / 38.16 / 40.83 | 47.22 x3 |
| loom_escape.GF_peak_hz | >= 33 | 48.01 / 59.70 / 45.42 | 49.39 / 49.92 / 52.95 | 32.26 F / 34.47 / 33.95 | 47.39 / 49.61 / 48.85 |
| loom_escape.escapes | >= 1 | 3 x3 | 3 x3 | 0 F / 1 / 1 | 3 x3 |
| motion.min_dsi | >= 0.1 | 0.246 / 0.246 / 0.241 | 0.233 x3 | 0.168-0.172 | 0.241 x3 |
| rotate.DNp20_flip_hz | < -2 | -36.56 / -32.21 / -35.71 | -32.24 / -34.04 / -31.90 | -16.89 / -32.09 / -27.55 | -35.38 / -32.49 / -34.91 |
| rotation.group_flip_hz | <= -3 | -10.17 / -9.46 / -9.25 | -9.61 / -10.18 / -10.39 | -7.15 / -7.08 / -7.06 | -9.75 / -9.19 / -9.89 |
| odour.apple_channel_clean_hz | <= 6 | 4.43 / 4.34 / 4.27 | 4.28 / 4.63 / 4.47 | 4.42 / 4.36 / 4.36 | 4.50 / 4.35 / 4.33 |
| taste.MN9 / smell.PN / smell.KC_active | > 2 / < 100 / > 0 | 10.93 / 7.86 / 816 x3 | same | 5.85 / 11.19 / 1426 | same as edited |
| bitter shiu_sugar / shiu_sugar_bitter / calibrated_sugar | > 50 / < 10 / > 2 | 139.90 / 0.82 / 5.52 x3 | same | 123.54 / 2.12 / 4.57 | same as edited |
| rest, dn.*, loom.escape_cm, motion.correct_directions, wind.*, odour.apple_8cm, calibrated_sugar_bitter | | all PASS, inside the round-4 ranges | | | |
| object.LC10a_flip_hz, compass.wedge_cells_persisting | gap | KNOWN GAP x3 | KNOWN GAP x3 | KNOWN GAP x3 | KNOWN GAP x3 |
| **pass / fail / known gap** | | **27 / 0 / 2 x3** | 26 / 1 / 2 x3 | 24/3/2, 26/1/2, 26/1/2 | 27 / 0 / 2 x3 |

**Criterion (a), no check worse in status than the pre-retirement default in 3/3: holds** -- the list of (check, run)
pairs where an edited-default status ranks below any pre-retirement status is empty; the one status change is
`walk.power_max` FAIL -> PASS in 3/3. **Criterion (b), `walk.power_max` PASS in 3/3: holds**, 48.48052978515625 in all
three -- bit-identical to the ten round-4 `no_gf_damping` draws (14/14 now), against a bound of 50 and a largest observed
baseline walk-section excursion of 0.63 Hz (round-4 corrections). The walk / smell / taste / Shiu values are bit-identical
to the round-4 `no_gf_damping` runs (`walk.GF_max` 4.629162311553955, `power_sustained` 20.109053071339925, taste
10.93417739868164, KC 816, PN 7.861156463623047, Shiu 139.89849777221679): the edited default is the configuration round 4
measured, reached through `brain.py` instead of `retire_measures.py`. Smallest PASS margins (min over the three runs):
`motion.min_dsi` 0.141, `bitter.calibrated_sugar_bitter` 1.00 Hz, `walk.power_max` 1.519 Hz, `odour.apple_channel_clean`
1.571 Hz, `loom_escape.escapes` 2. Value shifts vs the pre-retirement default beyond the replicate scatter: `walk.power_max`
-30.98 Hz, `walk.power_sustained` -17.78 Hz, `loom.GF_peak` 50.4-50.9 -> 47.22 (the round-4 default's own full-suite draws
span 46.80-54.27), `motion.min_dsi` 0.233 -> 0.241-0.246; everything else inside the round-4 ranges (`loom_escape.GF_peak`
45.4-59.7 vs 49.4-52.9; `walk_gf.p99` 15.8-20.5 vs 18.3-22.3).

### Take-offs: edited default vs the pre-retirement default vs off (live escape route, 3 brain RNGs x 16 flies x 300 s)

Per batch (brain seed 0 / 1 / 2, environment seeds 0-15 / 16-31 / 32-47), hops = escape + voluntary; the per-fly vectors
are in `out/r5_adopt_report.log`.

| arm | batch 1 | batch 2 | batch 3 | pooled 48 flies, 14,400 fly-s | per 1,000 fly-s all / escape / voluntary | walking-GF median (rows >= 33 Hz) |
|---|---|---|---|---|---|---|
| edited default (`r5_adopt_sustain_live`) | 19 = 7 + 12 | 21 = 11 + 10 | 16 = 6 + 10 | **56 = 24 + 32** | **3.89 / 1.67 / 2.22** | 31.90 Hz (19/48); per batch 31.15 / 33.32 / 31.22 |
| pre-retirement default (`r5_sustain_default_live`) | 25 = 12 + 13 | 25 = 11 + 14 | 25 = 8 + 17 | 75 = 31 + 44 | 5.21 / 2.15 / 3.06 | 32.25 Hz (22/48); 32.96 / 32.82 / 31.16 |
| off (`r5_sustain_off_live`) | 3 = 3 + 0 | 4 = 4 + 0 | 2 = 2 + 0 | 9 = 9 + 0 | 0.63 / 0.63 / 0.00 | 27.20 Hz (8/48); 27.22 / 26.83 / 27.99 |
| round-4 pre-retirement default (`r4_sustain_default`, no split) | 21 | 26 | 27 | 74 | 5.14 | -- |

Mann-Whitney over flies (asymptotic; counts are tied so no exact p), pooled 48 v 48, with both one-sided alternatives:

| contrast | metric | U | p two-sided | p(edited < other) | p(edited > other) |
|---|---|---|---|---|---|
| edited vs pre-retirement | hops | 926.0 | 0.087 | 0.044 | 0.957 |
| | escape | 1064.0 | 0.47 | 0.23 | 0.77 |
| | voluntary | 942.0 | 0.099 | 0.049 | 0.951 |
| | walking-GF max per row | 934.0 | 0.11 | 0.056 | 0.945 |
| edited vs off | hops | 1786.5 | 1.7e-07 | 1 | 8.3e-08 |
| | escape | 1423.0 | 0.012 | 0.994 | 0.0058 |
| | voluntary | 1656.0 | 3.2e-07 | 1 | 1.6e-07 |
| | walking-GF max per row | 1846.0 | 3.7e-07 | 1 | 1.9e-07 |
| pre-retirement vs off (the instrument task's contrast, for reference) | hops | 1974.0 | 6.6e-11 | -- | -- |

Per batch, edited vs pre-retirement (seed-matched): hops 19 / 21 / 16 vs 25 / 25 / 25 (two-sided p 0.29 / 0.53 / 0.20,
lower in 3 of 3); voluntary 12 / 10 / 10 vs 13 / 14 / 17 (p 0.76 / 0.45 / 0.13, lower in 3 of 3); escape 7 / 11 / 6 vs
12 / 11 / 8 (p 0.43 / 0.90 / 0.53, lower in 2 of 3, equal in 1); walking-GF median 31.15 / 33.32 / 31.22 vs 32.96 / 32.82 /
31.16 (rows at the threshold 6 / 9 / 4 vs 8 / 8 / 6 of 16). Edited vs off per batch: hops p 2.7e-04 / 4.1e-03 / 1.4e-02,
voluntary 12 / 10 / 10 vs 0 / 0 / 0 (p 1.6e-03 / 1.6e-03 / 1.8e-02), escape 7 / 11 / 6 vs 3 / 4 / 2 (p 0.23 / 0.047 / 0.33).
Meals / energy / path are not the question here; for the record, edited default meals 3 / 2 / 3 vs pre-retirement 1 / 2 / 0
vs off 1 / 5 / 4.

**Reading.** (1) The take-off rates are **not worse** than the pre-retirement default's in either route: every point
estimate is lower (all 0.75x, escape 0.77x, voluntary 0.73x) and the one-sided "edited hops more" p is 0.77 (escape) /
0.95 (voluntary) / 0.96 (all). (2) The excess **shrinks but survives**: of the pre-retirement excess over off (4.58 per
1,000 fly-s all; 1.53 escape; 3.06 voluntary) the retirement removes roughly a quarter to a half (point estimates 29 % / 32 % / 27 %, which move to 41 / 50 / 36 % when the shipped default's identical-seed rerun is substituted) -- 56 vs 75 hops, lower in 3 of 3
batches, two-sided p 0.087 (one-sided 0.044); against off the edited default is still 6.2x (56 vs 9, p 1.7e-07), takes off
by the voluntary route 32 times where off does 0, and its walking-GF tail (median 31.9 vs 27.2 Hz, 19/48 vs 8/48 rows at
the 33 Hz escape threshold, p 3.7e-07) is barely moved from the pre-retirement 32.25 Hz / 22/48 (p 0.11). Restoring the
842 shaped |W| of inhibition onto DNp01 therefore lowers the walking wing-power peak by 31 Hz in `sec_walk` but takes
only about a quarter of the room take-off excess with it: the link the round-4 critic wanted tested ("whether restoring
inhibition onto DNp01 lowers the walking-GF tail that fires the 33 Hz escape") is weak -- the GF tail is a property of
the receptor default's drive onto the GF that the five damped inputs do not control, and the voluntary route (wing power
>= 50 Hz held 0.3 s, a different readout from `walk.power_max`'s per-frame MN mean) stays 2.2 per 1,000 fly-s. (3) The
seed-0 reproduction of round 4 holds for the pre-retirement arm (25 hops here vs 21 / 17 / 24 in the three round-4
seed-0 draws; off 3 vs 1 / 0 / 2), so the two batches are comparable; the round-4 room runs give the same picture against
the edited default (56 vs 74, U 969, p 0.17; `out/r5_adopt_vs_r4_sustain.log`).

### The `hops` section on the edited default (one draw each; `out/r5_adopt_hops.json`, `r5_hops_default.json`, `r5_hops_off.json`)

16 flies x 150 s = 2,400 fly-s, brain seed 0, environment seeds 0-15, scored against the bounds in `benchmark.py` at run
time (`Ref(0.0, "<", 1.0, gap)` voluntary, `Ref(0.5, "<", 10.0)` escape, `Ref(28.0, "<", 33.0, gap)` GF median):

| arm | hops | voluntary per 1,000 fly-s | escape per 1,000 fly-s | walking-GF median | section tally |
|---|---|---|---|---|---|
| edited default | 3 = 2 escape + 1 voluntary | 0.42 PASS (gap closed) | 0.83 PASS | 29.27 PASS (gap closed) | 3 / 0 / 0 |
| pre-retirement default | 7 = 4 + 3 | 1.25 KNOWN GAP | 1.67 PASS | 30.11 PASS (gap closed) | 2 / 0 / 1 |
| off | 1 = 1 + 0 | 0.00 PASS (gap closed) | 0.42 PASS | 25.74 PASS (gap closed) | 3 / 0 / 0 |

The edited default's "PASS (gap closed)" on the voluntary entry is a single 2,400 fly-s draw and **must not be read as the
gap closing**: the 300-s room batches above put its voluntary rate at 2.22 per 1,000 fly-s (32 in 14,400), 2.2x the
bound, and a section at that rate expects 5.3 voluntary hops, not 1 (Poisson P(X <= 1) = 0.031). The same shortfall
appears in the pre-retirement draw (7 hops observed against 12.5 expected from its room rate, P(X <= 7) = 0.070;
voluntary 3 vs 7.3, P = 0.066), while off's 1 vs 1.5 is unremarkable -- so either both draws are low by chance (joint
probability ~1e-3) or the take-off rate is not stationary over the 300 s and the section's first 150 s (energy 0.9 ->
~0.1; the flies reach energy 0 at 150-180 s) under-samples the 300-s room rate. That is a question for the section's
owner (the instrument task's R5.4 references are derived from the 300-s off batches); it does not affect the decision here,
which rests on the room batches. Launch times in the three draws: edited 51.6 s (voluntary), 106.9, 121.9 (escape);
pre-retirement 13.4, 74.0, 79.0, 88.6, 102.3, 136.0, 149.1 s; off 125.2 s.

### Per measure: the GF input damping is retired

* **GF input damping x0.3 (five inputs, `LIFParams.type_path_gain`) -- retired; the retirement is kept.** It clears the
  rule of this document (a replacement -- here, nothing -- that passes in 3/3 with a status margin): 27 / 0 / 2 in 3/3
  on the shipped code path, `walk.power_max` 48.4805 in 14/14 draws over three batches (+1.52 Hz against a 50 Hz bound
  and a largest observed baseline excursion of 0.63 Hz), no check worse in status, and the additional condition set for
  this adoption -- the room take-off rates not worse than the pre-retirement default's -- holds in both routes with the
  point estimates a quarter to a half lower, inside rerun scatter. Costs on record, unchanged from round 4: `loom.GF_peak` 50.4-50.9 -> 47.2 and
  `loom_escape.GF_peak` 49-53 -> 45-60 (3 escapes in 3/3; bounds 20 / 33). What the adoption does **not** buy: the room
  take-off excess of the receptor default stays (3.89 vs off's 0.63 per 1,000 fly-s; voluntary 2.22 vs 0), and the
  `hops` section's voluntary entry stays a KNOWN GAP in substance even where a 2,400 fly-s draw passes it.
* The other two verdicts of round 4 stand: LPi x4 cannot be retired (x2 not adopted -- and `no_gf_damping +
  pair_gain_lpi_x2` remains untested, now as "default + pair_gain_lpi_x2"); the AL LN override cannot be retired. The
  `walk.power_max` bound stays at the hand-set 50 Hz (its note: 22 Hz at DN -> VNC x3, 50 at x6); the round-4 critic
  asked for that to be decided before anything is tuned to sit under it -- this adoption is a removal, not a tuning,
  and the check's value did not move between round 4 and here, so the bound question is left where it was.

### Caveats

(1) The pre-retirement and off room arms were run by the instrument task's batch, 8 min earlier on the same node with the
same command and seeds; the only difference between `r5_sustain_default_live_k` and `r5_adopt_sustain_live_k` is the
`brain.py` line (verified in both run directories), but the room JSONs record `receptor` and not `type_path_gain`, so
the arm labels rest on the run directories, not on the files. (2) The batched rollout is not deterministic (round 4:
17-24 hops in identical-seed reruns of the pre-retirement seed-0 batch, 0-2 for off), so the 19-hop difference (56 vs 75)
is ~2-3 rerun scatters wide and its two-sided p is 0.087; "shrinks by about a quarter" is a point estimate with that
uncertainty, and "not worse" is the robust statement. (3) Three room replicates per arm and one `hops` draw per arm; the
`hops` section's stationarity question above is open. (4) The suite's `walk.*` values are bit-reproducible here (3/3 and
14/14 across scripts), but round 4 showed the baseline's walk section can move by 0.63 Hz between draws, so the
`walk.power_max` margin is 1.52 Hz against that, not against zero.

### Corrections (closing-round verification)

* The shipped default's room take-off count at identical seeds scatters 19 -> 11 (7+12 -> 3+8) between two runs of the
  same command (U 182.5, p2 0.029), a larger separation than the retirement contrast itself (p2 0.087); the
  load-bearing statements are "not worse in either route" and "still 6.2x off (voluntary 32 vs 0)".
* The off arm used for the "excess" denominator carries the damping (it predates the retirement); off with the damping
  retired was run on six Brain-only sections only: walk.power_max 95.5-97.1 FAIL 3/3, walk.power_sustained 50.60 FAIL /
  49.18 PASS / 49.61 PASS. Under the shipped gains the receptor model is therefore ~48 Hz better than off on
  walk.power_max, where under the damped gains it was 6.3 Hz worse.
* The round-4 LPi scan (x1 / x2 / x3 / x4) and the AL LN verdict were measured under the damped gains only and must
  be re-scanned under the shipped default before "x2 is the only passing point" is quoted again.
* `retire_measures.py`: `baseline` and `no_gf_damping` are now the same weights; the new `gf_damped` configuration
  restores the damping for future ablations.

## Round 6 (2026-09-13): the `walk.power_max` bound, decided from the data (handover item 4)

**Decision.** `walk.power_max_hz` should be **de-scored** -- kept in the table as a reported number, not as a pass / fail
check -- and its referent should **not** be re-derived from a gain scan. The measured facts behind that: (a) under the
shipped defaults it is 48.4805 Hz in 12 of 12 independent GPU draws (this batch's 6 plus 6 on file), **1.5195 Hz**
under the 50 Hz bound, with zero scatter in the default itself but **more scatter on record for the very arms
section (c) uses than that margin**: `out/sk_les_holds_all/holdOptic_r{0..3}.json` span 11.86 Hz on this check and
this document's own `off` suite arm spans 1.56 Hz, so **the margin is inside the worst single-arm scatter on record**
(not "about two excursions wide", as an earlier form of this sentence said -- see the bit-stability paragraph below);
(b) it is
**non-monotone** in the one three-point scan available (LPi -> LPLC2 x1 / x2 / x4 = 51.51 / 48.00 / 48.48, 6 draws
each, under the shipped gains) while the quantity that gain was added for, `walk.GF_max`, is monotone (9.80 / 9.03 /
4.63); (c) across the four take-off hold arms its rank correlation with the room take-off rate is **negative**
(**Spearman -0.60**, which is the statistic to quote: it stays exactly -0.600 under every substitution tested --
report value, pooled 9 draws, eager mean 63.51, worst draw 56.17 -- while the Pearson moves -0.787 -> -0.664 and its
p 0.213 -> 0.336; n = 4, exact permutation p 0.42 against a floor of 0.083 -- descriptive only, but
the ordering is the opposite of what a spurious-take-off bound should produce); (d) its note's referent ("22 Hz with
DN -> VNC x3") measures 48.48 at that operating point under the shipped model, and the 50 it is scored against is the
body's voluntary-take-off threshold (`Flight.takeoff_power_hz`, 50 Hz **held 0.3 s**) applied to a **per-frame maximum**
-- the sustained form of that criterion is the separate check `walk.power_sustained_hz`, which keeps its referent and
stays scored. The exact line is named in (d) below; **it is not changed here** (the benchmark's owner does).
The drive-clip retirement candidate (`OpticParams.drive_clip_mv` 35) is 10 PASS / 0 FAIL in 6 of 6 draws at a
`walk.power_max` of 49.2480 (0.75 Hz under the bound) with `walk.GF_max` 4.63 -> 13.26; **nothing is adopted** -- the
adopt-alone rule needs its own 29-check suite run and the room take-off protocol, and those were not run.

### The batch (`scripts/pm_bound_batch.sh`; run dir `pmb-f59cd3` on four rented boxes; console `out/pm_bound_cluster.log`)

One `cluster_run.py` call, **36 jobs, 0 failed, 6.3 min wall**: six configurations x brain seeds 0, 1, 2 x two
independent draws, each job its own directory `out/pm_bound/<cfg>_s<seed>_d<k>/<cfg>.json` (+ `<cfg>.log`, the job's
`.txt`), each job

    mkdir -p out/pm_bound && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && \
      PYTHONIOENCODING=utf-8 python scripts/retire_measures.py --configs <cfg> --seeds <s> --sections walk,a,b --timeout 40 --out out/pm_bound/<cfg>_s<s>_d<k>

(`walk` = the legacy walking / loom / rotate section that carries the three `walk.*` checks, `a` = motion, `b` =
`loom_escape` at that one seed; receptor model `default` = LIFParams' own `sign` / `abs`, native backend). The boxes:
vast-a / vast-b (NVIDIA B200, 20 jobs), vast-c / vast-d (NVIDIA H200, 16 jobs); every JSON's `config.device` reads a
GPU, none `cpu`. `scripts/retire_measures.py` was edited for this batch (its owner's import fix was already in):
a `no_drive_clip` configuration (`optic: {"drive_clip_mv": 1e9}`, the same ablation as `scripts/audit_optic.py`'s)
and a `provenance` block in every JSON (`flyverse.interp.common.provenance`: the resolved LIFParams / OpticParams,
the compiled-connectome fingerprint, source fingerprint, execution record). Identity of the runs: compiled `W`
md5 `ef23cc27...` (sum|W| 121,460,584, 25,578,600 nnz, the shipped cache), `type_path_gain`
`[LC4|LPLC2 -> DNp01 x3]`, `path_gain` `[DN -> vnc_ x3, VP -> DN x2]`, `flyverse_commit` `unknown` in the run copy
(no `.git`; the 43-file source fingerprint is in each JSON), local commit `653179b4` with a dirty tree -- the
working-tree diff `cluster_run.py` ships (26 files) is the concurrent tasks' opt-in work (`senses.Proprioception`,
the optic stream hooks, `batch_sim`'s `proprioception=None` argument), every field defaulting to off; the check that
none of it touched these sections is that the baseline reproduces round 5's shipped-default **walk** values
**bit for bit** -- `walk.power_max` 48.48052978515625, `power_sustained` 20.109053071339925, `GF_max`
4.629162311553955, all three in 6/6 draws. **"Bit for bit" is the three walk values and no more**: `motion.min_dsi`
is bit-identical nowhere (the six baseline draws take **five distinct values**, 0.2410964184151673 ..
0.24109659755256557, and the round-5 files themselves split 0.24109651 / 0.24109660 / 0.24109660 against
0.24595012 / 0.24595003 in `out/r5_adopt_default_1.json` / `_2.json`), and `loom.GF_peak` spans
**43.5929-47.2241** across the six baseline draws. Those two reproduce to 4 dp / within scatter, not bit for bit.
Tables: `out/pm_bound/replicates.md`
/ `.json` (`retire_measures.py --report-replicates out/pm_bound`) and `out/pm_bound/pm_bound_report.md` /
`pm_bound_summary.json` (`scripts/pm_bound_report.py`, CPU; sections 1-7 there carry every number below).

### The walk section, every configuration x 6 draws (`out/pm_bound/pm_bound_report.md` section 1 / 7)

| config | walk.power_max (< 50) | walk.power_sustained (< 50) | walk.GF_max (< 38) | loom.GF_peak (>= 20) | rotate.DNp20_flip (< -2) | motion.min_dsi (>= 0.1) | loom_escape.GF_peak (>= 33) / escapes | tally x6 |
|---|---|---|---|---|---|---|---|---|
| baseline (shipped) | **48.4805 x6** | 20.1091 x6 | 4.6292 x6 | 43.59-47.22 | -43.9 to -26.9 | 0.24109642-0.24109660 (5 distinct values, not bit-identical) | 44.1-50.3 / 1 x6 | 10/0/0 x6 |
| no_dn_vnc_gain (DN -> VNC x1, VP -> DN x2 kept) | **32.4141 x6** | 10.7886 x6 | 9.9266 x6 | 53.9267 x6 | -37.5 to -29.9 | 0.2531-0.2532 | 44.5-48.4 / 1 x6 | 10/0/0 x6 |
| no_path_gain (both gains off) | **34.2405 x6** | 11.8857 / 15.9133 x3 / 15.8312 x2 | 0.0000 x6 | 40.72-46.79 | -23.2 to -20.1 | 0.2510-0.2531 | 37.8-47.8 / 1 x6 | 10/0/0 x6 |
| no_drive_clip (clip 35 -> 1e9 mV) | **49.2480 x6** | 25.2732 x6 | 13.2599 x6 | 42.68-46.62 | -47.1 to -36.3 | 0.23701883-0.23713313 (all 6 **below** every default draw) | 40.7-53.3 / 1 x6 | 10/0/0 x6 |
| pair_gain_lpi_x1 | **51.5078 FAIL x6** | 20.2129 x6 | 9.7951 x6 | 56.24-61.93 | -42.3 to -25.8 | 0.2377-0.2490 | 50.5-62.4 / 1 x6 | 9/1/0 x6 |
| pair_gain_lpi_x2 | **48.1205 x5, 47.4015 x1** | 27.9224 x5, 28.7598 x1 | 9.0295 x6 | 51.40-52.75 | -35.9 to -28.2 | 0.2445 x6 | 48.4-58.3 / 1 x6 | 10/0/0 x6 |

`--report-replicates`: no check differs in status between the six baseline draws; `breaks_in_all` is empty for
every configuration but `pair_gain_lpi_x1` (`walk.power_max_hz`, 6/6); `fixes_in_all` empty everywhere (the baseline
fails nothing in these sections). The x1 / x2 walk values are bit-for-bit round 4's 51.5078 / 48.1205 and the
optic-audit's (`out/optic_audit/pair_gain_lpi_{x1,x2}/`), and `no_drive_clip`'s 49.2480 / 25.2732 / 13.2599 are
bit-for-bit the optic audit's one draw (`out/optic_audit/no_drive_clip/no_drive_clip.json`).

**On "walk.* is bit-stable".** It is bit-stable in the default (12/12) and in four of the five ablations (6/6 each,
across both GPU types: `no_dn_vnc_gain`, `no_drive_clip`, `pair_gain_lpi_x1` ran on B200 and H200 and agree to the
last digit), but **not in every configuration**: `pair_gain_lpi_x2_s2_d1` (H200) gives 47.4015 / 28.7598 / leg MN
2.20 Hz against 48.1205 / 27.9224 / 2.99 Hz in its five siblings (three of them on the same box), and
`no_path_gain`'s `power_sustained` takes three values (11.8857, 15.9133, 15.8312) while its `power_max` 34.2405 does
not move. Within this batch the draw-to-draw excursion on `power_max`, when it happens, is 0.6-0.7 Hz (0.63 Hz in the
round-4 corrections, 0.72 Hz here) and up to 4 Hz on `power_sustained`; "zero scatter" is the default's own record,
not a property of the section.

**The scatter on record is much larger than this batch shows, on the arm section (c) leans on** (dynamics round 2
verification -- and it argues harder for de-scoring, not less). `out/sk_les_holds_all/holdOptic_r{0,1,2,3}.json`
(`lesion.id` `holdOptic`, `kind` `hold_table`, note `scripts/build_hold_tables.py group Optic` -- the **same arm**
section (c) uses) give `walk.power_max` **64.9147 / 56.1737 / 68.0376 / 64.9147**: an **11.86 Hz spread over four
replicates**. And this document's own `off` suite arm spans **1.5598 Hz** (95.54158-97.10139), already wider than the
**1.5195 Hz** margin it is defending. In fairness that lesion set is a different harness (tool `lesion`, sections
rest / taste / smell / walk / bitter, `provenance.execution.backend` showing `cuda_kernels` / `cuda_graphs` /
`event_driven` all false), so it is not a like-for-like replicate -- but its `baseline` (48.4805 x4), `holdBrain`
(47.0012 x4) and `off` (95.5416 x3) agree **bit for bit** with the native suite, so the divergence is specific to
`holdOptic` and is not obviously a backend artefact. The honest statement is that **the margin is inside the worst
single-arm scatter on record**, not that it is two excursions wide.

### (a) Inside or outside 50 across draws

Inside, 12 of 12: 48.4805 in this batch's six draws and in `out/r5_adopt_default_{1,2,3}.json`,
`out/r5_skeptic_default_4.json`, `out/r5_attr_default_1.json`, `out/optic_audit/baseline/baseline.json` (all B200;
`out/sk/sk_attr_default.json` makes 13). Margin **1.5195 Hz**, which is *inside the worst single-arm scatter on
record* for this check (11.86 Hz on `holdOptic`, 1.56 Hz on the `off` suite arm -- see the bit-stability paragraph),
not merely two excursions wide.
There is no distribution to speak of in the default -- the answer is a point 1.5 Hz under a hand-set line,
and the line's 3 % margin is smaller than the scatter the same check shows on other arms.

### (b) Monotone in any scanned measure?

* **LPi34/43 -> LPLC2 factor: no.** 51.5078 (x1) / 48.0007 mean, 47.40-48.12 (x2) / 48.4805 (x4): down then up,
  crossing the bound once between x1 and x2 -- the round-4 shape, now under the shipped gains and with six draws per
  point (round 4 had x3 at 60.42 under the damped gains; x3 was not re-run). `walk.GF_max` is monotone (9.7951 / 9.0295
  / 4.6292) and `walk_gf.p99` was not (round-4 corrections); the gain suppresses the walking giant-fibre drive as
  designed and every point is far under that check's 38 Hz bound.
* **DN -> VNC gain: two points only, 32.4141 (x1) -> 48.4805 (x3), +16.07 Hz;** the x6 point (the note's 50 Hz) was
  not measured -- no `dn_vnc_gain_x6` configuration exists and the task's list did not include one.
* **Both path gains off: 34.2405**, i.e. removing VP -> DN x2 on top of DN -> VNC x1 *raises* `power_max` by 1.83 Hz
  while taking `walk.GF_max` from 9.93 to 0.00 and `loom.GF_peak` from 53.9 to 40.7-46.8: the two gains do not act
  on this check in the same direction, so a one-dimensional "gain -> power" referent does not exist even in the
  measure the note names.
* **Drive clip: 48.4805 -> 49.2480 (+0.77 Hz)**, with `walk.GF_max` 4.63 -> 13.26 and `power_sustained` 20.11 -> 25.27.

`common.compare` on every one of these contrasts returns `undetermined` (null SD 0: the baseline's six draws are
bit-identical, so z is undefined and the exact U at 6 v 6 sits on its floor 0.0022); the differences are read as
magnitudes, which is what a deterministic section allows.

### (c) Correlation with the room take-off rate over the four hold arms (`pm_bound_report.md` section 4)

| arm | walk.power_max (suite draws) | room hops per 1,000 fly-s (4 matched batches, 19,200 fly-s) | escape / voluntary | walking-GF median per batch |
|---|---|---|---|---|
| shipped default | 48.4805 x5 | 3.906 (75 = 27 + 48; 19 / 21 / 16 / 19) | 1.406 / 2.500 | 31.15 / 33.32 / 31.22 / 31.56 |
| holdBrain (optic side only) | 47.0012 x5 | 3.073 (59 = 19 + 40; 14 / 22 / 16 / 7) | 0.990 / 2.083 | 31.15 / 33.01 / 31.77 / 30.29 |
| holdOptic (Brain side only) | 64.9147 x5 FAIL | 0.417 (8 = 8 + 0; 1 / 4 / 0 / 3) | 0.417 / 0.000 | 27.10 / 27.14 / 26.75 / 25.95 |
| off | 97.1014 / 95.5416 / 96.4573 FAIL (**3 draws**) | 0.573 (11 = 11 + 0; 3 / 3 / 1 / 4) | 0.573 / 0.000 | 26.82 / 28.20 / 26.18 / 27.76 |

Suite files `out/r5_attr_*.json`, `out/r5_attr_dup/`, `out/sk/sk_attr_*.json`, `out/r5_adopt_default_*.json`; room
files `out/r5_adopt_sustain_live_{1,2,3}.json` + `out/sk_d1_shipped_4.json`, `out/d1_{holdBrain,holdOptic,off}_{1,2,3}.json`
+ `out/sk_d1_{holdBrain,holdOptic,off}_4.json` (the receptor_integration.md G.5 pooling). Spearman rho(`walk.power_max`,
hops rate) = **-0.60** (exact permutation p 0.417), Pearson r = -0.79 (p 0.21); escape route rho -0.60. At n = 4 arms
the smallest exact two-sided p a rank correlation can reach is 0.083, so in `compare`'s
vocabulary this is `underpowered` and the reading is the ordering: by `walk.power_max` holdBrain < default < holdOptic
< off, by room take-offs holdOptic < off < holdBrain < default. The two arms that FAIL the check are the two quietest
rooms; the two that PASS are the two that take off 5-9x more. A bound meant to guard against spurious take-offs
cannot be that bound.

Three limits on this table, all from the dynamics round-2 verification, none of which changes the ordering:

* **Quote the rank correlation, not the Pearson magnitude.** Spearman stays exactly **-0.600** under every
  substitution tried -- the table's value, the pooled 9 `holdOptic` draws, the eager mean 63.51, the worst draw
  56.17 -- so the ordering finding is robust; the Pearson moves -0.787 -> -0.664 and its p 0.213 -> 0.336 under the
  same substitutions. Section 4 of `pm_bound_report.md` also presents `holdOptic` as five bit-identical draws
  (64.9147 x5) while the four-replicate record of that arm at 56.17-68.04 sits unmentioned on file
  (`out/sk_les_holds_all/holdOptic_r{0..3}.json`; see the bit-stability paragraph).
* **`spearman_voluntary` -0.80 is a tie-breaking artefact and is withdrawn.** `scripts/pm_bound_report.py:133` ranks
  with `np.argsort(np.argsort(x)) + 1`, which has **no tie correction**, and the voluntary rates contain a tie
  (0.000 for both `holdOptic` and `off`). Tie-corrected the value is **-0.7379**, and because of that tie
  |rho| = 1 is unreachable on the voluntary row, so **the quoted 0.083 floor does not apply to it**. The headline
  rows (`all` and `escape`) have no ties and are exact.
* **The `off` arm rests on 3 suite draws**, below `docs/INTERP.md` 10.2's ">= 4 runs per arm" (the table above says
  so: 3 suite / 4 room). It is also the only arm whose `walk.power_max` has scatter, and its 3-draw mean 96.3668 is
  what the Pearson consumes. It does not change the FAIL status (every draw 95.5-97.1, far above 50) or its rank, so
  the ordering is unaffected.

### (d) The referent, and which line

`scripts/benchmark.py` line 85:

    "walk.power_max_hz": Ref(22, "<", 50, "4", note="per-frame max of the wing-power MN mean (22 Hz with DN->VNC x3, 50 at x6)"),

The note's two anchors are session-4 numbers from a model without the receptor lookup, the LPi gain, the GF damping
or its retirement; at the same operating point (DN -> VNC x3) the shipped model measures 48.48, at x1 32.41, and the
x6 point was not re-measured. The 50 is not an animal number: it is `body.Flight.takeoff_power_hz`, the wing-power
level the body must hold for 0.3 s to launch, and `walk.power_sustained_hz` (line 86, `Ref(22, "<", 50, ...)`, the
0.3 s running mean) is the check that carries that referent -- the shipped default measures 20.11 on it, no clip
25.27, LPi x2 27.9-28.8, and receptor_integration.md G.4 records that off sits at 49.2-50.6 on it while making 0
voluntary take-offs in 52,800 fly-s, so even the sustained form is a weak predictor of the room. The per-frame
maximum has no referent of its own. **Recommendation: de-score it** -- keep the number in the table and the JSON,
stop scoring it: on line 85 change the op / bound to the form the suite already uses for an unscored, reported value
(`loom.escape_cm` on line 88: `Ref(3.5, "notnone", 0, ...)`), i.e. `Ref(48.5, "notnone", 0, "4, r6", note="reported,
not scored (docs/audits/anti_runaway.md round 6): per-frame max of the wing-power MN mean; the scored take-off
criterion is walk.power_sustained_hz")`, with the reference value updated to the shipped model's 48.48 so the table's
"reference" column stops printing 22. **Precision on what that form does:** it does **not** remove the row from the
tally. `scripts/benchmark.py:132` makes `notnone` pass iff the value is not `None`, so the check stays **in the pass
count** as a row that can only fail when the measurement is missing -- exactly what `loom.escape_cm` does today (it
prints PASS in every table, including for `out_norm_l2` at 50.0 cm). "De-scored, kept as report" is accurate in
effect; "unscored" is not literally true, and a tally of 10/0/0 with this form still counts `walk.power_max_hz` as a
PASS. **Not recommended: re-deriving the referent from x3 / x6.** The x6 point is
unmeasured under the shipped gains, the two path gains move the check in opposite directions (b), the LPi scan is
non-monotone at three points, and any number read off a gain scan is a hand-set bound on a hand-set gain -- the
project rule's "labelled control arm", not a referent. The line is **not changed here**; the benchmark's owner
changes it, and until then every verdict below is stated both ways.

### What this frees, and what it does not

* **LPi34/43 -> LPLC2 x4 (`optic.DEFAULT_PAIR_GAIN`).** With `walk.power_max` scored, x1 is 9/1/0 in 6/6 and x4
  cannot be retired (the round-4 / optic-audit verdict, reproduced bit for bit). With it de-scored, x1 is 10/10 in
  6/6 on these sections and the only remaining measured cost of x1 is the one the gain was added for -- the walking
  giant-fibre drive, `walk.GF_max` 4.63 -> 9.80 (bound 38) -- plus `loom.GF_peak` 56-62 and `loom_escape.GF_peak`
  50-62 *higher* than the baseline's 44-50. Whether x4 can then be retired is a full-suite question (29 checks, and
  the room walking-GF tail that fires the 33 Hz escape, which the pinned section does not measure -- G.4), not one
  these ten checks settle; it is not retired here.
* **`OpticParams.drive_clip_mv` 35 (the +-35 mV clip on the optic -> spiking injected current).** Seven draws now
  (six here + the optic audit's one): 10 PASS / 0 FAIL x6 on walk / a / b (11/11 with `walk_gf` in the audit), no
  check worse in status than the baseline, `walk.power_max` 49.2480 (PASS by 0.75 Hz -- one excursion wide, on the
  check this section de-scores), `power_sustained` 25.27 (margin 24.7), `walk.GF_max` 13.26 (margin 24.7; the clip
  binds on the walking GF's peak frames), `motion.min_dsi` 0.2411 -> 0.2371 (`compare` calls this `result` at 6 v 6
  within the batch, |diff| 0.004; **an earlier form of this sentence set that verdict aside as "inside the shipped
  default's 0.241-0.246 cross-batch band", which is wrong -- 0.2371 is BELOW 0.241, not inside the band.** All six
  `no_drive_clip` draws (0.23701883-0.23713313) lie below **every** shipped-default draw on file: this batch's
  0.24109642-0.24109660 and the round-5 files' 0.24109651-0.24595012. The cross-batch band therefore does not
  neutralise the `result`. The correct statement is that 0.2371 is **0.0040 below the lowest shipped-default draw on
  record**, small against the check's 0.1 bound, and irrelevant to this section's decision because **nothing is
  adopted**), `loom.GF_peak` 42.7-46.6 vs 43.6-47.2 and `loom_escape.GF_peak` 40.7-53.3 vs 44.1-50.3 (`null`),
  `rotate.DNp20_flip` -47 to -36 vs -44 to -27 (`null`, p 0.09). By this document's rule the ablation passes with a
  status margin on every check but the one under question. **Adopt nothing:** the adopt-alone rule (round 4, executed
  in round 5) requires the ablation's own full 29-check suite x >= 3 and the room take-off protocol (3-4 batches x 16
  flies x 300 s) against the shipped default before the default changes, and neither was run in this batch. The
  numbers above are the record for that run; the prediction it should test is that the clip's removal raises the
  room walking-GF tail (pinned GF max 4.6 -> 13.3 Hz is the same sign as the holdBrain / holdOptic 12.5 / 13.3 that
  G.4 showed does NOT predict the room, so the prediction is weak).
* **Unknown-NT AL LN -> GABA override.** Not in this batch. Its round-4 verdict rested on 26/1/2 with `walk.power_max`
  60.88 under the damped gains; with the check de-scored the deciding numbers are the olfactory ones (KC active 816
  -> 1805, LH apple clean 4.3-4.5 -> 5.2-5.5 against 6 -- a 0.5 Hz margin), and the verdict "cannot be retired,
  replacement = a transmitter prediction for the 27 cells" stands on those, unchanged.

### Caveats

(1) Ten checks per draw (sections walk, a, b), not the 29-check suite; the tallies above are section tallies.
(2) Six draws per configuration on two GPU types; the walk values agree across the types wherever both ran, but the
one 0.72 Hz excursion (`pair_gain_lpi_x2_s2_d1`) is a single draw on an H200 and is not attributed to the device.
(3) The x6 DN -> VNC point and the LPi x3 point were not re-measured under the shipped gains; (b)'s DN -> VNC
statement is two points. (4) The room correlation is over four arms and `underpowered` by construction; it is
reported because the ordering, not the p, is the finding. (5) The batch ran with the concurrent tasks' opt-in modules
in the working tree; the bit-identity of the baseline's **three walk values** to the round-5 record is the evidence
they were inert in these
sections. (6) `retire_measures.py`'s new `provenance.execution.device` field reads `null` (the realised device is in
`config.device` and `provenance.execution.device_requested`) -- a bookkeeping defect of this edit, to be fixed by
whoever next touches the script; `flyverse_commit.commit` is `unknown` in every cluster JSON as in every other cluster
Result (INTERP.md 10.2), and the identity is the cache md5 + source fingerprint.
(7) Extent of the round-6 edit: this section begins at line 758 and **runs to the end of the file** (it closed round 6
at line 947, +191 / -0, append-only; the dynamics round-2 verification corrections above extend it further). Nothing
in rounds 1-5 was edited by either pass.

Files: `scripts/pm_bound_batch.sh` (the batch), `scripts/pm_bound_report.py` (every table and statistic above ->
`out/pm_bound/pm_bound_report.md`, `pm_bound_summary.json`, console `pm_bound_report_console.txt`),
`scripts/retire_measures.py` (the `no_drive_clip` configuration and the provenance block), `out/pm_bound_cluster.log`
(`36 job(s), 0 failed  (6.3 min)`), `out/pm_bound/<cfg>_s<s>_d<k>/` (36 run directories), `out/pm_bound/replicates.md`
/ `.json`, `out/pm_bound/structure_check.txt` (`--check` of the six configurations, CPU). Files behind the dynamics
round-2 corrections above: `out/sk_les_holds_all/{baseline,holdBrain,holdOptic,off}_r*.json` (the 11.86 Hz
`holdOptic` scatter), `out/pm_bound/*/*.json` (the `motion.min_dsi` and `loom.GF_peak` per-draw values),
`out/r5_adopt_default_{1,2}.json` (the 0.24595 `min_dsi` draws), `scripts/pm_bound_report.py:133` (the
tie-uncorrected rank), `scripts/benchmark.py:132` (what `notnone` does).
