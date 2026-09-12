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
backend, the shared override cache (sum|W| 121,460,584), run dir `/mnt/beegfs/neurome/runs/r3-retire-33dbb1`, 10.3 min
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
