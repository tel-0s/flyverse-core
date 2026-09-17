# flyverse notes

Working notes for putting the MaleCNS v1.0 fly connectome into simulations and games.

## Dataset (local)

`D:\Datasets\male-cns-connectome-v1.0\flat-connectome\` (CC-BY 4.0, HHMI Janelia FlyEM + Google Research;
paper: Cell 2026-09-03, doi:10.1016/j.cell.2026.08.015; neuPrint dataset `male-cns:v1.0`).

The three files a point-neuron simulation needs (everything else is per-synapse detail):

| file | what we use |
|---|---|
| `body-annotations-...feather` (211,577 rows) | `bodyId, type, instance, superclass, class, subclass, somaSide, somaNeuromere, status, entryNerve, exitNerve, assignedOlHex1/2, somaLocation` |
| `body-neurotransmitters-...feather` (1.8 M rows) | `consensus_nt` (fallback `celltype_predicted_nt`, `predicted_nt`) |
| `connectome-weights-...feather` (151.9 M rows, 1.05 GB, loads in ~2 s) | `body_pre, body_post, weight` |

Facts established empirically (see `flyverse/connectome.py`):

* 165,122 bodies have status `Traced`. Restricting edges to Traced->Traced gives 25,563,197 edges /
  124.0 M synapses -- which is exactly the `-significant-only` file (25,568,639 rows), so that file is
  just "both endpoints traced".
* We keep Traced bodies + all photoreceptors (6,091; 1,983 of them are untraced axon fragments that
  still carry synapses): **N = 167,106 neurons, 25,578,600 edges, 124.2 M synapses**. Compiled to a
  signed CSR matrix (post x pre) in 15 s; cached in `cache/` (205 MB).
* NT labels for the kept neurons: ACh 104k, Glu 29.6k, GABA 22.1k, histamine 7.9k, unknown 2.5k,
  5-HT 404, DA 395, OA 141. Sign convention: ACh/DA/OA/5-HT +, GABA/Glu/histamine -, unknown +.
* Sensory inventory: photoreceptors R1-R6 (3,377), R7y/p/d/unclear (1,300), R8y/p/d/unclear (1,329);
  53 ORN types; gustatory GRNs by subclass: leg bristle 768, wing bristle 385, labellar bristle 163,
  taste peg 60, pharyngeal 48; Johnston's organ JO-A/B/C/D/E/F; mechanosensory ~5.7k.
* Motor inventory: 1,314 descending neurons (all the classics present as L/R pairs: DNa01, DNa02,
  DNp09, MDN, DNb01, DNp07, DNg13 ...); 708 VNC motor neurons labelled by neuromere (T1/T2/T3 x L/R
  ~86 each, A1-A10) and subclass (`fl/ml/hl` = front/mid/hind leg, `wm` wing, `nm` neck, `ad` abdominal);
  107 central-brain motor neurons incl. proboscis MN1..MN13 (MN9 = rostrum protractor, the classic
  proboscis-extension readout).

### Retinotopy

Photoreceptors have **no** `assignedOlHex1/2`. The hex-column grid (h1 1..36, h2 1..39) is annotated on
the columnar lamina/medulla cells (L1/L2/L3/L5, C2/C3, Mi1/Mi4/Mi9, Tm1/2/4/9/20, T1; ~1,770 each =
875 L + 892 R columns). We assign each photoreceptor the column of its synapse-weighted strongest hexed
postsynaptic partner (top-3 partners agree 94-99%). Result: 5,895/6,091 photoreceptors placed;
R1-R6 cover 300 L / 526 R columns, R7 535/682, R8 623/698 (the retina is partly outside the EM volume).

Axis orientation from soma-position regressions: `h1+h2` increases dorsally, `h1-h2` increases
anteriorly (lamina cell somata; medulla somata give the opposite AP sign, as the external chiasm
predicts). Check: DRA photoreceptors (R7d/R8d) land at the dorsal edge. The two hex axes are 120 deg
apart ((1,1) is a neighbour): with that embedding the eye is ~33 columns tall x ~26 wide, i.e.
~154 x 120 deg at 4.6 deg/column.

## Neuron model

Leaky integrate-and-fire after Shiu et al. 2024 (Nature 634:210; code
github.com/philshiu/Drosophila_brain_model), which is also what stonkfly/doomfly use on this dataset:
v_rest = v_reset = -52 mV, v_th = -45 mV, tau_m = 20 ms, tau_syn = 5 ms, refractory 2.2 ms, delay
1.8 ms, **w = 0.275 mV x synapse count x sign** (the single free parameter). We integrate with
exponential Euler at dt = 0.5 ms in torch on the GPU: whole CNS at ~1,700 steps/s on an RTX 4090
(~0.85x real time). Optional extras that Shiu's model does not have: constant current injection per
neuron (mV; a drive D > 7 mV makes a neuron fire at 1/(2.2 ms + 20 ms ln(D/(D-7)))), Poisson forcing,
and spike-frequency adaptation. [2026-09-13: the **realised** refractory period is **2.5 ms**, not 2.2 --
`t_ref` 2.2 ms holds a cell for 5 steps of dt = 0.5 ms and it fires again on the 6th, so the minimum ISI
is 3.0 ms (333 Hz) and the analytic rate above, plus every "refractory-limited" ceiling in this file, is
that bound. `docs/audits/interp_health.md` 4; see "Session 10, interpretability toolkit".]

### What we learned driving it

* With no input the network is silent (good: no spontaneous runaway).
* Driving photoreceptors alone does nothing downstream: they are histaminergic (inhibitory) and the
  lamina is silent. doomfly's fix -- a 12 mV tonic bias on L1-L5 -- makes L1/L3 fire ~50 Hz in the dark,
  and that alone drives 15% of the CNS above 1 Hz (central brain mean ~60 Hz) while light only lowers
  L1 from 49 to 36 Hz and the medulla stays silent. Conclusion: photoreceptor->lamina is a graded-potential
  stage and a spiking LIF of it is not a useful front end.
* Injecting graded photoreceptor/lamina signals into a *spiking* medulla does not propagate either:
  even 35 mV of drive on every right-eye Mi1+Tm3 leaves T4/T5 silent (Tm3 is suppressed by the
  GABAergic Pm cells; T4 needs ~5,000 synapse-spikes/s). The optic lobe up to T4/T5 is graded in the
  real fly, and flyvis (Lappalainen et al. 2024, Nature) shows a connectome-constrained *rate* model of it
  works. **So flyverse is a hybrid**: the 89,390 `ol_intrinsic` neurons are rate units
  (`flyverse/optic.py`; deviations from an operating point, L2-normalised signed weights so coherent
  fan-in is amplified ~sqrt(N), contrast input gain 3, slow adaptation), and the 71,625 remaining neurons
  (visual projection neurons onward) are Shiu-style LIF driven by injected current
  `gain_out * sum_i What_si (r_i - b)` (80 mV per unit of mean input deviation, clipped at 35 mV).
* With that, in a textured room (checked cloth, striped walls) a 90 deg/s yaw rotation modulates every
  optic-lobe layer by ~0.2 (L1..T4/T5), and drives DNa02 on one side at ~28 Hz vs ~0.4 Hz on the other,
  plus MDN/DNa01/DNp01, with <2% of the CNS above 1 Hz. An untextured room gives almost nothing (mean
  contrast 0.01): yaw rotation leaves horizontal edges invariant, so the world needs vertical structure.
* **Runaway excitation** was not visual: any sustained input tipped the central brain into a 300 Hz
  loop (KCs, PAM dopamine neurons, PEN, antennal-lobe LNs). Cause: the 2,494 neurons with no usable NT
  prediction were treated as excitatory (Shiu's default), and they include the largest antennal-lobe
  local neurons (v2LN30, lLN2F_a, lLN2T_d: 25-50k output synapses each, GABAergic as a class). Fix:
  unknown NT -> sign 0, AL LN types -> GABA. With that the brain is quiet without input again.
  Making DA/OA/5-HT purely modulatory (sign 0) was *not* sufficient on its own; they were later set to sign 0
  anyway (NT_SIGN, session 3), which is the current state -- see the session-9 NT audit.
* Synaptic budget: a target needs ~5,000 synapse-spikes/s (7 mV / (0.275 mV x 5 ms)) to reach
  threshold, so columnar medulla cells must fire ~100 Hz to drive T4/T5 (166 synapses from
  Mi1/Tm3/Mi4/Mi9/C3/CT1 per T4a). Shiu drove sensory inputs at 150 Hz Poisson for the same reason.

## Status of the room demo (2026-09-10)

* Pipeline: ray-traced spectral radiance per ommatidium (7 rays, 4.5 deg acceptance) -> photoreceptor
  low-pass (10 ms) + contrast adaptation (300 ms) -> graded optic lobe (1 ms substeps, tau 10 ms) ->
  LIF CNS (0.5 ms steps) -> motor readout (DNa02 L/R turning, DNp09/DNa01/DNa03/DNb01/DNa04 + leg MNs
  forward, MDN backward, MN9 proboscis) -> fly pose on the table -> next frame. 10 ms of brain per frame.
* Speed on the RTX 4090: 16.6 ms per frame without drawing (0.6x real time); brain.step(20) is 13 ms
  of that. The pygame window costs ~15 ms every 4th frame. A GPU->CPU sync per LIF step (`tensor.any()`)
  had silently doubled the frame time; avoid syncs inside the step loop.
* Intrinsic walking drive of 0.4 cm/s (body.py `baseline_speed`) keeps optic flow, and therefore the
  brain, alive; without it the fly stops, the contrast adapts away and the CNS goes silent.
* Taste: touching fruit (<1 cm) fires a placeholder "sweet" set (a random quarter of the front-leg
  bristle GRNs) at 120 Hz Poisson. MN9 responds weakly (2-9 Hz). Which leg GRNs are sugar cells is not
  in the annotations; the way to find them is connectivity to the known sweet second-order neurons
  (hemibrain names appear in `synonyms`: G2N-1 = GNG232, Rattle = GNG132, Usnea = GNG175, Phantom =
  GNG229, Bract = DNge173/174, Clavicle = ANXXX462a, Zorro = GNG215, Fudog = DNg67, Roundup = GNG108).
* Known issue: taste input can trip a 200 Hz loop among antennal-lobe local neurons + projection
  neurons (lLN1_bc, v2LN30, lLN2T_a, M_imPNl92). Some lLN types are cholinergic in reality, so the
  loop may be real wiring plus missing inhibition; adaptation/STD only partly damp it.
* The motor mapping is a hypothesis. Runs are sensitive to details (which eye has holes, room asymmetry),
  so left/right responses are not symmetric. The fly does turn and back up in response to optic flow.

## Session 2 (autonomous): taste, smell, flight, loom

* **Sweet GRNs found by connectivity** (`scripts/find_sweet_grns.py` -> `flyverse/data/taste_grns.csv`):
  scoring every gustatory neuron by synapses onto the known sweet second-order neurons (G2N-1/GNG232,
  Rattle/GNG132, Usnea/GNG175, Phantom/GNG229, Zorro/GNG215, Roundup/GNG108, Fudog/DNg67,
  Bract/DNge173-174, Clavicle/ANXXX462a) vs bitter ones (Bitter/DNg28, Scapula/GNG087) cleanly
  separates 165 sweet cells (labellar bristles LB3a/c/d, taste pegs, some pharyngeal) from 47 bitter
  (LB1a/c). Only 3 *leg* GRNs qualify -- the reference neurons are the labellar pathway; leg sugar GRNs
  need their own second-order reference set. Stimulating the labellar sweet set at 100 Hz drives the
  sweet interneurons (Usnea 67 Hz, Rattle 49, Phantom 47, G2N-1 28) and the proboscis motor neurons
  MN9 (4-14 Hz) and MN11D -- the Shiu et al. sugar -> proboscis result reproduces on MaleCNS.
* **Olfaction** (`flyverse/olfaction.py`): fruit = odour sources -> ORN classes by glomerulus (rough
  DoOR-style tuning table), Poisson rates by distance. PNs encode odour identity (DM2 PNs for banana,
  DM1 for apple) but the antennal lobe runs *hot*: PNs at ~200 Hz, GABAergic LNs at 100-250 Hz, from
  any tonic ORN input. Cause: Shiu-strength ORN->PN synapses (3,000+ per PN) saturate PNs at 8 Hz
  spontaneous ORN rate, and the cholinergic LN population (lLN1_bc: 30 cells with 86,864 synapses onto
  each other; lLN2T/lLN2X) excites itself. Neither input normalisation (alpha 0.75/1.0 -- which also
  kills the taste pathway) nor zeroing cholinergic LN outputs fixes it: the point model lacks the
  presynaptic (GABA-B on ORN terminals) inhibition and the gap-junction coupling that shape the real AL.
  Setting: ORN base 3 Hz. It does not destabilise the rest of the brain (MB silent, DNs quiet).
* **Fan-in cap** (`LIFParams.input_norm_ref = 5000, alpha = 1`): neurons with more than 5,000 input
  synapses get their unitary synapse scaled by 5000/total. Motivation: the giant fibre (~40k inputs)
  fired every ~0.5 s from walking-related central-brain input (DNp70, SAD073, PVLP010 -- not from
  LC4/LPLC2), making the fly hop continuously; with the cap it is silent during walking and the taste
  pathway is unchanged (MN9 3-5 Hz). Big neurons have low input resistance, so this is defensible.
* **Direction selectivity: not achieved.** Per-type time constants (Mi4/Mi9/CT1/Tm9 slow) and slower
  gratings give DSI 0.03-0.06 for all T4/T5 subtypes (`scripts/probe_motion.py`). The rate model is
  linear around its operating point; flyvis gets DS only after training. Consequence: LC4/LPLC2 respond
  to self-motion optic flow as much as to looms, so loom-vs-walking discrimination for the escape
  circuit is a gain trade-off, not a computation, until DS exists.
* **Flight** (`body.Flight`, `body.wing_groups`): takeoff on a giant-fibre spike (escape: ballistic
  hop at 0.6 m/s, 45 deg) or 0.1 s of sustained wing-power MN (DLMn/DVMn) activity; airborne thrust and
  lift from power-MN rate, yaw from steering-MN (b1-3, i1-2, iii1/3, hg1-4, ps, tp) asymmetry; without
  wingbeat the fly falls and lands (table or floor). GF -> TTMn is weak in the chemical-synapse table
  (90 synapses; the real GF-TTMn synapse is electrical), so the GF spike itself is the trigger.
* **Loom** (`scripts/probe_loom.py`, L key in the demo): a black 3 cm ball approaching at 1 m/s from the
  side drives DNp01/DNp11/DNp02/DNp04 and triggers the escape at ~3.5 cm range.
* **Settings that came out of this** (defaults now): optic->spiking gain 100 mV (80 left LC4/LPLC2
  below threshold for the loom; 120+ or L2-normalised output weights make them fire from walking flow),
  fan-in cap 5000/alpha 1 (alpha 0.5 already lets the GF fire spontaneously). Cost: DNa02's visual
  turning response (28 Hz vs 0.4 Hz before the cap) is now ~1-3 Hz; turning in the demo comes mostly
  from leg-MN asymmetry. A per-cell-type gain (learned, as in flyvis) is the real answer; a uniform
  synapse cannot serve both a 40k-input giant fibre and a 250-input T4.

## Session 3: tackling the negatives

* **Direction selectivity achieved.** First the anatomy check: in our hex embedding, each T4 subtype's
  Mi9 (leading) and Mi4 (trailing) inputs are offset from its Mi1 centre by one column in the expected
  directions -- T4c: Mi4 +2.9 deg up / Mi9 -3.1 deg down, T4d mirrored; T4a/T4b: +-3.6-4 deg along
  azimuth, mirrored between eyes. The rate model just wasn't using it: units sat at a mid-range
  operating point (linear) and the delayed inhibitory arms were weak. Fix (now default in optic.py):
  T4/T5 operating point 0 (pure ReLU), x5 gain on Mi4/Mi9/CT1/C3 -> T4 and Tm4/Tm9/CT1/TmY15 -> T5, slow
  cells (Mi4/Mi9/CT1/Tm9) at 150 ms. Result with a 60 deg/s, 30 deg grating: DSI 0.16-0.26 with the
  correct preferred direction for all eight subtypes (T4a/T5a front-to-back, b back-to-front, c up,
  d down); at 30 deg/s DSI halves (temporal-frequency tuning, as in the real cells). The loom still
  triggers the escape at 3.5 cm and walking leaves the GF at 0-2 Hz.
* **DN activation screen** (`scripts/screen_dns.py`, batched: 64 brains, one DN type each, 150 Hz
  for 400 ms, ~2 min for all 600 types): essentially *nothing* moves in the VNC. DNp09, MDN, DNa02,
  whose optogenetic activation walks real flies, change leg-MN rates by < 1 Hz; strongest wing-power
  driver DNg15 at 4.5 Hz; DNge062 -> MN9 14 Hz is the one clean hit. Not the fan-in cap (alpha 0 is the
  same). See below for the cause.
* **Cause: short-term depression.** With `std_u = 0` DNp09/MDN/DNa02 at 150 Hz drive 1,000-1,500 VNC
  interneurons and the leg MNs -- but *non-specifically* (all three give the same pattern, both sides
  equally, wing power at 250 Hz) and the cliques come back (DNg33, DLMn, AN27X013 at 300 Hz). So the
  point model has two regimes, silent or epileptic, and depression had been choosing "silent" for the
  VNC. Depression is presynaptic-rate dependent: a 150 Hz descending command is exactly what it
  suppresses (10% efficacy), which is why single-DN optogenetics-style tests fail here.
* Within-type synapses are 2.8% of the connectome but hold most of the cliques (lLN1_bc 2,900/cell,
  KC->KC 415k = 57% of KC input, FR1, PFN, PEN, DNg33 ...). `LIFParams.same_type_gain` scales them
  (populations like these are typically gap-junction coupled and fire in synchrony in the animal). It is
  not sufficient: with depression off a *cross-type* recurrent network in the AVLP (AVLP154/157/488 ->
  AVLP520/AVLP428/CL212/CL002) runs at 280 Hz. Spike-frequency adaptation (per neuron, so it does not
  block feed-forward commands) is the remaining physiological brake -- see the benchmark results.
* `scripts/benchmark.py` scores a parameter set on all known behaviours at once (rest, taste, smell,
  DN drive x3, walking GF, loom, rotation). Use it before changing defaults.
* **Per-connection saturation** (`LIFParams.conn_cap = 60`, new default, with depression off and
  adaptation 1.0 mV/spike): a connection contributes at most 60 synapse-equivalents (16.5 mV per
  presynaptic spike). Connections above 60 synapses are 0.6% of all connections (5.8% of synapses);
  the AVLP giants are ~435 per cell. Benchmark: rest silent; sugar -> MN9 8 Hz; loom -> GF 76 Hz,
  escape at 3.5 cm; walking GF 0; odour -> 3,261 KCs active (the MB is alive, not yet sparse);
  MDN drives a specific VNC set (IN06B020, GNG562, LBL40 at 70-80 Hz, no storm); DNa02_L gives an
  *ipsilateral* leg-MN bias (0.8 vs 0.1 Hz) for the first time. DNp09 still ignites the AVLP network.
  Cap 40 kills DNp09's effect and most of taste; same-type damping on top kills DNp09's effect too --
  so in this model DNp09's motor effect *is* that AVLP recurrence.
* **DNa02 is not an optomotor relay.** Of its 23,756 input synapses only 595 (2.5%) come from visual
  projection neurons; the top inputs are AN03A008, PS049/PS059, AN04B003, **PFL3** (the central-complex
  steering output), LAL126/083/179, VES052, AOTU019. During a 90 deg/s rotation its net synaptic drive
  is -0.6 mV (inhibited). The 28 Hz "turning response" seen before the fan-in cap was central-brain
  crosstalk. So the rotation benchmark now reports *which* DNs respond asymmetrically to rotation, and
  the body readout should use those; DNa02 should be read as a goal-steering command (PFL3 -> DNa02).
* **Measured optomotor readout.** Most lateralised DNs under rotation (DNg34, DNge119, DNp68) are
  lateralised the same way for both directions -- constant biases from the asymmetric eye (left-eye
  hole) and room, not turning signals. Ranking types by how their L-R asymmetry *flips* with rotation
  direction gives DNp04 (left rotation: L 17 / R 2.6 Hz; right rotation: L 1.5 / R 5.9), the
  lobula-plate tangential types LPT27, LPT30, and MeVP50/Nod4; HS/VS are weak (< 5 Hz) here. body.py now
  uses DNp04 + LPT27 + LPT30 (L - R) as a course-stabilising optomotor term, DNa02 as goal steering.
* Final defaults of this session: depression off, adaptation 1.5 mV/spike (tau 200 ms), conn_cap 60,
  same_type_gain 0.1, fan-in cap 5000, T4/T5 ReLU + x5 inhibition + x2 output, slow cells 150 ms.
  Benchmark: rest silent; sugar -> MN9 3.7 Hz (6.7 without same-type damping); odour -> 2,409 KCs
  active; DNa02_L -> ipsilateral leg bias; MDN -> specific VNC set (GNG562, IN06B020, IN07B010); DNp09
  -> no storm (and no VNC effect either); walking: GF 0, wing-power MNs 2.5 Hz (max 11) so no
  spurious takeoffs; loom GF 39 Hz, escape at 3.5 cm. Without same-type damping, walking optic flow
  drove the DLMn/DVMn clique to 30-90 Hz and the fly took off every half second. Runs are chaotic:
  repeat a benchmark before trusting a 20% difference.
* **Antennal lobe fixed by putting depression back where it is documented.** With global depression off
  the AL's PN <-> cholinergic-LN recurrence ran at 300 Hz from 1 Hz of spontaneous ORN input (and
  MBONs at 230 Hz). `LIFParams.std_u_by_type` now applies u = 0.2 (tau 300 ms) to ORNs, all AL local
  neurons and all PN types, nothing else; spontaneous ORN rate 1 Hz. Result: spontaneous PNs 4-55 Hz,
  odour-specific responses (apple: DM1 123 Hz, VA2 101, DM4 33; banana: DM2 66), LNs 20-200 Hz,
  Kenyon cells sparse (0-1 Hz), MBONs 1-4 Hz. u = 0.5 on ORNs alone silences PNs (strong depression
  equalises steady-state transmission across rates and leaves only onset transients).
* **Final benchmark (commit after this note)**: rest silent; sugar -> MN9 3.7 Hz; odour -> PN 13 Hz
  mean (max 108), 1,249 KCs weakly active, LNs 57 Hz; DNa02_L -> ipsilateral leg bias (0.6 / 0.0);
  MDN -> GNG562 / IN06B020 / IN07B010 at 60 Hz, no storm; DNp09 -> nothing (its VNC effect was the
  AVLP storm); walking: GF 0, wing power 2.5 Hz (max 11), 1.3% of CNS active; loom: GF peak 20 Hz,
  escape at 3.5 cm (borderline -- the escape threshold is 20 Hz); rotation: DNp04 and DNp20 flip
  with direction (DNp20 15/1 Hz during rightward rotation -- doomfly's arbitrary turning DN turns out to
  carry a rotation signal here too). Demo: one loom-triggered hop, one spurious GF hop in 14 s; the
  fly barely walks (baseline 0.4 cm/s minus an MDN term) -- the walking drive needs a real source.

## Session 4: per-pathway gains

* **First per-cell-type gain: descending -> VNC synapses x3** (`LIFParams.path_gain`, default
  `[("^descending_neuron$", "^vnc_", 3.0)]`). Benchmark: DNa02_L -> left leg MNs 3.1 Hz vs right 0.1
  (ipsilateral, through IN14B003 / IN13B001 / IN14B004); DNp09 -> its own premotor set (IN06B030,
  IN09A011, IN27X005 at 45 Hz); MDN -> the backward-walking interneurons (IN06B020 152 Hz, IN03B015,
  IN07B013, LBL40) -- three different DNs, three different VNC patterns, no storm. Walking: GF 0,
  wing power max 22 Hz (no spurious takeoffs); loom GF 30 Hz, escape at 3.5 cm; taste MN9 5.8 Hz.
  x6 gives stronger drive but wing power max 50 during walking (takeoffs). T4/T5 output x3 alone
  raises the loom margin (37 Hz) but costs walking-GF margin (max 14); kept at x2.
* body.py: baseline walking 0.8 cm/s; the MDN "back up" term only counts above 15 Hz (MDN fires a few
  Hz from self-motion optic flow).
* **Takeoff triggers.** With the DN->VNC gain, walking-related VNC input drove TTMn (the jump-muscle
  MN, 3,000 inputs, below the fan-in cap) to 20+ Hz and the fly hopped every second. In the animal TTMn
  fires one spike per giant-fibre spike, so the GF alone is now the escape trigger; voluntary takeoff
  needs a 50 Hz wingbeat command sustained for 0.3 s. Demo: 5.5 s of quiet walking, loom -> GF 21 Hz
  with TTMn following at 21 Hz, a 1.2 s flight off the table. Smell adds ~10 Hz to wing-power MNs
  during walking (benchmark walking test now has smell on, as the demo does).
* Loom -> GF varies run to run (peak 9-124 Hz); the fan-in cap scales the GF's synapses by 0.125,
  which under-weights LC4/LPLC2 -> GF, a known strong pathway. Fixed with a second pathway gain,
  visual_projection -> descending_neuron x2 (default): loom GF 26 Hz, escape at 3.5 cm, walking GF 0,
  taste unchanged. x3 re-ignites the AVLP network during walking and halves the taste response.
  The model's per-pathway gain table is therefore: DN -> VNC x3, VP -> DN x2, everything else Shiu's
  0.275 mV with the connection cap and fan-in cap.
* **DN screen, second pass (current defaults; `out/dn_screen.csv`)** -- now a real motor map:
  wing power: DNg02_a 146 Hz (10 cells, legs 0.5 Hz), DNa08 121, DNp31 100, DNpe036 47, DNp43 46,
  DNg02_f 42, DNp27 41, DNg37, pMP2, DNg02_e -- DNg02 is the wingbeat-amplitude DN cluster of Namiki et
  al. 2022, so the model recovers a known flight pathway. Giant fibre DNp01 -> TTMn 47 Hz (the jump).
  Legs: DNp27, DNp43, DNge035, DNg100, DNge130 (5-9 Hz both sides); MDN 3.7/2.9; DNp09 only 0.6.
  Proboscis: DNge080 28 Hz, DNge062 17, DNge059 10. 37 of 473 DN types drive wing power above 15 Hz.
  [2026-09-13: the 981-population atlas adds the *lateralised* map this screen does not have: **DNge035
  is the strongest lateralised leg driver**, +7.50 / -6.19 Hz of leg L-R and *contralateral*, three times
  DNa02's +2.24 / -1.45, then DNa13, DNge037, DNge049, DNge073. A measurement, not a readout change --
  `body.py` is untouched. `docs/audits/interp_atlas.md` 6.1; "Session 10, interpretability toolkit".]

* **RL, second attempt.** Retraining the DN-rate decoder under the new model with a 12 cm spawn
  curriculum gave the same nothing after 12 generations (mean return -5), so it was stopped. The
  observation was the problem, not the optimiser: DN rates are dominated by self-motion. New
  `EnvParams.obs = "pn"`: per-glomerulus projection-neuron means (71) plus their change over 0.5 s --
  the odour code plus the derivative a chemotaxing fly needs (klinotaxis). 142 dimensions instead of
  1,314. Results below when the run finishes.
* **Is the odour code decodable at all?** A hand-written klinotaxis policy on the PN observation
  (walk forward; turn at a rate proportional to the total drop in glomerular PN activity over the last
  0.5 s; `FlyRoomEnv.klinotaxis_action`) scores 45 +- 110 with 83 tasted frames and a final distance of
  4.0 cm, against random 34 / still 42 (47 tasted frames, 4.6 / 5.4 cm) and the oracle 271 (265 tasted
  frames). So the projection-neuron code carries a usable gradient, but a weak one at 5-12 cm from a
  fruit with our 10 cm odour half-distance -- and the reward's spawn variance (+-110-145) swamps it.
  A linear ES decoder cannot express klinotaxis (needs the rectified derivative), hence obs="pn3" adds
  relu(-delta). Better experiments: bilateral antennae (Gaudry et al. 2013 -- flies steer on
  left/right ORN asymmetry), a steeper plume, and scoring by tasted frames rather than return.
* **Result of the odour-code ES run** (64 flies x 40 generations x 5 s, 12 cm spawn curriculum,
  obs="pn"): the trained decoder scores 34 +- 124 with 16 tasted frames -- *worse* than random (47) and
  far below the hand-written klinotaxis (48 +- 116, 93 tasted frames). The two generations with high
  mean return (28: +3, 38: +390) were common spawns that started on fruit. Verdict: ES with a
  spawn-dominated return does not find chemotaxis in 40 generations even though a 3-line hand policy
  on the same observation does. Next: fitness = tasted frames with several spawns per policy, the
  "pn3" feature, and PPO on Linux via PufferLib.

## Demo UI (session 5)

* The UI is drawn on a fixed design canvas (1280 x 760, + 330 for the brain map) and scaled uniformly
  into a resizable window (`pygame.transform.smoothscale`, letterboxed); mouse events are mapped back
  through the same transform. GIFs capture the canvas.
* Scene view: `OrbitCam` (azimuth / elevation / distance around a target; `C` follows the fly). The
  view is re-rendered only when the camera key changes (3 ms at 480 x 300).
* `flyverse/brainmap.py`: 140,024 of the 167,106 neurons have a `somaLocation`; they are binned once
  into dorsal (z across, x down) and lateral (z across, y down) pixel grids at the 0.5-99.5 percentile
  extents; per frame the activity vector (spiking: rate / 40; optic-lobe units: |dr| x 2) is summed per
  pixel with `np.bincount` and divided by sqrt(count) so dense regions do not wash out. Type means for
  the "most active types" list are a bincount over type codes (a pandas groupby here cost 50 ms/frame).
* Body: `FlyState` has `pitch` and `roll`; `forward/left/up` give the full frame and `body_to_world`
  uses it, so the retina rays and the fly's-eye camera roll and pitch with the body. In flight the nose
  follows the velocity vector (climb/dive) and the wings bank into the turn (0.12 rad per rad/s of
  yaw, clipped at 60 deg); landing zeroes both. Trail: 20 samples/s, faded over `--trail-seconds`.

* **Texture at fly scale.** The room's checks (3 cm), stripes (50 cm) and planks were coarse, periodic
  and uniform at the scale a fly sees from 1 mm above a surface. `world.py` now multiplies every
  surface's reflectance by three octaves of value noise (2 cm, 8 mm, 3 mm; `World.detail` = 1.4,
  clamped at 0.1) -- weave, grain, fibres, fruit-skin speckle -- and the room has non-periodic
  landmarks (a dark picture on the +y wall, a bright door on the +x wall, a skirting strip). Mean
  |contrast| while walking forward went from 0.012 (flat) to 0.021 (detail 0.6) to 0.039 (detail 1.4);
  rotation 0.11-0.13. Walking still leaves the GF at 0.1 Hz; the loom escape still fires at 3.5 cm.
* **Roll is now commanded.** `Flight.k_roll`: the wing steering-muscle asymmetry (L - R) rolls the body
  (a banked turn is what an asymmetric stroke produces) on top of the bank that follows the yaw rate;
  pitch follows half the flight-path angle, clipped at 40 deg. Both relax towards level with a 120 ms
  time constant -- the haltere-mediated stabilising reflexes that keep a real fly upright, which the
  brain model has no haltere input to provide. Before this, pitch and roll were pure consequences of
  the flight path (the fly dove at -82 deg after hops).

## Session 6: food-finding -- wind, plumes, anemotaxis

* **Why**: flies find fruit by odour-gated anemotaxis (surge upwind on an odour hit, cast crosswind
  when it is lost; van Breugel & Dickinson 2014), not by gradient descent on an isotropic cloud.
  That needs a directional plume and a wind cue.
* `flyverse/air.py`: `Air` (wind vector with slow meander; Gaussian plume per fruit, sigma = 1.5 cm +
  0.12 x downwind distance, plus a near-source term and a per-source puff fluctuation; vectorised over
  positions: 64 flies' ORN rates in 4 ms), `BilateralOlfaction` (two antennae 1 mm apart; ORNs are
  assigned to an antenna by the laterality of their PN targets: 708 left / 1,396 right / 535
  bilateral), `WindSense` (JO-C/E neurons, sided the same way via their AMMC/WED targets, driven by
  the backward/forward deflection of each antenna, which point forward-lateral at +-45 deg).
* **Wind-direction descending neurons** (`scripts/probe_wind.py`, fly standing, wind from the front /
  left / right): the JO -> AMMC/WED -> DN pathway is the strongest lateralised signal in the model.
  DNp18 fires on the side the wind comes from (wind left: L - R = +60 Hz; wind right: -40), DNp33 on
  the opposite side (-37 / +64), then DNge016, DNg99, DNge175, DNg05_a, DNp19 and DNpe017 (doomfly's
  choice, which does carry a wind signal). The DN rates are identical with and without an odour plume:
  the odour gate on upwind turning is not at the DN level in this model.
  [2026-09-13: the atlas reproduces these flips **without a room**, by stimulating the 335 JO-C/E cells at
  the exact per-cell rates `senses.Wind` would produce (DNp18 +50.98 / +50.59 against the suite's +45.2; the
  +12 % is the atlas fixing the wind at 180 deg against the room's 20 deg meander -- with the meander, +47.5),
  so JO stimulation equals sense-driven wind through `fb.wind`. And under *symmetric* head-on drive the same
  DNs keep a **fixed anatomical L-R offset**: DNp18 +13.45, DNp73 +19.51, WED080 -17.07, DNge016 +9.89,
  DNp33 -10.68 Hz -- half of DNp73's apparent flip is that offset, not the wind.
  `docs/audits/interp_atlas.md` 3.1; "Session 10, interpretability toolkit".]
* body.py: anemotaxis term `k_wind * gate * upwind`, upwind = 0.5 [(DNp18-group L - R) - (DNp33-group
  L - R)]; gate = clip((mean PN rate - 10 Hz) / 15 Hz), held with a 1.5 s decay after the plume is lost
  (the surge), plus a small forward bonus at full gate. The gate is the one behavioural assumption
  (odour -> upwind) that the connectome model does not itself produce.
* **First foraging run: two bugs found by instrumenting one run per second.** (1) The fly was spinning
  at 50-70 deg/s: the optomotor term. DNp04's right side sits at 18-31 Hz against ~0 on the left even
  in still air (the eye is asymmetric), so `-k_opto * (opto_L - opto_R)` was a constant left turn
  that its own rotation response could not cancel. Fix: the term now uses the asymmetry minus its 2 s
  running mean (only *changes* in rotation count). (2) The odour gate barely opened (0.06-0.17): it was
  the mean over all 516 PNs, which hardly moves when four glomeruli respond. Fix: gate on the most
  active glomerulus's mean PN rate ((max - 25 Hz) / 60 Hz). (3) The JO wind input at 100 Hz max pushed
  the giant fibre over threshold (2-6 hops per 30 s); the GF does receive antennal mechanosensory
  input in the animal. JO max rate lowered to 50 Hz. Wind sensing itself was right: the DNp18/DNp33
  asymmetry flipped sign with the fly's heading relative to the wind (+47 Hz at 46 deg off the wind,
  -30 at -94 deg).
* **First fruit found by the brain's own signals.** With the three fixes, a fly started crosswind
  turns into the wind (heading relative to wind -74 -> -1 deg in 8 s, gate ~0.95), walks upwind at
  ~1.4 cm/s and reaches the blueberries on the plume axis (11 -> 0.8 cm in 11 s). `probe_foraging.py`,
  3 seeds x 30 s from 12 / 7 / 9 cm: anemotaxis term ON -> closest approach 0 / 3 / 0 cm, tasting
  0.5 / 0.1 / 0.7 s; OFF -> 4 / 2 / 3 cm, no tasting. What still spoils it: 8-10 giant-fibre hops per
  30 s in two of the three runs (the escapes eventually carry the fly off the table), and the fly
  overshoots after contact. Both are the next targets.
* **The hops were not the wind.** GF input-weighted drivers during foraging: SAD073, GNG300, LPLC2,
  DNp70, CL367 -- the same central-brain crosstalk as before, plus LPLC2 responding to the fruit itself
  (an approached fruit *is* an expanding object). Single crosstalk spikes vs a loom burst: the escape
  threshold went from 20 to 30 Hz of smoothed GF rate (>= 3 spikes in ~100 ms), the LC4 / LPLC2 -> GF
  synapses got a x3 type-level gain (`LIFParams.type_path_gain`; Ache et al. 2019) and LC4 / LPLC2 a
  and a fly that is tasting stops walking (tasting reach 1.5 cm). Foraging then: 0 hops in 5 of 6
  runs, and the flies that reach fruit stay on it (17-21 s of tasting).
* **Loom-detector gain is context dependent.** Raising LC4 / LPLC2's optic-lobe drive x1.5 fixed the
  loom in the bare loom probe but, with smell and wind on, put the GF at 20-55 Hz from the first
  half-second and produced 95 hops per 30 s (a jump-land-jump chain across the room). Tested in the
  full demo context (5 s walking + loom, 2 seeds): x1.0 / 30 Hz -> exactly one hop per run, the loom's
  (GF 35-46 on the loom vs <= 24 walking); x1.25 -> 3-4 spurious hops. Kept x1.0. Also added a 1 s
  post-landing escape refractory (`Flight.landing_refractory_s`) so a trigger cannot chain.
* **Foraging, final numbers for this session** (`probe_foraging.py`, 3 seeds x 30 s, start 7-12 cm
  downwind of fruit): anemotaxis term ON -> fruit reached and fed on in 2 of 3 runs (20.4 s and 17.4 s
  of tasting, closest approach 1 / 3 / 1 cm), 0 hops in all three; OFF -> 1 of 3 (by baseline
  walking), 8 hops in one run. The fly finds food by turning into the wind when it smells it, using
  DNp18 / DNp33 for the wind direction and its antennal lobe for the odour -- with the gate between
  them supplied by the body.

## Session 7: closing the loop (sustain)

* `body.Metabolism`: energy in [0, 1], resting drain 1/300 s plus 0.15 per metre walked, feeding refills
  at 1/15 s; hunger = 1 - energy scales the odour-gated upwind drive (0.3 at zero hunger -> 1.0 when
  starving), satiety at 0.95 ends the meal and a new one only starts below 0.7 (hysteresis, so a sated
  fly actually leaves). A feeding fly stands still; airborne flies do not feed. Demo-scale constants:
  a tank lasts ~4 min of walking, a meal ~15 s.
* Casting: when the gate falls below 0.25 after having been above 0.6 in the last 30 s, the body runs
  a crosswind zigzag (+-90 deg/s, sign flipping every 1.5 s) for up to 8 s -- the search program real
  flies show on plume loss (van Breugel & Dickinson 2014). Modes shown in the UI: searching / surging /
  casting / feeding, with an energy bar and meal count. `scripts/probe_sustain.py` scores minutes of
  autonomy: meals, minimum energy, hops, path length, time in each mode.

## Session 8: making the loop hold (sustain)

The first 5-minute sustain run (energy 0.4, fruit 10 cm away) failed in an instructive way: first
meal only at 125 s, 18 hops, and the fly ended on the floor. Each cause turned out to be a readout or
body assumption, not the connectome:

* **The odour gate was too slow.** With gate = (max-glomerulus PN rate - 25) / 60 the fly surged only
  when almost on the fruit. Second version: max-glomerulus rate against a 10 s adapting baseline
  ("onsets count, sustained odour fades"): first meal at 15 s instead of 125 s -- but on the floor,
  2 m upwind of any plume, it read "odour" 70-95% of the time. Recording the per-glomerulus PN means
  at four sites (30 s each, `scratchpad/record_pn.py`) showed why: the model's antennal lobe is
  noisy. 90 of 625 PNs exceed 40 Hz in 20 s with 1 Hz ORN input; the strongest bursters (110-154 Hz)
  are the 275 multiglomerular `M_` PNs (many GABAergic), which the gate had been treating as one
  glomerulus; and 2-cell glomeruli (DC1, DA4m, VP1m ...) burst to 50-85 Hz for a second at a time. The
  real odour code is a *sustained*, glomerulus-specific elevation: blueberry -> DM2 (4 PNs) 80 Hz;
  apple, 8 cm downwind -> DM1 71, VA2 58, DM2 52 Hz. Candidate statistics were scored offline
  (fraction of time gate > 0.6: want high next to fruit, ~0 on a plume-free spot):

  | statistic | blueberry | apple | plume-free table | floor |
  |---|---|---|---|---|
  | adaptive max (shrunk, 0.3 s smooth, 10 s baseline) | 0% | 0% | 0% | 0% |
  | per-glomerulus baseline, tau 30 s | 81% | 59% | 6% | 0% |
  | **level: 1 s smooth, glomeruli >= 3 PNs, max - median, (x - 20) / 40** | **100%** | **100%** | **0%** | **0%** |
  | top-3 mean - median | 100% | 0% | 0% | 0% |

  The level statistic is now the gate (`Locomotion.pn_smooth_s`, `pn_min_cells`, `pn_base_hz`,
  `pn_gate_hz`); uniglomerular PNs only (350 cells in 70 glomeruli, 46 with >= 3 PNs). No adaptation:
  a sated fly stays near its food, which is what flies do; hunger scaling (x0.3 at zero hunger) and
  satiety do the leaving.
* **The table-edge "nudge" was a 720 deg/s spin.** `Locomotion.step` turned the fly by pi/2 * 8 rad/s
  for every frame its next step lay outside the bounds. Pushed against the y-edge by the upwind
  drive, the fly pirouetted at 720 deg/s for a minute at a time (the commanded yaw never exceeded
  +-200 deg/s; the heading log gave it away), the rotating scene drove LPLC2 / LC4 (3,000-9,700 mV/s
  into the GF -- a genuine expansion signal from the walls), the giant fibre fired 60-80 Hz every
  second, the fly hopped off the table and then round the room (122 hops in 5 min), and the visual
  storm even showed up in the PNs. Replaced by contact-mediated edge behaviour: slide along the
  edge, turn towards the interior at 90 deg/s, heading wrapped. Corner-of-the-room check: 58 deg/s
  mean turning instead of 600, and the fly walks out of the corner.
* **Walls fire the giant fibre harder than a loom.** With the spin gone, a fly next to a wall (10 cm,
  filling the eye, expanding with every turn) still had the GF at 57-72 Hz *sustained* -- more than
  the 32-48 Hz peak of a 3 cm ball at 1 m/s. LPLC2 in the animal is inhibited by wide-field motion;
  the rate model's LPLC2 is not, and no threshold separates the two. Time course does: the loom is a
  burst from silence. `Flight.gf_habituation`: the escape threshold is `gf_hz` (30) plus the GF's 10 s
  running mean. Loom from rest: threshold ~33-35, escape fires (4 of 5 looms; real flies escape a
  similar fraction). Corner: 3-4 first-contact hops in 30 s instead of 11, then the threshold sits at
  ~40 above the sustained drive. Raising the LC4/LPLC2 -> GF gain to x4 did not raise loom peaks
  (32-48 Hz at x3 and x4: a ~100 ms burst can only carry so many spikes) and was reverted.
* The five central-brain inputs that fire the GF during ordinary walking (input-weighted: SAD073,
  GNG300, DNp70, CL367, PVLP010) are damped x0.3 (`DEFAULT_TYPE_PATH_GAIN`); walking GF peak 25-29 Hz.
* **The brain supplies the gate after all.** The open question from session 7 was whether some central
  population carries the odour signal cleanly enough to replace the hand-made PN statistic. Recording
  the mean rate of every LH / MB-output / DN / WED type (3,245 types, 12,460 cells) at the fruit sites
  and the plume-free ones, with heading-matched controls (the fly facing into vs away from the wind at
  both kinds of site), and ranking by d' between the worst odour site and the best clean site:

  | type (cells) | blueberry, into wind | apple | blueberry, away from wind | clean table, into wind | clean, away | floor |
  |---|---|---|---|---|---|---|
  | LHPD4d2_b (2) | 23.5 | 14.7 | 23.4 | 4.6 | 5.0 | 1.8 |
  | LHPD4a2 (4) | 22.4 | 14.7 | 22.3 | 4.8 | 5.2 | 1.8 |
  | LHAV3k1 (2) | 26.0 | 19.7 | 26.0 | 9.4 | 9.9 | 4.9 |
  | LHAV3h1 (2) | 26.7 | 22.1 | 26.8 | 9.9 | 10.3 | 3.7 |
  | LHPD5c1 (2) | 35.2 | 34.2 | 35.9 | 19.3 | 20.1 | 7.4 |
  | LHAD1f2 (2) | 24.4 | 17.8 | 25.0 | 8.1 | 8.8 | 4.2 |
  | WED080 (2) -- wind, not odour | 14.6 | 15.9 | 0.6 | 15.6 | 0.2 | 0.1 |
  | DNp18 (2) -- wind, not odour | 26.6 | 27.7 | 11.0 | 26.9 | 9.4 | 8.2 |

  Every one of the six lateral-horn types separates fruit from no-fruit 100% / 0% of the time at a
  midpoint threshold, independent of heading; the LH is the innate-valence output of the olfactory
  system, so this is the right place to find it. The wind-facing types (WED, DNp18, DNg05_a, DNge016)
  fall out of the same screen as a control: identical with and without odour at the same heading,
  which independently confirms the JO -> WED -> DNp18 wind-direction readout. `body.LH_ODOUR_TYPES`
  (14 cells) is now the default odour source (`Locomotion.odour_source = "lh"`, gate = (1 s-smoothed
  mean rate - 10) / 12); the PN level statistic remains as `odour_source = "pn"`.
* `scripts/probe_sustain.py` scores minutes of autonomy: meals, minimum energy, hops, path length,
  time in each mode. Numbers for the current defaults are below.
* **Search program.** With the gate fixed, the failure moved: the fly feeds once, and the sated fly's
  residual upwind pull (0.3 x) walks it past the fruit to the upwind table edge, where no plume can
  reach it; hungry again, "searching" had no program (baseline walking + DN noise) and it starved
  20-30 cm from the apple. Now: sated odour gain 0.1; and a hungry fly without a surge-level odour
  for 10 s runs the documented offset response (Alvarez-Salvado et al. 2018) -- a hunger-scaled
  downwind drift (the wind DNs with reversed sign, fading as the odour grows) with Ornstein-Uhlenbeck
  turning, and klinokinesis (turning x0.25 while the 1 s odour signal exceeds its 4 s average; Jung et
  al. 2015), which climbs the isotropic near-field of a fruit where an upwind surge goes nowhere.
* **Startup escape.** Every run's first hop came at 0.8-1.1 s: the brain starts at rest, the first
  frames of full sensory input are a brain-wide transient, and the habituation mean was zero exactly
  then. The habituation state now starts at `gf_hz` (threshold doubled, relaxing over ~10 s).
* **Overhaul merged** (branch `refactor/control-surface`, by another agent; `docs/ARCHITECTURE.md`,
  `docs/CONTROL_SURFACE.md`, `docs/PERFORMANCE.md`): `Connectome.subset` + `regions.py` modules,
  `FlyBrain` (senses in, `MotorRates` out), the demo and RL env rebuilt on it, CUDA-graph capture of
  neural frames and sensory rays, fp16 weights (opt-in), `step_budget` / `AsyncFlyBrain`, tests. The
  session-8 behaviour was ported onto `MotorRates` in the merge (`lh_odour` and `pn_glom_cells`
  readouts; `Locomotion` / `Flight` readouts take `dt_s`). Checks on the merged tree: loom escape
  (GF 46-50 Hz vs 23-25 walking), corner 3 hops / 20 s, odour gate 0% floor / 100% by the apple / 0%
  plume-free; 25 tests pass.
* **Where the behaviour lives -- the split.** By this point the body layer had accumulated the odour x
  wind product, casting, hunger scaling, a search program, escape habituation and an efference-copy
  discount: a behaviour layer bolted onto the full model, each piece with a citation and all of them
  hand-designed. Now: `body.Locomotion` / `body.Flight` are a physical readout only (two readout
  filters remain and are labelled as such: the optomotor high-pass for the model's eye asymmetry, the
  MDN threshold), and the programs live in `flyverse/programs.py` as opt-in, swappable stand-ins for
  circuits (`AnemotaxisProgram` for the LAL / CX steering integration, `EscapeGating` for the
  wide-field suppression the rate LPLC2 lacks). They read only brain signals (DNp18 / DNp33, the LH
  odour population) and the demo / probes run the plain model unless `--program anemotaxis` is given.
  The efference-copy escape discount (threshold x (1 + |yaw| / 90 deg/s)) was tried and rejected
  here: a loom makes the fly turn, so it blocked 3 of 3 real escapes (GF peaks 37-73 Hz).
* **Screens as the method.** `flyverse/screen.py` turns the LH screen into a tool: `TypeRecorder`
  (per-type or per-type-and-side population means from a rate snapshot), `record`, `rank` (d'
  between the worst positive and best negative condition, so confounds score low), `contrast`,
  `ablate` / `restore`. `scripts/screen_odour.py` reproduces the LH result; `scripts/screen_steering.py`
  asks whether any bilateral population's L - R asymmetry flips with wind side more under odour --
  i.e. whether the brain already computes the gated steering the program supplies.
* **Does the brain already compute the gated steering? No -- and the screen says where it would.**
  `scripts/screen_steering.py` (1,993 bilateral types by side, fruit vs plume-free site, wind on the
  fly's left vs right, pose pinned, 30 s each). The wind-side signal is enormous and purely
  mechanosensory: L - R flips by DNp33 -49 Hz, DNp18 +45, WED080 -41, DNge016 +30, DNg99 -19, DNp73
  +18, DNp19 +17 -- and *identical* with and without odour (DNp18 +44.8 vs +44.2 Hz). The best odour x
  wind-side interaction anywhere is d' 0.70 (DNge133, 2 cells at 4 Hz), i.e. nothing. The reference
  rows are the finding: the central-complex steering output is silent in the full model -- PFL3
  0.00 Hz, DNa02 0.01, LAL010 0.14 (the wind DNs and the LH odour population both fire; their
  integration point does not). So the anemotaxis program is standing in for a specific silent
  region, and that region -- not the body -- is where a simulated module belongs: a CX / LAL model
  taking the wind-direction DNs and the LH odour signal, driving DNa02 by stimulation.
* **The central complex is silent.** Census in the full model (fly 6 cm from the blueberries,
  walking, 17 s; mean rate and fraction of cells above 1 Hz): EPG compass 0.01 Hz / 0%, PEN, PEG,
  Delta7, PFL, PFN, PFR, hDelta, vDelta all 0.00 / 0%, ring neurons (ER) 0.08 / 2%, FB tangentials
  0.22 / 3%, LAL 0.87 / 13% -- against LH 1.7 / 20%, MBON 4.3 / 46%, KC 1.0 / 19%, DNs 2.0 / 22%,
  visual projection 0.7 / 11%. The model has no heading representation, so it has no goal steering
  (PFL3 -> DNa02), whatever the wind DNs and the LH say. Whether the CX is input-starved (the ring
  neurons' visual pathway MeTu -> AOTU -> TuBu -> ER) or cannot sustain its attractor under the
  anti-runaway settings is the next experiment, and the natural place for the first swapped-in
  module.
* **Why the CX is silent, causally** (`FlyBrain.stimulate` at the blueberry site):
  - the visual pathway to the compass dies before the ring neurons: MeTu 0.7 Hz -> TuBu 0.1 -> AOTU
    0.1 -> ER 0.0. Driving all ER ring neurons at 40 Hz leaves EPG at 0.0 (they are GABAergic onto
    EPG, so that is the expected sign, not a bug);
  - the attractor does not sustain: driving a quarter of the EPG population at 60 Hz recruits Delta7
    (11.6 Hz) and ExR (8.5) but PEN only 0.3 Hz, and 3 s after the pulse EPG is back to 0.4 -- the
    EPG <-> PEN recurrence that makes the compass a ring attractor does not close under the current
    synaptic scaling (candidates: the fan-in cap, conn_cap, and same-type damping on these dense,
    stereotyped connections);
  - the output works: PFL3-left at 80 Hz -> DNa02-right 22.6 Hz, DNa02-left 0.0 -- the contralateral
    steering projection of the animal, intact down to the legs.
  A swapped-in module therefore has a precise contract: supply the compass (heading -> EPG bump) and
  the goal comparison, drive PFL3 L / R by stimulation, and the connectome does the rest. That, not a
  body program, is the principled replacement for `AnemotaxisProgram`; alternatively, the two
  synaptic questions above (TuBu / AOTU silence, EPG <-> PEN gain) are tractable with the benchmark.
* **Can gains alone wake the compass? No.** Per-type gains on the two broken links, same
  calibration style as DN -> VNC: EPG <-> PEN / PEG x4 makes the network bistable between silent and
  a 300 Hz all-cell state (baseline 109 Hz on one seed, 0 on another; a driven bump recruits all 50
  EPG to 300+ Hz with Delta7 at 250 Hz sculpting nothing and the ring neurons dragged to 41 Hz);
  x8 the same; adding MeTu -> TuBu -> ER x4 lifts TuBu to 2.7 Hz and nothing else. In every case
  PFL3 stays at 0.00 Hz even with EPG at 300 -- the EPG -> PFN -> hDelta -> PFL chain is blocked as
  well. The point model's silent-or-epileptic regime again, now in the one circuit whose function is
  an attractor. This is a research thread (Delta7 inhibition vs EPG <-> PEN excitation under the
  connection and fan-in caps, and the FB columnar pathway), not a knob; the practical route to
  navigation in the meantime is the module contract above (compass + goal -> PFL3 stimulation).
* **The first swapped-in module: `flyverse/cx.py` `CompassSteering`.** Built to the contract the
  causal test gave: it reads the brain's own wind-direction DNs and LH odour signal from
  `MotorRates`, decides the steering error the LAL / FB circuit would (the same surge / cast / search
  logic as the body program, reused), and expresses it by stimulating PFL3 left / right (80 Hz at
  full error; PFL3 projects contralaterally, so the right PFL3 turns the fly left) and DNp09 for
  forward drive. Nothing is injected into the body: DNa02 and the leg motor neurons do the turning.
  30 s from 12 cm (seed 0, energy 0.4): plain model 12 -> 15 cm, no meal; CX module 12 -> 4 cm, one
  meal (PFL3 4.5 Hz mean, DNa02 |L - R| 0.6 Hz mean); body program 12 -> 12 cm, one meal. Demo and
  sustain probe: `--program cx`. It is the template for module swaps: same slot, brain-side output.
* `scripts/screen_odour.py` (the tool) reproduces the hand-made screen: LHPD4d2_b d' 7.04, LHPD4a2
  6.84, LHAV3h1 6.05, LHAV3k1 5.80, LHPD5c1 5.66, LHAD1f2 5.43, all 100% / 0% at a midpoint threshold,
  3,233 populations in ~6 x 30 s.
* **Sustain, honestly measured** (`probe_sustain.py`, 5 min, energy 0.4 at start, 10 cm from the
  blueberries; meals / final energy / hops):

  | seed | plain model | anemotaxis program + escape gating | CX module (PFL3 stimulation) + gating |
  |---|---|---|---|
  | 0 | 3 meals / 0.94 / 8 hops | 1 / 0.00 / 3 (hopped off the table at ~50 s) | 2 / 0.32 / 3 (off the table at ~200 s; min 0.28) |
  | 1 | 1 / 0.00 / 35 (off the table at ~50 s) | 2 / 0.75 / 3 | 2 / 0.02 / 2 (min 0.02, never zero) |
  | 2 | 1 / 0.50 / 5 | -- | -- |

  The CX module -- the same decisions, expressed through PFL3 -> DNa02 -> legs instead of the body --
  is the only configuration in which energy never reached zero in either seed. The plain connectome model, wandering (baseline walking, DN drive, optomotor) and relocated by
  escape hops, finds a fruit roughly every 100 s on a table with 19 fruit items, and so does the
  programmed forager; both lose runs to the table edge (the escape hop that lands on the floor,
  where no plume reaches and the fly cannot climb back). The upwind program's systematic bias is
  not an advantage here: it walks sated flies past the fruit to the upwind edge, and hungry ones into
  the corner. Two conclusions: (i) "sustains itself" is a property of the table as much as of the
  fly -- a single fruit source would separate the strategies; (ii) the remaining body-level
  failure is the edge, which is a physics question (a fly at a table edge grips, and a hop from the
  edge is a flight), not a behaviour one.
* Taste after the merge: forced sugar contact -> the 165 labellar GRNs at 122 Hz -> MN9 6 Hz, off ->
  0; the pathway is intact. `teleport_to_fruit` (the T key) was making the fly escape 40 ms after
  arrival -- a jump cut next to a 4 cm apple *is* a loom to LPLC2 / LC4 -- so the teleport now carries
  a 3 s escape refractory (UI convenience, documented as such). `probe_taste.py` defaults to the
  labellar set; its old "leg" default was 3 cells that drive nothing.
* **Optic-column prune** (`LIFParams.prune_frozen`, default on): the LIF matrix carried every
  synapse from and onto the 89k optic-lobe rate units although they never spike there; dropping them
  takes the matrix from 24.7M to 13.0M nnz with spike counts identical for 200 / 200 frames.
* **Per-module clocks** (`LIFParams.dt_by_module`, e.g. `{"vnc": 1.0}`; demo `--dt-by-module
  vnc=1.0`): a slow module integrates every k base steps from the spikes accumulated since its last
  update, through the delay buffer (the slowest clock may not exceed the 1.8 ms delay). Row-split
  matrices and per-phase coefficient vectors keep the elementwise work identical and the spmm
  proportional to the fast module's synapses; it captures into CUDA graphs (spike counts identical
  eager vs replay, 200 / 200). Behaviour with the VNC at 1 ms: loom escape fires (GF 65 / 44 Hz),
  rest rates within ~10% (central 1.57 vs 1.50 Hz, DNs 2.12 vs 1.92, VNC motor 3.87 vs 4.18), wind DN
  asymmetry +19 vs +20 Hz; with descending at 1 ms as well, +23 Hz and GF 67 / 61. Opt-in for now.
* **Clean timing with the prune and the clocks** (same protocol, idle 4090, 300 frames):

  | configuration | ms / frame | x real time |
  |---|---|---|
  | eager, no prune (previous default) | 22.5 | 0.44 |
  | eager, prune (new default) | 19.8 | 0.51 |
  | prune + VNC at 1 ms | 19.8 | 0.51 |
  | prune + VNC and descending at 1 ms | 19.0 | 0.53 |
  | prune + `--cuda-graphs` | 12.7 | 0.79 |
  | prune + VNC 1 ms + `--cuda-graphs` | 11.6 | 0.86 |
  | prune + VNC 1 ms + `--cuda-graphs --weight-dtype float16` | 10.4 | 0.96 |
  | prune + `--fast --cuda-graphs` | 7.7 | 1.31 |

  The eager loop is launch-bound, so halving the synapses (prune) buys 2.7 ms and halving the VNC's
  share (clock) buys nothing until the launches are captured; under CUDA graphs the kernels dominate
  and the clock is worth ~1 ms, fp16 another ~1 ms. The forecast of "~14 ms eager" was wrong for
  that reason. Real time for the full brain at B = 1 is now `--cuda-graphs --dt-by-module vnc=1.0
  --weight-dtype float16` (0.96x) or `--fast --cuda-graphs` (1.31x).
* **Surfaces: flies stick to what they stand on** (`flyverse/surfaces.py`). Every sustain
  configuration was losing runs the same way: an escape hop from the table edge lands on the floor,
  and a fly on the floor can neither smell the fruit nor get back up. That is a body-model artefact --
  the old walker was a rectangle with a clipped boundary and a turn-away rule, and the flight model
  landed only on the floor or the table top. Now the walkable world is a set of axis-aligned faces
  (the inside of the room, the outside of the table slab and its four legs, 36 faces) and the fly's
  pose on a surface is a forward vector in the face plane plus the face's outward normal; heading /
  pitch / roll are derived (and owned by the flight integrator while airborne). Walking over a convex
  edge rotates the fly onto the side face (new normal = the direction it was walking, new forward =
  minus the old normal), walking into a concave corner climbs it (new normal = the face it ran into,
  new forward = the old normal), and a flight lands on the first face its path crosses. No decision
  by the fly is involved; the contact-mediated edge turn is gone. Checks without a brain: table top
  -> side -> underside at 2 cm/s; floor -> wall -> ceiling (including the case where a step lands
  exactly on the wall plane, which stuck the first version); floor -> leg side -> slab underside ->
  other leg -> floor; a hop from the +x edge lands 14 cm out on the floor, a hop mid-table lands back
  on the top, a powered takeoff lands on the top. The eye sits `eye_height` along the surface normal,
  so a fly on a wall or the underside sees the room from there; the wind and odour senses use the
  same body frame. Save states carry the pose (the face is re-resolved on load).
* **Two bugs the surfaces exposed.** (i) Landing was only checked after 50 ms of flight; a
  voluntary takeoff (0.2 m/s) is a ~60 ms flight that could pass through the floor plane inside the
  unchecked window, after which no face is ever "entered" again -- one plain fly fell to z = -182 m.
  The gate is gone (the 3 mm launch offset already prevents re-landing at takeoff) and a fly found
  inside a solid snaps to the nearest face. (ii) `nearest_fruit` was 2-D: a fly on the floor 75 cm
  below a blueberry "tasted" it. Now 3-D. **Earlier sustain rows with meals at z = 0 were this bug**
  (the CX seeds' floor-side meals; the pre-surfaces plain seed 0's three meals were all on the table
  and stand).
* **Sustain with the surfaces, honestly** (5 min, seeds 0-2, meals / final energy / hops): plain
  0 / 0.00 / 26, 1 / 0.00 / 31, 0 / 0.00 / 20; CX module + gating 0 / 0.00 / 2, 1 / 0.00 / 5,
  1 / 0.00 / 5. Every fly leaves the table by an escape hop within the first minute (a hop from the
  edge lands 14 cm out on the floor; a hop from the side or underside lands further) and never
  returns: from a 16 m^2 floor a random walk does not find a 5 cm table leg, and the model's flight is
  a ballistic hop without a goal. The physics is now right; what remains is (a) the hop rate on the
  table -- the rate optic lobe's LPLC2 / LC4 respond to self-motion expansion, so a fly turning near
  a fruit or an edge escapes (20-30 hops / 5 min plain, 2-5 with habituation), and (b) the return,
  which in the animal is a flight towards light / odour, i.e. behaviour the model does not have.
  Both are model questions (the loom pathway's wide-field suppression; goal-directed flight), not
  body physics; the single-fruit table will be measured with the fly held on the table (a
  glass-walled table, `--fence`) so foraging can be scored independently of the escape problem.
* **A single fruit source, and a fence.** `make_room(seed, fruit_set="apple")` leaves only the apple
  (4 cm, at (0.25, 0.15)); the standard start 40 cm directly downwind of it is inside its plume.
  `--fence` replaces the walkable world by a 10 cm glass box around the table top (faces the fly can
  walk on, nothing visual), so a hop from the edge lands on the fence and the run scores foraging
  without the escape problem. Both are demo / probe options (`--fruit apple --fence`).
* **Where the hops that lose the table come from.** A hop log (takeoff face, GF vs threshold, landing
  face) over 90 s: the plain fly hops twice on the table top and lands on it (GF 30-31 vs 30 -- the
  plain model walks at the escape threshold), then hops **from the table side face within a frame of
  walking over the edge** (GF 35) and lands on the floor; the CX fly's only hop is the same event (GF
  43 vs 38, from the side face, 0% of its time spent there). The first surface model flipped the pose
  by 90 deg in one 10 ms frame at every edge, so the whole scene swung in a frame -- an expansion
  transient the loom detectors are built to report. A real fly bends over an edge across a couple of
  body lengths. Now the face changes at once but the *sensed* orientation rotates about the edge axis
  over `FlyState.edge_len` = 4 mm of travel (0.2 s at 2 cm/s; Rodrigues interpolation of the body
  frame, used by vision, wind and olfaction). Same on concave corners.
* **The escape threshold, from data.** With the gradual edge the plain fly still hopped every ~10 s at
  GF 30-34 against `gf_hz` 30: the threshold was set in session 6 when the walking GF peaked at 26 Hz
  and the model has moved since. Measured over six seeds (20 s of walking each, hops disabled, then a
  loom, then a walk over the +x edge): per-second maxima of the walking GF -- median 20 Hz, 90th
  percentile 28, 99th 32, one outlier at 41.5; loom bursts 67, 57, 53, 41 Hz in the four seeds that
  see the ball and 21-23 in the two that do not (the ball comes from behind the fly: a stimulus fact,
  not a threshold one); edge crossings 21-24 Hz in four seeds (the gradual edge does its job) and
  60-63 in the two loom-miss seeds, where the passed ball was still next to the edge. Threshold sweep:
  30 -> 4/6 looms, 1.2 spontaneous hops per minute; 36-40 -> 4/6 looms, 0.6 / min; 42 -> 3/6 looms,
  0 / min. `Flight.gf_hz` is now 38: above the 99th percentile of walking, below the weakest loom.
* **The single-apple table: 0 meals in 9 runs, and why.** Plain, body program and CX module, three
  seeds each, fenced, starting 40 cm directly downwind of the apple: nobody found it. The programmed
  flies never reached "surging": the LH odour gate never crossed 0.6, so the search program's
  downwind drift walked them to the downwind edge and left them there; the plain flies climbed the
  fence (a glass wall is walkable) and sat on its rim. Three defects, all in the world / programs:
  (a) every fruit emitted odour at strength 1 regardless of size -- a 4 cm apple and a 6 mm blueberry
  were the same source, and the LH gate had been calibrated on the blueberries (23 Hz next to one)
  while the apple gave 15 Hz at 8 cm and nothing at 40; emission now scales with radius (strength 1
  at 2 cm: apple 2x, blueberry 0.3x, banana 4.5x); (b) the fence is now a fixture -- the legacy plane
  bounds for walking and a clamp for flight -- not a climbable box; (c) the offset response is an
  episode (20 s after the 10 s delay), after which the hungry fly does pure local search rather
  than a permanent downwind pull.
* **Unfenced sustain with everything above** (gradual edge, gf_hz 38; 5 min, seeds 0-2, meals / hops):
  plain 0 / 13, 1 / 7, 0 / 2; CX module + gating 0 / 2, 1 / 2, 1 / 2 -- every fly still ends on the
  floor with energy 0. Hops are down from 19-36 to 2-13 per five minutes, but two hops near an edge
  are enough, and a fly on the floor has no way back (a 16 m^2 random walk does not find a 5 cm leg;
  the animal would fly up). So the table-edge item ends here with a clear statement rather than a
  number: the physics is right, the remaining loss is (a) the loom pathway's response to self-motion
  (a few escapes per five minutes on a fruit table) and (b) the absence of goal-directed flight --
  both model questions. Foraging is scored on the fenced table from here on.
* **The plume was 4 cm above the fly.** Measuring the LH signal along the apple's plume axis
  (fly pinned, facing upwind) gave concentrations of 0.22 at 8 cm and 0.07 at 40 cm: the plume was
  centred on the apple's centre, 4 cm above the table, with a vertical width of sigma / 2 (1.2 cm at
  8 cm downwind), so a walking fly sat exp(-5) below the core. The blueberries only ever worked
  because they are 6 mm tall and there are six. Odour from a fruit on a substrate fills the boundary
  layer down to the surface: `Air` sources now carry the fruit's radius and the vertical offset is
  measured outside the fruit's extent, so a fly under the rim is on the axis. Concentrations become
  1.87 at 8 cm, 0.79 at 15, 0.34 at 25, 0.14 at 40, 0.08 at 55 (the RL env's sources updated too).
  The LH population then reads 7.2 / 7.0 / 6.6 / 5.8 / 5.1 Hz at those distances against 3.5 Hz
  plume-free -- graded, but a lone apple drives the population found on the mixed-fruit table far
  below its 10 + 12 Hz gate, which is why the single-apple runs never surged. The gate is re-derived
  from a screen on the single-apple table rather than re-tuned by hand (next entry).
* **The apple has its own lateral-horn population.** `scripts/screen_odour.py --fruit apple` (8 cm
  downwind into and away from the wind, 40 cm downwind, plume-free into and away, floor; 3,233
  populations): LHPD4d1 d' 4.51 -- 20.6 Hz at 8 cm, 12.5 at 40 cm, 3.4 plume-free; LHAV4a1_b 4.20
  (19.0 / 12.7 / 4.7); LHAV4a1_a 3.92 (21.9 / 12.7 / 4.5); LHCENT12_a 3.75; LHPD2a1 3.65; all
  heading-invariant, 100% / 0% at a midpoint threshold, and graded with distance. None of the
  blueberry-selected types is in this list except LHPD5c1, which is in both. The LH is odour-tuned
  -- the single "food population" was a simplification of the mixed-fruit table. `motor.LH_ODOUR_CHANNELS`
  now carries a berry and an apple population; `MotorRates.lh_odour` is a per-channel dict; the
  program's gate reads each channel against its own baseline (berry 10 + 12 Hz, apple 5 + 8 Hz from
  the screens) and takes the strongest. A third fruit would be a third screen, not a retune.
* **The optomotor readout was on the wrong cells.** On the single-apple table the surge went
  crosswind for both programs. The yaw decomposition of a walking fly in wind: the optomotor term
  (DNp04 + LPT27/30, L - R) mean |48| deg/s, std 73; DNa02 0; leg MNs 1; heading std 31 deg. Under
  sustained imposed rotation, pinned, no wind, those DNs do not flip (DNp04 1.4 / 3.5 Hz at +90
  deg/s, 0.6 / 3.4 at -90; identical on the pre-merge commit, so not a regression: session 5's
  benchmark rotated for 0.8 s and measured an onset transient), they do not respond to wind (0.3-1 Hz
  pinned, wind on or off), and walking drives them to 15-35 Hz asymmetries. T4 / T5 direction
  selectivity is intact (`probe_motion`: T5a DSI 0.38, T4c / T4d 0.23 / 0.24, correct preferred
  directions). So the term was self-motion noise injected as "stabilisation". With it off, the body
  program turns from crosswind to upwind in 6 s and closes 36 -> 19 cm in 20 s.
  `scripts/screen_rotation.py` (every DN / LPT / HS / VS type by side, +90 vs -90 deg/s sustained,
  pinned, no wind) finds the real rotation populations: HSN d' 4.2 (L - R -1.4 Hz CCW, +5.7 CW), DNp20
  3.1 (-4.5 / +7.8, the strongest in Hz), Nod1 2.8, LPT26 2.7, LPT50 2.4, HSE 2.2, VS 1.8; DNp04 0.3,
  LPT27 0.1, LPT30 0.5. The optomotor group is now DNp20 + HSN + HSE (the horizontal-system cells
  are the textbook optomotor neurons), sign from the screen (L - R grows under clockwise rotation, so
  + k * (L - R) opposes it). Checked: under imposed +-90 deg/s the commanded yaw opposes the rotation
  (-4 / +5 / +12 deg/s at 8 deg/s per Hz: loop gain ~0.1) -- but a walking fly in wind drives the same
  term to mean |62| deg/s, std 80: HS cells respond to translational flow as they do in the animal, and
  the plain readout cannot separate rotation from translation (the animal uses matched filters
  across the VS / HS population and efference copies). So `k_opto` is 0 by default: the group and sign
  are recorded for whoever builds the separation; a stabilising reflex that fires on the fly's own
  walking is noise. The body program steers upwind correctly with it off.
* **The compass circuit's weights, audited.** Effective strength under the model's rules (cap, path
  gains, same-type damping, fan-in normalisation -- none of which bite here: the compass cells have
  1.1-4.2k inputs, scale x1.00, and 10-30 synapses per pair, under the cap of 60), in mV per
  presynaptic spike per pair, with the total if the whole presynaptic population fired once:
  EPG -> PEN +5.0 (+81), PEN -> EPG +7.7 (+117), EPG -> PEG +3.5 (+74), PEG -> EPG +0.9 (+6), EPG ->
  Delta7 +3.2 (+134), Delta7 -> EPG -2.9 (-29, only 506 pairs), Delta7 -> PEN -4.7 (-45), ER -> EPG -2.9
  (-676 over 11,719 pairs), ExR -> EPG -1.7 (-26), EPG -> PFN +1.6 (+3), PFN -> hDelta +2.7 (+44),
  hDelta -> PFL3 +5.8 (+117). Threshold is 7 mV from rest. So the recurrent excitation is not weak --
  one PEN spike nearly fires an EPG -- and the restoring inhibition that confines a bump is: Delta7's
  total is a quarter of the EPG <-> PEN loop's, and the ring neurons, the compass's shaping input in
  the animal, are silent (0.08 Hz) because their own visual pathway is (TuBu 0.1, AOTU 0.1). That is
  the silent-or-seizure bistability seen in the gain experiment, explained: nothing lights the
  compass, and once lit nothing confines it. The experiment it prescribes: raise Delta7 -> EPG / PEN
  to match the loop and ask whether a *localised* bump persists.
* **A driven bump dies in 500 ms whatever Delta7 does.** Quarter of the EPG population driven at 60 Hz
  for 2 s: during the drive 12 / 50 EPG fire (the driven ones), PEN 0.1-1.0 Hz, Delta7 8-12 Hz; 0.5 s
  after the pulse EPG 0.1 Hz, PEN 0, Delta7 0.1 -- identical with Delta7 -> EPG / PEN x4, x8, and with
  the ring-neuron inhibition cut to a quarter. PEN never fires during the drive despite +5 mV per EPG
  spike, yet at EPG <-> PEN x4 the same loop ran at 300 Hz: the attractor's regime lies between
  those two, and one global setting stands on it -- spike-frequency adaptation (1.5 mV per spike,
  tau 200 ms, added in session 3 to tame runaway cliques) puts ~15 mV of hyperpolarisation on any
  cell sustaining 50 Hz, which is what a compass bump is; the animal's EPG hold persistent activity
  for tens of seconds. The next measurement is the bump with adaptation off.
* **Without adaptation the compass wakes up.** Same pulse with `adapt_jump` 0 for the whole brain:
  0.5 s after the pulse all 50 EPG are above 5 Hz (EPG 20.6 Hz, PEN 15.8, Delta7 22.5) -- the EPG <->
  PEN recurrence closes -- as whole-ring activity that flickers (5 / 50 at 1 s, 1 / 50 at 2 s, 16 / 50
  at 3 s) rather than a confined bump; with Delta7 -> EPG / PEN x4 as well, a *localised* 9-cell bump
  at 0.5 s that fades by 1 s. The rest of the brain rose 2.3 -> 4 Hz over 5 s without adaptation, so
  the attractor's regime is between Delta7 x1 and x4 with no adaptation in the compass.
  `LIFParams.adapt_by_type` ({type regex: mV per spike}, like `std_u_by_type`) makes that a
  per-population setting: the AVLP cliques that needed 1.5 mV / spike are not the compass, and the
  animal's EPG hold their bump for tens of seconds.
* **Compass-only adaptation off is not enough.** With `adapt_by_type = {"^(EPG|PEN|PEG|Delta7)": 0}`
  and Delta7 -> EPG / PEN at x1, x1.5, x2, x3, the driven bump is gone 0.5 s after the pulse in every
  case (a 9-cell flicker at 1 s for x1.5). So the persistence seen with adaptation off *globally* came
  from outside the compass -- the whole brain's rate rose 2.3 -> 4 Hz and that background drove the
  ring -- not from the EPG <-> PEN loop on its own. The compass in the animal rides on tonic drive
  (ExR, PEN, and others). The thread's axes are now mapped: no adaptation in the compass, tonic
  drive, Delta7 inhibition matched to the loop, and the recurrence gain; the pieces behave as
  expected alone (recurrence x4 -> 300 Hz seizure; adaptation off + background -> whole-ring
  activity; + Delta7 x4 -> a localised bump that fades in 1 s) and no combination tried so far gives
  a stable localised bump. Left here, with `adapt_by_type` kept as a feature (empty default) and the
  CX module as the working stand-in.
* **Single apple, corrected readouts: still 0 / 9, but now for a nameable reason.** With the
  boundary-layer plume, the apple LH channel and the optomotor term off, every programmed fly (body
  program and CX module, three seeds each, fenced) reaches 12-20 cm of the apple inside the first
  minute -- and then sits upwind and beside it (x 0.36-0.58 vs the apple's 0.25) for four minutes:
  the surge overshoots the source and upwind of a source there is no plume. The plain fly does not
  move towards it at all (0 / 3). The walking-fly answer to overshoot is the offset response --
  turn and walk downwind 1-2 s after odour loss, back into the plume, surge again -- and the program
  had it 10 s after the last hit, behind an 8 s flight-style cast, which next to a flickering source
  never fires. Offset delay 1.5 s, cast 3 s, from the walking literature rather than the flight one.
* **Retimed: still 0 / 6, and the failure is now the last five centimetres.** With the walking-fly
  offset timing, every programmed fly reaches 3.5-9 cm of the apple within 30 s (CX seed 0 again
  at 300 s) and runs past it: "surging" is upwind, and within a few centimetres of a source the
  direction to the source is no longer upwind (the fly ends level with or beside the apple, then
  explores upwind of it against the fence). The animal closes this gap with local search on the
  near-field and, plausibly, vision -- a 4 cm apple at 3 cm fills a large part of the eye, and flies
  fixate and approach dark objects. Whether the model carries an object-position signal is a screen
  question (`scripts/screen_object.py`: apple 5 cm ahead-left vs ahead-right, every DN / LC / VP type
  by side) before it is a program question.
* **No object-position signal in the model.** `screen_object.py` (apple 5 cm ahead-left vs
  ahead-right, pinned, no wind; every DN / LC / LPLC / LT / MeTu / AOTU / LAL / PVLP / AVLP type by
  side): the strongest L - R flip is LAL207 at d' 0.64 (2 cells, 1.2 Hz) -- nothing. LC10a, the
  object-tracking population of the animal (275 cells), sits at 0.02 Hz; LC16 0.12; DNa02 0.01. Like
  the compass's visual input, the object pathway is silent in the rate optic lobe + LIF model. So the
  last five centimetres cannot be read out of the brain as it stands: it is either a program
  (near-field klinotaxis on the bilateral antennae, which the animal also does) or an optic-lobe
  question (why LC10 is silent when LC4 / LPLC2 are not). Left as the second open model question
  beside the compass; the single-apple table stays at 0 / 6 with the approach to 3.5-9 cm recorded.
* **Native CUDA backends merged** (branch `perf/neural-execution`: fused LIF / optic kernels, warp-CSR
  products, native event traversal, native motor readout; `docs/PERFORMANCE.md`). The native motor
  readout now carries the per-channel LH populations, and the native LIF falls back to the Torch
  path when `adapt_by_type` is set (the kernel takes the scalar jump). First timing, *contended* (a
  compass experiment shared the GPU): eager 23.6 ms/frame, `--cuda-graphs` 12.6, `--cuda-kernels
  --cuda-graphs` 16.4, `+ --cuda-sparse warp` 11.3, `+ --event-driven` 11.3; spike trains diverge
  from eager at frame 36 (atomic accumulation order; documented as non-bitwise). A clean timing on
  an idle GPU follows before any of this is called a speed-up.
* **LC10, first look.** LC10a (275 cells, 910 input synapses per cell) takes 10% of its input from
  itself, 7% from TuTuA_2 (0 Hz), 6% AOTU042 (0.2 Hz), 6% Tm5Y (0.47 rate units), 4% LC9 (0), 4%
  LC10c (0), 3% TmY21 (0.44 ru): active optic-lobe units at the same level that feeds LC4 (T2 0.51,
  TmY3 0.46, Tm4 0.47), plus silent central cells. With a *static* apple 5 cm ahead of a pinned fly
  both LC4 and LC10a are at 0 Hz -- and LC10 is a moving-object detector in the animal, so the
  object screen showed it the wrong stimulus. Next: the same with self-motion (the apple sweeping
  the eye as the fly turns). [2026-09-13: do not read "small-object" as LC11's optimum here --
  **LC10a's own preferred width/height is 15-30 deg** (Schretter et al. 2024, Fig 3a); the small-object
  optimum (8.8 deg height, ~4.4 deg width) is LC11's. See "Session 10, interpretability toolkit".]
* `programs.KlinotaxisProgram`: near-field chemotaxis from the two antennae (bilateral contrast ->
  turn towards the stronger side; turn less while the odour rises) -- a subsystem approximation
  that reads the sensor, meant for offloading the olfactory brain, composable as
  `--program anemotaxis+klinotaxis`.
* **Tonic drive, and the balance read the right way round.** With compass adaptation off, Delta7 x2 and
  a held Poisson background on every EPG (5 / 10 / 20 Hz through `FlyBrain.stimulate` -- a direct
  `brain.set_poisson` is wiped when a pulse expires, which voided the first attempt), a driven wedge
  still dies within 0.5 s, and the reason is visible in one column: **PEN is 0.0 Hz throughout**,
  with EPG at 9-20 Hz and Delta7 at 33-58. EPG -> Delta7 is the circuit's strongest projection (+134
  mV in total) and Delta7 -> PEN is -4.7 mV per pair, so Delta7 clamps PEN before EPG -> PEN can close
  the loop: the inhibition is too strong relative to the recurrence, not too weak as the totals
  suggested. The attractor's regime is a small grid -- EPG <-> PEN gain against Delta7 gain, no
  adaptation, tonic background -- scored by wedge persistence versus the rest of the ring.
* **LC10 with self-motion: still silent.** Heading oscillating +-20 deg at 0.5 Hz so the apple
  sweeps the eye, 5 cm ahead-left / ahead-right / absent: LC10a 0.02 Hz in all three; LC10b 1.4-2.2
  with or without the apple; nothing object-specific anywhere in the LC10 group, AOTU042 or TuTuA.
  Since the stimulus is now the right kind, the question moves to the optic -> spiking interface:
  spiking cells are driven by *changes* in optic-lobe rate through a uniform 100 mV output gain
  spread over each cell's inputs, and a small sweeping object may move LC10a's Tm5Y / TmY21 inputs
  by too little to matter where a loom moves LC4's T2 / TmY3 / Tm4 a lot (and gets a x3 pair gain).
  Measured next: the drive LC10a actually receives, apple vs none.
* **LC10: not a gain problem.** The drive LC10a receives from the optic lobe during the sweep is
  identical with and without the apple (mean -0.02 vs -0.00 mV, peak 11.9 vs 11.9 -- brief 1 ms peaks
  that a 20 ms membrane integrates to nothing), and so is its inputs' activity: |delta-rate| Tm5Y
  0.194 vs 0.193, TmY21 0.207 vs 0.206, T2 / TmY3 / Tm4 / Tm20 the same. The whole textured room
  sweeping the eye moves every Tm / TmY population by +-0.2 rate units; a 4 cm apple adds nothing
  measurable on top. So the rate optic lobe carries no small-object signal at the level of LC10's
  inputs -- the animal's object pathway rests on small-field selectivity (medulla / lobula surround
  inhibition through Dm / Pm / Li interneurons) that the L2-normalised rate model evidently washes
  out. An optic-lobe modelling thread, not a per-type gain; the second open model question beside
  the compass, now with a precise statement of what is missing. In the meantime the last five
  centimetres is `KlinotaxisProgram` (a sensor-side approximation, off by default).
  [2026-09-12, dynamics round 1: the per-type drives are now measured in the deterministic lobe
  (`gain_fb 0`, where the none-vs-none null is 1e-9-3e-7 mV): **LC11 +0.046 mV and LC10a +0.080
  against LPLC2's +0.54 and a 7 mV criterion** -- so the gap is ~10x on the LC cells that should
  carry the object, not a scaling nudge. The figure is already gone one stage earlier, at T2 / T3 /
  Tm5Y / TmY21, by ON/OFF cancellation, and is then diluted by l1 pooling at LC10 / LC11.]
* **The compass grid** (no compass adaptation, 10 Hz held background on every EPG, a 12-cell wedge
  driven at +40 Hz for 2 s; cells above 22 Hz in the wedge vs the other 38, at 0.5 / 2 / 5 s after):
  EPG <-> PEN x1.5 -- the ring stays at the background (EPG 10-13 Hz, PEN 1-5), wedge 0-3 / rest 2-10,
  no bump; x2 -- the whole ring is self-sustaining *before* the pulse (EPG 57-74 Hz, PEN 51-76, Delta7
  105-127), wedge 5-6 / rest 8-12 throughout; x3 -- 160-178 Hz whole-ring. Delta7 at x0.5 vs x1 barely
  changes any row, and the rest of the brain stays at 1-4 Hz (the seizure does not spread: adaptation
  elsewhere holds). So the transition from sub-threshold to whole-ring is sharp between x1.5 and x2
  and nowhere in this grid is there a confined bump: the inhibition that should let only one wedge
  win is weak relative to the recurrence at every gain tried. Last corner: x2 with Delta7 x2 / x4 / x8.
* **... and the last corner:** with EPG <-> PEN x2 (self-sustaining ring), Delta7 x2 and x4 leave it
  whole-ring at ~55 Hz with the driven wedge *less* active than the rest (4 / 12 vs 8-13 / 38), and x8
  suppresses the ring back to the background (EPG 10-15 Hz) without favouring the wedge either.
  Delta7's inhibition acts globally and never lets one wedge win. So: no confined, persistent bump in
  any cell of (recurrence x1.5-3) x (Delta7 x0.5-8) x (no compass adaptation) x (10 Hz background). The
  connectome has the wedge-specific Delta7 -> EPG structure the animal's attractor uses; under
  uniform 0.275 mV synapses, the cap and the L1 fan-in scaling it does not produce winner-take-all.
  The compass thread ends this session with that map; `cx.CompassSteering` stays the stand-in.
* **Native backends, clean timing** (idle 4090, 300 frames of 10 ms, seed 0, full brain, B = 1):

  | configuration | ms / frame | x real time |
  |---|---|---|
  | eager torch | 22.0 | 0.45 |
  | `--cuda-graphs` | 15.8 | 0.63 |
  | `--cuda-kernels --cuda-graphs` | 16.9 | 0.59 |
  | `--cuda-kernels --cuda-graphs --cuda-sparse warp` | 13.9 | 0.72 |
  | **`--cuda-kernels --cuda-graphs --event-driven --cuda-sparse warp`** | **8.0** | **1.24** |
  | ... + `--dt-by-module vnc=1.0` (without events) | 13.5 | 0.74 |
  | ... + `--weight-dtype float16` (without events) | 13.5 | 0.74 |
  | `--fast --cuda-kernels --cuda-graphs --cuda-sparse warp` | 8.9 | 1.13 |

  The native event traversal is the win (full dt, full brain, real time and a quarter); the fused
  kernels alone are not faster than Torch graphs, and `--fast` buys nothing on top of events. Spike
  trains diverge from eager after ~35 frames (atomic accumulation order; documented as non-bitwise).
  The sustain runs below use the event configuration. `probe_sustain.py` takes the backend flags.
* **Shiu et al. 2024's sugar / bitter result, on MaleCNS** (`scripts/probe_bitter.py`: the 165 labellar
  sweet GRNs and the 47 bitter GRNs from `taste_grns.csv` driven at 100 Hz, LIF alone, 1.5 s):

  | | sugar | sugar + bitter | bitter alone |
  |---|---|---|---|
  | Shiu's rules (uniform 0.275 mV synapses, no adaptation / cap / damping / fan-in scaling) | MN9 123.5 Hz | 2.1 Hz | 0.0 Hz |
  | this project's calibration | 4.6 Hz | 0.0 Hz | 0.0 Hz |

  Shiu reported 78 -> 3 Hz on FlyWire (female). On MaleCNS their rules give 124 -> 2: the proboscis
  motor neuron fires hard on sugar and bitter shuts it off. The calibrated model keeps the sign and
  the suppression at a tenth of the amplitude -- the price of the anti-runaway measures the whole
  brain with sensory input needs (session 3); the two-parameter-set comparison is the honest way to
  quote it.
* **Final sustain runs of the session** (`probe_sustain.py`, 5 min, energy 0.4 at start, native event
  backend; seeds 0 / 1 / 2 as meals, hops):

  | configuration | full table (unfenced) | single apple, fenced |
  |---|---|---|
  | plain model | 1, 0, 1 meals (3 / 4 / 2 hops) | 0, 0, 0 (closest 18 / 7 / 18 cm) |
  | anemotaxis program + gating | 0, 0, 0 (2 / 1 / 1 hops) | 0, 0, 0 (closest 12 / 9 / 6 cm) |
  | CX module + gating | 0, 0, 0 (1 / 1 / 1) | 0, 0, 0 (closest 12 / 9 / 8 cm) |
  | anemotaxis + klinotaxis | 0, 0, 0 (2 / 1 / 1) | 0, 0, 0 (closest 8 / 8 / 6 cm; klino mode 65-67%) |
  | CX + klinotaxis | 0, 0, 0 (1 / 1 / 3) | **0, 1, 1 meals** (closest 17 / 1.4 / 1.5 cm; seed 1 ended at 0.07) |
  | CX alone, the mis-dispatched "CX + klinotaxis" batch | -- | **1, 0, 1 meals** (closest 0.8 / 8.6 / 1.4 cm; seed 0 ended at energy 0.38) |

  (The composite rows are the rerun after two bugs: the first batch crashed the anemotaxis composite
  on an array-valued antennal sum and dispatched the CX composite without antennae.) Three readings.
  The klinotaxis subsystem is what closes the last centimetres for the CX fly (2 of 3 seeds fed, 1.4
  and 1.5 cm closest approach) but not for the body program (6-8 cm): the CX module's steering is
  expressed through DNa02 and the leg motor neurons, which the antennal contrast term adds to, while
  the body program's yaw injection swamps it. On the unfenced table every programmed fly hops off once or twice in five minutes
  and cannot return (the plain fly gets its meals by wandering into fruit before it leaves), so the
  full-table row measures the edge problem, not foraging. On the fenced single apple the CX module
  reached the apple and fed in 2 of 6 runs (the extra three came from a dispatch bug that ran plain
  CX with a different RNG stream -- useful as replication), which is the first success on a lone
  source; the body program's closest approach was 6 cm. Every fly starts hungry and is starving by
  ~110 s, so a first meal has to come inside two minutes.
## Session 9: improving the model itself

* **LPi -> LPLC2: the missing inhibition behind the self-motion escapes.** Structure first: under the
  uniform synapse the six T4 / T5 subtypes give each LPLC2 +414 synapse-equivalents of excitation and
  the two LPi types (glutamate, sign -1: correct) only -40 (LPi43 -32, LPi34 -8), while the LPi are
  themselves driven hard (T4c / T5c -> LPi34 +650 / +780 per cell). In the animal that inhibition is
  what makes LPLC2 expansion-selective. A pair-gain sweep on LPi34 / LPi43 -> LPLC2 (pinned fly,
  no wind; then the loom; then 15 s of free walking; native backend, contended):

  | gain | GF under +-90 deg/s rotation | loom GF peak | walking GF per-second max: median / 90% / max |
  |---|---|---|---|
  | x1 | 6.6 / 7.4 Hz (max 24) | 28.0 | 18 / 23 / 28 |
  | x4 | 3.7 / 2.9 (max 18) | 32.5 | 11 / 17 / 19 |
  | x10 | 0.9 / 1.1 (max 9) | 20.4 | 6 / 9 / 15 |

  x4 cuts the self-motion drive on the giant fibre and raises the loom peak; x10 overshoots. This is
  a model change with a physiological basis, not a readout, and it is the first that separates
  looms from walking at the source instead of at the threshold.
* **Where the optic lobe loses a small object -- found.** `scripts/probe_figure_ground.py`: pinned fly,
  apple 5 cm ahead-left, heading oscillating +-20 deg (self-motion), the same scene with no fruit; for
  every optic type the time-averaged |delta-rate| in the 159 columns that view the apple minus the
  same in 1,050 background columns, apple minus none, as a z-score against the background scatter.
  (Only 19,817 of 89,390 optic cells carry a hex column in the annotations; the rest inherit the
  column of their strongest input partner in three passes, 89,380 assigned.) The object is
  represented retinotopically in the medulla: Mi4 z 7.4 (0.14 vs 0.10 rate units in its columns --
  a 40% modulation), Tm3 4.0, Tm9 3.7, Mi9 3.6, L1 2.2, L3 1.7, and Tm1 / Tm2 carry it with the
  opposite sign (-3.9 / -2.4: the dark apple lowers their transients). It is gone one stage later, in
  the types that feed LC10a: Tm5Y z 0.5 (a 4% modulation), TmY21 -0.2, TmY3 -0.7, T2 -0.8, T3 -1.3.
  Per cell: the LPLC2 cells viewing the apple's direction receive +3.8 / +3.3 / +3.1 mV of extra
  drive (the loom detectors see a stationary-relative object), LC10a's best cell +0.85 mV against a
  7 mV threshold. So the small-object pathway is lost between the medulla and Tm5Y / TmY21 -- one
  synaptic stage -- and the population-mean screens could never have seen it. Next: is that stage
  a weight problem (the figure-carrying inputs are a small fraction of Tm5Y's normalised input) or
  a dynamics problem (the rate model flattens it)?
* **The signed figure, and where the object pathway loses it -- structurally.** Signed (apple - none)
  mean delta-rate per type, apple columns minus background: the dark object *lowers* the ON channel
  (Mi1 z -7.6, Tm3 -8.3) and *raises* the OFF channel (L2 +10.2, Tm1 +3.0, Tm2 +3.4, Tm4 +5.3), the
  lamina (L1 +12.8, L4 +4.4) and the Dm / Pm interneurons carry it (Pm5 +6.5, Dm16 +5.8, Dm12 +4.1,
  Pm1 -6.5) -- a physiologically coherent contrast figure. A linear estimate from each type's inputs
  (signed L1 input fraction x the input type's signed figure) recovers 30-40% of Tm3's and Tm4's
  own figure and ~15% of Tm20's: the rate dynamics amplify what the wiring delivers, 3-6x. At Tm5Y
  the estimate is -0.0002 against an own figure of +0.0002 -- its inputs (Tm20 13%, Li19 9%, Y3 7%,
  Tm32, Tm5a, T2a; photoreceptors 0.1%, unlike Tm5a / b / c at ~5%) deliver nothing to amplify;
  TmY21 (TmY5a, TmY13, Tm20; photoreceptors 0%) the same. So LC10a's input types are not fed by the
  contrast channels that carry a static figure in this connectome. The retina's contrast stage shows
  the apple as a 0.2-0.4% offset in its columns (adaptation removes a static object and leaves its
  edges). The stimulus LC10 is built for is an object moving *relative* to the background; that is
  the test that decides whether the pathway is absent or merely untested.
* **Relative motion: the small-object pathway is absent, not untested.** Pinned fly, empty table, a
  1 cm black ball 5 cm ahead sweeping 12 cm laterally in 3 s (~45 deg/s across ~11 deg of the eye --
  a textbook LC11 / LC10 stimulus), vs the same scene without the ball, drive per cell over 9 s:
  LC10a max cell +0.19 mV mean (+1.4 mV peak), LC11 +0.09 (+1.9 peak), LC10b +0.21 (+2.1), LC16
  +0.32 (+3.4) -- all against a 7 mV threshold, and their rates 0.0-1.3 Hz with or without the
  ball -- while the same object drives LPLC2 to +9.8 mV peak (0.09 -> 0.20 Hz). So the lobula's
  small-field object-motion channels (T2 / T3 / Tm -> LC11 / LC10) do not carry a small moving
  object in the rate optic lobe, whereas the wide-field loom channel does. This is the model item:
  the medulla -> lobula small-field stage, with `scripts/probe_figure_ground.py` (static figure,
  retinotopic) and this sweep as its two benchmarks. The escape-hop side of the same optic question
  is fixed at the source (LPi x4 above); the object side is not a gain.
* **LPi x4 is the default; the escape threshold re-derived from it.** Six-seed protocol (20 s of
  walking with hops disabled, then a loom, then a walk over the +x edge): walking per-second GF
  maxima median 12.1, 90th percentile 18.6, 99th 22.5, max 28 (were 20 / 28 / 32 / 41.5); loom peaks
  38, 42, 15, 34, 56, 52 -- five of six seen (four before). Threshold sweep: 30-33 -> 5 / 6 looms, 0
  spontaneous hops per minute; 36-38 -> 4 / 6. `Flight.gf_hz` 38 -> 33. The edge crossing became the
  dominant spurious trigger (27-49 Hz at the 4 mm edge rotation, 32-43 at 10 mm: a 90 deg sensed
  rotation over 0.2-0.5 s is a 180-450 deg/s sweep that LPi inhibition does not cancel), and at 20 mm
  -- ~90 deg/s at walking speed, the range the visual system operates in -- it is 18-23 Hz, below
  threshold on all six seeds. `FlyState.edge_len` 4 -> 20 mm. In the walking demo the change gives
  walking GF maxima 16 / 18 / 18 (were 34 / 17 / 25) and no hops in 60 s of wandering on the full table.
* **The looming fork** (`docs/BENCHMARK_BATTERY.md` assays 2a / 2b / 3; `start_loom(speed, radius, final)`):
  a 3 cm ball from 0.5 m at 2.0 / 1.0 / 0.5 / 0.25 / 0.12 m/s (l/v 15-250 ms), three seeds, peaks over
  a 3 s pre-loom baseline with the peak's time relative to arrival at 3.5 cm. GF +37 / +41 / +31 / +36
  / +27 Hz at +120 / +140 / +87 / +47 / +70 ms -- the same response at every rate, timed to the final
  expansion; with takeoff enabled the body escapes (short mode) in 13 of 15 runs, at +30 to +150 ms.
  Wing power +41 / +46 / +47 / +49 / +53 Hz at +490 / +457 / +113 / -923 / -1993 ms: larger and earlier
  for slower looms, the long mode's wing-raise signature, but never a mode switch because the GF
  fires anyway. Leg MNs +3.5-4.8 Hz throughout: no landing response. So the model's giant fibre
  encodes final size (the LC4 side), not expansion rate (the LPLC2 side) -- consistent with LPLC2
  sitting at 1-4 Hz throughout this session's measurements. The discriminating condition for
  landing is an approach that stops short.
* **Stop-short approaches** (the landing condition: the ball halts at 8 / 10 / 15 cm, final size 23-41
  deg, l/v 60-250 ms, two seeds): GF +7.5 to +15 Hz over baseline and no takeoff in any run -- the
  short-mode escape is gated by final size in the right direction; wing power +35 to +58 Hz without a
  jump -- a wing-raise-like preparation to a gentle approach, which is the long mode's first half;
  leg MNs +2.7 to +5.1 Hz -- no leg extension. Battery: 2a pass, 2b partial (wing preparation scales
  and precedes, but no GF-independent takeoff and the GF still fires for slow looms that reach the
  fly), 3 fail (no landing leg response).
* **Clean timing** (headless demo loop, 300 frames of 10 ms after 60 warm-up, one configuration at a
  time on an idle RTX 4090, B = 1, full brain):

  | configuration | ms / frame | x real time |
  |---|---|---|
  | eager (default) | 21.3 | 0.47 |
  | `--cuda-graphs` (neural frames + sensory rays) | 14.5 | 0.69 |
  | `--cuda-graphs --weight-dtype float16` | 13.7 | 0.73 |
  | `--cuda-graphs`, neural capture only (no sensory capture) | 20.8 | 0.48 |
  | `--fast` (brain dt 1 ms, optic dt 2 ms, camera 1/4) | 15.5 | 0.64 |
  | `--fast --cuda-graphs` | 9.3 | 1.07 |

  Nearly all of the gain is the captured ray tracing (the ~20 ray kernels per frame were
  launch-bound); capturing the neural frame itself saves ~0.5 ms. Determinism: with CUDA graphs the
  spike counts match the eager path for the first 136 frames and then differ by one spike (chaos
  amplifying a ~1e-6 rounding difference; the same fly, not bit-exact); float16 differs from frame 0.
  Both stay opt-in; the probes and the sustain test run eager.

### Session 9 addendum: headless runs move to the cluster

From now on GPU sweeps, screens and the benchmark suite run on the group's B200 cluster through its job
manager rather than on this desktop; the recipe, addresses and the first results live in `docs/CLUSTER.md`
(git-ignored on purpose: infrastructure stays out of the public repo). `scripts/cluster_run.py` is the
committed helper: it ships the local working-tree changes to a fresh per-run copy of the cluster checkout,
submits one job per command (several commands in one call run concurrently and pack onto a GPU by VRAM
budget), waits, prints the logs and fetches results back. The workflow agents of this session were
reloaded with the instruction to use it instead of the local GPU.

First cluster run of the fenced single-apple sustain probe (seed 0, `--program cx`, fast configuration,
`probe_sustain.py`), which by a scheduling accident ran twice concurrently on two GPUs: 5 min, meals 0, final
energy 0, hops 0 in both; path 3.78 m (exploring 52 %, surging 19 %, casting 17 %, searching 12 %) and
3.84 m (exploring 69 %, surging 18 %) -- the same seed giving two different trajectories is the run-to-run
nondeterminism of the native fast configuration (float atomics in the event-driven kernels), worth
remembering when comparing single runs. 9.6 min wall-clock, i.e. 1.9x slower than real time, against 1.24x real time on the local RTX 4090. That number includes the first-time compilation of the native kernels for sm_100 and two other jobs sharing the GPU for part of the run, so it is not yet a clean B200 timing; the sim is latency-bound (small kernels per 0.5 ms step), which is exactly the regime where a datacentre GPU does not beat a desktop one and where batching flies per process would pay. Same
outcome class as the local runs (session 9: CX fed in 2 of 6). The smoke job (control / cuda / world
tests) passed on the B200: 25 passed, 10 skipped.

Follow-up worth doing next on the performance side: a **BatchSim**. The brain (`Brain`, `OpticLobe`,
`FlyBrain`) is batched over B flies and `env.FlyRoomEnv` uses that, but the demo `Sim` -- body physics,
surfaces, metabolism, programs, the odour / wind / taste sensing -- is written for one fly, so a seed
sweep today is B processes packed onto a GPU (each at ~20 % of a B200's memory). Vectorising the body
over B would let one process run a whole sweep through a single batched brain step, which is roughly
B times cheaper than B processes; the warp-CSR path is batch-1 only, so the batched configuration would
use cuSPARSE (`--cuda-sparse torch`) with kernels + graphs.

### Session 9 addendum 2: the audit workflow (NT signs, compass structure, benchmark suite, anti-runaway retirement)

Nine agents (three audits, one retirement study, four skeptics, one critic; the second half on the cluster) --
reports in `docs/audits/`, every number below survived its skeptic unless marked.

**The compass circuit does support a ring attractor** (`docs/audits/cx_wedge.md`, `scripts/cx_wedge.py`, verified
bit-for-bit on the B200 by `scripts/cx_wedge_verify.py`). Session 8's "no confined bump in any gain grid" was a
consequence of where the Delta7 gain was applied. Structure, from the wedge identities (46 EPG in PB glomeruli
L1-L8 / R1-R8, 42 PEN, 18 PEG, 42 Delta7 whose instance names are their output glomeruli; ring order L1 R8 L2 R7 L3
R6 L4 R5 L5 R4 L6 R3 L7 R2 L8 R1): two-step EPG x EPG excitation through PEN is wedge-local (84 % within +-2 wedges),
Delta7 inhibition is cosine-shaped (own-wedge / opposite 0.10), and PEG is 20x weaker. What the grid missed is an
**untuned global feedback loop EPG -> ExR6 / ExR4 / ER6 / ER4m -> EPG, PEN** (two-step -2,800 to -3,200 mV^2 per
wedge at every distance, 7x the peak Delta7 inhibition): with Delta7 cut, a PEN next to a driven bump still nets
-7.7 mV from it. So at x1 this loop, not Delta7, shuts the recurrence, and gaining Delta7 -> PEN (the session-8
convention) clamps the bump's own PEN. Working settings (full connectome, compass adaptation 0, 10 Hz EPG
background, 4 wedges driven at +40 Hz for 2 s, 5 s free): EPG <-> PEN and EPG <-> PEG x2 with **Delta7 -> EPG x15-25
(Delta7 -> PEN x1)** gives a confined bump that persists for the full 5 s (driven wedges ~200 Hz, 11/11 cells above
22 Hz, 1/35 outside, vector strength 0.76; PEN 49, Delta7 101 Hz; rest of brain 0.03 Hz), captured by the pulse from
wherever the spontaneous bump had formed, in seeds 0-2; gE 1.75 also works, gE 1.5 is metastable, gE 1 dies, gD 60
drifts, gD 8 or gE 2.5 give a bump too stiff to relocate. Alternative: ER / ExR -> EPG / PEN / PEG damped x0.3 with
Delta7 -> EPG x4-15 at recurrence x1. Caveats the skeptic added: the bump fires 200-260 Hz per cell (refractory-
limited, 10x the animal); capture is not always exact (one seed lands one wedge off); nothing beyond 5 s was
sampled; several "capture" rows were really persistence because the seed's spontaneous bump had nucleated at the
driven wedges; and the whole loop hinges on four ExR4 / ExR6 cells (predicted glutamate) whose fast-vs-modulatory
role in the animal is unknown. The bump has not yet been run with sensory input, shown to move with a PEN L / R
asymmetry, or shown to reach PFN -> hDelta -> PFL3 (structurally PFN gets EPG +2.8 vs Delta7 -13.5 mV per cell with
every nodulus input but IbSpsP inhibitory) -- these are the next experiments. [2026-09-12, answered in dynamics
round 1: all three. With full senses the bump persists for the whole 38 s free window in 192/192 flies (12 gain runs, 11 configurations) at all
three operating points (219-261 Hz), but it is pinned to 5-7 attractor sites and tracks neither heading nor
anything else. A 1 s unilateral 40 Hz PEN drive moves it +0.038 +- 0.018 wedges with GLNO silent and
+0.301 +- 0.036 with GLNO relabelled GABA (6/6 seeds, sign-flip p 0.031) -- an elastic deflection that relaxes
when the drive stops, not an integrated heading update. PFN stays at 0.59-0.78 Hz and hDelta at 1.47-1.86 Hz
under a 220-260 Hz bump, so PFN -> hDelta -> PFL3 is NO. See "Session 10, dynamics round 1".]

**NT signs** (`docs/audits/nt_audit.md`, `scripts/audit_nt.py`, byte-reproducible on CPU): sign 0 silences 2.2 % of
synapses (unknown 0.7 %, DA / OA / 5-HT 0.5 % each), landing on the mushroom body (9.8 % of input, 12.5 % of output)
and the octopaminergic visual centrifugal cells (17.6 % of their output), while every optic, loom, optomotor,
antennal-lobe and DN population loses under 1.5 % of input and none of its output. So the sign-0 convention is not what
silences the compass or the object pathway. One link to test: GLNO (4 cells, all NT columns unclear, T-bars glutamate
51 % / ACh 37 %) is 19.4 % of PEN's raw input and is itself 41 % driven by PEN -- a silent PEN <-> GLNO loop. Two
sign disagreements rather than gaps: Tm5Y is ACh at confidence 0.96 (the "glutamate" was the audit's own unsourced
table entry), MN9 ACh vs glutamatergic motor neurons (726 output synapses; irrelevant). A type-majority rule would
give an NT to 496 of the 1,838 unknown presynaptic cells (25 % of the silenced synapses) at much higher confidence
than T-bar votes. The receptor-expression integration (`docs/NT_INTEGRATION.md`) remains the principled route.

**Benchmark suite** (`scripts/benchmark.py`, `docs/audits/benchmark_suite.md`): 14 sections, one entry point,
`REFERENCES` dict with the session that set each value, JSON for regression diffs; 3.8 min on the 4090, 5.3 min on a
shared B200; 24 pass / 3 fail / 2 known gaps at the defaults. Skeptic's corrections applied: `loom_escape` had run
against a stale body.py (gf_hz 38) -- at HEAD (33) the reruns give 2/2, 1/2 and 4/6 escapes with peaks 29-42 Hz, so
the check passes but is bistable at the threshold (`--seeds 0..5`); the motion DSI line now quotes measured values
(T4a 0.17 ... T5a 0.40, identical across 4090 and B200); min-DSI reference 0.16. Checks that do not discriminate and
should be repaired next: `loom.escape_cm`, `dn.DNp09_top_hz` / `MDN_top_hz` (report the driven type), `walk.power_max_hz`
(bistable per-frame transient), the legacy `rotate.DNp20_flip_hz` (2x scatter), and the object / compass sections
(population means at defaults, 0 in all 45 runs by construction).

**Anti-runaway retirement** (`docs/audits/anti_runaway.md`, `scripts/retire_measures.py`; 26 configurations, three
replicates, "broken" only when it fails in every replicate): adaptation is needed (6 breaks without it) but
`adapt_by_type` -- 0 mV in the CX columnar cells, ring neurons, motor neurons and GF, 1.5 mV elsewhere -- **breaks
nothing** (26/1; this is also the adaptation half of the compass fix); the cap is needed (DNa02 asymmetry, MDN storm,
odour gate), a soft cap 120(1-exp(-n/120)) breaks nothing; same-type damping x0.1 produces no replicated break when
removed but KC rates rise 2.9 -> 12 Hz and loom GF doubles, and the per-type list passes only without margin; fan-in
scaling is needed (walking GF p99 58 Hz without it), (5000/total)^0.5 passes; AL depression is needed (odour gate 92 Hz
without it); DN -> VNC x3 breaks nothing when removed but the DN motor maps collapse 4x; **the GF x0.3 input damping
breaks nothing when removed** and four of its five targets are inhibitory, so it goes (retracted in round 3 of the receptor
integration, reinstated in round 4 under the corrected benchmark, adopted in round 5 -- see session 10). Adoptions are one at a time
with a suite run each (`all_replacements` together breaks walking GF p99).

**Three bugs found by the study, fixed here**: (1) `brain._shaped_weights` with `conn_cap 0` aliased the shared
connectome and multiplied the path gains into `c.W` on every build (121.4M -> 125.6M -> 137.9M abs sum) -- it now
always copies; the local `no_cap` and the earlier Shiu-rules numbers in the retire study were contaminated (the cluster
replicate, built with a guard, was clean: Shiu 123.5 / 2.1 unchanged). (2) The PN entry in `DEFAULT_STD_U_BY_TYPE`
never matched a cell (patterns are applied with `re.match`), so the model has always depressed ORN and LN terminals
only; the entry is removed and the docs corrected -- no behavioural change. (3) `room_demo.update_loom` placed the
ball along world +y, which is the fly's left only at heading 0; in free-walking loom sections the stimulus azimuth
therefore depended on the seed's heading (session 9's two "ball not seen" seeds). It now approaches along
`fly.left`; the free-walking loom numbers above predate the fix and should be re-measured.

**Critic's follow-ups, in order**: run the two compass settings through the full suite with senses (does a 200 Hz
bump spin the walking fly through PFL3 -> DNa02? [2026-09-12, answered in dynamics round 1: **no**. The bump sets a
static PFL3 L-R offset of 2.09-3.31 Hz -- a fixed anatomical gradient the gains scale rather than create, the
shipped control reproducing the same phase at 0.19 Hz -- DNa02 follows at 0.08-0.24 Hz against a within-fly
temporal sd of 1.0-1.3 Hz, and the yaw first harmonic is 0.006-0.021 rad/s = 0.3-1.2 deg/s against a heading
circular sd of 20-31 deg. Statistically detectable at the two higher gains, behaviourally nothing.]); PEN L / R
shift and the PFN / hDelta / PFL3 readout; the GLNO sign;
T2 / T3 as ON-OFF units (their inputs sum Mi1 / Tm3 (figure z -8) and Tm1 / Tm2 / Tm4 (+3 to +5) linearly and cancel
the object exactly where NOTES 9 loses it; T3 is 21 % of LC11's input) via a rectified baseline in `optic.py`, scored
with `probe_figure_ground.py` and the sweeping-ball assay [2026-09-12, answered in dynamics round 1: the
cancellation is **confirmed structurally** -- T3 sums Mi1 22.5 % + Tm3 8.2 %, lowered by the dark object, against
Tm1 16.0 % + Tm4 6.7 %, raised, through excitatory synapses of equal weight, a linear estimate of -0.0004 against
its own -0.0011 -- but a rectified baseline is not the fix: round 3's `--rectify-t2t3` runs already left LC11 and
LC10a at the null (FAIL in 4 of 4, `docs/audits/object_sweep.md` section 2). The levers left are
physiology parameters, not a rectification]; the self-motion GF assays that decided LPi x4 and
edge_len 20 mm added to the suite, and the optic layer's own hand-crafted measures (five pair gains, gain_out, L1
normalisation, drive clip) audited the way the LIF's were; the odour gate at the 40 cm foraging start (channel 12.5 Hz
against a 13 Hz gate) and cross-channel specificity; an NT-rescue counterfactual through the LIF sections.

## Session 10: receptor-expression integration, round 1

The plan of `docs/NT_INTEGRATION.md` was executed end to end by a 17-agent workflow (GPU work on the cluster
through `scripts/cluster_run.py`): five transcriptomic / typing sources acquired and pinned (55 files,
`flyverse/data/manifest.json`, `scripts/fetch_data.py --external all`), mapped to MaleCNS type names with
confidence tiers, a per-type (transmitter x receptor) -> sign / gain table derived (607 profiled types), an optional
model stage `LIFParams.receptor_model` implemented (off byte-identical; `sign` / `sign+gain` / `full`), scored on the
benchmark suite and the optic probes, and adversarially verified. Section 7 of `docs/NT_INTEGRATION.md` has the
detail; `docs/audits/receptor_verification.md` the skeptics' text.

The scientific result is null and informative. (1) Coverage: the receptor route decides 25.7 % of synapses, and
almost none of the populations behind the open questions -- Tm5Y, TmY21, LC11, DNp04 / LPT, DNa02, PFL3, hDelta, the
sweet interneurons -- have a receptor profile in any source; the optic lobe's object pathway is out of reach of
the present data. (2) Where the data do reach, they confirm the current sign for the loom, optomotor and compass
populations (LC4 / LPLC2 / LPi34 / T4 / T5 / HSN / HSE / EPG all GluCl-dominant for glutamate; the ExR -> EPG loop
that `cx_wedge` found is fast inhibition by the E-PG driver data) -- the compass remains a gain / dynamics
question. (3) The one structural change the data propose, glutamate as excitatory onto medulla cells with NMDA /
kainate receptors (Mi4, L3, Mi9, T2a), moves the figure signal onto Mi4 (z 0 -> 3) but not down the pathway, and
breaks the Shiu-rules sugar -> MN9 replication; and 59 % of those flips are tertile-rank calls that the
absolute-level rule does not make, so the effect is not yet a finding. (4) The slow term as built is a
classical-metabotropic current rather than the monoamine tone the plan wanted, and storms at its first-guess
scale; it needs the per-class redesign before assay 7 (hunger) can be attempted.

Errors caught by verification: Davis 2020's KaiR1D column was a different gene (CG8916); single-nucleus dropout
profiles were allowed to silence ~180 k synapses on the photoreceptor -> medulla stage; Pm1 / Tm29 / typing
tier inflation; misquoted ranges. None affects the current default model (the stage is off). Round 2 fixes the
tables, scores the other net rules, runs the moving-ball object sweep, searches for the missing optic profiles,
applies the supported transmitter labels (TmY14 glutamate, Mi19 serotonin, aMe8 ACh; T1 histamine unsupported)
as an override table, and tests the GLNO sign in the compass.

Infrastructure notes from the round: `cluster_run.py --fetch` now accumulates; two budgeted jobs starting in the
same second once came up without CUDA (a node-agent race, reported); the round used ~5.0 M agent tokens.

## Session 10: receptor-expression integration, round 2

Round 2 (17 agents; skeptics on Opus) applied every round-1 table correction, added Kurmangaliyev 2020 as a sixth
source, rebuilt the receptor table under a stricter silencing rule (a synapse is silenced only when >= 2 sources
agree and no whole-cell profile has the receptor on: 286,600 -> 83,473 synapses), and ran the experiments the plan
had left open. `docs/NT_INTEGRATION.md` section 7 has the round-2 outcome in full; the verification record is in
`docs/audits/receptor_verification.md`. The five results:

1. **The small-object hypothesis is closed on the receptor route.** None of Tm5Y / TmY21 / TmY13 / LC11 / Y3 / Li19 /
   Tm32 has a profile in any source; the lookup changes no input of the object detectors; and the moving-ball assay
   (`scripts/probe_object_sweep.py`) shows no object signal at LC11 / LC10a in any mode, with the skeptic's positive
   control demonstrating that the loom chain does respond to size (a 43 deg object drives LPLC2 +3.7 mV, LC11 +0.2).
   The object item becomes a medulla -> lobula wiring / dynamics question.
2. **A suite-neutral receptor mode exists:** `sign` with the absolute-level rule (`abs`, 0.17 % of synapses changed)
   is 27/0/2 in four suite runs (off 26/1/2, 24/3/2), raises the loom GF and escapes, costs 10 Hz on the legacy loom,
   and touches nothing on the object, optomotor or compass populations. Held back until a contested-flip rule removes
   the 26 +1 rows another source contradicts (Tm9 above all) and a Tm9-held-at-minus-one control is run.
3. **Default model changed, data-driven:** `TYPE_NT_OVERRIDE` labels TmY14 glutamate, Mi19 serotonin and aMe8
   acetylcholine for their previously sign-0 cells (107 cells, 0.027 % of synapses; three-seed suite: no status change).
   The cache is rebuilt locally and on the cluster.
4. **Compass:** the receptor model is inert on the ring; the GLNO <-> PEN loop (19 % of PEN's input, sign 0 today)
   decides the persistence window -- glutamatergic GLNO kills the bump at gE 1.75 in 6/6 seeds, cholinergic holds it --
   and both EM predictions favour inhibitory, so the session-9 compass gains were tuned against a silent loop.
5. **Slow term:** per-class split removes the runaway; four settings pass the runaway checks but only on class weights;
   the 'gain' mode acts on the net input; remains an experiment flag.

Code fixes from the verification: a model option that forces the Torch path now downgrades `cuda_sparse=warp` to
cuSPARSE with a warning instead of aborting the room sections silently; `cx_wedge.py --nt-override` can no longer
overwrite the audit's structural files; `cluster_run.py --fetch out/` copies the directory's contents instead of
nesting; the two remaining scratchpad generators are in `scripts/`.

## Session 10: receptor-expression integration, round 3

Round 3 (13 agents: rule, adoption, four experiments, five Opus skeptics, one critic) closed the receptor plan's
open list; `docs/NT_INTEGRATION.md` section 7 has the outcome, `docs/audits/receptor_verification.md` the verdicts.

**The default model now uses receptor-derived signs.** `LIFParams.receptor_model = 'sign'` with the absolute-level
rule (`abs`) on the contested-flip table: a glutamatergic synapse is excitatory where the target's absolute iGluR
expression exceeds its GluCl expression and no other profiled source says otherwise (26 rows removed by that rule:
Tm9, L1, 24 ring-neuron rows); histamine is silenced where two or more sources agree the target has no ort / HisCl.
It changes 0.148 % of the synaptic weight (30,916 glutamate flips, 17,379 histamine silencings), 95 % of it on
optic rate units (T1, Dm9, Mi4, Mi1); `receptor_model=None` reproduces the previous weights byte for byte (pinned
test). Under the project's rule this is the right kind of change: a sign taken from expression data, no fitted
parameter, no gain, reversible by one flag. (The round-3 '27/0/2 in 10 of 10 runs' was measured with the benchmark's
walk / motion sections half applied; round 4 below has the corrected 26/1/2, equal to off's best, with no check worse in
status; the demo loom escapes 12/12 seeds; costs on record are the legacy loom GF (27-32 vs 37-44
Hz, still PASS), KC_active 816 vs 1426, and more spontaneous take-offs in the room (24 vs 3 in 16 flies x 5 min, one
batch). One caveat qualifies the headline: the critic found that benchmark.py's two legacy sections (walk, motion)
built their optic lobe without the receptor lookup, so the check that flips to PASS (walk.power_max) and the one
quantified cost were measured with the model half applied. That is fixed (and `--receptor-model off`, which had
become a silent no-op, again means the presynaptic rule); the re-score is the first item of round 4, and until it
runs those two numbers are not quoted as properties of the shipped model.

**Step 8 (stop-gap retirement) under the new default:** LPi x4, GF x0.3 damping and the AL LN GABA override each
break walk.power_max alone in 3/3 replicates, so none is retired; the session-9 "GF damping can go" is retracted
(its ablation now raises the walking GF instead of collapsing it); weakening LPi x4 -> x1 undoes the loom cost. To be
re-derived after the benchmark fix, since the deciding check lives in the affected section.

**Compass with the GLNO loop closed inhibitory** (both EM predictions favour it): the bump persists at gE 2 / gD 15
at 180-184 Hz (-10 %); over a seed-matched grid the silent and inhibitory rings persist in equal numbers of runs
(21/36 each) with the window shifted one gD step, not narrower; the session-9 window was a seed-0 statement and
fails at three seeds at its corners. Nothing in the session-9 compass conclusions moves; gE 2 / gD 15 lies in both
windows; the 200 Hz bump rate is untouched.

**Slow term on abs weights:** the hunger-assay precondition is not met (walk.power_max fails in 11/12 runs); the
Torch-path walk / motion sections are not bit-reproducible; the dop1r1 confound is unseparated. Stays a flag.

**Object sweep with a proper null:** the (ball - none) statistic's no-stimulus value is +0.07-0.19 mV (a max over
cells); only LPLC2 rises above it (+0.1-0.2 mV; z +5.4), an invisible-ball control confirms the null; the
medulla carries the ball (Mi4 z +22-29) and the lobula small-field stage does not. Hypothesis (a) is closed with an
error bar; the object item continues as a medulla -> lobula wiring / dynamics question.

Verification cost this round: attribution and wording errors only (a per-seed loom determinism that is run-to-run;
"narrower" compass window; "discrete" walk.power_max values; p 0.0003 -> 7.7e-5) -- none reversed a decision, and the
skeptics' added controls supported every direction. 64 tests pass; round-3 batches: six task + five skeptic, 0
failed jobs.

## Session 10: receptor-expression integration, round 4 (closing the plan)

Round 4 (13 agents; Opus on the run-and-tabulate stages and the skeptics) re-scored the default with the fixed
benchmark and finished the plan's open items; `docs/NT_INTEGRATION.md` section 7 has the outcome.

**The round-3 caveat resolved against the round-3 headline.** With the optic lobe under the receptor signs in every
section, the default scores 26/1/2 in 11 of 11 suite runs -- the FAIL is walk.power_max (79.5 Hz vs a hand-set 50 Hz
bound; off fails it too at 73.2) -- equal to off's best tally, not above it; the "-10 Hz legacy loom cost" was a
+10-14 Hz gain measured backwards, and direction selectivity rises 0.17 -> 0.23. The default stays, on the rule's
letter ("not worse in status than off", expression-derived, reversible), and the record now says so instead of
"27/0/2". Off is again producible from the scripts and reproduces the round-2 off values bit for bit.

**The one open threat to the default is behavioural, not a suite check.** Its take-off excess in the room replicates
across four brain RNGs (95 vs 16 hops over 64 flies x 5 min), and the skeptic's route split shows ~40 % are GF escape
jumps on room optic flow from a higher walking-GF tail (median 33 vs 28.5 Hz, 8/16 vs 2/16 flies at the 33 Hz
threshold); the rest are voluntary (10 vs 0 with the escape route disabled). Nothing in the suite scores it; a
`hops` section with the escape / voluntary split and a voluntary-only reference is owed (round 5).

**Step 8 reversed by the corrected benchmark:** the GF x0.3 input damping can be retired (its ablation is 27/0/2 in
10/10 draws and turns the default's one FAIL into a PASS at 48.5 Hz; damping only the cholinergic DNp70 is inert, so
the four inhibitory inputs are the whole effect) -- the session-9 recommendation reinstated on better evidence, to be
adopted alone with its own suite and hop batches, because restoring 2,899 |W| of inhibition onto DNp01 is exactly the
change that could move the take-off cost. LPi x4 and the AL LN override stay.

**Closed items:** compass operating points robust to the GLNO sign (gE 2 / gD 15, 2.25 / 25, 2.5 / 25; the receptor
default is byte-identical to off on the ring); the type-majority NT rule reaches 36 silent cells and would contradict
the AL LN regex where it overlaps -- the regex has no data replacement; the slow term's non-determinism is the sparse
product itself (a per-op probe: 20/20 distinct under the deterministic flag) and its cost is the term, not the
dopamine signs -- parked; neither KC nor DN1 flips carry the taste rise (fan-in normalisation suspected).

**Verification this round** refuted one premise ("spontaneous" hops), one reference scale, one determinism claim and
one causal reading; no decision reversed. The receptor data are exhausted -- six sources, 25.7 % of the weight
decided, the object-pathway types unprofiled everywhere, the ring inert -- so after the closing round the thread
hands over to the dynamics questions.

## Session 10: receptor-expression integration, round 5 (closed)

The closing round (7 agents; Opus skeptics) built the take-off instrument, adopted the GF-damping retirement and
settled the attribution. The plan is closed; `docs/NT_INTEGRATION.md` section 7 carries the outcome and the handover.

**The default model now.** Synapse signs from the presynaptic transmitter, corrected per postsynaptic type by the
receptor-expression table where six transcriptomic sources decide it (`receptor_model 'sign'` / `abs`, 48,295 entries =
0.15 % of the weight; `None` restores the presynaptic rule byte for byte); the GF x0.3 input damping retired
(`DEFAULT_TYPE_PATH_GAIN` keeps LC4/LPLC2 -> DNp01 x3; `GF_DAMPED_TYPE_PATH_GAIN` restores it) [2026-09-13: still
exactly this -- `brain.py:221` -- and the cluster runs' own provenance shows only the x3 pair, but
`docs/audits/interp_atlas.md` 6.1 and 8 quote the retired five-type x0.3 damp (SAD073 / GNG300 / DNp70 / CL367 /
PVLP010 -> DNp01) as if it were in force; it is not, and any reading resting on it (e.g. "DNp70 is damped") is
void. "Session 10, interpretability toolkit"]; `TYPE_NT_OVERRIDE`
for TmY14 / Mi19 / aMe8. Four weight md5s pinned in the tests. Suite 27/0/2 in 4 of 4 draws (walk.power_max 48.5 Hz
against the hand-set 50 Hz bound, in 14/14 draws) -- the tally gain is the retirement's, and the two changes
interact: off with the damping retired fails walk.power_max at 95-97 Hz, so under the shipped gains the receptor
model is ~48 Hz better than off on that check where under the damped gains it was 6 Hz worse.

**The take-off cost is now scored and split.** Every hop in the batched room records its route; `benchmark.py
--sections hops` (opt-in) scores voluntary and escape rates and the walking-GF median. Shipped default 56 = 24
escape + 32 voluntary vs pre-retirement 75 = 31 + 44 vs presynaptic model 9 = 9 + 0 over 14,400 fly-s each; off
never takes off voluntarily (0 in 28,800 fly-s). The retirement is not worse in either route and trims the excess
by a quarter to a half (inside rerun scatter: the same seed gives 19 hops in one run and 11 in another), but the
excess over off survives at 6x with an unmoved walking-GF tail (median 31.9 vs 27.2 Hz; 19/48 vs 8/48 flies at the
33 Hz escape threshold). The shipped default's voluntary entry is a KNOWN GAP in substance. Neither the KC / DN1
holds (round 4) nor the DNp01 inhibition (round 5) accounts for it, and walk.GF_max shows the two halves of the
receptor signs cancel (either half alone raises the walking GF 2.5-2.9x) [2026-09-13: **retracted as a
cancellation** -- that was a seed-0 reading. The seed-0 digits (off 4.964, default 4.629, holdBrain 12.517,
holdOptic 13.311 Hz) reproduce exactly, but seeds 0-2 scatter as widely as the effect (off 4.96 / 9.38 / 13.41,
default 4.63 / 4.96 / 0.00), every arm vs off is `null`, at n = 4 holdBrain minus off is +0.006 Hz, and DNp01's
1,455 input entries are *identical* in all four arms: the difference is in presynaptic rates, not in DNp01's
synapses, and `walk.GF_max` is not a quantity a cancellation can be read from at three runs.
`docs/audits/interp_decompose.md` 2; "Session 10, interpretability toolkit"] -- a dynamics question the sustain / RL
work inherits, with the room hold pair as the first experiment.

**Attribution closed.** Taste, smell and the sugar checks are Brain-side (bit-exact in three run dirs), direction
selectivity optic, the legacy loom mostly optic, walk.power_max both sides non-additively; fan-in normalisation is
refuted (the dissociation survives with the normalisation off); the CPU double dissociation names 123 histamine
silencings (282 synapses onto OA-AL2i3 / TmY14 / DNge14x) as what taste depends on and the 3,709 KC + DN1 glutamate
flips as what smell depends on -- dependence, not magnitude.

**Verification:** instrument mostly sound (its PENDING sections were filled from the landed batch; two of its three
reference values re-derived; a gap-style entry that could never fail made a plain check), adoption sound (a "gap
closed" reading was one low draw), attribution mostly sound (a PASS reported as FAIL; float32 counts; the decisive
alpha = 0 test was the skeptic's). No decision reversed. The session-9 "GF damping goes" is now adopted, on evidence
that survived a retraction and a reinstatement.

**Handover.** The receptor data are exhausted (six sources, 26 % of the weight decided; PEN, GLNO, PFL3, hDelta, the
object-pathway types, DNp04 / LPT unprofiled everywhere; the ring inert; 95 of 99 unknown-NT types unreachable). The
threads that continue are dynamics: compass with senses / PEN L-R / PFL3 readout from the GLNO-sign-robust operating
points; medulla -> lobula small-field wiring; a feeding-capable metabolism; the walk.power_max bound (re-derive its
referent under the shipped gains before tuning anything to sit under it); the take-off cost. First commands for each
are in the round-5 verification record. No round 6 of receptor work.

## Session 10, dynamics: why the plain fly walks straight

Observed in the room demo: the plain fly (program `none`) walks nearly straight, past fruit and eventually off the
table, even when starving, where an earlier model wandered. Bisected with `scripts/probe_walk_straightness.py`
(BatchSim, 16 plain flies, full table, no fence, 60 s; cluster batch walk-straight-fb28c0, 10 jobs; out/ws_*.json):
median yaw-rate SD over flies is 2.5-2.6 deg/s under the shipped default, 1.6-1.8 with the receptor model off, 1.7-1.9
with receptor off + the GF damping restored (the pre-session-10 weights), 2.6-2.8 for the round-3/4 default, and
1.6-1.7 for the true session-9 weights (the pre-TYPE_NT_OVERRIDE cache); straightness 0.99-1.00 in every arm; no arm
reaches a fruit (closest approach 2.5-9 cm; 0-1 of 16 leave the table within 60 s). **Every arm walks straight, so none
of the session-10 weight changes caused it** (the receptor default is if anything slightly turnier). The history
shows what did: `Locomotion.k_opto` was deg2rad(150)/15 rad/s per Hz of DNp04 / LPT asymmetry until session 9 zeroed
it (commit c4ec33d: the optomotor readout never flipped under imposed rotation and HSN / DNp20 / HSE respond to the
fly's own walking). The old fly's turns were that optomotor term reacting to self-motion -- an artefact -- and with
it gone the only turning signals left in the plain body, DNa02 R-L and leg-MN L-R asymmetry, are symmetric with no
directed input. [2026-09-13: localized, and it is not only an absence of input at DNa02 -- DNa02 is held **below
threshold** in the room by sign-correct tonic inhibition (-326 / -392 mV/s = -1.6 / -2.0 mV of steady conductance
against a 7 mV threshold gap) while its three lateralised excitatory classes are silent at source (PFL3 and
AOTU015 / AOTU001 at 0.000 Hz, most of LLPC1 never firing and LPT22 cancelling what survives at corr -0.86) and the
one live lateralised route, the wind, is ~90x under dose; and the **VNC runs open-loop** -- every `vnc_sensory` cell
is at 0 Hz in the room, so the walking body reports nothing back. `docs/audits/deficit_turning.md`; "Session 10,
interpretability toolkit".] [2026-09-13, dynamics round 2: the open-loop VNC was **closed and the fly still walks
straight**. `senses.Proprioception` (opt-in, default OFF) drives all 941 afferents from the body state, every silent
ascending stage fires, and DNa02's operating point moves only -1.52 -> -1.17 mV against the 7.0 mV gap while its L-R
stays a fixed offset; the yaw-rate SD that does rise (2.848 -> 3.317 deg/s, straightness 0.9956 -> 0.9818) is entirely
the leg-MN branch of `body.Locomotion`'s hand-written yaw law and never changes sign. `docs/audits/vnc_drive.md`;
"Session 10, dynamics round 2".] Starvation does nothing in the plain model by design: hunger-driven search exists only as programs
(`--program anemotaxis` / `cx`).

So this is a real, previously masked deficit of the full model rather than a regression: the connectome model has no
source of spontaneous turning. In the animal that comes from central-complex / LAL dynamics (the compass bump's
projection through PFL3 to DNa02, plus exploratory state), which is exactly what the session-10 dynamics round is
measuring (does a bump produce a PFL3 / DNa02 asymmetry in the room). Under the project rule the options are: leave
the plain fly straight and state it (the honest default); a compass module with documented gains as a swappable
stop-gap if the dynamics round shows the bump steers [2026-09-12: **closed negatively** in dynamics round 1 -- the
bump does not steer (yaw first harmonic 0.3-1.2 deg/s against a 20-31 deg heading sd), does not track heading
(circ corr -0.11 to +0.11) and is pinned to 5-7 attractor sites, so the compass gains stay experiment overrides
and the plain fly stays straight by default]; or a noise-driven turning term, which would be hand-crafting
and is not proposed. A benchmark assay for spontaneous turning (yaw-rate SD / straightness of the plain fly in the
room, against free-walking Drosophila turn statistics) is added to the battery as a known gap so this cannot slip
through the suite again -- none of the 29 existing checks scores it.

## Session 10, dynamics round 1 (2026-09-12)

Five threads (compass-room, compass-shift, optic-audit, takeoff-hold, feeding-horizon), five Opus skeptics,
~60 cluster jobs, 0 failed. **No model default changed**: `git diff -- flyverse/` is empty at 305f507, and the
compass gains (gE / gD) and the drain scales were experiment overrides, stated as such. Audits:
`docs/audits/compass_room.md` and `cx_shift.md` (both still stubs -- see the process lessons; R2-0 fills them),
`optic_measures.md`, `feeding_horizon.md`, `receptor_integration.md` G.5.

**The compass bump survives the senses and does nothing with them.** At all three GLNO-sign-robust operating
points the ring attractor lives the whole 38 s free window in 192/192 flies over 12 gain runs (bump 219-220 Hz at
gE 2 / gD 15, 231 at 2.25 / 25, 259-261 at 2.5 / 25; the shipped control dies within 0.1 s, PEN 0.0 Hz, 0/80
flies in 5 control runs) -- but it does not track heading (circ corr(centre, heading) -0.11 to +0.11 per run),
does not steer (yaw first harmonic 0.006-0.021 rad/s = 0.3-1.2 deg/s against a heading circular sd of
20-31 deg), and leaves PFN at 0.59-0.78 Hz and hDelta at 1.47-1.86 Hz under a 220-260 Hz bump, so **PFN ->
hDelta -> PFL3 is answered NO**. It is also pinned: the realised centre snaps to 5-7 attractor sites out of 16
wedges ({1.5, 3.9, 8.4, 10.6, 13.2} at gE 2/15, identical to 0.1 wedge across seeds and programs) and then
freezes (centre circular sd 0.02-0.09 wedges over 38 s, 0 jumps). [2026-09-13: `health` gives the pinning its
membrane signature -- the **undriven EPG sit ~250 mV below threshold** under gD 15 (there is no lower clamp on `v`
in the LIF), so a cell that far down cannot be recruited by a few Hz of ring input, and the bump's own cells are
refractory-limited at a load of 0.440 +- 0.004. `docs/audits/interp_health.md` 3; "Session 10, interpretability
toolkit".] PFL3's L-R offset is a fixed anatomical
gradient that the gains scale rather than create (amplitude 2.09-3.31 Hz, phase 12.3-12.9 wedges, r2 0.72-0.80,
p 0.018-0.042 with the bump position as the unit and duplicate tile pairs collapsed, n = 8; the shipped control
reproduces the same phase at 0.19 Hz), and DNa02 follows it at 0.08-0.24 Hz against a within-fly temporal sd of
1.0-1.3 Hz. `out/cxroom_orig/`, `out/cxvfy/verify_table.md`.

**PEN has no signed rotation input in this connectome, and the one silent link is an elastic lever.** GLNO
(4 cells, sign 0) is 19.4 % of PEN's raw input, fully contralateral, and is itself fed by PEN 37 % / PS196_b
19 % / EPG 8 % with 8 optic synapses -- efference-copy territory; LNO1 / LNO2 / LNOa / SpsP make 0-13 synapses
onto PEN. Relabelling GLNO GABA (both EM predictions are inhibitory and disagree, Glu 0.505 vs gaba at conf
0.30-0.33, and `NT_SIGN` maps both to -1, so the arm is a what-if that a glutamate arm would reproduce) opens a
one-sided lever: a 40 Hz 1 s unilateral PEN drive shifts the bump +0.301 +- 0.036 wedges against
+0.038 +- 0.018 with GLNO silent, 6/6 seeds, sign-flip p 0.031, R-minus-none null in both (+0.015 / -0.009).
The deflection is **elastic**, relaxing to or past its start once the drive stops (onset->end -0.07 to -0.14
wedges at seeds 0-2). Imposed visual rotation at 90 deg/s moves the bump 0.00 +- 0.01 wedges/s against a
4.0 w/s ideal while HSN / HSE / Nod1 / DNp20 flip at d' -2.5 to -4.5 in the same runs: **no visual route to
PEN**. The efferent route the structure points at is untested -- the rig teleports the fly (R2-1).
`out/verify_cx_shift_shift.md`, `out/vcx_*_s345.json`. [2026-09-13: the efferent route was run (the fly turning
itself at 86-113 deg/s) and the bump does not follow that either -- 0.00 +- 0.01 wedges/s. The block is two layers
deep: **PS196_b, which would bring the turn into GLNO, never fires** (0.03-0.16 Hz, null in 24/24 traces), **DNa02
makes 0 synapses onto the PS196_b / LAL / GLNO chain** (so the efferent arm tests visual reafference plus a VNC
loop, and the sided report the VNC does send up reaches nothing in the PEN chain), and what the eye delivers arrives
at GLNO / ER1_a **direction-blind** and at 0.1 % of GLNO's drive -- so signing GLNO would carry nothing (the gaba
arms: -3,752 / -3,797 vs -3,749 mV/s onto PEN_a). `docs/audits/deficit_rotation.md`; "Session 10, interpretability
toolkit".] [2026-09-13, dynamics round 2: **PS196_b now fires** -- 0.193/0.519 -> 1.437/1.505 Hz with the
proprioceptive / haltere transducer on, and GLNO with it (0.034/0.038 -> 0.664/0.657 Hz, the first time PEN's nodulus
input carries anything in the plain fly) -- **and the bump still does not move** (-0.0047 to +0.0036 wedges/s against
4.0 ideal). The route is **unsigned by construction**: the only afferent class two steps from PS196_b is the haltere
SApp and `MotorRates.haltere` is one bilateral number, so under the labelled Coriolis stop-gap, with PS196_b driven
13x harder (11.05/7.52 Hz), its L-R moves the *same* way in both turn directions (+3.52 ccw, +3.95 cw). Signing GLNO
stays moot. `docs/audits/vnc_drive.md` 6; "Session 10, dynamics round 2".]

**Where the small object is lost, named at last.** The static-apple figure is carried by the lamina (L1 z
+18.4, L2 +12.0), the medulla (Mi1 -8.8, Tm3 -6.8, Tm20 +5.3, Tm4 +4.8, Tm1 +4.5) and T4c/d / LPLC2, and it is
gone at the inputs of LC10 / LC11 (Tm5Y +0.2, TmY21 -0.6, TmY13 +2.2, T2 +2.7, T3 -1.0; none carries in 3/3
seeds) by sign-correct **ON/OFF cancellation through excitatory convergence** (T3 sums Mi1 22.5 % + Tm3 8.2 %,
lowered by the dark object, against Tm1 16.0 % + Tm4 6.7 %, raised: linear estimate -0.0004 against its own
-0.0011; T2 the same with a residual +2.7; Tm5Y is dilution instead -- one carrying input, Tm20 at 14 %), then
by **l1 pooling at the LC cells** (LC11 +0.046 mV and LC10a +0.080 against LPLC2's +0.54 and a 7 mV criterion,
in the deterministic lobe). No hand-crafted optic measure sits on those edges and none of the 16 ablations
makes them carry. `gain_fb = 0` is the deterministic null (every none-vs-none statistic falls to 1e-9-3e-7 mV,
five to seven orders under the shipped 0.06 mV null), and in that lobe the small-field types do pass a
single-cell signal of the medulla's size (best-cell |dev| diff T2 0.066, Tm5Y 0.072, TmY21 0.062, TmY5a 0.084
against Mi4 0.047, Tm3 0.104) -- a signal, not a retinotopic population figure and not a drive. The levers left
are **physiology parameters** (the receptor table's slow classes: GABA-B mid on T2 / high on T3, mGluR on T2;
a per-type output normalisation at LC10 / LC11), not data. Two bounds the skeptic added: the stage pipeline is
**not reproducible at fixed seed** (baseline seed 0 rerun: median |dz| 0.36-0.49, max 5.2-7.6, 15 of 234 apple
carry-threshold flips), so every per-type z is +-1.5 and the replicate unit is runs, not seeds; and LPLC2's
object null under the shipped gains is z +1.2 (rank-sum p 0.07, n 3 vs 5) where round 3 had +5.4, because the
null rose 0.139 -> 0.238 mV while the ball arm did not move. `out/optic_audit/`, `out/optic_verify/`,
`docs/audits/optic_measures.md`.

**The room take-off cost is optic-side.** Round 5's hold pair in the room (3 brain seeds x 16 flies x 300 s per
arm, 9 jobs, 0 failed) plus the skeptic's 4th matched batch at a fresh seed / env block gives, over 4 batches
of 19,200 fly-s per arm: shipped 75 hops = 27 escape + 48 voluntary, optic side (`holdBrain`, 44,463 medulla
entries) 59 = 19 + 40, Brain side (`holdOptic`, 3,832 entries) 8 = 8 + 0, off 11 = 11 + 0. The **Brain side
carries none of it** (vs off: hops U 1980 p 0.59, 0 voluntary) and the optic side **most of it** -- 75 % of the
hop excess, 83 % voluntary, 95 % of the GF-median shift, having read 92 / 106 / 103 % on three batches, so
these shares are not measured to better than a factor ~1.5. The escape route alone cannot be attributed at this
exposure (holdBrain vs off escape U 2300, p 0.095) and dose is uncontrolled (the optic half is 95 % of the
changed |W|). The halves do **not** cancel in the room as they do in `walk.GF_max` (4.6 / 12.5 / 13.3 / 5.0 Hz
pinned against a monotone room ordering off ~ Brain side << optic side ~ default), so the pinned walk section
is not a proxy for the room's escape route. Off with the GF damping retired is the same denominator as off with
it (0.486 vs 0.625 per 1,000 fly-s, p 0.58; off now has 0 voluntary take-offs in 52,800 fly-s over 11
batches), so the record's excess statements survive on the correct denominator. And 0 of the 48,295 changed
entries land on DNp01, MN9, LC4, LPLC2 or any motor superclass: the cost is an upstream medulla state change
reaching an untouched loom / wing-power pathway. `out/d1_*.json`, `out/sk_d1_*_4.json`,
`docs/audits/receptor_integration.md` G.5.

**Feeding cannot rank models, at any horizon or drain tested.** `meals` is identically equal to `contacts`
fly-for-fly in 12/12 runs (the metabolic re-feed channel contributes exactly zero; `meals` is an arrival
counter thresholded at 1.5 cm on the ground), 0.13-0.50 meals per fly per arm (0.06-0.50 per run), default vs
off p 0.40-0.78 everywhere, and the cumulative arrival curve saturates (0.03 added in the second five minutes).
Starvation is not the limit: at drain-scale 0.5 / 0.25 no fly reaches energy 0 and arrivals are unchanged, an
empty tank *raises* walking speed 12-15 % (hunger is the model's only motivational gain and it saturates), and
`batch_body._metabolism` has no death, immobilisation or episode end. The audit's one positive -- the default
nearer the fruit on closest approach in 3 of 3 cells, stratified p 0.016 -- **failed replication**: two new
cells, one reversed (U 145, p 0.53) and one null, 5 of 6 runs, sign test p 0.22. Its negative tail is entirely
airborne (a walking fly sits at z = 0.75, so the surface distance cannot go below 0) and it is confounded with
hops (Spearman rho -0.25 to -0.75 within arm). The drain's direction on food-finding is unresolved (3 of 4
seed-matched drain 1.0 -> 0.5 transitions raise meals). The mechanism is the **cx program's terminal
approach**: the plume is Gaussian downwind only (`air.py:104`), upwind of the source only the isotropic near
field remains, the rule is "steer upwind" with nothing that uses a concentration change to stop or reverse,
81.2 % of flies end past the apple going upwind (median final x 0.415, 16.5 cm past it), and the
closest-approach median is 2.3-7.2 cm against a 3.8 cm capture radius. `body.py` is untouched and stays so.
`out/feedh_*.json`, `out/skfeed_*.json`, `docs/audits/feeding_horizon.md`.

**Hand-crafted measures.** `walk.power_max`'s 50 Hz bound (`scripts/benchmark.py:85`) is **still owed**
(handover item 4): it fails under 13 of 16 optic ablations (51.5-198.8 Hz), is non-monotone in `gain_out`
(80 / 100 / 120 = 63.8 / 48.5 / 51.7) and **anti-correlates** with the room take-off rate across the four hold
arms (off 95.5-97.1 Hz FAIL is the quietest room arm; the default 48.5 PASS the noisiest), so nothing may be
tuned to it and no round-4/5 verdict resting on it may be quoted until it is re-derived or de-scored; and
`scripts/retire_measures.py:245` references `brain` without importing it, so the script cannot even be
imported. `DEFAULT_PAIR_GAIN[1]` (Tm4 / Tm9 / CT1 / TmY15 -> T5 x5) is **data-contradicted** -- 78,877 of its
101,619 edges are Tm4 / Tm9 -> T5 acetylcholine (Nern 2025 validated, nAChR-alpha5 on T5) against 22,742 GABA
edges -- but it is not retirable (loom 29-36 Hz and T5 DSI 0.15 without it), so it is **re-labelled a T5 drive
gain** and documented as a stop-gap. `OpticParams.drive_clip_mv = 35` is a retirement candidate (11/11 without
it) pending 3 independent draws. `DEFAULT_PAIR_GAIN[4]` (`.* -> LC4|LPLC2 x1`) is a literal no-op and is
retirable now, with no behaviour change.

**Process lessons.** Two threads reported finished batches as "still running" because `cluster_run.py --fetch
out/` failed on a 152 MB scratch `cache_*` directory inside the run's `out/` -- the jobs had completed, every
number was in the threads' own shipped console logs, and both audit documents shipped as stubs. Fetch a **named
subdirectory** (`--fetch out/<name>/`) or a file glob, and read the console log's `<n> job(s), 0 failed` line
before writing a status section. Run JSON headers must carry the resolved `LIFParams`, `type_path_gain` and the
**realised** device (`options.device` records the request and is `None`), so that an arm identifies itself
without run-dir md5s and prose -- md5s are not portable anyway (CRLF here vs LF on the cluster). And ship every
generator: the feeding thread's stratified / Fisher statistic, the LC10 / LC11 pooling arithmetic and three of
the take-off thread's logs have no script behind them.

**Round 2, in order, each one submission.** R2-0 (CPU, first): fetch the `cx_shift` originals into their
reserved paths and fill `cx_shift.md` sections 2-4 and `compass_room.md` sections 4-6 from data that already
landed, and fix the `retire_measures.py` import. R2-1 compass: the efferent rotation arm (the fly turning
itself instead of being teleported), GLNO silent vs GABA, with the teleport arm as the visual control, plus a
CPU test of whether the 5-7 pinned sites are the wedges of maximal recurrent weight. R2-2 object: make the
stage map an instrument first (three same-seed draws, `gain_fb 0` as the null), then the two data-backed arms
-- slow GABA-B on T2 / T3, and a per-type output normalisation at LC10 / LC11. R2-3 take-off: split the optic
side by transmitter (`holdOpticHis` vs `holdOpticGlu`) with a random 3,832-entry dose control, all five arms in
one submission at matched env seeds. R2-4 `walk.power_max`: three draws each of baseline / no_dn_vnc_gain /
no_path_gain / no_drive_clip under the shipped gains, then de-score or re-derive the referent (the drive-clip
retirement rides along). R2-5 feeding: re-instrument rather than replicate -- a ground-only closest approach
and `first_contact_s`, `--fruit all` (2.28 ideal encounters per fly per 5 min), the `--drain-scale 0` hunger
clamp as the ranking assay, and `cx+klinotaxis` as the terminal-approach test. R2-6 reporting: the quantifier
corrections into `receptor_integration.md` G.5, `optic_measures.md` and `feeding_horizon.md`, and the five
skeptic verdicts verbatim into `receptor_verification.md`.

The interpretability-toolkit workflow (`flyverse/interp/`) was launched immediately after this round.

## Session 10, interpretability toolkit (2026-09-12/13)

Eight tools, one skeptic pass each, three applications. **No model default changed**: `brain.py` / `optic.py` /
`body.py` and the weights are untouched, and the only code edited anywhere was the toolkit's own
(`flyverse/interp/trace.py`'s raw-count fix; a documented `--by-side` alias in `scripts/interp_atlas.py`).
Shipped: `flyverse/interp/` (**decompose** -- what drives a cell
set per frame by presynaptic type / transmitter / receptor tier; **trace** -- where along the depth a stimulus is
lost; **paths** -- effective k-step signed gains and which links are silent; **lesion** -- check x lesion delta
matrices and the double dissociations; **atlas** -- what every motor readout does when population X is stimulated;
**health** -- per-type operating point of a rollout; **ledger** -- measured responses against curated expectations;
**export** -- the Neurome serializer), one CLI per tool (`scripts/interp_*.py`), **65 CPU tests**
(`tests/test_interp.py`), a validation record per tool (`docs/audits/interp_<tool>.md`), three applications
(`docs/audits/deficit_{turning,object,rotation}.md`), the procedure (`docs/INTERP.md` 10) and ten open contract
defects in the shared files (`docs/INTERP.md` 11: `common.raw_counts`, a floor on the null SD, `MIN_REPLICATES`, the
`--null` flag, `VALIDATION['health']` 3,407 -> 3,312, the submodule/function namespace clash). It is **one system**
at the data layer -- one selection grammar, one weight accessor (`effective_weights` = `Brain._W_cpu` bit-for-bit,
md5 `ed1df661...` in every JSON), one Result schema (`Result.check()` empty on every shipped JSON of all eight
tools), one provenance block with the realised device and the cache fingerprint `ef23cc27...`, one comparison
primitive -- and **seven scripts at the statistics layer**: five private `raw_counts` fixes, four null-CLI
conventions, three answers to the deterministic null, and `result` / `reproduced` meaning something different per
tool. That seam is exactly the layer that decides whether a number is a finding, and it is what `INTERP.md` 11 fixes.

**Every tool reproduced a hand-made localization; two did not reproduce as written, and both are findings rather
than tool failures.** `paths` 63/63 checks, deterministic and independently re-derived by brute force (GLNO -> PEN
84 entries / 16,371 syn / 19.4 % / sign 0; the two-step ExR6 / ER4m / ExR4 ranking; only prose errors: one omitted
runner-up and "203 LAL types" for 204). The taste carrier bit-identically (123 entries / 282 synapses histamine
0 -> -1 onto OA-AL2i3 62 / TmY14 25 / DNge138 5; MN9 0 entries). The hold tables 8/8 entry counts and md5s and the
CPU double dissociation 32/32 per-seed rows in three independent runs. The compass bump seed-matched to the digit
(EPG_in 199.8 +- 1.6 Hz, refractory load 0.440 +- 0.004 inside the 0.40-0.57 band) and the NT-audit sign-0 shares to
the synapse (3,312 / 2,683 / 2,701,289 of 124,161,873). The LH odour gate to 0.00-0.17 Hz over 11,147 rows. The wind
/ PFL3 / DNa02 atlas arms at the criterion on fresh seeds (DNp18 +50.98 / +50.59 vs +45.2; PFL3_L at 80 Hz ->
DNa02_R 24.11 / 21.57 vs 22.6, DNa02_L exactly 0.00; DNa02_L -> leg L-R +2.13 / +2.24 / +2.29 vs 2.58). The round-3
object sweep and its retina bit-identically (round-trip max_abs_diff 1.4e-14, 16/16 Neurome edge counts exact). The
suite's own 9 references. **The two that did not:** (1) `walk.GF_max`'s G.4 "cancellation" is a **seed-0 reading** --
the seed-0 digits 4.964 / 4.629 / 12.517 / 13.311 Hz reproduce exactly on a second cluster batch, but seeds 0-2
scatter as widely as the effect (off 4.96 / 9.38 / 13.41, default 4.63 / 4.96 / 0.00; every arm vs off `null`; at
n = 4 holdBrain minus off is +0.006 Hz), and **DNp01's 1,455 input entries are identical in all four arms**, so the
tool localizes the arms to presynaptic rates, not to DNp01's synapses (`docs/audits/interp_decompose.md` 2).
(2) The GPU hold matrix's `walk.GF_max` / `walk.power_max` / `walk.power_sustained` / `loom.GF_peak_hz` rows are
**draw-dependent** -- 56 / 57 / 59 of 60 over three batches, with off `walk.power_max` reading 72.65 in one draw
against the published 95.5-97.1 -- so "59 of 60" is one batch's property and the shipped status is honestly `not
reproduced`; the reproducible core is the 44 bit-identical taste / smell / bitter / rest / escape rows and the
(holdDN1, holdKC) x (Shiu sugar, bitter) dissociation, the only one of 1 / 8 / 17 extracted dissociations that
survives all three batches (`docs/audits/interp_lesion.md`). One qualification to carry: the object-stage `trace`
reproduces **at the verdict level only** (Mi4 z +6.6 / Mi1 +17.2 / Tm3 +10.5 `result`, the six small-field types
`null`, 5 v 5) -- the z magnitudes do not (the reference's Mi4 +22.3 / +28.6 against a 5-draw null SD 3x larger
here), and under the contract's own default statistic (`figure_z`) the same target reads `not reproduced`. Read
every `validation.status = reproduced` with its tool's own definition of the word.

**Turning: DNa02 sits under tonic sign-correct inhibition and its lateralised excitation is silent at source.**
(`docs/audits/deficit_turning.md`; 3 room rollouts x 16 flies x 60 s plus a 963-population atlas x 3 runs; **nothing
changed in `flyverse/`**.) The plain fly walks straight -- yaw-rate SD 2.45-2.57 deg/s, straightness 0.99-1.00 --
and the deficit is **MISSING INPUT on three of four routes plus an OPERATING-POINT fact at DNa02** (dynamics), with
**WIRING** contributing on the fourth. DNa02's room input is net inhibitory, -326 / -392 mV/s per cell = **-1.6 /
-2.0 mV of steady conductance against a 7 mV threshold gap** (wired per volley it is net *excitatory*, +466 / +448
mV), from sign-correct GABA / glutamate cells firing at 3-14 Hz (IN12B014, GNG562, PS059, LT51, IN19A003, LPT22);
reproduced from the raw npz, from scratch through `effective_weights`, and on a fresh seed (-323 / -388). The three
lateralised excitatory classes are silent at source: **PFL3 0.000 Hz** in 24 cells x 48 flies -- though the atlas
shows PFL3 -> DNa02 works when driven (+45 Hz at 150 Hz) and **eight other CX populations move DNa02 by 0.6-1.6 Hz**
(hDeltaI both sides, ExR8 both, hDeltaA_L, vDeltaK_R, PFL2_L, PEN_b_L), so "PFL3 is the only CX gate" overstates;
**AOTU015 / AOTU001 0.000 Hz** (downstream of the object deficit); and **LLPC1**, with 63.5 % of its 104
DNa02-presynaptic cells firing at some point but the median cell at 0 Hz, and LPT22 (GABA) cancelling 85 % of the
surviving motion signal at corr -0.85 to -0.86 (the sd of the sum is 34-36 mV/s against 63-68 for each term:
wiring). The wind route (JO -> PS230 -> DNa02) is live and lateralised (r -0.61 with the wind side across 48 flies)
at ~0.08 mV -- **~90x under the dose needed, not the audit's 25x**. And the VNC runs **open-loop**: every
`vnc_sensory` cell is at 0 Hz in the room (SNpp39 / 45 / 50, LgLG), a missing-input fact of the body model.
[2026-09-13, dynamics round 2: **route (b) was built and run, and the input it supplied is not enough.**
`senses.Proprioception` (opt-in, default OFF) drives the 941 proprioceptors from the realised leg-MN / haltere-MN
rates and ground contact; AN04B003 goes 0.589/0.314 -> 3.713/3.690 Hz and the sided excitatory term the audit asked
for appears (AN04B003/L -> DNa02_L +9.09 -> +57.35 mV/s), but the same transducer's haltere channel cancels it on the
spot through PS059 (-90.8 -> -113.5), DNa02's net input moves only -303 -> -235 / -363 -> -342 mV/s (-1.52 -> -1.17
and -1.81 -> -1.71 mV against the 7.0 mV gap, 2-3 % of the required dose) and **DNa02 still does not fire** (0.02/0.10
Hz, `null`). The leg L-R that does grow (+0.151 -> +0.301 Hz) is a **fixed positive offset**, positive in all 400
fly-runs and still +0.09 to +0.33 Hz at 1.6 rad/s of imposed clockwise turn, so `missing input` is closed on this
route and shown insufficient: what remains is the operating point, the transducer's own cancellation, and a body
model with no leg cycle. `docs/audits/proprioception_transducer.md`, `vnc_drive.md`; "Session 10, dynamics round 2".]
Data-driven route: (a) the JO transducer's rate-vs-wind-speed calibration against measured JO responses (a sensor
calibration, not a gain); (b) proprioceptive / haltere drive of the `vnc_sensory` superclass from the body's
realised state, the cells picked from what MaleCNS says those ascending neurons receive (the same route as the
rotation deficit); (c) `health` on the tonic VNC / GNG / PS inhibitors and `trace` T4a / T5a -> LLPC1 / LPT22 (why
the majority of LLPC1 is silent; whether LPT22 sees the same flow). **Hand-crafting, named and not done:** any gain
on PFL3 / LLPC1 / PS230 -> DNa02, a DNa02 bias / noise / threshold, `k_turn` / `k_leg_turn` changes, reading the leg
DNs (DNge035 +7.1 / -6.4 Hz of leg L-R, 3x DNa02) as a readout redefinition, ablating LPT22 as a default, or the
`cx` / `anemotaxis` programs (already documented stop-gaps).

**Object: the ball dies at T3 by ON/OFF convergence, its residual is destroyed by the stochastic feedback, and what
survives is pooled away at the LCs.** (`docs/audits/deficit_object.md`; `apobj-les-e38d76` 96 jobs and
`apobj-lad-287a43` 40 jobs, 0 failed; **nothing changed in `flyverse/`**.) The moving ball / static apple is carried
by lamina, medulla, T4c/d and LPLC2 and lost at LC11 / LC10a's inputs: classified **WIRING + DYNAMICS** at the
small-field stage, **WIRING x output-normalisation (pooling)** at the LC cells; *not* a rate-model artefact, *not* a
sign fact, and specifically a **small-object** fact. Wiring: T3 sums ON (Mi1 22.5 %, Tm3 8.2 %) and OFF (Tm1 16.0 %,
Tm4 6.7 %) carriers through excitatory exact-tier synapses whose figures for the sweeping ball have opposite sign,
and holding either class at 0 restores T3 (0.062 / 0.072 against the shipped 0.024, z +3.5 / +4.8 at 4 v 4;
reproduced by the skeptic on a fresh batch at 0.063 / 0.074, and a size-matched control hold of non-carrier inputs
does not restore). Dynamics: with `gain_fb = 0` the deterministic lobe passes per-cell figures of the medulla's size
(T3 0.041, T2 0.066, Tm5Y 0.072, TmY21 0.062, four digits across batches) while under the shipped stochastic
feedback those very cells carry **7-26 %** of it and the medulla's keep 100 % -- a residual of two opposing sums is
scrambled, a direct response is not. The skeptic's bounds, which travel with the verdict: "lost at T3 / Tm5Y" is a
**threshold call** (Tm5Y reads z +0.9 / +2.2 / +2.6 / +3.65 over four batches of the same model); the "cancellation
fraction 0.96" is a descriptive ratio that does not predict T3's measured figure in sign or size (linear estimate
+4.2e-6 against the measured -2.25e-4); "LC11 / LC10a not restored in any arm" is an **underpowered non-detection**
(null SD 0.02-0.05 mV), not a measurement; at T3 the split is carriers-of-opposite-figure (Tm2, an OFF type, raises
with Mi1 / Tm3) rather than strictly ON vs OFF; and LPLC2's z at 11.4 deg flips between batches (+1.5 to +2.8 --
`object_sweep.md` 8.7's +5.4 does not reproduce). Data-driven route: receptor profiles for the unprofiled types
(Tm5Y, TmY21, LC11 are fallback tier on 100 %); a per-input-class rectification arm before the sum at T3 (a question
about the unit model that needs an `OpticParams` hook, not an edge lesion); a per-body comparison of the LC pooling
(94 / 32 columns, `out_norm l1`, `gain_out` 100 mV -- the one place no data in the tree decides) against Neurome's
LC11 / LC10a recordings at four sizes. **Hand-crafting, named and not done:** any optic measure on the small-field
edges, a sign flip of Tm1|Tm4 -> T3 (the counterfactual restores T3 at z +5.8, but the data say + at exact tier), or
a gain on LC10 / AOTU015.

**Rotation: the efference-copy chain never fires, the eye's report arrives direction-blind, and the one silent link
would carry nothing.** (`docs/audits/deficit_rotation.md`; two 20-job batches, 40 runs, pooled and per batch;
**nothing changed in `flyverse/`**.) The ring bump does not follow the fly's rotation -- 0.00 +- 0.01 wedges/s at
90 deg/s visual and now also under an **86-113 deg/s self-turn**, against 4.0 ideal, with all 80 shipped-cache
phases inside -0.008..+0.010 and a third independent batch agreeing. Classified **MISSING INPUT (primary) behind a
SIGN-0 LINK (structural)**. PEN's one nodulus input, GLNO -> PEN (84 entries, 16,371 synapses, 19.4 %,
contralateral, sign 0 because GLNO's transmitter is unknown: MaleCNS `unclear` at conf 0.48, T-bars Glu 0.505 / ACh
0.373, FlyWire gaba 0.30-0.33 -- all below the 0.5 the project accepts), is the strongest silent link **at k = 3 for
every yaw source and at k = 2 for `optic_yaw` and efference** (the audit's "k = 2 and 3 for every source" is false
for `jo` and `descending_yaw`), and no yaw carrier -- HS / VS / H2 / Nod / LPT, DNp20 / DNp15, JO, DNa02, PS196_b,
LAL139 / 184, WED040_a -- makes one synapse onto PEN. Dynamically **PS196_b, the population that would bring the
turn into GLNO on the efference-copy route, never fires** (0.03-0.16 Hz, max cell 0.57, null in 24/24 traces,
reproduced on fresh seeds), nor do LAL184 / WED040_a / CB2037 / LPsP or the ascending neurons that feed them
(AN07B037_a, PS239); the visual route delivers a signed flip to PLP078 / WED153 / PS047_b (+0.5-1 Hz) but arrives at
GLNO / ER1_a **direction-blind** (LAL139 -10 -> -20 mV/s equal for ccw and cw; 0.1 % of GLNO's +15,200 mV/s drive,
which is 97 % PEN + EPG -- GLNO is an efference copy of *the bump*, not of the body), so signing GLNO would carry
nothing (the GLNO = gaba arms: GLNO -> PEN_a -3,752 / -3,797 against -3,749 mV/s, z +1.5 / +0.3). Caveats: the
"efferent" arm drives DNa02 and the VNC but **DNa02 makes 0 synapses onto PS196_b / LAL / GLNO**, so it tests visual
reafference plus a VNC loop rather than the efference-copy chain; the VNC *does* send a sided report up (56
`vnc_intrinsic` types and the ascending AN07B035 / AN06A026 flip) that **reaches nothing in the PEN chain**; every
run tests one bump site (wedge ~1.5); and the 0.1 % is specific to the gE 2 / gD 15 operating point. Data-driven
route -- the connectome implies a missing afferent: (a) PS196_b's ascending inputs are named (AN07B037_a/b 419 / 52
syn, PS239 312, AN08B026 -> LAL104 269), (b) a `paths` query of what those ascending neurons receive in the VNC (leg
sensory, haltere, `vnc_intrinsic`), (c) drive those afferents as a sensor from the leg / haltere state the VNC motor
neurons actually produce -- **not** from `body.Locomotion`'s yaw scalar, which would close a loop through a
hand-written module -- then repeat the efferent arm in two batches with `verify-batch`; if PS196_b / LAL / WED fire
and GLNO gains a body-locked sided term, the GLNO sign becomes testable on a signal. [2026-09-13, dynamics round 2:
(a)-(c) were run and the prediction **failed on its second half**. The afferents were driven from the VNC's own motor
readout (never from `body.Locomotion`'s yaw scalar) and **PS196_b fires** -- 0.193/0.519 -> 1.437/1.505 Hz, GLNO
0.034/0.038 -> 0.664/0.657, both `result` at 5 runs/arm -- but the term GLNO gains is **not body-locked and not
sided**: bump drift stays -0.0047 to +0.0036 wedges/s against 4.0 ideal, GLNO's L-R stays +26.93 to +28.07 Hz in
every phase of every arm, and under the labelled Coriolis stop-gap (13x the drive, PS196_b at 11.05/7.52 Hz)
PS196_b's L-R moves the same way in both turn directions (+3.52 ccw, +3.95 cw) -- a **sign** failure, not a dose
failure. `AN07B037_a`, the named route, stays at 0.000-0.004 Hz in the room (a rate threshold, not a wiring one: it
fires at 3.1/2.2 Hz under the 80 Hz stop-gap turn); PS196_b is reached through CB0675 / GNG580 / PS047_b instead. So
the classification becomes **missing input (supplied) + a route that cannot be sided in this body + a sign-0 link
that is moot until it can be**, and the actionable item moves to the body model -- a side-split haltere MN readout in
`motor.py`, a leg cycle in `body.py`. `docs/audits/vnc_drive.md`; "Session 10, dynamics round 2".] **Hand-crafting,
named and not done:** a GLNO `TYPE_NT_OVERRIDE`, a gain on PS196_b / PLP078, the EPG <-> PEN gains as a default, or a module that
writes PFL3 / PEN.

**Facts that supersede earlier readings in this file** (each is cross-referenced at the entry it corrects):

* `walk.GF_max`'s "cancellation" (round 5) was a **seed-0 reading**: off 4.96 / 9.38 / 13.41 Hz over seeds 0-2,
  every arm-vs-off comparison `null`, and DNp01's 1,455 input entries identical across the arms.
* The LIF's **effective refractory period is 2.5 ms, not 2.2**: `t_ref` 2.2 ms holds a cell for 5 steps of
  dt = 0.5 ms and it fires again on the 6th, so the minimum ISI is 3.0 ms = **333 Hz** (`interp_health.md` 4).
* **Undriven EPG sit ~250 mV below threshold** under gD 15 -- there is no lower clamp on `v` in the LIF -- which is
  the membrane signature of the pinned bump: a cell 250 mV down cannot be recruited by a few Hz (`interp_health.md`
  3).
* The atlas' **JO stimulation equals sense-driven wind** through `fb.wind` (`INTERP.md` 9's open question, answered
  yes: +47.5 Hz with the room's meander against +51.3 at a fixed 180 deg and the room's own +45.3), and the wind DNs
  carry a **fixed anatomical L-R offset under symmetric drive** -- head-on, DNp18 +13.45, DNp73 +19.51, WED080
  -17.07 Hz, which is half of DNp73's apparent flip (`interp_atlas.md` 3.1).
* **DNge035 is the strongest lateralised leg driver** in this model: +7.50 / -6.19 Hz of leg L-R, contralateral,
  three times DNa02's +2.24 / -1.45 -- a measurement, not a readout change (`interp_atlas.md` 6.1).
* The retired `GF_DAMPED` list is **not in force**: `brain.py:221` `DEFAULT_TYPE_PATH_GAIN` is LC4|LPLC2 -> DNp01 x3
  alone, and the runs' own provenance shows only that pair; `interp_atlas.md` 6.1 and 8 quote the retired five-type
  x0.3 damp as if it were the default and are wrong.
* The VNC is **open-loop** (every `vnc_sensory` cell at 0 Hz in the room) and **DNa02 makes 0 synapses onto the
  PS196_b / LAL139 / LAL184 / WED040_a / GLNO chain** (its 14,446 outputs go to the VNC; 669 onto 1,846 ascending
  neurons in total): the body's own turn has no route back to the compass. [2026-09-13, dynamics round 2: the first
  half is now a statement about the *default*, not about the model -- with `senses.Proprioception` on (opt-in,
  default OFF) all 941 afferents fire and the ascending chain carries. The second half stands: the report that
  arrives is unsigned, so the body's own turn still has no *signed* route back to the compass.]

**Process rules this round paid for.** Replicates are **jobs, and >= 4 per arm** (5 for small effects), because
`common.compare`'s exact-U floor is p = 0.10 at 3 v 3 (0.029 at 4 v 4, 0.0079 at 5 v 5) -- a 3 v 3 "result" cannot
reach significance whatever the effect. The **GPU rollout is not seed-reproducible**: a verbatim `record --seed 0`
repeat gives leg L-R +0.190 against +0.208 (printed yaw-rate SD 9.42 vs 7.54) and same-seed ring histories differ
between batches (gaba-visual seed 4 rest drift -2.657 vs +0.005), so the replicate unit is the run and a second
batch is a replication, not a check -- the aggregates that carry a verdict move ~0.3 %. **Never let two clients
`--fetch` one directory** (35 of 80 files in `out/rot/` were an interleaving of two batches) and **verify a fetched
batch before analysing it** (`interp_apply_rotation.py verify-batch`). `CUDA_VISIBLE_DEVICES` must be `"-1"` on
Windows: the empty string is ignored and a "CPU smoke" then runs on this desktop's GPU. And the `commit` field is
`unknown` in every cluster JSON (the run directory is not a checkout), so code is identified by the compiled-`W` md5
`ef23cc27bea13be7f6a96f3c04fd3737`, the effective-weight md5 `ed1df661716d240b0f9289607f95320c` and `export`'s
source fingerprint -- md5s of run dirs are not portable.

**Neurome hand-off: the size ladder, a scene baseline with a size-dependent LC drive.** Four sizes at 5 v 5 runs
each with matched none-vs-none nulls and the retina replayed (`docs/audits/deficit_object.md` 5; every export
`verify: problems none`, LC11 143 / LC10a 275 bodies x {`upstream_drive_mV`, `output_Hz`}):
`out/export/objsize-d045-20260913T014556Z-47ed1383/` (4.5 deg),
`objsize-d114-20260913T014606Z-7a4eadbd/` (11.4), `objsize-d200-20260913T014616Z-5ece62d6/` (20),
`objsize-d300-20260913T014626Z-d5048f74/` (30), with the summary at
`export-20260913T014635Z-db22e3ea/` (`size_tuning` 288 rows, `retina_footprint` 4, `runs` 40). At 4.5 / 11.4 / 20
deg the two LC types and their small-field inputs sit at the null on the drive statistic; **only at 30 deg** do LC11
(+4.8), LC10a (+8.6) and the small-field stage (Tm5Y +38.5, plus T2 / T3 / TmY21 / TmY13 / TmY5a) reach `result`,
with the loom chain (LPLC2 / LC16 / LC4) coming in from 20 deg. The drive statistic is the **within-run maximum over
cells of the time-mean object-minus-blank drive difference**, compared with the same statistic in independent
blank/blank runs -- not an absolute voltage. What the ladder supports is a **weak, size-dependent LC drive response
and no detected object effect in the 16 exported population firing-rate comparisons**; LC11's missing small-object
response (preferred vertical extent 8.8 deg, width ~4.4 deg, Keles & Frye 2017) is the biological concern, while
LC10a's own preference is 15-30 deg (Schretter et al. 2024), so its 30-deg response is not a reversal. Size and
retinal position change together here (centre elevation 0.88 -> 13.71 deg), so this is a scene baseline, not yet a
controlled size-tuning assay.

[2026-09-13, Neurome intake: the hand-off's original wording -- "the model's size ordering is the animal's inverted"
and "LC11 / LC10a never spike" -- is **retracted**, and the corrections are accepted
(`D:\Projects\neurome\docs\flyverse-size-tuning-reply.md`, `D:\Projects\neurome\reports\flyverse-size-tuning-intake.md`;
`docs/NEUROME_INTERFACE.md` 3b). (1) **LC10a's own target is 15-30 deg** (Schretter 2024, Fig 3a), so a 30-deg LC10a
drive response is not inverted tuning; only LC11 is a small-object type (8.8 deg vertical extent, ~4.4 deg width,
Keles & Frye 2017 Fig 3D/E, calcium). (2) **Firing is not literally zero**: 6/143 LC11 and 19/275 LC10a bodies have
nonzero mean firing at 4.5 deg (LC11 `24647` 0.1833 Hz object vs 0.1167 Hz paired blank at 11.4 deg; LC10a `69463`
0.8333 vs 0.7500 Hz at 4.5 deg). Comparing a time-mean drive difference with the 7 mV threshold gap cannot establish
that no spikes occurred. (3) **Elevation confound**: 0.88 / 4.34 / 8.66 / 13.71 deg with columns dimmed > 50 % at
0.14 / 2.26 / 7.64 / 19.73 per frame; the retina table is a geometry **replay** at a pinned pose (not in-loop
capture) and the **matched blank radiance was omitted**. (4) **Field defect**: `control_ids` names the independent
blank/blank runs while `control_value` comes from arm b of the stimulus recording -- the next export splits
`paired_control_ids` from `null_reference_ids`, adds a blank-radiance table and a `retina.mode` field
(`docs/audits/interp_export.md`). (5) **Statistics**: Holm across the eight LC drive comparisons gives p 0.0635 --
exploratory; the exact U meets ties in 25/288 rows; `diff_signed_best_cell` and `diff_abs_best_cell_mean` stay
distinct (different 20-deg verdicts). (6) Literature inputs now in the ledger: Keles 2020 (LC11 Rdl + nAChR
alpha1/alpha6/alpha7; Rdl disruption -40 % small-dark-object response without releasing bars/gratings; T2/T3 ON+OFF;
T3 -> LC11 functionally excitatory) and Tanaka & Clark 2020 (pooling of fast-adapting size-tuned inputs -- a
competing hypothesis). No graph correction follows. **Agreed next round, "object round 2"**: a matched visual assay
(fixed-centre square ladder, then separate height and width ladders; constant elevation and angular speed; per-body
RF localizer; in-loop blank and object radiance; predeclared per-type primary statistics and Holm family; >= 5 runs
per arm in one submission), then a fixed-anatomy model comparison (sum / per-presynaptic-stream rectification with
signs preserved / adaptation + spatial suppression, with `gain_fb 0` and the feedback-hold as controls and a
bright-dark / ON-OFF / flicker / bar / grating specificity battery), then a transfer test with fixed parameters and
held-out stimuli.]

## Session 10, dynamics round 2 (2026-09-13)

Four threads (build:transducer, run:vnc-drive, run:takeoff-split, run:pm-bound), four Opus skeptics, ~100
cluster jobs. **No model default changed**: `git diff -- flyverse/brain.py flyverse/body.py` is empty, the
transducer is opt-in and OFF, and the two benchmark / optic lines the round decides about are named --
`scripts/benchmark.py:85` is de-scored (a measurement, not a model default) and `OpticParams.drive_clip_mv`
is untouched. Audits: `docs/audits/proprioception_transducer.md`, `vnc_drive.md`, `receptor_integration.md` G.6,
`anti_runaway.md` round 6. One verdict was **unsound on its numbers** (`vnc-drive` had quoted a partial
staging directory) and the audit was rewritten from the final 5-run data; its three conclusions were
re-derived at 5 v 5 by the skeptic and hold.

**A transducer on the never-driven VNC afferents: the wiring carries it, and DNa02 still does not fire.**
`senses.Proprioception` drives 941 wired-but-never-driven proprioceptors from the body state the VNC motor
neurons produce -- chordotonal (615 cells, `10 + 140 x legMN / 30` Hz), hair plate (113, `5 + 95 x legMN / 30`),
leg campaniform (12, 50 Hz on the ground, 0 airborne), haltere (201, = the haltere-MN rate) -- a
body-state -> afferent-rate model of the `Wind` -> JO kind, every law inside a literature bracket recorded as
an `op report` ledger row and none chosen against behaviour. In 60 s room runs, 16 flies, 5 runs per arm,
every stage the round-1 audits found silent now fires: **AN04B003 0.589/0.314 -> 3.713/3.690 Hz**,
AN07B035 +0.72/+0.31, AN06A026 +0.62/+0.18, **PS196_b 0.193/0.519 -> 1.437/1.505** -- the efference-copy input
of GLNO that `deficit_rotation.md` found at 0.03-0.16 Hz in all 160 phases -- and **GLNO 0.034/0.038 ->
0.664/0.657 Hz**, the first time the nodulus input of PEN carries anything in the plain fly (all `result`,
p 0.0079). The decompose of DNa02's input shows the sided excitatory term the audits asked for **does
appear** -- AN04B003/L -> DNa02_L +9.09 -> **+57.35 mV/s**, /R -> /R +4.75 -> +55.78, SNpp45 direct +39.97/+17.12,
the tonic VNC inhibitors relaxing by +5.0 to +7.4 each -- and is **cancelled on the spot** by the haltere
afferents' own second-step inhibition (PS059/L -90.8 -> -113.5, /R -49.4 -> -75.3; the tool's own
`cancelling_pair`). Net input -303 -> -235 and -363 -> -342 mV/s = **-1.52 -> -1.17 mV and -1.81 -> -1.71 mV of
steady `g` against a 7.0 mV threshold gap**: the operating point moves a fifth of the way, 2-3 % of the
~1,750 mV/s `deficit_turning.md` 7 computed as the requirement, and DNa02 stays at 0.02/0.10 Hz (`null`).
`missing input` is closed on this route and shown insufficient; what is left is the operating point, the
transducer's own cancellation, and a body model with no leg cycle.

**The steering that does change is a bias in a hand-written readout, and its sign never changes.** Yaw-rate
SD 2.848 -> 3.317 deg/s (`result`, p 0.032), straightness 0.9956 -> 0.9818 (z -20.7, every B/C/E run below
every A run) -- but the whole change is the leg-MN branch of `body.Locomotion`'s yaw law: leg L-R
**+0.151 -> +0.301 Hz** (z +33) = +0.50 -> +1.00 deg/s of fixed bias, with a wider temporal SD (0.414 ->
0.529 Hz), while DNa02 L-R stays a -0.07 to -0.10 Hz fixed offset. **The leg L-R is positive in all 400
fly-runs of all 25 room runs** (+0.098 .. +0.385) and stays +0.09 to +0.33 Hz at 1.6 rad/s *clockwise* with
DNa02 L-R = -18.2 Hz. This is not turning and must not be quoted as progress on the spontaneous-turning
gap. The leg channels alone (arm C) produce the whole effect; the haltere channel alone (D) none of it.

**The compass does not see the self-turn, with the sense on or off -- and the failure is sign, not dose.**
Bump drift -0.0047 .. +0.0036 wedges/s on the phase means of A / B / E against **4.0 ideal**, every one of
36 arm-phase-runs inside -0.0097 .. +0.0060, with realised self-turns of +96.5 to +108.3 / -85.9 to -99.1
deg/s; GLNO's L-R stays +26.93 to +28.07 Hz in every phase of every arm. Under the labelled stop-gap arm
(the Coriolis term on the haltere channel, `body.Locomotion`'s yaw scalar fed back as if it were a sense)
the efference-copy chain is driven **hard** -- the commanded haltere afferent rate reaches 80.10 Hz ccw /
74.50 cw and **PS196_b reaches 11.05/7.52 Hz, 13x its rest rate** -- and **its L-R still moves the same way
in both turn directions** (+3.52 ccw, +3.95 cw). The reason is structural: the only afferent class with a
two-step route into PS196_b is the haltere SApp, whose rate is one bilateral number in this body, and the
sided leg channels reach it only at k = 3. So `deficit_rotation.md`'s "missing input + a sign-0 link"
becomes **missing input (supplied), a route that cannot be sided in this body, and a GLNO -> PEN sign-0 link
that is moot until it is** -- and the actionable item moves to the body model (a side-split haltere MN
readout in `motor.py`, a leg cycle in `body.py`), not the connectome and not the GLNO transmitter call.
`AN07B037_a`, the route `deficit_rotation.md` 2.5 named, stays at 0.000-0.004 Hz in every room arm at 193
SApp synapses and 10.7 Hz of drive -- but fires at 3.1/2.2 Hz under the 80 Hz stop-gap turn, so it is a rate
threshold, not a wiring one. The stop-gap loop is positive but **stable** (haltere MN 9.38 -> 27.55 Hz,
realised in-turn gain ~0.26; 0.163 at the ground operating point), and stays a control arm, never a default.

**Nothing is adopted, and the suite says why.** With the sense on, `rest.spikes_per_step` reads **6.0
against `< 5`, FAIL in both GPU draws** -- by construction, because the check is defined as "no input" and
740 tonically-driven afferents are input -- and four further bench measures move by 37-81 % while still
passing loose one-sided bounds: `taste.MN9_hz` 10.93 -> 4.15 (-62 %), `smell.PN_hz` 7.86 -> 2.82 (-64 %),
`smell.KC_active` 816 -> 511 (-37 %), `walk.GF_max_hz` 4.63 -> 8.39 (+81 %). A leg/haltere transducer moving
taste and olfaction by 60 % is a broad cross-modal effect on a bare `brain.Brain`, not a `rest`
redefinition problem, and whether it is real or an artefact of the probe's body-less harness is unmeasured;
the two bench "draws" are bit-identical on 8 of 10 checks, so the bench carries **one effective replicate
per arm**. The transducer ships as a **swappable module, default OFF**. Closed this round: the opt-in
path's checkpoint gap (`BatchSim.state_dict` now carries the motor snapshot, `load_state_dict` restores it,
a partial `reset(rows=)` zeroes only the selected rows; `tests/test_proprioception.py::CheckpointAndResetTests`).

**The take-off cost of the receptor signs belongs to the histamine silencings, and the rule it questions is
not the flip rule.** Five arms in one submission, four runs each, then a fresh-seed replication at brain
seeds 4/5/6 (7 runs per arm pooled, 33,600 fly-s each): the 17,256 two-source histamine silencings alone
(90 % of their entries photoreceptor -> medulla) reproduce the shipped default -- `compare` **null** on hops,
escape, voluntary and the GF median (pooled hops 0.840x [0.669, 1.054]) -- and beat off on all four
(z +9.4 / +3.4 / **+29.2** / +4.0, 7.21x on hops); the **larger** 27,207-entry glutamate class (T1 / Dm9 /
Lai) reproduces off on all four, with **0 voluntary take-offs in 19,200 fly-s**. A 3,832-entry random draw
from the carrying class, matched to the Brain side's entry dose and |W|, also reproduces off -- so dose
measured in **entries or |W| does not order the arms**. What this design cannot close, and structurally
never can: **postsynaptic cells touched** orders every arm monotonically with the outcome (off 0, Glu 2,091,
Random 2,505, His 10,411, shipped 14,161) and an entry-matched subset can never match its class's cell
count. The **escape** channel stays unattributable, as in G.5: the reported batch's 103.6 % does not
replicate (fresh seeds 0.633x [0.353, 1.116]), and at 7 v 7 even `off` vs shipped is `compare`-null on
escape (|z| 1.92 < 3). Off's voluntary rate is not exactly zero either -- 1 in 86,400 fly-s over 18 batches.
The class is a product of the **silencing rule**, not the contested-flip rule (`fast_net_abs none`,
`fast_sign_abs 0`, `flip_contested` empty on 45/45, sources unanimous), so what goes on
`docs/NT_INTEGRATION.md`'s list is the silencing rule's premise on photoreceptor -> medulla edges -- and
first as an `optic.py` question (the receptor factor is applied to photoreceptor -> rate edges too), never
decided on the room take-off rate.

**`walk.power_max` decided from the data: de-score it.** Under the shipped defaults it is 48.4805 Hz in
12 of 12 independent GPU draws, **1.5195 Hz** under a hand-set 50 -- a margin **inside the worst single-arm
scatter on record** (`out/sk_les_holds_all/holdOptic_r{0..3}.json` span 11.86 Hz on the same arm; the `off`
suite arm spans 1.56 Hz). It is **non-monotone** in the one three-point scan (LPi x1/x2/x4 = 51.51 / 48.00 /
48.48) while the quantity that gain was added for is monotone; the two path gains move it in **opposite**
directions (DN->VNC x1 = 32.41, both gains off = 34.24); and across the four take-off hold arms its rank
correlation with the room take-off rate is **Spearman -0.600** -- the two arms that FAIL the check are the
two quietest rooms. The 50 is not an animal number: it is `body.Flight.takeoff_power_hz`, 50 Hz held 0.3 s,
applied to a per-frame maximum, and the sustained form (`walk.power_sustained_hz`) is the check that keeps
that referent. `scripts/benchmark.py:85` now takes the `notnone` form `loom.escape_cm` already uses -- with
the precision that `benchmark.py:135` makes `notnone` pass whenever the value exists, so the row stays **in
the pass tally** as a report. **Not** recommended: re-deriving the referent from a gain scan. The drive-clip
retirement candidate (`OpticParams.drive_clip_mv` 35, `optic.py:90`) is 10 PASS / 0 FAIL in 6 of 6 draws at
`walk.power_max` 49.2480 and `walk.GF_max` 4.63 -> 13.26; **nothing is adopted** -- the adopt-alone rule needs
its own 29-check suite x >= 3 and the room take-off protocol, and `motion.min_dsi` 0.2371 sits 0.0040 below
the lowest shipped-default draw on record (a `result`, small against the 0.1 bound, and irrelevant only
because nothing is adopted). De-scoring `walk.power_max` does **not** license retiring LPi x4, the drive
clip or the AL LN override.

## Session 10, object round 2 (2026-09-13)

Six threads (build:sphere, build:synthetic, build:hooks, then the baseline / compare / export
batch threads), five Opus skeptics, 758 cluster processes in two submissions (238 + 520). **No model
default changed**: the three new `OpticParams` fields (`stream_rectify`, `stream_adapt`,
`spatial_suppress`) plus `fb_hold` all default to `None`, are bit-identical off on a
deterministic backend (`tests/test_optic_hooks.py`, 14 tests; `git show HEAD:flyverse/optic.py`
loaded as a sibling module, `torch.equal` on drive/v/adapt/delta_rate over 12 frames of an
18-type subset), and **nothing is adopted**. Audits: `object_matched_assay.md`,
`object_synthetic_stimuli.md`, `optic_stream_hooks.md`, `object_baseline_r2.md`,
`object_compare_r2.md`, `object_export_r2.md`, `receptor_verification.md` "Object round 2".

**The matched assay Neurome asked for exists, and the LC size effect went away with the
confound.** `probe_object_matched.py` holds the ball at a fixed elevation, distance and angular
diameter on a constant-speed arc: realised centre-elevation deviation **0.0 deg** at every rung,
angular-diameter deviation **<= 1.4e-14 deg**, speed **40.000 deg/s**, against the old ladder's
elevation **0.88 / 4.34 / 8.66 / 13.71 deg**, diameter shrinking 30.18 -> 19.52 within a sweep
and speed 45.84 -> 18.79. Radiance is captured **in the loop, for both arms** (`retina.mode =
in_loop_capture`); the footprint recomputed from the exported Parquet alone reproduces the
probe's own numbers to 2.2e-16 at every rung. On that assay, at **6 runs per arm in one
submission** with a predeclared 12-member family per LC type: **LC11 12/12 `null`** (smallest
p 0.180, p_holm 1.000) and **LC10a 11 `null` + one `result`** -- 30 deg drive median
+0.0155 +- 0.0132 mV against a null of -0.0046 +- 0.0060, z +3.37, p 0.0087 -- which **fails
Holm at p_holm 0.104**. By the predeclared call rule **no size preference is called for either
type**. LC10a's one unadjusted `result` sits inside its own published 15-30 deg range
(Schretter 2024) and is exploratory; the LC11 excess over null is largest at the smallest rung
(+0.103 / +0.042 / +0.051 / +0.009 / +0.015 / +0.005 mV, Spearman rho -0.216, p_perm 0.206) --
the Keles & Frye direction, inside the scatter. `spikes_median` is exactly 0.000 +- 0.000 in
every arm of every rung, so six of each family's twelve members are structurally uninformative;
the LC populations are not silent, they emit essentially no spikes in this protocol.

**Upstream the figure is large, clean and size-monotone the wrong way.** `diff_signed_best_cell`
(a max over cells) reaches `result` surviving its own family's Holm at **T2 from 11 deg**
(+0.0141 -> +0.0442, z +3.8 -> +23.0 against a null of +0.0081 +- 0.0016), **Tm5Y from 11 deg**
(z +4.7 -> +28.2), **T3 and TmY21 from 20 deg** -- and 4.5 deg is null or negative at all four.
The population `diff_signed_mean` stays `null` at every rung for every type (|diff| <= 3.8e-4 mV)
and `diff_abs_best_cell_mean` gives a third answer again; the three statistics are kept apart, as
Neurome asked. The old headline `diff_max_over_cells_mean_mv` does move at the LC types -- LC11
+0.159 vs a null of +0.053 (z +10.6), LC10a +0.165 vs +0.074 (z +3.6) -- **only at 30 deg**, i.e.
at the large end, on a within-run maximum over 143 / 275 cells.

**The fixed-anatomy model comparison: no mechanism class passes.** Eight arms (base, gain_fb 0,
per-stream rectification, 100 and 300 ms adaptation, spatial suppression, and the two
combinations) on the matched ladder at 5 runs per arm, plus a seven-stimulus specificity battery
at 4 runs and three benchmark draws, all in one submission of 20 jobs = 520 processes.
**Rectification is the only arm that carries the predeclared carrier figure** -- T3
`diff_signed_best_cell` **6.0 / 10.5 / 9.0x** base at 4.5 / 8.8 / 11 deg and T2 3.1-3.7x, `result`
on both questions at every small rung -- and it fails on four counts. (1) It **releases exactly
what Keles 2020 says LC11 must not release**: at LC11 the grating rises 0.611 -> 3.202 mV
(z +98.8) and 0.042 -> 2.92 Hz, the full-field flicker 0.145 -> 2.994 mV (z +66.1) and
4.71 -> 10.67 Hz, the bar 0.182 -> 0.705 mV. (2) The figure it creates **grows with size**
(T3 0.021 / 0.037 / 0.045 / 0.055 / 0.058 across 4.5 -> 30 deg; LC11's max-over-cells excess
Spearman rho +0.937) -- a large-object figure. (3) It is **entirely in the extremum**: the
per-body RF-windowed T3 and T2 medians move by at most 27 % and read `null` against base in
**all 48 arm x type x rung rows of every arm**. (4) It **shifts the operating point** -- T3's
blank-arm mean deviation +0.00002 -> +0.0284, LC11's blank-arm drive 0.093 -> 0.348 mV -- which
is what the uniform flicker stimulus is in the battery to detect. **LC11 output does not follow
in any arm** (`lc11_follows` false 8/8). Adaptation at 100 and 300 ms is inert (T3 0.66-1.01x
base, every row null, no release, no bench cost). Spatial suppression produces no figure and
**costs the escape section**: `loom_escape.GF_peak_hz` 50.0 -> 31.2 Hz and escapes 1.0 -> 0.33,
two of three draws failing checks base passes in all three, replicated on the same GPU model by
the rect+suppress arm. Every hook parameter -- the streams, the modes, tau, gain, k, radius -- is
a **hand-set hypothesis, not a datum**; the `neg` mode preserves the sign of W but inverts the
sign of the signal, a stand-in for an unmodelled OFF pathway. Nothing is adopted and no default
moved.

**The ON/OFF question the battery could not answer was answered afterwards, on CPU.** The batch's
`flashon` / `flashoff` rows are a **bright** and a **dark** periodic square, and the spec statistic is a
whole-window mean, so they separate bright from dark and pool both transition polarities -- the
predeclared `T2_T3_keep_on_and_off = true` is a bright-vs-dark statement. Re-deriving the split
from the 2,048 stored recordings (`probe_synthetic_stimuli.py analyse --transition-window-s 0.3`
-> `out/interp/objr2c/spec_transitions.json`, 46,144 rows, problems none) gives the real answer:
**every arm keeps a T2 and T3 figure on both the ON and the OFF window in both flash families**,
never below 0.56x base's, so the "keep both transitions" half of the Keles 2020 constraint holds
for all eight arms. What rectification changes is the **asymmetry**: base's two windows sit within
6-30 % of each other, while `rectify` tilts the dark flash's T3 toward ON by 3.1x
(0.142 / 0.045 mV) and its T2 toward OFF by 1.5x, and `rect_supp` does the same at T3 (2.1x);
adaptation and suppression leave the ratios near base's. Against the matched blank/blank floor on
the same edge schedule, though, almost every 0.3 s window is at its own null (0.68-1.71x) -- only
`rectify` and `rect_supp` on the dark flash's ON window clear it (2.36x, 2.38x). Magnitudes over
4 runs of a max-over-cells statistic, no verdict. `object_compare_r2.md` 6.1.

**A finding about the shipped model that the battery produced for free:** base LC11 already
responds to a 30-deg grating (0.611 mV vs a blank/blank null of 0.085, z +21.5), to a 7-deg bar
(0.182, z +4.0) and to 2-Hz full-field flicker (4.71 Hz at the best cell), while an 11-deg dark
square sits at the null (0.081 vs 0.085, z -0.2) -- the opposite selectivity from the animal's
LC11, before any arm is applied.

**Export.** The matched ladder went to Neurome through the revision-2 exporter: 13 run directories
(`out/export/objr2-{ship,fb0}-d{045,088,110,150,200,300}-*` plus the ladder summary), 194 tables,
42,750,310 rows, `export.verify()` `problems: none` in all 13, with `paired_control_ids` /
`null_reference_ids` split (the section 3b field defect closed), `retina_radiance_blank`,
`retina.mode = in_loop_capture`, statistic definitions and both rank tests. It was delivered on
2026-09-13 (`objr2-*-20260913T2329*Z-*`, ladder `objr2-ladder-20260913T232912Z-72041020`) and
**re-emitted on 2026-09-14** from the corrected Result (`objr2-*-20260914T024*Z-*`, ladder
`objr2-ladder-20260914T024906Z-035363c0`); the 2026-09-13 directories stay on disk as the
superseded delivery. The compare arms and the synthetic rectangle ladders are **not** exported.

**Process, honestly.** Both batches lost their `cluster_run.py` client before the fetch, so
neither has a `<n> job(s), 0 failed` line and both were fetched by hand; the replacement is the
tool's own `verify` (`out/objr2/verify_sph.json` problems none over 84 runs; `out/objr2c/verify.json`
`expected_missing` 0 over 240 + 256 + 24). Only **14 of 84** baseline consoles came across, so the
console-vs-JSON device cross-check the process rule requires exists for 14 runs, not 84 (the
compare batch has 240/240). **Arm was confounded with box**: the scheduler put one job per arm on
the least-loaded target, so base ran on a B200 and the two arms `rectify` and `suppress` on H200s
-- and base itself sat on a different GPU model for the sphere (B200) than for the specificity and
bench sections (H200). The rectification result survives because the two combination arms carry
the same hook on base's own GPU model, and the suppression bench cost survives for the same
reason; this was luck, not design. The analysis code was edited **after** the predeclaration stamp
and after the Result was written -- `scripts/object_round2_baseline.py` at 2026-09-14T00:58Z
against `baseline.json` at 2026-09-13T23:17Z -- so the 2026-09-13 Result and the 2026-09-13
exported `preference.csv` carried the pre-fix `spearman_perm` floor: `spearman_p_perm = 5.0e-05`
with an empty rho on the four `spikes_median` preference rows (all `role = primary`), where the
correct value is NaN. **Both were re-emitted on CPU on 2026-09-14** -- `analyse` re-run on the
already-fetched batch (`out/interp/objr2/baseline.json`, now NaN on those four rows, every other
number unchanged and the primary families identical) and the ladder re-exported
(`objr2-ladder-20260914T024906Z-035363c0`, whose `preference.csv` carries empty `spearman_rho` /
`spearman_p_perm` there). One cost of the re-emit: the 2026-09-14 directories were written from a
working tree that had drifted since the batch, so their `flyverse_commit` reads `unknown` with
`source_match.verified false` (19/29 loaded, 30/43 glob), where the 2026-09-13 directories carry
commit `653179b4...` verified 29/29 and 43/43. The compare audit doc was stamped 22:07:27Z, after
the 22:06:37Z submission; its `predeclared.json` (22:04:04Z) is the real stamp and the two are
byte-identical.

## Session 11, behaviour round 3 (2026-09-14)

Five threads (build:body-state, run:monoamines, run:unitary, run:guards, then integrate), five Opus skeptics with their own
fresh-seed house-cluster replications, 165 cluster jobs in six completed submissions (vncd3b 22, mono 25, unit1 28, unit2 12,
guard7 27, r3int 51; 0 failed) plus 58 skeptic jobs (0 failed), split across the rented H200s and the house B200s. **No model
default changed and nothing is adopted**: `git diff -- flyverse/optic.py flyverse/data/receptors_by_type.csv` is empty, every
new field defaults OFF (`Locomotion.cycle = None`, `BatchBody.leg_cycle = None`, the spec tokens `leg_cycle` / `haltere_sided`
/ `haltere_coriolis` unnamed, `LIFParams.w_syn_by_nt = None`), the shipped path is bit-identical on CPU with the cycle attached
and the sense off (`tests/test_body_cycle.py`, atol 0 on every brain tensor), and the 13 new ledger rows (`unitary.*`, `lit.HS/Mi4/KC/MBON11/PN.*`)
are all `op report`. Audits: `docs/audits/body_sided_state.md`, `monoamine_slow_term.md`, `unitary_strength.md`, `guard_suites_r3.md`,
`anti_runaway.md` round 7, `round3_integration.md`; verdicts in `receptor_verification.md` "Behaviour round 3". All four thread
verdicts and the integration verdict are **mostly sound**; the refuted claims (a false 4-run/5-run agreement statement, a
CUDA-only suite pass, a per-file `--arm-block`, three overstated wordings) are corrected in the audits and none reverses a conclusion.

**A leg cycle and a side-split haltere readout make the never-driven afferents fire at literature-typical rates, DNa02 fires, and
the fly meanders and drifts left -- by the connectome's own asymmetry, not by sensing a turn.** `body.LegCycle` (tripod, stance
`0.9328 s x v^-1.025` from DeAngelis 2019, swing 30 ms from Mendes 2013 with the 30-50 ms bracket UNCERTAIN, per-leg ground
speed `v - s yaw b` with `half_width_m` 1.0 mm UNMEASURED) is a stance/swing phase derived from the realised speed that feeds
nothing back into the walk; `motor.read_haltere_sides` reads the 8 L / 8 R haltere MNs the anatomy already separates. With the
cycle on, the commanded chordotonal rate goes 23.4 -> **88.1 Hz** (hair plate 14 -> 47, campaniform 50 -> 25, every one inside its
bracket), AN04B003 3.7 -> 23 Hz, and the leg-afferent term on DNa02 outgrows the haltere-afferent inhibition (AN04B003/L +57.9 ->
**+360.4 mV/s** against PS059/L -114 -> -208; DNa02_L net -233 -> **+110**): DNa02_L / _R 0.031 / 0.104 -> **0.540 / 0.383 Hz**,
clean-frame yaw SD 3.35 -> **7.87 +- 0.14 deg/s** (per seed 7.84 / 7.96 / 8.01 / 7.66 / 7.88; z +51, `result` at 5 v 5),
straightness 0.979 -> 0.826, 7 of 16 flies off the table top within 60 s, hops 0.46 per fly -- replicated on a B200 on the eager
path (2.75 -> 7.67; DNa02 0.505 / 0.385) and at fresh seeds 5-8 (`out/vncd3sk`, 4 v 4: DNa02_L 0.024 -> 0.559, z +61.5). The
side-split haltere (D vs C) and the Coriolis control (E vs D) are `null` on every behavioural and DNa02 row. Three things stop
this from being turning: **no clean frame in any arm of any batch exceeds 100 deg/s** (the animal saccades at 200-450 deg/s every
~250 ms; max on record 63.8), the extra yaw is a **fixed left drift** (+1.23 +- 0.16 deg/s in 78 / 80 flies, DNa02 L > R in 68 / 80,
from the connectome's inhibitors -- IN12B014's capped wiring is identical on both sides and its RATE differs, 11.1 vs 8.4 Hz -- with
the excitatory rows equalised by `conn_cap`), and the sided afferent term that does reach DNa02 (-0.35 Hz per tripod half-cycle,
-3.39 Hz on AN04B003) alternates at 7.9 Hz under the 80 ms motor filter with a DC part 2.5 % of the alternation, and correlates
with the realised yaw at -0.06 to -0.15 with the **kinematic** sign (the turn shapes the afferent). The leg-MN L-R stays positive in
400 / 400 fly-runs. **The level confound qualifies all of it**: the round changed the afferents' LEVEL (23 -> 88 Hz) and their phase
structure together, and the level alone accounts for the DNa02 firing; the level-matched control (`'all'` with `mn_ref_hz` ~3.5 Hz)
has not been run. The module ships opt-in; what a default needs is `body_sided_state.md` 8.

**The compass receives no signed report of the self-turn under any combination, and forms no bump at the shipped gains under any
brain configuration.** Wedge protocol at gE = gD = gR = 1: `compass.EPG.bump_survival_s` **0.00 s in 48 / 48 runs** (unitary's four
brackets on H200, the integration's four brain configurations on B200, the skeptic's fresh seeds), rate and width
`NOT_APPLICABLE`, PEN at 0.04-0.15 Hz because its net drive is +1.2 to +2.3 mV against the 7 mV gap; a per-transmitter scale
multiplies the tuned Delta7 inhibition and the untuned ring feedback by the same factor and cannot supply the type-level ratio that
gE 2 / gD 15 did. Efferent protocol under the gains: bump drift |mean| <= 0.005 w/s in every phase of every arm against +-4.0 ideal.
The signed report now exists at depth 1 (AN04B003 flip -9.8 to -12.2 Hz against nulls of ~1 Hz, every seed), reaches PS196_b at
depth 2 (-1.1 under D, **-3.00 +- 0.29** under ABC) and is **gone at GLNO** (|flip| <= 0.5 Hz; GLNO L-R +27-34 Hz at every phase) and
at PEN / EPG. The block is the GLNO fan-in plus the sign-0 GLNO -> PEN link, and no attractor for a report to move.

**The monoamine slow class is inert at 0.02, runs away at 0.2, and fails the taste check on the CPU.** Coverage: 72.8 % of the
1.88 M monoamine synapses are silenced in every model, the VNC / ascending / OA-VUM targets have 0 signed synapses, DNa02 / DNp09
/ MDN / DNp01 / MN9 carry no monoamine load; the dopamine class is 59 % DAN -> KC (+1, Dop1R2-led). `add_low` (the shipped 0.02):
every behaviour key `null` at 5 v 5 and again at 4 v 4 on the B200; spikes +3 %; `taste.MN9_hz` 10.93 -> 2.48 seed-locked on
CUDA and **5.09 -> 1.97 on the CPU, i.e. FAIL against `> 2`** (MN9 is the residual of +1,786 / -1,677 mV/s inputs; a sub-mV 5-HT
tone on the SEZ tips it). `add_mid` 0.2 and `add_high` 1.0 additive: runaway in 5 / 5 runs each (665.6 +- 3.2 and 1,277.0 +- 0.5
spikes/step at the abort; the KC <-> DAN loop -- KCg-m gives PAM08 163 mV per volley and carries 75.8 dopamine syn-eq per cell,
0.83 mV of tone per Hz of the DAN population at 0.2; deleting `W_slow` onto the 4,064 KCs on CPU removes it, a cell-matched
control lesion does not). `gain_mid`: x2.55 spikes, the MBONs zeroed by their Dop2R rows, OA-VUMa2 at 329 Hz, 1.5 voluntary
take-offs per fly, DNa02 at 0.1 Hz -- a baseline change, not a behavioural gain. The two literature anchors are 50-100x apart on
one scalar (Cohn 2015: ~0 mV on KCs; Longden 2010 / Maimon 2010: x1.5-2 OA gain on HS / VS), so **one class scalar cannot carry
the three transmitters**; the split per transmitter, a KC>MBON plasticity module and VNC receptor rows are what an adoption needs.

**One 0.275 mV for every transmitter is a cholinergic calibration; every data-anchored inhibitory bracket breaks the suite and
the room.** The one anchor with both the EPSP and the count measured (ORN -> PN, ~5 mV over ~23 synapses) is x0.79 of 0.275; no
per-synapse fast IPSP in a Drosophila central neuron is on record (the one insect unitary I/E, 0.28, is Periplaneta -- J Neurosci
34:13039 -- and belongs in the ledger). Brackets ACh x0.5 / 0.8 / 1.0 with I/E 0.25 / 0.5 / 0.75: `walk.power_sustained_hz`
**89-194 Hz against < 50 in every bracket and every draw**, low also fires the GF at 128-130 Hz to a walk stimulus; in the room
`high` hops 39-42 times a minute and leaves the table in 3.5-7.9 s, `mid` is airborne two thirds of the time with a fixed DNa02
R-L bias of -11 Hz in 16 / 16 flies. Combined with the transducer (AC / ABC) the disinhibition turns the connectome's sidedness
into a +1.12 Hz DNa02 L-R of one sign in 80 / 80 fly-runs and a +5-6 deg/s left drift on a fly that is off the table in 11 s;
the transducer halves C's hops (43 -> 22) without making it steer. B does not interact with A (AB vs A `null` on every key).

**The two retirement candidates are refuted on the room, by the adopt-alone rule they were owed.** `no_drive_clip`: suite 27 / 0
/ 2 x3, room 5.069 vs 3.125 take-offs per 1,000 fly-s (CI 2.279-4.181), the excess entirely voluntary (3.958 vs 1.944), escape and
walking-GF tail unchanged -- so the clip binds on the wing-power route, and round 6's walking-GF prediction is **not confirmed**;
the skeptic's B200 rerun at 4 v 4 gives the first callable verdict, `hops_voluntary_total` +10.5, z 3.57, p 0.029 `result`.
`pair_gain_lpi_x1`: 34.6 per 1,000 fly-s (11x), 451 escapes against 17, walking-GF median 38.1 Hz with **48 / 48 flies above the
33 Hz escape threshold** -- the factor is what keeps the walking giant-fibre drive under the threshold in the closed loop, and the
pinned `walk.GF_max` (4.63 -> 9.80 against 38) does not predict it. The shipped default's reference for these boxes: 3.125 (H200),
3.750 (B200), 3.89 (round 5). The `walk.power_max` triple 48.4805 / 20.1091 / 4.6292 reproduced the recorded value exactly in
every native GPU draw of the round and reads 57.38 / 26.66 / 9.76 on the CPU: a backend property, not a model constant.

**Process, honestly.** Every predecessor agent died at a session limit and the rented boxes were stopped ~5 h for funds, so four
of six batches were fetched by hand (`scripts/box_status.py`, `scripts/fetch_run.py`); the body-state five-run tables span two
submissions (seed 2 from `vncd3-f3bb50`, seeds 0/1/3/4 from `vncd3b`) and the audit's claim that the single-submission four-run
tables differ in no verdict is false (15 of 765 pairwise and 7 of 548 room-table verdict cells flip; no headline row among them);
unitary batch 1's `--arm-block fam` resolved to one block per FILE and was dealt round-robin over two H200s (balanced, not
confounded, and reproduced in one block on the B200); the guards batch ran an intermediate `body.py` (md5 46e3c10a) so its
'transducer on' arms are the round-2 sense; every batch shipped uncommitted cross-task files. The integration batch (`r3int-9a78e4`,
51 jobs, one submission, one node, `verify` 88 / 88 JSONs 0 problems) is the round's clean design and its shipped arm on the eager
path is the reference for any future cross-backend replication.

**Next (the round-3 plan, in order; items 1-6 need the cluster, 7 does not).**

1. THE LEVEL-MATCHED CONTROL (house, ONE submission, ~20 jobs, blocks `fam_r<seed>`): arms shipped / `'all'` with
   `mn_ref_hz` set so the window-mean chordotonal rate is 88 Hz (~3.5 Hz; a LABELLED control) / `'all+leg_cycle'` /
   `'all+leg_cycle+haltere_sided'`, 5 brain seeds x 16 flies x 60 s, the `probe_vnc_drive` room protocol with the
   clean-frame statistics, DNa02 decompose per arm, and the same 4 arms on the efferent compass at 4 seeds (so the
   flip rows are callable). Decides the one thing the round could not: whether the per-leg / per-phase structure
   contributes anything beyond the afferent level (if the level control fires DNa02 as C does, the cycle's
   contribution is its level). Do first: add the ledger rows `lit.walk.step_frequency_hz_at_speed`,
   `lit.walk.stance_fraction_at_speed`, `lit.walk.swing_duration_ms` (30-50 UNCERTAIN),
   `lit.walk.outer_leg_step_ratio_in_turn` (op report; CPU) and fix the `sided_frames` lag + the DNa02 mask before
   the analysis.
2. THE ADOPTION-LICENSING RUN FOR THE MODULE (house, ONE submission, one block `fam_lic`): shipped vs
   `'all+leg_cycle+haltere_sided'` on `benchmark.py --sections hops` (2,400 fly-s) x 6 draws and the
   `batch_sustain` room take-off protocol (16 x 300 s) x 6 batches at seed-matched seeds, plus the 29-check suite
   x 3 for the record (with the statement that 28 of 29 checks cannot carry the sense), plus the room ledger rows
   under the sided spec; verdicts `common.compare` at 6 v 6 and the two-sample exact Poisson. This is what
   `body_sided_state.md` 8 item 3 and guards 1b jointly say a default needs; run it only if item 1 shows the phase
   structure matters, otherwise the module stays a module.
3. THE ACh-ONLY UNITARY FAMILY (house, ONE submission, `--arm-block-map` so the family is ONE block; ~24 jobs):
   `w_syn_by_nt` {acetylcholine: 0.8} and {acetylcholine: 0.5} with inhibition x1 (plus the KC / LHN anchors as arms
   if budget allows) through the suite x 4 draws, the wedge compass x 4 seeds and the room x 4 runs with the
   transducer OFF and ON (integration 9 item 5b); `taste.MN9_hz` re-read as the re-calibration it is, not re-passed.
   Before submission (CPU): pin Kazama & Wilson 2008's primary EPSP (5 vs 7 mV), add the Periplaneta unitary I/E
   0.28 as a ledger row, cite or relabel `unitary.IoverE.chloride_driving_force`, and fix `probe_unitary`'s
   rounding-before-compare.
4. THE MONOAMINE CLASS SPLIT (code first, then ONE submission): thread B/C work on CPU -- `receptor_signs`
   slow_class per presynaptic transmitter (DA / OA / 5-HT) and `LIFParams.slow_gain_by_class` / `slow_tau_by_class`
   accepting the three keys (default None, CPU bit-identity test), separate E/I accumulators in gain mode; then one
   house submission, 5 runs per arm, blocks `fam_r<seed>`: off / DA 0 + OA gain 1-3 (with the opt-in
   `OpticLobe(slow=)` term so the HS / VS / Mi4 anchors are reachable) + 5-HT additive 0.02-0.2, with health, the
   plain-fly room, the suite on the GPU AND the same suite sections on the CPU (the CPU-path requirement the round
   exposed), and the compass rows under the OA arm (the ring leaves silence under octopamine; is it heading-locked?).
   A KC>MBON plasticity module gated by the DAN rate against `lit.MBON11.kc_mbon_depression` is a separate build,
   not a scale.
5. A TYPE-LEVEL RING MECHANISM, OR NONE: the round proved a transmitter scale cannot set the Delta7 : ring ratio;
   before any compass batch, a CPU structure pass (`interp_paths` / `structure.json`) naming what data-implied fact
   could change that ratio (receptor tiers on ER / ExR -> EPG, the GLNO transmitter, a conductance-based synapse) --
   if none exists, the compass is parked at 'no attractor at shipped gains' and the free-walking compass room under
   A at the experiment gains (4 seeds, one block, bump metrics + `circ_corr_heading`) is run once to close
   `body_sided_state.md` 8 item 5 (expected negative: the report is gone by GLNO).
6. NO RERUN OF THE TWO RETIREMENT CANDIDATES. Close `drive_clip_mv` and LPi x1 as NOT adoptable in
   `anti_runaway.md`; the next submission on that thread is only for a candidate REPLACEMENT mechanism (a bound on
   the optic -> spiking injected current seen by the wing-power route; a sign-correct LPi -> LPLC2 strength from
   data), scored in the room at >= 6 runs per arm with the two-sample exact Poisson and the 29-check suite x 3, one
   submission, one block.
7. BOOKKEEPING, NO GPU: apply the section-7 audit corrections; regenerate the integration pairwise with z to one
   decimal and the mask caveat; add `fetch_run.py` sha256 receipts and a `cluster_run.py --attach` mode plus the
   one-job-block guard; write the object round 3 NOTES entry; commit the round-3 tree in one commit with the
   fingerprints quoted.

## Session 11, object round 3 (2026-09-14, Astra)

The object half of round 3 ran as a separate workflow. After the API 529s took this session off it mid-round it was
handed to **Astra** under `docs/HANDOFF_ROUND3_ASTRA.md` (git-ignored, one owner per file); Astra's reply is
`docs/HANDOFF_ROUND3_REPLY.md`. **No commit, no push, no model or default change**: `flyverse/optic.py` and
`flyverse/brain.py` are byte-identical to the handoff snapshot (`out/round3_astra/start/state.json`,
`brain.py` 7fadb8d6..., `optic.py` afb78cfe...), and `brain.py`'s only diff is the *behaviour* round's opt-in
`LIFParams.w_syn_by_nt`, recorded `null` in all 180 same-device runs. Audits:
`docs/audits/object_samedevice_r3.md`, `object_rectangles_r3.md`, `object_localizer_r3.md`, `object_export_r3.md`.

**What ran.** The three predecessor H200 batches were completed, checked and given recovered scheduler receipts
after their polling clients died -- `objr3sd-d6f06a` (5 jobs = 90 paired runs), `objr3rect-e9a8f3` (12 jobs = 324
paired runs), `objr3rf-a2fc62` (10 jobs = 10 runs) -- and each was replicated at fresh seeds on the house B200s:
`r3sdcheck-00126d` (5 jobs, 90 runs, seeds 1000-1004), `r3rectcheck-6f0cbc` (12 jobs, 324 runs, object seeds
2000-2005) and `r3rfcheck-c82c1e` (10 jobs, 10 runs, seeds 1000-1004) -- 27 house jobs, 424 paired runs,
**0 failed anywhere in either half**. Every replication was one `cluster_run.py --arm-block fam --target house
--node <cluster-node>` call with the reference arm and its treatments in the same block; every replication recording reports
NVIDIA B200 on host `<cluster-node>`, every original reports NVIDIA H200 on one host per batch, so the round-2 arm/box
confound is closed inside each batch. The `FETCH FAILED` lines in the native logs are **transfer** failures, not
failed simulations: the interrupted SCP transfers were resumed through compressed tar with SHA-256 checks, which
caught two partial files that a size-only compare had passed (one rectangle NPZ, one sphere JSON); both were
replaced and every final batch verifier reports no problems.

**Same-device (`object_samedevice_r3.md`): the H200 batch reproduces in full, the B200 batch is PARTIAL, and the
reference arm itself is not the same on the two boxes.** Rectify exceeds base at every small T3 / T2 maximum in
both batches (H200 ratios T3 6.281 / 9.528 / 13.063 and T2 2.356 / 3.369 / 3.068 at 4.5 / 8.8 / 11 deg, all six
rows Holm p .023810; B200 T3 6.131 / 9.028 / 12.598, T2 3.522 / 3.421 / 3.225). The declared reproduction rule
needs T3 *and* T2 at all three small rungs: on H200 both pass (`REPRODUCES`), on B200 rectify T2 at 4.5 deg reads
z **+2.66825** against its own blank/blank null and so is `null` under `common.compare`'s z >= 3 gate despite
Holm p .023810 -- hence **PARTIAL**. The separation there is complete (the five rectify runs all exceed all five
null runs, U 25/25, p at the exact 5 v 5 floor .0079365); the z gate is missed because one null run inflates the
null SD to .005302 against the H200 null's .001147. Two things the audit reported narrowly and the skeptic
corrected in the body: the **large-rung companion effect is a three-batch effect, not a B200 observation** --
rectify's windowed-median T3 companion exceeds base at 20 deg (.006050 +/- .000709 vs .004445 +/- .000414,
z +3.88) and 30 deg (.007377 +/- .000775 vs .004259 +/- .000400, z +7.80) and its T2 companion at 30 deg
(z +3.56), all Holm .039683, on **H200 as well as B200**, and round 2 already carried the T3-at-30 row (z +9.795,
ratio 1.827), so `companion_all_null_all_rungs` is false in both round-3 Results and the "typical cells unchanged"
reading holds only at the three small rungs; and the **base arm's own T2 maximum crosses the effect gate on the
B200 and not on the H200** (base T2 vs its own null: B200 z +6.12652 at 8.8 deg and +9.29988 at 11 deg, both
Holm .023810 = `result`; H200 z +0.35 / +1.02 / +1.96, `null` at every small rung). The two round-3 batches have
**identical source fingerprints** (all 44 `files` entries and all 29 `files_loaded` entries match) and differ only
in GPU model, box, seeds and execution history -- which still does not isolate a causal GPU-model effect, and no
base-vs-base row between them was computed. Suppress's primary `null` pattern and the absence of an LC11 rescue
reproduce in both (`both_null` on T3, T2 and LC11; `lc11_follows` false in every arm of both batches). **Nothing
adopted; rectify and suppress remain hand-set opt-in control arms.**

**Rectangles (`object_rectangles_r3.md`): 40 of 40 primary verdicts `null` in both batches, no animal-shape
preference called, and retinal contrast is matched only at width >= 8.8 deg.** Each batch is 216 object runs
(9 shapes x 2 contrasts x 6 seeds x 2 lobes) + 108 blank/blank null runs (9 shapes x 6 seeds x 2 lobes) = 324;
nulls carry no contrast, so a shape's six nulls serve both its dark and its bright family and those two families
are **not independent of each other**. The object's Weber contrast is fixed at +/- .995, but fractional coverage
of the retinal acceptance kernel caps the per-column change wherever a rectangle is narrower than a column's
4.5-deg acceptance: peak per-column effective contrast reaches +/- .995 **only at width >= 8.8 deg -- three of the
nine rectangles, all in the width ladder**, while the whole height ladder runs at width 4.4 and is capped at max
coverage .88530 / extreme -.65262 (its 2.2 and 4.4 rungs, .42649 / -.22826 and .65590 / -.53849, are matched to
nothing). The stamped `effective_contrast` clause had already declared that cap and its 0.885 value before
submission; what the radiance capture refutes is the *shorthand* "retinal contrast matched by construction", not
the design. The one sub-.05 row is B200 dark LC10a at width 15 deg (diff +.055055 mV, z **+2.17788**, Holm
p .021645) -- `null` under the z >= 3 gate, and the **H200 batch has the opposite sign at the same type, ladder,
contrast and rung** (-.015022 mV, z -.45490, Holm 1), so the question is open in both directions. The
size-monotonicity reading is a **two-directional knife-edge, not a non-replication**: of the 16 declared
(ladder x contrast x type) joint tests exactly one passes in each batch and it is a **different** one (B200
`hlad:dark:T2`, H200 `hlad:bright:Tm5Y`), the two batches agree in sign on both statistics, and both calls flip
under a different permutation seed (H200 median p .049248 -> .053047; B200 `hlad:bright:T2` .050797 -> .047048) --
so no upstream size-monotonicity conclusion leaves this round, and the declared rule (`docs/INTERP.md` 10.4 item
2) forbids the cross-batch row comparison anyway. Two protocol caveats belong beside every null: Keles & Frye's
Figure 3D height sweep held **width at 30 deg**, so the width-4.4 height ladder here is an adapted assay with no
published counterpart (the width ladder, at a fixed height of 8.8 deg, does match Figure 3E), and **Figure 3B says
maximum contrast is not LC11's optimum** -- dropping OFF-object Weber contrast from 100 % to 30 % "nearly doubled
the amplitude of the calcium response", while both ladders here run at |Weber| .995.

**Localizer (`object_localizer_r3.md`): neither LC population localizes under the static 4.5-deg probe, on either
lobe, in either batch.** LC11 has **zero fits at the fixed z = 5** on both lobes in both batches. In the B200 `fb0`
map the blank-selected threshold is z* = 4, which yields 4 of 143 LC11 fits against 1 of 143 blank fits -- and
still fails both the population-coverage and the spatial-enrichment criteria; shipped B200 LC10a has free-peak
enrichment above 2x chance but insufficient fitted coverage. **The Mi1 positive control (pooled coverage ~53 % in
both batches) is on `optic_dr`, not on the `drive_mv` quantity the LC negative is measured in**: every control type
(Mi1, T2, T3, Tm5Y, TmY21) is an optic-lobe rate unit, and the only `drive_mv` rows in the map are the 418 LC
bodies, so the control shows that the grid, the dwell, the pooling and the fitter work on rates, not that a
`drive_mv` receptive field of the same strength would have been detected. Two deviations from the request, both on
record: the probe was **4.5 deg, not the 2-4 deg asked for, and static rather than the moving physiological RF
probe**, and the presented grid was 1,466 of 1,787 reachable nodes (82 % of the eye; the per-cell box restriction
applies at fit time only). Read the result as stimulus-, prior- and fit-limited: it does **not** license "no
stimulus can localize LC cells", and it is not a statement about firing -- the fitted quantity is received drive,
and the LC populations emit essentially no spikes in this protocol (137 of 143 LC11 bodies emit exactly zero over
the 1,466.5 s recording). The round's ladder windows were left unchanged: all 143 LC11 windows are the anatomical
fallback, and using new RF maps in inference needs a separately stamped analysis. The final RF thresholding was
corrected to agree with the fitter on zero-MAD peaks and amplitude / node floors; both the original and the fresh
maps were re-analysed under the corrected rule, the fix changed no reported number (28 of 28 cells identical), and
all three stamped rule dictionaries are unchanged.

**Skeptic passes: all three `mostly sound`, with every number reproduced twice.** Same-device: every headline
statistic recomputed twice, once with the shipped script and once with an independent numpy/scipy implementation
that never imports the project's reducers, plus a raw-array check of the max-over-cells primary; 1,860
raw-derived values and 928 comparisons / 204 scored families matched per batch, 0 differences. Rectangles: an
independent primary reimplementation reproduces all 40 members per batch to max |deviation| 1.332e-15 (H200) /
1.665e-15 (B200) with 0 verdict mismatches, plus 1,280 comparisons / 236 scored families, 1,728 preference and 756
contrast statistics per batch and 3,888 raw values on the originals. Localizer: an independent scalar
implementation matched 7,031 bodies x five runs on each lobe in all four maps, 0 differences. The delivery checker
scans all 1,712 tables and 91 LC frame lattices and reproduces nine complete paired LC drive mean / SD traces
exactly. The governance caveat stands and is stated in each audit: **these passes are Astra's own separately
implemented checks plus fresh cluster runs, not a second agent or a human review** -- and in the same-device audit
the `## Report`'s self-review is now filed under `author_self_review`, with the independent pass as the `skeptic`
block.

**Export (`object_export_r3.md`).** Entry point `out/export/objr3_index.json`, schema
`flyverse.neurome.export/2`, revision 2: **95 directories, 1,712 tables, `problems` empty**, over four component
indices -- `objr3_r2compare_index.json` (41 directories), `objr3_samedevice_index.json` (16),
`objr3_rectangles_index.json` (36), `objr3_rfmap_index.json` (2) -- with `objr3_tables.json` listing every table
and `objr3_skeptic.json` the independent delivery checks. All 91 sphere / rectangle rungs carry all 418 LC bodies
at all 1,200 frames, with run means and sample SDs. The four fresh-seed **B200 replication Results are linked
separately** under `replication_evidence` -> `out/export/objr3_house_results/index.json` (sha256 466c4fb2...,
`n_results` 4: the house same-device Result, the house rectangles Result and the two house RF maps), kept as
native Results rather than folded into the 95 original directories. Per-section reference devices are explicit
(round-2 sphere base B200; specificity / benchmark base H200) and the literal verdict text `null` is preserved
through the CSV round trip. One honest limitation: **the long RF recorder retained only node / role-window means**,
so the delivery ships node response tables and says so, rather than a reconstructed per-cell frame chronology;
getting that would need a new recording, not a reconstruction from means.

**What the object half owes.** (1) A **separately declared** localizer -- a moving-probe RF assay with blank
controls and a level-matched, smaller probe -- before any absence claim under other stimuli; the static 4.5-deg
negative is scoped to its own protocol. (2) **More LC10a runs under a new declaration**, to settle the width-15
row the two batches sign-reverse on and to give the free-peak-enrichment coverage test real power. (3) The
**Neurome receptor-tier question**: LC11, T2 and T3 (and Tm5Y / TmY21 / TmY13) still run on the presynaptic-sign
fallback for 100 % of their input, and that is the single change that would replace a modelling assumption with
data. There is no adoption decision to execute from this round, and any future candidate still needs
specificity / ON-OFF evidence, the full benchmark draws and the applicable room / anti-runaway guards.

## Session 11, process, owner decisions and the FlyWire / BANC survey (2026-09-14)

**Process, across both workflows.** Every predecessor agent died at a session limit, and the rented H200s were
stopped about five hours when the account ran out of funds and then restarted, so four of the six behaviour
batches lost their `cluster_run.py` client and were attached to and pulled by hand with the two helpers written
for it -- `scripts/box_status.py` (`--wait` on a live submission; scheduler states by job ID) and
`scripts/fetch_run.py` (resume a named run directory, compare every local / remote file). Astra's object batches
hit the same dead-client failure on all three H200 submissions and recovered them from the scheduler's own job
IDs. API 529s took the object workflow off this session mid-round; it was handed to Astra under
`docs/HANDOFF_ROUND3_ASTRA.md`, which names every file's owner, and came back as `docs/HANDOFF_ROUND3_REPLY.md`.
The **rented boxes were destroyed after the fetches, and those fetches were size-verified, not hash-verified**:
nothing in `cluster_run.py` or `fetch_run.py` computes a digest, so unitary batch 1's "md5-verified against the
box" is corrected to "size-verified (`FETCHED.txt`)" and `r3-h200b` is gone, which is why it cannot be re-checked.
The rule that follows is `docs/INTERP.md` 10.4 item 19: `fetch_run.py` writes a per-file sha256 receipt computed
on the box before transfer, and only a receipt path licenses the word "verified". Astra's house transfers already
work that way -- compressed tar with SHA-256 compare -- which is how the two partial files with matching sizes
were caught; no object box was destroyed.

**Three owner decisions, recorded.** (a) **`LIFParams.w_syn_by_nt` is KEPT, as an opt-in instrument, not a
mechanism.** Default `None`; shaped weights byte-identical under `None` / `{}` / all-ones (md5 6c36faf3...), 47
CPU tests including bit-identity. The governance point goes on record with it: `flyverse/brain.py` was on this
round's never-edit list and thread unitary used the "a new `LIFParams` field is unavoidable" carve-out, whose two
conditions (default `None` plus a CPU bit-identity test) were met -- and which must from now on also carry a line
in the hand-off. No bracket of the field is adoptable (`unitary_strength.md` 4). (b) **`MotorRates.haltere_L` /
`_R` as fields is DEFERRED.** It is a one-line `read_motor` change plus `haltere_side_groups` cached on
`WingGroups`, but it forces regeneration of `tests/test_bit_identity.py`'s golden (which hashes
`asdict(fb.motor())`); until that is wanted, `motor.read_haltere_sides` stays a free function beside `MotorRates`.
(c) **The adopt-alone rate-half is restated as a two-sample exact Poisson comparison.** The CI-containment form the
guards audit cited was never the rule of rounds 4-6 (round 5 read it one-sided, "not worse in either route"), it
ignores the candidate's own sampling error (14.9-15.6 % false failure for an identical true rate at 3-6 batches
per arm, and it does not improve with n), and it was applied asymmetrically -- the transducer arm, also outside
the CI but *below* it, was passed. From round 7 on: the rate-half is a two-sample exact Poisson (conditional
binomial) at equal exposure, quoted beside the run-level `common.compare`; the direction is stated once for every
arm; and the prescribed replication comes from a power calculation on the observed contrast (>= 6 runs per arm
here: power 0.85 at n 6, 0.50 at n 4), not a fixed ">= 4".

**FlyWire FAFB v783 / BANC v888 survey.** `docs/audits/flywire_banc_survey.md` reads the two public Princeton /
FlyWire female releases on disk against the MaleCNS v1.0 graph flyverse ships on (CPU only, no model change);
`docs/CONNECTOME_BACKENDS_SPEC.md` is the implementation spec handed to Astra. FAFB v783 is the brain with both
optic lobes (139,255 cells, 50.7 M synapses, six-class per-cell NT probabilities, a `column_assignment` table of
45,528 cells over ~790 columns/side); BANC v888 is brain **and** VNC (158,262 cells, 23.6 M synapses, a *verified*
transmitter for 65,369 cells, `Body Part` / `Function` / `Nerve` labels for sensory cells, 259 hemilineages).
Exact type-name overlap with MaleCNS covers 59 % (FAFB) and 72 % (BANC) of MaleCNS cells -- BANC carries the VNC
types at identical cell counts (`AN04B003` 6/6, `IN12B014` 4/4, `PS059` 4/4, `GLNO` 4/4) -- and
`flyverse/data/type_aliases.csv` already holds 22,200 alias rows, so a loader adapter is a name normalisation, not
a re-typing. The round-1..3 turning anatomy re-reads on the female CNS with the same top rows (PS049 / PS059 ->
DNa02 GABA, VES051 / LAL126 / AOTU019, AN04B003 and LT51 excitation, IN12B014's symmetric contralateral pair,
PS196a -> PS059 contralateral), and DNa02's excitation : inhibition by presynaptic transmitter is 2.0 : 1 in
MaleCNS, 1.96 : 1 in FAFB and 1.7 : 1 in BANC -- so the net-inhibited resting state rounds 1-3 rest on is a **rate
statement, not a reconstruction artefact of the male graph**, which is what a reviewer will ask. Two caveats:
synapse yield scales MaleCNS : FAFB : BANC ~ 1 : 0.6 : 0.3 (counts are not comparable across releases without a
per-release scale; ratios within a release are), and BANC's optic lobes are under-proofread (T2 853 vs FAFB 1,466
vs MaleCNS 1,630). For the NT thread: about 400 of the 2,361 `unknown` cells MaleCNS silences carry a
classical-transmitter prediction in BANC, and MaleCNS's `serotonin` class splits SER / DA / tyramine across
sources. Ranked uses are in `TODO.md` section F; nothing here changed a model.

### Session 11 addendum: connectome backends merged (2026-09-14)

Astra implemented `docs/CONNECTOME_BACKENDS_SPEC.md` on `feat/connectome-backends` (d9f8cf2) and an independent Opus
review (`docs/audits/connectome_backends_review.md`) verified it with its own harness: a full MaleCNS recompile with the
branch code reproduces the shipped cache byte for byte, file for file (neurons.parquet c50c598a..., W_post_pre.npz
ac131529..., sign0_counts.npz bf01d724...; fingerprint md5 ef23cc27... unchanged, key set unchanged); the FAFB column map
hex1 = q + 18, hex2 = p + 20 passes the mirror (max 1.583 deg over 772 shared coordinates) and T4-offset (min cosine
0.9818; the axis-swapped control gives -0.949) validations and the column count (1,581), and fails the strict DRA-on-rim
check (100/126 on the flat dorsal band; 118/126 within two rows of the curved envelope) -- recorded as an expected failure
in the acceptance gate, not hidden. Verdict merge with fixes; B1-B4 (BANC wing MNs silently empty through a naming gap,
`$FLYVERSE_CACHE` moving the MaleCNS default, a FAFB cache built by an earlier compiler, the gate omitting the failing
validation) fixed in 34c2eb0 and merged fast-forward. CPU suite 360 passed / 19 skipped. `docs/audits/connectome_backends.md`
carries the numbers; the walking replicate (BANC, 5 seeds x 16 flies, house B200, one submission) is the first
cross-connectome behavioural result: the female CNS walks straighter than the male at the shipped defaults (clean yaw SD
0.275 +/- 0.004 vs 2.641 +/- 0.146 deg/s; DNa02 silent bilaterally) and the leg-cycle yaw increase replicates
qualitatively (0.381 -> 2.651 deg/s, `result` within-dataset at 5 v 5) while the MaleCNS neural pattern does not
(BANC DNa02_L silent under C, DNa02_R 0.0985 Hz vs 0.54/0.38; PS059 0.00087/0 Hz vs 20.06/16.77). Descriptive across
reconstructions, not a sex test: animal, lab, synapse threshold and yield, optic capability and GPU all differ. A
pre-existing bug in `probe_walk_straightness.py`'s table-exit counter (bounds tuple read as half-extents) was fixed as a
reporter-only change (`metrics_version` 2).

## Session 12, the level-matched control (2026-09-15)

**What ran.** ONE house submission, `vncd4-8dd183`: **24 job(s), 0 failed (32.3 min)** on 8 x B200 -- the
`probe_vnc_drive` room at four arms x five brain seeds (16 flies x 60 s, 5-60 s window, blocks `fam_r<seed>`) plus
the same four arms on the efferent compass at four seeds (`fam_c<seed>`). Arms: **A** shipped (no sense), **L** the
round-2 transducer (`all`) at `mn_ref_hz` **8.84** -- a LABELLED control, ONE hand-set parameter, the sense's default
30 untouched -- **C** `all+leg_cycle`, **D** `all+leg_cycle+haltere_sided`. Predeclared families (`predeclared.json`,
Holm m 5), decision pair **C v L**, precondition |L - C| <= 10 Hz on the commanded chordotonal rate.
`docs/audits/level_matched_control.md`.

**The `mn_ref_hz` derivation** (`out/vncd4/mn_ref_derivation.json`, CPU). `mn_ref` sets the afferent level through a
closed loop -- afferents -> VNC -> leg MNs -> afferents -- so the open-loop solve depends on which arm's MN
distribution it is done over: 4.16 / 4.99 / **8.45** Hz over the recorded round-3 A / B / C distributions. Two CPU
calibration runs give the loop slope k = 0.032 Hz/Hz (loop gain 0.53, stable), and the fixed point of the loop is
**8.8396 -> 8.84 Hz** (9.263 without the loop correction); d(level)/d(mn_ref) = -17.7 Hz per Hz, which is what the
10 Hz tolerance is. The critique's **~3.5 Hz** estimate ignored both the clip and the loop: recomputed on the round-3
C leg-MN distribution, mn_ref 3.0 / 3.5 / 4.0 gives **143-150 Hz** open loop (149-150 self-consistent) against the
88 Hz it was meant to hit -- the 150 Hz clip makes the lower end unreachable.

**The level match.** Realised window-mean commanded chordotonal rate **L 86.2 +- 0.5 Hz** against **C 87.4 +- 1.0**
(C v L +1.2 Hz, z +2.4, p 0.15, **`null`**), measured rates of the same 615 cells identical to 0.03 Hz: the
precondition is met with 1.2 Hz of a 10 Hz tolerance. Hair plate and campaniform were NOT matched, by construction
and as predeclared: L 56.8 / 49.7 vs C 47.1 / 24.9 Hz (-9.7 / -24.8, `result`).

**The family calls.** C v L is `result` on **DNa02_L +0.127 Hz** (0.431 +- 0.014 -> 0.558 +- 0.005; z +9.2, p 0.0079,
Holm 0.040), **DNa02_R +0.103** (z +16.7), **clean yaw SD +0.57 deg/s** (7.30 +- 0.08 -> 7.87 +- 0.16; z +7.1) and
**straightness +0.336** (z +15.3); DNa02 L-R +0.024 `null`. L v A is `result` on all five primaries; D v C on
DNa02_L only (-0.036, z -6.8, called; round 3 had this row `null`). Per seed (same seed, same block) the level
control reaches 0.794 0.716 0.765 0.759 0.791 of the cycle's DNa02_L rise, 0.597 0.640 0.668 0.727 0.715 of its
DNa02_R rise and 0.874 0.846 0.899 0.888 0.938 of its yaw-SD rise -- **76 / 67 / 89 %** with a scatter <= +-0.07 --
and 73 % of its DNa02-active fraction (0.0440 of 0.0605). On straightness the same fraction is **3.08**: the level
control overshoots the cycle by 3.1x, in the opposite direction (L's fly is the crooked one, 0.50 vs 0.83).

**The skeptic pass (independent Opus, verdict mostly sound; sixteen corrections applied).** Every one of the fifteen
family comparisons reproduced from the npz with the skeptic's own clean-frame rule, own exact Mann-Whitney (full
enumeration of the 252 / 70 rank splits), own z and own Holm -- **not one verdict flips** -- and so do the `mn_ref`
derivation (to four decimals), the level match, the decomposition magnitudes and both reducer fixes. What did not
survive is the **attribution**. The audit had dismissed the two unmatched channels with "L has more hair-plate and
campaniform drive than C and still less DNa02, so neither can account for the excess"; that needs the hair-plate ->
DNa02 transfer to be net positive, and the audit's own paths output says it is **negative**: the strongest three-step
afferent walk is `SNpp45 -> IN13B001 -| AN04B003 -> DNa02`, signs `+,-,+`, `gain_if_signed` -3.0e+04, and **SNpp45 is
a hair-plate afferent** (53 of the channel's 113 cells; the direct SNpp45 -> DNa02 link is only +2.0 mV/volley). The
intermediate moves exactly as that route predicts -- IN13B001 **76.4 Hz (L) vs 65.2 (C)** while AN04B003 runs
17.8 vs 23.4 -- and an additive LEVEL model with an inhibitory hair plate (w_chordotonal +0.53, w_hair_plate -0.51
Hz/Hz) reproduces BOTH the within-L left-vs-right side contrast AND the whole C-over-L AN04B003 difference with a
**zero structure term**. So L differs from C in three ways at once: per-phase modulation, the +9.7 / -24.8 Hz channel
mismatch, and the +13.6 Hz DC chordotonal L-R -- and no arm in this batch separates them. Two more rows were
re-read: **straightness is a sidedness row** (a per-fly regression inside L, straightness = 0.987 - 0.1424 x |signed
yaw|, r -0.78, predicts 0.806 at C's own drift against C's observed 0.833 -- 92 % of the gap; the reverse fit inside
C gives 82 %), and **DNa02_R is partly a per-side level deficit** (L's right side sees 79.28 Hz chordotonal against
C's 87.58, an 8.3 Hz deficit the whole-channel match hides, and AN04B003 is 94 % of that row's gap). **DNa02_L is the
row that survives every confound**: arm L's LEFT side carries more chordotonal (92.90 vs 87.24), hair-plate (61.25 vs
46.97) and campaniform (49.67 vs 24.88) drive than arm C's and still fires AN04B003_L lower (19.10 vs 23.13) and
DNa02_L lower (0.431 vs 0.558). Six further numbers did not match their own named files and are corrected in the
audit (the DNa02-active fraction 73 not 76 %; the 5-12 Hz band fraction 0.81 not 0.29, with the estimator now named;
two of the four compass drift ranges and the global bound, now -0.0087 .. +0.0094 w/s; the |DNa02 L-R| per-frame
levels; an incoherent `rate_hz` row that mixed AN04B003 with DNa02; the "whole difference is AN04B003" claim, true on
DNa02_R and a cancellation on DNa02_L). Two provenance corrections: the 16 compass run JSONs record `mn_ref_hz` null
(the flag is verifiable only from the job line and the realised pinned level), and the predeclaration's ordering
rests on local file mtimes, not on any absolute timestamp in the log.

The skeptic's closing paragraph, verbatim, is the reading this session adopts:

**What the round can now say.** With the CHORDOTONAL channel matched to 1.2 Hz (precondition met, `null`), **C v L is
`result` on DNa02_L (+0.127 Hz, z +9.2), DNa02_R (+0.103, z +16.7) and the clean yaw SD (+0.57 deg/s, z +7.1),
Holm-called at m 5** -- so **something other than the chordotonal mean separates the leg cycle from the round-2
transducer**, in the direction "more DNa02". The DNa02_L row is the robust one: arm L's left side carries *more*
chordotonal, hair-plate and campaniform drive than arm C's and still fires DNa02_L less, so no under-drive or
sidedness account of it survives. But **"the per-leg / per-phase STRUCTURE contributes beyond the LEVEL" is not yet
supported**: L differs from C in three ways at once -- per-phase modulation, a +9.7 / -24.8 Hz mismatch on the two
unmatched leg channels whose dominant route into DNa02 is sign-negative, and a +13.6 Hz DC chordotonal L-R -- and no
arm in this batch separates them. The straightness row is `result` but reads as the DC sidedness, not as structure;
the DNa02_R row is `result` but is partly a per-side level deficit; the DNa02 L-R row is `null`. `L v A` is `result`
on all five primaries and the afferent level (as delivered by the round-2 law, sidedness included) reaches 76 % / 67 %
/ 89 % of the cycle's DNa02_L / DNa02_R / yaw-SD rise over the shipped path, with tight per-seed scatter -- that part
of the framing is fair, and it is the round's solid finding. **D v C** stands as reported (DNa02_L -0.036 called, no
behavioural row). The compass answer stands at 4 v 4: no arm moves the bump, and AN04B003's signed report of the turn
(-10.5 Hz) exists under the cycle and not under the level control. Nothing is adopted, and adoption now needs three
controls, not two: the unsided level control, the modulation-only arm, **and a channel-matched level control**.

**Two reducer defects closed before the analysis** (both from the round-3 skeptic). The `sided_frames` recorder rows
now pair sample s with body frame k (lag 0) instead of k - 1: on `out/vncd3/room_C_r0` corr(AN04B003 L-R, chordotonal
cmd L-R) is **-0.339** at lag 0 against -0.294 at the old alignment, the tripod-conditioned swing -3.49 vs -3.37 Hz,
and the round-3 alignment is kept as labelled `<key>_lagm1` secondaries. And `summarise_room`'s watch loop no longer
overwrites the command-derived `DNa02_L_hz` / `DNa02_R_hz` with an all-window (airborne-inclusive) mean -- it stores
that under `DNa02_{L,R}_allwin_hz` -- so DNa02_L, DNa02_R and DNa02 L-R are on ONE mask and subtract exactly
(identity holds to < 1e-6 in all 20 runs).

**Four `lit.walk.*` ledger rows** were added to `flyverse/data/expected_responses.csv`, all `op report`, none scored:
`step_frequency_hz_at_speed` (7.1 Hz at 8 mm/s, 16.5 at 28), `stance_fraction_at_speed` (0.79 at 8, 0.51 at 28),
`swing_duration_ms` (30, bracket 30-50, **UNCERTAIN**) and `outer_leg_step_ratio_in_turn` -- the last with an **empty
value** and a note that DeAngelis 2019 Fig 6C gives the direction only and no ratio was read off the figure. All four
carry the caveat that the numbers are the *fit's* values, not tabulated ones (DeAngelis, Zavatone-Veth & Clark 2019,
eLife 8:e46409; Mendes, Bartos, Akay, Marka & Mann 2013, eLife 2:e00231). Nothing was invented to fill the blank.

**Next, and nothing adopted.** Adoption of the leg cycle now needs **three** controls, not two: (a) an **unsided**
level control (the round-2 law on the side-MEAN leg-MN rate at mn_ref 8.84 -- removes the +13.6 Hz DC L-R and the
+9.2 Hz hair-plate L-R at the same means), (b) a **modulation-only** cycle arm (per-leg amplitude held at 1 --
separates the temporal modulation from the turn kinematics but not from the channel mismatch), and (c) a
**channel-matched** level control (`hair_plate_max_hz` / `campaniform_load_hz` set so the realised hair-plate and
campaniform means match C's 47.1 / 24.9 Hz at the same chordotonal 86-88 Hz) -- the only one of the three that
addresses the hair-plate alternative. Beside them, a single-cell CPU check of **AN04B003** under (i) steady vs
8 Hz-modulated chordotonal input at the same mean with IN13B001's rate clamped and (ii) the hair-plate level varied
alone at a fixed chordotonal level; the two together settle whether the C-over-L AN04B003 difference is modulation or
hair-plate disinhibition. **Nothing is adopted and no default changed**: `senses.py`, `body.py`, `brain.py` are
untouched, the bit-identity golden passes, and the leg cycle and the level control both remain opt-in labelled
mechanisms.

### Session 12 addendum: the three level controls (2026-09-15)

**What ran.** ONE house submission, `vncd5-2ffbc3`: **30 job(s), 0 failed (22.6 min)** on B200 -- the
`probe_vnc_drive` room at `--family level2`, **six arms x five brain seeds** (16 flies x 60 s, 5-60 s window, blocks
`fam_r0..fam_r4`), no compass job (rounds 2-4 settled that question). Arms: **A** shipped (no sense); **L** round
4's level-matched control (`all`, `mn_ref_hz` 8.84), re-run so every pair is within-batch; **U** the UNSIDED control
(`all+unsided`: every leg cell reads the side-MEAN leg-MN rate, so L's +13.564 Hz DC chordotonal L-R and +9.204 Hz
hair-plate L-R become +0.000 at the same channel means); **K** the CHANNEL-MATCHED control (`all` at 8.84 with
`hair_plate_max_hz` **86.71** and `campaniform_load_hz` **25.05**, derived on CPU as a fixed point of the same
afferent -> leg-MN loop round 4 solved for `mn_ref_hz` and verified by a fourth CPU run before submission, realising
hair plate 44.34 / campaniform 24.78 against C's 46.36 / 24.91); **M** the MODULATION-ONLY control
(`all+leg_cycle+leg_cycle_flat`, per-leg amplitude held at 1.000000, |amp L-R| 0.000000 against C's 0.948 / 0.0170);
**C** the cycle arm. Two new mechanisms, both **opt-in and default OFF**, both LABELLED CONTROL constructions and
neither a candidate for a default: the `unsided` token in `senses.Proprioception.FLAGS` (four lines, one of them the
law; refused with `leg_cycle` and without an MN-rate leg channel) and `body.LegCycle.flat_amplitude` (the `amp` line
only; phase, stance, frequency, stance fraction and per-side loads bit-identical to the default cycle's). The
shipped CPU path is bit-identical with both absent -- `pytest tests/test_proprioception.py tests/test_body_cycle.py
tests/test_bit_identity.py` = **33 passed, 5 subtests**, the golden unchanged, reproduced independently on a second
machine. Predeclaration stamped **04:54:59Z** against the earliest run's own `started_utc` **04:56:11Z**, an
absolute timestamp inside the run JSONs, so round 4's "the ordering rests on file mtimes" caveat is closed.
`docs/audits/level_controls.md`.

**The sanctioned reading is the independent skeptic pass's closing paragraph, verbatim** (verdict **mostly sound**;
its ten corrections were applied to the audit):

**For NOTES -- what rounds 4 and 4b jointly say about the leg cycle.** Round 4 found the leg-cycle arm above a
level-matched steady control on DNa02 and the ascending relay but could not say why; round 4b ran the three controls
that split the difference and, within one 30-job submission whose predeclaration is stamped 72 s before the first run,
separated them: the round-2 law's DC sidedness owns the level control's drift and straightness and **none** of its
DNa02 rate (U v L `null` on DNa02_L, DNa02_R and the clean yaw SD while straightness goes +0.231 and the drift halves);
the unmatched hair plate is a real, sign-negative and **small** route (-0.07 Hz of AN04B003 per Hz in the room, -0.149
on an isolated cell, accounting for +0.72 Hz -- 14 % -- of the +5.22 Hz pooled C-over-L relay difference, with the
campaniform contributing nothing measurable); and the remainder, **~+4.5 Hz, is the per-phase modulation** -- a level
model fitted on three steady arms predicts a held-out steady arm to 0.33 Hz and under-predicts both cycle arms by
+3.9 to +5.7 Hz, a residual that survives every specification I could build and that no admissible hair-plate slope can
remove, with the single cell naming the mechanism directly (+1.63 Hz at a matched per-cell mean with IN13B001 clamped,
z +7.2, reproduced bit-for-bit). Against that, four things are **not** established: no row of either round is
Holm-called in round 4b, because the predeclared family size m = 7 is arithmetically unsatisfiable at 5 runs per arm
(0.0079 x 7 = 0.0556) -- a failure to apply a rule already in `docs/INTERP.md` 10.2; the DNa02 story rests on pairs and
on a level model the audit itself disqualifies, so "the structure raises DNa02" remains a `compare`-level pattern of
+0.09 to +0.18 Hz, not a level-controlled measurement; the amplitude-only control M bought 6.0 Hz of chordotonal and
still carries a structure residual 0.66-0.75 Hz **larger** than C's (z +3.5 to +3.8), so "the amplitude law adds no
drive" is not supported and F4 stays open; and the clean-yaw-SD row of C v L flipped from `result` to `null` between
batches only because the control arm's between-run scatter doubled -- the difference itself replicated (+0.57 then
+0.49, complete rank separation in both, pooled p 1.1e-05), so neither batch is an outlier. Nothing is adopted; the
next round's arms are named in section 10 and the first of them is a family whose Holm denominator can actually be
satisfied.

**The Holm defect, and the standing rule it violated.** The predeclared family size was **m = 7** at 5 runs per arm,
where the exact Mann-Whitney floor is `p_floor(5,5) = 2/C(10,5) = 0.0079365`: the smallest attainable adjusted p is
`0.0079 x 7 = 0.0556 > 0.05`, so the declared decision rule was **unsatisfiable before a single job ran** and no row
of any family is CALLED. The audit reports this rather than repairing it after the fact, and every table quotes the
`compare` verdict, the z, the per-run scatter and the Holm-adjusted p side by side. What the audit got wrong is
calling the fix "the rule for the next round": **`docs/INTERP.md` 10.2 already said it**, from object round 2 --
*"A family member whose statistic is constant by construction is not a test ... still inflates the Holm denominator
... Declare such quantities as reported magnitudes outside the family, and size the family (and therefore the arm
count) from the members that can move"* -- and that rule is recorded there as the reason that round raised its arms
from 5 to 6 runs. The predeclaration step failed to apply a standing house rule. The arithmetic for next time:
**m <= 6 at 5 v 5** (0.0079365 x 6 = 0.047619) and **m <= 23 at 6 v 6** (0.0021645 x 23 = 0.04978).

**The integrity defect, and the rule that follows.** The audit's section-5 sentence "Per-seed values behind the arms
(r0..r4)" printed DNa02_L per-run lists for **U, K, M and C** -- and a clean-yaw-SD list for U -- whose values occur
**nowhere in `out/vncd5`**. They had been constructed to carry the published means to four decimals rather than
transcribed from the file: A and L reproduced, the other four did not. The skeptic caught it (R1) while reproducing
**all 49 rows** of the decision table exactly from the raw recordings, so the table, the means, the SDs, the diffs,
the z, the p and every verdict were and are correct -- only the prose scatter was invented. The four lists were
replaced from `analysis/room_table.csv`'s `*_runs` columns (verified against that file before writing: U
[0.3917 0.3950 0.3720 0.3873 0.3892], K [0.3519 0.3632 0.3529 0.3261 0.3548], M [0.6187 0.6327 0.6227 0.6141
0.6165], C [0.5155 0.5343 0.5290 0.5144 0.5470]; U's clean yaw SD [6.8111 6.6278 6.9263 6.7598 6.6147]), and the
audit now says plainly in section 5 that the previous version of that sentence printed values that were in no file.
**The rule this adds** (`docs/INTERP.md` 10.4, new item): any per-run or per-seed scatter quoted in prose is
**emitted by the analysis script into a named file and pasted from that file** -- the file is named in the sentence
that quotes it -- never retyped by hand and never reconstructed from a mean; a quantity no file carries is not
quoted. A second, related correction in the same pass: `pairwise.csv` carries the run mean, SD and n only (51
columns, no `*_runs`), so `room_table.csv` and `pairs_console.txt` are where the per-run values live.

**What K and M actually were (the two construction caveats).** Both controls bought a level change with the thing
they were built to hold. **K** lost **8.7 Hz of chordotonal** because matching the hair plate and campaniform at a
fixed `mn_ref` 8.84 lowers the leg-MN rate that feeds the chordotonal channel -- the predeclared side effect P1b,
and exactly the round-2 law (`d(chord)/d(legMN)` = 15.84 Hz/Hz predicts -11.09 / -6.13 against -11.09 / -6.17
observed). So K v L is `result` **NEGATIVE** on AN04B003 (-0.94 / -0.61), the opposite of the predicted
disinhibition, the pair settles nothing on its own, and C v K / M v K are **upper bounds** by the predeclared rule
(the DNa02 one loose by about a sixth: K's deficit is worth only +0.028 Hz of the +0.178 DNa02_L difference).
**M** *gained* **6.0 Hz of chordotonal**, because the cycle law's realised amplitude is **0.948**, not 1.000, so the
flat arm sits above the treatment; M v K's realised gap is **+14.74 Hz**, outside the declared 10 Hz tolerance, and
F6 is VOID as a level contrast. Two further limits on the post-hoc level model that replaced the two-control rule:
its published slope SEs are **pseudo-replicated** (six distinct (arm, side) level points, 30 rows that are five runs
of each; on the six arm-side means `b +0.1848 +- 0.0355`, `c -0.0648 +- 0.0285`, so the hair-plate slope is 2.3
sigma from zero, not 6.8), and the three steady arms lie close to one line in the (chordotonal, hair-plate) plane
(corr 0.925), with C 9.1 Hz and M 11.8 Hz of hair plate off it -- the cycle predictions are **joint extrapolations**
even though both marginal ranges contain them. Neither limit removes the residual: a **held-out steady arm** is
predicted to 0.33 Hz, the residual survives twelve specifications (+2.50 .. +5.02 on C's left side) and **no
hair-plate slope compatible with the steady arms can eat it** (`c ~ -0.55` would be needed, degrading the steady-arm
RMSE 9 x). F5 (C v U) additionally leaves a **-24.7 Hz campaniform mismatch** the model cannot correct for, and the
only evidence it does not matter is the single cell's campaniform null on a 752-cell subset.

**Next, and nothing adopted.** No default moved, no mechanism was adopted, and `unsided`, `leg_cycle_flat` and the
leg cycle all remain opt-in with their defaults untouched. The arms `docs/audits/level_controls.md` section 10 names
first are **M at the cycle's own realised amplitude 0.948** (so that M and C sit at one chordotonal level and F4 --
does the amplitude law add drive? -- becomes answerable) and **a predeclared family whose Holm denominator can be
satisfied**: m <= 6 at 5 runs per arm, or 6 runs per arm for a family of up to 23. Behind them, unchanged from the
audit's list: a control that matches all three channels at once (`mn_ref_hz`, `hair_plate_max_hz` and
`campaniform_load_hz` as one three-parameter fixed point, which would turn C v K from an upper bound into a
measurement), the suite under `all+leg_cycle`, ledger values for `lit.walk.outer_leg_step_ratio_in_turn` and
`half_width_m`, a free-walking compass room under the cycle, and the FeCO walking-mean rate as a ledger number --
to which this round adds the cycle's realised per-leg amplitude (0.948) as a second unmeasured quantity of the same
kind.

## Session 12, compass round 5: the ring mechanism and the GLNO sign (2026-09-15)

Two threads, one house submission each, one independent skeptic pass each; **nothing adopted and no default moved.**

**5A -- is there a type-level ring mechanism, or none? (`docs/audits/compass_ring_mechanism.md`, batch `cx5-5cde5e`.)** ONE
`cluster_run.py` call, **8 job(s), 0 failed** (4 seeds x {GLNO silent, GLNO = glutamate}, six arms per job), **48 runs**
(12 arms x 4 seeds) on the house B200s with 0 provenance problems; a new CPU structure pass
(`scripts/cx_ring_structure.py`, `tests/test_cx_ring_structure.py` 5 passed) ranked the candidates first. **The answer is
no.** At the shipped gains (gE 1 / gD 1) the shipped path, the GLNO relabel, the receptor `sign+gain` tier and their pair
all score `bump_survival_s` 0.00 in 4/4 seeds, and **no arm meets the predeclared "working compass" rule** (survival >= 5 s
AND width 2.5-5 wedges AND rate 5-60 Hz in >= 3 of 4 seeds): **every surviving bump fails on rate** -- F (same-type damping
off, a labelled global INSTRUMENT) 151-156 Hz in 4/4, CFG 156-158, and the labelled references R 201-203 and RG 181-184, all
about 3x above the 5-60 Hz ledger row. The binding constraint is a **DC balance on the relays**, not a missing ring mode:
the EPG drives 2 ExR6 + 4 ER6 (+ 11 ER4m onto EPG) harder than it drives PEN (EPG -> ExR6 +11.4 mV per pair against
EPG -> PEN +5.05), so u_PEN sits at -11.1 mV at background and -15.8 mV *during* the pulse, and the measured PEN population
mean is 0.29-0.66 Hz during the pulse (0.02-0.06 after). The caveat the skeptic forced into the audit: **the per-type
ExR6 -14.8 mV at 183 Hz / ER6 -9.8 at 116 Hz figures are rate-model outputs, not measured**, and they overstate the DC term
by roughly 2x against this batch's own 308-cell ring population (244 Hz after release, 630 at the end of the pulse, peak
800); the conclusion rests on the measured PEN rate. The ER/ExR feedback carries no k = 1 content (-206, 2.5 % of PEN's
+8,327), so it is untuned in the bump mode rather than a flattened cosine.

**The structure pass's two defects, owned by that thread.** It missed the F-family bump, and not because of a
fluctuation regime: (i) the EPG-only reduction computes the one-step EPG -> EPG term and then drops it (its undamped k = 1
ring-Fourier coefficient is +59.76 mV, gamma_crit 3.35 Hz/mV, predicting saturation at 145 Hz against the observed
151-156), and (ii) `rate_fixed_point` enters the forced background as a **rate** rather than as a **current**, so the driven
EPG sits at u -22 to -68 mV while "firing" at 10-50 Hz and gamma_EPG = 0 by construction. The section 1.3 ranking is
therefore computed at a zero-gain state and is not to be trusted until both are fixed; the defects are named in the module
docstring of `scripts/cx_ring_structure.py`. **The next arm**, which the audit's own decomposition names and this batch did
not run: an `edges`-kind hold of **ExR6 / ER6 / ER4m at 0 onto PEN and EPG** -- with the ER/ExR term removed, u_PEN during
the pulse is +9.7 - 0.6 = **+9.1 mV**, above the 7 mV gap (f ~ 30 Hz) -- the one arm that turns "the DC inhibition keeps the
relays below threshold" from a decomposition into a tested attribution. Its data half is a **relabel with sources, not a
gain: ExR6's transmitter and receptor are unknown.** ExR6 is 2 cells, MaleCNS `nt` glutamate, sign -1, with no
receptor-table row (tier fallback = NT_SIGN), and its -1 onto EPG rides on the E-PG row's GluClalpha (Davis 2020 PB_2, tier
alias); no published transmitter or function for ExR6 could be verified (Hulse et al. 2021 defines ExR1-ExR8
morphologically and by connectivity, ExR2 is the dopaminergic PPM3 class, and the classical ER ring neurons are the
GABAergic E-PG inhibitors of Omoto 2017 / Fisher 2019 / Kim 2019), so it is to be stated as **unknown** rather than as "the
ring neurons inhibit EPG".

The independent 5A pass (Opus, verdict **mostly sound**, 14 corrections, all applied) closes with this, quoted verbatim:

**What the compass now needs.** The shipped-gain question is closed as a *null*, not as a *result*: every gE 1 / gD 1
arm is a structural zero-SD comparison against a reference that is also zero, read as magnitudes, and the two arms that
carry a bump are a labelled global INSTRUMENT (`same_type_gain` 1) and a labelled reference (gE 2 / gD 15) -- neither is
a mechanism the data imply, and both fail `compass.EPG.bump_rate_hz` 3x, so `compass.EPG.bump_survival_s` stays FAIL and
the rate and width rows stay NOT_APPLICABLE. Nothing here licenses an adoption. What it licenses is one more
**measurement** and one more **counterfactual arm**, in that order: first fix the structure tool (forced drive as a
current, the one-step term kept, per-cell gamma at the realised fixed point) and re-derive the ranking, because the
present ranking is computed at a state with zero gain everywhere and its miss on the F family shows it; then run the
single `edges`-kind arm the audit's own decomposition names -- ExR6 / ER6 / ER4m held at 0 onto PEN and EPG -- which
the fixed point predicts puts the driven PEN at +9.1 mV and ~30 Hz during the pulse, and which is the only thing that
turns "the DC inhibition keeps the relays silent" from a decomposition into a tested attribution. Alongside it, the
data question is a **relabel with sources**, not a gain: ExR6's transmitter confidence and whether it has any fast
receptor on E-PG at all (it has no receptor-table row; the -1 rides on the E-PG GluClalpha profile), reported as
`unknown` where it is unknown. The GLNO relabel is a `null` at these gains and should be handed to `glno_relabel.md`
saying exactly that -- a silent neuron's sign is untested, not confirmed harmless. And the F-family bump should not be
brought back as a candidate until it is run in the room with a world: at 151-158 Hz it is a KNOWN GAP swap, and the
rest-of-brain 0.04 Hz in this world-less protocol is no evidence at all about the runaway cliques the x0.1 was adopted
against.

**5B -- is GLNO = glutamate adoptable? (`docs/audits/glno_relabel.md`, batch `cx5b-d08be3`.)** ONE submission, **22 job(s),
0 failed (18.5 min)** on a B200, every one of the 22 JSONs carrying `problems []` and the md5 of the cache it was asked for:
the 29-check suite at 3 draws x {shipped, glutamate}, and the efferent compass (family `level`, arm C = `all+leg_cycle`) at
4 seeds x {shipped, glutamate} x {gE 2 / gD 15, shipped gains}. Structure, entry by entry on the CPU: the candidate cache
differs from the shipped one in **exactly 4 cells and 213 W entries** -- all with a GLNO presynaptic cell, all 0 ->
negative, |value| equal to the shipped sign-0 count on every one; 17,698 raw synapses; the 84 GLNO -> PEN edges at -16.50 mV
per GLNO spike (-33.0 per PEN per volley), all 84 above the connection cap -- plus a <= 1.2 % fan-in-scale shift on 20 minor
targets. **Under the shipped receptor model that W is bit-identical to the round-3/4 `gaba` cache** (md5 `7a10d93b`;
`fast_sign` identical between the two labels on all 25,578,600 entries), which is why the decision is about a sign and not
about a transmitter name. Suite: **27 PASS / 0 FAIL / 2 KNOWN GAP in all six draws, no check changing status**, with
`changed_pooled`, `changed_matched` and `unstable_shipped` all empty over 174 check rows. Compass: **every predeclared
member `null`** at 4 v 4 (all 15 members of F1-F5: the GLNO / PEN_a / PEN_b flips and the bump drift), and **the signed
self-turn report still stops at AN04B003** (`result`, -11.2 / -11.8 / -11.5 / -11.5 Hz, p 0.0286 = the exact-U floor, in
4/4 arms) and is **`null` at PS196_b**, GLNO, PEN and EPG at both gain settings. What the relabel moves is rates, not
reports: the ring -5..-15 %, FB4Y -31 %, FB1C -68 %, ExR8 silenced outright, and GLNO's standing L-R +27 -> +3..+4 Hz.

The independent 5B pass (Opus, verdict **mostly sound**, corrections C1-C14, all applied) closes with this, quoted verbatim:

**What rounds 5A and 5B jointly say about the compass and GLNO.** Taken together the two threads turn the GLNO sign
from an open question into a `null` with a known mechanism, and leave the compass's real failure where 5A found it.
5A establishes that at the shipped gains no type-level change reachable from the data -- the GLNO relabel included --
makes the ring hold a bump, because PEN sits 11-16 mV below threshold under ExR6 / ER6 and the relabel is therefore
inert (`bump_survival_s` 0.00 in 4/4 seeds, a structural zero-SD null); the only configuration that carries a bump at
those gains is the removal of a global hand rule, which is a labelled instrument, not a mechanism, and which the
correct GLNO sign then destroys. 5B runs the same candidate on the one instrument where a signed GLNO can act (gE 2 /
gD 15, body attached, leg cycle on, DNa02 driven) and finds the effect is entirely on rates -- ring -5 to -17 %, FB4Y
-31 %, FB1C -68 %, ExR8 silenced, GLNO's standing L-R compressed 86-88 % -- with every predeclared flip and the bump
drift `null` at 4 v 4, and with the signed self-turn report still terminating at AN04B003 (`result`, -11 Hz, p at the
0.0286 floor, 4/4 arms) and `null` at PS196_b, GLNO, PEN and EPG. The joint reading is therefore: GLNO's transmitter is
a **rate** parameter of the ring, not a **signal** parameter, and it is a parameter the fast model cannot even name,
since glutamate and gaba produce a bit-identical W and, under the shipped receptor model, an identical model
everywhere. The compass gap (`compass.wedge_cells_persisting` 0, KNOWN GAP in all six draws of both arms) is untouched
by it in both threads, and the self-turn report's break is upstream, at a PS196_b whose input to GLNO is symmetric in
both turn directions -- which is a body-model question, not a transmitter one. On the decision itself the two threads
now agree with one correction: the adoption is **adoptable by the suite half of the round-2 rule and by nothing else**,
the room rate-half is unrun, the source standard of the three existing `TYPE_NT_OVERRIDE` entries is unmet, and the
"hemibrain name" 5A counted as one of three sources is not a source at all -- leaving two low-confidence EM classifiers
calling glutamate and a third calling GABA, all three agreeing only that GLNO is inhibitory.

**The owner decision: GLNO -> glutamate is NOT adopted.** The suite half of the round-2 rule is met and nothing else is.
The **rate half was not run** (the room take-off protocol at >= 6 runs per arm, `guard_suites_r3.md` 4), and the **source
standard of the three existing `TYPE_NT_OVERRIDE` entries -- each names a transcriptome or an EASI-FISH source -- is not
met**: what exists for GLNO is two low-confidence EM classifiers calling glutamate (MaleCNS v1.0 T-bars 51 % glutamate /
37 % ACh over 783 T-bars, every consensus column `unclear` at conf 0.48; BANC v888 `Predicted NT type` GLUT on 4/4 at conf
0.47-0.50, unverified, taken by the backend without a threshold) and a third calling GABA (FlyWire v783 `top_nt` gaba 3/4
at conf 0.30-0.33, glutamate 1/4 at 0.30), all three agreeing only that GLNO is **inhibitory**. **The "hemibrain name"
cited to the owner as the third source is withdrawn.** No accessible text defines the "G" or attaches a transmitter
prediction to GLNO: Scheffer et al. 2020 says only that "the nodulus neurons are now 'LNO' and 'GLNO' instead of 'LN' and
'GLN'", Hulse et al. 2021's accessible text lists GLNO with no transmitter statement, and Wolff & Rubin 2018 is the source
of the light-level `LAL-NO1` name and its abbreviation table. The expansion "GLutamatergic LAL-NOduli neuron" was asserted
as a fact with no citation in `out/cx5/structure/evidence_glno.json` and repeated in `compass_ring_mechanism.md` 1.3(a);
both threads and both skeptic passes failed to verify it, and it is not to be cited again without a page reference. Because
the candidate's W is bit-identical to the gaba cache under the shipped model, **the decision actually on the table is "sign
GLNO -1", not "GLNO is glutamatergic"** -- the transmitter name is a tie-break the data do not make. **Nothing was adopted
and no default moved**: `cache/`, `connectome.TYPE_NT_OVERRIDE`, the receptor table and every gain are untouched, the
candidate stays in the scratch cache `out/cache_glno_glu/`, and `same_type_gain` keeps its x0.1. What the round leaves on
the list is in `TODO.md` B: the ExR6 / ER6 / ER4m hold arm (after the structure tool's two defects are fixed), ExR6's
transmitter as a relabel-with-sources question, the room rate-half if adoption is ever wanted, the inverted
`struct.GLNO_PEN.sign` ledger row, and the body-model question at PS196_b, where the self-turn report actually breaks.

### Session 12 addendum: round 4c, the modulation-only arm at the realised amplitude (2026-09-15)

**What ran.** ONE house submission, `vncd6-2ad71d`: **24 job(s), 0 failed (21.5 min)** on B200 -- the
`probe_vnc_drive` room at `--family level3`, **four arms x six brain seeds** (16 flies x 55 s window, blocks
`fam_r0..fam_r5`), no compass job. Arms: **A** shipped (no sense); **L** round 4's level-matched control (`all`,
`mn_ref_hz` 8.84), re-run inside this batch as the level reference; **M2** the MODULATION-ONLY control at the cycle's
OWN realised amplitude (`all+leg_cycle+leg_cycle_flat` with `body.LegCycle(flat_amplitude=True,
flat_amplitude_value=0.948)`: the phase / stance-swing modulation kept, the amplitude law removed, the afferent LEVEL
the cycle's); **C** the cycle arm. Six runs per arm so that the predeclared Holm family of m = 7 is satisfiable at all
(the 6 v 6 exact-U floor is `2/C(12, 6)` = 0.0021645 and 0.0021645 x 7 = **0.0152 <= 0.05**), which round 4b's m = 7
at 5 v 5 was not. One new mechanism, **opt-in and off the default path**: `body.LegCycle.flat_amplitude_value`
(default **1.0**, read only under `flat_amplitude`; one dataclass field and one line inside the flat branch, with the
amplitude law and every timing / tripod / load line untouched) -- `pytest tests/test_body_cycle.py
tests/test_proprioception.py tests/test_bit_identity.py` = **36 passed, 5 subtests**, including three new
`rtol=0, atol=0` `BatchSim` tests pinning that the default cycle with the value unread, and round 4b's flat cycle with
`=1.0`, are bit-identical, and the shipped golden unchanged. The value 0.948 is a summary statistic of round 4b's
recordings, derived on CPU before submission: the per-leg amplitude of `out/vncd5`'s cycle arm on walking frames, per
run [0.947665 0.947714 0.947744 0.947786 0.947730], run mean **0.947728 +- 0.000044**, rounded to three decimals (the
law at the recorded mean walking speed of 9.794 mm/s at yaw 0 gives 0.947396, so the recorded mean is the law and not
an artefact of the mask). Nothing about behaviour enters it. Predeclaration stamped **07:47:38Z** (archived
byte-identical) against the earliest run's own `started_utc` **07:48:15Z**; submission receipt 07:47:47Z.
`docs/audits/level_controls_r2.md`.

**The headline.** **F4 is CLOSED.** With M2 level-matched to C on all three leg channels (chordotonal -0.76, hair
plate -0.41, campaniform +0.03 Hz) and its per-leg amplitude exactly 0.948000 with |amp L-R| 0.000000, **M2 v C is
`null` on all seven primaries** (AN04B003_L -0.20, AN04B003_R -0.27, DNa02_L -0.037, DNa02_R +0.000, clean yaw SD
+0.07, straightness +0.031, DNa02 L-R -0.037) -- and at 6 v 6 a row is called only at |z| >= 3, i.e. at three times
C's own between-run SD (1.26 / 1.44 Hz of AN04B003, 0.055 / 0.041 of DNa02, 0.60 deg/s of clean yaw SD, 0.114 of
straightness, 0.071 of DNa02 L-R), every one of which round 4b's M-over-C excess (+1.62 / +1.65 Hz of AN04B003,
+0.093 of DNa02_L) clears. So the amplitude / turn law adds **no drive** at the relay or at DNa02, round 4b's excess
was its +6.0 Hz of level, and the +0.66 / +0.75 Hz structure-residual excess its skeptic pass found is -0.12 / -0.13
Hz and `null` at the right level. **The structure term is the modulation**: M2 v L is CALLED (Holm 0.015) on
AN04B003 +3.92 / +6.85 Hz, DNa02_L +0.111 and straightness +0.339 with an arm that cannot carry a turn term, and the
level-corrected pooled structure term is **+4.60 +- 0.31 Hz without the amplitude law against the cycle's
+4.72 +- 0.32** -- the same number. **The turn term owns the sided DNa02 signal, and owns it as a sidedness rather
than a rate**: `E[DNa02 L-R | chord L-R > 0] - E[. | < 0]` is -0.338 under C against -0.091 under M2 (z +7.9,
`result`), and the two arms do NOT see the same afferent waveform -- the alternation SIZE is the same (12.36 vs 12.16
Hz) and so is the fast tripod term (SD 14.02 vs 13.99), but the SLOW, yaw-locked component is 2.6x larger under C
(mean |slow chord L-R| 1.65 vs 0.63 Hz, corr with yaw -0.78 vs +0.01) and DNa02's sidedness follows that component
alone while the relay follows the fast one and is identical in both arms. No window rate, yaw SD, straightness or
behavioural primary moves with it.

**The sanctioned reading is the independent skeptic pass's closing paragraph, verbatim** (verdict **mostly sound**;
its thirteen corrections were applied to the audit):

**What rounds 4, 4b and 4c jointly say about the leg cycle.** Three batches of the same protocol now agree on an
attribution and on its size: against a level-matched round-2 transducer at the same chordotonal level, the leg cycle
raises the ascending relay AN04B003 by **+3.7 to +7.1 Hz per side** (`result`, CALLED in the one batch whose Holm
family was satisfiable) and DNa02_L by **+0.09 to +0.15 Hz** (`result` in all three) -- a replicated attribution, not a
single-batch finding. Round 4c splits that term cleanly in two. The rate part is the **per-phase modulation**: an arm
that keeps the phase/stance structure and holds the amplitude at the cycle's own realised 0.948 reproduces the cycle
arm on every rate row (fractions 0.94-1.00 against the level control's 0.69-0.88) and is `null` against it on all seven
primaries, with a level-corrected structure term of +4.60 +- 0.31 Hz against the cycle's +4.72 +- 0.32 -- so round 4b's
M-over-C excess was its +6.0 Hz of level, and the amplitude / turn law adds **no drive** at the relay or at DNa02. The
sided part is the **amplitude / turn law**, and it is a sidedness, not a rate: the law writes a slow, yaw-locked L-R
asymmetry into the afferents (mean |slow chord L-R| 1.65 Hz against the flat arm's 0.63, corr with yaw -0.78 against
+0.01) which DNa02 integrates into a -0.34 Hz tripod-conditioned L-R swing (against -0.09), while the relay, which
follows the fast tripod term, sees nothing of it -- and no window rate, yaw SD, straightness or behavioural primary
moves. Two rows stay unsettled across all three batches: DNa02_R (+0.09-0.11 Hz) and the clean yaw SD (+0.49-0.59
deg/s) over the level control reproduce as differences every time and as verdicts only sometimes, because the
denominator is the control arm's own between-run scatter. Nothing is adopted; `LegCycle.flat_amplitude_value` is an
opt-in labelled-control parameter whose default path is bit-identical to the shipped one, and the structure term still
carries a -9.5 Hz hair-plate and -24.7 Hz campaniform mismatch that is corrected at slopes extrapolated ~9.5 Hz off
their calibration manifold, and an unmeasured `half_width_m` that scales the one row the turn term owns.

**Integrity.** Round 4b's defect does not recur. **All 39 per-seed lists** quoted in the audit's prose and tables (28
in section 5, 4 in section 7, 7 in section 8.3) were machine-extracted and checked by the skeptic against their named
sources -- 33 against `analysis/per_seed.csv` column `value` and 6 against
`analysis/level_model_r2_residual_runs.csv` column `residual` -- with **zero mismatches**, so `docs/INTERP.md` 10.4
item 28 (any per-run or per-seed scatter quoted in prose is emitted by the analysis script into a named file and
pasted from that file, the file and column named beside the list) was followed. One predeclared reducer changed after
the first `pairs` run -- a one-line dedupe of `cmd_pairs`' key list in `scripts/probe_vnc_drive.py`, which had listed
`DNa02_L_hz` / `DNa02_R_hz` as both a room key and a watch key and written 48 duplicate rows into `per_seed.csv` --
and it was handled the way **`docs/INTERP.md` 10.4 item 11** requires: the whole `pairs` step was re-run on the same
fetch and every derived artefact re-emitted, `decision_table.csv` is byte-identical before and after (both files
kept), and `pairwise.csv` equals the before-file with its two duplicate rows dropped. The shipped analysis artefacts
therefore carry the POST-change reducer hash (`5a288184...`, recorded in `post_analysis_sha.json`) while the 24 room
jobs ran the stamped one (`f636d330...`), which every run JSON confirms independently through its 52-file
`source_fingerprint`.

**Next, and nothing adopted.** No default moved and no mechanism was adopted: `flat_amplitude_value` is an opt-in
labelled-control parameter whose default path is bit-identical to the shipped one, `leg_cycle`, `leg_cycle_flat` and
`unsided` all stay opt-in and OFF, and M2 is a LABELLED CONTROL. What the round leaves on the list (`TODO.md` B):
**`mn_ref_hz`, `hair_plate_max_hz` and `campaniform_load_hz` derived TOGETHER as one three-parameter fixed point**, so
that the +4.6 / +4.7 Hz structure term over L stops carrying a -9.5 Hz hair-plate and -24.7 Hz campaniform mismatch
corrected at slopes extrapolated about 9.5 Hz off their calibration manifold -- a within-batch calibration is not
available here, L being the only steady-input arm in the batch (two arm-side means against the model's four
parameters), so the cross-batch application is the only option rather than a preference; **more runs of the REFERENCE
arm** for DNa02_R and the clean yaw SD, whose differences replicate across all three batches while the z does not,
because L's own between-run scatter is the denominator every time; ledger values for `half_width_m` and
`lit.walk.outer_leg_step_ratio_in_turn`, which scale the one row the turn term owns; the FeCO walking-mean rate as a
ledger number (86-87 Hz and the cycle's realised amplitude 0.948 are both consequences of the laws, not measurements);
the suite under `all+leg_cycle`; and a free-walking compass room under the cycle.

## Session 12, compass round 6: the DC-balance test (2026-09-15)

One thread, one house submission, one independent skeptic pass; **nothing adopted and no default moved.**

**What ran (`docs/audits/compass_dc_balance.md`, batch `cx6-995cd5`).** ONE `cluster_run.py` call, **10 job(s), 0
failed (5.0 min)** on the house B200s, **40 runs** (8 arms x 5 seeds) at the **shipped gains** (gE 1 / gD 1), 40/40
consoles `device cuda`, 0 provenance problems, predeclaration stamped **08:09:25Z** and never amended. The arm is an
`edges`-kind LABELLED COUNTERFACTUAL: **ExR6 + ER6 + ER4m held at 0 onto PEN and EPG** -- 17 presynaptic cells (2 + 4 +
11) onto 88 postsynaptic (46 EPG + 42 PEN), **1,149 weight entries, 37,256 synapses**, with the three single-type arms
partitioning the entries exactly (174 + 287 + 688) and EPGt excluded by the post regex. It is installed by ONE new
flag, `scripts/cx_wedge.py --hold-edges PRE_REGEX:POST_REGEX`, **default `None`**, which appends `(pre, post, 0.0)` to
`LIFParams.type_path_gain`: with the flag absent the gain list is the previous one entry for entry and the shipped path
is **bit-identical** (the CPU default-path smoke against the pre-6A run, and S / F / R reproducing cx5's seeds 0-3 at
max |diff| 0.0). `scripts/cx_ring_structure.py` was fixed first -- the forced drive entered as a **current**, the
one-step EPG -> EPG term **kept**, per-cell gains at the realised fixed point -- with **`--legacy` reproducing 5A
exactly**, and was validated against the cx5 batch it had missed before the predeclaration was written.

**The call, in the predeclared words: NECESSARY BUT NOT SUFFICIENT.** The hold lifts the driven PEN population from S's
**0.29-0.66 Hz to 40.2-48.3 Hz during the pulse** (+43.53 Hz, z 294.0, p 0.0079, Holm 0.0317, `result`) and to
49.3-54.0 Hz after release -- the 5A attribution, tested -- and **buys no bump**: `bump_survival_s` **0.00** and
`frac_confined_post` 0.000 in **5 of 5 seeds**, a structural zero-vs-zero null against a reference that is also 0.00.
The ring does not settle into a confined bump, it **saturates**: at 5 s the EPG hump is centred at wedge **5.71-6.11**
rather than the driven 1.5, with 17-19 of the 35 off-block cells above 22 Hz. And **the hold also removes the resting
state** -- on the 10 Hz background alone, before any pulse, H3 sits at EPG 19.7-62.9 Hz, PEN 8.8-42.7, Delta7
28.3-103.5 and GLNO 32.5-131.4, against S's 8.7-10.0 / 0.00-0.02 / 8.4-11.8 / 0.00. Per type, **ExR6 carries most of
the PEN DC** (held alone: PEN 8.9-14.9 Hz, +10.37, z 70.0, `result`), **ER6 some** (2.4-3.1 Hz, +2.25, z 15.2,
`result`) and **ER4m none on PEN** (0.29-0.70 Hz, +0.03, z 0.22, p 0.69, **`null`**) -- ER4m is a -125.3 mV-per-volley
term onto EPG and only -2.0 onto PEN. The closest configuration to a bump at the shipped gains without a global
instrument is **H3G**, the hold with the GLNO relabel: survival 4.76-5.00 s, 159.0-164.2 Hz, width 3.85-4.00, confined
in 11-49 % of post-pulse frames -- but **at the driven tile in only 1 of 5 seeds** (centre 0.92; the others 5.30, 5.43,
10.01, 10.03), which is cx5's R175 caveat again. **No arm meets the predeclared working-compass rule in any seed**, and
every surviving bump fails `compass.EPG.bump_rate_hz` by 2.5-3.4x.

The independent 6A pass (Opus, verdict **mostly sound**, corrections 1-13, all applied) reproduced every measurement in
the batch from the raw `.npz` with max |diff| 0.0 and closes with this, quoted verbatim:

**What compass rounds 5 and 6 jointly say.** At the shipped gains the compass question is **closed as a null and opened
as an attribution**: 5A showed that no data-implied type-level change -- the GLNO relabel, the receptor tier, the
monoamine slow class, the fan-in, the cap -- moves the shipped ring off its uniform state, and that the binding
constraint is a DC balance on the relays rather than a missing k = 1 mode; 6A fixed the three tool defects 5A's skeptic
named and then ran the one `edges`-kind counterfactual that decision implied, and the answer is that the DC term is
**real and insufficient**: holding 2 ExR6 + 4 ER6 + 11 ER4m off PEN and EPG lifts the driven PEN from 0.29-0.66 Hz to
40.2-48.3 Hz (`result`, Holm 0.0317) -- ExR6 and ER6 carrying it, ER4m a `null` on PEN -- and buys zero seconds of
confined bump in 5 of 5 seeds, because the same term is what keeps the unstimulated ring near its drive, so removing it
saturates the ring instead of releasing it. Everything that holds a bump across both rounds is a **labelled instrument
or a labelled reference** (`same_type_gain` 1, gE 2 / gD 15, or the hold with the GLNO relabel), every one of them
fails `compass.EPG.bump_rate_hz` by 2.5-3.4x, and three of the four drift off the driven tile -- so
`compass.EPG.bump_survival_s` stays FAIL and the rate and width rows stay NOT_APPLICABLE, and **nothing in either round
licenses an adoption**. What the two rounds jointly license is one measurement and one data question: record per-type
ring rates in the protocol (the rate model's weakest link, 2.1-2.6x high on H3 and 3.5-4.6x on H_ER6, is its ring
rates, and no arm in either round measures them), and settle ExR6's transmitter, receptor and modulatory status in the
animal, which is `UNKNOWN` at the type level and is the single largest term in the balance both rounds are about.

**The corrections.** The slope question 5A and 6A had each answered differently is settled at **8.00 Hz/mV, not 8.27
and not 25.8**: the right comparison is the maximum slope of the *smoothed* f-I at the sigma the rate model **assumes**
(8.00 at u 8.61 by adaptive quadrature, where `lif_fi_prime`'s Gauss-Hermite estimate is not converged and
`SIGMA_MV = 2.0` has never been measured, while the 15-node `cx_wedge.lif_fi` the fixed point actually iterates is not
smooth near threshold), so the comparison's **sign** is earned, the word **"unreachable" is not**, and it is the
measurement -- no bump in S / G / C / CG in 4 of 4 seeds -- that carries the conclusion; the other twelve corrections
read "7 of 8 arms" as **3 of 4 distinct predictions** from a one-bit classifier that tracks `same_type_gain`, restate
the receptor tier's fall as a criterion change rather than a rank move (its tail is ordered by float underflow), fix
three pre-pulse EPG ranges against `state.csv`, withdraw a `silent` claim the protocol's group means cannot decide,
name the hold's one unnamed side effect (`input_norm` re-scaling one EPG cell's surviving inputs by 2 %), note that
`scripts/cx_ring_structure.py` was edited at 08:18:29Z after the `tree_state.json` stamp though nothing on the
simulated path changed, confine the predeclared call's three phrases to H3 where the rule defines them, and record that
the ExR6 -> EPG sign reproduces the presynaptic NT_SIGN rather than being created by the E-PG GluClalpha row.

**An incident, recorded.** An argument-less run of the fixed `scripts/cx_ring_structure.py` -- whose default `--out` is
`out/cx5/structure` -- **overwrote 5A's structure artefacts** (`structure.json`, `structure.md`, `matrices.npz`,
`evidence_glno.json`). `out/` is git-ignored, so they were regenerated at **2026-09-15T08:20:39Z** with `--legacy`,
reproducing every number `compass_ring_mechanism.md` quotes from that file **to the printed digit** (u_PEN -15.785 /
-11.057 mV, lambda_1 +10,019, gamma_crit(k1) 1.998, ExR6 -14.78 @ 183.2 Hz, ER6 -9.77 @ 116.3, ER4m -0.25 @ 22.3, and
the b / b+f / c / f rows), with `out/cx5/structure/REGENERATED_BY_6A.txt` alongside them. The overwrite's own time is
in no surviving file (between 08:03 and 08:05Z per the thread's log), because the only file that carried it was itself
overwritten by the regeneration; **making `--out` required** is the open item that leaves.

**Nothing adopted.** A hold is a counterfactual and can never be a default: `--hold-edges` defaults to `None`, the
shipped path is bit-identical with it absent, no gain, cache, receptor table or `TYPE_NT_OVERRIDE` entry moved, and
`compass.EPG.bump_survival_s` stays FAIL with the rate and width rows NOT_APPLICABLE. H3G's near-bump is a
counterfactual arm, so the GLNO relabel's status is unchanged and stays with `docs/audits/glno_relabel.md` -- what 6A
adds there is only that the sign is no longer untestable at the shipped gains (GLNO fires 135.9-137.6 Hz in H3 and
64.0-83.2 in H3G, against 0.01-0.34 in S). What the round leaves on the list is in `TODO.md` B: per-type ring rates in
`cx_wedge`'s recorded groups, a measurement of the spiking LIF's effective input noise, and the hold PLUS a
wedge-local recurrence as a mechanism question rather than an adoption.

## Session 12, compass round 6B: hold plus recurrence (2026-09-15)

**Answer: no working compass, 0/5 in every arm.** The hold plus EPG-only recurrence H3E and the hold plus
global recurrence H3F both have zero survival and confinement. Five F/H3G runs pass the individual survival
and width ledger rows but fail rate. The full record is `docs/audits/compass_local_recurrence.md` (cx7,
40 CUDA runs, 10 completed jobs, no provenance problems). Nothing adopted; full CPU closeout 441 passed,
19 skipped, 215 subtests, bit-identity gate green and cache MD5s unchanged.

**The two measurements.** Per-type ring rates close 6A's recording gap: S post-pulse ExR6 41.3-42.8 Hz,
ER6 23.1-23.9, below the sigma-4.6 rate model's 107.7 / 51.9. Never-spiked relay membrane sigma is
4.638 mV (128 cells, S CPU seeds 0-1), compared with the assumed 2.0; the smoothed-f-I maximum is lower
at 4.6, strengthening the shipped-ring slope comparison. EPG's 5.521 mV is a spiking-cell reading,
not a clean noise estimate; input sigma 11.800 supplies a quasi-static sensitivity scenario, not an
established upper bound. The scored cx5 validation is unchanged (3/4 distinct predictions, 2/3 rate calls),
while unscored relay predictions change. The fixed point misses H3E's placement in 4/5 seeds.

**What the skeptic refuted. Withdrawn:** "what is still missing is not excitation at the tile but
inhibition everywhere else." H3E's far half is already near background (8.7-12.3 Hz); excess lies next
to the driven block, and recurrence makes off-block activity quieter while recruiting Delta7. The failure
is a seven-wedge, high-rate hump with a systematic +1.58-1.73 wedge offset, not a marginal confinement miss.
The pulse moves a pre-existing high attractor. No arm isolates inhibition of the rest of the ring.
H3F is nearly flat within 15%, not 10%. The ledger, windows, Holm result counts, noise interpretation and
source-stamp limits are corrected in the audit. Fable described the independent pass as "mostly sound";
the reviewer reached its usage limit before a final verdict block, so its completed notes are preserved
without a fabricated summary. The ExR6 transmitter question is superseded by `exr6_evidence.md`;
receptor placement and kinetics remain open. Both experimental flags still default to None.

## Session 12, literature note: Wang's fly-circuit-exploration and two Rockefeller theses (2026-09-15)

Peter Wang's `fly-circuit-exploration` (github.com/pwang724/fly-circuit-exploration; findings index at
pwang724.github.io/fly-circuit-exploration/findings/) is an LLM-driven mining of the MaleCNS v1.0 type-aggregated
graph with hemibrain v1.2 as the replication -- no simulation, four findings, a confirmations page and a self-audit
(`audit-2026-09-13.md`). His own account of the project (X post, 2026-09-15): the first pass stated things as facts
that were wrong, missed two Rockefeller dissertations that had already done the work, and misattributed results;
Thornquist's criticism of the same. The theses: Janke 2025, "A Neuronal Circuit Motif for Leaky Vector
Integration" (Maimon lab; an hDeltaG bump built by integrating vDeltaE synaptic input) and Avritzer 2026, "An
Angular Working-Memory Signal that Guides Drosophila Navigational Trajectories" (hDeltaA integrates travel
direction over ~7-10 s; embargoed to 2027-05-30, abstract only).

**What checks against our graph** (CPU, `connectome.load()`, `abs(W)[post, pre]` summed over the cells of each
type, 2026-09-15). His finding 3, "the velocity signals come from cells nobody has named", gives PS196_b -> GLNO
1,801 syn (19 % of GLNO's input), -> LPsP 942, -> ExR2 899, -> ExR4 686, -> FB3A 195; AN07B037 -> PS196_b; FB3A
(4 glutamatergic tangentials) -> PFNd 5,765, -> hDeltaB 883. **Every count reproduces to the synapse in our
cache** (PS196_b -> GLNO 1,801 = 20.9 % of GLNO's 8,623 non-sign-0 input; AN07B037_a / _b -> PS196_b 419 / 52 --
the two numbers the compass-room entry above already quotes). His finding 4, "the compass has a built-in brake", is
the EPG -> PEN write-position recurrence (hemibrain PENa 3,844 syn at the write tile vs 1,281 at the read tile,
3 : 1, which he reports holds in MaleCNS) that "turns the shifters into anchors", with Delta7 / ExR4 / ExR6 named as
the inhibition "the models leave out" that might cancel it. Our cache stores no per-ROI split, so the 3 : 1 is not
checkable here (EPG -> PEN_a 6,100 / PEN_b 6,230 in total; PEN_a -> EPG 14,398 / PEN_b 9,671); 5A's per-side wedge
table (`compass_ring_mechanism.md`, the EPG -> PEN rows) is the same loop read by wedge -- EPG writes +2.16 / -2.18
wedges through PEN and returns to its own tile.

**Where it lands on ours.** (a) *The same unmapped cell, reached from the other side.* Rounds 1-4 reached PS196_b
from the body: AN04B003 -> PS196_b is where the self-turn report breaks, and PS196_b's L-R moves the same way in
both turn directions under the Coriolis stop-gap (unsigned by construction). Wang reaches it from the compass: it
is GLNO's largest input outside the ring. Both readings say the turn signal into PEN has to come up through
PS196_b, and nobody has recorded PS196_b. That is TODO B's "PS196_b body-model question" with a second,
independent argument behind it, and it makes PS196_b the cell to put an imaging question on if we ever write one.
(b) *The brake.* His is the EPG -> PEN recurrence; ours (5A / 6A) is the ExR6 / ER6 / ER4m DC term on the relays,
and in 6A the two meet: lifting our brake (the hold) does not give a bump, it gives a saturated ring (H3, 144-160
Hz, a five-wedge hump) -- the "anchor" his loop predicts once nothing brakes it -- and only the GLNO sign on top
gets it to a 159-164 Hz near-bump, at the driven tile in 1 of 5 seeds. The two readings are the two terms of one
balance, not competitors, and neither of us has the gains that make it a compass. (c) *hDelta path integration*
(his findings 1-2, both theses) is downstream of a heading bump we do not have, so it is not on the critical path;
it goes on the expectation side of the ledger -- a working compass room should show hDeltaB / hDeltaG / hDeltaA
activity that integrates, and the theses give the time constants to expect.

**The process lesson, and what is not taken.** His failures were literature failures -- facts without a source,
theses and preprints not searched, results attributed to the wrong lab -- the same class as our withdrawn "the
hemibrain name GLNO encodes glutamate" (5B, `glno_relabel.md`). `docs/INTERP.md` 10.4 items 25-27 govern sources
for model changes but said nothing about prose claims about the literature; item 29 now does. Nothing here is a
physiology result and his audit marks the functional claims "proposed here, untested", so no model change follows;
one TODO B item is added (PS196_b from the compass side). Pages read: `findings/03-velocity-sources.html`,
`findings/04-compass-brake.html`, `findings/index.html`, the two thesis records (Rockefeller Digital Commons 808
and 837). The tooling behind his project is his, and nothing here is a claim about it.

## Session 13, the instrumented preset: rounds 4d and 7, the transfer follow-up, and Astra's navigation stack (2026-09-17)

**What happened while the owner's sessions were down.** The owner's accounts were out for several days, so the
round ran on a handoff to Astra (GPT-6). It closed round 6B and round 4d from the batches that had already
landed, reviewed and merged the `feat/instruments` branch with fixes (`instruments_review.md`), and submitted
cx8 -- which it then invalidated itself, because the frozen declaration named `receptor_net_rule="class"` where
`cx_wedge.py --receptor-model shipped` resolves `abs`; the physical commands were right and the record was not.
It re-ran the identical 48 commands as cx8r and ran the one authorized follow-up, cx8t. The owner then asked for
a larger stand-in stack under `instrumented`: `CompassDriver` (imposed angular memory), `compass_ring` (Wang's
reduced EPG/PEN rate loop), `plume` (the published PFL3 comparator with an ideal goal memory), `hunger` (an
explicit metabolic gain) and `flight` (a wing-MN control experiment). Two of the owner's own observations then
forced corrections: flight continued while the fly starved (the flight-priority policy: ground search, bounded
bouts, reserve and odour interruptions) and the fly walked nearly straight past fruit (the plume-steering
correction: a bilateral walking goal and DNa02 L-R feedback onto the existing PFL3 inputs). The room demo's
giant-fibre escape threshold was exposed as `--gf-threshold` in the same stretch. Four independent Opus skeptic
passes then covered the lot -- round 4d, round 7 plus cx8t, the three navigation audits, and the shipped
instrument code -- and all four verdicts are **mostly sound**. Their verdict and claim lines are quoted verbatim
in each audit and in `receptor_verification.md`; their corrections are applied. **This entry closes every
"Independent skeptic pending (Fable, when accounts reset)" line in the round-4d, round-7, transfer-follow-up,
compass-stand-in, navigation, flight-priority and plume-steering entries further down this file**: those entries
are left as written, because they were true when written, and the pass that answers them is here.

**The scientific state, as the skeptics established it.** Round 7 is a **null**, and a well-localized one. The
signed afferent does reach GLNO with a side: V minus S raises GLNO L-R by **+2.2 Hz** (`result`, Holm p 0.0130),
and all six V values exceed all six S values. What does not happen is rotation. A hump **does** form in HG, HGV
and HGV- -- vector strength ~0.72, ~160 Hz, ~3.9 wedges -- and it does not turn: in the three runs that pass the
confinement gate the confined-frame centre slope is **+0.137, -0.086 and -0.069 wedges/s** against an ideal
**+4.0**, and HGV's nearest miss gives +0.171. The gate fails mostly on its `out_above <= 3` clause, not on an
absent bump, so "no eligible follow measurement" is an unavailable comparison, not a zero velocity. The sign
control fails where it matters: GLNO L-R does not reverse in HGV- (+0.69 +- 1.91 Hz, positive in four of six
seeds), and no sign-flipped counterpart of the V arm was declared, so primary 3's sign specificity is untested
where it was measured. HGVp -- the PEN-only hold -- has no bump at all.

**cx8 and cx8r are bit-identical**, which nobody claimed: 6,966 of 6,966 saved metric values and 1,560 of 1,560
NPZ array pairs are exactly equal, and only the run paths and `wall_s` differ. The invalidation really was a
declaration defect alone, and the replacement is a free replication -- two separate submissions twelve minutes
apart on the same house B200 reproduced the batch exactly. That is recorded as a **qualification of
`docs/INTERP.md` 10.4 item 2 for the cx_wedge protocol only**, not as licence to assume GPU reproducibility
anywhere else.

**The transfer follow-up (cx8t).** A direct 90 Hz GLNO challenge does put a correctly signed **~3 Hz** difference
into PEN (HL-HR +2.9939, z 3.1904), while suppressing PEN overall -- the side-balanced mean falls from 24.0 Hz to
5.0 / 3.6 Hz. The fourth contrast came out positive against a predeclared negative prediction, and the reason is
the operating state, not the biology: the cut alone moves PEN L-R by **+11.4 Hz** while the challenge moves it
about **-1.7 Hz**, so the challenge-attributable part carries the predicted sign in both edge contrasts and
contrast 4 as declared could not have been negative at any biology. No unitary transfer and no receptor sign
follows; the audit's refusal to infer one is right, and it now shows the decomposition that justifies it.

**Round 4d closed the level question.** The leg-cycle attribution replicates on a control matched on all three
leg channels at once: C minus L3 is **+2.7 / +3.9 Hz** at AN04B003 (pooled +3.27), **DNa02-left is a `result`**,
and nothing else is -- DNa02-right, the clean yaw SD, straightness and DNa02 L-R are all null -- while **M2
versus C is null on all seven primaries**, so the amplitude/turn law adds nothing beyond the per-phase
modulation. The old cross-batch level model leaves +1.20 / +1.23 Hz in a modulation-free arm at the cycle's own
level point: its extrapolation overestimated the structure term, and the direct matched difference is the number
to quote.

**The milestone, stated the way the navigation skeptic established it.** Under `instrumented` the fly **turns and
finds food** in a 60 s room -- six of six plume rows feed -- with no oracle anywhere in the loop and `raw`
untouched. The instrument's only body-derived inputs are antennal deflections, bilateral concentrations and
interoception flags; no world position, wind angle, fruit position or distance reaches it, and the probe's
`nearest_fruit()` is logging that never gets there. That is the real result, and it is smaller than "the fly
flies and feeds":

- **It never flies and feeds in the same episode.** `powered_s` is 0.00 wherever it feeds, in all 12
  flight-priority room rows and all 12 plume rows; under the corrected policy the odour gate latches for
  essentially the whole episode, so artificial flight is **unreachable in any room containing fruit**, not merely
  bounded. The only powered-flight rooms are the superseded navigation flight arm, which fed in none of six.
- **The plume cue is not something the model's own nose could resolve.** The law reads a bilateral contrast off
  the physical concentration field -- noise-free, pre-transduction -- and multiplies it by 0.1 m / 0.001 m = a
  gain of **200**. The observed contrast is 0.26-0.57 %, so this is close to a sign-of-contrast turn; through the
  model's own ORN law (1 + 150c/(c+0.5)) a 0.3 % difference is ~0.1 Hz per ORN, well under the Poisson noise of
  the 0.25 s window.
- **The feedback servo-inverts the biological stage it runs through.** The integral supplies more than half the
  PFL3 input in every room row, and the balanced control needs -5.94 Hz to hold DNa02 L-R at +0.06 Hz -- the loop
  is cancelling the parent circuit's own bias rather than consulting its gain. What the connectome clearly owns
  here is **DNa02 -> motor readout -> yaw**; the goal, the 200x gradient and the integral are the instrument's.
- **The rooms do not test hunger.** Every plume row hits energy exactly 0.0000 at **18.0-18.2 s**, before every
  first contact (20.1-37.3 s), so the gain is pinned at 1.0 for most of each episode and never leaves [0.90, 1.0].
- **The six environment seeds are near-repeats.** `world.make_room` places apple, orange, banana and lime at
  seed-independent coordinates, only the grapes and blueberries jitter, the start is identical and three rows
  share the same 5 deg initial heading; the lime is the nearest fruit at 22.6 cm and **four of the six rows fed
  there**.
- **None of these rooms pins a number.** The native event-driven CUDA path is not reproducible run to run: the
  exact-workload gate fails with per-cell rate differences to **44.9 Hz** (voltages to 77.6 mV, conductances to
  247) while `poisson_p` and `drive` are bit-identical, so the divergence is in the kernels and not the inputs.
  Re-running the same command with the same seeds gives a different trajectory and could give a different feeding
  outcome.

**The code state.** `raw` is byte-identical: the bit-identity golden predates the instrument branch (96c9a24, an
ancestor of it), the three cache MD5s are unchanged (`c50c598a...`, `ac131529...`, `bf01d724...`) with compiled
CSR `ef23cc27bea13be7f6a96f3c04fd3737`, and the CPU suite is 503 passed / 19 skipped. **No default moved**
anywhere in `flyverse/` or `scripts/`; every change is a new trailing keyword with a neutral default. Three
things are fixed in this closeout. `FlyBrain.attach` compared the brain's preset with the *object's* declared
preset, so an object declaring `required_preset="raw"` attached to a raw brain, drove Poisson and produced
provenance `preset: "raw"` with a non-empty instrument list -- the guard now refuses any instrument attach unless
the preset is `instrumented`, with a test. Every shipped `describe()` now carries a **`replaces`** field
(`"input"` for the sided afferent, `"configuration"` for the hold / relabel / gain records, `"computation"` for
`compass`, `compass_ring`, `plume`, `hunger`, `flight`), because the owner's section-5 extension admits
program-shaped stand-ins as instruments and they have to say so; `_check_instrument` now also requires
`law == "unverified"` or a non-empty `source` / `sources`. And `hunger`, `plume` and `flight` now declare their
body-derived inputs and effect route in `describe()`: `hunger`'s record used to read `reads {} / writes {}`, which
implied it did nothing while it gates plume's turn and flight's lift.

**Process notes.** The first is Fable's own: the WIP commit 62cdefb's message said "the Report block is complete"
while the file still carried five `PENDING_*` placeholders -- a breach of `docs/INTERP.md` 10.4 rule 7's
placeholder gate, recorded here rather than quietly fixed. `instruments_review.md` reviewed `a41d0f2..6d1c501`
and reported 468/469 passing; it never saw `compass.py`, `navigation.py`, `interoception()` or the four
navigation instruments, and it now says so at the top -- its "no new body-to-neural-module interface was
introduced" is scoped to that range, because `interoception()` plus the four `observe_*` receivers are exactly
such an interface. The `undetermined` label on the two empty follow comparisons is the frozen declaration's own
word, but `common.compare` is never called on those rows and returns `underpowered` on an empty sample, while
INTERP 2.4 / 10.2 reserve `undetermined` for a deterministic reference -- they are now read as unavailable
comparisons. Two verdicts are **orientation-dependent**, because `compare` divides by the reference arm's SD:
primary 3 reverses to |z| 2.96 and a null, and cx8t's transfer reverses to z -2.70 and a null, while the
symmetric Welch statistics are +7.26 and +5.05 and U is unchanged. Round 4d's `probe_vnc_drive.py` -- the script
every one of its 24 jobs ran -- is not one of the 52 files in `provenance.source_fingerprint`, so the driver is
identified by the scheduler's verbatim job lines instead. And the infrastructure-identifier scrub is still owed
before release: eight files by the code skeptic's list (`receptor_verification.md`, `NOTES.md`,
`docs/media/README.md`, and the five ignored `out/*/predeclared.json` families).

**What is next, as questions.** (1) Does the plume result survive a **goal-only arm with no DNa02 feedback**? No
such arm exists, mean |demand| did not rise between diagnostic and validation while mean |yaw| tripled, so the
feedback bridge is the load-bearing change -- but that is an inference, not a measurement. (2) Does a
**transduced-contrast law** work: read the ORNs' own rates instead of the physical field, and let the Poisson
noise in? If it does not, the 200x gradient is doing the navigating. (3) Does a **V- arm** (the sign-flipped
afferent without the hold) reverse GLNO L-R where primary 3 measured it? The held arms do not. (4) Can a
**deterministic-kernel gate** be met before any room number is quoted again -- an exact-workload pass, or the
torch-sparse path, or repeated draws with the spread reported? (5) What does the **29-check suite under
`instrumented`** say with the three-instrument list, at three draws, beside the `raw` column? The compass
stand-in's own run already rejected admission on `taste.MN9_hz` (seed 1, 1.690 -> 4.296 Hz, FAIL -> PASS outside
the declared gap), so the honest expectation is another rejection, and it is worth having on the record.

## Session 13, continued: the history rewrite (2026-09-17)

The release blocker in TODO A -- infrastructure identifiers in committed files -- was closed by rewriting the
whole history rather than the tip: `git filter-repo` over all 161 commits with a replacement map (hostnames,
the cluster user, the shared-filesystem root, the scheduler's name, three rented-box IPs and the workstation
home path -> `<cluster-host>`, `<cluster-node>`, `<cluster-node-2>`, `<cluster-user>`, `<cluster-fs>`,
`<scheduler>`, `<rented-box-ip>`, `<workstation-home>`), applied to blobs and commit messages, verified by a
full-history grep returning nothing. Every commit SHA changed; the ten remote branches were force-pushed; a
pre-rewrite bundle is kept outside the repository. Audit prose that quoted a run directory or a host now reads the
placeholder, which is the convention the audits already used. `tests/test_cluster_run.py`'s canned scheduler
fixture was renamed by the same map and still passes (74 passed with the bit-identity and instrument files).

## Batched brains and the RL environment

* `Brain(c, batch=B)` and `OpticLobe(c, r, batch=B)` keep state as (B, N): one sparse matmul serves all
  B flies. Measured on the RTX 4090: spmm 0.58 ms for B=1, 1.3 ms for B=32, 1.7 ms for B=64. Gotcha
  that cost a 4x slowdown: `W @ x.T` with a non-contiguous transpose (5.6 ms) -- use `.contiguous()`.
* `flyverse/env.py` `FlyRoomEnv(batch=B)`: B flies on the picnic table, vision (1 ray per ommatidium
  for speed), smell and taste as in the demo; obs = rates of the 1,314 descending neurons; action =
  (forward, yaw); reward = cm of progress towards the nearest fruit + 1 per frame of tasting - 0.01.
  Oracle (walk straight to the fruit) earns ~+13 per second, random ~-1. Throughput: 58 ms per step
  for B=32 (1.8 ms per fly-frame, 5.5 fly-seconds of brain per wall second); the ray tracer dominates
  at large B if 7 rays/ommatidium are used.
* `scripts/train_decoder.py`: OpenAI-ES over a linear decoder (1,315 x 2) with antithetic perturbations,
  rank fitness and common random numbers (all flies spawn at the same pose within a generation).
  The question it asks: does descending-neuron activity carry enough information to find the fruit?
  (The connectome is untouched; only the readout is learned.)
* **Result of the first ES run** (64 flies x 40 generations x 5 s, sigma 0.3, lr 0.1, ~40 min): no
  learning. Mean return stayed at the step-penalty floor (about -5) in every generation except one whose
  common spawn point was already touching fruit (+417). Evaluation over 96 random spawns, 5 s episodes:
  decoder 19 +- 101, random 23 +- 112, still 37 +- 138, oracle 123 +- 162 (final distance 11 / 13 / 13
  / 2.7 cm) -- the decoder is indistinguishable from doing nothing, and the variance shows the reward is
  dominated by spawns that happen to touch fruit. Reading: with a linear readout of 1,314 DN
  rates and 64 rollouts per generation the signal is too weak -- DN activity here is dominated by
  self-motion optic flow, the fruit are small in the visual field, odour gradients only reach the
  (saturating) antennal lobe, and ES on 2,630 parameters with returns that depend mostly on the spawn
  is a hard estimator. Things to try, in order: a curriculum (spawn 5-10 cm from fruit, then farther),
  observations from LC/MeTu/AL projection neurons rather than DNs, PPO (PufferLib on Linux) with
  frame-stacked observations, and actions that stimulate DN groups so the brain's own motor pathways
  do the walking. The env, batching and trainer are in place; the science is open.
* **PufferLib**: `pip install pufferlib` fails to build on this Windows / Python 3.13 box (needs its C
  extensions). The env follows the vectorised reset/step convention, so on Linux wrap it with
  `pufferlib.emulation` and train with PuffeRL/PPO; the batched brain is the throughput lever either way.

* **Colour** (`scripts/probe_colour.py`): sweeping a 2 cm sphere of each fruit material past the fly
  gives responses that differ by material only in the third decimal, but the pattern is the right one:
  Tm5a tracks UV/blue reflectance (r = +0.8), Dm8 / Tm20 / Mi15 anti-track green, LC15 tracks UV/blue,
  MDN anti-tracks UV/blue. So colour opponency exists in the wiring and reaches projection neurons,
  weakly. A local (receptive-field) analysis with a larger stimulus would quantify it properly.
* **Mushroom body**: Kenyon cells stay silent under odour (96 PN synapses per KC, PN population mean
  15 Hz -> ~1.3 mV of synaptic drive vs the 7 mV needed). Real KCs need ~half their claws active at
  high rates; with our saturating antennal lobe only the 3-4 odour glomeruli are active. In-brain
  learning (KC->MBON plasticity gated by PAM/PPL1 dopamine) therefore needs a working PN->KC stage
  first -- e.g. sparser, stronger ORN->PN transmission or a KC-specific gain.

## Ideas / next steps

Done in session 2: sweet GRNs, olfaction, colour probe, flight/loom, batched brains, RL env + ES
trainer. Still open:

* **Direction selectivity** (the biggest missing computation): per-type time constants alone give
  DSI ~0.05. Options: (a) fit per-cell-type gains/time constants to flyvis's published optimum;
  (b) a multiplicative (Reichardt-style) nonlinearity at T4/T5 dendrites; (c) use flyvis directly as
  the optic lobe (it is a torch model on the FlyWire optic lobe; map its cell types onto MaleCNS).
  With DS, LPLC2 becomes a real loom detector and the escape/self-motion trade-off disappears.
* **Antennal lobe realism**: presynaptic (GABA-B) inhibition of ORN terminals and depressing ORN->PN
  synapses would bring PN rates to physiological levels; then KCs can be made sparse and the
  KC->MBON plasticity (Huang & Luo 2024 rule, PAM/PPL1 gated) becomes meaningful. DA/OA/5-HT are
  already sign-0, i.e. reserved for modulation.
* **Per-cell-type synaptic gains** instead of one 0.275 mV synapse plus a fan-in cap: a small table
  (visual projection, DN, MN, KC ...) would let the GF, DNa02 and T4 all sit at sensible thresholds.
  Could be fit against the known behaviours (sugar -> MN9, loom -> GF, optic flow -> DNa02).
* **Sustained flight**: which descending neurons initiate wingbeat (DNa? DNp? "flight DNs" in the
  fly-escape pathways.json) -- stimulate candidates and see whether DLMn/DVMn stay on.
* **Leg sugar GRNs** need their own second-order reference set (the labellar set is done).
* **RL**: on Linux, wrap `FlyRoomEnv` with PufferLib and run PPO; try actions = stimulation of DN
  groups instead of direct body control, so the brain's own motor pathways do the walking.
* Try Shiu's dt = 0.1 ms and no STD/adaptation to see how far the vanilla model gets now that the
  NT-sign fixes are in; and the `-significant-only` weights file as a faster-loading alternative.

## Reference projects

* **stonkfly** (github.com/nftechie/stonkfly): MaleCNS v1.0, same three files; C++ event-driven LIF
  kernel with the Shiu constants; R1-R6 mapped via strongest L1/L2/L3 contact onto hex columns, R8p/R8y
  driven by blue/green of a price chart; readout DNp20 L/R + DNpe017 gate (admittedly arbitrary);
  anti-Hebbian KC->MBON learning with PAM/PPL1 reward pulses.
* **doomfly** (github.com/nftechie/doomfly): same brain, ViZDoom at 640x480; drive
  `30*lum/(0.02+lum)` mV, lamina bias 12 mV, 10 ms photoreceptor low-pass; turn = DNp20 R-L, move =
  DNpe017; ~0.2-0.66x real time on an M1 Pro (serial C++). "Visual, conditioning and survival
  validation gates" failed for their learning variant.
* **fly-escape** (github.com/dzhng/fly-escape): 3,016-neuron subgraph (min weight 5), dimensionless LIF
  in Rust/wasm, taps Tm2/Tm20 with luminance/blue, body IDs for olfactory/flight/landing/proboscis
  readouts, three.js front end.

## RL: PufferLib

For reinforcement learning on top of the brain (training a readout/decoder, evolving plasticity
parameters, or letting the fly learn), use **PufferLib** (github.com/PufferAI/PufferLib): vectorised
envs with far higher throughput than gymnasium/SB3 wrappers, native PPO ("PuffeRL"), and a C/Cython
env API if the room sim needs to be ported for speed. Not installed yet (`pip install pufferlib`).
Design constraint to remember: the brain step is the bottleneck (~0.6 ms/step for the whole CNS on
GPU), so batch many environment copies through one batched brain (spike vectors as a (B, N) matrix;
the sparse matmul cost is nearly flat in B) rather than running many brains in parallel processes.


## Backends: Apple MPS / CPU (session 5)

- `flyverse/device.py` picks cuda > mps > cpu (`FLYVERSE_DEVICE` overrides). Torch has no compressed
  sparse (CSR) kernels on MPS, so `sparse_matrix()` builds COO there; COO spmm on MPS is ~5-8 ms for the
  24.6 M-synapse brain matrix and 3.6 ms for the 8.8 M-synapse optic-lobe recurrence (about 5x off memory
  bandwidth; gather + `index_add_`, int32 indices, fp16 values and a cumsum-over-CSR-segments trick were
  all tried and are no faster or lose precision).
- The brain's synaptic input is now optionally event-driven (`LIFParams.event_driven`, default on for
  non-CUDA devices): the pre-major (CSC) weight layout is gathered for the neurons that fired this step
  and `index_add_`ed into g. Cost is spikes x fan-out (~150 synapses per neuron) instead of 24.6 M: 0.8 ms
  at the demo's 40-60 spikes/step, 1.6 ms in a 2%-of-neurons storm, vs 5-8 ms for the spmm. It works for
  any batch B (the gather carries a brain index). Results are bit-identical to the spmm on CPU and under
  deterministic drive on MPS; with Poisson input on MPS the two paths drift apart by fp32 rounding order
  (0.14 Hz max after 150 ms, 16 Hz on a handful of neurons after 500 ms) -- the network amplifies
  near-threshold differences, as it does between any two summation orders.
- Readouts: `Brain.rate_np()` caches one device->host copy of the rates per step for all B = 1 readouts
  (`rates`, `mean_rate`); the demo made ~30 such copies per frame, each a GPU sync (7 ms on MPS).
- Demo frame on an M-series Mac (FRAME_MS = 10): default 158 ms (optic 67, draw 57, brain 26, trace 5) =
  0.06x real time; `--fast` (brain dt 1 ms, optic dt 2 ms, camera at 240x150 upscaled) 70 ms = 0.14x.
  Before this work the frame was ~450 ms. The optic-lobe rate model is the floor now: every unit is active
  so there is no event-driven shortcut, and its cost is (frame_ms / dt_ms) x 4.5 ms. Further options, all
  model or engineering trade-offs: prune the smallest normalised optic-lobe weights, overlap the pygame
  drawing with the next frame's GPU work (MPS is asynchronous until a `.cpu()`), or a fused Metal kernel
  for the LIF update (the 10 elementwise ops per step cost ~0.3 ms in launches).

## Metal kernels for the Mac (session 6)

The MPS analogue of the CUDA-graph work: `flyverse/metal.py` (event scatter, fused LIF update,
SIMD-group CSR spmv, fused optic substep) through `torch.mps.compile_shader`. Full-fidelity demo
0.15x -> 0.29x real time, `--fast` 0.24x -> 0.39x; numbers, validation and what is left in
`docs/PERFORMANCE.md`. Spikes are identical to the torch path; continuous state agrees to ~1e-4
(fast-math contraction). Next target on the Mac is the ray tracer (~10 ms of small launches per frame).

Session 6, continued: the ray tracer as one Metal kernel plus a fix for a scene-repacking bug that hit every
MPS trace (device `mps:0` != `mps`). Full-fidelity demo 0.29x -> 0.78x real time, `--fast` 0.39x -> 1.19x.
The brain's own GPU time (~9 ms per 10 ms frame at full fidelity) is now the floor on the Mac.

## BatchSim follow-up (September 11, 2026)

The session 9 batching follow-up is implemented in `flyverse/batch_sim.py`, exported
as `BatchSim`. One batched brain now serves independent full room environments:
seed-specific fruit geometry and air phases, full sensory rays, surface walking,
flight, metabolism and optional programs. The body constants come from the scalar
objects; edge transitions/takeoff/landing use the scalar methods. Programs keep
independent state and their neural stimuli are coalesced by selector/duration into
batch-row rates. `scripts/batch_sustain.py` runs sweeps and saves per-row JSON;
`scripts/profile_batch.py` measures full frames and CPU/CUDA work.

Cluster checks: all 12 batching checks pass, including B=1 agreement with the demo
(ordinary programs and `cx`), 200 mixed body-state comparison subtests, per-row
scene/stimulus/reset isolation, and native B=4 checkpoints. The wider selected
suite had 48 passes and 10 opt-in native-kernel tests skipped. A four-fly `cx`
rollout completed, but this is engineering validation, not a new sustain result.

B200 development timings (same integration settings, full brain, native events,
cuSPARSE and graphs requested): plain mixed-fruit room, scalar 1.95 aggregate
fly-s/wall-s -> B=16 8.09; `cx` + apple + fence, scalar 0.54 -> B=64 11.68. The latter
is 54.8 ms per batch frame, so each individual rollout advances at 0.18x real time.
GPU operation counts stay approximately constant across batch sizes; physical
body/sense work for the `cx` configuration is 0.46 ms at B=1, 1.36 ms at B=64.
The machine was shared, including validation overlap late in the second sweep;
do not interpret the ratios as idle-GPU guarantees. Details and measurements:
`docs/BATCH_SIM.md`, `docs/batch_profile.json`.

RNG caveat: environment seeds retain their room/air meaning, but the existing
batched brain uses one generator over (B,N), so changing B changes the neural
draw layout. These are independent rollouts, not exact replays of B separately
seeded Sim processes. Program RNGs are now included in BatchSim checkpoints.

## Session 12, round 4d: three-channel matched level control (2026-09-15)

The fixed point landed: L3's realised chordotonal / hair-plate / campaniform means were 87.444 /
47.091 / 24.978 Hz, inside every per-side tolerance against this batch's cycle arm C. On 24 CUDA runs
(six per arm, one house submission), **C-L3 AN04B003 +2.6687 / +3.8782 Hz**, pooled **+3.2734**, is a
direct matched structure term. Both relay sides and DNa02-left (+0.1342 Hz) pass the primary family;
DNa02-right, clean yaw SD, straightness and DNa02 L-R are null. M2-L3 modulation alone gives +2.8521 /
+4.0290 Hz at the relay and passes DNa02-left and clean yaw too. M2-C is null on all seven primaries,
replicating round 4c's null; it is not an equivalence test. The sided secondary still separates the
DNa02 tripod-conditioned swing (-0.3543 C, -0.0884 M2).

The old level model leaves +1.2028 / +1.2344 Hz residual in modulation-free L3, an extrapolation error;
the earlier corrected +4.49 / +4.60 / +4.72 Hz terms are cross-batch context, not the new matched size.
Nothing adopted. `docs/audits/level_fixed_point.md` records every family and per-seed list, the three changed source-stamp entries (two scripts and the wrapper), and a misleading descriptive decomposition rate column unused by the
primaries. Sections 4-11 and Report were missing alongside section 0 and are reconstructed from the
saved batch with all analyses re-run. CPU suite 441 passed / 19 skipped, 215 subtests; golden and cache
unchanged. **Independent skeptic pending (Fable, when accounts reset).**

## Session 12, compass round 7: the signed afferent reaches GLNO (2026-09-15)

The input arrives with a side: V minus S GLNO L-R is **+2.2183 Hz**, result with Holm p 0.0130.
The raw SD is only 0.0013269 Hz, so the enormous z is not a large absolute physiological effect.
The compass still does not follow: HGV has **0/6 eligible runs** at the predeclared 50% confinement
gate; the two follow contrasts are **undetermined**, not zero and not null. PEN L-R HGV-HG is null
(+0.7813 Hz, z 0.4221), as is DNa02 L-R (all values zero). The PEN-only hold has no confined frames;
its contrast is nevertheless null under the full rule (|z| 2.3613 <3 despite Holm p 0.0130).
The descriptive low/high k levels have one eligible run each and select no gain. No arm passes the
joint survival/rate/width ledger. The signal reaches GLNO without demonstrating a working compass.

The valid batch is cx8r, 48 CUDA runs, six per arm, zero analysis problems. The initial cx8 attempt
is retained as invalid: Astra's new frozen record said class, whereas --receptor-model shipped runs
sign/abs. Fixed the metadata and actual-CLI guard, then repeated the complete unchanged experiment.
Independent CPU trace verification reproduces 972 measurements and 52 source hashes; every run
matches the corrected freeze. `docs/audits/compass_velocity_route.md` carries the six tests, all
per-seed lists and follow traces. CPU 469 passed / 19 skipped, 220 subtests; golden/cache unchanged.
Nothing adopted. Primary 3 result with no demonstrated follow selects the one authorized GLNO-to-PEN
transfer diagnostic; the conditional suite and odour room remain gated. **Independent skeptic pending
(Fable, when accounts reset).**

## Session 12, compass round 7 follow-up: direct GLNO input reaches PEN (2026-09-15)

The single authorized follow-up is complete: one house submission, six seeds in each of six conditions.
Under HG (ring DC held, GLNO signed), a fixed **unverified 90 Hz direct GLNO challenge** gives
HL-HR GLNO L-R **+149.0748 Hz** and PEN L-R **+2.9939 Hz** (z 3.1904), both results, Holm p 0.008658
in the four-test family. Every HL PEN L-R is positive and every HR value negative. This establishes
side transfer under the strong imposed input. Side-balanced PEN mean also falls from H0's 24.0426 Hz
to 5.0211 / 3.6345 Hz under left / right forcing. It is not a following compass.

The edge control is a substantial operating-state change: holding GLNO -> PEN gives C0 a PEN L-R
of -11.0738 Hz and mean 54.1743 Hz before adding either challenge. HL-CL +12.5861 Hz and HR-CR
+9.7340 Hz are both positive results, so the **right-edge negative sign prediction failed**.
The intact/cut contrasts do not isolate unitary transfer at matched presynaptic rates. No extra
interaction test or receptor mechanism is inferred. DNa02 L/R challenge-window means are zero in
all 36 runs; no run passes the joint compass ledger and no stimulated arm has a run above 50%
challenge-window confinement. The suite/odour-room gate from the afferent experiment remains unmet.

All 36 runs match the frozen declaration and all 648 trace checks pass. The source/control verifier
checks 72 frozen and 58 recorded source hashes against submitted commit dc98bfe; the three recorded
files outside the frozen list are identified as retrospective checks. All six H0 controls reproduce
the earlier HG metrics and arrays exactly. Every per-seed row and the operating-state plot are in
`docs/audits/compass_velocity_route.md` section 6. Nothing adopted; no second follow-up submitted.
Final CPU 473 passed / 19 skipped, 220 subtests; original golden and MaleCNS cache bytes unchanged.
**Independent skeptic pending (Fable, when accounts reset).**


## Session 12, imposed compass memory experiment (2026-09-15)

The owner authorized a fast program/control arm under instrumented while the biological compass remains
unresolved. CompassDriver integrates realized yaw and writes a continuous Poisson bump to the 46 EPGs
through the existing module surface. It supplies angular memory; there is no world-heading/goal oracle,
visual anchoring, motor overwrite or synapse change. The exact 50 Hz / 35 degree law is unverified.
Wang's requested findings were read at 80b94e6608cf927ca2c9f2bbce0577c5a998684c, including the corrected
qualification that fitted models can integrate with the feedback contacts present. Count-based failure
does not identify the biological correction. Sources and limits are in `audits/compass_standin.md`.

All 48 controlled trajectories (six seeds, B=8) pass the frozen engineering gates: stationary phase-error
p95 5.45-8.50 deg, moving p95 8.41-23.51 deg, including estimator lag. All six controlled traces reproduce
exactly after scheduler/capture optimization. The three-draw, 29-check comparison changes one taste status
FAIL to PASS outside the gap, so strict preset admission fails despite no passing-row regression. The legacy
pulse-memory compass row remains a known gap; this instrument follows its imposed phase, not an arbitrary
neural pulse. The 60 s room observations do not establish food finding or replace the full admission gate.

Native CUDA adds 54.22 us per 10 ms brain frame at B=1 (11.01%, misses the <=10% target), 3.19% at B=8,
1.22% at B=32. CPU/CUDA traces confirm zero scalar readbacks in the optimized steady frame. Native B=6
room median is 12.84/13.18 ms raw/instrumented; UI has only a visual smoke check. Two same-code eager
room repeats diverge, as does eager versus captured; full-brain exactness gates fail and remain recorded.
Two failed capture attempts are retained; the successful retry and final lifetime precaution pass the
house fixtures. Final CPU 482 passed / 19 skipped / 220 subtests; house 15 exact lifecycle checks plus
8 existing CUDA tests / 2 skipped / 3 subtests. Original golden and both worktrees' cache MD5s unchanged.
Experimental opt-in only; raw remains default and no physiology adopted. **Independent skeptic pending
(Fable, when accounts reset).**


## Navigation instruments: recurrent compass, plume, hunger and flight (Astra, 2026-09-15)

Owner-requested experiments now compose as `--preset instrumented --instruments compass plume hunger flight`.
`compass_ring` is an alternative Wang-style EPG/PEN recurrent rate model, with the EB-count-ratio brake
as a separate diagnostic. Its angular gain remains uncalibrated. Plume uses the published PFL3 comparator
plus synthetic goal memory/output mapping; hunger is explicit energy-to-navigation gain, not a claimed
insulin/sNPF circuit. Flight is a neural wing-MN servo targeting the current body's 100 Hz lift equilibrium.
No parent weights, receptors, body constants or raw default change. Incompatible/dependent combinations
fail early. Both room implementations supply held sensory/internal inputs and use normal neural readout.

Final CPU: 497 passed / 19 skipped / 220 subtests, original golden and both cache copies unchanged.
House: 26 exact multi-module CUDA lifecycle checks; the corrected harness records its actual instrumented
controllers. Full-brain neural forcing gives signed steering and about 100 Hz power, with sated/feeding
controls off. Six 60 s rooms with flight remain airborne 59.67 s each; plume+hunger yields one feeding
row (8.08 s), versus zero with compass alone. This does not establish robust food finding, source
localization or a recovered native circuit. No flight landing/altitude controller or added energy cost.

All-four brain-frame timing observations: 1.015 / 4.152 / 13.969 ms at B=1/8/32 on B200; the exact-workload
gate fails, including rate/spike differences at B=8/32, so no clean causal overhead or whole-room equality
claim. UI smoke with brain map passes. No gains fitted, no preset admission/default adoption; independent
skeptic pending. Sources, declarations, every per-room row, failures and reproducible derived tables:
`audits/navigation_instruments.md`. Implementation 3cac3cc; lifecycle provenance correction c7ead86.

## Flight gives foraging priority (Astra, 2026-09-15)

The owner observed starvation during sustained instrumented flight. The old `flight` request
stayed on down to energy 0.05; hunger strengthened steering without interrupting lift. The
optional instrument now searches on foot for 20 s, caps powered bouts at 8 s, withdraws its
wing drive at energy <= 0.35 (re-arm >= 0.50) or sustained strong neural LH odor, and holds an
airborne interruption until touchdown. These are declared engineering rules, not physiology.
Raw, body constants and metabolism are unchanged; native escape hops remain possible.

One frozen house submission passes 26 lifecycle checks, 17 exact captured/eager transition
checkpoints and both six-row room gates. Initially starving airborne flies land in 0.47-0.52 s.
The normal-energy room requests no artificial wing drive: odor keeps the flies investigating
on foot; one feeds 8.06 s. Two depleted rows feed 15 s each, four run out of energy. Food finding
is still unreliable. This fixes perpetual flight, not the remaining navigation problem.

CPU 500 passed / 19 skipped / 220 subtests; original golden and both cache copies unchanged.
All constants frozen before submission, no tuning or default adoption. Audit, exact per-row
table, provenance checks and reproduction: `audits/flight_foraging_priority.md`.
Implementation 266d274. Author self-review complete; independent skeptic pending Fable.

## Plume steering reaches the motor circuit and local odor sources (Astra, 2026-09-16)

The owner's `compass plume hunger flight` report still showed near-straight walking past fruit.
The house diagnostic finds a confident compass but only 0.281-0.687 Hz mean absolute DNa02
asymmetry; the old goal stays upwind throughout sustained odor. The optional plume controller
now uses bilateral smell samples for its walking goal and DNa02 feedback to correct its existing
PFL3 input. The gradient law and bounded integral feedback are explicitly unverified engineering;
no body command, parent synapse, transmitter, raw default or physical constant is changed.

Frozen full-brain controls give signed DNa02 responses +9.378 / -8.945 Hz, zero added input
while feeding/sated, and wind fallback when airborne or without physical smell. All 26 exact
CUDA lifecycle checks pass. Six of six compass room rows feed 15 s; five of six ring rows feed
at least 1 s. The scalar room-demo check reaches food at 21.6 s, feeds 15 s and ends at energy
0.8412. These are 60 s functional observations, not a broad success rate or adoption suite;
some flies reach zero energy before feeding, and zero is not death in the shipped metabolism.

CPU 503 passed / 19 skipped / 220 subtests; original golden and caches unchanged. Headless UI
with brain map passes, without an interactive FPS claim. Source ef90c23, scalar harness 74f64de.
Exact rows, source-verified reproduction, trajectories, source limits and author self-review:
`audits/plume_steering.md`. Independent skeptic pending Fable. Restart old plume episodes:
the expanded sensory/feedback checkpoint state is intentionally incompatible.
