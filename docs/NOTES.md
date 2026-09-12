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
and spike-frequency adaptation.

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
  both LC4 and LC10a are at 0 Hz -- and LC10 is a small-object motion detector in the animal, so the
  object screen showed it the wrong stimulus. Next: the same with self-motion (the apple sweeping
  the eye as the fly turns).
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
every nodulus input but IbSpsP inhibitory) -- these are the next experiments.

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
breaks nothing when removed** and four of its five targets are inhibitory, so it goes. Adoptions are one at a time
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
bump spin the walking fly through PFL3 -> DNa02?); PEN L / R shift and the PFN / hDelta / PFL3 readout; the GLNO sign;
T2 / T3 as ON-OFF units (their inputs sum Mi1 / Tm3 (figure z -8) and Tm1 / Tm2 / Tm4 (+3 to +5) linearly and cancel
the object exactly where NOTES 9 loses it; T3 is 21 % of LC11's input) via a rectified baseline in `optic.py`, scored
with `probe_figure_ground.py` and the sweeping-ball assay; the self-motion GF assays that decided LPi x4 and
edge_len 20 mm added to the suite, and the optic layer's own hand-crafted measures (five pair gains, gain_out, L1
normalisation, drive clip) audited the way the LIF's were; the odour gate at the 40 cm foraging start (channel 12.5 Hz
against a 13 Hz gate) and cross-channel specificity; an NT-rescue counterfactual through the LIF sections.

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
