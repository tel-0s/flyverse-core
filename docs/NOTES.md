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
  Making DA/OA/5-HT purely modulatory (sign 0) was *not* sufficient on its own and is left at +1.
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
