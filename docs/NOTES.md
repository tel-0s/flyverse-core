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

## Ideas / next steps

* Direction selectivity: T4/T5 need temporally asymmetric inputs (Mi4/Mi9/CT1 slow vs Mi1/Tm3 fast);
  give rate units per-type time constants (flyvis has them) and check T4a-d for preferred directions.
* Sweet GRN identification via the G2N-1/Rattle/Usnea connectivity above; then proboscis extension.
* Colour: fruit already differ in UV/B/G reflectance; measure Dm8/Tm5/Tm20 opponency by moving a coloured
  sphere through the receptive field; check whether MeTu/LC responses discriminate apple vs lime.
* Olfaction is trivial to add: ORN types are annotated by glomerulus (ORN_DA1 ...); a fruit odour = a
  set of ORN classes at Poisson rates falling off with distance (fly-escape does this).
* RL with PufferLib: the room is already a step()-able Sim; wrap it as a PufferLib env with the readout
  (or a small linear decoder over DN rates) as the trained policy, batching B fly copies through one
  brain (spikes as a (B, N) matrix). See the PufferLib section below.
* Learning inside the brain: KC->MBON anti-Hebbian rule with PAM/PPL1 reward pulses (stonkfly/doomfly
  do this with Huang & Luo 2024 equations); DA/OA/5-HT are currently sign-0 (modulatory), which is
  exactly where plasticity gating should hook in.
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
