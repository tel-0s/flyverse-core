# flyverse

Simulations, games and other strange places to put a fly connectome. The default model is **MaleCNS
v1.0** (HHMI Janelia FlyEM + Google Research, Sept 2026: 167,106 neurons, 25.6 M connections, 124 M
synapses, brain **and** ventral nerve cord). The fly gets colour vision, smell, taste, wind sense, a
body that walks, jumps and flies, and a room with a table with fruit on it. Two female FlyWire
releases load through the same API -- FAFB v783 (brain, complete optic lobe) and BANC v888 (brain +
VNC) -- as `connectome.load(dataset="fafb" | "banc")`.

The default `raw` brain uses the unchanged connectome plus the documented physiological assumptions
below. Explicit programs and `instrumented` experiments add labelled computations; their output is
not attributed to the wiring alone.

![The flyverse room console](docs/demo_ui.png)
*The room console: body camera, small orbit view, both retinal mosaics, antennal inputs, and
simultaneous population, motor and body readouts. `--brain-map` opens the soma map.*

Moving: [`docs/media/loom.gif`](docs/media/loom.gif) -- 25 s of the console, a loom at 8 s, the giant-fibre jump and the re-landing.
[`docs/media/wind_apple.mp4`](docs/media/wind_apple.mp4) -- 25 s with one apple and the default wind (what the fly did, honestly captioned in [`docs/media/README.md`](docs/media/README.md), with every command line).
Toolkit: [`docs/media/toolkit_loom_gf.png`](docs/media/toolkit_loom_gf.png) -- `interp` on the loom -> giant-fibre path: the `paths` stage map and the `atlas` readout.

## Run it

```
pip install -e .
python -m pytest -m "not gpu and not data and not cluster" -q   # needs no data, no cache, no GPU

python scripts/fetch_data.py --list         # what the manifest knows, and what is already present
python scripts/fetch_data.py --malecns      # ~3.7 GB from Janelia's public bucket, hash-verified
python -m flyverse.connectome               # compile the graph cache once (15 s), print N and nnz

python scripts/room_demo.py                 # live window: fly on the table
python scripts/room_demo.py --fruit apple --fence                # one apple, a fence around the table top
python scripts/room_demo.py --start floor                        # begin on the floor (it has to fly to get back up)
python scripts/room_demo.py --program cx --escape-gating         # simulated central-complex steering, plugged into the brain
python scripts/room_demo.py --gf-threshold 50                    # raise the giant-fibre escape threshold (default 33 Hz)
python scripts/room_demo.py --headless --seconds 20 --loom-at 4 --gif out/room.gif
```

The CPU path runs all of this, slowly; a GPU is what makes the demo interactive and what the
benchmark suite and the batched sweeps need. `SPACE` pauses, `L` looms a ball at the fly, `T`
teleports it to the apple, `?` lists the rest.

`--gf-threshold HZ` sets the body's base escape threshold; `--escape-gating` adds habituation on top.
It does not change voluntary wing-powered takeoff. Saved states retain the threshold; `--load`
restores that saved value.

Install details (CUDA wheels, extras, the female releases, the cluster runner): **`docs/INSTALL.md`**.
Controls, flags, speed numbers, the batched simulator and the file-by-file map: **`docs/OVERVIEW.md`**.

An experimental heading-memory stand-in is available with `--preset instrumented --instrument compass`.
It integrates realized yaw and continuously drives the biological EPG cells; it supplies angular memory,
not a recovered neural attractor or a food-seeking policy. The UI and saved provenance label it. `raw`
remains the default. See [instruments](docs/INSTRUMENTS.md) and the [validation audit](docs/audits/compass_standin.md).

Use `--instruments compass plume hunger` to add the experimental goal/steering bridge and metabolic gain;
add `flight` for the wing-control experiment. `compass_ring` is an alternative recurrent-model compass.
All five are **computation stand-ins**: they replace a computation the model cannot do rather than supply a missing
input, each says so in its `describe()` record (`replaces: "computation"`), and they are admitted as instruments
only under the owner's extension in [PRESETS_SPEC](docs/PRESETS_SPEC.md) section 5.
See [the instrument guide](docs/INSTRUMENTS.md) for dependencies, scientific limits and validation status.

## What is simulated

| stage | count | model |
|---|---|---|
| photoreceptors R1-R6, R7 (UV), R8 (blue / green) | 5,895 placed on 1,466 real hex columns (of 6,098 `ol_sensory`) | ray-traced spectral radiance [UV,B,G,R] per ommatidium, low-pass + contrast adaptation |
| optic lobe (lamina, medulla, lobula, lobula plate) | 89,390 | graded rate units on the signed, input-normalised connectome (flyvis-style); T4/T5 rectify with delayed inhibition and are direction selective |
| visual projection neurons, central brain, VNC, motor neurons | 71,618 | Shiu et al. 2024 leaky integrate-and-fire on the GPU, plus the additions in "What we added" below; synapse signs from the presynaptic transmitter, corrected per postsynaptic type by a receptor-expression table built from six transcriptomic sources |
| taste | 165 labellar sugar GRNs | Poisson while touching fruit -> Usnea / Rattle / Phantom / G2N-1 -> proboscis MN9 |
| smell | 2,639 ORNs in 53 glomeruli, sided by their PN targets | every fruit is a wind-blown plume (Gaussian, puffing) sampled by two antennae 1 mm apart |
| wind | Johnston's organ C / E neurons, sided by their AMMC/WED targets | antennal deflection per side from the wind vector in the body frame |
| body | -- | walking on any surface (table top, sides, underside, walls, ceiling: over edges, around corners), escape jumps, flight with pitch and roll; rates in, motion out, nothing decided |

The body reads named groups the connectome already separates: DNp20 + HSN/HSE (optomotor -- the
populations whose L-R actually flips under sustained yaw; DNp04 + LPT27/30 were used until session 8
and do not), DNp18 / DNp33 (wind direction, gated by the odour signal), DNa02 (goal steering), MDN
(back up), MN9 (proboscis), the giant fibre (escape jump), DLMn / DVMn (wingbeat), wing steering MNs
(yaw and roll). Per-stage detail and the files: `docs/OVERVIEW.md` section 3.

## What the connectome does, unprompted

Measured behaviours of the wiring under the model, each with a probe script that reproduces it.
Numbers in brackets are the nine round-3 suite runs at the shipped defaults
(`docs/audits/guard_suites_r3.md`); the rest name their own probe.

* **Direction selectivity** (`probe_motion.py`): T4a/T5a prefer front-to-back, b back-to-front, c up,
  d down, from the one-column offsets of their Mi4 / Mi9 inputs, speed-tuned [`motion.min_dsi` 0.241,
  correct preferred direction for all 8 subtypes in 9 of 9].
* **Looming escape** (`probe_loom.py`): a black ball approaching at 1 m/s drives LC4 / LPLC2 -> the
  giant fibre, which spikes at ~3.5 cm range; TTMn (the jump muscle) follows it 1:1. Walking through
  a textured room leaves the giant fibre silent [loom GF peak 46.7-49.5 Hz and an escape in 9 of 9,
  against a 33 Hz threshold and a 99th-percentile walking GF of 17-25 Hz].
* **Sugar -> proboscis, bitter shuts it off** (`probe_taste.py`, `probe_bitter.py`): the labellar
  sweet GRNs drive the known second-order neurons (Usnea, Rattle, Phantom, G2N-1) and MN9; adding the
  bitter GRNs suppresses it. Under Shiu et al.'s uniform-synapse rules on MaleCNS, MN9 goes
  139.9 Hz -> 0.8 Hz (they report 78 -> 3 on FlyWire); under this project's calibration 5.5 -> 0
  [all four values identical in 9 of 9; `taste.MN9_hz` 10.93].
* **Wind direction** (`probe_wind.py`): the strongest lateralised signal in the model. DNp18 fires on
  the side the wind comes from, DNp33 on the opposite side, then DNge016, DNg99, DNge175, DNg05_a,
  DNp19 and DNpe017 [`wind.DNp18_flip_hz` +44.7 to +45.8, `wind.DNp33_flip_hz` -49.4 to -50.0].
* **Rotation** (`screen_rotation.py`, `benchmark.py`): DNp20, HSN and HSE flip their left-right
  asymmetry with the direction of yaw rotation -- the optomotor signal the body reads
  [`rotation.group_flip_hz` -8.7 to -10.3]. DNp04 and LPT27/30, the group used until session 8, do
  **not** flip and are driven by walking instead. The flip is a readout the body can steer with; the
  connectome itself cannot, because DNa02 stays silent -- deficit 1 below.
* **Odour code** (`probe_smell.py`, `screen_odour.py`): glomerulus-specific PN responses (apple:
  DM1 / VA2 / DM4; banana: DM2 / VM2), sparse Kenyon cells, MBONs at 1-4 Hz; the lateral-horn apple
  channel separates plume from clean air [17.3-17.5 Hz at 8 cm vs 4.3-4.9 Hz plume-free].
* **Measured motor map** (`screen_dns.py`, every DN type stimulated in its own batched brain copy):
  DNg02 -> wing power muscles at 146 Hz with no leg drive (Namiki's wingbeat-amplitude cluster),
  DNa08 and DNp31 likewise; giant fibre -> TTMn 47 Hz; DNa02_L -> ipsilateral leg motor neurons;
  MDN -> the backward-walking premotor set; DNge080 / DNge062 -> proboscis.

`scripts/benchmark.py` scores any parameter set on all of these at once (14 sections, ~4 min on a
4090). Runs are chaotic: repeat before trusting a 20 % change.

## Where the model stands

Behavioural assays and how the plain model scores, from `docs/BENCHMARK_BATTERY.md` (`pass` = the
connectome model produces it with the plain body; `partial` = the neural signal is there but the
motor path is not, or one branch works; `program` = only a behaviour program or module produces it;
`gap` = a documented, localized model deficit; `untested` = cheap to run, not run):

| assay | status | one line |
|---|---|---|
| looming escape, short mode (giant fibre) | **pass** | loom GF peak 46.7-49.5 Hz, an escape in 9 of 9 suite runs, threshold 33; self-motion no longer fakes it |
| wind orientation | **pass** | kept in every suite run as a regression guard |
| optomotor response (rotating drum) | partial | HS / DNp20 flip with sustained yaw; DNa02 is held below threshold, so nothing reaches the legs |
| looming escape, long mode | partial | the wing-raise signature scales the right way with expansion rate, but there is no mode switch: the GF responds to final size |
| landing response | fail | no leg extension at any stop-short approach; GF correctly stays under threshold |
| cast-and-surge in a plume | program | the LH odour channels follow the puffs; surge / cast is `AnemotaxisProgram` or `cx.CompassSteering` |
| local search after food loss | program | the central complex is silent at the defaults |
| hunger-modulated thresholds | gap | dopamine / octopamine / serotonin synapses are sign 0: neuromodulation has no route |
| spontaneous turning (free walking) | gap | see "Three localized deficits" |
| small-object detection (LC11 / LC10a) | gap | see "Three localized deficits" |
| spontaneous take-off | gap (no animal reference) | 3.0 voluntary + 2.0 escape per 1,000 fly-s at the default |
| grooming cascade, flight saccades | untested | readouts exist, assays not written |

The 29-check regression suite at the shipped defaults reads **27 PASS / 0 FAIL / 2 KNOWN GAP** in
three of three independent draws; the two gaps are `object.LC10a_flip_hz` and
`compass.wedge_cells_persisting` (`docs/audits/guard_suites_r3.md`, batch `guard7-97ce35`, 27 jobs,
0 failed). Which numbers were measured at which commit: `docs/REPRODUCIBILITY.md` section 8.

## What we added to the point model, and why

Shiu et al.'s LIF with one 0.275 mV synapse per contact reproduces sugar -> proboscis but, on the
whole CNS with sensory input, has two regimes: silent or epileptic. What was added, and which of it
is a **stop-gap** (something standing in for a measurement nobody has made) rather than a mechanism:

| addition | value | mechanism or stop-gap |
|---|---|---|
| spike-frequency adaptation | 1.5 mV/spike, tau 200 ms | mechanism (real, uncalibrated magnitude): without it KCs / PEN / PAM lock at 300 Hz |
| per-connection saturation cap | 60 synapse-equivalents | mechanism: PSP amplitude does not grow linearly with synapse number |
| fan-in normalisation | above 5,000 input synapses, unitary x `(5000/total)` | mechanism: large neurons have low input resistance |
| same-type synapse damping | x0.1 | mechanism: such populations are typically gap-junction coupled in the animal, not chemically self-exciting |
| short-term depression | antennal lobe only (ORN and LN terminals, u 0.2) | mechanism where documented (Kazama & Wilson 2008); **globally it blocked every descending command**, so it is off elsewhere |
| graded rate optic lobe up to T4/T5 | `optic.py` | mechanism: these cells are graded in the animal; a spiking lamina / medulla does not propagate at all |
| descending -> VNC x3, visual projection -> descending x2 | `DEFAULT_PATH_GAIN` | **stop-gap** for per-cell-type synaptic strengths: the difference between motor commands that reach the legs and ones that do not |
| LC4 / LPLC2 -> DNp01 x3 | `DEFAULT_TYPE_PATH_GAIN` | mechanism (Ache et al. 2019), magnitude a stop-gap: the loom escape margin |
| "T5 delayed inhibition" x5 | `DEFAULT_PAIR_GAIN` | **stop-gap, mislabelled until session 10**: 78 % of the 101,619 edges it multiplies are cholinergic Tm9 / Tm4, so it is mostly a **drive** gain (`docs/audits/optic_measures.md`) |
| receptor-corrected synapse signs | `receptor_model='sign'`, 48,295 of 25.6 M entries | mechanism from data (six transcriptomic sources); `None` restores the presynaptic rule byte-for-byte |

Every field and value, with the opt-in modules that are **not** part of the default:
`docs/REPRODUCIBILITY.md` section 1. The argument for each: `docs/OVERVIEW.md` section 4.

Two things we measured and did **not** add: the monoamine synapses (2.7 M, sign 0) can be routed
through a slow receptor class, but one scalar cannot carry dopamine, octopamine and serotonin at once
-- at the value that leaves behaviour untouched it fails the taste check on CPU, at ten times that
the mushroom body runs away in half a second (`docs/audits/monoamine_slow_term.md`); and a
per-transmitter unitary strength breaks the take-off motor at every data-anchored bracket
(`docs/audits/unitary_strength.md`). Nothing from either was adopted and no default moved.

## Three localized deficits

What is *not* added is a fix for a behaviour the model fails. The three that matter are **localized
rather than tuned**: a named population, a named link, and what kind of fact the loss is, with a
data-driven next measurement and an explicit list of the hand-crafting that was not done.

**1. The fly walks straight.** Clean-frame yaw SD 2.6-2.8 deg/s and straightness 0.995 against an
animal that saccades 200-450 deg/s every ~250 ms. The one steering neuron, DNa02, is not merely unlit
but held below threshold by sign-correct tonic inhibition (-1.6 / -2.0 mV of steady conductance
against a 7.0 mV gap), while every lateralised excitatory route into it is silent at source: PFL3 and
AOTU015 at 0.000 Hz, most of LLPC1 never firing and LPT22 cancelling what survives, and every
`vnc_sensory` proprioceptor at 0 Hz -- the walking VNC runs open-loop
(`docs/audits/deficit_turning.md`).

*Round 3:* an opt-in body-model module closes that loop -- a tripod leg cycle (`body.LegCycle`) plus
a side-split haltere readout make the afferents fire at literature-typical rates, DNa02 fires (0.54 /
0.38 Hz) and yaw SD rises to 7.7-7.9 deg/s with straightness 0.83-0.87. **It moves yaw; it does not
turn.** No clean frame in any arm of any batch exceeds 100 deg/s; the extra yaw is a fixed left drift
of +1.0-1.2 deg/s in 15 of 16 flies, which is the connectome's own asymmetry; and the sided afferent
report follows the turn instead of leading it. A signed steering command and a saccade generator are
named as missing from the wired graph. **Nothing was adopted and no default changed**; the module
ships off by default (`docs/audits/body_sided_state.md`, `round3_integration.md`).

**2. Small objects do not reach the small-object channel.** The figure is carried by the lamina,
medulla, T4c/d and LPLC2 and lost at LC11 / LC10a's inputs, to opposite-signed carrier convergence at
T2/T3, the stochastic optic feedback destroying the residual, and l1 pooling over 94 / 32 columns at
the LC cells (`docs/audits/deficit_object.md`).

*Round 3:* on a matched assay (elevation, distance, angular size and angular speed held per frame) at
six runs per arm, **no size preference is called for either type** -- LC11 12/12 `null`, LC10a's one
`result` fails Holm. A fixed-anatomy comparison of eight physiological arms found **no mechanism that
passes**: per-stream rectification is the only arm that carries the upstream carrier figure and it
releases bar, grating and full-field flicker responses at LC11, the opposite of the animal. The three
owed items closed and changed no answer: neither population localizes under a static 4.5-deg probe,
40 of 40 contrast-matched rectangle verdicts are `null`, and the same-device re-run reproduces on an
H200 and is `PARTIAL` on a B200. **Nothing adopted, no default changed** (`object_matched_assay.md`,
`object_compare_r2.md`, `object_localizer_r3.md`, `object_rectangles_r3.md`,
`object_samedevice_r3.md`).

**3. The compass has no rotation input.** At 90 deg/s the bump moves 0.00 +/- 0.01 wedges/s against
4.0 ideal. GLNO -> PEN is sign 0 (19.4 % of PEN's raw input), PS196_b never fires, and the visual
route arrives direction-blind (`docs/audits/deficit_rotation.md`).

*Round 3:* the transducer supplies the missing input and the answer does not change. The signed
self-turn report exists at AN04B003, reaches PS196_b at 1-3 Hz and **is gone at GLNO**; the bump
drifts `|mean| <= 0.005` wedges/s in all 11 arms. At the shipped gains **no bump forms at all** --
`compass.EPG.bump_survival_s` is 0.00 s in 48 of 48 runs under every per-transmitter unitary bracket,
because a transmitter scale multiplies the tuned Delta7 inhibition and the untuned ring feedback by
the same factor and cannot set their ratio. The block is now two facts: no attractor at the shipped
gains, and the sign-0 GLNO -> PEN link. **Nothing adopted, no default changed**
(`docs/audits/unitary_strength.md`, `body_sided_state.md`).

## The same model on two female connectomes

`connectome.load(dataset="fafb" | "banc")` compiles FlyWire FAFB v783 and BANC v888 into their own
caches with `root_id -> bodyId`, alias-normalised type names and vocabulary maps for superclass, NT,
side and neuropil; the receptor table and the expectation ledger transfer by type name. MaleCNS
identity is preserved exactly (the three cache md5s, the CSR fingerprint and the
`test_bit_identity.py` golden are unchanged). Anatomical counts run about 1 : 0.59 : 0.28 across the
three releases, so nothing is compared unscaled.

**The result, descriptively.** On the walking replicate (five house jobs, five seeds x 16 flies x
60 s, shipped parameters, batch `cbwalk-384afd`, 30 of 30 simulations exit 0), **the female CNS walks
straighter than the male at the shipped defaults**: clean yaw SD 0.275 +/- 0.004 deg/s (BANC) against
2.641 +/- 0.146 (MaleCNS), straightness 0.9984 against 0.9943, with BANC's DNa02 silent bilaterally.
The leg-cycle increase in turning replicates qualitatively (BANC 0.381 -> 2.651 deg/s, `result`
within-dataset) but **the neural pattern does not**: BANC's DNa02_L stays silent while R reaches
0.0985 Hz, and its PS059 sits near 0 Hz where MaleCNS runs 20 Hz -- so the cancellation mechanism the
male audit named is not shown to be shared. **This is a descriptive cross-connectome comparison, not
a sex test**: one male reconstruction against one female reconstruction, with different coverage
(BANC's optic lobes are incomplete: T2 853 vs 1,466), different naming, and no per-individual
replication. A missing type match is not established sex specificity.
`docs/audits/connectome_backends.md`.

## Use the brain in your own simulation

The connectome model is a control system with one small surface (`docs/CONTROL_SURFACE.md`,
`docs/ARCHITECTURE.md`): the environment supplies physical sensor values, the brain returns named
rates, the body classes (or your own) turn rates into motion.

```python
from flyverse import FlyBrain
from flyverse.body import FlyState, Locomotion

fb = FlyBrain(modules=["antennal_lobe", "mushroom_body", "gustatory", "mechanosensory", "central", "descending", "vnc"])   # modules=None: everything
fb.smell({"DM1": 0.4, "VA2": 0.2}, {"DM1": 0.2, "VA2": 0.1})   # concentration per glomerulus, left / right antenna
fb.wind(0.3, 0.3); fb.taste(0.0)                              # antennal deflection; sugar contact
fb.step(10.0)                                                 # 10 ms of brain
motor = fb.motor()                                            # MotorRates: fwd_dn, turn_L/R, wind_ipsi_L/R, lh_odour, gf, power, ...
pose, legs = FlyState(), Locomotion()
cmd = legs.readout(motor, dt_s=0.01); legs.step(pose, cmd, 0.01, bounds=(-1, 1, -1, 1))
```

`modules` picks the parts of the CNS to simulate (`flyverse/regions.py`); retained synapses keep the
strength they have in the full model. `FlyBrain(cuda_graphs=True)` replays captured frames,
`fb.step_budget(wall_ms)` fits the brain into a game tick and reports the time dilation, and
`flyverse.async_brain.AsyncFlyBrain` runs it on its own thread so a fixed-tick host never waits on
the GPU.

If you need the brain to do something it does not do, add code to it rather than fork it:
`fb.add_hook` runs a named callable around each step; `fb.attach` registers a module that reads
selected cells and writes drive or Poisson forcing between frames (a small Torch network, or a whole
synthetic spiking sub-brain); `Connectome.extend` adds synthetic cells with negative bodyIds beside
the biological graph; `LIFParams(surrogate_grad=True)` backpropagates through a short window so a
boundary encoder or motor decoder can be trained against the connectome. All of it is opt-in,
recorded in provenance and checkpoints with a declared `kind`, and inert until used: with nothing
attached the simulation is bit-identical to the default path on CPU. See
**[Optional extensions](docs/EXTENSIBILITY.md)** and
**[Hooks, modules, graph extension](docs/CONTROL_SURFACE.md#hooks-modules-graph-extension)**.

## When a behaviour fails: run the toolkit

`flyverse/interp` is eight tools that localize a deficit on the shipped weights and never edit them:
`decompose` (what drives a cell set, per frame, by presynaptic type / transmitter / receptor tier),
`trace` (where along the depth from a sensory population a stimulus is lost), `paths` (effective
k-step signed gains A -> B, and which links are silent), `lesion` (check x lesion delta matrices and
double dissociations), `atlas` (what every motor readout does when population X is stimulated),
`health` (per-type operating point of a rollout), `ledger` (measured responses against the curated
expectations in `flyverse/data/expected_responses.csv`) and `export`.

**[`docs/INTERP.md` section 10](docs/INTERP.md)** is the seven-step procedure to run when a ledger
row or a benchmark check fails: name the readout and the expectation, do the structure pass on CPU,
record one GPU batch with runs as replicates, trace then decompose at the lost stage, classify the
loss with counterfactual arms, attribute with the lesion matrix, write it up and export it. The first
two steps cost minutes and no GPU:

```
python scripts/interp_ledger.py --results "out/interp/*/*.json" out/benchmark_suite.json --status "FAIL,KNOWN GAP" --json out/interp/ledger/fails.json
python scripts/interp_paths.py --a "class=olfactory" --b "DNa02" --k 3 --json out/interp/paths/orn_dna02.json
python scripts/interp_decompose.py analyse --static --target DNa02 --json out/interp/decompose/dna02_static.json
```

That is where GLNO -> PEN (sign 0) and the three silent lateralised routes into DNa02 were found,
before any GPU job ran. The toolkit's output is a **diagnosis**, never a fix: under the project rule
a diagnosis licenses a measurement, a mechanism the connectome data imply, or a documented swappable
module -- and a gain, a bias, a threshold or a sign the data cannot see is hand-crafting, which the
write-up must name as such.

## Reproducibility

**[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** records the shipped defaults field by field,
the cache fingerprint (`sum|W|` 121,460,584, compiled-W md5 `ef23cc27bea13be7f6a96f3c04fd3737`), the
dataset releases and their hashes, the commits, and which number here was measured at which one.

Two things to know before quoting anything:

* **The GPU rollout is not seed-reproducible.** Two identical runs of one tree at one seed on a B200
  diverge from frame 500 of 6,000 (`out/proprio_bitid/compare.txt`: *VERDICT: NOT identical*, with
  `cache_sum_abs_W` equal in both). **Runs are the replicate unit**, four minimum per arm.
* **Bit-identity claims are CPU claims only** (`tests/test_bit_identity.py`, a recorded golden over a
  deterministic synthetic graph). Nothing here claims a bit-identical GPU rollout.

## Where things are

`docs/OVERVIEW.md` has the full file-by-file map. The short version:

| | |
|---|---|
| `flyverse/connectome.py`, `flyverse/backends/` | compile a release into a signed CSR cache (MaleCNS, FAFB, BANC) |
| `flyverse/brain.py`, `flyverse/optic.py`, `flyverse/retina.py`, `flyverse/world.py` | the LIF CNS, the rate optic lobe, the eyes, the ray-traced room |
| `flyverse/fly.py`, `flyverse/motor.py`, `flyverse/senses.py`, `flyverse/body.py` | the control surface: senses in, `MotorRates` out, motion |
| `flyverse/programs.py`, `flyverse/cx.py` | opt-in behaviour programs and the compass-steering module |
| `flyverse/interp/` | the eight interpretability tools |
| `flyverse/batch_sim.py`, `flyverse/env.py` | batched room rollouts; the vectorised RL environment |
| `scripts/` | `room_demo.py`, `fetch_data.py`, `benchmark.py`, `probe_*.py`, `screen_*.py`, `interp_*.py`, `cross_connectome.py` |
| `docs/INSTALL.md`, `docs/OVERVIEW.md`, `docs/REPRODUCIBILITY.md` | install, the long form, what was measured where |
| `docs/BENCHMARK_BATTERY.md`, `docs/INTERP.md`, `docs/NOTES.md` | the assays, the toolkit contract, everything learned session by session |
| `docs/audits/` | one record per audit, every number with its batch line |

## Licence, citation and data

Code: **MIT** (`LICENSE`). Cite flyverse with `CITATION.cff`, and cite the MaleCNS v1.0 release and
Shiu et al. 2024 as well -- flyverse is a simulator built on their data and their neuron model.

The repository ships **no connectome data and no third-party tables**.
`flyverse/data/manifest.json` records the URL, SHA-256, citation and licence of every file
`scripts/fetch_data.py` can fetch:

* **MaleCNS v1.0** -- Janelia FlyEM, <https://male-cns.janelia.org>, CC BY 4.0 at the time of writing
  (check the download page); paper Cell 2026, S0092-8674(26)00942-6.
* **FlyWire FAFB v783** and **BANC v888** -- Codex public releases, CC BY 4.0; cite the releases and
  the original reconstruction papers. FlyWire NT predictions are used where the model reads them.
* **External expression tables** (Ozel 2021, Davis 2020, Kurmangaliyev 2020, Nern 2025, Fly Cell
  Atlas / Davie 2018, the typing tables) -- **not redistributed**; `data/external/` is git-ignored.
  Cite them from the manifest, do not re-host them.

Inspirations: stonkfly, doomfly (nftechie), fly-escape (dzhng).
