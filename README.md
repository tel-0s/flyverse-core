# flyverse

Simulations, games and other strange places to put a fly connectome. The default model is **MaleCNS
v1.0** (HHMI Janelia FlyEM + Google Research, Sept 2026: 167,106 neurons, 25.6 M connections, 124 M
synapses, brain **and** ventral nerve cord). The fly gets colour vision, smell, taste, wind sense, a
body that walks, jumps and flies, and a room with a table with fruit on it. Two female FlyWire
releases load through the same API -- FAFB v783 (brain, complete optic lobe) and BANC v888 (brain +
VNC) -- as `connectome.load(dataset="fafb" | "banc")`.

The default preset is **`raw`**: the unchanged connectome, the LIF, the receptor table, the senses
and the motor readout, bit-identical on the CPU path (`tests/test_bit_identity.py`, a golden over a
deterministic synthetic graph); no GPU rollout is claimed to repeat
([`determinism_gate.md`](docs/audits/determinism_gate.md)). Everything added on top is either named
in the additions table below or lives in the opt-in `instrumented` preset, where every instrument
declares what it replaces, whether its law is sourced or `unverified`, and what would retire it.
**Nothing is adopted into `raw`** ([`docs/PRESETS_SPEC.md`](docs/PRESETS_SPEC.md)).

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

The CPU path runs all of this, slowly; a GPU is what makes the demo interactive and what the benchmark
suite and the batched sweeps need. `SPACE` pauses, `L` looms a ball at the fly, `T` teleports it to
the apple, `?` lists the rest. `--gf-threshold HZ` sets the body's base escape threshold and
`--escape-gating` adds habituation on top; neither changes voluntary wing-powered takeoff, and saved
states retain the threshold (`--load` restores it).

Install details (CUDA wheels, extras, the female releases, the cluster runner): **`docs/INSTALL.md`**.
Controls, flags, speed numbers, the batched simulator and the file-by-file map: **`docs/OVERVIEW.md`**.

## What is simulated

| stage | count | model |
|---|---|---|
| photoreceptors R1-R6, R7 (UV), R8 (blue / green) | 5,895 placed on 1,466 real hex columns (of 6,098 `ol_sensory`) | ray-traced spectral radiance [UV,B,G,R] per ommatidium, low-pass + contrast adaptation |
| optic lobe (lamina, medulla, lobula, lobula plate) | 89,390 | graded rate units on the signed, input-normalised connectome (flyvis-style); T4/T5 rectify with delayed inhibition and are direction selective |
| visual projection neurons, central brain, VNC, motor neurons | 71,618 | Shiu et al. 2024 leaky integrate-and-fire on the GPU, plus the additions below; synapse signs from the presynaptic transmitter, corrected per postsynaptic type by a receptor-expression table built from six transcriptomic sources |
| taste | 165 labellar sugar GRNs | Poisson while touching fruit -> Usnea / Rattle / Phantom / G2N-1 -> proboscis MN9 |
| smell | 2,639 ORNs in 53 glomeruli, sided by their PN targets | every fruit is a wind-blown plume (Gaussian, puffing) sampled by two antennae 1 mm apart |
| wind | Johnston's organ C / E neurons, sided by their AMMC/WED targets | antennal deflection per side from the wind vector in the body frame |
| body | -- | walking on any surface (table top, sides, underside, walls, ceiling: over edges, around corners), escape jumps, flight with pitch and roll; rates in, motion out, nothing decided |

The body reads named groups the connectome already separates: DNp20 + HSN/HSE (optomotor -- the
populations whose L-R actually flips under sustained yaw; DNp04 + LPT27/30 were used until session 8
and do not), DNp18 / DNp33 (wind direction), DNa02 (goal steering), MDN (back up), MN9 (proboscis),
the giant fibre (jump), DLMn / DVMn (wingbeat), wing steering MNs. Detail: `docs/OVERVIEW.md` 3.

## Use the brain in your own simulation

The connectome model is a control system with one small surface (`docs/CONTROL_SURFACE.md`,
`docs/ARCHITECTURE.md`): the environment supplies physical sensor values, the brain returns named
rates, the body classes (or your own) turn rates into motion.

```python
from flyverse import FlyBrain
from flyverse.body import FlyState, Locomotion
fb = FlyBrain(modules=["antennal_lobe", "gustatory", "central", "descending", "vnc"])   # modules=None: everything
fb.smell({"DM1": 0.4, "VA2": 0.2}, {"DM1": 0.2, "VA2": 0.1})   # concentration per glomerulus, left / right antenna
fb.wind(0.3, 0.3); fb.taste(0.0)                              # antennal deflection; sugar contact
fb.step(10.0)                                                 # 10 ms of brain
motor = fb.motor()                                            # MotorRates: fwd_dn, turn_L/R, wind_ipsi_L/R, lh_odour, gf, power, ...
pose, legs = FlyState(), Locomotion()
cmd = legs.readout(motor, dt_s=0.01); legs.step(pose, cmd, 0.01, bounds=(-1, 1, -1, 1))
```

`modules` picks the parts of the CNS to simulate (`flyverse/regions.py`); `FlyBrain(cuda_graphs=True)`
replays captured frames, `fb.step_budget(wall_ms)` fits the brain into a game tick, `AsyncFlyBrain`
runs it on its own thread. To make the brain do something it does not do, add code rather than fork:
`fb.add_hook`, `fb.attach`, `Connectome.extend` (synthetic cells with negative bodyIds beside the
biological graph) and `LIFParams(surrogate_grad=True)` are all opt-in, recorded in provenance and
checkpoints with a declared `kind`, and inert until used -- with nothing attached the simulation is
bit-identical to the default path on CPU. See **[Optional extensions](docs/EXTENSIBILITY.md)** and
**[Hooks, modules, graph extension](docs/CONTROL_SURFACE.md#hooks-modules-graph-extension)**.

## What the raw model does unprompted, and where it stops

Out of the box, with nothing attached, the wiring produces a small set of real sensorimotor
behaviours and stops well short of a foraging animal. It **escapes a loom**: a black ball approaching
at 1 m/s drives LC4 / LPLC2 -> the giant fibre to 48.1 / 55.3 / 48.8 Hz against the 33 Hz threshold,
with an escape in all three draws, while the 99th-percentile giant-fibre rate during ordinary walking
stays at 13.5-22.5 Hz. It is **direction selective**: `motion.min_dsi` 0.241-0.246 with the correct
preferred direction for all 8 T4/T5 subtypes in every draw. Its **wind descending neurons lateralise
hard** -- DNp18 +44.8 to +47.0 Hz on the side the wind comes from, DNp33 -50.3 to -50.7 Hz on the
other -- which is the strongest lateralised signal in the model. **Sugar drives the proboscis and
bitter shuts it off** (calibrated MN9 3.93-5.52 Hz, 0 Hz with bitter added). And it **walks**, on any
surface, with the optomotor group's L-R flipping with yaw direction. What it does not do: the ring
holds no bump (`compass.wedge_cells_persisting` 0 in every draw); the fly does not turn (clean-frame
yaw SD 2.56 deg/s, 2.45-2.57 with no fence, straightness 0.995, against an animal that saccades
200-450 deg/s every ~250 ms, because DNa02 is held below threshold by sign-correct tonic inhibition
while its three lateralised excitatory classes are silent at their sources -- PFL3 0.000 Hz, AOTU001
/ AOTU015 at or near 0.000 Hz, most of LLPC1 never firing with LPT22 cancelling what survives -- and
the fourth class, the wind through PS230, is live and lateralised but ~90x under dose, at ~0.078 mV
against the 7.0 mV threshold gap); small objects never reach the small-object channel
(`object.LC10a_flip_hz` |0.0007-0.0098| against a 1.0 Hz criterion); there is no flight state,
because the octopaminergic drive that would gate the wing motor neurons has no route through a graph
whose monoamine synapses carry sign 0; and **no raw arm shows directed food search** -- the raw model
has no odour-guided goal, and a plain fly's meals come from wandering into fruit before it leaves the
table (`docs/NOTES.md` session 8). Sources: the 29-check ledger below and its caption,
[`deficit_turning.md`](docs/audits/deficit_turning.md), [`deficit_object.md`](docs/audits/deficit_object.md),
[`deficit_rotation.md`](docs/audits/deficit_rotation.md), `docs/NOTES.md` sessions 8 and 12-14.

## The 29-check ledger

| check (criterion) | `raw`, 3 draws | `instrumented`, 3 draws |
|---|---|---|
| `rest.spikes_per_step` (< 5) | PASS x3 | PASS x3 |
| `taste.MN9_hz` (> 2) | PASS, **FAIL**, PASS | PASS, **FAIL**, PASS |
| `smell.PN_hz` (< 100) | PASS x3 | PASS x3 |
| `smell.KC_active` (> 0) | PASS x3 | PASS x3 |
| `dn.DNa02_L_leg_asym_hz` (> 0.3) | PASS x3 | PASS x3 |
| `dn.MDN_top_hz` (< 250) | PASS x3 | PASS x3 |
| `dn.DNp09_top_hz` (< 250) | PASS x3 | PASS x3 |
| `walk.GF_max_hz` (< 38) | PASS x3 | PASS x3 |
| `walk.power_max_hz` (not none) | PASS x3 | PASS x3 |
| `walk.power_sustained_hz` (< 50) | PASS x3 | PASS x3 |
| `loom.GF_peak_hz` (>= 20) | PASS x3 | PASS x3 |
| `loom.escape_cm` (not none) | PASS x3 | PASS x3 |
| `rotate.DNp20_flip_hz` (< -2) | PASS x3 | PASS x3 |
| `motion.min_dsi` (>= 0.1) | PASS x3 | PASS x3 |
| `motion.correct_directions` (== 8) | PASS x3 | PASS x3 |
| `loom_escape.GF_peak_hz` (>= 33) | PASS x3 | PASS x3 |
| `loom_escape.escapes` (>= 1) | PASS x3 | PASS x3 |
| `walk_gf.p99_hz` (< 38) | PASS x3 | PASS x3 |
| `rotation.group_flip_hz` (<= -3) | PASS x3 | PASS x3 |
| `object.LC10a_flip_hz` (abs >= 1.0) | **KNOWN GAP x3** | **KNOWN GAP x3** |
| `bitter.calibrated_sugar_MN9_hz` (> 2) | PASS x3 | PASS x3 |
| `bitter.calibrated_sugar_bitter_MN9_hz` (< 1) | PASS x3 | PASS x3 |
| `bitter.shiu_sugar_MN9_hz` (> 50) | PASS x3 | PASS x3 |
| `bitter.shiu_sugar_bitter_MN9_hz` (< 10) | PASS x3 | PASS x3 |
| `wind.DNp18_flip_hz` (>= 15) | PASS x3 | PASS x3 |
| `wind.DNp33_flip_hz` (<= -15) | PASS x3 | PASS x3 |
| `odour.apple_channel_8cm_hz` (>= 10) | PASS x3 | PASS x3 |
| `odour.apple_channel_clean_hz` (<= 6) | PASS x3 | PASS x3 |
| `compass.wedge_cells_persisting` (>= 6) | **KNOWN GAP x3** | **KNOWN GAP x3** |

*Tallies 27/0/2, 26/1/2, 27/0/2 in both columns. Six runs on NVIDIA B200 GPUs, draw seeds 0, 1, 2
per preset, 0 analysis problems (batch `suite-inst`,
[`docs/audits/instrumented_suite.md`](docs/audits/instrumented_suite.md) section 2). The
`instrumented` column is the **admissible three-instrument list**: `sided_turn_afferent:k=0.5`, the
`ring_dc_hold` ExR6 / ER6 / ER4m hold on PEN / EPG, and the `glno_sign` GLNO-as-glutamate relabel.
No row changes status; the one non-uniform row, `taste.MN9_hz`, is `raw`'s own instability (1.690 Hz
in draw 1) and is bit-identical between the presets at every seed. Values move on 20 of the 29 rows;
8 rows are bit-identical to `raw`. Per-row values, criteria and the frozen rule are in the audit.
`scripts/benchmark.py --sections all` runs every probe behind these rows (`probe_motion.py`,
`probe_loom.py`, `probe_taste.py`, `probe_bitter.py`, `probe_wind.py`, `probe_smell.py`,
`screen_rotation.py`, `screen_odour.py`, `screen_dns.py`) and writes `out/benchmark_suite.json`.*

## Where the model stands

Three deficits are localized to a named population and a named link rather than tuned away. **The
fly walks straight**: DNa02 is held below a 7.0 mV gap by -1.6 / -2.0 mV of sign-correct steady
inhibition while PFL3 and AOTU015 sit at 0.000 Hz and every `vnc_sensory` proprioceptor is at 0 Hz,
so the walking VNC runs open-loop ([`deficit_turning.md`](docs/audits/deficit_turning.md)); the
opt-in leg-cycle module closes that loop and raises yaw SD to 7.7-7.9 deg/s, but no clean frame in
any arm exceeds 100 deg/s -- it moves yaw, it does not turn, and nothing was adopted
([`body_sided_state.md`](docs/audits/body_sided_state.md)). **Small objects are lost at LC11 /
LC10a's inputs**: on the matched sphere ladder no size preference is called for either type (LC11
12/12 `null`, smallest `p_holm` 1.000; LC10a's one `result` fails Holm at `p_holm` 0.104 --
[`object_export_r2.md`](docs/audits/object_export_r2.md),
[`object_baseline_r2.md`](docs/audits/object_baseline_r2.md)) and no mechanism passes among eight
fixed-anatomy physiological arms ([`object_compare_r2.md`](docs/audits/object_compare_r2.md),
[`deficit_object.md`](docs/audits/deficit_object.md)); the assay itself is built and smoke-tested in
[`object_matched_assay.md`](docs/audits/object_matched_assay.md), which claims no result about
either type. **The compass has no rotation input**: at 90 deg/s the bump moves 0.00 +/- 0.01
wedges/s against 4.0 ideal ([`deficit_rotation.md`](docs/audits/deficit_rotation.md)). Behind all
three sit physiological facts nobody has measured -- each the reason a behaviour is a gap rather
than a bug:

1. **The GLNO transmitter.** GLNO's 4 cells are `nt` `unknown` in every MaleCNS column, so their
   17,698 output synapses are stored as explicit zeros -- 16,371 of them onto PEN, **19.4 % of PEN's
   raw input**, every edge above the connection cap. Three EM classifiers agree only that GLNO is
   inhibitory and disagree on the transmitter (MaleCNS T-bar consensus `unclear` at confidence 0.48,
   glutamate 51 % / acetylcholine 37 %; BANC v888 glutamate 4/4 at ~0.5; FlyWire FAFB GABA 3/4 at
   ~0.3), no transcriptome profile covers it and the receptor table has no row for it
   ([`glno_relabel.md`](docs/audits/glno_relabel.md) 1.1-1.2, [`cx_glno.md`](docs/audits/cx_glno.md) 1).
2. **Receptors at the EB / GA contacts.** ExR6's glutamate and ER6's GABA identities have
   type-specific EASI-FISH support (Wolff et al. 2025, eLife, Figure 9: ExR6 vGlut strong, ER6 Gad1
   weak), so the transmitters are settled. What no source gives is the **strength, kinetics and
   receptor localization of their particular contacts onto EPG and the two PEN types** -- exactly what
   the model's large, fast, negative DC term would need
   ([`exr6_evidence.md`](docs/audits/exr6_evidence.md) 1, [`compass_dc_balance.md`](docs/audits/compass_dc_balance.md) 5).
3. **Neuromodulation.** 3,312 bodies carry sign 0 (2,361 `unknown`, 415 serotonin, 395 dopamine, 141
   octopamine, across 530 types; 2,683 of them presynaptic, 2,701,289 of 124,161,873 synapses), so
   dopamine, octopamine and serotonin have no route. One slow receptor class cannot carry all three
   on one scalar: at the value that leaves behaviour untouched it fails the taste check on CPU, at ten
   times that the mushroom body runs away in half a second ([`nt_audit.md`](docs/audits/nt_audit.md),
   [`monoamine_slow_term.md`](docs/audits/monoamine_slow_term.md), [`flywire_banc_survey.md`](docs/audits/flywire_banc_survey.md) 4).
4. **Intrinsic and baseline activity.** No per-cell-type resting drive is measured anywhere in this
   model, and the compass is where that shows: at the shipped gains the ring forms no bump at all --
   `compass.EPG.bump_survival_s` is 0.00 s in 16 of 16 runs across the default arm and all three
   per-transmitter unitary brackets, because a transmitter scale multiplies the tuned Delta7
   inhibition and the untuned ring feedback by one factor and cannot set their ratio. The compass in
   the animal rides on tonic drive (ExR, PEN and others) that nothing here pins
   ([`unitary_strength.md`](docs/audits/unitary_strength.md) 1, `docs/NOTES.md` session 8).
5. **PS196_b has never been recorded.** Wang 2026 finding 3 makes PS196_b GLNO's largest non-ring
   input (1,801 synapses, reproduced to the synapse in this cache), and in the shipped body the report
   that reaches it is unsigned. A search on 2026-09-15 over Wang's audit, Hulse 2021 and the two
   Rockefeller theses found no recording of PS196_b or AN07B037 during turning, so `sided_turn_afferent`'s
   gain is a declared level (the 1,801 synapses / 19.2 % of GLNO's input are
   [`cx_shift.md`](docs/audits/cx_shift.md) 1 and [`deficit_rotation.md`](docs/audits/deficit_rotation.md);
   the Wang attribution and the search are [`docs/INSTRUMENTS.md`](docs/INSTRUMENTS.md),
   [`vnc_drive.md`](docs/audits/vnc_drive.md) 6).
6. **What the antennal contrast is worth to the model's own ORNs.** At the shipped ORN law, 2,639
   cells, the 100 ms rate filter and the 250 ms contrast filter, a 0.57 % physical contrast at the
   historical median total concentration arrives as 0.022-0.056 Hz against 0.224-0.273 Hz of 250 ms
   counting-window noise in each of four odour-mixture scenarios -- cascade SNR 0.16-0.34, below 1.
   These are sensory-field evaluations, not neural room runs
   ([`plume_transduced.md`](docs/audits/plume_transduced.md) 3).

What each of these predicts, and which measurement would settle it, is written up separately in
[`docs/PREDICTIONS.md`](docs/PREDICTIONS.md).

## The raw / instrumented layer

**What `raw` already contains that is not the connectome.** Shiu et al.'s LIF with one 0.275 mV
synapse per contact reproduces sugar -> proboscis but, on the whole CNS with sensory input, has two
regimes: silence or seizure. What was added, and which of it is a **stop-gap** (something standing in
for a measurement nobody has made) rather than a mechanism:

| addition | value | mechanism or stop-gap |
|---|---|---|
| spike-frequency adaptation | 1.5 mV/spike, tau 200 ms | mechanism (real, uncalibrated magnitude): without it KCs / PEN / PAM lock at 300 Hz |
| per-connection saturation cap | 60 synapse-equivalents | mechanism: PSP amplitude does not grow linearly with synapse number |
| fan-in normalisation | above 5,000 input synapses, unitary x `(5000/total)` | mechanism: large neurons have low input resistance |
| same-type synapse damping | x0.1 | mechanism: such populations are typically gap-junction coupled in the animal |
| short-term depression | antennal lobe only (ORN and LN terminals, u 0.2) | mechanism where documented (Kazama & Wilson 2008); **globally it blocked every descending command**, so it is off elsewhere |
| graded rate optic lobe up to T4/T5 | `optic.py` | mechanism: these cells are graded in the animal; a spiking lamina / medulla does not propagate at all |
| descending -> VNC x3, visual projection -> descending x2 | `DEFAULT_PATH_GAIN` | **stop-gap** for per-cell-type synaptic strengths: the difference between motor commands that reach the legs and ones that do not |
| LC4 / LPLC2 -> DNp01 x3 | `DEFAULT_TYPE_PATH_GAIN` | mechanism (Ache et al. 2019), magnitude a stop-gap: the loom escape margin |
| "T5 delayed inhibition" x5 | `DEFAULT_PAIR_GAIN` | **stop-gap, mislabelled until session 10**: 78 % of the 101,619 edges it multiplies are cholinergic Tm9 / Tm4, so it is mostly a **drive** gain ([`optic_measures.md`](docs/audits/optic_measures.md)) |
| receptor-corrected synapse signs | `receptor_model='sign'`, 48,295 of 25.6 M entries | mechanism from data (six transcriptomic sources); `None` restores the presynaptic rule byte-for-byte |

Every field and value, with the opt-in modules that are **not** part of the default:
`docs/REPRODUCIBILITY.md` section 1. Two things measured and **not** added: the monoamine slow term,
and a per-transmitter unitary strength that breaks the take-off motor at every data-anchored bracket
([`monoamine_slow_term.md`](docs/audits/monoamine_slow_term.md),
[`unitary_strength.md`](docs/audits/unitary_strength.md)). No default moved for either.

**What an instrument is** ([`docs/PRESETS_SPEC.md`](docs/PRESETS_SPEC.md) sections 1-2, 5). It stands
in for a piece of physiology the connectome names but the model cannot supply, and is admitted only
with all of: a **named gap** pointing at the audit that established it; a **source for its law**,
quoted at page / figure level, or the word `unverified` in its docstring and in `describe()`; the
**neural boundary only** (it reads `rate_hz` / `drive_mv` and writes `poisson_hz` / `drive_mv`, never
a weight, a receptor, `NT_SIGN` or the body); **its own suite run** beside `raw`; and a **removal
condition**. Program-shaped stand-ins -- those that replace a *computation* rather than supply a missing
*input* -- are admitted **only** under the owner's section-5 extension, and every instrument must declare
what it replaces in a `replaces` field that `flyverse.instruments._check_instrument` enforces.

| instrument | kind | replaces | law | admission | gap it stands in for | retired by |
|---|---|---|---|---|---|---|
| `compass` | stop-gap | computation | **unverified** (integrated realized yaw; 50 Hz peak, 35 deg Gaussian) | **rejected** | imposed angular memory and a continuous EPG Poisson representation | a native circuit that maintains and integrates a heading bump under the same turn/reversal/dark tests |
| `compass_ring` | stop-gap | computation | **unverified**; EB feedback a declared textbook counterfactual, angular gain uncalibrated | untested by the gate | the same memory, as Wang's reduced recurrent EPG/PEN rate loop | a validated native heading circuit or a quantitatively validated connectome-fitted model |
| `sided_turn_afferent` | stop-gap | input | **unverified**: no PS196_b / AN07B037 recording exists; `k` a declared level, `sign` the control | admissible at `k=0.5` | PS196_b's signed turn input (unknown 5 above) | a recording of PS196_b / AN07B037 during turning; a sided ascending report that reaches PS196_b on its own; or a round-7 null on measure 3 at every declared k |
| `ring_dc_hold` | edges | configuration | a counterfactual: no transfer claimed | admissible | the ExR6 / ER6 / ER4m DC term on PEN / EPG (1,149 entries, 37,256 synapses) held at 0 | sourced receptor placement and kinetics at the EB / GA contacts (unknown 2) |
| `glno_sign` | relabel | configuration | **unverified**: the transmitter two of the three low-confidence EM classifiers call | admissible | GLNO's sign-0 output (unknown 1) | a transmitter source for GLNO that is not one EM classifier |
| `plume` | stop-gap | computation | **unverified**: goal policy, output bridge, 1 mm baseline, 0.1 m response length and feedback gain all declared engineering assumptions | untested by the gate | bilateral odour-gradient walking goal, PFL3 comparator with DNa02 feedback | native odor/wind goal memory and PFL3 steering pass the same sensory controls |
| `hunger` | stop-gap | computation | **unverified**: explicit engineering modulation, not a hormone or receptor model | untested by the gate | `(1-energy)*(not sated)` navigation gain | a sourced metabolic transducer with verified target cells and dynamics |
| `flight` | stop-gap | computation | **unverified**: timers, reserve / odour hysteresis and the 100 Hz target come from the body equations, not physiology | untested by the gate | bounded wing-power bouts and a PFL3-to-wing steering bridge | validated descending/VNC flight-state and steering mechanisms supply the same control |

Every cell in the last column is that instrument's own declared `removal` -- the field
`flyverse.instruments._check_instrument` refuses to leave empty, printed by `describe()`. The
admission column is the section-2 item-5 gate and nothing else: `compass` was **rejected** on a
`taste.MN9_hz` status change ([`compass_standin.md`](docs/audits/compass_standin.md)), and the other
four navigation stand-ins have never been through it, so *untested by the gate* is neither a pass nor
a failure. [`docs/INSTRUMENTS.md`](docs/INSTRUMENTS.md) carries the **Composition** table -- `compass`
with `compass_ring` is an error (two heading providers), `plume` needs `compass` or `compass_ring`,
`hunger` needs `plume` or `flight`, and no two instruments write the same cell and channel, each
checked at CLI parse time and again before installation -- and a block per instrument with the rest
of each one's limits: `plume` is an experimental control, given no food-location oracle and claiming
no native source-localization circuit; `flight` selects no landing site, controls no altitude and
adds no flight-energy model. None of those four is in the ledger's `instrumented` column.
`ring_dc_hold_pen`,
`glno_pen_hold` and the `plume` variants (`plume:bilateral=orn`, `plume:feedback=0`,
`plume:walking_goal=0`) are diagnostic arms of the same rows.

**The admissible list.** `sided_turn_afferent:k=0.5` + `ring_dc_hold` + `glno_sign` is admissible
under PRESETS_SPEC section 2 item 5: no row changes status in three instrumented draws beside three
raw draws, and the room rate-half passes at six seed-matched runs per arm (110 take-offs against 101
over 28,800 fly-s each; one-sided exact p 0.291; run-level `null`). Admissible is all it is -- nothing
is adopted, `raw` stays the default, the afferent's law is still `unverified` and the relabel still
has no transmitter source ([`instrumented_suite.md`](docs/audits/instrumented_suite.md) section 4).

**Two sentences the audits license.** The first is the statement the independent skeptic proposed for
this README, quoted in `docs/NOTES.md`, "Session 14, continued: the transduced-plume rooms":

> In six 60 s tabletop rooms per arm, the shipped `plume` instrument -- which reads the physical
> odour-concentration difference between the antennae -- fed in 6/6 rooms, while the opt-in
> `plume:bilateral=orn` variant, which substitutes the model's own ORN population rates at the same
> gain, fed in 1/6 and steered on a cue uncorrelated with the true lateral contrast; a descriptive
> room observation, not a significance test, an SNR measurement, or a claim about flies.

The second is carried by the three navigation audits
([`flight_foraging_priority.md`](docs/audits/flight_foraging_priority.md),
[`navigation_instruments.md`](docs/audits/navigation_instruments.md),
[`plume_steering.md`](docs/audits/plume_steering.md)):

> No run in these audits shows artificially powered flight and feeding in the same episode: the only
> powered-flight rooms are the superseded navigation flight arm (59.67 s airborne, zero feeding in all
> six rows), and `powered_s` is 0.00 in all 12 flight-priority room rows and all 12 plume room rows.

The rooms do **not** confirm the CPU SNR result and are not offered as confirming it; what they add
is that the measured in-room cue noise is 1.3-2.6x the analytical estimate and that the cue's sign
was right in only 0.52 +/- 0.10 of samples
([`plume_transduced.md`](docs/audits/plume_transduced.md), skeptic correction C7). The six runs of
an arm differ in start *and* seed, so the across-run SD is start heterogeneity; the arms share their
six starts, and the paired-by-start reading is the stronger statement. A third arm,
`plume:feedback=off` (physical goal, DNa02 gain 0), fed in 3/6 -- exactly the starts already
pointing near fruit. Nothing adopted, no default moved, the gain of 200 still underived.

## The same model on three connectomes

`connectome.load(dataset="fafb" | "banc")` compiles FlyWire FAFB v783 and BANC v888 into their own
caches with `root_id -> bodyId`, alias-normalised type names and vocabulary maps; the receptor table
and the expectation ledger transfer by type name. MaleCNS identity is preserved exactly (the three
cache md5s, the CSR fingerprint `ef23cc27bea13be7f6a96f3c04fd3737` over 167,106 neurons / 25,578,600
entries, and the `test_bit_identity.py` golden are unchanged). 4,715 type names are shared with FAFB
(59 % of MaleCNS cells) and 7,464 with BANC (72 %), and raw input counts scale about 1 : 0.6 : 0.3
across the three releases, so nothing is compared unscaled. BANC's optic lobes are under-proofread
(T2 853 vs FAFB 1,466 vs MaleCNS 1,630); FAFB has the complete optic lobe, BANC has the VNC
([`flywire_banc_survey.md`](docs/audits/flywire_banc_survey.md) sections 1-3).

`scripts/cross_connectome.py` generates the cross-connectome table -- 491 side-resolved edge rows,
100 NT budgets and 300 ranked input rows, distinguishing a missing population from a present one with
no retained edge ([`connectome_backends.md`](docs/audits/connectome_backends.md)). Descriptively, on
the walking replicate the female CNS walks straighter than the male at the shipped defaults (BANC
clean yaw SD 0.275 +/- 0.004 deg/s against MaleCNS 2.641 +/- 0.146; straightness 0.9984 against
0.9943), the leg-cycle increase replicates within-dataset as a `result` while the neural pattern does
not, and BANC's DNa02_L does not fire while R reaches 0.0985 Hz. **This is a descriptive
cross-connectome comparison, not a sex test**: one male reconstruction against one female one, with
different coverage, different naming and no per-individual replication; a missing type match is not
established sex specificity.

## When a behaviour fails: the toolkit, and the process

`flyverse/interp` is eight tools that localize a deficit on the shipped weights and never edit them:
`decompose` (what drives a cell set, by presynaptic type / transmitter / receptor tier), `trace`
(where a stimulus is lost along the depth from a sensory population), `paths` (effective k-step
signed gains A -> B, and which links carry nothing), `lesion` (check x lesion delta matrices),
`atlas` (what every motor readout does when population X is stimulated), `health`, `ledger`
(measured responses against `flyverse/data/expected_responses.csv`) and `export`.

**[`docs/INTERP.md` section 10](docs/INTERP.md)** is the seven-step procedure to run when a ledger row
or a benchmark check fails. The first two steps cost minutes and no GPU:

```
python scripts/interp_ledger.py --results "out/interp/*/*.json" out/benchmark_suite.json --status "FAIL,KNOWN GAP" --json out/interp/ledger/fails.json
python scripts/interp_paths.py --a "class=olfactory" --b "DNa02" --k 3 --json out/interp/paths/orn_dna02.json
python scripts/interp_decompose.py analyse --static --target DNa02 --json out/interp/decompose/dna02_static.json
```

That is where GLNO -> PEN (sign 0) and the three lateralised routes into DNa02 that carry nothing were
found, before any GPU job ran. The toolkit's output is a **diagnosis**, never a fix: a diagnosis
licenses a measurement, a mechanism the connectome data imply, or a documented swappable module -- and
a gain, a bias, a threshold or a sign the data cannot see is hand-crafting, named as such.

The process rules are `docs/INTERP.md` section 10.4, and they are why the numbers above read the way
they do. Every batch is **predeclared**: protocol, family, gate and source hashes frozen in an
`out/<batch>/predeclared.json` before submission, not overwritable after. The verdict vocabulary is
fixed and belongs to one function (`common.compare`): **`result`** (|z| >= z_min and p <= alpha),
**`null`**, **`underpowered`** (fewer than four runs in the smaller arm, or a rank-test floor above
alpha) and **`undetermined`** (a deterministic null arm, where z is undefined and only the magnitude
and p can be read). Replicates are cluster jobs, not seeds and not batch rows. Every round is then
re-checked by an **independent skeptic pass whose verdict line and claims are quoted verbatim**, with
corrections applied in place and superseded sentences kept marked **Withdrawn** -- the record is
[`docs/audits/receptor_verification.md`](docs/audits/receptor_verification.md). How a round is run
end to end is written up separately in [`docs/PROCESS.md`](docs/PROCESS.md).

## Reproducibility

**[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** records the shipped defaults field by field,
the cache fingerprint (`sum|W|` 121,460,584, compiled-W md5 `ef23cc27bea13be7f6a96f3c04fd3737`), the
dataset releases and hashes, the commits, and which number here was measured at which one.

**The determinism gate.** Each protocol was run twice on one pinned GPU and every saved array and
metric compared with no tolerance ([`determinism_gate.md`](docs/audits/determinism_gate.md)). The
`cx_wedge` protocol -- B=1, no world, no optic lobe, a fixed Poisson programme on the torch path --
**repeats exactly** (31/31 arrays, 546/546 metrics; the third such exact repeat). **No B=6 room
repeats on either GPU path**: the native path diverges within the first 4 (raw) / 30 (plume) frames,
the torch path at 86 / 60, and the first difference is always +/-1 spike in one row, so neither the
body integrator nor the odour field is the source (the native event-scatter kernel's float
`atomicAdd` is a sufficient mechanism there; which stage breaks the torch path is the gate's open
question). The frozen consequence: **rooms run at >= 6 draws with the run as the replicate unit and
are quoted as a mean +/- across-run SD, never beyond it**; bit-identity claims are CPU claims only
(`tests/test_bit_identity.py`, a golden over a deterministic synthetic graph), and nothing here
claims a bit-identical GPU rollout.

## Where things are

The results, arm by arm, with the batch that produced each one:
[`docs/RESULTS.md`](docs/RESULTS.md). One audit record per question, every number carrying its batch
line: [`docs/audits/`](docs/audits/). Session by session: [`docs/NOTES.md`](docs/NOTES.md). The full
file-by-file map: `docs/OVERVIEW.md`. In short:

| | |
|---|---|
| `flyverse/connectome.py`, `flyverse/backends/` | compile a release into a signed CSR cache (MaleCNS, FAFB, BANC) |
| `flyverse/brain.py`, `flyverse/optic.py`, `flyverse/retina.py`, `flyverse/world.py` | the LIF CNS, the rate optic lobe, the eyes, the ray-traced room |
| `flyverse/fly.py`, `flyverse/motor.py`, `flyverse/senses.py`, `flyverse/body.py` | the control surface: senses in, `MotorRates` out, motion |
| `flyverse/instruments.py`, `flyverse/navigation.py`, `flyverse/compass.py` | the `instrumented` preset's stand-ins and their `describe()` records |
| `flyverse/programs.py`, `flyverse/cx.py`, `flyverse/interp/` | opt-in behaviour programs, the compass-steering module, the eight interpretability tools |
| `flyverse/batch_sim.py`, `flyverse/env.py` | batched room rollouts; the vectorised RL environment |
| `scripts/` | `room_demo.py`, `fetch_data.py`, `benchmark.py`, `probe_*.py`, `screen_*.py`, `interp_*.py`, `cross_connectome.py` |
| `docs/INSTALL.md`, `docs/OVERVIEW.md`, `docs/REPRODUCIBILITY.md` | install, the long form, what was measured where |
| `docs/PRESETS_SPEC.md`, `docs/INSTRUMENTS.md`, `docs/BENCHMARK_BATTERY.md`, `docs/INTERP.md` | the preset contract, the instrument inventory, the assays, the toolkit contract |
| [`docs/RESULTS.md`](docs/RESULTS.md), [`docs/PREDICTIONS.md`](docs/PREDICTIONS.md) | what every round asked and what came back, arm by arm; the falsifiable predictions and the assay that would refute each |
| [`docs/PROCESS.md`](docs/PROCESS.md), [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) | how a round is run end to end; what is owed before the repository is public |

## Licence, citation and data

Code: **MIT** (`LICENSE`). Cite flyverse with `CITATION.cff`, and cite the MaleCNS v1.0 release and
Shiu et al. 2024 as well -- flyverse is a simulator built on their data and their neuron model. If you
use the female backends, cite the FlyWire FAFB v783 and/or BANC v888 releases and their reconstruction
papers. The repository ships **no connectome data and no third-party tables**;
`flyverse/data/manifest.json` records the URL, SHA-256, citation and licence of every file
`scripts/fetch_data.py` can fetch:

* **MaleCNS v1.0** -- Janelia FlyEM, <https://male-cns.janelia.org>, CC BY 4.0 at the time of writing
  (check the download page); the release paper is recorded in the manifest as Cell 2026, PII
  S0092-8674(26)00942-6, which has not been checked against a registered DOI, so `CITATION.cff`
  claims no `doi:` field for it.
* **FlyWire FAFB v783** and **BANC v888** -- Codex public releases, CC BY 4.0; cite the releases and
  the original reconstruction papers. FlyWire NT predictions are used where the model reads them.
* **External expression tables** (Ozel 2021, Davis 2020, Kurmangaliyev 2020, Nern 2025, Fly Cell
  Atlas / Davie 2018, the typing tables) -- **not redistributed**; `data/external/` is git-ignored.
  Cite them from the manifest, do not re-host them.

Inspirations: stonkfly, doomfly (nftechie), fly-escape (dzhng).
