# flyverse: the long form

Everything the `README.md` one-pager points at but does not have room for: the demo's controls and
flags, the speed numbers and backends, the batched simulator, the full account of what was added to
the point neuron model and why, the behaviour programs and what is and is not the brain's, the RL
environment, and the file-by-file repository map.

Read `README.md` first. Reproducibility -- the shipped defaults, the cache fingerprint, which number
was measured at which commit -- is `docs/REPRODUCIBILITY.md`.

## 1. The room demo: controls, flags, speed

### Keys

`SPACE` pause, `R` reset, `T` teleport to the apple (taste), `L` loom a black ball at the fly
(escape jump), `F` fire the giant fibre, `W` stimulate the flight DNs (DNg02_a / DNa08: a 2-3 s
powered flight); arrows / mouse drag orbit the scene camera, `+`/`-` or wheel zoom, `C` follow the
fly, `HOME` reset; `F5`/`F9` quick-save / quick-load, `S` timestamped save, `ESC` quit. `?` opens the
controls guide; `1`-`4` select inspector views and `V` cycles retinal both / colour / contrast. `N`
toggles the soma map to live NT levels. Save states are ~10 MB and resume bit-exactly.

### Flags

`--brain-map` (all 140k located somata in dorsal and lateral view, activity as highlights;
`--map-every N`, `--map-no-blur`), `--brain-map-mode nt` (open the map on live NT levels; this
requires an optional readout source, not NT labels -- see `docs/NT_READOUT.md`), `--trail-seconds`
(decaying trail in the scene view), `--start x,y[,z]` | `floor`, `--wind-speed`, `--wind-dir`,
`--load state.pt`, `--window WxH` (the window is resizable, the UI reflows above 1100x720 and scales
below it), `--fast` (preset for Macs / slower GPUs; `--brain-dt`, `--optic-dt`, `--cam-scale`),
`--screenshot out/console.png` (saves the final UI), `--gif out/room.gif` with `--headless
--seconds N`. UI controls and display details: `docs/ROOM_UI.md`.

### Speed

Eager, one fly on an RTX 4090, the demo runs at 0.5x real time (19.8 ms per 10 ms frame; the LIF
matrix no longer carries the optic lobe's synapses -- exact, 24.7 M -> 13.0 M nnz). `--cuda-graphs`
(capture and replay each neural frame and the sensory ray tracing) gives 0.8x; `--dt-by-module
vnc=1.0` (integrate the VNC at 1 ms, the brain at 0.5) and `--weight-dtype float16` bring it to real
time (10.4 ms); `--fast --cuda-graphs` is 1.3x. Graphs match the eager spike train for ~130 frames
and then diverge by a spike (chaos); fp16 and the coarser clock change spikes from the start, so the
probes run eager. Numbers and behavioural checks in `docs/NOTES.md` and `docs/PERFORMANCE.md`.

On an M-series Mac the brain and the ray tracer run on custom Metal kernels (`flyverse/metal.py`):
0.8x real time at full fidelity, 1.2x with `--fast`. Batched brains (`Brain(c, batch=64)`) run 64
flies at 1.8 ms per fly-frame -- one sparse matmul serves them all.

Everything also runs on CPU, slowly; the brain is 72k spiking neurons behind an 89k-unit graded optic
lobe and wants a GPU for anything real-time.

### Backends and data location

Torch backend: CUDA, else Apple MPS, else CPU (`FLYVERSE_DEVICE`). On MPS/CPU the synaptic input is
an event-driven gather instead of the sparse matmul (`flyverse/device.py`). On MPS the gather, the
LIF update, the optic lobe's sparse products and the ray tracer are hand-written Metal kernels
compiled at first use through `torch.mps.compile_shader` (`flyverse/metal.py`; `FLYVERSE_METAL=0` for
plain torch). Spike trains match the torch path exactly; continuous state and radiance agree to
~1e-4.

MaleCNS data defaults to `D:\Datasets\male-cns-connectome-v1.0\flat-connectome` (override with
`FLYVERSE_DATA`); only `body-annotations`, `body-neurotransmitters` and `connectome-weights` are used
(1.1 GB; `tbar-neurotransmitters`, 2.7 GB, only by the NT audit). None of it ships with the repo.
Female releases live under `FLYVERSE_DATA_FAFB` / `FLYVERSE_DATA_BANC` and compile into `cache/fafb/`
and `cache/banc/`. Full install and fetch instructions: `docs/INSTALL.md`.

## 2. Batched room simulations

For headless sweeps of the full room model, `BatchSim` runs independent environments through one
batched brain, including sensing, surface walking, flight and metabolism:

```powershell
python scripts/batch_sustain.py --batch 8 --minutes 5 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch --json out/batch.json
```

(A GPU command; `--cuda-sparse warp` is batch-1 only, batched runs need `torch`.) See
`docs/BATCH_SIM.md` for per-row configuration, checkpoints, random-stream semantics and profiling.
The interactive demo and the RL environment keep their existing APIs.

## 3. What is simulated, stage by stage

| stage | neurons | model | file |
|---|---|---|---|
| photoreceptors R1-R6, R7 (UV), R8 (blue / green) | 5,895 placed on 1,466 real hex columns of the two eyes (of 6,098 `ol_sensory`) | ray-traced spectral radiance [UV,B,G,R] per ommatidium, low-pass + contrast adaptation | `retina.py`, `world.py` |
| optic lobe (lamina, medulla, lobula, lobula plate) | 89,390 `ol_intrinsic` | graded rate units on the signed, input-normalised connectome (flyvis-style); T4/T5 rectify with strong delayed inhibition and are direction selective with the correct preferred direction for all eight subtypes | `optic.py` |
| everything else: visual projection neurons, central brain, VNC, motor neurons | 71,618 | Shiu et al. 2024 leaky integrate-and-fire on the GPU, plus adaptation, a per-connection saturation cap, a fan-in cap for giant neurons, same-type synapse damping and antennal-lobe-only depression; synapse signs from the presynaptic transmitter, corrected per postsynaptic type by the receptor-expression table | `brain.py` |
| taste | 165 labellar sugar GRNs, found by connectivity to the known sweet interneurons | Poisson while touching fruit -> Usnea / Rattle / Phantom / G2N-1 -> proboscis MN9 | `scripts/find_sweet_grns.py`, `flyverse/data/taste_grns.csv` |
| smell | 2,639 ORNs in 53 glomeruli, sided to the left / right antenna by their PN targets | every fruit is a wind-blown plume (Gaussian, puffing) sampled by two antennae 1 mm apart | `air.py` |
| wind | Johnston's organ C / E neurons, sided by their AMMC/WED targets | antennal deflection per side from the wind vector in the body frame | `air.py` |
| body | walking on any surface (table top, sides, underside, legs, walls, ceiling: the fly sticks to what it stands on, walks over edges and climbs corners), escape jumps, flight with pitch and roll that lands on the first surface it meets | named readout from measured groups (`motor.motor_groups`): DNp20 + HSN/HSE (optomotor -- DNp04 + LPT27/30 were used until session 8 and do not flip under imposed yaw), DNp18 / DNp33 (wind direction, gated by the odour signal), DNa02 (goal steering), MDN (back up), MN9 (proboscis); giant fibre -> escape jump; DLMn / DVMn -> wingbeat; wing steering MNs -> yaw and roll | `body.py`, `motor.py` |

Two details the one-line table cells cannot carry:

* **The T5 "delayed inhibition" pair gain is mostly a drive gain.** 78 % of the 101,619 edges it
  multiplies are cholinergic Tm9 / Tm4 input, so it is documented as a stop-gap rather than a
  mechanism (`docs/audits/optic_measures.md`; see section 4 below).
* **The receptor correction is 26 % of the weight.** Six transcriptomic sources decide it;
  `LIFParams.receptor_model = None` restores the presynaptic rule and the weights are then
  byte-identical to the model before the receptor block (`docs/NT_INTEGRATION.md`,
  `tests/test_receptor_model.py`).

The central-complex ring is silent at these defaults; no per-transmitter unitary bracket forms a bump
either; under experiment gains a bump persists with the senses on but does not track heading and does
not steer (`docs/audits/compass_room.md`, `cx_shift.md`), and `--program cx` food-finding fails in the
last few centimetres of the approach (`docs/audits/feeding_horizon.md`).

![retina](retina_map.png)

*The two eyes reconstructed from the connectome: 1,466 hex columns with R1-R6 (yellow), R7 (purple),
R8 (green) and dorsal-rim (red) photoreceptors placed by their strongest lamina/medulla partner.*

## 4. What we had to add to the point model, and why

Shiu et al.'s LIF with one 0.275 mV synapse per contact reproduces sugar -> proboscis but, on the
whole CNS with sensory input, it has two regimes: silent or epileptic. The notes document each
finding; the short version:

* neurons without a neurotransmitter prediction and monoamines are treated as non-additive (they were
  the first 300 Hz loops);
* giant connections are capped at 60 synapse-equivalents;
* same-type synapses (gap-junction-coupled populations) are damped;
* adaptation is 1.5 mV/spike;
* depression lives only in the antennal lobe, where it is documented -- globally it blocked every
  descending command;
* the optic lobe up to T4/T5 is graded in the animal and is a rate model here; a spiking
  lamina/medulla does not propagate at all;
* two pathway gains (descending -> VNC x3, visual projection -> descending x2) stand in for
  per-cell-type synaptic strengths; they were the difference between motor commands that reach the
  legs and ones that do not.

Every one of these is a stop-gap with a name and an audit. The exact fields and values are in
`docs/REPRODUCIBILITY.md` section 1.

What is *not* added is a fix for a behaviour the model fails: the remaining deficits are localized
rather than tuned -- named population, named link, and what kind of fact the loss is (missing input,
wiring, dynamics, operating point) with a data-driven next measurement and an explicit list of the
hand-crafting that was not done (`docs/audits/deficit_{turning,object,rotation}.md`). The procedure to
run when a behaviour fails is `docs/INTERP.md` section 10; the tools it uses never edit the model.

### What the body does not send back, and the module that tests it

Every `vnc_sensory` and `sensory_ascending` proprioceptor in MaleCNS is wired into this model and, in
the plain fly, sits at 0 Hz: the walking VNC runs open-loop. `senses.Proprioception` (opt-in, **off by
default**) closes that loop the way `Wind` feeds Johnston's organ -- leg chordotonal / hair-plate /
campaniform afferents from the leg motor-neuron rates and ground contact, haltere afferents from the
haltere motor-neuron rate, every rate law inside a published range recorded as an unscored ledger row.
Turning it on makes the wiring carry: the ascending cells fire, the efference-copy chain into the
compass fires for the first time, and a sided excitatory term appears on the steering neuron DNa02.

With a stance / swing leg cycle on the body (`body.LegCycle`, tripod timing from the realised speed,
DeAngelis 2019 / Mendes 2013) and the haltere motor readout split by side, the leg afferents fire at
literature-typical rates, DNa02 fires and the fly meanders (clean-frame yaw SD 2.7 -> 7.8 deg/s,
straightness 0.995 -> 0.85) -- but it does not turn the way the animal does: no frame above 100 deg/s,
a fixed left drift that is the connectome's own asymmetry, and a sided afferent report that follows
the turn instead of leading it. The compass still receives no signed report of a self-turn: it exists
at the ascending neurons and is diluted to nothing two synapses on (`docs/audits/body_sided_state.md`,
`round3_integration.md`).

All of it ships as a **module, not a default**: with the sense on, the suite's `rest` check fails by
construction -- a fly standing still has firing proprioceptors, and that check is defined as "no
input". `docs/audits/proprioception_transducer.md`, `docs/audits/vnc_drive.md`,
`docs/audits/body_sided_state.md`.

### Two things we measured and did not add

The monoamine synapses (2.7 M, sign 0) can be routed through a slow receptor class, but one scalar
cannot carry dopamine, octopamine and serotonin at once -- at the value that leaves behaviour
untouched it fails the taste check on CPU, at ten times that the mushroom body runs away in half a
second (`docs/audits/monoamine_slow_term.md`). And a per-transmitter unitary strength (the one
0.275 mV is a cholinergic calibration; the inhibitory ratio is an argument, not a measurement) breaks
the take-off motor at every data-anchored bracket (`docs/audits/unitary_strength.md`). Nothing from
either was adopted and no default moved.

## 5. Food-finding, and what is the brain's and what is not

The default model is the connectome plus a **plain body**: rates in, motion out (forward DNs ->
speed, DNa02 -> turning, the optomotor DNs -> stabilisation, MN9 -> proboscis, the giant fibre ->
jump, wing motor neurons -> flight), contact at table edges, an energy state. Nothing in that path
decides behaviour; what the fly does is what the wiring does.

Foraging by odour-gated anemotaxis -- surge upwind on an odour hit, cast when it is lost, search when
hungry -- is a **behaviour program** (`flyverse/programs.py`), an explicit, swappable stand-in for the
lateral-accessory-lobe / central-complex steering circuit, off by default and enabled with `--program
anemotaxis` (`--escape-gating` adds habituation of the escape). It reads only brain signals the model
carries -- the wind-direction DNs (DNp18 / DNp33) and the odour signal of six lateral-horn types
(`motor.LH_ODOUR_TYPES`, found by a screen: ~23 Hz next to fruit, ~5 Hz away, whichever way the fly
faces) -- and supplies the integration the model does not (yet) produce.

`scripts/probe_sustain.py` measures the fly for minutes with and without it. Where things stand (5
min, three seeds): on the full table every configuration loses the table to an escape hop within
minutes; on a fenced table with a single apple 40 cm upwind the plain model never finds it, the body
program gets to 6 cm, the CX module feeds in 2 of 6 runs, and CX + `KlinotaxisProgram` (a sensor-side
near-field subsystem, `--program cx+klinotaxis`) in 2 of 3. Read those as descriptions of single runs,
not as a ranking: meals are too rare to score anything at this exposure (0.13-0.50 per fly at every
horizon and metabolic drain tested, and identical to plain arrivals), so they cannot separate two
models (`docs/audits/feeding_horizon.md`).

The principled replacement for a program is a **module that plugs into the brain**: the screens showed
the model's central complex is silent -- no compass, no goal comparison -- while its inputs (wind DNs,
the LH odour signal) and its output (PFL3 -> DNa02 -> legs) work. `flyverse/cx.py` `CompassSteering`
fills exactly that hole: it reads the brain's signals and drives PFL3 by stimulation; the connectome
turns the fly (`--program cx`). Nothing is injected into the body.

The tooling for finding where such functions live is `flyverse/screen.py`: record every cell type
under contrasting conditions, rank by separability with heading / wind controls, ablate or stimulate a
candidate to test it causally. `scripts/screen_odour.py` reproduces the lateral-horn result;
`scripts/screen_steering.py` asks whether any population already carries the odour x wind-side
interaction the program computes.

## 6. RL on top of the brain

`flyverse/env.py` `FlyRoomEnv(batch=B)` is a vectorised environment (PufferLib convention; PufferLib
itself does not build on Windows/py3.13): observations are brain activity (descending-neuron rates, or
the per-glomerulus odour code), actions are walking commands, reward is progress to fruit + tasting.
`scripts/train_decoder.py` runs evolution strategies over a linear decoder. Results so far are
negative and documented: a hand-written klinotaxis policy on the odour code beats chance (93 vs 47
tasted frames), ES did not find it in 40 generations.

## 7. Repository map

```
flyverse/connectome.py   load + compile a release -> signed CSR (cached in cache/)
flyverse/backends/       per-release readers: malecns.py, fafb.py, banc.py, common.py
flyverse/retina.py       photoreceptor -> hex column -> viewing direction; spectral sensitivities
flyverse/world.py        torch ray tracer: room, table, fruit, 4-channel light, fly-scale textures
flyverse/optic.py        graded optic lobe + photoreceptor contrast stage + interface to the LIF
flyverse/brain.py        whole-CNS LIF (torch sparse, batched)
flyverse/metal.py        Metal kernels for the MPS backend: event scatter, fused LIF update, CSR spmv, optic substep, ray tracer
flyverse/cuda.py         nvcc-compiled CUDA kernels (opt-in)
flyverse/air.py          wind, plumes, bilateral olfaction, Johnston's organ wind sense
flyverse/regions.py      named brain modules, Connectome subsets, path-based paring
flyverse/fly.py          FlyBrain: the control surface (senses in, MotorRates out, step / budget / state)
flyverse/senses.py       vision / smell / wind / taste / proprioception encoders onto the sensory neurons
flyverse/motor.py        named readout groups and MotorRates (incl. the lateral-horn odour signal)
flyverse/modules.py      the attached-module protocol (fb.attach)
flyverse/async_brain.py  the brain on its own thread / CUDA stream for fixed-tick hosts
flyverse/body.py         fly pose (3-D), walking / flight / metabolism: MotorRates -> motion, nothing decided
flyverse/surfaces.py     walkable faces: edges, corners, landings
flyverse/programs.py     hand-designed behaviour programs (anemotaxis, escape gating), opt-in stand-ins for circuits
flyverse/cx.py           CompassSteering: a simulated central-complex stage that drives PFL3 inside the brain
flyverse/screen.py       condition screens, ranking, ablation: which cells carry what
flyverse/batch_sim.py    BatchSim: B independent room rollouts around one batched brain
flyverse/env.py          vectorised RL environment
flyverse/brainmap.py     soma projections for the --brain-map panel
flyverse/room_ui.py      dense room console, controls and neural telemetry
flyverse/nt_readout.py   live neurotransmitter readout for the UI
flyverse/interp/         the interpretability toolkit: eight tools that localize a deficit on the shipped weights (never edit them)
  decompose.py           what drives a cell set, per frame, by presynaptic type / transmitter / receptor tier
  trace.py               where along the depth from a sensory population a stimulus is lost
  paths.py               effective k-step signed gains A -> B, and which links are silent (sign 0, never firing)
  lesion.py              check x lesion delta matrices with scatter, and the double dissociations
  atlas.py               what every motor readout does when population X is stimulated
  health.py              per-type operating point of a rollout: silent, at threshold, refractory-limited, E/I, fan-in
  ledger.py              measured per-type responses against the curated expectations (flyverse/data/expected_responses.csv)
  export.py              the Neurome read-only probe export (manifest, tables, SHA-256)
  common.py              shared by all eight: selection grammar, shaped-weight accessor, Result schema, provenance, null helpers

scripts/room_demo.py     the interactive demo
scripts/fetch_data.py    fetch and verify every file in flyverse/data/manifest.json
scripts/benchmark.py     the benchmark suite (14 sections, JSON for regression diffs);  scripts/probe_*.py  one behaviour each
scripts/cross_connectome.py   MaleCNS / FAFB / BANC anatomy side by side
scripts/check_connectome_backends.py   the female-backend acceptance gate
scripts/audit_nt.py, scripts/cx_wedge.py, scripts/retire_measures.py   the session-9 audits (reports in docs/audits/)
scripts/interp_*.py      one CLI per interpretability tool (record / run on the GPU, analyse on the CPU)
scripts/interp_apply_*.py   the three deficit applications: turning, object, rotation
scripts/cluster_run.py   run a batch of commands on a GPU cluster job manager (config in a git-ignored .cluster.json)
scripts/screen_dns.py    the DN activation screen;  scripts/find_sweet_grns.py  sugar GRNs
scripts/screen_odour.py  the lateral-horn odour screen;  scripts/screen_steering.py  odour x wind steering
scripts/hash_weights.py  the compiled-W md5 and sum|W| of this checkout's cache
scripts/profile_room.py  per-frame profile of the demo loop;  scripts/profile_brain.py  the brain alone

tests/                   control-surface, world and integration tests;  tests/test_interp.py  the toolkit's CPU suite
tests/test_bit_identity.py   the recorded golden: the default path is byte-for-byte what it was

docs/INSTALL.md             clean clone -> a fly walking on a table
docs/REPRODUCIBILITY.md     shipped defaults, cache fingerprint, which number at which commit
docs/NOTES.md               everything learned, session by session, with numbers
docs/ARCHITECTURE.md, docs/CONTROL_SURFACE.md, docs/PERFORMANCE.md   the control surface and its cost
docs/EXTENSIBILITY.md       hooks, attached modules, graph extension, trainable boundaries
docs/BATCH_SIM.md           batched room rollouts
docs/ROOM_UI.md             the room console
docs/BENCHMARK_BATTERY.md   behavioural assays and how the plain model scores on each
docs/INTERP.md              the interpretability toolkit: contract, the procedure to run when a behaviour fails (10), open defects (11)
docs/NT_INTEGRATION.md      receptor-expression data integration: findings, sources, task outline
docs/NT_READOUT.md          the optional live neurotransmitter readout
docs/CONNECTOME_BACKENDS_SPEC.md   the FAFB / BANC backend specification
docs/NEUROME_INTERFACE.md   the read-only probe-export interchange
docs/audits/interp_*.md     one validation record per tool;  docs/audits/deficit_*.md  the turning / object / rotation localizations
docs/audits/connectome_backends.md   the cross-connectome result
docs/audits/receptor_integration.md, docs/audits/receptor_verification.md   the receptor integration's scoring record and the skeptics' verdicts (rounds 1-5)
```
