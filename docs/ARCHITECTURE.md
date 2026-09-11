# Architecture: the fly brain as a control system

Goal: the connectome model should plug into anything -- the picnic-table demo, an RL environment,
a game with a fixed update tick, an interpretability notebook -- through one small surface, and it
should be possible to run only the parts of the brain a simulation needs. This document describes
the implementation; `docs/NOTES.md` has the measurements that motivated it. See
`docs/CONTROL_SURFACE.md` for API examples, timing and validation details.

## The three layers

```
 environment side                 brain side                          body side
 ---------------                  ----------                          ---------
 world.py   ray tracer            fly.FlyBrain                        body.FlyState
 air.py     wind + plumes           .vision(radiance)   -> optic.py    body.Locomotion.step(state, motor)
 (a game)   anything               .smell(cL, cR)      -> senses.py   body.Flight.step(state, motor)
                                   .taste(sugar)                      body.Metabolism
                                   .wind(dL, dR)
                                   .stimulate(sel, hz, ms)
                                   .step(ms)           -> brain.py (LIF on a Connectome subset)
                                   .motor()            -> MotorRates (named group rates)
```

* **Environment side** produces physical quantities at the fly's sensors: radiance per ommatidium
  column, odour concentration per glomerulus at each antenna, wind deflection per antenna, sugar
  contact. Nothing here knows a neuron exists.
* **Brain side** (`FlyBrain`) turns those into spikes and rates on the connectome and exposes named
  readouts (`MotorRates`: forward DNs, DNa02 L/R, optomotor L/R, wind-direction L/R, MDN, MN9, giant
  fibre, wing power/steering MNs, PN rates by glomerulus). Nothing here knows what a table is.
* **Body side** turns `MotorRates` into a pose: speed, yaw, pitch/roll, jumps, flight, feeding. This
  is where the behavioural assumptions live (the odour gate, casting, hunger scaling, edge
  behaviour) and they are all named parameters of `Locomotion` / `Flight` / `Metabolism`. A game can
  use these classes or write its own from the same `MotorRates`.

The **room demo** is then `World + Air + FlyBrain(all modules) + body.*` -- the full-fidelity
configuration -- and `env.FlyRoomEnv` is the batched version of the same wiring. Neither wires
sensory neurons by hand any more (both did before; that duplication was the coupling).

## Modules: running part of the brain

`regions.py` names the parts of the connectome and builds a **`Connectome` subset** (induced
subgraph with re-indexed neurons and weights; everything downstream -- `Brain`, `OpticLobe`,
selectors, motor groups -- works on any `Connectome`, so a subset needs no special casing):

| module | contents | needed for |
|---|---|---|
| `optic` | photoreceptors + 89k `ol_intrinsic` (rate model) | vision |
| `visual_projection` | LC / LPLC / LPT / MeTu / ... + visual centrifugal cells | vision -> central brain |
| `antennal_lobe` | ORNs, AL local neurons, PNs | smell |
| `mushroom_body` | KCs, MBONs, DANs, APL / DPM | learning-related readouts |
| `gustatory` | GRNs + sweet second-order interneurons + central-brain motor cells | taste and proboscis |
| `mechanosensory` | JO neurons, AMMC / WED interneurons | wind, sound |
| `central` | the rest of `cb_intrinsic` (LH, CX, LAL, AVLP, GNG, ...) and unclassified cells | everything |
| `descending` | DNs | any motor output |
| `vnc` | VNC intrinsic + motor + sensory + ascending | leg / wing motor neurons |

`FlyBrain(modules=[...])` composes them; a sense whose neurons are not in the subset is simply
absent (`fb.vision` raises; `available_senses` lists the remaining senses). Labels form an exhaustive,
disjoint partition; DNs stay in `descending` even when they participate in taste pathways.

Subsets preserve **effective weights**, not the activity of a full brain. The full neuron table stores
`in_syn` (sum of absolute signed synapse counts, excluding sign-zero modulation) and `in_syn_l2` for
optic normalization. LIF fan-in uses the full graph **after** connection caps, pathway gains and
same-type damping, preserving the previous calibration even with custom `LIFParams`. The original
graph is shared on CPU and its normalization totals are cached by parameter configuration. Retained
DN -> VNC synapses therefore keep their strength when other modules are removed. Retinal viewing
directions and sensory laterality also derive from the original graph. Dropped neurons contribute nothing;
`regions.pare()` can also keep only neurons on paths between chosen sources and sinks (BFS both
ways on nonzero adjacency, optional total hop limit), which is the graph-theoretic form of the
"automated model paring" an RL environment can later do with activity statistics
(`FlyBrain.activity_mask(min_hz)` accumulates LIF spike counts; graded vision cells are conservatively
retained because their activity is not measured in spike Hz). With recurrent cycles, `pare` retains
neurons on directed walks, rather than enumerating simple paths.

Cost depends on retained edges, batch size and kernel overhead. The full CNS has 25.6M stored edges.
Optic + visual projection + descending + VNC has 124,137 neurons and 16.0M edges.
The broad nose configuration (AL + MB + **all central** + descending + VNC) has 53,446 neurons and
11.0M edges; the earlier ~5M estimate assumed LH alone, which is not a separate module. Use `pare`
for narrower pathways. Saving a subset also saves its original normalization graph; GPU work shrinks,
but CPU reference memory and serialized size need not shrink proportionally.

## Performance

The demo advances **10 ms** per frame: 20 LIF steps at dt 0.5 ms and 10 optic steps at dt 1 ms.
The 71k figure describes the non-optic population; the current full `Brain` still allocates state
for all 167,106 neurons, masking optic neurons from spiking. Use `scripts/profile_brain.py` for
current, synchronized measurements rather than the original 16.6 ms estimates. Available levers:

1. **Subsets** (above): fewer neurons and synapses on the device.
2. **CUDA graphs** (`FlyBrain(cuda_graphs=True)`): capture optic and LIF kernels for a frame on
   fixed buffers, including registered Poisson RNG state. Cache by frame length and delay/optic
   phase, up to eight captures. Pulses expiring inside a frame use the eager path.
   The demo and RL environment also capture sensory ray tracing on stable scene/ray buffers;
   `World.move_sphere` updates captured geometry in place. Texture-noise constants are reused.
   See `docs/PERFORMANCE.md` for independent sensory capture controls and CPU/CUDA traces.
3. **Half-precision LIF weights** (`LIFParams(weight_dtype="float16")`): CUDA sparse matmul with
   half weights/transmitted spikes and a float32 output/accumulator. Optic and membrane state remain
   float32. Opt-in: rounding can change spikes, and speed gains are workload dependent.
4. **Event-driven synaptic input on CUDA** (`LIFParams(event_driven=True)`): the existing gather
   backend is also usable on CUDA, but its dynamic host synchronization prevents graph capture.
   It remains the default on MPS/CPU; current CUDA measurements favor sparse matmul.
5. **Coarser dt** where the benchmark allows (1 ms instead of 0.5).

For hosts with a fixed update tick, `FlyBrain.step_budget(wall_ms)` runs as many LIF steps as fit
in a measured best-effort budget and reports actual simulated time, completed steps and time dilation.
It synchronizes GPU work, adapts chunk length, and can overrun during warmup or contention.
`AsyncFlyBrain` owns the controller on its own thread / CUDA stream: the game submits CPU sensor
frames and reads a published CPU `MotorRates` snapshot without waiting for GPU work.

## Status

- [x] design
- [x] `Connectome.subset`, normalization metadata on the neuron table, `regions.py`
- [x] `senses.py`, `fly.FlyBrain`, `MotorRates`; demo and env rebuilt on it; probes unchanged
- [x] profile script, CUDA graphs, fp16 spmm, step budget / async wrapper
