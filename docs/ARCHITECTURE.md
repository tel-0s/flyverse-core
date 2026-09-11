# Architecture: the fly brain as a control system

Goal: the connectome model should plug into anything -- the picnic-table demo, an RL environment,
a game with a fixed update tick, an interpretability notebook -- through one small surface, and it
should be possible to run only the parts of the brain a simulation needs. This document is the
design; `docs/NOTES.md` has the measurements that motivated it.

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
| `visual_projection` | LC / LPLC / LPT / MeTu / ... | vision -> central brain |
| `antennal_lobe` | ORNs, AL local neurons, PNs | smell |
| `mushroom_body` | KCs, MBONs, DANs, APL / DPM | learning-related readouts |
| `gustatory` | GRNs + the sweet second-order set | taste |
| `mechanosensory` | JO neurons, AMMC / WED interneurons | wind, sound |
| `central` | the rest of `cb_intrinsic` (LH, CX, LAL, AVLP, GNG, ...) | everything |
| `descending` | DNs | any motor output |
| `vnc` | VNC intrinsic + motor + sensory + ascending | leg / wing motor neurons |

`FlyBrain(modules=[...])` composes them; a sense whose neurons are not in the subset is simply
absent (`fb.vision` raises). Two things make subsets behave like the full model: the fan-in cap
uses each neuron's **full-connectome** synapse count (stored on the neuron table at compile time),
and the `DEFAULT_PATH_GAIN` / `TYPE_PATH_GAIN` rules apply by name, so a DN -> VNC synapse has the
same strength whether or not the optic lobe is loaded. Neurons that are dropped contribute nothing;
`regions.pare()` can also keep only neurons on paths between chosen sources and sinks (BFS both
ways on the boolean adjacency, optional hop limit), which is the graph-theoretic form of the
"automated model paring" an RL environment can later do with activity statistics
(`FlyBrain.activity_mask(min_hz)` accumulates them).

Cost scales with synapses kept: the full CNS is 25.6M edges; optic + VP + descending + VNC (the
"vision-to-legs" fly) is ~14M; AL + MB + LH + descending + VNC (the "nose" fly) ~5M.

## Performance

Measured per frame (16.6 ms of fly time, B = 1, RTX 4090) before this work: LIF 33 steps x
(spmm + ~15 elementwise kernels) ~ 10 ms, optic lobe 16 steps ~ 3 ms, ray tracing ~1.5 ms, readouts
(one device->host copy) < 0.5 ms. Since `(B, N)` state is 71k floats, the LIF is launch-bound, not
bandwidth-bound; the levers, in order of expected gain:

1. **Subsets** (above): fewer synapses, linearly.
2. **CUDA-graph capture of a whole frame** (`torch.cuda.CUDAGraph`): the 33-step loop is a fixed
   sequence of kernels on fixed buffers; replaying it removes ~500 launches of Python + driver
   overhead. Same for the optic lobe.
3. **Half-precision weights** for the sparse matmul (fp16 values, fp32 accumulate): halves the
   bytes of the one bandwidth-bound kernel.
4. **Event-driven synaptic input on CUDA** at low activity (already the path on MPS / CPU).
5. **Coarser dt** where the benchmark allows (1 ms instead of 0.5).

For hosts with a fixed update tick, `FlyBrain.step_budget(wall_ms)` runs as many LIF steps as fit
in the budget and reports the resulting time dilation (a slow-motion fly rather than a dropped
frame), and `AsyncFlyBrain` runs the brain on its own thread / CUDA stream: the game pushes the
latest sensory frame and reads the latest `MotorRates`, never blocking on the GPU.

## Status

- [x] design
- [ ] `Connectome.subset`, `in_syn` on the neuron table, `regions.py`
- [ ] `senses.py`, `fly.FlyBrain`, `MotorRates`; demo and env rebuilt on it; probes unchanged
- [ ] profile script, CUDA graphs, fp16 spmm, step budget / async wrapper
