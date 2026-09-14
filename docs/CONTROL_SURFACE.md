# Using the fly brain controller

`FlyBrain` owns the neural model and sensory encoding. Environments provide physical sensor
values; `MotorRates` exposes named rates in Hz. `Locomotion`, `Flight`, and `Metabolism` retain
their existing behavioral parameters. The demo and batched RL environment both use this interface.

## A controller without a room

```python
from flyverse import FlyBrain

fb = FlyBrain(modules=["antennal_lobe", "mushroom_body", "central", "descending", "vnc"])
fb.smell({"DM1": 0.4, "VA2": 0.2}, {"DM1": 0.2, "VA2": 0.1})
fb.step(10.0)
motor = fb.motor()
print(motor.fwd_dn, motor.turn_L, motor.turn_R, motor.pn_glomeruli)
fb.stimulate({"type": "DNa02", "somaSide": "L"}, hz=150, ms=100)
```

`modules=None` loads all neurons. `fb.available_senses` lists the present senses; calling an
absent sense raises `ValueError`. A module selects cells, not a guarantee that a particular
behavior will survive the loss of the rest of its recurrent network.

| input | values |
|---|---|
| `vision(radiance)` | Nonnegative radiance `[UV, B, G, R]`, `(columns, 4)` or `(B, columns, 4)`, in `fb.retina` column order. A single frame broadcasts to the batch. |
| `smell(cL, cR)` | Dictionaries keyed by glomerulus; values are nonnegative concentrations, scalar or `(B,)`. Omitted keys mean zero concentration. |
| `wind(dL, dR)` | Backward antennal deflection, scalar or `(B,)`; `+1` is full backward deflection, `-1` full forward deflection. |
| `taste(sugar)` | Sugar contact in `[0, 1]`, scalar or `(B,)`; full contact drives the sweet GRNs at 120 Hz. |
| `stimulate(selection, hz, ms)` | Local neuron indices, a boolean mask, or `Connectome.select` criteria. Rates broadcast to `(B, selected_neurons)`. Pulses combine with sensory forcing by maximum and expire on a LIF step boundary. |

Inputs stay in effect until replaced. `step(ms)` returns actual simulated milliseconds and carries
fractional LIF steps forward. The optic model carries its own fractional substeps forward too;
short calls do not inadvertently run the optic lobe faster than the LIF model. `step(0)` advances
nothing. `reset()` clears state, inputs, pulses and clocks; `reset(rows)` resets selected batch
members while preserving the shared simulation clock and random stream. Reset does not reseed RNG.

`motor()` returns an independent CPU snapshot: scalar fields for one fly, `(B,)` arrays for a batch.
Missing readout groups return zero. `motor.row(i)` produces one fly's rates for scalar body classes.
Calling `motor()` does not advance body adaptation, hunger or casting. The output includes
`time_ms`, forward/backward DNs, bilateral goal/optomotor/wind/leg signals, proboscis, giant fibre,
jump, wing power/steering, halteres and PN means by glomerulus.

```python
from flyverse.body import FlyState, Locomotion

pose = FlyState()
locomotion = Locomotion()
locomotion.step(pose, fb.motor(), dt_s=0.01, bounds=(-1, 1, -1, 1))
```

Body classes are scalar; use one per batch member or implement vectorized body dynamics. The RL
environment continues to apply the policy's walking commands directly, with its existing rewards
and observations. Configure subsets there with `EnvParams(modules=(...), obs="pn")`, for example;
missing senses are skipped and an observation mode with no retained neurons is rejected.

## Connecting a world

`Air.antennae(eye, left, forward)` supplies the two concentration dictionaries.
`Air.deflections(forward, left)` supplies normalized wind deflections. Both accept one pose or
arrays `(B, 3)`. `fb.retina.ray_directions()` provides the optical sampling directions and weights;
the world ray tracer produces radiance, and the controller handles the optic/central-brain interface.

The old `body.motor_groups`, `body.wing_groups`, `Locomotion.readout(brain, groups)`,
`Flight.readout(brain, wings)`, `air.BilateralOlfaction` and `air.WindSense` interfaces remain as
compatibility adapters for the existing probes. New worlds need no neuron indices.

## Subsets and graph paring

```python
from flyverse import connectome, regions

c = connectome.load()
small = regions.pare(c, sources={"class": "olfactory"},
                     sinks={"superclass": "descending_neuron"}, max_hops=5)
fb = FlyBrain(small)
```

`c.subset(indices_or_mask)` preserves selection order and body IDs while rebuilding local indices.
Nested subsets retain the same original reference graph. `regions.pare` intersects forward and
backward reachability over nonzero directed connections. `max_hops` limits the total length of a
source-to-sink walk. It can return an empty `Connectome`; `FlyBrain` rejects empty graphs.

`in_syn` and `in_syn_l2` preserve full-graph optic normalization. LIF normalization retains the
original **capped, pathway-scaled, same-type-damped** full-graph input total for each neuron,
including when nondefault LIF parameters are used. These totals differ from raw synapse counts;
using raw counts would have retuned the full brain. Source laterality and retinal directions also
remain fixed when their target cells or neighboring eye columns are removed.

The reference graph stays on CPU. Saving a subset writes that reference alongside it so reloading
with different LIF parameters remains correct. Existing caches acquire normalization columns in
memory; no dataset recompile or automatic rewrite is required. Explicit `connectome.save` writes
the metadata. Raw connectomes should be treated as immutable after controller construction.

`fb.activity_mask(min_hz)` thresholds accumulated LIF spike frequency since reset. For a batch it
returns `(B, N)`; use `.any(axis=0)` to keep cells active in any fly. Graded optic/photoreceptor
cells remain selected because they have no spike frequency. Removing cells based on activity
changes the model and does not preserve full-brain trajectories.

## Hooks, modules, graph extension

Everything in this section is **opt-in and off by default**, and a controller with nothing attached
follows the numerical path documented above, byte for byte (`tests/test_bit_identity.py`).
[Optional extensions](EXTENSIBILITY.md) is the reference; this is the shape of the surface.

```python
fb.add_hook(lambda f, t_ms: f.brain.set_poisson(f.c.select(type="DNa02"), 30.), when="pre", name="bias")
fb.attach(FunctionModule({"src": {"type": "LC4"}}, {"dst": {"type": "DNp01"}}, fn, name="relay"))
c2 = fb.c.extend(nodes, edges, cache_dir="out/aux-cache")     # a new Connectome; build a new FlyBrain from it
```

| surface | what it is |
|---|---|
| `add_hook(fn, when="pre"/"post", name=...)`, `remove_hook(name)` | A named callable run once per `step()` call, before or after the advance. It receives the controller and `t_ms` and may only use the existing `set_drive` / `set_poisson` / `stimulate` primitives. An exception aborts the call and names the hook. |
| `attach(module)`, `detach(name)`, `attached_modules` | A named object that reads selected cells and writes a neural channel between frames. `FunctionModule`, `TorchModule`, `SNNModule` (an independent synthetic `Brain`) and `ReadoutModule` (reads only, sampled through the [NT source interface](NT_READOUT.md)) ship in `flyverse.modules`. |
| `Connectome.extend(nodes, edges, cache_dir=...)` | Synthetic cells with **negative int64** bodyIds and their edges, in a scratch cache. Biological IDs and biological-to-biological weights never change, and `prune` inverts it exactly. Construct a new `FlyBrain` from the result. |
| `LIFParams(surrogate_grad=True)` | Functional Torch LIF and optic updates that backpropagate through a short window. Torch only: native CUDA/Metal kernels, CUDA graphs and explicit `event_driven` all raise. |

Selections use the `interp.common.resolve` grammar, so a module names cell types or body IDs, never
row indices of a particular subset. Writes are validated at attach: two modules may not write the
same channel on the same cell. Drive writes are additive and may be negative; Poisson writes combine
with the senses and with `stimulate` **by maximum**, and a negative rate clamps to zero. Weights are
never rewritten by a module — an extension is an external input, not a new synapse.

Frame timing is the part that bites. A module that **writes** (or a vision encoder that feeds the
optic frame) splits `step(ms)` into frames of at most 10 ms, because its output is the next frame's
input; a module that only reads — a `ReadoutModule`, any `kind="analysis"` observer — does not, and
`step(50)` stays bit-identical to the same call with nothing attached. Observation does not perturb
the model. Modules see the **previous** frame's state, and all inputs are gathered before any module
runs, so attachment order cannot create a zero-delay loop.

`fb.hooks`, `fb.module_records()` and `provenance(..., fb=fb)["model"]` record what was attached;
checkpoints carry hook identifiers, module descriptions and module state, but never code — reattach
matching code before `load_state_dict`. Every attached thing declares a `kind`
(`stop-gap` / `mechanism` / `sensor` / `decoder` / `analysis`), so a result says which of its numbers
came from the connectome and which from hand-written code.

## Execution options

```python
from flyverse.brain import LIFParams

fb = FlyBrain(cuda_graphs=True, lif_params=LIFParams(weight_dtype="float16"))
```

Both options are off by default. CUDA graphs capture the optic and LIF frame together, preserving
RNG and delay-buffer phase during warmup/capture. Inputs use stable device buffers; the first
encounter with a new frame length or phase has capture overhead. The cache is bounded to eight
graphs, with eager execution for additional shapes and pulses that expire inside a frame.

Half precision applies to LIF sparse weights and transmitted spikes only. Sparse multiplication
uses a float32 output and accumulation; membrane, adaptation and optic state remain float32.
This follows the mixed-precision path supported by
[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmm). Graph RNG handling follows
[PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
Half precision changes rounding and may change spike timing. It is an experimental performance
option, not a new calibrated default. Likewise, `LIFParams(dt=1.0)` changes integration accuracy.

`FlyBrain(cuda_kernels=True)` opts into fused native CUDA updates and compact GPU motor
readouts (requires nvcc and a host C++ compiler). With this option,
`LIFParams(event_driven=True)` uses device-only event traversal and supports CUDA graphs.
The Torch event fallback still synchronizes with the host and cannot be captured.
`cuda_sparse="warp"` selects experimental CSR for LIF products and optic recurrence;
keep the default `"torch"` for batched workloads. `cuda_compact=False` disables compact
event traversal and the frozen-at-rest fast path. Public state retains its full shape.
See [CUDA measurements and tradeoffs](PERFORMANCE.md#native-cuda-execution-september-11-2026).

```text
python scripts/room_demo.py --cuda-graphs
python scripts/room_demo.py --cuda-graphs --weight-dtype float16
python scripts/room_demo.py --cuda-graphs --cuda-kernels --event-driven --cuda-sparse warp
python scripts/profile_brain.py --frames 100 --cuda-graphs --json out/profile.json
python scripts/profile_brain.py --modules antennal_lobe,mushroom_body,central,descending,vnc
python scripts/profile_brain.py --event-driven
```

The profiler separates sensor/rendering, neural stepping and readout wall time, synchronizing the
GPU at each boundary. It reports initialization, warmup and steady-frame timing separately from
the demo's UI work. The rendering workload uses one pose broadcast to the batch; this measures
batched controller cost, not rendering B independent viewpoints. `EnvParams` also exposes
`cuda_graphs`, `weight_dtype`, `brain_dt_ms`, `cuda_kernels`, `cuda_sparse`, `cuda_compact`
and `event_driven`.

For the actual interactive demo loop, `scripts/profile_room.py` includes drawing and presentation
and can export a CPU/CUDA timeline. See [room-demo profiling](PERFORMANCE.md) for commands, the
distinction between visible and headless runs, and validation of optic input hoisting and sensory
ray capture. In the demo and RL environment, `cuda_graphs` also enables sensory ray capture;
`--no-sensory-cuda-graphs` / `EnvParams(sensory_cuda_graphs=False)` disables that part independently.

A local RTX 4090 measurement (PyTorch 2.10.0+cu128, B=1, 30 measured frames after 8 warmup frames,
10 ms of simulated time per frame) gave these mean latencies. They are a development snapshot on
a shared workstation, not a controlled hardware benchmark; absolute timings varied between runs.

| configuration | neural step | sensors + step + readout, excluding UI |
|---|---:|---:|
| full, eager fp32 | 29.70 ms | 55.38 ms |
| full, CUDA graphs fp32 | 11.75 ms | 36.02 ms |
| full, CUDA graphs fp16 weights | 10.44 ms | 29.72 ms |

CUDA graphs consistently helped. The smaller fp16 advantage varied between runs, so it remains
opt-in. PN readout metadata is cached; readout was about 0.5 ms in this measurement. The broad
nose subset has 11.0M stored edges versus the full graph's 25.6M; a separate run gave 6.64 ms neural
stepping with graphs, while the event-driven CUDA path took 43.23 ms on that workload.

For a best-effort wall budget, `result = fb.step_budget(wall_ms=4)` reports `simulated_ms`,
`wall_ms`, `steps` and `time_dilation` (simulated/elapsed wall time). It measures completed GPU
work and adapts the number of steps per chunk. Shorter chunks also change optic feedback cadence.
Warmup/capture and contention can overrun a budget; a warmed call can return zero steps if none
are estimated to fit. This is not a hard real-time deadline.

For a host that cannot wait for GPU work:

```python
from flyverse import AsyncFlyBrain

with AsyncFlyBrain(fb, frame_ms=10.0) as worker:
    worker.submit(smell=({"DM1": 0.4}, {"DM1": 0.2}))
    latest = worker.motor()  # immediately returns the last published CPU snapshot
```

The worker owns `fb` exclusively until closed. It waits for the first submission, then advances
continuously, holding the latest sensor values. Submit NumPy/CPU values; CUDA tensors are rejected
to keep GPU transfers off the host thread. Sensor submissions coalesce; stimulation pulses queue.
`motor().time_ms` identifies the published simulation time. `wait_for_update` is an optional
blocking helper for offline consumers. Worker failures surface through host operations, and the
context manager joins the worker on exit.

## State and validation

`fb.state_dict()` / `fb.load_state_dict(state)` preserve neural/optic state, RNG, held inputs,
pending pulses, fractional clocks and activity statistics. Load into the same neuron ordering,
batch shape and model parameters. CUDA graphs are rebuilt after load. The demo adds pose,
metabolism, flight and all locomotion timers, including casting and adaptive baselines; old demo
save files remain readable, although old files did not contain all of those body timers.

CUDA sparse reductions can differ slightly with execution/allocation order; there is no guarantee
of bit-identical long trajectories across execution modes or devices. The tests check exact state
restoration, spike/rate/readout agreement in short comparisons, and tight tolerances on continuous
subthreshold state. Treat fp16 or coarser dt as model variants and rerun behavioral probes.

```text
python -m unittest discover -s tests -v
```

The small contracts need no dataset; CUDA-specific cases skip when CUDA is unavailable. With a
local cache and CUDA, set `FLYVERSE_INTEGRATION=1` to add full-hybrid graph, sensory/retina subset,
and demo checkpoint tests. In PowerShell:

```powershell
$env:FLYVERSE_INTEGRATION = '1'
python -m unittest discover -s tests -v
```

During this refactor, the original 30-frame demo reference matched spikes, rates, commands and
pose, and the B=2 RL reference matched observations/rewards. The unchanged calibration harness
matched 24/28 fields exactly: rest, taste, smell, DN stimulation and walking readouts matched;
later looming/rotation values diverged. Escape remained at 3.5 cm. No constants in `Locomotion`,
`Flight` or `Metabolism` were retuned. These checks establish the structural refactor's baseline;
they do not establish behavioral equivalence of arbitrary subsets or reduced precision.
