# Optional extensions

Extensions are explicit experiments. A fresh `FlyBrain` has no attached code or
synthetic cells and follows the existing numerical path. Anatomical `modules=`
still selects a connectome subset; `attach()` registers an executable extension.
The shipped connectome, physiological defaults, and body readout are unchanged.

## Hooks and frame timing

```python
def background(fb, t_ms):
    fb.brain.set_poisson(fb.c.select(type="DNa02"), 30.)

fb.add_hook(background, when="pre", name="background")
fb.step(10)
fb.remove_hook("background")
```

`pre` runs once per `step()` call after held senses, before optic/LIF advance;
`post` runs once before returning. This includes calls too short to advance a LIF
step. Hooks receive the batched controller and simulation time in ms. Exceptions
abort the call and report the hook name; earlier hook side effects are not rolled
back. Use the existing `set_drive`, `set_poisson`, and `stimulate` primitives.

Hook drive writes are added to optic/sensory drive, including when vision replaces
its frame input. Poisson writes combine by **maximum** with senses and timed pulses,
matching `stimulate`; negative module Poisson rates clamp to zero. Drive can be
negative. Each hook owns its held inputs until it writes again, resets, or is
removed. Post-hook writes take effect on the next advance. Outside hooks, ordinary
Brain setters update the externally held input. Direct tensor mutation bypasses
input ownership and is not an extension API.

Modules run before each neural frame and receive the **previous frame's state**.
All their inputs are gathered before any module runs; attachment order cannot make
a zero-delay feedback loop. With extensions attached, long `step(ms)` calls split
into frames of at most 10 ms; the final shorter frame uses its actual duration.
`step_budget` can use shorter frames too. Fractional LIF time still carries over.
Hooks remain once per API call. Without extensions, `step(ms)` retains its existing
cadence. CUDA graphs capture only neural advancement: Python hooks and modules run
once outside warm-up/capture/replay, with held input buffers.

## Modules

```python
import torch
from flyverse.modules import FunctionModule

def relay(dt_ms, inputs):
    return {"target": inputs["source"] * .1}

extension = FunctionModule(
    reads={"source": "DNa02&somaSide=L"},
    writes={"target": "DNa02&somaSide=R"},
    fn=relay, name="experimental_relay", channel_out="drive_mv",
    kind="stop-gap", parameters={"gain_mv_per_hz": .1},
)
# This example requires equal source/target population sizes.
fb.attach(extension)
fb.detach("experimental_relay")
```

`Module` is a structural protocol in `flyverse/modules.py`. Selections use
`interp.common.resolve`: type, regex, superclass, anatomical module, bodyId, masks,
or row indices. Inputs and outputs are **(B, number of selected cells)** Torch
tensors on `fb.device`, including B=1. Output keys must match `writes` exactly;
values must be finite. Two modules cannot write overlapping cells on the same
channel, including overlaps between a module's own output labels.

| Field | Meaning |
|---|---|
| `quantity_in="rate_hz"` | Previous frame's running LIF rates, Hz |
| `quantity_in="drive_mv"` | Previous frame's applied drive, mV |
| `quantity_in="spike_count"` | Spikes during the previous frame, initially zero |
| `quantity_in="optic_rate"` | Previous optic rates; selectors must address graded cells |
| `channel_out="poisson_hz"` | Nonnegative forced rate, combined by maximum |
| `channel_out="drive_mv"` | Additive drive, mV per LIF step |
| `kind` in `describe()` | `stop-gap`, `mechanism`, `sensor`, `decoder`, or `analysis` |

Hand-designed behaviours must be classified `stop-gap`. A custom module implements
`reset(B, device)`, `step(dt_ms, inputs)`, `state_dict`, `load_state_dict`, and
`describe`. Descriptions should include class, parameters, checkpoint hash and
trainability. Treat declarations as fixed while attached; detach and reattach to
change populations/channels. A stateful module used with partial batch resets
also implements `reset_rows(rows)`; FlyBrain rejects an unsupported partial reset
before modifying neural state. Full reset uses the protocol's `reset`.

The included adapters are:

* `FunctionModule(reads, writes, fn)`: stateless `fn(dt_ms, inputs) -> outputs`.
  Stateful functions should implement the protocol so checkpoints remain complete.
* `TorchModule(reads, writes, net, channel_out)`: concatenates reads in dictionary
  order, runs `net(B, n_in)`, and splits `(B, n_out)` in writes order. Neural reads
  use ascending model-index order within each selection. `net` remains available
  to an optimizer; `describe()` hashes the current parameters and buffers.
* `SNNModule(reads, writes, c, input_cells, output_cells)`: runs a separate
  synthetic `Brain`. Input/output label mappings select child cells. Parent and
  child population widths must agree. Child inputs use `input_channel` (default
  `poisson_hz`); child rates feed parent writes. Child dynamics and RNG checkpoint
  independently. The parent's CSR is untouched.
* `ReadoutModule(reads, fn, channels=...)`: no writes. The function returns each
  named channel on the sorted union of read cells. `NTChannel` declares its unit
  and display range (without `channels`, outputs use arbitrary units and range 0–1).
  Supply channels for physical quantities. `readout(batch_index=...)` returns an immutable `NTSnapshot`
  through the [existing source interface](NT_READOUT.md). The observatory lists
  these computed channels under extensions; they are not automatically presented
  as neurotransmitter concentrations. An actual NT module can additionally be
  assigned to `fb.nt_source`.

## Synthetic graphs and scratch caches

```python
import pandas as pd

nodes = pd.DataFrame({"bodyId": [-1, -2], "type": ["aux_in", "aux_out"],
                      "superclass": ["auxiliary", "auxiliary"],
                      "side": ["L", "R"], "nt": ["acetylcholine", "gaba"]})
edges = pd.DataFrame({"body_pre": [-1], "body_post": [-2], "weight": [20], "sign": [1]})
c2 = fb.c.extend(nodes, edges, cache_dir="out/auxiliary-cache")
restored = c2.prune(c2.select(dataset="synthetic"))
```

Synthetic bodyIds are unique **negative int64** values. Biological IDs never
change. New nodes carry `dataset="synthetic"`, `release` (default `v1`), and a
superclass used as their anatomical module label (`synthetic` if unspecified).
Each new edge must touch a new node. Weight is a nonnegative magnitude; explicit
`sign` is -1/0/+1, or `nt` selects `NT_SIGN`. If both are missing, the presynaptic
node's NT supplies the sign. Old CSR entries, including stored zeros, are retained.
To add more edges later, create an extension that also introduces their new nodes;
`extend` does not edit existing-to-existing connections.

The original CPU reference is retained for subset normalization. Pruning all
new cells restores the original reference as well as its weights. `Connectome.prune`
removes rows; the older `Brain.prune` only removes outgoing synaptic contributions.
Construct a new Brain/FlyBrain from the result; kernels/graphs are built afresh.

`extend` writes a fresh scratch cache and refuses the shared cache or its children.
Without `cache_dir` it allocates a temporary directory, exposed as `c2.cache_dir`;
the caller owns cleanup. `save`/`load` preserve the extension, base graph, and
normalization reference. Fingerprints include the base fingerprint, extension
SHA-256, node/edge counts and namespaces. Neurome exports label synthetic cells
and both endpoints of cross-dataset edges explicitly.

## Boundary training

`TorchModule(..., boundary="vision", output_sizes={"intensity": n_pr})` replaces
the optic input's spectral projection with a learned network: flattened radiance
`(B, n_columns * 4)` -> nonnegative photoreceptor intensity `(B, n_pr)` in retinal
photoreceptor order. It uses empty neural reads/writes. The existing contrast
adaptation and optic dynamics follow it. Only one vision encoder can be attached.

`MotorDecoder(net, features=(...))` takes named `MotorRates` fields (Hz, divided
by `rate_scale`, default 50) and emits normalized forward/yaw. It is a `TorchModule`
with empty neural writes and a body program: `apply` replaces speed/yaw in the
command dictionary. Compose it through `programs.Composite`, after earlier body
readouts. `action(motor)` is its batched Torch path; scalar `apply` wraps it.

```python
from flyverse.env import EnvParams, FlyRoomEnv
from flyverse.modules import MotorDecoder

decoder = MotorDecoder(torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Tanh()))
env = FlyRoomEnv(batch=16, params=EnvParams(
    modules_attached=lambda env: [decoder], motor_decoder=decoder.name,
))
env.reset()
obs, reward, done, info = env.step()  # attached decoder's output
```

`EnvParams.modules_attached` accepts modules or a factory called once with the
constructed environment. Factories can resolve populations or network sizes from
`env.fb`. `step(actions)` still accepts policy output directly. For a policy that
emits decoder parameters, `action_adapter(env, policy_action)` can update the
attached decoder and return its `(B,2)` action; `env.provenance()` records the
adapter identifier and the decoder's current checkpoint hash. Keep learned state
in the attached module, not an unrecorded closure.

[examples/learned_motor_decoder.py](../examples/learned_motor_decoder.py) includes
CPU smoke training, supervision from `common.Recorder` NPZ motor fields, and a
small policy-gradient loop through FlyRoomEnv. It saves a checkpoint and JSON
description. The RL mode requires a graph cache; the smoke mode needs no dataset.
The artificial smoke teacher is a software check, not a claimed fly behaviour.

## Experimental gradients through neural dynamics

```python
from flyverse.brain import LIFParams
from flyverse.fly import FlyBrain

fb = FlyBrain(c_subset, device="cpu", lif_params=LIFParams(surrogate_grad=True))
drive = torch.full((fb.B, 1), 30., device=fb.device, requires_grad=True)
fb.brain.set_drive([0], drive)
fb.step(10)
fb.brain.rate.sum().backward()
fb.detach_state()  # retain state; truncate history before the next training window
```

Forward spikes remain Heaviside. Backward uses the fast-sigmoid derivative
`1 / (1 + 25 * abs(v - threshold))**2`; refractory/reset choices stay discrete.
Use continuous `drive_mv` to train through spikes. Poisson draws remain discrete
and do not provide a reparameterized rate gradient.

This enables functional Torch updates for both LIF and optics, including delayed
spikes, adaptation, depression, slow receptor tone, and module clocks. It is slow
and memory intensive: use small anatomical subsets and short windows, call
`detach_state` between truncated windows, and optimize only the intended boundary
parameters. Native CUDA/Metal kernels and CUDA graphs raise when requested;
explicit event-driven selection also raises because it discards subthreshold
gradients. Torch sparse matmul is selected automatically. The normal path is
unchanged. Checkpoints save values, not autograd graphs or hook/module code.

## Provenance and inspection

`fb.hooks`, `fb.attached_modules`, `fb.module_records()`, the observatory, and
`provenance(..., fb=fb)["model"]` expose attached code. Checkpoints contain hook
identifiers, module descriptions and states; reattach matching code before loading.
Hook code is never serialized. Legacy extension-free checkpoints still load.

`Recorder` and trace's `ArmAccumulator` retain extension inputs as
`module:<name>:drive_mv` / `module:<name>:poisson_hz` (hooks use `hook:<name>`).
Decompose separates their contributions from optic drive, preserving Hz versus
mV units (forced rates have their own `module_poisson` table); trace and decompose
include a `module_inputs` table. These are external
input classes, never new synaptic edges. Recordings carry the module declarations
so offline analyses retain the attribution.

Run dataset-free contracts with `python -m unittest discover -s tests -p test_modules.py`,
`test_extend.py`, and `test_surrogate.py`.

Implementation validation (against `d89155b`): the full CPU discovery run passed
252 tests with 42 data/accelerator checks skipped; the final module suite adds
one post-hook attribution check (15 module tests passed). Three additional CUDA
checks passed on a cluster worker: eager/captured hooks and module inputs, changing
vision-encoder input under capture, and backpropagation through an attached Torch
module. The unchanged path was bit-identical to that commit across six 1-second
CPU cases (B=1/4, motor/optic/clocked), comparing 110 state, spike-history and RNG
tensors. No timing or behavioural improvement is claimed by these checks.
