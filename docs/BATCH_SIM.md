# Batched room simulations

`BatchSim` runs independent room environments around one `FlyBrain(batch=B)`. It
retains the full room model: seven rays per ommatidium, bilateral odour and wind,
3-D fruit contact, surface walking, flight, feeding, and optional behaviour programs.
The default frame is still 10 ms, with LIF dt 0.5 ms and optic dt 1 ms. No brain
modules, sensory rays, or body mechanisms are removed to obtain batching.

This is a headless rollout API. The interactive `room_demo.Sim` remains the scalar
reference. `FlyRoomEnv` keeps its existing, simpler action/reward-based RL contract.

## Run a sweep

```powershell
python scripts/batch_sustain.py --batch 8 --minutes 5 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch --json out/batch.json
```

The JSON has a record per fly: energy, meals, hops, path length, distance to fruit,
mode fractions and final position. Progress prints every ten simulated seconds.
`--save out/batch.pt` saves the final state; `--load out/batch.pt` resumes a matching
configuration. A resumed run's path/mode/hop statistics describe the new measurement
interval; energy and meal counts retain the restored episode state.

`--seed` sets the batched brain RNG seed and the first environment seed. By default,
environments use consecutive seeds. `--seeds 2,7,11,19 --batch 4` selects them explicitly.
The runner uses those environment seeds for initial headings, as `probe_sustain.py`
does, and defaults to the same start position and energy as that probe.

**Batch rows are independent, but they are not bit-identical replays of separate
single-seed processes.** The existing batched brain uses one Torch RNG generating
independent draws over `(B,N)`. Changing B changes the draw layout. Environment seeds
control fruit geometry and plume phases; each program owns its own RNG instance
with the scalar program's existing initialization. Native event atomics retain
their existing run-to-run nondeterminism. Compare behavioural distributions, and
record both the master brain seed and the environment seeds.

## Python interface

```python
from flyverse import BatchSim

sim = BatchSim(8, seed=0, program="cx", fruit_set="apple", fence=True,
               cuda_graphs=True, cuda_kernels=True, event_driven=True,
               cuda_sparse="torch")
for _ in range(100):
    motor = sim.step()                 # one MotorRates snapshot for all B flies

positions = [fly.pos for fly in sim.flies]
energy = [m.energy for m in sim.metabolisms]
sim.start_loom(rows=[1, 6])             # independent objects in just those scenes
sim.stimulate_gf(rows=[2])              # other brain rows receive no pulse
sim.reset(rows=[3])                    # restart one episode
sim.save_state("out/batch.pt")
```

`start` accepts `(3,)` or `(B,3)` coordinates. `program` accepts a single name or one
name per row, including compositions such as `cx+klinotaxis`. `wind_speed` and
`wind_dir` accept scalars or `(B,)` arrays. Brain backend options, `modules`, and an
explicit connectome/device pass through to `FlyBrain`; missing senses are skipped.

Public `flies`, `locos`, `flights`, `metabolisms`, `programs`, `gatings`, and `airs`
contain independent scalar model objects. `commands`, `wcommands`, `feeding`, and
`tasting` describe the latest frame. Changing a body's parameters affects the next
batch update; constants are read from the scalar instances rather than duplicated.
Program code still receives its own scalar fly, metabolism and motor row.

`reset(rows=...)` restores those rows to constructor defaults and configured start
positions/programs, clearing their body state, plume clock and brain state. It
retains the shared neural clock and RNG, following `FlyBrain.reset(rows)`; other
rows continue uninterrupted. A full reset also resets the shared clock. Set custom
per-row body parameters again after a reset. The API does not automatically reset
starving flies or terminate episodes.

Checkpoints include the batched controller, body state/parameters, scene objects,
air clocks/phases, loom parameters, program state **and program RNGs**. Restoring an
initial checkpoint also removes dynamic program state created after it was saved.
Snapshots own their CPU data. Batch size, environment seeds, program names, fruit
set, fence and gating configuration must match; `FlyBrain` additionally validates
the connectome and integration parameters. Optional NT module state remains the
module owner's responsibility, as with the existing NT readout hook.

## What is batched

| Component | Work shared across rows |
|---|---|
| `batch_world.BatchWorld` | One `(B,rays,3)` trace including shadows and the existing procedural textures. Every row has separate geometry, materials and lights. Both intersections and shading run over B. CUDA graphs capture the batch. |
| `batch_air.BatchAir` | Bilateral concentration samples over `(B,2,sources)` and wind deflections. Each row keeps its scene, plume phases, wind settings and clock. |
| `batch_body.frames` | Full body axes, flight roll/pitch and edge-blend rotation, then all retinal ray transforms. |
| `BatchBody` | Motor readouts and optomotor adaptation, metabolic hysteresis, continuous walking and free-flight integration. |
| `BatchSurfaces` | Face crossing/containment and ordinary surface steps over B. Edge transitions, takeoff, landing and collision recovery call the scalar reference methods only for affected rows. |
| `FlyBrain` | One held sensory frame per input, one neural frame, and one motor readout for all rows. |
| Programs | Independent scalar state machines; same-frame brain pulses are coalesced by selector/duration into `(B,selected_neurons)` rates before controller calls. |

`BatchWorld` requires matching primitive counts and a shared texture-detail setting;
the individual primitive positions, materials and lights may differ. The room
factory meets that contract for a given fruit set. `BatchAir` similarly requires
matching source counts and glomerulus layouts. `BatchSim` shares the room factory's
walkable table/room surfaces, whose geometry is invariant across its seeds.

Use `sim.world.move_sphere(index, centers, rows=[...])` for dynamic geometry; it
updates captured buffers in place. For direct scene/material/light edits, edit
`sim.world.worlds` and call `sim.world.invalidate()`. The primitive count constraint
still applies. Moving visible fruit does not redefine odour/contact sources;
those are explicit environment configuration, as in the scalar room demo.

The native Metal ray kernel is not used by `BatchWorld`; its Torch path is portable,
and its CUDA graph path is the optimized target of this change. There is no new
CUDA kernel or change to synaptic weights, behavioural thresholds or integration dt.

## Verification and profiling

```powershell
python -m unittest discover -s tests -p test_batch_sim.py
python scripts/profile_batch.py --scalar --batches 1,4,8,16 --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch --trace out/batch_traces --json out/profile_batch.json
```

Use the project's configured cluster runner for full GPU checks and sweeps. Enable
`FLYVERSE_BATCH_CUDA=1` and `FLYVERSE_BATCH_INTEGRATION=1` there to run the full tests.
CPU tests require no dataset. They compare physical sensing and mixed body states
to the scalar model, test row-specific stimuli/reset, and round-trip program RNGs.
Full-connectome checks compare B=1 against `room_demo.Sim`, exercise B=4 with native
CUDA and `cx`, and verify captured ray scene isolation and checkpoint state.

The profiler measures complete frames, aggregate fly-seconds per wall second,
CUDA spans, peak allocated memory, and CPU body/physical-sense work. Optional
three-frame traces report device event counts and execution time. The isolated CPU
stage excludes ray tracing, neural sensory encoding and program execution; the
complete-frame measurement includes them. CUDA spans include idle gaps while the
host prepares work, so they are not pure kernel execution time.

Batching improves aggregate throughput, not necessarily the latency of one fly.
Scene tensors and state grow with B; increase batch size while monitoring memory.
The current `cx` program uses 15 ms pulses on 10 ms frames. Their expirations split
neural frames, so `FlyBrain` retains its existing eager fallback for those frames;
requesting CUDA graphs does not imply that every neural frame is captured.
Surface transitions and program dispatch still contain per-row Python work.

The initial B200 run (full mixed-fruit room, no programs, native CUDA events,
cuSPARSE, neural/sensory graphs, 12 warm-up + 50 measured frames) produced:

| Configuration | Frame ms | Aggregate fly-s / wall-s | CPU body + physical senses ms | GPU events / frame |
|---|---:|---:|---:|---:|
| Scalar demo | 5.13 | 1.95 | not isolated | 1,013 |
| BatchSim B=1 | 5.94 | 1.68 | 0.57 | 1,023 |
| BatchSim B=4 | 13.02 | 3.07 | 0.66 | 1,048 |
| BatchSim B=8 | 15.08 | 5.31 | 0.82 | 1,048 |
| BatchSim B=16 | 19.78 | 8.09 | 0.93 | 1,048 |

These are descriptive measurements on shared infrastructure, not idle-GPU speed
guarantees. The B=16 result is about 4.1 times the scalar aggregate throughput; its
individual rollout advances at about half real time. B=1 has batching overhead and
the scalar demo remains preferable for one fly. Event counts come from separate
three-frame CPU/CUDA traces: more flies increase the work within each operation,
while the number of operations stays nearly constant.

With `--program cx --fruit apple --fence`, the same protocol gave:

| Configuration | Frame ms | Aggregate fly-s / wall-s | CPU body + physical senses ms |
|---|---:|---:|---:|
| Scalar demo | 18.49 | 0.54 | not isolated |
| BatchSim B=1 | 11.68 | 0.86 | 0.46 |
| BatchSim B=8 | 19.22 | 4.16 | 0.63 |
| BatchSim B=16 | 25.43 | 6.29 | 0.69 |
| BatchSim B=32 | 35.27 | 9.07 | 0.97 |
| BatchSim B=64 | 54.78 | 11.68 | 1.36 |

The row stimulus adapter also caches resolved program selectors; it avoids repeated
dataframe queries even at B=1. The B=64 trace had about 1,325 GPU events per frame
versus 1,290 for the scalar `cx` loop, rather than 64 copies of the kernel sequence.
Peak Torch-allocated memory at B=64 was 1,092 MiB; this excludes CUDA context memory
and allocator reservations. The final part of this sweep overlapped validation work,
so retain the shared-machine qualification when using these numbers.

The measurements and trace summaries are recorded in [batch_profile.json](batch_profile.json).
Validation: 44 CPU/UI tests passed (GPU/integration checks disabled locally); the
cluster run passed 48 checks, including all 12 batching checks and 200 scalar body
comparison subtests. Ten separate opt-in native-kernel unit tests were not enabled
in that run. Native B=4 stepping, B=1 demo agreement with and without `cx`, ray
capture, and batched checkpoint restoration were exercised by the batching checks.
