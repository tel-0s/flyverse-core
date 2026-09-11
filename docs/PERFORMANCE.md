# Room-demo profiling and optimizations

## Native CUDA execution (September 11, 2026)

For the full-brain, single-fly room demo, try:

```powershell
python scripts/room_demo.py --cuda-graphs --cuda-kernels --event-driven --cuda-sparse warp --cam-scale 4 --trail-seconds 0
```

This retains all 167,106 neurons, the 0.5 ms LIF step, the 1 ms optic step and sensory
sampling. `--cam-scale` affects the display camera. Add `--fast --weight-dtype float16`
only if the existing coarser timesteps and half-weight rounding are acceptable.
Native kernels are opt-in; `--no-cuda-kernels` overrides `FLYVERSE_CUDA_KERNELS=1`.
For a dense or batched workload, start with `--cuda-kernels --cuda-graphs`, keeping
the default cuSPARSE products, then compare `--event-driven` with the actual activity.
Module clocks still require the matrix-product backend.

The implementation in `flyverse/cuda.py` builds `kernels/neural.cu` with `nvcc` and a
host C++ compiler (MSVC on Windows). Both must be discoverable by the CUDA toolchain.
It caches a small shared library under `~/.cache/flyverse/cuda` (override with
`FLYVERSE_CUDA_CACHE`), keyed by source, compiler version and build flags/device architecture.
The C interface avoids a PyTorch C++ ABI dependency. Kernels use Torch's current CUDA stream
and support graph capture. Compilation happens during construction, before capture.
The source ships in the wheel; compiled binaries are local. Windows, Torch 2.10.0+cu128,
CUDA toolkit 12.9 and an RTX 4090 were tested; Linux compilation has not been exercised here.

### What changed and what was retained

- LIF membrane, refractory period, adaptation, STD, rate and spike-count updates run in
  one kernel. Optic integration and the next rectified recurrent input run in one kernel.
  Poisson draws still use the same Torch generator and full tensor shape. Delays, update
  order and per-module clocks remain intact. Fast math is disabled; the adaptation
  increment explicitly follows Torch's contracted multiply-add.
- CUDA sparse matrices use 32-bit indices when dimensions/nnz fit, with an int64 fallback.
  This saves about 50 MiB of LIF index storage and 34 MiB of optic-recurrence index storage.
  CPU/MPS index choices are unchanged.
- Native events traverse outgoing CSC edges on the GPU, with no `nonzero`, dynamic
  host-sized work queue or per-step host read. Compact traversal launches only for
  neurons with outgoing edges. Atomic accumulation can change floating-point order;
  this is an optional backend, not a bitwise-deterministic mode.
- `--cuda-sparse warp` selects a custom warp-per-row CSR kernel for LIF products and
  optic recurrence. The other held optic products remain on cuSPARSE. It helps B=1
  here but repeats matrix reads across flies and loses at B=8, so it is not the default.
- Native motor readout reduces 90 groups on the GPU and transfers 91 doubles per fly,
  including the spike-history total: 728 bytes instead of the 668,424-byte full rate
  vector plus a separate total. PN means use double accumulation; ordinary group means
  can differ from NumPy's float32 reduction by roundoff. Returned snapshots own CPU data.
  The demo downloads contrast and full rates only at display cadence; explicit full
  probes still work. A standalone controller retains its diagnostic setting.

**Compaction decision:** full `(B,N)` state tensors remain publicly writable and keep
their original indices/checkpoint layout. Frozen optic cells can receive forced pulses,
and restored or directly edited state still needs integration. Native LIF skips a frozen
cell only while every relevant state/input is at rest; otherwise it uses the full update.
`--no-cuda-compact` disables this fast path and compact event traversal for comparison.
This is compact execution, not physically packed LIF storage. Physical state packing is
deferred: a captured full-brain LIF update is now about 6 us, and changing the public state
contract mainly promises memory savings for large batches. Removing empty CSR rows alone
barely helped B=1, before adding the cost of scattering their output back.

### Measurements on the shared machine

Fable was also using this machine. Kernel/transfer counts are the primary evidence;
component timings below are exploratory GPU measurements, **not reliable end-to-end
speedup estimates**. All UI traces use seed 0, 40 warmup + 8 measured frames, graph capture,
default timesteps, float32 weights, camera scale 4 and no trail. Each frame advances 10 ms.

| Per simulation frame | Torch at `7a0ceab` | Native fusion + cuSPARSE | Native events + warp CSR |
|---|---:|---:|---:|
| Neural GPU kernels | 1,090 | 261 | 151 |
| Captured neural/housekeeping host launches | 3 | 3 | 3 |
| Motor rate transfer | 668,424 bytes | 720 bytes before packing total | 728 bytes including total |
| Contrast transfers per 8 frames | 8 | 2 | 2 |

Headless/dummy-driver traces independently reproduced the 1,090-to-151 neural kernel
reduction. They still draw every fourth frame. The final packed total removes the separate
spike-history D2H copy/synchronization. Full-rate transfers move to `ui.draw`; they have not
disappeared. Removing the per-frame contrast copy also moves the GPU wait from `brain.frame`
into `brain.motor`, so inclusive CPU scope durations must not be interpreted as isolated
computation. Sensory rays still execute 796 GPU kernels per frame through one captured graph.

`benchmark_cuda.py` compares the actual pruned LIF matrix (12,991,449 nnz) and optic
recurrence (8,780,774 nnz), fp32/fp16 LIF weights, B=1/8, and 0.1/1/10/100% activity.
It alternates backend order and times captured repetitions with CUDA events. Example
medians in microseconds from `out/cuda_sparse_compact.json` (5 rounds, 8 products/round):

| Matrix / batch / activity | cuSPARSE int64 | cuSPARSE int32 | Warp CSR | Compact events, including clear |
|---|---:|---:|---:|---:|
| LIF / 1 / 1% | 187 | 166 | 148 | 12 |
| LIF / 8 / 1% | 618 | 530 | 1,173 | 67 |
| LIF / 1 / 100% | 184 | 160 | 147 | 315 |
| Optic / 1 / dense | 134 | 121 | 112 | — |
| Optic / 8 / dense | 415 | 354 | 845 | — |

Events win strongly at low activity and lose during storms; no automatic runtime switching
is implemented. For half weights, both matrix backends round the dense operand to half and
accumulate into float32. Events retain float32 transmitted amplitudes, matching the existing
event backend; the rounding contract therefore differs from half-precision matrix products.

The remaining neural target is optic recurrence (~107 us per substep in the captured UI
trace). Outside the brain, sensory uploads still synchronize, and display cameras still use
eager rendering. These need separate profiling rather than extrapolating neural speedups to UI FPS.

### Reproduce and validate

```powershell
python scripts/profile_room.py --warmup 40 --frames 8 --trace out/cuda_ui.trace.json --json out/cuda_ui.json --cuda-graphs --cuda-kernels --event-driven --cuda-sparse warp --cam-scale 4 --trail-seconds 0
python scripts/summarize_cuda_trace.py out/cuda_ui.trace.json --json out/cuda_work.json
python scripts/benchmark_cuda.py --rounds 5 --repeats 8 --json out/cuda_sparse.json
$env:FLYVERSE_CUDA_TESTS='1'
$env:FLYVERSE_INTEGRATION='1'
python -m unittest discover -s tests -v
```

Repeat the profile with `--headless` and a separate filename. Use longer runs without
`--trace` for throughput when the machine is available. `summarize_cuda_trace.py` associates
GPU work with CPU scopes through runtime correlation IDs, including graph replay; it does
not infer GPU ownership from overlapping wall-clock intervals.

Validation covers neuron/optic arithmetic, STD, clocks, identical Torch RNG progression,
delay-buffer storage offsets, nondefault streams, sparse/event products, empty matrices,
bad tensor layouts/dtypes, motor groups and independent snapshots, direct probe edits,
partial resets, pulse expiry, graph replay and full-connectome/B=2 environment runs.
Short full-hybrid comparisons preserve exact spike histories within the tested cases;
subthreshold sparse reductions are compared with tolerances. Long chaotic trajectories
are not guaranteed identical across sparse algorithms.
The final full run passed 39 tests and skipped 8 Metal-only tests on Windows. Wheel
construction also succeeded and included the CUDA source and loader.

Checkpoint testing also found and fixed missing per-clock spike accumulators. New saves
include them and clock multipliers; older unclocked saves still load, while older clocked
saves are rejected because their pending inputs cannot be reconstructed. Replacing sparse
weights or clock partitions invalidates captured frames. One pre-existing demo test was
updated from removed casting timers to the body's current optomotor adaptation state.

Local artifacts are ignored under `out/`: `cuda_before_ui`, `cuda_after_ui`, `cuda_events_ui`,
`cuda_final_ui`, `cuda_torch_headless`, `cuda_events_headless`, their `.trace.json` timelines
and `*_trace.json` reports, `cuda_final_headless`, plus `cuda_sparse_compact.json`. The initial UI baseline uses
int64 indices; the later Torch headless control already uses int32. Counts agree, but their
timings should not be treated as a matched index-width experiment.

## Earlier optic input optimization

The first optimization after the control-surface refactor computes the photoreceptor-input
and spiking-feedback products once per optic frame. Their source vectors are held constant
inside `OpticLobe.step_frame`; recurrent optic input and adaptation still change each substep.
The two fixed terms remain separate to preserve the existing addition order. A frame with
no completed optic substeps does not compute them, and they are recalculated on the next frame.
All neurons, weights, timesteps, sensory sampling and body constants are retained.

## Profiling the actual demo

`scripts/profile_room.py` wraps the existing `room_demo.main` loop. It includes sensory
ray tracing, body updates, drawing every fourth simulation frame, display presentation,
event handling and the 240 Hz frame limiter. Unrecognized arguments are forwarded to the demo.
Run from the refactor worktree:

```powershell
python scripts/profile_room.py --warmup 40 --frames 160 --json out/room_ui.json --cuda-graphs --weight-dtype float16 --fast --cam-scale 4 --trail-seconds 0
```

Run the same command with `--headless` and a different output filename for a separate dummy-display
measurement. **Headless still executes the drawing code.** It is neither a no-render benchmark
nor a substitute for measuring the visible window. Leave the window unpaused and avoid resets,
loads, camera changes and other interactions during matched runs. A fixed camera also keeps
the cached scene overview representative of the normal startup view; moving it is a different
rendering workload.

Capture a short CPU/CUDA timeline in a separate run, after warmup and graph capture:

```powershell
python scripts/profile_room.py --warmup 40 --frames 8 --trace out/room_ui.trace.json --json out/room_ui_trace.json --cuda-graphs --weight-dtype float16 --fast --cam-scale 4 --trail-seconds 0
```

The Chrome-format trace can be opened in a compatible trace viewer such as Perfetto. It includes
named CPU ranges (`sensor.raytrace`, `brain.frame`, `brain.motor`, `ui.draw`, `ui.present`,
`room.frame`) and CUDA kernels, including graph replay. Trace collection adds overhead: use
the run without `--trace` for throughput. CUDA events measure neural-frame stream elapsed time;
there are no added GPU synchronizations at individual stage boundaries. Inclusive CPU scopes
overlap and include existing waits, so their times must not be summed. Stream elapsed time can
include host submission gaps and contention; it is not the sum of kernel execution times.
Initialization and warmup are excluded from measured frame times.

Optional `--state out/room.pt` saves the final controller, pose and commands for comparisons.
This is a profiling snapshot, not the demo's F5/F9 save format. Closing the window early marks
the report incomplete (or exits without a report if no measured frames completed).

## Sensory ray tracing

In the demo, `--cuda-graphs` now enables both neural-frame capture and sensory ray capture.
The existing fast command therefore enables the new path automatically:

```powershell
python scripts/room_demo.py --cuda-graphs --weight-dtype float16 --fast --cam-scale 4 --trail-seconds 0
```

`--no-sensory-cuda-graphs` disables just the ray graph for comparisons; `--sensory-cuda-graphs`
can enable it independently of neural capture. `EnvParams.sensory_cuda_graphs` defaults to
`None`, which follows `EnvParams.cuda_graphs`. `FlyBrain` itself still handles only the brain;
ray capture belongs to the environment. Direct callers use
`world.trace(origins, directions, cuda_graphs=True)`. CPU/MPS rendering remains eager.

The renderer keeps the same intersections, four spectral channels, hard shadows, procedural
textures and arithmetic order. Eight integer texture-noise corners are uploaded once when
packing the scene, replacing 24 small per-trace uploads. CUDA capture then replays the packed
ray computation with fresh origin/direction buffers. The result is copied to independently
owned storage so subsequent sensory or display-camera calls cannot overwrite it. The demo's
display cameras use eager rendering; ray sampling and camera resolution are unchanged.

`world.move_sphere(index, center, radii)` updates the packed tensors in place, including a
captured looming object, without rebuilding the graph. Call `world.invalidate()` after other
direct edits to geometry, material properties or lighting. Changes in object counts and device
trigger repacking automatically. The cache holds at most two combinations of ray shapes,
detail setting and CUDA stream; additional combinations use eager execution. Repacking clears
the cache. Capture has startup cost and retains a GPU memory pool, so warm it before timing.

With the same full-brain fast configuration and 8 measured frames after 40 warmup frames,
the CPU/CUDA traces show this change in the sensory ray-tracing scope:

| Per sensory frame | Before ray optimization (`1d85c09`) | Captured ray path |
|---|---:|---:|
| Individual CPU calls to `cudaLaunchKernel` | 796 | 2 |
| CPU calls to `cudaGraphLaunch` | 0 | 1 |
| Host-to-device copies | 26 | 2 |
| Explicit `cudaStreamSynchronize` calls | 26 | 2 |
| Executed GPU kernels | 796 | 796 |

The two remaining individual launches average the rays into ommatidial columns outside
the graph. Three extra device-to-device copies fill the two graph input buffers and return
an owned result. The visible and headless captured traces had identical API call counts.
This change reduces host submission and synchronization overhead; it does not fuse or remove
the rendering math. The observed UI CPU ray scope fell from about 17.2 ms to 1.7 ms, but some
GPU-completion waiting moved into `sensor.vision`. That is not a tenfold improvement in the
time until sensory data is ready. Trace overhead and Fable's concurrent workload also prevent
turning these observations into a reliable end-to-end speedup claim.

Validation covered changing viewpoints, noncontiguous ray inputs, moving/resizing objects,
shadows, material/lighting edits, topology changes, empty scenes, device changes, a nondefault
CUDA stream, bounded caching and image lifetime across display-camera calls. The full suite
passed 29 tests with `FLYVERSE_INTEGRATION=1`. A 12-frame demo comparison with looming matched
radiance, spikes, motor commands and pose exactly; a B=2 RL environment also produced identical
radiance with either ray backend. Comparing the renderer directly with the pre-change source
at three viewpoints (512 rays each on CPU, 10,262 each on CUDA) gave bit-identical radiance.
These comparisons validate the tested inputs rather than guaranteeing all long trajectories.

The separate 200-frame UI runs ended on different neural/body trajectories. Follow-up lockstep
comparisons located the first difference in neural conductance (about 3.8e-6 at frame 1), while
radiance stayed bit-identical until the body poses diverged. An eager-versus-eager control also
showed neural differences followed by trajectory divergence. This is evidence of existing neural
numerical variability, not a guarantee of deterministic behavior from the optimized renderer.
The isolated renderer comparisons avoid that feedback confound.

One pair of non-trace visible-UI runs measured 20.93 ms/frame with eager rays and 13.65 ms/frame
with captured rays (160 measured frames after 40 warmup frames). Both used shared noise constants
and neural CUDA graphs. These are provisional observations under contention, with diverging
closed-loop trajectories, rather than a controlled 1.53x throughput result.

Local artifacts: `out/ray_before_ui.trace.json`, `out/ray_after_ui.trace.json`,
`out/ray_after_headless.trace.json`, their `*_trace.json` reports and
`out/ray_reference_validation.json`. `ray_demo_validation.json` records the longer-run differences;
`ray_first_difference_capture.json` and `ray_first_difference_eager.json` locate their onset.
The non-trace `ray_eager_ui.json` / `ray_after_ui.json`
runs compare graph-disabled and graph-enabled rendering with the new shared noise constants
in both; they are also subject to workstation contention.

## Optic input-hoisting validation: September 10, 2026

Measured on RTX 4090 / PyTorch 2.10.0+cu128 with the full 167,106-neuron connectome, seed 0,
CUDA graphs, fp16 LIF weights, `--fast --cam-scale 4 --trail-seconds 0`. Baseline code was
commit `56e5dc6`. Fable was concurrently using the workstation for sensory-pathway work;
**these runs do not establish a wall-time speedup or a UI-versus-headless speed comparison**.

Eight-frame traces, after 40 warmup frames, gave identical work counts in both display modes:

| Per 10 ms simulation frame | Before | After |
|---|---:|---:|
| Optic sparse products, including photoreceptor averaging and output drive | 17 | 9 |
| LIF sparse products | 10 | 10 |
| Total sparse products | 27 | 19 |
| Neural CUDA kernels | 620 | 580 |
| Sensory ray-tracing CUDA kernels | 796 | 796 |
| UI drawing CUDA kernels, per drawn frame | 830 | 830 |

The installed backend dispatched the batch-one products to cuSPARSE `csrmv_v3_kernel`.
The removed products also remove associated partitioning, setup and scaling kernels.
With the original optic dt of 1 ms, the source change removes 18 sparse products per 10 ms
frame instead of 8; that configuration's work reduction is derived from the loop rather than
measured in these fast-mode traces.

The sensory trace contained roughly 1.2 ms of kernel execution within a CPU ray-tracing scope
of roughly 18 ms. This makes submission overhead and gaps between kernels worth investigating
next, but the concurrent workload prevents isolating their causes or predicting a speedup.

Validation completed:

- All 21 existing tests passed with `FLYVERSE_INTEGRATION=1`, including full-hybrid CUDA graph
  agreement, fractional optic time and checkpoint/resume.
- After 200 matched visible-demo frames (40 warmup + 160 measured), spike counts, transmitted
  spike buffers, neural rates, RNG state, motor commands and body pose matched exactly.
- Continuous internal state had small differences: maximum absolute differences were about
  0.00016 for membrane voltage, 0.00034 for optic-to-LIF drive, and 0.000021 for optic voltage.
  This does not establish bit-identical continuous state or long-run trajectory equivalence.

Local artifacts are in the ignored `out/` directory: `optic_before_ui.json`,
`optic_after_ui.json`, the corresponding `*_headless.json` reports, and four
`optic_{before,after}_{ui,headless}.trace.json` timelines. The separate `*_trace.json`
reports contain instrumented timings. Preserve these labels when comparing results.

## Apple MPS: Metal kernels (September 11, 2026)

Measured on an M-series Mac, torch 2.14.0, full connectome, seed 0, `profile_room.py --headless
--cam-scale 4 --trail-seconds 0`, 20 warmup + 80 measured frames. CUDA graphs, fp16 weights and
per-module clocks are unavailable on MPS; the optic input hoisting and `prune_frozen` apply.

| per 10 ms frame | torch on MPS | Metal kernels |
|---|---:|---:|
| full fidelity (LIF dt 0.5, optic dt 1): frame wall | 66.5 ms (0.15x) | **34.3 ms (0.29x)** |
| `--fast` (LIF dt 1, optic dt 2): frame wall | 41.6 ms (0.24x) | **25.9 ms (0.39x)** |
| LIF step (event-driven, ~10-60 spikes/step) | 0.76 ms + host sync | 0.12 ms |
| optic `step_frame`, 10 substeps | 38.0 ms | 7.2 ms |
| optic recurrence W_rr @ dr (8.8M nnz) | 3.57 ms (COO spmm) | 0.61 ms (SIMD-group CSR) |

Why: torch's MPS backend has ~40 us per launch and no graph capture, so a LIF step was ~20 launches plus
the `nonzero` host sync of the event-driven gather, and its COO sparse kernel runs ~5x off memory
bandwidth on the optic lobe's skewed row lengths (median 69, max 11,421 synapses per row).
`flyverse/metal.py` compiles five kernels through `torch.mps.compile_shader`: `event_scatter` (one SIMD
group per presynaptic neuron, early exit when silent, float atomics into g; no host sync, any batch
size), `lif_update` (the whole state update in one pass; Poisson draws still come from the torch
generator, so seeds and saved RNG state are unchanged), `csr_spmv` (one SIMD group per row, `simd_sum`),
`optic_dr` and `optic_substep`. A LIF step is three launches; an optic substep is two.

Validation (`tests/test_metal.py`, plus a full-connectome comparison): over 150 ms with 13.6k active
neurons, deterministic or Poisson drive, B = 1 and B = 3, the Metal and torch backends produce identical
spike counts per neuron (37,834 and 57,696 spikes in the two runs); v, g and rates differ by at most
1.5e-4 (Metal compiles with fast-math and may contract multiply-adds). Optic drive differs by 1.4e-5 of
35 mV. The headless demo's loom escape is unchanged in both presets. The kernels read raw storage, so the
wrappers reject non-contiguous tensors (a view with a storage offset, such as a delay-buffer slot, is fine).

What remains per frame at full fidelity: the brain's ~15 ms of GPU time (20 LIF steps + 10 optic
substeps; MPS only enqueues in `brain.frame`, so the wait appears in `brain.motor`), sensory ray tracing
~10 ms (hundreds of small torch ops; the same treatment would apply), senses ~2 ms, and drawing ~4 ms
amortised over every fourth frame.
