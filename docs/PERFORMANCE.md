# Room-demo profiling and optimizations

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
