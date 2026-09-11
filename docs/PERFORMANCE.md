# Room-demo profiling and optic input hoisting

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

## Validation snapshot: September 10, 2026

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
