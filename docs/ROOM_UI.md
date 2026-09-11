# The room console

The room demo uses a dedicated pygame presentation layer in `flyverse/room_ui.py`.
The default 1360x820 window is a compact monospace instrument console: thin rules,
aligned numeric columns, model readout names, and restrained colour for different signals.
The body camera is larger than the orbit camera, which serves as a small locator view.
Both retinal mosaics, spike history and the strongest antennal inputs stay visible.
The default **live** view shows populations, walking and flight readouts, body commands,
pose and metabolism together, without scrolling at the default size.

![The room console](demo_ui.png)

```powershell
python scripts/room_demo.py --cuda-graphs --cuda-kernels --event-driven --cuda-sparse warp --cam-scale 2
```

The backend flags retain their existing meaning. The header shows backend, integration
steps, weight precision and the CUDA-graph setting alongside simulation time and step
count. The presentation layer also supports plain Torch and Metal. It uses installed
Cascadia Mono, Consolas, DejaVu Sans Mono or Liberation Mono; no downloaded font or new
dependency is required.

## Explore

The toolbar exposes pause, reset, teleport, looming stimuli, giant-fibre/flight stimulation,
save and load. Hover for a short explanation. Existing keyboard shortcuts still work.

| Control | Action |
|---|---|
| Space | Pause / resume |
| R / T | Reset the fly's pose / move next to the apple |
| L / F / W | Loom / giant fibre / flight DN stimulation |
| Drag / scroll over habitat | Orbit / zoom |
| Arrows / + / − | Orbit / zoom with the keyboard |
| C / Home | Follow the fly / restore the overview camera |
| 1 / 2 / 3 / 4 | Live telemetry / full motor list / all senses / brain map |
| V | Cycle retinal both / false colour / R1–R6 contrast |
| N | Brain map: neural activity / live NT levels |
| F5 / F9 / S | Quick-save / quick-load / timestamped save |
| ? or H | Controls guide; pauses and restores the previous playback state when closed |
| Escape | Close the guide, or exit when it is closed |

Scroll telemetry if extra program readouts or a smaller window require it. The **motor**
view gives extended names and values the full width. **Senses** shows contact, wind and
every bilateral antennal odor concentration, sorted by L+R strength. The permanent
antennae table shows the strongest entries, with an explicit shown/total count and L−R.

Population values are mean firing rates in Hz. The graded optic lobe instead shows mean
absolute change from its operating point as a percentage of its unit rate range.
Motor labels use the model's own keys; program gates and x10 diagnostics have no Hz unit.

**Map** lazily loads soma positions, showing dorsal and lateral activity side by side,
with population rates and the most active cell types below. `--brain-map` opens this
view at launch; `4` selects it and `1` returns to live telemetry. Hidden maps do not
sample the brain. The brain map never adds an extra column or hides the sensory views.

**NT levels** displays live data from an optional `NTSource`, with per-channel units,
fixed colour ranges, timestamps and mean/max readouts. Click a channel to select it.
`--brain-map-mode nt` opens this view but does not enable NT dynamics. With no NT
module/readout, it reports unavailable and the activity view still works. Labels and
firing rates are never used as substitute concentrations. Integration details and
missing-data semantics are in [NT_READOUT.md](NT_READOUT.md).

![The toggleable brain atlas](demo_ui_atlas.png)

The body camera is a display in human-visible colour. Compound-eye colour uses the actual
four-band sensory radiance, with UV mapped to magenta, blue to blue and green to green.
Contrast is shown alongside it by default, with OFF dark and ON light. Either view can
also use the full retinal display width. The spike chart contains the
last LIF step sampled every 10 ms frame, over a four-second window.

## Window, captures and timing

The layout uses the available window size at native resolution above 1100x720. Smaller
windows scale the minimum canvas uniformly; pointer hit testing uses that same transform.
Text has explicit width budgets, and telemetry clips to its own scrolling viewport.
The closer initial orbit and camera pixel sizes affect only the display, not sensory
sampling or neural/body parameters.

Paused eye images and the static room background are cached. Camera/geometry changes
refresh the room image, including the moving loom. Atlas fading advances with simulation
frames, so a paused brain does not fade just because the UI redraws. The guide consumes
input, and save/load actions give visible success or error feedback.

```powershell
python scripts/room_demo.py --headless --seconds 2 --screenshot out/console.png
python scripts/room_demo.py --brain-map --window 1920x1080
python scripts/room_demo.py --headless --seconds 2 --gif out/console.gif
```

Screenshots save the final canvas; GIF frames retain their initial dimensions even if the
window is resized during recording. Headless mode still renders the UI. `profile_room.py`
continues to wrap the real demo's `draw` function. UI timings from the older 1280x760
prototype are not directly comparable to the new layout and cameras.

Validation uses the real pygame event loop with deterministic display data to check
resized click targets, telemetry scrolling, the three retinal modes, modal playback,
save/load dispatch, redraw caching, and paused map stability. A separate check verifies
that the default view exposes core population, walking, flight and body readouts inside
the visible viewport, without scrolling, and draws both retinal mosaics.
Real-connectome renders were inspected at 1100x720, 1360x820 and 1920x1080. Neural and
body stepping are unchanged by the console revision; timings on the shared machine
are not used as throughput benchmarks.
The six UI tests pass, including NT sampling/selection and launch without a module.
Five NT contract/projection tests and eighteen control-surface tests pass. The NT
layout was inspected with explicitly labelled synthetic levels at all three sizes;
the existing activity projection remains pixel-identical on the full connectome.
A live Windows run exported a 25-frame 1360x820 GIF, and the
existing profiler completed a Windows-display smoke run with drawing, presentation
and CUDA frame timing intact.
