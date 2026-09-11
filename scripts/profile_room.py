"""Profile the actual room-demo loop, including drawing and presentation.

python scripts/profile_room.py --json out/room.json --cuda-graphs --weight-dtype float16 --fast --cam-scale 4 --trail-seconds 0
Add --trace out/room.trace.json --frames 8 for a separate, instrumented CPU/CUDA trace.
All unrecognized arguments go to room_demo.py; --headless uses its dummy SDL display.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from functools import wraps
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pygame
import torch

import room_demo as demo


class _Finished(Exception):
    pass


def summarize(values):
    return {"count": len(values), "mean": float(np.mean(values)),
            "p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--state", type=Path, help="save final controller/body state for comparisons")
    args, demo_args = ap.parse_known_args()
    if args.frames < 1 or args.warmup < 1:
        ap.error("frames and warmup must be positive")
    for path in (args.json, args.trace, args.state):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap the real event loop instead of maintaining a second simulation/drawing loop.
    # No per-stage GPU synchronization: CPU scopes measure submission plus existing waits;
    # CUDA events separately measure elapsed stream time for the controller frame.
    frame = 0
    frame_start = None
    frame_scope = None
    sim = None
    tracing = False
    profiler = None
    timings = {}
    neural_events = []
    frame_ms = []
    simulated_ms = []
    previous_sim_time = 0.0
    driver = None

    @contextmanager
    def phase(name):
        start = time.perf_counter()
        scope = torch.profiler.record_function(name) if tracing else None
        if scope:
            scope.__enter__()
        try:
            yield
        finally:
            if scope:
                scope.__exit__(None, None, None)
            if frame_start is not None and frame >= args.warmup:
                timings.setdefault(name, []).append((time.perf_counter() - start) * 1000)

    def instrument(fn, name):
        @wraps(fn)
        def wrapped(*pos, **kw):
            with phase(name):
                return fn(*pos, **kw)
        return wrapped

    step = demo.Sim.step

    def sim_step(instance):
        nonlocal sim
        sim = instance
        with phase("room.simulation"):
            return step(instance)

    brain_step = demo.FlyBrain.step

    def neural_step(instance, *pos, **kw):
        events = None
        if frame >= args.warmup and instance.device.type == "cuda":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record(torch.cuda.current_stream(instance.device))
        with phase("brain.frame"):
            result = brain_step(instance, *pos, **kw)
        if events:
            events[1].record(torch.cuda.current_stream(instance.device))
            neural_events.append(events)
        return result

    get_events = pygame.event.get

    def begin_frame(*pos, **kw):
        nonlocal frame_start, frame_scope, driver
        frame_start = time.perf_counter()
        driver = pygame.display.get_driver()
        if tracing:
            frame_scope = torch.profiler.record_function("room.frame")
            frame_scope.__enter__()
        with phase("ui.events"):
            return get_events(*pos, **kw)

    real_clock = pygame.time.Clock

    class FrameClock:
        def __init__(self):
            self.clock = real_clock()

        def tick(self, *pos, **kw):
            nonlocal frame, frame_start, frame_scope, tracing, profiler
            nonlocal previous_sim_time
            with phase("ui.frame_limit"):
                result = self.clock.tick(*pos, **kw)
            if frame_scope:
                frame_scope.__exit__(None, None, None)
                frame_scope = None
            if frame >= args.warmup:
                frame_ms.append((time.perf_counter() - frame_start) * 1000)
                simulated_ms.append(sim.brain.t - previous_sim_time)
            previous_sim_time = sim.brain.t
            frame_start = None
            frame += 1
            if frame == args.warmup:
                sim.fb._synchronize()
                if args.trace:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if sim.fb.device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    profiler = torch.profiler.profile(activities=activities)
                    profiler.start()
                    tracing = True
            if frame >= args.warmup + args.frames:
                raise _Finished()
            return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", ["room_demo.py", *demo_args]))
        for obj, attr, replacement in (
            (demo.Sim, "step", sim_step),
            (demo.FlyBrain, "step", neural_step),
            (pygame.event, "get", begin_frame),
            (pygame.time, "Clock", FrameClock),
        ):
            stack.enter_context(patch.object(obj, attr, replacement))
        for obj, attr, name in (
            (demo.Sim, "column_radiance", "sensor.raytrace"),
            (demo.FlyBrain, "vision", "sensor.vision"),
            (demo.FlyBrain, "smell", "sensor.smell"),
            (demo.FlyBrain, "wind", "sensor.wind"),
            (demo.FlyBrain, "taste", "sensor.taste"),
            (demo.FlyBrain, "motor", "brain.motor"),
            (demo, "draw", "ui.draw"),
            (pygame.display, "flip", "ui.present"),
        ):
            stack.enter_context(patch.object(obj, attr, instrument(getattr(obj, attr), name)))
        try:
            demo.main()
        except _Finished:
            pass
        finally:
            if frame_scope:
                frame_scope.__exit__(None, None, None)
            if sim is not None:
                sim.fb._synchronize()
            if profiler is not None:
                profiler.stop()
                profiler.export_chrome_trace(str(args.trace))
            pygame.quit()

    if not frame_ms:
        raise SystemExit("Demo ended before any measured frames completed.")
    if args.state:
        torch.save({"controller": sim.fb.state_dict(), "pose": vars(sim.fly),
                    "command": sim.cmd, "wing_command": sim.wcmd}, args.state)
    report = {
        "torch": torch.__version__, "device": str(sim.fb.device),
        "gpu": torch.cuda.get_device_name(sim.fb.device) if sim.fb.device.type == "cuda" else None,
        "demo_args": demo_args, "sdl_driver": driver, "neurons": sim.c.n,
        "brain_dt_ms": sim.brain.p.dt, "optic_dt_ms": sim.optic.p.dt_ms,
        "cuda_kernels": sim.brain.cuda, "cuda_sparse": sim.brain.cuda_sparse,
        "cuda_compact": sim.brain.cuda_compact, "event_driven": sim.brain.event_driven,
        "sensory_cuda_graphs": sim.sensory_cuda_graphs,
        "ray_graphs_cached": len(getattr(sim.world, "_trace_graphs", {})),
        "warmup_frames": args.warmup, "measured_frames": len(frame_ms),
        "complete": len(frame_ms) == args.frames, "trace_instrumented": bool(args.trace),
        "frame_wall_ms": summarize(frame_ms),
        "simulated_ms": sum(simulated_ms),
        "real_time_factor": sum(simulated_ms) / sum(frame_ms),
        "cpu_scopes_ms": {name: summarize(values) for name, values in timings.items()},
        "brain_cuda_stream_ms": summarize([a.elapsed_time(b) for a, b in neural_events]) if neural_events else None,
        "notes": "CPU scopes are inclusive and overlap; CUDA events include stream gaps. "
                 "Frame wall time includes drawing, presentation and the demo's frame limiter. "
                 "Headless still draws. Use a separate run without --trace for throughput. "
                 "Leave the demo unpaused and avoid resets, loads or camera changes for matched runs.",
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
