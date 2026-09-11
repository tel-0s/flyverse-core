"""Measure controller frame latency, with subsets, graphs and precision as explicit options.

python scripts/profile_brain.py --frames 100 --cuda-graphs --json out/profile_graphs.json
python scripts/profile_brain.py --modules antennal_lobe,central,descending,vnc --event-driven
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import FlyBrain, body, connectome, regions, world
from flyverse.brain import LIFParams


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modules", default="all", help="comma-separated module names, or all")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--frame-ms", type=float, default=10.)
    ap.add_argument("--dt", type=float, default=.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cuda-graphs", action="store_true")
    ap.add_argument("--event-driven", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--weight-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.batch < 1 or args.frame_ms < args.dt:
        ap.error("frames and batch must be positive, warmup nonnegative, frame-ms >= dt")
    start = time.perf_counter()
    c = connectome.load(verbose=False)
    modules = None if args.modules == "all" else args.modules.split(",")
    fb = FlyBrain(c, modules=modules, batch=args.batch, device=args.device, cuda_graphs=args.cuda_graphs,
                  lif_params=LIFParams(dt=args.dt, event_driven=args.event_driven, weight_dtype=args.weight_dtype))
    if fb.optic is not None:
        fb.optic.diagnostics = False  # UI copies are measured separately by the demo
    initialization = (time.perf_counter() - start) * 1000
    wd = dirs = ray_weights = None
    pose = body.FlyState(x=-.3, y=0)
    if fb.optic is not None:
        wd, _ = world.make_room()
        wd.device = fb.device
        dirs, ray_weights = fb.retina.ray_directions()
        ray_weights = torch.as_tensor(ray_weights, device=wd.device, dtype=torch.float32)
    measurements = {key: [] for key in ("sensory_ms", "step_ms", "motor_ms", "total_ms")}
    startup = time.perf_counter()
    warmup_ms = 0.0
    for frame in range(args.warmup + args.frames):
        fb._synchronize()
        t0 = time.perf_counter()
        if frame == args.warmup:
            warmup_ms = (t0 - startup) * 1000
        if wd is not None:
            pose.heading = frame * .005
            directions = pose.body_to_world(dirs.reshape(-1, 3))
            origin = np.broadcast_to(pose.eye_pos, directions.shape).copy()
            rad = wd.trace(torch.as_tensor(origin, dtype=torch.float32), torch.as_tensor(directions, dtype=torch.float32))
            rad = (rad.reshape(dirs.shape[0], dirs.shape[1], 4) * ray_weights[None, :, None]).sum(1)
            fb.vision(rad)
        if "smell" in fb.available_senses:
            fb.smell({"DM1":.2, "VA2":.1}, {"DM1":.1, "VA2":.05})
        if "wind" in fb.available_senses:
            fb.wind(.5,.2)
        if "taste" in fb.available_senses:
            fb.taste(0)
        fb._synchronize(); t1 = time.perf_counter()
        fb.step(args.frame_ms)
        fb._synchronize(); t2 = time.perf_counter()
        rates = fb.motor()
        t3 = time.perf_counter()
        if frame >= args.warmup:
            for name, value in zip(measurements, (t1-t0,t2-t1,t3-t2,t3-t0)):
                measurements[name].append(value * 1000)
    summary = {name: {"mean":float(np.mean(v)), "p50":float(np.median(v)), "p95":float(np.percentile(v,95))}
               for name,v in measurements.items()}
    report = {"device":str(fb.device), "torch":torch.__version__, "neurons":fb.c.n,
              "stored_edges":fb.c.W.nnz, "modules":modules or list(regions.MODULES), "batch":fb.B,
              "lif":asdict(fb.brain.p), "frame_ms":args.frame_ms, "frames":args.frames,
              "cuda_graphs":args.cuda_graphs, "captured_graphs":len(fb._graphs),
              "initialization_ms":initialization, "timing":summary,
              "warmup_ms":warmup_ms,
              "real_time_factor":args.frame_ms / summary['total_ms']['mean'],
              "warmup_and_measurement_ms":(time.perf_counter()-startup)*1000}
    print(json.dumps(report,indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
