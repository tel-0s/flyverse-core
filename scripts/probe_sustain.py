"""Sustain test: can the fly keep itself fed? Runs the demo Sim headless for minutes and logs energy,
meals, mode (searching / surging / casting / feeding) and distance to fruit every 10 s.

    python scripts/probe_sustain.py --minutes 5 [--seed 0] [--start=-0.15,0.15]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

pygame.init(); pygame.display.set_mode((64, 64))
sys.path.insert(0, os.path.dirname(__file__))
import room_demo as rd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=str, default="-0.15,0.15")
    ap.add_argument("--energy", type=float, default=0.4)
    ap.add_argument("--program", default="none")
    ap.add_argument("--escape-gating", action="store_true")
    ap.add_argument("--fruit", default="all", choices=["all", "apple"])
    ap.add_argument("--fence", action="store_true")
    args = ap.parse_args()
    x, y = [float(v) for v in args.start.split(",")]
    sim = rd.Sim(args.seed, start=(x, y, 0.75), trail_seconds=0.0, program=args.program, escape_gating=args.escape_gating,
                 fruit_set=args.fruit, fence=args.fence)
    print(f"program {args.program}, escape gating {args.escape_gating}, fruit {args.fruit}, fence {args.fence}")
    sim.fly.heading = np.random.default_rng(args.seed).uniform(-np.pi, np.pi)
    sim.metabolism.energy = args.energy
    n = int(args.minutes * 60 * 100)
    modes = {}; hops = 0; was = False; min_energy = 1.0; path = 0.0; last = np.array([sim.fly.x, sim.fly.y])
    print("  t(s)  energy  meals  state     mode       dist_cm  pos")
    for k in range(n):
        sim.step()
        mb = sim.metabolism
        mode = "feeding" if getattr(sim, "feeding", False) else sim.cmd.get("mode", "plain")
        modes[mode] = modes.get(mode, 0) + 1
        min_energy = min(min_energy, mb.energy)
        if sim.fly.airborne and not was: hops += 1
        was = sim.fly.airborne
        cur = np.array([sim.fly.x, sim.fly.y]); path += np.linalg.norm(cur - last); last = cur
        if k % 1000 == 999:
            print(f"{(k + 1) / 100:6.0f}  {mb.energy:6.2f}  {mb.meals:5d}  {mb.state:8s}  {mode:9s}  {sim.nearest_fruit()[1] * 100:6.1f}  ({sim.fly.x:+.2f},{sim.fly.y:+.2f},{sim.fly.z:.2f})")
    tot = sum(modes.values())
    print(f"\nsummary: {args.minutes:.0f} min, meals {mb.meals}, final energy {mb.energy:.2f} (min {min_energy:.2f}), hops {hops}, path {path:.2f} m, "
          + ", ".join(f"{m} {v / tot * 100:.0f}%" for m, v in sorted(modes.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
