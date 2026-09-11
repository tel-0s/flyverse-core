"""Foraging test: the fly starts downwind of the apple; does it get there? Runs the full demo Sim
headless for `--seconds` and reports distance to the nearest fruit over time, with the wind-gated
anemotaxis term on (default) or off (--no-wind-term), for several seeds.

    python scripts/probe_foraging.py --seeds 3 --seconds 30
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


def run(seed, seconds, wind_term, start, wind_speed):
    sim = rd.Sim(seed, start=start, trail_seconds=0.0, wind_speed=wind_speed, program="anemotaxis", escape_gating=True)
    if not wind_term:
        sim.program.k_wind = 0.0; sim.program.wind_speed_bonus = 0.0
    sim.fly.heading = np.random.default_rng(seed).uniform(-np.pi, np.pi)
    d0 = sim.nearest_fruit()[1]
    dist = []; tasted = 0; hops = 0; was_air = False
    for k in range(int(seconds * 100)):
        sim.step()
        if k % 100 == 0:
            dist.append(sim.nearest_fruit()[1] * 100)
        tasted += sim.tasting
        if sim.fly.airborne and not was_air:
            hops += 1
        was_air = sim.fly.airborne
    return d0 * 100, dist, tasted, hops, sim.fly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--start", type=str, default="-0.15,0.15", help="x,y on the table (apple at 0.25,0.15; wind blows towards -x)")
    ap.add_argument("--wind-speed", type=float, default=0.3)
    ap.add_argument("--no-wind-term", action="store_true")
    args = ap.parse_args()
    x, y = [float(v) for v in args.start.split(",")]
    for wind_term in ([False, True] if not args.no_wind_term else [False]):
        print(f"\n=== anemotaxis readout {'ON' if wind_term else 'OFF'}")
        finals = []
        for seed in range(args.seeds):
            d0, dist, tasted, hops, fly = run(seed, args.seconds, wind_term, (x, y, 0.75), args.wind_speed)
            finals.append(min(dist))
            print(f"seed {seed}: start {d0:.0f} cm -> " + " ".join(f"{d:.0f}" for d in dist[::5]) + f" | min {min(dist):.0f} cm, tasted {tasted / 100:.1f} s, hops {hops}, end ({fly.x:+.2f},{fly.y:+.2f},{fly.z:.2f})")
        print(f"   mean closest approach {np.mean(finals):.0f} cm")


if __name__ == "__main__":
    main()
