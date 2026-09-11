"""Which cell types report fruit odour, independent of heading?  (flyverse/screen.py on the room demo)

Records every population matching --pattern at fruit and plume-free sites, facing into and away from
the wind, with the pose pinned, and ranks them by d' between the worst fruit condition and the best
plume-free one. This is the screen that found body.LH_ODOUR_TYPES (NOTES, session 8).

    python scripts/screen_odour.py [--seconds 30] [--pattern "^(LH|MBON|DN[a-z]|WED|AVLP|SLP|LAL)"] [--top 20]
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
from flyverse import screen  # noqa: E402

SITES_ALL = {  # name: (x, y, z, heading_deg); the wind blows towards -x, so heading 0 faces into it
    "blueberry_into_wind": (-0.14, 0.30, 0.75, 0), "blueberry_away": (-0.14, 0.30, 0.75, 180),
    "apple_into_wind": (0.12, 0.15, 0.75, 0),
    "clean_into_wind": (0.55, 0.35, 0.75, 0), "clean_away": (0.55, 0.35, 0.75, 180), "floor_away": (0.8, 0.8, 0.0, 180),
}
SITES_APPLE = {  # a single apple at (0.25, 0.15): its plume runs towards -x
    "apple8_into_wind": (0.17, 0.15, 0.75, 0), "apple8_away": (0.17, 0.15, 0.75, 180), "apple40_into_wind": (-0.15, 0.15, 0.75, 0),
    "clean_into_wind": (0.55, 0.35, 0.75, 0), "clean_away": (0.55, 0.35, 0.75, 180), "floor_away": (0.8, 0.8, 0.0, 180),
}
SETS = {"all": (SITES_ALL, ["blueberry_into_wind", "blueberry_away", "apple_into_wind"], ["clean_into_wind", "clean_away", "floor_away"]),
        "apple": (SITES_APPLE, ["apple8_into_wind", "apple8_away", "apple40_into_wind"], ["clean_into_wind", "clean_away", "floor_away"])}


def pinned_step(sim, x, y, heading):
    def step():
        sim.fly.x, sim.fly.y, sim.fly.heading = x, y, heading
        sim.fly.speed = 0.0; sim.fly.yaw_rate = 0.0
        sim.step()
        return sim.fb.brain.rate_np()
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--pattern", default=r"^(LH|MBON|PPL1|PAM|DN[a-z]|MDN|WED|AVLP|PLP|SLP|SIP|SMP|CRE|LAL)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fruit", default="all", choices=["all", "apple"])
    args = ap.parse_args()
    SITES, POSITIVE, NEGATIVE = SETS[args.fruit]
    args.out = args.out or f"out/screen_odour_{args.fruit}.csv"
    runs = {}; rec = None
    for name, (x, y, z, hd) in SITES.items():
        sim = rd.Sim(0, start=(x, y, z), trail_seconds=0.0, fruit_set=args.fruit)
        rec = rec or screen.TypeRecorder.build(sim.c, pattern=args.pattern)
        runs[name] = screen.record(pinned_step(sim, x, y, np.deg2rad(hd)), rec, args.seconds)
        print(f"{name}: done")
    table = screen.rank(runs, POSITIVE, NEGATIVE, rec)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); table.to_csv(args.out, index=False)
    cols = ["key", "cells", "d_prime", "pos_min_hz", "neg_max_hz", "above_thr_pos", "above_thr_neg"] + list(SITES)
    print(table[cols].head(args.top).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n{len(rec.keys)} populations screened; full table in {args.out}")


if __name__ == "__main__":
    main()
