"""Which populations report a nearby object's side?  (flyverse/screen.py, by side)

A fly that has surged to within a few centimetres of a fruit runs past it: upwind is no longer the
direction to the source. Flies fixate and approach dark objects; does this model carry an
object-position signal? Pins the fly on the single-apple table with the apple 5 cm ahead-left, then
ahead-right, then absent (plume-free spot, same headings), and ranks every bilateral population by
the flip of its L - R asymmetry between left and right, against its L - R with no object.

    python scripts/screen_object.py [--seconds 20] [--pattern "^(DN|LC|LPLC|LT|MeTu|AOTU|LAL|PVLP|AVLP)"]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

pygame.init(); pygame.display.set_mode((64, 64))
sys.path.insert(0, os.path.dirname(__file__))
import room_demo as rd  # noqa: E402
from flyverse import screen  # noqa: E402

APPLE = np.array([0.25, 0.15])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--dist", type=float, default=0.05, help="m from the apple's surface")
    ap.add_argument("--pattern", default=r"^(DN[a-z]|MDN|LC|LPLC|LT|MeTu|AOTU|LAL|PVLP|AVLP|PLP|CL0)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="out/screen_object.csv")
    args = ap.parse_args()
    r = 0.04 + args.dist
    # the fly faces +y; the apple is at 45 deg ahead-left or ahead-right
    sites = {}
    for name, ang in [("apple_left", np.deg2rad(135)), ("apple_right", np.deg2rad(45))]:
        pos = APPLE - r * np.array([np.cos(ang), np.sin(ang)])          # apple is at +ang from the fly's position... place the fly so the apple sits ahead-left / ahead-right
        sites[name] = (float(pos[0]), float(pos[1]), np.pi / 2)
    sites["none_a"] = (0.55, 0.35, np.pi / 2); sites["none_b"] = (0.55, -0.35, np.pi / 2)
    runs = {}; rec = None
    for name, (x, y, hd) in sites.items():
        sim = rd.Sim(0, start=(x, y, 0.75), trail_seconds=0.0, fruit_set="apple", fence=True, wind_speed=0.0)
        rec = rec or screen.TypeRecorder.build(sim.c, pattern=args.pattern, by_side=True)
        def step(sim=sim, x=x, y=y, hd=hd):
            sim.fly.place(x, y, 0.75, heading=hd); sim.step(); return sim.fb.brain.rate_np()
        runs[name] = screen.record(step, rec, args.seconds)
        d = sim.nearest_fruit()[1]
        print(f"{name}: done (apple {d * 100:.1f} cm away)")
    S = {k: screen._smooth(v, 1.0)[300:] for k, v in runs.items()}
    rows = []
    for t, iL, iR in screen.lateral_pairs(rec):
        asym = {k: S[k][:, iL] - S[k][:, iR] for k in S}
        m = {k: float(v.mean()) for k, v in asym.items()}
        sd = float(np.sqrt(np.mean([v.std() ** 2 for v in asym.values()]))) + 1.0
        flip = m["apple_left"] - m["apple_right"]; none = 0.5 * (m["none_a"] + m["none_b"])
        rows.append({"type": t, "cells": int(rec.n_cells[iL] + rec.n_cells[iR]), "flip_d": flip / sd, "flip_hz": flip,
                     "left_LR": m["apple_left"], "right_LR": m["apple_right"], "none_LR": none,
                     "rate_hz": float(np.mean([S[k][:, [iL, iR]].mean() for k in S]))})
    df = pd.DataFrame(rows); df = df.reindex(df.flip_d.abs().sort_values(ascending=False).index).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); df.to_csv(args.out, index=False)
    print(f"\nstrongest object-side signal (L - R with the apple ahead-left minus ahead-right, {args.dist * 100:.0f} cm away):")
    print(df.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nreference:", df[df.type.isin(["DNa02", "DNp09", "LC10a", "LC10", "LC16", "DNa03", "DNa01"])].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
