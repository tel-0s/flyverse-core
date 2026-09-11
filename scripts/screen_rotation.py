"""Which populations report yaw rotation, sustained?  (flyverse/screen.py, by side)

The optomotor readout (DNp04 + LPT27/30, L - R) turned out not to flip under sustained rotation while
walking drives it hard (NOTES, session 8). This screen imposes +90 and -90 deg/s yaw on a pinned fly in
the room (no wind) and ranks every bilateral population by the sign flip of its L - R asymmetry between
the two directions, against its L - R at rest -- the signature a course-stabilising reflex needs.

    python scripts/screen_rotation.py [--seconds 10] [--rate 90] [--pattern "^(DN|LPT|HS|VS|H2|LPi|MeVP|Nod)"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=90.0)
    ap.add_argument("--pattern", default=r"^(DN[a-z]|MDN|LPT|HS|VS|H2|LPi|MeVP|Nod|CH|LLPC|LPC)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="out/screen_rotation.csv")
    args = ap.parse_args()
    sim = rd.Sim(0, start=(0.0, 0.0, 0.75), trail_seconds=0.0, wind_speed=0.0)
    rec = screen.TypeRecorder.build(sim.c, pattern=args.pattern, by_side=True)
    state = {"h": 0.0}

    def stepper(rate_dps):
        def step():
            state["h"] += np.deg2rad(rate_dps) * 0.01
            sim.fly.place(0.0, 0.0, 0.75, heading=state["h"])
            sim.step()
            return sim.fb.brain.rate_np()
        return step

    runs = {}
    for name, rate in [("rest", 0.0), ("ccw", args.rate), ("rest2", 0.0), ("cw", -args.rate)]:
        runs[name] = screen.record(stepper(rate), rec, args.seconds)
        print(f"{name}: done")
    skip = 300
    S = {k: screen._smooth(v, 1.0)[skip:] for k, v in runs.items()}
    rows = []
    for t, iL, iR in screen.lateral_pairs(rec):
        asym = {k: S[k][:, iL] - S[k][:, iR] for k in S}
        m = {k: float(v.mean()) for k, v in asym.items()}
        sd = float(np.sqrt(np.mean([v.std() ** 2 for v in asym.values()]))) + 1.0
        flip = m["ccw"] - m["cw"]
        rest = 0.5 * (m["rest"] + m["rest2"])
        rows.append({"type": t, "cells": int(rec.n_cells[iL] + rec.n_cells[iR]), "flip_d": flip / sd, "flip_hz": flip,
                     "ccw_LR": m["ccw"], "cw_LR": m["cw"], "rest_LR": rest, "rate_hz": float(np.mean([S[k][:, [iL, iR]].mean() for k in S]))})
    df = pd.DataFrame(rows)
    df["signed_flip_d"] = df.flip_d
    df = df.reindex(df.flip_d.abs().sort_values(ascending=False).index).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); df.to_csv(args.out, index=False)
    print(f"\nstrongest sustained rotation signal (L - R flips between +{args.rate:.0f} and -{args.rate:.0f} deg/s; sign: + = larger L - R for CCW):")
    print(df.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nreference:", df[df.type.isin(["DNp04", "LPT27", "LPT30", "DNp20", "DNa02"])].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
