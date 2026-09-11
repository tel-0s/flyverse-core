"""Does any population already carry odour-gated wind steering?  (flyverse/screen.py, by side)

The anemotaxis program multiplies the wind-direction DNs' left-right asymmetry by an odour gate. If the
full model has a circuit that does this, some population's L - R asymmetry should flip with wind side
*more strongly when odour is present*. Records every population matching --pattern by soma side at a
fruit site and a plume-free site with the wind on the fly's left or right (pose pinned), and ranks
types by the interaction (odour x wind-side) d', alongside the pure wind-side and odour effects.

    python scripts/screen_steering.py [--seconds 30] [--pattern "^(DN|LAL|PFL|CL|AVLP|WED)"]
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
from screen_odour import pinned_step  # noqa: E402

# wind blows towards -x (comes from +x). Heading -90 (facing -y): +x is on the fly's LEFT; heading +90: on the RIGHT
SITES = {"fruit_windL": (-0.14, 0.30, 0.75, -90), "fruit_windR": (-0.14, 0.30, 0.75, 90),
         "clean_windL": (0.55, 0.35, 0.75, -90), "clean_windR": (0.55, 0.35, 0.75, 90)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--pattern", default=r"^(DN[a-z]|MDN|LAL|PFL|PFR|CL0|AVLP|WED|LH|MBON)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="out/screen_steering.csv")
    args = ap.parse_args()
    runs = {}; rec = None
    for name, (x, y, z, hd) in SITES.items():
        sim = rd.Sim(0, start=(x, y, z), trail_seconds=0.0)
        rec = rec or screen.TypeRecorder.build(sim.c, pattern=args.pattern, by_side=True)
        runs[name] = screen.record(pinned_step(sim, x, y, np.deg2rad(hd)), rec, args.seconds)
        print(f"{name}: done")
    skip = 300
    S = {k: screen._smooth(v, 1.0)[skip:] for k, v in runs.items()}
    rows = []
    for t, iL, iR in screen.lateral_pairs(rec):
        asym = {k: S[k][:, iL] - S[k][:, iR] for k in S}
        m = {k: v.mean() for k, v in asym.items()}
        sd = np.sqrt(np.mean([v.std() ** 2 for v in asym.values()])) + 1.0
        wind_fruit = m["fruit_windL"] - m["fruit_windR"]; wind_clean = m["clean_windL"] - m["clean_windR"]
        odour = 0.5 * (S["fruit_windL"][:, [iL, iR]].mean() + S["fruit_windR"][:, [iL, iR]].mean()) - 0.5 * (S["clean_windL"][:, [iL, iR]].mean() + S["clean_windR"][:, [iL, iR]].mean())
        rows.append({"type": t, "cells": int(rec.n_cells[iL] + rec.n_cells[iR]), "interaction_d": (wind_fruit - wind_clean) / sd,
                     "wind_flip_fruit_hz": wind_fruit, "wind_flip_clean_hz": wind_clean, "odour_mean_hz": odour,
                     "rate_hz": float(np.mean([S[k][:, [iL, iR]].mean() for k in S]))})
    df = pd.DataFrame(rows).sort_values("interaction_d", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); df.to_csv(args.out, index=False)
    fmt = lambda v: f"{v:.2f}"
    print("\nstrongest odour x wind-side interaction (the gated steering signal, if the brain has one):")
    print(df.head(args.top).to_string(index=False, float_format=fmt))
    print("\nstrongest pure wind-side flip (clean site):")
    print(df.reindex(df.wind_flip_clean_hz.abs().sort_values(ascending=False).index).head(8).to_string(index=False, float_format=fmt))
    print("\nreference types:")
    print(df[df.type.isin(["DNa02", "DNp18", "DNp33", "DNg05_a", "DNge016", "LAL010", "PFL3"])].to_string(index=False, float_format=fmt))
    print(f"\n{len(rows)} bilateral types; full table in {args.out}")


if __name__ == "__main__":
    main()
