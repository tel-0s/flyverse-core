"""Diagnostic: stimulate the putative sweet GRNs (flyverse/data/taste_grns.csv) and see whether the
sweet pathway reaches the proboscis motor neurons, and whether anything runs away.

    python scripts/probe_taste.py [--alpha 0.5] [--rate 100] [--group leg|labellar|all]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import brain, connectome  # noqa: E402

PROBE = ["MN9", "MN1", "MN2Da", "MN6", "MN11D", "MN12D", "GNG232", "GNG132", "GNG175", "GNG229", "GNG215", "GNG108",
         "DNge173", "DNg67", "lLN1_bc", "v2LN30", "DNa02", "MDN", "DNp09"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--group", default="labellar")   # the 165 labellar sugar GRNs; "leg" is 3 cells and drives nothing
    ap.add_argument("--ms", type=float, default=1500)
    ap.add_argument("--norm-ref", type=float, default=5000.0)
    ap.add_argument("--norm-alpha", type=float, default=1.0)
    args = ap.parse_args()
    c = connectome.load(verbose=False)
    n = c.neurons
    taste = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "flyverse", "data", "taste_grns.csv"))
    sweet = c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
    sub = n.subclass.to_numpy()[sweet]
    if args.group == "leg":
        sweet = sweet[sub == "leg bristle"]
    elif args.group == "labellar":
        sweet = sweet[np.isin(sub, ["labellar bristle", "taste peg"])]
    print(f"stimulating {len(sweet)} sweet GRNs ({args.group}) at {args.rate} Hz")
    b = brain.Brain(c, brain.LIFParams(input_norm_alpha=args.norm_alpha, input_norm_ref=args.norm_ref))
    b.set_poisson(sweet, args.rate)
    for k in range(int(args.ms / 100)):
        b.run_ms(100)
        rt = b.rate[0].cpu().numpy()
        if k % 3 == 2:
            print(f"  t={b.t:.0f}ms spikes/step {b.total_spikes():.0f} frac>1Hz {(rt > 1).mean():.3f} " +
                  " ".join(f"{t}={rt[c.select(type=t)].mean():.0f}" for t in PROBE))
    b.set_poisson(sweet, 0.0)
    b.run_ms(600)
    rt = b.rate[0].cpu().numpy()
    print(f"  600 ms after stimulus: spikes/step {b.total_spikes():.0f} frac>1Hz {(rt > 1).mean():.3f} " +
          " ".join(f"{t}={rt[c.select(type=t)].mean():.0f}" for t in PROBE))
    g = n.assign(rate=rt).groupby("type").rate.agg(["mean", "size"])
    print("  top after:", g[g["size"] >= 2].sort_values("mean", ascending=False).head(8)["mean"].round(0).to_dict())


if __name__ == "__main__":
    main()
