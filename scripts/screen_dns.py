"""Descending-neuron activation screen, batched: stimulate each DN type (both sides, 150 Hz Poisson for
400 ms) in its own copy of the brain and record what the VNC does -- leg motor neurons per side and
neuromere, wing power / steering MNs, haltere, neck, proboscis. This is a measured motor map for this
model (as opposed to the hypothesised readout in body.py), and finds flight-initiating DNs.

    python scripts/screen_dns.py [--batch 64] [--min-cells 2] [--rate 150]
Writes out/dn_screen.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import body, brain, connectome  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-cells", type=int, default=2)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--ms", type=float, default=400.0)
    ap.add_argument("--superclass", default="descending_neuron")
    args = ap.parse_args()
    c = connectome.load(verbose=False)
    n = c.neurons
    dn = n[(n.superclass == args.superclass) & n.type.notna()]
    counts = dn.type.value_counts()
    dn_types = sorted(counts[counts >= args.min_cells].index.tolist())
    print(f"{len(dn_types)} {args.superclass} types with >= {args.min_cells} cells")
    g = body.motor_groups(c); wg = body.wing_groups(c)
    side = n.somaSide.to_numpy(); neuromere = n.somaNeuromere.fillna("").to_numpy()
    leg = c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"])
    groups = {"legL": leg[side[leg] == "L"], "legR": leg[side[leg] == "R"],
              "T1L": leg[(side[leg] == "L") & (neuromere[leg] == "T1")], "T1R": leg[(side[leg] == "R") & (neuromere[leg] == "T1")],
              "T2L": leg[(side[leg] == "L") & (neuromere[leg] == "T2")], "T2R": leg[(side[leg] == "R") & (neuromere[leg] == "T2")],
              "T3L": leg[(side[leg] == "L") & (neuromere[leg] == "T3")], "T3R": leg[(side[leg] == "R") & (neuromere[leg] == "T3")],
              "power": wg.power, "steerL": wg.steer_L, "steerR": wg.steer_R, "haltere": wg.haltere, "ttm": wg.ttm,
              "neck": c.select(superclass="vnc_motor", subclass="nm"), "abd": c.select(superclass="vnc_motor", subclass="ad"),
              "MN9": g.proboscis, "cb_motor": c.select(superclass="cb_motor"),
              "DNa02L": c.select(type="DNa02", somaSide="L"), "DNa02R": c.select(type="DNa02", somaSide="R"), "MDN": g.back_dn, "GF": wg.gf}
    B = args.batch
    b = brain.Brain(c, batch=B)
    rows = []
    t0 = time.time()
    for start in range(0, len(dn_types), B):
        chunk = dn_types[start:start + B]
        b.reset()
        b.poisson_p.zero_()
        for k, t in enumerate(chunk):
            idx = c.select(type=t)
            b.poisson_p[k, b._idx(idx)] = args.rate * (b.p.dt / 1000.0)
        b._poisson_on = True
        b.run_ms(args.ms)
        rate = b.rate.cpu().numpy()
        spikes_total = b.spikes.sum(1).cpu().numpy()
        for k, t in enumerate(chunk):
            row = {"type": t, "n_cells": int(counts[t]), "spikes_per_step": float(spikes_total[k]),
                   "frac_active": float((rate[k] > 1).mean())}
            for name, idx in groups.items():
                row[name] = float(rate[k, idx].mean()) if len(idx) else 0.0
            rows.append(row)
        print(f"  {start + len(chunk)}/{len(dn_types)} types, {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    df["leg_asym"] = df.legL - df.legR
    df["leg_total"] = (df.legL + df.legR) / 2
    os.makedirs("out", exist_ok=True)
    df.to_csv("out/dn_screen.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["type", "n_cells", "spikes_per_step", "frac_active", "legL", "legR", "power", "steerL", "steerR", "haltere", "neck", "MN9", "ttm"]
    print("\nstrongest wing-power drivers (flight?):"); print(df.sort_values("power", ascending=False)[cols].head(12).round(1).to_string(index=False))
    print("\nstrongest leg drivers:"); print(df.sort_values("leg_total", ascending=False)[cols].head(12).round(1).to_string(index=False))
    print("\nmost lateralised leg drivers (L-R):"); print(df.reindex(df.leg_asym.abs().sort_values(ascending=False).index)[cols + ["leg_asym"]].head(12).round(1).to_string(index=False))
    print("\nproboscis (MN9) drivers:"); print(df.sort_values("MN9", ascending=False)[cols].head(6).round(1).to_string(index=False))
    print("\nbrain-wide activation (frac_active) top:"); print(df.sort_values("frac_active", ascending=False)[cols].head(6).round(2).to_string(index=False))
    for t in ["DNa02", "DNp09", "MDN", "DNa01", "DNp01", "DNg13", "DNb01", "DNp07", "DNa03"]:
        if t in set(df.type):
            print("  ", df[df.type == t][cols].round(1).to_string(index=False, header=False))


if __name__ == "__main__":
    main()
