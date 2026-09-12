"""Shiu et al. 2024's sugar / bitter result on MaleCNS: sugar GRNs -> MN9 (proboscis extension) fires;
adding bitter GRNs suppresses it. Runs the LIF alone (no senses, no body), driving the labellar sweet
set from flyverse/data/taste_grns.csv at `--rate` Hz with and without the bitter set, under this
project's calibrated parameters and under Shiu's pure rules (uniform 0.275 mV synapses: no adaptation,
no connection cap, no same-type damping, no fan-in scaling).

    python scripts/probe_bitter.py [--rate 100] [--ms 1500]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import brain, connectome  # noqa: E402

SECOND_ORDER = {"Usnea": r"^Usnea", "Rattle": r"^Rattle", "Phantom": r"^Phantom", "G2N-1": r"^G2N", "Bract": r"^Bract", "Bitter": r"^Bitter", "Scapula": r"^Scapula"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--ms", type=float, default=1500.0)
    ap.add_argument("--receptor-model", default="off", choices=["off", "sign", "sign+gain", "full"],
                    help="LIFParams.receptor_model (default off = the presynaptic NT_SIGN rule)")
    ap.add_argument("--receptor-net-rule", default="class", choices=["class", "abs", "nonmda"])
    args = ap.parse_args()
    receptor = dict(receptor_model=None if args.receptor_model == "off" else args.receptor_model,
                    receptor_net_rule=args.receptor_net_rule)
    c = connectome.load(verbose=False)
    if receptor["receptor_model"]:
        rs = connectome.receptor_signs(c, net_rule=args.receptor_net_rule)
        cov = rs.coverage(c.W)
        m = cov[cov.tier == "matched"].iloc[0]
        print(f"receptor model {args.receptor_model} ({args.receptor_net_rule}): matched {m.edges:,} edges = {m.edges_frac:.1%}, "
              f"{m.syn_W:,.0f} |W| synapses = {m.syn_W_frac:.1%}; fast sign changed on "
              f"{int((rs.fast_sign != np.sign(c.W.data)).sum()):,} entries")
    import torch
    print(f"device: {brain.resolve(None)}; cuda available: {torch.cuda.is_available()}")
    table = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "flyverse", "data", "taste_grns.csv"))
    ids = {k: table.bodyId[table.taste == k].to_numpy() for k in ("sweet", "bitter")}
    sweet = c.index_of(ids["sweet"][np.isin(ids["sweet"], c.neurons.bodyId)])
    bitter = c.index_of(ids["bitter"][np.isin(ids["bitter"], c.neurons.bodyId)])
    mn9 = c.select(type="MN9")
    inst = c.neurons.instance.fillna("").to_numpy(); ty = c.neurons.type.fillna("").to_numpy()
    second = {}
    for name, pat in SECOND_ORDER.items():
        idx = np.flatnonzero(pd.Series(inst).str.match(pat).to_numpy() | pd.Series(ty).str.match(pat).to_numpy())
        if len(idx):
            second[name] = idx
    print(f"sweet GRNs {len(sweet)}, bitter GRNs {len(bitter)}, MN9 {len(mn9)}; second-order sets: " + str({k: len(v) for k, v in second.items()}))
    settings = {
        "this project (calibrated)": brain.LIFParams(**receptor),
        "Shiu et al. rules (uniform 0.275 mV)": brain.LIFParams(adapt_jump=0.0, conn_cap=0.0, same_type_gain=1.0, input_norm_alpha=0.0,
                                                               std_u_by_type={}, path_gain=[], type_path_gain=[], **receptor),
    }
    steps = int(args.ms / 0.5)
    for label, p in settings.items():
        for cond, drive in [("sugar", {"sweet": args.rate}), ("sugar + bitter", {"sweet": args.rate, "bitter": args.rate}), ("bitter", {"bitter": args.rate})]:
            b = brain.Brain(c, p, seed=0)
            for k, hz in drive.items():
                b.set_poisson(sweet if k == "sweet" else bitter, hz)
            b.step(steps // 3)                                        # let it settle, then measure the last two thirds
            acc = {k: 0.0 for k in ["MN9", *second]}; n = 0
            for _ in range(20):
                b.step(steps // 30); r = b.rate_np(); n += 1
                acc["MN9"] += r[mn9].mean()
                for k, idx in second.items():
                    acc[k] += r[idx].mean()
            print(f"{label:38s} {cond:16s}: MN9 {acc['MN9'] / n:5.1f} Hz | " + " ".join(f"{k} {v / n:5.1f}" for k, v in acc.items() if k != "MN9") + f" | spikes/step {float(b.total_spikes()):.0f}")


if __name__ == "__main__":
    main()
