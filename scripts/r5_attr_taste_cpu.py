"""Round 5: which Brain-side entries carry taste.MN9 / smell.KC_active -- benchmark.py's own taste and smell
protocols on the CPU, over more hold tables than one cluster batch can afford.

    PYTHONIOENCODING=utf-8 python scripts/r5_attr_taste_cpu.py --seeds 0,1,2 --json out/r5_attr_taste_cpu.json

sec_taste and sec_smell build brain.Brain on the whole connectome with no optic lobe, so they run in seconds on the
CPU; this reproduces them verbatim (sweet labellar-bristle / taste-peg GRNs at 100 Hz for 600 ms; the apple plume at
the benchmark's coordinates for 800 ms) under every table scripts/build_hold_tables.py can build, including the two
that split the 3,832 Brain-side entries into their transmitter groups:

    holdBrainGlu  holds the 3,709 glutamate flips (= the KC and DN1 groups together, which round 4 only ever held
                  one at a time), leaving the 123 Brain-side histamine silencings + the optic side
    holdBrainHis  holds those 123 silencings, leaving the KC / DN1 flips + the optic side

CPU Poisson draws differ from the CUDA ones, so the absolute rates are not the suite's; the contrast between
conditions at a shared seed is what this measures (the GPU numbers are in out/r5_attr_*.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from flyverse import brain, connectome as cn, olfaction              # noqa: E402
import build_hold_tables as bht                                     # noqa: E402

CONDS = ["off", "default", "holdBrain", "holdOptic", "holdBrainGlu", "holdBrainHis", "holdKC", "holdDN1"]


def params(cond: str, out_dir: str):
    if cond == "off":
        return brain.LIFParams(receptor_model=None)
    if cond == "default":
        return brain.LIFParams(receptor_model="sign", receptor_net_rule="abs")
    g = cond[len("hold"):]
    p = os.path.join(out_dir, f"receptors_hold{g}.csv")
    if not os.path.isfile(p):
        header, t, n = bht.hold_table(g)
        bht.write_atomic(p, header + t.to_csv(index=False, lineterminator="\n"))
    return brain.LIFParams(receptor_model="sign", receptor_net_rule="abs", receptor_table=p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    conds = [x for x in a.conds.split(",") if x]
    c = cn.load(verbose=False)
    n = c.neurons
    taste = pd.read_csv(os.path.join(ROOT, "flyverse", "data", "taste_grns.csv"))
    sweet = c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
    sweet = sweet[np.isin(n.subclass.to_numpy()[sweet], ["labellar bristle", "taste peg"])]
    mn9, gng175 = c.select(type="MN9"), c.select(type="GNG175")
    pn = c.select(type="~_l2PN|_adPN|_lPN|_lvPN|_ilPN"); kc = c.select(type="~^KC")
    olf = olfaction.Olfaction(c, [("apple", (0.25, 0.15, 0.79), 1.0)])
    res = {}
    for cond in conds:
        p = params(cond, a.out_dir)
        r = cn.receptor_signs(c, table_path=p.receptor_table, net_rule="abs") if p.receptor_model else None
        changed = int((r.fast_sign != np.sign(c.W.data)).sum()) if r is not None else 0
        res[cond] = {"changed_entries": changed, "table": p.receptor_table, "seeds": {}}
        for s in seeds:
            t0 = time.time()
            b = brain.Brain(c, p, device="cpu", seed=s)
            b.set_poisson(sweet, 100.0); b.run_ms(600); rt = b.rate_np()
            d = {"MN9_hz": float(rt[mn9].mean()), "GNG175_hz": float(rt[gng175].mean()),
                 "frac_active": float((rt > 1).mean())}
            del b
            b = brain.Brain(c, p, device="cpu", seed=s)
            olf.apply(b, (0.19, 0.15, 0.75)); b.run_ms(800); rt = b.rate_np()
            d.update({"PN_hz": float(rt[pn].mean()), "KC_hz": float(rt[kc].mean()),
                      "KC_active": int((rt[kc] > 1).sum())})
            del b
            d["wall_s"] = round(time.time() - t0, 1)
            res[cond]["seeds"][s] = d
            print(f"{cond:14s} seed {s}  changed {changed:6d}  MN9 {d['MN9_hz']:7.4f}  GNG175 {d['GNG175_hz']:7.2f}  "
                  f"PN {d['PN_hz']:6.3f}  KC_active {d['KC_active']:5d}  ({d['wall_s']}s)")
    print("\n-- summary (mean over seeds) --")
    print(f"{'condition':14s} {'changed':>8s} {'MN9_hz':>9s} {'GNG175':>8s} {'PN_hz':>8s} {'KC_active':>10s}")
    for cond in conds:
        v = res[cond]["seeds"]
        m = lambda k: np.mean([v[s][k] for s in seeds])
        rng = lambda k: f"{min(v[s][k] for s in seeds):.4g}-{max(v[s][k] for s in seeds):.4g}"
        print(f"{cond:14s} {res[cond]['changed_entries']:8d} {m('MN9_hz'):9.4f} {m('GNG175_hz'):8.2f} "
              f"{m('PN_hz'):8.3f} {m('KC_active'):10.1f}   [MN9 {rng('MN9_hz')}; KC_active {rng('KC_active')}]")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
