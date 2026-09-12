"""Skeptic test of the round-5 claim "the fan-in-normalisation hypothesis for the taste rise is REFUTED".

    PYTHONIOENCODING=utf-8 python scripts/skeptic_attr_fanin_test.py --seeds 0,1,2 --json out/sk_attr_fanin.json

The reported attribution shows taste.MN9 is set by the 123 Brain-side histamine silencings and by nothing else
(holdBrainHis == off, holdBrainGlu == default, bit-exact).  It also shows those silencings are the only receptor
change in the whole graph that moves an `input_scale` (7 cells: OA-AL2i1/2, OA-VUMa1, AVLP476, DNge138 x2,
DNge149), and that their presynaptic partners are photoreceptors (R8p / R8y / R8_unclear / HBeyelet), which
sec_taste never drives.  That makes fan-in normalisation ON THOSE SEVEN CELLS the obvious candidate mechanism,
not a refuted one -- the report's refutation only rules out fan-in from the 44,463 OPTIC-side entries.

The discriminating run: repeat sec_taste's protocol with input_norm_alpha = 0 (Brain.__init__ then skips the
fan-in block entirely, brain.py "if p.input_norm_alpha > 0").  If the off -> default taste difference survives
alpha = 0, fan-in is refuted as the mechanism; if it vanishes while alpha = 1 keeps it, fan-in IS the mechanism.
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

from flyverse import brain, connectome as cn                          # noqa: E402
import build_hold_tables as bht                                       # noqa: E402

CONDS = ["off", "default", "holdBrainGlu", "holdBrainHis"]


def params(cond: str, out_dir: str, alpha: float):
    kw = dict(input_norm_alpha=alpha)
    if cond == "off":
        return brain.LIFParams(receptor_model=None, **kw)
    if cond == "default":
        return brain.LIFParams(receptor_model="sign", receptor_net_rule="abs", **kw)
    g = cond[len("hold"):]
    p = os.path.join(out_dir, f"receptors_hold{g}.csv")
    if not os.path.isfile(p):
        header, t, n = bht.hold_table(g)
        bht.write_atomic(p, header + t.to_csv(index=False, lineterminator="\n"))
    return brain.LIFParams(receptor_model="sign", receptor_net_rule="abs", receptor_table=p, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alphas", default="1.0,0.0")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    conds = [x for x in a.conds.split(",") if x]
    c = cn.load(verbose=False)
    n = c.neurons
    taste = pd.read_csv(os.path.join(ROOT, "flyverse", "data", "taste_grns.csv"))
    sweet = c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
    sweet = sweet[np.isin(n.subclass.to_numpy()[sweet], ["labellar bristle", "taste peg"])]
    mn9, gng175 = c.select(type="MN9"), c.select(type="GNG175")
    res = {}
    for alpha in alphas:
        for cond in conds:
            p = params(cond, a.out_dir, alpha)
            r = cn.receptor_signs(c, table_path=p.receptor_table, net_rule="abs") if p.receptor_model else None
            changed = int((r.fast_sign != np.sign(c.W.data)).sum()) if r is not None else 0
            key = f"alpha{alpha:g}/{cond}"
            res[key] = {"changed_entries": changed, "input_norm_alpha": alpha, "seeds": {}}
            for s in seeds:
                t0 = time.time()
                b = brain.Brain(c, p, device="cpu", seed=s)
                b.set_poisson(sweet, 100.0); b.run_ms(600); rt = b.rate_np()
                d = {"MN9_hz": float(rt[mn9].mean()), "GNG175_hz": float(rt[gng175].mean()),
                     "frac_active": float((rt > 1).mean()), "wall_s": round(time.time() - t0, 1)}
                del b
                res[key]["seeds"][s] = d
                print(f"{key:24s} seed {s}  changed {changed:6d}  MN9 {d['MN9_hz']:8.4f}  "
                      f"GNG175 {d['GNG175_hz']:7.2f}  frac {d['frac_active']:.4f}  ({d['wall_s']}s)")
    print("\n-- MN9_hz per seed --")
    print(f"{'condition':24s} " + " ".join(f"{'seed %d' % s:>10s}" for s in seeds) + "      mean")
    for key in res:
        v = res[key]["seeds"]
        print(f"{key:24s} " + " ".join(f"{v[s]['MN9_hz']:10.4f}" for s in seeds) +
              f" {np.mean([v[s]['MN9_hz'] for s in seeds]):9.4f}")
    print("\n-- default - off per seed (the 'taste rise') --")
    for alpha in alphas:
        o, d = res[f"alpha{alpha:g}/off"]["seeds"], res[f"alpha{alpha:g}/default"]["seeds"]
        print(f"alpha {alpha:g}: " + "  ".join(f"seed {s} {d[s]['MN9_hz'] - o[s]['MN9_hz']:+.4f}" for s in seeds))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
