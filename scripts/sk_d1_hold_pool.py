"""Skeptic: fold the independent 4th replicate (brain seed 3, env 48-63, all four arms in ONE cluster
submission, run dir sk-d1-hold-f691cf) into the takeoff-hold report's four-arm table and redo every
headline number.  Also the within-batch matched-room analysis of my own batch alone."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"D:\Projects\flyverse")
AUD = {"shipped":   [f"out/r5_adopt_sustain_live_{i}.json" for i in (1, 2, 3)],
       "holdBrain": [f"out/d1_holdBrain_{i}.json" for i in (1, 2, 3)],
       "holdOptic": [f"out/d1_holdOptic_{i}.json" for i in (1, 2, 3)],
       "off":       [f"out/d1_off_{i}.json" for i in (1, 2, 3)]}
MINE = {"shipped": ["out/sk_d1_shipped_4.json"], "holdBrain": ["out/sk_d1_holdBrain_4.json"],
        "holdOptic": ["out/sk_d1_holdOptic_4.json"], "off": ["out/sk_d1_off_4.json"]}
L = lambda ps: [json.loads((ROOT / p).read_text(encoding="utf-8")) for p in ps]
A = {k: L(v) for k, v in AUD.items()}
M = {k: L(v) for k, v in MINE.items()}
P = {k: A[k] + M[k] for k in A}


def rows(ds, key):
    return np.array([r[key] for d in ds for r in d["rows"]], float)


def tot(ds, key):
    return int(rows(ds, key).sum())


for label, D in (("AUDITED 3 batches per arm", A), ("MY 4th batch alone", M), ("POOLED 4 batches per arm", P)):
    print(f"\n===== {label} =====")
    fs = sum(d["fly_s"] for d in D["shipped"])
    hdr = f"{'arm':10s} {'hops':>5s} {'esc':>4s} {'vol':>4s} {'/1000':>7s} {'esc':>7s} {'vol':>7s} {'GFmed':>7s} {'rows>=33':>9s}"
    print(hdr)
    for k in ("shipped", "holdBrain", "holdOptic", "off"):
        h, e, v = (tot(D[k], m) for m in ("hops", "hops_escape", "hops_voluntary"))
        g = rows(D[k], "gf_max_walk_hz")
        print(f"{k:10s} {h:5d} {e:4d} {v:4d} {h/fs*1000:7.3f} {e/fs*1000:7.3f} {v/fs*1000:7.3f} "
              f"{np.median(g):7.2f} {int((g>=33).sum()):4d}/{len(g):<4d}")
    print("  share of the shipped default's excess over off, and the count ratio to the default (Jeffreys 95 %):")
    for m in ("hops", "hops_escape", "hops_voluntary"):
        base, top = tot(D["off"], m) / fs * 1000, tot(D["shipped"], m) / fs * 1000
        for k in ("holdBrain", "holdOptic"):
            val = tot(D[k], m) / fs * 1000
            ka, ks = tot(D[k], m), tot(D["shipped"], m)
            share = (val - base) / (top - base) * 100 if top != base else float("nan")
            if ka + ks:
                lo, hi = stats.beta.ppf([0.025, 0.975], ka + .5, ks - ka + .5)
                r = lambda p: p / (1 - p) if p < 1 else float("inf")
                ci = f"[{r(lo):.2f}, {r(hi):.2f}]x"
                pb = stats.binomtest(ka, ka + ks, 0.5).pvalue
            else:
                ci, pb = "n/a", float("nan")
            print(f"    {m:15s} {k:10s} {val:6.3f} -> {share:7.1f} % of the excess; ratio {ka/max(ks,1):5.3f}x {ci} binom p {pb:.3g}")
    gm = {k: float(np.median(rows(D[k], "gf_max_walk_hz"))) for k in D}
    span = gm["shipped"] - gm["off"]
    for k in ("holdBrain", "holdOptic"):
        print(f"    gf median      {k:10s} {gm[k]:6.2f} (off {gm['off']:6.2f}, shipped {gm['shipped']:6.2f}) -> {(gm[k]-gm['off'])/span*100:7.1f} % of the shift")
    print("  pooled two-sided Mann-Whitney (asymptotic) over rows:")
    for a, b in (("holdBrain", "shipped"), ("holdOptic", "off"), ("holdBrain", "off"), ("holdOptic", "shipped")):
        out = []
        for m in ("hops", "hops_escape", "hops_voluntary", "gf_max_walk_hz"):
            u = stats.mannwhitneyu(rows(D[a], m), rows(D[b], m), alternative="two-sided", method="asymptotic")
            out.append(f"{m} U {u.statistic:.1f} p {u.pvalue:.3g}")
        print(f"    {a:10s} vs {b:10s} " + " | ".join(out))

print("\n===== MY batch: matched rooms (16 env seeds x 4 arms, one submission) =====")
r = {k: {x["environment_seed"]: x for x in M[k][0]["rows"]} for k in M}
ks = sorted(r["shipped"])
print(f"{'env':>4s} " + " ".join(f"{k[:9]:>20s}" for k in ("shipped", "holdBrain", "holdOptic", "off")))
for e in ks:
    print(f"{e:4d} " + " ".join(f"{r[k][e]['hops']:3d}({r[k][e]['hops_escape']}+{r[k][e]['hops_voluntary']}) gf {r[k][e]['gf_max_walk_hz']:5.1f}"
                               for k in ("shipped", "holdBrain", "holdOptic", "off")))
for a, b in (("holdBrain", "shipped"), ("holdOptic", "off"), ("holdBrain", "off")):
    for m in ("hops", "hops_voluntary", "gf_max_walk_hz"):
        x = np.array([r[a][e][m] for e in ks], float); y = np.array([r[b][e][m] for e in ks], float)
        d = x - y; nz = d[d != 0]
        sp = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
        print(f"  paired {a:10s} vs {b:10s} {m:15s} mean d {d.mean():+7.3f} nonzero {len(nz):2d} (+{int((nz>0).sum())}) sign p {sp:.3g}")
