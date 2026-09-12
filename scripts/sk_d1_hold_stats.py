"""Skeptic re-derivation of the takeoff-hold statistics: tie-aware exact permutation p, Kruskal-Wallis,
batch-level (n=3) tests, and the sensitivity of the 'share of the excess' to the 4th shipped-default batch
on record (out/r5_skeptic_sustain_live_4.json, the round-5 skeptic's identical-seed rerun, 11 hops)."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"D:\Projects\flyverse")
ARMS = {
    "shipped":     [f"out/r5_adopt_sustain_live_{i}.json" for i in (1, 2, 3)],
    "shipped+sk":  [f"out/r5_adopt_sustain_live_{i}.json" for i in (1, 2, 3)] + ["out/r5_skeptic_sustain_live_4.json"],
    "holdBrain":   [f"out/d1_holdBrain_{i}.json" for i in (1, 2, 3)],
    "holdOptic":   [f"out/d1_holdOptic_{i}.json" for i in (1, 2, 3)],
    "off_shipped": [f"out/d1_off_{i}.json" for i in (1, 2, 3)],
    "off_damped":  [f"out/r5_sustain_off_live_{i}.json" for i in (1, 2, 3)],
}
M = ("hops", "hops_escape", "hops_voluntary", "gf_max_walk_hz")
data = {k: [json.loads((ROOT / p).read_text(encoding="utf-8")) for p in v] for k, v in ARMS.items()}


def rows(name, key):
    return np.array([r[key] for d in data[name] for r in d["rows"]], float)


def batchvals(name, key):
    if key == "gf_max_walk_hz":
        return [float(np.median([r[key] for r in d["rows"]])) for d in data[name]]
    return [float(sum(r[key] for r in d["rows"])) for d in data[name]]


def perm_p(x, y, n=200000, rng=0):
    """Tie-aware exact-style permutation p on the Mann-Whitney U statistic (two-sided)."""
    r = np.random.default_rng(rng)
    obs = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic
    pool = np.concatenate([x, y]); nx = len(x)
    mid = nx * len(y) / 2
    cnt = 0
    for _ in range(n):
        r.shuffle(pool)
        u = stats.mannwhitneyu(pool[:nx], pool[nx:], alternative="two-sided", method="asymptotic").statistic
        cnt += abs(u - mid) >= abs(obs - mid) - 1e-9
    return (cnt + 1) / (n + 1)


PAIRS = [("holdBrain", "shipped"), ("holdOptic", "shipped"), ("off_shipped", "shipped"),
         ("holdBrain", "off_shipped"), ("holdOptic", "off_shipped"), ("holdBrain", "holdOptic"),
         ("off_shipped", "off_damped"), ("holdBrain", "shipped+sk")]
print("== tie-aware permutation p (20,000 shuffles of the pooled 48+48 rows) vs the report's asymptotic p ==")
for a, b in PAIRS:
    for m in M:
        x, y = rows(a, m), rows(b, m)
        u = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        pp = perm_p(x, y, n=20000)
        print(f"  {a:12s} vs {b:12s} {m:15s} U {u.statistic:7.1f} p_asym {u.pvalue:.3g}  p_perm {pp:.4g}")
    print()

print("== Kruskal-Wallis across an arm's 3 batches (16 rows each) ==")
for name in ("shipped", "holdBrain", "holdOptic", "off_shipped", "off_damped"):
    for m in M:
        g = [[r[m] for r in d["rows"]] for d in data[name]]
        try:
            h = stats.kruskal(*g)
            print(f"  {name:12s} {m:15s} H {h.statistic:6.3f} p {h.pvalue:.4g}")
        except ValueError as e:
            print(f"  {name:12s} {m:15s} n/a ({e})")

print("\n== batch-level values (n=3 per arm), the unit that is actually independent ==")
for m in M:
    print(f"  {m}")
    for name in ("shipped", "holdBrain", "holdOptic", "off_shipped", "off_damped"):
        print(f"     {name:12s} {[round(v,3) for v in batchvals(name, m)]}")

print("\n== batch-level paired tests (3 matched brain seeds / env blocks) ==")
for a, b in PAIRS[:7]:
    for m in M:
        xa, xb = batchvals(a, m), batchvals(b, m)
        d = np.array(xa) - np.array(xb)
        # exact sign-flip permutation on 3 paired differences: 8 sign patterns
        obs = abs(d.sum())
        tot = 0; ge = 0
        for s0 in (1, -1):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    tot += 1
                    ge += abs((d * np.array([s0, s1, s2])).sum()) >= obs - 1e-12
        print(f"  {a:12s} vs {b:12s} {m:15s} diffs {[round(v,3) for v in d]} exact sign-flip p {ge/tot:.3f}")
    print()

print("== share of the shipped default's excess over off, with and without the 4th shipped batch on record ==")
for ref in ("shipped", "shipped+sk"):
    n_ref = len(data[ref])
    for m in ("hops", "hops_escape", "hops_voluntary"):
        base = rows("off_shipped", m).sum() / 14400 * 1000
        top = rows(ref, m).sum() / (4800 * n_ref) * 1000
        for arm in ("holdBrain", "holdOptic"):
            v = rows(arm, m).sum() / 14400 * 1000
            print(f"  ref {ref:11s} {m:15s} {arm:10s} {v:6.3f} (off {base:6.3f}, ref {top:6.3f}) -> {(v-base)/(top-base)*100:7.1f} %")
    print()

print("== Poisson two-sided ratio test, optic side vs the shipped default (counts over equal exposure) ==")
for m in ("hops", "hops_escape", "hops_voluntary"):
    k1, k2 = int(rows("holdBrain", m).sum()), int(rows("shipped", m).sum())
    p = stats.binomtest(k1, k1 + k2, 0.5).pvalue if k1 + k2 else float("nan")
    print(f"  {m:15s} {k1} vs {k2}  binomial p {p:.4g}")

print("\n== how far the escape share could be from 100 %: profile on the ratio (holdBrain-off)/(shipped-off) ==")
kb, ks, ko = (int(rows(a, "hops_escape").sum()) for a in ("holdBrain", "shipped", "off_shipped"))
print(f"  escape counts holdBrain {kb} shipped {ks} off {ko} over 14,400 fly-s each")
lo, hi = stats.beta.ppf([0.025, 0.975], kb + .5, ks - kb + .5) if kb < ks else (np.nan, np.nan)
print(f"  Jeffreys CI on holdBrain/shipped escape rate ratio: [{lo/(1-lo):.3f}, {hi/(1-hi):.3f}]x")
