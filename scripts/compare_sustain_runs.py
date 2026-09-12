"""Compare batch_sustain.py JSON rollouts between two receptor conditions.

    python scripts/compare_sustain_runs.py --a out/r4_sustain_default_*.json --b out/r4_sustain_off_*.json \
        --label-a default --label-b off

Prints, per input batch and pooled over all rows of a condition: hops, meals, final energy, minimum
energy, path length and distance to the nearest fruit (mean +- sd, and the per-fly vectors for hops),
then a Mann-Whitney U test (two-sided) per metric with the asymptotic and the exact p-value at full
precision and at three significant digits.  Each JSON's receptor header and brain seed are echoed so
that the condition labels can be checked against what the run actually used.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = ('hops', 'meals', 'energy', 'min_energy', 'path_m', 'distance_cm')


def load(patterns):
    files = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        files.extend(hits if hits else [p])
    runs = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding='utf-8'))
        runs.append(dict(file=f, data=d))
    return runs


def sig3(x):
    return f'{x:.3g}'


def describe(name, runs):
    print(f'=== {name}: {len(runs)} batch(es) ===')
    for r in runs:
        d = r['data']
        rec = d.get('receptor', {})
        rows = d['rows']
        seeds = [row['environment_seed'] for row in rows]
        print(f'  {r["file"]}: brain_seed={d.get("brain_seed")} batch={d.get("batch")} '
              f'env_seeds={seeds[0]}..{seeds[-1]} receptor={rec.get("model")}/{rec.get("net_rule")} '
              f'changed={rec.get("fast_sign_changed_entries")} simulated_s={d.get("simulated_s")} '
              f'wall_s={d.get("wall_s", float("nan")):.1f}')
        for m in METRICS:
            v = np.array([row[m] for row in rows], dtype=float)
            extra = f' total={int(v.sum())}' if m in ('hops', 'meals') else ''
            print(f'      {m:12s} mean {v.mean():9.4f}  sd {v.std(ddof=1):8.4f}  '
                  f'min {v.min():8.4f}  max {v.max():8.4f}{extra}')
        print(f'      hops per fly {[row["hops"] for row in rows]}')
        print(f'      meals per fly {[row["meals"] for row in rows]}')


def pooled(runs, metric):
    return np.array([row[metric] for r in runs for row in r['data']['rows']], dtype=float)


def mw(a, b):
    """Two-sided Mann-Whitney U on a vs b; returns (U, asymptotic p, exact p or nan)."""
    res = stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic')
    try:
        ex = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact').pvalue
    except ValueError:
        ex = float('nan')
    return float(res.statistic), float(res.pvalue), float(ex)


def compare(label_a, runs_a, label_b, runs_b):
    print(f'=== pooled {label_a} (n={sum(len(r["data"]["rows"]) for r in runs_a)}) vs '
          f'{label_b} (n={sum(len(r["data"]["rows"]) for r in runs_b)}) ===')
    for m in METRICS:
        a, b = pooled(runs_a, m), pooled(runs_b, m)
        u, p, ex = mw(a, b)
        print(f'  {m:12s} {label_a} {a.mean():9.4f} +- {a.std(ddof=1):7.4f}   '
              f'{label_b} {b.mean():9.4f} +- {b.std(ddof=1):7.4f}   '
              f'U {u:8.1f}  p_asym {p:.6g} ({sig3(p)})  p_exact {ex:.6g} ({sig3(ex)})')
    print(f'=== per-batch {label_a} vs {label_b} (paired by position in the argument lists) ===')
    for i, (ra, rb) in enumerate(zip(runs_a, runs_b), 1):
        print(f'  batch {i}: {Path(ra["file"]).name} vs {Path(rb["file"]).name} '
              f'(brain seeds {ra["data"].get("brain_seed")} / {rb["data"].get("brain_seed")})')
        for m in METRICS:
            a = np.array([row[m] for row in ra['data']['rows']], dtype=float)
            b = np.array([row[m] for row in rb['data']['rows']], dtype=float)
            u, p, ex = mw(a, b)
            print(f'      {m:12s} {a.mean():9.4f} +- {a.std(ddof=1):7.4f}   '
                  f'{b.mean():9.4f} +- {b.std(ddof=1):7.4f}   U {u:7.1f}  '
                  f'p_asym {p:.6g} ({sig3(p)})  p_exact {ex:.6g} ({sig3(ex)})')
    print('=== between-batch scatter within a condition (batch means) ===')
    for label, runs in ((label_a, runs_a), (label_b, runs_b)):
        for m in METRICS:
            means = [float(np.mean([row[m] for row in r['data']['rows']])) for r in runs]
            spread = (max(means) - min(means)) if means else float('nan')
            print(f'  {label:8s} {m:12s} batch means {[round(v, 4) for v in means]}  range {spread:.4f}')
        if len(runs) > 1:
            groups = [[row['hops'] for row in r['data']['rows']] for r in runs]
            try:
                kw = stats.kruskal(*groups)
                print(f'  {label:8s} hops Kruskal-Wallis across batches H {kw.statistic:.4f} '
                      f'p {kw.pvalue:.6g} ({sig3(kw.pvalue)})')
            except ValueError as e:
                print(f'  {label:8s} hops Kruskal-Wallis not defined ({e})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', nargs='+', required=True, help='condition A JSONs (globs allowed)')
    ap.add_argument('--b', nargs='+', required=True, help='condition B JSONs (globs allowed)')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    args = ap.parse_args()
    runs_a, runs_b = load(args.a), load(args.b)
    describe(args.label_a, runs_a)
    describe(args.label_b, runs_b)
    compare(args.label_a, runs_a, args.label_b, runs_b)


if __name__ == '__main__':
    main()
