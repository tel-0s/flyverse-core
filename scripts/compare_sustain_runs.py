"""Compare batch_sustain.py JSON rollouts between two receptor conditions.

    python scripts/compare_sustain_runs.py --a out/r4_sustain_default_*.json --b out/r4_sustain_off_*.json \
        --label-a default --label-b off

Prints, per input batch and pooled over all rows of a condition: hops (and, when the JSONs carry the round-5 split,
hops_escape / hops_voluntary with their rates per 1,000 fly-s and the per-row walking-GF maximum), meals, final
energy, minimum energy, path length and distance to the nearest fruit (mean +- sd, and the per-fly vectors for
hops), then a two-sided Mann-Whitney U test per metric with the asymptotic p-value at full precision and at three
significant digits.  The exact p-value is printed only when the pooled data carry no ties (scipy's 'exact' method
is invalid under ties, and hop counts are heavily tied); otherwise the column reads 'n/a (ties)'.  Each JSON's
receptor header, brain seed and gf_hz are echoed so that the condition labels can be checked against what the run
actually used.

Round-5 additions:
    --ref-proposal      print the reference values and bounds for benchmark.py's opt-in 'hops' section derived from
                        condition B (the off runs): voluntary-only and escape rates per 1,000 fly-s, the walking-GF
                        median, with the shipped default's (condition A) values beside them and the Poisson pass /
                        fail probabilities of a 2,400 fly-s section at both measured rates for candidate bounds
    --benchmark-json F  re-score the 'hops' section of one or more benchmark.py JSONs against the REFERENCES now in
                        scripts/benchmark.py (the measured values do not depend on the bounds the run was scored with)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import stats

METRICS = ('hops', 'hops_escape', 'hops_voluntary', 'gf_max_walk_hz', 'meals', 'energy', 'min_energy', 'path_m', 'distance_cm')
RATE_METRICS = ('hops', 'hops_escape', 'hops_voluntary')


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


def metrics_present(runs):
    keys = set(METRICS)
    for r in runs:
        for row in r['data']['rows']:
            keys &= set(row.keys())
    return [m for m in METRICS if m in keys]


def fly_s(run):
    d = run['data']
    return float(d.get('fly_s', d['batch'] * d['simulated_s']))


def gf_threshold(run):
    d = run['data']
    return float(d.get('flight', {}).get('gf_hz', d.get('gf_hz', 33.0)))


def describe(name, runs):
    print(f'=== {name}: {len(runs)} batch(es) ===')
    metrics = metrics_present(runs)
    for r in runs:
        d = r['data']
        rec = d.get('receptor', {})
        rows = d['rows']
        seeds = [row['environment_seed'] for row in rows]
        print(f'  {r["file"]}: brain_seed={d.get("brain_seed")} batch={d.get("batch")} '
              f'env_seeds={seeds[0]}..{seeds[-1]} receptor={rec.get("model")}/{rec.get("net_rule")} '
              f'changed={rec.get("fast_sign_changed_entries")} simulated_s={d.get("simulated_s")} '
              f'fly_s={fly_s(r):.0f} gf_hz={gf_threshold(r):g} wall_s={d.get("wall_s", float("nan")):.1f}')
        for m in metrics:
            v = np.array([row[m] for row in rows], dtype=float)
            extra = f' total={int(v.sum())}' if m in ('hops', 'meals', 'hops_escape', 'hops_voluntary') else ''
            if m in RATE_METRICS:
                extra += f' per_1000_fly_s={v.sum() / fly_s(r) * 1000:.3f}'
            if m == 'gf_max_walk_hz':
                extra += f' median={np.median(v):.2f} rows>=gf_hz={int((v >= gf_threshold(r)).sum())}/{len(v)}'
            print(f'      {m:16s} mean {v.mean():9.4f}  sd {v.std(ddof=1):8.4f}  '
                  f'min {v.min():8.4f}  max {v.max():8.4f}{extra}')
        print(f'      hops per fly {[row["hops"] for row in rows]}')
        if 'hops_escape' in metrics:
            print(f'      escape per fly {[row["hops_escape"] for row in rows]}')
            print(f'      voluntary per fly {[row["hops_voluntary"] for row in rows]}')
        if 'gf_max_walk_hz' in metrics:
            print(f'      walking GF max per fly {[round(row["gf_max_walk_hz"], 1) for row in rows]}')
        print(f'      meals per fly {[row["meals"] for row in rows]}')


def pooled(runs, metric):
    return np.array([row[metric] for r in runs for row in r['data']['rows']], dtype=float)


def has_ties(a, b):
    v = np.concatenate([a, b])
    return len(np.unique(v)) < len(v)


def mw(a, b):
    """Two-sided Mann-Whitney U on a vs b; returns (U, asymptotic p, exact p or nan when the data carry ties)."""
    res = stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic')
    ex = float('nan')
    if not has_ties(a, b):
        try:
            ex = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact').pvalue
        except ValueError:
            pass
    return float(res.statistic), float(res.pvalue), float(ex)


def fmt_exact(ex):
    return 'n/a (ties)' if np.isnan(ex) else f'{ex:.6g} ({sig3(ex)})'


def rate_line(label, runs, metric):
    total = sum(sum(row[metric] for row in r['data']['rows']) for r in runs)
    fs = sum(fly_s(r) for r in runs)
    per = [sum(row[metric] for row in r['data']['rows']) / fly_s(r) * 1000 for r in runs]
    return f'  {label:8s} {metric:16s} total {int(total):4d} over {fs:,.0f} fly-s = {total / fs * 1000:.3f} per 1,000 fly-s  (per batch {[round(p, 3) for p in per]})'


def compare(label_a, runs_a, label_b, runs_b):
    metrics = [m for m in metrics_present(runs_a) if m in metrics_present(runs_b)]
    print(f'=== pooled {label_a} (n={sum(len(r["data"]["rows"]) for r in runs_a)}) vs '
          f'{label_b} (n={sum(len(r["data"]["rows"]) for r in runs_b)}) ===')
    for m in metrics:
        a, b = pooled(runs_a, m), pooled(runs_b, m)
        u, p, ex = mw(a, b)
        print(f'  {m:16s} {label_a} {a.mean():9.4f} +- {a.std(ddof=1):7.4f}   '
              f'{label_b} {b.mean():9.4f} +- {b.std(ddof=1):7.4f}   '
              f'U {u:8.1f}  p_asym {p:.6g} ({sig3(p)})  p_exact {fmt_exact(ex)}')
    for m in RATE_METRICS:
        if m in metrics:
            print(rate_line(label_a, runs_a, m)); print(rate_line(label_b, runs_b, m))
    if 'gf_max_walk_hz' in metrics:
        for label, runs in ((label_a, runs_a), (label_b, runs_b)):
            v = pooled(runs, 'gf_max_walk_hz'); thr = gf_threshold(runs[0])
            print(f'  {label:8s} walking GF max per row: median {np.median(v):.2f} Hz  mean {v.mean():.2f}  max {v.max():.2f}  '
                  f'rows >= {thr:g} Hz {int((v >= thr).sum())}/{len(v)}  (per batch medians '
                  f'{[round(float(np.median([row["gf_max_walk_hz"] for row in r["data"]["rows"]])), 2) for r in runs]})')
    print(f'=== per-batch {label_a} vs {label_b} (paired by position in the argument lists) ===')
    for i, (ra, rb) in enumerate(zip(runs_a, runs_b), 1):
        print(f'  batch {i}: {Path(ra["file"]).name} vs {Path(rb["file"]).name} '
              f'(brain seeds {ra["data"].get("brain_seed")} / {rb["data"].get("brain_seed")})')
        for m in metrics:
            a = np.array([row[m] for row in ra['data']['rows']], dtype=float)
            b = np.array([row[m] for row in rb['data']['rows']], dtype=float)
            u, p, ex = mw(a, b)
            extra = ''
            if m in RATE_METRICS:
                extra = f'  rate {a.sum() / fly_s(ra) * 1000:.3f} vs {b.sum() / fly_s(rb) * 1000:.3f} per 1,000 fly-s'
            print(f'      {m:16s} {a.mean():9.4f} +- {a.std(ddof=1):7.4f}   '
                  f'{b.mean():9.4f} +- {b.std(ddof=1):7.4f}   U {u:7.1f}  '
                  f'p_asym {p:.6g} ({sig3(p)})  p_exact {fmt_exact(ex)}{extra}')
    print('=== between-batch scatter within a condition (batch means) ===')
    for label, runs in ((label_a, runs_a), (label_b, runs_b)):
        for m in metrics:
            means = [float(np.mean([row[m] for row in r['data']['rows']])) for r in runs]
            spread = (max(means) - min(means)) if means else float('nan')
            print(f'  {label:8s} {m:16s} batch means {[round(v, 4) for v in means]}  range {spread:.4f}')
        if len(runs) > 1:
            for m in [m for m in RATE_METRICS if m in metrics]:
                groups = [[row[m] for row in r['data']['rows']] for r in runs]
                try:
                    kw = stats.kruskal(*groups)
                    print(f'  {label:8s} {m} Kruskal-Wallis across batches H {kw.statistic:.4f} '
                          f'p {kw.pvalue:.6g} ({sig3(kw.pvalue)}) (brain RNG and environment seeds confounded)')
                except ValueError as e:
                    print(f'  {label:8s} {m} Kruskal-Wallis not defined ({e})')


def poisson_cdf(k, mu):
    return float(stats.poisson.cdf(k, mu))


def ref_proposal(label_a, runs_a, label_b, runs_b, section_fly_s=2400.0):
    """Reference values for benchmark.py's 'hops' section from condition B (off), with the shipped default beside them."""
    print(f'=== reference proposal for benchmark.py section hops (reference = {label_b}; section size {section_fly_s:,.0f} fly-s) ===')
    out = {}
    for m in ('hops_voluntary', 'hops_escape'):
        for label, runs in ((label_b, runs_b), (label_a, runs_a)):
            if not runs or m not in metrics_present(runs):
                continue
            total = sum(sum(row[m] for row in r['data']['rows']) for r in runs)
            fs = sum(fly_s(r) for r in runs)
            out[(label, m)] = total / fs * 1000
            print(f'  {label:8s} {m:16s} {total} / {fs:,.0f} fly-s = {total / fs * 1000:.3f} per 1,000 fly-s')
    for label, runs in ((label_b, runs_b), (label_a, runs_a)):
        if runs and 'gf_max_walk_hz' in metrics_present(runs):
            v = pooled(runs, 'gf_max_walk_hz')
            out[(label, 'gf')] = float(np.median(v))
            print(f'  {label:8s} walking GF max median {np.median(v):.2f} Hz (per batch '
                  f'{[round(float(np.median([row["gf_max_walk_hz"] for row in r["data"]["rows"]])), 2) for r in runs]}); '
                  f'rows >= {gf_threshold(runs[0]):g}: {int((v >= gf_threshold(runs[0])).sum())}/{len(v)}')
    for m in ('hops_voluntary', 'hops_escape'):
        ra, rb = out.get((label_a, m)), out.get((label_b, m))
        if ra is None or rb is None:
            continue
        ea, eb = ra * section_fly_s / 1000, rb * section_fly_s / 1000
        print(f'  {m}: expected count in the section {label_a} {ea:.2f} vs {label_b} {eb:.2f} (Poisson)')
        print(f'      bound (per 1,000 fly-s) -> max passing count | P({label_a} passes) | P({label_b} fails)')
        for bound in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
            kmax = int(np.ceil(bound * section_fly_s / 1000) - 1)          # measured < bound  <=>  count <= kmax
            pa = poisson_cdf(kmax, ea); pb = 1 - poisson_cdf(kmax, eb)
            print(f'      < {bound:5.2f}  -> <= {kmax:2d} hops | {pa:.3f} | {pb:.3f}')


def rescore_benchmark(files):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import benchmark as bm
    print('=== re-scoring the hops section against the REFERENCES now in scripts/benchmark.py ===')
    for f in files:
        d = json.loads(Path(f).read_text(encoding='utf-8'))
        sec = d.get('sections', {}).get('hops')
        if not sec or 'error' in sec:
            print(f'  {f}: no hops section ({sec.get("error") if sec else "absent"})'); continue
        cfg = d.get('config', {})
        print(f'  {f}: receptor {cfg.get("receptor", {}).get("model")} (flag {cfg.get("receptor", {}).get("flag")}), device {cfg.get("device")}, '
              f'{sec["batch"]} flies x {sec["frames"] / 100:.0f} s = {sec["fly_s"]:,.0f} fly-s; hops {sec["hops_total"]} = escape {sec["escape_total"]} + '
              f'voluntary {sec["voluntary_total"]}; split consistent {sec.get("route_split_consistent")}')
        for key, val in (('hops.voluntary_per_1000_fly_s', sec['voluntary_per_1000_fly_s']), ('hops.escape_per_1000_fly_s', sec['escape_per_1000_fly_s']),
                         ('hops.walk_gf_max_median_hz', sec['walk_gf_max_median_hz'])):
            ref = bm.REFERENCES[key]
            old = next((c['status'] for c in d.get('checks', []) if c['key'] == key), 'absent')
            print(f'      {key:32s} measured {val:8.3f}  ref {ref.value}  criterion {ref.op} {ref.bound}  gap {ref.gap}  -> {bm.evaluate(ref, val):18s} (run-time status {old})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', nargs='+', default=[], help='condition A JSONs (globs allowed)')
    ap.add_argument('--b', nargs='+', default=[], help='condition B JSONs (globs allowed)')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    ap.add_argument('--ref-proposal', action='store_true', help="derive benchmark.py 'hops' references from condition B")
    ap.add_argument('--section-fly-s', type=float, default=2400.0, help='fly-seconds of the benchmark section for --ref-proposal')
    ap.add_argument('--benchmark-json', nargs='*', default=[], help="benchmark.py JSONs whose 'hops' section is re-scored")
    args = ap.parse_args()
    runs_a, runs_b = load(args.a), load(args.b)
    if runs_a:
        describe(args.label_a, runs_a)
    if runs_b:
        describe(args.label_b, runs_b)
    if runs_a and runs_b:
        compare(args.label_a, runs_a, args.label_b, runs_b)
    if args.ref_proposal:
        ref_proposal(args.label_a, runs_a, args.label_b, runs_b, args.section_fly_s)
    if args.benchmark_json:
        rescore_benchmark(args.benchmark_json)


if __name__ == '__main__':
    main()
