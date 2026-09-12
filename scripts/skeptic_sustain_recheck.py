"""Independent recomputation of the round-4 sustain claims from the batch_sustain JSONs.

Written by the round-4 sustain skeptic.  Does not import the task's generator; reads the JSONs
directly and recomputes every number quoted in the report (headers, per-batch and pooled stats,
Mann-Whitney asymptotic + exact, histograms, rates, mode fractions, Kruskal-Wallis, dispersion).

    PYTHONIOENCODING=utf-8 python scripts/skeptic_sustain_recheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / 'out'
DEF = ['r4_sustain_default_1.json', 'r4_sustain_default_2.json', 'r4_sustain_default_3.json']
OFF = ['r4_sustain_off_1.json', 'r4_sustain_off_2.json', 'r4_sustain_off_3.json']
REP = ['r4_sustain_default_1b.json', 'r4_sustain_off_1b.json']
METRICS = ('hops', 'meals', 'energy', 'min_energy', 'path_m', 'distance_cm')


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def hdr(name):
    d = load(name)
    o = d['options']
    rec = d['receptor']
    rows = d['rows']
    seeds = [r['environment_seed'] for r in rows]
    print(f'{name}: brain_seed={d["brain_seed"]} batch={d["batch"]} frames={d["frames"]} '
          f'sim_s={d["simulated_s"]} wall_s={d["wall_s"]:.1f} flys_per_walls={d["aggregate_fly_s_per_wall_s"]:.2f}')
    print(f'    receptor={rec} ')
    print(f'    env_seeds={seeds} (n={len(seeds)}, unique={len(set(seeds))})')
    keys = ('seed', 'program', 'fruit', 'fence', 'escape_gating', 'brain_dt', 'optic_dt',
            'cuda_graphs', 'cuda_kernels', 'event_driven', 'cuda_sparse', 'weight_dtype',
            'device', 'receptor_model', 'receptor_net_rule', 'batch', 'seeds', 'minutes',
            'start', 'energy', 'log_every', 'json', 'save', 'load')
    print('    options: ' + ', '.join(f'{k}={o.get(k)!r}' for k in keys if k in o))
    return d


def vec(runs, m):
    return np.array([r[m] for d in runs for r in d['rows']], dtype=float)


def mw(a, b):
    ra = stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic')
    try:
        ex = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact').pvalue
    except ValueError:
        ex = float('nan')
    return float(ra.statistic), float(ra.pvalue), float(ex)


def main():
    print('### scipy', stats.__name__, __import__('scipy').__version__)
    print('\n### HEADERS (default)')
    dd = [hdr(n) for n in DEF]
    print('\n### HEADERS (off)')
    oo = [hdr(n) for n in OFF]
    print('\n### HEADERS (fixed-seed replicate)')
    rr = [hdr(n) for n in REP]

    print('\n### PER-BATCH hops + pairwise MW (seed-matched)')
    for i, (a, b) in enumerate(zip(dd, oo), 1):
        ha = np.array([r['hops'] for r in a['rows']], float)
        hb = np.array([r['hops'] for r in b['rows']], float)
        u, p, ex = mw(ha, hb)
        print(f'  batch {i}: default total {int(ha.sum())} mean {ha.mean():.4f} +- {ha.std(ddof=1):.4f} '
              f'| off total {int(hb.sum())} mean {hb.mean():.4f} +- {hb.std(ddof=1):.4f} '
              f'| U {u:.1f} p_asym {p:.6g} p_exact {ex:.6g}')
        print(f'      default hops {[r["hops"] for r in a["rows"]]}')
        print(f'      off     hops {[r["hops"] for r in b["rows"]]}')

    print('\n### POOLED 48 v 48, all metrics')
    for m in METRICS:
        a, b = vec(dd, m), vec(oo, m)
        u, p, ex = mw(a, b)
        print(f'  {m:12s} default {a.mean():9.4f} +- {a.std(ddof=1):7.4f} (n={a.size}) '
              f'off {b.mean():9.4f} +- {b.std(ddof=1):7.4f} (n={b.size}) '
              f'U {u:8.1f} p_asym {p:.6g} p_exact {ex:.6g}')

    ha, hb = vec(dd, 'hops'), vec(oo, 'hops')
    print(f'  hops totals default {int(ha.sum())} off {int(hb.sum())}')
    print(f'  rates default {ha.sum()/(ha.size*300):.6g} off {hb.sum()/(hb.size*300):.6g} '
          f'ratio {(ha.sum()/ha.size)/(hb.sum()/hb.size):.4f}')
    print(f'  per-fly hist 0..4 default {[int((ha==k).sum()) for k in range(5)]} '
          f'off {[int((hb==k).sum()) for k in range(5)]}')
    print(f'  frac >=1 default {(ha>=1).mean():.4f} off {(hb>=1).mean():.4f}')
    print(f'  per-1000-fly-s default {ha.sum()/(ha.size*300)*1000:.4f} off {hb.sum()/(hb.size*300)*1000:.4f}')

    print('\n### KRUSKAL within condition (hops across the 3 batches)')
    for lbl, runs in (('default', dd), ('off', oo)):
        g = [[r['hops'] for r in d['rows']] for d in runs]
        kw = stats.kruskal(*g)
        print(f'  {lbl}: H {kw.statistic:.4f} p {kw.pvalue:.6g}  batch totals {[int(sum(x)) for x in g]}')
        tot = np.array([sum(x) for x in g], float)
        print(f'      batch-total mean {tot.mean():.4f} sd {tot.std(ddof=1):.4f} '
              f'Poisson sd sqrt(mean) {np.sqrt(tot.mean()):.4f}')

    print('\n### PER-BATCH means, every metric')
    for lbl, runs in (('default', dd), ('off', oo)):
        for m in METRICS:
            means = [float(np.mean([r[m] for r in d['rows']])) for d in runs]
            print(f'  {lbl:8s} {m:12s} {[round(v,4) for v in means]} range {max(means)-min(means):.4f}')

    print('\n### PER-BATCH path_m MW p (seed-matched)')
    for i, (a, b) in enumerate(zip(dd, oo), 1):
        pa = np.array([r['path_m'] for r in a['rows']], float)
        pb = np.array([r['path_m'] for r in b['rows']], float)
        u, p, ex = mw(pa, pb)
        print(f'  batch {i}: {pa.mean():.4f} vs {pb.mean():.4f} U {u:.1f} p_asym {p:.6g} p_exact {ex:.6g}')
    pa, pb = vec(dd, 'path_m'), vec(oo, 'path_m')
    print(f'  pooled path pct diff {(pa.mean()/pb.mean()-1)*100:.4f} %')

    print('\n### FIXED-SEED REPLICATE (brain seed 0, env 0-15)')
    d1, o1 = dd[0], oo[0]
    d1b, o1b = rr[0], rr[1]
    for lbl, x, y in (('default', d1, d1b), ('off', o1, o1b)):
        hx = np.array([r['hops'] for r in x['rows']], float)
        hy = np.array([r['hops'] for r in y['rows']], float)
        print(f'  {lbl}: hops {int(hx.sum())} -> {int(hy.sum())} '
              f'(mean {hx.mean():.4f} -> {hy.mean():.4f}); '
              f'meals {int(sum(r["meals"] for r in x["rows"]))} -> {int(sum(r["meals"] for r in y["rows"]))}; '
              f'mean final energy {np.mean([r["energy"] for r in x["rows"]]):.4f} -> '
              f'{np.mean([r["energy"] for r in y["rows"]]):.4f}; '
              f'mean path {np.mean([r["path_m"] for r in x["rows"]]):.4f} -> '
              f'{np.mean([r["path_m"] for r in y["rows"]]):.4f}')
        print(f'      hops per fly {[r["hops"] for r in x["rows"]]} -> {[r["hops"] for r in y["rows"]]}')
        ident = all(r1 == r2 for r1, r2 in zip(x['rows'], y['rows']))
        print(f'      rows byte-identical between the two runs: {ident}')
    print('  seed-0 pooled 32 v 32 (r4_sustain_default_1 + _1b vs off_1 + _1b):')
    a = np.array([r['hops'] for d in (d1, d1b) for r in d['rows']], float)
    b = np.array([r['hops'] for d in (o1, o1b) for r in d['rows']], float)
    u, p, ex = mw(a, b)
    print(f'      default {a.mean():.4f} +- {a.std(ddof=1):.4f} off {b.mean():.4f} +- {b.std(ddof=1):.4f} '
          f'U {u:.1f} p_asym {p:.6g} p_exact {ex:.6g}')

    print('\n### MODE FRACTIONS (mean over 48 flies)')
    for lbl, runs in (('default', dd), ('off', oo)):
        keys = sorted({k for d in runs for r in d['rows'] for k in r['modes']})
        out = {}
        for k in keys:
            out[k] = float(np.mean([r['modes'].get(k, 0.0) for d in runs for r in d['rows']]))
        print(f'  {lbl}: ' + ', '.join(f'{k} {v:.4f}' for k, v in
                                      sorted(out.items(), key=lambda kv: -kv[1])))
        print(f'      sum {sum(out.values()):.4f}')

    print('\n### FLIES ABOVE 0 ENERGY AT t=300 (from rows energy)')
    for lbl, runs in (('default', dd), ('off', oo), ('rep', rr)):
        for d, name in zip(runs, (DEF if lbl == 'default' else OFF if lbl == 'off' else REP)):
            e = np.array([r['energy'] for r in d['rows']], float)
            print(f'  {name}: >0 {int((e>0).sum())}/16 mean {e.mean():.6f} '
                  f'min_energy mean {np.mean([r["min_energy"] for r in d["rows"]]):.6f}')

    print('\n### ROUND-3 BATCH (sanity, for the generator-validation claim)')
    try:
        r3d, r3o = load('r3_sustain_default.json'), load('r3_sustain_off.json')
        a = np.array([r['hops'] for r in r3d['rows']], float)
        b = np.array([r['hops'] for r in r3o['rows']], float)
        u, p, ex = mw(a, b)
        print(f'  r3 hops {int(a.sum())} vs {int(b.sum())} U {u:.1f} p_asym {p:.6g} p_exact {ex:.6g}')
        print(f'  r3 receptor headers: {r3d["receptor"]} | {r3o["receptor"]}')
        print(f'  r3 energy option: {r3d["options"].get("energy")} / {r3o["options"].get("energy")}')
        print(f'  r3 meals {int(sum(r["meals"] for r in r3d["rows"]))} vs '
              f'{int(sum(r["meals"] for r in r3o["rows"]))}')
    except FileNotFoundError as e:
        print('  missing:', e)


if __name__ == '__main__':
    sys.exit(main())
