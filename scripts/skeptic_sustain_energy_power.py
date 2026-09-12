"""Independent recomputation of the round-4 sustain energy traces and the proposed hop-check power table.

The round-4 sustain task produced out/r4_sustain_energy.log and out/r4_hopcheck_power.log with no
generator in scripts/; this script recomputes both from the fetched artefacts so the numbers can be
checked.  Energy traces come from the batch_sustain progress lines in out/r4_sustain_*.txt; the power
table is a Poisson calculation at the two measured rates (from the JSONs, not from the log).

    PYTHONIOENCODING=utf-8 python scripts/skeptic_sustain_energy_power.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / 'out'
JOBS = ['default_1', 'default_2', 'default_3', 'off_1', 'off_2', 'off_3', 'default_1b', 'off_1b']
LINE = re.compile(r't=([\d.]+)s energy=\[([^\]]*)\] meals=\[([^\]]*)\] hops=\[([^\]]*)\]')


def trace(job):
    txt = (OUT / f'r4_sustain_{job}.txt').read_text(encoding='utf-8')
    rows = []
    for m in LINE.finditer(txt):
        t = float(m.group(1))
        e = np.array([float(v) for v in m.group(2).split(',')])
        meals = np.array([int(v) for v in m.group(3).split(',')])
        hops = np.array([int(v) for v in m.group(4).split(',')])
        rows.append((t, e, meals, hops))
    return rows


def main():
    print('### ENERGY TRACES recomputed from the .txt progress lines')
    for job in JOBS:
        rows = trace(job)
        d = {t: (e, m, h) for t, e, m, h in rows}
        cells = []
        for t in (10., 30., 60., 90., 120., 180., 240., 300.):
            if t in d:
                e = d[t][0]
                cells.append(f't{int(t)}: mean {e.mean():.3f} zeros {int((e <= 0).sum()):2d}/{e.size}')
        allzero = next((t for t, e, _, _ in rows if (e <= 0).all()), None)
        last = rows[-1]
        print(f'  {job:11s} | ' + ' | '.join(cells))
        print(f'              first progress line with all at 0: {allzero}; >0 at {last[0]:.0f} s: '
              f'{int((last[1] > 0).sum())}; meals {int(last[2].sum())}; hops {int(last[3].sum())}; '
              f'lines {len(rows)}')

    print('\n### POISSON POWER TABLE recomputed from the JSON hop totals')
    dd = [json.loads((OUT / f'r4_sustain_{j}.json').read_text(encoding="utf-8")) for j in JOBS[:3]]
    oo = [json.loads((OUT / f'r4_sustain_{j}.json').read_text(encoding="utf-8")) for j in JOBS[3:6]]
    hd = np.array([r['hops'] for d in dd for r in d['rows']], float)
    ho = np.array([r['hops'] for d in oo for r in d['rows']], float)
    rate_d, rate_o = hd.sum() / (hd.size * 300), ho.sum() / (ho.size * 300)
    print(f'  rates default {rate_d:.6f} off {rate_o:.6f} hops/fly-s; ratio {rate_d/rate_o:.4f}')
    print(f'  per-batch totals default {[int(sum(r["hops"] for r in d["rows"])) for d in dd]} '
          f'off {[int(sum(r["hops"] for r in d["rows"])) for d in oo]}')
    for B, T in ((16, 300), (16, 150), (16, 60), (8, 300), (4, 300)):
        n = B * T
        ed, eo = rate_d * n, rate_o * n
        best = None
        for K in range(0, 60):
            # check "hops < K": default fails (passes the check) with prob P(X < K) at the default rate;
            # off fails the check with prob P(X >= K) at the off rate.
            p_def_pass = stats.poisson.cdf(K - 1, ed)
            p_off_fail = 1 - stats.poisson.cdf(K - 1, eo)
            score = max(p_def_pass, p_off_fail)
            if best is None or score < best[0]:
                best = (score, K, p_def_pass, p_off_fail)
        _, K, pdp, pof = best
        print(f'  B={B:2d} x {T:3d} s = {n:5d} fly-s: E[default] {ed:5.1f}, E[off] {eo:4.2f}; '
              f'best bound K={K:2d}: P(default passes)={pdp:.4f}, P(off fails)={pof:.4f}')

    print('\n### PER-FLY NONPARAMETRIC VIEW')
    for lbl, h in (('default', hd), ('off', ho)):
        print(f'  {lbl:8s} n={h.size} mean {h.mean():.4f} median {np.median(h)} '
              f'frac>=1 {(h>=1).mean():.3f} frac>=2 {(h>=2).mean():.3f} max {int(h.max())} '
              f'counts {np.bincount(h.astype(int))}')

    print('\n### ROUND-1 FULL CEILING')
    print('  benchmark walk_gf voluntary_takeoffs 31 in 15 s of one fly = '
          f'{31/15:.4f} hops/fly-s = {31/15*1000:.1f} per 1,000 fly-s = {31/15*300:.1f} per fly per 5 min')
    print(f'  shipped default {rate_d*1000:.2f} per 1,000 fly-s; off {rate_o*1000:.2f} per 1,000 fly-s')


if __name__ == '__main__':
    main()
