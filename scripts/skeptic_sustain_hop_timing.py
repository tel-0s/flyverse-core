"""Hop timing per 10 s from the batch_sustain progress lines, as a starvation confound check.

The round-4 sustain runs end with almost every fly at energy 0, and the two conditions differ in meals
(11 vs 8) and final energy, so a hop excess could in principle be a consequence of differing energy
trajectories rather than of the receptor model.  This splits the cumulative `hops=` vectors that
batch_sustain.py prints every 10 simulated seconds into per-interval counts and compares the two
conditions over the window in which no fly has yet reached energy 0.

    PYTHONIOENCODING=utf-8 python scripts/skeptic_sustain_hop_timing.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / 'out'
DEF = ['default_1', 'default_2', 'default_3']
OFF = ['off_1', 'off_2', 'off_3']
LINE = re.compile(r't=([\d.]+)s energy=\[([^\]]*)\] meals=\[([^\]]*)\] hops=\[([^\]]*)\]')


def series(job):
    txt = (OUT / f'r4_sustain_{job}.txt').read_text(encoding='utf-8')
    ts, hops, zeros = [], [], []
    for m in LINE.finditer(txt):
        ts.append(float(m.group(1)))
        e = np.array([float(v) for v in m.group(2).split(',')])
        hops.append(np.array([int(v) for v in m.group(4).split(',')]))
        zeros.append(int((e <= 0).sum()))
    return np.array(ts), np.stack(hops), np.array(zeros)


def main():
    print('### cumulative hops and starving flies per 10 s')
    per = {}
    for job in DEF + OFF:
        ts, h, z = series(job)
        tot = h.sum(axis=1)
        first_zero = ts[np.argmax(z > 0)] if (z > 0).any() else None
        per[job] = (ts, h, z)
        print(f'  {job:10s} first fly at energy 0: t={first_zero}; cumulative hops '
              f'{dict(zip(ts[::3].astype(int).tolist(), tot[::3].tolist()))}')

    for cut in (120., 150., 300.):
        a, b = [], []
        for job, store in ((j, a) for j in DEF):
            ts, h, _ = per[job]
            i = int(np.argmax(ts >= cut))
            store.extend(h[i].tolist())
        for job, store in ((j, b) for j in OFF):
            ts, h, _ = per[job]
            i = int(np.argmax(ts >= cut))
            store.extend(h[i].tolist())
        a, b = np.array(a, float), np.array(b, float)
        u = stats.mannwhitneyu(a, b, alternative='two-sided')
        ue = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact')
        print(f'\n  hops by t={cut:.0f} s: default {int(a.sum())} (mean {a.mean():.4f}) vs '
              f'off {int(b.sum())} (mean {b.mean():.4f}); U {u.statistic:.1f} '
              f'p_asym {u.pvalue:.4g} p_exact {ue.pvalue:.4g}')
        mz = max(int(per[j][2][int(np.argmax(per[j][0] >= cut))]) for j in DEF + OFF)
        print(f'      max flies already at energy 0 at that time, over the six jobs: {mz}/16')


if __name__ == '__main__':
    main()
