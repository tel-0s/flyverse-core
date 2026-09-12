"""Skeptic verification of docs/audits/feeding_horizon.md (thread: feeding). CPU only, no model change.

    PYTHONIOENCODING=utf-8 python scripts/sk_feeding_horizon_verify.py            # all sections
    PYTHONIOENCODING=utf-8 python scripts/sk_feeding_horizon_verify.py --section G # one section

Every number is recomputed from the rows[] of the audit's own JSONs (out/feedh_*.json, the 26 re-read
out/r{4,5}_*sustain*.json / out/sk5_sustain_*.json) rather than from its summary blocks or its --report log, and
from the shipped code's own constants.  It adds four controls the audit did not run:

  B  the ground floor on `min_distance_cm`.  Under --fence, BatchBody._walk updates only x, y and heading
     (flyverse/batch_body.py:145-157), so a walking fly sits at exactly z = table_top = 0.75 while the apple's
     centre is at z = 0.79 with radius 0.04 (flyverse/world.py:332-339).  `nearest_fruit` returns
     |p - centre| - radius (flyverse/batch_sim.py:151-155), so on the ground that is >= sqrt(0.04^2) - 0.04 = 0.
     Every negative min_distance_cm is therefore an AIRBORNE frame, not "the fly walks through the apple's
     footprint" (feeding_horizon.md sections 5.1 item 2 and 5.3 item 4).
  C  the same comparison with min_distance clipped at that ground floor, and the take-off <-> approach coupling
     (within-arm Spearman, hop-matched split) that the audit's 15x take-off difference puts under its headline
     approach measure.
  D  the replicate-level (brain-seed-level) test of the approach difference, which is the unit THE PROJECT RULE
     asks for -- the audit's stratified p is a 32-fly-level statistic.
  E  the shell enrichment of the 416 final positions against an x-matched null (observed x marginal, y uniform)
     instead of a uniform point, which isolates the plume-specific (lateral) part of the pile-up; and the
     directed route the protocol actually offers, the start being exactly on the apple's y axis.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out'
CELLS = (('1.0', 10), ('0.5', 5), ('0.25', 5))
PRIOR = ('r4_sustain_default_*.json', 'r4_sustain_off_*.json', 'r5_adopt_sustain_live_*.json',
         'r5_sustain_default_*.json', 'r5_sustain_off_*.json', 'r5_skeptic_sustain_live_*.json',
         'sk5_sustain_default_live_1r.json', 'sk5_sustain_off_live_1r.json')


def load(pattern):
    return {Path(f).stem: json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(str(OUT / pattern)))}


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def mwu(a, b):
    r = stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic')
    try:
        pe = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact').pvalue
    except Exception:
        pe = float('nan')
    return float(r.statistic), float(r.pvalue), float(pe)


def zstat(a, b):
    """Standardised Mann-Whitney U and its variance weight, for the stratified combination."""
    n1, n2 = len(a), len(b)
    U, _, _ = mwu(a, b)
    var = n1 * n2 * (n1 + n2 + 1) / 12
    return (U - n1 * n2 / 2) / np.sqrt(var), var


def stratified(parts):
    num = sum(z * np.sqrt(v) for z, v in parts)
    z = num / np.sqrt(sum(v for _, v in parts))
    return z, 2 * stats.norm.sf(abs(z))


def fisher(ps):
    chi = -2 * sum(np.log(p) for p in ps)
    return chi, float(stats.chi2.sf(chi, 2 * len(ps)))


def cell(D, drain, minutes):
    return {arm: [r for k in sorted(D) if f'_{arm}_' in k and f'd{drain}_' in k and f'{minutes}min' in k
                  for r in D[k]['rows']] for arm in ('default', 'off')}


# ------------------------------------------------------------------ A: per-run recomputation
def sec_A(D):
    print('A. per-run recomputation from rows[] (audit sections 5.1-5.2)')
    for k in sorted(D):
        d = D[k]; rows = d['rows']; B = len(rows); s = d['summary']
        meals, cont = col(rows, 'meals'), col(rows, 'contacts')
        mind, en = col(rows, 'min_distance_cm'), col(rows, 'energy')
        tz = [r['t_zero_s'] for r in rows if r['t_zero_s'] is not None]
        print(f'  {k}  B={B} T={d["simulated_s"]:g}s device_option={d["options"]["device"]} '
              f'receptor={d["receptor"]["model"]}/{d["receptor"]["net_rule"]} '
              f'fast_sign_changed={d["receptor"]["fast_sign_changed_entries"]:,} '
              f'drain_scale={d["metabolism"]["drain_scale"]:g} gf_hz={d["flight"]["gf_hz"]:g}')
        print(f'     meals {meals.sum():.0f} ({meals.mean():.4f}/fly, summary {s["meals_per_fly"]:.4f}) '
              f'contacts {cont.sum():.0f}  meals==contacts per fly: {bool((meals == cont).all())}  '
              f'flies fed {int((meals > 0).sum())}  flies inside 1.5 cm {int((mind < 1.5).sum())}  '
              f'same set: {bool(((mind < 1.5) == (meals > 0)).all())}')
        print(f'     min_energy {col(rows, "min_energy").min():.4f}..{col(rows, "min_energy").max():.4f}  '
              f'final energy mean {en.mean():.4f}  at zero {int((en <= 1e-9).sum())}  '
              f'frac_at_zero {col(rows, "frac_at_zero").mean():.4f}  '
              f't_zero median {np.median(tz) if tz else float("nan"):.1f} first {min(tz) if tz else float("nan"):.1f}')
        print(f'     min_distance median {np.median(mind):.3f} mean {mind.mean():.3f} '
              f'[{mind.min():.3f}, {mind.max():.3f}]  path {col(rows, "path_m").mean():.4f} m  '
              f'hops {col(rows, "hops").sum():.0f} = {col(rows, "hops_escape").sum():.0f} esc + '
              f'{col(rows, "hops_voluntary").sum():.0f} vol  gf_walk median {np.median(col(rows, "gf_max_walk_hz")):.2f} Hz')
        mods = {}
        for r in rows:
            for m, v in r['modes'].items():
                mods[m] = mods.get(m, 0.) + v / B
        print('     modes ' + ', '.join(f'{m} {v*100:.2f}%' for m, v in sorted(mods.items(), key=lambda kv: -kv[1])))


# ------------------------------------------------------------------ B: the ground floor
def sec_B(D):
    print('B. CONTROL: is a negative min_distance_cm reachable on the ground?')
    from flyverse.world import make_room
    _, info = make_room(0, 'apple')
    name, centre, radius = info['fruit'][0]
    z = info['table_top_z']; h = float(centre[2]) - z
    print(f'  {name} centre z {centre[2]:g}, radius {radius:g}, table top {z:g} -> a walking fly (fenced _walk '
          f'never changes z) is h = {h:g} m below the centre')
    print(f'  ground-reachable minimum of |p - centre| - radius = {max(h - radius, 0.)*100:.4f} cm; '
          f'negative values require |p - centre| < {radius:g} m, i.e. z > {centre[2]-radius:g} -- flight only')
    ground = [r for k in D for r in D[k]['rows'] if r['hops'] == 0]
    hopped = [r for k in D for r in D[k]['rows'] if r['hops'] > 0]
    gm, hm = col(ground, 'min_distance_cm'), col(hopped, 'min_distance_cm')
    print(f'  flies that never took off: n {len(ground)}, min_distance min {gm.min():+.3f} cm, all >= 0: {bool((gm >= 0).all())}')
    print(f'  flies that took off:       n {len(hopped)}, min_distance min {hm.min():+.3f} cm, '
          f'{int((hm < 0).sum())} of them negative')
    neg = [r for k in D for r in D[k]['rows'] if r['min_distance_cm'] < 0]
    print(f'  ALL negative-minimum flies: {len(neg)}; of those with hops > 0: {sum(1 for r in neg if r["hops"] > 0)}')
    print(f'  => the audit\'s stated cause ("the fly walks through the apple\'s footprint", 5.1 item 2 / 5.3 item 4) '
          f'is geometrically impossible; the negative tail is airborne')
    print(f'  ground-only closest approach: median {np.median(gm):.2f} cm; within 5 cm {int((gm <= 5).sum())}/{len(gm)}, '
          f'within 1.5 cm {int((gm < 1.5).sum())}, fed {sum(1 for r in ground if r["meals"] > 0)}')
    print(f'  hopped closest approach:      median {np.median(hm):.2f} cm; within 1.5 cm {int((hm < 1.5).sum())}/{len(hm)}, '
          f'fed {sum(1 for r in hopped if r["meals"] > 0)}')


# ------------------------------------------------------------------ C: stats + coupling
def sec_C(D):
    print('C. per-cell Mann-Whitney (tie-aware), the ground-floor clip, and the take-off <-> approach coupling')
    pm, pd, pdc, zd, zdc = [], [], [], [], []
    for drain, minutes in CELLS:
        c = cell(D, drain, minutes); a, b = c['default'], c['off']
        if not a or not b:
            continue
        print(f'  cell drain {drain}, {minutes} min (n {len(a)} / {len(b)})')
        for key in ('meals', 'contacts', 'min_distance_cm', 'path_m', 'hops'):
            x, y = col(a, key), col(b, key)
            U, p, pe = mwu(x, y)
            print(f'    {key:16s} default {x.mean():8.4f} (med {np.median(x):7.3f}) off {y.mean():8.4f} '
                  f'(med {np.median(y):7.3f})  U {U:7.1f}  p {p:.4f}  p_exact {pe:.4f}')
            if key == 'meals': pm.append(p)
            if key == 'min_distance_cm': pd.append(p); zd.append(zstat(x, y))
        x, y = np.clip(col(a, 'min_distance_cm'), 0, None), np.clip(col(b, 'min_distance_cm'), 0, None)
        U, p, _ = mwu(x, y); pdc.append(p); zdc.append(zstat(x, y))
        print(f'    {"minD clip>=0":16s} default {x.mean():8.4f} off {y.mean():8.4f}  U {U:7.1f}  p {p:.4f}')
        for arm, rows in (('default', a), ('off', b)):
            h, md = col(rows, 'hops'), col(rows, 'min_distance_cm')
            if len(set(h)) > 1:
                rho, pr = stats.spearmanr(h, md)
                print(f'    coupling {arm:8s}: spearman(hops, min_distance) rho {rho:+.3f} p {pr:.4f}')
    if pm:
        print(f'  Fisher meals        chi2 {fisher(pm)[0]:.3f} p {fisher(pm)[1]:.4f}   (audit 0.756)')
        print(f'  Fisher min_distance chi2 {fisher(pd)[0]:.3f} p {fisher(pd)[1]:.4f}   (audit 0.072)')
        print(f'  Fisher minD clipped chi2 {fisher(pdc)[0]:.3f} p {fisher(pdc)[1]:.4f}')
        z, p = stratified(zd);  print(f'  stratified z raw     {z:+.3f} p {p:.4f}   (audit z -2.42 p 0.016)')
        z, p = stratified(zdc); print(f'  stratified z clipped {z:+.3f} p {p:.4f}')
    print('  the audit\'s own control -- flies that never took off:')
    parts = []
    for drain, minutes in CELLS:
        c = cell(D, drain, minutes)
        a = [r for r in c['default'] if r['hops'] == 0]; b = [r for r in c['off'] if r['hops'] == 0]
        print(f'    drain {drain} {minutes}min: default {len(a)}/{len(c["default"])}, off {len(b)}/{len(c["off"])}')
        if len(a) >= 2 and len(b) >= 2:
            x, y = col(a, 'min_distance_cm'), col(b, 'min_distance_cm')
            U, p, pe = mwu(x, y); parts.append(zstat(x, y))
            print(f'      default {x.mean():.3f} vs off {y.mean():.3f}  U {U:.1f} p {p:.4f} p_exact {pe:.4f}')
    if parts:
        z, p = stratified(parts)
        print(f'    stratified over those cells: z {z:+.3f} p {p:.4f}   (audit z -1.81 p 0.070)')


# ------------------------------------------------------------------ D: replicate level
def sec_D(D):
    print('D. CONTROL: replicate (brain-seed) level, the unit THE PROJECT RULE asks for')
    runs = {}
    for k in sorted(D):
        arm = 'default' if '_default_' in k else 'off'
        key = k.replace('feedh_default_', '').replace('feedh_off_', '').replace('skfeed_default_', '').replace('skfeed_off_', '')
        md = col(D[k]['rows'], 'min_distance_cm')
        runs.setdefault(key, {})[arm] = dict(mean=float(md.mean()), median=float(np.median(md)),
                                             meals=float(D[k]['summary']['meals_per_fly']))
    pairs = [(k, v['default'], v['off']) for k, v in sorted(runs.items()) if 'default' in v and 'off' in v]
    dd = np.array([a['mean'] for _, a, _ in pairs]); oo = np.array([b['mean'] for _, _, b in pairs])
    mm = np.array([a['meals'] for _, a, _ in pairs]); nn = np.array([b['meals'] for _, _, b in pairs])
    for k, a, b in pairs:
        print(f'    {k:24s} min_distance default {a["mean"]:7.3f} vs off {b["mean"]:7.3f} '
              f'{"d<o" if a["mean"] < b["mean"] else "d>o"} | meals {a["meals"]:.3f} vs {b["meals"]:.3f}')
    n = len(dd); k = int((dd < oo).sum())
    print(f'  min_distance: default closer in {k}/{n} runs; exact sign test p {2*stats.binom.sf(k-1, n, .5):.4f}')
    if n >= 2:
        print(f'    Wilcoxon signed-rank p {stats.wilcoxon(dd, oo).pvalue:.4f}; '
              f'paired t p {stats.ttest_rel(dd, oo).pvalue:.4f}')
    print(f'  meals: default higher in {int((mm > nn).sum())}/{n}, lower {int((mm < nn).sum())}, tied {int((mm == nn).sum())}')
    ten = [(kk, a, b) for kk, a, b in pairs if '10min' in kk]
    if len(ten) >= 2:
        v = np.array([a['mean'] for _, a, _ in ten] + [b['mean'] for _, _, b in ten])
        m = len(ten); obs = v[:m].mean() - v[m:].mean(); tot = hit = 0
        for c in itertools.combinations(range(2 * m), m):
            g1 = v[list(c)]; g2 = v[[i for i in range(2 * m) if i not in c]]
            tot += 1; hit += abs(g1.mean() - g2.mean()) >= abs(obs) - 1e-12
        print(f'  10-min cell alone, exact permutation over the {m}+{m} replicate means: p {hit}/{tot} = {hit/tot:.3f}')


# ------------------------------------------------------------------ E: prior + nulls
def sec_E():
    print('E. the 26 re-read batches, the x-matched null, and the directed route the protocol offers')
    rows, arms = [], {'default': [], 'off': []}
    for pat in PRIOR:
        for f in sorted(glob.glob(str(OUT / pat))):
            d = json.load(open(f, encoding='utf-8'))
            arms[d['options']['receptor_model']].append((Path(f).name, d['rows'])); rows += d['rows']
    for arm in ('default', 'off'):
        meals = np.array([r['meals'] for _, rs in arms[arm] for r in rs], float)
        en = np.array([r['energy'] for _, rs in arms[arm] for r in rs], float)
        per = [float(np.mean([r['meals'] for r in rs])) for _, rs in arms[arm]]
        print(f'  {arm:8s} batches {len(arms[arm])} flies {len(meals)} meals {meals.sum():.0f} '
              f'({meals.mean():.4f}/fly) fed {int((meals > 0).sum())} ({(meals > 0).mean()*100:.1f} %) '
              f'at zero {int((en <= 1e-9).sum())} ({(en <= 1e-9).mean()*100:.1f} %) '
              f'per-batch {min(per):.3f}-{max(per):.3f}')
    meals = np.array([r['meals'] for r in rows], float); en = np.array([r['energy'] for r in rows], float)
    print(f'  pooled {len(rows)} rollouts: {meals.mean():.4f} meals/fly (audit 0.139); '
          f'{int((en <= 1e-9).sum())}/{len(rows)} = {(en <= 1e-9).mean()*100:.1f} % at energy 0 (audit 390, 93.8 %)')
    d48 = [r for n, rs in arms['default'] if n.startswith('r5_adopt_sustain_live') for r in rs]
    o48 = [r for n, rs in arms['off'] if n.startswith('r5_sustain_off_live') for r in rs]
    for key, ref in (('meals', 'U 1106 p 0.586'), ('energy', 'U 1196 p 0.447')):
        U, p, _ = mwu(col(d48, key), col(o48, key))
        print(f'  r5 live arms (48 v 48) {key:7s} U {U:.0f} p {p:.4f}   (audit {ref})')
    mods = {}
    for r in rows:
        for k, v in r.get('modes', {}).items():
            mods[k] = mods.get(k, 0.) + v / len(rows)
    print('  mode fractions ' + ', '.join(f'{k} {v*100:.2f}%' for k, v in sorted(mods.items(), key=lambda kv: -kv[1])))
    pos = np.array([r['position'] for r in rows], float)
    print(f'  final-position z values: {sorted(set(np.round(pos[:, 2], 4)))} (a walking fly never leaves the table top)')
    centre = np.array([.25, .15, .79]); d = (np.linalg.norm(pos - centre, axis=1) - .04) * 100
    rng = np.random.default_rng(1); N = 400_000
    ux, uy = rng.uniform(-.6, .6, N), rng.uniform(-.4, .4, N)
    uA = (np.sqrt((ux - .25) ** 2 + (uy - .15) ** 2 + .04 ** 2) - .04) * 100
    bx, by = rng.choice(pos[:, 0], N), rng.uniform(-.4, .4, N)
    uB = (np.sqrt((bx - .25) ** 2 + (by - .15) ** 2 + .04 ** 2) - .04) * 100
    print(f'  CONTROL shell enrichment vs a uniform point (the audit) and vs an x-MATCHED null (observed x, uniform y):')
    print(f'    {"shell":10s} {"obs %":>7s} {"unif %":>7s} {"enr":>6s} {"xmatch %":>9s} {"enr":>6s}')
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30), (30, 40), (40, 70)):
        o = ((d >= lo) & (d < hi)).mean() * 100
        a = ((uA >= lo) & (uA < hi)).mean() * 100; b = ((uB >= lo) & (uB < hi)).mean() * 100
        print(f'    {lo:2d}-{hi:2d} cm {o:>7.1f} {a:>7.1f} {o/a:>6.2f} {b:>9.1f} {o/max(b,1e-9):>6.2f}')
    for cut in (8, 12, 20):
        o = (d <= cut).mean()
        print(f'    <= {cut:2d} cm: obs {o*100:5.1f} %  uniform enr {o/(uA <= cut).mean():.2f}  '
              f'x-matched enr {o/max((uB <= cut).mean(), 1e-12):.2f}')
    print(f'  upwind of the fruit (x > 0.25): {(pos[:, 0] > .25).mean()*100:.1f} %; median final x {np.median(pos[:, 0]):.4f}')
    start, apple = np.array([-.15, .15]), np.array([.25, .15])
    r = float(np.sqrt(.055 ** 2 - .04 ** 2)); L = float(np.linalg.norm(apple - start))
    print(f'  the protocol\'s directed route: start and apple share y ({start[1] == apple[1]}), separation {L*100:.0f} cm '
          f'straight upwind; capture radius {r*100:.2f} cm')
    print(f'    a straight upwind walker holding heading within {np.degrees(np.arctan(r/L)):.2f} deg reaches the disc in '
          f'{(L-r)/.01223:.1f} s = {(L-r)/.01223/60:.2f} min, against the random-sweep 17.3 min')
    print(f'    => the "ideal non-revisiting searcher" figure is a BLIND-SEARCH BASELINE, not a ceiling for an '
          f'odour-guided fly; the model running at 43-53 % of it indicts the searcher, not the assay')


def sec_F(D):
    print('F. instrument identity: energy0 - drain*T - walk*path vs the recorded energy')
    for k in sorted(D):
        d = D[k]; m = d['metabolism']; T = d['simulated_s']
        res = np.array([r['energy'] - (m['energy0'] - m['drain_per_s'] * T - m['walk_drain_per_m'] * r['path_m'])
                        for r in d['rows'] if r['meals'] == 0 and r['hops'] == 0 and r['frac_at_zero'] == 0])
        if len(res):
            print(f'  {k:34s} n {len(res):2d} median {np.median(res):+.2e} min {res.min():+.2e} max {res.max():+.2e} '
                  f'all negative: {bool((res < 0).all())}')
        else:
            print(f'  {k:34s} no qualifying fly (the audit\'s "3-13 per arm" covers only the 5-minute arms)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--section', default='', help='one of A B C D E F (default: all)')
    ap.add_argument('--glob', default='feedh_*.json', help='the audit\'s run JSONs; add skfeed_*.json to include the replicate batch')
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    D = {}
    for pat in args.glob.split(','):
        D.update(load(pat.strip()))
    print(f'{len(D)} run JSON(s): {", ".join(sorted(D))}\n')
    todo = args.section.upper() or 'ABCDEF'
    for name in todo:
        fn = globals().get(f'sec_{name}')
        if fn is None:
            continue
        print('=' * 118)
        fn() if name == 'E' else fn(D)
        print()


if __name__ == '__main__':
    main()
