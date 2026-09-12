"""Report the route attribution of the round-4 sustain take-offs and the 4th-brain-seed replication.

Reads the JSONs written by scripts/probe_hop_route.py (route attribution, and the --gf-hz 1e9 arm that
disables the GF escape route) and by scripts/batch_sustain.py (the seed-3 pair), and prints them beside
the round-4 sustain task's own batches.

    PYTHONIOENCODING=utf-8 python scripts/skeptic_sustain_route_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / 'out'


def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def mw(a, b):
    ra = stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic')
    try:
        ex = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact').pvalue
    except ValueError:
        ex = float('nan')
    return float(ra.statistic), float(ra.pvalue), float(ex)


def main():
    print('### ROUTE ATTRIBUTION (scripts/probe_hop_route.py, brain seed 0, env seeds 0-15, 16 x 300 s)')
    for name in ('sk4_route_default_1.json', 'sk4_route_default_nogf.json', 'sk4_route_off_1.json'):
        d = load(name)
        if d is None:
            print(f'  {name}: NOT FETCHED')
            continue
        h = np.array([r['hops'] for r in d['rows']], float)
        e = np.array([r['escape_hops'] for r in d['rows']], float)
        v = np.array([r['voluntary_hops'] for r in d['rows']], float)
        print(f'  {name}: receptor {d["receptor"]["model"]}/{d["receptor"]["net_rule"]} '
              f'changed={d["receptor"]["fast_sign_changed_entries"]}, gf_hz={d["gf_hz"]:g}, '
              f'device from log; wall {d["wall_s"]:.1f} s')
        print(f'      hops {int(h.sum())} = escape {int(e.sum())} + voluntary {int(v.sum())} '
              f'(per fly mean {h.mean():.4f} +- {h.std(ddof=1):.4f})')
        print(f'      hops per fly {[r["hops"] for r in d["rows"]]}')
        print(f'      escape      {[r["escape_hops"] for r in d["rows"]]}')
        print(f'      voluntary   {[r["voluntary_hops"] for r in d["rows"]]}')
        print(f'      GF max over rows {max(r["gf_max_hz"] for r in d["rows"]):.2f} Hz, '
              f'mean of per-row GF means {np.mean([r["gf_mean_hz"] for r in d["rows"]]):.2f} Hz; '
              f'power max {max(r["power_max_hz"] for r in d["rows"]):.2f} Hz')
        print(f'      airborne frac mean {np.mean([r["airborne_frac"] for r in d["rows"]]):.4f}; '
              f'meals {sum(r["meals"] for r in d["rows"])}; path mean '
              f'{np.mean([r["path_m"] for r in d["rows"]]):.4f} m')

    a, b = load('sk4_route_default_1.json'), load('sk4_route_off_1.json')
    if a and b:
        ha = np.array([r['hops'] for r in a['rows']], float)
        hb = np.array([r['hops'] for r in b['rows']], float)
        u, p, ex = mw(ha, hb)
        print(f'\n  route-probe default vs off, 16 v 16 at the same seeds: '
              f'{ha.mean():.4f} vs {hb.mean():.4f}; U {u:.1f} p_asym {p:.4g} p_exact {ex:.4g}')
    c = load('sk4_route_default_nogf.json')
    if a and c:
        ha = np.array([r['hops'] for r in a['rows']], float)
        hc = np.array([r['hops'] for r in c['rows']], float)
        u, p, ex = mw(ha, hc)
        print(f'  default with the GF escape route disabled (--gf-hz 1e9) vs the same command with it live: '
              f'{int(hc.sum())} vs {int(ha.sum())} hops; U {u:.1f} p_asym {p:.4g} p_exact {ex:.4g}')

    print('\n### FOURTH BRAIN SEED (scripts/batch_sustain.py --seed 3, env seeds 48-63)')
    d4, o4 = load('sk4_sustain_default_4.json'), load('sk4_sustain_off_4.json')
    if d4 and o4:
        for lbl, d in (('default_4', d4), ('off_4', o4)):
            h = np.array([r['hops'] for r in d['rows']], float)
            print(f'  {lbl}: receptor {d["receptor"]["model"]}/{d["receptor"]["net_rule"]} '
                  f'changed={d["receptor"]["fast_sign_changed_entries"]}; brain_seed={d["brain_seed"]}; '
                  f'env {d["rows"][0]["environment_seed"]}-{d["rows"][-1]["environment_seed"]}; '
                  f'wall {d["wall_s"]:.1f} s')
            print(f'      hops {int(h.sum())} mean {h.mean():.4f} +- {h.std(ddof=1):.4f}; '
                  f'per fly {[r["hops"] for r in d["rows"]]}')
            print(f'      meals {sum(r["meals"] for r in d["rows"])}; final energy mean '
                  f'{np.mean([r["energy"] for r in d["rows"]]):.4f}; path mean '
                  f'{np.mean([r["path_m"] for r in d["rows"]]):.4f} m; distance mean '
                  f'{np.mean([r["distance_cm"] for r in d["rows"]]):.2f} cm')
        ha = np.array([r['hops'] for r in d4['rows']], float)
        hb = np.array([r['hops'] for r in o4['rows']], float)
        u, p, ex = mw(ha, hb)
        print(f'  seed-3 pair: U {u:.1f} p_asym {p:.4g} p_exact {ex:.4g}')
        pa = np.array([r['path_m'] for r in d4['rows']], float)
        pb = np.array([r['path_m'] for r in o4['rows']], float)
        u2, p2, e2 = mw(pa, pb)
        print(f'  seed-3 path: {pa.mean():.4f} vs {pb.mean():.4f} m '
              f'({(pa.mean()/pb.mean()-1)*100:+.2f} %); U {u2:.1f} p_asym {p2:.4g}')

        print('\n### POOLED WITH THE TASK BATCHES (4 brain seeds per condition, 64 v 64)')
        dd = [json.loads((OUT / f'r4_sustain_default_{k}.json').read_text(encoding='utf-8')) for k in (1, 2, 3)] + [d4]
        oo = [json.loads((OUT / f'r4_sustain_off_{k}.json').read_text(encoding='utf-8')) for k in (1, 2, 3)] + [o4]
        A = np.array([r['hops'] for d in dd for r in d['rows']], float)
        B = np.array([r['hops'] for d in oo for r in d['rows']], float)
        u, p, ex = mw(A, B)
        print(f'  hops default {int(A.sum())} ({A.mean():.4f} +- {A.std(ddof=1):.4f}, n={A.size}) vs '
              f'off {int(B.sum())} ({B.mean():.4f} +- {B.std(ddof=1):.4f}, n={B.size}); '
              f'U {u:.1f} p_asym {p:.4g} p_exact {ex:.4g}')
        print(f'  per-batch totals default {[int(sum(r["hops"] for r in d["rows"])) for d in dd]} '
              f'off {[int(sum(r["hops"] for r in d["rows"])) for d in oo]}')
        print(f'  rates per 1,000 fly-s: default {A.sum()/(A.size*300)*1000:.2f} '
              f'off {B.sum()/(B.size*300)*1000:.2f}; ratio {(A.mean()/B.mean()):.2f}')


if __name__ == '__main__':
    main()
