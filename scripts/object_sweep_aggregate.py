"""Aggregate the round-3 object-sweep batch (cluster run r3-obj-*): replicates, none-vs-none null, z-scores, ladder.

    PYTHONIOENCODING=utf-8 python out/r3obj/aggregate.py > out/r3obj/aggregate.md

Inputs (all written by scripts/probe_object_sweep.py, this directory):
    ball_{off,abs}_s{0..4}.json   ball vs none, 5 seeds per mode
    null_{off,abs}_s{0..4}.json   none vs none (--null), 5 seeds per mode -- the null distribution of every diff_* field
    lad_{11deg_static,22deg,28deg,43deg}.json   angular-size ladder, off, seed 0
Scratch script kept with its outputs (out/ is git-ignored), as out/obj/obj_aggregate.py in round 2.
"""
import json
import os

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
TYPES = ["LC11", "LC10a", "LC10b", "LC16", "LPLC2", "LC4"]
SEEDS = [0, 1, 2, 3, 4]
STATS = ["diff_max_over_cells_mean_mv", "diff_mean_over_cells_mean_mv", "diff_tuning_peak_mv",
         "diff_peak_100ms_mv", "diff_rate_hz_max_cell", "diff_rate_hz_mean"]


def load(name):
    p = os.path.join(D, name + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def series(prefix, mode, stat, t):
    out = []
    for s in SEEDS:
        d = load(f"{prefix}_{mode}_s{s}")
        out.append(float("nan") if d is None else d["ball"][t][stat])
    return np.array(out)


def msd(x):
    x = x[~np.isnan(x)]
    return (float("nan"), float("nan"), 0) if len(x) == 0 else (x.mean(), x.std(ddof=1) if len(x) > 1 else float("nan"), len(x))


def fmt(v, n=4):
    return "n/a" if v != v else f"{v:+.{n}f}"


print("# Round-3 object sweep: replicates, none-vs-none null, z-scores\n")
# provenance
for name in ["ball_off_s0", "ball_abs_s0", "null_off_s0", "null_abs_s0", "lad_43deg"]:
    d = load(name)
    if d:
        c = d["config"]
        print(f"* `{name}`: mode {c['mode']}, receptor_model {c['receptor_model']}/{c['receptor_net_rule']}, null {c['null']}, "
              f"ball {c['ball_radius_m']} m at {c['ahead_m']} m ({c.get('angular_diameter_deg', float('nan')):.1f} deg), "
              f"half sweep {c['half_sweep_m']} m, {c['seconds']} s after {c['settle']} s, cache `{c['cache_dir']}`, {c['device']}, torch {c['torch']}")
print()

for stat in STATS:
    print(f"\n## {stat}\n")
    print("| type | off ball-none per seed | off mean | off SD | off null per seed | null mean | null SD | z(off) | "
          "abs ball-none per seed | abs mean | abs SD | abs null per seed | null mean | null SD | z(abs) |")
    print("|" + "---|" * 15)
    for t in TYPES:
        row = [t]
        for mode in ("off", "abs"):
            b = series("ball", mode, stat, t); n = series("null", mode, stat, t)
            bm, bs, _ = msd(b); nm, ns, _ = msd(n)
            z = (bm - nm) / ns if ns == ns and ns > 0 else float("nan")
            row += [" ".join(fmt(v, 3) for v in b), fmt(bm), fmt(bs), " ".join(fmt(v, 3) for v in n), fmt(nm), fmt(ns),
                    "n/a" if z != z else f"{z:+.1f}"]
        print("| " + " | ".join(row) + " |")

# z with the standard error of the two means (Welch), primary statistic only
print("\n## z-scores, primary statistic diff_max_over_cells_mean_mv\n")
print("| type | mode | mean(ball-none) | SD | mean(none-none) | SD | z = (mB-mN)/SD_null | Welch t = (mB-mN)/sqrt(sB^2/5+sN^2/5) | signal at z>3 |")
print("|" + "---|" * 9)
for t in TYPES:
    for mode in ("off", "abs"):
        b = series("ball", mode, "diff_max_over_cells_mean_mv", t); n = series("null", mode, "diff_max_over_cells_mean_mv", t)
        bm, bs, bn = msd(b); nm, ns, nn = msd(n)
        z = (bm - nm) / ns if ns == ns and ns > 0 else float("nan")
        w = (bm - nm) / np.sqrt(bs ** 2 / max(bn, 1) + ns ** 2 / max(nn, 1)) if bs == bs and ns == ns else float("nan")
        print(f"| {t} | {'off' if mode == 'off' else 'sign-abs'} | {fmt(bm)} | {fmt(bs)} | {fmt(nm)} | {fmt(ns)} | "
              f"{'n/a' if z != z else f'{z:+.1f}'} | {'n/a' if w != w else f'{w:+.1f}'} | "
              f"{'YES' if z == z and z > 3 else 'no'} |")

# absolute levels the difference is built from
print("\n## Levels (mean over the 5 seeds; A / B = ball / none, or none / none under the null)\n")
print("| type | mode | A time-mean drive mV | B time-mean drive mV | A max-cell rate Hz | B max-cell rate Hz | A peak mV | B peak mV |")
print("|" + "---|" * 8)
for t in TYPES:
    for prefix in ("ball", "null"):
        for mode in ("off", "abs"):
            va = {k: [] for k in ("drive_mean_mv", "rate_hz_max_cell", "drive_peak_mv")}
            vb = {k: [] for k in va}
            for s in SEEDS:
                d = load(f"{prefix}_{mode}_s{s}")
                if d:
                    for k in va:
                        va[k].append(d["ball"][t][k]); vb[k].append(d["none"][t][k])
            if not va["drive_mean_mv"]:
                continue
            m = lambda dd, k: float(np.mean(dd[k]))
            print(f"| {t} | {prefix} {mode} | {m(va,'drive_mean_mv'):+.3f} | {m(vb,'drive_mean_mv'):+.3f} | "
                  f"{m(va,'rate_hz_max_cell'):.2f} | {m(vb,'rate_hz_max_cell'):.2f} | {m(va,'drive_peak_mv'):+.2f} | {m(vb,'drive_peak_mv'):+.2f} |")

# verdict tally
print("\n## Per-run verdict of the 7 mV / 1 Hz criterion\n")
for prefix in ("ball", "null"):
    for mode in ("off", "abs"):
        v = []
        for s in SEEDS:
            d = load(f"{prefix}_{mode}_s{s}")
            if d:
                v.append(("PASS" if d["verdict"]["pass"] else "FAIL"))
        print(f"* {prefix} {mode}: " + " ".join(v))

# ladder
print("\n## Angular-size ladder (off, seed 0), diff_max_over_cells_mean_mv (mV)\n")
lad = [("11.4 deg static", "lad_11deg_static"), ("22.6 deg", "lad_22deg"), ("28.1 deg", "lad_28deg"), ("43.6 deg", "lad_43deg")]
print("| object | r (m) | ahead (m) | half sweep (m) | " + " | ".join(TYPES) + " | LPLC2 rate mean ball/none Hz | LPLC2 max cell ball/none Hz |")
print("|" + "---|" * (5 + len(TYPES)))
for label, name in lad:
    d = load(name)
    if not d:
        print(f"| {label} | MISSING |")
        continue
    c = d["config"]
    cells = " | ".join(fmt(d["ball"][t]["diff_max_over_cells_mean_mv"], 3) for t in TYPES)
    print(f"| {label} | {c['ball_radius_m']} | {c['ahead_m']} | {c['half_sweep_m']} | {cells} | "
          f"{d['ball']['LPLC2']['rate_hz_mean']:.3f}/{d['none']['LPLC2']['rate_hz_mean']:.3f} | "
          f"{d['ball']['LPLC2']['rate_hz_max_cell']:.2f}/{d['none']['LPLC2']['rate_hz_max_cell']:.2f} |")

# rate units
print("\n## Optic rate units, diff_abs_best_cell_mean (max over cells of the (A-B) time-mean |dev|)\n")
RATE = ["T2", "T3", "Tm5Y", "TmY21", "TmY13", "TmY5a", "Mi4", "Tm3", "Mi1"]
print("| type | ball off mean | null off mean | null off SD | z(off) | ball abs mean | null abs mean | null abs SD | z(abs) |")
print("|" + "---|" * 9)
for t in RATE:
    row = [t]
    for mode in ("off", "abs"):
        b = np.array([load(f"ball_{mode}_s{s}")["ball"][t]["diff_abs_best_cell_mean"] if load(f"ball_{mode}_s{s}") else np.nan for s in SEEDS])
        n = np.array([load(f"null_{mode}_s{s}")["ball"][t]["diff_abs_best_cell_mean"] if load(f"null_{mode}_s{s}") else np.nan for s in SEEDS])
        bm, _, _ = msd(b); nm, ns, _ = msd(n)
        z = (bm - nm) / ns if ns == ns and ns > 0 else float("nan")
        row += [fmt(bm), fmt(nm), fmt(ns), "n/a" if z != z else f"{z:+.1f}"]
    print("| " + " | ".join(row) + " |")
