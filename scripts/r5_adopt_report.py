"""Receptor round 5, GF-damping adoption: the decision table (CPU only; JSON parsing + scipy).

    PYTHONIOENCODING=utf-8 python scripts/r5_adopt_report.py \
        --new out/r5_adopt_default_{1,2,3}.json --pre out/r4_default_{1,2,3}.json --off out/r4_off_{1,2,3}.json \
        --retire out/retire_r4/no_gf_damping_r*/no_gf_damping.json \
        --sustain-new out/r5_adopt_sustain_live_{1,2,3}.json --sustain-pre out/r5_sustain_default_live_{1,2,3}.json \
        --sustain-off out/r5_sustain_off_live_{1,2,3}.json \
        --hops out/r5_adopt_hops.json out/r5_hops_default.json out/r5_hops_off.json

Part 1, the suite: every check of scripts/benchmark.py --seeds 0,1,2 under the edited default (NEW: brain.DEFAULT_TYPE_PATH_GAIN
without the GF x0.3 damping), the pre-retirement default (PRE: round-4 r4_default_*), off (OFF: round-4 r4_off_*) and, when
given, the round-4 retire_measures no_gf_damping runs (the same LIFParams reached through retire_measures.py).  The criterion of
docs/audits/anti_runaway.md ("adopt alone, with a status margin"): no check worse in status than PRE in 3/3, walk.power_max PASS
in 3/3; the margins to the bound are printed for every NEW run.
Part 2, take-offs: scripts/batch_sustain.py rows (16 flies x 300 s x 3 brain RNGs, live escape route) of NEW vs PRE and NEW vs
OFF: totals, rates per 1,000 fly-s (all / escape / voluntary), per-batch seed-matched and pooled Mann-Whitney with the
two-sided asymptotic p AND both one-sided alternatives ('new < other' and 'new > other'), the walking-GF tail (median, rows at
the threshold), and how much of the PRE - OFF excess the retirement removes.
Part 3, the benchmark 'hops' section JSONs: measured values and the status the run was scored with.
Console: out/r5_adopt_report.log (tee it); the numbers are also written to --json.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

def norm(status):
    """benchmark.py writes 'PASS (gap closed)' for a gap=True entry that passes; rank it as PASS."""
    return "PASS" if str(status).startswith("PASS") else status


RANK = {"PASS": 2, "KNOWN GAP": 1, "MISSING": 0, "FAIL": 0}
ABBR = {"PASS": "P", "FAIL": "F", "KNOWN GAP": "G", "MISSING": "M"}


def expand(patterns):
    out = []
    for p in patterns or []:
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else [p])
    return out


def load_suite(path):
    d = json.load(open(path, encoding="utf-8"))
    checks = {c["key"]: c for c in d["checks"]}
    tally = {s: sum(1 for c in checks.values() if c["status"] == s) for s in ("PASS", "FAIL", "KNOWN GAP")}
    return d, checks, tally


def margin(c):
    """Distance from the measured value to the bound in the direction of passing (positive = inside)."""
    m = c["measured"]
    if m is None:
        return None
    crit = c["criterion"]
    mm = re.match(r"^(abs)?\s*(<=|>=|<|>|==)\s*(-?[\d.]+)$", crit.replace("abs>=", "abs >="))
    if not mm:
        return None
    ab, op, b = mm.group(1), mm.group(2), float(mm.group(3))
    v = abs(m) if ab else m
    if op in ("<", "<="):
        return b - v
    if op in (">", ">="):
        return v - b
    return -abs(v - b)


def fmt(c):
    return "-- M" if c is None or c["measured"] is None else f"{c['measured']:.2f} {ABBR[norm(c['status'])]}"


def part1(a, out):
    groups = [("NEW", expand(a.new)), ("PRE", expand(a.pre)), ("OFF", expand(a.off)), ("RETIRE", expand(a.retire))]
    groups = [(n, f) for n, f in groups if f]
    loaded = {n: [load_suite(f) for f in fs] for n, fs in groups}
    print("=== Part 1: the suite ===")
    for n, fs in groups:
        for f, (d, ch, t) in zip(fs, loaded[n]):
            cfg = d["config"]
            print(f"  {n:6s} {f}: {t['PASS']}/{t['FAIL']}/{t['KNOWN GAP']}  receptor {cfg.get('receptor', {}).get('model')} "
                  f"({cfg.get('receptor', {}).get('flag')})  type_path_gain {cfg.get('type_path_gain')}  device {cfg.get('device')}  "
                  f"date {d.get('date')}  runtime {d.get('total_runtime_s', float('nan')):.0f} s")
    keys = []
    for n in loaded:
        for _, ch, _ in loaded[n]:
            for k in ch:
                if k not in keys:
                    keys.append(k)
    header = "| check | criterion | " + " | ".join(f"{n} x{len(loaded[n])}" for n in loaded) + " |"
    print(header); print("|---|---|" + "---|" * len(loaded))
    table = {}
    for k in keys:
        crit = next((ch[k]["criterion"] for n in loaded for _, ch, _ in loaded[n] if k in ch), "")
        cells = []
        for n in loaded:
            cells.append(" / ".join(fmt(ch.get(k)) for _, ch, _ in loaded[n]))
        print(f"| {k} | {crit} | " + " | ".join(cells) + " |")
        table[k] = {n: [(ch.get(k) or {}).get("measured") for _, ch, _ in loaded[n]] for n in loaded}
        table[k]["status"] = {n: [(ch.get(k) or {}).get("status") for _, ch, _ in loaded[n]] for n in loaded}
    out["suite_table"] = table
    if "NEW" in loaded and "PRE" in loaded:
        worse = []
        for k in keys:
            for i, (_, chn, _) in enumerate(loaded["NEW"]):
                for j, (_, chp, _) in enumerate(loaded["PRE"]):
                    if k in chn and k in chp and RANK[norm(chn[k]["status"])] < RANK[norm(chp[k]["status"])]:
                        worse.append((k, i + 1, chn[k]["status"], j + 1, chp[k]["status"]))
        print(f"\ncriterion (a) no check worse in status than any PRE run, for every NEW run: {'HOLDS' if not worse else 'VIOLATED'} {worse}")
        better = sorted({k for k in keys for _, chn, _ in loaded["NEW"] for _, chp, _ in loaded["PRE"]
                         if k in chn and k in chp and RANK[norm(chn[k]["status"])] > RANK[norm(chp[k]["status"])]})
        print(f"checks better in status in some NEW vs some PRE run: {better}")
        pm = [norm(ch["walk.power_max_hz"]["status"]) for _, ch, _ in loaded["NEW"]]
        pv = [ch["walk.power_max_hz"]["measured"] for _, ch, _ in loaded["NEW"]]
        print(f"criterion (b) walk.power_max PASS in 3/3: {'HOLDS' if pm.count('PASS') == len(pm) == 3 else 'VIOLATED'} {pv} {pm}")
        tallies = [f"{t['PASS']}/{t['FAIL']}/{t['KNOWN GAP']}" for _, _, t in loaded["NEW"]]
        print(f"NEW tallies: {tallies}; PRE: {[f'{t['PASS']}/{t['FAIL']}/{t['KNOWN GAP']}' for _, _, t in loaded['PRE']]}")
        out["criterion_a_worse"] = worse; out["criterion_b_power_max"] = pv; out["new_tallies"] = tallies
        print("\nmargins to the bound, NEW runs (min over the three), sorted ascending; 'x' marks a FAIL / KNOWN GAP:")
        rows = []
        for k in keys:
            ms = [margin(ch[k]) for _, ch, _ in loaded["NEW"] if k in ch]
            st = [norm(ch[k]["status"]) for _, ch, _ in loaded["NEW"] if k in ch]
            if ms and all(m is not None for m in ms):
                rows.append((min(ms), k, st))
        for m, k, st in sorted(rows):
            print(f"   {m:9.3f}  {k:40s} {'x' if any(s != 'PASS' for s in st) else ' '}")
        out["new_margins"] = {k: m for m, k, _ in rows}
        print("\nvalue shifts NEW - PRE (mean over runs) for the checks whose |shift| > 0.01:")
        for k in keys:
            vn = [ch[k]["measured"] for _, ch, _ in loaded["NEW"] if k in ch and ch[k]["measured"] is not None]
            vp = [ch[k]["measured"] for _, ch, _ in loaded["PRE"] if k in ch and ch[k]["measured"] is not None]
            if vn and vp and abs(np.mean(vn) - np.mean(vp)) > 0.01:
                print(f"   {k:40s} NEW {np.mean(vn):9.3f} [{min(vn):.3f}, {max(vn):.3f}]   PRE {np.mean(vp):9.3f} [{min(vp):.3f}, {max(vp):.3f}]   "
                      f"shift {np.mean(vn) - np.mean(vp):+.3f}")
    if "NEW" in loaded and "RETIRE" in loaded:
        print("\nNEW vs the round-4 retire_measures no_gf_damping runs (same LIFParams by a different script), bit-comparison of the walk / smell / taste checks:")
        for k in ("walk.power_max_hz", "walk.GF_max_hz", "walk.power_sustained_hz", "taste.MN9_hz", "smell.KC_active", "smell.PN_hz",
                  "bitter.shiu_sugar_MN9_hz", "motion.min_dsi", "loom.GF_peak_hz"):
            vn = [ch[k]["measured"] for _, ch, _ in loaded["NEW"] if k in ch]
            vr = [ch[k]["measured"] for _, ch, _ in loaded["RETIRE"] if k in ch]
            print(f"   {k:28s} NEW {vn}   RETIRE {vr}   identical set: {set(vn) == set(vr)}")


def sustain_rows(files):
    runs = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        runs.append((f, d))
    return runs


def rates(runs):
    fs = sum(float(d.get("fly_s", d["batch"] * d["simulated_s"])) for _, d in runs)
    tot = {m: sum(int(r[m]) for _, d in runs for r in d["rows"]) for m in ("hops", "hops_escape", "hops_voluntary")}
    return fs, tot, {m: tot[m] / fs * 1000 for m in tot}


def mw3(a, b):
    """U and the asymptotic p two-sided, one-sided a<b ('less') and a>b ('greater')."""
    two = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    less = stats.mannwhitneyu(a, b, alternative="less", method="asymptotic").pvalue
    greater = stats.mannwhitneyu(a, b, alternative="greater", method="asymptotic").pvalue
    return float(two.statistic), float(two.pvalue), float(less), float(greater)


def contrast(label_a, ra, label_b, rb, out, key):
    print(f"\n--- {label_a} vs {label_b} ---")
    res = {}
    for m in ("hops", "hops_escape", "hops_voluntary", "gf_max_walk_hz"):
        print(f"  {m}:")
        for (fa, da), (fb, db) in zip(ra, rb):
            va = np.array([r[m] for r in da["rows"]], float); vb = np.array([r[m] for r in db["rows"]], float)
            U, p2, pl, pg = mw3(va, vb)
            tot = f"total {int(va.sum())} vs {int(vb.sum())}" if m != "gf_max_walk_hz" else f"median {np.median(va):.2f} vs {np.median(vb):.2f}"
            print(f"     batch brain_seed {da.get('brain_seed')} / {db.get('brain_seed')}: mean {va.mean():.3f}+-{va.std(ddof=1):.3f} vs "
                  f"{vb.mean():.3f}+-{vb.std(ddof=1):.3f}  {tot}  U {U:.1f}  p2 {p2:.3g}  p({label_a}<{label_b}) {pl:.3g}  p({label_a}>{label_b}) {pg:.3g}")
        va = np.array([r[m] for _, d in ra for r in d["rows"]], float); vb = np.array([r[m] for _, d in rb for r in d["rows"]], float)
        U, p2, pl, pg = mw3(va, vb)
        if m == "gf_max_walk_hz":
            ga = [float(d.get("flight", {}).get("gf_hz", 33.0)) for _, d in ra][0]
            tot = (f"median {np.median(va):.2f} vs {np.median(vb):.2f}; rows >= {ga:g} Hz {int((va >= ga).sum())}/{len(va)} vs "
                   f"{int((vb >= ga).sum())}/{len(vb)}")
        else:
            fsa, _, rta = rates(ra); fsb, _, rtb = rates(rb)
            tot = f"total {int(va.sum())} vs {int(vb.sum())}; per 1,000 fly-s {rta[m]:.3f} vs {rtb[m]:.3f} ({fsa:.0f} / {fsb:.0f} fly-s)"
        print(f"     pooled {len(va)} v {len(vb)}: mean {va.mean():.4f}+-{va.std(ddof=1):.4f} vs {vb.mean():.4f}+-{vb.std(ddof=1):.4f}  {tot}  "
              f"U {U:.1f}  p2 {p2:.3g}  p({label_a}<{label_b}) {pl:.3g}  p({label_a}>{label_b}) {pg:.3g}")
        res[m] = {"a": va.tolist(), "b": vb.tolist(), "U": U, "p_two_sided": p2, "p_a_less": pl, "p_a_greater": pg}
    out[key] = res


def part2(a, out):
    rn, rp, ro = sustain_rows(expand(a.sustain_new)), sustain_rows(expand(a.sustain_pre)), sustain_rows(expand(a.sustain_off))
    if not rn:
        return
    print("\n=== Part 2: take-offs (batch_sustain rows, live escape route) ===")
    for label, runs in (("NEW", rn), ("PRE", rp), ("OFF", ro)):
        for f, d in runs:
            rec = d.get("receptor", {}); fl = d.get("flight", {})
            v = np.array([r["hops"] for r in d["rows"]]); e = np.array([r["hops_escape"] for r in d["rows"]]); w = np.array([r["hops_voluntary"] for r in d["rows"]])
            g = np.array([r["gf_max_walk_hz"] for r in d["rows"]], float)
            print(f"  {label:4s} {f}: brain_seed {d.get('brain_seed')} receptor {rec.get('model')}/{rec.get('net_rule')} changed {rec.get('fast_sign_changed_entries')} "
                  f"gf_hz {fl.get('gf_hz')} fly_s {d.get('fly_s')} wall {d.get('wall_s', float('nan')):.0f} s  hops {int(v.sum())} = escape {int(e.sum())} + voluntary {int(w.sum())}"
                  f"  walking-GF median {np.median(g):.2f} rows>=thr {int((g >= float(fl.get('gf_hz', 33))).sum())}/{len(g)}  meals {sum(r['meals'] for r in d['rows'])}")
            print(f"         hops per fly {v.tolist()}  escape {e.tolist()}  voluntary {w.tolist()}")
        if runs:
            fs, tot, rt = rates(runs)
            print(f"  {label:4s} pooled: {fs:.0f} fly-s, hops {tot['hops']} (escape {tot['hops_escape']}, voluntary {tot['hops_voluntary']}); per 1,000 fly-s "
                  f"all {rt['hops']:.3f} escape {rt['hops_escape']:.3f} voluntary {rt['hops_voluntary']:.3f}")
            out[f"rates_{label}"] = {"fly_s": fs, "totals": tot, "per_1000_fly_s": rt}
    if rp:
        contrast("NEW", rn, "PRE", rp, out, "new_vs_pre")
    if ro:
        contrast("NEW", rn, "OFF", ro, out, "new_vs_off")
    if rp and ro:
        contrast("PRE", rp, "OFF", ro, out, "pre_vs_off")
        _, _, rtn = rates(rn); _, _, rtp = rates(rp); _, _, rto = rates(ro)
        print("\nthe excess (rate per 1,000 fly-s above OFF) and how much of it the retirement removes:")
        for m in ("hops", "hops_escape", "hops_voluntary"):
            ex_pre, ex_new = rtp[m] - rto[m], rtn[m] - rto[m]
            frac = (ex_pre - ex_new) / ex_pre if ex_pre else float("nan")
            print(f"   {m:15s} PRE {rtp[m]:.3f}  NEW {rtn[m]:.3f}  OFF {rto[m]:.3f}   excess PRE {ex_pre:+.3f}  NEW {ex_new:+.3f}   removed {100 * frac:.0f} %   "
                  f"NEW/PRE {rtn[m] / rtp[m] if rtp[m] else float('nan'):.2f}x")
            out[f"excess_{m}"] = {"pre": ex_pre, "new": ex_new, "fraction_removed": frac}


def part3(a, out):
    files = expand(a.hops)
    if not files:
        return
    print("\n=== Part 3: benchmark.py 'hops' section JSONs (status = as scored by the bounds in benchmark.py at run time) ===")
    res = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        cfg = d["config"]; sec = d.get("sections", {}).get("hops", {})
        print(f"  {f}: receptor {cfg.get('receptor', {}).get('model')} ({cfg.get('receptor', {}).get('flag')}) type_path_gain {cfg.get('type_path_gain')} "
              f"device {cfg.get('device')} fly_s {sec.get('fly_s')} hops {sec.get('hops_total')} = escape {sec.get('escape_total')} + voluntary {sec.get('voluntary_total')} "
              f"route_split_consistent {sec.get('route_split_consistent')}")
        for c in d["checks"]:
            if c["key"].startswith("hops."):
                print(f"     {c['key']:32s} {c['measured']:.4f}  {c['criterion']:8s} {c['status']}")
        res[f] = {c["key"]: (c["measured"], c["status"]) for c in d["checks"] if c["key"].startswith("hops.")}
    out["hops_sections"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs="*", default=[]); ap.add_argument("--pre", nargs="*", default=[]); ap.add_argument("--off", nargs="*", default=[])
    ap.add_argument("--retire", nargs="*", default=[])
    ap.add_argument("--sustain-new", nargs="*", default=[]); ap.add_argument("--sustain-pre", nargs="*", default=[]); ap.add_argument("--sustain-off", nargs="*", default=[])
    ap.add_argument("--hops", nargs="*", default=[])
    ap.add_argument("--json", default="out/r5_adopt_report.json")
    a = ap.parse_args()
    out = {}
    if a.new:
        part1(a, out)
    part2(a, out); part3(a, out)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print("\nwrote", a.json)


if __name__ == "__main__":
    main()
