"""Closing-round recount for the receptor-expression integration (CPU, no simulation).

Re-derives, from the JSON files named below, every number the closing statement quotes: the suite tallies of the
shipped default vs the pre-retirement default vs off, the room take-off split (escape / voluntary) per batch and
pooled with Mann-Whitney over flies, the identical-seed rerun scatter, the four `hops` section draws, and the
attribution endpoints over three run directories.  Writes out/r5_close_recount.log.

    PYTHONIOENCODING=utf-8 python scripts/r5_close_recount.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

try:
    from scipy.stats import mannwhitneyu, poisson
except ImportError:  # pragma: no cover
    mannwhitneyu = poisson = None

OUT = "out/r5_close_recount.log"
_lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _lines.append(s)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------- suite runs
SUITE = {
    "shipped default (r5_adopt_default x3 + r5_skeptic_default_4)": ["out/r5_adopt_default_1.json", "out/r5_adopt_default_2.json",
                                                                     "out/r5_adopt_default_3.json", "out/r5_skeptic_default_4.json"],
    "pre-retirement default (r4_default x3 + sk4_default_4)": ["out/r4_default_1.json", "out/r4_default_2.json", "out/r4_default_3.json",
                                                              "out/sk4_default_4.json"],
    "off, pre-retirement gains (r4_off x3 + sk4_off_4)": ["out/r4_off_1.json", "out/r4_off_2.json", "out/r4_off_3.json", "out/sk4_off_4.json"],
}
KEYS = ["walk.power_max_hz", "walk.power_sustained_hz", "walk.GF_max_hz", "loom.GF_peak_hz", "loom_escape.GF_peak_hz",
        "loom_escape.escapes", "motion.min_dsi", "taste.MN9_hz", "smell.KC_active", "bitter.shiu_sugar_MN9_hz", "walk_gf.p99_hz"]


def suite():
    say("=" * 100)
    say("A. SUITE (benchmark.py --seeds 0,1,2, 29 checks)")
    for label, paths in SUITE.items():
        say(f"\n{label}")
        for p in paths:
            if not os.path.exists(p):
                say(f"  MISSING {p}"); continue
            d = load(p)
            checks = {c["key"]: c for c in d["checks"]}
            tally = {}
            for c in d["checks"]:
                tally[c["status"]] = tally.get(c["status"], 0) + 1
            fails = [c["key"] for c in d["checks"] if c["status"] == "FAIL"]
            tpg = d["config"].get("type_path_gain")
            rc = d["config"].get("receptor", {})
            say(f"  {p}: PASS {tally.get('PASS', 0)} FAIL {tally.get('FAIL', 0)} GAP {tally.get('KNOWN GAP', 0)} "
                f"fails={fails} type_path_gain entries={len(tpg) if tpg else None} receptor={rc.get('model')}/{rc.get('net_rule')} "
                f"changed={rc.get('fast_sign_changed_entries')} device={d['config'].get('device')}")
            say("    " + "  ".join(f"{k.split('.')[1]}={checks[k]['measured']:.4f}" if k in checks and isinstance(checks[k]['measured'], float)
                                   else f"{k.split('.')[1]}={checks[k]['measured']}" for k in KEYS if k in checks))


# ------------------------------------------------------------------------------------------------- room batches
ROOM = {
    "shipped default, live (r5_adopt_sustain_live)": sorted(glob.glob("out/r5_adopt_sustain_live_[123].json")),
    "pre-retirement default, live (r5_sustain_default_live)": sorted(glob.glob("out/r5_sustain_default_live_[123].json")),
    "off, live (r5_sustain_off_live)": sorted(glob.glob("out/r5_sustain_off_live_[123].json")),
    "pre-retirement default, gf_hz 1e9 (r5_sustain_default_nogf)": sorted(glob.glob("out/r5_sustain_default_nogf_[123].json")),
    "off, gf_hz 1e9 (r5_sustain_off_nogf)": sorted(glob.glob("out/r5_sustain_off_nogf_[123].json")),
}
RERUNS = {
    "shipped default seed 0 (r5_adopt_sustain_live_1 vs r5_skeptic_sustain_live_4)": ["out/r5_adopt_sustain_live_1.json", "out/r5_skeptic_sustain_live_4.json"],
    "pre-retirement default seed 0 (r5_sustain_default_live_1 vs sk5_sustain_default_live_1r vs round-4 r4_sustain_default_1 / _1b / sk4_route_default_1)":
        ["out/r5_sustain_default_live_1.json", "out/sk5_sustain_default_live_1r.json", "out/r4_sustain_default_1.json",
         "out/r4_sustain_default_1b.json", "out/sk4_route_default_1.json"],
    "off seed 0 (r5_sustain_off_live_1 vs sk5_sustain_off_live_1r vs r4_sustain_off_1 / _1b / sk4_route_off_1)":
        ["out/r5_sustain_off_live_1.json", "out/sk5_sustain_off_live_1r.json", "out/r4_sustain_off_1.json",
         "out/r4_sustain_off_1b.json", "out/sk4_route_off_1.json"],
}


def rows_of(d):
    rows = d.get("rows") or d.get("flies") or []
    return rows


def room_arm(paths):
    per = []
    for p in paths:
        d = load(p)
        rows = rows_of(d)
        hops = np.array([r.get("hops", 0) for r in rows]); esc = np.array([r.get("hops_escape", -1) for r in rows])
        vol = np.array([r.get("hops_voluntary", -1) for r in rows]); gf = np.array([r.get("gf_max_walk_hz", np.nan) for r in rows], float)
        thr = (d.get("flight") or {}).get("gf_hz")
        per.append(dict(path=p, hops=hops, esc=esc, vol=vol, gf=gf, fly_s=d.get("fly_s"), thr=thr,
                        receptor=(d.get("receptor") or {}).get("model"), tpg=d.get("options", {}).get("type_path_gain")))
    return per


def mw(a, b):
    if mannwhitneyu is None or len(a) == 0 or len(b) == 0:
        return "n/a"
    r2 = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    rl = mannwhitneyu(a, b, alternative="less", method="asymptotic")
    rg = mannwhitneyu(a, b, alternative="greater", method="asymptotic")
    return f"U {r2.statistic:.1f} p2 {r2.pvalue:.3g} p(a<b) {rl.pvalue:.3g} p(a>b) {rg.pvalue:.3g}"


def room():
    say("\n" + "=" * 100)
    say("B. ROOM TAKE-OFFS (batch_sustain.py --batch 16 --minutes 5 --energy 0.9, cx / apple / fence)")
    arms = {}
    for label, paths in ROOM.items():
        per = room_arm(paths)
        arms[label] = per
        say(f"\n{label}: {len(per)} batches")
        for b in per:
            split_ok = bool(np.all(b["hops"] == b["esc"] + b["vol"])) if b["esc"].min() >= 0 else None
            med = float(np.nanmedian(b["gf"])) if np.isfinite(b["gf"]).any() else float("nan")
            at = int(np.sum(b["gf"] >= 33.0)) if np.isfinite(b["gf"]).any() else -1
            say(f"  {b['path']}: hops {int(b['hops'].sum())} = escape {int(b['esc'].sum())} + voluntary {int(b['vol'].sum())} "
                f"(split==hops: {split_ok}); fly_s {b['fly_s']}; flight.gf_hz {b['thr']}; receptor {b['receptor']}; "
                f"walking-GF median {med:.2f} Hz, rows >= 33 Hz {at}/{len(b['gf'])}")
        H = np.concatenate([b["hops"] for b in per]); E = np.concatenate([b["esc"] for b in per]); V = np.concatenate([b["vol"] for b in per])
        G = np.concatenate([b["gf"] for b in per]); fs = sum(b["fly_s"] for b in per)
        say(f"  POOLED n={len(H)} fly-s {fs:.0f}: hops {int(H.sum())} = {int(E.sum())} + {int(V.sum())}; per 1,000 fly-s all {1000*H.sum()/fs:.3f} "
            f"escape {1000*E.sum()/fs:.3f} voluntary {1000*V.sum()/fs:.3f}; per-batch rates all "
            f"{[round(1000*b['hops'].sum()/b['fly_s'],3) for b in per]} escape {[round(1000*b['esc'].sum()/b['fly_s'],3) for b in per]} "
            f"voluntary {[round(1000*b['vol'].sum()/b['fly_s'],3) for b in per]}; walking-GF median {np.nanmedian(G):.2f} Hz, "
            f"rows >= 33 Hz {int(np.sum(G >= 33.0))}/{len(G)}; per-batch medians {[round(float(np.nanmedian(b['gf'])),2) for b in per]}")
    L = list(ROOM)
    pairs = [(L[0], L[1]), (L[0], L[2]), (L[1], L[2]), (L[3], L[4])]
    say("\nMann-Whitney over flies (asymptotic; counts are tied so no exact p), a vs b:")
    for a, b in pairs:
        A = arms[a]; B = arms[b]
        for metric in ["hops", "esc", "vol", "gf"]:
            x = np.concatenate([p[metric] for p in A]); y = np.concatenate([p[metric] for p in B])
            x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
            say(f"  [{a.split(' (')[0]}] vs [{b.split(' (')[0]}] {metric:4s}: {mw(x, y)}")
    say("\nSeed-matched per-batch two-sided p, shipped vs pre-retirement (hops / escape / voluntary):")
    for i in range(3):
        A = arms[L[0]][i]; B = arms[L[1]][i]
        say(f"  batch {i+1}: " + "  ".join(f"{m} {mannwhitneyu(A[m], B[m], alternative='two-sided', method='asymptotic').pvalue:.3f}" for m in ["hops", "esc", "vol"]))
    say("\nIDENTICAL-SEED RERUNS (brain seed 0, environment seeds 0-15, identical command):")
    for label, paths in RERUNS.items():
        say(f"  {label}")
        vals = []
        for p in paths:
            if not os.path.exists(p):
                say(f"    MISSING {p}"); continue
            d = load(p); rows = rows_of(d)
            hops = np.array([r.get("hops", 0) for r in rows]); esc = [r.get("hops_escape") for r in rows]; vol = [r.get("hops_voluntary") for r in rows]
            gf = np.array([r.get("gf_max_walk_hz", np.nan) for r in rows], float)
            e = int(sum(x for x in esc if x is not None)) if all(x is not None for x in esc) else "-"
            v = int(sum(x for x in vol if x is not None)) if all(x is not None for x in vol) else "-"
            med = f"{np.nanmedian(gf):.2f}" if np.isfinite(gf).any() else "-"
            say(f"    {p}: hops {int(hops.sum())} = escape {e} + voluntary {v}; walking-GF median {med}; rows>=33 "
                f"{int(np.sum(gf >= 33)) if np.isfinite(gf).any() else '-'}/{len(rows)}")
            vals.append(hops)
        if len(vals) >= 2 and mannwhitneyu is not None:
            say(f"    first two reruns against each other: {mw(vals[0], vals[1])}")


# ------------------------------------------------------------------------------------------------- hops sections
HOPS = ["out/r5_hops_default.json", "out/r5_hops_off.json", "out/r5_adopt_hops.json", "out/r5_skeptic_hops.json", "out/sk5_hops_default_rerun.json"]


def hops():
    say("\n" + "=" * 100)
    say("C. THE `hops` SECTION (benchmark.py --sections hops: 16 flies x 150 s = 2,400 fly-s, one draw each)")
    for p in HOPS:
        if not os.path.exists(p):
            say(f"  MISSING {p}"); continue
        d = load(p); s = d["sections"]["hops"]; tpg = d["config"].get("type_path_gain"); rc = d["config"].get("receptor", {})
        st = {c["key"].split(".")[1]: (round(c["measured"], 3), c["status"]) for c in d["checks"]}
        launches = s.get("launches") or []
        first = min((l[1] if isinstance(l, (list, tuple)) else l.get("t")) for l in launches) if launches else None
        say(f"  {p}: gains {'retired' if tpg and len(tpg) == 1 else 'damped'} ({len(tpg) if tpg else '?'} entries), receptor {rc.get('model')}; "
            f"hops {s['hops_total']} = escape {s['escape_total']} + voluntary {s['voluntary_total']}; "
            f"voluntary {s['voluntary_per_1000_fly_s']:.3f} escape {s['escape_per_1000_fly_s']:.3f} per 1,000 fly-s; "
            f"walking-GF median {s['walk_gf_max_median_hz']:.2f}; rows>=33 {s.get('rows_gf_at_threshold')}; split consistent {s.get('route_split_consistent')}; "
            f"statuses {st}; first launch t={first}")
    if poisson is not None:
        say("  Poisson, 2,400 fly-s section at the 300-s room rates: "
            f"P(X<=1 | 2.222*2.4={2.222*2.4:.2f}) = {poisson.cdf(1, 2.222*2.4):.4f}; "
            f"P(X<=3 | 3.056*2.4={3.056*2.4:.2f}) = {poisson.cdf(3, 3.056*2.4):.4f}; "
            f"P(X<=7 | 5.208*2.4={5.208*2.4:.2f}) = {poisson.cdf(7, 5.208*2.4):.4f}; "
            f"P(default passes '< 1.0' i.e. X<=2 | 2.222*2.4) = {poisson.cdf(2, 2.222*2.4):.4f}; "
            f"P(X<=2 | 3.056*2.4) = {poisson.cdf(2, 3.056*2.4):.4f}")


# ------------------------------------------------------------------------------------------------- attribution
ATTR = {
    "default": ["out/r5_attr_default_1.json", "out/r5_attr_dup/r5_attr_default_1.json", "out/sk/sk_attr_default.json"],
    "holdBrain (optic side only)": ["out/r5_attr_holdBrain_1.json", "out/r5_attr_holdBrain_2.json", "out/r5_attr_dup/r5_attr_holdBrain_1.json",
                                    "out/r5_attr_dup/r5_attr_holdBrain_2.json", "out/sk/sk_attr_holdBrain.json"],
    "holdOptic (Brain side only)": ["out/r5_attr_holdOptic_1.json", "out/r5_attr_holdOptic_2.json", "out/r5_attr_dup/r5_attr_holdOptic_1.json",
                                    "out/r5_attr_dup/r5_attr_holdOptic_2.json", "out/sk/sk_attr_holdOptic.json"],
    "off": ["out/r5_attr_off_1.json", "out/r5_attr_dup/r5_attr_off_1.json", "out/sk/sk_attr_off.json"],
}
AKEYS = ["taste.MN9_hz", "smell.KC_active", "smell.PN_hz", "bitter.shiu_sugar_MN9_hz", "walk.power_max_hz", "walk.power_sustained_hz",
         "walk.GF_max_hz", "loom.GF_peak_hz", "motion.min_dsi", "rotate.DNp20_flip_hz"]


def attribution():
    say("\n" + "=" * 100)
    say("D. ATTRIBUTION (benchmark.py --sections rest,taste,smell,walk,bitter,motion; three run dirs: r5-attr-6aa260, r5-attr-05bc30, sk-attr-6666df)")
    for label, paths in ATTR.items():
        say(f"\n{label}:")
        vals = {k: [] for k in AKEYS}; tallies = []
        for p in paths:
            if not os.path.exists(p):
                say(f"  MISSING {p}"); continue
            d = load(p); checks = {c["key"]: c for c in d["checks"]}
            t = {}
            for c in d["checks"]:
                t[c["status"]] = t.get(c["status"], 0) + 1
            tallies.append((os.path.basename(p), f"{t.get('PASS',0)}/{t.get('FAIL',0)}", [c['key'] for c in d['checks'] if c['status']=='FAIL'],
                            d["config"].get("receptor", {}).get("fast_sign_changed_entries")))
            for k in AKEYS:
                if k in checks:
                    vals[k].append(checks[k]["measured"])
        for tl in tallies:
            say(f"  {tl[0]}: PASS/FAIL {tl[1]} fails {tl[2]} changed entries {tl[3]}")
        for k in AKEYS:
            v = vals[k]
            if not v:
                continue
            fmt = (lambda x: f"{x:.4f}") if isinstance(v[0], float) else str
            say(f"  {k}: {[fmt(x) for x in v]}  (min {fmt(min(v))} max {fmt(max(v))}, distinct {len(set(v))})")


def main():
    suite(); room(); hops(); attribution()
    os.makedirs("out", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
