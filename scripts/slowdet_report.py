"""Score the round-4 `slowdet` batch (scripts/slowdet_batch.py): is the walk / motion pair bit-reproducible under
`benchmark.py --deterministic`, does the term-off control's motion scatter vanish, and does the walk.power_max cost
belong to the slow term or to the --dopamine-lead dop1r1 slow signs.

Reads out/r4_sd_<arm>_<i>.json, prints (1) a provenance line per run, (2) per arm the exact float repr of every
reported check plus a leaf-by-leaf count of how many of the sections' numbers differ between replicates, (3) the
cross-arm table, (4) runtimes.  Nothing is rounded: bit-identity is decided on `repr(float)`.

    python scripts/slowdet_report.py                       # default glob, writes out/r4_slowdet_scores.txt
    python scripts/slowdet_report.py --glob 'out/r4_sd_*.json' --out out/r4_slowdet_scores.txt
"""
from __future__ import annotations

import argparse
import glob as globmod
import itertools
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARM_ORDER = ["det_ctl", "nd_ctl", "det_dop", "nd_dop", "det_ecr", "nd_ecr"]
KEYS = ["walk.power_max_hz", "walk.power_sustained_hz", "walk.GF_max_hz", "loom.GF_peak_hz", "loom.escape_cm",
        "rotate.DNp20_flip_hz", "motion.min_dsi", "motion.correct_directions"]


def leaves(obj, prefix=""):
    """{dotted path: value} over every scalar leaf of a nested dict / list."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(leaves(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def arm_of(path):
    m = re.search(r"r4_sd_(.+)_(\d+)\.json$", os.path.basename(path))
    return (m.group(1), int(m.group(2))) if m else (os.path.basename(path), 0)


def check_map(d):
    return {c["key"]: c for c in d.get("checks", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="out/r4_sd_*.json")
    ap.add_argument("--out", default="out/r4_slowdet_scores.txt")
    a = ap.parse_args()
    files = sorted(globmod.glob(os.path.join(ROOT, a.glob)))
    if not files:
        raise SystemExit(f"no files match {a.glob}")
    runs = {}
    for f in files:
        arm, i = arm_of(f)
        runs.setdefault(arm, {})[i] = (f, json.load(open(f, encoding="utf-8")))
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p(f"== {len(files)} run(s) from {a.glob}")
    p()
    p("-- provenance")
    for arm in [x for x in ARM_ORDER if x in runs] + [x for x in runs if x not in ARM_ORDER]:
        for i in sorted(runs[arm]):
            f, d = runs[arm][i]
            cfg, rec = d["config"], d["config"].get("receptor", {})
            det = cfg.get("deterministic", {})
            slow = rec.get("slow", {})
            lead = (rec.get("dopamine_lead") or {}).get("lead", "shipped(DopEcR)")
            err = [k for k, v in d["sections"].items() if isinstance(v, dict) and "error" in v]
            p(f"  {arm}_{i}  {os.path.basename(f):28s} det={bool(det.get('flag'))} "
              f"cublas={det.get('cublas_workspace_config')} torch={det.get('torch','-')} dev={cfg.get('device')} "
              f"backend={cfg.get('backend')} model={rec.get('model')}/{rec.get('net_rule')} lead={lead} "
              f"slow.active={slow.get('active')} gain={slow.get('gain_by_class')} mode={slow.get('mode')} "
              f"changed={rec.get('fast_sign_changed_entries')} cache={cfg.get('cache_dir')}"
              + (f"  ERRORS={err}" if err else ""))
            for k in err:
                p(f"      {k}: {d['sections'][k]['error']}")
    p()
    p("-- reported checks (exact repr; one column per replicate)")
    hdr = f"  {'key':28s} " + "  ".join(f"{arm}" for arm in ARM_ORDER if arm in runs)
    p(hdr)
    for key in KEYS:
        row = [f"  {key:28s}"]
        for arm in [x for x in ARM_ORDER if x in runs]:
            vals = []
            for i in sorted(runs[arm]):
                cm = check_map(runs[arm][i][1])
                vals.append(repr(cm[key]["measured"]) if key in cm else "-")
            row.append(f"{arm}: " + " | ".join(vals))
        p(row[0])
        for r in row[1:]:
            p(f"      {r}")
    p()
    p("-- status tally per run (PASS/FAIL/GAP/MISSING)")
    for arm in [x for x in ARM_ORDER if x in runs]:
        t = []
        for i in sorted(runs[arm]):
            cs = runs[arm][i][1]["checks"]
            t.append(f"{sum(c['status'].startswith('PASS') for c in cs)}/"
                     f"{sum(c['status'] == 'FAIL' for c in cs)}/"
                     f"{sum(c['status'] == 'KNOWN GAP' for c in cs)}/"
                     f"{sum(c['status'] == 'MISSING' for c in cs)}")
        p(f"  {arm:9s} " + "  ".join(t) + "   fails: " +
          "; ".join(",".join(c["key"] for c in runs[arm][i][1]["checks"] if c["status"] == "FAIL") or "-"
                    for i in sorted(runs[arm])))
    p()
    p("-- within-arm leaf diff (sections tree; 'identical' = every scalar leaf equal)")
    for arm in [x for x in ARM_ORDER if x in runs]:
        idx = sorted(runs[arm])
        if len(idx) < 2:
            p(f"  {arm:9s} 1 replicate only")
            continue
        for i, j in itertools.combinations(idx, 2):
            la, lb = leaves(runs[arm][i][1]["sections"]), leaves(runs[arm][j][1]["sections"])
            keys = sorted(set(la) | set(lb))
            diff = [k for k in keys if la.get(k) != lb.get(k)]
            for sec in ("walk", "motion"):
                sk = [k for k in keys if k.startswith(sec + ".")]
                sd = [k for k in diff if k.startswith(sec + ".")]
                p(f"  {arm}_{i} vs _{j}  {sec:7s} {len(sd):4d} / {len(sk):4d} leaves differ"
                  + ("   IDENTICAL" if not sd else "   e.g. " + "; ".join(
                      f"{k} {la.get(k)!r} vs {lb.get(k)!r}" for k in sd[:3])))
    p()
    p("-- cross-arm: distinct values per arm")
    for key in KEYS:
        p(f"  {key}")
        for arm in [x for x in ARM_ORDER if x in runs]:
            vals = [check_map(runs[arm][i][1]).get(key, {}).get("measured") for i in sorted(runs[arm])]
            uniq = sorted({repr(v) for v in vals})
            p(f"      {arm:9s} n={len(vals)} distinct={len(uniq)}  " + " | ".join(repr(v) for v in vals))
    p()
    p("-- attribution: condition = {ctl: term off, ecr: term on + shipped DopEcR table, dop: term on + dop1r1 table}")
    p("   term = ecr - ctl (the slow term at fixed fast weights), lead = dop - ecr (the dop1r1 slow signs alone)")
    cond = {"ctl": ["det_ctl", "nd_ctl"], "ecr": ["det_ecr", "nd_ecr"], "dop": ["det_dop", "nd_dop"]}
    for key in KEYS:
        vals = {}
        for cname, arms in cond.items():
            v = [check_map(runs[arm][i][1]).get(key, {}).get("measured")
                 for arm in arms if arm in runs for i in sorted(runs[arm])]
            v = [x for x in v if isinstance(x, (int, float))]
            vals[cname] = v
        if not all(vals.values()):
            continue
        rng = {k: (min(v), max(v), len(v)) for k, v in vals.items()}
        mean = {k: sum(v) / len(v) for k, v in vals.items()}
        spread = max(hi - lo for lo, hi, _ in rng.values())
        term, lead = mean["ecr"] - mean["ctl"], mean["dop"] - mean["ecr"]
        p(f"  {key:28s} ctl {mean['ctl']:.6g} [{rng['ctl'][0]:.6g},{rng['ctl'][1]:.6g}] n={rng['ctl'][2]}  "
          f"ecr {mean['ecr']:.6g} [{rng['ecr'][0]:.6g},{rng['ecr'][1]:.6g}] n={rng['ecr'][2]}  "
          f"dop {mean['dop']:.6g} [{rng['dop'][0]:.6g},{rng['dop'][1]:.6g}] n={rng['dop'][2]}")
        p(f"  {'':28s} term {term:+.6g}   lead {lead:+.6g}   max within-condition spread {spread:.6g}"
          f"   -> term {'>' if abs(term) > spread else '<='} spread, lead {'>' if abs(lead) > spread else '<='} spread")
    p()
    p("-- runtime (s)")
    for arm in [x for x in ARM_ORDER if x in runs]:
        for i in sorted(runs[arm]):
            d = runs[arm][i][1]
            rt = d.get("runtime_s", {})
            p(f"  {arm}_{i}  walk {rt.get('walk', float('nan')):.1f}  motion {rt.get('motion', float('nan')):.1f}  "
              f"total {d.get('total_runtime_s', float('nan')):.1f}")
    det_t = [runs[a2][i][1]["runtime_s"] for a2 in runs if a2.startswith("det_") for i in runs[a2]
             if "walk" in runs[a2][i][1].get("runtime_s", {})]
    nd_t = [runs[a2][i][1]["runtime_s"] for a2 in runs if a2.startswith("nd_") for i in runs[a2]
            if "walk" in runs[a2][i][1].get("runtime_s", {})]
    for name, ts in (("deterministic", det_t), ("non-deterministic", nd_t)):
        if ts:
            w = [t["walk"] for t in ts]; m = [t["motion"] for t in ts]
            p(f"  {name:18s} n={len(ts)}  walk {min(w):.1f}-{max(w):.1f} (mean {sum(w)/len(w):.1f})  "
              f"motion {min(m):.1f}-{max(m):.1f} (mean {sum(m)/len(m):.1f})")
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
