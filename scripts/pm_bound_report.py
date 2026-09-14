"""The walk.power_max bound study: tables and statistics from out/pm_bound/ (docs/audits/anti_runaway.md round 6,
handover item 4). CPU only; no simulation.

    python scripts/pm_bound_report.py                       # -> out/pm_bound/pm_bound_report.md, pm_bound_summary.json
    python scripts/pm_bound_report.py --root out/pm_bound

Inputs: out/pm_bound/<cfg>_s<seed>_d<draw>/<cfg>.json (scripts/pm_bound_batch.sh -> scripts/retire_measures.py, sections
walk,a,b), the shipped-default / LPi / drive-clip draws already on file (out/r5_adopt_default_*.json,
out/r5_skeptic_default_4.json, out/r5_attr_*.json, out/optic_audit/<cfg>/<cfg>.json, out/optic_verify/...), the four
take-off hold arms' pinned-walk values (out/r5_attr_{default,holdBrain,holdOptic,off}_*.json) and their room batches
(out/r5_adopt_sustain_live_{1,2,3}.json + out/sk_d1_shipped_4.json; out/d1_{holdBrain,holdOptic,off}_{1,2,3}.json +
out/sk_d1_{holdBrain,holdOptic,off}_4.json).

Questions answered (each a numbered table in the report):
  (a) the distribution of walk.power_max under the shipped defaults across independent GPU draws, against the 50 Hz bound
  (b) whether the check is monotone in the scanned measures (LPi -> LPLC2 x1 / x2 / x4; DN -> VNC x1 / x3; every path
      gain off; the +-35 mV optic drive clip)
  (c) its rank / linear correlation with the room take-off rate over the four hold arms (n = 4: descriptive only)
  (d) the numbers the referent question needs (the note on benchmark.py's REFERENCES line: 22 Hz at DN -> VNC x3, 50 at x6)
  plus the no_drive_clip retirement candidate against the baseline on every check of the three sections, 6 v 6 draws,
  with flyverse.interp.common.compare's verdict vocabulary (result / null / underpowered / undetermined).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from flyverse.interp import common  # noqa: E402

BOUND = 50.0
WALK_KEYS = ["walk.power_max_hz", "walk.power_sustained_hz", "walk.GF_max_hz"]
ALL_KEYS = ["walk.power_max_hz", "walk.power_sustained_hz", "walk.GF_max_hz", "loom.GF_peak_hz", "loom.escape_cm",
            "rotate.DNp20_flip_hz", "motion.min_dsi", "motion.correct_directions", "loom_escape.GF_peak_hz", "loom_escape.escapes"]
CONFIG_ORDER = ["baseline", "no_dn_vnc_gain", "no_path_gain", "no_drive_clip", "pair_gain_lpi_x1", "pair_gain_lpi_x2"]

# prior draws of the same sections on file (config -> [(label, path)])
PRIOR = {
    "baseline": [("r5_adopt_default_1", "out/r5_adopt_default_1.json"), ("r5_adopt_default_2", "out/r5_adopt_default_2.json"),
                 ("r5_adopt_default_3", "out/r5_adopt_default_3.json"), ("r5_skeptic_default_4", "out/r5_skeptic_default_4.json"),
                 ("r5_attr_default_1", "out/r5_attr_default_1.json"), ("optic_audit baseline", "out/optic_audit/baseline/baseline.json"),
                 ("optic_verify baseline", "out/optic_verify/baseline/baseline.json")],
    "pair_gain_lpi_x1": [("optic_audit", "out/optic_audit/pair_gain_lpi_x1/pair_gain_lpi_x1.json"),
                         ("optic_verify", "out/optic_verify/pair_gain_lpi_x1/pair_gain_lpi_x1.json")],
    "pair_gain_lpi_x2": [("optic_audit", "out/optic_audit/pair_gain_lpi_x2/pair_gain_lpi_x2.json")],
    "no_drive_clip": [("optic_audit", "out/optic_audit/no_drive_clip/no_drive_clip.json")],
}
# the four take-off hold arms: pinned-walk suite JSONs and room batches (docs/audits/receptor_integration.md G.4 / G.5)
HOLD_ARMS = {
    "shipped default": {"suite": ["out/r5_attr_default_1.json", "out/sk/sk_attr_default.json", "out/r5_adopt_default_1.json", "out/r5_adopt_default_2.json", "out/r5_adopt_default_3.json"],
                        "room": ["out/r5_adopt_sustain_live_1.json", "out/r5_adopt_sustain_live_2.json", "out/r5_adopt_sustain_live_3.json", "out/sk_d1_shipped_4.json"]},
    "holdBrain (optic side only)": {"suite": ["out/r5_attr_holdBrain_1.json", "out/r5_attr_holdBrain_2.json", "out/r5_attr_dup/r5_attr_holdBrain_1.json",
                                              "out/r5_attr_dup/r5_attr_holdBrain_2.json", "out/sk/sk_attr_holdBrain.json"],
                                    "room": ["out/d1_holdBrain_1.json", "out/d1_holdBrain_2.json", "out/d1_holdBrain_3.json", "out/sk_d1_holdBrain_4.json"]},
    "holdOptic (Brain side only)": {"suite": ["out/r5_attr_holdOptic_1.json", "out/r5_attr_holdOptic_2.json", "out/r5_attr_dup/r5_attr_holdOptic_1.json",
                                              "out/r5_attr_dup/r5_attr_holdOptic_2.json", "out/sk/sk_attr_holdOptic.json"],
                                    "room": ["out/d1_holdOptic_1.json", "out/d1_holdOptic_2.json", "out/d1_holdOptic_3.json", "out/sk_d1_holdOptic_4.json"]},
    "off": {"suite": ["out/r5_attr_off_1.json", "out/r5_attr_dup/r5_attr_off_1.json", "out/sk/sk_attr_off.json"],
            "room": ["out/d1_off_1.json", "out/d1_off_2.json", "out/d1_off_3.json", "out/sk_d1_off_4.json"]},
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def checks_of(run):
    return {c["key"]: (c["measured"], c["status"]) for c in run["checks"]}


def load_runs(root):
    runs = []
    for d in sorted(os.listdir(root)):
        m = re.match(r"^(?P<cfg>[a-z0-9_]+)_s(?P<seed>\d+)_d(?P<draw>\d+)$", d)
        p = os.path.join(root, d)
        if not m or not os.path.isdir(p):
            continue
        fn = os.path.join(p, m["cfg"] + ".json")
        if not os.path.exists(fn):
            runs.append({"cfg": m["cfg"], "seed": int(m["seed"]), "draw": int(m["draw"]), "dir": d, "missing": True}); continue
        run = load_json(fn)
        prov = run.get("provenance", {})
        runs.append({"cfg": m["cfg"], "seed": int(m["seed"]), "draw": int(m["draw"]), "dir": d, "file": fn.replace("\\", "/"),
                     "checks": checks_of(run), "device": run["config"].get("device"), "date": run.get("date"),
                     "runtime_min": round(run.get("total_runtime_s", float("nan")) / 60, 1),
                     "lif_receptor": (run["config"]["lif"].get("receptor_model"), run["config"]["lif"].get("receptor_net_rule")),
                     "path_gain": run["config"].get("path_gain"), "type_path_gain": run["config"].get("type_path_gain"),
                     "sum_absW": run["config"].get("connectome", {}).get("sum_absW"),
                     "commit": (prov.get("flyverse_commit") or {}).get("commit"),
                     "source_fingerprint": (prov.get("source_fingerprint") or {}).get("sha256") if isinstance(prov.get("source_fingerprint"), dict) else None,
                     "cache_md5": (prov.get("compiled_connectome") or {}).get("md5"),
                     "realised_device": (prov.get("execution") or {}).get("device"),
                     "optic_drive_clip": ((prov.get("model") or {}).get("optic") or {}).get("drive_clip_mv"),
                     "optic_pair_gain": ((prov.get("model") or {}).get("optic") or {}).get("pair_gain"),
                     "walk_top": run["sections"].get("walk", {}).get("walk", {}).get("top")})
    return runs


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def arm(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "bit_identical": bool(np.all(v == v[0])),
            "n_under_bound": int((v < BOUND).sum()), "values": [float(x) for x in v]}


def spearman_exact(x, y):
    """Spearman rho over n points with the exact two-sided permutation p (n! permutations of y)."""
    n = len(x)
    rx = np.argsort(np.argsort(x)) + 1; ry = np.argsort(np.argsort(y)) + 1
    def rho(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    r = rho(rx, ry)
    count = 0; total = 0
    for perm in itertools.permutations(ry):
        total += 1
        if abs(rho(rx, np.array(perm))) >= abs(r) - 1e-12:
            count += 1
    return r, count / total


def pearson_p(x, y):
    from scipy import stats
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def compare_row(stim, null):
    try:
        c = common.compare(list(stim), list(null))
        return {"verdict": c.get("verdict"), "diff": c.get("diff"), "z": c.get("z"), "p": c.get("p"), "p_floor": c.get("p_floor"),
                "null_sd_zero": c.get("null_sd_zero")}
    except Exception as e:      # noqa: BLE001
        return {"verdict": f"error: {e!r}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "out", "pm_bound"))
    args = ap.parse_args()
    root = args.root
    runs = load_runs(root)
    missing = [r["dir"] for r in runs if r.get("missing")]
    runs = [r for r in runs if not r.get("missing")]
    by_cfg = {c: [r for r in runs if r["cfg"] == c] for c in CONFIG_ORDER if any(r["cfg"] == c for r in runs)}
    lines = [f"# walk.power_max bound study ({root})", "",
             f"Generated by scripts/pm_bound_report.py on {time.strftime('%Y-%m-%d %H:%M')}; {len(runs)} runs "
             f"({', '.join(f'{c} x{len(v)}' for c, v in by_cfg.items())}); missing JSONs: {', '.join(missing) or 'none'}.", ""]
    summary = {"root": root, "n_runs": len(runs), "missing": missing, "bound": BOUND, "runs": runs}

    # ---- 0. provenance roll-call --------------------------------------------------------------------------------
    lines += ["## 0. Provenance of the runs", "", "| run | device (config) | realised device | commit | cache md5 | sum|W| | receptor | drive_clip | runtime min |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['dir']} | {r['device']} | {r['realised_device']} | {(r['commit'] or 'unknown')[:8]} | {(r['cache_md5'] or '?')[:8]} | "
                     f"{fmt(r['sum_absW'], 0)} | {r['lif_receptor'][0]} / {r['lif_receptor'][1]} | {fmt(r['optic_drive_clip'], 0)} | {r['runtime_min']} |")
    devs = sorted({str(r["device"]) for r in runs}); caches = sorted({str(r["cache_md5"]) for r in runs})
    lines += ["", f"Devices: {devs}; cache md5s: {caches}; receptor pairs: {sorted({str(r['lif_receptor']) for r in runs})}.", ""]
    summary["devices"] = devs; summary["cache_md5s"] = caches

    # ---- 1. the walk section per configuration x draw ------------------------------------------------------------
    lines += ["## 1. The pinned walk section, every draw (walk.* draws no RNG: differences between draws are GPU nondeterminism)", "",
              "| config | draw (seed, k) | walk.power_max (< 50) | walk.power_sustained (< 50) | walk.GF_max (< 38) | loom.GF_peak (>= 20) | rotate.DNp20_flip (< -2) | motion.min_dsi (>= 0.1) | loom_escape.GF_peak (>= 33) | escapes (>= 1) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    per_cfg = {}
    for cfg, rs in by_cfg.items():
        for r in sorted(rs, key=lambda r: (r["seed"], r["draw"])):
            ch = r["checks"]
            cells = []
            for k in ["walk.power_max_hz", "walk.power_sustained_hz", "walk.GF_max_hz", "loom.GF_peak_hz", "rotate.DNp20_flip_hz",
                      "motion.min_dsi", "loom_escape.GF_peak_hz", "loom_escape.escapes"]:
                v, s = ch.get(k, (None, None))
                cells.append(fmt(v) + ("" if s is None or s.startswith("PASS") else f" **{s}**"))
            lines.append(f"| {cfg} | s{r['seed']} d{r['draw']} | " + " | ".join(cells) + " |")
        per_cfg[cfg] = {k: arm([r["checks"].get(k, (None, None))[0] for r in rs]) for k in ALL_KEYS}
        per_cfg[cfg]["status"] = {k: sorted({r["checks"].get(k, (None, "n/a"))[1] for r in rs}) for k in ALL_KEYS}
        per_cfg[cfg]["tally"] = sorted({f"{sum(s.startswith('PASS') for _, s in r['checks'].values())}/{sum(s == 'FAIL' for _, s in r['checks'].values())}/{sum(s == 'KNOWN GAP' for _, s in r['checks'].values())}" for r in rs})
    summary["per_config"] = per_cfg
    lines.append("")

    # ---- (a) the baseline distribution -------------------------------------------------------------------------
    lines += ["## 2. (a) walk.power_max under the shipped defaults across draws", ""]
    base = per_cfg.get("baseline", {}).get("walk.power_max_hz", {"n": 0})
    prior = []
    for cfg, items in PRIOR.items():
        for label, path in items:
            p = os.path.join(ROOT, path)
            if os.path.exists(p):
                try:
                    d = load_json(p); ch = checks_of(d)
                    prior.append({"cfg": cfg, "label": label, "file": path, "walk.power_max_hz": ch.get("walk.power_max_hz", (None,))[0],
                                  "walk.power_sustained_hz": ch.get("walk.power_sustained_hz", (None,))[0], "walk.GF_max_hz": ch.get("walk.GF_max_hz", (None,))[0],
                                  "device": d.get("config", {}).get("device")})
                except Exception as e:  # noqa: BLE001
                    prior.append({"cfg": cfg, "label": label, "file": path, "error": repr(e)})
    summary["prior_draws"] = prior
    if base["n"]:
        lines += [f"This batch, baseline x{base['n']}: values {[round(v, 4) for v in base['values']]}; min {base['min']:.4f}, max {base['max']:.4f}, "
                  f"mean {base['mean']:.4f}, sd {base['sd']:.4f}; bit-identical across draws: {base['bit_identical']}; "
                  f"{base['n_under_bound']} of {base['n']} under the 50 Hz bound; margin to the bound {BOUND - base['max']:.4f} Hz (min over draws).", ""]
    pb = [x for x in prior if x["cfg"] == "baseline" and x.get("walk.power_max_hz") is not None]
    if pb:
        lines += ["Shipped-default draws already on file (same section, same code path):", "", "| file | walk.power_max | power_sustained | GF_max | device |", "|---|---|---|---|---|"]
        lines += [f"| {x['file']} | {fmt(x['walk.power_max_hz'])} | {fmt(x['walk.power_sustained_hz'])} | {fmt(x['walk.GF_max_hz'])} | {x['device']} |" for x in pb]
        allv = base.get("values", []) + [x["walk.power_max_hz"] for x in pb]
        lines += ["", f"Pooled shipped-default draws (this batch + on file): n {len(allv)}, min {min(allv):.4f}, max {max(allv):.4f}, "
                  f"{sum(v < BOUND for v in allv)} of {len(allv)} under 50.", ""]
        summary["baseline_pooled"] = arm(allv)

    # ---- (b) monotonicity -------------------------------------------------------------------------------------
    lines += ["## 3. (b) Is the bound monotone in any scanned measure?", "",
              "| scan | point | config | walk.power_max mean (min-max) | vs baseline: diff, verdict (compare, 6 v 6) | walk.GF_max mean | power_sustained mean |",
              "|---|---|---|---|---|---|---|"]
    scans = [("LPi34/43 -> LPLC2 factor", [("x1", "pair_gain_lpi_x1"), ("x2", "pair_gain_lpi_x2"), ("x4 (shipped)", "baseline")]),
             ("DN -> VNC path gain", [("x1 (no_dn_vnc_gain; VP -> DN x2 kept)", "no_dn_vnc_gain"), ("x3 (shipped)", "baseline")]),
             ("both path gains", [("none (no_path_gain)", "no_path_gain"), ("DN -> VNC x3, VP -> DN x2 (shipped)", "baseline")]),
             ("optic drive clip", [("1e9 mV (no_drive_clip)", "no_drive_clip"), ("+-35 mV (shipped)", "baseline")])]
    mono = {}
    for scan, points in scans:
        vals = []
        for label, cfg in points:
            a = per_cfg.get(cfg, {}).get("walk.power_max_hz", {"n": 0})
            if not a["n"]:
                lines.append(f"| {scan} | {label} | {cfg} | n/a | | | |"); continue
            cmp = compare_row(a["values"], per_cfg["baseline"]["walk.power_max_hz"]["values"]) if cfg != "baseline" and "baseline" in per_cfg else {"verdict": "-", "diff": 0.0}
            vals.append(a["mean"])
            gf = per_cfg[cfg]["walk.GF_max_hz"]; ps = per_cfg[cfg]["walk.power_sustained_hz"]
            lines.append(f"| {scan} | {label} | {cfg} | {a['mean']:.4f} ({a['min']:.4f}-{a['max']:.4f}) | {fmt(cmp.get('diff'))}, {cmp.get('verdict')}"
                         f"{' (null SD 0: bit-identical draws)' if cmp.get('null_sd_zero') else ''} | {gf['mean']:.4f} | {ps['mean']:.4f} |")
        d = np.diff(vals) if len(vals) > 1 else np.array([])
        mono[scan] = {"means": vals, "monotone": bool(d.size and (np.all(d >= 0) or np.all(d <= 0))), "n_points": len(vals)}
    summary["monotonicity"] = mono
    lines += ["", "Monotone (with only 2 points a scan is trivially monotone; the 3-point LPi scan is the one that can fail): " +
              "; ".join(f"{k}: {'yes' if v['monotone'] else 'NO'} over {v['n_points']} points {[round(x, 2) for x in v['means']]}" for k, v in mono.items()), ""]

    # ---- (c) the room correlation --------------------------------------------------------------------------------
    lines += ["## 4. (c) walk.power_max against the room take-off rate over the four hold arms", "",
              "| arm | walk.power_max (suite draws) | room hops per 1,000 fly-s (4 matched batches, 19,200 fly-s) | escape | voluntary | walking-GF median per batch | files |",
              "|---|---|---|---|---|---|---|"]
    hold = {}
    for name, spec in HOLD_ARMS.items():
        pm = []
        for p in spec["suite"]:
            fp = os.path.join(ROOT, p)
            if os.path.exists(fp):
                pm.append(checks_of(load_json(fp)).get("walk.power_max_hz", (None,))[0])
        tot = esc = vol = flys = 0; med = []; per = []
        for p in spec["room"]:
            fp = os.path.join(ROOT, p)
            if not os.path.exists(fp):
                continue
            d = load_json(fp); tot += d["hops_total"]; esc += d["hops_escape_total"]; vol += d["hops_voluntary_total"]; flys += d["fly_s"]
            med.append(round(d["gf_max_walk_median_hz"], 2)); per.append(d["hops_total"])
        rate = 1000 * tot / flys if flys else float("nan")
        hold[name] = {"power_max": [float(x) for x in pm if x is not None], "power_max_mean": float(np.mean(pm)) if pm else None,
                      "hops": tot, "escape": esc, "voluntary": vol, "fly_s": flys, "rate_all": rate, "rate_escape": 1000 * esc / flys if flys else None,
                      "rate_voluntary": 1000 * vol / flys if flys else None, "gf_median_per_batch": med, "hops_per_batch": per,
                      "suite_files": spec["suite"], "room_files": spec["room"]}
        lines.append(f"| {name} | {', '.join(fmt(x) for x in pm)} | {rate:.3f} ({tot} = {esc} + {vol}; per batch {per}) | {1000 * esc / flys:.3f} | {1000 * vol / flys:.3f} | {med} | {len(spec['suite'])} suite / {len(spec['room'])} room |")
    names = [n for n in hold if hold[n]["power_max_mean"] is not None and hold[n]["fly_s"]]
    x = np.array([hold[n]["power_max_mean"] for n in names]); y = np.array([hold[n]["rate_all"] for n in names])
    rho, p_rho = spearman_exact(x, y)
    try:
        pr, p_pr = pearson_p(x, y)
    except Exception:  # noqa: BLE001
        pr, p_pr = float("nan"), float("nan")
    ye = np.array([hold[n]["rate_escape"] for n in names]); yv = np.array([hold[n]["rate_voluntary"] for n in names])
    rho_e, p_e = spearman_exact(x, ye); rho_v, p_v = spearman_exact(x, yv)
    lines += ["", f"Over the {len(names)} arms: Spearman rho(walk.power_max, hops rate) = {rho:+.2f} (exact permutation p {p_rho:.3f}); "
              f"Pearson r = {pr:+.2f} (p {p_pr:.3f}); escape route rho {rho_e:+.2f} (p {p_e:.3f}); voluntary route rho {rho_v:+.2f} (p {p_v:.3f}). "
              f"With n = 4 arms the smallest attainable exact two-sided p for a rank correlation is {2 / math.factorial(4):.3f} (|rho| = 1), so this "
              f"table is descriptive: in common.compare's vocabulary it is `underpowered` for any verdict, and its reading is the ordering only.", ""]
    order_pm = sorted(names, key=lambda n: hold[n]["power_max_mean"]); order_room = sorted(names, key=lambda n: hold[n]["rate_all"])
    lines += [f"Ordering by walk.power_max (low -> high): {' < '.join(order_pm)}.", f"Ordering by room hops rate (low -> high): {' < '.join(order_room)}.", ""]
    summary["hold_arms"] = hold
    summary["room_correlation"] = {"arms": names, "spearman": rho, "spearman_p_exact": p_rho, "pearson": pr, "pearson_p": p_pr,
                                   "spearman_escape": rho_e, "spearman_voluntary": rho_v, "p_floor_n4": 2 / math.factorial(4)}

    # ---- (d) the referent ---------------------------------------------------------------------------------------
    lines += ["## 5. (d) The referent (benchmark.py REFERENCES: `Ref(22, \"<\", 50, \"4\", note=\"... 22 Hz with DN->VNC x3, 50 at x6\")`)", ""]
    b = per_cfg.get("baseline", {}).get("walk.power_max_hz", {"n": 0}); d1 = per_cfg.get("no_dn_vnc_gain", {}).get("walk.power_max_hz", {"n": 0})
    np_ = per_cfg.get("no_path_gain", {}).get("walk.power_max_hz", {"n": 0})
    if b["n"] and d1["n"]:
        lines += [f"* the shipped model at DN -> VNC x3 (the referent's own operating point) measures {b['mean']:.4f} Hz "
                  f"({b['min']:.4f}-{b['max']:.4f} over {b['n']} draws), not the 22 Hz the note records (session 4, before the receptor model, the "
                  f"LPi gain, the GF damping and its retirement);",
                  f"* at DN -> VNC x1 (`no_dn_vnc_gain`) it measures {d1['mean']:.4f} Hz ({d1['min']:.4f}-{d1['max']:.4f}); with every path gain off "
                  f"(`no_path_gain`) {np_['mean']:.4f} Hz ({np_['min']:.4f}-{np_['max']:.4f});",
                  f"* the x6 point (the note's 50 Hz) was not part of this batch's six configurations (no `dn_vnc_gain_x6` configuration exists in "
                  f"retire_measures.py); a re-derivation from x3 / x6 under the shipped gains needs that one arm.", ""]

    # ---- drive clip --------------------------------------------------------------------------------------------
    lines += ["## 6. The drive-clip retirement candidate: no_drive_clip vs baseline, every check of walk / a / b, 6 v 6 draws", "",
              "| check | criterion | baseline x6 (min-max; statuses) | no_drive_clip x6 (min-max; statuses) | diff of means | compare verdict | p | p_floor |",
              "|---|---|---|---|---|---|---|---|"]
    crit = {"walk.power_max_hz": "< 50", "walk.power_sustained_hz": "< 50", "walk.GF_max_hz": "< 38", "loom.GF_peak_hz": ">= 20", "loom.escape_cm": "notnone",
            "rotate.DNp20_flip_hz": "< -2", "motion.min_dsi": ">= 0.1", "motion.correct_directions": "== 8", "loom_escape.GF_peak_hz": ">= 33", "loom_escape.escapes": ">= 1"}
    dc = {}
    if "no_drive_clip" in per_cfg and "baseline" in per_cfg:
        for k in ALL_KEYS:
            a0 = per_cfg["baseline"][k]; a1 = per_cfg["no_drive_clip"][k]
            if not a0["n"] or not a1["n"]:
                continue
            cmp = compare_row(a1["values"], a0["values"])
            dc[k] = {"baseline": a0, "no_drive_clip": a1, "compare": cmp, "status_baseline": per_cfg["baseline"]["status"][k], "status_no_drive_clip": per_cfg["no_drive_clip"]["status"][k]}
            lines.append(f"| {k} | {crit[k]} | {a0['min']:.4f}-{a0['max']:.4f}; {'/'.join(per_cfg['baseline']['status'][k])} | "
                         f"{a1['min']:.4f}-{a1['max']:.4f}; {'/'.join(per_cfg['no_drive_clip']['status'][k])} | {a1['mean'] - a0['mean']:+.4f} | "
                         f"{cmp.get('verdict')}{' (null SD 0)' if cmp.get('null_sd_zero') else ''} | {fmt(cmp.get('p'))} | {fmt(cmp.get('p_floor'))} |")
        lines += ["", f"Tallies over the sections run (pass/fail/gap per draw): baseline {per_cfg['baseline']['tally']}, no_drive_clip {per_cfg['no_drive_clip']['tally']}.",
                  "Prior draw on file: " + "; ".join(f"{x['label']} {x['file']}: walk.power_max {fmt(x.get('walk.power_max_hz'))}, GF_max {fmt(x.get('walk.GF_max_hz'))}" for x in prior if x["cfg"] == "no_drive_clip"), ""]
    summary["drive_clip"] = dc

    # ---- every configuration vs baseline, every check (status view) -----------------------------------------------
    lines += ["## 7. Every configuration vs baseline, status over its 6 draws", "", "| check | " + " | ".join(by_cfg) + " |", "|---|" + "---|" * len(by_cfg)]
    for k in ALL_KEYS:
        row = []
        for cfg in by_cfg:
            a = per_cfg[cfg][k]
            row.append(f"{a['min']:.3f}-{a['max']:.3f} {'/'.join(per_cfg[cfg]['status'][k])}" if a["n"] else "n/a")
        lines.append(f"| {k} | " + " | ".join(row) + " |")
    lines.append("| pass/fail/gap | " + " | ".join(", ".join(per_cfg[c]["tally"]) for c in by_cfg) + " |")
    lines.append("")

    # local provenance for the summary (the shipped defaults on this machine's cache)
    try:
        from flyverse import brain, connectome, optic
        c = connectome.load(verbose=False)
        summary["provenance_local"] = common.provenance(c, lif=brain.LIFParams(), optic=optic.OpticParams(), device="cpu (report only)")
    except Exception as e:  # noqa: BLE001
        summary["provenance_local"] = {"error": repr(e)}
    summary["generator"] = "scripts/pm_bound_report.py"; summary["date"] = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(root, "pm_bound_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(root, "pm_bound_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\n".join(lines))
    print(f"\nwritten {os.path.join(root, 'pm_bound_report.md')} and pm_bound_summary.json")


if __name__ == "__main__":
    main()
