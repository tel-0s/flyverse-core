"""Suite-run comparisons for the receptor-integration rounds. CPU only (JSON / CSV / txt parsing).

    python scripts/compare_suite_runs.py            # round 3: the no-flag default runs vs round-2 off / abs and round-3 abs
                                                    #   -> out/r3_adopt_compare.log, out/r3_adopt_suite_table.md
    python scripts/compare_suite_runs.py --round4   # round 4: fixed-benchmark default x3 vs off x3 (+ r3 half-applied, r2 off,
                                                    #   holdKC / holdDN1) -> out/r4_compare.log, out/r4_suite_table.md
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd

ROOT = r"D:\Projects\flyverse"
os.chdir(ROOT)
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.append(s)


RANK = {"PASS": 2, "KNOWN GAP": 1, "MISSING": 0, "FAIL": 0}
STATUS_ABBR = {"PASS": "P", "FAIL": "F", "KNOWN GAP": "G", "MISSING": "M"}


def load_suite(path):
    d = json.load(open(path, encoding="utf-8"))
    checks = {c["key"]: c for c in d["checks"]}
    tally = {s: sum(1 for c in checks.values() if c["status"] == s) for s in ("PASS", "FAIL", "KNOWN GAP", "MISSING")}
    return d, checks, tally


def fmt_check(c):
    m = c["measured"]
    return f"-- {STATUS_ABBR[c['status']]}" if m is None else f"{m:.2f} {STATUS_ABBR[c['status']]}"


def round4():
    """Round-4 re-score with the fixed benchmark (walk / motion sections build the optic lobe with the receptor lookup;
    --receptor-model off sets receptor_model=None explicitly): default x3 vs off x3, with the round-3 half-applied
    defaults and the round-2 off runs beside them, and the Brain-side hold tables (holdKC / holdDN1).
    Writes out/r4_compare.log and out/r4_suite_table.md."""
    G = {
        "r4_default": sorted(glob.glob("out/r4_default_[123].json")),
        "r4_off": sorted(glob.glob("out/r4_off_[123].json")),
        "r3_default": sorted(glob.glob("out/r3_default_[123].json")),
        "r2_off": ["out/rm2_off.json", "out/rm2_off_r2.json", "out/rm2_off_r3.json", "out/skeptic2/rm_off_r4.json"],
        "holdKC": sorted(glob.glob("out/r4_holdKC_[12].json")),
        "holdDN1": sorted(glob.glob("out/r4_holdDN1_[12].json")),
    }
    S = {}
    for g, paths in G.items():
        S[g] = []
        for p in paths:
            if not os.path.exists(p):
                say(f"MISSING FILE {p}"); continue
            d, checks, tally = load_suite(p)
            S[g].append((p, d, checks, tally))
            cfg = d["config"]; rc = cfg.get("receptor", {})
            say(f"{p}: PASS {tally['PASS']} FAIL {tally['FAIL']} GAP {tally['KNOWN GAP']} MISSING {tally['MISSING']} | device {cfg.get('device')} "
                f"backend {cfg.get('backend')} seeds {cfg.get('seeds')} cache_dir {cfg.get('cache_dir')} | receptor model {rc.get('model')} "
                f"rule {rc.get('net_rule')} flag {rc.get('flag')} changed {rc.get('fast_sign_changed_entries')} table {os.path.basename(str(rc.get('table')))} | "
                f"nt glu {cfg.get('nt_counts', {}).get('glutamate')} | runtime {d.get('total_runtime_s', 0) / 60:.1f} min | {d.get('date')}")
    keys = []
    for g in ("r4_default", "r4_off", "r3_default", "r2_off", "holdKC", "holdDN1"):
        for _, _, checks, _ in S[g]:
            for k in checks:
                if k not in keys:
                    keys.append(k)
    # ---- per-check table + the adoption criterion (r4 default vs r4 off) ----------------------------------------
    cols_g = ("r4_default", "r4_off", "r3_default", "r2_off", "holdKC", "holdDN1")
    rows = ["| check | criterion | r4 default x3 (fixed benchmark) | r4 off x3 | r3 default x3 (half-applied) | r2 off x4 | holdKC x2 | holdDN1 x2 |",
            "|---|---|---|---|---|---|---|---|"]
    worse_best, worse_worst = [], []
    for k in keys:
        cols, crit = [], None
        for g in cols_g:
            vals = []
            for _, _, checks, _ in S[g]:
                if k in checks:
                    vals.append(fmt_check(checks[k])); crit = crit or checks[k]["criterion"]
                else:
                    vals.append("absent")
            cols.append(" / ".join(vals) if vals else "--")
        rows.append(f"| {k} | {crit} | " + " | ".join(cols) + " |")
        off_ranks = [RANK[checks[k]["status"]] for _, _, checks, _ in S["r4_off"] if k in checks]
        for p, _, checks, _ in S["r4_default"]:
            if k not in checks or not off_ranks:
                continue
            r = RANK[checks[k]["status"]]
            offs = [c[k]["status"] for _, _, c, _ in S["r4_off"] if k in c]
            if r < max(off_ranks):
                worse_best.append((k, os.path.basename(p), checks[k]["status"], offs))
            if r < min(off_ranks):
                worse_worst.append((k, os.path.basename(p), checks[k]["status"], offs))
    open("out/r4_suite_table.md", "w", encoding="utf-8").write(
        "# Round-4 re-score: benchmark suite with the fixed walk / motion sections and an explicit off\n\n" + "\n".join(rows) + "\n")
    say("\nCRITERION (r4 default x3 vs r4 off x3): status worse than the BEST off status (strict):", worse_best if worse_best else "none")
    say("CRITERION: status worse than the WORST off status (lenient):", worse_worst if worse_worst else "none")
    if not S["r4_default"] or not S["r4_off"]:
        say("CRITERION: cannot be applied -- no r4 default / off JSONs found (a vacuous 'none' is NOT a pass)")
    for g in ("r4_default", "r4_off"):
        say(f"  {g} tallies: " + "; ".join(f"{os.path.basename(p)} {t['PASS']}/{t['FAIL']}/{t['KNOWN GAP']}" for p, _, _, t in S[g]))
    # ---- off must reproduce round-2 off's deterministic values -------------------------------------------------------
    say("\nOFF REPRODUCTION (r4 off vs out/rm2_off.json; deterministic Brain-only checks must be bit-identical):")
    ref = S["r2_off"][0][2] if S["r2_off"] else {}
    # the checks that are bit-stable run to run at fixed weights (walk.power_sustained, loom.GF_peak, rotate.DNp20 and
    # motion.min_dsi scatter within the r2-off group itself, so they are compared as ranges elsewhere, not here)
    det = ["rest.spikes_per_step", "taste.MN9_hz", "smell.PN_hz", "smell.KC_active", "dn.DNa02_L_leg_asym_hz", "dn.MDN_top_hz", "dn.DNp09_top_hz",
           "walk.GF_max_hz", "walk.power_max_hz", "loom.escape_cm", "motion.correct_directions", "bitter.calibrated_sugar_MN9_hz",
           "bitter.calibrated_sugar_bitter_MN9_hz", "bitter.shiu_sugar_MN9_hz", "bitter.shiu_sugar_bitter_MN9_hz"]
    for p, _, checks, _ in S["r4_off"]:
        same = [k for k in det if k in checks and k in ref and checks[k]["measured"] == ref[k]["measured"]]
        diff = [(k, checks[k]["measured"], ref[k]["measured"]) for k in det if k in checks and k in ref and checks[k]["measured"] != ref[k]["measured"]]
        say(f"  {os.path.basename(p)}: identical {len(same)} of {len(det)}; differing: " +
            ("; ".join(f"{k} {a:.4f} vs {b:.4f}" for k, a, b in diff) if diff else "none"))
    # ---- per-group value ranges for the moved checks -------------------------------------------------------------------
    say("\nVALUES (min-max over replicates; * = identical in every replicate of the group):")
    for k in keys:
        line = f"  {k:38s}"
        for g in cols_g:
            v = [checks[k]["measured"] for _, _, checks, _ in S[g] if k in checks and checks[k]["measured"] is not None]
            if not v:
                line += f" | {g} --"
            elif len(set(v)) == 1:
                line += f" | {g} {v[0]:.4f}*"
            else:
                line += f" | {g} {min(v):.4f}-{max(v):.4f}"
        say(line)
    # ---- hold tables: which Brain-side group carries each deterministic change ---------------------------------------
    say("\nHOLD TABLES (Brain-only sections; value under default / holdKC / holdDN1 / off):")
    for k in ["taste.MN9_hz", "smell.PN_hz", "smell.KC_active", "walk.GF_max_hz", "walk.power_max_hz", "walk.power_sustained_hz", "loom.GF_peak_hz",
              "loom.escape_cm", "rotate.DNp20_flip_hz", "bitter.calibrated_sugar_MN9_hz", "bitter.calibrated_sugar_bitter_MN9_hz",
              "bitter.shiu_sugar_MN9_hz", "bitter.shiu_sugar_bitter_MN9_hz", "rest.spikes_per_step"]:
        parts = []
        for g in ("r4_default", "holdKC", "holdDN1", "r4_off"):
            v = [checks[k]["measured"] for _, _, checks, _ in S[g] if k in checks]
            parts.append(f"{g} " + (" / ".join(f"{x:.4f}" for x in v) if v else "--"))
        say(f"  {k:38s} " + " | ".join(parts))
    # ---- legacy walk section detail (top types, loom peak) for the default vs off -----------------------------------
    say("\nLEGACY walk section detail (sections.walk / loom):")
    for g in ("r4_default", "r4_off", "r3_default", "r2_off", "holdKC", "holdDN1"):
        for p, d, _, _ in S[g]:
            w = d["sections"].get("walk", {})
            wk, lm = w.get("walk", {}), w.get("loom", {})
            say(f"  {g:11s} {os.path.basename(p):22s} GF_mean {wk.get('GF_mean_hz', float('nan')):.3f} GF_max {wk.get('GF_max_hz', float('nan')):.2f} "
                f"power_mean {wk.get('power_mean_hz', float('nan')):.2f} power_max {wk.get('power_max_hz', float('nan')):.4f} "
                f"sustained {wk.get('power_sustained_hz', float('nan')):.2f} leg {wk.get('leg_hz', float('nan')):.2f} "
                f"loom_peak {lm.get('GF_peak_hz', float('nan')):.4f} escape_cm {lm.get('escape_cm')} top {wk.get('top')}")
    say("\nMOTION section (DSI per subtype):")
    for g in ("r4_default", "r4_off", "r3_default", "r2_off"):
        for p, d, _, _ in S[g]:
            m = d["sections"].get("motion", {}).get("subtypes", {})
            say(f"  {g:11s} {os.path.basename(p):22s} " + " ".join(f"{t} {v['dsi']:.4f}{'' if v.get('correct') else '!'}" for t, v in m.items()))
    say("\nloom_escape per seed (GF_loom_peak_hz, escape, t_after_loom_s):")
    for g in ("r4_default", "r4_off", "r3_default", "r2_off"):
        for p, d, _, _ in S[g]:
            le = d["sections"].get("loom_escape", {}).get("seeds", {})
            say(f"  {g:11s} {os.path.basename(p):22s} " + "  ".join(
                f"s{s}: {v['GF_loom_peak_hz']:.1f} {'E' if v['escape'] else '-'}" + (f"@{v['escape_info']['t_after_loom_s']:.2f}" if v.get('escape_info') else "")
                for s, v in le.items()))
    open("out/r4_compare.log", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
    print("\nwritten out/r4_compare.log, out/r4_suite_table.md")


if "--round4" in sys.argv:
    round4()
    sys.exit(0)


groups = {
    "default": sorted(glob.glob("out/r3_default_[123].json")),
    "default_flagged": ["out/r3_default_flagged.json"],
    "r3_abs": sorted(glob.glob("out/r3_abs_c[123].json")),
    "r2_abs": ["out/rm2_abs.json", "out/rm2_abs_r2.json", "out/rm2_abs_r3.json", "out/skeptic2/rm_abs_r4.json"],
    "off": ["out/rm2_off.json", "out/rm2_off_r2.json", "out/rm2_off_r3.json", "out/skeptic2/rm_off_r4.json"],
}
suites = {}
for g, paths in groups.items():
    suites[g] = []
    for p in paths:
        if not os.path.exists(p):
            say(f"MISSING FILE {p}"); continue
        d, checks, tally = load_suite(p)
        suites[g].append((p, d, checks, tally))
        cfg = d["config"]; rc = cfg.get("receptor", {})
        say(f"{p}: PASS {tally['PASS']} FAIL {tally['FAIL']} GAP {tally['KNOWN GAP']} MISSING {tally['MISSING']} | device {cfg.get('device')} "
            f"backend {cfg.get('backend')} seeds {cfg.get('seeds')} cache_dir {cfg.get('cache_dir')} | receptor cfg model {rc.get('model')} "
            f"rule {rc.get('net_rule')} changed {rc.get('fast_sign_changed_entries')} table {os.path.basename(str(rc.get('table')))} | "
            f"nt glu {cfg.get('nt_counts', {}).get('glutamate')} | runtime {d.get('total_runtime_s', 0) / 60:.1f} min | {d.get('date')}")

# ---- per-check table ------------------------------------------------------------------------------------------------
keys = []
for g in ("default", "r3_abs", "r2_abs", "off"):
    for _, _, checks, _ in suites[g]:
        for k in checks:
            if k not in keys:
                keys.append(k)


def fmt(c):
    m = c["measured"]
    if m is None:
        return f"-- {STATUS_ABBR[c['status']]}"
    return f"{m:.2f} {STATUS_ABBR[c['status']]}"


rows = ["| check | criterion | default x3 (no flags) | default flagged | r3 abs x3 | r2 abs x4 | off x4 |", "|---|---|---|---|---|---|---|"]
worse_any, worse_all, worse_value = [], [], []
for k in keys:
    cols = []
    crit = None
    for g in ("default", "default_flagged", "r3_abs", "r2_abs", "off"):
        vals = []
        for _, _, checks, _ in suites[g]:
            if k in checks:
                vals.append(fmt(checks[k])); crit = crit or checks[k]["criterion"]
            else:
                vals.append("absent")
        cols.append(" / ".join(vals))
    rows.append(f"| {k} | {crit} | " + " | ".join(cols) + " |")
    # status comparison: default replicate vs the off runs
    off_ranks = [RANK[checks[k]["status"]] for _, _, checks, _ in suites["off"] if k in checks]
    for p, _, checks, _ in suites["default"]:
        if k not in checks or not off_ranks:
            continue
        r = RANK[checks[k]["status"]]
        if r < max(off_ranks):
            worse_any.append((k, os.path.basename(p), checks[k]["status"], [c[k]["status"] for _, _, c, _ in suites["off"] if k in c]))
        if r < min(off_ranks):
            worse_all.append((k, os.path.basename(p), checks[k]["status"], [c[k]["status"] for _, _, c, _ in suites["off"] if k in c]))
open("out/r3_adopt_suite_table.md", "w", encoding="utf-8").write(
    "# Round-3 adoption: benchmark suite, default (no flags) vs round-3 abs, round-2 abs and off\n\n" + "\n".join(rows) + "\n")
say("\nSTATUS worse than the BEST off status (strict):", worse_any if worse_any else "none")
say("STATUS worse than the WORST off status (lenient):", worse_all if worse_all else "none")

# ---- deterministic Brain-only values: default vs r3_abs bit-identity -----------------------------------------------
say("\nBit-identity of measured values, default runs vs r3 abs c1 (same weights expected):")
if suites["default"] and suites["r3_abs"]:
    ref = suites["r3_abs"][0][2]
    for p, _, checks, _ in suites["default"] + suites["default_flagged"]:
        same = [k for k in keys if k in checks and k in ref and checks[k]["measured"] == ref[k]["measured"]]
        diff = [(k, checks[k]["measured"], ref[k]["measured"]) for k in keys if k in checks and k in ref and checks[k]["measured"] != ref[k]["measured"]]
        say(f"  {os.path.basename(p)}: identical {len(same)} checks; differing {len(diff)}: " +
            "; ".join(f"{k} {a:.4g} vs {b:.4g}" if a is not None and b is not None else f"{k} {a} vs {b}" for k, a, b in diff))
# loom_escape per seed
say("\nloom_escape per seed (GF_loom_peak_hz, escape, t_after_loom_s):")
for g in ("default", "default_flagged", "r3_abs", "r2_abs", "off"):
    for p, d, _, _ in suites[g]:
        le = d["sections"].get("loom_escape", {}).get("seeds", {})
        say(f"  {g:16s} {os.path.basename(p):24s} " + "  ".join(
            f"s{s}: {v['GF_loom_peak_hz']:.1f} {'E' if v['escape'] else '-'}" + (f"@{v['escape_info']['t_after_loom_s']:.2f}" if v.get('escape_info') else "")
            for s, v in le.items()))

# ---- figure-ground ----------------------------------------------------------------------------------------------------
say("\nFIGURE-GROUND (figure_z per type; r3 default s0 vs fg2 off s0/s1 and fg2 abs s0/s1):")
fg = {}
for name, path in [("default_s0", "out/r3_fg_default_s0.csv"), ("off_s0", "out/fg2_off_s0.csv"), ("off_s1", "out/fg2_off_s1.csv"),
                   ("abs_s0", "out/fg2_abs_s0.csv"), ("abs_s1", "out/fg2_abs_s1.csv")]:
    if os.path.exists(path):
        fg[name] = pd.read_csv(path).set_index("type")
    else:
        say(f"  MISSING {path}")
if fg:
    types = ["Mi4", "Mi1", "Tm5Y", "TmY21", "L1", "L2", "C2", "Tm9", "T1", "Dm9", "T4a", "T5a", "Mi9", "L3"]
    hdr = "| type | " + " | ".join(fg.keys()) + " |"
    say(hdr); say("|---" * (len(fg) + 1) + "|")
    for t in types:
        say(f"| {t} | " + " | ".join(f"{fg[k].figure_z.get(t, float('nan')):+.2f} (fig {fg[k].figure.get(t, float('nan')):+.4f})" for k in fg) + " |")
    # rank correlation of the whole z column with each reference
    if "default_s0" in fg:
        for k in fg:
            if k == "default_s0":
                continue
            j = fg["default_s0"].figure_z.to_frame("a").join(fg[k].figure_z.to_frame("b"), how="inner")
            say(f"  default_s0 vs {k}: {len(j)} types, spearman {j.a.corr(j.b, method='spearman'):.3f}, pearson {j.a.corr(j.b):.3f}, "
                f"max |dz| {float((j.a - j.b).abs().max()):.2f} at {(j.a - j.b).abs().idxmax()}")
# LC10a / LPLC2 lines from the txt (per-cell drive block)
for path in ["out/r3_fg_default_s0.txt", "out/fg2_off_s0.txt", "out/fg2_off_s1.txt", "out/fg2_abs_s0.txt", "out/fg2_abs_s1.txt"]:
    if os.path.exists(path):
        txt = open(path, encoding="utf-8", errors="replace").read()
        m = re.findall(r"^(LC10a|LC10b|LPLC2|LC11|LC4)\b.*$", txt, flags=re.M)
        dev = re.findall(r"device \w+", txt)
        say(f"  {path}: {dev[:1]} " + " || ".join(x.strip()[:110] for x in m[:5]))

# ---- object sweep -----------------------------------------------------------------------------------------------------
say("\nOBJECT SWEEP (verdict; LC11 / LC10a ball vs none):")
FIELDS = ["drive_mean_mv", "drive_best_cell_mean_mv", "drive_peak_mv", "drive_peak_100ms_mv", "rate_mean_hz", "rate_max_cell_hz", "cells_over_1hz"]
for name, path in [("default_s0", "out/r3_obj_default_s0.json"), ("default_s1", "out/r3_obj_default_s1.json"),
                   ("off_s0", "out/obj/obj_off_s0.json"), ("off_s1", "out/obj/obj_off_s1.json"),
                   ("abs_s0", "out/obj/obj_abs_s0.json"), ("abs_s1", "out/obj/obj_abs_s1.json")]:
    if not os.path.exists(path):
        say(f"  MISSING {path}"); continue
    d = json.load(open(path, encoding="utf-8"))
    v = d["verdict"]
    line = f"  {name:11s} mode {d['config'].get('mode')} rm {d['config'].get('receptor_model')} verdict {'PASS' if v['pass'] else 'FAIL'} " \
           f"(LC11 drive {v['LC11']['drive_pass']} rate {v['LC11']['rate_pass']}; LC10a drive {v['LC10a']['drive_pass']} rate {v['LC10a']['rate_pass']})"
    for t in ("LC11", "LC10a", "LPLC2"):
        b, n = d["ball"][t], d["none"][t]
        avail = [f for f in FIELDS if f in b]
        line += f"\n      {t}: " + ", ".join(f"{f.replace('drive_', '').replace('_mv', '').replace('_hz', 'Hz')} {b[f]:.2f}/{n[f]:.2f}" for f in avail)
    say(line)

# ---- sustain ----------------------------------------------------------------------------------------------------------
say("\nSUSTAIN 16 flies x 5 min, program cx, apple, fence (default vs off):")
sus = {}
for name, path in [("default", "out/r3_sustain_default.json"), ("off", "out/r3_sustain_off.json")]:
    if not os.path.exists(path):
        say(f"  MISSING {path}"); continue
    d = json.load(open(path, encoding="utf-8"))
    rows_ = pd.DataFrame(d["rows"])
    sus[name] = rows_
    say(f"  {name}: receptor {d.get('receptor')} frames {d['frames']} simulated {d['simulated_s']} s wall {d['wall_s']:.0f} s "
        f"({d['aggregate_fly_s_per_wall_s']:.2f} fly-s/wall-s)")
    for col in ("meals", "energy", "min_energy", "hops", "path_m", "distance_cm"):
        x = rows_[col].to_numpy(float)
        say(f"    {col:12s} mean {x.mean():.3f} sd {x.std(ddof=1):.3f} median {np.median(x):.3f} min {x.min():.3f} max {x.max():.3f} | " +
            " ".join(f"{v:.2f}" if col != "meals" and col != "hops" else f"{int(v)}" for v in x))
    modes = pd.DataFrame(list(rows_["modes"])).fillna(0)
    say("    mode fractions (mean over flies): " + ", ".join(f"{k} {v:.3f}" for k, v in modes.mean().sort_values(ascending=False).items()))
if len(sus) == 2:
    from scipy import stats
    for col in ("meals", "energy", "min_energy", "hops", "path_m", "distance_cm"):
        a, b = sus["default"][col].to_numpy(float), sus["off"][col].to_numpy(float)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        say(f"  {col:12s} default mean {a.mean():.3f} vs off {b.mean():.3f}; Mann-Whitney U {u.statistic:.0f} p {u.pvalue:.3f}")
    a, b = sus["default"]["meals"].to_numpy(int), sus["off"]["meals"].to_numpy(int)
    say(f"  flies with >= 1 meal: default {int((a >= 1).sum())}/16, off {int((b >= 1).sum())}/16; total meals {a.sum()} vs {b.sum()}")

# ---- pinned loom and bitter: byte comparison of the probe lines against the r3 abs runs ---------------------------------
say("\nPINNED LOOM / BITTER (default-flag runs vs the r3 abs runs of the rule task):")


def probe_lines(path):
    if not os.path.exists(path):
        return None
    L = [l.rstrip() for l in open(path, encoding="utf-8", errors="replace")]
    return [l for l in L if l and not l.startswith(("cuda ok", "/mnt", "  return", "pygame", "Hello", "ready in", "seed ")) and "Warning" not in l]


for new, old in [("out/r3_loom_default_s0.txt", "out/r3_loom_abs_s0.txt"), ("out/r3_bitter_default_s0.txt", "out/r3_bitter_abs_s0.txt"),
                 ("out/r3_bitter_default_s1.txt", "out/r3_bitter_abs_s1.txt"), ("out/r3_bitter_default_s2.txt", "out/r3_bitter_abs_s2.txt")]:
    a, b = probe_lines(new), probe_lines(old)
    if a is None or b is None:
        say(f"  MISSING {new if a is None else old}"); continue
    same = a == b
    say(f"  {new} vs {old}: {'IDENTICAL probe lines' if same else 'DIFFER'} ({len(a)} vs {len(b)} lines)")
    if not same:
        import difflib
        for l in list(difflib.unified_diff(b, a, lineterm="", n=0))[:30]:
            say("    " + l[:160])
    key = [l for l in a if re.search(r"DNp01|escape|GF|MN9|sugar|bitter|receptor model|device", l)]
    for l in key[:14]:
        say("    " + l[:170])

open("out/r3_adopt_compare.log", "w", encoding="utf-8").write("\n".join(OUT) + "\n")
print("\nwritten out/r3_adopt_compare.log, out/r3_adopt_suite_table.md")
