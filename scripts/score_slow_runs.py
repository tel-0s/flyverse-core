"""Score the round-3 slow-term-on-abs batch (out/r3_slow_abs_*.json) in the slow_term.md section-5b format.
Usage: python score_slow_abs.py [glob-prefix]   (default out/r3_slow_abs_)"""
import glob, json, os, sys, hashlib
import numpy as np
ROOT = r"D:\Projects\flyverse"
prefix = sys.argv[1] if len(sys.argv) > 1 else "out/r3_slow_abs_"
files = sorted(glob.glob(os.path.join(ROOT, prefix + "*.json")))
RUNAWAY = {"rest.spikes_per_step": "PASS", "walk_gf.p99_hz": "PASS", "loom_escape.GF_peak_hz": "PASS", "wind.DNp18_flip_hz": "PASS"}

def load(f):
    d = json.load(open(f, encoding="utf-8"))
    s = d["sections"]; ch = {c["key"]: c for c in d["checks"]}
    st = [c["status"] for c in d["checks"]]
    tally = f"{st.count('PASS')}/{st.count('FAIL')}/{st.count('KNOWN GAP')}"
    slow = d["config"]["receptor"].get("slow") or {}
    le = s.get("loom_escape", {}).get("seeds", {})
    per_seed = [round(v["GF_loom_peak_hz"], 1) for k, v in sorted(le.items())]
    escapes = sum(1 for v in le.values() if v.get("escape"))
    row = {
        "file": os.path.basename(f).replace(".json", ""),
        "mode": slow.get("mode") if slow.get("active") else "-",
        "gain": (slow.get("gain_by_class") or {}).get("monoamine", 0.0),
        "lead": (d["config"]["receptor"].get("dopamine_lead") or {}).get("lead", "all") if d["config"]["receptor"].get("dopamine_lead") else "all",
        "rest": s["rest"]["spikes_per_step"],
        "KC_hz": s["smell"]["KC_hz"], "KC_active": s["smell"]["KC_active"],
        "walk_gf_p99": s["walk_gf"]["p99_hz"], "walk_gf_med": s["walk_gf"]["median_hz"],
        "loom_escape_peak": ch["loom_escape.GF_peak_hz"]["measured"], "loom_per_seed": per_seed, "escapes": escapes,
        "wind_DNp18": s["wind"]["types"]["DNp18"]["flip_hz"],
        "taste_MN9": s["taste"]["MN9_hz"],
        "cal_sugar": s["bitter"]["calibrated_sugar_MN9_hz"], "cal_sugar_bitter": s["bitter"]["calibrated_sugar_bitter_MN9_hz"],
        "shiu_sugar": s["bitter"]["shiu_sugar_MN9_hz"], "shiu_sugar_bitter": s["bitter"]["shiu_sugar_bitter_MN9_hz"],
        "rotation": s["rotation"]["group_flip_hz"], "rotation_status": ch["rotation.group_flip_hz"]["status"],
        "walk_power_max": s["walk"]["walk"]["power_max_hz"], "loom_GF_peak_legacy": s["walk"]["loom"]["GF_peak_hz"],
        "tally": tally,
        "runaway4": "PASS" if all(ch[k]["status"] == v for k, v in RUNAWAY.items()) else "FAIL " + ",".join(k.split(".")[0] for k, v in RUNAWAY.items() if ch[k]["status"] != v),
        "fails": [c["key"] for c in d["checks"] if c["status"] == "FAIL"],
        "statuses": {c["key"]: c["status"] for c in d["checks"]},
        "measured": {c["key"]: c["measured"] for c in d["checks"]},
        "device": d["config"].get("device"), "backend": d["config"].get("backend"), "cache_dir": d["config"].get("cache_dir"),
        "nt_serotonin": (d["config"].get("nt_counts") or {}).get("serotonin"),
        "fast_changed": d["config"]["receptor"].get("fast_sign_changed_entries"),
        "net_rule": d["config"]["receptor"].get("net_rule"), "gain_classes": d["config"]["receptor"].get("gain_classes"),
        "slow_entries": (slow.get("entries_by_class") or {}).get("monoamine"),
        "table": d["config"]["receptor"].get("table"),
        "dop_info": d["config"]["receptor"].get("dopamine_lead"),
        "runtime_min": d["total_runtime_s"] / 60.0, "seeds": d["config"].get("seeds"),
    }
    return row

rows = [load(f) for f in files]
if not rows:
    sys.exit("no files")
hdr = ["run", "mode", "monoamine gain", "rest spk/step", "KC Hz (active)", "walk_gf p99 (median)", "loom GF peak (per seed)", "escapes",
       "wind DNp18", "taste MN9", "bitter cal sugar / +bitter", "Shiu sugar / +bitter", "rotation flip", "walk.power_max", "loom.GF_peak (a)", "pass/fail/gap", "runaway 4"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for r in rows:
    print(f"| {r['file']} | {r['mode']} | {r['gain']:g} | {r['rest']:.2f} | {r['KC_hz']:.2f} ({r['KC_active']}) | {r['walk_gf_p99']:.1f} ({r['walk_gf_med']:.0f}) | "
          f"{r['loom_escape_peak']:.1f} ({', '.join(f'{x:.1f}' for x in r['loom_per_seed'])}) | {r['escapes']} | {r['wind_DNp18']:.1f} | {r['taste_MN9']:.2f} | "
          f"{r['cal_sugar']:.2f} / {r['cal_sugar_bitter']:.2f} | {r['shiu_sugar']:.1f} / {r['shiu_sugar_bitter']:.2f} | {r['rotation']:.2f} {r['rotation_status']} | "
          f"{r['walk_power_max']:.1f} | {r['loom_GF_peak_legacy']:.1f} | {r['tally']} | {r['runaway4']} |")
print()
for r in rows:
    print(r["file"], "device", r["device"], r["backend"], "cache", r["cache_dir"], "5-HT", r["nt_serotonin"], "net_rule", r["net_rule"],
          "gain_classes", r["gain_classes"], "fast_changed", r["fast_changed"], "slow mono entries", r["slow_entries"], "lead", r["lead"],
          "runtime %.1f min" % r["runtime_min"], "seeds", r["seeds"], "FAILS", r["fails"])
# control scatter and per-check status differences vs the control
ctl = [r for r in rows if r["mode"] == "-"]
act = [r for r in rows if r["mode"] != "-"]
def scat(key, rs):
    v = [r[key] for r in rs]
    return f"{min(v):.2f}..{max(v):.2f} (n={len(v)}: {', '.join(f'{x:.2f}' for x in v)})"
if ctl:
    print("\nCONTROL scatter:")
    for k in ["walk_gf_p99", "loom_escape_peak", "rotation", "wind_DNp18", "KC_hz", "taste_MN9", "shiu_sugar", "shiu_sugar_bitter", "cal_sugar", "walk_power_max", "loom_GF_peak_legacy", "rest"]:
        print(" ", k, scat(k, ctl))
    # deterministic sections bit-identical across the controls?
    det = ["KC_hz", "taste_MN9", "shiu_sugar", "cal_sugar", "walk_power_max"]
    print("  deterministic keys identical across controls:", {k: len({round(r[k], 6) for r in ctl}) == 1 for k in det},
          {k: len({round(r["measured"][k], 6) for r in ctl}) == 1 for k in ["dn.MDN_top_hz", "odour.apple_channel_8cm_hz", "motion.min_dsi"]})
    ctl_status = {}
    for k in ctl[0]["statuses"]:
        ctl_status[k] = {r["statuses"][k] for r in ctl}
    print("  control checks not PASS in some replicate:", {k: v for k, v in ctl_status.items() if v != {"PASS"}})
    print("\nACTIVE runs: checks whose status differs from every control replicate:")
    for r in act:
        diffs = {k: (r["statuses"][k], sorted(ctl_status[k])) for k in r["statuses"] if r["statuses"][k] not in ctl_status[k]}
        print(" ", r["file"], diffs if diffs else "none")
# pairs (replicates) of the active settings
print("\nACTIVE settings, replicate values (walk_gf p99 | loom_escape | rotation | KC | Shiu sugar):")
import collections
by = collections.defaultdict(list)
for r in act:
    by[(r["mode"], r["gain"])].append(r)
for k, rs in sorted(by.items()):
    print(" ", k, [(f"{r['walk_gf_p99']:.1f}", f"{r['loom_escape_peak']:.1f}", f"{r['rotation']:.2f}", f"{r['KC_hz']:.2f}", f"{r['shiu_sugar']:.1f}", r["tally"], r["runaway4"]) for r in rs])
    det = ["KC_hz", "taste_MN9", "shiu_sugar", "cal_sugar", "walk_power_max"]
    print("    deterministic keys identical across the pair:", {d: len({round(r[d], 6) for r in rs}) == 1 for d in det})
# dop1r1 race check: identical slow entries / dopamine-lead info across every active run
print("\nDOP1R1 table race check (all active runs must show the same monoamine slow entries and dopamine-lead counts):")
print("  slow monoamine entries:", {json.dumps(r["slow_entries"]) for r in act})
print("  dopamine_lead info (rows, slow_net, lead):", {json.dumps({k: v for k, v in (r["dop_info"] or {}).items() if k in ("dopamine_rows", "slow_net", "slow_pos_lead")}, sort_keys=True) for r in act})
print("  fast_sign_changed_entries:", {r["fast_changed"] for r in rows})
