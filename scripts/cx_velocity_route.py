"""Compass round 7, the PS196_b velocity route (docs/audits/compass_velocity_route.md): the batch plan and the analysis.

    PYTHONIOENCODING=utf-8 python scripts/cx_velocity_route.py --plan-batch out/cx8 --seeds 0-5 --minutes 30
        -> out/cx8/batch.sh (a DRAFT: nothing here submits; `set -o pipefail`, one cluster_run.py call per <= 24 jobs,
           `--arm-block fam`, the docs/INTERP.md 10.3 job line: mkdir, venv, cuda assert, `st=\\$?; tail; exit \\$st`,
           `--fetch out/cx8/`) and out/cx8/arms.json (the arm table the analysis checks every run against).
    PYTHONIOENCODING=utf-8 python scripts/cx_velocity_route.py --analyse --runs out/cx8 --out out/cx8/analysis
        -> runs.csv (one row per run with its checks), per_seed.csv (arm, key, seeds, values: the per-seed lists every
           quoted number is pasted from, docs/INTERP.md 10.4 rule 28), compare.csv (the six predeclared measures,
           `common.compare` 6 v 6 exact U with Holm over the family of m = 6), descriptive.csv (the per-arm MEAN pivot
           of every key -- bump survival / rate / width, the k sweep, the L-R of every group -- which is the table
           analysis.md renders; the per-seed values with their sd and n are in per_seed.csv) and analysis.md.
    PYTHONIOENCODING=utf-8 python scripts/cx_velocity_route.py --analyse --runs out/cx6 --out out/cx6/analysis_cx8_pathcheck \\
        --label "PATH CHECK on cx6 runs: NOT cx8" --alias HG=H3 --alias HGV=H3G
        -> the same files on another batch's runs, every file headed by the label; `--alias NEW=OLD` maps that batch's
           arm labels onto round 7's so the code path runs. A path check carries no result of round 7.

Arms (6 seeds each; the cx_wedge free-turn protocol of 6A -- full connectome, no world, compass adaptation 0, 10 Hz
Poisson background on the 46 EPG, 1 s settle, wedges 0-3 at +40 Hz for 2 s, 5 s free -- plus a PRESCRIBED turn of
+90 deg/s (a left turn) over 0.5-3.5 s after the pulse end, fed to the afferent instrument when one is attached and
recorded either way; cx_wedge has no body, so the "fly turning itself" of the efferent protocol is a protocol
parameter here, the analogue of the 90 deg/s imposed visual rotation of deficit_rotation.md):

| arm | preset | hold | GLNO | instrument | role |
| S | raw | -- | sign 0 | -- | reference |
| V | instrumented | -- | sign 0 | sided_turn_afferent k 0.5 | the afferent alone |
| HG | instrumented | ring_dc_hold | glutamate (glno_sign) | -- | 6A's H3G |
| HGV | instrumented | ring_dc_hold | glutamate | sided_turn_afferent k 0.5 | THE ARM |
| HGV- | instrumented | ring_dc_hold | glutamate | sided_turn_afferent k 0.5 sign -1 | the sign control |
| HGVp | instrumented | ring_dc_hold_pen (`^(ExR6|ER6|ER4m)$:^PEN_`: EPG keeps its ring input) | glutamate | sided_turn_afferent k 0.5 | HGV with the PEN-side hold only (6B: a separate test of retaining EPG DC input; no causal inference from off-block counts) |
| HGVk025 / HGVk1 | as HGV at k 0.25 / 1.0 | descriptive (the k sweep on HGV only) |

Predeclared family (Holm, m = 6; 6 v 6 exact-U floor 0.0022 x 6 = 0.013, satisfiable): (1) bump_follow_wedges_per_s
HGV vs HG; (2) bump_follow_wedges_per_s HGV vs HGV-; (3) GLNO_LR_hz V vs S; (4) PEN_LR_hz HGV vs HG; (5) DNa02_LR_hz
HGV vs HG; (6) frac_confined_post HGVp vs HGV. `bump_follow_wedges_per_s` is the slope of
the unwrapped bump centre over the turn window times the sign of the turn (ideal 4.0 w/s at 90 deg/s); it is read
beside `bump_follow_confined_frac`, because on a dead bump the centre is noise -- the analysis reports both and never
calls a follow on a bump that was confined in fewer than half the turn-window frames (the gate is named in the row).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flyverse.interp import common  # noqa: E402

HOLD = "^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)"          # 6A's hold, on PEN and EPG: `ring_dc_hold`
HOLD_PEN = "^(ExR6|ER6|ER4m)$:^PEN_"              # the PEN-side hold only (EPG keeps its ExR6 / ER6 / ER4m input): `ring_dc_hold_pen`
TURN = ("90", "0.5:3.5")
# label, glutamate, hold spec (None = none), instrument spec, role
ARMS = [("S", False, None, None, "raw reference"),
        ("V", False, None, "sided_turn_afferent:k=0.5", "the afferent alone"),
        ("HG", True, HOLD, None, "ring_dc_hold + glno_sign (6A's H3G)"),
        ("HGV", True, HOLD, "sided_turn_afferent:k=0.5", "the arm"),
        ("HGV-", True, HOLD, "sided_turn_afferent:k=0.5:sign=-1", "the sign control"),
        ("HGVp", True, HOLD_PEN, "sided_turn_afferent:k=0.5",
         "HGV with the hold on the PEN side only (6B: test the EPG hold separately; the earlier rest-of-ring inhibition inference was withdrawn)"),
        ("HGVk025", True, HOLD, "sided_turn_afferent:k=0.25", "descriptive: k 0.25 on HGV"),
        ("HGVk1", True, HOLD, "sided_turn_afferent:k=1.0", "descriptive: k 1.0 on HGV")]
PRIMARY = ["S", "V", "HG", "HGV", "HGV-", "HGVp"]
FAMILY = [("1_bump_follow_HGV_vs_HG", "bump_follow_wedges_per_s", "HGV", "HG"),
          ("2_bump_follow_HGV_vs_HGV-", "bump_follow_wedges_per_s", "HGV", "HGV-"),
          ("3_GLNO_LR_V_vs_S", "GLNO_LR_hz", "V", "S"),
          ("4_PEN_LR_HGV_vs_HG", "PEN_LR_hz", "HGV", "HG"),
          ("5_DNa02_LR_HGV_vs_HG", "DNa02_LR_hz", "HGV", "HG"),
          ("6_frac_confined_post_HGVp_vs_HGV", "frac_confined_post", "HGVp", "HGV")]
HOLD_NAMES = {HOLD: "ring_dc_hold", HOLD_PEN: "ring_dc_hold_pen"}
KEYS = ["survival_s", "bump_hz_post", "width_half_post", "frac_confined_post", "bump_follow_wedges_per_s",
        "bump_follow_confined_frac", "bump_follow_ideal_wedges_per_s", "GLNO_LR_hz", "PEN_LR_hz", "DNa02_LR_hz",
        "PS196b_LR_hz", "AFF_LR_hz", "GLNO_LR_hz_rest", "PEN_LR_hz_rest", "PEN_mean_post", "GLNO_mean_post",
        "PS196b_L_hz_turn", "PS196b_R_hz_turn", "epg_in_mean_post", "epg_out_mean_post", "vs_post_all"]
MAX_JOBS_PER_CALL = 24
FOLLOW_GATE = 0.5              # a follow is read only when the bump was confined in >= this fraction of the turn-window frames


def sh_token(s: str) -> str:
    return "'" + s + "'" if any(ch in s for ch in "()|$^") else s


def arm_command(label, glu, hold, spec, seed, rel):
    """The cx_wedge.py invocation of one arm (the flags of 6A's job lines, plus round 7's)."""
    stem = f"{rel}/{label}_s{seed}"
    parts = ["python scripts/cx_wedge.py --no-structure --sim 1:1 --ledger", f"--seed {seed} --arm {label} --block fam_s{seed}",
             "--receptor-model shipped", f"--turn {TURN[0]} --turn-window {TURN[1]}"]
    if glu:
        parts.append("--nt-override GLNO=glutamate")
    if hold:
        parts.append(f"--hold-edges {sh_token(hold)}")
    if glu or hold or spec:
        parts.append("--preset instrumented")
    if spec:
        parts.append(f"--instrument {spec}")
    parts.append(f"--sim-out {stem}.json > {stem}.txt 2>&1")
    return " ".join(parts), stem


def job_line(cmd, stem, rel):
    """docs/INTERP.md 10.3: mkdir, venv, cuda assert, the command, `st=\\$?; tail; exit \\$st` (\\$ escaped so THIS
    shell leaves it for the job's shell)."""
    return (f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"{cmd}; st=\\$?; tail -3 {stem}.txt; exit \\$st")


def plan_batch(out_dir: Path, seeds, minutes=30, name="cx8"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rel = out_dir.as_posix() if not out_dir.is_absolute() else f"out/{out_dir.name}"
    if (out_dir / "predeclared.json").exists() or (out_dir / "submitted_at.txt").exists():
        raise ValueError("batch is frozen; use a new directory for a new predeclaration")
    jobs = []
    for seed in seeds:
        for label, glu, hold, spec, _ in ARMS:
            cmd, stem = arm_command(label, glu, hold, spec, seed, rel)
            jobs.append(dict(seed=seed, arm=label, block=f"fam_s{seed}", line=job_line(cmd, stem, rel)))
    # <= 24 jobs per cluster_run.py call, splitting only between seeds so a block never straddles two calls
    calls, current = [], []
    for j in jobs:
        if current and (len(current) >= MAX_JOBS_PER_CALL or (j["seed"] != current[-1]["seed"] and len(current) + len(ARMS) > MAX_JOBS_PER_CALL)):
            calls.append(current); current = []
        current.append(j)
    if current:
        calls.append(current)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = ["#!/bin/bash", "set -o pipefail",
             f"# DRAFT -- NOT SUBMITTED. Compass round 7 / batch {name}: docs/audits/compass_velocity_route.md (predeclared 2026-09-15).",
             f"# {len(jobs)} jobs = {len(ARMS)} arms x {len(seeds)} seeds, one arm per job, blocks fam_s<seed> (--arm-block fam: every seed's",
             f"# arms on one target), {len(calls)} cluster_run.py call(s) of <= {MAX_JOBS_PER_CALL} jobs. Generated by scripts/cx_velocity_route.py --plan-batch on {stamp}.",
             "# Submission is the owner's call: run from the repo root on a machine that reaches the cluster; read '<n> job(s), 0 failed' per call,",
             f"# then `ls {rel}/*.json | wc -l` = {len(jobs)} and `grep -c 'device cuda' {rel}/*.txt`, then --analyse.",
             "# Arms: " + "; ".join(f"{a[0]} = {a[4]}" for a in ARMS)]
    for i, call in enumerate(calls):
        lines.append(f"python scripts/cluster_run.py --target house --name {name} --minutes {minutes} --arm-block fam "
                     + " ".join('"' + j["line"] + '"' for j in call) + f" --fetch {rel}/ 2>&1 | tee {rel}/client_stdout_{i}.txt || exit $?")
    (out_dir / "batch.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    arms = {a[0]: dict(glutamate=a[1], hold=a[2], instrument=a[3], role=a[4], **{k: v for k, v in expected(a[0]).items() if k in ("preset", "instruments")})
            for a in ARMS}
    (out_dir / "arms.json").write_text(json.dumps({"batch": name, "seeds": list(seeds), "turn_deg_s": float(TURN[0]), "turn_window_s": TURN[1],
                                                   "holds": HOLD_NAMES, "arms": arms, "family": FAMILY, "primary_arms": PRIMARY,
                                                   "follow_gate_confined_frac": FOLLOW_GATE, "generated_utc": stamp,
                                                   "status": "DRAFT, not submitted", "protocol": PROTOCOL}, indent=1), encoding="utf-8")
    print(f"{len(jobs)} jobs in {len(calls)} call(s) -> {out_dir / 'batch.sh'} (DRAFT, not submitted); {out_dir / 'arms.json'}")
    return jobs, calls


# Fixed protocol values are checked independently in each run header and stimulus record.
PROTOCOL = {"gE": 1.0, "gD": 1.0, "gR": 1.0, "delta7_pen": True, "background_hz": 10.0,
            "pulse_hz": 40.0, "pulse_s": 2.0, "seconds_after": 5.0, "width": 4, "start_wedge": 0,
            "settle_s": 1.0, "receptor_model": "sign", "receptor_net_rule": "abs"}
HOLD_COUNTS = {HOLD: (1149, 37256.0, 88), HOLD_PEN: (402, 7893.0, 42)}
CACHE_MD5 = {False: "ef23cc27bea13be7f6a96f3c04fd3737", True: "7a10d93ba2086f2c76bcdabdca79b4ec"}
KEYS = list(dict.fromkeys(KEYS + [f"{g}_{side}_hz_turn" for g in ("GLNO", "PEN", "DNa02", "PS196b", "AFF")
                                for side in ("L", "R")] + [f"{g}_mean_{w}" for g in ("ExR6", "ER6", "ER4m")
                                                           for w in ("pre", "during", "post")]))


def source_hashes():
    paths = sorted(set(ROOT.glob("flyverse/**/*.py")) | set(ROOT.glob("flyverse/data/*.csv")) |
                   {Path(__file__).resolve(), ROOT / "scripts/cx_wedge.py", ROOT / "scripts/probe_compass_room.py"})
    # Git/cluster copies use LF; record that normalization explicitly for Windows checkouts.
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in paths}


def resolved_lif_by_arm():
    """Resolve the declared shipped-gain protocol before submission, including the diagnostic adaptation hold."""
    from flyverse import brain
    from cx_wedge import COMPASS_RE, RING_RE
    result = {}
    for name, _, hold, _, _ in ARMS:
        gains = list(brain.DEFAULT_TYPE_PATH_GAIN) + [
            (r"^EPG$", r"^PEN_", 1.0), (r"^PEN_", r"^EPG$", 1.0),
            (r"^EPG$", r"^PEG$", 1.0), (r"^PEG$", r"^EPG$", 1.0),
            (r"^Delta7$", r"^(EPG$|PEN_)", 1.0), (RING_RE, r"^(EPG$|PEN_|PEG$)", 1.0)]
        if hold:
            gains.append((*hold.split(":", 1), 0.0))
        params = brain.LIFParams(adapt_by_type={COMPASS_RE: 0.0}, type_path_gain=gains,
                                 receptor_model=PROTOCOL["receptor_model"], receptor_net_rule=PROTOCOL["receptor_net_rule"])
        result[name] = common.to_jsonable(common.model_record(params)["lif"])
    return result


def predeclare(out_dir):
    """Freeze the reviewed plan before submission. An existing declaration is never overwritten."""
    out_dir = Path(out_dir)
    if (out_dir / "submitted_at.txt").exists() or list(out_dir.glob("*_s*.json")):
        raise ValueError("cannot predeclare after submission or results exist")
    plan = json.loads((out_dir / "arms.json").read_text(encoding="utf-8"))
    if plan["seeds"] != list(range(6)) or plan["family"] != [list(x) for x in FAMILY]:
        raise ValueError("cx8 requires the reviewed six seeds and six primary contrasts")
    record = dict(plan, status="PREDECLARED, not submitted", written_before_submission=True,
                  stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  batch_sha256=hashlib.sha256((out_dir / "batch.sh").read_bytes()).hexdigest(),
                  arms_sha256=hashlib.sha256((out_dir / "arms.json").read_bytes()).hexdigest(),
                  source_sha256_lf=source_hashes(), resolved_lif_by_arm=resolved_lif_by_arm(),
                  replicate_unit="runs", multiplicity="one Holm family, m=6; missing p values count toward m",
                  follow_eligibility="confined in >= 0.5 of turn-window frames; disclose excluded run ids and actual n; no data is undetermined",
                  outcome_rule="primaries 1 and 2 must be positive results, HGV mean > 0 and HGV- mean < 0, with eligible bumps; otherwise no demonstrated turn-following compass",
                  followup="at most one predeclared batch: alternate afferents if primary 3 is null; GLNO-PEN transfer if 3 is result and no demonstrated follow; insufficient data remain undetermined")
    with (out_dir / "predeclared.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"froze {out_dir / 'predeclared.json'} before submission")
    return record


def instrument_parameters(spec):
    values = dict(k_hz_per_deg_s=0.5, sign=1, cells="AN07B037", max_hz=250.0)
    for item in (spec or "").split(":")[1:]:
        key, value = item.split("=", 1)
        key = "k_hz_per_deg_s" if key == "k" else key
        values[key] = value if key == "cells" else float(value)
    return values


def check_run(r, exp, path):
    """Refuse mislabeled protocols before interpreting any number. No simulation or cache mutation."""
    bad = []
    prov, m = r.get("provenance", {}), r.get("metrics", {})
    def check(ok, reason):
        if not ok:
            bad.append(reason)
    check(r.get("preset") == prov.get("preset") == exp["preset"], "preset/provenance mismatch")
    records = prov.get("instruments", [])
    names = [d.get("name") for d in records if isinstance(d, dict)]
    check(r.get("instruments") == names == exp["instruments"], "instruments/provenance mismatch")
    check(r.get("instrument_specs", []) == ([exp["spec"]] if exp["spec"] else []), "instrument spec mismatch")
    byname = {d["name"]: d for d in records if isinstance(d, dict) and "name" in d}
    for d in records:
        check(isinstance(d, dict) and all(d.get(k) for k in ("name", "kind", "law", "gap", "removal", "audits")),
              "instrument description incomplete")
    if exp["spec"]:
        d = byname.get("sided_turn_afferent", {})
        check(d.get("kind") == "stop-gap" and d.get("law") == "unverified", "afferent classification mismatch")
        pars = d.get("parameters", {})
        check(all(pars.get(k) == v for k, v in instrument_parameters(exp["spec"]).items()), "afferent parameters mismatch")
        check(d.get("n_cells") == 6 and all(len(d.get("cells", {}).get(side, [])) == 3 for side in ("L", "R")),
              "afferent cell selection mismatch")
    hold = exp["hold"]
    hold_list = [[*hold.split(":", 1), 0.0]] if hold else []
    check(r.get("hold_edges", []) == hold_list, "hold spec/factor mismatch")
    resolved = r.get("hold_edges_resolved", [])
    if hold:
        n, syn, npost = HOLD_COUNTS[hold]
        h = resolved[0] if len(resolved) == 1 else {}
        check(h.get("n_entries") == n and h.get("synapses") == syn and h.get("n_post_cells") == npost
              and h.get("n_pre_cells") == 17 and h.get("factor") == 0.0, "hold resolved counts mismatch")
        d = byname.get(HOLD_NAMES[hold], {})
        check(d.get("resolved") == h and d.get("parameters") == dict(pre=hold_list[0][0], post=hold_list[0][1], factor=0.0),
              "hold provenance mismatch")
    else:
        check(resolved == [], "unexpected resolved hold")
    override = {"GLNO": "glutamate"} if exp["glutamate"] else {}
    check(r.get("nt_override", {}) == override, "GLNO relabel mismatch")
    check(r.get("glno_nt") == (["glutamate"] if exp["glutamate"] else ["unknown"]), "GLNO labels mismatch")
    if exp["glutamate"]:
        check(byname.get("glno_sign", {}).get("parameters") == {"type": "GLNO", "nt": "glutamate"}, "relabel provenance mismatch")
    check(prov.get("compiled_connectome", {}).get("md5") == CACHE_MD5[exp["glutamate"]], "compiled cache mismatch")
    stimulus = prov.get("stimulus", {}).get("params", {})
    for key, value in PROTOCOL.items():
        if key not in ("pulse_s", "seconds_after"):
            check(r.get(key) == value, f"protocol {key} mismatch")
        check(stimulus.get(key) == value, f"stimulus {key} mismatch")
    check(r.get("lif_overrides", {}) == {} and r.get("edge_gains", []) == [], "unexpected LIF override/edge gain")
    check(r.get("turn_deg_s") == 90.0 and r.get("turn_window_s") == [0.5, 3.5], "turn/window mismatch")
    check(m.get("turn_on_s") == 3.5 and m.get("turn_off_s") == 6.5 and m.get("turn_deg_s") == 90.0,
          "measured turn window mismatch")
    check(m.get("turn_fed") is bool(exp["spec"]), "turn feeding mismatch")
    check(m.get("frames") == 800 and m.get("frame_s") == 0.01 and m.get("bump_follow_n_frames") == 300,
          "recording frame count mismatch")
    check(str(r.get("device", "")).startswith("cuda"), "device is not CUDA")
    check(r.get("block") == f"fam_s{r.get('seed')}", "seed block mismatch")
    lif = prov.get("model", {}).get("lif", {})
    check(lif.get("same_type_gain") == 0.1 and lif.get("dt") == 0.5 and lif.get("receptor_model") == "sign",
          "resolved LIF defaults mismatch")
    check(lif.get("adapt_by_type") == {"^(EPG|PEN|PEG|Delta7)": 0.0}, "compass adaptation mismatch")
    if hold_list:
        check(hold_list[0] in lif.get("type_path_gain", []), "resolved LIF lacks hold")
    for key in ("GLNO_LR_hz", "PEN_LR_hz", "DNa02_LR_hz", "frac_confined_post", "bump_follow_confined_frac"):
        check(isinstance(m.get(key), (int, float)) and np.isfinite(m[key]), f"missing/nonfinite {key}")
    check(0 <= m.get("bump_follow_confined_frac", -1) <= 1, "invalid confinement fraction")
    ledger = r.get("ledger_npz")
    check(bool(ledger) and (path.parent / Path(str(ledger)).name).is_file(), "missing ledger NPZ")
    console = path.with_suffix(".txt")
    check(console.is_file() and "device cuda" in console.read_text(encoding="utf-8", errors="replace"), "missing CUDA console")
    return bad


# ---------------------------------------------------------------------------------------------- analysis
def expected(label):
    for a in ARMS:
        if a[0] == label:
            names = ([a[3].split(":")[0]] if a[3] else []) + ([HOLD_NAMES[a[2]]] if a[2] else []) + (["glno_sign"] if a[1] else [])
            return dict(preset="instrumented" if (a[1] or a[2] or a[3]) else "raw", instruments=names, glutamate=a[1], hold=a[2], spec=a[3])
    return None


def holm(pvals: dict, m: int | None = None) -> dict:
    """Holm step-down over a family of `m` tests (default: the finite p's); a missing p counts toward m but gets NaN."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None and np.isfinite(p)), key=lambda kv: kv[1])
    m = len(items) if m is None else int(m)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = float(min(1.0, running))
    for k in pvals:
        adj.setdefault(k, float("nan"))
    return adj


def load_runs(runs_dir: Path, alias: dict) -> tuple[list, list]:
    rows, problems, identities, fingerprints = [], [], set(), set()
    declaration = Path(runs_dir) / "predeclared.json"
    frozen = json.loads(declaration.read_text(encoding="utf-8")) if declaration.exists() else None
    for path in sorted(Path(runs_dir).glob("*_s*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            problems.append(f"{path.name}: unreadable ({e})"); continue
        for i, r in enumerate(data if isinstance(data, list) else [data]):
            if not isinstance(r, dict) or "metrics" not in r:
                problems.append(f"{path.name}: missing metrics"); continue
            arm = alias.get(r.get("arm"), r.get("arm"))
            m, prov = r["metrics"], r.get("provenance", {})
            seed = r.get("seed")
            run_id = f"{path.name}#{i}"
            row = dict(arm=arm, arm_in_file=r.get("arm"), seed=seed, file=path.name, run_id=run_id,
                       device=r.get("device"), preset=r.get("preset", prov.get("preset", "raw")),
                       instruments=",".join(r.get("instruments", [])), nt_override=json.dumps(r.get("nt_override", {})),
                       hold_entries=sum(h.get("n_entries", 0) for h in r.get("hold_edges_resolved", [])),
                       turn_deg_s=r.get("turn_deg_s"), turn_fed=m.get("turn_fed"),
                       md5=prov.get("compiled_connectome", {}).get("md5"), wall_s=r.get("wall_s"))
            for k in KEYS:
                v = m.get(k)
                row[k] = float(v) if isinstance(v, (int, float)) else float("nan")
            row["follow_gated"] = bool(np.isfinite(row["bump_follow_confined_frac"]) and row["bump_follow_confined_frac"] >= FOLLOW_GATE)
            bad = []
            exp = expected(arm)
            if not alias:
                if exp is None or seed not in range(6):
                    bad.append("unexpected arm/seed")
                else:
                    bad.extend(check_run(r, exp, path))
                    if frozen and prov.get("model", {}).get("lif") != frozen.get("resolved_lif_by_arm", {}).get(arm):
                        bad.append("resolved LIF differs from frozen protocol")
                if (arm, seed) in identities:
                    bad.append("duplicate arm/seed")
                identities.add((arm, seed))
                fp = prov.get("source_fingerprint", {}).get("files", {})
                if not fp:
                    bad.append("missing source fingerprint")
                if frozen:
                    loaded = prov.get("source_fingerprint", {}).get("files_loaded", {})
                    if not all(f in loaded for f in ("scripts/cx_wedge.py", "scripts/probe_compass_room.py")):
                        bad.append("missing loaded probe source hashes")
                    for file, h in loaded.items():
                        wanted_hash = frozen.get("source_sha256_lf", {}).get(file)
                        if wanted_hash and h != wanted_hash:
                            bad.append(f"loaded simulation source differs from predeclaration: {file}")
                fingerprints.add(json.dumps(fp, sort_keys=True))
            row["checks"] = "; ".join(bad)
            problems.extend(f"{run_id}: {item}" for item in bad)
            rows.append(row)
    if not alias:
        wanted = {(a[0], seed) for a in ARMS for seed in range(6)}
        if identities != wanted or len(rows) != len(wanted):
            problems.append(f"expected 48 distinct arm/seed runs; got {len(rows)} rows; missing {sorted(wanted-identities)}")
        if len(fingerprints) != 1:
            problems.append("source fingerprints differ across runs")
        if frozen:
            if frozen.get("family") != [list(x) for x in FAMILY] or frozen.get("follow_gate_confined_frac") != FOLLOW_GATE:
                problems.append("analysis disagrees with frozen family/eligibility gate")
            for fp in fingerprints:
                for file, h in json.loads(fp).items():
                    want = frozen.get("source_sha256_lf", {}).get(file)
                    if want and want != h:
                        problems.append(f"simulation source differs from predeclaration: {file}")
    return rows, problems


def analyse(runs_dir: Path, out_dir: Path, label: str | None, alias: dict):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows, problems = load_runs(Path(runs_dir), alias)
    head = label or f"compass round 7 / cx8 analysis of {Path(runs_dir).as_posix()}"
    if not rows:
        raise SystemExit(f"no *_s*.json runs with a metrics block under {runs_dir}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs.csv", index=False)
    # per-seed lists (rule 28): every key of every arm, seeds and values side by side
    per = []
    for arm, g in df.groupby("arm", sort=False):
        g = g.sort_values("seed")
        for k in KEYS:
            vals = g[k].to_numpy(float)
            per.append(dict(arm=arm, key=k, seeds=",".join(str(s) for s in g.seed), files=",".join(g.file), run_ids=",".join(g.run_id), values=",".join(f"{v:.4f}" for v in vals),
                            mean=float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
                            sd=float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan"),
                            n=int(np.isfinite(vals).sum())))
    per_df = pd.DataFrame(per); per_df.to_csv(out_dir / "per_seed.csv", index=False)
    desc = per_df.pivot_table(index="arm", columns="key", values="mean", aggfunc="first").reindex([a[0] for a in ARMS if a[0] in set(df.arm)])
    # descriptive.csv is the per-arm mean pivot the analysis.md table renders (one row per arm, one column per key).
    # The per-seed values, their SD and n stay in per_seed.csv; this file used to be a byte copy of it.
    desc.to_csv(out_dir / "descriptive.csv", index=True)
    # the predeclared family
    comp = []
    for name, key, a, b in FAMILY:
        sa, sb = df[df.arm == a], df[df.arm == b]
        if key == "bump_follow_wedges_per_s":                       # the gate: a follow is read on a bump that was there
            sa, sb = sa[sa.follow_gated], sb[sb.follow_gated]
        va, vb = sa[key].to_numpy(float), sb[key].to_numpy(float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        rec = dict(test=name, key=key, stim=a, null=b, n_stim=int(len(va)), n_null=int(len(vb)),
                   stim_values=json.dumps([round(float(x), 4) for x in va]), null_values=json.dumps([round(float(x), 4) for x in vb]),
                   stim_run_ids=sa.loc[np.isfinite(sa[key]), "run_id"].tolist(),
                   null_run_ids=sb.loc[np.isfinite(sb[key]), "run_id"].tolist())
        if len(va) and len(vb):
            c = common.compare(va, vb)
            rec.update(verdict=c["verdict"], diff=c["diff"], z=c["z"], U=c["U"], p=c["p"], p_floor=c["p_floor"], null_sd_zero=c["null_sd_zero"], n_min=c["n_min"])
        else:
            rec.update(verdict="undetermined", diff=float("nan"), z=float("nan"), U=float("nan"), p=float("nan"), p_floor=float("nan"), null_sd_zero=None, n_min=0)
        # This is the first-step bound, not a per-test impossibility claim: other smaller p values can
        # put a test later in Holm's ordering. common.compare supplies the actual replicate/p-floor guard.
        rec["first_step_holm_floor"] = float(len(FAMILY) * 2 / math.comb(len(va) + len(vb), len(va))) if len(va) and len(vb) else float("nan")
        comp.append(rec)
    adj = holm({r["test"]: r["p"] for r in comp}, m=len(FAMILY))
    for r in comp:
        r["p_holm"] = adj[r["test"]]; r["m"] = len(FAMILY)
        r["verdict_holm"] = ("result" if (r["verdict"] == "result" and np.isfinite(r["p_holm"]) and r["p_holm"] <= 0.05)
                             else ("null" if r["verdict"] == "result" else r["verdict"]))
        if problems:
            r["unchecked_verdict"] = r["verdict"]
            r["verdict"] = r["verdict_holm"] = "undetermined"
    comp_df = pd.DataFrame(comp); comp_df.to_csv(out_dir / "compare.csv", index=False)
    summary = {"label": head, "runs_dir": Path(runs_dir).as_posix(), "n_runs": int(len(df)), "arms": {a: int(n) for a, n in df.arm.value_counts().items()},
               "problems": problems, "alias": alias, "family": comp, "follow_gate_confined_frac": FOLLOW_GATE,
               "files": {k: str(out_dir / f"{k}.csv") for k in ("runs", "per_seed", "compare", "descriptive")},
               "valid_batch": not problems and not alias, "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    md = [f"# {head}", "",
          (f"**{label}** -- the files under `{out_dir.as_posix()}` are a code-path check on another batch's runs and carry no result of round 7." if label else
           f"Runs: `{Path(runs_dir).as_posix()}`, {len(df)} runs, arms {dict(df.arm.value_counts())}."),
          "", f"n_runs {len(df)}; problems {len(problems)}" + (": " + "; ".join(problems[:20]) if problems else ""), "",
          "## The predeclared family (Holm, m = 6; `compare.csv`)", "",
          "| test | key | stim | null | n | verdict | verdict (Holm) | diff | z | p | p_holm | p_floor |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp:
        md.append(f"| {r['test']} | {r['key']} | {r['stim']} | {r['null']} | {r['n_stim']} v {r['n_null']} | {r['verdict']} | {r['verdict_holm']} | "
                  f"{r['diff']:+.4f} | {r['z']:+.2f} | {r['p']:.4f} | {r['p_holm']:.4f} | {r['p_floor']:.4f} |")
    md += ["", f"`bump_follow_wedges_per_s` rows use only runs whose bump was confined in >= {FOLLOW_GATE:.0%} of the turn-window frames "
               "(`follow_gated` in runs.csv); the ungated values are in per_seed.csv.", "",
           "## Per arm (`descriptive.csv`: mean over seeds)", "", desc.round(4).to_markdown() if hasattr(desc, "to_markdown") else desc.round(4).to_string(), "",
           "## Per-seed lists (`per_seed.csv`, columns arm,key,seeds,values -- pasted, never retyped: INTERP 10.4 rule 28)", "",
           "| arm | key | seeds | values |", "|---|---|---|---|"]
    for r in per:
        if r["key"] in ("survival_s", "bump_follow_wedges_per_s", "bump_follow_confined_frac", "GLNO_LR_hz", "PEN_LR_hz", "DNa02_LR_hz", "PS196b_LR_hz", "AFF_LR_hz"):
            md.append(f"| {r['arm']} | {r['key']} | {r['seeds']} | {r['values']} |")
    (out_dir / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{head}\nn_runs {len(df)} arms {summary['arms']} problems {len(problems)} -> {out_dir}")
    common.print_table(comp_df[["test", "key", "stim", "null", "n_stim", "n_null", "verdict", "verdict_holm", "diff", "z", "p", "p_holm"]], floatfmt="{:+.4f}")
    return summary


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-", 1); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-batch", default=None, metavar="DIR", help="write DIR/batch.sh (DRAFT) and DIR/arms.json")
    ap.add_argument("--seeds", default="0-5"); ap.add_argument("--minutes", type=int, default=30); ap.add_argument("--name", default="cx8")
    ap.add_argument("--predeclare", default=None, metavar="DIR", help="freeze the reviewed plan before submission; never overwrite")
    ap.add_argument("--analyse", action="store_true"); ap.add_argument("--runs", default="out/cx8"); ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None, help="a heading for every output file (REQUIRED when the runs are not cx8's)")
    ap.add_argument("--alias", action="append", default=None, metavar="NEW=OLD", help="map another batch's arm label onto a round-7 label (path checks only)")
    a = ap.parse_args()
    if a.plan_batch:
        plan_batch(Path(a.plan_batch), parse_seeds(a.seeds), minutes=a.minutes, name=a.name)
    if a.predeclare:
        predeclare(a.predeclare)
    if a.analyse:
        alias = {}
        for it in a.alias or []:
            new, old = it.split("=", 1); alias[old] = new
        if alias and not a.label:
            raise SystemExit("--alias is for path checks on another batch: give --label saying so")
        result = analyse(Path(a.runs), Path(a.out or (Path(a.runs) / "analysis")), a.label, alias)
        if result["problems"] and not alias:
            raise SystemExit(2)
    if not (a.plan_batch or a.predeclare or a.analyse):
        ap.print_help()


if __name__ == "__main__":
    main()
