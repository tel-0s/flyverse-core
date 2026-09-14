"""Object round 3, task same-device: the ARM/BOX CONFOUND of object round 2, closed on one GPU model.

Round 2 (docs/audits/object_compare_r2.md section 8 item 1) ran `base` on a B200 and `rectify` / `suppress` on H200s
because one job per arm made the arm collinear with the box (docs/INTERP.md 10.4 item 9). Every rectify-vs-base and
suppress-vs-base row of the sphere ladder was therefore device-crossed. This batch re-runs the three arms -- `base`,
`rectify`, `suppress`, the `--optic` overrides copied VERBATIM from `object_round2_compare.ARMS` -- on the MATCHED sphere
ladder (scripts/probe_object_matched.py run: 4.5 / 8.8 / 11 / 20 / 30 deg, 5 object runs per rung + 5 blank/blank nulls
per arm, the round-2 protocol) in ONE submission whose every job carries the block token `fam_samedev`, so
`scripts/cluster_run.py --arm-block fam` puts every job -- every arm -- on ONE box. The round-2 analysis
(`object_round2_compare.analyse_sphere`, its predeclared families, Holm and window rule, unchanged) is then run on this
batch, every arm-vs-base row carrying `same_device_as_reference = True`, and its numbers are quoted BESIDE round 2's.

Jobs are blocked by SEED, not by arm: job `seed<s>` runs base, rectify and suppress in turn at every rung and the null,
so the three arms share the box AND the same load history; five jobs = five processes-at-a-time on a 5-slot box.

The per-body RF-windowed T3 / T2 medians (`abs_drive_median`, the companion every max-over-cells figure must travel
with) read the per-frame graded arrays `a__optic_dr` / `b__optic_dr` of each run npz (117 MB of a 190 MB npz). The
round-2 export's `slim` drops them outright, which would lose exactly that statistic, so this script's `slim` keeps the
COLUMNS of the four upstream types the analysis reads (T3, T2, Tm5Y, TmY21: 4,840 of 12,235) and drops the other five
types' columns (Tm3, Mi1, Mi4, TmY5a, TmY13 -- read by nothing in this analysis). The run JSON, the console, and every
other array are untouched; the npz records what was kept. `run-job` slims its own runs on the box after its check, and
copies the JSON + console into `meta/`, so the fetch pulls `meta/` (small) before `sph_slim/` (~45 MB per run).

    python scripts/object_round3_samedevice.py plan    [--out out/objr3sd --name objr3sd --minutes 240 --runs 5 --seconds 12 --settle 3]
    python scripts/object_round3_samedevice.py submit  [--out out/objr3sd]          # ONE cluster_run.py call, --arm-block fam, log out/objr3sd_cluster.log
    python scripts/object_round3_samedevice.py run-job --job seed0 --fam fam_samedev --out out/objr3sd ...    # on the box
    python scripts/object_round3_samedevice.py slim    --out out/objr3sd            # on the box, by hand, if a job died before its slim
    python scripts/object_round3_samedevice.py assemble [--out out/objr3sd]         # local: meta/ + sph_slim/ -> sph/ (the round-2 layout)
    python scripts/object_round3_samedevice.py verify   [--out out/objr3sd --log out/objr3sd_cluster.log --json out/objr3sd/verify.json]
    python scripts/object_round3_samedevice.py analyse  [--out out/objr3sd --json out/objr3sd/samedevice.json --round2 out/interp/objr2c/compare.json]

The reading rules (PREDECLARED) are stamped to out/objr3sd/predeclared.json by `plan`, before submission; `analyse`
copies them into the Result unchanged and records the sha256 of the analysis code it ran (docs/INTERP.md 10.4 item 11).
Nothing is adopted; every hook stays off by default (the project rule).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flyverse.interp import common                     # noqa: E402  (CPU-only)
import object_round2_compare as orc                    # noqa: E402  (the round-2 arms, analysis and verifier: one implementation)
import object_round2_baseline as orb                   # noqa: E402  (compare_row, tree_state)

# ==================================================================================================== the design
ARMS3 = ("base", "rectify", "suppress")
ARMS = {a: orc.ARMS[a] for a in ARMS3}                 # the round-2 overrides, verbatim (object_compare_r2.md section 1)
DIAMS, SMALL_RUNGS, LARGE_RUNGS = orc.DIAMS, orc.SMALL_RUNGS, orc.LARGE_RUNGS
N_RUNS = orc.N_RUNS
BLOCK_KEY, BLOCK = "fam", "fam_samedev"               # every job line carries `fam_samedev`; cluster_run --arm-block fam keeps them on one box
KEEP_TYPES = tuple(orc.UPSTREAM)                       # the graded columns the analysis reads: T3, T2, Tm5Y, TmY21
REFERENCE = orc.REFERENCE_ARM
ANALYSIS_CODE = ("scripts/object_round3_samedevice.py", "scripts/object_round2_compare.py", "scripts/object_round2_baseline.py",
                 "scripts/probe_object_matched.py", "flyverse/interp/common.py")
ROUND2_RESULT = "out/interp/objr2c/compare.json"
ROUND2_DEVICES = {"base": ["NVIDIA B200"], "rectify": ["NVIDIA H200"], "suppress": ["NVIDIA H200"]}   # object_compare_r2.md section 8 item 1

PREDECLARED = {
    "written": "2026-09-14, before the batch was submitted (out/objr3sd/predeclared.json is stamped by `plan`; the stamp, not this text, is the predeclaration)",
    "question": "do round 2's rectify-vs-base results and suppress's nulls on the matched sphere ladder reproduce when base, rectify and suppress run on ONE GPU model in ONE submission -- the arm/box confound of object_compare_r2.md section 8 item 1 closed; a diagnosis on fixed anatomy, nothing adopted, every hook off by default",
    "arms": {a: {"letter": ARMS[a]["letter"], "optic": ARMS[a]["optic"], "note": ARMS[a]["note"]} for a in ARMS3},
    "why_these_arms": "rectify is the one arm that carried the carrier figure in round 2 (T3 6.0 / 10.5 / 9.0x base at 4.5 / 8.8 / 11 deg) and suppress the one whose sphere rows were all null while it cost the escape benchmark; both ran on H200s against a base on a B200, so both sets of vs-base rows were device-crossed",
    "replicate_unit": "one process = one (A, B) pair under one brain seed = one run; a cluster job is a container of sequential processes; runs, not cells, not seeds",
    "n_runs": {"sphere_per_arm_per_rung": N_RUNS, "sphere_nulls_per_arm": N_RUNS, "rungs_deg": list(DIAMS)},
    "one_submission_one_box": "ONE cluster_run.py call; every job line carries the block token `fam_samedev` and the call passes --arm-block fam, so every job of every arm is placed on ONE target; jobs are blocked by SEED (job seed<s> runs base, rectify, suppress in turn at every rung + null), so the arms share the box and its load history",
    "device_criterion": "the batch answers the question only if every run of every arm records the same provenance.execution.device_name, object_round2_compare.verify_all raises no device problem, and every arm-vs-base comparison row carries same_device_as_reference = True; otherwise the batch is reported as not having closed the confound and no vs-base row is quoted without that caveat",
    "gpu_model_note": "the task named a B200; at plan time the two B200 boxes of round 2 (vast-a / vast-b) were gone (ssh refused, disabled in .cluster.json) and the two live boxes are both NVIDIA H200 -- the model that hosted rectify and suppress in round 2. Base therefore lands on the treatment arms' GPU model, which is what object_compare_r2.md section 8 item 1 asked for ('a re-run of base on the box that hosted rectify and suppress'). The realised device_name is read from every run, never assumed",
    "window_rule": orc.PREDECLARED["window_rule"],
    "primary": {
        "P1_carrier": orc.PREDECLARED["primary"]["P1_carrier"],
        "P2_lc11": orc.PREDECLARED["primary"]["P2_lc11"],
        "note": "the round-2 families and calls verbatim (object_round2_compare.PREDECLARED['primary']): per arm the three small rungs {4.5, 8.8, 11} = 3 members, Holm within, question (i) arm object runs vs the arm's own blank/blank runs, question (ii) arm object runs vs the base arm's object runs at the same rung, tie-aware exact permutation U, verdict from common.compare; base's vs-base rows are the identity and are not called",
    },
    "companion": "beside every P1 / T2 row the per-body RF-windowed median |drive| of the same type and rung (`abs_drive_median`, role upstream_windowed, the same two questions): the headline is a within-run maximum over 1,940 T3 / 1,630 T2 cells and never travels without it (docs/INTERP.md 10.2)",
    "secondary_exploratory": orc.PREDECLARED["secondary_exploratory"],
    "reproduction": {
        "rectify_vs_base": "REPRODUCES if, on this batch, rectify reads `result` with p_holm <= 0.05 and diff > 0 vs base AND `result` with p_holm <= 0.05 vs its own null at every small rung on T3 and on T2 (round 2: result x3 on both questions at both types, carries_small_field = true); PARTIALLY if at least one small rung on T3 satisfies both; otherwise NOT. The ratio to base per rung is quoted beside round 2's 6.01 / 10.49 / 8.99 (T3) and 3.75 / 3.27 / 3.13 (T2) as magnitudes -- two batches are two draws and are never tested row by row (docs/INTERP.md 10.4 item 2)",
        "suppress_nulls": "REPRODUCE if, on this batch, suppress reads `null` on both questions at every small rung on T3, T2 and LC11 drive_median (round 2: null x3 on both questions at all three); a `null` is an absence at 5 v 5 and is reported with its z and p, never as evidence of equality",
        "lc11_follows": "LC11 FOLLOWS in neither arm is REPRODUCED if the P2 family reads no rung with `result` on both questions for rectify and for suppress (round 2: false 8/8)",
        "companion_null": "the round-2 finding that rectification leaves the typical cell where it was is REPRODUCED if every abs_drive_median vs-base row of rectify at T3 and T2 (small rungs; the large rungs reported beside) reads `null` (round 2: null in all 48 arm x type x rung rows, ratios 0.73-1.27)",
        "levels": "the blank-arm operating point (T3 / T2 dev_mean_b, LC11 drive_mean_mv_b) per arm, a magnitude beside round 2's section 5 table",
    },
    "cross_batch_exploratory": "base (this batch, one H200) vs base (round 2, B200): per type x statistic x rung, the 5 object runs of each batch through compare_row, and the 5 nulls likewise; role cross_batch_base, EXPLORATORY, no call -- the two batches differ in more than the GPU model (submission time, concurrent load, the shipped tree), so a difference here is a box-or-batch effect and an agreement is one draw's worth of evidence that the reference arm is stable across models",
    "analysis_code": "the sha256 of every analysis source (this script, object_round2_compare.py, object_round2_baseline.py, probe_object_matched.py, flyverse/interp/common.py) is stamped here at plan time and again in the Result at analysis time; a difference is reported, and a change to the reducers after the Result is written obliges a re-run on the same fetched batch (docs/INTERP.md 10.4 item 11)",
    "classification": "each finding is classified under the project rule: a mechanism the physiology implies vs a hand-set number (the rectify modes and the suppression k / radius are hypotheses, not data); NOTHING IS ADOPTED, every hook stays off by default and is bit-identical off on the CPU",
}


# ==================================================================================================== helpers
def analysis_code_sha() -> dict:
    out = {}
    for f in ANALYSIS_CODE:
        p = ROOT / f
        out[f] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if p.exists() else None
    return out


def _job_line(out: str, job: str, a) -> str:
    return (f"mkdir -p {out}/sph {out}/sph_slim {out}/meta && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/object_round3_samedevice.py run-job --job {job} --{BLOCK_KEY} {BLOCK} --out {out} --runs {a.runs} "
            f"--seconds {a.seconds:g} --settle {a.settle:g} --seed-offset {getattr(a, 'seed_offset', 0)}; st=$?; echo \"run-job {job} exit $st\"; exit $st")


def build_jobs(out: str, a) -> list[dict]:
    """One job per seed: base, rectify, suppress in turn at every rung, then the three nulls (3 x 6 = 18 processes)."""
    jobs = []
    for s in range(a.runs):
        procs = [orc._sphere_cmd(out, arm, d, s + getattr(a, 'seed_offset', 0), a.seconds, a.settle) for d in list(DIAMS) + [None] for arm in ARMS3]
        jobs.append({"job": f"seed{s}", "kind": "sph", "block": BLOCK, "stems": [p[0] for p in procs], "commands": [p[1] for p in procs],
                     "expect": [p[0] + ".json" for p in procs], "line": _job_line(out, f"seed{s}", a)})
    return jobs


# ==================================================================================================== plan / submit
def cmd_plan(args) -> int:
    out = args.out.rstrip("/")
    Path(out).mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(out, args)
    n_proc = sum(len(j["stems"]) for j in jobs)
    log = f"out/{args.name}_cluster.log"
    fetch = [f"{out}/meta/", f"{out}/sph_slim/"]
    sh = ["#!/bin/sh", f"# generated by scripts/object_round3_samedevice.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs, {n_proc} processes, ONE block ({BLOCK})", "set -e",
          f"mkdir -p {out} out", f'if [ -f {log} ]; then mv {log} "out/{args.name}_cluster.$(date +%Y%m%dT%H%M%S).log"; fi',
          f"python scripts/cluster_run.py --name {args.name} --minutes {args.minutes} --arm-block {BLOCK_KEY} --no-balance-blocks \\"]
    sh += [f"  {shlex.quote(j['line'])} \\" for j in jobs]
    sh += [f"  --fetch {' '.join(fetch)} 2>&1 | tee {log}"]
    (Path(out) / "batch.sh").write_text("\n".join(sh) + "\n", encoding="utf-8")
    with open(Path(out) / "jobs.json", "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "minutes": args.minutes, "runs": args.runs, "seconds": args.seconds, "settle": args.settle, "out": out,
                   "diameters_deg": list(DIAMS), "arms": ARMS, "arm_block": BLOCK_KEY, "block": BLOCK, "keep_types": list(KEEP_TYPES),
                   "jobs": jobs, "n_processes": n_proc, "fetch": fetch, "log": log}, f, indent=1)
    with open(Path(out) / "predeclared.json", "w", encoding="utf-8") as f:
        json.dump({"predeclared": PREDECLARED, "arms": ARMS, "analysis_code_sha256": analysis_code_sha(),
                   "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generator": " ".join(sys.argv)}, f, indent=1)
    ts = orb.tree_state()
    ts["sha256_lf"]["scripts/object_round3_samedevice.py"] = analysis_code_sha()["scripts/object_round3_samedevice.py"]
    ts["sha256_lf"]["scripts/object_round2_compare.py"] = analysis_code_sha()["scripts/object_round2_compare.py"]
    with open(Path(out) / "tree_state.json", "w", encoding="utf-8") as f:
        json.dump(ts, f, indent=1)
    print(f"{len(jobs)} job(s), {n_proc} processes, one block {BLOCK} (--arm-block {BLOCK_KEY}) -> {out}/ (log {log}); batch.sh, jobs.json, predeclared.json, tree_state.json written")
    for j in jobs:
        print(f"  {j['job']:8s} {len(j['stems']):3d} processes: {Path(j['stems'][0]).name} .. {Path(j['stems'][-1]).name}")
    return 0


def cmd_submit(args) -> int:
    out = args.out.rstrip("/")
    with open(Path(out) / "jobs.json", encoding="utf-8") as f:
        plan = json.load(f)
    lines = [j["line"] for j in plan["jobs"]]
    # --no-balance-blocks: the ONE block resolves to the least-loaded box (round-robin would deal it to the first target whatever its queue)
    cmd = [sys.executable, "scripts/cluster_run.py", "--name", plan["name"], "--minutes", str(plan["minutes"]), "--arm-block", plan["arm_block"], "--no-balance-blocks", *lines, "--fetch", *plan["fetch"]]
    cmd[2:2] = [v for key in ("target", "node") if getattr(args, key, None) for v in ("--" + key, getattr(args, key))]
    log = Path(plan["log"])
    if log.exists():
        log.rename(log.with_name(log.stem + time.strftime(".%Y%m%dT%H%M%S") + ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"submitting {len(lines)} job(s) ({plan['n_processes']} processes) in ONE block through scripts/cluster_run.py --arm-block {plan['arm_block']}; log {log}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"# submitted {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by scripts/object_round3_samedevice.py submit: {' '.join(cmd[1:6])} ... --fetch {' '.join(plan['fetch'])}\n"); f.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=env)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line); f.flush()
        rc = p.wait()
    print(f"cluster_run exit {rc}; log {log}")
    return rc


# ==================================================================================================== on the box: run-job, slim
def slim_npz(src: Path, dst: Path, keep_types=KEEP_TYPES) -> dict:
    """Rewrite one run npz keeping every array, but `a__optic_dr` / `b__optic_dr` (and the aligned `rate_idx` /
    `rate_body_ids` / `rate_types`) restricted to the columns whose type is in keep_types; records what was kept."""
    z = np.load(src, allow_pickle=False)
    keep = {k: z[k] for k in z.files if not k.startswith(("a__optic_dr", "b__optic_dr", "rate_"))}
    rt = z["rate_types"].astype(str)
    m = np.isin(rt, list(keep_types))
    keep["rate_idx"] = z["rate_idx"][m]; keep["rate_body_ids"] = z["rate_body_ids"][m]; keep["rate_types"] = z["rate_types"][m]
    keep["a__optic_dr"] = z["a__optic_dr"][:, m]; keep["b__optic_dr"] = z["b__optic_dr"][:, m]
    keep["slim_kept_types"] = np.array(list(keep_types)); keep["slim_dropped_types"] = np.array(sorted(set(rt[~m].tolist())))
    keep["slim_n_columns_before"] = np.array(int(len(rt))); keep["slim_n_columns_after"] = np.array(int(m.sum()))
    z.close()
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **keep)
    return {"file": src.name, "bytes_in": src.stat().st_size, "bytes_out": dst.stat().st_size, "columns_before": int(len(rt)), "columns_after": int(m.sum())}


def slim_and_stage(out: str, stems: list[str]) -> list[dict]:
    """For each finished stem: npz -> sph_slim/, json + txt -> meta/ (the fetch pulls meta/ first)."""
    rep = []
    for stem in stems:
        p = Path(stem)
        src = p.with_suffix(".npz")
        if not src.exists() or not p.with_suffix(".json").exists():
            rep.append({"file": src.name, "skipped": "npz or json missing"}); continue
        dst = Path(out) / "sph_slim" / src.name
        if not dst.exists():
            r = slim_npz(src, dst)
            print(f"  slim {r['file']}: {r['bytes_in'] / 1e6:.1f} -> {r['bytes_out'] / 1e6:.1f} MB, columns {r['columns_before']} -> {r['columns_after']}", flush=True)
            rep.append(r)
        for ext in (".json", ".txt"):
            q = p.with_suffix(ext)
            if q.exists():
                Path(out, "meta").mkdir(parents=True, exist_ok=True)
                shutil.copy2(q, Path(out) / "meta" / q.name)
    return rep


def cmd_run_job(args) -> int:
    jobs = {j["job"]: j for j in build_jobs(args.out.rstrip("/"), args)}
    j = jobs[args.job]
    print(f"run-job {args.job} [block {args.fam}]: {len(j['commands'])} process(es); arms {ARMS3} in turn per rung", flush=True)
    t0 = time.time()
    for stem, cmd in zip(j["stems"], j["commands"]):
        t1 = time.time()
        rc = subprocess.call(cmd, shell=True, cwd=str(ROOT))
        txt = Path(stem + ".txt")
        tail = [l for l in txt.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()][-1:] if txt.exists() else ["(no console)"]
        print(f"  {Path(stem).name:28s} exit {rc} {time.time() - t1:6.0f} s  {tail[0][:160]}", flush=True)
    print(f"run-job {args.job}: {(time.time() - t0) / 60:.1f} min", flush=True)
    args.expect = j["expect"]
    rc = orc.cmd_check_job(args)
    slim_and_stage(args.out.rstrip("/"), j["stems"])
    return rc


def cmd_slim(args) -> int:
    out = args.out.rstrip("/")
    stems = sorted(p[:-5] for p in glob.glob(f"{out}/sph/*.json") if orc.parse_sphere_name(Path(p).stem))
    rep = slim_and_stage(out, stems)
    done = [r for r in rep if "bytes_in" in r]
    print(f"slim: {len(stems)} run(s), {len(done)} rewritten, {sum(r['bytes_in'] for r in done) / 1e9:.2f} -> {sum(r['bytes_out'] for r in done) / 1e9:.2f} GB; kept {list(KEEP_TYPES)}")
    return 0


# ==================================================================================================== local: assemble, verify
def cmd_assemble(args) -> int:
    """meta/*.json + *.txt and sph_slim/*.npz -> sph/ (the layout every round-2 reader expects)."""
    out = args.out.rstrip("/")
    dst = Path(out) / "sph"; dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub, pat in (("meta", "*.json"), ("meta", "*.txt"), ("sph_slim", "*.npz")):
        for p in sorted(Path(out, sub).glob(pat)):
            q = dst / p.name
            if not q.exists() or q.stat().st_size != p.stat().st_size:
                shutil.copy2(p, q); n += 1
    js = sorted(dst.glob("*.json")); npz = sorted(dst.glob("*.npz")); txt = sorted(dst.glob("*.txt"))
    print(f"assemble: {n} file(s) copied into {dst}; now {len(js)} json, {len(npz)} npz, {len(txt)} consoles")
    return 0


def verify_extra(out: str, rep: dict) -> dict:
    """Beyond object_round2_compare.verify_all: consoles counted against runs, one device across the batch, the slim
    record of every npz, and the analysis code against the stamp."""
    ex = {"n_json": 0, "n_npz": 0, "n_txt": 0, "devices": {}, "slim": {"n_slim": 0, "n_full": 0, "kept_types": None, "columns_after": None},
          "analysis_code_now": analysis_code_sha(), "analysis_code_stamped": None, "analysis_code_drift": [], "problems": []}
    files = sorted(p for p in glob.glob(f"{out}/sph/*.json") if orc.parse_sphere_name(Path(p).stem))
    ex["n_json"] = len(files); ex["n_npz"] = sum(Path(p[:-5] + ".npz").exists() for p in files); ex["n_txt"] = sum(Path(p[:-5] + ".txt").exists() for p in files)
    if ex["n_txt"] != ex["n_json"]:
        ex["problems"].append(f"{ex['n_json'] - ex['n_txt']} run(s) without a console")
    for p in files:
        d = json.load(open(p, encoding="utf-8"))
        dn = str(d["provenance"]["execution"].get("device_name")); ex["devices"][dn] = ex["devices"].get(dn, 0) + 1
        npz = Path(p[:-5] + ".npz")
        if npz.exists():
            z = np.load(npz, allow_pickle=False)
            if "slim_kept_types" in z.files:
                ex["slim"]["n_slim"] += 1; ex["slim"]["kept_types"] = z["slim_kept_types"].tolist(); ex["slim"]["columns_after"] = int(z["slim_n_columns_after"])
                if sorted(z["slim_kept_types"].tolist()) != sorted(KEEP_TYPES) or int(z["slim_n_columns_after"]) != int(len(z["rate_types"])):
                    ex["problems"].append(f"{npz.name}: slim record inconsistent")
            else:
                ex["slim"]["n_full"] += 1
            z.close()
    if len(ex["devices"]) > 1:
        ex["problems"].append(f"more than one realised device across the batch: {ex['devices']}")
    st = Path(out) / "predeclared.json"
    if st.exists():
        s = json.load(open(st, encoding="utf-8"))
        ex["analysis_code_stamped"] = s.get("analysis_code_sha256")
        ex["stamped_utc"] = s.get("stamped_utc")
        for f, h in (ex["analysis_code_stamped"] or {}).items():
            if ex["analysis_code_now"].get(f) != h:
                ex["analysis_code_drift"].append(f)
        if json.dumps(common.to_jsonable(s.get("predeclared")), sort_keys=True) != json.dumps(common.to_jsonable(PREDECLARED), sort_keys=True):
            ex["problems"].append("the live PREDECLARED dict differs from the stamped predeclared.json")
    else:
        ex["problems"].append("no predeclared.json stamp")
    return ex


def verify_all(out: str, log: str | None) -> dict:
    rep = orc.verify_all(out, log)
    rep["extra"] = verify_extra(out, rep)
    rep["problems"] += rep["extra"]["problems"]
    if rep["extra"]["analysis_code_drift"]:
        rep["analysis_code_drift_note"] = (f"analysis code edited after the stamp: {rep['extra']['analysis_code_drift']} (the reading rules are the stamp; "
                                           "the Result records the code it ran)")
    return rep


def cmd_verify(args) -> int:
    out = args.out.rstrip("/")
    rep = verify_all(out, args.log)
    print(f"cluster log: {rep['log']}")
    if rep["sphere"]:
        print(f"sphere: {rep['sphere']['n_runs']} runs on {rep['sphere']['devices']}; centroid-elevation spread {rep['sphere']['el_band_spread_deg']} deg -> {rep['sphere']['el_band_verdict']}")
        print(f"hook liveness (Torch-substep warning per arm): {rep['hook_liveness']}")
        print(f"devices per arm (reference {REFERENCE}): {json.dumps(rep['devices_by_arm'].get('sphere'))}")
    e = rep["extra"]
    print(f"files: {e['n_json']} json / {e['n_npz']} npz / {e['n_txt']} consoles; devices {e['devices']}; slim {e['slim']}")
    print(f"analysis code drift vs stamp: {e['analysis_code_drift'] or 'none'}")
    print(f"expected outputs missing: {len(rep['expected_missing'])}")
    print("problems: " + ("; ".join(rep["problems"][:40]) if rep["problems"] else "none") + (f" ... ({len(rep['problems'])} total)" if len(rep["problems"]) > 40 else ""))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(common.to_jsonable(dict(rep, generator=" ".join(sys.argv))), f, indent=1)
        print(f"written {args.json}")
    return 1 if rep["problems"] else 0


# ==================================================================================================== analyse
KEY = ("arm", "type", "diam_deg", "statistic", "against", "role")
CARRY = ("stim_mean", "stim_sd", "null_mean", "null_sd", "diff", "z", "p", "p_holm", "verdict", "survives_holm", "same_device_as_reference", "stim_n", "null_n")
SIDE_ROLES = ("P1_carrier", "T2_carrier", "P2_lc11", "upstream_windowed", "large_rungs", "lc10a")


def _key(c: dict) -> tuple:
    return tuple(str(c.get(k)) for k in KEY)


def load_round2(path: str) -> dict:
    if not Path(path).exists():
        return {}
    d = json.load(open(path, encoding="utf-8"))
    comps = [c for c in d["tables"]["sphere_comparisons"] if c.get("arm") in ARMS3]
    per_run = pd.DataFrame(d["tables"]["sphere_per_run"]); per_run = per_run[per_run.arm.isin(ARMS3)]
    up = pd.DataFrame(d["tables"]["sphere_upstream_runs"]); up = up[up.arm.isin(ARMS3)]
    levels = pd.DataFrame(d["tables"]["levels"]); levels = levels[levels.arm.isin(ARMS3)]
    return {"path": path, "comparisons": comps, "per_run": per_run, "upstream_runs": up, "levels": levels, "answers": d["summary"]["answers"]["sphere"],
            "classification": d["summary"]["classification"], "devices_by_arm": (d["summary"].get("devices_by_arm") or {}).get("sphere") or ROUND2_DEVICES,
            "created_utc": d.get("created_utc"), "run_id": d.get("run_id")}


def side_by_side(comps3: list, comps2: list) -> pd.DataFrame:
    """Round-3 row beside the round-2 row with the same (arm, type, rung, statistic, question, role)."""
    r2 = {_key(c): c for c in comps2}
    rows = []
    for c in comps3:
        if c.get("role") not in SIDE_ROLES:
            continue
        o = r2.get(_key(c), {})
        row = {k: c.get(k) for k in KEY}
        for k in CARRY:
            row[f"r3_{k}"] = c.get(k); row[f"r2_{k}"] = o.get(k)
        if c.get("against") == "base":
            row["r3_ratio_to_base"] = (c["stim_mean"] / c["null_mean"]) if c.get("null_mean") not in (None, 0) and np.isfinite(c.get("null_mean", np.nan)) else None
            row["r2_ratio_to_base"] = (o["stim_mean"] / o["null_mean"]) if o and o.get("null_mean") not in (None, 0) and np.isfinite(o.get("null_mean", np.nan)) else None
        row["verdict_agrees"] = (o.get("verdict") == c.get("verdict")) if o else None
        rows.append(row)
    return pd.DataFrame(rows)


def cross_batch_base(comps3: list, comps2: list) -> list:
    """base (this batch) vs base (round 2), object runs and nulls, per type x statistic x rung: exploratory, no call."""
    r2 = {_key(c): c for c in comps2 if c.get("against") == "null" and c.get("arm") == REFERENCE}
    out = []
    for c in comps3:
        if c.get("against") != "null" or c.get("arm") != REFERENCE or c.get("role") not in SIDE_ROLES:
            continue
        o = r2.get(_key(c))
        if not o:
            continue
        lab = {"type": c["type"], "diam_deg": c["diam_deg"], "statistic": c["statistic"], "role": "cross_batch_base", "arm": "base:r3", "against": "base:r2",
               "same_device_as_reference": False, "note": "EXPLORATORY: two batches on two GPU models differ in more than the model; no call"}
        out.append(orb.compare_row(c["stim_values"], o["stim_values"], dict(lab, kind="object_runs")))
        out.append(orb.compare_row(c["null_values"], o["null_values"], dict(lab, kind="blank_blank_runs")))
    return out


def _rung_rows(cdf: pd.DataFrame, arm: str, role: str, rungs) -> list:
    rows = []
    for dd in rungs:
        gn = cdf[(cdf.arm == arm) & (cdf.role == role) & (cdf.against == "null") & (cdf.diam_deg == dd)]
        gb = cdf[(cdf.arm == arm) & (cdf.role == role) & (cdf.against == "base") & (cdf.diam_deg == dd)]
        r = {"diam_deg": dd}
        for q, g in (("vs_null", gn), ("vs_base", gb)):
            if len(g):
                x = g.iloc[0]
                r.update({f"{q}_verdict": x.verdict, f"{q}_z": float(x.z), f"{q}_p": float(x.p), f"{q}_p_holm": float(x.p_holm), f"{q}_survives": bool(x.survives_holm), f"{q}_diff": float(x["diff"])})
                if q == "vs_base":
                    r["same_device"] = bool(x.get("same_device_as_reference", False))
                    r["ratio_to_base"] = float(x.stim_mean / x.null_mean) if x.null_mean not in (0, None) and np.isfinite(x.null_mean) else None
                    r["arm_mean"] = float(x.stim_mean); r["arm_sd"] = float(x.stim_sd); r["base_mean"] = float(x.null_mean); r["base_sd"] = float(x.null_sd)
                else:
                    r["null_mean"] = float(x.null_mean)
        r["both_result"] = bool(r.get("vs_null_survives") and r.get("vs_base_survives") and r.get("vs_base_diff", 0) > 0)
        r["both_null"] = bool(r.get("vs_null_verdict") == "null" and r.get("vs_base_verdict") == "null")
        rows.append(r)
    return rows


def reproduction(sph: dict, r2: dict) -> dict:
    cdf = pd.DataFrame(sph["comparisons"]); c2 = pd.DataFrame(r2["comparisons"]) if r2 else pd.DataFrame()
    devs = sph["devices_by_arm"]
    vs_base = cdf[cdf.against == "base"]
    rep = {"device": {"devices_by_arm": devs, "one_model": bool(devs and all(len(v) == 1 for v in devs.values()) and len({tuple(v) for v in devs.values()}) == 1),
                      "same_device_rows": f"{int(vs_base.same_device_as_reference.sum())}/{len(vs_base)}",
                      "all_vs_base_rows_same_device": bool(len(vs_base) and vs_base.same_device_as_reference.all()),
                      "round2_devices_by_arm": (r2 or {}).get("devices_by_arm")}}
    rep["closed"] = rep["device"]["one_model"] and rep["device"]["all_vs_base_rows_same_device"]

    def per_arm(arm):
        a = {}
        for t, role in (("T3", "P1_carrier"), ("T2", "T2_carrier"), ("LC11", "P2_lc11")):
            a[t] = {"r3": _rung_rows(cdf, arm, role, SMALL_RUNGS), "r2": _rung_rows(c2, arm, role, SMALL_RUNGS) if len(c2) else None}
        comp = {}
        for t in ("T3", "T2"):
            g3 = cdf[(cdf.arm == arm) & (cdf.role == "upstream_windowed") & (cdf.type == t) & (cdf.against == "base")]
            g2 = c2[(c2.arm == arm) & (c2.role == "upstream_windowed") & (c2.type == t) & (c2.against == "base")] if len(c2) else pd.DataFrame()
            rows = []
            for _, x in g3.sort_values("diam_deg").iterrows():
                r = {"diam_deg": float(x.diam_deg), "r3_verdict": x.verdict, "r3_z": float(x.z), "r3_p": float(x.p), "r3_ratio": float(x.stim_mean / x.null_mean) if x.null_mean else None,
                     "r3_arm_median_of_medians": float(x.stim_mean), "r3_base": float(x.null_mean)}
                y = g2[g2.diam_deg == x.diam_deg] if len(g2) else g2
                if len(y):
                    y = y.iloc[0]; r.update({"r2_verdict": y.verdict, "r2_z": float(y.z), "r2_ratio": float(y.stim_mean / y.null_mean) if y.null_mean else None})
                rows.append(r)
            comp[t] = rows
        a["companion"] = comp
        a["answers_r3"] = sph["answers"].get(arm); a["answers_r2"] = (r2 or {}).get("answers", {}).get(arm)
        return a

    # rectify: results vs base on T3 and T2 at every small rung, both questions
    rc = per_arm("rectify")
    t3_all = all(r["both_result"] for r in rc["T3"]["r3"]); t2_all = all(r["both_result"] for r in rc["T2"]["r3"]); t3_any = any(r["both_result"] for r in rc["T3"]["r3"])
    rc["verdict"] = "REPRODUCES" if (t3_all and t2_all) else ("PARTIAL" if t3_any else "NOT REPRODUCED")
    rc["carries_small_field"] = {"r3": bool(sph["answers"]["rectify"]["carries_small_field"]), "r2": bool((rc["answers_r2"] or {}).get("carries_small_field")) if rc["answers_r2"] else None}
    rc["ratio_to_base"] = {t: {"r3": [r.get("ratio_to_base") for r in rc[t]["r3"]], "r2": [r.get("ratio_to_base") for r in rc[t]["r2"]] if rc[t]["r2"] else None} for t in ("T3", "T2")}
    rc["companion_all_null_small"] = all(x["r3_verdict"] == "null" for t in ("T3", "T2") for x in rc["companion"][t] if x["diam_deg"] in SMALL_RUNGS)
    rc["companion_all_null_all_rungs"] = all(x["r3_verdict"] == "null" for t in ("T3", "T2") for x in rc["companion"][t])
    # suppress: null on both questions at every small rung on T3, T2, LC11
    sc = per_arm("suppress")
    sc["all_null"] = {t: all(r["both_null"] for r in sc[t]["r3"]) for t in ("T3", "T2", "LC11")}
    sc["verdict"] = "REPRODUCE" if all(sc["all_null"].values()) else "NOT REPRODUCED"
    sc["ratio_to_base"] = {t: {"r3": [r.get("ratio_to_base") for r in sc[t]["r3"]], "r2": [r.get("ratio_to_base") for r in sc[t]["r2"]] if sc[t]["r2"] else None} for t in ("T3", "T2")}
    sc["companion_all_null_all_rungs"] = all(x["r3_verdict"] == "null" for t in ("T3", "T2") for x in sc["companion"][t])
    rep["rectify"] = rc; rep["suppress"] = sc
    rep["lc11_follows"] = {a: {"r3": bool(sph["answers"][a]["lc11_follows"]), "r2": bool((r2 or {}).get("answers", {}).get(a, {}).get("lc11_follows")) if r2 else None} for a in ARMS3}
    rep["lc11_follows_in_no_arm"] = not any(v["r3"] for v in rep["lc11_follows"].values())
    lv = sph["levels"].set_index("arm"); lv2 = r2["levels"].set_index("arm") if r2 and len(r2["levels"]) else None
    rep["levels"] = {a: {k: {"r3": float(lv.loc[a, k]) if a in lv.index else None, "r2": float(lv2.loc[a, k]) if lv2 is not None and a in lv2.index else None}
                         for k in ("T3.dev_mean_b", "T3.dev_abs_mean_b", "T2.dev_mean_b", "LC11.drive_mean_mv_b", "LC11.rate_hz_mean_b", "LC10a.drive_mean_mv_b")} for a in ARMS3}
    return rep


def _fmt_rung(r: dict) -> str:
    return (f"{r['diam_deg']:>5.1f} deg  arm {r.get('arm_mean', float('nan')):+.5f} +- {r.get('arm_sd', float('nan')):.5f}  base {r.get('base_mean', float('nan')):+.5f}  null {r.get('null_mean', float('nan')):+.5f}  "
            f"(i) {r.get('vs_null_verdict')} z {r.get('vs_null_z', float('nan')):+.2f} p_holm {r.get('vs_null_p_holm', float('nan')):.4f}  "
            f"(ii) {r.get('vs_base_verdict')} z {r.get('vs_base_z', float('nan')):+.2f} p_holm {r.get('vs_base_p_holm', float('nan')):.4f}  ratio {r.get('ratio_to_base') if r.get('ratio_to_base') is None else round(r['ratio_to_base'], 3)}"
            f"{'' if r.get('same_device', True) else '  DEVICE-CROSSED'}")


def cmd_analyse(args) -> int:
    out = args.out.rstrip("/")
    rep = None
    if not args.no_verify:
        rep = verify_all(out, args.log)
        print(f"verify: log {rep['log']!r}; problems: {len(rep['problems'])} {rep['problems'][:8]}")
        if rep["problems"] and not args.force:
            sys.exit("batch verification failed (pass --force to analyse anyway; the Result then records the problems)")
    win, win_summary = orc.load_windows(args.rf_maps)
    print(f"windows: maps {win_summary['maps']}; sources {win_summary['sources']}; box widths {win_summary['box_width_by_type_deg']}")
    sph = orc.analyse_sphere(out, win)
    if not sph:
        sys.exit(f"no sphere runs under {out}/sph")
    r2 = load_round2(args.round2)
    sbs = side_by_side(sph["comparisons"], r2["comparisons"]) if r2 else pd.DataFrame()
    xb = cross_batch_base(sph["comparisons"], r2["comparisons"]) if r2 else []
    repro = reproduction(sph, r2)
    stamp = json.load(open(Path(out) / "predeclared.json", encoding="utf-8")) if (Path(out) / "predeclared.json").exists() else None
    prov = sph["prov0"] or common.provenance(__import__("flyverse.connectome", fromlist=["load"]).load(verbose=False))
    res = common.Result.new("lesion", prov)
    res.run_id = "objr3sd-samedevice"; res.tool_version = "object_round3_samedevice/1"
    res.replicates = {"n": {"sphere": N_RUNS}, "unit": "runs", "null": "each arm's own blank/blank runs of the same submission; the base arm's object runs of the SAME submission and box for the model comparison",
                      "n_runs": sph["n_runs"], "batch": {"jobs_json": f"{out}/jobs.json", "cluster_log": args.log, "block": BLOCK}}
    sph["per_body"].to_csv(f"{out}/per_body_sphere.csv", index=False)
    res.add_table("sphere_per_run", sph["per_run"]); res.add_table("sphere_upstream_runs", sph["upstream_runs"]); res.add_table("sphere_comparisons", sph["comparisons"])
    res.add_table("sphere_preference", sph["preference"]); res.add_table("sphere_time_course", sph["time_course"]); res.add_table("sphere_footprint", sph["footprint"])
    res.add_table("sphere_footprint_runs", sph["footprint_runs"]); res.add_table("levels", sph["levels"]); res.add_table("levels_runs", sph["levels_runs"])
    res.add_table("windows", win)
    if len(sbs):
        res.add_table("side_by_side_round2", sbs)
    if xb:
        res.add_table("cross_batch_base", xb)
    cls = {a: {"letter": ARMS[a]["letter"], **{k: sph["answers"][a][k] for k in ("carries_small_field", "carries_T3_only", "lc11_follows")}, "adopted": False} for a in sph["arms"]}
    res.summary = {"predeclared": (stamp or {}).get("predeclared", PREDECLARED), "predeclared_stamped_utc": (stamp or {}).get("stamped_utc"),
                   "predeclared_live_equals_stamp": bool(stamp) and json.dumps(common.to_jsonable(stamp["predeclared"]), sort_keys=True) == json.dumps(common.to_jsonable(PREDECLARED), sort_keys=True),
                   "arms": ARMS, "windows": win_summary, "answers": sph["answers"], "classification": cls, "reproduction": repro,
                   "devices_by_arm": {"reference_arm": REFERENCE, "sphere": sph["devices_by_arm"], "round2_sphere": (r2 or {}).get("devices_by_arm")},
                   "round2": {k: r2.get(k) for k in ("path", "run_id", "created_utc", "classification")} if r2 else None,
                   "analysis_code_sha256": analysis_code_sha(), "analysis_code_stamped_sha256": (stamp or {}).get("analysis_code_sha256"),
                   "analysis_code_drift": (rep or {}).get("extra", {}).get("analysis_code_drift") if rep else [f for f, h in ((stamp or {}).get("analysis_code_sha256") or {}).items() if analysis_code_sha().get(f) != h],
                   "verify": rep,
                   "reading": "verdicts are common.compare's; p the tie-aware exact permutation U; p_holm Holm within the round-2 family (3 small rungs per arm x type x question); "
                              "every arm-vs-base row carries same_device_as_reference, read off the realised device_name of every run of this batch; "
                              "round-2 numbers are quoted BESIDE round-3 numbers (side_by_side_round2) and never tested against them except in cross_batch_base, which is exploratory; "
                              "P1 / T2 are within-run maxima over cells and travel with the per-body windowed median (abs_drive_median, role upstream_windowed); nothing is adopted"}
    res.validation = {"name": "same-device replication of object round 2's base / rectify / suppress on the matched sphere ladder (one submission, one block, one box)",
                      "reference": {"round2": args.round2, "P1": "object_compare_r2.md section 4: rectify 6.01 / 10.49 / 8.99x base at T3, result x3 on both questions; suppress null x3"},
                      "measured": {"closed": repro["closed"], "rectify": repro["rectify"]["verdict"], "suppress": repro["suppress"]["verdict"], "lc11_follows_in_no_arm": repro["lc11_follows_in_no_arm"],
                                   "rectify_companion_all_null_small": repro["rectify"]["companion_all_null_small"]},
                      "status": "measured", "source": "scripts/object_round3_samedevice.py analyse"}
    res.files = {"generator": " ".join(sys.argv), "per_body_sphere": f"{out}/per_body_sphere.csv", "predeclared": f"{out}/predeclared.json", "jobs": f"{out}/jobs.json",
                 "cluster_log": args.log, "rf_maps": win_summary["maps"], "round2_result": args.round2, "tree_state": f"{out}/tree_state.json"}
    p = res.save(args.json)
    if len(sbs):
        sbs.to_csv(f"{out}/side_by_side_round2.csv", index=False)
    # ---------------------------------------------------------------- console
    print(f"\n== DEVICES per arm (sphere, reference {REFERENCE}): r3 {json.dumps(sph['devices_by_arm'])}   r2 {json.dumps((r2 or {}).get('devices_by_arm'))}")
    print(f"   confound closed: {repro['closed']} (one model {repro['device']['one_model']}; same-device vs-base rows {repro['device']['same_device_rows']})")
    print("\n== SPHERE: footprint per arm x rung (captured radiance; mean over runs)")
    common.print_table(sph["footprint"], floatfmt="{:.3f}", max_rows=60)
    print("\n== SPHERE: blank-arm levels per arm (r3 mean over all 30 runs of the arm | r2)")
    for a, v in repro["levels"].items():
        print(f"  {a:9s} " + "  ".join(f"{k} {x['r3']:+.5f}|{x['r2'] if x['r2'] is None else format(x['r2'], '+.5f')}" for k, x in v.items()))
    for arm in ("rectify", "suppress"):
        for t in ("T3", "T2", "LC11"):
            print(f"\n== {arm} {t} {'diff_signed_best_cell (MAX OVER CELLS)' if t != 'LC11' else 'windowed drive_median'}, small rungs, Holm within 3: ROUND 3 (this batch) then ROUND 2")
            for r in repro[arm][t]["r3"]:
                print("  r3 " + _fmt_rung(r))
            for r in (repro[arm][t]["r2"] or []):
                print("  r2 " + _fmt_rung(r))
        print(f"\n== {arm}: per-body RF-windowed median |drive| (abs_drive_median, the companion), vs base, all rungs")
        for t in ("T3", "T2"):
            for x in repro[arm]["companion"][t]:
                print(f"  {t} {x['diam_deg']:>5.1f} deg  r3 {x['r3_verdict']:12s} z {x['r3_z']:+.2f} ratio {x['r3_ratio']:.3f} (arm {x['r3_arm_median_of_medians']:.5f} base {x['r3_base']:.5f})   "
                      + (f"r2 {x['r2_verdict']:12s} z {x['r2_z']:+.2f} ratio {x['r2_ratio']:.3f}" if x.get("r2_verdict") else "r2 (no row)"))
    print("\n== REPRODUCTION")
    print(json.dumps(common.to_jsonable({"closed": repro["closed"], "rectify": {k: repro["rectify"][k] for k in ("verdict", "carries_small_field", "ratio_to_base", "companion_all_null_small", "companion_all_null_all_rungs")},
                                         "suppress": {k: repro["suppress"][k] for k in ("verdict", "all_null", "ratio_to_base", "companion_all_null_all_rungs")},
                                         "lc11_follows": repro["lc11_follows"]}), indent=1))
    if xb:
        print("\n== CROSS-BATCH base r3 (H200) vs base r2 (B200), EXPLORATORY, no call")
        for c in xb:
            if c["role"] == "cross_batch_base" and c["statistic"] in ("diff_signed_best_cell", "drive_median", "abs_drive_median"):
                print(f"  {c['type']:5s} {c['diam_deg']:>5} {c['statistic']:22s} {c['kind']:16s} r3 {c['stim_mean']:+.5f} +- {c['stim_sd']:.5f}  r2 {c['null_mean']:+.5f} +- {c['null_sd']:.5f}  z {c['z']:+.2f} p {c['p']:.4f} -> {c['verdict']}")
    print("\n== CLASSIFICATION (r3)"); print(json.dumps(common.to_jsonable(cls), indent=1))
    print(f"\nanalysis code drift vs stamp: {res.summary['analysis_code_drift'] or 'none'}; problems: {res.check() or 'none'}; written {p}")
    return 0


# ==================================================================================================== main
def _plan_args(p):
    p.add_argument("--seed-offset", type=int, default=0, help="offset object and null seeds for an independent replication")
    p.add_argument("--runs", type=int, default=N_RUNS); p.add_argument("--seconds", type=float, default=12.0); p.add_argument("--settle", type=float, default=3.0)


def cmd_check_raw(args):
    """Independent vectorized raw-to-run reduction (no baseline window reducer)."""
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    win = pd.DataFrame(result["tables"]["windows"]).set_index("bodyId")
    runs = pd.DataFrame(result["tables"]["sphere_per_run"])
    rows = []; checked = 0
    for path in sorted(Path(args.out, "sph").glob("*.json")):
        name = path.stem; nm = orc.parse_sphere_name(name)
        if nm is None: continue
        recorded = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            az, el = z["track_az_deg"], z["track_el_deg"]
            for block, types in (("lc", orc.LC_TYPES), ("rate", KEEP_TYPES)):
                labels = z[block + "_types"].astype(str); ids = z[block + "_body_ids"].astype(str)
                ix = np.flatnonzero(np.isin(labels, types)); labels = labels[ix]; ids = ids[ix]
                key = "lc_drive_mv" if block == "lc" else "optic_dr"
                d = z["a__" + key][:, ix].astype(float) - z["b__" + key][:, ix].astype(float)
                for t in types:
                    mask = labels == t
                    if block == "rate":
                        measured = np.abs(d[:, mask].mean(0)).max()
                        want = recorded["summary"]["diff_signed_best_cell"][t]
                        if not np.isclose(measured, want, atol=1e-7, rtol=1e-5):
                            rows.append({"run":name,"type":t,"statistic":"diff_signed_best_cell","got":measured,"want":want})
                        checked += 1
                ww = win.reindex(ids)
                for diam in (DIAMS if nm["null"] else [nm["diam"]]):
                    valid = ((np.abs(az[:,None] - ww.az_deg.to_numpy()[None]) <= (ww.width_deg.to_numpy()[None]+diam)/2)
                             & (np.abs(el[:,None] - ww.el_deg.to_numpy()[None]) <= (ww.height_deg.to_numpy()[None]+diam)/2))
                    valid[:, ww.window_source.isna().to_numpy() | (ww.window_source == "whole").to_numpy()] = True
                    n = valid.sum(0)
                    means = np.divide((d*valid).sum(0), n, out=np.full(len(ids),np.nan), where=n>0)
                    target = runs[(runs.run==name)&(runs.diam_deg==diam)].set_index("type")
                    for t in types:
                        values = means[(labels==t)&(n>0)]
                        for statistic, measured in (("drive_median",float(np.median(values)) if len(values) else np.nan),
                                                   ("abs_drive_median",float(np.median(abs(values))) if len(values) else np.nan)):
                            if statistic == "abs_drive_median" and block == "lc": continue
                            want = target.loc[t,statistic]
                            if not np.isclose(measured,want,atol=1e-9,rtol=1e-8,equal_nan=True):
                                rows.append({"run":name,"type":t,"diam":diam,"statistic":statistic,"got":measured,"want":want})
                            checked += 1
    report = {"schema":"flyverse.sphere_raw_check/1","generator":" ".join(sys.argv),"provenance":result["provenance"],
              "analysis_code":analysis_code_sha(),"values_checked":checked,"differences":rows,
              "scope":"Every run, every ladder window, LC median plus upstream median and maximum, directly from frame arrays",
              "verdict":"sound" if not rows else "unsound"}
    Path(args.json).write_text(json.dumps(common.to_jsonable(report),indent=1),encoding="utf-8")
    print(f"independent raw check: {checked} values, {len(rows)} differences",flush=True)
    return int(bool(rows))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", default="out/objr3sd"); p.add_argument("--name", default="objr3sd"); p.add_argument("--minutes", type=int, default=240)
    _plan_args(p); p.set_defaults(func=cmd_plan)
    s = sub.add_parser("submit"); s.add_argument("--out", default="out/objr3sd"); s.set_defaults(func=cmd_submit)
    s.add_argument("--target"); s.add_argument("--node")
    rj = sub.add_parser("run-job"); rj.add_argument("--job", required=True); rj.add_argument(f"--{BLOCK_KEY}", default=BLOCK, help="the block token cluster_run --arm-block reads off the command line")
    rj.add_argument("--out", default="out/objr3sd"); _plan_args(rj); rj.set_defaults(func=cmd_run_job)
    sl = sub.add_parser("slim"); sl.add_argument("--out", default="out/objr3sd"); sl.set_defaults(func=cmd_slim)
    asm = sub.add_parser("assemble"); asm.add_argument("--out", default="out/objr3sd"); asm.set_defaults(func=cmd_assemble)
    v = sub.add_parser("verify"); v.add_argument("--out", default="out/objr3sd"); v.add_argument("--log", default="out/objr3sd_cluster.log"); v.add_argument("--json", default="out/objr3sd/verify.json"); v.set_defaults(func=cmd_verify)
    a = sub.add_parser("analyse"); a.add_argument("--out", default="out/objr3sd"); a.add_argument("--log", default="out/objr3sd_cluster.log"); a.add_argument("--json", default="out/objr3sd/samedevice.json")
    a.add_argument("--round2", default=ROUND2_RESULT); a.add_argument("--rf-maps", nargs="+", default=orc.DEFAULT_MAPS)
    a.add_argument("--no-verify", action="store_true"); a.add_argument("--force", action="store_true"); a.set_defaults(func=cmd_analyse)
    c = sub.add_parser("check-raw"); c.add_argument("--out", default="out/objr3sd"); c.add_argument("--result", default="out/objr3sd/samedevice.json")
    c.add_argument("--json", default="out/objr3sd/skeptic_raw.json"); c.set_defaults(func=cmd_check_raw)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
