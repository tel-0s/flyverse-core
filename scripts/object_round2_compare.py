"""Round-2 fixed-anatomy MODEL COMPARISON (Neurome intake, experiment 2 + 3): the batch builder, the batch check, and the
analysis. A diagnosis, not a tuning: every arm is an opt-in OpticParams override (docs/audits/optic_stream_hooks.md), the
graph is untouched, the shipped defaults stay off, nothing is adopted.

ONE cluster submission (docs/INTERP.md 2.4 / 10.4: runs are the replicate unit, every arm of a comparison in ONE
submission), three parts, all under out/objr2c/:

  sph/    the MATCHED sphere ladder (scripts/probe_object_matched.py run: elevation 0, 5 cm, +-50 deg at 40 deg/s, dark
          ball, eye 0.15 m, headlamp) at 4.5 / 8.8 / 11 / 20 / 30 deg, N_RUNS object runs per rung + N_RUNS blank/blank
          nulls, under EVERY arm of ARMS (the same seeds, the same protocol, the same submission).
  spec/   the SPECIFICITY battery (scripts/probe_synthetic_stimuli.py record, radiance path, contrast-matched Weber
          +-0.995): the 11-deg dark and bright squares on the sphere's arc, a 7-deg dark bar, a 30-deg grating, 2-Hz
          full-field flicker, isolated ON and OFF 15-deg flashes at (-30, 0) deg, and a blank/blank null; N_SPEC runs
          each, under every arm.
  bench/  TRANSFER: scripts/benchmark.py --sections motion,loom_escape under every arm (the arm installed through the
          same override path), N_BENCH draws per arm.

Arms (ARMS; `base` = A, `fb0` = B, `rectify` = C, `adapt100` / `adapt300` = D, `suppress` = E, `rect_adapt` = F,
`rect_supp` = G). The streams are the carrier classes docs/audits/deficit_object.md 2 names: at T3 the raising
{Mi1, Tm3, Tm2} and the lowering {Tm1, Tm4} (all excitatory, exact tier); at T2 the raising {Tm2, L5, Tm3, Mi1} and the
lowering {C3} (C3 -> T2 is INHIBITORY: the 'neg' half-wave keeps the sign of W, so the inhibitory guarantee of the hook
is exercised on a real stream here, not only in the unit test). spatial_suppress has no post regex (it transforms the
presynaptic cell's signal for every target), so E / G suppress the medulla carriers Mi1 / Tm1 / Tm2 / Tm3 / Tm4 for
every target, T2 / T3 included.

    python scripts/object_round2_compare.py plan   [--out out/objr2c --name objr2c --minutes 240 --runs 5 --spec-runs 4 --bench-draws 3 --seconds 12 --settle 3 --spec-seconds 6]
    python scripts/object_round2_compare.py submit [--out out/objr2c]           # runs the plan through scripts/cluster_run.py, tees out/objr2c_cluster.log
    python scripts/object_round2_compare.py run-job --job NAME --out ...        # on the box: the job's processes in sequence, then check-job
    python scripts/object_round2_compare.py bench --arm ARM --seed K --out STEM # GPU: benchmark.py motion + loom_escape under the arm
    python scripts/object_round2_compare.py verify  [--out out/objr2c --log out/objr2c_cluster.log --json out/objr2c/verify.json]
    python scripts/object_round2_compare.py analyse [--out out/objr2c --json out/interp/objr2c/compare.json --rf-maps CSV...]
    python scripts/object_round2_compare.py streams                             # CPU: the arms' OpticParams overrides as the probes receive them

The reading rules (PREDECLARED below) are written before the batch is submitted and dumped to out/objr2c/predeclared.json
by `plan`; `analyse` copies them into the Result unchanged.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
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
import probe_object_matched as pom                     # noqa: E402  (CPU-only module functions)
import probe_synthetic_stimuli as pss                  # noqa: E402  (CPU-only: verify_dir, family_stats)
import object_round2_baseline as orb                   # noqa: E402  (the baseline's statistics helpers and window rule: one implementation)

# ==================================================================================================== the design
DIAMS = [4.5, 8.8, 11.0, 20.0, 30.0]
SMALL_RUNGS, LARGE_RUNGS = (4.5, 8.8, 11.0), (20.0, 30.0)
N_RUNS, N_SPEC, N_BENCH = 5, 4, 3
NULL_DIAM = 11.0
ARC = {"half_span_deg": 50.0, "speed_deg_s": 40.0, "elevation_deg": 0.0}
FLASH = {"size_deg": 15.0, "az_deg": -30.0, "el_deg": 0.0, "on_s": 0.5, "period_s": 1.5,
         "why_here": "the retina has no column within 4.4 deg of (0, 0) and 2 within 7.5; (-30, 0) on the same arc has 3 within 4.4 and 8 within 7.5 (col_az_el of the shipped retina)"}
LC_TYPES = ("LC11", "LC10a")
UPSTREAM = ("T3", "T2", "Tm5Y", "TmY21")
HOOKED = ("T3", "T2")
ALPHA = 0.05

T3_RAISE, T3_LOWER = "^(Mi1|Tm3|Tm2)$", "^(Tm1|Tm4)$"
T2_RAISE, T2_LOWER = "^(Tm2|L5|Tm3|Mi1)$", "^C3$"
T3_ALL, T2_ALL = "^(Mi1|Tm3|Tm2|Tm1|Tm4)$", "^(Tm2|L5|Tm3|Mi1|C3)$"
MEDULLA = "^(Mi1|Tm1|Tm2|Tm3|Tm4)$"
RECT = [[T3_RAISE, "^T3$", "pos"], [T3_LOWER, "^T3$", "neg"], [T2_RAISE, "^T2$", "pos"], [T2_LOWER, "^T2$", "neg"]]
SUPP = [[MEDULLA, 0.5, 10.0]]


def ADAPT(tau_ms: float) -> list:
    return [[T3_ALL, "^T3$", float(tau_ms), 1.0], [T2_ALL, "^T2$", float(tau_ms), 1.0]]


ARMS = {
    "base":       {"letter": "A", "optic": {}, "note": "the shipped sum (OpticParams defaults)"},
    "fb0":        {"letter": "B", "optic": {"gain_fb": 0.0}, "note": "gain_fb = 0: the deterministic lobe (the exact-arithmetic control)"},
    "rectify":    {"letter": "C", "optic": {"stream_rectify": RECT},
                   "note": "per-stream half-wave rectification of the T3 and T2 carriers before their sums: raising classes 'pos', lowering classes 'neg'; weights and signs untouched"},
    "adapt100":   {"letter": "D", "optic": {"stream_adapt": ADAPT(100.0)}, "note": "the same carrier streams minus a 100 ms fast-adaptation state (gain 1: high-passed)"},
    "adapt300":   {"letter": "D'", "optic": {"stream_adapt": ADAPT(300.0)}, "note": "the same streams, 300 ms"},
    "suppress":   {"letter": "E", "optic": {"spatial_suppress": SUPP}, "note": "centre-surround (k 0.5 within 10 deg, same type) on the medulla carriers Mi1 / Tm1 / Tm2 / Tm3 / Tm4, every target"},
    "rect_adapt": {"letter": "F", "optic": {"stream_rectify": RECT, "stream_adapt": ADAPT(100.0)}, "note": "C + D (100 ms)"},
    "rect_supp":  {"letter": "G", "optic": {"stream_rectify": RECT, "spatial_suppress": SUPP}, "note": "C + E"},
}
HOOK_ARMS = tuple(a for a in ARMS if a not in ("base", "fb0"))

SPEC = {   # name -> (label, record arguments); every moving stimulus rides the sphere's arc (ARC); flashes at FLASH
    "dark110":   ("11-deg dark square on the arc (the contrast-matched partner of the sphere's 11-deg rung)", f"--stimulus rect --width 11 --height 11 --contrast {pss.DARK_CONTRAST}"),
    "bright110": ("11-deg bright square on the arc (the contrast-matched bright object)", f"--stimulus rect --width 11 --height 11 --contrast {-pss.DARK_CONTRAST}"),
    "bar":       ("7-deg dark vertical bar on the arc (every elevation)", f"--stimulus bar --width 7 --contrast {pss.DARK_CONTRAST}"),
    "grating":   ("30-deg sine grating, contrast 0.5, front->back", "--stimulus grating --period 30 --contrast 0.5"),
    "flicker":   ("2-Hz full-field square-wave flicker, contrast 0.5 (spatially uniform: the level-shift control)", "--stimulus flicker --hz 2 --contrast 0.5"),
    "flashon":   ("isolated ON transitions: 15-deg bright square at (-30, 0), 0.5 s on / 1.5 s period", f"--stimulus flash --width {FLASH['size_deg']:g} --contrast {-pss.DARK_CONTRAST} --azimuth {FLASH['az_deg']:g} --elevation {FLASH['el_deg']:g} --on-s {FLASH['on_s']:g} --period-s {FLASH['period_s']:g}"),
    "flashoff":  ("isolated OFF transitions: the same square, dark", f"--stimulus flash --width {FLASH['size_deg']:g} --contrast {pss.DARK_CONTRAST} --azimuth {FLASH['az_deg']:g} --elevation {FLASH['el_deg']:g} --on-s {FLASH['on_s']:g} --period-s {FLASH['period_s']:g}"),
    "null":      ("blank vs blank (the null of every spec statistic under the arm)", f"--stimulus rect --width 4.4 --height 8.8 --contrast {pss.DARK_CONTRAST} --null"),
}
SPEC_MOVING = ("dark110", "bright110", "bar", "grating", "null")
BENCH_SECTIONS = "motion,loom_escape"
BENCH_KEYS = ("motion.min_dsi", "motion.correct_directions", "loom_escape.GF_peak_hz", "loom_escape.escapes")

PREDECLARED = {
    "written": "2026-09-13/14, before the batch was submitted (out/objr2c/predeclared.json is stamped by `plan`)",
    "question": "which mechanism class (rectification / adaptation / spatial suppression / none) makes the small-field stage (T3, T2) carry a small-object figure on the matched ladder with specificity preserved, at what magnitude, and whether LC11 output follows -- a diagnosis on fixed anatomy; nothing is adopted, the defaults stay off",
    "replicate_unit": "one process = one (A, B) pair under one brain seed = one run; a cluster job is a container of sequential processes; runs, not cells, not seeds",
    "n_runs": {"sphere_per_arm_per_rung": N_RUNS, "sphere_nulls_per_arm": N_RUNS, "spec_per_arm_per_stimulus": N_SPEC, "spec_nulls_per_arm": N_SPEC, "bench_draws_per_arm": N_BENCH},
    "why_five": "the primary families have 3 members (the three small rungs) so Holm's smallest threshold is 0.05/3 = 0.0167 > the 5 v 5 exact-U floor 0.0079: a fully separated member can survive Holm at five runs per arm. The spec battery at 4 v 4 (floor 0.029) is decidable per stimulus unadjusted; it is read as an exploratory screen per stimulus, not as one Holm family",
    "window_rule": "the baseline's (object_round2_baseline.PREDECLARED['window_rule']): fitted in the shipped localizer map, else the fb0 map, else the anatomical column with the type's median fitted width, else whole. The maps are the ones on disk at analysis time (--rf-maps; default out/objr2/rfmap_ship.csv + out/objr2/rfmap_fb0.csv if the baseline batch has landed, else out/synth2/rfmap_150_fb0_p3.csv): a stimulus-driven RF map is a property of the retinotopy, which no arm changes, so one map serves every arm; the map's provenance is recorded",
    "primary": {
        "P1_carrier": {"statistic": "T3 diff_signed_best_cell (max over cells of |mean_t dr_A - mean_t dr_B|, whole window, probe_object_matched's definition) per run",
                       "family": "per hook arm: the three small rungs {4.5, 8.8, 11} = 3 members, Holm within, two questions each: (i) arm object runs vs the arm's own blank/blank runs (is there a figure), (ii) arm object runs vs the base arm's object runs at the same rung (does the mechanism change it); tie-aware exact permutation U; verdict from common.compare",
                       "call": "an arm CARRIES the small-object figure at T3 if at least one small rung reads `result` with p_holm <= 0.05 on BOTH questions (with a positive diff on (ii)); the magnitude is the arm's mean and SD over runs against the base's; T2 the same family as the second carrier (reported, a call needs both T3 and T2 to hold for 'the small-field stage')"},
        "P2_lc11": {"statistic": "LC11: per run the population MEDIAN over the 143 bodies of the per-body RF-windowed time-mean received drive (A - B, mV); the same median of the windowed spike-count difference is the secondary member set",
                    "family": "per hook arm: {4.5, 8.8, 11} x drive_median = 3 members, Holm within, the same two questions (vs the arm's null; vs base)",
                    "call": "LC11 FOLLOWS if at least one small rung reads `result` with p_holm <= 0.05 vs the arm's null AND vs base (positive diff)"},
        "fb0": "arm B: its blank/blank null has SD ~0 for optic quantities, so question (i) returns `undetermined` (read as magnitudes with p); question (ii) vs base is a plain 5 v 5 comparison and is called",
    },
    "secondary_exploratory": {
        "rungs": "the large rungs {20, 30} on the same statistics (LC10a's target range: Schretter 2024): reported with unadjusted p and Holm within each arm x type x statistic set, exploratory",
        "lc": "LC11 / LC10a spikes_median, drive_mean, drive_max, the whole-window diff_max_over_cells_mean_mv and diff_rate_hz_max_cell; LC10a everything; preference tests (Spearman over the 25 object runs, small-vs-large contrast) per arm",
        "upstream": "T2 (family), Tm5Y / TmY21 (not hooked): diff_signed_best_cell, diff_signed_mean, diff_abs_best_cell_mean kept distinct; RF-windowed medians for T2 / T3 / Tm5Y over the map's fitted bodies",
        "level": "the level-shift question the hooks audit raised: the blank-arm levels per arm (T3 / T2 dev_mean_b, dev_abs_mean_b; LC11 / LC10a drive_mean_mv_b, rate_hz_mean_b) and the flicker stimulus (spatially uniform; a pure level shift gives a figure there, a rectified transient of a local object does not)",
    },
    "specificity": {
        "stimuli": list(SPEC),
        "statistics": "LC11 / LC10a diff_max_over_cells_mean_mv, diff_mean_over_cells_mean_mv, diff_rate_hz_max_cell; T2 / T3 diff_signed_best_cell, diff_abs_best_cell_mean (probe_synthetic_stimuli.family_stats)",
        "comparisons": "per arm x stimulus: (i) vs the arm's blank/blank null (4 v 4), (ii) vs base on the same stimulus (4 v 4); unadjusted, exploratory screen",
        "released": "an arm RELEASES bar / grating / flicker responses at LC11 if, for that stimulus, (ii) reads `result` with diff > 0 on diff_max_over_cells_mean_mv or diff_rate_hz_max_cell (Keles et al. 2020: LC11-specific Rdl disruption reduced small-object responses without releasing bar / grating responses)",
        "on_off": "T2 and T3 KEEP both transitions if the arm's mean diff_signed_best_cell on flashon and on flashoff are each >= 0.5 x the base's mean (a magnitude rule, stated as such; the 4-run scatter is reported beside it)",
        "bright_dark": "bright110 vs dark110 under each arm: the ratio of LC11 diff_max_over_cells_mean_mv means, with both arms' 4-run scatter; no call, a magnitude",
    },
    "transfer": {"sections": BENCH_SECTIONS, "draws": N_BENCH,
                 "reading": "per arm: the benchmark's own PASS / FAIL / KNOWN GAP on motion.min_dsi, motion.correct_directions, loom_escape.GF_peak_hz, loom_escape.escapes per draw, and the values' mean and SD over draws against the base arm's draws (3 v 3: magnitudes, `underpowered` by construction); an arm COSTS a section if any draw fails a check the base passes in every draw"},
    "classification": "each finding is classified under the project rule: a mechanism the physiology implies (Keles 2020: T2 / T3 respond to both ON and OFF; T3 -> LC11 excitatory; Tanaka & Clark 2020: fast-adapting size-tuned inputs) vs a hand-set number (the modes, tau, gain, k, radius are hypotheses, not data); NOTHING IS ADOPTED this round, every hook stays off by default",
}


# ==================================================================================================== helpers
def optic_flags(arm: str) -> str:
    """The arm's OpticParams overrides as --optic KEY=JSON flags (common.parse_kv reads JSON first)."""
    return " ".join(f"--optic {shlex.quote(k + '=' + json.dumps(v))}" for k, v in ARMS[arm]["optic"].items())


def sphere_stem(arm: str, diam: float | None, seed: int) -> str:
    return f"{arm}_null_s{seed}" if diam is None else f"{arm}_d{int(round(diam * 10)):03d}_obj_s{seed}"


def parse_sphere_name(stem: str) -> dict | None:
    m = re.match(rf"({'|'.join(ARMS)})_(?:d(\d{{3}})_obj|null)_s(\d+)$", stem)
    if not m:
        return None
    return {"arm": m.group(1), "null": m.group(2) is None, "diam": None if m.group(2) is None else int(m.group(2)) / 10.0, "seed": int(m.group(3))}


def parse_spec_name(stem: str) -> dict | None:
    m = re.match(rf"({'|'.join(ARMS)})_({'|'.join(SPEC)})_s(\d+)$", stem)
    return {"arm": m.group(1), "stim": m.group(2), "null": m.group(2) == "null", "seed": int(m.group(3))} if m else None


def _sphere_cmd(out: str, arm: str, diam: float | None, seed: int, seconds: float, settle: float) -> tuple[str, str]:
    stem = f"{out}/sph/{sphere_stem(arm, diam, seed)}"
    d = NULL_DIAM if diam is None else diam
    fl = optic_flags(arm)
    cmd = (f"python scripts/probe_object_matched.py run --diam-deg {d:g} --seconds {seconds:g} --settle {settle:g} --seed {seed}"
           f"{' --null' if diam is None else ''}{' ' + fl if fl else ''} --out {stem} > {stem}.txt 2>&1")
    return stem, cmd


def _spec_cmd(out: str, arm: str, stim: str, seed: int, seconds: float, settle: float) -> tuple[str, str]:
    stem = f"{out}/spec/{arm}_{stim}_s{seed}"
    _, a = SPEC[stim]
    arc = f" --speed {ARC['speed_deg_s']:g} --elevation {ARC['elevation_deg']:g} --half-span {ARC['half_span_deg']:g}" if stim in SPEC_MOVING else ""
    fl = optic_flags(arm)
    cmd = (f"python scripts/probe_synthetic_stimuli.py record {a}{arc} --seconds {seconds:g} --settle {settle:g} --seed {seed}"
           f"{' ' + fl if fl else ''} --out {stem} > {stem}.txt 2>&1")
    return stem, cmd


def _bench_cmd(out: str, arm: str, k: int) -> tuple[str, str]:
    stem = f"{out}/bench/{arm}_k{k}"
    cmd = f"python scripts/object_round2_compare.py bench --arm {arm} --seed {k} --out {stem} > {stem}.txt 2>&1"
    return stem, cmd


def _job_line(out: str, job: str, a) -> str:
    return (f"mkdir -p {out}/sph {out}/spec {out}/bench && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/object_round2_compare.py run-job --job {job} --out {out} --runs {a.runs} --spec-runs {a.spec_runs} --bench-draws {a.bench_draws} "
            f"--seconds {a.seconds:g} --settle {a.settle:g} --spec-seconds {a.spec_seconds:g} --spec-settle {a.spec_settle:g}")


def build_jobs(out: str, a) -> list[dict]:
    """20 jobs: 8 sphere jobs (one per arm: every rung x every seed + the nulls, 30 processes), 4 spec jobs (two arms
    each: 8 stimuli x N_SPEC seeds x 2 = 64 short processes), 8 bench jobs (one per arm, N_BENCH draws)."""
    jobs = []

    def add(name, kind, procs):
        exp = {"sph": ".json", "spec": "_summary.json", "bench": ".json"}[kind]
        jobs.append({"job": name, "kind": kind, "stems": [p[0] for p in procs], "commands": [p[1] for p in procs],
                     "expect": [p[0] + exp for p in procs], "line": _job_line(out, name, a)})

    for arm in ARMS:
        add(f"sph_{arm}", "sph", [_sphere_cmd(out, arm, d, s, a.seconds, a.settle) for s in range(a.runs) for d in DIAMS + [None]])
    arms = list(ARMS)
    for i in range(0, len(arms), 2):
        pair = arms[i:i + 2]
        add("spec_" + "_".join(pair), "spec", [_spec_cmd(out, arm, st, s, a.spec_seconds, a.spec_settle) for arm in pair for s in range(a.spec_runs) for st in SPEC])
    for arm in ARMS:
        add(f"bench_{arm}", "bench", [_bench_cmd(out, arm, k) for k in range(a.bench_draws)])
    return jobs


# ==================================================================================================== plan / submit / run-job
def cmd_plan(args) -> int:
    out = args.out.rstrip("/")
    Path(out).mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(out, args)
    n_proc = sum(len(j["stems"]) for j in jobs)
    if len(jobs) > 20:
        raise SystemExit(f"{len(jobs)} jobs > the 20-job cap of one submission on the rented boxes")
    log = f"out/{args.name}_cluster.log"
    sh = ["#!/bin/sh", f"# generated by scripts/object_round2_compare.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs, {n_proc} processes", "set -e",
          f"mkdir -p {out} out", f'if [ -f {log} ]; then mv {log} "out/{args.name}_cluster.$(date +%Y%m%dT%H%M%S).log"; fi',
          f"python scripts/cluster_run.py --name {args.name} --minutes {args.minutes} \\"]
    sh += [f"  {shlex.quote(j['line'])} \\" for j in jobs]
    sh += [f"  --fetch {out}/ 2>&1 | tee {log}"]
    (Path(out) / "batch.sh").write_text("\n".join(sh) + "\n", encoding="utf-8")
    with open(Path(out) / "jobs.json", "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "minutes": args.minutes, "runs": args.runs, "spec_runs": args.spec_runs, "bench_draws": args.bench_draws,
                   "seconds": args.seconds, "settle": args.settle, "spec_seconds": args.spec_seconds, "spec_settle": args.spec_settle, "out": out,
                   "diameters_deg": DIAMS, "arms": ARMS, "spec": {k: {"label": v[0], "args": v[1]} for k, v in SPEC.items()}, "arc": ARC, "flash": FLASH,
                   "bench_sections": BENCH_SECTIONS, "jobs": jobs, "n_processes": n_proc, "fetch": out + "/", "log": log}, f, indent=1)
    with open(Path(out) / "predeclared.json", "w", encoding="utf-8") as f:
        json.dump({"predeclared": PREDECLARED, "arms": ARMS, "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generator": " ".join(sys.argv)}, f, indent=1)
    with open(Path(out) / "tree_state.json", "w", encoding="utf-8") as f:
        json.dump(orb.tree_state(), f, indent=1)
    print(f"{len(jobs)} job(s), {n_proc} processes -> {out}/ (log {log}); batch.sh, jobs.json, predeclared.json, tree_state.json written")
    for j in jobs:
        print(f"  {j['job']:22s} {len(j['stems']):3d} processes: {Path(j['stems'][0]).name} .. {Path(j['stems'][-1]).name}")
    return 0


def cmd_submit(args) -> int:
    out = args.out.rstrip("/")
    with open(Path(out) / "jobs.json", encoding="utf-8") as f:
        plan = json.load(f)
    lines = [j["line"] for j in plan["jobs"]]
    cmd = [sys.executable, "scripts/cluster_run.py", "--name", plan["name"], "--minutes", str(plan["minutes"]), *lines, "--fetch", plan["fetch"]]
    log = Path(plan["log"])
    if log.exists():
        log.rename(log.with_name(log.stem + time.strftime(".%Y%m%dT%H%M%S") + ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"submitting {len(lines)} job(s) ({plan['n_processes']} processes) through scripts/cluster_run.py; log {log}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=env)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line); f.flush()
        rc = p.wait()
    print(f"cluster_run exit {rc}; log {log}")
    return rc


def cmd_run_job(args) -> int:
    jobs = {j["job"]: j for j in build_jobs(args.out.rstrip("/"), args)}
    j = jobs[args.job]
    print(f"run-job {args.job}: {len(j['commands'])} process(es)", flush=True)
    t0 = time.time()
    for stem, cmd in zip(j["stems"], j["commands"]):
        t1 = time.time()
        rc = subprocess.call(cmd, shell=True, cwd=str(ROOT))
        txt = Path(stem + ".txt")
        tail = [l for l in txt.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()][-1:] if txt.exists() else ["(no console)"]
        print(f"  {Path(stem).name:28s} exit {rc} {time.time() - t1:6.0f} s  {tail[0][:160]}", flush=True)
    print(f"run-job {args.job}: {(time.time() - t0) / 60:.1f} min", flush=True)
    args.expect = j["expect"]
    return cmd_check_job(args)


def cmd_check_job(args) -> int:
    bad = []
    for e in args.expect:
        p = Path(e)
        if not p.exists():
            bad.append(f"missing {e}")
        txt = Path(str(e).replace("_summary.json", "").replace(".json", "") + ".txt")
        t = txt.read_text(encoding="utf-8", errors="replace") if txt.exists() else ""
        if "device cuda" not in t:
            bad.append(f"{txt}: no 'device cuda' line")
        if "Traceback" in t:
            bad.append(f"{txt}: Traceback")
    print(f"check-job: {len(args.expect)} expected, {len(bad)} problem(s)" + ("; " + "; ".join(bad) if bad else ""))
    return 1 if bad else 0


# ==================================================================================================== bench (GPU)
def cmd_bench(args) -> int:
    """scripts/benchmark.py --sections motion,loom_escape --seeds K under the arm's OpticParams overrides (installed the
    way every probe installs them: interp_trace.install_overrides -> every OpticParams built from here on), plus a
    provenance sidecar <out>_prov.json (resolved LIFParams / OpticParams, device, cache fingerprint, source fingerprint)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import torch
    assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    arm = ARMS[args.arm]
    import interp_trace as it
    it.install_overrides({}, dict(arm["optic"]), "default", "abs", None)
    print(f"bench: arm {args.arm} ({arm['letter']}: {arm['note']}); overrides {json.dumps(arm['optic'])}; sections {BENCH_SECTIONS}; loom seed {args.seed}; "
          f"device cuda ({torch.cuda.get_device_name(0)}); torch {torch.__version__}", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    js = str(out.with_suffix(".json"))
    import benchmark
    argv0 = sys.argv
    sys.argv = ["benchmark.py", "--sections", BENCH_SECTIONS, "--seeds", str(args.seed), "--json", js]
    t0 = time.time()
    try:
        benchmark.main()
    finally:
        sys.argv = argv0
    wall = time.time() - t0
    from flyverse import brain, connectome, optic
    c = connectome.load(verbose=False)
    lif, op = brain.LIFParams(), optic.OpticParams()
    prov = common.provenance(c, lif, op, device="cuda", seeds=[args.seed], env_seeds=[args.seed], batch=1,
                             stimulus={"protocol": f"benchmark:{BENCH_SECTIONS}", "params": {"seeds": [args.seed], "arm": args.arm, "optic_overrides": arm["optic"]},
                                       "control": "the base arm's draws in the same submission"})
    prov["execution"]["device"] = "cuda"; prov["execution"]["device_name"] = torch.cuda.get_device_name(0)
    bench = json.load(open(js, encoding="utf-8")) if Path(js).exists() else None
    side = {"schema": "flyverse.object_round2_compare.bench/1", "arm": args.arm, "letter": arm["letter"], "note": arm["note"], "optic_overrides": arm["optic"], "seed": args.seed,
            "sections": BENCH_SECTIONS, "wall_s": wall, "benchmark_json": js, "checks": (bench or {}).get("checks"), "benchmark_config": (bench or {}).get("config"),
            "generator": " ".join(argv0), "provenance": prov}
    with open(str(out) + "_prov.json", "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(side), f, indent=1)
    for ch in (bench or {}).get("checks", []):
        print(f"  {ch['key']:30s} {ch['measured']!s:>12s} {ch['criterion']:>10s} {ch['status']}")
    print(f"written {js} and {out}_prov.json ({wall:.0f} s wall; device cuda)")
    return 0


# ==================================================================================================== verify (CPU)
def _optic_matches(rec: dict, arm: str) -> list[str]:
    """The resolved OpticParams of a run against the arm's overrides (every hook field: the arm's list or None)."""
    probs = []
    want = ARMS[arm]["optic"]
    for k in ("stream_rectify", "stream_adapt", "spatial_suppress", "fb_hold"):
        have = rec.get(k)
        exp = want.get(k)
        if exp is None and have not in (None, [], ()):
            probs.append(f"{k} set on arm {arm}")
        if exp is not None and json.dumps(common.to_jsonable(have)) != json.dumps(common.to_jsonable(exp)):
            probs.append(f"{k} differs from arm {arm}: {have}")
    if (arm == "fb0") != (rec.get("gain_fb") == 0):
        probs.append(f"gain_fb {rec.get('gain_fb')} on arm {arm}")
    return probs


def verify_all(out: str, log: str | None) -> dict:
    rep = {"problems": [], "log": None, "sphere": None, "spec": None, "bench": None, "expected_missing": [], "hook_liveness": {}}
    if log and Path(log).exists():
        lines = [l for l in Path(log).read_text(encoding="utf-8", errors="replace").splitlines() if "job(s)" in l and "failed" in l]
        rep["log"] = lines[-1] if lines else None
        if not lines or ", 0 failed" not in lines[-1]:
            rep["problems"].append(f"cluster log: {rep['log']!r}")
    plan_p = Path(out) / "jobs.json"
    if plan_p.exists():
        plan = json.load(open(plan_p, encoding="utf-8"))
        for j in plan["jobs"]:
            for e in j["expect"]:
                if not Path(e).exists():
                    rep["expected_missing"].append(e)
        if rep["expected_missing"]:
            rep["problems"].append(f"{len(rep['expected_missing'])} expected outputs missing")
    sph = sorted(p for p in glob.glob(f"{out}/sph/*.json") if parse_sphere_name(Path(p).stem))
    if sph:
        df, probs = pom.verify_runs(sph)
        rep["problems"] += probs
        for p in sph:
            d = json.load(open(p, encoding="utf-8")); nm = parse_sphere_name(Path(p).stem)
            rep["problems"] += [f"{p}: {x}" for x in _optic_matches(d["provenance"]["model"]["optic"], nm["arm"])]
            if bool(d["summary"]["null"]) != nm["null"]:
                rep["problems"].append(f"{p}: null flag {d['summary']['null']} vs name")
            if not nm["null"] and abs(float(d["provenance"]["stimulus"]["params"]["diam_deg"]) - nm["diam"]) > 1e-6:
                rep["problems"].append(f"{p}: diam vs name")
            if d["replicates"].get("seed") != nm["seed"]:
                rep["problems"].append(f"{p}: seed vs name")
            txt = Path(p[:-5] + ".txt")
            console = txt.read_text(encoding="utf-8", errors="replace") if txt.exists() else ""
            live = "per-stream hooks disable; using 'torch'" in console        # the optic's warp -> torch downgrade: the hook path ran
            rep["hook_liveness"].setdefault(nm["arm"], {"torch_substep_warning": 0, "runs": 0})
            rep["hook_liveness"][nm["arm"]]["runs"] += 1; rep["hook_liveness"][nm["arm"]]["torch_substep_warning"] += int(live)
            if nm["arm"] in HOOK_ARMS and not live:
                rep["problems"].append(f"{p}: hook arm without the Torch-substep warning in its console (hook not live?)")
            if nm["arm"] not in HOOK_ARMS and live:
                rep["problems"].append(f"{p}: plain arm shows the Torch-substep warning")
        rep["sphere"] = {"n_runs": int(len(df)), "el_band_spread_deg": df.attrs.get("el_band_spread_deg"), "el_band_verdict": df.attrs.get("el_band_verdict"),
                         "devices": sorted(set(df.device_name.dropna().astype(str))), "runs": df.to_dict("records")}
    d_spec = f"{out}/spec"
    if Path(d_spec).exists() and glob.glob(f"{d_spec}/*_summary.json"):
        df = pss.verify_dir(d_spec)
        rep["spec"] = {"n_runs": int(len(df)), "n_bad": int((~df.ok).sum()) if len(df) else 0, "runs": df.to_dict("records")}
        if len(df) and (~df.ok).any():
            rep["problems"].append(f"spec: {int((~df.ok).sum())} run(s) not verified: {df[~df.ok].run.tolist()}")
        for r in df.to_dict("records"):
            nm = parse_spec_name(r["run"])
            if nm is None:
                rep["problems"].append(f"spec/{r['run']}: unexpected file name"); continue
            if r["seed"] != nm["seed"]:
                rep["problems"].append(f"spec/{r['run']}: seed vs name")
            ov = json.loads(r["optic"] or "{}")
            if json.dumps(common.to_jsonable(ov), sort_keys=True) != json.dumps(common.to_jsonable(ARMS[nm["arm"]]["optic"]), sort_keys=True):
                rep["problems"].append(f"spec/{r['run']}: optic overrides {ov} vs arm {nm['arm']}")
            sm = json.load(open(f"{d_spec}/{r['run']}_summary.json", encoding="utf-8"))
            hi = sm.get("hook_info") or {}
            if (nm["arm"] in HOOK_ARMS) != bool(hi.get("active")):
                rep["problems"].append(f"spec/{r['run']}: hook_info.active {hi.get('active')} on arm {nm['arm']}")
    bench = sorted(glob.glob(f"{out}/bench/*_prov.json"))
    if bench:
        rows = []
        for p in bench:
            s = json.load(open(p, encoding="utf-8"))
            txt = Path(p.replace("_prov.json", ".txt")); console = txt.read_text(encoding="utf-8", errors="replace") if txt.exists() else ""
            ok = ("device cuda" in console) and s.get("checks") is not None and Path(s["benchmark_json"]).exists()
            if not ok:
                rep["problems"].append(f"{p}: console / checks / benchmark json incomplete")
            if json.dumps(common.to_jsonable(s["provenance"]["model"]["optic"].get("stream_rectify"))) != json.dumps(common.to_jsonable(ARMS[s["arm"]]["optic"].get("stream_rectify"))):
                rep["problems"].append(f"{p}: resolved stream_rectify differs from arm {s['arm']}")
            rows.append({"file": p, "arm": s["arm"], "seed": s["seed"], "ok": ok, "device": s["provenance"]["execution"].get("device_name"), "wall_s": s.get("wall_s"),
                         **{ch["key"]: ch["status"] for ch in (s.get("checks") or [])}})
        rep["bench"] = {"n_runs": len(rows), "runs": rows}
    return rep


def cmd_verify(args) -> int:
    rep = verify_all(args.out.rstrip("/"), args.log)
    print(f"cluster log: {rep['log']}")
    if rep["sphere"]:
        print(f"sphere: {rep['sphere']['n_runs']} runs on {rep['sphere']['devices']}; centroid-elevation spread {rep['sphere']['el_band_spread_deg']} deg -> {rep['sphere']['el_band_verdict']}")
        print(f"hook liveness (Torch-substep warning per arm): {rep['hook_liveness']}")
    if rep["spec"]:
        print(f"spec: {rep['spec']['n_runs']} runs, {rep['spec']['n_bad']} not verified")
    if rep["bench"]:
        print(f"bench: {rep['bench']['n_runs']} runs")
    print(f"expected outputs missing: {len(rep['expected_missing'])}")
    print("problems: " + ("; ".join(rep["problems"][:40]) if rep["problems"] else "none") + (f" ... ({len(rep['problems'])} total)" if len(rep["problems"]) > 40 else ""))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(common.to_jsonable(dict(rep, generator=" ".join(sys.argv))), f, indent=1)
        print(f"written {args.json}")
    return 1 if rep["problems"] else 0


# ==================================================================================================== windows
DEFAULT_MAPS = ["out/objr2/rfmap_ship.csv", "out/objr2/rfmap_fb0.csv", "out/synth2/rfmap_150_fb0_p3.csv"]


def load_windows(paths: list[str]) -> tuple[pd.DataFrame, dict]:
    """The baseline's window rule on an ordered list of RF-map CSVs (first map wins per body, then the anatomical column
    with the type's median fitted width over all maps, else whole)."""
    maps = []
    for p in paths:
        if Path(p).exists():
            m = pd.read_csv(p, dtype={"bodyId": str}); m["fitted"] = m["fitted"].astype(str).str.lower().isin(("true", "1", "1.0")); maps.append((p, m))
    if not maps:
        raise SystemExit(f"no RF map found among {paths}")
    base = maps[0][1]
    widths = {}
    for t in sorted(set(base.type)):
        w = pd.concat([m[(m.type == t) & m.fitted].width_deg for _, m in maps])
        widths[t] = float(w.median()) if len(w) else 15.0
    rows = []
    for _, r in base.iterrows():
        b = r.bodyId; t = r.type; done = False
        for p, m in maps:
            q = m[(m.bodyId == b) & m.fitted]
            if len(q):
                q = q.iloc[0]; rows.append({"bodyId": b, "type": t, "window_source": "rf:" + Path(p).stem, "az_deg": float(q.az_deg), "el_deg": float(q.el_deg),
                                            "width_deg": float(q.width_deg), "height_deg": float(q.width_deg)}); done = True; break
        if done:
            continue
        if np.isfinite(r.get("anat_az_deg", np.nan)):
            rows.append({"bodyId": b, "type": t, "window_source": "anat", "az_deg": float(r.anat_az_deg), "el_deg": float(r.anat_el_deg), "width_deg": widths[t], "height_deg": widths[t]})
        else:
            rows.append({"bodyId": b, "type": t, "window_source": "whole", "az_deg": np.nan, "el_deg": np.nan, "width_deg": np.nan, "height_deg": np.nan})
    win = pd.DataFrame(rows).drop_duplicates("bodyId").reset_index(drop=True)
    summary = {"maps": [p for p, _ in maps], "box_width_by_type_deg": widths,
               "sources": {t: win[win.type == t].window_source.value_counts().to_dict() for t in sorted(set(win.type))},
               "fitted_bodies": {p: {t: int(((m.type == t) & m.fitted).sum()) for t in sorted(set(m.type))} for p, m in maps}}
    return win, summary


# ==================================================================================================== the sphere analysis
def _holm_family(rows: list[dict]) -> list[dict]:
    ph = orb.holm([c["p"] for c in rows])
    for c, h in zip(rows, ph):
        c["p_holm"] = float(h); c["survives_holm"] = bool(np.isfinite(h) and h <= ALPHA and c["verdict"] == "result")
    return rows


def analyse_sphere(out: str, win: pd.DataFrame) -> dict:
    files = sorted(p for p in glob.glob(f"{out}/sph/*.json") if parse_sphere_name(Path(p).stem))
    if not files:
        return {}
    per_body_rows, per_run_rows, up_rows, lvl_rows, fp_rows, tc_rows = [], [], [], [], [], []
    prov0 = None; az_edges = None
    for p in files:
        nm = parse_sphere_name(Path(p).stem); d = json.load(open(p, encoding="utf-8")); z = np.load(p[:-5] + ".npz", allow_pickle=False)
        if prov0 is None and nm["arm"] == "base":
            prov0 = d["provenance"]
        sm = d["summary"]; pt = {r["type"]: r for r in d["tables"]["per_type"]}
        stem = Path(p).stem
        for dd in (DIAMS if nm["null"] else [nm["diam"]]):
            zero = np.zeros_like(z["a__optic_dr"])
            pb_lc = orb.windowed_body_stats(z["a__lc_drive_mv"], z["b__lc_drive_mv"], z["a__lc_spikes"], z["b__lc_spikes"], z["lc_body_ids"], z["lc_types"], win, z["track_az_deg"], z["track_el_deg"], dd, dd, LC_TYPES)
            pb_up = orb.windowed_body_stats(z["a__optic_dr"], z["b__optic_dr"], zero, zero, z["rate_body_ids"], z["rate_types"], win, z["track_az_deg"], z["track_el_deg"], dd, dd, UPSTREAM)
            pb = pd.concat([pb_lc, pb_up], ignore_index=True)
            pb.insert(0, "run", stem); pb.insert(1, "arm", nm["arm"]); pb.insert(2, "null", bool(nm["null"])); pb.insert(3, "diam_deg", dd); pb.insert(4, "seed", nm["seed"])
            per_body_rows.append(pb)
            for t, v in orb.pop_stats(pb).items():
                row = dict(run=stem, arm=nm["arm"], null=bool(nm["null"]), diam_deg=dd, seed=nm["seed"], type=t, **v, device=d["provenance"]["execution"].get("device_name"))
                if t in LC_TYPES:
                    row.update(diff_max_over_cells_mean_mv=sm["diff_max_over_cells_mean_mv"][t], diff_rate_hz_max_cell=sm["diff_rate_hz_max_cell"][t])
                else:   # upstream: the windowed median of |A - B| over fitted bodies beside the signed one
                    g = pb[(pb.type == t) & (pb.n_frames_win > 0)]
                    row.update(abs_drive_median=float(g.drive_diff_win.abs().median()) if len(g) else np.nan)
                per_run_rows.append(row)
        for t in UPSTREAM:
            up_rows.append(dict(run=stem, arm=nm["arm"], null=bool(nm["null"]), diam_deg=(np.nan if nm["null"] else nm["diam"]), seed=nm["seed"], type=t,
                                diff_signed_best_cell=sm["diff_signed_best_cell"][t], diff_signed_mean=sm["diff_signed_mean"][t], diff_abs_best_cell_mean=sm["diff_abs_best_cell_mean"][t]))
        lvl = dict(run=stem, arm=nm["arm"], null=bool(nm["null"]), diam_deg=(np.nan if nm["null"] else nm["diam"]), seed=nm["seed"])
        for t in UPSTREAM:
            lvl.update({f"{t}.dev_mean_b": pt[t]["dev_mean_b"], f"{t}.dev_abs_mean_b": pt[t]["dev_abs_mean_b"], f"{t}.dev_mean_a": pt[t]["dev_mean"], f"{t}.dev_abs_mean_a": pt[t]["dev_abs_mean"]})
        for t in LC_TYPES:
            lvl.update({f"{t}.drive_mean_mv_b": pt[t]["drive_mean_mv_b"], f"{t}.rate_hz_mean_b": pt[t]["rate_hz_mean_b"], f"{t}.drive_mean_mv_a": pt[t]["drive_mean_mv"],
                        f"{t}.rate_hz_mean_a": pt[t]["rate_hz_mean"], f"{t}.cells_firing_a": pt[t]["cells_firing"], f"{t}.drive_best_cell_mean_mv_a": pt[t]["drive_best_cell_mean_mv"]})
        lvl_rows.append(lvl)
        fp = sm["footprint"]; cr = orb.contrast_readout(z["rad_a"], z["rad_b"]) if not nm["null"] else {}
        fp_rows.append(dict(run=stem, arm=nm["arm"], null=bool(nm["null"]), diam_deg=(np.nan if nm["null"] else nm["diam"]), seed=nm["seed"],
                            n_dimmed_50pct_mean=fp["n_dimmed_50pct_mean"], n_changed_5pct_mean=fp["n_changed_5pct_mean"], centroid_el_mean_deg=fp["centroid_el_mean_deg"],
                            band_lo=fp["dimmed_el_band_deg"][0], band_hi=fp["dimmed_el_band_deg"][1], blank_lum=fp["blank_lum_under_object_mean"],
                            realised_el_maxdev=sm["track_check"]["realised_el_deg_maxdev"], realised_diam_maxdev=sm["track_check"]["realised_diam_deg_maxdev"],
                            speed_min=sm["track_check"]["speed_deg_s_min"], speed_max=sm["track_check"]["speed_deg_s_max"], wall_a_s=sm["wall_s"]["a"], wall_b_s=sm["wall_s"]["b"], **cr))
        tun = z["a__lc_drive_by_az_bin"] - z["b__lc_drive_by_az_bin"]; az_edges = z["az_bin_edges_deg"]
        for t in LC_TYPES:
            mi = np.flatnonzero(z["lc_types"] == t)
            tc_rows.append({"run": stem, "arm": nm["arm"], "null": bool(nm["null"]), "diam_deg": (np.nan if nm["null"] else nm["diam"]), "type": t, "pop_mean": tun[:, mi].mean(1).astype(np.float32),
                            "top_body": str(int(z["lc_body_ids"][mi[int(np.argmax(np.abs(tun[:, mi]).max(0)))]])), "top_tc": tun[:, mi[int(np.argmax(np.abs(tun[:, mi]).max(0)))]].astype(np.float32)})
    per_body = pd.concat(per_body_rows, ignore_index=True); per_run = pd.DataFrame(per_run_rows); up = pd.DataFrame(up_rows); lvl = pd.DataFrame(lvl_rows); fp_df = pd.DataFrame(fp_rows)
    arms_present = [a for a in ARMS if (per_run.arm == a).any()]
    # ---------------------------------------------------------------- comparisons: per arm, two questions, Holm within the small-rung family
    comps, pref = [], []
    base_obj_run = per_run[(per_run.arm == "base") & ~per_run.null]; base_obj_up = up[(up.arm == "base") & ~up.null]

    def fam(objs, nulls, base_objs, t, q, arm, role, rungs, key="diam_deg"):
        rows_null, rows_base = [], []
        for dd in rungs:
            s = objs[objs[key] == dd][q].to_numpy()
            n = nulls[(nulls[key] == dd)][q].to_numpy() if key in nulls.columns and (nulls[key] == dd).any() else nulls[q].to_numpy()
            rows_null.append(orb.compare_row(s, n, {"family": f"{role}:{arm}:{t}:{q}:vs_null", "arm": arm, "type": t, "diam_deg": dd, "statistic": q, "against": "null", "role": role}))
            b = base_objs[base_objs[key] == dd][q].to_numpy()
            rows_base.append(orb.compare_row(s, b, {"family": f"{role}:{arm}:{t}:{q}:vs_base", "arm": arm, "type": t, "diam_deg": dd, "statistic": q, "against": "base", "role": role}))
        return _holm_family(rows_null) + _holm_family(rows_base)

    for arm in arms_present:
        objs = per_run[(per_run.arm == arm) & ~per_run.null]; nulls = per_run[(per_run.arm == arm) & per_run.null]
        for t in LC_TYPES:
            o, n, b = objs[objs.type == t], nulls[nulls.type == t], base_obj_run[base_obj_run.type == t]
            comps += fam(o, n, b, t, "drive_median", arm, "P2_lc11" if t == "LC11" else "lc10a", SMALL_RUNGS)
            comps += fam(o, n, b, t, "drive_median", arm, "large_rungs", LARGE_RUNGS)
            for q in ("spikes_median", "drive_mean", "drive_max", "diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell"):
                comps += fam(o, n, b, t, q, arm, "lc_exploratory", DIAMS)
            for q in ("drive_median", "diff_max_over_cells_mean_mv"):
                sp = orb.spearman_perm(o.diam_deg.to_numpy(), o[q].to_numpy()); ct = orb.contrast_perm(o[o.diam_deg.isin(SMALL_RUNGS)][q], o[o.diam_deg.isin(LARGE_RUNGS)][q])
                pref.append({"arm": arm, "type": t, "statistic": q, "spearman_rho": sp["rho"], "spearman_p_perm": sp["p_perm"], "n_runs": sp["n"],
                             "contrast_small_minus_large": ct["contrast"], "contrast_p_perm": ct["p_perm"],
                             "excess_by_rung": {str(dd): float(o[o.diam_deg == dd][q].mean() - n[n.diam_deg == dd][q].mean()) for dd in DIAMS}})
        for t in UPSTREAM:
            o, n, b = up[(up.arm == arm) & ~up.null & (up.type == t)], up[(up.arm == arm) & up.null & (up.type == t)], base_obj_up[base_obj_up.type == t]
            role = "P1_carrier" if t == "T3" else ("T2_carrier" if t == "T2" else "upstream_exploratory")
            comps += fam(o, n, b, t, "diff_signed_best_cell", arm, role, SMALL_RUNGS)
            comps += fam(o, n, b, t, "diff_signed_best_cell", arm, "large_rungs", LARGE_RUNGS)
            for q in ("diff_signed_mean", "diff_abs_best_cell_mean"):
                comps += fam(o, n, b, t, q, arm, "upstream_exploratory", DIAMS)
            ow, nw, bw = objs[objs.type == t], nulls[nulls.type == t], base_obj_run[base_obj_run.type == t]
            if len(ow):
                comps += fam(ow, nw, bw, t, "abs_drive_median", arm, "upstream_windowed", DIAMS)
    # ---------------------------------------------------------------- per-arm answers
    cdf = pd.DataFrame(comps)
    answers = {}
    for arm in arms_present:
        a = {"letter": ARMS[arm]["letter"]}
        for t, role in (("T3", "P1_carrier"), ("T2", "T2_carrier"), ("LC11", "P2_lc11")):
            g = cdf[(cdf.arm == arm) & (cdf.role == role)]
            both = []
            for dd in SMALL_RUNGS:
                gn = g[(g.against == "null") & (g.diam_deg == dd)]; gb = g[(g.against == "base") & (g.diam_deg == dd)]
                okn = bool(len(gn) and gn.survives_holm.iloc[0]) or (arm == "fb0" and bool(len(gn)) and gn.verdict.iloc[0] == "undetermined" and gn["diff"].iloc[0] > 0)
                okb = bool(len(gb) and gb.survives_holm.iloc[0] and gb["diff"].iloc[0] > 0)
                both.append({"diam_deg": dd, "vs_null_verdict": gn.verdict.iloc[0] if len(gn) else None, "vs_null_p_holm": float(gn.p_holm.iloc[0]) if len(gn) else None, "vs_null_z": float(gn.z.iloc[0]) if len(gn) else None,
                             "vs_base_verdict": gb.verdict.iloc[0] if len(gb) else None, "vs_base_p_holm": float(gb.p_holm.iloc[0]) if len(gb) else None, "vs_base_z": float(gb.z.iloc[0]) if len(gb) else None,
                             "arm_mean": float(gn.stim_mean.iloc[0]) if len(gn) else None, "arm_sd": float(gn.stim_sd.iloc[0]) if len(gn) else None,
                             "null_mean": float(gn.null_mean.iloc[0]) if len(gn) else None, "base_mean": float(gb.null_mean.iloc[0]) if len(gb) else None,
                             "ratio_to_base": float(gb.stim_mean.iloc[0] / gb.null_mean.iloc[0]) if len(gb) and gb.null_mean.iloc[0] not in (0, None) else None,
                             "called": okn and okb})
            a[t] = {"rungs": both, "called": any(r["called"] for r in both), "note": "undetermined vs a deterministic null (fb0) counts as a figure when diff > 0; vs base is always a 5 v 5 call" if arm == "fb0" else ""}
        a["carries_small_field"] = bool(a["T3"]["called"] and a["T2"]["called"]); a["carries_T3_only"] = bool(a["T3"]["called"] and not a["T2"]["called"]); a["lc11_follows"] = bool(a["LC11"]["called"])
        answers[arm] = a
    # ---------------------------------------------------------------- time courses (population mean over runs per arm x rung) and footprint
    tc = pd.DataFrame(tc_rows); tc_out = []
    for arm in arms_present:
        for t in LC_TYPES:
            for dd in DIAMS + [None]:
                g = tc[(tc.arm == arm) & (tc.type == t) & (tc.null if dd is None else (tc.diam_deg == dd))]
                if not len(g):
                    continue
                M = np.stack(g.pop_mean.to_numpy())
                for k in range(M.shape[1]):
                    tc_out.append({"arm": arm, "type": t, "diam_deg": dd, "arm_kind": "null" if dd is None else "object", "az_bin": k, "az_centre_deg": float((az_edges[k] + az_edges[k + 1]) / 2),
                                   "pop_mean_diff_mv": float(M[:, k].mean()), "sd_runs_mv": float(M[:, k].std(ddof=1)) if len(M) > 1 else np.nan, "n_runs": int(len(M))})
    fps = fp_df[~fp_df.null].groupby(["arm", "diam_deg"]).agg(n_runs=("run", "size"), n_dimmed_50pct=("n_dimmed_50pct_mean", "mean"), n_changed_5pct=("n_changed_5pct_mean", "mean"),
                                                            centroid_el=("centroid_el_mean_deg", "mean"), band_lo=("band_lo", "min"), band_hi=("band_hi", "max"),
                                                            extreme_rel_change_median=("extreme_rel_change_median", "mean"), mean_rel_change_over_changed_set=("mean_rel_change_over_changed_set", "mean"),
                                                            realised_el_maxdev=("realised_el_maxdev", "max"), speed_min=("speed_min", "min"), speed_max=("speed_max", "max"),
                                                            wall_a_s=("wall_a_s", "mean")).reset_index()
    levels = lvl.groupby("arm").agg(**{c: (c, "mean") for c in lvl.columns if "." in c}, n_runs=("run", "size")).reset_index()
    levels_sd = lvl.groupby("arm").agg(**{c + "_sd": (c, "std") for c in lvl.columns if "." in c}).reset_index()
    return {"per_body": per_body, "per_run": per_run, "upstream_runs": up, "levels": levels.merge(levels_sd, on="arm"), "levels_runs": lvl, "comparisons": comps, "preference": pref,
            "answers": answers, "time_course": pd.DataFrame(tc_out), "footprint": fps, "footprint_runs": fp_df, "prov0": prov0, "arms": arms_present,
            "n_runs": {f"{a}:{'null' if n else 'obj'}": int(((per_run.arm == a) & (per_run.null == n)).sum() / len(LC_TYPES + UPSTREAM)) for a in arms_present for n in (False, True)}}


# ==================================================================================================== the specificity analysis
SPEC_LC = ("diff_max_over_cells_mean_mv", "diff_mean_over_cells_mean_mv", "diff_rate_hz_max_cell")
SPEC_UP = ("diff_signed_best_cell", "diff_abs_best_cell_mean", "diff_signed_mean")


def analyse_spec(out: str) -> dict:
    files = sorted(glob.glob(f"{out}/spec/*_summary.json"))
    rows = []; prov0 = None
    for sf in files:
        stem = sf[:-len("_summary.json")]; nm = parse_spec_name(Path(stem).name)
        if nm is None:
            continue
        sm = json.load(open(sf, encoding="utf-8"))
        a_name, b_name = ("blank_a", "blank_b") if nm["null"] else ("stim", "blank")
        A = common.Recording.load(Path(stem + f"_{a_name}.npz")); B = common.Recording.load(Path(stem + f"_{b_name}.npz"))
        if prov0 is None and os.path.exists(stem + "_prov.json"):
            prov0 = json.load(open(stem + "_prov.json", encoding="utf-8"))
        fs = pss.family_stats(A, B)
        row = dict(run=Path(stem).name, arm=nm["arm"], stim=nm["stim"], null=nm["null"], seed=nm["seed"], device=sm.get("device"), hook_active=bool((sm.get("hook_info") or {}).get("active")),
                   wall_s=sum((sm.get("wall_s") or {}).values()))
        for t in LC_TYPES:
            for q in SPEC_LC:
                row[f"{t}.{q}"] = fs.get(t, {}).get(q, np.nan)
            row[f"{t}.rate_hz_mean_blank"] = fs.get(t, {}).get("rate_hz_mean_blank", np.nan); row[f"{t}.drive_mean_blank_mv"] = fs.get(t, {}).get("drive_mean_blank_mv", np.nan)
        for t in UPSTREAM:
            for q in SPEC_UP:
                row[f"{t}.{q}"] = fs.get(t, {}).get(q, np.nan)
            row[f"{t}.dev_mean_blank"] = fs.get(t, {}).get("dev_mean_blank", np.nan)
        rows.append(row)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    arms_present = [a for a in ARMS if (df.arm == a).any()]
    comps = []
    stats = [(t, q) for t in LC_TYPES for q in SPEC_LC] + [(t, q) for t in UPSTREAM for q in SPEC_UP]
    for arm in arms_present:
        nulls = df[(df.arm == arm) & df.null]
        for st in SPEC:
            if st == "null":
                continue
            o = df[(df.arm == arm) & (df.stim == st)]; b = df[(df.arm == "base") & (df.stim == st)]
            for t, q in stats:
                col = f"{t}.{q}"
                comps.append(orb.compare_row(o[col].to_numpy(), nulls[col].to_numpy(), {"family": f"spec:{arm}:{st}:{t}:{q}", "arm": arm, "stim": st, "type": t, "statistic": q, "against": "null", "role": "specificity"}))
                comps.append(orb.compare_row(o[col].to_numpy(), b[col].to_numpy(), {"family": f"spec:{arm}:{st}:{t}:{q}", "arm": arm, "stim": st, "type": t, "statistic": q, "against": "base", "role": "specificity"}))
    cdf = pd.DataFrame(comps)
    answers = {}
    for arm in arms_present:
        a = {}
        rel = []
        for st in ("bar", "grating", "flicker"):
            for q in ("diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell"):
                g = cdf[(cdf.arm == arm) & (cdf.stim == st) & (cdf.type == "LC11") & (cdf.statistic == q) & (cdf.against == "base")]
                if len(g):
                    r = g.iloc[0]
                    rel.append({"stim": st, "statistic": q, "verdict": r.verdict, "z": float(r.z), "p": float(r.p), "arm_mean": float(r.stim_mean), "arm_sd": float(r.stim_sd), "base_mean": float(r.null_mean), "base_sd": float(r.null_sd),
                                "released": bool(r.verdict == "result" and r["diff"] > 0)})
        a["lc11_release_rows"] = rel; a["releases_bar_grating_flicker"] = any(r["released"] for r in rel)
        onoff = {}
        for t in HOOKED:
            e = {}
            for st in ("flashon", "flashoff"):
                col = f"{t}.diff_signed_best_cell"
                o = df[(df.arm == arm) & (df.stim == st)][col]; b = df[(df.arm == "base") & (df.stim == st)][col]; n = df[(df.arm == arm) & df.null][col]
                e[st] = {"arm_mean": float(o.mean()) if len(o) else None, "arm_sd": float(o.std(ddof=1)) if len(o) > 1 else None, "arm_values": o.tolist(), "base_mean": float(b.mean()) if len(b) else None,
                         "null_mean": float(n.mean()) if len(n) else None, "ratio_to_base": float(o.mean() / b.mean()) if len(o) and len(b) and b.mean() != 0 else None}
            e["keeps_both"] = bool(all(e[st]["ratio_to_base"] is not None and e[st]["ratio_to_base"] >= 0.5 for st in ("flashon", "flashoff")))
            onoff[t] = e
        a["on_off"] = onoff; a["T2_T3_keep_on_and_off"] = bool(all(onoff[t]["keeps_both"] for t in HOOKED))
        bd = {}
        for t in LC_TYPES:
            col = f"{t}.diff_max_over_cells_mean_mv"
            br = df[(df.arm == arm) & (df.stim == "bright110")][col]; dk = df[(df.arm == arm) & (df.stim == "dark110")][col]; n = df[(df.arm == arm) & df.null][col]
            bd[t] = {"bright_mean": float(br.mean()) if len(br) else None, "bright_sd": float(br.std(ddof=1)) if len(br) > 1 else None, "dark_mean": float(dk.mean()) if len(dk) else None,
                     "dark_sd": float(dk.std(ddof=1)) if len(dk) > 1 else None, "null_mean": float(n.mean()) if len(n) else None, "bright_over_dark": float(br.mean() / dk.mean()) if len(br) and len(dk) and dk.mean() != 0 else None}
        a["bright_dark"] = bd
        answers[arm] = a
    return {"per_run": df, "comparisons": comps, "answers": answers, "prov0": prov0, "arms": arms_present,
            "n_runs": {f"{a}:{st}": int(((df.arm == a) & (df.stim == st)).sum()) for a in arms_present for st in SPEC}}


# ==================================================================================================== the transfer analysis
def analyse_bench(out: str) -> dict:
    files = sorted(glob.glob(f"{out}/bench/*_prov.json"))
    rows = []
    for p in files:
        s = json.load(open(p, encoding="utf-8"))
        bj = json.load(open(s["benchmark_json"], encoding="utf-8")) if Path(s["benchmark_json"]).exists() else {}
        row = {"arm": s["arm"], "seed": s["seed"], "file": p, "device": s["provenance"]["execution"].get("device_name"), "wall_s": s.get("wall_s"),
               "runtime": bj.get("runtime_s"), "errors": {k: v.get("error") for k, v in (bj.get("sections") or {}).items() if isinstance(v, dict) and v.get("error")}}
        for ch in s.get("checks") or []:
            row[ch["key"]] = ch["measured"]; row[ch["key"] + ".status"] = ch["status"]
        sec = bj.get("sections") or {}
        if "motion" in sec and isinstance(sec["motion"], dict) and "subtypes" in sec["motion"]:
            for t, v in sec["motion"]["subtypes"].items():
                row[f"motion.{t}.dsi"] = v["dsi"]; row[f"motion.{t}.correct"] = v["correct"]
        if "loom_escape" in sec and isinstance(sec["loom_escape"], dict) and "seeds" in sec["loom_escape"]:
            for sd, v in sec["loom_escape"]["seeds"].items():
                row["loom.GF_walk_max_hz"] = v["GF_walk_max_hz"]; row["loom.hops_before_loom"] = v["hops_before_loom"]; row["loom.escape"] = v["escape"]
        rows.append(row)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    arms_present = [a for a in ARMS if (df.arm == a).any()]
    comps, summ = [], {}
    base = df[df.arm == "base"]
    for arm in arms_present:
        g = df[df.arm == arm]
        s = {"n_draws": int(len(g)), "errors": [e for e in g.errors if e]}
        for k in BENCH_KEYS:
            if k in g.columns:
                v = pd.to_numeric(g[k], errors="coerce")
                s[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "values": v.tolist(), "status": g[k + ".status"].tolist() if k + ".status" in g.columns else None}
                if arm != "base" and k in base.columns:
                    comps.append(orb.compare_row(v.to_numpy(), pd.to_numeric(base[k], errors="coerce").to_numpy(), {"family": f"bench:{arm}:{k}", "arm": arm, "statistic": k, "against": "base", "role": "transfer"}))
        costs = []
        for k in BENCH_KEYS:
            if k + ".status" in g.columns and k + ".status" in base.columns:
                base_all_pass = bool(len(base)) and all(str(x).startswith("PASS") for x in base[k + ".status"])
                arm_any_fail = any(str(x) == "FAIL" for x in g[k + ".status"])
                if base_all_pass and arm_any_fail:
                    costs.append(k)
        s["costs_sections"] = costs
        summ[arm] = s
    return {"per_run": df, "comparisons": comps, "summary": summ, "arms": arms_present}


# ==================================================================================================== analyse
def _fmt(c: dict) -> str:
    key = c.get("diam_deg", c.get("stim", c.get("statistic")))
    return (f"{c['arm']:10s} {c['type'] if 'type' in c else '':6s} {str(key):>9s} {c['statistic']:28s} vs {c['against']:4s} arm {c['stim_mean']:+.4f} +- {c['stim_sd']:.4f} (n {c['stim_n']}) "
            f"ref {c['null_mean']:+.4f} +- {c['null_sd']:.4f} (n {c['null_n']}) z {c['z']:+.2f} p {c['p']:.4f} (ties {c['n_tied']}) p_holm {c.get('p_holm', float('nan')):.4f} -> {c['verdict']}"
            f"{' SURVIVES HOLM' if c.get('survives_holm') else ''}")


def cmd_analyse(args) -> int:
    out = args.out.rstrip("/")
    rep = None
    if not args.no_verify:
        rep = verify_all(out, args.log)
        print(f"verify: log {rep['log']!r}; problems: {len(rep['problems'])} {rep['problems'][:8]}")
        if rep["problems"] and not args.force:
            sys.exit("batch verification failed (pass --force to analyse anyway; the Result then records the problems)")
    win, win_summary = load_windows(args.rf_maps)
    print(f"windows: maps {win_summary['maps']}; sources {win_summary['sources']}; box widths {win_summary['box_width_by_type_deg']}")
    sph = analyse_sphere(out, win) if not args.skip_sphere else {}
    spec = analyse_spec(out) if not args.skip_spec else {}
    bench = analyse_bench(out) if not args.skip_bench else {}
    prov = (sph or spec).get("prov0")
    if prov is None and bench:
        prov = json.load(open(bench["per_run"].file.iloc[0], encoding="utf-8"))["provenance"]
    if prov is None:
        prov = common.provenance(__import__("flyverse.connectome", fromlist=["load"]).load(verbose=False))
    res = common.Result.new("lesion", prov)
    res.run_id = "objr2c-compare"; res.tool_version = "object_round2_compare/1"
    res.replicates = {"n": {"sphere": N_RUNS, "spec": N_SPEC, "bench": N_BENCH}, "unit": "runs", "null": "each arm's own blank/blank runs of the same submission; the base arm's object runs for the model comparison",
                      "n_runs": {"sphere": (sph or {}).get("n_runs"), "spec": (spec or {}).get("n_runs"), "bench": {a: v["n_draws"] for a, v in (bench or {}).get("summary", {}).items()}},
                      "batch": {"jobs_json": f"{out}/jobs.json", "cluster_log": args.log}}
    Path(out).mkdir(parents=True, exist_ok=True)
    if sph:
        sph["per_body"].to_csv(f"{out}/per_body_sphere.csv", index=False)
        res.add_table("sphere_per_run", sph["per_run"]); res.add_table("sphere_upstream_runs", sph["upstream_runs"]); res.add_table("sphere_comparisons", sph["comparisons"])
        res.add_table("sphere_preference", sph["preference"]); res.add_table("sphere_time_course", sph["time_course"]); res.add_table("sphere_footprint", sph["footprint"])
        res.add_table("sphere_footprint_runs", sph["footprint_runs"]); res.add_table("levels", sph["levels"]); res.add_table("levels_runs", sph["levels_runs"])
    if spec:
        res.add_table("spec_per_run", spec["per_run"]); res.add_table("spec_comparisons", spec["comparisons"])
    if bench:
        res.add_table("bench_per_run", bench["per_run"]); res.add_table("bench_comparisons", bench["comparisons"])
    res.add_table("windows", win)
    answers = {"sphere": (sph or {}).get("answers"), "specificity": (spec or {}).get("answers"), "transfer": (bench or {}).get("summary")}
    # the one-line classification per arm
    cls = {}
    for arm in (sph or {}).get("arms", []):
        a = sph["answers"][arm]; s = (spec or {}).get("answers", {}).get(arm, {}); b = (bench or {}).get("summary", {}).get(arm, {})
        cls[arm] = {"letter": ARMS[arm]["letter"], "carries_small_field": a["carries_small_field"], "carries_T3_only": a["carries_T3_only"], "lc11_follows": a["lc11_follows"],
                    "releases_bar_grating_flicker": s.get("releases_bar_grating_flicker"), "T2_T3_keep_on_and_off": s.get("T2_T3_keep_on_and_off"),
                    "costs_benchmark_sections": b.get("costs_sections"), "adopted": False}
    res.summary = {"predeclared": PREDECLARED, "arms": ARMS, "windows": win_summary, "answers": answers, "classification": cls, "verify": rep,
                   "reading": "verdicts are common.compare's (z on the null SD, exact U, p_floor); p is the tie-aware exact permutation U; p_holm is Holm within the named family; only a `result` with p_holm <= 0.05 is called; nothing is adopted"}
    res.validation = {"name": "fixed-anatomy model comparison on the matched sphere ladder: predeclared P1 (T3 / T2 carrier figure) and P2 (LC11 windowed drive) per arm, the specificity battery, the benchmark transfer",
                      "reference": {"P1": "docs/audits/deficit_object.md 4.3: the held-carrier arms gave T3 0.062 / 0.072 +- 0.002 against 0.024 +- 0.005 (scale reference, another protocol)",
                                    "specificity": "Keles et al. 2020: no release of bar / grating responses; T2 / T3 respond to both ON and OFF"},
                      "measured": cls, "status": "measured", "source": "scripts/object_round2_compare.py analyse"}
    res.files = {"generator": " ".join(sys.argv), "per_body_sphere": f"{out}/per_body_sphere.csv" if sph else None, "predeclared": f"{out}/predeclared.json", "jobs": f"{out}/jobs.json",
                 "cluster_log": args.log, "rf_maps": win_summary["maps"]}
    p = res.save(args.json)
    # ---------------------------------------------------------------- console
    if sph:
        print("\n== SPHERE: footprint per arm x rung (captured radiance; mean over runs)")
        common.print_table(sph["footprint"], floatfmt="{:.3f}", max_rows=200)
        print("\n== SPHERE: blank-arm levels per arm (mean over all runs of the arm)")
        common.print_table(sph["levels"][["arm", "n_runs"] + [c for c in sph["levels"].columns if c.endswith("_b")]], floatfmt="{:+.4f}", max_rows=20)
        for role, title in (("P1_carrier", "P1: T3 diff_signed_best_cell, small rungs, Holm within 3 (vs null; vs base)"), ("T2_carrier", "T2 diff_signed_best_cell, small rungs"),
                            ("P2_lc11", "P2: LC11 windowed drive_median, small rungs"), ("lc10a", "LC10a windowed drive_median, small rungs"), ("large_rungs", "large rungs (exploratory)")):
            print(f"\n== SPHERE: {title}")
            for c in sph["comparisons"]:
                if c["role"] == role:
                    print("  " + _fmt(c))
        print("\n== SPHERE: preference tests per arm")
        common.print_table(pd.DataFrame([{k: v for k, v in p_.items() if k != "excess_by_rung"} for p_ in sph["preference"]]), floatfmt="{:+.4f}", max_rows=100)
    if spec:
        print("\n== SPECIFICITY: LC11 / T3 per arm x stimulus (vs base)")
        for c in spec["comparisons"]:
            if c["against"] == "base" and c["type"] in ("LC11", "T3") and c["statistic"] in ("diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell", "diff_signed_best_cell"):
                print("  " + _fmt(c))
    if bench:
        print("\n== TRANSFER: benchmark motion / loom_escape per arm")
        print(json.dumps(common.to_jsonable(bench["summary"]), indent=1)[:6000])
    print("\n== CLASSIFICATION"); print(json.dumps(common.to_jsonable(cls), indent=1))
    print("\n== ANSWERS (sphere)"); print(json.dumps(common.to_jsonable(answers["sphere"]), indent=1)[:12000])
    print(f"\nproblems: {res.check() or 'none'}; written {p}")
    return 0


def cmd_streams(args) -> int:
    for a, v in ARMS.items():
        print(f"{v['letter']:3s} {a:11s} {v['note']}")
        for k, val in v["optic"].items():
            print(f"      --optic {k}={json.dumps(val)}")
    print(f"\nflash position {FLASH}")
    return 0


# ==================================================================================================== main
def _common_plan_args(p):
    p.add_argument("--runs", type=int, default=N_RUNS); p.add_argument("--spec-runs", type=int, default=N_SPEC); p.add_argument("--bench-draws", type=int, default=N_BENCH)
    p.add_argument("--seconds", type=float, default=12.0); p.add_argument("--settle", type=float, default=3.0)
    p.add_argument("--spec-seconds", type=float, default=6.0); p.add_argument("--spec-settle", type=float, default=2.0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", default="out/objr2c"); p.add_argument("--name", default="objr2c"); p.add_argument("--minutes", type=int, default=240)
    _common_plan_args(p); p.set_defaults(func=cmd_plan)
    s = sub.add_parser("submit"); s.add_argument("--out", default="out/objr2c"); s.set_defaults(func=cmd_submit)
    rj = sub.add_parser("run-job"); rj.add_argument("--job", required=True); rj.add_argument("--out", default="out/objr2c"); _common_plan_args(rj); rj.set_defaults(func=cmd_run_job)
    c = sub.add_parser("check-job"); c.add_argument("--expect", nargs="+", required=True); c.set_defaults(func=cmd_check_job)
    b = sub.add_parser("bench"); b.add_argument("--arm", required=True, choices=list(ARMS)); b.add_argument("--seed", type=int, default=0); b.add_argument("--out", required=True); b.set_defaults(func=cmd_bench)
    v = sub.add_parser("verify"); v.add_argument("--out", default="out/objr2c"); v.add_argument("--log", default="out/objr2c_cluster.log"); v.add_argument("--json", default="out/objr2c/verify.json"); v.set_defaults(func=cmd_verify)
    a = sub.add_parser("analyse"); a.add_argument("--out", default="out/objr2c"); a.add_argument("--log", default="out/objr2c_cluster.log"); a.add_argument("--json", default="out/interp/objr2c/compare.json")
    a.add_argument("--rf-maps", nargs="+", default=DEFAULT_MAPS); a.add_argument("--no-verify", action="store_true"); a.add_argument("--force", action="store_true")
    a.add_argument("--skip-sphere", action="store_true"); a.add_argument("--skip-spec", action="store_true"); a.add_argument("--skip-bench", action="store_true"); a.set_defaults(func=cmd_analyse)
    st = sub.add_parser("streams"); st.set_defaults(func=cmd_streams)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
