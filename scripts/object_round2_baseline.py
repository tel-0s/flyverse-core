"""Round-2 MATCHED object baseline (Neurome intake, experiment 1): the batch builder, the batch check, and the analysis.

One cluster submission (docs/INTERP.md 2.4 / 10.4: runs are the replicate unit, every arm of a comparison in ONE
submission), three parts, all under out/objr2/:

  sph/  the matched ray-traced sphere ladder (scripts/probe_object_matched.py run): elevation 0 deg, 5 cm, +-50 deg
        at 40 deg/s, dark ball, eye 0.15 m, headlamp; diameters 4.5 / 8.8 / 11 / 15 / 20 / 30 deg; two lobes --
        `ship` (the shipped OpticParams) and `fb0` (`--optic gain_fb=0`, the deterministic lobe, the exact-arithmetic
        control); N_RUNS object runs per rung + N_RUNS blank/blank nulls per lobe.
  syn/  the synthetic radiance ladders (scripts/probe_synthetic_stimuli.py record) on the shipped lobe: the height
        ladder (width 4.4), the width ladder (height 8.8), the square ladder, all dark (-0.995), plus the square ladder
        bright (+0.995); the SAME arc as the sphere (elevation 0, +-50 deg at 40 deg/s); N_RUNS runs per rung + N_RUNS
        blank/blank nulls.
  loc/  the per-body RF localizer (15-deg dark square, 3 passes: the only configuration that localized an LC type,
        docs/audits/object_synthetic_stimuli.md 8): fb0 x 1 + shipped x 3.

Why N_RUNS = 6 and not 5: the predeclared primary family is 6 rungs x {drive, spikes} = 12 members per LC type with Holm
inside it. The exact Mann-Whitney floor at 5 v 5 is p = 0.0079 (common.p_floor) which is ABOVE the Holm threshold for
the smallest member, 0.05 / 12 = 0.0042 -- at 5 runs per arm no member of a 12-member family can survive Holm however
large the effect. At 6 v 6 the floor is 2 / C(12, 6) = 0.00216 < 0.0042, so a fully separated member can. Six runs per
arm is the smallest count at which the predeclared family is decidable.

Job packing: a cluster job is one shell line; the rented boxes take ~24 concurrent jobs, so each job runs several
`run` / `record` PROCESSES in sequence (one process = one (A, B) pair under one brain seed = one run = the replicate
unit; the job is the container). Every job line ends with `check-job`, which exits 1 unless every expected output of
the job exists and every console says `device cuda`, so the cluster log's '<n> job(s), 0 failed' means what it says.

    python scripts/object_round2_baseline.py plan [--out out/objr2 --name objr2 --minutes 120 --runs 6 --seconds 12 --settle 3]
    python scripts/object_round2_baseline.py submit [--out out/objr2]          # runs the plan through scripts/cluster_run.py, tees out/objr2_cluster.log
    python scripts/object_round2_baseline.py check-job --expect STEM...        # on the box, at the end of every job line
    python scripts/object_round2_baseline.py verify [--out out/objr2 --log out/objr2_cluster.log --json out/objr2/verify.json]
    python scripts/object_round2_baseline.py rfmap [--out out/objr2]           # the RF maps of this submission's localizer runs
    python scripts/object_round2_baseline.py analyse [--out out/objr2 --json out/interp/objr2/baseline.json]
    python scripts/object_round2_baseline.py oldladder                          # the old (scene-baseline) ladder's geometry from probe_object_sweep's constants

The reading rules (PREDECLARED below) were written before the batch was submitted and are dumped to
out/objr2/predeclared.json by `plan`; `analyse` copies them into the Result unchanged.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
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
import probe_object_matched as pom                     # noqa: E402  (CPU-only module functions: rf_frames, footprint, verify_runs)
import probe_synthetic_stimuli as pss                  # noqa: E402  (CPU-only: verify_dir, family_stats, ladders)

DIAMS = [4.5, 8.8, 11.0, 15.0, 20.0, 30.0]
N_RUNS = 6
SEEDS = list(range(N_RUNS))
LOBES = {"ship": "", "fb0": "--optic gain_fb=0"}
LC_TYPES = ("LC11", "LC10a")
UPSTREAM = ("T2", "T3", "Tm5Y", "TmY21")
SMALL_RUNGS, LARGE_RUNGS = (4.5, 8.8, 11.0), (20.0, 30.0)
ALPHA = 0.05
NULL_DIAM = 11.0                          # the diameter flag a null run carries (a null shows no ball; the window rule uses the rung's diameter)
SYN_HALF_SPAN, SYN_SPEED, SYN_EL = 50.0, 40.0, 0.0    # the sphere's arc, so the two assays share the trajectory
LOC_SIZE_DEG, LOC_PASSES = 15.0, 3
SYN_FAMILIES = {                          # name -> (ladder label, [(rung value, width, height)], contrast)
    "hlad_dark": ("height ladder, width 4.4, dark", [(h, pss.HEIGHT_LADDER["width"], h) for h in pss.HEIGHT_LADDER["heights"]], pss.DARK_CONTRAST),
    "wlad_dark": ("width ladder, height 8.8, dark", [(w, w, pss.WIDTH_LADDER["height"]) for w in pss.WIDTH_LADDER["widths"]], pss.DARK_CONTRAST),
    "sqlad_dark": ("square ladder, dark", [(s, s, s) for s in pss.SQUARE_LADDER], pss.DARK_CONTRAST),
    "sqlad_bright": ("square ladder, bright", [(s, s, s) for s in pss.SQUARE_LADDER], -pss.DARK_CONTRAST),
}

PREDECLARED = {
    "written": "2026-09-13, before the batch was submitted (out/objr2/predeclared.json is stamped by `plan`)",
    "replicate_unit": "one process = one (A, B) pair under one brain seed = one run; a cluster job is a container of sequential processes",
    "n_runs_per_arm": N_RUNS,
    "why_six": "Holm inside a 12-member family needs the smallest member at p <= 0.05/12 = 0.00417; the exact-U floor is 0.0079 at 5 v 5 and 0.00216 at 6 v 6",
    "window_rule": {
        "order": ["fitted in the shipped localizer map (15-deg square, 3 passes, fitted in all 3 runs, z_min 5)",
                  "else fitted in the fb0 localizer map (1 run)",
                  "else the anatomical column of trace.column_of_cells (the rfmap's anat_az_deg / anat_el_deg), box = the type's median fitted width over both maps, else 15 deg",
                  "else the whole window"],
        "sphere_frames": "frames on which the disc overlaps the body's box: |d az| <= (w + diam)/2 and |d el| <= (h + diam)/2 (probe_object_matched.rf_frames)",
        "rect_frames": "frames on which the rectangle overlaps the box: |d az| <= (w + width)/2 and |d el| <= (h + height)/2",
        "null_runs": "the same rule with the rung's diameter (rectangle) applied to the blank/blank run's track: the same frames, no object",
        "recorded": "per body: window_source in {rf_ship, rf_fb0, anat, whole}",
    },
    "primary": {
        "LC11": {"statistic": "per run: population MEDIAN over the 143 LC11 bodies of the per-body RF-windowed time-mean received drive (A - B, mV), and the same median of the per-body windowed spike-count difference (A - B, spikes in the window)",
                 "family": "6 rungs {4.5, 8.8, 11, 15, 20, 30} x {drive, spikes} = 12 members; Holm within; tie-aware exact permutation U (two-sided) per member; verdict from common.compare (z on the null SD, exact U, p_floor)",
                 "null": "the six blank/blank runs of the same lobe in the same submission, windowed with the rung's diameter",
                 "expectation": "a preference for small objects (4-9 deg; Keles & Frye 2017 Fig 3D/E, calcium): the excess over the null largest at 4.5-8.8 deg and falling by 20-30 deg",
                 "preference_test": "Spearman rank correlation of the per-run statistic with the diameter over the 36 object runs, permutation p (20,000 label shuffles, seed 0); small-vs-large contrast = mean over runs at {4.5, 8.8, 11} minus at {20, 30}, permutation p over run labels"},
        "LC10a": {"statistic": "the same two statistics over the 275 LC10a bodies",
                  "family": "the same 12-member family, Holm within",
                  "expectation": "a preference for 15-30 deg (Schretter et al. 2024 Fig 3a): the excess largest at 15-30 deg",
                  "preference_test": "the same two tests; the expected sign is positive"},
        "call": "a size preference is CALLED only if at least one member reads `result` from common.compare AND survives Holm (p_holm <= 0.05); the preference test then says which way. Otherwise the family is exploratory and the answer is 'no detected size preference at 6 v 6 on this statistic'",
    },
    "secondary_exploratory": {
        "upstream": "T2 / T3 / Tm5Y / TmY21: diff_signed_best_cell (max over cells of |mean dr_A - mean dr_B|) AND population diff_signed_mean, kept as two statistics, plus diff_abs_best_cell_mean as a third; whole-window, probe_object_matched's definitions; per rung vs the same lobe's nulls; unadjusted p and Holm reported, all EXPLORATORY",
        "lc_other": "population mean and best-cell (max over bodies) of the windowed per-body drive; the whole-window diff_max_over_cells_mean_mv (the old ladder's headline statistic) for continuity; diff_rate_hz_max_cell; all exploratory",
        "fb0": "the deterministic lobe: its blank/blank null has SD ~0 for optic quantities, so compare returns `undetermined`; read as magnitudes (diff) with the LIF's own spike scatter",
        "synthetic": "the rectangle ladders: the same per-body windowed medians, the same compare, Holm within each (family x type) = 12, reported as the matched-rectangle families; the sphere family is the primary",
    },
    "footprint": "per rung from the CAPTURED radiance: columns dimmed > 50 % and changed > 5 % per frame, centroid elevation (mean, band), and the effective-contrast readout the skeptic asked for: the per-frame extreme relative luminance change (median over frames) and the mean relative change over the dimmed (> 5 %) set",
    "not_matched": "the effective retinal contrast is NOT matched across the sphere ladder (a 4.5-deg ball only partially fills a 4.6-deg column); the synthetic ladders are contrast-matched by construction (Weber +-0.995 per covered fraction) and the bright synthetic ladder is the contrast-matched bright control; the sphere `lamp` ball is not contrast-matched and is not run here",
}


# ==================================================================================================== statistics helpers
def tie_aware_exact_u(a, b, max_enum: int = 200000) -> dict:
    """Two-sided exact permutation test of the Mann-Whitney U with average ranks for ties (the tie-aware U Neurome
    asked for): every C(n_a + n_b, n_a) assignment of the pooled values to the arms is enumerated (when that count is at
    most `max_enum`), U computed with 0.5 per tie, and p = the fraction of assignments whose |U - n_a n_b / 2| is at least
    the observed one. Unlike scipy's `method='exact'`, ties are part of the reference distribution."""
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return {"U": float("nan"), "p_tie_exact": float("nan"), "n_tied": 0, "enumerated": 0}
    pooled = np.concatenate([a, b]); r = rankdata(pooled)          # average ranks: ties count 0.5
    u_obs = r[:n].sum() - n * (n + 1) / 2
    centre = n * m / 2
    n_tied = int(len(pooled) - len(np.unique(pooled)))
    from math import comb
    total = comb(n + m, n)
    if total > max_enum:
        rng = np.random.RandomState(0); cnt = 0; N = max_enum
        for _ in range(N):
            perm = rng.permutation(n + m); u = r[perm[:n]].sum() - n * (n + 1) / 2
            cnt += abs(u - centre) >= abs(u_obs - centre) - 1e-9
        return {"U": float(u_obs), "p_tie_exact": float(cnt / N), "n_tied": n_tied, "enumerated": 0, "monte_carlo": N}
    cnt = 0
    for idx in itertools.combinations(range(n + m), n):
        u = r[list(idx)].sum() - n * (n + 1) / 2
        cnt += abs(u - centre) >= abs(u_obs - centre) - 1e-9
    return {"U": float(u_obs), "p_tie_exact": float(cnt / total), "n_tied": n_tied, "enumerated": int(total)}


def holm(pvals, m: int | None = None) -> np.ndarray:
    """Holm step-down adjusted p-values. m is the DECLARED family size (default len(pvals), NaN members included): a
    member whose run set is missing or whose test is undefined still counts toward m, so a family can only get more
    conservative when a rung drops out, never quietly smaller (the round-2 compare skeptic's finding)."""
    p = np.asarray(pvals, float); out = np.full(p.shape, np.nan)
    ok = np.isfinite(p); m = int(len(p)) if m is None else int(m)
    if int(ok.sum()) == 0:
        return out
    m = max(m, int(ok.sum()))
    k = int(ok.sum())
    order = np.argsort(p[ok]); ps = p[ok][order]
    adj = np.maximum.accumulate([(m - i) * ps[i] for i in range(k)])
    adj = np.minimum(adj, 1.0)
    tmp = np.empty(k); tmp[order] = adj
    out[ok] = tmp
    return out


def compare_row(stim, null, label: dict) -> dict:
    """common.compare (the verdict vocabulary) + the tie-aware exact U on the same values."""
    stim = [float(v) for v in stim if np.isfinite(v)]; null = [float(v) for v in null if np.isfinite(v)]
    if stim and null:
        c = common.compare(stim, null)
    else:                                                            # an arm is empty: magnitudes of the other arm, verdict underpowered
        a = common.ArmStats.of(stim) if stim else None; b = common.ArmStats.of(null) if null else None
        c = {"stim": a.record() if a else {"n": 0, "mean": np.nan, "sd": np.nan, "values": []}, "null": b.record() if b else {"n": 0, "mean": np.nan, "sd": np.nan, "values": []},
             "diff": np.nan, "z": np.nan, "welch": np.nan, "U": np.nan, "p": np.nan, "verdict": "underpowered", "p_floor": np.nan, "null_sd_zero": False}
    t = tie_aware_exact_u(stim, null)
    return dict(label, stim_n=c["stim"]["n"], stim_mean=c["stim"]["mean"], stim_sd=c["stim"]["sd"], stim_values=c["stim"]["values"],
                null_n=c["null"]["n"], null_mean=c["null"]["mean"], null_sd=c["null"]["sd"], null_values=c["null"]["values"],
                diff=c["diff"], z=c["z"], welch=c["welch"], U=c["U"], p_scipy_exact=c["p"], p_floor=c["p_floor"], null_sd_zero=c["null_sd_zero"],
                verdict=c["verdict"], U_tie=t["U"], p=t["p_tie_exact"], n_tied=t["n_tied"], p_method="tie-aware exact permutation U (two-sided)")


def spearman_perm(x, y, n_perm: int = 20000, seed: int = 0) -> dict:
    """Spearman rho of y on x with a permutation p (labels of y shuffled)."""
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 4:
        return {"rho": np.nan, "p_perm": np.nan, "n": int(len(x))}
    rho = float(spearmanr(x, y).correlation)
    if not np.isfinite(rho):                  # constant x or y (e.g. every spike median 0.0): no rank test exists
        return {"rho": np.nan, "p_perm": np.nan, "n": int(len(x)), "n_perm": n_perm, "note": "rho undefined (constant input)"}
    rng = np.random.RandomState(seed); cnt = 0
    for _ in range(n_perm):
        cnt += abs(float(spearmanr(x, rng.permutation(y)).correlation)) >= abs(rho) - 1e-12
    return {"rho": rho, "p_perm": float((cnt + 1) / (n_perm + 1)), "n": int(len(x)), "n_perm": n_perm}


def contrast_perm(vals_small, vals_large, n_perm: int = 20000, seed: int = 0) -> dict:
    """mean(small rungs) - mean(large rungs) over runs with a permutation p over run labels."""
    a = np.asarray(vals_small, float); b = np.asarray(vals_large, float); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"contrast": np.nan, "p_perm": np.nan}
    obs = a.mean() - b.mean(); pooled = np.concatenate([a, b]); rng = np.random.RandomState(seed); cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled); cnt += abs(p[:len(a)].mean() - p[len(a):].mean()) >= abs(obs) - 1e-12
    return {"contrast": float(obs), "p_perm": float((cnt + 1) / (n_perm + 1)), "n_small": int(len(a)), "n_large": int(len(b))}


# ==================================================================================================== the plan
def sphere_stem(lobe: str, diam: float | None, seed: int) -> str:
    return f"{lobe}_null_s{seed}" if diam is None else f"{lobe}_d{int(round(diam * 10)):03d}_obj_s{seed}"


def syn_stem(fam: str, rung: float, seed: int) -> str:
    return f"{fam}_r{int(round(rung * 10)):03d}_s{seed}"


def _sphere_cmd(out: str, lobe: str, diam: float | None, seed: int, seconds: float, settle: float) -> tuple[str, str]:
    stem = f"{out}/sph/{sphere_stem(lobe, diam, seed)}"
    flags = LOBES[lobe]
    d = NULL_DIAM if diam is None else diam
    cmd = (f"python scripts/probe_object_matched.py run --diam-deg {d:g} --seconds {seconds:g} --settle {settle:g} --seed {seed}"
           f"{' --null' if diam is None else ''}{' ' + flags if flags else ''} --out {stem} > {stem}.txt 2>&1")
    return stem, cmd


def _syn_cmd(out: str, fam: str | None, rung: float | None, seed: int, seconds: float, settle: float) -> tuple[str, str]:
    common_ = f"--speed {SYN_SPEED:g} --elevation {SYN_EL:g} --half-span {SYN_HALF_SPAN:g} --seconds {seconds:g} --settle {settle:g} --seed {seed}"
    if fam is None:
        stem = f"{out}/syn/null_s{seed}"
        cmd = f"python scripts/probe_synthetic_stimuli.py record --stimulus rect --width 4.4 --height 8.8 --contrast {pss.DARK_CONTRAST} {common_} --null --out {stem} > {stem}.txt 2>&1"
        return stem, cmd
    _, rungs, contrast = SYN_FAMILIES[fam]
    w, h = [(w_, h_) for r_, w_, h_ in rungs if r_ == rung][0]
    stem = f"{out}/syn/{syn_stem(fam, rung, seed)}"
    cmd = f"python scripts/probe_synthetic_stimuli.py record --stimulus rect --width {w:g} --height {h:g} --contrast {contrast:g} {common_} --out {stem} > {stem}.txt 2>&1"
    return stem, cmd


def _loc_cmd(out: str, lobe: str, seed: int, settle: float = 2.0) -> tuple[str, str]:
    stem = f"{out}/loc/loc150_{lobe}_s{seed}"
    flags = LOBES[lobe]
    cmd = (f"python scripts/probe_synthetic_stimuli.py record --stimulus localizer --width {LOC_SIZE_DEG:g} --passes {LOC_PASSES} --settle {settle:g}"
           f"{' ' + flags if flags else ''} --seed {seed} --out {stem} > {stem}.txt 2>&1")
    return stem, cmd


def _job_line(out: str, job: str, runs: int, seconds: float, settle: float, loc_ship_runs: int) -> str:
    """One cluster job: the round's prefix (mkdir, venv, CUDA assert), then `run-job`, which regenerates this job's
    process list from the same arguments ON THE BOX (this script ships with the batch), runs the processes in
    sequence (a failed process does not stop the next), and exits 1 unless every expected output exists and every
    console says `device cuda` (check-job). The expanded process commands are in jobs.json for the record. (A single
    shell line carrying 14-25 process commands exceeded the Windows 32 k command-line limit at submission.)"""
    return (f"mkdir -p {out}/sph {out}/syn {out}/loc && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/object_round2_baseline.py run-job --job {job} --out {out} --runs {runs} --seconds {seconds:g} --settle {settle:g} --loc-ship-runs {loc_ship_runs}")


def build_jobs(out: str, runs: int, seconds: float, settle: float, loc_ship_runs: int = 3) -> list[dict]:
    seeds = list(range(runs))
    jobs = []

    def add(name, kind, procs):
        jobs.append({"job": name, "kind": kind, "stems": [p[0] for p in procs], "commands": [p[1] for p in procs],
                     "expect": [f"{p[0]}{'.json' if kind == 'sph' else '_summary.json'}" for p in procs],
                     "line": _job_line(out, name, runs, seconds, settle, loc_ship_runs)})

    for s in seeds:                                                  # sphere: one job per seed, both lobes, every rung + the null
        add(f"sph_s{s}", "sph", [_sphere_cmd(out, lobe, d, s, seconds, settle) for lobe in LOBES for d in DIAMS + [None]])
    for s in seeds:                                                  # synthetic: one job per seed, the four ladders + the null
        procs = [_syn_cmd(out, fam, r, s, seconds, settle) for fam, (_, rungs, _) in SYN_FAMILIES.items() for r, _, _ in rungs]
        procs.append(_syn_cmd(out, None, None, s, seconds, settle))
        add(f"syn_s{s}", "syn", procs)
    add("loc_fb0_s0", "loc", [_loc_cmd(out, "fb0", 0)])
    for s in range(loc_ship_runs):
        add(f"loc_ship_s{s}", "loc", [_loc_cmd(out, "ship", s)])
    return jobs


def cmd_run_job(args) -> int:
    """On the box: the processes of one job, in sequence, each with its console file; then the check."""
    jobs = {j["job"]: j for j in build_jobs(args.out.rstrip("/"), args.runs, args.seconds, args.settle, args.loc_ship_runs)}
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


def tree_state() -> dict:
    """What the working tree looked like when the plan was written (cluster_run ships every file that differs from
    origin/main, other tasks' uncommitted edits included): commit, status, and the sha256 of the simulation sources."""
    def run(cmd):
        try:
            return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"unavailable: {e!r}"
    files = sorted(glob.glob(str(ROOT / "flyverse" / "*.py")) + glob.glob(str(ROOT / "flyverse" / "interp" / "*.py")) +
                   [str(ROOT / "scripts" / f) for f in ("probe_object_matched.py", "probe_synthetic_stimuli.py", "object_round2_baseline.py", "room_demo.py", "interp_trace.py")])
    sha = {}
    for f in files:
        p = Path(f)
        if p.exists():
            sha[str(p.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"commit": run(["git", "rev-parse", "HEAD"]), "status_porcelain": run(["git", "status", "--porcelain"]).splitlines(),
            "diff_stat": run(["git", "diff", "--stat"]).splitlines()[-1:] , "sha256_lf": sha, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def cmd_plan(args) -> int:
    out = args.out.rstrip("/")
    Path(out).mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(out, args.runs, args.seconds, args.settle, args.loc_ship_runs)
    n_proc = sum(len(j["stems"]) for j in jobs)
    log = f"out/{args.name}_cluster.log"
    lines = [j["line"] for j in jobs]
    sh = ["#!/bin/sh", f"# generated by scripts/object_round2_baseline.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs, {n_proc} processes", "set -e",
          f"mkdir -p {out} out", f'if [ -f {log} ]; then mv {log} "out/{args.name}_cluster.$(date +%Y%m%dT%H%M%S).log"; fi',
          f"python scripts/cluster_run.py --name {args.name} --minutes {args.minutes} \\"]
    sh += [f"  {shlex.quote(l)} \\" for l in lines]
    sh += [f"  --fetch {out}/ 2>&1 | tee {log}"]
    (Path(out) / "batch.sh").write_text("\n".join(sh) + "\n", encoding="utf-8")
    with open(Path(out) / "jobs.json", "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "minutes": args.minutes, "runs": args.runs, "seconds": args.seconds, "settle": args.settle, "out": out,
                   "diameters_deg": DIAMS, "lobes": LOBES, "synthetic_families": {k: {"label": v[0], "rungs": v[1], "contrast": v[2]} for k, v in SYN_FAMILIES.items()},
                   "synthetic_arc": {"half_span_deg": SYN_HALF_SPAN, "speed_deg_s": SYN_SPEED, "elevation_deg": SYN_EL},
                   "localizer": {"size_deg": LOC_SIZE_DEG, "passes": LOC_PASSES, "fb0_runs": 1, "ship_runs": args.loc_ship_runs},
                   "jobs": jobs, "n_processes": n_proc, "fetch": out + "/", "log": log}, f, indent=1)
    with open(Path(out) / "predeclared.json", "w", encoding="utf-8") as f:
        json.dump({"predeclared": PREDECLARED, "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generator": " ".join(sys.argv)}, f, indent=1)
    with open(Path(out) / "tree_state.json", "w", encoding="utf-8") as f:
        json.dump(tree_state(), f, indent=1)
    print(f"{len(jobs)} job(s), {n_proc} processes -> {out}/ (log {log}); batch.sh, jobs.json, predeclared.json, tree_state.json written")
    for j in jobs:
        print(f"  {j['job']:12s} {len(j['stems']):3d} processes: {j['stems'][0]} .. {j['stems'][-1]}")
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
            sys.stdout.write(line); sys.stdout.flush(); f.write(line)
        rc = p.wait()
    print(f"cluster_run exit {rc}; log {log}")
    return rc


def cmd_check_job(args) -> int:
    """On the box: every expected output file exists and every console says `device cuda` (exit 1 otherwise)."""
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


# ==================================================================================================== verify (CPU)
def parse_sphere_name(stem: str) -> dict | None:
    m = re.match(r"(ship|fb0)_(?:d(\d{3})_obj|null)_s(\d+)$", stem)
    if not m:
        return None
    return {"lobe": m.group(1), "null": m.group(2) is None, "diam": None if m.group(2) is None else int(m.group(2)) / 10.0, "seed": int(m.group(3))}


def parse_syn_name(stem: str) -> dict | None:
    m = re.match(r"(hlad_dark|wlad_dark|sqlad_dark|sqlad_bright)_r(\d{3})_s(\d+)$", stem)
    if m:
        return {"family": m.group(1), "null": False, "rung": int(m.group(2)) / 10.0, "seed": int(m.group(3))}
    m = re.match(r"null_s(\d+)$", stem)
    return {"family": None, "null": True, "rung": None, "seed": int(m.group(1))} if m else None


def verify_all(out: str, log: str | None) -> dict:
    rep = {"problems": [], "log": None, "sphere": None, "synth": None, "loc": None, "expected_missing": []}
    if log and Path(log).exists():
        lines = [l for l in Path(log).read_text(encoding="utf-8", errors="replace").splitlines() if "job(s)" in l and "failed" in l]
        rep["log"] = lines[-1] if lines else None
        if not lines or ", 0 failed" not in lines[-1]:
            rep["problems"].append(f"cluster log: {rep['log']!r}")
    plan_p = Path(out) / "jobs.json"
    if plan_p.exists():
        plan = json.load(open(plan_p, encoding="utf-8"))
        for j in plan["jobs"]:
            for s in j["stems"]:
                f = s + (".json" if j["kind"] == "sph" else "_summary.json")
                if not Path(f).exists():
                    rep["expected_missing"].append(f)
        if rep["expected_missing"]:
            rep["problems"].append(f"{len(rep['expected_missing'])} expected outputs missing")
    sph = sorted(glob.glob(f"{out}/sph/*.json"))
    if sph:
        df, probs = pom.verify_runs(sph)
        rep["problems"] += probs
        # consistency: the file name's lobe / null / diameter against the JSON's own record
        for p in sph:
            d = json.load(open(p, encoding="utf-8")); nm = parse_sphere_name(Path(p).stem)
            if nm is None:
                rep["problems"].append(f"{p}: unexpected file name"); continue
            gfb = d["provenance"]["model"]["optic"].get("gain_fb")
            if (nm["lobe"] == "fb0") != (gfb == 0):
                rep["problems"].append(f"{p}: lobe {nm['lobe']} but gain_fb {gfb}")
            if bool(d["summary"]["null"]) != nm["null"]:
                rep["problems"].append(f"{p}: null flag {d['summary']['null']} vs name")
            if not nm["null"] and abs(float(d["provenance"]["stimulus"]["params"]["diam_deg"]) - nm["diam"]) > 1e-6:
                rep["problems"].append(f"{p}: diam {d['provenance']['stimulus']['params']['diam_deg']} vs name {nm['diam']}")
            if d["replicates"].get("seed") != nm["seed"]:
                rep["problems"].append(f"{p}: seed {d['replicates'].get('seed')} vs name")
        rep["sphere"] = {"n_runs": int(len(df)), "el_band_spread_deg": df.attrs.get("el_band_spread_deg"), "el_band_verdict": df.attrs.get("el_band_verdict"),
                         "devices": sorted(set(df.device_name.dropna().astype(str))), "runs": df.to_dict("records")}
    for key in ("syn", "loc"):
        d = f"{out}/{key}"
        if Path(d).exists():
            df = pss.verify_dir(d)
            rep[key if key == "loc" else "synth"] = {"n_runs": int(len(df)), "n_bad": int((~df.ok).sum()) if len(df) else 0, "runs": df.to_dict("records")}
            if len(df) and (~df.ok).any():
                rep["problems"].append(f"{key}: {int((~df.ok).sum())} run(s) not verified: {df[~df.ok].run.tolist()}")
            for r in df.to_dict("records"):
                if key == "syn":
                    nm = parse_syn_name(r["run"])
                    if nm is None:
                        rep["problems"].append(f"{key}/{r['run']}: unexpected file name"); continue
                    if r["seed"] != nm["seed"]:
                        rep["problems"].append(f"{key}/{r['run']}: seed {r['seed']} vs name")
                    if json.loads(r["optic"] or "{}"):
                        rep["problems"].append(f"{key}/{r['run']}: optic overrides {r['optic']} on a shipped-lobe run")
                else:
                    fb0 = "fb0" in r["run"]; ov = json.loads(r["optic"] or "{}")
                    if fb0 != (ov.get("gain_fb") == 0):
                        rep["problems"].append(f"{key}/{r['run']}: optic overrides {ov} vs name")
    return rep


def cmd_verify(args) -> int:
    rep = verify_all(args.out.rstrip("/"), args.log)
    print(f"cluster log: {rep['log']}")
    if rep["sphere"]:
        print(f"sphere: {rep['sphere']['n_runs']} runs on {rep['sphere']['devices']}; centroid-elevation spread across object runs {rep['sphere']['el_band_spread_deg']} deg -> {rep['sphere']['el_band_verdict']}")
    if rep["synth"]:
        print(f"synthetic: {rep['synth']['n_runs']} runs, {rep['synth']['n_bad']} not verified")
    if rep["loc"]:
        print(f"localizer: {rep['loc']['n_runs']} runs, {rep['loc']['n_bad']} not verified")
    print(f"expected outputs missing: {len(rep['expected_missing'])}")
    print("problems: " + ("; ".join(rep["problems"]) if rep["problems"] else "none"))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(common.to_jsonable(dict(rep, generator=" ".join(sys.argv))), f, indent=1)
        print(f"written {args.json}")
    return 1 if rep["problems"] else 0


# ==================================================================================================== RF maps
def cmd_rfmap(args) -> int:
    out = args.out.rstrip("/")
    rc = 0
    for lobe, pattern in (("ship", f"{out}/loc/loc150_ship_s*_nodes.npz"), ("fb0", f"{out}/loc/loc150_fb0_s*_nodes.npz")):
        runs = sorted(glob.glob(pattern))
        if not runs:
            print(f"{lobe}: no localizer runs match {pattern}"); rc = 1; continue
        cmd = [sys.executable, "scripts/probe_synthetic_stimuli.py", "rfmap", "--runs", *runs, "--csv", f"{out}/rfmap_{lobe}.csv",
               "--json", f"out/interp/objr2/rfmap_{lobe}.json", "--z-min", str(args.z_min)]
        print("+ " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        rc |= r.returncode
    return rc


def load_windows(out: str) -> tuple[pd.DataFrame, dict]:
    """The predeclared window rule applied to the localizer maps of this submission (PREDECLARED['window_rule'])."""
    maps = {}
    for lobe in ("ship", "fb0"):
        p = Path(out) / f"rfmap_{lobe}.csv"
        if p.exists():
            m = pd.read_csv(p, dtype={"bodyId": str}); m["fitted"] = m["fitted"].astype(str).str.lower().isin(("true", "1", "1.0")); maps[lobe] = m
    if not maps:
        raise SystemExit(f"no rfmap_ship.csv / rfmap_fb0.csv under {out}: run `rfmap` first")
    base = maps.get("ship", maps.get("fb0"))
    widths = {}
    for t in sorted(set(base.type)):
        w = pd.concat([m[(m.type == t) & m.fitted].width_deg for m in maps.values()])
        widths[t] = float(w.median()) if len(w) else 15.0
    rows = []
    for _, r in base.iterrows():
        b = r.bodyId; t = r.type; src = None
        for lobe in ("ship", "fb0"):
            m = maps.get(lobe)
            if m is None:
                continue
            q = m[(m.bodyId == b) & m.fitted]
            if len(q):
                q = q.iloc[0]; rows.append({"bodyId": b, "type": t, "window_source": f"rf_{lobe}", "az_deg": float(q.az_deg), "el_deg": float(q.el_deg),
                                            "width_deg": float(q.width_deg), "height_deg": float(q.width_deg)}); src = lobe; break
        if src:
            continue
        if np.isfinite(r.get("anat_az_deg", np.nan)):
            rows.append({"bodyId": b, "type": t, "window_source": "anat", "az_deg": float(r.anat_az_deg), "el_deg": float(r.anat_el_deg),
                         "width_deg": widths[t], "height_deg": widths[t]})
        else:
            rows.append({"bodyId": b, "type": t, "window_source": "whole", "az_deg": np.nan, "el_deg": np.nan, "width_deg": np.nan, "height_deg": np.nan})
    win = pd.DataFrame(rows).drop_duplicates("bodyId").reset_index(drop=True)
    summary = {"maps": {k: str(Path(out) / f"rfmap_{k}.csv") for k in maps}, "box_width_by_type_deg": widths,
               "sources": {t: win[win.type == t].window_source.value_counts().to_dict() for t in sorted(set(win.type))},
               "fitted_bodies": {k: {t: int(((m.type == t) & m.fitted).sum()) for t in sorted(set(m.type))} for k, m in maps.items()}}
    return win, summary


# ==================================================================================================== per-body windowed statistics
def box_frames(track_az, track_el, obj_w, obj_h, az, el, w, h) -> np.ndarray:
    """Frames on which an object of angular size (obj_w, obj_h) centred on the track overlaps the (w x h) box at (az, el)."""
    return (np.abs(np.asarray(track_az) - az) <= (w + obj_w) / 2) & (np.abs(np.asarray(track_el) - el) <= (h + obj_h) / 2)


def windowed_body_stats(A_drive, B_drive, A_spk, B_spk, bodies, types, win: pd.DataFrame, track_az, track_el, obj_w, obj_h, sel_types=LC_TYPES) -> pd.DataFrame:
    """Per body of `sel_types`: the window (by the predeclared rule), the windowed time-mean A - B drive, the windowed
    A - B spike count, both arms' windowed means, and the whole-window values."""
    wi = win.set_index("bodyId")
    T = min(len(A_drive), len(B_drive))
    rows = []
    for i in range(len(bodies)):
        t = str(types[i])
        if t not in sel_types:
            continue
        b = str(int(bodies[i]))
        if b in wi.index and wi.loc[b, "window_source"] != "whole":
            r = wi.loc[b]; m = box_frames(track_az[:T], track_el[:T], obj_w, obj_h, r.az_deg, r.el_deg, r.width_deg, r.height_deg); src = r.window_source
        else:
            m = np.ones(T, bool); src = "whole"
        da = A_drive[:T, i].astype(np.float64); db = B_drive[:T, i].astype(np.float64); sa = A_spk[:T, i].astype(np.float64); sb = B_spk[:T, i].astype(np.float64)
        n = int(m.sum())
        rows.append({"bodyId": b, "type": t, "window_source": src, "n_frames_win": n, "n_frames": T,
                     "drive_a_win": float(da[m].mean()) if n else np.nan, "drive_b_win": float(db[m].mean()) if n else np.nan,
                     "drive_diff_win": float((da[m] - db[m]).mean()) if n else np.nan,
                     "spikes_a_win": float(sa[m].sum()) if n else np.nan, "spikes_b_win": float(sb[m].sum()) if n else np.nan,
                     "spikes_diff_win": float((sa[m] - sb[m]).sum()) if n else np.nan,
                     "drive_diff_whole": float((da - db).mean()), "spikes_a_whole": float(sa.sum()), "spikes_b_whole": float(sb.sum())})
    return pd.DataFrame(rows)


def pop_stats(pb: pd.DataFrame) -> dict:
    """Per run and type: the population median / mean / max over bodies (bodies with >= 1 window frame) of the windowed
    per-body statistics -- the values common.compare receives, one per run."""
    out = {}
    for t, g in pb.groupby("type"):
        g = g[g.n_frames_win > 0]
        out[t] = {"n_bodies_windowed": int(len(g)),
                  "drive_median": float(g.drive_diff_win.median()) if len(g) else np.nan, "drive_mean": float(g.drive_diff_win.mean()) if len(g) else np.nan,
                  "drive_max": float(g.drive_diff_win.max()) if len(g) else np.nan,
                  "spikes_median": float(g.spikes_diff_win.median()) if len(g) else np.nan, "spikes_mean": float(g.spikes_diff_win.mean()) if len(g) else np.nan,
                  "spikes_max": float(g.spikes_diff_win.max()) if len(g) else np.nan,
                  "spikes_a_total": float(g.spikes_a_win.sum()) if len(g) else np.nan, "spikes_b_total": float(g.spikes_b_win.sum()) if len(g) else np.nan,
                  "bodies_firing_a": int((g.spikes_a_whole > 0).sum()), "bodies_firing_b": int((g.spikes_b_whole > 0).sum())}
    return out


def contrast_readout(rad_a, rad_b, change_thresh: float = 0.05) -> dict:
    """The effective-contrast readout: per frame the extreme relative luminance change over columns (median over
    frames that have a changed column) and the mean relative change over the changed (> 5 %) set."""
    lum_o = np.asarray(rad_a, np.float64).sum(-1); lum_b = np.asarray(rad_b, np.float64).sum(-1)
    rel = (lum_o - lum_b) / (lum_b + 1e-9)
    ext = np.where(np.abs(rel.min(1)) >= np.abs(rel.max(1)), rel.min(1), rel.max(1))
    ch = np.abs(rel) > change_thresh
    has = ch.any(1)
    mean_ch = np.array([rel[j][ch[j]].mean() if ch[j].any() else np.nan for j in range(len(rel))])
    return {"extreme_rel_change_median": float(np.median(ext[has])) if has.any() else np.nan,
            "mean_rel_change_over_changed_set": float(np.nanmean(mean_ch)) if has.any() else np.nan,
            "frames_with_change_frac": float(has.mean())}


# ==================================================================================================== the sphere analysis
def load_sphere_runs(out: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(f"{out}/sph/*.json")):
        if p.endswith("verify.json"):
            continue
        nm = parse_sphere_name(Path(p).stem)
        if nm is None:
            continue
        rows.append(dict(nm, json=p, npz=p[:-5] + ".npz", stem=Path(p).stem))
    return pd.DataFrame(rows)


def analyse_sphere(out: str, win: pd.DataFrame) -> dict:
    runs = load_sphere_runs(out)
    if not len(runs):
        return {}
    per_body_rows, per_run_rows, tc_rows, fp_rows, up_rows = [], [], [], [], []
    prov0 = None
    for _, r in runs.iterrows():
        d = json.load(open(r.json, encoding="utf-8")); z = np.load(r.npz, allow_pickle=False)
        if prov0 is None and r.lobe == "ship":
            prov0 = d["provenance"]
        sm = d["summary"]; T = int(sm["n_frames"])
        diams = DIAMS if r.null else [r.diam]
        for dd in diams:                                                # a null run is windowed once per rung
            pb = windowed_body_stats(z["a__lc_drive_mv"], z["b__lc_drive_mv"], z["a__lc_spikes"], z["b__lc_spikes"], z["lc_body_ids"], z["lc_types"], win,
                                     z["track_az_deg"], z["track_el_deg"], dd, dd)
            pb.insert(0, "run", r.stem); pb.insert(1, "lobe", r.lobe); pb.insert(2, "null", bool(r.null)); pb.insert(3, "diam_deg", dd); pb.insert(4, "seed", int(r.seed))
            per_body_rows.append(pb)
            ps = pop_stats(pb)
            for t, v in ps.items():
                per_run_rows.append(dict(run=r.stem, lobe=r.lobe, null=bool(r.null), diam_deg=dd, seed=int(r.seed), type=t, **v,
                                         diff_max_over_cells_mean_mv=sm["diff_max_over_cells_mean_mv"][t], diff_rate_hz_max_cell=sm["diff_rate_hz_max_cell"][t],
                                         device=d["provenance"]["execution"].get("device_name")))
        for t in UPSTREAM:
            up_rows.append(dict(run=r.stem, lobe=r.lobe, null=bool(r.null), diam_deg=(np.nan if r.null else r.diam), seed=int(r.seed), type=t,
                                diff_signed_best_cell=sm["diff_signed_best_cell"][t], diff_signed_mean=sm["diff_signed_mean"][t], diff_abs_best_cell_mean=sm["diff_abs_best_cell_mean"][t]))
        fp = sm["footprint"]; cr = contrast_readout(z["rad_a"], z["rad_b"]) if not r.null else {}
        fp_rows.append(dict(run=r.stem, lobe=r.lobe, null=bool(r.null), diam_deg=(np.nan if r.null else r.diam), seed=int(r.seed),
                            n_dimmed_50pct_mean=fp["n_dimmed_50pct_mean"], n_changed_5pct_mean=fp["n_changed_5pct_mean"], centroid_el_mean_deg=fp["centroid_el_mean_deg"],
                            centroid_el_sd_deg=fp["centroid_el_sd_deg"], band_lo=fp["dimmed_el_band_deg"][0], band_hi=fp["dimmed_el_band_deg"][1],
                            centroid_az_min=fp["centroid_az_min_deg"], centroid_az_max=fp["centroid_az_max_deg"], blank_lum=fp["blank_lum_under_object_mean"],
                            blank_lum_cv=fp["blank_lum_under_object_cv"], realised_el_maxdev=sm["track_check"]["realised_el_deg_maxdev"],
                            realised_diam_maxdev=sm["track_check"]["realised_diam_deg_maxdev"], speed_min=sm["track_check"]["speed_deg_s_min"], speed_max=sm["track_check"]["speed_deg_s_max"], **cr))
        # sweep-locked per-body time course along the arc (20 azimuth bins), A - B
        tun = z["a__lc_drive_by_az_bin"] - z["b__lc_drive_by_az_bin"]
        for t in LC_TYPES:
            mi = np.flatnonzero(z["lc_types"] == t)
            for i in mi:
                tc_rows.append({"run": r.stem, "lobe": r.lobe, "null": bool(r.null), "diam_deg": (np.nan if r.null else r.diam), "seed": int(r.seed), "type": t,
                                "bodyId": str(int(z["lc_body_ids"][i])), "tc": tun[:, i].astype(np.float32)})
        az_edges = z["az_bin_edges_deg"]
    per_body = pd.concat(per_body_rows, ignore_index=True); per_run = pd.DataFrame(per_run_rows); fp_df = pd.DataFrame(fp_rows); up = pd.DataFrame(up_rows)
    # ---------------------------------------------------------------- primary comparisons and Holm
    comps, pref = [], []
    for lobe in LOBES:
        for t in LC_TYPES:
            nulls = per_run[(per_run.lobe == lobe) & per_run.null & (per_run.type == t)]
            objs = per_run[(per_run.lobe == lobe) & ~per_run.null & (per_run.type == t)]
            fam = []
            for dd in DIAMS:
                for q in ("drive_median", "spikes_median"):
                    s = objs[objs.diam_deg == dd][q].to_numpy(); n = nulls[nulls.diam_deg == dd][q].to_numpy()
                    fam.append(compare_row(s, n, {"family": f"primary:{lobe}:{t}", "lobe": lobe, "type": t, "diam_deg": dd, "statistic": q, "role": "primary" if lobe == "ship" else "control_fb0"}))
            ph = holm([c["p"] for c in fam])
            for c, h in zip(fam, ph):
                c["p_holm"] = float(h); c["survives_holm"] = bool(np.isfinite(h) and h <= ALPHA and c["verdict"] == "result")
            comps += fam
            for q in ("drive_median", "spikes_median", "drive_mean", "drive_max", "diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell"):   # exploratory
                if q in ("drive_median", "spikes_median"):
                    pass
                else:
                    ex = []
                    for dd in DIAMS:
                        s = objs[objs.diam_deg == dd][q].to_numpy(); n = nulls[nulls.diam_deg == dd][q].to_numpy()
                        ex.append(compare_row(s, n, {"family": f"exploratory:{lobe}:{t}:{q}", "lobe": lobe, "type": t, "diam_deg": dd, "statistic": q, "role": "exploratory"}))
                    ph = holm([c["p"] for c in ex])
                    for c, h in zip(ex, ph):
                        c["p_holm"] = float(h); c["survives_holm"] = bool(np.isfinite(h) and h <= ALPHA and c["verdict"] == "result")
                    comps += ex
                # the preference tests, per statistic
                x = objs.diam_deg.to_numpy(); y = objs[q].to_numpy()
                sp = spearman_perm(x, y); ct = contrast_perm(objs[objs.diam_deg.isin(SMALL_RUNGS)][q], objs[objs.diam_deg.isin(LARGE_RUNGS)][q])
                # the same statistic in the nulls: does the null itself trend with the window? (the window widens with the rung)
                xn = nulls.diam_deg.to_numpy(); yn = nulls[q].to_numpy(); spn = spearman_perm(xn, yn, n_perm=2000)
                pref.append({"lobe": lobe, "type": t, "statistic": q, "role": "primary" if q in ("drive_median", "spikes_median") else "exploratory",
                             "spearman_rho": sp["rho"], "spearman_p_perm": sp["p_perm"], "n_runs": sp["n"], "contrast_small_minus_large": ct["contrast"], "contrast_p_perm": ct["p_perm"],
                             "null_spearman_rho": spn["rho"], "null_spearman_p_perm": spn["p_perm"],
                             "excess_by_rung": {str(dd): float(objs[objs.diam_deg == dd][q].mean() - nulls[nulls.diam_deg == dd][q].mean()) for dd in DIAMS}})
    # secondary: upstream statistics (whole-window, the probe's definitions), exploratory
    ups = []
    for lobe in LOBES:
        for t in UPSTREAM:
            nulls = up[(up.lobe == lobe) & up.null & (up.type == t)]; objs = up[(up.lobe == lobe) & ~up.null & (up.type == t)]
            for q in ("diff_signed_best_cell", "diff_signed_mean", "diff_abs_best_cell_mean"):
                fam = [compare_row(objs[objs.diam_deg == dd][q].to_numpy(), nulls[q].to_numpy(),
                                   {"family": f"secondary:{lobe}:{t}:{q}", "lobe": lobe, "type": t, "diam_deg": dd, "statistic": q, "role": "secondary_exploratory"}) for dd in DIAMS]
                ph = holm([c["p"] for c in fam])
                for c, h in zip(fam, ph):
                    c["p_holm"] = float(h); c["survives_holm"] = bool(np.isfinite(h) and h <= ALPHA and c["verdict"] == "result")
                ups += fam
    # ---------------------------------------------------------------- per-body time courses for the top-10 bodies by RF coverage
    tc_df = pd.DataFrame(tc_rows)
    tc_out = []
    cov = per_body[~per_body.null].groupby(["lobe", "diam_deg", "type", "bodyId"]).n_frames_win.mean().reset_index()
    for lobe in LOBES:
        for dd in DIAMS:
            for t in LC_TYPES:
                top = cov[(cov.lobe == lobe) & (cov.diam_deg == dd) & (cov.type == t)].sort_values(["n_frames_win", "bodyId"], ascending=[False, True]).head(10)
                for b in top.bodyId:
                    for is_null in (False, True):
                        g = tc_df[(tc_df.lobe == lobe) & (tc_df.type == t) & (tc_df.bodyId == b) & (tc_df.null == is_null) & ((tc_df.diam_deg == dd) | is_null)]
                        if not len(g):
                            continue
                        M = np.stack(g.tc.to_numpy())
                        src = win.set_index("bodyId").window_source.get(b, "whole")
                        for k in range(M.shape[1]):
                            tc_out.append({"lobe": lobe, "diam_deg": dd, "type": t, "bodyId": b, "window_source": src, "arm": "null" if is_null else "object",
                                           "az_bin": k, "az_centre_deg": float((az_edges[k] + az_edges[k + 1]) / 2), "mean_diff_mv": float(M[:, k].mean()),
                                           "sd_runs_mv": float(M[:, k].std(ddof=1)) if len(M) > 1 else np.nan, "n_runs": int(len(M))})
    # ---------------------------------------------------------------- footprint per rung
    fps = fp_df[~fp_df.null].groupby(["lobe", "diam_deg"]).agg(n_runs=("run", "size"), n_dimmed_50pct=("n_dimmed_50pct_mean", "mean"), n_dimmed_50pct_sd=("n_dimmed_50pct_mean", "std"),
                                                              n_changed_5pct=("n_changed_5pct_mean", "mean"), centroid_el=("centroid_el_mean_deg", "mean"), centroid_el_sd_runs=("centroid_el_mean_deg", "std"),
                                                              band_lo=("band_lo", "min"), band_hi=("band_hi", "max"), blank_lum=("blank_lum", "mean"), blank_lum_cv=("blank_lum_cv", "mean"),
                                                              extreme_rel_change_median=("extreme_rel_change_median", "mean"), mean_rel_change_over_changed_set=("mean_rel_change_over_changed_set", "mean"),
                                                              realised_el_maxdev=("realised_el_maxdev", "max"), realised_diam_maxdev=("realised_diam_maxdev", "max"),
                                                              speed_min=("speed_min", "min"), speed_max=("speed_max", "max")).reset_index()
    return {"runs": runs.drop(columns=["json", "npz"]).to_dict("records"), "per_body": per_body, "per_run": per_run, "comparisons": comps + ups, "preference": pref,
            "time_course": pd.DataFrame(tc_out), "footprint_runs": fp_df, "footprint": fps, "upstream_runs": up, "prov0": prov0,
            "n_runs": {f"{l}:{'null' if n else 'obj'}": int(((runs.lobe == l) & (runs.null == n)).sum()) for l in LOBES for n in (False, True)}}


# ==================================================================================================== the synthetic analysis
def analyse_synth(out: str, win: pd.DataFrame) -> dict:
    files = sorted(glob.glob(f"{out}/syn/*_summary.json"))
    if not files:
        return {}
    per_body_rows, per_run_rows, up_rows, fp_rows = [], [], [], []
    prov0 = None
    for sf in files:
        stem = sf[:-len("_summary.json")]; nm = parse_syn_name(Path(stem).name)
        if nm is None:
            continue
        sm = json.load(open(sf, encoding="utf-8"))
        a_name, b_name = ("blank_a", "blank_b") if nm["null"] else ("stim", "blank")
        A = common.Recording.load(Path(stem + f"_{a_name}.npz")); B = common.Recording.load(Path(stem + f"_{b_name}.npz"))
        rad = dict(np.load(stem + "_radiance.npz", allow_pickle=False))
        if prov0 is None and os.path.exists(stem + "_prov.json"):
            prov0 = json.load(open(stem + "_prov.json", encoding="utf-8"))
        sa = np.diff(A.quantities["spike_count"], axis=0, prepend=A.quantities["spike_count"][:1]); sb = np.diff(B.quantities["spike_count"], axis=0, prepend=B.quantities["spike_count"][:1])
        rungs = [(fam, r, w, h) for fam, (_, rr, _) in SYN_FAMILIES.items() for r, w, h in rr] if nm["null"] else \
                [(nm["family"], nm["rung"], *[(w, h) for r, w, h in SYN_FAMILIES[nm["family"]][1] if r == nm["rung"]][0])]
        for fam, rung, w, h in rungs:
            pb = windowed_body_stats(A.quantities["drive_mv"], B.quantities["drive_mv"], sa, sb, A.body_ids, A.types.astype(str), win,
                                     rad["track__az_deg"], rad["track__el_deg"], w, h)
            pb.insert(0, "run", Path(stem).name); pb.insert(1, "family", fam); pb.insert(2, "null", nm["null"]); pb.insert(3, "rung", rung); pb.insert(4, "seed", nm["seed"])
            per_body_rows.append(pb)
            for t, v in pop_stats(pb).items():
                per_run_rows.append(dict(run=Path(stem).name, family=fam, null=nm["null"], rung=rung, seed=nm["seed"], type=t, **v, device=sm.get("device")))
        fs = pss.family_stats(A, B)
        for t in UPSTREAM + LC_TYPES:
            if t in fs:
                v = fs[t]
                up_rows.append(dict(run=Path(stem).name, family=nm["family"], null=nm["null"], rung=nm["rung"], seed=nm["seed"], type=t,
                                    **{k: v.get(k, np.nan) for k in ("diff_signed_best_cell", "diff_signed_mean", "diff_abs_best_cell_mean", "diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell")}))
        if not nm["null"]:
            P = rad["radiance"]; Bl = np.broadcast_to(rad["blank"][None], P.shape)
            fpx = pom.footprint(P, Bl, rad["col_az_el"])["summary"]; cr = contrast_readout(P, Bl)
            fp_rows.append(dict(run=Path(stem).name, family=nm["family"], rung=nm["rung"], seed=nm["seed"], n_dimmed_50pct_mean=fpx["n_dimmed_50pct_mean"], n_changed_5pct_mean=fpx["n_changed_5pct_mean"],
                                n_brightened_50pct_mean=fpx["n_brightened_50pct_mean"], centroid_el_mean_deg=fpx["centroid_el_mean_deg"], band_lo=fpx["dimmed_el_band_deg"][0], band_hi=fpx["dimmed_el_band_deg"][1],
                                frames_with_change=fpx["frames_with_dimmed"], **cr))
    per_body = pd.concat(per_body_rows, ignore_index=True); per_run = pd.DataFrame(per_run_rows); up = pd.DataFrame(up_rows); fp = pd.DataFrame(fp_rows)
    comps, pref = [], []
    for fam, (label, rungs, contrast) in SYN_FAMILIES.items():
        rv = [r for r, _, _ in rungs]
        for t in LC_TYPES:
            nulls = per_run[(per_run.family == fam) & per_run.null & (per_run.type == t)]; objs = per_run[(per_run.family == fam) & ~per_run.null & (per_run.type == t)]
            famc = []
            for r in rv:
                for q in ("drive_median", "spikes_median"):
                    famc.append(compare_row(objs[objs.rung == r][q].to_numpy(), nulls[nulls.rung == r][q].to_numpy(),
                                            {"family": f"synthetic:{fam}:{t}", "lobe": "ship", "type": t, "rung": r, "statistic": q, "role": "synthetic_family"}))
            ph = holm([c["p"] for c in famc])
            for c, h in zip(famc, ph):
                c["p_holm"] = float(h); c["survives_holm"] = bool(np.isfinite(h) and h <= ALPHA and c["verdict"] == "result")
            comps += famc
            for q in ("drive_median", "spikes_median", "drive_mean", "drive_max"):
                sp = spearman_perm(objs.rung, objs[q]); small = [x for x in rv[:3]]; large = rv[-2:]
                ct = contrast_perm(objs[objs.rung.isin(small)][q], objs[objs.rung.isin(large)][q])
                pref.append({"family": fam, "type": t, "statistic": q, "spearman_rho": sp["rho"], "spearman_p_perm": sp["p_perm"], "n_runs": sp["n"],
                             "contrast_small_minus_large": ct["contrast"], "contrast_p_perm": ct["p_perm"], "small_rungs": small, "large_rungs": large,
                             "excess_by_rung": {str(r): float(objs[objs.rung == r][q].mean() - nulls[nulls.rung == r][q].mean()) for r in rv}})
        for t in UPSTREAM + LC_TYPES:
            nulls = up[up.null & (up.type == t)]; objs = up[(up.family == fam) & (up.type == t)]
            for q in (("diff_signed_best_cell", "diff_signed_mean", "diff_abs_best_cell_mean") if t in UPSTREAM else ("diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell")):
                famc = [compare_row(objs[objs.rung == r][q].to_numpy(), nulls[q].to_numpy(), {"family": f"synthetic_secondary:{fam}:{t}:{q}", "lobe": "ship", "type": t, "rung": r, "statistic": q, "role": "synthetic_secondary"}) for r in rv]
                ph = holm([c["p"] for c in famc])
                for c, h in zip(famc, ph):
                    c["p_holm"] = float(h); c["survives_holm"] = bool(np.isfinite(h) and h <= ALPHA and c["verdict"] == "result")
                comps += famc
    fps = fp.groupby(["family", "rung"]).agg(n_runs=("run", "size"), n_dimmed_50pct=("n_dimmed_50pct_mean", "mean"), n_changed_5pct=("n_changed_5pct_mean", "mean"), n_brightened_50pct=("n_brightened_50pct_mean", "mean"),
                                             centroid_el=("centroid_el_mean_deg", "mean"), band_lo=("band_lo", "min"), band_hi=("band_hi", "max"), frames_with_change=("frames_with_change", "mean"),
                                             extreme_rel_change_median=("extreme_rel_change_median", "mean"), mean_rel_change_over_changed_set=("mean_rel_change_over_changed_set", "mean")).reset_index() if len(fp) else pd.DataFrame()
    return {"per_body": per_body, "per_run": per_run, "comparisons": comps, "preference": pref, "footprint": fps, "upstream_runs": up, "prov0": prov0,
            "n_runs": {f"{fam}:{'null' if n else 'obj'}": int(((per_run.family == fam) & (per_run.null == n)).sum() / max(len(LC_TYPES), 1)) for fam in SYN_FAMILIES for n in (False, True)}}


# ==================================================================================================== the old ladder (scene baseline)
def old_ladder_geometry(nominal=(4.5, 11.4, 20.0, 30.0)) -> pd.DataFrame:
    """The old ladder's geometry from probe_object_sweep.py's constants (AHEAD, HALF_SWEEP, SWEEP_S) and body.py's
    eye_height: the ball rests on the table (centre r above it, the eye 1.2 mm above it) and slides laterally at
    constant world speed. Radius from the probe's own convention 2 atan(r / AHEAD) = nominal."""
    src = (ROOT / "scripts" / "probe_object_sweep.py").read_text(encoding="utf-8")
    consts = {k: float(re.search(rf"^{k}\s*=\s*([0-9.eE+-]+)", src, re.M).group(1)) for k in ("AHEAD", "HALF_SWEEP", "SWEEP_S", "BALL_R")}
    eye_h = float(re.search(r"eye_height:\s*float\s*=\s*([0-9.eE+-]+)", (ROOT / "flyverse" / "body.py").read_text(encoding="utf-8")).group(1))
    a, s_max, sweep_s = consts["AHEAD"], consts["HALF_SWEEP"], consts["SWEEP_S"]
    v = 2 * s_max / sweep_s
    rows = []
    for deg in nominal:
        r = a * np.tan(np.radians(deg) / 2); dz = r - eye_h
        d0 = np.hypot(a, dz); d1 = np.sqrt(a ** 2 + s_max ** 2 + dz ** 2)
        rows.append({"nominal_deg": deg, "radius_mm": r * 1000, "centre_elevation_deg_centre": float(np.degrees(np.arctan2(dz, a))),
                     "centre_elevation_deg_end": float(np.degrees(np.arcsin(dz / d1))),
                     "angular_diameter_deg_centre": float(2 * np.degrees(np.arcsin(r / d0))), "angular_diameter_deg_end": float(2 * np.degrees(np.arcsin(r / d1))),
                     "angular_speed_deg_s_centre": float(np.degrees(v / a)), "angular_speed_deg_s_end": float(np.degrees(v * a / (a ** 2 + s_max ** 2))),
                     "azimuth_max_deg": float(np.degrees(np.arctan2(s_max, a)))})
    df = pd.DataFrame(rows); df.attrs["constants"] = dict(consts, eye_height_m=eye_h, lateral_speed_m_s=v)
    return df


def cmd_oldladder(args) -> int:
    df = old_ladder_geometry(args.nominal)
    print(f"probe_object_sweep constants {df.attrs['constants']}")
    common.print_table(df, floatfmt="{:.2f}")
    return 0


def old_ladder_tables(export_dir: str) -> dict:
    p = Path(export_dir)
    if not (p / "size_tuning.csv").exists():
        return {}
    st = pd.read_csv(p / "size_tuning.csv"); fp = pd.read_csv(p / "retina_footprint.csv")
    keep = st[st.type.isin(LC_TYPES + UPSTREAM) & st.statistic.isin(("diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell", "diff_signed_best_cell", "diff_signed_mean", "diff_abs_best_cell_mean"))]
    cols = ["size_id", "nominal_deg", "centre_elevation_deg_at_azimuth_0", "type", "statistic", "stim_mean", "stim_sd", "null_mean", "null_sd", "z", "p", "p_method", "verdict", "p_holm"]
    return {"dir": str(p), "size_tuning": keep[[c for c in cols if c in keep.columns]].to_dict("records"),
            "retina_footprint": fp[["size_id", "columns_dimmed_5pct_per_frame_mean", "columns_dimmed_50pct_per_frame_mean", "min_relative_radiance", "elevation_deg_min", "elevation_deg_max", "centre_elevation_deg_at_azimuth_0"]].to_dict("records")}


# ==================================================================================================== analyse
def _fmt_comp(c: dict) -> str:
    key = c.get("diam_deg", c.get("rung"))
    return (f"{c['type']:6s} {str(key):>5s} {c['statistic']:28s} stim {c['stim_mean']:+.4f} +- {c['stim_sd']:.4f} (n {c['stim_n']}) null {c['null_mean']:+.4f} +- {c['null_sd']:.4f} (n {c['null_n']}) "
            f"z {c['z']:+.2f} U {c['U_tie']:.1f} p {c['p']:.4f} (scipy {c['p_scipy_exact']:.4f}, ties {c['n_tied']}) p_holm {c.get('p_holm', float('nan')):.4f} -> {c['verdict']}{' SURVIVES HOLM' if c.get('survives_holm') else ''}")


def cmd_analyse(args) -> int:
    out = args.out.rstrip("/")
    rep = None
    if not args.no_verify:
        rep = verify_all(out, args.log)
        print(f"verify: log {rep['log']!r}; problems: {rep['problems'] or 'none'}")
        if rep["problems"] and not args.force:
            sys.exit("batch verification failed (pass --force to analyse anyway; the Result then records the problems)")
    win, win_summary = load_windows(out)
    print(f"windows: {win_summary['sources']}; box widths {win_summary['box_width_by_type_deg']}; fitted {win_summary['fitted_bodies']}")
    sph = analyse_sphere(out, win)
    syn = analyse_synth(out, win) if not args.skip_synth else {}
    old = old_ladder_tables(args.old_ladder) if args.old_ladder else {}
    oldgeo = old_ladder_geometry()
    prov = (sph or syn).get("prov0") or common.provenance(__import__("flyverse.connectome", fromlist=["load"]).load(verbose=False))
    res = common.Result.new("trace", prov)
    res.run_id = "objr2-baseline"; res.tool_version = "object_round2_baseline/1"
    res.replicates = {"n": N_RUNS, "unit": "runs", "runs": (sph or {}).get("runs"), "null": "the blank/blank runs of the same lobe in the same submission",
                      "n_runs": {"sphere": (sph or {}).get("n_runs"), "synthetic": (syn or {}).get("n_runs")}, "batch": {"jobs_json": f"{out}/jobs.json", "cluster_log": args.log}}
    Path(out).mkdir(parents=True, exist_ok=True)
    if sph:
        sph["per_body"].to_csv(f"{out}/per_body_sphere.csv", index=False)
        res.add_table("sphere_per_run", sph["per_run"]); res.add_table("sphere_comparisons", sph["comparisons"]); res.add_table("sphere_preference", sph["preference"])
        res.add_table("sphere_time_course", sph["time_course"]); res.add_table("sphere_footprint", sph["footprint"]); res.add_table("sphere_footprint_runs", sph["footprint_runs"])
        res.add_table("sphere_upstream_runs", sph["upstream_runs"])
    if syn:
        syn["per_body"].to_csv(f"{out}/per_body_synth.csv", index=False)
        res.add_table("synth_per_run", syn["per_run"]); res.add_table("synth_comparisons", syn["comparisons"]); res.add_table("synth_preference", syn["preference"])
        res.add_table("synth_footprint", syn["footprint"]); res.add_table("synth_upstream_runs", syn["upstream_runs"])
    res.add_table("windows", win); res.add_table("old_ladder_geometry", oldgeo)
    if old:
        res.add_table("old_ladder_size_tuning", old["size_tuning"]); res.add_table("old_ladder_footprint", old["retina_footprint"])
    # ---------------------------------------------------------------- the answers
    answers = {}
    comps = pd.DataFrame(sph["comparisons"]) if sph else pd.DataFrame()
    if len(comps):
        prim = comps[comps.role == "primary"]
        for t in LC_TYPES:
            g = prim[prim.type == t]
            surv = g[g.survives_holm.astype(bool)]
            res_rows = g[g.verdict == "result"]
            pf = [p for p in sph["preference"] if p["lobe"] == "ship" and p["type"] == t and p["statistic"] == "drive_median"][0]
            answers[t] = {"family_members": int(len(g)), "verdicts": g.verdict.value_counts().to_dict(), "min_p": float(g.p.min()), "min_p_holm": float(g.p_holm.min()),
                          "members_result_unadjusted": res_rows[["diam_deg", "statistic", "z", "p", "p_holm"]].to_dict("records"),
                          "members_surviving_holm": surv[["diam_deg", "statistic", "z", "p", "p_holm"]].to_dict("records"),
                          "size_preference_called": bool(len(surv)), "drive_median_spearman_rho": pf["spearman_rho"], "drive_median_spearman_p_perm": pf["spearman_p_perm"],
                          "drive_median_small_minus_large_mv": pf["contrast_small_minus_large"], "drive_median_contrast_p_perm": pf["contrast_p_perm"],
                          "excess_over_null_by_rung_mv": pf["excess_by_rung"]}
        sec = comps[comps.role == "secondary_exploratory"]
        answers["small_field"] = {}
        for t in ("T2", "T3"):
            g = sec[(sec.type == t) & (sec.lobe == "ship") & (sec.diam_deg.isin(SMALL_RUNGS))]
            answers["small_field"][t] = {"rows": g[["diam_deg", "statistic", "z", "p", "p_holm", "verdict"]].to_dict("records"),
                                        "any_result_unadjusted": bool((g.verdict == "result").any()), "any_survives_holm": bool(g.survives_holm.astype(bool).any())}
    res.summary = {"predeclared": PREDECLARED, "windows": win_summary, "answers": answers, "verify": rep, "synthetic_arc": {"half_span_deg": SYN_HALF_SPAN, "speed_deg_s": SYN_SPEED, "elevation_deg": SYN_EL},
                   "old_ladder_constants": oldgeo.attrs.get("constants"), "old_ladder_export": (old or {}).get("dir"),
                   "reading": "verdicts are common.compare's (z on the null SD, exact U, p_floor); p is the tie-aware exact permutation U; p_holm is Holm within the named family; only a `result` with p_holm <= 0.05 is called"}
    res.validation = {"name": "matched baseline: predeclared LC11 / LC10a families on the matched sphere ladder", "reference": {"LC11": "small-object preference (Keles & Frye 2017)", "LC10a": "15-30 deg preference (Schretter 2024)"},
                      "measured": answers, "status": "measured", "source": "scripts/object_round2_baseline.py analyse"}
    res.files = {"generator": " ".join(sys.argv), "per_body_sphere": f"{out}/per_body_sphere.csv" if sph else None, "per_body_synth": f"{out}/per_body_synth.csv" if syn else None,
                 "predeclared": f"{out}/predeclared.json", "jobs": f"{out}/jobs.json", "cluster_log": args.log, "rf_maps": win_summary["maps"]}
    p = res.save(args.json)
    # ---------------------------------------------------------------- console
    if sph:
        print("\n== SPHERE: footprint per rung (captured radiance; mean over runs)")
        common.print_table(sph["footprint"], floatfmt="{:.3f}")
        print("\n== SPHERE: primary family (per-run population MEDIAN over bodies of the RF-windowed per-body A - B; Holm within 12 per type)")
        for c in sph["comparisons"]:
            if c["role"] in ("primary", "control_fb0"):
                print(("  " if c["lobe"] == "ship" else "  [fb0] ") + _fmt_comp(c))
        print("\n== SPHERE: preference tests")
        common.print_table(pd.DataFrame([{k: v for k, v in p_.items() if k != "excess_by_rung"} for p_ in sph["preference"]]), floatfmt="{:+.4f}", max_rows=100)
        print("\n== SPHERE: exploratory LC statistics (unadjusted; Holm within each 6-rung set)")
        for c in sph["comparisons"]:
            if c["role"] == "exploratory" and c["lobe"] == "ship":
                print("  " + _fmt_comp(c))
        print("\n== SPHERE: upstream (secondary, exploratory), shipped lobe")
        for c in sph["comparisons"]:
            if c["role"] == "secondary_exploratory" and c["lobe"] == "ship":
                print("  " + _fmt_comp(c))
    if syn:
        print("\n== SYNTHETIC: footprint per family x rung")
        common.print_table(syn["footprint"], floatfmt="{:.3f}", max_rows=100)
        print("\n== SYNTHETIC: LC families (per-run population median of the windowed per-body A - B; Holm within 12 per family x type)")
        for c in syn["comparisons"]:
            if c["role"] == "synthetic_family":
                print("  " + _fmt_comp(c))
        print("\n== SYNTHETIC: preference tests")
        common.print_table(pd.DataFrame([{k: v for k, v in p_.items() if k != "excess_by_rung"} for p_ in syn["preference"]]), floatfmt="{:+.4f}", max_rows=100)
        print("\n== SYNTHETIC: secondary (upstream and whole-window LC statistics), exploratory")
        for c in syn["comparisons"]:
            if c["role"] == "synthetic_secondary" and c["type"] in ("T2", "T3", "LC11", "LC10a"):
                print("  " + _fmt_comp(c))
    print("\n== ANSWERS"); print(json.dumps(common.to_jsonable(answers), indent=1))
    print(f"\nproblems: {res.check() or 'none'}; written {p}")
    return 0


# ==================================================================================================== main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", default="out/objr2"); p.add_argument("--name", default="objr2"); p.add_argument("--minutes", type=int, default=120)
    p.add_argument("--runs", type=int, default=N_RUNS); p.add_argument("--seconds", type=float, default=12.0); p.add_argument("--settle", type=float, default=3.0)
    p.add_argument("--loc-ship-runs", type=int, default=3); p.set_defaults(func=cmd_plan)
    s = sub.add_parser("submit"); s.add_argument("--out", default="out/objr2"); s.set_defaults(func=cmd_submit)
    c = sub.add_parser("check-job"); c.add_argument("--expect", nargs="+", required=True); c.set_defaults(func=cmd_check_job)
    rj = sub.add_parser("run-job"); rj.add_argument("--job", required=True); rj.add_argument("--out", default="out/objr2"); rj.add_argument("--runs", type=int, default=N_RUNS)
    rj.add_argument("--seconds", type=float, default=12.0); rj.add_argument("--settle", type=float, default=3.0); rj.add_argument("--loc-ship-runs", type=int, default=3); rj.set_defaults(func=cmd_run_job)
    v = sub.add_parser("verify"); v.add_argument("--out", default="out/objr2"); v.add_argument("--log", default="out/objr2_cluster.log"); v.add_argument("--json", default="out/objr2/verify.json"); v.set_defaults(func=cmd_verify)
    m = sub.add_parser("rfmap"); m.add_argument("--out", default="out/objr2"); m.add_argument("--z-min", type=float, default=5.0); m.set_defaults(func=cmd_rfmap)
    a = sub.add_parser("analyse"); a.add_argument("--out", default="out/objr2"); a.add_argument("--log", default="out/objr2_cluster.log"); a.add_argument("--json", default="out/interp/objr2/baseline.json")
    a.add_argument("--old-ladder", default="out/export/export-20260913T065509Z-5dc2aa41", help="the old ladder's summary export (revision 2) for the comparison table")
    a.add_argument("--no-verify", action="store_true"); a.add_argument("--force", action="store_true"); a.add_argument("--skip-synth", action="store_true"); a.set_defaults(func=cmd_analyse)
    o = sub.add_parser("oldladder"); o.add_argument("--nominal", type=float, nargs="+", default=[4.5, 11.4, 20.0, 30.0]); o.set_defaults(func=cmd_oldladder)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
