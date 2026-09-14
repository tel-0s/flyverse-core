"""Round-3 CONTRAST-MATCHED rectangle ladders (Neurome intake, the family round 2 recorded and never fetched).

One cluster submission, one --arm-block (every job carries the token `fam_rect`, so the shipped lobe and the
`gain_fb = 0` lobe land on ONE box), everything under out/objr3rect/:

  syn/   scripts/probe_synthetic_stimuli.py record  --stimulus rect  on the sphere's arc (elevation 0, +-50 deg at
         40 deg/s, 12 s after 3 s settle): the HEIGHT ladder at width 4.4 deg (heights 2.2 / 4.4 / 8.8 / 15 / 30) and
         the WIDTH ladder at height 8.8 deg (widths 2.2 / 4.4 / 8.8 / 15 / 30), dark (Weber -0.995) and bright
         (+0.995), two lobes (`ship` = the shipped OpticParams, `fb0` = `--optic gain_fb=0`, the deterministic
         control); N_RUNS object runs per rectangle x contrast and N_RUNS blank/blank nulls PER RECTANGLE (its own
         seeds), so the members of a family share no null. The 4.4 x 8.8 rectangle is a member of both ladders
         (height 8.8 at width 4.4 == width 4.4 at height 8.8): 9 distinct rectangles, one set of runs each.

The per-cell recordings (~64 MB per 12-s run) never leave the box: every `record` is followed on the box by `reduce`,
which writes <stem>_reduced.npz -- the per-cell time-mean of both arms (the whole-window statistics), the per-type
population means per frame (the time course), and the SWEEP-LATTICE sums: the rectangle's azimuth is a triangle wave
in angle at constant speed, so every frame sits on one of P = 251 lattice azimuths (0.4 deg apart at 40 deg/s and
10-ms frames), and the sum + count of every cell's value at each lattice azimuth reproduces the time-mean over ANY
azimuth-interval window EXACTLY (the RF-window rule is an azimuth interval x a constant elevation test). `reduce`
checks that identity on random boxes against the raw frames of the same run and records the deviation.

    python scripts/object_round3_rectangles.py plan     [--out out/objr3rect --name objr3rect --minutes 60 --runs 6]
    python scripts/object_round3_rectangles.py submit   [--out out/objr3rect]     # cluster_run.py --arm-block fam --fetch out/objr3rect/
    python scripts/object_round3_rectangles.py run-job  --fam fam_rect --job ship_s0 ...   # on the box (one job = 27 processes)
    python scripts/object_round3_rectangles.py reduce   --raw <raw stem> --out <syn stem>   # on the box, after every record
    python scripts/object_round3_rectangles.py verify   [--out out/objr3rect --log out/objr3rect_cluster.log]
    python scripts/object_round3_rectangles.py analyse  [--out out/objr3rect --json out/interp/objr3rect/rectangles.json]
    python scripts/object_round3_rectangles.py selftest --raw <raw stem>          # the lattice identity on a raw run (CPU)

The reading rules (PREDECLARED) are dumped to out/objr3rect/predeclared.json by `plan` BEFORE submission; `analyse`
copies the stamped file into the Result, checks it against the live dict, and records the sha256 of the analysis code
(this script and every reducer it imports) -- docs/INTERP.md 10.4 items 10 / 11.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
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
import object_round2_baseline as orb                   # noqa: E402  (holm, compare_row, spearman_perm, contrast_perm, contrast_readout, tree_state)
import object_round2_compare as orc                    # noqa: E402  (load_windows: the round-2 window rule over an ordered list of RF maps)
import probe_synthetic_stimuli as pss                  # noqa: E402  (DARK_CONTRAST, footprint helpers)
import probe_object_matched as pom                     # noqa: E402  (footprint)

# ---------------------------------------------------------------------------------------------------- design constants
N_RUNS = 6
FIXED_WIDTH, FIXED_HEIGHT = 4.4, 8.8
HEIGHTS = [2.2, 4.4, 8.8, 15.0, 30.0]                   # the height ladder at width 4.4 (Keles & Frye 2017: peak near 8.8 deg vertical)
WIDTHS = [2.2, 4.4, 8.8, 15.0, 30.0]                    # the width ladder at height 8.8 (Keles & Frye 2017: ~4.4 deg width)
LADDERS = {"hlad": ("height ladder, width 4.4 deg", [(FIXED_WIDTH, h) for h in HEIGHTS], "height_deg"),
           "wlad": ("width ladder, height 8.8 deg", [(w, FIXED_HEIGHT) for w in WIDTHS], "width_deg")}
RECTS: list[tuple[float, float]] = []                   # the distinct (width, height) rectangles, ladder order, 4.4 x 8.8 once
for _lad in LADDERS.values():
    for _wh in _lad[1]:
        if _wh not in RECTS:
            RECTS.append(_wh)
CONTRASTS = {"dark": pss.DARK_CONTRAST, "bright": -pss.DARK_CONTRAST}
LOBES = {"ship": "", "fb0": "--optic gain_fb=0"}
LC_TYPES = ("LC11", "LC10a")
UPSTREAM = ("T2", "T3", "Tm5Y", "TmY21")
SMALL_RUNGS, LARGE_RUNGS = (2.2, 4.4, 8.8), (15.0, 30.0)
TARGET_RUNGS = {"LC11": {"hlad": [8.8], "wlad": [4.4]}, "LC10a": {"hlad": [15.0, 30.0], "wlad": [15.0, 30.0]}}
ALPHA = 0.05
ARC = {"half_span_deg": 50.0, "speed_deg_s": 40.0, "elevation_deg": 0.0}   # the sphere ladder's arc (object_round2_baseline SYN_*)
SECONDS, SETTLE = 12.0, 3.0
FAM_TOKEN = "fam_rect"                                  # the --arm-block key `fam`: one block for the whole family
RF_MAPS = ["out/objr2/rfmap_ship.csv", "out/objr2/rfmap_fb0.csv", "out/synth2/rfmap_150_fb0_p3.csv"]   # the round-2 window rule's maps
LATTICE_DECIMALS = 4
ANALYSIS_SOURCES = ["scripts/object_round3_rectangles.py", "scripts/object_round2_baseline.py", "scripts/object_round2_compare.py",
                    "scripts/probe_synthetic_stimuli.py", "scripts/probe_object_matched.py", "flyverse/interp/common.py", "flyverse/interp/trace.py"]

PREDECLARED = {
    "written": "2026-09-14, before the batch was submitted (out/objr3rect/predeclared.json is stamped by `plan`; the stamp, not this dict or the audit, is the predeclaration)",
    "question": "With retinal contrast matched by construction and height / width varied separately, does any LC population show the animal's "
                "size preference (LC11: peak near height 8.8 deg at width 4.4 and near width 4.4 deg at height 8.8, Keles & Frye 2017; "
                "LC10a: 15-30 deg, Schretter et al. 2024); is the upstream T2 / T3 figure still size-monotone; how does it compare with the sphere ladder",
    "replicate_unit": "one process = one (A, B) pair under one brain seed = one run; a cluster job is a container of sequential processes; runs, not cells, not seeds",
    "n_runs_per_arm": N_RUNS,
    "design": {"ladders": {k: {"label": v[0], "rects_w_h": v[1], "varied": v[2]} for k, v in LADDERS.items()},
               "contrasts": CONTRASTS, "lobes": LOBES, "arc": ARC, "seconds": SECONDS, "settle_s": SETTLE,
               "nulls": "N_RUNS blank/blank runs PER RECTANGLE (their own seeds), windowed with that rectangle: the members of a family share no null run",
               "shared_rung": "the 4.4 x 8.8 rectangle is height 8.8 of the height ladder AND width 4.4 of the width ladder: one set of runs, a member of both families",
               "one_submission": "every arm of every family, both lobes, one cluster_run.py call with --arm-block fam (every job carries fam_rect): one box hosts the whole family"},
    "why_six": "the primary family has 5 members (the rungs; drive only), so Holm's smallest threshold is 0.05 / 5 = 0.01; the exact-U floor is 0.0079 at 5 v 5 "
               "and 0.00216 at 6 v 6 (common.p_floor); six runs per arm keeps a fully separated member clear of the floor with margin, and matches round 2's arm size",
    "window_rule": {"order": ["fitted in out/objr2/rfmap_ship.csv (15-deg square, 3 passes, all 3 runs, z_min 5)", "else fitted in out/objr2/rfmap_fb0.csv",
                              "else fitted in out/synth2/rfmap_150_fb0_p3.csv", "else the anatomical column of trace.column_of_cells with the type's median fitted width over the maps",
                              "else the whole window"],
                    "frames": "frames on which the rectangle overlaps the body's box: |d az| <= (w + width)/2 and |d el| <= (h + height)/2 (object_round2_baseline.box_frames); "
                              "a null run is windowed with the rectangle's width / height on its own (identical) track",
                    "maps": RF_MAPS, "reader": "object_round2_compare.load_windows (the same maps as the round-2 comparison)",
                    "caveat": "no map fits an LC11 body (0 of 143): every LC11 window is an anatomical box, not a measured receptive field; say so beside every LC11 number"},
    "primary": {
        "statistic": "per run: the population MEDIAN over the type's bodies (those with >= 1 window frame) of the per-body RF-windowed time-mean received drive (A - B, mV): `drive_median`",
        "families": "one family per (ladder x contrast x LC type) on the shipped lobe: hlad / wlad x dark / bright x LC11 / LC10a = 8 families of 5 members (the rungs); "
                    "Holm within each family with m = 5; each member = the six object runs at that rung vs the six blank/blank runs of THAT rectangle; "
                    "p = the tie-aware exact permutation U (two-sided, all C(12, 6) assignments); verdict = common.compare (z on the null SD, exact U, p_floor)",
        "spikes": "the per-body windowed spike-count difference median (`spikes_median`) is REPORTED AS A MAGNITUDE OUTSIDE THE FAMILY: it was exactly 0.000 in every run of "
                  "round 2 and a member constant by construction is not a test (docs/INTERP.md 10.2)",
        "LC11": {"expectation": "Keles & Frye 2017: the excess over the null largest at height 8.8 (width ladder: at width 4.4), falling by 15-30 deg",
                 "target_rungs": TARGET_RUNGS["LC11"]},
        "LC10a": {"expectation": "Schretter et al. 2024: the excess largest at 15-30 deg on both ladders", "target_rungs": TARGET_RUNGS["LC10a"]},
        "preference_tests": "per family and statistic: Spearman rho of the per-run statistic against the varied dimension over the 30 object runs (permutation p, 20,000 label "
                            "shuffles, seed 0); small-minus-large = mean at {2.2, 4.4, 8.8} minus mean at {15, 30} (permutation p over run labels); "
                            "peak contrast = mean at the type's target rungs minus mean at the other rungs (permutation p); the null runs' own Spearman against the "
                            "rung (the window widens with the rung) beside each",
        "call": "a size preference is CALLED for a family only if at least one member reads `result` from common.compare AND survives Holm (p_holm <= 0.05); the animal's "
                "SHAPE is shown only if, in addition, the peak contrast is positive with p_perm <= 0.05. Otherwise: 'no detected preference at 6 v 6 on this statistic'",
        "fb0": "the deterministic lobe's blank/blank null has SD ~0, so compare returns `undetermined` or a structural p: its rows are magnitudes (diff with the object runs' "
               "scatter), never tabulated beside a Holm call (docs/INTERP.md 10.2)",
    },
    "secondary_exploratory": {
        "upstream": "T2 / T3 / Tm5Y / TmY21: `diff_signed_best_cell` (max over cells of |mean dr_A - mean dr_B|, a within-run population MAXIMUM), the population "
                    "`diff_signed_mean`, and `diff_abs_best_cell_mean` (max over cells of mean|dr_A| - mean|dr_B|) kept as three statistics, whole-window "
                    "(probe_synthetic_stimuli.family_stats definitions); per rung vs the rectangle's nulls; Holm within each (ladder x contrast x type x statistic) "
                    "5-rung set; all exploratory",
        "companion": "beside every max-over-cells row, in the same table: `abs_drive_median` = the per-body RF-windowed median over the type's windowed bodies of "
                     "|mean_window(dr_A - dr_B)| (rf and anatomical windows) and `abs_drive_median_rf` (fitted bodies only) -- a max-over-cells statistic never "
                     "travels without its per-body median (docs/INTERP.md 10.2)",
        "size_monotone": "per (ladder x contrast x type x statistic): Spearman rho of the per-run value against the varied dimension over the 30 object runs; "
                         "'size-monotone' = rho > 0 with p_perm <= 0.05 on diff_signed_best_cell AND on abs_drive_median",
        "lc_other": "LC11 / LC10a: `drive_mean`, `drive_max` (windowed, over bodies), the whole-window `diff_max_over_cells_mean_mv` (the old headline statistic) and "
                    "`diff_rate_hz_max_cell`, Holm within each 5-rung set, exploratory; the per-body median (the primary) is always in the same table",
    },
    "effective_contrast": "per rectangle from the synthesised radiance (the array handed to FlyBrain.vision): the object's Weber contrast (fixed +-0.995 by construction), "
                          "the per-frame extreme relative luminance change (median over frames), the mean relative change over the changed (> 5 %) set, the maximum column "
                          "coverage over the run, the column-equivalents per frame (sum of coverage; median over frames), the fraction of frames in which no column is "
                          "covered > 5 %, and the columns dimmed / brightened > 50 % per frame. Stated up front: the family is contrast-matched in the object's Weber "
                          "contrast; the per-COLUMN extreme change is capped by coverage where a rectangle is narrower than a column's 4.5-deg acceptance (a 4.4-deg "
                          "width never covers a whole column: max coverage 0.885), and the elevation-0 arc runs through the sparsest patch of this retina "
                          "(docs/audits/object_synthetic_stimuli.md 6.2), so the column-equivalents per rung are the number to read beside every small-rung null",
    "sphere_comparison": "magnitudes only: the round-2 sphere ladder (out/interp/objr2/baseline.json, 6 v 6, B200 / H200 boxes of round 2) beside these rectangles at the "
                         "matching sizes (4.5 ~ 4.4 x 4.4; 8.8 ~ 8.8 x 8.8; 15; 30); two batches on different boxes are never compared row by row (docs/INTERP.md 10.4 item 2)",
    "devices": "the realised device_name of every run is recorded; both lobes are one --arm-block block and the verifier raises a problem if the two lobes' device sets differ",
    "adoption": "nothing is adopted; no default changes; this is a measurement on the shipped model and its deterministic control",
}


# ==================================================================================================== names and plan
def rect_tag(w: float, h: float) -> str:
    return f"w{int(round(w * 10)):03d}_h{int(round(h * 10)):03d}"


def stem_of(lobe: str, w: float, h: float, con: str, seed: int) -> str:
    return f"{lobe}_{rect_tag(w, h)}_{con}_s{seed}"


def parse_stem(stem: str) -> dict | None:
    m = re.match(r"(ship|fb0)_w(\d{3})_h(\d{3})_(dark|bright|null)_s(\d+)$", stem)
    if not m:
        return None
    return {"lobe": m.group(1), "width": int(m.group(2)) / 10.0, "height": int(m.group(3)) / 10.0, "contrast_name": m.group(4),
            "null": m.group(4) == "null", "seed": int(m.group(5))}


def null_seed(rect_index: int, k: int) -> int:
    """Each rectangle's nulls are their own draws: seeds 100 (ri + 1) + k, disjoint from the object runs' 0..N-1."""
    return 100 * (rect_index + 1) + k


def _record_cmd(raw: str, syn: str, lobe: str, w: float, h: float, con: str, seed: int, seconds: float, settle: float, allow_cpu: bool = False) -> tuple[str, str]:
    stem = stem_of(lobe, w, h, con, seed)
    contrast = CONTRASTS["dark"] if con == "null" else CONTRASTS[con]
    flags = LOBES[lobe]
    cmd = (f"python scripts/probe_synthetic_stimuli.py record --stimulus rect --width {w:g} --height {h:g} --contrast {contrast:g} "
           f"--speed {ARC['speed_deg_s']:g} --elevation {ARC['elevation_deg']:g} --half-span {ARC['half_span_deg']:g} --seconds {seconds:g} --settle {settle:g} "
           f"--seed {seed}{' ' + flags if flags else ''}{' --null' if con == 'null' else ''}{' --allow-cpu' if allow_cpu else ''} --out {raw}/{stem} > {syn}/{stem}.txt 2>&1")
    red = f"python scripts/object_round3_rectangles.py reduce --raw {raw}/{stem} --out {syn}/{stem} >> {syn}/{stem}.txt 2>&1"
    return f"{syn}/{stem}", f"{cmd}; st=$?; {red}; st2=$?; test $st -eq 0 && exit $st2 || exit $st"


def _job_line(out: str, raw: str, job: str, a) -> str:
    return (f"mkdir -p {out}/syn {raw} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/object_round3_rectangles.py run-job --fam {FAM_TOKEN} --job {job} --out {out} --raw {raw} --runs {a.runs} "
            f"--seconds {a.seconds:g} --settle {a.settle:g} --seed-offset {getattr(a, 'seed_offset', 0)}")


def build_jobs(out: str, raw: str, a, allow_cpu: bool = False) -> list[dict]:
    """One job per (lobe, seed): the 9 rectangles x 2 contrasts at that seed + the 9 nulls of that seed index = 27 processes."""
    jobs = []
    syn = f"{out}/syn"
    for lobe in LOBES:
        for k in range(a.runs):
            offset = getattr(a, "seed_offset", 0)
            procs = [_record_cmd(raw, syn, lobe, w, h, con, k + offset, a.seconds, a.settle, allow_cpu) for (w, h) in RECTS for con in CONTRASTS]
            procs += [_record_cmd(raw, syn, lobe, w, h, "null", null_seed(ri, k) + offset, a.seconds, a.settle, allow_cpu) for ri, (w, h) in enumerate(RECTS)]
            jobs.append({"job": f"{lobe}_s{k}", "lobe": lobe, "seed": k, "stems": [p[0] for p in procs], "commands": [p[1] for p in procs],
                         "expect": [p[0] + "_summary.json" for p in procs] + [p[0] + "_reduced.npz" for p in procs], "line": _job_line(out, raw, f"{lobe}_s{k}", a)})
    return jobs


def analysis_code_sha256() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in ANALYSIS_SOURCES if (ROOT / f).exists()}


def cmd_plan(args) -> int:
    out = args.out.rstrip("/"); raw = args.raw.rstrip("/")
    Path(out).mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(out, raw, args)
    n_proc = sum(len(j["stems"]) for j in jobs)
    if len(jobs) > 24:
        raise SystemExit(f"{len(jobs)} jobs > the ~24-job cap of one submission on the rented boxes")
    log = f"out/{args.name}_cluster.log"
    sh = ["#!/bin/sh", f"# generated by scripts/object_round3_rectangles.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs, {n_proc} processes, one --arm-block block ({FAM_TOKEN})",
          "set -e", f"mkdir -p {out} out", f'if [ -f {log} ]; then mv {log} "out/{args.name}_cluster.$(date +%Y%m%dT%H%M%S).log"; fi',
          f"python scripts/cluster_run.py --name {args.name} --minutes {args.minutes} --arm-block fam \\"]
    sh += [f"  {shlex.quote(j['line'])} \\" for j in jobs]
    sh += [f"  --fetch {out}/ 2>&1 | tee {log}"]
    (Path(out) / "batch.sh").write_text("\n".join(sh) + "\n", encoding="utf-8")
    with open(Path(out) / "jobs.json", "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "minutes": args.minutes, "runs": args.runs, "seconds": args.seconds, "settle": args.settle, "out": out, "raw": raw,
                   "arm_block": "fam", "block_token": FAM_TOKEN, "rects_w_h": RECTS, "ladders": {k: {"label": v[0], "rects_w_h": v[1], "varied": v[2]} for k, v in LADDERS.items()},
                   "contrasts": CONTRASTS, "lobes": LOBES, "arc": ARC, "jobs": jobs, "n_processes": n_proc, "fetch": out + "/", "log": log}, f, indent=1)
    with open(Path(out) / "predeclared.json", "w", encoding="utf-8") as f:
        json.dump({"predeclared": PREDECLARED, "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generator": " ".join(sys.argv),
                   "analysis_code_sha256_at_stamp": analysis_code_sha256()}, f, indent=1)
    ts = orb.tree_state(); ts["sha256_lf"].update({k: v for k, v in analysis_code_sha256().items()})
    with open(Path(out) / "tree_state.json", "w", encoding="utf-8") as f:
        json.dump(ts, f, indent=1)
    print(f"{len(jobs)} job(s), {n_proc} processes -> {out}/ (raw recordings stay on the box under {raw}/; log {log}); batch.sh, jobs.json, predeclared.json, tree_state.json written")
    for j in jobs:
        print(f"  {j['job']:10s} {len(j['stems']):3d} processes: {Path(j['stems'][0]).name} .. {Path(j['stems'][-1]).name}")
    return 0


def cmd_submit(args) -> int:
    out = args.out.rstrip("/")
    with open(Path(out) / "jobs.json", encoding="utf-8") as f:
        plan = json.load(f)
    lines = [j["line"] for j in plan["jobs"]]
    cmd = [sys.executable, "scripts/cluster_run.py", "--name", plan["name"], "--minutes", str(plan["minutes"]), "--arm-block", plan["arm_block"], *lines, "--fetch", plan["fetch"]]
    cmd[2:2] = [v for key in ("target", "node") if getattr(args, key, None) for v in ("--" + key, getattr(args, key))]
    log = Path(plan["log"])
    if log.exists():
        log.rename(log.with_name(log.stem + time.strftime(".%Y%m%dT%H%M%S") + ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"submitting {len(lines)} job(s) ({plan['n_processes']} processes) through scripts/cluster_run.py --arm-block {plan['arm_block']}; log {log}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=env)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line); f.flush()
        rc = p.wait()
    print(f"cluster_run exit {rc}; log {log}")
    return rc


def cmd_run_job(args) -> int:
    """On the box: the processes of one job in sequence (record, then reduce; a failed process does not stop the next), then the check."""
    jobs = {j["job"]: j for j in build_jobs(args.out.rstrip("/"), args.raw.rstrip("/"), args, allow_cpu=args.allow_cpu)}
    j = jobs[args.job]
    print(f"run-job {args.job} ({args.fam}): {len(j['commands'])} process(es)", flush=True)
    t0 = time.time()
    for stem, cmd in zip(j["stems"], j["commands"]):
        t1 = time.time()
        rc = subprocess.call(cmd, shell=True, cwd=str(ROOT))
        txt = Path(stem + ".txt")
        tail = [l for l in txt.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()][-1:] if txt.exists() else ["(no console)"]
        print(f"  {Path(stem).name:30s} exit {rc} {time.time() - t1:6.0f} s  {tail[0][:160]}", flush=True)
    print(f"run-job {args.job}: {(time.time() - t0) / 60:.1f} min", flush=True)
    args.expect = j["expect"]
    return cmd_check_job(args)


def cmd_check_job(args) -> int:
    """Every expected output exists, every console says `device cuda` (or `device cpu` under --allow-cpu) and `reduced`, no Traceback."""
    bad = []
    want_dev = "device cpu" if getattr(args, "allow_cpu", False) else "device cuda"
    seen = set()
    for e in args.expect:
        if not Path(e).exists():
            bad.append(f"missing {e}")
        stem = re.sub(r"(_summary\.json|_reduced\.npz)$", "", str(e))
        if stem in seen:
            continue
        seen.add(stem)
        txt = Path(stem + ".txt")
        t = txt.read_text(encoding="utf-8", errors="replace") if txt.exists() else ""
        if want_dev not in t:
            bad.append(f"{txt}: no '{want_dev}' line")
        if "reduced " not in t:
            bad.append(f"{txt}: no 'reduced' line")
        if "Traceback" in t:
            bad.append(f"{txt}: Traceback")
    print(f"check-job: {len(args.expect)} expected, {len(bad)} problem(s)" + ("; " + "; ".join(bad) if bad else ""))
    return 1 if bad else 0


# ==================================================================================================== reduce (on the box)
def per_frame_spikes(cum: np.ndarray) -> np.ndarray:
    return np.diff(cum.astype(np.float64), axis=0, prepend=cum[:1].astype(np.float64))


def lattice_of(az: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lattice azimuths (P,), frame -> lattice index (T,), frames per lattice azimuth (P,))."""
    key = np.round(np.asarray(az, np.float64), LATTICE_DECIMALS)
    lat, inv = np.unique(key, return_inverse=True)
    return lat, inv, np.bincount(inv, minlength=len(lat)).astype(np.int64)


def lattice_sums(x: np.ndarray, inv: np.ndarray, P: int) -> np.ndarray:
    """(P, n) sums of x (T, n) over the frames at each lattice azimuth (float64)."""
    out = np.zeros((P, x.shape[1]), np.float64)
    np.add.at(out, inv, np.asarray(x, np.float64))
    return out


def window_mean_from_lattice(sums: np.ndarray, counts: np.ndarray, lat: np.ndarray, az_b: float, half: float) -> tuple[np.ndarray, int]:
    """The time-mean over the frames with |az - az_b| <= half, from the lattice sums: exact for any azimuth interval."""
    m = np.abs(lat - az_b) <= half
    n = int(counts[m].sum())
    return (sums[m].sum(0) / n if n else np.full(sums.shape[1], np.nan)), n


def reduce_run(raw_stem: str, out_stem: str, n_check: int = 40, seed: int = 0) -> dict:
    """<raw>_{stim|blank_a}.npz / _{blank|blank_b}.npz + <raw>_radiance.npz -> <out>_reduced.npz (+ the summary / prov /
    radiance copied beside it). Returns the check record."""
    sm = json.load(open(raw_stem + "_summary.json", encoding="utf-8"))
    a_name, b_name = sm["arms"][0], sm["arms"][1]
    A = common.Recording.load(Path(raw_stem + f"_{a_name}.npz")); B = common.Recording.load(Path(raw_stem + f"_{b_name}.npz"))
    rad = np.load(raw_stem + "_radiance.npz", allow_pickle=False)
    T = min(A.n_frames, B.n_frames, len(rad["track__az_deg"]))
    az = np.asarray(rad["track__az_deg"][:T], np.float64); el = np.asarray(rad["track__el_deg"][:T], np.float64)
    lat, inv, counts = lattice_of(az)
    P = len(lat)
    types = A.types.astype(str); body_ids = A.body_ids.astype(np.int64)
    is_lc = np.isin(types, LC_TYPES)
    dr_a = A.quantities["optic_dr"][:T].astype(np.float64); dr_b = B.quantities["optic_dr"][:T].astype(np.float64)
    dv_a = A.quantities["drive_mv"][:T].astype(np.float64); dv_b = B.quantities["drive_mv"][:T].astype(np.float64)
    sp_a = per_frame_spikes(A.quantities["spike_count"][:T]); sp_b = per_frame_spikes(B.quantities["spike_count"][:T])
    rate_ok = np.isfinite(dr_a).all(0) & np.isfinite(dr_b).all(0)          # the rate units (optic_dr defined); LC types are NaN there
    up = np.flatnonzero(rate_ok & ~is_lc); lc = np.flatnonzero(is_lc)
    win_s = T * common.FRAME_MS / 1000.0
    # whole-window per-cell means (family_stats' inputs) and totals
    cell = {"mean_drive_a": dv_a.mean(0), "mean_drive_b": dv_b.mean(0), "mean_dr_a": dr_a.mean(0), "mean_dr_b": dr_b.mean(0),
            "mean_drabs_a": np.abs(dr_a).mean(0), "mean_drabs_b": np.abs(dr_b).mean(0), "spikes_a": sp_a.sum(0), "spikes_b": sp_b.sum(0)}
    # per-type population means per frame (time courses), the type's native quantity
    tkeys = sorted(set(types)); pop_a = np.zeros((T, len(tkeys)), np.float32); pop_b = np.zeros((T, len(tkeys)), np.float32)
    for j, t in enumerate(tkeys):
        m = types == t
        if t in LC_TYPES or not rate_ok[m].all():
            pop_a[:, j] = dv_a[:, m].mean(1); pop_b[:, j] = dv_b[:, m].mean(1)
        else:
            pop_a[:, j] = dr_a[:, m].mean(1); pop_b[:, j] = dr_b[:, m].mean(1)
    # the sweep-lattice sums: LC drive and spikes (both arms, float64), upstream optic_dr difference (float32)
    lat_drive_a = lattice_sums(dv_a[:, lc], inv, P); lat_drive_b = lattice_sums(dv_b[:, lc], inv, P)
    lat_spk_a = lattice_sums(sp_a[:, lc], inv, P); lat_spk_b = lattice_sums(sp_b[:, lc], inv, P)
    lat_dr_diff = lattice_sums(dr_a[:, up] - dr_b[:, up], inv, P).astype(np.float32)
    # the identity check on random boxes against the raw frames
    rng = np.random.RandomState(seed); dev = []
    for _ in range(n_check):
        az_b = rng.uniform(az.min() - 10, az.max() + 10); half = rng.uniform(1.0, 40.0)
        m = np.abs(az - az_b) <= half
        if not m.any():
            continue
        cl = rng.randint(len(lc)); cu = rng.randint(len(up)) if len(up) else None
        v, n = window_mean_from_lattice(lat_drive_a[:, [cl]] - lat_drive_b[:, [cl]], counts, lat, az_b, half)
        dev.append(abs(float(v[0]) - float((dv_a[m, lc[cl]] - dv_b[m, lc[cl]]).mean()))); dev.append(abs(n - int(m.sum())))
        v, _ = window_mean_from_lattice(lat_spk_a[:, [cl]] - lat_spk_b[:, [cl]], counts, lat, az_b, half)
        dev.append(abs(float(v[0]) * n - float((sp_a[m, lc[cl]] - sp_b[m, lc[cl]]).sum())))
        if cu is not None:
            v, _ = window_mean_from_lattice(lat_dr_diff[:, [cu]].astype(np.float64), counts, lat, az_b, half)
            dev.append(abs(float(v[0]) - float((dr_a[m, up[cu]] - dr_b[m, up[cu]]).mean())))
    check = {"n_boxes": n_check, "max_abs_dev": float(max(dev)) if dev else float("nan"), "n_lattice": int(P), "lattice_step_deg": float(np.median(np.diff(lat))) if P > 1 else float("nan"),
             "n_frames": int(T), "n_lc": int(len(lc)), "n_upstream": int(len(up)), "frames_per_lattice_min_max": [int(counts.min()), int(counts.max())]}
    out = Path(out_stem); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + "_reduced.npz", t_ms=A.t_ms[:T], body_ids=body_ids, types=types, idx=A.idx, az_deg=az, el_deg=el, lattice_az=lat, lattice_count=counts,
                        lc_idx=lc, up_idx=up, lat_drive_a=lat_drive_a, lat_drive_b=lat_drive_b, lat_spk_a=lat_spk_a, lat_spk_b=lat_spk_b, lat_dr_diff=lat_dr_diff,
                        pop_types=np.array(tkeys), pop_a=pop_a, pop_b=pop_b, win_s=np.float64(win_s), arms=np.array([a_name, b_name]),
                        width_deg=np.float64(sm["params"]["width_deg"]), height_deg=np.float64(sm["params"]["height_deg"]), contrast=np.float64(sm["params"]["contrast"]),
                        null=np.bool_(sm["arms"][0] == "blank_a"), check=np.str_(json.dumps(check)), **{f"cell__{k}": v.astype(np.float64) for k, v in cell.items()})
    for suf in ("_summary.json", "_prov.json", "_radiance.npz"):
        if Path(raw_stem + suf).exists() and Path(raw_stem + suf).resolve() != Path(str(out) + suf).resolve():
            shutil.copyfile(raw_stem + suf, str(out) + suf)
    return check


def cmd_reduce(args) -> int:
    t0 = time.time()
    check = reduce_run(args.raw, args.out, args.n_check)
    tol = 1e-6
    ok = np.isfinite(check["max_abs_dev"]) and check["max_abs_dev"] <= tol
    size = Path(args.out + "_reduced.npz").stat().st_size / 1e6
    print(f"reduced {args.out}_reduced.npz ({size:.1f} MB): lattice {check['n_lattice']} azimuths x {check['n_frames']} frames, {check['n_lc']} LC + {check['n_upstream']} upstream cells; "
          f"identity check on {check['n_boxes']} random boxes max |dev| {check['max_abs_dev']:.2e} ({'ok' if ok else 'FAILED'}); {time.time() - t0:.0f} s", flush=True)
    return 0 if ok else 1


def cmd_selftest(args) -> int:
    """The lattice identity against object_round2_baseline.windowed_body_stats on a raw run, for every LC body under the
    real window rule (CPU; the smoke's validation of the reduction)."""
    win, _ = orc.load_windows(args.rf_maps)
    tmp = Path(args.tmp); tmp.mkdir(parents=True, exist_ok=True)
    stem = str(tmp / Path(args.raw).name)
    check = reduce_run(args.raw, stem)
    z = np.load(stem + "_reduced.npz", allow_pickle=False)
    sm = json.load(open(args.raw + "_summary.json", encoding="utf-8")); a_name, b_name = sm["arms"]
    A = common.Recording.load(Path(args.raw + f"_{a_name}.npz")); B = common.Recording.load(Path(args.raw + f"_{b_name}.npz"))
    T = int(z["t_ms"].shape[0])
    sa = per_frame_spikes(A.quantities["spike_count"][:T]); sb = per_frame_spikes(B.quantities["spike_count"][:T])
    w, h = float(z["width_deg"]), float(z["height_deg"])
    direct = orb.windowed_body_stats(A.quantities["drive_mv"][:T], B.quantities["drive_mv"][:T], sa, sb, A.body_ids, A.types.astype(str), win, z["az_deg"], z["el_deg"], w, h, LC_TYPES)
    zero = np.zeros((T, len(A.idx)))
    direct_up = orb.windowed_body_stats(A.quantities["optic_dr"][:T], B.quantities["optic_dr"][:T], zero, zero, A.body_ids, A.types.astype(str), win, z["az_deg"], z["el_deg"], w, h, UPSTREAM)
    mine = per_body_from_reduced(z, win, w, h)
    d = direct.merge(mine[mine.type.isin(LC_TYPES)], on="bodyId", suffixes=("", "_lat"))
    du = direct_up.merge(mine[mine.type.isin(UPSTREAM)], on="bodyId", suffixes=("", "_lat"))
    devs = {"lc_drive_diff_win": float(np.nanmax(np.abs(d.drive_diff_win - d.drive_diff_win_lat))), "lc_n_frames_win": int(np.abs(d.n_frames_win - d.n_frames_win_lat).max()),
            "lc_spikes_diff_win": float(np.nanmax(np.abs(d.spikes_diff_win - d.spikes_diff_win_lat))), "lc_drive_a_win": float(np.nanmax(np.abs(d.drive_a_win - d.drive_a_win_lat))),
            "up_drive_diff_win": float(np.nanmax(np.abs(du.drive_diff_win - du.drive_diff_win_lat))), "up_n_frames_win": int(np.abs(du.n_frames_win - du.n_frames_win_lat).max()),
            "n_lc_bodies": int(len(d)), "n_up_bodies": int(len(du)), "reduce_check": check}
    ok = devs["lc_drive_diff_win"] < 1e-9 and devs["lc_n_frames_win"] == 0 and devs["lc_spikes_diff_win"] < 1e-9 and devs["up_drive_diff_win"] < 1e-6 and devs["up_n_frames_win"] == 0
    print(json.dumps(common.to_jsonable(devs), indent=1)); print(f"selftest {'PASSED' if ok else 'FAILED'}: lattice-reconstructed windows vs per-frame windows on {args.raw}")
    return 0 if ok else 1


# ==================================================================================================== per-body statistics from the reduction
def _window_masks(z, win: pd.DataFrame, sel: np.ndarray, obj_w: float, obj_h: float) -> tuple[np.ndarray, list, list]:
    """(n_sel, P) lattice masks of the selected cells' windows under the round-2 rule (object_round2_baseline.box_frames
    on the lattice azimuths; the elevation test is constant along the arc), their window sources and bodyIds."""
    lat = z["lattice_az"]; el_c = float(np.median(z["el_deg"]))
    bodies = pd.DataFrame({"bodyId": [str(int(b)) for b in z["body_ids"][sel]]})
    w = bodies.merge(win[["bodyId", "window_source", "az_deg", "el_deg", "width_deg", "height_deg"]], on="bodyId", how="left")
    src = w.window_source.fillna("whole").astype(str).to_numpy()
    boxed = src != "whole"
    az_b = w.az_deg.to_numpy(float); el_b = w.el_deg.to_numpy(float); wb = w.width_deg.to_numpy(float); hb = w.height_deg.to_numpy(float)
    M = np.ones((len(sel), len(lat)), bool)
    with np.errstate(invalid="ignore"):
        el_ok = np.abs(el_c - el_b) <= (hb + obj_h) / 2
        Mb = np.abs(lat[None, :] - az_b[:, None]) <= ((wb + obj_w) / 2)[:, None]
    M[boxed] = Mb[boxed] & el_ok[boxed, None]
    return M, src.tolist(), bodies.bodyId.tolist()


def per_body_from_reduced(z, win: pd.DataFrame, obj_w: float, obj_h: float) -> pd.DataFrame:
    """Per body of the LC and upstream types: the window by the round-2 rule and its lattice-reconstructed time-means
    (object_round2_baseline.windowed_body_stats' columns, from the lattice sums), vectorised over bodies."""
    counts = z["lattice_count"].astype(np.float64); types = z["types"].astype(str)
    n_total = int(counts.sum())
    frames = []
    for kind, sel in (("lc", z["lc_idx"]), ("up", z["up_idx"])):
        keep = np.isin(types[sel], LC_TYPES if kind == "lc" else UPSTREAM)
        sel = sel[keep]
        if not len(sel):
            continue
        M, src, bid = _window_masks(z, win, sel, obj_w, obj_h)
        n = M @ counts                                                     # frames in the window per body
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "lc":
                pos = np.flatnonzero(keep)                                  # columns of the lattice arrays for the kept LC cells
                A = z["lat_drive_a"][:, pos]; B = z["lat_drive_b"][:, pos]; KA = z["lat_spk_a"][:, pos]; KB = z["lat_spk_b"][:, pos]
                sa = (M.T * A).sum(0); sb = (M.T * B).sum(0); ka = (M.T * KA).sum(0); kb = (M.T * KB).sum(0)
                df = pd.DataFrame({"bodyId": bid, "type": types[sel], "window_source": src, "n_frames_win": n.astype(int), "n_frames": n_total,
                                   "drive_a_win": np.where(n > 0, sa / n, np.nan), "drive_b_win": np.where(n > 0, sb / n, np.nan), "drive_diff_win": np.where(n > 0, (sa - sb) / n, np.nan),
                                   "spikes_a_win": np.where(n > 0, ka, np.nan), "spikes_b_win": np.where(n > 0, kb, np.nan), "spikes_diff_win": np.where(n > 0, ka - kb, np.nan),
                                   "drive_diff_whole": (A.sum(0) - B.sum(0)) / n_total, "spikes_a_whole": KA.sum(0), "spikes_b_whole": KB.sum(0)})
            else:
                pos = np.flatnonzero(keep)
                D = z["lat_dr_diff"][:, pos].astype(np.float64)
                s = (M.T * D).sum(0)
                df = pd.DataFrame({"bodyId": bid, "type": types[sel], "window_source": src, "n_frames_win": n.astype(int), "n_frames": n_total,
                                   "drive_a_win": np.nan, "drive_b_win": np.nan, "drive_diff_win": np.where(n > 0, s / n, np.nan),
                                   "spikes_a_win": 0.0, "spikes_b_win": 0.0, "spikes_diff_win": 0.0, "drive_diff_whole": D.sum(0) / n_total, "spikes_a_whole": 0.0, "spikes_b_whole": 0.0})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


_COMBOS: dict = {}


def tie_aware_exact_u(a, b) -> dict:
    """object_round2_baseline.tie_aware_exact_u (every C(n + m, n) assignment, average ranks for ties, two-sided), the
    enumeration vectorised: identical U and p (checked on the smoke), ~100x faster per row."""
    from itertools import combinations
    from math import comb
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float); n, m = len(a), len(b)
    if n == 0 or m == 0:
        return {"U": float("nan"), "p_tie_exact": float("nan"), "n_tied": 0, "enumerated": 0}
    pooled = np.concatenate([a, b]); r = rankdata(pooled)
    u_obs = r[:n].sum() - n * (n + 1) / 2; centre = n * m / 2
    n_tied = int(len(pooled) - len(np.unique(pooled))); total = comb(n + m, n)
    if total > 200000:
        return orb.tie_aware_exact_u(a, b)
    if (n, m) not in _COMBOS:
        _COMBOS[(n, m)] = np.array(list(combinations(range(n + m), n)), dtype=np.int64)
    idx = _COMBOS[(n, m)]
    u = r[idx].sum(1) - n * (n + 1) / 2
    cnt = int((np.abs(u - centre) >= abs(u_obs - centre) - 1e-9).sum())
    return {"U": float(u_obs), "p_tie_exact": float(cnt / total), "n_tied": n_tied, "enumerated": int(total)}


def compare_row(stim, null, label: dict) -> dict:
    """object_round2_baseline.compare_row (common.compare's verdict + the tie-aware exact U on the same values) with the
    vectorised enumeration."""
    stim = [float(v) for v in stim if np.isfinite(v)]; null = [float(v) for v in null if np.isfinite(v)]
    if stim and null:
        c = common.compare(stim, null)
    else:
        a = common.ArmStats.of(stim) if stim else None; b = common.ArmStats.of(null) if null else None
        c = {"stim": a.record() if a else {"n": 0, "mean": np.nan, "sd": np.nan, "values": []}, "null": b.record() if b else {"n": 0, "mean": np.nan, "sd": np.nan, "values": []},
             "diff": np.nan, "z": np.nan, "welch": np.nan, "U": np.nan, "p": np.nan, "verdict": "underpowered", "p_floor": np.nan, "null_sd_zero": False}
    t = tie_aware_exact_u(stim, null)
    return dict(label, stim_n=c["stim"]["n"], stim_mean=c["stim"]["mean"], stim_sd=c["stim"]["sd"], stim_values=c["stim"]["values"],
                null_n=c["null"]["n"], null_mean=c["null"]["mean"], null_sd=c["null"]["sd"], null_values=c["null"]["values"],
                diff=c["diff"], z=c["z"], welch=c["welch"], U=c["U"], p_scipy_exact=c["p"], p_floor=c["p_floor"], null_sd_zero=c["null_sd_zero"],
                verdict=c["verdict"], U_tie=t["U"], p=t["p_tie_exact"], n_tied=t["n_tied"], p_method="tie-aware exact permutation U (two-sided)")


def contrast_perm(vals_small, vals_large, n_perm: int = 20000, seed: int = 0) -> dict:
    """object_round2_baseline.contrast_perm, vectorised (the same RandomState permutation sequence and count)."""
    a = np.asarray(vals_small, float); b = np.asarray(vals_large, float); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"contrast": np.nan, "p_perm": np.nan}
    obs = a.mean() - b.mean(); pooled = np.concatenate([a, b]); rng = np.random.RandomState(seed)
    perms = np.stack([rng.permutation(len(pooled)) for _ in range(n_perm)])
    P = pooled[perms]
    d = P[:, :len(a)].mean(1) - P[:, len(a):].mean(1)
    cnt = int((np.abs(d) >= abs(obs) - 1e-12).sum())
    return {"contrast": float(obs), "p_perm": float((cnt + 1) / (n_perm + 1)), "n_small": int(len(a)), "n_large": int(len(b))}


def whole_window_stats(z) -> dict:
    """probe_synthetic_stimuli.family_stats' per-type statistics from the reduced per-cell means (the three upstream
    statistics kept apart; the LC max-over-cells headline and the best-cell rate)."""
    types = z["types"].astype(str); win_s = float(z["win_s"])
    out = {}
    for t in sorted(set(types)):
        m = types == t
        if t in LC_TYPES:
            d = z["cell__mean_drive_a"][m] - z["cell__mean_drive_b"][m]; hz = (z["cell__spikes_a"][m] - z["cell__spikes_b"][m]) / win_s
            out[t] = {"kind": "spiking", "n_cells": int(m.sum()), "diff_max_over_cells_mean_mv": float(d.max()), "diff_mean_over_cells_mean_mv": float(d.mean()),
                      "diff_rate_hz_max_cell": float(hz.max()), "diff_rate_hz_mean": float(hz.mean()), "rate_hz_mean_a": float(z["cell__spikes_a"][m].mean() / win_s),
                      "rate_hz_mean_b": float(z["cell__spikes_b"][m].mean() / win_s), "cells_firing_a": int((z["cell__spikes_a"][m] > 0).sum()), "cells_firing_b": int((z["cell__spikes_b"][m] > 0).sum()),
                      "drive_mean_a_mv": float(z["cell__mean_drive_a"][m].mean()), "drive_mean_b_mv": float(z["cell__mean_drive_b"][m].mean())}
        else:
            d = z["cell__mean_dr_a"][m] - z["cell__mean_dr_b"][m]; da = z["cell__mean_drabs_a"][m] - z["cell__mean_drabs_b"][m]
            out[t] = {"kind": "rate", "n_cells": int(m.sum()), "diff_signed_best_cell": float(np.abs(d).max()), "diff_signed_mean": float(d.mean()),
                      "diff_abs_best_cell_mean": float(da.max()), "diff_abs_mean": float(da.mean()), "dev_mean_a": float(z["cell__mean_dr_a"][m].mean()),
                      "dev_mean_b": float(z["cell__mean_dr_b"][m].mean()), "dev_abs_mean_b": float(z["cell__mean_drabs_b"][m].mean())}
    return out


def effective_contrast(rad, contrast: float, change_thresh: float = 0.05) -> dict:
    """From the synthesised radiance of an object run: coverage = (r / bg - 1) / contrast per column and frame (channel 0;
    every channel is the same by construction), the column-equivalents per frame, the maximum coverage, the frames with
    no column above `change_thresh`, and object_round2_baseline.contrast_readout / probe_object_matched.footprint."""
    P = np.asarray(rad["radiance"], np.float64); bg = np.asarray(rad["blank"], np.float64)
    cov = (P[:, :, 0] / bg[None, :, 0] - 1.0) / contrast
    ch = np.abs(cov) > change_thresh
    fp = pom.footprint(P, np.broadcast_to(bg[None], P.shape), rad["col_az_el"])["summary"]
    cr = orb.contrast_readout(P, np.broadcast_to(bg[None], P.shape))
    spread = float(np.abs(P[:, :, 1:] / bg[None, :, 1:] - P[:, :, :1] / bg[None, :, :1]).max())
    return {"weber_contrast": float(contrast), "max_coverage": float(cov.max()), "col_equiv_median": float(np.median(cov.sum(1))), "col_equiv_mean": float(cov.sum(1).mean()),
            "col_equiv_max": float(cov.sum(1).max()), "frames_no_column_5pct_frac": float((~ch.any(1)).mean()), "n_cols_5pct_mean": float(ch.sum(1).mean()),
            "n_cols_50pct_mean": float((np.abs(cov) > 0.5).sum(1).mean()), "n_cols_full_mean": float((cov >= 0.999).sum(1).mean()),
            "channel_spread": spread, "extreme_rel_change_median": cr["extreme_rel_change_median"], "mean_rel_change_over_changed_set": cr["mean_rel_change_over_changed_set"],
            "n_dimmed_50pct_mean": fp["n_dimmed_50pct_mean"], "n_brightened_50pct_mean": fp["n_brightened_50pct_mean"], "n_changed_5pct_mean": fp["n_changed_5pct_mean"],
            "centroid_el_mean_deg": fp["centroid_el_mean_deg"], "band_lo": fp["dimmed_el_band_deg"][0], "band_hi": fp["dimmed_el_band_deg"][1]}


# ==================================================================================================== verify (CPU)
def verify_all(out: str, log: str | None) -> dict:
    rep = {"problems": [], "log": None, "expected_missing": [], "runs": [], "devices_by_lobe": {}, "n_runs": 0, "n_consoles": 0, "reduce_check_max_dev": None}
    if log and Path(log).exists():
        lines = [l for l in Path(log).read_text(encoding="utf-8", errors="replace").splitlines() if "job(s)" in l and "failed" in l]
        rep["log"] = lines[-1] if lines else None
        if not lines or ", 0 failed" not in lines[-1]:
            rep["problems"].append(f"cluster log: {rep['log']!r}")
    else:
        rep["problems"].append(f"cluster log: {log!r} missing")
    plan_p = Path(out) / "jobs.json"
    if plan_p.exists():
        plan = json.load(open(plan_p, encoding="utf-8"))
        for j in plan["jobs"]:
            for e in j["expect"]:
                if not Path(e).exists():
                    rep["expected_missing"].append(e)
        if rep["expected_missing"]:
            rep["problems"].append(f"{len(rep['expected_missing'])} expected outputs missing")
    syn = f"{out}/syn"
    devs = {}
    max_dev = 0.0
    for sf in sorted(glob.glob(f"{syn}/*_summary.json")):
        stem = sf[:-len("_summary.json")]; name = Path(stem).name; nm = parse_stem(name)
        row = {"run": name, "ok": True}
        if nm is None:
            rep["problems"].append(f"{name}: unexpected file name"); row["ok"] = False; rep["runs"].append(row); continue
        sm = json.load(open(sf, encoding="utf-8"))
        pv = Path(stem + "_prov.json"); prov = json.load(open(pv, encoding="utf-8")) if pv.exists() else {}
        txt = Path(stem + ".txt"); console = txt.read_text(encoding="utf-8", errors="replace") if txt.exists() else ""
        red = Path(stem + "_reduced.npz")
        probs = []
        if sm.get("device") != "cuda":
            probs.append(f"summary device {sm.get('device')}")
        if (prov.get("execution") or {}).get("device") != "cuda":
            probs.append(f"prov device {(prov.get('execution') or {}).get('device')}")
        if not txt.exists():
            probs.append("no console")
        else:
            rep["n_consoles"] += 1
            if "device cuda" not in console:
                probs.append("console without 'device cuda'")
            if "reduced " not in console:
                probs.append("console without 'reduced'")
            if "Traceback" in console:
                probs.append("Traceback in console")
        if not all(v is True for v in (sm.get("checksum_per_arm") or {}).values()) or not sm.get("checksum_per_arm"):
            probs.append(f"checksums {sm.get('checksum_per_arm')}")
        if (sm["arms"] == ["blank_a", "blank_b"]) != nm["null"]:
            probs.append(f"arms {sm['arms']} vs name")
        if sm.get("seed") != nm["seed"]:
            probs.append(f"seed {sm.get('seed')} vs name")
        p = sm["params"]
        if abs(float(p["width_deg"]) - nm["width"]) > 1e-6 or abs(float(p["height_deg"]) - nm["height"]) > 1e-6:
            probs.append(f"rect {p['width_deg']}x{p['height_deg']} vs name")
        if not nm["null"] and abs(float(p["contrast"]) - CONTRASTS[nm["contrast_name"]]) > 1e-9:
            probs.append(f"contrast {p['contrast']} vs {nm['contrast_name']}")
        for k, v in (("speed_deg_s", ARC["speed_deg_s"]), ("elevation_deg", ARC["elevation_deg"]), ("half_span_deg", ARC["half_span_deg"])):
            if abs(float(p[k]) - v) > 1e-9:
                probs.append(f"{k} {p[k]} vs {v}")
        ov = sm.get("optic_overrides") or {}
        if (nm["lobe"] == "fb0") != (ov.get("gain_fb") == 0) or (nm["lobe"] == "ship" and ov):
            probs.append(f"optic overrides {ov} vs lobe {nm['lobe']}")
        gfb = ((prov.get("model") or {}).get("optic") or {}).get("gain_fb")
        if prov and ((nm["lobe"] == "fb0") != (gfb == 0)):
            probs.append(f"resolved gain_fb {gfb} vs lobe")
        if (sm.get("hook_info") or {}).get("active"):
            probs.append("hook_info.active true on a plain run")
        if not red.exists():
            probs.append("no _reduced.npz")
        else:
            try:
                z = np.load(red, allow_pickle=False); chk = json.loads(str(z["check"]))
                max_dev = max(max_dev, float(chk["max_abs_dev"]))
                if not (np.isfinite(chk["max_abs_dev"]) and chk["max_abs_dev"] <= 1e-6):
                    probs.append(f"reduce identity check {chk['max_abs_dev']}")
                if abs(float(z["width_deg"]) - nm["width"]) > 1e-6 or abs(float(z["height_deg"]) - nm["height"]) > 1e-6 or bool(z["null"]) != nm["null"]:
                    probs.append("reduced npz params vs name")
                if int(z["t_ms"].shape[0]) != int(sm["n_frames"]):
                    probs.append(f"reduced frames {z['t_ms'].shape[0]} vs summary {sm['n_frames']}")
                row["n_lattice"] = int(len(z["lattice_az"])); row["n_frames"] = int(z["t_ms"].shape[0])
            except Exception as e:  # noqa: BLE001
                probs.append(f"reduced npz unreadable: {e!r}")
        row.update(lobe=nm["lobe"], null=nm["null"], device_name=(prov.get("execution") or {}).get("device_name"), problems=probs, ok=not probs,
                   wall_s=sm.get("wall_s"), fps=sm.get("fps"))
        devs.setdefault(nm["lobe"], set()).add(str(row["device_name"]))
        if probs:
            rep["problems"].append(f"{name}: " + "; ".join(probs))
        rep["runs"].append(row)
    rep["n_runs"] = len(rep["runs"]); rep["reduce_check_max_dev"] = max_dev
    rep["devices_by_lobe"] = {k: sorted(v) for k, v in devs.items()}
    if rep["n_consoles"] != rep["n_runs"]:
        rep["problems"].append(f"consoles {rep['n_consoles']} != runs {rep['n_runs']}")
    if len(devs) == 2 and devs["ship"] != devs["fb0"]:
        rep["problems"].append(f"lobes on different devices: {rep['devices_by_lobe']} (the block did not hold)")
    if any("None" in v or len(v) != 1 for v in devs.values()):
        rep["problems"].append(f"device set per lobe not a single known GPU: {rep['devices_by_lobe']}")
    return rep


def cmd_verify(args) -> int:
    rep = verify_all(args.out.rstrip("/"), args.log)
    print(f"cluster log: {rep['log']}")
    print(f"runs: {rep['n_runs']} ({rep['n_consoles']} consoles); devices per lobe {rep['devices_by_lobe']}; reduce identity max |dev| {rep['reduce_check_max_dev']:.2e}")
    print(f"expected outputs missing: {len(rep['expected_missing'])}")
    print("problems: " + ("; ".join(rep["problems"][:30]) if rep["problems"] else "none") + (f" ... ({len(rep['problems'])} total)" if len(rep["problems"]) > 30 else ""))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(common.to_jsonable(dict(rep, generator=" ".join(sys.argv))), f, indent=1)
        print(f"written {args.json}")
    return 1 if rep["problems"] else 0


# ==================================================================================================== analyse (CPU)
def _holm_family(rows: list[dict], m: int | None = None) -> list[dict]:
    ph = orb.holm([c["p"] for c in rows], m=m)
    for c, h in zip(rows, ph):
        c["p_holm"] = float(h); c["survives_holm"] = bool(np.isfinite(h) and h <= ALPHA and c["verdict"] == "result")
    return rows


def _rung_of(w: float, h: float, ladder: str) -> float | None:
    if ladder == "hlad" and abs(w - FIXED_WIDTH) < 1e-6 and any(abs(h - x) < 1e-6 for x in HEIGHTS):
        return float(h)
    if ladder == "wlad" and abs(h - FIXED_HEIGHT) < 1e-6 and any(abs(w - x) < 1e-6 for x in WIDTHS):
        return float(w)
    return None


def load_reduced_runs(out: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(f"{out}/syn/*_reduced.npz")):
        stem = p[:-len("_reduced.npz")]; nm = parse_stem(Path(stem).name)
        if nm is None:
            continue
        rows.append(dict(nm, stem=stem, run=Path(stem).name))
    return pd.DataFrame(rows)


def analyse(out: str, win: pd.DataFrame, sphere_json: str | None) -> dict:
    runs = load_reduced_runs(out)
    if not len(runs):
        raise SystemExit(f"no *_reduced.npz under {out}/syn")
    per_body_rows, per_run_rows, up_rows, fp_rows = [], [], [], []
    prov0 = None
    for _, r in runs.iterrows():
        z = np.load(r.stem + "_reduced.npz", allow_pickle=False)
        sm = json.load(open(r.stem + "_summary.json", encoding="utf-8"))
        pv = Path(r.stem + "_prov.json"); prov = json.load(open(pv, encoding="utf-8")) if pv.exists() else {}
        if prov0 is None and prov and r.lobe == "ship":
            prov0 = prov
        dev = (prov.get("execution") or {}).get("device_name")
        pb = per_body_from_reduced(z, win, r.width, r.height)
        pb.insert(0, "run", r.run); pb.insert(1, "lobe", r.lobe); pb.insert(2, "null", bool(r.null)); pb.insert(3, "width_deg", r.width); pb.insert(4, "height_deg", r.height)
        pb.insert(5, "contrast_name", r.contrast_name); pb.insert(6, "seed", int(r.seed))
        per_body_rows.append(pb)
        ww = whole_window_stats(z)
        for t, v in orb.pop_stats(pb[pb.type.isin(LC_TYPES)]).items():
            per_run_rows.append(dict(run=r.run, lobe=r.lobe, null=bool(r.null), width_deg=r.width, height_deg=r.height, contrast_name=r.contrast_name, seed=int(r.seed), type=t, **v,
                                     diff_max_over_cells_mean_mv=ww[t]["diff_max_over_cells_mean_mv"], diff_rate_hz_max_cell=ww[t]["diff_rate_hz_max_cell"],
                                     rate_hz_mean_a=ww[t]["rate_hz_mean_a"], rate_hz_mean_b=ww[t]["rate_hz_mean_b"], drive_mean_b_mv=ww[t]["drive_mean_b_mv"], device=dev))
        for t in UPSTREAM:
            g = pb[(pb.type == t) & (pb.n_frames_win > 0)]; grf = g[g.window_source.str.startswith("rf")]
            up_rows.append(dict(run=r.run, lobe=r.lobe, null=bool(r.null), width_deg=r.width, height_deg=r.height, contrast_name=r.contrast_name, seed=int(r.seed), type=t,
                                diff_signed_best_cell=ww[t]["diff_signed_best_cell"], diff_signed_mean=ww[t]["diff_signed_mean"], diff_abs_best_cell_mean=ww[t]["diff_abs_best_cell_mean"],
                                abs_drive_median=float(g.drive_diff_win.abs().median()) if len(g) else np.nan, abs_drive_median_rf=float(grf.drive_diff_win.abs().median()) if len(grf) else np.nan,
                                drive_median_signed=float(g.drive_diff_win.median()) if len(g) else np.nan, n_bodies_windowed=int(len(g)), n_bodies_rf=int(len(grf)),
                                dev_mean_b=ww[t]["dev_mean_b"], dev_abs_mean_b=ww[t]["dev_abs_mean_b"], device=dev))
        if not r.null:
            rad = np.load(r.stem + "_radiance.npz", allow_pickle=False)
            fp_rows.append(dict(run=r.run, lobe=r.lobe, width_deg=r.width, height_deg=r.height, contrast_name=r.contrast_name, seed=int(r.seed),
                                **effective_contrast(rad, float(z["contrast"]))))
    per_body = pd.concat(per_body_rows, ignore_index=True); per_run = pd.DataFrame(per_run_rows); up = pd.DataFrame(up_rows); fp = pd.DataFrame(fp_rows)
    for df in (per_run, up):
        for lad in LADDERS:
            df[f"rung_{lad}"] = [_rung_of(w, h, lad) for w, h in zip(df.width_deg, df.height_deg)]
    # ---------------------------------------------------------------- families
    comps, pref = [], []
    n_runs = {}
    for lobe in LOBES:
        for lad, (label, rects, varied) in LADDERS.items():
            rung_col = f"rung_{lad}"; rv = [h if lad == "hlad" else w for w, h in rects]
            for con in CONTRASTS:
                objs_all = per_run[(per_run.lobe == lobe) & ~per_run.null & (per_run.contrast_name == con) & per_run[rung_col].notna()]
                nulls_all = per_run[(per_run.lobe == lobe) & per_run.null & per_run[rung_col].notna()]
                n_runs[f"{lobe}:{lad}:{con}"] = {str(rr): [int((objs_all[(objs_all[rung_col] == rr) & (objs_all.type == 'LC11')]).shape[0]),
                                                           int((nulls_all[(nulls_all[rung_col] == rr) & (nulls_all.type == 'LC11')]).shape[0])] for rr in rv}
                for t in LC_TYPES:
                    objs = objs_all[objs_all.type == t]; nulls = nulls_all[nulls_all.type == t]
                    lab = {"lobe": lobe, "ladder": lad, "contrast_name": con, "type": t}
                    role_p = "primary" if lobe == "ship" else "control_fb0"
                    fam = [compare_row(objs[objs[rung_col] == rr].drive_median.to_numpy(), nulls[nulls[rung_col] == rr].drive_median.to_numpy(),
                                           dict(lab, family=f"primary:{lobe}:{lad}:{con}:{t}", rung=rr, statistic="drive_median", role=role_p)) for rr in rv]
                    comps += _holm_family(fam, m=len(rv))
                    mag = [compare_row(objs[objs[rung_col] == rr].spikes_median.to_numpy(), nulls[nulls[rung_col] == rr].spikes_median.to_numpy(),
                                           dict(lab, family=f"magnitude:{lobe}:{lad}:{con}:{t}:spikes_median", rung=rr, statistic="spikes_median", role="reported_magnitude")) for rr in rv]
                    for c in mag:
                        c["p_holm"] = float("nan"); c["survives_holm"] = False
                    comps += mag
                    for q in ("drive_mean", "drive_max", "diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell"):
                        ex = [compare_row(objs[objs[rung_col] == rr][q].to_numpy(), nulls[nulls[rung_col] == rr][q].to_numpy(),
                                              dict(lab, family=f"exploratory:{lobe}:{lad}:{con}:{t}:{q}", rung=rr, statistic=q, role="exploratory")) for rr in rv]
                        comps += _holm_family(ex, m=len(rv))
                    for q in ("drive_median", "spikes_median", "drive_mean", "drive_max", "diff_max_over_cells_mean_mv", "diff_rate_hz_max_cell"):
                        pref.append(dict(lab, statistic=q, role=("primary" if q == "drive_median" else "reported_magnitude" if q == "spikes_median" else "exploratory"),
                                         **preference_tests(objs, nulls, rung_col, q, rv, TARGET_RUNGS[t][lad])))
                for t in UPSTREAM:
                    objs = up[(up.lobe == lobe) & ~up.null & (up.contrast_name == con) & up[rung_col].notna() & (up.type == t)]
                    nulls = up[(up.lobe == lobe) & up.null & up[rung_col].notna() & (up.type == t)]
                    lab = {"lobe": lobe, "ladder": lad, "contrast_name": con, "type": t}
                    for q in ("diff_signed_best_cell", "diff_signed_mean", "diff_abs_best_cell_mean", "abs_drive_median", "abs_drive_median_rf"):
                        fam = [compare_row(objs[objs[rung_col] == rr][q].to_numpy(), nulls[nulls[rung_col] == rr][q].to_numpy(),
                                               dict(lab, family=f"secondary:{lobe}:{lad}:{con}:{t}:{q}", rung=rr, statistic=q, role="secondary_exploratory")) for rr in rv]
                        comps += _holm_family(fam, m=len(rv))
                        pref.append(dict(lab, statistic=q, role="secondary_exploratory", **preference_tests(objs, nulls, rung_col, q, rv, None)))
    # ---------------------------------------------------------------- effective contrast per rectangle x contrast
    fpc = [c for c in fp.columns if c not in ("run", "lobe", "seed", "width_deg", "height_deg", "contrast_name")]
    fps = fp.groupby(["width_deg", "height_deg", "contrast_name"]).agg(n_runs=("run", "size"), **{c: (c, "mean") for c in fpc},
                                                                       **{c + "_sd_runs": (c, "std") for c in ("max_coverage", "col_equiv_median", "extreme_rel_change_median")}).reset_index() if len(fp) else pd.DataFrame()
    for lad in LADDERS:
        fps[f"rung_{lad}"] = [_rung_of(w, h, lad) for w, h in zip(fps.width_deg, fps.height_deg)]
    # ---------------------------------------------------------------- answers
    answers = {}
    cdf = pd.DataFrame(comps)
    for lobe in ("ship",):
        for lad in LADDERS:
            for con in CONTRASTS:
                for t in LC_TYPES:
                    g = cdf[(cdf.role == "primary") & (cdf.lobe == lobe) & (cdf.ladder == lad) & (cdf.contrast_name == con) & (cdf.type == t)]
                    pf = [p for p in pref if p["lobe"] == lobe and p["ladder"] == lad and p["contrast_name"] == con and p["type"] == t and p["statistic"] == "drive_median"][0]
                    surv = g[g.survives_holm.astype(bool)]
                    called = bool(len(surv))
                    shape = bool(called and np.isfinite(pf["peak_contrast"]) and pf["peak_contrast"] > 0 and pf["peak_p_perm"] <= ALPHA)
                    answers[f"{lad}:{con}:{t}"] = {"family_members": int(len(g)), "verdicts": g.verdict.value_counts().to_dict(), "min_p": float(g.p.min()) if len(g) else np.nan,
                                                   "min_p_holm": float(g.p_holm.min()) if len(g) else np.nan,
                                                   "members_result_unadjusted": g[g.verdict == "result"][["rung", "z", "p", "p_holm"]].to_dict("records"),
                                                   "members_surviving_holm": surv[["rung", "z", "p", "p_holm"]].to_dict("records"),
                                                   "excess_over_null_by_rung_mv": pf["excess_by_rung"], "n_bodies_windowed_by_rung": pf["n_bodies_windowed_by_rung"],
                                                   "spearman_rho": pf["spearman_rho"], "spearman_p_perm": pf["spearman_p_perm"], "small_minus_large": pf["small_minus_large"],
                                                   "small_minus_large_p_perm": pf["small_minus_large_p_perm"], "peak_contrast": pf["peak_contrast"], "peak_p_perm": pf["peak_p_perm"],
                                                   "target_rungs": TARGET_RUNGS[t][lad], "size_preference_called": called, "animal_shape_shown": shape}
    mono = {}
    for lad in LADDERS:
        for con in CONTRASTS:
            for t in ("T2", "T3", "Tm5Y", "TmY21"):
                pa = [p for p in pref if p["lobe"] == "ship" and p["ladder"] == lad and p["contrast_name"] == con and p["type"] == t]
                best = [p for p in pa if p["statistic"] == "diff_signed_best_cell"][0]; comp = [p for p in pa if p["statistic"] == "abs_drive_median"][0]
                mono[f"{lad}:{con}:{t}"] = {"best_cell_rho": best["spearman_rho"], "best_cell_p_perm": best["spearman_p_perm"], "best_cell_excess_by_rung": best["excess_by_rung"],
                                            "abs_drive_median_rho": comp["spearman_rho"], "abs_drive_median_p_perm": comp["spearman_p_perm"], "abs_drive_median_excess_by_rung": comp["excess_by_rung"],
                                            "size_monotone": bool(np.isfinite(best["spearman_rho"]) and best["spearman_rho"] > 0 and best["spearman_p_perm"] <= ALPHA
                                                                  and np.isfinite(comp["spearman_rho"]) and comp["spearman_rho"] > 0 and comp["spearman_p_perm"] <= ALPHA),
                                            "best_cell_monotone_alone": bool(np.isfinite(best["spearman_rho"]) and best["spearman_rho"] > 0 and best["spearman_p_perm"] <= ALPHA)}
    sphere = sphere_tables(sphere_json, per_run, up, cdf, pref) if sphere_json and Path(sphere_json).exists() else {}
    return {"runs": runs.drop(columns=["stem"]).to_dict("records"), "per_body": per_body, "per_run": per_run, "upstream_runs": up, "comparisons": comps, "preference": pref,
            "footprint": fps, "footprint_runs": fp, "answers": answers, "size_monotone": mono, "prov0": prov0, "n_runs": n_runs, "sphere": sphere,
            "devices_by_lobe": {l: sorted(set(per_run[per_run.lobe == l].device.dropna().astype(str))) for l in LOBES}}


def spearman_perm(x, y, n_perm: int = 20000, seed: int = 0) -> dict:
    """object_round2_baseline.spearman_perm, vectorised: Spearman rho = Pearson on average ranks, the label permutations
    drawn from the same RandomState sequence (rng.permutation over an n-vector), the same two-sided count -- identical
    p to the loop over scipy.spearmanr (checked on the smoke) at ~1/500 of the time. rho undefined (constant input)
    -> NaN / NaN, never a floor p (the round-2 defect)."""
    from scipy.stats import rankdata
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    n = int(len(x))
    if n < 4:
        return {"rho": np.nan, "p_perm": np.nan, "n": n}
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return {"rho": np.nan, "p_perm": np.nan, "n": n, "n_perm": n_perm, "note": "rho undefined (constant input)"}
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    rho = float((rx * ry).mean())
    rng = np.random.RandomState(seed)
    perms = np.stack([rng.permutation(n) for _ in range(n_perm)])
    rho_p = (rx[None, :] * ry[perms]).mean(1)
    cnt = int((np.abs(rho_p) >= abs(rho) - 1e-12).sum())
    return {"rho": rho, "p_perm": float((cnt + 1) / (n_perm + 1)), "n": n, "n_perm": n_perm}


def preference_tests(objs: pd.DataFrame, nulls: pd.DataFrame, rung_col: str, q: str, rv: list, targets: list | None) -> dict:
    x = objs[rung_col].to_numpy(float); y = objs[q].to_numpy(float)
    sp = spearman_perm(x, y)
    ct = contrast_perm(objs[objs[rung_col].isin(SMALL_RUNGS)][q], objs[objs[rung_col].isin(LARGE_RUNGS)][q])
    pk = contrast_perm(objs[objs[rung_col].isin(targets)][q], objs[~objs[rung_col].isin(targets)][q]) if targets else {"contrast": np.nan, "p_perm": np.nan}
    spn = spearman_perm(nulls[rung_col].to_numpy(float), nulls[q].to_numpy(float), n_perm=2000)
    return {"spearman_rho": sp["rho"], "spearman_p_perm": sp["p_perm"], "n_runs": sp["n"], "small_minus_large": ct["contrast"], "small_minus_large_p_perm": ct["p_perm"],
            "peak_contrast": pk["contrast"], "peak_p_perm": pk["p_perm"], "peak_rungs": targets, "null_spearman_rho": spn["rho"], "null_spearman_p_perm": spn["p_perm"],
            "excess_by_rung": {str(rr): float(objs[objs[rung_col] == rr][q].mean() - nulls[nulls[rung_col] == rr][q].mean()) for rr in rv},
            "stim_mean_by_rung": {str(rr): float(objs[objs[rung_col] == rr][q].mean()) for rr in rv}, "null_mean_by_rung": {str(rr): float(nulls[nulls[rung_col] == rr][q].mean()) for rr in rv},
            "n_bodies_windowed_by_rung": ({str(rr): float(objs[objs[rung_col] == rr].n_bodies_windowed.mean()) for rr in rv} if "n_bodies_windowed" in objs.columns else None)}


def sphere_tables(sphere_json: str, per_run: pd.DataFrame, up: pd.DataFrame, cdf: pd.DataFrame, pref: list) -> dict:
    """The round-2 sphere ladder beside these rectangles at the matching sizes: magnitudes, two batches on different boxes."""
    d = json.load(open(sphere_json, encoding="utf-8"))
    sc = pd.DataFrame(d["tables"]["sphere_comparisons"]); sp = d["tables"]["sphere_preference"]; sup = pd.DataFrame(d["tables"]["sphere_upstream_runs"]); sfp = pd.DataFrame(d["tables"]["sphere_footprint"])
    match = [(4.5, 4.4, 4.4, "hlad"), (8.8, 8.8, 8.8, "wlad"), (15.0, 4.4, 15.0, "hlad"), (15.0, 15.0, 8.8, "wlad"), (30.0, 4.4, 30.0, "hlad"), (30.0, 30.0, 8.8, "wlad")]
    rows = []
    for t in LC_TYPES:
        pf = [p for p in sp if p["lobe"] == "ship" and p["type"] == t and p["statistic"] == "drive_median"][0]
        for diam, w, h, lad in match:
            s = sc[(sc.role == "primary") & (sc.type == t) & (sc.statistic == "drive_median") & (sc.diam_deg == diam)]
            rr = h if lad == "hlad" else w
            for con in CONTRASTS:
                r = cdf[(cdf.role == "primary") & (cdf.lobe == "ship") & (cdf.type == t) & (cdf.ladder == lad) & (cdf.contrast_name == con) & (cdf.rung == rr)]
                rows.append({"type": t, "statistic": "drive_median", "sphere_diam_deg": diam, "rect_w_deg": w, "rect_h_deg": h, "rect_ladder": lad, "rect_contrast": con,
                             "sphere_excess_mv": pf["excess_by_rung"].get(str(diam)), "sphere_z": float(s.z.iloc[0]) if len(s) else np.nan, "sphere_p_holm": float(s.p_holm.iloc[0]) if len(s) else np.nan,
                             "sphere_verdict": s.verdict.iloc[0] if len(s) else None, "rect_excess_mv": float(r["diff"].iloc[0]) if len(r) else np.nan,
                             "rect_z": float(r.z.iloc[0]) if len(r) else np.nan, "rect_p_holm": float(r.p_holm.iloc[0]) if len(r) else np.nan, "rect_verdict": r.verdict.iloc[0] if len(r) else None,
                             "sphere_extreme_rel_change": float(sfp[(sfp.lobe == "ship") & (sfp.diam_deg == diam)].extreme_rel_change_median.iloc[0]) if len(sfp) else np.nan})
    for t in ("T2", "T3", "Tm5Y", "TmY21"):
        so = sup[(sup.lobe == "ship") & (sup.type == t)]
        nul = so[so.null].diff_signed_best_cell.mean()
        for diam, w, h, lad in match:
            rr = h if lad == "hlad" else w
            for con in CONTRASTS:
                pf = [p for p in pref if p["lobe"] == "ship" and p["type"] == t and p["ladder"] == lad and p["contrast_name"] == con and p["statistic"] == "diff_signed_best_cell"][0]
                pc = [p for p in pref if p["lobe"] == "ship" and p["type"] == t and p["ladder"] == lad and p["contrast_name"] == con and p["statistic"] == "abs_drive_median"][0]
                rows.append({"type": t, "statistic": "diff_signed_best_cell", "sphere_diam_deg": diam, "rect_w_deg": w, "rect_h_deg": h, "rect_ladder": lad, "rect_contrast": con,
                             "sphere_excess_mv": float(so[~so.null & (so.diam_deg == diam)].diff_signed_best_cell.mean() - nul), "rect_excess_mv": pf["excess_by_rung"].get(str(rr)),
                             "rect_abs_drive_median_excess": pc["excess_by_rung"].get(str(rr))})
    sph_pref = {t: {k: v for k, v in [p for p in sp if p["lobe"] == "ship" and p["type"] == t and p["statistic"] == "drive_median"][0].items() if k != "excess_by_rung"} for t in LC_TYPES}
    return {"table": pd.DataFrame(rows), "sphere_preference_ship": sph_pref, "source": sphere_json, "sphere_devices": sorted(set(pd.DataFrame(d["tables"]["sphere_per_run"]).device.dropna().astype(str)))}


def _fmt(c: dict) -> str:
    return (f"{c['lobe']:4s} {c['ladder']} {c['contrast_name']:6s} {c['type']:6s} rung {c['rung']:>5} {c['statistic']:26s} stim {c['stim_mean']:+.4f} +- {c['stim_sd']:.4f} (n {c['stim_n']}) "
            f"null {c['null_mean']:+.4f} +- {c['null_sd']:.4f} (n {c['null_n']}) z {c['z']:+.2f} p {c['p']:.4f} p_holm {c.get('p_holm', float('nan')):.4f} -> {c['verdict']}"
            f"{' SURVIVES HOLM' if c.get('survives_holm') else ''}")


def cmd_analyse(args) -> int:
    out = args.out.rstrip("/")
    rep = None
    if not args.no_verify:
        rep = verify_all(out, args.log)
        print(f"verify: log {rep['log']!r}; runs {rep['n_runs']} ({rep['n_consoles']} consoles); devices {rep['devices_by_lobe']}; problems: {len(rep['problems'])} {rep['problems'][:6]}")
        if rep["problems"] and not args.force:
            sys.exit("batch verification failed (pass --force to analyse anyway; the Result then records the problems)")
    win, win_summary = orc.load_windows(args.rf_maps)
    print(f"windows: maps {win_summary['maps']}; sources {win_summary['sources']}; box widths {win_summary['box_width_by_type_deg']}")
    res_ = analyse(out, win, args.sphere)
    stamp_p = Path(out) / "predeclared.json"
    stamp = json.load(open(stamp_p, encoding="utf-8")) if stamp_p.exists() else None
    stamp_match = bool(stamp and json.dumps(common.to_jsonable(stamp["predeclared"]), sort_keys=True) == json.dumps(common.to_jsonable(PREDECLARED), sort_keys=True))
    if stamp and not stamp_match:
        print("WARNING: the live PREDECLARED dict differs from the stamped out/objr3rect/predeclared.json -- the stamp is the predeclaration and is what the Result carries")
    prov = res_["prov0"] or common.provenance(__import__("flyverse.connectome", fromlist=["load"]).load(verbose=False))
    res = common.Result.new("trace", prov)
    res.run_id = "objr3rect-rectangles"; res.tool_version = "object_round3_rectangles/1"
    res.replicates = {"n": N_RUNS, "unit": "runs", "runs": res_["runs"], "null": "the six blank/blank runs of the same rectangle and lobe in the same submission (their own seeds)",
                      "n_runs": res_["n_runs"], "batch": {"jobs_json": f"{out}/jobs.json", "cluster_log": args.log}}
    Path(out).mkdir(parents=True, exist_ok=True)
    res_["per_body"].to_csv(f"{out}/per_body.csv", index=False)
    res.add_table("per_run", res_["per_run"]); res.add_table("upstream_runs", res_["upstream_runs"]); res.add_table("comparisons", res_["comparisons"]); res.add_table("preference", res_["preference"])
    res.add_table("effective_contrast", res_["footprint"]); res.add_table("effective_contrast_runs", res_["footprint_runs"]); res.add_table("windows", win)
    if res_["sphere"]:
        res.add_table("sphere_vs_rectangles", res_["sphere"]["table"])
    res.summary = {"predeclared": (stamp or {}).get("predeclared", PREDECLARED), "predeclared_stamped_utc": (stamp or {}).get("stamped_utc"), "predeclared_matches_live_dict": stamp_match,
                   "windows": win_summary, "answers": res_["answers"], "size_monotone": res_["size_monotone"], "verify": rep, "devices_by_lobe": res_["devices_by_lobe"],
                   "sphere": {k: v for k, v in (res_["sphere"] or {}).items() if k != "table"},
                   "reading": "verdicts are common.compare's (z on the null SD, exact U, p_floor); p is the tie-aware exact permutation U; p_holm is Holm within the named 5-member family; "
                              "only a `result` with p_holm <= 0.05 is called; spikes_median rows are reported magnitudes outside every family; fb0 rows are magnitudes; "
                              "every max-over-cells row has its per-body median in the same table; nothing is adopted"}
    res.validation = {"name": "contrast-matched rectangle ladders: predeclared LC11 / LC10a families (height at width 4.4, width at height 8.8; dark and bright)",
                      "reference": {"LC11": "peak near height 8.8 deg / width ~4.4 deg (Keles & Frye 2017)", "LC10a": "15-30 deg (Schretter et al. 2024)"},
                      "measured": res_["answers"], "status": "measured", "source": "scripts/object_round3_rectangles.py analyse"}
    res.files = {"generator": " ".join(sys.argv), "per_body": f"{out}/per_body.csv", "predeclared": str(stamp_p), "jobs": f"{out}/jobs.json", "cluster_log": args.log, "rf_maps": win_summary["maps"],
                 "sphere_result": args.sphere, "analysis_code_sha256": analysis_code_sha256(), "analysis_code_sha256_at_stamp": (stamp or {}).get("analysis_code_sha256_at_stamp"),
                 "analysis_code_matches_stamp": bool(stamp and stamp.get("analysis_code_sha256_at_stamp") == analysis_code_sha256())}
    p = res.save(args.json)
    # ---------------------------------------------------------------- console
    print("\n== EFFECTIVE CONTRAST per rectangle x contrast (synthesised radiance; mean over runs; *_sd_runs must be 0: the stimulus does not depend on the seed)")
    common.print_table(res_["footprint"][["width_deg", "height_deg", "contrast_name", "n_runs", "weber_contrast", "max_coverage", "col_equiv_median", "frames_no_column_5pct_frac", "n_cols_50pct_mean",
                                          "extreme_rel_change_median", "mean_rel_change_over_changed_set", "max_coverage_sd_runs"]], floatfmt="{:.3f}", max_rows=40)
    for role, title in (("primary", "PRIMARY (shipped lobe): LC drive_median, per (ladder x contrast x type) family of 5, Holm within"),
                        ("control_fb0", "fb0 control (magnitudes)"), ("reported_magnitude", "spikes_median (reported magnitude, outside the families)")):
        print(f"\n== {title}")
        for c in res_["comparisons"]:
            if c["role"] == role:
                print("  " + _fmt(c))
    print("\n== PREFERENCE TESTS (shipped lobe, LC drive_median)")
    common.print_table(pd.DataFrame([{k: v for k, v in p_.items() if not isinstance(v, (dict, list))} for p_ in res_["preference"] if p_["lobe"] == "ship" and p_["statistic"] == "drive_median"]),
                       floatfmt="{:+.4f}", max_rows=40)
    print("\n== SECONDARY upstream (shipped lobe, dark): the three whole-window statistics and the per-body companion, per rung")
    for c in res_["comparisons"]:
        if c["role"] == "secondary_exploratory" and c["lobe"] == "ship" and c["contrast_name"] == "dark" and c["type"] in ("T2", "T3") and c["statistic"] in ("diff_signed_best_cell", "abs_drive_median"):
            print("  " + _fmt(c))
    print("\n== SIZE-MONOTONE (upstream, shipped lobe)"); print(json.dumps(common.to_jsonable(res_["size_monotone"]), indent=1)[:8000])
    if res_["sphere"]:
        print("\n== SPHERE vs RECTANGLES (magnitudes; two batches, different boxes)")
        common.print_table(res_["sphere"]["table"], floatfmt="{:+.4f}", max_rows=80)
    print("\n== ANSWERS"); print(json.dumps(common.to_jsonable(res_["answers"]), indent=1))
    print(f"\nproblems: {res.check() or 'none'}; predeclared matches stamp: {stamp_match}; written {p}")
    return 0


# ==================================================================================================== main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", default="out/objr3rect"); p.add_argument("--raw", default="out/objr3rect_raw"); p.add_argument("--name", default="objr3rect")
    p.add_argument("--minutes", type=int, default=60); p.add_argument("--runs", type=int, default=N_RUNS); p.add_argument("--seconds", type=float, default=SECONDS)
    p.add_argument("--settle", type=float, default=SETTLE); p.set_defaults(func=cmd_plan)
    p.add_argument("--seed-offset", type=int, default=0, help="offset stimulus and null seeds for an independent replication")
    s = sub.add_parser("submit"); s.add_argument("--out", default="out/objr3rect"); s.set_defaults(func=cmd_submit)
    s.add_argument("--target"); s.add_argument("--node")
    rj = sub.add_parser("run-job"); rj.add_argument("--job", required=True); rj.add_argument("--fam", default=FAM_TOKEN, help="the --arm-block token (recorded; every job of the family carries it)")
    rj.add_argument("--out", default="out/objr3rect"); rj.add_argument("--raw", default="out/objr3rect_raw"); rj.add_argument("--runs", type=int, default=N_RUNS)
    rj.add_argument("--seconds", type=float, default=SECONDS); rj.add_argument("--settle", type=float, default=SETTLE); rj.add_argument("--allow-cpu", action="store_true", help="CPU smoke")
    rj.add_argument("--seed-offset", type=int, default=0)
    rj.set_defaults(func=cmd_run_job)
    c = sub.add_parser("check-job"); c.add_argument("--expect", nargs="+", required=True); c.add_argument("--allow-cpu", action="store_true"); c.set_defaults(func=cmd_check_job)
    r = sub.add_parser("reduce"); r.add_argument("--raw", required=True, help="the record stem (<raw>_stim.npz ...)"); r.add_argument("--out", required=True, help="the reduced stem (<out>_reduced.npz)")
    r.add_argument("--n-check", type=int, default=40); r.set_defaults(func=cmd_reduce)
    st = sub.add_parser("selftest"); st.add_argument("--raw", required=True); st.add_argument("--tmp", default="out/objr3rect_smoke/selftest"); st.add_argument("--rf-maps", nargs="+", default=RF_MAPS)
    st.set_defaults(func=cmd_selftest)
    v = sub.add_parser("verify"); v.add_argument("--out", default="out/objr3rect"); v.add_argument("--log", default="out/objr3rect_cluster.log"); v.add_argument("--json", default="out/objr3rect/verify.json")
    v.set_defaults(func=cmd_verify)
    a = sub.add_parser("analyse"); a.add_argument("--out", default="out/objr3rect"); a.add_argument("--log", default="out/objr3rect_cluster.log"); a.add_argument("--json", default="out/interp/objr3rect/rectangles.json")
    a.add_argument("--rf-maps", nargs="+", default=RF_MAPS); a.add_argument("--sphere", default="out/interp/objr2/baseline.json", help="the round-2 sphere Result for the comparison table")
    a.add_argument("--no-verify", action="store_true"); a.add_argument("--force", action="store_true"); a.set_defaults(func=cmd_analyse)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
