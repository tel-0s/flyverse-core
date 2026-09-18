"""Round 8, item 4 -- the goal-only plume arm (docs/audits/plume_goal_only.md, batch plume-go).

plume_steering.md's correction bundled a bilateral walking goal with a DNa02 L-R integral feedback onto PFL3; the
skeptic inferred the feedback is the load-bearing part. Three arms of the shipped `compass plume hunger flight` room:

| arm           | plume instrument            | what it is                                                            |
| full          | plume                       | as shipped: walking goal + DNa02 L-R integral feedback (gain 5 /s)    |
| goal-only     | plume:feedback=0            | the walking goal with the one-way clip(80 * turn) bridge (gain 0)     |
| feedback-only | plume:walking_goal=0        | the feedback with the pre-correction upwind / entry-memory goal law   |

Two start sets: `shipped` (the plume_steering.md rooms: default start, headings 5 / 90 / -90 / 185 / 5 / 5 deg) and
`drawn` (six starts and headings drawn once from numpy seed 8 inside the table top: x in [-0.55, 0.55], y in
[-0.35, 0.35], heading in [-180, 180) deg; the same six for every arm and run). Six RUNS per (arm, start set), each a
B=6 room (env seeds 0-5, all fruit, no fence, energy 0.1, 60 s) with its own brain seed 31 + r -- the run is the
replicate unit (determinism_gate.md: no room path repeats, so a run is a draw whatever its seed, and the seeds keep the
draws honest). 3 x 2 x 6 = 36 jobs, two cluster_run.py calls of 18. The shipped native path throughout (the path
plume_steering.md ran; determinism_gate.md found the torch path no better).

    PYTHONIOENCODING=utf-8 python scripts/plume_goal_only.py plan --out out/plume-go
    PYTHONIOENCODING=utf-8 python scripts/plume_goal_only.py predeclare out/plume-go
    python scripts/plume_goal_only.py room --arm full --starts shipped --run 0 --out STEM        (the job)
    PYTHONIOENCODING=utf-8 python scripts/plume_goal_only.py analyse --runs out/plume-go --out out/plume-go/analysis
        -> runs.csv (one row per run: fed rows, first contact, DNa02 |L-R|, path, checks), rows.csv (one row per
           room row -- rule 28), compare.csv (the six predeclared contrasts, common.compare 6 v 6, Holm m = 6),
           descriptive.csv, analysis.md, summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ARMS = {"full": ["compass", "plume", "hunger", "flight"],
        "goal-only": ["compass", "plume:feedback=0", "hunger", "flight"],
        "feedback-only": ["compass", "plume:walking_goal=0", "hunger", "flight"]}
VARIANT = {"full": "full", "goal-only": "goal-only", "feedback-only": "feedback-only"}
STARTS = ("shipped", "drawn")
RUNS = list(range(6))
BRAIN_SEED0 = 31
SHIPPED_HEADINGS_DEG = [5, 90, -90, 185, 5, 5]
DRAW_SEED = 8
TABLE = dict(x=(-0.55, 0.55), y=(-0.35, 0.35), top_z=0.75)
ENERGY = 0.1
SECONDS = 60.0
FLAGS = dict(cuda_graphs=True, cuda_kernels=True, event_driven=True, cuda_sparse="torch")
GPU_POOL = "4,5,6,7"
MAX_JOBS_PER_CALL = 24
FAMILY = [("1_fed_rows_full_vs_goal-only_shipped", "fed_rows", "full", "goal-only", "shipped", "positive"),
          ("2_fed_rows_full_vs_feedback-only_shipped", "fed_rows", "full", "feedback-only", "shipped", "none"),
          ("3_dna_lr_full_vs_goal-only_shipped", "mean_abs_dna_lr_hz", "full", "goal-only", "shipped", "positive"),
          ("4_fed_rows_full_vs_goal-only_drawn", "fed_rows", "full", "goal-only", "drawn", "positive"),
          ("5_fed_rows_full_vs_feedback-only_drawn", "fed_rows", "full", "feedback-only", "drawn", "none"),
          ("6_dna_lr_full_vs_goal-only_drawn", "mean_abs_dna_lr_hz", "full", "goal-only", "drawn", "positive")]
ROW_KEYS = ["feeding_s", "first_contact_s", "path_m", "mean_abs_dna_lr_hz", "mean_abs_turn", "mean_abs_steer_input_hz",
            "mean_abs_integral_hz", "mean_abs_yaw_rad_s", "min_fruit_distance_m", "end_energy", "zero_energy_s"]
RUN_KEYS = ["fed_rows", "fed_fraction", "mean_first_contact_s", "mean_abs_dna_lr_hz", "mean_path_m", "mean_abs_turn",
            "mean_abs_steer_input_hz", "mean_abs_integral_hz", "mean_abs_yaw_rad_s"]


def drawn_starts():
    rng = np.random.default_rng(DRAW_SEED)
    x = rng.uniform(*TABLE["x"], size=6); y = rng.uniform(*TABLE["y"], size=6); h = rng.uniform(-180.0, 180.0, size=6)
    return [dict(x=float(a), y=float(b), z=TABLE["top_z"], heading_deg=float(c)) for a, b, c in zip(x, y, h)]


def starts_of(name: str):
    if name == "shipped":
        return [dict(x=-0.5, y=0.05, z=TABLE["top_z"], heading_deg=float(h)) for h in SHIPPED_HEADINGS_DEG]
    return drawn_starts()


def room_command(arm: str, starts: str, run: int, rel: str) -> tuple[str, str]:
    stem = f"{rel}/{arm}_{starts}_r{run}"
    return f"PYTHONIOENCODING=utf-8 python scripts/plume_goal_only.py room --arm {arm} --starts {starts} --run {run} --out {stem}", stem


def job_line(cmd: str, stem: str, rel: str) -> str:
    return (f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"{cmd} > {stem}.txt 2>&1; st=\\$?; tail -3 {stem}.txt; exit \\$st")


def plan(out_dir: Path, minutes: int = 90, name: str = "plume-go"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rel = f"out/{out_dir.name}"
    if (out_dir / "predeclared.json").exists():
        raise ValueError("batch is frozen; use a new directory for a new predeclaration")
    jobs = []
    for starts in STARTS:
        for run in RUNS:
            for arm in ARMS:
                cmd, stem = room_command(arm, starts, run, rel)
                jobs.append(dict(arm=arm, starts=starts, run=run, brain_seed=BRAIN_SEED0 + run, stem=stem, line=job_line(cmd, stem, rel)))
    calls = [jobs[i:i + 18] for i in range(0, len(jobs), 18)]           # 18 = one start set; never more than MAX_JOBS_PER_CALL
    assert all(len(c) <= MAX_JOBS_PER_CALL for c in calls)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = ["#!/bin/bash", "set -o pipefail",
             f"# DRAFT -- NOT SUBMITTED. Round 8 / batch {name}: docs/audits/plume_goal_only.md.",
             f"# {len(jobs)} jobs = {len(ARMS)} arms x {len(STARTS)} start sets x {len(RUNS)} runs (brain seeds {BRAIN_SEED0}-{BRAIN_SEED0 + 5}), one B=6 room per job,",
             f"# {len(calls)} cluster_run.py calls of <= {MAX_JOBS_PER_CALL}; every job pinned to one GPU of the pool {GPU_POOL} (gpus 1 + gpu_ids [one id]).",
             "# The second cluster node is named by the CLUSTER_NODE_2 environment variable (docs/CLUSTER.md, git-ignored); committed",
             "# text never carries the name. Generated by scripts/plume_goal_only.py plan on " + stamp + ".",
             f"# Read: `ls {rel}/*_r?.json | wc -l` = {len(jobs)}, `grep -c 'device cuda' {rel}/*_r?.txt`, then analyse (zero problems)."]
    for i, call in enumerate(calls):
        lines.append("python scripts/cluster_run.py --target house --node \"${CLUSTER_NODE_2:?set CLUSTER_NODE_2 to the name of the second cluster node}\" "
                     f"--gpu-ids {GPU_POOL} --ship flyverse,scripts --name {name} --minutes {minutes} "
                     + " ".join('"' + j["line"] + '"' for j in call) + f" --fetch {rel}/ 2>&1 | tee {rel}/client_stdout_{i}.txt || exit $?")
    (out_dir / "batch.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "jobs.json").write_text(json.dumps({"batch": name, "arms": ARMS, "starts": {s: starts_of(s) for s in STARTS}, "runs": RUNS,
                                                   "brain_seed0": BRAIN_SEED0, "draw_seed": DRAW_SEED, "seconds": SECONDS, "energy": ENERGY,
                                                   "flags": FLAGS, "family": FAMILY, "gpu_pool": GPU_POOL, "node": "<cluster-node-2>",
                                                   "jobs": jobs, "generated_utc": stamp, "status": "DRAFT, not submitted"}, indent=1),
                                       encoding="utf-8")
    print(f"{len(jobs)} jobs in {len(calls)} call(s) -> {out_dir / 'batch.sh'} (DRAFT, not submitted); {out_dir / 'jobs.json'}")


def source_hashes() -> dict:
    paths = sorted(set(ROOT.glob("flyverse/**/*.py")) | set(ROOT.glob("flyverse/data/*.csv")) |
                   {Path(__file__).resolve(), ROOT / "scripts/plume_steering_probe.py", ROOT / "scripts/navigation_probe.py",
                    ROOT / "scripts/cluster_run.py"})
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}


def predeclare(out_dir: Path):
    out_dir = Path(out_dir)
    if list(out_dir.glob("*_r?.json")):
        raise ValueError("cannot predeclare after results exist")
    plan_rec = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
    record = dict(plan_rec, status="PREDECLARED, not submitted", written_before_submission=True,
                  stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  batch_sha256=hashlib.sha256((out_dir / "batch.sh").read_bytes()).hexdigest(),
                  jobs_sha256=hashlib.sha256((out_dir / "jobs.json").read_bytes()).hexdigest(),
                  source_sha256_lf=source_hashes(),
                  replicate_unit="runs: one B=6 room per job, six per (arm, start set), brain seeds 31-36; a row is not a replicate",
                  multiplicity="one Holm family, m = 6 (6 v 6 exact-U floor 0.0021645 x 6 = 0.013, satisfiable); a missing p counts toward m",
                  measures={"fed_rows": "rows of the six that feed for >= 1 s in 60 s (the plume_steering.md criterion)",
                            "mean_first_contact_s": "mean over the fed rows of the first frame with feeding (NaN when none fed; descriptive only)",
                            "mean_abs_dna_lr_hz": "mean over rows and 0.1 s samples of |DNa02 L - R| (Hz)",
                            "mean_path_m": "mean over rows of the path length (m)"},
                  predictions={f[0]: f[5] for f in FAMILY},
                  prediction_note="`positive` = full minus the other arm > 0 (the feedback is load-bearing: goal-only feeds fewer rows and moves "
                                  "DNa02 less); `none` = no sign predicted for feedback-only (the walking goal's own contribution is the question)",
                  verdict_rule="flyverse.interp.common.compare (result / null / underpowered / undetermined); a result with the unpredicted sign is "
                               "`result, opposite sign`; fed_rows is an integer count so ties are expected and p_floor is reported",
                  quoting_rule="determinism_gate.md: no number is quoted to more than its across-run SD; the shipped-seed run (full, shipped, r0, "
                               "brain seed 31) is the plume_steering.md configuration and its values are one draw beside the audit's",
                  validity="36 runs; device cuda; backend native flags; preset instrumented with the four instruments; the plume record's "
                           "`variant` equal to the arm's; brain seed 31 + r; the six starts / headings equal to the frozen list; every loaded "
                           "source at the predeclared hash",
                  nothing_adopted="no admission claim, no default change, no gain selected")
    with (out_dir / "predeclared.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"froze {out_dir / 'predeclared.json'} before submission")


# ------------------------------------------------------------------------------------------------ the room job
def room(arm: str, starts: str, run: int, out: str, seconds: float = SECONDS):
    import os
    import torch
    from flyverse import BatchSim, connectome
    from flyverse.interp.common import provenance, to_jsonable

    out = Path(out)
    if out.with_suffix(".json").exists():
        raise FileExistsError(out)
    assert torch.cuda.is_available(), "GPU required"
    print("device cuda", torch.cuda.get_device_name(0), "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"),
          "arm", arm, "starts", starts, "run", run, flush=True)
    t0 = time.time()
    c = connectome.load(verbose=False)
    st = starts_of(starts)
    seed = BRAIN_SEED0 + run
    sim = BatchSim(6, c=c, seed=seed, seeds=range(6), device="cuda", program="none", fruit_set="all", fence=False,
                   start=np.array([[s["x"], s["y"], s["z"]] for s in st]), preset="instrumented", instruments=list(ARMS[arm]), **FLAGS)
    for f, m, s in zip(sim.flies, sim.metabolisms, st):
        f.heading = np.deg2rad(s["heading_deg"])
        m.energy = ENERGY
    plume = sim.fb.instruments["plume"]
    b = sim.brain
    frames = int(round(seconds * 100))
    samples = []
    feeding = np.zeros(6); dist = np.zeros(6); first = np.full(6, np.nan); zero_energy = np.full(6, np.nan)
    previous = np.array([f.pos for f in sim.flies])
    for frame in range(frames):
        sim.step()
        current = np.array([f.pos for f in sim.flies])
        dist += np.linalg.norm(current - previous, axis=1); previous = current
        feeding += sim.feeding * 0.01
        newly = (sim.feeding > 0) & np.isnan(first); first[newly] = (frame + 1) / 100.0
        energy = np.array([m.energy for m in sim.metabolisms])
        ze = (energy <= 0) & np.isnan(zero_energy); zero_energy[ze] = (frame + 1) / 100.0
        if frame % 10 == 9:
            dna = [b.rate[:, plume.reads["dna_" + s]].mean(1, keepdim=True) for s in ("L", "R")]
            neural = torch.cat([plume.turn, plume.steer_input, plume.steer_integral, plume.goal, plume.strength, plume.odor,
                                plume.bilateral, *dna], 1).detach().cpu().numpy()
            left, right = [np.asarray(sum(d.values()), float) if d else np.zeros(6) for d in sim.smell_values]
            samples.append(np.column_stack([[f.x for f in sim.flies], [f.y for f in sim.flies], [f.heading for f in sim.flies],
                                            [f.yaw_rate for f in sim.flies], energy, left, right, neural, sim.nearest_fruit()[1], sim.feeding]))
    torch.cuda.synchronize()
    data = np.asarray(samples)
    columns = ["x", "y", "heading", "yaw", "energy", "antenna_L", "antenna_R", "turn", "steer_input", "steer_integral", "goal",
               "strength", "odor", "bilateral", "dna_L", "dna_R", "fruit_distance", "feeding"]
    col = {k: i for i, k in enumerate(columns)}
    rows = []
    for i in range(6):
        rows.append(dict(row=i, env_seed=i, start=st[i], feeding_s=float(feeding[i]), first_contact_s=(None if np.isnan(first[i]) else float(first[i])),
                         path_m=float(dist[i]), mean_abs_dna_lr_hz=float(np.mean(np.abs(data[:, i, col["dna_L"]] - data[:, i, col["dna_R"]]))),
                         mean_abs_turn=float(np.mean(np.abs(data[:, i, col["turn"]]))), mean_abs_steer_input_hz=float(np.mean(np.abs(data[:, i, col["steer_input"]]))),
                         mean_abs_integral_hz=float(np.mean(np.abs(data[:, i, col["steer_integral"]]))), mean_abs_yaw_rad_s=float(np.mean(np.abs(data[:, i, col["yaw"]]))),
                         min_fruit_distance_m=float(data[:, i, col["fruit_distance"]].min()), end_energy=float(sim.metabolisms[i].energy),
                         zero_energy_s=(None if np.isnan(zero_energy[i]) else float(zero_energy[i]))))
    fed = [r for r in rows if r["feeding_s"] >= 1.0]
    metrics = dict(fed_rows=len(fed), fed_fraction=len(fed) / 6.0,
                   mean_first_contact_s=(float(np.mean([r["first_contact_s"] for r in fed])) if fed else None),
                   mean_abs_dna_lr_hz=float(np.mean([r["mean_abs_dna_lr_hz"] for r in rows])), mean_path_m=float(np.mean([r["path_m"] for r in rows])),
                   mean_abs_turn=float(np.mean([r["mean_abs_turn"] for r in rows])), mean_abs_steer_input_hz=float(np.mean([r["mean_abs_steer_input_hz"] for r in rows])),
                   mean_abs_integral_hz=float(np.mean([r["mean_abs_integral_hz"] for r in rows])), mean_abs_yaw_rad_s=float(np.mean([r["mean_abs_yaw_rad_s"] for r in rows])),
                   frames=frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), samples=data)
    record = dict(arm=arm, starts=starts, run=run, brain_seed=seed, env_seeds=list(range(6)), instruments=list(ARMS[arm]), seconds=seconds,
                  energy=ENERGY, flags=FLAGS, starts_used=st, columns=columns, rows=rows, metrics=metrics, device="cuda",
                  device_name=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  plume_record=plume.describe(), wall_s=round(time.time() - t0, 1),
                  provenance=provenance(c, fb=sim.fb, seeds=[seed], env_seeds=list(range(6)), batch=6,
                                        stimulus={"name": "plume goal-only room", "arm": arm, "starts": starts, "run": run, "seconds": seconds,
                                                  "energy": ENERGY, "fruit": "all", "fence": False, "program": "none", "flags": FLAGS}))
    out.with_suffix(".json").write_text(json.dumps(to_jsonable(record), indent=1) + "\n", encoding="utf-8")
    print(f"{arm} {starts} r{run}: fed rows {metrics['fed_rows']}, feeding_s {[round(r['feeding_s'], 2) for r in rows]}, first contact "
          f"{[r['first_contact_s'] for r in rows]}, |DNa02 L-R| {metrics['mean_abs_dna_lr_hz']:.3f} Hz, path {metrics['mean_path_m']:.3f} m, "
          f"variant {record['plume_record']['parameters'].get('variant')}, {record['wall_s']} s", flush=True)


# ------------------------------------------------------------------------------------------------ the analysis
def analyse(runs_dir: Path, out_dir: Path):
    import pandas as pd
    from flyverse.interp import common
    from cx_sign_control import stale_sources
    from cx_velocity_route import holm
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((runs_dir / "predeclared.json").read_text(encoding="utf-8")) if (runs_dir / "predeclared.json").exists() else {}
    want = frozen.get("source_sha256_lf", {})
    run_rows, row_rows, problems, seen = [], [], [], set()
    for starts in STARTS:
        for run in RUNS:
            for arm in ARMS:
                p = runs_dir / f"{arm}_{starts}_r{run}.json"
                if not p.is_file():
                    problems.append(f"{p.name}: missing"); continue
                r = json.loads(p.read_text(encoding="utf-8"))
                m, prov = r.get("metrics", {}), r.get("provenance", {})
                ex = prov.get("execution", {})
                bad = []
                if not str(ex.get("device", "")).startswith("cuda"):
                    bad.append(f"device {ex.get('device')!r}")
                be = ex.get("backend", {})
                for k in ("cuda_kernels", "event_driven", "cuda_sparse"):
                    if be.get(k) != FLAGS[k]:
                        bad.append(f"backend {k} {be.get(k)!r}")
                if prov.get("preset") != "instrumented" or [d.get("name") for d in prov.get("instruments", [])] != ["compass", "plume", "hunger", "flight"]:
                    bad.append(f"preset / instruments {prov.get('preset')} {[d.get('name') for d in prov.get('instruments', [])]}")
                pr = next((d for d in prov.get("instruments", []) if d.get("name") == "plume"), {}).get("parameters", {})
                if pr.get("variant") != VARIANT[arm]:
                    bad.append(f"plume variant {pr.get('variant')!r} != {VARIANT[arm]!r}")
                if r.get("brain_seed") != BRAIN_SEED0 + run or r.get("arm") != arm or r.get("starts") != starts:
                    bad.append("arm / starts / seed mismatch")
                if r.get("starts_used") != (frozen.get("starts", {}) or {"shipped": starts_of("shipped"), "drawn": starts_of("drawn")}).get(starts):
                    bad.append("starts differ from the frozen list")
                if m.get("frames") != int(SECONDS * 100):
                    bad.append(f"frames {m.get('frames')}")
                stale, loaded = stale_sources(prov.get("source_fingerprint", {}), want)
                if stale:
                    bad.append(f"loaded source differs from the predeclared tree: {stale[:4]}")
                if frozen and not loaded:
                    bad.append("missing source fingerprint")
                console = p.with_suffix(".txt")
                if not (console.is_file() and "device cuda" in console.read_text(encoding="utf-8", errors="replace")):
                    bad.append("console lacks 'device cuda'")
                if (arm, starts, run) in seen:
                    bad.append("duplicate")
                seen.add((arm, starts, run))
                problems.extend(f"{p.name}: {b}" for b in bad)
                run_rows.append(dict(arm=arm, starts=starts, run=run, brain_seed=r.get("brain_seed"), file=p.name, device_name=ex.get("device_name"),
                                     gpu=r.get("cuda_visible_devices"), wall_s=r.get("wall_s"), variant=pr.get("variant"),
                                     integral_gain=pr.get("steering_integral_gain_per_s"), walking_goal=pr.get("walking_goal_enabled"),
                                     **{k: (float(m[k]) if isinstance(m.get(k), (int, float)) else float("nan")) for k in RUN_KEYS}, checks="; ".join(bad)))
                for rr in r.get("rows", []):
                    row_rows.append(dict(arm=arm, starts=starts, run=run, brain_seed=r.get("brain_seed"), row=rr["row"], env_seed=rr["env_seed"],
                                         start_x=rr["start"]["x"], start_y=rr["start"]["y"], heading_deg=rr["start"]["heading_deg"],
                                         **{k: rr.get(k) for k in ROW_KEYS}))
    if len(seen) != 36:
        problems.append(f"expected 36 runs, got {len(seen)}")
    df = pd.DataFrame(run_rows); df.to_csv(out_dir / "runs.csv", index=False)
    rows_df = pd.DataFrame(row_rows); rows_df.to_csv(out_dir / "rows.csv", index=False)
    desc = []
    for (arm, starts), g in df.groupby(["arm", "starts"], sort=False):
        d = dict(arm=arm, starts=starts, n=len(g))
        for k in RUN_KEYS:
            v = g[k].to_numpy(float)
            d[f"{k}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            d[f"{k}_sd"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")
            d[f"{k}_runs"] = ",".join("nan" if not np.isfinite(x) else f"{x:.4f}" for x in v)
        desc.append(d)
    desc_df = pd.DataFrame(desc); desc_df.to_csv(out_dir / "descriptive.csv", index=False)
    comp = []
    for name, key, a, b_, starts, predicted in FAMILY:
        va = df[(df.arm == a) & (df.starts == starts)][key].to_numpy(float); vb = df[(df.arm == b_) & (df.starts == starts)][key].to_numpy(float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        rec = dict(test=name, key=key, stim=a, null=b_, starts=starts, predicted_sign=predicted, n_stim=len(va), n_null=len(vb),
                   stim_values=json.dumps([round(float(x), 4) for x in va]), null_values=json.dumps([round(float(x), 4) for x in vb]))
        if len(va) and len(vb):
            c = common.compare(va, vb)
            rec.update(verdict=c["verdict"], diff=c["diff"], z=c["z"], U=c["U"], p=c["p"], p_floor=c["p_floor"], null_sd_zero=c["null_sd_zero"], n_min=c["n_min"])
            sa, sb = np.std(va, ddof=1), np.std(vb, ddof=1)
            rec["welch_t"] = float((va.mean() - vb.mean()) / math.sqrt(sa ** 2 / len(va) + sb ** 2 / len(vb))) if (sa > 0 or sb > 0) else float("nan")
        else:
            rec.update(verdict="underpowered", diff=float("nan"), z=float("nan"), U=float("nan"), p=float("nan"), p_floor=float("nan"), null_sd_zero=None, n_min=0)
        comp.append(rec)
    adj = holm({r["test"]: r["p"] for r in comp}, m=len(FAMILY))
    for r in comp:
        r["p_holm"] = adj[r["test"]]; r["m"] = len(FAMILY)
        holm_ok = r["verdict"] == "result" and np.isfinite(r["p_holm"]) and r["p_holm"] <= 0.05
        sign_ok = (r["predicted_sign"] == "none") or (np.isfinite(r["diff"]) and ((r["diff"] > 0) if r["predicted_sign"] == "positive" else (r["diff"] < 0)))
        r["verdict_holm"] = ("result" if (holm_ok and sign_ok) else "result, opposite sign" if holm_ok else "null" if r["verdict"] == "result" else r["verdict"])
        if problems:
            r["unchecked_verdict"] = r["verdict_holm"]; r["verdict"] = r["verdict_holm"] = "undetermined (invalid batch)"
    pd.DataFrame(comp).to_csv(out_dir / "compare.csv", index=False)
    summary = dict(runs_dir=runs_dir.as_posix(), n_runs=len(df), problems=problems, family=comp, valid_batch=not problems,
                   stamped_utc=frozen.get("stamped_utc"), analysis_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    md = [f"# plume-go analysis of `{runs_dir.as_posix()}`", "", f"n_runs {len(df)}; problems {len(problems)}" + (": " + "; ".join(problems[:12]) if problems else ""), "",
          "## The predeclared family (Holm, m = 6; `compare.csv`)", "",
          "| test | key | stim | null | starts | predicted | n | verdict | verdict (Holm, sign) | diff | z (ref SD) | Welch t | p | p_holm | p_floor |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp:
        md.append(f"| {r['test']} | {r['key']} | {r['stim']} | {r['null']} | {r['starts']} | {r['predicted_sign']} | {r['n_stim']} v {r['n_null']} | {r['verdict']} | "
                  f"{r['verdict_holm']} | {r['diff']:+.4f} | {r['z']:+.2f} | {r.get('welch_t', float('nan')):+.2f} | {r['p']:.4f} | {r['p_holm']:.4f} | {r['p_floor']:.4f} |")
    md += ["", "## Per (arm, start set): mean +/- SD over the six runs, and the six run values (`descriptive.csv`)", "",
           "| arm | starts | fed rows | first contact s (fed rows) | mean abs DNa02 L-R Hz | path m | abs turn | abs PFL3 input Hz | abs integral Hz | abs yaw rad/s |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for d in desc:
        def f(k):
            return f"{d[k + '_mean']:.3f} +/- {d[k + '_sd']:.3f} [{d[k + '_runs']}]"
        md.append(f"| {d['arm']} | {d['starts']} | {f('fed_rows')} | {f('mean_first_contact_s')} | {f('mean_abs_dna_lr_hz')} | {f('mean_path_m')} | "
                  f"{f('mean_abs_turn')} | {f('mean_abs_steer_input_hz')} | {f('mean_abs_integral_hz')} | {f('mean_abs_yaw_rad_s')} |")
    md += ["", "## Per room row (`rows.csv`; one line per run: feeding s of rows 0-5 | first contact s of rows 0-5)", "",
           "| arm | starts | run | brain seed | feeding s | first contact s | min fruit distance m |", "|---|---|---|---|---|---|---|"]
    for (arm, starts, run), g in rows_df.groupby(["arm", "starts", "run"], sort=False):
        g = g.sort_values("row")
        md.append(f"| {arm} | {starts} | {run} | {int(g.brain_seed.iloc[0])} | {','.join(f'{v:.2f}' for v in g.feeding_s)} | "
                  f"{','.join('none' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:.1f}' for v in g.first_contact_s)} | "
                  f"{','.join(f'{v:.3f}' for v in g.min_fruit_distance_m)} |")
    (out_dir / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:14]))
    print(f"-> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", default="out/plume-go"); p.add_argument("--minutes", type=int, default=90); p.add_argument("--name", default="plume-go")
    p = sub.add_parser("predeclare"); p.add_argument("dir")
    p = sub.add_parser("room"); p.add_argument("--arm", choices=tuple(ARMS), required=True); p.add_argument("--starts", choices=STARTS, required=True)
    p.add_argument("--run", type=int, required=True); p.add_argument("--out", required=True); p.add_argument("--seconds", type=float, default=SECONDS)
    p = sub.add_parser("analyse"); p.add_argument("--runs", default="out/plume-go"); p.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(Path(a.out), a.minutes, a.name)
    elif a.cmd == "predeclare":
        predeclare(Path(a.dir))
    elif a.cmd == "room":
        room(a.arm, a.starts, a.run, a.out, a.seconds)
    else:
        s = analyse(Path(a.runs), Path(a.out or (Path(a.runs) / "analysis")))
        if s["problems"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
