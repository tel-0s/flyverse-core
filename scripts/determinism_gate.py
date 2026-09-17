"""Round 8, item 1 -- the deterministic-kernel gate (docs/audits/determinism_gate.md, batch det1).

Which execution paths repeat exactly on one GPU? Every protocol is run TWICE, sequentially, inside one scheduler job
pinned to one GPU (cluster_run.py --gpu-ids), and every saved array and every numeric metric of the pair is compared
EXACTLY (np.array_equal with NaN == NaN; numeric leaves ==). The rule is frozen in out/det1/predeclared.json before
submission and applied by `analyse` without a tolerance: a pair either repeats exactly or it does not.

    PYTHONIOENCODING=utf-8 python scripts/determinism_gate.py plan --out out/det1
        -> out/det1/batch.sh (DRAFT: `set -o pipefail`, ONE cluster_run.py call of 5 jobs, the docs/INTERP.md 10.3 job
           line, `--target house --node "${CLUSTER_NODE_2:?}" --gpu-ids 4,5,6,7`), out/det1/jobs.json
    PYTHONIOENCODING=utf-8 python scripts/determinism_gate.py predeclare out/det1
        -> out/det1/predeclared.json (stamped; never overwritten)
    python scripts/determinism_gate.py room --config plume|raw --path native|torch --seconds 10 --seed 0 --out STEM
        -> STEM.npz (every recorded array) + STEM.json (metrics + provenance); the job's own command
    PYTHONIOENCODING=utf-8 python scripts/determinism_gate.py analyse --runs out/det1 --out out/det1/analysis
        -> pairs.csv (one row per pair: arrays / metrics compared, equal, max |diff|), arrays.csv (one row per array of
           every pair), analysis.md, summary.json

Protocols (a run = one process; the pair = the two runs of one job):
  (a) cxS      scripts/cx_wedge.py, the round-7 S arm (raw, no instrument, prescribed turn 90 deg/s over 0.5-3.5 s),
               seed 0 -- the protocol cx8 == cx8r reproduced across two submissions. cx_wedge builds FlyBrain with
               the constructor defaults: cuda_kernels None (-> the FLYVERSE_CUDA_KERNELS env, unset on the box: off),
               event_driven None (-> False on CUDA), cuda_sparse "torch": the TORCH path, B = 1.
  (b) plume    the plume_steering_probe room: BatchSim B=6, env seeds 0-5, `compass plume hunger flight`, initial
               headings 5 / 90 / -90 / 185 / 5 / 5 deg, energy 0.1, all fruit, no fence, no program -- 10 s, brain
               seed 0; on the NATIVE path (cuda_kernels, event_driven, cuda_graphs; cuda_sparse torch, warp is B=1
               only) and on the TORCH path (no native kernels, no event selection: sparse matmul + torch LIF).
  (c) raw      the plain room: BatchSim B=6, preset raw, no instruments, no program, all fruit, default start and
               heading -- 10 s, brain seed 0; both paths.
The recorded arrays are the body trace per frame (x, y, z, heading, yaw_rate, energy, feeding, tasting, nearest
fruit distance, antenna L / R, motor turn_L / turn_R / power), the cumulative spike count per row per frame (the
first differing frame of a non-repeating pair), the per-row plume state per frame (turn, steer input, goal,
strength, DNa02 L / R) for (b), and the FINAL brain tensors (FlyBrain.BRAIN_TENSORS: v, g, rate, spike_counts, ...)
for (b) and (c); for (a) the cx_wedge ledger NPZ and its JSON row.
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

NAV = ["compass", "plume", "hunger", "flight"]
PLUME_HEADINGS_DEG = [5, 90, -90, 185, 5, 5]
PLUME_ENERGY = 0.1
PATHS = {"native": dict(cuda_graphs=True, cuda_kernels=True, event_driven=True, cuda_sparse="torch"),
         "torch": dict(cuda_graphs=True, cuda_kernels=False, event_driven=False, cuda_sparse="torch")}
# (pair label, protocol, path, the two run stems relative to the batch dir)
PAIRS = [("cxS", "cx_wedge S arm seed 0", "torch (cx_wedge default)", "cxS"),
         ("plume_native", "plume room 10 s B=6 seed 0", "native", "plume_native"),
         ("plume_torch", "plume room 10 s B=6 seed 0", "torch", "plume_torch"),
         ("raw_native", "raw room 10 s B=6 seed 0", "native", "raw_native"),
         ("raw_torch", "raw room 10 s B=6 seed 0", "torch", "raw_torch")]
# bookkeeping keys that legitimately differ between two runs of one command (never compared)
SKIP_KEYS = {"wall_s", "host", "platform", "created_utc", "stamped_utc", "started_utc", "finished_utc", "time_s",
             "ledger_npz", "sim_out", "out", "path", "npz", "json", "elapsed_s", "seconds_wall", "torch", "cuda_visible_devices"}
GPU_POOL = "4,5,6,7"
MAX_JOBS_PER_CALL = 24


def sh_job(cmds: list[str], logs: list[str], rel: str) -> str:
    """docs/INTERP.md 10.3: mkdir, venv, cuda assert, the commands (each to its own console), `st=\\$?; tail; exit \\$st`
    (\\$ escaped so THIS shell leaves it for the job's shell)."""
    body = " && ".join(f"{c} > {l} 2>&1" for c, l in zip(cmds, logs))
    tails = "; ".join(f"tail -3 {l}" for l in logs)
    return (f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"( {body} ); st=\\$?; {tails}; exit \\$st")


def cx_command(stem: str) -> str:
    return (f"python scripts/cx_wedge.py --no-structure --sim 1:1 --ledger --seed 0 --arm S --block det_cxS --receptor-model shipped "
            f"--turn 90 --turn-window 0.5:3.5 --sim-out {stem}.json")


def room_command(config: str, path: str, seconds: float, seed: int, stem: str) -> str:
    return (f"PYTHONIOENCODING=utf-8 python scripts/determinism_gate.py room --config {config} --path {path} --seconds {seconds:g} "
            f"--seed {seed} --out {stem}")


def plan(out_dir: Path, seconds: float = 10.0, minutes: int = 40, name: str = "det1"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rel = f"out/{out_dir.name}"
    if (out_dir / "predeclared.json").exists() or (out_dir / "submitted_at.txt").exists():
        raise ValueError("batch is frozen; use a new directory for a new predeclaration")
    jobs = []
    for label, protocol, path, stem in PAIRS:
        stems = [f"{rel}/{stem}_r{i}" for i in (1, 2)]
        if label == "cxS":
            cmds = [cx_command(s) for s in stems]
        else:
            config, p = label.split("_")
            cmds = [room_command(config, p, seconds, 0, s) for s in stems]
        jobs.append(dict(pair=label, protocol=protocol, path=path, runs=stems, commands=cmds,
                         line=sh_job(cmds, [f"{s}.txt" for s in stems], rel)))
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = ["#!/bin/bash", "set -o pipefail",
             f"# DRAFT -- NOT SUBMITTED. Round 8 / batch {name}: docs/audits/determinism_gate.md.",
             f"# {len(jobs)} jobs = {len(PAIRS)} pairs, each job runs its protocol TWICE sequentially on ONE pinned GPU "
             f"(--gpu-ids {GPU_POOL}: gpus 1 + gpu_ids [one id] per job), one cluster_run.py call.",
             "# The second cluster node is named by the CLUSTER_NODE_2 environment variable (docs/CLUSTER.md, git-ignored);",
             "# committed text never carries the name. Generated by scripts/determinism_gate.py plan on " + stamp + ".",
             "# Read: '5 job(s), 0 failed'; then `grep -c 'device cuda' " + rel + "/*_r?.txt` = 10; then analyse.",
             "python scripts/cluster_run.py --target house --node \"${CLUSTER_NODE_2:?set CLUSTER_NODE_2 to the name of the second cluster node}\" "
             f"--gpu-ids {GPU_POOL} --name {name} --minutes {minutes} "
             + " ".join('"' + j["line"] + '"' for j in jobs) + f" --fetch {rel}/ 2>&1 | tee {rel}/client_stdout_0.txt || exit $?"]
    (out_dir / "batch.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "jobs.json").write_text(json.dumps({"batch": name, "seconds": seconds, "gpu_pool": GPU_POOL, "node": "<cluster-node-2>",
                                                   "paths": PATHS, "pairs": jobs, "generated_utc": stamp,
                                                   "status": "DRAFT, not submitted"}, indent=1), encoding="utf-8")
    print(f"{len(jobs)} jobs (pairs) in 1 call -> {out_dir / 'batch.sh'} (DRAFT, not submitted); {out_dir / 'jobs.json'}")


def source_hashes() -> dict:
    paths = sorted(set(ROOT.glob("flyverse/**/*.py")) | set(ROOT.glob("flyverse/data/*.csv")) |
                   {Path(__file__).resolve(), ROOT / "scripts/cx_wedge.py", ROOT / "scripts/probe_compass_room.py",
                    ROOT / "scripts/cluster_run.py"})
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}


COMPARISON_RULE = {
    "unit": "the pair: two runs of one command, sequential, in one scheduler job pinned to one GPU (gpus 1, gpu_ids [id])",
    "arrays": "every array of the run's NPZ compared with numpy.array_equal(a, b, equal_nan=True) after a shape check; "
              "max |a - b| reported per array (NaN-aware; inf where shapes differ)",
    "metrics": "every numeric leaf of the run's JSON (ints, floats, bools; lists and dicts recursed) compared with ==; "
               "NaN == NaN; max |diff| over the numeric leaves reported",
    "excluded_keys": sorted(SKIP_KEYS),
    "excluded_reason": "wall-clock, host / platform, timestamps and file paths differ between two runs by construction",
    "verdict_per_pair": "`repeats exactly` iff every compared array and every compared metric is equal; otherwise "
                        "`one draw` -- there is no tolerance and no partial credit",
    "outcome_table": "per (path, protocol): repeats exactly -> numbers from that path / protocol may be quoted to the digit "
                     "as 'reproduced exactly in 2 draws on <backend>' (INTERP 10.4 rule 16: never 'bit-identical' of a GPU "
                     "number); one draw -> no number from it is quoted to more than its across-draw spread",
    "decision_for_items_3_and_4": "if the torch path repeats exactly on both rooms and the native path does not, items 3 "
                                  "and 4 run on the torch path; if neither repeats, items 3-4 are >= 6 draws with the run "
                                  "as the replicate unit and no number quoted to more than its across-draw SD; if both "
                                  "repeat, the shipped native path stays",
    "backend_check": "every run's provenance.execution.backend must carry the flags the plan names for its path "
                     "(cuda_kernels / event_driven / cuda_sparse) and provenance.execution.device must start with cuda; "
                     "a mismatch invalidates the pair",
}


def predeclare(out_dir: Path):
    out_dir = Path(out_dir)
    if (out_dir / "submitted_at.txt").exists() or list(out_dir.glob("*_r?.json")):
        raise ValueError("cannot predeclare after submission or results exist")
    plan_rec = json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))
    record = dict(plan_rec, status="PREDECLARED, not submitted", written_before_submission=True,
                  stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  batch_sha256=hashlib.sha256((out_dir / "batch.sh").read_bytes()).hexdigest(),
                  jobs_sha256=hashlib.sha256((out_dir / "jobs.json").read_bytes()).hexdigest(),
                  source_sha256_lf=source_hashes(), comparison_rule=COMPARISON_RULE,
                  replicate_unit="the pair (one job); no statistics, an exact-equality gate",
                  nothing_adopted="no default changes; the gate decides only which path later items run on and how numbers are quoted")
    with (out_dir / "predeclared.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"froze {out_dir / 'predeclared.json'} before submission")


# ------------------------------------------------------------------------------------------------ the room job
def room(config: str, path: str, seconds: float, seed: int, out: str):
    import os
    import torch
    from flyverse import BatchSim, connectome
    from flyverse.fly import FlyBrain
    from flyverse.interp.common import provenance, to_jsonable

    out = Path(out)
    if out.with_suffix(".npz").exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    assert torch.cuda.is_available(), "GPU required"
    print("device cuda", torch.cuda.get_device_name(0), "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"),
          "config", config, "path", path, flush=True)
    t0 = time.time()
    c = connectome.load(verbose=False)
    flags = PATHS[path]
    kw = dict(preset="instrumented", instruments=list(NAV)) if config == "plume" else dict(preset="raw")
    sim = BatchSim(6, c=c, seed=seed, seeds=range(6), device="cuda", program="none", fruit_set="all", fence=False, **flags, **kw)
    plume = sim.fb.instruments["plume"] if config == "plume" else None
    if config == "plume":
        for f, m, angle in zip(sim.flies, sim.metabolisms, PLUME_HEADINGS_DEG):
            f.heading = np.deg2rad(angle)
            m.energy = PLUME_ENERGY
    b = sim.brain
    frames = int(round(seconds * 100))
    body, plume_rows, spikes = [], [], []
    feeding = np.zeros(6)
    dist = np.zeros(6)
    previous = np.array([f.pos for f in sim.flies])
    first_contact = np.full(6, np.nan)
    for frame in range(frames):
        sim.step()
        current = np.array([f.pos for f in sim.flies])
        dist += np.linalg.norm(current - previous, axis=1)
        previous = current
        feeding += sim.feeding * 0.01
        newly = (sim.feeding > 0) & np.isnan(first_contact)
        first_contact[newly] = (frame + 1) / 100.0
        left, right = [np.asarray(sum(d.values()), float) if d else np.zeros(6) for d in sim.smell_values]
        m = sim.motor
        body.append(np.column_stack([[f.x for f in sim.flies], [f.y for f in sim.flies], [f.z for f in sim.flies],
                                     [f.heading for f in sim.flies], [f.yaw_rate for f in sim.flies],
                                     [mm.energy for mm in sim.metabolisms], sim.feeding, sim.tasting, sim.nearest_fruit()[1],
                                     left, right, np.broadcast_to(m.turn_L, (6,)), np.broadcast_to(m.turn_R, (6,)),
                                     np.broadcast_to(m.power, (6,))]))
        spikes.append(b.spike_counts.sum(1).detach().cpu().numpy().copy())
        if plume is not None:
            dna = [b.rate[:, plume.reads["dna_" + s]].mean(1, keepdim=True) for s in ("L", "R")]
            plume_rows.append(torch.cat([plume.turn, plume.steer_input, plume.goal, plume.strength, plume.steer_integral, *dna], 1)
                              .detach().cpu().numpy().copy())
    torch.cuda.synchronize()
    final = {}
    for k in FlyBrain.BRAIN_TENSORS:
        t = getattr(b, k, None)
        if isinstance(t, torch.Tensor) and t.numel():
            final["final__" + k] = t.detach().cpu().numpy().copy()
    arrays = dict(body=np.asarray(body), spikes_cum=np.asarray(spikes), **final)
    if plume is not None:
        arrays["plume"] = np.asarray(plume_rows)
    body_columns = ["x", "y", "z", "heading", "yaw_rate", "energy", "feeding", "tasting", "fruit_distance", "antenna_L", "antenna_R",
                    "turn_L", "turn_R", "power"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    metrics = dict(feeding_s=feeding.tolist(), path_m=dist.tolist(), first_contact_s=[None if np.isnan(v) else float(v) for v in first_contact],
                   end_energy=[float(mm.energy) for mm in sim.metabolisms], min_fruit_distance_m=np.asarray(body)[:, :, 8].min(0).tolist(),
                   final_spike_counts=spikes[-1].tolist(), final_rate_mean_hz=[float(v) for v in final["final__rate"].mean(1)],
                   final_v_mean=[float(v) for v in final["final__v"].mean(1)], frames=frames)
    record = dict(config=config, path=path, flags=flags, seconds=seconds, brain_seed=seed, env_seeds=list(range(6)),
                  initial_heading_deg=PLUME_HEADINGS_DEG if config == "plume" else [5] * 6,
                  initial_energy=PLUME_ENERGY if config == "plume" else None, device="cuda",
                  device_name=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  columns=dict(body=body_columns, spikes_cum=["cumulative spike count per row"],
                               plume=["turn", "steer_input", "goal", "strength", "steer_integral", "dna_L_hz", "dna_R_hz"]),
                  metrics=metrics, wall_s=round(time.time() - t0, 1),
                  provenance=provenance(c, fb=sim.fb, seeds=[seed], env_seeds=list(range(6)), batch=6,
                                        stimulus={"name": f"determinism gate {config} room", "seconds": seconds, "path": path,
                                                  "flags": flags, "fruit": "all", "fence": False, "program": "none"}))
    out.with_suffix(".json").write_text(json.dumps(to_jsonable(record), indent=1) + "\n", encoding="utf-8")
    print(f"{config} {path}: feeding_s {metrics['feeding_s']}, path_m {[round(v, 4) for v in dist]}, final spikes {metrics['final_spike_counts']}, "
          f"backend {record['provenance']['execution']['backend']}, {record['wall_s']} s", flush=True)


# ------------------------------------------------------------------------------------------------ the analysis
def numeric_leaves(obj, prefix=""):
    """(path, value) of every numeric leaf, skipping SKIP_KEYS at any depth."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            yield from numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from numeric_leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        yield prefix, float(obj)
    elif isinstance(obj, (int, float)) and obj is not None:
        yield prefix, float(obj)


def compare_arrays(a: dict, b: dict) -> list[dict]:
    rows = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            rows.append(dict(array=k, shape_a=str(a[k].shape) if k in a else None, shape_b=str(b[k].shape) if k in b else None,
                             equal=False, max_abs_diff=float("inf"), first_diff_index=None))
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            rows.append(dict(array=k, shape_a=str(x.shape), shape_b=str(y.shape), equal=False, max_abs_diff=float("inf"), first_diff_index=None))
            continue
        if x.dtype.kind in "biu" and y.dtype.kind in "biu":
            eq = bool(np.array_equal(x, y))
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        else:
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            eq = bool(np.array_equal(xf, yf, equal_nan=True))
            d = np.abs(xf - yf)
            d = np.where(np.isnan(xf) & np.isnan(yf), 0.0, d)
        first = None
        if not eq:
            idx = np.argwhere(~np.isclose(d, 0.0, rtol=0, atol=0) | np.isnan(d))
            first = idx[0].tolist() if len(idx) else None
        rows.append(dict(array=k, shape_a=str(x.shape), shape_b=str(y.shape), equal=eq,
                         max_abs_diff=float(np.nanmax(d)) if d.size else 0.0, first_diff_index=first))
    return rows


def compare_metrics(a, b) -> tuple[int, int, float, list]:
    la, lb = dict(numeric_leaves(a)), dict(numeric_leaves(b))
    keys = sorted(set(la) | set(lb))
    n_eq, worst, unequal = 0, 0.0, []
    for k in keys:
        if k not in la or k not in lb:
            unequal.append((k, float("inf"))); worst = float("inf"); continue
        x, y = la[k], lb[k]
        if (math.isnan(x) and math.isnan(y)) or x == y:
            n_eq += 1
        else:
            d = float("inf") if (math.isnan(x) or math.isnan(y)) else abs(x - y)
            unequal.append((k, d)); worst = max(worst, d)
    return len(keys), n_eq, worst, unequal


def load_run(stem: Path):
    """(json record, arrays) of a run; cx_wedge writes a LIST of rows and names its NPZ after the (gE, gD, seed)."""
    rec = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if isinstance(rec, list):
        rec = rec[0]
        npz = stem.parent / Path(str(rec.get("ledger_npz", ""))).name
    else:
        npz = stem.with_suffix(".npz")
    arrays = {k: v for k, v in np.load(npz, allow_pickle=False).items()} if npz.is_file() else {}
    return rec, arrays, npz


def expected_backend(path_label: str) -> dict:
    p = "torch" if path_label.startswith("torch") else "native"
    f = PATHS[p]
    return {"cuda_kernels": f["cuda_kernels"], "event_driven": f["event_driven"], "cuda_sparse": f["cuda_sparse"]}


def analyse(runs_dir: Path, out_dir: Path):
    import pandas as pd
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((runs_dir / "predeclared.json").read_text(encoding="utf-8")) if (runs_dir / "predeclared.json").exists() else None
    pair_rows, array_rows, problems = [], [], []
    for label, protocol, path, stem in PAIRS:
        stems = [runs_dir / f"{stem}_r{i}" for i in (1, 2)]
        if not all(s.with_suffix(".json").is_file() for s in stems):
            problems.append(f"{label}: missing run JSON(s)")
            pair_rows.append(dict(pair=label, protocol=protocol, path=path, verdict="missing"))
            continue
        (ra, aa, npz_a), (rb, ab, npz_b) = load_run(stems[0]), load_run(stems[1])
        arr = compare_arrays(aa, ab)
        for r in arr:
            array_rows.append(dict(pair=label, **r))
        n_m, n_eq, worst, unequal = compare_metrics(ra, rb)
        prov = [r.get("provenance", {}).get("execution", {}) for r in (ra, rb)]
        backends = [p.get("backend", {}) for p in prov]
        exp = expected_backend(path)
        bad = []
        for i, (p, be) in enumerate(zip(prov, backends)):
            if not str(p.get("device", "")).startswith("cuda"):
                bad.append(f"r{i + 1} device {p.get('device')!r}")
            for k, v in exp.items():
                if be.get(k) != v:
                    bad.append(f"r{i + 1} backend {k} {be.get(k)!r} != {v!r}")
            console = stems[i].with_suffix(".txt")
            if not (console.is_file() and "device cuda" in console.read_text(encoding="utf-8", errors="replace")):
                bad.append(f"r{i + 1} console lacks 'device cuda'")
        devnames = [p.get("device_name") for p in prov]
        if len(set(devnames)) != 1:
            bad.append(f"device names differ {devnames}")
        # the code the run LOADED is the predeclared tree, file for file (a target checkout behind origin/main runs its
        # own stale copy of every file outside the shipped diff; cluster_run.py --ship names what must be copied)
        want = (frozen or {}).get("source_sha256_lf", {})
        for i, r in enumerate((ra, rb)):
            fp = r.get("provenance", {}).get("source_fingerprint", {})
            loaded = dict(fp.get("files", {})); loaded.update(fp.get("files_loaded", {}))
            stale = sorted(f for f, h in loaded.items() if f in want and want[f] != h)
            if stale:
                bad.append(f"r{i + 1} loaded {len(stale)} file(s) that differ from the predeclared tree: {stale[:6]}")
            if want and not loaded:
                bad.append(f"r{i + 1} has no source fingerprint")
        problems.extend(f"{label}: {b}" for b in bad)
        arrays_equal = all(r["equal"] for r in arr) and bool(arr)
        verdict = "invalid" if bad else ("repeats exactly" if (arrays_equal and n_eq == n_m) else "one draw")
        spikes = next((r for r in arr if r["array"] == "spikes_cum"), None)
        pair_rows.append(dict(pair=label, protocol=protocol, path=path, verdict=verdict,
                              n_arrays=len(arr), n_arrays_equal=sum(r["equal"] for r in arr),
                              max_abs_diff_arrays=max((r["max_abs_diff"] for r in arr), default=0.0),
                              n_metrics=n_m, n_metrics_equal=n_eq, max_abs_diff_metrics=worst,
                              first_diff_frame_spikes=(spikes["first_diff_index"][0] if spikes and spikes["first_diff_index"] else None),
                              worst_metrics=json.dumps(sorted(unequal, key=lambda kv: -kv[1])[:5]),
                              device_name=devnames[0], backend=json.dumps(backends[0], sort_keys=True), cuda_visible=[r.get("cuda_visible_devices") for r in (ra, rb)],
                              npz=[npz_a.name, npz_b.name], problems="; ".join(bad)))
    pairs = pd.DataFrame(pair_rows); pairs.to_csv(out_dir / "pairs.csv", index=False)
    arrays = pd.DataFrame(array_rows); arrays.to_csv(out_dir / "arrays.csv", index=False)
    by_path = {}
    for r in pair_rows:
        if r["pair"] == "cxS":
            continue
        by_path.setdefault(r["path"], []).append(r["verdict"])
    torch_ok = all(v == "repeats exactly" for v in by_path.get("torch", [])) and bool(by_path.get("torch"))
    native_ok = all(v == "repeats exactly" for v in by_path.get("native", [])) and bool(by_path.get("native"))
    decision = ("torch path for items 3-4 (torch repeats exactly on both rooms; native does not)" if (torch_ok and not native_ok) else
                "shipped native path stays (both repeat exactly)" if (torch_ok and native_ok) else
                "native path for items 3-4 (native repeats exactly; torch does not)" if (native_ok and not torch_ok) else
                "neither path repeats: items 3-4 are >= 6 draws, run = replicate unit, nothing quoted beyond its across-draw SD")
    if problems:
        decision = "UNDETERMINED (invalid or missing pairs: " + "; ".join(problems[:6]) + ")"
    summary = dict(runs_dir=runs_dir.as_posix(), pairs=pair_rows, problems=problems, decision=decision,
                   torch_repeats=torch_ok, native_repeats=native_ok,
                   frozen_rule=(frozen or {}).get("comparison_rule"), stamped_utc=(frozen or {}).get("stamped_utc"),
                   analysis_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    md = [f"# det1 analysis of `{runs_dir.as_posix()}`", "",
          f"problems {len(problems)}" + (": " + "; ".join(problems) if problems else ""), "",
          f"**Decision (frozen rule): {decision}**", "",
          "| pair | protocol | path | verdict | arrays equal / n | max abs diff (arrays) | metrics equal / n | max abs diff (metrics) | first diff frame (spikes) | device |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in pair_rows:
        if r["verdict"] == "missing":
            md.append(f"| {r['pair']} | {r['protocol']} | {r['path']} | missing | | | | | | |")
            continue
        md.append(f"| {r['pair']} | {r['protocol']} | {r['path']} | **{r['verdict']}** | {r['n_arrays_equal']} / {r['n_arrays']} | "
                  f"{r['max_abs_diff_arrays']:.6g} | {r['n_metrics_equal']} / {r['n_metrics']} | {r['max_abs_diff_metrics']:.6g} | "
                  f"{r['first_diff_frame_spikes']} | {r['device_name']} |")
    md += ["", "## Per array (`arrays.csv`)", "", "| pair | array | shape | equal | max abs diff | first differing index |", "|---|---|---|---|---|---|"]
    for r in array_rows:
        md.append(f"| {r['pair']} | {r['array']} | {r['shape_a']} | {r['equal']} | {r['max_abs_diff']:.6g} | {r['first_diff_index']} |")
    (out_dir / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:len(pair_rows) + 8]))
    print(f"-> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", default="out/det1"); p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--minutes", type=int, default=40); p.add_argument("--name", default="det1")
    p = sub.add_parser("predeclare"); p.add_argument("dir")
    p = sub.add_parser("room"); p.add_argument("--config", choices=("plume", "raw"), required=True)
    p.add_argument("--path", choices=tuple(PATHS), required=True); p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", required=True)
    p = sub.add_parser("analyse"); p.add_argument("--runs", default="out/det1"); p.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(Path(a.out), a.seconds, a.minutes, a.name)
    elif a.cmd == "predeclare":
        predeclare(Path(a.dir))
    elif a.cmd == "room":
        room(a.config, a.path, a.seconds, a.seed, a.out)
    elif a.cmd == "analyse":
        s = analyse(Path(a.runs), Path(a.out or (Path(a.runs) / "analysis")))
        if s["problems"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
