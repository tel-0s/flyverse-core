#!/usr/bin/env python
"""The monoamine slow class at data-anchored magnitudes: baseline activity, the plain fly, the suite (docs/audits/monoamine_slow_term.md).

    python scripts/probe_monoamines.py arms                                             # the arm table (LIFParams / benchmark flags)
    python scripts/probe_monoamines.py batch --name mono --minutes 45 [--runs 5]        # write out/monoamines/cmds.txt + the cluster_run line
    # GPU (one job per arm x run; the job = rollout + health + benchmark sections):
    python scripts/probe_monoamines.py record --arm add_mid --run 2 --block fam_r2 --out out/monoamines/runs/r2_add_mid
    # CPU:
    PYTHONIOENCODING=utf-8 python scripts/probe_monoamines.py analyse --runs "out/monoamines/runs" --json out/monoamines/analysis.json
    # CPU: the benchmark's seed-locked taste section replayed under one arm, with the realised tone g_slow per type
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python scripts/probe_monoamines.py taste --arm add_low --out out/monoamines/taste/add_low.json --null out/monoamines/taste/off.json

Arms (`ARMS`): `off` = the shipped default (LIFParams(): receptor_model 'sign', net rule 'abs', no slow term); the others
= receptor_model 'full' on the SAME fast weights (`receptor_gain` 1/1/1 = the 'sign' weights; docs/audits/slow_term.md 5b)
with the monoamine class at the scale / mode named, classical class 0 (its default). Every arm runs on the eager Torch
path (the slow term has no native kernel; the off arm is run on the same path so the backend is not a factor).

`record` (GPU): (1) rest guard -- the batched Brain runs 500 ms with no input and spikes/step is measured (scripts/benchmark.py
sec_rest's quantity; bound < 5): above `--abort-rest` (10 x 5 = 50) the arm is ABORTED (status 'aborted_rest', exit 0,
no rollout, no benchmark). (2) the plain-fly rollout: BatchSim(B, program 'none', fence on, fruit 'all') for `--seconds`,
every frame's spikes/step recorded (abort above `--abort-rollout`, status 'aborted_rollout'), the health target set
(descending / ascending neurons, CX, mushroom body, antennal lobe, VNC motor, every monoaminergic cell, MN9) captured
every `--every` frames as rate_hz + spike_count; per fly: yaw-rate SD and mean |yaw| on walking frames, straightness,
walking speed, DNa02 R-L (the readout's rates), leg L-R, take-offs (escape / voluntary), airborne fraction. (3) the health
tool (flyverse.interp.health) per batch row over the window [--skip, --seconds], averaged over rows into one per-run
per-type table (`<out>_health_type.json`) and one per-module table (`<out>_health_module.json`); B rows of one BatchSim
are ONE draw layout (docs/BATCH_SIM.md), the replicate unit is the job. (4) `scripts/benchmark.py --eager --sections
rest,taste,smell,walk,loom_escape --seeds 0,1,2` with the arm's flags (`<out>_bench.json` / `.txt`); the probe's exit
code is the benchmark's. Every JSON carries common.provenance (resolved LIFParams / OpticParams, realised device, cache
fingerprint).

`analyse` (CPU): per arm over runs, common.compare against `off` (runs are the replicates; `result` needs >= 4 runs and
|z| >= 3, p <= 0.05): behaviour, health per type (rate_mean / silent_frac; which types leave 0 Hz), per module, the
benchmark checks (status counts and measured values with scatter), the runaway record (rest / rollout spikes per step,
aborted arms). Writes a Result (tool 'health') with those tables and the md5 of every input file.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

UNIT_GAIN = {"none": 1.0, "low": 1.0, "mid": 1.0, "high": 1.0}      # 'full' on the shipped 'sign' fast weights
REST_BOUND = 5.0                                                     # scripts/benchmark.py REFERENCES rest.spikes_per_step (< 5)
# The bracket (docs/audits/monoamine_slow_term.md section 2; out/monoamines/anchor_bracket.csv): 0.02 is the hand-set
# default and the KC-anchored ceiling (Cohn 2015: no DAN tone on KCs; 0.7 mV per KC at 10 Hz); 1.0 is the LOW edge of the
# octopamine visual-gain anchor on the best-loaded HS cell (HSS: 3.5 mV at 10 Hz); 0.2 is the geometric middle.
ARMS = {
    "off": {"lif": {}, "bench": ["--receptor-model", "default"], "mode": None, "gain": 0.0},
    "add_low": {"mode": "additive", "gain": 0.02},
    "add_mid": {"mode": "additive", "gain": 0.2},
    "add_high": {"mode": "additive", "gain": 1.0},
    "gain_mid": {"mode": "gain", "gain": 0.2},
}
for _k, _a in ARMS.items():
    if _k != "off":
        _a["lif"] = dict(receptor_model="full", receptor_net_rule="abs", receptor_gain=dict(UNIT_GAIN), slow_mode=_a["mode"], slow_gain=float(_a["gain"]))
        _a["bench"] = ["--receptor-model", "full", "--receptor-net-rule", "abs", "--receptor-gain", "1,1,1", "--slow-mode", _a["mode"], "--slow-gain-monoamine", str(_a["gain"])]
BENCH_SECTIONS = "rest,taste,smell,walk,loom_escape"
HEALTH_TARGET = ["superclass=descending_neuron", "superclass=ascending_neuron", "class=CX", "module=mushroom_body", "module=antennal_lobe",
                 "superclass=vnc_motor", "nt:dopamine|octopamine|serotonin", "MN9"]
HEALTH_STATS = ("rate_mean", "silent_frac", "rate_max", "spike_rate_hz")
BEHAVIOUR_KEYS = ("yaw_sd_deg_s", "yaw_abs_deg_s", "straightness", "speed_mm_s", "dna02_rl_hz", "dna02_abs_rl_hz", "leg_lr_hz", "takeoffs",
                  "takeoffs_escape", "takeoffs_voluntary", "airborne_frac", "spikes_per_step_mean", "spikes_per_step_max", "rest_spikes_per_step")


def md5(path) -> str:
    h = hashlib.md5(); h.update(Path(path).read_bytes()); return h.hexdigest()


def patch_lif(overrides: dict):
    """Every brain.LIFParams built from here on (BatchSim's included) carries the arm (scripts/probe_walk_straightness.py's pattern)."""
    from flyverse import brain
    L = brain.LIFParams

    def make(**kw):
        for k, v in overrides.items():
            kw.setdefault(k, v)
        return L(**kw)
    brain.LIFParams = make
    return L


def lif_for(arm: str):
    from flyverse import brain
    return brain.LIFParams(**ARMS[arm]["lif"])


# ---------------------------------------------------------------------------------------------- arms
def cmd_arms(args):
    from flyverse import brain
    for name, a in ARMS.items():
        p = brain.LIFParams(**a["lif"]); spec = brain._slow_spec(p)
        print(f"{name:9s} receptor_model {p.receptor_model!s:5s} net {p.receptor_net_rule} gain_classes {brain._receptor_gain(p)} slow {('off' if spec is None else f'{spec.mode} {spec.gain} tau {spec.tau}')}; "
              f"benchmark flags {' '.join(a['bench'])}")


# ---------------------------------------------------------------------------------------------- batch
def cmd_batch(args):
    out = ROOT / "out" / "monoamines"; out.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in range(args.runs):
        for arm in ARMS:
            stem = f"out/monoamines/runs/r{r}_{arm}"
            lines.append(f"mkdir -p out/monoamines/runs && source .venv/bin/activate && python scripts/probe_monoamines.py record --arm {arm} --run {r} --block fam_r{r} "
                         f"--batch {args.batch} --seconds {args.seconds} --out {stem} > {stem}.txt 2>&1; st=$?; tail -4 {stem}.txt; exit $st")
    (out / "cmds.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'cmds.txt'} ({len(lines)} job lines; blocks fam_r0..fam_r{args.runs - 1}, every arm of a replicate on one box)")
    print(f"python scripts/cluster_run.py --name {args.name} --minutes {args.minutes} --arm-block fam $(cat out/monoamines/cmds.txt as separate args) --fetch out/monoamines/runs/")


# ---------------------------------------------------------------------------------------------- record
def cmd_record(args):
    import torch
    from flyverse import brain, connectome
    from flyverse.interp import common, health as H
    from flyverse.interp.common import Recorder
    gpu = torch.cuda.is_available() and args.device != "cpu"
    if not gpu and not args.allow_cpu:
        raise SystemExit("record needs a GPU (pass --allow-cpu for a smoke test)")
    arm = ARMS[args.arm]
    t0 = time.time()
    L0 = patch_lif(arm["lif"])
    from flyverse.batch_sim import BatchSim
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    env_seeds = list(range(args.run * 1000, args.run * 1000 + args.batch))
    sim = BatchSim(args.batch, seed=args.run, seeds=env_seeds, c=c, program="none", fruit_set="all", fence=not args.no_fence,
                   cuda_graphs=False, cuda_kernels=False, event_driven=False, cuda_sparse=args.cuda_sparse, device=args.device)
    b = sim.fb.brain; lp = b.p; op = sim.fb.optic.p if sim.fb.optic is not None else None
    b.record_activity = True
    spec = brain._slow_spec(lp)
    S_cpu = getattr(b, "_W_slow_cpu", None)
    slow_entries = {k: {"entries": int(S.nnz), "syn_eq_mv": float(np.abs(S.data).sum())} for k, S in S_cpu.items()} if isinstance(S_cpu, dict) else None
    print(f"arm {args.arm} run {args.run}: receptor_model {lp.receptor_model} ({lp.receptor_net_rule}), gain classes {brain._receptor_gain(lp)}, "
          f"slow {'off' if spec is None else f'{spec.mode} {spec.gain} tau {spec.tau} ms'} (entries {slow_entries}); device {b.device} "
          f"(kernels {b.cuda}, event_driven {b.event_driven}, cuda_graphs {sim.fb.cuda_graphs}); B {args.batch} env seeds {env_seeds}; setup {time.time() - t0:.0f} s", flush=True)
    steps_per_frame = int(round(BatchSim.FRAME_MS / lp.dt))
    # ---- (1) rest guard: the unstimulated batched Brain for 500 ms (sec_rest's quantity)
    n_rest = int(round(500.0 / lp.dt))
    s0 = b.spike_counts.sum(dim=1).detach().cpu().numpy().copy()
    b.step(n_rest)
    s1 = b.spike_counts.sum(dim=1).detach().cpu().numpy().copy()
    rest_sps = (s1 - s0) / n_rest
    rest_last = np.asarray(b.total_spikes(), dtype=float).reshape(-1)
    status = "ok"
    print(f"rest: spikes/step over 500 ms {rest_sps.mean():.3f} (rows {np.round(rest_sps, 3).tolist()}; last step {rest_last.mean():.2f}); bound {REST_BOUND}, abort above {args.abort_rest}", flush=True)
    if rest_sps.mean() > args.abort_rest:
        status = "aborted_rest"
    # ---- (2) rollout
    target = common.resolve(c, HEALTH_TARGET)
    rec = Recorder(c, target, quantities=("rate_hz", "spike_count"))
    n = int(round(args.seconds * 100)); B = args.batch
    heading = np.zeros((n, B)); pos = np.zeros((n, B, 3)); speed = np.zeros((n, B)); yaw = np.zeros((n, B)); airborne = np.zeros((n, B), bool)
    dna_l = np.zeros((n, B)); dna_r = np.zeros((n, B)); leg_l = np.zeros((n, B)); leg_r = np.zeros((n, B)); sps = np.zeros((n, B))
    wall = 0.0; frames_done = 0
    if status == "ok":
        t1 = time.time(); prev = b.spike_counts.sum(dim=1).detach().cpu().numpy().copy()
        for k in range(n):
            sim.step()
            cur = b.spike_counts.sum(dim=1).detach().cpu().numpy(); sps[k] = (cur - prev) / steps_per_frame; prev = cur.copy()
            for i, f in enumerate(sim.flies):
                heading[k, i] = float(f.heading); pos[k, i] = f.pos; speed[k, i] = float(f.speed); yaw[k, i] = float(f.yaw_rate); airborne[k, i] = bool(f.airborne)
                r = sim.commands[i]["rates"]
                dna_l[k, i] = r["DNa02_L"]; dna_r[k, i] = r["DNa02_R"]; leg_l[k, i] = r["legMN_L"]; leg_r[k, i] = r["legMN_R"]
            if k % args.every == 0:
                rec.capture(sim.fb, t_ms=(k + 1) * BatchSim.FRAME_MS, motor=sim.motor)   # rollout time (the brain's clock includes the 500 ms rest phase)
            frames_done = k + 1
            run_mean = sps[max(0, k - 49):k + 1].mean()                                # 50-frame (0.5 s) running mean over the rows
            if k >= 50 and run_mean > args.abort_rollout:                              # never on the vision-onset transient (frames 1-10 reach ~300 in the off arm)
                status = "aborted_rollout"
                print(f"  ABORT at t {(k + 1) / 100:.2f} s: 0.5 s running mean {run_mean:.1f} spikes/step > {args.abort_rollout}", flush=True)
                break
            if (k + 1) % 500 == 0:
                print(f"  t {(k + 1) / 100:5.1f} s  spikes/step {sps[max(0, k - 499):k + 1].mean():7.2f}  airborne {airborne[k].mean():.2f}  "
                      f"speed {speed[k].mean() * 1000:.1f} mm/s  {(time.time() - t1) / (k + 1) * 1000:.0f} ms/frame", flush=True)
        wall = time.time() - t1
    # ---- behaviour per fly over [skip, done] (an aborted rollout is scored over its second half, and says so)
    skip_s = args.skip if frames_done >= 2 * args.skip * 100 else 0.5 * frames_done / 100.0
    k0 = min(int(round(skip_s * 100)), max(frames_done - 1, 0)); k1 = frames_done
    flies = []
    for i in range(B):
        w = ~airborne[k0:k1, i]
        path = float(np.linalg.norm(np.diff(pos[k0:k1, i, :2], axis=0), axis=1).sum()) if k1 - k0 > 1 else 0.0
        net = float(np.linalg.norm(pos[k1 - 1, i, :2] - pos[k0, i, :2])) if k1 > k0 else 0.0
        yw = np.degrees(yaw[k0:k1, i][w])
        flies.append({"row": i, "env_seed": env_seeds[i], "yaw_sd_deg_s": float(yw.std()) if yw.size > 1 else None, "yaw_abs_deg_s": float(np.abs(yw).mean()) if yw.size else None,
                      "straightness": net / path if path > 0 else None, "path_m": path, "net_m": net,
                      "speed_mm_s": float(speed[k0:k1, i][w].mean() * 1000) if w.any() else None,
                      "dna02_rl_hz": float((dna_r[k0:k1, i] - dna_l[k0:k1, i]).mean()) if k1 > k0 else None, "dna02_abs_rl_hz": float(np.abs(dna_r[k0:k1, i] - dna_l[k0:k1, i]).mean()) if k1 > k0 else None,
                      "dna02_mean_hz": float((0.5 * (dna_r[k0:k1, i] + dna_l[k0:k1, i])).mean()) if k1 > k0 else None,
                      "leg_lr_hz": float((leg_l[k0:k1, i] - leg_r[k0:k1, i]).mean()) if k1 > k0 else None,
                      "takeoffs": int(np.sum(np.diff(airborne[:k1, i].astype(int)) == 1)), "takeoffs_escape": int(sim.hops_escape[i]), "takeoffs_voluntary": int(sim.hops_voluntary[i]),
                      "airborne_frac": float(airborne[k0:k1, i].mean()) if k1 > k0 else None,
                      "spikes_per_step_mean": float(sps[k0:k1, i].mean()) if k1 > k0 else None, "spikes_per_step_max": float(sps[:k1, i].max()) if k1 > 0 else None,
                      "rest_spikes_per_step": float(rest_sps[i])})
    def agg(key):
        v = [f[key] for f in flies if f[key] is not None]
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)} if v else None
    behaviour = {k: agg(k) for k in BEHAVIOUR_KEYS}
    # ---- (3) health per row, averaged over rows
    recording = rec.finish(meta={"protocol": "plain_fly_rollout", "seed": args.run, "label": "health_target", "selection": HEALTH_TARGET, "arm": args.arm})
    stimulus = {"protocol": "plain_fly_rollout", "params": {"arm": args.arm, "lif_overrides": common.to_jsonable(arm["lif"]), "seconds": args.seconds, "skip_s": args.skip,
                                                            "fence": not args.no_fence, "fruit_set": "all", "program": "none", "batch": B, "every_frames": args.every,
                                                            "rest_guard_ms": 500, "abort_rest": args.abort_rest, "abort_rollout": args.abort_rollout},
                "control": {"arm": "off", "note": "the shipped LIFParams() on the same eager Torch path"}}
    prov = common.provenance(c, lp, op, fb=sim.fb, device=args.device, seeds=[args.run], env_seeds=env_seeds, batch=B,
                             backend=dict(cuda_graphs=False, cuda_kernels=False, event_driven=False, cuda_sparse=args.cuda_sparse), stimulus=stimulus, cache_dir=args.cache_dir)
    recording.meta["provenance"] = prov
    health_files = {}
    if status != "aborted_rest" and frames_done > k0 + 1:
        th = time.time()
        ew = common.effective_weights(c, lp)
        counts, _ = H.full_counts(c)
        for by in ("type", "module"):
            per = []
            for i in range(B):
                R = recording.row(i)
                res = H.health(R, c=c, params=lp, window=(skip_s, frames_done / 100.0), by=by, ew=ew, fb=sim.fb, counts=counts, readout_rows=False)
                per.append(res.table("per_type").assign(row=i))
                last = res
            import pandas as pd
            allp = pd.concat(per, ignore_index=True)
            num = [k for k in allp.columns if allp[k].dtype.kind in "fi" and k not in ("row",)]
            mean = allp.groupby("group", sort=True).agg({**{k: "mean" for k in num}, "state_flags": lambda s: "|".join(sorted({f for x in s.fillna("") for f in x.split("|") if f})),
                                                          "by": "first", "unit_kind": "first"}).reset_index()
            mean["silent_all_rows"] = allp.groupby("group").silent_frac.apply(lambda s: bool((s >= 1.0).all())).to_numpy()
            out = common.Result.new("health", prov)
            out.add_table("per_type", mean); out.add_table("per_type_rows", allp)
            out.summary = dict(last.summary, by=by, rows_averaged=B, arm=args.arm, run=args.run, status=status, window_s=[skip_s, frames_done / 100.0],
                               note="per-row health() averaged over the B rows of one BatchSim (one draw layout); replicate unit = the job; an aborted rollout is scored over its second half")
            out.replicates = {"n": 1, "unit": "runs", "runs": [{"run_index": args.run, "seed": args.run, "file": str(args.out) + ".json", "device": prov["execution"]["device"]}], "null": None}
            out.files = {"generator": "scripts/probe_monoamines.py record", "run": str(args.out) + ".json"}
            path = Path(str(args.out) + f"_health_{by}.json"); out.save(path); health_files[by] = str(path)
        print(f"health: {len(target)} target cells, window [{skip_s}, {frames_done / 100:.1f}] s, {time.time() - th:.0f} s", flush=True)
    # ---- slim per-cell means (raw data) and the run JSON
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rate = recording.quantities["rate_hz"]; kk = rate.shape[0]
    wk = int(np.searchsorted(recording.t_ms, skip_s * 1000.0)) if kk else 0
    np.savez_compressed(str(args.out) + "_cells.npz", idx=recording.idx, body_ids=recording.body_ids, types=recording.types, window_s=np.array([skip_s, frames_done / 100.0]),
                        rate_mean=rate[wk:].mean(axis=0).reshape(B, -1) if kk > wk else np.zeros((B, len(target)), np.float32),
                        rate_max=rate[wk:].max(axis=0).reshape(B, -1) if kk > wk else np.zeros((B, len(target)), np.float32),
                        t_ms=recording.t_ms, spikes_per_step=sps[:frames_done].astype(np.float32), airborne=airborne[:frames_done],
                        heading=heading[:frames_done].astype(np.float32), pos=pos[:frames_done].astype(np.float32), speed=speed[:frames_done].astype(np.float32),
                        dna02_L=dna_l[:frames_done].astype(np.float32), dna02_R=dna_r[:frames_done].astype(np.float32))
    run = {"arm": args.arm, "run": args.run, "status": status, "frames_done": frames_done, "seconds": args.seconds, "skip_s": skip_s, "window_s": [skip_s, frames_done / 100.0], "batch": B, "env_seeds": env_seeds,
           "rest_spikes_per_step": {"mean": float(rest_sps.mean()), "rows": rest_sps.tolist(), "bound": REST_BOUND, "abort_above": args.abort_rest},
           "rollout_spikes_per_step": {"mean": float(sps[k0:k1].mean()) if k1 > k0 else None, "max_frame": float(sps[:k1].max()) if k1 else None, "abort_above": args.abort_rollout},
           "behaviour": behaviour, "flies": flies, "slow": None if spec is None else {"mode": spec.mode, "gain": spec.gain, "tau": spec.tau, "entries": slow_entries},
           "lif": common.to_jsonable(dataclasses.asdict(lp)), "device": str(b.device), "wall_s": round(wall, 1), "health_target_cells": int(len(target)),
           "files": {"health": health_files, "cells": str(args.out) + "_cells.npz", "generator": "scripts/probe_monoamines.py " + " ".join(sys.argv[1:])}, "provenance": prov}
    print(f"behaviour: " + ", ".join(f"{k} {v['mean']:.3f}+-{v['sd']:.3f}" for k, v in behaviour.items() if v), flush=True)
    del sim
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ---- (4) the benchmark sections
    bench_rc = 0
    if status == "ok" and not args.no_bench:
        bj = str(args.out) + "_bench.json"; bt = str(args.out) + "_bench.txt"
        cmd = [sys.executable, str(ROOT / "scripts" / "benchmark.py"), "--eager", "--sections", BENCH_SECTIONS, "--seeds", "0,1,2", "--json", bj] + arm["bench"]
        if args.cache_dir:
            cmd += ["--cache-dir", args.cache_dir]
        print("benchmark:", " ".join(cmd), flush=True)
        with open(bt, "w", encoding="utf-8") as f:
            bench_rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        run["files"]["bench"] = bj; run["bench_exit_code"] = bench_rc
        tail = Path(bt).read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
        print("\n".join(tail), flush=True)
    Path(str(args.out) + ".json").write_text(json.dumps(common.to_jsonable(run), indent=1), encoding="utf-8")
    print(f"wrote {args.out}.json (status {status}, device {run['device']}, rollout {wall:.0f} s, benchmark exit {bench_rc})", flush=True)
    return bench_rc


# ---------------------------------------------------------------------------------------------- analyse
def _local(path) -> Path:
    """A path recorded by a job (relative to the repo root on the box) resolved against this checkout."""
    q = Path(str(path).replace("\\", "/"))
    return q if q.is_absolute() else ROOT / q


def _load_runs(pattern):
    runs = []
    for p in sorted(glob.glob(str(Path(pattern) / "*.json"))):
        name = Path(p).name
        if name.endswith(("_health_type.json", "_health_module.json", "_bench.json")):
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if "arm" in d and "behaviour" in d:
            d["_path"] = p; runs.append(d)
    return runs


def cmd_analyse(args):
    import pandas as pd
    from flyverse.interp import common, health as H
    runs = _load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no run JSONs under {args.runs}")
    by_arm = {}
    for d in runs:
        by_arm.setdefault(d["arm"], []).append(d)
    if "off" not in by_arm:
        raise SystemExit("no 'off' arm")
    print(f"{len(runs)} run(s) from {args.runs}: " + ", ".join(f"{a} n={len(v)}" for a, v in by_arm.items()))
    for a, v in by_arm.items():
        print(f"  {a}: " + "; ".join(f"r{d['run']} {d['status']} dev {d['device']} rest {d['rest_spikes_per_step']['mean']:.2f} roll {d['rollout_spikes_per_step']['mean']}" for d in v))
    files = {Path(d["_path"]).name: md5(d["_path"]) for d in runs}
    # ---- behaviour
    beh = []
    for a, v in by_arm.items():
        for k in BEHAVIOUR_KEYS:
            vals = [d["behaviour"][k]["mean"] for d in v if d["behaviour"].get(k)]
            null = [d["behaviour"][k]["mean"] for d in by_arm["off"] if d["behaviour"].get(k)]
            row = {"arm": a, "key": k, "n": len(vals), "mean": float(np.mean(vals)) if vals else None, "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "values": vals}
            if a != "off" and vals and null:
                cmp = common.compare(np.asarray(vals, float), np.asarray(null, float))
                row.update(diff=cmp["diff"], z=cmp["z"], p=cmp["p"], verdict=cmp["verdict"])
            beh.append(row)
    beh_df = pd.DataFrame(beh)
    # ---- health per type / module
    health_tabs = {}; movers = {}; leave_zero = {}
    for by in ("type", "module"):
        res = {a: [common.Result.load(_local(d["files"]["health"][by])) for d in v if d["files"].get("health", {}).get(by) and _local(d["files"]["health"][by]).exists()] for a, v in by_arm.items()}
        for a, rr in res.items():
            for r in rr:
                files[Path(r.files.get("run", "")).name + f"_health_{by}"] = None
        rep = {a: H.replicate_table(rr) for a, rr in res.items() if rr}
        health_tabs[f"replicates_{by}"] = pd.concat([t.assign(arm=a) for a, t in rep.items()], ignore_index=True) if rep else pd.DataFrame()
        comp = []
        for a, rr in res.items():
            if a == "off" or not rr or not res["off"]:
                continue
            t = H.compare_arms(rr, res["off"], stats=HEALTH_STATS).assign(arm=a)
            comp.append(t)
        comp_df = pd.concat(comp, ignore_index=True) if comp else pd.DataFrame()
        health_tabs[f"compare_{by}"] = comp_df
        if len(comp_df):
            sil = comp_df[comp_df.stat == "silent_frac"]
            leave_zero[by] = sil[(sil.null_mean >= 0.999) & (sil.stim_mean < 0.999)][["arm", "group", "stim_mean", "null_mean", "stim_n", "null_n", "verdict"]]
            rm = comp_df[comp_df.stat == "rate_mean"].copy()
            rm["abs_diff"] = rm["diff"].abs()
            movers[by] = rm.sort_values("abs_diff", ascending=False).groupby("arm").head(args.top)
    # ---- benchmark
    bench_rows = []; bench_cmp = []
    checks = {}
    for a, v in by_arm.items():
        for d in v:
            bj = d.get("files", {}).get("bench")
            if not bj or not _local(bj).exists():
                continue
            bj = _local(bj); files[bj.name] = md5(bj)
            bd = json.loads(bj.read_text(encoding="utf-8"))
            for ch in bd["checks"]:
                checks.setdefault((a, ch["key"]), []).append((ch["status"], ch["measured"], d["run"]))
            bench_rows.append({"arm": a, "run": d["run"], "device": bd["config"].get("device"), "backend": bd["config"].get("backend"), "runtime_min": bd.get("total_runtime_s", 0) / 60.0,
                               "rest_spikes_per_step": bd["sections"].get("rest", {}).get("spikes_per_step"), "KC_hz": bd["sections"].get("smell", {}).get("KC_hz"),
                               "KC_active": bd["sections"].get("smell", {}).get("KC_active"), "PN_hz": bd["sections"].get("smell", {}).get("PN_hz"),
                               "taste_MN9": bd["sections"].get("taste", {}).get("MN9_hz"), "walk_GF_max": bd["sections"].get("walk", {}).get("walk", {}).get("GF_max_hz"),
                               "walk_power_max": bd["sections"].get("walk", {}).get("walk", {}).get("power_max_hz"), "walk_power_sustained": bd["sections"].get("walk", {}).get("walk", {}).get("power_sustained_hz"),
                               "loom_GF_peak": bd["sections"].get("walk", {}).get("loom", {}).get("GF_peak_hz"), "loom_escape_peak": bd["sections"].get("loom_escape", {}).get("GF_peak_hz"),
                               "loom_escapes": bd["sections"].get("loom_escape", {}).get("escapes"),
                               "hops_before_loom": sum(s.get("hops_before_loom", 0) for s in bd["sections"].get("loom_escape", {}).get("seeds", {}).values()) if "loom_escape" in bd["sections"] else None,
                               "tally": f"{sum(c['status'] == 'PASS' for c in bd['checks'])}/{sum(c['status'] == 'FAIL' for c in bd['checks'])}/{sum(c['status'] == 'KNOWN GAP' for c in bd['checks'])}",
                               "fails": [c["key"] for c in bd["checks"] if c["status"] == "FAIL"]})
    keys = sorted({k for _, k in checks})
    for a in by_arm:
        for k in keys:
            v = checks.get((a, k), [])
            if not v:
                continue
            meas = [m for _, m, _ in v if isinstance(m, (int, float))]
            null = [m for _, m, _ in checks.get(("off", k), []) if isinstance(m, (int, float))]
            row = {"arm": a, "check": k, "n": len(v), "pass": sum(s == "PASS" for s, _, _ in v), "fail": sum(s == "FAIL" for s, _, _ in v), "gap": sum(s == "KNOWN GAP" for s, _, _ in v),
                   "mean": float(np.mean(meas)) if meas else None, "sd": float(np.std(meas, ddof=1)) if len(meas) > 1 else None, "values": meas}
            if a != "off" and meas and null:
                cmp = common.compare(np.asarray(meas, float), np.asarray(null, float))
                row.update(diff=cmp["diff"], z=cmp["z"], p=cmp["p"], verdict=cmp["verdict"], null_sd_zero=cmp.get("null_sd_zero"))
            bench_cmp.append(row)
    bench_df = pd.DataFrame(bench_rows); bench_cmp_df = pd.DataFrame(bench_cmp)
    # ---- runaway record
    run_rows = [{"arm": d["arm"], "run": d["run"], "status": d["status"], "device": d["device"], "frames_done": d["frames_done"], "rest_spikes_per_step": d["rest_spikes_per_step"]["mean"],
                 "rollout_spikes_per_step": d["rollout_spikes_per_step"]["mean"], "rollout_max_frame": d["rollout_spikes_per_step"]["max_frame"],
                 "slow": d.get("slow"), "wall_s": d.get("wall_s"), "bench_exit": d.get("bench_exit_code")} for d in runs]
    run_df = pd.DataFrame(run_rows)
    off_roll = [d["rollout_spikes_per_step"]["mean"] for d in by_arm["off"] if d["rollout_spikes_per_step"]["mean"] is not None]
    run_df["rollout_ratio_vs_off"] = run_df.rollout_spikes_per_step / (np.mean(off_roll) if off_roll else np.nan)
    run_df["runaway_10x_off"] = run_df.rollout_ratio_vs_off > 10
    # ---- print
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print("\n== runs"); print(run_df.drop(columns=["slow"]).round(3).to_string(index=False))
    print("\n== behaviour (per-run value = mean over the run's flies; compare vs off over runs)")
    print(beh_df.drop(columns=["values"]).round(4).to_string(index=False))
    for by in ("type", "module"):
        if by in leave_zero and len(leave_zero[by]):
            lz = leave_zero[by]
            print(f"\n== {by}s leaving 0 Hz (silent in every off run, not in the arm): " + ", ".join(f"{a} {int(n)}" for a, n in lz.groupby("arm").size().items()) + f" (first {args.top} per arm)")
            print(lz.sort_values(["arm", "stim_mean"]).groupby("arm").head(args.top).round(3).to_string(index=False))
        else:
            print(f"\n== {by}s leaving 0 Hz: none")
        if by in movers:
            print(f"\n== {by}: top {args.top} |rate_mean| movers per arm"); print(movers[by].drop(columns=["abs_diff"]).round(3).to_string(index=False))
    if len(bench_df):
        print("\n== benchmark runs"); print(bench_df.drop(columns=["fails"]).round(2).to_string(index=False))
        print("\n== benchmark checks per arm"); print(bench_cmp_df.drop(columns=["values"]).round(3).to_string(index=False))
    prov = dict(runs[0]["provenance"]); prov["analysis"] = {"flyverse_commit": common.git_state(), "generator": "scripts/probe_monoamines.py analyse " + " ".join(sys.argv[2:]),
                                                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "note": "the runs' provenance is the recording jobs'"}
    out = common.Result.new("health", prov)
    out.add_table("runs", run_df); out.add_table("behaviour", beh_df)
    for k, t in health_tabs.items():
        out.add_table("health_" + k, t)
    for by, t in leave_zero.items():
        out.add_table(f"leave_zero_{by}", t)
    for by, t in movers.items():
        out.add_table(f"movers_{by}", t.drop(columns=["abs_diff"]))
    out.add_table("bench_runs", bench_df); out.add_table("bench_checks", bench_cmp_df)
    out.replicates = {"n": {a: len(v) for a, v in by_arm.items()}, "unit": "runs", "runs": [{"arm": d["arm"], "run": d["run"], "file": d["_path"], "device": d["device"], "status": d["status"]} for d in runs], "null": "off"}
    out.summary = {"arms": {a: ARMS[a] for a in by_arm}, "n_runs": {a: len(v) for a, v in by_arm.items()}, "aborted": [(d["arm"], d["run"], d["status"]) for d in runs if d["status"] != "ok"],
                   "runaway_10x_off": run_df[run_df.runaway_10x_off][["arm", "run"]].to_dict("records"), "health_stats": list(HEALTH_STATS), "bench_sections": BENCH_SECTIONS}
    out.files = {"inputs_md5": {k: v for k, v in files.items() if v}, "generator": "scripts/probe_monoamines.py analyse"}
    path = Path(args.json); out.save(path)
    print(f"\nwrote {path}; problems: {out.check() or 'none'}")


# ---------------------------------------------------------------------------------------------- taste (CPU)
MONO_NTS = ("dopamine", "octopamine", "serotonin")
TASTE_NAMED = ["MN9", "GNG175", "GNG141", "GNG038", "GNG042", "GNG101", "GNG550", "GNG002", "GNG572", "GNG067", "GNG644", "GNG540", "GNG137", "MNx05"]


def cmd_taste(args):
    """scripts/benchmark.py sec_taste (sweet labellar-bristle / taste-peg GRNs at 100 Hz, 600 ms, Brain seed 0) replayed
    under one arm with the realised monoamine tone g_slow (mV) read off the Brain per postsynaptic type, the per-type
    rate / tone time course (10 ms frames), and the input current onto MN9 attributed per presynaptic type
    (A x rate: decompose's dynamic quantity, mV/s per MN9 cell, with A = common.effective_weights). Deterministic at a
    fixed seed on the CPU; the H200 benchmark runs are another device (a replication across devices, not a bit check).
    With --null <off taste JSON> the per-type tables are joined and the largest movers printed."""
    import pandas as pd
    from flyverse import brain, connectome
    from flyverse.interp import common
    arm = ARMS[args.arm]
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    n = c.neurons
    lp = brain.LIFParams(**arm["lif"])
    taste = pd.read_csv(ROOT / "flyverse" / "data" / "taste_grns.csv")
    sweet = c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
    sweet = sweet[np.isin(n.subclass.to_numpy()[sweet], ["labellar bristle", "taste peg"])]
    t0 = time.time()
    b = brain.Brain(c, lp, device=args.device, seed=args.seed)
    b.record_activity = True                                          # spike_counts accumulate only when asked (record's pattern)
    types = n.type.fillna("").to_numpy().astype(str)
    uniq, code = np.unique(types, return_inverse=True); ncell = np.bincount(code, minlength=len(uniq)).astype(float)
    nt = n.nt.fillna("unknown").to_numpy().astype(str); from flyverse import regions; mod = regions.labels(c)
    print(f"taste arm {args.arm}: {len(sweet)} sweet GRNs at 100 Hz for {args.ms:.0f} ms; device {b.device}; slow "
          f"{'off' if not b._slow_active else f'{b.slow.mode} {b.slow.gain}'}; build {time.time() - t0:.0f} s", flush=True)
    b.set_poisson(sweet, 100.0)
    frames = int(round(args.ms / 10.0)); steps = int(round(10.0 / lp.dt))
    rate_t = np.zeros((frames, len(uniq)), np.float32); tone_t = np.zeros((frames, len(uniq)), np.float32); sps = np.zeros(frames)
    prev = float(b.spike_counts.sum()); t1 = time.time()
    for k in range(frames):
        b.step(steps)
        cur = float(b.spike_counts.sum()); sps[k] = (cur - prev) / steps; prev = cur
        rate_t[k] = np.bincount(code, weights=b.rate_np(), minlength=len(uniq)) / ncell
        if b._slow_active:
            tone_t[k] = np.bincount(code, weights=b.g_slow[0].detach().cpu().numpy(), minlength=len(uniq)) / ncell
    rt = b.rate_np().astype(np.float64)
    g_end = b.g_slow[0].detach().cpu().numpy().astype(np.float64) if b._slow_active else np.zeros(c.n)
    k0 = frames // 2
    per_type = pd.DataFrame({"type": uniq, "cells": ncell.astype(int), "module": pd.Series(mod).groupby(code).first().reindex(range(len(uniq))).to_numpy(),
                             "nt": pd.Series(nt).groupby(code).first().reindex(range(len(uniq))).to_numpy(),
                             "rate_end_hz": np.bincount(code, weights=rt, minlength=len(uniq)) / ncell, "rate_2nd_half_hz": rate_t[k0:].mean(axis=0),
                             "tone_end_mv": np.bincount(code, weights=g_end, minlength=len(uniq)) / ncell, "tone_2nd_half_mv": tone_t[k0:].mean(axis=0),
                             "tone_end_min_mv": pd.Series(g_end).groupby(code).min().reindex(range(len(uniq))).to_numpy(),
                             "tone_end_max_mv": pd.Series(g_end).groupby(code).max().reindex(range(len(uniq))).to_numpy()})
    per_type["abs_tone"] = per_type.tone_2nd_half_mv.abs()
    mn9 = c.select(type="MN9"); gng175 = c.select(type="GNG175")
    # MN9 input by presynaptic type: A[MN9 cells] x rate (mV/s per MN9 cell), split E / I
    ew = common.effective_weights(c, lp); rows = ew.A[mn9].tocsr()
    contrib = rows.multiply(rt[None, :]).tocsr()
    coo = contrib.tocoo(); pre_code = code[coo.col]
    tot = np.bincount(pre_code, weights=coo.data, minlength=len(uniq)) / max(len(mn9), 1)
    pos = np.bincount(pre_code, weights=np.where(coo.data > 0, coo.data, 0), minlength=len(uniq)) / max(len(mn9), 1)
    neg = np.bincount(pre_code, weights=np.where(coo.data < 0, coo.data, 0), minlength=len(uniq)) / max(len(mn9), 1)
    mn9_in = pd.DataFrame({"pre_type": uniq, "mv_per_s": tot, "E": pos, "I": neg, "pre_rate_hz": per_type.rate_end_hz.to_numpy(), "pre_tone_mv": per_type.tone_end_mv.to_numpy()})
    mn9_in = mn9_in[mn9_in.mv_per_s != 0].copy(); mn9_in["abs"] = mn9_in.mv_per_s.abs(); mn9_in = mn9_in.sort_values("abs", ascending=False)
    res = {"arm": args.arm, "device": str(b.device), "ms": args.ms, "seed": args.seed, "n_sweet_grn": int(len(sweet)),
           "MN9_hz": float(rt[mn9].mean()), "GNG175_hz": float(rt[gng175].mean()), "frac_active": float((rt > 1).mean()),
           "spikes_per_step_2nd_half": float(sps[k0:].mean()), "spikes_per_step_max_frame": float(sps.max()),
           "MN9_input_mv_per_s": {"total": float(tot.sum()), "E": float(pos.sum()), "I": float(neg.sum())},
           "mono_types_active": int(((per_type.nt.isin(MONO_NTS)) & (per_type.rate_end_hz > 1)).sum()),
           "tone_mv_over_types": {"min": float(per_type.tone_end_mv.min()), "max": float(per_type.tone_end_mv.max()), "n_types_below_-3.5": int((per_type.tone_end_mv < -3.5).sum()),
                                  "n_types_above_3.5": int((per_type.tone_end_mv > 3.5).sum()), "n_types_nonzero": int((per_type.tone_end_mv != 0).sum())},
           "wall_s": round(time.time() - t1, 1)}
    print(f"MN9 {res['MN9_hz']:.2f} Hz  GNG175 {res['GNG175_hz']:.1f}  frac {res['frac_active']:.4f}  spikes/step {res['spikes_per_step_2nd_half']:.2f} (max frame {res['spikes_per_step_max_frame']:.1f}); "
          f"MN9 input {res['MN9_input_mv_per_s']['total']:.0f} mV/s (E {res['MN9_input_mv_per_s']['E']:.0f}, I {res['MN9_input_mv_per_s']['I']:.0f}); tone over types {res['tone_mv_over_types']}; {res['wall_s']:.0f} s", flush=True)
    pd.set_option("display.width", 250)
    named = per_type[per_type.type.isin(TASTE_NAMED)].set_index("type").reindex(TASTE_NAMED).reset_index()
    print("== named SEZ types"); print(named.drop(columns=["abs_tone"]).round(3).to_string(index=False))
    mono_on = per_type[per_type.nt.isin(MONO_NTS) & (per_type.rate_end_hz > 1)].sort_values("rate_end_hz", ascending=False)
    print(f"== monoaminergic types above 1 Hz ({len(mono_on)})"); print(mono_on.drop(columns=["abs_tone"]).head(25).round(2).to_string(index=False))
    tones = per_type.sort_values("abs_tone", ascending=False).head(args.top)
    print(f"== largest |tone| per type ({args.top})"); print(tones.drop(columns=["abs_tone"]).round(2).to_string(index=False))
    print("== MN9 input by presynaptic type (mV/s per MN9 cell)"); print(mn9_in.drop(columns=["abs"]).head(args.top).round(2).to_string(index=False))
    stimulus = {"protocol": "benchmark.sec_taste", "params": {"grn_hz": 100.0, "ms": args.ms, "sweet_subclasses": ["labellar bristle", "taste peg"], "arm": args.arm, "lif_overrides": common.to_jsonable(arm["lif"])},
                "control": {"arm": "off"}}
    prov = common.provenance(c, lp, None, device=str(b.device), seeds=[args.seed], batch=1, backend=dict(cuda_graphs=False, cuda_kernels=False, event_driven=False),
                             stimulus=stimulus, cache_dir=args.cache_dir)
    out = common.Result.new("health", prov)
    out.add_table("per_type", per_type.drop(columns=["abs_tone"])); out.add_table("mn9_input_by_pre_type", mn9_in.drop(columns=["abs"]))
    out.add_table("rate_time_course_named", pd.DataFrame(rate_t[:, np.isin(uniq, TASTE_NAMED)], columns=[t for t in uniq if t in TASTE_NAMED]).assign(t_ms=(np.arange(frames) + 1) * 10.0))
    out.add_table("tone_time_course_named", pd.DataFrame(tone_t[:, np.isin(uniq, TASTE_NAMED)], columns=[t for t in uniq if t in TASTE_NAMED]).assign(t_ms=(np.arange(frames) + 1) * 10.0))
    out.summary = dict(res, note="one deterministic CPU replay of the benchmark's taste section per arm (seed-locked: the H200 benchmark's own 5 runs x 3 seeds are the replicates of the measured value)")
    out.replicates = {"n": 1, "unit": "runs", "runs": [{"arm": args.arm, "seed": args.seed, "device": str(b.device)}], "null": "off"}
    out.files = {"generator": "scripts/probe_monoamines.py taste " + " ".join(sys.argv[2:])}
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); out.save(path)
    if args.null:
        null = common.Result.load(Path(args.null)); nt_ = null.table("per_type").set_index("type"); me = per_type.set_index("type")
        j = me.join(nt_[["rate_end_hz", "tone_end_mv"]], rsuffix="_off"); j["d_rate"] = j.rate_end_hz - j.rate_end_hz_off; j["abs_d"] = j.d_rate.abs()
        print(f"== rate movers vs {args.null} (Hz, end of the 600 ms)"); print(j.sort_values("abs_d", ascending=False).head(args.top)[["cells", "module", "nt", "rate_end_hz_off", "rate_end_hz", "d_rate", "tone_end_mv"]].round(2).to_string())
        mi = null.table("mn9_input_by_pre_type").set_index("pre_type"); k = mn9_in.set_index("pre_type").join(mi[["mv_per_s", "pre_rate_hz"]], rsuffix="_off", how="outer").fillna(0)
        k["d"] = k.mv_per_s - k.mv_per_s_off; k["abs_d"] = k.d.abs()
        print("== MN9 input movers by presynaptic type (mV/s per MN9 cell)"); print(k.sort_values("abs_d", ascending=False).head(args.top)[["mv_per_s_off", "mv_per_s", "d", "pre_rate_hz_off", "pre_rate_hz", "pre_tone_mv"]].round(1).to_string())
    print(f"wrote {path}; problems: {out.check() or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("arms")
    b = sub.add_parser("batch"); b.add_argument("--name", default="mono"); b.add_argument("--minutes", type=int, default=45); b.add_argument("--runs", type=int, default=5)
    b.add_argument("--batch", type=int, default=8); b.add_argument("--seconds", type=float, default=30.0)
    r = sub.add_parser("record")
    r.add_argument("--arm", choices=sorted(ARMS), required=True); r.add_argument("--run", type=int, default=0)
    r.add_argument("--block", default=None, help="the cluster_run --arm-block token (fam_r<run>); no effect on the run")
    r.add_argument("--batch", type=int, default=8); r.add_argument("--seconds", type=float, default=30.0); r.add_argument("--skip", type=float, default=2.0)
    r.add_argument("--every", type=int, default=10, help="capture the health target every N frames (10 = 100 ms)")
    r.add_argument("--no-fence", action="store_true"); r.add_argument("--no-bench", action="store_true")
    r.add_argument("--abort-rest", type=float, default=10 * REST_BOUND, help="abort the arm when the 500 ms rest phase exceeds this many spikes/step (10 x the rest bound)")
    r.add_argument("--abort-rollout", type=float, default=500.0, help="abort the arm when a rollout frame exceeds this many spikes/step (~6.5 Hz mean over the 77k spiking cells)")
    r.add_argument("--device", default=None); r.add_argument("--cuda-sparse", default="torch"); r.add_argument("--cache-dir", default=None); r.add_argument("--allow-cpu", action="store_true")
    r.add_argument("--out", required=True)
    a = sub.add_parser("analyse"); a.add_argument("--runs", default="out/monoamines/runs"); a.add_argument("--json", default="out/monoamines/analysis.json"); a.add_argument("--top", type=int, default=15)
    t = sub.add_parser("taste"); t.add_argument("--arm", choices=sorted(ARMS), required=True); t.add_argument("--ms", type=float, default=600.0); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu"); t.add_argument("--cache-dir", default=None); t.add_argument("--top", type=int, default=20); t.add_argument("--null", default=None, help="the off arm's taste JSON to diff against")
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "arms":
        cmd_arms(args)
    elif args.cmd == "taste":
        cmd_taste(args)
    elif args.cmd == "batch":
        cmd_batch(args)
    elif args.cmd == "record":
        sys.exit(cmd_record(args))
    else:
        cmd_analyse(args)


if __name__ == "__main__":
    main()
