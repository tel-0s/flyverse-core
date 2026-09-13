"""interp_apply_turning -- localize the plain fly's missing turning with the interpretability toolkit (apply:turning).

The plain model walks straight (docs/NOTES.md 'why the plain fly walks straight'; probe_walk_straightness.py: yaw SD
1.6-2.8 deg/s in every weight configuration). body.py's only turning inputs are DNa02 R-L (k_turn) and the leg-MN L-R
asymmetry (k_leg_turn). This script applies three validated tools to that readout and writes one audit's worth of raw
data (docs/audits/deficit_turning.md):

  (a) decompose  -- DNa02 L / R and the leg-MN L / R groups during a plain-fly room rollout (BatchSim, 16 flies, 60 s,
                    default weights): which presynaptic types drive them, the L-R difference per type, and whether any
                    type carries an asymmetry that the sum cancels (a per-fly time-series variance decomposition).
  (b) atlas      -- which populations, when stimulated, produce a DNa02 or leg asymmetry at all: DNa02's presynaptic
                    types, every LAL type, the CX output / columnar types (by side), plus the lateralised DN movers of
                    docs/audits/interp_atlas.md, each against the atlas' in-batch null rows, 3 independent runs.
  (c) paths      -- from each mover back to the CX and to the sensory populations (visual projection neurons, JO,
                    ORNs, VNC sensory), k <= 3, silent links flagged, never_firing judged on the room rollout itself.

GPU (cluster; one batch, `plan` writes the exact lines):
    python scripts/interp_apply_turning.py record    --seed 0 --out out/turn/plain_r0        # (a) one run = 16 flies x 60 s
    python scripts/interp_apply_turning.py atlas-run --seed 0 --out out/turn/atlas_r0        # (b) one run
CPU (desktop):
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_turning.py plan --out out/interp/apply_turning
    PYTHONIOENCODING=utf-8 python scripts/interp_apply_turning.py analyse --recordings "out/turn/plain_r*" \
        --atlas-runs "out/turn/atlas_r*" --json out/interp/apply_turning/turning.json

What `record` stores per run (out/turn/plain_r<seed>.npz + .json): per captured frame (every `--every` frames) and per
fly the rate-weighted synaptic input I_g(t) = mean over post cells of sum_j A[post, j] r_j(t) (mV/s per post cell;
`decompose`'s 'current', A = common.effective_weights = Brain._W_cpu) of each of the four targets split by presynaptic
(type, soma side) group -- kept in full for the union of the top-K groups of each L / R pair and as E / I / net totals
for all groups -- the per-fly mean rate of every presynaptic cell over the rollout (which, with A, gives the exact
per-fly mean input of every group), the per-cell max rate (for `never_firing`), the DNa02 / leg / watch-type rates,
the body's yaw command, heading, speed, wind angle and airborne flag per frame. The per-fly means are also written as a
`common.Recording` whose 'frames' are flies (`plain_r<seed>_flies.npz`) so `decompose` runs on them unchanged: its
window mean over frames is then the mean over the 16 flies of one run, and 3 runs = 3 replicates (BATCH_SIM.md: rows
of one batch are one draw layout, not replicates). The replicate unit everywhere is the run; per-fly numbers are quoted
with their scatter over 48 flies and never called a result on their own.

The toolkit reads the model: nothing under flyverse/ is edited or overridden here (the BatchSim runs the shipped
default weights; no program, no gains, no hold table).
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flyverse import connectome as cn                                  # noqa: E402
from flyverse.interp import common                                     # noqa: E402

TOOL = "apply_turning"
OUT_DIR = Path(ROOT) / "out" / "interp" / "apply_turning"
TARGETS = {"DNa02_L": "type=DNa02&somaSide=L", "DNa02_R": "type=DNa02&somaSide=R",
           "leg_L": "superclass=vnc_motor&subclass:fl|ml|hl&somaSide=L", "leg_R": "superclass=vnc_motor&subclass:fl|ml|hl&somaSide=R"}
PAIRS = (("DNa02_L", "DNa02_R"), ("leg_L", "leg_R"))
# types whose by-side rates are recorded every frame (the DN movers of interp_atlas.md leg_LR / turn_LR, the CX output
# stage and the wind DNs)
WATCH = ["DNa02", "DNge035", "DNa13", "DNge037", "DNge049", "DNge073", "DNae007", "DNpe024", "DNde003", "PFL3", "PFL2",
         "DNp18", "DNp33", "DNa01", "DNa03", "DNp09", "LLPC1", "LT51", "AOTU015", "PS077"]
CX_SPEC = "~^(PFL[123]|PFR|FC[123]|FS[1-4]|hDelta|vDelta|EPG|PEN_|PEG|Delta7|ExR|ER\\d|FB\\d)"
CX_OUT_SPEC = "~^(PFL[123]|PFR|FC[123]|FS[1-4]|hDelta|vDelta|EPG|PEN_|PEG|Delta7|ExR|ER\\d)"
DN_MOVERS = ["DNge035", "DNa13", "DNge037", "DNge049", "DNge073", "DNa02", "DNae007", "DNpe024", "DNde003"]
SOURCES = {"cx": CX_SPEC, "lal": "~^LAL", "visual_projection": "superclass=visual_projection", "wind_JO": "~^JO",
           "ORN": "~^ORN", "vnc_sensory": "superclass=vnc_sensory"}
SENSORY_SOURCES = ("visual_projection", "wind_JO", "ORN", "vnc_sensory")


def _log(*a, **k):
    print(*a, **k, flush=True)


def _load(args):
    return cn.load(cache_dir=Path(args.cache_dir) if args.cache_dir else cn.CACHE_DIR, verbose=False)


# ------------------------------------------------------------------------------------------------ groups / weights
def target_groups(c) -> dict[str, np.ndarray]:
    return {k: common.resolve(c, v) for k, v in TARGETS.items()}


def input_vectors(c, ew, targets: dict[str, np.ndarray]):
    """(pre_idx, {target: w}) with w[j] = mean over the target's post cells of A[post, pre_idx[j]] (mV per spike per
    post cell): the linear map from presynaptic rates (Hz) to the target's mean synaptic input (mV/s)."""
    A = ew.A.tocsr()
    rows = np.unique(np.concatenate(list(targets.values())))
    pre_idx = np.unique(A[rows].tocoo().col)
    w = {}
    for k, idx in targets.items():
        sub = A[idx][:, pre_idx]
        w[k] = np.asarray(sub.sum(axis=0)).ravel().astype(np.float32) / max(len(idx), 1)
    return pre_idx, w


def group_labels(c, pre_idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(labels, inverse, types) -- presynaptic (type, side) groups 'TYPE/S' in decompose's 'type,side' naming."""
    n = c.neurons
    ty = n.type.fillna("").to_numpy()[pre_idx].astype(str)
    sd = n.somaSide.fillna("?").to_numpy()[pre_idx].astype(str) if "somaSide" in n else np.full(len(pre_idx), "?")
    ty = np.where(ty == "", "(untyped)", ty)
    lab = np.array([f"{t}/{s}" for t, s in zip(ty, sd)])
    keys, inv = np.unique(lab, return_inverse=True)
    return keys, inv, ty


def flies_recording(c, sel, rate_mean_bs, rate_max, meta, sides) -> common.Recording:
    """A common.Recording whose frames are flies: quantities['rate_hz'][b] = fly b's rollout-mean rate of every cell in
    `sel`. `decompose` treats the window mean over frames as the mean over flies; t_ms = 1000 b labels the fly."""
    n = c.neurons
    B = rate_mean_bs.shape[0]
    return common.Recording(np.arange(B, dtype=np.float64) * 1000.0, np.asarray(sel), n.bodyId.to_numpy()[sel],
                            n.type.fillna("").to_numpy()[sel].astype(str), {"rate_hz": rate_mean_bs.astype(np.float32)},
                            {}, dict(meta, sides=sides, frames_are="flies", max_rate_hz_per_cell="in the npz sidecar of the run"))


# ------------------------------------------------------------------------------------------------ record (GPU)
def cmd_record(args) -> int:
    import torch
    from flyverse import BatchSim, world
    if args.device != "cpu":
        assert torch.cuda.is_available(), "CUDA is not available on this node (resubmit the job)"
    t0 = time.time()
    c = _load(args)
    B = int(args.batch)
    env_seeds = [args.seed * 100 + i for i in range(B)]
    start = (-0.15, 0.15, world.make_room(0, "all")[1]["table_top_z"])
    sim = BatchSim(B, seeds=env_seeds, seed=args.seed, c=c, start=start, program="none", fruit_set="all", fence=not args.no_fence,
                   cuda_graphs=args.device != "cpu", cuda_kernels=(args.device != "cpu"), event_driven=(args.device != "cpu"),
                   cuda_sparse="torch", device=args.device)
    fb, brain = sim.fb, sim.fb.brain
    lp, op = brain.p, (sim.optic.p if sim.optic is not None else None)
    dev = str(fb.device)
    rng_head = {}
    for seed, fly, m in zip(sim.seeds, sim.flies, sim.metabolisms):
        fly.heading = float(np.random.default_rng(seed).uniform(-np.pi, np.pi)); rng_head[seed] = fly.heading
        m.energy = args.energy
    _log(f"device {dev} ({torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'cpu'}); torch {torch.__version__}")
    _log(f"BatchSim B={B} program=none fruit=all fence={not args.no_fence} env seeds={env_seeds} brain seed={args.seed} "
         f"receptor_model {lp.receptor_model} type_path_gain {'DEFAULT' if lp.type_path_gain is None else lp.type_path_gain} "
         f"cuda kernels {brain.cuda} event_driven {brain.event_driven} cuda_graphs {fb.cuda_graphs}")
    # the linear map from presynaptic rates to each target's mean synaptic input
    ew = common.effective_weights(c, lp)
    targets = target_groups(c)
    pre_idx, w = input_vectors(c, ew, targets)
    keys, inv, pre_types = group_labels(c, pre_idx)
    G = len(keys)
    tgt_names = list(targets)
    Ind = torch.zeros((len(pre_idx), G), dtype=torch.float32, device=fb.device)
    Ind[torch.arange(len(pre_idx), device=fb.device), torch.as_tensor(inv, device=fb.device)] = 1.0
    W_t = torch.stack([torch.as_tensor(w[k], device=fb.device) for k in tgt_names])            # (4, n_pre)
    pre_t = torch.as_tensor(pre_idx, device=fb.device)
    tgt_all = np.unique(np.concatenate(list(targets.values())))
    sel = np.union1d(tgt_all, pre_idx)
    sel_t = torch.as_tensor(sel, device=fb.device)
    tgt_pos = {k: torch.as_tensor(np.searchsorted(sel, idx), device=fb.device) for k, idx in targets.items()}
    # watch types by side
    n = c.neurons
    watch = {}
    for t in WATCH:
        for s in ("L", "R"):
            idx = common.resolve(c, f"type={t}&somaSide={s}")
            if len(idx):
                watch[f"{t}_{s}"] = torch.as_tensor(idx, device=fb.device)
    watch_names = list(watch)
    _log(f"targets {[(k, len(v)) for k, v in targets.items()]}; presynaptic cells {len(pre_idx)} in {G} (type, side) groups; "
         f"recorded cells {len(sel)}; watch {len(watch_names)} type-sides; ew md5 {ew.md5}")
    n_frames = int(round(args.seconds * 100))
    every, mean_every = int(args.every), int(args.mean_every)
    T = (n_frames + every - 1) // every
    contrib = np.zeros((T, len(tgt_names), B, G), np.float32)
    tot = np.zeros((T, len(tgt_names), B), np.float32); totE = np.zeros_like(tot); totI = np.zeros_like(tot)
    watch_rates = np.zeros((T, B, len(watch_names)), np.float32)
    dna02_drive = np.zeros((T, B, 2), np.float32)
    motor_fields = ("turn_L", "turn_R", "leg_L", "leg_R", "fwd_dn", "back_dn", "wind_ipsi_L", "wind_ipsi_R", "wind_contra_L", "wind_contra_R", "opto_L", "opto_R", "gf", "power")
    motor = {k: np.zeros((T, B), np.float32) for k in motor_fields}
    yaw_cmd = np.zeros((T, B), np.float32); speed_cmd = np.zeros((T, B), np.float32)
    heading = np.zeros((T, B), np.float32); pos = np.zeros((T, B, 3), np.float32); airborne = np.zeros((T, B), bool)
    wind_angle = np.zeros((T, B), np.float32)
    rate_sum = torch.zeros((B, len(sel)), dtype=torch.float64, device=fb.device)
    rate_max = torch.zeros((B, len(sel)), dtype=torch.float32, device=fb.device)
    n_mean = 0
    heading_all = np.zeros((n_frames, B), np.float32)
    wind_dir = np.deg2rad(180.0)
    k_out = 0
    for k in range(n_frames):
        sim.step()
        for i, f in enumerate(sim.flies):
            heading_all[k, i] = f.heading
        if k % mean_every == 0:
            r_sel = brain.rate[:, sel_t]
            rate_sum += r_sel.double(); rate_max = torch.maximum(rate_max, r_sel); n_mean += 1
        if k % every == 0:
            rate = brain.rate
            r_pre = rate[:, pre_t]                                                               # (B, n_pre)
            cg = torch.einsum("gj,bj,jh->gbh", W_t, r_pre, Ind)                                  # (4, B, G) mV/s
            cgn = cg.cpu().numpy()
            contrib[k_out] = cgn
            tot[k_out] = cgn.sum(-1); totE[k_out] = np.clip(cgn, 0, None).sum(-1); totI[k_out] = np.clip(cgn, None, 0).sum(-1)
            if watch_names:
                watch_rates[k_out] = torch.stack([rate[:, v].mean(1) for v in watch.values()], 1).cpu().numpy()
            dna02_drive[k_out] = torch.stack([brain.drive[:, sel_t[tgt_pos["DNa02_L"]]].mean(1), brain.drive[:, sel_t[tgt_pos["DNa02_R"]]].mean(1)], 1).cpu().numpy()
            m = sim.motor
            for name in motor_fields:
                motor[name][k_out] = np.asarray(getattr(m, name, 0.0), np.float32)
            for i, f in enumerate(sim.flies):
                cmd = sim.commands[i]
                yaw_cmd[k_out, i] = cmd["yaw"]; speed_cmd[k_out, i] = cmd["speed"]
                heading[k_out, i] = f.heading; pos[k_out, i] = f.pos; airborne[k_out, i] = bool(f.airborne)
                wind_angle[k_out, i] = float((wind_dir - f.heading + np.pi) % (2 * np.pi) - np.pi)
            k_out += 1
        if k % 1000 == 999:
            _log(f"  t {(k + 1) / 100:5.0f} s  airborne {airborne[k_out - 1].mean():.2f}  DNa02 L-R {float((motor['turn_L'][k_out - 1] - motor['turn_R'][k_out - 1]).mean()):+.2f} Hz  "
                 f"leg L-R {float((motor['leg_L'][k_out - 1] - motor['leg_R'][k_out - 1]).mean()):+.2f} Hz  wall {time.time() - t0:.0f} s")
    rate_mean = (rate_sum / max(n_mean, 1)).float().cpu().numpy()                                # (B, n_sel)
    rate_max_np = rate_max.cpu().numpy()
    # per-fly per-group mean input (exact, from the per-fly mean rates) and the top-K union per pair
    pre_pos = np.searchsorted(sel, pre_idx)
    Ind_np = sp.csr_matrix((np.ones(len(pre_idx), np.float32), (np.arange(len(pre_idx)), inv)), shape=(len(pre_idx), G))
    group_mean = np.stack([np.asarray((rate_mean[:, pre_pos] * w[k][None]) @ Ind_np) for k in tgt_names])           # (4, B, G)
    keep = {}
    series = {}
    for a, b in PAIRS:
        ia, ib = tgt_names.index(a), tgt_names.index(b)
        score = np.abs(group_mean[ia]).mean(0) + np.abs(group_mean[ib]).mean(0)
        sel_g = np.argsort(-score)[: int(args.top_groups)]
        keep[f"{a}|{b}"] = sel_g
        series[a] = contrib[:, ia][:, :, sel_g].astype(np.float16)
        series[b] = contrib[:, ib][:, :, sel_g].astype(np.float16)
    yaw_rate = np.diff(np.unwrap(heading_all.astype(np.float64), axis=0), axis=0) * 100.0          # rad/s per frame
    yaw_rate = np.concatenate([yaw_rate[:1], yaw_rate], 0)[::every][:T].astype(np.float32)
    stim = {"protocol": "room_plain", "params": {"batch": B, "seconds": args.seconds, "program": "none", "fruit_set": "all", "fence": not args.no_fence,
                                                 "start": list(map(float, start)), "energy": args.energy, "wind": "0.3 m/s towards 180 deg (BatchSim default)",
                                                 "initial_heading": "uniform(-pi, pi) per environment seed (probe_compass_room's rule)",
                                                 "capture_every_frames": every, "mean_every_frames": mean_every, "top_groups_per_pair": int(args.top_groups)},
            "control": "none: the question is the L-R structure of the input; the replicate unit is the run (3 seeds), the null of a per-type L-R offset is its scatter over runs and flies"}
    prov = common.provenance(c, lp, op, fb=fb, device=args.device, seeds=[args.seed], env_seeds=env_seeds, batch=B,
                             backend={"cuda_kernels": bool(brain.cuda), "event_driven": bool(brain.event_driven), "cuda_graphs": bool(fb.cuda_graphs), "cuda_sparse": "torch"},
                             stimulus=stim, cache_dir=args.cache_dir)
    meta = {"protocol": "room_plain", "seed": int(args.seed), "env_seeds": env_seeds, "initial_heading": rng_head, "targets": {k: v for k, v in TARGETS.items()},
            "target_n_cells": {k: int(len(v)) for k, v in targets.items()}, "n_presynaptic": int(len(pre_idx)), "n_groups": int(G), "n_recorded": int(len(sel)),
            "ew_md5": ew.md5, "shaped_md5": ew.shaped_md5, "frames": n_frames, "captured_frames": int(T), "every": every, "mean_every": mean_every,
            "watch": watch_names, "motor_fields": list(motor_fields), "units": {"contrib": "mV/s per post cell (A[mV per spike] x rate[Hz])", "rates": "Hz", "yaw": "rad/s"},
            "provenance": prov, "generator": "scripts/interp_apply_turning.py record " + " ".join(sys.argv[2:]), "wall_s": round(time.time() - t0, 1),
            "device": prov["execution"]["device"]}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"t_ms": (np.arange(T) * every * 10.0).astype(np.float64), "group_keys": keys.astype(str), "target_names": np.array(tgt_names),
              "pre_idx": pre_idx, "pre_group": inv.astype(np.int32), "sel_idx": sel, "rate_mean_flies": rate_mean, "rate_max_flies": rate_max_np,
              "group_mean_flies": group_mean, "tot": tot, "totE": totE, "totI": totI, "watch_rates": watch_rates, "dna02_drive": dna02_drive,
              "yaw_cmd": yaw_cmd, "speed_cmd": speed_cmd, "heading": heading, "pos": pos, "airborne": airborne, "wind_angle": wind_angle, "yaw_rate": yaw_rate}
    for k in tgt_names:
        arrays[f"w__{k}"] = w[k]
    for pair, sel_g in keep.items():
        arrays[f"keep__{pair}"] = sel_g.astype(np.int32)
    for k, v in series.items():
        arrays[f"series__{k}"] = v
    for k, v in motor.items():
        arrays[f"m__{k}"] = v
    np.savez(out.with_suffix(".npz"), **arrays)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(meta), f, indent=1)
    # the per-fly recording for `decompose` and the max-rate recording for `paths --recording`
    sides = n.somaSide.fillna("?").to_numpy()[sel].astype(str).tolist()
    R = flies_recording(c, sel, rate_mean, rate_max_np, dict(meta, file=str(out.with_suffix(".npz"))), sides)
    R.meta.pop("initial_heading", None)
    R.save(out.with_name(out.name + "_flies"))
    Rm = common.Recording(np.zeros(1), np.asarray(sel), n.bodyId.to_numpy()[sel], n.type.fillna("").to_numpy()[sel].astype(str),
                          {"rate_hz": rate_max_np.max(0)[None, :].astype(np.float32)}, {},
                          {"protocol": "room_plain", "seed": int(args.seed), "what": "one frame = per-cell max rate over all frames and flies of the run (for paths never_firing)",
                           "provenance": prov, "sides": sides})
    Rm.save(out.with_name(out.name + "_max"))
    lr = float((motor["turn_L"] - motor["turn_R"]).mean()); leg = float((motor["leg_L"] - motor["leg_R"]).mean())
    _log(f"device {prov['execution']['device']}  frames {n_frames} captured {T}  DNa02 L-R mean {lr:+.3f} Hz (fly sd {np.std((motor['turn_L'] - motor['turn_R']).mean(0)):.3f})  "
         f"leg L-R mean {leg:+.3f} Hz  yaw-rate sd {np.degrees(np.std(yaw_rate[~airborne])):.2f} deg/s  airborne frac {airborne.mean():.3f}  wall {meta['wall_s']} s")
    _log(f"wrote {out.with_suffix('.npz')} (+ _flies, _max)")
    return 0


# ------------------------------------------------------------------------------------------------ atlas run (GPU)
def atlas_population_specs(c, min_entries: int = 4) -> list[str]:
    """The stimulation list of (b): DNa02's presynaptic types with >= `min_entries` stored entries onto the two DNa02
    cells, every LAL type, the CX output / columnar types, and the DN movers of interp_atlas.md -- split by type and
    soma side by the atlas tool itself."""
    ew = common.effective_weights(c)
    A = ew.A.tocsr()
    dna = common.resolve(c, "DNa02")
    coo = A[dna].tocoo()
    ty = c.neurons.type.fillna("").to_numpy()
    cnt = pd.Series(ty[coo.col]).value_counts()
    pre_types = sorted(t for t, v in cnt.items() if v >= min_entries and t)
    return ["type:" + "|".join(pre_types), "~^LAL", CX_OUT_SPEC, "type:" + "|".join(DN_MOVERS)]


def cmd_atlas_run(args) -> int:
    import torch
    from flyverse.interp import atlas as atlas_mod
    if args.device != "cpu":
        assert torch.cuda.is_available(), "CUDA is not available on this node (resubmit the job)"
    c = _load(args)
    lif, optic = common.params_from_args(args)
    specs = atlas_population_specs(c, args.min_entries)
    pops = atlas_mod.make_populations(c, specs, by_side=True, split="type", hz=args.hz, ms=args.ms, min_cells=1)
    if args.limit:
        pops = pops[: int(args.limit)]                                                            # smoke tests only
    pattern = "^(" + "|".join(WATCH) + ")$"
    _log(f"atlas-run: {len(pops)} populations from {len(specs)} specs, batch {args.batch} ({args.n_null} null rows), {args.hz} Hz for {args.ms} ms after {args.settle_ms} ms, seed {args.seed}")
    run = atlas_mod.run_once(c, pops, hz=args.hz, ms=args.ms, settle_ms=args.settle_ms, batch=args.batch, readouts=("motor",), pattern=pattern,
                             by_side=True, n_null=args.n_null, params=lif, optic_params=optic, device=args.device, context=None, seed=args.seed,
                             per_body=False, quiet=args.quiet)
    out = Path(args.out)
    run.meta["file"] = str(out.with_suffix(".npz"))
    run.meta["population_specs"] = specs
    run.save(out)
    _log(f"device {run.meta['provenance']['execution']['device']}   {run.meta['n_populations']} populations x {len(run.readouts)} readouts   "
         f"{len(run.null_ids)} null rows   {run.meta['wall_s']} s   -> {out.with_suffix('.npz')}")
    df = run.frame("mean")
    cols = [x for x in ("turn_LR", "leg_LR", "turn_L", "turn_R") if x in df.columns]
    if cols:
        common.print_table(df[cols].reindex(df[cols].abs().max(axis=1).sort_values(ascending=False).index).head(20).reset_index().rename(columns={"index": "population"}), max_rows=20)
    return 0


# ------------------------------------------------------------------------------------------------ plan
def cmd_plan(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pre = "python -c 'import torch; assert torch.cuda.is_available()' && "
    cmds = []
    for s in range(args.replicates):
        cmds.append(f"mkdir -p out/turn && {pre}python scripts/interp_apply_turning.py record --seed {s} --seconds {args.seconds} --out out/turn/plain_r{s} > out/turn/plain_r{s}.txt 2>&1; cat out/turn/plain_r{s}.txt")
    for s in range(args.replicates):
        cmds.append(f"mkdir -p out/turn && {pre}python scripts/interp_apply_turning.py atlas-run --seed {s} --out out/turn/atlas_r{s} > out/turn/atlas_r{s}.txt 2>&1; cat out/turn/atlas_r{s}.txt")
    line = f"python scripts/cluster_run.py --name turn --minutes {args.minutes} " + " ".join('"' + c_.replace('"', '\\"') + '"' for c_ in cmds) + " --fetch out/turn/ 2>&1 | tee out/turn_cluster.log"
    (out / "batch.sh").write_text("#!/bin/bash\n# one batch: 3 plain-fly recordings (a) + 3 atlas runs (b); written by scripts/interp_apply_turning.py plan\ncd \"$(dirname \"$0\")/../../..\"\n" + line + "\n", encoding="utf-8")
    print(line)
    print(f"wrote {out / 'batch.sh'}")
    return 0


# ------------------------------------------------------------------------------------------------ analyse (CPU)
class Run:
    def __init__(self, prefix: str):
        self.prefix = Path(prefix)
        z = np.load(self.prefix.with_suffix(".npz"), allow_pickle=False)
        self.z = {k: z[k] for k in z.files}
        with open(self.prefix.with_suffix(".json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.keys = self.z["group_keys"].astype(str)
        self.targets = self.z["target_names"].astype(str).tolist()
        self.B = self.z["tot"].shape[2]
        self.seed = int(self.meta["seed"])
        self.flies = common.Recording.load(self.prefix.with_name(self.prefix.name + "_flies"))

    def series(self, target: str, pair: str):
        return self.z[f"series__{target}"].astype(np.float32), self.z[f"keep__{pair}"]

    def motor(self, name: str):
        return self.z[f"m__{name}"]


def _pair_of(target: str) -> str:
    for a, b in PAIRS:
        if target in (a, b):
            return f"{a}|{b}"
    raise KeyError(target)


def lr_tables(runs: list[Run], c) -> dict[str, pd.DataFrame]:
    """Per pair: the per-(pre type) and per-(pre type, side) mean input to the L and the R target over flies and runs,
    the L-R difference with its scatter over 48 flies and over the 3 run means (the replicate unit), and the static
    weight (mV per volley) alongside."""
    out = {}
    for a, b in PAIRS:
        ia, ib = runs[0].targets.index(a), runs[0].targets.index(b)
        keys = runs[0].keys
        assert all(np.array_equal(r.keys, keys) for r in runs), "runs disagree on the group keys"
        gm = np.concatenate([r.z["group_mean_flies"] for r in runs], axis=1)                     # (4, n_flies, G)
        run_of = np.concatenate([[k] * r.B for k, r in enumerate(runs)])
        types = np.array([k.split("/")[0] for k in keys]); sides = np.array([k.split("/")[1] for k in keys])
        wa = runs[0].z[f"w__{a}"]; wb = runs[0].z[f"w__{b}"]
        inv = runs[0].z["pre_group"]
        G = len(keys)
        w_group_a = np.bincount(inv, weights=wa, minlength=G); w_group_b = np.bincount(inv, weights=wb, minlength=G)
        n_cells_g = np.bincount(inv, minlength=G)
        # the presynaptic cells' own room activity: per-cell mean rate over flies and runs, per-cell max over everything
        pre_pos = np.searchsorted(runs[0].z["sel_idx"], runs[0].z["pre_idx"])
        r_mean_cell = np.concatenate([r.z["rate_mean_flies"][:, pre_pos] for r in runs], 0).mean(0)                # (n_pre,)
        r_max_cell = np.max(np.stack([r.z["rate_max_flies"][:, pre_pos].max(0) for r in runs]), 0)                  # (n_pre,)
        mean_g_sum = np.bincount(inv, weights=r_mean_cell, minlength=G)
        firing_g = np.bincount(inv, weights=(r_max_cell >= common.NEVER_FIRING_HZ).astype(float), minlength=G)
        max_g = np.zeros(G); np.maximum.at(max_g, inv, r_max_cell)
        for level, lab in (("type_side", keys), ("type", types)):
            uk, uinv = np.unique(lab, return_inverse=True)
            M = sp.csr_matrix((np.ones(G), (np.arange(G), uinv)), shape=(G, len(uk)))
            IL = np.asarray(gm[ia] @ M); IR = np.asarray(gm[ib] @ M)                              # (n_flies, n_groups)
            D = IL - IR
            run_means = np.stack([D[run_of == k].mean(0) for k in range(len(runs))])              # (n_runs, n_groups)
            n_lab = np.asarray(n_cells_g @ M).ravel()
            max_lab = np.zeros(len(uk)); np.maximum.at(max_lab, uinv, max_g)
            with np.errstate(divide="ignore", invalid="ignore"):
                z_runs = (run_means.mean(0) / (run_means.std(0, ddof=1) / np.sqrt(len(runs)))) if len(runs) > 1 else np.full(len(uk), np.nan)
            rows = pd.DataFrame({"pre_group": uk, "n_pre_cells": n_lab.astype(int),
                                 f"w_{a}_mv_per_volley": np.asarray(w_group_a @ M).ravel(), f"w_{b}_mv_per_volley": np.asarray(w_group_b @ M).ravel(),
                                 "room_mean_hz": np.asarray(mean_g_sum @ M).ravel() / np.maximum(n_lab, 1), "room_max_hz": max_lab,
                                 "frac_cells_firing": np.asarray(firing_g @ M).ravel() / np.maximum(n_lab, 1),
                                 f"I_{a}_mean": IL.mean(0), f"I_{b}_mean": IR.mean(0), "LR_mean": D.mean(0), "LR_sd_flies": D.std(0, ddof=1),
                                 "LR_n_flies": D.shape[0], "LR_run_means": [list(map(float, run_means[:, j])) for j in range(len(uk))],
                                 "LR_run_sd": run_means.std(0, ddof=1) if len(runs) > 1 else np.nan,
                                 "LR_z_runs": z_runs,
                                 "abs_I_mean": (np.abs(IL).mean(0) + np.abs(IR).mean(0)) / 2,
                                 "abs_w_mean": (np.abs(np.asarray(w_group_a @ M).ravel()) + np.abs(np.asarray(w_group_b @ M).ravel())) / 2,
                                 "LR_sign_consistency_flies": (np.sign(D) == np.sign(D.mean(0))[None]).mean(0)})
            if level == "type_side":
                rows["side"] = sides[np.searchsorted(keys, uk)] if np.array_equal(uk, keys) else [k.split("/")[1] for k in uk]
                rows["pre_type"] = [k.split("/")[0] for k in uk]
            rows = rows.sort_values("abs_I_mean", ascending=False).reset_index(drop=True)
            out[f"{a}|{b}:{level}"] = rows
        # the totals
        totL = np.concatenate([r.z["tot"][:, ia].mean(0) for r in runs]); totR = np.concatenate([r.z["tot"][:, ib].mean(0) for r in runs])
        out[f"{a}|{b}:totals"] = pd.DataFrame({"fly": np.arange(len(totL)), "run": run_of, f"I_{a}": totL, f"I_{b}": totR, "LR": totL - totR,
                                               f"E_{a}": np.concatenate([r.z["totE"][:, ia].mean(0) for r in runs]), f"I_neg_{a}": np.concatenate([r.z["totI"][:, ia].mean(0) for r in runs]),
                                               f"E_{b}": np.concatenate([r.z["totE"][:, ib].mean(0) for r in runs]), f"I_neg_{b}": np.concatenate([r.z["totI"][:, ib].mean(0) for r in runs])})
    return out


def _block_shuffle_null(x: np.ndarray, y: np.ndarray, block: int, n: int, rng) -> float:
    """95th percentile of |corr(x, shuffled y)| over `n` block permutations of y (blocks of `block` samples)."""
    T = len(x)
    nb = max(T // block, 1)
    yb = y[: nb * block].reshape(nb, block)
    xb = x[: nb * block]
    vals = []
    for _ in range(n):
        ys = yb[rng.permutation(nb)].ravel()
        vals.append(abs(np.corrcoef(xb, ys)[0, 1]) if xb.std() > 0 and ys.std() > 0 else 0.0)
    return float(np.percentile(vals, 95))


def cancellation_tables(runs: list[Run], t_skip_s: float, n_shuffle: int, seed: int = 0) -> dict:
    """Per pair, per fly: the time series of the L-R input asymmetry of every kept group, its variance share in the
    total asymmetry's variance (Var(total) = sum_X Cov(asym_X, total)), the pairs of groups whose asymmetries
    anti-correlate most (cancellation), and the correlation of the total and of each group's asymmetry with the
    DNa02 rate L-R and the body's yaw command, against a 1 s block-shuffle null."""
    rng = np.random.default_rng(seed)
    out = {}
    for a, b in PAIRS:
        pair = f"{a}|{b}"
        keys = runs[0].keys
        # each run kept its own top-K groups: the analysis runs on the groups every run kept (their intersection)
        kept_labels = [set(keys[r.z[f"keep__{pair}"]].tolist()) for r in runs]
        common_set = set.intersection(*kept_labels)
        labels = np.array([k for k in keys[runs[0].z[f"keep__{pair}"]] if k in common_set])
        types = np.array([k.split("/")[0] for k in labels])
        ia, ib = runs[0].targets.index(a), runs[0].targets.index(b)
        per_fly = []; corr_rows = []; pair_rows = []
        cov_share = []; var_group = []; var_tot = []; var_explained = []; sd_tot = []; sd_dna = []
        for r in runs:
            sa_all, keep_r = r.series(a, pair); sb_all, _ = r.series(b, pair)                       # (T, B, K_run)
            pos = {k: i for i, k in enumerate(keys[keep_r])}
            cols = [pos[k] for k in labels]
            sa, sb = sa_all[:, :, cols], sb_all[:, :, cols]                                         # (T, B, K)
            dt_s = float(np.diff(r.z["t_ms"][:2])[0]) / 1000.0
            k0 = int(round(t_skip_s / dt_s))
            asym = (sa - sb)[k0:]
            block = max(1, min(int(round(1.0 / dt_s)), asym.shape[0] // 4))                   # 1 s blocks (shorter on a smoke run)                                                                   # (T', B, K)
            tot = (r.z["tot"][:, ia] - r.z["tot"][:, ib])[k0:]                                     # (T', B)
            dna = (r.motor("turn_L") - r.motor("turn_R"))[k0:]
            leg = (r.motor("leg_L") - r.motor("leg_R"))[k0:]
            yaw = r.z["yaw_cmd"][k0:]; air = r.z["airborne"][k0:]
            for bi in range(r.B):
                ok = ~air[:, bi]
                A_ = asym[ok, bi]; t_ = tot[ok, bi]
                if ok.sum() < 3 * block:
                    continue
                vt = t_.var(ddof=1)
                cov = np.array([np.cov(A_[:, j], t_)[0, 1] for j in range(A_.shape[1])])
                var_tot.append(vt); cov_share.append(cov / vt if vt > 0 else np.full(len(cov), np.nan))
                var_group.append(A_.var(0, ddof=1)); var_explained.append(A_.sum(1).var(ddof=1) / vt if vt > 0 else np.nan)
                sd_tot.append(np.sqrt(vt)); sd_dna.append(dna[ok, bi].std(ddof=1))
                # correlations with the readouts
                for name, y in (("DNa02_LR_hz", dna[ok, bi]), ("leg_LR_hz", leg[ok, bi]), ("yaw_cmd", yaw[ok, bi])):
                    if y.std() == 0 or t_.std() == 0:
                        rr, null95 = np.nan, np.nan
                    else:
                        rr = float(np.corrcoef(t_, y)[0, 1]); null95 = _block_shuffle_null(t_, y, block, n_shuffle, rng)
                    corr_rows.append({"run": r.seed, "fly": bi, "x": f"total_LR_input[{pair}]", "y": name, "r": rr, "null95_abs_r": null95, "n": int(ok.sum())})
                per_fly.append({"run": r.seed, "fly": bi, "n_frames": int(ok.sum()), "sd_total_LR_input": float(np.sqrt(vt)), "mean_total_LR_input": float(t_.mean()),
                                "sd_DNa02_LR_hz": float(dna[ok, bi].std(ddof=1)), "mean_DNa02_LR_hz": float(dna[ok, bi].mean()),
                                "sd_leg_LR_hz": float(leg[ok, bi].std(ddof=1)), "mean_leg_LR_hz": float(leg[ok, bi].mean()),
                                "sd_yaw_cmd_deg_s": float(np.degrees(yaw[ok, bi].std(ddof=1))), "kept_groups_explained_var_share": float(var_explained[-1])})
            # the most anti-correlated pairs of group asymmetries, pooled over this run's flies
            A_all = np.concatenate([asym[~air[:, bi], bi] for bi in range(r.B)], 0)
            A_all = A_all - A_all.mean(0)
            C = np.cov(A_all.T)
            sdv = np.sqrt(np.diag(C)); Cn = C / np.outer(sdv, sdv + 1e-30)
            iu = np.triu_indices(len(labels), 1)
            order = np.argsort(C[iu])[:12]
            for o in order:
                i, j = iu[0][o], iu[1][o]
                pair_rows.append({"run": r.seed, "group_i": labels[i], "group_j": labels[j], "cov": float(C[i, j]), "corr": float(Cn[i, j]),
                                  "sd_i": float(sdv[i]), "sd_j": float(sdv[j]), "sd_sum": float(np.sqrt(max(C[i, i] + C[j, j] + 2 * C[i, j], 0)))})
        if not cov_share:
            out[pair] = {"groups": pd.DataFrame(), "per_fly": pd.DataFrame(), "correlations": pd.DataFrame(), "anticorrelated_pairs": pd.DataFrame(),
                         "summary": {"kept_groups": int(len(labels)), "note": "no fly had enough non-airborne frames after --skip"}}
            continue
        cs = np.array(cov_share); vg = np.array(var_group)
        groups = pd.DataFrame({"group": labels, "pre_type": types, "mean_var_share_of_total": np.nanmean(cs, 0), "sd_var_share_flies": np.nanstd(cs, 0, ddof=1),
                               "mean_sd_asym_mV_s": np.sqrt(vg).mean(0), "sd_asym_over_sd_total": (np.sqrt(vg) / np.array(sd_tot)[:, None]).mean(0),
                               "n_flies": len(cs)}).sort_values("mean_sd_asym_mV_s", ascending=False).reset_index(drop=True)
        out[pair] = {"groups": groups, "per_fly": pd.DataFrame(per_fly), "correlations": pd.DataFrame(corr_rows), "anticorrelated_pairs": pd.DataFrame(pair_rows),
                     "summary": {"kept_groups": int(len(labels)), "kept_groups_per_run": [len(x) for x in kept_labels], "explained_var_share_mean": float(np.nanmean(var_explained)), "explained_var_share_min": float(np.nanmin(var_explained)),
                                 "sum_of_group_variances_over_total_var_mean": float(np.mean(vg.sum(1) / np.array(var_tot))),
                                 "sd_total_LR_input_mV_s_mean": float(np.mean(sd_tot)), "sd_DNa02_LR_hz_mean": float(np.mean(sd_dna))}}
    return out


def watch_tables(runs: list[Run], t_skip_s: float) -> pd.DataFrame:
    rows = []
    names = runs[0].meta["watch"]
    for r in runs:
        dt_s = float(np.diff(r.z["t_ms"][:2])[0]) / 1000.0; k0 = int(round(t_skip_s / dt_s))
        wr = r.z["watch_rates"][k0:]                                                                 # (T, B, n)
        for j, nm in enumerate(names):
            rows.append({"run": r.seed, "type_side": nm, "mean_hz": float(wr[:, :, j].mean()), "fly_sd_of_mean": float(wr[:, :, j].mean(0).std(ddof=1)),
                         "temporal_sd_mean": float(wr[:, :, j].std(0, ddof=1).mean()), "max_hz": float(wr[:, :, j].max())})
    df = pd.DataFrame(rows)
    g = df.groupby("type_side").agg(mean_hz=("mean_hz", "mean"), run_sd=("mean_hz", "std"), fly_sd=("fly_sd_of_mean", "mean"), temporal_sd=("temporal_sd_mean", "mean"), max_hz=("max_hz", "max"), n_runs=("run", "nunique")).reset_index()
    g["type"] = g.type_side.str.rsplit("_", n=1).str[0]; g["side"] = g.type_side.str.rsplit("_", n=1).str[1]
    piv = g.pivot(index="type", columns="side", values="mean_hz")
    lr = (piv.get("L") - piv.get("R")) if "L" in piv and "R" in piv else None
    g = g.merge(pd.DataFrame({"type": piv.index, "LR_mean_hz": lr.to_numpy() if lr is not None else np.nan}), on="type", how="left")
    return g.sort_values(["type", "side"]).reset_index(drop=True)


def behaviour_table(runs: list[Run], t_skip_s: float) -> pd.DataFrame:
    rows = []
    for r in runs:
        dt_s = float(np.diff(r.z["t_ms"][:2])[0]) / 1000.0; k0 = int(round(t_skip_s / dt_s))
        yr = r.z["yaw_rate"][k0:]; air = r.z["airborne"][k0:]; yc = r.z["yaw_cmd"][k0:]
        dna = (r.motor("turn_L") - r.motor("turn_R"))[k0:]; leg = (r.motor("leg_L") - r.motor("leg_R"))[k0:]
        wa = r.z["wind_angle"][k0:]
        for bi in range(r.B):
            ok = ~air[:, bi]
            rows.append({"run": r.seed, "fly": bi, "env_seed": r.meta["env_seeds"][bi], "yaw_rate_sd_deg_s": float(np.degrees(yr[ok, bi].std(ddof=1))),
                         "yaw_cmd_sd_deg_s": float(np.degrees(yc[ok, bi].std(ddof=1))), "yaw_cmd_mean_deg_s": float(np.degrees(yc[ok, bi].mean())),
                         "DNa02_LR_mean_hz": float(dna[ok, bi].mean()), "DNa02_LR_sd_hz": float(dna[ok, bi].std(ddof=1)), "DNa02_L_mean_hz": float(r.motor("turn_L")[k0:][ok, bi].mean()),
                         "DNa02_R_mean_hz": float(r.motor("turn_R")[k0:][ok, bi].mean()), "leg_LR_mean_hz": float(leg[ok, bi].mean()), "leg_LR_sd_hz": float(leg[ok, bi].std(ddof=1)),
                         "leg_L_mean_hz": float(r.motor("leg_L")[k0:][ok, bi].mean()), "airborne_frac": float(air[:, bi].mean()),
                         "wind_angle_mean_deg": float(np.degrees(np.arctan2(np.sin(wa[ok, bi]).mean(), np.cos(wa[ok, bi]).mean()))),
                         "wind_left_frac": float((wa[ok, bi] > 0).mean()), "speed_cmd_mean_m_s": float(r.z["speed_cmd"][k0:][ok, bi].mean())})
    return pd.DataFrame(rows)


def wind_side_table(runs: list[Run], t_skip_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per fly: the fraction of frames with the wind on the left and the rollout-mean L-R of the lateralised readouts
    (the wind DNs, the DNa02 movers, DNa02, the leg MNs, LLPC1, and the PS230 -> DNa02 input); then, per column, the
    correlation with the wind side over all flies and per run, and the slope per unit wind_left_frac. The wind is the
    one lateralised sensory signal of the room, so this is the cross-fly test of which stage it reaches."""
    rows = []
    for r in runs:
        dt_s = float(np.diff(r.z["t_ms"][:2])[0]) / 1000.0; k0 = int(round(t_skip_s / dt_s))
        wa = r.z["wind_angle"][k0:]; wr = r.z["watch_rates"][k0:]; j = {nm: i for i, nm in enumerate(r.meta["watch"])}
        gm = r.z["group_mean_flies"]; tn = r.targets; keys = r.keys
        def group(target, key):
            w = np.where(keys == key)[0]
            return gm[tn.index(target)][:, w[0]] if len(w) else np.zeros(r.B)
        ps = group("DNa02_L", "PS230/L") - group("DNa02_R", "PS230/R")
        for b in range(r.B):
            row = {"run": r.seed, "fly": b, "wind_left_frac": float((wa[:, b] > 0).mean()), "PS230_LR_input_DNa02_mV_s": float(ps[b]),
                   "leg_LR": float((r.motor("leg_L")[k0:, b] - r.motor("leg_R")[k0:, b]).mean())}
            for t in ("DNp18", "DNp33", "DNae007", "DNge035", "DNa02", "LLPC1", "DNa13", "DNge037"):
                if f"{t}_L" in j and f"{t}_R" in j:
                    row[f"{t}_LR"] = float(wr[:, b, j[f"{t}_L"]].mean() - wr[:, b, j[f"{t}_R"]].mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    cols = [c_ for c_ in df.columns if c_.endswith("_LR") or c_.startswith("PS230")]
    stats = []
    for col in cols:
        x = df.wind_left_frac.to_numpy(); y = df[col].to_numpy()
        r_all = float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 else np.nan
        r_runs = [float(np.corrcoef(d.wind_left_frac, d[col])[0, 1]) if d[col].std() > 0 else np.nan for _, d in df.groupby("run")]
        stats.append({"readout": col, "r_all_flies": r_all, "r_per_run": r_runs, "slope_per_unit_wind_left": float(np.polyfit(x, y, 1)[0]) if y.std() > 0 else np.nan,
                      "mean": float(y.mean()), "sd_flies": float(y.std(ddof=1)), "n_flies": int(len(y)),
                      "consistent_sign_over_runs": bool(len(set(np.sign(r_runs))) == 1)})
    return df, pd.DataFrame(stats)


def cmd_wind_side(args) -> int:
    files = sorted({str(Path(f).with_suffix("")) for pat in args.recordings for f in globmod.glob(pat + ".npz") + globmod.glob(pat)
                    if f.endswith(".npz") and not f.endswith(("_flies.npz", "_max.npz"))})
    runs = [Run(f) for f in files]
    df, st = wind_side_table(runs, args.skip)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False); st.to_csv(out.with_name(out.stem + "_stats.csv"), index=False)
    print(f"{len(runs)} runs, {len(df)} flies; wind_left_frac range {df.wind_left_frac.min():.2f}-{df.wind_left_frac.max():.2f}")
    common.print_table(st, max_rows=20)
    print(f"wrote {out} and {out.with_name(out.stem + '_stats.csv')}")
    return 0


def run_decompose(c, runs: list[Run], window, top: int, out_dir: Path) -> dict:
    """The decompose tool on the per-fly recordings (frames = flies), 3 runs as the arm, by (type, side) and by type."""
    from flyverse.interp import decompose as dec
    lif = None
    res = {}
    recs = [r.flies for r in runs]
    for k, spec in TARGETS.items():
        for by in (("type", "side"), ("type",)):
            key = f"{k}:{'_'.join(by)}"
            R = dec.decompose(c, spec, recording={"plain": recs}, params=lif, by=by, tiers=False, top=top, keep_links=False)
            R.files["generator"] = f"scripts/interp_apply_turning.py analyse (decompose {k} by {','.join(by)}; frames = flies)"
            R.summary["frames_are"] = "flies: each recording frame is one fly's rollout-mean rate; window mean = mean over the run's 16 flies; the 3 runs are the replicates"
            p = out_dir / f"decompose_{key.replace(':', '_')}.json"
            R.save(p)
            res[key] = {"json": str(p), "result": R}
    return res


def run_paths(c, movers: list[str], recording_max: str | None, k_max: int, top: int, out_dir: Path, sources: dict) -> dict:
    from flyverse.interp import paths as P
    ew = common.effective_weights(c)
    counts, _ = P.raw_counts(c)
    aif = P.if_signed_magnitudes(c, ew, counts)
    out = {}
    for mover in movers:
        for sname, spec in sources.items():
            t0 = time.time()
            try:
                R = P.paths(c, spec, mover, k_max=k_max, top=top, recording=recording_max, frozen="static", ew=ew, counts=counts, aif=aif, contributions_per_link=20)
            except Exception as e:                                                                   # a source with no path is a finding, not a crash
                out[f"{sname}->{mover}"] = {"error": str(e)}
                _log(f"paths {sname} -> {mover}: {e}")
                continue
            R.files["generator"] = f"scripts/interp_apply_turning.py analyse (paths {sname} -> {mover}, k <= {k_max})"
            p = out_dir / f"paths_{sname}_to_{re.sub(r'[^A-Za-z0-9_]+', '_', mover)}.json"
            R.save(p)
            pt = R.table("paths")
            summ = {"json": str(p), "n_nodes": R.summary.get("n_nodes"), "seconds": round(time.time() - t0, 1),
                    "strongest_silent_link_per_k": R.summary.get("strongest_silent_link_per_k"), "direct": R.summary.get("direct"),
                    "top_signed_walk_per_k": {}, "top_silent_walk_per_k": {}}
            for k in range(1, k_max + 1):
                for kind in ("signed", "silent"):
                    g = pt[(pt.k == k) & (pt.kind == kind)].head(3) if len(pt) else pt
                    summ[f"top_{kind}_walk_per_k"][str(k)] = g[["path", "gain", "gain_if_signed", "signs", "silent_links", "raw_counts"]].to_dict("records") if len(g) else []
            bi = R.table("b_inputs")
            if len(bi):
                summ["b_inputs_top"] = bi.head(12)[["rank", "pre_type", "n_pre_cells", "raw_count", "share_of_b_input", "nt", "sign", "mv_per_post_volley", "mv_per_post_volley_if_signed", "silent", "silent_partial"]].to_dict("records")
                summ["b_silent_input_share"] = R.summary.get("b_silent_input_share"); summ["b_sign0_input_share"] = R.summary.get("b_sign0_input_share")
                nf = bi[bi.silent.astype(str).str.contains("never_firing")] if "silent" in bi else bi.iloc[0:0]
                summ["b_never_firing_input_share"] = float(nf.share_of_b_input.sum()) if len(nf) else None
            out[f"{sname}->{mover}"] = summ
            _log(f"paths {sname} -> {mover}: {summ['n_nodes']} nodes, {summ['seconds']} s; strongest silent per k: " +
                 "; ".join(f"k{k}: {v['pre']}->{v['post']} [{v['silent']}] {v['mv_per_post_volley_if_signed']:+.1f}" if v else f"k{k}: none" for k, v in (summ['strongest_silent_link_per_k'] or {}).items()))
    return out


def movers_from_atlas(res, readouts, top: int, exclude_self: bool = True) -> pd.DataFrame:
    mv = res.table("movers") if "movers" in res.tables else res.table("atlas")
    if not len(mv):
        return mv
    mv = mv[mv.readout.isin(readouts) & (mv.verdict == "result")]
    if exclude_self and "self_drive" in mv:
        mv = mv[~mv.self_drive.astype(bool)]
    mv = mv.assign(abs_diff=mv["diff"].abs()).sort_values(["readout", "abs_diff"], ascending=[True, False])
    mv = mv.groupby("readout").head(top).drop(columns="abs_diff").reset_index(drop=True)
    mv["base_population"] = mv.population.astype(str).str.replace(r"#\d+$", "", regex=True)      # '#n' = the same cells listed twice
    return mv


def cmd_analyse(args) -> int:
    out_dir = Path(args.json).parent if args.json else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    c = _load(args)
    files = []
    for pat in args.recordings:
        files += [str(Path(f).with_suffix("")) for f in sorted(globmod.glob(pat + ".npz")) + sorted(globmod.glob(pat)) if f.endswith(".npz") and not f.endswith(("_flies.npz", "_max.npz"))]
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no recordings match {args.recordings}")
    runs = [Run(f) for f in files]
    devices = sorted({r.meta["device"] for r in runs})
    _log(f"{len(runs)} runs: seeds {[r.seed for r in runs]}, devices {devices}, B {[r.B for r in runs]}, captured frames {[r.z['tot'].shape[0] for r in runs]}")
    summary = {"tool": TOOL, "generator": "scripts/interp_apply_turning.py " + " ".join(sys.argv[1:]), "runs": [{"seed": r.seed, "file": str(r.prefix), "device": r.meta["device"], "wall_s": r.meta["wall_s"]} for r in runs],
               "replicates": {"n": len(runs), "unit": "runs", "flies_per_run": runs[0].B}, "provenance": runs[0].meta["provenance"], "window_skip_s": args.skip}
    if len(runs) < common.MIN_REPLICATES:
        summary["warning"] = f"fewer than {common.MIN_REPLICATES} runs: every difference below is underpowered"

    # ---- behaviour and the readouts
    beh = behaviour_table(runs, args.skip)
    summary["behaviour"] = {k: {"mean": float(beh[k].mean()), "sd_flies": float(beh[k].std(ddof=1)), "median": float(beh[k].median()), "min": float(beh[k].min()), "max": float(beh[k].max()),
                                "run_means": [float(beh[beh.run == r.seed][k].mean()) for r in runs]} for k in ("yaw_rate_sd_deg_s", "yaw_cmd_sd_deg_s", "DNa02_LR_mean_hz", "DNa02_LR_sd_hz", "DNa02_L_mean_hz", "DNa02_R_mean_hz", "leg_LR_mean_hz", "leg_LR_sd_hz", "leg_L_mean_hz", "airborne_frac")}
    # does the DNa02 asymmetry follow the wind angle across flies?  (the one lateralised sensory signal the room has)
    wl = beh.wind_left_frac.to_numpy(); d = beh.DNa02_LR_mean_hz.to_numpy(); lg = beh.leg_LR_mean_hz.to_numpy()
    summary["wind_side_vs_asymmetry"] = {"r_windleft_DNa02_LR": float(np.corrcoef(wl, d)[0, 1]) if d.std() > 0 else None, "r_windleft_leg_LR": float(np.corrcoef(wl, lg)[0, 1]) if lg.std() > 0 else None, "n_flies": int(len(beh))}
    watch = watch_tables(runs, args.skip)
    wind_df, wind_stats = wind_side_table(runs, args.skip)
    summary["wind_side_per_readout"] = wind_stats.to_dict("records")

    # ---- (a) decompose: the tool on the per-fly recordings, then the L-R tables and the cancellation analysis
    dec_res = run_decompose(c, runs, None, args.top, out_dir) if not args.no_decompose else {}
    lr = lr_tables(runs, c)
    canc = cancellation_tables(runs, args.skip, args.shuffles)
    # ---- (b) atlas
    atlas_summary = {}
    movers = pd.DataFrame()
    if args.atlas_runs:
        from flyverse.interp import atlas as atlas_mod
        afiles = []
        for pat in args.atlas_runs:
            afiles += [str(Path(f).with_suffix("")) for f in sorted(globmod.glob(pat + ".npz")) + sorted(globmod.glob(pat)) if f.endswith(".npz")]
        afiles = sorted(set(afiles))
        aruns = [atlas_mod.AtlasRun.load(f) for f in afiles]
        for f, r in zip(afiles, aruns):
            r.meta.setdefault("file", f + ".npz")
        ares = atlas_mod.analyse_runs(aruns, c=c, top=args.top, generator="scripts/interp_apply_turning.py analyse (atlas)")
        ap = out_dir / "atlas_turning.json"; ares.save(ap)
        readouts = ["turn_LR", "leg_LR", "type.DNa02_LR", "type.DNge035_LR", "type.DNa13_LR", "type.PFL3_LR"]
        readouts = [r_ for r_ in readouts if r_ in set(ares.table("movers").readout)] if len(ares.table("movers")) else readouts
        movers = movers_from_atlas(ares, readouts, args.top)
        movers["atlas"] = "turning"
        atlas_summary = {"json": str(ap), "n_runs": len(aruns), "devices": [r.meta["provenance"]["execution"].get("device") for r in aruns],
                         "n_populations": ares.summary.get("n_populations"), "n_readouts": ares.summary.get("n_readouts"),
                         "top_population_per_readout": {k: ares.summary["top_population_per_readout"][k] for k in readouts if k in ares.summary.get("top_population_per_readout", {})}}
    if args.dn_atlas and os.path.exists(args.dn_atlas):
        dres = common.Result.load(args.dn_atlas)
        dm = movers_from_atlas(dres, ["turn_LR", "leg_LR"], args.top); dm["atlas"] = "dn_sensory"
        movers = pd.concat([movers, dm], ignore_index=True) if len(movers) else dm
        atlas_summary["dn_sensory_json"] = args.dn_atlas
    # ---- (c) paths
    paths_summary = {}
    if not args.no_paths:
        if args.movers:
            mover_list = args.movers
        else:
            cand = movers[movers.readout.isin(["turn_LR", "leg_LR"])] if len(movers) else movers
            names = []
            for lab in (cand.population.tolist() if len(cand) else []):
                t = re.sub(r"_(L|R|\?)$", "", re.sub(r"#\d+$", "", str(lab)))          # the atlas suffixes a duplicate label '#n'
                if t.startswith(("class.", "superclass.", "channel.")) or not len(common.resolve(c, f"type={t}")):
                    continue
                if t not in names:
                    names.append(t)
            mover_list = names[: args.n_movers] if names else DN_MOVERS[:6]
            for t in ("DNa02",):
                if t not in mover_list:
                    mover_list.append(t)
        rec_max = str(runs[0].prefix.with_name(runs[0].prefix.name + "_max"))
        sources = {k: SOURCES[k] for k in (args.sources.split(",") if args.sources else SOURCES)}
        paths_summary = run_paths(c, mover_list, rec_max, args.k, args.top, out_dir, sources)
        paths_summary["_movers"] = mover_list; paths_summary["_never_firing_recording"] = rec_max; paths_summary["_sources"] = sources

    # ---- write
    tables = {"behaviour_per_fly": beh, "watch_types": watch, "atlas_movers": movers, "per_fly_wind_side": wind_df, "wind_side_per_readout": wind_stats}
    for k, v in lr.items():
        tables[f"lr_{k}"] = v
    for pair, d in canc.items():
        for k, v in d.items():
            if isinstance(v, pd.DataFrame):
                tables[f"cancel_{pair}_{k}"] = v
        summary.setdefault("cancellation", {})[pair] = d["summary"]
    summary["lr_top"] = {}
    for a, b in PAIRS:
        t = lr[f"{a}|{b}:type"]
        cols = ["pre_group", "n_pre_cells", f"w_{a}_mv_per_volley", f"w_{b}_mv_per_volley", "room_mean_hz", "room_max_hz", "frac_cells_firing", f"I_{a}_mean", f"I_{b}_mean", "LR_mean", "LR_sd_flies", "LR_run_means", "LR_z_runs", "LR_sign_consistency_flies"]
        by_w = t.sort_values("abs_w_mean", ascending=False).head(30)
        summary["lr_top"][f"{a}|{b}"] = {"by_abs_input": t.head(20)[cols].to_dict("records"),
                                          "by_LR_z_runs": t.reindex(t.LR_z_runs.abs().sort_values(ascending=False).index).head(20)[cols].to_dict("records"),
                                          "by_static_weight": by_w[cols].to_dict("records"),
                                          "silent_at_source": {"n_types_top30_by_weight_never_firing": int((by_w.room_max_hz < common.NEVER_FIRING_HZ).sum()),
                                                               "types": by_w[by_w.room_max_hz < common.NEVER_FIRING_HZ].pre_group.tolist(),
                                                               "share_of_abs_static_weight_never_firing": float(t[t.room_max_hz < common.NEVER_FIRING_HZ].abs_w_mean.sum() / max(t.abs_w_mean.sum(), 1e-9))},
                                          "totals": {k_: {"mean": float(lr[f"{a}|{b}:totals"][k_].mean()), "sd_flies": float(lr[f"{a}|{b}:totals"][k_].std(ddof=1)),
                                                          "run_means": [float(lr[f"{a}|{b}:totals"][lr[f"{a}|{b}:totals"].run == i][k_].mean()) for i in range(len(runs))]} for k_ in (f"I_{a}", f"I_{b}", "LR", f"E_{a}", f"I_neg_{a}", f"E_{b}", f"I_neg_{b}")}}
    summary["decompose"] = {k: {"json": v["json"], "coverage": v["result"].summary.get("coverage"), "dynamic": v["result"].summary.get("dynamic")} for k, v in dec_res.items()}
    summary["atlas"] = atlas_summary
    summary["paths"] = paths_summary
    summary["watch_LR"] = watch.drop_duplicates("type")[["type", "LR_mean_hz"]].set_index("type").LR_mean_hz.to_dict()
    doc = {"schema": "flyverse.interp.apply_turning/1", "summary": summary, "tables": {k: common.to_jsonable(v.to_dict("records")) for k, v in tables.items()}}
    path = Path(args.json) if args.json else out_dir / "turning.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(doc), f, indent=1)
    # ---- print
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n== behaviour (per-fly stats over the window; runs = replicates) ==")
    for k, v in summary["behaviour"].items():
        print(f"  {k:22s} mean {v['mean']:+8.3f}  fly sd {v['sd_flies']:7.3f}  run means {['%+.3f' % x for x in v['run_means']]}")
    print(f"  wind side vs asymmetry: {summary['wind_side_vs_asymmetry']}")
    print("-- per-fly wind side (fraction of frames with the wind on the left) vs the lateralised readouts, 48 flies --")
    common.print_table(wind_stats, max_rows=20)
    print("\n== watch types (mean Hz over flies and runs; L-R) ==")
    common.print_table(watch[["type_side", "mean_hz", "run_sd", "fly_sd", "temporal_sd", "max_hz", "LR_mean_hz"]], max_rows=80)
    for a, b in PAIRS:
        t = lr[f"{a}|{b}:type"]
        print(f"\n== (a) {a} vs {b}: input by presynaptic type (mV/s per post cell; mean over 48 flies; L-R with fly sd and the 3 run means) ==")
        common.print_table(t.head(args.top)[["pre_group", "n_pre_cells", f"w_{a}_mv_per_volley", f"w_{b}_mv_per_volley", "room_mean_hz", "room_max_hz", f"I_{a}_mean", f"I_{b}_mean", "LR_mean", "LR_sd_flies", "LR_z_runs", "LR_sign_consistency_flies"]], max_rows=args.top)
        print(f"-- the same, ranked by static weight |w| (the room rate says which wired inputs fire at all) --")
        common.print_table(t.sort_values("abs_w_mean", ascending=False).head(args.top)[["pre_group", "n_pre_cells", f"w_{a}_mv_per_volley", f"w_{b}_mv_per_volley", "room_mean_hz", "room_max_hz", "frac_cells_firing", f"I_{a}_mean", f"I_{b}_mean", "LR_mean"]], max_rows=args.top)
        print(f"-- the same, ranked by |L-R z over runs| --")
        common.print_table(t.reindex(t.LR_z_runs.abs().sort_values(ascending=False).index).head(15)[["pre_group", "n_pre_cells", f"I_{a}_mean", f"I_{b}_mean", "LR_mean", "LR_sd_flies", "LR_run_means", "LR_z_runs"]], max_rows=15)
        tt = summary["lr_top"][f"{a}|{b}"]["totals"]
        print("-- totals: " + "; ".join(f"{k_} {v['mean']:+.1f} (fly sd {v['sd_flies']:.1f}; runs {['%+.1f' % x for x in v['run_means']]})" for k_, v in tt.items()))
        cs = canc[f"{a}|{b}"]
        print(f"\n== cancellation {a}|{b}: per-fly time series of the group L-R asymmetries ({cs['summary']}) ==")
        common.print_table(cs["groups"].head(20), max_rows=20)
        print("-- most anti-correlated group pairs (pooled flies per run) --")
        common.print_table(cs["anticorrelated_pairs"].sort_values("corr").head(12), max_rows=12)
        cr = cs["correlations"]
        if len(cr):
            g = cr.groupby("y").agg(r_mean=("r", "mean"), r_sd=("r", "std"), r_abs_mean=("r", lambda x: np.nanmean(np.abs(x))), null95_mean=("null95_abs_r", "mean"), n=("r", "size")).reset_index()
            print("-- corr(total L-R input, readout) per fly, against a 1 s block-shuffle null (95th pct |r|) --")
            common.print_table(g, max_rows=10)
    if len(movers):
        print("\n== (b) atlas movers of the turning readouts (not self-drive; verdict result) ==")
        common.print_table(movers[[x for x in ("atlas", "readout", "population", "n_cells", "diff", "z_floor", "stim_sd", "n_runs", "verdict") if x in movers.columns]], max_rows=120)
    if paths_summary:
        print("\n== (c) paths: strongest silent link and top signed walk per k, per source -> mover ==")
        for key, s in paths_summary.items():
            if key.startswith("_") or "error" in s:
                continue
            print(f"\n{key}  ({s['n_nodes']} nodes; direct {s['direct']})")
            for k, v in (s["strongest_silent_link_per_k"] or {}).items():
                print(f"   silent k={k}: " + ("none" if v is None else f"{v['pre']} -> {v['post']} [{v['silent']}] {v['mv_per_post_volley_if_signed']:+.2f} mV/volley if signed, {v['raw_count']:.0f} syn, {100 * v['share_of_post_input']:.1f} % of post input, on {v['on_path']}"))
            for k, ws in s["top_signed_walk_per_k"].items():
                for w_ in ws[:2]:
                    print(f"   signed k={k}: {w_['path']}  gain {w_['gain']:+.4g}  silent {w_['silent_links']}")
            if s.get("b_inputs_top"):
                print(f"   b inputs (silent share {s.get('b_silent_input_share')}, never_firing share {s.get('b_never_firing_input_share')}): " +
                      ", ".join(f"{r_['pre_type']} {100 * r_['share_of_b_input']:.1f}%{'[' + r_['silent'] + ']' if r_['silent'] else ''}" for r_ in s["b_inputs_top"][:10]))
    print(f"\nwrote {path}")
    return 0


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="GPU: one plain-fly room rollout (16 flies x 60 s) with the targets' presynaptic input recorded")
    r.add_argument("--out", required=True); r.add_argument("--batch", type=int, default=16); r.add_argument("--seconds", type=float, default=60.0)
    r.add_argument("--every", type=int, default=2, help="capture every n-th 10 ms frame (rate_tau is 100 ms)")
    r.add_argument("--mean-every", type=int, default=1, help="accumulate the per-cell mean / max rate every n-th frame")
    r.add_argument("--top-groups", type=int, default=64, help="kept time series: the union of the top-K groups of each L / R pair")
    r.add_argument("--no-fence", action="store_true"); r.add_argument("--energy", type=float, default=0.9)
    a = sub.add_parser("atlas-run", help="GPU: one atlas run over the turning candidates (DNa02 inputs, LAL, CX outputs, DN movers)")
    a.add_argument("--out", required=True); a.add_argument("--hz", type=float, default=150.0); a.add_argument("--ms", type=float, default=400.0)
    a.add_argument("--settle-ms", type=float, default=200.0); a.add_argument("--batch", type=int, default=64); a.add_argument("--n-null", type=int, default=4)
    a.add_argument("--min-entries", type=int, default=4, help="DNa02 presynaptic types with at least this many stored entries")
    a.add_argument("--limit", type=int, default=0, help="smoke tests: keep only the first n populations")
    p = sub.add_parser("plan", help="write the one-batch cluster line")
    p.add_argument("--out", default=str(OUT_DIR)); p.add_argument("--minutes", type=int, default=40); p.add_argument("--seconds", type=float, default=60.0)
    n = sub.add_parser("analyse", help="CPU: decompose + L-R + cancellation, atlas movers, paths")
    n.add_argument("--recordings", action="append", required=True, help="run prefix or glob (repeatable)")
    n.add_argument("--atlas-runs", action="append", default=[], help="atlas run prefix or glob (repeatable)")
    n.add_argument("--dn-atlas", default=str(Path(ROOT) / "out" / "interp" / "atlas" / "dn_sensory.json"), help="the DN + sensory atlas Result (interp_atlas.md)")
    n.add_argument("--skip", type=float, default=5.0, help="seconds dropped at the start of the rollout (the brain's start-up transient)")
    n.add_argument("--top", type=int, default=30); n.add_argument("--shuffles", type=int, default=100)
    n.add_argument("--k", type=int, default=3); n.add_argument("--n-movers", type=int, default=6)
    n.add_argument("--movers", action="append", default=[], help="paths targets (types); default: the atlas movers of turn_LR / leg_LR")
    n.add_argument("--sources", default=None, help="comma-separated subset of " + ",".join(SOURCES))
    n.add_argument("--no-paths", action="store_true"); n.add_argument("--no-decompose", action="store_true")
    w = sub.add_parser("wind-side", help="CPU: the 48-fly wind-side table alone (which lateralised stage the room's wind reaches)")
    w.add_argument("--recordings", action="append", required=True); w.add_argument("--skip", type=float, default=5.0)
    w.add_argument("--out", default=str(OUT_DIR / "per_fly_wind_side.csv"))
    for x in (r, a, p, n, w):
        common.add_common_args(x)
    args = ap.parse_args(argv)
    return {"record": cmd_record, "atlas-run": cmd_atlas_run, "plan": cmd_plan, "analyse": cmd_analyse, "wind-side": cmd_wind_side}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
