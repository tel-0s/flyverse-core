"""run:vnc-drive -- the VNC afferent test (docs/audits/vnc_drive.md).

Does driving the proprioceptive afferents from the body state the VNC motor neurons produce (the opt-in
senses.Proprioception transducer, docs/audits/proprioception_transducer.md) reach the ascending cells and beyond, change
the plain fly's steering, and give the compass loop a report of a self-turn?  Three GPU protocols, one CPU analysis, all
on the shipped model (nothing in flyverse/ is edited; every arm is the shipped default with or without the sense
attached):

    room     (GPU)  ONE run of scripts/probe_walk_straightness.py's plain-fly protocol -- BatchSim 16 flies x 60 s, program
                    none, no fence, the full table, the shipped default weights -- under one arm:
                      A  shipped (no sense)            B  proprioception 'all' (default laws)
                      C  leg channels only ('chordotonal,hair_plate,campaniform')     D  the haltere channel only
                      E  'all+haltere_coriolis' -- the LABELLED STOP-GAP control arm (body.Locomotion's yaw scalar fed back)
                    recording per fly: yaw-rate SD, straightness, time to leave the table top, DNa02 L-R, leg L-R, the
                    per-frame rates of the watch types (AN04B003, AN07B035, AN07B037_a, AN06A026, PS196_b, LAL139, GLNO,
                    IN12B014, IN19A003, DNa02, by side) and of the afferent channels (flyverse.interp.common.Recorder), the
                    commanded afferent Hz read back from FlyBrain._base_poisson, plus the per-fly window-mean rate of EVERY
                    cell (a Recording whose frames are flies: the decompose / trace input) and the per-cell max (paths'
                    never_firing flag).
    compass  (GPU)  the efferent rotation arm of scripts/interp_apply_rotation.py (`record --mode efferent`: pinned single
                    fly, compass gains gE 2 / gD 15, rest / ccw / rest2 / cw, DNa02 of one side pulsed at 20 Hz, the body
                    integrates the turn) under arms A, B and E -- scripts/cx_shift.py has no efferent mode; the closest
                    is interp_apply_rotation's, whose helpers (build_sim, bump metrics, phases) are imported unchanged.
                    Per phase: the per-cell time-means of every cell (trace.ArmAccumulator), a per-frame Recorder of the
                    chain cells and afferents, the bump track, the realised heading, the commanded afferent Hz.
    bench    (GPU)  scripts/benchmark.py sections rest / taste / smell / walk under arm A (unchanged) or B: the legacy
                    sections build a bare brain.Brain, so arm B installs a Brain subclass that feeds the transducer's laws
                    from the Brain's OWN leg / haltere motor-neuron readout every 10 ms frame (on the ground, yaw 0) -- the
                    same variable BatchSim feeds it, without a body.  Nothing in benchmark.py is edited.
    analyse  (CPU)  the tables of the audit: per-arm behaviour and rates with the run scatter and common.compare verdicts
                    against arm A; trace from the afferent channels to DNa02 / PS196_b / GLNO (stimulus = B, control = A,
                    null = control pairs); decompose of DNa02_L / DNa02_R by (type, side) under every arm against A; paths
                    from the afferents to DNa02 / PS196_b / GLNO with never_firing from the A and B rollouts; the compass
                    flip table, chain rates and bump drift per arm; the benchmark checks A vs B.
    plan     (CPU)  writes the ONE cluster submission (out/vncd/batch.sh).

    PYTHONIOENCODING=utf-8 python scripts/probe_vnc_drive.py plan --runs 5 --compass-seeds 3 --draws 2
    bash out/vncd/batch.sh                       # one cluster_run.py call, --fetch out/vncd/
    PYTHONIOENCODING=utf-8 python scripts/probe_vnc_drive.py analyse --dir out/vncd --out out/vncd/analysis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if any(sys.argv[i] == "--device" and sys.argv[i + 1].startswith("cpu") for i in range(len(sys.argv) - 1)):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # a CPU smoke never touches this desktop's GPU (the cluster rule); before torch is imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flyverse.interp import common  # noqa: E402
from flyverse.interp.common import Recording  # noqa: E402

ARMS = {"A": None, "B": "all", "C": "chordotonal,hair_plate,campaniform", "D": "haltere", "E": "all+haltere_coriolis"}
ARM_LABEL = {"A": "shipped (sense off)", "B": "proprioception all", "C": "leg channels only", "D": "haltere only",
             "E": "all + haltere_coriolis (labelled stop-gap control arm)"}
WATCH = ["AN04B003", "AN07B035", "AN07B037_a", "AN06A026", "PS196_b", "LAL139", "GLNO", "IN12B014", "IN19A003", "DNa02"]
CHAIN = WATCH + ["PS047_b", "PS239", "LAL184", "WED040_a", "PEN_a(PEN1)", "PEN_b(PEN2)", "EPG", "Delta7", "PEG", "HSN", "Nod1", "PLP078", "LPsP"]
CHANNELS = ("chordotonal", "hair_plate", "campaniform", "haltere")
SUBSAMPLE = 12                       # afferent cells recorded per frame per channel (the channel mean comes from the window accumulator)
MOTOR_FIELDS = ("turn_L", "turn_R", "leg_L", "leg_R", "fwd_dn", "back_dn", "power", "haltere", "gf", "steer_L", "steer_R")


def _log(msg):
    print(msg, flush=True)


def spec_of(arm):
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {sorted(ARMS)}")
    return ARMS[arm]


def channel_subsample(sense):
    """Up to SUBSAMPLE cells per channel, spread over the channel's cell list (recorded every frame)."""
    out = {}
    for ch in sense.channels:
        idx = np.asarray(sense.idx[ch])
        step = max(1, len(idx) // SUBSAMPLE)
        out[ch] = idx[::step][:SUBSAMPLE]
    return out


def flies_recording(c, rate_flies, meta):
    """A common.Recording whose frames are flies (docs/audits/deficit_turning.md 2): rate_hz[b] = fly b's window-mean
    rate of every cell; t_ms = 1000 b labels the fly. decompose / trace read the window mean as the mean over flies."""
    n = c.neurons
    B = rate_flies.shape[0]
    return Recording(np.arange(B, dtype=np.float64) * 1000.0, np.arange(c.n), n.bodyId.to_numpy(), n.type.fillna("").to_numpy().astype(str),
                     {"rate_hz": rate_flies.astype(np.float32)}, {}, dict(meta, frames_are="flies"))


def max_recording(c, rate_max, meta):
    n = c.neurons
    return Recording(np.zeros(1), np.arange(c.n), n.bodyId.to_numpy(), n.type.fillna("").to_numpy().astype(str),
                     {"rate_hz": np.asarray(rate_max, np.float32)[None]}, {}, dict(meta, frames_are="max over flies and frames"))


# ---------------------------------------------------------------------------------------------- room (GPU)
def cmd_room(args) -> int:
    import torch
    from flyverse import senses, world
    from flyverse.batch_sim import BatchSim
    from flyverse.interp import trace as tr
    gpu = not (args.device and str(args.device).startswith("cpu"))
    if gpu:
        assert torch.cuda.is_available(), "no CUDA device (run this on the cluster; --device cpu for a synthetic smoke)"
    t0 = time.time()
    spec = spec_of(args.arm)
    B = int(args.batch)
    seeds = list(range(args.seed * 100, args.seed * 100 + B))
    # scripts/probe_walk_straightness.py's protocol: BatchSim default start (-0.5, 0.05, table top), heading 5 deg, program
    # none, fruit all, NO fence, default energy; the only addition is the opt-in sense of the arm
    sim = BatchSim(B, seed=args.seed, seeds=seeds, program="none", fruit_set="all", fence=False,
                   cuda_graphs=gpu, cuda_kernels=True if gpu else None, event_driven=True if gpu else None,
                   cuda_sparse=args.cuda_sparse, device=args.device, proprioception=spec)
    c, fb, brain = sim.fb.c, sim.fb, sim.fb.brain
    lp, op = brain.p, (sim.optic.p if sim.optic is not None else None)
    sense = fb.proprioception_sense if spec is not None else senses.Proprioception(c, "all")   # A: the same cells, recorded
    counts = sense.counts()
    info = world.make_room(0, "all")[1]; top_z = float(info["table_top_z"]); x0, x1, y0, y1 = [float(v) for v in info["table_extent"]]
    _log(f"[room {args.arm}] spec {spec or 'off'} channels {sense.channels} coriolis {sense.haltere_coriolis} params {sense.params} mn_ref {sense.mn_ref_hz}")
    _log(f"[room {args.arm}] BatchSim B={B} neurons={c.n:,} device={brain.device} env seeds={seeds} brain seed={args.seed} receptor_model {lp.receptor_model} "
         f"type_path_gain {lp.type_path_gain} cuda kernels {brain.cuda} event_driven {brain.event_driven} cuda_graphs {fb.cuda_graphs} fence False")
    n = c.neurons; ty = n.type.fillna("").to_numpy(); side = n.somaSide.fillna("?").to_numpy()
    watch_idx = {t: c.select(type=t) for t in WATCH}
    sub = channel_subsample(sense)
    all_chan = {ch: np.asarray(sense.idx[ch]) for ch in sense.channels}
    rec_idx = np.unique(np.concatenate(list(watch_idx.values()) + list(sub.values())))
    rec = common.Recorder(c, rec_idx, quantities=("rate_hz",))
    n_frames = int(round(args.seconds * 100)); every = int(args.every); mean_every = int(args.mean_every)
    skip_f = int(round(args.skip * 100))
    T = (n_frames + every - 1) // every
    heading = np.zeros((n_frames, B), np.float32); pos = np.zeros((n_frames, B, 3), np.float32)
    airborne = np.zeros((n_frames, B), bool); on_top = np.zeros((n_frames, B), bool)
    yaw_cmd = np.zeros((n_frames, B), np.float32); speed_cmd = np.zeros((n_frames, B), np.float32)
    rates_cmd = {k: np.zeros((n_frames, B), np.float32) for k in ("DNa02_L", "DNa02_R", "legMN_L", "legMN_R")}
    motor = {k: np.zeros((n_frames, B), np.float32) for k in MOTOR_FIELDS}
    commanded = {ch: np.zeros((n_frames, B), np.float32) for ch in sense.channels}
    chan_t = {ch: brain._idx(idx) for ch, idx in all_chan.items()} if spec is not None else {}
    chan_pos_t = {ch: brain._idx(idx) for ch, idx in all_chan.items()}
    rate_max = torch.zeros(c.n, dtype=torch.float32, device=fb.device)
    chan_sum = {ch: torch.zeros(B, dtype=torch.float64, device=fb.device) for ch in all_chan}
    counts0 = None; n_mean = 0
    for k in range(n_frames):
        sim.step()
        if k == skip_f:
            counts0 = brain.spike_counts.detach().clone()
        if k % every == 0:
            rec.capture(fb, t_ms=10.0 * k, motor=sim.motor)
        if k >= skip_f and k % mean_every == 0:
            r = brain.rate
            rate_max = torch.maximum(rate_max, r.max(dim=0).values); n_mean += 1
            for ch, ti in chan_pos_t.items():
                chan_sum[ch] += r[:, ti].mean(dim=1).double()
        if chan_t:
            base = fb._base_poisson
            cmdv = torch.stack([base[:, ti].mean(dim=1) for ti in chan_t.values()], 1).cpu().numpy() * (1000.0 / brain.p.dt)
            for j, ch in enumerate(chan_t):
                commanded[ch][k] = cmdv[:, j]
        m = sim.motor
        for name in MOTOR_FIELDS:
            motor[name][k] = np.broadcast_to(np.asarray(getattr(m, name, 0.0), np.float32), (B,))
        for i, f in enumerate(sim.flies):
            cmd = sim.commands[i]
            heading[k, i] = f.heading; pos[k, i] = f.pos; airborne[k, i] = bool(f.airborne)
            on_top[k, i] = (x0 - 1e-3 <= f.pos[0] <= x1 + 1e-3) and (y0 - 1e-3 <= f.pos[1] <= y1 + 1e-3) and abs(f.pos[2] - top_z) < 0.01
            yaw_cmd[k, i] = cmd["yaw"]; speed_cmd[k, i] = cmd["speed"]
            for key in rates_cmd:
                rates_cmd[key][k, i] = cmd["rates"][key]
        if k % 1000 == 999:
            _log(f"  t {(k + 1) / 100:5.0f} s  on table {on_top[k].mean():.2f}  airborne {airborne[k].mean():.2f}  DNa02 L-R {float((motor['turn_L'][k] - motor['turn_R'][k]).mean()):+.3f} Hz  "
                 f"leg L-R {float((motor['leg_L'][k] - motor['leg_R'][k]).mean()):+.3f} Hz  wall {time.time() - t0:.0f} s")
    window_s = (n_frames - skip_f) / 100.0
    if counts0 is None:
        counts0 = torch.zeros_like(brain.spike_counts); window_s = n_frames / 100.0
    rate_flies = ((brain.spike_counts - counts0).double() / max(window_s, 1e-9)).float().cpu().numpy()           # (B, N)
    rate_max_np = rate_max.cpu().numpy()
    chan_measured = {ch: (v / max(n_mean, 1)).cpu().numpy() for ch, v in chan_sum.items()}                       # (B,) per channel
    wall = time.time() - t0
    stim = {"protocol": "room_plain_walk_straightness", "arm": args.arm, "params": {"proprioception": spec, "channels": list(sense.channels),
            "haltere_coriolis": sense.haltere_coriolis, "rate_params": sense.params, "mn_ref_hz": sense.mn_ref_hz, "counts": counts,
            "batch": B, "seconds": args.seconds, "skip_s": args.skip, "program": "none", "fruit_set": "all", "fence": False,
            "start": "BatchSim default (-0.5, 0.05, table top), heading 5 deg", "energy": "Metabolism default", "wind": "0.3 m/s towards 180 deg (BatchSim default)",
            "capture_every_frames": every, "mean_every_frames": mean_every},
            "control": {"arm": "A", "proprioception": "off (the shipped path)"}}
    prov = common.provenance(c, lp, op, fb=fb, device=args.device, seeds=[args.seed], env_seeds=seeds, batch=B,
                             backend={"cuda_kernels": bool(brain.cuda), "event_driven": bool(brain.event_driven), "cuda_graphs": bool(fb.cuda_graphs), "cuda_sparse": args.cuda_sparse},
                             stimulus=stim)
    recording = rec.finish(meta={"protocol": "room_plain_walk_straightness", "arm": args.arm, "proprioception": spec, "seed": args.seed, "env_seeds": seeds,
                                 "watch": WATCH, "channel_subsample": {ch: [int(i) for i in v] for ch, v in sub.items()}, "every": every,
                                 "provenance": prov, "generator": "scripts/probe_vnc_drive.py room " + " ".join(sys.argv[2:])})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta_f = {"protocol": "room_plain_walk_straightness", "arm": "as-given", "vnc_arm": args.arm, "proprioception": spec, "seed": args.seed, "env_seeds": seeds,
              "window_s": [args.skip, args.seconds], "default_quantity": {"spiking": "rate_hz"}, "provenance": prov,
              "run_id": f"vncd-room-{args.arm}-r{args.seed}", "generator": "scripts/probe_vnc_drive.py room " + " ".join(sys.argv[2:])}
    tr.save_recording(flies_recording(c, rate_flies, meta_f), Path(str(out) + "_flies"))
    tr.save_recording(max_recording(c, rate_max_np, meta_f), Path(str(out) + "_max"))
    recording.save(Path(str(out) + "_rec"))
    np.savez_compressed(str(out) + "_body.npz", heading=heading, pos=pos, airborne=airborne, on_top=on_top, yaw_cmd=yaw_cmd, speed_cmd=speed_cmd,
                        **{f"cmd__{k}": v for k, v in rates_cmd.items()}, **{f"m__{k}": v for k, v in motor.items()},
                        **{f"commanded__{ch}": v for ch, v in commanded.items()}, env_seeds=np.asarray(seeds))
    summary = summarise_room(c, recording, heading, pos, airborne, on_top, yaw_cmd, speed_cmd, rates_cmd, motor, commanded, chan_measured, all_chan, sub, watch_idx, skip_f, every)
    summary.update(arm=args.arm, arm_label=ARM_LABEL[args.arm], proprioception=spec, seed=args.seed, env_seeds=seeds, batch=B, seconds=args.seconds, skip_s=args.skip,
                   wall_s=round(wall, 1), counts=counts, rate_params=sense.params, mn_ref_hz=sense.mn_ref_hz, channel_names=list(sense.channels),
                   haltere_coriolis=sense.haltere_coriolis, device=prov["execution"]["device"], device_name=prov["execution"].get("device_name"),
                   provenance=prov, lif={"receptor_model": lp.receptor_model, "receptor_net_rule": lp.receptor_net_rule, "type_path_gain": lp.type_path_gain},
                   files={"generator": "scripts/probe_vnc_drive.py room " + " ".join(sys.argv[2:]), "recording": str(out) + "_rec.npz", "flies": str(out) + "_flies.npz",
                          "max": str(out) + "_max.npz", "body": str(out) + "_body.npz"})
    Path(str(out) + ".json").write_text(json.dumps(common.to_jsonable(summary), indent=1), encoding="utf-8")
    s = summary["run"]
    _log(f"[room {args.arm} seed {args.seed}] device {summary['device']} ({summary['device_name']}), {wall:.0f} s wall; yaw SD {s['yaw_sd_deg_s']:.2f} deg/s (median {s['yaw_sd_deg_s_median']:.2f}), "
         f"straightness {s['straightness']:.3f}, left the table {s['n_left_table']}/{B} (median {s['left_table_s_median']}), DNa02 L-R {s['DNa02_LR_hz']:+.3f} Hz, "
         f"leg L-R {s['leg_LR_hz']:+.3f} Hz, power max {s['power_max_hz']:.1f} Hz, airborne {s['airborne_frac']:.3f}")
    _log("  channel      n   commanded Hz   measured Hz")
    for ch, v in summary["channels"].items():
        _log(f"  {ch:12s} {v['n']:4d}  {v['commanded_hz']:10.2f}   {v['measured_hz']:8.2f}")
    _log("  " + "; ".join(f"{k} {v['hz']:.3f}" for k, v in summary["watch"].items()))
    _log(f"wrote {out}.json")
    if summary["device"] is None or "cpu" in str(summary["device"]):
        _log("WARNING: device cpu -- resubmit (the JSON records the realised device)")
    return 0


def summarise_room(c, recording, heading, pos, airborne, on_top, yaw_cmd, speed_cmd, rates_cmd, motor, commanded, chan_measured, all_chan, sub, watch_idx, skip_f, every):
    n_frames, B = heading.shape
    side = c.neurons.somaSide.fillna("?").to_numpy()
    dh = np.diff(np.unwrap(heading.astype(np.float64), axis=0), axis=0) * 100.0                # rad/s per frame
    walking = ~airborne[1:]
    win = np.zeros(n_frames, bool); win[skip_f:] = True
    rate = recording.quantities["rate_hz"]                      # (T, B, n)
    t_ms = recording.t_ms; wmask = t_ms >= skip_f * 10.0
    pos_of = {int(i): j for j, i in enumerate(recording.idx)}
    rows = []
    for i in range(B):
        w_all = walking[:, i]; w_win = walking[:, i] & win[1:]
        path = float(np.linalg.norm(np.diff(pos[:, i, :2], axis=0), axis=1).sum()); net = float(np.linalg.norm(pos[-1, i, :2] - pos[0, i, :2]))
        left = np.flatnonzero(~on_top[:, i])
        ok = win & ~airborne[:, i]
        d = rates_cmd
        row = {"row": i, "yaw_sd_deg_s": float(np.degrees(np.std(dh[w_win, i]))) if w_win.any() else None,
               "yaw_sd_all_deg_s": float(np.degrees(np.std(dh[w_all, i]))) if w_all.any() else None,
               "yaw_mean_abs_deg_s": float(np.degrees(np.mean(np.abs(dh[w_win, i])))) if w_win.any() else None,
               "yaw_cmd_sd_deg_s": float(np.degrees(np.std(yaw_cmd[ok, i]))) if ok.any() else None,
               "yaw_cmd_mean_deg_s": float(np.degrees(np.mean(yaw_cmd[ok, i]))) if ok.any() else None,
               "speed_cmd_mean_m_s": float(np.mean(speed_cmd[ok, i])) if ok.any() else None,
               "path_m": path, "net_m": net, "straightness": net / path if path > 0 else None,
               "left_table_s": float(left[0] / 100.0) if len(left) else None, "frac_on_table": float(on_top[:, i].mean()),
               "hops": int(np.sum(np.diff(airborne[:, i].astype(int)) == 1)), "airborne_frac": float(airborne[win, i].mean()),
               "DNa02_L_hz": float(d["DNa02_L"][ok, i].mean()) if ok.any() else None, "DNa02_R_hz": float(d["DNa02_R"][ok, i].mean()) if ok.any() else None,
               "DNa02_LR_hz": float((d["DNa02_L"][ok, i] - d["DNa02_R"][ok, i]).mean()) if ok.any() else None,
               "DNa02_LR_sd_hz": float((d["DNa02_L"][ok, i] - d["DNa02_R"][ok, i]).std()) if ok.any() else None,
               "DNa02_abs_LR_hz": float(np.abs(d["DNa02_L"][ok, i] - d["DNa02_R"][ok, i]).mean()) if ok.any() else None,
               "leg_L_hz": float(d["legMN_L"][ok, i].mean()) if ok.any() else None, "leg_R_hz": float(d["legMN_R"][ok, i].mean()) if ok.any() else None,
               "leg_LR_hz": float((d["legMN_L"][ok, i] - d["legMN_R"][ok, i]).mean()) if ok.any() else None,
               "leg_LR_sd_hz": float((d["legMN_L"][ok, i] - d["legMN_R"][ok, i]).std()) if ok.any() else None,
               "leg_abs_LR_hz": float(np.abs(d["legMN_L"][ok, i] - d["legMN_R"][ok, i]).mean()) if ok.any() else None,
               "power_mean_hz": float(motor["power"][win, i].mean()), "power_max_hz": float(motor["power"][:, i].max()),
               "power_sustained_hz": float(np.max(np.convolve(motor["power"][:, i], np.ones(30) / 30, mode="valid"))) if n_frames >= 30 else float(motor["power"][:, i].max()),
               "haltere_mn_hz": float(motor["haltere"][win, i].mean()), "fwd_dn_hz": float(motor["fwd_dn"][win, i].mean()), "gf_max_hz": float(motor["gf"][:, i].max())}
        for ch in all_chan:
            row[f"commanded_{ch}_hz"] = float(commanded[ch][win, i].mean()) if ch in commanded else 0.0
            row[f"measured_{ch}_hz"] = float(chan_measured[ch][i])
        for t, idx in watch_idx.items():
            for s in ("L", "R"):
                sel = idx[side[idx] == s]
                if len(sel):
                    cols = [pos_of[int(j)] for j in sel]
                    row[f"{t}_{s}_hz"] = float(rate[wmask][:, i, cols].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    run = {}
    for k in df.columns:
        if k == "row":
            continue
        v = df[k].dropna().to_numpy(dtype=float)
        run[k] = float(v.mean()) if len(v) else None
        run[k + "_fly_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
    run["yaw_sd_deg_s_median"] = float(df.yaw_sd_deg_s.dropna().median()) if df.yaw_sd_deg_s.notna().any() else None
    run["n_left_table"] = int(df.left_table_s.notna().sum())
    run["left_table_s_median"] = float(df.left_table_s.dropna().median()) if df.left_table_s.notna().any() else None
    run["left_table_s_mean_capped"] = float(df.left_table_s.fillna(n_frames / 100.0).mean())      # flies that never left count the full run
    channels = {ch: {"n": int(len(all_chan[ch])), "n_recorded": int(len(sub.get(ch, []))), "commanded_hz": run.get(f"commanded_{ch}_hz"),
                     "measured_hz": run.get(f"measured_{ch}_hz")} for ch in all_chan}
    watch = {f"{t}_{s}": {"hz": run.get(f"{t}_{s}_hz"), "fly_sd": run.get(f"{t}_{s}_hz_fly_sd"), "n_cells": int((side[watch_idx[t]] == s).sum())}
             for t in watch_idx for s in ("L", "R") if f"{t}_{s}_hz" in run}
    return {"window_skip_s": skip_f / 100.0, "frames": int(n_frames), "captured_every": every, "run": run, "rows": common.to_jsonable(df.to_dict("records")),
            "channels": channels, "watch": watch}


# ---------------------------------------------------------------------------------------------- compass (GPU)
def cmd_compass(args) -> int:
    cpu = bool(args.device and str(args.device).startswith("cpu"))
    import torch
    if not cpu:
        assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    import pygame
    pygame.init(); pygame.display.set_mode((64, 64))
    from flyverse import senses
    from flyverse.interp import trace as tr
    import cx_wedge
    import interp_apply_rotation as rot
    t0 = time.time()
    spec = spec_of(args.arm)
    if args.arm not in ("A", "B", "E"):
        _log("note: the compass protocol was specified for arms A, B and E; running the requested arm anyway")
    seed = int(args.seed)
    gains = rot.parse_gains(args.gains)
    c, cdir = rot.load_condition_connectome("default", args.cache_dir, verbose=False)
    sim = rot.build_sim(c, gains, seed, args.device, cpu, sparse=args.sparse)
    fb = sim.fb
    if spec is not None:
        fb.proprioception_sense = senses.Proprioception(c, spec)
    sense = fb.proprioception_sense if spec is not None else senses.Proprioception(c, "all")
    _log(f"[compass {args.arm}] spec {spec or 'off'} channels {sense.channels} coriolis {sense.haltere_coriolis}; available senses {fb.available_senses}")
    seconds = 2.0 if args.quick else float(args.seconds)
    skip = int(round((0.5 if args.quick else float(args.skip)) * 100)); n_frames = int(round(seconds * 100))
    rate_dps = float(args.rate)
    cells = cx_wedge.compass_cells(c); epg = cells["EPG"]; idx_epg = epg["idx"]
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, list(rot.PULSE_WEDGES))
    if gains:
        fb.stimulate(idx_epg, rot.BACKGROUND_HZ, 4 * seconds * 1000 + 1000)
        fb.stimulate(idx_epg[inside], rot.PULSE_HZ, rot.PULSE_S * 1000 if not args.quick else 200.0)
    dna02 = {s: c.select(type="DNa02", somaSide=s) for s in ("L", "R")}
    sub = channel_subsample(sense)
    chain_idx = np.unique(np.concatenate([c.select(type=t) for t in CHAIN if len(c.select(type=t))] + list(sub.values())))
    all_chan = {ch: np.asarray(sense.idx[ch]) for ch in sense.channels}
    chan_t = {ch: fb.brain._idx(idx) for ch, idx in all_chan.items()} if spec is not None else {}
    lif_p = fb.brain.p; optic_p = fb.optic.p if fb.optic is not None else None
    stim_common = {"protocol": "rotation", "mode": "efferent", "condition": "default", "arm": args.arm, "gains": list(gains) if gains else None,
                   "params": {"pos": list(rot.POS), "rate_dps": rate_dps, "seconds": seconds, "skip_s": skip / 100.0, "phases": [p for p, _ in rot.PHASES],
                              "wind_speed": 0.0, "background_hz": rot.BACKGROUND_HZ if gains else 0.0, "pulse_hz": rot.PULSE_HZ if gains else 0.0,
                              "pulse_s": rot.PULSE_S, "pulse_wedges": list(rot.PULSE_WEDGES), "dna02_hz": float(args.dna02_hz),
                              "compass_adaptation": 0.0 if gains else None, "sparse": args.sparse, "proprioception": spec,
                              "channels": list(sense.channels), "haltere_coriolis": sense.haltere_coriolis, "rate_params": sense.params, "mn_ref_hz": sense.mn_ref_hz},
                   "control": "the rest phases of the same run (rest = control, rest2 = control again = the null); arm A = the sense off"}
    prov = common.provenance(c, lif_p, optic_p, fb=fb, device=args.device, seeds=[seed], env_seeds=[seed], batch=1, stimulus=stim_common,
                             retina=tr.retina_record(fb.retina, c), cache_dir=str(cdir) if cdir else args.cache_dir)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    summary_phases = {}; written = []
    motor = None
    for name, sgn in rot.PHASES:
        t1 = time.time()
        acc = tr.ArmAccumulator(fb, keep_series=False)
        rec = common.Recorder(c, chain_idx, quantities=("rate_hz",))
        track, headings, yaws = [], [], []
        commanded = {ch: np.zeros(n_frames, np.float32) for ch in all_chan}
        airborne = 0
        if sgn != 0.0:
            fb.stimulate(dna02["L" if sgn > 0 else "R"], float(args.dna02_hz), seconds * 1000)
        for k in range(n_frames):
            sim.fly.place(*rot.POS, heading=sim.fly.heading)          # efferent: position pinned, heading integrated by the body
            if spec is not None:
                if motor is None:
                    fb.proprioception(0.0, 0.0, 0.0, False, 0.0)                  # tonic / load only before the first readout (BatchSim's rule)
                else:
                    fb.proprioception(float(motor.leg_L), float(motor.leg_R), float(motor.haltere), bool(sim.fly.airborne), float(sim.fly.yaw_rate))
            sim.step()
            motor = fb.motor()
            acc.add(fb, k, skip)
            rec.capture(fb, t_ms=10.0 * k, motor=motor)
            if chan_t:
                base = fb._base_poisson
                for ch, ti in chan_t.items():
                    commanded[ch][k] = float(base[0, ti].mean()) * (1000.0 / fb.brain.p.dt)
            r = fb.brain.rate_np(); re_ = r[idx_epg]
            prof = np.array([float(re_[wedge_of == w].mean()) for w in range(16)])
            cc, vs = rot.circ_centre(prof)
            track.append((cc, vs, float(prof.max())))
            headings.append(float(sim.fly.heading)); yaws.append(float(sim.fly.yaw_rate)); airborne += int(bool(sim.fly.airborne))
        bump = rot.bump_metrics(track, skip); head = rot.heading_metrics(headings, skip)
        meta = {"protocol": "rotation", "arm": name, "phase": name, "rotation_sign": sgn, "mode": "efferent", "condition": "default", "vnc_arm": args.arm,
                "proprioception": spec, "gains": list(gains) if gains else None, "seed": seed, "window_s": [skip / 100.0, n_frames / 100.0],
                "stimulus": dict(stim_common, phase=name, rotation_sign=sgn), "default_quantity": {"spiking": "rate_hz"}, "provenance": prov,
                "run_id": f"vncd-compass-{args.arm}-{name}-r{seed}", "bump": bump, "heading": head, "airborne_frames": airborne,
                "yaw_rate_abs_mean_rad_s": float(np.mean(np.abs(yaws[skip:]))) if len(yaws) > skip else float("nan"),
                "commanded_hz": {ch: float(v[skip:].mean()) for ch, v in commanded.items()},
                "generator": "scripts/probe_vnc_drive.py compass " + " ".join(sys.argv[2:])}
        cells_rec, _ = acc.finish(meta)
        stem = Path(f"{out}_{name}")
        tr.save_recording(cells_rec, stem); written.append(str(stem.with_suffix(".npz")))
        R = rec.finish(dict(meta, channel_subsample={ch: [int(i) for i in v] for ch, v in sub.items()}, chain=CHAIN))
        R.motor = dict(R.motor, **{f"commanded.{ch}": v for ch, v in commanded.items()}, yaw_rate=np.asarray(yaws, np.float32), heading=np.asarray(headings, np.float32))
        tr.save_recording(R, Path(f"{out}_{name}_rec")); written.append(str(Path(f"{out}_{name}_rec").with_suffix(".npz")))
        summary_phases[name] = {"bump": bump, "heading": head, "airborne_frames": airborne, "commanded_hz": meta["commanded_hz"],
                                "yaw_rate_abs_mean_rad_s": meta["yaw_rate_abs_mean_rad_s"], "wall_s": round(time.time() - t1, 1)}
        _log(f"  {name} ({('DNa02_%s %.0f Hz' % ('L' if sgn > 0 else 'R', args.dna02_hz)) if sgn else 'no stimulus'}, {seconds} s): {time.time() - t1:.0f} s wall; "
             f"bump vs {bump['vs']:.2f} peak {bump['peak']:.0f} Hz drift {bump['drift_wedges_per_s']:+.3f} w/s; heading {head['rate_dps']:+.1f} deg/s; "
             f"|yaw| {meta['yaw_rate_abs_mean_rad_s']:.2f} rad/s; commanded " + ", ".join(f"{ch} {v:.1f}" for ch, v in meta["commanded_hz"].items()))
    dev = prov["execution"]["device"]
    with open(f"{out}_run.json", "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable({"arm": args.arm, "arm_label": ARM_LABEL[args.arm], "proprioception": spec, "mode": "efferent", "condition": "default", "seed": seed,
                                      "gains": gains, "phases": summary_phases, "device": dev, "device_name": prov["execution"].get("device_name"),
                                      "counts": sense.counts(), "rate_params": sense.params, "files": written, "provenance": prov,
                                      "wall_s": round(time.time() - t0, 1), "generator": "scripts/probe_vnc_drive.py compass " + " ".join(sys.argv[2:])}), f, indent=1)
    _log(f"[compass {args.arm} seed {seed}] device {dev} ({prov['execution'].get('device_name')}), {time.time() - t0:.0f} s wall; written {out}_<phase>.npz / _rec.npz and {out}_run.json")
    if dev is None or "cpu" in str(dev):
        _log("WARNING: device cpu -- resubmit (the JSON records the realised device)")
    return 0


# ---------------------------------------------------------------------------------------------- bench (GPU)
def install_bench_brain(spec: str):
    """Arm B of the benchmark: brain.Brain -> a subclass that, every 10 ms frame, feeds senses.Proprioception's laws from
    the Brain's own leg / haltere MN readout (motor.read_motor; on the ground, yaw 0) through Brain.set_poisson -- the
    body-less analogue of BatchSim's per-frame call (the previous frame's MotorRates; zeros before the first frame).
    Arm A leaves brain.Brain untouched."""
    from flyverse import brain as brain_mod, motor as motor_mod, senses
    Base = brain_mod.Brain
    log = {"frames": 0, "commanded": {}}

    class ProprioBrain(Base):
        def __init__(self, c, params=None, *a, **kw):
            super().__init__(c, params, *a, **kw)
            self._pp_sense = senses.Proprioception(c, spec)
            self._pp_groups = motor_mod.motor_groups(c); self._pp_wings = motor_mod.wing_groups(c)
            self._pp_frame_steps = max(1, int(round(10.0 / self.p.dt)))
            self._pp_motor = None

        def _pp_apply(self):
            m = self._pp_motor
            lL, lR, h = (0.0, 0.0, 0.0) if m is None else (m.leg_L, m.leg_R, m.haltere)
            for ch, idx, hz in self._pp_sense.rates(lL, lR, h, False, 0.0, self.B):
                self.set_poisson(idx, hz)
                log["commanded"].setdefault(ch, []).append(float(np.mean(hz)))
            log["frames"] += 1

        def step(self, n_steps: int = 1) -> None:
            left = int(n_steps)
            while left > 0:
                self._pp_apply()
                k = min(left, self._pp_frame_steps)
                super().step(k)
                left -= k
                self._pp_motor = motor_mod.read_motor(self, self._pp_groups, self._pp_wings)

    brain_mod.Brain = ProprioBrain
    return log


def cmd_bench(args) -> int:
    import torch
    if not (args.device and str(args.device).startswith("cpu")):
        assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    import benchmark
    spec = spec_of(args.arm)
    log = install_bench_brain(spec) if spec is not None else None
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    argv = ["benchmark.py", "--sections", args.sections, "--json", str(out)] + (["--fast"] if args.fast else []) + (["--eager"] if args.eager else [])
    _log(f"[bench {args.arm}] spec {spec or 'off'}; benchmark argv {argv[1:]}")
    sys.argv = argv
    benchmark.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    d["vnc_arm"] = args.arm; d["arm_label"] = ARM_LABEL[args.arm]; d["proprioception"] = spec
    d["proprioception_frames"] = None if log is None else log["frames"]
    d["proprioception_commanded_hz_mean"] = None if log is None else {ch: float(np.mean(v)) for ch, v in log["commanded"].items()}
    d["generator"] = f"scripts/probe_vnc_drive.py bench --arm {args.arm} --sections {args.sections} --out {out}"
    out.write_text(json.dumps(d, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    _log(f"[bench {args.arm}] device {d['config'].get('device')}; checks " + "; ".join(f"{c_['key']} {c_['measured']} {c_['status']}" for c_ in d["checks"])
         + (f"; transducer frames {log['frames']}, commanded " + ", ".join(f"{k} {v:.1f}" for k, v in d["proprioception_commanded_hz_mean"].items()) if log else ""))
    return 0


# ---------------------------------------------------------------------------------------------- plan (CPU)
def cmd_plan(args) -> int:
    d = args.dir.rstrip("/")
    pre = f"mkdir -p {d} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
    cmds = []
    for arm in "ABCDE":
        for s in range(args.runs):
            stem = f"{d}/room_{arm}_r{s}"
            cmds.append(f"{pre}python scripts/probe_vnc_drive.py room --arm {arm} --seed {s} --batch 16 --seconds 60 --out {stem} > {stem}.txt 2>&1; tail -6 {stem}.txt")
    for s in range(args.compass_seeds):
        parts = [f"python scripts/probe_vnc_drive.py compass --arm {arm} --seed {s} --out {d}/compass_{arm}_r{s} > {d}/compass_{arm}_r{s}.txt 2>&1; tail -6 {d}/compass_{arm}_r{s}.txt" for arm in "ABE"]
        cmds.append(pre + "; ".join(parts))
    for arm in "AB":
        parts = [f"python scripts/probe_vnc_drive.py bench --arm {arm} --sections rest,taste,smell,walk --out {d}/bench_{arm}_d{k}.json > {d}/bench_{arm}_d{k}.txt 2>&1; tail -12 {d}/bench_{arm}_d{k}.txt" for k in range(args.draws)]
        cmds.append(pre + "; ".join(parts))
    line = f"python scripts/cluster_run.py --name vncd --minutes {args.minutes} " + " ".join('"' + c_.replace('"', '\\"') + '"' for c_ in cmds) + f" --fetch {d}/ 2>&1 | tee out/vncd_cluster.log"
    Path(d).mkdir(parents=True, exist_ok=True)
    Path(d, "batch.sh").write_text("#!/bin/bash\n# ONE submission: " + f"{len(cmds)} jobs = {5 * args.runs} room + {args.compass_seeds} compass (3 arms each, sequential) + 2 bench ({args.draws} draws each, sequential)\n" + line + "\n", encoding="utf-8")
    _log(f"wrote {d}/batch.sh ({len(cmds)} jobs)")
    return 0


# ---------------------------------------------------------------------------------------------- analyse (CPU)
ROOM_KEYS = [("yaw_sd_deg_s", "yaw-rate SD (deg/s, walking frames, 5-60 s)"), ("yaw_sd_deg_s_median", "yaw-rate SD, median over flies"),
             ("yaw_cmd_sd_deg_s", "yaw command SD (deg/s)"), ("straightness", "straightness (net / path)"),
             ("left_table_s_mean_capped", "time to leave the table top (s; never = 60)"), ("n_left_table", "flies that left the table top (of 16)"),
             ("frac_on_table", "fraction of frames on the table top"), ("hops", "take-offs per fly"), ("airborne_frac", "airborne fraction"),
             ("DNa02_LR_hz", "DNa02 L-R (Hz)"), ("DNa02_abs_LR_hz", "|DNa02 L-R| (Hz)"), ("DNa02_L_hz", "DNa02_L (Hz)"), ("DNa02_R_hz", "DNa02_R (Hz)"),
             ("leg_LR_hz", "leg MN L-R (Hz)"), ("leg_abs_LR_hz", "|leg MN L-R| (Hz)"), ("leg_L_hz", "leg MN L (Hz)"), ("leg_R_hz", "leg MN R (Hz)"),
             ("power_mean_hz", "wing power MN mean (Hz)"), ("power_max_hz", "wing power MN per-frame max (Hz; the room analogue of walk.power_max)"),
             ("power_sustained_hz", "wing power 0.3 s running-mean max (Hz)"), ("haltere_mn_hz", "haltere MN mean (Hz)"), ("fwd_dn_hz", "fwd DN mean (Hz)"), ("gf_max_hz", "GF per-frame max (Hz)")]


def load_room(d):
    runs = {}
    import re
    for p in sorted(glob.glob(os.path.join(d, "room_*_r*.json"))):
        if not re.fullmatch(r"room_[A-E]_r\d+\.json", Path(p).name):
            continue
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        runs.setdefault(j["arm"], []).append((p, j))
    return runs


def room_tables(runs, out):
    rows = []
    arms = [a for a in "ABCDE" if a in runs]
    keys = list(ROOM_KEYS)
    j0 = runs[arms[0]][0][1]
    for ch in j0["channels"]:
        keys += [(f"commanded_{ch}_hz", f"channel {ch} commanded Hz"), (f"measured_{ch}_hz", f"channel {ch} measured Hz")]
    for w in j0["watch"]:
        keys.append((f"{w}_hz", f"{w} (Hz)"))
    for key, label in keys:
        row = {"key": key, "label": label}
        vals = {}
        for a in arms:
            v = np.array([j["run"].get(key) if j["run"].get(key) is not None else np.nan for _, j in runs[a]], float)
            vals[a] = v
            row[f"{a}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            row[f"{a}_run_sd"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan
            row[f"{a}_runs"] = [round(float(x), 4) for x in v]
        for a in arms:
            if a == "A" or "A" not in vals:
                continue
            s, nl = vals[a][np.isfinite(vals[a])], vals["A"][np.isfinite(vals["A"])]
            if len(s) and len(nl):
                cmp_ = common.compare(s, nl)
                row[f"{a}_vs_A_diff"] = cmp_["diff"]; row[f"{a}_vs_A_z"] = cmp_["z"]; row[f"{a}_vs_A_p"] = cmp_["p"]; row[f"{a}_vs_A_verdict"] = cmp_["verdict"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out, "room_table.csv"), index=False)
    lines = [f"{'key':28s} " + " | ".join(f"{a:^22s}" for a in arms) + "   verdicts vs A"]
    for _, r in df.iterrows():
        cells = [f"{r[f'{a}_mean']:9.3f} +- {r[f'{a}_run_sd']:7.3f}" if np.isfinite(r[f"{a}_mean"]) else " " * 20 for a in arms]
        vs = " ".join(f"{a}:{r.get(f'{a}_vs_A_diff', np.nan):+.3f}/z{r.get(f'{a}_vs_A_z', np.nan):+.1f}/p{r.get(f'{a}_vs_A_p', np.nan):.3f} {r.get(f'{a}_vs_A_verdict', '')}" for a in arms if a != "A")
        lines.append(f"{r['key']:28s} " + " | ".join(cells) + "   " + vs)
    text = "\n".join(lines)
    Path(out, "room_table.txt").write_text(text, encoding="utf-8")
    print(text)
    return df


def load_flies(runs, arm, kind="flies"):
    recs = []
    for p, j in runs.get(arm, []):
        stem = p[:-5]
        recs.append(Recording.load(Path(stem + f"_{kind}.npz")))
    return recs


def afferent_source(c):
    from flyverse.senses import Proprioception
    sense = Proprioception(c, "all")
    return np.unique(np.concatenate([sense.idx[ch] for ch in sense.channels])), sense


def analyse_trace(c, lp, runs, out, arms=("B", "E"), decompose_at=("DNa02", "PS196_b", "GLNO")):
    from flyverse.interp import trace as tr
    src_idx, sense = afferent_source(c)
    ctrl = load_flies(runs, "A")
    results = {}
    for arm in arms:
        stim = load_flies(runs, arm)
        if not stim or not ctrl:
            continue
        t0 = time.time()
        res = tr.trace(c, src_idx, stimulus=stim, control=ctrl, params=lp, stat="mean", min_cells=2, depth_max=6,
                       decompose_at=list(decompose_at), per_body="none", max_lost=len(decompose_at))
        res.summary["source"] = "senses.Proprioception('all').idx (the 941 afferent cells of the four channels)"
        res.summary["arms"] = {"stimulus": arm, "control": "A", "null": "control pairs (A vs A)"}
        res.files["generator"] = "scripts/probe_vnc_drive.py analyse (trace)"
        path = Path(out, f"trace_{arm}_vs_A.json"); res.save(path)
        pt = res.table("per_type")
        results[arm] = pt
        print(f"\n== trace {arm} vs A ({time.time() - t0:.0f} s): carriers per depth (verdict result) -> {path}; check {res.check()}")
        if len(pt):
            for dep in sorted(pt.depth.dropna().unique()):
                sub = pt[(pt.depth == dep) & (pt.verdict == "result")].sort_values("z", key=np.abs, ascending=False)
                print(f"  depth {int(dep)}: {len(sub)} carriers; top " + "; ".join(f"{r.type} {r.diff:+.2f} Hz z{r.z:+.1f}" for r in sub.head(12).itertuples()))
            watch = pt[pt.type.isin(CHAIN)].sort_values("depth")
            cols = [x for x in ("type", "depth", "n_cells", "stim_mean", "ctrl_level", "diff", "z", "p", "verdict") if x in watch]
            common.print_table(watch[cols], max_rows=40)
            watch[cols].to_csv(Path(out, f"trace_{arm}_vs_A_watch.csv"), index=False)
    return results


def analyse_decompose(c, lp, runs, out, arms=("B", "C", "D", "E")):
    from flyverse.interp import decompose as dc
    rec = {a: load_flies(runs, a) for a in arms if a in runs}
    null = {"A": load_flies(runs, "A")}
    if not null["A"] or not rec:
        return {}
    ew = common.effective_weights(c, lp)
    results = {}
    for tgt, spec in (("DNa02_L", "type=DNa02&somaSide=L"), ("DNa02_R", "type=DNa02&somaSide=R"), ("PS196_b", "PS196_b"), ("GLNO", "GLNO")):
        t0 = time.time()
        res = dc.decompose(c, spec, recording=rec, null_recording=null, params=lp, by=("type", "side"), tiers=False, top=30, ew=ew, keep_links=False,
                           arm_weights_override={a: lp for a in list(rec) + ["A"]})      # every arm is the shipped weights; the label is the sense arm
        res.files["generator"] = "scripts/probe_vnc_drive.py analyse (decompose)"
        path = Path(out, f"decompose_{tgt}.json"); res.save(path)
        pt = res.table("per_type")
        results[tgt] = (res, pt)
        print(f"\n== decompose {tgt} by (type, side), arms {list(rec)} vs null A ({time.time() - t0:.0f} s) -> {path}; check {res.check()}")
        dyn = res.summary.get("dynamic", {})
        for arm_, v in dyn.items():
            for t_, d_ in v.items():
                cp = d_.get("cancelling_pair", {})
                print(f"  arm {arm_} {t_}: E {d_['E_total']:+.1f} I {d_['I_total']:+.1f} net {d_['net']:+.1f} mV/s per cell; cancelling pair {cp.get('E')} {cp.get('E_value', 0):+.1f} vs "
                      f"{cp.get('I')} {cp.get('I_value', 0):+.1f}")
        cols = [x for x in pt.columns if x in ("pre_group", "sign_rule", "raw_count", "weight_mv_per_volley", "A_mean", "A_sd") or (x.endswith("_mean") and not x.startswith("A_")) or x.endswith("_z") or x.endswith("_verdict") or x.endswith("_diff")]
        if len(pt):
            key = "A_mean" if "A_mean" in pt else cols[0]
            top = pt.reindex(pt[key].abs().sort_values(ascending=False).index).head(30) if key in pt else pt.head(30)
            print("  top 30 groups by |input under A| (mV/s per post cell):")
            common.print_table(top[cols], max_rows=32)
            diffcols = [x for x in pt.columns if x.endswith("_diff")]
            if diffcols:
                mv = pt[diffcols].abs().max(axis=1)
                print("  top 20 groups by |change vs A| over the arms:")
                common.print_table(pt.reindex(mv.sort_values(ascending=False).index).head(20)[cols], max_rows=22)
            pt.to_csv(Path(out, f"decompose_{tgt}_per_type.csv"), index=False)
    return results


def analyse_paths(c, lp, runs, out, arms=("A", "B")):
    from flyverse.interp import paths as pa
    src_idx, _ = afferent_source(c)
    rows = []
    for arm in arms:
        recs = load_flies(runs, arm, kind="max")
        if not recs:
            continue
        for tgt in ("DNa02", "PS196_b", "GLNO"):
            t0 = time.time()
            res = pa.paths(c, src_idx, tgt, params=lp, k_max=3, top=20, recording=recs[0], frozen="static")
            res.files["generator"] = "scripts/probe_vnc_drive.py analyse (paths)"
            path = Path(out, f"paths_afferents_to_{tgt}_nf{arm}.json"); res.save(path)
            s = res.summary
            row = {"arm_never_firing": arm, "target": tgt, "file": str(path), "wall_s": round(time.time() - t0, 1)}
            d_ = s.get("direct") or {}
            row["direct_raw_syn"] = d_.get("raw_count"); row["direct_mv_per_post_volley"] = d_.get("mv_per_post_volley"); row["direct_never_firing_frac"] = d_.get("never_firing_frac")
            row["b_silent_input_share"] = s.get("b_silent_input_share"); row["b_sign0_input_share"] = s.get("b_sign0_input_share")
            dom = s.get("dominant_silent_input_of_b") or {}
            row["dominant_silent_input_of_b"] = f"{dom.get('pre_type')} {dom.get('silent')} share {dom.get('share_of_b_input', 0):.3f}" if dom else None
            def _k(dct, k):
                dct = dct or {}
                return dct.get(k) if k in dct else dct.get(int(k))
            for k in ("1", "2", "3"):
                w = _k(s.get("top_walk_per_k"), k)
                row[f"k{k}_top_signed"] = f"{w['path']} {w['gain']:+.4g} [{w['silent_links'] or 'live'}]" if w else None
                w = _k(s.get("top_silent_walk_per_k"), k)
                row[f"k{k}_top_silent"] = f"{w['path']} if-signed {w['gain_if_signed']:+.4g} [{w['silent_links']}]" if w else None
                l_ = _k(s.get("strongest_silent_link_per_k"), k)
                row[f"k{k}_silent_link"] = (f"{l_['pre_type']}->{l_['post_type']} {l_.get('silent')}{'/' + l_['silent_partial'] if l_.get('silent_partial') else ''} "
                                            f"{l_['mv_per_post_volley_if_signed']:+.2f} mV/volley ({l_['raw_count']:.0f} syn, nf {l_.get('never_firing_frac', 0):.2f})") if l_ else None
            pt_ = res.table("links")
            if len(pt_) and "never_firing_frac" in pt_:
                row["links_listed"] = int(len(pt_)); row["links_never_firing"] = int((pt_.never_firing_frac >= 0.999).sum())
            rows.append(row)
            print(f"\n== paths afferents -> {tgt}, never_firing from the arm-{arm} max recording ({row['wall_s']} s):")
            for k_, v_ in row.items():
                if k_ not in ("file", "wall_s", "arm_never_firing", "target"):
                    print(f"    {k_}: {v_}")
            pt = res.table("paths")
            if len(pt):
                cols = [x for x in ("k", "kind", "rank", "path", "gain", "gain_if_signed", "silent_links") if x in pt]
                common.print_table(pt[pt["rank"] <= 5][cols] if "rank" in pt else pt[cols], max_rows=36)
    df = pd.DataFrame(rows); df.to_csv(Path(out, "paths_summary.csv"), index=False)
    return df


def load_compass(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "compass_*_r*_run.json"))):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        runs.setdefault(j["arm"], []).append((p, j))
    return runs


def analyse_compass(c, runs, out):
    import interp_apply_rotation as rot
    from flyverse.interp import trace as tr
    phases = [p for p, _ in rot.PHASES]
    rows = []; flips = {}; chain = {}
    for arm, items in runs.items():
        cells = {ph: [] for ph in phases}
        for p, j in items:
            stem = p[: -len("_run.json")]
            for ph in phases:
                cells[ph].append(tr.load_run(Path(f"{stem}_{ph}.npz")))
        n_runs = len(items)
        for ph in phases:
            drift = np.array([j["phases"][ph]["bump"]["drift_wedges_per_s"] for _, j in items], float)
            head = np.array([j["phases"][ph]["heading"]["rate_dps"] for _, j in items], float)
            vs = np.array([j["phases"][ph]["bump"]["vs"] for _, j in items], float)
            yaw = np.array([j["phases"][ph].get("yaw_rate_abs_mean_rad_s", np.nan) for _, j in items], float)
            cmdh = {ch: float(np.mean([j["phases"][ph]["commanded_hz"].get(ch, 0.0) for _, j in items])) for ch in CHANNELS}
            rows.append({"arm": arm, "phase": ph, "n_runs": n_runs, "drift_w_s": float(drift.mean()), "drift_sd": float(drift.std(ddof=1)) if n_runs > 1 else np.nan,
                         "drift_runs": [round(float(x), 4) for x in drift], "heading_dps": float(head.mean()), "heading_sd": float(head.std(ddof=1)) if n_runs > 1 else np.nan,
                         "bump_vs": float(vs.mean()), "yaw_abs_rad_s": float(np.nanmean(yaw)), **{f"cmd_{ch}": v for ch, v in cmdh.items()}})
        rests = np.array([j["phases"][ph]["bump"]["drift_wedges_per_s"] for _, j in items for ph in ("rest", "rest2")], float)
        for ph in ("ccw", "cw"):
            v = np.array([j["phases"][ph]["bump"]["drift_wedges_per_s"] for _, j in items], float)
            cmp_ = common.compare(v, rests)
            rows.append({"arm": arm, "phase": f"{ph}_vs_rests", "n_runs": n_runs, "drift_w_s": cmp_["diff"], "drift_sd": np.nan, "drift_runs": [], "heading_dps": np.nan,
                         "z": cmp_["z"], "p": cmp_["p"], "verdict": cmp_["verdict"]})
        ft = rot.flip_table(c, cells)
        ft.to_csv(Path(out, f"compass_flip_{arm}.csv"), index=False); flips[arm] = ft
        cr = rot.chain_rates(c, cells, types=[t for t in CHAIN])
        cr.to_csv(Path(out, f"compass_chain_{arm}.csv"), index=False); chain[arm] = cr
        print(f"\n== compass arm {arm} ({ARM_LABEL[arm]}), {n_runs} runs: bump drift per phase (w/s; ideal +-4.0 at 90 deg/s)")
        common.print_table(pd.DataFrame([r for r in rows if r["arm"] == arm])[["phase", "n_runs", "drift_w_s", "drift_sd", "heading_dps", "bump_vs", "yaw_abs_rad_s"] + [f"cmd_{ch}" for ch in CHANNELS]].fillna(""), max_rows=12)
        print(f"  flip table (L-R at ccw minus cw against the rest2-rest null), key types:")
        sel = ft[ft.type.isin(CHAIN + ["HSN", "Nod1", "DNp20", "IN13B001", "IN14B003"])]
        common.print_table(sel[["type", "n_L", "n_R", "flip_mean", "null_mean", "null_sd", "z", "p", "verdict", "LR_rest", "LR_ccw", "LR_rest2", "LR_cw"]], max_rows=40)
        print("  chain rates per phase (L / R Hz, max cell):")
        common.print_table(cr[cr.type.isin(["PS196_b", "GLNO", "PEN_a(PEN1)", "PEN_b(PEN2)", "AN07B037_a", "AN06A026", "AN07B035", "AN04B003", "LAL139", "DNa02"])][["phase", "type", "L", "L_sd", "R", "R_sd", "max_cell"]], max_rows=48)
    df = pd.DataFrame(rows); df.to_csv(Path(out, "compass_table.csv"), index=False)
    return df, flips, chain


def analyse_bench(d, out):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "bench_*_d*.json"))):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        for ch in j["checks"]:
            rows.append({"arm": j.get("vnc_arm"), "draw": Path(p).stem, "key": ch["key"], "measured": ch["measured"], "reference": ch["reference"], "criterion": ch["criterion"],
                         "status": ch["status"], "device": j["config"].get("device"), "commanded": j.get("proprioception_commanded_hz_mean")})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="key", columns="draw", values="measured", aggfunc="first")
    st = df.pivot_table(index="key", columns="draw", values="status", aggfunc="first")
    print("\n== benchmark sections rest / taste / smell / walk, arm A vs B (measured per draw; status)")
    print(piv.round(3).to_string()); print(st.to_string())
    df.to_csv(Path(out, "bench_table.csv"), index=False)
    return df


def cmd_analyse(args) -> int:
    from flyverse import connectome
    from flyverse.interp import trace as tr
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = load_room(args.dir)
    print(f"room runs per arm: {[(a, len(v)) for a, v in runs.items()]}")
    for a, items in runs.items():
        devs = sorted({j['device'] for _, j in items})
        print(f"  arm {a} ({ARM_LABEL[a]}): {[Path(p).name for p, _ in items]} device {devs}")
        if any("cpu" in str(x) for x in devs):
            print(f"  WARNING: arm {a} has a cpu run -- resubmit it, do not report it")
    summary = {"room_runs": {a: [p for p, _ in v] for a, v in runs.items()}, "generator": "scripts/probe_vnc_drive.py analyse " + " ".join(sys.argv[2:])}
    if runs:
        df = room_tables(runs, out)
        summary["room_table"] = common.to_jsonable(df.to_dict("records"))
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    lp = None
    if runs:
        prov0 = next(iter(runs.values()))[0][1]["provenance"]
        lp = tr.params_from_provenance(prov0)
        summary["provenance_of_first_run"] = {"compiled_connectome_md5": prov0["compiled_connectome"].get("md5"), "device": prov0["execution"].get("device"),
                                              "lif_receptor_model": prov0["model"]["lif"].get("receptor_model"), "type_path_gain": prov0["model"]["lif"].get("type_path_gain")}
    if runs and not args.skip_trace:
        analyse_trace(c, lp, runs, out)
    if runs and not args.skip_decompose:
        analyse_decompose(c, lp, runs, out)
    if runs and not args.skip_paths:
        summary["paths"] = common.to_jsonable(analyse_paths(c, lp, runs, out).to_dict("records"))
    cr = load_compass(args.dir)
    if cr:
        df, flips, chain = analyse_compass(c, cr, out)
        summary["compass_table"] = common.to_jsonable(df.to_dict("records"))
    bench = analyse_bench(args.dir, out)
    if bench is not None:
        summary["bench_table"] = common.to_jsonable(bench.to_dict("records"))
    summary["wall_s"] = round(time.time() - t0, 1)
    Path(out, "summary.json").write_text(json.dumps(common.to_jsonable(summary), indent=1), encoding="utf-8")
    print(f"\nwrote {out}/summary.json ({time.time() - t0:.0f} s)")
    return 0


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("room", help="one plain-fly room run (GPU)")
    r.add_argument("--arm", required=True, choices=sorted(ARMS)); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch", type=int, default=16); r.add_argument("--seconds", type=float, default=60.0); r.add_argument("--skip", type=float, default=5.0)
    r.add_argument("--every", type=int, default=2, help="capture the watch / afferent Recorder every N frames"); r.add_argument("--mean-every", type=int, default=5)
    r.add_argument("--device", default=None); r.add_argument("--cuda-sparse", default="torch"); r.add_argument("--out", required=True)
    k = sub.add_parser("compass", help="the efferent rotation arm under one vnc arm (GPU)")
    k.add_argument("--arm", required=True, choices=sorted(ARMS)); k.add_argument("--seed", type=int, default=0)
    k.add_argument("--gains", default="2:15"); k.add_argument("--seconds", type=float, default=10.0); k.add_argument("--skip", type=float, default=3.0)
    k.add_argument("--rate", type=float, default=90.0); k.add_argument("--dna02-hz", type=float, default=20.0); k.add_argument("--sparse", default="warp", choices=["warp", "torch"])
    k.add_argument("--quick", action="store_true"); k.add_argument("--device", default=None); k.add_argument("--cache-dir", default=None); k.add_argument("--out", required=True)
    b = sub.add_parser("bench", help="scripts/benchmark.py sections under arm A or B (GPU)")
    b.add_argument("--arm", required=True, choices=sorted(ARMS)); b.add_argument("--sections", default="rest,taste,smell,walk")
    b.add_argument("--fast", action="store_true"); b.add_argument("--eager", action="store_true"); b.add_argument("--device", default=None); b.add_argument("--out", required=True)
    p = sub.add_parser("plan", help="write the one cluster submission")
    p.add_argument("--dir", default="out/vncd"); p.add_argument("--runs", type=int, default=5); p.add_argument("--compass-seeds", type=int, default=3)
    p.add_argument("--draws", type=int, default=2); p.add_argument("--minutes", type=int, default=60)
    a = sub.add_parser("analyse", help="the audit's tables (CPU)")
    a.add_argument("--dir", default="out/vncd"); a.add_argument("--out", default="out/vncd/analysis"); a.add_argument("--cache-dir", default=None)
    a.add_argument("--skip-trace", action="store_true"); a.add_argument("--skip-decompose", action="store_true"); a.add_argument("--skip-paths", action="store_true")
    args = ap.parse_args(argv)
    return {"room": cmd_room, "compass": cmd_compass, "bench": cmd_bench, "plan": cmd_plan, "analyse": cmd_analyse}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
