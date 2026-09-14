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
# Round 3 (thread body-state, docs/audits/body_sided_state.md): the same protocols under the sided body state.
#   B = the round-2 transducer ('all'); C = + the stance / swing leg cycle (body.LegCycle attached to the body, the leg channels
#   read it per leg and per phase); D = C + the side-split haltere readout (motor.read_haltere_sides); E = D + the labelled
#   Coriolis stop-gap (the positive control). The compass protocol runs A, D, E.
ARMS_BODY = {"A": None, "B": "all", "C": "all+leg_cycle", "D": "all+leg_cycle+haltere_sided", "E": "all+leg_cycle+haltere_sided+haltere_coriolis"}
ARM_LABEL_BODY = {"A": "shipped (sense off)", "B": "proprioception all (round-2 form: leg channels on the side's MN rate, one bilateral haltere)",
                  "C": "all + leg cycle (per-leg, per-phase leg channels)", "D": "all + leg cycle + side-split haltere",
                  "E": "all + leg cycle + side-split haltere + haltere_coriolis (labelled stop-gap, positive control)"}
FAMILIES = {"vncd": (ARMS, ARM_LABEL, "ABE"), "body": (ARMS_BODY, ARM_LABEL_BODY, "ADE")}     # arms, labels, compass arms
WATCH = ["AN04B003", "AN07B035", "AN07B037_a", "AN06A026", "PS196_b", "LAL139", "GLNO", "IN12B014", "IN19A003", "DNa02"]
WATCH_BODY = WATCH + ["PS059"]
CHAIN = WATCH_BODY + ["PS047_b", "PS239", "LAL184", "WED040_a", "PEN_a(PEN1)", "PEN_b(PEN2)", "EPG", "Delta7", "PEG", "HSN", "Nod1", "PLP078", "LPsP"]
CHANNELS = ("chordotonal", "hair_plate", "campaniform", "haltere")
SUBSAMPLE = 12                       # afferent cells recorded per frame per channel (the channel mean comes from the window accumulator)
MOTOR_FIELDS = ("turn_L", "turn_R", "leg_L", "leg_R", "fwd_dn", "back_dn", "power", "haltere", "gf", "steer_L", "steer_R")
LEG_NAMES = ("L1", "R1", "L2", "R2", "L3", "R3")


def _log(msg):
    print(msg, flush=True)


def family_of(args):
    fam = getattr(args, "family", None) or "vncd"
    if fam not in FAMILIES:
        raise SystemExit(f"family must be one of {sorted(FAMILIES)}")
    return fam


def spec_of(arm, family="vncd"):
    arms = FAMILIES[family][0]
    if arm not in arms:
        raise SystemExit(f"arm must be one of {sorted(arms)}")
    return arms[arm]


def watch_of(family):
    return WATCH_BODY if family == "body" else WATCH


def afferent_groups(sense):
    """Per (channel, side) and per (channel, leg) cell groups for the sided report: leg channels by the side rule and the
    entry-nerve segment (senses.Proprioception.leg_of), the haltere channel by side. Keys 'chordotonal:L', 'chordotonal:L2', 'haltere:R' ..."""
    out = {}
    for ch in sense.channels:
        idx = np.asarray(sense.idx[ch]); side = sense.side[ch]
        for s, name in ((1, "L"), (-1, "R")):
            if (side == s).any():
                out[f"{ch}:{name}"] = idx[side == s]
        if ch != "haltere":
            _, seg = sense.leg_of(ch)
            for j, leg in enumerate(LEG_NAMES):
                m = (side == (1 if leg[0] == "L" else -1)) & (seg == int(leg[1]))
                if m.any():
                    out[f"{ch}:{leg}"] = idx[m]
    return out


def attach_cycle(sense, target):
    """Attach body.LegCycle to a BatchBody (`leg_cycle`) or a body.Locomotion (`cycle`) when the sense names 'leg_cycle'."""
    from flyverse import body
    if sense is None or not sense.leg_cycle:
        return None
    cyc = body.LegCycle()
    if hasattr(target, "leg_cycle"):
        target.leg_cycle = cyc
    else:
        target.cycle = cyc
    return cyc


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
    fam = family_of(args); arms, labels, _ = FAMILIES[fam]
    spec = spec_of(args.arm, fam)
    B = int(args.batch)
    seeds = list(range(args.seed * 100, args.seed * 100 + B))
    # scripts/probe_walk_straightness.py's protocol: BatchSim default start (-0.5, 0.05, table top), heading 5 deg, program
    # none, fruit all, NO fence, default energy; the only addition is the opt-in sense of the arm (and, under a 'leg_cycle'
    # spec, the opt-in body.LegCycle attached to the batch body -- a readout of the realised speed / yaw the walk never reads)
    sim = BatchSim(B, seed=args.seed, seeds=seeds, program="none", fruit_set="all", fence=False,
                   cuda_graphs=gpu, cuda_kernels=True if gpu else None, event_driven=True if gpu else None,
                   cuda_sparse=args.cuda_sparse, device=args.device, proprioception=spec)
    c, fb, brain = sim.fb.c, sim.fb, sim.fb.brain
    lp, op = brain.p, (sim.optic.p if sim.optic is not None else None)
    sense = fb.proprioception_sense if spec is not None else senses.Proprioception(c, "all")   # A: the same cells, recorded
    cycle = attach_cycle(fb.proprioception_sense if spec is not None else None, sim.body)
    counts = sense.counts()
    info = world.make_room(0, "all")[1]; top_z = float(info["table_top_z"]); x0, x1, y0, y1 = [float(v) for v in info["table_extent"]]
    _log(f"[room {args.arm}] family {fam} spec {spec or 'off'} channels {sense.channels} coriolis {sense.haltere_coriolis} leg_cycle {sense.leg_cycle} "
         f"haltere_sided {sense.haltere_sided} params {sense.params} mn_ref {sense.mn_ref_hz} cycle {vars(cycle) if cycle else None} block {args.block}")
    _log(f"[room {args.arm}] BatchSim B={B} neurons={c.n:,} device={brain.device} env seeds={seeds} brain seed={args.seed} receptor_model {lp.receptor_model} "
         f"type_path_gain {lp.type_path_gain} cuda kernels {brain.cuda} event_driven {brain.event_driven} cuda_graphs {fb.cuda_graphs} fence False")
    n = c.neurons; ty = n.type.fillna("").to_numpy(); side = n.somaSide.fillna("?").to_numpy()
    WATCH_ = watch_of(fam)
    watch_idx = {t: c.select(type=t) for t in WATCH_}
    sub = channel_subsample(sense)
    all_chan = {ch: np.asarray(sense.idx[ch]) for ch in sense.channels}
    groups = afferent_groups(sense)                                    # per (channel, side) and (channel, leg): the sided report
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
    commanded_g = {g: np.zeros((n_frames, B), np.float32) for g in groups}
    cyc_keys = ("load_L", "load_R", "amp_L", "amp_R", "stance_frac", "step_hz", "haltere_L", "haltere_R")
    cyc_t = {k: np.zeros((n_frames, B), np.float32) for k in cyc_keys}
    chan_t = {ch: brain._idx(idx) for ch, idx in all_chan.items()} if spec is not None else {}
    group_t = {g: brain._idx(idx) for g, idx in groups.items()}
    chan_pos_t = {ch: brain._idx(idx) for ch, idx in all_chan.items()}
    rate_max = torch.zeros(c.n, dtype=torch.float32, device=fb.device)
    chan_sum = {ch: torch.zeros(B, dtype=torch.float64, device=fb.device) for ch in all_chan}
    group_sum = {g: torch.zeros(B, dtype=torch.float64, device=fb.device) for g in groups}
    from flyverse.motor import haltere_side_groups
    hm_groups = sense.haltere_mn_groups if sense.haltere_mn_groups is not None else haltere_side_groups(c)
    hm_t = {s: brain._idx(hm_groups[s]) for s in ("L", "R")}
    counts0 = None; n_mean = 0
    leg_side = np.array([1, -1, 1, -1, 1, -1])
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
            for g, ti in group_t.items():
                group_sum[g] += r[:, ti].mean(dim=1).double()
        if chan_t:
            base = fb._base_poisson
            cmdv = torch.stack([base[:, ti].mean(dim=1) for ti in chan_t.values()] + [base[:, ti].mean(dim=1) for ti in group_t.values()], 1).cpu().numpy() * (1000.0 / brain.p.dt)
            for j, ch in enumerate(chan_t):
                commanded[ch][k] = cmdv[:, j]
            for j, g in enumerate(group_t):
                commanded_g[g][k] = cmdv[:, len(chan_t) + j]
        hm = torch.stack([brain.rate[:, hm_t["L"]].mean(dim=1), brain.rate[:, hm_t["R"]].mean(dim=1)], 1).cpu().numpy()   # the side-split haltere MN readout, every arm
        cyc_t["haltere_L"][k] = hm[:, 0]; cyc_t["haltere_R"][k] = hm[:, 1]
        m = sim.motor
        for name in MOTOR_FIELDS:
            motor[name][k] = np.broadcast_to(np.asarray(getattr(m, name, 0.0), np.float32), (B,))
        if cycle is not None:
            st = cycle.state(sim.flies)
            cyc_t["load_L"][k] = st["load_L"]; cyc_t["load_R"][k] = st["load_R"]; cyc_t["stance_frac"][k] = st["beta"]
            cyc_t["amp_L"][k] = st["amp"][:, leg_side > 0].mean(1); cyc_t["amp_R"][k] = st["amp"][:, leg_side < 0].mean(1)
            cyc_t["step_hz"][k] = cycle.timing(np.array([f.speed for f in sim.flies]))[1]
        for i, f in enumerate(sim.flies):
            cmd = sim.commands[i]
            heading[k, i] = f.heading; pos[k, i] = f.pos; airborne[k, i] = bool(f.airborne)
            on_top[k, i] = (x0 - 1e-3 <= f.pos[0] <= x1 + 1e-3) and (y0 - 1e-3 <= f.pos[1] <= y1 + 1e-3) and abs(f.pos[2] - top_z) < 0.01
            yaw_cmd[k, i] = cmd["yaw"]; speed_cmd[k, i] = cmd["speed"]
            for key in rates_cmd:
                rates_cmd[key][k, i] = cmd["rates"][key]
        if k % 1000 == 999:
            _log(f"  t {(k + 1) / 100:5.0f} s  on table {on_top[k].mean():.2f}  airborne {airborne[k].mean():.2f}  DNa02 L-R {float((motor['turn_L'][k] - motor['turn_R'][k]).mean()):+.3f} Hz  "
                 f"leg L-R {float((motor['leg_L'][k] - motor['leg_R'][k]).mean()):+.3f} Hz  haltere L-R {float((cyc_t['haltere_L'][k] - cyc_t['haltere_R'][k]).mean()):+.3f} Hz  "
                 f"step {float(cyc_t['step_hz'][k].mean()):.1f} Hz  wall {time.time() - t0:.0f} s")
    window_s = (n_frames - skip_f) / 100.0
    if counts0 is None:
        counts0 = torch.zeros_like(brain.spike_counts); window_s = n_frames / 100.0
    rate_flies = ((brain.spike_counts - counts0).double() / max(window_s, 1e-9)).float().cpu().numpy()           # (B, N)
    rate_max_np = rate_max.cpu().numpy()
    chan_measured = {ch: (v / max(n_mean, 1)).cpu().numpy() for ch, v in chan_sum.items()}                       # (B,) per channel
    group_measured = {g: (v / max(n_mean, 1)).cpu().numpy() for g, v in group_sum.items()}
    wall = time.time() - t0
    stim = {"protocol": "room_plain_walk_straightness", "arm": args.arm, "family": fam, "params": {"proprioception": spec, "channels": list(sense.channels),
            "haltere_coriolis": sense.haltere_coriolis, "leg_cycle": sense.leg_cycle, "haltere_sided": sense.haltere_sided, "leg_cycle_params": vars(cycle) if cycle else None,
            "rate_params": sense.params, "mn_ref_hz": sense.mn_ref_hz, "counts": counts, "haltere_mn_sides": {s: int(len(hm_groups[s])) for s in hm_groups},
            "batch": B, "seconds": args.seconds, "skip_s": args.skip, "program": "none", "fruit_set": "all", "fence": False,
            "start": "BatchSim default (-0.5, 0.05, table top), heading 5 deg", "energy": "Metabolism default", "wind": "0.3 m/s towards 180 deg (BatchSim default)",
            "capture_every_frames": every, "mean_every_frames": mean_every, "block": args.block},
            "control": {"arm": "A", "proprioception": "off (the shipped path)"}}
    prov = common.provenance(c, lp, op, fb=fb, device=args.device, seeds=[args.seed], env_seeds=seeds, batch=B,
                             backend={"cuda_kernels": bool(brain.cuda), "event_driven": bool(brain.event_driven), "cuda_graphs": bool(fb.cuda_graphs), "cuda_sparse": args.cuda_sparse},
                             stimulus=stim)
    recording = rec.finish(meta={"protocol": "room_plain_walk_straightness", "arm": args.arm, "family": fam, "proprioception": spec, "seed": args.seed, "env_seeds": seeds,
                                 "watch": WATCH_, "channel_subsample": {ch: [int(i) for i in v] for ch, v in sub.items()}, "every": every,
                                 "provenance": prov, "generator": "scripts/probe_vnc_drive.py room " + " ".join(sys.argv[2:])})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta_f = {"protocol": "room_plain_walk_straightness", "arm": "as-given", "vnc_arm": args.arm, "family": fam, "proprioception": spec, "seed": args.seed, "env_seeds": seeds,
              "window_s": [args.skip, args.seconds], "default_quantity": {"spiking": "rate_hz"}, "provenance": prov,
              "run_id": f"vncd-{fam}-room-{args.arm}-r{args.seed}", "generator": "scripts/probe_vnc_drive.py room " + " ".join(sys.argv[2:])}
    tr.save_recording(flies_recording(c, rate_flies, meta_f), Path(str(out) + "_flies"))
    tr.save_recording(max_recording(c, rate_max_np, meta_f), Path(str(out) + "_max"))
    recording.save(Path(str(out) + "_rec"))
    np.savez_compressed(str(out) + "_body.npz", heading=heading, pos=pos, airborne=airborne, on_top=on_top, yaw_cmd=yaw_cmd, speed_cmd=speed_cmd,
                        **{f"cmd__{k}": v for k, v in rates_cmd.items()}, **{f"m__{k}": v for k, v in motor.items()},
                        **{f"commanded__{ch}": v for ch, v in commanded.items()}, **{f"commandedg__{g}": v for g, v in commanded_g.items()},
                        **{f"cyc__{k}": v for k, v in cyc_t.items()}, env_seeds=np.asarray(seeds))
    summary = summarise_room(c, recording, heading, pos, airborne, on_top, yaw_cmd, speed_cmd, rates_cmd, motor, commanded, chan_measured, all_chan, sub, watch_idx, skip_f, every,
                             commanded_g=commanded_g, group_measured=group_measured, groups=groups, cyc=cyc_t, cycle_on=cycle is not None)
    summary.update(arm=args.arm, arm_label=labels[args.arm], family=fam, proprioception=spec, seed=args.seed, env_seeds=seeds, batch=B, seconds=args.seconds, skip_s=args.skip,
                   wall_s=round(wall, 1), counts=counts, rate_params=sense.params, mn_ref_hz=sense.mn_ref_hz, channel_names=list(sense.channels),
                   haltere_coriolis=sense.haltere_coriolis, leg_cycle=sense.leg_cycle, haltere_sided=sense.haltere_sided, leg_cycle_params=vars(cycle) if cycle else None,
                   block=args.block, device=prov["execution"]["device"], device_name=prov["execution"].get("device_name"),
                   provenance=prov, lif={"receptor_model": lp.receptor_model, "receptor_net_rule": lp.receptor_net_rule, "type_path_gain": lp.type_path_gain},
                   files={"generator": "scripts/probe_vnc_drive.py room " + " ".join(sys.argv[2:]), "recording": str(out) + "_rec.npz", "flies": str(out) + "_flies.npz",
                          "max": str(out) + "_max.npz", "body": str(out) + "_body.npz"})
    Path(str(out) + ".json").write_text(json.dumps(common.to_jsonable(summary), indent=1), encoding="utf-8")
    s = summary["run"]
    _log(f"[room {args.arm} seed {args.seed}] device {summary['device']} ({summary['device_name']}), {wall:.0f} s wall; yaw SD {s['yaw_sd_deg_s']:.2f} deg/s (median {s['yaw_sd_deg_s_median']:.2f}), "
         f"straightness {s['straightness']:.3f}, left the table {s['n_left_table']}/{B} (median {s['left_table_s_median']}), DNa02 L-R {s['DNa02_LR_hz']:+.3f} Hz, "
         f"leg L-R {s['leg_LR_hz']:+.3f} Hz, haltere MN L-R {s['haltere_LR_hz']:+.3f} Hz, power max {s['power_max_hz']:.1f} Hz, airborne {s['airborne_frac']:.3f}")
    _log("  channel / group        n   commanded Hz   measured Hz")
    for ch, v in summary["channels"].items():
        _log(f"  {ch:20s} {v['n']:4d}  {v['commanded_hz']:10.2f}   {v['measured_hz']:8.2f}")
    for g, v in summary["groups"].items():
        _log(f"  {g:20s} {v['n']:4d}  {v['commanded_hz']:10.2f}   {v['measured_hz']:8.2f}")
    if cycle is not None:
        _log(f"  cycle: step {s['step_hz']:.2f} Hz, stance fraction {s['stance_frac']:.3f}, load L-R {s['load_LR']:+.4f} (|.| {s['abs_load_LR']:.3f}), amp L-R {s['amp_LR']:+.4f} (|.| {s['abs_amp_LR']:.4f})")
    _log("  " + "; ".join(f"{k} {v['hz']:.3f}" for k, v in summary["watch"].items()))
    _log(f"wrote {out}.json")
    if summary["device"] is None or "cpu" in str(summary["device"]):
        _log("WARNING: device cpu -- resubmit (the JSON records the realised device)")
    return 0


def summarise_room(c, recording, heading, pos, airborne, on_top, yaw_cmd, speed_cmd, rates_cmd, motor, commanded, chan_measured, all_chan, sub, watch_idx, skip_f, every,
                   commanded_g=None, group_measured=None, groups=None, cyc=None, cycle_on=False):
    commanded_g = commanded_g or {}; group_measured = group_measured or {}; groups = groups or {}; cyc = cyc or {}
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
        for g in groups:
            row[f"commanded_{g}_hz"] = float(commanded_g[g][win, i].mean()) if g in commanded_g else 0.0
            row[f"measured_{g}_hz"] = float(group_measured[g][i])
        for ch in ("chordotonal", "hair_plate", "campaniform", "haltere"):
            if f"{ch}:L" in groups and f"{ch}:R" in groups:
                row[f"commanded_{ch}_LR_hz"] = row[f"commanded_{ch}:L_hz"] - row[f"commanded_{ch}:R_hz"]
                row[f"commanded_{ch}_absLR_hz"] = float(np.abs(commanded_g[f"{ch}:L"][ok, i] - commanded_g[f"{ch}:R"][ok, i]).mean()) if (ok.any() and f"{ch}:L" in commanded_g) else 0.0
        if "haltere_L" in cyc:
            row["haltere_L_hz"] = float(cyc["haltere_L"][ok, i].mean()) if ok.any() else None; row["haltere_R_hz"] = float(cyc["haltere_R"][ok, i].mean()) if ok.any() else None
            row["haltere_LR_hz"] = float((cyc["haltere_L"][ok, i] - cyc["haltere_R"][ok, i]).mean()) if ok.any() else None
            row["haltere_abs_LR_hz"] = float(np.abs(cyc["haltere_L"][ok, i] - cyc["haltere_R"][ok, i]).mean()) if ok.any() else None
        if cycle_on and ok.any():
            row["step_hz"] = float(cyc["step_hz"][ok, i].mean()); row["stance_frac"] = float(cyc["stance_frac"][ok, i].mean())
            row["load_LR"] = float((cyc["load_L"][ok, i] - cyc["load_R"][ok, i]).mean()); row["abs_load_LR"] = float(np.abs(cyc["load_L"][ok, i] - cyc["load_R"][ok, i]).mean())
            row["amp_LR"] = float((cyc["amp_L"][ok, i] - cyc["amp_R"][ok, i]).mean()); row["abs_amp_LR"] = float(np.abs(cyc["amp_L"][ok, i] - cyc["amp_R"][ok, i]).mean())
            yaw_f = dh[:, i]; a_lr = (cyc["amp_L"][1:, i] - cyc["amp_R"][1:, i])[ok[1:]]; y_f = yaw_f[ok[1:]]
            row["amp_LR_yaw_corr"] = float(np.corrcoef(a_lr, y_f)[0, 1]) if len(y_f) > 2 and a_lr.std() > 0 and y_f.std() > 0 else None   # the kinematics: -1 expected (left turn -> left legs shorter)
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
    group_out = {g: {"n": int(len(groups[g])), "commanded_hz": run.get(f"commanded_{g}_hz"), "measured_hz": run.get(f"measured_{g}_hz")} for g in groups}
    watch = {f"{t}_{s}": {"hz": run.get(f"{t}_{s}_hz"), "fly_sd": run.get(f"{t}_{s}_hz_fly_sd"), "n_cells": int((side[watch_idx[t]] == s).sum())}
             for t in watch_idx for s in ("L", "R") if f"{t}_{s}_hz" in run}
    return {"window_skip_s": skip_f / 100.0, "frames": int(n_frames), "captured_every": every, "run": run, "rows": common.to_jsonable(df.to_dict("records")),
            "channels": channels, "groups": group_out, "watch": watch}


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
    fam = family_of(args); arms, labels, compass_arms = FAMILIES[fam]
    spec = spec_of(args.arm, fam)
    if args.arm not in compass_arms:
        _log(f"note: the compass protocol was specified for arms {compass_arms} of family {fam}; running the requested arm anyway")
    seed = int(args.seed)
    gains = rot.parse_gains(args.gains)
    c, cdir = rot.load_condition_connectome("default", args.cache_dir, verbose=False)
    sim = rot.build_sim(c, gains, seed, args.device, cpu, sparse=args.sparse)
    fb = sim.fb
    if spec is not None:
        fb.proprioception_sense = senses.Proprioception(c, spec)
    sense = fb.proprioception_sense if spec is not None else senses.Proprioception(c, "all")
    cycle = attach_cycle(fb.proprioception_sense if spec is not None else None, sim.loco)     # scalar body: Locomotion.cycle
    groups = afferent_groups(sense)
    from flyverse.motor import haltere_side_groups
    hm_groups = sense.haltere_mn_groups if sense.haltere_mn_groups is not None else haltere_side_groups(c)
    hm_t = {s: fb.brain._idx(hm_groups[s]) for s in ("L", "R")}
    _log(f"[compass {args.arm}] family {fam} spec {spec or 'off'} channels {sense.channels} coriolis {sense.haltere_coriolis} leg_cycle {sense.leg_cycle} "
         f"haltere_sided {sense.haltere_sided} cycle {vars(cycle) if cycle else None} block {args.block}; available senses {fb.available_senses}")
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
    group_t = {g: fb.brain._idx(idx) for g, idx in groups.items()} if spec is not None else {}
    lif_p = fb.brain.p; optic_p = fb.optic.p if fb.optic is not None else None
    stim_common = {"protocol": "rotation", "mode": "efferent", "condition": "default", "arm": args.arm, "family": fam, "gains": list(gains) if gains else None,
                   "params": {"pos": list(rot.POS), "rate_dps": rate_dps, "seconds": seconds, "skip_s": skip / 100.0, "phases": [p for p, _ in rot.PHASES],
                              "wind_speed": 0.0, "background_hz": rot.BACKGROUND_HZ if gains else 0.0, "pulse_hz": rot.PULSE_HZ if gains else 0.0,
                              "pulse_s": rot.PULSE_S, "pulse_wedges": list(rot.PULSE_WEDGES), "dna02_hz": float(args.dna02_hz),
                              "compass_adaptation": 0.0 if gains else None, "sparse": args.sparse, "proprioception": spec,
                              "channels": list(sense.channels), "haltere_coriolis": sense.haltere_coriolis, "leg_cycle": sense.leg_cycle, "haltere_sided": sense.haltere_sided,
                              "leg_cycle_params": vars(cycle) if cycle else None, "haltere_mn_sides": {s: int(len(hm_groups[s])) for s in hm_groups},
                              "rate_params": sense.params, "mn_ref_hz": sense.mn_ref_hz, "block": args.block},
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
        commanded_g = {g: np.zeros(n_frames, np.float32) for g in groups}
        cyc_t = {k: np.zeros(n_frames, np.float32) for k in ("load_L", "load_R", "amp_L", "amp_R", "stance_frac", "step_hz", "haltere_L", "haltere_R", "speed")}
        leg_side = np.array([1, -1, 1, -1, 1, -1])
        airborne = 0
        if sgn != 0.0:
            fb.stimulate(dna02["L" if sgn > 0 else "R"], float(args.dna02_hz), seconds * 1000)
        for k in range(n_frames):
            sim.fly.place(*rot.POS, heading=sim.fly.heading)          # efferent: position pinned, heading integrated by the body
            if spec is not None:
                # the previous frame's readout (zeros before the first: BatchSim's rule), the leg-cycle state and the side-split
                # haltere rates travel through the sense exactly as BatchSim.step feeds them
                state = sim.loco.proprio_state(sim.fly, motor, haltere_sides=sense.haltere_sides(fb.brain))
                fb.proprioception(**sense.take_body(state))
            sim.step()
            motor = fb.motor()
            acc.add(fb, k, skip)
            rec.capture(fb, t_ms=10.0 * k, motor=motor)
            if chan_t:
                base = fb._base_poisson
                for ch, ti in chan_t.items():
                    commanded[ch][k] = float(base[0, ti].mean()) * (1000.0 / fb.brain.p.dt)
                for g, ti in group_t.items():
                    commanded_g[g][k] = float(base[0, ti].mean()) * (1000.0 / fb.brain.p.dt)
            r = fb.brain.rate_np(); re_ = r[idx_epg]
            cyc_t["haltere_L"][k] = float(r[hm_groups["L"]].mean()); cyc_t["haltere_R"][k] = float(r[hm_groups["R"]].mean()); cyc_t["speed"][k] = sim.fly.speed
            if cycle is not None:
                fl = sim.fly
                cyc_t["load_L"][k] = fl.stance_load_L; cyc_t["load_R"][k] = fl.stance_load_R; cyc_t["stance_frac"][k] = fl.stance_frac
                cyc_t["amp_L"][k] = fl.leg_amp[leg_side > 0].mean(); cyc_t["amp_R"][k] = fl.leg_amp[leg_side < 0].mean()
                cyc_t["step_hz"][k] = cycle.timing([fl.speed])[1][0]
            prof = np.array([float(re_[wedge_of == w].mean()) for w in range(16)])
            cc, vs = rot.circ_centre(prof)
            track.append((cc, vs, float(prof.max())))
            headings.append(float(sim.fly.heading)); yaws.append(float(sim.fly.yaw_rate)); airborne += int(bool(sim.fly.airborne))
        bump = rot.bump_metrics(track, skip); head = rot.heading_metrics(headings, skip)
        cyc_summary = {k: float(v[skip:].mean()) for k, v in cyc_t.items()}
        cyc_summary["haltere_LR_hz"] = cyc_summary["haltere_L"] - cyc_summary["haltere_R"]
        cyc_summary["load_LR"] = cyc_summary["load_L"] - cyc_summary["load_R"]; cyc_summary["amp_LR"] = cyc_summary["amp_L"] - cyc_summary["amp_R"]
        meta = {"protocol": "rotation", "arm": name, "phase": name, "rotation_sign": sgn, "mode": "efferent", "condition": "default", "vnc_arm": args.arm, "family": fam,
                "proprioception": spec, "gains": list(gains) if gains else None, "seed": seed, "window_s": [skip / 100.0, n_frames / 100.0],
                "stimulus": dict(stim_common, phase=name, rotation_sign=sgn), "default_quantity": {"spiking": "rate_hz"}, "provenance": prov,
                "run_id": f"vncd-{fam}-compass-{args.arm}-{name}-r{seed}", "bump": bump, "heading": head, "airborne_frames": airborne,
                "yaw_rate_abs_mean_rad_s": float(np.mean(np.abs(yaws[skip:]))) if len(yaws) > skip else float("nan"),
                "commanded_hz": {ch: float(v[skip:].mean()) for ch, v in commanded.items()},
                "commanded_group_hz": {g: float(v[skip:].mean()) for g, v in commanded_g.items()}, "cycle": cyc_summary,
                "generator": "scripts/probe_vnc_drive.py compass " + " ".join(sys.argv[2:])}
        cells_rec, _ = acc.finish(meta)
        stem = Path(f"{out}_{name}")
        tr.save_recording(cells_rec, stem); written.append(str(stem.with_suffix(".npz")))
        R = rec.finish(dict(meta, channel_subsample={ch: [int(i) for i in v] for ch, v in sub.items()}, chain=CHAIN))
        R.motor = dict(R.motor, **{f"commanded.{ch}": v for ch, v in commanded.items()}, **{f"commandedg.{g}": v for g, v in commanded_g.items()},
                       **{f"cyc.{k}": v for k, v in cyc_t.items()}, yaw_rate=np.asarray(yaws, np.float32), heading=np.asarray(headings, np.float32))
        tr.save_recording(R, Path(f"{out}_{name}_rec")); written.append(str(Path(f"{out}_{name}_rec").with_suffix(".npz")))
        summary_phases[name] = {"bump": bump, "heading": head, "airborne_frames": airborne, "commanded_hz": meta["commanded_hz"], "commanded_group_hz": meta["commanded_group_hz"],
                                "cycle": cyc_summary, "yaw_rate_abs_mean_rad_s": meta["yaw_rate_abs_mean_rad_s"], "wall_s": round(time.time() - t1, 1)}
        _log(f"  {name} ({('DNa02_%s %.0f Hz' % ('L' if sgn > 0 else 'R', args.dna02_hz)) if sgn else 'no stimulus'}, {seconds} s): {time.time() - t1:.0f} s wall; "
             f"bump vs {bump['vs']:.2f} peak {bump['peak']:.0f} Hz drift {bump['drift_wedges_per_s']:+.3f} w/s; heading {head['rate_dps']:+.1f} deg/s; "
             f"|yaw| {meta['yaw_rate_abs_mean_rad_s']:.2f} rad/s; haltere MN L-R {cyc_summary['haltere_LR_hz']:+.2f} Hz; amp L-R {cyc_summary['amp_LR']:+.3f}; "
             f"commanded " + ", ".join(f"{ch} {v:.1f}" for ch, v in meta["commanded_hz"].items())
             + "; sided " + ", ".join(f"{g} {v:.1f}" for g, v in meta["commanded_group_hz"].items() if ":L" in g or ":R" in g))
    dev = prov["execution"]["device"]
    with open(f"{out}_run.json", "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable({"arm": args.arm, "arm_label": labels[args.arm], "family": fam, "proprioception": spec, "mode": "efferent", "condition": "default", "seed": seed,
                                      "leg_cycle": sense.leg_cycle, "haltere_sided": sense.haltere_sided, "haltere_coriolis": sense.haltere_coriolis,
                                      "leg_cycle_params": vars(cycle) if cycle else None, "block": args.block,
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
    fam = family_of(args); labels = FAMILIES[fam][1]
    spec = spec_of(args.arm, fam)
    if spec is not None and ("leg_cycle" in spec or "haltere_sided" in spec):
        raise SystemExit("the bench protocol has no body (a bare brain.Brain): only the round-2 specs run there; use --family vncd")
    log = install_bench_brain(spec) if spec is not None else None
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    argv = ["benchmark.py", "--sections", args.sections, "--json", str(out)] + (["--fast"] if args.fast else []) + (["--eager"] if args.eager else [])
    _log(f"[bench {args.arm}] spec {spec or 'off'}; benchmark argv {argv[1:]}")
    sys.argv = argv
    benchmark.main()
    d = json.loads(out.read_text(encoding="utf-8"))
    d["vnc_arm"] = args.arm; d["arm_label"] = labels[args.arm]; d["proprioception"] = spec; d["family"] = fam; d["block"] = args.block
    d["proprioception_frames"] = None if log is None else log["frames"]
    d["proprioception_commanded_hz_mean"] = None if log is None else {ch: float(np.mean(v)) for ch, v in log["commanded"].items()}
    d["generator"] = f"scripts/probe_vnc_drive.py bench --arm {args.arm} --sections {args.sections} --out {out}"
    out.write_text(json.dumps(d, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    _log(f"[bench {args.arm}] device {d['config'].get('device')}; checks " + "; ".join(f"{c_['key']} {c_['measured']} {c_['status']}" for c_ in d["checks"])
         + (f"; transducer frames {log['frames']}, commanded " + ", ".join(f"{k} {v:.1f}" for k, v in d["proprioception_commanded_hz_mean"].items()) if log else ""))
    return 0


# ---------------------------------------------------------------------------------------------- plan (CPU)
def cmd_plan(args) -> int:
    """ONE cluster submission. Every job line preserves python's exit code (`st=$?; tail; exit $st`, docs/INTERP.md 10.4
    item 4), and every command carries a `fam_<block>` token so `--arm-block fam` keeps a comparison family on ONE box:
    the block is the SEED -- all five arms of seed s (the reference A with its treatments) run on the same target -- and
    each compass job (its arms sequential inside one python process each) is its own block."""
    fam = family_of(args); arms, _, compass_arms = FAMILIES[fam]
    d = args.dir.rstrip("/")
    name = args.name or ("vncd" if fam == "vncd" else "vncd3")
    pre = f"mkdir -p {d} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
    fam_flag = f" --family {fam}" if fam != "vncd" else ""
    cmds = []

    # The job strings are written into batch.sh inside DOUBLE quotes, so every `$` below is escaped as `\$` there: the
    # local shell must hand `st=$?; ...; exit $st` to cluster_run verbatim (the first vncd3 submission lost them --
    # `st=0; ...; exit ` -- and was cancelled; docs/audits/body_sided_state.md 3).
    def job(cmd, log):
        return f"{cmd} > {log} 2>&1; st=$?; tail -4 {log}; exit $st"

    room_seeds = [int(x) for x in args.only_seeds.split(",")] if args.only_seeds else list(range(args.runs))
    compass_seeds = [int(x) for x in args.only_compass_seeds.split(",")] if args.only_compass_seeds else list(range(args.compass_seeds))
    for s in room_seeds:
        for arm in "ABCDE":
            stem = f"{d}/room_{arm}_r{s}"
            cmds.append(pre + job(f"python scripts/probe_vnc_drive.py room{fam_flag} --arm {arm} --seed {s} --batch 16 --seconds 60 --block fam_r{s} --out {stem}", f"{stem}.txt"))
    for s in compass_seeds:
        parts = []; sts = []
        for j, arm in enumerate(compass_arms):
            stem = f"{d}/compass_{arm}_r{s}"
            parts.append(f"python scripts/probe_vnc_drive.py compass{fam_flag} --arm {arm} --seed {s} --block fam_c{s} --out {stem} > {stem}.txt 2>&1; s{j}=$?; tail -4 {stem}.txt")
            sts.append(f"s{j}")
        cmds.append(pre + "; ".join(parts) + f"; exit $(({' | '.join(sts)}))")
    if args.draws:
        for arm in "AB":
            parts = []; sts = []
            for k in range(args.draws):
                stem = f"{d}/bench_{arm}_d{k}"
                parts.append(f"python scripts/probe_vnc_drive.py bench{fam_flag} --arm {arm} --sections rest,taste,smell,walk --block fam_bench --out {stem}.json > {stem}.txt 2>&1; s{k}=$?; tail -12 {stem}.txt")
                sts.append(f"s{k}")
            cmds.append(pre + "; ".join(parts) + f"; exit $(({' | '.join(sts)}))")

    def quoted(c_):
        return '"' + c_.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'

    targets = f" --targets {args.targets}" if args.targets else ""
    line = (f"python scripts/cluster_run.py --name {name} --minutes {args.minutes} --arm-block fam{targets} " + " ".join(quoted(c_) for c_ in cmds)
            + f" --fetch {d}/ 2>&1 | tee out/{name}_cluster.log")
    Path(d).mkdir(parents=True, exist_ok=True)
    script = args.script or "batch.sh"
    Path(d, script).write_text("#!/bin/bash\n# ONE submission: " + f"{len(cmds)} jobs = {5 * len(room_seeds)} room (seeds {room_seeds}; blocks fam_r<seed>: the five arms of one seed on one box) + "
                               f"{len(compass_seeds)} compass (seeds {compass_seeds}; {len(compass_arms)} arms {compass_arms} each, sequential, blocks fam_c<seed>)"
                               + (f" + 2 bench ({args.draws} draws each, sequential)" if args.draws else "") + f"; family {fam}; targets {args.targets or 'default'}\n" + line + "\n", encoding="utf-8")
    _log(f"wrote {d}/{script} ({len(cmds)} jobs, family {fam}, name {name}, targets {args.targets or 'default'})")
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


ROBUST_KEYS = [("yaw_sd_clean_deg_s", "yaw-rate SD on CLEAN walking frames (on the table top, no airborne frame within 0.5 s; deg/s)"),
               ("yaw_median_abs_clean_deg_s", "median |yaw rate| on clean frames (deg/s)"), ("yaw_p99_abs_clean_deg_s", "99th percentile |yaw rate| on clean frames (deg/s)"),
               ("yaw_frac_gt30_clean", "fraction of clean frames with |yaw| > 30 deg/s"), ("yaw_frac_gt100_clean", "fraction of clean frames with |yaw| > 100 deg/s"),
               ("yaw_max_abs_deg_s", "max |yaw rate| over ALL walking frames (deg/s; the hop / edge artefact the plain SD carries)"),
               ("yaw_artefact_frames", "walking frames per fly with |yaw| > 720 deg/s (a table-edge crossing or landing: not a turn)"),
               ("clean_frac", "fraction of window frames that are clean"), ("net_turns", "net heading change over the window (turns, |.| per fly mean)"),
               ("leg_LR_neg_flies", "flies (of 16) whose window-mean leg L-R is negative"), ("DNa02_LR_pos_flies", "flies (of 16) whose window-mean DNa02 L-R is positive"),
               ("DNa02_active_frac", "fraction of walking frames with DNa02_L or DNa02_R > 5 Hz"),
               ("cmd_chordotonal_LR_flip_frac", "fraction of clean frames where the chordotonal command L-R has the sign the kinematics predict for the realised yaw (yaw > 0 -> R > L)")]


def robust_room(body_npz, skip_f=500, guard=50):
    """Per-run robust yaw statistics from the per-frame body arrays (ROBUST_KEYS), computed by analyse for every run
    (the summary's yaw_sd_deg_s counts every walking frame, and a single hop landing or table-edge crossing contributes
    a 90-180 deg heading jump = 9,000-18,000 deg/s to it; docs/audits/body_sided_state.md 4). A clean frame is on the
    table top before and after, no airborne frame within `guard` frames, and |yaw| <= 720 deg/s (7.2 deg per 10 ms
    frame: above that the heading representation flipped over an edge, it is not a turn)."""
    z = np.load(body_npz)
    h = z["heading"].astype(np.float64); air = z["airborne"]; top = z["on_top"]
    dh = np.degrees(np.diff(np.unwrap(h, axis=0), axis=0) * 100.0)
    T, B = dh.shape
    clean = np.abs(dh) <= 720.0
    for i in range(B):
        for k in np.flatnonzero(air[:, i]):
            clean[max(0, k - guard):k + guard + 1, i] = False
        clean[:, i] &= top[1:, i] & top[:-1, i]
    win = np.zeros(T, bool); win[skip_f:] = True
    walking = ~air[1:]
    tL, tR = z["cmd__DNa02_L"], z["cmd__DNa02_R"]; lL, lR = z["cmd__legMN_L"], z["cmd__legMN_R"]
    rows = []
    for i in range(B):
        c = clean[:, i] & win; w = dh[c, i]; wa = dh[win & walking[:, i], i]
        okf = np.zeros(T + 1, bool); okf[skip_f:] = True; okf &= ~air[:, i]
        row = {"yaw_sd_clean_deg_s": float(w.std()) if len(w) > 10 else None, "yaw_median_abs_clean_deg_s": float(np.median(np.abs(w))) if len(w) else None,
               "yaw_p99_abs_clean_deg_s": float(np.percentile(np.abs(w), 99)) if len(w) else None,
               "yaw_frac_gt30_clean": float((np.abs(w) > 30).mean()) if len(w) else None, "yaw_frac_gt100_clean": float((np.abs(w) > 100).mean()) if len(w) else None,
               "yaw_max_abs_deg_s": float(np.abs(wa).max()) if len(wa) else None, "yaw_artefact_frames": float((np.abs(wa) > 720.0).sum()), "clean_frac": float(c[skip_f:].mean()),
               "net_turns": float(abs(h[-1, i] - h[skip_f, i]) / (2 * np.pi)),
               "leg_LR_neg_flies": float((lL[okf, i] - lR[okf, i]).mean() < 0), "DNa02_LR_pos_flies": float((tL[okf, i] - tR[okf, i]).mean() > 0),
               "DNa02_active_frac": float(((tL[okf, i] > 5) | (tR[okf, i] > 5)).mean())}
        if "commandedg__chordotonal:L" in z:
            cL, cR = z["commandedg__chordotonal:L"][1:, i], z["commandedg__chordotonal:R"][1:, i]
            m = c & (np.abs(dh[:, i]) > 5) & win
            row["cmd_chordotonal_LR_flip_frac"] = float((np.sign(cR - cL)[m] == np.sign(dh[m, i])).mean()) if m.sum() > 10 else None
        rows.append(row)
    out = {}
    for k, _ in ROBUST_KEYS:
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        out[k] = (float(v.sum()) if k.endswith("_flies") else float(v.mean())) if len(v) else None
    return out


def load_room(d, only_seeds=None):
    runs = {}
    import re
    for p in sorted(glob.glob(os.path.join(d, "room_*_r*.json"))):
        m = re.fullmatch(r"room_[A-E]_r(\d+)\.json", Path(p).name)
        if not m or (only_seeds is not None and int(m.group(1)) not in only_seeds):
            continue
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        body = Path(p[:-5] + "_body.npz")
        if body.exists():
            try:
                j["run"].update(robust_room(body, skip_f=int(round(j.get("skip_s", 5.0) * 100))))
            except Exception as e:                                       # noqa: BLE001 - the summary keys still tabulate
                print(f"  robust_room failed on {body.name}: {e}")
        runs.setdefault(j["arm"], []).append((p, j))
    return runs


def label_of(items, arm):
    """The arm label recorded in the run JSONs (the family's), falling back to the round-2 table."""
    for _, j in items:
        if j.get("arm_label"):
            return j["arm_label"]
    return ARM_LABEL.get(arm, arm)


CYCLE_KEYS = [("haltere_L_hz", "haltere MN L (Hz; side-split readout, every arm)"), ("haltere_R_hz", "haltere MN R (Hz)"), ("haltere_LR_hz", "haltere MN L-R (Hz)"),
              ("haltere_abs_LR_hz", "|haltere MN L-R| (Hz)"), ("step_hz", "leg cycle step frequency (Hz)"), ("stance_frac", "leg cycle stance fraction"),
              ("load_LR", "stance load L-R (share of body weight)"), ("abs_load_LR", "|stance load L-R|"), ("amp_LR", "leg amplitude L-R (stance path / step_ref)"),
              ("abs_amp_LR", "|leg amplitude L-R|"), ("amp_LR_yaw_corr", "corr(amp L-R, realised yaw) per fly (-1 = the kinematics)")]


def room_tables(runs, out):
    rows = []
    arms = [a for a in "ABCDE" if a in runs]
    keys = list(ROOM_KEYS) + list(ROBUST_KEYS) + list(CYCLE_KEYS)
    j0 = runs[arms[0]][0][1]
    for ch in j0["channels"]:
        keys += [(f"commanded_{ch}_hz", f"channel {ch} commanded Hz"), (f"measured_{ch}_hz", f"channel {ch} measured Hz")]
    groups = sorted({g for items in runs.values() for _, j in items for g in j.get("groups", {})}, key=lambda g: (g.split(":")[0], len(g), g))
    for g in groups:
        keys += [(f"commanded_{g}_hz", f"group {g} commanded Hz"), (f"measured_{g}_hz", f"group {g} measured Hz")]
    for ch in ("chordotonal", "hair_plate", "campaniform", "haltere"):
        keys += [(f"commanded_{ch}_LR_hz", f"channel {ch} commanded L-R (Hz)"), (f"commanded_{ch}_absLR_hz", f"channel {ch} commanded |L-R| per frame (Hz)")]
    watch_keys = sorted({w for items in runs.values() for _, j in items for w in j["watch"]})
    for w in watch_keys:
        keys.append((f"{w}_hz", f"{w} (Hz)"))
    keys = [(k, l) for k, l in keys if any(j["run"].get(k) is not None for items in runs.values() for _, j in items)]
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
        if arm not in runs:
            continue
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


def load_compass(d, only_seeds=None):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "compass_*_r*_run.json"))):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        if only_seeds is not None and int(j["seed"]) not in only_seeds:
            continue
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
        for ph in phases:                                          # the sided afferent input and the body's own sided state per phase (round 3)
            r = next(r_ for r_ in rows if r_["arm"] == arm and r_["phase"] == ph)
            gk = sorted({g for _, j in items for g in j["phases"][ph].get("commanded_group_hz", {})})
            for g in gk:
                r[f"cmdg_{g}"] = float(np.mean([j["phases"][ph]["commanded_group_hz"].get(g, np.nan) for _, j in items]))
            for k in ("haltere_LR_hz", "amp_LR", "load_LR", "step_hz", "stance_frac", "speed"):
                v = [j["phases"][ph].get("cycle", {}).get(k, np.nan) for _, j in items]
                r[f"cyc_{k}"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                r[f"cyc_{k}_runs"] = [round(float(x), 4) for x in v]
            for ch in ("chordotonal", "hair_plate", "campaniform", "haltere"):
                if f"cmdg_{ch}:L" in r and f"cmdg_{ch}:R" in r:
                    r[f"cmd_{ch}_LR"] = r[f"cmdg_{ch}:L"] - r[f"cmdg_{ch}:R"]
        ft = rot.flip_table(c, cells)
        ft.to_csv(Path(out, f"compass_flip_{arm}.csv"), index=False); flips[arm] = ft
        cr = rot.chain_rates(c, cells, types=[t for t in CHAIN])
        cr.to_csv(Path(out, f"compass_chain_{arm}.csv"), index=False); chain[arm] = cr
        print(f"\n== compass arm {arm} ({label_of(items, arm)}), {n_runs} runs: bump drift per phase (w/s; ideal +-4.0 at 90 deg/s)")
        common.print_table(pd.DataFrame([r for r in rows if r["arm"] == arm])[["phase", "n_runs", "drift_w_s", "drift_sd", "heading_dps", "bump_vs", "yaw_abs_rad_s"] + [f"cmd_{ch}" for ch in CHANNELS]].fillna(""), max_rows=12)
        sided_cols = [c_ for c_ in ("cmd_chordotonal_LR", "cmd_hair_plate_LR", "cmd_campaniform_LR", "cmd_haltere_LR", "cyc_haltere_LR_hz", "cyc_amp_LR", "cyc_load_LR", "cyc_step_hz", "cyc_stance_frac", "cyc_speed")
                      if any(c_ in r for r in rows if r["arm"] == arm)]
        if sided_cols:
            print("  sided afferent commands (L-R, Hz) and the body's sided state per phase:")
            common.print_table(pd.DataFrame([r for r in rows if r["arm"] == arm and r["phase"] in phases])[["phase"] + sided_cols].fillna(""), max_rows=8)
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
    only = set(int(x) for x in args.only_seeds.split(",")) if args.only_seeds else None
    runs = load_room(args.dir, only)
    print(f"room runs per arm: {[(a, len(v)) for a, v in runs.items()]}" + (f"  (seeds restricted to {sorted(only)})" if only else ""))
    for a, items in runs.items():
        devs = sorted({j['device'] for _, j in items}); names = sorted({str(j.get('device_name')) for _, j in items})
        fams = sorted({str(j.get('family', 'vncd')) for _, j in items}); specs = sorted({str(j.get('proprioception')) for _, j in items})
        print(f"  arm {a} ({label_of(items, a)}): {[Path(p).name for p, _ in items]} device {devs} {names} family {fams} spec {specs} "
              f"blocks {sorted({str(j.get('block')) for _, j in items})}")
        if any("cpu" in str(x) for x in devs):
            print(f"  WARNING: arm {a} has a cpu run -- resubmit it, do not report it")
        if len(specs) != 1 or len(fams) != 1:
            print(f"  WARNING: arm {a} mixes specs / families -- the directory holds runs of more than one family")
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
        analyse_trace(c, lp, runs, out, arms=tuple(a.strip() for a in args.trace_arms.split(",") if a.strip()))
    if runs and not args.skip_decompose:
        analyse_decompose(c, lp, runs, out, arms=tuple(a for a in "BCDE" if a in runs))
    if runs and not args.skip_paths:
        summary["paths"] = common.to_jsonable(analyse_paths(c, lp, runs, out).to_dict("records"))
    cr = load_compass(args.dir, only)
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


# ---------------------------------------------------------------------------------------------- pairs (CPU)
PAIRS = (("B", "A"), ("C", "B"), ("D", "C"), ("E", "D"), ("D", "B"))
SIDED_KEYS = [("yaw_signed_mean_deg_s", "mean SIGNED yaw rate on clean frames per fly (deg/s; + = left), mean over flies"),
              ("yaw_pos_flies", "flies (of 16) whose clean-frame mean yaw is positive (left)"),
              ("corr_yaw_chordLR", "corr(realised yaw, commanded chordotonal L-R) per fly on clean frames (kinematics: -1)"),
              ("corr_yaw_hairLR", "corr(realised yaw, commanded hair-plate L-R)"), ("corr_yaw_haltLR", "corr(realised yaw, commanded haltere L-R)"),
              ("corr_yaw_haltMN_LR", "corr(realised yaw, haltere MN L-R readout)"), ("corr_yaw_legMN_LR", "corr(realised yaw, leg MN L-R)"),
              ("corr_dna02LR_chordLR", "corr(DNa02 L-R command, commanded chordotonal L-R) per fly"), ("corr_dna02LR_haltLR", "corr(DNa02 L-R command, commanded haltere L-R)"),
              ("corr_an04LR_chordLR", "corr(AN04B003 L-R rate, commanded chordotonal L-R) per fly (rec frames)"), ("corr_an04LR_yaw", "corr(AN04B003 L-R rate, realised yaw)"),
              ("corr_ps059LR_haltLR", "corr(PS059 L-R rate, commanded haltere L-R)"), ("corr_ps196LR_haltLR", "corr(PS196_b L-R rate, commanded haltere L-R)"),
              ("corr_ps196LR_yaw", "corr(PS196_b L-R rate, realised yaw)"),
              ("dna02LR_given_chordLR_pos_minus_neg", "E[DNa02 L-R | chordotonal L-R > 0] - E[. | < 0] (Hz; the tripod alternation seen by DNa02)"),
              ("an04LR_given_chordLR_pos_minus_neg", "E[AN04B003 L-R | chordotonal L-R > 0] - E[. | < 0] (Hz)")]


def _clean_frames(z, skip_f=500, guard=50):
    """The clean-frame mask of robust_room (window, on the table top before and after, no airborne frame within `guard`,
    |yaw| <= 720 deg/s) and the per-frame yaw (deg/s), both (T-1, B)."""
    h = z["heading"].astype(np.float64); air = z["airborne"]; top = z["on_top"]
    dh = np.degrees(np.diff(np.unwrap(h, axis=0), axis=0) * 100.0)
    T, B = dh.shape
    clean = np.abs(dh) <= 720.0
    for i in range(B):
        for k in np.flatnonzero(air[:, i]):
            clean[max(0, k - guard):k + guard + 1, i] = False
        clean[:, i] &= top[1:, i] & top[:-1, i]
    clean[:skip_f] = False
    return dh, clean


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 and a.std() > 0 and b.std() > 0 else np.nan


def sided_frames(run_json, c):
    """Per-fly, per-frame sidedness statistics of ONE room run on its clean walking frames (SIDED_KEYS): does the body's
    sided state (the commanded afferent L-R per frame) reach DNa02 / AN04B003 / PS059 / PS196_b in a sided way, and does
    the realised yaw follow the afferent asymmetry the kinematics impose? Returns the run mean over flies (sums for the
    *_flies keys)."""
    p = Path(run_json); j = json.loads(p.read_text(encoding="utf-8"))
    z = np.load(str(p)[:-5] + "_body.npz"); rec = np.load(str(p)[:-5] + "_rec.npz")
    skip_f = int(round(j.get("skip_s", 5.0) * 100))
    dh, clean = _clean_frames(z, skip_f)
    T, B = dh.shape
    side = c.neurons.somaSide.fillna("?").to_numpy()[rec["idx"]]; types = rec["types"]; q = rec["q__rate_hz"]
    fr = np.clip(np.round(rec["t_ms"] / 10.0).astype(int) - 2, 0, T - 1)                   # rec sample (frame k, t = (k+1) dt) -> dh row k-1
    def lr(t):
        L = q[:, :, (types == t) & (side == "L")].mean(2); R = q[:, :, (types == t) & (side == "R")].mean(2)
        return L - R                                                                          # (S, B)
    an04, ps059, ps196 = lr("AN04B003"), lr("PS059"), lr("PS196_b")
    def g(name):
        return z[name] if name in z else None
    chord = g("commandedg__chordotonal:L"); chordR = g("commandedg__chordotonal:R")
    hair = g("commandedg__hair_plate:L"); hairR = g("commandedg__hair_plate:R")
    halt = g("commandedg__haltere:L"); haltR = g("commandedg__haltere:R")
    rows = []
    for i in range(B):
        m = clean[:, i]; y = dh[m, i]
        row = {"yaw_signed_mean_deg_s": float(y.mean()) if len(y) else np.nan, "yaw_pos_flies": float(y.mean() > 0) if len(y) else np.nan}
        dna = (z["cmd__DNa02_L"][1:, i] - z["cmd__DNa02_R"][1:, i])
        legmn = (z["cmd__legMN_L"][1:, i] - z["cmd__legMN_R"][1:, i]); hmn = (z["cyc__haltere_L"][1:, i] - z["cyc__haltere_R"][1:, i])
        row["corr_yaw_legMN_LR"] = _corr(y, legmn[m]); row["corr_yaw_haltMN_LR"] = _corr(y, hmn[m])
        if chord is not None:
            cl = (chord[1:, i] - chordR[1:, i]); hl = (hair[1:, i] - hairR[1:, i]); tl = (halt[1:, i] - haltR[1:, i])
            row["corr_yaw_chordLR"] = _corr(y, cl[m]); row["corr_yaw_hairLR"] = _corr(y, hl[m]); row["corr_yaw_haltLR"] = _corr(y, tl[m])
            row["corr_dna02LR_chordLR"] = _corr(dna[m], cl[m]); row["corr_dna02LR_haltLR"] = _corr(dna[m], tl[m])
            pos = m & (cl > 0); neg = m & (cl < 0)
            row["dna02LR_given_chordLR_pos_minus_neg"] = float(dna[pos].mean() - dna[neg].mean()) if pos.sum() > 10 and neg.sum() > 10 else np.nan
            ms = m[fr]                                                                        # the rec samples that fall on clean frames
            row["corr_an04LR_chordLR"] = _corr(an04[ms, i], cl[fr][ms]); row["corr_an04LR_yaw"] = _corr(an04[ms, i], dh[fr, i][ms])
            row["corr_ps059LR_haltLR"] = _corr(ps059[ms, i], tl[fr][ms]); row["corr_ps196LR_haltLR"] = _corr(ps196[ms, i], tl[fr][ms])
            row["corr_ps196LR_yaw"] = _corr(ps196[ms, i], dh[fr, i][ms])
            posS = ms & (cl[fr] > 0); negS = ms & (cl[fr] < 0)
            row["an04LR_given_chordLR_pos_minus_neg"] = float(an04[posS, i].mean() - an04[negS, i].mean()) if posS.sum() > 10 and negS.sum() > 10 else np.nan
        rows.append(row)
    out = {}
    for k, _ in SIDED_KEYS:
        v = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))], float)
        out[k] = (float(v.sum()) if k.endswith("_flies") else float(v.mean())) if len(v) else None
    return out


def decompose_summary(analysis_dir, target="DNa02", arms="ABCDE", top=12):
    """Per arm and side of `target`: the rate-weighted input totals (E, I, net; mV/s per post cell) from the analyse
    step's decompose_<target>_<side>_per_type.csv, plus the named rows (the PS059 cancellation, the AN04B003 term)."""
    named = ("PS059/L", "PS059/R", "AN04B003/L", "AN04B003/R", "AN06A026/L", "AN06A026/R", "AN07B035/L", "AN07B035/R", "IN12B014/L", "IN12B014/R",
             "IN19A003/L", "IN19A003/R", "GNG562/L", "GNG562/R", "LT51/L", "LT51/R", "PFL3/L", "PFL3/R")
    rows = []
    for side in "LR":
        f = Path(analysis_dir, f"decompose_{target}_{side}_per_type.csv")
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for arm in arms:
            col = f"{arm}_mean"
            if col not in df:
                continue
            v = df[col].fillna(0.0)
            tot = {"post": f"{target}_{side}", "arm": arm, "E_total": float(v[v > 0].sum()), "I_total": float(v[v < 0].sum()), "net": float(v.sum()),
                   "rate_hz": float(df[f"{arm}_rate_hz"].iloc[0]) if f"{arm}_rate_hz" in df else np.nan}
            for n in named:
                r = df[df.pre_group == n]
                tot[n] = float(r[col].iloc[0]) if len(r) else np.nan
                if len(r) and f"{arm}_sd" in df:
                    tot[n + "_sd"] = float(r[f"{arm}_sd"].iloc[0])
            top_e = df[v > 0].assign(_v=v[v > 0]).nlargest(top, "_v"); top_i = df[v < 0].assign(_v=v[v < 0]).nsmallest(top, "_v")
            tot["top_E"] = "; ".join(f"{g} {x:+.1f}" for g, x in zip(top_e.pre_group, top_e._v)); tot["top_I"] = "; ".join(f"{g} {x:+.1f}" for g, x in zip(top_i.pre_group, top_i._v))
            rows.append(tot)
    return pd.DataFrame(rows)


def cmd_pairs(args) -> int:
    """CPU: the adjacent-arm comparisons the audit's questions turn on (C vs B: the leg cycle; D vs C: the side-split
    haltere; E vs D: the stop-gap), the per-frame sidedness statistics, the DNa02 decomposition summary and the compass
    flip table of the chain types -- all from the artefacts `analyse` already read or wrote."""
    from flyverse import connectome
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    only = set(int(x) for x in args.only_seeds.split(",")) if args.only_seeds else None
    runs = load_room(args.dir, only)
    print(f"room runs per arm: {[(a, len(v)) for a, v in runs.items()]}" + (f"  (seeds restricted to {sorted(only)})" if only else ""))
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    # per-frame sidedness per run (attached to the run dicts so the pairwise table can compare them too)
    sided = {}
    for arm, items in runs.items():
        for p, j in items:
            s = sided_frames(p, c); j["run"].update(s); sided.setdefault(arm, []).append((Path(p).name, s))
    sd_rows = []
    for k, label in SIDED_KEYS:
        row = {"key": k, "label": label}
        for arm in "ABCDE":
            v = np.array([s[k] for _, s in sided.get(arm, []) if s.get(k) is not None], float)
            row[f"{arm}_mean"] = float(v.mean()) if len(v) else np.nan; row[f"{arm}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            row[f"{arm}_runs"] = [round(float(x), 4) for x in v]
        sd_rows.append(row)
    sdf = pd.DataFrame(sd_rows); sdf.to_csv(Path(out, "sided_frames.csv"), index=False)
    print("\n== per-frame sidedness on clean walking frames (run mean +- SD over runs; per-run values in sided_frames.csv)")
    for _, r in sdf.iterrows():
        print(f"  {r.key:40s} " + " | ".join(f"{a}: {r[f'{a}_mean']:+.3f} +- {r[f'{a}_sd']:.3f}" if np.isfinite(r[f"{a}_mean"]) else f"{a}: --" for a in "ABCDE"))
    # pairwise verdicts on every tabulated key
    keys = [k for k, _ in ROOM_KEYS + ROBUST_KEYS + CYCLE_KEYS + SIDED_KEYS]
    j0 = next(iter(runs.values()))[0][1]
    keys += [k for k in sorted(j0["run"]) if k.startswith(("commanded_", "measured_")) and k.endswith("_hz")]
    keys += sorted({f"{w}_hz" for items in runs.values() for _, j in items for w in j["watch"]})
    rows = []
    for key in keys:
        vals = {a: np.array([j["run"][key] for _, j in items if j["run"].get(key) is not None], float) for a, items in runs.items()}
        if not any(len(v) for v in vals.values()):
            continue
        row = {"key": key}
        for a, v in vals.items():
            row[f"{a}_mean"] = float(v.mean()) if len(v) else np.nan; row[f"{a}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan; row[f"{a}_n"] = int(len(v))
        for t, r in PAIRS:
            if len(vals.get(t, [])) and len(vals.get(r, [])):
                cmp_ = common.compare(vals[t], vals[r])
                row.update({f"{t}v{r}_diff": cmp_["diff"], f"{t}v{r}_z": cmp_["z"], f"{t}v{r}_p": cmp_["p"], f"{t}v{r}_verdict": cmp_["verdict"]})
        rows.append(row)
    pdf = pd.DataFrame(rows); pdf.to_csv(Path(out, "pairwise.csv"), index=False)
    print(f"\n== adjacent-arm verdicts (common.compare; {'5' if only is None else len(only)} runs per arm) -> {out}/pairwise.csv")
    show = [k for k, _ in ROOM_KEYS + ROBUST_KEYS + CYCLE_KEYS + SIDED_KEYS] + [k for k in keys if k.startswith("commanded_") and ("_LR_" in k or (k.endswith("_hz") and ":" not in k))] + [k for k in keys if k.endswith("_hz") and any(k.startswith(w) for w in WATCH_BODY)]
    for _, r in pdf[pdf.key.isin(show)].iterrows():
        cells = []
        for t, ref in PAIRS:
            if f"{t}v{ref}_verdict" in r and isinstance(r[f"{t}v{ref}_verdict"], str):
                cells.append(f"{t}v{ref} {r[f'{t}v{ref}_diff']:+.3f} z{r[f'{t}v{ref}_z']:+.1f} p{r[f'{t}v{ref}_p']:.3f} {r[f'{t}v{ref}_verdict']}")
        print(f"  {r.key:36s} " + " | ".join(cells))
    # DNa02 decomposition summary
    dd = decompose_summary(args.analysis or out, "DNa02", "ABCDE")
    if len(dd):
        dd.to_csv(Path(out, "dna02_decompose_summary.csv"), index=False)
        print(f"\n== DNa02 rate-weighted input per arm (mV/s per post cell; decompose_DNa02_*_per_type.csv) -> {out}/dna02_decompose_summary.csv")
        cols = ["post", "arm", "rate_hz", "E_total", "I_total", "net", "PS059/L", "PS059/R", "AN04B003/L", "AN04B003/R", "AN06A026/L", "AN06A026/R", "AN07B035/L", "AN07B035/R", "IN12B014/L", "IN12B014/R", "IN19A003/L", "IN19A003/R", "GNG562/L", "GNG562/R"]
        common.print_table(dd[[c_ for c_ in cols if c_ in dd]], max_rows=12)
        for _, r in dd.iterrows():
            print(f"  {r.post} {r.arm}: top E {r.top_E}\n      top I {r.top_I}")
    # compass flip rows of the chain types
    fl = []
    for arm in "ABCDE":
        f = Path(args.analysis or out, f"compass_flip_{arm}.csv")
        if f.exists():
            ft = pd.read_csv(f); ft = ft[ft.type.isin(CHAIN)].copy(); ft.insert(0, "arm", arm); fl.append(ft)
    if fl:
        fdf = pd.concat(fl); fdf.to_csv(Path(out, "compass_flip_chain.csv"), index=False)
        print(f"\n== compass flip table, chain types (L-R at ccw minus cw against the rest2 - rest null; per-seed values in compass_flip_chain.csv)")
        common.print_table(fdf[["arm", "type", "n_L", "n_R", "LR_rest", "LR_ccw", "LR_rest2", "LR_cw", "flip_mean", "flip_sd", "null_mean", "null_sd", "p", "verdict"]], max_rows=80)
    Path(out, "pairs_summary.json").write_text(json.dumps(common.to_jsonable({"generator": "scripts/probe_vnc_drive.py pairs " + " ".join(sys.argv[2:]), "room_runs": {a: [p for p, _ in v] for a, v in runs.items()},
                                                                               "pairs": list(PAIRS), "sided_frames": sd_rows, "pairwise": rows, "decompose_DNa02": dd.to_dict("records") if len(dd) else None,
                                                                               "wall_s": round(time.time() - t0, 1)}), indent=1), encoding="utf-8")
    print(f"\nwrote {out}/pairs_summary.json ({time.time() - t0:.0f} s)")
    return 0


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fam_help = "arm family: 'vncd' (round 2: B all / C legs / D haltere / E coriolis) or 'body' (round 3: C + leg cycle / D + side-split haltere / E + coriolis; docs/audits/body_sided_state.md)"
    r = sub.add_parser("room", help="one plain-fly room run (GPU)")
    r.add_argument("--arm", required=True, choices=sorted(ARMS)); r.add_argument("--seed", type=int, default=0); r.add_argument("--family", default="vncd", choices=sorted(FAMILIES), help=fam_help)
    r.add_argument("--batch", type=int, default=16); r.add_argument("--seconds", type=float, default=60.0); r.add_argument("--skip", type=float, default=5.0)
    r.add_argument("--every", type=int, default=2, help="capture the watch / afferent Recorder every N frames"); r.add_argument("--mean-every", type=int, default=5)
    r.add_argument("--device", default=None); r.add_argument("--cuda-sparse", default="torch"); r.add_argument("--out", required=True)
    r.add_argument("--block", default=None, help="inert: the cluster_run --arm-block key this job was scheduled under (recorded in the JSON)")
    k = sub.add_parser("compass", help="the efferent rotation arm under one vnc arm (GPU)")
    k.add_argument("--arm", required=True, choices=sorted(ARMS)); k.add_argument("--seed", type=int, default=0); k.add_argument("--family", default="vncd", choices=sorted(FAMILIES), help=fam_help)
    k.add_argument("--gains", default="2:15"); k.add_argument("--seconds", type=float, default=10.0); k.add_argument("--skip", type=float, default=3.0)
    k.add_argument("--rate", type=float, default=90.0); k.add_argument("--dna02-hz", type=float, default=20.0); k.add_argument("--sparse", default="warp", choices=["warp", "torch"])
    k.add_argument("--quick", action="store_true"); k.add_argument("--device", default=None); k.add_argument("--cache-dir", default=None); k.add_argument("--out", required=True)
    k.add_argument("--block", default=None, help="inert: the cluster_run --arm-block key (recorded)")
    b = sub.add_parser("bench", help="scripts/benchmark.py sections under arm A or B (GPU)")
    b.add_argument("--arm", required=True, choices=sorted(ARMS)); b.add_argument("--sections", default="rest,taste,smell,walk"); b.add_argument("--family", default="vncd", choices=sorted(FAMILIES), help=fam_help)
    b.add_argument("--fast", action="store_true"); b.add_argument("--eager", action="store_true"); b.add_argument("--device", default=None); b.add_argument("--out", required=True)
    b.add_argument("--block", default=None, help="inert: the cluster_run --arm-block key (recorded)")
    p = sub.add_parser("plan", help="write the one cluster submission")
    p.add_argument("--dir", default="out/vncd"); p.add_argument("--runs", type=int, default=5); p.add_argument("--compass-seeds", type=int, default=3)
    p.add_argument("--draws", type=int, default=2, help="benchmark draws per arm (0 = no bench jobs)"); p.add_argument("--minutes", type=int, default=60)
    p.add_argument("--family", default="vncd", choices=sorted(FAMILIES), help=fam_help); p.add_argument("--name", default=None, help="cluster_run --name (default vncd / vncd3 by family)")
    p.add_argument("--only-seeds", default=None, help="comma list of room seeds to plan (default 0..runs-1)"); p.add_argument("--only-compass-seeds", default=None, help="comma list of compass seeds")
    p.add_argument("--targets", default=None, help="cluster_run --targets (comma list); default = the config's default list"); p.add_argument("--script", default=None, help="file name under --dir (default batch.sh)")
    a = sub.add_parser("analyse", help="the audit's tables (CPU)")
    a.add_argument("--dir", default="out/vncd"); a.add_argument("--out", default="out/vncd/analysis"); a.add_argument("--cache-dir", default=None)
    a.add_argument("--skip-trace", action="store_true"); a.add_argument("--skip-decompose", action="store_true"); a.add_argument("--skip-paths", action="store_true")
    a.add_argument("--trace-arms", default="B,E", help="arms traced against A (each trace is slow)")
    a.add_argument("--only-seeds", default=None, help="comma list: analyse only these run seeds (room and compass), e.g. the runs of one submission")
    q = sub.add_parser("pairs", help="adjacent-arm verdicts, per-frame sidedness, DNa02 decomposition summary, compass flips of the chain (CPU; after analyse)")
    q.add_argument("--dir", default="out/vncd3"); q.add_argument("--out", default="out/vncd3/analysis"); q.add_argument("--analysis", default=None, help="where analyse wrote decompose_* / compass_flip_* (default --out)")
    q.add_argument("--cache-dir", default=None); q.add_argument("--only-seeds", default=None, help="comma list: only these run seeds (e.g. the runs of one submission)")
    args = ap.parse_args(argv)
    return {"room": cmd_room, "compass": cmd_compass, "bench": cmd_bench, "plan": cmd_plan, "analyse": cmd_analyse, "pairs": cmd_pairs}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
