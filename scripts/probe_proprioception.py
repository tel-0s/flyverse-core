"""Smoke of the opt-in proprioception transducer (flyverse.senses.Proprioception; docs/audits/proprioception_transducer.md):
BatchSim flies in the fenced room, the sense on ('all', or any spec) or off (the shipped path), recording every frame the
rates of the afferent channels themselves, of the ascending / VNC / DN watch types (AN04B003, AN07B035, AN07B037_a,
AN06A026, PS196_b, IN12B014, IN19A003, DNa02 L / R) and the motor readout.

    python scripts/probe_proprioception.py --batch 4 --seconds 5 --proprioception all --seed 0 --out out/proprio/on_r0
    python scripts/probe_proprioception.py --batch 4 --seconds 5 --proprioception off --seed 0 --out out/proprio/off_r0
    python scripts/probe_proprioception.py report --runs "out/proprio/on_r*.json" "out/proprio/off_r*.json"
    python scripts/probe_proprioception.py structure --out out/proprio/structure.json          # CPU: cell sets and the side rule

Writes <out>.json (provenance from flyverse.interp.common.provenance: resolved LIFParams / OpticParams / type_path_gain,
the realised device, the cache fingerprint, the git commit; the channel cell counts and rate-law parameters; per-channel
commanded and measured afferent Hz; the watch-type rates per side with the per-fly scatter) and <out>_rec.npz /
<out>_rec.json (the flyverse.interp.recording of every recorded cell, every frame). `report` tabulates several runs per arm with the run scatter (the run is the replicate
unit; `common.compare` for on-vs-off when >= 4 runs per arm). This is a smoke, not a behaviour probe: nothing here is a
verdict on turning.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

WATCH = ["AN04B003", "AN07B035", "AN07B037_a", "AN06A026", "PS196_b", "IN12B014", "IN19A003", "DNa02"]


def record(args):
    import torch
    from flyverse import senses, world
    from flyverse.batch_sim import BatchSim
    from flyverse.interp import common
    if args.device is None or args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "no CUDA device (run this on the cluster, or pass --device cpu for a synthetic smoke)"
    spec = None if args.proprioception in (None, "", "off", "none") else args.proprioception
    seeds = list(range(args.seed * 100, args.seed * 100 + args.batch))
    info = world.make_room(0, "all")[1]
    start = (-.15, .15, float(info["table_top_z"]))
    gpu = args.device is None or args.device.startswith("cuda")
    sim = BatchSim(args.batch, seed=args.seed, seeds=seeds, start=start, program="none", fruit_set="all", fence=True,
                   cuda_graphs=gpu, cuda_kernels=gpu if gpu else None, event_driven=gpu if gpu else None,
                   cuda_sparse=args.cuda_sparse, device=args.device, proprioception=spec)
    for seed, fly, m in zip(sim.seeds, sim.flies, sim.metabolisms):
        fly.heading = np.random.default_rng(seed).uniform(-np.pi, np.pi); m.energy = args.energy
    c = sim.fb.c
    sense = sim.fb.proprioception_sense if spec is not None else senses.Proprioception(c, "all")   # off: the same cells, recorded
    counts = sense.counts()
    lp, op = sim.fb.brain.p, (sim.fb.optic.p if sim.fb.optic is not None else None)
    print(f"proprioception {spec or 'off'}: channels {sense.channels} coriolis {sense.haltere_coriolis} params {sense.params} mn_ref {sense.mn_ref_hz} Hz",
          flush=True)
    print(f"cells per channel: " + "; ".join(f"{ch} n {v['n']} (L {v['L']} / R {v['R']} / both {v['both']}; vnc_sensory {v['vnc_sensory']}, "
          f"sensory_ascending {v['sensory_ascending']}; side from instance {v['side_source']['instance']}, laterality {v['side_source']['laterality']})"
          for ch, v in counts.items()), flush=True)
    print(f"BatchSim B={sim.B} neurons={c.n:,} device={sim.fb.brain.device} env seeds={sim.seeds} receptor_model {lp.receptor_model} "
          f"type_path_gain {lp.type_path_gain}", flush=True)
    watch_idx = {t: c.select(type=t) for t in WATCH}
    chan_idx = {ch: sense.idx[ch] for ch in sense.channels}
    all_idx = np.unique(np.concatenate([v for v in watch_idx.values()] + [v for v in chan_idx.values()]))
    rec = common.Recorder(c, all_idx, quantities=("rate_hz",))
    n = int(round(args.seconds * 100)); dt = sim.fb.brain.p.dt
    commanded = {ch: np.zeros((n, sim.B)) for ch in chan_idx}
    airborne = np.zeros((n, sim.B), bool); yaw = np.zeros((n, sim.B))
    pos_idx = {ch: sim.fb.brain._idx(idx) for ch, idx in chan_idx.items()}
    t0 = time.time()
    for k in range(n):
        sim.step()
        rec.capture(sim.fb, motor=sim.motor)
        base = sim.fb._base_poisson
        for ch, ti in pos_idx.items():
            commanded[ch][k] = (base[:, ti].mean(dim=1) * (1000.0 / dt)).cpu().numpy()
        airborne[k] = [f.airborne for f in sim.flies]; yaw[k] = [f.yaw_rate for f in sim.flies]
        if (k + 1) % 100 == 0:
            print(f"  t {(k + 1) / 100:.0f} s airborne {airborne[k].mean():.2f}", flush=True)
    wall = time.time() - t0
    recording = rec.finish(meta={"protocol": "proprioception_smoke", "proprioception": spec, "watch": WATCH})
    prov = common.provenance(c, lp, op, fb=sim.fb, device=args.device, seeds=[args.seed], env_seeds=seeds, batch=sim.B,
                             backend=dict(cuda_graphs=gpu, cuda_kernels=gpu, event_driven=gpu, cuda_sparse=args.cuda_sparse),
                             stimulus={"protocol": "proprioception_smoke", "params": {"proprioception": spec, "channels": list(sense.channels),
                                       "haltere_coriolis": sense.haltere_coriolis, "rate_params": sense.params, "mn_ref_hz": sense.mn_ref_hz,
                                       "counts": counts, "seconds": args.seconds, "skip_s": args.skip, "fence": True, "program": "none"},
                                       "control": {"proprioception": "off"}})
    out = summarise(recording, chan_idx, watch_idx, commanded, airborne, yaw, c, args.skip)
    out.update(arm="off" if spec is None else spec, seed=args.seed, env_seeds=seeds, batch=sim.B, seconds=args.seconds, wall_s=wall,
               counts=counts, rate_params=sense.params, mn_ref_hz=sense.mn_ref_hz, channels=list(sense.channels),
               haltere_coriolis=sense.haltere_coriolis, provenance=prov, device=str(sim.fb.brain.device),
               files={"generator": "scripts/probe_proprioception.py record", "recording": str(args.out) + "_rec.npz"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    recording.save(Path(str(args.out) + "_rec"))                       # <out>_rec.npz + <out>_rec.json (the recording meta)
    Path(str(args.out) + ".json").write_text(json.dumps(common.to_jsonable(out), indent=1), encoding="utf-8")
    print_run(out)
    print(f"wrote {args.out}.json (device {out['device']}, wall {wall:.0f} s)", flush=True)


def summarise(recording, chan_idx, watch_idx, commanded, airborne, yaw, c, skip_s):
    rate = recording.quantities["rate_hz"]                      # (T, B, n)
    T = rate.shape[0]; k0 = min(int(round(skip_s * 100)), max(T - 1, 0))
    pos = {int(i): j for j, i in enumerate(recording.idx)}
    side = c.neurons.somaSide.fillna("?").to_numpy()
    chan = {}
    for ch, idx in chan_idx.items():
        cols = [pos[int(i)] for i in idx]
        per_fly = rate[k0:, :, cols].mean(axis=(0, 2))
        chan[ch] = dict(n=len(idx), commanded_hz_mean=float(commanded[ch][k0:].mean()), commanded_hz_max=float(commanded[ch].max()),
                        measured_hz_mean=float(per_fly.mean()), measured_hz_fly_sd=float(per_fly.std(ddof=1)) if len(per_fly) > 1 else 0.0,
                        measured_hz_per_fly=[float(v) for v in per_fly])
    watch = []
    for t, idx in watch_idx.items():
        for s in ("L", "R"):
            sel = idx[side[idx] == s]
            if not len(sel): continue
            cols = [pos[int(i)] for i in sel]
            per_fly = rate[k0:, :, cols].mean(axis=(0, 2))
            watch.append(dict(type=t, side=s, n_cells=int(len(sel)), hz_mean=float(per_fly.mean()),
                              hz_fly_sd=float(per_fly.std(ddof=1)) if len(per_fly) > 1 else 0.0,
                              hz_max_cell=float(rate[k0:, :, cols].max()), hz_per_fly=[float(v) for v in per_fly]))
    motor = {k: float(v[k0:].mean()) for k, v in recording.motor.items() if k in ("leg_L", "leg_R", "haltere", "turn_L", "turn_R", "power", "fwd_dn")}
    return dict(window_skip_s=skip_s, frames=T, channels_measured=chan, watch=watch, motor_mean=motor,
                airborne_frac=float(airborne[k0:].mean()), yaw_rate_abs_mean_deg_s=float(np.degrees(np.abs(yaw[k0:]).mean())))


def print_run(out):
    print(f"arm {out['arm']} seed {out['seed']} (skip {out['window_skip_s']} s; airborne {out['airborne_frac']:.3f}; |yaw| {out['yaw_rate_abs_mean_deg_s']:.2f} deg/s; "
          f"motor {', '.join(f'{k} {v:.2f}' for k, v in out['motor_mean'].items())})")
    print("  channel        n  commanded Hz  measured Hz (fly sd)")
    for ch, v in out["channels_measured"].items():
        print(f"  {ch:12s} {v['n']:4d}  {v['commanded_hz_mean']:10.2f}   {v['measured_hz_mean']:8.2f} ({v['measured_hz_fly_sd']:.2f})")
    print("  type        side  n   Hz mean (fly sd)   max cell")
    for w in out["watch"]:
        print(f"  {w['type']:10s} {w['side']}  {w['n_cells']:2d}  {w['hz_mean']:8.3f} ({w['hz_fly_sd']:.3f})   {w['hz_max_cell']:.1f}")


def report(args):
    from flyverse.interp import common
    runs = {}
    for pattern in args.runs:
        for p in sorted(glob.glob(pattern)):
            if p.endswith(".json") and not p.endswith(".meta.json"):
                d = json.loads(Path(p).read_text(encoding="utf-8"))
                if "arm" in d: runs.setdefault(d["arm"], []).append((p, d))
    if not runs: sys.exit("no run JSONs matched")
    rows = []
    for arm, items in runs.items():
        devices = sorted({d["device"] for _, d in items})
        print(f"arm {arm}: {len(items)} run(s) {[Path(p).name for p, _ in items]} device {devices}")
        for ch in items[0][1]["channels_measured"]:
            cmd = [d["channels_measured"][ch]["commanded_hz_mean"] for _, d in items]; meas = [d["channels_measured"][ch]["measured_hz_mean"] for _, d in items]
            rows.append(dict(arm=arm, key=f"channel.{ch}", n_runs=len(items), commanded_hz=np.mean(cmd), measured_hz=np.mean(meas),
                             measured_run_sd=np.std(meas, ddof=1) if len(meas) > 1 else 0.0, runs=meas))
        keys = [(w["type"], w["side"]) for w in items[0][1]["watch"]]
        for t, s in keys:
            vals = [next(w["hz_mean"] for w in d["watch"] if (w["type"], w["side"]) == (t, s)) for _, d in items]
            rows.append(dict(arm=arm, key=f"{t}_{s}", n_runs=len(items), commanded_hz=np.nan, measured_hz=np.mean(vals),
                             measured_run_sd=np.std(vals, ddof=1) if len(vals) > 1 else 0.0, runs=vals))
        for k in items[0][1]["motor_mean"]:
            vals = [d["motor_mean"][k] for _, d in items]
            rows.append(dict(arm=arm, key=f"motor.{k}", n_runs=len(items), commanded_hz=np.nan, measured_hz=np.mean(vals),
                             measured_run_sd=np.std(vals, ddof=1) if len(vals) > 1 else 0.0, runs=vals))
    import pandas as pd
    df = pd.DataFrame(rows)
    table = df.pivot(index="key", columns="arm", values="measured_hz")
    sd = df.pivot(index="key", columns="arm", values="measured_run_sd")
    n = df.pivot(index="key", columns="arm", values="n_runs")
    lines = []
    for key in table.index:
        cells = [f"{table.loc[key, a]:9.3f} +- {sd.loc[key, a]:6.3f} (n {int(n.loc[key, a])})" if a in table.columns and np.isfinite(table.loc[key, a]) else " " * 30 for a in table.columns]
        verdict = ""
        if "off" in table.columns:
            for a in table.columns:
                if a == "off": continue
                stim = df[(df.arm == a) & (df.key == key)].runs.iloc[0]; null = df[(df.arm == "off") & (df.key == key)].runs.iloc[0]
                cmp_ = common.compare(np.asarray(stim, float), np.asarray(null, float))
                verdict += f" {a} vs off: diff {cmp_['diff']:+.3f} z {cmp_['z']:+.2f} p {cmp_['p']} -> {cmp_['verdict']}"
        lines.append(f"{key:22s} " + " | ".join(cells) + verdict)
    header = f"{'key':22s} " + " | ".join(f"{a:^30s}" for a in table.columns)
    print(header); print("\n".join(lines))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False); print(f"wrote {args.out}")


def structure(args):
    """CPU, no rollout: the channel cell sets of the shipped cache, the side rule's two sources and their agreement, the
    per-type L / R split, and which side each ascending watch cell's proprioceptive input comes from (the numbers of
    docs/audits/proprioception_transducer.md section 1)."""
    from flyverse import connectome
    from flyverse.senses import Proprioception, laterality
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else connectome.load(verbose=False)
    sense = Proprioception(c, "all")
    n = c.neurons; types = n.type.fillna("").to_numpy().astype(str); sc = n.superclass.fillna("").to_numpy().astype(str)
    nerve = n.entryNerve.fillna("").to_numpy().astype(str); soma = n.somaSide.to_numpy()
    tL, tR = np.flatnonzero(soma == "L"), np.flatnonzero(soma == "R")
    out = {"counts": sense.counts(), "per_type": {}, "instance_vs_laterality": {}, "ascending_inputs": []}
    print(f"cache n {c.n:,}; proprioceptors with somaSide: {int(pd.notna(soma[(n['class'] == 'mechanosensory_proprioceptive').to_numpy()]).sum())}")
    for ch, idx in sense.idx.items():
        side = sense.side[ch]; inst = sense._instance_side(idx); lat = laterality(c.reference, c.reference.index_of(n.bodyId.to_numpy()[idx]), tL, tR)
        has = inst != 0
        agree = int(((inst == 1) & (lat > 0)).sum() + ((inst == -1) & (lat < 0)).sum()); disagree = int(((inst == 1) & (lat < 0)).sum() + ((inst == -1) & (lat > 0)).sum())
        weak = int((has & (np.abs(lat) <= 0.2)).sum())
        out["instance_vs_laterality"][ch] = dict(instance_labelled=int(has.sum()), agree_sign=agree, disagree_sign=disagree, weak_laterality_among_labelled=weak)
        cnt = out["counts"][ch]
        print(f"{ch}: n {cnt['n']} L {cnt['L']} / R {cnt['R']} / both {cnt['both']}; vnc_sensory {cnt['vnc_sensory']} sensory_ascending {cnt['sensory_ascending']}; "
              f"side from instance {cnt['side_source']['instance']} / laterality {cnt['side_source']['laterality']}; on the instance-labelled cells the laterality sign "
              f"agrees {agree}, disagrees {disagree} (|lat| <= 0.2 on {weak})")
        print("   entry nerves: " + ", ".join(f"{k} {v}" for k, v in pd.Series(nerve[idx]).value_counts().items()))
        tab = {}
        for t in pd.Series(types[idx]).value_counts().index:
            m = types[idx] == t
            tab[t or "<untyped>"] = dict(n=int(m.sum()), L=int((side[m] > 0).sum()), R=int((side[m] < 0).sum()), both=int((side[m] == 0).sum()),
                                         nerves=sorted(set(nerve[idx][m])), superclass=sorted(set(sc[idx][m])))
        out["per_type"][ch] = tab
        shown = [k for k in tab if tab[k]["n"] >= 10 or k in ("SNpp39", "SNpp50", "SNpp60", "SNpp52", "SNppxx", "SApp23", "SNpp45", "SNpp19", "SNpp53", "SApp")]
        print("   types: " + "; ".join(f"{k} {v['n']} ({v['L']}/{v['R']}/{v['both']}; {'/'.join(v['nerves'])})" for k, v in tab.items() if k in shown))
    pro = (n["class"] == "mechanosensory_proprioceptive").to_numpy()
    Wabs = abs(c.W).tocsr()
    all_idx = np.concatenate(list(sense.idx.values())); all_side = np.concatenate(list(sense.side.values()))
    side_of = dict(zip(all_idx.tolist(), all_side.tolist()))
    print("ascending / VNC watch cells: proprioceptive input synapses by the afferents' assigned side")
    for t in WATCH:
        for j in c.select(type=t):
            row = Wabs[j].toarray().ravel(); pre = np.flatnonzero(row > 0); pre = pre[pro[pre]]
            syn_L = float(sum(row[p] for p in pre if side_of.get(int(p), 0) > 0)); syn_R = float(sum(row[p] for p in pre if side_of.get(int(p), 0) < 0))
            syn_0 = float(sum(row[p] for p in pre if int(p) in side_of and side_of[int(p)] == 0)); other = float(sum(row[p] for p in pre if int(p) not in side_of))
            rec = dict(type=t, index=int(j), bodyId=int(n.bodyId[j]), somaSide=str(soma[j]), n_pre=int(len(pre)), syn_from_L=syn_L, syn_from_R=syn_R, syn_from_unsided=syn_0, syn_from_unselected_proprio=other)
            out["ascending_inputs"].append(rec)
            print(f"   {t:10s} {str(soma[j])} idx {j:6d}: {len(pre):3d} proprioceptive inputs; synapses from L-assigned {syn_L:6.0f}, R-assigned {syn_R:6.0f}, unsided {syn_0:5.0f}, outside the channels {other:5.0f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8"); print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("structure"); s.add_argument("--cache-dir", default=None); s.add_argument("--out", default=None)
    r = sub.add_parser("record"); r.add_argument("--batch", type=int, default=4); r.add_argument("--seconds", type=float, default=5.0)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--proprioception", default="all"); r.add_argument("--energy", type=float, default=.9)
    r.add_argument("--skip", type=float, default=1.0, help="seconds dropped from the window means"); r.add_argument("--device", default=None)
    r.add_argument("--cuda-sparse", default="torch"); r.add_argument("--out", required=True)
    p = sub.add_parser("report"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", default=None)
    argv = sys.argv[1:]
    if argv and argv[0] not in ("record", "report", "structure"): argv = ["record"] + argv
    args = ap.parse_args(argv)
    if args.cmd == "report": report(args)
    elif args.cmd == "structure": structure(args)
    else: record(args)


if __name__ == "__main__":
    main()
