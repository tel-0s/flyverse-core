"""Does the plain fly (no program) still wander, or does it walk straight off the table?  Bisect over the session-10
default changes with BatchSim: 16 flies, plain body, no fence, the full table, 60 s.

    python scripts/probe_walk_straightness.py --arm default|off|off-damped|default-damped|session9 --seed 0 --out out/ws_default_s0.json

Arms: default = the shipped model; off = receptor_model None on the shipped gains; off-damped = None + the GF x0.3 damping
(the pre-session-10 weights on the current cache); default-damped = the round-3/4 default; session9 = off-damped on the
pre-TYPE_NT_OVERRIDE cache (--cache-dir, e.g. out/cache_pre_override or the cluster's backup) = the session-9 weights.
Per fly: yaw-rate SD, straightness (net displacement / path), time to leave the table top, min distance to a fruit, hops,
mean |DNa02 R-L| / |leg L-R| from the readout, plus the walking trajectory every 0.5 s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from flyverse import body, brain, connectome, world  # noqa: E402
from flyverse.batch_sim import BatchSim  # noqa: E402

ARMS = {
    "default":        dict(receptor="default", gains="shipped"),
    "off":            dict(receptor=None, gains="shipped"),
    "off-damped":     dict(receptor=None, gains="damped"),
    "default-damped": dict(receptor="default", gains="damped"),
    "session9":       dict(receptor=None, gains="damped"),             # + --cache-dir pre-override
}


def patch_lif(receptor, gains):
    """Every brain.LIFParams built from here on (BatchSim's included) carries the arm's receptor model and type gains."""
    L = brain.LIFParams
    damped = list(brain.GF_DAMPED_TYPE_PATH_GAIN)

    def make(**kw):
        if receptor != "default":
            kw.setdefault("receptor_model", receptor)
        if gains == "damped":
            kw.setdefault("type_path_gain", damped)
        return L(**kw)
    brain.LIFParams = make
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--program", default="none")
    ap.add_argument("--fence", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="connectome cache to use (the session9 arm wants the pre-override backup)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cuda-sparse", default="torch")
    ap.add_argument("--proprioception", default=None, metavar="SPEC",
                    help="opt-in senses.Proprioception ('all', a comma list of channels, 'all+haltere_coriolis' for the labelled "
                         "stop-gap control arm; round 3: '+leg_cycle' reads the body's stance / swing cycle per leg, '+haltere_sided' "
                         "the side-split haltere MN readout -- docs/audits/body_sided_state.md); default off = the shipped path "
                         "(docs/audits/proprioception_transducer.md)")
    ap.add_argument("--leg-cycle", action="store_true",
                    help="attach body.LegCycle to the batch body (a readout of the realised speed / yaw; the walk itself is untouched). "
                         "Attached automatically when --proprioception names leg_cycle.")
    args = ap.parse_args()
    import torch
    assert torch.cuda.is_available(), "no CUDA device (run this on the cluster)"
    arm = ARMS[args.arm]
    L0 = patch_lif(arm["receptor"], arm["gains"])
    c = connectome.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else None
    seeds = list(range(args.seed * 100, args.seed * 100 + args.batch))
    sim = BatchSim(args.batch, seed=args.seed, seeds=seeds, c=c, program=args.program, fruit_set="all", fence=args.fence,
                   cuda_graphs=True, cuda_kernels=True, event_driven=True, cuda_sparse=args.cuda_sparse,
                   proprioception=args.proprioception)
    lp = sim.fb.brain.p
    sense = getattr(sim.fb, "proprioception_sense", None)
    if args.leg_cycle or (sense is not None and sense.leg_cycle):
        sim.body.leg_cycle = body.LegCycle()
    print(f"arm {args.arm}: receptor_model {lp.receptor_model} ({lp.receptor_net_rule}), type_path_gain {lp.type_path_gain}, "
          f"cache {args.cache_dir or 'default'}, sum|W| {float(abs(sim.fb.c.W).sum()):,.0f}, device {sim.fb.brain.device}, B {args.batch}, "
          f"proprioception {sense.spec if sense is not None else 'off'}, leg cycle {'on' if sim.body.leg_cycle is not None else 'off'}")
    info = world.make_room(0, "all")[1]; top_z = float(info["table_top_z"]); ext = info["table_extent"]
    fruit = np.array([s.center for s in sim.fb_world_spheres()]) if hasattr(sim, "fb_world_spheres") else None
    B = args.batch; n = int(args.seconds * 100)
    heading = np.zeros((n, B)); pos = np.zeros((n, B, 3)); on_top = np.zeros((n, B), bool); airborne = np.zeros((n, B), bool)
    dna02 = np.zeros((n, B)); legasym = np.zeros((n, B)); dist_fruit = np.zeros((n, B))
    t0 = time.time()
    for k in range(n):
        sim.step()
        for i, f in enumerate(sim.flies):
            heading[k, i] = float(f.heading); pos[k, i] = f.pos; airborne[k, i] = bool(f.airborne)
            on_top[k, i] = (abs(f.pos[0]) <= ext[0] + 1e-3) and (abs(f.pos[1]) <= ext[1] + 1e-3) and abs(f.pos[2] - top_z) < 0.01 if len(ext) >= 2 else abs(f.pos[2] - top_z) < 0.01
            cmd = sim.commands[i] if sim.commands else {}
            r = cmd.get("rates", {}) if isinstance(cmd, dict) else {}
            dna02[k, i] = float(r.get("DNa02 R-L", r.get("DNa02_R-L", np.nan))) if r else np.nan
            legasym[k, i] = float(r.get("leg L-R", r.get("legs L-R", np.nan))) if r else np.nan
        _names, d = sim.nearest_fruit()                                 # (names, surface distances) per row
        dist_fruit[k] = np.asarray(d, dtype=float).ravel()[:B]
        if k % 1000 == 999:
            print(f"  t {(k + 1) / 100:5.0f} s  on table {on_top[k].mean():.2f}  airborne {airborne[k].mean():.2f}  min fruit dist {dist_fruit[k].min() * 100:.1f} cm", flush=True)
    dh = np.diff(np.unwrap(heading, axis=0), axis=0) * 100.0                               # rad/s per frame
    walking = ~airborne[1:]
    rows = []
    for i in range(B):
        w = walking[:, i]
        path = float(np.linalg.norm(np.diff(pos[:, i, :2], axis=0), axis=1).sum())
        net = float(np.linalg.norm(pos[-1, i, :2] - pos[0, i, :2]))
        left = np.flatnonzero(~on_top[:, i])
        rows.append({"row": i, "seed": seeds[i], "yaw_sd_deg_s": float(np.degrees(np.std(dh[w, i]))) if w.any() else None,
                     "yaw_mean_abs_deg_s": float(np.degrees(np.mean(np.abs(dh[w, i])))) if w.any() else None,
                     "path_m": path, "net_m": net, "straightness": net / path if path > 0 else None,
                     "left_table_s": float(left[0] / 100.0) if len(left) else None, "frac_on_table": float(on_top[:, i].mean()),
                     "min_fruit_cm": float(dist_fruit[:, i].min() * 100), "frames_within_2cm": int((dist_fruit[:, i] < 0.02).sum()),
                     "hops": int(np.sum(np.diff(airborne[:, i].astype(int)) == 1)),
                     "dna02_abs_mean": float(np.nanmean(np.abs(dna02[:, i]))) if np.isfinite(dna02[:, i]).any() else None,
                     "leg_abs_mean": float(np.nanmean(np.abs(legasym[:, i]))) if np.isfinite(legasym[:, i]).any() else None,
                     "track": [[round(float(x), 4) for x in pos[k, i]] + [round(float(heading[k, i]), 3)] for k in range(0, n, 50)]})
    def agg(key):
        v = [r[key] for r in rows if r[key] is not None]
        return {"mean": float(np.mean(v)), "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)} if v else None
    summary = {k: agg(k) for k in ("yaw_sd_deg_s", "yaw_mean_abs_deg_s", "straightness", "path_m", "left_table_s", "frac_on_table", "min_fruit_cm", "frames_within_2cm", "hops", "dna02_abs_mean", "leg_abs_mean")}
    summary["n_left_table"] = int(sum(r["left_table_s"] is not None for r in rows))
    out = {"arm": args.arm, "seed": args.seed, "seconds": args.seconds, "program": args.program, "fence": args.fence, "cache_dir": args.cache_dir,
           "proprioception": sim.proprioception, "leg_cycle": None if sim.body.leg_cycle is None else {k: v for k, v in vars(sim.body.leg_cycle).items()},
           "lif": {"receptor_model": lp.receptor_model, "receptor_net_rule": lp.receptor_net_rule, "type_path_gain": lp.type_path_gain},
           "sum_abs_W": float(abs(sim.fb.c.W).sum()), "wall_s": time.time() - t0, "summary": summary, "rows": rows,
           "cmd_keys": sorted(sim.commands[0].keys()) if sim.commands and isinstance(sim.commands[0], dict) else None,
           "rate_keys": sorted(sim.commands[0].get("rates", {}).keys()) if sim.commands and isinstance(sim.commands[0], dict) else None}
    print(f"\n{args.arm} seed {args.seed}: yaw SD {summary['yaw_sd_deg_s']['median']:.1f} deg/s (median over flies), straightness {summary['straightness']['median']:.2f}, "
          f"left the table {summary['n_left_table']}/{B} (median t {summary['left_table_s']['median'] if summary['left_table_s'] else float('nan'):.0f} s), "
          f"min fruit dist {summary['min_fruit_cm']['median']:.1f} cm, hops {summary['hops']['mean']:.2f}/fly, wall {out['wall_s']:.0f} s")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
    brain.LIFParams = L0


if __name__ == "__main__":
    main()
