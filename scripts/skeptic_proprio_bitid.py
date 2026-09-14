"""SKEPTIC check (build:transducer): is the shipped, proprioception-OFF path bit-identical to the shipped commit?

Not part of the task's deliverables -- written by the reviewing pass. One BatchSim room rollout is run against a chosen
source tree (the working tree, or a `git archive HEAD flyverse` snapshot placed in `_skepthead/`), with no
`proprioception` argument at all (so the HEAD signature works) or with `proprioception=None` (`--pass-kwarg`, what
`scripts/batch_sustain.py` now passes by default). The run dumps per-row hops / GF / positions like
`scripts/batch_sustain.py` plus md5 hashes of the whole brain state at checkpoints, so two trees can be compared byte
for byte. A second working-tree run at the same seed is the determinism control: if it does not match the first, a
head-vs-work mismatch says nothing.

    python scripts/skeptic_proprio_bitid.py run --tree _skepthead --batch 16 --minutes 1 --out out/x/head.json
    python scripts/skeptic_proprio_bitid.py run --tree .          --batch 16 --minutes 1 --out out/x/work.json
    python scripts/skeptic_proprio_bitid.py compare out/x/head.json out/x/work.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

STATE = ("v", "g", "rate", "poisson_p", "spike_counts")


def md5_of(arrays):
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def brain_hash(fb):
    out = {}
    for k in STATE:
        t = getattr(fb.brain, k, None)
        if t is None:
            continue
        out[k] = md5_of([t.detach().cpu().numpy()])
    return out


def run(args):
    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(tree))
    import torch
    from flyverse.batch_sim import BatchSim
    import flyverse
    loaded = Path(flyverse.__file__).resolve()
    if tree not in loaded.parents:
        sys.exit(f"imported flyverse from {loaded}, not from the requested tree {tree}")
    if args.device is None or str(args.device).startswith("cuda"):
        assert torch.cuda.is_available(), "no CUDA device (run this on the cluster)"
    gpu = args.device is None or str(args.device).startswith("cuda")
    kw = {}
    if args.pass_kwarg:
        kw["proprioception"] = None                      # exactly what scripts/batch_sustain.py passes by default
    sim = BatchSim(args.batch, seed=args.seed, start=(-.15, .15, .75), program="none", fruit_set="all", fence=True,
                   cuda_graphs=gpu, cuda_kernels=gpu if gpu else None, event_driven=gpu if gpu else None,
                   cuda_sparse=args.cuda_sparse, device=args.device, **kw)
    for seed, fly, m in zip(sim.seeds, sim.flies, sim.metabolisms):
        fly.heading = np.random.default_rng(seed).uniform(-np.pi, np.pi)
        m.energy = args.energy
    print(f"tree {tree} -> flyverse {loaded}; B {sim.B} neurons {sim.c.n:,} device {sim.fb.device} "
          f"kwarg {'proprioception=None' if args.pass_kwarg else '(absent)'} sum|W| {float(abs(sim.c.W).sum()):,.0f}", flush=True)
    count = max(1, round(args.minutes * 60_000 / sim.FRAME_MS))
    hops = np.zeros(sim.B, dtype=int)
    previous_air = np.array([f.airborne for f in sim.flies])
    gf_max_walk = np.zeros(sim.B)
    power_max = np.zeros(sim.B)
    checkpoints = {}
    t0 = time.perf_counter()
    for k in range(count):
        sim.step()
        gf = np.array([w["gf"] for w in sim.wcommands])
        power = np.array([w["power"] for w in sim.wcommands])
        gf_max_walk = np.where(previous_air, gf_max_walk, np.maximum(gf_max_walk, gf))
        power_max = np.maximum(power_max, power)
        current_air = np.array([f.airborne for f in sim.flies])
        hops += current_air & ~previous_air
        previous_air = current_air
        if (k + 1) % args.hash_every == 0 or k + 1 == count:
            checkpoints[k + 1] = dict(brain=brain_hash(sim.fb),
                                      pos=md5_of([np.array([f.pos for f in sim.flies], dtype=np.float64)]),
                                      heading=md5_of([np.array([f.heading for f in sim.flies], dtype=np.float64)]))
            print(f"  frame {k+1}: rate {checkpoints[k+1]['brain'].get('rate')} pos {checkpoints[k+1]['pos']}", flush=True)
    wall = time.perf_counter() - t0
    escape, voluntary = sim.hops_escape.copy(), sim.hops_voluntary.copy()
    rows = [dict(row=i, environment_seed=sim.seeds[i], energy=float(m.energy), meals=int(m.meals), hops=int(hops[i]),
                 hops_escape=int(escape[i]), hops_voluntary=int(voluntary[i]), gf_max_walk_hz=float(gf_max_walk[i]),
                 power_max_hz=float(power_max[i]), position=[float(v) for v in sim.flies[i].pos],
                 heading=float(sim.flies[i].heading))
            for i, m in enumerate(sim.metabolisms)]
    result = dict(tree=str(tree), flyverse_file=str(loaded), pass_kwarg=bool(args.pass_kwarg), batch=sim.B,
                  seed=args.seed, frames=count, device=str(sim.fb.device),
                  device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                  torch=torch.__version__, wall_s=wall,
                  cache_sum_abs_W=float(abs(sim.c.W).sum()),
                  hops_total=int(hops.sum()), hops_escape_total=int(escape.sum()), hops_voluntary_total=int(voluntary.sum()),
                  gf_max_walk_median_hz=float(np.median(gf_max_walk)),
                  rows=rows, checkpoints={str(k): v for k, v in checkpoints.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {args.out} (device {result['device_name']}, wall {wall:.0f} s, hops {result['hops_total']}, "
          f"GF walk median {result['gf_max_walk_median_hz']:.2f} Hz)", flush=True)


def compare(args):
    runs = [(p, json.loads(Path(p).read_text(encoding="utf-8"))) for p in args.files]
    base_p, base = runs[0]
    print(f"reference {base_p}: tree {base['tree']} kwarg {base['pass_kwarg']} device {base['device_name']} frames {base['frames']}")
    for p, d in runs[1:]:
        same_state = all(base["checkpoints"][k] == d["checkpoints"].get(k) for k in base["checkpoints"])
        first_bad = next((k for k in sorted(base["checkpoints"], key=int) if base["checkpoints"][k] != d["checkpoints"].get(k)), None)
        same_rows = base["rows"] == d["rows"]
        print(f"\n{p}: tree {d['tree']} kwarg {d['pass_kwarg']} device {d['device_name']}")
        print(f"  every checkpoint hash equal: {same_state}" + ("" if same_state else f" (first difference at frame {first_bad})"))
        print(f"  per-row table identical:     {same_rows}")
        for k in ("hops_total", "hops_escape_total", "hops_voluntary_total", "gf_max_walk_median_hz", "cache_sum_abs_W"):
            print(f"  {k:24s} {base[k]} vs {d[k]}" + ("" if base[k] == d[k] else "   <-- DIFFERS"))
        if not same_rows:
            for a, b in zip(base["rows"], d["rows"]):
                if a != b:
                    print(f"  row {a['row']}: {a} vs {b}")
        print(f"  VERDICT: {'BIT-IDENTICAL' if same_state and same_rows else 'NOT identical'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True, help="directory containing the flyverse package to import")
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--minutes", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--energy", type=float, default=.9)
    r.add_argument("--device", default=None)
    r.add_argument("--cuda-sparse", default="torch")
    r.add_argument("--hash-every", type=int, default=500)
    r.add_argument("--pass-kwarg", action="store_true", help="pass proprioception=None explicitly (the batch_sustain default)")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    args = ap.parse_args()
    (run if args.cmd == "run" else compare)(args)


if __name__ == "__main__":
    main()
