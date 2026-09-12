"""Round-4 `slowdet` batch: pin the slow term's non-determinism and separate the --dopamine-lead confound.

Generates the 12 commands of ONE cluster batch and submits them through scripts/cluster_run.py (all concurrent,
one run directory, `--fetch out/`).  Every arm is `--eager --sections walk,motion --seeds 0,1,2 --receptor-model full
--receptor-net-rule abs --receptor-gain 1,1,1` (the round-3 slow-term protocol on the adopted `abs` weights, with
`1,1,1` making `full`'s fast weights identical to the default `sign` ones); the arms differ only in

  det_dop  x3  --deterministic, --slow-mode threshold --slow-gain-monoamine 0.01 --dopamine-lead dop1r1
  nd_dop   x2  the same without --deterministic
  det_ctl  x2  --deterministic, term off (--slow-gain-monoamine 0), shipped table
  nd_ctl   x2  the same without --deterministic
  det_ecr  x2  --deterministic, threshold 0.01, shipped (DopEcR-led) table -- the dop1r1 control
  nd_ecr   x1  the same without --deterministic (insurance: if --deterministic raises, this arm still
               pairs with nd_dop for the dopamine-lead comparison)

The deterministic commands export CUBLAS_WORKSPACE_CONFIG=:4096:8 before python, as cuBLAS requires it to be set
before its first handle is created.  No --cache-dir anywhere (the cluster's shared TYPE_NT_OVERRIDE cache).

    python scripts/slowdet_batch.py --print          # just the commands
    python scripts/slowdet_batch.py                  # submit the batch (cluster_run.py --name r4-slowdet)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BASE = ("--eager --sections walk,motion --seeds 0,1,2 "
        "--receptor-model full --receptor-net-rule abs --receptor-gain 1,1,1")
TERM = "--slow-mode threshold --slow-gain-monoamine 0.01"
OFF = "--slow-gain-monoamine 0"

# (tag, replicates, deterministic, extra flags)
ARMS = [
    ("det_dop", 3, True,  f"{TERM} --dopamine-lead dop1r1"),
    ("nd_dop",  2, False, f"{TERM} --dopamine-lead dop1r1"),
    ("det_ctl", 2, True,  OFF),
    ("nd_ctl",  2, False, OFF),
    ("det_ecr", 2, True,  TERM),
    ("nd_ecr",  1, False, TERM),
]


def commands() -> list[str]:
    out = []
    for tag, n, det, extra in ARMS:
        for i in range(1, n + 1):
            stem = f"out/r4_sd_{tag}_{i}"
            env = "CUBLAS_WORKSPACE_CONFIG=:4096:8 " if det else ""
            flag = "--deterministic " if det else ""
            out.append("python -c 'import torch; assert torch.cuda.is_available()' && "
                       f"{env}python scripts/benchmark.py {flag}{BASE} {extra} "
                       f"--json {stem}.json > {stem}.txt; cat {stem}.txt")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true", help="print the commands instead of submitting them")
    ap.add_argument("--name", default="r4-slowdet")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--cmds", default="out/r4_slowdet_cmds.txt", help="write the command list here")
    a = ap.parse_args()
    cmds = commands()
    os.makedirs(os.path.join(ROOT, os.path.dirname(a.cmds)), exist_ok=True)
    with open(os.path.join(ROOT, a.cmds), "w", encoding="utf-8") as f:
        f.write("\n".join(cmds) + "\n")
    if a.print:
        print("\n".join(cmds))
        return
    argv = [sys.executable, os.path.join(ROOT, "scripts", "cluster_run.py"),
            "--name", a.name, "--minutes", str(a.minutes), *cmds, "--fetch", "out/"]
    print(f"{len(cmds)} job(s) -> cluster_run.py --name {a.name}", flush=True)
    raise SystemExit(subprocess.call(argv, cwd=ROOT))


if __name__ == "__main__":
    main()
