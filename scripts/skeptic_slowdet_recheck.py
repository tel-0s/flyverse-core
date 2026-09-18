"""Round-4 skeptic re-check of the `slowdet` batch (docs/audits/slow_term.md section 8).

Independent replicates of three of that batch's arms, same protocol, same flags, submitted as ONE cluster batch
through scripts/cluster_run.py (all concurrent, one run directory, `--fetch out/`; no --cache-dir, so the
cluster's shared TYPE_NT_OVERRIDE cache):

  sk_ctl x2   --deterministic, term off (--slow-gain-monoamine 0), shipped table
  sk_ecr x1   --deterministic, --slow-mode threshold --slow-gain-monoamine 0.01, shipped (DopEcR) table
  sk_dop x1   the same plus --dopamine-lead dop1r1

Tests, against out/r4_sd_*.json: (a) walk.power_max_hz / power_sustained / GF_max reproduce ONE value per
condition (79.46501159667969 / 52.44422912597656 / 55.62504196166992 for ctl / ecr / dop), (b) motion.min_dsi
still differs between two deterministic term-off controls, (c) --deterministic raises no
"no deterministic implementation" error and no check reports MISSING, (d) the --receptor-model full run with an
active slow class does not crash under the with_counts fix in scripts/benchmark.py.

    python scripts/skeptic_slowdet_recheck.py --print     # just the commands
    python scripts/skeptic_slowdet_recheck.py             # submit the batch
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

ARMS = [("sk_ctl", 2, OFF), ("sk_ecr", 1, TERM), ("sk_dop", 1, f"{TERM} --dopamine-lead dop1r1")]


def commands() -> list[str]:
    out = []
    for tag, n, extra in ARMS:
        for i in range(1, n + 1):
            stem = f"out/r4_{tag}_{i}"
            out.append("python -c 'import torch; assert torch.cuda.is_available()' && "
                       f"CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/benchmark.py --deterministic {BASE} {extra} "
                       f"--json {stem}.json > {stem}.txt; cat {stem}.txt")
    return out


def cpu_checks():
    """CPU-only re-checks of the two scripts/benchmark.py edits (no GPU, no simulation):

    1. the with_counts crash the round-4 report claims for HEAD 79769c3: `rs = brain._receptor(c, lif)` (no counts)
       handed to a Brain whose slow class has a non-zero scale must raise
       ValueError('the slow term needs receptor_signs(..., with_counts=True)') in brain._slow_weights;
    2. the fix's inertness: receptor_signs(c, net_rule='abs') with and without with_counts must agree entry for
       entry on fast_sign / slow_sign / slow_class / tier / fast_gain / slow_gain.
    """
    import numpy as np
    sys.path.insert(0, ROOT)
    from flyverse import brain as br, connectome as cn
    gain = {"none": 1.0, "low": 1.0, "mid": 1.0, "high": 1.0}
    c = cn.load(verbose=False)
    print(f"connectome n={c.n} nnz={c.W.nnz} sum|W|={abs(c.W.data).sum():.0f}")
    lif = br.LIFParams(receptor_model="full", receptor_net_rule="abs", receptor_gain=gain,
                       slow_mode="threshold", slow_gain_by_class={"monoamine": 0.01})
    spec = br._slow_spec(lif)
    print(f"slow spec classes={spec.classes if spec else None}")
    rs0 = br._receptor(c, lif)                    # HEAD's call: no with_counts
    rs1 = br._receptor(c, lif, with_counts=True)  # the fixed call
    print(f"rs0.count={type(rs0.count).__name__}  rs1.count={type(rs1.count).__name__}")
    try:
        br._slow_weights(c, lif, rs0, spec)
        print("CHECK 1 FAILED: no ValueError without with_counts")
    except ValueError as e:
        print(f"CHECK 1 OK: ValueError({e})")
    S = br._slow_weights(c, lif, rs1, spec)
    print("CHECK 1b OK: with counts the slow matrices build: "
          + ", ".join(f"{k} nnz={v.nnz}" for k, v in S.items()))
    for f in ("fast_sign", "slow_sign", "slow_class", "tier", "fast_gain", "slow_gain"):
        a, b = getattr(rs0, f), getattr(rs1, f)
        if a is None or b is None:
            print(f"CHECK 2 {f}: {type(a).__name__} vs {type(b).__name__}")
        else:
            print(f"CHECK 2 {f}: equal={bool(np.array_equal(a, b))} n={len(a)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-checks", action="store_true", help="run the two CPU-only checks of the benchmark.py edits")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--name", default="r4-skslowdet")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--cmds", default="out/r4_sk_slowdet_cmds.txt")
    a = ap.parse_args()
    if a.cpu_checks:
        cpu_checks()
        return
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
