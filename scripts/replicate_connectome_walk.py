"""Pack one female walking replicate per process seed, preserving child exit status.

Plan via probe_vnc_drive.py plan --family body --dataset banc --runs 5.
Five jobs each run the five body arms and the plain-walking probe sequentially.
The JSON plan can be passed as structured subprocess arguments on any platform.
"""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys


def write_plan(args):
    if args.family != "body":
        raise ValueError("the BANC replicate uses --family body")
    out = Path(args.dir); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.only_seeds.split(",")] if args.only_seeds else list(range(args.runs))
    if len(seeds) > 24:
        raise ValueError("keep a house submission at or below 24 jobs")
    commands = [f"mkdir -p {shlex.quote(args.dir)} && source .venv/bin/activate && "
                f"python scripts/replicate_connectome_walk.py --seed {s} --out {shlex.quote(args.dir)} --block fam_r{s}"
                for s in seeds]
    argv = ["scripts/cluster_run.py", "--name", args.name or "bancwalk", "--minutes", str(args.minutes),
            "--arm-block", "fam", "--target", "house", *commands, "--fetch", args.dir.rstrip("/") + "/"]
    (out / "plan.json").write_text(json.dumps(dict(dataset="banc", family="body", seeds=seeds, jobs=len(commands),
        simulations=6 * len(seeds), argv=argv,
        omitted="compass and benchmark protocols require separate annotation/capability audits; this batch is walking only"), indent=2), encoding="utf-8")
    (out / (args.script or "batch.sh")).write_text("#!/bin/bash\nset -euo pipefail\n" + shlex.join(["python", *argv]) + "\n", encoding="utf-8")
    print(f"wrote {out}/plan.json: {len(commands)} house jobs, {6 * len(seeds)} simulations, one submission")
    return 0


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Rotate order across seeds, so a fixed arm is not always warmed up last.
    arms = list("ABCDE"); arms = arms[args.seed % 5:] + arms[:args.seed % 5]
    commands = [[sys.executable, "scripts/probe_vnc_drive.py", "room", "--dataset", "banc", "--family", "body",
        "--arm", arm, "--seed", str(args.seed), "--batch", "16", "--seconds", "60", "--block", args.block,
        "--out", str(out / f"room_{arm}_r{args.seed}")] for arm in arms]
    commands += [[sys.executable, "scripts/probe_walk_straightness.py", "--dataset", "banc", "--arm", "default",
                  "--seed", str(args.seed), "--batch", "16", "--seconds", "60", "--out", str(out / f"straight_r{args.seed}.json")]]
    results = []
    for i, cmd in enumerate(commands):
        log = out / f"seed{args.seed}_step{i}.txt"
        with log.open("w", encoding="utf-8") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        results.append(dict(argv=cmd, exit_code=result.returncode, log=str(log)))
        print(f"seed {args.seed} step {i}: exit {result.returncode} ({log})", flush=True)
        if result.returncode:
            print(log.read_text(encoding="utf-8")[-5000:], flush=True)
    (out / f"jobs_r{args.seed}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return int(any(r["exit_code"] for r in results))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", required=True)
    sys.exit(run(ap.parse_args()))
