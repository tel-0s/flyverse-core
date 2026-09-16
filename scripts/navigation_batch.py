"""Freeze the navigation assays before a single sequential house submission."""

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(out, lifecycle_only=False, flight_priority=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "predeclared.json").exists():
        raise FileExistsError("already frozen")
    rel = f"out/{out.name}"
    commands = []
    for mode in ("lifecycle", "recurrent", "neural"):
        commands.append(
            f"python scripts/navigation_probe.py --mode {mode} --out {rel}/{mode}.json"
        )
    for arm, names in (
        ("compass", "compass"),
        ("plume", "compass plume hunger"),
        ("flight", "compass plume hunger flight"),
    ):
        commands.append(
            f"python scripts/navigation_probe.py --mode room --instruments {names} --out {rel}/room_{arm}.json"
        )
    commands.append(
        f"python scripts/navigation_probe.py --mode profile --out {rel}/profile.json"
    )
    commands.append(
        f"python scripts/room_demo.py --headless --seconds 1 --preset instrumented --instruments compass plume hunger flight --brain-map --cuda-graphs --cuda-kernels --event-driven --cuda-sparse warp --screenshot {rel}/ui.png"
    )
    if lifecycle_only:
        commands = commands[:1]
    if flight_priority:
        commands = [
            f"python scripts/flight_priority_probe.py --mode {mode} --out {rel}/{mode}.json"
            for mode in ("lifecycle", "transitions", "room", "depleted")
        ]
    chain = " && ".join(commands)
    log = f"{rel}/fam_navigation.txt"
    job = f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && ( {chain} ) > {log} 2>&1; st=$?; tail -8 {log}; exit $st"
    batch = (
        "#!/bin/bash\nset -o pipefail\n"
        + f"python scripts/cluster_run.py --target house --name navigation --minutes 30 --arm-block fam {shlex.quote(job)} --fetch {rel}/ 2>&1 | tee {rel}/client_stdout.txt\n"
    )
    (out / "batch.sh").write_text(batch, encoding="utf-8", newline="\n")
    paths = [
        p
        for p in (ROOT / "flyverse").rglob("*")
        if p.suffix in (".py", ".cu", ".metal", ".csv")
    ]
    paths += [
        ROOT / p
        for p in (
            "scripts/navigation_probe.py",
            "scripts/navigation_batch.py",
            "scripts/room_demo.py",
            "scripts/benchmark.py",
            "scripts/cx_wedge.py",
            "tests/test_navigation_instruments.py",
            "docs/audits/navigation_instruments.md",
        )
    ]
    if flight_priority:
        paths += [
            ROOT / p
            for p in (
                "scripts/flight_priority_probe.py",
                "docs/audits/flight_foraging_priority.md",
            )
        ]
    hashes = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in paths
    }
    declaration = {
        "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "status": "frozen before submission",
        "commands": commands,
        "source_sha256_lf": hashes,
        "batch_sha256": hashlib.sha256((out / "batch.sh").read_bytes()).hexdigest(),
        "rules": (
            "docs/audits/flight_foraging_priority.md section 2; no fit or adoption"
            if flight_priority
            else "docs/audits/navigation_instruments.md section 3; no fit or adoption"
        ),
        "room_seconds": 60,
        "room_seeds": list(range(6)),
        "neural_seed": 31,
    }
    (out / "predeclared.json").write_text(
        json.dumps(declaration, indent=2) + "\n", encoding="utf-8"
    )
    print("froze", len(commands), "sequential assays in", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/navigation_v1")
    ap.add_argument(
        "--lifecycle-only",
        action="store_true",
        help="metadata/lifecycle correction; no repeated behavioral assays",
    )
    ap.add_argument(
        "--flight-priority",
        action="store_true",
        help="flight interruption, capture transitions and two room observations",
    )
    args = ap.parse_args()
    if args.lifecycle_only and args.flight_priority:
        ap.error("choose either --lifecycle-only or --flight-priority")
    build(args.out, args.lifecycle_only, args.flight_priority)
