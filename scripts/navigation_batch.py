"""Freeze the navigation assays before a single sequential house submission."""

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(
    out,
    lifecycle_only=False,
    flight_priority=False,
    plume_diagnostic=False,
    plume_validation=False,
    plume_scalar=False,
):
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
    if plume_diagnostic:
        commands = [
            f"python scripts/plume_steering_probe.py --heading {heading} --out {rel}/{heading}.json"
            for heading in ("compass", "compass_ring")
        ]
    if plume_validation:
        commands = [
            f"python scripts/plume_steering_probe.py --mode {mode} --out {rel}/{mode}.json"
            for mode in ("lifecycle", "boundary")
        ] + [
            f"python scripts/plume_steering_probe.py --mode room --heading {heading} --out {rel}/{heading}.json"
            for heading in ("compass", "compass_ring")
        ]
    if plume_scalar:
        commands = [
            f"python scripts/plume_steering_probe.py --mode scalar --out {rel}/scalar.json",
            f"python scripts/room_demo.py --headless --seconds 1 --preset instrumented --instruments compass plume hunger flight --brain-map --cuda-graphs --cuda-kernels --event-driven --cuda-sparse warp --screenshot {rel}/ui.png",
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
    if plume_diagnostic or plume_validation or plume_scalar:
        paths += [
            ROOT / p
            for p in (
                "scripts/plume_steering_probe.py",
                "docs/audits/plume_steering.md",
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
    if plume_diagnostic:
        declaration["rules"] = (
            "docs/audits/plume_steering.md section 1; diagnostic only, no tuning/adoption"
        )
    if plume_validation:
        declaration["rules"] = (
            "docs/audits/plume_steering.md section 4; functional engineering checks, no fitting/adoption"
        )
    if plume_scalar:
        declaration["rules"] = (
            "docs/audits/plume_steering.md section 5; scalar integration and UI smoke only"
        )
        declaration["neural_seed"] = 0
        declaration["room_seeds"] = [0]
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
    ap.add_argument("--plume-diagnostic", action="store_true")
    ap.add_argument("--plume-validation", action="store_true")
    ap.add_argument("--plume-scalar", action="store_true")
    args = ap.parse_args()
    if (
        sum(
            (
                args.lifecycle_only,
                args.flight_priority,
                args.plume_diagnostic,
                args.plume_validation,
                args.plume_scalar,
            )
        )
        > 1
    ):
        ap.error("choose one assay family")
    build(
        args.out,
        args.lifecycle_only,
        args.flight_priority,
        args.plume_diagnostic,
        args.plume_validation,
        args.plume_scalar,
    )
