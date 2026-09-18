"""Prepare (never submit), run, or summarize the predeclared transduced-plume rooms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

ARMS = {
    "full": ["compass", "plume", "hunger", "flight"],
    "transduced": ["compass", "plume:bilateral=orn", "hunger", "flight"],
    "goal_only": ["compass", "plume:feedback=off", "hunger", "flight"],
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def hashes():
    paths = [
        p
        for p in (ROOT / "flyverse").rglob("*")
        if p.suffix in (".py", ".cu", ".metal", ".csv")
    ]
    paths += [Path(__file__), ROOT / "scripts/plume_transduced_characterise.py"]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in sorted(paths)
    }


def plan(
    out, published_plan=Path("docs/audits/data/plume_transduced/predeclared.json")
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from flyverse import connectome, world
    from flyverse.instruments import make_instrument

    published_plan = (ROOT / published_plan).resolve()
    public_rel = published_plan.relative_to(ROOT).as_posix()
    if out.exists() or published_plan.exists():
        raise FileExistsError("use a fresh plan directory and published declaration")
    c = connectome.load()
    out.mkdir(parents=True)
    rel = out.relative_to(ROOT).as_posix()
    starts = []
    for seed in range(6):
        _, info = world.make_room(seed)
        rng = np.random.default_rng(20260917 + seed)
        # Whole tabletop, 3 cm edge margin; reject starts already touching fruit.
        # This is fixture construction only; the controller never sees fruit positions.
        for _ in range(1000):
            x, y = rng.uniform(-0.57, 0.57), rng.uniform(-0.37, 0.37)
            if all(
                np.hypot(x - p[0], y - p[1]) > r + 0.015 for _, p, r in info["fruit"]
            ):
                break
        else:
            raise RuntimeError("failed to draw a free start")
        starts.append(
            {
                "env_seed": seed,
                "neural_seed": 31 + seed,
                "start_seed": 20260917 + seed,
                "position": [float(x), float(y), info["table_top_z"]],
                "heading_deg": float(rng.uniform(-180, 180)),
            }
        )
    protocol = {
        "stamped_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PREPARED ONLY; no submission authorized until Fable releases the pool",
        "seconds": 60,
        "results_dir": rel,
        "published_plan": public_rel,
        "initial_energy": 0.1,
        "batch": 1,
        "fruit": "all",
        "fence": False,
        "brain_dt_ms": 0.5,
        "optic_dt_ms": 1.0,
        "frame_ms": 10,
        "sample_ms": 100,
        "cuda_graphs": True,
        "cuda_kernels": True,
        "event_driven": True,
        "cuda_sparse": "torch",
        "program": "none",
        "escape_gating": False,
        "gf_threshold_hz": 33.0,
        "arms": ARMS,
        "starts": starts,
        "replicate_unit": "independent process/run, six per arm",
        "measurements": [
            "feeding_s (>=1 s contact criterion)",
            "first_feed_s (null if censored at 60 s)",
            "final energy",
            "minimum fruit distance",
            "mean absolute goal offset",
            "airborne_s",
            "weighted ORN L/R brain.rate, filtered contrast, target/measured DNa02 difference",
        ],
        "analysis": "descriptive mean and across-run sample SD; report six rows and all failures; no samples-as-replicates, significance claim, fit or adoption",
        "comparisons": [
            "transduced versus full: sensory-cue replacement with the same feedback",
            "goal_only versus full: physical goal retained, DNa02 feedback removed",
        ],
        "contrast_compression": "at median historical total, rate/physical contrast is 0.371 lime, 0.429 apple, 0.447 banana, 0.639 all-fruit; gain 200 is unverified and underived, so transduced/full also changes small-contrast gain to 0.37-0.64x",
        "precision": "mean and across-run SD at two significant digits in SD, unless a matching execution path passes docs/audits/determinism_gate.md; exact JSON retained as machine records",
        "provenance": {
            "preset": "instrumented",
            "dataset": c.dataset,
            "release": c.release,
            "cache_md5": {
                name: hashlib.md5(
                    (connectome.CACHE_DIR / name).read_bytes()
                ).hexdigest()
                for name in ("neurons.parquet", "W_post_pre.npz", "sign0_counts.npz")
            },
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256_lf": hashes(),
            "instruments": {
                arm: [make_instrument(c, name).describe() for name in names]
                for arm, names in ARMS.items()
            },
        },
    }
    commands = []
    for seed in range(6):
        # Rotate arm order across matched starts; each is a fresh independent process.
        names = list(ARMS)
        for arm in names[seed % 3 :] + names[: seed % 3]:
            commands.append(
                f"python scripts/plume_transduced_batch.py room --plan {public_rel} --arm {arm} --seed {seed} --out {rel}/{arm}_{seed}.json"
            )
    protocol["commands"] = commands
    job_lines = []
    for command in commands:
        stem = Path(command.split("--out ")[1]).stem
        filename = f"{rel}/fam_plume/{stem}.log"
        job = (
            f"mkdir -p {rel}/fam_plume && source .venv/bin/activate && FLYVERSE_PLUME_GPU_RELEASED=1 {command} > {filename} 2>&1; "
            f"st=$?; tail -8 {filename}; exit $st"
        )
        job_lines.append(shlex.quote(job))
    batch = (
        "#!/bin/bash\nset -euo pipefail\n"
        "# DRAFT: do not run until Fable explicitly releases the GPU pool.\n"
        'if [ "${FLYVERSE_PLUME_GPU_RELEASED:-0}" != 1 ]; then\n'
        '  echo "Prepared only: wait for Fable, then set FLYVERSE_PLUME_GPU_RELEASED=1." >&2\n'
        "  exit 2\nfi\n"
        "python scripts/cluster_run.py --target house --name plume_transduced --minutes 45 --arm-block fam \\\n"
        + " \\\n".join("  " + j for j in job_lines)
        + f" --fetch {rel}/ 2>&1 | tee {rel}/client_stdout.txt\n"
    )
    # cluster_run's explicit arm block is carried by every job's log path.
    (out / "batch.sh").write_text(batch, encoding="utf-8", newline="\n")
    protocol["batch_sha256"] = hashlib.sha256(
        (out / "batch.sh").read_bytes()
    ).hexdigest()
    save(out / "predeclared.json", protocol)
    # A tracked/public declaration is shipped; ignored out/ is not in the source overlay.
    save(published_plan, protocol)
    print(f"Prepared {len(commands)} independent rooms in {rel}; NOT submitted")


def room_sample(sim, plume, ix, w, time_s, distance):
    """One post-frame row, using the same boundary as the CUDA room logger."""
    import torch

    # Post-frame neural measurements are explicitly labelled; the controller
    # used frame-start rates. Its filtered cue/goal/servo state is sampled too.
    neural = (
        torch.cat(
            [sim.brain.rate[:, ix[s]] @ w[s] for s in ("L", "R")]
            + [
                plume.odor,
                plume.strength,
                plume.heading,
                plume.goal,
                plume.bilateral,
                plume.target_difference,
                plume.steer_input,
            ],
            1,
        )
        .cpu()
        .numpy()[0]
    )
    f, m, motor = sim.flies[0], sim.metabolisms[0], sim.motor.row(0)
    return [
        time_s,
        f.x,
        f.y,
        f.z,
        f.heading,
        f.yaw_rate,
        m.energy,
        distance,
        float(sim.feeding[0]),
        *[float(sum(d.values())[0]) for d in sim.smell_values],
        *neural.tolist(),
        float(motor.turn_L),
        float(motor.turn_R),
    ]


def room(args):
    if os.environ.get("FLYVERSE_PLUME_GPU_RELEASED") != "1":
        raise RuntimeError("room execution blocked pending Fable's GPU-pool release")
    import torch

    from flyverse import BatchSim, connectome
    from flyverse.interp.common import provenance, to_jsonable
    from flyverse.navigation import bilateral_orn_groups

    if not torch.cuda.is_available():
        raise RuntimeError("the prepared rooms require house CUDA execution")
    p = json.loads(args.plan.read_text(encoding="utf-8"))
    if p["provenance"]["source_sha256_lf"] != hashes() or p["arms"] != ARMS:
        raise ValueError("source or protocol differs from frozen plan; do not run")
    if args.out.exists():
        raise FileExistsError(args.out)
    start = p["starts"][args.seed]
    c = connectome.load()
    if any(
        hashlib.md5((connectome.CACHE_DIR / name).read_bytes()).hexdigest() != digest
        for name, digest in p["provenance"]["cache_md5"].items()
    ):
        raise ValueError("MaleCNS cache differs from frozen plan")
    sim = BatchSim(
        1,
        c=c,
        seed=start["neural_seed"],
        seeds=[start["env_seed"]],
        start=start["position"],
        device="cuda",
        cuda_graphs=True,
        cuda_kernels=True,
        event_driven=True,
        cuda_sparse="torch",
        program="none",
        fruit_set="all",
        preset="instrumented",
        instruments=ARMS[args.arm],
    )
    sim.flies[0].heading = np.deg2rad(start["heading_deg"])
    sim.metabolisms[0].energy = 0.1
    plume = sim.fb.instruments["plume"]
    groups, weights, _ = bilateral_orn_groups(c)
    ix = {s: torch.as_tensor(v, device="cuda") for s, v in groups.items()}
    w = {s: torch.as_tensor(v, device="cuda") for s, v in weights.items()}
    print("device", sim.brain.device, "arm", args.arm, "seed", args.seed, flush=True)
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    columns = [
        "time_s",
        "x",
        "y",
        "z",
        "heading",
        "yaw",
        "energy",
        "fruit_distance",
        "feeding",
        "antenna_concentration_L",
        "antenna_concentration_R",
        "orn_rate_L",
        "orn_rate_R",
        "odor",
        "strength",
        "neural_heading",
        "goal",
        "bilateral",
        "target_DNa02_difference",
        "PFL3_input_difference",
        "DNa02_L",
        "DNa02_R",
    ]
    samples = []
    feeding = airborne = 0.0
    first_feed = None
    min_distance = float("inf")
    for frame in range(6000):
        sim.step()
        feeding += float(sim.feeding[0]) * 0.01
        airborne += float(sim.flies[0].airborne) * 0.01
        distance = float(sim.nearest_fruit()[1][0])
        min_distance = min(min_distance, distance)
        if sim.feeding[0] and first_feed is None:
            first_feed = (frame + 1) * 0.01
        if frame % 10 == 9:
            samples.append(
                room_sample(sim, plume, ix, w, (frame + 1) * 0.01, distance)
            )
        if (frame + 1) % 1000 == 0:
            print(f"frame {frame + 1}/6000", flush=True)
    a = np.asarray(samples)
    error = np.angle(
        np.exp(
            1j * (a[:, columns.index("goal")] - a[:, columns.index("neural_heading")])
        )
    )
    result = {
        "arm": args.arm,
        "seed": args.seed,
        "start": start,
        "columns": columns,
        "samples": samples,
        "summary": {
            "feeding_s": feeding,
            "fed_at_least_1s": feeding >= 1,
            "first_feed_s": first_feed,
            "final_energy": sim.metabolisms[0].energy,
            "airborne_s": airborne,
            "min_fruit_distance_m": min_distance,
            "mean_abs_goal_offset_deg": float(np.rad2deg(np.abs(error)).mean()),
        },
        "sampling": "neural ORN rates post-frame; plume state derived from frame-start brain.rate; every 100 ms",
        "declaration_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "provenance": provenance(
            c,
            fb=sim.fb,
            seeds=[start["neural_seed"]],
            env_seeds=[start["env_seed"]],
            batch=1,
            stimulus={"protocol": p, "arm": args.arm, "start": start},
        ),
    }
    save(args.out, to_jsonable(result))
    print(json.dumps(result["summary"]), flush=True)


def analyse(args):
    records = []
    digest = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    p = json.loads(args.plan.read_text(encoding="utf-8"))
    result_dir = ROOT / p["results_dir"]
    for arm in ARMS:
        for seed in range(6):
            r = json.loads(
                (result_dir / f"{arm}_{seed}.json").read_text(encoding="utf-8")
            )
            if (r["arm"], r["seed"], r["declaration_sha256"]) != (arm, seed, digest):
                raise ValueError("run identity differs from declaration")
            if (
                r["provenance"]["preset"] != "instrumented"
                or r["provenance"]["instruments"] != p["provenance"]["instruments"][arm]
            ):
                raise ValueError("run instruments differ from declaration")
            if not str(r["provenance"]["execution"]["device"]).startswith("cuda"):
                raise ValueError("room did not execute on CUDA")
            if r["start"] != p["starts"][seed]:
                raise ValueError("run start differs from declaration")
            values = np.asarray(r["samples"])
            if (
                values.shape != (600, len(r["columns"]))
                or not np.isfinite(values).all()
                or values[-1, 0] != 60
            ):
                raise ValueError("incomplete or invalid room trace")
            records.append(r)
    rows = []
    for arm in ARMS:
        summaries = [r["summary"] for r in records if r["arm"] == arm]
        row = {
            "arm": arm,
            "n": 6,
            "feed_count": sum(r["fed_at_least_1s"] for r in summaries),
            "first_feed_censored_count": sum(
                r["first_feed_s"] is None for r in summaries
            ),
        }
        for key in (
            "feeding_s",
            "final_energy",
            "airborne_s",
            "min_fruit_distance_m",
            "mean_abs_goal_offset_deg",
        ):
            values = [r[key] for r in summaries]
            row[key] = {
                "mean": float(np.mean(values)),
                "across_run_sd": float(np.std(values, ddof=1)),
                "values": values,
            }
        rows.append(row)
    save(
        args.out,
        {
            "status": "descriptive only; no precision beyond across-run SD; independent skeptic pending",
            "rows": rows,
            "provenance": p["provenance"],
            "declaration_sha256": digest,
        },
    )
    for row in rows:
        stats = row["feeding_s"]
        sd = stats["across_run_sd"]
        places = 1 - math.floor(math.log10(sd)) if sd > 0 else 1
        mean = round(stats["mean"], places)
        print(
            f"{row['arm']}: fed {row['feed_count']}/6; feeding {mean:.{max(0, places)}f} +/- {sd:.2g} s (across-run SD)"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("plan", "room", "analyse"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", type=Path)
    ap.add_argument(
        "--published-plan",
        type=Path,
        default=Path("docs/audits/data/plume_transduced/predeclared.json"),
    )
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--seed", type=int, choices=range(6))
    args = ap.parse_args()
    if args.mode == "plan":
        plan(args.out.resolve(), args.published_plan)
    else:
        if args.plan is None or (
            args.mode == "room" and (args.arm is None or args.seed is None)
        ):
            ap.error("room requires --plan, --arm and --seed; analyse requires --plan")
        (room if args.mode == "room" else analyse)(args)
