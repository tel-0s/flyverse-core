"""Regenerate public navigation tables without copying host/path-bearing raw provenance."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify_frozen(source, commit):
    frozen = json.loads((source / "predeclared.json").read_text())
    hashes = frozen["source_sha256_lf"]
    runtime = {}
    for name, wanted in hashes.items():
        data = subprocess.check_output(
            ["git", "show", f"{commit}:{name}"], cwd=ROOT
        ).replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == wanted, (
            f"frozen source mismatch: {name}"
        )
        # Shipping preserves mixed Windows line endings; the archive is the exact remote source.
        # Validate its normalized content against the PRE-submission hash and its bytes against
        # runtime provenance, rather than treating an arbitrary hash as an encoding equivalent.
        shipped = (source / "source" / name).read_bytes()
        assert hashlib.sha256(shipped.replace(b"\r\n", b"\n")).hexdigest() == wanted, (
            name
        )
        runtime[name] = {hashlib.sha256(shipped).hexdigest()}
    return frozen, runtime


def analyse(source, destination, commit, correction, correction_commit):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    frozen, runtime = verify_frozen(source, commit)
    hashes = frozen["source_sha256_lf"]
    checked = []

    def verify(p, label, expected=runtime):
        files = p["source_fingerprint"]["files"]
        common = set(files) & set(expected)
        bad = [n for n in common if files[n] not in expected[n]]
        if bad:
            raise AssertionError(f"{label}: runtime source differs: {bad}")
        assert "flyverse/navigation.py" in common
        checked.append({"artifact": label, "shared_source_files": len(common)})

    def read(name):
        d = json.loads((source / (name + ".json")).read_text())
        if "provenance" in d:
            verify(d["provenance"], name)
        for i, p in enumerate(d.get("controllers", [])):
            verify(p, f"{name}[{i}]")
        return d

    lifecycle, recurrent, neural = read("lifecycle"), read("recurrent"), read("neural")
    rooms = {arm: read("room_" + arm) for arm in ("compass", "plume", "flight")}
    profile = read("profile")
    correction = Path(correction)
    corrected_frozen, corrected_runtime = verify_frozen(correction, correction_commit)
    corrected = json.loads((correction / "lifecycle.json").read_text())
    assert corrected["records"] == lifecycle["records"]
    for i, p in enumerate(corrected["controllers"]):
        verify(p, f"lifecycle-correction[{i}]", corrected_runtime)
        assert p["preset"] == "instrumented"
        assert len(p["instruments"]) == 4
    summary = {
        "source_commit": commit,
        "frozen_sources": len(hashes),
        "runtime_verification": checked,
        "runtime_source_sha256": {k: next(iter(v)) for k, v in runtime.items()},
        "device": neural["provenance"]["execution"]["device_name"],
        "compiled_connectome_md5": neural["provenance"]["compiled_connectome"]["md5"],
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.glob("*.json")
        },
        "lifecycle": lifecycle["records"],
        "lifecycle_correction": {
            "source_commit": correction_commit,
            "frozen_sources": len(corrected_frozen["source_sha256_lf"]),
            "artifact_sha256": hashlib.sha256(
                (correction / "lifecycle.json").read_bytes()
            ).hexdigest(),
            "controllers": len(corrected["controllers"]),
            "runtime_source_sha256": {
                k: next(iter(v)) for k, v in corrected_runtime.items()
            },
            "exact_checks_unchanged": True,
        },
        "recurrent": [
            {k: v for k, v in r.items() if k != "instrument"}
            for r in recurrent["records"]
        ],
        "neural": {"columns": neural["columns"], "rows": neural["rows"]},
        "rooms": {},
        "profile_records": profile["records"],
        "profile": [],
        "workload_identity": profile["workload_identity"],
        "identity": profile["identity"],
    }
    for arm, d in rooms.items():
        positions = np.asarray(d["body"])
        summary["rooms"][arm] = [
            {
                "seed": i,
                "airborne_s": d["airborne_s"][i],
                "feeding_s": d["feeding_s"][i],
                "distance_m": d["distance_m"][i],
                "nearest_fruit_m": d["nearest_fruit_m"][i],
                "z_min_m": float(positions[:, i, 2].min()),
                "z_max_m": float(positions[:, i, 2].max()),
                "mean_wing_power_hz": float(np.asarray(d["wing_power"])[:, i].mean()),
            }
            for i in range(6)
        ]
    for batch in (1, 8, 32):
        selected = [r for r in profile["records"] if r["batch"] == batch]
        variants = list(dict.fromkeys(tuple(r["instruments"]) for r in selected))
        reference = np.median(
            [r["cuda_ms"] for r in selected if r["instruments"] == ["compass"]]
        )
        for names in variants:
            rows = [r for r in selected if tuple(r["instruments"]) == names]
            median = float(np.median([r["cuda_ms"] for r in rows]))
            summary["profile"].append(
                {
                    "batch": batch,
                    "instruments": list(names),
                    "cuda_ms_median": median,
                    "wall_ms_median": float(np.median([r["wall_ms"] for r in rows])),
                    "extra_us": (median - reference) * 1000,
                    "extra_percent": 100 * (median / reference - 1),
                }
            )
    (destination / "navigation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    declaration = {
        k: frozen[k]
        for k in (
            "stamped_utc",
            "commit",
            "status",
            "commands",
            "source_sha256_lf",
            "batch_sha256",
            "rules",
        )
    }
    (destination / "navigation_predeclared.json").write_text(
        json.dumps(declaration, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "navigation_lifecycle_predeclared.json").write_text(
        json.dumps(corrected_frozen, indent=2) + "\n", encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "flyverse-navigation"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, (arm, d) in zip(axes, rooms.items()):
        p = np.asarray(d["body"])
        for i in range(6):
            ax.plot(
                p[:, i, 0] * 100, p[:, i, 1] * 100, lw=0.9, alpha=0.8, label=f"seed {i}"
            )
            ax.scatter(p[-1, i, 0] * 100, p[-1, i, 1] * 100, s=8)
        ax.scatter([-15], [15], marker="x", color="black", s=25)
        ax.set(title=arm + " (60 s)", xlabel="x (cm)", ylabel="y (cm)", aspect="equal")
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle(
        "Experimental room trajectories; flight panel is horizontal projection; no food-finding claim",
        fontsize=10,
    )
    fig.savefig(destination / "navigation_room.svg", metadata={"Date": None})
    svg = destination / "navigation_room.svg"
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "recurrent",
                    "neural",
                    "rooms",
                    "profile",
                    "workload_identity",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="out/navigation_v1")
    ap.add_argument("--destination", default="docs/audits/data/navigation")
    ap.add_argument("--source-commit", default="3cac3cc")
    ap.add_argument("--lifecycle-correction", default="out/navigation_v2")
    ap.add_argument("--correction-commit", default="c7ead86")
    a = ap.parse_args()
    analyse(
        a.source,
        a.destination,
        a.source_commit,
        a.lifecycle_correction,
        a.correction_commit,
    )
