"""Regenerate public flight-priority tables from frozen, source-verified raw artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from navigation_analyse import verify_frozen


def analyse(source, destination):
    source, destination = Path(source), Path(destination)
    declaration = json.loads((source / "predeclared.json").read_text(encoding="utf-8"))
    frozen, runtime = verify_frozen(source, declaration["commit"])
    results, checked = {}, []
    for name in ("lifecycle", "transitions", "room", "depleted"):
        result = json.loads((source / (name + ".json")).read_text(encoding="utf-8"))
        records = result.get("controllers", [])
        if "provenance" in result:
            records = [result["provenance"]]
        assert records
        for p in records:
            assert p["preset"] == "instrumented" and len(p["instruments"]) == 4
            files = p["source_fingerprint"]["files"]
            common = set(files) & set(runtime)
            assert "flyverse/navigation.py" in common
            assert all(files[k] in runtime[k] for k in common), name
            checked.append({"artifact": name, "shared_source_files": len(common)})
        results[name] = result
    summary = {
        "source_commit": declaration["commit"],
        "frozen_source_files": len(runtime),
        "runtime_verification": checked,
        "runtime_source_sha256": {k: next(iter(v)) for k, v in runtime.items()},
        "artifact_sha256": {
            name + ".json": hashlib.sha256(
                (source / (name + ".json")).read_bytes()
            ).hexdigest()
            for name in ("predeclared", "lifecycle", "transitions", "room", "depleted")
        },
        "device": results["room"]["provenance"]["execution"]["device_name"],
        "compiled_connectome_md5": results["room"]["provenance"]["compiled_connectome"][
            "md5"
        ],
        "lifecycle": results["lifecycle"]["records"],
        "transition_frames": results["transitions"]["checks"],
        "gates": {
            name: results[name]["gates"] for name in ("transitions", "room", "depleted")
        },
        "rooms": {},
    }
    table = [
        "| start | env seed | airborne s | powered s | longest bout s | first ground s | feeding s | end energy | escape / voluntary hops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("room", "depleted"):
        result = results[name]
        energy = np.asarray(result["energy"])
        summary["rooms"][name] = []
        for seed in range(6):
            row = {
                key: result[key][seed]
                for key in (
                    "airborne_s",
                    "powered_s",
                    "max_power_bout_s",
                    "first_grounded_s",
                    "feeding_s",
                    "reserve_violations",
                    "hops_escape",
                    "hops_voluntary",
                )
            }
            row.update(seed=seed, end_energy=float(energy[-1, seed]))
            summary["rooms"][name].append(row)
            table.append(
                f"| {name} | {seed} | {row['airborne_s']:.2f} | {row['powered_s']:.2f} | "
                f"{row['max_power_bout_s']:.2f} | {row['first_grounded_s']:.2f} | "
                f"{row['feeding_s']:.2f} | {row['end_energy']:.4f} | "
                f"{row['hops_escape']} / {row['hops_voluntary']} |"
            )
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in (("predeclared", frozen), ("summary", summary)):
        (destination / (name + ".json")).write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
    (destination / "room_table.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    print("\n".join(table))
    print("gates:", json.dumps(summary["gates"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="out/flight_priority_v1")
    ap.add_argument("--out", default="docs/audits/data/flight_priority")
    args = ap.parse_args()
    analyse(args.source, args.out)
