"""Reproduce plume correction tables and trajectories, verifying frozen and runtime source."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from navigation_analyse import verify_frozen


def analyse(out):
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    results, snapshots = {}, {}
    summary = {"runtime_verification": [], "artifacts": {}, "rooms": {}}
    for family, names in (
        ("diagnostic", ("compass", "compass_ring")),
        ("validation", ("lifecycle", "boundary", "compass", "compass_ring")),
        ("scalar", ("scalar",)),
    ):
        source = Path("out") / f"plume_{family}_v1"
        declaration = json.loads(
            (source / "predeclared.json").read_text(encoding="utf-8")
        )
        frozen, runtime = verify_frozen(source, declaration["commit"])
        snapshots[family] = runtime
        (destination / f"{family}_predeclared.json").write_text(
            json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
        )
        for name in names:
            path = source / (name + ".json")
            record = json.loads(path.read_text(encoding="utf-8"))
            label = family + "/" + name
            summary["artifacts"][label] = hashlib.sha256(path.read_bytes()).hexdigest()
            for p in record.get("controllers", [record.get("provenance")]):
                assert (
                    p and p["preset"] == "instrumented" and len(p["instruments"]) == 4
                )
                files = p["source_fingerprint"]["files"]
                shared = set(files) & set(runtime)
                assert {"flyverse/navigation.py", "flyverse/fly.py"} <= shared
                assert all(files[k] in runtime[k] for k in shared), label
                summary["runtime_verification"].append(
                    {"artifact": label, "shared_source_files": len(shared)}
                )
            results[label] = record
    # The scalar follow-up changes only the harness/declaration, never the controller laws.
    for name in ("flyverse/navigation.py", "flyverse/fly.py"):
        assert snapshots["validation"][name] == snapshots["scalar"][name]
    summary["runtime_source_sha256"] = {
        family: {k: next(iter(v)) for k, v in hashes.items()}
        for family, hashes in snapshots.items()
    }
    summary["device"] = results["validation/boundary"]["provenance"]["execution"][
        "device_name"
    ]
    summary["compiled_connectome_md5"] = results["validation/boundary"]["provenance"][
        "compiled_connectome"
    ]["md5"]
    summary["boundary"] = {
        k: results["validation/boundary"][k] for k in ("columns", "rows", "gates")
    }
    summary["lifecycle"] = results["validation/lifecycle"]["records"]
    table = [
        "| arm | heading | env seed | initial heading deg | feeding s | first feeding s | heading span deg | mean absolute DNa02 L-R Hz | nearest fruit m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("diagnostic", "validation"):
        for heading in ("compass", "compass_ring"):
            label = family + "/" + heading
            record = results[label]
            data = np.asarray(record["samples"])
            rows = []
            for row in record["summary"]:
                row = dict(row)
                i = row["env_seed"]
                contact = np.flatnonzero(data[:, i, 18])
                row["first_feeding_s"] = (
                    float((contact[0] + 1) / 10) if len(contact) else None
                )
                row["end_energy"] = float(data[-1, i, 4])
                rows.append(row)
                first = f"{row['first_feeding_s']:.1f}" if len(contact) else "none"
                table.append(
                    f"| {family} | {heading} | {i} | {row['initial_heading_deg']} | {row['feeding_s']:.2f} | {first} | {row['heading_range_deg']:.2f} | {row['mean_abs_DNa02_difference_hz']:.3f} | {row['min_fruit_distance_m']:.4f} |"
                )
            summary["rooms"][label] = {"rows": rows, "gates": record.get("gates", {})}
    scalar = results["scalar/scalar"]
    data = np.asarray(scalar["samples"])
    contact = np.flatnonzero(data[:, 7])
    summary["scalar"] = {
        "feeding_s": scalar["feeding_s"],
        "gates": scalar["gates"],
        "first_feeding_s": float((contact[0] + 1) / 10) if len(contact) else None,
        "end_energy": float(data[-1, 5]),
        "heading_span_deg": float(np.rad2deg(np.ptp(np.unwrap(data[:, 3])))),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "rooms.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    plot(results, destination)
    print("\n".join(table))
    print("scalar:", summary["scalar"])


def plot(results, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    matplotlib.rcParams["svg.hashsalt"] = "plume-steering"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10").colors
    for row, heading in enumerate(("compass", "compass_ring")):
        for col, family in enumerate(("diagnostic", "validation")):
            ax = axes[row, col]
            data = np.asarray(results[family + "/" + heading]["samples"])
            ax.add_patch(Rectangle((-0.6, -0.4), 1.2, 0.8, color="0.95", zorder=-2))
            for i in range(6):
                ax.plot(
                    data[:, i, 0],
                    data[:, i, 1],
                    color=colors[i],
                    lw=1.2,
                    label=f"seed {i}",
                )
                fed = np.flatnonzero(data[:, i, 18])
                if len(fed):
                    ax.scatter(
                        *data[fed[0], i, :2],
                        color=colors[i],
                        s=35,
                        marker="o",
                        edgecolors="black",
                        linewidths=0.6,
                    )
            ax.set_title(f"{heading}: {'before' if col == 0 else 'corrected'}")
            ax.set(
                aspect="equal",
                xlabel="x (m)",
                ylabel="y (m)",
            )
            ax.grid(alpha=0.15)
    axes[0, 0].legend(ncol=3, fontsize=8, loc="lower left")
    fig.suptitle(
        "60 s room traces; circles mark first feeding\nTop-down projection; grey rectangle is the table",
        fontsize=12,
    )
    fig.tight_layout()
    svg = destination / "trajectories.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    fig.savefig("out/plume_validation_v1/trajectories.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/audits/data/plume_steering")
    analyse(ap.parse_args().out)
