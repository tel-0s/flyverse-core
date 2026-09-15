"""Compare walking cohorts descriptively; statistical calls stay within a dataset.

The historical MaleCNS vncd3 cohort uses seeds 0,1,3,4 from one submission.
Seed 2 belongs to a different submission and is deliberately excluded by default.
No batch row is treated as an independent replicate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse.interp import common
from flyverse import world
from probe_vnc_drive import robust_room, ARMS_BODY
from probe_walk_straightness import on_table


METRICS = (
    "yaw_sd_clean_deg_s", "yaw_p99_abs_clean_deg_s", "clean_frac",
    "straightness", "path_m", "n_left_table", "hops", "airborne_frac",
    "DNa02_L_hz", "DNa02_R_hz", "DNa02_LR_hz", "leg_LR_hz",
    "AN04B003_L_hz", "AN04B003_R_hz", "PS059_L_hz", "PS059_R_hz",
    "commanded_chordotonal_hz", "commanded_haltere_hz", "haltere_LR_hz",
)


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    return dict(n=len(v), mean=float(v.mean()) if len(v) else None,
                run_sd=float(v.std(ddof=1)) if len(v) > 1 else None, runs=v.tolist())


def cohort(root, dataset, seeds, batch_name):
    """Read exact summary names and require complete arms, provenance and body traces."""
    root = Path(root)
    arms, sources = {}, {}
    for arm, spec in ARMS_BODY.items():
        rows = []
        for seed in seeds:
            p = root / f"room_{arm}_r{seed}.json"
            body = p.with_name(p.stem + "_body.npz")
            d = json.loads(p.read_text(encoding="utf-8"))
            if (d["family"], d["seed"], d["batch"], d["seconds"], d["proprioception"]) != ("body", seed, 16, 60, spec):
                raise ValueError(f"unexpected walking protocol: {p}")
            provenance = d["provenance"]
            name = provenance["dataset_release"]["name"]
            if name != ("male-cns" if dataset == "malecns" else dataset):
                raise ValueError(f"dataset mismatch in {p}: {name}")
            values = dict(d["run"])
            values.update(robust_room(body, skip_f=round(d["skip_s"] * 100)))
            rows.append({k: values.get(k) for k in METRICS})
            sources[p.name] = dict(sha256=sha256(p), body_sha256=sha256(body),
                                   provenance=provenance, skip_s=d["skip_s"], device=d["device_name"])
        arms[arm] = dict(label=spec or "shipped default", metrics={k: stats([r[k] for r in rows]) for k in METRICS})
    comparisons = []
    for stimulus, control in (("B", "A"), ("C", "B"), ("D", "C"), ("E", "D")):
        for key in METRICS:
            comparisons.append(dict(stimulus=stimulus, control=control, metric=key,
                **common.compare(arms[stimulus]["metrics"][key]["runs"], arms[control]["metrics"][key]["runs"])))
    return dict(dataset=dataset, release=provenance["dataset_release"]["release"], batch=batch_name,
                seeds=seeds, batch_rows=16, seconds=60, arms=arms, within_dataset=comparisons, sources=sources)


def generate(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cohorts = {
        "malecns": cohort(args.malecns, "malecns", args.malecns_seeds, args.malecns_batch),
        "banc": cohort(args.banc, "banc", args.banc_seeds, args.banc_batch),
    }
    plain = {k: [] for k in ("straightness", "path_m", "n_left_table", "hops", "yaw_sd_deg_s")}
    invalid_plain_table_counts = []
    table = world.make_room(0, "all")[1]
    samples_on_table = samples_total = 0
    for seed in args.banc_seeds:
        p = Path(args.banc) / f"straight_r{seed}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["provenance"]["model"]["dataset"] != "banc" or d["seed"] != seed or len(d["rows"]) != 16 or d["seconds"] != 60:
            raise ValueError(f"unexpected plain-walking protocol: {p}")
        cohorts["banc"]["sources"][p.name] = dict(sha256=sha256(p), provenance=d["provenance"])
        for row in d["rows"]:
            for sample in row["track"]:
                samples_total += 1
                samples_on_table += int(on_table(sample[:3], table["table_extent"], table["table_top_z"]))
        if d.get("metrics_version", 1) < 2:
            invalid_plain_table_counts.append(dict(seed=seed, reported_n_left_table=d["summary"]["n_left_table"],
                reason="legacy probe treated (xmin,xmax,ymin,ymax) as half-extents, declaring every initial pose off-table"))
        for key in plain:
            value = d["summary"][key]
            plain[key].append(None if key == "n_left_table" and d.get("metrics_version", 1) < 2 else
                              value["mean"] if isinstance(value, dict) else value)
    rows = []
    for dataset, data in cohorts.items():
        for arm, values in data["arms"].items():
            for key, value in values["metrics"].items():
                rows.append(dict(dataset=dataset, release=data["release"], arm=arm, metric=key, **value))
        pd.DataFrame(data["within_dataset"]).to_csv(out / f"{dataset}_within.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "descriptive.csv", index=False)
    report = dict(generator="scripts/summarize_connectome_walk.py", generator_sha256=sha256(__file__),
        cohorts=cohorts, plain_banc={k: stats(v) for k, v in plain.items()}, invalid_plain_table_counts=invalid_plain_table_counts,
        plain_table_samples=dict(on_table=samples_on_table, total=samples_total, cadence_s=.5,
            bounds=list(table["table_extent"]), top_z=table["table_top_z"], world_source_sha256=sha256(world.__file__),
            interpretation="retained trajectory samples under current room bounds; not full-frame exit counters"),
        caveats=["Cross-dataset differences are descriptive: sex, individual, annotation, reconstruction, threshold, optic capability and hardware differ.",
                 "Only within-dataset arm comparisons use common.compare; independent process seeds, not flies, are the replicate unit.",
                 "Historical MaleCNS seeds 0,1,3,4 share vncd3b-c2eeaf; seed 2 from vncd3-f3bb50 is excluded.",
                 "Room yaw uses the existing clean-frame mask after 5 s. Plain-walking yaw includes edge/landing jumps and is not compared to clean yaw.",
                 "Legacy plain-probe table counters are excluded; body-arm table counters use the correct bounds and remain valid. Original result JSONs are retained.",
                 "Body arms B-E are opt-in controls; no parameter was tuned for BANC."])
    (out / "report.json").write_text(json.dumps(common.to_jsonable(report), indent=2), encoding="utf-8")
    for dataset, data in cohorts.items():
        for arm, values in data["arms"].items():
            m = values["metrics"]
            def display(key):
                v = m[key]
                return f"{key}: {v['mean']:.4g} +/- {v['run_sd']:.3g}" if v["n"] > 1 else f"{key}: {v['mean']} (n={v['n']})"
            print(dataset, arm, "; ".join(display(k) for k in
                ("yaw_sd_clean_deg_s", "straightness", "n_left_table", "DNa02_L_hz", "DNa02_R_hz")))
    print(f"wrote {out}: separate cohorts, no cross-dataset statistical calls")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--malecns", required=True, help="historical out/vncd3 directory")
    ap.add_argument("--banc", default="out/connectome_backends/walking")
    ap.add_argument("--out", default="out/connectome_backends/walking_comparison")
    seeds = lambda value: [int(x) for x in value.split(",")]
    ap.add_argument("--malecns-seeds", type=seeds, default=[0, 1, 3, 4])
    ap.add_argument("--banc-seeds", type=seeds, default=[0, 1, 2, 3, 4])
    ap.add_argument("--malecns-batch", default="vncd3b-c2eeaf", help="declared historical submission, verified against its audit")
    ap.add_argument("--banc-batch", default="cbwalk-384afd")
    generate(ap.parse_args())
