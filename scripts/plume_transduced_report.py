"""Validate finished plume rooms and emit descriptive run-level tables (no verdict)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plume_transduced_batch as batch


def table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def formatted(mean, sd):
    if sd == 0:
        return f"{mean:.3g} +/- 0"
    places = 1 - math.floor(math.log10(sd))
    rounded = round(mean, places)
    if rounded == 0:
        rounded = 0.0
    sd_text = f"{sd:#.2g}".removesuffix(".")
    return f"{rounded:.{max(0, places)}f} +/- {sd_text}"


def report(plan_path, out):
    from flyverse.interp.export import match_sources

    out.mkdir(parents=True, exist_ok=False)
    batch.analyse(SimpleNamespace(plan=plan_path, out=out / "summary.json"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    directory = ROOT / plan["results_dir"]
    rows, validation, raw_hashes = [], [], {}
    for arm in batch.ARMS:
        for seed in range(6):
            stem = f"{arm}_{seed}"
            path = directory / f"{stem}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            raw_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            execution = record["provenance"]["execution"]
            expected = {"cuda_kernels": True, "event_driven": True,
                        "cuda_graphs": True, "cuda_sparse": "torch"}
            assert all(execution["backend"][k] == v for k, v in expected.items()), stem
            assert execution["batch"] == 1 and execution["dt"]["lif_ms"] == 0.5, stem
            assert execution["seeds"] == {"brain": [31 + seed], "env": [seed]}, stem
            job_index = next(i for i, cmd in enumerate(plan["commands"])
                             if f"--arm {arm} --seed {seed} " in cmd)
            gpu = str(plan["execution_placement"]["gpu_pool"][job_index % 4])
            assert record["cuda_visible_devices"] == gpu, stem
            assert (directory / "fam_plume" / f"{stem}.gpu.txt").read_text().strip() == gpu
            log = (directory / "fam_plume" / f"{stem}.log").read_text(encoding="utf-8")
            assert f"device cuda arm {arm} seed {seed}" in log, stem
            assert "frame 6000/6000" in log and "Traceback" not in log, stem
            assert json.loads(log.splitlines()[-1]) == record["summary"], stem
            sources = match_sources(record["provenance"]["source_fingerprint"], ROOT)
            assert sources["verified"], (stem, sources)
            validation.append({"run": stem, "device": execution["device"],
                               "device_name": execution["device_name"], "gpu": gpu,
                               "source_match": sources})
            # Check each imported file directly too, with Windows/Linux line endings.
            for name, digest in record["provenance"]["source_fingerprint"]["files_loaded"].items():
                raw = (ROOT / name).read_bytes()
                content = raw.replace(b"\r\n", b"\n")
                assert digest in {hashlib.sha256(b).hexdigest() for b in
                                  (raw, content, content.replace(b"\n", b"\r\n"))}, (stem, name)
            values = np.asarray(record["samples"])
            assert np.allclose(values[:, 0], np.arange(1, 601) / 10, rtol=0, atol=1e-14)
            columns = dict(zip(record["columns"], values.T))
            difference = columns["orn_rate_L"] - columns["orn_rate_R"]
            measured = columns["DNa02_L"] - columns["DNa02_R"]
            total = columns["antenna_concentration_L"] + columns["antenna_concentration_R"]
            physical = np.divide(columns["antenna_concentration_L"] - columns["antenna_concentration_R"],
                                 total, out=np.zeros_like(total), where=total > 0)
            row = {"arm": arm, "env_seed": seed, "neural_seed": 31 + seed, "gpu": gpu,
                   **record["summary"], "first_feed_censored": record["summary"]["first_feed_s"] is None}
            for key in ("orn_rate_L", "orn_rate_R", "DNa02_L", "DNa02_R",
                        "target_DNa02_difference", "PFL3_input_difference", "bilateral"):
                row["mean_" + key] = float(columns[key].mean())
            row.update(
                mean_orn_difference_hz=float(difference.mean()),
                temporal_sd_orn_difference_hz=float(difference.std(ddof=1)),
                mean_DNa02_difference_hz=float(measured.mean()),
                mean_abs_tracking_error_hz=float(np.abs(measured - columns["target_DNa02_difference"]).mean()),
                mean_physical_contrast=float(physical.mean()),
                temporal_sd_filtered_contrast=float(columns["bilateral"].std(ddof=1)),
            )
            rows.append(row)
    table(out / "per_run.csv", rows)
    metrics = [key for key in rows[0] if key not in
               {"arm", "env_seed", "neural_seed", "gpu", "fed_at_least_1s", "first_feed_censored"}]
    stats = []
    for arm in batch.ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        for key in metrics:
            vals = [r[key] for r in selected if r[key] is not None]
            stats.append({"arm": arm, "metric": key, "n": len(vals),
                          "mean": float(np.mean(vals)) if vals else None,
                          "across_run_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                          "conditioning": "uncensored contacts only" if key == "first_feed_s" else "all six runs"})
    table(out / "descriptive.csv", stats)
    index = {(s["arm"], s["metric"]): s for s in stats}

    def cell(arm, metric):
        s = index[arm, metric]
        return formatted(s["mean"], s["across_run_sd"])

    lines = ["Means +/- across-run sample SD (six independent runs per arm).", "",
             "| Arm | Fed >=1 s | Feeding s | Final energy | Minimum fruit-surface distance m | Goal offset deg | Airborne s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in batch.ARMS:
        n = sum(r["fed_at_least_1s"] for r in rows if r["arm"] == arm)
        cells = [cell(arm, k) for k in ("feeding_s", "final_energy", "min_fruit_distance_m",
                                       "mean_abs_goal_offset_deg", "airborne_s")]
        lines.append(f"| {arm} | {n}/6 | " + " | ".join(cells) + " |")
    lines += ["", "| Arm | ORN L Hz | ORN R Hz | ORN L-R Hz | Temporal SD of ORN L-R Hz | DNa02 target Hz | DNa02 measured L-R Hz | Mean absolute tracking error Hz |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in batch.ARMS:
        cells = [cell(arm, k) for k in ("mean_orn_rate_L", "mean_orn_rate_R", "mean_orn_difference_hz",
                                       "temporal_sd_orn_difference_hz", "mean_target_DNa02_difference",
                                       "mean_DNa02_difference_hz", "mean_abs_tracking_error_hz")]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "All trace-derived quantities use the same 600 post-frame samples, 0.1-60 s, without a behavior mask.",
              "ORN rates are post-frame; target/plume state used frame-start rates; DNa02 motor rates are the post-frame readout.",
              "Temporal SD includes changing stimuli and behavior; it is not a stationary noise estimate or an SNR.",
              "Tracking error is the sampled target-minus-measured magnitude, not a causal lag or servo-gain estimate.",
              "Minimum fruit distance is distance to the fruit surface, including the fly's height.",
              "first_feed_s is first contact at any frame; blank means censored at 60 s. Contact-time statistics condition on uncensored runs.",
              "The >=1 s indicator uses cumulative feeding duration. Goal offset is commanded-turn magnitude, not true-bearing error."]
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    batch.save(out / "validation.json", {"runs": validation, "result_sha256": raw_hashes,
                                          "analysis_source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()})
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="fresh analysis directory")
    args = parser.parse_args()
    report(args.plan, args.out)
