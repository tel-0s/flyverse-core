"""Pin diagnostic coordinates and predeclare synthetic weights before functional testing.

Run on CPU: python scripts/build_banc_candidate.py out/banc_candidate_source
The source diagnostic must come from a clean commit, with all review controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import connectome as cn
from flyverse.banc_vision import DATA, QUALIFICATION, annotation_sha256
from flyverse.interp.common import connectome_fingerprint, raw_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diagnostic", type=Path)
    ap.add_argument(
        "--acceptance",
        type=Path,
        help="attach completed functional evidence for this exact candidate",
    )
    args = ap.parse_args()
    report = json.loads((args.diagnostic / "report.json").read_text(encoding="utf-8"))
    if (
        report["provenance"]["flyverse_commit"]["dirty"]
        or "review_controls" not in report
    ):
        raise ValueError("pin a clean committed diagnostic with --review-controls")
    banc, male = cn.load(dataset="banc", verbose=False), cn.load(verbose=False)
    fp = connectome_fingerprint(banc)
    if fp["md5"] != report["provenance"]["compiled_connectome"]["md5"]:
        raise ValueError("diagnostic and runtime biological graphs differ")
    raw_b, _ = raw_counts(banc, build=False)
    raw_m, _ = raw_counts(male, build=False)
    counts = {
        "banc": float(raw_b[banc.select(type="DNa02")].sum()),
        "malecns": float(raw_m[male.select(type="DNa02")].sum()),
    }
    scale = counts["banc"] / counts["malecns"]
    pr = np.flatnonzero(
        (male.neurons.type.eq("R1-R6") & male.neurons.hex_side.eq("R")).to_numpy()
    )
    measured, weights = {}, {}
    for t in ("L1", "L2", "L3"):
        values = raw_m[male.select(type=t)][:, pr].data
        median = float(np.median(values))
        measured[t] = {
            "pairs": len(values),
            "median": median,
            "quartiles": np.quantile(values, [0.25, 0.75]).tolist(),
        }
        weights[t] = median * scale
    content = (
        (args.diagnostic / "banc_R_candidate.csv").read_bytes().replace(b"\r\n", b"\n")
    )
    meta = {
        "mode": "candidate",
        "candidate_id": "banc-v888-right-k6-v1",
        "release": banc.release,
        "qualification": QUALIFICATION,
        "eye": "R",
        "anatomically_validated": False,
        "functional_gate": "pending one predeclared house comparison; shelve if any of eight motion directions fails",
        "columns_sha256": hashlib.sha256(content).hexdigest(),
        "base_csr_md5": fp["md5"],
        "base_annotations_sha256": annotation_sha256(banc),
        "base_manifest": banc._manifest,
        "reconstruction": {
            key: report[key]
            for key in ("generator", "versions", "thread_environment", "parameters")
        },
        "source_commit": report["provenance"]["flyverse_commit"],
        "checks": report["checks"],
        "right_eye": report["eyes"]["R"],
        "review_controls": report["review_controls"],
        "accuracy_estimate": {
            "exact_fraction": [0.80, 0.85],
            "within_one_column_fraction": [0.97, 0.98],
            "method": "label-free split-half self-consistency calibrated on FAFB, from independent review",
            "source": "docs/audits/banc_column_reconstruction_review.md",
            "assumptions": "approximate transfer of FAFB split-half calibration to BANC; not biological ground truth",
            "errors": "about one cell in six misplaced; concentrated at the rim and T4 home columns",
            "fafb_tuned_optimum": {
                "neighbours": 6,
                "exact_percent": 94.38,
                "k5_percent": 78.54,
                "k7_percent": 67.82,
            },
        },
        "dra_interpretation": "gate retained but intrinsically closed: FAFB published proxy 80/90; proxy precision 68%, recall 95%; BANC mapping 11/28 reflects annotation incompleteness",
        "input_layer": {
            "cells_per_cartridge": 6,
            "transmitter": "histamine",
            "targets": ["L1", "L2", "L3"],
            "wiring": "neural-superposition cartridge: six distinct receptors sharing a viewing direction, each wired only to mapped native lamina cells in that column",
            "weight_by_target": weights,
            "male_right_native_pair_counts": measured,
            "dn_raw_input_counts": counts,
            "banc_to_male_sampling_scale": scale,
            "reference_csr_md5": connectome_fingerprint(male)["md5"],
            "assumption": "unmeasured synthetic weights: MaleCNS per-edge median times global DNa02 input-yield ratio; not a local lamina calibration",
            "native_synapses_modified": 0,
            "missing_targets": "left missing; no borrowed FAFB graph or imputed biological edges",
        },
    }
    if args.acceptance:
        accepted = json.loads(args.acceptance.read_text(encoding="utf-8"))
        tested = accepted["provenance"]["model"]["vision"]
        for key in (
            "columns_sha256",
            "base_csr_md5",
            "base_annotations_sha256",
            "input_layer",
            "checks",
        ):
            if tested[key] != meta[key]:
                raise ValueError(
                    f"functional evidence belongs to a different candidate: {key}"
                )
        meta["functional_gate"] = (
            "recorded engineering acceptance below; anatomical gates remain closed"
        )
        meta["functional_validation"] = {
            "status": accepted["validation"]["status"],
            "summary": accepted["summary"],
            "result_run_id": accepted["run_id"],
            "result_sha256": hashlib.sha256(args.acceptance.read_bytes()).hexdigest(),
            "source": "docs/audits/banc_candidate_experiment.md",
            "scope": "one fixed right-eye motion/loom protocol; not a replicated behavioural claim or anatomical validation",
        }
    (DATA / "banc_candidate_columns.csv").write_bytes(content)
    (DATA / "banc_candidate_vision.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "columns_sha256": meta["columns_sha256"],
                "weights": weights,
                "reference": measured,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
