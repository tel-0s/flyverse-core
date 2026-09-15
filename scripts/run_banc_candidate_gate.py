"""One house submission: fresh right-eye MaleCNS/BANC motion and looming probes.

No tuning or retries of failed functional gates. Exit 0 means measurements completed;
acceptance.json alone records acceptance/shelving. Infrastructure errors remain nonzero.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyverse import connectome as cn
from flyverse.banc_vision import QUALIFICATION
from flyverse.interp.common import Result, provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    # New female cache compiled with this snapshot's aliases; shared caches untouched.
    os.environ["FLYVERSE_DATA_BANC"] = str(out / "raw")
    os.environ["FLYVERSE_CACHE"] = str(out / "biological")

    def run(script, flags, label):
        cmd = [sys.executable, str(ROOT / "scripts" / script), *map(str, flags)]
        print(label, " ".join(cmd), flush=True)
        with (results / f"{label}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode:
            print((results / f"{label}.log").read_text(encoding="utf-8"), flush=True)
            raise subprocess.CalledProcessError(completed.returncode, cmd)

    run("fetch_data.py", ["--banc"], "fetch")
    base = cn.load(dataset="banc", verbose=False)
    candidate_path = out / "candidate"
    c = cn.load(
        dataset="banc",
        vision="candidate",
        vision_cache_dir=candidate_path,
        verbose=False,
    )
    biological = c.W[: base.n, : base.n].tocsr()
    if any(
        getattr(biological, key).tobytes() != getattr(base.W, key).tobytes()
        for key in ("data", "indices", "indptr")
    ):
        raise RuntimeError("candidate changed the biological CSR block")
    measured = {}
    for dataset in ("malecns", "banc"):
        shared = ["--dataset", dataset, "--eye", "right", "--device", "cuda"]
        if dataset == "banc":
            shared += ["--vision", "candidate", "--vision-cache", candidate_path]
        for probe in ("motion", "loom"):
            label = f"{dataset}_{probe}"
            path = results / f"{label}.json"
            flags = shared + ["--json", path]
            if probe == "loom":
                flags += ["--side", "right", "--seed", "0"]
            run(f"probe_{probe}.py", flags, label)
            measured[label] = Result.load(path)
            print(label, measured[label].summary, flush=True)
    motion_ok = all(
        measured[f"{ds}_motion"].summary["all_eight_correct"]
        for ds in ("malecns", "banc")
    )
    loom_ok = all(
        measured[f"{ds}_loom"].summary["gf_gate_pass"] for ds in ("malecns", "banc")
    )
    status = "functional gates passed" if motion_ok and loom_ok else "shelved"
    p = provenance(
        c,
        device="cuda",
        seeds=[0],
        batch=1,
        stimulus={
            "protocol": "banc_candidate_functional_gate",
            "params": {"eye": "right"},
            "control": "fresh matched MaleCNS; cross-dataset comparison, not a sex-effect estimate",
        },
    )
    aggregate = Result.new(
        "ledger",
        p,
        summary={
            "status": status,
            "qualification": QUALIFICATION,
            "motion_pass": motion_ok,
            "loom_pass": loom_ok,
            "measurements": {key: value.summary for key, value in measured.items()},
            "anatomically_validated": False,
        },
        tables={
            "motion": [
                dict(dataset=ds, **row)
                for ds in ("malecns", "banc")
                for row in measured[f"{ds}_motion"].tables["per_type"]
            ]
        },
        validation={
            "name": "BANC candidate functional acceptance",
            "reference": {},
            "source": "owner request",
            "status": "pass" if motion_ok and loom_ok else "fail",
            "measured": {"motion": motion_ok, "loom": loom_ok},
        },
        files={key: f"{key}.json" for key in measured},
    )
    aggregate.save(results / "acceptance.json")
    print(json.dumps(aggregate.summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
