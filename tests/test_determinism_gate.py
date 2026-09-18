"""scripts/determinism_gate.py (round 8, item 1): the generated plan's shape, including --ship.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_determinism_gate.py -q

No connectome, no GPU: the plan is text. `--ship PATH[,PATH]` exists so that the committed plan reproduces a
submission made against a target checkout behind origin/main (docs/audits/determinism_gate.md sections 2 and 5).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

dg = pytest.importorskip("determinism_gate")
_git_bash = Path(shutil.which("git") or ".").resolve().parent.parent / "bin/bash.exe"
BASH = str(_git_bash) if _git_bash.is_file() else shutil.which("bash")


def test_plan_without_ship_names_no_ship_flag(tmp_path):
    out = tmp_path / "det1"
    dg.plan(out)
    sh = (out / "batch.sh").read_text(encoding="utf-8")
    assert sh.startswith("#!/bin/bash\nset -o pipefail\n")
    assert "--ship" not in sh
    assert "--gpu-ids 4,5,6,7 --name det1 --minutes 40 " in sh
    assert "ship" not in json.loads((out / "jobs.json").read_text(encoding="utf-8"))


def test_plan_with_ship_puts_the_flag_on_the_call_and_in_jobs_json(tmp_path):
    out = tmp_path / "det1"
    dg.plan(out, ship="flyverse")
    sh = (out / "batch.sh").read_text(encoding="utf-8")
    assert sh.count("--gpu-ids 4,5,6,7 --ship flyverse --name det1 --minutes 40 ") == 1
    assert sh.count("python scripts/cluster_run.py ") == 1          # one call, five jobs
    assert "# --ship flyverse: the named tracked paths are copied" in sh
    assert json.loads((out / "jobs.json").read_text(encoding="utf-8"))["ship"] == "flyverse"
    # the two-path form travels verbatim, and an empty value is the same as no flag
    dg.plan(tmp_path / "b" / "det1", ship="flyverse,scripts")
    assert "--ship flyverse,scripts --name" in (tmp_path / "b" / "det1" / "batch.sh").read_text(encoding="utf-8")
    dg.plan(tmp_path / "c" / "det1", ship="  ")
    assert "--ship" not in (tmp_path / "c" / "det1" / "batch.sh").read_text(encoding="utf-8")


def test_a_bad_ship_value_is_refused_and_the_plan_is_bash_clean(tmp_path):
    for bad in ("--fetch", "flyverse scripts"):
        with pytest.raises(ValueError):
            dg.plan(tmp_path / bad.replace(" ", "_").replace("-", "") / "det1", ship=bad)
    out = tmp_path / "ok" / "det1"
    dg.plan(out, ship="flyverse")
    if BASH:
        assert subprocess.run([BASH, "-n", str(out / "batch.sh")], capture_output=True).returncode == 0
