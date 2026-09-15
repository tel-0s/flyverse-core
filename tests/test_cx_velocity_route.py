"""scripts/cx_velocity_route.py (compass round 7): the batch plan's shape and the analysis on synthetic run files.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_cx_velocity_route.py -q

No connectome, no GPU: the plan is text and the analysis reads JSON rows shaped like cx_wedge's ledger rows.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

cvr = pytest.importorskip("cx_velocity_route")


def test_plan_batch_shape_and_job_lines():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cx8"
        jobs, calls = cvr.plan_batch(out, seeds=range(6), minutes=30, name="cx8")
        sh = (out / "batch.sh").read_text(encoding="utf-8")
        assert sh.startswith("#!/bin/bash\nset -o pipefail\n")
        assert "DRAFT" in sh and "NOT SUBMITTED" in sh
        assert len(jobs) == 48 and len(calls) == 2 and all(len(c) <= cvr.MAX_JOBS_PER_CALL for c in calls)
        assert sh.count("python scripts/cluster_run.py --name cx8 --minutes 30 --arm-block fam ") == 2
        assert sh.count("--fetch out/cx8/") == 2
        # a block never straddles two calls
        for c in calls:
            seeds = {j["seed"] for j in c}
            assert all(sum(1 for j in c if j["seed"] == s) == len(cvr.ARMS) for s in seeds)
        lines = {(j["arm"], j["seed"]): j["line"] for j in jobs}
        s0 = lines[("S", 0)]
        assert s0.startswith("mkdir -p out/cx8 && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && ")
        assert s0.endswith("; st=\\$?; tail -3 out/cx8/S_s0.txt; exit \\$st")
        assert "--preset" not in s0 and "--instrument" not in s0 and "--turn 90 --turn-window 0.5:3.5" in s0
        hgv = lines[("HGV", 3)]
        assert "--nt-override GLNO=glutamate" in hgv and "--hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)'" in hgv
        assert "--preset instrumented --instrument sided_turn_afferent:k=0.5 " in hgv
        assert "--instrument sided_turn_afferent:k=0.5:sign=-1" in lines[("HGV-", 0)]
        assert "--instrument sided_turn_afferent:k=0.25" in lines[("HGVk025", 5)]
        assert "--preset instrumented --sim-out" in lines[("HG", 1)] and "--instrument" not in lines[("HG", 1)]
        assert "--nt-override" not in lines[("V", 2)] and "--hold-edges" not in lines[("V", 2)] and "--preset instrumented" in lines[("V", 2)]
        hgvp = lines[("HGVp", 4)]                                                   # the PEN-side hold only
        assert "--hold-edges '^(ExR6|ER6|ER4m)$:^PEN_' " in hgvp and "EPG" not in hgvp.split("--hold-edges")[1].split("--preset")[0]
        assert "--nt-override GLNO=glutamate" in hgvp and "--instrument sided_turn_afferent:k=0.5 " in hgvp
        arms = json.loads((out / "arms.json").read_text(encoding="utf-8"))
        assert arms["arms"]["HGV"]["instruments"] == ["sided_turn_afferent", "ring_dc_hold", "glno_sign"]
        assert arms["arms"]["HGVp"]["instruments"] == ["sided_turn_afferent", "ring_dc_hold_pen", "glno_sign"]
        assert arms["arms"]["HGVp"]["hold"] == cvr.HOLD_PEN and arms["arms"]["HGV"]["hold"] == cvr.HOLD
        assert len(arms["family"]) == 6 and arms["family"][5][1:] == ["frac_confined_post", "HGVp", "HGV"]
        assert arms["arms"]["S"]["preset"] == "raw" and arms["status"].startswith("DRAFT")
        # no infrastructure identifiers (no user@host, no mounted run directory): the job lines name only repo paths
        assert "@" not in sh and "/mnt/" not in sh and "ssh " not in sh


def _run(arm, seed, follow, glno_lr, pen_lr, dna_lr, confined=1.0, preset=None, instruments=None, glu=None, hold=None, turn_fed=None,
         frac_confined_post=1.0):
    exp = cvr.expected(arm)
    inst = exp["instruments"] if instruments is None else instruments
    hold = exp["hold"] if hold is None else hold
    pre, post = hold.split(":", 1) if hold else (None, None)
    return {"arm": arm, "seed": seed, "device": "cuda", "preset": exp["preset"] if preset is None else preset, "instruments": inst,
            "nt_override": ({"GLNO": "glutamate"} if (exp["glutamate"] if glu is None else glu) else {}),
            "hold_edges": ([[pre, post, 0.0]] if hold else []),
            "hold_edges_resolved": ([{"n_entries": 1149 if post != "^PEN_" else 461}] if hold else []),
            "turn_deg_s": 90.0, "wall_s": 20.0,
            "provenance": {"preset": exp["preset"], "compiled_connectome": {"md5": "abc"}},
            "metrics": {"survival_s": 5.0, "bump_hz_post": 100.0, "width_half_post": 3.0, "frac_confined_post": frac_confined_post,
                        "bump_follow_wedges_per_s": follow, "bump_follow_confined_frac": confined, "bump_follow_ideal_wedges_per_s": 4.0,
                        "GLNO_LR_hz": glno_lr, "PEN_LR_hz": pen_lr, "DNa02_LR_hz": dna_lr, "PS196b_LR_hz": 1.0, "AFF_LR_hz": 22.0,
                        "turn_fed": (bool(exp["spec"]) if turn_fed is None else turn_fed)}}


def test_analyse_computes_the_family_with_holm_and_the_follow_gate():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "cx8"; runs.mkdir()
        for seed in range(6):
            n = rng.normal
            rows = {"S": _run("S", seed, 0.0 + n(0, 0.01), 0.0 + n(0, 0.05), 0.0, 0.0),
                    "V": _run("V", seed, 0.0 + n(0, 0.01), 3.0 + n(0, 0.2), 0.0, 0.0),
                    "HG": _run("HG", seed, 0.0 + n(0, 0.02), 0.0 + n(0, 0.05), 0.0 + n(0, 0.05), 0.0),
                    "HGV": _run("HGV", seed, 3.5 + n(0, 0.2), 3.0 + n(0, 0.2), 2.0 + n(0, 0.1), 0.0),
                    "HGV-": _run("HGV-", seed, -3.5 + n(0, 0.2), -3.0 + n(0, 0.2), -2.0 + n(0, 0.1), 0.0),
                    "HGVp": _run("HGVp", seed, 3.5 + n(0, 0.2), 3.0, 2.0, 0.0, frac_confined_post=0.95 + n(0, 0.01)),
                    "HGVk025": _run("HGVk025", seed, 1.5, 1.5, 1.0, 0.0), "HGVk1": _run("HGVk1", seed, 4.0, 6.0, 4.0, 0.0)}
            rows["HGV"]["metrics"]["frac_confined_post"] = 0.6 + n(0, 0.02)                # HGVp confines better: test 6
            for arm, row in rows.items():
                (runs / f"{arm}_s{seed}.json").write_text(json.dumps([row]), encoding="utf-8")
        # one HGV run with a dead bump: gated out of the follow comparisons, kept everywhere else
        dead = _run("HGV", 5, 15.0, 3.0, 2.0, 0.0, confined=0.0, frac_confined_post=0.6)
        (runs / "HGV_s5.json").write_text(json.dumps([dead]), encoding="utf-8")
        out = runs / "analysis"
        summary = cvr.analyse(runs, out, None, {})
        assert summary["n_runs"] == 48 and summary["problems"] == []
        comp = {r["test"]: r for r in summary["family"]}
        assert len(comp) == 6
        assert comp["6_frac_confined_post_HGVp_vs_HGV"]["verdict_holm"] == "result" and comp["6_frac_confined_post_HGVp_vs_HGV"]["n_stim"] == 6
        assert comp["1_bump_follow_HGV_vs_HG"]["n_stim"] == 5 and comp["1_bump_follow_HGV_vs_HG"]["n_null"] == 6      # the gate
        assert comp["1_bump_follow_HGV_vs_HG"]["verdict"] == "result" and comp["1_bump_follow_HGV_vs_HG"]["verdict_holm"] == "result"
        assert comp["2_bump_follow_HGV_vs_HGV-"]["verdict_holm"] == "result"
        assert comp["3_GLNO_LR_V_vs_S"]["verdict_holm"] == "result"
        assert comp["4_PEN_LR_HGV_vs_HG"]["verdict_holm"] == "result"
        assert comp["5_DNa02_LR_HGV_vs_HG"]["verdict"] in ("null", "undetermined")           # all zeros: no result
        assert all(r["m"] == 6 for r in comp.values())
        ps = {k: r["p"] for k, r in comp.items()}
        adj = cvr.holm(ps, m=6)
        assert all(abs(adj[k] - comp[k]["p_holm"]) < 1e-12 for k in comp)
        assert min(adj.values()) >= min(v for v in ps.values() if np.isfinite(v))
        for f in ("runs.csv", "per_seed.csv", "compare.csv", "descriptive.csv", "analysis.md", "analysis.json"):
            assert (out / f).exists(), f
        per = (out / "per_seed.csv").read_text(encoding="utf-8").splitlines()
        assert per[0] == "arm,key,seeds,values,mean,sd,n"
        hgv_follow = [ln for ln in per if ln.startswith("HGV,bump_follow_wedges_per_s,")][0]
        assert '"0,1,2,3,4,5"' in hgv_follow and "15.0000" in hgv_follow                     # ungated list carries the dead run
        # per-run checks catch a mislabelled arm
        bad = _run("HGV", 0, 3.5, 3.0, 2.0, 0.0, instruments=["sided_turn_afferent"])         # missing the hold / relabel records
        (runs / "HGV_s0.json").write_text(json.dumps([bad]), encoding="utf-8")
        summary = cvr.analyse(runs, out, None, {})
        assert any("HGV_s0.json" in p and "instruments" in p for p in summary["problems"])


def test_analyse_with_alias_needs_a_label_and_reports_no_data():
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "other"; runs.mkdir()
        row = {"arm": "H3G", "seed": 0, "device": "cuda", "metrics": {"survival_s": 5.0}, "provenance": {}}
        (runs / "H3G_s0.json").write_text(json.dumps([row]), encoding="utf-8")
        summary = cvr.analyse(runs, runs / "an", "PATH CHECK: NOT cx8", {"H3G": "HGV"})
        assert summary["arms"] == {"HGV": 1} and all(r["verdict"] == "no data" for r in summary["family"])
        md = (runs / "an" / "analysis.md").read_text(encoding="utf-8")
        assert md.startswith("# PATH CHECK: NOT cx8") and "carry no result of round 7" in md


def test_holm_step_down():
    adj = cvr.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan"), "e": None}, m=5)
    assert adj["a"] == pytest.approx(0.05) and adj["c"] == pytest.approx(0.12) and adj["b"] == pytest.approx(0.12)
    assert np.isnan(adj["d"]) and np.isnan(adj["e"])
