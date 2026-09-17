"""scripts/cx_velocity_route.py (compass round 7): the batch plan's shape and the analysis on synthetic run files.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_cx_velocity_route.py -q

No connectome, no GPU: the plan is text and the analysis reads JSON rows shaped like cx_wedge's ledger rows.
"""
from __future__ import annotations

import json
import copy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

cvr = pytest.importorskip("cx_velocity_route")
_git_bash = Path(shutil.which("git") or ".").resolve().parent.parent / "bin/bash.exe"
BASH = str(_git_bash) if _git_bash.is_file() else shutil.which("bash")


def test_plan_batch_shape_and_job_lines():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cx8"
        jobs, calls = cvr.plan_batch(out, seeds=range(6), minutes=30, name="cx8")
        sh = (out / "batch.sh").read_text(encoding="utf-8")
        assert sh.startswith("#!/bin/bash\nset -o pipefail\n")
        assert "DRAFT" in sh and "NOT SUBMITTED" in sh
        assert len(jobs) == 48 and len(calls) == 2 and all(len(c) <= cvr.MAX_JOBS_PER_CALL for c in calls)
        assert sh.count("python scripts/cluster_run.py --target house --name cx8 --minutes 30 --arm-block fam ") == 2
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
    params = dict(k_hz_per_deg_s=0.5, sign=1, cells="AN07B037", max_hz=250.0)
    if arm == "HGV-": params["sign"] = -1
    if arm == "HGVk025": params["k_hz_per_deg_s"] = 0.25
    if arm == "HGVk1": params["k_hz_per_deg_s"] = 1.0
    resolved = []
    if hold:
        n, syn, posts = (1149, 37256.0, 88) if post != "^PEN_" else (402, 7893.0, 42)
        resolved = [dict(pre=pre, post=post, factor=0.0, n_entries=n, synapses=syn, n_pre_cells=17, n_post_cells=posts)]
    records = []
    for name in exp["instruments"]:
        d = dict(name=name, kind="stop-gap", law="unverified", gap="test gap", removal="test recording", audits=["test audit"])
        if name == "sided_turn_afferent":
            d.update(parameters=params, n_cells=6, cells={"L": [1,2,3], "R": [4,5,6]})
        elif name.startswith("ring_dc_hold"):
            d.update(kind="edges", parameters=dict(pre=pre,post=post,factor=0.0),resolved=resolved[0])
        else:
            d.update(kind="relabel",parameters={"type":"GLNO","nt":"glutamate"})
        records.append(d)
    protocol = dict(gE=1.0,gD=1.0,gR=1.0,delta7_pen=True,background_hz=10.0,pulse_hz=40.0,pulse_s=2.0,
                    seconds_after=5.0,width=4,start_wedge=0,settle_s=1.0,receptor_model="sign",receptor_net_rule="abs")
    return {**protocol, "arm": arm, "seed": seed, "device": "cuda", "preset": exp["preset"] if preset is None else preset,
            "instruments": inst, "instrument_specs": [exp["spec"]] if exp["spec"] else [],
            "nt_override": ({"GLNO": "glutamate"} if (exp["glutamate"] if glu is None else glu) else {}),
            "glno_nt": ["glutamate"] if exp["glutamate"] else ["unknown"],
            "hold_edges": ([[pre, post, 0.0]] if hold else []), "hold_edges_resolved": resolved,
            "turn_deg_s": 90.0, "turn_window_s": [0.5,3.5], "wall_s": 20.0, "block": f"fam_s{seed}",
            "ledger_npz": f"{arm}_s{seed}.npz",
            "provenance": {"preset": exp["preset"], "instruments": records,
                           "compiled_connectome": {"md5": "7a10d93ba2086f2c76bcdabdca79b4ec" if exp["glutamate"] else "ef23cc27bea13be7f6a96f3c04fd3737"},
                           "source_fingerprint": {"files": {"test.py": "same-for-every-run"}},
                           "stimulus": {"params": protocol},
                           "model": {"lif": {"same_type_gain": 0.1, "dt": 0.5, "receptor_model": "sign",
                                             "adapt_by_type": {"^(EPG|PEN|PEG|Delta7)": 0.0},
                                             "type_path_gain": [[pre,post,0.0]] if hold else []}}},
            "metrics": {"survival_s": 5.0, "bump_hz_post": 100.0, "width_half_post": 3.0, "frac_confined_post": frac_confined_post,
                        "bump_follow_wedges_per_s": follow, "bump_follow_confined_frac": confined, "bump_follow_ideal_wedges_per_s": 4.0,
                        "GLNO_LR_hz": glno_lr, "PEN_LR_hz": pen_lr, "DNa02_LR_hz": dna_lr, "PS196b_LR_hz": 1.0, "AFF_LR_hz": 22.0,
                        "turn_on_s":3.5,"turn_off_s":6.5,"turn_deg_s":90.0,"frames":800,"frame_s":0.01,"bump_follow_n_frames":300,
                        "turn_fed": (bool(exp["spec"]) if turn_fed is None else turn_fed)}}


def _save_run(runs, arm, row):
    path = runs / f"{arm}_s{row['seed']}.json"
    path.write_text(json.dumps([row]),encoding="utf-8")
    path.with_suffix(".txt").write_text("synthetic fixture, device cuda",encoding="utf-8")
    np.savez(runs/row["ledger_npz"], synthetic=np.zeros(1))
    return path


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
                _save_run(runs, arm, row)
        # one HGV run with a dead bump: gated out of the follow comparisons, kept everywhere else
        dead = _run("HGV", 5, 15.0, 3.0, 2.0, 0.0, confined=0.0, frac_confined_post=0.6)
        _save_run(runs, "HGV", dead)
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
        assert per[0] == "arm,key,seeds,files,run_ids,values,mean,sd,n"
        # descriptive.csv is the per-arm MEAN pivot the analysis.md table renders, not a second copy of per_seed.csv
        # (the docstring advertised a per-arm file and the script wrote per_df to both; skeptic correction 15).
        desc = pd.read_csv(out / "descriptive.csv")
        assert desc.columns[0] == "arm" and len(desc) == len(set(desc.arm)) == 8
        assert "bump_follow_wedges_per_s" in desc.columns and "seeds" not in desc.columns
        assert (out / "descriptive.csv").read_bytes() != (out / "per_seed.csv").read_bytes()
        per_tbl = pd.read_csv(out / "per_seed.csv")
        for _, row in desc.iterrows():
            want = per_tbl[(per_tbl.arm == row["arm"]) & (per_tbl.key == "GLNO_LR_hz")]["mean"].iloc[0]
            assert abs(float(row["GLNO_LR_hz"]) - float(want)) < 1e-9, row["arm"]
        hgv_follow = [ln for ln in per if ln.startswith("HGV,bump_follow_wedges_per_s,")][0]
        assert '"0,1,2,3,4,5"' in hgv_follow and "15.0000" in hgv_follow                     # ungated list carries the dead run
        # Four eligible runs still use the actual Holm ordering. m*p_floor alone is only the first-step bound;
        # it must not discard a test that passes at a later step after smaller p values from the other primaries.
        _save_run(runs, "HGV", _run("HGV", 4, 15.0, 3.0, 2.0, 0.0, confined=0.0, frac_confined_post=0.6))
        smaller = cvr.analyse(runs, out, None, {})["family"][0]
        assert smaller["n_stim"] == 4 and smaller["first_step_holm_floor"] > 0.05
        assert smaller["verdict_holm"] == "result"
        # per-run checks catch a mislabelled arm
        bad = _run("HGV", 0, 3.5, 3.0, 2.0, 0.0, instruments=["sided_turn_afferent"])         # missing the hold / relabel records
        _save_run(runs, "HGV", bad)
        summary = cvr.analyse(runs, out, None, {})
        assert any("HGV_s0.json" in p and "instruments" in p for p in summary["problems"])


def test_analyse_with_alias_needs_a_label_and_reports_no_data():
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "other"; runs.mkdir()
        row = {"arm": "H3G", "seed": 0, "device": "cuda", "metrics": {"survival_s": 5.0}, "provenance": {}}
        (runs / "H3G_s0.json").write_text(json.dumps([row]), encoding="utf-8")
        summary = cvr.analyse(runs, runs / "an", "PATH CHECK: NOT cx8", {"H3G": "HGV"})
        assert summary["arms"] == {"HGV": 1} and all(r["verdict"] == "undetermined" for r in summary["family"])
        md = (runs / "an" / "analysis.md").read_text(encoding="utf-8")
        assert md.startswith("# PATH CHECK: NOT cx8") and "carry no result of round 7" in md


def test_holm_step_down():
    adj = cvr.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan"), "e": None}, m=5)
    assert adj["a"] == pytest.approx(0.05) and adj["c"] == pytest.approx(0.12) and adj["b"] == pytest.approx(0.12)
    assert np.isnan(adj["d"]) and np.isnan(adj["e"])


def test_independent_provenance_and_parameters_cannot_be_mislabeled(tmp_path):
    row = _run("HGV-",0,-3.5,3,2,1)
    path = _save_run(tmp_path,"HGV-",row)
    assert cvr.check_run(row,cvr.expected("HGV-"),path)==[]
    mutations = [
        lambda r:r["provenance"].update(preset="raw"),
        lambda r:r["provenance"]["instruments"][0]["parameters"].update(sign=1),
        lambda r:r["provenance"]["instruments"][0]["parameters"].update(k_hz_per_deg_s=1),
        lambda r:r.update(turn_window_s=[0,3]),
        lambda r:r["hold_edges"][0].__setitem__(2,1.0),
        lambda r:r["hold_edges_resolved"][0].update(n_entries=402),
        lambda r:r["provenance"]["model"]["lif"].update(same_type_gain=1.0),
        lambda r:r.update(lif_overrides={"conn_cap":0}),
    ]
    for change in mutations:
        bad=copy.deepcopy(row);change(bad)
        assert cvr.check_run(bad,cvr.expected("HGV-"),path)


def test_duplicate_and_missing_runs_never_form_a_valid_batch(tmp_path):
    row=_run("V",0,0,3,0,0)
    path=_save_run(tmp_path,"V",row)
    (tmp_path/"copy_s0.json").write_text(path.read_text(),encoding="utf-8")
    summary=cvr.analyse(tmp_path,tmp_path/"analysis",None,{})
    assert any("duplicate" in p for p in summary["problems"])
    assert any("48 distinct" in p for p in summary["problems"])
    assert not summary["valid_batch"]
    assert all(r["verdict_holm"]=="undetermined" for r in summary["family"])


def test_predeclaration_is_immutable_and_records_the_job_hash(tmp_path):
    import hashlib
    cvr.plan_batch(tmp_path,range(6))
    frozen=cvr.predeclare(tmp_path)
    assert frozen["batch_sha256"]==hashlib.sha256((tmp_path/"batch.sh").read_bytes()).hexdigest()
    assert frozen["written_before_submission"] and frozen["follow_gate_confined_frac"]==0.5
    assert len(frozen["family"])==6 and frozen["protocol"]["seconds_after"]==5.0
    with pytest.raises(FileExistsError): cvr.predeclare(tmp_path)
    with pytest.raises(ValueError): cvr.plan_batch(tmp_path,range(6))


def test_frozen_protocol_checks_the_probe_actually_loaded(tmp_path):
    cvr.plan_batch(tmp_path, range(6))
    frozen = cvr.predeclare(tmp_path)
    row = _run("V", 0, 0, 0, 0, 0)
    row["provenance"]["model"]["lif"] = frozen["resolved_lif_by_arm"]["V"]
    loaded = {p: frozen["source_sha256_lf"][p] for p in
              ("scripts/cx_wedge.py", "scripts/probe_compass_room.py")}
    row["provenance"]["source_fingerprint"]["files_loaded"] = loaded
    _save_run(tmp_path, "V", row)
    _, problems = cvr.load_runs(tmp_path, {})
    assert not any("loaded" in p for p in problems)  # batch is incomplete, but its probe hashes match
    loaded["scripts/cx_wedge.py"] = "different-code"
    _save_run(tmp_path, "V", row)
    _, problems = cvr.load_runs(tmp_path, {})
    assert any("loaded simulation source differs" in p for p in problems)


@pytest.mark.skipif(BASH is None, reason="bash is needed to exercise submission wrappers")
def test_shell_preserves_job_failure_and_stops_before_second_client(tmp_path):
    jobs, _ = cvr.plan_batch(tmp_path / "out/cx8", range(6))
    activate = tmp_path / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("# synthetic environment\n", encoding="utf-8")
    job = jobs[0]["line"].replace("\\$", "$")
    # This shell function replaces Python completely. No simulation, CUDA import or remote request occurs.
    prefix = 'python() { if [ "$1" = "-c" ]; then return 0; fi; return 19; };\n'
    proc = subprocess.run([BASH, "-c", prefix + job], cwd=tmp_path, capture_output=True)
    assert proc.returncode == 19
    wrapper = (tmp_path / "out/cx8/batch.sh").read_text(encoding="utf-8")
    prefix = 'python() { echo client-attempt >> client_calls.txt; return 17; };\n'
    # Exercise the script as a file, as in production; its 48 job arguments exceed some Windows -c limits.
    stub = tmp_path / "wrapper_test.sh"
    stub.write_text(prefix + wrapper, encoding="utf-8", newline="\n")
    proc = subprocess.run([BASH, stub.name], cwd=tmp_path, capture_output=True)
    assert proc.returncode == 17
    assert (tmp_path / "client_calls.txt").read_text().splitlines() == ["client-attempt"]
