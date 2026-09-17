"""The optional antenna-rate cue stays on the neural boundary."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_navigation_instruments import graph, plume_inputs

from flyverse.brain import LIFParams
from flyverse.connectome import Connectome
from flyverse.fly import FlyBrain
from flyverse.instruments import make_instrument
from flyverse.navigation import PlumeNavigation


def sensory_graph():
    rows = graph().neurons.to_dict("records")
    links = []
    for glom, counts in (("DM1", (2, 5)), ("DM2", (4, 1))):
        for side, count in zip(("L", "R"), counts):
            pn = len(rows)
            rows.append(
                dict(
                    bodyId=pn + 1,
                    type=glom + "_adPN",
                    instance=glom + "_adPN_" + side,
                    somaSide=side,
                    superclass="cb_intrinsic",
                    nt="acetylcholine",
                    **{"class": "", "subclass": ""},
                )
            )
            for _ in range(count):
                pre = len(rows)
                rows.append(
                    dict(
                        bodyId=pre + 1,
                        type="ORN_" + glom,
                        instance="ORN_" + glom,
                        somaSide=None,
                        superclass="cb_sensory",
                        nt="acetylcholine",
                        **{"class": "olfactory", "subclass": ""},
                    )
                )
                links.append((pn, pre))
    n = pd.DataFrame(rows)
    w = sp.lil_matrix((len(n), len(n)), dtype=np.float32)
    for post, pre in links:
        w[post, pre] = 20
    return Connectome(n, w.tocsr(), pd.Series(np.arange(len(n)), index=n.bodyId))


def brain(c, plume, batch=2):
    return FlyBrain(
        c,
        device="cpu",
        optic=None,
        seed=17,
        batch=batch,
        preset="instrumented",
        instruments=["compass", plume, "hunger"],
        lif_params=LIFParams(receptor_model=None),
    )


def test_default_and_explicit_legacy_are_exact():
    c = graph()
    a, b = [
        brain(c, name)
        for name in ("plume", "plume:bilateral=concentration:feedback=on")
    ]
    for i in range(35):
        for fb in (a, b):
            fb.smell({"DM1": [0.11, 0.1]}, {"DM1": [0.1, 0.11]})
            fb.wind([1, 0], [0, 1])
            fb.interoception([0.1, 0.6], airborne=[False, i > 20])
            fb.proprioception(0, 0, 0, False, yaw_rate=[0.3, -0.2])
            fb.step(10)
        torch.testing.assert_close(a.brain.rate, b.brain.rate, rtol=0, atol=0)
        for key, value in a.instruments["plume"].state_dict().items():
            torch.testing.assert_close(
                value, b.instruments["plume"].state_dict()[key], rtol=0, atol=0
            )
    assert a.instruments["plume"].describe() == b.instruments["plume"].describe()


def test_balanced_groups_no_artificial_laterality_and_correct_rate_sign():
    c = sensory_graph()
    m = PlumeNavigation(c, bilateral="orn")
    m.reset(3, "cpu")
    inputs = plume_inputs(m)
    # Same per-glomerulus rates, different numbers of cells on each side.
    for side in ("L", "R"):
        ty = c.neurons.type.to_numpy()[m.reads["antenna_" + side]]
        r = np.where(ty == "ORN_DM1", 15.0, 55.0)
        inputs["antenna_" + side] = torch.tensor(
            np.tile(r, (3, 1)), dtype=torch.float32
        )
    inputs["antenna_L"][1] *= 1.1
    inputs["antenna_R"][2] *= 1.1
    for _ in range(300):
        m.step(10, inputs)
    assert abs(m.bilateral[0].item()) < 1e-7
    assert m.goal[1] > 0 and m.goal[2] < 0
    torch.testing.assert_close(m.goal[1], -m.goal[2], atol=2e-6, rtol=0)


def test_frame_reads_brain_rate_and_never_physical_concentrations():
    c = sensory_graph()
    fb = brain(c, "plume:bilateral=orn")
    m = fb.instruments["plume"]
    assert m.observe_smell is None
    assert m.quantity_in == "rate_hz"
    values = plume_inputs(m)
    for side, hz in (("L", 25.0), ("R", 20.0)):
        values["antenna_" + side] = torch.full((2, len(m.reads["antenna_" + side])), hz)
    for key, value in values.items():
        fb.brain.rate[:, m.reads[key]] = value
    state = m.state_dict()
    fb.smell({"DM1": 1e6}, {"DM1": 0.0})
    for key, value in state.items():
        torch.testing.assert_close(value, m.state_dict()[key], rtol=0, atol=0)
    fb.step(10)
    torch.testing.assert_close(m.odor_L, torch.full((2, 1), 25.0), atol=2e-6, rtol=0)
    torch.testing.assert_close(m.odor_R, torch.full((2, 1), 20.0), atol=2e-6, rtol=0)
    assert torch.all(m.bilateral > 0)
    # Reversing measured neural rates reverses the cue despite opposite physical input.
    m.bilateral.zero_()
    fb.brain.rate[:, m.reads["antenna_L"]] = 20
    fb.brain.rate[:, m.reads["antenna_R"]] = 25
    fb.step(10)
    assert torch.all(m.bilateral < 0)


def test_transduced_lifecycle_and_configuration_identity():
    c = sensory_graph()
    fb = brain(c, "plume:bilateral=orn")
    fb.smell({"DM1": [0.7, 0.2]}, {"DM1": [0.2, 0.7]})
    fb.step(150)
    state = fb.state_dict()
    fb.step(50)
    expected = fb.brain.rate.clone()
    cue = fb.instruments["plume"].bilateral.clone()
    fb.load_state_dict(state)
    fb.step(50)
    torch.testing.assert_close(fb.brain.rate, expected, rtol=0, atol=0)
    torch.testing.assert_close(fb.instruments["plume"].bilateral, cue, rtol=0, atol=0)
    ptr = fb.instruments["plume"].bilateral.data_ptr()
    fb.reset([0])
    assert fb.instruments["plume"].bilateral.data_ptr() == ptr
    assert fb.instruments["plume"].bilateral[0] == 0
    assert fb.instruments["plume"].bilateral[1] == cue[1]
    with pytest.raises(ValueError):
        brain(c, "plume").load_state_dict(state)
    description = fb.instruments["plume"].describe()
    assert (
        description["replaces"] == "computation" and description["law"] == "unverified"
    )
    assert "brain.rate" in description["input"]
    assert "no physical concentration read" in description["input"]
    assert description["parameters"]["rate_contrast_gain_status"].startswith(
        "unverified"
    )
    json.dumps(description)


def test_missing_groups_and_invalid_options_fail():
    with pytest.raises(ValueError, match="matched"):
        PlumeNavigation(graph(), bilateral="orn")
    for name in (
        "plume:bilateral=pn",
        "plume:feedback=yes",
        "plume:bilateral=orn:bilateral=orn",
        "plume:unknown=1",
    ):
        with pytest.raises(ValueError):
            make_instrument(graph(), name)


def test_goal_only_has_no_feedback_even_after_loading_nonzero_integral():
    c = graph()
    a, b = [PlumeNavigation(c, feedback=False) for _ in range(2)]
    for m in (a, b):
        m.reset(2, "cpu")
        m.observe_wind([0, 1], [1, 0])
        m.steer_integral.fill_(50)
    ia, ib = plume_inputs(a), plume_inputs(b)
    ib["dna_L"].fill_(100)
    for _ in range(100):
        oa, ob = a.step(10, ia), b.step(10, ib)
    for key in oa:
        torch.testing.assert_close(oa[key], ob[key], rtol=0, atol=0)
    assert torch.count_nonzero(a.steer_integral) == 0
    assert "disabled" in a.describe()["parameters"]["steering_feedback"]


def test_draft_has_eighteen_independent_guarded_rooms(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import plume_transduced_batch as batch

    from flyverse import connectome

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "hashes", lambda: {"fixture": "hash"})
    monkeypatch.setattr(
        batch.subprocess, "check_output", lambda *a, **k: "test-commit\n"
    )
    monkeypatch.setattr(connectome, "load", sensory_graph)
    monkeypatch.setattr(connectome, "CACHE_DIR", tmp_path)
    for name in ("neurons.parquet", "W_post_pre.npz", "sign0_counts.npz"):
        (tmp_path / name).write_bytes(b"fixture")
    out = tmp_path / "draft"
    batch.plan(out)
    p = json.loads((out / "predeclared.json").read_text(encoding="utf-8"))
    assert len(p["commands"]) == 18
    assert len({tuple(s["position"]) for s in p["starts"]}) == 6
    assert p["arms"]["goal_only"][1] == "plume:feedback=off"
    for arm in batch.ARMS:
        assert sum(f"--arm {arm} " in command for command in p["commands"]) == 6
    shell = (out / "batch.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in shell and "FLYVERSE_PLUME_GPU_RELEASED:-0" in shell
    assert "--target house" in shell and "--arm-block fam" in shell
    with pytest.raises(FileExistsError):
        batch.plan(out)
    monkeypatch.delenv("FLYVERSE_PLUME_GPU_RELEASED", raising=False)
    with pytest.raises(RuntimeError, match="pending"):
        batch.room(None)  # guard precedes any graph, CUDA or room construction


def test_room_analysis_uses_runs_and_rejects_incomplete_records(tmp_path):
    import hashlib
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import plume_transduced_batch as batch

    plan = {
        "starts": [{"seed": s} for s in range(6)],
        "provenance": {
            "preset": "instrumented",
            "instruments": {a: [a] for a in batch.ARMS},
        },
    }
    path = tmp_path / "predeclared.json"
    batch.save(path, plan)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for arm in batch.ARMS:
        for seed in range(6):
            summary = {
                "feeding_s": seed,
                "fed_at_least_1s": seed > 0,
                "first_feed_s": 30 if seed else None,
                "final_energy": 0.1,
                "airborne_s": 0,
                "min_fruit_distance_m": 0.02,
                "mean_abs_goal_offset_deg": 5,
            }
            batch.save(
                tmp_path / f"{arm}_{seed}.json",
                {
                    "arm": arm,
                    "seed": seed,
                    "declaration_sha256": digest,
                    "start": plan["starts"][seed],
                    "provenance": {
                        "preset": "instrumented",
                        "instruments": [arm],
                        "execution": {"device": "cuda:0"},
                    },
                    "columns": ["time_s"],
                    "samples": [[(i + 1) * 0.1] for i in range(600)],
                    "summary": summary,
                },
            )
    args = SimpleNamespace(plan=path, out=tmp_path / "summary.json")
    batch.analyse(args)
    data = json.loads(args.out.read_text(encoding="utf-8"))
    assert data["rows"][0]["feeding_s"]["mean"] == 2.5
    assert data["rows"][0]["feeding_s"]["across_run_sd"] == pytest.approx(
        np.std(range(6), ddof=1)
    )
    assert data["rows"][0]["feed_count"] == 5
    assert data["rows"][0]["first_feed_censored_count"] == 1
    bad = tmp_path / "full_0.json"
    record = json.loads(bad.read_text(encoding="utf-8"))
    record["samples"].pop()
    bad.write_text(json.dumps(record), encoding="utf-8")
    args.out = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="incomplete"):
        batch.analyse(args)
