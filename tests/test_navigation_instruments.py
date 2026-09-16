"""Navigation laws, composition, body inputs, and module lifecycle on a small CPU graph."""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from test_compass_driver import graph as epg_graph

from flyverse.brain import LIFParams
from flyverse.connectome import Connectome
from flyverse.fly import FlyBrain
from flyverse.instruments import add_cli_arguments
from flyverse.motor import LH_ODOUR_CHANNELS
from flyverse.navigation import (
    HungerGain,
    PlumeNavigation,
    RecurrentCompass,
    pfl3_rates,
)


def graph():
    rows = epg_graph().neurons.to_dict("records")
    for ty in [
        *dict.fromkeys(t for ts in LH_ODOUR_CHANNELS.values() for t in ts),
        "PFL3",
        "DNp09",
        "DLMn1",
        "b1 MN",
    ]:
        for side in ("L", "R"):
            motor = ty in ("DLMn1", "b1 MN")
            rows.append(
                dict(
                    bodyId=len(rows) + 1,
                    type=ty,
                    instance=ty + "_" + side,
                    somaSide=side,
                    superclass="vnc_motor" if motor else "cb_intrinsic",
                    nt="acetylcholine",
                    **{
                        "class": "motor" if motor else "",
                        "subclass": "wm" if motor else "",
                    },
                )
            )
    n = pd.DataFrame(rows)
    return Connectome(
        n,
        sp.csr_matrix((len(n), len(n)), dtype=np.float32),
        pd.Series(np.arange(len(n)), index=n.bodyId),
    )


def make(names, batch=2):
    return FlyBrain(
        graph(),
        device="cpu",
        optic=None,
        seed=17,
        batch=batch,
        preset="instrumented",
        instruments=names,
        lif_params=LIFParams(receptor_model=None),
    )


def plume_inputs(m, heading=0.0, odor=1.0):
    headings = np.broadcast_to(heading, (m.B,))[:, None]
    angles = np.angle(m._epg_weights[:, 0] + 1j * m._epg_weights[:, 1])
    delta = (angles - headings + math.pi) % (2 * math.pi) - math.pi
    values = {
        "epg": torch.tensor(
            50 * np.exp(-0.5 * (delta / 0.35) ** 2), dtype=torch.float32
        )
    }
    for k, base in m.parameters["odor_baseline_hz"].items():
        values["odor_" + k] = torch.full(
            (m.B, len(m.reads["odor_" + k])),
            base + odor * m.parameters["odor_range_hz"][k],
        )
    return values


def test_plural_alias_order_and_repeated_options():
    p = argparse.ArgumentParser()
    add_cli_arguments(p)
    assert p.parse_args([]).instrument == []
    assert p.parse_args(
        [
            "--instruments",
            "compass",
            "plume",
            "--instrument",
            "hunger",
            "--instruments",
            "flight",
        ]
    ).instrument == ["compass", "plume", "hunger", "flight"]
    assert p.parse_args(
        ["--instrument", "compass", "--instrument", "plume"]
    ).instrument == ["compass", "plume"]
    with pytest.raises(SystemExit):
        p.parse_args(["--instruments"])


@pytest.mark.parametrize(
    "names,match",
    [
        (["compass", "compass_ring"], "incompatible"),
        (["compass", "compass"], "unique"),
        (["hunger"], "requires"),
        (["plume"], "requires"),
        (["not-an-instrument"], "unknown"),
    ],
)
def test_invalid_composition_precedes_brain_allocation(names, match, monkeypatch):
    def unexpected(*a, **kw):
        raise AssertionError("allocated brain before rejecting composition")

    monkeypatch.setattr("flyverse.brain.Brain", unexpected)
    with pytest.raises(ValueError, match=match):
        make(names)


def test_dynamic_attach_detach_keeps_dependencies_and_hunger_binding():
    fb = make(["compass"])
    with pytest.raises(ValueError, match="incompatible"):
        fb.attach(RecurrentCompass(fb.c))
    with pytest.raises(ValueError, match="requires"):
        fb.attach(HungerGain(fb.c))
    p = fb.attach(PlumeNavigation(fb.c))
    h = fb.attach(HungerGain(fb.c))
    assert p.hunger is h
    with pytest.raises(ValueError, match="requires"):
        fb.detach("compass")
    with pytest.raises(ValueError, match="requires"):
        fb.detach("plume")
    fb.detach("hunger")
    assert p.hunger is None
    fb.detach("plume")
    fb.detach("compass")
    assert fb.instruments == {} and fb.attached_modules == {}


def test_capture_excludes_replaced_spike_count_input_storage():
    fb = make(["compass", "plume", "hunger", "flight"])
    assert fb._extensions.can_capture_frame()
    # Even an explicitly safe module cannot make the scheduler's replaced count buffer persistent.
    fb.instruments["plume"].quantity_in = "spike_count"
    assert not fb._extensions.can_capture_frame()


def test_composition_order_neural_boundary_and_checkpoint():
    names = ["compass", "plume", "hunger", "flight"]
    a, b = make(names), make(names[::-1])
    before = a.c.W.copy()
    for fb in (a, b):
        fb.interoception([0.2, 0.8], airborne=[True, False])
        fb.proprioception(0, 0, 0, False, yaw_rate=[0.4, -0.2])
        fb.step(70.0)
    torch.testing.assert_close(a.brain.rate, b.brain.rate, rtol=0, atol=0)
    for k in names:
        for field, value in a.instruments[k].state_dict().items():
            torch.testing.assert_close(
                value, b.instruments[k].state_dict()[field], rtol=0, atol=0
            )
    state = a.state_dict()
    a.step(50.0)
    expected = a.brain.rate.clone()
    a.load_state_dict(state)
    a.step(50.0)
    torch.testing.assert_close(expected, a.brain.rate, rtol=0, atol=0)
    assert np.array_equal(before.data, a.c.W.data) and np.array_equal(
        before.indices, a.c.W.indices
    )
    from flyverse.interp.common import provenance

    for rec in provenance(a.c, fb=a)["instruments"]:
        assert rec["law"] == "unverified" and rec["audits"]
    preserved = {
        k: {n: v[1].clone() for n, v in m.state_dict().items()}
        for k, m in a.instruments.items()
    }
    a.reset([0])
    assert a.instruments["hunger"].level[0] == 1
    for k, m in a.instruments.items():
        for n, v in m.state_dict().items():
            torch.testing.assert_close(v[1], preserved[k][n], rtol=0, atol=0)


def test_interoception_validates_all_fields_before_changing_state():
    fb = make(["compass", "plume", "hunger"])
    assert "interoception" in fb.available_senses
    fb.interoception([0.1, 0.8], sated=[False, True])
    torch.testing.assert_close(
        fb.instruments["hunger"].level, torch.tensor([[0.9], [0.0]])
    )
    for kwargs in (
        {"energy": [-0.1, 0.5]},
        {"energy": float("nan")},
        {"energy": 0.4, "sated": 0.5},
        {"energy": [0.4]},
    ):
        saved = fb.instruments["hunger"].level.clone()
        with pytest.raises(ValueError):
            fb.interoception(**kwargs)
        torch.testing.assert_close(
            saved, fb.instruments["hunger"].level, rtol=0, atol=0
        )
    state = fb.instruments["hunger"].state_dict()
    state["level"][0] = -1
    with pytest.raises(ValueError, match="outside"):
        fb.instruments["hunger"].load_state_dict(state)


def test_published_pfl_comparator_turns_toward_goal_over_heading_grid():
    heading = torch.linspace(-math.pi, math.pi, 97)[:, None]
    for error in (-math.pi / 2, -math.pi / 4, math.pi / 4, math.pi / 2):
        rates = pfl3_rates(heading, heading + error)
        turn = rates[:, 1].mean(1) - rates[:, 0].mean(1)
        assert bool((turn * error > 0).all())
        torch.testing.assert_close(
            rates,
            pfl3_rates(heading + 2 * math.pi, heading + error + 2 * math.pi),
            rtol=1e-5,
            atol=5e-5,
        )
    # Independent scalar evaluation of one published model cell, not a regression snapshot.
    r = pfl3_rates(torch.tensor([[0.0]]), torch.tensor([[math.pi / 2]]))[0, 0, 0]
    x = math.cos(math.radians(67.5)) + 0.63 * math.cos(math.radians(105))
    assert r.item() == pytest.approx(
        29.23 * math.log1p(math.exp(2.17 * (x - 0.7))), rel=1e-6
    )


def test_plume_uses_heading_wind_odor_and_optional_hunger_without_oracles():
    fb = make(["compass", "plume", "hunger"])
    m = fb.instruments["plume"]
    fb.interoception([0.2, 0.8])
    fb.wind([1.0, 1.0], [0.0, 0.0])  # incoming wind 45 degrees left
    inputs = plume_inputs(m)
    for _ in range(200):
        out = m.step(10, inputs)
    assert bool((out["pfl_R"] > 0).all()) and torch.count_nonzero(out["pfl_L"]) == 0
    assert (out["pfl_R"][0] / out["pfl_R"][1]).mean().item() == pytest.approx(
        4.0, rel=1e-5
    )
    entry = m.entry_x.clone()
    m.observe_wind([0.0, 0.0], [1.0, 1.0])
    out = m.step(10, inputs)
    assert bool((out["pfl_L"] > 0).all())
    # After odor loss, walking returns toward the learned entry heading, despite changed wind.
    no_odor = plume_inputs(m, heading=math.pi / 2, odor=0.0)
    for _ in range(200):
        out = m.step(10, no_odor)
    assert bool((out["pfl_L"] > 0).all())
    torch.testing.assert_close(m.entry_x, entry, rtol=0, atol=0)
    # Weak/unconfined compass has no meaningful goal steering or forward demand.
    no_odor["epg"].fill_(10.0)
    out = m.step(10, no_odor)
    assert all(torch.count_nonzero(v) == 0 for v in out.values())
    fb.interoception(0.2, feeding=True)
    out = m.step(10, inputs)
    assert all(torch.count_nonzero(v) == 0 for v in out.values())


def test_recurrent_memory_moves_both_ways_and_eb_brake_is_separate_counterfactual():
    for eb in (0.0, 2.7):
        m = RecurrentCompass(graph(), eb_ratio=eb)
        m.reset(3, "cpu")
        m.observe_turn([math.pi / 2, -math.pi / 2, 0])
        phase = []
        for _ in range(400):
            m.step(10, {})
            phase.append(
                np.angle(m.e.numpy() @ np.exp(1j * np.arange(16) * 2 * np.pi / 16))
            )
        delta = np.unwrap(phase, axis=0)[-1] - np.unwrap(phase, axis=0)[100]
        if eb == 0:
            assert delta[0] > math.pi and delta[1] < -math.pi and abs(delta[2]) < 0.01
        else:
            assert np.max(np.abs(delta)) < 0.01
        assert not hasattr(
            m, "phase"
        )  # memory is the recurrent activity, not an integrated scalar


def test_flight_needs_explicit_body_input_and_has_sided_bounded_neural_output():
    fb = make(["flight"])
    m = fb.instruments["flight"]
    inputs = {k: torch.zeros((2, len(ix))) for k, ix in m.reads.items()}
    assert all(torch.count_nonzero(v) == 0 for v in m.step(10, inputs).values())
    fb.interoception([0.6, 0.01])
    inputs["pfl_R"].fill_(40.0)
    # Healthy flies first get time to search on foot; a starving row never requests flight.
    for _ in range(1900):
        assert all(torch.count_nonzero(v) == 0 for v in m.step(10, inputs).values())
    for _ in range(110):
        out = m.step(10, inputs)
    assert (
        out["power"][0].mean() > 100
        and out["steer_L"][0].mean() > out["steer_R"][0].mean()
    )
    assert all(torch.count_nonzero(v[1]) == 0 for v in out.values())
    fb.interoception(0.5, feeding=True)
    assert all(torch.count_nonzero(v) == 0 for v in m.step(10, inputs).values())


def test_flight_bouts_end_and_do_not_relaunch_until_ground_search():
    fb = make(["flight"])
    m = fb.instruments["flight"]
    inputs = {k: torch.zeros((2, len(ix))) for k, ix in m.reads.items()}
    # A spontaneous/native takeoff may be assisted, but only for one bounded bout.
    fb.interoception(0.8, airborne=True)
    powered = 0
    for _ in range(1200):
        powered += bool(m.step(10, inputs)["power"].any())
    assert 800 <= powered <= 801
    assert bool((m.landing == 1).all())
    # Losing the cue or restoring energy cannot restart lift before touchdown.
    fb.interoception(1.0, airborne=True)
    assert not m.step(10, inputs)["power"].any()
    fb.interoception(0.8)
    for _ in range(2000):
        assert not m.step(10, inputs)["power"].any()
    for _ in range(3):
        out = m.step(10, inputs)
    assert bool((out["power"] > 0).all())
    # Even a failed takeoff times out; it cannot accumulate a permanent launch request.
    for _ in range(900):
        out = m.step(10, inputs)
    assert not out["power"].any() and not m.active.any()


def test_flight_reserve_and_neural_odor_land_independently_with_hysteresis():
    fb = make(["flight", "hunger"])
    m = fb.instruments["flight"]
    inputs = {k: torch.zeros((2, len(ix))) for k, ix in m.reads.items()}
    fb.interoception([0.6, 0.6], airborne=True)
    assert bool((m.step(10, inputs)["power"] > 0).all())
    fb.interoception([0.3, 0.6], airborne=True)
    for k in m.parameters["odor_channels"]:
        inputs["odor_" + k][1].fill_(40.0)
    for _ in range(100):
        out = m.step(10, inputs)
    assert not out["power"].any()
    assert m.reserve_low[:, 0].tolist() == [1, 0]
    assert m.odor_near[:, 0].tolist() == [0, 1]
    saved = m.state_dict()
    # Odor lost during descent; low energy raised but still below the re-arm threshold.
    for value in inputs.values():
        value.zero_()
    fb.interoception([0.45, 0.6], airborne=True)
    for _ in range(250):
        assert not m.step(10, inputs)["power"].any()
    assert not m.odor_near.any() and bool((m.landing == 1).all())
    fb.interoception([0.45, 0.6])
    for _ in range(2010):
        out = m.step(10, inputs)
    assert not out["power"][0].any() and out["power"][1].any()
    # Restoring sufficient reserves still requires the full grounded interval.
    fb.interoception([0.6, 0.6], feeding=[False, True])
    assert not m.step(10, inputs)["power"].any()
    for _ in range(2010):
        out = m.step(10, inputs)
    assert out["power"][0].any() and not out["power"][1].any()
    m.load_state_dict(saved)
    for key, value in saved.items():
        torch.testing.assert_close(value, m.state_dict()[key], rtol=0, atol=0)
    m.reset_rows([0])
    for key, value in saved.items():
        torch.testing.assert_close(value[1], m.state_dict()[key][1], rtol=0, atol=0)
    assert not m.observed[0].any() and not m.landing[0].any()


def test_flight_odor_is_optional_and_independent_of_plume_attachment_order():
    # A reduced graph without LH cells keeps reserve and time limits; no fabricated odor.
    from flyverse.navigation import FlightDrive

    c = graph()
    keep = ~c.neurons.type.str.startswith("LH")
    m = FlightDrive(c.subset(np.flatnonzero(keep.to_numpy())))
    m.reset(1, "cpu")
    assert m.parameters["odor_channels"] == {}
    m.observe_internal(0.1, False, True, False)
    inputs = {k: torch.zeros((1, len(ix))) for k, ix in m.reads.items()}
    assert not m.step(10, inputs)["power"].any()

    names = ["compass", "plume", "hunger", "flight"]
    a, b = make(names), make(names[::-1])
    for fb in (a, b):
        fb.interoception(0.8, airborne=True)
        flight = fb.instruments["flight"]
        for k in flight.parameters["odor_channels"]:
            fb.brain.rate[:, flight.reads["odor_" + k]] = 40
        # Place the filter just below its switch, then cross it through normal scheduling.
        flight.odor.fill_(0.599)
        fb.step(10)
        assert bool((flight.landing == 1).all())
    torch.testing.assert_close(a.brain.rate, b.brain.rate, rtol=0, atol=0)


def test_batch_environment_feeds_internal_state_and_resets_rows():
    from flyverse.batch_sim import BatchSim

    sim = BatchSim(
        c=graph(),
        batch=2,
        device="cpu",
        preset="instrumented",
        instruments=["flight", "hunger"],
    )
    sim.metabolisms[0].energy = 0.1
    sim.metabolisms[1].energy = 0.8
    sim.step()
    torch.testing.assert_close(
        sim.fb.instruments["hunger"].level, torch.tensor([[0.9], [0.2]])
    )
    sim.reset([0])
    sim.step()
    assert sim.fb.instruments["hunger"].level[0].item() == pytest.approx(0.4, abs=1e-6)
