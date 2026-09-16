"""Frozen functional checks of the optional flight foraging policy; house GPU only.

This is an engineering regression assay, not a physiological or food-finding claim.
See docs/audits/flight_foraging_priority.md for the protocol frozen before submission.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from navigation_probe import NAV, lifecycle, write

from flyverse import BatchSim, connectome
from flyverse.brain import LIFParams
from flyverse.fly import FlyBrain
from flyverse.instruments import make_instrument
from flyverse.interp.common import provenance


def transitions(c):
    selected = set()
    for name in NAV:
        m = make_instrument(c, name)
        for idx in [*m.reads.values(), *m.writes.values()]:
            selected.update(idx)
    sub = c.subset(np.array(sorted(selected)))
    a, b = [
        FlyBrain(
            sub,
            device="cuda",
            batch=3,
            optic=None,
            seed=31,
            preset="instrumented",
            instruments=NAV,
            cuda_graphs=True,
            lif_params=LIFParams(receptor_model=None),
        )
        for _ in range(2)
    ]
    for m in b.instruments.values():
        m.cuda_graph_safe = False
        m.cuda_async_validation = False
        m.poisson_always_on = False
    checkpoints = {
        0,
        99,
        100,
        199,
        300,
        500,
        1000,
        1999,
        2000,
        2001,
        2040,
        2799,
        2800,
        2801,
        2802,
        2900,
        3099,
    }
    checks, sampled = [], []
    for frame in range(3100):
        for fb in (a, b):
            # Prescribed body observations, not simulated behavior. Rows: timed bout,
            # odor interruption, depleted reserves. LH rates clamped at frame boundaries.
            fb.interoception(
                [0.8, 0.8, 0.2],
                airborne=[2040 <= frame < 2900, frame < 1000, frame < 500],
            )
            m = fb.instruments["flight"]
            for key in m.parameters["odor_channels"]:
                idx = m.reads["odor_" + key]
                fb.brain.rate[:, idx] = 0
                if 100 <= frame < 300:
                    fb.brain.rate[1, idx] = 40
            fb.step(10)
        if frame in checkpoints:
            for key in FlyBrain.BRAIN_TENSORS:
                torch.testing.assert_close(
                    getattr(a.brain, key), getattr(b.brain, key), rtol=0, atol=0
                )
            for name in NAV:
                for key, value in a.instruments[name].state_dict().items():
                    torch.testing.assert_close(
                        value, b.instruments[name].state_dict()[key], rtol=0, atol=0
                    )
            checks.append(frame)
        m = a.instruments["flight"]
        sampled.append(torch.cat([m.active, m.landing, m.power_input], dim=1).clone())
    data = torch.stack(sampled).cpu().numpy()
    gates = {
        "captured_module_frame": any(isinstance(g, tuple) for g in a._graphs.values()),
        "ground_search_before_launch": not data[:1999, 0, 0].any(),
        "healthy_bout_starts": bool(data[2002, 0, 0]),
        "healthy_bout_times_out": bool(data[2802, 0, 1])
        and not data[2802:2900, 0, 2].any(),
        "odor_interrupts": not data[200:1000, 1, 2].any(),
        "depleted_never_powered": not data[:, 2, 2].any(),
    }
    return {
        "checks": checks,
        "gates": gates,
        "samples": data[::10],
        "columns": ["active", "landing", "power_input_hz"],
        "provenance": provenance(
            sub,
            fb=a,
            seeds=[31],
            batch=3,
            stimulus={
                "name": "prescribed body and neural observations",
                "duration_s": 31,
                "LH_rate_clamp_hz": [0, 40],
                "biological_result": False,
            },
        ),
    }


def room(c, depleted=False):
    sim = BatchSim(
        6,
        c=c,
        seed=0,
        seeds=range(6),
        device="cuda",
        cuda_graphs=True,
        cuda_kernels=True,
        event_driven=True,
        program="none",
        fruit_set="apple",
        fence=True,
        start=(-0.15, 0.15, 0.75),
        preset="instrumented",
        instruments=NAV,
    )
    if depleted:
        for f, metabolism in zip(sim.flies, sim.metabolisms):
            f.z = 1.0
            f.airborne = True
            f.air_time = 0.1
            f.vx = f.vy = f.vz = 0.0
            metabolism.energy = 0.1
    m = sim.fb.instruments["flight"]
    air, feeding, drive, max_bout, current_bout = [np.zeros(6) for _ in range(5)]
    first_land = np.full(6, np.nan)
    reserve_violations = np.zeros(6, dtype=int)
    samples, policy, energies = [], [], []
    for frame in range(6000):
        sim.step()
        states = (
            torch.cat(
                [m.active, m.landing, m.reserve_low, m.power_input, m.odor, m.energy],
                dim=1,
            )
            .cpu()
            .numpy()
        )
        flying = np.array([f.airborne for f in sim.flies])
        first_land[np.isnan(first_land) & ~flying] = (frame + 1) / 100
        air += flying * 0.01
        feeding += sim.feeding * 0.01
        powered = states[:, 0] > 0
        drive += powered * 0.01
        current_bout = np.where(powered, current_bout + 0.01, 0)
        max_bout = np.maximum(max_bout, current_bout)
        reserve_violations += (states[:, 5] <= 0.35) & (states[:, 3] > 0)
        if frame % 10 == 9:
            samples.append([[f.x, f.y, f.z] for f in sim.flies])
            policy.append(states[:, :5])
            energies.append([metabolism.energy for metabolism in sim.metabolisms])
    gates = {
        "zero_artificial_power_at_low_reserves": not bool(reserve_violations.any()),
        "bounded_power_bouts": bool((max_bout <= 8.02).all()),
    }
    if depleted:
        gates["initial_airborne_rows_land_within_5s"] = bool((first_land <= 5).all())
    return {
        "gates": gates,
        "airborne_s": air,
        "feeding_s": feeding,
        "powered_s": drive,
        "max_power_bout_s": max_bout,
        "first_grounded_s": first_land,
        "reserve_violations": reserve_violations,
        "body": np.array(samples),
        "policy": np.array(policy),
        "energy": np.array(energies),
        "policy_columns": [
            "active",
            "landing",
            "reserve_low",
            "power_input_hz",
            "odor",
        ],
        "hops_escape": sim.hops_escape.copy(),
        "hops_voluntary": sim.hops_voluntary.copy(),
        "provenance": provenance(
            c,
            fb=sim.fb,
            seeds=[0],
            env_seeds=list(range(6)),
            batch=6,
            stimulus={
                "name": "flight priority room observation",
                "duration_s": 60,
                "initial_energy": 0.1 if depleted else 0.6,
                "initial_airborne": depleted,
                "initial_z_m": 1.0 if depleted else 0.75,
                "program": "none",
                "fruit_set": "apple",
                "fence": True,
                "not_adoption_rate_half": True,
            },
        ),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("lifecycle", "transitions", "room", "depleted"),
        required=True,
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "house GPU required"
    print("device cuda", torch.cuda.get_device_name(), "mode", args.mode, flush=True)
    c = connectome.load()
    result = (
        lifecycle(c)
        if args.mode == "lifecycle"
        else transitions(c)
        if args.mode == "transitions"
        else room(c, args.mode == "depleted")
    )
    write(args.out, result)
    print("gates", result.get("gates", "exact lifecycle checks passed"), flush=True)
    if not all(result.get("gates", {}).values()):
        raise SystemExit("functional gate failed; results saved")
