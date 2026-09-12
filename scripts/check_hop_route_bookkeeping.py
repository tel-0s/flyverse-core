"""CPU check that flyverse.batch_body attributes every take-off to its route without changing the dynamics.

Drives a BatchBody the way tests/test_batch_sim.py does (random motor samples, forced GF burst on one row,
forced wing power on another, both fence settings) next to a copy stepped with the scalar reference methods,
and asserts (1) the flies' states stay equal to the reference, (2) hops_escape + hops_voluntary equals the
number of false->true airborne transitions on every row, (3) the forced rows launched by the expected route.

    python scripts/check_hop_route_bookkeeping.py
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import body, surfaces, world  # noqa: E402
from flyverse.batch_body import BatchBody  # noqa: E402
from flyverse.motor import MotorRates  # noqa: E402


def motor_sample(batch, rng):
    values = {f.name: rng.uniform(0, 20, batch).astype(np.float32) for f in fields(MotorRates)
              if f.name not in ("lh_odour", "pn_glomeruli", "pn_glom_cells", "time_ms")}
    values.update(lh_odour={"apple": rng.uniform(0, 30, batch)}, pn_glomeruli={"DM1": rng.uniform(0, 40, batch)},
                  pn_glom_cells={"DM1": 5})
    return MotorRates(**values)


def scalar_body_step(model, commands, wings, tasting, dt):
    for i, (f, l, flight, m) in enumerate(zip(model.flies, model.locos, model.flights, model.metabolisms)):
        if f.airborne:
            model.feeding[i] = False; m.update(False, 0., dt)
            if model.fence is None:
                flight.step(f, wings[i], dt, model.surfaces.scalar, (-2, 2, -2, 2, 2.6))
            else:
                x0, x1, y0, y1, z = model.fence
                flight.step(f, wings[i], dt, lambda x, y: z, (x0, x1, y0, y1, 2.6))
        elif not flight.maybe_takeoff(f, wings[i], dt):
            model.feeding[i] = m.update(bool(tasting[i]), f.speed, dt)
            cmd = dict(commands[i], speed=0., yaw=0.) if model.feeding[i] else commands[i]
            l.step(f, cmd, dt, model.surfaces.scalar if model.fence is None else model.fence[:4])


def equal(a, b, tol=2e-12):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(a, (float, np.floating)):
        return np.allclose(a, b, rtol=tol, atol=tol)
    return a == b


def main():
    _, info = world.make_room()
    geo = surfaces.Surfaces(info["room"], info["solids"], info["solid_labels"])
    for fence in (None, (*info["table_extent"], .75)):
        B = 9
        flies = [body.FlyState(x=0, y=0, z=.75, heading=.15) for _ in range(B)]
        actual = BatchBody(flies, geo, fence)
        actual.flights[7]._power_hold = .29
        reference = BatchBody(copy.deepcopy(flies), geo, fence)
        reference.flights = copy.deepcopy(actual.flights)
        rng = np.random.default_rng(3)
        transitions = np.zeros(B, dtype=int); prev = np.array([f.airborne for f in actual.flies])
        for step in range(400):
            motor = motor_sample(B, rng)
            commands, wings = actual.readout(motor, .01)
            refw = [f.readout(motor.row(i)) for i, f in enumerate(reference.flights)]
            refcmd = [l.readout(motor.row(i)) for i, l in enumerate(reference.locos)]
            if step == 0:
                wings[6]["gf"] = refw[6]["gf"] = 80.          # GF burst above gf_hz 33 -> escape route
                wings[7]["power"] = refw[7]["power"] = 60.    # 0.29 s of hold + this frame -> voluntary route
            if step == 200:
                wings[6]["gf"] = refw[6]["gf"] = 80.          # a second escape after the landing refractory
            taste = np.zeros(B)
            actual.step(commands, wings, taste, .01)
            scalar_body_step(reference, refcmd, refw, taste, .01)
            cur = np.array([f.airborne for f in actual.flies]); transitions += cur & ~prev; prev = cur
            for a, b in zip(actual.flies, reference.flies):
                assert equal(asdict(a), asdict(b)), f"fly state diverged at step {step} (fence {fence is not None})"
        split = actual.hops_escape + actual.hops_voluntary
        print(f"fence={fence is not None}: airborne transitions {transitions.tolist()} escape {actual.hops_escape.tolist()} "
              f"voluntary {actual.hops_voluntary.tolist()}")
        assert np.array_equal(split, transitions), "hops_escape + hops_voluntary != airborne transitions"
        assert actual.hops_escape[6] >= 1 and actual.hops_voluntary[6] == 0, "row 6 should have launched by the GF route"
        assert actual.hops_voluntary[7] >= 1, "row 7 should have launched by the voluntary route"
    print("ok: route split equals the airborne-transition count on every row, dynamics unchanged vs the scalar reference")


if __name__ == "__main__":
    main()
