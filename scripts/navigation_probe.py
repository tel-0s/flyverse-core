"""Frozen navigation engineering assays. GPU execution belongs on the house cluster.

No fit, native-circuit claim, or preset adoption. See docs/audits/navigation_instruments.md.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from flyverse import BatchSim, connectome
from flyverse.brain import LIFParams
from flyverse.fly import FlyBrain
from flyverse.instruments import make_instrument
from flyverse.interp.common import provenance, to_jsonable
from flyverse.navigation import RecurrentCompass

NAV = ["compass", "plume", "hunger", "flight"]


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2) + "\n", encoding="utf-8")


def lifecycle(c):
    """Biological subgraph; tests reads, changing sensory inputs, pulses and graph storage lifetime."""
    selected = set()
    for name in NAV:
        m = make_instrument(c, name)
        for idx in [*m.reads.values(), *m.writes.values()]:
            selected.update(idx)
    sub = c.subset(np.array(sorted(selected)))
    records = []
    for heading in ("compass", "compass_ring"):
        names = [heading, *NAV[1:]]
        a, b = [
            FlyBrain(
                sub,
                batch=3,
                device="cuda",
                optic=None,
                preset="instrumented",
                instruments=names,
                seed=31,
                lif_params=LIFParams(receptor_model=None),
                cuda_graphs=True,
            )
            for _ in range(2)
        ]
        for m in b.instruments.values():
            m.cuda_graph_safe = False
            m.cuda_async_validation = False
            m.poisson_always_on = False
        checks = []

        def equal(label, a=a, b=b, names=names, checks=checks):
            for key in FlyBrain.BRAIN_TENSORS:
                torch.testing.assert_close(
                    getattr(a.brain, key),
                    getattr(b.brain, key),
                    rtol=0,
                    atol=0,
                    msg=label + ":" + key,
                )
            for name in names:
                for key, value in a.instruments[name].state_dict().items():
                    torch.testing.assert_close(
                        value,
                        b.instruments[name].state_dict()[key],
                        rtol=0,
                        atol=0,
                        msg=label + ":" + name + ":" + key,
                    )
            ai, bi = a.module_inputs(), b.module_inputs()
            assert ai.keys() == bi.keys()
            for key in ai:
                torch.testing.assert_close(ai[key], bi[key], rtol=0, atol=0)
            checks.append(label)

        for frame in range(80):
            for fb in (a, b):
                fb.proprioception(
                    0,
                    0,
                    0,
                    False,
                    yaw_rate=np.array([1.0, -2.0, 0.5]) * (1 if frame < 40 else -1),
                )
                fb.interoception(
                    [0.2, 0.8, 0.5], airborne=[True, False, True], feeding=frame >= 50
                )
                fb.instruments["plume"].observe_wind([0.6, -0.6, 0.2], [0.2, 0.4, -0.3])
                if frame == 20:
                    fb.stimulate(np.arange(8), 100.0, 3.0)
                if frame == 30:
                    fb.brain.set_drive([0], 0.25)
                if frame == 45:
                    fb.step(5.0)
                    fb.step(5.0)
                else:
                    fb.step(10.0)
                torch.cuda.synchronize()
            if frame == 0:
                assert any(isinstance(g, tuple) for g in a._graphs.values()), (
                    "no captured module frame"
                )
            if frame in (0, 20, 21, 30, 40, 45, 46, 50, 79):
                equal(f"frame {frame}")
        for fb in (a, b):
            fb.reset(rows=[1])
            fb.step(10.0)
        equal("partial reset")
        state = a.state_dict()
        a.step(50.0)
        expected = {k: getattr(a.brain, k).clone() for k in FlyBrain.BRAIN_TENSORS}
        a.load_state_dict(state)
        a.step(50.0)
        for k, v in expected.items():
            torch.testing.assert_close(getattr(a.brain, k), v, rtol=0, atol=0)
        b.step(50.0)
        equal("checkpoint replay")
        for fb in (a, b):
            for _ in range(20):
                fb.step(10.0)
            fb.reset()
            fb.step(10.0)
        equal("queued reset")
        for fb in (a, b):
            for _ in range(20):
                fb.step(10.0)
            for name in ("hunger", "flight", "plume", heading):
                fb.detach(name)
            fb.step(10.0)
        for k in FlyBrain.BRAIN_TENSORS:
            torch.testing.assert_close(
                getattr(a.brain, k), getattr(b.brain, k), rtol=0, atol=0
            )
        checks.append("queued detach")
        records.append({"heading": heading, "checks": checks})
        print("lifecycle exact", heading, len(checks), flush=True)
    return {
        "records": records,
        "provenance": provenance(
            sub,
            device="cuda",
            seeds=[31],
            batch=3,
            stimulus={
                "name": "captured vs checked multi-module scheduler",
                "instruments": [NAV, ["compass_ring", *NAV[1:]]],
            },
        ),
    }


def recurrent(c, device):
    records = []
    for eb in (0.0, 2.7):
        m = RecurrentCompass(c, eb_ratio=eb)
        m.reset(3, device)
        traces = []
        for frame in range(1000):
            if frame in (0, 400, 700):
                m.observe_turn(
                    [math.pi / 2, -math.pi / 2, 0.0] if frame == 0 else [0.0, 0.0, 0.0]
                )
            m.step(10, {})
            traces.append(m.e.clone())
        rates = torch.stack(traces).cpu().numpy()
        z = rates @ np.exp(1j * np.arange(16) * 2 * np.pi / 16)
        phase = np.unwrap(np.angle(z), axis=0)
        records.append(
            {
                "eb_ratio": eb,
                "velocity_deg_s": np.rad2deg((phase[399] - phase[100]) / 2.99),
                "stopped_drift_deg": np.rad2deg(phase[-1] - phase[699]),
                "final_strength": np.abs(z[-1]) / rates[-1].sum(-1),
                "instrument": m.describe(),
            }
        )
    return {
        "records": records,
        "provenance": provenance(
            c,
            device=device,
            stimulus={
                "name": "reduced ring control",
                "yaw_deg_s": [90, -90, 0],
                "moving_s": 4,
                "stationary_s": 6,
                "noise": 0,
                "not_velocity_calibrated": True,
            },
        ),
    }


def neural(c):
    fb = FlyBrain(
        c,
        device="cuda",
        batch=6,
        optic=None,
        preset="instrumented",
        instruments=NAV,
        cuda_graphs=True,
        cuda_kernels=True,
        cuda_sparse="torch",
        lif_params=LIFParams(event_driven=True),
        seed=31,
    )
    m = fb.instruments["plume"]
    rows = []
    idx = np.unique(
        np.concatenate([v for k, v in m.reads.items() if k.startswith("odor_")])
    )
    fb.brain.set_poisson(idx, 40.0)
    fb.interoception(
        [0.2, 0.2, 1.0, 0.2, 0.2, 0.2],
        sated=[0, 0, 1, 0, 0, 0],
        feeding=[0, 0, 0, 1, 0, 0],
        airborne=[0, 0, 0, 0, 1, 1],
    )
    m.observe_wind([1.0, 0.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    power = fb.instruments["flight"].reads["power"]
    for frame in range(3000):
        fb.step(10.0)
        if frame >= 2000 and frame % 10 == 9:
            rows.append(
                torch.cat(
                    [
                        fb.brain.rate[:, fb.brain._idx(power)].mean(1, keepdim=True),
                        m.turn,
                        m.strength,
                        fb.instruments["flight"].power_input,
                    ],
                    dim=1,
                ).clone()
            )
    mean = torch.stack(rows).mean(0).cpu().numpy()
    return {
        "columns": [
            "wing_power_hz",
            "plume_turn",
            "heading_strength",
            "power_input_hz",
        ],
        "rows": mean,
        "provenance": provenance(
            c,
            fb=fb,
            seeds=[31],
            batch=6,
            stimulus={
                "name": "neural boundary control",
                "duration_s": 30,
                "tail_s": 10,
                "odor_LH_poisson_hz": 40,
                "rows": [
                    "left",
                    "right",
                    "sated",
                    "feeding",
                    "airborne_left",
                    "airborne_right",
                ],
                "yaw_rate": 0,
                "body_commands": False,
                "optic": None,
            },
        ),
    }


def room(c, names, seconds=60):
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
        instruments=names,
    )
    sampled = []
    powers = []
    dist = np.zeros(6)
    feeding = np.zeros(6)
    airtime = np.zeros(6)
    minimum = np.full(6, np.inf)
    previous = np.array([f.pos for f in sim.flies])
    for frame in range(round(seconds * 100)):
        sim.step()
        current = np.array([f.pos for f in sim.flies])
        dist += np.linalg.norm(current - previous, axis=1)
        previous = current
        minimum = np.minimum(minimum, sim.nearest_fruit()[1])
        feeding += sim.feeding * 0.01
        airtime += np.array([f.airborne for f in sim.flies]) * 0.01
        if frame % 10 == 9:
            sampled.append([[f.x, f.y, f.z, f.heading, f.yaw_rate] for f in sim.flies])
            powers.append(np.asarray(sim.motor.power).copy())
    return {
        "body": np.asarray(sampled),
        "wing_power": np.asarray(powers),
        "distance_m": dist,
        "feeding_s": feeding,
        "airborne_s": airtime,
        "nearest_fruit_m": minimum,
        "seconds": seconds,
        "provenance": provenance(
            c,
            fb=sim.fb,
            seeds=[0],
            env_seeds=list(range(6)),
            batch=6,
            stimulus={
                "name": "navigation room observation",
                "program": "none",
                "start": [-0.15, 0.15, 0.75],
                "fruit_set": "apple",
                "fence": True,
                "not_adoption_rate_half": True,
            },
        ),
    }


def profile(c):
    records = []
    controllers = []
    identity = []
    for batch in (1, 8, 32):
        namesets = [
            ["compass"],
            ["compass", "plume", "hunger"],
            NAV,
            ["compass_ring", "plume", "hunger"],
        ]
        brains = []
        for names in namesets:
            fb = FlyBrain(
                c,
                device="cuda",
                batch=batch,
                optic=None,
                preset="instrumented",
                instruments=names,
                cuda_graphs=True,
                cuda_kernels=True,
                cuda_sparse="warp" if batch == 1 else "torch",
                lif_params=LIFParams(event_driven=True),
                seed=31,
            )
            # Matched high external forcing dominates all module outputs, equalizing network workload.
            forced = np.unique(
                np.concatenate(
                    [
                        ix
                        for name in NAV
                        for ix in make_instrument(c, name).writes.values()
                    ]
                )
            )
            fb.brain.set_poisson(forced, 250.0)
            if "interoception" in fb.available_senses:
                fb.interoception(0.5)
            for _ in range(50):
                fb.step(10.0)
            brains.append(fb)
            controllers.append(provenance(c, fb=fb, seeds=[31], batch=batch))
        for repeat in range(4):
            for i in range(4) if repeat % 2 == 0 else reversed(range(4)):
                fb = brains[i]
                torch.cuda.synchronize()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                t = time.perf_counter()
                begin.record()
                for _ in range(100):
                    fb.step(10.0)
                end.record()
                end.synchronize()
                records.append(
                    {
                        "batch": batch,
                        "instruments": namesets[i],
                        "repeat": repeat,
                        "cuda_ms": begin.elapsed_time(end) / 100,
                        "wall_ms": (time.perf_counter() - t) * 10,
                        "module_graphs": sum(
                            isinstance(g, tuple) for g in fb._graphs.values()
                        ),
                    }
                )
        reference = brains[0].brain
        for fb in brains[1:]:
            tensors = {
                key: torch.equal(getattr(reference, key), getattr(fb.brain, key))
                for key in FlyBrain.BRAIN_TENSORS
            }
            maximum = {
                key: float(
                    (getattr(reference, key).float() - getattr(fb.brain, key).float())
                    .abs()
                    .max()
                )
                if getattr(reference, key).numel()
                else 0.0
                for key in FlyBrain.BRAIN_TENSORS
            }
            identity.append(
                {
                    "batch": batch,
                    "instruments": list(fb.instruments),
                    "tensors": tensors,
                    "max_abs": maximum,
                }
            )
        del brains, fb
        torch.cuda.empty_cache()
    return {
        "records": records,
        "controllers": controllers,
        "workload_identity": all(all(row["tensors"].values()) for row in identity),
        "identity": identity,
        "protocol": "4 alternating repeats x 100 frames; 50 warmup; matched 250 Hz forcing; B=1,8,32; shared-machine observations",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["lifecycle", "recurrent", "neural", "room", "profile"],
        required=True,
    )
    ap.add_argument("--instruments", nargs="+", default=NAV)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    if a.device == "cpu" and a.mode != "recurrent":
        ap.error("only the reduced ring assay runs on CPU here")
    c = connectome.load(verbose=False)
    result = {
        "lifecycle": lambda: lifecycle(c),
        "recurrent": lambda: recurrent(c, a.device),
        "neural": lambda: neural(c),
        "room": lambda: room(c, a.instruments),
        "profile": lambda: profile(c),
    }[a.mode]()
    write(a.out, result)
    print("completed", a.mode, flush=True)
