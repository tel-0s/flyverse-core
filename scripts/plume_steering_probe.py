"""Trace sensory, heading, goal and motor stages of the optional plume controller."""

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
from flyverse.interp.common import provenance


def diagnostic(c, heading="compass", *, validation=False):
    names = [heading, *NAV[1:]]
    sim = BatchSim(
        6,
        c=c,
        seed=31,
        seeds=range(6),
        device="cuda",
        cuda_graphs=True,
        cuda_kernels=True,
        event_driven=True,
        program="none",
        fruit_set="all",
        preset="instrumented",
        instruments=names,
    )
    initial = [5, 90, -90, 185, 5, 5]
    for f, m, angle in zip(sim.flies, sim.metabolisms, initial):
        f.heading = np.deg2rad(angle)
        m.energy = 0.1
    plume = sim.fb.instruments["plume"]
    rows = []
    feeding = np.zeros(6)
    control = []
    for frame in range(6000):
        sim.step()
        feeding += sim.feeding * 0.01
        if frame % 10 == 9:
            neural = (
                torch.cat(
                    [
                        plume.odor,
                        plume.strength,
                        plume.heading,
                        plume.goal,
                        plume.turn,
                        plume.inside,
                        sim.brain.rate[:, plume.writes["pfl_L"]].mean(1, keepdim=True),
                        sim.brain.rate[:, plume.writes["pfl_R"]].mean(1, keepdim=True),
                    ],
                    1,
                )
                .cpu()
                .numpy()
            )
            left, right = [sum(d.values()) for d in sim.smell_values]
            rows.append(
                np.column_stack(
                    [
                        [f.x for f in sim.flies],
                        [f.y for f in sim.flies],
                        [f.heading for f in sim.flies],
                        [f.yaw_rate for f in sim.flies],
                        [m.energy for m in sim.metabolisms],
                        left,
                        right,
                        neural,
                        sim.motor.turn_L,
                        sim.motor.turn_R,
                        sim.nearest_fruit()[1],
                        sim.feeding,
                    ]
                )
            )
            if validation:
                control.append(
                    torch.cat(
                        [plume.target_difference, plume.steer_input, plume.bilateral], 1
                    )
                    .cpu()
                    .numpy()
                )
    columns = [
        "x",
        "y",
        "body_heading",
        "yaw",
        "energy",
        "antenna_L",
        "antenna_R",
        "odor",
        "heading_strength",
        "heading",
        "goal",
        "turn_demand",
        "inside",
        "pfl_L",
        "pfl_R",
        "dna_L",
        "dna_R",
        "fruit_distance",
        "feeding",
    ]
    data = np.array(rows)
    summary = []
    for i in range(6):
        row = {
            "env_seed": i,
            "initial_heading_deg": initial[i],
            "feeding_s": feeding[i],
            "heading_range_deg": np.rad2deg(np.ptp(np.unwrap(data[:, i, 2]))),
            "heading_valid_fraction": np.mean(data[:, i, 8] >= 0.6),
            "mean_abs_demand": np.mean(np.abs(data[:, i, 11])),
            "mean_abs_DNa02_difference_hz": np.mean(
                np.abs(data[:, i, 15] - data[:, i, 16])
            ),
            "min_fruit_distance_m": np.min(data[:, i, 17]),
        }
        summary.append(row)
        print(row, flush=True)
    result = {
        "columns": columns,
        "samples": data,
        "summary": summary,
        "provenance": provenance(
            c,
            fb=sim.fb,
            seeds=[31],
            env_seeds=list(range(6)),
            batch=6,
            stimulus={
                "name": "plume steering diagnostic",
                "seconds": 60,
                "initial_energy": 0.1,
                "initial_heading_deg": initial,
                "start": "room default",
                "fruit": "all",
                "fence": False,
            },
        ),
    }
    if validation:
        result["control_columns"] = [
            "target_DNa02_difference_hz",
            "PFL3_input_difference_hz",
            "bilateral_contrast",
        ]
        result["control"] = np.asarray(control)
        result["gates"] = {
            "at_least_four_rows_feed_for_one_second": bool((feeding >= 1).sum() >= 4)
        }
    return result


def boundary(c):
    fb = FlyBrain(
        c,
        batch=8,
        device="cuda",
        optic=None,
        seed=31,
        preset="instrumented",
        instruments=NAV,
        cuda_graphs=True,
        cuda_kernels=True,
        lif_params=LIFParams(event_driven=True),
    )
    m = fb.instruments["plume"]
    idx = np.unique(
        np.concatenate([ix for key, ix in m.reads.items() if key.startswith("odor_")])
    )
    fb.brain.set_poisson(idx, 40)
    fb.interoception(
        0.1,
        sated=[0, 0, 0, 0, 1, 0, 0, 0],
        feeding=[0, 0, 0, 1, 0, 0, 0, 0],
        airborne=[0, 0, 0, 0, 0, 1, 1, 0],
    )
    fb.smell(
        {"DM1": [1.02, 1, 1, 1.02, 1, 1.02, 1, 0]},
        {"DM1": [1, 1.02, 1, 1, 1.02, 1, 1.02, 0]},
    )
    fb.wind([0, 1, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 0, 0])
    sampled = []
    for frame in range(3000):
        fb.step(10)
        if frame >= 2000 and frame % 10 == 9:
            actual = fb.brain.rate[:, m.reads["dna_L"]].mean(
                1, keepdim=True
            ) - fb.brain.rate[:, m.reads["dna_R"]].mean(1, keepdim=True)
            sampled.append(
                torch.cat(
                    [m.target_difference, actual, m.steer_input, m.strength], 1
                ).clone()
            )
    values = torch.stack(sampled).mean(0).cpu().numpy()
    gates = {
        "bilateral_turn_reaches_DNa02": bool(values[0, 1] > 2 and values[1, 1] < -2),
        "tracking_mean_error_below_2hz": bool(
            (np.abs(values[[0, 1, 5, 6, 7], 0] - values[[0, 1, 5, 6, 7], 1]) < 2).all()
        ),
        "balanced_input_has_little_turn": bool(abs(values[2, 1]) < 2),
        "feeding_and_satiety_stop_input": bool((values[[3, 4], 2] == 0).all()),
        "airborne_and_no_smell_retain_wind": bool(
            values[5, 1] < -2 and values[6, 1] > 2 and values[7, 1] > 2
        ),
    }
    print("boundary means", values.tolist(), "gates", gates, flush=True)
    return {
        "columns": [
            "target_DNa02_difference_hz",
            "DNa02_difference_hz",
            "PFL3_input_difference_hz",
            "heading_strength",
        ],
        "rows": values,
        "gates": gates,
        "provenance": provenance(
            c,
            fb=fb,
            seeds=[31],
            batch=8,
            stimulus={
                "name": "bilateral and neural feedback controls",
                "duration_s": 30,
                "tail_s": 10,
                "LH_poisson_hz": 40,
                "rows": [
                    "left",
                    "right",
                    "balanced",
                    "feeding",
                    "sated",
                    "airborne_right",
                    "airborne_left",
                    "no_smell_left",
                ],
                "concentrations": "1.02 versus 1.0; body-relative wind opposes walking gradient",
            },
        ),
    }


def scalar():
    from room_demo import Sim

    sim = Sim(
        seed=0,
        preset="instrumented",
        instruments=NAV,
        cuda_graphs=True,
        cuda_kernels=True,
        event_driven=True,
        cuda_sparse="warp",
    )
    sim.metabolism.energy = 0.1
    samples, feeding = [], 0.0
    for frame in range(6000):
        sim.step()
        feeding += bool(sim.feeding) * 0.01
        if frame % 10 == 9:
            samples.append(
                [
                    *sim.fly.pos,
                    sim.fly.heading,
                    sim.fly.yaw_rate,
                    sim.metabolism.energy,
                    sim.nearest_fruit()[1],
                    bool(sim.feeding),
                ]
            )
    result = {
        "columns": [
            "x",
            "y",
            "z",
            "heading",
            "yaw",
            "energy",
            "fruit_distance",
            "feeding",
        ],
        "samples": samples,
        "feeding_s": feeding,
        "gates": {"scalar_demo_feeds_for_one_second": feeding >= 1},
        "provenance": provenance(
            sim.c,
            fb=sim.fb,
            seeds=[0],
            env_seeds=[0],
            batch=1,
            stimulus={
                "name": "room_demo.Sim scalar integration",
                "seconds": 60,
                "initial_energy": 0.1,
                "start": "default",
                "fruit": "all",
                "cuda_sparse": "warp",
                "rendering": False,
            },
        ),
    }
    print("scalar feeding", feeding, "end energy", sim.metabolism.energy, flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heading", choices=("compass", "compass_ring"), default="compass")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode",
        choices=("diagnostic", "room", "boundary", "lifecycle", "scalar"),
        default="diagnostic",
    )
    args = ap.parse_args()
    assert torch.cuda.is_available(), "house GPU required"
    print("device cuda", torch.cuda.get_device_name(), args.heading, flush=True)
    c = None if args.mode == "scalar" else connectome.load()
    if args.mode == "scalar":
        result = scalar()
    elif args.mode == "lifecycle":
        result = lifecycle(c, smell=True)
    elif args.mode == "boundary":
        result = boundary(c)
    else:
        result = diagnostic(c, args.heading, validation=args.mode == "room")
    write(args.out, result)
    print("gates", result.get("gates", "diagnostic/lifecycle"), flush=True)
    # Save failures and continue the frozen family so all arms remain available for diagnosis.
