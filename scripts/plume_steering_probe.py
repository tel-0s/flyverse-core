"""Trace sensory, heading, goal and motor stages of the optional plume controller."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from navigation_probe import NAV, write

from flyverse import BatchSim, connectome
from flyverse.interp.common import provenance


def diagnostic(c, heading="compass"):
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
    return {
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heading", choices=("compass", "compass_ring"), default="compass")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "house GPU required"
    print("device cuda", torch.cuda.get_device_name(), args.heading, flush=True)
    write(args.out, diagnostic(connectome.load(), args.heading))
