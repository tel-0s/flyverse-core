"""Where does the optic lobe lose a small object?  Retinotopic figure-ground measurement.

The fly is pinned on the single-apple table with the apple 5 cm ahead-left, heading oscillating +-20 deg
at 0.5 Hz (self-motion), and the same again with no fruit in the scene. For every optic-lobe cell type
the per-cell time-averaged SIGNED delta-rate (deviation from the operating point) is compared between the
columns that view the apple and the columns that do not, in both scenes. A type that carries the
object shows a larger (apple - none) difference in the apple columns than elsewhere. Spiking targets
(LC10a, LC4, LPLC2) are compared by the drive they receive.

    python scripts/probe_figure_ground.py [--seconds 15] [--radius-deg 30]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

pygame.init(); pygame.display.set_mode((64, 64))
sys.path.insert(0, os.path.dirname(__file__))
import room_demo as rd  # noqa: E402
from flyverse import brain  # noqa: E402

APPLE = np.array([0.25, 0.15, 0.79])


def patch_receptor(model, net_rule):
    """Make every brain.LIFParams built from here on (room_demo.Sim's included) carry the receptor model
    (LIFParams.receptor_model; docs/NT_INTEGRATION.md). 'off' leaves the class untouched."""
    if model in (None, "off"):
        return
    L = brain.LIFParams

    def make(**kw):
        p = L(**kw); p.receptor_model = model; p.receptor_net_rule = net_rule
        return p
    brain.LIFParams = make


def print_coverage(sim, model, net_rule):
    if sim.fb.receptor is None:
        return
    cov = sim.fb.receptor.coverage(sim.c.W)
    print(f"receptor model {model} ({net_rule}); fast sign changed on {int((sim.fb.receptor.fast_sign != np.sign(sim.c.W.data)).sum()):,} "
          f"of {sim.c.W.nnz:,} entries; coverage by tier:")
    print(cov.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
TYPES = ["L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "T2", "T3", "TmY3", "Tm5Y", "TmY21", "Tm20", "TmY17", "Tm34",
         "T4a", "T5a", "LPi34", "LPi43", "Dm8", "Dm9", "Pm2", "Li14"]
SPIKING = ["LC10a", "LC10b", "LC4", "LPLC2", "LC16"]


def run(fruit, seconds, pos, heading0, seed=0):
    sim = rd.Sim(seed, start=(float(pos[0]), float(pos[1]), 0.75), trail_seconds=0.0, fruit_set=fruit, fence=True, wind_speed=0.0,
                 cuda_kernels=True, cuda_graphs=False, event_driven=True, cuda_sparse="warp")
    if fruit == "all":
        for i, s in enumerate(sim.world.spheres):
            if s.material in ("apple", "orange", "banana", "lime", "grape", "blueberry"):
                sim.world.move_sphere(i, (9, 9, 9))
    o = sim.fb.optic; n_rate = len(o.rate_idx)
    acc = np.zeros(n_rate); drive = np.zeros(sim.c.n); n = 0
    # column viewing directions in the body frame, per rate cell: through the retina's hex columns
    for k in range(int(seconds * 100)):
        h = heading0 + np.deg2rad(20) * np.sin(2 * np.pi * 0.5 * k / 100)
        sim.fly.place(float(pos[0]), float(pos[1]), 0.75, heading=h); sim.step()
        if k >= 300:
            n += 1
            acc += (o.rates()[0] - o.r0.reshape(-1)).detach().cpu().numpy().ravel(); drive += sim.fb.brain.drive[0].detach().cpu().numpy().ravel()   # SIGNED mean deviation
    return sim, acc / n, drive / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--radius-deg", type=float, default=30.0)
    ap.add_argument("--out", default="out/figure_ground_signed.csv")
    ap.add_argument("--receptor-model", default="off", choices=["off", "sign", "sign+gain", "full"],
                    help="LIFParams.receptor_model (default off = the presynaptic NT_SIGN rule)")
    ap.add_argument("--receptor-net-rule", default="class", choices=["class", "abs", "nonmda"])
    ap.add_argument("--seed", type=int, default=0, help="room_demo.Sim seed (Brain RNG); replicates otherwise sample the native backend's nondeterminism")
    args = ap.parse_args()
    patch_receptor(args.receptor_model, args.receptor_net_rule)
    print(f"seed {args.seed}")
    r = 0.04 + 0.05
    pos = APPLE[:2] - r * np.array([np.cos(np.deg2rad(135)), np.sin(np.deg2rad(135))])
    heading0 = np.pi / 2
    sim_a, a_apple, d_apple = run("apple", args.seconds, pos, heading0, args.seed)
    print_coverage(sim_a, args.receptor_model, args.receptor_net_rule)
    sim_n, a_none, d_none = run("all", args.seconds, pos, heading0, args.seed)
    c = sim_a.c; o = sim_a.fb.optic
    # which columns view the apple: the column direction (body frame at the mean heading) vs the apple's direction
    eye = np.array([pos[0], pos[1], 0.75 + 0.0012]); to_apple = APPLE - eye; to_apple /= np.linalg.norm(to_apple)
    fly = sim_a.fly; fly.place(float(pos[0]), float(pos[1]), 0.75, heading=heading0)
    col_dir_world = fly.body_to_world(np.asarray(sim_a.fb.retina.col_dir))             # (n_col, 3)
    ang = np.degrees(np.arccos(np.clip(col_dir_world @ to_apple, -1, 1)))               # per column
    in_obj = ang <= args.radius_deg; far = ang >= args.radius_deg + 20
    print(f"columns viewing the apple (within {args.radius_deg:.0f} deg): {int(in_obj.sum())} of {len(ang)}; background columns (> {args.radius_deg + 20:.0f} deg): {int(far.sum())}")
    # map rate cells to columns: the retina/optic model assigns each ol cell a column index
    ret = sim_a.fb.retina
    key = {(str(sd), int(h[0]), int(h[1])): i for i, (sd, h) in enumerate(zip(ret.col_side, ret.col_hex))}
    neurons = c.neurons.iloc[np.asarray(o.rate_idx)]
    h1 = neurons["hex1"].to_numpy(); h2 = neurons["hex2"].to_numpy(); sd_ = neurons["hex_side"].fillna("").to_numpy()
    col_of = np.array([key.get((str(s_), int(a_), int(b_)), -1) if (a_ == a_ and b_ == b_) else -1 for s_, a_, b_ in zip(sd_, h1, h2)])
    print(f"rate cells with an annotated retina column: {int((col_of >= 0).sum())} of {len(col_of)}")
    # propagate: an unassigned cell takes the column of its strongest input partner that has one (3 passes)
    import scipy.sparse as sp
    ridx = np.asarray(o.rate_idx); Wabs = sp.csr_matrix(abs(c.W)); Wr = Wabs[ridx][:, ridx].tocsr()
    for _ in range(3):
        todo = np.flatnonzero(col_of < 0)
        for i in todo:
            s0, s1 = Wr.indptr[i], Wr.indptr[i + 1]; pre = Wr.indices[s0:s1]; w = Wr.data[s0:s1]
            ok = col_of[pre] >= 0
            if ok.any():
                col_of[i] = col_of[pre[ok][np.argmax(w[ok])]]
        print(f"  after a propagation pass: {int((col_of >= 0).sum())} of {len(col_of)}")
    types = c.neurons.type.fillna("").to_numpy()[np.asarray(o.rate_idx)]
    rows = []
    all_types = sorted(set(types) - {""})
    for ty in all_types:                                                  # every optic-lobe type, not a hand list
        m = (types == ty) & (col_of >= 0)
        if not m.any():
            continue
        obj = m & in_obj[np.clip(col_of, 0, len(in_obj) - 1)]; bg = m & far[np.clip(col_of, 0, len(far) - 1)]
        if obj.sum() < 3 or bg.sum() < 3:
            continue
        d_obj = (a_apple[obj] - a_none[obj]).mean(); d_bg = (a_apple[bg] - a_none[bg]).mean()
        sd = np.std(np.r_[a_apple[bg] - a_none[bg]]) + 1e-6
        rows.append({"type": ty, "cells_obj": int(obj.sum()), "cells_bg": int(bg.sum()), "resp_obj_apple": a_apple[obj].mean(), "resp_obj_none": a_none[obj].mean(),
                     "figure": d_obj - d_bg, "figure_z": (d_obj - d_bg) / sd, "bg_level": a_none[bg].mean()})
    df = pd.DataFrame(rows).sort_values("figure_z", ascending=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); df.to_csv(args.out, index=False)
    print("\nfigure signal per type: (apple - none) in the apple's columns minus the same in background columns, in rate units; z vs background scatter")
    print(df.head(25).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("...  bottom 10 (the dark object lowers these):")
    print(df.tail(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"{len(df)} types scored; full table in {args.out}")
    t_all = c.neurons.type.fillna("").to_numpy()
    print("\nspiking targets, mean |drive| mV apple vs none: " + "  ".join(f"{ty} {d_apple[t_all == ty].mean():.2f}/{d_none[t_all == ty].mean():.2f}" for ty in SPIKING))
    pos_r = {int(i): k for k, i in enumerate(ridx)}
    print("per-cell view: (apple - none) |drive| per cell; top 5 cells with the angle of the column they mostly view to the apple:")
    for ty in SPIKING:
        cells = np.flatnonzero(t_all == ty); diff = d_apple[cells] - d_none[cells]; order = np.argsort(-np.abs(diff))[:5]
        desc = []
        for j in order:
            s0, s1 = Wabs.indptr[cells[j]], Wabs.indptr[cells[j] + 1]; pre = Wabs.indices[s0:s1]; w = Wabs.data[s0:s1]
            m = np.array([int(pp) in pos_r and col_of[pos_r[int(pp)]] >= 0 for pp in pre])
            colj = col_of[pos_r[int(pre[m][np.argmax(w[m])])]] if m.any() else -1
            desc.append(f"{diff[j]:+.2f} mV @ {ang[colj]:.0f} deg" if colj >= 0 else f"{diff[j]:+.2f} mV @ ?")
        print(f"  {ty:6s}: median {np.median(diff):+.3f}, 90% {np.percentile(diff, 90):+.2f}, max {diff.max():+.2f} mV; top cells: " + "; ".join(desc))


if __name__ == "__main__":
    main()
