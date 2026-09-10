"""Loom probe: a black sphere rushes at the fly from one side. Does the loom pathway (LPLC2 / LC4 ->
giant fibre DNp01 / DNp11 / DNp02...) fire, and does the giant fibre trigger the escape jump?

    python scripts/probe_loom.py [--side left|right|front] [--speed 1.0]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import body, brain, connectome, optic, retina, world  # noqa: E402

PROBE_OL = ["T4a", "T5a", "LPi34"]
PROBE = ["LPLC2", "LC4", "LPLC1", "LC6", "LC16", "LC11", "DNp01", "DNp11", "DNp02", "DNp04", "DNp06", "TTMn", "DLMn c-f", "DNa02", "MDN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left")
    ap.add_argument("--speed", type=float, default=1.0, help="m/s approach speed")
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--norm-ref", type=float, default=5000.0)
    ap.add_argument("--norm-alpha", type=float, default=1.0)
    ap.add_argument("--out-norm", default="l1")
    ap.add_argument("--gain-out", type=float, default=80.0)
    args = ap.parse_args()
    c = connectome.load(verbose=False)
    r = retina.build_retina(c)
    types = c.neurons.type.fillna("").to_numpy()
    w, info = world.make_room()
    w.spheres.append(world.Sphere((9, 9, 9), (args.radius,) * 3, "black"))
    loom_idx = len(w.spheres) - 1
    dirs_b, wts = r.ray_directions()
    wts_t = torch.from_numpy(wts).float().to(w.device)
    ol = optic.OpticLobe(c, r, optic.OpticParams(gain_out_mv=args.gain_out, out_norm=args.out_norm)); ol.relax()
    rt = types[ol.rate_idx]
    b = brain.Brain(c, brain.LIFParams(input_norm_alpha=args.norm_alpha, input_norm_ref=args.norm_ref)); b.freeze(ol.rate_idx)
    wg = body.wing_groups(c); flight = body.Flight()
    print("wing groups:", {k: len(v) for k, v in vars(wg).items()})
    fly = body.FlyState(x=-0.3, y=0.0, z=info["table_top_z"], heading=0.0)

    def col_radiance():
        d = fly.body_to_world(dirs_b.reshape(-1, 3)); o = np.broadcast_to(fly.eye_pos, d.shape)
        rad = w.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        return (rad.reshape(dirs_b.shape[0], dirs_b.shape[1], 4) * wts_t[None, :, None]).sum(1)

    def rep(lab):
        rts = b.rate.cpu().numpy(); dr = ol.last["dr"].cpu().numpy()
        print(f"[{lab}] spikes/step {b.total_spikes():.0f} | OL " + " ".join(f"{t}={np.abs(dr[rt == t]).mean():.2f}" for t in PROBE_OL)
              + " | " + " ".join(f"{t}={rts[c.select(type=t)].mean():.0f}" for t in PROBE))

    # adapt to the scene, then loom
    for k in range(150):   # walk for 1.5 s first: the GF must NOT fire from self-motion optic flow
        fly.x += 0.004 * 0.01 * np.cos(fly.heading); fly.y += 0.004 * 0.01 * np.sin(fly.heading)
        b.drive = ol.step_frame(col_radiance(), b.rate, 10.0); b.step(20)
        if k % 50 == 49:
            rep(f"walking {k // 100 + 1}")
    start = {"left": np.array([0.0, 0.5, 0.0]), "right": np.array([0.0, -0.5, 0.0]), "front": np.array([0.5, 0.0, 0.0])}[args.side]
    eye = fly.eye_pos + np.array([0, 0, 0.01])
    dist0 = np.linalg.norm(start)
    t = 0.0
    jumped = False
    while t < dist0 / args.speed + 0.3:
        d = max(dist0 - args.speed * t, args.radius + 0.005)
        w.move_sphere(loom_idx, eye + start / dist0 * d)
        b.drive = ol.step_frame(col_radiance(), b.rate, 10.0); b.step(20)
        t += 0.01
        wv = flight.readout(b, wg)
        if not jumped and (wv["gf"] >= flight.gf_hz or wv["ttm"] >= flight.gf_hz):
            jumped = True
            print(f"  *** escape triggered at t={t:.2f}s, object {d * 100:.1f} cm away (GF {wv['gf']:.0f} Hz, TTMn {wv['ttm']:.0f} Hz)")
        if int(t * 100) % 10 == 0:
            rep(f"t={t:.1f}s d={d * 100:.0f}cm")
    print("escape:", jumped)


if __name__ == "__main__":
    main()
