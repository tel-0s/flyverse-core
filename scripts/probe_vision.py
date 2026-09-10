"""Diagnostic: run the hybrid brain through a fixed stimulus protocol in the room and print what
responds at each stage (optic-lobe delta-rates by type, spiking visual projection / descending / motor
neurons, motor readout). Also writes out/view_*.png test renders.

    python scripts/probe_vision.py [--norm l2] [--gain-in 3] [--gain-out 80]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import body, brain, connectome, optic, retina, world  # noqa: E402

OL_TYPES = ["L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9", "Mi4", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b",
            "Dm8a", "Tm5c", "Tm20", "LPi34"]
SPK_TYPES = ["LC4", "LPLC2", "LC10a", "LC11", "LC12", "LC15", "LC16", "LC17", "LC21", "LPLC1", "HSE", "VS", "MeTu3c",
             "DNa02", "DNp09", "MDN", "DNa01", "DNp01", "DNp07", "MN9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="l2")
    ap.add_argument("--gain-in", type=float, default=3.0)
    ap.add_argument("--gain-out", type=float, default=80.0)
    ap.add_argument("--gain-rr", type=float, default=1.0)
    args = ap.parse_args()

    c = connectome.load(verbose=False)
    r = retina.build_retina(c)
    nrn = c.neurons
    types = nrn.type.fillna("").to_numpy()
    w, info = world.make_room()
    dirs_b, wts = r.ray_directions()
    wts_t = torch.from_numpy(wts).float().to(w.device)

    img = w.render_camera(pos=(-0.55, 0.0, info["table_top_z"] + 0.0012), forward=(1, 0.0, 0.05), up=(0, 0, 1), width=480, height=300, fov_deg=110)
    Image.fromarray(world.to_rgb8(img, exposure=2.0)).save("out/view_fly_height.png")
    Image.fromarray(world.to_fly_false_color(img, exposure=2.0)).save("out/view_fly_height_uv.png")
    img2 = w.render_camera(pos=(-1.5, -1.3, 1.8), forward=(1.5, 1.3, -1.05), up=(0, 0, 1), width=480, height=300, fov_deg=60)
    Image.fromarray(world.to_rgb8(img2, exposure=2.0)).save("out/view_room.png")

    def col_radiance(fly):
        d = fly.body_to_world(dirs_b.reshape(-1, 3))
        o = np.broadcast_to(fly.eye_pos, d.shape)
        rad = w.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        rad = rad.reshape(dirs_b.shape[0], dirs_b.shape[1], 4)
        return (rad * wts_t[None, :, None]).sum(1)

    g = body.motor_groups(c)
    loco = body.Locomotion()
    ol = optic.OpticLobe(c, r, optic.OpticParams(norm=args.norm, gain_in=args.gain_in, gain_out_mv=args.gain_out, gain_rr=args.gain_rr))
    ol.relax()
    rt = types[ol.rate_idx]
    b = brain.Brain(c)
    b.freeze(ol.rate_idx)
    fly = body.FlyState(x=-0.45, y=0.0, z=info["table_top_z"], heading=0.0)

    def report(lab):
        rts = b.rate.cpu().numpy()
        n = nrn.assign(rate=rts)
        sc = n.groupby("superclass").rate.mean()
        dr = ol.last["dr"].cpu().numpy()
        ct = ol.last["contrast"][:, 0]
        print(f"[{lab}] spikes/step {b.total_spikes():.0f} frac>1Hz {(rts > 1).mean():.3f} contrast |c| mean {np.abs(ct).mean():.3f} | "
              + ", ".join(f"{k}={v:.1f}" for k, v in sc.sort_values(ascending=False).head(4).items()))
        print("    OL |dr|:", {t: round(float(np.abs(dr[rt == t]).mean()), 3) for t in OL_TYPES},
              " sat %.3f zero %.3f" % ((ol.rates() >= 1).float().mean().item(), (ol.rates() <= 0).float().mean().item()))
        print("    spiking:", {t: round(float(rts[c.select(type=t)].mean()), 1) for t in SPK_TYPES})
        d = b.drive.cpu().numpy()
        print("    drive>7mV: %d, <-7: %d, max %.1f" % ((d > 7).sum(), (d < -7).sum(), d.max()))
        g2 = n.groupby("type").rate.agg(["mean", "size"])
        print("    top spiking:", g2[g2["size"] >= 4].sort_values("mean", ascending=False).head(8)["mean"].round(0).to_dict())
        print("    motor:", {k: round(v, 1) for k, v in loco.readout(b, g)["rates"].items()})

    def frames(n, move):
        nonlocal rad
        for _ in range(n):
            move()
            rad = col_radiance(fly)
            b.drive = ol.step_frame(rad, b.rate, 10.0)
            b.step(20)

    rad = col_radiance(fly)
    frames(40, lambda: None); report("static 400ms")
    frames(80, lambda: setattr(fly, "heading", fly.heading + np.deg2rad(90) * 0.01)); report("rotating left 90deg/s")
    frames(80, lambda: setattr(fly, "heading", fly.heading - np.deg2rad(90) * 0.01)); report("rotating right 90deg/s")

    def fwd():
        fly.x += 0.02 * 0.01 * np.cos(fly.heading); fly.y += 0.02 * 0.01 * np.sin(fly.heading)
    frames(80, fwd); report("walking fwd 2cm/s")
    frames(80, lambda: None); report("static 800ms after")


if __name__ == "__main__":
    main()
