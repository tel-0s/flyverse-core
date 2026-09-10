"""Colour probe: sweep a sphere of each fruit material past the fly (same size, same path) and compare
the responses of the colour pathway (R7/R8 targets Dm8, Tm5a/b/c, Tm20, MeTu) and downstream visual
projection neurons. If the responses differ by material beyond what luminance explains, the fly sees
colour (UV/blue/green opponency from the connectome wiring), not just brightness.

    python scripts/probe_colour.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import body, brain, connectome, optic, retina, world  # noqa: E402

OL_TYPES = ["Dm8a", "Dm8b", "Tm5a", "Tm5b", "Tm5c", "Tm20", "Mi15", "Dm9", "Tm1", "Mi1"]
SPK_TYPES = ["MeTu1", "MeTu2a", "MeTu3a", "MeTu3b", "MeTu3c", "MeTu4a", "LC10a", "LC15", "LC11", "aMe12", "DNa02", "MDN"]
MATERIALS = ["apple", "banana", "orange", "lime", "grape", "blueberry", "plate", "black"]


def main():
    c = connectome.load(verbose=False)
    r = retina.build_retina(c)
    types = c.neurons.type.fillna("").to_numpy()
    w, info = world.make_room()
    w.spheres.append(world.Sphere((9, 9, 9), (0.02, 0.02, 0.02), "apple"))
    idx = len(w.spheres) - 1
    dirs_b, wts = r.ray_directions()
    wts_t = torch.from_numpy(wts).float().to(w.device)
    ol = optic.OpticLobe(c, r); ol.relax()
    rt = types[ol.rate_idx]
    fly = body.FlyState(x=-0.3, y=0.0, z=info["table_top_z"], heading=0.0)

    def col_radiance():
        d = fly.body_to_world(dirs_b.reshape(-1, 3)); o = np.broadcast_to(fly.eye_pos, d.shape)
        rad = w.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        return (rad.reshape(dirs_b.shape[0], dirs_b.shape[1], 4) * wts_t[None, :, None]).sum(1)

    rows = []
    for mat in MATERIALS:
        w.spheres[idx].material = mat
        w.move_sphere(idx, (9, 9, 9))
        ol.reset(); b = brain.Brain(c); b.freeze(ol.rate_idx)
        for k in range(40):                                   # adapt to the empty scene
            b.drive = ol.step_frame(col_radiance(), b.rate, 10.0); b.step(20)
        acc_ol = {t: [] for t in OL_TYPES}; acc_spk = {t: [] for t in SPK_TYPES}; lum = []
        for k in range(120):                                  # sphere sweeps left->right 8 cm in front, 1.2 s
            y = 0.12 - 0.24 * k / 120
            w.move_sphere(idx, (fly.x + 0.08, fly.y + y, fly.z + 0.02))
            rad = col_radiance()
            b.drive = ol.step_frame(rad, b.rate, 10.0); b.step(20)
            dr = ol.last["dr"][0].cpu().numpy(); rts = b.rate[0].cpu().numpy()
            for t in OL_TYPES:
                acc_ol[t].append(np.abs(dr[rt == t]).mean())
            for t in SPK_TYPES:
                acc_spk[t].append(rts[c.select(type=t)].mean())
            lum.append(float(rad[:, 1:].mean()))
        row = {"material": mat, "UV": world.MATERIALS[mat].refl[0], "B": world.MATERIALS[mat].refl[1], "G": world.MATERIALS[mat].refl[2],
               "lum": np.mean(lum)}
        row.update({t: np.mean(v) for t, v in acc_ol.items()})
        row.update({t: np.mean(v) for t, v in acc_spk.items()})
        rows.append(row)
        print(f"{mat:10s} " + " ".join(f"{t}={np.mean(v):.3f}" for t, v in acc_ol.items()) + " | " + " ".join(f"{t}={np.mean(v):.1f}" for t, v in acc_spk.items()))
    df = pd.DataFrame(rows).set_index("material")
    os.makedirs("out", exist_ok=True)
    df.to_csv("out/colour_probe.csv")
    # does anything track UV / blue / green beyond luminance?
    print("\ncorrelation of each response with material reflectance across materials (UV, B, G, lum):")
    for t in OL_TYPES + SPK_TYPES:
        v = df[t].to_numpy()
        if v.std() < 1e-9:
            continue
        cc = [np.corrcoef(v, df[k].to_numpy())[0, 1] for k in ["UV", "B", "G", "lum"]]
        print(f"  {t:8s} UV {cc[0]:+.2f}  B {cc[1]:+.2f}  G {cc[2]:+.2f}  lum {cc[3]:+.2f}")


if __name__ == "__main__":
    main()
