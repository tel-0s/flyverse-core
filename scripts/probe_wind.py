"""Wind / plume probe: does any descending neuron encode wind direction (anemotaxis), and is it gated
by odour? The fly stands still; the wind blows from the front, the left or the right (1 s each), with
and without an odour plume on it. For each condition the left-right asymmetry of every DN type is
recorded; types whose asymmetry flips between left and right wind are the wind-direction candidates.

    python scripts/probe_wind.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import air, body, brain, connectome, optic, retina, world  # noqa: E402


def main():
    c = connectome.load(verbose=False)
    n = c.neurons
    r = retina.build_retina(c)
    w, info = world.make_room()
    dirs_b, wts = r.ray_directions(); wts_t = torch.from_numpy(wts).float().to(w.device)
    ol = optic.OpticLobe(c, r); ol.relax()
    fly = body.FlyState(x=-0.3, y=0.0, z=info["table_top_z"], heading=0.0)

    def col_rad():
        d = fly.body_to_world(dirs_b.reshape(-1, 3)); o = np.broadcast_to(fly.eye_pos, d.shape)
        rad = w.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        return (rad.reshape(dirs_b.shape[0], dirs_b.shape[1], 4) * wts_t[None, :, None]).sum(1)

    side = n.somaSide.to_numpy(dtype=object)
    dn = c.select(superclass="descending_neuron"); dn_t = n.type.fillna("").to_numpy()[dn]; dn_s = side[dn]
    jo = c.select(type="~^JO-[CE]")
    results = {}
    for odour in (False, True):
        # a single apple source upwind of the fly when the wind blows from the front; the plume passes over it
        src = [("apple", (fly.x + 0.3, fly.y, fly.z + 0.01), 1.0)] if odour else []
        for name, towards in (("front", 180.0), ("left", -90.0), ("right", 90.0)):
            a = air.Air(src, air.WindParams(direction_deg=towards, meander_deg=0.0))
            olf = air.BilateralOlfaction(c, a); ws = air.WindSense(c, a)
            b = brain.Brain(c); b.freeze(ol.rate_idx); ol.reset()
            acc = np.zeros(len(dn)); nacc = 0
            for k in range(120):
                a.step(0.01); olf.apply(b, fly); ws.apply(b, fly)
                b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
                if k >= 40:
                    acc += b.rate_np()[dn]; nacc += 1
            acc /= nacc
            rt = b.rate_np()
            df = pd.DataFrame({"type": dn_t, "side": dn_s, "rate": acc}).groupby(["type", "side"]).rate.mean().unstack().fillna(0)
            results[(odour, name)] = df
            print(f"odour={odour!s:5s} wind from {name:5s}: JO-C/E mean {rt[jo].mean():.0f} Hz, dL/dR {ws.last['dL']:+.2f}/{ws.last['dR']:+.2f}, "
                  f"ORN mean {rt[olf.orn_idx].mean():.1f} Hz, frac>1Hz {(rt > 1).mean():.3f}, DN mean {acc.mean():.2f} Hz")
    for odour in (False, True):
        L, R, F = results[(odour, "left")], results[(odour, "right")], results[(odour, "front")]
        asym = pd.DataFrame({"wind_left": L.get("L", 0) - L.get("R", 0), "wind_right": R.get("L", 0) - R.get("R", 0), "front_mean": F.mean(axis=1)})
        asym["flip"] = asym.wind_left - asym.wind_right
        act = pd.concat([L, R, F], axis=1).max(axis=1)
        asym = asym[act > 3]
        print(f"\nodour={odour}: DN types whose L-R asymmetry flips with wind side (top 10):")
        print(asym.reindex(asym.flip.abs().sort_values(ascending=False).index).head(10).round(1).to_string())


if __name__ == "__main__":
    main()
