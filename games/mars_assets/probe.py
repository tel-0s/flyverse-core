"""Dev probe for games/mars.py (dev seeds >= 100 only): drive the fly's eye straight at a boulder and log what the
loom pathway, the giant fibre and MDN do, tick by tick. Not a game; measurements for the Design section.

    python games/mars_assets/probe.py --seed 100 --set A --out out/games/mars/probe_A_s100.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import games.common as gc  # noqa: E402
from games import mars as M  # noqa: E402

import torch  # noqa: E402

# condition: (name, speed m/s, eye height m, lateral offset of the rock m (+ = left), rock size m, flat terrain, dark)
SETS = {
    "A": [("base_v6_h2", 6.0, 2.2, None, 1.2, True, True),
          ("c_v3_h2", 3.0, 2.2, 0.0, 1.2, True, True),
          ("c_v6_h2", 6.0, 2.2, 0.0, 1.2, True, True),
          ("c_v10_h2", 10.0, 2.2, 0.0, 1.2, True, True),
          ("L_v6_h2", 6.0, 2.2, 1.6, 1.2, True, True),
          ("R_v6_h2", 6.0, 2.2, -1.6, 1.2, True, True)],
    "B": [("base_v6_h1", 6.0, 1.1, None, 1.2, True, True),
          ("c_v3_h1", 3.0, 1.1, 0.0, 1.2, True, True),
          ("c_v6_h1", 6.0, 1.1, 0.0, 1.2, True, True),
          ("c_v10_h1", 10.0, 1.1, 0.0, 1.2, True, True),
          ("L_v6_h1", 6.0, 1.1, 1.6, 1.2, True, True),
          ("R_v6_h1", 6.0, 1.1, -1.6, 1.2, True, True)],
    "C": [("big_v6_h2", 6.0, 2.2, 0.0, 2.2, True, True),
          ("light_v6_h2", 6.0, 2.2, 0.0, 1.2, True, False),
          ("terrain_v6_h2", 6.0, 2.2, None, 1.2, False, True),
          ("L_v10_h2", 10.0, 2.2, 1.6, 1.2, True, True),
          ("R_v10_h2", 10.0, 2.2, -1.6, 1.2, True, True),
          ("static_h2", 0.0, 2.2, None, 1.2, True, True)],
}
WARM_S, DRIVE_S, HOLD_S = 1.0, 3.0, 0.4


def groups(c):
    side = c.neurons.somaSide.fillna("").to_numpy()
    g = {}
    for t in ("LPLC2", "LC4", "DNp01", "MDN", "LPLC1", "LC6", "DNp02", "DNp11"):
        idx = c.select(type=t)
        for s in "LR":
            sel = idx[side[idx] == s]
            if len(sel):
                g[f"{t}_{s}"] = sel
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--set", default="A")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.seed >= 100, "dev seeds only"
    from flyverse import FlyBrain
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fb = FlyBrain(device=dev, seed=a.seed)
    eyes = gc.Eyes(fb)
    g = groups(fb.c)
    names = list(g)
    flat_idx = np.concatenate([g[k] for k in names])
    seg = np.cumsum([0] + [len(g[k]) for k in names])
    ridx = torch.as_tensor(flat_idx, device=fb.device)
    scenes = {}
    out = {"seed": a.seed, "set": a.set, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
           "groups": {k: int(len(v)) for k, v in g.items()}, "conditions": []}

    for (name, v, eh, off, size, flat, dark) in SETS[a.set]:
        if flat not in scenes:
            scenes[flat] = M.MarsScene(dev, flat=flat)
        sc = scenes[flat]
        d0 = max(3.0 * v, 6.0) + size + 2.0
        x_rock = d0
        if off is not None:
            rk = M.make_rock(np.random.default_rng(11), (x_rock, off, sc.height_np(x_rock, off)), size, dark=dark)
            sc.set_rocks([rk])
        else:
            rk = None
            sc.set_rocks([])
        fb.reset()
        wd = torch.as_tensor(eyes.world_dirs((1, 0, 0), (0, 1, 0), (0, 0, 1)), device=fb.device)
        n_ticks = int(round((WARM_S + DRIVE_S + HOLD_S) * 100))
        rows = []
        x = 0.0
        t_wall = time.time()
        stopped = False
        for k in range(n_ticks):
            t = k * 0.01
            if WARM_S <= t < WARM_S + DRIVE_S and not stopped:
                x += v * 0.01
            if rk is not None and off == 0.0:
                clear = (rk.pos[0] - rk.r_ground) - x          # eye to the rock's near face, horizontally
                if clear < 0.6:
                    stopped = True
            else:
                clear = None
            z = sc.height_np(x, 0.0) + eh
            o = torch.tensor([x, 0.0, z], dtype=torch.float32, device=fb.device)
            rgb, _ = sc.trace(o, wd)
            rad = eyes.pool(gc.rgb_to_radiance(rgb))
            fb.vision(rad)
            fb.step(gc.TICK_MS)
            r = fb.brain.rate[0, ridx].float().cpu().numpy()
            m = fb.motor()
            row = {"t": round(t + 0.01, 3), "x": round(x, 3), "clear": None if clear is None else round(clear, 3),
                   "gf": float(m.gf), "mdn": float(m.back_dn), "moving": bool(WARM_S <= t < WARM_S + DRIVE_S and not stopped)}
            if rk is not None:
                dx, dy = rk.pos[0] - x, rk.pos[1]
                row["ang_deg"] = round(2 * math.degrees(math.atan2(rk.r_ground, max(math.hypot(dx, dy) - 0.0, 1e-3))), 2)
                row["dist"] = round(math.hypot(dx, dy), 3)
            for i, kname in enumerate(names):
                row[kname] = round(float(r[seg[i]:seg[i + 1]].mean()), 3)
            rows.append(row)
        out["conditions"].append({"name": name, "v": v, "eye_h": eh, "offset": off, "size": size, "flat": flat,
                                  "dark": dark, "rock_x": x_rock, "wall_s": round(time.time() - t_wall, 1), "rows": rows})
        print(f"{name}: {time.time() - t_wall:.1f} s wall; peak GF {max(r_['gf'] for r_ in rows):.1f}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
