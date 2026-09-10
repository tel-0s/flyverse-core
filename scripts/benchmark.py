"""Calibration harness: score one LIF/optic parameter set on every behaviour we know the model should
show, in one run (~3 min on the 4090). Use it before changing defaults.

    python scripts/benchmark.py                        # current defaults
    python scripts/benchmark.py --std-u 0 --same-type-gain 0.1
    python scripts/benchmark.py --json out/bench_x.json

Tests
  rest      : no input for 500 ms                  -> spikes/step (want ~0)
  taste     : labellar sweet GRNs 100 Hz, 600 ms   -> MN9 Hz (want > 3), Usnea/GNG175 Hz, brain-wide frac
  smell     : ORNs at 3 Hz base + apple odour       -> PN Hz (want < 100), KC Hz (want > 0), LN Hz, frac
  dn_drive  : DNa02_L / DNp09 / MDN at 150 Hz       -> leg MN L, R (want asymmetry for DNa02_L, activity
              for DNp09/MDN, and NOT the same pattern for all), wing power, frac active, top clique rate
  walk      : fly walks 1.5 s in the room, smell on -> GF Hz (want ~0), wing power (want < 30), frac active
  loom      : black ball at 1 m/s from the left     -> GF Hz at 3.5 cm (want > 20), escape distance
  rotate    : 90 deg/s yaw for 0.8 s each way       -> DNa02 ipsi/contra Hz (want asymmetric)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import body, brain, connectome, olfaction, optic, retina, world  # noqa: E402


def top_types(c, rt, k=4):
    g = c.neurons.assign(rate=rt).groupby("type").rate.agg(["mean", "size"])
    return g[g["size"] >= 2].sort_values("mean", ascending=False).head(k)["mean"].round(0).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std-u", type=float, default=None)
    ap.add_argument("--std-tau", type=float, default=None)
    ap.add_argument("--adapt-jump", type=float, default=None)
    ap.add_argument("--same-type-gain", type=float, default=None)
    ap.add_argument("--norm-alpha", type=float, default=None)
    ap.add_argument("--norm-ref", type=float, default=None)
    ap.add_argument("--w-syn", type=float, default=None)
    ap.add_argument("--conn-cap", type=float, default=None)
    ap.add_argument("--dn-vnc-gain", type=float, default=None, help="gain on descending -> VNC synapses")
    ap.add_argument("--gain-out", type=float, default=None)
    ap.add_argument("--t4-gain", type=float, default=None, help="T4/T5 output gain (default 2)")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()
    lif = brain.LIFParams()
    for k, v in [("std_u", args.std_u), ("std_tau", args.std_tau), ("adapt_jump", args.adapt_jump), ("same_type_gain", args.same_type_gain),
                 ("input_norm_alpha", args.norm_alpha), ("input_norm_ref", args.norm_ref), ("w_syn", args.w_syn), ("conn_cap", args.conn_cap)]:
        if v is not None:
            setattr(lif, k, v)
    if args.dn_vnc_gain is not None:
        lif.path_gain = [(r"^descending_neuron$", r"^vnc_", args.dn_vnc_gain)]
    op = optic.OpticParams()
    if args.gain_out is not None:
        op.gain_out_mv = args.gain_out
    if args.t4_gain is not None:
        op.pair_gain = [g for g in optic.DEFAULT_PAIR_GAIN if not g[0].startswith("^T[45]")] + [(r"^T[45][abcd]$", r".*", args.t4_gain)]
    print("LIF:", {k: getattr(lif, k) for k in ["std_u", "std_tau", "adapt_jump", "same_type_gain", "input_norm_alpha", "input_norm_ref", "w_syn", "conn_cap"]}, " gain_out", op.gain_out_mv)

    c = connectome.load(verbose=False)
    n = c.neurons
    side = n.somaSide.to_numpy()
    leg = c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"]); legL = leg[side[leg] == "L"]; legR = leg[side[leg] == "R"]
    wg = body.wing_groups(c)
    res = {}

    def B():
        return brain.Brain(c, lif)

    # ---- rest
    b = B(); b.run_ms(500); res["rest_spikes_per_step"] = float(b.total_spikes())

    # ---- taste
    taste = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "flyverse", "data", "taste_grns.csv"))
    sweet = c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
    sweet = sweet[np.isin(n.subclass.to_numpy()[sweet], ["labellar bristle", "taste peg"])]
    b = B(); b.set_poisson(sweet, 100.0); b.run_ms(600); rt = b.rate[0].cpu().numpy()
    res["taste_MN9"] = float(rt[c.select(type="MN9")].mean()); res["taste_GNG175"] = float(rt[c.select(type="GNG175")].mean())
    res["taste_frac"] = float((rt > 1).mean()); res["taste_top"] = top_types(c, rt)

    # ---- smell
    olf = olfaction.Olfaction(c, [("apple", (0.25, 0.15, 0.79), 1.0)])
    b = B(); olf.apply(b, (0.19, 0.15, 0.75)); b.run_ms(800); rt = b.rate[0].cpu().numpy()
    pn = c.select(type="~_l2PN|_adPN|_lPN|_lvPN|_ilPN"); kc = c.select(type="~^KC"); ln = c.select(type="~^(lLN|v2LN)")
    res["smell_PN"] = float(rt[pn].mean()); res["smell_PN_max"] = float(rt[pn].max()); res["smell_KC"] = float(rt[kc].mean())
    res["smell_KC_active"] = int((rt[kc] > 1).sum()); res["smell_LN"] = float(rt[ln].mean()); res["smell_frac"] = float((rt > 1).mean())
    res["smell_top"] = top_types(c, rt)

    # ---- DN drive
    for name, idx in [("DNa02_L", c.select(type="DNa02", somaSide="L")), ("DNp09", c.select(type="DNp09")), ("MDN", c.select(type="MDN"))]:
        b = B(); b.set_poisson(idx, 150.0); b.run_ms(400); rt = b.rate[0].cpu().numpy()
        res[f"dn_{name}"] = {"legL": float(rt[legL].mean()), "legR": float(rt[legR].mean()), "power": float(rt[wg.power].mean()),
                             "frac": float((rt > 1).mean()), "top": top_types(c, rt)}

    # ---- walk / loom / rotate (hybrid)
    r = retina.build_retina(c)
    w, info = world.make_room()
    w.spheres.append(world.Sphere((9, 9, 9), (0.03,) * 3, "black")); loom_idx = len(w.spheres) - 1
    dirs_b, wts = r.ray_directions(); wts_t = torch.from_numpy(wts).float().to(w.device)
    ol = optic.OpticLobe(c, r, op); ol.relax()
    b = B(); b.freeze(ol.rate_idx)
    fly = body.FlyState(x=-0.3, y=0.0, z=info["table_top_z"], heading=0.0)

    def col_rad():
        d = fly.body_to_world(dirs_b.reshape(-1, 3)); o = np.broadcast_to(fly.eye_pos, d.shape)
        rad = w.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        return (rad.reshape(dirs_b.shape[0], dirs_b.shape[1], 4) * wts_t[None, :, None]).sum(1)

    gf = c.select(type="DNp01"); a02L = c.select(type="DNa02", somaSide="L"); a02R = c.select(type="DNa02", somaSide="R")
    gf_walk = []; pw_walk = []
    olf_walk = olfaction.Olfaction(c, [(name, cen, 1.0) for name, cen, rad in info["fruit"]])   # smell on, as in the demo
    for k in range(150):
        fly.x += 0.004 * 0.01; olf_walk.apply(b, fly.eye_pos); b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
        if k >= 50:
            gf_walk.append(float(b.rate[0, b._idx(gf)].mean())); pw_walk.append(float(b.rate[0, b._idx(wg.power)].mean()))
    rt = b.rate[0].cpu().numpy()
    res["walk_GF_mean"] = float(np.mean(gf_walk)); res["walk_GF_max"] = float(np.max(gf_walk)); res["walk_frac"] = float((rt > 1).mean())
    res["walk_power_mean"] = float(np.mean(pw_walk)); res["walk_power_max"] = float(np.max(pw_walk))
    res["walk_leg"] = float(rt[leg].mean())
    res["walk_top"] = top_types(c, rt)
    eye = fly.eye_pos + np.array([0, 0, 0.01]); esc = None; gf_peak = 0.0
    for k in range(80):
        d = max(0.5 - 1.0 * k * 0.01, 0.035)
        w.move_sphere(loom_idx, eye + np.array([0.0, d, 0.0]))
        b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
        g = float(b.rate[0, b._idx(gf)].mean()); gf_peak = max(gf_peak, g)
        if esc is None and g >= 20:
            esc = d * 100
    res["loom_GF_peak"] = gf_peak; res["loom_escape_cm"] = esc
    w.move_sphere(loom_idx, (9, 9, 9))
    out = {}
    dn_all = c.select(superclass="descending_neuron"); dn_side = side[dn_all]; dn_type = n.type.fillna("").to_numpy()[dn_all]
    for name, sgn in [("left", 1), ("right", -1)]:
        acc = np.zeros(len(dn_all))
        for k in range(80):
            fly.heading += sgn * np.deg2rad(90) * 0.01
            b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
            if k >= 20:
                acc += b.rate[0, b._idx(dn_all)].cpu().numpy()
        acc /= 60
        out[name] = (float(b.rate[0, b._idx(a02L)].mean()), float(b.rate[0, b._idx(a02R)].mean()))
        df = pd.DataFrame({"type": dn_type, "side": dn_side, "rate": acc})
        piv = df.groupby(["type", "side"]).rate.mean().unstack().fillna(0)
        piv["asym"] = piv.get("L", 0) - piv.get("R", 0)
        piv = piv[(piv[["L", "R"]].max(axis=1) > 3)]
        res[f"rotate_{name}_asym_dns"] = piv.reindex(piv.asym.abs().sort_values(ascending=False).index).head(6).round(1).to_dict("index")
        for k in range(50):
            b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
    res["rotate_left_DNa02_L_R"] = out["left"]; res["rotate_right_DNa02_L_R"] = out["right"]

    # ---- report
    print(f"rest      spikes/step {res['rest_spikes_per_step']:.0f}")
    print(f"taste     MN9 {res['taste_MN9']:.1f} Hz  GNG175 {res['taste_GNG175']:.0f}  frac {res['taste_frac']:.3f}  top {res['taste_top']}")
    print(f"smell     PN {res['smell_PN']:.0f} (max {res['smell_PN_max']:.0f})  KC {res['smell_KC']:.2f} ({res['smell_KC_active']} active)  LN {res['smell_LN']:.0f}  frac {res['smell_frac']:.3f}  top {res['smell_top']}")
    for name in ["DNa02_L", "DNp09", "MDN"]:
        d = res[f"dn_{name}"]
        print(f"dn {name:8s} legL {d['legL']:.1f} legR {d['legR']:.1f} power {d['power']:.1f} frac {d['frac']:.3f} top {d['top']}")
    print(f"walk      GF mean {res['walk_GF_mean']:.1f} max {res['walk_GF_max']:.0f}  wing power mean {res['walk_power_mean']:.1f} max {res['walk_power_max']:.0f}  leg MN {res['walk_leg']:.1f}  frac {res['walk_frac']:.3f}  top {res['walk_top']}")
    print(f"loom      GF peak {res['loom_GF_peak']:.0f} Hz  escape at {res['loom_escape_cm']} cm")
    print(f"rotate    left: DNa02 L/R {out['left'][0]:.1f}/{out['left'][1]:.1f}   right: {out['right'][0]:.1f}/{out['right'][1]:.1f}")
    for name in ["left", "right"]:
        print(f"  rotate {name:5s} most lateralised DNs (L, R, L-R Hz): " + "; ".join(f"{t}: {v['L']:.0f}/{v['R']:.0f} ({v['asym']:+.0f})" for t, v in res[f"rotate_{name}_asym_dns"].items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
