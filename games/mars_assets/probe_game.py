"""Dev probe for games/mars.py (dev seeds >= 100 only; run on the cluster's GPUs, never on the desktop).

The game's own `MarsRover`, ticked headless without the camera or the HUD, in its control arm (`--control off`: both
decoders read their cells, neither is applied), so the rover's path is the GAME's and is identical in every arm: the
autopilot drives the route line into H1 (and, after the detour, H2). The clips' build (games/mars.py) changed the
layout and the devil after this probe's measurements; see the caption for which build each number came from. For one eye mount, two arms, each with a fresh
brain on the same seed:

    visible  the game's scene, exactly as the fly sees it in the clips
    hidden   the hazard boulders (and their shadows) removed from the scene the fly's eye traces; the camera, the
             collision rule and the tally are unchanged

A difference between the arms is what the boulders do to the decoders' inputs; flow, terrain, the start and the
collision stops are common to both. One row per 10 ms tick is kept.

    python games/mars_assets/probe_game.py --seed 100 --eye mast_post --seconds 16 --out out/games/mars/probe2/pg_s100.json
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

COLS = ["t", "x", "y", "v", "state", "gap", "eye_gap", "ang_deg", "gf", "DNp01_L", "DNp01_R", "L", "R", "side",
        "LPLC2_L", "LPLC2_R", "LC4_L", "LC4_R", "mdn"]
STATES = {"wait": 0, "drive": 1, "stop": 2, "reverse": 3, "bump": 4}


def stats(a):
    a = np.asarray(a, float)
    if not len(a):
        return None
    return {"n": int(len(a)), "max": round(float(a.max()), 2), "min": round(float(a.min()), 2),
            "mean": round(float(a.mean()), 3), "sd": round(float(a.std()), 3)}


def make_game(seed, eye, arm, seconds):
    ap = M.add_args(gc.standard_args("probe", seconds=seconds))
    args = ap.parse_args(["--seed", str(seed), "--control", "off", "--eye", eye, "--seconds", str(seconds)])
    g = M.MarsRover(args)
    g.hide_hazards = arm == "hidden"
    return g


def rover_columns(g):
    """Columns' worth of eye rays (retina-weighted) that hit the rover's own body at the start pose."""
    w = torch.as_tensor(np.tile(g.eyes.weights, g.eyes.n_col), device=g.dev)
    R = g.pose_R
    d = torch.as_tensor(g.eyes.world_dirs(R[:, 0], R[:, 1], R[:, 2]), device=g.dev)
    o = torch.tensor(g.eye_pos(), dtype=torch.float32, device=g.dev)
    _, _, _, occ = g.rover.hit(o, d, torch.full((d.shape[0],), float("inf"), device=g.dev), dense=True)
    return round(float((w * occ.float()).sum()), 1)


def run_arm(seed, eye, arm, seconds):
    t_build = time.time()
    g = make_game(seed, eye, arm, seconds)
    info = {"arm": arm, "build_s": round(time.time() - t_build, 1), "preset": g.fb.preset,
            "instruments": sorted(g.fb.instruments), "eye_local_m": g.eye_local.tolist(),
            "hazards_hidden": g.hide_hazards, "rover_columns_at_start": rover_columns(g),
            "modules": [{k: r.get(k) for k in ("name", "kind", "n_writes")} for r in g.fb.module_records()]}
    print(arm, json.dumps(info), flush=True)
    rows = []
    t0 = time.time()
    for k in range(int(round(seconds * 100))):
        g.tick()
        st, sp, rr = g.steer_dec.value, g.stop_dec.value, g.rates
        e = g.eye_pos()
        live = [h for h in g.tally.rows if h["outcome"] is None]
        if live:
            h = min(live, key=lambda q: M.rect_circle_clearance(g.x, g.y, g.yaw, M.ROVER_L / 2, M.ROVER_W / 2,
                                                                 q["x"], q["y"], q["r"]))
            gap = M.rect_circle_clearance(g.x, g.y, g.yaw, M.ROVER_L / 2, M.ROVER_W / 2, h["x"], h["y"], h["r"])
            dist = math.hypot(h["x"] - e[0], h["y"] - e[1])
            eye_gap = dist - h["r"]
            ang = 2 * math.degrees(math.atan2(h["r"], max(dist, 1e-3))) if dist > h["r"] else 180.0
        else:
            gap = eye_gap = 99.0
            ang = 0.0
        rows.append([round(g.t_s, 3), g.x, g.y, g.drive.v, STATES[g.drive.state], gap, eye_gap, ang, sp["gf"],
                     rr["DNp01_L"], rr["DNp01_R"], st["L"], st["R"], st["side"], rr["LPLC2_L"], rr["LPLC2_R"],
                     rr["LC4_L"], rr["LC4_R"], 0.5 * (rr["MDN_L"] + rr["MDN_R"])])
        if k % 100 == 0:
            print(f"  {arm} t {g.t_s:5.2f} x {g.x:6.2f} gap {gap:6.2f} gf {sp['gf']:5.1f} side {st['side']:+5.2f} "
                  f"L {st['L']:4.2f} R {st['R']:4.2f} wall {time.time() - t0:6.1f}", flush=True)
    A = np.asarray(rows, float)
    c = {kk: A[:, i] for i, kk in enumerate(COLS)}
    driving = c["state"] == 1
    wins = {"warm_t<1s": c["t"] < M.WARM_S,
            "cruise_gap>15m": driving & (c["v"] > 4.0) & (c["gap"] > 15.0),
            "approach_gap<8m": driving & (c["v"] > 1.0) & (c["gap"] < 8.0),
            "approach_gap<3m": driving & (c["v"] > 1.0) & (c["gap"] < 3.0),
            "all_t>=1s": c["t"] >= M.WARM_S}
    keys = ("gf", "side", "L", "R", "LPLC2_L", "LPLC2_R", "LC4_L", "LC4_R", "mdn", "DNp01_L", "DNp01_R")
    out = dict(info, wall_s=round(time.time() - t0, 1), ticks=len(rows), x_end=round(float(c["x"][-1]), 2))
    for name, m in wins.items():
        out[name] = {kk: stats(c[kk][m]) for kk in keys}
        out[name]["frac_|side|>dead"] = round(float((np.abs(c["side"][m]) > M.STEER_DEAD).mean()), 3) if m.any() else None
    ev = g.log.events
    out["events"] = {kk: [e for e in ev if e["kind"] == kk] for kk in ("gf_cross", "steer_on", "collision")}
    pre = []
    for e in out["events"]["collision"]:
        for span in (1.0, 0.3):
            m = (c["t"] > e["t_s"] - span) & (c["t"] <= e["t_s"])
            pre.append({"hazard": e["hazard"], "t_s": e["t_s"], "window_s": span,
                        **{kk: stats(c[kk][m]) for kk in ("gf", "side", "L", "R", "ang_deg", "eye_gap")}})
    out["before_each_collision"] = pre
    i = int(np.argmax(c["gf"]))
    out["gf_peak"] = {"hz": round(float(c["gf"][i]), 2), "t_s": float(c["t"][i]), "gap_m": round(float(c["gap"][i]), 2),
                      "state": int(c["state"][i])}
    out["cols"] = COLS
    out["rows"] = np.round(A, 4).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eye", choices=sorted(M.EYE_MOUNTS), required=True)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--arms", default="visible,hidden")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.seed >= 100, "dev seeds only (>= 100)"
    assert torch.cuda.is_available(), "cluster GPU only"
    res = {"seed": a.seed, "eye": a.eye, "device": torch.cuda.get_device_name(0), "seconds": a.seconds,
           "sources": gc.source_hashes([M.__file__, __file__, gc.__file__]), "arms": {}}
    for arm in a.arms.split(","):
        res["arms"][arm] = run_arm(a.seed, a.eye, arm, a.seconds)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res))
        r = res["arms"][arm]
        print(arm, "gf peak", r["gf_peak"], "gf_cross", len(r["events"]["gf_cross"]), "steer_on",
              len(r["events"]["steer_on"]), "collisions", [(e["hazard"], e["t_s"]) for e in r["events"]["collision"]],
              "approach<3m gf", r["approach_gap<3m"]["gf"], flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
