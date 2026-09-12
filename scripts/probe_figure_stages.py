"""Per-stage figure-propagation map: where in the rate optic lobe is a small object lost?

Extends scripts/probe_figure_ground.py's retinotopic figure statistic (the signed per-column (stimulus - none)
deviation in the columns that view the object minus the same in background columns, as a z against the background
scatter) to EVERY optic-lobe type -- rate units (L1-L5 -> Mi/Tm/TmY/Dm -> T2/T3/T4/T5 -> LPi/Li) by their deviation
from the operating point and the spiking visual projection neurons (LC / LPLC / LLPC / LPT / MeVP ...) by the drive
they receive -- ordered by processing stage, for two stimuli:

  * `apple`: the static apple of probe_figure_ground (fly pinned 9 cm from the apple, heading oscillating +-20 deg at
    0.5 Hz = self-motion; the same scene with every fruit moved out);
  * `ball`:  the moving ball of probe_object_sweep (empty table, a 1 cm black ball 5 cm ahead sweeping +-6 cm in 3 s
    per pass; the same timeline with the ball parked out of the scene).

Every stimulus runs THREE simulations with the same seed: A = stimulus, B = none, C = none again.  (A - B) is the
figure, (C - B) is the none-vs-none NULL of the identical statistic (the native backend's run-to-run scatter; the
round-3 object-sweep convention, docs/audits/object_sweep.md section 8), so a type "carries" the object only where its
(A - B) figure stands above its own (C - B) null.  Two measures per type: `signed` (time-mean deviation, as
probe_figure_ground) and `abs` (time-mean |deviation|; for the moving ball the ON / OFF transients of a passing object
cancel in the signed mean, not in the rectified one).

Stage assignment: the Nern et al. 2025 figure groups (Primary_cell_type_table.xlsx of the supplementary code repository,
CC BY 4.0; data/external/nern2025/github_code_params/, git-ignored) when the file is present, else the same grouping
by type-name family (the table's groups are the paper's naming families: L / C / Lawf / T1 = lamina; Mi / Dm / Pm / Cm
/ Sm = medulla intrinsic; Tm / TmY / MeLo = medulla -> lobula projection; T2 / T3 / T4 / T5 / CT1 / Tlp / Y =
the T cells and lobula-plate connecting neurons (the table's group 4); Li / LPi = lobula (plate) intrinsic; the VPN families).  The two agree on every type in the tree (checked
in main() when the file is present).

    python scripts/probe_figure_stages.py --stimulus apple|ball|both [--seed 0] [--out out/x.json]
                                          [--optic-config NAME]      # an OpticParams override set from scripts/audit_optic.py
                                          [--receptor-model default|off|sign] [--receptor-net-rule abs]

Output: <out>.json with, per stimulus, one record per type (stage, kind, cells in the object / background columns,
figure and z for signed / abs, A-B and C-B), the column counts and the geometry; the console prints the stage-ordered
table.  CUDA only (the sims use the native backend), as the other probes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402

pygame.init(); pygame.display.set_mode((64, 64))
sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402

import room_demo as rd  # noqa: E402
from flyverse import brain, optic  # noqa: E402

FRAME_S = rd.FRAME_MS / 1000.0
NERN_TABLE = os.path.join(os.path.dirname(__file__), "..", "data", "external", "nern2025", "github_code_params", "Primary_cell_type_table.xlsx")

# ---- the static apple (probe_figure_ground) ----------------------------------------------------------------
APPLE = np.array([0.25, 0.15, 0.79])
APPLE_RADIUS_DEG = 30.0          # columns within this angle of the apple's direction view it (probe_figure_ground --radius-deg)
APPLE_SECONDS = 15.0             # 3 s skipped, 12 s scored
# ---- the moving ball (probe_object_sweep) --------------------------------------------------------------------
POS = (-0.20, 0.10, 0.75); HEADING = -np.pi / 2
BALL_R = 0.005; AHEAD = 0.05; HALF_SWEEP = 0.06; SWEEP_S = 3.0
BALL_SETTLE = 3.0; BALL_SECONDS = 12.0
BALL_RADIUS_DEG = 8.0            # columns within this angle of the ball's path (its centre) at some moment of the sweep
BG_MARGIN_DEG = 20.0             # background columns: farther than the object radius + this from the object
MIN_OBJ_CELLS = 5; MIN_BG_CELLS = 20   # a type is scored only with this many cells in the object / background columns

# ---- stages ---------------------------------------------------------------------------------------------------
STAGES = {1: "1 lamina (L1-L5, C2/C3, Lawf, T1)", 2: "2a medulla intrinsic (Mi, Dm, Pm, Cm, Sm)",
          3: "2b medulla -> lobula projection (Tm, TmY, MeLo)", 4: "3 T cells + lobula-plate connecting (T2/T3, T4/T5, CT1, Tlp, Y)",
          5: "4 lobula / lobula-plate intrinsic (Li, LPi)", 6: "5 visual projection neurons (spiking; drive)",
          7: "6 visual centrifugal (spiking; drive)"}
# Nern 2025 figure groups -> stage (Primary_cell_type_table.xlsx, column figure_group)
NERN_GROUP_STAGE = {1: 1, 5: 2, 6: 2, 7: 2, 8: 2, 2: 3, 3: 3, 4: 4, 9: 5, 10: 5, 11: 5, 12: 5}
# the same grouping by name family (used when the table is absent; the two are checked against each other)
FAMILY_STAGE = [(r"^(L[1-5]|C[23]|Lawf[12]|T1|Lai)$", 1), (r"^(Mi|Dm|Pm|Cm|Sm)", 2), (r"^(Tm(?!23|24|Y)|TmY(?!20)|MeLo)", 3),
                (r"^(T2a?|T3|T4[abcd]|T5[abcd]|T4_unclear|T5a_unclear|CT1|TmY20|Am1|LOLP1|Tlp|Y\d|Y_unclear)", 4),
                (r"^(Li|LPi|Tm23|Tm24|LT33|aMe6b|HBeyelet)", 5)]
# (checked against the Nern table locally: the family rule reproduces the table's group -> stage for every ol_intrinsic type
#  it covers; Tlp / Y / TmY20 sit with the T cells in the table's group 4, so they do here)


def stage_table():
    """{type: stage} from the Nern table when present (plus the family rule for anything not in it), else the family rule."""
    from_table = {}
    if os.path.isfile(NERN_TABLE):
        try:
            import pandas as pd
            x = pd.read_excel(NERN_TABLE)
            for t, g, mg in zip(x["type"], x["figure_group"], x["main_groups"]):
                try:
                    from_table[str(t)] = NERN_GROUP_STAGE.get(int(g))
                except (TypeError, ValueError):
                    from_table[str(t)] = 6 if str(mg) == "VPN" else 7 if str(mg) == "VCN" else None
        except Exception as e:  # noqa: BLE001
            print(f"(Nern table unreadable: {e!r}; family rule only)")
    return from_table


def stage_of(t, superclass, table):
    if superclass == "visual_projection":
        return 6
    if superclass == "visual_centrifugal":
        return 7
    if t in table and table[t] is not None:
        return table[t]
    for pat, s in FAMILY_STAGE:
        if re.match(pat, t):
            return s
    return None


# ---- OpticParams overrides (scripts/audit_optic.py configurations) ---------------------------------------------
def install_optic_overrides(overrides: dict):
    """Every optic.OpticParams built from here on (room_demo.Sim's included) carries `overrides` (field -> value)."""
    if not overrides:
        return
    O = optic.OpticParams

    def make(**kw):
        p = O(**kw)
        for k, v in overrides.items():
            setattr(p, k, v)
        return p
    optic.OpticParams = make


def patch_receptor(model, net_rule):
    """As probe_object_sweep.patch_receptor: 'default' leaves LIFParams alone; 'off' APPLIES receptor_model=None."""
    if model == "default":
        return
    L = brain.LIFParams
    rm = None if model in (None, "off") else model

    def make(**kw):
        p = L(**kw); p.receptor_model = rm; p.receptor_net_rule = net_rule
        return p
    brain.LIFParams = make


# ---- retinotopy ---------------------------------------------------------------------------------------------
def column_of_cells(sim):
    """Column index per connectome cell (-1 = none): rate cells from their hex annotation, then three propagation passes
    (an unassigned rate cell takes the column of its strongest rate input partner); spiking cells the column of their
    strongest rate input partner.  probe_figure_ground's rule, applied to every cell."""
    import scipy.sparse as sp
    c = sim.c; o = sim.fb.optic; ret = sim.fb.retina
    key = {(str(sd), int(h[0]), int(h[1])): i for i, (sd, h) in enumerate(zip(ret.col_side, ret.col_hex))}
    ridx = np.asarray(o.rate_idx)
    nr = c.neurons.iloc[ridx]
    h1 = nr["hex1"].to_numpy(); h2 = nr["hex2"].to_numpy(); sd_ = nr["hex_side"].fillna("").to_numpy()
    col_r = np.array([key.get((str(s_), int(a_), int(b_)), -1) if (a_ == a_ and b_ == b_) else -1 for s_, a_, b_ in zip(sd_, h1, h2)])
    n_ann = int((col_r >= 0).sum())
    Wabs = sp.csr_matrix(abs(c.W)); Wr = Wabs[ridx][:, ridx].tocsr()
    for _ in range(3):
        todo = np.flatnonzero(col_r < 0)
        for i in todo:
            s0, s1 = Wr.indptr[i], Wr.indptr[i + 1]; pre = Wr.indices[s0:s1]; w = Wr.data[s0:s1]
            ok = col_r[pre] >= 0
            if ok.any():
                col_r[i] = col_r[pre[ok][np.argmax(w[ok])]]
    col = -np.ones(c.n, np.int64); col[ridx] = col_r
    # spiking cells: strongest rate input partner that has a column
    Wsr = Wabs[:, ridx].tocsr()
    spk = np.flatnonzero(col < 0)
    for i in spk:
        s0, s1 = Wsr.indptr[i], Wsr.indptr[i + 1]
        if s1 == s0:
            continue
        pre = Wsr.indices[s0:s1]; w = Wsr.data[s0:s1]; ok = col_r[pre] >= 0
        if ok.any():
            col[i] = col_r[pre[ok][np.argmax(w[ok])]]
    return col, n_ann


# ---- the two protocols ----------------------------------------------------------------------------------------
def new_sim(seed, fruit_set, start):
    sim = rd.Sim(seed, start=start, trail_seconds=0.0, fruit_set=fruit_set, fence=True, wind_speed=0.0,
                 cuda_kernels=True, cuda_graphs=False, event_driven=True, cuda_sparse="warp")
    dev = sim.fb.brain.device
    assert dev.type == "cuda", f"device {dev}: not CUDA (node race; resubmit)"
    return sim


def record(sim, n_frames, skip, place_fn):
    """Time-mean signed and |.| deviation of every rate unit and drive of every spiking cell over frames >= skip."""
    o = sim.fb.optic; b = sim.fb.brain; n = 0
    acc_s = torch.zeros(len(o.rate_idx), device=b.device); acc_a = torch.zeros_like(acc_s)
    drv_s = torch.zeros(sim.c.n, device=b.device); drv_a = torch.zeros_like(drv_s)
    t0 = time.time()
    for k in range(n_frames):
        place_fn(k); sim.step()
        if k >= skip:
            n += 1
            dr = (o.rates()[0] - o.r0[0]).reshape(-1); d = b.drive[0].reshape(-1)
            acc_s += dr; acc_a += dr.abs(); drv_s += d; drv_a += d.abs()
        if k % 500 == 499:
            print(f"    frame {k + 1}/{n_frames} ({time.time() - t0:.0f} s)", flush=True)
    f = lambda x: (x / max(n, 1)).detach().cpu().numpy()
    return {"rate_signed": f(acc_s), "rate_abs": f(acc_a), "drive_signed": f(drv_s), "drive_abs": f(drv_a), "frames": n}


def run_apple(seed, with_apple):
    r = 0.04 + 0.05
    pos = APPLE[:2] - r * np.array([np.cos(np.deg2rad(135)), np.sin(np.deg2rad(135))])
    heading0 = np.pi / 2
    sim = new_sim(seed, "apple" if with_apple else "all", (float(pos[0]), float(pos[1]), 0.75))
    if not with_apple:
        for i, s in enumerate(sim.world.spheres):
            if s.material in ("apple", "orange", "banana", "lime", "grape", "blueberry"):
                sim.world.move_sphere(i, (9, 9, 9))

    def place(k):
        h = heading0 + np.deg2rad(20) * np.sin(2 * np.pi * 0.5 * k / 100)
        sim.fly.place(float(pos[0]), float(pos[1]), 0.75, heading=h)
    rec = record(sim, int(APPLE_SECONDS * 100), 300, place)
    # object columns: column direction (body frame at the mean heading, to world) vs the apple's direction
    eye = np.array([pos[0], pos[1], 0.75 + 0.0012]); to_apple = APPLE - eye; to_apple /= np.linalg.norm(to_apple)
    sim.fly.place(float(pos[0]), float(pos[1]), 0.75, heading=heading0)
    col_dir_world = sim.fly.body_to_world(np.asarray(sim.fb.retina.col_dir))
    ang = np.degrees(np.arccos(np.clip(col_dir_world @ to_apple, -1, 1)))
    geom = {"pos": [float(pos[0]), float(pos[1]), 0.75], "heading0": float(heading0), "apple": APPLE.tolist(),
            "object_radius_deg": APPLE_RADIUS_DEG, "bg_margin_deg": BG_MARGIN_DEG, "seconds": APPLE_SECONDS, "skip_s": 3.0}
    return sim, rec, ang, geom


def ball_offset(t):
    phase = (t / SWEEP_S) % 2.0
    frac = phase if phase < 1.0 else 2.0 - phase
    return HALF_SWEEP - 2 * HALF_SWEEP * frac


def run_ball(seed, with_ball):
    sim = new_sim(seed, "apple", POS)
    for i, s in enumerate(sim.world.spheres):
        if s.material == "apple":
            sim.world.move_sphere(i, (9, 9, 9))
    sim.world.move_sphere(sim.loom_idx, (9, 9, 9), (BALL_R, BALL_R, BALL_R))
    fly = sim.fly; fly.place(*POS, heading=HEADING)
    eye0 = fly.eye_pos.copy(); fwd = fly.forward.copy(); left = fly.left.copy()
    ball_z = sim.info["table_top_z"] + BALL_R
    n_settle = int(round(BALL_SETTLE / FRAME_S)); n_sweep = int(round(BALL_SECONDS / FRAME_S))

    def place(k):
        fly.place(*POS, heading=HEADING)
        if with_ball and k >= n_settle:
            s = ball_offset((k - n_settle) * FRAME_S)
            centre = eye0 + AHEAD * fwd + s * left; centre[2] = ball_z
            sim.world.move_sphere(sim.loom_idx, centre)
    rec = record(sim, n_settle + n_sweep, n_settle, place)
    # object columns: the ball's direction in the body frame (x forward, y left, z up) over the sweep; per column the
    # minimum angle to the ball's centre over the positions it takes
    cd = np.asarray(sim.fb.retina.col_dir)
    ang = np.full(len(cd), 180.0)
    for s in np.linspace(-HALF_SWEEP, HALF_SWEEP, 49):
        rel = np.array([AHEAD, s, ball_z - eye0[2]]); rel /= np.linalg.norm(rel)
        ang = np.minimum(ang, np.degrees(np.arccos(np.clip(cd @ rel, -1, 1))))
    geom = {"pos": list(POS), "heading": float(HEADING), "ball_radius_m": BALL_R, "ahead_m": AHEAD, "half_sweep_m": HALF_SWEEP,
            "sweep_s": SWEEP_S, "angular_diameter_deg": float(2 * np.degrees(np.arctan(BALL_R / AHEAD))),
            "object_radius_deg": BALL_RADIUS_DEG, "bg_margin_deg": BG_MARGIN_DEG, "seconds": BALL_SECONDS, "settle_s": BALL_SETTLE}
    return sim, rec, ang, geom


# ---- statistics -----------------------------------------------------------------------------------------------
def figure_stats(xA, xB, obj, bg):
    """(A - B) in the object cells minus (A - B) in the background cells; z against the background scatter."""
    d = xA - xB
    d_obj = d[obj].mean(); d_bg = d[bg].mean(); sd = d[bg].std() + 1e-9
    return {"figure": float(d_obj - d_bg), "z": float((d_obj - d_bg) / sd), "sd_bg": float(sd),
            "resp_obj_A": float(xA[obj].mean()), "resp_obj_B": float(xB[obj].mean()), "bg_level_B": float(xB[bg].mean())}


def per_type(sim, recs, ang, radius_deg, stages):
    """recs = {'A': rec, 'B': rec, 'C': rec}; returns {type: record} for every type with >= MIN_OBJ_CELLS / MIN_BG_CELLS cells."""
    c = sim.c; o = sim.fb.optic
    col, n_ann = column_of_cells(sim)
    in_obj = ang <= radius_deg; far = ang >= radius_deg + BG_MARGIN_DEG
    types = c.neurons.type.fillna("").to_numpy(); sc = c.neurons.superclass.fillna("").to_numpy()
    ridx = np.asarray(o.rate_idx); is_rate = np.zeros(c.n, bool); is_rate[ridx] = True
    pos_r = -np.ones(c.n, np.int64); pos_r[ridx] = np.arange(len(ridx))
    has = col >= 0
    cell_obj = has & in_obj[np.clip(col, 0, len(in_obj) - 1)]; cell_bg = has & far[np.clip(col, 0, len(far) - 1)]
    out = {}
    for ty in sorted(set(types) - {""}):
        m = types == ty
        if not (is_rate[m].all() or sc[m][0] in ("visual_projection", "visual_centrifugal")):
            continue
        obj = m & cell_obj; bg = m & cell_bg
        if obj.sum() < MIN_OBJ_CELLS or bg.sum() < MIN_BG_CELLS:
            continue
        kind = "rate" if is_rate[m].all() else "spiking"
        rec = {"stage": stage_of(ty, sc[m][0], stages), "kind": kind, "superclass": sc[m][0], "n_cells": int(m.sum()),
               "cells_obj": int(obj.sum()), "cells_bg": int(bg.sum())}
        if kind == "rate":
            oi = pos_r[obj]; bi = pos_r[bg]
            get = lambda r, meas: r["rate_" + meas]
            sel_o, sel_b = oi, bi
        else:
            get = lambda r, meas: r["drive_" + meas]
            sel_o, sel_b = np.flatnonzero(obj), np.flatnonzero(bg)
        for meas in ("signed", "abs"):
            xA, xB, xC = (get(recs[k], meas) for k in ("A", "B", "C"))
            n = len(xA); obj_m = np.zeros(n, bool); obj_m[sel_o] = True; bg_m = np.zeros(n, bool); bg_m[sel_b] = True
            rec[meas] = {"AB": figure_stats(xA, xB, obj_m, bg_m), "CB": figure_stats(xC, xB, obj_m, bg_m)}
        out[ty] = rec
    return out, {"columns_obj": int(in_obj.sum()), "columns_bg": int(far.sum()), "columns": int(len(ang)),
                 "rate_cells_annotated": n_ann, "cells_with_column": int(has.sum())}


def print_table(res, cols, label):
    print(f"\n== {label}: {cols['columns_obj']} object columns, {cols['columns_bg']} background columns of {cols['columns']}; "
          f"{cols['cells_with_column']:,} cells with a column ({cols['rate_cells_annotated']:,} rate cells annotated)")
    print(f"{'type':10s} {'kind':8s} {'obj':>4s} {'bg':>5s} | signed: figure(A-B)   z(A-B)  z(C-B) | abs: figure(A-B)   z(A-B)  z(C-B)")
    for s in sorted(STAGES):
        rows = [(t, r) for t, r in res.items() if r["stage"] == s]
        if not rows:
            continue
        rows.sort(key=lambda tr: -abs(tr[1]["signed"]["AB"]["z"]))
        n_carry = sum(abs(r["signed"]["AB"]["z"]) >= 3 or abs(r["abs"]["AB"]["z"]) >= 3 for _, r in rows)
        n_null = sum(abs(r["signed"]["CB"]["z"]) >= 3 or abs(r["abs"]["CB"]["z"]) >= 3 for _, r in rows)
        print(f"-- stage {STAGES[s]}: {len(rows)} types; |z| >= 3 on either measure: {n_carry} (A-B) vs {n_null} (C-B null)")
        for t, r in rows[:12]:
            sg, ab = r["signed"], r["abs"]
            print(f"{t:10s} {r['kind']:8s} {r['cells_obj']:4d} {r['cells_bg']:5d} | {sg['AB']['figure']:+.5f} {sg['AB']['z']:+7.2f} {sg['CB']['z']:+7.2f} | "
                  f"{ab['AB']['figure']:+.5f} {ab['AB']['z']:+7.2f} {ab['CB']['z']:+7.2f}")
        if len(rows) > 12:
            print(f"  ... {len(rows) - 12} more")


def run_stimulus(stim, seed, stages):
    runner = run_apple if stim == "apple" else run_ball
    recs = {}; sim = None
    for tag, on in (("A", True), ("B", False), ("C", False)):
        print(f"[{stim} {tag}: {'stimulus' if on else 'none'}] seed {seed}", flush=True)
        t0 = time.time()
        s, rec, ang, geom = runner(seed, on)
        recs[tag] = rec
        print(f"    {rec['frames']} frames scored ({time.time() - t0:.0f} s)", flush=True)
        if sim is None:
            sim = s                                    # kept for the connectome / retina / column map
        else:
            del s; torch.cuda.empty_cache()
    res, cols = per_type(sim, recs, ang, geom["object_radius_deg"], stages)
    print_table(res, cols, f"{stim} seed {seed}")
    del sim; torch.cuda.empty_cache()
    return {"types": res, "columns": cols, "geometry": geom}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus", default="both", choices=["apple", "ball", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/figure_stages.json")
    ap.add_argument("--optic-config", default=None, help="an OpticParams override set by name (scripts/audit_optic.py CONFIGS)")
    ap.add_argument("--receptor-model", default="default", choices=["default", "off", "sign"])
    ap.add_argument("--receptor-net-rule", default="abs", choices=["class", "abs", "nonmda"])
    ap.add_argument("--quick", action="store_true", help="smoke test: 4 s apple window, 1 + 3 s ball window")
    args = ap.parse_args(argv)
    if args.quick:
        global APPLE_SECONDS, BALL_SETTLE, BALL_SECONDS
        APPLE_SECONDS, BALL_SETTLE, BALL_SECONDS = 4.0, 1.0, 3.0
    assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    patch_receptor(args.receptor_model, args.receptor_net_rule)
    overrides = {}
    if args.optic_config:
        import audit_optic
        overrides = audit_optic.optic_overrides(args.optic_config)
        install_optic_overrides(overrides)
    stages = stage_table()
    src = "Nern 2025 figure groups" if stages else "type-name family rule"
    if stages:   # the family rule must agree with the table on every type it covers
        dis = [(t, s) for t, s in stages.items() if s is not None and s <= 5 and stage_of(t, "ol_intrinsic", {}) not in (s, None)]
        print(f"stage source: {src} ({len(stages)} types); family rule disagrees on {len(dis)}: {dis[:10]}")
    else:
        print(f"stage source: {src}")
    rm = brain.LIFParams()
    print(f"figure stages: stimulus {args.stimulus}, seed {args.seed}, optic config {args.optic_config or 'baseline'} {overrides}, "
          f"receptor {rm.receptor_model}/{rm.receptor_net_rule}; torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)
    out = {"config": {"stimulus": args.stimulus, "seed": args.seed, "optic_config": args.optic_config or "baseline",
                      "optic_overrides": {k: (v if not isinstance(v, (list, dict)) else json.loads(json.dumps(v))) for k, v in overrides.items()},
                      "receptor_model": rm.receptor_model, "receptor_net_rule": rm.receptor_net_rule, "stage_source": src,
                      "stages": STAGES, "device": torch.cuda.get_device_name(0), "torch": torch.__version__}}
    for stim in (["apple", "ball"] if args.stimulus == "both" else [args.stimulus]):
        out[stim] = run_stimulus(stim, args.seed, stages)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
