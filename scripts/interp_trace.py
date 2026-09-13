"""CLI of the trace tool (flyverse/interp/trace.py; docs/INTERP.md 4.2): where along the depth from a sensory population
is a stimulus lost?

    record   (GPU, on the cluster)  one arm of one protocol -> <out>.npz/.json (per-cell time-means over the window,
                                     provenance with the realised device, the column map) + <out>_series.npz (per-frame
                                     pooled series per type)
    analyse  (CPU)                   stimulus / control / null recordings -> the trace Result JSON + the printed table
    audit    (CPU)                   a trace JSON side by side with the validation reference (VALIDATION['trace'])

Protocols (`--protocol`), each an existing probe's protocol, reused not re-invented:
  object   scripts/probe_object_sweep.py: pinned fly on the empty fenced apple table, a 1 cm black ball 5 cm ahead sweeping
           +-6 cm in 3 s per pass, 12 s scored after 3 s settle; arms stim (ball) / ctrl (no ball) / null (no ball again);
           column map and object / background columns as scripts/probe_figure_stages.py run_ball (8 deg / +20 deg margin)
  apple    scripts/probe_figure_stages.py run_apple: the static apple 9 cm ahead-left, heading oscillating +-20 deg at
           0.5 Hz, 15 s with 3 s skipped; arms stim (apple) / ctrl (no fruit) / null (no fruit again)
  odour    scripts/screen_odour.py --fruit apple sites, pinned, 30 s with 3 s skipped: arms stim (apple8_into_wind) /
           ctrl (clean_into_wind) / null (clean_into_wind again), or any site name of SITES_APPLE

  The validation batch of docs/audits/interp_trace.md (30 jobs in ONE call: 2 protocols x 3 arms x 5 runs; the exact
  Mann-Whitney p of n v n runs floors at 2 / C(2n, n) -- 0.10 at 3 v 3, 0.029 at 4 v 4, 0.0079 at 5 v 5 -- so a verdict
  'result' (p <= 0.05) needs >= 4 runs per arm; five matches docs/audits/object_sweep.md 8.4):

    cmds=(); for proto in object odour; do pre=$([ "$proto" = object ] && echo obj || echo od); for a in stim ctrl null; do for s in 0 1 2 3 4; do
      cmds+=("python -c 'import torch; assert torch.cuda.is_available()' && mkdir -p out/trv && python scripts/interp_trace.py record --protocol $proto --arm $a --seed $s --series-every 5 --out out/trv/${pre}_${a}_r$s > out/trv/${pre}_${a}_r$s.txt 2>&1; tail -4 out/trv/${pre}_${a}_r$s.txt"); done; done; done
    python scripts/cluster_run.py --name trv --minutes 40 "${cmds[@]}" --fetch out/trv/ > out/trv_cluster.log 2>&1
    PYTHONIOENCODING=utf-8 python scripts/interp_trace.py analyse --source "type:R1-R6|R7y|R7p|R7d|R7_unclear|R8y|R8p|R8_unclear" \
      --stimulus "out/trv/obj_stim_r*" --control "out/trv/obj_ctrl_r*" --null-runs "out/trv/obj_null_r*" --stat best_cell --stage-table family \
      --decompose-at T3,T2,Tm5Y,TmY21 --json out/interp/trace/object_stage.json        # (--decompose-at first_lost: object_default.json)
    PYTHONIOENCODING=utf-8 python scripts/interp_trace.py analyse --source "class=olfactory" --stimulus "out/trv/od_stim_r*" \
      --control "out/trv/od_ctrl_r*" --null-runs "out/trv/od_null_r*" --stat mean --min-cells 2 --json out/interp/trace/odour_mean.json
    PYTHONIOENCODING=utf-8 python scripts/interp_trace.py audit --json out/interp/trace/object_stage.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flyverse.interp import common  # noqa: E402
from flyverse.interp import trace as tr  # noqa: E402

PHOTORECEPTORS = "type:R1-R6|R7y|R7p|R7d|R7_unclear|R8y|R8p|R8_unclear"
ODOUR_ARMS = {"stim": "apple8_into_wind", "ctrl": "clean_into_wind", "null": "clean_into_wind"}
ODOUR_SECONDS, ODOUR_SKIP_S = 30.0, 3.0


# ------------------------------------------------------------------------------------------------ record (GPU)
def install_overrides(lif_kw: dict, optic_kw: dict, receptor_model: str, net_rule: str, table: str | None):
    """Every LIFParams / OpticParams built from here on (room_demo.Sim's included) carries the CLI overrides -- the
    probes' patch_receptor / install_optic_overrides pattern; nothing in flyverse/ is edited."""
    from flyverse import brain, optic
    L, O = brain.LIFParams, optic.OpticParams
    rm = {"default": "keep", "off": None}.get(receptor_model, receptor_model)

    def make_lif(**kw):
        p = L(**kw)
        if rm != "keep":
            p.receptor_model = rm
            if rm is not None:
                p.receptor_net_rule = net_rule
        if table:
            p.receptor_table = table
        for k, v in lif_kw.items():
            setattr(p, k, v)
        return p

    def make_optic(**kw):
        p = O(**kw)
        for k, v in optic_kw.items():
            setattr(p, k, v)
        return p
    if rm != "keep" or table or lif_kw:
        brain.LIFParams = make_lif
    if optic_kw:
        optic.OpticParams = make_optic


def patch_cache(cache_dir):
    if not cache_dir:
        return
    from flyverse import connectome
    orig = connectome.load

    def load(cache_dir_=None, **kw):
        return orig(cache_dir=Path(cache_dir), **kw)
    connectome.load = load


def new_sim(seed, start, fruit_set, fence, wind_speed, allow_cpu, device):
    import torch
    import room_demo as rd
    flags = dict(cuda_kernels=True, cuda_graphs=False, event_driven=True, cuda_sparse="warp") if (torch.cuda.is_available() and not allow_cpu) else {}
    sim = rd.Sim(seed, start=start, trail_seconds=0.0, fruit_set=fruit_set, fence=fence, wind_speed=wind_speed, **flags)
    dev = sim.fb.brain.device
    print(f"sim ready; device {dev} (requested {device}); cuda available {torch.cuda.is_available()}", flush=True)
    if not allow_cpu:
        assert dev.type == "cuda", f"device {dev}: not CUDA (node race; resubmit)"
    return sim


def protocol_object(args, seed, arm):
    """probe_object_sweep's protocol with probe_figure_stages' column geometry."""
    import probe_object_sweep as pos_
    import probe_figure_stages as pfs
    with_ball = arm == "stim"
    sim = new_sim(seed, pos_.POS, "apple", True, 0.0, args.allow_cpu, args.device)
    for i, s in enumerate(sim.world.spheres):
        if s.material == "apple":
            sim.world.move_sphere(i, (9, 9, 9))
    sim.world.move_sphere(sim.loom_idx, (9, 9, 9), (pos_.BALL_R, pos_.BALL_R, pos_.BALL_R))
    fly = sim.fly; fly.place(*pos_.POS, heading=pos_.HEADING)
    eye0 = fly.eye_pos.copy(); fwd = fly.forward.copy(); left = fly.left.copy()
    ball_z = sim.info["table_top_z"] + pos_.BALL_R
    settle_s, seconds = (1.0, 3.0) if args.quick else (args.settle, args.seconds)
    n_settle = int(round(settle_s / pos_.FRAME_S)); n_sweep = int(round(seconds / pos_.FRAME_S))

    def place(k):
        fly.place(*pos_.POS, heading=pos_.HEADING)
        if with_ball and k >= n_settle:
            s = pos_.ball_offset((k - n_settle) * pos_.FRAME_S)
            centre = eye0 + pos_.AHEAD * fwd + s * left; centre[2] = ball_z
            sim.world.move_sphere(sim.loom_idx, centre)
    cd = np.asarray(sim.fb.retina.col_dir)
    ang = np.full(len(cd), 180.0)
    for s in np.linspace(-pos_.HALF_SWEEP, pos_.HALF_SWEEP, 49):
        rel = np.array([pos_.AHEAD, s, ball_z - eye0[2]]); rel /= np.linalg.norm(rel)
        ang = np.minimum(ang, np.degrees(np.arccos(np.clip(cd @ rel, -1, 1))))
    cols_obj, cols_bg = tr.figure_columns(cd, ang, pfs.BALL_RADIUS_DEG, pfs.BG_MARGIN_DEG)
    stim = {"protocol": "object", "arm": arm, "condition": "ball" if with_ball else "none",
            "params": {"pos": list(pos_.POS), "heading_rad": float(pos_.HEADING), "ball_radius_m": pos_.BALL_R, "ahead_m": pos_.AHEAD,
                       "half_sweep_m": pos_.HALF_SWEEP, "sweep_s": pos_.SWEEP_S, "angular_diameter_deg": float(2 * np.degrees(np.arctan(pos_.BALL_R / pos_.AHEAD))),
                       "settle_s": settle_s, "seconds": seconds, "object_radius_deg": pfs.BALL_RADIUS_DEG, "bg_margin_deg": pfs.BG_MARGIN_DEG,
                       "fruit_set": "apple (removed)", "fence": True, "wind_speed": 0.0},
            "control": "the same timeline with the ball parked out of the scene (probe_object_sweep --null: none vs none)"}
    return sim, place, n_settle + n_sweep, n_settle, cols_obj, cols_bg, stim, {"spiking": "drive_mv"}


def protocol_apple(args, seed, arm):
    """probe_figure_stages.run_apple: the static apple, heading oscillating +-20 deg at 0.5 Hz."""
    import probe_figure_stages as pfs
    with_apple = arm == "stim"
    r = 0.04 + 0.05
    pos = pfs.APPLE[:2] - r * np.array([np.cos(np.deg2rad(135)), np.sin(np.deg2rad(135))])
    heading0 = np.pi / 2
    sim = new_sim(seed, (float(pos[0]), float(pos[1]), 0.75), "apple" if with_apple else "all", True, 0.0, args.allow_cpu, args.device)
    if not with_apple:
        for i, s in enumerate(sim.world.spheres):
            if s.material in ("apple", "orange", "banana", "lime", "grape", "blueberry"):
                sim.world.move_sphere(i, (9, 9, 9))
    seconds = 4.0 if args.quick else pfs.APPLE_SECONDS
    skip = 100 if args.quick else 300

    def place(k):
        h = heading0 + np.deg2rad(20) * np.sin(2 * np.pi * 0.5 * k / 100)
        sim.fly.place(float(pos[0]), float(pos[1]), 0.75, heading=h)
    eye = np.array([pos[0], pos[1], 0.75 + 0.0012]); to_apple = pfs.APPLE - eye; to_apple /= np.linalg.norm(to_apple)
    sim.fly.place(float(pos[0]), float(pos[1]), 0.75, heading=heading0)
    col_dir_world = sim.fly.body_to_world(np.asarray(sim.fb.retina.col_dir))
    ang = np.degrees(np.arccos(np.clip(col_dir_world @ to_apple, -1, 1)))
    cols_obj, cols_bg = tr.figure_columns(col_dir_world, ang, pfs.APPLE_RADIUS_DEG, pfs.BG_MARGIN_DEG)
    stim = {"protocol": "apple", "arm": arm, "condition": "apple" if with_apple else "none",
            "params": {"pos": [float(pos[0]), float(pos[1]), 0.75], "heading0": float(heading0), "apple": pfs.APPLE.tolist(), "oscillation_deg": 20.0,
                       "oscillation_hz": 0.5, "seconds": seconds, "skip_s": skip / 100.0, "object_radius_deg": pfs.APPLE_RADIUS_DEG,
                       "bg_margin_deg": pfs.BG_MARGIN_DEG, "fence": True, "wind_speed": 0.0},
            "control": "the same scene with every fruit moved out"}
    return sim, place, int(seconds * 100), skip, cols_obj, cols_bg, stim, {"spiking": "drive_mv"}


def protocol_odour(args, seed, arm):
    """screen_odour --fruit apple: pinned at a site facing into / away from the wind."""
    import screen_odour as so
    site = ODOUR_ARMS.get(arm, arm)
    if site not in so.SITES_APPLE:
        raise SystemExit(f"odour arm {arm!r}: not stim / ctrl / null nor a site of screen_odour.SITES_APPLE {list(so.SITES_APPLE)}")
    x, y, z, hd = so.SITES_APPLE[site]
    sim = new_sim(seed, (x, y, z), "apple", False, 0.3, args.allow_cpu, args.device)
    seconds = 4.0 if args.quick else args.seconds if args.seconds != 12.0 else ODOUR_SECONDS
    skip = 100 if args.quick else int(ODOUR_SKIP_S * 100)

    def place(k):
        sim.fly.place(x, y, z, heading=np.deg2rad(hd))
    stim = {"protocol": "odour", "arm": arm, "site": site, "condition": "apple plume" if site.startswith("apple") else "plume-free",
            "params": {"site": site, "pos": [x, y, z], "heading_deg": hd, "seconds": seconds, "skip_s": skip / 100.0, "fruit_set": "apple",
                       "wind_speed": 0.3, "sites": so.SITES_APPLE},
            "control": "clean_into_wind (the plume-free spot, facing into the wind)"}
    return sim, place, int(seconds * 100), skip, None, None, stim, {"spiking": "rate_hz"}


PROTOCOLS = {"object": protocol_object, "apple": protocol_apple, "odour": protocol_odour}


def cmd_record(args):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if args.device and str(args.device).startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # a CPU smoke test never touches this machine's GPU (the cluster rule)
        args.allow_cpu = True
    import torch
    if not args.allow_cpu:
        assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    install_overrides(common.parse_kv(args.lif), common.parse_kv(args.optic), args.receptor_model, args.receptor_net_rule, args.receptor_table)
    patch_cache(args.cache_dir)
    seed = args.seed
    t0 = time.time()
    sim, place, n_frames, skip, cols_obj, cols_bg, stim, default_q = PROTOCOLS[args.protocol](args, seed, args.arm)
    fb = sim.fb
    acc = tr.ArmAccumulator(fb, keep_series=not args.no_series, series_every=args.series_every)
    for k in range(n_frames):
        place(k); sim.step(); acc.add(fb, k, skip)
        if k % 500 == 499:
            print(f"    frame {k + 1}/{n_frames} ({time.time() - t0:.0f} s)", flush=True)
    lif_p = fb.brain.p; optic_p = fb.optic.p if fb.optic is not None else None
    prov = common.provenance(sim.c, lif_p, optic_p, fb=fb, device=args.device, seeds=[seed], env_seeds=[seed], batch=1, stimulus=stim,
                             retina=tr.retina_record(fb.retina, sim.c), cache_dir=args.cache_dir)
    meta = {"protocol": args.protocol, "arm": args.arm, "seed": seed, "window_s": [skip / 100.0, n_frames / 100.0], "stimulus": stim,
            "default_quantity": default_q, "provenance": prov, "run_id": f"{args.protocol}-{args.arm}-r{seed}", "generator": " ".join(sys.argv)}
    if cols_obj is not None:
        col, n_ann = tr.column_of_cells(sim.c, fb.retina, fb.optic.rate_idx)
        meta.update({"column": col.tolist(), "columns_obj": cols_obj.tolist(), "columns_bg": cols_bg.tolist(), "rate_cells_annotated": n_ann,
                     "cells_with_column": int((col >= 0).sum()), "n_columns": int(fb.retina.n_columns)})
    cells, series = acc.finish(meta)
    out = Path(args.out)
    tr.save_recording(cells, out)
    if series is not None:
        tr.save_recording(series, Path(str(out) + "_series"))
    # a short console record: the named types' headline numbers
    ty = sim.c.neurons.type.fillna("").to_numpy()
    q = cells.quantities
    named = ["LC11", "LC10a", "LPLC2", "LC4", "Mi4", "Mi1", "Tm3", "T2", "T3", "Tm5Y", "TmY21", "LHPD4d1", "LHAV4a1_b", "DNp18", "DNa02"]
    parts = []
    for t in named:
        m = ty == t
        if not m.any():
            continue
        if np.isfinite(q["optic_dr"][0, m]).all():
            parts.append(f"{t} |dr| {np.nanmean(q['optic_dr_abs'][0, m]):.4f} (best {np.nanmax(q['optic_dr_abs'][0, m]):.4f})")
        else:
            parts.append(f"{t} {np.nanmean(q['rate_hz'][0, m]):.2f} Hz, drive {np.nanmean(q['drive_mv'][0, m]):+.3f} mV (best {np.nanmax(q['drive_mv'][0, m]):+.3f})")
    print(f"[{args.protocol} {args.arm} seed {seed}] {acc.n} frames scored, window {meta['window_s']} s, device {prov['execution']['device']} "
          f"({prov['execution'].get('device_name')}), {time.time() - t0:.0f} s wall")
    print("  " + "; ".join(parts))
    print(f"written {out.with_suffix('.npz')}" + (f" and {out}_series.npz" if series is not None else ""))
    if prov["execution"]["device"] is None or "cpu" in str(prov["execution"]["device"]):
        print("WARNING: device cpu -- resubmit (the JSON records the realised device)")


# ------------------------------------------------------------------------------------------------ analyse (CPU)
def cmd_analyse(args):
    from flyverse import connectome as cn
    c = cn.load(cache_dir=Path(args.cache_dir), verbose=False) if args.cache_dir else cn.load(verbose=False)
    stim = tr.load_runs(args.stimulus); ctrl = tr.load_runs(args.control); nul = tr.load_runs(args.null_runs) if args.null_runs else []
    if not stim or not ctrl:
        raise SystemExit(f"no recordings: stimulus {args.stimulus} -> {len(stim)}, control {args.control} -> {len(ctrl)}")
    print(f"{len(stim)} stimulus, {len(ctrl)} control, {len(nul)} null recordings; protocol {stim[0].meta.get('protocol')}; "
          f"devices {sorted({str(((r.meta.get('provenance') or {}).get('execution') or {}).get('device')) for r in stim + ctrl + nul})}")
    params = None
    if args.lif or args.receptor_model != "default" or args.receptor_table:
        params, _ = common.params_from_args(args)
    stage = None if args.stage_table in (None, "none") else args.stage_table
    dec = args.decompose_at
    if dec not in (None, "first_lost", "none"):
        dec = int(dec) if dec.lstrip("-").isdigit() else dec.split(",")
    elif dec == "none":
        dec = None
    quantity = args.quantity
    if quantity and "=" in quantity:                  # per unit kind: graded=optic_dr_abs,spiking=drive_mv_abs
        quantity = {k.strip(): v.strip() for k, v in (kv.split("=", 1) for kv in quantity.split(","))}
    res = tr.trace(c, args.source, stimulus=stim, control=ctrl, null=nul or None, params=params, stat=args.stat, depth_max=args.depth_max,
                   stage_table=stage, decompose_at=dec, quantity=quantity, min_cells=args.min_cells, min_share=args.min_share,
                   per_body=args.per_body, window=tuple(float(x) for x in args.window.split(",")) if args.window else None)
    res.files["generator"] = " ".join(sys.argv)
    path = Path(args.json) if args.json else common.default_json_path("trace", res.run_id)
    res.save(path)
    if not args.quiet:
        tr.print_trace(res, max_rows=args.max_rows)
    problems = res.check()
    if problems:
        print("CHECK: " + "; ".join(problems))
    print(f"written {path}")
    return res


# ------------------------------------------------------------------------------------------------ audit (CPU)
def cmd_audit(args):
    """The side-by-side of a trace JSON with VALIDATION['trace'] (docs/audits/interp_trace.md's table generator)."""
    res = common.Result.load(args.json)
    pt = res.table("per_type"); row = {t: r for t, r in zip(pt.type, pt.to_dict("records"))}
    ref = common.VALIDATION["trace"]["reference"]
    print(f"{args.json}: stat {res.summary.get('stat')}, {res.summary.get('n_stim_runs')} runs vs {res.summary.get('n_null_draws')} null draws; validation {res.validation.get('status')}")
    print(f"{'type':10s} {'depth':>5s} {'ref z (8.7, both modes)':>26s} {'measured z':>11s} {'stim mean':>10s} {'null mean +- sd':>18s} {'U':>4s} {'p':>7s} verdict")
    for t in ["Mi4", "Mi1", "Tm3"] + list(ref["at_null"]) + ["LPLC2", "LC10b", "LC16", "LC4"]:
        rz = ref.get(f"{t}_z") or ref["at_null"].get(t)
        rs = f"{rz[0]:+.1f} / {rz[1]:+.1f}" if rz else "--"
        if t in row:
            r = row[t]
            print(f"{t:10s} {r['depth']:5d} {rs:>26s} {r['z']:+11.1f} {r['stim_mean']:10.4f} {r['null_mean']:9.4f} +- {r['null_sd']:6.4f} {r['U']:4.0f} {r['p']:7.4f} {r['verdict']}")
        else:
            print(f"{t:10s} {'--':>5s} {rs:>26s} {'unscored':>11s}")
    for t in ("LHPD4d1", "LHAV4a1_a", "LHAV4a1_b", "LHCENT12_a", "LHPD2a1"):
        if t in row:
            r = row[t]
            print(f"{t:10s} depth {r['depth']}  stim {r['stim_level']:.2f} Hz vs ctrl {r['ctrl_level']:.2f} Hz (ref LHPD4d1 20.6 / 3.4; benchmark 17.4 / 4.4); "
                  f"diff {r['diff']:+.2f} z {r['z']:+.1f} verdict {r['verdict']}")
    print(f"first lost depth {res.summary.get('first_lost_depth')}, lost types {res.summary.get('lost_types')}, decomposed {res.summary.get('decomposed')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="GPU: one arm of one protocol -> recordings")
    r.add_argument("--protocol", required=True, choices=list(PROTOCOLS))
    r.add_argument("--arm", required=True, help="stim | ctrl | null (odour: or a site of screen_odour.SITES_APPLE)")
    r.add_argument("--out", required=True, help="output stem: <out>.npz/.json and <out>_series.npz/.json")
    r.add_argument("--seconds", type=float, default=12.0, help="scored window (object: 12 s; odour: 30 s unless given)")
    r.add_argument("--settle", type=float, default=3.0, help="object: settle before the window (ball hidden)")
    r.add_argument("--quick", action="store_true", help="smoke test: a few seconds")
    r.add_argument("--allow-cpu", action="store_true", help="do not insist on CUDA (smoke tests only; the JSON records the realised device)")
    r.add_argument("--no-series", action="store_true", help="skip the per-frame pooled series")
    r.add_argument("--series-every", type=int, default=1, help="keep every k-th scored frame of the pooled series (5 = 50 ms; the d' statistic smooths over 1 s)")
    common.add_common_args(r)
    a = sub.add_parser("analyse", help="CPU: recordings -> the trace Result")
    a.add_argument("--source", default=PHOTORECEPTORS, help="the sensory population (common.resolve grammar); default the photoreceptors")
    a.add_argument("--stimulus", nargs="+", required=True, help="glob(s) of the stimulus recordings")
    a.add_argument("--control", nargs="+", required=True)
    a.add_argument("--null-runs", nargs="+", default=None, dest="null_runs", help="glob(s) of the control-again recordings (the null arm)")
    a.add_argument("--stat", default="best_cell", choices=list(tr.STATS))
    a.add_argument("--quantity", default=None, help="the recorded quantity spiking types are scored on (rate_hz | drive_mv | ...); or per unit kind, "
                                                    "e.g. graded=optic_dr_abs,spiking=drive_mv_abs (probe_figure_stages' 'abs' measure under --stat figure_z)")
    a.add_argument("--depth-max", type=int, default=6)
    a.add_argument("--min-share", type=float, default=tr.DEFAULT_MIN_SHARE, help="type-level input share that makes a depth edge (0 = |A| > 0)")
    a.add_argument("--min-cells", type=int, default=3)
    a.add_argument("--stage-table", default=None, help="none | family | <path to the Nern 2025 Primary_cell_type_table.xlsx>")
    a.add_argument("--decompose-at", default="first_lost", help="first_lost | <depth> | type,type,... | none")
    a.add_argument("--per-body", default="lost", choices=["lost", "all", "none"])
    a.add_argument("--window", default=None, help="start_s,end_s for per-frame recordings")
    a.add_argument("--max-rows", type=int, default=80)
    common.add_common_args(a)
    u = sub.add_parser("audit", help="CPU: a trace JSON against the validation reference")
    u.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "record":
        cmd_record(args)
    elif args.cmd == "analyse":
        cmd_analyse(args)
    else:
        cmd_audit(args)


if __name__ == "__main__":
    main()
