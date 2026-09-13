"""interp_decompose -- CLI of the decompose tool (docs/INTERP.md 4.1; flyverse/interp/decompose.py).

    # GPU (cluster): record the presynaptic rates of a target over a protocol, one arm, one seed = one run
    python scripts/interp_decompose.py record --target DNp01 --protocol walk --arm holdBrain --seed 0 --out out/dec/holdBrain_r0
    # CPU: the dynamic decomposition over arms of runs (label=glob), compared with a null arm
    PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py analyse --target DNp01 --recordings "default=out/dec/default_r*.npz" \
        --recordings "holdBrain=out/dec/holdBrain_r*.npz" --null-runs "off=out/dec/off_r*.npz" --by type --window 0.5,1.5 --json out/interp/decompose/gf_walk.json
    # CPU: the static decomposition under one arm, and the structural contrast of two arms
    PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py analyse --static --target "OA-AL2i3|TmY14|DNge138|DNge149|DNge150" --arm holdBrainHis
    PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py contrast --target "OA-AL2i3|TmY14|DNge138|DNge149|DNge150" --arms default,holdBrainHis
    # the validation targets (VALIDATION['decompose']): the walk.GF_max cancellation and the 282-synapse taste dependence
    PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py validate --case taste --json out/interp/decompose/validate_taste.json
    PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py validate --case gf --dir out/dec --json out/interp/decompose/validate_gf.json

Protocols of `record` (each reproduces the benchmark section it is named after, verbatim, on the model the common
flags / --arm select; the Recorder captures every 10 ms frame):
    walk   scripts/benchmark.py sec_walk's walking phase: pinned fly on the room table, smell on, 150 frames, the
           GF maximum over frames >= 50 is walk.GF_max_hz (meta 'scalars'); needs the ray tracer -> a GPU job
    taste  sec_taste: sweet labellar-bristle / taste-peg GRNs at 100 Hz for 600 ms, no optic lobe (CPU-able)
    smell  sec_smell: the apple plume at the benchmark's coordinates for 800 ms, no optic lobe (CPU-able)
    rest   nothing driven, 500 ms
t_ms of a frame is its START time (10 k ms); every quantity is the state at its end, so `--window 0.5,1.5` on the walk
protocol is exactly the section's frames 50-149.

Arms (--arm): off | default | holdBrain | holdOptic | holdBrainGlu | holdBrainHis | holdKC | holdDN1 | <table.csv> |
as-given (the common flags alone). Hold tables are rebuilt by scripts/build_hold_tables.py when missing (out/ is not
shipped to the cluster).
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from flyverse import connectome as cn                                  # noqa: E402
from flyverse.interp import common                                     # noqa: E402
from flyverse.interp import decompose as dec                           # noqa: E402

PROTOCOLS = ("walk", "taste", "smell", "rest")
GF_ARMS = ("off", "default", "holdBrain", "holdOptic")
TASTE_TARGETS = "OA-AL2i3|TmY14|DNge138|DNge149|DNge150"


def _load(args):
    return cn.load(verbose=False) if not args.cache_dir else cn.load(cache_dir=args.cache_dir, verbose=False)


def _arm_lif(args):
    lif, op = common.params_from_args(args)
    return dec.arm_params(args.arm, lif, out_dir=os.path.join(ROOT, "out")), op


# ---------------------------------------------------------------------------------------------- record
def _protocol_walk(c, lif, op, seed, device, frames, on_frame):
    """sec_walk's walking phase, verbatim (scripts/benchmark.py): the section's GF_max is max over frames >= 50."""
    import torch
    from flyverse import brain, optic, retina, world, body, olfaction, motor
    r = retina.build_retina(c)
    w, info = world.make_room()
    w.spheres.append(world.Sphere((9, 9, 9), (0.03,) * 3, "black"))
    dirs_b, wts = r.ray_directions(); wts_t = torch.from_numpy(wts).float().to(w.device)
    rs = brain._receptor(c, lif, with_counts=lif.receptor_model == "full")
    ol = optic.OpticLobe(c, r, op, device=device, receptor=rs, receptor_gain=brain._receptor_gain(lif)); ol.relax()
    b = brain.Brain(c, lif, device=device, seed=seed, receptor=rs); b.freeze(ol.rate_idx)
    fly = body.FlyState(x=-0.3, y=0.0, z=info["table_top_z"], heading=0.0)

    def col_rad():
        d = fly.body_to_world(dirs_b.reshape(-1, 3)); o = np.broadcast_to(fly.eye_pos, d.shape)
        rad = w.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        return (rad.reshape(dirs_b.shape[0], dirs_b.shape[1], 4) * wts_t[None, :, None]).sum(1)

    gf = c.select(type="DNp01"); wg = motor.wing_groups(c)
    olf_walk = olfaction.Olfaction(c, [(name, cen, 1.0) for name, cen, rad in info["fruit"]])
    gf_walk, pw_walk = [], []
    fb = SimpleNamespace(brain=b, optic=ol, c=c, cuda_graphs=False)
    groups = motor.motor_groups(c)
    for k in range(frames):
        fly.x += 0.004 * 0.01; olf_walk.apply(b, fly.eye_pos); b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
        if k >= 50:
            gf_walk.append(float(b.rate[0, b._idx(gf)].mean())); pw_walk.append(float(b.rate[0, b._idx(wg.power)].mean()))
        on_frame(fb, k, motor.read_motor(b, groups, wg))
    scalars = {"walk.GF_max_hz": float(np.max(gf_walk)) if gf_walk else None, "walk.GF_mean_hz": float(np.mean(gf_walk)) if gf_walk else None,
               "walk.power_max_hz": float(np.max(pw_walk)) if pw_walk else None,
               "walk.power_sustained_hz": float(np.max(np.convolve(pw_walk, np.ones(30) / 30, mode="valid"))) if len(pw_walk) >= 30 else None,
               "frac_active": float((b.rate_np() > 1).mean())}
    stim = {"protocol": "walk", "params": {"frames": frames, "frame_ms": 10.0, "fly": {"x0": -0.3, "y": 0.0, "heading": 0.0, "dx_per_frame_m": 0.004 * 0.01},
                                            "smell": "room fruit plumes (olfaction.Olfaction on world.make_room()['fruit'])", "gf_window_frames": [50, frames - 1],
                                            "loom": "sphere parked at (9, 9, 9): the walking phase only"}, "control": "arm off (receptor_model None)"}
    return fb, scalars, stim, {"file": None, "n_columns": int(r.n_columns), "column_to_bodies": "retina.build_retina(c) (pr_index / pr_column; not serialised here)"}


def _protocol_taste(c, lif, op, seed, device, frames, on_frame):
    from flyverse import brain, motor
    n = c.neurons
    taste = pd.read_csv(os.path.join(ROOT, "flyverse", "data", "taste_grns.csv"))
    sweet = c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
    sweet = sweet[np.isin(n.subclass.to_numpy()[sweet], ["labellar bristle", "taste peg"])]
    b = brain.Brain(c, lif, device=device, seed=seed); b.set_poisson(sweet, 100.0)
    fb = SimpleNamespace(brain=b, optic=None, c=c, cuda_graphs=False)
    groups, wg = motor.motor_groups(c), motor.wing_groups(c)
    for k in range(frames):
        b.step(20); on_frame(fb, k, motor.read_motor(b, groups, wg))
    rt = b.rate_np()
    scalars = {"taste.MN9_hz": float(rt[c.select(type="MN9")].mean()), "taste.GNG175_hz": float(rt[c.select(type="GNG175")].mean()),
               "frac_active": float((rt > 1).mean())}
    stim = {"protocol": "taste", "params": {"frames": frames, "frame_ms": 10.0, "grns": "sweet labellar bristle + taste peg (flyverse/data/taste_grns.csv)",
                                             "n_grn": int(len(sweet)), "hz": 100.0}, "control": "arm off / holdBrainHis (receptor_integration.md E.4)"}
    return fb, scalars, stim, None


def _protocol_smell(c, lif, op, seed, device, frames, on_frame):
    from flyverse import brain, motor, olfaction
    olf = olfaction.Olfaction(c, [("apple", (0.25, 0.15, 0.79), 1.0)])
    b = brain.Brain(c, lif, device=device, seed=seed); olf.apply(b, (0.19, 0.15, 0.75))
    fb = SimpleNamespace(brain=b, optic=None, c=c, cuda_graphs=False)
    groups, wg = motor.motor_groups(c), motor.wing_groups(c)
    for k in range(frames):
        b.step(20); on_frame(fb, k, motor.read_motor(b, groups, wg))
    rt = b.rate_np()
    pn = c.select(type="~_l2PN|_adPN|_lPN|_lvPN|_ilPN"); kc = c.select(type="~^KC")
    scalars = {"smell.PN_hz": float(rt[pn].mean()), "smell.KC_hz": float(rt[kc].mean()), "smell.KC_active": int((rt[kc] > 1).sum()),
               "frac_active": float((rt > 1).mean())}
    stim = {"protocol": "smell", "params": {"frames": frames, "frame_ms": 10.0, "source": ["apple", [0.25, 0.15, 0.79], 1.0], "fly": [0.19, 0.15, 0.75]},
            "control": "arm off"}
    return fb, scalars, stim, None


def _protocol_rest(c, lif, op, seed, device, frames, on_frame):
    from flyverse import brain, motor
    b = brain.Brain(c, lif, device=device, seed=seed)
    fb = SimpleNamespace(brain=b, optic=None, c=c, cuda_graphs=False)
    groups, wg = motor.motor_groups(c), motor.wing_groups(c)
    for k in range(frames):
        b.step(20); on_frame(fb, k, motor.read_motor(b, groups, wg))
    return fb, {"rest.spikes_per_step": float(b.total_spikes()), "frac_active": float((b.rate_np() > 1).mean())}, \
        {"protocol": "rest", "params": {"frames": frames, "frame_ms": 10.0}, "control": None}, None


PROTOCOL_FN = {"walk": (_protocol_walk, 150), "taste": (_protocol_taste, 60), "smell": (_protocol_smell, 80), "rest": (_protocol_rest, 50)}


def cmd_record(args) -> int:
    t0 = time.time()
    c = _load(args)
    lif, op = _arm_lif(args)
    fn, default_frames = PROTOCOL_FN[args.protocol]
    frames = args.frames or default_frames
    target_idx = common.resolve(c, args.target)
    if len(target_idx) == 0:
        raise SystemExit(f"--target {args.target!r} selects no cell")
    pre_idx = np.unique(c.W.tocsr()[target_idx].tocoo().col)
    sel = np.union1d(target_idx, pre_idx)
    quantities = tuple(args.quantities.split(","))
    rec = common.Recorder(c, sel, quantities=quantities)
    arm = dec.arm_of(lif)
    print(f"record {args.protocol}: target {args.target} ({len(target_idx)} cells), {len(pre_idx)} presynaptic cells, {len(sel)} recorded; "
          f"arm {arm} (receptor_model {lif.receptor_model!r}, table {lif.receptor_table}); seed {args.seed}; device request {args.device}", flush=True)

    def on_frame(fb, k, mot):
        rec.capture(fb, t_ms=10.0 * k, motor=mot)

    fb, scalars, stim, retina_rec = fn(c, lif, op, args.seed, args.device, frames, on_frame)
    prov = common.provenance(c, lif, op, fb=fb, device=args.device, seeds=[args.seed], stimulus=stim, retina=retina_rec, cache_dir=args.cache_dir)
    rs_changed = None
    if lif.receptor_model is not None:
        rs_changed = int((fb.brain.receptor.fast_sign != np.sign(c.W.data)).sum())
    meta = {"protocol": args.protocol, "arm": arm, "seed": int(args.seed), "target": args.target, "target_idx": target_idx.tolist(),
            "n_presynaptic": int(len(pre_idx)), "scalars": scalars, "stimulus": stim, "provenance": prov, "t_convention": "t_ms = 10 k is the frame's start; state at its end",
            "receptor_changed_entries": rs_changed, "shaped_md5": common._md5_csr(fb.brain._W_cpu) if hasattr(fb.brain, "_W_cpu") else None,
            "generator": "scripts/interp_decompose.py record " + " ".join(sys.argv[2:]), "wall_s": round(time.time() - t0, 1)}
    R = rec.finish(meta=meta)
    out = Path(args.out)
    meta["file"] = str(out.with_suffix(".npz"))
    R.meta["file"] = meta["file"]
    path = R.save(out)
    print(f"device {prov['execution']['device']}  frames {R.n_frames}  scalars {json.dumps(scalars)}  receptor changed entries {rs_changed}  "
          f"W md5 {meta['shaped_md5']}  wall {meta['wall_s']} s\nwrote {path}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- analyse / contrast
def _arms_from_items(items) -> dict:
    """['default=out/x_r*.npz', 'out/y_r*.npz'] -> {label: glob}."""
    out = {}
    for it in items or []:
        if "=" in it and not os.path.exists(it):
            k, v = it.split("=", 1)
        else:
            k, v = f"arm{len(out)}", it
        out[k.strip()] = v.strip()
    return out


def _window(s):
    if not s:
        return None
    a, b = (float(x) for x in s.split(","))
    return (a, b)


def cmd_analyse(args) -> int:
    c = _load(args)
    lif, op = _arm_lif(args)
    by = tuple(x.strip() for x in args.by.split(",") if x.strip())
    if args.static:
        res = dec.decompose(c, args.target, params=lif, optic_params=op, by=by, tiers=not args.no_tiers, top=args.top, arm_label=dec.arm_of(lif))
    else:
        arms = _arms_from_items(args.recordings)
        if not arms:
            raise SystemExit("analyse needs --recordings label=glob (or --static)")
        null = _arms_from_items(args.null_runs) if args.null_runs else None
        res = dec.decompose(c, args.target, recording=arms, null_recording=null, params=lif, optic_params=op, by=by, tiers=not args.no_tiers,
                            window=_window(args.window), top=args.top)
    res.files["generator"] = "scripts/interp_decompose.py " + " ".join(sys.argv[1:])
    if not args.quiet:
        dec.print_per_type(res, top=args.top)
        s = res.summary
        for t, v in (s.get("dynamic", {}).get(next(iter(s.get("arms", {"x": 0}))), s.get("static", {})) if "dynamic" in s else s.get("static", {})).items():
            print(f"{t}: E {v['E_total']:+.4g}  I {v['I_total']:+.4g}  net {v['net']:+.4g}  cancelling pair {v['cancelling_pair']}")
    problems = res.check()
    if problems:
        print("CHECK:", problems)
    path = Path(args.json) if args.json else common.default_json_path("decompose", res.run_id)
    res.save(path)
    print("wrote", path)
    return 0


def cmd_contrast(args) -> int:
    c = _load(args)
    base, op = common.params_from_args(args)
    arms = [a.strip() for a in args.arms.split(",")]
    if len(arms) != 2:
        raise SystemExit("--arms needs exactly two arm names, e.g. default,holdBrainHis")
    pa, pb = dec.arm_params(arms[0], base, out_dir=os.path.join(ROOT, "out")), dec.arm_params(arms[1], base, out_dir=os.path.join(ROOT, "out"))
    by = tuple(x.strip() for x in args.by.split(",") if x.strip())
    res = dec.contrast(c, args.target, pa, pb, labels=tuple(arms), by=by, optic_params=op)
    res.files["generator"] = "scripts/interp_decompose.py " + " ".join(sys.argv[1:])
    if not args.quiet:
        _print_contrast(res, args.target, args.top)
    path = Path(args.json) if args.json else common.default_json_path("decompose", res.run_id)
    res.save(path)
    print("wrote", path)
    return 0


def _print_contrast(res, target, top) -> None:
    s = res.summary
    print(f"contrast {s['labels'][0]} -> {s['labels'][1]} on {target}: of {s['entries_total']} stored entries {s['entries_sign_changed']} change sign / gain "
          f"({s['synapses_sign_changed']:.0f} raw synapses) and {s['entries_rescaled_only']} are rescaled only (fan-in scale of "
          f"{s['target_cells_with_moved_fanin_scale']} target cells moved; sum |delta| {s['abs_delta_rescaled_mv']:.3f} mV); "
          f"by transmitter {s['by_transmitter']}; pre types {s['pre_types_changed']}")
    print("-- sign / gain changes by (post type, pre group, transmitter, tier, sign a -> b) --")
    common.print_table(res.table("delta_per_type"), max_rows=top)
    print("-- by post type --")
    common.print_table(res.table("delta_by_post_type"), max_rows=top)
    if len(res.table("rescaled_cells")):
        print("-- target cells whose fan-in scale moved (tot = shaped |W| row sum; every input entry of the cell is rescaled) --")
        common.print_table(res.table("rescaled_cells"), floatfmt="{:+.6f}", max_rows=top)


# ---------------------------------------------------------------------------------------------- validate
def _glob_arms(d, arms, pattern="{arm}_r*.npz") -> dict:
    out = {}
    for a in arms:
        files = sorted(globmod.glob(os.path.join(d, pattern.format(arm=a))))
        if files:
            out[a] = files
    return out


def _validate_taste(args, c, base, op, ref) -> int:
    """(b) the 282-synapse taste dependence. Static: the contrast default -> holdBrainHis over the whole brain (the 123
    entries / 282 synapses of E.4, and E.3's seven rescaled cells), on the five named targets and on MN9 (0 entries).
    Dynamic (recordings in --dir, CPU taste protocol): MN9's and the targets' rate-weighted input under default / off
    with holdBrainHis as the matched control -- which presynaptic groups differ, whether the histamine group carries
    any current in the protocol at all, and whether holdBrainHis is bit-identical to off."""
    out_dir = os.path.join(ROOT, "out")
    pa, pb = dec.arm_params("default", base, out_dir=out_dir), dec.arm_params("holdBrainHis", base, out_dir=out_dir)
    targets = args.target or TASTE_TARGETS
    labels = ("default", "holdBrainHis")
    res = dec.contrast(c, targets, pa, pb, labels=labels, by=("type",), optic_params=op, top=args.top)
    mn9 = dec.contrast(c, "MN9", pa, pb, labels=labels, by=("type",), optic_params=op, top=args.top)
    whole = dec.contrast(c, np.arange(c.n), pa, pb, labels=labels, by=("type",), optic_params=op, top=args.top)
    keys = ("entries_total", "entries_sign_changed", "synapses_sign_changed", "entries_rescaled_only", "abs_delta_rescaled_mv", "by_transmitter",
            "by_post_type", "pre_types_changed", "target_cells_with_moved_fanin_scale", "rescaled_cells")
    measured = {"targets": targets, "static": {"targets": {k: res.summary[k] for k in keys}, "whole_brain": {k: whole.summary[k] for k in keys},
                                               "MN9": {k: mn9.summary[k] for k in keys}}}
    tc = ref["taste_carrier"]
    w = whole.summary
    ok_static = (w["entries_sign_changed"] == tc["entries"] and abs(w["synapses_sign_changed"] - tc["synapses"]) < 0.5
                 and set(tc["pre"]) <= set(w["pre_types_changed"]) and all(k.startswith(tc["transmitter"]) for k in w["by_transmitter"])
                 and all(t in w["by_post_type"] for t in tc["targets"]) and mn9.summary["entries_sign_changed"] == 0)
    res.add_table("whole_brain_delta_per_type", whole.table("delta_per_type"))
    res.add_table("whole_brain_delta_by_post_type", whole.table("delta_by_post_type"))
    res.add_table("whole_brain_rescaled_cells", whole.table("rescaled_cells"))
    if not args.quiet:
        print("== static: whole brain =="); _print_contrast(whole, "every cell", args.top)
        print("\n== static: the five named targets =="); _print_contrast(res, targets, args.top)
        print(f"\n== static: MN9 == sign-changed entries {mn9.summary['entries_sign_changed']}, rescaled-only {mn9.summary['entries_rescaled_only']}, "
              f"fan-in scale moved on {mn9.summary['target_cells_with_moved_fanin_scale']} cell(s)")
    # the dynamic half: recordings of the taste protocol (default / off as arms, holdBrainHis as the matched control)
    arms = _glob_arms(args.dir, ("default", "off", "holdBrainGlu", "holdBrain", "holdOptic"))
    null_files = _glob_arms(args.dir, ("holdBrainHis",))
    if arms and null_files:
        null = {"holdBrainHis": null_files["holdBrainHis"]}
        window = _window(args.window)
        dyn = dec.decompose(c, "MN9", recording=arms, null_recording=null, params=pa, optic_params=op, by=("type",), tiers=False, window=window, top=args.top)
        dyn_t = dec.decompose(c, targets, recording=arms, null_recording=null, params=pa, optic_params=op, by=("type",), tiers=True, window=window, top=args.top)
        dyn_nt = dec.decompose(c, targets, recording=arms, null_recording=null, params=pa, optic_params=op, by=("transmitter",), tiers=False, window=window, top=args.top)
        recs = {a: [common.Recording.load(f) for f in fs] for a, fs in {**arms, **null}.items()}
        scal = {a: [r.meta.get("scalars", {}) for r in rs] for a, rs in recs.items()}
        d = {"taste.MN9_hz_by_arm": {a: [x.get("taste.MN9_hz") for x in v] for a, v in scal.items()},
             "taste.GNG175_hz_by_arm": {a: [x.get("taste.GNG175_hz") for x in v] for a, v in scal.items()},
             "seeds_by_arm": {a: [r.meta.get("seed") for r in rs] for a, rs in recs.items()},
             "devices": {a: sorted(set((r.meta.get("provenance") or {}).get("execution", {}).get("device") for r in rs)) for a, rs in recs.items()},
             "shaped_md5": {a: sorted(set(r.meta.get("shaped_md5") for r in rs)) for a, rs in recs.items()},
             "holdBrainHis_equals_off": ("off" in scal and [x.get("taste.MN9_hz") for x in scal["off"]] == [x.get("taste.MN9_hz") for x in scal["holdBrainHis"]]
                                        and [x.get("taste.GNG175_hz") for x in scal["off"]] == [x.get("taste.GNG175_hz") for x in scal["holdBrainHis"]])}
        d["taste.MN9_hz_default_vs_holdBrainHis"] = common.compare(d["taste.MN9_hz_by_arm"].get("default", []), d["taste.MN9_hz_by_arm"]["holdBrainHis"])
        wm = dyn.table("per_type")
        d["MN9_groups_z_ge_3_default_vs_holdBrainHis"] = wm[wm.default_z.abs() >= 3][["pre_group", "default_mean", "holdBrainHis_mean", "default_z", "default_verdict"]].to_dict("records") if "default_z" in wm else []
        d["MN9_verdict_counts"] = wm.default_verdict.value_counts().to_dict() if "default_verdict" in wm else {}
        d["MN9_totals"] = dyn.summary.get("dynamic")
        wn = dyn_nt.table("per_type")
        his = wn[wn.pre_group == "histamine"]
        d["targets_histamine_group"] = his[[c_ for c_ in wn.columns if c_.endswith("_mean") or c_.endswith("_rate_hz") or c_.endswith("_z") or c_.endswith("_verdict") or c_ in ("post_type", "pre_group", "weight_mv_per_volley", "raw_count")]].to_dict("records")
        # do the presynaptic histamine cells fire at all in the protocol?
        rec0 = recs["default"][0]
        his_pre = np.isin(rec0.types, np.array(tc["pre"] + ["R7y", "R7p", "R7_unclear", "R8d", "T1", "AN27X004", "AN27X008", "GNG043", "IN27X004"]))
        d["histamine_presynaptic_cells_recorded"] = int(his_pre.sum())
        d["histamine_presynaptic_max_rate_hz"] = float(rec0.quantities["rate_hz"][:, his_pre].max()) if his_pre.any() else None
        d["coverage"] = {"MN9": dyn.summary.get("coverage", {}).get("default[0]"), "targets": dyn_t.summary.get("coverage", {}).get("default[0]")}
        d["target_totals"] = dyn_t.summary.get("dynamic")
        measured["dynamic"] = d
        if args.json:
            for r_, suffix in ((dyn, "_MN9_dynamic"), (dyn_t, "_targets_dynamic"), (dyn_nt, "_targets_by_transmitter")):
                p_ = Path(args.json).with_name(Path(args.json).stem + suffix + ".json"); r_.save(p_); measured["dynamic"][suffix.strip("_") + "_json"] = str(p_)
        if not args.quiet:
            print("\n== dynamic: taste protocol, arms = runs (seeds), null = holdBrainHis ==")
            print("taste.MN9_hz by arm:", json.dumps(d["taste.MN9_hz_by_arm"]), " holdBrainHis == off (MN9 and GNG175, every seed):", d["holdBrainHis_equals_off"])
            print("default vs holdBrainHis on taste.MN9_hz:", {k: d["taste.MN9_hz_default_vs_holdBrainHis"][k] for k in ("diff", "z", "welch", "U", "p", "verdict")})
            print(f"histamine presynaptic cells recorded {d['histamine_presynaptic_cells_recorded']}, their max rate over the protocol {d['histamine_presynaptic_max_rate_hz']} Hz")
            print("\n-- MN9 input by presynaptic type (mV/s per MN9 cell; window mean; z against holdBrainHis) --")
            dec.print_per_type(dyn, top=args.top)
            print("\n-- the five targets' input by presynaptic transmitter --")
            dec.print_per_type(dyn_nt, top=args.top)
            print("\n-- the five targets' input by presynaptic type (top groups) --")
            dec.print_per_type(dyn_t, top=min(args.top, 12))
    else:
        measured["dynamic"] = f"no recordings in {args.dir} (need default_r*.npz and holdBrainHis_r*.npz from `record --protocol taste`)"
    res.validation["measured"] = measured; res.validation["status"] = "reproduced" if ok_static else "not reproduced"
    res.files["generator"] = "scripts/interp_decompose.py " + " ".join(sys.argv[1:])
    if not args.quiet:
        print(f"\nvalidation (static, whole brain): {res.validation['status']} -- measured {w['entries_sign_changed']} entries / {w['synapses_sign_changed']:.0f} synapses, "
              f"{w['by_transmitter']}, targets {list(w['by_post_type'])[:8]}...; reference {tc}")
    path = Path(args.json) if args.json else common.default_json_path("decompose", res.run_id)
    res.save(path); print("wrote", path)
    return 0


def cmd_validate(args) -> int:
    c = _load(args)
    base, op = common.params_from_args(args)
    ref = common.VALIDATION["decompose"]["reference"]
    measured, notes = {}, []
    if args.case == "taste":
        return _validate_taste(args, c, base, op, ref)
    # (a) the walk.GF_max cancellation over the four arms of recordings in --dir
    arms = _glob_arms(args.dir, GF_ARMS)
    missing = [a for a in GF_ARMS if a not in arms]
    if missing:
        raise SystemExit(f"--dir {args.dir}: no recordings for arms {missing} (expected <arm>_r*.npz)")
    null = {"off": arms.pop("off")}
    ordered = {k: arms[k] for k in ("default", "holdBrain", "holdOptic") if k in arms}
    res = dec.decompose(c, args.target or "DNp01", recording=ordered, null_recording=null, params=dec.arm_params("default", base), optic_params=op,
                        by=tuple(args.by.split(",")), tiers=not args.no_tiers, window=_window(args.window or "0.5,1.5"), top=args.top)
    recs = {a: [common.Recording.load(f) for f in fs] for a, fs in {**null, **ordered}.items()}
    gf = {a: [r.meta.get("scalars", {}).get("walk.GF_max_hz") for r in rs] for a, rs in recs.items()}
    gf_from_rec = {}
    for a, rs in recs.items():
        vals = []
        for r in rs:
            w = r.window(*_window(args.window or "0.5,1.5"))
            tpos = np.isin(w.idx, common.resolve(c, args.target or "DNp01"))
            vals.append(float(w.quantities["rate_hz"][:, tpos].mean(axis=1).max()))
        gf_from_rec[a] = vals
    devices = {a: [(r.meta.get("provenance") or {}).get("execution", {}).get("device") for r in rs] for a, rs in recs.items()}
    md5s = {a: sorted(set(r.meta.get("shaped_md5") for r in rs)) for a, rs in recs.items()}
    # the scatter rule on the check itself: each arm's GF maxima over runs against the off arm's (common.compare)
    gf_compare = {a: common.compare(gf[a], gf["off"]) for a in ("default", "holdBrain", "holdOptic") if a in gf}
    measured = {"walk.GF_max_hz": gf, "walk.GF_max_hz_from_recording": gf_from_rec, "seeds": {a: [r.meta.get("seed") for r in rs] for a, rs in recs.items()},
                "devices": devices, "shaped_md5": md5s, "receptor_changed_entries": {a: sorted(set(r.meta.get("receptor_changed_entries") for r in rs)) for a, rs in recs.items()},
                "walk.GF_max_hz_arm_vs_off": gf_compare,
                "replicated_over_runs": bool(any(v["verdict"] == "result" for v in gf_compare.values()))}
    # the cancellation table: per group, arm means and deltas against off
    wide = res.table("per_type")
    for a in ("default", "holdBrain", "holdOptic"):
        wide[f"d_{a}_vs_off"] = wide[f"{a}_mean"] - wide["off_mean"]
    wide["d_sum_halves"] = wide["d_holdBrain_vs_off"] + wide["d_holdOptic_vs_off"]
    wide["nonadditivity"] = wide["d_default_vs_off"] - wide["d_sum_halves"]
    res.add_table("cancellation", wide[["post_type", "pre_group", "off_mean", "default_mean", "holdBrain_mean", "holdOptic_mean", "d_default_vs_off",
                                        "d_holdBrain_vs_off", "d_holdOptic_vs_off", "d_sum_halves", "nonadditivity", "weight_mv_per_volley", "raw_count", "sign_rule",
                                        "off_sd", "default_sd", "holdBrain_sd", "holdOptic_sd"] + [c_ for c_ in wide.columns if c_.endswith("_verdict") or c_.endswith("_z")]])
    totals = {}
    for a in ("off", "default", "holdBrain", "holdOptic"):
        v = wide[wide.pre_group != "optic_drive"][f"{a}_mean"]
        vp = wide[wide.pre_group != "optic_drive"][f"{a}_value_at_peak"] if f"{a}_value_at_peak" in wide else None
        totals[a] = {"E": float(v[v > 0].sum()), "I": float(v[v < 0].sum()), "net": float(v.sum()),
                     "E_at_peak": float(vp[vp > 0].sum()) if vp is not None else None, "I_at_peak": float(vp[vp < 0].sum()) if vp is not None else None,
                     "optic_drive_mv": float(wide[wide.pre_group == "optic_drive"][f"{a}_g_mv"].mean()) if (wide.pre_group == "optic_drive").any() and f"{a}_g_mv" in wide else None}
    measured["input_totals_mV_per_s"] = totals
    # the groups that differ from off in every arm (|z| >= 3 in all three) and the ones that differ in one half only
    zc = [f"{a}_z" for a in ("default", "holdBrain", "holdOptic")]
    z = wide[zc].fillna(0.0)
    measured["groups_differing_in_every_arm"] = wide.pre_group[(z.abs() >= 3).all(axis=1)].tolist()
    measured["groups_differing_in_one_half_only"] = {
        "holdBrain_only": wide.pre_group[(z.holdBrain_z.abs() >= 3) & (z.holdOptic_z.abs() < 3) & (z.default_z.abs() < 3)].tolist(),
        "holdOptic_only": wide.pre_group[(z.holdOptic_z.abs() >= 3) & (z.holdBrain_z.abs() < 3) & (z.default_z.abs() < 3)].tolist()}
    measured["verdict_counts"] = {a: wide[f"{a}_verdict"].value_counts().to_dict() for a in ("default", "holdBrain", "holdOptic")}
    measured["summary_dynamic"] = res.summary.get("dynamic")
    # the reference check: at seed 0 the four GF maxima are the audit's to 3 decimals (deterministic section)
    ok = True
    for a, r_ in ref["walk.GF_max_hz"].items():
        vals = [v for v, s_ in zip(gf.get(a, []), measured["seeds"].get(a, [])) if s_ == 0 and v is not None]
        if not vals or abs(vals[0] - r_) > 5e-3:
            ok = False
    res.validation["measured"] = measured; res.validation["status"] = "reproduced" if ok else "not reproduced"
    res.files["generator"] = "scripts/interp_decompose.py " + " ".join(sys.argv[1:])
    if not args.quiet:
        print("walk.GF_max_hz per arm (meta scalars, per run):", json.dumps(gf))
        print("  from the recording (max over frames of the DNp01 mean rate in the window):", json.dumps(gf_from_rec))
        print("  reference (receptor_integration.md G.4, seed 0):", ref["walk.GF_max_hz"], "->", res.validation["status"])
        print("  each arm vs off over runs (common.compare):", json.dumps({a: {k: v[k] for k in ("diff", "z", "welch", "U", "p", "verdict")} for a, v in gf_compare.items()}))
        print("  devices:", devices, " shaped md5:", md5s)
        print("input totals (mV/s per DNp01 cell, groups summed; window mean and at the frame of the DNp01 peak):", json.dumps(totals, indent=None))
        print("groups with |z| >= 3 vs off in every arm:", measured["groups_differing_in_every_arm"], "; in one half only:", measured["groups_differing_in_one_half_only"])
        cols = ["post_type", "pre_group", "off_mean", "default_mean", "holdBrain_mean", "holdOptic_mean", "d_default_vs_off", "d_holdBrain_vs_off", "d_holdOptic_vs_off",
                "nonadditivity", "weight_mv_per_volley", "raw_count", "sign_rule"]
        t = res.table("cancellation")
        print("\n-- groups by |delta holdBrain - off| + |delta holdOptic - off| (window means over 3 runs per arm) --")
        common.print_table(t.reindex((t.d_holdBrain_vs_off.abs() + t.d_holdOptic_vs_off.abs()).sort_values(ascending=False).index)[cols], max_rows=args.top)
        print("\n-- groups by |off mean| --")
        common.print_table(t.reindex(t.off_mean.abs().sort_values(ascending=False).index)[cols], max_rows=args.top)
        pk = [f"{a}_value_at_peak" for a in ("off", "default", "holdBrain", "holdOptic") if f"{a}_value_at_peak" in wide]
        if pk:
            print("\n-- input at the frame of the DNp01 peak (mean over runs of each run's peak-frame value; the frame walk.GF_max reads) --")
            common.print_table(wide.reindex(wide[pk].abs().max(axis=1).sort_values(ascending=False).index)[["post_type", "pre_group"] + pk +
                                                                                                          [f"{a}_output_peak_hz" for a in ("off", "default", "holdBrain", "holdOptic") if f"{a}_output_peak_hz" in wide]], max_rows=args.top)
    path = Path(args.json) if args.json else common.default_json_path("decompose", res.run_id)
    res.save(path); print("wrote", path)
    return 0


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="GPU / CPU: run a protocol and record the target's presynaptic rates (one arm, one seed = one run)")
    r.add_argument("--target", required=True); r.add_argument("--protocol", required=True, choices=PROTOCOLS)
    r.add_argument("--arm", default="as-given", help="off | default | hold<Group> | <table.csv> | as-given")
    r.add_argument("--out", required=True, help="recording path without suffix (.npz + .json)")
    r.add_argument("--frames", type=int, default=None, help="override the protocol's frame count")
    r.add_argument("--quantities", default="rate_hz,drive_mv", help="Recorder quantities (comma-separated)")
    a = sub.add_parser("analyse", help="CPU: the decomposition (static, or over recordings)")
    a.add_argument("--target", required=True); a.add_argument("--static", action="store_true")
    a.add_argument("--arm", default="as-given", help="static: the arm whose weights to decompose")
    a.add_argument("--recordings", action="append", default=[], metavar="LABEL=GLOB", help="an arm of runs (repeatable)")
    a.add_argument("--by", default="type", help="comma-separated subset of " + ",".join(dec.GROUPINGS))
    a.add_argument("--no-tiers", action="store_true"); a.add_argument("--window", default=None, help="start_s,end_s")
    a.add_argument("--top", type=int, default=40)
    k = sub.add_parser("contrast", help="CPU: the structural difference between two arms on the target's input")
    k.add_argument("--target", required=True); k.add_argument("--arms", required=True, help="two arm names, e.g. default,holdBrainHis")
    k.add_argument("--by", default="type"); k.add_argument("--top", type=int, default=40)
    v = sub.add_parser("validate", help="the VALIDATION['decompose'] targets: --case gf (walk recordings in --dir) | taste (CPU)")
    v.add_argument("--case", required=True, choices=["gf", "taste"]); v.add_argument("--dir", default=os.path.join(ROOT, "out", "dec"))
    v.add_argument("--target", default=None); v.add_argument("--by", default="type"); v.add_argument("--no-tiers", action="store_true")
    v.add_argument("--window", default=None); v.add_argument("--top", type=int, default=30)
    v.add_argument("--whole-brain", action="store_true", help="taste: also count the changed entries over every cell (slow: ~1 min)")
    for p in (r, a, k, v):
        common.add_common_args(p)
    args = ap.parse_args()
    return {"record": cmd_record, "analyse": cmd_analyse, "contrast": cmd_contrast, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
