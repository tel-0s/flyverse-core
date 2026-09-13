#!/usr/bin/env python
"""The health tool's CLI (docs/INTERP.md 4.6 / 5): record on the GPU, analyse on the CPU.

    # GPU (cluster): the two validation protocols, one job per run
    python scripts/interp_health.py record --protocol compass --gE 2 --gD 15 --receptor-model off --seed 0 --out out/health/compass_r0
    python scripts/interp_health.py record --protocol walk --receptor-model default --seed 0 --out out/health/walk_default_r0
    python scripts/interp_health.py record --protocol walk --receptor-model off     --seed 0 --out out/health/walk_off_r0
    # CPU
    PYTHONIOENCODING=utf-8 python scripts/interp_health.py analyse --recordings "out/health/compass_r*" --window 5.5,8 --groups meta --json out/interp/health/compass.json
    PYTHONIOENCODING=utf-8 python scripts/interp_health.py analyse --recordings "out/health/walk_default_r*" --null-runs "out/health/walk_off_r*" --window 0.5,1.5 --groups meta --json out/interp/health/walk.json
    PYTHONIOENCODING=utf-8 python scripts/interp_health.py structure --by module --json out/interp/health/nt_structure_module.json
    PYTHONIOENCODING=utf-8 python scripts/interp_health.py validate --compass "out/health/compass_r*" --walk-default "out/health/walk_default_r*" --walk-off "out/health/walk_off_r*" --json out/interp/health/validation.json

Protocols (`record --protocol`):
  compass   scripts/cx_wedge.py simulate's ring protocol as the six-seed grid ran it (docs/audits/cx_glno.md 5): FlyBrain on the
            full connectome, compass adaptation 0, 10 Hz Poisson background on every EPG, 1 s settle, wedges `--start-wedge` ..
            +`--width` at +`--pulse-hz` for `--pulse-s`, then `--seconds` free; Delta7 gain on Delta7 -> EPG only, ER/ExR x1,
            receptor model as given (the grid: off). Records rate / v / adapt / refrac / spike_count of every EPG, EPGt, PEN,
            PEG, Delta7, ring (ER / ExR) and GLNO cell every 10 ms frame, plus rate_hz of all their presynaptic cells
            (<stem>_presyn), and cx_wedge's in / out / PEN / Delta7 / GLNO / vector-strength samples at its marks (meta).
  walk      scripts/benchmark.py sec_walk's walking + loom phases, step for step (Brain + OpticLobe, eager, smell on, the fly
            walking forward at 4 mm/s for 1.5 s, then a loom from the front over 0.8 s): records every descending neuron,
            DNp01, the wing-power MNs and the leg MNs (all LIF quantities) plus rate_hz of their presynaptic cells; meta
            carries the section's own walk.GF_max_hz / power_max_hz / loom.GF_peak_hz from the same run.

Every recording's json sidecar carries the provenance block (realised device, resolved LIFParams / OpticParams, fingerprint,
seeds, stimulus) so `analyse` on another machine is self-describing. The replicate unit is the job: submit >= 3 seeds per
arm in ONE cluster batch and fetch a named directory (docs/INTERP.md section 5).
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from flyverse.interp import common  # noqa: E402
from flyverse.interp.common import Recorder, Recording, Result  # noqa: E402

# cx_wedge.py's constants (scripts/cx_wedge.py COMPASS_RE / RING_RE), copied so this CLI imports nothing heavy at import time
COMPASS_RE = r"^(EPG|PEN|PEG|Delta7)"
RING_RE = r"^(ER|ExR)"
COMPASS_TARGET = ["EPG", "EPGt", "~^PEN_", "PEG", "Delta7", "~^(ER|ExR)", "GLNO"]


def parse_window(text):
    if text in (None, ""):
        return None
    a, b = (float(x) for x in str(text).split(","))
    return (a, b)


def compass_type_path_gain(gE: float, gD: float) -> list:
    """cx_wedge.simulate's type_path_gain with delta7_pen=False and gR=1 (= scripts/probe_compass_room.compass_type_path_gain):
    the shipped LC4 / LPLC2 -> DNp01 x3, EPG <-> PEN / PEG x gE, Delta7 -> EPG x gD, ER / ExR -> EPG / PEN / PEG x1."""
    from flyverse import brain
    return list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", float(gE)), (r"^PEN_", r"^EPG$", float(gE)),
                                                 (r"^EPG$", r"^PEG$", float(gE)), (r"^PEG$", r"^EPG$", float(gE)),
                                                 (r"^Delta7$", r"^EPG$", float(gD)), (RING_RE, r"^(EPG$|PEN_|PEG$)", 1.0)]


def load_connectome(args):
    from flyverse import connectome
    if getattr(args, "cache_dir", None):
        return connectome.load(cache_dir=Path(args.cache_dir), verbose=False)
    return connectome.load(verbose=False)


def _device_ok(args):
    """Refuse to start a GPU protocol on the CPU unless --allow-cpu (a 'device cpu' log means resubmission)."""
    import torch
    if args.device == "cpu" or not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit("record needs a GPU (torch.cuda.is_available() is False); pass --allow-cpu for a smoke test")
        return False
    return True


# ---------------------------------------------------------------------------------------------- record: compass
def _compass_sample(fb, idx_epg, inside, wedge_of, pen_idx, d7_idx, glno_idx, thresh_hz=22.0) -> dict:
    """cx_wedge.simulate's `sample`: in / out means and counts above 22 Hz, PEN / Delta7 / GLNO means, vector strength, centre."""
    r = fb.brain.rates(idx_epg).copy()
    d = {"in_mean": float(r[inside].mean()), "out_mean": float(r[~inside].mean()),
         "in_above": int((r[inside] > thresh_hz).sum()), "out_above": int((r[~inside] > thresh_hz).sum()),
         "pen": float(fb.brain.mean_rate(pen_idx)), "delta7": float(fb.brain.mean_rate(d7_idx)), "glno": float(fb.brain.mean_rate(glno_idx)),
         "wedge_profile": [float(r[wedge_of == w].mean()) for w in range(16)]}
    ang = 2 * np.pi * wedge_of / 16
    z = np.sum(r * np.exp(1j * ang)) / max(r.sum(), 1e-9)
    d["vector_strength"] = float(np.abs(z)); d["centre_wedge"] = float((np.angle(z) % (2 * np.pi)) / (2 * np.pi) * 16)
    return d


def record_compass(args, lif, optic_params):
    import torch
    from flyverse.fly import FlyBrain
    sys.path.insert(0, str(ROOT / "scripts"))
    import cx_wedge                                                   # compass_cells: wedge identity of every EPG (cx_wedge.md 1)
    from flyverse.interp.health import presynaptic_indices
    gpu = _device_ok(args)
    t0 = time.time()
    c = load_connectome(args)
    cells = cx_wedge.compass_cells(c)
    epg = cells["EPG"]; idx_epg = np.asarray(epg["idx"])
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, [(args.start_wedge + j) % 16 for j in range(args.width)])
    pen_idx, d7_idx = np.asarray(cells["PEN"]["idx"]), np.asarray(cells["Delta7"]["idx"])
    glno_idx = common.resolve(c, "GLNO")
    tpg = compass_type_path_gain(args.gE, args.gD)
    params = dataclasses.replace(lif, adapt_by_type={COMPASS_RE: 0.0}, type_path_gain=tpg)
    fb = FlyBrain(c, lif_params=params, optic_params=optic_params, seed=args.seed, device=args.device, cuda_graphs=gpu and not args.no_graphs)
    target = common.resolve(c, COMPASS_TARGET)
    frozen = common.frozen_indices(fb)
    presyn = presynaptic_indices(c, target, exclude=frozen)
    rec = Recorder(c, target, quantities=("rate_hz", "v_mv", "adapt_mv", "refrac", "spike_count"))
    prec = Recorder(c, presyn, quantities=("rate_hz",))
    total_ms = (args.settle_s + args.pulse_s + args.seconds) * 1000.0
    fb.stimulate(idx_epg, args.background, total_ms + 100)          # the held 10 Hz background, as the grid's long pulse
    n_frames = int(round(total_ms / common.FRAME_MS)); marks = {}
    k_pulse = int(round(args.settle_s * 1000.0 / common.FRAME_MS))
    mark_at = {"pre": args.settle_s, "during": args.settle_s + args.pulse_s}
    mark_at.update({f"t{m}": args.settle_s + args.pulse_s + m for m in (0.5, 1.0, 2.0, 3.0, 5.0) if m <= args.seconds + 1e-9})
    print(f"compass gE {args.gE} gD {args.gD} seed {args.seed} receptor {params.receptor_model} device {fb.brain.device} "
          f"(cuda graphs {fb.cuda_graphs}, event_driven {fb.brain.event_driven}, kernels {fb.brain.cuda}); "
          f"{len(target)} target cells, {len(presyn)} presynaptic, {n_frames} frames; setup {time.time() - t0:.0f} s", flush=True)
    t1 = time.time()
    for k in range(n_frames):
        if k == k_pulse:                                               # t = settle_s: the pulse on the driven wedges
            fb.stimulate(idx_epg[inside], args.background + args.pulse_hz, args.pulse_s * 1000.0)
        fb.step(common.FRAME_MS)
        rec.capture(fb); prec.capture(fb)
        t_s = round(fb.t / 1000.0, 3)
        for tag, at in mark_at.items():
            if abs(t_s - at) < 1e-6:
                marks[tag] = _compass_sample(fb, idx_epg, inside, wedge_of, pen_idx, d7_idx, glno_idx)
    wall = time.time() - t1
    groups = {"EPG_in": idx_epg[inside].tolist(), "EPG_out": idx_epg[~inside].tolist(), "PEN": pen_idx.tolist(),
              "Delta7": d7_idx.tolist(), "PEG": np.asarray(cells["PEG"]["idx"]).tolist(), "Ring": np.asarray(cells["Ring"]["idx"]).tolist(),
              "GLNO": glno_idx.tolist(), "EPGt": np.asarray(cells["EPGt"]["idx"]).tolist()}
    stimulus = {"protocol": "compass", "params": {"gE": args.gE, "gD": args.gD, "delta7_pen": False, "gR": 1.0, "background_hz": args.background,
                                                  "pulse_hz": args.pulse_hz, "pulse_s": args.pulse_s, "settle_s": args.settle_s, "free_s": args.seconds,
                                                  "start_wedge": args.start_wedge, "width": args.width, "adapt_by_type": {COMPASS_RE: 0.0},
                                                  "type_path_gain": tpg, "n_in": int(inside.sum()), "n_out": int((~inside).sum())},
                "control": "the undriven wedges of the same ring (EPG_out) and cx_glno.md section 5's six-seed table"}
    prov = common.provenance(c, params, optic_params, fb=fb, device=args.device, seeds=[args.seed], stimulus=stimulus, cache_dir=args.cache_dir)
    meta = {"protocol": "compass", "seed": args.seed, "label": "compass_cells", "selection": COMPASS_TARGET, "groups": groups,
            "wedge_of": wedge_of.tolist(), "inside_idx": idx_epg[inside].tolist(), "cx_wedge_marks": marks, "wall_s": round(wall, 1),
            "windows": {"pre": [0.0, args.settle_s], "pulse": [args.settle_s, args.settle_s + args.pulse_s], "free": [args.settle_s + args.pulse_s, total_ms / 1000.0],
                        "scored": [max(args.settle_s + args.pulse_s, total_ms / 1000.0 - 2.5), total_ms / 1000.0]},
            "provenance": prov, "file": str(Path(args.out).with_suffix(".npz")), "generator": " ".join(sys.argv)}
    R = rec.finish(meta); path = R.save(args.out)
    P = prec.finish({"protocol": "compass", "seed": args.seed, "role": "presyn", "of": str(path), "provenance": prov, "file": str(Path(str(args.out) + "_presyn").with_suffix(".npz"))})
    P.save(str(args.out) + "_presyn")
    m = marks.get("t5.0", {})
    print(f"t5.0: in {m.get('in_mean', float('nan')):.1f} ({m.get('in_above')}/{int(inside.sum())}) out {m.get('out_mean', float('nan')):.1f} "
          f"({m.get('out_above')}/{int((~inside).sum())}) PEN {m.get('pen', float('nan')):.1f} D7 {m.get('delta7', float('nan')):.1f} "
          f"GLNO {m.get('glno', float('nan')):.1f} vs {m.get('vector_strength', float('nan')):.2f}; {wall:.0f} s; device {prov['execution']['device']}", flush=True)
    print(f"wrote {path} and {str(args.out) + '_presyn.npz'}", flush=True)
    return path


# ---------------------------------------------------------------------------------------------- record: walk
def record_walk(args, lif, optic_params):
    import torch
    from flyverse import body, brain, motor, olfaction, optic, retina, world
    from flyverse.interp.health import presynaptic_indices
    gpu = _device_ok(args)
    t0 = time.time()
    c = load_connectome(args); n = c.neurons
    # ---- scripts/benchmark.py sec_walk, line for line (the protocol is the reference; only the capture calls are added)
    leg = c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"]); wg = motor.wing_groups(c)
    r = retina.build_retina(c)
    w, info = world.make_room()
    w.spheres.append(world.Sphere((9, 9, 9), (0.03,) * 3, "black")); loom_idx = len(w.spheres) - 1
    dirs_b, wts = r.ray_directions(); wts_t = torch.from_numpy(wts).float().to(w.device)
    rs = brain._receptor(c, lif, with_counts=lif.receptor_model == "full")
    ol = optic.OpticLobe(c, r, optic_params, device=args.device, receptor=rs, receptor_gain=brain._receptor_gain(lif)); ol.relax()
    b = brain.Brain(c, lif, receptor=rs, seed=args.seed, device=args.device); b.freeze(ol.rate_idx)
    dev = ol.device                                                    # the world traces on the default device; the lobe may be elsewhere (--device cpu)
    b.record_activity = True
    fly = body.FlyState(x=-0.3, y=0.0, z=info["table_top_z"], heading=0.0)

    def col_rad():
        d = fly.body_to_world(dirs_b.reshape(-1, 3)); o = np.broadcast_to(fly.eye_pos, d.shape)
        rad = w.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        return (rad.reshape(dirs_b.shape[0], dirs_b.shape[1], 4) * wts_t[None, :, None]).sum(1).to(dev)

    gf = c.select(type="DNp01")
    dn_all = c.select(superclass="descending_neuron")
    target = np.unique(np.concatenate([dn_all, gf, np.asarray(wg.power), leg]))
    presyn = presynaptic_indices(c, target, exclude=ol.rate_idx)
    holder = SimpleNamespace(brain=b, optic=ol, c=c, cuda_graphs=False)
    rec = Recorder(c, target, quantities=("rate_hz", "v_mv", "adapt_mv", "refrac", "spike_count"))
    prec = Recorder(c, presyn, quantities=("rate_hz",))
    print(f"walk seed {args.seed} receptor {lif.receptor_model} device {b.device} (event_driven {b.event_driven}, kernels {b.cuda}); "
          f"{len(target)} target cells ({len(dn_all)} DNs), {len(presyn)} presynaptic; setup {time.time() - t0:.0f} s", flush=True)
    gf_walk, pw_walk = [], []
    olf_walk = olfaction.Olfaction(c, [(name, cen, 1.0) for name, cen, rad in info["fruit"]])   # smell on, as in the demo
    t1 = time.time()
    for k in range(args.frames_walk):
        fly.x += 0.004 * 0.01; olf_walk.apply(b, fly.eye_pos); b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
        rec.capture(holder); prec.capture(holder)
        if k >= args.frames_walk // 3:
            gf_walk.append(float(b.rate[0, b._idx(gf)].mean())); pw_walk.append(float(b.rate[0, b._idx(wg.power)].mean()))
    rt = b.rate_np()
    pw_sustained = float(np.max(np.convolve(pw_walk, np.ones(min(30, len(pw_walk))) / min(30, len(pw_walk)), mode="valid")))
    walk = {"GF_mean_hz": float(np.mean(gf_walk)), "GF_max_hz": float(np.max(gf_walk)), "frac_active": float((rt > 1).mean()),
            "power_mean_hz": float(np.mean(pw_walk)), "power_max_hz": float(np.max(pw_walk)), "power_sustained_hz": pw_sustained,
            "leg_hz": float(rt[leg].mean())}
    eye = fly.eye_pos + np.array([0, 0, 0.01]); esc = None; gf_peak = 0.0
    for k in range(args.frames_loom):
        d = max(0.5 - 1.0 * k * 0.01, 0.035)
        w.move_sphere(loom_idx, eye + np.array([0.0, d, 0.0]))
        b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
        rec.capture(holder); prec.capture(holder)
        g = float(b.rate[0, b._idx(gf)].mean()); gf_peak = max(gf_peak, g)
        if esc is None and g >= 20:
            esc = d * 100
    loom = {"GF_peak_hz": gf_peak, "escape_cm": esc}
    wall = time.time() - t1
    groups = {"DN_all": dn_all.tolist(), "DNp01": gf.tolist(), "DNa02": common.resolve(c, "DNa02").tolist(), "DNp09": common.resolve(c, "DNp09").tolist(),
              "MDN": common.resolve(c, "MDN").tolist(), "wing_power": np.asarray(wg.power).tolist(), "leg_MN": leg.tolist()}
    stimulus = {"protocol": "walk", "params": {"walk_s": 1.5, "walk_speed_m_s": 0.004, "smell": "on (room fruit)", "loom_s": 0.8,
                                               "loom_from_m": 0.5, "scored_walk_window_s": [0.5, 1.5], "frame_ms": 10.0, "lif_steps_per_frame": 20},
                "control": "the same protocol under the other receptor arm (default vs off); docs/audits/receptor_integration.md G.4"}
    prov = common.provenance(c, lif, optic_params, fb=holder, device=args.device, seeds=[args.seed], stimulus=stimulus, cache_dir=args.cache_dir)
    meta = {"protocol": "walk", "seed": args.seed, "label": "descending+motor", "selection": "superclass=descending_neuron | DNp01 | wing power MNs | leg MNs",
            "groups": groups, "walk": walk, "loom": loom, "wall_s": round(wall, 1),
            "windows": {"walk": [0.0, 1.5], "scored": [0.5, 1.5], "loom": [1.5, 2.3]},
            "provenance": prov, "file": str(Path(args.out).with_suffix(".npz")), "generator": " ".join(sys.argv)}
    R = rec.finish(meta); path = R.save(args.out)
    P = prec.finish({"protocol": "walk", "seed": args.seed, "role": "presyn", "of": str(path), "provenance": prov, "file": str(Path(str(args.out) + "_presyn").with_suffix(".npz"))})
    P.save(str(args.out) + "_presyn")
    print(f"walk      GF mean {walk['GF_mean_hz']:.2f} max {walk['GF_max_hz']:.3f}  wing power max {walk['power_max_hz']:.1f} (sustained {pw_sustained:.1f})  "
          f"leg MN {walk['leg_hz']:.2f} frac {walk['frac_active']:.3f}; loom GF peak {gf_peak:.1f} escape {esc}; {wall:.0f} s; device {prov['execution']['device']}", flush=True)
    print(f"wrote {path} and {str(args.out) + '_presyn.npz'}", flush=True)
    return path


# ---------------------------------------------------------------------------------------------- analyse
def _stems(pattern) -> list[str]:
    if isinstance(pattern, (list, tuple)):
        out = []
        for p in pattern:
            out.extend(_stems(p))
        return sorted(set(out))
    files = sorted(glob.glob(str(pattern) + ".npz")) or sorted(glob.glob(str(pattern)))
    stems = [f[:-4] if f.endswith(".npz") else (f[:-5] if f.endswith(".json") else f) for f in files]
    return sorted({s for s in stems if not s.endswith("_presyn")})


def _load_run(stem: str):
    R = Recording.load(stem)
    P = Recording.load(stem + "_presyn") if Path(stem + "_presyn.npz").exists() else None
    R.meta.setdefault("file", stem + ".npz")
    return R, P


def _groups_for(R: Recording, groups):
    if groups is None:
        return None
    if isinstance(groups, dict):
        return {k: np.asarray(v) if not isinstance(v, str) else v for k, v in groups.items()}
    if groups == "meta":
        g = R.meta.get("groups")
        return {k: np.asarray(v, dtype=np.int64) for k, v in g.items()} if g else None
    raise ValueError(f"unknown groups {groups!r}: None, 'meta' or a dict")


def analyse_runs(stems, c, params=None, window=None, by="type", groups="meta", null_stems=None, counts=None, thresholds=None,
                 ew_cache=None) -> Result:
    """health() on every run (a recording stem = path without .npz; <stem>_presyn is used when present), merged into one
    Result: per-run tables, the replicate table (mean / sd / n / values per group x statistic), the Neurome rows with
    n_trials = runs, and -- with `null_stems` (the other arm) -- common.compare per group x statistic."""
    from flyverse.interp import health as H
    ew_cache = {} if ew_cache is None else ew_cache

    def run_all(stem_list):
        out = []
        for stem in stem_list:
            R, P = _load_run(stem)
            p = params if params is not None else H.resolve_params(None, R)
            ew = None
            if c is not None:
                key = repr(common.to_jsonable(dataclasses.asdict(p)))
                if key not in ew_cache:
                    ew_cache[key] = common.effective_weights(c, p)
                ew = ew_cache[key]
            res = H.health(R, c=c, params=p, window=window, by=by, ew=ew, presyn=P, counts=counts, thresholds=thresholds)
            g = _groups_for(R, groups) if c is not None else None
            gres = H.health(R, c=c, params=p, window=window, by=g, ew=ew, presyn=P, counts=counts, thresholds=thresholds, readout_rows=False) if g else None
            out.append((stem, R, res, gres))
        return out

    runs = run_all(_stems(stems) if isinstance(stems, str) else list(stems))
    if not runs:
        raise SystemExit(f"no recordings match {stems}")
    nulls = run_all(_stems(null_stems)) if null_stems else []   # _stems takes one glob or a list of them (--null-runs)
    first = runs[0][2]
    prov = dict(first.provenance)
    prov["analysis"] = {"flyverse_commit": common.git_state(), "note": "the recordings' provenance is the recording job's (a cluster run dir is not a git checkout: "
                                                                 "its commit reads 'unknown'); this block is the checkout that ran analyse"}
    res = Result.new("health", prov)
    res.add_table("per_type", pd.concat([r.table("per_type").assign(run_index=k, seed=R.meta.get("seed")) for k, (_, R, r, _) in enumerate(runs)], ignore_index=True))
    res.add_table("replicates", H.replicate_table([r for _, _, r, _ in runs]))
    if any(g is not None for *_, g in runs):
        res.add_table("per_group", pd.concat([g.table("per_type").assign(run_index=k, seed=R.meta.get("seed")) for k, (_, R, _, g) in enumerate(runs) if g is not None], ignore_index=True))
        res.add_table("replicates_group", H.replicate_table([g for *_, g in runs if g is not None]))
    rb = pd.concat([r.table("readout_per_body").assign(run_index=k) for k, (_, _, r, _) in enumerate(runs) if "readout_per_body" in r.tables], ignore_index=True)
    if len(rb):
        agg = rb.groupby(["bodyId", "quantity"], sort=False).agg(model_index=("model_index", "first"), type=("type", "first"), unit_kind=("unit_kind", "first"),
                                                                window_start_s=("window_start_s", "first"), window_end_s=("window_end_s", "first"),
                                                                stimulus_value=("stimulus_value", "mean"), trial_sd=("stimulus_value", lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan),
                                                                n_trials=("stimulus_value", "size"), unit=("unit", "first")).reset_index()
        agg["control_value"] = None; agg["stimulus_minus_control"] = None; agg["control_ids"] = [[] for _ in range(len(agg))]
        res.add_table("readout_per_body", agg[common.EXPORT_TABLES["readout_per_body"]])
    if nulls:
        res.add_table("per_type_null", pd.concat([r.table("per_type").assign(run_index=k, seed=R.meta.get("seed")) for k, (_, R, r, _) in enumerate(nulls)], ignore_index=True))
        res.add_table("compare", H.compare_arms([r for _, _, r, _ in runs], [r for _, _, r, _ in nulls]))
        if any(g is not None for *_, g in nulls) and any(g is not None for *_, g in runs):
            res.add_table("per_group_null", pd.concat([g.table("per_type").assign(run_index=k, seed=R.meta.get("seed")) for k, (_, R, _, g) in enumerate(nulls) if g is not None], ignore_index=True))
            res.add_table("compare_group", H.compare_arms([g for *_, g in runs if g is not None], [g for *_, g in nulls if g is not None]))
    for pop in first.populations:
        res.populations.append(pop)
    res.replicates = {"n": len(runs), "unit": "runs",
                      "runs": [{"run_index": k, "seed": R.meta.get("seed"), "file": R.meta.get("file"), "device": r.provenance.get("execution", {}).get("device"),
                                "wall_s": R.meta.get("wall_s")} for k, (_, R, r, _) in enumerate(runs)],
                      "null": None if not nulls else {"n": len(nulls), "unit": "runs",
                                                      "runs": [{"run_index": k, "seed": R.meta.get("seed"), "file": R.meta.get("file"),
                                                                "device": r.provenance.get("execution", {}).get("device")} for k, (_, R, r, _) in enumerate(nulls)]}}
    res.summary = {"protocol": runs[0][1].meta.get("protocol"), "n_runs": len(runs), "n_null_runs": len(nulls), "window_s": list(window) if window else None,
                   "by": by, "groups": list(_groups_for(runs[0][1], groups) or {}) if c is not None else [],
                   "runs": [{"seed": R.meta.get("seed"), "n_cells": r.summary["n_cells"], "flagged": r.summary["flagged"],
                             "group_flags": (g.summary["flagged"] if g is not None else None),
                             "meta": {k: R.meta.get(k) for k in ("walk", "loom", "cx_wedge_marks", "wall_s") if k in R.meta}} for _, R, r, g in runs],
                   "null_runs": [{"seed": R.meta.get("seed"), "flagged": r.summary["flagged"], "group_flags": (g.summary["flagged"] if g is not None else None),
                                  "meta": {k: R.meta.get(k) for k in ("walk", "loom", "cx_wedge_marks") if k in R.meta}} for _, R, r, g in nulls],
                   "sign0": first.summary.get("sign0"), "thresholds": first.summary["thresholds"], "lif": first.summary["lif"],
                   "presyn": first.summary.get("presyn"), "ew_md5": first.summary.get("ew_md5")}
    res.files = {"recordings": [s + ".npz" for s, *_ in runs], "presyn": [s + "_presyn.npz" for s, *_ in runs if Path(s + "_presyn.npz").exists()],
                 "null_recordings": [s + ".npz" for s, *_ in nulls], "generator": "scripts/interp_health.py analyse"}
    return res


def structure(c, by="module", params=None, counts=None, thresholds=None) -> Result:
    """The NT audit's group tables on the cached connectome (health with recording=None)."""
    from flyverse.interp import health as H
    res = H.health(None, c=c, params=params, by=by, counts=counts, thresholds=thresholds)
    res.files = {"generator": "scripts/interp_health.py structure", "recordings": []}
    return res


# ---------------------------------------------------------------------------------------------- validate
def _in(v, lo, hi):
    return bool(np.isfinite(v)) and lo <= v <= hi


def validate(args, c, params_override=None):
    """The validation targets of common.VALIDATION['health'] side by side with what the tool measures."""
    from flyverse.interp import health as H
    ref = common.VALIDATION["health"]["reference"]
    measured, parts, tables = {}, {}, {}
    counts, sign0_avail = H.full_counts(c)
    # ---- (1) the refractory-limited bump
    if args.compass:
        cw = parse_window(args.compass_window) or (5.5, 8.0)
        comp = analyse_runs(args.compass, c, params=params_override, window=cw, by="type", groups="meta", counts=counts)
        tables["compass_per_group"] = comp.table("per_group"); tables["compass_replicates_group"] = comp.table("replicates_group")
        tables["compass_per_type"] = comp.table("per_type")
        rows = []
        pg = comp.table("per_group")
        for run in comp.summary["runs"]:
            k = run["seed"]; m = run["meta"].get("cx_wedge_marks", {}).get("t5.0", {})
            sub = pg[pg.seed == k].set_index("group")
            rows.append({"seed": k, "EPG_in_rate_mean": sub.loc["EPG_in"].rate_mean, "EPG_in_spike_rate": sub.loc["EPG_in"].spike_rate_hz,
                         "EPG_in_refractory_load": sub.loc["EPG_in"].refractory_load, "EPG_in_refrac_measured": sub.loc["EPG_in"].refrac_measured,
                         "EPG_in_flags": sub.loc["EPG_in"].state_flags, "EPG_out_rate_mean": sub.loc["EPG_out"].rate_mean,
                         "PEN_rate_mean": sub.loc["PEN"].rate_mean, "PEN_flags": sub.loc["PEN"].state_flags, "PEN_refractory_load": sub.loc["PEN"].refractory_load,
                         "Delta7_rate_mean": sub.loc["Delta7"].rate_mean, "Delta7_flags": sub.loc["Delta7"].state_flags, "Delta7_refractory_load": sub.loc["Delta7"].refractory_load,
                         "GLNO_rate_mean": sub.loc["GLNO"].rate_mean, "GLNO_flags": sub.loc["GLNO"].state_flags,
                         "cx_wedge_t5_in": m.get("in_mean"), "cx_wedge_t5_out": m.get("out_mean"), "cx_wedge_t5_in_above": m.get("in_above"),
                         "cx_wedge_t5_out_above": m.get("out_above"), "cx_wedge_t5_pen": m.get("pen"), "cx_wedge_t5_delta7": m.get("delta7"),
                         "cx_wedge_t5_glno": m.get("glno"), "cx_wedge_t5_vs": m.get("vector_strength"), "device": [r["device"] for r in comp.replicates["runs"] if r["seed"] == k][0]})
        cr = pd.DataFrame(rows); tables["compass_validation"] = cr
        ok = [(_in(r.EPG_in_rate_mean, *ref["bump_hz"]) and _in(r.EPG_in_refractory_load, 0.40, 0.57) and _in(r.PEN_rate_mean, *ref["PEN_hz"])
               and _in(r.Delta7_rate_mean, *ref["Delta7_hz"]) and "refractory_limited" in r.EPG_in_flags) for r in cr.itertuples()]
        parts["bump"] = {"reproduced": bool(all(ok)) and len(ok) >= common.MIN_REPLICATES, "runs_ok": int(sum(ok)), "n_runs": len(ok),
                         "reference": {"bump_hz": ref["bump_hz"], "refractory_load": [0.40, 0.57], "PEN_hz": ref["PEN_hz"], "Delta7_hz": ref["Delta7_hz"],
                                       "seed_matched_cx_glno_r4": {"bump_hz_seeds012": [201.0, 201.4, 204.3], "PEN": "47.9 +- 0.8", "Delta7": "100.1 +- 1.6", "GLNO": "132.5 +- 3.7", "out_hz": 10.2}},
                         "window_s": list(cw)}
        measured["compass"] = cr.to_dict("records")
    # ---- (2) the walk-section operating points, default vs off
    if args.walk_default:
        ww = parse_window(args.walk_window) or (0.5, 1.5)
        walk = analyse_runs(args.walk_default, c, params=params_override, window=ww, by="type", groups="meta", null_stems=args.walk_off or None, counts=counts)
        tables["walk_per_group"] = walk.table("per_group"); tables["walk_replicates_group"] = walk.table("replicates_group")
        if "compare_group" in walk.tables:
            tables["walk_compare_group"] = walk.table("compare_group"); tables["walk_per_group_null"] = walk.table("per_group_null")
        gf = {"default": [r["meta"]["walk"]["GF_max_hz"] for r in walk.summary["runs"]],
              "off": [r["meta"]["walk"]["GF_max_hz"] for r in walk.summary["null_runs"]]}
        pg = walk.table("per_group"); dn = pg[pg.group == "DN_all"]
        dn_null = walk.table("per_group_null") if "per_group_null" in walk.tables else pd.DataFrame()
        dn_null = dn_null[dn_null.group == "DN_all"] if len(dn_null) else dn_null
        measured["walk"] = {"window_s": list(ww), "GF_max_hz": gf,
                            "GF_max_hz_reference_G4": {"off": 4.964, "default": 4.629, "note": "single runs of a chaotic section (receptor_integration.md G.4)"},
                            "DN_all_default": dn[["seed", "n_cells", "rate_mean", "silent_frac", "at_threshold_frac", "refractory_load", "v_margin_mv", "ei_balance", "input_coverage", "state_flags"]].to_dict("records"),
                            "DN_all_off": dn_null[["seed", "n_cells", "rate_mean", "silent_frac", "at_threshold_frac", "refractory_load", "v_margin_mv", "ei_balance", "input_coverage", "state_flags"]].to_dict("records") if len(dn_null) else [],
                            "compare_DN_all": walk.table("compare_group")[walk.table("compare_group").group == "DN_all"].to_dict("records") if "compare_group" in walk.tables else []}
        parts["walk"] = {"reproduced": None, "note": "no prior per-type operating-point reference exists for the walk section; these are the tool's first numbers "
                                                    "(3 runs per arm) and the GF_max cross-check against G.4's single runs", "n_runs": len(gf["default"]), "n_null_runs": len(gf["off"])}
    # ---- (3) the sign-0 populations of the NT audit
    sm = structure(c, by="module", params=params_override, counts=counts); ss = structure(c, by="superclass", params=params_override, counts=counts)
    tables["structure_module"] = sm.table("per_type"); tables["structure_superclass"] = ss.table("per_type")
    mod = sm.table("per_type").set_index("group"); sc = ss.table("per_type").set_index("group")
    s0 = sm.summary["sign0"]
    measured["structure"] = {"sign0": s0, "mushroom_body_in_share": float(mod.loc["mushroom_body"].sign0_in_share), "mushroom_body_out_share": float(mod.loc["mushroom_body"].sign0_out_share),
                             "gustatory_in_share": float(mod.loc["gustatory"].sign0_in_share), "gustatory_out_share": float(mod.loc["gustatory"].sign0_out_share),
                             "descending_out_share": float(mod.loc["descending"].sign0_out_share), "central_in_share": float(mod.loc["central"].sign0_in_share),
                             "visual_centrifugal_out_share": float(sc.loc["visual_centrifugal"].sign0_out_share), "visual_centrifugal_pre_sign0": int(sc.loc["visual_centrifugal"].n_sign0_pre),
                             "ol_intrinsic_out_share": float(sc.loc["ol_intrinsic"].sign0_out_share), "cb_intrinsic_out_share": float(sc.loc["cb_intrinsic"].sign0_out_share),
                             "descending_neuron_out_share": float(sc.loc["descending_neuron"].sign0_out_share)}
    st = measured["structure"]
    audit = {"sign0_neurons": 3312, "sign0_presynaptic": 2683, "sign0_synapses": 2701289, "synapses_total": 124161873, "sign0_synapse_share": 0.022,
             "mushroom_body_in": ref["mushroom_body_input_share_silenced"], "mushroom_body_out": 0.125, "visual_centrifugal_out": ref["visual_centrifugal_output_silenced"],
             "visual_centrifugal_pre_sign0": 73, "gustatory_in": 0.055, "gustatory_out": 0.084, "descending_out": 0.047, "central_in": 0.037}
    parts["sign0"] = {"reproduced": (abs(st["mushroom_body_in_share"] - audit["mushroom_body_in"]) <= 0.002
                                     and abs(st["mushroom_body_out_share"] - audit["mushroom_body_out"]) <= 0.002
                                     and abs(st["visual_centrifugal_out_share"] - audit["visual_centrifugal_out"]) <= 0.002
                                     and abs(s0["sign0_synapse_share"] - audit["sign0_synapse_share"]) <= 0.002
                                     and s0["n_sign0_cells"] == audit["sign0_neurons"] and s0["n_sign0_presynaptic"] == audit["sign0_presynaptic"]
                                     and int(s0["sign0_synapses"]) == audit["sign0_synapses"] and int(s0["synapses_total"]) == audit["synapses_total"]),
                      "reference": dict(audit, sign0_presynaptic_bodies_VALIDATION=ref["sign0_presynaptic_bodies"]),
                      "note": "docs/audits/nt_audit.md is the reference (generated from the adopted cache: 3,312 sign-0 neurons, 2,683 presynaptic); "
                              "VALIDATION['health']'s 3,407 is the pre-override cache's count (docs/audits/receptor_verification.md: sum|W| 121,427,136), "
                              "not this model's; the tool reports the count of the cache it loaded and its fingerprint",
                      "sign0_counts_available": s0["sign0_counts_available"]}
    status_parts = [k for k, v in parts.items() if v.get("reproduced") is True]
    failed = [k for k, v in parts.items() if v.get("reproduced") is False]
    status = "reproduced" if (not failed and "bump" in parts and "sign0" in parts) else ("not reproduced: " + ", ".join(failed) if failed else "partially run: " + ", ".join(status_parts))
    res = Result.new("health", (comp.provenance if args.compass else sm.provenance))
    for k, t in tables.items():
        res.add_table(k, t)
    res.validation["measured"] = measured; res.validation["status"] = status; res.validation["parts"] = parts
    res.summary = {"status": status, "parts": {k: v.get("reproduced") for k, v in parts.items()}}
    res.replicates = comp.replicates if args.compass else {"n": 0, "unit": "runs", "runs": [], "null": None}
    res.files = {"generator": "scripts/interp_health.py validate " + " ".join(sys.argv[2:]), "compass": _stems(args.compass) if args.compass else [],
                 "walk_default": _stems(args.walk_default) if args.walk_default else [], "walk_off": _stems(args.walk_off) if args.walk_off else []}
    return res


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="GPU: run a protocol and save the Recording (+ provenance)")
    r.add_argument("--protocol", choices=("compass", "walk"), required=True)
    r.add_argument("--out", required=True, help="recording stem (writes <out>.npz / .json and <out>_presyn.*)")
    r.add_argument("--gE", type=float, default=2.0); r.add_argument("--gD", type=float, default=15.0)
    r.add_argument("--background", type=float, default=10.0); r.add_argument("--pulse-hz", type=float, default=40.0)
    r.add_argument("--pulse-s", type=float, default=2.0); r.add_argument("--seconds", type=float, default=5.0)
    r.add_argument("--start-wedge", type=int, default=0); r.add_argument("--width", type=int, default=4)
    r.add_argument("--no-graphs", action="store_true"); r.add_argument("--allow-cpu", action="store_true")
    r.add_argument("--settle-s", type=float, default=1.0, help="compass: seconds on the background before the pulse (the grid: 1)")
    r.add_argument("--frames-walk", type=int, default=150, help="walk: walking frames (sec_walk: 150 = 1.5 s; the last two thirds are scored)")
    r.add_argument("--frames-loom", type=int, default=80, help="walk: loom frames (sec_walk: 80)")
    common.add_common_args(r)
    a = sub.add_parser("analyse", help="CPU: health() over recordings, merged over runs")
    a.add_argument("--recordings", required=True, help="glob of recording stems / .npz files (>= 3 runs = one arm)")
    a.add_argument("--presyn", default=None, help="(unused: <stem>_presyn.npz is picked up automatically)")
    a.add_argument("--window", default=None, help="start,end seconds"); a.add_argument("--by", default="type")
    a.add_argument("--groups", default="meta", help="'meta' (the recording's groups), 'none', or a JSON dict {label: spec}")
    a.add_argument("--no-structure", action="store_true", help="dynamic columns only (no Connectome load)")
    a.add_argument("--params-from", choices=("meta", "args"), default="meta", help="LIFParams from the recording's provenance (default) or the --lif / --receptor-* flags")
    common.add_common_args(a)
    s = sub.add_parser("structure", help="CPU: the sign-0 / frozen shares of every group of the cached connectome")
    s.add_argument("--by", default="module"); common.add_common_args(s)
    v = sub.add_parser("validate", help="CPU: the validation targets side by side with the tool's numbers")
    v.add_argument("--compass", default=None, help="glob of compass recording stems"); v.add_argument("--compass-window", default="5.5,8")
    v.add_argument("--walk-default", default=None); v.add_argument("--walk-off", default=None); v.add_argument("--walk-window", default="0.5,1.5")
    v.add_argument("--params-from", choices=("meta", "args"), default="meta")
    common.add_common_args(v)
    args = ap.parse_args(argv)
    lif, op = common.params_from_args(args)
    if args.cmd == "record":
        path = record_compass(args, lif, op) if args.protocol == "compass" else record_walk(args, lif, op)
        return 0
    if args.cmd == "analyse":
        c = None if args.no_structure else load_connectome(args)
        groups = None if args.groups in ("none", "") else (json.loads(args.groups) if args.groups.startswith("{") else args.groups)
        res = analyse_runs(args.recordings, c, params=(lif if args.params_from == "args" else None), window=parse_window(args.window), by=args.by,
                           groups=groups, null_stems=(args.null_runs or None))
        table = res.table("replicates_group") if "replicates_group" in res.tables else res.table("replicates")
        if not args.quiet:
            common.print_table(table.drop(columns=["values"]) if "values" in table else table, max_rows=80)
    elif args.cmd == "structure":
        c = load_connectome(args)
        res = structure(c, by=args.by, params=(lif if args.receptor_model != "default" or args.lif else None))
        if not args.quiet:
            t = res.table("per_type")
            common.print_table(t[["group", "n_cells", "n_sign0_pre", "in_syn", "sign0_in_share", "out_syn", "sign0_out_share", "frozen_in_share", "frozen_out_share", "fanin_scale_med", "state_flags"]], floatfmt="{:.4f}", max_rows=80)
            print(json.dumps(res.summary["sign0"], indent=1))
    else:
        c = load_connectome(args)
        res = validate(args, c, params_override=(lif if args.params_from == "args" else None))
        if not args.quiet:
            for name in ("compass_validation", "walk_compare_group"):
                if name in res.tables:
                    print(f"\n[{name}]"); common.print_table(res.table(name), max_rows=40)
            print("\n[validation]"); print(json.dumps(common.to_jsonable(res.validation["parts"]), indent=1)); print("status:", res.validation["status"])
    problems = res.check()
    if problems:
        print("Result.check():", problems)
    path = Path(args.json) if args.json else common.default_json_path("health", res.run_id)
    res.save(path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
