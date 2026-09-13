"""CLI for the Neurome probe export (flyverse/interp/export.py; docs/NEUROME_INTERFACE.md section 1, docs/INTERP.md 4.8).

    # CPU: serialize any Result JSON into out/export/<run_id>/ (the plain form of docs/INTERP.md section 7)
    PYTHONIOENCODING=utf-8 python scripts/interp_export.py --result out/interp/trace/ball.json --retina out/tr/retina.npz --out out/export

    # CPU: check a finished export (hashes, decimal bodyIds, the two-rows-per-body rule, ids in cache/neurons.parquet)
    PYTHONIOENCODING=utf-8 python scripts/interp_export.py verify --run-dir out/export/<run_id> --neurons cache/neurons.parquet

    # GPU (cluster only): re-run the object-sweep protocol of scripts/probe_object_sweep.py and capture what the probe
    # JSON pools away -- per-cell drive and rate for every recorded body, and the retinal sampling actually presented
    python scripts/cluster_run.py --name exp-obj --minutes 30 \
      "python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_export.py record --receptor-model off --seed 0 --retina --out out/expobj/stim_s0 > out/expobj/stim_s0.txt; cat out/expobj/stim_s0.txt" \
      "... --null --out out/expobj/null_s0 ..." --fetch out/expobj/

    # CPU: the recorded runs -> one Result -> the export, with the earlier run side by side
    PYTHONIOENCODING=utf-8 python scripts/interp_export.py run --stim "out/expobj/stim_s*.json" --null-runs "out/expobj/null_s*.json" \
      --reference out/r3obj/ball_off_s0.json --reference-null out/r3obj/null_off_s0.json --retina out/expobj/stim_s0_retina.npz \
      --json out/interp/export/object_sweep.json --out out/export

    # CPU: the structural (static) decomposition of a target's input, exported through the contributions table
    PYTHONIOENCODING=utf-8 python scripts/interp_export.py static-decompose --target "LC11|LC10a" --json out/interp/decompose/lc_static.json --out out/export

    # CPU: re-export a recorded size ladder (one run directory per size + the summary) from the recordings on file
    PYTHONIOENCODING=utf-8 python scripts/interp_export.py ladder --runs-csv out/export/<summary run>/runs.csv \
      --out out/export --family lc_drive --expect-counts '{"LC11": 143, "LC10a": 275}'

`record` is the only GPU subcommand and runs on the cluster; everything else is CPU. Every run records the REALISED
device (`sim.fb.brain.device`), the cache fingerprint and the resolved LIFParams / OpticParams, so the manifest alone
reconstructs the run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyverse.interp import common                      # noqa: E402
from flyverse.interp import export as ex                # noqa: E402

SPIKING_FRAMES = ("LC11", "LC10a")                      # the two populations Neurome asked for keep their time course


# ------------------------------------------------------------------------------------------------- record (GPU)
def _probe():
    """scripts/probe_object_sweep.py as a module (it initialises pygame and torch at import: GPU jobs only)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import probe_object_sweep as probe                   # noqa: PLC0415
    return probe


def capture_retina(sim, probe, n_frames: int, with_ball: bool) -> dict:
    """The retinal sampling the probe presented: per-column radiance [UV, B, G, R] per frame at the pinned pose with
    the ball at the frame's own sweep offset, plus the column -> photoreceptor-body map (flyverse/retina.py).

    A replay, not a second rollout: the pose is pinned every frame exactly as scripts/probe_object_sweep.py pins it
    and the ray tracer is deterministic, so the radiance is the one the optic lobe received. The radiance_check block
    the probe writes at four frames is the cross-check (scripts/interp_export.py run reports it)."""
    fly = sim.fly
    fly.place(*probe.POS, heading=probe.HEADING)
    eye0 = fly.eye_pos.copy(); fwd = fly.forward.copy(); left = fly.left.copy()
    ball_z = sim.info["table_top_z"] + probe.BALL_R
    r = sim.fb.retina
    rad = np.zeros((n_frames, len(r.col_dir), 4), np.float32)
    off = np.zeros(n_frames, np.float64)
    t0 = time.time()
    for j in range(n_frames):
        s = probe.ball_offset(j * probe.FRAME_S)
        off[j] = s if with_ball else np.nan
        if with_ball:
            centre = eye0 + probe.AHEAD * fwd + s * left
            centre[2] = ball_z
            sim.world.move_sphere(sim.loom_idx, centre)
        else:
            sim.world.move_sphere(sim.loom_idx, (9, 9, 9))
        fly.place(*probe.POS, heading=probe.HEADING)
        rad[j] = sim.column_radiance().detach().cpu().numpy()
        if j % 300 == 0:
            print(f"  retina replay frame {j}/{n_frames} ({time.time() - t0:.0f} s)", flush=True)
    c = sim.c
    types = c.neurons.type.fillna("").to_numpy()
    return {"radiance": rad, "t_s": np.arange(n_frames) * probe.FRAME_S, "ball_offset_m": off,
            "col_dir": np.asarray(r.col_dir), "col_az_el": np.asarray(r.col_az_el), "col_hex": np.asarray(r.col_hex),
            "col_side": np.asarray(r.col_side).astype(str), "pr_index": np.asarray(r.pr_index).astype(np.int64),
            "pr_body": c.neurons.bodyId.to_numpy()[np.asarray(r.pr_index)].astype(np.int64),
            "pr_column": np.asarray(r.pr_column).astype(np.int64), "pr_sens": np.asarray(r.pr_sens, np.float32),
            "pr_type": types[np.asarray(r.pr_index)].astype(str),
            "sampling": f"replay of the presented geometry, every frame ({probe.FRAME_S * 1000:.0f} ms), pinned pose"}


def cell_arrays(c, kinds, A: dict, B: dict) -> dict:
    """The per-cell numbers the probe JSON pools away: for every recorded body the window-mean optic drive (mV),
    the spike rate over the window (Hz) and, for the optic rate units, the mean and mean-|.| deviation from the
    operating point -- under condition A (ball, or the second none run under --null) and condition B (none).
    `kinds` is common.unit_kinds(c, fb) of the recording FlyBrain (graded = its optic rate units)."""
    types = c.neurons.type.fillna("").to_numpy()
    bodies = c.neurons.bodyId.to_numpy()
    out = {"types": np.array(list(A["drive"].keys()) + list(A["rate"].keys()), dtype=object)}
    for t, ix in A["spk_cells"].items():
        out[f"body__{t}"] = bodies[ix].astype(np.int64)
        out[f"index__{t}"] = np.asarray(ix, np.int64)
        out[f"unitkind__{t}"] = kinds[ix].astype(str)
        for lab, res in (("a", A), ("b", B)):
            out[f"drive_mean__{t}__{lab}"] = res["drive"][t].mean().astype(np.float32)
            out[f"drive_peak__{t}__{lab}"] = res["drive"][t].max.astype(np.float32)
            out[f"rate_hz__{t}__{lab}"] = np.asarray(res["rates_hz"][t], np.float32)
        if t in SPIKING_FRAMES:
            for lab, res in (("a", A), ("b", B)):
                out[f"drive_frames__{t}__{lab}"] = np.asarray(res["drive_frames"][t], np.float32)
                out[f"spk_frames__{t}__{lab}"] = np.clip(np.asarray(res["spk_frames"][t]), 0, 255).astype(np.uint8)
    for t in A["rate"]:
        ix = np.flatnonzero(types == t)
        out[f"body__{t}"] = bodies[ix].astype(np.int64)
        out[f"index__{t}"] = ix.astype(np.int64)
        out[f"unitkind__{t}"] = kinds[ix].astype(str)
        for lab, res in (("a", A), ("b", B)):
            r = res["rate"][t]
            out[f"dev_mean__{t}__{lab}"] = r.mean().astype(np.float32)
            out[f"dev_absmean__{t}__{lab}"] = (r.sum_abs / max(r.n, 1)).astype(np.float32)
            out[f"dev_max__{t}__{lab}"] = r.max.astype(np.float32)
            out[f"dev_min__{t}__{lab}"] = r.min.astype(np.float32)
    return out


def cmd_record(args) -> int:
    """Run the object-sweep protocol on the GPU and write <out>.json (probe-shaped), <out>_cells.npz (per body),
    <out>_prov.json (the provenance block with the realised device) and, with --retina, <out>_retina.npz."""
    import torch                                          # noqa: PLC0415
    probe = _probe()
    from flyverse import brain, connectome, optic         # noqa: PLC0415
    assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    probe.patch_receptor(args.receptor_model, args.receptor_net_rule)
    probe.patch_cache(args.cache_dir)
    probe.BALL_R, probe.AHEAD, probe.HALF_SWEEP = args.ball_radius, args.ahead, args.half_sweep
    rm, rule = probe.resolved_receptor()
    mode = "off" if rm is None else f"{rm}-{rule}"
    pargs = SimpleNamespace(settle=args.settle, seconds=args.seconds, null=args.null)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"interp_export record: object sweep{' NULL (none vs none)' if args.null else ''}; mode {mode}; seed {args.seed}; "
          f"window {args.seconds} s after {args.settle} s settle; ball r {probe.BALL_R} m at {probe.AHEAD} m "
          f"({2 * np.degrees(np.arctan(probe.BALL_R / probe.AHEAD)):.1f} deg), half sweep {probe.HALF_SWEEP} m; "
          f"cache {args.cache_dir or connectome.CACHE_DIR}; torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)
    sim_a, A = probe.run(not args.null, pargs, args.seed)
    n_sweep = int(round(args.seconds / probe.FRAME_S))
    retina = capture_retina(sim_a, probe, n_sweep, with_ball=not args.null) if args.retina else None
    c = sim_a.c                                            # the Connectome is CPU-side: keeping it frees no GPU memory
    fb = sim_a.fb
    kinds = common.unit_kinds(c, fb)                       # graded = this FlyBrain's optic rate units
    n_col, n_pr = int(len(fb.retina.col_dir)), int(len(fb.retina.pr_index))
    stimulus = {"protocol": "object_sweep", "generator": "scripts/probe_object_sweep.py::run (scripts/interp_export.py record)",
                "params": {"mode": mode, "receptor_model": rm, "receptor_net_rule": rule, "null": bool(args.null),
                           "condition_a": "none" if args.null else "ball", "condition_b": "none",
                           "ball_radius_m": probe.BALL_R, "ahead_m": probe.AHEAD, "half_sweep_m": probe.HALF_SWEEP,
                           "sweep_s": probe.SWEEP_S, "angular_diameter_deg": float(2 * np.degrees(np.arctan(probe.BALL_R / probe.AHEAD))),
                           "pos": list(probe.POS), "heading_rad": float(probe.HEADING), "seconds": args.seconds,
                           "settle": args.settle, "fruit_set": "apple (removed from the scene)", "fence": True,
                           "wind_speed": 0.0, "contrast": "black ball on the plain wall at y = -2"},
                "control": {"condition": "none (ball parked at (9, 9, 9))", "matched": "same seed, same code path"}}
    prov = common.provenance(c, lif=getattr(fb.brain, "p", None), optic=getattr(fb.optic, "p", None), fb=fb,
                             device=args.device, seeds=[args.seed], env_seeds=[args.seed], batch=1,
                             backend={"cuda_kernels": True, "event_driven": True, "cuda_sparse": "warp", "cuda_graphs": False},
                             stimulus=stimulus,
                             retina={"source_npz": str(out) + "_retina.npz" if args.retina else None,
                                     "n_columns": n_col, "n_photoreceptors": n_pr,
                                     "column_to_bodies": "retina_columns.csv:photoreceptor_bodies" if args.retina else None},
                             cache_dir=args.cache_dir)
    # a cluster job has no .git, so git_state() is 'unknown' here: hash the source files the job actually loaded and
    # let the CPU analysis step match them against the checkout (flyverse/interp/export.py::match_sources)
    prov["flyverse_commit"]["source_fingerprint"] = ex.source_fingerprint(
        ROOT, include_loaded=True, extra=[p for p in [getattr(fb.brain.p, "receptor_table", None)] if p])
    with open(str(out) + "_prov.json", "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable({"provenance": prov, "argv": sys.argv}), f, indent=1)
    del sim_a, fb                                          # free the GPU before the second rollout (as the probe does)
    torch.cuda.empty_cache()
    sim_b, B = probe.run(False, pargs, args.seed)
    cells = cell_arrays(c, kinds, A, B)
    del sim_b
    torch.cuda.empty_cache()
    S_a, S_b = probe.summarize(A, B), probe.summarize(B)
    cfg = {"mode": mode, "receptor_model": rm, "receptor_net_rule": rule, "receptor_model_flag": args.receptor_model,
           "rectify_t2t3": False, "null": bool(args.null), "condition_a": "none" if args.null else "ball",
           "condition_b": "none", "angular_diameter_deg": float(2 * np.degrees(np.arctan(probe.BALL_R / probe.AHEAD))),
           "seed": args.seed, "seconds": args.seconds, "settle": args.settle,
           "cache_dir": str(args.cache_dir or connectome.CACHE_DIR), "pos": list(probe.POS), "heading_rad": float(probe.HEADING),
           "ball_radius_m": probe.BALL_R, "ahead_m": probe.AHEAD, "half_sweep_m": probe.HALF_SWEEP, "sweep_s": probe.SWEEP_S,
           "pass_drive_mv": probe.PASS_DRIVE_MV, "pass_rate_hz": probe.PASS_RATE_HZ,
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    payload = {"config": cfg, "ball": S_a, "none": S_b, "radiance_check": A["radiance"],
               "window_s": A["window_s"], "n_frames": int(n_sweep)}
    with open(str(out) + ".json", "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(payload), f, indent=1)
    np.savez_compressed(str(out) + "_cells.npz", **{k: v for k, v in cells.items() if k != "types"},
                        types=np.array([str(t) for t in cells["types"]]))
    if retina is not None:
        np.savez_compressed(str(out) + "_retina.npz", **{k: v for k, v in retina.items() if k != "sampling"},
                            sampling=np.array(retina["sampling"]))
        print(f"retina: {retina['radiance'].shape[0]} frames x {retina['radiance'].shape[1]} columns x 4 channels; "
              f"{len(retina['pr_body'])} photoreceptors", flush=True)
    print("\nper type (condition A / condition B): drive mean mV, best-cell mean, rate Hz mean, diff max over cells")
    rows = [{"type": t, "n_cells": S_a[t]["n_cells"], "A_drive_mean_mv": S_a[t]["drive_mean_mv"],
             "B_drive_mean_mv": S_b[t]["drive_mean_mv"], "A_best_cell_mv": S_a[t]["drive_best_cell_mean_mv"],
             "A_rate_hz_mean": S_a[t]["rate_hz_mean"], "diff_max_over_cells_mean_mv": S_a[t]["diff_max_over_cells_mean_mv"]}
            for t in probe.SPIKING]
    common.print_table(pd.DataFrame(rows), floatfmt="{:+.4f}")
    print(f"written {out}.json, {out}_cells.npz, {out}_prov.json" + (f", {out}_retina.npz" if retina is not None else ""))
    return 0


# ------------------------------------------------------------------------------------------------- run (CPU)
SIDECARS = ("_prov.json",)                              # `record` writes these beside <out>.json; a glob must not count them


def _glob(patterns) -> list:
    """The probe JSONs a `--stim` / `--null-runs` pattern names, forward-slashed and WITHOUT the sidecars `record`
    writes beside them (`<out>_prov.json`): `out/expobj/stim_s*.json` matches `stim_s0_prov.json` too, and counting
    it would inflate the run count and the control_ids while contributing no arm value."""
    out = []
    for p in patterns or []:
        hits = sorted(glob.glob(p))
        out += hits if hits else ([p] if Path(p).exists() else [])
    return [Path(p).as_posix() for p in out if not any(str(p).endswith(s) for s in SIDECARS)]


def _sibling(paths, suffix) -> list:
    out = []
    for p in paths:
        q = Path(str(p)[:-len(".json")] + suffix)
        if q.exists():
            out.append(q.as_posix())
    return out


def _family(spec):
    """`--family`: '' / 'none' -> nothing declared (the default), a key of `export.FAMILY_SPECS`, or a JSON spec.

    A multiple-comparison family is a PREDECLARATION by the caller; the export never invents one, and `p_holm` never
    moves an unadjusted p or a verdict (docs/audits/interp_export.md revision 2)."""
    s = (spec or "").strip()
    if not s or s.lower() == "none":
        return None
    if s in ex.FAMILY_SPECS:
        return s
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    raise SystemExit(f"--family {spec!r}: not a key of {list(ex.FAMILY_SPECS)} and not a JSON spec")


def _geometry(stim_jsons) -> dict:
    """The ball geometry of a recording set, from the probe JSON's own config -- what `retina_object_track` needs."""
    if not stim_jsons:
        return {}
    with open(stim_jsons[0], encoding="utf-8") as f:
        cfg = json.load(f).get("config") or {}
    if cfg.get("ball_radius_m") is None or cfg.get("ahead_m") is None:
        return {}
    return {"ball_radius_m": float(cfg["ball_radius_m"]), "ahead_m": float(cfg["ahead_m"]),
            "eye_above_table_m": ex.EYE_ABOVE_TABLE_M}


def _size_geometry(size_id: str, stim_jsons, geom: dict) -> dict:
    """One ladder rung's size, in the column names the earlier export used plus the ones that say WHERE the numbers
    hold: the probe's nominal 2 atan(r / ahead), the exact 2 asin(r / distance) from the eye AT AZIMUTH 0, and the
    centre elevation there. Along the sweep both shrink (`retina_object_track` has them per frame)."""
    if not geom:
        return {"size_id": size_id}
    at0 = ex.object_track([0.0], **geom).iloc[0]
    with open(stim_jsons[0], encoding="utf-8") as f:
        cfg = json.load(f).get("config") or {}
    nominal = cfg.get("angular_diameter_deg")
    try:
        nominal = float(re.sub(r"^d", "", size_id)) / 10.0
    except ValueError:
        pass
    return {"size_id": size_id, "nominal_deg": nominal, **geom,
            "angular_diameter_deg_probe": float(cfg.get("angular_diameter_deg", float("nan"))),
            "angular_diameter_deg_from_eye": float(at0.angular_diameter_deg),
            "centre_elevation_deg_at_azimuth_0": float(at0.centre_elevation_deg),
            "distance_eye_to_centre_m_at_azimuth_0": float(at0.distance_eye_to_centre_m),
            "half_sweep_m": cfg.get("half_sweep_m"), "sweep_s": cfg.get("sweep_s")}


def _stamp_commit(res) -> dict:
    """Name the commit a cluster-recorded run ran, by matching the source hashes the record job wrote against this
    checkout. A GPU job runs from an rsynced tree with no `.git`, so its own `git_state()` is `commit 'unknown'`;
    when every file it loaded is byte-identical here, the run IS this checkout's commit and the manifest says so
    (`commit_verified`, with the per-file evidence kept)."""
    fc = res.provenance.setdefault("flyverse_commit", {})
    unknown = fc.get("commit") in (None, "", "unknown")
    fp = fc.get("source_fingerprint")
    if not fp:
        if unknown:
            fc["commit_verified"] = ("no source fingerprint recorded: a job that ran outside a git checkout cannot be "
                                     "pinned to a commit (flyverse/interp/export.py::source_fingerprint)")
            print("source match: the run recorded no source fingerprint; its commit stays 'unknown'", flush=True)
        return {}
    match = ex.match_sources(fp)
    fc["source_match"] = match
    if match["verified"] and unknown:
        fc["commit"] = match["analysis_git"]["commit"]
        fc["dirty"] = match["analysis_git"]["dirty"]
        fc["commit_verified"] = (f"by source hash ({match['scope']} scope): every one of the {match['n_recorded']} "
                                 "source files the run loaded holds the content of this checkout")
    elif unknown:
        fc["commit_verified"] = match["note"]
    print(f"source match ({match['scope']}): {match['n_identical']}/{match['n_recorded']} files identical to this "
          f"checkout" + (f", differ {match['differ']}" if match["differ"] else "") +
          f"; commit {fc.get('commit')} ({fc.get('commit_verified', 'recorded by the job itself')})", flush=True)
    return match


def cmd_run(args) -> int:
    """The recorded object-sweep runs -> one Result -> the export directory (CPU)."""
    stim = _glob(args.stim)
    nulls = _glob(args.null_runs)
    if not stim:
        sys.exit("no --stim runs found")
    cells, null_cells = _sibling(stim, "_cells.npz"), _sibling(nulls, "_cells.npz")
    prov = None
    for p in _sibling(stim, "_prov.json"):
        prov = p
        break
    res = ex.result_from_object_sweep(stim, nulls, cells=cells, null_cells=null_cells, provenance=prov,
                                      reference=_glob(args.reference), reference_null=_glob(args.reference_null),
                                      retina=args.retina, null_reference_ids=[Path(p).stem for p in nulls] or None,
                                      family=_family(getattr(args, "family", None)),
                                      generator="scripts/interp_export.py record + run")
    _stamp_commit(res)
    problems = res.check()
    if problems:
        print("Result.check():", "; ".join(problems))
    json_path = Path(args.json) if args.json else common.default_json_path("export", res.run_id)
    res.save(json_path)
    per = res.table("per_type")
    show = per[per.statistic == "diff_max_over_cells_mean_mv"][["type", "n_cells", "stim_n", "stim_mean", "stim_sd",
                                                                "null_n", "null_mean", "null_sd", "z", "p", "verdict"]]
    common.print_table(show, floatfmt="{:+.4f}")
    rep = (res.validation.get("measured") or {}).get("reproduction") or {}
    if rep:
        print("\nthis run vs the reference runs (diff_max_over_cells_mean_mv, mV):")
        common.print_table(pd.DataFrame([
            {"type": t, "ref_n": len(d["reference_runs"]), "ref_mean": d["reference_mean"], "ref_sd": d["reference_sd"],
             "ref_null_mean": d["reference_null_mean"], "ref_z": d["reference_z"], "n": len(d["reproduced_runs"]),
             "mean": d["reproduced_mean"], "sd": d["reproduced_sd"], "null_mean": d["null_mean"], "z": d["z"],
             "verdict": d["verdict"]} for t, d in rep.items()]), floatfmt="{:+.4f}")
    if problems:
        sys.exit(f"not exported: {problems}")
    blank = args.retina_blank or (_sibling(nulls, "_retina.npz") or [None])[0]
    run_dir = ex.export(res, out_root=args.out, retina=args.retina, retina_blank=blank,
                        retina_geometry=_geometry(stim), parquet_rows=args.parquet_rows)
    _report(run_dir, res, args)
    print(f"Result {json_path}\nexport {run_dir}")
    return 0


def _report(run_dir, res, args) -> None:
    """Print the manifest's table list and run the export's own validation (VALIDATION['export']): the round trip
    Result -> files -> the same numbers, and `export.verify` (hashes, decimal ids, ids in cache/neurons.parquet, the
    two-rows-per-body rule). Both land in `<run_dir>/checks.json`, which is written AFTER manifest.json and is
    therefore not one of the hashed tables -- it is a report of the checks, not part of the interchange."""
    trip = ex.round_trip_check(res, run_dir)
    info = ex.verify(run_dir, neurons=args.neurons if args.neurons else True,
                     expect_paired=tuple(t for t in (getattr(args, "paired", None) or "").split(",") if t.strip()),
                     expect_counts=json.loads(args.expect_counts) if args.expect_counts else None)
    with open(Path(run_dir) / "manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    with open(Path(run_dir) / "checks.json", "w", encoding="utf-8") as f:
        json.dump({"note": "written after manifest.json; not a hashed interchange table",
                   "round_trip": common.to_jsonable(trip), "verify": common.to_jsonable(info)}, f, indent=1)
    common.print_table(pd.DataFrame(man["tables"])[["name", "file", "format", "rows", "sha256"]], floatfmt="{:.4f}")
    print("round trip:", json.dumps(trip))
    print("verify:", json.dumps({k: v for k, v in info.items() if k != "problems"}))
    print("problems:", info["problems"] or "none")


def cmd_analyse(args) -> int:
    """Serialize an existing Result JSON (any tool) into the export directory (CPU)."""
    res = common.Result.load(args.result)
    _stamp_commit(res)                                     # a GPU-recorded Result says commit 'unknown': pin it here
    problems = res.check()
    if problems:
        sys.exit(f"the export refuses {args.result}: {problems}")
    run_dir = ex.export(res, out_root=args.out, run_id=args.run_id, retina=args.retina,
                        retina_blank=args.retina_blank, parquet_rows=args.parquet_rows,
                        control_ids=args.control_ids.split(",") if args.control_ids else None,
                        paired_control_ids=args.paired_control_ids.split(",") if args.paired_control_ids else None,
                        null_reference_ids=args.null_reference_ids.split(",") if args.null_reference_ids else None)
    _report(run_dir, res, args)
    print(f"export {run_dir}")
    return 0


def cmd_verify(args) -> int:
    """Re-check a finished export directory (CPU)."""
    info = ex.verify(args.run_dir, neurons=args.neurons if args.neurons else True,
                     expect_paired=tuple(t for t in (args.paired or "").split(",") if t.strip()),
                     expect_counts=json.loads(args.expect_counts) if args.expect_counts else None)
    print(json.dumps({k: v for k, v in info.items() if k != "problems"}, indent=1))
    print("problems:", info["problems"] or "none")
    return 1 if info["problems"] else 0


# ------------------------------------------------------------------------------------------------- ladder (CPU)
def ladder_runs(runs_csv=None, recordings_dir=None, sizes=None) -> dict:
    """{size_id: {'stim': [...], 'null': [...]}} of probe JSONs, from a previous summary export's `runs.csv` (its
    `size_id` / `arm` / `file` columns name every recording the ladder was built from) or from the recordings
    directory itself (`<dir>/<size>_<arm>_s*.json`). Re-exporting reads the recordings, never a finished export."""
    out = {}
    if runs_csv:
        tab = pd.read_csv(runs_csv, keep_default_na=False, na_values=[""])
        for r in tab.itertuples():
            if Path(str(r.file)).exists():
                out.setdefault(str(r.size_id), {}).setdefault(str(r.arm), []).append(Path(str(r.file)).as_posix())
            else:
                print(f"[{r.size_id}] missing recording {r.file}", flush=True)
    else:
        d = Path(recordings_dir)
        for p in sorted(d.glob("*_s*.json")):
            m = re.match(r"^(?P<sid>[^_]+)_(?P<arm>stim|null)_s\d+$", p.stem)
            if m:
                out.setdefault(m["sid"], {}).setdefault(m["arm"], []).append(p.as_posix())
    if sizes:
        keep = {s.strip() for s in sizes.split(",") if s.strip()}
        out = {k: v for k, v in out.items() if k in keep}
    return {k: {a: sorted(v) for a, v in arms.items()} for k, arms in sorted(out.items())}


def _footprint(stim_npz, blank_npz, geom: dict) -> dict:
    """One size's retinal footprint from the two replayed radiance records: the columns the ball dims, their extent,
    and the geometry that moved with the size (centre elevation at azimuth 0, angular diameter from the eye)."""
    a, b = np.load(stim_npz), np.load(blank_npz)
    ra, rb = a["radiance"].sum(-1), b["radiance"].sum(-1)
    rel = ra / np.maximum(rb[0][None], 1e-9) - 1.0
    hit = rel < -0.05
    any_col = np.flatnonzero(hit.any(0))
    az_el = a["col_az_el"]
    track = ex.object_track(a["ball_offset_m"], **geom) if geom else pd.DataFrame()
    out = {"n_columns": int(ra.shape[1]), "n_frames": int(ra.shape[0]),
           "columns_dimmed_5pct_any_frame": int(len(any_col)),
           "columns_dimmed_5pct_per_frame_mean": float(hit.sum(1).mean()),
           "columns_dimmed_50pct_per_frame_mean": float((rel < -0.5).sum(1).mean()),
           "min_relative_radiance": float(1 + rel.min()),
           "azimuth_deg_min": float(az_el[any_col, 0].min()) if len(any_col) else float("nan"),
           "azimuth_deg_max": float(az_el[any_col, 0].max()) if len(any_col) else float("nan"),
           "elevation_deg_min": float(az_el[any_col, 1].min()) if len(any_col) else float("nan"),
           "elevation_deg_max": float(az_el[any_col, 1].max()) if len(any_col) else float("nan"),
           "blank_radiance_identical_over_frames": bool(np.allclose(rb, rb[0][None])),
           "stim_sampling": str(a["sampling"]) if "sampling" in a else "",
           "blank_sampling": str(b["sampling"]) if "sampling" in b else ""}
    if len(track):
        out.update({"centre_elevation_deg_at_azimuth_0": float(track.centre_elevation_deg.max()),
                    "centre_elevation_deg_min": float(track.centre_elevation_deg.min()),
                    "angular_diameter_deg_max": float(track.angular_diameter_deg.max()),
                    "angular_diameter_deg_min": float(track.angular_diameter_deg.min()),
                    "centre_azimuth_deg_min": float(track.centre_azimuth_deg.min()),
                    "centre_azimuth_deg_max": float(track.centre_azimuth_deg.max())})
    return out


def cmd_ladder(args) -> int:
    """Re-export a recorded size ladder from the recordings on file: one NEW run directory per size plus a summary
    directory, never a write into an existing one (a run is a directory and is immutable, NEUROME_INTERFACE 1).

    Every number comes from the recordings, not from the earlier export: the probe JSONs give the arms, the
    `_cells.npz` the per-body rows, the `_retina.npz` of each arm the object AND blank radiance, and the probe's own
    config the ball geometry. The declared `--family` is applied to the ladder-wide `size_tuning` table, which is the
    only table where a family spanning the sizes can be formed."""
    runs = ladder_runs(args.runs_csv, args.dir, args.sizes)
    if not runs:
        sys.exit(f"no ladder recordings found ({args.runs_csv or args.dir})")
    family = _family(args.family)
    json_dir = Path(args.json_dir); json_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dirs, rows, foot, runs_tab, results = {}, [], [], [], {}
    for sid, arms in runs.items():
        stim, nul = arms.get("stim", []), arms.get("null", [])
        if not stim:
            print(f"[{sid}] no stimulus runs; skipped"); continue
        cells, ncells = _sibling(stim, "_cells.npz"), _sibling(nul, "_cells.npz")
        prov = (_sibling(stim, "_prov.json") or [None])[0]
        ret = (_sibling(stim, "_retina.npz") or [None])[0]
        ret_blank = (_sibling(nul, "_retina.npz") or [None])[0]
        geom = _geometry(stim)
        res = ex.result_from_object_sweep(
            stim, nul, cells=cells, null_cells=ncells, provenance=prov, retina=ret, family=None,
            generator="scripts/interp_export.py ladder (re-export of the recorded size ladder)",
            run_id=f"objsize-{sid}-{stamp}-{os.urandom(4).hex()}")
        size_geom = _size_geometry(sid, stim, geom)
        res.provenance["stimulus"]["size_geometry"] = common.to_jsonable(size_geom)
        res.summary["size_geometry"] = common.to_jsonable(size_geom)
        res.provenance["retina"] = {**(res.provenance.get("retina") or {}),
                                    "blank_npz": ret_blank, "object_npz": ret}
        _stamp_commit(res)
        problems = res.check()
        if problems:
            print(f"[{sid}] Result.check(): {problems}")
        res.save(json_dir / f"ladder_{sid}_{stamp}.json")
        run_dir = ex.export(res, out_root=args.out, retina=ret, retina_blank=ret_blank, retina_geometry=geom,
                            parquet_rows=args.parquet_rows)
        _report(run_dir, res, SimpleNamespace(neurons=args.neurons, expect_counts=args.expect_counts,
                                              paired=args.paired))
        run_dirs[sid] = str(run_dir); results[sid] = res
        per = res.table("per_type")
        geo_cols = {k: v for k, v in (res.summary.get("size_geometry") or {}).items() if k != "size_id"}
        for r in per.to_dict("records"):
            rows.append({"size_id": sid, **geo_cols, **r, "export_run_dir": str(run_dir)})
        runs_tab += [{"size_id": sid, "arm": "stimulus", "record_id": ex._run_stem(p), "file": p,
                      "reference_role": "object arm (a) and, as `paired_control_ids`, its own blank arm (b)"} for p in stim]
        runs_tab += [{"size_id": sid, "arm": "null", "record_id": ex._run_stem(p), "file": p,
                      "reference_role": "independent blank/blank reference run (`null_reference_ids`)"} for p in nul]
        if ret and ret_blank:
            foot.append({"size_id": sid, **_footprint(ret, ret_blank, geom), "stim_retina": ret, "null_retina": ret_blank})
        print(f"[{sid}] {len(stim)} stim + {len(nul)} null runs -> {run_dir}", flush=True)
    if not run_dirs:
        sys.exit("nothing to summarise")
    tab = ex.add_family_columns(pd.DataFrame(rows), family)      # the family spans the ladder, so it is applied here
    first = results[list(run_dirs)[0]]
    prov = dict(first.provenance)
    # the summary carries no radiance of its own: one size's object arm here would be a mislabelled retina record
    prov["retina"] = {"source_npz": None, "per_size_exports": run_dirs,
                      "summary_note": "the retinal record of each size -- retina_radiance, retina_radiance_blank, "
                                      "retina_object_track and the mode block -- lives in that size's run directory; "
                                      "this summary carries only the reduced `retina_footprint` table"}
    prov["stimulus"] = {"protocol": "object_sweep size ladder",
                        "params": {"sizes": [results[s].summary.get("size_geometry") or {"size_id": s} for s in run_dirs],
                                   "runs_per_arm": {s: len(runs[s].get("stim", [])) for s in run_dirs}},
                        "control": "per size: the matched blank arm (b) of each stimulus recording for every "
                                   "`control_value` / `diff_*`, and the independent blank/blank runs of the same "
                                   "protocol for every null column (see conventions.controls)",
                        "per_size_exports": run_dirs}
    out = common.Result.new("export", prov)
    out.add_table("size_tuning", tab)
    out.add_table("retina_footprint", foot)
    out.add_table("runs", runs_tab)
    key = tab[tab.statistic.isin(["diff_max_over_cells_mean_mv", "diff_abs_best_cell_mean"])]
    out.summary = {"per_size_exports": run_dirs, "family": family if isinstance(family, (str, type(None))) else "custom",
                   "p_methods": tab.p_method.value_counts().to_dict(),
                   "n_rows_with_ties": int((tab.n_tied_values > 0).sum()),
                   "lc_tuning": {t: {r.size_id: {"z": float(r.z), "stim_mean": float(r.stim_mean),
                                                 "null_mean": float(r.null_mean), "null_sd": float(r.null_sd),
                                                 "p": float(r.p), "p_method": r.p_method, "p_holm": float(r.p_holm),
                                                 "verdict": r.verdict, "statistic": r.statistic}
                                     for r in key[key.type == t].itertuples()}
                                 for t in ["LC11", "LC10a", "T2", "T3", "Tm5Y", "TmY21", "LPLC2"] if t in set(key.type)}}
    out.replicates = {"n": int(np.median([len(runs[s].get("stim", [])) for s in run_dirs])), "unit": "runs",
                      "runs": runs_tab, "null": {"per_size": True}}
    out.files = {"generator": " ".join(sys.argv), "per_size_results": {s: str(json_dir / f"ladder_{s}_{stamp}.json") for s in run_dirs},
                 "source": args.runs_csv or args.dir}
    out.validation = dict(out.validation, status="not run",
                          measured={"note": "each per-size run directory carries its own round-trip and verify report "
                                            "in checks.json; this summary re-reduces their per_type tables"})
    _stamp_commit(out)
    jp = Path(args.json) if args.json else json_dir / f"ladder_summary_{stamp}.json"
    out.save(jp)
    sum_dir = ex.export(out, out_root=args.out, parquet_rows=args.parquet_rows)
    _report(sum_dir, out, SimpleNamespace(neurons=args.neurons, expect_counts=None, paired=""))
    show = key[key.statistic == "diff_max_over_cells_mean_mv"]
    common.print_table(show[["size_id", "type", "stim_mean", "null_mean", "z", "p", "p_method", "p_holm", "verdict"]],
                       floatfmt="{:+.4f}", max_rows=60)
    if foot:
        common.print_table(pd.DataFrame(foot)[["size_id", "columns_dimmed_5pct_per_frame_mean",
                                               "columns_dimmed_50pct_per_frame_mean", "centre_elevation_deg_at_azimuth_0",
                                               "angular_diameter_deg_max", "min_relative_radiance"]], floatfmt="{:.2f}")
    print(f"\nsummary Result {jp}\nsummary export {sum_dir}")
    for sid, rd in run_dirs.items():
        print(f"  {sid}: {rd}")
    return 0


def cmd_static_decompose(args) -> int:
    """The structural decomposition of a target's input -> a Result -> the export (CPU; no rollout)."""
    from flyverse import brain, connectome as cn          # noqa: PLC0415
    lif, _optic = common.params_from_args(args)
    c = cn.load(cache_dir=Path(args.cache_dir)) if args.cache_dir else cn.load()
    receptor = brain._receptor(c, lif)                     # the model's own lookup (None when receptor_model is off)
    res = ex.static_decompose_result(c, args.target, args.pre, params=lif, receptor=receptor, cache_dir=args.cache_dir)
    res.files = {"generator": f"scripts/interp_export.py static-decompose --target {args.target!r}"}
    json_path = Path(args.json) if args.json else common.default_json_path("decompose", res.run_id)
    res.save(json_path)
    per = res.table("per_type")
    common.print_table(per.head(args.top), floatfmt="{:+.4f}")
    run_dir = ex.export(res, out_root=args.out, parquet_rows=args.parquet_rows)
    _report(run_dir, res, args)
    print(f"Result {json_path}\nexport {run_dir}")
    return 0


# ------------------------------------------------------------------------------------------------- argv
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common_export_args(p):
        p.add_argument("--out", default="out/export", help="export root (the run directory is <out>/<run_id>)")
        p.add_argument("--parquet-rows", type=int, default=1_000_000, help="tables above this many rows go to Parquet")
        p.add_argument("--neurons", default=None, help="cache/neurons.parquet to check bodyIds against (default: the cache)")
        p.add_argument("--expect-counts", default=None, help='JSON {"LC11": 143, "LC10a": 275} the verifier must find')
        p.add_argument("--paired", default="LC11,LC10a", help="types whose every body must carry BOTH quantities "
                       "(upstream_drive_mV and output_Hz) in readout_per_body; '' turns the check off")
        p.add_argument("--family", default=None, help=f"multiple-comparison family to predeclare for `p_holm`: a key "
                       f"of {list(ex.FAMILY_SPECS)}, a JSON spec, or nothing (the default: no family, p_holm empty)")

    rec = sub.add_parser("record", help="GPU: run the object-sweep protocol and capture per-body readouts + the retina")
    common.add_common_args(rec)
    rec.add_argument("--protocol", default="object_sweep", choices=["object_sweep"])
    rec.add_argument("--seconds", type=float, default=12.0)
    rec.add_argument("--settle", type=float, default=3.0)
    rec.add_argument("--ball-radius", type=float, default=0.005)
    rec.add_argument("--ahead", type=float, default=0.05)
    rec.add_argument("--half-sweep", type=float, default=0.06)
    rec.add_argument("--null", action="store_true", help="record the matched control-vs-control arm (none vs none) "
                     "instead of the stimulus arm; the bare switch belongs to `record`, --null-runs to the analysis")
    rec.add_argument("--retina", action="store_true", help="also capture the retinal sampling presented (frames x 1,466 columns)")
    rec.add_argument("--out", required=True, help="output prefix (<out>.json, <out>_cells.npz, <out>_prov.json, <out>_retina.npz)")
    rec.set_defaults(func=cmd_record)

    run = sub.add_parser("run", help="CPU: recorded object-sweep runs -> Result -> export")
    common.add_common_args(run)
    run.add_argument("--stim", nargs="+", required=True, help="probe JSONs of the stimulus runs (globs allowed)")
    run.add_argument("--reference", nargs="*", default=[], help="earlier runs of the same protocol, globs allowed "
                                                                "(out/r3obj/ball_off_s*.json)")
    run.add_argument("--reference-null", nargs="*", default=[], help="their nulls (out/r3obj/null_off_s*.json)")
    run.add_argument("--retina", default=None, help="the retina npz of one of the stimulus runs")
    run.add_argument("--retina-blank", default=None, help="the retina npz of the MATCHED blank run (default: the "
                     "sibling of the first --null-runs entry); written as retina_radiance_blank")
    common_export_args(run)
    run.set_defaults(func=cmd_run)

    an = sub.add_parser("analyse", help="CPU: serialize a Result JSON into the export")
    common.add_common_args(an)
    an.add_argument("--result", required=True)
    an.add_argument("--run-id", default=None)
    an.add_argument("--retina", default=None)
    an.add_argument("--retina-blank", default=None, help="the matched blank arm's retina npz (retina_radiance_blank)")
    an.add_argument("--control-ids", default=None, help="DEPRECATED spelling of --null-reference-ids (what this flag "
                    "has always held); kept for one revision")
    an.add_argument("--paired-control-ids", default=None, help="comma-separated record/arm ids the `control_value` "
                    "column was computed from (arm b of the stimulus recordings)")
    an.add_argument("--null-reference-ids", default=None, help="comma-separated ids of the INDEPENDENT blank/blank "
                    "runs behind null_mean / z_vs_null / verdict")
    common_export_args(an)
    an.set_defaults(func=cmd_analyse)

    lad = sub.add_parser("ladder", help="CPU: re-export a recorded size ladder into NEW run directories + a summary")
    lad.add_argument("--runs-csv", default=None, help="runs.csv of an earlier summary export (size_id / arm / file)")
    lad.add_argument("--dir", default=None, help="the recordings directory instead (<dir>/<size>_<arm>_s*.json)")
    lad.add_argument("--sizes", default=None, help="comma-separated size ids to re-export (default: all found)")
    lad.add_argument("--json", default=None, help="the summary Result JSON (default: <json-dir>/ladder_summary_<utc>.json)")
    lad.add_argument("--json-dir", default="out/interp/export", help="where the per-size Result JSONs are written")
    common_export_args(lad)
    lad.set_defaults(func=cmd_ladder)

    ver = sub.add_parser("verify", help="CPU: re-check a finished export directory")
    ver.add_argument("--run-dir", required=True)
    ver.add_argument("--neurons", default=None)
    ver.add_argument("--paired", default="LC11,LC10a", help="types that must carry both quantities per body")
    ver.add_argument("--expect-counts", default=None)
    ver.set_defaults(func=cmd_verify)

    sd = sub.add_parser("static-decompose", help="CPU: structural decomposition of a target -> contributions -> export")
    common.add_common_args(sd)
    sd.add_argument("--target", required=True, help="population spec (docs/INTERP.md 2.1)")
    sd.add_argument("--pre", default=None, help="presynaptic spec (default: every cell with an entry onto the target)")
    sd.add_argument("--top", type=int, default=40)
    common_export_args(sd)
    sd.set_defaults(func=cmd_static_decompose)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"record", "run", "analyse", "verify", "static-decompose", "ladder", "-h", "--help"}
    if not argv or argv[0] not in known:
        argv = ["analyse"] + argv                          # docs/INTERP.md section 7: `--result ... --out out/export`
    args = build_parser().parse_args(argv)
    if not hasattr(args, "func"):
        build_parser().print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
