"""Round-2 MATCHED-baseline export for Neurome: the matched sphere ladder through the REVISED probe exporter.

What this ships (docs/NEUROME_INTERFACE.md section 1 + 3/3b, docs/audits/interp_export.md revision 2):

  * ONE run directory per rung of the matched sphere ladder (`out/export/objr2-<lobe>-d<rung>-<ts>-<id>/`) and one
    ladder summary directory, written by `flyverse.interp.export.export` -- this module computes no comparison of its
    own: every arm, z, U, p, verdict and Holm value it writes was produced by `scripts/object_round2_baseline.py
    analyse` (the predeclared analysis, `out/interp/objr2/baseline.json`) and is copied through.
  * `readout_per_body` with the per-body time-mean rows Neurome asked for: two quantities per spiking body
    (`upstream_drive_mV`, `output_Hz`) and two per graded unit (`rate_deviation`, `abs_rate_deviation`), LC11 and
    LC10a never pooled, T2 / T3 (and the other recorded medulla types) in the same table, each row carrying BOTH
    references apart -- `paired_control_ids` (arm b of the same recording) and `null_reference_ids` (the independent
    blank/blank runs) -- plus the RF-window columns of the predeclared window rule.
  * the IN-LOOP captured radiance of BOTH arms: `retina_radiance` (object) and `retina_radiance_blank` (the matched
    blank), identical schema, `retina.mode = "in_loop_capture"` (the round-1 ladder shipped a geometry REPLAY and no
    blank radiance at all -- Neurome's intake, 'The retinal comparison is not yet a controlled size-tuning assay').
  * `retina_object_track`: the ball's azimuth / elevation / angular diameter / distance per frame as the probe
    measured them IN the loop, with the empirical dimmed-column counts computed from the two radiance tables.
  * the per-body RF map as a table (`rf_map`: the localizer fit, the anatomical fallback and the window each body
    actually got), the per-body sweep-locked time courses (`time_course`) before any population maximum, and the
    predeclared-family Holm results (`size_tuning` in the summary directory, `family` / `p_holm` columns).

    PYTHONIOENCODING=utf-8 python scripts/object_round2_export.py ladder --out out/objr2 --baseline out/interp/objr2/baseline.json
    PYTHONIOENCODING=utf-8 python scripts/object_round2_export.py compare --out out/objr2c --baseline out/interp/objr2c/compare.json
    PYTHONIOENCODING=utf-8 python scripts/object_round2_export.py tables --run-dir out/export/<id> [--json out/export/<id>_tables.json]
    PYTHONIOENCODING=utf-8 python scripts/object_round2_export.py verify --run-dir out/export/<id>

CPU only (no torch is needed for the export itself; `flyverse.retina` is imported for the photoreceptor spectral
table and does import torch, so a desktop run wants CUDA_VISIBLE_DEVICES=-1). Nothing under flyverse/ is edited.

Two things this module does that the shipped exporter cannot do for it, both deliberate and both recorded in
`docs/audits/object_export_r2.md`:
  1. `export.object_track()` models the OLD probe's geometry (a ball resting on the table sliding along a lateral
     line), which cannot express the matched ARC (constant elevation, constant distance, constant angular speed). The
     track is therefore taken from the probe's own in-loop measurement and written with `export.write_table` under
     the contract name `retina_object_track`, its manifest entry appended to `manifest.json` after `export()`
     returns. `export.verify()` re-reads and re-hashes it like any other table.
  2. `export.STATISTIC_DEFINITIONS` has no entry for this round's statistics (`drive_median`, `spikes_median`, ...),
     so the definitions are registered into that dict at runtime (the module file is NOT edited) and are also carried
     per row in a `statistic_definition` column, so they survive with the table if the registration ever goes away.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flyverse.interp import common                         # noqa: E402  (CPU-only)
from flyverse.interp import export as ex                    # noqa: E402  (CPU-only, no torch)
import object_round2_baseline as b2                         # noqa: E402  (the predeclared analysis: CPU-only)
import probe_object_matched as pom                          # noqa: E402  (CPU-only module constants / helpers)

SPIKING, RATE = tuple(pom.SPIKING), tuple(pom.RATE)
LC_TYPES = b2.LC_TYPES                                      # ('LC11', 'LC10a') -- the two populations, never pooled
# The predeclared RF window is a property of the PRIMARY statistic and is applied to the two LC populations only;
# the upstream types (T2 / T3 / Tm5Y / TmY21) are predeclared WHOLE-WINDOW (out/objr2/predeclared.json,
# secondary_exploratory.upstream), and their per-body rows are the probe's own whole-window means.
WIN_TYPES = LC_TYPES
EXPECT_COUNTS = {"LC11": 143, "LC10a": 275}
FRAME_S = pom.FRAME_S

# probe_object_matched's per-body quantity -> (the interchange quantity name, unit, unit_kind, the window unit)
QUANTITY = {
    "drive_mv":     ("upstream_drive_mV", "mV", "spiking", "mV"),
    "rate_hz":      ("output_Hz", "Hz", "spiking", "spikes (sum over the window frames)"),
    "optic_dr":     ("rate_deviation", "rate units [0-1]", "graded", "rate units [0-1]"),
    "optic_dr_abs": ("abs_rate_deviation", "rate units [0-1]", "graded", "rate units [0-1]"),
}

# ---------------------------------------------------------------------------------------------------------------
# The statistics of THIS round, one sentence each. Registered into export.STATISTIC_DEFINITIONS at import time (the
# module file is not edited) so the manifest defines every `statistic` its tables use, and carried per row as well.
STATISTIC_DEFINITIONS = {
    "drive_median": "PREDECLARED PRIMARY. Per run: the MEDIAN over the bodies of the type (bodies with at least one "
                    "window frame) of that body's RF-windowed time-mean object-minus-blank received optic drive "
                    "(mV), the blank being arm b of the SAME recording. Compared with the same statistic in "
                    "independent blank/blank runs of the same lobe, windowed with the same rung's diameter. A "
                    "difference of time-means over a per-body window, never an absolute membrane voltage.",
    "spikes_median": "PREDECLARED PRIMARY. Per run: the MEDIAN over the bodies of the type of that body's RF-windowed "
                     "object-minus-blank SPIKE COUNT (spikes summed over the window frames), against the same "
                     "statistic in independent blank/blank runs.",
    "drive_mean": "Exploratory: the MEAN over the bodies of the type of the RF-windowed time-mean object-minus-blank "
                  "drive (mV), against the same statistic in blank/blank runs.",
    "drive_max": "Exploratory: the MAXIMUM over the bodies of the type of the RF-windowed time-mean object-minus-blank "
                 "drive (mV) -- a within-run maximum over cells, so the maximising cell may differ from run to run.",
    "spikes_mean": "Exploratory: the mean over bodies of the RF-windowed object-minus-blank spike count.",
    "spikes_max": "Exploratory: the maximum over bodies of the RF-windowed object-minus-blank spike count.",
    "n_bodies_windowed": "How many bodies of the type had at least one frame in their RF window at this rung (the "
                         "denominator of every `*_median` / `*_mean` above).",
}
# window columns of readout_per_body (documented in the manifest through `column_definitions`, a tool table)
COLUMN_DEFINITIONS = {
    "stimulus_value": "the object arm (a) of the recording, time-averaged over the whole analysis window, then "
                      "averaged over the runs of this rung",
    "control_value": "the matched blank arm (b) of the SAME recording (`paired_control_ids`), the same way",
    "stimulus_minus_control": "object minus its own matched blank, per run, averaged over runs",
    "trial_sd": "SD over runs of `stimulus_minus_control` (the replicate unit is the run)",
    "null_mean": "the same object-minus-blank difference in the INDEPENDENT blank/blank runs (`null_reference_ids`), "
                 "averaged over those runs -- arm a of a null run is a second blank",
    "null_sd": "SD over the null runs of that difference",
    "z_vs_null": "(stimulus_minus_control - null_mean) / null_sd; a per-BODY z on a null SD from 6 runs, not a "
                 "population verdict -- the verdict of record is the per-type arm in `per_type` / `size_tuning`",
    "window_source": "which rule gave this body its RF window (PREDECLARED, out/objr2/predeclared.json): `rf_ship` a "
                     "fit in the shipped-lobe localizer, `rf_fb0` a fit in the fb0 localizer, `anat` the anatomical "
                     "column of trace.column_of_cells with the type's median fitted box, `whole` no window (the whole "
                     "sweep). Empty = the window rule was not applied to this type.",
    "window_n_frames": "frames of the sweep on which the object overlapped this body's window box",
    "window_stimulus_minus_control": "object minus its own matched blank over the body's window frames only, averaged "
                                     "over runs; `window_unit` says the unit (mV for drive, spike COUNT for output_Hz)",
    "window_null_mean": "the same windowed difference in the independent blank/blank runs (same window, no object)",
    "paired_control_ids": "the recordings whose arm b produced `control_value` (id suffix '#arm_b')",
    "null_reference_ids": "the independent blank/blank runs behind `null_mean` / `null_sd` / `z_vs_null`",
}
ex.STATISTIC_DEFINITIONS.update(STATISTIC_DEFINITIONS)      # runtime registration; flyverse/interp/export.py untouched

PRIMARY_FAMILY = {
    "name": "predeclared_primary",
    "where": {"type": list(LC_TYPES), "statistic": ["drive_median", "spikes_median"],
              "role": ["primary"], "lobe": ["ship"]},
    "by": ["type"],
}
PRIMARY_FAMILY_NOTE = ("the predeclared primary family of out/objr2/predeclared.json (stamped 2026-09-13T21:45:39Z, "
                       "before the batch was submitted): per LC type, 6 rungs x {drive_median, spikes_median} = 12 "
                       "members, Holm within. It is applied where the whole family is present -- the ladder summary's "
                       "`size_tuning` -- and not inside a per-rung directory, where only 2 of the 12 members exist.")


# ==================================================================================================== small helpers
def sha256_file(p) -> str:
    return ex.sha256_file(p)


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def stamp_commit(res: common.Result) -> dict:
    """Name the commit a cluster-recorded run ran by matching the source hashes the job wrote against this checkout
    (`export.match_sources`). A cluster job has no `.git`, so its own provenance says `commit unknown`; when every
    loaded file is identical here the manifest says which commit it was and HOW that is known."""
    fc = res.provenance.setdefault("flyverse_commit", {})
    unknown = fc.get("commit") in (None, "", "unknown")
    # `scripts/interp_export.py record` writes the fingerprint INSIDE flyverse_commit; `common.provenance` (which is
    # what every probe uses, this batch included) writes it as its own top-level provenance block. Take either.
    fp = fc.get("source_fingerprint") or res.provenance.get("source_fingerprint")
    if fp and not (fp.get("files") or fp.get("files_loaded")):
        fp = None
    if not fp:
        if unknown:
            fc["commit_verified"] = "no source fingerprint recorded: the run cannot be pinned to a commit"
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
    print(f"  source match ({match['scope']}): {match['n_identical']}/{match['n_recorded']} identical"
          + (f", differ {match['differ']}" if match["differ"] else "") + f"; commit {fc.get('commit')}", flush=True)
    return match


# ==================================================================================================== the retina record
_NEURONS = None


def neurons_table(cache_dir=None) -> pd.DataFrame:
    """cache/neurons.parquet (bodyId, type) with the model index = the row position, which is what every tool means
    by `model_index` (flyverse/connectome.py builds its matrices in this order)."""
    global _NEURONS
    if _NEURONS is None:
        p = Path(cache_dir or (ROOT / "cache")) / "neurons.parquet"
        df = pd.read_parquet(p, columns=["bodyId", "type"])
        df["model_index"] = np.arange(len(df), dtype=np.int64)
        df["bodyId"] = df.bodyId.astype(np.int64)
        _NEURONS = df
    return _NEURONS


def retina_maps(prov_retina: dict, col_az_el: np.ndarray, col_side: np.ndarray, cache_dir=None) -> dict:
    """The column / photoreceptor map the exporter's `retina_tables` needs, rebuilt on the CPU from what the probe
    recorded plus `cache/neurons.parquet`.

    `scripts/probe_object_matched.py` stores the in-loop radiance and the column azimuth / elevation but not the
    photoreceptor table, so `column_to_bodies` (its provenance `retina` block, written by
    `flyverse.interp.trace.retina_record`) gives column -> bodyIds, the cache gives each body's type and model index,
    and `flyverse.retina.SENSITIVITY` the four spectral sensitivities. `col_dir` is the unit view direction
    reconstructed from (azimuth, elevation) by flyverse/retina.py's own convention
    [cos(el)cos(az), cos(el)sin(az), sin(el)] -- exactly what `Retina.ray_directions` builds."""
    from flyverse.retina import SENSITIVITY                 # noqa: PLC0415  (imports torch: CPU is fine)
    c2b = prov_retina.get("column_to_bodies") or {}
    if not c2b:
        raise SystemExit("the run's provenance carries no retina.column_to_bodies: cannot build the retina tables")
    n_col = int(len(col_az_el))
    nrn = neurons_table(cache_dir)
    idx_of = dict(zip(nrn.bodyId.to_numpy(), nrn.model_index.to_numpy()))
    type_of = dict(zip(nrn.bodyId.to_numpy(), nrn.type.fillna("").to_numpy()))
    bodies, cols = [], []
    for col, ids in sorted(((int(k), v) for k, v in c2b.items()), key=lambda kv: kv[0]):
        for b in ids:
            bodies.append(int(b)); cols.append(col)
    pr_body = np.asarray(bodies, np.int64); pr_column = np.asarray(cols, np.int64)
    missing = [b for b in pr_body if b not in idx_of]
    if missing:
        raise SystemExit(f"{len(missing)} photoreceptor bodies are not in cache/neurons.parquet (e.g. {missing[:3]})")
    pr_index = np.asarray([idx_of[b] for b in pr_body], np.int64)
    pr_type = np.asarray([type_of[b] for b in pr_body], dtype="<U16")
    unknown = sorted({t for t in pr_type if t not in SENSITIVITY})
    if unknown:
        raise SystemExit(f"photoreceptor types with no spectral sensitivity: {unknown}")
    pr_sens = np.asarray([SENSITIVITY[t] for t in pr_type], np.float32)
    az = np.radians(np.asarray(col_az_el, float)[:, 0]); el = np.radians(np.asarray(col_az_el, float)[:, 1])
    col_dir = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)
    col_hex = np.asarray(prov_retina.get("col_hex", np.full((n_col, 2), -1.0)), float)
    return {"col_dir": col_dir, "col_az_el": np.asarray(col_az_el, float), "col_hex": col_hex,
            "col_side": np.asarray(col_side).astype(str), "pr_index": pr_index, "pr_body": pr_body,
            "pr_column": pr_column, "pr_sens": pr_sens, "pr_type": pr_type, "n_columns": n_col}


def radiance_records(z, maps: dict, sampling: str) -> tuple[dict, dict]:
    """The object and matched-blank radiance in the one npz-like shape `export.retina_tables` reads.

    `sampling` travels as a 0-d numpy array on purpose: `export._as_npz` drops every scalar entry of the record it is
    given (`if not np.isscalar(v)`), so a plain `str` would be lost and the manifest would fall back to the generic
    default text instead of the probe's own statement of what it captured."""
    t_s = np.asarray(z["t_s"], float)
    s = np.asarray(str(sampling))
    obj = dict(maps, radiance=np.asarray(z["rad_a"], np.float32), t_s=t_s, sampling=s)
    blank = dict(maps, radiance=np.asarray(z["rad_b"], np.float32), t_s=t_s, sampling=s)
    for d in (obj, blank):
        d.pop("n_columns", None)
    return obj, blank


def object_track_table(z, geom: dict, blank_frame0: bool = False) -> tuple[pd.DataFrame, dict]:
    """`retina_object_track` for the MATCHED arc, from the probe's own in-loop measurement.

    `export.object_track()` cannot produce this: it models the old probe's geometry (a ball resting on the table and
    sliding along a lateral LINE, so elevation and angular diameter move with the radius and along the sweep). Here
    the columns are the values `scripts/probe_object_matched.py` measured from the eye every frame (`seen_*`) beside
    the intended trajectory (`track_*`), plus the empirical footprint computed from the two captured radiance arrays.

    The empirical columns use the MATCHED blank frame (`rad_b[j]`), not the blank's first frame as
    `export.retina_tables` does; the returned report says by how much the two differ (the blank is a static scene with
    the fly pinned, so it is 0 unless something moved)."""
    ra = np.asarray(z["rad_a"], np.float64).sum(-1)
    rb = np.asarray(z["rad_b"], np.float64).sum(-1)
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]
    rel_frame = ra / np.maximum(rb, 1e-9) - 1.0
    rel_zero = ra / np.maximum(rb[0][None], 1e-9) - 1.0
    rel = rel_zero if blank_frame0 else rel_frame
    az_el = np.asarray(z["col_az_el"], float)
    hit = rel < -0.05
    w = np.where(hit, -rel, 0.0)
    tot = w.sum(1)
    t = pd.DataFrame({
        "frame": np.arange(n), "t_s": np.asarray(z["t_s"], float)[:n],
        "centre_azimuth_deg": np.asarray(z["seen_az_deg"], float)[:n],
        "centre_elevation_deg": np.asarray(z["seen_el_deg"], float)[:n],
        "angular_diameter_deg": np.asarray(z["seen_diam_deg"], float)[:n],
        "distance_eye_to_centre_m": np.asarray(z["seen_dist_m"], float)[:n],
        "intended_azimuth_deg": np.asarray(z["track_az_deg"], float)[:n],
        "intended_elevation_deg": np.asarray(z["track_el_deg"], float)[:n],
        "intended_diameter_deg": np.asarray(z["track_diam_deg"], float)[:n],
        "angular_speed_deg_s": np.asarray(z["track_speed_deg_s"], float)[:n],
        "turn_frame": np.asarray(z["track_turn"], bool)[:n],
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        t["columns_dimmed_5pct"] = hit.sum(1)
        t["columns_dimmed_50pct"] = (rel < -0.5).sum(1)
        t["min_relative_radiance"] = 1.0 + rel.min(1)
        t["dimmed_centroid_azimuth_deg"] = np.where(tot > 0, w @ az_el[:, 0] / np.maximum(tot, 1e-12), np.nan)
        t["dimmed_centroid_elevation_deg"] = np.where(tot > 0, w @ az_el[:, 1] / np.maximum(tot, 1e-12), np.nan)
    report = {
        "definition": "per frame, as the probe measured it IN the loop: the ball's azimuth / centre elevation / "
                      "angular diameter / distance from the eye (`centre_*`, `seen_*` in the run npz) beside the "
                      "intended arc (`intended_*`), and the columns the object actually dimmed, computed from "
                      "retina_radiance against retina_radiance_blank frame by frame.",
        "why_not_export_object_track": "flyverse.interp.export.object_track models the OLD probe's geometry (a ball "
                                       "resting on the table sliding along a lateral line): it cannot express a "
                                       "constant-elevation, constant-distance, constant-speed arc, so the track is "
                                       "the probe's measurement, written through export.write_table under the same "
                                       "contract name and hashed into the manifest by this script.",
        "geometry": common.to_jsonable(geom),
        "blank_is_static": bool(np.allclose(rb, rb[0][None])),
        "max_abs_diff_vs_blank_frame0_convention": float(np.nanmax(np.abs(rel_frame - rel_zero))),
        "blank_reference": "the matched blank frame of the same frame index (an in-loop capture of both arms)",
    }
    return t, report


# ==================================================================================================== per-body tables
def per_body_whole(runs: pd.DataFrame) -> pd.DataFrame:
    """Every run's per-body whole-window table (the probe's own `per_body`) stacked, with the interchange quantity
    names. One row per (run, body, quantity): `a_mean` the object arm, `b_mean` its matched blank arm."""
    frames = []
    for _, r in runs.iterrows():
        d = _load(r.json)
        pb = pd.DataFrame(d["tables"]["per_body"])
        pb["run"] = r.stem
        frames.append(pb)
    df = pd.concat(frames, ignore_index=True)
    q = df.quantity.map({k: v[0] for k, v in QUANTITY.items()})
    if q.isna().any():
        raise SystemExit(f"unmapped per-body quantities: {sorted(set(df.quantity[q.isna()]))}")
    df["quantity"] = q
    df["unit"] = df.quantity.map({v[0]: v[1] for v in QUANTITY.values()})
    return df


def window_mask(z, win_row, obj_w: float, obj_h: float, n: int) -> np.ndarray:
    if win_row is None or str(win_row["window_source"]) == "whole":
        return np.ones(n, bool)
    return b2.box_frames(np.asarray(z["track_az_deg"], float)[:n], np.asarray(z["track_el_deg"], float)[:n],
                         obj_w, obj_h, float(win_row["az_deg"]), float(win_row["el_deg"]),
                         float(win_row["width_deg"]), float(win_row["height_deg"]))


def per_body_windowed(runs: pd.DataFrame, win: pd.DataFrame, obj_w: float, obj_h: float,
                      types=WIN_TYPES) -> pd.DataFrame:
    """Per (run, body, quantity) over the body's RF window: the object arm, its matched blank arm, the difference and
    how many frames the window held. The window rule is the predeclared one (`scripts/object_round2_baseline.py`
    `load_windows` / `box_frames`), applied here with the rung's angular size."""
    wi = win.set_index("bodyId")
    rows = []
    for _, r in runs.iterrows():
        z = np.load(r.npz, allow_pickle=False)
        blocks = []
        if set(types) & set(SPIKING):
            blocks.append(("lc", z["lc_body_ids"], z["lc_types"],
                           {"upstream_drive_mV": (z["a__lc_drive_mv"], z["b__lc_drive_mv"], "mean"),
                            "output_Hz": (z["a__lc_spikes"], z["b__lc_spikes"], "sum")}))
        if set(types) & set(RATE):       # the per-frame graded arrays are the bulk of a run npz: touched only if asked
            blocks.append(("rate", z["rate_body_ids"], z["rate_types"],
                           {"rate_deviation": (z["a__optic_dr"], z["b__optic_dr"], "mean"),
                            "abs_rate_deviation": (np.abs(z["a__optic_dr"]), np.abs(z["b__optic_dr"]), "mean")}))
        for _, bodies, btypes, quants in blocks:
            sel = np.flatnonzero(np.isin(btypes.astype(str), list(types)))
            if not len(sel):
                continue
            n = min(*[min(len(a), len(b)) for a, b, _ in quants.values()])
            masks = {}
            for i in sel:
                b = str(int(bodies[i])); t = str(btypes[i])
                key = None
                wr = wi.loc[b] if b in wi.index else None
                if wr is not None and str(wr["window_source"]) != "whole":
                    key = (float(wr["az_deg"]), float(wr["el_deg"]), float(wr["width_deg"]), float(wr["height_deg"]))
                if key not in masks:
                    masks[key] = window_mask(z, None if key is None else wr, obj_w, obj_h, n)
                m = masks[key]
                nf = int(m.sum())
                src = "whole" if wr is None else str(wr["window_source"])
                for qname, (A, B, how) in quants.items():
                    a = A[:n, i].astype(np.float64)[m]; bb = B[:n, i].astype(np.float64)[m]
                    if how == "sum":
                        va, vb = (float(a.sum()), float(bb.sum())) if nf else (np.nan, np.nan)
                    else:
                        va, vb = (float(a.mean()), float(bb.mean())) if nf else (np.nan, np.nan)
                    rows.append({"run": r.stem, "bodyId": b, "type": t, "quantity": qname,
                                 "window_source": src, "window_n_frames": nf, "n_frames": int(n),
                                 "window_az_deg": np.nan if key is None else key[0],
                                 "window_el_deg": np.nan if key is None else key[1],
                                 "window_width_deg": np.nan if key is None else key[2],
                                 "window_height_deg": np.nan if key is None else key[3],
                                 "window_stimulus_value": va, "window_control_value": vb,
                                 "window_stimulus_minus_control": va - vb if nf else np.nan})
        z.close()
    return pd.DataFrame(rows)


def pool(df: pd.DataFrame, value_cols: dict) -> pd.DataFrame:
    """Mean / SD over runs per (bodyId, type, quantity) for the named columns (`{out_name: in_name}`)."""
    g = df.groupby(["bodyId", "type", "quantity"], sort=False)
    out = g.size().rename("n_trials").reset_index()
    for name, col in value_cols.items():
        m = g[col].mean().reset_index(name=name)
        s = g[col].std(ddof=1).reset_index(name=name + "_sd")
        out = out.merge(m, on=["bodyId", "type", "quantity"]).merge(s, on=["bodyId", "type", "quantity"])
    return out


def readout_per_body(obj_runs: pd.DataFrame, null_runs: pd.DataFrame, win: pd.DataFrame, diam: float,
                     window_s: tuple, paired_ids: str, null_ids: str) -> tuple[pd.DataFrame, dict]:
    """The interchange table: one row per (body, quantity), the two references named apart, the RF-window columns of
    the predeclared rule beside the whole-window ones."""
    ow, nw = per_body_whole(obj_runs), per_body_whole(null_runs)
    keys = ["bodyId", "type", "quantity"]
    base = ow.groupby(keys + ["model_index", "unit_kind", "unit"], sort=False).agg(
        stimulus_value=("a_mean", "mean"), control_value=("b_mean", "mean"),
        stimulus_minus_control=("diff", "mean"),
        stimulus_sd=("a_mean", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
        control_sd=("b_mean", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
        trial_sd=("diff", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
        n_trials=("diff", "size")).reset_index()
    nl = nw.groupby(keys, sort=False).agg(null_mean=("diff", "mean"),
                                          null_sd=("diff", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
                                          n_null=("diff", "size")).reset_index()
    df = base.merge(nl, on=keys, how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        df["z_vs_null"] = (df.stimulus_minus_control - df.null_mean) / df.null_sd
    df["verdict"] = np.where(df.n_trials < common.CALL_REPLICATES, "underpowered",
                             np.where(np.isfinite(df.z_vs_null) & (np.abs(df.z_vs_null) >= common.Z_RESULT),
                                      "result", "null"))
    # the RF-windowed columns (the predeclared primary window), for the types the rule is applied to
    ow_w = per_body_windowed(obj_runs, win, diam, diam)
    nw_w = per_body_windowed(null_runs, win, diam, diam)
    if len(ow_w):
        po = pool(ow_w, {"window_stimulus_value": "window_stimulus_value",
                         "window_control_value": "window_control_value",
                         "window_stimulus_minus_control": "window_stimulus_minus_control"})
        po = po.rename(columns={"n_trials": "window_n_trials",
                                "window_stimulus_minus_control_sd": "window_trial_sd"})
        meta = ow_w.groupby(keys, sort=False).agg(window_source=("window_source", "first"),
                                                  window_n_frames=("window_n_frames", "mean"),
                                                  window_az_deg=("window_az_deg", "first"),
                                                  window_el_deg=("window_el_deg", "first"),
                                                  window_width_deg=("window_width_deg", "first"),
                                                  window_height_deg=("window_height_deg", "first")).reset_index()
        pn = nw_w.groupby(keys, sort=False).agg(
            window_null_mean=("window_stimulus_minus_control", "mean"),
            window_null_sd=("window_stimulus_minus_control", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
            window_n_null=("window_stimulus_minus_control", "size")).reset_index()
        df = df.merge(meta, on=keys, how="left").merge(po, on=keys, how="left").merge(pn, on=keys, how="left")
        with np.errstate(invalid="ignore", divide="ignore"):
            df["window_z_vs_null"] = (df.window_stimulus_minus_control - df.window_null_mean) / df.window_null_sd
    for c in ("window_source",):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    df["window_unit"] = df.quantity.map({v[0]: v[3] for v in QUANTITY.values()})
    df["window_rule"] = "predeclared (out/objr2/predeclared.json): the object overlaps the body's RF box"
    df["window_start_s"] = float(window_s[0]); df["window_end_s"] = float(window_s[1])
    df["object_angular_size_deg"] = float(diam)
    df["paired_control_ids"] = paired_ids
    df["null_reference_ids"] = null_ids
    df["control_ids"] = null_ids                              # the deprecated alias, equal row for row
    df["n_null"] = df.n_null.fillna(0).astype(int)
    info = {"rows": int(len(df)), "bodies": int(df.bodyId.nunique()),
            "types": {t: int(df[df.type == t].bodyId.nunique()) for t in sorted(set(df.type))},
            "quantities": {q: int((df.quantity == q).sum()) for q in sorted(set(df.quantity))},
            "windowed_rows": int((df.get("window_n_frames", pd.Series(dtype=float)).notna()).sum())
            if "window_n_frames" in df.columns else 0}
    return df, info


# ==================================================================================================== the analysis tables
def comparisons_table(baseline: dict, key: str = "sphere_comparisons") -> pd.DataFrame:
    """`scripts/object_round2_baseline.py analyse`'s comparison rows, with the export's own contract columns added:
    `p_method` / `n_tied_values` / `U` from `export.compare_tie_aware` on the SAME per-run values, beside the
    predeclared tie-aware exact permutation U the analysis computed (`p`, `U_tie`, `n_tied`).

    Nothing is recomputed here: `z`, `welch`, `verdict`, `p` and the arm values are copied from the analysis; the
    Mann-Whitney columns are the export's contract vocabulary (`p_method` in {exact, asymptotic_tie_corrected, none})
    added beside them so a reader of the manifest's `conventions.p_method` finds what it describes."""
    rows = pd.DataFrame(baseline["tables"][key])
    if not len(rows):
        return rows
    # the analysis' own columns are kept under `analysis_*` names: it declares one family per (lobe, type) for the
    # primary rows and one per (lobe, type, statistic) for each exploratory / secondary set, and its Holm values are
    # the ones `docs/audits/object_baseline_r2.md` quotes. The export's `family` / `p_holm` (added below) carry only
    # the PREDECLARED primary family, so both multiplicity corrections travel and neither overwrites the other.
    rows = rows.rename(columns={c: f"analysis_{c}" for c in ("family", "p_holm", "survives_holm")
                                if c in rows.columns})
    if "p_method" in rows.columns:            # the analysis' own constant label; kept under an unambiguous name
        rows = rows.rename(columns={"p_method": "p_predeclared_method"})
    add = []
    for _, r in rows.iterrows():
        s = [float(v) for v in (r.get("stim_values") or [])]
        n = [float(v) for v in (r.get("null_values") or [])]
        mw = ex.mann_whitney(s, n)
        add.append({"p_mannwhitney": mw["p"], "p_method": mw["p_method"], "n_tied_values": mw["n_tied_values"],
                    "U_mannwhitney": mw["U"]})
    out = pd.concat([rows.reset_index(drop=True), pd.DataFrame(add)], axis=1)
    out["p_predeclared_method"] = "tie-aware exact permutation U (two-sided), the predeclared test"
    out["statistic_definition"] = out.statistic.astype(str).map(lambda s: ex.STATISTIC_DEFINITIONS.get(
        s, "not defined in this revision of the export"))
    return out


def with_family(df: pd.DataFrame, spec=None) -> pd.DataFrame:
    """`family` / `p_holm` on the PREDECLARED p (the tie-aware exact permutation U), and `p_holm_mannwhitney` on the
    export's contract `p_mannwhitney`, so the two multiplicity corrections can be read side by side."""
    out = ex.add_family_columns(df, spec, p_col="p")
    fam = out.family.to_numpy()
    out["p_holm_mannwhitney"] = ex.holm(out.p_mannwhitney.to_numpy() if "p_mannwhitney" in out else
                                        np.full(len(out), np.nan), fam)
    out["p_holm_source"] = np.where([bool(f) for f in fam],
                                    "Holm within `family` of `p` (the predeclared tie-aware exact permutation U)", "")
    return out


def rf_map_table(out_dir: str, win: pd.DataFrame) -> pd.DataFrame:
    """The per-body RF map as ONE table, one row per body: the window the predeclared rule gave it, and the localizer
    fit it came from in each lobe (`rfmap_ship.csv` / `rfmap_fb0.csv`: centre, width, peak, z, whether it was fitted,
    and the anatomical column that is the fallback), the two lobes side by side as `ship_*` / `fb0_*` columns."""
    keep = ["bodyId", "type", "az_deg", "el_deg", "width_deg", "peak", "z_peak", "fitted", "n_runs_fitted",
            "n_nodes_above_threshold", "sign", "noise_mad", "quantity", "window", "model_index",
            "anat_column", "anat_az_deg", "anat_el_deg", "hex_annotated", "anat_distance_deg", "z_blank"]
    shared = ("model_index", "anat_column", "anat_az_deg", "anat_el_deg", "hex_annotated")
    df = win.rename(columns={"az_deg": "window_az_deg", "el_deg": "window_el_deg",
                             "width_deg": "window_width_deg", "height_deg": "window_height_deg"}).copy()
    for lobe in ("ship", "fb0"):
        p = Path(out_dir) / f"rfmap_{lobe}.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p, dtype={"bodyId": str})
        m = m[[c for c in keep if c in m.columns]].drop_duplicates("bodyId").copy()
        ren = {c: (c if c in shared else f"{lobe}_fit_{c}") for c in m.columns if c not in ("bodyId", "type")}
        ren.update({"quantity": f"{lobe}_localizer_quantity", "window": f"{lobe}_localizer_window"})
        m = m.rename(columns={k: v for k, v in ren.items() if k in m.columns})
        drop = [c for c in m.columns if c in df.columns and c not in ("bodyId", "type")]
        df = df.merge(m.drop(columns=drop), on=["bodyId", "type"], how="outer")
    df["localizer"] = "15-deg dark square, 3 passes (scripts/probe_synthetic_stimuli.py record --stimulus localizer)"
    df["window_rule"] = ("PREDECLARED (out/objr2/predeclared.json): a fit in the shipped-lobe map, else a fit in the "
                         "fb0 map, else the anatomical column with the type's median fitted box, else the whole sweep")
    return df


# ==================================================================================================== building a run dir
def write_run_dir(res: common.Result, prefix: str, export_root: str, retina=None, retina_blank=None,
                  track=None, track_report=None, paired_ids="", null_ids="", parquet_rows=1_000_000,
                  expect_paired=("LC11", "LC10a"), expect_counts=None, json_dir=None) -> dict:
    """`export.export` -> the run directory; then the matched-arc `retina_object_track` (written with the exporter's
    own `write_table` and appended to the manifest), then the exporter's own validation into `checks.json`."""
    rid = run_id(prefix)
    if json_dir:    # the run directory already holds the Result verbatim as result.json (58 MB for a rung, mostly
        Path(json_dir).mkdir(parents=True, exist_ok=True)   # readout_per_body): a second copy is opt-in
        res.save(Path(json_dir) / f"{rid}.json")
    run_dir = ex.export(res, out_root=export_root, run_id=rid, retina=retina, retina_blank=retina_blank,
                        retina_in_loop=retina is not None, parquet_rows=parquet_rows,
                        paired_control_ids=paired_ids or None, null_reference_ids=null_ids or None)
    if track is not None and len(track):
        meta = ex.write_table(track, run_dir, "retina_object_track", parquet_rows, role="interchange")
        with open(Path(run_dir) / "manifest.json", encoding="utf-8") as f:
            man = json.load(f)
        man["tables"].append(meta)
        man.setdefault("retina", {})["object_track"] = dict(track_report or {}, file=meta["file"],
                                                            written_by="scripts/object_round2_export.py")
        man["retina"].setdefault("files", {})["object_track"] = meta["file"]
        with open(Path(run_dir) / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(common.to_jsonable(man), f, indent=1)
    trip = ex.round_trip_check(res, run_dir)
    info = ex.verify(run_dir, neurons=True, expect_paired=tuple(expect_paired), expect_counts=expect_counts)
    with open(Path(run_dir) / "checks.json", "w", encoding="utf-8") as f:
        json.dump({"note": "written after manifest.json; not a hashed interchange table",
                   "round_trip": common.to_jsonable(trip), "verify": common.to_jsonable(info)}, f, indent=1)
    with open(Path(run_dir) / "manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    print(f"  {run_dir}: {len(man['tables'])} tables, problems {info['problems'] or 'none'}", flush=True)
    return {"run_dir": str(run_dir).replace("\\", "/"), "run_id": rid, "problems": info["problems"],
            "tables": [{"name": t["name"], "file": t["file"], "format": t["format"], "rows": t["rows"],
                        "role": t.get("role"), "sha256": t["sha256"]} for t in man["tables"]],
            "round_trip": common.to_jsonable(trip), "verify": common.to_jsonable(info)}


def runs_table(runs: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for _, r in runs.iterrows():
        d = _load(r.json)
        ex_ = d["provenance"]["execution"]
        rows.append({"record_id": r.stem, "arm": "null" if r.null else "object", group_col: r[group_col],
                     "diam_deg": np.nan if r.null else float(r.diam), "seed": int(r.seed),
                     "reference_role": "null_reference" if r.null else "stimulus",
                     "file": str(r.json).replace("\\", "/"), "sha256": sha256_file(r.json),
                     "npz": str(r.npz).replace("\\", "/"), "npz_sha256": sha256_file(r.npz),
                     "npz_arrays": int(len(np.load(r.npz, allow_pickle=False).files)),
                     "npz_note": ("the run npz AS FETCHED: the two (frames, 12235) graded per-frame arrays "
                                  "a__optic_dr / b__optic_dr were dropped on the box before the hand fetch "
                                  "(scripts/object_round2_export.py slim); the JSON beside it is untouched and "
                                  "carries those types' per-body whole-window means, which is what this export and "
                                  "the predeclared analysis use"),
                     "device": ex_.get("device"), "device_name": ex_.get("device_name"),
                     "n_frames": int(d["summary"]["n_frames"]), "window_s": float(d["summary"]["window_s"]),
                     "realised_el_maxdev_deg": d["summary"]["track_check"]["realised_el_deg_maxdev"],
                     "realised_diam_maxdev_deg": d["summary"]["track_check"]["realised_diam_deg_maxdev"],
                     "speed_deg_s_min": d["summary"]["track_check"]["speed_deg_s_min"],
                     "speed_deg_s_max": d["summary"]["track_check"]["speed_deg_s_max"]})
    return pd.DataFrame(rows)


# ==================================================================================================== the ladder command
def load_runs(out: str, group_col: str, parse) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(f"{out}/sph/*.json")):
        if p.endswith("verify.json"):
            continue
        nm = parse(Path(p).stem)
        if nm is None or not os.path.exists(p[:-5] + ".npz"):
            continue
        rows.append(dict(nm, json=p, npz=p[:-5] + ".npz", stem=Path(p).stem))
    df = pd.DataFrame(rows)
    if len(df) and "lobe" in df.columns and group_col != "lobe":
        df = df.rename(columns={"lobe": group_col})
    if len(df) and "arm" in df.columns and group_col != "arm":
        df = df.rename(columns={"arm": group_col})
    return df


def ladder(args) -> int:
    out = args.out.rstrip("/")
    baseline = _load(args.baseline)
    if baseline.get("schema") != common.SCHEMA:
        raise SystemExit(f"{args.baseline}: not a flyverse Result")
    group_col = args.group_col
    parse = b2.parse_sphere_name if group_col == "lobe" else __import__("object_round2_compare").parse_sphere_name
    runs = load_runs(out, group_col, parse)
    if not len(runs):
        raise SystemExit(f"no sphere recordings under {out}/sph/")
    win = pd.DataFrame(baseline["tables"]["windows"])
    win["bodyId"] = win.bodyId.astype(str)
    rf = rf_map_table(out, win)
    comps = with_family(comparisons_table(baseline, args.comparisons_table), args.family_spec)
    per_run = pd.DataFrame(baseline["tables"].get(args.per_run_table, []))
    footprint = pd.DataFrame(baseline["tables"].get(args.footprint_table, []))
    pref = pd.DataFrame(baseline["tables"].get(args.preference_table, []))
    tc = pd.DataFrame(baseline["tables"].get(args.time_course_table, []))
    ups = pd.DataFrame(baseline["tables"].get(args.upstream_table, []))
    groups = [args.group] if args.group else sorted(set(runs[group_col]))
    sizes = args.sizes or sorted({float(d) for d in runs[~runs.null].diam})
    written, summary_rows = [], []
    for g in groups:
        gr = runs[runs[group_col] == g]
        nulls = gr[gr.null]
        if not len(nulls):
            print(f"[{g}] no blank/blank runs: skipped"); continue
        for diam in sizes:
            obj = gr[(~gr.null) & (np.abs(gr.diam.astype(float) - diam) < 1e-6)]
            if not len(obj):
                continue
            rec = export_one_size(out, g, group_col, diam, obj, nulls, win, rf, comps, per_run, footprint, tc, ups,
                                  baseline, args)
            written.append(rec); summary_rows.append({"group": g, "diam_deg": diam, **{k: rec[k] for k in ("run_id", "run_dir")},
                                                      "problems": "; ".join(rec["problems"]) or "none"})
    # ------------------------------------------------------------------ the ladder summary directory
    rec = export_summary(out, runs, group_col, win, rf, comps, per_run, footprint, pref, tc, ups, baseline, written, args)
    written.append(rec)
    summary_rows.append({"group": "ALL", "diam_deg": np.nan, "run_id": rec["run_id"], "run_dir": rec["run_dir"],
                         "problems": "; ".join(rec["problems"]) or "none"})
    index = {"generator": " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "baseline_result": args.baseline, "baseline_sha256": sha256_file(args.baseline),
             "export_schema": ex.EXPORT_SCHEMA, "export_revision": ex.EXPORT_REVISION,
             "directories": written}
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    with open(args.index, "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(index), f, indent=1)
    print("\n== run directories")
    common.print_table(pd.DataFrame(summary_rows), floatfmt="{:.1f}", max_rows=60)
    bad = [r for r in written if r["problems"]]
    print(f"\nindex {args.index}; {len(written)} directories, {len(bad)} with problems")
    return 1 if bad else 0


def export_one_size(out, g, group_col, diam, obj, nulls, win, rf, comps, per_run, footprint, tc, ups, baseline,
                    args) -> dict:
    tag = f"d{int(round(diam * 10)):03d}"
    print(f"[{g} {diam:g} deg] {len(obj)} object runs, {len(nulls)} null runs", flush=True)
    seed0 = obj.sort_values("seed").iloc[0]
    d0 = _load(seed0.json)
    prov = json.loads(json.dumps(d0["provenance"]))          # a copy: never edit the recording's own block
    paired = "|".join(f"{s}{ex.ARM_B_SUFFIX}" for s in sorted(obj.stem))
    nullids = "|".join(sorted(nulls.stem))
    window_s = (float(d0["provenance"]["stimulus"]["params"]["settle_s"]),
                float(d0["provenance"]["stimulus"]["params"]["settle_s"]) + float(d0["summary"]["window_s"]))
    rpb, info = readout_per_body(obj, nulls, win, diam, window_s, paired, nullids)
    # ---- the stimulus block: the rung, the controls, the predeclaration
    stim = dict(prov.get("stimulus") or {})
    stim["protocol"] = "object_matched_sphere"
    stim["control"] = {"condition": "arm b: the same timeline with the ball parked outside the room",
                       "null": "independent blank/blank runs of the same lobe in the same submission",
                       "n_stim_runs": int(len(obj)), "n_null_runs": int(len(nulls))}
    stim["runs"] = {"stimulus": [str(p).replace("\\", "/") for p in obj.json], "null": [str(p).replace("\\", "/") for p in nulls.json]}
    stim["matched"] = {"elevation_deg": stim["params"].get("elevation_deg"), "distance_m": stim["params"].get("distance_m"),
                       "deg_per_s": stim["params"].get("deg_per_s"), "az_max_deg": stim["params"].get("az_max_deg"),
                       "note": "constant centre elevation, constant distance from the eye, constant angular speed "
                               "along the arc and one contrast for every rung: the confounds Neurome's intake found "
                               "in the old scene ladder (size with retinal elevation, size with angular speed)"}
    stim["predeclared"] = predeclared_block(out)
    stim["controls"] = controls_block(paired, nullids)
    prov["stimulus"] = stim
    res = common.Result.new("export", prov)
    res.run_id = f"objr2-{g}-{tag}"
    res.tool_version = "object_round2_export/1"
    stamp_commit(res)
    res.add_table("readout_per_body", rpb)
    sub = comps[(comps[group_col] == g) & (np.abs(comps.diam_deg.astype(float) - diam) < 1e-6)] if len(comps) else comps
    res.add_table("per_type", sub)
    if len(per_run):
        pr = per_run[(per_run[group_col] == g) & (np.abs(per_run.diam_deg.astype(float) - diam) < 1e-6)]
        res.add_table("per_run", pr)
    if len(ups):
        u = ups[(ups[group_col] == g) & ((np.abs(ups.diam_deg.astype(float) - diam) < 1e-6) | ups.null)]
        res.add_table("upstream_per_run", u)
    if len(tc):
        t = tc[(tc[group_col] == g) & (np.abs(tc.diam_deg.astype(float) - diam) < 1e-6)]
        res.add_table("time_course", t)
    if len(footprint):
        f = footprint[(footprint[group_col] == g) & (np.abs(footprint.diam_deg.astype(float) - diam) < 1e-6)]
        res.add_table("retina_footprint_runs", f)
    agg = pd.DataFrame(baseline["tables"].get(args.footprint_agg_table, []))
    if len(agg):
        a = agg[(agg[group_col] == g) & (np.abs(agg.diam_deg.astype(float) - diam) < 1e-6)]
        if len(a):
            res.add_table("retina_footprint", a)
    res.add_table("rf_map", rf)
    res.add_table("column_definitions", [{"table": "readout_per_body", "column": k, "definition": v}
                                         for k, v in COLUMN_DEFINITIONS.items()])
    res.add_table("runs", runs_table(pd.concat([obj, nulls]), group_col))
    for t in SPIKING + RATE:
        sub_b = rpb[rpb.type == t].drop_duplicates("bodyId")
        res.add_population(common.Population(t, t, sub_b.model_index.to_numpy(np.int64),
                                             sub_b.bodyId.to_numpy(np.int64)),
                           unit_kind="spiking" if t in SPIKING else "graded", keep_ids=False)
    # ---- the retina: the in-loop capture of both arms, the matched-arc track
    z = np.load(seed0.npz, allow_pickle=False)
    maps = retina_maps(prov["retina"], z["col_az_el"], z["col_side"], cache_dir=args.cache_dir)
    check_model_index(maps, d0)
    obj_rad, blank_rad = radiance_records(z, maps, str(prov["retina"].get("sampling", "")))
    track, track_report = object_track_table(z, d0["summary"]["geometry"])
    track_report["record"] = seed0.stem
    lean = {k: v for k, v in prov["retina"].items() if k not in ("column_to_bodies", "col_side", "col_hex", "col_az_el")}
    prov["retina"] = dict(lean, radiance_record=seed0.stem, radiance_npz=str(seed0.npz).replace("\\", "/"),
                          radiance_npz_sha256=sha256_file(seed0.npz),
                          n_runs_with_radiance=int(len(obj)) + int(len(nulls)),
                          note="the radiance tables are the seed-0 recording of this rung, captured IN the loop for "
                               "both arms; the ray tracer is deterministic and the fly is pinned, so every run of the "
                               "rung presented the same stimulus (the run-to-run scatter is in the brain, not in the "
                               "retina). The column / photoreceptor map is in retina_columns.csv and "
                               "retina_bodies.csv, rebuilt on the CPU from the record's own column_to_bodies and "
                               "cache/neurons.parquet (scripts/object_round2_export.py::retina_maps).")
    res.provenance["retina"] = prov["retina"]
    res.summary = {"protocol": "object_matched_sphere", "group": g, "group_kind": group_col,
                   "angular_diameter_deg": float(diam), "window_s": list(window_s),
                   "n_stim_runs": int(len(obj)), "n_null_runs": int(len(nulls)),
                   "readout": info, "track": track_report,
                   "predeclared_primary": primary_rows(sub),
                   "reading": "verdicts are common.compare's (z on the null SD, an exact/tie-aware rank test and the "
                              "p floor); `p` is the predeclared tie-aware exact permutation U; `p_mannwhitney` / "
                              "`p_method` are the export's contract columns; Holm lives in the ladder summary, where "
                              "the whole predeclared family is present"}
    res.validation = {"name": "matched sphere rung exported for Neurome", "reference": common.VALIDATION["export"],
                      "measured": {"readout": info, "track": track_report}, "status": "measured",
                      "source": "docs/audits/object_export_r2.md"}
    res.files = {"generator": " ".join(sys.argv), "baseline_result": args.baseline,
                 "recordings": [str(p).replace("\\", "/") for p in list(obj.json) + list(nulls.json)],
                 "radiance_npz": str(seed0.npz).replace("\\", "/")}
    rec = write_run_dir(res, f"objr2-{g}-{tag}", args.export_root, retina=obj_rad, retina_blank=blank_rad,
                        track=track, track_report=track_report, paired_ids=paired, null_ids=nullids,
                        parquet_rows=args.parquet_rows, expect_counts=EXPECT_COUNTS,
                        json_dir=args.json_dir if args.save_results else None)
    z.close()
    rec.update(group=g, diam_deg=float(diam), kind="size")
    return rec


def primary_rows(sub: pd.DataFrame) -> list:
    if not len(sub):
        return []
    p = sub[sub.get("role", pd.Series([""] * len(sub))) == "primary"]
    cols = [c for c in ("type", "statistic", "stim_n", "stim_mean", "stim_sd", "null_n", "null_mean", "null_sd",
                        "z", "p", "p_mannwhitney", "p_method", "verdict") if c in p.columns]
    return common.to_jsonable(p[cols].to_dict("records"))


def check_model_index(maps: dict, d0: dict) -> None:
    """The bodyId -> model index map rebuilt from cache/neurons.parquet must agree with the indices the RUN wrote."""
    pb = pd.DataFrame(d0["tables"]["per_body"])
    nrn = neurons_table()
    idx_of = dict(zip(nrn.bodyId.to_numpy(), nrn.model_index.to_numpy()))
    got = pb.drop_duplicates("bodyId")
    bad = [b for b, i in zip(got.bodyId.astype(np.int64), got.model_index.astype(np.int64)) if idx_of.get(b) != i]
    if bad:
        raise SystemExit(f"cache/neurons.parquet row order disagrees with the run's model_index for "
                         f"{len(bad)} bodies (e.g. {bad[:3]}): the export would ship a wrong photoreceptor map")


def controls_block(paired: str, nullids: str) -> dict:
    """The manifest's one copy of the two references (revision 2, `docs/audits/interp_export.md` 14.1); the same two
    strings are on every `readout_per_body` row."""
    return {"paired_control_ids": paired, "null_reference_ids": nullids,
            "paired_control": "arm b (the matched blank: the same timeline with the ball parked outside the room) of "
                              "each stimulus recording -- the source of `control_value`, `stimulus_minus_control` "
                              "and every windowed difference",
            "null_reference": "independent blank/blank runs of the same lobe in the same submission (arm a is a "
                              "second blank) -- the source of `null_mean` / `null_sd` / `z_vs_null` and of the "
                              "comparison arm of every per-type statistic",
            "deprecated_control_ids": "an alias of null_reference_ids (export.CONVENTIONS['control_ids'])"}


def predeclared_block(out: str) -> dict:
    p = Path(out) / "predeclared.json"
    if not p.exists():
        return {"file": None}
    return {"file": str(p).replace("\\", "/"), "sha256": sha256_file(p), **_load(p)}


def export_summary(out, runs, group_col, win, rf, comps, per_run, footprint, pref, tc, ups, baseline, written,
                   args) -> dict:
    """The ladder summary: every rung x type x statistic in one `size_tuning` table with the predeclared family's
    Holm columns, the retinal footprint per rung, the preference tests, the RF map and every recording."""
    print("[summary] the whole ladder", flush=True)
    seed0 = runs[~runs.null].sort_values(["seed"]).iloc[0]
    d0 = _load(seed0.json)
    prov = json.loads(json.dumps(d0["provenance"]))
    stim = dict(prov.get("stimulus") or {})
    stim["protocol"] = "object_matched_sphere_ladder"
    stim["control"] = {"condition": "arm b of each recording", "null": "independent blank/blank runs per lobe",
                       "n_stim_runs": int((~runs.null).sum()), "n_null_runs": int(runs.null.sum())}
    stim["runs"] = {"stimulus": [str(p).replace("\\", "/") for p in runs[~runs.null].json],
                    "null": [str(p).replace("\\", "/") for p in runs[runs.null].json]}
    stim["ladder"] = {"rungs_deg": sorted({float(d) for d in runs[~runs.null].diam}),
                      group_col + "s": sorted(set(runs[group_col]))}
    stim["predeclared"] = predeclared_block(out)
    stim["controls"] = controls_block("|".join(f"{s_}{ex.ARM_B_SUFFIX}" for s_ in sorted(runs[~runs.null].stem)),
                                      "|".join(sorted(runs[runs.null].stem)))
    prov["stimulus"] = stim
    res = common.Result.new("export", prov)
    res.run_id = "objr2-ladder"
    res.tool_version = "object_round2_export/1"
    stamp_commit(res)
    res.add_table("size_tuning", comps)
    if len(per_run):
        res.add_table("per_run", per_run)
    if len(pref):
        res.add_table("preference", pref)
    if len(footprint):
        res.add_table("retina_footprint_runs", footprint)
    agg = pd.DataFrame(baseline["tables"].get(args.footprint_agg_table, []))
    if len(agg):
        res.add_table("retina_footprint", agg)
    if len(ups):
        res.add_table("upstream_per_run", ups)
    if len(tc):
        res.add_table("time_course", tc)
    res.add_table("rf_map", rf)
    res.add_table("runs", runs_table(runs, group_col))
    res.add_table("column_definitions", [{"table": "readout_per_body", "column": k, "definition": v}
                                         for k, v in COLUMN_DEFINITIONS.items()]
                  + [{"table": "size_tuning", "column": k, "definition": v}
                     for k, v in {"p": "the PREDECLARED test: two-sided tie-aware exact permutation U over the per-run "
                                       "values of the two arms (scripts/object_round2_baseline.py::tie_aware_exact_u)",
                                  "p_mannwhitney": "the export's contract rank test (export.mann_whitney): exact only "
                                                   "on untied data, otherwise the tie-corrected normal approximation",
                                  "p_method": "which test produced `p_mannwhitney` (exact | asymptotic_tie_corrected | none)",
                                  "p_holm": "Holm-Bonferroni of `p` within `family` (the predeclared primary family)",
                                  "p_holm_mannwhitney": "Holm-Bonferroni of `p_mannwhitney` within `family`",
                                  "analysis_family": "the family the predeclared ANALYSIS declared for this row: one "
                                                     "per (lobe, type) for the primary rows, one per (lobe, type, "
                                                     "statistic) for each exploratory / secondary set",
                                  "analysis_p_holm": "Holm-Bonferroni within `analysis_family` "
                                                     "(scripts/object_round2_baseline.py; the values "
                                                     "docs/audits/object_baseline_r2.md quotes)",
                                  "analysis_survives_holm": "the analysis' call rule: verdict `result` AND "
                                                            "`analysis_p_holm` <= 0.05",
                                  "verdict": "common.compare's vocabulary (result | null | underpowered | undetermined)",
                                  "family": "the predeclared family this row belongs to ('' = none declared)"}.items()])
    for name, key in (("synthetic_size_tuning", "synth_comparisons"), ("synthetic_footprint", "synth_footprint"),
                      ("synthetic_preference", "synth_preference"), ("synthetic_per_run", "synth_per_run"),
                      ("old_ladder_size_tuning", "old_ladder_size_tuning"),
                      ("old_ladder_footprint", "old_ladder_footprint"),
                      ("old_ladder_geometry", "old_ladder_geometry")):
        rows = baseline["tables"].get(key)
        if rows:
            df = pd.DataFrame(rows)
            if key == "synth_comparisons":
                df = with_family(comparisons_table(baseline, key), None)
            res.add_table(name, df)
    res.add_table("export_directories", [{k: v for k, v in r.items() if k in ("run_id", "run_dir", "group",
                                                                              "diam_deg", "kind")} for r in written])
    fam = comps[comps.family.astype(str) != ""] if "family" in comps.columns else comps.iloc[:0]
    res.summary = {"protocol": "object_matched_sphere_ladder",
                   "rungs_deg": sorted({float(d) for d in runs[~runs.null].diam}),
                   "groups": sorted(set(runs[group_col])), "n_runs": int(len(runs)),
                   "family": {"spec": args.family_spec, "declaration": PRIMARY_FAMILY_NOTE, "members": int(len(fam)),
                              "labels": sorted(set(fam.family.astype(str))) if len(fam) else [],
                              "min_p": float(fam.p.min()) if len(fam) else None,
                              "min_p_holm": float(fam.p_holm.min()) if len(fam) else None,
                              "survivors": common.to_jsonable(
                                  fam[(fam.p_holm <= 0.05) & (fam.verdict == "result")][
                                      [c for c in ("type", "statistic", "diam_deg", "z", "p", "p_holm", "verdict")
                                       if c in fam.columns]].to_dict("records")) if len(fam) else []},
                   "answers": (baseline.get("summary") or {}).get("answers"),
                   "windows": (baseline.get("summary") or {}).get("windows"),
                   "predeclared": (baseline.get("summary") or {}).get("predeclared"),
                   "reading": "the per-rung directories carry the per-body interchange rows and the captured "
                              "radiance; this directory carries the ladder-wide comparison with the predeclared "
                              "family's Holm columns. No verdict rests on p_holm."}
    res.validation = {"name": "matched sphere ladder exported for Neurome", "reference": common.VALIDATION["export"],
                      "measured": res.summary["family"], "status": "measured",
                      "source": "docs/audits/object_export_r2.md"}
    res.files = {"generator": " ".join(sys.argv), "baseline_result": args.baseline,
                 "baseline_sha256": sha256_file(args.baseline),
                 "per_size_directories": [r["run_dir"] for r in written]}
    rec = write_run_dir(res, "objr2-ladder", args.export_root, paired_ids="", null_ids="",
                        parquet_rows=args.parquet_rows, expect_paired=(), json_dir=args.json_dir)
    rec.update(group="ALL", diam_deg=float("nan"), kind="ladder_summary")
    return rec


# ==================================================================================================== tables / verify
def cmd_tables(args) -> int:
    """Every table of a finished run directory: name, role, format, rows, columns and SHA-256 (the list this round's
    audit quotes, and the list Neurome checks a delivery against)."""
    rows = []
    for rd in args.run_dir:
        with open(Path(rd) / "manifest.json", encoding="utf-8") as f:
            man = json.load(f)
        for t in man["tables"]:
            rows.append({"run_dir": str(rd).replace("\\", "/"), "name": t["name"], "role": t.get("role"),
                         "format": t["format"], "rows": t["rows"], "n_columns": len(t.get("columns", [])),
                         "bytes": (Path(rd) / t["file"]).stat().st_size, "sha256": t["sha256"]})
    df = pd.DataFrame(rows)
    common.print_table(df.drop(columns=["run_dir"]) if df.run_dir.nunique() == 1 else df, max_rows=400)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(args.json, orient="records", indent=1)
        print(f"written {args.json}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    print(f"{len(df)} tables over {df.run_dir.nunique()} run director(ies); "
          f"{df.rows.sum():,} rows, {df.bytes.sum() / 1e6:.1f} MB")
    return 0


SLIM_DROP = ("a__optic_dr", "b__optic_dr")


def cmd_slim(args) -> int:
    """Rewrite each run npz without the named arrays, for the hand fetch (run ON the box, over ssh).

    A 12 s matched-sphere run npz is 114 MB, of which 106 MB is the two (1200, 12235) graded per-frame arrays
    `a__optic_dr` / `b__optic_dr`; the captured radiance of both arms compresses to 0.5 MB because the scene is
    static. Neither the predeclared analysis (`scripts/object_round2_baseline.py analyse`, which reads the graded
    types from each run's JSON summary) nor this export (whose graded per-body rows are the probe's own whole-window
    means from the run JSON, the predeclared treatment of the upstream types) reads those two arrays, so dropping
    them turns a ~10 GB fetch into a ~0.7 GB one and loses nothing either of them uses. Every other array is copied
    through unchanged and the console prints the byte counts, so what was dropped is on the record."""
    src, dst = Path(args.dir), Path(args.slim_out)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))
    tot_in = tot_out = 0
    for p in files:
        if p.name.endswith("_slim.npz"):
            continue
        if args.require_json and not p.with_suffix(".json").exists():
            print(f"  {p.name}: no sibling .json yet (the probe writes the npz first): skipped", flush=True)
            continue
        if (dst / p.name).exists() and not args.overwrite:
            continue
        z = np.load(p, allow_pickle=False)
        keep = {k: z[k] for k in z.files if k not in set(args.drop)}
        q = dst / p.name
        np.savez_compressed(q, **keep)
        z.close()
        tot_in += p.stat().st_size; tot_out += q.stat().st_size
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} -> {q.stat().st_size / 1e6:.1f} MB "
              f"({len(z.files)} -> {len(keep)} arrays)", flush=True)
    print(f"slim: {len(files)} files, {tot_in / 1e9:.2f} -> {tot_out / 1e9:.2f} GB, dropped {list(args.drop)}")
    return 0


def cmd_check(args) -> int:
    """Independent checks OF a finished per-rung directory, each one a number this round's audit quotes:

      retina   -- the footprint recomputed from `retina_radiance` against `retina_radiance_blank` reproduces the
                  probe's own `summary.footprint` (columns changed / dimmed per frame, the dimmed centroid's
                  elevation, the blank luminance under the object): the exported radiance IS the presented stimulus;
      window   -- the RF-windowed per-body values in `readout_per_body` reproduce the analysis' own per-body CSV
                  (`out/objr2/per_body_sphere.csv`), which was computed by a different function;
      readout  -- body counts per type, both quantities per LC body, the two reference id sets present and disjoint;
      track    -- the matched arc as measured in the loop: the elevation / diameter / speed bands, and the geometric
                  azimuth against the radiance-weighted centroid of the columns the object dimmed;
      tests    -- the predeclared tie-aware exact permutation U against the export's contract rank test, row by row.
    """
    rd = Path(args.run_dir[0])
    with open(rd / "manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    rep = {"run_dir": str(rd).replace("\\", "/"), "run_id": man["run_id"]}
    rec = man["retina"]["object_track"]["record"]
    probe = _load(Path(args.out) / "sph" / f"{rec}.json")
    fp = probe["summary"]["footprint"]
    ra = ex.read_table(rd, "retina_radiance"); rb = ex.read_table(rd, "retina_radiance_blank")
    n_col = int(man["retina"]["n_columns"])
    def lum(df):
        v = df[["radiance_uv", "radiance_b", "radiance_g", "radiance_r"]].to_numpy(float).sum(1)
        return v.reshape(-1, n_col)
    la, lb = lum(ra), lum(rb)
    rel = (la - lb) / (lb + 1e-9)
    az_el = ex.read_table(rd, "retina_columns")[["azimuth_deg", "elevation_deg"]].to_numpy(float)
    strong = np.abs(rel) > 0.5                       # probe_object_matched.footprint's own `strong` set and centroid
    cen_el = np.array([az_el[m, 1].mean() if m.any() else np.nan for m in strong])
    bg = np.array([lb[j][strong[j]].mean() if strong[j].any() else np.nan for j in range(len(rel))])
    got = {"n_changed_5pct_mean": float((np.abs(rel) > 0.05).sum(1).mean()),
           "n_dimmed_50pct_mean": float((rel < -0.5).sum(1).mean()),
           "centroid_el_mean_deg": float(np.nanmean(cen_el)),
           "blank_lum_under_object_mean": float(np.nanmean(bg))}
    rep["retina"] = {"record": rec, "probe": {k: fp.get(k) for k in got}, "from_exported_tables": got,
                     "max_abs_diff": float(max(abs(got[k] - float(fp[k])) for k in got if fp.get(k) is not None)),
                     "n_columns": n_col, "n_photoreceptors": int(man["retina"]["n_photoreceptors"]),
                     "columns_with_no_photoreceptor": int((ex.read_table(rd, "retina_columns")
                                                           .n_photoreceptors == 0).sum())}
    # ---- the RF window, against the analysis' own per-body CSV
    r = ex.read_table(rd, "readout_per_body")
    csv = Path(args.out) / "per_body_sphere.csv"
    if csv.exists():
        pb = pd.read_csv(csv, dtype={"bodyId": str})
        diam = float(r.object_angular_size_deg.iloc[0])
        g = man.get("summary", {}).get("group") or "ship"
        sub = pb[(pb.lobe == g) & (~pb["null"].astype(bool)) & (np.abs(pb.diam_deg - diam) < 1e-6)]
        m = sub.groupby(["bodyId", "type"]).agg(drive=("drive_diff_win", "mean"),
                                                spikes=("spikes_diff_win", "mean"),
                                                nf=("n_frames_win", "mean")).reset_index()
        j = r[r.quantity == "upstream_drive_mV"].merge(m, on=["bodyId", "type"])
        k = r[r.quantity == "output_Hz"].merge(m, on=["bodyId", "type"])
        d1 = np.abs(j.window_stimulus_minus_control.to_numpy(float) - j.drive.to_numpy(float))
        d2 = np.abs(k.window_stimulus_minus_control.to_numpy(float) - k.spikes.to_numpy(float))
        d3 = np.abs(j.window_n_frames.to_numpy(float) - j.nf.to_numpy(float))
        rep["window"] = {"csv": str(csv).replace("\\", "/"), "bodies_joined": int(len(j)),
                         "max_abs_diff_drive_mv": float(np.nanmax(d1)) if len(d1) else None,
                         "max_abs_diff_spikes": float(np.nanmax(d2)) if len(d2) else None,
                         "max_abs_diff_window_frames": float(np.nanmax(d3)) if len(d3) else None,
                         "note": "the export's per-body window (scripts/object_round2_export.py::per_body_windowed) "
                                 "against the analysis' own (scripts/object_round2_baseline.py::windowed_body_stats)"}
    def both(ty):                      # every body of `ty` carries BOTH paired quantities (verify's rule, re-checked)
        sub = r[r.type == ty]
        if not len(sub):
            return False
        q = sub.groupby("bodyId").quantity.apply(set)
        return bool(len(q) and all(set(ex.PAIRED_QUANTITIES) <= s for s in q))
    rep["readout"] = {"rows": int(len(r)), "bodies": int(r.bodyId.nunique()),
                      "bodies_per_type": {t: int(r[r.type == t].bodyId.nunique()) for t in sorted(set(r.type))},
                      "LC11_both_quantities": both("LC11"), "LC10a_both_quantities": both("LC10a"),
                      "paired_and_null_disjoint": bool(set(str(r.paired_control_ids.iloc[0]).split("|")).isdisjoint(
                          set(str(r.null_reference_ids.iloc[0]).split("|")))),
                      "control_ids_equals_null_reference_ids": bool((r.control_ids == r.null_reference_ids).all()),
                      "window_sources": r[r.window_source != ""].window_source.value_counts().to_dict()}
    t = ex.read_table(rd, "retina_object_track")
    rep["track"] = {"frames": int(len(t)),
                    "elevation_deg": [float(t.centre_elevation_deg.min()), float(t.centre_elevation_deg.max())],
                    "diameter_deg": [float(t.angular_diameter_deg.min()), float(t.angular_diameter_deg.max())],
                    "azimuth_deg": [float(t.centre_azimuth_deg.min()), float(t.centre_azimuth_deg.max())],
                    "speed_deg_s_straight": [float(t[~t.turn_frame.astype(bool)].angular_speed_deg_s.min()),
                                             float(t[~t.turn_frame.astype(bool)].angular_speed_deg_s.max())],
                    "columns_dimmed_5pct_mean": float(t.columns_dimmed_5pct.mean()),
                    "columns_dimmed_50pct_mean": float(t.columns_dimmed_50pct.mean()),
                    "min_relative_radiance": float(t.min_relative_radiance.min()),
                    "max_abs_az_minus_dimmed_centroid_deg": float(np.nanmax(np.abs(
                        t.centre_azimuth_deg - t.dimmed_centroid_azimuth_deg))),
                    "blank_is_static": man["retina"]["object_track"].get("blank_is_static"),
                    "max_abs_diff_vs_blank_frame0_convention":
                        man["retina"]["object_track"].get("max_abs_diff_vs_blank_frame0_convention")}
    st_dir = args.summary_dir or str(rd)
    try:
        st = ex.read_table(st_dir, "size_tuning")
    except (KeyError, FileNotFoundError):
        st = ex.read_table(rd, "per_type")
    ok = st[np.isfinite(pd.to_numeric(st.p, errors="coerce")) & np.isfinite(pd.to_numeric(st.p_mannwhitney, errors="coerce"))]
    untied = ok[pd.to_numeric(ok.n_tied_values, errors="coerce") == 0]
    rep["tests"] = {"rows": int(len(st)), "rows_with_both_p": int(len(ok)),
                    "untied_rows": int(len(untied)),
                    "max_abs_p_diff_untied": float(np.nanmax(np.abs(pd.to_numeric(untied.p, errors="coerce")
                                                                    - pd.to_numeric(untied.p_mannwhitney, errors="coerce"))))
                    if len(untied) else None,
                    "tied_rows": int((pd.to_numeric(ok.n_tied_values, errors="coerce") > 0).sum()),
                    "p_methods": ok.p_method.value_counts().to_dict(),
                    "family_members": int((st.family.fillna("").astype(str) != "").sum()) if "family" in st else 0,
                    "verdicts": st.verdict.value_counts().to_dict()}
    print(json.dumps(common.to_jsonable(rep), indent=1))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(common.to_jsonable(rep), f, indent=1)
        print(f"written {args.json}")
    return 0


def cmd_verify(args) -> int:
    rc = 0
    for rd in args.run_dir:
        info = ex.verify(rd, neurons=True,
                         expect_paired=tuple(t for t in (args.paired or "").split(",") if t.strip()),
                         expect_counts=json.loads(args.expect_counts) if args.expect_counts else None)
        print(f"{rd}: " + json.dumps({k: v for k, v in info.items() if k != "problems"}))
        print("  problems:", info["problems"] or "none")
        rc |= 1 if info["problems"] else 0
    return rc


# ==================================================================================================== main
def _family(spec):
    if not spec or spec == "none":
        return None
    if spec == "primary":
        return PRIMARY_FAMILY
    if spec in ex.FAMILY_SPECS:
        return spec
    return json.loads(spec)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_args(p, out, baseline, group_col, prefix):
        p.add_argument("--out", default=out)
        p.add_argument("--baseline", default=baseline)
        p.add_argument("--export-root", default="out/export")
        p.add_argument("--json-dir", default="out/interp/objr2")
        p.add_argument("--index", default=f"out/export/{prefix}_index.json")
        p.add_argument("--sizes", type=float, nargs="*", default=None)
        p.add_argument("--group", default=None, help=f"only this {group_col} (default: every one present)")
        p.add_argument("--family", dest="family", default="primary", help="primary | none | a FAMILY_SPECS key | a JSON spec")
        p.add_argument("--parquet-rows", type=int, default=1_000_000)
        p.add_argument("--save-results", action="store_true",
                       help="also write each per-rung Result under --json-dir (the run directory already holds it "
                            "verbatim as result.json, ~58 MB per rung)")
        p.add_argument("--cache-dir", default=None)
        p.set_defaults(group_col=group_col, func=ladder)

    l = sub.add_parser("ladder", help="the matched baseline sphere ladder -> one run directory per rung + the summary")
    common_args(l, "out/objr2", "out/interp/objr2/baseline.json", "lobe", "objr2")
    tabs = dict(comparisons_table="sphere_comparisons", per_run_table="sphere_per_run",
                footprint_table="sphere_footprint_runs", footprint_agg_table="sphere_footprint",
                preference_table="sphere_preference", time_course_table="sphere_time_course",
                upstream_table="sphere_upstream_runs")
    l.set_defaults(**tabs)
    c = sub.add_parser("compare", help="the model-comparison batch's arms -> one run directory per arm x rung")
    common_args(c, "out/objr2c", "out/interp/objr2c/compare.json", "arm", "objr2c")
    c.set_defaults(**tabs)

    t = sub.add_parser("tables"); t.add_argument("--run-dir", nargs="+", required=True)
    t.add_argument("--json", default=None); t.add_argument("--csv", default=None); t.set_defaults(func=cmd_tables)
    s = sub.add_parser("slim", help="rewrite run npz files without the named arrays (run on the box, for the fetch)")
    s.add_argument("--dir", required=True); s.add_argument("--slim-out", required=True)
    s.add_argument("--drop", nargs="*", default=list(SLIM_DROP))
    s.add_argument("--require-json", action="store_true", default=True)
    s.add_argument("--overwrite", action="store_true"); s.set_defaults(func=cmd_slim)
    k = sub.add_parser("check", help="independent checks of a finished per-rung directory (the audit's numbers)")
    k.add_argument("--run-dir", nargs=1, required=True); k.add_argument("--out", default="out/objr2")
    k.add_argument("--summary-dir", default=None); k.add_argument("--json", default=None)
    k.set_defaults(func=cmd_check)
    v = sub.add_parser("verify"); v.add_argument("--run-dir", nargs="+", required=True)
    v.add_argument("--paired", default="LC11,LC10a"); v.add_argument("--expect-counts", default=None)
    v.set_defaults(func=cmd_verify)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "family"):
        args.family_spec = _family(args.family)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
