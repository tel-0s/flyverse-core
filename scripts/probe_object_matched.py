"""The MATCHED ray-traced sphere assay (Neurome intake, recommended experiment 1; docs/audits/object_matched_assay.md).

What the old ladder (scripts/probe_object_sweep.py) confounded, and what this probe holds constant:

  * the ball RESTED ON THE TABLE, so its centre elevation grew with its radius (0.88 / 4.34 / 8.66 / 13.71 deg across
    4.5 / 11.4 / 20 / 30 deg) -- here the centre is placed FLY-RELATIVE at a fixed elevation (--elevation-deg, default
    0) and a fixed distance from the eye (--distance-m, default 0.05), independent of the radius;
  * it moved on a LATERAL LINE at constant world speed (45.8 deg/s at the centre, 18.8 at the ends; the angular
    diameter shrank 11.4 -> 7.3 deg along the sweep) -- here it moves on an ARC of constant distance at constant
    angular speed (--deg-per-s, default 40; a triangle wave between +-(--az-max), default 50 deg, left -> right first),
    so angular diameter and angular speed are constant along the trajectory;
  * the retina table was a geometry REPLAY at a pinned pose and the blank radiance was omitted -- here `sim.col_rad`,
    the (n_col, 4) radiance the optic lobe received, is stored after EVERY sim.step() of BOTH arms into the run's npz
    (`retina.mode = "in_loop_capture"` in the JSON), with the per-frame object track (azimuth, elevation, angular
    diameter as seen from the eye).

Two scene facts decide two flags (numbers in the audit, section 2):

  * the eye of a fly standing on the table is 1.2 mm above it, so a ball at elevation 0 has its lower limb clipped by
    the table top (the table horizon at the ball's tangent distance is -1.4 deg): every diameter would be clipped, by
    a size-dependent fraction. --eye-height-m (default 0.15, the tethered-fly analogue: the body stays on the table,
    the eye is raised) puts the substrate below -16.7 deg at the table edge; the probe REFUSES a configuration whose
    lower limb is below the substrate horizon unless --allow-substrate-clip is given (the check is recorded);
  * the room's ceiling lamp makes the ball cast a moving, size-dependent shadow on the table. --light eye (default)
    puts the point light at the eye (a headlamp: every shadow is hidden behind its caster, exactly); --light ceiling
    keeps the room's lamp (the shadow is then part of the captured stimulus and shows in the footprint diagnostic).

Arms: A = object, B = blank (same timeline, ball parked at (9, 9, 9), outside the room); --null makes A a second blank
(blank/blank: the null distribution of every diff_* statistic). One process = one run = one (A, B) pair with one brain
seed, as probe_object_sweep does; runs are the replicate unit (docs/INTERP.md 2.4).

Recorded per frame (10 ms) through flyverse.interp.common.Recorder over the scored window: LC11 / LC10a / LC10b /
LPLC2 / LC4 (drive_mv, spike_count) and T2 / T3 / Tm5Y / TmY21 / TmY13 / TmY5a / Mi4 / Tm3 / Mi1 (optic_dr).
Statistics: probe_object_sweep's (diff_max_over_cells_mean_mv, diff_abs_best_cell_mean, diff_signed_best_cell, the
rates), PLUS per-body time-mean tables (before any population maximum) and an RF-windowed variant when --rf-map names
a localizer (CSV bodyId,az_deg,el_deg,width_deg[,height_deg], or a Result JSON with a table `rf_map`).

    python scripts/probe_object_matched.py run --diam-deg 11 --seed 0 --out out/objm/smoke/d110_obj_s0
    python scripts/probe_object_matched.py run --diam-deg 11 --seed 0 --null --out out/objm/smoke/d110_null_s0
    python scripts/probe_object_matched.py verify out/objm/smoke --json out/objm/smoke/verify.json      # CPU
    python scripts/probe_object_matched.py geometry --diam-deg 4.5 11 20 30                              # CPU

Every GPU run goes through scripts/cluster_run.py (the cluster rule); a CPU smoke needs --device cpu --allow-cpu
(CUDA_VISIBLE_DEVICES=-1 is set before torch is imported). Nothing under flyverse/ is edited by this probe; the model
is read, not changed (OpticParams / LIFParams overrides go through --optic / --lif as in the interp tools).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flyverse.interp import common                 # noqa: E402  (CPU-only import: no torch, no pygame)

SPIKING = ["LC11", "LC10a", "LC10b", "LPLC2", "LC4"]
RATE = ["T2", "T3", "Tm5Y", "TmY21", "TmY13", "TmY5a", "Mi4", "Tm3", "Mi1"]
POS = (-0.20, 0.10, 0.75)            # probe_object_sweep's pinned pose: on the table top, facing -y (the plain wall)
HEADING = -np.pi / 2
FRAME_S = 0.01
DIAMETERS_DEG = [4.5, 11.0, 20.0, 30.0]
EXPECTED_BODIES = {"LC11": 143, "LC10a": 275}   # docs/NEUROME_INTERFACE.md section 3 (the anatomy / readout join)
PARK = (9.0, 9.0, 9.0)               # outside the 4 x 4 x 2.6 m room: invisible
N_AZ_BINS = 20                        # sweep-locked tuning bins over [-az_max, az_max]
MATERIALS = ("black", "plate", "lamp")   # dark (refl 0.01-0.02) / bright reflective (0.6-0.95, shaded) / bright emissive (0.6-1.0, uniform)


# ================================================================================================ geometry (CPU, pure numpy)
def radius_for(diam_deg: float, dist_m: float) -> float:
    """Ball radius (m) whose angular diameter from the eye at `dist_m` (eye -> centre) is `diam_deg`: 2 asin(r / d)."""
    return float(dist_m * np.sin(np.radians(diam_deg) / 2))


def angular_diameter_deg(r_m: float, dist_m: float) -> float:
    return float(2 * np.degrees(np.arcsin(min(1.0, r_m / dist_m))))


def azimuth_track(t_s, az_max_deg: float, deg_per_s: float) -> np.ndarray:
    """Azimuth (deg, + left) at sweep time t: a triangle wave at constant |d az / dt| = deg_per_s between +az_max and
    -az_max, starting at +az_max (left) and moving right, as probe_object_sweep's lateral sweep did."""
    t = np.asarray(t_s, dtype=np.float64)
    one_way = 2 * az_max_deg / deg_per_s
    phase = np.mod(t, 2 * one_way)
    return np.where(phase < one_way, az_max_deg - deg_per_s * phase, -az_max_deg + deg_per_s * (phase - one_way))


def direction(az_deg, el_deg) -> np.ndarray:
    """Body-frame unit vector(s) (x fwd, y left, z up) of (azimuth, elevation) -- flyverse/retina.py's convention."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def centre_world(eye, fwd, left, up, az_deg: float, el_deg: float, dist_m: float) -> np.ndarray:
    """World position of the ball centre at (az, el) and distance `dist_m` from the eye, in the fly's body frame."""
    d = direction(az_deg, el_deg)
    return np.asarray(eye, float) + dist_m * (d[0] * np.asarray(fwd, float) + d[1] * np.asarray(left, float) + d[2] * np.asarray(up, float))


def seen_from_eye(centre, eye, fwd, left, up, r_m: float) -> dict:
    """What the eye sees of a ball at `centre`: azimuth / elevation of its centre, its distance and angular diameter."""
    v = np.asarray(centre, float) - np.asarray(eye, float)
    dist = float(np.linalg.norm(v))
    x, y, z = float(v @ fwd), float(v @ left), float(v @ up)
    return {"az_deg": float(np.degrees(np.arctan2(y, x))), "el_deg": float(np.degrees(np.arcsin(np.clip(z / dist, -1, 1)))),
            "dist_m": dist, "diam_deg": angular_diameter_deg(r_m, dist)}


def make_track(n_frames: int, az_max_deg: float, deg_per_s: float, el_deg: float, dist_m: float, diam_deg: float,
               frame_s: float = FRAME_S) -> dict:
    """The intended per-frame track: t_s, az_deg, el_deg, diam_deg, dist_m, speed_deg_s (|d az / dt| by forward
    difference; the turn frames carry the mean of the two legs and are flagged in `turn`)."""
    t = np.arange(n_frames) * frame_s
    az = azimuth_track(t, az_max_deg, deg_per_s)
    speed = np.full(n_frames, np.nan)
    if n_frames > 1:
        speed[:-1] = np.abs(np.diff(az)) / frame_s
        speed[-1] = speed[-2]
    one_way = 2 * az_max_deg / deg_per_s
    leg = np.floor((t + 1e-9) / one_way); leg_next = np.floor((t + frame_s + 1e-9) / one_way)
    turn = leg != leg_next                                                  # the frame whose step crosses a reversal
    return {"t_s": t, "az_deg": az, "el_deg": np.full(n_frames, float(el_deg)), "diam_deg": np.full(n_frames, float(diam_deg)),
            "dist_m": np.full(n_frames, float(dist_m)), "speed_deg_s": speed, "turn": turn,
            "one_way_s": float(one_way), "period_s": float(2 * one_way)}


def substrate_clearance(el_deg: float, diam_deg: float, dist_m: float, eye_height_m: float) -> dict:
    """Does the substrate the fly stands on (a plane `eye_height_m` below the eye) clip the ball's lower limb?

    The lower limb is at elevation el - diam/2; the ray to it touches the sphere at the tangent distance
    sqrt(d^2 - r^2), and a ray at elevation e < 0 meets the substrate plane at distance eye_height / sin(-e): the limb
    is clear when the tangent point comes first, i.e. when e_low > -asin(eye_height / tangent_distance). With the eye
    1.2 mm above the table and the ball 5 cm away the horizon is -1.4 deg, so every ball at elevation 0 is clipped
    (a size-dependent fraction); at 0.15 m the horizon is below -90 deg (the whole ball is in the clear)."""
    r = radius_for(diam_deg, dist_m)
    tangent = float(np.sqrt(max(dist_m ** 2 - r ** 2, 1e-12)))
    ratio = eye_height_m / tangent
    horizon = -90.0 if ratio >= 1.0 else float(-np.degrees(np.arcsin(ratio)))
    e_low = float(el_deg - diam_deg / 2)
    return {"lower_limb_deg": e_low, "upper_limb_deg": float(el_deg + diam_deg / 2), "substrate_horizon_deg": horizon,
            "clearance_deg": e_low - horizon, "clear": bool(e_low > horizon), "tangent_distance_m": tangent,
            "eye_height_m": float(eye_height_m), "radius_m": r}


def geometry_record(diam_deg: float, dist_m: float, el_deg: float, az_max_deg: float, deg_per_s: float, eye_height_m: float) -> dict:
    r = radius_for(diam_deg, dist_m)
    return {"diam_deg": float(diam_deg), "radius_m": r, "distance_m": float(dist_m), "elevation_deg": float(el_deg),
            "az_max_deg": float(az_max_deg), "deg_per_s": float(deg_per_s), "one_way_s": float(2 * az_max_deg / deg_per_s),
            "angular_diameter_check_deg": angular_diameter_deg(r, dist_m), "substrate": substrate_clearance(el_deg, diam_deg, dist_m, eye_height_m)}


# ================================================================================================ radiance footprint (CPU)
def footprint(rad_obj: np.ndarray, rad_blank: np.ndarray, col_az_el: np.ndarray, dim_thresh: float = 0.5,
              change_thresh: float = 0.05, eps: float = 1e-9) -> dict:
    """Neurome's own diagnostic from the CAPTURED radiance of the two arms: per frame the columns whose summed
    radiance the object dims by more than `dim_thresh` (50 %), those it brightens by more than `dim_thresh` (a
    bright object), those it changes by more than `change_thresh` (5 %), the unweighted az / el centroid of the
    strongly changed set (dimmed OR brightened, so a `lamp` ball has a centroid too), and the blank (background)
    luminance under it. `frames_with_dimmed` counts frames with a strongly changed column (either sign)."""
    lum_o = np.asarray(rad_obj, np.float64).sum(-1); lum_b = np.asarray(rad_blank, np.float64).sum(-1)     # (T, n_col)
    rel = (lum_o - lum_b) / (lum_b + eps)
    dimmed = rel < -dim_thresh; changed = np.abs(rel) > change_thresh; brightened = rel > dim_thresh
    strong = dimmed | brightened
    T = rel.shape[0]
    az, el = np.asarray(col_az_el)[:, 0], np.asarray(col_az_el)[:, 1]
    n_d = dimmed.sum(1); n_c = changed.sum(1); n_b = brightened.sum(1); n_s = strong.sum(1)
    cen_az = np.full(T, np.nan); cen_el = np.full(T, np.nan); bg = np.full(T, np.nan); el_lo = np.full(T, np.nan); el_hi = np.full(T, np.nan)
    for j in range(T):
        m = strong[j]
        if m.any():
            cen_az[j] = az[m].mean(); cen_el[j] = el[m].mean(); bg[j] = lum_b[j, m].mean(); el_lo[j] = el[m].min(); el_hi[j] = el[m].max()
    frames_with = n_s > 0
    static = float(np.abs(lum_b - lum_b[0:1]).max()) if T else float("nan")
    return {"per_frame": {"n_dimmed_50pct": n_d, "n_changed_5pct": n_c, "n_brightened_50pct": n_b, "n_strong_50pct": n_s, "centroid_az_deg": cen_az,
                          "centroid_el_deg": cen_el, "el_min_deg": el_lo, "el_max_deg": el_hi, "blank_lum_under_object": bg,
                          "min_rel": rel.min(1) if T else np.zeros(0), "max_rel": rel.max(1) if T else np.zeros(0)},
            "summary": {"frames": int(T), "frames_with_dimmed": int(frames_with.sum()),
                        "n_dimmed_50pct_mean": float(n_d.mean()) if T else float("nan"),
                        "n_changed_5pct_mean": float(n_c.mean()) if T else float("nan"),
                        "n_brightened_50pct_mean": float(n_b.mean()) if T else float("nan"),
                        "n_strong_50pct_mean": float(n_s.mean()) if T else float("nan"),
                        "centroid_set": "columns with |rel change| > dim_thresh (dimmed or brightened)",
                        "centroid_el_mean_deg": float(np.nanmean(cen_el)) if frames_with.any() else float("nan"),
                        "centroid_el_sd_deg": float(np.nanstd(cen_el)) if frames_with.any() else float("nan"),
                        "centroid_el_min_deg": float(np.nanmin(cen_el)) if frames_with.any() else float("nan"),
                        "centroid_el_max_deg": float(np.nanmax(cen_el)) if frames_with.any() else float("nan"),
                        "centroid_az_min_deg": float(np.nanmin(cen_az)) if frames_with.any() else float("nan"),
                        "centroid_az_max_deg": float(np.nanmax(cen_az)) if frames_with.any() else float("nan"),
                        "dimmed_el_band_deg": [float(np.nanmin(el_lo)), float(np.nanmax(el_hi))] if frames_with.any() else [None, None],
                        "blank_lum_under_object_mean": float(np.nanmean(bg)) if frames_with.any() else float("nan"),
                        "blank_lum_under_object_cv": float(np.nanstd(bg) / max(np.nanmean(bg), 1e-12)) if frames_with.any() else float("nan"),
                        "blank_static_max_abs_diff": static, "dim_thresh": dim_thresh, "change_thresh": change_thresh}}


# ================================================================================================ statistics (CPU)
def summarize_arms(A: dict, B: dict, window_s: float, track_az: np.ndarray, az_max: float) -> tuple[dict, pd.DataFrame]:
    """probe_object_sweep.summarize's statistics (scripts/probe_object_sweep.py:247-294, same names, same definitions:
    every diff_* field is arm A minus arm B) on per-frame per-cell arrays, plus the per-body table.

    A / B: {"drive": {type: (T, n)}, "spk": {type: (T, n) spikes per frame}, "dr": {type: (T, n)}, "cells": {type: idx},
    "bodies": {type: bodyIds}}. The sweep-locked tuning is binned on the track AZIMUTH (degrees), not metres."""
    out = {}; rows = []
    bins = np.clip(((track_az + az_max) / (2 * az_max) * N_AZ_BINS).astype(int), 0, N_AZ_BINS - 1)
    cnt = np.bincount(bins, minlength=N_AZ_BINS)

    def tuning(fr):
        t = np.zeros((N_AZ_BINS, fr.shape[1])); np.add.at(t, bins[:fr.shape[0]], fr)
        return t / np.maximum(cnt, 1)[:, None]

    for t in A["drive"]:
        fa, fb = A["drive"][t], B["drive"][t]; n = min(len(fa), len(fb)); fa, fb = fa[:n], fb[:n]
        ma, mb = fa.mean(0), fb.mean(0); md = ma - mb
        hz_a = A["spk"][t][:n].sum(0) / window_s; hz_b = B["spk"][t][:n].sum(0) / window_s
        ta, tb = tuning(fa), tuning(fb)
        out[t] = {"n_cells": int(len(ma)), "kind": "spiking",
                  "drive_mean_mv": float(ma.mean()), "drive_best_cell_mean_mv": float(ma.max()), "drive_peak_mv": float(fa.max()),
                  "drive_mean_mv_b": float(mb.mean()), "drive_best_cell_mean_mv_b": float(mb.max()), "drive_peak_mv_b": float(fb.max()),
                  "rate_hz_mean": float(hz_a.mean()), "rate_hz_max_cell": float(hz_a.max()), "cells_over_1hz": int((hz_a > 1.0).sum()),
                  "rate_hz_mean_b": float(hz_b.mean()), "rate_hz_max_cell_b": float(hz_b.max()), "cells_over_1hz_b": int((hz_b > 1.0).sum()),
                  "cells_firing": int((hz_a > 0).sum()), "cells_firing_b": int((hz_b > 0).sum()),
                  # diff_max_over_cells_mean_mv: max OVER CELLS of the per-cell (A - B) time-mean drive (the headline
                  # statistic of the ladder; a within-run maximum over cells, not a membrane voltage)
                  "diff_max_over_cells_mean_mv": float(md.max()), "diff_mean_over_cells_mean_mv": float(md.mean()),
                  "diff_argmax_body": str(int(A["bodies"][t][int(np.argmax(md))])),
                  "diff_tuning_peak_mv": float((ta - tb).max()), "tuning_bins_frames": [int(v) for v in cnt],
                  "diff_rate_hz_max_cell": float((hz_a - hz_b).max()), "diff_rate_hz_mean": float((hz_a - hz_b).mean())}
        for i in range(len(ma)):
            rows.append({"bodyId": str(int(A["bodies"][t][i])), "model_index": int(A["cells"][t][i]), "type": t, "unit_kind": "spiking",
                         "quantity": "drive_mv", "a_mean": float(ma[i]), "b_mean": float(mb[i]), "diff": float(md[i]),
                         "a_sd_frames": float(fa[:, i].std()), "b_sd_frames": float(fb[:, i].std()), "n_frames": int(n)})
            rows.append({"bodyId": str(int(A["bodies"][t][i])), "model_index": int(A["cells"][t][i]), "type": t, "unit_kind": "spiking",
                         "quantity": "rate_hz", "a_mean": float(hz_a[i]), "b_mean": float(hz_b[i]), "diff": float(hz_a[i] - hz_b[i]),
                         "a_sd_frames": float(A["spk"][t][:n, i].std() / FRAME_S), "b_sd_frames": float(B["spk"][t][:n, i].std() / FRAME_S), "n_frames": int(n)})
    for t in A["dr"]:
        fa, fb = A["dr"][t], B["dr"][t]; n = min(len(fa), len(fb)); fa, fb = fa[:n], fb[:n]
        ma, mb = fa.mean(0), fb.mean(0); aa, ab = np.abs(fa).mean(0), np.abs(fb).mean(0)
        out[t] = {"n_cells": int(len(ma)), "kind": "rate",
                  "dev_mean": float(ma.mean()), "dev_abs_mean": float(aa.mean()), "dev_abs_best_cell_mean": float(aa.max()),
                  "dev_max": float(fa.max()), "dev_min": float(fa.min()),
                  "dev_mean_b": float(mb.mean()), "dev_abs_mean_b": float(ab.mean()), "dev_abs_best_cell_mean_b": float(ab.max()),
                  # the two upstream statistics stay DISTINCT (Neurome intake): |deviation| vs signed
                  "diff_abs_mean": float((aa - ab).mean()), "diff_abs_best_cell_mean": float((aa - ab).max()),
                  "diff_signed_mean": float((ma - mb).mean()), "diff_signed_best_cell": float(np.abs(ma - mb).max()),
                  "diff_signed_argmax_body": str(int(A["bodies"][t][int(np.argmax(np.abs(ma - mb)))]))}
        for i in range(len(ma)):
            rows.append({"bodyId": str(int(A["bodies"][t][i])), "model_index": int(A["cells"][t][i]), "type": t, "unit_kind": "graded",
                         "quantity": "optic_dr", "a_mean": float(ma[i]), "b_mean": float(mb[i]), "diff": float(ma[i] - mb[i]),
                         "a_sd_frames": float(fa[:, i].std()), "b_sd_frames": float(fb[:, i].std()), "n_frames": int(n)})
            rows.append({"bodyId": str(int(A["bodies"][t][i])), "model_index": int(A["cells"][t][i]), "type": t, "unit_kind": "graded",
                         "quantity": "optic_dr_abs", "a_mean": float(aa[i]), "b_mean": float(ab[i]), "diff": float(aa[i] - ab[i]),
                         "a_sd_frames": float(np.abs(fa[:, i]).std()), "b_sd_frames": float(np.abs(fb[:, i]).std()), "n_frames": int(n)})
    return out, pd.DataFrame(rows)


# ================================================================================================ RF windowing (CPU)
RF_COLUMNS = ("bodyId", "az_deg", "el_deg", "width_deg")


def load_rf_map(path) -> pd.DataFrame:
    """A per-body receptive-field localizer: CSV with bodyId, az_deg, el_deg, width_deg[, height_deg] (height defaults
    to width), or a Result JSON carrying a table `rf_map` with those columns. Written against the documented CSV
    interface; the synthetic-stimuli task's `rfmap --csv` (scripts/probe_synthetic_stimuli.py: bodyId, type, az_deg,
    el_deg, width_deg, peak, n_nodes_above_threshold, fitted, ...) is read through the same columns: a row whose
    `fitted` is false, or whose centre / width is not finite, has no localizer and is dropped (its rf_* fields stay
    NaN in the per-body table -- never 0)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        recs = d.get("tables", {}).get("rf_map") if isinstance(d, dict) else None
        if recs is None and isinstance(d, list):
            recs = d
        if recs is None:
            raise ValueError(f"{path}: no tables.rf_map (and not a list of records)")
        df = pd.DataFrame(recs)
    else:
        df = pd.read_csv(path)
    missing = [c for c in RF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: RF map lacks columns {missing}; expected {RF_COLUMNS} (+ optional height_deg)")
    if "height_deg" not in df.columns:
        df["height_deg"] = df["width_deg"]
    df["bodyId"] = df["bodyId"].astype(np.int64).astype(str)
    ok = np.isfinite(df[["az_deg", "el_deg", "width_deg", "height_deg"]].to_numpy(dtype=float)).all(1)
    if "fitted" in df.columns:
        ok &= df["fitted"].astype(str).str.lower().isin(("true", "1", "1.0")).to_numpy()
    df = df[ok]
    return df[list(RF_COLUMNS) + ["height_deg"]].drop_duplicates("bodyId").reset_index(drop=True)


def rf_frames(track_az, track_el, track_diam, rf_az: float, rf_el: float, width_deg: float, height_deg: float) -> np.ndarray:
    """Frames on which the object disc overlaps the body's RF box: |d az| <= (width + diam) / 2 and
    |d el| <= (height + diam) / 2 (the disc touches the rectangle; a box window matches a rectangle localizer)."""
    daz = np.abs(np.asarray(track_az) - rf_az); dele = np.abs(np.asarray(track_el) - rf_el); d2 = np.asarray(track_diam) / 2
    return (daz <= width_deg / 2 + d2) & (dele <= height_deg / 2 + d2)


def rf_windowed(per_body: pd.DataFrame, frames: dict, rf: pd.DataFrame, track: dict) -> pd.DataFrame:
    """Per-body means over the body's RF frames (the same frames in both arms) joined onto the per-body table:
    rf_az_deg, rf_el_deg, rf_n_frames, rf_a_mean, rf_b_mean, rf_diff (NaN for a body without a localizer row).
    `frames`: {(type, quantity): (A (T, n), B (T, n), bodyIds)}."""
    per_body = per_body.copy()
    for k in ("rf_az_deg", "rf_el_deg", "rf_width_deg", "rf_n_frames", "rf_a_mean", "rf_b_mean", "rf_diff"):
        per_body[k] = np.nan
    rfi = rf.set_index("bodyId")
    for (t, q), (fa, fb, bodies) in frames.items():
        for i, b in enumerate(bodies):
            b = str(int(b))
            if b not in rfi.index:
                continue
            r = rfi.loc[b]
            m = rf_frames(track["az_deg"], track["el_deg"], track["diam_deg"], float(r.az_deg), float(r.el_deg), float(r.width_deg), float(r.height_deg))
            n = min(len(fa), len(fb)); m = m[:n]
            sel = (per_body.bodyId == b) & (per_body.type == t) & (per_body.quantity == q)
            va = float(fa[:n][m, i].mean()) if m.any() else np.nan; vb = float(fb[:n][m, i].mean()) if m.any() else np.nan
            per_body.loc[sel, ["rf_az_deg", "rf_el_deg", "rf_width_deg", "rf_n_frames", "rf_a_mean", "rf_b_mean", "rf_diff"]] = \
                [float(r.az_deg), float(r.el_deg), float(r.width_deg), int(m.sum()), va, vb, va - vb]
    return per_body


def rf_population_stats(per_body: pd.DataFrame) -> dict:
    """The population statistics of the RF-windowed per-body diffs, per type: max over cells / mean over cells of
    rf_diff (drive_mv for spiking types; optic_dr signed and optic_dr_abs for graded), and how many bodies had a
    localizer row with >= 1 RF frame."""
    out = {}
    if "rf_diff" not in per_body.columns:
        return out
    for (t, q), g in per_body.groupby(["type", "quantity"]):
        v = g.rf_diff.to_numpy(dtype=float); ok = np.isfinite(v) & (g.rf_n_frames.to_numpy(dtype=float) > 0)
        if not ok.any():
            out.setdefault(t, {})[q] = {"n_bodies_windowed": 0}
            continue
        vv = v[ok]
        out.setdefault(t, {})[q] = {"n_bodies_windowed": int(ok.sum()), "rf_diff_max_over_cells": float(vv.max()),
                                    "rf_diff_mean_over_cells": float(vv.mean()), "rf_diff_abs_max_over_cells": float(np.abs(vv).max()),
                                    "rf_frames_mean": float(g.rf_n_frames.to_numpy(dtype=float)[ok].mean())}
    return out


# ================================================================================================ the rollout (GPU)
def _arm_label(with_object: bool) -> str:
    return "object" if with_object else "blank"


def run_arm(args, seed: int, with_object: bool, geom: dict) -> dict:
    """One arm: a fresh room_demo.Sim (same seed for both arms of a run), the pinned pose re-placed every frame, the ball
    on its arc (or parked), `sim.col_rad` captured after every step, the Recorders over the scored window."""
    import torch
    import room_demo as rd
    from flyverse.interp import trace as tr
    flags = dict(cuda_kernels=True, cuda_graphs=False, event_driven=True, cuda_sparse="warp") if (torch.cuda.is_available() and not args.allow_cpu) else {}
    sim = rd.Sim(seed, start=POS, trail_seconds=0.0, fruit_set="apple", fence=True, wind_speed=0.0, **flags)   # receptor / LIF / optic overrides: install_overrides
    dev = sim.fb.brain.device
    print(f"[{_arm_label(with_object)}] sim ready; device {dev} (requested {args.device}); cuda available {torch.cuda.is_available()}", flush=True)
    if not args.allow_cpu:
        assert dev.type == "cuda", f"device {dev}: not CUDA (node race; resubmit)"
    for i, s in enumerate(sim.world.spheres):                    # empty table: the apple goes out of the scene
        if s.material == "apple":
            sim.world.move_sphere(i, PARK)
    r = geom["radius_m"]
    ball = sim.world.spheres[sim.loom_idx]
    ball.material = args.material                                # the tracer packs materials per object: repack below
    sim.world.move_sphere(sim.loom_idx, PARK, (r, r, r))
    fly = sim.fly
    fly.place(*POS, heading=HEADING); fly.eye_height = args.eye_height_m
    eye = fly.eye_pos.copy(); fwd = fly.forward.copy(); left = fly.left.copy(); up = fly.up.copy()
    light_before = tuple(float(v) for v in sim.world.light_pos)
    if args.light == "eye":
        sim.world.light_pos = tuple(float(v) for v in eye)     # headlamp: no shadow is visible from the eye
    sim.world.invalidate()                                       # material / light edits -> repack on the next trace
    n_settle = int(round(args.settle / FRAME_S)); n_sweep = int(round(args.seconds / FRAME_S))
    track = make_track(n_sweep, args.az_max, args.deg_per_s, args.elevation_deg, args.distance_m, geom["diam_deg"])
    c = sim.c; fb = sim.fb; b = fb.brain
    rec_lc = common.Recorder(c, SPIKING, quantities=("drive_mv", "spike_count"))
    rec_rate = common.Recorder(c, RATE, quantities=("optic_dr",))
    n_col = fb.retina.n_columns
    rad = np.zeros((n_sweep, n_col, 4), np.float32)
    seen = {k: np.full(n_sweep, np.nan) for k in ("az_deg", "el_deg", "dist_m", "diam_deg")}
    centres = np.full((n_sweep, 3), np.nan)
    count0 = None
    t0 = time.time()
    print(f"  eye {np.round(eye, 4).tolist()} (height {args.eye_height_m} m above the body) fwd {np.round(fwd, 3).tolist()} left {np.round(left, 3).tolist()}; "
          f"ball r {r * 1000:.3f} mm at {args.distance_m} m, {geom['diam_deg']} deg, elevation {args.elevation_deg} deg, +-{args.az_max} deg at {args.deg_per_s} deg/s; "
          f"material {args.material}; light {args.light} {tuple(round(float(v), 3) for v in sim.world.light_pos)} (room lamp {tuple(round(v, 3) for v in light_before)})", flush=True)
    for k in range(n_settle + n_sweep):
        fly.place(*POS, heading=HEADING); fly.eye_height = args.eye_height_m
        j = k - n_settle
        if with_object and j >= 0:
            cen = centre_world(eye, fwd, left, up, float(track["az_deg"][j]), args.elevation_deg, args.distance_m)
            sim.world.move_sphere(sim.loom_idx, cen)
            s = seen_from_eye(cen, fly.eye_pos, fly.forward, fly.left, fly.up, r)   # from the pinned pose the trace below uses
            for kk in seen:
                seen[kk][j] = s[kk]
            centres[j] = cen
        else:
            sim.world.move_sphere(sim.loom_idx, PARK)
        sim.step()                                                # traces at the pinned pose, then fb.vision(col_rad), then the brain step
        if k == n_settle - 1:
            count0 = common._np(b.spike_counts)[0].copy()
        if j < 0:
            continue
        rad[j] = sim.col_rad.detach().cpu().numpy()             # IN-LOOP capture: the radiance fb.vision() received this frame
        rec_lc.capture(fb); rec_rate.capture(fb)
        if j % 300 == 0:
            d = common._np(b.drive)[0]
            ty = rec_lc.types
            print(f"  t {j * FRAME_S:5.1f} s  az {track['az_deg'][j]:+6.1f}  LC11 drive max {d[rec_lc.idx[ty == 'LC11']].max():+.2f} mV  "
                  f"LC10a {d[rec_lc.idx[ty == 'LC10a']].max():+.2f}  LPLC2 {d[rec_lc.idx[ty == 'LPLC2']].max():+.2f}  ({time.time() - t0:.0f} s wall)", flush=True)
    if count0 is None:
        count0 = np.zeros(c.n, np.float32)
    lc = rec_lc.finish({"arm": _arm_label(with_object)}); rt = rec_rate.finish({"arm": _arm_label(with_object)})
    types_lc = lc.types; types_rt = rt.types
    sc = lc.quantities["spike_count"]                             # cumulative (T, n)
    spk = np.diff(np.concatenate([count0[lc.idx][None], sc], 0), axis=0)
    drive = lc.quantities["drive_mv"]; dr = rt.quantities["optic_dr"]
    assert np.isfinite(dr).all(), "a recorded rate-type cell is not an optic rate unit"
    arm = {"drive": {t: drive[:, types_lc == t] for t in SPIKING}, "spk": {t: spk[:, types_lc == t] for t in SPIKING}, "spk_all": spk,
           "dr": {t: dr[:, types_rt == t] for t in RATE},
           "cells": {**{t: lc.idx[types_lc == t] for t in SPIKING}, **{t: rt.idx[types_rt == t] for t in RATE}},
           "bodies": {**{t: lc.body_ids[types_lc == t] for t in SPIKING}, **{t: rt.body_ids[types_rt == t] for t in RATE}},
           "rad": rad, "track": track, "seen": seen, "centres": centres, "lc": lc, "rt": rt, "count0": count0[lc.idx],
           "eye": eye, "fwd": fwd, "left": left, "up": up, "light_pos": tuple(float(v) for v in sim.world.light_pos), "room_lamp": light_before,
           "device": str(dev), "wall_s": time.time() - t0, "window_s": n_sweep * FRAME_S}
    # provenance while the brain is alive (the realised device, backend flags, dt); the stimulus / retina blocks are
    # filled in by cmd_run (they are plain dicts on the returned record)
    arm["retina"] = dict(tr.retina_record(fb.retina, c), col_az_el=np.asarray(fb.retina.col_az_el).tolist())
    arm["col_az_el"] = np.asarray(fb.retina.col_az_el).copy(); arm["col_side"] = np.asarray(fb.retina.col_side).astype(str)
    arm["prov"] = common.provenance(c, fb.brain.p, fb.optic.p, fb=fb, device=args.device, seeds=[seed], env_seeds=[seed], batch=1,
                                    stimulus=None, retina=None, cache_dir=args.cache_dir)
    return arm, sim


def cmd_run(args) -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if args.device and str(args.device).startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # a CPU smoke never touches this machine's GPU (docs/INTERP.md 10.4 rule 5)
        args.allow_cpu = True
    import torch
    if not args.allow_cpu:
        assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    import interp_trace as it                                   # the interp tools' override installer (LIF / optic / receptor / cache)
    it.install_overrides(common.parse_kv(args.lif), common.parse_kv(args.optic), args.receptor_model, args.receptor_net_rule, args.receptor_table)
    it.patch_cache(args.cache_dir)
    if args.material not in MATERIALS:
        raise SystemExit(f"--material {args.material}: choose from {MATERIALS}")
    geom = geometry_record(args.diam_deg, args.distance_m, args.elevation_deg, args.az_max, args.deg_per_s, args.eye_height_m)
    sub = geom["substrate"]
    print(f"matched sphere assay{' NULL (blank vs blank)' if args.null else ''}: {geom['diam_deg']} deg (r {geom['radius_m'] * 1000:.3f} mm at {args.distance_m} m), "
          f"elevation {args.elevation_deg} deg, arc +-{args.az_max} deg at {args.deg_per_s} deg/s (one way {geom['one_way_s']:.2f} s), eye height {args.eye_height_m} m; "
          f"lower limb {sub['lower_limb_deg']:+.2f} deg vs substrate horizon {sub['substrate_horizon_deg']:+.2f} deg -> {'clear' if sub['clear'] else 'CLIPPED'}; "
          f"seed {args.seed}; window {args.seconds} s after {args.settle} s settle; torch {torch.__version__}", flush=True)
    if not sub["clear"] and not args.allow_substrate_clip:
        raise SystemExit("the ball's lower limb is below the substrate horizon (the table clips it, size-dependently): raise --eye-height-m or "
                         "--elevation-deg, or pass --allow-substrate-clip to record it anyway (the JSON records the check)")
    seed = args.seed
    A, sim_a = run_arm(args, seed, not args.null, geom)
    del sim_a                                                    # one brain on the GPU at a time (probe_object_sweep's pattern)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    B, sim_b = run_arm(args, seed, False, geom)
    dev_b = B["device"]
    del sim_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ------------------------------------------------------------------ statistics
    window_s = A["window_s"]; track = A["track"]
    per_type, per_body = summarize_arms(A, B, window_s, track["az_deg"], args.az_max)
    frames = {(t, "drive_mv"): (A["drive"][t], B["drive"][t], A["bodies"][t]) for t in SPIKING}
    frames.update({(t, "optic_dr"): (A["dr"][t], B["dr"][t], A["bodies"][t]) for t in RATE})
    rf_stats = None; rf_map_rec = None
    if args.rf_map:
        rf = load_rf_map(args.rf_map)
        per_body = rf_windowed(per_body, frames, rf, track)
        rf_stats = rf_population_stats(per_body)
        rf_map_rec = {"file": str(args.rf_map), "n_bodies": int(len(rf)), "window_rule": "disc overlaps the RF box: |d az| <= (width + diam)/2 and |d el| <= (height + diam)/2"}
    col_az_el = A["col_az_el"]
    fp = footprint(A["rad"], B["rad"], col_az_el)
    fp_b = footprint(B["rad"], B["rad"], col_az_el)               # blank vs itself: exactly 0 changed columns
    track_real = {k: v for k, v in A["seen"].items()}
    seen_ok = np.isfinite(track_real["az_deg"])
    track_check = {"intended_el_deg": float(args.elevation_deg), "realised_el_deg_mean": float(np.nanmean(track_real["el_deg"])) if seen_ok.any() else None,
                   "realised_el_deg_maxdev": float(np.nanmax(np.abs(track_real["el_deg"] - args.elevation_deg))) if seen_ok.any() else None,
                   "realised_diam_deg_mean": float(np.nanmean(track_real["diam_deg"])) if seen_ok.any() else None,
                   "realised_diam_deg_maxdev": float(np.nanmax(np.abs(track_real["diam_deg"] - geom["diam_deg"]))) if seen_ok.any() else None,
                   "realised_dist_m_maxdev": float(np.nanmax(np.abs(track_real["dist_m"] - args.distance_m))) if seen_ok.any() else None,
                   "az_vs_intended_maxdev_deg": float(np.nanmax(np.abs(track_real["az_deg"] - track["az_deg"]))) if seen_ok.any() else None,
                   "speed_deg_s_median": float(np.nanmedian(track["speed_deg_s"][~track["turn"]])), "speed_deg_s_max": float(np.nanmax(track["speed_deg_s"][~track["turn"]])),
                   "speed_deg_s_min": float(np.nanmin(track["speed_deg_s"][~track["turn"]])), "turn_frames": int(track["turn"].sum())}
    # ------------------------------------------------------------------ files
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out.with_suffix(".npz")
    arrays = {"t_s": track["t_s"], "track_az_deg": track["az_deg"], "track_el_deg": track["el_deg"], "track_diam_deg": track["diam_deg"],
              "track_speed_deg_s": track["speed_deg_s"], "track_turn": track["turn"],
              "seen_az_deg": track_real["az_deg"], "seen_el_deg": track_real["el_deg"], "seen_diam_deg": track_real["diam_deg"], "seen_dist_m": track_real["dist_m"],
              "centre_world_m": A["centres"], "eye_world_m": A["eye"], "fwd": A["fwd"], "left": A["left"], "up": A["up"],
              "rad_a": A["rad"], "rad_b": B["rad"], "col_az_el": col_az_el, "col_side": A["col_side"],
              "fp_n_dimmed_50pct": fp["per_frame"]["n_dimmed_50pct"], "fp_n_changed_5pct": fp["per_frame"]["n_changed_5pct"],
              "fp_n_brightened_50pct": fp["per_frame"]["n_brightened_50pct"], "fp_n_strong_50pct": fp["per_frame"]["n_strong_50pct"],
              "fp_centroid_az_deg": fp["per_frame"]["centroid_az_deg"], "fp_centroid_el_deg": fp["per_frame"]["centroid_el_deg"],
              "fp_blank_lum_under_object": fp["per_frame"]["blank_lum_under_object"],
              "lc_idx": A["lc"].idx, "lc_body_ids": A["lc"].body_ids, "lc_types": A["lc"].types.astype(str),
              "a__lc_drive_mv": A["lc"].quantities["drive_mv"], "b__lc_drive_mv": B["lc"].quantities["drive_mv"],
              "a__lc_spikes": A["spk_all"], "b__lc_spikes": B["spk_all"],
              "rate_idx": A["rt"].idx, "rate_body_ids": A["rt"].body_ids, "rate_types": A["rt"].types.astype(str),
              "a__optic_dr": A["rt"].quantities["optic_dr"], "b__optic_dr": B["rt"].quantities["optic_dr"]}
    # sweep-locked per-body tuning along the arc (mean per azimuth bin), both arms: the per-body time course before any maximum
    bins = np.clip(((track["az_deg"] + args.az_max) / (2 * args.az_max) * N_AZ_BINS).astype(int), 0, N_AZ_BINS - 1)
    cnt = np.bincount(bins, minlength=N_AZ_BINS)
    for tag, arm in (("a", A), ("b", B)):
        tun = np.zeros((N_AZ_BINS, arm["lc"].quantities["drive_mv"].shape[1])); np.add.at(tun, bins, arm["lc"].quantities["drive_mv"])
        arrays[f"{tag}__lc_drive_by_az_bin"] = tun / np.maximum(cnt, 1)[:, None]
    arrays["az_bin_edges_deg"] = np.linspace(-args.az_max, args.az_max, N_AZ_BINS + 1); arrays["az_bin_frames"] = cnt
    np.savez_compressed(npz_path, **arrays)
    if args.save_recordings:
        for tag, arm in (("a", A), ("b", B)):
            arm["lc"].save(str(out) + f"_{tag}_lc"); arm["rt"].save(str(out) + f"_{tag}_rate")
    # ------------------------------------------------------------------ the Result
    stim = {"protocol": "object_matched_sphere",
            "params": {"diam_deg": geom["diam_deg"], "radius_m": geom["radius_m"], "distance_m": args.distance_m, "elevation_deg": args.elevation_deg,
                       "az_max_deg": args.az_max, "deg_per_s": args.deg_per_s, "one_way_s": geom["one_way_s"], "material": args.material,
                       "light": args.light, "light_pos_m": A["light_pos"], "room_lamp_m": A["room_lamp"], "eye_height_m": args.eye_height_m,
                       "pos": list(POS), "heading_rad": float(HEADING), "seconds": args.seconds, "settle_s": args.settle, "frame_ms": FRAME_S * 1000,
                       "park": list(PARK), "start": "+az_max (left) moving right, triangle wave"},
            "control": {"arm_b": "blank: same timeline, the ball parked outside the room", "null": bool(args.null),
                        "arm_a": "blank (NULL run)" if args.null else "object"},
            "geometry_check": geom, "track_check": track_check, "rf_map": rf_map_rec}
    retina = dict(A["retina"], file=str(npz_path), mode="in_loop_capture",
                  sampling="sim.col_rad after every sim.step() of BOTH arms (the (n_col, 4) radiance fb.vision() received that frame), "
                           "stored as rad_a / rad_b in the npz; NOT a geometry replay")
    prov = A["prov"]
    prov["stimulus"] = common.to_jsonable(stim); prov["retina"] = common.to_jsonable(retina)
    prov["execution"]["device_by_arm"] = {"a": A["device"], "b": dev_b}
    run_id = f"objm-d{int(round(geom['diam_deg'] * 10)):03d}-{'null' if args.null else 'obj'}-r{seed}"
    res = common.Result.new("trace", prov, validation={"name": "matched sphere assay: constant elevation / diameter / speed along the arc, in-loop capture",
                                                       "reference": {"elevation_deg": args.elevation_deg, "diam_deg": geom["diam_deg"], "deg_per_s": args.deg_per_s},
                                                       "measured": track_check, "status": "geometry checked in-loop (see summary.footprint for the retina)",
                                                       "source": "docs/audits/object_matched_assay.md"})
    res.run_id = run_id; res.tool_version = "probe_object_matched/1"
    for t in SPIKING + RATE:
        res.add_population(common.Population(t, t, np.asarray(A["cells"][t]), np.asarray(A["bodies"][t])), unit_kind="spiking" if t in SPIKING else "graded", keep_ids=False)
    res.replicates = {"n": 1, "unit": "runs", "runs": [run_id], "null": bool(args.null), "seed": seed,
                      "note": "one run = one (A, B) pair under one brain seed; replicate over jobs (>= 5 per arm, one submission) before any verdict"}
    res.add_table("per_type", [dict(type=t, **v) for t, v in per_type.items()])
    res.add_table("per_body", per_body)
    res.add_table("footprint_per_frame", pd.DataFrame({k: v for k, v in fp["per_frame"].items()}).assign(t_s=track["t_s"], az_deg=track["az_deg"]))
    res.summary = {"arms": {"a": "blank" if args.null else "object", "b": "blank"}, "null": bool(args.null), "window_s": window_s, "n_frames": int(len(track["t_s"])),
                   "diff_max_over_cells_mean_mv": {t: per_type[t]["diff_max_over_cells_mean_mv"] for t in SPIKING},
                   "diff_rate_hz_max_cell": {t: per_type[t]["diff_rate_hz_max_cell"] for t in SPIKING},
                   "diff_abs_best_cell_mean": {t: per_type[t]["diff_abs_best_cell_mean"] for t in RATE},
                   "diff_signed_best_cell": {t: per_type[t]["diff_signed_best_cell"] for t in RATE},
                   "diff_signed_mean": {t: per_type[t]["diff_signed_mean"] for t in RATE},
                   "footprint": fp["summary"], "footprint_blank_vs_itself": {"n_changed_5pct_mean": fp_b["summary"]["n_changed_5pct_mean"]},
                   "rf_windowed": rf_stats, "geometry": geom, "track_check": track_check,
                   "statistic_definitions": {
                       "diff_max_over_cells_mean_mv": "max over cells of the per-cell time-mean (A - B) optic drive, mV (scripts/probe_object_sweep.py:277); a within-run maximum, not a membrane voltage",
                       "diff_abs_best_cell_mean": "max over cells of (mean_t |dr_A| - mean_t |dr_B|) (probe_object_sweep.py:291)",
                       "diff_signed_best_cell": "max over cells of |mean_t dr_A - mean_t dr_B| (probe_object_sweep.py:292)",
                       "diff_signed_mean": "mean over cells of (mean_t dr_A - mean_t dr_B)",
                       "rate_hz": "spikes over the scored window / window_s per cell (spike_count differences)",
                       "rf_diff": "per body: mean over the body's RF frames of A minus the same frames of B"},
                   "wall_s": {"a": A["wall_s"], "b": B["wall_s"]}}
    res.files = {"generator": " ".join(sys.argv), "npz": str(npz_path), "json": str(out.with_suffix(".json")),
                 "recordings": [str(out) + f"_{t}_{k}.npz" for t in ("a", "b") for k in ("lc", "rate")] if args.save_recordings else None}
    res.save(out.with_suffix(".json"))
    # ------------------------------------------------------------------ console
    print_run(per_type, fp["summary"], track_check, args, geom, prov, rf_stats)
    problems = res.check()
    print(f"written {out.with_suffix('.json')} and {npz_path}  (check: {problems or 'ok'})")
    if "cpu" in str(prov["execution"]["device"]):
        print("WARNING: device cpu -- resubmit (the JSON records the realised device)")
    return 0


def print_run(per_type: dict, fps: dict, tc: dict, args, geom: dict, prov: dict, rf_stats) -> None:
    a = "blank" if args.null else "object"
    print(f"\n{geom['diam_deg']} deg at elevation {args.elevation_deg} deg, {args.deg_per_s} deg/s, {args.material}, light {args.light}; seed {args.seed}; "
          f"{args.seconds:g} s window after {args.settle:g} s settle; A = {a}, B = blank; device {prov['execution']['device']} ({prov['execution'].get('device_name')})")
    print(f"track: realised elevation {tc['realised_el_deg_mean']} (max dev {tc['realised_el_deg_maxdev']}), diameter {tc['realised_diam_deg_mean']} "
          f"(max dev {tc['realised_diam_deg_maxdev']}), speed {tc['speed_deg_s_min']:.2f}-{tc['speed_deg_s_max']:.2f} deg/s off the {tc['turn_frames']} turn frames")
    print(f"footprint (captured radiance, A vs B): columns dimmed > 50 % {fps['n_dimmed_50pct_mean']:.2f} / frame, changed > 5 % {fps['n_changed_5pct_mean']:.2f}, "
          f"brightened > 50 % {fps['n_brightened_50pct_mean']:.2f}; centroid el {fps['centroid_el_mean_deg']:+.2f} +- {fps['centroid_el_sd_deg']:.2f} deg "
          f"(band {[None if v is None else round(v, 2) for v in fps['dimmed_el_band_deg']]}), az {fps['centroid_az_min_deg']:+.1f}..{fps['centroid_az_max_deg']:+.1f}; blank luminance under the object "
          f"{fps['blank_lum_under_object_mean']:.3f} (cv {fps['blank_lum_under_object_cv']:.2f}); blank static to {fps['blank_static_max_abs_diff']:.2e}")
    print(f"{'type':7s} {'cells':>5s} | {'drive mean mV A/B':>18s} {'best cell mean':>15s} {'peak':>13s} | {'rate Hz mean':>13s} {'max cell':>11s} {'firing':>9s} | diff maxcell / meancell / tuning mV | diff rate max/mean")
    for t in SPIKING:
        v = per_type[t]
        print(f"{t:7s} {v['n_cells']:5d} | {v['drive_mean_mv']:+7.3f}/{v['drive_mean_mv_b']:+7.3f}   {v['drive_best_cell_mean_mv']:+6.3f}/{v['drive_best_cell_mean_mv_b']:+6.3f} "
              f"{v['drive_peak_mv']:+6.2f}/{v['drive_peak_mv_b']:+6.2f} | {v['rate_hz_mean']:6.3f}/{v['rate_hz_mean_b']:6.3f} {v['rate_hz_max_cell']:5.2f}/{v['rate_hz_max_cell_b']:5.2f} "
              f"{v['cells_firing']:4d}/{v['cells_firing_b']:<4d} | {v['diff_max_over_cells_mean_mv']:+.4f} / {v['diff_mean_over_cells_mean_mv']:+.4f} / {v['diff_tuning_peak_mv']:+.3f} | "
              f"{v['diff_rate_hz_max_cell']:+.3f}/{v['diff_rate_hz_mean']:+.4f}")
    print(f"{'type':7s} {'cells':>5s} | {'|dev| mean A/B':>16s} {'best cell |dev|':>15s} | {'signed mean A/B':>17s} | diff |dev| mean / best | diff signed mean / best cell")
    for t in RATE:
        v = per_type[t]
        print(f"{t:7s} {v['n_cells']:5d} | {v['dev_abs_mean']:7.4f}/{v['dev_abs_mean_b']:7.4f} {v['dev_abs_best_cell_mean']:6.3f}/{v['dev_abs_best_cell_mean_b']:6.3f} | "
              f"{v['dev_mean']:+8.5f}/{v['dev_mean_b']:+8.5f} | {v['diff_abs_mean']:+.5f} / {v['diff_abs_best_cell_mean']:+.4f} | {v['diff_signed_mean']:+.5f} / {v['diff_signed_best_cell']:+.4f}")
    if rf_stats:
        print("RF-windowed (per-body localizer): " + "; ".join(f"{t} {q} n {v['n_bodies_windowed']} max {v.get('rf_diff_max_over_cells', float('nan')):+.4f}"
                                                              for t, d in rf_stats.items() for q, v in d.items()))


# ================================================================================================ verify (CPU)
def verify_runs(paths: list) -> tuple[pd.DataFrame, list]:
    """Per run JSON (+ npz + console): the footprint recomputed from the captured radiance, the elevation band, the body
    coverage of every recorded type against the JSON's own population counts and EXPECTED_BODIES, and the console-vs-
    meta device check (docs/INTERP.md 10.4 rule 4)."""
    rows, problems = [], []
    for p in sorted(paths):
        p = Path(p)
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        prov = d.get("provenance", {}); ex = prov.get("execution", {}); st = prov.get("stimulus", {}); pr = st.get("params", {})
        npz = p.with_suffix(".npz"); txt = p.with_suffix(".txt")
        sm = d.get("summary", {})
        row = {"run": p.stem, "diam_deg": pr.get("diam_deg"), "elevation_deg": pr.get("elevation_deg"), "null": bool(sm.get("null")),
               "material": pr.get("material"), "light": pr.get("light"), "eye_height_m": pr.get("eye_height_m"),
               "device": ex.get("device"), "device_name": ex.get("device_name"), "retina_mode": prov.get("retina", {}).get("mode"), "console_device_cuda": None,
               "window_s": sm.get("window_s"), "wall_s_a": (sm.get("wall_s") or {}).get("a"), "wall_s_b": (sm.get("wall_s") or {}).get("b"),
               "commit": (prov.get("flyverse_commit") or {}).get("commit"), "W_md5": (prov.get("compiled_connectome") or {}).get("md5")}
        # the headline statistics of the run (one draw each; the ladder's compare() reads >= 5 of these per arm)
        for t in ("LC11", "LC10a", "LPLC2"):
            row[f"{t}_diff_max_mv"] = (sm.get("diff_max_over_cells_mean_mv") or {}).get(t)
            row[f"{t}_diff_rate_max_hz"] = (sm.get("diff_rate_hz_max_cell") or {}).get(t)
        for t in ("T3", "T2", "Tm5Y", "Mi1"):
            row[f"{t}_diff_signed_best"] = (sm.get("diff_signed_best_cell") or {}).get(t)
            row[f"{t}_diff_abs_best"] = (sm.get("diff_abs_best_cell_mean") or {}).get(t)
        if txt.exists():
            t = txt.read_text(encoding="utf-8", errors="replace")
            row["console_device_cuda"] = ("device cuda" in t) and ("device cpu" not in t)
        if not str(ex.get("device", "")).startswith("cuda"):
            problems.append(f"{p.stem}: realised device {ex.get('device')} (resubmit)")
        if row["console_device_cuda"] is False:
            problems.append(f"{p.stem}: console does not say device cuda")
        if row["retina_mode"] != "in_loop_capture":
            problems.append(f"{p.stem}: retina.mode {row['retina_mode']!r}")
        pb = pd.DataFrame(d.get("tables", {}).get("per_body", []))
        pops = {q["label"]: q["n_cells"] for q in d.get("populations", [])}
        for t, n_exp in EXPECTED_BODIES.items():
            n_rows = int(((pb.type == t) & (pb.quantity == "drive_mv")).sum()) if len(pb) else 0
            row[f"bodies_{t}"] = n_rows
            if n_rows != n_exp or pops.get(t) != n_exp:
                problems.append(f"{p.stem}: {t} bodies per_body {n_rows} / population {pops.get(t)} != expected {n_exp}")
        for t in SPIKING + RATE:
            q = "drive_mv" if t in SPIKING else "optic_dr"
            n_rows = int(((pb.type == t) & (pb.quantity == q)).sum()) if len(pb) else 0
            if n_rows != pops.get(t):
                problems.append(f"{p.stem}: {t} per_body rows {n_rows} != population {pops.get(t)}")
        if npz.exists():
            z = np.load(npz)
            fp = footprint(z["rad_a"], z["rad_b"], z["col_az_el"])["summary"]
            row.update({k: fp[k] for k in ("n_dimmed_50pct_mean", "n_changed_5pct_mean", "centroid_el_mean_deg", "centroid_el_sd_deg", "centroid_az_min_deg",
                                           "centroid_az_max_deg", "blank_lum_under_object_mean", "blank_lum_under_object_cv", "blank_static_max_abs_diff")})
            row["dimmed_el_band"] = fp["dimmed_el_band_deg"]
            row["seen_el_maxdev"] = float(np.nanmax(np.abs(z["seen_el_deg"] - pr.get("elevation_deg", 0)))) if np.isfinite(z["seen_el_deg"]).any() else None
            row["seen_diam_maxdev"] = float(np.nanmax(np.abs(z["seen_diam_deg"] - pr.get("diam_deg", 0)))) if np.isfinite(z["seen_diam_deg"]).any() else None
            js = d.get("summary", {}).get("footprint", {})
            if js and abs(js.get("n_dimmed_50pct_mean", np.nan) - fp["n_dimmed_50pct_mean"]) > 1e-9:
                problems.append(f"{p.stem}: JSON footprint {js.get('n_dimmed_50pct_mean')} != recomputed {fp['n_dimmed_50pct_mean']}")
        else:
            problems.append(f"{p.stem}: no npz")
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        obj = df[~df.null.astype(bool)]
        if len(obj) > 1 and obj.centroid_el_mean_deg.notna().any():
            spread = float(obj.centroid_el_mean_deg.max() - obj.centroid_el_mean_deg.min())
            df.attrs["el_band_spread_deg"] = spread
            df.attrs["el_band_verdict"] = "same band" if spread < 4.6 else "DIFFERENT bands (> one interommatidial angle)"
    return df, problems


def cmd_verify(args) -> int:
    paths = []
    for g in args.paths:
        paths += glob.glob(str(Path(g) / "*.json")) if Path(g).is_dir() else glob.glob(g)
    paths = [p for p in paths if not p.endswith("verify.json") and not p.endswith("_prov.json")]
    df, problems = verify_runs(paths)
    cols = [c for c in ("run", "diam_deg", "elevation_deg", "null", "device", "console_device_cuda", "retina_mode", "bodies_LC11", "bodies_LC10a",
                        "n_dimmed_50pct_mean", "n_changed_5pct_mean", "centroid_el_mean_deg", "centroid_el_sd_deg", "centroid_az_min_deg", "centroid_az_max_deg",
                        "dimmed_el_band", "blank_lum_under_object_mean", "blank_lum_under_object_cv", "seen_el_maxdev", "seen_diam_maxdev") if c in df.columns]
    common.print_table(df[cols], floatfmt="{:.3f}")
    cols2 = [c for c in ("run", "material", "light", "device_name", "wall_s_a", "wall_s_b", "LC11_diff_max_mv", "LC10a_diff_max_mv", "LPLC2_diff_max_mv",
                         "LC11_diff_rate_max_hz", "LC10a_diff_rate_max_hz", "T3_diff_signed_best", "T3_diff_abs_best", "Mi1_diff_signed_best", "Mi1_diff_abs_best") if c in df.columns]
    print("per-run headline statistics (ONE draw each; not a comparison):")
    common.print_table(df[cols2], floatfmt="{:+.4f}")
    spread = df.attrs.get("el_band_spread_deg")
    if spread is not None:
        print(f"footprint centroid elevation across object runs: spread {spread:.3f} deg -> {df.attrs['el_band_verdict']}")
    print("problems: " + ("; ".join(problems) if problems else "none"))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(common.to_jsonable({"generator": " ".join(sys.argv), "runs": df.to_dict("records"), "el_band_spread_deg": spread,
                                          "el_band_verdict": df.attrs.get("el_band_verdict"), "problems": problems, "expected_bodies": EXPECTED_BODIES}), f, indent=1)
        print(f"written {args.json}")
    return 1 if problems else 0


def cmd_geometry(args) -> int:
    rows = []
    for deg in args.diam_deg:
        g = geometry_record(deg, args.distance_m, args.elevation_deg, args.az_max, args.deg_per_s, args.eye_height_m)
        s = g["substrate"]
        rows.append({"diam_deg": deg, "radius_mm": g["radius_m"] * 1000, "check_deg": g["angular_diameter_check_deg"], "lower_limb_deg": s["lower_limb_deg"],
                     "substrate_horizon_deg": s["substrate_horizon_deg"], "clear": s["clear"], "one_way_s": g["one_way_s"]})
    common.print_table(pd.DataFrame(rows), floatfmt="{:.3f}")
    return 0


# ================================================================================================ batch (the cluster generator)
def _job(dir_: str, stem: str, extra: str, seconds: float, settle: float, seed: int) -> str:
    """One cluster job line (docs/INTERP.md 10.4 rules 3-4: mkdir -p in every line, the venv, the CUDA assert, the console
    into the run's .txt beside its JSON / npz, the tail echoed into the job log)."""
    return (f"mkdir -p {dir_} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/probe_object_matched.py run --diam-deg {extra['diam']} --seconds {seconds:g} --settle {settle:g} --seed {seed}"
            f"{' ' + extra['flags'] if extra['flags'] else ''} --out {dir_}/{stem} > {dir_}/{stem}.txt 2>&1; tail -4 {dir_}/{stem}.txt")


def batch_jobs(kind: str, dir_: str, seconds: float, settle: float, seeds: list, diams: list) -> list:
    """The job lines of a named batch. `smoke`: one object run per diameter + one blank/blank null at 11 deg + the
    bright (lamp) and ceiling-light variants at 11 deg, one seed (the validation batch of the audit). `ladder`: every
    diameter x every seed (object) + every seed (null at the first diameter) -- the >= 5-runs-per-arm submission."""
    jobs = []
    if kind == "smoke":
        s = seeds[0]
        for d in diams:
            jobs.append(_job(dir_, f"d{int(round(d * 10)):03d}_obj_s{s}", {"diam": d, "flags": ""}, seconds, settle, s))
        jobs.append(_job(dir_, f"d110_null_s{s}", {"diam": 11, "flags": "--null"}, seconds, settle, s))
        jobs.append(_job(dir_, f"d110_lamp_s{s}", {"diam": 11, "flags": "--material lamp"}, seconds, settle, s))
        jobs.append(_job(dir_, f"d110_ceil_s{s}", {"diam": 11, "flags": "--light ceiling"}, seconds, settle, s))
    elif kind == "ladder":
        for d in diams:
            for s in seeds:
                jobs.append(_job(dir_, f"d{int(round(d * 10)):03d}_obj_s{s}", {"diam": d, "flags": ""}, seconds, settle, s))
        for s in seeds:
            jobs.append(_job(dir_, f"d{int(round(diams[0] * 10)):03d}_null_s{s}", {"diam": diams[0], "flags": "--null"}, seconds, settle, s))
    else:
        raise ValueError(kind)
    return jobs


def cmd_batch(args) -> int:
    import shlex
    import subprocess
    dir_ = args.dir.rstrip("/")
    jobs = batch_jobs(args.kind, dir_, args.seconds, args.settle, args.seeds, args.diam_deg)
    cmd = [sys.executable, "scripts/cluster_run.py", "--name", args.name, "--minutes", str(args.minutes), *jobs, "--fetch", dir_ + "/"]
    log = Path("out") / f"{args.name.replace('-', '_')}_cluster.log"
    print(f"{len(jobs)} job(s) -> {dir_}/ (console log {log}):")
    for j in jobs:
        print("  " + j)
    print("cluster_run: " + " ".join(shlex.quote(c) for c in cmd))
    if not args.submit:
        return 0
    if len(jobs) > 20:
        raise SystemExit(f"{len(jobs)} jobs > 20: split the batch (the rented boxes are few)")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")      # stream the child's console into the log as it happens
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=env)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line)
        rc = p.wait()
    print(f"cluster_run exit {rc}; log {log}")
    return rc


def add_geometry_args(ap, multi: bool = False) -> None:
    if multi:
        ap.add_argument("--diam-deg", type=float, nargs="+", default=DIAMETERS_DEG, help="angular diameters from the eye (deg)")
    else:
        ap.add_argument("--diam-deg", type=float, default=11.0, help="angular diameter from the eye (deg); the radius is derived from --distance-m")
    ap.add_argument("--distance-m", type=float, default=0.05, help="eye -> ball centre distance (m), constant along the arc")
    ap.add_argument("--elevation-deg", type=float, default=0.0, help="centre elevation in the fly's frame (deg), constant and radius-independent")
    ap.add_argument("--az-max", type=float, default=50.0, help="the arc spans +-az_max deg of azimuth (+ = left)")
    ap.add_argument("--deg-per-s", type=float, default=40.0, help="constant angular speed along the arc")
    ap.add_argument("--eye-height-m", type=float, default=0.15, help="eye above the body's surface (m); 0.0012 is the walking fly (then a ball at elevation 0 is clipped by the table)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one run = arm A (object, or blank under --null) then arm B (blank), same seed (GPU)")
    add_geometry_args(r)
    r.add_argument("--material", default="black", choices=MATERIALS, help="black = dark object; plate = bright reflective; lamp = bright emissive (uniform)")
    r.add_argument("--light", default="eye", choices=["eye", "ceiling"], help="eye = point light at the eye (no visible shadows); ceiling = the room's lamp")
    r.add_argument("--allow-substrate-clip", action="store_true", help="record even when the table clips the ball's lower limb")
    r.add_argument("--seconds", type=float, default=12.0, help="scored window (s)")
    r.add_argument("--settle", type=float, default=3.0, help="settling time before the window, ball parked (s)")
    r.add_argument("--null", action="store_true", help="blank vs blank: arm A is a second blank run under the same seed")
    r.add_argument("--rf-map", default=None, help="per-body RF localizer (CSV bodyId,az_deg,el_deg,width_deg[,height_deg] or a Result JSON with tables.rf_map)")
    r.add_argument("--save-recordings", action="store_true", help="also write the flyverse.interp.recording npz + json per arm and cell set")
    r.add_argument("--allow-cpu", action="store_true")
    r.add_argument("--out", required=True, help="output stem: <out>.json (Result) + <out>.npz (per-frame arrays)")
    common.add_common_args(r)
    r.set_defaults(func=cmd_run)
    v = sub.add_parser("verify", help="CPU: footprint from the captured radiance, elevation band across diameters, body coverage, device checks")
    v.add_argument("paths", nargs="+", help="run JSONs, globs, or directories")
    v.add_argument("--json", default=None)
    v.set_defaults(func=cmd_verify)
    g = sub.add_parser("geometry", help="CPU: the geometry table for a ladder (radius, limbs, substrate clearance)")
    add_geometry_args(g, multi=True)
    g.set_defaults(func=cmd_geometry)
    b = sub.add_parser("batch", help="print (or --submit through scripts/cluster_run.py) the job lines of the smoke / ladder batch")
    b.add_argument("--kind", choices=["smoke", "ladder"], default="smoke")
    b.add_argument("--dir", default="out/objm/smoke", help="the NAMED fetch directory (never a bare out/)")
    b.add_argument("--name", default="objm-smoke", help="cluster job name; the console log goes to out/<name>_cluster.log")
    b.add_argument("--minutes", type=int, default=30)
    b.add_argument("--seconds", type=float, default=3.0); b.add_argument("--settle", type=float, default=3.0)
    b.add_argument("--seeds", type=int, nargs="+", default=[0], help="one job per seed per arm (runs are the replicate unit; the seed does not pin a GPU draw)")
    b.add_argument("--diam-deg", type=float, nargs="+", default=DIAMETERS_DEG)
    b.add_argument("--submit", action="store_true", help="run cluster_run.py (the cluster rule: every GPU workload goes through it)")
    b.set_defaults(func=cmd_batch)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
