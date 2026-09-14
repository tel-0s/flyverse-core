"""Object round 3, task run:localizer -- a stimulus-driven LC11 / LC10a receptive-field localizer that can fit these cells.

Round 2's localizer (scripts/probe_synthetic_stimuli.py: a 15-deg dark square, 200 ms per node, a 10-deg grid over the
whole eye, three passes in ONE run, z_min 5 on the stimulus arm's own MAD) fitted 0 of 143 LC11 bodies and 13 of 275
LC10a bodies, so 405 of 418 LC windows of the matched sphere ladder were anatomical boxes (docs/audits/object_baseline_r2.md
0b, object_synthetic_stimuli.md 4 / 8, docs/INTERP.md 2.7). This script is the round-3 design:

* a SMALL, MAXIMUM-CONTRAST probe: a 4.5-deg dark square at Weber contrast -0.995. 4.5 deg is the smallest square that
  contains a column's whole 7-ray acceptance kernel (`Retina.ray_directions`, acceptance 4.5 deg): on this retina a
  2 / 3 / 4-deg square dims the median column by 12 % / 43 % / 54 % at its best grid node (maxima 43 % / 54 % / 89 %;
  the grid is 4 deg and `preview` prints these), so anything smaller than 4.5 deg is a LOWER-contrast probe, not a
  finer one;
* a LONGER dwell: 0.5 s on, 0.5 s of background between nodes (`base` = the last 0.2 s of that blank), one
  presentation per node per run;
* a FINE grid (4-deg spacing, sub-column against the 4.6-deg column pitch) over the anatomical boxes of the LC cells --
  the nodes within `--grid-radius` (20 deg) of any LC11 / LC10a cell's INPUT-WEIGHTED anatomical centroid -- not the
  whole eye;
* >= 4 runs POOLED per node (5 by default), each run its own brain seed AND its own node order, on the shipped lobe
  and on the `gain_fb = 0` deterministic lobe, both lobes in ONE submission under one `--arm-block` (`fam_locr3`);
* the fit on the per-body RECEIVED DRIVE (mV) of the spiking LC cells (`drive_mv`; `optic_dr` for the rate units),
  never on spikes (the LC populations emit essentially none in this protocol: object_baseline_r2.md 0b);
* the false-fit rate measured by the SAME rule on the pooled BLANK arm (the same node schedule, nothing presented) at
  every threshold of a z ladder, and the threshold chosen per type from the blank arm (predeclared rule below);
* the ANATOMICAL COLUMN SET as the prior: per cell the |W|-weighted set of columns of its rate inputs
  (`trace.column_of_cells` over the optic-lobe rate units), its weighted centroid, its 50 % / 80 % weight radii, and
  the round-2 single column beside it. The fit is done inside a `--box-radius` (20 deg) disc around the centroid; the
  peak over the WHOLE grid ('free peak') is the criterion-free check that the prior did not manufacture the fit.

The anatomical prior, measured (the reason the round-2 boxes could not stand in for receptive fields): `column_of_cells`
assigns a spiking cell the column of its strongest rate input, and the 143 LC11 cells land on 13 distinct columns (50
on one, 44 on another); the 275 LC10a cells on 88. The input-weighted column set of an LC11 cell has a median 100
distinct columns, half its input weight within 26 deg of the centroid (LC10a: 48 columns, 45 deg), and the round-2
single column sits a median 65 deg (LC11) / 47 deg (LC10a) from that centroid. Every number is written to the map.

    PYTHONIOENCODING=utf-8 python scripts/object_round3_localizer.py preview                       # CPU: the grid, the probe's coverage, the prior
    PYTHONIOENCODING=utf-8 python scripts/object_round3_localizer.py plan --out out/objr3rf         # batch.sh, jobs.json, predeclared.json (stamped), tree_state.json
    PYTHONIOENCODING=utf-8 python scripts/object_round3_localizer.py submit --out out/objr3rf       # ONE cluster_run.py call, --arm-block fam, fetch out/objr3rf/loc/
    PYTHONIOENCODING=utf-8 python scripts/object_round3_localizer.py verify --out out/objr3rf       # CPU: consoles, devices, checksums, grids, the log's job(s) line
    PYTHONIOENCODING=utf-8 python scripts/object_round3_localizer.py rfmap --out out/objr3rf --lobe ship   # CPU: the pooled map (docs/INTERP.md 2.7 format)
    PYTHONIOENCODING=utf-8 python scripts/object_round3_localizer.py selfcheck                     # CPU: the pooled fit on synthetic bumps
    # CPU smoke of a run (Windows: CUDA_VISIBLE_DEVICES=-1):
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python scripts/object_round3_localizer.py record --lobe ship --run 0 --allow-cpu \
        --grid-spacing 40 --grid-radius 40 --flash-s 0.05 --blank-s 0.05 --settle 0.1 --out out/objr3rf_smoke/loc/loc45_ship_r0

Nothing in flyverse/ is edited; the model is read through FlyBrain with the common CLI's overrides (`--lobe fb0` =
`--optic gain_fb=0`). The stimulus path, the recorder and the RF-map format are scripts/probe_synthetic_stimuli.py's
(imported, not modified). Every JSON carries provenance; the Result records the sha256 of this script and of the
probe module it imports (docs/INTERP.md 10.4 item 11).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from flyverse.interp import common                       # noqa: E402
from flyverse.interp.common import Result, to_jsonable   # noqa: E402
import probe_synthetic_stimuli as pss                     # noqa: E402

# ---------------------------------------------------------------------------------------------------- constants
PROBE_DEG = 4.5                 # the smallest square that contains a column's whole 7-ray kernel (acceptance 4.5 deg)
CONTRAST = pss.DARK_CONTRAST    # -0.995: the black ball's Weber contrast (the darkest object the room presents)
SPACING_DEG = 4.0               # grid spacing: sub-column against the retina's 4.6-deg pitch
GRID_RADIUS_DEG = 20.0          # nodes within this of any LC11 / LC10a input-weighted centroid are presented
BOX_RADIUS_DEG = 20.0           # the per-cell fit uses the nodes within this of the cell's input-weighted centroid
FLASH_S, BLANK_S = 0.5, 0.5     # dwell per node and the background between nodes
BASE_FRAMES, ON1_FRAMES, OFF_FRAMES = 20, 10, 10   # base = last 0.2 s of the blank; on1 = first 0.1 s of the flash; off = first 0.1 s after
RUNS = 5                        # runs per lobe, pooled per node (>= 4 asked; 5 = one spare)
LOBES = {"ship": [], "fb0": ["gain_fb=0"]}
Z_LADDER = (3.0, 4.0, 5.0, 6.0, 8.0)
FALSE_FIT_MAX = 0.01            # the threshold per type is the smallest z of the ladder whose blank-arm false-fit rate is <= this
MIN_BOX_NODES = 12              # a cell whose box holds fewer grid nodes is not fitted (NaN, never 0)
TAG = "fam_locr3"               # the --arm-block key/value every job command carries (one block = one box)
SELECTION = list(pss.SELECTION)  # LC11, LC10a, T2, T3, Tm5Y, TmY21, Mi1 (Mi1 = the localizer's ground truth)
LC_TYPES = ("LC11", "LC10a")
QUANTITIES = ("drive_mv", "spike_count", "optic_dr")
ROLES = ("on", "on1", "off", "base")
RETINA_MODE = pss.RETINA_MODE
STIM_NAME = "localizer_r3"

PREDECLARED = {
    "question": "Can a small, maximum-contrast dark probe with a long dwell, a fine grid over the anatomical boxes and >= 4 pooled runs "
                "fit a stimulus-driven receptive-field centre for the LC11 and LC10a bodies of this model, on the shipped lobe and on the "
                "gain_fb = 0 lobe -- and if not for LC11, at what measured false-fit rate is that the answer.",
    "stimulus": {"probe": f"{PROBE_DEG} deg dark square, Weber contrast {CONTRAST} (the smallest square containing a column's whole 7-ray kernel)",
                 "dwell_s": FLASH_S, "blank_between_s": BLANK_S, "grid_spacing_deg": SPACING_DEG,
                 "grid": f"nodes within {GRID_RADIUS_DEG} deg of any LC11 / LC10a input-weighted anatomical centroid (not the whole eye)",
                 "presentations_per_node_per_run": 1, "order": "a fixed shuffled order per run (order seed = run index), so the pooled "
                 "per-node response samples the lobe's autonomous oscillation at different phases in different runs"},
    "replicate_unit": "one process = one (stimulus, blank) pair under one brain seed and one node order = one run; the map pools the "
                      "per-node responses over the runs of one lobe BEFORE the fit (the per-run fits are on file as n_runs_fitted and "
                      "centre_spread_deg; the split-half pooled fits as split_half_centre_distance_deg)",
    "runs_per_lobe": RUNS, "lobes": {"ship": "OpticParams defaults", "fb0": "gain_fb=0 (deterministic lobe)"},
    "one_submission": f"both lobes x {RUNS} runs in ONE cluster_run.py call, --arm-block fam (every job carries {TAG}) so both lobes sit on one box",
    "quantity": "per-body received drive (drive_mv, the LIF's input in mV) for the spiking LC cells; optic_dr for the rate units; "
                "spikes are reported (spikes_on_minus_base_hz_peak) and never fitted",
    "windows": {"on": "the 50 flash frames minus base (PRIMARY)", "on1": "the first 10 flash frames minus base (secondary)",
                "off": "the 10 frames after the offset minus base (secondary)", "base": "the last 20 blank frames before the onset"},
    "prior": {"column_set": "per cell, the |W|-weighted columns of its rate inputs (trace.column_of_cells over the optic-lobe rate units): "
                            "weighted centroid (anat_input_az_deg / _el_deg), 50 % and 80 % weight radii, the number of distinct columns; "
                            "the round-2 single column (anat_az_deg / anat_el_deg, the strongest input's column) is kept beside it",
              "box": f"the fit uses the grid nodes within {BOX_RADIUS_DEG} deg of the input-weighted centroid (>= {MIN_BOX_NODES} nodes, else NaN)",
              "criterion_free_check": "free_peak_in_box: the argmax |R| over the WHOLE grid falls inside the box, against chance_in_box = "
                                      "the fraction of grid nodes inside the box"},
    "fit_rule": "per cell, on the run-pooled per-node response R restricted to the box: peak = argmax |R|, noise = 1.4826 x MAD over the box "
                "nodes, FITTED when |peak| >= z_min x noise; centre = the (R - half-max)-weighted centroid of the nodes at or above half-maximum "
                "of the peak's sign, width = the equivalent-disc FWHM 2 sqrt(n spacing^2 / pi) (at least one spacing) -- probe_synthetic_stimuli.fit_rf "
                "applied to the box nodes with spacing 4 deg",
    "false_fit": "the same rule on the run-pooled BLANK arm (the same node schedule, the background presented) per type at every z of the ladder",
    "z_ladder": list(Z_LADDER),
    "threshold_rule": f"per type and lobe, z* = the smallest z of the ladder at which the pooled blank-arm false-fit rate is <= {FALSE_FIT_MAX}; "
                      "if none, z* = the largest z and the type is reported 'no threshold reaches the false-fit bound'; z_min 5 (round 2's) "
                      "is reported beside it for continuity",
    "answer": "per type and lobe: the fraction of bodies fitted at z*, stated WITH the blank-arm false-fit rate at z*; the distribution of "
              "fitted centres and widths; the agreement with the anatomical prior (median great-circle distance of the fitted centre to the "
              "input-weighted centroid and to the round-2 single column, in deg); the free-peak-in-box fraction against chance",
    "localizable_call": "a type is LOCALIZED on a lobe when (a) its coverage at z* exceeds the blank-arm false-fit rate at z* by >= 0.05 "
                        "(5 points of the population) AND (b) free_peak_in_box >= 2 x chance_in_box; otherwise 'not localized by this stimulus'. "
                        "If LC11 is not localized on either lobe the finding is stated as such: the cell has no stimulus-driven receptive "
                        "field to a 4.5-deg dark square in this model at the measured false-fit rate -- never 'silent' (the cells fire) and "
                        "never 'inverted'",
    "no_verdict_vocabulary": "a map is magnitudes: no compare() verdict, no Holm; the only inferential quantities are the false-fit rate "
                             "(measured) and the chance level of the free-peak check (computed)",
    "controls": "Mi1 (hex-annotated: the ground truth for a column-sized RF), T2 / T3 (LC11's inputs) and Tm5Y / TmY21 (LC10a's) are "
                "recorded and fitted by the same rule so 'the inputs are retinotopic, the output is not' is measured, not assumed",
    "consumers": "the CSV / JSON follow docs/INTERP.md 2.7 (flyverse.interp.rfmap/1): the first seven columns are the interface; fitted "
                 "rows are the ones with a finite az_deg; anat_az_deg / anat_el_deg keep round 2's single-column meaning so the ladder "
                 "window rule's fallback is unchanged, and the input-weighted centroid is the new anat_input_* pair",
    "not_matched": "the probe's effective contrast at a column depends on the node-to-column offset (coverage 0.54-0.89 at the best node "
                   "of a 4-deg grid, median 0.66); the map's width is bounded below by one grid spacing (4 deg) and above by the box",
}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def analysis_code() -> dict:
    """sha256 of the analysis code (this script and the probe module it imports): docs/INTERP.md 10.4 item 11."""
    return {"scripts/object_round3_localizer.py": sha256_file(__file__), "scripts/probe_synthetic_stimuli.py": sha256_file(pss.__file__),
            "flyverse/interp/common.py": sha256_file(common.__file__)}


# ---------------------------------------------------------------------------------------------------- anatomy
def load_model(cache_dir=None):
    from flyverse import connectome, retina as retina_mod
    c = connectome.load(cache_dir=Path(cache_dir), verbose=False) if cache_dir else connectome.load(verbose=False)
    return c, retina_mod.build_retina(c)


def unit_vectors(az_deg, el_deg) -> np.ndarray:
    a, e = np.deg2rad(np.asarray(az_deg, float)), np.deg2rad(np.asarray(el_deg, float))
    return np.c_[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]


def anatomical_prior(c, retina, idx) -> pd.DataFrame:
    """Per cell (model index): the round-2 single column (`trace.column_of_cells`: anat_column / anat_az_deg / anat_el_deg /
    hex_annotated) AND the input-weighted column set -- the |W|-weighted mean direction of the columns of the cell's rate
    inputs (anat_input_az_deg / _el_deg), the weighted RMS and the 50 % / 80 % weight radii (deg), the number of distinct
    input columns and the fraction of the cell's input weight that carries a column. NaN when no input has a column."""
    import scipy.sparse as sp
    from flyverse.connectome import PHOTORECEPTOR_TYPES
    from flyverse.interp import trace as tr
    nrn = c.neurons
    rate_idx = np.flatnonzero((nrn.superclass == "ol_intrinsic").to_numpy() & ~nrn.type.isin(PHOTORECEPTOR_TYPES).to_numpy())
    col, _ = tr.column_of_cells(c, retina, rate_idx)
    cae = np.asarray(retina.col_az_el); U = unit_vectors(cae[:, 0], cae[:, 1])
    W = sp.csr_matrix(abs(c.W))                                   # W[post, pre] (column_of_cells' orientation)
    hex_ann = nrn.hex1.notna().to_numpy()
    rows = []
    for i in np.asarray(idx):
        cc = int(col[i])
        row = {"anat_column": cc, "anat_az_deg": float(cae[cc, 0]) if cc >= 0 else np.nan, "anat_el_deg": float(cae[cc, 1]) if cc >= 0 else np.nan,
               "hex_annotated": bool(hex_ann[i]), "anat_input_az_deg": np.nan, "anat_input_el_deg": np.nan, "anat_input_rms_deg": np.nan,
               "anat_input_r50_deg": np.nan, "anat_input_r80_deg": np.nan, "n_input_columns": 0, "input_weight_with_column": 0.0}
        s0, s1 = W.indptr[i], W.indptr[i + 1]; pre = W.indices[s0:s1]; w = W.data[s0:s1]
        pc = col[pre]; ok = pc >= 0
        if ok.any() and w.sum() > 0:
            ww, cols = w[ok], pc[ok]
            v = (ww[:, None] * U[cols]).sum(0); v /= max(np.linalg.norm(v), 1e-12)
            az = float(np.degrees(np.arctan2(v[1], v[0]))); el = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
            d = pss.angular_distance_deg(cae[cols, 0], cae[cols, 1], az, el)
            o = np.argsort(d); cw = np.cumsum(ww[o]) / ww.sum()
            row.update({"anat_input_az_deg": az, "anat_input_el_deg": el, "anat_input_rms_deg": float(np.sqrt((ww * d ** 2).sum() / ww.sum())),
                        "anat_input_r50_deg": float(d[o][min(np.searchsorted(cw, 0.5), len(d) - 1)]),
                        "anat_input_r80_deg": float(d[o][min(np.searchsorted(cw, 0.8), len(d) - 1)]),
                        "n_input_columns": int(len(np.unique(cols))), "input_weight_with_column": float(ww.sum() / w.sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def lc_centroids(c, retina, types=LC_TYPES) -> pd.DataFrame:
    """The input-weighted centroids of the LC cells (the centres the grid is built around)."""
    idx = np.flatnonzero(c.neurons.type.isin(list(types)).to_numpy())
    pr = anatomical_prior(c, retina, idx)
    pr.insert(0, "type", c.neurons.type.to_numpy()[idx]); pr.insert(0, "model_index", idx)
    return pr


# ---------------------------------------------------------------------------------------------------- the grid and the stimulus
def grid_nodes(col_az_el, centres_az, centres_el, spacing_deg=SPACING_DEG, radius_deg=GRID_RADIUS_DEG, size_deg=PROBE_DEG,
               az_range=(-120.0, 120.0), el_range=(-70.0, 70.0), acceptance_deg=pss.ACCEPTANCE_DEG) -> np.ndarray:
    """(n_nodes, 2) azimuth / elevation of the grid nodes that (a) reach at least one column (a column centre within
    size / 2 + acceptance / 2, as probe_synthetic_stimuli.localizer_grid) and (b) lie within `radius_deg` (great circle)
    of at least one centre. Row-major over elevation then azimuth, so the node index is a stable function of the grid."""
    azs = np.arange(az_range[0], az_range[1] + 1e-9, spacing_deg); els = np.arange(el_range[0], el_range[1] + 1e-9, spacing_deg)
    A, E = np.meshgrid(azs, els); A, E = A.ravel(), E.ravel()
    c = np.asarray(col_az_el)
    reach = size_deg / 2.0 + acceptance_deg / 2.0
    ok_reach = np.array([np.hypot(c[:, 0] - a, c[:, 1] - e).min() <= reach for a, e in zip(A, E)])
    ca, ce = np.asarray(centres_az, float), np.asarray(centres_el, float)
    fin = np.isfinite(ca) & np.isfinite(ce)
    near = np.zeros(len(A), bool)
    for a0, e0 in zip(ca[fin], ce[fin]):
        near |= pss.angular_distance_deg(A, E, a0, e0) <= radius_deg
    keep = ok_reach & near
    return np.c_[A[keep], E[keep]].astype(np.float64)


def localizer_boxed(col_az_el, nodes: np.ndarray, size_deg=PROBE_DEG, contrast=CONTRAST, flash_s=FLASH_S, blank_s=BLANK_S, order_seed=0,
                    background=pss.BACKGROUND, acceptance_deg=pss.ACCEPTANCE_DEG, rays=pss.RAYS, dt_s=pss.FRAME_S, extra_params=None) -> pss.Stimulus:
    """The round-3 localizer: a `size_deg` square of `contrast` shown for `flash_s` at every node of `nodes`, `blank_s` of
    background before each (and after the last), in the shuffled order of `order_seed` (a different order per run),
    ONE presentation per node. Stored as pattern (blank + one frame per node) and a per-frame index; `track['node']` is
    the node presented (-1 = blank), `track['on']` the flash frames -- probe_synthetic_stimuli.localizer's layout."""
    az, el, w = pss.sample_points(col_az_el, acceptance_deg, rays)
    blank = pss.blank_frame(len(col_az_el), background)
    pattern = np.empty((len(nodes) + 1, len(col_az_el), 4), np.float32); pattern[0] = blank
    cov = np.zeros(len(nodes))
    for i, (a, e) in enumerate(nodes):
        cv = pss.rect_coverage(az, el, w, a, e, size_deg, size_deg); cov[i] = cv.sum()
        pattern[i + 1] = pss.radiance_from_coverage(cv, contrast, background)
    order = np.random.RandomState(int(order_seed)).permutation(len(nodes))
    n_on, n_off = int(round(flash_s / dt_s)), int(round(blank_s / dt_s))
    index, node, on = [], [], []
    for i in order:
        index += [0] * n_off + [i + 1] * n_on; node += [-1] * n_off + [int(i)] * n_on; on += [False] * n_off + [True] * n_on
    index += [0] * n_off; node += [-1] * n_off; on += [False] * n_off
    n = len(index)
    params = {"spacing_deg": None, "size_deg": size_deg, "contrast": contrast, "flash_s": flash_s, "blank_s": blank_s, "order_seed": int(order_seed),
              "n_nodes": int(len(nodes)), "passes": 1, "seconds": n * dt_s, "background": list(background), "acceptance_deg": acceptance_deg, "rays": rays,
              "column_equivalents_per_node_median": float(np.median(cov)) if len(cov) else np.nan, "column_equivalents_per_node_min": float(cov.min()) if len(cov) else np.nan}
    params.update(extra_params or {})
    return pss.Stimulus(STIM_NAME, params, blank, dt_s, pattern=pattern, index=np.asarray(index, np.int64),
                        track={"t_s": np.arange(n) * dt_s, "node": np.asarray(node, np.int64), "on": np.asarray(on, bool),
                               "node_az_deg": nodes[:, 0], "node_el_deg": nodes[:, 1], "order": order.astype(np.int64), "node_column_equivalents": cov})


# ---------------------------------------------------------------------------------------------------- recording
class BoxAccumulator:
    """Online per-node reductions, one array per role: `on` (every flash frame), `on1` (the first ON1_FRAMES flash frames),
    `off` (the first OFF_FRAMES frames after the offset), `base` (the last BASE_FRAMES frames of the blank before the
    onset), of ONE quantity per cell -- drive_mv for a spiking cell, optic_dr for a rate unit (decided on the first
    frame from which cells carry a finite optic_dr) -- plus spikes per frame for the spiking cells. (n_roles, n_nodes,
    n_cells) float64 sums, written as float32 means; nothing per frame per cell is kept."""

    def __init__(self, stim: pss.Stimulus, n_cells: int, base_frames=BASE_FRAMES, on1_frames=ON1_FRAMES, off_frames=OFF_FRAMES):
        node, on = stim.track["node"], stim.track["on"]
        n_nodes = int(stim.params["n_nodes"]); T = len(node)
        self.role = [np.full(T, -1, np.int64) for _ in ROLES]           # per role: the node the frame belongs to, -1 = none
        onset = np.flatnonzero(on & ~np.r_[False, on[:-1]]); offset = np.flatnonzero(~on & np.r_[False, on[:-1]])
        for s in onset:
            e = s
            while e < T and on[e]:
                e += 1
            self.role[0][s:e] = node[s]; self.role[1][s:min(e, s + on1_frames)] = node[s]
            self.role[3][max(0, s - base_frames):s] = node[s]
        for s in offset:
            self.role[2][s:s + off_frames] = node[s - 1]
        self.sum = np.zeros((len(ROLES), n_nodes, n_cells), np.float64)
        self.cnt = np.zeros((len(ROLES), n_nodes), np.int64)
        self.is_rate = None; self.spk_sum = None; self.spk_idx = None; self._prev_sc = None
        self.n_cells = n_cells

    def add(self, t: int, values: dict) -> None:
        if self.is_rate is None:
            self.is_rate = np.isfinite(np.asarray(values["optic_dr"], np.float64))
            self.spk_idx = np.flatnonzero(~self.is_rate)
            self.spk_sum = np.zeros((len(ROLES), self.sum.shape[1], len(self.spk_idx)), np.float64)
        sc = np.asarray(values["spike_count"], np.float64)
        spk = sc - (self._prev_sc if self._prev_sc is not None else sc); self._prev_sc = sc
        x = np.where(self.is_rate, np.asarray(values["optic_dr"], np.float64), np.asarray(values["drive_mv"], np.float64))
        for r in range(len(ROLES)):
            nd = self.role[r][t]
            if nd >= 0:
                self.sum[r, nd] += x; self.spk_sum[r, nd] += spk[self.spk_idx]; self.cnt[r, nd] += 1

    def finish(self) -> dict:
        out = {f"n_{lab}": self.cnt[r] for r, lab in enumerate(ROLES)}
        out["quantity"] = np.where(self.is_rate, pss.RATE_Q, pss.SPIKING_Q).astype(str)
        out["spiking_cells"] = self.spk_idx
        for r, lab in enumerate(ROLES):
            den = np.maximum(self.cnt[r], 1)[:, None]
            out[f"resp__{lab}"] = (self.sum[r] / den).astype(np.float32)
            out[f"spk__{lab}"] = (self.spk_sum[r] / den).astype(np.float32)
        return out


def run_arm(fb, stim: pss.Stimulus, present_stimulus: bool, settle_s: float, selection, label: str, quiet: bool = False):
    """probe_synthetic_stimuli.run_arm with the BoxAccumulator: settle on the blank, present the stimulus (or the blank for
    the same frames), record the per-node reductions of every cell and the per-type population means per frame."""
    import torch
    rec = pss.GatherRecorder(fb.c, selection, quantities=QUANTITIES)
    n_settle = int(round(settle_s / stim.dt_s))
    blank_t = torch.as_tensor(np.ascontiguousarray(stim.blank), dtype=torch.float32, device=fb.device).clone()
    for _ in range(n_settle):
        fb.vision(blank_t); fb.step(common.FRAME_MS)
    acc = BoxAccumulator(stim, len(rec.idx))
    series = {q: [] for q in QUANTITIES}
    trec = common.Recording(np.zeros(0), rec.idx, rec.body_ids, rec.types).recorder()
    T = stim.n_frames; t0 = time.time()

    def on_frame(t):
        rec.capture(fb)
        vals = {q: rec._frames[q].pop() for q in QUANTITIES}; rec._t.pop()
        acc.add(t, vals)
        for q in QUANTITIES:
            series[q].append(trec.snapshot(vals[q]).astype(np.float32))
        if not quiet and (t % 2000 == 0 or t == T - 1):
            print(f"  [{label}] frame {t + 1}/{T} ({time.time() - t0:.0f} s wall; {(t + 1) / max(time.time() - t0, 1e-9):.1f} fps)", flush=True)

    checksum = pss.play(fb, stim, present_stimulus, on_frame)
    wall = time.time() - t0
    meta = {"arm": label, "settle_s": settle_s, "stimulus": stim.name, "presented": present_stimulus, "wall_s": wall, "fps": T / max(wall, 1e-9)}
    recording = common.Recording(np.arange(T) * stim.dt_s * 1000.0, rec.idx, rec.body_ids, rec.types,
                                 {f"pooled_{q}": np.stack(series[q]) for q in QUANTITIES}, {},
                                 dict(meta, pooled_keys=list(trec.keys), note="localizer_r3: per-type population means per frame; per-cell node reductions in <out>_nodes*.npz"))
    return recording, acc.finish(), checksum, wall


def cmd_record(args) -> int:
    from flyverse.interp import trace as tr
    if args.lobe not in LOBES:
        sys.exit(f"--lobe {args.lobe!r}: choose from {list(LOBES)}")
    args.optic = list(args.optic) + LOBES[args.lobe]
    seed = args.seed + args.run; order_seed = args.order_seed if args.order_seed is not None else args.run
    c, retina = load_model(args.cache_dir)
    lif, op = common.params_from_args(args)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    sel = [s for s in args.select.split(",") if s]
    fb = pss.build_fb(c, lif, op, seed, args.device)
    dev = fb.brain.device
    hook_info = getattr(fb.optic, "hook_info", None)
    print(f"FlyBrain ready: {c.n} neurons, {fb.retina.n_columns} columns, device {dev}, optic overrides {common.parse_kv(args.optic)}, "
          f"lif overrides {common.parse_kv(args.lif)}, receptor {lif.receptor_model}/{lif.receptor_net_rule}, hooks {hook_info}, "
          f"lobe {args.lobe}, run {args.run} (seed {seed}, order seed {order_seed}), tag {args.tag}", flush=True)
    if dev.type != "cuda" and not args.allow_cpu:
        sys.exit(f"device {dev}: not CUDA (node race; resubmit) -- pass --allow-cpu for a CPU smoke")
    col_az_el = np.asarray(fb.retina.col_az_el)
    cen = lc_centroids(c, retina)
    nodes = grid_nodes(col_az_el, cen["anat_input_az_deg"], cen["anat_input_el_deg"], args.grid_spacing, args.grid_radius, args.width)
    grid_sha = hashlib.sha256(np.ascontiguousarray(nodes).tobytes()).hexdigest()
    stim = localizer_boxed(col_az_el, nodes, args.width, args.contrast, args.flash_s, args.blank_s, order_seed,
                           extra_params={"grid_spacing_deg": args.grid_spacing, "grid_radius_deg": args.grid_radius, "grid_sha256": grid_sha,
                                         "grid_centres": f"input-weighted anatomical centroids of {len(cen)} LC11 / LC10a cells", "run": args.run, "lobe": args.lobe})
    print(f"stimulus {stim.name}: {stim.n_frames} frames ({stim.n_frames * stim.dt_s:.1f} s), {len(nodes)} nodes (grid sha256 {grid_sha[:12]}), "
          f"{json.dumps(to_jsonable({k: v for k, v in stim.params.items() if k != 'background'}))}", flush=True)
    arms = [("stim", True), ("blank", False)]
    recs, nodes_out, sums, walls = {}, {}, {}, {}
    for i, (label, present) in enumerate(arms):
        if i:
            fb = pss.build_fb(c, lif, op, seed, args.device)             # a fresh brain per arm, same seed (the probe's pattern)
        recs[label], nodes_out[label], sums[label], walls[label] = run_arm(fb, stim, present, args.settle, sel, label, args.quiet)
    retina_rec = dict(tr.retina_record(fb.retina, c), mode=RETINA_MODE, file=str(out) + "_radiance.npz",
                      sampling=f"synthetic per-column radiance handed to FlyBrain.vision every frame ({stim.dt_s * 1000:.0f} ms); no ray tracing; "
                               "the stored pattern + index IS the presented input of the stimulus arm (the blank arm presents `blank` every frame)",
                      pinned_pose=None, geometry="fly pinned, no body")
    prov = common.provenance(c, lif, op, fb=fb, device=args.device, seeds=[seed], batch=1,
                             stimulus={"protocol": f"synthetic:{stim.name}", "params": stim.params, "arms": [a for a, _ in arms], "settle_s": args.settle,
                                       "control": "the constant background for the same frames (matched blank arm, fresh brain, same seed)",
                                       "selection": sel, "quantities": list(QUANTITIES), "roles": list(ROLES), "lobe": args.lobe, "run": args.run,
                                       "order_seed": order_seed, "tag": args.tag},
                             retina=retina_rec, cache_dir=args.cache_dir)
    prov["execution"]["device"] = str(dev)
    try:
        import torch
        prov["execution"]["device_name"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    except Exception:  # noqa: BLE001
        prov["execution"]["device_name"] = None
    if hook_info is not None:
        prov["model"]["optic_hook_info"] = to_jsonable(hook_info)
    gen = "scripts/object_round3_localizer.py record " + " ".join(map(shlex.quote, sys.argv[2:]))
    for label, r in recs.items():
        r.meta.update({"provenance": {k: prov[k] for k in ("flyverse_commit", "source_fingerprint", "compiled_connectome", "model", "execution", "stimulus")},
                       "retina_mode": RETINA_MODE, "seed": seed, "generator": gen})
        tr.save_recording(r, str(out) + f"_{label}")
    rad = stim.arrays(); rad.update({"col_az_el": col_az_el, "col_dir": np.asarray(fb.retina.col_dir), "col_side": np.asarray(fb.retina.col_side).astype(str),
                                     "col_hex": np.asarray(fb.retina.col_hex), "retina_mode": np.str_(RETINA_MODE), **{f"checksum__{k}": v for k, v in sums.items()}})
    np.savez_compressed(str(out) + "_radiance.npz", **rad)
    for label, nd in nodes_out.items():
        suffix = "_nodes.npz" if label == "stim" else "_nodes_blank.npz"
        np.savez_compressed(str(out) + suffix, node_az_deg=stim.track["node_az_deg"], node_el_deg=stim.track["node_el_deg"], node_order=stim.track["order"],
                            node_column_equivalents=stim.track["node_column_equivalents"], grid_sha256=np.str_(grid_sha), arm=np.str_(label),
                            presented=np.bool_(label == "stim"), idx=recs[label].idx, body_ids=recs[label].body_ids, types=recs[label].types.astype(str), **nd)
    with open(str(out) + "_prov.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(prov), f, indent=1)
    presented_sum = np.asarray(stim.presented(), np.float64).sum((1, 2)); blank_sum = float(np.asarray(stim.blank, np.float64).sum())
    checks = {lab: bool(np.allclose(sums[lab], presented_sum if present else blank_sum)) for lab, present in arms}
    summary = {"stimulus": stim.name, "params": stim.params, "arms": [a for a, _ in arms], "device": str(dev), "device_name": prov["execution"].get("device_name"),
               "seed": seed, "run": args.run, "lobe": args.lobe, "order_seed": order_seed, "tag": args.tag,
               "optic_overrides": common.parse_kv(args.optic), "lif_overrides": common.parse_kv(args.lif), "n_frames": stim.n_frames, "n_nodes": int(len(nodes)),
               "grid_sha256": grid_sha, "retina_mode": RETINA_MODE, "checksum_equal_stim_vs_presented": checks.get("stim"), "checksum_per_arm": checks,
               "wall_s": walls, "fps": {k: stim.n_frames / max(v, 1e-9) for k, v in walls.items()},
               "hook_info": to_jsonable(hook_info) if hook_info is not None else None, "generator": gen}
    with open(str(out) + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=1)
    print(f"written {out}_{{stim,blank}}.npz, {out}_radiance.npz, {out}_nodes[_blank].npz, {out}_prov.json, {out}_summary.json; device {dev} "
          f"({summary['device_name']}); checksums {checks}; wall {', '.join(f'{k} {v:.0f} s' for k, v in walls.items())}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- the batch
def run_stem(out: str, lobe: str, run: int) -> str:
    return f"{out}/loc/loc45_{lobe}_r{run}"


def job_line(out: str, lobe: str, run: int, minutes_note: str = "", seed_offset: int = 0) -> str:
    stem = run_stem(out, lobe, run)
    return (f"mkdir -p {out}/loc && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"python scripts/object_round3_localizer.py record --lobe {lobe} --run {run} --seed {seed_offset} --order-seed {seed_offset + run} --tag {TAG} --out {stem} > {stem}.txt 2>&1; "
            f"st=$?; tail -4 {stem}.txt; exit $st")


def build_jobs(out: str, runs: int, seed_offset: int = 0) -> list[dict]:
    return [{"job": f"loc45_{lobe}_r{k}", "lobe": lobe, "run": k, "stem": run_stem(out, lobe, k), "expect": run_stem(out, lobe, k) + "_summary.json",
             "line": job_line(out, lobe, k, seed_offset=seed_offset)} for lobe in LOBES for k in range(runs)]


def tree_state() -> dict:
    """The working tree at plan time (cluster_run ships every file that differs from origin/main): commit, porcelain status,
    the sha256 of the simulation sources and of this analysis code."""
    def run(cmd):
        try:
            return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"unavailable: {e!r}"
    files = sorted(glob.glob(str(ROOT / "flyverse" / "*.py")) + glob.glob(str(ROOT / "flyverse" / "interp" / "*.py")) +
                   [str(ROOT / "scripts" / f) for f in ("probe_synthetic_stimuli.py", "object_round3_localizer.py", "cluster_run.py")])
    sha = {str(Path(f).relative_to(ROOT)).replace("\\", "/"): sha256_file(f) for f in files if Path(f).exists()}
    return {"commit": run(["git", "rev-parse", "HEAD"]), "status_porcelain": run(["git", "status", "--porcelain"]).splitlines(),
            "diff_stat_vs_origin_main": run(["git", "diff", "--stat", "origin/main"]).splitlines()[-1:], "sha256_lf": sha,
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def cmd_plan(args) -> int:
    out = args.out.rstrip("/")
    Path(out).mkdir(parents=True, exist_ok=True); Path(out, "loc").mkdir(exist_ok=True)
    jobs = build_jobs(out, args.runs, args.seed_offset)
    log = f"out/{args.name}_cluster.log"
    lines = [j["line"] for j in jobs]
    sh = ["#!/bin/sh", f"# generated by scripts/object_round3_localizer.py plan on {time.strftime('%Y-%m-%d %H:%M')}: {len(jobs)} jobs, one process each", "set -e",
          f"mkdir -p {out}/loc out", f'if [ -f {log} ]; then mv {log} "out/{args.name}_cluster.$(date +%Y%m%dT%H%M%S).log"; fi',
          f"python scripts/cluster_run.py --name {args.name} --minutes {args.minutes} --arm-block fam \\"]
    sh += [f"  {shlex.quote(l)} \\" for l in lines]
    sh += [f"  --fetch {out}/loc/ 2>&1 | tee {log}"]
    (Path(out) / "batch.sh").write_text("\n".join(sh) + "\n", encoding="utf-8")
    plan = {"name": args.name, "minutes": args.minutes, "runs": args.runs, "out": out, "lobes": LOBES, "tag": TAG, "arm_block": "fam",
            "probe": {"size_deg": PROBE_DEG, "contrast": CONTRAST, "flash_s": FLASH_S, "blank_s": BLANK_S, "grid_spacing_deg": SPACING_DEG,
                      "grid_radius_deg": GRID_RADIUS_DEG, "box_radius_deg": BOX_RADIUS_DEG},
            "jobs": jobs, "n_processes": len(jobs), "fetch": out + "/loc/", "log": log}
    with open(Path(out) / "jobs.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=1)
    with open(Path(out) / "predeclared.json", "w", encoding="utf-8") as f:
        json.dump({"predeclared": PREDECLARED, "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generator": " ".join(sys.argv),
                   "analysis_code_sha256_at_stamp": analysis_code()}, f, indent=1)
    with open(Path(out) / "tree_state.json", "w", encoding="utf-8") as f:
        json.dump(tree_state(), f, indent=1)
    print(f"{len(jobs)} job(s) -> {out}/ (log {log}); batch.sh, jobs.json, predeclared.json (stamped), tree_state.json written; block key fam = {TAG}")
    for j in jobs:
        print(f"  {j['job']:16s} {j['stem']}")
    return 0


def cmd_submit(args) -> int:
    out = args.out.rstrip("/")
    with open(Path(out) / "jobs.json", encoding="utf-8") as f:
        plan = json.load(f)
    lines = [j["line"] for j in plan["jobs"]]
    cmd = [sys.executable, "scripts/cluster_run.py", "--name", plan["name"], "--minutes", str(plan["minutes"]), "--arm-block", plan["arm_block"],
           *lines, "--fetch", plan["fetch"]]
    cmd[2:2] = [v for key in ("target", "node") if getattr(args, key, None) for v in ("--" + key, getattr(args, key))]
    log = Path(plan["log"])
    if log.exists():
        log.rename(log.with_name(log.stem + time.strftime(".%Y%m%dT%H%M%S") + ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"submitting {len(lines)} job(s) through scripts/cluster_run.py --arm-block {plan['arm_block']}; log {log}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=env)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line); f.flush()
        rc = p.wait()
    print(f"cluster_run exit {rc}; log {log}")
    return rc


# ---------------------------------------------------------------------------------------------------- verify (CPU)
def verify_all(out: str, log: str | None, runs: int) -> dict:
    d = f"{out}/loc"
    df = pss.verify_dir(d)
    expected = [j["expect"] for j in build_jobs(out, runs)]
    missing = [e for e in expected if not Path(e).exists()]
    rows = []
    for j in build_jobs(out, runs):
        s = j["stem"]
        summ = json.load(open(s + "_summary.json", encoding="utf-8")) if Path(s + "_summary.json").exists() else {}
        rows.append({"job": j["job"], "lobe": j["lobe"], "run": j["run"], "device": summ.get("device"), "device_name": summ.get("device_name"),
                     "optic": json.dumps(summ.get("optic_overrides")), "optic_ok": (summ.get("optic_overrides") == common.parse_kv(LOBES[j["lobe"]])) if summ else False,
                     "n_nodes": summ.get("n_nodes"), "grid_sha256": (summ.get("grid_sha256") or "")[:12], "seed": summ.get("seed"), "order_seed": summ.get("order_seed"),
                     "checksums": summ.get("checksum_per_arm"), "console": Path(s + ".txt").exists(),
                     "wall_min": (sum(summ.get("wall_s", {}).values()) / 60.0) if summ else None})
    jobs_df = pd.DataFrame(rows)
    grids = set(g for g in jobs_df["grid_sha256"].dropna() if g)
    devices = sorted(set(str(x) for x in jobs_df["device_name"].dropna()))
    problems = [f"missing {m}" for m in missing]
    node_files = 0
    for j in build_jobs(out, runs):
        identity = None
        for suffix in ("_nodes.npz", "_nodes_blank.npz", "_radiance.npz", "_stim.npz", "_blank.npz"):
            path = Path(j["stem"] + suffix)
            try:
                with np.load(path, allow_pickle=False) as z:
                    if suffix.startswith("_nodes"):
                        required = {"idx", "body_ids", "types", "grid_sha256", "node_az_deg", "node_el_deg",
                                    "resp__on", "resp__base", "resp__on1", "resp__off"}
                        if not required <= set(z.files):
                            raise ValueError(f"missing arrays {required - set(z.files)}")
                        current = (z["body_ids"], z["idx"], str(z["grid_sha256"]))
                        if identity is not None and (not np.array_equal(current[0], identity[0])
                                                   or not np.array_equal(current[1], identity[1]) or current[2] != identity[2]):
                            raise ValueError("stimulus/blank body or grid mismatch")
                        identity = current; node_files += 1
            except Exception as exc:
                problems.append(f"{path}: {exc}")
    if len(df) and int((~df["ok"]).sum()):
        problems += [f"{r['run']}: device {r['device_summary']}/{r['device_prov']} console_cuda {r['console_cuda']} checksums {r['checksums_ok']} arms {r['arms_ok']}"
                     for r in df.to_dict("records") if not r["ok"]]
    if len(grids) > 1:
        problems.append(f"grids differ across runs: {sorted(grids)}")
    if len(devices) != 1 or devices[0] in ("cpu", "None", ""):
        problems.append(f"expected one known CUDA device model: {devices}")
    if not jobs_df["optic_ok"].all():
        problems.append("optic overrides differ from the lobe in " + ", ".join(jobs_df.loc[~jobs_df['optic_ok'], 'job']))
    n_consoles = int(jobs_df["console"].sum())
    if n_consoles != len(expected):
        problems.append(f"consoles {n_consoles} of {len(expected)} runs")
    line = None
    if log and Path(log).exists():
        ls = [l for l in Path(log).read_text(encoding="utf-8", errors="replace").splitlines() if "job(s)" in l]
        line = ls[-1] if ls else None
    if line is None:
        problems.append(f"cluster log: no 'job(s)' line ({log})")
    elif not re.search(r"\b0 failed\b", line):
        problems.append(f"cluster log: {line}")
    return {"dir": d, "n_expected": len(expected), "n_present": len(expected) - len(missing), "n_consoles": n_consoles, "n_node_files": node_files, "devices": devices, "grids": sorted(grids),
            "cluster_log_line": line, "runs": to_jsonable(jobs_df.to_dict("records")), "verify_dir": to_jsonable(df.to_dict("records")) if len(df) else [],
            "problems": problems}


def cmd_verify(args) -> int:
    rep = verify_all(args.out.rstrip("/"), args.log, args.runs)
    common.print_table(pd.DataFrame(rep["runs"]).drop(columns=["checksums"]), max_rows=50)
    print(f"cluster log: {rep['cluster_log_line']}\ndevices {rep['devices']}; grids {rep['grids']}; {rep['n_present']} of {rep['n_expected']} runs, {rep['n_consoles']} consoles")
    print(f"problems: {rep['problems'] or 'none'}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    return 1 if rep["problems"] else 0


# ---------------------------------------------------------------------------------------------------- the pooled fit
def load_nodes(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def response(nd: dict, window: str) -> np.ndarray:
    """(n_nodes, n_cells) `window` - base of one arm."""
    return np.asarray(nd[f"resp__{window}"], np.float64) - np.asarray(nd["resp__base"], np.float64)


def box_masks(node_az, node_el, cen_az, cen_el, radius_deg) -> np.ndarray:
    """(n_nodes, n_cells) bool: node inside the cell's box (NaN centroid -> no nodes)."""
    M = np.zeros((len(node_az), len(cen_az)), bool)
    for j in range(len(cen_az)):
        if np.isfinite(cen_az[j]) and np.isfinite(cen_el[j]):
            M[:, j] = pss.angular_distance_deg(node_az, node_el, cen_az[j], cen_el[j]) <= radius_deg
    return M


def fit_boxed(R: np.ndarray, node_az, node_el, M: np.ndarray, z_min: float, spacing_deg: float = SPACING_DEG, min_nodes: int = MIN_BOX_NODES) -> pd.DataFrame:
    """PREDECLARED['fit_rule']: probe_synthetic_stimuli.fit_rf applied per cell to the nodes of its box; `free_peak_*` is the
    argmax |R| over the whole grid (criterion-free); cells with < min_nodes box nodes get NaN everywhere and fitted False."""
    n_nodes, n_cells = R.shape
    Rm = np.where(M, R, np.nan)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        med = np.nanmedian(Rm, axis=0); mad = 1.4826 * np.nanmedian(np.abs(Rm - med[None, :]), axis=0)
    n_box = M.sum(0)
    absR = np.where(M & np.isfinite(R), np.abs(R), -np.inf)               # outside the box (or NaN) never wins the argmax
    ip = np.argmax(absR, axis=0)
    peak = np.where(n_box > 0, R[ip, np.arange(n_cells)], np.nan)
    ifree = np.argmax(np.abs(np.nan_to_num(R)), axis=0)
    rows = []
    for j in range(n_cells):
        pk = float(peak[j]) if np.isfinite(peak[j]) else np.nan; nz = float(mad[j]) if np.isfinite(mad[j]) else np.nan
        usable = n_box[j] >= min_nodes and np.isfinite(pk)
        s = 1.0 if not usable or pk >= 0 else -1.0
        fitted = bool(usable and abs(pk) > 1e-9 and ((nz == 0 and abs(pk) > 0) or (nz > 0 and abs(pk) >= z_min * nz)))
        if fitted:
            thr = 0.5 * abs(pk); r = s * Rm[:, j]
            above = np.flatnonzero(np.nan_to_num(r, nan=-np.inf) >= thr)
            w = r[above] - thr + 1e-12 * abs(pk)
            az0 = float((w * node_az[above]).sum() / w.sum()); el0 = float((w * node_el[above]).sum() / w.sum())
            width = float(max(2.0 * np.sqrt(len(above) * spacing_deg ** 2 / np.pi), spacing_deg)); n_above = int(len(above))
        else:
            az0 = el0 = width = np.nan; n_above = 0
        rows.append({"az_deg": az0, "el_deg": el0, "width_deg": width, "peak": pk, "n_nodes_above_threshold": n_above, "sign": int(s), "noise_mad": nz,
                     "z_peak": float(abs(pk) / nz) if (usable and nz > 0) else (float("inf") if usable and abs(pk) > 0 else np.nan), "fitted": fitted,
                     "peak_node_az_deg": float(node_az[ip[j]]) if usable else np.nan, "peak_node_el_deg": float(node_el[ip[j]]) if usable else np.nan,
                     "n_box_nodes": int(n_box[j]), "free_peak_az_deg": float(node_az[ifree[j]]), "free_peak_el_deg": float(node_el[ifree[j]]),
                     "free_peak_in_box": bool(M[ifree[j], j]) if n_box[j] > 0 else False, "free_peak_value": float(R[ifree[j], j])})
    return pd.DataFrame(rows)


def fitted_fraction(R, node_az, node_el, M, types, z_min) -> dict:
    f = fit_boxed(R, node_az, node_el, M, z_min)["fitted"].to_numpy(bool)
    return {t: float(f[types == t].mean()) for t in sorted(set(types))}


def threshold_by_type(sens: pd.DataFrame, bound: float = FALSE_FIT_MAX) -> dict:
    """PREDECLARED['threshold_rule']: per type the smallest z of the ladder whose blank-arm false-fit rate <= bound."""
    out = {}
    for t, g in sens.groupby("type"):
        g = g.sort_values("z_min")
        ok = g[g["false_fit_rate_blank_pooled"] <= bound]
        out[t] = {"z_min": float(ok["z_min"].iloc[0]) if len(ok) else float(g["z_min"].iloc[-1]), "reached_bound": bool(len(ok))}
    return out


def passes_threshold(fit: pd.DataFrame, z_min) -> np.ndarray:
    """Apply fit_boxed's exact rule, including a nonzero signal with zero MAD.

    Infinite z is evidence of zero estimated noise, not a missing observation.
    The amplitude floor and minimum node count still apply to every map row.
    """
    peak = np.abs(fit["peak"].to_numpy(float))
    noise = fit["noise_mad"].to_numpy(float)
    return (np.isfinite(peak) & (peak > 1e-9)
            & (fit["n_box_nodes"].to_numpy() >= MIN_BOX_NODES)
            & ((noise == 0) | ((noise > 0) & (peak >= np.asarray(z_min) * noise))))


def cmd_rfmap(args) -> int:
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)                        # nanmedian over all-NaN slices (unfitted types) is expected
    out = args.out.rstrip("/")
    stems = [run_stem(out, args.lobe, k) for k in range(args.runs)]
    missing = [s + suffix for s in stems for suffix in ("_nodes.npz", "_nodes_blank.npz")
               if not Path(s + suffix).exists() or Path(s + suffix).stat().st_size == 0]
    if missing:
        sys.exit(f"incomplete localizer: {len(missing)} missing/empty node files: {missing}")
    if not args.no_verify:
        bad = pss.verify_runs([f"{out}/loc"], require_cuda=not args.allow_cpu)
        if bad:
            sys.exit(f"batch verification failed: {bad}")
    print(f"n_runs {len(stems)} ({out}/loc/loc45_{args.lobe}_r*_nodes.npz), window {args.window}", flush=True)
    S = [load_nodes(s + "_nodes.npz") for s in stems]; B = [load_nodes(s + "_nodes_blank.npz") for s in stems]
    grids = {str(nd["grid_sha256"]) for nd in S + B}
    if len(grids) != 1:
        sys.exit(f"the runs do not share one grid: {grids}")
    node_az, node_el = S[0]["node_az_deg"].astype(float), S[0]["node_el_deg"].astype(float)
    types = S[0]["types"].astype(str); body = S[0]["body_ids"]; idx = S[0]["idx"].astype(int); quantity = S[0]["quantity"].astype(str)
    for nd in S + B:
        assert np.array_equal(nd["idx"].astype(int), idx), "the runs do not record the same cells"
    c, retina = load_model(args.cache_dir)
    prior = anatomical_prior(c, retina, idx)
    M = box_masks(node_az, node_el, prior["anat_input_az_deg"].to_numpy(float), prior["anat_input_el_deg"].to_numpy(float), args.box_radius)
    Rs = [response(nd, args.window) for nd in S]; Rb = [response(nd, args.window) for nd in B]
    Rp, Rbp = np.mean(Rs, axis=0), np.mean(Rb, axis=0)
    # the z ladder: coverage on the pooled stimulus arm, false-fit rate on the pooled blank arm (and per run, any run)
    sens_rows = []
    for zm in Z_LADDER:
        cov = fitted_fraction(Rp, node_az, node_el, M, types, zm); ff = fitted_fraction(Rbp, node_az, node_el, M, types, zm)
        per_run_s = np.stack([fit_boxed(R, node_az, node_el, M, zm)["fitted"].to_numpy(bool) for R in Rs])
        per_run_b = np.stack([fit_boxed(R, node_az, node_el, M, zm)["fitted"].to_numpy(bool) for R in Rb])
        for t in sorted(set(types)):
            m = types == t
            sens_rows.append({"type": t, "z_min": zm, "coverage_pooled": cov[t], "false_fit_rate_blank_pooled": ff[t],
                              "coverage_all_runs": float(per_run_s[:, m].all(0).mean()), "coverage_any_run": float(per_run_s[:, m].any(0).mean()),
                              "false_fit_rate_blank_any_run": float(per_run_b[:, m].any(0).mean()), "n_bodies": int(m.sum())})
    sens = pd.DataFrame(sens_rows)
    zstar = threshold_by_type(sens)
    # the map at z* (per type), z 5 beside it
    z_of_cell = np.array([zstar[t]["z_min"] for t in types])
    fit_star = fit_boxed(Rp, node_az, node_el, M, z_min=0.0)              # centres for every usable cell; `fitted` applied below
    fit_star["z_min_used"] = z_of_cell
    zp = fit_star["z_peak"].to_numpy(float)
    fitted = passes_threshold(fit_star, z_of_cell)
    fit_star["fitted"] = fitted
    fit_star.loc[~fitted, ["az_deg", "el_deg", "width_deg"]] = np.nan
    fit_star.loc[~fitted, "n_nodes_above_threshold"] = 0
    fit_star["fitted_z5"] = passes_threshold(fit_star, 5.0)
    blank_star = fit_boxed(Rbp, node_az, node_el, M, z_min=0.0)
    zb = blank_star["z_peak"].to_numpy(float)
    fit_star["blank_fitted_at_z"] = passes_threshold(blank_star, z_of_cell)
    # per-run fits at z* (reproducibility), split-half pooled fits
    per_run = [fit_boxed(R, node_az, node_el, M, z_min=0.0) for R in Rs]
    F = np.stack([passes_threshold(f, z_of_cell) for f in per_run])
    A = np.stack([np.where(F[i], f["az_deg"].to_numpy(float), np.nan) for i, f in enumerate(per_run)])
    E = np.stack([np.where(F[i], f["el_deg"].to_numpy(float), np.nan) for i, f in enumerate(per_run)])
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        fit_star["n_runs_fitted"] = F.sum(0)
        fit_star["az_sd_runs"] = np.nanstd(A, axis=0, ddof=0) if len(Rs) > 1 else np.nan
        fit_star["el_sd_runs"] = np.nanstd(E, axis=0, ddof=0) if len(Rs) > 1 else np.nan
        fit_star["centre_spread_deg"] = (np.nanmax([pss.angular_distance_deg(A[i], E[i], A[k], E[k]) for i in range(len(Rs)) for k in range(i + 1, len(Rs))], axis=0)
                                         if len(Rs) > 1 else np.nan)
        if len(Rs) >= 2:
            h1 = fit_boxed(np.mean(Rs[0::2], axis=0), node_az, node_el, M, z_min=0.0); h2 = fit_boxed(np.mean(Rs[1::2], axis=0), node_az, node_el, M, z_min=0.0)
            ok1 = passes_threshold(h1, z_of_cell); ok2 = passes_threshold(h2, z_of_cell)
            d = pss.angular_distance_deg(h1["az_deg"], h1["el_deg"], h2["az_deg"], h2["el_deg"])
            fit_star["split_half_centre_distance_deg"] = np.where(ok1 & ok2, d, np.nan)
            fit_star["split_half_both_fitted"] = ok1 & ok2
        else:
            fit_star["split_half_centre_distance_deg"] = np.nan; fit_star["split_half_both_fitted"] = False
        # blank-arm noise as an independent z, the anatomical comparison, the free-peak chance
        bsd = np.mean([np.where(M, R, np.nan) for R in Rb], axis=0)
        bsd = np.nanstd(bsd, axis=0)
        fit_star["blank_node_sd"] = bsd; fit_star["z_blank"] = np.abs(fit_star["peak"].to_numpy(float)) / np.where(bsd > 0, bsd, np.nan)
        fit_star["chance_in_box"] = M.mean(0)
        fit_star = pd.concat([fit_star.reset_index(drop=True), prior.reset_index(drop=True)], axis=1)
        fit_star["anat_distance_deg"] = pss.angular_distance_deg(fit_star["az_deg"], fit_star["el_deg"], fit_star["anat_az_deg"], fit_star["anat_el_deg"])
        fit_star["anat_input_distance_deg"] = pss.angular_distance_deg(fit_star["az_deg"], fit_star["el_deg"], fit_star["anat_input_az_deg"], fit_star["anat_input_el_deg"])
        fit_star["free_peak_anat_input_distance_deg"] = pss.angular_distance_deg(fit_star["free_peak_az_deg"], fit_star["free_peak_el_deg"],
                                                                                 fit_star["anat_input_az_deg"], fit_star["anat_input_el_deg"])
    spk = np.full(len(types), np.nan)
    if "spk__on" in S[0]:
        sp_idx = S[0]["spiking_cells"].astype(int)
        spk[sp_idx] = np.mean([np.abs(nd["spk__on"] - nd["spk__base"]).max(0) for nd in S], axis=0) * (1000.0 / common.FRAME_MS)
    fit_star["spikes_on_minus_base_hz_peak"] = spk
    fit_star.insert(0, "type", types); fit_star.insert(0, "bodyId", [str(int(b)) for b in body])
    fit_star["quantity"] = quantity; fit_star["window"] = args.window; fit_star["model_index"] = idx
    fit_star["box_radius_deg"] = args.box_radius; fit_star["lobe"] = args.lobe
    m0 = fit_star[pss.RF_CSV_COLUMNS + [c_ for c_ in fit_star.columns if c_ not in pss.RF_CSV_COLUMNS]]
    # per type
    rows = []
    for t, g in m0.groupby("type", sort=True):
        ok = g["fitted"].to_numpy(bool); zs = zstar[t]
        ffb = float(g["blank_fitted_at_z"].to_numpy(bool).mean())
        dd = g.loc[ok, "anat_input_distance_deg"].to_numpy(float); dd = dd[np.isfinite(dd)]
        d1 = g.loc[ok, "anat_distance_deg"].to_numpy(float); d1 = d1[np.isfinite(d1)]
        fp = g["free_peak_in_box"].to_numpy(bool); ch = float(np.nanmean(g["chance_in_box"]))
        cov = float(ok.mean())
        localized = bool((cov - ffb) >= 0.05 and fp.mean() >= 2.0 * ch)
        rows.append({"type": t, "lobe": args.lobe, "n_bodies": int(len(g)), "z_min": zs["z_min"], "z_reached_false_fit_bound": zs["reached_bound"],
                     "n_fitted": int(ok.sum()), "coverage": cov, "false_fit_rate_blank_arm_mean": ffb, "coverage_minus_false_fit": cov - ffb,
                     "coverage_z5": float(g["fitted_z5"].to_numpy(bool).mean()),
                     "false_fit_rate_z5": float(sens.loc[(sens["type"] == t) & (sens["z_min"] == 5.0), "false_fit_rate_blank_pooled"].iloc[0]),
                     "n_fitted_all_runs": int((g["n_runs_fitted"] == len(Rs)).sum()), "n_fitted_any_run": int((g["n_runs_fitted"] > 0).sum()), "n_runs": len(Rs),
                     "width_median_deg": float(np.nanmedian(g.loc[ok, "width_deg"])) if ok.any() else np.nan,
                     "width_iqr_deg": [float(np.nanpercentile(g.loc[ok, "width_deg"], 25)), float(np.nanpercentile(g.loc[ok, "width_deg"], 75))] if ok.any() else None,
                     "peak_median": float(np.nanmedian(np.abs(g.loc[ok, "peak"]))) if ok.any() else np.nan, "peak_median_all": float(np.nanmedian(np.abs(g["peak"]))),
                     "z_peak_median_all": float(np.nanmedian(g["z_peak"].replace(np.inf, np.nan))), "z_blank_median": float(np.nanmedian(g["z_blank"])),
                     "blank_node_sd_median": float(np.nanmedian(g["blank_node_sd"])),
                     "anat_input_distance_median_deg": float(np.median(dd)) if len(dd) else np.nan,
                     "anat_input_within_10deg": float((dd <= 10).mean()) if len(dd) else np.nan, "anat_input_within_15deg": float((dd <= 15).mean()) if len(dd) else np.nan,
                     "anat_distance_median_deg": float(np.median(d1)) if len(d1) else np.nan,
                     "anat_within_10deg": float((d1 <= 10).mean()) if len(d1) else np.nan, "anat_within_15deg": float((d1 <= 15).mean()) if len(d1) else np.nan,
                     "free_peak_in_box": float(fp.mean()), "chance_in_box": ch, "free_peak_in_box_over_chance": float(fp.mean() / ch) if ch > 0 else np.nan,
                     "free_peak_anat_input_distance_median_deg": float(np.nanmedian(g["free_peak_anat_input_distance_deg"])),
                     "centre_spread_median_deg": float(np.nanmedian(g.loc[ok, "centre_spread_deg"])) if (len(Rs) > 1 and ok.any()) else np.nan,
                     "split_half_centre_distance_median_deg": float(np.nanmedian(g.loc[ok, "split_half_centre_distance_deg"])) if ok.any() else np.nan,
                     "split_half_both_fitted_frac": float(g["split_half_both_fitted"].to_numpy(bool).mean()),
                     "anat_input_r50_median_deg": float(np.nanmedian(g["anat_input_r50_deg"])), "n_input_columns_median": float(np.nanmedian(g["n_input_columns"])),
                     "box_nodes_median": float(np.median(g["n_box_nodes"])), "spikes_on_minus_base_hz_peak_median": float(np.nanmedian(g["spikes_on_minus_base_hz_peak"])),
                     "localized": localized,
                     "az_median": float(np.nanmedian(g.loc[ok, "az_deg"])) if ok.any() else np.nan, "el_median": float(np.nanmedian(g.loc[ok, "el_deg"])) if ok.any() else np.nan})
    pt = pd.DataFrame(rows)
    show = ["type", "n_bodies", "z_min", "n_fitted", "coverage", "false_fit_rate_blank_arm_mean", "coverage_z5", "false_fit_rate_z5", "width_median_deg",
            "anat_input_distance_median_deg", "anat_input_within_15deg", "free_peak_in_box", "chance_in_box", "z_blank_median", "split_half_centre_distance_median_deg", "localized"]
    common.print_table(pt[show], floatfmt="{:+.3f}")
    print("z ladder: coverage (pooled) / false-fit rate (pooled blank arm):")
    common.print_table(sens.pivot(index="type", columns="z_min", values=["coverage_pooled", "false_fit_rate_blank_pooled"]).reset_index(), floatfmt="{:.3f}")
    csv = Path(args.csv or f"{out}/rfmap_r3_{args.lobe}.csv"); csv.parent.mkdir(parents=True, exist_ok=True)
    m0.to_csv(csv, index=False)
    prov_path = stems[0] + "_prov.json"
    prov0 = json.load(open(prov_path, encoding="utf-8")) if os.path.exists(prov_path) else common.provenance(c)
    prov0["analysis_code"] = analysis_code()
    res = Result.new("trace", prov0)
    res.replicates = {"n": len(stems), "unit": "runs", "runs": [{"run_index": i, "file": s + "_nodes.npz", "seed": pss._seed_of(s + "_nodes.npz"),
                                                                  "device": pss._device_of(s + "_nodes.npz"), "device_name": pss._summary_of(s + "_nodes.npz").get("device_name"),
                                                                  "order_seed": pss._summary_of(s + "_nodes.npz").get("order_seed")} for i, s in enumerate(stems)],
                      "null": {"blank_arm_files": [s + "_nodes_blank.npz" for s in stems],
                               "reading": "the same fit on the run-pooled blank arm: its 'fitted' fraction per type at each z is the false-fit rate"},
                      "pooling": "per-node responses averaged over the runs before the fit; per-run fits on file (n_runs_fitted, centre_spread_deg), split-half pooled fits (split_half_centre_distance_deg)"}
    per_run_rows = []
    for i, f in enumerate(per_run):
        bf = fit_boxed(Rb[i], node_az, node_el, M, 0.0)
        for t in sorted(set(types)):
            mask = types == t
            per_run_rows.append({"run": stems[i], "type": t, "lobe": args.lobe,
                                 "coverage": float(F[i, mask].mean()),
                                 "false_fit_rate": float(passes_threshold(bf, z_of_cell)[mask].mean()),
                                 "z_min": zstar[t]["z_min"]})
    res.add_table("rf_per_type", pt); res.add_table("rf_map", m0.drop(columns=["model_index"])); res.add_table("rf_sensitivity", sens)
    res.add_table("rf_per_run", per_run_rows)
    loc_params = pss._summary_of(stems[0] + "_nodes.npz").get("params", {})
    res.summary = {"schema": pss.RFMAP_SCHEMA, "rf_map_csv": str(csv), "n_runs": len(stems), "lobe": args.lobe, "window": args.window,
                   "z_min": {t: zstar[t]["z_min"] for t in zstar}, "z_ladder": list(Z_LADDER), "false_fit_bound": FALSE_FIT_MAX, "box_radius_deg": args.box_radius,
                   "fit_rule": PREDECLARED["fit_rule"], "threshold_rule": PREDECLARED["threshold_rule"], "localizable_call": PREDECLARED["localizable_call"],
                   "predeclared": PREDECLARED, "localizer_params": loc_params, "grid_sha256": next(iter(grids)), "n_nodes": int(len(node_az)),
                   "coverage": {r["type"]: r["coverage"] for r in rows}, "false_fit_rate_blank_arm": {r["type"]: r["false_fit_rate_blank_arm_mean"] for r in rows},
                   "localized": {r["type"]: r["localized"] for r in rows},
                   "anat_input_distance_median_deg": {r["type"]: r["anat_input_distance_median_deg"] for r in rows},
                   "free_peak_in_box": {r["type"]: (r["free_peak_in_box"], r["chance_in_box"]) for r in rows},
                   "optic_overrides": pss._optic_overrides_of(stems[0] + "_nodes.npz"), "columns": pss.RF_CSV_COLUMNS,
                   "reading": "a pooled map's magnitudes: coverage is stated with the blank-arm false-fit rate at the same z; anat_az_deg / anat_el_deg keep round 2's "
                              "single-column meaning (the ladder fallback), anat_input_* is the input-weighted column set; free_peak_in_box vs chance_in_box is the "
                              "criterion-free check that the prior box did not manufacture the fit"}
    stamp_path = Path(out) / "predeclared.json"
    stamp = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.exists() else {}
    res.summary.update(predeclared=stamp.get("predeclared", PREDECLARED),
                       predeclared_stamped_utc=stamp.get("stamped_utc"),
                       predeclared_matches_live_dict=stamp.get("predeclared") == to_jsonable(PREDECLARED),
                       analysis_code_stamped=stamp.get("analysis_code_sha256_at_stamp"),
                       analysis_code_sha256=analysis_code())
    res.validation = {"name": "RF-map coverage at a measured false-fit rate, and agreement with the anatomical column set (trace.column_of_cells, input-weighted)",
                      "measured": {r["type"]: {k: r[k] for k in ("z_min", "coverage", "false_fit_rate_blank_arm_mean", "anat_input_distance_median_deg", "anat_input_within_15deg",
                                                                 "free_peak_in_box", "chance_in_box", "z_blank_median", "localized")} for r in rows},
                      "status": "measured", "reference": {"Mi1": "hex-annotated medulla cells: the localizer's own ground truth for a column-sized RF"},
                      "source": "scripts/object_round3_localizer.py rfmap"}
    res.files = {"generator": "scripts/object_round3_localizer.py rfmap " + " ".join(map(shlex.quote, sys.argv[2:])), "runs": stems, "csv": str(csv),
                 "analysis_code": analysis_code(), "predeclared": f"{out}/predeclared.json"}
    p = res.save(args.json or f"{ROOT}/out/interp/objr3rf/rfmap_r3_{args.lobe}.json")
    print(f"problems: {res.check() or 'none'}; written {csv} and {p}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------- preview, selfcheck
def cmd_preview(args) -> int:
    c, retina = load_model(args.cache_dir)
    cae = np.asarray(retina.col_az_el)
    cen = lc_centroids(c, retina)
    rep = {"n_columns": int(len(cae))}
    for t, g in cen.groupby("type"):
        rep[f"prior_{t}"] = {"n_cells": int(len(g)), "distinct_single_columns": int(g["anat_column"].nunique()),
                             "single_column_top_counts": g["anat_column"].value_counts().head(4).tolist(),
                             "input_columns_median": float(g["n_input_columns"].median()), "input_r50_median_deg": float(g["anat_input_r50_deg"].median()),
                             "input_r80_median_deg": float(g["anat_input_r80_deg"].median()), "input_rms_median_deg": float(g["anat_input_rms_deg"].median()),
                             "single_column_to_input_centroid_median_deg": float(np.nanmedian(pss.angular_distance_deg(g["anat_az_deg"], g["anat_el_deg"], g["anat_input_az_deg"], g["anat_input_el_deg"]))),
                             "input_weight_with_column_median": float(g["input_weight_with_column"].median())}
    nodes = grid_nodes(cae, cen["anat_input_az_deg"], cen["anat_input_el_deg"], args.grid_spacing, args.grid_radius, args.width)
    whole = pss.localizer_grid(cae, args.grid_spacing, args.width)
    stim = localizer_boxed(cae, nodes, args.width, args.contrast, args.flash_s, args.blank_s, 0)
    rep.update({"grid": {"spacing_deg": args.grid_spacing, "radius_deg": args.grid_radius, "n_nodes": int(len(nodes)), "n_nodes_whole_eye": int(len(whole)),
                         "seconds_per_arm": stim.params["seconds"], "frames_per_arm": stim.n_frames, "grid_sha256": hashlib.sha256(np.ascontiguousarray(nodes).tobytes()).hexdigest(),
                         "column_equivalents_per_node_median": stim.params["column_equivalents_per_node_median"]}})
    az, el, w = pss.sample_points(cae)
    azs = np.arange(-120, 120 + 1e-9, args.grid_spacing); els = np.arange(-70, 70 + 1e-9, args.grid_spacing)
    cover = {}
    for size in (2.0, 3.0, 4.0, 4.5, 8.8):
        best = np.zeros(len(cae))
        for e in els:
            for a in azs:
                best = np.maximum(best, pss.rect_coverage(az, el, w, a, e, size, size))
        cover[f"{size:g}"] = {"best_node_coverage_median": float(np.median(best)), "p10": float(np.percentile(best, 10)), "max": float(best.max()),
                              "columns_ge_0.5": float((best >= 0.5).mean())}
    rep["probe_coverage_on_this_grid"] = cover
    rep["reading"] = ("a column's 7-ray kernel spans 4.5 deg, so a square < 4.5 deg cannot cover it; the best-node coverage per column (the probe's "
                      "effective contrast fraction) is what each size buys on this grid")
    M = box_masks(nodes[:, 0], nodes[:, 1], cen["anat_input_az_deg"].to_numpy(float), cen["anat_input_el_deg"].to_numpy(float), args.box_radius)
    rep["box_nodes_per_lc_cell"] = {"median": float(np.median(M.sum(0))), "min": int(M.sum(0).min()), "max": int(M.sum(0).max())}
    print(json.dumps(to_jsonable(rep), indent=1))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(dict(rep, generator=" ".join(sys.argv), analysis_code=analysis_code())), f, indent=1)
    return 0


def cmd_selfcheck(args) -> int:
    """The pooled fit on synthetic per-node responses: Gaussian bumps recovered inside their boxes, a noise-only cell rejected,
    the blank-arm false-fit rate at the ladder, and the threshold rule -- CPU, no connectome."""
    rng = np.random.default_rng(0)
    az, el = np.meshgrid(np.arange(-60, 61, 4.0), np.arange(-40, 41, 4.0)); az, el = az.ravel(), el.ravel()
    n_cells, n_runs = 200, 5
    cen_az = rng.uniform(-40, 40, n_cells); cen_el = rng.uniform(-25, 25, n_cells)
    true_az = cen_az + rng.normal(0, 5, n_cells); true_el = cen_el + rng.normal(0, 5, n_cells)
    amp = np.where(np.arange(n_cells) < 100, 2.0, 0.0)                        # cells 0-99 respond (pooled z ~ 9), 100-199 do not
    sigma = 5.0
    Rs, Rb = [], []
    for k in range(n_runs):
        G = amp[None, :] * np.exp(-0.5 * (pss.angular_distance_deg(az[:, None], el[:, None], true_az[None, :], true_el[None, :]) / sigma) ** 2)
        Rs.append(G + 0.5 * rng.normal(size=G.shape)); Rb.append(0.5 * rng.normal(size=G.shape))
    M = box_masks(az, el, cen_az, cen_el, 20.0)
    types = np.array(["A"] * 100 + ["B"] * 100)
    Rp, Rbp = np.mean(Rs, axis=0), np.mean(Rb, axis=0)
    sens = []
    for zm in Z_LADDER:
        cov = fitted_fraction(Rp, az, el, M, types, zm); ff = fitted_fraction(Rbp, az, el, M, types, zm)
        for t in ("A", "B"):
            sens.append({"type": t, "z_min": zm, "coverage_pooled": cov[t], "false_fit_rate_blank_pooled": ff[t]})
    sens = pd.DataFrame(sens); zs = threshold_by_type(sens)
    f = fit_boxed(Rp, az, el, M, zs["A"]["z_min"])
    okA = f["fitted"].to_numpy(bool)[:100]
    d = pss.angular_distance_deg(f["az_deg"][:100], f["el_deg"][:100], true_az[:100], true_el[:100])
    single = fit_boxed(Rs[0], az, el, M, zs["A"]["z_min"])["fitted"].to_numpy(bool)[:100].mean()
    print(sens.pivot(index="type", columns="z_min", values=["coverage_pooled", "false_fit_rate_blank_pooled"]).to_string(float_format="{:.3f}".format))
    print(f"z* A {zs['A']}, B {zs['B']}; responders fitted (pooled) {okA.mean():.3f} vs one run {single:.3f}; centre error median {np.nanmedian(d[okA]):.2f} deg, "
          f"p90 {np.nanpercentile(d[okA], 90):.2f}; width median {np.nanmedian(f['width_deg'][:100][okA]):.1f} deg (true FWHM {2.355 * sigma:.1f}); "
          f"non-responders fitted {f['fitted'].to_numpy(bool)[100:].mean():.3f}; free peak in box (responders) {f['free_peak_in_box'][:100].mean():.3f} "
          f"vs chance {M.mean(0)[:100].mean():.3f}; (non-responders) {f['free_peak_in_box'][100:].mean():.3f}")
    ok = okA.mean() > 0.9 and np.nanmedian(d[okA]) < 2.0 and f["fitted"].to_numpy(bool)[100:].mean() <= 0.02 and single < okA.mean()
    edge = np.zeros((len(az), 3)); edge[len(az) // 2, 0] = 1.0; edge[len(az) // 2, 1] = 1e-12
    ef = fit_boxed(edge, az, el, np.ones_like(edge, dtype=bool), 5.0)
    edge_ok = np.array_equal(passes_threshold(ef, 5.0), [True, False, False])
    edge_ok &= np.array_equal(passes_threshold(ef, 5.0), ef.fitted.to_numpy())
    print("zero-MAD / amplitude-floor / zero-signal consistency", "OK" if edge_ok else "FAILED")
    ok = ok and edge_ok
    print("selfcheck", "OK" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def stim_args(p):
        p.add_argument("--width", type=float, default=PROBE_DEG, help="probe square size (deg)")
        p.add_argument("--contrast", type=float, default=CONTRAST, help="Weber contrast of the probe")
        p.add_argument("--flash-s", type=float, default=FLASH_S); p.add_argument("--blank-s", type=float, default=BLANK_S)
        p.add_argument("--grid-spacing", type=float, default=SPACING_DEG); p.add_argument("--grid-radius", type=float, default=GRID_RADIUS_DEG)
        p.add_argument("--box-radius", type=float, default=BOX_RADIUS_DEG)

    r = sub.add_parser("record", help="one run: the localizer + its matched blank on the GPU"); stim_args(r); common.add_common_args(r)
    r.add_argument("--lobe", required=True, choices=list(LOBES)); r.add_argument("--run", type=int, required=True, help="run index: seed = --seed + run, order seed = run")
    r.add_argument("--order-seed", type=int, default=None); r.add_argument("--tag", default=TAG, help="the --arm-block token (recorded)")
    r.add_argument("--out", required=True); r.add_argument("--settle", type=float, default=2.0)
    r.add_argument("--select", default=",".join(SELECTION)); r.add_argument("--allow-cpu", action="store_true")
    pv = sub.add_parser("preview", help="CPU: the grid, the probe's coverage per size, the anatomical prior"); stim_args(pv)
    pv.add_argument("--cache-dir", default=None); pv.add_argument("--json", default=None)
    pl = sub.add_parser("plan", help="write batch.sh, jobs.json, predeclared.json (stamped), tree_state.json")
    pl.add_argument("--out", default="out/objr3rf"); pl.add_argument("--name", default="objr3rf"); pl.add_argument("--minutes", type=int, default=90)
    pl.add_argument("--runs", type=int, default=RUNS)
    pl.add_argument("--seed-offset", type=int, default=0, help="offset both brain seeds and node-order seeds for a fresh replication")
    sm = sub.add_parser("submit", help="ONE cluster_run.py call from jobs.json (--arm-block fam)"); sm.add_argument("--out", default="out/objr3rf")
    sm.add_argument("--target"); sm.add_argument("--node")
    v = sub.add_parser("verify", help="CPU: the fetched batch"); v.add_argument("--out", default="out/objr3rf"); v.add_argument("--log", default="out/objr3rf_cluster.log")
    v.add_argument("--runs", type=int, default=RUNS); v.add_argument("--json", default=None)
    m = sub.add_parser("rfmap", help="CPU: the pooled RF map of one lobe (docs/INTERP.md 2.7 format)")
    m.add_argument("--out", default="out/objr3rf"); m.add_argument("--lobe", required=True, choices=list(LOBES)); m.add_argument("--runs", type=int, default=RUNS)
    m.add_argument("--window", default="on", choices=list(ROLES[:3])); m.add_argument("--box-radius", type=float, default=BOX_RADIUS_DEG)
    m.add_argument("--csv", default=None); m.add_argument("--json", default=None); m.add_argument("--cache-dir", default=None)
    m.add_argument("--no-verify", action="store_true"); m.add_argument("--allow-cpu", action="store_true")
    sub.add_parser("selfcheck", help="CPU: the pooled fit on synthetic bumps")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"record": cmd_record, "preview": cmd_preview, "plan": cmd_plan, "submit": cmd_submit, "verify": cmd_verify, "rfmap": cmd_rfmap,
            "selfcheck": cmd_selfcheck}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
