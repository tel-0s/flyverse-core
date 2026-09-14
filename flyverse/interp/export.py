"""The read-only Neurome probe export (docs/NEUROME_INTERFACE.md section 1) -- a serializer over `common.Result`.

`export(result)` writes one run directory `out/export/<run_id>/` holding

    manifest.json          every field of NEUROME_INTERFACE section 1, taken from the Result's provenance block:
                           run_id, flyverse_commit (+ dirty), dataset_release (the four MaleCNS files + SHA-256),
                           compiled_connectome (the cache fingerprint), model (resolved LIFParams / OpticParams /
                           body thresholds), execution (the REALISED device), stimulus, retina, units, tables
    result.json            the source Result, byte-for-byte, with its SHA-256 in the manifest
    readout_per_body.csv   one row per (body, quantity); bodyId a decimal string; LC11 / LC10a never pooled
    contributions.csv      body_pre / body_post edge contributions with the sign and gain rules that made them
    sensitivity.csv        lesion / hold deltas with replicate scatter
    retina_columns.csv     the 1,466 hex columns: direction, azimuth / elevation and the photoreceptor bodies in each
    retina_bodies.csv      the 5,895 photoreceptors: bodyId, type, column, spectral sensitivity
    retina_radiance.csv    the radiance of the OBJECT arm, per column per frame, [UV, B, G, R]  (Parquet above
                           `parquet_rows`, as every table is)
    retina_radiance_blank  the matched BLANK arm's radiance in the SAME schema, from the blank run's own record --
                           written whenever the caller supplies it, so the footprint is portable without the npz
    retina_object_track    the ball per frame as the eye sees it: offset, azimuth, centre elevation, angular diameter,
                           and (with the blank) the columns it dims -- size and retinal position both on file
    <tool tables>.csv      the tool's own tables (`per_type`, `reference_per_type`, `delta_links`, ...), hashed the
                           same way and marked `role: tool`, so the directory does not depend on parsing result.json

Nothing is computed here that a tool did not already compute: the Result schema (docs/INTERP.md section 3) carries
the numbers, the provenance and the populations, so this module is field mapping, hashing and the refusal rule
(a Result whose `check()` is non-empty is not exportable).

`source_fingerprint` / `match_sources` close the one hole in that provenance: a GPU job runs from a cluster copy
with no `.git`, so its own `git_state()` is `commit 'unknown'`. The record job hashes the modules it actually
imported and the CPU analysis step matches them against the checkout, which names the commit (and says so).

Two adapters live here because the export's own validation needs them and they are pure field mapping:

  * `result_from_object_sweep` turns the round-3 object-sweep probe runs (`scripts/probe_object_sweep.py`, plus the
    per-cell npz that `scripts/interp_export.py record` captures from the same protocol) into a Result -- per-type
    arms with `common.compare` against the none-vs-none null, and two `readout_per_body` rows per spiking body
    (`upstream_drive_mV`, `output_Hz`).
  * `links_to_contributions` maps a `common.links` edge table onto the Neurome `contributions` columns, and
    `static_decompose_result` wraps it into a Result for the structural (static) decomposition.

CPU only; no torch, no GPU. `verify(run_dir)` re-reads a finished export and re-checks it (hashes, keys, row counts,
the two-rows-per-body rule, bodyId membership in cache/neurons.parquet).

Revision 2 (Neurome's intake of the size ladder, `D:\\Projects\\neurome\\reports\\flyverse-size-tuning-intake.md`):

  * the two references are **split**. `control_value` is the matched blank arm (arm b) of the SAME recording ->
    `paired_control_ids`; `null_mean` / `null_sd` / `z_vs_null` / `verdict` come from INDEPENDENT blank/blank runs ->
    `null_reference_ids`. Revision 1 wrote the second set under `control_ids` while the first set produced the
    numbers; `control_ids` survives as a **deprecated alias of `null_reference_ids`** for one revision and the
    manifest's `conventions` block says so.
  * `manifest.statistic_definitions` gives one sentence per quantity / per-type statistic, so a headline drive figure
    cannot be read as an absolute membrane voltage.
  * the retina record declares `retina.mode` (`geometry_replay` | `in_loop_capture`) with the pinned pose and what
    was replayed -- never "capture" where it is replay.
  * the rank test is tie-aware: exact U only on untied data, otherwise the asymptotic test WITH the tie correction,
    and every row says which was used (`p_method`). A caller may predeclare a family and get `p_holm` beside it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common
from .common import EXPORT_TABLES, Result, body_str, to_jsonable

EXPORT_SCHEMA = "flyverse.neurome.export/2"
EXPORT_REVISION = 2                                          # revision 2: the Neurome intake corrections (module docstring)
INTERCHANGE_KEY = ("dataset.name", "dataset.release", "bodyId")
NEUROME_TABLES = ("readout_per_body", "contributions", "sensitivity")
RETINA_TABLES = ("retina_columns", "retina_bodies", "retina_radiance", "retina_radiance_blank", "retina_object_track")
# the two references, split in revision 2, plus the deprecated alias that still equals `null_reference_ids`
CONTROL_ID_COLUMNS = ("paired_control_ids", "null_reference_ids", "control_ids")
ID_COLUMNS = ("bodyId", "body_pre", "body_post", "pre_bodies", "post_bodies", "bodies") + CONTROL_ID_COLUMNS
PAIRED_QUANTITIES = ("upstream_drive_mV", "output_Hz")      # LC11 / LC10a get both, per body, never pooled
ARM_B_SUFFIX = "#arm_b"                                      # a paired-control id names the record AND the arm within it
RETINA_MODES = ("geometry_replay", "in_loop_capture", "unknown")

# Physical units of the columns the export writes (the manifest declares them per table).
COLUMN_UNITS = {
    "window_start_s": "s", "window_end_s": "s", "window": "s", "trial_sd": "see `unit`", "stimulus_value": "see `unit`",
    "control_value": "see `unit`", "stimulus_minus_control": "see `unit`", "stimulus_sd": "see `unit`",
    "control_sd": "see `unit`", "null_mean": "see `unit`", "null_sd": "see `unit`", "z_vs_null": "SD of the null arm",
    "value": "see `kind`", "baseline": "see `check`", "delta": "see `check`", "replicate_sd": "see `check`",
    "synaptic_pair_count": "synapses (raw, unsigned, uncapped)", "effective_mv": "mV per presynaptic spike",
    "fanin_scale_post": "dimensionless", "azimuth_deg": "deg", "elevation_deg": "deg", "t_s": "s",
    "radiance_uv": "radiance (model units)", "radiance_b": "radiance (model units)", "radiance_g": "radiance (model units)",
    "radiance_r": "radiance (model units)", "ball_offset_m": "m", "sens_uv": "dimensionless", "sens_b": "dimensionless",
    "sens_g": "dimensionless", "sens_r": "dimensionless", "dir_x": "unit vector (body frame, x forward)",
    "dir_y": "unit vector (body frame, y left)", "dir_z": "unit vector (body frame, z up)",
    "p": "probability (two-sided rank test; see `p_method`)", "p_holm": "probability, Holm-Bonferroni adjusted within `family`",
    "centre_azimuth_deg": "deg (+ left of the pinned heading)", "centre_elevation_deg": "deg (+ above the eye)",
    "angular_diameter_deg": "deg (from the eye, 2 asin(r / distance))", "distance_eye_to_centre_m": "m",
    "min_relative_radiance": "fraction of the matched blank frame", "columns_dimmed_5pct": "columns",
    "columns_dimmed_50pct": "columns", "dimmed_centroid_azimuth_deg": "deg", "dimmed_centroid_elevation_deg": "deg",
}

# One sentence per exported quantity / per-type statistic (manifest `statistic_definitions`). Neurome read revision 1's
# headline drive numbers as membrane voltages; every one of these is a DIFFERENCE of time-means, and the `*_over_cells`
# ones take a maximum over cells WITHIN each run, so the maximising cell may differ from run to run.
_DRIVE = ("the received optic drive of one cell (mV), i.e. the rate lobe's input to that spiking cell, time-averaged over "
          "the analysis window -- not an absolute membrane voltage and not a distance to threshold")
STATISTIC_DEFINITIONS = {
    # ---- readout_per_body quantities (one row per body per quantity; the two arms of ONE recording)
    "upstream_drive_mV": f"per body: {_DRIVE}. `stimulus_value` is the object arm (a) and `control_value` the matched "
                         "blank arm (b) of the SAME recording (`paired_control_ids`); `null_mean` / `null_sd` / "
                         "`z_vs_null` come from the independent blank/blank runs (`null_reference_ids`).",
    "output_Hz": "per body: spikes emitted over the analysis window divided by its length (Hz), object arm minus the "
                 "matched blank arm of the same recording; the null columns come from the independent blank/blank runs.",
    "rate_deviation": "per graded (optic rate) unit: the time-mean signed deviation of its rate from its operating "
                      "point (rate units [0-1]), object arm minus the matched blank arm of the same recording.",
    "abs_rate_deviation": "per graded (optic rate) unit: the time-mean of |rate - operating point| (rate units [0-1]), "
                          "object arm minus the matched blank arm of the same recording.",
    # ---- per_type / size_tuning statistics (one row per type per statistic; arms of INDEPENDENT runs)
    "diff_max_over_cells_mean_mv": "max over the cells of the type, within each run, of that cell's time-mean object-"
                                   "minus-blank received optic drive (mV), compared with the same statistic in "
                                   "independent blank/blank runs. A difference of time-means, never an absolute "
                                   "membrane voltage; the maximising cell may differ from run to run.",
    "diff_mean_over_cells_mean_mv": "mean over the cells of the type of that cell's time-mean object-minus-blank "
                                    "received optic drive (mV), against the same statistic in blank/blank runs.",
    "diff_tuning_peak_mv": "max over (cell, sweep-position bin) of the sweep-locked mean drive in the object arm minus "
                           "the same bin in the blank arm (mV), against the same statistic in blank/blank runs.",
    "diff_peak_100ms_mv": "peak over cells and time of the 100 ms boxcar of the frame-by-frame drive difference (mV). A "
                          "frame-wise subtraction of two stochastic runs: read it only against its own blank/blank null.",
    "diff_rate_hz_max_cell": "max over the cells of the type of (object-arm firing rate - blank-arm firing rate) over "
                             "the window (Hz), against the same statistic in blank/blank runs.",
    "diff_rate_hz_mean": "mean over the cells of the type of (object-arm - blank-arm) firing rate over the window (Hz), "
                         "against the same statistic in blank/blank runs.",
    "diff_abs_mean": "mean over the graded units of the type of the object-minus-blank change in mean |rate deviation| "
                     "(rate units [0-1]), against the same statistic in blank/blank runs.",
    "diff_abs_best_cell_mean": "max over the graded units of the type of the object-minus-blank change in mean |rate "
                               "deviation| (rate units [0-1]), against the same statistic in blank/blank runs.",
    "diff_signed_mean": "mean over the graded units of the type of the object-minus-blank change in signed mean rate "
                        "deviation (rate units [0-1]), against the same statistic in blank/blank runs.",
    "diff_signed_best_cell": "max over the graded units of |object-minus-blank change in signed mean rate deviation| "
                             "(rate units [0-1]). Distinct from `diff_abs_best_cell_mean`: the two give different "
                             "verdicts at 20 deg and must not be collapsed into one statement.",
}

# The manifest's `conventions` block: how to read the files, and -- since revision 2 -- which reference each column
# came from and which field is deprecated.
CONVENTIONS = {
    "missing": "an empty field is the only missing value; read CSV with keep_default_na=False, na_values=['']",
    "not_missing": "'null' is a value of `verdict` (common.compare: the stimulus arm sits inside the null), "
                   "not a missing value; pandas' default NA list would eat it",
    "ids": "bodyId / body_pre / body_post are decimal strings (int64 in the cache), never floats",
    "lists": "a list-valued id column is '|'-joined in one field",
    "roles": "role 'interchange' = the tables of docs/NEUROME_INTERFACE.md section 1; role 'tool' = the "
             "tool's own tables, written and hashed but outside the (dataset, release, bodyId) contract",
    "controls": "TWO references travel, and they are different runs. `control_value` (and `stimulus_minus_control`) "
                f"is the matched blank arm -- arm b, id suffix '{ARM_B_SUFFIX}' -- of the SAME recording that gave "
                "`stimulus_value`: `paired_control_ids`. `null_mean` / `null_sd` / `z_vs_null` / `verdict` come from "
                "INDEPENDENT blank/blank runs of the same protocol: `null_reference_ids`.",
    "control_ids": "DEPRECATED alias of `null_reference_ids`, kept for ONE revision (export schema "
                   f"{EXPORT_SCHEMA}, revision {EXPORT_REVISION}) so revision-1 readers do not break, and equal to it "
                   "row for row. Revision 1 wrote the independent blank/blank run ids here while `control_value` was "
                   "computed from arm b of the stimulus recordings; use `paired_control_ids` / `null_reference_ids` "
                   "and expect this column to be removed in the next revision.",
    "statistics": "every `quantity` (readout_per_body) and `statistic` (per_type / size_tuning) is defined in "
                  "manifest.statistic_definitions; the drive figures are DIFFERENCES of time-means in mV and the "
                  "`max_over_cells` ones take the maximum within each run, so they are never absolute membrane "
                  "voltages and never the tuning curve of one fixed cell",
    "p_method": "`p_method` says which rank test produced `p` in that row: 'exact' (no ties in the pooled sample), "
                "'asymptotic_tie_corrected' (ties present: the normal approximation WITH the tie correction, "
                "continuity-corrected), or 'none' (no test was possible). An exact U on tied data is not computed.",
    "family": "`family` is the multiple-comparison family the CALLER predeclared for that row ('' = none declared) "
              "and `p_holm` the Holm-Bonferroni adjustment of `p` within it. Unadjusted `p` is unchanged; no verdict "
              "in this export rests on `p_holm`.",
    "retina_mode": "manifest.retina.mode is 'geometry_replay' (the presented scene re-rendered at the pinned pose) or "
                   "'in_loop_capture' (the radiance recorded during the rollout itself). A replay is never labelled a "
                   "capture; manifest.retina.replayed says exactly what was re-rendered.",
}


# ------------------------------------------------------------------------------------------------- hashing / tables
def sha256_file(path) -> str:
    """SHA-256 of a file (the manifest's integrity field for every table it lists)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decimal(v) -> str:
    """One interchange key value as a decimal string ('' for a missing one)."""
    if v is None or isinstance(v, (list, tuple, np.ndarray)):
        return "|".join(_decimal(x) for x in np.atleast_1d(v)) if v is not None else ""
    if isinstance(v, str):
        return v
    if isinstance(v, float) and not np.isfinite(v):
        return ""
    return str(int(v))


def _decimal_ids(df: pd.DataFrame) -> pd.DataFrame:
    """bodyId / body_pre / body_post (and the body-list columns) as decimal strings, never floats or ints."""
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = [_decimal(v) for v in df[col]]
    return df


def _order_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """The contract's columns first (common.EXPORT_TABLES), the tool's extra columns after, in their own order.

    The two split reference columns of revision 2 take the place of the contract's `control_ids` (which stays, as the
    deprecated alias), so the three references read together instead of one of them landing among the tool's extras."""
    lead = [c for c in EXPORT_TABLES.get(name, ()) if c in df.columns]
    if "control_ids" in lead:
        lead[lead.index("control_ids"):lead.index("control_ids") + 1] = [c for c in CONTROL_ID_COLUMNS if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]


def _table_units(df: pd.DataFrame) -> dict:
    return {c: COLUMN_UNITS[c] for c in df.columns if c in COLUMN_UNITS}


def write_table(df: pd.DataFrame, out_dir: Path, name: str, parquet_rows: int = 1_000_000, role: str = "interchange") -> dict:
    """Write one table (CSV, Parquet above `parquet_rows`) and return its manifest entry: file, format, role, rows,
    columns, units and SHA-256.

    `role` is `interchange` for the three tables of docs/NEUROME_INTERFACE.md section 1 and the retina record, and
    `tool` for a tool's own table (`per_type`, `reference_per_type`, `delta_links`, ...) -- written so the run
    directory stands on its own, hashed like the rest, but not part of the (dataset, release, bodyId) contract."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "parquet" if len(df) > int(parquet_rows) else "csv"
    path = out_dir / f"{name}.{fmt}"
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return {"name": name, "file": path.name, "format": fmt, "role": role, "rows": int(len(df)),
            "columns": list(df.columns), "units": _table_units(df), "sha256": sha256_file(path)}


# ------------------------------------------------------------------------------------------- which code actually ran
SOURCE_PATTERNS = ("flyverse/*.py", "flyverse/interp/*.py", "flyverse/data/receptors_by_type.csv",
                   "scripts/probe_object_sweep.py", "scripts/interp_export.py")
TEXT_SUFFIXES = (".py", ".csv", ".md", ".txt", ".json", ".toml", ".cfg")
LINE_ENDING_NOTE = ("text files are matched on content, not bytes: a run's file counts as identical when its SHA-256 "
                    "equals this checkout's raw, LF-normalised or CRLF-normalised hash. `scripts/cluster_run.py` "
                    "copies the CLUSTER's own checkout (LF) and overlays only the files that differ from origin/main, "
                    "so a Windows working tree (CRLF) and the job's tree hash differently on identical source.")


def _content_hashes(path) -> set:
    """Every SHA-256 the same content can have on either side of the line-ending divide (one hash for binary)."""
    b = Path(path).read_bytes()
    out = {hashlib.sha256(b).hexdigest()}
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        lf = b.replace(b"\r\n", b"\n")
        out.add(hashlib.sha256(lf).hexdigest())
        out.add(hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest())
    return out


def loaded_sources(root=None, extra=()) -> dict:
    """{relative path: SHA-256} of every module this process has actually imported from under `root` (`sys.modules`),
    plus any `extra` files. This -- not a glob -- is the set of source that can have affected the run: a sibling
    tool's module that was never imported cannot have, and on a shared tree it is edited while a job runs."""
    root = Path(root or common.ROOT).resolve()
    files = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        if p.is_file():
            files[rel] = sha256_file(p)
    for e in extra or ():
        p = Path(e).resolve()
        if p.is_file():
            try:
                files[p.relative_to(root).as_posix()] = sha256_file(p)
            except ValueError:
                files[str(p)] = sha256_file(p)
    return dict(sorted(files.items()))


def source_fingerprint(root=None, patterns=SOURCE_PATTERNS, include_loaded: bool = False, extra=()) -> dict:
    """SHA-256 of every simulation / probe source file present under `root`, plus `common.git_state()`.

    A cluster job runs from an rsynced copy with no `.git`, so `git_state()` there is `commit 'unknown'` and the
    manifest would not name the commit -- the one thing the export must let a skeptic reconstruct. Hashing the files
    the job actually loaded fixes that from the other side: `match_sources` compares those hashes against the local
    checkout, and a full match pins the run to the local commit (the procedure `docs/audits/object_sweep.md` 8.2
    already uses by hand). `files` holds the raw hash of each file (the form 8.2 quotes) and `files_lf` the
    LF-normalised one, so a match is possible whichever side wrote it. With `include_loaded`, `files_loaded` adds
    `loaded_sources` -- the modules this process really imported, which is the set `match_sources` verifies on."""
    root = Path(root or common.ROOT)
    files, lf = {}, {}
    for pat in patterns or ():
        for p in sorted(root.glob(pat)):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                files[rel] = sha256_file(p)
                if p.suffix.lower() in TEXT_SUFFIXES:
                    lf[rel] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    out = {"root": str(root), "git": common.git_state(), "n_files": len(files), "files": files, "files_lf": lf,
           "line_endings": LINE_ENDING_NOTE}
    if include_loaded:
        out["files_loaded"] = loaded_sources(root, extra)
        out["n_loaded"] = len(out["files_loaded"])
    return out


def _match_one(rec: dict, root: Path) -> dict:
    same, differ, missing = [], [], []
    for f, h in (rec or {}).items():
        p = root / f
        if not p.is_file():
            missing.append(f)
        elif h in _content_hashes(p):
            same.append(f)
        else:
            differ.append(f)
    return {"n_recorded": len(rec or {}), "n_identical": len(same), "differ": differ, "missing_here": missing,
            "verified": bool(rec) and not differ and not missing}


def match_sources(recorded: dict, root=None) -> dict:
    """A recorded `source_fingerprint` against this checkout: which files hold the same content, which differ, which
    are absent here, and -- when every one matches -- the local commit the run therefore ran.

    The verdict is taken on `files_loaded` (the modules the run actually imported) when the record has one, and on
    the glob otherwise; the other scope's counts travel alongside, because a file that was never imported cannot
    have changed the numbers. Matching is line-ending-insensitive for text files (LINE_ENDING_NOTE); anything else
    is a real content difference."""
    root = Path(root or common.ROOT)
    rec = recorded or {}
    glob_m = _match_one(dict(rec.get("files") or {}), root)
    load_m = _match_one(dict(rec.get("files_loaded") or {}), root) if rec.get("files_loaded") else None
    chosen = load_m if load_m else glob_m
    scope = "loaded" if load_m else "glob"
    out = {"scope": scope, **chosen, "analysis_git": common.git_state(), "line_endings": LINE_ENDING_NOTE,
           "note": (f"every source file the run imported ({scope}) holds the content this checkout has" if chosen["verified"]
                    else f"the run's {scope} source files are NOT all the content of this checkout; see `differ` / `missing_here`")}
    if load_m:
        out["glob_scope"] = glob_m
    return out


# ------------------------------------------------------------------------------------- tie-aware rank test, Holm
def n_tied_values(stim, null) -> int:
    """How many observations of the pooled two-arm sample repeat a value already present (0 = no ties at all).

    The exact Mann-Whitney null distribution assumes no ties: it enumerates rank assignments, and tied observations
    have no unique rank. `common.compare` asks scipy for `method='exact'` whenever the two arms hold <= 40 runs, which
    is every arm this toolkit produces -- so on tied data it returns an exact p for a distribution the data do not
    follow (25 of the 288 ladder statistics; Neurome's intake, 'Corrected interpretation of the numbers')."""
    a = np.asarray([v for v in np.atleast_1d(stim) if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in np.atleast_1d(null) if np.isfinite(v)], dtype=float)
    pooled = np.concatenate([a, b]) if (len(a) or len(b)) else np.zeros(0)
    return int(len(pooled) - len(np.unique(pooled))) if len(pooled) else 0


def mann_whitney(stim, null, *, exact_max_n: int = 40) -> dict:
    """Two-sided Mann-Whitney U and p by the method the data allow, with the method named.

    `p_method`: `'exact'` when the pooled sample has NO ties and the arms are small enough to enumerate;
    `'asymptotic_tie_corrected'` when there are ties (scipy's normal approximation applies the tie correction to the
    variance and a continuity correction) or the arms are too large to enumerate; `'none'` when no test is possible.
    Returns {'U', 'p', 'p_method', 'n_tied_values'}."""
    from scipy import stats
    a = [float(v) for v in np.atleast_1d(stim) if np.isfinite(v)]
    b = [float(v) for v in np.atleast_1d(null) if np.isfinite(v)]
    ties = n_tied_values(a, b)
    out = {"U": float("nan"), "p": float("nan"), "p_method": "none", "n_tied_values": ties}
    if not a or not b or (len(a) + len(b)) < 3:
        return out
    method = "exact" if (ties == 0 and (len(a) + len(b)) <= int(exact_max_n)) else "asymptotic"
    try:
        r = stats.mannwhitneyu(a, b, alternative="two-sided", method=method)
    except ValueError:
        return out
    out.update(U=float(r.statistic), p=float(r.pvalue),
               p_method="exact" if method == "exact" else "asymptotic_tie_corrected")
    return out


def _verdict_from(cmp: dict, *, z_min: float, min_n: int, alpha: float) -> str:
    """`common.compare`'s verdict rule re-applied to a comparison whose `p` has been recomputed (docstring of
    `common.compare`, in its order): underpowered -> undetermined -> result -> null. Tested against `common.compare`
    itself on untied data, where the tie-aware p IS the exact p and the two must agree row for row."""
    na, nb = int(cmp["stim"]["n"]), int(cmp["null"]["n"])
    floor, z, p, diff = cmp["p_floor"], cmp["z"], cmp["p"], cmp["diff"]
    min_n = max(int(min_n), common.CALL_REPLICATES)          # the call rule: >= 4 runs per arm, whatever the other arm has
    if min(na, nb) < min_n or (np.isfinite(floor) and floor > alpha):
        return "underpowered"
    if cmp.get("null_sd_zero") and diff != 0 and not (np.isfinite(p) and p > alpha):
        return "undetermined"
    if np.isfinite(z) and abs(z) >= z_min and (not np.isfinite(p) or p <= alpha):
        return "result"
    return "null"


def compare_tie_aware(stim, null, *, z_min: float = common.Z_RESULT, min_n: int = common.CALL_REPLICATES,
                      alpha: float = 0.05) -> dict:
    """`common.compare` with the rank test taken by `mann_whitney` -- exact only on untied data -- and the verdict
    re-derived from the p that resulted. Adds `p_method` and `n_tied_values`; every other field is `compare`'s own."""
    cmp = dict(common.compare(stim, null, z_min=z_min, min_n=min_n, alpha=alpha))
    mw = mann_whitney(stim, null)
    cmp.update(U=mw["U"], p=mw["p"], p_method=mw["p_method"], n_tied_values=mw["n_tied_values"])
    cmp["verdict"] = _verdict_from(cmp, z_min=z_min, min_n=min_n, alpha=alpha)
    return cmp


# A family is a PREDECLARATION by the caller, never a default: `where` selects the rows, `by` (optional) splits them
# into one family per group of those columns, `name` labels it. Nothing here changes an unadjusted p or any verdict.
FAMILY_SPECS = {
    "lc_drive": {"name": "LC drive comparisons (LC11 / LC10a x the size ladder)",
                 "where": {"type": ["LC11", "LC10a"], "statistic": ["diff_max_over_cells_mean_mv"]}},
}


def family_labels(df: pd.DataFrame, spec=None) -> np.ndarray:
    """One family label per row of `df` ('' = the row is in no declared family).

    `spec`: None (nothing declared -- the default), a key of `FAMILY_SPECS`, one dict
    `{'name': str, 'where': {column: [values]}, 'by': [column, ...]}`, or a list of such (first match wins)."""
    n = len(df)
    out = np.array([""] * n, dtype=object)
    if spec is None:
        return out
    specs = [spec] if isinstance(spec, (str, dict)) else list(spec)
    for s in specs:
        s = FAMILY_SPECS[s] if isinstance(s, str) else dict(s)
        m = np.ones(n, bool)
        for col, vals in (s.get("where") or {}).items():
            if col not in df.columns:
                m[:] = False
                break
            m &= df[col].astype(str).isin([str(v) for v in np.atleast_1d(vals)]).to_numpy()
        name = str(s.get("name", "family"))
        by = [c for c in (s.get("by") or []) if c in df.columns]
        for i in np.flatnonzero(m & (out == "")):
            out[i] = name + ("" if not by else " | " + " ".join(str(df[c].iloc[i]) for c in by))
    return out


def holm(p, family=None) -> np.ndarray:
    """Holm-Bonferroni adjusted p WITHIN each family label; NaN for a row in no family or without a finite p.

    Holm: sort the family's m p-values ascending, multiply the k-th (0-based) by (m - k), then take the running
    maximum so the adjusted values stay monotone, and cap at 1. A post-hoc sensitivity calculation, not a pass rule:
    eight LC drive comparisons at p 0.0079365 come out at 0.0635 (Neurome's intake)."""
    p = np.asarray([float(v) if v is not None else np.nan for v in np.atleast_1d(p)], dtype=float)
    fam = np.array([""] * len(p), dtype=object) if family is None else np.asarray(family, dtype=object)
    out = np.full(len(p), np.nan)
    for label in {f for f in fam if str(f)}:
        idx = np.flatnonzero(np.array([str(f) == str(label) for f in fam]) & np.isfinite(p))
        if not len(idx):
            continue
        order = idx[np.argsort(p[idx], kind="stable")]
        m = len(order)
        adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
        out[order] = adj
    return out


def add_family_columns(df: pd.DataFrame, spec=None, *, p_col: str = "p") -> pd.DataFrame:
    """`family` and `p_holm` on a table that already carries `p` (both empty / NaN when nothing was declared)."""
    fam = family_labels(df, spec)
    df = df.copy()
    df["family"] = [str(f) for f in fam]
    df["p_holm"] = holm(df[p_col].to_numpy() if p_col in df.columns else np.full(len(df), np.nan), fam)
    return df


# ------------------------------------------------------------------------------------------------- the retina record
EYE_ABOVE_TABLE_M = 0.0012      # the probe's pinned pose: eye z 0.7512 over a table top at 0.750035 m (the record logs)


def retina_mode_of(sampling: str, *, in_loop: bool = False) -> dict:
    """`retina.mode` from what the record itself says it did -- never a guess in the optimistic direction.

    `'in_loop_capture'` only when the caller asserts it (the radiance was read out of the rollout that produced the
    spikes); `'geometry_replay'` when the record's own `sampling` string says it replayed the presented geometry;
    `'unknown'` otherwise, quoting the string verbatim. Revision 1 shipped a replay with no such field, and Neurome's
    intake had to infer it from prose ('Do not silently relabel replayed input as recorded input')."""
    s = str(sampling or "")
    if in_loop:
        mode, what = "in_loop_capture", "the radiance the optic lobe read during the rollout that produced the spikes"
    elif "replay" in s.lower():
        mode = "geometry_replay"
        what = ("the presented scene geometry re-rendered frame by frame at the pinned pose, AFTER the rollout, by the "
                "same deterministic ray tracer the rollout used -- the stimulus side is byte-identical across runs "
                "while the spiking side is not, so this is the radiance the optic lobe received, but it is a replay "
                "and no in-loop capture was recorded to test that equivalence against")
    else:
        mode, what = "unknown", "the record does not say how its radiance was obtained"
    return {"mode": mode, "replayed": what, "sampling_recorded": s,
            "in_loop": bool(in_loop), "modes": list(RETINA_MODES)}


def object_track(offset_m, *, ball_radius_m, ahead_m, eye_above_table_m: float = EYE_ABOVE_TABLE_M) -> pd.DataFrame:
    """Where the ball is, from the eye, at each frame: azimuth, centre elevation, angular diameter and distance.

    The probe pins the fly and slides a ball of radius r along the eye's `left` axis at a fixed distance `ahead`,
    resting on the table, so the centre sits `r - eye_above_table` above the eye whatever its size -- which is why
    elevation and size move together along the ladder (0.88 deg at 4.5 deg, 13.71 deg at 30 deg: Neurome's intake,
    'The retinal comparison is not yet a controlled size-tuning assay'). Angular diameter is 2 asin(r / distance) from
    the eye, not the 2 atan(r / ahead) the probe's flags use."""
    s = np.asarray(offset_m, dtype=float)
    r, ahead = float(ball_radius_m), float(ahead_m)
    dz = r - float(eye_above_table_m)
    horiz = np.sqrt(ahead ** 2 + s ** 2)
    dist = np.sqrt(horiz ** 2 + dz ** 2)
    return pd.DataFrame({"ball_offset_m": s,
                         "centre_azimuth_deg": np.degrees(np.arctan2(s, ahead)),
                         "centre_elevation_deg": np.degrees(np.arctan2(dz, horiz)),
                         "angular_diameter_deg": 2 * np.degrees(np.arcsin(np.clip(r / np.maximum(dist, 1e-12), 0, 1))),
                         "distance_eye_to_centre_m": dist})


def _radiance_long(z: dict, n_col: int) -> pd.DataFrame:
    """The (frames, columns, 4) radiance array as the long table the manifest's row count refers to."""
    rad = np.asarray(z["radiance"], dtype=np.float32)
    n_f = rad.shape[0]
    t_s = np.asarray(z.get("t_s", np.arange(n_f) * common.FRAME_MS / 1000.0), dtype=np.float64)
    df = pd.DataFrame({"frame": np.repeat(np.arange(n_f), n_col), "t_s": np.repeat(t_s, n_col),
                       "column_id": np.tile(np.arange(n_col), n_f),
                       "radiance_uv": rad[:, :, 0].ravel(), "radiance_b": rad[:, :, 1].ravel(),
                       "radiance_g": rad[:, :, 2].ravel(), "radiance_r": rad[:, :, 3].ravel()})
    if "ball_offset_m" in z:
        df["ball_offset_m"] = np.repeat(np.asarray(z["ball_offset_m"], dtype=np.float64), n_col)
    return df


def _as_npz(x):
    """A path to an npz, a dict or an npz-like -> {name: array} (None stays None), plus the source path."""
    if x is None:
        return None, None
    src = str(x) if isinstance(x, (str, Path)) else None
    if src is not None:
        x = dict(np.load(x, allow_pickle=False))
    return {k: np.asarray(v) for k, v in dict(x).items() if not np.isscalar(v)}, src


def retina_tables(retina, out_dir, parquet_rows: int = 1_000_000, *, blank=None, in_loop: bool = False,
                  geometry=None, pose=None) -> tuple[list, dict]:
    """The retinal sampling actually presented -> the retina tables + the manifest's `retina` block.

    `retina`: a path to the npz `scripts/interp_export.py record --retina` writes, or a dict / npz-like with
    `radiance` (n_frames, n_columns, 4 = [UV, B, G, R]), `t_s`, optionally `ball_offset_m`, and the column / body map
    `col_side`, `col_hex`, `col_az_el`, `col_dir`, `pr_body`, `pr_column`, `pr_sens`, `pr_type`, `pr_index`
    (flyverse/retina.py's Retina: 1,466 columns, 5,895 photoreceptors). Radiance is written long
    (frame x column), which is the form the row counts and the SHA-256 in the manifest refer to.

    `blank`: the MATCHED blank arm's record in the same form. Its radiance is written as `retina_radiance_blank` with
    the identical schema, so the object-minus-blank footprint is reconstructible from the run directory alone --
    revision 1 shipped only the object arm and Neurome could only rebuild it because the blank npz happened to exist
    in this checkout. `in_loop`: assert an in-loop capture (see `retina_mode_of`; the shipped protocol is a replay).
    `geometry`: `{'ball_radius_m', 'ahead_m'[, 'eye_above_table_m']}` -- with it, and with `ball_offset_m` in the
    record, `retina_object_track` gives the ball's azimuth / centre elevation / angular diameter per frame (and, with
    the blank, the columns it dims), so size and retinal position are both on file. `pose`: the pinned pose the
    sampling was taken at, which travels beside the mode (`export` fills it from the stimulus block).
    """
    z, src = _as_npz(retina)
    if z is None:                       # no retinal record: no mode to declare, and `verify` asks for one only with radiance
        return [], {"file": None, "n_columns": None, "column_to_bodies": None, "mode": None, "replayed": None}
    zb, src_b = _as_npz(blank)
    out_dir = Path(out_dir)
    metas = []
    pr_body = z["pr_body"].astype(np.int64)
    pr_col = z["pr_column"].astype(np.int64)
    n_col = int(z["col_dir"].shape[0])
    # columns: direction, azimuth / elevation and the photoreceptor bodies that feed each column
    order = np.argsort(pr_col, kind="stable")
    bodies_by_col = {k: [] for k in range(n_col)}
    for i in order:
        bodies_by_col[int(pr_col[i])].append(str(int(pr_body[i])))
    cols = pd.DataFrame({"column_id": np.arange(n_col),
                         "side": z["col_side"].astype(str) if "col_side" in z else ["?"] * n_col,
                         "hex1": z["col_hex"][:, 0] if "col_hex" in z else -1,
                         "hex2": z["col_hex"][:, 1] if "col_hex" in z else -1,
                         "azimuth_deg": z["col_az_el"][:, 0] if "col_az_el" in z else np.nan,
                         "elevation_deg": z["col_az_el"][:, 1] if "col_az_el" in z else np.nan,
                         "dir_x": z["col_dir"][:, 0], "dir_y": z["col_dir"][:, 1], "dir_z": z["col_dir"][:, 2],
                         "n_photoreceptors": [len(bodies_by_col[k]) for k in range(n_col)],
                         "photoreceptor_bodies": ["|".join(bodies_by_col[k]) for k in range(n_col)]})
    metas.append(write_table(cols, out_dir, "retina_columns", parquet_rows))
    bodies = pd.DataFrame({"bodyId": body_str(pr_body), "model_index": z.get("pr_index", np.full(len(pr_body), -1)).astype(np.int64),
                           "type": z["pr_type"].astype(str) if "pr_type" in z else "",
                           "unit_kind": "photoreceptor", "column_id": pr_col})
    if "pr_sens" in z:
        for j, ch in enumerate(("uv", "b", "g", "r")):
            bodies[f"sens_{ch}"] = z["pr_sens"][:, j]
    metas.append(write_table(bodies, out_dir, "retina_bodies", parquet_rows))
    sampling = str(z.get("sampling", "per frame of the probe window, at the pinned pose"))
    block = {"n_columns": n_col, "n_photoreceptors": int(len(pr_body)), "channels": ["UV", "B", "G", "R"],
             "column_to_bodies": "retina_columns.csv:photoreceptor_bodies",
             "source_npz": src, "frame_ms": common.FRAME_MS, "sampling": sampling,
             **retina_mode_of(sampling, in_loop=in_loop), "pinned_pose": to_jsonable(pose) if pose else None,
             "files": {"columns": "retina_columns.csv", "bodies": "retina_bodies.csv"}}
    if "radiance" in z:
        n_f = int(np.asarray(z["radiance"]).shape[0])
        meta = write_table(_radiance_long(z, n_col), out_dir, "retina_radiance", parquet_rows)
        metas.append(meta)
        block.update({"n_frames": n_f, "file": meta["file"], "shape": [n_f, n_col, 4], "arm": "object (condition a)",
                      "files": dict(block["files"], radiance=meta["file"])})
    # the matched blank arm, same schema, same replay: the footprint travels with the run directory
    if zb is not None and "radiance" in zb:
        nb = int(np.asarray(zb["radiance"]).shape[0])
        if int(np.asarray(zb["radiance"]).shape[1]) != n_col:
            raise ValueError(f"the blank retina record has {np.asarray(zb['radiance']).shape[1]} columns, "
                             f"the object record {n_col}: they are not the same sampling")
        meta_b = write_table(_radiance_long(zb, n_col), out_dir, "retina_radiance_blank", parquet_rows)
        metas.append(meta_b)
        rb = np.asarray(zb["radiance"], dtype=np.float32).sum(-1)
        block["blank"] = {"file": meta_b["file"], "source_npz": src_b, "n_frames": nb, "arm": "blank (condition b)",
                          "sampling_recorded": str(zb.get("sampling", "")),
                          "identical_over_frames": bool(np.allclose(rb, rb[0][None])),
                          "note": "the matched blank arm of the same protocol, replayed the same way; the same schema "
                                  "as retina_radiance, so object / blank is a join on (frame, column_id)"}
        block["files"] = dict(block["files"], radiance_blank=meta_b["file"])
    elif "radiance" in z:
        block["blank"] = {"file": None, "note": "no matched blank radiance was supplied to this export; the "
                                                "object-minus-blank footprint cannot be rebuilt from this directory alone"}
    # the ball itself, frame by frame, in the eye's own coordinates
    if geometry and "radiance" in z and "ball_offset_m" in z:
        g = dict(geometry)
        track = object_track(np.asarray(z["ball_offset_m"], dtype=float), ball_radius_m=g["ball_radius_m"],
                             ahead_m=g["ahead_m"], eye_above_table_m=g.get("eye_above_table_m", EYE_ABOVE_TABLE_M))
        n_f = int(np.asarray(z["radiance"]).shape[0])
        track.insert(0, "frame", np.arange(n_f))
        track.insert(1, "t_s", np.asarray(z.get("t_s", np.arange(n_f) * common.FRAME_MS / 1000.0), dtype=np.float64))
        if zb is not None and "radiance" in zb:               # what the ball actually did to the columns, per frame
            ra = np.asarray(z["radiance"], np.float64).sum(-1)
            rb0 = np.asarray(zb["radiance"], np.float64).sum(-1)[0][None]
            rel = ra / np.maximum(rb0, 1e-9) - 1.0
            hit = rel < -0.05
            az_el = np.asarray(z["col_az_el"], float) if "col_az_el" in z else np.full((n_col, 2), np.nan)
            w = np.where(hit, -rel, 0.0)
            tot = w.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                track["columns_dimmed_5pct"] = hit.sum(1)
                track["columns_dimmed_50pct"] = (rel < -0.5).sum(1)
                track["min_relative_radiance"] = 1.0 + rel.min(1)
                track["dimmed_centroid_azimuth_deg"] = np.where(tot > 0, w @ az_el[:, 0] / np.maximum(tot, 1e-12), np.nan)
                track["dimmed_centroid_elevation_deg"] = np.where(tot > 0, w @ az_el[:, 1] / np.maximum(tot, 1e-12), np.nan)
        meta_t = write_table(track, out_dir, "retina_object_track", parquet_rows)
        metas.append(meta_t)
        block["object_track"] = {
            "file": meta_t["file"], "geometry": to_jsonable({**g, "eye_above_table_m": g.get("eye_above_table_m", EYE_ABOVE_TABLE_M)}),
            "definition": "per frame: the ball's offset along the eye's left axis, its azimuth atan2(offset, ahead), "
                          "its centre elevation atan2(r - eye_above_table, hypot(ahead, offset)) and its angular "
                          "diameter 2 asin(r / distance) from the eye. The ball rests on the table, so centre "
                          "elevation rises with radius: size and retinal position change together along the ladder.",
            "empirical_columns": ("the dimmed-column counts and centroid are computed from retina_radiance against "
                                  "retina_radiance_blank, so the geometric track can be checked against the replay"
                                  if "columns_dimmed_5pct" in track.columns else None)}
    return metas, block


# ------------------------------------------------------------------------------------------------- the export itself
def _stamp_ids(df: pd.DataFrame, column: str, ids) -> pd.DataFrame:
    """`column` = the '|'-joined `ids` on every row that does not already carry its own value."""
    if ids is None:
        return df
    joined = "|".join(str(x) for x in np.atleast_1d(ids))
    if column not in df.columns:
        df[column] = joined
    else:
        df[column] = [joined if (v is None or v == "" or (isinstance(v, float) and not np.isfinite(v))) else v
                      for v in df[column]]
    return df


def _statistic_definitions(res) -> dict:
    """The `statistic_definitions` this Result needs: every value its tables actually use under `quantity` /
    `statistic`, defined. A name with no definition is listed with an explicit 'not defined in this revision' so the
    gap is visible in the manifest rather than silently absent."""
    names = []
    for rows in res.tables.values():
        for key in ("quantity", "statistic"):
            names += [str(r.get(key)) for r in rows if isinstance(r, dict) and r.get(key)]
    out = {}
    for n in sorted(set(names)):
        out[n] = STATISTIC_DEFINITIONS.get(n, "not defined in this revision of the export "
                                              "(flyverse/interp/export.py::STATISTIC_DEFINITIONS)")
    return out


def _dataset_columns(df, prov):
    """Carry each endpoint's namespace across biological/synthetic graph boundaries."""
    ds = prov.get("dataset_release", {})
    default = (ds.get("name", common.DATASET_NAME), ds.get("release", common.DATASET_RELEASE))
    synthetic = {}
    fp = prov.get("compiled_connectome", {})
    while fp.get("extension"):
        ext = fp["extension"]
        synthetic.update({str(i): ("synthetic", release) for i, release in zip(ext["body_ids"], ext["releases"])})
        fp = ext.get("base", {})
    def identity(ids):
        return [synthetic.get(str(i), default) for i in ids]
    if "bodyId" in df:
        namespaces = identity(df.bodyId)
    elif "body_post" in df:
        namespaces = identity(df.body_post)
    else:
        namespaces = [default] * len(df)
    for pos, (key, values) in enumerate((("dataset", [v[0] for v in namespaces]), ("release", [v[1] for v in namespaces]))):
        if key in df:
            df[key] = df[key].fillna(pd.Series(values, index=df.index))
        else:
            df.insert(pos, key, values)
    if synthetic:
        for endpoint in ("pre", "post"):
            if "body_" + endpoint in df:
                ns = identity(df["body_" + endpoint])
                df["dataset_" + endpoint] = [v[0] for v in ns]
                df["release_" + endpoint] = [v[1] for v in ns]
    return df


def export(result, *, out_root="out/export", run_id=None, retina=None, parquet_rows: int = 1_000_000,
           control_ids=None, paired_control_ids=None, null_reference_ids=None, retina_blank=None,
           retina_in_loop: bool = False, retina_geometry=None) -> Path:
    """Serialize a `common.Result` into a Neurome probe-export run directory and return its path.

    `result`: a Result or the path of a Result JSON. `out_root` / `run_id`: the run directory is
    `<out_root>/<run_id>` (`run_id` defaults to the Result's own). `retina`: the retinal sampling record (a path to
    the npz `scripts/interp_export.py record --retina` writes, or a dict; see `retina_tables`), `retina_blank` the
    matched blank arm's record, `retina_geometry` the ball geometry for `retina_object_track`, `retina_in_loop` the
    assertion that the radiance was captured in the loop (it is a replay in the shipped protocol). Tables above
    `parquet_rows` rows are written as Parquet instead of CSV.

    The two references (revision 2): `paired_control_ids` names the record / arm each row's `control_value` was
    computed from (arm b of the SAME recording, id suffix `#arm_b`), `null_reference_ids` the INDEPENDENT blank/blank
    runs behind `null_mean` / `z_vs_null` / `verdict`. `control_ids` is the deprecated alias of the second and is
    always written equal to it; a caller that passes only `control_ids` (the revision-1 spelling) is taken to mean
    `null_reference_ids`, which is what that argument has always held.

    Refuses (ValueError) a Result whose `Result.check()` is non-empty -- a missing provenance block, a table lacking
    the contract's columns, or an `execution.device` that is not the realised device. Every table is listed in
    manifest.json with its row count, columns, units and SHA-256, and the source Result is copied in beside it, so a
    reader can reconstruct which commit, cache fingerprint, LIFParams and stimulus produced the numbers from the
    manifest alone.

    Tables carry a `role`: `interchange` for the three of docs/NEUROME_INTERFACE.md section 1 and the retina record
    (the (dataset, release, bodyId) contract), `tool` for the tool's own tables (`per_type`, `reference_per_type`,
    `delta_links`, ...), which are written and hashed as well so the run directory does not depend on a reader
    parsing result.json.
    """
    res = result if isinstance(result, Result) else Result.load(result)
    problems = res.check()
    if problems:
        raise ValueError(f"the export refuses this Result ({res.tool} {res.run_id}): " + "; ".join(problems))
    run_id = str(run_id or res.run_id)
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = res.provenance
    ds = prov.get("dataset_release", {})
    if null_reference_ids is None and control_ids is not None:
        null_reference_ids = control_ids                       # the revision-1 spelling of the independent-null ids
    tables = []
    for name in NEUROME_TABLES:
        rows = res.tables.get(name)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if name == "readout_per_body":
            df = _stamp_ids(_stamp_ids(df, "paired_control_ids", paired_control_ids),
                            "null_reference_ids", null_reference_ids)
            if "null_reference_ids" not in df.columns and "control_ids" in df.columns:
                df["null_reference_ids"] = df["control_ids"]   # a revision-1 Result: its control_ids WERE the nulls
            if "null_reference_ids" in df.columns:
                df["control_ids"] = df["null_reference_ids"]   # the deprecated alias, equal row for row
        df = _order_columns(name, _decimal_ids(df))
        df = _dataset_columns(df, prov)
        tables.append(write_table(df, out_dir, name, parquet_rows))
    sp = (prov.get("stimulus") or {}).get("params") or {}
    pose = {"pos_m": sp.get("pos"), "heading_rad": sp.get("heading_rad"), "frame_ms": common.FRAME_MS,
            "pinned": "the fly is placed at this pose every frame and does not move (the probe pins it); the only "
                      "thing that moves in the scene is the object"} if sp.get("pos") is not None else None
    retina_meta, retina_block = retina_tables(retina if retina is not None else prov.get("retina", {}).get("source_npz"),
                                              out_dir, parquet_rows, blank=retina_blank, in_loop=retina_in_loop,
                                              geometry=retina_geometry, pose=pose)
    tables += retina_meta
    for name, rows in res.tables.items():                 # the tool's own tables, so the run directory stands alone
        if name in NEUROME_TABLES or name in RETINA_TABLES or not rows:
            continue                                       # (a tool table may not shadow a retina file)
        tables.append(write_table(_decimal_ids(pd.DataFrame(rows)), out_dir, name, parquet_rows, role="tool"))
    if isinstance(prov.get("retina"), dict):
        retina_block = {**prov["retina"], **retina_block}
    result_path = out_dir / "result.json"
    res.save(result_path)
    manifest = {
        "schema": EXPORT_SCHEMA, "run_id": run_id, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": res.tool, "tool_version": res.tool_version, "result_run_id": res.run_id,
        "dataset": {"name": ds.get("name", common.DATASET_NAME), "release": ds.get("release", common.DATASET_RELEASE)},
        "interchange_key": list(INTERCHANGE_KEY),
        "export_revision": EXPORT_REVISION,
        "revision_note": "revision 2: paired vs independent controls split (`paired_control_ids` / "
                         "`null_reference_ids`, `control_ids` deprecated), `statistic_definitions`, `retina.mode` "
                         "with the matched blank radiance and the object track, and a tie-aware rank test "
                         "(`p_method`) with an optional predeclared family (`family`, `p_holm`)",
        "conventions": dict(CONVENTIONS),
        "statistic_definitions": _statistic_definitions(res),
        "flyverse_commit": prov["flyverse_commit"], "dataset_release": ds, "compiled_connectome": prov["compiled_connectome"],
        "model": prov["model"], "execution": prov["execution"], "stimulus": prov["stimulus"], "retina": retina_block,
        "units": prov["units"], "populations": res.populations, "replicates": res.replicates,
        "summary": to_jsonable(res.summary), "validation": to_jsonable(res.validation), "files": to_jsonable(res.files),
        "source_result": {"file": result_path.name, "schema": res.schema, "sha256": sha256_file(result_path),
                          "tables": {k: len(v) for k, v in res.tables.items()}},
        "tables": tables,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest), f, indent=1)
    return out_dir


def read_table(run_dir, name: str) -> pd.DataFrame:
    """One table of a finished export, by name (CSV or Parquet, whichever the manifest lists).

    Only an empty field is missing: `keep_default_na=False, na_values=[""]`. pandas' default NA list contains the
    literal `null` (and `NA`, `None`, `nan`, `N/A`, ...), and `verdict` -- `common.compare`'s three-valued answer --
    takes the value `null` for 'the stimulus arm sits inside the null'. Read with the defaults, 23,794 of the
    26,482 `readout_per_body` rows of the object-sweep export come back as NaN verdicts, i.e. a verdict is silently
    turned into a missing value. Readers outside this module must do the same (docs/audits/interp_export.md)."""
    run_dir = Path(run_dir)
    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    for t in man["tables"]:
        if t["name"] == name:
            p = run_dir / t["file"]
            return pd.read_parquet(p) if t["format"] == "parquet" else pd.read_csv(
                p, dtype={c: str for c in ID_COLUMNS}, keep_default_na=False, na_values=[""])
    raise KeyError(f"{run_dir}: no table {name!r} (has {[t['name'] for t in man['tables']]})")


def verify(run_dir, *, neurons=None, expect_paired=("LC11", "LC10a"), expect_counts=None) -> dict:
    """Re-read a finished export and check it: every table's SHA-256 matches the file, every id column holds decimal
    strings, every bodyId exists in `neurons` (a path to cache/neurons.parquet, a DataFrame or an array of ids; the
    cache is used when neurons is True), every type in `expect_paired` has both `upstream_drive_mV` and `output_Hz`
    for each of its bodies, `expect_counts` = {type: n_bodies} holds, and the manifest carries every mandatory
    provenance field. Returns {'problems': [...], ...counts}; an empty `problems` is a clean export."""
    run_dir = Path(run_dir)
    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    problems, info = [], {"run_id": man.get("run_id"), "tables": {}}
    for field in ("run_id", "flyverse_commit", "dataset_release", "compiled_connectome", "model", "execution",
                  "stimulus", "retina", "units", "tables"):
        if field not in man:
            problems.append(f"manifest lacks {field}")
    if man.get("execution", {}).get("device") in (None, "", "None"):
        problems.append("manifest.execution.device is not the realised device")
    if not man.get("dataset_release", {}).get("files"):
        problems.append("manifest.dataset_release lists no MaleCNS files")
    for t in man.get("tables", []):
        p = run_dir / t["file"]
        if not p.exists():
            problems.append(f"{t['name']}: {t['file']} missing")
            continue
        if sha256_file(p) != t["sha256"]:
            problems.append(f"{t['name']}: SHA-256 mismatch")
        df = read_table(run_dir, t["name"])
        info["tables"][t["name"]] = int(len(df))
        if len(df) != t["rows"]:
            problems.append(f"{t['name']}: {len(df)} rows, manifest says {t['rows']}")
        for col in ("bodyId", "body_pre", "body_post"):
            if col in df.columns:
                bad = [v for v in df[col].astype(str).head(200000) if v and not v.lstrip("-").isdigit()]
                if bad:
                    problems.append(f"{t['name']}.{col}: {len(bad)} values are not decimal ids (e.g. {bad[0]!r})")
    if neurons is not None and neurons is not False:
        if neurons is True:
            neurons = Path(common.ROOT) / "cache" / "neurons.parquet"
        if isinstance(neurons, (str, Path)):
            ids = pd.read_parquet(neurons, columns=["bodyId"]).bodyId.to_numpy()
        elif isinstance(neurons, pd.DataFrame):
            ids = neurons.bodyId.to_numpy()
        else:
            ids = np.asarray(neurons)
        known = set(str(int(b)) for b in ids)
        info["n_known_bodies"] = len(known)
        for t in man.get("tables", []):
            if t["name"] in ("readout_per_body", "contributions", "retina_bodies"):
                df = read_table(run_dir, t["name"])
                for col in ("bodyId", "body_pre", "body_post"):
                    if col in df.columns:
                        miss = sorted({v for v in df[col].astype(str) if v and v not in known})
                        if miss:
                            problems.append(f"{t['name']}.{col}: {len(miss)} ids not in the connectome (e.g. {miss[:3]})")
    names = [t["name"] for t in man.get("tables", [])]
    if "readout_per_body" in names:
        df = read_table(run_dir, "readout_per_body")
        info["readout_types"] = int(df.type.nunique()) if "type" in df.columns else 0
        for ty in expect_paired or ():
            sub = df[df.type == ty] if "type" in df.columns else df.iloc[:0]
            if not len(sub):
                problems.append(f"readout_per_body: no rows for {ty}")
                continue
            per = sub.groupby("bodyId").quantity.apply(lambda s: set(s))
            info[f"{ty}_bodies"] = int(len(per))
            missing = [b for b, q in per.items() if not set(PAIRED_QUANTITIES) <= q]
            if missing:
                problems.append(f"readout_per_body: {len(missing)} {ty} bodies lack both of {PAIRED_QUANTITIES}")
        for ty, n in (expect_counts or {}).items():
            got = int(df[df.type == ty].bodyId.nunique()) if "type" in df.columns else 0
            if got != int(n):
                problems.append(f"readout_per_body: {ty} has {got} bodies, expected {n}")
        problems += _check_control_ids(df, man)
    if any(t.get("name") == "retina_radiance" for t in man.get("tables", [])):
        if man.get("retina", {}).get("mode") not in RETINA_MODES:
            problems.append("manifest.retina has radiance but no mode in "
                            f"{RETINA_MODES} (a replay must not pass as a capture)")
    info["problems"] = problems
    return info


def _check_control_ids(df: pd.DataFrame, man: dict) -> list:
    """The revision-2 rule on the two references: the deprecated `control_ids` must be exactly `null_reference_ids`,
    the paired and the independent ids must not be the same runs, and the manifest must document the alias. This is
    the defect Neurome's intake found -- one column naming the blank/blank runs while `control_value` came from arm b
    of the stimulus recordings -- so the export checks it rather than trusting the writer."""
    def col(name):                                             # '' is the missing value; a str dtype reads it back as NA
        return df[name].fillna("").astype(str)

    out = []
    has_null, has_paired = "null_reference_ids" in df.columns, "paired_control_ids" in df.columns
    if "control_ids" in df.columns:
        if not has_null:
            out.append("readout_per_body: control_ids without null_reference_ids (the alias must name what it aliases)")
        elif not (col("control_ids") == col("null_reference_ids")).all():
            out.append("readout_per_body: control_ids is not equal to null_reference_ids (its documented alias)")
        if not (man.get("conventions") or {}).get("control_ids"):
            out.append("manifest.conventions does not document the deprecated control_ids alias")
    if has_paired and has_null:
        a, b = col("paired_control_ids"), col("null_reference_ids")
        live = (a != "") & (b != "")
        if live.any() and (a[live] == b[live]).all():
            out.append("readout_per_body: paired_control_ids and null_reference_ids name the same runs; "
                       "the paired control is arm b of the stimulus recordings, the null reference is the "
                       "independent blank/blank runs (docs/audits/interp_export.md revision 2)")
    return out


# ------------------------------------------------------------------------------------------------- adapters
def raw_counts(c) -> tuple:
    """Raw synapse counts per stored entry (post x pre, unsigned, uncapped), sign-0 entries included --
    `common.raw_counts`.

    The merge this function used to perform privately (|W.data| with the explicit zeros filled from
    `connectome.sign0_counts`, because the shared accessor substituted that array for the whole count vector and
    every ordinary edge came out as 0) is `common.raw_counts`'s own now -- with the same
    `maximum(|W.data|, sign0_counts)` rule. docs/INTERP.md 11, defect 1, closed. Returns (csr, sign0_available)."""
    return common.raw_counts(c)


def silent_flags(c, pre_idx, frozen_idx=None, rates=None) -> pd.DataFrame:
    """`common.silent_flags`, whose `never_firing` is False (not evaluated) when no rollout is given.

    This wrapper used to patch NaN -> False, because `common.links` read `bool(NaN)` as True and marked every
    structural row 'never_firing'; the shared function returns a boolean column now (docs/INTERP.md 11, defect 2,
    closed). Kept as the export's named accessor; the table's own `silent_rule` says whether rates were supplied."""
    return common.silent_flags(c, pre_idx, frozen_idx=frozen_idx, rates=rates)


def links_to_contributions(links: pd.DataFrame, *, kind: str = "effective_weight_mV", normalisation: str = "",
                           reference_graph: str = "", window=None) -> pd.DataFrame:
    """A `common.links` edge table -> the Neurome `contributions` columns (docs/NEUROME_INTERFACE.md section 1).

    Pure field mapping: `effective_mv` becomes `value` with `kind`, the edge's `sign_rule` and `gain_rule` travel
    unchanged, `normalisation` names the input normalisation in force, `reference_graph` is the cache fingerprint the
    weights were read from, and `window` is the analysis window (None for a structural table)."""
    df = pd.DataFrame({"body_pre": links["body_pre"].astype(str), "body_post": links["body_post"].astype(str),
                       "pre_type": links["pre_type"], "post_type": links["post_type"],
                       "value": links["effective_mv"].astype(float), "kind": kind,
                       "sign_rule": links["sign_rule"], "gain_rule": links["gain_rule"],
                       "normalisation": normalisation, "reference_graph": reference_graph,
                       "window": [list(window) if window is not None else None] * len(links),
                       "synaptic_pair_count": links["synaptic_pair_count"].astype(float)})
    for extra in ("pre_nt", "sign", "silent", "fanin_scale_post", "pre_index", "post_index"):
        if extra in links.columns:
            df[extra] = links[extra].to_numpy()
    return df


def static_decompose_result(c, target, pre=None, *, params=None, receptor=None, cache_dir=None, nonzero_only=False,
                            label=None, frozen=None, rates=None) -> Result:
    """The structural (static) decomposition of `target`'s input as a Result -- the minimal form the export
    validation needs: `common.effective_weights` + `common.links` mapped onto the Neurome `contributions` columns
    through `links_to_contributions`, plus a per-type summary (mV per post cell per presynaptic volley, the raw
    synapse count and its share of the target's raw input -- docs/audits/cx_glno.md section 1).

    `flyverse/interp/decompose.py` is the tool that owns this analysis and supersedes this function; it is here
    because the export must be able to demonstrate the `contributions` path end to end on a CPU, with or without
    that module. `pre` defaults to every presynaptic cell with a stored entry onto the target.
    """
    from .. import brain
    p = params or brain.LIFParams()
    ew = common.effective_weights(c, p, receptor)
    post = common.population(c, target, label=label or "target")
    if pre is None:
        W = c.W.tocsr()[post.idx]
        pre_idx = np.unique(W.indices)
    else:
        pre_idx = common.resolve(c, pre)
    counts, sign0_ok = raw_counts(c)
    lk = common.links(c, ew, pre_idx, post.idx, receptor=receptor, counts=counts,
                      flags=silent_flags(c, pre_idx, frozen_idx=frozen), nonzero_only=nonzero_only)
    lk["silent_rule"] = ("sign0 | frozen (an optic rate unit) | pruned; never_firing not evaluated (no rollout)"
                         if rates is None else "sign0 | frozen | pruned | never_firing (max rate over the rollout)")
    fp = common.connectome_fingerprint(c, cache_dir)
    prov = common.provenance(c, lif=p, fb=None, device="cpu", cache_dir=cache_dir,
                             stimulus={"protocol": "static", "params": {"target": common.spec_repr(target)}, "control": None})
    prov["execution"]["device"] = "cpu"                       # structural: no rollout, the CPU is the realised device
    res = Result.new("decompose", prov)
    res.add_population(post, unit_kind=None)
    contrib = links_to_contributions(lk, kind="effective_weight_mV",
                                     normalisation=f"input_norm ref {p.input_norm_ref} alpha {p.input_norm_alpha}",
                                     reference_graph=fp["md5"], window=None)
    res.add_table("contributions", contrib)
    raw_total = lk.groupby("post_type").synaptic_pair_count.sum().to_dict()
    per = (lk.groupby(["post_type", "pre_type"])
             .agg(value=("effective_mv", "sum"), n_entries=("effective_mv", "size"),
                  raw_count=("synaptic_pair_count", "sum"), n_pre=("pre_index", "nunique"), n_post=("post_index", "nunique"))
             .reset_index())
    n_post_by_type = {t: len(ix) for t, ix in common.by_type(c, post.idx).items()}
    per["value"] = [v / max(n_post_by_type.get(t, 1), 1) for v, t in zip(per.value, per.post_type)]
    per["kind"] = "effective_weight_mV_per_post_per_volley"
    per["share_of_raw_input"] = [rc / max(raw_total.get(t, 1.0), 1.0) for rc, t in zip(per.raw_count, per.post_type)]
    per["silent"] = [("sign0" if v == 0 else "") for v in per.value]
    res.add_table("per_type", per.sort_values(["post_type", "value"], key=lambda s: s.abs() if s.name == "value" else s,
                                              ascending=[True, False]))
    res.summary = {"n_pre_cells": int(len(pre_idx)), "n_post_cells": int(post.n), "n_entries": int(len(lk)),
                   "raw_synapses": float(lk.synaptic_pair_count.sum()), "sign0_counts_available": bool(sign0_ok),
                   "total_positive_mv": float(lk.effective_mv[lk.effective_mv > 0].sum()),
                   "total_negative_mv": float(lk.effective_mv[lk.effective_mv < 0].sum()),
                   "sign0_entries": int((lk.effective_mv == 0).sum()),
                   "sign0_raw_synapses": float(lk.synaptic_pair_count[lk.effective_mv == 0].sum())}
    res.validation = dict(res.validation, measured={"static": True}, status="not run")
    return res


# ------------------------------------------------------------------------------------------------- object-sweep adapter
SWEEP_STATS = ("diff_max_over_cells_mean_mv", "diff_mean_over_cells_mean_mv", "diff_tuning_peak_mv",
               "diff_peak_100ms_mv", "diff_rate_hz_max_cell", "diff_rate_hz_mean", "diff_abs_mean",
               "diff_abs_best_cell_mean", "diff_signed_mean", "diff_signed_best_cell")
CELL_QUANTITIES = {"drive_mean": ("upstream_drive_mV", "mV", "spiking"),
                   "rate_hz": ("output_Hz", "Hz", "spiking"),
                   "dev_mean": ("rate_deviation", "rate units [0-1]", "graded"),
                   "dev_absmean": ("abs_rate_deviation", "rate units [0-1]", "graded")}


def _load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _arm_values(jsons: list, stat: str, type_: str) -> list:
    out = []
    for d in jsons:
        v = d.get("ball", {}).get(type_, {}).get(stat)
        if v is not None and np.isfinite(v):
            out.append(float(v))
    return out


def _cells_frame(npz_paths, window_s=None) -> pd.DataFrame:
    """The per-cell arrays of `scripts/interp_export.py record` -> one long frame
    (run, type, bodyId, model_index, unit_kind, quantity, stimulus_value, control_value)."""
    rows = []
    for run, path in enumerate(npz_paths):
        z = np.load(path, allow_pickle=False)
        types = [str(t) for t in z["types"]]
        for t in types:
            bodies = z[f"body__{t}"]
            idx = z[f"index__{t}"]
            kinds = [str(x) for x in z[f"unitkind__{t}"]] if f"unitkind__{t}" in z.files else None
            for key, (quantity, unit, default_kind) in CELL_QUANTITIES.items():
                ka, kb = f"{key}__{t}__a", f"{key}__{t}__b"
                if ka not in z.files or kb not in z.files:
                    continue
                a, b = np.asarray(z[ka], float), np.asarray(z[kb], float)
                rows.append(pd.DataFrame({"run": run, "file": str(path), "type": t, "bodyId": body_str(bodies),
                                          "model_index": idx.astype(np.int64),
                                          "unit_kind": kinds if kinds is not None else default_kind,
                                          "quantity": quantity, "unit": unit,
                                          "stimulus_value": a, "control_value": b}))
    if not rows:
        return pd.DataFrame(columns=["run", "type", "bodyId", "model_index", "unit_kind", "quantity", "unit",
                                     "stimulus_value", "control_value"])
    df = pd.concat(rows, ignore_index=True)
    df["stimulus_minus_control"] = df.stimulus_value - df.control_value
    return df


def _arm_table(stim: list, null: list, *, paired_control_ids="", null_reference_ids="", family=None) -> pd.DataFrame:
    """One per-type arm table from loaded probe JSONs: for every type and every statistic of SWEEP_STATS the stimulus
    arm, the null arm and the comparison's z / Welch / U / p / verdict, with the per-run values kept. The same
    function builds the reproduction and the reference arm, so the two tables are directly comparable row by row.

    The rank test is `compare_tie_aware`'s (exact only on untied data) and every row says which it used (`p_method`,
    `n_tied_values`); `statistic_definition` spells out what the number is, and the two reference id columns say which
    runs each arm came from -- a `diff_*` value is already object-minus-blank WITHIN each stimulus recording
    (`paired_control_ids`, arm b), and the comparator arm is the independent blank/blank runs (`null_reference_ids`).
    `family`, when the caller predeclares one, adds `family` / `p_holm` (see `add_family_columns`)."""
    head = (stim[0] if stim else null[0])["ball"]
    rows = []
    for t in head:
        for stat in SWEEP_STATS:
            a, b = _arm_values(stim, stat, t), _arm_values(null, stat, t)
            if not a and not b:
                continue
            cmp = compare_tie_aware(a, b) if b else {"z": float("nan"), "welch": float("nan"), "U": float("nan"),
                                                     "p": float("nan"), "verdict": "underpowered", "p_method": "none",
                                                     "n_tied_values": n_tied_values(a, b)}
            rows.append({"type": t, "statistic": stat, "kind": head[t]["kind"], "n_cells": head[t]["n_cells"],
                         "stim_n": len(a), "stim_mean": float(np.mean(a)) if a else np.nan,
                         "stim_sd": float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
                         "null_n": len(b), "null_mean": float(np.mean(b)) if b else np.nan,
                         "null_sd": float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
                         "z": cmp["z"], "welch": cmp["welch"], "U": cmp["U"], "p": cmp["p"],
                         "p_method": cmp["p_method"], "n_tied_values": cmp["n_tied_values"], "verdict": cmp["verdict"],
                         "stim_values": a, "null_values": b,
                         "statistic_definition": STATISTIC_DEFINITIONS.get(stat, ""),
                         "paired_control_ids": paired_control_ids, "null_reference_ids": null_reference_ids})
    return add_family_columns(pd.DataFrame(rows), family)


def _as_list(x) -> list:
    """One path, a list of paths, or None -> a list of strings."""
    if x is None:
        return []
    return [str(x)] if isinstance(x, (str, Path)) else [str(p) for p in x]


def _pool(df: pd.DataFrame, window) -> pd.DataFrame:
    """Per (type, body, quantity) over runs: means, the trial SD of the difference and the run count."""
    g = df.groupby(["type", "bodyId", "model_index", "unit_kind", "quantity", "unit"], sort=False)
    out = g.agg(stimulus_value=("stimulus_value", "mean"), control_value=("control_value", "mean"),
                stimulus_minus_control=("stimulus_minus_control", "mean"),
                stimulus_sd=("stimulus_value", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
                control_sd=("control_value", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
                trial_sd=("stimulus_minus_control", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
                n_trials=("stimulus_minus_control", "size")).reset_index()
    out["window_start_s"] = float(window[0]); out["window_end_s"] = float(window[1])
    return out


def _run_stem(p) -> str:
    """The record's own id from any of its files: `out/.../d045_stim_s0_cells.npz` -> `d045_stim_s0`."""
    stem = Path(str(p)).stem
    for suffix in ("_cells", "_prov", "_retina"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def result_from_object_sweep(stim_jsons, null_jsons=(), *, cells=(), null_cells=(), provenance=None, reference=None,
                             reference_null=None, retina=None, control_ids=None, paired_control_ids=None,
                             null_reference_ids=None, family=None, run_id=None, generator=None,
                             label="object_sweep") -> Result:
    """The round-3 object sweep (`scripts/probe_object_sweep.py`) as an exportable Result.

    `stim_jsons` / `null_jsons`: the probe JSONs of the ball-vs-none runs and of the none-vs-none null runs (>= 3
    each for a verdict that is not 'underpowered'; `common.compare` decides). `cells` / `null_cells`: the matching
    per-cell npz files that `scripts/interp_export.py record` captures from the same protocol -- they carry the
    per-body numbers the probe JSON pools away, and give two `readout_per_body` rows per spiking body
    (`upstream_drive_mV` = the window-mean optic drive, `output_Hz` = spikes over the window) plus one per graded
    optic rate unit. `provenance`: the provenance block a record job wrote (its REALISED device); without one the
    Result will not pass `Result.check()` and the export refuses it -- deliberately. `reference` / `reference_null`:
    an earlier run of the same protocol -- one path or a list (`out/r3obj/ball_off_s*.json` and
    `out/r3obj/null_off_s*.json`, the five round-3 seeds of `docs/audits/object_sweep.md` 8.4). The reference arm is
    reduced exactly as the reproduction is (the same `_arm_table`), so `tables['reference_per_type']` and
    `tables['per_type']` are comparable row by row, and `validation.measured.reproduction` holds the two arms side by
    side on the primary statistic with both files' SHA-256 in `stimulus.reference_runs`.

    The two references (revision 2): `paired_control_ids` defaults to the stimulus records' own ids with the
    `#arm_b` suffix -- every `control_value` and every `diff_*` statistic is object minus the blank arm of the SAME
    recording -- and `null_reference_ids` to the independent blank/blank records, which is what `null_mean` /
    `z_vs_null` / the per-type comparison arm are. `control_ids` is accepted as the revision-1 spelling of
    `null_reference_ids`. `family`: a multiple-comparison family predeclared by the caller (`FAMILY_SPECS`, a dict, or
    None), which adds `family` / `p_holm` to the per-type tables.

    Tables: 'per_type' (type, statistic, the stimulus and null arms, z / Welch / U / p / p_method / verdict),
    'readout_per_body', 'reference_per_type'. Summary: the LC11 / LC10a / LPLC2 statistics of object_sweep.md 8.4.
    """
    stim = [_load_json(p) for p in stim_jsons]
    null = [_load_json(p) for p in null_jsons]
    cfg = stim[0]["config"] if stim else (null[0]["config"] if null else {})
    window = (float(cfg.get("settle", 0.0)), float(cfg.get("settle", 0.0)) + float(cfg.get("seconds", 0.0)))
    prov = dict(to_jsonable(provenance)) if isinstance(provenance, dict) else (      # a copy: never edit the caller's
        _load_json(provenance) if provenance is not None else {})
    prov = prov.get("provenance", prov)
    stimulus = dict(prov.get("stimulus") or {})
    stimulus.setdefault("protocol", "object_sweep")
    stimulus["params"] = {**cfg, **(stimulus.get("params") or {})}
    stimulus["control"] = {"condition": "none (ball parked out of the scene)", "null": "none vs none",
                           "n_stim_runs": len(stim), "n_null_runs": len(null)}
    stimulus["runs"] = {"stimulus": [str(p) for p in stim_jsons], "null": [str(p) for p in null_jsons]}
    ref_paths, ref_null_paths = _as_list(reference), _as_list(reference_null)
    if ref_paths:
        stimulus["reference_runs"] = [{"file": p, "arm": "stimulus", "sha256": sha256_file(p)} for p in ref_paths] + \
                                     [{"file": p, "arm": "null", "sha256": sha256_file(p)} for p in ref_null_paths]
    prov["stimulus"] = stimulus
    if retina is not None:
        prov["retina"] = {**(prov.get("retina") or {}), "source_npz": str(retina)}
    # ---- the two references, named apart (revision 2: Neurome's control-labelling defect)
    if null_reference_ids is None:
        null_reference_ids = control_ids if control_ids is not None else [
            _run_stem(p) for p in (null_cells or null_jsons)]
    if paired_control_ids is None:
        paired_control_ids = [f"{_run_stem(p)}{ARM_B_SUFFIX}" for p in (cells or stim_jsons)]
    paired_ids = "|".join(str(x) for x in np.atleast_1d(paired_control_ids))
    null_ids = "|".join(str(x) for x in np.atleast_1d(null_reference_ids))
    stimulus["controls"] = {"paired_control_ids": paired_ids, "null_reference_ids": null_ids,
                            "paired_control": "arm b (the matched blank) of each stimulus recording -- the source of "
                                              "`control_value` and of every `diff_*` per-type statistic",
                            "null_reference": "independent blank/blank runs of the same protocol -- the source of "
                                              "`null_mean` / `null_sd` / `z_vs_null` and of the per-type comparison arm",
                            "deprecated_control_ids": "an alias of null_reference_ids (CONVENTIONS['control_ids'])"}
    prov["stimulus"] = stimulus
    res = Result.new("export", prov)
    if run_id:
        res.run_id = str(run_id)
    # ---- per-type arms
    res.add_table("per_type", _arm_table(stim, null, paired_control_ids=paired_ids, null_reference_ids=null_ids,
                                         family=family))
    # ---- per-body readouts
    if cells:
        df = _cells_frame(list(cells))
        pooled = _pool(df, window)
        if null_cells:
            nd = _pool(_cells_frame(list(null_cells)), window)[["type", "bodyId", "quantity", "stimulus_minus_control", "trial_sd", "n_trials"]]
            nd = nd.rename(columns={"stimulus_minus_control": "null_mean", "trial_sd": "null_sd", "n_trials": "n_null"})
            pooled = pooled.merge(nd, on=["type", "bodyId", "quantity"], how="left")
            with np.errstate(invalid="ignore", divide="ignore"):
                pooled["z_vs_null"] = (pooled.stimulus_minus_control - pooled.null_mean) / pooled.null_sd
            pooled["verdict"] = np.where(pooled.n_trials < common.CALL_REPLICATES, "underpowered",
                                         np.where(np.abs(pooled.z_vs_null) >= common.Z_RESULT, "result", "null"))
        pooled["paired_control_ids"] = paired_ids
        pooled["null_reference_ids"] = null_ids
        pooled["control_ids"] = null_ids                       # the deprecated alias, equal to null_reference_ids
        pooled["run_files"] = "|".join(Path(str(p)).name for p in cells)
        # `quantity` is defined in manifest.statistic_definitions rather than repeated on all 26,482 rows
        res.add_table("readout_per_body", pooled)
        res.replicates = {"n": len(cells), "unit": "runs",
                          "runs": [{"run_index": i, "file": str(p)} for i, p in enumerate(cells)],
                          "null": {"n": len(null_cells), "runs": [str(p) for p in null_cells]}}
    else:
        res.replicates = {"n": len(stim), "unit": "runs", "runs": [{"run_index": i, "file": str(p)} for i, p in enumerate(stim_jsons)],
                          "null": {"n": len(null), "runs": [str(p) for p in null_jsons]}}
    # ---- the earlier runs, side by side (the same reduction applied to both arms)
    measured = {}
    if ref_paths:
        ref_tab = _arm_table([_load_json(p) for p in ref_paths], [_load_json(p) for p in ref_null_paths],
                             paired_control_ids="|".join(f"{_run_stem(p)}{ARM_B_SUFFIX}" for p in ref_paths),
                             null_reference_ids="|".join(_run_stem(p) for p in ref_null_paths), family=family)
        ref_tab.insert(0, "source", "reference")
        res.add_table("reference_per_type", ref_tab)
        per, primary = res.table("per_type"), "diff_max_over_cells_mean_mv"
        for t in per.type.unique() if len(per) else []:
            r = per[(per.type == t) & (per.statistic == primary)]
            q = ref_tab[(ref_tab.type == t) & (ref_tab.statistic == primary)]
            if not len(r) or not len(q):
                continue
            measured[t] = {"statistic": primary, "reference_files": ref_paths, "reference_null_files": ref_null_paths,
                           "reference_runs": list(q.stim_values.iloc[0]), "reference_mean": float(q.stim_mean.iloc[0]),
                           "reference_sd": float(q.stim_sd.iloc[0]), "reference_null_mean": float(q.null_mean.iloc[0]),
                           "reference_null_sd": float(q.null_sd.iloc[0]), "reference_z": float(q.z.iloc[0]),
                           "reference_verdict": q.verdict.iloc[0],
                           "reproduced_runs": list(r.stim_values.iloc[0]), "reproduced_mean": float(r.stim_mean.iloc[0]),
                           "reproduced_sd": float(r.stim_sd.iloc[0]), "null_mean": float(r.null_mean.iloc[0]),
                           "null_sd": float(r.null_sd.iloc[0]), "z": float(r.z.iloc[0]), "verdict": r.verdict.iloc[0]}
    res.summary = {"protocol": "object_sweep", "mode": cfg.get("mode"), "angular_diameter_deg": cfg.get("angular_diameter_deg"),
                   "window_s": list(window), "n_stim_runs": len(stim), "n_null_runs": len(null),
                   "n_bodies": int(res.table("readout_per_body").bodyId.nunique()) if res.tables.get("readout_per_body") else 0}
    res.validation = dict(res.validation, measured={"round_trip": None, "reproduction": measured},
                          status="not run" if not measured else "measured")
    res.files = {"generator": generator or "scripts/interp_export.py record + run",
                 "probe_jsons": [str(p) for p in stim_jsons] + [str(p) for p in null_jsons],
                 "cells": [str(p) for p in cells] + [str(p) for p in null_cells],
                 "reference": ref_paths + ref_null_paths,
                 "retina": str(retina) if retina is not None else None}
    return res


def round_trip_check(res: Result, run_dir) -> dict:
    """Every value of the Result's Neurome tables found again in the written files (the export's own validation,
    `common.VALIDATION['export']`): the numbers within `max_abs_diff`, the ids as decimal strings, and the text
    columns character for character -- the last is what catches a value a CSV reader turns into something else
    (`verdict = 'null'` read back as NaN). Returns {table: {'rows', 'rows_written', 'max_abs_diff',
    'n_numeric_columns', 'ids_match', 'text_columns_match', 'mismatched_text_columns'}}."""
    out = {}
    for name in NEUROME_TABLES:
        rows = res.tables.get(name)
        if not rows:
            continue
        src = pd.DataFrame(rows)
        got = read_table(run_dir, name)
        rec = {"rows": int(len(src)), "rows_written": int(len(got))}
        num = [c for c in src.columns if c in got.columns and pd.api.types.is_numeric_dtype(pd.Series(src[c]))]
        worst = 0.0
        for c in num:
            a = pd.to_numeric(src[c], errors="coerce").to_numpy(float)
            b = pd.to_numeric(got[c], errors="coerce").to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                worst = max(worst, float(np.max(np.abs(a[m] - b[m]))))
        rec["max_abs_diff"] = worst
        rec["n_numeric_columns"] = len(num)
        bad = []
        for c in src.columns:
            if c in num or c not in got.columns or c in ID_COLUMNS:
                continue
            a = ["" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in src[c]]
            b = ["" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in got[c]]
            if a != b:
                bad.append(c)
        rec["text_columns_match"] = not bad
        rec["mismatched_text_columns"] = bad
        for col in ("bodyId", "body_pre"):
            if col in src.columns and col in got.columns:
                rec["ids_match"] = bool(list(map(_decimal, src[col])) == list(got[col].astype(str)))
        out[name] = rec
    return out
