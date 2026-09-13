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
    retina_radiance.csv    the radiance actually presented, per column per frame, [UV, B, G, R]  (Parquet above
                           `parquet_rows`, as every table is)
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

EXPORT_SCHEMA = "flyverse.neurome.export/1"
INTERCHANGE_KEY = ("dataset.name", "dataset.release", "bodyId")
NEUROME_TABLES = ("readout_per_body", "contributions", "sensitivity")
RETINA_TABLES = ("retina_columns", "retina_bodies", "retina_radiance")
ID_COLUMNS = ("bodyId", "body_pre", "body_post", "pre_bodies", "post_bodies", "bodies", "control_ids")
PAIRED_QUANTITIES = ("upstream_drive_mV", "output_Hz")      # LC11 / LC10a get both, per body, never pooled

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
    """The contract's columns first (common.EXPORT_TABLES), the tool's extra columns after, in their own order."""
    lead = [c for c in EXPORT_TABLES.get(name, ()) if c in df.columns]
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


# ------------------------------------------------------------------------------------------------- the retina record
def retina_tables(retina, out_dir, parquet_rows: int = 1_000_000) -> tuple[list, dict]:
    """The retinal sampling actually presented -> three tables + the manifest's `retina` block.

    `retina`: a path to the npz `scripts/interp_export.py record --retina` writes, or a dict / npz-like with
    `radiance` (n_frames, n_columns, 4 = [UV, B, G, R]), `t_s`, optionally `ball_offset_m`, and the column / body map
    `col_side`, `col_hex`, `col_az_el`, `col_dir`, `pr_body`, `pr_column`, `pr_sens`, `pr_type`, `pr_index`
    (flyverse/retina.py's Retina: 1,466 columns, 5,895 photoreceptors). Radiance is written long
    (frame x column), which is the form the row counts and the SHA-256 in the manifest refer to.
    """
    if retina is None:
        return [], {"file": None, "n_columns": None, "column_to_bodies": None}
    src = None
    if isinstance(retina, (str, Path)):
        src = str(retina)
        retina = dict(np.load(retina, allow_pickle=False))
    z = {k: np.asarray(v) for k, v in dict(retina).items() if not np.isscalar(v)}
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
    block = {"n_columns": n_col, "n_photoreceptors": int(len(pr_body)), "channels": ["UV", "B", "G", "R"],
             "column_to_bodies": "retina_columns.csv:photoreceptor_bodies",
             "source_npz": src, "frame_ms": common.FRAME_MS,
             "sampling": str(z.get("sampling", "per frame of the probe window, at the pinned pose")),
             "files": {"columns": "retina_columns.csv", "bodies": "retina_bodies.csv"}}
    if "radiance" in z:
        rad = np.asarray(z["radiance"], dtype=np.float32)
        n_f = rad.shape[0]
        t_s = np.asarray(z.get("t_s", np.arange(n_f) * common.FRAME_MS / 1000.0), dtype=np.float64)
        frame = np.repeat(np.arange(n_f), n_col)
        df = pd.DataFrame({"frame": frame, "t_s": np.repeat(t_s, n_col), "column_id": np.tile(np.arange(n_col), n_f),
                           "radiance_uv": rad[:, :, 0].ravel(), "radiance_b": rad[:, :, 1].ravel(),
                           "radiance_g": rad[:, :, 2].ravel(), "radiance_r": rad[:, :, 3].ravel()})
        if "ball_offset_m" in z:
            df["ball_offset_m"] = np.repeat(np.asarray(z["ball_offset_m"], dtype=np.float64), n_col)
        meta = write_table(df, out_dir, "retina_radiance", parquet_rows)
        metas.append(meta)
        block.update({"n_frames": int(n_f), "file": meta["file"], "shape": [int(n_f), n_col, 4],
                      "files": dict(block["files"], radiance=meta["file"])})
    return metas, block


# ------------------------------------------------------------------------------------------------- the export itself
def export(result, *, out_root="out/export", run_id=None, retina=None, parquet_rows: int = 1_000_000,
           control_ids=None) -> Path:
    """Serialize a `common.Result` into a Neurome probe-export run directory and return its path.

    `result`: a Result or the path of a Result JSON. `out_root` / `run_id`: the run directory is
    `<out_root>/<run_id>` (`run_id` defaults to the Result's own). `retina`: the retinal sampling record (a path to
    the npz `scripts/interp_export.py record --retina` writes, or a dict; see `retina_tables`). Tables above
    `parquet_rows` rows are written as Parquet instead of CSV. `control_ids`: the matched control run ids to stamp
    on every `readout_per_body` row that does not carry its own.

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
    tables = []
    for name in NEUROME_TABLES:
        rows = res.tables.get(name)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if name == "readout_per_body" and control_ids is not None:
            ids = "|".join(str(x) for x in np.atleast_1d(control_ids))
            if "control_ids" not in df.columns:
                df["control_ids"] = ids
            else:
                df["control_ids"] = [ids if (v is None or v == "" or (isinstance(v, float) and not np.isfinite(v))) else v
                                     for v in df["control_ids"]]
        df = _order_columns(name, _decimal_ids(df))
        df.insert(0, "release", ds.get("release", common.DATASET_RELEASE))
        df.insert(0, "dataset", ds.get("name", common.DATASET_NAME))
        tables.append(write_table(df, out_dir, name, parquet_rows))
    retina_meta, retina_block = retina_tables(retina if retina is not None else prov.get("retina", {}).get("source_npz"),
                                              out_dir, parquet_rows)
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
        "conventions": {
            "missing": "an empty field is the only missing value; read CSV with keep_default_na=False, na_values=['']",
            "not_missing": "'null' is a value of `verdict` (common.compare: the stimulus arm sits inside the null), "
                           "not a missing value; pandas' default NA list would eat it",
            "ids": "bodyId / body_pre / body_post are decimal strings (int64 in the cache), never floats",
            "lists": "a list-valued id column is '|'-joined in one field",
            "roles": "role 'interchange' = the tables of docs/NEUROME_INTERFACE.md section 1; role 'tool' = the "
                     "tool's own tables, written and hashed but outside the (dataset, release, bodyId) contract"},
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
    info["problems"] = problems
    return info


# ------------------------------------------------------------------------------------------------- adapters
def raw_counts(c) -> tuple:
    """Raw synapse counts per stored entry (post x pre, unsigned, uncapped), sign-0 entries included.

    `common.raw_counts` REPLACES the whole count vector with `connectome.sign0_counts`, which is "non-zero only where
    W.data == 0" (flyverse/connectome.py:308) -- so every ordinary edge comes out as 0 and only the sign-0 entries
    carry a count. The export needs the real `synaptic_pair_count` (Neurome joins on it), so it takes |W.data| and
    fills the explicit zeros from sign0_counts instead. Returns (csr, sign0_available); see the report in
    docs/audits/interp_export.md."""
    from .. import connectome as cn
    C = c.W.tocsr().copy()
    C.data = np.abs(C.data).astype(np.float32)
    ok = False
    try:
        s0 = cn.sign0_counts(c)
        if s0 is not None and len(np.asarray(s0)) == C.nnz:
            C.data = np.maximum(C.data, np.asarray(s0, dtype=np.float32))
            ok = True
    except Exception:  # noqa: BLE001 -- the raw weights table is not on every machine
        pass
    return C, ok


def silent_flags(c, pre_idx, frozen_idx=None, rates=None) -> pd.DataFrame:
    """`common.silent_flags` with `never_firing` left False instead of NaN when no rollout is given.

    `common.links` builds its `silent` string with `bool(flag) is True`, and `bool(float('nan'))` is True, so a
    structural table built without rates marks every entry 'never_firing'. Here the column is False (not evaluated)
    unless `rates` are supplied; the table's own `silent_rule` says which."""
    f = common.silent_flags(c, pre_idx, frozen_idx=frozen_idx, rates=rates)
    if rates is None:
        f["never_firing"] = False
    return f


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


def _arm_table(stim: list, null: list) -> pd.DataFrame:
    """One per-type arm table from loaded probe JSONs: for every type and every statistic of SWEEP_STATS the stimulus
    arm, the null arm and `common.compare`'s z / Welch / U / p / verdict, with the per-run values kept. The same
    function builds the reproduction and the reference arm, so the two tables are directly comparable row by row."""
    head = (stim[0] if stim else null[0])["ball"]
    rows = []
    for t in head:
        for stat in SWEEP_STATS:
            a, b = _arm_values(stim, stat, t), _arm_values(null, stat, t)
            if not a and not b:
                continue
            cmp = common.compare(a, b) if b else {"z": float("nan"), "welch": float("nan"), "U": float("nan"),
                                                  "p": float("nan"), "verdict": "underpowered"}
            rows.append({"type": t, "statistic": stat, "kind": head[t]["kind"], "n_cells": head[t]["n_cells"],
                         "stim_n": len(a), "stim_mean": float(np.mean(a)) if a else np.nan,
                         "stim_sd": float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
                         "null_n": len(b), "null_mean": float(np.mean(b)) if b else np.nan,
                         "null_sd": float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
                         "z": cmp["z"], "welch": cmp["welch"], "U": cmp["U"], "p": cmp["p"], "verdict": cmp["verdict"],
                         "stim_values": a, "null_values": b})
    return pd.DataFrame(rows)


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


def result_from_object_sweep(stim_jsons, null_jsons=(), *, cells=(), null_cells=(), provenance=None, reference=None,
                             reference_null=None, retina=None, control_ids=None, run_id=None, generator=None,
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

    Tables: 'per_type' (type, statistic, the stimulus and null arms, compare's z / Welch / U / p / verdict),
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
    res = Result.new("export", prov)
    if run_id:
        res.run_id = str(run_id)
    # ---- per-type arms
    res.add_table("per_type", _arm_table(stim, null))
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
            pooled["verdict"] = np.where(pooled.n_trials < common.MIN_REPLICATES, "underpowered",
                                         np.where(np.abs(pooled.z_vs_null) >= common.Z_RESULT, "result", "null"))
        ids = control_ids if control_ids is not None else [Path(str(p)).stem for p in cells]
        pooled["control_ids"] = "|".join(str(x) for x in np.atleast_1d(ids))
        pooled["run_files"] = "|".join(Path(str(p)).name for p in cells)
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
        ref_tab = _arm_table([_load_json(p) for p in ref_paths], [_load_json(p) for p in ref_null_paths])
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
