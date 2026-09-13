"""The expectation ledger: score every probe's per-population output against a curated table (docs/INTERP.md 4.7).

One question -- *does the model's measured response to a named stimulus match what the literature and this project's
own audits say it should be?* -- asked once, of every file the project has already produced, with the citation and the
replicate scatter attached to each answer. Nothing here simulates, tunes or changes the model: the ledger reads
finished outputs (`flyverse.interp.result/1` JSONs, `scripts/benchmark.py` JSONs, `probe_object_sweep.py` /
`probe_compass_room.py` / `batch_sustain.py` / `probe_figure_stages.py` JSONs, `r5_attr_taste_cpu.py`'s arm x seed
table, the loom / bitter console logs and the rotation / figure-ground CSVs) and the shipped table
`flyverse/data/expected_responses.csv`, and writes PASS / FAIL / MISSING per (row, arm).

Three things it is deliberately NOT:

* it is not a fitting target. A row is an expectation with a citation, not a knob; the project rule ("the full plain
  model is not hand-tuned toward behaviour") means a FAIL is a localization task for `trace` / `lesion` / `paths`.
* it is not a second opinion on the battery. Where a row carries `check_key`, the ledger's status must equal the
  status `scripts/benchmark.py` wrote for the same key on the same JSON; disagreement is a bug in one of the two and
  the Result reports it (`summary.battery_disagree`). A stored JSON keeps the criterion that was in force when it
  ran, so a status difference the FILE's own criterion explains is reported separately
  (`summary.battery_criterion_mismatch`) -- a stale file, not a scoring bug.
* it is not a place to hide a bad bound. `walk.power_max_hz` carries op `report`: dynamics round 1 showed the 50 Hz
  bound is unfit (it fails under 13 of 16 optic ablations and anti-correlates with room take-offs), so the ledger
  records the number and never turns it into a verdict.

Layout
------
`load_table()`             the CSV (see COLUMNS) with the ops validated.
`read_sources(paths)`      every supported file -> a flat `Observation` frame (population, stimulus, quantity, value,
                           arm, run, source) plus the benchmark `checks` and a per-source provenance row.
`ledger(results, ...)`     match, pool replicates per arm, score, compare against the auto-detected null arm,
                           and return a `common.Result` (tables `ledger`, `sources`, `observations`).

`validate()`               reproduce VALIDATION['ledger'] and return (Result, the side-by-side frame).

The stochastic half is handled the way `docs/audits/object_sweep.md` 8.4 does it: sources that declare themselves a
none-vs-none null (`config.null == true` in a `probe_object_sweep.py` JSON, the `CB` block of a stage JSON, or
anything passed as `null=`) form the null arm, and every row whose quantity is a difference statistic additionally
reports `common.compare`'s dict -- z against the null SD, Welch, exact Mann-Whitney U and p, and the verdict, which
is `underpowered` below three runs per arm whatever the numbers. Rows scored from a single run carry `n = 1` and say so.
"""
from __future__ import annotations

import dataclasses
import glob as _glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import common
from .common import MIN_REPLICATES, ROOT, Result, Z_RESULT

TOOL = "ledger"
TABLE_PATH = ROOT / "flyverse" / "data" / "expected_responses.csv"

#: Columns of flyverse/data/expected_responses.csv. `population`, `stimulus`, `quantity`, `expected`, `op`, `unit`,
#: `source`, `model_reference` and `notes` are the contract (docs/INTERP.md 4.7); `row_id`, `label`, `bound`, `gap`
#: and `check_key` are this implementation's additions (see docs/audits/interp_ledger.md section 2):
#:   row_id     a stable id, '<stimulus family>.<population>.<quantity>'
#:   label      the short name a probe file uses for the population (the match key; `population` is the resolvable
#:              flyverse.interp.common grammar spec, used for n_cells / bodyIds and the export)
#:   bound      the criterion value the op compares against (`expected` stays the reference value, as
#:              scripts/benchmark.py's Ref.value does); 'lo,hi' for op range
#:   gap        1 = a documented model deficit: failing prints KNOWN GAP and meeting it PASS (gap closed), exactly
#:              as scripts/benchmark.py's Ref.gap
#:   check_key  the scripts/benchmark.py REFERENCES key this row mirrors, so the two statuses can be compared
#:   requires   another row_id this row is only meaningful when: unless THAT row passes in the same arm, this one is
#:              NOT_APPLICABLE rather than a verdict (the compass bump's rate and width mean nothing in an arm whose
#:              bump does not exist -- the shipped default reads a 34.9 Hz 'bump' that dies in 0.03 s)
COLUMNS = ["row_id", "population", "label", "stimulus", "quantity", "expected", "op", "bound", "unit", "gap",
           "check_key", "requires", "source", "model_reference", "notes"]

#: Comparison operators. `>`, `<`, `>=`, `<=`, `sign`, `range` and `abs>=` are the contract; `==`, `abs<=`, `notnone`
#: and `is` are needed to reproduce scripts/benchmark.py's own criteria, and `report` records a value without ever
#: scoring it (the walk.power_max_hz rule of dynamics round 1).
OPS = (">", "<", ">=", "<=", "==", "abs>=", "abs<=", "sign", "range", "notnone", "is", "report")
STATUSES = ("PASS", "PASS (gap closed)", "FAIL", "KNOWN GAP", "MISSING", "RECORDED", "NOT_APPLICABLE")
PASS_STATUSES = ("PASS", "PASS (gap closed)")

#: Quantities whose value is a stimulus-minus-control difference, so a null arm is meaningful.
DIFFERENCE_QUANTITIES = ("diff_abs_best_cell_mean", "diff_max_over_cells_mean_mv", "diff_mean_over_cells_mean_mv",
                         "diff_signed_best_cell", "diff_rate_hz_max_cell", "diff_tuning_peak_mv", "figure_z",
                         "figure_z_abs", "figure", "figure_abs", "flip_hz", "dprime", "stimulus_minus_control")


# --------------------------------------------------------------------------------------------------- observations
@dataclass
class Observation:
    """One measured number from one finished run: which population, under which stimulus, of which quantity."""
    population: str
    stimulus: str
    quantity: str
    value: object
    unit: str = ""
    arm: str = "unknown"
    arm_source: str = "unknown"
    run: str = ""
    source: str = ""
    source_kind: str = ""
    is_null: bool = False
    detail: str = ""


def _obs_frame(obs: list) -> pd.DataFrame:
    cols = [f.name for f in dataclasses.fields(Observation)]
    if not obs:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    return pd.DataFrame([dataclasses.asdict(o) for o in obs])[cols]


# --------------------------------------------------------------------------------------------------- the table
def load_table(path=None) -> pd.DataFrame:
    """The expectation table: `flyverse/data/expected_responses.csv` (or `path`), with COLUMNS and OPS validated.

    One row = one expectation: a population (in `common.resolve`'s grammar), a stimulus protocol id, a quantity, the
    reference value, and the criterion `op` / `bound` the measurement has to meet, with the literature citation
    (`source`) and the file that measured it in this model (`model_reference`)."""
    p = Path(path or TABLE_PATH)
    df = pd.read_csv(p, comment="#", dtype=str, keep_default_na=False).rename(columns=str.strip)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: expectation table is missing columns {missing}")
    df = df[[c for c in COLUMNS]].copy()
    for c in COLUMNS:
        df[c] = df[c].astype(str).str.strip()
    bad = sorted(set(df.op) - set(OPS))
    if bad:
        raise ValueError(f"{p}: unknown op(s) {bad}; choose from {list(OPS)}")
    dup = df.row_id[df.row_id.duplicated()].tolist()
    if dup:
        raise ValueError(f"{p}: duplicate row_id(s) {sorted(set(dup))}")
    unknown = sorted({r for r in df.requires if r} - set(df.row_id))
    if unknown:
        raise ValueError(f"{p}: `requires` names row_id(s) that are not in the table: {unknown}")
    df["gap"] = df.gap.replace("", "0").astype(int).astype(bool)
    df["label"] = np.where(df.label == "", df.population, df.label)
    df["table_path"] = str(p)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------------------- scoring
def _f(v):
    """A float, or None when the value is not a finite number."""
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _bounds(bound: str) -> list:
    return [_f(b) for b in str(bound).split(",")]


def evaluate(op: str, value, bound, tolerance: float = 0.0, gap: bool = False) -> str:
    """One row's status: PASS / FAIL / MISSING / RECORDED, with the battery's gap labels when `gap`.

    `tolerance` relaxes the bound by `tolerance * max(|bound|, 1)` in the direction that makes the row easier; it is 0
    by default so the ledger reproduces `scripts/benchmark.py`'s own criteria exactly. Mirrors benchmark.evaluate for
    the ops the battery uses (`>`, `<`, `>=`, `<=`, `==`, `abs>=`, `notnone`)."""
    if op == "report":
        return "RECORDED"
    x = _f(value)
    if op == "is":
        ok = (value is not None) and str(value).strip() == str(bound).strip()
        if value is None or str(value).strip() == "":
            return "MISSING"
    elif op == "notnone":
        return "MISSING" if (value is None or (isinstance(value, str) and value.strip() == "")) else _label(True, gap)
    else:
        if x is None:
            return "MISSING"
        bs = _bounds(bound)
        if any(b is None for b in bs) or not bs:
            raise ValueError(f"op {op!r} needs a numeric bound, got {bound!r}")
        tol = [tolerance * max(abs(b), 1.0) for b in bs]
        if op == ">":
            ok = x > bs[0] - tol[0]
        elif op == ">=":
            ok = x >= bs[0] - tol[0]
        elif op == "<":
            ok = x < bs[0] + tol[0]
        elif op == "<=":
            ok = x <= bs[0] + tol[0]
        elif op == "==":
            ok = abs(x - bs[0]) <= tol[0]
        elif op == "abs>=":
            ok = abs(x) >= bs[0] - tol[0]
        elif op == "abs<=":
            ok = abs(x) <= bs[0] + tol[0]
        elif op == "sign":
            ok = (x == 0.0) if bs[0] == 0 else (x * bs[0] > 0)
        elif op == "range":
            if len(bs) != 2:
                raise ValueError(f"op 'range' needs 'lo,hi', got {bound!r}")
            ok = (bs[0] - tol[0]) <= x <= (bs[1] + tol[1])
        else:  # pragma: no cover -- load_table rejects unknown ops
            raise ValueError(f"unknown op {op!r}")
    return _label(bool(ok), gap)


def _label(ok: bool, gap: bool) -> str:
    if ok:
        return "PASS (gap closed)" if gap else "PASS"
    return "KNOWN GAP" if gap else "FAIL"


# --------------------------------------------------------------------------------------------------- source readers
_ARM_FILENAME = {"off": "off", "sign": "sign-class", "abs": "sign-abs", "class": "sign-class",
                 "classfb": "sign-class+fb", "nonmda": "sign-nonmda", "signgain": "sign+gain", "default": "default",
                 "full": "full", "ctrl": "control", "baseline": "baseline"}


def _arm_from_header(text: str):
    """The receptor arm a probe's console log declares ('receptor model sign (abs)' / 'receptor model None ...')."""
    m = re.search(r"receptor model\s+(None|\w+)\s*(?:\((\w+)\))?", text)
    if not m:
        return None
    if m.group(1) == "None":
        return "off"
    return f"{m.group(1)}-{m.group(2)}" if m.group(2) else m.group(1)


def _arm_from_name(path) -> str:
    stem = Path(path).stem
    for tok in re.split(r"[_\-.]", stem):
        if tok in _ARM_FILENAME:
            return _ARM_FILENAME[tok]
    return "unknown"


def _arm_for_probe(path, text=None) -> tuple:
    """(arm, how it was determined) for a probe output: the console header first, then the file name."""
    if text is None:
        sib = Path(path).with_suffix(".txt")
        text = sib.read_text(encoding="utf-8", errors="replace") if sib.exists() else ""
    a = _arm_from_header(text or "")
    if a:
        return a, "header"
    return _arm_from_name(path), "filename"


def _device_from_text(text: str):
    m = re.findall(r"device (\w+)", text or "")
    return m[0] if m else None


def _arm_from_receptor_cfg(cfg) -> str:
    """The receptor arm a JSON's config block declares. A missing / null receptor block is the presynaptic-sign model
    ('off'), which is how scripts/benchmark.py and scripts/batch_sustain.py write `--receptor-model off`."""
    if not isinstance(cfg, dict):
        return "off"
    model = cfg.get("model", cfg.get("receptor_model"))
    rule = cfg.get("net_rule", cfg.get("receptor_net_rule"))
    if model in (None, "None", "", "off"):
        return "off"
    arm = f"{model}-{rule}" if rule else str(model)
    table = cfg.get("table", cfg.get("receptor_table"))
    if table:                                   # a hold table names the arm (holdBrain, holdOptic, holdBrainHis, ...)
        stem = Path(str(table)).stem
        arm = stem[len("receptors_"):] if stem.startswith("receptors_") else stem
    return arm


def _get(d: dict, path: str):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# --- scripts/benchmark.py JSONs (out/benchmark_suite.json, out/rm2_*.json, out/r5_attr_*.json, ...)
#: (section, key path inside the section, population label, stimulus, quantity, unit)
BENCH_SCALARS = [
    ("rest", "spikes_per_step", "brain", "rest", "spikes_per_step", "spikes/step"),
    ("taste", "MN9_hz", "MN9", "sugar", "rate_hz", "Hz"),
    ("taste", "GNG175_hz", "GNG175", "sugar", "rate_hz", "Hz"),
    ("taste", "frac_active", "brain", "sugar", "frac_active", "fraction"),
    ("smell", "PN_hz", "PN", "apple_odour", "rate_hz", "Hz"),
    ("smell", "PN_max_hz", "PN", "apple_odour", "rate_max_hz", "Hz"),
    ("smell", "KC_hz", "KC", "apple_odour", "rate_hz", "Hz"),
    ("smell", "KC_active", "KC", "apple_odour", "n_active", "cells"),
    ("smell", "LN_hz", "LN", "apple_odour", "rate_hz", "Hz"),
    ("walk", "walk.GF_mean_hz", "DNp01", "walking", "rate_hz", "Hz"),
    ("walk", "walk.GF_max_hz", "DNp01", "walking", "rate_max_hz", "Hz"),
    ("walk", "walk.power_max_hz", "wing_power_MN", "walking", "rate_max_hz", "Hz"),
    ("walk", "walk.power_sustained_hz", "wing_power_MN", "walking", "rate_sustained_hz", "Hz"),
    ("walk", "walk.leg_hz", "leg_MN", "walking", "rate_hz", "Hz"),
    ("walk", "loom.GF_peak_hz", "DNp01", "loom_left", "rate_peak_hz", "Hz"),
    ("walk", "loom.escape_cm", "DNp01", "loom_left", "escape_range_cm", "cm"),
    ("loom_escape", "GF_peak_hz", "DNp01", "loom_demo", "rate_peak_hz", "Hz"),
    ("loom_escape", "escapes", "body", "loom_demo", "escapes", "seeds"),
    ("walk_gf", "p99_hz", "DNp01", "walking", "rate_p99_hz", "Hz"),
    ("walk_gf", "median_hz", "DNp01", "walking", "rate_median_hz", "Hz"),
    ("walk_gf", "voluntary_takeoffs", "body", "walking", "voluntary_takeoffs", "count"),
    ("rotation", "group_flip_hz", "optomotor_group", "rotation_cw_vs_ccw", "flip_hz", "Hz"),
    ("object", "LC10a_flip_hz", "LC10a", "apple_left_vs_right", "flip_hz", "Hz"),
    ("object", "LC10a_hz", "LC10a", "apple_left_vs_right", "rate_hz", "Hz"),
    ("bitter", "calibrated_sugar_MN9_hz", "MN9", "sugar", "rate_hz_calibrated", "Hz"),
    ("bitter", "calibrated_sugar_bitter_MN9_hz", "MN9", "sugar+bitter", "rate_hz_calibrated", "Hz"),
    ("bitter", "shiu_sugar_MN9_hz", "MN9", "sugar", "rate_hz_shiu", "Hz"),
    ("bitter", "shiu_sugar_bitter_MN9_hz", "MN9", "sugar+bitter", "rate_hz_shiu", "Hz"),
    ("odour", "apple_channel_8cm_hz", "LH_apple_channel", "apple_8cm", "rate_hz", "Hz"),
    ("odour", "apple_channel_clean_hz", "LH_apple_channel", "apple_clean", "rate_hz", "Hz"),
    ("compass", "during.wedge_hz", "EPG", "compass_pulse", "rate_hz", "Hz"),
    ("compass", "after.wedge_hz", "EPG", "compass_pulse_after", "rate_hz", "Hz"),
    ("compass", "wedge_cells_persisting", "EPG", "compass_pulse_after", "cells_persisting", "cells"),
    ("compass", "during.PEN_hz", "PEN", "compass_pulse", "rate_hz", "Hz"),
    ("compass", "during.Delta7_hz", "Delta7", "compass_pulse", "rate_hz", "Hz"),
    ("compass", "during.PFL3_hz", "PFL3", "compass_pulse", "rate_hz", "Hz"),
    ("hops", "voluntary_per_1000_fly_s", "body", "room_walking", "voluntary_per_1000_fly_s", "per 1000 fly-s"),
    ("hops", "escape_per_1000_fly_s", "body", "room_walking", "escape_per_1000_fly_s", "per 1000 fly-s"),
    ("hops", "walk_gf_max_median_hz", "DNp01", "room_walking", "rate_max_median_hz", "Hz"),
]


def read_benchmark(doc: dict, source: str) -> tuple:
    """A `scripts/benchmark.py` JSON -> (observations, {check key: check record}). Every REFERENCES check is also
    emitted as an observation under the population `check:<key>` so a ledger row can fall back to it."""
    arm = _arm_from_receptor_cfg(doc.get("config", {}).get("receptor", {}))
    run = Path(source).stem
    sec = doc.get("sections", {})
    out = []

    def add(pop, stim, q, val, unit, detail=""):
        if val is not None and not isinstance(val, (dict, list)):
            out.append(Observation(pop, stim, q, val, unit, arm, "config", run, source, "benchmark", detail=detail))

    for section, path, pop, stim, q, unit in BENCH_SCALARS:
        add(pop, stim, q, _get(sec.get(section, {}) or {}, path), unit)
    for t, v in (_get(sec, "motion.subtypes") or {}).items():
        add(t, "motion", "dsi", v.get("dsi"), "index")
        add(t, "motion", "preferred_direction", v.get("best"), "direction")
        add(t, "motion", "direction_correct", int(bool(v.get("correct"))), "0/1")
    subs = _get(sec, "motion.subtypes") or {}
    if subs:
        add("T4/T5", "motion", "min_dsi", float(min(v["dsi"] for v in subs.values())), "index")
        add("T4/T5", "motion", "correct_directions", int(sum(bool(v.get("correct")) for v in subs.values())), "count")
    for group, stim in (("rotation.types", "rotation_cw_vs_ccw"), ("wind.types", "wind_left_vs_right"),
                        ("object.types", "apple_left_vs_right")):
        for t, v in (_get(sec, group) or {}).items():
            for key, q, unit in (("flip_hz", "flip_hz", "Hz"), ("rate_hz", "rate_hz", "Hz")):
                add(t, stim, q, v.get(key), unit)
    for t, v in (_get(sec, "odour.types") or {}).items():
        add(t, "apple_8cm", "rate_hz", v.get("apple8"), "Hz")
        add(t, "apple_clean", "rate_hz", v.get("clean"), "Hz")
    for dn, v in (sec.get("dn") or {}).items():
        if not isinstance(v, dict):
            continue
        add(dn, f"{dn}_stim_150hz", "leg_L_hz", v.get("legL"), "Hz")
        add(dn, f"{dn}_stim_150hz", "leg_R_hz", v.get("legR"), "Hz")
        add(dn, f"{dn}_stim_150hz", "power_hz", v.get("power"), "Hz")
        if v.get("legL") is not None and v.get("legR") is not None:
            add(dn, f"{dn}_stim_150hz", "leg_asym_hz", float(v["legL"]) - float(v["legR"]), "Hz")
        for t, hz in (v.get("top") or {}).items():
            add(t, f"{dn}_stim_150hz", "rate_hz", hz, "Hz")
    for section in ("walk", "smell", "taste"):
        top = _get(sec, f"{section}.top") or _get(sec, f"{section}.walk.top") or {}
        stim = {"walk": "walking", "smell": "apple_odour", "taste": "sugar"}[section]
        for t, hz in top.items():
            add(t, stim, "rate_hz", hz, "Hz", detail=f"{section} section top-type mean")
    checks = {}
    for ch in doc.get("checks", []):
        checks[ch["key"]] = dict(ch, source=source, arm=arm)
        add(f"check:{ch['key']}", "benchmark", "measured", ch.get("measured"), "")
    return out, checks


# --- scripts/probe_object_sweep.py JSONs (out/r3obj/{ball,null}_*.json, out/obj/*.json)
def read_object_sweep(doc: dict, source: str) -> list:
    """A `probe_object_sweep.py` JSON. The condition-A block ('ball') carries the diff statistics; `config.null`
    marks a none-vs-none run, which becomes the null arm of `common.compare` (docs/audits/object_sweep.md 8.4)."""
    cfg = doc.get("config", {})
    arm = str(cfg.get("mode") or _arm_from_receptor_cfg(cfg))
    is_null = bool(cfg.get("null"))
    run = Path(source).stem
    out = []
    for t, v in (doc.get("ball") or {}).items():
        for q, unit in (("diff_abs_best_cell_mean", "rate units"), ("diff_signed_best_cell", "rate units"),
                        ("diff_max_over_cells_mean_mv", "mV"), ("diff_mean_over_cells_mean_mv", "mV"),
                        ("diff_rate_hz_max_cell", "Hz"), ("drive_mean_mv", "mV"), ("rate_hz_mean", "Hz"),
                        ("rate_hz_max_cell", "Hz"), ("dev_abs_best_cell_mean", "rate units")):
            if q in v:
                out.append(Observation(t, "object_sweep", q, v[q], unit, arm, "config", run, source,
                                       "object_sweep", is_null=is_null))
    return out


# --- scripts/probe_compass_room.py JSONs (out/cxroom/*.json)
_COMPASS = [("bump_hz_post", "EPG", "bump_rate_hz", "Hz"), ("width_half_post", "EPG", "bump_width_wedges", "wedges"),
            ("width_22_post", "EPG", "bump_width_22hz_wedges", "wedges"), ("survival_s", "EPG", "bump_survival_s", "s"),
            ("frac_confined_post", "EPG", "frac_confined", "fraction"),
            ("circ_corr_centre_heading", "EPG", "circ_corr_heading", "r"),
            ("r_bumpvel_yaw", "EPG", "r_bump_velocity_yaw", "r"),
            ("pen_L_post", "PEN_L", "rate_hz", "Hz"), ("pen_R_post", "PEN_R", "rate_hz", "Hz"),
            ("d7_L_post", "Delta7_L", "rate_hz", "Hz"), ("d7_R_post", "Delta7_R", "rate_hz", "Hz"),
            ("peg_post", "PEG", "rate_hz", "Hz"), ("ring_post", "ER_ExR", "rate_hz", "Hz"),
            ("glno_post", "GLNO", "rate_hz", "Hz"), ("pfn_post", "PFN", "rate_hz", "Hz"),
            ("hdelta_post", "hDelta", "rate_hz", "Hz"), ("pfl3_L_post", "PFL3_L", "rate_hz", "Hz"),
            ("pfl3_R_post", "PFL3_R", "rate_hz", "Hz"), ("dna02_L_post", "DNa02_L", "rate_hz", "Hz"),
            ("dna02_R_post", "DNa02_R", "rate_hz", "Hz")]


def read_compass_room(doc: dict, source: str) -> list:
    """A `probe_compass_room.py` JSON: the per-run summary of the 38 s post-pulse window, one observation per
    population (the run's mean over its 16 flies).

    The stimulus is `compass_room` in every run -- the same 2 s 40 Hz wedge pulse in the same room. What differs is
    the ARM: the ring gain pair (`control` = the shipped default, which has no attractor), with the walking program
    appended when one drove the fly (`gE2/gD15+cx`), since the program changes the self-motion the bump sees and not
    the protocol being scored."""
    cfg = doc.get("config", {})
    gE, gD = cfg.get("gE"), cfg.get("gD")
    arm = "control" if gE is None else f"gE{gE:g}/gD{gD:g}"
    program = str(cfg.get("program") or "none")
    if program not in ("none", "None", ""):
        arm = f"{arm}+{program}"
    stim = "compass_room"
    run = Path(source).stem
    s = doc.get("summary", {})
    out = []
    for key, pop, q, unit in _COMPASS:
        v = s.get(key)
        val = v.get("mean") if isinstance(v, dict) else v
        if val is not None:
            out.append(Observation(pop, stim, q, val, unit, arm, "config", run, source, "compass_room",
                                   detail=f"mean over {v.get('n')} flies" if isinstance(v, dict) else ""))
    return out


# --- scripts/batch_sustain.py JSONs (out/r5_sustain_*.json, out/feedh_*.json)
def read_sustain(doc: dict, source: str) -> list:
    """A `batch_sustain.py` room-rollout JSON: the take-off rates and the walking-phase GF maximum.

    The arm is the receptor block (`receptor_model` / `receptor_net_rule`, or the hold table's name when one was
    passed), so the round-5 take-off arms -- off / default / holdBrain / holdOptic -- separate on their own."""
    arm = _arm_from_receptor_cfg(doc.get("receptor") or doc.get("options") or {})
    run = Path(source).stem
    stim = "room_walking"
    pairs = [("hops_voluntary_per_1000_fly_s", "body", "voluntary_per_1000_fly_s", "per 1000 fly-s"),
             ("hops_escape_per_1000_fly_s", "body", "escape_per_1000_fly_s", "per 1000 fly-s"),
             ("gf_max_walk_median_hz", "DNp01", "rate_max_median_hz", "Hz")]
    return [Observation(pop, stim, q, doc[key], unit, arm, "config", run, source, "batch_sustain",
                        detail=f"{doc.get('fly_s')} fly-s") for key, pop, q, unit in pairs if key in doc]


# --- scripts/r5_attr_taste_cpu.py JSON (out/r5_attr_taste_cpu.json): arms x seeds, CPU taste / smell
def read_taste_cpu(doc: dict, source: str) -> list:
    """`scripts/r5_attr_taste_cpu.py`'s arm x seed table (the CPU taste / smell protocol of the round-5 double
    dissociation). Each (arm, seed) is one run; the arm name is the hold table's id."""
    out = []
    for arm, block in doc.items():
        for seed, v in (block.get("seeds") or {}).items():
            run = f"{Path(source).stem}:{arm}:s{seed}"
            for key, pop, stim, q, unit in (("MN9_hz", "MN9", "sugar", "rate_hz", "Hz"),
                                            ("GNG175_hz", "GNG175", "sugar", "rate_hz", "Hz"),
                                            ("PN_hz", "PN", "apple_odour", "rate_hz", "Hz"),
                                            ("KC_hz", "KC", "apple_odour", "rate_hz", "Hz"),
                                            ("KC_active", "KC", "apple_odour", "n_active", "cells")):
                if key in v:
                    out.append(Observation(pop, stim, q, v[key], unit, arm, "config", run, source, "taste_cpu"))
    return out


# --- scripts/probe_loom.py console logs (out/loom*.txt)
_LOOM_TOKEN = re.compile(r"([A-Za-z][A-Za-z0-9_\- ]*?)=(-?\d+(?:\.\d+)?)")


def read_loom_text(text: str, source: str) -> list:
    """`scripts/probe_loom.py`'s console log: the per-100 ms population rates (Hz) during the approach, the walking
    baseline, and the escape the body took."""
    arm, how = _arm_for_probe(source, text)
    run = Path(source).stem
    loom, walk = {}, {}
    for line in text.splitlines():
        if line.startswith("[t="):
            tgt = loom
        elif line.startswith("[walking"):
            tgt = walk
        else:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        for name, val in _LOOM_TOKEN.findall(parts[2]):
            tgt[name.strip()] = max(float(val), tgt.get(name.strip(), -np.inf))
    out = [Observation(t, "loom_left", "rate_peak_hz", v, "Hz", arm, how, run, source, "loom") for t, v in loom.items()]
    out += [Observation(t, "walking", "rate_max_hz", v, "Hz", arm, how, run, source, "loom") for t, v in walk.items()]
    m = re.search(r"escape triggered at t=([\d.]+)s, object ([\d.]+) cm away", text)
    if m:
        out.append(Observation("DNp01", "loom_left", "escape_range_cm", float(m.group(2)), "cm", arm, how, run, source, "loom"))
        out.append(Observation("body", "loom_left", "escape_t_s", float(m.group(1)), "s", arm, how, run, source, "loom"))
    m = re.search(r"^escape: (True|False)", text, re.M)
    if m:
        out.append(Observation("body", "loom_left", "escape", int(m.group(1) == "True"), "0/1", arm, how, run, source, "loom"))
    return out


# --- scripts/probe_bitter.py console logs (out/bitter*.txt)
_BITTER = re.compile(r"^(.*?)\s\s+(sugar \+ bitter|sugar|bitter)\s*:\s*MN9\s+(-?[\d.]+)\s*Hz", re.M)


def read_bitter_text(text: str, source: str) -> list:
    """`scripts/probe_bitter.py`'s table: MN9 under sugar / sugar + bitter / bitter, under this project's calibrated
    rules and under Shiu et al. 2024's uniform rules."""
    arm, how = _arm_for_probe(source, text)
    run = Path(source).stem
    out = []
    for ruleset, cond, hz in _BITTER.findall(text):
        rs = "shiu" if "Shiu" in ruleset else "calibrated"
        out.append(Observation("MN9", cond.replace(" ", ""), f"rate_hz_{rs}", float(hz), "Hz", arm, how, run, source, "bitter"))
    return out


# --- scripts/screen_rotation.py / probe_figure_ground.py CSVs
def read_rotation_csv(df: pd.DataFrame, source: str) -> list:
    """`scripts/screen_rotation.py`'s per-type table: the L - R flip between +90 and -90 deg/s imposed yaw."""
    arm, how = _arm_for_probe(source)
    run = Path(source).stem
    out = []
    for _, r in df.iterrows():
        for col, q, unit in (("flip_d", "dprime", "d'"), ("flip_hz", "flip_hz", "Hz"), ("rate_hz", "rate_hz", "Hz"),
                             ("ccw_LR", "ccw_LR_hz", "Hz"), ("cw_LR", "cw_LR_hz", "Hz"), ("rest_LR", "rest_LR_hz", "Hz")):
            if col in df.columns and pd.notna(r[col]):
                out.append(Observation(str(r["type"]), "rotation_cw_vs_ccw", q, float(r[col]), unit, arm, how, run,
                                       source, "rotation"))
    return out


# --- scripts/probe_figure_stages.py / scripts/audit_optic.py stage JSONs (out/optic_audit/<config>/stages_s*.json)
#: which stage-file block is which protocol, and which arm pair inside it is the stimulus and which the null.
_STAGE_STIM = {"apple": "figure_ground_apple", "ball": "figure_ground_ball"}
_STAGE_FAMILY = {"signed": "figure_z", "abs": "figure_z_abs"}


def read_stages(doc: dict, source: str) -> list:
    """A `scripts/probe_figure_stages.py` stage JSON -- the file that produced `docs/audits/optic_measures.md` 5.1.

    Per type and per stage: the signed and |dev| retinotopic figure statistic of the static apple (`apple`) and the
    moving ball (`ball`). The stimulus arm is the file's `AB` block (object vs matched none); its `CB` block is the
    file's own **none-vs-none null** and is emitted with `is_null`, so `common.compare` has a matched null without a
    second file. The arm is the receptor model, with the optic configuration appended when it is not `baseline`
    (`out/optic_audit/no_spk_feedback/stages_s0.json` -> `sign-abs+no_spk_feedback`)."""
    cfg = doc.get("config", {})
    arm = _arm_from_receptor_cfg(cfg)
    conf = str(cfg.get("optic_config") or "baseline")
    if conf and conf != "baseline":
        arm = f"{arm}+{conf}"
    p = Path(source)
    run = f"{p.parent.name}/{p.stem}"
    out = []
    for block, stim in _STAGE_STIM.items():
        for t, v in ((doc.get(block) or {}).get("types") or {}).items():
            if not isinstance(v, dict):
                continue
            detail = f"stage {v.get('stage')} {v.get('kind')} n_cells {v.get('n_cells')}"
            for family, q in _STAGE_FAMILY.items():
                for pair, is_null in (("AB", False), ("CB", True)):
                    d = (v.get(family) or {}).get(pair)
                    if not isinstance(d, dict):
                        continue
                    out.append(Observation(t, stim, q, d.get("z"), "z", arm, "config", run, source, "figure_stages",
                                           is_null=is_null, detail=detail))
                    out.append(Observation(t, stim, q.replace("figure_z", "figure"), d.get("figure"), "rate units",
                                           arm, "config", run, source, "figure_stages", is_null=is_null, detail=detail))
    return out


def read_figure_csv(df: pd.DataFrame, source: str) -> list:
    """`scripts/probe_figure_ground.py`'s per-type table: the signed retinotopic figure statistic and its z."""
    arm, how = _arm_for_probe(source)
    run = Path(source).stem
    out = []
    for _, r in df.iterrows():
        for col, q, unit in (("figure_z", "figure_z", "z"), ("figure", "figure", "rate units"),
                             ("bg_level", "bg_level", "rate units")):
            if col in df.columns and pd.notna(r[col]):
                out.append(Observation(str(r["type"]), "figure_ground_apple", q, float(r[col]), unit, arm, how, run,
                                       source, "figure_ground"))
    return out


# --- flyverse.interp.result/1 JSONs (any tool's Result)
_RESULT_SKIP = {"ledger", "sources", "observations"}
_RESULT_META = {"type", "population", "pre_type", "post_type", "stimulus", "quantity", "unit", "bodyId", "model_index",
                "unit_kind", "verdict", "sign_rule", "gain_rule", "kind", "stage", "depth", "n_cells", "control_ids",
                "window", "window_start_s", "window_end_s", "normalisation", "reference_graph"}


def read_result(doc: dict, source: str) -> list:
    """Any `flyverse.interp.result/1` JSON: every table row that names a population (`type`, `population`, or a
    `pre_type` / `post_type` pair) contributes its numeric columns as observations, the stimulus coming from the row
    or from `provenance.stimulus.protocol`. `readout_per_body` rows are pooled per (type, quantity)."""
    prov = doc.get("provenance", {})
    stim0 = str(_get(prov, "stimulus.protocol") or doc.get("tool") or "unknown")
    arm = str(_get(prov, "model.lif.receptor_model") or "off")
    rule = _get(prov, "model.lif.receptor_net_rule")
    arm = "off" if arm in ("None", "none", "") else (f"{arm}-{rule}" if rule else arm)
    run = str(doc.get("run_id") or Path(source).stem)
    out = []
    for name, rows in (doc.get("tables") or {}).items():
        if name in _RESULT_SKIP or not isinstance(rows, list):
            continue
        if name == "readout_per_body":
            out += _readout_rows(rows, stim0, arm, run, source)
            continue
        for rec in rows:
            if not isinstance(rec, dict):
                continue
            pop = rec.get("type") or rec.get("population")
            if pop is None and rec.get("pre_type") and rec.get("post_type"):
                pop = f"{rec['pre_type']}->{rec['post_type']}"
            if pop is None:
                continue
            stim = str(rec.get("stimulus") or stim0)
            if rec.get("quantity") is not None and any(k in rec for k in ("value", "measured")):
                val = rec.get("value", rec.get("measured"))
                out.append(Observation(str(pop), stim, str(rec["quantity"]), val, str(rec.get("unit", "")), arm,
                                       "provenance", run, source, "result", detail=name))
                continue
            for k, v in rec.items():
                if k in _RESULT_META or isinstance(v, (dict, list)) or isinstance(v, bool) or _f(v) is None:
                    continue
                out.append(Observation(str(pop), stim, k, float(v), "", arm, "provenance", run, source, "result",
                                       detail=name))
    return out


def _readout_rows(rows: list, stim0: str, arm: str, run: str, source: str) -> list:
    df = pd.DataFrame(rows)
    if df.empty or "type" not in df.columns:
        return []
    q = df["quantity"].astype(str) if "quantity" in df.columns else pd.Series(["value"] * len(df))
    out = []
    for (t, qq), g in df.assign(_q=q).groupby(["type", "_q"], sort=True):
        for col, suffix in (("stimulus_value", ""), ("stimulus_minus_control", "_minus_control")):
            if col in g.columns:
                v = pd.to_numeric(g[col], errors="coerce").mean()
                if np.isfinite(v):
                    unit = str(g["unit"].iloc[0]) if "unit" in g.columns else ""
                    out.append(Observation(str(t), stim0, f"{qq}{suffix}", float(v), unit, arm, "provenance", run,
                                           source, "result", detail=f"readout_per_body, {len(g)} bodies"))
    return out


# --------------------------------------------------------------------------------------------------- dispatch
def _sniff_json(doc: dict) -> str:
    if doc.get("schema") == common.SCHEMA:
        return "result"
    if "sections" in doc and "checks" in doc:
        return "benchmark"
    if "ball" in doc and "none" in doc and "verdict" in doc:
        return "object_sweep"
    if any(isinstance(doc.get(b), dict) and "types" in doc[b] for b in _STAGE_STIM):
        return "figure_stages"
    if "n_epg" in doc and "summary" in doc:
        return "compass_room"
    if "hops_total" in doc:
        return "batch_sustain"
    if doc and all(isinstance(v, dict) and "seeds" in v for v in doc.values()):
        return "taste_cpu"
    return "unknown"


def read_source(src) -> tuple:
    """One file (or an in-memory Result / dict) -> (observations, checks, provenance row). Unreadable or unknown
    files are reported in the provenance row's `error` field rather than raising."""
    if isinstance(src, Result):
        src = src.to_dict()
    if isinstance(src, dict):
        kind = _sniff_json(src)
        name = str(src.get("run_id") or kind)
        obs, checks = (read_benchmark(src, name) if kind == "benchmark" else (_read_json(src, kind, name), {}))
        return obs, checks, {"source": name, "kind": kind, "n_observations": len(obs),
                             "device": _source_device(src, kind), "error": ""}
    path = Path(src)
    row = {"source": str(path), "kind": "unknown", "n_observations": 0, "device": None, "error": ""}
    try:
        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            kind = _sniff_json(doc)
            row["kind"] = kind
            row["device"] = _source_device(doc, kind)
            obs, checks = (read_benchmark(doc, str(path)) if kind == "benchmark" else (_read_json(doc, kind, str(path)), {}))
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            checks = {}
            if "figure_z" in df.columns:
                row["kind"], obs = "figure_ground", read_figure_csv(df, str(path))
            elif "flip_d" in df.columns:
                row["kind"], obs = "rotation", read_rotation_csv(df, str(path))
            else:
                obs = []
                row["error"] = f"unrecognised CSV columns {list(df.columns)[:6]}"
        elif path.suffix in (".txt", ".log"):
            text = path.read_text(encoding="utf-8", errors="replace")
            checks = {}
            row["device"] = _device_from_text(text)
            if _BITTER.search(text):
                row["kind"], obs = "bitter", read_bitter_text(text, str(path))
            elif "escape" in text and "DNp01=" in text:
                row["kind"], obs = "loom", read_loom_text(text, str(path))
            else:
                obs = []
                row["error"] = "not a loom or bitter console log"
        else:
            return [], {}, dict(row, error=f"unsupported suffix {path.suffix!r}")
    except Exception as e:  # noqa: BLE001 -- one unreadable file must not stop the ledger
        return [], {}, dict(row, error=repr(e))
    row["n_observations"] = len(obs)
    if not obs and not row["error"]:
        row["error"] = "no observations extracted"
    return obs, checks, row


def _read_json(doc: dict, kind: str, source: str) -> list:
    return {"result": read_result, "object_sweep": read_object_sweep, "compass_room": read_compass_room,
            "batch_sustain": read_sustain, "taste_cpu": read_taste_cpu,
            "figure_stages": read_stages}.get(kind, lambda *a: [])(doc, source)


def _source_device(doc: dict, kind: str):
    for path in ("provenance.execution.device", "config.device", "device", "options.device"):
        v = _get(doc, path)
        if isinstance(v, str) and v:
            return v
    return None


def expand(paths) -> list:
    """Paths / globs / Results / dicts -> a flat list, globs sorted, duplicates dropped, order preserved."""
    if paths is None:
        return []
    if isinstance(paths, (str, Path, dict, Result)):
        paths = [paths]
    out, seen = [], set()
    for p in paths:
        if isinstance(p, (dict, Result)):
            out.append(p)
            continue
        s = str(p)
        hits = sorted(_glob.glob(s, recursive=True)) if any(ch in s for ch in "*?[") else [s]
        for h in hits:
            if h not in seen:
                seen.add(h)
                out.append(h)
    return out


def read_sources(paths, force_null: bool = False) -> tuple:
    """Every source -> (observation DataFrame, {check key: record}, provenance DataFrame)."""
    obs, checks, rows = [], {}, []
    for p in expand(paths):
        o, ch, row = read_source(p)
        if force_null:
            for x in o:
                x.is_null = True
        obs += o
        checks.update(ch)
        rows.append(row)
    return _obs_frame(obs), checks, pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "kind", "n_observations", "device", "error"])


# --------------------------------------------------------------------------------------------------- pooling
def _pool(values: list) -> dict:
    """Replicate pooling: numeric arms give mean / sd / n, string arms the majority value and whether runs disagree."""
    nums = [_f(v) for v in values]
    if all(n is not None for n in nums) and nums:
        a = np.asarray(nums, dtype=np.float64)
        return {"measured": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
                "n": int(len(a)), "values": [float(x) for x in a]}
    s = [str(v) for v in values]
    top = max(set(s), key=s.count) if s else None
    return {"measured": top, "sd": float("nan"), "n": len(s), "values": s}


def row_quantity(row) -> tuple:
    """(the quantity a source measures, what the row is scored on). A table quantity ending in '@z' is matched on the
    base quantity and scored on the z against the null arm, the statistic docs/audits/object_sweep.md 8.4 uses."""
    q = str(row.quantity)
    return (q[:-2], "z") if q.endswith("@z") else (q, "value")


def _match(obs: pd.DataFrame, row, check_obs_ok: bool = True) -> pd.DataFrame:
    if obs.empty:
        return obs
    names = {row.label, row.population}
    base, _ = row_quantity(row)
    m = obs[obs.population.isin(names) & obs.stimulus.eq(row.stimulus) & obs.quantity.eq(base)]
    if m.empty and check_obs_ok and row.check_key:
        m = obs[obs.population.eq(f"check:{row.check_key}")]
    return m


# --------------------------------------------------------------------------------------------------- the tool
def ledger(results, *, table=None, tolerance=0.0, strict=False, null=(), c=None, group_by="arm",
           min_replicates=MIN_REPLICATES, z_min=Z_RESULT, generator=None, cache_dir=None, lif=None, optic=None):
    """Score every probe's per-population output against `flyverse/data/expected_responses.csv`.

    `results`: Result objects, dicts or paths / globs of any supported output (a `flyverse.interp.result/1` JSON from
    any tool, a `scripts/benchmark.py` JSON, a `probe_object_sweep.py` / `probe_compass_room.py` / `batch_sustain.py`
    JSON, a `r5_attr_taste_cpu.py` JSON, a loom / bitter console log, a rotation / figure-ground CSV). Every row of
    the table whose (population, stimulus, quantity) a source measured is scored PASS / FAIL / MISSING -- with the
    battery's `KNOWN GAP` / `PASS (gap closed)` labels on rows marked `gap`, and `RECORDED` on rows whose op is
    `report` (a number the project has decided is not a verdict) -- and carries the measured value, the replicate
    scatter, the citation and, where the row mirrors a `scripts/benchmark.py` check, that check's own status so the
    two can be compared.

    Replicates and the null: runs are pooled per arm (`group_by='arm'`; `None` pools everything into one row) and the
    row reports mean / sd / n and how many runs pass on their own. Sources that declare themselves a none-vs-none
    null, and anything passed in `null=`, form the null arm; for the difference quantities of DIFFERENCE_QUANTITIES
    the row additionally carries `common.compare`'s z / Welch / U / p / verdict, which reads `underpowered` below
    `min_replicates` runs per arm whatever the numbers.

    `tolerance` relaxes every bound by that fraction (0 = the battery's own criteria); `strict` makes a MISSING row
    count as a failure in `summary.ok`. `c` is the Connectome used to resolve each row's population spec to cell
    counts and bodyIds (loaded from the cache when None; the scoring itself needs no connectome).

    Tables: 'ledger' (every row with status, measured, sd, n and the citation), 'sources' (one row per file read,
    with the realised device it recorded), 'observations' (every measurement that matched a row).
    Validation: VALIDATION['ledger'] -- the T4/T5 DSI, loom GF, HSN / DNp20 / HSE d' and sugar / bitter MN9 rows
    reproduce `scripts/benchmark.py`'s statuses on its own JSON.
    """
    rows = load_table(table)
    obs, checks, srcs = read_sources(results)
    nobs, _, nsrcs = read_sources(null, force_null=True)
    if not nobs.empty:
        obs = pd.concat([obs, nobs], ignore_index=True)
        srcs = pd.concat([srcs, nsrcs], ignore_index=True)
    stim_obs = obs[~obs.is_null.astype(bool)] if not obs.empty else obs
    null_obs = obs[obs.is_null.astype(bool)] if not obs.empty else obs

    records, used = [], set()
    for row in rows.itertuples(index=False):
        m = _match(stim_obs, row)
        used.update(m.index.tolist())
        arms = sorted(m.arm.unique()) if (group_by == "arm" and not m.empty) else [None]
        for arm in arms:
            sub = m if arm is None else m[m.arm.eq(arm)]
            rec = _base_record(row, tolerance)
            rec["arm"] = arm if arm is not None else ("pooled" if not m.empty else "")
            if not sub.empty:
                pooled = _pool(list(sub.value))
                rec.update(pooled)
                rec["values"] = pooled["values"]
                rec["sources"] = sorted(set(sub.source))
                rec["runs"] = sorted(set(sub.run))
                rec["source_kinds"] = sorted(set(sub.source_kind))     # >1 kind = two protocols pooled; read the sd
                rec["replicates_ok"] = bool(pooled["n"] >= min_replicates)
            _add_null(rec, row, null_obs, arm, z_min, min_replicates)
            scored_on = rec["scored_on"]
            scored = rec["z"] if scored_on == "z" else rec["measured"]
            rec["status"] = evaluate(row.op, scored, row.bound, tolerance, row.gap)
            if not sub.empty and scored_on == "value":
                per_run = [evaluate(row.op, v, row.bound, tolerance, row.gap) for v in sub.value]
                rec["runs_passing"] = f"{sum(s.startswith('PASS') for s in per_run)}/{len(per_run)}"
                rec["unstable"] = len(set(per_run)) > 1
            _add_battery(rec, row, checks)
            records.append(rec)

    led = _apply_preconditions(pd.DataFrame(records))
    prov = _provenance(c, cache_dir, lif, optic, srcs)
    res = Result.new(TOOL, prov)
    res.add_table("ledger", led)
    res.add_table("sources", srcs)
    res.add_table("observations", stim_obs.loc[sorted(used)] if used else stim_obs.iloc[0:0])
    _add_populations(res, rows, led, c)
    res.summary = _summary(led, srcs, rows, tolerance, strict, min_replicates)
    res.validation = dict(res.validation, **_validate(led))
    res.files = {"table": str(rows.table_path.iloc[0]) if len(rows) else str(TABLE_PATH),
                 "sources": [str(s) for s in srcs.source] if len(srcs) else [],
                 "generator": generator or "flyverse.interp.ledger.ledger"}
    return res


def _base_record(row, tolerance: float) -> dict:
    return {"row_id": row.row_id, "population": row.population, "label": row.label, "stimulus": row.stimulus,
            "quantity": row.quantity, "arm": "", "expected": row.expected, "op": row.op, "bound": row.bound,
            "unit": row.unit, "measured": None, "sd": float("nan"), "n": 0, "values": [], "runs_passing": "0/0",
            "unstable": False, "replicates_ok": False, "scored_on": row_quantity(row)[1],
            "status": "MISSING", "gap": bool(row.gap), "requires": row.requires, "precondition": "",
            "source_kinds": [],
            "tolerance": float(tolerance), "check_key": row.check_key, "battery_status": "",
            "battery_agrees": None, "battery_criterion": "", "battery_criterion_same": None, "battery_measured": None,
            "null_mean": float("nan"), "null_sd": float("nan"), "null_n": 0,
            "z": float("nan"), "welch": float("nan"), "U": float("nan"), "p": float("nan"), "null_verdict": "",
            "source": row.source, "model_reference": row.model_reference, "notes": row.notes, "sources": [],
            "runs": []}


def _apply_preconditions(led: pd.DataFrame) -> pd.DataFrame:
    """A row whose `requires` row did not pass **in the same arm** is NOT_APPLICABLE, not a verdict.

    The compass is why this exists: `bump_hz_post` in an arm with no attractor is the peak of a wedge that dies in
    0.03 s, and scoring it against Seelig & Jayaraman's bump rate turns 'there is no bump' into a PASS. The
    precondition makes the ledger say what the run actually shows -- the bump row is the one that failed."""
    if not len(led) or "requires" not in led.columns:
        return led
    ok = {(r.row_id, r.arm): r.status in PASS_STATUSES for r in led.itertuples(index=False)}
    for i, r in enumerate(led.itertuples(index=False)):
        if not r.requires or r.status in ("MISSING", "NOT_APPLICABLE"):
            continue
        met = ok.get((r.requires, r.arm))
        if met is False:
            led.iat[i, led.columns.get_loc("status")] = "NOT_APPLICABLE"
            led.iat[i, led.columns.get_loc("precondition")] = f"{r.requires} did not pass in arm {r.arm!r}"
        elif met is None:
            led.iat[i, led.columns.get_loc("precondition")] = f"{r.requires} not measured in arm {r.arm!r}"
    return led


def _add_null(rec: dict, row, null_obs: pd.DataFrame, arm, z_min: float, min_n: int) -> None:
    base, _ = row_quantity(row)
    if null_obs.empty or base not in DIFFERENCE_QUANTITIES or rec["n"] == 0:
        return
    nm = _match(null_obs, row, check_obs_ok=False)
    if arm is not None and not nm.empty:
        same = nm[nm.arm.eq(arm)]
        nm = same if not same.empty else nm
    vals = [_f(v) for v in nm.value]
    vals = [v for v in vals if v is not None]
    if not vals:
        return
    cmp = common.compare(rec["values"], vals, z_min=z_min, min_n=min_n)
    rec.update({"null_mean": cmp["null"]["mean"], "null_sd": cmp["null"]["sd"], "null_n": cmp["null"]["n"],
                "z": cmp["z"], "welch": cmp["welch"], "U": cmp["U"], "p": cmp["p"], "null_verdict": cmp["verdict"]})


def _same_criterion(op: str, bound: str, criterion: str) -> bool:
    """Does this row's `op` / `bound` say the same thing as the criterion string a benchmark JSON stored
    (`f'{ref.op} {ref.bound}'`, scripts/benchmark.py:289)? A criterion the stored file does not carry is 'unknown'."""
    if not criterion:
        return True
    parts = str(criterion).split()
    if len(parts) != 2 or parts[0] != op:
        return False
    a, b = _f(parts[1]), _f(bound)
    return (a == b) if (a is not None and b is not None) else (parts[1] == str(bound))


def _add_battery(rec: dict, row, checks: dict) -> None:
    """The status `scripts/benchmark.py` itself wrote for the check this row mirrors, and whether the two criteria
    are the same one. A stored JSON keeps the criterion that was in force when it ran, so a bound the battery has
    since retuned shows up as `battery_criterion_same = False` -- a stale file, not a scoring bug."""
    ch = checks.get(row.check_key) if row.check_key else None
    if ch is None:
        return
    rec["battery_status"] = ch.get("status", "")
    rec["battery_criterion"] = str(ch.get("criterion", ""))
    rec["battery_criterion_same"] = _same_criterion(row.op, row.bound, rec["battery_criterion"])
    rec["battery_measured"] = ch.get("measured")
    if rec["status"] == "RECORDED":
        rec["battery_agrees"] = None            # deliberately not scored here; see the module docstring
    else:
        rec["battery_agrees"] = bool(rec["status"] == ch.get("status"))


def _add_populations(res: Result, rows: pd.DataFrame, led: pd.DataFrame, c) -> None:
    if c is None:
        return
    seen = set(led.label[led.n > 0]) if len(led) else set()
    for spec, label in rows[["population", "label"]].drop_duplicates().itertuples(index=False):
        if label not in seen:
            continue
        try:
            pop = common.population(c, spec, label)
        except Exception:  # noqa: BLE001 -- a label that is not a connectome spec (e.g. 'body', 'check:...')
            continue
        res.add_population(pop, keep_ids=pop.n <= 10000)


def _devices(srcs: pd.DataFrame) -> list:
    """The realised devices the scored runs recorded, NaN / empty dropped (a source that recorded none is reported
    by its row in the 'sources' table, not by a 'nan' entry here)."""
    if not len(srcs):
        return []
    return sorted({str(d) for d in srcs.device if isinstance(d, str) and d.strip() and d.strip().lower() != "nan"})


def _summary(led: pd.DataFrame, srcs: pd.DataFrame, rows: pd.DataFrame, tolerance, strict, min_replicates) -> dict:
    counts = {s: int((led.status == s).sum()) for s in STATUSES} if len(led) else {s: 0 for s in STATUSES}
    scored = led[led.status.isin(list(PASS_STATUSES) + ["FAIL", "KNOWN GAP"])] if len(led) else led
    bat = led[led.battery_agrees.notna()] if len(led) else led
    disagree = bat[~bat.battery_agrees.astype(bool)] if len(bat) else bat
    # rows whose stored check used a different criterion from the table's: the file predates (or postdates) the bound.
    # `report` rows are excluded -- their divergence from the battery is deliberate (see the module docstring).
    stale = led[led.battery_criterion_same.eq(False) & led.op.ne("report")] if len(led) else led
    if len(disagree) and len(stale):            # a disagreement the stored file's own criterion explains is not a bug
        disagree = disagree[~disagree.row_id.isin(set(stale.row_id))]
    under = scored[(scored.n > 0) & (scored.n < min_replicates)] if len(scored) else scored
    return {"n_table_rows": int(len(rows)), "n_ledger_rows": int(len(led)), "status_counts": counts,
            "n_scored": int(len(scored)), "n_failing": counts["FAIL"], "n_known_gap": counts["KNOWN GAP"],
            "n_missing": counts["MISSING"], "n_recorded": counts["RECORDED"],
            "n_not_applicable": counts["NOT_APPLICABLE"],
            "not_applicable": led.loc[led.status.eq("NOT_APPLICABLE"), ["row_id", "arm", "precondition"]]
            .to_dict("records") if len(led) else [],
            "battery_rows": int(len(bat)), "battery_agree": int(bat.battery_agrees.sum()) if len(bat) else 0,
            "battery_disagree": disagree[["row_id", "arm", "status", "battery_status", "measured"]].to_dict("records")
            if len(disagree) else [],
            "battery_criterion_mismatch": stale[["row_id", "arm", "op", "bound", "battery_criterion", "status",
                                                 "battery_status"]].to_dict("records") if len(stale) else [],
            "underpowered_rows": sorted(set(under.row_id)) if len(under) else [],
            "arms": sorted({a for a in led.arm if a}) if len(led) else [],
            "sources": {"n": int(len(srcs)), "ok": int((srcs.error == "").sum()) if len(srcs) else 0,
                        "kinds": {k: int(v) for k, v in srcs.kind.value_counts().items()} if len(srcs) else {},
                        "devices": _devices(srcs) if len(srcs) else []},
            "tolerance": float(tolerance), "strict": bool(strict), "min_replicates": int(min_replicates),
            "ok": bool(counts["FAIL"] == 0 and (counts["MISSING"] == 0 or not strict))}


#: The validation target of docs/INTERP.md section 6: these row ids must reproduce scripts/benchmark.py's statuses.
VALIDATION_ROWS = ["motion.T4_T5.min_dsi", "motion.T4_T5.correct_directions", "loom.DNp01.rate_peak_hz",
                   "rotation.HSN.dprime", "rotation.DNp20.dprime", "rotation.HSE.dprime",
                   "taste.MN9.rate_hz_calibrated", "taste.MN9.rate_hz_calibrated_bitter",
                   "taste.MN9.rate_hz_shiu", "taste.MN9.rate_hz_shiu_bitter"]


def _validate(led: pd.DataFrame) -> dict:
    if not len(led):
        return {"measured": {}, "status": "not run"}
    sel = led[led.row_id.isin(VALIDATION_ROWS)]
    measured = {r.row_id: {"arm": r.arm, "measured": r.measured, "n": int(r.n), "sd": r.sd, "status": r.status,
                           "battery_status": r.battery_status, "battery_agrees": r.battery_agrees}
                for r in sel.itertuples(index=False)}
    got = sel[sel.n > 0]
    bat = sel[sel.battery_agrees.notna() & sel.battery_criterion_same.ne(False)]
    if not len(got):
        status = "not run"
    elif len(bat) and not bool(bat.battery_agrees.astype(bool).all()):
        status = "not reproduced"
    elif len(got) < len(VALIDATION_ROWS):
        status = f"partially reproduced ({len(got)}/{len(VALIDATION_ROWS)} rows measured)"
    else:
        status = "reproduced"
    return {"measured": measured, "status": status}


def _provenance(c, cache_dir, lif, optic, srcs: pd.DataFrame) -> dict:
    """The ledger runs no simulation: `execution.device` is the CPU that scored, and every scored run's own realised
    device is in the 'sources' table and echoed in `execution.scored_devices`."""
    if c is None:
        try:
            from .. import connectome as cn
            c = cn.load(cache_dir=cache_dir, verbose=False) if cache_dir else cn.load(verbose=False)
        except Exception:  # noqa: BLE001 -- the ledger scores files; the cache is only needed to resolve populations
            c = None
    if c is None:
        prov = {"flyverse_commit": common.git_state(), "dataset_release": common.dataset_release(),
                "compiled_connectome": {"error": "no connectome loaded (scoring is file-only)", "cache_dir": str(cache_dir)},
                "model": common.model_record(lif, optic), "execution": common.execution_record(device="cpu"),
                "stimulus": {"protocol": "ledger", "params": {}, "control": None},
                "retina": {"file": None, "n_columns": None, "column_to_bodies": None}, "units": common.UNITS}
    else:
        prov = common.provenance(c, lif, optic, device="cpu", cache_dir=cache_dir,
                                 stimulus={"protocol": "ledger", "params": {"table": str(TABLE_PATH)}, "control": None})
    prov["execution"]["device"] = "cpu"
    prov["execution"]["note"] = "analysis only: the ledger scores finished runs and simulates nothing"
    prov["execution"]["scored_devices"] = _devices(srcs) if len(srcs) else []
    return prov


PRINT_COLUMNS = ["row_id", "label", "stimulus", "quantity", "arm", "measured", "sd", "n", "op", "bound", "status",
                 "battery_status", "z", "null_verdict"]


# --------------------------------------------------------------------------------------------------- validation
#: The files the validation target of docs/INTERP.md section 6 lives in. `scripts/benchmark.py`'s own JSON carries the
#: T4/T5, loom, sugar / bitter and wind numbers with the statuses it wrote; the rotation d' is in the screen CSV.
VALIDATION_SOURCES = ("out/benchmark_suite.json", "out/screen_rotation.csv")

#: (item, the reference in common.VALIDATION['ledger'], the ledger row it is scored by, the arm to quote).
#: `abs` = the reference is the magnitude (screen_rotation.py stores the signed d', Schnell 2010's tuning is a sign).
VALIDATION_MAP = [
    ("motion.min_dsi", "motion.min_dsi", "motion.T4_T5.min_dsi", False),
    # the 34-56 Hz band is session 9's loom DEMO (benchmark.py's `loom_escape.GF_peak_hz`, the comment above that Ref),
    # not the older per-100 ms approach probe `loom.GF_peak_hz`, which reads 30.8 Hz in out/benchmark_suite.json
    ("loom.GF_peak_hz", "loom.GF_peak_hz", "loom.DNp01.rate_peak_hz_demo", False),
    ("rotation d' HSN", ("rotation_dprime", "HSN"), "rotation.HSN.dprime", True),
    ("rotation d' DNp20", ("rotation_dprime", "DNp20"), "rotation.DNp20.dprime", True),
    ("rotation d' HSE", ("rotation_dprime", "HSE"), "rotation.HSE.dprime", True),
    ("bitter.calibrated_sugar_MN9_hz", "bitter.calibrated_sugar_MN9_hz", "taste.MN9.rate_hz_calibrated", False),
    ("bitter.calibrated_sugar_bitter_MN9_hz", "bitter.calibrated_sugar_bitter_MN9_hz",
     "taste.MN9.rate_hz_calibrated_bitter", False),
    ("bitter.shiu_sugar_MN9_hz", "bitter.shiu_sugar_MN9_hz", "taste.MN9.rate_hz_shiu", False),
    ("bitter.shiu_sugar_bitter_MN9_hz", "bitter.shiu_sugar_bitter_MN9_hz", "taste.MN9.rate_hz_shiu_bitter", False),
]


def _reference(ref: dict, key):
    return ref[key[0]][key[1]] if isinstance(key, tuple) else ref.get(key)


def _within(measured, reference, use_abs: bool) -> object:
    """Does the measured number agree with the reference -- a range [lo, hi] contains it, a scalar within 25 % or
    0.5 of it (the object-sweep / figure-stage scatter rule: the ordering is the claim, not the digits)."""
    x = _f(measured)
    if x is None:
        return None
    if use_abs:
        x = abs(x)
    if isinstance(reference, (list, tuple)) and len(reference) == 2:
        return bool(reference[0] <= x <= reference[1])
    r = _f(reference)
    if r is None:
        return None
    return bool(abs(x - r) <= max(0.25 * abs(r), 0.5))


def validate(results=None, *, table=None, c=None, cache_dir=None, null=(), log=print) -> tuple:
    """Reproduce `common.VALIDATION['ledger']` and return `(Result, side_by_side DataFrame)`.

    The target of docs/INTERP.md section 6: the suite's own references -- `motion.min_dsi` 0.16, the loom GF peak
    34-56 Hz, the optomotor d' (HSN 4.2, DNp20 3.1, HSE 2.2) and the sugar / bitter MN9 numbers (4.6 -> 0.0
    calibrated, 123.5 -> 2.1 under Shiu et al.'s rules) -- scored from `scripts/benchmark.py`'s own JSON and
    `scripts/screen_rotation.py`'s CSV, with the status `benchmark.py` wrote for the same key beside each row.
    `status` is `reproduced` only when every row that scores a battery check agrees with it and every reference
    number is matched (`_within`); otherwise it says which rows did not."""
    res = ledger(list(results or [str(ROOT / p) for p in VALIDATION_SOURCES]), table=table, c=c, null=null,
                 cache_dir=cache_dir, generator="flyverse.interp.ledger.validate")
    led = res.table("ledger")
    ref = common.VALIDATION[TOOL]["reference"]
    rows = []
    for item, key, row_id, use_abs in VALIDATION_MAP:
        sel = led[led.row_id.eq(row_id)] if len(led) else led
        r = sel.iloc[0] if len(sel) else None
        reference = _reference(ref, key)
        measured = None if r is None else r.measured
        rows.append({"item": item, "reference": reference, "row_id": row_id, "arm": None if r is None else r.arm,
                     "measured": measured, "abs": use_abs, "n": 0 if r is None else int(r.n),
                     "sd": float("nan") if r is None else r.sd,
                     "status": "MISSING" if r is None else r.status,
                     "battery_status": "" if r is None else r.battery_status,
                     "battery_agrees": None if r is None else r.battery_agrees,
                     "battery_criterion": "" if r is None else r.battery_criterion,
                     "battery_criterion_same": None if r is None else r.battery_criterion_same,
                     "matches_reference": _within(measured, reference, use_abs),
                     "sources": [] if r is None else list(r.sources)})
    side = pd.DataFrame(rows)
    bad_ref = sorted(side.loc[side.matches_reference.ne(True), "item"])
    # a status difference the stored file's own criterion explains is not a disagreement (see _add_battery)
    bad_bat = sorted(side.loc[side.battery_agrees.eq(False) & side.battery_criterion_same.ne(False), "item"])
    status = ("reproduced" if not bad_ref and not bad_bat else
              f"not reproduced (reference: {bad_ref or 'ok'}; battery: {bad_bat or 'ok'})")
    res.add_table("validation", side)
    res.validation = dict(res.validation, measured=side.to_dict("records"), status=status,
                          battery_disagree=res.summary.get("battery_disagree", []),
                          battery_criterion_mismatch=res.summary.get("battery_criterion_mismatch", []))
    if log:
        log(f"[ledger validate] {status}")
    return res, side
