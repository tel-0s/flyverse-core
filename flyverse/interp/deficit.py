"""The toolkit's front door (docs/INTERP.md 10.1; the design defect of section 11.2, item 14): from one ledger row that
fails to steps 1-2 of the procedure, run, and steps 3-7, written out.

    from flyverse import connectome
    from flyverse.interp import deficit
    rep = deficit.diagnose(connectome.load(), "compass.wedge_cells_persisting", out_dir="out/interp/deficit/compass")
    print(deficit.render(rep))

**Step 1 -- name the readout and the expectation.** The row comes from `flyverse/data/expected_responses.csv`, by its
`row_id` or by the `scripts/benchmark.py` check key it mirrors (`check_key`), with its citation, model reference and
`requires` precondition. When finished results are given (`results=`, anything `ledger.ledger` reads), the row's status
is scored from them; otherwise the report says it was not scored and names the row's `gap` flag.

**Step 2 -- structure first.** `paths` from each sensory entry point of the row's stimulus (`STIMULUS_SOURCES`, or
`sources=`) to the readout, k <= 3, and `decompose` statically on the readout -- each writing its own
`flyverse.interp.result/1` JSON, exactly as `scripts/interp_paths.py` / `interp_decompose.py analyse --static` would.
The report condenses them into the three things section 10.1 step 2 says to read: the strongest **silent** link per k,
the readout's E / I per volley and its largest cancelling pair, and whether any route into the readout is signed at
every link.

**Steps 3-7** need a GPU batch, a predeclaration and a person. The report writes them out as command templates with the
row's populations filled in, and runs none of them.

This is a diagnosis aid and never a fix (section 10's opening): nothing here edits a weight, a sign, a gain, a
threshold or a default, and the tools it calls are the shipped ones, unchanged. The two declared tables below
(`STIMULUS_SOURCES`, `READOUT_LABELS` / `BODY_READOUTS`) are choices about where to LOOK, each with its reason;
`sources=` / `readout=` override them, and a row whose stimulus has no entry asks for `sources=` rather than guessing.

Validation (`validate`): on the shipped cache the front door re-finds, from the row alone, the two structural
localizations that were found by hand before any GPU job ran -- GLNO -> PEN (sign 0) as the strongest silent link into
the compass (`docs/audits/cx_glno.md` 1), and DNa02's net-excitatory wiring with LLPC1 as its largest excitatory input
(`docs/audits/deficit_turning.md` 0).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import connectome as cn
from ..motor import LH_ODOUR_CHANNELS, POWER_MN_TYPES
from . import common
from . import decompose as D
from . import ledger as L
from . import paths as P

SCHEMA = "flyverse.interp.deficit/1"
ROOT = common.ROOT

# ---------------------------------------------------------------------------------------------- where to look
# The sensory side of each stimulus, in the population grammar of docs/INTERP.md 2.1. `paths` runs from each entry that
# resolves; one that does not reach the readout within k synapses says so in the report. Two visual entries because the
# photoreceptors reach the optic lobe within 3 synapses and the central brain does not -- the visual projection neurons
# are where the optic lobe hands over.
PHOTORECEPTORS = ("photoreceptors", "class=visual")
VISUAL_PROJECTION = ("visual projection neurons", "superclass=visual_projection")
ORNS = ("ORNs", "class=olfactory")
GUSTATORY = ("gustatory receptor neurons", "class=gustatory")
JOHNSTONS_ORGAN = ("Johnston's organ C / E", "~^JO-(C|E)")                       # senses.Wind's own selection
PROPRIOCEPTORS = ("proprioceptors", "class=mechanosensory_proprioceptive")
RING_ROTATION_INPUTS = ("rotation inputs of the ring", "LNO1|LNO2|LNOa|SpsP|PS196_b|~^LAL")   # scripts/interp_paths.py's
#                                                                                   example; docs/audits/cx_shift.md 1
VISUAL = (PHOTORECEPTORS, VISUAL_PROJECTION)
STIMULUS_SOURCES = {
    **{s: VISUAL for s in ("motion", "loom_left", "loom_demo", "rotation_cw_vs_ccw", "yaw_onset", "apple_left_vs_right",
                           "object_sweep", "figure_ground_apple", "rectangle_height_ladder", "rectangle_width_ladder",
                           "object_size_ladder", "contrast_transitions", "small_dark_object_rdl_disruption")},
    **{s: (ORNS,) for s in ("apple_odour", "apple_8cm", "apple_clean", "odour_paired_with_PPL1_g1pedc")},
    **{s: (GUSTATORY,) for s in ("sugar", "sugar+bitter", "bitter")},
    "wind_left_vs_right": (JOHNSTONS_ORGAN,),
    **{s: (PROPRIOCEPTORS,) for s in ("proprioception_transducer", "leg_cycle")},
    **{s: (RING_ROTATION_INPUTS,) for s in ("compass_room", "compass_pulse_after")},
    # the walking room drives every sense the room has at once
    **{s: (*VISUAL, ORNS, PROPRIOCEPTORS) for s in ("walking", "room_walking")},
}
STIM_SUFFIX = "_stim_150hz"          # the DN screen rows: the stimulated population is the source (screen_dns.py)
LEG_MN = "superclass=vnc_motor&subclass:fl|ml|hl"                                  # motor.motor_groups' leg_L / leg_R

# Ledger population labels that are not population specs (ledger._add_populations skips them): the cells the quantity
# is read from, and where that is defined.
READOUT_LABELS = {
    "wing_power_MN": (POWER_MN_TYPES, "flyverse/motor.py POWER_MN_TYPES: DLMn + DVMn, the wing power readout"),
    "LH_apple_channel": ("|".join(LH_ODOUR_CHANNELS["apple"]), "flyverse/motor.py LH_ODOUR_CHANNELS['apple']"),
}
# Rows whose population is the body: the neural readout that decides the body event (body.py; the take-off rule quoted
# in docs/audits/receptor_integration.md S.1: an escape is the giant fibre >= its threshold, a voluntary take-off is wing
# power >= 50 Hz for 0.3 s). Keyed by the row's quantity.
BODY_READOUTS = {
    "escapes": ("DNp01", "the giant fibre: an escape is DNp01 above the body's gf threshold"),
    "escape_per_1000_fly_s": ("DNp01", "the giant fibre: an escape is DNp01 above the body's gf threshold"),
    "voluntary_per_1000_fly_s": (POWER_MN_TYPES, "wing power MNs: a voluntary take-off is power >= 50 Hz for 0.3 s"),
    "voluntary_takeoffs": (POWER_MN_TYPES, "wing power MNs: a voluntary take-off is power >= 50 Hz for 0.3 s"),
}
MAX_READOUT_CELLS = 5000             # a module-wide readout (rest.brain) is not a decomposition target
MAX_TYPES_SHOWN = 8                  # readout types printed in report.md; report.json keeps them all

# The gaps that already have a localization audit: read it before running anything (INTERP 10.1 step 7 is its shape).
KNOWN_AUDITS = {
    "rotation.DNa02": "docs/audits/deficit_turning.md",
    "compass.": "docs/audits/deficit_rotation.md (and docs/audits/cx_glno.md for GLNO -> PEN)",
    "struct.GLNO_PEN": "docs/audits/cx_glno.md, docs/audits/glno_relabel.md",
    "object.": "docs/audits/deficit_object.md",
    "objsweep.": "docs/audits/deficit_object.md, docs/audits/object_sweep.md",
    "fg.": "docs/audits/deficit_object.md",
}

# Where the recording of step 3 comes from, per stimulus: (script, protocol, arms). None = no record protocol wraps it.
TRACE = "scripts/interp_trace.py"
DECOMPOSE = "scripts/interp_decompose.py"
HEALTH = "scripts/interp_health.py"
RECORD_PROTOCOLS = {
    **{s: (TRACE, "odour") for s in ("apple_odour", "apple_8cm", "apple_clean")},
    **{s: (TRACE, "apple") for s in ("apple_left_vs_right", "figure_ground_apple")},
    "object_sweep": (TRACE, "object"),
    **{s: (DECOMPOSE, "taste") for s in ("sugar", "sugar+bitter", "bitter")},
    "walking": (DECOMPOSE, "walk"),
    "rest": (DECOMPOSE, "rest"),
    **{s: (HEALTH, "compass") for s in ("compass_room", "compass_pulse_after")},
}

VALIDATION = {
    "name": "front door: the two structural localizations found before any GPU job, re-found from the row alone",
    "reference": {
        "compass.EPG.cells_persisting": {"strongest_silent_link": ["GLNO", "PEN_a(PEN1)", "sign0"], "k": 3,
                                         "source": "docs/audits/cx_glno.md 1 (GLNO -> PEN sign 0, 19.4 % of PEN's input)"},
        "rotation.DNa02.rate_hz": {"net_per_volley": "> 0", "top_excitatory_input": "LLPC1",
                                   "source": "docs/audits/deficit_turning.md 0.1 (net +466 / +448 mV per volley) and 0.2 "
                                             "(LLPC1 the strongest DNa02 mover)"},
    },
}


# ---------------------------------------------------------------------------------------------- step 1
def find_row(row: str, table=None) -> pd.Series:
    """The ledger row named by its `row_id`, or by the `scripts/benchmark.py` check key it mirrors."""
    t = L.load_table(table)
    hit = t[t.row_id == row]
    if hit.empty:
        hit = t[t.check_key == row]
    if hit.empty:
        raise KeyError(f"{row!r} is neither a row_id nor a check_key of {t.table_path.iloc[0] if len(t) else table}; "
                       "scripts/interp_deficit.py --list prints them")
    if len(hit) > 1:
        raise KeyError(f"check key {row!r} is mirrored by {len(hit)} rows ({', '.join(hit.row_id)}); name one row_id")
    return hit.iloc[0]


def known_audit(row_id: str) -> str | None:
    for prefix, audit in KNOWN_AUDITS.items():
        if row_id.startswith(prefix):
            return audit
    return None


def status_of(row: pd.Series, results=(), null=(), table=None, c=None) -> list[dict] | None:
    """The row's ledger status per arm from finished results (None when nothing was given), with the status of the row
    it `requires`, if any."""
    if not results:
        return None
    res = L.ledger(list(results), table=table, null=list(null), c=c)
    led = res.table("ledger")
    keep = ["row_id", "arm", "status", "measured", "sd", "n", "runs_passing", "unstable", "battery_status"]
    rows = led[led.row_id.isin([row.row_id] + ([row.requires] if row.requires else []))]
    return common.to_jsonable(rows[[k for k in keep if k in rows.columns]].to_dict("records"))


# ---------------------------------------------------------------------------------------------- where to look
def _stimulated_population(c: cn.Connectome, stimulus: str) -> str:
    """'DNa02_L_stim_150hz' -> 'type=DNa02&somaSide=L'; 'MDN_stim_150hz' -> 'MDN'."""
    tok = stimulus[: -len(STIM_SUFFIX)]
    if len(common.resolve(c, tok)):
        return tok
    m = re.match(r"^(.+)_([LR])$", tok)
    if m and len(common.resolve(c, m.group(1))):
        return f"type={m.group(1)}&somaSide={m.group(2)}"
    return tok


def readout_of(c: cn.Connectome, row: pd.Series, override=None) -> dict:
    """{'spec', 'why', 'kind'} of the cells the row's quantity is read from. kind: 'population' | 'edge' (a structural
    row naming one link) | 'self' (a DN screen row that reads the stimulated population itself)."""
    if override:
        return {"spec": override, "why": "given (readout=)", "kind": "population"}
    pop, stim, q = row.population, row.stimulus, row.quantity
    if "->" in pop:
        pre, post = (s.strip() for s in pop.split("->", 1))
        return {"spec": post, "pre": pre, "why": f"the structural row's own link {pre} -> {post}", "kind": "edge"}
    if stim.endswith(STIM_SUFFIX):
        if q.startswith("leg"):
            return {"spec": LEG_MN, "why": "the leg MNs of motor.motor_groups (the DN screen's leg readout)", "kind": "population"}
        return {"spec": pop, "why": "the stimulated population's own rate", "kind": "self"}
    if pop in READOUT_LABELS:
        spec, why = READOUT_LABELS[pop]
        return {"spec": spec, "why": why, "kind": "population"}
    if pop == "body":
        if q in BODY_READOUTS:
            spec, why = BODY_READOUTS[q]
            return {"spec": spec, "why": why, "kind": "population"}
        raise ValueError(f"row {row.row_id} reads the body ({q}); no neural readout is declared for that quantity -- "
                         "name the cells with readout=")
    return {"spec": pop, "why": "the row's population", "kind": "population"}


def sources_of(c: cn.Connectome, row: pd.Series, readout: dict, override=None) -> list[tuple[str, str]]:
    """[(label, spec)] of the entry points `paths` runs from."""
    if override:
        return [tuple(s) if isinstance(s, (tuple, list)) else (common.spec_repr(s), s) for s in override]
    if readout["kind"] == "edge":
        return [(f"the link's presynaptic side ({readout['pre']})", readout["pre"])]
    if row.stimulus.endswith(STIM_SUFFIX):
        spec = _stimulated_population(c, row.stimulus)
        return [(f"the stimulated population ({spec})", spec)]
    if row.stimulus in STIMULUS_SOURCES:
        return list(STIMULUS_SOURCES[row.stimulus])
    return []


# ---------------------------------------------------------------------------------------------- step 2
def _link(L_: dict | None) -> dict | None:
    if not L_:
        return None
    keep = ("pre", "post", "silent", "silent_partial", "sign", "raw_count", "pairs", "share_of_post_input",
            "mv_per_post_volley_if_signed", "on_path")
    return {k: L_[k] for k in keep if k in L_}


def _walk(w: dict | None) -> dict | None:
    if not w:
        return None
    return {k: w[k] for k in ("path", "gain", "gain_if_signed", "signs", "silent_links", "raw_counts", "unit") if k in w}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:60] or "source"


def as_spec(spec):
    """A spec string as `scripts/interp_paths.py --a/--b` reads it (`paths.spec_from_cli`): 'LNO1|SpsP|~^LAL' is the
    OR list ['LNO1', 'SpsP', '~^LAL']. Passed whole, the same string is ONE unanchored regex on type (it contains a
    metacharacter): it selects every type that merely contains 'SpsP' (IbSpsP ...) and its '~^LAL' branch matches
    nothing."""
    return P.spec_from_cli(spec)


def structure(c: cn.Connectome, readout: dict, sources: list, *, params, k_max: int = 3, top: int = 5,
              out_dir: Path | None = None, cache_dir=None, counts=None, log=print) -> dict:
    """Step 2 on the CPU: `paths` per source and a static `decompose` of the readout, each saved as its own Result
    when `out_dir` is given; returns the condensed record the report reads. `counts` is `paths.raw_counts(c)`'s matrix
    when the caller has it (a synthetic graph must pass it: `raw_counts` would otherwise try to build the shipped
    cache's sign-0 file from the raw weights table)."""
    ro_spec = as_spec(readout["spec"])
    ro_idx = P.resolve_loose(c, ro_spec)
    kinds = common.unit_kinds(c)[ro_idx] if len(ro_idx) else np.array([])
    out = {"readout": {**readout, "n_cells": int(len(ro_idx)),
                       "types": sorted(set(c.neurons.type.fillna(P.UNTYPED).to_numpy()[ro_idx].tolist()))[:20],
                       "unit_kinds": sorted(set(kinds.tolist()))},
           "sources": [], "decompose": None, "readout_inputs": None, "notes": []}
    if not len(ro_idx):
        out["notes"].append(f"the readout {readout['spec']!r} selects no cell in this connectome")
        return out
    if len(ro_idx) > MAX_READOUT_CELLS:
        out["notes"].append(f"the readout is {len(ro_idx):,} cells -- a population-wide quantity, not a decomposition "
                            f"target; read `{HEALTH} structure --by module` instead")
        return out
    if len(kinds) and set(kinds.tolist()) <= {"graded", "photoreceptor"}:
        out["notes"].append("the readout is graded optic-lobe units: `paths` / `decompose` read the LIF's weights, in "
                            "which every optic rate unit is `frozen|pruned` by construction, so a silent flag there is "
                            "not a loss; the optic lobe's own route is `trace` with the column map (INTERP 4.2)")
    ew = common.effective_weights(c, params)
    if counts is None:
        counts, sign0_ok = P.raw_counts(c)
    else:
        counts, sign0_ok = counts.tocsr(), True
    aif = P.if_signed_magnitudes(c, ew, counts)
    if not sign0_ok:
        out["notes"].append("cache/sign0_counts.npz is missing: sign-0 links carry raw count 0 here (run "
                            "`python -m flyverse.connectome`, or any interp tool with the raw weights table present)")
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # every presynaptic type of the readout, pooled -- `paths`' own b_inputs table, with the same flags (the static
    # frozen rule, prune_frozen from params), so a row with no source still gets where its input is silent
    frozen_idx = P.frozen_from("static", c)
    bi = P.inputs_of(c, ew, ro_idx, counts=counts, Aif=aif, frozen_idx=frozen_idx,
                     prune_frozen=bool(getattr(params, "prune_frozen", True)), min_hz=common.NEVER_FIRING_HZ)
    total = float(bi.raw_count.sum()) if len(bi) else 0.0
    silent = bi[bi.silent != ""] if len(bi) else bi
    out["readout_inputs"] = {
        "raw_input_total": total,
        "silent_share": float(silent.raw_count.sum() / total) if total > 0 else None,
        "sign0_share": float(bi.raw_count.mul(bi["sign0_frac"]).sum() / total) if (total > 0 and "sign0_frac" in bi) else 0.0,
        "sign0_counts_available": bool(sign0_ok),
        "top_silent": common.to_jsonable(silent.head(top)[[k for k in ("pre_type", "raw_count", "share_of_b_input", "nt", "silent",
                                                                      "mv_per_post_volley_if_signed") if k in silent.columns]].to_dict("records"))}

    for i, (label, spec) in enumerate(sources):
        rec = {"label": label, "spec": spec}
        try:
            a_spec = as_spec(spec)
            a_idx = P.resolve_loose(c, a_spec)
            rec["n_cells"] = int(len(a_idx))
            if not len(a_idx):
                rec["error"] = "selects no cell"; out["sources"].append(rec); continue
            if np.array_equal(np.sort(a_idx), np.sort(ro_idx)):
                rec["error"] = "is the readout itself"; out["sources"].append(rec); continue
            res = P.paths(c, a_spec, ro_spec, params=params, k_max=k_max, top=top, frozen="static", ew=ew,
                          counts=counts, aif=aif, cache_dir=cache_dir)
        except ValueError as e:
            rec["error"] = str(e); out["sources"].append(rec); continue
        s = res.summary
        n_walks = {int(k): int(v) for k, v in s["n_walks"].items()}
        reached = [k for k, v in n_walks.items() if v]
        rec.update(n_walks=n_walks, reachable_k=min(reached) if reached else None,
                   top_signed={int(k): _walk(v) for k, v in s["top_walk_per_k"].items()},
                   strongest_silent_link={int(k): _link(v) for k, v in s["strongest_silent_link_per_k"].items()},
                   direct=_link(s.get("direct")))
        rec["any_signed_route"] = any(v is not None for v in rec["top_signed"].values())
        if out_dir is not None:
            rec["json"] = str(res.save(out_dir / f"paths_{i}_{_slug(label)}.json"))
        out["sources"].append(rec)
        log(f"  paths {label}: walks per k {n_walks}")

    if readout["kind"] != "edge":
        res = D.decompose(c, ro_spec, params=params, ew=ew, counts=counts, top=top)
        per = res.table("per_type")
        dec = {"by_type": {}}
        for t, rec in res.summary["static"].items():
            sub = per[per.post_type == t]
            dec["by_type"][t] = {**rec,
                                 "top_E": common.to_jsonable(sub[sub.value > 0].nlargest(top, "value")[["pre_group", "value", "raw_count", "silent_entries"]].to_dict("records")),
                                 "top_I": common.to_jsonable(sub[sub.value < 0].nsmallest(top, "value")[["pre_group", "value", "raw_count", "silent_entries"]].to_dict("records"))}
        dec["silent_entries"] = res.summary["silent_entries"]
        dec["unit"] = "mV per post cell per presynaptic volley"
        if out_dir is not None:
            dec["json"] = str(res.save(out_dir / "decompose_static.json"))
        out["decompose"] = dec
        log(f"  decompose static: {len(dec['by_type'])} readout type(s)")
    return out


def answers(st: dict) -> dict:
    """The three things INTERP 10.1 step 2 says to read, over every source."""
    best = {}
    for s in st["sources"]:
        for k, link in (s.get("strongest_silent_link") or {}).items():
            if link and (k not in best or abs(link["mv_per_post_volley_if_signed"]) > abs(best[k]["mv_per_post_volley_if_signed"])):
                best[k] = {**link, "source": s["label"]}
    signed = {s["label"]: next(({"k": k, **w} for k, w in sorted(s["top_signed"].items()) if w), None)
              for s in st["sources"] if "top_signed" in s}
    ei = {t: {k: r[k] for k in ("E_total", "I_total", "net", "cancelling_pair")}
          for t, r in ((st.get("decompose") or {}).get("by_type") or {}).items()}
    return {"strongest_silent_link_per_k": best, "first_signed_route_per_source": signed, "E_I_per_volley": ei}


# ---------------------------------------------------------------------------------------------- steps 3-7
def plan(row: pd.Series, readout: dict, sources: list, name: str) -> list[str]:
    """Steps 3-7 as lines: commands with the row's populations filled in, and the rules that govern them. Runs nothing."""
    ro, src = readout["spec"], (sources[0][1] if sources else "<source spec>")
    if readout["kind"] == "edge":
        return [f"# {row.row_id} is a structural row: step 2 above answers it (the link's raw count, sign and silence flag).",
                "# A counterfactual on the link (a hold, a relabel) is a question for its own predeclared batch, never a fix."]
    lines = [f"# Steps 3-7 for {row.row_id} (docs/INTERP.md 10.1). Nothing below has been run. Predeclare first: protocol,",
             f"# family, gate and source hashes frozen in out/{name}/predeclared.json before submission (INTERP 10.4; PROCESS.md).",
             "", "# 3. ONE cluster batch; the job is the replicate unit; min(n_a, n_b) >= 4 runs, 5 for a small effect."]
    rec = RECORD_PROTOCOLS.get(row.stimulus)
    job_tail = f"> out/{name}/${{a}}_r$s.txt 2>&1; st=\\$?; tail -4 out/{name}/${{a}}_r$s.txt; exit \\$st"
    if rec and rec[0] == TRACE:
        lines += [f'cmds=(); for a in stim ctrl null; do for s in 0 1 2 3 4; do cmds+=("mkdir -p out/{name} && python {TRACE} record '
                  f'--protocol {rec[1]} --arm $a --seed $s --out out/{name}/${{a}}_r$s {job_tail}"); done; done']
    elif rec and rec[0] == DECOMPOSE:
        lines += [f"# The {rec[1]} protocol has no stim / ctrl pair: the matched control and the null are yours to declare (the",
                  "# same protocol under a hold arm, --arm holdBrain / holdKC / ..., or --protocol rest), each as >= 4 jobs.",
                  f'cmds=(); a=default; for s in 0 1 2 3 4; do cmds+=("mkdir -p out/{name} && python {DECOMPOSE} record '
                  f'--target \\"{ro}\\" --protocol {rec[1]} --arm $a --seed $s --out out/{name}/${{a}}_r$s {job_tail}"); done']
    elif rec and rec[0] == HEALTH:
        lines += ["# The ring protocol of docs/audits/cx_glno.md 5 at its defaults; declare the arm that removes the question's input.",
                  f'cmds=(); a=default; for s in 0 1 2 3 4; do cmds+=("mkdir -p out/{name} && python {HEALTH} record '
                  f'--protocol {rec[1]} --seed $s --out out/{name}/${{a}}_r$s {job_tail}"); done']
    else:
        lines += [f"# No record protocol wraps the stimulus {row.stimulus!r}: the recording is the probe of the row's",
                  f"# model reference ({row.model_reference or 'none given'}), run as >= 4 independent jobs per arm."]
    if rec:
        lines += [f'python scripts/cluster_run.py --name {name} --minutes 40 "${{cmds[@]}}" --fetch out/{name}/ 2>&1 | tee out/{name}_cluster.log',
                  f"#    read '<n> job(s), 0 failed', count the artefacts, check every run block's device and arm, verify-batch, THEN analyse."]
    lines += ["", "# 4. Trace, then decompose at the lost stage (CPU); feed a recording back to paths for never_firing flags."]
    if rec and rec[0] == TRACE:
        lines += [f'python {TRACE} analyse --source "{src}" --stimulus "out/{name}/stim_r*.npz" --control "out/{name}/ctrl_r*.npz" '
                  f'--null-runs "out/{name}/null_r*.npz" --stat mean --min-cells 2 --decompose-at "{ro}" --json out/interp/trace/{name}.json']
    elif rec and rec[0] == DECOMPOSE:
        lines += [f'python {DECOMPOSE} analyse --target "{ro}" --recordings "default=out/{name}/default_r*.npz" '
                  f'--null-runs "<control>=out/{name}/<control>_r*.npz" --by type --json out/interp/decompose/{name}.json']
    lines += [f'python scripts/interp_paths.py --a "{src}" --b "{ro}" --k 3 --recording out/{name}/<a stimulus run> --json out/interp/paths/{name}_nf.json',
              "", "# 5. Classify the loss with ONE more batch, only the arms that decide a kind (INTERP 10.1 step 5):",
              "#    wiring      -> hold one presynaptic class onto the lost type at 0 (lesion `edges` kind, interp_apply_object.py)",
              "#    sign        -> a hold table / receptor tier, `decompose contrast`; a counterfactual flip is a question, never a proposal",
              "#    rate model  -> OpticParams overrides (lesion kind `optic`)",
              "#    dynamics    -> the deterministic arm (gain_fb=0) against the shipped feedback, per cell",
              "#    missing input -> never_firing at the source of every wired route + `interp_atlas.py run --populations <source>`"]
    if row.check_key:
        lines += ["", f"# 6. {row.check_key} is a suite check: attribute it with the lesion matrix, >= 4-5 draws where it scatters.",
                  f"python scripts/interp_lesion.py plan --manifest holds --replicates 4 --out-dir out/{name}_lesion"]
    lines += ["", f"# 7. docs/audits/deficit_{_slug(row.row_id.split('.')[0])}.md in the shape of the three shipped ones: the answer first,",
              "#    the data-driven route if one exists, the hand-crafting named and NOT done; `interp_export.py analyse --result <Result>`",
              "#    for every JSON a partner reads; a ledger row for the new localization; an independent skeptic pass."]
    return lines


# ---------------------------------------------------------------------------------------------- the report
def diagnose(c: cn.Connectome, row: str, *, table=None, results=(), null=(), sources=None, readout=None, k_max: int = 3,
             top: int = 5, params=None, out_dir=None, cache_dir=None, name: str | None = None, counts=None, log=print) -> dict:
    """Steps 1-2 run, 3-7 written out, for one ledger row (a row_id or a check key). Returns the report dict; with
    `out_dir`, writes the tools' Result JSONs, report.json and report.md there."""
    from .. import brain
    params = params if params is not None else brain.LIFParams()
    r = find_row(row, table)
    name = name or _slug(r.row_id.replace(".", "-"))
    ro = readout_of(c, r, readout)
    srcs = sources_of(c, r, ro, sources)
    if r.row_id.startswith("walk.power_M") or r.check_key == "walk.power_max_hz":
        log("note: never use walk.power_max as the failing check (INTERP 10.1 step 1; docs/audits/anti_runaway.md round 6)")
    log(f"{r.row_id}: readout {ro['spec']!r} ({ro['why']}); {len(srcs)} source(s)")
    k_max = 1 if ro["kind"] == "edge" else int(k_max)                 # a structural row names one link
    st = structure(c, ro, srcs, params=params, k_max=k_max, top=top,
                   out_dir=out_dir, cache_dir=cache_dir, counts=counts, log=log)
    if not srcs and ro["kind"] != "self":
        st["notes"].append(f"no sensory entry is declared for the stimulus {r.stimulus!r} (deficit.STIMULUS_SOURCES): pass "
                           "sources= / --source LABEL=SPEC to run `paths`; the readout's own inputs are below")
    if ro["kind"] == "self":
        st["notes"].append("the row measures the stimulated population's own rate: structure has nothing upstream to add")
    report = {"schema": SCHEMA,
              "row": {k: (bool(v) if k == "gap" else v) for k, v in r.drop(labels=["table_path"]).to_dict().items()},
              "table": str(r.table_path), "known_audit": known_audit(r.row_id),
              "status": status_of(r, results, null, table, c),
              "structure": st, "answers": answers(st), "plan": plan(r, ro, srcs, name),
              "model": {"lif": common.to_jsonable(common.model_record(params)["lif"]),
                        "shipped": common.to_jsonable(common.model_record(params)["lif"]) == common.to_jsonable(common.model_record(brain.LIFParams())["lif"]),
                        "compiled_connectome": common.connectome_fingerprint(c)},
              "k_max": k_max, "note": "a diagnosis, never a fix (docs/INTERP.md 10): nothing here edits the model"}
    report = common.to_jsonable(report)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        (out_dir / "report.md").write_text(render(report), encoding="utf-8")
        report["files"] = {"report": str(out_dir / "report.json"), "markdown": str(out_dir / "report.md")}
    return report


def _fmt(v, spec="+.3f"):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def render(rep: dict) -> str:
    """The report as Markdown: what the row expects, what the structure says, what to run next."""
    r, st, ans = rep["row"], rep["structure"], rep["answers"]
    out = [f"# {r['row_id']}: {r['label']} under `{r['stimulus']}`", "",
           f"*{rep['note']}.*", "", "## 1. The readout and the expectation", "",
           f"* **Expectation**: {r['quantity']} {r['op']} {r['bound']} {r['unit']} (reference {r['expected'] or '--'}); "
           f"row `gap` {'set -- a KNOWN GAP' if r['gap'] else 'not set'}.",
           f"* **Source**: {r['source'] or '--'}", f"* **Measured in this model by**: {r['model_reference'] or '--'}",
           f"* **Readout**: `{st['readout']['spec']}` -- {st['readout']['why']}; {st['readout']['n_cells']:,} cells "
           f"({', '.join(st['readout']['types'][:8])}{', ...' if len(st['readout']['types']) > 8 else ''})."]
    if r.get("check_key"):
        out.append(f"* **Suite check**: `{r['check_key']}` (scripts/benchmark.py).")
    if r.get("requires"):
        out.append(f"* **Requires** `{r['requires']}`: when that row fails, this one is NOT_APPLICABLE, not a failure.")
    if rep.get("known_audit"):
        out.append(f"* **Already localized**: {rep['known_audit']} -- read it before running anything.")
    if rep.get("status") is None:
        out.append("* **Status**: not scored here (give finished results: `--results`).")
    else:
        for s in rep["status"]:
            out.append(f"* **Status** `{s.get('row_id')}` [{s.get('arm') or '-'}]: {s.get('status')} -- measured "
                       f"{_fmt(s.get('measured'), '.4g')} +- {_fmt(s.get('sd'), '.2g')} over n {s.get('n')}"
                       f"{'; UNSTABLE across runs' if s.get('unstable') else ''}.")
    shipped = (rep.get("model") or {}).get("shipped", True)
    out += ["", f"## 2. Structure (CPU; {'the shipped weights' if shipped else 'the OVERRIDDEN model in report.json `model`, not the shipped one'})", ""]
    for n in st["notes"]:
        out.append(f"> {n}")
    if st["notes"]:
        out.append("")
    ri = st.get("readout_inputs")
    if ri:
        out.append(f"The readout's raw input: {ri['raw_input_total']:,.0f} synapses, **{_fmt(ri['silent_share'], '.1%')} silent**, "
                   f"{_fmt(ri['sign0_share'], '.1%')} sign 0.")
        if ri["top_silent"]:
            out += ["", "| silent input type | raw synapses | share | nt | flag | mV per volley if signed |", "|---|---|---|---|---|---|"]
            out += [f"| {x['pre_type']} | {x['raw_count']:,.0f} | {_fmt(x.get('share_of_b_input'), '.1%')} | {x.get('nt')} | "
                    f"{x.get('silent')} | {_fmt(x.get('mv_per_post_volley_if_signed'))} |" for x in ri["top_silent"]]
        out.append("")
    out += ["**(a) The strongest silent link per k** (largest |if-signed mV per post volley| on a listed silent walk):", ""]
    if ans["strongest_silent_link_per_k"]:
        out += ["| k | link | flag | raw synapses | share of post input | mV if signed | from source |", "|---|---|---|---|---|---|---|"]
        for k, x in sorted(ans["strongest_silent_link_per_k"].items(), key=lambda kv: int(kv[0])):
            out.append(f"| {k} | {x['pre']} -> {x['post']} | {x.get('silent')} | {_fmt(x.get('raw_count'), ',.0f')} | "
                       f"{_fmt(x.get('share_of_post_input'), '.1%')} | {_fmt(x.get('mv_per_post_volley_if_signed'))} | {x['source']} |")
    else:
        out.append("none on any listed walk.")
    out += ["", "**(b) E / I per volley at the readout** (static; mV per post cell per presynaptic volley):", ""]
    dec = st.get("decompose") or {}
    name = lambda g: g or P.UNTYPED                                                       # noqa: E731
    by_type = list((dec.get("by_type") or {}).items())
    for t, x in by_type[:MAX_TYPES_SHOWN]:
        cp = x.get("cancelling_pair")
        out.append(f"* `{t}`: E {x['E_total']:+.1f} / I {x['I_total']:+.1f} = net **{x['net']:+.1f}**"
                   + (f"; largest cancelling pair {name(cp['E'])} ({cp['E_value']:+.1f}) vs {name(cp['I'])} ({cp['I_value']:+.1f})" if cp else "")
                   + ". Top E: " + ", ".join(f"{name(e['pre_group'])} {e['value']:+.1f}" for e in x["top_E"][:5])
                   + ". Top I: " + ", ".join(f"{name(i['pre_group'])} {i['value']:+.1f}" for i in x["top_I"][:5]) + ".")
    if len(by_type) > MAX_TYPES_SHOWN:
        out.append(f"* ... {len(by_type) - MAX_TYPES_SHOWN} more readout types in report.json (`structure.decompose.by_type`).")
    if not dec:
        out.append("not run (see the notes above).")
    out += ["", "**(c) Is any route into the readout signed at every link?**", ""]
    for s in st["sources"]:
        if "error" in s:
            out.append(f"* {s['label']} (`{s['spec']}`): {s['error']}.")
            continue
        first = ans["first_signed_route_per_source"].get(s["label"])
        if first:
            out.append(f"* {s['label']}: **yes**, first at k {first['k']}: `{first['path']}` (gain {first['gain']:+.4g} {first.get('unit', '')}).")
        elif s.get("reachable_k"):
            out.append(f"* {s['label']}: **no** -- every listed walk (from k {s['reachable_k']}) carries a silent link.")
        else:
            out.append(f"* {s['label']}: no walk within k {rep['k_max']}.")
    if not st["sources"]:
        out.append("no source was run.")
    out += ["", "## 3-7. What to run next", "", "```bash", *rep["plan"], "```", ""]
    files = [s["json"] for s in st["sources"] if s.get("json")] + ([dec["json"]] if dec.get("json") else [])
    if files:
        out += ["Result JSONs (`flyverse.interp.result/1`): " + ", ".join(f"`{f}`" for f in files), ""]
    return "\n".join(out)


def validate(c: cn.Connectome, *, out_dir=None, log=print) -> dict:
    """VALIDATION on the shipped cache: returns {'status': 'reproduced' | 'not reproduced', 'measured': {...}}."""
    measured, ok = {}, True
    rep = diagnose(c, "compass.EPG.cells_persisting", out_dir=None if out_dir is None else Path(out_dir) / "compass", log=log)
    link = rep["answers"]["strongest_silent_link_per_k"].get("3") or {}     # JSON keys: the report is to_jsonable'd
    got = [link.get("pre"), link.get("post"), link.get("silent")]
    measured["compass.EPG.cells_persisting"] = {"strongest_silent_link_k3": got, "mv_if_signed": link.get("mv_per_post_volley_if_signed")}
    ok &= got == VALIDATION["reference"]["compass.EPG.cells_persisting"]["strongest_silent_link"]
    rep = diagnose(c, "rotation.DNa02.rate_hz", out_dir=None if out_dir is None else Path(out_dir) / "turning", log=log)
    x = rep["structure"]["decompose"]["by_type"].get("DNa02", {})
    top_e = x["top_E"][0]["pre_group"] if x.get("top_E") else None
    measured["rotation.DNa02.rate_hz"] = {"E_total": x.get("E_total"), "I_total": x.get("I_total"), "net": x.get("net"), "top_E": top_e}
    ok &= bool(x) and x["net"] > 0 and top_e == VALIDATION["reference"]["rotation.DNa02.rate_hz"]["top_excitatory_input"]
    return {**VALIDATION, "measured": measured, "status": "reproduced" if ok else "not reproduced"}
