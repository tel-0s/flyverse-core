"""Audit of the rate optic lobe's hand-crafted measures (flyverse/optic.py), the way docs/audits/anti_runaway.md audited
the LIF's: each measure removed alone (OpticParams overrides only; optic.py is not edited), scored on

  * scripts/benchmark.py --sections walk,a,b,walk_gf   (motion DSI, the pinned loom + walking of the legacy `walk`
    section, the demo loom-escape probe [b], walking GF p99 [walk_gf]; 2 seeds where seeds apply = section b),
  * scripts/probe_figure_stages.py --stimulus both       (the per-stage figure map: static apple + moving ball, with the
    none-vs-none null), 2 seeds (3 for the baseline),
  * scripts/probe_object_sweep.py (ball vs none) and --null (none vs none), 2 seeds (3 for the baseline),

one cluster job per configuration, all in ONE batch (`--batch --submit`).  The measures and what each stands in for are
listed in docs/audits/optic_measures.md; `--check` prints, per configuration, the effective OpticParams and the
structure of the edges it changes (edges, |W|, presynaptic transmitter, the receptor table's sign) on the CPU.

    python scripts/audit_optic.py --list
    python scripts/audit_optic.py --check baseline,no_t4_pair_gain            # CPU, no simulation
    python scripts/audit_optic.py --batch [--submit] [--minutes 120]           # print / submit the cluster batch
    python scripts/audit_optic.py --one CFG --what bench,stages,object --seeds 0,1 --out out/optic_audit   # one job (GPU)
    python scripts/audit_optic.py --report out/optic_audit                     # tables -> <dir>/report.md / report.json (CPU)

The benchmark half reuses scripts/retire_measures.py's machinery (its Context applies OpticParams overrides through
benchmark.Context._apply_optic and records `config.pair_gain`); the configurations below are registered into
retire_measures.CONFIGS in-process, so that file is not edited.  The two probes take the overrides through
probe_figure_stages.install_optic_overrides (every optic.OpticParams built by room_demo.Sim carries them).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flyverse import optic  # noqa: E402

SECTIONS = "walk,a,b,walk_gf"
T4_IN = (r"^(Mi4|Mi9|CT1|C3)$", r"^T4[abcd]$")
T5_IN = (r"^(Tm4|Tm9|CT1|TmY15)$", r"^T5[abcd]$")
LPI = (r"^LPi(34|43)$", r"^LPLC2$")
T4T5_OUT = (r"^T[45][abcd]$", r".*")
LC_IN = (r".*", r"^(LC4|LPLC2)$")


def pair_gain_with(changes: dict):
    """optic.DEFAULT_PAIR_GAIN with the factors of the named (pre, post) entries replaced; every entry must exist once."""
    out, seen = [], set()
    for pre, post, g in optic.DEFAULT_PAIR_GAIN:
        if (pre, post) in changes:
            seen.add((pre, post)); out.append((pre, post, float(changes[(pre, post)])))
        else:
            out.append((pre, post, g))
    missing = set(changes) - seen
    if missing:
        raise SystemExit(f"optic.DEFAULT_PAIR_GAIN lacks {missing}")
    return out


# name -> {"measure", "kind", "optic": OpticParams overrides (field -> value or callable of nothing), "note"}
CONFIGS = {
    "baseline": {"measure": "-", "kind": "baseline", "optic": {}, "note": "shipped OpticParams / DEFAULT_PAIR_GAIN / DEFAULT_TAU_BY_TYPE / DEFAULT_BASELINE_BY_TYPE"},
    # pair gains (optic.DEFAULT_PAIR_GAIN)
    "no_t4_pair_gain": {"measure": "pair gain Mi4/Mi9/CT1/C3 -> T4 x5", "kind": "ablation", "optic": lambda: {"pair_gain": pair_gain_with({T4_IN: 1.0})},
                        "note": "the T4 delayed-inhibition arm back to the uniform synapse (session 3: needed for DSI)"},
    "no_t5_pair_gain": {"measure": "pair gain Tm4/Tm9/CT1/TmY15 -> T5 x5", "kind": "ablation", "optic": lambda: {"pair_gain": pair_gain_with({T5_IN: 1.0})},
                        "note": "the T5 arm back to x1 (Tm4 / Tm9 are cholinergic: the entry amplifies excitation too)"},
    "pair_gain_lpi_x1": {"measure": "pair gain LPi34/43 -> LPLC2 x4", "kind": "ablation", "optic": lambda: {"pair_gain": pair_gain_with({LPI: 1.0})},
                         "note": "LPi -> LPLC2 back to x1 (session 9's self-motion escape fix); re-scanned under the shipped gains"},
    "pair_gain_lpi_x2": {"measure": "pair gain LPi34/43 -> LPLC2 x4", "kind": "replacement", "optic": lambda: {"pair_gain": pair_gain_with({LPI: 2.0})},
                         "note": "the round-4 'only passing point' at x2, under the shipped gains (handover item 4)"},
    "no_t4t5_out_gain": {"measure": "pair gain T4/T5 -> * x2", "kind": "ablation", "optic": lambda: {"pair_gain": pair_gain_with({T4T5_OUT: 1.0})},
                         "note": "T4 / T5 output back to x1 (session 3: 'rectified DS units respond weakly to natural scenes')"},
    "no_pair_gain": {"measure": "all pair gains", "kind": "ablation", "optic": {"pair_gain": []},
                     "note": "every DEFAULT_PAIR_GAIN entry off (the .* -> LC4/LPLC2 x1 entry is a no-op either way)"},
    # per-type operating points and time constants
    "no_t4t5_rectify": {"measure": "baseline_by_type T4/T5 = 0 (ReLU)", "kind": "ablation", "optic": {"baseline_by_type": {}},
                        "note": "T4 / T5 back to the linear operating point 0.5 (session 3: the ReLU is what makes them DS)"},
    "no_tau_by_type": {"measure": "tau_by_type (Mi4/Mi9/CT1/Tm9 150 ms, L1/L2 6, Mi1/Tm3/Tm1/2/4 8, L3 40)", "kind": "ablation",
                       "optic": {"tau_by_type": {}}, "note": "every rate unit at tau_ms 10 (session 2-3: the T4/T5 delay line)"},
    # normalisation, output stage
    "norm_l1": {"measure": "input normalisation norm = l2", "kind": "replacement", "optic": {"norm": "l1"},
                "note": "optic-lobe weights as fractions of total input (l1) instead of L2 (coherent fan-in amplified ~sqrt N)"},
    "out_norm_l2": {"measure": "out_norm = l1", "kind": "replacement", "optic": {"out_norm": "l2"},
                    "note": "optic -> spiking weights L2-normalised (NOTES 2: 'L2-normalised output weights make LC4/LPLC2 fire from walking flow')"},
    "gain_out_80": {"measure": "gain_out_mv = 100", "kind": "replacement", "optic": {"gain_out_mv": 80.0},
                    "note": "session 2's value ('80 left LC4/LPLC2 below threshold for the loom')"},
    "gain_out_120": {"measure": "gain_out_mv = 100", "kind": "replacement", "optic": {"gain_out_mv": 120.0},
                     "note": "the value NOTES 2 says fires LC4/LPLC2 from walking flow"},
    "no_drive_clip": {"measure": "drive_clip_mv = 35", "kind": "ablation", "optic": {"drive_clip_mv": 1e9},
                      "note": "the +-35 mV clip on the injected current removed"},
    # dynamics
    "no_optic_adapt": {"measure": "adapt_gain 1 / adapt_tau_ms 400", "kind": "ablation", "optic": {"adapt_gain": 0.0},
                       "note": "the slow relaxation of every rate unit towards its operating point off (after-images stay)"},
    "no_spk_feedback": {"measure": "gain_fb = 0.5", "kind": "ablation", "optic": {"gain_fb": 0.0},
                        "note": "spiking -> optic-lobe feedback off (is the object-sweep noise floor the LIF's feedback?)"},
    "gain_in_1": {"measure": "gain_in = 3", "kind": "replacement", "optic": {"gain_in": 1.0},
                  "note": "photoreceptor contrast -> lamina at x1 instead of x3 ('lamina cells saturate at ~20 % contrast')"},
}
BASELINE_SEEDS = "0,1,2"
SEEDS = "0,1"


def optic_overrides(name):
    cfg = CONFIGS[name]
    o = cfg["optic"]
    return copy.deepcopy(o() if callable(o) else o)


def effective_params(name):
    p = optic.OpticParams()
    for k, v in optic_overrides(name).items():
        setattr(p, k, v)
    return p


# ---- --check: structure of what a configuration changes (CPU) --------------------------------------------------
def pair_gain_structure(c, pair_gain, receptor=None):
    """Per pair-gain entry: rate -> rate edges matched, |W|, by presynaptic transmitter, and the share of the targets'
    L2-normalised input the entry multiplies; `receptor` = brain._receptor(...) gives the effective fast sign."""
    import scipy.sparse as sp
    from flyverse.connectome import PHOTORECEPTOR_TYPES
    nrn = c.neurons; types = nrn.type.fillna("").to_numpy(); nt = nrn.nt.fillna("").to_numpy()
    is_pr = nrn.type.isin(PHOTORECEPTOR_TYPES).to_numpy()
    rate = (nrn.superclass == "ol_intrinsic").to_numpy() & ~is_pr
    W = c.W.tocsr(); coo = W.tocoo()
    sign = np.sign(coo.data) if receptor is None else receptor.fast_sign
    l2 = nrn.in_syn_l2.to_numpy()
    rows = []
    for pre_re, post_re, f in pair_gain:
        pre_m = np.array([bool(re.match(pre_re, t)) for t in types]); post_m = np.array([bool(re.match(post_re, t)) for t in types])
        sel = pre_m[coo.col] & post_m[coo.row]
        sel_rr = sel & rate[coo.col] & rate[coo.row]                      # W_rr entries (rate -> rate)
        sel_sr = sel & rate[coo.col] & ~rate[coo.row] & ~is_pr[coo.row]    # W_sr entries (rate -> spiking)
        by_nt = {}
        for m, lab in ((sel_rr, "rate->rate"), (sel_sr, "rate->spiking")):
            for t in sorted(set(nt[coo.col[m]])):
                mm = m & (nt[coo.col] == t)
                by_nt[f"{lab} {t}"] = {"edges": int(mm.sum()), "absW": float(np.abs(coo.data[mm]).sum()),
                                       "exc_edges": int((sign[mm] > 0).sum()), "inh_edges": int((sign[mm] < 0).sum()), "zero_edges": int((sign[mm] == 0).sum())}
        # share of the post targets' L2-normalised |input| that the entry multiplies (rate -> rate part)
        post_cells = np.flatnonzero(post_m & rate)
        tot = np.abs(coo.data[rate[coo.col] & post_m[coo.row]] / l2[coo.row[rate[coo.col] & post_m[coo.row]]]).sum()
        part = np.abs(coo.data[sel_rr] / l2[coo.row[sel_rr]]).sum()
        pre_types = sorted(set(types[coo.col[sel_rr | sel_sr]]))
        rows.append({"pre": pre_re, "post": post_re, "factor": f, "edges_rr": int(sel_rr.sum()), "edges_sr": int(sel_sr.sum()),
                     "absW_rr": float(np.abs(coo.data[sel_rr]).sum()), "absW_sr": float(np.abs(coo.data[sel_sr]).sum()),
                     "post_rate_cells": int(len(post_cells)), "share_of_post_l2_input_rr": float(part / tot) if tot else None,
                     "pre_types": pre_types[:40], "by_nt": by_nt})
    return rows


def check(names):
    from flyverse import brain, connectome
    c = connectome.load(verbose=False)
    lif = brain.LIFParams()
    rs = brain._receptor(c, lif, with_counts=False)
    base = optic.OpticParams()
    out = {}
    for name in names:
        p = effective_params(name)
        diff = {k: getattr(p, k) for k in vars(p) if getattr(p, k) != getattr(base, k)}
        rec = {"name": name, "measure": CONFIGS[name]["measure"], "kind": CONFIGS[name]["kind"], "note": CONFIGS[name]["note"],
               "optic_overrides": {k: (v if not callable(v) else str(v)) for k, v in diff.items()}}
        pg = optic.DEFAULT_PAIR_GAIN if p.pair_gain is None else p.pair_gain
        rec["pair_gain"] = pg
        rec["pair_gain_structure"] = pair_gain_structure(c, pg, rs)
        rec["tau_by_type"] = optic.DEFAULT_TAU_BY_TYPE if p.tau_by_type is None else p.tau_by_type
        rec["baseline_by_type"] = optic.DEFAULT_BASELINE_BY_TYPE if p.baseline_by_type is None else p.baseline_by_type
        out[name] = rec
        print(f"\n== {name}: {rec['measure']} ({rec['kind']}) -- {rec['note']}")
        print("   OpticParams changed:", rec["optic_overrides"] or "(none)")
        for r in rec["pair_gain_structure"]:
            print(f"   pair gain {r['pre']} -> {r['post']} x{r['factor']}: {r['edges_rr']:,} rate->rate edges / {r['absW_rr']:,.0f} |W| "
                  f"({(r['share_of_post_l2_input_rr'] or 0) * 100:.1f} % of the post types' L2-normalised rate input), "
                  f"{r['edges_sr']:,} rate->spiking edges / {r['absW_sr']:,.0f} |W|; pre types {r['pre_types'][:12]}")
            for k, v in r["by_nt"].items():
                print(f"       {k:28s} edges {v['edges']:7,d}  |W| {v['absW']:10,.0f}  sign under the shipped model: +{v['exc_edges']:,} / -{v['inh_edges']:,} / 0 {v['zero_edges']:,}")
    return out


# ---- --one: one configuration on the GPU ---------------------------------------------------------------------
def _import_retire_measures():
    """scripts/retire_measures.py references `brain` at import time (its round-5 `gf_damped` entry, line 245) without
    importing it, so a plain `import retire_measures` raises NameError.  That file is not this task's to edit; the module is
    loaded here with `brain` pre-bound in its namespace (the same object it would import), which changes nothing else."""
    if "retire_measures" in sys.modules:
        return sys.modules["retire_measures"]
    import importlib.util
    from flyverse import brain
    spec = importlib.util.spec_from_file_location("retire_measures", os.path.join(os.path.dirname(os.path.abspath(__file__)), "retire_measures.py"))
    rm = importlib.util.module_from_spec(spec)
    rm.brain = brain
    sys.modules["retire_measures"] = rm
    spec.loader.exec_module(rm)
    return rm


def run_bench(name, seeds, out_dir):
    rm = _import_retire_measures()
    cfg = CONFIGS[name]
    rm.CONFIGS[name] = {"measure": cfg["measure"], "kind": cfg["kind"], "note": cfg["note"], "optic": (lambda: optic_overrides(name))}
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    rm.run_one(name, SECTIONS, False, d, seeds, "default", "abs")       # writes <d>/<name>.json
    return os.path.join(d, f"{name}.json")


def run_stages(name, seeds, out_dir):
    import probe_figure_stages as pfs
    pfs.install_optic_overrides(optic_overrides(name))
    d = os.path.join(out_dir, name)
    for s in seeds.split(","):
        pfs.main(["--stimulus", "both", "--seed", s, "--out", os.path.join(d, f"stages_s{s}.json")])


def run_object(name, seeds, out_dir):
    import probe_figure_stages as pfs
    import probe_object_sweep as pos
    pfs.install_optic_overrides(optic_overrides(name))
    d = os.path.join(out_dir, name)
    for s in seeds.split(","):
        for null in (False, True):
            argv = ["--seed", s, "--out", os.path.join(d, f"obj_{'null' if null else 'ball'}_s{s}.json")] + (["--null"] if null else [])
            sys.argv = ["probe_object_sweep.py"] + argv
            pos.main()


def run_one(name, what, seeds, out_dir):
    import torch
    assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    print(f"audit_optic --one {name}: {CONFIGS[name]['measure']} ({CONFIGS[name]['kind']}); overrides {optic_overrides(name)}; "
          f"seeds {seeds}; device {torch.cuda.get_device_name(0)}; torch {torch.__version__}", flush=True)
    t0 = time.time()
    for w in what.split(","):
        t1 = time.time()
        if w == "bench":
            run_bench(name, seeds, out_dir)
        elif w == "stages":
            run_stages(name, seeds, out_dir)
        elif w == "object":
            run_object(name, seeds, out_dir)
        else:
            raise SystemExit(f"unknown --what {w}")
        print(f"[{name} {w}: {(time.time() - t1) / 60:.1f} min]", flush=True)
    print(f"[{name} done: {(time.time() - t0) / 60:.1f} min]", flush=True)


# ---- --batch: the cluster commands ----------------------------------------------------------------------------
def batch_commands(out_dir="out/optic_audit", names=None):
    cmds = []
    for name in (names or list(CONFIGS)):
        seeds = BASELINE_SEEDS if name == "baseline" else SEEDS
        log = f"{out_dir}/{name}.txt"
        cmds.append(f"python -c 'import torch; assert torch.cuda.is_available()' && mkdir -p {out_dir} && "
                    f"python scripts/audit_optic.py --one {name} --what bench,stages,object --seeds {seeds} --out {out_dir} > {log}; cat {log}")
    return cmds


# ---- --report ------------------------------------------------------------------------------------------------
CHECKS = ["motion.min_dsi", "motion.correct_directions", "loom_escape.GF_peak_hz", "loom_escape.escapes", "walk_gf.p99_hz",
          "walk.GF_max_hz", "walk.power_max_hz", "walk.power_sustained_hz", "loom.GF_peak_hz", "loom.escape_cm", "rotate.DNp20_flip_hz"]
OBJ_SPIKING = ["LC11", "LC10a", "LC10b", "LC16", "LPLC2", "LC4"]
OBJ_RATE = ["Mi4", "Mi1", "Tm3", "Tm5Y", "TmY21", "T2", "T3", "TmY13", "TmY5a"]


def _load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _msd(x):
    x = np.array([v for v in x if v is not None and v == v], float)
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"), int(len(x))


def report(out_dir, targets):
    names = [n for n in CONFIGS if os.path.isdir(os.path.join(out_dir, n))]
    lines = [f"# Optic-measure audit ({out_dir})", "", f"Generated by scripts/audit_optic.py --report on {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    summary = {}
    # 1. benchmark checks
    lines += ["## 1. Benchmark sections walk / a (motion) / b (loom_escape) / c (walk_gf), per configuration", "",
              "| config | measure | " + " | ".join(CHECKS) + " | pass/fail/gap | runtime min |", "|" + "---|" * (len(CHECKS) + 4)]
    for n in names:
        r = _load(os.path.join(out_dir, n, f"{n}.json"))
        if r is None:
            lines.append(f"| {n} | {CONFIGS[n]['measure']} | " + " | ".join("--" for _ in CHECKS) + " | missing | -- |"); continue
        st = {c["key"]: c for c in r["checks"]}
        cells = []
        for k in CHECKS:
            c = st.get(k)
            if c is None or c["measured"] is None:
                cells.append("--"); continue
            v = c["measured"]; s = c["status"]
            cells.append((f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else f"{v:.1f}" if isinstance(v, float) else str(v)) + ("" if s.startswith("PASS") else f" {s}"))
        tally = f"{sum(c['status'].startswith('PASS') for c in r['checks'])}/{sum(c['status'] == 'FAIL' for c in r['checks'])}/{sum(c['status'] == 'KNOWN GAP' for c in r['checks'])}"
        lines.append(f"| {n} | {CONFIGS[n]['measure']} | " + " | ".join(cells) + f" | {tally} | {r['total_runtime_s'] / 60:.1f} |")
        summary.setdefault(n, {})["checks"] = {k: (st[k]["measured"], st[k]["status"]) for k in st}
        summary[n]["motion_subtypes"] = {t: v["dsi"] for t, v in r["sections"].get("motion", {}).get("subtypes", {}).items()} if "motion" in r["sections"] else {}
        summary[n]["loom_escape_seeds"] = r["sections"].get("loom_escape", {}).get("seeds")
    # per-subtype DSI
    lines += ["", "Per-subtype DSI (section a; the check is the minimum):", "", "| config | " + " | ".join(optic.T4T5) + " |", "|" + "---|" * 9]
    for n in names:
        d = summary.get(n, {}).get("motion_subtypes") or {}
        lines.append(f"| {n} | " + " | ".join(f"{d[t]:.3f}" if t in d else "--" for t in optic.T4T5) + " |")
    # 2. stage maps
    lines += ["", "## 2. Figure-propagation map per stage (probe_figure_stages; A = stimulus, B / C = none)", "",
              "Per configuration and stimulus, per stage: types scored; types with |z(A-B)| >= 3 in EVERY seed on the signed / abs measure "
              "(vs the same count for the none-vs-none pair C-B); the best type by mean |z(A-B)| with its mean z(A-B) and z(C-B); "
              "the null-referenced z_null = (mean figure(A-B) - mean figure(C-B)) / SD(figure(C-B) over seeds) needs >= 3 seeds.", ""]
    stage_summary = {}
    for n in names:
        d = os.path.join(out_dir, n)
        runs = [_load(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith("stages_s") and f.endswith(".json")]
        runs = [r for r in runs if r]
        if not runs:
            continue
        stages = runs[0]["config"]["stages"]
        for stim in ("apple", "ball"):
            rs = [r[stim] for r in runs if stim in r]
            if not rs:
                continue
            types = sorted(set.intersection(*[set(r["types"]) for r in rs]))
            per_type = {}
            for t in types:
                recs = [r["types"][t] for r in rs]
                rec = {"stage": recs[0]["stage"], "kind": recs[0]["kind"], "cells_obj": recs[0]["cells_obj"], "cells_bg": recs[0]["cells_bg"]}
                for meas in ("signed", "abs"):
                    zAB = [x[meas]["AB"]["z"] for x in recs]; zCB = [x[meas]["CB"]["z"] for x in recs]
                    fAB = [x[meas]["AB"]["figure"] for x in recs]; fCB = [x[meas]["CB"]["figure"] for x in recs]
                    mA, sA, k = _msd(fAB); mC, sC, _ = _msd(fCB)
                    rec[meas] = {"zAB": zAB, "zCB": zCB, "zAB_mean": float(np.mean(zAB)), "zCB_mean": float(np.mean(zCB)),
                                 "abs_zAB_min": float(np.min(np.abs(zAB))), "abs_zCB_max": float(np.max(np.abs(zCB))),
                                 "figAB_mean": mA, "figCB_mean": mC, "figCB_sd": sC, "n": k,
                                 "z_null": float((mA - mC) / sC) if (sC == sC and sC > 1e-9 and k >= 3) else None,
                                 "carries": bool(np.min(np.abs(zAB)) >= 3 and len(set(np.sign(zAB))) == 1 and np.min(np.abs(zAB)) > np.max(np.abs(zCB)))}
                per_type[t] = rec
            stage_summary.setdefault(n, {})[stim] = per_type
            lines += [f"### {n} / {stim} ({len(rs)} seeds; {rs[0]['columns']['columns_obj']} object / {rs[0]['columns']['columns_bg']} background columns)", "",
                      "| stage | types | carry signed / abs (all seeds |z|>=3, consistent sign, above own null) | null |z(C-B)|>=3 signed / abs | best type (signed): z(A-B) mean, z(C-B) mean, z_null | best type (abs): z(A-B), z(C-B), z_null |",
                      "|---|---|---|---|---|---|"]
            for s in sorted(int(k) for k in stages):
                rows = {t: r for t, r in per_type.items() if r["stage"] == s}
                if not rows:
                    continue
                cs = sum(r["signed"]["carries"] for r in rows.values()); ca = sum(r["abs"]["carries"] for r in rows.values())
                ns = sum(any(abs(z) >= 3 for z in r["signed"]["zCB"]) for r in rows.values()); na = sum(any(abs(z) >= 3 for z in r["abs"]["zCB"]) for r in rows.values())
                bs = max(rows, key=lambda t: abs(rows[t]["signed"]["zAB_mean"])); ba = max(rows, key=lambda t: abs(rows[t]["abs"]["zAB_mean"]))
                f = lambda r: f"{r['zAB_mean']:+.1f}, {r['zCB_mean']:+.1f}, " + (f"{r['z_null']:+.1f}" if r["z_null"] is not None else "n/a")
                lines.append(f"| {stages[str(s)]} | {len(rows)} | {cs} / {ca} | {ns} / {na} | {bs}: {f(rows[bs]['signed'])} | {ba}: {f(rows[ba]['abs'])} |")
            # the named small-field / loom types
            lines += ["", "Named types (mean over seeds; signed z(A-B) / z(C-B); abs z(A-B) / z(C-B)):", ""]
            named = ["L1", "L2", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "Tm20", "Tm5Y", "TmY21", "TmY13", "TmY5a", "TmY3", "Y3", "T2", "T2a", "T3",
                     "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "LPi34", "LPi43", "Li19", "LC11", "LC10a", "LC10b", "LC16", "LPLC2", "LC4", "LC17", "LC12"]
            lines.append("| type | stage | kind | signed z(A-B) / z(C-B) | abs z(A-B) / z(C-B) | carries (signed / abs) |")
            lines.append("|---|---|---|---|---|---|")
            for t in named:
                if t in per_type:
                    r = per_type[t]
                    lines.append(f"| {t} | {r['stage']} | {r['kind']} | {r['signed']['zAB_mean']:+.1f} / {r['signed']['zCB_mean']:+.1f} | "
                                 f"{r['abs']['zAB_mean']:+.1f} / {r['abs']['zCB_mean']:+.1f} | {r['signed']['carries']} / {r['abs']['carries']} |")
            lines.append("")
    # 3. object sweep
    lines += ["## 3. Object sweep (probe_object_sweep): (ball - none) vs the none-vs-none null, per configuration", "",
              "Spiking types: diff_max_over_cells_mean_mv (mV). Rate units: diff_abs_best_cell_mean (rate units). Per seed values ball / null; "
              "z = (mean ball - mean null) / SD(null) (n null runs as given; a z at n = 2 is not a result).", ""]
    obj_summary = {}
    for n in names:
        d = os.path.join(out_dir, n)
        balls = {int(m.group(1)): _load(os.path.join(d, f)) for f in os.listdir(d) for m in [re.match(r"obj_ball_s(\d+)\.json$", f)] if m}
        nulls = {int(m.group(1)): _load(os.path.join(d, f)) for f in os.listdir(d) for m in [re.match(r"obj_null_s(\d+)\.json$", f)] if m}
        if not balls:
            continue
        lines += [f"### {n} ({len(balls)} ball, {len(nulls)} null runs)", "", "| type | stat | ball per seed | mean | null per seed | null mean | null SD | z |", "|---|---|---|---|---|---|---|---|"]
        obj_summary[n] = {}
        for t in OBJ_SPIKING + OBJ_RATE:
            stat = "diff_max_over_cells_mean_mv" if t in OBJ_SPIKING else "diff_abs_best_cell_mean"
            b = [balls[s]["ball"][t][stat] for s in sorted(balls) if balls[s] and t in balls[s]["ball"]]
            u = [nulls[s]["ball"][t][stat] for s in sorted(nulls) if nulls[s] and t in nulls[s]["ball"]]
            mb, sb, kb = _msd(b); mu, su, ku = _msd(u)
            z = (mb - mu) / su if (su == su and su > 1e-6) else float("nan")      # a zero null SD (deterministic lobe) has no z
            obj_summary[n][t] = {"ball": b, "null": u, "ball_mean": mb, "null_mean": mu, "null_sd": su, "z": z}
            lines.append(f"| {t} | {stat} | {' '.join(f'{v:+.4f}' for v in b)} | {mb:+.4f} | {' '.join(f'{v:+.4f}' for v in u)} | {mu:+.4f} | {su:.4f} | {z:+.1f} |")
        # the per-run verdicts and LPLC2 / LC11 rates
        v = [balls[s]["verdict"]["pass"] for s in sorted(balls) if balls[s]]
        lines += ["", f"per-run 7 mV / 1 Hz verdict PASS in {sum(v)} of {len(v)} ball runs (a coin flip at the noise floor; magnitudes above are the result)", ""]
    # 4. connectivity between the carrying and the losing stage (baseline, CPU)
    if targets:
        lines += ["## 4. Connectivity into the losing stage (baseline weights, CPU)", "",
                  "For each target type: its inputs by type as the share of its L2-normalised rate-unit input (|W| / in_syn_l2, summed over the "
                  "type's cells and divided by the type's total), the sign under the shipped receptor model (+ / - / 0 edges), the pair-gain factor "
                  "in force, and the input type's baseline figure z (apple signed, ball abs; mean over seeds).  `linear estimate` = "
                  "sum over inputs of (signed share x input figure(A-B)) against the target's own figure(A-B) (NOTES 9's test).", ""]
        conn = connectivity(targets, stage_summary.get("baseline", {}))
        for t, rec in conn.items():
            lines += [f"### {t} (stage {rec['stage']}, {rec['n_cells']} cells; rate input share covered by the top entries {rec['top_share'] * 100:.0f} %)", "",
                      "| input type | share of L2 input | sign (+/-/0 edges) | pair gain | figure z apple signed | figure z ball abs | share x figure (apple signed) |",
                      "|---|---|---|---|---|---|---|"]
            for r in rec["inputs"]:
                lines.append(f"| {r['type']} | {r['share'] * 100:.1f} % | {r['pos']}/{r['neg']}/{r['zero']} | x{r['gain']:.0f} | "
                             f"{'n/a' if r['z_apple'] is None else f'{r['z_apple']:+.1f}'} | {'n/a' if r['z_ball'] is None else f'{r['z_ball']:+.1f}'} | "
                             f"{'n/a' if r['contrib_apple'] is None else f'{r['contrib_apple']:+.6f}'} |")
            lines += ["", f"linear estimate (apple, signed): inputs deliver {rec['estimate_apple']:+.6f} vs the target's own figure "
                      f"{'n/a' if rec['own_apple'] is None else f'{rec['own_apple']:+.6f}'}; (ball, abs): {rec['estimate_ball']:+.6f} vs "
                      f"{'n/a' if rec['own_ball'] is None else f'{rec['own_ball']:+.6f}'}", ""]
        summary["connectivity"] = conn
    summary["stages"] = stage_summary; summary["object"] = obj_summary
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\n".join(lines))
    print(f"\nwritten {out_dir}/report.md, report.json")


def connectivity(targets, stage_maps, top=14):
    """Inputs of each target type in the baseline rate lobe: the L2-normalised, pair-gain-scaled |W| share per input type, the sign
    under the shipped receptor model, and the input types' baseline figures (from the stage maps, mean over seeds)."""
    from flyverse import brain, connectome
    from flyverse.connectome import PHOTORECEPTOR_TYPES
    c = connectome.load(verbose=False)
    nrn = c.neurons; types = nrn.type.fillna("").to_numpy()
    is_pr = nrn.type.isin(PHOTORECEPTOR_TYPES).to_numpy()
    rate = (nrn.superclass == "ol_intrinsic").to_numpy() & ~is_pr
    lif = brain.LIFParams(); rs = brain._receptor(c, lif, with_counts=False)
    W = c.W.tocsr(); coo = W.tocoo()
    sign = rs.fast_sign if rs is not None else np.sign(coo.data)
    l2 = nrn.in_syn_l2.to_numpy()
    gain = np.ones(coo.nnz, np.float32)
    for pre_re, post_re, f in optic.DEFAULT_PAIR_GAIN:
        pre_m = np.array([bool(re.match(pre_re, t)) for t in types]); post_m = np.array([bool(re.match(post_re, t)) for t in types])
        gain[pre_m[coo.col] & post_m[coo.row]] *= f
    apple = stage_maps.get("apple", {}); ball = stage_maps.get("ball", {})
    out = {}
    for t in targets:
        post = types == t
        m = post[coo.row] & rate[coo.col] & rate[coo.row]
        if not m.any():
            continue
        w = np.abs(coo.data[m]) / np.maximum(l2[coo.row[m]], 1.0) * gain[m]; sg = sign[m]; pre_t = types[coo.col[m]]
        tot = w.sum()
        rows = []
        for pt in sorted(set(pre_t), key=lambda x: -w[pre_t == x].sum()):
            mm = pre_t == pt
            g = float(np.unique(gain[m][mm])[0]) if len(np.unique(gain[m][mm])) == 1 else float(gain[m][mm].mean())
            pos_, neg_, zero_ = int((sg[mm] > 0).sum()), int((sg[mm] < 0).sum()), int((sg[mm] == 0).sum())
            signed_share = float((w[mm] * sg[mm]).sum() / tot)
            fa = apple.get(pt, {}).get("signed", {}).get("figAB_mean"); fb = ball.get(pt, {}).get("abs", {}).get("figAB_mean")
            rows.append({"type": pt, "share": float(w[mm].sum() / tot), "signed_share": signed_share, "pos": pos_, "neg": neg_, "zero": zero_, "gain": g,
                         "z_apple": apple.get(pt, {}).get("signed", {}).get("zAB_mean"), "z_ball": ball.get(pt, {}).get("abs", {}).get("zAB_mean"),
                         "contrib_apple": None if fa is None else signed_share * fa, "contrib_ball": None if fb is None else abs(signed_share) * fb})
        est_a = sum(r["contrib_apple"] for r in rows if r["contrib_apple"] is not None)
        est_b = sum(r["contrib_ball"] for r in rows if r["contrib_ball"] is not None)
        out[t] = {"stage": apple.get(t, ball.get(t, {})).get("stage"), "n_cells": int(post.sum()), "inputs": rows[:top], "top_share": float(sum(r["share"] for r in rows[:top])),
                  "estimate_apple": float(est_a), "own_apple": apple.get(t, {}).get("signed", {}).get("figAB_mean"),
                  "estimate_ball": float(est_b), "own_ball": ball.get(t, {}).get("abs", {}).get("figAB_mean")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", default=None, help="comma-separated configuration names (or all): effective parameters and edge structure, CPU")
    ap.add_argument("--one", default=None, help="run one configuration in this process (GPU)")
    ap.add_argument("--what", default="bench,stages,object")
    ap.add_argument("--seeds", default=SEEDS)
    ap.add_argument("--out", default="out/optic_audit")
    ap.add_argument("--batch", action="store_true", help="print the cluster_run.py batch (one job per configuration)")
    ap.add_argument("--submit", action="store_true", help="with --batch: submit it (console output -> out/optic_audit_cluster.log)")
    ap.add_argument("--configs", default=None, help="with --batch: a subset of configurations")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--report", default=None, metavar="DIR")
    ap.add_argument("--targets", default="T2,T2a,T3,Tm5Y,TmY21,TmY13,TmY5a,TmY3,Tm20,Y3,LPi34,LPi43",
                    help="with --report: target types for the connectivity table (empty = skip)")
    args = ap.parse_args()
    if args.list:
        for n, c in CONFIGS.items():
            print(f"{n:20s} {c['kind']:12s} {c['measure']:60s} {c['note']}")
        return
    if args.check:
        names = list(CONFIGS) if args.check == "all" else [s.strip() for s in args.check.split(",")]
        out = check(names)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "check.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(f"\nwritten {args.out}/check.json")
        return
    if args.one:
        run_one(args.one, args.what, args.seeds, args.out)
        return
    if args.batch:
        names = [s.strip() for s in args.configs.split(",")] if args.configs else None
        cmds = batch_commands(args.out, names)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "cluster_run.py"), "--name", "optic-audit", "--minutes", str(args.minutes)] + cmds + ["--fetch", "out/"]
        print("cluster batch (%d jobs):" % len(cmds))
        for c in cmds:
            print("  ", c)
        if args.submit:
            os.makedirs("out", exist_ok=True)
            log = os.path.join(ROOT, "out", "optic_audit_cluster.log")
            with open(log, "w", encoding="utf-8") as f:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                for line in p.stdout:
                    sys.stdout.write(line); f.write(line)
                rc = p.wait()
            print(f"cluster_run exit {rc}; console log {log}")
            sys.exit(rc)
        return
    if args.report:
        report(args.report, [t for t in args.targets.split(",") if t])
        return
    ap.print_help()


if __name__ == "__main__":
    main()
