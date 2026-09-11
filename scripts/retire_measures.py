"""Retire the anti-runaway measures one at a time: run scripts/benchmark.py's suite with each hand-crafted measure
removed alone, and with a literature-supported per-type replacement in its place, and tabulate which checks break.

    python scripts/retire_measures.py                        # every configuration, one subprocess each (~4 min per run)
    python scripts/retire_measures.py --configs baseline,no_adapt,adapt_by_type
    python scripts/retire_measures.py --fast --sections legacy,b,c
    python scripts/retire_measures.py --report               # only rebuild out/retire/comparison.md from the JSONs
    python scripts/retire_measures.py --one no_cap           # (internal) one configuration in this process

The measures (brain.LIFParams defaults; docs/NOTES.md session 3-4 and 8 record why each was added):
  adaptation      adapt_jump 1.5 mV / spike, tau 200 ms, every neuron
  cap             conn_cap 60 synapse-equivalents per connection
  same-type       same_type_gain 0.1 on every within-type synapse
  fan-in          input_norm_alpha 1, ref 5000: unitary synapse x (5000 / total inputs) for large neurons
  AL depression   DEFAULT_STD_U_BY_TYPE: u 0.2 on ORNs, AL local neurons and PNs
  path gains      DN -> VNC x3, visual projection -> DN x2
  type gains      LC4 / LPLC2 -> GF x3; SAD073 / GNG300 / DNp70 / CL367 / PVLP010 -> GF x0.3

Each configuration runs in its own subprocess (the fan-in normalisation cache is keyed on the parameter set, and the
weight hooks below change what a parameter set means). The replacements that need a rule rather than a LIFParams
value (a smooth saturation instead of the hard cap; same-type damping restricted to listed types) are installed
by wrapping brain._shaped_weights. adapt_by_type forces the Torch LIF path (the native kernel takes one scalar
jump), so that run and its eager baseline use --eager.

Results: <out>/<config>.json (benchmark.py's JSON layout plus the configuration), <out>/<config>.log, and
<out>/comparison.md: every check x every configuration, and per configuration the checks that break relative to
the baseline of the same backend. docs/audits/anti_runaway.md is written from those.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

DN_VNC = (r"^descending_neuron$", r"^vnc_", 3.0)
VP_DN = (r"^visual_projection$", r"^descending_neuron$", 2.0)
LOOM_GF = (r"^(LC4|LPLC2)$", r"^DNp01$", 3.0)
GF_DAMP = (r"^(SAD073|GNG300|DNp70|CL367|PVLP010)$", r"^DNp01$", 0.3)

# ---- per-type replacements -------------------------------------------------------------------------------------
# Adaptation only where it is documented or harmless; none in populations that hold persistent / tonic activity:
#  - compass and fan-shaped-body columnar cells (EPG / PEN / PEG / Delta7 / PFN / PFL / PFR / hDelta / vDelta / FC / FR / FS):
#    persistent heading and goal activity for tens of seconds (Seelig & Jayaraman 2015; Kim et al. 2017; Turner-Evans
#    et al. 2017; Hulse et al. 2021)
#  - ring neurons (ER): tonic, visually / self-motion tuned (Omoto et al. 2017; Sun et al. 2017)
#  - motor neurons: one spike per muscle potential, sustained for hours in flight (DLMn / DVMn) and tonic leg posture
#    (Azevedo et al. 2020); the GF fires single spikes (adaptation is irrelevant either way)
# Everything else keeps the global 1.5 mV / spike (ORNs and PNs adapt strongly in the animal: Nagel & Wilson 2011;
# the AVLP / CL cliques of session 3 have no physiology to cite either way).
PERSISTENT_TYPES = r"^(EPG|PEN|PEG|Delta7|PFN|PFL|PFR|PFG|hDelta|vDelta|FC|FR|FS|ER)"
MOTOR_TYPES = r"^(MN\d|DLMn|DVMn|TTMn|PSI|DNp01$)"


def adapt_by_type_replacement(c):
    """{type regex: mV per spike}: 0 for persistent-activity and motor populations, the default elsewhere."""
    vm = sorted(t for t in c.neurons.type[c.neurons.superclass == "vnc_motor"].fillna("").unique() if t)
    return {PERSISTENT_TYPES: 0.0, MOTOR_TYPES: 0.0, "^(" + "|".join(re.escape(t) for t in vm) + ")$": 0.0}


# Same-type damping only where the within-type synapses are documented to be something other than chemical
# recurrent excitation, or where session 3 measured the clique:
#  - antennal-lobe local neurons: LN <-> LN electrical coupling (Yaksi & Wilson 2010; Huang et al. 2010) and the
#    cholinergic lLN1 loop that ran at 300 Hz
#  - Kenyon cells: 415k KC -> KC synapses (57% of KC input; Takemura et al. 2017, Zheng et al. 2018 find the same in
#    hemibrain / FAFB) with no measured recurrent excitation -- KC output is sparse and APL-gated (Lin et al. 2014)
#  - ORNs: within-glomerulus ORN -> ORN axo-axonic synapses, presynaptic and modulatory in the animal (Tobin et al. 2017)
#  - the session-3 cliques: FR1, DNg33, and the AVLP / CL giants
# Everything else (the compass, DNs, VNC interneurons, motor neurons) gets its within-type synapses at full strength
# (DLMn / DVMn have 44 within-type synapses in total; the GF 2; TTMn 0: the "DLMn clique" of session 3 is not
# within-type wiring).
SAME_TYPE_DAMPED = r"^(lLN|v2LN|v3LN|il3LN|l2LN|vLN|KC|ORN_|FR1|DNg33|AVLP(154|157|488|520|428)|CL(212|002))"
# ... plus the visual projection neurons: LC / LPLC / LLPC / MeTu / LT types carry 60-280 within-type synapses per cell
# (LC17 284, LPLC2 250, LC9 186, LPLC1 155, LC12 152, LC4 146, MeTu1 109, LC10a 92), axo-axonic contacts inside the
# optic glomeruli (Wu et al. 2016) with no documented recurrent excitation; undamped, LC4 / LPLC2 amplify their own
# loom volley (loom GF peak 31 -> 81-93 Hz in the no_same_type / same_type_by_type runs).
SAME_TYPE_DAMPED_VP = SAME_TYPE_DAMPED[:-1] + r"|LC\d|LPLC\d|LLPC\d|LPC\d|MeTu|LT\d)"


def _hook_soft_cap(orig, mult=1.0):
    """Replace the hard cap min(n, cap) by the conductance-like saturation A (1 - exp(-n / A)), A = mult * cap: linear
    for small connections, saturating like the driving force of a conductance synapse. mult 1: the same asymptote
    (a 60-synapse connection drops to 38); mult 2: asymptote 120 (60 -> 47, 435 -> 117)."""
    def shaped(c, p):
        if p.conn_cap <= 0:
            return orig(c, p)
        cap = np.float32(p.conn_cap * mult)
        W = c.W.tocsr().copy()
        W.data = (np.sign(W.data) * cap * (1.0 - np.exp(-np.abs(W.data) / cap))).astype(np.float32)
        old = c.W; c.W = W
        try:
            return orig(c, dataclasses.replace(p, conn_cap=0.0))
        finally:
            c.W = old
    return shaped


def _hook_same_type_by_type(orig, pattern=SAME_TYPE_DAMPED):
    """same_type_gain applied only to postsynaptic types matching `pattern`; 1.0 elsewhere."""
    def shaped(c, p):
        if p.same_type_gain == 1.0:
            return orig(c, p)
        W = orig(c, dataclasses.replace(p, same_type_gain=1.0)).tocoo()
        types = c.neurons.type.fillna("").to_numpy()
        sel = np.array([bool(re.match(pattern, t)) for t in types])
        same = (types[W.row] == types[W.col]) & (types[W.row] != "") & sel[W.row]
        W.data[same] *= np.float32(p.same_type_gain)
        return W.tocsr()
    return shaped


def _hook_fan_in_by_size(orig, threshold=10000.0, ref=5000.0):
    """Fan-in scaling only for the giants: neurons with more than `threshold` input synapses get every input scaled by
    ref / total (the same factor the global rule gives them); everyone else stays at the uniform synapse. Installed as a
    weight rule so the run can set input_norm_alpha 0 (667 neurons above 10,000 inputs; 2,717 above 5,000)."""
    def shaped(c, p):
        if p.input_norm_alpha == 0 and p.conn_cap == 0:                 # Shiu-rules brain (bitter section): leave it alone
            return orig(c, p)
        W = orig(c, p).tocsr()
        tot = np.asarray(abs(W).sum(axis=1)).ravel()          # as brain.Brain: totals of the shaped matrix
        scale = np.where(tot > threshold, np.clip(ref / np.maximum(tot, 1.0), 0.02, 1.0), 1.0).astype(np.float32)
        import scipy.sparse as sp
        return (sp.diags(scale) @ W).tocsr()
    return shaped


HOOKS = {"soft_cap": _hook_soft_cap, "soft_cap_120": lambda orig: _hook_soft_cap(orig, 2.0),
         "same_type_by_type": _hook_same_type_by_type, "same_type_by_type_vp": lambda orig: _hook_same_type_by_type(orig, SAME_TYPE_DAMPED_VP),
         "fan_in_giants": _hook_fan_in_by_size}

# ---- configurations ----------------------------------------------------------------------------------------------
# name -> {"measure": which measure, "kind": ablation | replacement | baseline, "lif": LIFParams overrides (a callable
# of the connectome gives a dict), "hook": weight rule, "backend": native | eager}
CONFIGS = {
    "baseline": {"measure": "-", "kind": "baseline", "note": "current defaults, native backend"},
    "baseline_eager": {"measure": "-", "kind": "baseline", "backend": "eager", "note": "current defaults, torch path (the adapt_by_type run cannot use the native kernel)"},
    # adaptation
    "no_adapt": {"measure": "adaptation", "kind": "ablation", "lif": {"adapt_jump": 0.0}},
    "adapt_by_type": {"measure": "adaptation", "kind": "replacement", "backend": "eager", "lif": lambda c: {"adapt_by_type": adapt_by_type_replacement(c)},
                      "note": "0 mV/spike in CX columnar + ring neurons + motor neurons + GF; 1.5 elsewhere"},
    # connection cap
    "no_cap": {"measure": "cap", "kind": "ablation", "lif": {"conn_cap": 0.0}},
    "soft_cap": {"measure": "cap", "kind": "replacement", "hook": "soft_cap", "note": "60 (1 - exp(-n/60)) instead of min(n, 60)"},
    "soft_cap_120": {"measure": "cap", "kind": "replacement", "hook": "soft_cap_120", "note": "120 (1 - exp(-n/120)): the same rule with the knee at the old cap"},
    # same-type damping
    "no_same_type": {"measure": "same-type", "kind": "ablation", "lif": {"same_type_gain": 1.0}},
    "same_type_by_type": {"measure": "same-type", "kind": "replacement", "hook": "same_type_by_type",
                          "note": "x0.1 only on AL LNs, KCs, ORNs, FR1, DNg33, AVLP/CL giants; x1 elsewhere"},
    "same_type_by_type_vp": {"measure": "same-type", "kind": "replacement", "hook": "same_type_by_type_vp",
                             "note": "the same list plus the visual projection types (LC/LPLC/LLPC/LPC/MeTu/LT); x1 elsewhere"},
    # fan-in scaling
    "no_fan_in": {"measure": "fan-in", "kind": "ablation", "lif": {"input_norm_alpha": 0.0}},
    "fan_in_sqrt": {"measure": "fan-in", "kind": "replacement", "lif": {"input_norm_alpha": 0.5}, "note": "(5000 / total)^0.5"},
    "fan_in_ref_10000": {"measure": "fan-in", "kind": "replacement", "lif": {"input_norm_ref": 10000.0},
                         "note": "10000 / total: only neurons above 10k shaped inputs are scaled, by half as much (GF x0.24 instead of x0.12)"},
    "fan_in_giants": {"measure": "fan-in", "kind": "replacement", "lif": {"input_norm_alpha": 0.0}, "hook": "fan_in_giants",
                      "note": "no global rule; neurons above 10k shaped inputs keep their old factor 5000 / total (GF x0.12), everyone else x1"},
    # antennal-lobe depression
    "no_al_std": {"measure": "AL depression", "kind": "ablation", "lif": {"std_u_by_type": {}}},
    "std_orn_only": {"measure": "AL depression", "kind": "replacement", "lif": {"std_u_by_type": {r"^ORN_": 0.2}},
                     "note": "u 0.2 on ORN terminals only (Kazama & Wilson 2008)"},
    "std_orn_ln": {"measure": "AL depression", "kind": "replacement", "lif": {"std_u_by_type": {r"^ORN_": 0.2, r"^(lLN|v2LN|v3LN|il3LN|l2LN|vLN)": 0.2}},
                   "note": "u 0.2 on ORN and AL local-neuron terminals (ORN -> PN depression + LN gain control), PN terminals undepressed"},
    # pathway gains
    "no_path_gain": {"measure": "path gains", "kind": "ablation", "lif": {"path_gain": []}},
    "no_dn_vnc_gain": {"measure": "path gains", "kind": "ablation", "lif": {"path_gain": [VP_DN]}, "note": "DN -> VNC x3 removed, VP -> DN x2 kept"},
    "no_vp_dn_gain": {"measure": "path gains", "kind": "ablation", "lif": {"path_gain": [DN_VNC]}, "note": "VP -> DN x2 removed, DN -> VNC x3 kept"},
    "path_gain_typed": {"measure": "path gains", "kind": "replacement",
                        "lif": {"path_gain": [], "type_path_gain": [(r"^(LC4|LPLC2)$", r"^DNp01$", 6.0), GF_DAMP, (r"^DNp01$", r"^(TTMn|PSI)$", 10.0)]},
                        "note": "no superclass gains; LC4/LPLC2 -> GF x6 (the same effective loom gain), GF -> TTMn/PSI x10 (electrical synapse stand-in)"},
    # type gains
    "no_type_gain": {"measure": "type gains", "kind": "ablation", "lif": {"type_path_gain": []}},
    "no_gf_damping": {"measure": "type gains", "kind": "ablation", "lif": {"type_path_gain": [LOOM_GF]}, "note": "LC4/LPLC2 -> GF x3 kept, the x0.3 damping removed"},
    "gf_damping_dnp70": {"measure": "type gains", "kind": "replacement", "lif": {"type_path_gain": [LOOM_GF, (r"^DNp70$", r"^DNp01$", 0.3)]},
                         "note": "x0.3 only on the one excitatory input of the five (SAD073 / GNG300 / CL367 are GABA, PVLP010 glutamate)"},
    # every replacement at once (the per-measure winners): adaptation by type (eager only), the saturating cap with the
    # knee at 60, same-type damping restricted to the listed types + VP, fan-in (5000/total)^0.5, AL depression kept as
    # it is (no replacement passes), typed pathway gains without the GF damping
    "all_replacements": {"measure": "all", "kind": "replacement", "backend": "eager", "hooks": ["soft_cap_120", "same_type_by_type_vp"],
                         "lif": lambda c: dict(adapt_by_type=adapt_by_type_replacement(c), input_norm_alpha=0.5, path_gain=[],
                                               type_path_gain=[(r"^(LC4|LPLC2)$", r"^DNp01$", 6.0), (r"^DNp01$", r"^(TTMn|PSI)$", 10.0)]),
                         "note": "adapt_by_type + soft_cap_120 + same_type_by_type_vp + fan_in_sqrt + AL std + typed path gains, no GF damping"},
    "all_replacements_native": {"measure": "all", "kind": "replacement", "hooks": ["soft_cap_120", "same_type_by_type_vp"],
                                "lif": {"input_norm_alpha": 0.5, "path_gain": [],
                                        "type_path_gain": [(r"^(LC4|LPLC2)$", r"^DNp01$", 6.0), (r"^DNp01$", r"^(TTMn|PSI)$", 10.0)]},
                                "note": "the same with the global 1.5 mV adaptation kept (native kernel)"},
}


# ---- one configuration in this process ------------------------------------------------------------------------
def run_one(name, sections, fast, out_dir, seeds):
    import torch
    import benchmark as bm
    from flyverse import brain
    cfg = CONFIGS[name]
    eager = cfg.get("backend", "native") == "eager"
    args = argparse.Namespace(sections=sections, json="", fast=fast, eager=eager, seeds=seeds,
                              std_u=None, std_tau=None, adapt_jump=None, same_type_gain=None, norm_alpha=None, norm_ref=None,
                              w_syn=None, conn_cap=None, dn_vnc_gain=None, vp_dn_gain=None, gain_out=None, t4_gain=None)

    class RetireContext(bm.Context):
        def __init__(self, args, overrides):
            super().__init__(args)
            self.overrides = overrides(self.c) if callable(overrides) else (overrides or {})

        @property
        def has_overrides(self):
            return bool(self.overrides)

        def _apply_lif(self, p):
            for k, v in self.overrides.items():
                setattr(p, k, copy.deepcopy(v))
            return p

    # brain._shaped_weights aliases c.W when conn_cap == 0 (c.W.tocsr() is c.W; only the cap branch copies) and the
    # path / type gains are then multiplied into the shared connectome in place, compounding with every Brain built in
    # the process (verified: abs sum 121.4M -> 125.6M -> 137.9M over two calls with conn_cap 0). Guard every run.
    _orig_shaped = brain._shaped_weights

    def _protected(c, p):
        old = c.W; c.W = old.copy()
        try:
            return _orig_shaped(c, p)
        finally:
            c.W = old

    brain._shaped_weights = _protected
    for hook in cfg.get("hooks", [cfg["hook"]] if cfg.get("hook") else []):
        brain._shaped_weights = HOOKS[hook](brain._shaped_weights)
    t_all = time.time()
    ctx = RetireContext(args, cfg.get("lif"))
    lif = ctx.lif()
    config = {"name": name, "measure": cfg["measure"], "kind": cfg["kind"], "note": cfg.get("note", ""),
              "hooks": cfg.get("hooks", [cfg["hook"]] if cfg.get("hook") else []),
              "backend": "eager torch" if eager else "native (cuda_kernels, cuda_graphs, event_driven, warp)", "fast": fast, "seeds": ctx.seeds,
              "lif": {k: getattr(lif, k) for k in ["adapt_jump", "adapt_tau", "adapt_by_type", "std_u", "std_u_by_type", "same_type_gain",
                                                    "input_norm_alpha", "input_norm_ref", "conn_cap"]},
              "path_gain": brain.DEFAULT_PATH_GAIN if lif.path_gain is None else lif.path_gain,
              "type_path_gain": brain.DEFAULT_TYPE_PATH_GAIN if lif.type_path_gain is None else lif.type_path_gain,
              "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"}
    print("config:", json.dumps(config, default=str), flush=True)
    for letter, sname, fn in bm.select_sections(sections):
        print(f"\n=== [{letter or '-'}] {sname}", flush=True)
        t0 = time.time()
        try:
            ctx.results[sname] = fn(ctx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ctx.results[sname] = {"error": repr(e)}
            for key in bm.REFERENCES:
                if key.split(".")[0] == sname or (sname == "walk" and key.split(".")[0] in ("walk", "loom", "rotate")):
                    ctx.report(key, None)
        ctx.runtime[sname] = time.time() - t0
        print(f"    [{sname}: {ctx.runtime[sname]:.0f} s]", flush=True)
    total = time.time() - t_all
    print("\n" + bm.summary_table(ctx))
    n_pass = sum(c["status"].startswith("PASS") for c in ctx.checks); n_fail = sum(c["status"] == "FAIL" for c in ctx.checks)
    n_gap = sum(c["status"] == "KNOWN GAP" for c in ctx.checks); n_miss = sum(c["status"] == "MISSING" for c in ctx.checks)
    print(f"\n{name}: {n_pass} pass, {n_fail} fail, {n_gap} known gap, {n_miss} missing; runtime {total / 60:.1f} min")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump({"config": config, "sections": ctx.results, "checks": ctx.checks, "runtime_s": ctx.runtime, "total_runtime_s": total,
                   "date": time.strftime("%Y-%m-%d %H:%M")}, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ---- the comparison table ------------------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


MARK = {"PASS": "", "PASS (gap closed)": " (gap closed)", "FAIL": " FAIL", "KNOWN GAP": " gap", "MISSING": " MISSING"}


def report(out_dir, names):
    runs = {}
    for name in names:
        path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                runs[name] = json.load(f)
    if not runs:
        print("no results in", out_dir); return
    base = {"native": runs.get("baseline"), "eager": runs.get("baseline_eager")}
    keys = [c["key"] for c in next(iter(runs.values()))["checks"]]
    for r in runs.values():
        for c in r["checks"]:
            if c["key"] not in keys:
                keys.append(c["key"])
    # checks whose status differs between baseline runs (both backends, plus any extra baseline_*.json such as the plain
    # benchmark.py run) are run-to-run noise (the loom-escape bistability of NOTES session 8), not evidence
    base_runs = [r for n, r in runs.items() if n.startswith("baseline")]
    for fn in os.listdir(out_dir):
        if fn.startswith("baseline") and fn.endswith(".json") and fn[:-5] not in runs:
            with open(os.path.join(out_dir, fn)) as f:
                base_runs.append(json.load(f))
    ok = lambda s: s is not None and s.startswith("PASS")
    unstable = sorted({k for k in keys if len({ok({c["key"]: c["status"] for c in r["checks"]}.get(k)) for r in base_runs}) > 1})
    lines = ["# Anti-runaway measures: the suite with each measure removed / replaced", "",
             f"Generated by scripts/retire_measures.py on {time.strftime('%Y-%m-%d %H:%M')}. Each column is one run of scripts/benchmark.py "
             "(full protocol unless the config says fast) in its own process; 'breaks' lists the checks that pass in the baseline of the same "
             "backend and fail (or go missing) in the run, 'fixes' the reverse.", ""]
    head = ["check"] + list(runs)
    rows = [head]
    for k in keys:
        row = [k]
        for name, r in runs.items():
            ch = {c["key"]: c for c in r["checks"]}.get(k)
            row.append("n/a" if ch is None else _fmt(ch["measured"]) + MARK.get(ch["status"], ""))
        rows.append(row)
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(head))]
    lines += ["| " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)) + " |" for r in rows[:1]]
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    lines += ["| " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)) + " |" for r in rows[1:]]
    lines += ["", "## Per configuration", ""]
    summary = {}
    for name, r in runs.items():
        cfg = r["config"]; b = base["eager" if cfg["backend"].startswith("eager") else "native"]
        st = {c["key"]: c["status"] for c in r["checks"]}
        bst = {c["key"]: c["status"] for c in b["checks"]} if b else {}
        breaks = [k for k in keys if k not in unstable and ok(bst.get(k)) and not ok(st.get(k))]
        fixes = [k for k in keys if k not in unstable and k in st and not ok(bst.get(k)) and ok(st.get(k))]
        noise = [k for k in unstable if k in st and ok(bst.get(k)) != ok(st.get(k))]
        # quantitative shifts: measured values that moved by more than 50% of the baseline value (and by more than 1 unit),
        # whether or not the check still passes -- a pass by a narrower margin is part of the cost of a change
        meas = {c["key"]: c["measured"] for c in r["checks"]}
        bmeas = {c["key"]: c["measured"] for c in b["checks"]} if b else {}
        shifts = []
        for k in keys:
            v, bv = meas.get(k), bmeas.get(k)
            if isinstance(v, (int, float)) and isinstance(bv, (int, float)) and not isinstance(v, bool) and k not in unstable:
                if abs(v - bv) > max(0.5 * abs(bv), 1.0):
                    shifts.append(f"{k} {_fmt(bv)}->{_fmt(v)}")
        n_pass = sum(s.startswith("PASS") for s in st.values()); n_fail = sum(s == "FAIL" for s in st.values())
        n_gap = sum(s == "KNOWN GAP" for s in st.values()); n_miss = sum(s == "MISSING" for s in st.values())
        summary[name] = {"measure": cfg["measure"], "kind": cfg["kind"], "note": cfg.get("note", ""), "backend": cfg["backend"], "pass": n_pass, "fail": n_fail,
                         "gap": n_gap, "missing": n_miss, "breaks": breaks, "fixes": fixes, "noise": noise, "shifts": shifts,
                         "runtime_min": r["total_runtime_s"] / 60, "measured": meas}
        lines.append(f"* **{name}** ({cfg['measure']}, {cfg['kind']}{'; ' + cfg['note'] if cfg.get('note') else ''}; {cfg['backend'].split(' ')[0]}; "
                     f"{r['total_runtime_s'] / 60:.1f} min): {n_pass} pass, {n_fail} fail, {n_gap} gap, {n_miss} missing. "
                     f"breaks: {', '.join(breaks) or 'none'}. fixes: {', '.join(fixes) or 'none'}. "
                     f"shifts > 50%: {', '.join(shifts) or 'none'}."
                     + (f" (bistable in the baseline, ignored: {', '.join(noise)})" if noise else ""))
    lines.append("")
    lines.append(f"Checks whose status differs between the {len(base_runs)} baseline runs (ignored in 'breaks' / 'fixes'): {', '.join(unstable) or 'none'}.")
    with open(os.path.join(out_dir, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="all", help="comma-separated configuration names (see CONFIGS); 'all'")
    ap.add_argument("--sections", default="all", help="passed to benchmark.py")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "retire"))
    ap.add_argument("--skip-existing", action="store_true", help="do not rerun configurations that already have a JSON")
    ap.add_argument("--timeout", type=float, default=30.0, help="minutes per configuration")
    ap.add_argument("--report", action="store_true", help="only rebuild the comparison from existing JSONs")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    names = list(CONFIGS) if args.configs == "all" else [s.strip() for s in args.configs.split(",")]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configurations {unknown}; known: {list(CONFIGS)}")
    if args.one:
        run_one(args.one, args.sections, args.fast, args.out, args.seeds)
        return
    os.makedirs(args.out, exist_ok=True)
    if not args.report:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", SDL_VIDEODRIVER="dummy")
        for name in names:
            if args.skip_existing and os.path.exists(os.path.join(args.out, f"{name}.json")):
                print(f"[{name}] exists, skipped", flush=True); continue
            cmd = [sys.executable, os.path.abspath(__file__), "--one", name, "--sections", args.sections, "--seeds", args.seeds, "--out", args.out]
            if args.fast:
                cmd.append("--fast")
            t0 = time.time()
            print(f"[{name}] running ...", flush=True)
            with open(os.path.join(args.out, f"{name}.log"), "w", encoding="utf-8") as log:
                try:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.timeout * 60).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min (rc {rc})", flush=True)
    report(args.out, list(CONFIGS) if args.configs == "all" else names)


if __name__ == "__main__":
    main()
