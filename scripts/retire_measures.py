"""Retire the anti-runaway measures one at a time: run scripts/benchmark.py's suite with each hand-crafted measure
removed alone, and with a literature-supported per-type replacement in its place, and tabulate which checks break.

    python scripts/retire_measures.py                        # every configuration, one subprocess each (~4 min per run)
    python scripts/retire_measures.py --configs baseline,no_adapt,adapt_by_type
    python scripts/retire_measures.py --fast --sections legacy,b,c
    python scripts/retire_measures.py --report               # only rebuild out/retire/comparison.md from the JSONs
    python scripts/retire_measures.py --check --configs pair_gain_lpi_x1   # what a configuration changes, no simulation
    python scripts/retire_measures.py --report-replicates out/retire_r3    # <dir>/<config>_r<N>/<config>.json over replicates
    python scripts/retire_measures.py --one no_cap           # (internal) one configuration in this process

The measures (brain.LIFParams defaults; docs/NOTES.md session 3-4 and 8 record why each was added):
  adaptation      adapt_jump 1.5 mV / spike, tau 200 ms, every neuron
  cap             conn_cap 60 synapse-equivalents per connection
  same-type       same_type_gain 0.1 on every within-type synapse
  fan-in          input_norm_alpha 1, ref 5000: unitary synapse x (5000 / total inputs) for large neurons
  AL depression   DEFAULT_STD_U_BY_TYPE: u 0.2 on ORNs, AL local neurons and PNs
  path gains      DN -> VNC x3, visual projection -> DN x2
  type gains      LC4 / LPLC2 -> GF x3; SAD073 / GNG300 / DNp70 / CL367 / PVLP010 -> GF x0.3
  optic pair gains  optic.DEFAULT_PAIR_GAIN: LPi34 / LPi43 -> LPLC2 x4 (NOTES session 9), T4/T5 out x2, ...
  AL LN NT override connectome.UNKNOWN_NT_OVERRIDE_REGEX: unknown-NT antennal-lobe local neurons relabelled GABA

Receptor model: `--receptor-model` / `--receptor-net-rule` are passed through to every configuration exactly as
benchmark.py takes them. The default is `default`, which leaves brain.LIFParams' own defaults in force (since round 3:
receptor_model 'sign', receptor_net_rule 'abs'; docs/NT_INTEGRATION.md section 7); `off` means receptor_model=None,
the presynaptic-sign rule (the pre-round-3 weights). Each run records what it actually ran under in the JSON
(config.lif.receptor_model / receptor_net_rule, read back off the LIFParams the sections were built with).

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


# The hooks take brain._shaped_weights' signature (c, p, receptor=None) -- the third argument is the per-edge
# ReceptorSigns of the receptor model, computed once per Brain and aligned with c.W.nnz, so a hook may rewrite
# c.W.data (as the soft cap does) but must not change its sparsity pattern.
def _hook_soft_cap(orig, mult=1.0):
    """Replace the hard cap min(n, cap) by the conductance-like saturation A (1 - exp(-n / A)), A = mult * cap: linear
    for small connections, saturating like the driving force of a conductance synapse. mult 1: the same asymptote
    (a 60-synapse connection drops to 38); mult 2: asymptote 120 (60 -> 47, 435 -> 117)."""
    def shaped(c, p, receptor=None):
        if p.conn_cap <= 0:
            return orig(c, p, receptor)
        cap = np.float32(p.conn_cap * mult)
        W = c.W.tocsr().copy()
        W.data = (np.sign(W.data) * cap * (1.0 - np.exp(-np.abs(W.data) / cap))).astype(np.float32)
        old = c.W; c.W = W
        try:
            return orig(c, dataclasses.replace(p, conn_cap=0.0), receptor)
        finally:
            c.W = old
    return shaped


def _hook_same_type_by_type(orig, pattern=SAME_TYPE_DAMPED):
    """same_type_gain applied only to postsynaptic types matching `pattern`; 1.0 elsewhere."""
    def shaped(c, p, receptor=None):
        if p.same_type_gain == 1.0:
            return orig(c, p, receptor)
        W = orig(c, dataclasses.replace(p, same_type_gain=1.0), receptor).tocoo()
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
    def shaped(c, p, receptor=None):
        if p.input_norm_alpha == 0 and p.conn_cap == 0:                 # Shiu-rules brain (bitter section): leave it alone
            return orig(c, p, receptor)
        W = orig(c, p, receptor).tocsr()
        tot = np.asarray(abs(W).sum(axis=1)).ravel()          # as brain.Brain: totals of the shaped matrix
        scale = np.where(tot > threshold, np.clip(ref / np.maximum(tot, 1.0), 0.02, 1.0), 1.0).astype(np.float32)
        import scipy.sparse as sp
        return (sp.diags(scale) @ W).tocsr()
    return shaped


HOOKS = {"soft_cap": _hook_soft_cap, "soft_cap_120": lambda orig: _hook_soft_cap(orig, 2.0),
         "same_type_by_type": _hook_same_type_by_type, "same_type_by_type_vp": lambda orig: _hook_same_type_by_type(orig, SAME_TYPE_DAMPED_VP),
         "fan_in_giants": _hook_fan_in_by_size}

# ---- the two stop-gaps outside brain.LIFParams ---------------------------------------------------------------
LPI_LPLC2 = (r"^LPi(34|43)$", r"^LPLC2$")     # optic.DEFAULT_PAIR_GAIN entry: the x4 added in NOTES session 9


def pair_gain_lpi(factor):
    """optic.DEFAULT_PAIR_GAIN with the LPi34 / LPi43 -> LPLC2 factor set to `factor` (1.0 is the uniform-synapse
    value), everything else untouched. The x4 was hand-set so that the fly's own turning stopped driving the giant
    fibre through LPLC2; it is the only optic-lobe pair gain that stands in for a sign / strength the receptor table
    could now decide (round 3: the table matches 100 % of LPLC2's input and confirms the LPi glutamate as GluCl -1
    without licensing any strength, so the open question is where between x1 and x4 the factor sits)."""
    def build():
        from flyverse import optic
        out, found = [], 0
        for pre, post, g in optic.DEFAULT_PAIR_GAIN:
            if (pre, post) == LPI_LPLC2:
                found += 1
                out.append((pre, post, float(factor)))
            else:
                out.append((pre, post, g))
        if found != 1:
            raise SystemExit(f"optic.DEFAULT_PAIR_GAIN has {found} LPi -> LPLC2 entries {LPI_LPLC2}, expected 1")
        return {"pair_gain": out}
    return build


pair_gain_lpi_x1 = pair_gain_lpi(1.0)


def no_al_ln_override_connectome():
    """The connectome with connectome.UNKNOWN_NT_OVERRIDE_REGEX emptied for this process only: the antennal-lobe local
    neurons whose MaleCNS prediction is 'unknown' keep nt 'unknown' (sign 0, no fast synaptic effect) instead of being
    relabelled GABA. Compiled here (14 s) rather than through connectome.load(rebuild=True), which would SAVE the
    result over the shared cache; connectome.py itself is not edited. TYPE_NT_OVERRIDE stays at its default."""
    from flyverse import connectome as cn
    old = cn.UNKNOWN_NT_OVERRIDE_REGEX
    cn.UNKNOWN_NT_OVERRIDE_REGEX = {}
    try:
        c = cn.compile_connectome(verbose=True)
    finally:
        cn.UNKNOWN_NT_OVERRIDE_REGEX = old
    n_unknown = int((c.neurons.nt == "unknown").sum())
    print(f"no_al_ln_override: compiled without {old}; unknown-NT cells {n_unknown}, sum|W| {abs(c.W).sum():,.0f}", flush=True)
    return c


# ---- configurations ----------------------------------------------------------------------------------------------
# name -> {"measure": which measure, "kind": ablation | replacement | baseline, "lif": LIFParams overrides (a callable
# of the connectome gives a dict), "optic": OpticParams overrides (a callable of nothing gives a dict),
# "connectome": a callable returning the Connectome the run uses, "hook": weight rule, "backend": native | eager}
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
    "no_gf_damping": {"measure": "type gains", "kind": "ablation", "lif": {"type_path_gain": [LOOM_GF]}, "note": "LC4/LPLC2 -> GF x3 kept, the x0.3 damping removed (== the round-5 default)"},
    "gf_damped": {"measure": "type gains", "kind": "restoration", "lif": {"type_path_gain": list(brain.GF_DAMPED_TYPE_PATH_GAIN)},
                  "note": "the GF x0.3 input damping restored (the round-3/4 default; retired in round 5)"},
    "gf_damping_dnp70": {"measure": "type gains", "kind": "replacement", "lif": {"type_path_gain": [LOOM_GF, (r"^DNp70$", r"^DNp01$", 0.3)]},
                         "note": "x0.3 only on the one excitatory input of the five (SAD073 / GNG300 / CL367 are GABA, PVLP010 glutamate)"},
    # optic-lobe pair gains (optic.DEFAULT_PAIR_GAIN)
    "pair_gain_lpi_x1": {"measure": "optic pair gains", "kind": "ablation", "optic": pair_gain_lpi_x1,
                         "note": "LPi34 / LPi43 -> LPLC2 back to x1 (the uniform synapse); every other pair gain kept"},
    # the x1 / x2 / x3 / x4 scan the round-3 retire study named as its cheapest next test: where between the uniform
    # synapse and the hand-set x4 the factor crosses walk.power_max's 50 Hz bound, and what that does to loom.GF_peak
    "pair_gain_lpi_x2": {"measure": "optic pair gains", "kind": "replacement", "optic": pair_gain_lpi(2.0),
                         "note": "LPi34 / LPi43 -> LPLC2 at x2 instead of x4; every other pair gain kept"},
    "pair_gain_lpi_x3": {"measure": "optic pair gains", "kind": "replacement", "optic": pair_gain_lpi(3.0),
                         "note": "LPi34 / LPi43 -> LPLC2 at x3 instead of x4; every other pair gain kept"},
    # the unknown-NT antennal-lobe local-neuron relabelling (connectome.UNKNOWN_NT_OVERRIDE_REGEX)
    "no_al_ln_override": {"measure": "AL LN NT override", "kind": "ablation", "connectome": no_al_ln_override_connectome,
                          "note": "UNKNOWN_NT_OVERRIDE_REGEX emptied: unknown-NT AL local neurons keep sign 0 instead of GABA"},
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
# benchmark.Context reads its argparse namespace directly, so every flag it touches must exist here. These are
# scripts/benchmark.py's own defaults; BenchArgs returns None for any flag added to benchmark.py later (None is
# "not given" for all of its numeric / path options), which keeps this script from failing on an unrelated addition.
BENCH_DEFAULTS = dict(sections="all", json="", fast=False, eager=False, seeds="0,1",
                      std_u=None, std_tau=None, adapt_jump=None, same_type_gain=None, norm_alpha=None, norm_ref=None,
                      w_syn=None, conn_cap=None, dn_vnc_gain=None, vp_dn_gain=None, gain_out=None, t4_gain=None,
                      receptor_model="off", receptor_net_rule="class", receptor_table=None,
                      receptor_nt_class_fallback=False, receptor_gain=None, cache_dir=None,
                      slow_mode="additive", slow_gain_monoamine=None, slow_gain_classical=None,
                      slow_tau_monoamine=None, slow_tau_classical=None, dopamine_lead="all")


class BenchArgs(argparse.Namespace):
    def __getattr__(self, name):        # only reached when the attribute is absent
        return None


def bench_args(**kw):
    return BenchArgs(**dict(BENCH_DEFAULTS, **kw))


def make_context(name, args, bm):
    """benchmark.Context with the configuration's LIFParams / OpticParams overrides and, if it has one, its own
    connectome. The weight hooks are installed on brain._shaped_weights by the caller."""
    cfg = CONFIGS[name]
    if cfg.get("connectome"):
        from flyverse import connectome as cn
        c_cfg = cfg["connectome"]()
        cn.load = lambda *a, **k: c_cfg            # bm.Context.__init__ calls connectome.load(verbose=False)

    class RetireContext(bm.Context):
        def __init__(self, args, overrides, optic_overrides):
            super().__init__(args)
            self.overrides = overrides(self.c) if callable(overrides) else (overrides or {})
            self.optic_overrides = optic_overrides() if callable(optic_overrides) else (optic_overrides or {})

        @property
        def has_overrides(self):
            # also when only the optic lobe or a benchmark.py CLI flag (--receptor-model ...) changes, so that the
            # demo Sims are built through patched_params too
            return bool(self.overrides) or bool(self.optic_overrides) or bm.Context.has_overrides.fget(self)

        def _apply_lif(self, p):
            super()._apply_lif(p)                  # benchmark.py's CLI flags, including the receptor model
            for k, v in self.overrides.items():
                setattr(p, k, copy.deepcopy(v))
            return p

        def _apply_optic(self, op):
            super()._apply_optic(op)
            for k, v in self.optic_overrides.items():
                setattr(op, k, copy.deepcopy(v))
            return op

    return RetireContext(args, cfg.get("lif"), cfg.get("optic"))


def run_config(name, args, bm):
    """The context and the JSON `config` block of one configuration (no simulation): also what --check prints."""
    import torch
    from flyverse import brain, optic
    cfg = CONFIGS[name]
    ctx = make_context(name, args, bm)
    lif, op = ctx.lif(), ctx.optic_params()
    config = {"name": name, "measure": cfg["measure"], "kind": cfg["kind"], "note": cfg.get("note", ""),
              "hooks": cfg.get("hooks", [cfg["hook"]] if cfg.get("hook") else []),
              "backend": "eager torch" if args.eager else "native (cuda_kernels, cuda_graphs, event_driven, warp)",
              "fast": args.fast, "seeds": ctx.seeds,
              "lif": {k: getattr(lif, k) for k in ["adapt_jump", "adapt_tau", "adapt_by_type", "std_u", "std_u_by_type", "same_type_gain",
                                                   "input_norm_alpha", "input_norm_ref", "conn_cap",
                                                   "receptor_model", "receptor_net_rule", "receptor_nt_class_fallback", "receptor_table"]},
              "receptor_flags": {"receptor_model": args.receptor_model, "receptor_net_rule": args.receptor_net_rule},
              "path_gain": brain.DEFAULT_PATH_GAIN if lif.path_gain is None else lif.path_gain,
              "type_path_gain": brain.DEFAULT_TYPE_PATH_GAIN if lif.type_path_gain is None else lif.type_path_gain,
              "pair_gain": optic.DEFAULT_PAIR_GAIN if op.pair_gain is None else op.pair_gain,
              "connectome": {"n": int(ctx.c.n), "nnz": int(ctx.c.W.nnz), "sum_absW": float(abs(ctx.c.W).sum()),
                             "unknown_nt_cells": int((ctx.c.neurons.nt == "unknown").sum())},
              "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"}
    return ctx, config


def run_one(name, sections, fast, out_dir, seeds, receptor_model="off", receptor_net_rule="class"):
    import benchmark as bm
    from flyverse import brain
    cfg = CONFIGS[name]
    eager = cfg.get("backend", "native") == "eager"
    args = bench_args(sections=sections, fast=fast, eager=eager, seeds=seeds,
                      receptor_model=receptor_model, receptor_net_rule=receptor_net_rule)

    # brain._shaped_weights used to alias c.W when conn_cap == 0 (c.W.tocsr() is c.W; only the cap branch copied) and
    # the path / type gains were then multiplied into the shared connectome in place, compounding with every Brain
    # built in the process (abs sum 121.4M -> 125.6M -> 137.9M over two calls with conn_cap 0; that is what invalidated
    # the local no_cap run of out/retire/). It copies unconditionally since the receptor block was added; the guard is
    # kept because the hooks above hand it a substituted c.W.
    _orig_shaped = brain._shaped_weights

    def _protected(c, p, receptor=None):
        old = c.W; c.W = old.copy()
        try:
            return _orig_shaped(c, p, receptor)
        finally:
            c.W = old

    brain._shaped_weights = _protected
    for hook in cfg.get("hooks", [cfg["hook"]] if cfg.get("hook") else []):
        brain._shaped_weights = HOOKS[hook](brain._shaped_weights)
    t_all = time.time()
    ctx, config = run_config(name, args, bm)
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
        fixes = [k for k in keys if bst and k not in unstable and k in st and not ok(bst.get(k)) and ok(st.get(k))]   # no baseline -> no 'fixes'
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


# ---- the replicate table -----------------------------------------------------------------------------------
def _load_replicates(root):
    """{config: {replicate: run}} from <root>/<dir>/<config>.json; the replicate id is the trailing _r<N> of the
    directory name (the whole directory name if it has none). comparison.json / replicates.json are skipped."""
    runs = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        m = re.search(r"_r(\d+)$", d)
        rep = m.group(1) if m else d
        for fn in sorted(os.listdir(p)):
            if not fn.endswith(".json") or fn in ("comparison.json", "replicates.json"):
                continue
            with open(os.path.join(p, fn), encoding="utf-8") as f:
                runs.setdefault(fn[:-5], {})[rep] = json.load(f)
    return runs


def report_replicates(root, baseline="baseline"):
    """Every configuration x replicate x check, and per configuration the checks that break (or are fixed) in EVERY
    replicate against the baseline replicates -- the one-at-a-time rule of docs/audits/anti_runaway.md, applied with
    the run-to-run scatter visible in the table. Checks whose pass/fail differs between baseline replicates are
    listed as unstable and excluded from breaks / fixes."""
    runs = _load_replicates(root)
    if not runs:
        print("no <config>_r<N>/<config>.json under", root); return
    order = [n for n in CONFIGS if n in runs] + [n for n in runs if n not in CONFIGS]
    keys = []
    for name in order:
        for run in runs[name].values():
            for c in run["checks"]:
                if c["key"] not in keys:
                    keys.append(c["key"])
    st = {(n, r): {c["key"]: c["status"] for c in run["checks"]} for n, v in runs.items() for r, run in v.items()}
    ms = {(n, r): {c["key"]: c["measured"] for c in run["checks"]} for n, v in runs.items() for r, run in v.items()}
    ok = lambda s: s is not None and s.startswith("PASS")
    base_reps = sorted(runs.get(baseline, {}))
    unstable = sorted({k for k in keys if len({ok(st[(baseline, r)].get(k)) for r in base_reps}) > 1})

    cols = [(n, r) for n in order for r in sorted(runs[n])]
    rows = [["check"] + [f"{n} r{r}" for n, r in cols]]
    for k in keys:
        row = [k]
        for n, r in cols:
            s = st[(n, r)].get(k)
            row.append("n/a" if s is None else _fmt(ms[(n, r)].get(k)) + MARK.get(s, ""))
        rows.append(row)
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines = [f"# Retire measures over replicates ({root})", "",
             f"Generated by scripts/retire_measures.py --report-replicates on {time.strftime('%Y-%m-%d %H:%M')}. "
             f"{len(cols)} runs: {', '.join(f'{n} x{len(runs[n])}' for n in order)}. "
             "A check counts as broken only when it fails in EVERY replicate of the configuration while passing in "
             "every baseline replicate.", ""]
    lines += ["| " + " | ".join(str(v).ljust(x) for v, x in zip(rows[0], w)) + " |",
              "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    lines += ["| " + " | ".join(str(v).ljust(x) for v, x in zip(r, w)) + " |" for r in rows[1:]]

    summary = {}
    lines += ["", "## Per configuration (over its replicates)", ""]
    for n in order:
        my = sorted(runs[n])
        allp = lambda k, f: all(f(st[(n, r)].get(k)) for r in my)
        base_ok = lambda k: bool(base_reps) and all(ok(st[(baseline, r)].get(k)) for r in base_reps)
        base_bad = lambda k: bool(base_reps) and all(not ok(st[(baseline, r)].get(k)) for r in base_reps)
        breaks = [k for k in keys if k not in unstable and base_ok(k) and allp(k, lambda s: not ok(s))]
        fixes = [k for k in keys if k not in unstable and base_bad(k) and allp(k, ok)]
        flaky = [k for k in keys if k not in unstable and len({ok(st[(n, r)].get(k)) for r in my}) > 1]
        tally = [f"{sum(s.startswith('PASS') for s in st[(n, r)].values())}/"
                 f"{sum(s == 'FAIL' for s in st[(n, r)].values())}/"
                 f"{sum(s == 'KNOWN GAP' for s in st[(n, r)].values())}" for r in my]
        summary[n] = {"measure": runs[n][my[0]]["config"]["measure"], "note": runs[n][my[0]]["config"].get("note", ""),
                      "replicates": my, "pass_fail_gap": tally, "breaks_in_all": breaks, "fixes_in_all": fixes,
                      "differs_between_replicates": flaky,
                      "runtime_min": [round(runs[n][r]["total_runtime_s"] / 60, 1) for r in my],
                      "measured": {r: ms[(n, r)] for r in my}}
        lines.append(f"* **{n}** ({summary[n]['measure']}{'; ' + summary[n]['note'] if summary[n]['note'] else ''}): "
                     f"pass/fail/gap {', '.join(tally)}. breaks in all replicates: {', '.join(breaks) or 'none'}. "
                     f"fixes in all: {', '.join(fixes) or 'none'}. differs between its own replicates: {', '.join(flaky) or 'none'}.")
    lines += ["", f"Checks whose status differs between the {len(base_reps)} baseline replicates (excluded from "
                  f"breaks / fixes): {', '.join(unstable) or 'none'}."]
    with open(os.path.join(root, "replicates.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(root, "replicates.json"), "w", encoding="utf-8") as f:
        json.dump({"unstable_in_baseline": unstable, "configs": summary}, f, indent=1)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="all", help="comma-separated configuration names (see CONFIGS); 'all'")
    ap.add_argument("--sections", default="all", help="passed to benchmark.py")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--receptor-model", default="default", choices=["default", "off", "sign", "sign+gain", "full"],
                    help="passed to benchmark.py's Context for every configuration; 'off' leaves LIFParams alone, i.e. "
                         "the run inherits brain.LIFParams' defaults (round 3: 'sign' / 'abs')")
    ap.add_argument("--receptor-net-rule", default="abs", choices=["class", "abs", "nonmda"], help="passed through with --receptor-model (ignored under default / off)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "retire"))
    ap.add_argument("--skip-existing", action="store_true", help="do not rerun configurations that already have a JSON")
    ap.add_argument("--timeout", type=float, default=30.0, help="minutes per configuration")
    ap.add_argument("--report", action="store_true", help="only rebuild the comparison from existing JSONs")
    ap.add_argument("--report-replicates", default=None, metavar="DIR",
                    help="aggregate <DIR>/<config>_r<N>/<config>.json over replicates into <DIR>/replicates.md / .json "
                         "(a check counts as broken only when it breaks in every replicate)")
    ap.add_argument("--check", action="store_true",
                    help="print the effective configuration (LIFParams, pair gains, connectome) of each named "
                         "configuration and exit; no simulation, no GPU")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.report_replicates:
        report_replicates(args.report_replicates)
        return
    names = list(CONFIGS) if args.configs == "all" else [s.strip() for s in args.configs.split(",")]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configurations {unknown}; known: {list(CONFIGS)}")
    if args.check:
        import benchmark as bm
        for name in names:
            cfg = CONFIGS[name]
            a = bench_args(sections=args.sections, fast=args.fast, eager=cfg.get("backend", "native") == "eager",
                           seeds=args.seeds, receptor_model=args.receptor_model, receptor_net_rule=args.receptor_net_rule)
            _, config = run_config(name, a, bm)
            print(json.dumps(config, indent=1, default=str), flush=True)
        return
    if args.one:
        run_one(args.one, args.sections, args.fast, args.out, args.seeds, args.receptor_model, args.receptor_net_rule)
        return
    os.makedirs(args.out, exist_ok=True)
    if not args.report:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", SDL_VIDEODRIVER="dummy")
        for name in names:
            if args.skip_existing and os.path.exists(os.path.join(args.out, f"{name}.json")):
                print(f"[{name}] exists, skipped", flush=True); continue
            cmd = [sys.executable, os.path.abspath(__file__), "--one", name, "--sections", args.sections, "--seeds", args.seeds,
                   "--out", args.out, "--receptor-model", args.receptor_model, "--receptor-net-rule", args.receptor_net_rule]
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
