"""Whole-CNS leaky integrate-and-fire simulation of the MaleCNS connectome on the GPU (torch), batched.

Model (Shiu et al. 2024, Nature 634:210; same constants as stonkfly/doomfly):
    dv/dt = (v_rest - v + g + I_ext) / tau_m          v_rest = v_reset = -52 mV, v_th = -45 mV
    dg/dt = -g / tau_syn                              tau_m = 20 ms, tau_syn = 5 ms
    presynaptic spike (after 1.8 ms delay): g_post += 0.275 mV * sign * synapse_count
    refractory period 2.2 ms.
Integration is exponential-Euler with a configurable dt (default 0.5 ms; Shiu/stonkfly use 0.1 ms).
Extras (all optional, see LIFParams): spike-frequency adaptation, short-term synaptic depression, a
fan-in cap on unitary synaptic strength for very large neurons, current injection and Poisson forcing.

Batching: `Brain(c, batch=B)` simulates B independent brains (same connectome) with state tensors of
shape (B, N). One sparse matmul serves all B: the cost is nearly flat in B up to ~64 on an RTX 4090.
Every method accepts per-brain values where it makes sense; with B = 1 the scalar API is unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .connectome import Connectome, ReceptorSigns, receptor_signs
from .device import resolve, sparse_matrix
from . import metal, cuda


@dataclass
class LIFParams:
    v_rest: float = -52.0
    v_reset: float = -52.0
    v_th: float = -45.0
    tau_m: float = 20.0      # ms
    tau_syn: float = 5.0     # ms
    t_ref: float = 2.2       # ms
    delay: float = 1.8       # ms
    w_syn: float = 0.275     # mV per synapse
    dt: float = 0.5          # ms
    rate_tau: float = 100.0  # ms, time constant of the running firing-rate estimate
    # spike-frequency adaptation (not in Shiu et al.): each spike adds `adapt_jump` mV of hyperpolarising
    # current that decays with `adapt_tau`. At rate R the steady adaptation is R*jump*tau, e.g. 100 Hz ->
    # 6 mV with the defaults, so runaway loops (KCs/PEN/PAM at 300 Hz) throttle themselves. 0 disables.
    adapt_jump: float = 1.5  # mV (was 0.3; raised when depression was switched off, see NOTES)
    adapt_tau: float = 200.0 # ms
    # per-type adaptation jump {type_regex: mV}: the compass (EPG / PEN / PEG / Delta7) holds persistent
    # activity in the animal and cannot with 1.5 mV / spike (NOTES, session 8). None = DEFAULT_ADAPT_BY_TYPE.
    adapt_by_type: dict | None = None
    # short-term synaptic depression (Tsodyks-Markram style, per presynaptic neuron; not in Shiu et al.):
    # each spike uses a fraction `std_u` of the available resource x, which recovers with `std_tau`. At
    # rate R the steady resource is 1/(1 + u R tau): 20 Hz -> 0.45, 100 Hz -> 0.14, 300 Hz -> 0.05, which
    # kills the self-exciting cliques (FR1, DLMn, ...) that otherwise lock up at 300 Hz. 0 disables.
    std_u: float = 0.0       # off by default: depression blocked descending commands (NOTES, session 3)
    std_tau: float = 300.0   # ms
    # per-presynaptic-type depression: {type_regex: u}. ORN -> PN synapses are the classic strongly
    # depressing synapse of the fly (Kazama & Wilson 2008); without it 3 Hz of spontaneous ORN input
    # saturates the projection neurons and the antennal lobe runs hot.
    std_u_by_type: dict = None
    # Large neurons have low input resistance: neurons whose total input synapse count exceeds
    # `input_norm_ref` get their unitary synapse scaled by (ref / total)^alpha (down only, floor 0.02).
    # E.g. the giant fibre (~40k inputs) would otherwise fire from a few hundred active synapses of
    # walking-related central-brain input; with ref 5000 it needs the coherent LC4+LPLC2 loom volley.
    # alpha 0 = Shiu's uniform 0.275 mV everywhere.
    input_norm_alpha: float = 1.0
    input_norm_ref: float = 5000.0
    # Synapses between neurons of the SAME cell type are scaled by this factor. Dense within-type
    # excitatory connections (FR1, lLN1_bc, DLMn, DNg33 ...) are the runaway cliques of the point model;
    # in the animal such populations are typically gap-junction coupled and fire in synchrony rather
    # than exciting each other chemically. 1.0 = untouched.
    same_type_gain: float = 0.1
    # Per-connection saturation: a connection of `count` synapses contributes min(count, conn_cap)
    # synapse-equivalents (0 = linear, as in Shiu et al.). PSP amplitude does not grow linearly with
    # synapse number in real neurons; the linear rule turns the few giant connections (>100 synapses,
    # e.g. AVLP488->AVLP520 at ~435 per cell = 120 mV per presynaptic spike) into runaway drivers.
    conn_cap: float = 60.0
    # Per-pathway gains on the LIF weights: list of (pre_superclass_regex, post_superclass_regex, factor).
    # The first per-cell-type gain of the model: descending -> VNC synapses. With the anti-runaway
    # settings, single DN pairs at 150 Hz no longer reach the leg motor neurons; the animal's DN->VNC
    # synapses are strong (DNp09 / MDN optogenetics walks the fly).
    path_gain: list = None
    # Same, keyed on cell TYPE regexes: (pre_type_regex, post_type_regex, factor). Default: the direct
    # LC4 / LPLC2 -> giant fibre synapses x3 (Ache et al. 2019: the GF's loom input is these two types;
    # here it lets the escape threshold sit above the single GF spikes that central-brain crosstalk
    # produces while a loom still gives a burst).
    type_path_gain: list = None
    # Synaptic input as an event-driven gather over the outputs of the neurons that fired (cost ~ spikes x
    # fan-out, ~100x less than the full 24.6M-synapse spmm at a few % activity) or as one sparse matmul.
    # None = matmul on CUDA (cuSPARSE spmm is fast and the batched RL flies are dense in spikes), events
    # elsewhere (MPS/CPU sparse matmul is 5-30x slower than the gather).
    event_driven: bool | None = None
    weight_dtype: str = "float32"   # "float16" on CUDA; spike products accumulate into float32
    # Per-module integration clocks {module: dt_ms} (regions.py names), each a multiple of dt and at most the
    # synaptic delay. Neurons on a slow clock are integrated every k steps with dt_k = k dt; their synaptic
    # input is the spikes accumulated since their last update, and their own spikes reach everyone through
    # the delay buffer. E.g. {"vnc": 1.0} halves the VNC's share of the sparse matmul. FlyBrain resolves it.
    dt_by_module: dict | None = None
    # Drop the synapses from / onto frozen rate units (the optic lobe) from the LIF matrix. Exact: frozen
    # neurons never spike, so those entries only cost time (~40% of the nnz in the full brain).
    prune_frozen: bool = True
    # ---- receptor model (docs/NT_INTEGRATION.md step 5; docs/audits/receptor_rules.md). DEFAULT since round 3
    # (docs/audits/receptor_integration.md "Round 3: adoption"): "sign" with the absolute-level net rule "abs" on the
    # contested-flip table -- 30,916 glutamate entries flipped to +1 (iGluR targets no other source contradicts) and
    # 17,379 histamine entries silenced (>= 2 sources with ort / HisCl off) = 48,295 of 25,578,600 entries, 0.15 % of
    # |W|; adopted after 27 PASS / 0 FAIL / 2 KNOWN GAP in 3 of 3 suite replicates with no check worse than any off run.
    # None: the previous rule (a synapse carries NT_SIGN of its presynaptic cell); still selectable, and the weights
    #   are then byte-identical to the model before the receptor block (tests/test_receptor_model.py pins the hash).
    # "sign": where (postsynaptic type, presynaptic transmitter) has a row in flyverse/data/receptors_by_type.csv
    #   (29.7 % of edges / 25.7 % of |W| synapses of the whole CNS; optic module 52 %), the fast sign of that row
    #   replaces the presynaptic sign (glutamate -> +1 on iGluR targets, 0 where the target has no fast receptor
    #   for that transmitter, ...); unmatched edges keep NT_SIGN. Applied to abs(W) BEFORE the connection cap.
    # "sign+gain": as "sign", times the gain-class factor `receptor_gain` of the row (expression tertiles, not
    #   conductances -- a parameter to sweep), also before the cap.
    # "full": "sign+gain" plus a SLOW tone g_slow per neuron driven by the row's slow (metabotropic / monoamine) sign
    #   and gain class through sparse matrices W_slow (entries with a non-zero slow sign only), ONE PER SLOW CLASS
    #   (connectome.SLOW_CLASSES: 'metabotropic_classical' = mAChR-A/-B, GABA-B, mGluR on ACh / GABA / glutamate
    #   edges; 'monoamine' = the dopamine / octopamine / serotonin receptors), each with its own scale
    #   (`slow_gain_by_class`, x w_syn per synapse per presynaptic spike) and time constant (`slow_tau_by_class`).
    #   Round 2 (docs/audits/slow_term.md): the classical class defaults to 0 -- 98 % of the round-1 slow entries were
    #   mAChR-B / GABA-B / mGluR, which act mostly presynaptically in the animal and turned into a runaway current --
    #   and the monoamine class is the term the plan asked for. How g_slow acts is `slow_mode`:
    #     "additive"  : g_slow (mV) is added to the membrane target like g (round 1's form);
    #     "gain"      : the fast synaptic input g is multiplied by clamp(1 + g_slow / (v_th - v_rest), slow_gain_clip)
    #                   -- NOTE: g is the NET synaptic input (excitation minus inhibition), so a +7 mV tone doubles it and a
    #                   -7 mV tone zeroes it whatever its sign: on a net-inhibited target a negative tone REDUCES the
    #                   inhibition (disinhibits). A per-sign gain would need separate excitatory / inhibitory accumulators;
    #     "threshold" : the spike threshold becomes v_th - min(g_slow, 0.9 (v_th - v_rest)) -- a +7 mV tone puts the
    #                   threshold 0.7 mV above rest, a negative tone raises it without bound.
    #   In every mode g_slow = sum over classes of a per-class state that jumps by slow_gain[class] x w_syn x count x
    #   slow sign x gain factor per transmitted presynaptic spike and decays with slow_tau[class]; the steady tone of a
    #   presynaptic cell at rate R is therefore slow_gain x tau_slow / tau_syn times the fast conductance the same
    #   synapses would carry (gain 0.02, tau 200 ms: 0.8x). Monoamine synapses are explicit zeros in the cached W, so
    #   their counts come from cache/sign0_counts.npz (connectome.sign0_counts, built from the raw weights table on
    #   first use). The native CUDA / Metal kernels do not carry g_slow: "full" forces the Torch path (a warning if
    #   kernels were requested), like adapt_by_type; when every class gain is 0 no slow matrix is built and the step
    #   loop does no slow work (the weights and the dynamics then equal "sign+gain" on the Torch path).
    receptor_model: str | None = "sign"       # round-3 default; None = the presynaptic-sign rule (previous weights)
    receptor_net_rule: str = "abs"            # "class" | "abs" | "nonmda": which column set of the table decides the sign
    receptor_nt_class_fallback: bool = False  # unprofiled targets take the Davis 2020 whole-class baseline (tier nt_class)
    receptor_table: str | None = None         # path override for receptors_by_type.csv
    receptor_gain: dict | None = None         # {gain class: factor}; None = DEFAULT_RECEPTOR_GAIN (low 0.5, mid 1, high 1.5)
    slow_tau_ms: float = 200.0                # monoamine time constant (ms; GPCR cascades are 100s of ms) -- see slow_tau_by_class
    slow_gain: float = 0.02                   # monoamine scale (x w_syn per synapse per spike; round 1 used 0.1 for every class)
    # {slow class: scale} / {slow class: tau ms}; None = DEFAULT_SLOW_GAIN_BY_CLASS / DEFAULT_SLOW_TAU_BY_CLASS with the
    # monoamine entries taken from slow_gain / slow_tau_ms. Keys must be connectome.SLOW_CLASSES[1:].
    slow_gain_by_class: dict | None = None
    slow_tau_by_class: dict | None = None
    slow_mode: str = "additive"               # "additive" | "gain" | "threshold" (see above)
    slow_gain_clip: tuple = (0.0, 4.0)        # bounds of the multiplicative factor in "gain" mode
    # Per-transmitter unitary strength (docs/audits/unitary_strength.md; opt-in, thread:unitary). {transmitter: factor}
    # keyed by connectome.NT_SIGN names: every entry whose PRESYNAPTIC cell carries that transmitter is multiplied by
    # the factor on |W| in _shaped_weights BEFORE the connection cap (so the cap still saturates at conn_cap synapse-
    # equivalents of the SCALED count), after the receptor sign / gain-class stage and before the path gains, the
    # same-type damping and the fan-in normalisation (which therefore partly renormalises a uniform per-transmitter
    # scale on cells above input_norm_ref). Transmitters absent from the dict keep x1; sign-0 entries stay zero. None
    # (the default) executes nothing and the shaped weights are byte-identical (tests/test_unitary.py). The optic-lobe
    # rate model normalises its own weights by in_syn and is NOT touched. w_syn stays the single global scale.
    w_syn_by_nt: dict | None = None
    surrogate_grad: bool = False            # experimental Torch-only gradients through short windows


# Gain-class factors of the receptor model ("sign+gain" / "full"): the table's classes are per-source expression
# tertiles of the winning receptor group ("none" only occurs with sign 0). Parameters, not measurements.
DEFAULT_RECEPTOR_GAIN = {"none": 1.0, "low": 0.5, "mid": 1.0, "high": 1.5}
RECEPTOR_MODELS = (None, "sign", "sign+gain", "full")
SLOW_MODES = ("additive", "gain", "threshold")
# Slow-term defaults per class (docs/audits/slow_term.md). The classical metabotropic class is OFF (scale 0): mAChR-B /
# GABA-B / mGluR act largely presynaptically (release, adaptation) and as an additive current they were round 1's
# runaway; its tau (100 ms, the GABA-B IPSP scale) is a parameter with no calibration behind it. The monoamine
# entries come from LIFParams.slow_gain / slow_tau_ms.
DEFAULT_SLOW_GAIN_BY_CLASS = {"metabotropic_classical": 0.0, "monoamine": None}
DEFAULT_SLOW_TAU_BY_CLASS = {"metabotropic_classical": 100.0, "monoamine": None}
SLOW_THRESHOLD_MAX_FRAC = 0.9             # "threshold" mode: the threshold never drops below v_rest + 0.1 (v_th - v_rest)


@dataclass
class SlowSpec:
    """The resolved slow term of receptor_model == 'full' (brain._slow_spec): mode, per-class scale and time constant
    (classes with scale 0 are dropped), the fast synaptic time constant the scales are relative to, and the
    normalisation of the multiplicative modes (norm_mv = v_th - v_rest for the LIF). Shared with optic.OpticLobe."""
    mode: str
    gain: dict            # {class name: scale}, non-zero entries only
    tau: dict             # {class name: ms}, same keys
    tau_syn: float        # ms
    norm_mv: float        # mV: the tone that doubles the fast input ('gain') / would put the threshold at rest ('threshold')
    clip: tuple = (0.0, 4.0)

    @property
    def classes(self) -> list:
        return list(self.gain)


def _slow_gains(p: LIFParams) -> dict:
    g = dict(DEFAULT_SLOW_GAIN_BY_CLASS); g["monoamine"] = float(p.slow_gain)
    if p.slow_gain_by_class:
        g.update({k: float(v) for k, v in p.slow_gain_by_class.items()})
    return g


def _slow_taus(p: LIFParams) -> dict:
    t = dict(DEFAULT_SLOW_TAU_BY_CLASS); t["monoamine"] = float(p.slow_tau_ms)
    if p.slow_tau_by_class:
        t.update({k: float(v) for k, v in p.slow_tau_by_class.items()})
    return t


def _slow_spec(p: LIFParams) -> SlowSpec | None:
    """None unless receptor_model == 'full' AND some class has a non-zero scale (then the term costs nothing)."""
    if p.receptor_model != "full":
        return None
    from .connectome import SLOW_CLASSES
    if p.slow_mode not in SLOW_MODES:
        raise ValueError(f"slow_mode must be one of {SLOW_MODES}")
    gains, taus = _slow_gains(p), _slow_taus(p)
    bad = set(gains) | set(taus)
    bad -= set(SLOW_CLASSES[1:])
    if bad:
        raise ValueError(f"unknown slow classes {sorted(bad)}; use {SLOW_CLASSES[1:]}")
    for k, tau in taus.items():
        if not (math.isfinite(tau) and tau > 0):
            raise ValueError(f"slow tau of class {k!r} must be positive and finite")
    active = {k: g for k, g in gains.items() if g != 0.0}
    if not active:
        return None
    return SlowSpec(p.slow_mode, active, {k: taus[k] for k in active}, float(p.tau_syn), float(p.v_th - p.v_rest),
                    tuple(float(x) for x in p.slow_gain_clip))


# Depression only in the antennal lobe (ORN -> PN and the LN/PN recurrence are documented depressing
# synapses; without it the AL's PN <-> cholinergic-LN loop runs at 300 Hz). Elsewhere depression is off
# because it blocks descending commands.
DEFAULT_TYPE_PATH_GAIN = [(r"^(LC4|LPLC2)$", r"^DNp01$", 3.0)]     # loom escape margin (with the x2 pathway gain: x6 in total)

# The default until receptor round 5 (docs/audits/anti_runaway.md "Round 5: GF damping adoption"): the central-brain inputs
# that fired the GF during ordinary walking / feeding in this model (input-weighted: SAD073, GNG300, DNp70, CL367,
# PVLP010) were damped x0.3 on the argument that the animal's GF is hard to fire except by looms and mechanical shocks.
# Four of the five are inhibitory (SAD073 / GNG300 / CL367 GABA, PVLP010 glutamate; 2,899 of the 4,315 |W|), so the
# damping removed inhibition from DNp01; its ablation turned walk.power_max 79.47 FAIL into 48.48 PASS in 10/10 draws
# (round 4) and it was retired in round 5.  Kept as a named list so LIFParams(type_path_gain=GF_DAMPED_TYPE_PATH_GAIN)
# reproduces the previous weights byte for byte (tests/test_receptor_model.py) and scripts/retire_measures.py's
# no_gf_damping / gf_damping_dnp70 configurations keep their meaning relative to it.
GF_DAMPED_TYPE_PATH_GAIN = [(r"^(LC4|LPLC2)$", r"^DNp01$", 3.0),
                            (r"^(SAD073|GNG300|DNp70|CL367|PVLP010)$", r"^DNp01$", 0.3)]

DEFAULT_PATH_GAIN = [(r"^descending_neuron$", r"^vnc_", 3.0),          # benchmarked: specific, ipsilateral leg drive, no storms
                     (r"^visual_projection$", r"^descending_neuron$", 2.0)]   # LC4/LPLC2 -> GF etc.: loom escape margin (x3 re-ignites the AVLP network)

DEFAULT_ADAPT_BY_TYPE: dict = {}      # filled in when the compass benchmark settles; {} = uniform adapt_jump

# depressing terminals: ORNs (ORN -> PN, Kazama & Wilson 2008) and antennal-lobe LNs. (A PN entry that used to be here
# never matched a cell -- the patterns are applied with re.match -- so the model has always been ORN + LN only.)
DEFAULT_STD_U_BY_TYPE = {r"^ORN_": 0.2, r"^(lLN|v2LN|v3LN|il3LN|l2LN|vLN)": 0.2}


def _receptor_gain(p: LIFParams) -> dict | None:
    """The gain-class factors in force: None under "sign" (signs only), the dict under "sign+gain" / "full"."""
    if p.receptor_model in ("sign+gain", "full"):
        return DEFAULT_RECEPTOR_GAIN if p.receptor_gain is None else p.receptor_gain
    return None


def _receptor_key(p: LIFParams):
    """The receptor settings that change the shaped weights (part of the fan-in normalization cache key)."""
    if p.receptor_model is None:
        return None
    g = _receptor_gain(p)
    return (p.receptor_model, p.receptor_net_rule, p.receptor_nt_class_fallback, p.receptor_table,
            tuple(sorted(g.items())) if g else None)


def _receptor(c: Connectome, p: LIFParams, receptor: ReceptorSigns | None = None,
              with_counts: bool = False) -> ReceptorSigns | None:
    """Validate / compute the per-edge receptor lookup for `c` under `p` (None when the model is off)."""
    if p.receptor_model not in RECEPTOR_MODELS:
        raise ValueError(f"receptor_model must be one of {RECEPTOR_MODELS}")
    if p.receptor_model is None:
        return None
    if receptor is None:
        receptor = receptor_signs(c, table_path=p.receptor_table, net_rule=p.receptor_net_rule,
                                  nt_class_fallback=p.receptor_nt_class_fallback, with_counts=with_counts)
    if len(receptor.fast_sign) != c.W.nnz:
        raise ValueError("receptor signs are not aligned with this connectome's W")
    return receptor


def _nt_factor(c: Connectome, by_nt: dict) -> np.ndarray:
    """(N,) float32 per-cell factor of LIFParams.w_syn_by_nt: the factor of the cell's transmitter (NT_SIGN names;
    cells without a label count as 'unknown'), 1 for transmitters absent from the dict."""
    from .connectome import NT_SIGN
    bad = sorted(set(by_nt) - set(NT_SIGN))
    if bad:
        raise ValueError(f"w_syn_by_nt: unknown transmitters {bad}; use {sorted(NT_SIGN)}")
    nt = c.neurons.nt.fillna("unknown").to_numpy() if "nt" in c.neurons.columns else np.full(c.n, "unknown")
    f = np.ones(c.n, dtype=np.float32)
    for k, v in by_nt.items():
        v = float(v)
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"w_syn_by_nt[{k!r}] must be finite and >= 0, got {v}")
        f[nt == k] = np.float32(v)
    return f


def _shaped_weights(c: Connectome, p: LIFParams, receptor: ReceptorSigns | None = None):
    """Apply the calibrated connection rules before fan-in normalization."""
    W = c.W.tocsr().copy()                                          # always a copy: the gain loops below write into W.data
    receptor = _receptor(c, p, receptor)
    if receptor is not None:
        # sign (and gain class) per edge from the receptor table, on the magnitudes, BEFORE the cap; unmatched edges keep
        # NT_SIGN of the presynaptic cell (fast_sign == sign(W.data) there), explicit zeros stay zero
        W.data = np.abs(W.data) * receptor.fast_factor(_receptor_gain(p))
    if p.w_syn_by_nt:
        W.data = W.data * _nt_factor(c, p.w_syn_by_nt)[W.indices]      # csr: indices = PREsynaptic column
    if p.conn_cap > 0:
        W.data = np.sign(W.data) * np.minimum(np.abs(W.data), np.float32(p.conn_cap))
    path_gain = DEFAULT_PATH_GAIN if p.path_gain is None else p.path_gain
    if path_gain:
        import re
        sc = c.neurons.superclass.fillna("").to_numpy()
        Wc = W.tocoo()
        for pre_re, post_re, f in path_gain:
            pre_m = np.array([bool(re.match(pre_re, t)) for t in sc]); post_m = np.array([bool(re.match(post_re, t)) for t in sc])
            Wc.data[pre_m[Wc.col] & post_m[Wc.row]] *= np.float32(f)
        W = Wc.tocsr()
    type_path_gain = DEFAULT_TYPE_PATH_GAIN if p.type_path_gain is None else p.type_path_gain
    if type_path_gain:
        import re
        ty = c.neurons.type.fillna("").to_numpy()
        Wc = W.tocoo()
        for pre_re, post_re, f in type_path_gain:
            pre_m = np.array([bool(re.match(pre_re, t)) for t in ty]); post_m = np.array([bool(re.match(post_re, t)) for t in ty])
            Wc.data[pre_m[Wc.col] & post_m[Wc.row]] *= np.float32(f)
        W = Wc.tocsr()
    if p.same_type_gain != 1.0:
        types = c.neurons.type.fillna("").to_numpy()
        Wc = W.tocoo()
        same = (types[Wc.row] == types[Wc.col]) & (types[Wc.row] != "")
        Wc.data[same] *= np.float32(p.same_type_gain)
        W = Wc.tocsr()
    return W


def _slow_weights(c: Connectome, p: LIFParams, receptor: ReceptorSigns, spec: SlowSpec) -> dict:
    """The slow (metabotropic / monoamine) matrices of the "full" receptor model, one per active slow class
    ({class name: csr}), in synapse-equivalents: per edge count x slow sign x gain factor, capped like the fast
    synapses (conn_cap on the magnitude); entries with slow sign 0 or of another class are dropped. Path gains and
    same-type damping (fast-synapse stop-gaps) are NOT applied; the fan-in scale and w_syn x slow_gain[class] are
    applied by Brain."""
    if receptor.count is None:
        raise ValueError("the slow term needs receptor_signs(..., with_counts=True)")
    out = {}
    for cls in spec.classes:
        S = c.W.tocsr().copy()
        S.data = receptor.count * receptor.slow_factor(_receptor_gain(p), slow_class=cls)
        if p.conn_cap > 0:
            S.data = np.sign(S.data) * np.minimum(np.abs(S.data), np.float32(p.conn_cap))
        S.eliminate_zeros()
        out[cls] = S
    return out


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad):
        (voltage,) = ctx.saved_tensors
        return grad / (1. + 25. * voltage.abs()).square()


class Brain:
    def __init__(self, c: Connectome, params: LIFParams | None = None, device: str | None = None,
                 seed: int = 0, batch: int = 1, metal_kernels: bool | None = None,
                 cuda_kernels: bool | None = None, cuda_sparse: str = "torch", cuda_compact: bool = True,
                 receptor: ReceptorSigns | None = None):
        """metal_kernels: use the custom Metal kernels (flyverse/metal.py) for the event-driven synaptic input
        and the LIF update; None = automatically on MPS when available.
        receptor: a precomputed connectome.receptor_signs(c, ...) for LIFParams.receptor_model (computed here
        when None; FlyBrain passes the one it shares with the optic lobe)."""
        self.c = c
        self.p = params or LIFParams()
        self.device = resolve(device)
        self.n = c.n
        self.B = int(batch)
        p = self.p

        if p.surrogate_grad:
            if cuda_kernels or metal_kernels or cuda_sparse != "torch":
                raise ValueError("surrogate_grad requires the Torch path; native CUDA/Metal kernels cannot backpropagate")
            if p.event_driven:
                raise ValueError("surrogate_grad requires sparse matmul; event selection discards subthreshold gradients")
            cuda_kernels = metal_kernels = False

        if self.B < 1 or not math.isfinite(p.dt) or p.dt <= 0:
            raise ValueError("batch must be positive and dt must be positive and finite")

        # receptor_model == "full" keeps the Torch path (no native kernel carries g_slow); the slow term itself is
        # active only when some class has a non-zero scale (self.slow, a SlowSpec) -- otherwise nothing is built.
        self._slow_on = p.receptor_model == "full"
        self.slow = _slow_spec(p)
        self._slow_active = self.slow is not None
        self.slow_classes = self.slow.classes if self._slow_active else []
        self.receptor = _receptor(c, p, receptor, with_counts=self._slow_active)
        W = _shaped_weights(c, p, self.receptor)
        S = _slow_weights(c, p, self.receptor, self.slow) if self._slow_active else None
        if p.input_norm_alpha > 0:
            ref = c.reference
            key = repr((p.conn_cap, DEFAULT_PATH_GAIN if p.path_gain is None else p.path_gain,
                        DEFAULT_TYPE_PATH_GAIN if p.type_path_gain is None else p.type_path_gain,
                        p.same_type_gain, _receptor_key(p))
                       + ((tuple(sorted(p.w_syn_by_nt.items())),) if p.w_syn_by_nt else ()))
            if ref is c:
                tot = np.asarray(abs(W).sum(axis=1)).ravel()
                ref._norm_cache[key] = tot
            else:
                if key not in ref._norm_cache:
                    ref._norm_cache[key] = np.asarray(abs(_shaped_weights(ref, p)).sum(axis=1)).ravel()
                tot = ref._norm_cache[key][ref.index_of(c.neurons.bodyId)]
            scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
            import scipy.sparse as sp
            W = (sp.diags(scale) @ W).tocsr()
            if S is not None:
                S = {k: (sp.diags(scale) @ Sk).tocsr() for k, Sk in S.items()}
            self.input_scale = scale
        W = W.copy(); W.data = W.data * np.float32(p.w_syn)
        if S is not None:
            S = {k: Sk.copy() for k, Sk in S.items()}
            for k, Sk in S.items():
                Sk.data = Sk.data * np.float32(p.w_syn * self.slow.gain[k])
        self.event_driven = False if p.surrogate_grad else ((self.device.type != "cuda") if p.event_driven is None else bool(p.event_driven))
        adapt_map_early = DEFAULT_ADAPT_BY_TYPE if p.adapt_by_type is None else p.adapt_by_type
        if adapt_map_early and cuda_kernels:
            import warnings
            warnings.warn("per-type adaptation (adapt_by_type) is not in the native LIF kernel yet; using the Torch path")
        if self._slow_on and cuda_kernels:
            import warnings
            warnings.warn("the slow receptor term (receptor_model='full') is not in the native LIF kernel; using the Torch path")
        self.cuda = cuda.use(self.device, False if (adapt_map_early or self._slow_on) else cuda_kernels)
        if cuda_sparse not in ("torch", "warp"):
            raise ValueError("cuda_sparse must be torch or warp")
        if cuda_sparse == "warp" and not self.cuda:
            if cuda_kernels and (adapt_map_early or self._slow_on):          # the model option, not the caller, turned the kernels off
                import warnings
                warnings.warn("cuda_sparse=warp needs the native kernels, which this model configuration disables; using cuSPARSE")
                cuda_sparse = "torch"
            else:
                raise ValueError("cuda_sparse=warp requires CUDA kernels")
        self.cuda_sparse, self.cuda_compact = cuda_sparse, cuda_compact
        self.metal = (metal.use(self.device, metal_kernels) and self.event_driven and p.weight_dtype == "float32"
                      and not self._slow_on)
        if metal_kernels and not self.metal:
            raise ValueError("Metal kernels need the event-driven backend with float32 weights (and no slow receptor term)")
        if p.weight_dtype not in ("float32", "float16"):
            raise ValueError("weight_dtype must be float32 or float16")
        if p.weight_dtype == "float16" and self.device.type != "cuda":
            raise ValueError("float16 sparse weights require CUDA")
        self._W_cpu = W
        self._W_slow_cpu = S
        self.K = 1; self._kvec = None; self._W_k = {}; self._acc = {}; self._W_slow_k = {}
        self.set_weights(W)
        self._set_slow_weights(S)

        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        B, N, dev = self.B, self.n, self.device
        self.v = torch.full((B, N), p.v_rest, device=dev)
        self.g = torch.zeros(B, N, device=dev)
        # slow receptor tone (mV): per active slow class (K, B, N) and its sum (B, N); K = 0 when the term is off
        self.g_slow_cls = torch.zeros(len(self.slow_classes), B, N, device=dev)
        self.g_slow = torch.zeros(B, N, device=dev)
        self._a_slow = [math.exp(-p.dt / self.slow.tau[k]) for k in self.slow_classes]
        if self._slow_active and self.slow.mode == "threshold":
            self._th_max = torch.tensor(SLOW_THRESHOLD_MAX_FRAC * self.slow.norm_mv, device=dev)
        self._th = torch.full((B, N), p.v_th, device=dev) if self._slow_active and self.slow.mode == "threshold" else None
        self.refrac = torch.zeros(B, N, device=dev)          # ms left in refractory period
        self.drive = torch.zeros(B, N, device=dev)           # I_ext (mV), set by the environment
        self.poisson_p = torch.zeros(B, N, device=dev)       # per-step spike prob for forced neurons
        self.rate = torch.zeros(B, N, device=dev)            # running rate estimate (Hz)
        self.spikes = torch.zeros(B, N, device=dev)          # spikes this step (0/1)
        self.spike_counts = torch.zeros(B, N, device=dev)
        self.record_activity = False
        self.adapt = torch.zeros(B, N, device=dev)           # adaptation current (mV)
        self.res = torch.ones(B, N, device=dev)              # synaptic resource x (STD)
        self.active = torch.ones(N, device=dev)              # 0 = frozen (simulated elsewhere)
        self.n_delay = max(1, int(round(p.delay / p.dt)))
        self.spike_buf = torch.zeros(self.n_delay, B, N, device=dev)
        self.buf_pos = 0
        self._rate_np = None                                  # CPU snapshot of rate[0] (B = 1 readouts), see rate_np()
        self._rate_np_key = None
        self.t = 0.0                                          # ms
        self.step_count = 0
        self._poisson_on = False
        self._a_m = math.exp(-p.dt / p.tau_m)
        self._a_s = math.exp(-p.dt / p.tau_syn)
        self._a_r = math.exp(-p.dt / p.rate_tau)
        self._a_ad = math.exp(-p.dt / p.adapt_tau) if p.adapt_jump > 0 else 0.0
        adapt_map = DEFAULT_ADAPT_BY_TYPE if p.adapt_by_type is None else p.adapt_by_type
        self.adapt_jump_vec = None
        if adapt_map:
            import re
            types_ = c.neurons.type.fillna("").to_numpy()
            aj = np.full(N, p.adapt_jump, dtype=np.float32)
            for pat, jump in adapt_map.items():
                aj[np.array([bool(re.match(pat, t_)) for t_ in types_])] = jump
            self.adapt_jump_vec = torch.from_numpy(aj).to(dev)
            self._a_ad = math.exp(-p.dt / p.adapt_tau)
        std_map = DEFAULT_STD_U_BY_TYPE if p.std_u_by_type is None else p.std_u_by_type
        u = np.full(N, p.std_u, dtype=np.float32)
        if std_map:
            import re
            types = c.neurons.type.fillna("").to_numpy()
            for pat, uu in std_map.items():
                u[np.array([bool(re.match(pat, t)) for t in types])] = uu
        self.std_u_vec = torch.from_numpy(u).to(dev)
        self._std_on = bool((u > 0).any())
        self._a_std = math.exp(-p.dt / p.std_tau) if self._std_on else 1.0
        if self.cuda:
            self._cuda_P = torch.tensor([p.v_rest, p.v_reset, p.v_th, self._a_m, p.dt, p.t_ref, p.adapt_jump,
                                        self._a_ad, self._a_std, self._a_r, (1-self._a_r)*1000.0/p.dt],
                                       dtype=torch.float32, device=dev)
        if self.metal:   # coefficients for metal.lif_update, see SOURCE
            self._P = torch.tensor([p.v_rest, p.v_reset, p.v_th, self._a_m, p.dt, p.t_ref, p.adapt_jump, self._a_ad,
                                    self._a_std, self._a_r, (1 - self._a_r) * 1000.0 / p.dt], dtype=torch.float32, device=dev)

    def set_weights(self, W) -> None:
        """Install a (post, pre) scipy sparse matrix of synaptic weights in mV (already scaled by w_syn)."""
        import scipy.sparse as sp
        self._weights_version = getattr(self, "_weights_version", 0) + 1
        if self.event_driven:
            Wc = sp.csc_matrix(W); Wc.sum_duplicates(); Wc.sort_indices()   # pre-major: outputs of each neuron
            self.W = None
            itype = torch.int32 if self.metal or self.cuda else torch.int64
            if self.cuda and max(*Wc.shape, Wc.nnz) >= 2**31:
                raise ValueError("CUDA events require 32-bit dimensions and indices")
            self._out_ptr = torch.from_numpy(Wc.indptr.astype(np.int64)).to(self.device, itype)
            self._out_post = torch.from_numpy(Wc.indices.astype(np.int64)).to(self.device, itype)
            dtype = torch.float16 if self.p.weight_dtype == "float16" else torch.float32
            self._out_w = torch.from_numpy(Wc.data.astype(np.float32)).to(self.device, dtype)
            self._out_pre = (torch.as_tensor(np.flatnonzero(np.diff(Wc.indptr)), device=self.device, dtype=torch.int32)
                             if self.cuda and self.cuda_compact else None)
        else:
            dtype = torch.float16 if self.p.weight_dtype == "float16" else torch.float32
            self.W = self._sparse(W, dtype)
            self._syn_input = torch.empty(self.n, self.B, dtype=torch.float32, device=self.device)

    def _set_slow_weights(self, S) -> None:
        """Install the slow-term matrices ({class: csr}, mV per transmitted spike; None = no slow term) as plain torch
        sparse matrices in class order (self.slow_classes)."""
        self.W_slow = None if S is None else [sparse_matrix(S[k], self.device, dtype=torch.float32) for k in self.slow_classes]

    def _slow_update(self, W_slow: list, x: torch.Tensor, a_slow=None) -> None:
        """Decay every class tone and add the slow input of transmitted spikes x (B, N); refresh the summed tone."""
        for k, Wk in enumerate(W_slow):
            gk = self.g_slow_cls[k]
            gk.mul_(self._a_slow[k] if a_slow is None else a_slow[k])
            gk.add_((Wk @ x.T.contiguous()).T)
        torch.sum(self.g_slow_cls, dim=0, out=self.g_slow)

    def _slow_decay(self, a_slow=None) -> None:
        """Decay only (clocked path: the input is added per clock)."""
        for k in range(len(self.slow_classes)):
            self.g_slow_cls[k].mul_(self._a_slow[k] if a_slow is None else a_slow[k])

    def _slow_add(self, W_slow: list, x: torch.Tensor) -> None:
        for k, Wk in enumerate(W_slow):
            self.g_slow_cls[k].add_((Wk @ x.T.contiguous()).T)

    def _membrane_target(self) -> torch.Tensor:
        """v_rest + synaptic input + drive - adaptation, with the slow tone applied per slow_mode."""
        p = self.p
        if not self._slow_active or self.slow.mode == "threshold":
            return p.v_rest + self.g + self.drive - self.adapt
        if self.slow.mode == "additive":
            return p.v_rest + self.g + self.drive - self.adapt + self.g_slow
        # "gain": the fast synaptic input scaled by the normalised tone
        f = torch.clamp(1.0 + self.g_slow / self.slow.norm_mv, self.slow.clip[0], self.slow.clip[1])
        return p.v_rest + self.g * f + self.drive - self.adapt

    def _threshold(self):
        """The spike threshold: v_th, or v_th - min(g_slow, 0.9 gap) in "threshold" mode (a (B, N) tensor)."""
        if self._th is None:
            return self.p.v_th
        torch.minimum(self.g_slow, self._th_max, out=self._th)
        self._th.neg_().add_(self.p.v_th)
        return self._th

    def _sparse(self, W, dtype):
        return cuda.CSR(W, self.device, dtype) if self.cuda_sparse == "warp" else sparse_matrix(W, self.device, dtype=dtype)

    def _matmul_add(self, W, x: torch.Tensor) -> None:
        if isinstance(W, cuda.CSR):
            self.g.add_(W.matvec(x))
            return
        if self.p.weight_dtype == "float16":
            # cuSPARSE accepts half A/B with float C and compute type. Passing an
            # fp32 out buffer avoids the overflow/rounding of a half-precision output.
            torch.mm(W, x.T.to(torch.float16).contiguous(), out=self._syn_input)
            self.g.add_(self._syn_input.T)
        else:
            self.g.add_((W @ x.T.contiguous()).T)                      # contiguous: 4x faster spmm

    def prune(self, idx) -> None:
        """Drop every synapse from and onto neurons `idx` (frozen rate units) from the LIF matrix."""
        import scipy.sparse as sp
        keep = np.ones(self.n, np.float32); keep[np.asarray(idx)] = 0.0
        W = (sp.diags(keep) @ self._W_cpu @ sp.diags(keep)).tocsr(); W.eliminate_zeros()
        self._W_cpu = W
        self.set_weights(W)
        if self._W_slow_cpu is not None:
            S = {}
            for k, Sk in self._W_slow_cpu.items():
                Sk = (sp.diags(keep) @ Sk @ sp.diags(keep)).tocsr(); Sk.eliminate_zeros(); S[k] = Sk
            self._W_slow_cpu = S
            self._set_slow_weights(S)
        if self._kvec is not None:
            self._split_weights()

    def set_clocks(self, multiplier) -> None:
        """Integrate neuron i every multiplier[i] base steps (dt_i = multiplier[i] * dt). See LIFParams.dt_by_module."""
        k = np.asarray(multiplier, dtype=np.int64)
        if k.shape != (self.n,) or (k < 1).any():
            raise ValueError("multiplier must be an (N,) array of positive integers")
        if self.event_driven:
            raise ValueError("per-module clocks need the sparse-matmul backend")
        p = self.p
        if int(k.max()) * p.dt > p.delay + 1e-9:
            raise ValueError("the slowest clock must not exceed the synaptic delay")
        self._kvec = k; self.K = int(np.lcm.reduce(np.unique(k)))
        dev = self.device
        kt = torch.from_numpy(k.astype(np.float32)).to(dev)
        self._phase = []
        for ph in range(self.K):
            u = torch.from_numpy(((ph % k) == 0).astype(np.float32)).to(dev)
            one = torch.ones_like(u)
            coef = lambda tau: torch.where(u > 0, torch.exp(-p.dt * kt / tau), one)
            self._phase.append({"u": u, "a_m": coef(p.tau_m), "a_s": coef(p.tau_syn), "a_r": coef(p.rate_tau),
                                "a_slow": [coef(self.slow.tau[k]) for k in self.slow_classes],
                                "a_ad": coef(p.adapt_tau) if p.adapt_jump > 0 else torch.zeros_like(u),
                                "a_std": coef(p.std_tau) if self._std_on else one,
                                "dt": u * kt * p.dt, "rate_gain": u * (1 - torch.exp(-p.dt * kt / p.rate_tau)) * 1000.0 / (kt * p.dt),
                                "pois": u * kt})
            if self.cuda:
                c = self._phase[-1]
                c["cuda"] = torch.stack([c[key] for key in
                    ("a_m", "dt", "a_ad", "a_std", "a_r", "rate_gain", "u", "pois")])
        self._split_weights()

    def _split_weights(self) -> None:
        import scipy.sparse as sp
        self._weights_version += 1
        dtype = torch.float16 if self.p.weight_dtype == "float16" else torch.float32
        self._W_k = {}; self._acc = {}; self._W_slow_k = {}
        for kk in np.unique(self._kvec):
            rows = (self._kvec == kk).astype(np.float32)
            Wk = (sp.diags(rows) @ self._W_cpu).tocsr(); Wk.eliminate_zeros()
            self._W_k[int(kk)] = self._sparse(Wk, dtype)
            if self._W_slow_cpu is not None:
                mats = []
                for cls in self.slow_classes:
                    Sk = (sp.diags(rows) @ self._W_slow_cpu[cls]).tocsr(); Sk.eliminate_zeros()
                    mats.append(sparse_matrix(Sk, self.device, dtype=torch.float32))
                self._W_slow_k[int(kk)] = mats
            if kk > 1:
                self._acc[int(kk)] = torch.zeros(self.B, self.n, device=self.device)
        self.W = None
        self.W_slow = None

    def _add_synaptic_input(self, x: torch.Tensor) -> None:
        """g += W @ x for transmitted spikes x (B, N); event-driven or one sparse matmul for all B brains."""
        if not self.event_driven:
            self._matmul_add(self.W, x)
            return
        if self.metal:
            metal.event_scatter(self._out_ptr, self._out_post, self._out_w, x, self.g)
            return
        if self.cuda:
            cuda.event_scatter(self._out_ptr, self._out_post, self._out_w, x, self.g, self._out_pre)
            return
        nz = torch.nonzero(x)                                           # (K, 2) [brain, pre]; syncs with host
        if nz.shape[0] == 0:
            return
        bidx, pre = nz[:, 0], nz[:, 1]
        start = self._out_ptr[pre]
        cnt = self._out_ptr[pre + 1] - start
        total = int(cnt.sum())
        if total == 0:
            return
        seg = torch.repeat_interleave(torch.arange(nz.shape[0], device=self.device), cnt, output_size=total)
        first = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt, output_size=total)
        e = start[seg] + torch.arange(total, device=self.device) - first          # synapse ids, pre-major
        tgt = self._out_post[e] + bidx[seg] * self.n
        self.g.view(-1).index_add_(0, tgt, self._out_w[e] * x[bidx[seg], pre[seg]])

    # ------------------------------------------------------------------ input helpers
    def _idx(self, idx) -> torch.Tensor:
        return torch.as_tensor(idx, device=self.device, dtype=torch.long)

    def set_drive(self, idx, mv) -> None:
        """Injected current (mV) of neurons `idx`: scalar, (len(idx),) or (B, len(idx))."""
        if getattr(self, "_input_target", None) is not None:
            return self._input_target("drive_mv", idx, mv)
        value = torch.as_tensor(mv, dtype=torch.float32, device=self.device)
        if self.p.surrogate_grad:
            self.drive = self.drive.clone()
        self.drive[:, self._idx(idx)] = value

    def set_poisson(self, idx, rate_hz) -> None:
        """Force Poisson spikes at rate_hz on neurons `idx`: scalar, (len(idx),) or (B, len(idx))."""
        if getattr(self, "_input_target", None) is not None:
            return self._input_target("poisson_hz", idx, rate_hz)
        r = torch.as_tensor(rate_hz, dtype=torch.float32, device=self.device)
        self.poisson_p[:, self._idx(idx)] = r * (self.p.dt / 1000.0)
        self._poisson_on = bool((r > 0).any()) or bool(self.poisson_p.count_nonzero() > 0)

    def freeze(self, idx) -> None:
        """Neurons `idx` never spike in the LIF (they are simulated as rate units in optic.py)."""
        self.active[self._idx(idx)] = 0.0

    def reset(self, rows=None) -> None:
        """Reset the state of brains `rows` (all if None) to rest."""
        if self.p.surrogate_grad:
            self.detach_state()
        sel = slice(None) if rows is None else torch.as_tensor(np.asarray(rows), device=self.device, dtype=torch.long)
        self.v[sel] = self.p.v_rest
        for t in (self.g, self.g_slow, self.refrac, self.drive, self.poisson_p, self.rate, self.spikes, self.adapt,
                  self.spike_counts):
            t[sel] = 0.0
        self.g_slow_cls[:, sel] = 0.0
        self.res[sel] = 1.0
        self.spike_buf[:, sel] = 0.0
        for acc in self._acc.values():
            acc[sel] = 0.0
        self._rate_np_key = None

    # ------------------------------------------------------------------ dynamics
    def step(self, n_steps: int = 1) -> None:
        if self.p.surrogate_grad:
            if self.cuda or self.metal:
                raise ValueError("surrogate_grad cannot use native kernels")
            return self._step_surrogate(n_steps)
        return self._step_inference(n_steps)

    def detach_state(self):
        """End a truncated backpropagation window without resetting simulated state."""
        for name in ("v", "g", "g_slow", "g_slow_cls", "refrac", "drive", "poisson_p", "rate", "spikes",
                     "adapt", "res", "spike_buf", "spike_counts"):
            setattr(self, name, getattr(self, name).detach().clone())
        self._acc = {k: v.detach().clone() for k, v in self._acc.items()}

    def _step_surrogate(self, n_steps):
        """Functional counterpart of the Torch LIF update, including clocks and slow tone."""
        p = self.p
        for _ in range(n_steps):
            phase = self._phase[self.step_count % self.K] if self._kvec is not None else None
            a_s = phase["a_s"] if phase else self._a_s
            a_m = phase["a_m"] if phase else self._a_m
            a_r = phase["a_r"] if phase else self._a_r
            a_ad = phase["a_ad"] if phase else self._a_ad
            a_std = phase["a_std"] if phase else self._a_std
            dt = phase["dt"] if phase else p.dt
            x = self.spike_buf[self.buf_pos]
            self.g = self.g * a_s
            slow = [self.g_slow_cls[k] * (phase["a_slow"][k] if phase else self._a_slow[k])
                    for k in range(len(self.slow_classes))]
            weights = self._W_k if phase else {1: self.W}
            for kk, W in weights.items():
                if kk == 1:
                    incoming = x
                else:
                    self._acc[kk] = self._acc[kk] + x
                    if self.step_count % kk:
                        continue
                    incoming = self._acc[kk]
                    self._acc[kk] = torch.zeros_like(incoming)
                self.g = self.g + (W @ incoming.T.contiguous()).T
                if self._slow_active:
                    matrices = self._W_slow_k[kk] if phase else self.W_slow
                    slow = [value + (Wk @ incoming.T.contiguous()).T for value, Wk in zip(slow, matrices)]
            if self._slow_active:
                self.g_slow_cls = torch.stack(slow)
                self.g_slow = self.g_slow_cls.sum(0)
            target = self._membrane_target()
            voltage = target + (self.v - target) * a_m
            voltage = torch.where(self.refrac > 0, p.v_reset, voltage)
            self.refrac = (self.refrac - dt).clamp_min(0)
            threshold = p.v_th
            if self._slow_active and self.slow.mode == "threshold":
                threshold = p.v_th - self.g_slow.clamp(max=SLOW_THRESHOLD_MAX_FRAC * self.slow.norm_mv)
            spikes = _SurrogateSpike.apply(voltage - threshold) * self.active
            if phase:
                spikes = spikes * phase["u"]
            if self._poisson_on:
                probability = self.poisson_p * phase["pois"] if phase else self.poisson_p
                forced = (torch.rand(voltage.shape, generator=self.gen, device=self.device) < probability).float()
                spikes = torch.maximum(spikes, forced)
            # Hard reset decisions have no derivative; the spike signal carries the surrogate.
            fired = spikes.detach() > 0
            self.v = torch.where(fired, p.v_reset, voltage)
            self.refrac = torch.where(fired, p.t_ref, self.refrac)
            self.spikes = spikes
            if self.adapt_jump_vec is not None or p.adapt_jump > 0:
                jump = self.adapt_jump_vec if self.adapt_jump_vec is not None else p.adapt_jump
                self.adapt = self.adapt * a_ad + spikes * jump
            transmitted = spikes * self.res if self._std_on else spikes
            if self._std_on:
                self.res = 1. - (1. - torch.where(fired, self.res * (1. - self.std_u_vec), self.res)) * a_std
            buffers = list(self.spike_buf.unbind(0))
            buffers[self.buf_pos] = transmitted
            self.spike_buf = torch.stack(buffers)
            self.buf_pos = (self.buf_pos + 1) % self.n_delay
            if self.record_activity:
                self.spike_counts = self.spike_counts + spikes
            rate_gain = phase["rate_gain"] if phase else (1. - self._a_r) * 1000. / p.dt
            self.rate = self.rate * a_r + spikes * rate_gain
            self.t += p.dt
            self.step_count += 1

    @torch.no_grad()
    def _step_inference(self, n_steps: int = 1) -> None:
        if self._kvec is not None:
            self._step_clocked(n_steps)
            return
        if self.metal:
            self._step_metal(n_steps)
            return
        if self.cuda:
            self._step_cuda(n_steps)
            return
        p = self.p
        for _ in range(n_steps):
            # synaptic input from spikes emitted `delay` ago: one sparse matmul for all B brains
            self.g.mul_(self._a_s)
            self._add_synaptic_input(self.spike_buf[self.buf_pos])          # spikes emitted `delay` ago, (B, N)
            if self._slow_active:
                # slow (metabotropic / monoamine) tone per class: same delayed spikes, its own time constant
                self._slow_update(self.W_slow, self.spike_buf[self.buf_pos])

            # membrane (exponential Euler with g and drive held constant over the step)
            target = self._membrane_target()
            torch.add(target, (self.v - target) * self._a_m, out=self.v)
            in_ref = self.refrac > 0
            torch.where(in_ref, torch.full_like(self.v, p.v_reset), self.v, out=self.v)
            torch.clamp(self.refrac - p.dt, min=0.0, out=self.refrac)

            torch.mul((self.v >= self._threshold()).float(), self.active, out=self.spikes)
            spikes = self.spikes
            if self._poisson_on:
                forced = (torch.rand(self.v.shape, generator=self.gen, device=self.device) < self.poisson_p).float()
                torch.maximum(spikes, forced, out=spikes)
            fired = spikes > 0
            torch.where(fired, torch.full_like(self.v, p.v_reset), self.v, out=self.v)
            torch.where(fired, torch.full_like(self.refrac, p.t_ref), self.refrac, out=self.refrac)
            if self.adapt_jump_vec is not None:
                self.adapt.mul_(self._a_ad).add_(spikes * self.adapt_jump_vec)
            elif p.adapt_jump > 0:
                self.adapt.mul_(self._a_ad).add_(spikes, alpha=p.adapt_jump)

            if self._std_on:
                # transmit with the currently available resource, then deplete and recover
                self.spike_buf[self.buf_pos] = spikes * self.res
                torch.where(fired, self.res * (1 - self.std_u_vec), self.res, out=self.res)
                torch.sub(1.0, (1.0 - self.res) * self._a_std, out=self.res)
            else:
                self.spike_buf[self.buf_pos] = spikes
            self.buf_pos = (self.buf_pos + 1) % self.n_delay
            if self.record_activity:
                self.spike_counts.add_(spikes)
            self.rate.mul_(self._a_r).add_(spikes * ((1 - self._a_r) * 1000.0 / p.dt))
            self.t += p.dt
            self.step_count += 1

    @torch.no_grad()
    def _step_cuda(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.g.mul_(self._a_s)
            self._add_synaptic_input(self.spike_buf[self.buf_pos])
            rnd = torch.rand(self.v.shape, generator=self.gen, device=self.device) if self._poisson_on else self.poisson_p
            cuda.lif_update(self, rnd)
            self.buf_pos = (self.buf_pos + 1) % self.n_delay
            self.t += self.p.dt
            self.step_count += 1

    @torch.no_grad()
    def _step_metal(self, n_steps: int) -> None:
        """The step loop as three launches: synaptic scatter, Poisson draws (torch generator), fused update."""
        p = self.p
        flags = (1 if self._poisson_on else 0) | (2 if self._std_on else 0) | (4 if self.record_activity else 0) | (8 if p.adapt_jump > 0 else 0)
        for _ in range(n_steps):
            self.g.mul_(self._a_s)
            self._add_synaptic_input(self.spike_buf[self.buf_pos])
            rnd = torch.rand(self.v.shape, generator=self.gen, device=self.device) if self._poisson_on else self.poisson_p
            metal.lif_update(self.v, self.g, self.drive, self.adapt, self.refrac, self.active, self.poisson_p, rnd, self.res,
                             self.std_u_vec, self.spikes, self.spike_buf[self.buf_pos], self.rate, self.spike_counts, self._P, flags)
            self.buf_pos = (self.buf_pos + 1) % self.n_delay
            self.t += p.dt
            self.step_count += 1

    @torch.no_grad()
    def _step_clocked(self, n_steps: int) -> None:
        """The step loop with per-neuron clocks: non-updating neurons get coefficient 1 / mask 0 this phase."""
        p = self.p
        for _ in range(n_steps):
            c = self._phase[self.step_count % self.K]
            x = self.spike_buf[self.buf_pos]
            self.g.mul_(c["a_s"])
            if self._slow_active:
                self._slow_decay(c["a_slow"])
            for kk, Wk in self._W_k.items():
                if kk == 1:
                    self._matmul_add(Wk, x)
                    if self._slow_active:
                        self._slow_add(self._W_slow_k[kk], x)
                else:
                    acc = self._acc[kk]; acc.add_(x)
                    if (self.step_count % kk) == 0:
                        self._matmul_add(Wk, acc)
                        if self._slow_active:
                            self._slow_add(self._W_slow_k[kk], acc)
                        acc.zero_()
            if self._slow_active:
                torch.sum(self.g_slow_cls, dim=0, out=self.g_slow)
            if self.cuda:
                rnd = torch.rand(self.v.shape, generator=self.gen, device=self.device) if self._poisson_on else self.poisson_p
                cuda.lif_update(self, rnd, c["cuda"])
                self.buf_pos = (self.buf_pos + 1) % self.n_delay
                self.t += p.dt
                self.step_count += 1
                continue
            target = self._membrane_target()
            torch.add(target, (self.v - target) * c["a_m"], out=self.v)
            in_ref = self.refrac > 0
            torch.where(in_ref, torch.full_like(self.v, p.v_reset), self.v, out=self.v)
            torch.clamp(self.refrac - c["dt"], min=0.0, out=self.refrac)
            torch.mul((self.v >= self._threshold()).float() * c["u"], self.active, out=self.spikes)
            spikes = self.spikes
            if self._poisson_on:
                forced = (torch.rand(self.v.shape, generator=self.gen, device=self.device) < self.poisson_p * c["pois"]).float()
                torch.maximum(spikes, forced, out=spikes)
            fired = spikes > 0
            torch.where(fired, torch.full_like(self.v, p.v_reset), self.v, out=self.v)
            torch.where(fired, torch.full_like(self.refrac, p.t_ref), self.refrac, out=self.refrac)
            if self.adapt_jump_vec is not None:
                self.adapt.mul_(c["a_ad"]).add_(spikes * self.adapt_jump_vec)
            elif p.adapt_jump > 0:
                self.adapt.mul_(c["a_ad"]).add_(spikes, alpha=p.adapt_jump)
            if self._std_on:
                self.spike_buf[self.buf_pos] = spikes * self.res
                torch.where(fired, self.res * (1 - self.std_u_vec), self.res, out=self.res)
                torch.sub(1.0, (1.0 - self.res) * c["a_std"], out=self.res)
            else:
                self.spike_buf[self.buf_pos] = spikes
            self.buf_pos = (self.buf_pos + 1) % self.n_delay
            if self.record_activity:
                self.spike_counts.add_(spikes)
            self.rate.mul_(c["a_r"]).add_(spikes * c["rate_gain"])
            self.t += p.dt
            self.step_count += 1

    def run_ms(self, ms: float) -> None:
        self.step(int(round(ms / self.p.dt)))

    # ------------------------------------------------------------------ readout helpers
    def rate_np(self) -> np.ndarray:
        """(N,) CPU copy of the B = 1 rates, fetched once per step however many readouts ask (every
        device->host copy is a sync; the demo makes ~30 readouts per frame)."""
        key = (self.step_count, self.t)
        if self._rate_np_key != key:
            self._rate_np = self.rate[0].detach().cpu().numpy()
            self._rate_np_key = key
        return self._rate_np

    def rates(self, idx) -> np.ndarray:
        """(len(idx),) for B = 1, else (B, len(idx))."""
        if self.B == 1:
            return self.rate_np()[np.asarray(idx)]
        return self.rate[:, self._idx(idx)].detach().cpu().numpy()

    def mean_rate(self, idx):
        """float for B = 1, else (B,) array."""
        if len(idx) == 0:
            return 0.0 if self.B == 1 else np.zeros(self.B)
        if self.B == 1:
            return float(self.rate_np()[np.asarray(idx)].mean())
        return self.rate[:, self._idx(idx)].mean(dim=1).detach().cpu().numpy()

    def total_spikes(self):
        cached = getattr(self, "_cuda_spike_total", None)
        if self.cuda and cached is not None and cached[0] == (self.step_count,self.t,self.spikes._version):
            value = cached[1]
            return float(value[0]) if self.B == 1 else value.copy()
        s = self.spikes.sum(dim=1)
        return float(s[0]) if self.B == 1 else s.cpu().numpy()


def drive_from_intensity(intensity, gain: float = 30.0, half: float = 0.02):
    """Saturating light -> injected current (mV): 30*I/(0.02+I) (doomfly). Threshold-crossing at I~0.006."""
    return gain * intensity / (half + intensity)
