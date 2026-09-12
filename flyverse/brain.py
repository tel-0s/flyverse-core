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
    # ---- receptor model (docs/NT_INTEGRATION.md step 5; docs/audits/receptor_rules.md): OPTIONAL, default off.
    # None: the present rule (a synapse carries NT_SIGN of its presynaptic cell); the weights are byte-identical
    #   to the model without this block.
    # "sign": where (postsynaptic type, presynaptic transmitter) has a row in flyverse/data/receptors_by_type.csv
    #   (29.7 % of edges / 25.7 % of |W| synapses of the whole CNS; optic module 52 %), the fast sign of that row
    #   replaces the presynaptic sign (glutamate -> +1 on iGluR targets, 0 where the target has no fast receptor
    #   for that transmitter, ...); unmatched edges keep NT_SIGN. Applied to abs(W) BEFORE the connection cap.
    # "sign+gain": as "sign", times the gain-class factor `receptor_gain` of the row (expression tertiles, not
    #   conductances -- a parameter to sweep), also before the cap.
    # "full": "sign+gain" plus a SLOW conductance g_slow per neuron driven by the row's slow (metabotropic /
    #   monoamine) sign and gain class through a second sparse matrix W_slow (entries with a non-zero slow sign
    #   only: mAChR, GABA-B, mGluR, dopamine / octopamine / serotonin receptors), time constant `slow_tau_ms`,
    #   scale `slow_gain` x w_syn per synapse, added to the membrane input like g. Monoamine synapses are explicit
    #   zeros in the cached W, so their counts come from cache/sign0_counts.npz (connectome.sign0_counts, built
    #   from the raw weights table on first use). The native CUDA / Metal kernels do not carry g_slow: "full"
    #   forces the Torch path (a warning if kernels were requested), like adapt_by_type.
    receptor_model: str | None = None
    receptor_net_rule: str = "class"          # "class" | "abs" | "nonmda": which column set of the table decides the sign
    receptor_nt_class_fallback: bool = False  # unprofiled targets take the Davis 2020 whole-class baseline (tier nt_class)
    receptor_table: str | None = None         # path override for receptors_by_type.csv
    receptor_gain: dict | None = None         # {gain class: factor}; None = DEFAULT_RECEPTOR_GAIN (low 0.5, mid 1, high 1.5)
    slow_tau_ms: float = 200.0                # time constant of g_slow (a parameter: GPCR cascades are 100s of ms)
    slow_gain: float = 0.1                    # g_slow jump per synapse = slow_gain x w_syn x slow sign x gain factor


# Gain-class factors of the receptor model ("sign+gain" / "full"): the table's classes are per-source expression
# tertiles of the winning receptor group ("none" only occurs with sign 0). Parameters, not measurements.
DEFAULT_RECEPTOR_GAIN = {"none": 1.0, "low": 0.5, "mid": 1.0, "high": 1.5}
RECEPTOR_MODELS = (None, "sign", "sign+gain", "full")


# Depression only in the antennal lobe (ORN -> PN and the LN/PN recurrence are documented depressing
# synapses; without it the AL's PN <-> cholinergic-LN loop runs at 300 Hz). Elsewhere depression is off
# because it blocks descending commands.
DEFAULT_TYPE_PATH_GAIN = [(r"^(LC4|LPLC2)$", r"^DNp01$", 3.0),
                          # the central-brain inputs that fire the GF during ordinary walking / feeding in
                          # this model (input-weighted: SAD073, GNG300, DNp70, CL367, PVLP010) are damped;
                          # the animal's GF is notoriously hard to fire except by looms and mechanical shocks
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


def _shaped_weights(c: Connectome, p: LIFParams, receptor: ReceptorSigns | None = None):
    """Apply the calibrated connection rules before fan-in normalization."""
    W = c.W.tocsr().copy()                                          # always a copy: the gain loops below write into W.data
    receptor = _receptor(c, p, receptor)
    if receptor is not None:
        # sign (and gain class) per edge from the receptor table, on the magnitudes, BEFORE the cap; unmatched edges keep
        # NT_SIGN of the presynaptic cell (fast_sign == sign(W.data) there), explicit zeros stay zero
        W.data = np.abs(W.data) * receptor.fast_factor(_receptor_gain(p))
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


def _slow_weights(c: Connectome, p: LIFParams, receptor: ReceptorSigns):
    """The slow (metabotropic / monoamine) matrix of the "full" receptor model, in synapse-equivalents: per edge
    count x slow sign x gain factor, capped like the fast synapses (conn_cap on the magnitude); entries with slow
    sign 0 are dropped. Path gains and same-type damping (fast-synapse stop-gaps) are NOT applied; the fan-in scale
    and w_syn x slow_gain are applied by Brain."""
    if receptor.count is None:
        raise ValueError("the slow term needs receptor_signs(..., with_counts=True)")
    S = c.W.tocsr().copy()
    S.data = receptor.count * receptor.slow_factor(_receptor_gain(p))
    if p.conn_cap > 0:
        S.data = np.sign(S.data) * np.minimum(np.abs(S.data), np.float32(p.conn_cap))
    S.eliminate_zeros()
    return S


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

        if self.B < 1 or not math.isfinite(p.dt) or p.dt <= 0:
            raise ValueError("batch must be positive and dt must be positive and finite")

        self._slow_on = p.receptor_model == "full"
        if self._slow_on and not (math.isfinite(p.slow_tau_ms) and p.slow_tau_ms > 0):
            raise ValueError("slow_tau_ms must be positive and finite")
        self.receptor = _receptor(c, p, receptor, with_counts=self._slow_on)
        W = _shaped_weights(c, p, self.receptor)
        S = _slow_weights(c, p, self.receptor) if self._slow_on else None
        if p.input_norm_alpha > 0:
            ref = c.reference
            key = repr((p.conn_cap, DEFAULT_PATH_GAIN if p.path_gain is None else p.path_gain,
                        DEFAULT_TYPE_PATH_GAIN if p.type_path_gain is None else p.type_path_gain,
                        p.same_type_gain, _receptor_key(p)))
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
                S = (sp.diags(scale) @ S).tocsr()
            self.input_scale = scale
        W = W.copy(); W.data = W.data * np.float32(p.w_syn)
        if S is not None:
            S = S.copy(); S.data = S.data * np.float32(p.w_syn * p.slow_gain)
        self.event_driven = (self.device.type != "cuda") if p.event_driven is None else bool(p.event_driven)
        adapt_map_early = DEFAULT_ADAPT_BY_TYPE if p.adapt_by_type is None else p.adapt_by_type
        if adapt_map_early and cuda_kernels:
            import warnings
            warnings.warn("per-type adaptation (adapt_by_type) is not in the native LIF kernel yet; using the Torch path")
        if self._slow_on and cuda_kernels:
            import warnings
            warnings.warn("the slow receptor term (receptor_model='full') is not in the native LIF kernel; using the Torch path")
        self.cuda = cuda.use(self.device, False if (adapt_map_early or self._slow_on) else cuda_kernels)
        if cuda_sparse not in ("torch", "warp") or (cuda_sparse == "warp" and not self.cuda):
            raise ValueError("cuda_sparse must be torch or warp; warp requires CUDA kernels")
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
        self.g_slow = torch.zeros(B, N, device=dev)          # slow receptor conductance (mV), receptor_model == "full"
        self._a_slow = math.exp(-p.dt / p.slow_tau_ms) if self._slow_on else 1.0
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
        """Install the slow-term matrix (mV per transmitted spike; None = no slow term). Always a plain sparse matmul."""
        self.W_slow = None if S is None else sparse_matrix(S, self.device, dtype=torch.float32)

    def _add_slow_input(self, W_slow, x: torch.Tensor) -> None:
        self.g_slow.add_((W_slow @ x.T.contiguous()).T)

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
            S = (sp.diags(keep) @ self._W_slow_cpu @ sp.diags(keep)).tocsr(); S.eliminate_zeros()
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
                                "a_slow": coef(p.slow_tau_ms) if self._slow_on else one,
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
                Sk = (sp.diags(rows) @ self._W_slow_cpu).tocsr(); Sk.eliminate_zeros()
                self._W_slow_k[int(kk)] = sparse_matrix(Sk, self.device, dtype=torch.float32)
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
        return torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)

    def set_drive(self, idx, mv) -> None:
        """Injected current (mV) of neurons `idx`: scalar, (len(idx),) or (B, len(idx))."""
        self.drive[:, self._idx(idx)] = torch.as_tensor(np.asarray(mv, dtype=np.float32), device=self.device)

    def set_poisson(self, idx, rate_hz) -> None:
        """Force Poisson spikes at rate_hz on neurons `idx`: scalar, (len(idx),) or (B, len(idx))."""
        r = torch.as_tensor(np.asarray(rate_hz, dtype=np.float32), device=self.device)
        self.poisson_p[:, self._idx(idx)] = r * (self.p.dt / 1000.0)
        self._poisson_on = bool(np.any(np.asarray(rate_hz) > 0)) or bool(self.poisson_p.count_nonzero() > 0)

    def freeze(self, idx) -> None:
        """Neurons `idx` never spike in the LIF (they are simulated as rate units in optic.py)."""
        self.active[self._idx(idx)] = 0.0

    def reset(self, rows=None) -> None:
        """Reset the state of brains `rows` (all if None) to rest."""
        sel = slice(None) if rows is None else torch.as_tensor(np.asarray(rows), device=self.device, dtype=torch.long)
        self.v[sel] = self.p.v_rest
        for t in (self.g, self.g_slow, self.refrac, self.drive, self.poisson_p, self.rate, self.spikes, self.adapt,
                  self.spike_counts):
            t[sel] = 0.0
        self.res[sel] = 1.0
        self.spike_buf[:, sel] = 0.0
        for acc in self._acc.values():
            acc[sel] = 0.0
        self._rate_np_key = None

    # ------------------------------------------------------------------ dynamics
    @torch.no_grad()
    def step(self, n_steps: int = 1) -> None:
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
            if self._slow_on:
                # slow (metabotropic / monoamine) conductance: same delayed spikes, its own time constant
                self.g_slow.mul_(self._a_slow)
                self._add_slow_input(self.W_slow, self.spike_buf[self.buf_pos])

            # membrane (exponential Euler with g and drive held constant over the step)
            target = p.v_rest + self.g + self.drive - self.adapt
            if self._slow_on:
                target = target + self.g_slow
            torch.add(target, (self.v - target) * self._a_m, out=self.v)
            in_ref = self.refrac > 0
            torch.where(in_ref, torch.full_like(self.v, p.v_reset), self.v, out=self.v)
            torch.clamp(self.refrac - p.dt, min=0.0, out=self.refrac)

            torch.mul((self.v >= p.v_th).float(), self.active, out=self.spikes)
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
            if self._slow_on:
                self.g_slow.mul_(c["a_slow"])
            for kk, Wk in self._W_k.items():
                if kk == 1:
                    self._matmul_add(Wk, x)
                    if self._slow_on:
                        self._add_slow_input(self._W_slow_k[kk], x)
                else:
                    acc = self._acc[kk]; acc.add_(x)
                    if (self.step_count % kk) == 0:
                        self._matmul_add(Wk, acc)
                        if self._slow_on:
                            self._add_slow_input(self._W_slow_k[kk], acc)
                        acc.zero_()
            if self.cuda:
                rnd = torch.rand(self.v.shape, generator=self.gen, device=self.device) if self._poisson_on else self.poisson_p
                cuda.lif_update(self, rnd, c["cuda"])
                self.buf_pos = (self.buf_pos + 1) % self.n_delay
                self.t += p.dt
                self.step_count += 1
                continue
            target = p.v_rest + self.g + self.drive - self.adapt
            if self._slow_on:
                target = target + self.g_slow
            torch.add(target, (self.v - target) * c["a_m"], out=self.v)
            in_ref = self.refrac > 0
            torch.where(in_ref, torch.full_like(self.v, p.v_reset), self.v, out=self.v)
            torch.clamp(self.refrac - c["dt"], min=0.0, out=self.refrac)
            torch.mul((self.v >= p.v_th).float() * c["u"], self.active, out=self.spikes)
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
            self._rate_np = self.rate[0].cpu().numpy()
            self._rate_np_key = key
        return self._rate_np

    def rates(self, idx) -> np.ndarray:
        """(len(idx),) for B = 1, else (B, len(idx))."""
        if self.B == 1:
            return self.rate_np()[np.asarray(idx)]
        return self.rate[:, self._idx(idx)].cpu().numpy()

    def mean_rate(self, idx):
        """float for B = 1, else (B,) array."""
        if len(idx) == 0:
            return 0.0 if self.B == 1 else np.zeros(self.B)
        if self.B == 1:
            return float(self.rate_np()[np.asarray(idx)].mean())
        return self.rate[:, self._idx(idx)].mean(dim=1).cpu().numpy()

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
