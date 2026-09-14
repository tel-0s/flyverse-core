"""Graded (rate-model) optic lobe coupled to the spiking rest of the CNS, batched over B flies.

Physiology: photoreceptors, lamina cells, medulla columnar cells and T4/T5 are graded-potential
neurons. Simulating them as leaky integrate-and-fire units fails (docs/NOTES.md): Shiu-style 0.275 mV
synapses need ~5,000 synapse-spikes/s per target, so nothing propagates beyond the first medulla layer.
flyvis (Lappalainen et al. 2024) shows the connectome-constrained *rate* model of the optic lobe works.

So: every `ol_intrinsic` neuron (89k: L1-L5, C2/C3, Mi, Tm, TmY, Dm, Pm, T4, T5, LPi ...) is a rate unit

    tau dv_i/dt = -v_i + g_rr * sum_j What_ij (r_j - b) + g_in * sum_p What_ip a_p + g_fb * sum_s What_is s_s
    r_i = clip(v_i + b, 0, 1)

i.e. dynamics of *deviations* from an operating point b = 0.5 at which every unit rests (tonic inputs
are assumed balanced there, as homeostasis would make them). What_ij = sign(NT_j) |synapses_ij| normalised
per target (L2 norm of its input synapse counts by default: coherent fan-in of N equal inputs is
amplified ~sqrt(N), noise is averaged; "l1" = fractions of total input). Photoreceptor activity a_p =
spectral-band contrast of p's column; spiking-neuron feedback s_s = rate_s / 100 Hz (visual centrifugal
neurons). Sign-inverting synapses work in both directions around the operating point (L1 is
hyperpolarised by light, Mi1 is *dis*inhibited: ON pathway; L2/L3 targets get OFF), and the clip keeps
recurrent excitation bounded. The spiking neurons receive injected current

    drive_s (mV) = gain_out * sum_i What_si (r_i - b)

so a visual projection neuron (LC4, LPLC2, LC10 ...) whose optic-lobe inputs rise by 0.3 on average gets
~0.3 * gain_out mV. Everything downstream (central brain, VNC) is the Shiu-style LIF in brain.py.

Batching: all state is (B, n) and `step_frame` takes (B, n_col, 4) radiance; with B = 1 the demo
passes (n_col, 4) and gets (1, N) drive back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from .connectome import Connectome, PHOTORECEPTOR_TYPES, ReceptorSigns
from .retina import Retina
from .device import resolve, sparse_matrix
from . import metal, cuda

FAMILY_OF_TYPE = {"R1-R6": 0, "R7p": 1, "R7y": 2, "R7d": 1, "R7_unclear": 2, "R8p": 3, "R8y": 4,
                  "R8d": 1, "R8_unclear": 4, "R7R8_unclear": 2}


@dataclass
class OpticParams:
    """Parameters of the rate optic lobe (the module docstring gives the model). The last four fields are OPT-IN
    per-presynaptic-stream hooks (docs/audits/optic_stream_hooks.md); all default to None = off, and off is
    bit-identical to the shipped model (tests/test_optic_hooks.py). A stream is the block of W_rr / W_sr entries
    whose presynaptic type full-matches pre_regex and whose postsynaptic type full-matches post_regex; the hooks
    replace what the block MULTIPLIES -- the presynaptic deviation dr_j = r_j - b_j becomes x_j before the sum --
    and never the weights, so every synaptic sign is preserved. Per entry (i <- j), in this order:

        spatial_suppress  [(pre_regex, k, radius_deg)]        u_j = dr_j - k * mean_{j' in N(j)} dr_j'
                          N(j) = the cells of j's type within radius_deg of j's retinal column, j included
                          (columns from interp.trace.column_of_cells); a cell without a column keeps u_j = dr_j
        stream_adapt      [(pre_regex, post_regex, tau_ms, gain)]   y_j = u_j - gain * A_j,
                          A_j <- u_j + (A_j - u_j) exp(-dt / tau_ms) after each substep (one state per entry,
                          OpticLobe.stream_adapt_state, beside the lobe's own adapt)
        stream_rectify    [(pre_regex, post_regex, mode)]     x_j = max(y_j, 0) 'pos' | max(-y_j, 0) 'neg' | |y_j| 'abs'
                          (x_j >= 0: an entry contributes W_ij x_j with the sign of W_ij under every mode)
        input_i = gain_rr (sum_{j unmatched} W_ij dr_j + sum_{j matched} W_ij x_j) + ...;  the output sum W_sr alike.

    The first matching entry of a list wins on overlap. Any of the three active runs the Torch substep (the CUDA /
    Metal optic kernels are untouched). fb_hold [(spiking_pre_regex, rate_post_regex)] removes the matched
    spiking -> rate entries of W_rs at build time (a weight edit; the native kernels stay in use); [('.*', '.*')]
    is gain_fb = 0."""
    tau_ms: float = 10.0
    dt_ms: float = 1.0
    baseline: float = 0.5
    gain_rr: float = 1.0          # recurrent optic-lobe gain
    gain_in: float = 3.0          # photoreceptor contrast -> lamina (lamina cells saturate at ~20% contrast)
    norm: str = "l2"              # input normalisation of the optic-lobe weights: "l1" (fractions) or "l2"
                                  # (coherent fan-in of N equal inputs is amplified ~sqrt(N), noise averaged)
    gain_fb: float = 0.5          # spiking neurons -> optic lobe (rate/100 Hz)
    adapt_tau_ms: float = 400.0   # slow adaptation of every rate unit towards its operating point
    adapt_gain: float = 1.0       # (removes after-images / persistent states from recurrent gain > 1)
    # per-type membrane time constants (ms): the T4/T5 motion detectors need temporally asymmetric
    # inputs (Mi4/Mi9/CT1 and Tm9 are slow, Mi1/Tm3 and Tm1/Tm2/Tm4 fast) -- flyvis learns the same.
    tau_by_type: dict = None
    # per-type operating points (rest rate in [0,1]); a low value makes a unit rectify (ReLU-like),
    # which the T4/T5 motion detectors need (null-direction suppression must be able to clip to zero)
    baseline_by_type: dict = None
    # extra multiplicative gains on specific connections: list of (pre_type_regex, post_type_regex, factor)
    pair_gain: list = None
    gain_out_mv: float = 100.0    # optic lobe delta-rate -> injected current in spiking targets
    out_norm: str = "l1"          # normalisation of the optic-lobe -> spiking weights ("l1" fractions, "l2")
    drive_clip_mv: float = 35.0
    # photoreceptor stage
    tau_lp_ms: float = 10.0
    tau_adapt_ms: float = 300.0
    contrast_clip: float = 2.0
    eps: float = 0.02
    # ---- opt-in per-presynaptic-stream hooks (docs/audits/optic_stream_hooks.md). ALL default to None = off, and off is
    # bit-identical to the model above (tests/test_optic_hooks.py). A 'stream' is the block of synapses from the rate
    # units whose type full-matches pre_regex onto the rate units / spiking cells whose type full-matches post_regex
    # (re.fullmatch on the type string; the weights of the block are the shipped W_rr / W_sr entries, untouched).
    # The hooks change what the block MULTIPLIES -- the presynaptic deviation dr_j = r_j - b_j is replaced by a
    # transformed x_j BEFORE the sum over j -- never the weights, so every synaptic sign is preserved. Per entry
    # (i <- j), in this order (u -> y -> x):
    #   spatial_suppress   u_j = dr_j - k * mean_{j' in N(j)} dr_j'   (N(j): the cells of j's TYPE whose retinal column
    #                      lies within radius_deg of j's column, j itself included; cells without a column: u_j = dr_j)
    #   stream_adapt       y_j = u_j - gain * A_j,  A_j <- u_j + (A_j - u_j) exp(-dt / tau_ms) after each substep
    #                      (a separate fast-adaptation state per hook entry, kept beside OpticLobe.adapt; 'fast-adapting
    #                      inputs', Tanaka & Clark 2020)
    #   stream_rectify     x_j = max(y_j, 0) ('pos') | max(-y_j, 0) ('neg') | |y_j| ('abs')   -- a half-wave / full-wave
    #                      rectification of the presynaptic deviation: x_j >= 0, so an entry contributes W_ij x_j with
    #                      the sign of W_ij under every mode (an inhibitory entry never excites; 'neg' is the OFF
    #                      half-wave signalled as a positive drive, 'abs' both transitions)
    #   contribution_ij = W_ij x_j;  input_i = gain_rr (sum_{j unmatched} W_ij dr_j + sum_{j matched} W_ij x_j) + ...
    # The first matching hook entry wins where two entries of the same list overlap. Any of the three active forces
    # the Torch substep (the CUDA / Metal optic kernels receive the recurrent product only for the plain sum; a
    # 'warp' cuda_sparse request is downgraded to 'torch' with a warning). fb_hold is a build-time weight edit and
    # keeps the native kernels.
    stream_rectify: list = None   # [(pre_regex, post_regex, mode)], mode in {'pos', 'neg', 'abs'}
    stream_adapt: list = None     # [(pre_regex, post_regex, tau_ms, gain)]
    spatial_suppress: list = None # [(pre_regex, k, radius_deg)]  (all postsynaptic targets of the stream)
    # feedback hold: [(spiking_pre_regex, rate_post_regex)] -- the spiking -> rate entries of W_rs in the matched
    # blocks are removed (held at 0). [('.*', '.*')] is gain_fb = 0 (the deterministic lobe of optic_measures.md 6);
    # a narrower block holds one feedback pathway (LoVC16 -> T3, say) and leaves the rest of the feedback stochastic.
    fb_hold: list = None


DEFAULT_TAU_BY_TYPE = {"Mi4": 150.0, "Mi9": 150.0, "CT1": 150.0, "Tm9": 150.0, "L3": 40.0, "Mi1": 8.0, "Tm3": 8.0,
                       "Tm1": 8.0, "Tm2": 8.0, "Tm4": 8.0, "L1": 6.0, "L2": 6.0, "T4a": 10.0, "T4b": 10.0, "T4c": 10.0, "T4d": 10.0,
                       "T5a": 10.0, "T5b": 10.0, "T5c": 10.0, "T5d": 10.0}
# T4/T5 as rectifying (ReLU) units with strong delayed inhibition: this is what makes them direction
# selective (DSI 0.16-0.26 with the correct preferred direction for all 8 subtypes; see NOTES).
T4T5 = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
DEFAULT_BASELINE_BY_TYPE = {t: 0.0 for t in T4T5}
DEFAULT_PAIR_GAIN = [# T4 input x5: the delayed-inhibition arm of the direction-selective motion detector
                     (r"^(Mi4|Mi9|CT1|C3)$", r"^T4[abcd]$", 5.0),
                     # T5 input x5 -- annotated as "delayed inhibition" until session 10, but 78 % of the 101,619 edges it
                     # multiplies are cholinergic Tm9 / Tm4 -> T5 (T5a input: Tm9 28.5 %, Tm4 17.3 %; CT1 / TmY15 GABA are
                     # 22 %), so it is mostly a DRIVE gain on T5's excitatory centre; ablating it costs T5's drive (loom
                     # 29-36 Hz, DSI 0.44 -> 0.15), not its selectivity. A documented stop-gap (docs/audits/optic_measures.md).
                     (r"^(Tm4|Tm9|CT1|TmY15)$", r"^T5[abcd]$", 5.0),
                     # LPi -> LPLC2: the lobula-plate inhibitory interneurons make LPLC2 expansion-selective in the animal;
                     # under the uniform synapse they were a tenth of its T4/T5 excitation, so the fly's own turning
                     # drove the giant fibre. x4 halves the walking GF and keeps the loom (NOTES, session 9).
                     (r"^LPi(34|43)$", r"^LPLC2$", 4.0),
                     # T4/T5 outputs x2 (the factor below; an older comment said x4): rectified, strongly inhibited DS units respond weakly to natural
                     # scenes; this restores drive to LPi / HS / VS / LPLC and the descending neurons
                     (r"^T[45][abcd]$", r".*", 2.0)]
# The loom detectors LC4 / LPLC2 keep x1 optic-lobe drive (the T4/T5 -> LC4/LPLC2 edges carry the x2 above and nothing
# else): in the full sensory context x1.25 already gave 3-4 spontaneous GF escapes per 5 s of walking; the loom margin
# comes from the x3 LC4|LPLC2 -> DNp01 type gain in brain.py instead. Until session 10 this was a literal no-op entry
# (r".*", r"^(LC4|LPLC2)$", 1.0) in the list; removed (factor 1.0, bit-identical weights).


def _csr(D: sp.spmatrix, device, use_metal: bool = False, cuda_sparse: str = "torch"):
    if use_metal:
        return metal.MetalCSR(D, device)
    if cuda_sparse == "warp":
        return cuda.CSR(D, device)
    return sparse_matrix(D.astype(np.float32), device)


def _mv(M, x: torch.Tensor) -> torch.Tensor:
    """(B, rows) = M @ x for x (B, cols), M a torch sparse matrix or a MetalCSR."""
    if isinstance(M, (metal.MetalCSR, cuda.CSR)):
        return M.matvec(x)
    return (M @ x.T.contiguous()).T


class OpticLobe:
    def __init__(self, c: Connectome, retina: Retina, params: OpticParams | None = None, device=None, batch: int = 1,
                 metal_kernels: bool | None = None, cuda_kernels: bool | None = None, cuda_sparse: str = "torch",
                 receptor: ReceptorSigns | None = None, receptor_gain: dict | None = None, slow=None,
                 surrogate_grad: bool = False):
        """metal_kernels: custom Metal kernels for the sparse products and the substep (flyverse/metal.py);
        None = automatically on MPS when available.
        receptor: an optional connectome.receptor_signs(c, ...) (LIFParams.receptor_model): every optic-lobe edge
        (rate <-> rate, photoreceptor -> rate, spiking -> rate, rate -> spiking) takes abs(count) x the row's fast
        sign (x the gain-class factor `receptor_gain`, {class: factor}, when given) instead of sign(NT_pre) x count;
        unmatched edges are unchanged. The normalisation denominators (in_syn / in_syn_l2 of the neuron table) are
        the unmodified totals.
        slow: an optional brain.SlowSpec (brain._slow_spec(LIFParams) under receptor_model == 'full'): the slow
        (metabotropic / monoamine) term of the LIF, applied to the rate units in the same form. Per active slow class k
        the rate unit i carries a tone g_k,i that relaxes with tau_k towards
            scale_k x (gain_fb x sum_s Wslow_is s_s + gain_rr x sum_j Wslow_ij dr_j),   scale_k = gain_k x tau_k / tau_syn,
        where Wslow = count x slow sign x gain-class factor of the row, normalised by the same denominators as the fast
        weights (no pair gains), s = spiking rate / 100 Hz and dr the rate units' deviations -- i.e. the steady tone of a
        presynaptic cell is gain_k x tau_k / tau_syn times the fast input the same synapses would give, as in the LIF.
        g_slow = sum_k g_k acts per slow.mode: 'additive' adds it to the unit's input; 'gain' multiplies the fast
        NET synaptic input (recurrent + photoreceptor + spiking, signed) by clamp(1 + g_slow / (1 - baseline), clip) -- so
        a negative tone disinhibits a net-inhibited unit rather than silencing it -- the
        normaliser is the distance from the operating point to saturation, the rate-model counterpart of the LIF's
        rest-to-threshold gap; 'threshold' shifts the output nonlinearity, r = clip(v + b + min(g_slow, 0.9 (1 - b)), 0, 1).
        Spiking targets of rate-unit monoamine cells (Mi19 serotonin -> central brain) are outside both models (the LIF
        prunes frozen presynaptic cells). The native CUDA / Metal optic kernels do not carry the tone: an active slow
        term runs the Torch substep (a warning if kernels were requested).
        surrogate_grad (LIFParams.surrogate_grad): run the frame functionally so autograd can reach it. Torch only --
        native kernels and cuda_sparse='warp' raise. The substep is the SAME model: it goes through _recurrent() /
        _output() like inference, so the per-stream hooks above act on it identically, and with the hooks off the two
        paths evaluate identical expressions (bit-identical; tests/test_surrogate.py). Only the storage differs (state
        tensors are rebuilt rather than written in place)."""
        self.c, self.r, self.p = c, retina, params or OpticParams()
        self.receptor = receptor
        self.slow = slow if (slow is not None and receptor is not None and slow.gain) else None
        self.B = int(batch)
        self.device = resolve(device)
        self.surrogate_grad = surrogate_grad
        if surrogate_grad:
            if cuda_kernels or metal_kernels or cuda_sparse != "torch":
                raise ValueError("surrogate_grad requires Torch optic operations, not native CUDA/Metal kernels")
            cuda_kernels = metal_kernels = False
        # the per-stream hooks (OpticParams.stream_rectify / stream_adapt / spatial_suppress): Torch substep only
        self._hooks = bool(self.p.stream_rectify or self.p.stream_adapt or self.p.spatial_suppress)
        torch_only = self.slow is not None or self._hooks
        if torch_only and (metal_kernels or cuda_kernels):
            import warnings
            what = "slow receptor term" if self.slow is not None else "per-stream hooks"
            warnings.warn(f"the optic lobe's {what} are not in the native optic kernels; using the Torch substep")
        self.metal = metal.use(self.device, False if torch_only else metal_kernels)
        self.cuda = cuda.use(self.device, False if torch_only else cuda_kernels)
        if cuda_sparse == "warp" and self._hooks and not self.cuda:
            import warnings
            warnings.warn("cuda_sparse 'warp' needs the CUDA optic kernels, which the per-stream hooks disable; using 'torch'")
            cuda_sparse = "torch"
        if cuda_sparse not in ("torch", "warp") or (cuda_sparse == "warp" and not self.cuda):
            raise ValueError("cuda_sparse must be torch or warp; warp requires CUDA kernels")
        if not np.isfinite(self.p.dt_ms) or self.p.dt_ms <= 0:
            raise ValueError("optic dt_ms must be positive and finite")
        self._pending_ms = 0.0
        nrn = c.neurons
        types = nrn.type.fillna("").to_numpy()
        is_pr = nrn.type.isin(PHOTORECEPTOR_TYPES).to_numpy()
        self.rate_idx = np.flatnonzero((nrn.superclass == "ol_intrinsic").to_numpy() & ~is_pr)
        self.pr_idx = retina.pr_index
        is_rate = np.zeros(c.n, bool); is_rate[self.rate_idx] = True
        self.spk_idx = np.flatnonzero(~is_rate & ~is_pr)
        self.n_rate, self.n_pr, self.n_spk = len(self.rate_idx), len(self.pr_idx), len(self.spk_idx)

        W = c.W.tocsr()                                   # (post, pre) signed counts
        if receptor is not None:
            if len(receptor.fast_sign) != W.nnz:
                raise ValueError("receptor signs are not aligned with this connectome's W")
            W = W.copy()
            W.data = np.abs(W.data) * receptor.fast_factor(receptor_gain)     # unmatched edges: identical to W.data
        tot = c.neurons.in_syn.to_numpy()
        Wn = (sp.diags(1.0 / np.maximum(tot, 1.0)) @ W).tocsr()      # L1: fractions of total input
        if self.p.norm == "l2":
            l2 = c.neurons.in_syn_l2.to_numpy()
            Wn_ol = (sp.diags(1.0 / np.maximum(l2, 1.0)) @ W).tocsr()
        else:
            Wn_ol = Wn
        pair_gain = DEFAULT_PAIR_GAIN if self.p.pair_gain is None else self.p.pair_gain

        def apply_pair_gain(M, pre_types, post_types):
            if not pair_gain:
                return M
            import re
            M = M.tocoo()
            for pre_re, post_re, f in pair_gain:
                pre_m = np.array([bool(re.match(pre_re, t)) for t in pre_types]); post_m = np.array([bool(re.match(post_re, t)) for t in post_types])
                sel = pre_m[M.col] & post_m[M.row]
                M.data[sel] *= f
            return M.tocsr()

        rt = types[self.rate_idx]; st = types[self.spk_idx]
        M_rr = apply_pair_gain(Wn_ol[self.rate_idx][:, self.rate_idx], rt, rt)
        M_rs = Wn_ol[self.rate_idx][:, self.spk_idx].tocsr()
        M_sr = apply_pair_gain((Wn_ol if self.p.out_norm == "l2" else Wn)[self.spk_idx][:, self.rate_idx], rt, st)
        if self.p.fb_hold:                                     # spiking -> rate feedback of the matched blocks held at 0
            M_rs = self._hold_feedback(M_rs, st, rt)
        self.W_rr = _csr(M_rr, self.device, self.metal, cuda_sparse)
        self.W_rp = _csr(Wn_ol[self.rate_idx][:, self.pr_idx], self.device, self.metal)
        self.W_rs = _csr(M_rs, self.device, self.metal)
        self.W_sr = _csr(M_sr, self.device, self.metal)
        self.rate_idx_t = torch.as_tensor(self.rate_idx, device=self.device)
        self.spk_idx_t = torch.as_tensor(self.spk_idx, device=self.device)
        # per-stream hooks: the (rate <- rate) and (spiking <- rate) matrices split into the matched blocks (one per
        # distinct (rectify entry, adapt entry) combination) and the rest; the spatial operator; the adaptation states
        self.streams, self.W_rr_rest, self.W_sr_rest, self.G_supp = [], None, None, None
        self.hook_info = {"active": self._hooks}
        if self._hooks:
            self._build_streams(M_rr, M_sr, rt, st, cuda_sparse)
        K_a = len(self.p.stream_adapt or [])
        self.stream_adapt_state = torch.zeros(K_a, self.B, self.n_rate, device=self.device)   # A_k, one per stream_adapt entry

        # slow (metabotropic / monoamine) term: per active class the spiking -> rate and rate -> rate slow matrices
        # (None where a class has no entries), normalised like Wn_ol; scale_k = gain_k tau_k / tau_syn; a_k = exp(-dt / tau_k)
        self.slow_classes = list(self.slow.classes) if self.slow is not None else []
        self.W_slow_rs, self.W_slow_rr, self._slow_scale, self._a_slow = [], [], [], []
        self.slow_entries = {}
        if self.slow is not None:
            if receptor.count is None:
                raise ValueError("the optic lobe's slow term needs receptor_signs(..., with_counts=True)")
            denom = c.neurons.in_syn_l2.to_numpy() if self.p.norm == "l2" else tot
            for cls in self.slow_classes:
                S = c.W.tocsr().copy()
                S.data = receptor.count * receptor.slow_factor(receptor_gain, slow_class=cls)
                S.eliminate_zeros()
                Sn = (sp.diags(1.0 / np.maximum(denom, 1.0)) @ S).tocsr()
                rs = Sn[self.rate_idx][:, self.spk_idx].tocsr(); rr = Sn[self.rate_idx][:, self.rate_idx].tocsr()
                self.slow_entries[cls] = {"spiking_to_rate": int(rs.nnz), "rate_to_rate": int(rr.nnz),
                                          "syn_eq_spiking_to_rate": float(np.abs(S[self.rate_idx][:, self.spk_idx].data).sum()),
                                          "syn_eq_rate_to_rate": float(np.abs(S[self.rate_idx][:, self.rate_idx].data).sum())}
                self.W_slow_rs.append(_csr(rs, self.device) if rs.nnz else None)
                self.W_slow_rr.append(_csr(rr, self.device) if rr.nnz else None)
                self._slow_scale.append(float(self.slow.gain[cls] * self.slow.tau[cls] / self.slow.tau_syn))
                self._a_slow.append(float(np.exp(-self.p.dt_ms / self.slow.tau[cls])))
            self._slow_norm = float(1.0 - self.p.baseline)          # operating point -> saturation
            if self._slow_norm <= 0:
                raise ValueError("the optic slow term needs baseline < 1")
        K = len(self.slow_classes)
        self.g_slow_cls = torch.zeros(K, self.B, self.n_rate, device=self.device)      # per-class tone
        self.g_slow = torch.zeros(self.B, self.n_rate, device=self.device)             # its sum
        self._slow_in_s = torch.zeros(K, self.B, self.n_rate, device=self.device)      # spiking part, held per frame

        # photoreceptor stage: per (column, family) intensity, low-pass + contrast adaptation
        self.pr_family = np.array([FAMILY_OF_TYPE[t] for t in types[self.pr_idx]])
        n_col = retina.n_columns
        self.has = np.zeros((n_col, 5), bool)
        np.add.at(self.has, (retina.pr_column, self.pr_family), True)
        # photoreceptor -> (column, family) averaging matrix, (n_col*5, n_pr)
        cnt = np.zeros((n_col, 5), np.float32)
        np.add.at(cnt, (retina.pr_column, self.pr_family), 1.0)
        rows = retina.pr_column * 5 + self.pr_family
        A = sp.csr_matrix((1.0 / cnt[retina.pr_column, self.pr_family], (rows, np.arange(self.n_pr))), shape=(n_col * 5, self.n_pr))
        self.avg = _csr(A, self.device, self.metal)
        self.sens = torch.from_numpy(retina.pr_sens).to(self.device)                       # (n_pr, 4)
        self.pr_column_t = torch.as_tensor(retina.pr_column, device=self.device)
        self.pr_cell_t = torch.as_tensor(rows, device=self.device)                          # (n_pr,) -> col*5+fam
        self.has_t = torch.from_numpy(self.has.reshape(-1)).to(self.device)
        self.I_lp = torch.zeros(self.B, n_col * 5, device=self.device)
        self.I_mean = torch.zeros(self.B, n_col * 5, device=self.device)
        self._fresh = torch.ones(self.B, dtype=torch.bool, device=self.device)

        self.v = torch.zeros(self.B, self.n_rate, device=self.device)
        self.adapt = torch.zeros(self.B, self.n_rate, device=self.device)
        self.delta_rate = torch.zeros(self.B, self.n_rate, device=self.device)
        self._dr = torch.zeros(self.B, self.n_rate, device=self.device)                   # metal substeps: current dr
        self._cuda_dr = self._dr   # scratch recomputed at each CUDA frame, including after restore
        self.r0 = None
        self.last = {}
        self.diagnostics = True
        self.contrast = torch.zeros(self.B, n_col * 5, device=self.device)
        tau_map = DEFAULT_TAU_BY_TYPE if self.p.tau_by_type is None else self.p.tau_by_type
        tau = np.array([tau_map.get(t, self.p.tau_ms) for t in types[self.rate_idx]], dtype=np.float32)
        self._a = torch.from_numpy(np.exp(-self.p.dt_ms / tau)).to(self.device)             # (n_rate,)
        bl_map = DEFAULT_BASELINE_BY_TYPE if self.p.baseline_by_type is None else self.p.baseline_by_type
        self.b_vec = torch.from_numpy(np.array([bl_map.get(t, self.p.baseline) for t in types[self.rate_idx]], dtype=np.float32)).to(self.device)
        self._a_ad = float(np.exp(-self.p.dt_ms / self.p.adapt_tau_ms))

    # ------------------------------------------------------------------ per-stream hooks (opt-in; OpticParams docstring)
    @staticmethod
    def _type_mask(regex: str, type_list) -> np.ndarray:
        """Boolean mask over cells whose type full-matches `regex` (re.fullmatch; pair_gain's rule is re.match)."""
        import re
        pat = re.compile(str(regex))
        ok = {t: bool(pat.fullmatch(t)) for t in np.unique(type_list)}
        return np.array([ok[t] for t in type_list], dtype=bool)

    @staticmethod
    def _csr_select(C: sp.csr_matrix, keep: np.ndarray) -> sp.csr_matrix:
        """The entries of C (CSR order) flagged by `keep`, in their original order: a sparse product over the result sums
        the surviving terms of each row in the same sequence as over C, so an unselected entry changes nothing (bit for bit)."""
        C = C.tocsr(); rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        indptr = np.concatenate([[0], np.cumsum(np.bincount(rows[keep], minlength=C.shape[0]))])
        return sp.csr_matrix((C.data[keep], C.indices[keep], indptr), shape=C.shape)

    def _hold_feedback(self, M_rs, spk_types, rate_types):
        """fb_hold: the spiking (pre) -> rate (post) entries of the matched blocks removed from W_rs (the other entries keep
        their order, so their products are unchanged bit for bit)."""
        C = M_rs.tocsr(); rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr)); keep = np.ones(C.nnz, bool); held = []
        for pre_re, post_re in self.p.fb_hold:
            sel = self._type_mask(pre_re, spk_types)[C.indices] & self._type_mask(post_re, rate_types)[rows]
            held.append({"pre": pre_re, "post": post_re, "entries": int(sel.sum()), "syn_eq": float(np.abs(C.data[sel]).sum())})
            keep &= ~sel
        self.hook_info_fb = {"held": held, "entries_before": int(C.nnz), "entries_after": int(keep.sum())}
        return self._csr_select(C, keep)

    def _build_streams(self, M_rr, M_sr, rt, st, cuda_sparse):
        """Split M_rr / M_sr into the hook blocks and the rest, build the spatial operator and the adaptation constants."""
        p = self.p
        rect = list(p.stream_rectify or []); adapt = list(p.stream_adapt or []); supp = list(p.spatial_suppress or [])
        for k, (pre_re, post_re, mode) in enumerate(rect):
            if mode not in ("pos", "neg", "abs"):
                raise ValueError(f"stream_rectify[{k}]: mode must be pos, neg or abs, got {mode!r}")
        for k, (pre_re, post_re, tau, gain) in enumerate(adapt):
            if not (np.isfinite(tau) and tau > 0):
                raise ValueError(f"stream_adapt[{k}]: tau_ms must be positive and finite")
        # per presynaptic rate cell: which spatial_suppress entry applies (first match wins)
        supp_id = -np.ones(self.n_rate, np.int64)
        for k, (pre_re, kk, radius) in enumerate(supp):
            m = self._type_mask(pre_re, rt) & (supp_id < 0); supp_id[m] = k
        self._a_stream = [float(np.exp(-p.dt_ms / float(tau))) for (_, _, tau, _) in adapt]
        self._g_stream = [float(gain) for (_, _, _, gain) in adapt]

        def split(M, post_types, tag):
            # entries in the CSR order of M, so that every sub-matrix keeps M's per-row summation order (bit-identity)
            C = M.tocsr(); row = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr)); col = C.indices
            rect_id = -np.ones(C.nnz, np.int64); adapt_id = -np.ones(C.nnz, np.int64)
            for k, (pre_re, post_re, _) in enumerate(rect):
                sel = self._type_mask(pre_re, rt)[col] & self._type_mask(post_re, post_types)[row] & (rect_id < 0); rect_id[sel] = k
            for k, (pre_re, post_re, _, _) in enumerate(adapt):
                sel = self._type_mask(pre_re, rt)[col] & self._type_mask(post_re, post_types)[row] & (adapt_id < 0); adapt_id[sel] = k
            active = (rect_id >= 0) | (adapt_id >= 0) | (supp_id[col] >= 0)
            blocks = {}
            for key in sorted({(int(r), int(a)) for r, a in zip(rect_id[active], adapt_id[active])}):
                sel = active & (rect_id == key[0]) & (adapt_id == key[1])
                blocks[key] = {"M": self._csr_select(C, sel), "entries": int(sel.sum()), "syn_eq": float(np.abs(C.data[sel]).sum()),
                               "n_pre": int(len(np.unique(col[sel]))), "n_post": int(len(np.unique(row[sel])))}
            return blocks, self._csr_select(C, ~active), int(active.sum())

        b_rr, rest_rr, n_rr = split(M_rr, rt, "rr"); b_sr, rest_sr, n_sr = split(M_sr, st, "sr")
        self.W_rr_rest = _csr(rest_rr, self.device, self.metal, cuda_sparse); self.W_sr_rest = _csr(rest_sr, self.device, self.metal)
        info_streams = []
        for key in sorted(set(b_rr) | set(b_sr)):
            r_id, a_id = key
            s = {"rect_id": r_id, "adapt_id": a_id, "mode": rect[r_id][2] if r_id >= 0 else None,
                 "W_rr": _csr(b_rr[key]["M"], self.device, self.metal, cuda_sparse) if key in b_rr else None,
                 "W_sr": _csr(b_sr[key]["M"], self.device, self.metal) if key in b_sr else None}
            self.streams.append(s)
            info_streams.append({"rect": list(rect[r_id]) if r_id >= 0 else None, "adapt": list(adapt[a_id]) if a_id >= 0 else None,
                                 "rate_to_rate": {k: v for k, v in b_rr[key].items() if k != "M"} if key in b_rr else None,
                                 "rate_to_spiking": {k: v for k, v in b_sr[key].items() if k != "M"} if key in b_sr else None})
        self.hook_info.update({"streams": info_streams, "entries_rr_matched": n_rr, "entries_rr_rest": int(rest_rr.nnz),
                               "entries_sr_matched": n_sr, "entries_sr_rest": int(rest_sr.nnz), "torch_substep": not (self.cuda or self.metal)})
        # spatial suppression: G[j, j'] = k / n_j over the cells j' of j's type whose column is within radius_deg of j's
        # column (j included); u = dr - G dr.  Columns from trace.column_of_cells (hex annotation + propagation).
        if supp:
            from .interp.trace import column_of_cells
            col_all, n_ann = column_of_cells(self.c, self.r, self.rate_idx)
            col = col_all[self.rate_idx]
            cd = np.asarray(self.r.col_dir, dtype=np.float64)
            cosang = np.clip(cd @ cd.T, -1.0, 1.0)                                       # (n_col, n_col)
            rows, cols, vals = [], [], []; info_s = []
            for k, (pre_re, kk, radius) in enumerate(supp):
                A = sp.csr_matrix(cosang >= np.cos(np.radians(float(radius))))            # column adjacency, diagonal included
                cells = np.flatnonzero(supp_id == k); with_col = cells[col[cells] >= 0]
                n_nb = []
                for t in np.unique(rt[with_col]):
                    ct = with_col[rt[with_col] == t]
                    E = sp.csr_matrix((np.ones(len(ct)), (np.arange(len(ct)), col[ct])), shape=(len(ct), self.r.n_columns))
                    Mt = (E @ A @ E.T).tocoo()                                            # (cells of t) x (cells of t): within radius
                    cnt = np.asarray(Mt.sum(1)).ravel()
                    rows.append(ct[Mt.row]); cols.append(ct[Mt.col]); vals.append(float(kk) / cnt[Mt.row]); n_nb.append(cnt)
                n_nb = np.concatenate(n_nb) if n_nb else np.zeros(0)
                info_s.append({"pre": pre_re, "k": float(kk), "radius_deg": float(radius), "cells": int(len(cells)), "cells_with_column": int(len(with_col)),
                               "neighbours_mean": float(n_nb.mean()) if len(n_nb) else None, "neighbours_min": int(n_nb.min()) if len(n_nb) else None,
                               "neighbours_max": int(n_nb.max()) if len(n_nb) else None})
            G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n_rate, self.n_rate)) if rows else \
                sp.csr_matrix((self.n_rate, self.n_rate))
            self.G_supp = _csr(G, self.device, self.metal)
            self.hook_info["spatial"] = {"entries": info_s, "rate_cells_annotated": int(n_ann), "G_nnz": int(G.nnz)}

    def _stream_signals(self, dr: torch.Tensor, update: bool = False) -> list:
        """x per stream block from the presynaptic deviations dr (B, n_rate): u = dr - G dr, y = u - gain A, x = f(y).
        `update` advances every adaptation state A_k towards u (once per substep)."""
        u = dr if self.G_supp is None else dr - _mv(self.G_supp, dr)
        xs = []
        for s in self.streams:
            y = u if s["adapt_id"] < 0 else u - self._g_stream[s["adapt_id"]] * self.stream_adapt_state[s["adapt_id"]]
            m = s["mode"]
            x = y if m is None else (y.clamp(min=0.0) if m == "pos" else ((-y).clamp(min=0.0) if m == "neg" else y.abs()))
            xs.append(x)
        if update and self._a_stream:
            if self.surrogate_grad:
                # functional update: the differentiable path must not write in place into a tensor autograd holds
                self.stream_adapt_state = torch.stack([u + (self.stream_adapt_state[k] - u) * a
                                                       for k, a in enumerate(self._a_stream)])
            else:
                for k in range(len(self._a_stream)):
                    Ak = self.stream_adapt_state[k]
                    torch.add(u, (Ak - u) * self._a_stream[k], out=Ak)
        return xs

    def _recurrent(self, dr: torch.Tensor, update: bool = False) -> torch.Tensor:
        """gain_rr x the recurrent input: the plain product without hooks (the shipped expression, bit-identical), else
        the rest block on dr plus every stream block on its transformed x."""
        if not self._hooks:
            return self.p.gain_rr * _mv(self.W_rr, dr)
        acc = _mv(self.W_rr_rest, dr)
        for s, x in zip(self.streams, self._stream_signals(dr, update)):
            if s["W_rr"] is not None:
                acc = acc + _mv(s["W_rr"], x)
        return self.p.gain_rr * acc

    def _output(self, dr: torch.Tensor) -> torch.Tensor:
        """W_sr @ dr (the drive before gain_out / clip), with the stream blocks on their transformed x under the hooks."""
        if not self._hooks:
            return _mv(self.W_sr, dr)
        acc = _mv(self.W_sr_rest, dr)
        for s, x in zip(self.streams, self._stream_signals(dr, False)):
            if s["W_sr"] is not None:
                acc = acc + _mv(s["W_sr"], x)
        return acc

    # ------------------------------------------------------------------ photoreceptors
    @torch.no_grad()
    def photoreceptor_activity(self, col_radiance: torch.Tensor, dt_ms: float, *, intensity=None) -> torch.Tensor:
        """col_radiance (B, n_col, 4) -> photoreceptor contrast activity (B, n_pr)."""
        p = self.p
        rad = col_radiance.to(self.device, torch.float32)
        if rad.dim() == 2:
            rad = rad[None]
        I_pr = (rad[:, self.pr_column_t, :] * self.sens[None]).sum(-1) if intensity is None else intensity
        I = _mv(self.avg, I_pr)                                                               # (B, n_col*5)
        fresh = self._fresh[:, None]
        torch.where(fresh, I, self.I_lp, out=self.I_lp)
        torch.where(fresh, I, self.I_mean, out=self.I_mean)
        self._fresh[:] = False
        a_lp = float(np.exp(-dt_ms / p.tau_lp_ms)); a_ad = float(np.exp(-dt_ms / p.tau_adapt_ms))
        torch.add(a_lp * self.I_lp, (1 - a_lp) * I, out=self.I_lp)
        torch.add(a_ad * self.I_mean, (1 - a_ad) * self.I_lp, out=self.I_mean)
        contrast = ((self.I_lp - self.I_mean) / (self.I_mean + p.eps)).clamp(-1.0, p.contrast_clip)
        torch.where(self.has_t[None], contrast, torch.zeros_like(contrast), out=self.contrast)
        contrast = self.contrast
        if self.diagnostics:
            self.last["contrast"] = self.contrast.view(self.B, -1, 5).cpu().numpy()                  # (B, n_col, 5)
        return contrast[:, self.pr_cell_t]                                                   # (B, n_pr)

    # ------------------------------------------------------------------ rate dynamics
    def rates(self) -> torch.Tensor:
        if self.slow is not None and self.slow.mode == "threshold":
            return (self.v + self.b_vec[None] + self.g_slow.clamp(max=0.9 * self._slow_norm)).clamp(0.0, 1.0)
        return (self.v + self.b_vec[None]).clamp(0.0, 1.0)

    @torch.no_grad()
    def _substep(self, pr_input: torch.Tensor, spk_input: torch.Tensor) -> None:
        p = self.p
        dr = self.rates() - self.b_vec[None]                                                 # (B, n_rate)
        if self.slow is None:
            # Keep the addition order: combining the held inputs would change rounding.
            inp = self._recurrent(dr, update=True) + pr_input - p.adapt_gain * self.adapt
            inp = inp + spk_input
        else:
            # slow tone per class: relax towards scale x (held spiking part + recurrent part)
            for k in range(len(self.slow_classes)):
                target = self._slow_in_s[k]
                if self.W_slow_rr[k] is not None:
                    target = target + p.gain_rr * _mv(self.W_slow_rr[k], dr)
                gk = self.g_slow_cls[k]
                torch.add(target * self._slow_scale[k], (gk - target * self._slow_scale[k]) * self._a_slow[k], out=gk)
            torch.sum(self.g_slow_cls, dim=0, out=self.g_slow)
            syn = self._recurrent(dr, update=True) + pr_input + spk_input
            if self.slow.mode == "additive":
                inp = syn - p.adapt_gain * self.adapt + self.g_slow
            elif self.slow.mode == "gain":
                f = torch.clamp(1.0 + self.g_slow / self._slow_norm, self.slow.clip[0], self.slow.clip[1])
                inp = syn * f - p.adapt_gain * self.adapt
            else:                                                     # "threshold": the shift acts in rates()
                inp = syn - p.adapt_gain * self.adapt
        torch.add(inp, (self.v - inp) * self._a[None], out=self.v)
        torch.add(dr, (self.adapt - dr) * self._a_ad, out=self.adapt)

    def relax(self, ms: float = 0.0) -> None:
        """Rest state: by construction every unit sits at the operating point (kept for API symmetry)."""
        self.v.zero_()
        self.r0 = self.b_vec[None].clone()

    def step_frame(self, col_radiance: torch.Tensor, spk_rate_hz: torch.Tensor, frame_ms: float, *, intensity=None) -> torch.Tensor:
        if self.surrogate_grad:
            return self._step_frame_grad(col_radiance, spk_rate_hz, frame_ms, intensity=intensity)
        return self._step_frame_inference(col_radiance, spk_rate_hz, frame_ms, intensity=intensity)

    @torch.no_grad()
    def _step_frame_inference(self, col_radiance, spk_rate_hz, frame_ms, *, intensity=None):
        """Advance the optic lobe by one frame; returns the drive (B, N) in mV for the spiking brain.
        spk_rate_hz: the brain's rate tensor, (B, N) (or (N,))."""
        if self.r0 is None:
            self.relax()
        a_pr = self.photoreceptor_activity(col_radiance, frame_ms, intensity=intensity)
        if spk_rate_hz.dim() == 1:
            spk_rate_hz = spk_rate_hz[None]
        s = (spk_rate_hz[:, self.spk_idx_t] / 100.0).clamp(0, 3)
        total = self._pending_ms + frame_ms
        steps = int(np.floor((total + 1e-9) / self.p.dt_ms))
        self._pending_ms = max(0.0, total - steps * self.p.dt_ms)
        if steps:
            # Photoreceptor activity and spiking feedback are held for this frame.
            # Only the recurrent optic product changes between substeps.
            pr_input = self.p.gain_in * _mv(self.W_rp, a_pr)
            spk_input = self.p.gain_fb * _mv(self.W_rs, s)
            if self.slow is not None:                                # the spiking part of the slow input, held too
                for k in range(len(self.slow_classes)):
                    if self.W_slow_rs[k] is not None:
                        self._slow_in_s[k] = self.p.gain_fb * _mv(self.W_slow_rs[k], s)
            if self.cuda:
                cuda.optic_dr(self.v, self.b_vec, self._cuda_dr)
                pr_input, spk_input = pr_input.contiguous(), spk_input.contiguous()
                for _ in range(steps):
                    cuda.optic_update(self, _mv(self.W_rr, self._cuda_dr), pr_input, spk_input)
            elif self.metal:
                p = self.p
                metal.optic_dr(self.v, self.b_vec, self._dr)
                for _ in range(steps):
                    metal.optic_substep(self.v, self.adapt, self._dr, self.W_rr.matvec(self._dr), pr_input.contiguous(),
                                        spk_input.contiguous(), self._a, self.b_vec, p.gain_rr, p.adapt_gain, self._a_ad)
            else:
                for _ in range(steps):
                    self._substep(pr_input, spk_input)
        torch.sub(self.rates(), self.r0, out=self.delta_rate)
        dr = self.delta_rate
        drive = torch.zeros(self.B, self.c.n, device=self.device)
        drive[:, self.spk_idx_t] = (self.p.gain_out_mv * self._output(dr)).clamp(-self.p.drive_clip_mv, self.p.drive_clip_mv)
        self.last["dr"] = dr
        return drive

    def _step_frame_grad(self, col_radiance, spk_rate_hz, frame_ms, *, intensity=None):
        """Functional Torch optics for a short differentiable simulation window.

        The substep is the same model as `_step_frame_inference`: the recurrent input goes through
        `self._recurrent(dr, update=True)` and the output through `self._output(...)`, so the opt-in per-stream
        hooks (OpticParams.stream_rectify / stream_adapt / spatial_suppress) act here exactly as they do in
        inference, and with the hooks off the two paths evaluate the identical expressions (bit-identical;
        tests/test_surrogate.py::test_grad_and_inference_optic_agree_bitwise). Only the *storage* differs:
        every state tensor is rebuilt functionally instead of being written in place."""
        p = self.p
        if self.r0 is None:
            self.relax()
        rad = col_radiance.to(self.device, torch.float32)
        if rad.ndim == 2:
            rad = rad[None]
        I_pr = (rad[:, self.pr_column_t, :] * self.sens[None]).sum(-1) if intensity is None else intensity
        I = _mv(self.avg, I_pr)
        lp = torch.where(self._fresh[:, None], I, self.I_lp)
        mean = torch.where(self._fresh[:, None], I, self.I_mean)
        self._fresh = torch.zeros_like(self._fresh)
        a_lp = float(np.exp(-frame_ms / p.tau_lp_ms)); a_ad = float(np.exp(-frame_ms / p.tau_adapt_ms))
        self.I_lp = a_lp * lp + (1. - a_lp) * I
        self.I_mean = a_ad * mean + (1. - a_ad) * self.I_lp
        contrast = ((self.I_lp - self.I_mean) / (self.I_mean + p.eps)).clamp(-1., p.contrast_clip)
        self.contrast = torch.where(self.has_t[None], contrast, 0.)
        a_pr = self.contrast[:, self.pr_cell_t]
        if spk_rate_hz.ndim == 1:
            spk_rate_hz = spk_rate_hz[None]
        s = (spk_rate_hz[:, self.spk_idx_t] / 100.).clamp(0, 3)
        total = self._pending_ms + frame_ms
        steps = int(np.floor((total + 1e-9) / p.dt_ms))
        self._pending_ms = max(0., total - steps * p.dt_ms)
        if steps:
            pr_input = p.gain_in * _mv(self.W_rp, a_pr)
            spk_input = p.gain_fb * _mv(self.W_rs, s)
            if self.slow is not None:
                self._slow_in_s = torch.stack([p.gain_fb * _mv(W, s) if W is not None else torch.zeros_like(self.v)
                                              for W in self.W_slow_rs])
            for _ in range(steps):
                dr = self.rates() - self.b_vec[None]
                if self.slow is None:
                    # Keep the addition order of _substep: combining the held inputs would change rounding.
                    inp = self._recurrent(dr, update=True) + pr_input - p.adapt_gain * self.adapt
                    inp = inp + spk_input
                else:
                    slow = []
                    for k, W in enumerate(self.W_slow_rr):
                        target = self._slow_in_s[k]
                        if W is not None:
                            target = target + p.gain_rr * _mv(W, dr)
                        target = target * self._slow_scale[k]
                        slow.append(target + (self.g_slow_cls[k] - target) * self._a_slow[k])
                    self.g_slow_cls = torch.stack(slow)
                    self.g_slow = self.g_slow_cls.sum(0)
                    syn = self._recurrent(dr, update=True) + pr_input + spk_input
                    if self.slow.mode == "additive":
                        inp = syn - p.adapt_gain * self.adapt + self.g_slow
                    elif self.slow.mode == "gain":
                        factor = (1. + self.g_slow / self._slow_norm).clamp(*self.slow.clip)
                        inp = syn * factor - p.adapt_gain * self.adapt
                    else:
                        inp = syn - p.adapt_gain * self.adapt
                self.v = inp + (self.v - inp) * self._a[None]
                self.adapt = dr + (self.adapt - dr) * self._a_ad
        self.delta_rate = self.rates() - self.r0
        drive = torch.zeros(self.B, self.c.n, device=self.device)
        drive[:, self.spk_idx_t] = (p.gain_out_mv * self._output(self.delta_rate)).clamp(-p.drive_clip_mv, p.drive_clip_mv)
        self.last["dr"] = self.delta_rate
        if self.diagnostics:
            self.last["contrast"] = self.contrast.detach().view(self.B, -1, 5).cpu().numpy()
        return drive

    def detach_state(self):
        for name in ("v", "adapt", "I_lp", "I_mean", "_fresh", "contrast", "delta_rate", "g_slow", "g_slow_cls",
                     "_slow_in_s", "stream_adapt_state"):
            setattr(self, name, getattr(self, name).detach().clone())

    def reset(self, rows=None) -> None:
        if self.surrogate_grad:
            self.detach_state()
        if rows is None:
            self._fresh[:] = True; self.v.zero_(); self.adapt.zero_()
            self.g_slow_cls.zero_(); self.g_slow.zero_(); self._slow_in_s.zero_(); self.stream_adapt_state.zero_()
            self._pending_ms = 0.0
        else:
            sel = torch.as_tensor(np.asarray(rows), device=self.device, dtype=torch.long)
            self._fresh[sel] = True; self.v[sel] = 0.0; self.adapt[sel] = 0.0
            self.g_slow_cls[:, sel] = 0.0; self.g_slow[sel] = 0.0; self._slow_in_s[:, sel] = 0.0; self.stream_adapt_state[:, sel] = 0.0

    # ------------------------------------------------------------------ inspection
    def delta_rate_by_type(self, top: int = 15, row: int = 0):
        import pandas as pd
        if "dr" not in self.last:
            return None
        dr = self.last["dr"][row].cpu().numpy()
        t = self.c.neurons.type.fillna("").to_numpy()[self.rate_idx]
        df = pd.DataFrame({"type": t, "dr": dr, "absdr": np.abs(dr)})
        return df.groupby("type").agg(mean_dr=("dr", "mean"), mean_abs=("absdr", "mean"), n=("dr", "size")).sort_values("mean_abs", ascending=False).head(top)
