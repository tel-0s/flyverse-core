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


DEFAULT_TAU_BY_TYPE = {"Mi4": 150.0, "Mi9": 150.0, "CT1": 150.0, "Tm9": 150.0, "L3": 40.0, "Mi1": 8.0, "Tm3": 8.0,
                       "Tm1": 8.0, "Tm2": 8.0, "Tm4": 8.0, "L1": 6.0, "L2": 6.0, "T4a": 10.0, "T4b": 10.0, "T4c": 10.0, "T4d": 10.0,
                       "T5a": 10.0, "T5b": 10.0, "T5c": 10.0, "T5d": 10.0}
# T4/T5 as rectifying (ReLU) units with strong delayed inhibition: this is what makes them direction
# selective (DSI 0.16-0.26 with the correct preferred direction for all 8 subtypes; see NOTES).
T4T5 = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
DEFAULT_BASELINE_BY_TYPE = {t: 0.0 for t in T4T5}
DEFAULT_PAIR_GAIN = [(r"^(Mi4|Mi9|CT1|C3)$", r"^T4[abcd]$", 5.0), (r"^(Tm4|Tm9|CT1|TmY15)$", r"^T5[abcd]$", 5.0),
                     # LPi -> LPLC2: the lobula-plate inhibitory interneurons make LPLC2 expansion-selective in the animal;
                     # under the uniform synapse they were a tenth of its T4/T5 excitation, so the fly's own turning
                     # drove the giant fibre. x4 halves the walking GF and keeps the loom (NOTES, session 9).
                     (r"^LPi(34|43)$", r"^LPLC2$", 4.0),
                     # T4/T5 outputs x4: rectified, strongly inhibited DS units respond weakly to natural
                     # scenes; this restores drive to LPi / HS / VS / LPLC and the descending neurons
                     (r"^T[45][abcd]$", r".*", 2.0),
                     # the loom detectors LC4 / LPLC2 keep x1 optic-lobe drive: in the full sensory context
                     # (smell + wind on) x1.25 already gives 3-4 spontaneous GF escapes per 5 s of walking;
                     # the loom margin comes from the x3 LC4/LPLC2 -> GF synapses in brain.py instead
                     (r".*", r"^(LC4|LPLC2)$", 1.0)]


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
        term runs the Torch substep (a warning if kernels were requested)."""
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
        if self.slow is not None and (metal_kernels or cuda_kernels):
            import warnings
            warnings.warn("the optic lobe's slow receptor term is not in the native optic kernels; using the Torch substep")
        self.metal = metal.use(self.device, False if self.slow is not None else metal_kernels)
        self.cuda = cuda.use(self.device, False if self.slow is not None else cuda_kernels)
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

        rt = types[self.rate_idx]
        self.W_rr = _csr(apply_pair_gain(Wn_ol[self.rate_idx][:, self.rate_idx], rt, rt), self.device, self.metal, cuda_sparse)
        self.W_rp = _csr(Wn_ol[self.rate_idx][:, self.pr_idx], self.device, self.metal)
        self.W_rs = _csr(Wn_ol[self.rate_idx][:, self.spk_idx], self.device, self.metal)
        self.W_sr = _csr(apply_pair_gain((Wn_ol if self.p.out_norm == "l2" else Wn)[self.spk_idx][:, self.rate_idx], rt, types[self.spk_idx]), self.device, self.metal)
        self.rate_idx_t = torch.as_tensor(self.rate_idx, device=self.device)
        self.spk_idx_t = torch.as_tensor(self.spk_idx, device=self.device)

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
            inp = p.gain_rr * _mv(self.W_rr, dr) + pr_input - p.adapt_gain * self.adapt
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
            syn = p.gain_rr * _mv(self.W_rr, dr) + pr_input + spk_input
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
        drive[:, self.spk_idx_t] = (self.p.gain_out_mv * _mv(self.W_sr, dr)).clamp(-self.p.drive_clip_mv, self.p.drive_clip_mv)
        self.last["dr"] = dr
        return drive

    def _step_frame_grad(self, col_radiance, spk_rate_hz, frame_ms, *, intensity=None):
        """Functional Torch optics for a short differentiable simulation window."""
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
                    inp = p.gain_rr * _mv(self.W_rr, dr) + pr_input - p.adapt_gain * self.adapt
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
                    syn = p.gain_rr * _mv(self.W_rr, dr) + pr_input + spk_input
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
        drive[:, self.spk_idx_t] = (p.gain_out_mv * _mv(self.W_sr, self.delta_rate)).clamp(-p.drive_clip_mv, p.drive_clip_mv)
        self.last["dr"] = self.delta_rate
        if self.diagnostics:
            self.last["contrast"] = self.contrast.detach().view(self.B, -1, 5).cpu().numpy()
        return drive

    def detach_state(self):
        for name in ("v", "adapt", "I_lp", "I_mean", "_fresh", "contrast", "delta_rate", "g_slow", "g_slow_cls", "_slow_in_s"):
            setattr(self, name, getattr(self, name).detach().clone())

    def reset(self, rows=None) -> None:
        if self.surrogate_grad:
            self.detach_state()
        if rows is None:
            self._fresh[:] = True; self.v.zero_(); self.adapt.zero_()
            self.g_slow_cls.zero_(); self.g_slow.zero_(); self._slow_in_s.zero_()
            self._pending_ms = 0.0
        else:
            sel = torch.as_tensor(np.asarray(rows), device=self.device, dtype=torch.long)
            self._fresh[sel] = True; self.v[sel] = 0.0; self.adapt[sel] = 0.0
            self.g_slow_cls[:, sel] = 0.0; self.g_slow[sel] = 0.0; self._slow_in_s[:, sel] = 0.0

    # ------------------------------------------------------------------ inspection
    def delta_rate_by_type(self, top: int = 15, row: int = 0):
        import pandas as pd
        if "dr" not in self.last:
            return None
        dr = self.last["dr"][row].cpu().numpy()
        t = self.c.neurons.type.fillna("").to_numpy()[self.rate_idx]
        df = pd.DataFrame({"type": t, "dr": dr, "absdr": np.abs(dr)})
        return df.groupby("type").agg(mean_dr=("dr", "mean"), mean_abs=("absdr", "mean"), n=("dr", "size")).sort_values("mean_abs", ascending=False).head(top)
