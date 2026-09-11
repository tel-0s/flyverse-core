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

from .connectome import Connectome, PHOTORECEPTOR_TYPES
from .retina import Retina
from .device import resolve, sparse_matrix

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
                     # T4/T5 outputs x4: rectified, strongly inhibited DS units respond weakly to natural
                     # scenes; this restores drive to LPi / HS / VS / LPLC and the descending neurons
                     (r"^T[45][abcd]$", r".*", 2.0),
                     # the loom detectors LC4 / LPLC2 keep x1 optic-lobe drive: in the full sensory context
                     # (smell + wind on) x1.25 already gives 3-4 spontaneous GF escapes per 5 s of walking;
                     # the loom margin comes from the x3 LC4/LPLC2 -> GF synapses in brain.py instead
                     (r".*", r"^(LC4|LPLC2)$", 1.0)]


def _csr(D: sp.spmatrix, device) -> torch.Tensor:
    return sparse_matrix(D.astype(np.float32), device)


class OpticLobe:
    def __init__(self, c: Connectome, retina: Retina, params: OpticParams | None = None, device=None, batch: int = 1):
        self.c, self.r, self.p = c, retina, params or OpticParams()
        self.B = int(batch)
        self.device = resolve(device)
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
        self.W_rr = _csr(apply_pair_gain(Wn_ol[self.rate_idx][:, self.rate_idx], rt, rt), self.device)
        self.W_rp = _csr(Wn_ol[self.rate_idx][:, self.pr_idx], self.device)
        self.W_rs = _csr(Wn_ol[self.rate_idx][:, self.spk_idx], self.device)
        self.W_sr = _csr(apply_pair_gain((Wn_ol if self.p.out_norm == "l2" else Wn)[self.spk_idx][:, self.rate_idx], rt, types[self.spk_idx]), self.device)
        self.rate_idx_t = torch.as_tensor(self.rate_idx, device=self.device)
        self.spk_idx_t = torch.as_tensor(self.spk_idx, device=self.device)

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
        self.avg = _csr(A, self.device)
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
    def photoreceptor_activity(self, col_radiance: torch.Tensor, dt_ms: float) -> torch.Tensor:
        """col_radiance (B, n_col, 4) -> photoreceptor contrast activity (B, n_pr)."""
        p = self.p
        rad = col_radiance.to(self.device, torch.float32)
        if rad.dim() == 2:
            rad = rad[None]
        I_pr = (rad[:, self.pr_column_t, :] * self.sens[None]).sum(-1)                     # (B, n_pr)
        I = (self.avg @ I_pr.T.contiguous()).T                                                            # (B, n_col*5)
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
        return (self.v + self.b_vec[None]).clamp(0.0, 1.0)

    @torch.no_grad()
    def _substep(self, pr_input: torch.Tensor, spk_input: torch.Tensor) -> None:
        p = self.p
        dr = self.rates() - self.b_vec[None]                                                 # (B, n_rate)
        # Keep the addition order: combining the held inputs would change rounding.
        inp = p.gain_rr * (self.W_rr @ dr.T.contiguous()).T + pr_input - p.adapt_gain * self.adapt
        inp = inp + spk_input
        torch.add(inp, (self.v - inp) * self._a[None], out=self.v)
        torch.add(dr, (self.adapt - dr) * self._a_ad, out=self.adapt)

    def relax(self, ms: float = 0.0) -> None:
        """Rest state: by construction every unit sits at the operating point (kept for API symmetry)."""
        self.v.zero_()
        self.r0 = self.b_vec[None].clone()

    @torch.no_grad()
    def step_frame(self, col_radiance: torch.Tensor, spk_rate_hz: torch.Tensor, frame_ms: float) -> torch.Tensor:
        """Advance the optic lobe by one frame; returns the drive (B, N) in mV for the spiking brain.
        spk_rate_hz: the brain's rate tensor, (B, N) (or (N,))."""
        if self.r0 is None:
            self.relax()
        a_pr = self.photoreceptor_activity(col_radiance, frame_ms)
        if spk_rate_hz.dim() == 1:
            spk_rate_hz = spk_rate_hz[None]
        s = (spk_rate_hz[:, self.spk_idx_t] / 100.0).clamp(0, 3)
        total = self._pending_ms + frame_ms
        steps = int(np.floor((total + 1e-9) / self.p.dt_ms))
        self._pending_ms = max(0.0, total - steps * self.p.dt_ms)
        if steps:
            # Photoreceptor activity and spiking feedback are held for this frame.
            # Only the recurrent optic product changes between substeps.
            pr_input = self.p.gain_in * (self.W_rp @ a_pr.T.contiguous()).T
            spk_input = self.p.gain_fb * (self.W_rs @ s.T.contiguous()).T
            for _ in range(steps):
                self._substep(pr_input, spk_input)
        torch.sub(self.rates(), self.r0, out=self.delta_rate)
        dr = self.delta_rate
        drive = torch.zeros(self.B, self.c.n, device=self.device)
        drive[:, self.spk_idx_t] = (self.p.gain_out_mv * (self.W_sr @ dr.T.contiguous()).T).clamp(-self.p.drive_clip_mv, self.p.drive_clip_mv)
        self.last["dr"] = dr
        return drive

    def reset(self, rows=None) -> None:
        if rows is None:
            self._fresh[:] = True; self.v.zero_(); self.adapt.zero_()
            self._pending_ms = 0.0
        else:
            sel = torch.as_tensor(np.asarray(rows), device=self.device, dtype=torch.long)
            self._fresh[sel] = True; self.v[sel] = 0.0; self.adapt[sel] = 0.0

    # ------------------------------------------------------------------ inspection
    def delta_rate_by_type(self, top: int = 15, row: int = 0):
        import pandas as pd
        if "dr" not in self.last:
            return None
        dr = self.last["dr"][row].cpu().numpy()
        t = self.c.neurons.type.fillna("").to_numpy()[self.rate_idx]
        df = pd.DataFrame({"type": t, "dr": dr, "absdr": np.abs(dr)})
        return df.groupby("type").agg(mean_dr=("dr", "mean"), mean_abs=("absdr", "mean"), n=("dr", "size")).sort_values("mean_abs", ascending=False).head(top)
