"""Graded (rate-model) optic lobe coupled to the spiking rest of the CNS.

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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from .connectome import Connectome, PHOTORECEPTOR_TYPES
from .retina import Retina

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
    gain_out_mv: float = 100.0    # optic lobe delta-rate -> injected current in spiking targets
    out_norm: str = "l1"          # normalisation of the optic-lobe -> spiking weights ("l1" fractions, "l2")
    drive_clip_mv: float = 35.0
    # photoreceptor stage
    tau_lp_ms: float = 10.0
    tau_adapt_ms: float = 300.0
    contrast_clip: float = 2.0
    eps: float = 0.02


DEFAULT_TAU_BY_TYPE = {"Mi4": 60.0, "Mi9": 60.0, "CT1": 80.0, "Tm9": 60.0, "L3": 40.0, "Mi1": 8.0, "Tm3": 8.0,
                       "Tm1": 8.0, "Tm2": 8.0, "Tm4": 8.0, "L1": 6.0, "L2": 6.0, "T4a": 10.0, "T4b": 10.0, "T4c": 10.0, "T4d": 10.0,
                       "T5a": 10.0, "T5b": 10.0, "T5c": 10.0, "T5d": 10.0}


def _csr(D: sp.spmatrix, device) -> torch.Tensor:
    D = D.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(torch.from_numpy(D.indptr.astype(np.int64)), torch.from_numpy(D.indices.astype(np.int64)),
                                   torch.from_numpy(D.data), size=D.shape).to(device)


class OpticLobe:
    def __init__(self, c: Connectome, retina: Retina, params: OpticParams | None = None, device=None):
        self.c, self.r, self.p = c, retina, params or OpticParams()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        nrn = c.neurons
        types = nrn.type.fillna("").to_numpy()
        is_pr = nrn.type.isin(PHOTORECEPTOR_TYPES).to_numpy()
        self.rate_idx = np.flatnonzero((nrn.superclass == "ol_intrinsic").to_numpy() & ~is_pr)
        self.pr_idx = retina.pr_index
        is_rate = np.zeros(c.n, bool); is_rate[self.rate_idx] = True
        self.spk_idx = np.flatnonzero(~is_rate & ~is_pr)
        self.n_rate, self.n_pr, self.n_spk = len(self.rate_idx), len(self.pr_idx), len(self.spk_idx)

        W = c.W.tocsr()                                   # (post, pre) signed counts
        tot = np.asarray(abs(W).sum(axis=1)).ravel()
        Wn = (sp.diags(1.0 / np.maximum(tot, 1.0)) @ W).tocsr()      # L1: fractions of total input
        if self.p.norm == "l2":
            l2 = np.sqrt(np.asarray(W.multiply(W).sum(axis=1)).ravel())
            Wn_ol = (sp.diags(1.0 / np.maximum(l2, 1.0)) @ W).tocsr()
        else:
            Wn_ol = Wn
        self.W_rr = _csr(Wn_ol[self.rate_idx][:, self.rate_idx], self.device)
        self.W_rp = _csr(Wn_ol[self.rate_idx][:, self.pr_idx], self.device)
        self.W_rs = _csr(Wn_ol[self.rate_idx][:, self.spk_idx], self.device)
        self.W_sr = _csr((Wn_ol if self.p.out_norm == "l2" else Wn)[self.spk_idx][:, self.rate_idx], self.device)
        self.rate_idx_t = torch.as_tensor(self.rate_idx, device=self.device)
        self.spk_idx_t = torch.as_tensor(self.spk_idx, device=self.device)

        # photoreceptor stage: per (column, family) intensity, low-pass + contrast adaptation
        self.pr_family = np.array([FAMILY_OF_TYPE[t] for t in types[self.pr_idx]])
        n_col = retina.n_columns
        self.has = np.zeros((n_col, 5), bool)
        np.add.at(self.has, (retina.pr_column, self.pr_family), True)
        self.I_lp = np.zeros((n_col, 5), np.float32)
        self.I_mean = np.zeros((n_col, 5), np.float32)
        self._fresh = True

        self.v = torch.zeros(self.n_rate, device=self.device)
        self.adapt = torch.zeros(self.n_rate, device=self.device)
        self.r0 = None
        self.last = {}
        tau_map = DEFAULT_TAU_BY_TYPE if self.p.tau_by_type is None else self.p.tau_by_type
        tau = np.array([tau_map.get(t, self.p.tau_ms) for t in types[self.rate_idx]], dtype=np.float32)
        self._a = torch.from_numpy(np.exp(-self.p.dt_ms / tau)).to(self.device)
        self._a_ad = float(np.exp(-self.p.dt_ms / self.p.adapt_tau_ms))

    # ------------------------------------------------------------------ photoreceptors
    def photoreceptor_activity(self, col_radiance: torch.Tensor, dt_ms: float) -> torch.Tensor:
        p = self.p
        rad = col_radiance.detach().cpu().numpy().astype(np.float32)
        I_pr = np.einsum("pc,pc->p", rad[self.r.pr_column], self.r.pr_sens)
        I = np.zeros_like(self.I_lp); cnt = np.zeros_like(self.I_lp)
        np.add.at(I, (self.r.pr_column, self.pr_family), I_pr)
        np.add.at(cnt, (self.r.pr_column, self.pr_family), 1.0)
        I[self.has] /= cnt[self.has]
        if self._fresh:
            self.I_lp[:] = I; self.I_mean[:] = I; self._fresh = False
        a_lp = np.exp(-dt_ms / p.tau_lp_ms); a_ad = np.exp(-dt_ms / p.tau_adapt_ms)
        self.I_lp = a_lp * self.I_lp + (1 - a_lp) * I
        self.I_mean = a_ad * self.I_mean + (1 - a_ad) * self.I_lp
        contrast = np.clip((self.I_lp - self.I_mean) / (self.I_mean + p.eps), -1.0, p.contrast_clip)
        contrast[~self.has] = 0.0
        self.last["contrast"] = contrast
        a = contrast[self.r.pr_column, self.pr_family]                      # per photoreceptor
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)

    # ------------------------------------------------------------------ rate dynamics
    def rates(self) -> torch.Tensor:
        return (self.v + self.p.baseline).clamp(0.0, 1.0)

    @torch.no_grad()
    def _substep(self, a_pr: torch.Tensor, s_spk: torch.Tensor | None) -> None:
        p = self.p
        dr = self.rates() - p.baseline
        inp = p.gain_rr * (self.W_rr @ dr) + p.gain_in * (self.W_rp @ a_pr) - p.adapt_gain * self.adapt
        if s_spk is not None:
            inp = inp + p.gain_fb * (self.W_rs @ s_spk)
        self.v = inp + (self.v - inp) * self._a
        self.adapt = dr + (self.adapt - dr) * self._a_ad

    def relax(self, ms: float = 0.0) -> None:
        """Rest state: by construction every unit sits at the operating point (kept for API symmetry)."""
        self.v.zero_()
        self.r0 = torch.full((self.n_rate,), self.p.baseline, device=self.device)

    @torch.no_grad()
    def step_frame(self, col_radiance: torch.Tensor, spk_rate_hz: torch.Tensor, frame_ms: float) -> torch.Tensor:
        """Advance the optic lobe by one frame and return the drive vector (N,) mV for the spiking brain."""
        if self.r0 is None:
            self.relax()
        a_pr = self.photoreceptor_activity(col_radiance, frame_ms)
        s = (spk_rate_hz[self.spk_idx_t] / 100.0).clamp(0, 3)
        for _ in range(max(1, int(round(frame_ms / self.p.dt_ms)))):
            self._substep(a_pr, s)
        dr = self.rates() - self.r0
        drive = torch.zeros(self.c.n, device=self.device)
        drive[self.spk_idx_t] = (self.p.gain_out_mv * (self.W_sr @ dr)).clamp(-self.p.drive_clip_mv, self.p.drive_clip_mv)
        self.last["dr"] = dr
        return drive

    def reset(self) -> None:
        self._fresh = True
        self.v.zero_()
        self.adapt.zero_()

    # ------------------------------------------------------------------ inspection
    def delta_rate_by_type(self, top: int = 15):
        import pandas as pd
        if "dr" not in self.last:
            return None
        dr = self.last["dr"].cpu().numpy()
        t = self.c.neurons.type.fillna("").to_numpy()[self.rate_idx]
        df = pd.DataFrame({"type": t, "dr": dr, "absdr": np.abs(dr)})
        return df.groupby("type").agg(mean_dr=("dr", "mean"), mean_abs=("absdr", "mean"), n=("dr", "size")).sort_values("mean_abs", ascending=False).head(top)
