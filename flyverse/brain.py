"""Whole-CNS leaky integrate-and-fire simulation of the MaleCNS connectome on the GPU (torch).

Model (Shiu et al. 2024, Nature 634:210; same constants as stonkfly/doomfly):
    dv/dt = (v_rest - v + g + I_ext) / tau_m          v_rest = v_reset = -52 mV, v_th = -45 mV
    dg/dt = -g / tau_syn                              tau_m = 20 ms, tau_syn = 5 ms
    presynaptic spike (after 1.8 ms delay): g_post += 0.275 mV * sign * synapse_count
    refractory period 2.2 ms.
Integration is exponential-Euler with a configurable dt (default 0.5 ms; Shiu/stonkfly use 0.1 ms).

I_ext is a per-neuron constant "drive" in mV (doomfly-style current injection: a drive of D mV above
rest makes a neuron fire at 1/(t_ref + tau_m*ln(D/(D-7))) Hz once D > 7 mV). You can also force Poisson
spikes on chosen neurons (Shiu-style optogenetic input) with `poisson_rate_hz`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .connectome import Connectome


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
    adapt_jump: float = 0.3  # mV
    adapt_tau: float = 200.0 # ms
    # short-term synaptic depression (Tsodyks-Markram style, per presynaptic neuron; not in Shiu et al.):
    # each spike uses a fraction `std_u` of the available resource x, which recovers with `std_tau`. At
    # rate R the steady resource is 1/(1 + u R tau): 20 Hz -> 0.45, 100 Hz -> 0.14, 300 Hz -> 0.05, which
    # kills the self-exciting cliques (FR1, DLMn, ...) that otherwise lock up at 300 Hz. 0 disables.
    std_u: float = 0.2
    std_tau: float = 300.0   # ms
    # per-postsynaptic-neuron synaptic gain (median total input synapses / own total)^alpha, clipped to
    # [0.5, 5]: small neurons (T4 has ~250 input synapses vs a median of ~1000) get larger unitary PSPs.
    # 0 = Shiu's uniform 0.275 mV.
    input_norm_alpha: float = 0.0


class Brain:
    def __init__(self, c: Connectome, params: LIFParams | None = None, device: str | None = None,
                 seed: int = 0):
        self.c = c
        self.p = params or LIFParams()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n = c.n
        p = self.p

        W = c.W.tocsr()
        if p.input_norm_alpha > 0:
            tot = np.asarray(abs(W).sum(axis=1)).ravel()
            med = np.median(tot[tot > 0])
            scale = np.clip((med / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.5, 5.0).astype(np.float32)
            import scipy.sparse as sp
            W = (sp.diags(scale) @ W).tocsr()
        self.W = torch.sparse_csr_tensor(
            torch.from_numpy(W.indptr.astype(np.int64)), torch.from_numpy(W.indices.astype(np.int64)),
            torch.from_numpy(W.data * np.float32(p.w_syn)), size=(self.n, self.n), dtype=torch.float32,
        ).to(self.device)

        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.v = torch.full((self.n,), p.v_rest, device=self.device)
        self.g = torch.zeros(self.n, device=self.device)
        self.refrac = torch.zeros(self.n, device=self.device)          # ms left in refractory period
        self.drive = torch.zeros(self.n, device=self.device)           # I_ext (mV), set by the environment
        self.poisson_p = torch.zeros(self.n, device=self.device)       # per-step spike prob for forced neurons
        self.rate = torch.zeros(self.n, device=self.device)            # running rate estimate (Hz)
        self.spikes = torch.zeros(self.n, device=self.device)          # spikes this step (0/1)
        self.adapt = torch.zeros(self.n, device=self.device)           # adaptation current (mV)
        self.res = torch.ones(self.n, device=self.device)              # synaptic resource x (STD)
        self.active = torch.ones(self.n, device=self.device)           # 0 = frozen (simulated elsewhere)
        self.n_delay = max(1, int(round(p.delay / p.dt)))
        self.spike_buf = torch.zeros(self.n_delay, self.n, device=self.device)
        self.buf_pos = 0
        self.t = 0.0                                                   # ms
        self.step_count = 0
        self._a_m = math.exp(-p.dt / p.tau_m)
        self._a_s = math.exp(-p.dt / p.tau_syn)
        self._a_r = math.exp(-p.dt / p.rate_tau)
        self._a_ad = math.exp(-p.dt / p.adapt_tau) if p.adapt_jump > 0 else 0.0
        self._a_std = math.exp(-p.dt / p.std_tau) if p.std_u > 0 else 1.0

    # ------------------------------------------------------------------ input helpers
    def set_drive(self, idx, mv) -> None:
        """Set the injected current (mV) of neurons `idx` (numpy indices) to `mv` (scalar or array)."""
        idx_t = torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)
        self.drive[idx_t] = torch.as_tensor(np.asarray(mv, dtype=np.float32), device=self.device)

    def freeze(self, idx) -> None:
        """Neurons `idx` never spike in the LIF (they are simulated as rate units in optic.py)."""
        self.active[torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)] = 0.0

    def set_poisson(self, idx, rate_hz) -> None:
        idx_t = torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)
        r = torch.as_tensor(np.asarray(rate_hz, dtype=np.float32), device=self.device)
        self.poisson_p[idx_t] = r * (self.p.dt / 1000.0)
        self._poisson_on = bool(np.any(np.asarray(rate_hz) > 0)) or bool(self.poisson_p.count_nonzero() > 0)

    # ------------------------------------------------------------------ dynamics
    @torch.no_grad()
    def step(self, n_steps: int = 1) -> None:
        p = self.p
        for _ in range(n_steps):
            # synaptic input from spikes emitted `delay` ago
            delayed = self.spike_buf[self.buf_pos]
            self.g.mul_(self._a_s).add_(self.W @ delayed)

            # membrane (exponential Euler with g and drive held constant over the step)
            target = p.v_rest + self.g + self.drive - self.adapt
            self.v = target + (self.v - target) * self._a_m
            in_ref = self.refrac > 0
            self.v = torch.where(in_ref, torch.full_like(self.v, p.v_reset), self.v)
            self.refrac = torch.clamp(self.refrac - p.dt, min=0.0)

            spikes = (self.v >= p.v_th).float() * self.active
            if getattr(self, "_poisson_on", False):   # python flag: avoids a GPU sync per step
                forced = (torch.rand(self.n, generator=self.gen, device=self.device) < self.poisson_p).float()
                spikes = torch.maximum(spikes, forced)
            fired = spikes > 0
            self.v = torch.where(fired, torch.full_like(self.v, p.v_reset), self.v)
            self.refrac = torch.where(fired, torch.full_like(self.refrac, p.t_ref), self.refrac)
            if p.adapt_jump > 0:
                self.adapt.mul_(self._a_ad).add_(spikes, alpha=p.adapt_jump)

            if p.std_u > 0:
                # transmit with the currently available resource, then deplete and recover
                self.spike_buf[self.buf_pos] = spikes * self.res
                self.res = torch.where(fired, self.res * (1 - p.std_u), self.res)
                self.res = 1.0 - (1.0 - self.res) * self._a_std
            else:
                self.spike_buf[self.buf_pos] = spikes
            self.buf_pos = (self.buf_pos + 1) % self.n_delay
            self.spikes = spikes
            self.rate.mul_(self._a_r).add_(spikes * ((1 - self._a_r) * 1000.0 / p.dt))
            self.t += p.dt
            self.step_count += 1

    def run_ms(self, ms: float) -> None:
        self.step(int(round(ms / self.p.dt)))

    # ------------------------------------------------------------------ readout helpers
    def rates(self, idx) -> np.ndarray:
        return self.rate[torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)].cpu().numpy()

    def mean_rate(self, idx) -> float:
        if len(idx) == 0:
            return 0.0
        return float(self.rate[torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)].mean())

    def total_spikes(self) -> float:
        return float(self.spikes.sum())


def drive_from_intensity(intensity, gain: float = 30.0, half: float = 0.02):
    """Saturating light -> injected current (mV): 30*I/(0.02+I) (doomfly). Threshold-crossing at I~0.006."""
    return gain * intensity / (half + intensity)
