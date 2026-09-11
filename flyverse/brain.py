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

from .connectome import Connectome
from .device import resolve, sparse_matrix


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

DEFAULT_STD_U_BY_TYPE = {r"^ORN_": 0.2, r"^(lLN|v2LN|v3LN|il3LN|l2LN|vLN)": 0.2, r"(_l2PN|_adPN|_lPN|_lvPN|_ilPN|_ivPN|_vPN|PN\d)": 0.2}


class Brain:
    def __init__(self, c: Connectome, params: LIFParams | None = None, device: str | None = None,
                 seed: int = 0, batch: int = 1):
        self.c = c
        self.p = params or LIFParams()
        self.device = resolve(device)
        self.n = c.n
        self.B = int(batch)
        p = self.p

        W = c.W.tocsr()
        if p.conn_cap > 0:
            W = W.copy()
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
        if p.input_norm_alpha > 0:
            tot = np.asarray(abs(W).sum(axis=1)).ravel()
            scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
            import scipy.sparse as sp
            W = (sp.diags(scale) @ W).tocsr()
            self.input_scale = scale
        W = W.copy(); W.data = W.data * np.float32(p.w_syn)
        self.event_driven = (self.device.type != "cuda") if p.event_driven is None else bool(p.event_driven)
        self.set_weights(W)

        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        B, N, dev = self.B, self.n, self.device
        self.v = torch.full((B, N), p.v_rest, device=dev)
        self.g = torch.zeros(B, N, device=dev)
        self.refrac = torch.zeros(B, N, device=dev)          # ms left in refractory period
        self.drive = torch.zeros(B, N, device=dev)           # I_ext (mV), set by the environment
        self.poisson_p = torch.zeros(B, N, device=dev)       # per-step spike prob for forced neurons
        self.rate = torch.zeros(B, N, device=dev)            # running rate estimate (Hz)
        self.spikes = torch.zeros(B, N, device=dev)          # spikes this step (0/1)
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

    def set_weights(self, W) -> None:
        """Install a (post, pre) scipy sparse matrix of synaptic weights in mV (already scaled by w_syn)."""
        import scipy.sparse as sp
        if self.event_driven:
            Wc = sp.csc_matrix(W)                                      # pre-major: outputs of each neuron
            self.W = None
            self._out_ptr = torch.from_numpy(Wc.indptr.astype(np.int64)).to(self.device)
            self._out_post = torch.from_numpy(Wc.indices.astype(np.int64)).to(self.device)
            self._out_w = torch.from_numpy(Wc.data.astype(np.float32)).to(self.device)
        else:
            self.W = sparse_matrix(W, self.device)

    def _add_synaptic_input(self, x: torch.Tensor) -> None:
        """g += W @ x for transmitted spikes x (B, N); event-driven or one sparse matmul for all B brains."""
        if not self.event_driven:
            self.g.add_((self.W @ x.T.contiguous()).T)                  # contiguous: 4x faster spmm
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
        for t in (self.g, self.refrac, self.drive, self.poisson_p, self.rate, self.spikes, self.adapt):
            t[sel] = 0.0
        self.res[sel] = 1.0
        self.spike_buf[:, sel] = 0.0
        self._rate_np_key = None

    # ------------------------------------------------------------------ dynamics
    @torch.no_grad()
    def step(self, n_steps: int = 1) -> None:
        p = self.p
        for _ in range(n_steps):
            # synaptic input from spikes emitted `delay` ago: one sparse matmul for all B brains
            self.g.mul_(self._a_s)
            self._add_synaptic_input(self.spike_buf[self.buf_pos])          # spikes emitted `delay` ago, (B, N)

            # membrane (exponential Euler with g and drive held constant over the step)
            target = p.v_rest + self.g + self.drive - self.adapt
            self.v = target + (self.v - target) * self._a_m
            in_ref = self.refrac > 0
            self.v = torch.where(in_ref, torch.full_like(self.v, p.v_reset), self.v)
            self.refrac = torch.clamp(self.refrac - p.dt, min=0.0)

            spikes = (self.v >= p.v_th).float() * self.active
            if self._poisson_on:
                forced = (torch.rand(self.v.shape, generator=self.gen, device=self.device) < self.poisson_p).float()
                spikes = torch.maximum(spikes, forced)
            fired = spikes > 0
            self.v = torch.where(fired, torch.full_like(self.v, p.v_reset), self.v)
            self.refrac = torch.where(fired, torch.full_like(self.refrac, p.t_ref), self.refrac)
            if p.adapt_jump > 0:
                self.adapt.mul_(self._a_ad).add_(spikes, alpha=p.adapt_jump)

            if self._std_on:
                # transmit with the currently available resource, then deplete and recover
                self.spike_buf[self.buf_pos] = spikes * self.res
                self.res = torch.where(fired, self.res * (1 - self.std_u_vec), self.res)
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
        s = self.spikes.sum(dim=1)
        return float(s[0]) if self.B == 1 else s.cpu().numpy()


def drive_from_intensity(intensity, gain: float = 30.0, half: float = 0.02):
    """Saturating light -> injected current (mV): 30*I/(0.02+I) (doomfly). Threshold-crossing at I~0.006."""
    return gain * intensity / (half + intensity)
