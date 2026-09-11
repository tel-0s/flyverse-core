"""Custom Metal kernels for the Apple-GPU (MPS) backend.

Torch's MPS backend has no graph capture and ~40 us per kernel launch, so the ~20 elementwise ops of a LIF
step and the COO sparse matmuls (5x off memory bandwidth) dominate on a Mac. `torch.mps.compile_shader`
(torch >= 2.7) compiles Metal source at runtime and calls it on torch tensors; the kernels here replace

    * the event-driven synaptic input (brain.py): one launch, one thread per presynaptic neuron, no host
      sync (the torch gather chain needs `nonzero` and ~12 launches);
    * the whole LIF state update: one launch instead of ~18;
    * the optic lobe's sparse products: CSR with one SIMD group (32 lanes) per row -- 0.6 ms for the
      8.8M-synapse recurrence against 3.6 ms for torch's COO kernel -- and its fused substep.

Arithmetic follows the torch implementations op for op; Metal compiles with fast-math, so multiply-adds may
be contracted and results are equal to ~1e-6 relative rather than bit-identical. Everything falls back to the
torch paths when Metal is unavailable or FLYVERSE_METAL=0.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
import torch

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// g[b, post] += w * x[b, pre] over the outputs of every neuron that transmitted (x != 0). One SIMD group
// per (brain, presynaptic neuron): silent neurons exit at once, long output lists are strided over 32 lanes.
kernel void event_scatter(device const int* out_ptr [[buffer(0)]], device const int* out_post [[buffer(1)]],
                          device const float* out_w [[buffer(2)]], device const float* x [[buffer(3)]],
                          device atomic_float* g [[buffer(4)]], constant uint& N [[buffer(5)]],
                          constant uint& total [[buffer(6)]],
                          uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint gid = tid / 32;
    if (gid >= total) return;
    float xv = x[gid];
    if (xv == 0.0f) return;
    uint b = gid / N; uint pre = gid - b * N;
    device atomic_float* gb = g + b * N;
    int end = out_ptr[pre + 1];
    for (int j = out_ptr[pre] + (int)lane; j < end; j += 32)
        atomic_fetch_add_explicit(&gb[out_post[j]], out_w[j] * xv, memory_order_relaxed);
}

// The LIF update after the synaptic input (brain.Brain.step), one thread per (brain, neuron).
// P = [v_rest, v_reset, v_th, a_m, dt, t_ref, adapt_jump, a_ad, a_std, a_r, rate_gain]
// flags: 1 poisson forcing on, 2 synaptic depression on, 4 record spike counts, 8 adaptation on
kernel void lif_update(device float* v [[buffer(0)]], device const float* g [[buffer(1)]],
                       device const float* drive [[buffer(2)]], device float* adapt [[buffer(3)]],
                       device float* refrac [[buffer(4)]], device const float* active [[buffer(5)]],
                       device const float* poisson_p [[buffer(6)]], device const float* rnd [[buffer(7)]],
                       device float* res [[buffer(8)]], device const float* std_u [[buffer(9)]],
                       device float* spikes [[buffer(10)]], device float* out_buf [[buffer(11)]],
                       device float* rate [[buffer(12)]], device float* counts [[buffer(13)]],
                       device const float* P [[buffer(14)]], constant uint& N [[buffer(15)]],
                       constant uint& total [[buffer(16)]], constant uint& flags [[buffer(17)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    uint i = tid % N;
    float target = P[0] + g[tid] + drive[tid] - adapt[tid];
    float vv = target + (v[tid] - target) * P[3];
    float rf = refrac[tid];
    if (rf > 0.0f) vv = P[1];
    rf = max(rf - P[4], 0.0f);
    float s = (vv >= P[2] ? 1.0f : 0.0f) * active[i];
    if (flags & 1u) {
        float forced = rnd[tid] < poisson_p[tid] ? 1.0f : 0.0f;
        s = max(s, forced);
    }
    bool fired = s > 0.0f;
    if (fired) { vv = P[1]; rf = P[5]; }
    v[tid] = vv; refrac[tid] = rf;
    if (flags & 8u) adapt[tid] = adapt[tid] * P[7] + s * P[6];
    if (flags & 2u) {
        float r = res[tid];
        out_buf[tid] = s * r;
        if (fired) r = r * (1.0f - std_u[i]);
        res[tid] = 1.0f - (1.0f - r) * P[8];
    } else {
        out_buf[tid] = s;
    }
    spikes[tid] = s;
    if (flags & 4u) counts[tid] += s;
    rate[tid] = rate[tid] * P[9] + s * P[10];
}

// y[b, row] = sum_j vals[j] * x[b, indices[j]] for a CSR matrix; one SIMD group per (brain, row).
kernel void csr_spmv(device const int* indptr [[buffer(0)]], device const int* indices [[buffer(1)]],
                     device const float* vals [[buffer(2)]], device const float* x [[buffer(3)]],
                     device float* y [[buffer(4)]], constant uint& n_rows [[buffer(5)]],
                     constant uint& n_cols [[buffer(6)]], constant uint& B [[buffer(7)]],
                     uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint gid = tid / 32;
    if (gid >= n_rows * B) return;
    uint b = gid / n_rows; uint row = gid - b * n_rows;
    device const float* xb = x + b * n_cols;
    float acc = 0.0f;
    int end = indptr[row + 1];
    for (int j = indptr[row] + (int)lane; j < end; j += 32) acc += vals[j] * xb[indices[j]];
    acc = simd_sum(acc);
    if (lane == 0) y[gid] = acc;
}

// dr = clamp(v + b, 0, 1) - b  (optic.OpticLobe.rates() minus the operating point)
kernel void optic_dr(device const float* v [[buffer(0)]], device const float* bvec [[buffer(1)]],
                     device float* dr [[buffer(2)]], constant uint& n [[buffer(3)]], constant uint& total [[buffer(4)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    float bb = bvec[tid % n];
    dr[tid] = clamp(v[tid] + bb, 0.0f, 1.0f) - bb;
}

// One optic-lobe substep after y = W_rr @ dr (optic.OpticLobe._substep); writes the next dr.
kernel void optic_substep(device float* v [[buffer(0)]], device float* adapt [[buffer(1)]], device float* dr [[buffer(2)]],
                          device const float* y [[buffer(3)]], device const float* pr_in [[buffer(4)]],
                          device const float* spk_in [[buffer(5)]], device const float* a [[buffer(6)]],
                          device const float* bvec [[buffer(7)]], constant float& gain_rr [[buffer(8)]],
                          constant float& adapt_gain [[buffer(9)]], constant float& a_ad [[buffer(10)]],
                          constant uint& n [[buffer(11)]], constant uint& total [[buffer(12)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    uint i = tid % n;
    float d = dr[tid];
    float inp = gain_rr * y[tid] + pr_in[tid] - adapt_gain * adapt[tid];
    inp = inp + spk_in[tid];
    float vv = inp + (v[tid] - inp) * a[i];
    v[tid] = vv;
    adapt[tid] = d + (adapt[tid] - d) * a_ad;
    float bb = bvec[i];
    dr[tid] = clamp(vv + bb, 0.0f, 1.0f) - bb;
}
"""

_LIB = None
GROUP = 256          # threads per threadgroup (a multiple of the 32-lane SIMD width)


def available() -> bool:
    """Metal kernels can be used: MPS present, torch has compile_shader, not disabled by FLYVERSE_METAL=0."""
    if os.environ.get("FLYVERSE_METAL", "1") == "0":
        return False
    return bool(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"))


def use(device, explicit=None) -> bool:
    """Backend decision for a component on `device`: explicit True/False, else auto (MPS and available)."""
    if explicit is not None:
        if explicit and not (torch.device(device).type == "mps" and available()):
            raise ValueError("Metal kernels need an MPS device with torch.mps.compile_shader (and FLYVERSE_METAL != 0)")
        return bool(explicit)
    return torch.device(device).type == "mps" and available()


def lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(SOURCE)
    return _LIB


def _c(t: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """The kernels read raw storage: a non-contiguous tensor would be silently misread."""
    if not t.is_contiguous():
        raise ValueError("Metal kernels need contiguous tensors")
    if t.dtype != dtype:
        raise ValueError(f"expected {dtype}, got {t.dtype}")
    return t


class MetalCSR:
    """A scipy sparse matrix on the GPU for batched matrix-vector products y (B, rows) = M @ x (B, cols)."""

    def __init__(self, M: sp.spmatrix, device):
        M = sp.csr_matrix(M, dtype=np.float32)
        M.sum_duplicates(); M.sort_indices()
        if M.nnz >= 2**31:
            raise ValueError("MetalCSR needs fewer than 2^31 nonzeros")
        self.shape = tuple(int(s) for s in M.shape)
        self.nnz = int(M.nnz)
        self.device = torch.device(device)
        self.indptr = torch.from_numpy(M.indptr.astype(np.int32)).to(self.device)
        self.indices = torch.from_numpy(M.indices.astype(np.int32)).to(self.device)
        self.values = torch.from_numpy(M.data.astype(np.float32)).to(self.device)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, cols) or (cols,) -> (B, rows) (or (rows,) for 1-D input)."""
        squeeze = x.dim() == 1
        x = _c(x[None] if squeeze else x)
        B, cols = x.shape
        rows = self.shape[0]
        if cols != self.shape[1]:
            raise ValueError(f"matvec: x has {cols} columns, matrix has {self.shape[1]}")
        y = torch.empty(B, rows, dtype=torch.float32, device=self.device)
        if rows and B:
            lib().csr_spmv(self.indptr, self.indices, self.values, x, y, rows, cols, B, threads=B * rows * 32, group_size=GROUP)
        return y[0] if squeeze else y

    def _nnz(self) -> int:
        return self.nnz


def event_scatter(out_ptr, out_post, out_w, x: torch.Tensor, g: torch.Tensor) -> None:
    """g (B, N) += W @ x for transmitted spikes x (B, N), W given pre-major (CSC: out_ptr, out_post, out_w)."""
    x = _c(x); g = _c(g)
    B, N = x.shape
    lib().event_scatter(out_ptr, out_post, out_w, x, g, N, B * N, threads=B * N * 32, group_size=GROUP)


def lif_update(v, g, drive, adapt, refrac, active, poisson_p, rnd, res, std_u, spikes, out_buf, rate, counts, P, flags: int) -> None:
    B, N = v.shape
    args = [_c(t) for t in (v, g, drive, adapt, refrac, active, poisson_p, rnd, res, std_u, spikes, out_buf, rate, counts, P)]
    lib().lif_update(*args, N, B * N, int(flags), threads=B * N, group_size=GROUP)


def optic_dr(v, bvec, dr) -> None:
    B, n = v.shape
    lib().optic_dr(_c(v), _c(bvec), _c(dr), n, B * n, threads=B * n, group_size=GROUP)


def optic_substep(v, adapt, dr, y, pr_in, spk_in, a, bvec, gain_rr: float, adapt_gain: float, a_ad: float) -> None:
    B, n = v.shape
    args = [_c(t) for t in (v, adapt, dr, y, pr_in, spk_in, a, bvec)]
    lib().optic_substep(*args, float(gain_rr), float(adapt_gain), float(a_ad), n, B * n, threads=B * n, group_size=GROUP)
