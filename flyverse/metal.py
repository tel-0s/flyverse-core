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
// ---------------------------------------------------------------- ray tracer (world.World._trace)
// One thread per ray; the same intersections, precedence (spheres, then boxes, then planes, strict-less
// override), shading, shadow ray, patterns and value noise as the torch implementation.
#define TRACE_INF 1e9f

struct Scene {
    device const float* sc; device const float* sr;       // (S, 3) ellipsoid centres, radii
    device const float* blo; device const float* bhi;     // (Bx, 3) boxes
    device const float* pp; device const float* pn;       // (P, 3) planes: point, unit normal
    uint S; uint Bx; uint P;
};

static float intersect(thread const Scene& sc, float3 o, float3 d, thread float3& best_n, thread int& best_id, int miss_id) {
    const float eps = 1e-4f;
    float best_t = TRACE_INF; best_id = miss_id; best_n = float3(0.0f);
    for (uint i = 0; i < sc.S; ++i) {
        float3 c = float3(sc.sc[3*i], sc.sc[3*i+1], sc.sc[3*i+2]);
        float3 r = float3(sc.sr[3*i], sc.sr[3*i+1], sc.sr[3*i+2]);
        float3 oc = (o - c) / r; float3 dd = d / r;
        float a = dot(dd, dd); float b = 2.0f * dot(oc, dd); float cc = dot(oc, oc) - 1.0f;
        float disc = b * b - 4.0f * a * cc;
        bool ok = disc > 0.0f;
        float sq = sqrt(max(disc, 0.0f));
        float t0 = (-b - sq) / (2.0f * a); float t1 = (-b + sq) / (2.0f * a);
        float t = t0 > eps ? t0 : t1;
        t = (ok && t > eps) ? t : TRACE_INF;
        if (t < best_t) {
            best_t = t; best_id = (int)i;
            float3 pt = o + d * t;
            float3 n = (pt - c) / (r * r);
            best_n = n / max(length(n), 1e-9f);
        }
    }
    for (uint i = 0; i < sc.Bx; ++i) {
        float3 lo = float3(sc.blo[3*i], sc.blo[3*i+1], sc.blo[3*i+2]);
        float3 hi = float3(sc.bhi[3*i], sc.bhi[3*i+1], sc.bhi[3*i+2]);
        float3 dsafe = float3(fabs(d.x) < 1e-9f ? 1e-9f : d.x, fabs(d.y) < 1e-9f ? 1e-9f : d.y, fabs(d.z) < 1e-9f ? 1e-9f : d.z);
        float3 inv = 1.0f / dsafe;
        float3 tlo = (lo - o) * inv; float3 thi = (hi - o) * inv;
        float3 tn = min(tlo, thi); float3 tf = max(tlo, thi);
        int ax = 0; float t_enter = tn.x;
        if (tn.y > t_enter) { t_enter = tn.y; ax = 1; }
        if (tn.z > t_enter) { t_enter = tn.z; ax = 2; }
        float t_exit = min(tf.x, min(tf.y, tf.z));
        float t = (t_exit > t_enter && t_enter > eps) ? t_enter : TRACE_INF;
        if (t < best_t) {
            best_t = t; best_id = (int)(sc.S + i);
            float dax = ax == 0 ? d.x : (ax == 1 ? d.y : d.z);
            float sgn = -(dax > 0.0f ? 1.0f : (dax < 0.0f ? -1.0f : 0.0f));
            best_n = float3(ax == 0 ? sgn : 0.0f, ax == 1 ? sgn : 0.0f, ax == 2 ? sgn : 0.0f);
        }
    }
    for (uint i = 0; i < sc.P; ++i) {
        float3 pt = float3(sc.pp[3*i], sc.pp[3*i+1], sc.pp[3*i+2]);
        float3 n = float3(sc.pn[3*i], sc.pn[3*i+1], sc.pn[3*i+2]);
        float denom = dot(d, n);
        float t = dot(pt - o, n) / (fabs(denom) < 1e-9f ? 1e-9f : denom);
        t = (t > eps && fabs(denom) > 1e-9f) ? t : TRACE_INF;
        if (t < best_t) {
            best_t = t; best_id = (int)(sc.S + sc.Bx + i);
            best_n = dot(n, d) > 0.0f ? -n : n;
        }
    }
    return best_t;
}

static float pymod(float x, float m) { float r = fmod(x, m); return r < 0.0f ? r + m : r; }

static float hash3(long i0, long i1, long i2) {
    const long M = 2147483647L;
    long h = i0 * 374761393L + i1 * 668265263L + i2 * 2147483647L;
    h = h % M; if (h < 0) h += M;
    h = (h ^ (h >> 13)) * 1274126177L;
    h = h % M; if (h < 0) h += M;
    long r = h % 65536L; if (r < 0) r += 65536L;
    return (float)r / 32768.0f - 1.0f;
}

static float value_noise(float3 p, float scale) {
    float3 q = p / scale;
    float3 fl = floor(q);
    long i0 = (long)fl.x, i1 = (long)fl.y, i2 = (long)fl.z;
    float3 f = q - fl;
    f = f * f * (3.0f - 2.0f * f);
    float out = 0.0f;
    for (int dx = 0; dx < 2; ++dx) {
        float wx = dx ? f.x : 1.0f - f.x;
        for (int dy = 0; dy < 2; ++dy) {
            float wy = dy ? f.y : 1.0f - f.y;
            for (int dz = 0; dz < 2; ++dz) {
                float wz = dz ? f.z : 1.0f - f.z;
                out += wx * wy * wz * hash3(i0 + dx, i1 + dy, i2 + dz);
            }
        }
    }
    return out;
}

// light = [lp(3), lc(4), amb(4)]; mats: refl, refl2, emit (K, 4), pattern (K) int, pscale (K); miss id = K - 1
kernel void trace_rays(device const float* o [[buffer(0)]], device const float* d [[buffer(1)]],
                       device const float* sc [[buffer(2)]], device const float* sr [[buffer(3)]],
                       device const float* blo [[buffer(4)]], device const float* bhi [[buffer(5)]],
                       device const float* pp [[buffer(6)]], device const float* pn [[buffer(7)]],
                       device const float* refl [[buffer(8)]], device const float* refl2 [[buffer(9)]],
                       device const float* emit [[buffer(10)]], device const int* pattern [[buffer(11)]],
                       device const float* pscale [[buffer(12)]], device const float* light [[buffer(13)]],
                       device float* out [[buffer(14)]],
                       constant uint& S [[buffer(15)]], constant uint& Bx [[buffer(16)]], constant uint& P [[buffer(17)]],
                       constant uint& K [[buffer(18)]], constant uint& M [[buffer(19)]], constant float& detail [[buffer(20)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= M) return;
    Scene scene = {sc, sr, blo, bhi, pp, pn, S, Bx, P};
    float3 ro = float3(o[3*tid], o[3*tid+1], o[3*tid+2]);
    float3 rd = float3(d[3*tid], d[3*tid+1], d[3*tid+2]);
    float3 n; int mid;
    float t = intersect(scene, ro, rd, n, mid, (int)K - 1);
    if (!(t < TRACE_INF)) { out[4*tid] = 0.0f; out[4*tid+1] = 0.0f; out[4*tid+2] = 0.0f; out[4*tid+3] = 0.0f; return; }
    float3 p = ro + rd * t;
    float3 lp = float3(light[0], light[1], light[2]);
    float3 to_l = lp - p;
    float dist = length(to_l);
    float3 l = to_l / max(dist, 1e-9f);
    float lam = max(dot(n, l), 0.0f);
    float3 n2; int id2;
    float ts = intersect(scene, p + n * 1e-3f, l, n2, id2, (int)K - 1);
    float lit = ts >= dist ? 1.0f : 0.0f;
    float falloff = 4.0f / (1.0f + dist * dist);
    // texture
    int pat = pattern[mid]; float psc = pscale[mid];
    float fx = floor(p.x / psc); float fy = floor(p.y / psc);
    bool checks = pymod(fx + fy, 2.0f) == 1.0f;
    float horiz = fabs(n.x) > 0.5f ? fy : fx;
    bool stripes = pymod(horiz, 2.0f) == 1.0f;
    bool planks = (pymod(fy, 2.0f) == 1.0f) || (pymod(fx + 3.0f * fy, 7.0f) == 0.0f);
    bool second = pat == 1 ? checks : (pat == 2 ? stripes : (pat == 3 ? planks : false));
    device const float* rf = second ? refl2 + 4 * mid : refl + 4 * mid;
    float m = 1.0f;
    if (detail > 0.0f) {
        m = 1.0f + detail * (0.5f * value_noise(p, 0.02f) + 0.3f * value_noise(p, 0.008f) + 0.2f * value_noise(p, 0.003f));
        m = max(m, 0.1f);
    }
    float shade = lam * lit * falloff;
    for (int k = 0; k < 4; ++k) {
        float r = rf[k];
        if (detail > 0.0f) r = r * m;
        out[4*tid + k] = r * (light[7 + k] + light[3 + k] * shade) + emit[4 * mid + k];
    }
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


def trace_rays(o, d, sc, sr, blo, bhi, pp, pn, refl, refl2, emit, pattern, pscale, light, detail: float) -> torch.Tensor:
    """Radiance (M, 4) for rays o, d (M, 3) in the packed scene (world.World). Empty object classes pass a
    dummy row with count 0."""
    o = _c(o); d = _c(d)
    M = o.shape[0]
    out = torch.empty(M, 4, dtype=torch.float32, device=o.device)
    if M == 0:
        return out
    def buf(t, n):
        return _c(t) if n else torch.zeros(1, 3, dtype=torch.float32, device=o.device)
    S, Bx, P = sc.shape[0], blo.shape[0], pp.shape[0]
    lib().trace_rays(o, d, buf(sc, S), buf(sr, S), buf(blo, Bx), buf(bhi, Bx), buf(pp, P), buf(pn, P),
                     _c(refl), _c(refl2), _c(emit), _c(pattern, torch.int32), _c(pscale), _c(light), out,
                     S, Bx, P, refl.shape[0], M, float(detail), threads=M, group_size=GROUP)
    return out
