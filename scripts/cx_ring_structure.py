"""Thread 5A -- a type-level ring mechanism, or none (docs/audits/compass_ring_mechanism.md). CPU structure pass.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python scripts/cx_ring_structure.py [--out out/cx5/structure] [--no-banc]

The effective ring circuit under the SHIPPED rules (`brain._shaped_weights`: receptor sign stage, connection cap 60,
path gains, same-type damping x0.1; then the fan-in scale and w_syn 0.275 mV -- exactly what Brain installs) among EPG /
PEN_a / PEN_b / PEG / Delta7 (+ EPGt, the ER / ExR ring neurons, GLNO), at the cell level and at the 16-wedge level:

  * offset matrices: EPG -> PEN, PEN -> EPG (the +-1 wedge shift, by PEN side), PEG -> EPG, Delta7 -> EPG (cosine
    profile against the Delta7's output tile), ER / ExR -> EPG (no wedge: per-type volleys);
  * the EPG x EPG two-step matrices through each relay (mV^2 per spike, cx_wedge's aggregate) and their ring-Fourier
    spectrum: lambda_k = Re(e_k^H M e_k) / 16 for k = 0 (uniform) and k = 1 (bump), plus the exact eigenvalues of M16;
  * the linearised rate model. The threshold-linear model r = f(tau_syn A r + forced) (cx_wedge section 5) linearised at
    a uniform state with the same slope gamma (Hz/mV) for every cell gives dr = gamma tau A dr on the compass
    sub-circuit, so a mode is self-sustaining when gamma tau Re(mu) > 1 for an eigenvalue mu of A (mV per spike);
    for the EPG-only two-step reduction (relays adiabatic at the same gamma) the loop gain of ring mode k is
    gamma^2 tau^2 lambda_k, so the critical gain is gamma_crit(k) = 1 / (tau sqrt(lambda_k)) when lambda_k > 0.
    Both are reported; the k = 1 mode's gain, the k = 0 mode's gain and the Delta7 : ring-feedback ratio;
  * the threshold-linear fixed point itself under the cx_wedge protocol (10 Hz EPG background, 4 wedges +40 Hz,
    release) at gE = gD = 1 -- the number that decides, because the linear mode gain is not the binding constraint
    when the relays sit below threshold (the DC balance on PEN is reported as u_PEN against the 7 mV gap).

Then, one change at a time, each labelled with its evidence (docs/audits/compass_ring_mechanism.md section 2):
  (a) GLNO = glutamate  (scratch cache out/cache_<hash>, cx_wedge.load_connectome({'GLNO': 'glutamate'}));
  (b) the connection cap lifted (LIFParams(conn_cap=0)) -- an INSTRUMENT (a global default), capped vs uncapped weights;
  (c) the receptor tiers: what receptors_by_type.csv says for the ring's rows and what 'sign' / 'sign+gain' / 'full' do;
  (d) the monoamine slow class restricted to the ring (the 5-HT / DA / OA rows on EPG / PEN under 'full');
  (e) the fan-in normalisation on PEN / EPG (totals and scale; scale 1.00 per cx_glno -- confirmed here);
  (f) the same-type damping x0.1 (EPG->EPG, PEN_a->PEN_a, PEN_b->PEN_b, Delta7->Delta7 edges; LIFParams(same_type_gain=1));
  and the pairwise combinations with (a). Every configuration gets the same tables; the ranking is by the rate model's
  post-release bump at the shipped gains, then by gamma_crit(k = 1) and the PEN DC margin.

Outputs (out/cx5/structure/): structure.json (every number, provenance), structure.md (the tables), matrices.npz
(cell-level A blocks and M16 per configuration), evidence_glno.json (MaleCNS / BANC GLNO transmitter read here).
Nothing here changes a default; no file outside out/ is written.

THE THREE DEFECTS OF THE 5A PASS, FIXED HERE (thread 6A, docs/audits/compass_dc_balance.md; the 5A statements are in
docs/audits/compass_ring_mechanism.md sections 1.3 / 3.3 and its skeptic pass R2 / R3). Each fix is the DEFAULT and the
5A behaviour is reachable for comparison with `--legacy` (or the individual `--drive rate`, `--reduction two-step`,
`--gain uniform`); `--legacy` reproduces the 5A numbers exactly and is what the validation table compares against:

  1. FORCED DRIVE AS A CURRENT (`--drive current`, default; `--drive rate` = 5A). 5A's rate_fixed_point added the
     forced background as a RATE (r = f(tau A r) + forced), so a driven EPG sat at u = -22 to -68 mV while 'firing' at
     10-50 Hz and gamma_EPG = f'(u_EPG) = 0 BY CONSTRUCTION -- no recurrent EPG term could ever engage. The drive now
     enters as a current u_forced = f^-1(rate) under the same smoothed LIF f-I (`u_for_rate`: 10 Hz -> 6.628 mV,
     50 Hz -> 11.993 mV at sigma 2 mV; derived, not hardcoded), i.e. r = f(tau A r + u_forced).
  2. THE ONE-STEP EPG -> EPG TERM IS KEPT (`--reduction with-direct`, default; `--reduction two-step` = 5A). The
     EPG-only reduction's loop gain of ring mode k is now gamma tau d_k + gamma^2 tau^2 lambda_k (d_k = the one-step
     EPG -> EPG ring-Fourier coefficient in mV, lambda_k = the two-step one in mV^2), so gamma_crit(k) solves that
     quadratic (`gamma_crit_combined`). 5A computed lambda_k alone and printed d_k in the same row without using it.
     The rate at which a supercritical one-step ring saturates -- the rate where f'(u) falls back to gamma_crit,
     `rate_at_gain` -- is reported per configuration and is what the F-family validation checks.
  3. PER-CELL GAIN AT THE REALISED FIXED POINT (`--gain per-cell`, default; `--gain uniform` = 5A's gamma 6 reading).
     The linearisation is the true Jacobian J = diag(gamma_i) tau A with gamma_i = f'(u_i) (`lif_fi_prime`, the exact
     derivative of the same Gauss-Hermite-smoothed f-I) at the fixed point the model actually occupies, reported per
     state (leading eigenvalue, spectral radius, and the leading mode whose EPG profile is ring index k = 1: 'the k = 1
     gain'), alongside the weight-matrix eigenvalues, which are a NECESSARY CONDITION and not the attractor.

`--holds` adds the 6A counterfactual configurations (ExR6 / ER6 / ER4m held at 0 onto PEN and EPG, and each type
alone) through the same `type_path_gain` stage `cx_wedge.py --hold-edges` uses, so the structure pass and the batch
hold the same edges. `--validate-batch DIR` scores the fixed tool against an already-fetched batch (out/cx5): the
predicted saturation rate per configuration against the measured `bump_hz_post`, and the predicted / measured bump.
Nothing here changes a default; no file outside out/ is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyverse import brain, connectome  # noqa: E402
import cx_wedge  # noqa: E402

RING16 = cx_wedge.RING16
GROUPS = ("EPG", "PEN", "PEG", "Delta7", "EPGt", "Ring", "GLNO")
THETA_MV = 7.0          # v_th - v_rest of LIFParams()
BUMP_THRESH_HZ = 22.0   # probe_compass_room.THRESH_HZ: the per-cell rate the ledger's bump rule counts as 'in the bump'
SIGMA_MV = 2.0          # the input-noise smoothing of cx_wedge.lif_fi, unchanged from 5A. AN ASSUMPTION, NEVER
                        # MEASURED: nothing in this project has measured the spiking LIF's effective input noise, and
                        # no arm of cx6 varies it. Every slope bound below is a property of this number (max slope
                        # 25.3 Hz/mV at sigma 0.25, 11.0 at 1.0, 8.00 at 2.0; gamma_crit 33.3 is reached at 0.17).
# The 6A hold: the DC inhibition the 5A decomposition names, held at 0 onto the relays and the ring (an `edges`-kind
# LABELLED COUNTERFACTUAL; the same regex pair scripts/cx_wedge.py --hold-edges installs).
HOLD_PRE = r"^(ExR6|ER6|ER4m)$"
HOLD_POST = r"^(PEN_|EPG$)"


# ------------------------------------------------------------------------------------------------ the LIF f-I, its
# derivative and its inverse (6A fixes 1 and 3; the f-I itself is cx_wedge.lif_fi, unchanged)
def lif_fi_det(u, p: brain.LIFParams | None = None):
    """The deterministic LIF f-I (Hz), 1000 / (t_ref + tau_m ln(u / (u - theta))) above threshold and 0 below."""
    p = p or brain.LIFParams()
    theta = p.v_th - p.v_rest
    u = np.atleast_1d(np.asarray(u, dtype=float))
    out = np.zeros_like(u)
    m = u > theta + 1e-12
    if m.any():
        out[m] = 1000.0 / (p.t_ref + p.tau_m * np.log(u[m] / (u[m] - theta)))
    return out


def lif_fi_prime(u, p: brain.LIFParams | None = None, sigma: float = SIGMA_MV, nodes: int = 61):
    """d/du of the smoothed LIF f-I (Hz/mV), the cell gain gamma_i the linearisation needs.

    cx_wedge.lif_fi is the Gaussian convolution f_sigma(u) = E[f_det(u + sigma x)] (Gauss-Hermite, 15 nodes). Its
    derivative E[f_det'(u + sigma x)] cannot be taken node by node: f_det'(u) = f^2 tau_m theta / (1000 u (u - theta))
    DIVERGES as u -> theta+ (the deterministic f-I has infinite slope at threshold; 5A's '25.8 Hz/mV at u 7.1' is a
    point on that divergence, not a maximum), so a node landing near threshold returns hundreds of Hz/mV. Integrating by
    parts removes the singularity exactly -- phi'(x) = -x phi(x), so

        f_sigma'(u) = (1 / sigma) E[x f_det(u + sigma x)]

    -- and the estimate becomes finite, but it is NOT CONVERGED: the maximum over u reads 8.66 Hz/mV at 15 nodes,
    8.31 at 31, 8.27 at 61 (this function's default), 8.19 at 101 and 8.13 at 201, still falling, and numpy's
    `hermegauss` overflows to NaN above ~201 nodes. Exact adaptive quadrature of the same integral gives
    **8.00 Hz/mV at u 8.61 (25.5 Hz)** at sigma 2 mV, so the 61-node default is 3.4 % high. `structure.json` records
    the 61-node value (`max_lif_slope` 8.2747 at u 8.6018, 25.13 Hz).

    Two limits on what any of this means. (i) The bound is a property of the ASSUMED sigma: `SIGMA_MV = 2.0` is
    hard-coded and the effective input noise of the spiking LIF has never been measured in this project. The exact
    maximum slope is 25.3 Hz/mV at sigma 0.25 mV, 11.0 at 1.0 and 8.00 at 2.0, and a gamma_crit of 33.3 Hz/mV would be
    reached at sigma 0.17 mV. (ii) The f-I the fixed point actually iterates is `cx_wedge.lif_fi` at 15 nodes, which is
    not smooth near threshold and does not respect the bound. So the honest statement is: at sigma 2 mV a gamma_crit of
    28.8-33.3 Hz/mV is above every operating point of the exact smoothed f-I by a factor of 3.6-4.2 -- not that such a
    gain is unreachable. 5A's '25.8 Hz/mV at u 7.1' is right for the DETERMINISTIC f-I and is a sample on its
    divergence, so it was never a maximum either."""
    p = p or brain.LIFParams()
    xs, ws = np.polynomial.hermite_e.hermegauss(nodes)
    ws = ws / ws.sum()
    u = np.atleast_1d(np.asarray(u, dtype=float))
    out = np.zeros_like(u)
    for x, w in zip(xs, ws):
        out += w * x * lif_fi_det(u + sigma * x, p)
    return out / sigma


def u_for_rate(hz: float, p: brain.LIFParams | None = None, sigma: float = SIGMA_MV, lo: float = -60.0, hi: float = 400.0) -> float:
    """The mean input u (mV above rest) at which the smoothed LIF f-I fires `hz`: f^-1, by bisection on a monotone
    function. This is how a forced Poisson drive of `hz` enters the rate model as a CURRENT (6A fix 1)."""
    f = lambda x: float(cx_wedge.lif_fi(np.array([x]), p, sigma)[0])   # noqa: E731
    if hz <= 0:
        return float(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < hz:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


_MAX_SLOPE_CACHE = {}


def max_slope(p: brain.LIFParams | None = None, sigma: float = SIGMA_MV) -> tuple:
    """(max f'(u), the u where it peaks, f(u) there), computed from `lif_fi_prime` at its default quadrature order:
    the largest slope the SMOOTHED f-I has at any operating point, at the ASSUMED `sigma` (`SIGMA_MV` = 2.0, never
    measured). The Gauss-Hermite estimate is not converged -- see `lif_fi_prime`; this returns 8.27 Hz/mV where exact
    adaptive quadrature gives 8.00 at u 8.61 (25.5 Hz). A gamma_crit above it does not close a loop in this rate model
    at this sigma (shipped: gamma_crit(k1) 33.4 vs 8.0-8.3), which is a statement about the rate model and its assumed
    noise, not about the spiking LIF."""
    p = p or brain.LIFParams()
    key = (p.t_ref, p.tau_m, p.v_th - p.v_rest, sigma)
    if key not in _MAX_SLOPE_CACHE:
        u = np.linspace(p.v_th - p.v_rest - 4 * sigma, 200.0, 100001)
        d = lif_fi_prime(u, p, sigma)
        i = int(np.argmax(d))
        _MAX_SLOPE_CACHE[key] = (float(d[i]), float(u[i]), float(cx_wedge.lif_fi(np.array([u[i]]), p, sigma)[0]))
    return _MAX_SLOPE_CACHE[key]


def rate_at_gain(gamma: float, p: brain.LIFParams | None = None, sigma: float = SIGMA_MV) -> float:
    """The rate at which a loop whose critical gain is `gamma` saturates: f(u*) at the u* ABOVE the slope peak where
    f'(u*) = gamma. The slope falls monotonically there, so a supercritical loop grows until the gain has fallen back
    to 1 / (loop weight). NaN when gamma exceeds `max_slope`: the loop is not supercritical at any operating point of
    the smoothed f-I at the assumed sigma. (gamma 3.35 -> 144 Hz, 3.01 -> 160 Hz, 33.4 -> NaN.)"""
    p = p or brain.LIFParams()
    if not np.isfinite(gamma) or gamma <= 0:
        return float("nan")
    dmax, u_peak, _ = max_slope(p, sigma)
    if gamma >= dmax:
        return float("nan")
    lo, hi = u_peak, 5000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(lif_fi_prime(np.array([mid]), p, sigma)[0]) > gamma:
            lo = mid
        else:
            hi = mid
    return float(cx_wedge.lif_fi(np.array([0.5 * (lo + hi)]), p, sigma)[0])


def gamma_crit_combined(lam: float, d: float, tau: float) -> float:
    """The uniform cell gain at which ring mode k of the EPG-only reduction reaches loop gain 1 when BOTH the one-step
    EPG -> EPG coefficient d_k (mV) and the two-step coefficient lambda_k (mV^2) are kept (6A fix 2):
    gamma tau d + gamma^2 tau^2 lambda = 1. With d = 0 this is 5A's 1 / (tau sqrt(lambda)); with lambda = 0 it is
    1 / (tau d). inf when no positive gamma solves it."""
    lam, d = float(lam), float(d)
    if abs(lam) < 1e-12:
        return float(1.0 / (tau * d)) if d > 0 else float("inf")
    disc = d * d + 4.0 * lam
    if disc < 0:
        return float("inf")
    g = (-d + np.sqrt(disc)) / (2.0 * tau * lam)
    return float(g) if g > 0 and np.isfinite(g) else float("inf")


# ------------------------------------------------------------------------------------------------ small helpers
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def ring_offset(pos_post, pos_pre, n=16):
    """(post - pre) mod n mapped into [-n/2, n/2): the wedge offset of a projection."""
    d = (np.asarray(pos_post)[:, None] - np.asarray(pos_pre)[None, :]) % n
    return np.where(d >= n / 2, d - n, d)


def offset_profile(B, pos_post, pos_pre, n=16):
    """For a block B[post, pre] (mV per spike): mean over post cells of the summed weight from pre cells at each ring
    offset (post - pre, wedges), offsets -8..7. Cells with NaN positions are dropped."""
    ok_r = np.isfinite(pos_post); ok_c = np.isfinite(pos_pre)
    B = B[ok_r][:, ok_c]; pp = np.asarray(pos_post)[ok_r]; pq = np.asarray(pos_pre)[ok_c]
    off = np.round(ring_offset(pp, pq, n)).astype(int)
    offs = list(range(-n // 2, n // 2))
    prof = np.zeros(len(offs))
    for j, o in enumerate(offs):
        m = off == o
        prof[j] = (B * m).sum(axis=1).mean() if B.shape[0] else 0.0
    return offs, prof


def fourier_modes(M, ks=(0, 1, 2, 3, 4, 8)):
    """Ring-Fourier Rayleigh quotients of a wedge-level matrix M (rows post, cols pre; n x n, ring order):
    lambda_k = Re(e_k^H M e_k) / n with e_k[j] = exp(2 pi i k j / n). Exact eigenvalues for a circulant M."""
    n = M.shape[0]
    j = np.arange(n)
    out = {}
    for k in ks:
        e = np.exp(2j * np.pi * k * j / n)
        out[int(k)] = float(np.real(np.conj(e) @ M @ e) / n)
    return out


def eig_modes(M, n_report=6):
    """Exact eigenvalues of a wedge-level matrix with the dominant ring-Fourier index of each eigenvector."""
    w, V = np.linalg.eig(M)
    n = M.shape[0]
    rows = []
    order = np.argsort(-np.real(w))
    for i in order[:n_report]:
        v = V[:, i]
        spec = np.abs(np.fft.fft(v)) ** 2
        k = int(np.argmin([min(kk, n - kk) for kk in range(n)] if False else [-s for s in spec]))
        k = min(k, n - k)
        rows.append(dict(re=float(np.real(w[i])), im=float(np.imag(w[i])), k=k, k_power=float(spec.max() / max(spec.sum(), 1e-12))))
    return rows


def circuit_modes(A_sub, epg_slice, wedge_of, tau_s, n_report=8):
    """Eigenvalues mu of the compass sub-circuit matrix A (mV per spike, cell level); dr = gamma tau A dr, so a mode is
    self-sustaining at gamma > 1 / (tau Re mu). Each eigenvector is classified by the ring-Fourier index of its EPG part
    (per-wedge mean of the EPG components)."""
    w, V = np.linalg.eig(A_sub)
    rows = []
    for i in np.argsort(-np.real(w)):
        v = V[epg_slice, i]
        prof = np.array([v[wedge_of == k].mean() if (wedge_of == k).any() else 0.0 for k in range(16)])
        spec = np.abs(np.fft.fft(prof)) ** 2
        kk = int(np.argmax(spec)); kk = min(kk, 16 - kk)
        epg_share = float(np.sum(np.abs(v) ** 2) / max(np.sum(np.abs(V[:, i]) ** 2), 1e-12))
        rows.append(dict(re=float(np.real(w[i])), im=float(np.imag(w[i])), k=kk,
                         k_power=float(spec.max() / max(spec.sum(), 1e-12)), epg_share=epg_share,
                         gamma_crit=float(1.0 / (tau_s * np.real(w[i]))) if np.real(w[i]) > 0 else float("inf")))
        if len(rows) >= n_report:
            break
    # the leading mode of each ring index among the top 40 eigenvalues
    lead = {}
    for i in np.argsort(-np.real(w))[:40]:
        v = V[epg_slice, i]
        prof = np.array([v[wedge_of == k].mean() if (wedge_of == k).any() else 0.0 for k in range(16)])
        spec = np.abs(np.fft.fft(prof)) ** 2
        kk = int(np.argmax(spec)); kk = min(kk, 16 - kk)
        if kk not in lead:
            lead[kk] = dict(re=float(np.real(w[i])), im=float(np.imag(w[i])),
                            gamma_crit=float(1.0 / (tau_s * np.real(w[i]))) if np.real(w[i]) > 0 else float("inf"))
    return rows, lead


# ------------------------------------------------------------------------------------------------ the circuit
class Circuit:
    """Effective weights among the compass groups for one (connectome, LIFParams)."""

    def __init__(self, c, p: brain.LIFParams, label: str, evidence: str, cells=None):
        self.c, self.p, self.label, self.evidence = c, p, label, evidence
        self.cells = cells or cx_wedge.compass_cells(c)
        ty = c.neurons.type.fillna("").to_numpy()
        self.cells["GLNO"] = dict(idx=np.flatnonzero(ty == "GLNO"), pos=np.full(int((ty == "GLNO").sum()), np.nan),
                                  label=["GLNO"] * int((ty == "GLNO").sum()), body=c.neurons.bodyId.to_numpy()[ty == "GLNO"])
        self.A, self.scale, self.tot = cx_wedge.effective_weights(c, p)
        self.idx = {g: self.cells[g]["idx"] for g in GROUPS}
        self.pos = {g: np.asarray(self.cells[g]["pos"], float) for g in GROUPS}
        self.wedge_of = np.asarray(np.round(self.pos["EPG"]), int) % 16

    def block(self, post, pre):
        return self.A[self.idx[post]][:, self.idx[pre]].toarray().astype(np.float64)

    def sub(self, groups=GROUPS):
        idx = np.concatenate([self.idx[g] for g in groups])
        return self.A[idx][:, idx].toarray().astype(np.float64), idx

    def one_step(self) -> dict:
        out = {}
        pairs = {"EPG->PEN": ("EPG", "PEN"), "PEN->EPG": ("PEN", "EPG"), "EPG->PEG": ("EPG", "PEG"), "PEG->EPG": ("PEG", "EPG"),
                 "EPG->Delta7": ("EPG", "Delta7"), "Delta7->EPG": ("Delta7", "EPG"), "Delta7->PEN": ("Delta7", "PEN"),
                 "Delta7->PEG": ("Delta7", "PEG"), "Delta7->Delta7": ("Delta7", "Delta7"), "EPG->EPG": ("EPG", "EPG"),
                 "PEN->PEN": ("PEN", "PEN"), "PEN->GLNO": ("PEN", "GLNO"), "GLNO->PEN": ("GLNO", "PEN"), "GLNO->GLNO": ("GLNO", "GLNO"),
                 "EPG->Ring": ("EPG", "Ring"), "Ring->EPG": ("Ring", "EPG"), "Ring->PEN": ("Ring", "PEN"), "Ring->PEG": ("Ring", "PEG"),
                 "PEN->Delta7": ("PEN", "Delta7"), "EPGt->Delta7": ("EPGt", "Delta7"), "Delta7->EPGt": ("Delta7", "EPGt")}
        for name, (pre, post) in pairs.items():
            B = self.block(post, pre); nz = B != 0
            out[name] = dict(pairs=int(nz.sum()), mean_per_pair=float(B[nz].mean()) if nz.any() else 0.0,
                             total_per_post=float(B.sum(axis=1).mean()) if B.size else 0.0)
        return out

    def same_type_blocks(self) -> dict:
        """The same-type edges the x0.1 damping touches, per type, with damped / undamped effective weights."""
        ty = self.c.neurons.type.fillna("").to_numpy()
        out = {}
        for g, pat in (("EPG", "EPG"), ("PEN_a", "PEN_a(PEN1)"), ("PEN_b", "PEN_b(PEN2)"), ("PEG", "PEG"), ("Delta7", "Delta7")):
            idx = np.flatnonzero(ty == pat)
            B = self.A[idx][:, idx].toarray().astype(np.float64); nz = B != 0
            out[g] = dict(cells=int(len(idx)), pairs=int(nz.sum()), mean_per_pair=float(B[nz].mean()) if nz.any() else 0.0,
                          total_per_post=float(B.sum(axis=1).mean()) if B.size else 0.0)
        # cross-type PEN_a <-> PEN_b (NOT damped: different type strings)
        a = np.flatnonzero(ty == "PEN_a(PEN1)"); b = np.flatnonzero(ty == "PEN_b(PEN2)")
        for name, (r, q) in (("PEN_a->PEN_b", (b, a)), ("PEN_b->PEN_a", (a, b))):
            B = self.A[r][:, q].toarray().astype(np.float64); nz = B != 0
            out[name] = dict(pairs=int(nz.sum()), mean_per_pair=float(B[nz].mean()) if nz.any() else 0.0, total_per_post=float(B.sum(axis=1).mean()))
        return out

    def offsets(self) -> dict:
        """Wedge-offset profiles of the ring projections (post - pre in wedges of 22.5 deg)."""
        out = {}
        E, P, G, D = "EPG", "PEN", "PEG", "Delta7"
        side = np.array([str(l)[0] for l in self.cells["PEN"]["label"]])      # PEN side from its glomerulus (L / R)
        for name, post, pre in (("EPG->PEN", P, E), ("PEN->EPG", E, P), ("EPG->PEG", G, E), ("PEG->EPG", E, G),
                                ("EPG->Delta7", D, E), ("Delta7->EPG", E, D), ("Delta7->PEN", P, D), ("EPG->EPG", E, E)):
            B = self.block(post, pre)
            offs, prof = offset_profile(B, self.pos[post], self.pos[pre])
            d = dict(offsets=offs, profile=prof.tolist())
            if pre == P or post == P:
                for s in ("L", "R"):
                    m = side == s
                    if pre == P:
                        _, ps = offset_profile(B[:, m], self.pos[post], self.pos[pre][m])
                    else:
                        _, ps = offset_profile(B[m], self.pos[post][m], self.pos[pre])
                    d[f"profile_{s}"] = ps.tolist()
                    d[f"centroid_{s}"] = float(np.sum(np.array(offs) * np.abs(ps)) / max(np.abs(ps).sum(), 1e-12))
            out[name] = d
        # ring neurons: per-type volleys onto EPG / PEN / PEG (no wedge)
        rt = np.array(self.cells["Ring"]["label"]); types = {}
        for t in sorted(set(rt)):
            sel = rt == t
            row = dict(cells=int(sel.sum()), nt=str(self.c.neurons.nt.to_numpy()[self.idx["Ring"][sel]][0]))
            for post in ("EPG", "PEN", "PEG"):
                B = self.block(post, "Ring")[:, sel]; nz = B != 0
                row[f"to_{post}_per_pair"] = float(B[nz].mean()) if nz.any() else 0.0
                row[f"to_{post}_volley"] = float(B.sum(axis=1).mean())
            Bf = self.block("Ring", "EPG")[sel]; nz = Bf != 0
            row["from_EPG_per_pair"] = float(Bf[nz].mean()) if nz.any() else 0.0
            row["from_EPG_volley"] = float(Bf.sum(axis=1).mean()) if Bf.size else 0.0
            row["two_step_EPG"] = float((self.block("EPG", "Ring")[:, sel] @ Bf).sum(axis=1).mean())
            row["two_step_PEN"] = float((self.block("PEN", "Ring")[:, sel] @ Bf).sum(axis=1).mean())
            if abs(row["two_step_EPG"]) + abs(row["two_step_PEN"]) > 0:
                types[t] = row
        out["Ring_types"] = types
        return out

    def two_step(self) -> dict:
        """EPG x EPG two-step matrices through each relay (mV^2 per spike, cell level and 16-wedge aggregate)."""
        E = "EPG"
        K = {"PEN": self.block(E, "PEN") @ self.block("PEN", E), "PEG": self.block(E, "PEG") @ self.block("PEG", E),
             "Delta7": self.block(E, "Delta7") @ self.block("Delta7", E), "Ring": self.block(E, "Ring") @ self.block("Ring", E)}
        K["direct"] = self.block(E, E)
        # PEN-level loops: PEN -> GLNO -> PEN (mV^2) and PEN -> PEN (direct), reported as EPG x EPG four-step / three-step
        # is not meaningful in mV units; keep them at the PEN level
        Kp = {"GLNO": self.block("PEN", "GLNO") @ self.block("GLNO", "PEN"), "direct": self.block("PEN", "PEN"),
              "Delta7": self.block("PEN", "Delta7") @ self.block("Delta7", E), "Ring": self.block("PEN", "Ring") @ self.block("Ring", E),
              "EPG": self.block("PEN", E)}
        bin16 = lambda x: np.asarray(np.round(x), int) % 16   # noqa: E731
        M16 = {k: cx_wedge.aggregate(v, self.pos["EPG"], self.pos["EPG"], 16, bin16) for k, v in K.items()}
        M16["net"] = M16["PEN"] + M16["PEG"] + M16["Delta7"] + M16["Ring"]
        M16["net_tuned"] = M16["PEN"] + M16["PEG"] + M16["Delta7"]
        Mp16 = {k: cx_wedge.aggregate(v, self.pos["PEN"], self.pos["EPG"] if k != "GLNO" and k != "direct" else self.pos["PEN"], 16, bin16)
                for k, v in Kp.items()}
        return K, M16, Kp, Mp16

    def fan_in(self) -> dict:
        return {g: dict(total_min=float(self.tot[self.idx[g]].min()), total_max=float(self.tot[self.idx[g]].max()),
                        total_mean=float(self.tot[self.idx[g]].mean()), scale_min=float(self.scale[self.idx[g]].min()),
                        scale_max=float(self.scale[self.idx[g]].max()), n_below_1=int((self.scale[self.idx[g]] < 0.999).sum()))
                for g in GROUPS if len(self.idx[g])}


# ------------------------------------------------------------------------------------------------ rate model
def rate_fixed_point(A_sub, nE, inside, p: brain.LIFParams, background_hz=10.0, pulse_hz=40.0, sigma=SIGMA_MV, iters=4000,
                     alpha=0.05, drive="current"):
    """The threshold-linear fixed point of an explicit sub-circuit matrix under the cx_wedge protocol (background ->
    pulse -> release, each relaxed), with the LIF f-I smoothed over sigma mV of input noise.

    drive 'current' (6A fix 1, the default): r = max(f(tau A r + u_forced), r_forced), u_forced = f^-1(r_forced) on the
    EPG. The drive enters as the CURRENT that produces the forced rate, so a driven EPG that is not held below threshold
    by its own inputs has a real gain f'(u) -- 5A's defect. The forced rate stays a FLOOR because `FlyBrain.stimulate`
    forces spikes through `poisson_p` (brain.py: forced = rand < poisson_p, ORed with the threshold crossing), so a
    hyperpolarised EPG still fires at the driven rate; without the floor the same fixed point collapses the whole ring
    to 2.7 Hz (ring inhibition cancels the 6.63 mV drive), against a measured 9.5-14 Hz. Gains follow the same rule:
    gamma_i = f'(u_i) where the cell fires intrinsically, 0 where its rate is the forced floor.
    drive 'current-nofloor' is the literal reading (no floor), 'rate' is 5A's r = f(tau A r) + r_forced, which leaves
    the driven EPG at u -22 to -68 mV while 'firing' and so has gamma_EPG = 0 by construction. Both kept for
    comparison."""
    tau = p.tau_syn / 1000.0
    n = A_sub.shape[0]
    if drive not in ("current", "current-nofloor", "rate"):
        raise ValueError(f"drive must be 'current', 'current-nofloor' or 'rate', got {drive!r}")
    if drive.startswith("current"):
        floor_on = drive == "current"
        u_bg_drive, u_pulse_drive = u_for_rate(background_hz, p, sigma), u_for_rate(background_hz + pulse_hz, p, sigma)
        f_bg = np.zeros(n); f_bg[:nE] = u_bg_drive
        f_pulse = f_bg.copy(); f_pulse[:nE][inside] = u_pulse_drive
        r_floor_bg = np.zeros(n); r_floor_bg[:nE] = background_hz
        r_floor_pulse = r_floor_bg.copy(); r_floor_pulse[:nE][inside] = background_hz + pulse_hz
        floors = {id(f_bg): r_floor_bg, id(f_pulse): r_floor_pulse}
        r0 = np.maximum(cx_wedge.lif_fi(f_bg, p, sigma), r_floor_bg) if floor_on else cx_wedge.lif_fi(f_bg, p, sigma)

        def relax(fu, r):
            fl = floors[id(fu)] if floor_on else None
            u = tau * (A_sub @ r) + fu
            for _ in range(iters):
                u = tau * (A_sub @ r) + fu
                tgt = cx_wedge.lif_fi(u, p, sigma)
                if fl is not None:
                    tgt = np.maximum(tgt, fl)
                r = (1 - alpha) * r + alpha * tgt
            return r, u
    else:
        f_bg = np.zeros(n); f_bg[:nE] = background_hz
        f_pulse = f_bg.copy(); f_pulse[:nE][inside] += pulse_hz
        r0 = f_bg.copy()

        def relax(fr, r):
            u = tau * (A_sub @ r)
            for _ in range(iters):
                u = tau * (A_sub @ r)
                r = (1 - alpha) * r + alpha * (cx_wedge.lif_fi(u, p, sigma) + fr)
            return r, u

    r_bg, u_bg = relax(f_bg, r0)
    r_pulse, u_pulse = relax(f_pulse, r_bg)
    r_after, u_after = relax(f_bg, r_pulse)
    out = dict(background=(r_bg, u_bg), pulse=(r_pulse, u_pulse), after=(r_after, u_after))
    out["drive"] = drive
    cur = drive.startswith("current")
    out["forced_mV"] = dict(background=float(f_bg[0]) if cur else None,
                            pulse=float(f_pulse[:nE][inside][0]) if cur else None)
    out["floor"] = (dict(background=r_floor_bg, pulse=r_floor_pulse, after=r_floor_bg) if drive == "current" else None)
    return out


def jacobian_modes(A_sub, u, tau, epg_slice, wedge_of, p: brain.LIFParams, sigma=SIGMA_MV, n_report=6, floor=None):
    """The TRUE linearisation at a realised state (6A fix 3): J = diag(f'(u_i)) tau A, whose eigenvalues are loop gains
    (dimensionless; a mode grows when Re mu > 1). Returns the leading eigenvalue, the spectral radius, the per-group
    gains, and the leading mode whose EPG profile has ring index k -- 'the k = 1 gain' is lead['1']."""
    gamma = lif_fi_prime(u, p, sigma)
    if floor is not None:                    # a cell whose rate is the forced floor has no gain: its spikes are forced
        gamma = np.where(cx_wedge.lif_fi(u, p, sigma) > np.asarray(floor), gamma, 0.0)
    J = (gamma[:, None] * tau) * A_sub
    w, V = np.linalg.eig(J)
    order = np.argsort(-np.real(w))
    rows, lead = [], {}
    for i in order:                      # scan every mode so the leading k = 1 mode ('the k = 1 gain') is always found
        v = V[epg_slice, i]
        prof = np.array([v[wedge_of == k].mean() if (wedge_of == k).any() else 0.0 for k in range(16)])
        spec = np.abs(np.fft.fft(prof)) ** 2
        kk = int(np.argmax(spec)); kk = min(kk, 16 - kk)
        row = dict(re=float(np.real(w[i])), im=float(np.imag(w[i])), k=kk,
                   k_power=float(spec.max() / max(spec.sum(), 1e-12)),
                   epg_share=float(np.sum(np.abs(v) ** 2) / max(np.sum(np.abs(V[:, i]) ** 2), 1e-12)))
        if len(rows) < n_report:
            rows.append(row)
        lead.setdefault(str(kk), row)
    return dict(leading_re=float(np.real(w[order[0]])), spectral_radius=float(np.abs(w).max()),
                gamma=gamma, top=rows, leading_by_k=lead)


def summarise_state(r, u, groups_slices, inside, wedge_of, nE):
    e = r[:nE]
    d = dict(epg_in=float(e[inside].mean()), epg_out=float(e[~inside].mean()), epg_in_min=float(e[inside].min()),
             epg_out_max=float(e[~inside].max()), profile=[float(e[wedge_of == w].mean()) for w in range(16)],
             # 6B: the ledger's confinement count on the fixed point itself (bump_frames: >= 8 of 11 in, <= 3 of 35 out above 22 Hz)
             in_above_22=int((e[inside] > BUMP_THRESH_HZ).sum()), out_above_22=int((e[~inside] > BUMP_THRESH_HZ).sum()),
             n_in=int(inside.sum()), n_out=int((~inside).sum()))
    d["confined_by_ledger_rule"] = bool(d["in_above_22"] >= 8 and d["out_above_22"] <= 3)
    for g, sl in groups_slices.items():
        d[g] = float(r[sl].mean()) if sl.stop > sl.start else 0.0
        d[f"u_{g}"] = float(u[sl].mean()) if sl.stop > sl.start else 0.0
    # DC balance on the bump's own PEN / EPG and elsewhere
    return d


# ------------------------------------------------------------------------------------------------ analysis of one circuit
def analyse(cir: Circuit, log=print, drive="current", reduction="with-direct", gain="per-cell", sigma=SIGMA_MV) -> dict:
    """One configuration's structure pass. `sigma` (6B) is the input-noise width of the smoothed f-I used by the fixed
    point, the gains and the slope bound -- SIGMA_MV = 2.0 by default (the 5A / 6A assumption), or a MEASURED value
    (scripts/measure_lif_sigma.py); every number below that depends on it is labelled with it in `lif.sigma_mV`."""
    p = cir.p
    tau = p.tau_syn / 1000.0
    res = dict(label=cir.label, evidence=cir.evidence, lif=dict(conn_cap=p.conn_cap, same_type_gain=p.same_type_gain,
               receptor_model=p.receptor_model, receptor_net_rule=p.receptor_net_rule, w_syn=p.w_syn,
               input_norm_ref=p.input_norm_ref, input_norm_alpha=p.input_norm_alpha, sigma_mV=float(sigma),
               type_path_gain_extra=[list(x) for x in (p.type_path_gain or [])[len(brain.DEFAULT_TYPE_PATH_GAIN):]]),
               n_cells={g: int(len(cir.idx[g])) for g in GROUPS}, glno_nt=sorted(set(cir.c.neurons.nt.to_numpy()[cir.idx["GLNO"]].tolist())))
    res["one_step"] = cir.one_step()
    res["same_type"] = cir.same_type_blocks()
    res["fan_in"] = cir.fan_in()
    res["offsets"] = cir.offsets()
    K, M16, Kp, Mp16 = cir.two_step()
    d = cx_wedge.ring_dist(np.arange(16), np.arange(16))
    prof = {k: [float(M16[k][d == j].mean()) for j in range(9)] for k in M16}
    res["profile16"] = prof
    res["fourier"] = {k: fourier_modes(M16[k]) for k in M16}
    res["eig16"] = {k: eig_modes(M16[k]) for k in ("net", "net_tuned", "PEN", "Delta7", "Ring")}
    # Delta7 : ring-feedback ratio (three readings)
    fD, fR = res["fourier"]["Delta7"], res["fourier"]["Ring"]
    res["delta7_vs_ring"] = dict(k0_delta7=fD[0], k0_ring=fR[0], k0_ratio=float(fD[0] / fR[0]) if fR[0] else float("inf"),
                                 k1_delta7=fD[1], k1_ring=fR[1], k1_pen=res["fourier"]["PEN"][1],
                                 peak_delta7_opposite=prof["Delta7"][8], flat_ring_mean=float(np.mean(prof["Ring"])),
                                 ratio_peak=float(prof["Delta7"][8] / np.mean(prof["Ring"])) if np.mean(prof["Ring"]) else float("inf"),
                                 volley_delta7_to_epg=res["one_step"]["Delta7->EPG"]["total_per_post"],
                                 volley_ring_to_epg=res["one_step"]["Ring->EPG"]["total_per_post"])
    lam1 = res["fourier"]["net"][1]; lam0 = res["fourier"]["net"][0]
    d1 = res["fourier"]["direct"][1]; d0 = res["fourier"]["direct"][0]     # the ONE-STEP EPG -> EPG coefficients (mV)
    keep_direct = reduction == "with-direct"
    g1 = gamma_crit_combined(lam1, d1 if keep_direct else 0.0, tau)
    g0 = gamma_crit_combined(lam0, d0 if keep_direct else 0.0, tau)
    g1_direct_only = gamma_crit_combined(0.0, d1, tau)
    dmax, u_peak, f_peak = max_slope(p, sigma)
    res["two_step_gain"] = dict(
        lambda_k1_net=lam1, lambda_k0_net=lam0, lambda_k1_tuned=res["fourier"]["net_tuned"][1],
        direct_k1_mV=d1, direct_k0_mV=d0, reduction=reduction,
        # 6A fix 2: gamma tau d_k + gamma^2 tau^2 lambda_k = 1 (the one-step term kept); 5A's value is the d = 0 root
        gamma_crit_k1=g1, gamma_crit_k0=g0,
        gamma_crit_k1_two_step_only=float(1.0 / (tau * np.sqrt(lam1))) if lam1 > 0 else float("inf"),
        gamma_crit_k1_direct_only=g1_direct_only,
        loop_gain_k1_at_6=float(36.0 * tau * tau * lam1 + 6.0 * tau * (d1 if keep_direct else 0.0)),
        loop_gain_k0_at_6=float(36.0 * tau * tau * lam0 + 6.0 * tau * (d0 if keep_direct else 0.0)),
        # the rate at which a supercritical mode saturates (f'(u) back to gamma_crit); NaN = never supercritical
        saturation_hz_k1_uniform=rate_at_gain(g1, p, sigma), saturation_hz_k0_uniform=rate_at_gain(g0, p, sigma),
        saturation_hz_k1_direct_only=rate_at_gain(g1_direct_only, p, sigma),
        max_lif_slope=dmax, max_lif_slope_at_u=u_peak, max_lif_slope_at_hz=f_peak,
        supercritical_k1_uniform=bool(np.isfinite(g1) and g1 < dmax))
    # full sub-circuit linearisation
    A_sub, idx = cir.sub()
    nE = len(cir.idx["EPG"])
    rows, lead = circuit_modes(A_sub, slice(0, nE), cir.wedge_of, tau)
    res["circuit_modes"] = dict(top=rows, leading_by_k={str(k): v for k, v in lead.items()})
    # threshold-linear fixed point at the shipped gains, cx_wedge protocol (wedges 0-3)
    inside = np.isin(cir.wedge_of, [0, 1, 2, 3])
    sl = {}; start = 0
    for g in GROUPS:
        sl[g] = slice(start, start + len(cir.idx[g])); start += len(cir.idx[g])
    st = rate_fixed_point(A_sub, nE, inside, p, sigma=sigma, drive=drive)
    states = {k: v for k, v in st.items() if k in ("background", "pulse", "after")}
    res["rate_model"] = {k: summarise_state(r, u, sl, inside, cir.wedge_of, nE) for k, (r, u) in states.items()}
    res["rate_model"]["drive"] = drive
    res["rate_model"]["sigma_mV"] = float(sigma)
    res["rate_model"]["forced_mV"] = st["forced_mV"]
    a = res["rate_model"]["after"]
    res["rate_model"]["bump_after"] = bool(a["epg_in"] > 2 * a["epg_out"] and a["epg_in"] > 15.0)
    res["rate_model"]["runaway_after"] = bool(a["epg_out"] > 60.0)
    # 6B: a bump the ledger would score as confined (the 6A miss: H3's fixed point was a 'bump' with every off-block cell above 22 Hz)
    res["rate_model"]["confined_bump_after"] = bool(res["rate_model"]["bump_after"] and a["confined_by_ledger_rule"])
    # 6A fix 3: the true Jacobian diag(f'(u_i)) tau A at each realised state (a mode grows when Re mu > 1)
    jac = {}
    for k, (r, u) in states.items():
        jm = jacobian_modes(A_sub, u, tau, slice(0, nE), cir.wedge_of, p, sigma=sigma, floor=(st["floor"] or {}).get(k))
        gam = jm.pop("gamma")
        jm["gamma_by_group"] = {g: dict(mean=float(gam[sl[g]].mean()), max=float(gam[sl[g]].max())) for g in GROUPS if sl[g].stop > sl[g].start}
        jm["gamma_EPG_driven"] = float(gam[:nE][inside].mean())
        jm["loop_gain_k1"] = float(jm["leading_by_k"].get("1", {}).get("re", float("nan")))
        jm["loop_gain_k0"] = float(jm["leading_by_k"].get("0", {}).get("re", float("nan")))
        jac[k] = jm
    res["jacobian"] = jac
    # 6A: THE BUMP CRITERION the tool is validated on -- the EPG ring mode k = 1 with per-cell gains. An increment of
    # the driven EPG rate returns to the EPG as gamma_E tau d_1 (its own synapses, one step) + gamma_E gamma_relay
    # tau^2 lambda_1 (through PEN / PEG / Delta7, two steps), so the loop closes at
    #     gamma_E_crit = 1 / (tau d_1 + gamma_relay tau^2 lambda_1),
    # with gamma_relay READ OFF THE REALISED FIXED POINT (0 while the relays are below threshold -- the shipped case,
    # where the criterion reduces to the one-step term 5A dropped). A loop that closes grows until f'(u) has fallen
    # back to gamma_E_crit, which is `rate_at_gain`: the predicted bump rate. gamma_E_crit above the LIF's maximum
    # slope (8.3 Hz/mV at sigma 2 mV) is never reached at any operating point -> no bump.
    local = {k: float(prof[k][0] + 2 * prof[k][1]) for k in prof}     # the wedge-local band |post - pre| <= 1
    crit = {}
    for k in ("background", "pulse", "after"):
        gam = jac[k]["gamma_by_group"]
        g = {x: float(gam.get(x, {}).get("mean", 0.0)) for x in ("PEN", "PEG", "Delta7", "Ring")}
        # (i) the EPG's OWN recurrence, wedge-local and one-sided -- the criterion this tool is validated on
        den_epg = tau * (local["direct"] if keep_direct else 0.0)
        # (ii) the same with the relays adiabatic at their REALISED gains (0 while they are below threshold)
        den_all = den_epg + tau * tau * sum(g[x] * local[x] for x in ("PEN", "PEG", "Delta7", "Ring"))
        out = {}
        for name, den in (("epg_recurrent", den_epg), ("with_relays", den_all)):
            gc = float(1.0 / den) if den > 0 else float("inf")
            sat = rate_at_gain(gc, p, sigma)
            out[name] = dict(denominator_mV_per_Hz=den, gamma_EPG_crit=gc, saturation_hz=sat,
                             closes=bool(np.isfinite(sat)), predicted_bump=bool(np.isfinite(sat) and sat > BUMP_THRESH_HZ))
        # 'the k = 1 gain' with per-cell gains: the loop gain of ring mode k = 1 of the EPG-only reduction,
        # gamma_E (tau d_1 + tau^2 sum_X gamma_X lambda_1^X) -- a projection, always defined (the Jacobian's own
        # k = 1 CLASSIFICATION is nan whenever no eigenvector's EPG profile peaks at k = 1, e.g. when gamma_EPG = 0)
        lam1_by = {x: float(res["fourier"][x][1]) for x in ("PEN", "PEG", "Delta7", "Ring")}
        gE = float(jac[k]["gamma_EPG_driven"])
        k1_gain = gE * (tau * (d1 if keep_direct else 0.0) + tau * tau * sum(g[x] * lam1_by[x] for x in lam1_by))
        k1_gain_epg_only = gE * tau * (d1 if keep_direct else 0.0)
        crit[k] = dict(gamma_relay=g, gamma_EPG=gE, local_kernels=local, k1_gain=float(k1_gain),
                       k1_gain_epg_only=float(k1_gain_epg_only), lambda_1_by_relay=lam1_by,
                       jacobian_k1_classified=float(jac[k]["loop_gain_k1"]), max_lif_slope=dmax, **out)
    res["bump_criterion"] = crit
    res["local_kernels"] = local
    res["two_step_gain"]["gamma_crit_k1_recurrent"] = crit["pulse"]["epg_recurrent"]["gamma_EPG_crit"]
    res["two_step_gain"]["saturation_hz_k1"] = crit["pulse"]["epg_recurrent"]["saturation_hz"]
    res["two_step_gain"]["supercritical_k1"] = crit["pulse"]["epg_recurrent"]["closes"]
    res["two_step_gain"]["predicted_bump"] = crit["pulse"]["epg_recurrent"]["predicted_bump"]
    res["two_step_gain"]["gamma_crit_with_relays"] = crit["pulse"]["with_relays"]["gamma_EPG_crit"]
    res["two_step_gain"]["saturation_hz_with_relays"] = crit["pulse"]["with_relays"]["saturation_hz"]
    res["two_step_gain"]["predicted_bump_with_relays"] = crit["pulse"]["with_relays"]["predicted_bump"]
    # PEN DC margin during the pulse: mean input of the PENs whose glomerulus lies in the driven wedges
    pen_in = np.isin(np.asarray(np.round(cir.pos["PEN"]), int) % 16, [0, 1, 2, 3])
    for k, (r, u) in states.items():
        up = u[sl["PEN"]]
        res["rate_model"][k]["u_PEN_in"] = float(up[pen_in].mean()); res["rate_model"][k]["u_PEN_out"] = float(up[~pen_in].mean())
        res["rate_model"][k]["PEN_in"] = float(r[sl["PEN"]][pen_in].mean()); res["rate_model"][k]["PEN_out"] = float(r[sl["PEN"]][~pen_in].mean())
        res["rate_model"][k]["u_EPG_in"] = float(u[:nE][inside].mean()); res["rate_model"][k]["u_EPG_out"] = float(u[:nE][~inside].mean())
    # decomposition of the mean input of the driven-wedge PEN / EPG at each fixed point by presynaptic group (mV), and the
    # ring-neuron types that carry the Ring term (their fixed-point rates and their share)
    dec = {}
    rt = np.array(cir.cells["Ring"]["label"])
    for k, (r, u) in states.items():
        d = {}
        d["forced_mV"] = float(st["forced_mV"]["pulse" if k == "pulse" else "background"] or 0.0) if drive == "current" else 0.0
        for post, rows_ in (("PEN_in", pen_in), ("EPG_in", inside)):
            grp = "PEN" if post == "PEN_in" else "EPG"
            for g in GROUPS:
                Bg = cir.block(grp, g)[rows_]
                d[f"{post}<-{g}"] = float(tau * (Bg @ r[sl[g]]).mean()) if Bg.size else 0.0
            Bg = cir.block(grp, "Ring")[rows_]
            per_type = {}
            for t in sorted(set(rt)):
                m = rt == t
                v = float(tau * (Bg[:, m] @ r[sl["Ring"]][m]).mean())
                if abs(v) > 0.05:
                    per_type[t] = dict(mV=v, rate_hz=float(r[sl["Ring"]][m].mean()))
            d[f"{post}<-Ring_by_type"] = dict(sorted(per_type.items(), key=lambda kv: kv[1]["mV"]))
        dec[k] = d
    res["decomposition"] = dec
    B_pe = cir.block("PEN", "EPG")[pen_in][:, inside]
    res["pen_drive_per_hz"] = dict(direct_EPG=float(B_pe.sum(axis=1).mean() * tau))
    tg = res["two_step_gain"]; jp = res["jacobian"]["pulse"]; pu = res["rate_model"]["pulse"]
    log(f"[{cir.label}] k1 net {lam1:+.0f} (PEN {res['fourier']['PEN'][1]:+.0f}, D7 {fD[1]:+.0f}, Ring {fR[1]:+.0f}), direct k1 {d1:+.2f} mV; "
        f"gamma_crit(k1) {tg['gamma_crit_k1']:.2f} Hz/mV (two-step only {tg['gamma_crit_k1_two_step_only']:.2f}; max LIF slope {tg['max_lif_slope']:.1f}) "
        f"-> saturation {tg['saturation_hz_k1']:.0f} Hz; Jacobian at the pulse: lead {jp['leading_re']:+.3f}, k1 gain {jp['loop_gain_k1']:+.3f}, "
        f"gamma_EPG(driven) {jp['gamma_EPG_driven']:.2f}; pulse PEN {pu['PEN_in']:.1f} Hz u {pu['u_PEN_in']:+.2f} mV; after: in {a['epg_in']:.1f} "
        f"out {a['epg_out']:.1f} PEN {a['PEN']:.1f} D7 {a['Delta7']:.1f} Ring {a['Ring']:.2f} GLNO {a['GLNO']:.1f} "
        f"{'BUMP' if res['rate_model']['bump_after'] else ('RUNAWAY' if res['rate_model']['runaway_after'] else 'no bump')}"
        f" (in>22: {a['in_above_22']}/{a['n_in']}, out>22: {a['out_above_22']}/{a['n_out']}; sigma {sigma:g} mV)")
    return res, dict(K=K, M16=M16, Kp=Kp, Mp16=Mp16)


# ------------------------------------------------------------------------------------------------ the 6A hold configs
def hold_params(pre: str, post: str, base: brain.LIFParams | None = None, factor: float = 0.0) -> brain.LIFParams:
    """LIFParams with one `edges`-kind hold appended to type_path_gain -- the same stage and the same regex pair that
    `cx_wedge.py --hold-edges` installs, so the structure pass and the batch hold exactly the same edges."""
    import dataclasses
    base = base or brain.LIFParams()
    tpg = list(base.type_path_gain if base.type_path_gain is not None else brain.DEFAULT_TYPE_PATH_GAIN) + [(pre, post, float(factor))]
    return dataclasses.replace(base, type_path_gain=tpg)


def config_for_arm(arm, c, c_glu, cells, cells_glu):
    """(label, evidence, connectome, LIFParams, cells) for one arm of a batch table, read off its own command flags --
    so the structure pass and the batch are configured from ONE source. Returns None for an arm at gains other than
    1:1 (the rate model here is built at the shipped path gains; the R references are not predicted)."""
    label, desc, gains, glu, extra, cls = arm
    if gains != "1:1":
        return None
    kw = {}
    if "sign+gain" in extra:
        kw["receptor_model"] = "sign+gain"
        kw["receptor_net_rule"] = "abs"
    if "same_type_gain=1" in extra:
        kw["same_type_gain"] = 1.0
    p = brain.LIFParams(**kw)
    for i, x in enumerate(extra):
        if x == "--hold-edges":
            pre, post = extra[i + 1].split(":", 1)
            p = hold_params(pre, post, p)
        elif x == "--edge-gain":                          # 6B: a per-type gain through the same stage (cx_wedge.parse_edge_gains)
            pre, post, fac = cx_wedge.parse_edge_gains([extra[i + 1]])[0]
            p = hold_params(pre, post, p, factor=fac)
    return (label, f"{desc} [{cls}]", c_glu if glu else c, p, cells_glu if glu else cells)


def hold_configs(c, c_glu, cells, cells_glu) -> list:
    """The 6A counterfactual configurations: the DC inhibition the 5A decomposition names, held at 0 onto PEN and EPG --
    all three types together, each alone, and the three together with GLNO = glutamate. LABELLED COUNTERFACTUALS."""
    ev = ("LABELLED COUNTERFACTUAL (an `edges`-kind hold, docs/INTERP.md 10.1 step 5), not a candidate default: "
          "the DC term 5A's decomposition names, set to 0")
    out = [("H3: ExR6+ER6+ER4m -> PEN,EPG held 0", ev, c, hold_params(HOLD_PRE, HOLD_POST), cells)]
    for t in ("ExR6", "ER6", "ER4m"):
        out.append((f"H_{t}: {t} -> PEN,EPG held 0", ev + f" ({t} alone)", c, hold_params(rf"^{t}$", HOLD_POST), cells))
    out.append(("H3+GLNO=glu", ev + "; with the GLNO relabel (the correct-sign ring under the hold)", c_glu,
                hold_params(HOLD_PRE, HOLD_POST), cells_glu))
    return out


# ------------------------------------------------------------------------------------------------ the 6B configs
EPG_EPG_GAIN = r"^EPG$:^EPG$:10"      # x10 BEFORE the shipped same_type_gain 0.1 = exactly x1.0 on the 842 EPG -> EPG pairs


def recurrence_configs(c, c_glu, cells, cells_glu) -> list:
    """Thread 6B: the hold PLUS a wedge-local recurrence -- TWO labelled instruments at once (a counterfactual hold and
    a hand-rule removal or a per-type gain), which decide a MECHANISM question, never an adoption. H3F = H3 with the
    global same-type damping off; H3E = H3 with only the EPG -> EPG pairs undamped (`--edge-gain ^EPG$:^EPG$:10`,
    which composes with the x0.1 to x1.0 on exactly those 842 pairs); each with and without the GLNO relabel. E and EG
    (the per-type undamping WITHOUT the hold) are structural references only, not batch arms."""
    ev_h = "LABELLED COUNTERFACTUAL (the 6A hold: ExR6 + ER6 + ER4m -> PEN, EPG at 0)"
    pre, post, fac = cx_wedge.parse_edge_gains([EPG_EPG_GAIN])[0]
    P = brain.LIFParams
    out = [
        ("H3F: hold + same-type damping off", ev_h + " + LABELLED INSTRUMENT (global same_type_gain 1)", c,
         hold_params(HOLD_PRE, HOLD_POST, P(same_type_gain=1.0)), cells),
        ("H3E: hold + EPG->EPG undamped (per-type)", ev_h + " + LABELLED INSTRUMENT (per-type gain: only the 842 EPG -> EPG pairs at x1.0, every other same-type clique x0.1)", c,
         hold_params(pre, post, hold_params(HOLD_PRE, HOLD_POST), factor=fac), cells),
        ("H3FG: H3F + GLNO=glu", ev_h + " + LABELLED INSTRUMENT (global same_type_gain 1) + the GLNO relabel", c_glu,
         hold_params(HOLD_PRE, HOLD_POST, P(same_type_gain=1.0)), cells_glu),
        ("H3EG: H3E + GLNO=glu", ev_h + " + LABELLED INSTRUMENT (per-type EPG->EPG undamped) + the GLNO relabel", c_glu,
         hold_params(pre, post, hold_params(HOLD_PRE, HOLD_POST), factor=fac), cells_glu),
        ("E: EPG->EPG undamped (per-type), no hold", "LABELLED INSTRUMENT (per-type gain) without the hold -- a structural reference, not a batch arm", c,
         hold_params(pre, post, P(), factor=fac), cells),
    ]
    return out


# ------------------------------------------------------------------------------------------------ evidence blocks
def glno_evidence(c, c_glu, with_banc=True) -> dict:
    ty = c.neurons.type.fillna("").to_numpy()
    glno = np.flatnonzero(ty == "GLNO"); pen = np.flatnonzero(np.char.startswith(ty.astype(str), "PEN_"))
    W = c.W.tocsr(); coo = W.tocoo()
    cnt = connectome.sign0_counts(c, W=W, build=False)
    raw = np.abs(coo.data).astype(np.float64)
    if cnt is not None:
        raw = np.where(coo.data == 0, cnt, raw)
    g2p = np.isin(coo.row, pen) & np.isin(coo.col, glno)
    ev = dict(malecns=dict(cells=int(len(glno)), nt=sorted(set(c.neurons.nt.to_numpy()[glno].tolist())), sign=sorted(set(c.neurons.sign.to_numpy()[glno].tolist())),
                           glno_to_pen_edges=int(g2p.sum()), glno_to_pen_raw_syn=float(raw[g2p].sum()) if cnt is not None else None,
                           edge_min=float(raw[g2p].min()) if cnt is not None else None, edge_max=float(raw[g2p].max()) if cnt is not None else None,
                           edge_mean=float(raw[g2p].mean()) if cnt is not None else None, edges_above_cap_60=int((raw[g2p] > 60).sum()) if cnt is not None else None,
                           tbar_prediction="glutamate 51 % / acetylcholine 37 % over 783 T-bars, consensus 'unclear' conf 0.48 (docs/audits/nt_audit.md, cx_glno.md 1)",
                           hemibrain_name="GLNO = GLutamatergic LAL-NOduli neuron (the hemibrain / Hulse et al. 2021 naming)"),
              glu_cache=dict(nt=sorted(set(c_glu.neurons.nt.to_numpy()[np.flatnonzero(c_glu.neurons.type.fillna("").to_numpy() == "GLNO")].tolist())),
                             sum_abs_W=float(abs(c_glu.W).sum()), nnz=int(c_glu.W.nnz), cache_dir=str(getattr(c_glu, "cache_dir", None) or "")),
              shipped_cache=dict(sum_abs_W=float(abs(c.W).sum()), nnz=int(c.W.nnz)))
    if with_banc:
        try:
            b = connectome.load(dataset="banc", verbose=False)
            tb = b.neurons.type.fillna("").to_numpy(); m = tb == "GLNO"
            ev["banc"] = dict(cells=int(m.sum()), nt=b.neurons.nt[m].value_counts().to_dict(), release=str(b.release),
                              nt_verified=b.neurons.nt_verified[m].fillna("").astype(str).tolist() if "nt_verified" in b.neurons.columns else None)
        except Exception as e:  # noqa: BLE001
            ev["banc"] = dict(error=repr(e))
    return ev


def receptor_rows() -> dict:
    """The receptor-table rows of the ring types (fast / slow sign, gain class, tier, lead receptors)."""
    rt = connectome.read_receptor_table()
    keep = rt[rt.malecns_type.astype(str).str.match(r"^(EPG$|EPGt$|PEN_a|PEN_b|PEG$|Delta7$|GLNO$|ER[0-9]|ExR[0-9])") &
              rt.transmitter.isin(["acetylcholine", "gaba", "glutamate", "dopamine", "octopamine", "serotonin"])]
    cols = [c for c in ["malecns_type", "transmitter", "fast_sign", "fast_gain_class", "fast_pos_lead", "fast_neg_lead", "fast_sign_abs", "fast_gain_class_abs",
                        "slow_sign", "slow_gain_class", "slow_pos_lead", "slow_neg_lead", "tier", "source", "source_name"] if c in keep.columns]
    tab = keep[cols].copy()
    ring_types = sorted(set(tab.malecns_type))
    core = tab[tab.malecns_type.str.match(r"^(EPG$|EPGt$|PEN_a|PEN_b|PEG$|Delta7$|GLNO$)")]
    return dict(rows=core.to_dict("records"), n_ring_rows=int(len(tab)), ring_types_with_rows=[t for t in ring_types if t.startswith(("ER", "ExR"))],
                missing=[t for t in ("EPGt", "PEG", "GLNO", "ExR4", "ExR5", "ExR6", "ExR1", "ExR7", "ExR8") if t not in ring_types])


def receptor_stage_effects(c, cells) -> dict:
    """Entries on the ring core whose fast factor differs between receptor modes off / sign / sign+gain, by (post group,
    pre transmitter); and the monoamine slow matrices under 'full' restricted to EPG / PEN as post (mV per presynaptic
    Hz at steady state: w_syn x slow_gain x count x sign x gain x tau_slow)."""
    out = {}
    ty = c.neurons.type.fillna("").to_numpy()
    coo = c.W.tocoo()
    groups = {"EPG": cells["EPG"]["idx"], "PEN": cells["PEN"]["idx"], "PEG": cells["PEG"]["idx"], "Delta7": cells["Delta7"]["idx"], "Ring": cells["Ring"]["idx"]}
    nt = c.neurons.nt.fillna("unknown").to_numpy()
    for rule in ("abs", "class"):
        rs = connectome.receptor_signs(c, net_rule=rule, with_counts=True)
        f_sign = rs.fast_factor(None); f_gain = rs.fast_factor(brain.DEFAULT_RECEPTOR_GAIN)
        pre_sign = c.neurons.sign.to_numpy(np.float32)[coo.col]
        d = {}
        for g, idx in groups.items():
            sel = np.isin(coo.row, idx)
            chg_sign = sel & (f_sign != pre_sign)
            chg_gain = sel & (f_gain != f_sign) & (f_sign != 0)
            by = pd.DataFrame({"nt": nt[coo.col[sel]], "gain": f_gain[sel] / np.where(f_sign[sel] == 0, 1, f_sign[sel]), "syn": np.abs(coo.data[sel])})
            by = by[by.gain != 0].groupby("nt").agg(entries=("syn", "size"), syn=("syn", "sum"), gain_mean=("gain", "mean")).reset_index()
            d[g] = dict(entries=int(sel.sum()), sign_changed=int(chg_sign.sum()), gain_changed=int(chg_gain.sum()),
                        gain_by_pre_nt={r.nt: dict(entries=int(r.entries), syn=float(r.syn), factor=float(r.gain_mean)) for r in by.itertuples()})
        out[rule] = d
    # slow monoamine matrices under 'full' (shipped net rule)
    p_full = brain.LIFParams(receptor_model="full")
    spec = brain._slow_spec(p_full)
    rs = connectome.receptor_signs(c, net_rule=p_full.receptor_net_rule, with_counts=True)
    S = brain._slow_weights(c, p_full, rs, spec)
    slow = {}
    for cls, Sk in S.items():
        Sk = Sk.tocsr(); per_hz = p_full.w_syn * spec.gain[cls] * spec.tau[cls] / 1000.0    # mV per presynaptic Hz per synapse-equivalent
        for g in ("EPG", "PEN"):
            sub = Sk[groups[g]].tocoo()
            pre_ty = pd.Series(ty[sub.col]); pre_nt = pd.Series(nt[sub.col])
            df = pd.DataFrame({"pre_type": pre_ty, "pre_nt": pre_nt, "w": sub.data * per_hz, "post": sub.row})
            tot = df.groupby("post").w.sum()
            top = df.groupby(["pre_type", "pre_nt"]).agg(entries=("w", "size"), mv_per_hz_sum=("w", "sum")).reset_index().sort_values("mv_per_hz_sum", key=np.abs, ascending=False).head(8)
            slow[f"{cls}:{g}"] = dict(entries=int(sub.nnz), mv_per_hz_per_post_mean=float(tot.reindex(range(len(groups[g]))).fillna(0).mean()),
                                      mv_per_hz_per_post_max=float(np.abs(tot).max()) if len(tot) else 0.0,
                                      top_pre=[dict(pre_type=r.pre_type, pre_nt=r.pre_nt, entries=int(r.entries), mv_per_hz_sum_over_post=float(r.mv_per_hz_sum)) for r in top.itertuples()],
                                      per_hz_per_syn=per_hz, tau_ms=spec.tau[cls], gain=spec.gain[cls])
    out["slow_full"] = slow
    return out


def cap_effects(c, cells, p_cap: brain.LIFParams, p_nocap: brain.LIFParams) -> dict:
    """Capped vs uncapped effective weights of the ring's large edges (per-pair mean, per-post volley, share of pairs
    above the cap) and the fan-in scales they imply."""
    A1, s1, t1 = cx_wedge.effective_weights(c, p_cap); A0, s0, t0 = cx_wedge.effective_weights(c, p_nocap)
    ty = c.neurons.type.fillna("").to_numpy()
    idx = {g: cells[g]["idx"] for g in ("EPG", "PEN", "PEG", "Delta7", "Ring")}
    idx["GLNO"] = np.flatnonzero(ty == "GLNO")
    for t in ("ExR4", "ExR6", "ER4m", "ER6", "ExR5", "ER4d"):
        idx[t] = np.flatnonzero(ty == t)
    raw = c.W.tocsr()
    out = {}
    for name, (pre, post) in {"EPG->PEN": ("EPG", "PEN"), "PEN->EPG": ("PEN", "EPG"), "EPG->PEG": ("EPG", "PEG"), "PEG->EPG": ("PEG", "EPG"),
                              "EPG->Delta7": ("EPG", "Delta7"), "Delta7->EPG": ("Delta7", "EPG"), "Delta7->PEN": ("Delta7", "PEN"),
                              "GLNO->PEN": ("GLNO", "PEN"), "PEN->GLNO": ("PEN", "GLNO"), "ExR4->PEN": ("ExR4", "PEN"), "ExR6->PEN": ("ExR6", "PEN"),
                              "ExR6->EPG": ("ExR6", "EPG"), "ExR4->EPG": ("ExR4", "EPG"), "ER4m->EPG": ("ER4m", "EPG"), "ER6->EPG": ("ER6", "EPG"),
                              "ExR5->EPG": ("ExR5", "EPG"), "ER4d->EPG": ("ER4d", "EPG"), "ER6->PEN": ("ER6", "PEN"), "Ring->EPG": ("Ring", "EPG"), "Ring->PEN": ("Ring", "PEN")}.items():
        B1 = A1[idx[post]][:, idx[pre]].toarray().astype(np.float64); B0 = A0[idx[post]][:, idx[pre]].toarray().astype(np.float64)
        R = np.abs(raw[idx[post]][:, idx[pre]].toarray()); nz = R != 0
        out[name] = dict(pairs=int(nz.sum()), syn_min=float(R[nz].min()) if nz.any() else 0.0, syn_max=float(R[nz].max()) if nz.any() else 0.0,
                         syn_mean=float(R[nz].mean()) if nz.any() else 0.0, pairs_above_cap=int((R > p_cap.conn_cap).sum()),
                         capped_per_pair=float(B1[nz].mean()) if nz.any() else 0.0, uncapped_per_pair=float(B0[nz].mean()) if nz.any() else 0.0,
                         capped_volley=float(B1.sum(axis=1).mean()), uncapped_volley=float(B0.sum(axis=1).mean()))
    out["fan_in"] = {g: dict(scale_capped=[float(s1[idx[g]].min()), float(s1[idx[g]].max())], scale_uncapped=[float(s0[idx[g]].min()), float(s0[idx[g]].max())],
                             total_capped=[float(t1[idx[g]].min()), float(t1[idx[g]].max())], total_uncapped=[float(t0[idx[g]].min()), float(t0[idx[g]].max())])
                     for g in ("EPG", "PEN", "PEG", "Delta7", "GLNO")}
    return out


# ------------------------------------------------------------------------------------------------ markdown
def fmt_prof(v, w=7, d=0):
    return " ".join(f"{x:{w}.{d}f}" for x in v)


def write_md(res: dict, path: Path):
    L = []
    L.append("# cx_ring_structure -- the effective ring circuit under the shipped rules (CPU)\n")
    L.append(f"Generated {res['generated_utc']} by `{res['generator']}`; HEAD {res['git']['head']} ({'dirty' if res['git']['dirty'] else 'clean'}); "
             f"shipped cache sum|W| {res['glno_evidence']['shipped_cache']['sum_abs_W']:.0f}, GLNO=glutamate cache sum|W| {res['glno_evidence']['glu_cache']['sum_abs_W']:.0f}.\n")
    ev = res["glno_evidence"]
    L.append("## GLNO transmitter evidence\n")
    L.append(f"* MaleCNS: {ev['malecns']['cells']} cells, nt {ev['malecns']['nt']}, sign {ev['malecns']['sign']}; GLNO -> PEN {ev['malecns']['glno_to_pen_edges']} edges, "
             f"{ev['malecns']['glno_to_pen_raw_syn']:.0f} raw synapses, edges {ev['malecns']['edge_min']:.0f}-{ev['malecns']['edge_max']:.0f} (mean {ev['malecns']['edge_mean']:.0f}), "
             f"{ev['malecns']['edges_above_cap_60']} of {ev['malecns']['glno_to_pen_edges']} above conn_cap 60. T-bars: {ev['malecns']['tbar_prediction']}. Name: {ev['malecns']['hemibrain_name']}.")
    if "banc" in ev:
        L.append(f"* BANC ({ev['banc'].get('release')}): {ev['banc'].get('cells')} GLNO cells, nt {ev['banc'].get('nt')} (`connectome.load(dataset='banc')`, cache/banc).")
    L.append(f"* GLNO = glutamate scratch cache: nt {ev['glu_cache']['nt']}, nnz {ev['glu_cache']['nnz']}, sum|W| {ev['glu_cache']['sum_abs_W']:.0f}.\n")
    # receptor rows
    L.append("## Receptor-table rows of the ring core (fast sign / gain class; slow sign / class; tier / source)\n")
    L.append("| type | transmitter | fast sign | fast class | fast + lead | fast - lead | abs sign | abs class | slow sign | slow class | slow + lead | slow - lead | tier | source |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["receptor_rows"]["rows"]:
        L.append("| " + " | ".join(str(r.get(k, "")) for k in ("malecns_type", "transmitter", "fast_sign", "fast_gain_class", "fast_pos_lead", "fast_neg_lead", "fast_sign_abs",
                                                         "fast_gain_class_abs", "slow_sign", "slow_gain_class", "slow_pos_lead", "slow_neg_lead", "tier", "source")) + " |")
    L.append(f"\nTypes with no row (tier fallback = NT_SIGN of the presynaptic cell, gain class none = factor 1): {res['receptor_rows']['missing']}; "
             f"ER / ExR types with rows: {res['receptor_rows']['ring_types_with_rows']}.\n")
    # receptor stage effects
    rse = res["receptor_stage"]
    L.append("## What the receptor stage does on the ring core (entries whose fast factor differs from NT_SIGN; gain factors under sign+gain)\n")
    L.append("| net rule | post | entries | sign changed | gain changed | gain factor by presynaptic transmitter (entries, factor) |")
    L.append("|---|---|---|---|---|---|")
    for rule in ("abs", "class"):
        for g, d in rse[rule].items():
            gb = ", ".join(f"{k} ({v['entries']}, x{v['factor']:.2f})" for k, v in sorted(d["gain_by_pre_nt"].items(), key=lambda kv: -kv[1]["entries"])[:5])
            L.append(f"| {rule} | {g} | {d['entries']} | {d['sign_changed']} | {d['gain_changed']} | {gb} |")
    L.append("\nMonoamine slow class under `full` (shipped slow_gain 0.02, tau 200 ms; steady tone per presynaptic Hz = w_syn x slow_gain x count x sign x class x tau):\n")
    L.append("| class : post | entries | mV per Hz per post (mean) | max | top presynaptic types (sum over post, mV per Hz) |")
    L.append("|---|---|---|---|---|")
    for k, d in rse["slow_full"].items():
        top = "; ".join(f"{t['pre_type']} ({t['pre_nt']}, {t['entries']} e, {t['mv_per_hz_sum_over_post']:+.3f})" for t in d["top_pre"][:5])
        L.append(f"| {k} | {d['entries']} | {d['mv_per_hz_per_post_mean']:+.4f} | {d['mv_per_hz_per_post_max']:.4f} | {top} |")
    # cap effects
    ce = res["cap_effects"]
    L.append("\n## The connection cap on the ring's edges (shipped conn_cap 60 vs lifted; mV per presynaptic spike)\n")
    L.append("| projection | pairs | synapses min / mean / max | pairs above 60 | capped per pair | uncapped per pair | capped volley per post | uncapped volley |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, d in ce.items():
        if k == "fan_in":
            continue
        L.append(f"| {k} | {d['pairs']} | {d['syn_min']:.0f} / {d['syn_mean']:.0f} / {d['syn_max']:.0f} | {d['pairs_above_cap']} | {d['capped_per_pair']:+.2f} | {d['uncapped_per_pair']:+.2f} | {d['capped_volley']:+.1f} | {d['uncapped_volley']:+.1f} |")
    L.append("\nFan-in (input_norm_ref 5000, alpha 1): " + "; ".join(f"{g} total {v['total_capped'][0]:.0f}-{v['total_capped'][1]:.0f} scale {v['scale_capped'][0]:.2f}-{v['scale_capped'][1]:.2f} capped, "
                                                                f"total {v['total_uncapped'][0]:.0f}-{v['total_uncapped'][1]:.0f} scale {v['scale_uncapped'][0]:.2f}-{v['scale_uncapped'][1]:.2f} uncapped" for g, v in ce["fan_in"].items()) + ".\n")
    # per configuration
    cfgs = res["configs"]
    L.append("## Configurations: ring-mode gains and the threshold-linear fixed point at the shipped gains (gE 1 / gD 1)\n")
    L.append("lambda_k = ring-Fourier component k of the net EPG x EPG two-step matrix (PEN + PEG + Delta7 + Ring; mV^2 per spike); gamma_crit(k1) = 1 / (tau sqrt(lambda_1)) is the "
             "uniform cell gain (Hz/mV) at which the bump mode is self-sustaining in the two-step reduction; 'circuit k1' is the same from the full sub-circuit eigenvalue with a k=1 EPG profile. "
             "The fixed-point columns are the threshold-linear rate model after release (EPG in / out of wedges 0-3, PEN / Delta7 / Ring / GLNO, mean PEN input of the driven glomeruli in mV against the 7 mV gap).\n")
    L.append("| # | configuration | evidence | lambda_1 net | lambda_1 PEN / D7 / Ring / PEG / direct | lambda_0 net | gamma_crit k1 | circuit k1 mu (gamma_crit) | k0 mu | after: EPG in / out | PEN in / out | D7 | Ring | GLNO | u_PEN in (pulse / after) | bump |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(cfgs):
        f = r["fourier"]; g = r["two_step_gain"]; a = r["rate_model"]["after"]; pu = r["rate_model"]["pulse"]; lead = r["circuit_modes"]["leading_by_k"]
        k1 = lead.get("1", {}); k0 = lead.get("0", {})
        L.append(f"| {i} | {r['label']} | {r['evidence']} | {f['net'][1]:+.0f} | {f['PEN'][1]:+.0f} / {f['Delta7'][1]:+.0f} / {f['Ring'][1]:+.0f} / {f['PEG'][1]:+.0f} / {f['direct'][1]:+.1f} | {f['net'][0]:+.0f} | "
                 f"{g['gamma_crit_k1']:.2f} | {k1.get('re', float('nan')):+.1f} ({k1.get('gamma_crit', float('nan')):.2f}) | {k0.get('re', float('nan')):+.1f} | {a['epg_in']:.1f} / {a['epg_out']:.1f} | {a['PEN_in']:.1f} / {a['PEN_out']:.1f} | "
                 f"{a['Delta7']:.1f} | {a['Ring']:.2f} | {a['GLNO']:.1f} | {pu['u_PEN_in']:+.2f} / {a['u_PEN_in']:+.2f} | {'BUMP' if r['rate_model']['bump_after'] else 'no'} |")
    L.append("\n### Where the driven-wedge PEN's and EPG's input comes from at the fixed points (mV, mean over the cells of wedges 0-3; each presynaptic group at its fixed-point rate)\n")
    L.append("| configuration | state | PEN <- EPG | PEN <- PEN | PEN <- Delta7 | PEN <- Ring | PEN <- GLNO | PEN <- PEG | PEN total | Ring types onto PEN (mV @ Hz) | EPG <- PEN | EPG <- PEG | EPG <- Delta7 | EPG <- Ring | EPG <- EPG | EPG total | Ring types onto EPG |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cfgs:
        for k in ("background", "pulse"):
            d = r["decomposition"][k]
            tp = "; ".join(f"{t} {v['mV']:+.1f} @ {v['rate_hz']:.0f}" for t, v in list(d["PEN_in<-Ring_by_type"].items())[:4])
            te = "; ".join(f"{t} {v['mV']:+.1f} @ {v['rate_hz']:.0f}" for t, v in list(d["EPG_in<-Ring_by_type"].items())[:4])
            L.append(f"| {r['label']} | {k} | {d['PEN_in<-EPG']:+.1f} | {d['PEN_in<-PEN']:+.1f} | {d['PEN_in<-Delta7']:+.1f} | {d['PEN_in<-Ring']:+.1f} | {d['PEN_in<-GLNO']:+.1f} | {d['PEN_in<-PEG']:+.1f} | {r['rate_model'][k]['u_PEN_in']:+.1f} | {tp} | "
                     f"{d['EPG_in<-PEN']:+.1f} | {d['EPG_in<-PEG']:+.1f} | {d['EPG_in<-Delta7']:+.1f} | {d['EPG_in<-Ring']:+.1f} | {d['EPG_in<-EPG']:+.1f} | {r['rate_model'][k]['u_EPG_in']:+.1f} | {te} |")
    L.append("\n### Ring-distance profiles of the two-step matrices per configuration (16-wedge level, distance 0..8 wedges, mV^2 per spike)\n")
    for r in cfgs:
        L.append(f"**{r['label']}**\n")
        L.append("| path | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k in ("PEN", "PEG", "Delta7", "Ring", "direct", "net"):
            L.append(f"| {k} | " + " | ".join(f"{x:+.0f}" if k != "direct" else f"{x:+.2f}" for x in r["profile16"][k]) + " |")
        L.append("")
    L.append("### One-step volleys per configuration (mV per post cell if the whole presynaptic group fires once; pairs, mV per pair)\n")
    keys = ["EPG->PEN", "PEN->EPG", "EPG->PEG", "PEG->EPG", "EPG->Delta7", "Delta7->EPG", "Delta7->PEN", "Delta7->Delta7", "EPG->EPG", "PEN->PEN", "GLNO->PEN", "PEN->GLNO", "Ring->EPG", "Ring->PEN", "EPG->Ring"]
    L.append("| configuration | " + " | ".join(keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    for r in cfgs:
        L.append(f"| {r['label']} | " + " | ".join(f"{r['one_step'][k]['total_per_post']:+.1f} ({r['one_step'][k]['pairs']}, {r['one_step'][k]['mean_per_pair']:+.2f})" for k in keys) + " |")
    L.append("\n### Same-type edges (the x0.1 damping) per configuration\n")
    st_keys = ["EPG", "PEN_a", "PEN_b", "PEG", "Delta7", "PEN_a->PEN_b", "PEN_b->PEN_a"]
    L.append("| configuration | " + " | ".join(st_keys) + " |")
    L.append("|---|" + "---|" * len(st_keys))
    for r in cfgs:
        L.append(f"| {r['label']} | " + " | ".join(f"{r['same_type'][k]['total_per_post']:+.1f} ({r['same_type'][k]['pairs']}, {r['same_type'][k]['mean_per_pair']:+.2f})" for k in st_keys) + " |")
    # offsets for the shipped configuration
    r0 = cfgs[0]
    L.append("\n### Wedge-offset profiles (shipped): mean input to a post cell from pre cells at offset post - pre (wedges of 22.5 deg), mV per spike\n")
    offs = r0["offsets"]["EPG->PEN"]["offsets"]
    L.append("| projection | " + " | ".join(str(o) for o in offs) + " | centroid L / R |")
    L.append("|---|" + "---|" * (len(offs) + 1))
    for k in ("EPG->PEN", "PEN->EPG", "EPG->PEG", "PEG->EPG", "Delta7->EPG", "EPG->Delta7", "Delta7->PEN", "EPG->EPG"):
        d = r0["offsets"][k]
        cen = f"{d.get('centroid_L', float('nan')):+.2f} / {d.get('centroid_R', float('nan')):+.2f}" if "centroid_L" in d else ""
        L.append(f"| {k} | " + " | ".join(f"{x:+.1f}" for x in d["profile"]) + f" | {cen} |")
        for s in ("L", "R"):
            if f"profile_{s}" in d:
                L.append(f"| {k} ({s} PEN) | " + " | ".join(f"{x:+.1f}" for x in d[f"profile_{s}"]) + " |  |")
    L.append("\nRing / ExR types (shipped): EPG -> type -> EPG two-step total per EPG cell (mV^2) and onto PEN\n")
    L.append("| type | cells | nt | EPG->type per pair | type->EPG per pair | type->PEN per pair | two-step EPG | two-step PEN |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t, v in sorted(r0["offsets"]["Ring_types"].items(), key=lambda kv: kv[1]["two_step_EPG"]):
        L.append(f"| {t} | {v['cells']} | {v['nt']} | {v['from_EPG_per_pair']:+.2f} | {v['to_EPG_per_pair']:+.2f} | {v['to_PEN_per_pair']:+.2f} | {v['two_step_EPG']:+.0f} | {v['two_step_PEN']:+.0f} |")
    # ranking
    L.append("\n## Ranking of the single changes (by the fixed point's post-release bump, then gamma_crit(k1), then the PEN DC margin during the pulse)\n")
    L.append("| rank | configuration | bump after release | EPG in - out (Hz) | gamma_crit k1 (Hz/mV) | u_PEN in during pulse (mV) | lambda_1 net | lambda_0 net |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(res["ranking"]):
        L.append(f"| {i + 1} | {r['label']} | {'BUMP' if r['bump'] else 'no'} | {r['in_minus_out']:+.1f} | {r['gamma_crit_k1']:.2f} | {r['u_pen_pulse']:+.2f} | {r['lambda_1']:+.0f} | {r['lambda_0']:+.0f} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the batch (plan / analyse)
# Arms of the wedge-compass batch (docs/audits/compass_ring_mechanism.md section 3; out/cx5/predeclared.json). Every arm is
# scripts/cx_wedge.py --sim --ledger under the cx_wedge protocol (10 Hz Poisson background on the 46 EPG, 1 s settle,
# wedges 0-3 at +40 Hz for 2 s, 5 s free, compass adaptation 0); the label letters: G = GLNO = glutamate (scratch cache),
# C = receptor 'sign+gain' (net rule abs, the shipped rule), F = same_type_gain 1 (a global INSTRUMENT), R = the experiment
# gains gE 2 / gD 15 on Delta7 -> EPG only (the LABELLED reference), R175 = gE 1.75 / gD 15 (the point cx_glno abolished).
BATCH_ARMS = [
    ("S", "shipped: gE 1 / gD 1, LIFParams() (receptor sign/abs), GLNO sign 0", "1:1", False, ["--receptor-model", "shipped"], "the shipped path (reference)"),
    ("G", "GLNO = glutamate at the shipped gains", "1:1", True, ["--receptor-model", "shipped"], "data-implied relabel (MaleCNS T-bars 51 % glu; hemibrain name; BANC v888 4/4)"),
    ("C", "receptor sign+gain (abs) at the shipped gains", "1:1", False, ["--receptor-model", "sign+gain", "--receptor-net-rule", "abs"], "data-implied tier (expression tertiles) with a parameter factor map {0.5, 1, 1.5}; an existing opt-in mode"),
    ("CG", "sign+gain + GLNO = glutamate", "1:1", True, ["--receptor-model", "sign+gain", "--receptor-net-rule", "abs"], "C and G"),
    ("F", "same-type damping off (same_type_gain 1) at the shipped gains", "1:1", False, ["--receptor-model", "shipped", "--lif", "same_type_gain=1"], "INSTRUMENT: a global default (the connectome's same-type synapses at face value; the x0.1 is the hand rule)"),
    ("FG", "damping off + GLNO = glutamate", "1:1", True, ["--receptor-model", "shipped", "--lif", "same_type_gain=1"], "F and G"),
    ("CF", "sign+gain + damping off", "1:1", False, ["--receptor-model", "sign+gain", "--receptor-net-rule", "abs", "--lif", "same_type_gain=1"], "C and F (the structure pass's lowest gamma_crit)"),
    ("CFG", "sign+gain + damping off + GLNO = glutamate", "1:1", True, ["--receptor-model", "sign+gain", "--receptor-net-rule", "abs", "--lif", "same_type_gain=1"], "C, F and G"),
    ("R", "experiment gains gE 2 / gD 15 (Delta7 -> EPG only), GLNO silent", "2:15", False, ["--no-delta7-pen", "--receptor-model", "shipped"], "LABELLED reference: the bump that exists (cx_wedge / cx_glno)"),
    ("RG", "gE 2 / gD 15 + GLNO = glutamate", "2:15", True, ["--no-delta7-pen", "--receptor-model", "shipped"], "LABELLED reference: cx_glno's glu row at 2/15 (holds, -10 % rate)"),
    ("R175", "gE 1.75 / gD 15, GLNO silent", "1.75:15", False, ["--no-delta7-pen", "--receptor-model", "shipped"], "LABELLED reference: the silent point that held 3/6 in cx_glno"),
    ("RG175", "gE 1.75 / gD 15 + GLNO = glutamate", "1.75:15", True, ["--no-delta7-pen", "--receptor-model", "shipped"], "LABELLED reference: the point cx_glno's glutamatergic GLNO abolished 6/6"),
]
PRIMARIES = ("survival_s", "bump_hz_post", "width_half_post", "frac_confined_post")
SECONDARIES = ("epg_in_mean_post", "epg_out_mean_post", "PEN_mean_post", "Delta7_mean_post", "Ring_mean_post", "GLNO_mean_post", "rest_mean_post",
               "PEN_mean_during", "vs_post_all")

# ---- thread 6A (docs/audits/compass_dc_balance.md; out/cx6/predeclared.json). The cx5 protocol at the SHIPPED gains,
# with the DC term of 5A's decomposition held at 0 (an `edges`-kind LABELLED COUNTERFACTUAL, never a candidate default).
HOLD3 = r"^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)"
CX6_ARMS = [
    ("S", "shipped: gE 1 / gD 1, LIFParams() (receptor sign/abs), GLNO sign 0, no hold", "1:1", False,
     ["--receptor-model", "shipped"], "the shipped path (reference)"),
    ("H3", "ExR6 + ER6 + ER4m -> PEN, EPG held at 0", "1:1", False,
     ["--receptor-model", "shipped", "--hold-edges", HOLD3],
     "LABELLED COUNTERFACTUAL (edges hold): the DC term 5A's fixed point names (-24.9 mV on PEN during the pulse)"),
    ("H_ExR6", "ExR6 -> PEN, EPG held at 0", "1:1", False,
     ["--receptor-model", "shipped", "--hold-edges", r"^ExR6$:^(PEN_|EPG$)"],
     "LABELLED COUNTERFACTUAL: the largest single term (2 cells, -14.8 mV on PEN in the rate model)"),
    ("H_ER6", "ER6 -> PEN, EPG held at 0", "1:1", False,
     ["--receptor-model", "shipped", "--hold-edges", r"^ER6$:^(PEN_|EPG$)"],
     "LABELLED COUNTERFACTUAL: 4 cells, -9.8 mV on PEN in the rate model"),
    ("H_ER4m", "ER4m -> PEN, EPG held at 0", "1:1", False,
     ["--receptor-model", "shipped", "--hold-edges", r"^ER4m$:^(PEN_|EPG$)"],
     "LABELLED COUNTERFACTUAL: 11 cells, -12.6 mV on EPG in the rate model"),
    ("F", "same-type damping off (same_type_gain 1) at the shipped gains", "1:1", False,
     ["--receptor-model", "shipped", "--lif", "same_type_gain=1"],
     "LABELLED INSTRUMENT (global), carried over from cx5 for continuity: the only shipped-gain arm that held a bump"),
    ("R", "experiment gains gE 2 / gD 15 (Delta7 -> EPG only), GLNO silent", "2:15", False,
     ["--no-delta7-pen", "--receptor-model", "shipped"], "LABELLED reference (cx5): the bump that exists"),
    ("H3G", "H3 + GLNO = glutamate", "1:1", True,
     ["--receptor-model", "shipped", "--hold-edges", HOLD3],
     "LABELLED COUNTERFACTUAL + the data-implied relabel: the correct-sign ring under the hold"),
]
# 5 v 5 -> p_floor 0.0079, so a Holm family of m <= 6 is satisfiable (6 x 0.0079 = 0.047 <= 0.05).
PRIMARIES_CX6 = ("survival_s", "bump_hz_post", "width_half_post", "frac_confined_post", "PEN_mean_during", "PEN_mean_post")
SECONDARIES_CX6 = ("epg_in_mean_post", "epg_out_mean_post", "epg_in_mean_during", "epg_out_mean_during", "Delta7_mean_post",
                   "Ring_mean_post", "GLNO_mean_post", "rest_mean_post", "vs_post_all", "Ring_mean_during", "Delta7_mean_during")


# ---- thread 6B (docs/audits/compass_local_recurrence.md; out/cx7/predeclared.json). THE HOLD PLUS A WEDGE-LOCAL
# RECURRENCE: two labelled instruments at once, at the SHIPPED gains, deciding whether this ring can hold a bump AT THE
# DRIVEN TILE once the DC brake is off and the local recurrence is on -- a mechanism statement, never an adoption.
CX7_ARMS = [
    ("S", "shipped: gE 1 / gD 1, LIFParams() (receptor sign/abs), GLNO sign 0, no hold", "1:1", False,
     ["--receptor-model", "shipped"], "the shipped path (reference)"),
    ("H3", "ExR6 + ER6 + ER4m -> PEN, EPG held at 0 (6A)", "1:1", False,
     ["--receptor-model", "shipped", "--hold-edges", HOLD3],
     "LABELLED COUNTERFACTUAL (edges hold), the 6A arm: the second reference"),
    ("H3F", "H3 + same-type damping off (same_type_gain 1)", "1:1", False,
     ["--receptor-model", "shipped", "--hold-edges", HOLD3, "--lif", "same_type_gain=1"],
     "TWO INSTRUMENTS: the hold + the GLOBAL hand-rule removal (every same-type clique at its connectome weight)"),
    ("H3E", "H3 + only the EPG -> EPG pairs undamped (--edge-gain ^EPG$:^EPG$:10, x10 before the x0.1 = x1.0)", "1:1", False,
     ["--receptor-model", "shipped", "--hold-edges", HOLD3, "--edge-gain", EPG_EPG_GAIN],
     "TWO INSTRUMENTS: the hold + a PER-TYPE gain (842 EPG -> EPG pairs at x1.0; PEN_a, PEN_b, Delta7, PEG cliques stay x0.1)"),
    ("F", "same-type damping off (same_type_gain 1) at the shipped gains, no hold", "1:1", False,
     ["--receptor-model", "shipped", "--lif", "same_type_gain=1"],
     "LABELLED INSTRUMENT (global), carried from cx5 / cx6: the recurrence WITHOUT the hold (what the hold adds to F)"),
    ("H3G", "H3 + GLNO = glutamate (6A)", "1:1", True,
     ["--receptor-model", "shipped", "--hold-edges", HOLD3],
     "LABELLED COUNTERFACTUAL + the data-implied relabel, the 6A arm: the third reference"),
    ("H3FG", "H3F + GLNO = glutamate", "1:1", True,
     ["--receptor-model", "shipped", "--hold-edges", HOLD3, "--lif", "same_type_gain=1"],
     "TWO INSTRUMENTS + the relabel"),
    ("H3EG", "H3E + GLNO = glutamate", "1:1", True,
     ["--receptor-model", "shipped", "--hold-edges", HOLD3, "--edge-gain", EPG_EPG_GAIN],
     "TWO INSTRUMENTS (per-type) + the relabel"),
]
# The family that can move: 5 members vs each of the two references (S and H3), Holm within each family; 5 v 5 ->
# p_floor 0.0079, 5 x 0.0079 = 0.040 <= 0.05, satisfiable. bump_hz_post / width_half_post are undefined in both references
# (no confined frame in S or H3 in cx5 / cx6) and are PREDECLARED as magnitudes outside the family. centre_dist_t5 is the
# ring distance (wedges) of the EPG profile's vector centre at 5 s from the driven block's centre (1.5) -- the quantity
# 6A found decisive -- and is defined for every run.
PRIMARIES_CX7 = ("survival_s", "frac_confined_post", "centre_dist_t5", "PEN_mean_during", "PEN_mean_post")
MAGNITUDES_CX7 = ("bump_hz_post", "width_half_post")
SECONDARIES_CX7 = ("epg_in_mean_post", "epg_out_mean_post", "epg_in_mean_during", "epg_out_mean_during", "Delta7_mean_post",
                   "Ring_mean_post", "GLNO_mean_post", "rest_mean_post", "vs_post_all", "Ring_mean_during", "Delta7_mean_during",
                   "ExR6_mean_pre", "ExR6_mean_during", "ExR6_mean_post", "ER6_mean_pre", "ER6_mean_during", "ER6_mean_post",
                   "ER4m_mean_pre", "ER4m_mean_during", "ER4m_mean_post", "EPGt_mean_post", "PEG_mean_post", "epg_mean_pre",
                   "PEN_mean_pre", "GLNO_mean_pre", "Delta7_mean_pre", "frac_confined_during", "centre_dist_post_confined")
DRIVEN_CENTRE = 1.5           # wedges 0-3 -> centre 1.5
DRIVEN_TILE_TOL = 1.5         # the driven-tile rule: centre within 1.5 wedges of the driven block's centre


def ring_distance(a, b, n=16) -> float:
    d = abs(float(a) - float(b)) % n
    return float(min(d, n - d))


def arm_tables(arms):
    """(primaries, secondaries) for an arm table."""
    if arms is CX7_ARMS:
        return (PRIMARIES_CX7, SECONDARIES_CX7)
    return (PRIMARIES_CX6, SECONDARIES_CX6) if arms is CX6_ARMS else (PRIMARIES, SECONDARIES)


def sh_token(tok: str) -> str:
    """Single-quote a command token that carries shell metacharacters (the hold regexes carry $ ( ) |), so the line is
    safe both in batch.sh's double-quoted argument and in the remote shell that finally runs it."""
    return f"'{tok}'" if any(ch in tok for ch in "$()|*?&;<>\\\"` ") else tok


def write_tree_state(out_dir: Path):
    """<out_dir>/tree_state.json: HEAD, the diff against origin/main and the sha256 of every file that ships with the
    batch (cluster_run.py copies every file that differs from origin/main onto the run copy)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # tree state: git diff --stat and the sha256 of every file that differs from origin/main (cluster_run ships them all)
    status = git("status", "--porcelain").splitlines()
    files = sorted({ln[3:].strip() for ln in status if ln.strip()} | {"scripts/cx_wedge.py", "scripts/cx_ring_structure.py", "scripts/cx_glno.py",
                                                                        "scripts/probe_compass_room.py", "flyverse/brain.py", "flyverse/connectome.py",
                                                                        "flyverse/fly.py", "flyverse/interp/common.py", "flyverse/data/receptors_by_type.csv"})
    sha = {}
    for f in files:
        p = ROOT / f
        if p.is_file():
            sha[f] = sha256_file(p)
    tree = dict(stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), head=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                origin_main=git("rev-parse", "origin/main"), diff_stat=git("diff", "--stat").splitlines(), status_short=status, sha256=sha,
                note="cluster_run.py ships every file that differs from origin/main: the uncommitted edits above travel with the run (this task may not commit). "
                     "Another thread is concurrently editing scripts/probe_vnc_drive.py, flyverse/senses.py and docs/audits/level_controls_r2.md; "
                     "cx_wedge.py --sim builds FlyBrain without a world, so senses.py / body.py are not on the simulated path.")
    (out_dir / "tree_state.json").write_text(json.dumps(tree, indent=1), encoding="utf-8")
    print(f"tree_state.json ({len(sha)} files hashed) -> {out_dir / 'tree_state.json'}")
    return tree


def plan_batch(out_dir: Path, seeds=(0, 1, 2, 3), minutes=30, name="cx5", arms=None):
    """Writes <out_dir>/batch.sh (ONE cluster_run.py submission; one job per seed x GLNO condition, the arms of a job run
    sequentially, blocks fam_s<seed>), and tree_state.json."""
    arms_all = list(arms or BATCH_ARMS)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rel = out_dir.as_posix()
    jobs = []
    for s in seeds:
        for glu in (False, True):
            arms_j = [a for a in arms_all if a[3] == glu]
            if not arms_j:
                continue
            parts, sts = [], []
            for i, (label, _, gains, _, extra, _) in enumerate(arms_j):
                stem = f"{rel}/{label}_s{s}"
                cmd = (f"python scripts/cx_wedge.py --no-structure --sim {gains} --ledger --seed {s} --arm {label} --block fam_s{s} "
                       + ("--nt-override GLNO=glutamate " if glu else "") + " ".join(sh_token(x) for x in extra)
                       + f" --sim-out {stem}.json > {stem}.txt 2>&1; s{i}=\\$?; tail -3 {stem}.txt")
                parts.append(cmd); sts.append(f"s{i}")
            line = (f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
                    + "; ".join(parts) + f"; exit \\$(({' | '.join(sts)}))")     # bash arithmetic: bare names, no $ (a $s0 in the double-quoted line would expand at submission)
            jobs.append(dict(seed=s, glu=glu, arms=[a[0] for a in arms_j], line=line))
    call = (f"python scripts/cluster_run.py --name {name} --minutes {minutes} --arm-block fam " + " ".join('"' + j["line"] + '"' for j in jobs)
            + f" --fetch {rel}/")
    sh = (f"#!/bin/bash\n# ONE submission: {len(jobs)} jobs = {len(seeds)} seeds x 2 GLNO conditions; each job runs its arms sequentially; "
          f"blocks fam_s<seed> (every seed's arms on one target). Generated by scripts/cx_ring_structure.py --plan-batch on {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
          f"{call} 2>&1 | tee {rel}/client_stdout.txt\n")
    (out_dir / "batch.sh").write_text(sh, encoding="utf-8", newline="\n")
    tree = write_tree_state(out_dir)
    print(f"{len(jobs)} jobs -> {out_dir / 'batch.sh'}; tree_state.json ({len(tree['sha256'])} files hashed)")
    return jobs


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a family {key: p}."""
    items = [(k, p) for k, p in pvals.items() if p is not None and np.isfinite(p)]
    items.sort(key=lambda kv: kv[1])
    m = len(items); adj = {}; running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = float(min(1.0, running))
    for k, p in pvals.items():
        adj.setdefault(k, float("nan"))
    return adj


def analyse_batch(out_dir: Path, ref="S", arms=None, seeds=(0, 1, 2, 3)):
    """Tables, verdicts and the per-seed scatter from the fetched <out_dir>/<arm>_s<seed>.json rows (CPU)."""
    from flyverse.interp import common
    ARMS = list(arms or BATCH_ARMS)
    PRIM, SEC = arm_tables(arms or BATCH_ARMS)
    seeds = tuple(seeds)
    out_dir = Path(out_dir); an = out_dir / "analysis"; an.mkdir(parents=True, exist_ok=True)
    rows = []
    problems = []
    for label, desc, gains, glu, extra, cls in ARMS:
        gE, gD = (float(x) for x in gains.split(":"))
        for f in sorted(out_dir.glob(f"{label}_s*.json")):
            try:
                rr = json.load(open(f, encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                problems.append(f"{f.name}: unreadable ({e!r})"); continue
            for r in rr:
                m = r.get("metrics") or {}
                prov = r.get("provenance") or {}
                ex = prov.get("execution") or {}
                cc = prov.get("compiled_connectome") or {}
                d = dict(arm=label, seed=r["seed"], file=f.name, gE=r["gE"], gD=r["gD"], device=ex.get("device"), device_name=ex.get("device_name"),
                         host=ex.get("host"), md5=cc.get("md5"), nt_override=json.dumps(r.get("nt_override") or {}), lif_overrides=json.dumps(r.get("lif_overrides") or {}),
                         receptor=f"{r.get('receptor_model')}/{r.get('receptor_net_rule')}", delta7_pen=r.get("delta7_pen"), wall_s=r.get("wall_s"),
                         ledger_survival=(r.get("ledger") or {}).get("compass.EPG.bump_survival_s", {}).get("status"),
                         ledger_rate=(r.get("ledger") or {}).get("compass.EPG.bump_rate_hz", {}).get("status"),
                         ledger_width=(r.get("ledger") or {}).get("compass.EPG.bump_width_wedges", {}).get("status"),
                         hold_edges=json.dumps(r.get("hold_edges") or []),
                         hold_entries=int(sum(h.get("n_entries", 0) for h in (r.get("hold_edges_resolved") or []))),
                         hold_pre_cells=int(sum(h.get("n_pre_cells", 0) for h in (r.get("hold_edges_resolved") or []))),
                         hold_post_cells=int(sum(h.get("n_post_cells", 0) for h in (r.get("hold_edges_resolved") or []))),
                         profile_post=r.get("wedge_profile_post"))
                for k in PRIM + SEC:
                    d[k] = m.get(k)
                # checks: device / arm / gains / override
                if str(d["device"]) != "cuda":
                    problems.append(f"{f.name}: device {d['device']} (expected cuda)")
                if r.get("arm") != label:
                    problems.append(f"{f.name}: arm {r.get('arm')} != {label}")
                if abs(float(r["gE"]) - gE) > 1e-9 or abs(float(r["gD"]) - gD) > 1e-9:
                    problems.append(f"{f.name}: gains {r['gE']}:{r['gD']} != {gains}")
                if bool((r.get("nt_override") or {}).get("GLNO") == "glutamate") != glu:
                    problems.append(f"{f.name}: nt_override {r.get('nt_override')} does not match arm {label}")
                if sorted(set(r.get("glno_nt") or [])) != (["glutamate"] if glu else ["unknown"]):
                    problems.append(f"{f.name}: GLNO nt {r.get('glno_nt')} does not match arm {label}")
                want_same = 1.0 if "--lif" in extra and "same_type_gain=1" in extra else 0.1
                if abs(float((prov.get("model") or {}).get("lif", {}).get("same_type_gain", want_same)) - want_same) > 1e-9:
                    problems.append(f"{f.name}: same_type_gain {(prov.get('model') or {}).get('lif', {}).get('same_type_gain')} != {want_same}")
                want_rm = "sign+gain" if "sign+gain" in extra else brain.LIFParams().receptor_model
                if r.get("receptor_model") != want_rm:
                    problems.append(f"{f.name}: receptor_model {r.get('receptor_model')} != {want_rm}")
                want_hold = [extra[i + 1] for i, x in enumerate(extra) if x == "--hold-edges"]
                got_hold = [h[0] + ":" + h[1] for h in (r.get("hold_edges") or [])]
                if got_hold != want_hold:
                    problems.append(f"{f.name}: hold_edges {got_hold} != {want_hold} (arm {label})")
                if want_hold and not all(abs(float(h[2])) < 1e-12 for h in (r.get("hold_edges") or [])):
                    problems.append(f"{f.name}: hold factor is not 0: {r.get('hold_edges')}")
                if want_hold and d["hold_entries"] == 0:
                    problems.append(f"{f.name}: hold {want_hold} matched 0 entries")
                rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows"); return None
    df = df.sort_values(["arm", "seed"], key=lambda s: s.map({a[0]: i for i, a in enumerate(ARMS)}) if s.name == "arm" else s).reset_index(drop=True)
    counts = df.groupby("arm").seed.size()
    expected = {a[0] for a in ARMS}
    for a in expected - set(counts.index):
        problems.append(f"arm {a}: 0 runs")
    md5s = df.groupby("arm").md5.agg(lambda s: sorted(set(s)))
    # verdicts vs the reference per arm, primaries with Holm within the arm's family
    ref_df = df[df.arm == ref]
    comp = []
    for label, desc, gains, glu, extra, cls in ARMS:
        if label == ref:
            continue
        sub = df[df.arm == label]
        if sub.empty:
            continue
        pv = {}
        res_k = {}
        for k in PRIM:
            a_ = [x for x in sub[k].tolist() if x is not None and np.isfinite(x)]
            b_ = [x for x in ref_df[k].tolist() if x is not None and np.isfinite(x)]
            if len(a_) == 0 or len(b_) == 0:
                res_k[k] = dict(verdict="no data" if len(a_) == 0 and len(b_) == 0 else "one-sided (reference has no confined frames)", diff=float("nan"), z=float("nan"), p=float("nan"),
                                n_stim=len(a_), n_null=len(b_), null_sd_zero=None,
                                stim_values=[round(x, 4) for x in a_], null_values=[round(x, 4) for x in b_])
                continue
            c_ = common.compare(a_, b_)
            res_k[k] = dict(verdict=c_["verdict"], diff=c_["diff"], z=c_["z"], p=c_["p"], p_floor=c_["p_floor"], null_sd_zero=c_["null_sd_zero"],
                            n_stim=len(a_), n_null=len(b_), stim_values=[round(x, 4) for x in a_], null_values=[round(x, 4) for x in b_])
            pv[k] = c_["p"]
        adj = holm(pv)
        for k in PRIM:
            res_k[k]["p_holm"] = adj.get(k, float("nan"))
            comp.append(dict(arm=label, key=k, **res_k[k]))
        for k in SEC:
            a_ = [x for x in sub[k].tolist() if x is not None and np.isfinite(x)]
            b_ = [x for x in ref_df[k].tolist() if x is not None and np.isfinite(x)]
            if a_ and b_:
                c_ = common.compare(a_, b_)
                comp.append(dict(arm=label, key=k, verdict=c_["verdict"], diff=c_["diff"], z=c_["z"], p=c_["p"], p_floor=c_["p_floor"], null_sd_zero=c_["null_sd_zero"],
                                 n_stim=len(a_), n_null=len(b_), stim_values=[round(x, 4) for x in a_], null_values=[round(x, 4) for x in b_], p_holm=float("nan")))
    cdf = pd.DataFrame(comp)
    # the predeclared decision rule per arm
    rule = []
    for label, desc, gains, glu, extra, cls in ARMS:
        sub = df[df.arm == label]
        ok = [(bool(s >= 5.0) and bool(np.isfinite(w) and 2.5 <= w <= 5.0) and bool(np.isfinite(h) and 5.0 <= h <= 60.0))
              for s, w, h in zip(sub.survival_s.fillna(0), sub.width_half_post.astype(float), sub.bump_hz_post.astype(float))]
        conf = [bool(s >= 5.0) for s in sub.survival_s.fillna(0)]
        rule.append(dict(arm=label, gains=gains, runs=int(len(sub)), working_compass_seeds=int(sum(ok)), bump_survives_seeds=int(sum(conf)),
                         working=bool(sum(ok) >= 3 and len(sub) >= 4), survival=sub.survival_s.tolist(), rate=sub.bump_hz_post.tolist(), width=sub.width_half_post.tolist(),
                         frac_confined_post=sub.frac_confined_post.tolist(), devices=sorted(set(map(str, sub.device_name))), md5=md5s.get(label, [])))
    rdf = pd.DataFrame(rule)
    # the 6A three-way call, in the predeclared words (out/cx6/predeclared.json): applied to every arm, read for H3
    if arms is CX6_ARMS:
        calls = []
        for label, desc, gains, glu, extra, cls in ARMS:
            sub = df[df.arm == label]
            if sub.empty:
                continue
            r_ = rdf[rdf.arm == label].iloc[0]
            pen_d = sub.PEN_mean_during.astype(float)
            call = ("the DC balance is the whole story" if int(r_.working_compass_seeds) >= 3 else
                    ("not the story (PEN stays below 1 Hz)" if float(pen_d.max()) < 1.0 else "necessary but not sufficient (PEN fires, no working bump)"))
            calls.append(dict(arm=label, gains=gains, runs=int(len(sub)), working_compass_seeds=int(r_.working_compass_seeds),
                              bump_survives_seeds=int(r_.bump_survives_seeds), pen_during_min=float(pen_d.min()), pen_during_max=float(pen_d.max()),
                              pen_post_max=float(sub.PEN_mean_post.astype(float).max()), call=call))
        pd.DataFrame(calls).to_csv(an / "call.csv", index=False)
    # the per-run STATE table (what the ring is doing before / during / after the pulse, and WHERE the bump is), emitted
    # to a named file so the audit pastes rather than retypes (docs/INTERP.md 10.4 rule 28)
    strows = []
    for label, desc, gains, glu, extra, cls in ARMS:
        for f in sorted(out_dir.glob(f"{label}_s*.json")):
            for r in json.load(open(f, encoding="utf-8")):
                m = r.get("metrics") or {}
                strows.append(dict(arm=label, seed=r["seed"],
                                   **{k: m.get(k) for k in ("epg_mean_pre", "PEN_mean_pre", "Delta7_mean_pre", "GLNO_mean_pre",
                                                            "Ring_mean_pre", "rest_mean_pre", "frac_confined_pre", "frac_confined_during",
                                                            "frac_confined_post", "epg_in_mean_during", "epg_out_mean_during",
                                                            "epg_in_mean_post", "epg_out_mean_post", "epg_max_post",
                                                            "in_above_end", "out_above_end", "PEN_mean_during", "PEN_mean_post",
                                                            "Delta7_mean_post", "GLNO_mean_post", "Ring_mean_post", "rest_mean_post")},
                                   centre_wedge_t5=r.get("t5.0_centre_wedge"), vs_t5=r.get("t5.0_vector_strength"),
                                   in_above_t5=r.get("t5.0_in_above"), out_above_t5=r.get("t5.0_out_above"),
                                   profile_end=";".join(f"{x:.0f}" for x in (r.get("wedge_profile_end") or []))))
    pd.DataFrame(strows).to_csv(an / "state.csv", index=False)
    # per-seed scatter, emitted to a named file (docs/INTERP.md 10.4 rule 28: prose pastes from this file, never retypes)
    sc = []
    for label, desc, gains, glu, extra, cls in ARMS:
        sub = df[df.arm == label].sort_values("seed")
        if sub.empty:
            continue
        for k in PRIM:
            sc.append(dict(arm=label, key=k, seeds=",".join(str(int(x)) for x in sub.seed),
                           values=",".join("nan" if not np.isfinite(float(v)) else f"{float(v):.4f}" for v in sub[k].astype(float)),
                           mean=float(np.nanmean(sub[k].astype(float))) if np.isfinite(sub[k].astype(float)).any() else float("nan"),
                           sd=float(np.nanstd(sub[k].astype(float), ddof=1)) if np.isfinite(sub[k].astype(float)).sum() > 1 else float("nan")))
    pd.DataFrame(sc).to_csv(an / "scatter.csv", index=False)
    # console-vs-json device check: every .txt should say the device
    txt_missing = [a[0] + f"_s{s}" for a in ARMS for s in seeds if not (out_dir / f"{a[0]}_s{s}.txt").exists()]
    # write
    df.drop(columns=["profile_post"]).to_csv(an / "runs.csv", index=False)
    cdf.to_csv(an / "compare.csv", index=False)
    rdf.to_csv(an / "decision.csv", index=False)
    json.dump(dict(runs=df.drop(columns=["profile_post"]).to_dict("records"), compare=comp, decision=rule, problems=problems, txt_missing=txt_missing,
                   n_runs=int(len(df)), n_expected=len(ARMS) * len(seeds), generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   generator="python " + " ".join(sys.argv), analysis_sha256=sha256_file(Path(__file__)),
                   cx_wedge_sha256=sha256_file(ROOT / "scripts" / "cx_wedge.py"), common_sha256=sha256_file(ROOT / "flyverse" / "interp" / "common.py")),
              open(an / "analysis.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    # markdown
    L = [f"# batch analysis -- n_runs {len(df)} of {len(ARMS) * len(seeds)} expected ({out_dir.as_posix()}/<arm>_s<seed>.json); generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"]
    L.append(f"Problems ({len(problems)}): " + ("; ".join(problems) if problems else "none") + (f"; console .txt missing: {txt_missing}" if txt_missing else "") + "\n")
    L.append("## Per run\n")
    cols = ["arm", "seed", "gE", "gD", "survival_s", "bump_hz_post", "width_half_post", "frac_confined_post", "epg_in_mean_post", "epg_out_mean_post", "PEN_mean_during",
            "PEN_mean_post", "Delta7_mean_post", "Ring_mean_post", "GLNO_mean_post", "rest_mean_post", "vs_post_all", "ledger_survival", "ledger_rate", "ledger_width",
            "device_name", "md5", "wall_s"]
    L.append("| " + " | ".join(cols) + " |"); L.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        L.append("| " + " | ".join((f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    L.append("\n## Decision rule per arm ('a working compass' = survival >= 5 s AND width 2.5-5 wedges AND rate 5-60 Hz in >= 3 of 4 seeds)\n")
    L.append("| arm | gains | runs | seeds working | seeds with a surviving bump | working | survival per seed | rate per seed | width per seed | frac confined post | devices | compiled W md5 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in rdf.iterrows():
        fmtl = lambda v: "/".join("nan" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.3g}" for x in v)   # noqa: E731
        L.append(f"| {r.arm} | {r.gains} | {r.runs} | {r.working_compass_seeds} | {r.bump_survives_seeds} | {'YES' if r.working else 'no'} | {fmtl(r.survival)} | {fmtl(r.rate)} | {fmtl(r.width)} | {fmtl(r.frac_confined_post)} | {r.devices} | {r.md5} |")
    L.append(f"\n## Verdicts vs {ref} (common.compare, runs = the unit; Holm within each arm's four primaries)\n")
    L.append("| arm | key | verdict | diff | z | p | p Holm | null SD zero | stim values | null values |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in cdf.iterrows():
        L.append(f"| {r.arm} | {r.key} | {r.verdict} | {r['diff']:+.3g} | {r.z:+.2f} | {r.p:.3g} | {r.p_holm:.3g} | {r.null_sd_zero} | {r.stim_values} | {r.null_values} |")
    (an / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    # scatter figure: per-seed points of the four primaries per arm
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        order = [a[0] for a in ARMS]
        for ax, k in zip(axes, PRIM[:4]):
            for i, arm in enumerate(order):
                v = df[df.arm == arm][k].astype(float).to_numpy()
                v = np.where(np.isfinite(v), v, np.nan)
                ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=18)
                if np.isfinite(v).any():
                    ax.plot([i - 0.25, i + 0.25], [np.nanmean(v)] * 2, color="k", lw=1)
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=60, fontsize=7); ax.set_title(k, fontsize=9)
        fig.tight_layout(); fig.savefig(an / "scatter.png", dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"scatter figure skipped: {e!r}")
    print(f"n_runs {len(df)} ({out_dir.as_posix()}/<arm>_s<seed>.json); problems {len(problems)}")
    for p_ in problems:
        print("  PROBLEM", p_)
    print(rdf[["arm", "gains", "runs", "working_compass_seeds", "bump_survives_seeds", "working"]].to_string(index=False))
    print(f"-> {an / 'analysis.md'}, runs.csv, compare.csv, decision.csv, scatter.csv, analysis.json, scatter.png"
          + ", state.csv" + (", call.csv" if arms is CX6_ARMS else ""))
    return df, cdf, rdf


# ------------------------------------------------------------------------------------------------ thread 6B analysis
def _run_rows_cx7(out_dir: Path, arms, problems: list) -> list:
    """One row per run of a cx7-table batch: the recorded metrics, the per-type ring rates, the per-cell maxima, the
    driven-tile distance at 5 s and (from the .npz) over the confined post-pulse frames, plus the provenance checks."""
    import probe_compass_room as pcr
    rows = []
    for label, desc, gains, glu, extra, cls in arms:
        gE, gD = (float(x) for x in gains.split(":"))
        for f in sorted(out_dir.glob(f"{label}_s*.json")):
            try:
                rr = json.load(open(f, encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                problems.append(f"{f.name}: unreadable ({e!r})"); continue
            for r in rr:
                m = r.get("metrics") or {}
                prov = r.get("provenance") or {}
                ex = prov.get("execution") or {}
                cc = prov.get("compiled_connectome") or {}
                lif = (prov.get("model") or {}).get("lif", {})
                d = dict(arm=label, seed=r["seed"], file=f.name, gE=r["gE"], gD=r["gD"], device=ex.get("device"), device_name=ex.get("device_name"),
                         host=ex.get("host"), md5=cc.get("md5"), nt_override=json.dumps(r.get("nt_override") or {}), lif_overrides=json.dumps(r.get("lif_overrides") or {}),
                         same_type_gain=lif.get("same_type_gain"), receptor=f"{r.get('receptor_model')}/{r.get('receptor_net_rule')}", wall_s=r.get("wall_s"),
                         ledger_survival=(r.get("ledger") or {}).get("compass.EPG.bump_survival_s", {}).get("status"),
                         ledger_rate=(r.get("ledger") or {}).get("compass.EPG.bump_rate_hz", {}).get("status"),
                         ledger_width=(r.get("ledger") or {}).get("compass.EPG.bump_width_wedges", {}).get("status"),
                         hold_edges=json.dumps(r.get("hold_edges") or []), edge_gains=json.dumps(r.get("edge_gains") or []),
                         hold_entries=int(sum(h.get("n_entries", 0) for h in (r.get("hold_edges_resolved") or []))),
                         gain_entries=int(sum(h.get("n_entries", 0) for h in (r.get("edge_gains_resolved") or []))),
                         gain_entries_same_type=int(sum(h.get("n_entries_same_type", 0) for h in (r.get("edge_gains_resolved") or []))),
                         gain_effective_same_type=(r.get("edge_gains_resolved") or [{}])[0].get("effective_factor_same_type") if r.get("edge_gains_resolved") else None,
                         centre_wedge_t5=r.get("t5.0_centre_wedge"), vs_t5=r.get("t5.0_vector_strength"),
                         in_above_t5=r.get("t5.0_in_above"), out_above_t5=r.get("t5.0_out_above"),
                         profile_end=";".join(f"{x:.0f}" for x in (r.get("wedge_profile_end") or [])))
                d["centre_dist_t5"] = ring_distance(r["t5.0_centre_wedge"], DRIVEN_CENTRE) if r.get("t5.0_centre_wedge") is not None else float("nan")
                for k in set(PRIMARIES_CX7 + MAGNITUDES_CX7 + SECONDARIES_CX7) - {"centre_dist_t5", "centre_dist_post_confined"}:
                    d[k] = m.get(k)
                for g in ("PEN", "Delta7", "PEG", "GLNO", "EPGt", "ExR6", "ER6", "ER4m"):
                    for w in ("pre", "during", "post"):
                        d[f"{g}_cell_max_{w}"] = m.get(f"{g}_cell_max_{w}")
                # from the per-frame record: where the confined bump sits, and how long it is confined AT THE DRIVEN TILE
                npz = out_dir / f"{label}_s{r['seed']}_gE{r['gE']:g}_gD{r['gD']:g}_s{r['seed']}.npz"
                d["centre_dist_post_confined"] = float("nan"); d["survival_at_tile_s"] = float("nan"); d["frac_at_tile_post"] = float("nan")
                if npz.is_file():
                    z = np.load(npz)
                    b = pcr.bump_frames(z["epg"], z["wedge_of"])
                    tt = z["t"]; t_end = tt + 0.01
                    sp = ((prov.get("stimulus") or {}).get("params") or {})
                    t_rel = float(sp.get("settle_s", 1.0)) + float(sp.get("pulse_s", 2.0))      # the pulse end, from the run's own record
                    post = tt >= t_rel - 1e-9
                    dist = np.array([ring_distance(cw_, DRIVEN_CENTRE) for cw_ in b["centre"]])
                    conf = b["confined"]
                    at_tile = conf & (dist <= DRIVEN_TILE_TOL)
                    d["centre_dist_post_confined"] = float(dist[post & conf].mean()) if (post & conf).any() else float("nan")
                    last = np.flatnonzero(at_tile & post)
                    d["survival_at_tile_s"] = float(t_end[last[-1]] - t_rel) if len(last) else 0.0
                    d["frac_at_tile_post"] = float(at_tile[post].mean())
                else:
                    problems.append(f"{f.name}: per-frame record {npz.name} missing")
                # checks
                if str(d["device"]) != "cuda":
                    problems.append(f"{f.name}: device {d['device']} (expected cuda)")
                if r.get("arm") != label:
                    problems.append(f"{f.name}: arm {r.get('arm')} != {label}")
                if abs(float(r["gE"]) - gE) > 1e-9 or abs(float(r["gD"]) - gD) > 1e-9:
                    problems.append(f"{f.name}: gains {r['gE']}:{r['gD']} != {gains}")
                if bool((r.get("nt_override") or {}).get("GLNO") == "glutamate") != glu:
                    problems.append(f"{f.name}: nt_override {r.get('nt_override')} does not match arm {label}")
                if sorted(set(r.get("glno_nt") or [])) != (["glutamate"] if glu else ["unknown"]):
                    problems.append(f"{f.name}: GLNO nt {r.get('glno_nt')} does not match arm {label}")
                want_same = 1.0 if "same_type_gain=1" in extra else 0.1
                if abs(float(lif.get("same_type_gain", want_same)) - want_same) > 1e-9:
                    problems.append(f"{f.name}: same_type_gain {lif.get('same_type_gain')} != {want_same}")
                if r.get("receptor_model") != brain.LIFParams().receptor_model:
                    problems.append(f"{f.name}: receptor_model {r.get('receptor_model')} != shipped")
                want_hold = [extra[i + 1] for i, x in enumerate(extra) if x == "--hold-edges"]
                got_hold = [h[0] + ":" + h[1] for h in (r.get("hold_edges") or [])]
                if got_hold != want_hold:
                    problems.append(f"{f.name}: hold_edges {got_hold} != {want_hold} (arm {label})")
                if want_hold and not all(abs(float(h[2])) < 1e-12 for h in (r.get("hold_edges") or [])):
                    problems.append(f"{f.name}: hold factor is not 0: {r.get('hold_edges')}")
                if want_hold and d["hold_entries"] != 1149:
                    problems.append(f"{f.name}: hold matched {d['hold_entries']} entries, expected 1149")
                want_gain = [extra[i + 1] for i, x in enumerate(extra) if x == "--edge-gain"]
                got_gain = [f"{h[0]}:{h[1]}:{h[2]:g}" for h in (r.get("edge_gains") or [])]
                if got_gain != want_gain:
                    problems.append(f"{f.name}: edge_gains {got_gain} != {want_gain} (arm {label})")
                if want_gain:
                    eg = (r.get("edge_gains_resolved") or [{}])[0]
                    if eg.get("n_entries") != 842 or eg.get("n_entries_same_type") != 842:
                        problems.append(f"{f.name}: edge gain matched {eg.get('n_entries')} entries ({eg.get('n_entries_same_type')} same-type), expected 842 / 842")
                    if abs(float(eg.get("effective_factor_same_type", 0.0)) - 1.0) > 1e-6:
                        problems.append(f"{f.name}: effective EPG->EPG factor {eg.get('effective_factor_same_type')} != 1.0")
                for g in ("ExR6", "ER6", "ER4m", "EPGt"):
                    if m.get(f"{g}_mean_post") is None:
                        problems.append(f"{f.name}: per-type ring rate {g}_mean_post not recorded")
                rows.append(d)
    return rows


def analyse_batch_cx7(out_dir: Path, refs=("S", "H3"), seeds=(0, 1, 2, 3, 4)):
    """Thread 6B analysis (CPU): the 6A path extended with two references, the driven-tile rule, the per-type ring
    rates, the per-cell maxima and the predeclared calls for H3E / H3F, every table emitted to a named file."""
    from flyverse.interp import common
    ARMS = list(CX7_ARMS)
    out_dir = Path(out_dir); an = out_dir / "analysis"; an.mkdir(parents=True, exist_ok=True)
    problems = []
    rows = _run_rows_cx7(out_dir, ARMS, problems)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows"); return None
    order = {a[0]: i for i, a in enumerate(ARMS)}
    df = df.sort_values(["arm", "seed"], key=lambda s: s.map(order) if s.name == "arm" else s).reset_index(drop=True)
    counts = df.groupby("arm").seed.size()
    for a_ in {a[0] for a in ARMS} - set(counts.index):
        problems.append(f"arm {a_}: 0 runs")
    for a_, n_ in counts.items():
        if n_ != len(seeds):
            problems.append(f"arm {a_}: {n_} runs, expected {len(seeds)}")
    md5s = df.groupby("arm").md5.agg(lambda s: sorted(set(s)))

    def vals(sub, k):
        return [float(x) for x in sub[k].tolist() if x is not None and np.isfinite(float(x))]

    # verdicts vs EACH reference: Holm within the (arm, ref) family of the five primaries; magnitudes and secondaries outside
    comp = []
    for ref in refs:
        ref_df = df[df.arm == ref]
        for label, desc, gains, glu, extra, cls in ARMS:
            if label == ref:
                continue
            sub = df[df.arm == label]
            if sub.empty:
                continue
            pv, res_k = {}, {}
            for k in PRIMARIES_CX7 + MAGNITUDES_CX7 + SECONDARIES_CX7 + ("survival_at_tile_s", "frac_at_tile_post"):
                a_, b_ = vals(sub, k), vals(ref_df, k)
                fam = "primary" if k in PRIMARIES_CX7 else ("magnitude" if k in MAGNITUDES_CX7 else "secondary")
                if len(a_) == 0 or len(b_) == 0:
                    res_k[k] = dict(family=fam, verdict="no data" if len(a_) == 0 and len(b_) == 0 else "one-sided (reference has no confined frames)",
                                    diff=float("nan"), z=float("nan"), p=float("nan"), p_floor=float("nan"), null_sd_zero=None, n_stim=len(a_), n_null=len(b_),
                                    stim_values=[round(x, 4) for x in a_], null_values=[round(x, 4) for x in b_])
                    continue
                c_ = common.compare(a_, b_)
                res_k[k] = dict(family=fam, verdict=c_["verdict"], diff=c_["diff"], z=c_["z"], p=c_["p"], p_floor=c_["p_floor"], null_sd_zero=c_["null_sd_zero"],
                                n_stim=len(a_), n_null=len(b_), stim_values=[round(x, 4) for x in a_], null_values=[round(x, 4) for x in b_])
                if k in PRIMARIES_CX7:
                    pv[k] = c_["p"]
            adj = holm(pv)
            for k, v in res_k.items():
                comp.append(dict(arm=label, ref=ref, key=k, p_holm=adj.get(k, float("nan")) if k in PRIMARIES_CX7 else float("nan"), m_holm=len(pv) if k in PRIMARIES_CX7 else None, **v))
    cdf = pd.DataFrame(comp)
    # the two predeclared rules, per seed
    rule = []
    for label, desc, gains, glu, extra, cls in ARMS:
        sub = df[df.arm == label].sort_values("seed")
        if sub.empty:
            continue
        s_ = sub.survival_s.astype(float).fillna(0).to_numpy(); w_ = sub.width_half_post.astype(float).to_numpy()
        h_ = sub.bump_hz_post.astype(float).to_numpy(); dist = sub.centre_dist_t5.astype(float).to_numpy()
        surv = s_ >= 5.0 - 1e-9
        width_ok = np.isfinite(w_) & (w_ >= 2.5) & (w_ <= 5.0)
        rate_ok = np.isfinite(h_) & (h_ >= 5.0) & (h_ <= 60.0)
        tile = np.isfinite(dist) & (dist <= DRIVEN_TILE_TOL)
        working = surv & width_ok & rate_ok
        bump_at_tile = surv & width_ok & tile
        rule.append(dict(arm=label, gains=gains, runs=int(len(sub)), working_compass_seeds=int(working.sum()), driven_tile_seeds=int(tile.sum()),
                         bump_survives_seeds=int(surv.sum()), bump_at_tile_seeds=int(bump_at_tile.sum()), compass_at_tile_seeds=int((working & tile).sum()),
                         working=bool(working.sum() >= 3), at_tile=bool(tile.sum() >= 3),
                         survival=s_.tolist(), rate=h_.tolist(), width=w_.tolist(), centre_dist_t5=dist.tolist(), centre_t5=sub.centre_wedge_t5.tolist(),
                         survival_at_tile=sub.survival_at_tile_s.tolist(), frac_confined_post=sub.frac_confined_post.tolist(),
                         pen_during=sub.PEN_mean_during.tolist(), devices=sorted(set(map(str, sub.device_name))), md5=md5s.get(label, [])))
    rdf = pd.DataFrame(rule)
    # the predeclared phrases (worded for H3E and H3F; printed per arm, read for those two)
    calls = []
    for _, r_ in rdf.iterrows():
        if r_.working_compass_seeds >= 3 and r_.driven_tile_seeds >= 3:
            call = "a compass at the driven tile"
        elif r_.bump_at_tile_seeds >= 3:
            call = "a bump at the driven tile, not a compass (rate outside 5-60 Hz)"
        elif r_.bump_survives_seeds >= 3:
            call = "a bump, not at the driven tile"
        else:
            call = "no bump"
        calls.append(dict(arm=r_.arm, runs=r_.runs, working_compass_seeds=r_.working_compass_seeds, driven_tile_seeds=r_.driven_tile_seeds,
                          bump_survives_seeds=r_.bump_survives_seeds, bump_at_tile_seeds=r_.bump_at_tile_seeds, call=call,
                          in_rule=r_.arm in ("H3E", "H3F")))
    cl = pd.DataFrame(calls)
    rank = {"no bump": 0, "a bump, not at the driven tile": 1, "a bump at the driven tile, not a compass (rate outside 5-60 Hz)": 2, "a compass at the driven tile": 3}
    ce = cl[cl.arm == "H3E"].call.iloc[0] if (cl.arm == "H3E").any() else None
    cf = cl[cl.arm == "H3F"].call.iloc[0] if (cl.arm == "H3F").any() else None
    contrast = None
    if ce and cf:
        contrast = ("the per-type EPG->EPG recurrence is sufficient (H3E in the same class as H3F)" if rank[ce] >= rank[cf]
                    else "the other same-type cliques are needed (H3F in a better class than H3E)")
    cl.to_csv(an / "call.csv", index=False)
    # per-type ring rates and per-cell maxima per arm (closing 6A's weak link)
    rr = []
    for label, *_ in ARMS:
        sub = df[df.arm == label]
        if sub.empty:
            continue
        for g in ("ExR6", "ER6", "ER4m", "EPGt", "PEG", "PEN", "Delta7", "GLNO", "Ring", "rest"):
            for w in ("pre", "during", "post"):
                k = f"{g}_mean_{w}"
                if k in sub:
                    v = sub[k].astype(float)
                    rr.append(dict(arm=label, group=g, window=w, mean_min=float(v.min()), mean_max=float(v.max()), mean_mean=float(v.mean()),
                                   values=",".join(f"{x:.4f}" for x in v),
                                   cell_max_min=float(sub[f"{g}_cell_max_{w}"].astype(float).min()) if f"{g}_cell_max_{w}" in sub and sub[f"{g}_cell_max_{w}"].notna().any() else float("nan"),
                                   cell_max_max=float(sub[f"{g}_cell_max_{w}"].astype(float).max()) if f"{g}_cell_max_{w}" in sub and sub[f"{g}_cell_max_{w}"].notna().any() else float("nan")))
    ring_df = pd.DataFrame(rr)
    ring_df.to_csv(an / "ring_rates.csv", index=False)
    # state.csv (rule 28: every per-seed number the audit quotes comes from here or scatter.csv)
    state_cols = ["arm", "seed", "epg_mean_pre", "PEN_mean_pre", "Delta7_mean_pre", "GLNO_mean_pre", "Ring_mean_pre", "ExR6_mean_pre", "ER6_mean_pre", "ER4m_mean_pre",
                  "frac_confined_pre", "frac_confined_during", "frac_confined_post", "epg_in_mean_during", "epg_out_mean_during", "epg_in_mean_post", "epg_out_mean_post",
                  "in_above_t5", "out_above_t5", "centre_wedge_t5", "centre_dist_t5", "vs_t5", "centre_dist_post_confined", "survival_at_tile_s", "frac_at_tile_post",
                  "PEN_mean_during", "PEN_mean_post", "Delta7_mean_post", "GLNO_mean_post", "Ring_mean_post", "ExR6_mean_post", "ER6_mean_post", "ER4m_mean_post",
                  "EPGt_mean_post", "PEG_mean_post", "rest_mean_post", "profile_end"]
    df[[c for c in state_cols if c in df]].to_csv(an / "state.csv", index=False)
    # scatter.csv: primaries + magnitudes + the at-tile survival, per arm
    sc = []
    for label, *_ in ARMS:
        sub = df[df.arm == label].sort_values("seed")
        if sub.empty:
            continue
        for k in PRIMARIES_CX7 + MAGNITUDES_CX7 + ("survival_at_tile_s", "centre_wedge_t5"):
            v = sub[k].astype(float)
            sc.append(dict(arm=label, key=k, seeds=",".join(str(int(x)) for x in sub.seed),
                           values=",".join("nan" if not np.isfinite(float(x)) else f"{float(x):.4f}" for x in v),
                           mean=float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"),
                           sd=float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")))
    pd.DataFrame(sc).to_csv(an / "scatter.csv", index=False)
    txt_missing = [a[0] + f"_s{s}" for a in ARMS for s in seeds if not (out_dir / f"{a[0]}_s{s}.txt").exists()]
    df.to_csv(an / "runs.csv", index=False)
    cdf.to_csv(an / "compare.csv", index=False)
    rdf.to_csv(an / "decision.csv", index=False)
    json.dump(dict(runs=df.to_dict("records"), compare=comp, decision=rule, calls=calls, contrast_H3E_vs_H3F=contrast, problems=problems, txt_missing=txt_missing,
                   n_runs=int(len(df)), n_expected=len(ARMS) * len(seeds), refs=list(refs), primaries=list(PRIMARIES_CX7), magnitudes=list(MAGNITUDES_CX7),
                   driven_centre=DRIVEN_CENTRE, driven_tile_tol=DRIVEN_TILE_TOL, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   generator="python " + " ".join(sys.argv), analysis_sha256=sha256_file(Path(__file__)),
                   cx_wedge_sha256=sha256_file(ROOT / "scripts" / "cx_wedge.py"), common_sha256=sha256_file(ROOT / "flyverse" / "interp" / "common.py"),
                   probe_compass_room_sha256=sha256_file(ROOT / "scripts" / "probe_compass_room.py")),
              open(an / "analysis.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    L = [f"# batch analysis (cx7) -- n_runs {len(df)} of {len(ARMS) * len(seeds)} expected ({out_dir.as_posix()}/<arm>_s<seed>.json); generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
         f"Problems ({len(problems)}): " + ("; ".join(problems) if problems else "none") + (f"; console .txt missing: {txt_missing}" if txt_missing else "") + "\n",
         "## Per run\n"]
    cols = ["arm", "seed", "survival_s", "survival_at_tile_s", "bump_hz_post", "width_half_post", "frac_confined_post", "centre_wedge_t5", "centre_dist_t5",
            "epg_in_mean_post", "epg_out_mean_post", "PEN_mean_during", "PEN_mean_post", "Delta7_mean_post", "GLNO_mean_post", "ExR6_mean_post", "ER6_mean_post",
            "ER4m_mean_post", "rest_mean_post", "ledger_survival", "ledger_rate", "ledger_width", "device_name", "md5", "wall_s"]
    L.append("| " + " | ".join(cols) + " |"); L.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        L.append("| " + " | ".join((f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    L.append("\n## The two rules per arm (working compass: survival >= 5 AND width 2.5-5 AND rate 5-60, in >= 3 of 5; driven tile: centre within 1.5 wedges of 1.5 at 5 s, in >= 3 of 5)\n")
    L.append("| arm | runs | working seeds | driven-tile seeds | surviving bump seeds | bump-at-tile seeds | compass-at-tile seeds | survival per seed | rate | width | centre dist t5 | survival at tile | call |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    fmtl = lambda v: "/".join("nan" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.3g}" for x in v)   # noqa: E731
    for _, r in rdf.iterrows():
        call = cl[cl.arm == r.arm].call.iloc[0]
        L.append(f"| {r.arm} | {r.runs} | {r.working_compass_seeds} | {r.driven_tile_seeds} | {r.bump_survives_seeds} | {r.bump_at_tile_seeds} | {r.compass_at_tile_seeds} | "
                 f"{fmtl(r.survival)} | {fmtl(r.rate)} | {fmtl(r.width)} | {fmtl(r.centre_dist_t5)} | {fmtl(r.survival_at_tile)} | {call} |")
    L.append(f"\nH3E vs H3F: {contrast}\n")
    L.append("\n## Verdicts (common.compare, runs = the unit, 5 v 5; Holm within each (arm, reference) family of the five primaries)\n")
    L.append("| arm | ref | key | family | verdict | diff | z | p | p Holm | null SD zero | stim values | null values |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in cdf[cdf.family != "secondary"].iterrows():
        L.append(f"| {r.arm} | {r.ref} | {r.key} | {r.family} | {r.verdict} | {r['diff']:+.4g} | {r.z:+.1f} | {r.p:.3g} | {r.p_holm:.3g} | {r.null_sd_zero} | {r.stim_values} | {r.null_values} |")
    L.append("\n## Per-type ring rates (population means, Hz; min-max over seeds; per-cell max = max over cells of the window-mean rate)\n")
    L.append("| arm | group | window | mean min-max | per-cell max min-max |"); L.append("|---|---|---|---|---|")
    for _, r in ring_df.iterrows():
        L.append(f"| {r.arm} | {r.group} | {r.window} | {r.mean_min:.3f}-{r.mean_max:.3f} | {r.cell_max_min:.3f}-{r.cell_max_max:.3f} |")
    (an / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = list(PRIMARIES_CX7) + ["bump_hz_post", "survival_at_tile_s"]
        fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6))
        order_ = [a[0] for a in ARMS]
        for ax, k in zip(axes, keys):
            for i, arm in enumerate(order_):
                v = df[df.arm == arm][k].astype(float).to_numpy()
                v = np.where(np.isfinite(v), v, np.nan)
                ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=18)
                if np.isfinite(v).any():
                    ax.plot([i - 0.25, i + 0.25], [np.nanmean(v)] * 2, color="k", lw=1)
            ax.set_xticks(range(len(order_))); ax.set_xticklabels(order_, rotation=60, fontsize=7); ax.set_title(k, fontsize=9)
        fig.tight_layout(); fig.savefig(an / "scatter.png", dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"scatter figure skipped: {e!r}")
    print(f"n_runs {len(df)} ({out_dir.as_posix()}/<arm>_s<seed>.json); problems {len(problems)}")
    for p_ in problems:
        print("  PROBLEM", p_)
    print(cl.to_string(index=False))
    print("H3E vs H3F:", contrast)
    print(f"-> {an / 'analysis.md'}, runs.csv, compare.csv, decision.csv, call.csv, scatter.csv, state.csv, ring_rates.csv, analysis.json, scatter.png")
    return df, cdf, rdf


def write_predeclaration_cx7(out_dir: Path, seeds, structure_json: Path, sigma_json: Path | None, name="cx7", minutes=30):
    """The stamped 6B predeclaration (docs/INTERP.md 10.4 rule 10): arms, the two references, the family (m = 5) and its
    drop rule, the two rules and the four-way phrases for H3E / H3F in words, the sigma measurement's consequence, and the
    fixed tool's predictions per arm at the MEASURED sigma read out of `structure_json`. Archives any existing file."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    st = json.load(open(structure_json, encoding="utf-8"))
    by_label = {r["label"].split(":")[0].strip(): r for r in st["configs"]}
    arm_cfg = {"S": "shipped", "H3": "H3", "H3G": "H3+GLNO=glu", "F": "f", "H3F": "H3F", "H3E": "H3E", "H3FG": "H3FG", "H3EG": "H3EG"}
    pred = {}
    for label, desc, gains, glu, extra, cls in CX7_ARMS:
        r = by_label.get(arm_cfg.get(label))
        if r is None:
            pred[label] = dict(note="no rate-model configuration for this arm"); continue
        row = {}
        for stt in ("background", "pulse", "after"):
            m = r["rate_model"][stt]; b = r["bump_criterion"][stt]; j = r["jacobian"][stt]
            row[stt] = dict(EPG_in_hz=m["epg_in"], EPG_out_hz=m["epg_out"], EPG_out_max_hz=m["epg_out_max"], in_above_22=m["in_above_22"], out_above_22=m["out_above_22"],
                            confined_by_ledger_rule=m["confined_by_ledger_rule"], PEN_in_hz=m["PEN_in"], PEN_out_hz=m["PEN_out"],
                            Delta7_hz=m["Delta7"], Ring_hz=m["Ring"], GLNO_hz=m["GLNO"], u_PEN_in_mV=m["u_PEN_in"], u_EPG_in_mV=m["u_EPG_in"],
                            k1_gain=b["k1_gain"], gamma_EPG=b["gamma_EPG"], gamma_PEN=b["gamma_relay"]["PEN"], jacobian_leading=j["leading_re"],
                            gamma_E_crit_epg_recurrent=b["epg_recurrent"]["gamma_EPG_crit"], saturation_hz_epg_recurrent=b["epg_recurrent"]["saturation_hz"],
                            gamma_E_crit_with_relays=b["with_relays"]["gamma_EPG_crit"], saturation_hz_with_relays=b["with_relays"]["saturation_hz"])
        row["fixed_point_bump_after_release"] = r["rate_model"]["bump_after"]
        row["fixed_point_confined_bump_after_release"] = r["rate_model"]["confined_bump_after"]
        row["fixed_point_runaway_after_release"] = r["rate_model"]["runaway_after"]
        row["epg_recurrence_predicts_a_bump"] = r["bump_criterion"]["pulse"]["epg_recurrent"]["predicted_bump"]
        row["epg_recurrence_saturation_hz"] = r["bump_criterion"]["pulse"]["epg_recurrent"]["saturation_hz"]
        row["local_EPG_EPG_mV"] = r["local_kernels"]["direct"]
        row["ring_rates_hz"] = {k: v for k, v in r["decomposition"]["pulse"].get("PEN_in<-Ring_by_type", {}).items()}
        row["predicted_bump_at_driven_tile"] = bool(r["rate_model"]["confined_bump_after"])
        row["predicted_PEN_during_hz"] = r["rate_model"]["pulse"]["PEN_in"]
        row["predicted_PEN_post_hz"] = r["rate_model"]["after"]["PEN_in"]
        row["predicted_bump_rate_hz"] = (r["rate_model"]["after"]["epg_in"] if r["rate_model"]["bump_after"] else r["bump_criterion"]["pulse"]["epg_recurrent"]["saturation_hz"])
        pred[label] = row
    sig = json.load(open(sigma_json, encoding="utf-8")) if sigma_json and Path(sigma_json).is_file() else None
    doc = {
        "schema": "flyverse.predeclaration/1", "stamped_utc": stamp, "written_before_submission": True,
        "thread": "6B: the hold plus a wedge-local recurrence (docs/audits/compass_local_recurrence.md)",
        "question": ("6A found the DC brake on the relays (ExR6 + ER6 + ER4m -> PEN, EPG) necessary and not sufficient: held off, PEN fires 40-48 Hz "
                     "and the ring saturates into a five-wedge hump at the wrong place. The only wedge-local recurrence at the shipped gains is the "
                     "x0.1-damped EPG -> EPG synapses. Can this ring hold a bump AT THE DRIVEN TILE once the DC brake is off AND the local recurrence "
                     "is on? Two labelled instruments at once -- a mechanism question, never an adoption."),
        "batch": {"name": name, "target": "house (<cluster-node>; .cluster.json default)", "submissions": 1, "jobs": len(seeds) * 2, "runs": len(CX7_ARMS) * len(seeds),
                  "arms": len(CX7_ARMS), "seeds_per_arm": list(seeds),
                  "generator": f"python scripts/cx_ring_structure.py --batch cx7 --plan-batch {out_dir.as_posix()} --minutes {minutes}",
                  "blocks": "fam_s<seed>: every arm of one seed on ONE target; GLNO-silent arms (S, H3, H3F, H3E, F) in one job, GLNO = glutamate arms (H3G, H3FG, H3EG) in another",
                  "job_line": ("mkdir -p out/cx7 && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && python scripts/cx_wedge.py "
                               "--no-structure --sim 1:1 --ledger --seed <s> --arm <A> --block fam_s<s> [--nt-override GLNO=glutamate] <arm flags> --sim-out out/cx7/<A>_s<s>.json "
                               "> out/cx7/<A>_s<s>.txt 2>&1; s<i>=$?; tail -3 ...; exit $((s0 | s1 | ...)) -- tested on CPU before submission (out/cx7/smoke/jobline_check.txt)"),
                  "protocol": ("the cx5 / cx6 protocol, unchanged: cx_wedge.simulate --ledger, FlyBrain on the full MaleCNS connectome, no world, compass adaptation 0, "
                               "10 Hz Poisson background on all 46 EPG for the whole run, 1 s settle, wedges 0-3 (11 EPG) at +40 Hz for 2 s, then 5 s free; EPG per 10 ms "
                               "frame scored by probe_compass_room.bump_frames; survival = end of the last confined post-pulse frame minus the pulse end (max 5.00 s). "
                               "NEW RECORDED KEYS ONLY (the simulation is untouched): per-type ring rates ExR6 / ER6 / ER4m and EPGt as g__ groups, and the per-cell "
                               "rates of the small compass groups with their per-cell maxima"),
                  "gains": "SHIPPED (gE 1 / gD 1) in every arm",
                  "cache": "GLNO-silent arms read the shared cache (compiled W md5 ef23cc27bea13be7f6a96f3c04fd3737); the GLNO = glutamate arms compile the 7a10d93b cache via --nt-override, as in cx6",
                  "batch_sh_sha256": sha256_file(out_dir / "batch.sh") if (out_dir / "batch.sh").is_file() else None,
                  "tree_state": (out_dir / "tree_state.json").as_posix()},
        "instruments": {
            "hold": "scripts/cx_wedge.py --hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)' (6A; default None): 17 pre cells onto 88 post cells, 1,149 entries, 37,256 synapses at 0 -- a LABELLED COUNTERFACTUAL",
            "global_recurrence": "--lif same_type_gain=1 (the global hand rule removed; every same-type clique at its connectome weight) -- a LABELLED INSTRUMENT (cx5's F)",
            "per_type_recurrence": ("--edge-gain '^EPG$:^EPG$:10' (NEW in 6B, default None): a type_path_gain entry applied BEFORE same_type_gain, so x10 then x0.1 lands the 842 "
                                    "EPG -> EPG pairs at exactly x1.0 (bit-identical to same_type_gain 1 on that block; tests/test_cx_wedge_hold.py) while PEN_a, PEN_b, Delta7 and PEG "
                                    "cliques stay x0.1. No brain.py change was needed: the stage order allows it. A LABELLED INSTRUMENT (a per-type gain)"),
            "bit_identity": "shipped path on CPU with both flags absent: out/cx7/smoke/smoke_default_path_noledger.json equals out/cx6/smoke/smoke_default_path.json on 120 of 121 shared fields (wall_s excepted), two new record-keeping keys",
            "not_adoptable": "two instruments decide a MECHANISM question; neither the hold nor a per-type gain nor the global damping removal is a candidate default",
        },
        "arms": {label: dict(label=desc, gains_gE_gD=gains, glno_glutamate=glu, flags=list(extra), classification=cls) for label, desc, gains, glu, extra, cls in CX7_ARMS},
        "references": ["S", "H3"],
        "primaries": {
            "keys": list(PRIMARIES_CX7),
            "definitions": {"survival_s": "bump_survival_s (ledger)", "frac_confined_post": "fraction of post-pulse frames confined",
                            "centre_dist_t5": "ring distance (wedges, 0-8) between the EPG profile's vector centre at 5 s (t5.0_centre_wedge) and the driven block's centre 1.5",
                            "PEN_mean_during": "PEN population mean during the pulse", "PEN_mean_post": "after release"},
            "family": ("Holm within each (arm, reference) family of the FIVE primaries, references S and H3 separately. m = 5 at 5 v 5 (p_floor 0.0079): "
                       "5 x 0.0079 = 0.040 <= 0.05, satisfiable. bump_hz_post and width_half_post are undefined in both references (no confined frame in S or H3 in "
                       "cx5 / cx6) and are declared MAGNITUDES outside the family; if a reference should turn out to have confined frames they stay magnitudes. "
                       "A primary that returns no p is dropped from the denominator and reported as a magnitude (docs/INTERP.md 10.2)."),
            "m_max": 5,
            "magnitudes_outside_the_family": list(MAGNITUDES_CX7) + ["survival_at_tile_s (confined AND centre within 1.5 wedges of the driven block, from the .npz)",
                                                                     "frac_at_tile_post", "centre_dist_post_confined"] + list(SECONDARIES_CX7),
            "comparison": "flyverse.interp.common.compare, runs = the unit, 5 v 5; |z| >= 3 and p <= 0.05 -> result; a zero-SD null is structural (undetermined = read the magnitude with p)",
        },
        "decision_rule": {
            "working_compass": "bump_survival_s >= 5 s AND bump_width_wedges in [2.5, 5] AND bump_rate_hz in [5, 60] Hz, per seed; the arm meets it in >= 3 of 5 seeds",
            "driven_tile": f"centre_dist_t5 <= {DRIVEN_TILE_TOL} wedges (the driven block is wedges 0-3, centre {DRIVEN_CENTRE}), per seed; the arm meets it in >= 3 of 5 seeds",
            "phrases (worded for H3E and H3F; printed per arm)": {
                "a compass at the driven tile": "the working-compass rule AND the driven-tile rule each in >= 3 of 5 seeds",
                "a bump at the driven tile, not a compass (rate outside 5-60 Hz)": "survival >= 5 s AND width 2.5-5 AND centre within 1.5 wedges in >= 3 of 5, but the rate row fails",
                "a bump, not at the driven tile": "survival >= 5 s in >= 3 of 5 seeds but the driven-tile rule in < 3 of 5",
                "no bump": "survival >= 5 s in < 3 of 5 seeds",
            },
            "H3E_vs_H3F": "'the per-type EPG->EPG recurrence is sufficient' when H3E lands in the same or a better class than H3F; 'the other same-type cliques are needed' when H3F lands in a better class",
            "GLNO": "H3EG vs H3E and H3FG vs H3F: whether the relabel moves the class (read as magnitudes; the relabel's status belongs to docs/audits/glno_relabel.md)",
            "what_the_hold_adds": "H3F vs F and H3E vs F (F carried as a reference): read as magnitudes",
        },
        "sigma": (dict(source=str(sigma_json), assumed_mV=SIGMA_MV, measured=sig.get("headline") if isinstance(sig, dict) else None,
                       consequence=sig.get("consequence") if isinstance(sig, dict) else None) if sig else {"note": "no sigma measurement supplied"}),
        "predictions_from_the_structure_pass": {
            "source": f"{structure_json.as_posix()} (CPU; drive as a current with the forced rate as a floor, one-step EPG -> EPG term kept, per-cell gamma; sigma {st.get('modes', {}).get('sigma_mV')} mV)",
            "reading": ("bump_at_driven_tile = the fixed point after release is a bump (in > 2 x out, in > 15 Hz) AND passes the ledger's confinement count on its own "
                        "cells (>= 8 of 11 driven above 22 Hz, <= 3 of 35 outside) -- the criterion 6A's H3 prediction lacked; the rate is the fixed point's EPG-in "
                        "when it holds a bump, else the EPG-recurrence saturation rate"),
            "per_arm": pred,
            "falsifiers": ("H3E or H3F holding a confined bump at the driven tile in >= 3/5 seeds where the fixed point predicts a saturated ring (or the reverse) is a "
                           "tool miss to record. H3E in the same class as H3F says the EPG's own synapses carry the recurrence; H3F better than H3E says the PEN / PEG "
                           "cliques matter. A working compass (rate 5-60 Hz) in any arm would be the first at the shipped gains."),
        },
        "analysis": {"path": f"python scripts/cx_ring_structure.py --batch cx7 --analyse {out_dir.as_posix()} (CPU), writing analysis/{{analysis.md,runs.csv,compare.csv,decision.csv,call.csv,scatter.csv,state.csv,ring_rates.csv,analysis.json,scatter.png}}",
                     "per_seed_scatter": "analysis/scatter.csv and analysis/state.csv -- the audit pastes from those files (docs/INTERP.md 10.4 rule 28)",
                     "checks": "per run: device cuda, arm label, gains, nt_override / glno_nt, same_type_gain, receptor_model, the hold (spec, factor 0, 1,149 entries), the edge gain (spec, 842 entries all same-type, effective factor 1.0), the per-type ring rates present"},
        "adoption": "NOTHING IS ADOPTED BY THIS THREAD. Two instruments decide a mechanism question; every new flag defaults to None; no default changes.",
    }
    path = out_dir / "predeclared.json"
    if path.exists():
        arch = out_dir / f"predeclared.archived_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        arch.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"archived the previous predeclaration -> {arch}")
    path.write_text(json.dumps(doc, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    print(f"-> {path} (stamped {stamp})")
    return doc


# ------------------------------------------------------------------------------------------------ predeclaration
def write_predeclaration(out_dir: Path, arms, seeds, structure_json: Path, name="cx6", minutes=30, ref="S"):
    """The stamped predeclaration (docs/INTERP.md 10.4 rule 10: the predeclaration is this JSON, not the audit).
    Arms, primaries, the Holm family and its size, the decision rule in words, and the structure pass's PREDICTIONS
    per arm read out of `structure_json` -- all written before the batch is submitted. If a predeclaration is already
    there it is archived next to it (rule 6 of the 5A skeptic pass: archive before any amendment)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    st = json.load(open(structure_json, encoding="utf-8"))
    by_label = {}
    for r in st["configs"]:
        key = r["label"].split(":")[0].strip()
        by_label[key] = r
    pred = {}
    # the structure configurations that correspond to the batch's arms
    arm_cfg = {"S": "shipped", "H3": "H3", "H_ExR6": "H_ExR6", "H_ER6": "H_ER6", "H_ER4m": "H_ER4m",
               "H3G": "H3+GLNO=glu", "F": "f", "R": None}
    for label, desc, gains, glu, extra, cls in arms:
        cfg = arm_cfg.get(label)
        r = by_label.get(cfg) if cfg else None
        if r is None:
            pred[label] = dict(note="no rate-model prediction (the reference gains gE 2 / gD 15 are outside the "
                                    "shipped-gain rate model); the cx5 measurement is the expectation: a bump at 201-203 Hz in 4/4 seeds")
            continue
        row = {}
        for stt in ("background", "pulse", "after"):
            m = r["rate_model"][stt]; b = r["bump_criterion"][stt]; j = r["jacobian"][stt]
            row[stt] = dict(EPG_in_hz=m["epg_in"], EPG_out_hz=m["epg_out"], PEN_in_hz=m["PEN_in"], PEN_out_hz=m["PEN_out"],
                            Delta7_hz=m["Delta7"], Ring_hz=m["Ring"], GLNO_hz=m["GLNO"], u_PEN_in_mV=m["u_PEN_in"],
                            u_EPG_in_mV=m["u_EPG_in"], k1_gain=b["k1_gain"], k1_gain_epg_only=b["k1_gain_epg_only"],
                            gamma_EPG=b["gamma_EPG"], gamma_PEN=b["gamma_relay"]["PEN"],
                            jacobian_leading=j["leading_re"], spectral_radius=j["spectral_radius"])
        row["fixed_point_bump_after_release"] = r["rate_model"]["bump_after"]
        row["epg_recurrence_predicts_a_bump"] = r["bump_criterion"]["pulse"]["epg_recurrent"]["predicted_bump"]
        row["epg_recurrence_saturation_hz"] = r["bump_criterion"]["pulse"]["epg_recurrent"]["saturation_hz"]
        row["decomposition_pulse_PEN"] = {k.replace("PEN_in<-", ""): v for k, v in r["decomposition"]["pulse"].items()
                                          if k.startswith("PEN_in<-") and not k.endswith("by_type")}
        row["decomposition_pulse_EPG"] = {k.replace("EPG_in<-", ""): v for k, v in r["decomposition"]["pulse"].items()
                                          if k.startswith("EPG_in<-") and not k.endswith("by_type")}
        pred[label] = row
    doc = {
        "schema": "flyverse.predeclaration/1",
        "stamped_utc": stamp,
        "written_before_submission": True,
        "thread": "6A: the DC-balance test (docs/audits/compass_dc_balance.md)",
        "question": ("5A's fixed point says the DC inhibition through 2 ExR6 + 4 ER6 + 11 ER4m onto PEN and EPG "
                     "(-24.9 mV on PEN during the pulse, against a 7 mV gap) is what keeps the compass relays below "
                     "threshold. Holding exactly those edges at 0 turns that decomposition into a tested attribution."),
        "batch": {
            "name": name, "target": "house (<cluster-node>; .cluster.json default)", "submissions": 1,
            "jobs": len(seeds) * 2, "runs": len(arms) * len(seeds), "arms": len(arms), "seeds_per_arm": list(seeds),
            "generator": f"python scripts/cx_ring_structure.py --batch cx6 --plan-batch {out_dir.as_posix()} --minutes {minutes}",
            "blocks": "fam_s<seed>: every arm of one seed on ONE target (an experimental factor is never the unit of scheduling)",
            "job_line": ("mkdir -p out/cx6 && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' "
                         "&& python scripts/cx_wedge.py --no-structure --sim <gE:gD> --ledger --seed <s> --arm <A> --block fam_s<s> "
                         "[--nt-override GLNO=glutamate] <arm flags> --sim-out out/cx6/<A>_s<s>.json > out/cx6/<A>_s<s>.txt 2>&1; "
                         "s<i>=$?; tail -3 ...; exit $((s0 | s1 | ...)) -- the exit expression carries the python codes (tested on CPU "
                         "before submission; cx5's first submission died on a $((...)) expansion)"),
            "protocol": ("the cx5 protocol, unchanged: cx_wedge.simulate --ledger, FlyBrain on the full MaleCNS connectome, no world, "
                         "compass adaptation 0, 10 Hz Poisson background on all 46 EPG for the whole run, 1 s settle, wedges 0-3 (11 EPG) "
                         "at +40 Hz for 2 s, then 5 s free; EPG per 10 ms frame scored by probe_compass_room.bump_frames (vs > 0.6, "
                         ">= 8 of 11 block cells > 22 Hz, <= 3 of 35 outside); survival = end of the last confined post-pulse frame "
                         "minus the pulse end (max 5.00 s)"),
            "gains": "SHIPPED (gE 1 / gD 1) in every arm except the labelled reference R (gE 2 / gD 15, Delta7 -> EPG only)",
            "cache": ("GLNO-silent arms read the cluster's shared cache (compiled W md5 ef23cc27bea13be7f6a96f3c04fd3737); the H3G arm "
                      "compiles GLNO = glutamate into the run dir's out/cache_<hash>/ via --nt-override GLNO=glutamate, the same scratch "
                      "table cx5 used (local out/cache_c51b23e2, compiled W md5 7a10d93ba2086f2c76bcdabdca79b4ec; identical content to "
                      "the cx5b scratch cache out/cache_glno_glu)"),
            "batch_sh_sha256": sha256_file(out_dir / "batch.sh") if (out_dir / "batch.sh").is_file() else None,
            "tree_state": (out_dir / "tree_state.json").as_posix(),
            "job_line_tested_on_cpu": ("out/cx6/smoke/jobline_check.txt + exitcheck_*.sh / holdcheck.sh: the job lines were extracted "
                                       "from batch.sh BY BASH (out/cx6/smoke/argv_shim.py), the exit expression returns 0 when every arm "
                                       "succeeds and 7 when any one of them exits 7, and the --hold-edges regexes reach python's argv "
                                       "intact through both quoting levels"),
        },
        "hold": {
            "kind": "edges (docs/INTERP.md 2 / 10.1 step 5) -- a LABELLED COUNTERFACTUAL, not a candidate default",
            "mechanism": ("scripts/cx_wedge.py --hold-edges PRE_REGEX:POST_REGEX, a new flag defaulting to None, which appends "
                          "(pre, post, 0.0) to LIFParams.type_path_gain -- the same stage of brain._shaped_weights that carries gE / gD. "
                          "With the flag absent the installed gain list is entry-for-entry the previous one: the shipped default path is "
                          "bit-identical on CPU (out/cx6/smoke/smoke_default_path.json vs out/cx5/smoke/smoke_default_path.json, 117 of 117 "
                          "recorded fields equal)."),
            "spec_H3": HOLD3,
            "resolves_to": "17 presynaptic cells (2 ExR6 + 4 ER6 + 11 ER4m) onto 88 postsynaptic cells (46 EPG + 42 PEN); 1,149 W entries, 37,256 synapses. EPGt is NOT held.",
            "not_adoptable": "a hold is a counterfactual and can never become a default; what it can do is decide whether the DC balance is the story",
        },
        "arms": {label: dict(label=desc, gains_gE_gD=gains, glno_glutamate=glu, flags=list(extra), classification=cls)
                 for label, desc, gains, glu, extra, cls in arms},
        "reference": ref,
        "primaries": {
            "keys": list(PRIMARIES_CX6),
            "family": ("Holm within each arm's family of the six primaries vs S. m <= 6 is satisfiable at 5 v 5 "
                       "(p_floor 0.0079, 6 x 0.0079 = 0.047 <= 0.05). A member that returns no p -- bump_hz_post and "
                       "width_half_post are undefined in an arm with no confined frame, and are constant by construction "
                       "when neither arm has one -- is DROPPED from the Holm denominator and reported as a magnitude "
                       "(docs/INTERP.md 10.2: a quantity that cannot move is not a test)."),
            "m_max": 6,
            "reported_outside_the_family": list(SECONDARIES_CX6) + ["epg_in/out during and post (magnitudes)", "per-type ring rates (not recorded by this protocol)"],
            "comparison": ("flyverse.interp.common.compare, runs = the unit, 5 v 5; |z| >= 3 and p <= 0.05 -> result; a zero-SD null is "
                           "structural and reads as the magnitude diff with p (compare returns undetermined)"),
            "ledger_rows": {"compass.EPG.bump_survival_s": ">= 5 s", "compass.EPG.bump_rate_hz": "5-60 Hz (NOT_APPLICABLE unless survival >= 5)",
                            "compass.EPG.bump_width_wedges": "2.5-5 wedges (same)", "frac_confined_post": "reported, no bound"},
        },
        "decision_rule": {
            "working_compass": "bump_survival_s >= 5 s AND bump_width_wedges in [2.5, 5] AND bump_rate_hz in [5, 60] Hz, per seed",
            "the DC balance is the whole story": "H3 carries a bump meeting the working-compass rule at the shipped gains in >= 3 of 5 seeds",
            "necessary but not sufficient": "PEN fires under H3 (PEN_mean_during >= 1 Hz in the majority of seeds) but no arm meets the working-compass rule",
            "not the story": "PEN stays below 1 Hz under H3 (PEN_mean_during < 1 Hz in every seed)",
            "attribution_per_type": "H_ExR6 / H_ER6 / H_ER4m against H3 say which of the three carries the DC term; each is read as a magnitude against S",
            "GLNO": "H3G vs H3 is the first informative test of the GLNO sign at the SHIPPED gains, because it is the first shipped-gain configuration in which PEN and GLNO fire",
        },
        "predictions_from_the_structure_pass": {
            "source": f"{structure_json.as_posix()} (CPU; drive as a current with the forced rate as a floor, the one-step "
                      f"EPG -> EPG term kept, per-cell gamma at the realised fixed point)",
            "tool_validation": ("scripts/cx_ring_structure.py --validate-batch out/cx5: the fixed tool calls bump / no-bump correctly in 7 of "
                                "8 shipped-gain cx5 arms and predicts the saturation rate of the arms that held a bump to 5.4 % (F: 144 vs 152.5 Hz) "
                                "and 1.7 % (CFG: 160 vs 157.1); it predicts NO bump for S, G, C and CG (gamma_crit 33.3 / 28.8 Hz/mV against a maximum "
                                "LIF slope of 8.27). The one miss is FG (predicted a bump, measured none): the EPG-recurrence criterion cannot see the "
                                "glutamatergic GLNO's brake on PEN."),
            "statement": ("H3: PEN crosses threshold and the loop closes. The first-order DC number is 5A's: with the ring term at 0 the driven PEN's "
                          "input during the pulse is +9.7 (EPG) - 0.6 (Delta7) = +9.1 mV against the 7 mV gap. Self-consistently (the relays then drive "
                          "the EPG back) the fixed point runs much higher -- PEN 103 Hz at u +27.0 mV during the pulse, EPG in 260 Hz -- and holds after "
                          "release (EPG in 252 Hz, out 25.8, PEN 103, Delta7 143, GLNO 247), a state whose off-bump EPGs are ABOVE the ledger's 22 Hz "
                          "threshold, so a confinement failure is as likely as a clean bump. H_ExR6 alone: PEN 14.7 Hz during the pulse (u +1.3 mV), no "
                          "bump after release. H_ER6 alone: PEN 10.8 Hz (u -4.7). H_ER4m alone: PEN 0.0 Hz (u -15.5) -- ER4m is onto EPG, not PEN. "
                          "H3G: PEN 40.2 Hz during the pulse (GLNO -26.8 mV brakes it), bump after release at EPG in 158.7 Hz. S: PEN 0.0 Hz, u -15.8, "
                          "no bump (as measured in cx5). The k = 1 loop gain with per-cell gains is 0.000 in every shipped arm (gamma_EPG = 0: the driven "
                          "EPG's spikes are forced and its membrane is 33 mV below threshold), 0.434 during the pulse / 0.495 after under H3, and "
                          "0.731 / 1.024 under H3G."),
            "per_arm": pred,
            "falsifiers": ("H3 with PEN below 1 Hz would refute the DC-balance attribution outright. H3 with PEN firing but no bump would say the DC "
                           "balance is necessary and not sufficient. A bump in H3 whose rate is 150-250 Hz would meet the survival row and fail "
                           "compass.EPG.bump_rate_hz, exactly as every bump this project has produced so far."),
        },
        "analysis": {
            "path": f"python scripts/cx_ring_structure.py --batch cx6 --analyse {out_dir.as_posix()} (CPU), writing analysis/{{analysis.md,runs.csv,compare.csv,decision.csv,call.csv,scatter.csv,analysis.json,scatter.png}}",
            "per_seed_scatter": "analysis/scatter.csv (one row per arm x primary, the per-seed values in the `values` column) -- the audit pastes from that file (docs/INTERP.md 10.4 rule 28)",
            "checks": "per run: device cuda, arm label, gains, nt_override / glno_nt, same_type_gain, receptor_model, AND the hold (hold_edges equals the arm's declared spec, factor 0, non-zero matched entries)",
        },
        "adoption": "NOTHING IS ADOPTED BY THIS THREAD. A hold is a counterfactual; the new flag defaults to None; no default changes.",
    }
    path = out_dir / "predeclared.json"
    if path.exists():
        arch = out_dir / f"predeclared.archived_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        arch.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"archived the previous predeclaration -> {arch}")
    path.write_text(json.dumps(doc, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    print(f"-> {path} (stamped {stamp})")
    return doc


# ------------------------------------------------------------------------------------------------ validation of the
# fixed tool against a batch that is already on disk (thread 6A step 1: the 5A batch this pass previously missed)
def validate_batch(batch_dir: Path, out_dir: Path, drive="current", reduction="with-direct", gain="per-cell",
                   arms=None, tol=0.10, sigma=SIGMA_MV):
    """For every shipped-gain arm of `arms` (default the cx5 table) that has runs in `batch_dir`: build the same
    configuration from the arm's own flags, run the structure pass, and compare
      * the predicted saturation rate of the k = 1 mode (the rate at which the LIF slope falls back to gamma_crit)
        with the MEASURED `bump_hz_post` of the runs that held a bump -- `tol` relative;
      * the predicted bump (is the k = 1 mode supercritical at any operating point?) with the measured survival.
    Writes <out_dir>/validation.{json,md} and prints the table. No default changes; nothing is fitted."""
    arms = list(arms or BATCH_ARMS)
    batch_dir, out_dir = Path(batch_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = connectome.load(verbose=False)
    c_glu, glu_dir, _ = cx_wedge.load_connectome({"GLNO": "glutamate"}, scratch=True, verbose=False)
    cells, cells_glu = cx_wedge.compass_cells(c), cx_wedge.compass_cells(c_glu)
    rows = []
    for arm in arms:
        label = arm[0]
        cfg = config_for_arm(arm, c, c_glu, cells, cells_glu)
        files = sorted(batch_dir.glob(f"{label}_s*.json"))
        if cfg is None or not files:
            continue
        meas = []
        for f in files:
            for r in json.load(open(f, encoding="utf-8")):
                m = r.get("metrics") or {}
                meas.append((m.get("survival_s"), m.get("bump_hz_post"), m.get("PEN_mean_during"), m.get("epg_in_mean_during")))
        surv = [x[0] for x in meas if x[0] is not None]
        hz = [x[1] for x in meas if x[1] is not None and np.isfinite(x[1])]
        cir = Circuit(cfg[2], cfg[3], cfg[0], cfg[1], cells=dict(cfg[4]))
        res, _ = analyse(cir, drive=drive, reduction=reduction, gain=gain, sigma=sigma)
        tg = res["two_step_gain"]
        pred_hz = tg["saturation_hz_k1"]
        bc = res["bump_criterion"]["pulse"]
        held = [s for s in surv if s >= 5.0]
        meas_hz = float(np.mean([h for s, h, _, _ in meas if s is not None and s >= 5.0 and h is not None and np.isfinite(h)])) if held else float("nan")
        rel = abs(pred_hz - meas_hz) / meas_hz if (np.isfinite(pred_hz) and np.isfinite(meas_hz) and meas_hz > 0) else float("nan")
        pred_bump = bool(tg["predicted_bump"])
        meas_bump = bool(len(held) >= max(1, len(surv) // 2))
        rows.append(dict(arm=label, runs=len(meas), gamma_crit_k1=bc["epg_recurrent"]["gamma_EPG_crit"],
                         gamma_crit_k1_uniform=tg["gamma_crit_k1"], gamma_crit_with_relays=bc["with_relays"]["gamma_EPG_crit"],
                         saturation_with_relays=bc["with_relays"]["saturation_hz"], predicted_bump_with_relays=bc["with_relays"]["predicted_bump"],
                         local_direct_mV=bc["local_kernels"]["direct"], local_PEN=bc["local_kernels"]["PEN"], local_Ring=bc["local_kernels"]["Ring"],
                         local_Delta7=bc["local_kernels"]["Delta7"], gamma_relay_pulse=bc["gamma_relay"], direct_k1_mV=tg["direct_k1_mV"],
                         lambda_k1=tg["lambda_k1_net"], max_lif_slope=tg["max_lif_slope"], predicted_supercritical=pred_bump,
                         predicted_hz=pred_hz, measured_hz=meas_hz, rel_error=rel, seeds_with_bump=len(held), seeds=len(surv),
                         measured_survival=surv, measured_hz_per_seed=[round(h, 3) for h in hz],
                         bump_call_ok=bool(pred_bump == meas_bump),
                         rate_call_ok=bool(np.isfinite(rel) and rel <= tol) if meas_bump else None,
                         fixed_point_bump=res["rate_model"]["bump_after"], fixed_point_runaway=res["rate_model"]["runaway_after"],
                         pen_pulse_hz=res["rate_model"]["pulse"]["PEN_in"], u_pen_pulse=res["rate_model"]["pulse"]["u_PEN_in"],
                         jacobian_lead_pulse=res["jacobian"]["pulse"]["leading_re"], k1_gain_pulse=res["jacobian"]["pulse"]["loop_gain_k1"]))
    df = pd.DataFrame(rows)
    ok_bump = int(df.bump_call_ok.sum()) if len(df) else 0
    rate_rows = df[df.rate_call_ok.notna()] if len(df) else df
    ok_rate = int(rate_rows.rate_call_ok.sum()) if len(rate_rows) else 0
    L = [f"# cx_ring_structure validation against {batch_dir.as_posix()} (drive {drive}, reduction {reduction}, gain {gain}); "
         f"generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
         f"Bump / no-bump called correctly in {ok_bump} of {len(df)} arms; the saturation rate within {100 * tol:.0f} % in "
         f"{ok_rate} of {len(rate_rows)} arms that held a bump.\n",
         "| arm | gamma_E crit (Hz/mV) | local EPG->EPG (mV) | direct k1 (mV) | lambda_1 (mV^2) | loop closes (max LIF slope {:.2f} Hz/mV) | predicted Hz | measured Hz | rel err | seeds with a bump | fixed point | PEN during pulse (Hz / mV) | Jacobian lead / k1 gain (pulse) |".format(float(df.max_lif_slope.iloc[0]) if len(df) else float("nan")),
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rel_txt = "--" if not np.isfinite(r["rel_error"]) else "{:.1f} %".format(100 * r["rel_error"])
        L.append(f"| {r['arm']} | {r['gamma_crit_k1']:.2f} | {r['local_direct_mV']:+.2f} | {r['direct_k1_mV']:+.2f} | {r['lambda_k1']:+.0f} | {r['predicted_supercritical']} | "
                 f"{'--' if not np.isfinite(r['predicted_hz']) else format(r['predicted_hz'], '.0f')} | {r['measured_hz']:.1f} | {rel_txt} | "
                 f"{r['seeds_with_bump']}/{r['seeds']} | {'BUMP' if r['fixed_point_bump'] else ('RUNAWAY' if r['fixed_point_runaway'] else 'no bump')} | "
                 f"{r['pen_pulse_hz']:.1f} / {r['u_pen_pulse']:+.1f} | {r['jacobian_lead_pulse']:+.3f} / {r['k1_gain_pulse']:+.3f} |")
    (out_dir / "validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(dict(batch=batch_dir.as_posix(), drive=drive, reduction=reduction, gain=gain, sigma_mV=float(sigma), tol=tol, rows=rows,
                   bump_calls_ok=ok_bump, bump_calls=len(df), rate_calls_ok=ok_rate, rate_calls=len(rate_rows),
                   generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), generator="python " + " ".join(sys.argv),
                   generator_sha256=sha256_file(Path(__file__)), glu_cache=str(glu_dir)),
              open(out_dir / "validation.json", "w", encoding="utf-8"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(L))
    print(f"-> {out_dir / 'validation.md'}, {out_dir / 'validation.json'}")
    return df


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/cx5/structure")
    ap.add_argument("--no-banc", action="store_true", help="skip the BANC transmitter read")
    ap.add_argument("--quick", action="store_true", help="fewer rate-model iterations (smoke)")
    ap.add_argument("--plan-batch", default=None, metavar="DIR", help="write DIR/batch.sh, DIR/tree_state.json for the wedge-compass batch (no structure pass)")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--analyse", default=None, metavar="DIR", help="analyse the fetched batch in DIR (no structure pass)")
    # thread 6A: the three fixes are the defaults; --legacy (or the individual switches) reproduces the 5A behaviour
    ap.add_argument("--drive", default="current", choices=["current", "rate"],
                    help="how the forced EPG background / pulse enters the rate model: 'current' (u = f^-1(rate); the fix) or 'rate' (5A)")
    ap.add_argument("--reduction", default="with-direct", choices=["with-direct", "two-step"],
                    help="EPG-only reduction: keep the one-step EPG->EPG term in the loop gain (the fix) or the two-step terms only (5A)")
    ap.add_argument("--gain", default="per-cell", choices=["per-cell", "uniform"],
                    help="linearisation: per-cell gamma_i = f'(u_i) at the realised fixed point (the fix) or a uniform gamma (5A)")
    ap.add_argument("--legacy", action="store_true", help="the 5A behaviour: --drive rate --reduction two-step --gain uniform")
    ap.add_argument("--holds", action="store_true", help="add the 6A hold configurations (ExR6 / ER6 / ER4m -> PEN,EPG at 0)")
    # thread 6B (docs/audits/compass_local_recurrence.md)
    ap.add_argument("--local-recurrence", action="store_true", help="add the 6B configurations (the hold + same_type_gain 1 / + EPG->EPG undamped per type, each with GLNO = glu; and E alone)")
    ap.add_argument("--sigma", type=float, default=SIGMA_MV, help=f"input-noise width (mV) of the smoothed f-I for the fixed point, gains and slope bound (default the assumed {SIGMA_MV}; pass the MEASURED value from scripts/measure_lif_sigma.py)")
    ap.add_argument("--sigma-json", default=None, help="--predeclare (cx7): the sigma summary JSON whose headline / consequence are copied into the predeclaration")
    ap.add_argument("--batch", default="cx5", choices=["cx5", "cx6", "cx7"], help="which arm table --plan-batch / --analyse use")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds for --plan-batch (default: 0-3 for cx5, 0-4 for cx6 / cx7)")
    ap.add_argument("--name", default=None, help="batch name for --plan-batch (default: the --batch value)")
    ap.add_argument("--tree-state", default=None, metavar="DIR", help="rewrite DIR/tree_state.json only (batch.sh untouched)")
    ap.add_argument("--predeclare", default=None, metavar="DIR", help="write DIR/predeclared.json (stamped; archives any existing one) from the arm table and DIR/structure/structure.json")
    ap.add_argument("--structure-json", default=None, help="--predeclare: the structure pass to read the predictions from (default <DIR>/structure/structure.json)")
    ap.add_argument("--predictions-csv", nargs="+", default=None, metavar="STRUCTURE_JSON",
                    help="thread 6B: write <--out>/predictions.csv -- one row per (configuration, sigma) with the fixed point's PEN / EPG rates, "
                         "its confinement count on its own cells and the predicted bump -- from one or more structure.json files (no structure pass)")
    ap.add_argument("--validate-batch", default=None, metavar="DIR",
                    help="score this structure pass's predictions against an already-fetched batch (out/cx5): saturation rate vs measured bump_hz_post")
    a = ap.parse_args()
    if a.legacy:
        a.drive, a.reduction, a.gain = "rate", "two-step", "uniform"
    arms = {"cx5": BATCH_ARMS, "cx6": CX6_ARMS, "cx7": CX7_ARMS}[a.batch]
    seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else ((0, 1, 2, 3, 4) if a.batch in ("cx6", "cx7") else (0, 1, 2, 3))
    if a.tree_state:
        write_tree_state(Path(a.tree_state)); return
    if a.predeclare:
        d = Path(a.predeclare)
        sj = Path(a.structure_json) if a.structure_json else d / "structure" / "structure.json"
        if a.batch == "cx7":
            write_predeclaration_cx7(d, seeds, sj, Path(a.sigma_json) if a.sigma_json else None, name=a.name or a.batch, minutes=a.minutes); return
        write_predeclaration(d, arms, seeds, sj, name=a.name or a.batch, minutes=a.minutes); return
    if a.plan_batch:
        plan_batch(Path(a.plan_batch), seeds=seeds, minutes=a.minutes, name=a.name or a.batch, arms=arms); return
    if a.analyse:
        if a.batch == "cx7":
            analyse_batch_cx7(Path(a.analyse), seeds=seeds); return
        analyse_batch(Path(a.analyse), arms=arms, seeds=seeds); return
    if a.predictions_csv:
        rows = []
        for path in a.predictions_csv:
            st = json.load(open(path, encoding="utf-8"))
            sig = st.get("modes", {}).get("sigma_mV")
            for r in st["configs"]:
                rm, bc = r["rate_model"], r["bump_criterion"]["pulse"]
                pu, af = rm["pulse"], rm["after"]
                rows.append(dict(config=r["label"].split(":")[0].strip(), sigma_mV=sig, source=Path(path).as_posix(),
                                 max_lif_slope=st["modes"]["max_lif_slope"], local_EPG_EPG_mV=r["local_kernels"]["direct"],
                                 gamma_E_crit=bc["epg_recurrent"]["gamma_EPG_crit"], gamma_E_crit_with_relays=bc["with_relays"]["gamma_EPG_crit"],
                                 recurrence_saturation_hz=bc["epg_recurrent"]["saturation_hz"], gamma_EPG_driven=bc["gamma_EPG"],
                                 gamma_relay_PEN=bc["gamma_relay"]["PEN"], PEN_pulse_hz=pu["PEN_in"], u_PEN_pulse_mV=pu["u_PEN_in"],
                                 PEN_post_hz=af["PEN_in"], EPG_in_post_hz=af["epg_in"], EPG_out_post_hz=af["epg_out"],
                                 in_above_22=af["in_above_22"], out_above_22=af["out_above_22"], bump_after=rm["bump_after"],
                                 confined_bump_after=rm["confined_bump_after"], runaway_after=rm["runaway_after"]))
        df = pd.DataFrame(rows)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "predictions.csv", index=False)
        print(df.to_string(index=False)); print(f"-> {out / 'predictions.csv'} ({len(df)} rows)")
        return
    if a.validate_batch:
        validate_batch(Path(a.validate_batch), Path(a.out), drive=a.drive, reduction=a.reduction, gain=a.gain, sigma=a.sigma); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    c = connectome.load(verbose=False)
    c_glu, glu_dir, glu_table = cx_wedge.load_connectome({"GLNO": "glutamate"}, scratch=True, verbose=False)
    print(f"shipped cache: {c.n} cells, nnz {c.W.nnz}; GLNO=glutamate cache {glu_dir} (table {glu_table}): nnz {c_glu.W.nnz}; {time.time() - t0:.0f} s")
    cells = cx_wedge.compass_cells(c); cells_glu = cx_wedge.compass_cells(c_glu)
    res = dict(schema="flyverse.cx_ring_structure/1", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               generator="python " + " ".join(sys.argv), git=dict(head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
               status=git("status", "--porcelain").splitlines()), lif_defaults=json.loads(json.dumps(brain.LIFParams().__dict__, default=str)),
               shipped_cache_md5=hashlib.md5(c.W.tocsr().data.tobytes()).hexdigest(), glu_cache_dir=str(glu_dir), glu_table=glu_table)
    res["glno_evidence"] = glno_evidence(c, c_glu, with_banc=not a.no_banc)
    res["receptor_rows"] = receptor_rows()
    res["receptor_stage"] = receptor_stage_effects(c, cells)
    res["cap_effects"] = cap_effects(c, cells, brain.LIFParams(), brain.LIFParams(conn_cap=0.0))
    P = brain.LIFParams
    configs = [
        ("shipped", "LIFParams() on the shipped cache (receptor sign/abs, cap 60, same-type x0.1, GLNO sign 0)", c, P(), cells),
        ("a: GLNO=glutamate", "MaleCNS T-bars glu 51 % / ACh 37 % (conf 0.48 'unclear'); hemibrain name GLNO; BANC v888 glutamate 4/4", c_glu, P(), cells_glu),
        ("b: cap lifted (instrument)", "conn_cap 0 (global; the ring's EPG<->PEN / GLNO->PEN edges are 88-345 synapses, all capped at 60) -- a labelled instrument, not adoptable per type", c, P(conn_cap=0.0), cells),
        ("c: receptor sign+gain", "receptor_model 'sign+gain': the table's expression-tertile gain class x {low 0.5, mid 1, high 1.5} on sub-cap edges (the tertiles are data; the factor map is a parameter)", c, P(receptor_model="sign+gain"), cells),
        ("f: same-type damping off (instrument)", "same_type_gain 1 (global): EPG->EPG / PEN_a->PEN_a / PEN_b->PEN_b / Delta7->Delta7 at their connectome weight -- the damping is the hand rule, the synapses are data", c, P(same_type_gain=1.0), cells),
        ("a+c: GLNO=glu + sign+gain", "(a) and (c)", c_glu, P(receptor_model="sign+gain"), cells_glu),
        ("a+b: GLNO=glu + cap lifted", "(a) and (b): GLNO->PEN at its uncapped weight", c_glu, P(conn_cap=0.0), cells_glu),
        ("a+f: GLNO=glu + damping off", "(a) and (f)", c_glu, P(same_type_gain=1.0), cells_glu),
        ("c+f: sign+gain + damping off", "(c) and (f)", c, P(receptor_model="sign+gain", same_type_gain=1.0), cells),
        ("b+f: cap lifted + damping off (instrument)", "(b) and (f)", c, P(conn_cap=0.0, same_type_gain=1.0), cells),
    ]
    if a.holds:
        configs += hold_configs(c, c_glu, cells, cells_glu)
    if a.local_recurrence:
        configs += recurrence_configs(c, c_glu, cells, cells_glu)
    rows, mats = [], {}
    for label, ev, cc, p, cl in configs:
        cir = Circuit(cc, p, label, ev, cells=dict(cl))
        r, m = analyse(cir, drive=a.drive, reduction=a.reduction, gain=a.gain, sigma=a.sigma)
        rows.append(r)
        key = label.split(":")[0].replace(" ", "_").replace("+", "")
        for k, v in m["M16"].items():
            mats[f"M16_{key}_{k}"] = v
        for k, v in m["K"].items():
            mats[f"K_{key}_{k}"] = v
    res["configs"] = rows
    # ranking of the single and combined changes (6A: by the fixed point with the drive as a current, then the PEN rate
    # it reaches during the pulse, then the combined gamma_crit(k1) with the one-step term kept, then the DC margin)
    rk = []
    for r in rows:
        a_ = r["rate_model"]["after"]; pu_ = r["rate_model"]["pulse"]; tg_ = r["two_step_gain"]
        rk.append(dict(label=r["label"], bump=r["rate_model"]["bump_after"], runaway=r["rate_model"]["runaway_after"],
                       in_minus_out=a_["epg_in"] - a_["epg_out"], epg_in_after=a_["epg_in"], epg_out_after=a_["epg_out"],
                       pen_pulse_hz=pu_["PEN_in"], pen_after_hz=a_["PEN_in"], u_pen_pulse=pu_["u_PEN_in"],
                       gamma_crit_k1=tg_["gamma_crit_k1"], gamma_crit_k1_two_step_only=tg_["gamma_crit_k1_two_step_only"],
                       saturation_hz_k1=tg_["saturation_hz_k1"], supercritical_k1=tg_["supercritical_k1"],
                       predicted_bump=tg_["predicted_bump"], gamma_crit_k1_recurrent=tg_["gamma_crit_k1_recurrent"],
                       jacobian_lead_pulse=r["jacobian"]["pulse"]["leading_re"], k1_gain_pulse=r["jacobian"]["pulse"]["loop_gain_k1"],
                       lambda_1=r["fourier"]["net"][1], lambda_0=r["fourier"]["net"][0], direct_k1=r["fourier"]["direct"][1]))
    rk.sort(key=lambda d: (-int(d["bump"]), -d["pen_pulse_hz"], -d["in_minus_out"], d["gamma_crit_k1"], -d["u_pen_pulse"]))
    res["ranking"] = rk
    res["modes"] = dict(drive=a.drive, reduction=a.reduction, gain=a.gain, legacy=bool(a.legacy), sigma_mV=float(a.sigma), sigma_assumed_mV=SIGMA_MV,
                        u_for_rate={"10": u_for_rate(10.0, sigma=a.sigma), "50": u_for_rate(50.0, sigma=a.sigma)},
                        max_lif_slope=max_slope(sigma=a.sigma)[0], max_lif_slope_at_u=max_slope(sigma=a.sigma)[1], max_lif_slope_at_hz=max_slope(sigma=a.sigma)[2])
    res["wall_s"] = time.time() - t0
    with open(out / "structure.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez_compressed(out / "matrices.npz", **mats, epg_pos=cells["EPG"]["pos"], epg_body=cells["EPG"]["body"])
    with open(out / "evidence_glno.json", "w", encoding="utf-8") as f:
        json.dump(res["glno_evidence"], f, indent=1, default=str)
    write_md(res, out / "structure.md")
    print(f"-> {out / 'structure.json'}, {out / 'structure.md'}, {out / 'matrices.npz'}; {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
