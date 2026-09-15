"""CPU tests of the wedge-matrix / eigenmode code of scripts/cx_ring_structure.py on a synthetic ring (no connectome)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

crs = pytest.importorskip("cx_ring_structure")
from flyverse import brain  # noqa: E402


def circulant(profile):
    """16 x 16 circulant from a ring-distance profile p[0..8] (symmetric)."""
    n = 16
    d = crs.cx_wedge.ring_dist(np.arange(n), np.arange(n))
    return np.asarray(profile, float)[d]


def test_fourier_modes_of_a_circulant_are_its_eigenvalues():
    prof = np.array([5.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, 0.0])
    M = circulant(prof)
    fm = crs.fourier_modes(M, ks=(0, 1, 2, 8))
    # exact circulant eigenvalue: sum_d p(d) cos(2 pi k d / 16), each d in 1..7 counted twice
    for k in (0, 1, 2, 8):
        lam = prof[0] + 2 * sum(prof[d] * np.cos(2 * np.pi * k * d / 16) for d in range(1, 8)) + prof[8] * np.cos(np.pi * k)
        assert fm[k] == pytest.approx(lam, rel=1e-9)
    eig = np.sort(np.real(np.linalg.eigvals(M)))
    assert fm[0] == pytest.approx(eig[-1], rel=1e-9)          # the uniform mode is the largest for a positive local profile


def test_eig_modes_classifies_a_bump_mode_as_k1():
    # local excitation minus global inhibition: the k=1 mode leads, k=0 is negative
    prof = np.array([5.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, 0.0]) - 1.5
    M = circulant(prof)
    rows = crs.eig_modes(M, n_report=3)
    assert rows[0]["k"] == 1 and rows[0]["re"] == pytest.approx(crs.fourier_modes(M)[1], rel=1e-9)
    assert crs.fourier_modes(M)[0] < 0


def test_circuit_modes_two_population_ring_reduces_to_two_step_gain():
    """EPG (2 per wedge) -> relay (1 per wedge, local) -> EPG: the sub-circuit eigenvalue mu of the k=1 mode obeys
    mu^2 = lambda_1 of the two-step EPG x EPG matrix, so gamma_crit = 1 / (tau mu) = 1 / (tau sqrt(lambda_1))."""
    rng = np.random.default_rng(0)
    n = 16; nE = 32; nR = 16
    wedge_of = np.repeat(np.arange(n), 2)
    pos_r = np.arange(n)
    d_er = crs.cx_wedge.ring_dist(wedge_of, pos_r)            # EPG x relay
    A = np.zeros((nE + nR, nE + nR))
    A[:nE, nE:] = 3.0 * np.exp(-d_er / 1.5)                      # relay -> EPG, local
    A[nE:, :nE] = 2.0 * np.exp(-d_er.T / 1.5)                    # EPG -> relay, local
    A[:nE, nE:] -= 0.4                                           # global inhibition through the relay
    tau = 0.005
    rows, lead = crs.circuit_modes(A, slice(0, nE), wedge_of, tau)
    K = A[:nE, nE:] @ A[nE:, :nE]
    M16 = crs.cx_wedge.aggregate(K, wedge_of.astype(float), wedge_of.astype(float), 16, lambda x: np.asarray(np.round(x), int) % 16)
    lam1 = crs.fourier_modes(M16)[1]
    mu1 = lead[1]["re"]
    assert mu1 > 0 and lam1 > 0
    assert mu1 ** 2 == pytest.approx(lam1, rel=1e-6)
    assert lead[1]["gamma_crit"] == pytest.approx(1.0 / (tau * np.sqrt(lam1)), rel=1e-6)
    assert rows[0]["k"] == 1


def test_offset_profile_recovers_a_one_wedge_shift():
    pos_pre = np.arange(16, dtype=float); pos_post = np.arange(16, dtype=float)
    B = np.zeros((16, 16))
    for i in range(16):
        B[(i + 1) % 16, i] = 2.0                                  # post = pre + 1
    offs, prof = crs.offset_profile(B, pos_post, pos_pre)
    assert prof[offs.index(1)] == pytest.approx(2.0)
    assert sum(abs(v) for o, v in zip(offs, prof) if o != 1) == 0.0


def test_rate_fixed_point_holds_a_bump_on_a_synthetic_ring_and_not_without_local_excitation():
    """A threshold-linear ring with local relay excitation and global inhibition keeps the 4-wedge bump after release;
    the same ring with the local excitation removed relaxes back to the uniform background."""
    p = brain.LIFParams()
    n = 16; nE = 32; nR = 16
    wedge_of = np.repeat(np.arange(n), 2)
    d_er = crs.cx_wedge.ring_dist(wedge_of, np.arange(n))
    A = np.zeros((nE + nR, nE + nR))
    A[:nE, nE:] = 40.0 * np.exp(-d_er / 1.2) - 8.0               # relay -> EPG: local excitation, global inhibition
    A[nE:, :nE] = 20.0 * np.exp(-d_er.T / 1.2)                   # EPG -> relay: local
    inside = np.isin(wedge_of, [0, 1, 2, 3])
    st = crs.rate_fixed_point(A, nE, inside, p, iters=1500)
    r_after = st["after"][0]
    assert r_after[:nE][inside].mean() > 3 * r_after[:nE][~inside].mean()
    assert r_after[:nE][inside].mean() > 30.0
    A0 = A.copy(); A0[:nE, nE:] = -8.0
    st0 = crs.rate_fixed_point(A0, nE, inside, p, iters=1500)
    r0 = st0["after"][0]
    assert abs(r0[:nE][inside].mean() - r0[:nE][~inside].mean()) < 1e-3
