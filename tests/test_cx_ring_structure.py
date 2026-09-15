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


# ---------------------------------------------------------------- thread 6A: the three fixes (compass_dc_balance.md)
def test_u_for_rate_inverts_the_smoothed_f_I():
    """Fix 1: the forced drive enters as a current, u = f^-1(rate) -- derived from the shipped LIF f-I, not hardcoded."""
    p = brain.LIFParams()
    for hz in (5.0, 10.0, 25.0, 50.0, 120.0):
        u = crs.u_for_rate(hz, p)
        assert crs.cx_wedge.lif_fi(np.array([u]), p, crs.SIGMA_MV)[0] == pytest.approx(hz, rel=1e-6)
    assert crs.u_for_rate(10.0) == pytest.approx(6.6276, abs=1e-3)      # the audit's 6.63 mV
    assert crs.u_for_rate(50.0) == pytest.approx(11.9933, abs=1e-3)     # and 11.99 mV
    assert crs.u_for_rate(10.0) < crs.u_for_rate(50.0)


def test_lif_fi_prime_matches_a_finite_difference_and_is_bounded():
    """Fix 3: gamma_i = f'(u_i). Integration by parts must agree with a central difference of cx_wedge.lif_fi away
    from threshold, and -- unlike the deterministic f-I, whose slope diverges at threshold -- must be bounded."""
    p = brain.LIFParams()
    for u, tol in ((9.0, 0.10), (12.0, 0.03), (20.0, 0.02), (40.0, 0.02), (80.0, 0.02)):
        h = 1e-3
        fd = (crs.cx_wedge.lif_fi(np.array([u + h]), p, crs.SIGMA_MV)[0] - crs.cx_wedge.lif_fi(np.array([u - h]), p, crs.SIGMA_MV)[0]) / (2 * h)
        # the 15-node quadrature cx_wedge.lif_fi uses is itself coarse near threshold, so the tolerance widens there
        assert crs.lif_fi_prime(np.array([u]), p)[0] == pytest.approx(fd, rel=tol)
    dmax, u_peak, f_peak = crs.max_slope(p)
    assert 7.0 < dmax < 10.0 and 7.0 < u_peak < 12.0 and 15.0 < f_peak < 40.0
    assert crs.lif_fi_prime(np.array([200.0]), p)[0] < crs.lif_fi_prime(np.array([40.0]), p)[0] < dmax
    assert crs.lif_fi_prime(np.array([-5.0]), p)[0] == pytest.approx(0.0, abs=1e-6)


def test_rate_at_gain_is_the_saturation_rate_and_is_nan_above_the_maximum_slope():
    """A supercritical loop grows until f'(u) has fallen back to gamma_crit; a gamma_crit above the LIF's maximum
    slope is never reached at any operating point (the shipped ring's 33.4 Hz/mV)."""
    p = brain.LIFParams()
    assert crs.rate_at_gain(3.35, p) == pytest.approx(144.0, rel=0.02)      # the F family's undamped EPG -> EPG ring
    assert crs.rate_at_gain(3.01, p) == pytest.approx(160.0, rel=0.02)      # the cf one
    assert np.isnan(crs.rate_at_gain(33.4, p))                              # the shipped (x0.1) one: never
    assert crs.rate_at_gain(2.0, p) > crs.rate_at_gain(3.35, p)             # lower critical gain -> higher saturation
    assert np.isnan(crs.rate_at_gain(0.0, p)) and np.isnan(crs.rate_at_gain(float("inf"), p))


def test_gamma_crit_combined_keeps_the_one_step_term():
    """Fix 2: gamma tau d + gamma^2 tau^2 lambda = 1. It must reduce to 5A's two values in the two limits and lie
    between them when both terms are present."""
    tau = 0.005
    lam, d = 10019.0, 6.0
    assert crs.gamma_crit_combined(lam, 0.0, tau) == pytest.approx(1.0 / (tau * np.sqrt(lam)), rel=1e-9)
    assert crs.gamma_crit_combined(0.0, d, tau) == pytest.approx(1.0 / (tau * d), rel=1e-9)
    g = crs.gamma_crit_combined(lam, d, tau)
    assert g < crs.gamma_crit_combined(lam, 0.0, tau)                       # keeping a positive term lowers the bar
    assert g * tau * d + (g * tau) ** 2 * lam == pytest.approx(1.0, rel=1e-9)
    assert crs.gamma_crit_combined(-1.0, -1.0, tau) == float("inf")


def test_forced_drive_enters_as_a_current_and_the_forced_rate_is_a_floor():
    """Fix 1 in the fixed point. With A = 0: 'current' puts the driven cells at u = f^-1(rate) and their rate at the
    forced rate; 'rate' (5A) leaves u = 0 while they 'fire', which is how gamma_EPG became 0 by construction."""
    p = brain.LIFParams()
    nE = 4
    A = np.zeros((6, 6))
    inside = np.array([True, True, False, False])
    cur = crs.rate_fixed_point(A, nE, inside, p, iters=400, drive="current")
    r, u = cur["pulse"]
    assert u[:nE][inside][0] == pytest.approx(crs.u_for_rate(50.0, p), rel=1e-6)
    assert u[:nE][~inside][0] == pytest.approx(crs.u_for_rate(10.0, p), rel=1e-6)
    assert r[:nE][inside][0] == pytest.approx(50.0, rel=1e-3) and r[:nE][~inside][0] == pytest.approx(10.0, rel=1e-3)
    old = crs.rate_fixed_point(A, nE, inside, p, iters=400, drive="rate")
    r0, u0 = old["pulse"]
    assert np.allclose(u0, 0.0) and r0[:nE][inside][0] == pytest.approx(50.0, rel=1e-3)
    # 5A's gamma_EPG = 0 by construction: u = 0 is 3.5 sigma below threshold (0.009 Hz/mV against a maximum of 8.3),
    # and in the real circuit 5A's driven EPG sat at u = -22 to -68 mV, where it is exactly 0
    assert crs.lif_fi_prime(u0[:nE], p)[0] < 0.01
    assert crs.lif_fi_prime(np.array([-22.0]), p)[0] == 0.0
    # the floor: strong inhibition cannot push a FORCED cell below its driven rate (FlyBrain forces the spikes)
    A2 = np.zeros((6, 6)); A2[4, :nE] = 20.0; A2[:nE, 4] = -200.0        # cell 4 is driven by the EPGs and inhibits them
    r2, u2 = crs.rate_fixed_point(A2, nE, inside, p, iters=400, drive="current")["pulse"]
    assert r2[:nE][inside][0] == pytest.approx(50.0, rel=1e-3)
    assert u2[:nE][inside][0] < 0.0                                         # and it is held far below threshold there
    r3, _ = crs.rate_fixed_point(A2, nE, inside, p, iters=400, drive="current-nofloor")["pulse"]
    assert r3[:nE][inside][0] < 0.7 * r2[:nE][inside][0]                     # the literal reading loses the forced drive


def test_jacobian_modes_is_the_true_linearisation_and_the_floor_zeroes_a_forced_cell_s_gain():
    """Fix 3: J = diag(f'(u_i)) tau A, and a cell whose rate is the forced floor has no gain."""
    p = brain.LIFParams()
    rng = np.random.default_rng(0)
    n, nE = 8, 4
    A = rng.normal(size=(n, n)) * 5.0
    u = np.array([30.0, 30.0, 30.0, 30.0, 12.0, 12.0, 12.0, 12.0])
    tau = 0.005
    wedge_of = np.arange(nE) % 16
    jm = crs.jacobian_modes(A, u, tau, slice(0, nE), wedge_of, p)
    J = (crs.lif_fi_prime(u, p)[:, None] * tau) * A
    assert jm["leading_re"] == pytest.approx(float(np.real(np.linalg.eigvals(J)).max()), rel=1e-9)
    assert jm["spectral_radius"] == pytest.approx(float(np.abs(np.linalg.eigvals(J)).max()), rel=1e-9)
    assert jm["gamma"][0] > 0
    floor = np.zeros(n); floor[:nE] = 1e6                                   # every EPG pinned at its forced rate
    jf = crs.jacobian_modes(A, u, tau, slice(0, nE), wedge_of, p, floor=floor)
    assert np.all(jf["gamma"][:nE] == 0.0) and np.all(jf["gamma"][nE:] > 0)


def test_hold_params_appends_one_edges_hold_and_leaves_the_defaults_alone():
    base = brain.LIFParams()
    p = crs.hold_params(crs.HOLD_PRE, crs.HOLD_POST, base)
    assert base.type_path_gain is None                                      # the shipped default is untouched
    assert p.type_path_gain[:-1] == list(brain.DEFAULT_TYPE_PATH_GAIN)
    assert p.type_path_gain[-1] == (crs.HOLD_PRE, crs.HOLD_POST, 0.0)
    assert p.same_type_gain == base.same_type_gain and p.conn_cap == base.conn_cap


def test_the_cx6_arm_table_is_callable_at_five_runs_and_its_holds_parse():
    """The predeclared family must be satisfiable at 5 v 5 (m x p_floor <= 0.05) and every arm's hold must parse."""
    from flyverse.interp import common
    assert len(crs.PRIMARIES_CX6) <= 6
    assert len(crs.PRIMARIES_CX6) * common.p_floor(5, 5) <= 0.05
    labels = [a[0] for a in crs.CX6_ARMS]
    assert labels[0] == "S" and "H3" in labels and len(set(labels)) == len(labels)
    for label, desc, gains, glu, extra, cls in crs.CX6_ARMS:
        for i, x in enumerate(extra):
            if x == "--hold-edges":
                assert len(crs.cx_wedge.parse_hold_edges([extra[i + 1]])) == 1
        assert gains in ("1:1", "2:15")


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


# ------------------------------------------------------------------------------------------------ thread 6B
# (docs/audits/compass_local_recurrence.md): sigma as an argument, the two-instrument configurations, the cx7 table,
# the driven-tile distance and the fixed point's own confinement count.
def test_sigma_is_an_argument_and_the_slope_bound_falls_with_it():
    """The maximum slope of the smoothed f-I is a property of the assumed sigma (6A skeptic R1): higher sigma, lower
    bound; the default is still the assumed 2.0 mV, and the MEASURED 4.6 mV gives a bound below it."""
    d2, u2, f2 = crs.max_slope(sigma=2.0)
    d46, u46, f46 = crs.max_slope(sigma=4.6)
    d1, _, _ = crs.max_slope(sigma=1.0)
    assert crs.max_slope() == crs.max_slope(sigma=crs.SIGMA_MV) and crs.SIGMA_MV == 2.0
    assert d1 > d2 > d46 > 0
    assert 7.5 < d2 < 8.7                                                   # the 61-node estimate (8.27) of the 8.00
    assert 4.5 < d46 < 6.5
    # rate_at_gain follows the same sigma: a gain reachable at 2 mV may not be at 4.6 mV
    assert np.isfinite(crs.rate_at_gain(3.35, sigma=2.0)) and np.isfinite(crs.rate_at_gain(3.35, sigma=4.6))
    assert np.isnan(crs.rate_at_gain(7.5, sigma=4.6)) and np.isfinite(crs.rate_at_gain(7.5, sigma=2.0))
    # u_for_rate inverts the f-I at the sigma it is given
    for s in (2.0, 4.6):
        assert abs(crs.cx_wedge.lif_fi(np.array([crs.u_for_rate(10.0, sigma=s)]), sigma=s)[0] - 10.0) < 1e-5


def test_recurrence_configs_compose_the_hold_with_each_recurrence_and_e_alone(monkeypatch):
    """H3F carries the hold with same_type_gain 1; H3E carries the hold AND the EPG->EPG x10 entry with the shipped
    same_type_gain 0.1; the GLNO variants use the glutamate connectome; E carries the x10 entry without the hold."""
    calls = []
    fake_c, fake_glu = object(), object()
    out = crs.recurrence_configs(fake_c, fake_glu, {"cells": 1}, {"cells": 2})
    labels = [o[0].split(":")[0] for o in out]
    assert labels == ["H3F", "H3E", "H3FG", "H3EG", "E"]
    by = {o[0].split(":")[0]: o for o in out}
    hold = (crs.HOLD_PRE, crs.HOLD_POST, 0.0)
    gain = (r"^EPG$", r"^EPG$", 10.0)
    assert by["H3F"][3].same_type_gain == 1.0 and hold in by["H3F"][3].type_path_gain and gain not in by["H3F"][3].type_path_gain
    assert by["H3E"][3].same_type_gain == 0.1 and hold in by["H3E"][3].type_path_gain and gain in by["H3E"][3].type_path_gain
    assert by["E"][3].same_type_gain == 0.1 and hold not in by["E"][3].type_path_gain and gain in by["E"][3].type_path_gain
    assert by["H3FG"][2] is fake_glu and by["H3EG"][2] is fake_glu and by["H3F"][2] is fake_c and by["H3E"][2] is fake_c
    for lab in ("H3F", "H3E", "H3FG", "H3EG", "E"):
        assert by[lab][3].type_path_gain[:len(brain.DEFAULT_TYPE_PATH_GAIN)] == list(brain.DEFAULT_TYPE_PATH_GAIN)
        assert "INSTRUMENT" in by[lab][1]


def test_config_for_arm_reads_the_edge_gain_flag():
    arm = ("H3E", "desc", "1:1", False, ["--receptor-model", "shipped", "--hold-edges", crs.HOLD3, "--edge-gain", crs.EPG_EPG_GAIN], "cls")
    c, cglu = object(), object()
    label, ev, cc, p, cells = crs.config_for_arm(arm, c, cglu, {"a": 1}, {"b": 2})
    assert cc is c and cells == {"a": 1} and label == "H3E"
    assert p.type_path_gain[-2] == (crs.HOLD_PRE, crs.HOLD_POST, 0.0) and p.type_path_gain[-1] == (r"^EPG$", r"^EPG$", 10.0)
    assert p.same_type_gain == 0.1
    armF = ("H3F", "desc", "1:1", True, ["--receptor-model", "shipped", "--hold-edges", crs.HOLD3, "--lif", "same_type_gain=1"], "cls")
    _, _, cc2, p2, cells2 = crs.config_for_arm(armF, c, cglu, {"a": 1}, {"b": 2})
    assert cc2 is cglu and cells2 == {"b": 2} and p2.same_type_gain == 1.0 and p2.type_path_gain[-1] == (crs.HOLD_PRE, crs.HOLD_POST, 0.0)


def test_the_cx7_arm_table_and_its_family_are_satisfiable_at_five_runs():
    from flyverse.interp import common
    assert len(crs.PRIMARIES_CX7) == 5 and len(crs.PRIMARIES_CX7) * common.p_floor(5, 5) <= 0.05
    labels = [a[0] for a in crs.CX7_ARMS]
    assert labels[:2] == ["S", "H3"] and {"H3F", "H3E", "H3FG", "H3EG", "H3G", "F"} <= set(labels) and len(set(labels)) == len(labels)
    assert set(crs.MAGNITUDES_CX7).isdisjoint(crs.PRIMARIES_CX7) and "centre_dist_t5" in crs.PRIMARIES_CX7
    for label, desc, gains, glu, extra, cls in crs.CX7_ARMS:
        assert gains == "1:1"
        assert glu == label.endswith("G")
        for i, x in enumerate(extra):
            if x == "--hold-edges":
                assert crs.cx_wedge.parse_hold_edges([extra[i + 1]]) == [(crs.HOLD_PRE, crs.HOLD_POST, 0.0)]
            if x == "--edge-gain":
                assert crs.cx_wedge.parse_edge_gains([extra[i + 1]]) == [(r"^EPG$", r"^EPG$", 10.0)]
        assert ("--edge-gain" in extra) == label.startswith("H3E")
        assert ("same_type_gain=1" in extra) == (label.startswith("H3F") or label == "F")
        assert ("--hold-edges" in extra) == label.startswith("H3")
    assert crs.arm_tables(crs.CX7_ARMS) == (crs.PRIMARIES_CX7, crs.SECONDARIES_CX7)


def test_ring_distance_and_the_driven_tile_rule():
    assert crs.ring_distance(1.5, 1.5) == 0.0
    assert crs.ring_distance(0.92, 1.5) == pytest.approx(0.58)
    assert crs.ring_distance(15.5, 1.5) == pytest.approx(2.0)                # wraps: 15.5 -> 1.5 is 2 wedges the short way
    assert crs.ring_distance(9.5, 1.5) == 8.0                                # the far side
    assert crs.ring_distance(5.9, 1.5) > crs.DRIVEN_TILE_TOL                 # 6A's H3 hump (5.7-6.1) fails the rule
    assert crs.ring_distance(0.92, 1.5) <= crs.DRIVEN_TILE_TOL               # 6A's H3G seed 1 passes it


def test_summarise_state_counts_the_ledger_confinement_on_the_fixed_point():
    nE = 46
    wedge_of = np.arange(nE) % 16
    inside = np.isin(wedge_of, [0, 1, 2, 3])
    r = np.full(nE, 10.0); r[inside] = 150.0
    d = crs.summarise_state(r, np.zeros(nE), {}, inside, wedge_of, nE)
    assert d["in_above_22"] == int(inside.sum()) and d["out_above_22"] == 0 and d["confined_by_ledger_rule"]
    r2 = np.full(nE, 60.0); r2[inside] = 150.0                              # the 6A H3 regime: every off-block cell above 22 Hz
    d2 = crs.summarise_state(r2, np.zeros(nE), {}, inside, wedge_of, nE)
    assert d2["out_above_22"] == int((~inside).sum()) and not d2["confined_by_ledger_rule"]


# ------------------------------------------------------------------------------------------------ thread 6B:
# scripts/measure_lif_sigma.py (the input-noise measurement). Pure functions only -- no connectome, no FlyBrain.
mls = pytest.importorskip("measure_lif_sigma")


def test_window_stats_drops_the_transient_and_measures_the_sd_of_g_and_of_the_free_membrane():
    rng = np.random.default_rng(0)
    steps, cells = 400, 3
    g = rng.normal(loc=-7.0, scale=3.0, size=(steps, cells))
    v = rng.normal(loc=-50.0, scale=1.5, size=(steps, cells))
    g[:100] += 100.0                                                    # a transient the window must drop
    v[:100] += 100.0
    refrac = np.zeros((steps, cells)); spikes = np.zeros((steps, cells))
    refrac[:, 2] = 1.0                                                  # cell 2 is refractory throughout
    spikes[150, 1] = 1.0                                                # cell 1 spikes once
    st = mls.window_stats(v, g, refrac, spikes, transient_steps=100)
    assert st["n_steps"] == 300 and st["n_cells"] == 3
    assert st["mean_g_mV"][0] == pytest.approx(-7.0, abs=0.6) and st["sigma_g_mV"][0] == pytest.approx(3.0, rel=0.25)
    assert st["sigma_v_free_mV"][0] == pytest.approx(1.5, rel=0.25)
    assert st["spiked"] == [False, True, False]
    assert st["free_fraction"][0] == 1.0 and st["free_fraction"][2] == 0.0
    assert not np.isfinite(st["sigma_v_free_mV"][2])                    # a wholly refractory cell has no free sample
    assert st["rate_hz"][1] == pytest.approx(1.0 / (300 * 0.5e-3), rel=1e-6)


def test_summarise_separates_the_never_spiked_cells():
    """The headline restricts to never-spiked cells because reset-to-threshold clips a spiking cell's membrane SD."""
    st = dict(n_cells=4, n_steps=10, mean_g_mV=[1.0, 2.0, 3.0, 4.0], sigma_g_mV=[10.0, 11.0, 12.0, 13.0],
              mean_v_free_mV=[-50.0] * 4, sigma_v_free_mV=[2.2, 2.3, 5.0, 6.0], free_fraction=[1.0] * 4,
              rate_hz=[100.0, 80.0, 0.0, 0.0], spiked=[True, True, False, False])
    s = mls.summarise(st)
    assert s["n_cells"] == 4 and s["n_never_spiked"] == 2
    assert s["sigma_v_free_mV"]["median"] == pytest.approx(3.65)                    # all four cells
    assert s["sigma_v_free_mV_never_spiked"]["median"] == pytest.approx(5.5)       # the two sub-threshold ones
    assert s["sigma_v_free_mV_never_spiked"]["n"] == 2
    assert s["rate_hz"]["max"] == 100.0


def test_summarise_dir_pools_the_shipped_arm_and_writes_the_table():
    """The pooled headline is the never-spiked relay membrane SD of arm S only, and the per-seed CSV rule-28 file is written."""
    def doc(label, seed, sv, spiked, sg=12.0):
        pc = dict(n_cells=len(sv), n_steps=100, mean_g_mV=[-7.0] * len(sv), sigma_g_mV=[sg] * len(sv),
                  mean_v_free_mV=[-50.0] * len(sv), sigma_v_free_mV=list(sv), free_fraction=[1.0] * len(sv),
                  rate_hz=[0.0] * len(sv), spiked=list(spiked))
        g = {"summary": mls.summarise(pc), "per_cell": pc}
        return dict(label=label, seed=seed, windows={"settle": {k: g for k in ("EPG", "PEN", "PEG", "EPGt")}})
    import json, tempfile
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    for d in (doc("S", 0, [4.0, 5.0], [False, False]), doc("S", 1, [6.0, 7.0], [False, True]),
              doc("H3", 0, [1.0, 1.0], [True, True])):                      # the held arm must not enter the headline
        (tmp_path / f"sigma_{d['label']}_s{d['seed']}.json").write_text(json.dumps(d), encoding="utf-8")
    out = mls.summarise_dir(tmp_path)
    h = out["headline"]
    assert h["seeds"] == [0, 1] and h["n_relay_cells"] == 9               # S's never-spiked PEN + PEG + EPGt cells (2+2+2 and 1+1+1)
    assert h["sigma_v_relays_subthreshold_mV"] == pytest.approx(5.0)      # median of 4,5,6 over three groups
    assert out["measured_sigma_mV"] == 5.0 and out["measured_sigma_g_mV"] == 12.0
    assert "never measured" not in out["consequence"] and "2.0" in out["consequence"]
    table = (tmp_path / "sigma_table.csv").read_text().splitlines()
    assert table[0].startswith("label,seed,window,group,n,never_spiked,sigma_v_median")
    assert len(table) == 1 + 3 * 4                                        # 3 runs x 4 groups, one window
    assert (tmp_path / "sigma_summary.json").is_file()
    tmp.cleanup()
