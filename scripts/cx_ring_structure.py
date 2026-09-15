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

KNOWN DEFECTS OF THIS STRUCTURE PASS -- read the ranking it prints with these in mind. Found by the independent skeptic
pass of 2026-09-15 and recorded in docs/audits/compass_ring_mechanism.md sections 1.3 and 3.3; NOT fixed here (this note
is documentation only -- no behaviour has changed):

  1. The EPG-only two-step reduction DROPS THE ONE-STEP EPG -> EPG TERM. M16['net'] is PEN + PEG + Delta7 + Ring only;
     the 'direct' one-step matrix is computed, printed (in mV, in the same table row as the mV^2 lambdas) and then never
     used. Its k = 1 ring-Fourier coefficient is +6.00 mV damped (gamma_crit = 1 / (tau d_1) = 33.4 Hz/mV, never
     reached) and +59.76 mV undamped (3.35 Hz/mV, reached by the LIF at any u below ~34 mV), so the undamped
     EPG -> EPG ring is supercritical on its own and predicts, to 4-7 %, the bump the same_type_gain = 1 arms actually
     hold (145 Hz predicted for f, 161 Hz for cf, against 151-156 and 156-158 observed) -- the bump this pass misses.
  2. rate_fixed_point ENTERS THE FORCED BACKGROUND AS A RATE, NOT AS A CURRENT (r = f(tau A r) + forced), so a driven
     EPG sits at u = -22 to -68 mV while 'firing' at 10-50 Hz and gamma_EPG = f'(u_EPG) = 0 BY CONSTRUCTION -- no
     recurrent EPG term, direct or two-step, can ever engage. Put the same drive in as a current (10 Hz <-> 6.63 mV,
     50 Hz <-> 11.99 mV) and the same deterministic model, at the same sigma = 2 mV, separates the families: shipped
     returns to 14.3 / 15.3 Hz after release (no bump), undamped runs away to 264 / 272 Hz.

Consequence: the fixed point this script ranks configurations by has gamma = 0 on every compass cell except Delta7, so
THE RANKING IT PRINTS IS COMPUTED AT A ZERO-GAIN STATE. At that state the true Jacobian diag(gamma_i) tau A is stable in
every mode (leading eigenvalue +0.045 at background, +0.079 during the pulse), against +3.55 under a uniform gamma = 6,
and the ranking puts the one arm that does hold a bump (f) last-equal. Fix both before the ranking is relied on again.
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
def rate_fixed_point(A_sub, nE, inside, p: brain.LIFParams, background_hz=10.0, pulse_hz=40.0, sigma=2.0, iters=4000, alpha=0.05):
    """cx_wedge.rate_model on an explicit sub-circuit matrix: r = f(tau_syn A r + forced), LIF f-I smoothed over
    sigma mV of input noise, background -> pulse -> release, each relaxed to a fixed point."""
    tau = p.tau_syn / 1000.0
    n = A_sub.shape[0]
    forced = np.zeros(n); forced[:nE] = background_hz

    def relax(f, r):
        for _ in range(iters):
            u = tau * (A_sub @ r)
            r = (1 - alpha) * r + alpha * (cx_wedge.lif_fi(u, p, sigma) + f)
        return r, u

    r_bg, u_bg = relax(forced, forced.copy())
    fp = forced.copy(); fp[:nE][inside] += pulse_hz
    r_pulse, u_pulse = relax(fp, r_bg)
    r_after, u_after = relax(forced, r_pulse)
    return dict(background=(r_bg, u_bg), pulse=(r_pulse, u_pulse), after=(r_after, u_after))


def summarise_state(r, u, groups_slices, inside, wedge_of, nE):
    e = r[:nE]
    d = dict(epg_in=float(e[inside].mean()), epg_out=float(e[~inside].mean()), epg_in_min=float(e[inside].min()),
             epg_out_max=float(e[~inside].max()), profile=[float(e[wedge_of == w].mean()) for w in range(16)])
    for g, sl in groups_slices.items():
        d[g] = float(r[sl].mean()) if sl.stop > sl.start else 0.0
        d[f"u_{g}"] = float(u[sl].mean()) if sl.stop > sl.start else 0.0
    # DC balance on the bump's own PEN / EPG and elsewhere
    return d


# ------------------------------------------------------------------------------------------------ analysis of one circuit
def analyse(cir: Circuit, log=print) -> dict:
    p = cir.p
    tau = p.tau_syn / 1000.0
    res = dict(label=cir.label, evidence=cir.evidence, lif=dict(conn_cap=p.conn_cap, same_type_gain=p.same_type_gain,
               receptor_model=p.receptor_model, receptor_net_rule=p.receptor_net_rule, w_syn=p.w_syn,
               input_norm_ref=p.input_norm_ref, input_norm_alpha=p.input_norm_alpha),
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
    res["two_step_gain"] = dict(lambda_k1_net=lam1, lambda_k0_net=lam0, lambda_k1_tuned=res["fourier"]["net_tuned"][1],
                                gamma_crit_k1=float(1.0 / (tau * np.sqrt(lam1))) if lam1 > 0 else float("inf"),
                                gamma_crit_k0=float(1.0 / (tau * np.sqrt(lam0))) if lam0 > 0 else float("inf"),
                                loop_gain_k1_at_6=float(36.0 * tau * tau * lam1), loop_gain_k0_at_6=float(36.0 * tau * tau * lam0))
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
    st = rate_fixed_point(A_sub, nE, inside, p)
    res["rate_model"] = {k: summarise_state(r, u, sl, inside, cir.wedge_of, nE) for k, (r, u) in st.items()}
    a = res["rate_model"]["after"]
    res["rate_model"]["bump_after"] = bool(a["epg_in"] > 2 * a["epg_out"] and a["epg_in"] > 15.0)
    # PEN DC margin during the pulse: mean input of the PENs whose glomerulus lies in the driven wedges
    pen_in = np.isin(np.asarray(np.round(cir.pos["PEN"]), int) % 16, [0, 1, 2, 3])
    for k, (r, u) in st.items():
        up = u[sl["PEN"]]
        res["rate_model"][k]["u_PEN_in"] = float(up[pen_in].mean()); res["rate_model"][k]["u_PEN_out"] = float(up[~pen_in].mean())
        res["rate_model"][k]["PEN_in"] = float(r[sl["PEN"]][pen_in].mean()); res["rate_model"][k]["PEN_out"] = float(r[sl["PEN"]][~pen_in].mean())
        res["rate_model"][k]["u_EPG_in"] = float(u[:nE][inside].mean()); res["rate_model"][k]["u_EPG_out"] = float(u[:nE][~inside].mean())
    # decomposition of the mean input of the driven-wedge PEN / EPG at each fixed point by presynaptic group (mV), and the
    # ring-neuron types that carry the Ring term (their fixed-point rates and their share)
    dec = {}
    rt = np.array(cir.cells["Ring"]["label"])
    for k, (r, u) in st.items():
        d = {}
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
    log(f"[{cir.label}] k1 net {lam1:+.0f} (PEN {res['fourier']['PEN'][1]:+.0f}, D7 {fD[1]:+.0f}, Ring {fR[1]:+.0f}); k0 net {lam0:+.0f}; "
        f"gamma_crit(k1) {res['two_step_gain']['gamma_crit_k1']:.2f} Hz/mV; circuit lead k1 mu {lead.get(1, {}).get('re', float('nan')):+.1f} "
        f"(gamma_crit {lead.get(1, {}).get('gamma_crit', float('nan')):.2f}); rate model after: in {a['epg_in']:.1f} out {a['epg_out']:.1f} "
        f"PEN {a['PEN']:.1f} D7 {a['Delta7']:.1f} Ring {a['Ring']:.2f} GLNO {a['GLNO']:.1f} u_PEN_in {a['u_PEN_in']:+.2f} mV {'BUMP' if res['rate_model']['bump_after'] else ''}")
    return res, dict(K=K, M16=M16, Kp=Kp, Mp16=Mp16)


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


def plan_batch(out_dir: Path, seeds=(0, 1, 2, 3), minutes=30, name="cx5"):
    """Writes out/cx5/batch.sh (ONE cluster_run.py submission; one job per seed x GLNO condition, the arms of a job run
    sequentially, blocks fam_s<seed>), the predeclaration skeleton and tree_state.json."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rel = out_dir.as_posix()
    jobs = []
    for s in seeds:
        for glu in (False, True):
            arms = [a for a in BATCH_ARMS if a[3] == glu]
            parts, sts = [], []
            for i, (label, _, gains, _, extra, _) in enumerate(arms):
                stem = f"{rel}/{label}_s{s}"
                cmd = (f"python scripts/cx_wedge.py --no-structure --sim {gains} --ledger --seed {s} --arm {label} --block fam_s{s} "
                       + ("--nt-override GLNO=glutamate " if glu else "") + " ".join(extra) + f" --sim-out {stem}.json > {stem}.txt 2>&1; s{i}=\\$?; tail -3 {stem}.txt")
                parts.append(cmd); sts.append(f"s{i}")
            line = (f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && "
                    + "; ".join(parts) + f"; exit \\$(({' | '.join(sts)}))")     # bash arithmetic: bare names, no $ (a $s0 in the double-quoted line would expand at submission)
            jobs.append(dict(seed=s, glu=glu, arms=[a[0] for a in arms], line=line))
    call = (f"python scripts/cluster_run.py --name {name} --minutes {minutes} --arm-block fam " + " ".join('"' + j["line"] + '"' for j in jobs)
            + f" --fetch {rel}/")
    sh = (f"#!/bin/bash\n# ONE submission: {len(jobs)} jobs = {len(seeds)} seeds x 2 GLNO conditions; each job runs its {len(BATCH_ARMS) // 2} arms sequentially; "
          f"blocks fam_s<seed> (every seed's arms on one target). Generated by scripts/cx_ring_structure.py --plan-batch on {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
          f"{call} 2>&1 | tee {rel}/client_stdout.txt\n")
    (out_dir / "batch.sh").write_text(sh, encoding="utf-8", newline="\n")
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
                     "flyverse/body.py, flyverse/senses.py, scripts/probe_vnc_drive.py, tests/test_*.py and docs/audits/level_controls.md are another thread's "
                     "concurrent edits; cx_wedge.py --sim builds FlyBrain without a world, so senses.py / body.py are not on the simulated path.")
    (out_dir / "tree_state.json").write_text(json.dumps(tree, indent=1), encoding="utf-8")
    print(f"{len(jobs)} jobs -> {out_dir / 'batch.sh'}; tree_state.json ({len(sha)} files hashed)")
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


def analyse_batch(out_dir: Path, ref="S"):
    """Tables, verdicts and the per-seed scatter from the fetched out/cx5/<arm>_s<seed>.json rows (CPU)."""
    from flyverse.interp import common
    out_dir = Path(out_dir); an = out_dir / "analysis"; an.mkdir(parents=True, exist_ok=True)
    rows = []
    problems = []
    for label, desc, gains, glu, extra, cls in BATCH_ARMS:
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
                         profile_post=r.get("wedge_profile_post"))
                for k in PRIMARIES + SECONDARIES:
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
                rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows"); return None
    df = df.sort_values(["arm", "seed"], key=lambda s: s.map({a[0]: i for i, a in enumerate(BATCH_ARMS)}) if s.name == "arm" else s).reset_index(drop=True)
    counts = df.groupby("arm").seed.size()
    expected = {a[0] for a in BATCH_ARMS}
    for a in expected - set(counts.index):
        problems.append(f"arm {a}: 0 runs")
    md5s = df.groupby("arm").md5.agg(lambda s: sorted(set(s)))
    # verdicts vs the reference per arm, primaries with Holm within the arm's family
    ref_df = df[df.arm == ref]
    comp = []
    for label, desc, gains, glu, extra, cls in BATCH_ARMS:
        if label == ref:
            continue
        sub = df[df.arm == label]
        if sub.empty:
            continue
        pv = {}
        res_k = {}
        for k in PRIMARIES:
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
        for k in PRIMARIES:
            res_k[k]["p_holm"] = adj.get(k, float("nan"))
            comp.append(dict(arm=label, key=k, **res_k[k]))
        for k in SECONDARIES:
            a_ = [x for x in sub[k].tolist() if x is not None and np.isfinite(x)]
            b_ = [x for x in ref_df[k].tolist() if x is not None and np.isfinite(x)]
            if a_ and b_:
                c_ = common.compare(a_, b_)
                comp.append(dict(arm=label, key=k, verdict=c_["verdict"], diff=c_["diff"], z=c_["z"], p=c_["p"], p_floor=c_["p_floor"], null_sd_zero=c_["null_sd_zero"],
                                 n_stim=len(a_), n_null=len(b_), stim_values=[round(x, 4) for x in a_], null_values=[round(x, 4) for x in b_], p_holm=float("nan")))
    cdf = pd.DataFrame(comp)
    # the predeclared decision rule per arm
    rule = []
    for label, desc, gains, glu, extra, cls in BATCH_ARMS:
        sub = df[df.arm == label]
        ok = [(bool(s >= 5.0) and bool(np.isfinite(w) and 2.5 <= w <= 5.0) and bool(np.isfinite(h) and 5.0 <= h <= 60.0))
              for s, w, h in zip(sub.survival_s.fillna(0), sub.width_half_post.astype(float), sub.bump_hz_post.astype(float))]
        conf = [bool(s >= 5.0) for s in sub.survival_s.fillna(0)]
        rule.append(dict(arm=label, gains=gains, runs=int(len(sub)), working_compass_seeds=int(sum(ok)), bump_survives_seeds=int(sum(conf)),
                         working=bool(sum(ok) >= 3 and len(sub) >= 4), survival=sub.survival_s.tolist(), rate=sub.bump_hz_post.tolist(), width=sub.width_half_post.tolist(),
                         frac_confined_post=sub.frac_confined_post.tolist(), devices=sorted(set(map(str, sub.device_name))), md5=md5s.get(label, [])))
    rdf = pd.DataFrame(rule)
    # console-vs-json device check: every .txt should say the device
    txt_missing = [a[0] + f"_s{s}" for a in BATCH_ARMS for s in range(4) if not (out_dir / f"{a[0]}_s{s}.txt").exists()]
    # write
    df.drop(columns=["profile_post"]).to_csv(an / "runs.csv", index=False)
    cdf.to_csv(an / "compare.csv", index=False)
    rdf.to_csv(an / "decision.csv", index=False)
    json.dump(dict(runs=df.drop(columns=["profile_post"]).to_dict("records"), compare=comp, decision=rule, problems=problems, txt_missing=txt_missing,
                   n_runs=int(len(df)), n_expected=len(BATCH_ARMS) * 4, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   generator="python " + " ".join(sys.argv), analysis_sha256=sha256_file(Path(__file__)),
                   cx_wedge_sha256=sha256_file(ROOT / "scripts" / "cx_wedge.py"), common_sha256=sha256_file(ROOT / "flyverse" / "interp" / "common.py")),
              open(an / "analysis.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    # markdown
    L = [f"# cx5 analysis -- n_runs {len(df)} of {len(BATCH_ARMS) * 4} expected ({out_dir.as_posix()}/<arm>_s<seed>.json); generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"]
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
        order = [a[0] for a in BATCH_ARMS]
        for ax, k in zip(axes, PRIMARIES):
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
    print(f"-> {an / 'analysis.md'}, runs.csv, compare.csv, decision.csv, analysis.json, scatter.png")
    return df, cdf, rdf


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/cx5/structure")
    ap.add_argument("--no-banc", action="store_true", help="skip the BANC transmitter read")
    ap.add_argument("--quick", action="store_true", help="fewer rate-model iterations (smoke)")
    ap.add_argument("--plan-batch", default=None, metavar="DIR", help="write DIR/batch.sh, DIR/tree_state.json for the wedge-compass batch (no structure pass)")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--analyse", default=None, metavar="DIR", help="analyse the fetched batch in DIR (no structure pass)")
    a = ap.parse_args()
    if a.plan_batch:
        plan_batch(Path(a.plan_batch), minutes=a.minutes); return
    if a.analyse:
        analyse_batch(Path(a.analyse)); return
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
    rows, mats = [], {}
    for label, ev, cc, p, cl in configs:
        cir = Circuit(cc, p, label, ev, cells=dict(cl))
        r, m = analyse(cir)
        rows.append(r)
        key = label.split(":")[0].replace(" ", "_").replace("+", "")
        for k, v in m["M16"].items():
            mats[f"M16_{key}_{k}"] = v
        for k, v in m["K"].items():
            mats[f"K_{key}_{k}"] = v
    res["configs"] = rows
    # ranking of the single and combined changes
    rk = []
    for r in rows:
        a_ = r["rate_model"]["after"]
        rk.append(dict(label=r["label"], bump=r["rate_model"]["bump_after"], in_minus_out=a_["epg_in"] - a_["epg_out"],
                       gamma_crit_k1=r["two_step_gain"]["gamma_crit_k1"], u_pen_pulse=r["rate_model"]["pulse"]["u_PEN_in"],
                       lambda_1=r["fourier"]["net"][1], lambda_0=r["fourier"]["net"][0]))
    rk.sort(key=lambda d: (-int(d["bump"]), -d["in_minus_out"], d["gamma_crit_k1"], -d["u_pen_pulse"]))
    res["ranking"] = rk
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
