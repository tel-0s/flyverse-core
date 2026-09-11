"""Does the connectome's central-complex wiring support a ring attractor under this model's synaptic rules?

Structural audit (no simulation needed):
  * wedge identity of every EPG / PEN / PEG / Delta7 cell from the instance names (PB glomerulus L1-L9 / R1-R9;
    the Delta7 instance carries its OUTPUT glomeruli, e.g. Delta7(PB15)_L1L9R8_R);
  * the 16-wedge ring order of the glomeruli, read off the direct EPG -> EPG matrix (L_i neighbours R_(9-i) and
    R_(8-i)): L1 R8 L2 R7 L3 R6 L4 R5 L5 R4 L6 R3 L7 R2 L8 R1;
  * effective weights A[post, pre] in mV per presynaptic spike = w_syn * fan-in scale[post] * _shaped_weights
    (connection cap 60, path gains, same-type damping 0.1) -- exactly what Brain installs;
  * EPG x EPG two-step matrices through PEN, PEG and Delta7 (A[EPG, X] @ A[X, EPG], mV^2 per spike, one step each
    way), cell-level ordered by wedge and aggregated to 16 wedges / 8 tiles ("input to a typical cell of wedge i
    if every cell of wedge k fires once, relayed through X");
  * the ring-attractor window: for a bump of k wedges at uniform rate, the net two-step input to wedges inside
    versus outside the bump as a function of rho = gD / gE^2 (gD = Delta7 -> EPG gain, gE = EPG <-> PEN gain, both
    links), in the linear-rate estimate; the window is (max_out E/|I|, min_in E/|I|).

Simulation cross-check (--sim gE:gD [gE:gD ...]): FlyBrain on the full connectome, no world, compass adaptation
off, 10 Hz Poisson background on every EPG, one tile (two wedges) driven at +40 Hz for 2 s, then 5 s free;
reports cells above threshold inside / outside the driven wedge at 0.5 / 1 / 2 / 3 / 5 s after the pulse.

Usage:
    python scripts/cx_wedge.py                      # structure only; PNGs + JSON under docs/audits/
    python scripts/cx_wedge.py --sim 1:1 1.5:2 2:8  # + simulation at those (EPG<->PEN gain : Delta7 gain)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyverse import brain, connectome  # noqa: E402

AUDIT_DIR = Path(__file__).resolve().parent.parent / "docs" / "audits"
RING16 = ["L1", "R8", "L2", "R7", "L3", "R6", "L4", "R5", "L5", "R4", "L6", "R3", "L7", "R2", "L8", "R1"]
POS16 = {g: i for i, g in enumerate(RING16)}
POS16["L9"] = 0     # L9 wraps onto L1 (Delta7_L1L9R8), R9 onto R1 (Delta7_L8R1R9)
POS16["R9"] = 15
COMPASS_RE = r"^(EPG|PEN|PEG|Delta7)"


def effective_weights(c, p: brain.LIFParams):
    """A[post, pre] in mV per presynaptic spike, as Brain installs it (before the optic prune, which does not
    touch these cells)."""
    W = brain._shaped_weights(c, p)
    tot = np.asarray(abs(W).sum(axis=1)).ravel()
    scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
    A = (sp.diags(scale) @ W).tocsr()
    A.data *= np.float32(p.w_syn)
    return A, scale, tot


def glomerulus(instance: str) -> str | None:
    m = re.search(r"_([LR]\d)(?:$|_)", instance)
    return m.group(1) if m else None


def delta7_outputs(instance: str) -> list[str]:
    return re.findall(r"[LR]\d", instance.split("_")[1])


def compass_cells(c):
    n = c.neurons
    ty = n.type.fillna("").to_numpy()
    inst = n.instance.fillna("").to_numpy()
    cells = {}
    for name, pat in [("EPG", r"^EPG$"), ("EPGt", r"^EPGt$"), ("PEN", r"^PEN_"), ("PEN_a", r"^PEN_a"), ("PEN_b", r"^PEN_b"),
                      ("PEG", r"^PEG$"), ("Delta7", r"^Delta7$")]:
        idx = np.flatnonzero([bool(re.match(pat, t)) for t in ty])
        if name == "Delta7":
            outs = [delta7_outputs(inst[i]) for i in idx]
            pos = np.array([np.mean([POS16[g] for g in o if g in POS16]) for o in outs])   # mean output position
            # adjacent output wedges (e.g. L1=0, R8=1); L8R1R9 -> (14, 15); mean is the tile centre
            lab = ["".join(o) for o in outs]
        else:
            gl = [glomerulus(inst[i]) for i in idx]
            pos = np.array([POS16[g] for g in gl], dtype=float)
            lab = gl
        order = np.lexsort((idx, pos))
        cells[name] = dict(idx=idx[order], pos=pos[order], label=[lab[i] for i in order],
                           body=n.bodyId.to_numpy()[idx[order]])
    return cells


def ring_dist(a, b, n=16):
    d = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :]) % n
    return np.minimum(d, n - d)


def aggregate(K, pos_post, pos_pre, n_bins, bin_of):
    """M[i, k] = mean over post cells in bin i of the sum over pre cells in bin k of K[post, pre]."""
    bp, bq = bin_of(pos_post), bin_of(pos_pre)
    M = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        rows = bp == i
        if not rows.any():
            continue
        for k in range(n_bins):
            cols = bq == k
            M[i, k] = K[rows][:, cols].sum(axis=1).mean() if cols.any() else 0.0
    return M


def bump_window(E, I, k):
    """Bump = k contiguous bins starting at every offset (the ring is not perfectly uniform); per bin the two-step
    excitation E and inhibition I (negative) it receives from the bump; window of rho = gD / gE^2 such that every
    inside bin is net-excited and every outside bin net-inhibited. Returns the offset-averaged profile and the
    window (rho_low, rho_high) worst-case over offsets, plus the per-offset windows."""
    n = E.shape[0]
    rows = []
    for s in range(n):
        inb = np.zeros(n, bool)
        inb[[(s + j) % n for j in range(k)]] = True
        e = E[:, inb].sum(axis=1)
        i = I[:, inb].sum(axis=1)
        ratio = np.where(np.abs(i) > 1e-9, e / np.maximum(np.abs(i), 1e-9), np.inf)
        rho_low = ratio[~inb].max() if (~inb).any() else 0.0   # outside must be net inhibited: rho > E/|I|
        rho_high = ratio[inb].min()                            # inside must stay net excited: rho < E/|I|
        rows.append(dict(offset=s, e_in=e[inb].mean(), e_out=e[~inb].mean() if (~inb).any() else 0.0,
                         i_in=i[inb].mean(), i_out=i[~inb].mean() if (~inb).any() else 0.0,
                         e_in_min=e[inb].min(), e_out_max=e[~inb].max() if (~inb).any() else 0.0,
                         rho_low=float(rho_low), rho_high=float(rho_high)))
    df = pd.DataFrame(rows)
    return df


def heatmap(M, labels, title, path, cmap="RdBu_r", symmetric=True, fmt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = np.abs(M).max() if M.size else 1.0
    fig, ax = plt.subplots(figsize=(max(4, 0.42 * len(labels) + 1.5), max(3.5, 0.42 * len(labels) + 1.2)))
    im = ax.imshow(M, cmap=cmap, vmin=-v if symmetric else 0, vmax=v, aspect="equal")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("presynaptic EPG wedge (ring order)"); ax.set_ylabel("postsynaptic EPG wedge")
    ax.set_title(title, fontsize=9)
    if fmt and len(labels) <= 16:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt % M[i, j], ha="center", va="center", fontsize=5.5,
                        color="white" if abs(M[i, j]) > 0.6 * v else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def structure(out_dir: Path, gamma_nominal=6.0, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    c = connectome.load(verbose=False)
    p = brain.LIFParams()
    A, scale, tot = effective_weights(c, p)
    cells = compass_cells(c)
    epg, pen, peg, d7, epgt = (cells[k] for k in ("EPG", "PEN", "PEG", "Delta7", "EPGt"))
    res = {"n_cells": {k: int(len(v["idx"])) for k, v in cells.items()},
           "ring16": RING16, "cells_per_wedge": {g: int((np.array(epg["label"]) == g).sum()) for g in RING16},
           "fan_in_scale": {k: [float(scale[v["idx"]].min()), float(scale[v["idx"]].max())] for k, v in cells.items()},
           "fan_in_total": {k: [float(tot[v["idx"]].min()), float(tot[v["idx"]].max())] for k, v in cells.items()}}
    log("cells:", res["n_cells"], "\nEPG per wedge:", res["cells_per_wedge"])
    log("Delta7 output labels:", sorted(set(d7["label"])))

    def block(post, pre):
        return A[post["idx"]][:, pre["idx"]].toarray().astype(np.float64)

    # ---- one-step pair statistics (mV per presynaptic spike per pair)
    one = {}
    for name, (pre, post) in {"EPG->PEN": (epg, pen), "PEN->EPG": (pen, epg), "EPG->PEG": (epg, peg), "PEG->EPG": (peg, epg),
                              "EPG->Delta7": (epg, d7), "Delta7->EPG": (d7, epg), "Delta7->PEN": (d7, pen), "Delta7->PEG": (d7, peg),
                              "Delta7->Delta7": (d7, d7), "EPG->EPG": (epg, epg), "PEN->PEN": (pen, pen), "PEG->PEN": (peg, pen),
                              "EPGt->Delta7": (epgt, d7), "Delta7->EPGt": (d7, epgt)}.items():
        B = block(post, pre); nz = B != 0
        one[name] = dict(pairs=int(nz.sum()), mean_per_pair=float(B[nz].mean()) if nz.any() else 0.0,
                         total_per_post=float(B.sum(axis=1).mean()))
    res["one_step"] = one
    for k, v in one.items():
        log(f"  {k:>15}: {v['pairs']:5d} pairs, {v['mean_per_pair']:+6.2f} mV/pair, {v['total_per_post']:+7.1f} mV per post if all pre fire once")

    # ---- Delta7 own-wedge structure at the one-step level
    dpos = d7["pos"]
    d_in = block(d7, epg)     # [d7, epg]
    d_out = block(epg, d7)    # [epg, d7]
    dist_in = ring_dist(dpos, epg["pos"])      # distance from the Delta7's output tile to the EPG wedge
    dist_out = ring_dist(epg["pos"], dpos).T   # same, for the output block (transposed to [d7, epg])
    prof_in = np.array([d_in[dist_in <= 0.5].sum() / max((dist_in <= 0.5).sum(), 1)] +
                       [d_in[(dist_in > k - 1) & (dist_in <= k)].sum() / max(((dist_in > k - 1) & (dist_in <= k)).sum(), 1) for k in range(1, 9)])
    prof_out = np.array([d_out.T[dist_out <= 0.5].sum() / max((dist_out <= 0.5).sum(), 1)] +
                        [d_out.T[(dist_out > k - 1) & (dist_out <= k)].sum() / max(((dist_out > k - 1) & (dist_out <= k)).sum(), 1) for k in range(1, 9)])
    res["delta7_profile_by_distance"] = dict(distance_wedges=list(range(9)), epg_to_delta7_mean_pair_mV=prof_in.tolist(),
                                             delta7_to_epg_mean_pair_mV=prof_out.tolist())
    log("Delta7 one-step profile vs ring distance (wedges, 0 = the Delta7's own output tile), mean mV per cell pair:")
    log("   EPG -> Delta7:", np.round(prof_in, 2).tolist())
    log("   Delta7 -> EPG:", np.round(prof_out, 2).tolist())

    # ---- two-step EPG x EPG matrices (mV^2 per spike; one step each way through X)
    K = {"PEN": block(epg, pen) @ block(pen, epg), "PEG": block(epg, peg) @ block(peg, epg),
         "Delta7": block(epg, d7) @ block(d7, epg), "PEN_a": block(epg, cells["PEN_a"]) @ block(cells["PEN_a"], epg),
         "PEN_b": block(epg, cells["PEN_b"]) @ block(cells["PEN_b"], epg)}
    K["direct"] = block(epg, epg)   # mV per spike, no intermediate
    # three-step: EPG -> Delta7 -> PEN -> EPG (the Delta7 clamp on PEN), mV^3
    K["Delta7_PEN"] = block(epg, pen) @ block(pen, d7) @ block(d7, epg)
    bin16 = lambda x: np.asarray(np.round(x), int) % 16
    bin8 = lambda x: (np.asarray(np.round(x), int) % 16) // 2
    tiles = [f"{RING16[2 * i]}/{RING16[2 * i + 1]}" for i in range(8)]
    M16 = {k: aggregate(v, epg["pos"], epg["pos"], 16, bin16) for k, v in K.items()}
    M8 = {k: aggregate(v, epg["pos"], epg["pos"], 8, bin8) for k, v in K.items()}
    res["M16"] = {k: np.round(v, 2).tolist() for k, v in M16.items()}
    res["M8"] = {k: np.round(v, 2).tolist() for k, v in M8.items()}
    res["tiles"] = tiles
    labels46 = [f"{g}" for g in epg["label"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ("PEN", "PEG", "Delta7", "direct", "Delta7_PEN"):
        unit = {"direct": "mV/spike", "Delta7_PEN": "mV^3 (3 steps)"}.get(k, "mV^2 (2 steps)")
        heatmap(K[k], labels46, f"EPG x EPG through {k}, per cell pair [{unit}]", out_dir / f"cx_wedge_{k}_cells.png")
        heatmap(M16[k], RING16, f"EPG x EPG through {k}, 16 wedges (sum over pre wedge, mean over post) [{unit}]",
                out_dir / f"cx_wedge_{k}_16.png", fmt="%.0f")
        heatmap(M8[k], tiles, f"EPG x EPG through {k}, 8 tiles [{unit}]", out_dir / f"cx_wedge_{k}_8.png", fmt="%.0f")
    pd.set_option("display.width", 250)
    for k in ("PEN", "PEG", "Delta7", "direct"):
        log(f"\n{k}: 16-wedge matrix (rows post, cols pre; ring order)")
        log(pd.DataFrame(np.round(M16[k], 1), index=RING16, columns=RING16).to_string())
        log(f"{k}: 8-tile matrix")
        log(pd.DataFrame(np.round(M8[k], 1), index=tiles, columns=tiles).to_string())

    # ---- profile of each path against ring distance (16-wedge level, averaged over the diagonal bands)
    prof = {}
    for k in ("PEN", "PEG", "Delta7", "direct", "Delta7_PEN"):
        M = M16[k]
        d = ring_dist(np.arange(16), np.arange(16))
        prof[k] = [float(M[d == j].mean()) for j in range(9)]
    res["profile16_by_distance"] = dict(distance_wedges=list(range(9)), **prof)
    log("\nring-distance profiles (16-wedge level, mean over bands; distance in wedges of 22.5 deg):")
    for k, v in prof.items():
        log(f"   {k:>10}: " + " ".join(f"{x:8.1f}" for x in v))
    # locality of PEN excitation: share of the total row mass within +-1 tile (+-2 wedges) and the +-1 wedge
    d = ring_dist(np.arange(16), np.arange(16))
    for k in ("PEN", "PEG"):
        M = M16[k]; total = M.sum()
        res[f"{k}_locality"] = dict(within_1_wedge=float(M[d <= 1].sum() / total), within_2_wedges=float(M[d <= 2].sum() / total),
                                    within_4_wedges=float(M[d <= 4].sum() / total), opposite_half=float(M[d >= 5].sum() / total),
                                    diag_over_mean=float(np.mean(np.diag(M)) / (total / 256)))
        log(f"{k} locality: {res[f'{k}_locality']}")
    Md = M16["Delta7"]
    own = np.mean(np.diag(Md)); nb = Md[d == 1].mean(); far = Md[d >= 5].mean(); other = Md[d >= 1].mean(); opp = Md[d == 8].mean()
    res["Delta7_inhibition"] = dict(own_wedge=float(own), neighbour_wedge=float(nb), other_wedges_mean=float(other),
                                    far_half_mean=float(far), opposite_wedge=float(opp),
                                    ratio_own_over_other=float(own / other), ratio_own_over_opposite=float(own / opp),
                                    ratio_own_over_far_half=float(own / far))
    log("Delta7 two-step inhibition (16-wedge):", {k: round(v, 3) for k, v in res["Delta7_inhibition"].items()})

    # ---- ring-attractor window as a function of rho = gD / gE^2, bump width k, at 16 and 8 resolution
    windows = {}
    for level, M, n in (("16", M16, 16), ("8", M8, 8)):
        E = M["PEN"] + M["PEG"]; I = M["Delta7"]
        for k in range(1, (n // 2) + 1):
            df = bump_window(E, I, k)
            w = dict(k=k, e_in=float(df.e_in.mean()), e_out=float(df.e_out.mean()), i_in=float(df.i_in.mean()), i_out=float(df.i_out.mean()),
                     locality_ratio=float((df.e_in.mean() / df.e_out.mean()) / (df.i_in.mean() / df.i_out.mean())),
                     rho_low_worst=float(df.rho_low.max()), rho_high_worst=float(df.rho_high.min()),
                     rho_low_mean=float(df.rho_low.mean()), rho_high_mean=float(df.rho_high.mean()),
                     offsets_with_window=int((df.rho_high > df.rho_low).sum()), n_offsets=int(len(df)))
            windows[f"{level}:{k}"] = w
    res["windows"] = windows
    log("\nring-attractor window (two-step, linear): rho = gD / gE^2; bump of k bins; E = PEN+PEG, I = Delta7 (mean per post cell, sum over bump)")
    log("  level k   E_in    E_out    I_in    I_out  E_in/E_out  |I_in|/|I_out|  rho_low(max out E/|I|)  rho_high(min in E/|I|)  offsets with window")
    for key, w in windows.items():
        lvl, k = key.split(":")
        log(f"  {lvl:>4} {k:>2} {w['e_in']:7.0f} {w['e_out']:7.0f} {w['i_in']:8.0f} {w['i_out']:8.0f}   {w['e_in']/w['e_out']:6.2f}      "
            f"{w['i_in']/w['i_out']:6.2f}        {w['rho_low_worst']:7.3f} (mean {w['rho_low_mean']:.3f})       {w['rho_high_worst']:7.3f} (mean {w['rho_high_mean']:.3f})     {w['offsets_with_window']}/{w['n_offsets']}")

    # ---- absolute scale: what the bump wedge receives at nominal intermediate gain gamma (Hz per mV), tau_syn
    tau = p.tau_syn / 1000.0
    res["gamma_nominal_hz_per_mV"] = gamma_nominal
    scale_2 = gamma_nominal * tau * tau   # mV of mean depolarisation per Hz of presynaptic rate per mV^2 of two-step weight
    res["mV_per_Hz_per_mV2"] = scale_2
    abs_rows = {}
    for key in ("16:2", "16:3", "16:4", "8:1", "8:2"):
        w = windows[key]
        abs_rows[key] = dict(exc_mV_per_Hz_in=w["e_in"] * scale_2, inh_mV_per_Hz_in=w["i_in"] * scale_2,
                             exc_mV_per_Hz_out=w["e_out"] * scale_2, inh_mV_per_Hz_out=w["i_out"] * scale_2,
                             direct_mV_per_Hz_in=float(np.mean([M16["direct"][:, :].sum(axis=1).mean()])) * tau)
    res["absolute_at_gamma"] = abs_rows
    log(f"\nabsolute (gamma = {gamma_nominal} Hz/mV, tau_syn {p.tau_syn} ms): mean depolarisation of an EPG per Hz of bump rate (two-step, gains x1)")
    for key, r in abs_rows.items():
        log(f"  {key}: inside exc {r['exc_mV_per_Hz_in']:+.3f} inh {r['inh_mV_per_Hz_in']:+.3f} mV/Hz; outside exc {r['exc_mV_per_Hz_out']:+.3f} inh {r['inh_mV_per_Hz_out']:+.3f} mV/Hz")

    # Delta7 -> PEN clamp: input to a PEN from the bump directly vs through Delta7, per Hz of bump rate at gamma
    Apen_e = block(pen, epg); Apen_d = block(pen, d7); Ad_e = block(d7, epg)
    direct_pen = aggregate(Apen_e, pen["pos"], epg["pos"], 16, bin16)          # mV per spike, [pen wedge, epg wedge]
    via_d7 = aggregate(Apen_d @ Ad_e, pen["pos"], epg["pos"], 16, bin16)        # mV^2
    dd = ring_dist(np.arange(16), np.arange(16))
    res["PEN_drive_profile"] = dict(distance_wedges=list(range(9)),
                                    direct_EPG_to_PEN_mV=[float(direct_pen[dd == j].mean()) for j in range(9)],
                                    via_Delta7_mV2=[float(via_d7[dd == j].mean()) for j in range(9)],
                                    via_Delta7_at_gamma_mV=[float(via_d7[dd == j].mean() * gamma_nominal * tau) for j in range(9)])
    log("PEN drive from an EPG wedge vs distance (PEN wedge = its glomerulus): direct (mV/spike) and via Delta7 (mV, at gamma):")
    log("   direct     :", np.round(res["PEN_drive_profile"]["direct_EPG_to_PEN_mV"], 1).tolist())
    log("   via Delta7 :", np.round(res["PEN_drive_profile"]["via_Delta7_at_gamma_mV"], 1).tolist())
    with open(out_dir / "cx_wedge.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez(out_dir / "cx_wedge_matrices.npz", **{f"K_{k}": v for k, v in K.items()}, **{f"M16_{k}": v for k, v in M16.items()},
             **{f"M8_{k}": v for k, v in M8.items()}, epg_pos=epg["pos"], epg_body=epg["body"])
    return res, cells, c


def gained_blocks(c, cells, gE, gD, delta7_pen=True):
    """Effective matrix among the compass cells (mV per spike) with the gains applied as Brain would apply
    type_path_gain (before the fan-in scale, which stays 1.00 for these cells up to gains of ~x1.7 on EPG)."""
    tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", gE), (r"^PEN_", r"^EPG$", gE),
                                                (r"^EPG$", r"^PEG$", gE), (r"^PEG$", r"^EPG$", gE),
                                                (r"^Delta7$", r"^(EPG$|PEN_)" if delta7_pen else r"^EPG$", gD)]
    p = brain.LIFParams(type_path_gain=tpg)
    A, scale, tot = effective_weights(c, p)
    idx = np.concatenate([cells[k]["idx"] for k in ("EPG", "PEN", "PEG", "Delta7")])
    return A[idx][:, idx].toarray().astype(np.float64), idx, p, scale[idx]


def lif_fi(u, p=None, sigma=2.0):
    """Mean firing rate (Hz) of the LIF at mean input u (mV above rest), smoothed over Gaussian input fluctuations
    of sigma mV (Poisson input); deterministic f-I: 1 / (t_ref + tau_m ln(u / (u - theta)))."""
    p = p or brain.LIFParams()
    theta = p.v_th - p.v_rest
    xs, ws = np.polynomial.hermite_e.hermegauss(15)
    ws = ws / ws.sum()
    out = np.zeros_like(u, dtype=float)
    for x, w in zip(xs, ws):
        uu = u + sigma * x
        m = uu > theta + 1e-6
        f = np.zeros_like(u, dtype=float)
        f[m] = 1000.0 / (p.t_ref + p.tau_m * np.log(uu[m] / (uu[m] - theta)))
        out += w * f
    return out


def rate_model(c, cells, gE, gD, delta7_pen=True, background_hz=10.0, pulse_hz=40.0, start_wedge=0, width=4,
               sigma=2.0, iters=4000, alpha=0.05):
    """Threshold-linear mean-field fixed point of the compass circuit: r = f(tau_syn * A r) with the EPG's forced
    Poisson background / pulse added to the intrinsic rate. Returns the state after the pulse and after release."""
    A, idx, p, scale = gained_blocks(c, cells, gE, gD, delta7_pen)
    nE = len(cells["EPG"]["idx"]); n = len(idx)
    nP, nG = len(cells["PEN"]["idx"]), len(cells["PEG"]["idx"])
    tau = p.tau_syn / 1000.0
    wedge_of = np.asarray(np.round(cells["EPG"]["pos"]), int) % 16
    inside = np.isin(wedge_of, [(start_wedge + j) % 16 for j in range(width)])
    forced = np.zeros(n); forced[:nE] = background_hz
    r = forced.copy()

    def relax(forced, r):
        for _ in range(iters):
            u = tau * (A @ r)
            target = lif_fi(u, p, sigma) + forced
            r = (1 - alpha) * r + alpha * target
        return r

    r_bg = relax(forced, r)
    f_pulse = forced.copy(); f_pulse[:nE][inside] += pulse_hz
    r_pulse = relax(f_pulse, r_bg)
    r_after = relax(forced, r_pulse)

    def summ(r):
        e = r[:nE]
        return dict(epg_in=float(e[inside].mean()), epg_out=float(e[~inside].mean()),
                    epg_in_min=float(e[inside].min()), epg_out_max=float(e[~inside].max()),
                    pen=float(r[nE:nE + nP].mean()), peg=float(r[nE + nP:nE + nP + nG].mean()),
                    delta7=float(r[nE + nP + nG:].mean()),
                    profile=[float(e[wedge_of == w].mean()) for w in range(16)])
    return dict(gE=gE, gD=gD, delta7_pen=delta7_pen, background=summ(r_bg), pulse=summ(r_pulse), after=summ(r_after))


def rate_grid(c, cells, gEs, gDs, delta7_pen, log=print, **kw):
    rows = []
    bg = kw.get("background_hz", 10.0)
    log(f"threshold-linear rate model (Delta7 -> PEN {'x gD' if delta7_pen else 'x1'}): EPG in / out after release "
        f"(background-state ring / PEN / Delta7 in brackets; BUMP = in > 2 x out and in > bg + 5 Hz)")
    for gE in gEs:
        for gD in gDs:
            rm = rate_model(c, cells, gE, gD, delta7_pen, **kw)
            a, b = rm["after"], rm["background"]
            bump = a["epg_in"] > 2 * a["epg_out"] and a["epg_in"] > bg + 5
            rm["bump"] = bool(bump)
            rows.append(rm)
            log(f"  gE {gE:<4} gD {gD:<4}: after in {a['epg_in']:6.1f} out {a['epg_out']:6.1f} (max {a['epg_out_max']:6.1f}) "
                f"PEN {a['pen']:5.1f} D7 {a['delta7']:5.1f}  [bg ring {b['epg_in']:5.1f} PEN {b['pen']:5.1f} D7 {b['delta7']:5.1f}]  {'BUMP' if bump else ''}")
    return rows


def simulate(c, cells, gains, seconds=5.0, pulse_s=2.0, background_hz=10.0, pulse_hz=40.0, start_wedge=0, width=4, seed=0,
             thresh_hz=22.0, cuda_graphs=True, delta7_pen=True, verbose=True):
    """FlyBrain on the full connectome; drive `width` contiguous wedges (of 16) of the EPG ring from `start_wedge`;
    report persistence and confinement after the pulse."""
    from flyverse.fly import FlyBrain
    log = print if verbose else (lambda *a, **k: None)
    epg = cells["EPG"]
    wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    inside = np.isin(wedge_of, [(start_wedge + j) % 16 for j in range(width)])
    idx_epg = epg["idx"]
    pen_idx, d7_idx, peg_idx = cells["PEN"]["idx"], cells["Delta7"]["idx"], cells["PEG"]["idx"]
    others = np.setdiff1d(np.arange(c.n), np.concatenate([idx_epg, pen_idx, d7_idx, peg_idx, cells["EPGt"]["idx"]]))
    out = []
    for gE, gD in gains:
        tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", gE), (r"^PEN_", r"^EPG$", gE),
                                                    (r"^EPG$", r"^PEG$", gE), (r"^PEG$", r"^EPG$", gE),
                                                    (r"^Delta7$", r"^(EPG$|PEN_)" if delta7_pen else r"^EPG$", gD)]
        params = brain.LIFParams(adapt_by_type={COMPASS_RE: 0.0}, type_path_gain=tpg)
        t0 = time.time()
        fb = FlyBrain(c, lif_params=params, seed=seed, cuda_graphs=cuda_graphs)
        # background: FlyBrain.stimulate pulses expire, so hold the background as a long pulse (as the grid did)
        total_ms = (1.0 + pulse_s + seconds) * 1000
        fb.stimulate(idx_epg, background_hz, total_ms + 100)

        def sample(tag):
            r = fb.brain.rates(idx_epg).copy()
            d = {f"{tag}_in_mean": float(r[inside].mean()), f"{tag}_out_mean": float(r[~inside].mean()),
                 f"{tag}_in_above": int((r[inside] > thresh_hz).sum()), f"{tag}_out_above": int((r[~inside] > thresh_hz).sum()),
                 f"{tag}_pen": float(fb.brain.mean_rate(pen_idx)), f"{tag}_delta7": float(fb.brain.mean_rate(d7_idx)),
                 f"{tag}_peg": float(fb.brain.mean_rate(peg_idx)), f"{tag}_rest": float(fb.brain.mean_rate(others)),
                 f"{tag}_wedge_profile": [float(r[wedge_of == w].mean()) for w in range(16)]}
            ang = 2 * np.pi * wedge_of / 16
            z = np.sum(r * np.exp(1j * ang)) / max(r.sum(), 1e-9)
            d[f"{tag}_vector_strength"] = float(np.abs(z)); d[f"{tag}_centre_wedge"] = float((np.angle(z) % (2 * np.pi)) / (2 * np.pi) * 16)
            return d

        row = dict(gE=gE, gD=gD, delta7_pen=delta7_pen, background_hz=background_hz, pulse_hz=pulse_hz, start_wedge=start_wedge,
                   width=width, seed=seed, n_in=int(inside.sum()), n_out=int((~inside).sum()))
        fb.step(1000.0)                                             # 1 s settle on background
        row.update(sample("pre"))
        fb.stimulate(idx_epg[inside], background_hz + pulse_hz, pulse_s * 1000)
        fb.step(pulse_s * 1000)
        row.update(sample("during"))
        t = 0.0
        marks = (0.5, 1.0, 2.0, 3.0, 5.0)
        for mark in marks:
            fb.step((mark - t) * 1000); t = mark
            row.update(sample(f"t{mark}"))
        row["wall_s"] = round(time.time() - t0, 1)

        def fmt(tag):
            return (f"in {row[f'{tag}_in_mean']:.1f} ({row[f'{tag}_in_above']}/{row['n_in']}) out {row[f'{tag}_out_mean']:.1f} "
                    f"({row[f'{tag}_out_above']}/{row['n_out']}) PEN {row[f'{tag}_pen']:.1f} D7 {row[f'{tag}_delta7']:.1f} vs {row[f'{tag}_vector_strength']:.2f}")
        log(f"gE {gE} gD {gD} (D7->PEN {'x gD' if delta7_pen else 'x1'}, bg {background_hz} Hz, width {width}): pre {fmt('pre')}; "
            f"during {fmt('during')}; " + "; ".join(f"{m}s {fmt(f't{m}')}" for m in marks)
            + f"; PEG {row['t5.0_peg']:.1f} rest {row['t5.0_rest']:.2f} Hz; {row['wall_s']} s")
        out.append(row)
        del fb
        import torch; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(AUDIT_DIR))
    ap.add_argument("--sim", nargs="*", default=None, help="gE:gD pairs, e.g. 1:1 1.5:2")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--background", type=float, default=10.0)
    ap.add_argument("--pulse-hz", type=float, default=40.0)
    ap.add_argument("--start-wedge", type=int, default=0)
    ap.add_argument("--width", type=int, default=4, help="driven wedges (of 16); 4 = two tiles = 90 deg")
    ap.add_argument("--no-structure", action="store_true", help="skip the PNG / JSON structural outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-delta7-pen", action="store_true", help="apply gD to Delta7 -> EPG only (not Delta7 -> PEN)")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--sim-out", default=None, help="JSON file for the simulation rows")
    ap.add_argument("--rate-grid", nargs=2, default=None, metavar=("GE", "GD"),
                    help="threshold-linear rate-model grid, comma-separated gains, e.g. --rate-grid 0.8,1,1.5 1,4,15")
    ap.add_argument("--rate-out", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out)
    if a.no_structure:
        c = connectome.load(verbose=False); cells = compass_cells(c)
    else:
        res, cells, c = structure(out_dir)
    if a.rate_grid is not None:
        gEs = [float(x) for x in a.rate_grid[0].split(",")]; gDs = [float(x) for x in a.rate_grid[1].split(",")]
        rows = rate_grid(c, cells, gEs, gDs, not a.no_delta7_pen, background_hz=a.background, pulse_hz=a.pulse_hz,
                         start_wedge=a.start_wedge, width=a.width)
        if a.rate_out:
            with open(a.rate_out, "w") as f:
                json.dump(rows, f, indent=1)
    if a.sim is not None:
        gains = [tuple(float(x) for x in g.split(":")) for g in a.sim] or [(1.0, 1.0)]
        rows = simulate(c, cells, gains, seconds=a.seconds, background_hz=a.background, pulse_hz=a.pulse_hz, start_wedge=a.start_wedge, width=a.width,
                        seed=a.seed, cuda_graphs=not a.no_graphs, delta7_pen=not a.no_delta7_pen)
        if a.sim_out:
            path = Path(a.sim_out)
            old = json.load(open(path)) if path.exists() else []
            with open(path, "w") as f:
                json.dump(old + rows, f, indent=1)


if __name__ == "__main__":
    main()
