"""The compass under sensory input: the ring attractor's operating points in the walking fly (docs/audits/compass_room.md).

Every ring-attractor result so far (docs/audits/cx_wedge.md, cx_glno.md) was measured with the LIF alone: no senses,
a held 10 Hz Poisson background on the EPG, the rest of the brain at 0.03 Hz. This probe puts the same gains into
BatchSim (16 independent rooms through one FlyBrain: full senses, the fenced single-apple table, program 'none' or 'cx')
and asks, per fly and per frame,

  (a) whether a bump forms spontaneously (0 .. --pulse-at s, no EPG drive of any kind) and whether a bump written into
      the ring by a 2 s pulse (4 wedges at +--pulse-hz Hz, a different tile per fly: row i starts at wedge 2 i mod 16)
      persists for the rest of the rollout under the sensory-driven brain (ER ring neurons, ExR feedback) or is
      destroyed / pinned -- EPG vector strength, width, centre, rate, cx_glno's confinement rule;
  (b) whether the bump centre tracks the fly's heading (circular correlation of centre with heading; Pearson r of the
      bump's angular velocity with the yaw rate) -- the model has no verified rotation input to PEN, so the expected
      answer is 'no' and the point is to state it precisely;
  (c) whether the bump produces a steering asymmetry -- PFL3 L - R and DNa02 L - R regressed on the bump's position
      (first harmonic in the ring coordinate, which is body-fixed anatomy up to a per-fly offset; and on the bump
      position relative to the heading) -- with the no-bump control as the null.

Compass gains are LIFParams overrides, exactly cx_glno's: adapt_by_type {'^(EPG|PEN|PEG|Delta7)': 0}, type_path_gain =
DEFAULT_TYPE_PATH_GAIN + EPG <-> PEN x gE, EPG <-> PEG x gE, Delta7 -> EPG x gD (Delta7 -> PEN x1, ER/ExR x1). They are
an EXPERIMENT's gains (the project rule: the plain model is not hand-tuned; --gE omitted = the shipped default, the
control). The receptor default is irrelevant on the ring (0 of 27,553 ring-core entries change) but is threaded through
(--receptor-model default | off) so the control is the shipped model.

Backend: one FlyBrain(batch=B) on the Torch path (adapt_by_type disables the native LIF kernel, so the control runs the
same path: cuda_kernels False, sparse matmul, CUDA graphs), 10 ms frames, LIF dt 0.5 ms, optic dt 1 ms.

GPU (cluster):
  python scripts/probe_compass_room.py --gE 2 --gD 15 --program none --seed 0 --seconds 60 --out out/cxroom/cxroom_g2-15_none_s0.json
    -> the JSON (config, per-fly metrics, per-condition summary) and cxroom_g2-15_none_s0.npz (per-frame per-fly records:
       EPG rates per cell in cx_wedge's wedge order, PEN / Delta7 / PFL3 / DNa02 L and R, PEG, ER/ExR, GLNO, PFN, hDelta,
       rest-of-brain rates, heading, yaw rate, speed, airborne, program mode / steering error, position).
CPU (local): python scripts/probe_compass_room.py --report --files "out/cxroom/cxroom_*.json" --table out/cxroom/compass_room
    -> per-condition tables (markdown + csv), recomputed from the NPZs (the metrics in the JSON are the same code).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyverse import brain  # noqa: E402
import cx_wedge  # noqa: E402

COMPASS_RE = cx_wedge.COMPASS_RE
RING_RE = cx_wedge.RING_RE
THRESH_HZ = 22.0            # cx_glno's cell threshold
VS_MIN = 0.6                # confinement: vector strength above this
IN_FRAC = 8.0 / 11.0        # ... and at least this fraction of the 4-wedge block's cells above THRESH_HZ
OUT_MAX = 3                 # ... and at most this many cells outside the block above THRESH_HZ
BLOCK = 4                   # wedges in the bump block (cx_wedge's driven width: two tiles, 90 deg)
FRAME_MS = 10.0


# ------------------------------------------------------------------------------------------------ gains
def compass_type_path_gain(gE: float, gD: float) -> list:
    """cx_wedge.simulate's list with delta7_pen=False, gR=1 (Delta7 -> EPG only; ER/ExR x1)."""
    return list(brain.DEFAULT_TYPE_PATH_GAIN) + [(r"^EPG$", r"^PEN_", gE), (r"^PEN_", r"^EPG$", gE),
                                                 (r"^EPG$", r"^PEG$", gE), (r"^PEG$", r"^EPG$", gE),
                                                 (r"^Delta7$", r"^EPG$", gD),
                                                 (RING_RE, r"^(EPG$|PEN_|PEG$)", 1.0)]


def patch_lif_params(gE, gD, receptor_model: str):
    """Every brain.LIFParams built from here on (BatchSim's included) carries the compass overrides and the receptor
    choice ('default' = LIFParams' own sign/abs, 'off' = the presynaptic rule). Returns the type_path_gain in force."""
    L = brain.LIFParams
    tpg = None if gE is None else compass_type_path_gain(float(gE), float(gD))
    adapt = None if gE is None else {COMPASS_RE: 0.0}

    def make(**kw):
        p = L(**kw)
        if tpg is not None:
            p.type_path_gain = tpg
            p.adapt_by_type = adapt
        if receptor_model == "off":
            p.receptor_model = None
        return p
    brain.LIFParams = make
    return tpg, adapt


# ------------------------------------------------------------------------------------------------ analysis
def circ_mean(a):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(a))))) if len(a) else float("nan")


def circ_corr(a, b):
    """Jammalamadaka-Sarma circular correlation of two angle series."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    sa, sb = np.sin(a - circ_mean(a)), np.sin(b - circ_mean(b))
    d = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / d) if d > 0 else float("nan")


def circ_sd_wedges(theta):
    """Circular standard deviation in wedge units (16 wedges = 2 pi)."""
    if len(theta) < 2:
        return float("nan")
    R = min(np.abs(np.mean(np.exp(1j * np.asarray(theta)))), 1.0)
    return float(np.sqrt(-2 * np.log(max(R, 1e-12))) / (2 * np.pi) * 16)


def harmonic_fit(y, theta, extra=None, min_spread_wedges=1.0):
    """y ~ b0 + b1 cos theta + b2 sin theta (+ b3 extra): amplitude, phase (wedge units of the peak), R^2, n.
    The fit is ill-conditioned when theta hardly varies (cos / sin collinear with the intercept), so it is refused
    (nan) unless the circular sd of theta is at least `min_spread_wedges` (1 wedge = 22.5 deg)."""
    y, theta = np.asarray(y, float), np.asarray(theta, float)
    m = np.isfinite(y) & np.isfinite(theta)
    if extra is not None:
        extra = np.asarray(extra, float); m &= np.isfinite(extra)
    nan = dict(amp=float("nan"), phase_wedge=float("nan"), r2=float("nan"), n=int(m.sum()), offset=float("nan"), slope_extra=float("nan"),
               spread_wedges=circ_sd_wedges(theta[m]) if m.sum() > 1 else float("nan"))
    if m.sum() < 8 or not (nan["spread_wedges"] >= min_spread_wedges):
        return nan
    cols = [np.ones(m.sum()), np.cos(theta[m]), np.sin(theta[m])]
    if extra is not None:
        cols.append(extra[m])
    X = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    pred = X @ beta
    ss_res = float(((y[m] - pred) ** 2).sum()); ss_tot = float(((y[m] - y[m].mean()) ** 2).sum())
    amp = float(np.hypot(beta[1], beta[2]))
    phase = float((np.arctan2(beta[2], beta[1]) % (2 * np.pi)) / (2 * np.pi) * 16)
    return dict(amp=amp, phase_wedge=phase, r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), n=int(m.sum()),
                offset=float(beta[0]), slope_extra=float(beta[3]) if extra is not None else float("nan"), spread_wedges=nan["spread_wedges"])


def block_shuffle_null(y, theta, extra=None, block=100, reps=200, seed=0):
    """Amplitude of the first-harmonic fit when theta is shuffled in blocks of `block` frames (1 s at 10 ms): the
    amplitude a fit of this length produces with no relation between y and the bump position (95th percentile)."""
    y, theta = np.asarray(y, float), np.asarray(theta, float)
    n = len(y)
    if n < 2 * block:
        return float("nan")
    rng = np.random.default_rng(seed)
    nb = n // block
    amps = []
    for _ in range(reps):
        order = rng.permutation(nb)
        th = np.concatenate([theta[b * block:(b + 1) * block] for b in order] + [theta[nb * block:]])
        amps.append(harmonic_fit(y, th, extra)["amp"])
    return float(np.nanpercentile(amps, 95))


def bump_frames(epg, wedge_of):
    """Per-frame bump statistics from (F, 46) EPG rates in wedge order. Returns dict of (F,) arrays."""
    epg = np.asarray(epg, np.float64)
    F = epg.shape[0]
    ang = 2 * np.pi * wedge_of / 16
    tot = epg.sum(1)
    z = (epg * np.exp(1j * ang)[None]).sum(1) / np.maximum(tot, 1e-9)
    vs = np.abs(z); vs[tot <= 1e-9] = 0.0
    centre = (np.angle(z) % (2 * np.pi)) / (2 * np.pi) * 16
    w0 = np.round(centre - (BLOCK - 1) / 2).astype(int) % 16
    in_block = np.zeros_like(epg, bool)
    for j in range(BLOCK):
        in_block |= wedge_of[None, :] == ((w0[:, None] + j) % 16)
    above = epg > THRESH_HZ
    in_above = (above & in_block).sum(1); n_in = in_block.sum(1)
    out_above = (above & ~in_block).sum(1); n_out = (~in_block).sum(1)
    in_frac = in_above / np.maximum(n_in, 1)
    confined = (vs > VS_MIN) & (in_frac >= IN_FRAC - 1e-9) & (out_above <= OUT_MAX)
    bump_hz = np.where(n_in > 0, (epg * in_block).sum(1) / np.maximum(n_in, 1), 0.0)
    out_hz = (epg * ~in_block).sum(1) / np.maximum(n_out, 1)
    prof = np.stack([epg[:, wedge_of == w].mean(1) for w in range(16)], 1)      # (F, 16)
    peak = prof.max(1)
    width_half = (prof > 0.5 * peak[:, None]).sum(1).astype(float); width_half[peak <= THRESH_HZ] = np.nan
    width_22 = (prof > THRESH_HZ).sum(1).astype(float)
    return dict(vs=vs, centre=centre, in_above=in_above, n_in=n_in, out_above=out_above, n_out=n_out, in_frac=in_frac,
                confined=confined, bump_hz=bump_hz, out_hz=out_hz, epg_mean=epg.mean(1), epg_max=epg.max(1),
                active=epg.max(1) > THRESH_HZ, width_half=width_half, width_22=width_22, profile=prof)


def fly_metrics(rec: dict, i: int, pulse_at: float, pulse_s: float, seconds: float, dt: float = FRAME_MS / 1000):
    """All per-fly numbers of the audit, from the per-frame record (arrays (F, B, ...)); i = the fly's row."""
    t = rec["t"]
    b = bump_frames(rec["epg"][:, i, :], rec["wedge_of"])
    pre = t < pulse_at
    during = (t >= pulse_at) & (t < pulse_at + pulse_s)
    post = t >= pulse_at + pulse_s
    late = t >= max(pulse_at + pulse_s, seconds - 30.0)      # the last 30 s (task: 'over 30-60 s')
    have_pulse = rec.get("pulse_hz", 0.0) > 0
    start = int(rec["start_wedge"][i]) if have_pulse else -1
    block_centre = (start + (BLOCK - 1) / 2) % 16 if have_pulse else float("nan")
    heading = rec["heading"][:, i]; yaw = rec["yaw"][:, i]
    theta = 2 * np.pi * b["centre"] / 16
    conf = b["confined"]
    m = dict(row=i, env_seed=int(rec["env_seed"][i]), start_wedge=start, block_centre=block_centre)

    def frac(mask):
        return float(conf[mask].mean()) if mask.any() else float("nan")

    def mean_if(x, mask):
        mm = mask & np.isfinite(x)
        return float(np.mean(x[mm])) if mm.any() else float("nan")

    m.update(frac_confined_pre=frac(pre), frac_confined_during=frac(during), frac_confined_post=frac(post), frac_confined_late=frac(late),
             frac_active_pre=float(b["active"][pre].mean()) if pre.any() else float("nan"),
             frac_active_post=float(b["active"][post].mean()) if post.any() else float("nan"),
             epg_mean_pre=mean_if(b["epg_mean"], pre), epg_mean_post=mean_if(b["epg_mean"], post), epg_mean_late=mean_if(b["epg_mean"], late),
             bump_hz_post=mean_if(b["bump_hz"], post & conf), bump_hz_late=mean_if(b["bump_hz"], late & conf),
             out_hz_post=mean_if(b["out_hz"], post & conf), vs_post=mean_if(b["vs"], post & conf), vs_late=mean_if(b["vs"], late & conf),
             vs_post_all=mean_if(b["vs"], post), width_half_post=mean_if(b["width_half"], post & conf),
             width_22_post=mean_if(b["width_22"], post & conf), in_above_post=mean_if(b["in_above"].astype(float), post & conf),
             n_in_post=mean_if(b["n_in"].astype(float), post & conf), out_above_post=mean_if(b["out_above"].astype(float), post & conf),
             in_frac_post_all=mean_if(b["in_frac"], post), out_above_post_all=mean_if(b["out_above"].astype(float), post))
    # first / last confined frame after the pulse; survival = last confined time - pulse end (s)
    pc = np.flatnonzero(post & conf)
    m["t_first_confined_post"] = float(t[pc[0]]) if len(pc) else float("nan")
    m["t_last_confined_post"] = float(t[pc[-1]]) if len(pc) else float("nan")
    m["survival_s"] = float(t[pc[-1]] - (pulse_at + pulse_s)) if len(pc) else 0.0
    pp = np.flatnonzero(pre & conf)
    m["t_first_confined_pre"] = float(t[pp[0]]) if len(pp) else float("nan")
    # rates of the other populations, post-pulse mean (all frames) and over confined frames
    for k in ("pen_L", "pen_R", "d7_L", "d7_R", "peg", "ring", "glno", "pfn", "hdelta", "pfl3_L", "pfl3_R", "dna02_L", "dna02_R", "rest"):
        if k in rec:
            m[f"{k}_post"] = mean_if(rec[k][:, i].astype(float), post)
            m[f"{k}_post_conf"] = mean_if(rec[k][:, i].astype(float), post & conf)
    m["pen_LR_post_conf"] = m.get("pen_L_post_conf", np.nan) - m.get("pen_R_post_conf", np.nan)
    # (a) drift: centre over post-pulse confined frames
    if len(pc) >= 2:
        c = b["centre"][pc]
        d = ((np.diff(c) + 8) % 16) - 8                      # wedge step per frame (signed, shortest arc)
        gaps = np.diff(t[pc])                                # s between consecutive confined frames
        speed = np.abs(d) / np.maximum(gaps, dt)
        m["drift_abs_wedges_per_s"] = float(np.mean(speed))
        m["drift_net_wedges"] = float(np.sum(d))
        # displacement over 1 s windows (100 frames) between confined frames: the drift with the per-frame centre jitter averaged out
        full = np.full(len(t), np.nan); full[pc] = c
        d1 = ((full[100:] - full[:-100] + 8) % 16) - 8
        m["drift_1s_abs_wedges"] = float(np.nanmean(np.abs(d1))) if np.isfinite(d1).any() else float("nan")
        m["drift_1s_max_wedges"] = float(np.nanmax(np.abs(d1))) if np.isfinite(d1).any() else float("nan")
        m["centre_circ_sd_wedges"] = circ_sd_wedges(theta[pc])
        dist0 = np.abs(((c - block_centre + 8) % 16) - 8) if have_pulse else np.full(len(c), np.nan)
        m["dist_from_pulse_block_mean"] = float(np.nanmean(dist0)); m["dist_from_pulse_block_max"] = float(np.nanmax(dist0))
        m["dist_from_pulse_block_final"] = float(dist0[-1])
        m["frac_within_2p5_of_pulse_block"] = float(np.mean(dist0 <= 2.5)) if have_pulse else float("nan")
        # jumps: displacement over 0.5 s (50 frames) exceeding 2 wedges, counted over consecutive confined stretches
        idx = pc
        big = 0
        for j in range(len(idx) - 50):
            if idx[j + 50] - idx[j] == 50:
                if abs(((c[j + 50] - c[j] + 8) % 16) - 8) > 2:
                    big += 1
        m["jump_frames_0p5s"] = int(big)
    else:
        for k in ("drift_abs_wedges_per_s", "drift_net_wedges", "drift_1s_abs_wedges", "drift_1s_max_wedges", "centre_circ_sd_wedges",
                  "dist_from_pulse_block_mean", "dist_from_pulse_block_max", "dist_from_pulse_block_final", "frac_within_2p5_of_pulse_block"):
            m[k] = float("nan")
        m["jump_frames_0p5s"] = 0
    # (b) heading tracking over post-pulse confined frames
    sel = post & conf
    m["heading_range_rad_post"] = float(np.ptp(np.unwrap(heading[post]))) if post.any() else float("nan")
    m["yaw_abs_mean_post"] = float(np.mean(np.abs(yaw[post]))) if post.any() else float("nan")
    m["yaw_mean_post"] = float(np.mean(yaw[post])) if post.any() else float("nan")
    m["heading_circ_sd_deg_post"] = float(np.degrees(circ_sd_wedges(heading[post]) / 16 * 2 * np.pi)) if post.sum() > 1 else float("nan")
    if sel.sum() >= 30:
        m["circ_corr_centre_heading"] = circ_corr(theta[sel], heading[sel])
        # bump angular velocity (rad/s, 0.1 s box) vs yaw rate on the confined stretch
        th_u = np.unwrap(theta)
        vel = np.gradient(th_u, dt)
        k = np.ones(10) / 10
        vel_s = np.convolve(vel, k, mode="same"); yaw_s = np.convolve(yaw, k, mode="same")
        v, y = vel_s[sel], yaw_s[sel]
        if np.std(v) > 0 and np.std(y) > 0:
            m["r_bumpvel_yaw"] = float(np.corrcoef(v, y)[0, 1])
            m["slope_bumpvel_per_yaw"] = float(np.polyfit(y, v, 1)[0])
        else:
            m["r_bumpvel_yaw"] = float("nan"); m["slope_bumpvel_per_yaw"] = float("nan")
        m["bumpvel_abs_mean_rad_s"] = float(np.mean(np.abs(v)))
        m["yaw_abs_mean_conf_rad_s"] = float(np.mean(np.abs(y)))
        # the same with the bump position taken relative to the heading (allocentric anchoring test): if the bump were
        # heading-anchored, theta - heading would be constant -> small circular sd
        m["circ_sd_centre_minus_heading_wedges"] = circ_sd_wedges(theta[sel] - heading[sel])
        m["circ_sd_centre_wedges_conf"] = circ_sd_wedges(theta[sel])
    else:
        for k in ("circ_corr_centre_heading", "r_bumpvel_yaw", "slope_bumpvel_per_yaw", "bumpvel_abs_mean_rad_s", "yaw_abs_mean_conf_rad_s",
                  "circ_sd_centre_minus_heading_wedges", "circ_sd_centre_wedges_conf"):
            m[k] = float("nan")
    # (c) steering asymmetry vs bump position (ring coordinate; and relative to the heading), post-pulse confined frames
    err = rec["prog_err"][:, i] if "prog_err" in rec else None
    for name, L, R in (("pfl3", "pfl3_L", "pfl3_R"), ("dna02", "dna02_L", "dna02_R")):
        if L not in rec:
            continue
        lr = rec[L][:, i].astype(float) - rec[R][:, i].astype(float)
        m[f"{name}_LR_mean_post"] = float(np.mean(lr[post])) if post.any() else float("nan")
        m[f"{name}_LR_sd_post"] = float(np.std(lr[post])) if post.any() else float("nan")
        m[f"{name}_LR_absmean_post"] = float(np.mean(np.abs(lr[post]))) if post.any() else float("nan")
        m[f"{name}_LR_mean_post_conf"] = float(np.mean(lr[sel])) if sel.any() else float("nan")
        ex = err[sel] if err is not None else None
        fit = harmonic_fit(lr[sel], theta[sel], ex)
        m[f"{name}_amp_ring"] = fit["amp"]; m[f"{name}_phase_ring"] = fit["phase_wedge"]; m[f"{name}_r2_ring"] = fit["r2"]; m[f"{name}_n_fit"] = fit["n"]
        m[f"{name}_slope_err"] = fit["slope_extra"]; m[f"{name}_fit_spread_wedges"] = fit["spread_wedges"]
        m[f"{name}_amp_ring_null95"] = block_shuffle_null(lr[sel], theta[sel], ex, seed=i) if (sel.sum() >= 200 and np.isfinite(fit["amp"])) else float("nan")
        fit2 = harmonic_fit(lr[sel], theta[sel] - heading[sel], ex)
        m[f"{name}_amp_rel_heading"] = fit2["amp"]; m[f"{name}_phase_rel_heading"] = fit2["phase_wedge"]; m[f"{name}_r2_rel_heading"] = fit2["r2"]
        # and the yaw rate itself on the bump position (the body-level consequence)
    fit_y = harmonic_fit(yaw[sel], theta[sel], err[sel] if err is not None else None)
    m["yaw_amp_ring"] = fit_y["amp"]; m["yaw_phase_ring"] = fit_y["phase_wedge"]; m["yaw_r2_ring"] = fit_y["r2"]
    # the wedge-binned asymmetry (16 bins) for the pooled table
    bins = np.floor(b["centre"][sel]).astype(int) % 16 if sel.any() else np.zeros(0, int)
    m["_bins"] = bins
    m["_pfl3_lr_sel"] = (rec["pfl3_L"][:, i] - rec["pfl3_R"][:, i]).astype(float)[sel] if "pfl3_L" in rec else np.zeros(0)
    m["_dna02_lr_sel"] = (rec["dna02_L"][:, i] - rec["dna02_R"][:, i]).astype(float)[sel] if "dna02_L" in rec else np.zeros(0)
    m["_yaw_sel"] = yaw[sel]
    # body summary
    m["airborne_frac"] = float(rec["airborne"][:, i].mean())
    m["speed_mean"] = float(rec["speed"][:, i].mean())
    m["path_m"] = float(np.sum(np.hypot(np.diff(rec["x"][:, i]), np.diff(rec["y"][:, i]))))
    return m


def condition_summary(flies: list[dict]) -> dict:
    """Mean / sd / min / max over flies of every numeric metric, plus the 16-bin pooled asymmetry."""
    keys = [k for k in flies[0] if not k.startswith("_") and isinstance(flies[0][k], (int, float, np.floating, np.integer))]
    s = {}
    for k in keys:
        v = np.array([f[k] for f in flies], float)
        ok = np.isfinite(v)
        s[k] = dict(mean=float(v[ok].mean()) if ok.any() else float("nan"), sd=float(v[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"),
                    min=float(v[ok].min()) if ok.any() else float("nan"), max=float(v[ok].max()) if ok.any() else float("nan"), n=int(ok.sum()))
    bins = np.concatenate([f["_bins"] for f in flies]); pf = np.concatenate([f["_pfl3_lr_sel"] for f in flies])
    dn = np.concatenate([f["_dna02_lr_sel"] for f in flies]); yw = np.concatenate([f["_yaw_sel"] for f in flies])
    s["by_wedge"] = dict(n=[int((bins == w).sum()) for w in range(16)],
                         pfl3_LR=[float(pf[bins == w].mean()) if (bins == w).any() else None for w in range(16)],
                         dna02_LR=[float(dn[bins == w].mean()) if (bins == w).any() else None for w in range(16)],
                         yaw=[float(yw[bins == w].mean()) if (bins == w).any() else None for w in range(16)])
    # pooled first-harmonic fit over all flies' confined frames (ring coordinate)
    if len(bins) >= 8:
        th = 2 * np.pi * (bins + 0.5) / 16
        s["pooled_fit"] = dict(pfl3=harmonic_fit(pf, th), dna02=harmonic_fit(dn, th), yaw=harmonic_fit(yw, th), n=int(len(bins)))
    # per-fly tile position vs per-fly mean asymmetry (16 flies at 8 tiles): the across-fly test
    tiles = np.array([f["block_centre"] for f in flies], float)
    if np.isfinite(tiles).all() and len(flies) >= 6:
        th = 2 * np.pi * tiles / 16
        # over confined frames (bump runs) and over every post-pulse frame (defined for the no-bump control as well: its
        # amplitude is what 16 flies' mean asymmetries produce against the tile they were pulsed at with no bump present)
        s["across_fly_fit"] = dict(pfl3=harmonic_fit([f.get("pfl3_LR_mean_post_conf", np.nan) for f in flies], th),
                                   dna02=harmonic_fit([f.get("dna02_LR_mean_post_conf", np.nan) for f in flies], th))
        s["across_fly_fit_all"] = dict(pfl3=harmonic_fit([f.get("pfl3_LR_mean_post", np.nan) for f in flies], th),
                                       dna02=harmonic_fit([f.get("dna02_LR_mean_post", np.nan) for f in flies], th),
                                       yaw_mean=harmonic_fit([f.get("yaw_mean_post", np.nan) for f in flies], th))
    return s


# ------------------------------------------------------------------------------------------------ run (GPU)
def run(a):
    import torch
    assert torch.cuda.is_available() or a.device == "cpu", "CUDA is not available on this node (resubmit the job)"
    from flyverse import BatchSim
    tpg, adapt = patch_lif_params(a.gE, a.gD, a.receptor_model)
    t0 = time.perf_counter()
    B = a.batch
    env_seeds = [a.seed * B + i for i in range(B)] if not a.seeds else [int(s) for s in a.seeds.split(",")]
    start = tuple(float(v) for v in a.start.split(","))
    if len(start) == 2:
        start += (0.75,)
    sim = BatchSim(B, seeds=env_seeds, seed=a.seed, start=start, program=a.program, fruit_set=a.fruit, fence=not a.no_fence,
                   cuda_graphs=not a.no_graphs and a.device != "cpu", cuda_kernels=False, event_driven=None, cuda_sparse="torch",
                   device=a.device)
    fb, c = sim.fb, sim.c
    lp = fb.brain.p
    dev = str(fb.device)
    print(f"device {dev} ({torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'cpu'}); torch {torch.__version__}", flush=True)
    print(f"BatchSim B={B} neurons={c.n:,} program={a.program} fruit={a.fruit} fence={not a.no_fence} env seeds={env_seeds} brain seed={a.seed}", flush=True)
    print(f"LIFParams: receptor_model {lp.receptor_model} ({lp.receptor_net_rule if lp.receptor_model else None}); adapt_by_type {lp.adapt_by_type}; "
          f"type_path_gain {lp.type_path_gain if lp.type_path_gain is not None else 'DEFAULT ' + str(brain.DEFAULT_TYPE_PATH_GAIN)}", flush=True)
    print(f"backend: cuda kernels {fb.brain.cuda} event_driven {fb.brain.event_driven} cuda_graphs {fb.cuda_graphs}; sum|W| {float(abs(c.W).sum()):.0f}", flush=True)
    for seed, fly, m in zip(sim.seeds, sim.flies, sim.metabolisms):
        fly.heading = np.random.default_rng(seed).uniform(-np.pi, np.pi)
        m.energy = a.energy
    # cells
    cells = cx_wedge.compass_cells(c)
    epg = cells["EPG"]; idx_epg = epg["idx"]; wedge_of = np.asarray(np.round(epg["pos"]), int) % 16
    side = c.neurons.somaSide.fillna("").to_numpy()
    ty = c.neurons.type.fillna("").to_numpy()

    def by_side(idx, s):
        return idx[side[idx] == s]
    groups = {"pen_L": by_side(cells["PEN"]["idx"], "L"), "pen_R": by_side(cells["PEN"]["idx"], "R"),
              "d7_L": by_side(cells["Delta7"]["idx"], "L"), "d7_R": by_side(cells["Delta7"]["idx"], "R"),
              "peg": cells["PEG"]["idx"], "ring": cells["Ring"]["idx"], "glno": np.flatnonzero(ty == "GLNO"),
              "pfn": np.flatnonzero([t.startswith("PFN") for t in ty]), "hdelta": np.flatnonzero([t.startswith("hDelta") for t in ty]),
              "pfl3_L": c.select(type="PFL3", somaSide="L"), "pfl3_R": c.select(type="PFL3", somaSide="R"),
              "dna02_L": fb.groups.turn_L, "dna02_R": fb.groups.turn_R}
    others = np.setdiff1d(np.arange(c.n), np.concatenate([idx_epg, cells["PEN"]["idx"], cells["Delta7"]["idx"], cells["PEG"]["idx"], cells["EPGt"]["idx"]]))
    print("cells: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()) + f"; EPG {len(idx_epg)}; rest {len(others)}", flush=True)
    all_idx = np.concatenate([idx_epg] + [groups[k] for k in groups])
    bounds = np.cumsum([0, len(idx_epg)] + [len(groups[k]) for k in groups])
    names = ["epg"] + list(groups)
    count = max(1, round(a.seconds * 1000 / FRAME_MS))
    F = count
    rec = {"t": (np.arange(F) + 1) * FRAME_MS / 1000, "epg": np.zeros((F, B, len(idx_epg)), np.float16),
           "heading": np.zeros((F, B), np.float32), "yaw": np.zeros((F, B), np.float32), "speed": np.zeros((F, B), np.float32),
           "x": np.zeros((F, B), np.float32), "y": np.zeros((F, B), np.float32), "airborne": np.zeros((F, B), bool),
           "mode": np.zeros((F, B), np.int8), "prog_err": np.full((F, B), np.nan, np.float32), "rest": np.zeros((F, B), np.float32),
           "energy": np.zeros((F, B), np.float32)}
    for k in groups:
        rec[k] = np.zeros((F, B), np.float32)
    modes = {}
    start_wedge = np.array([(2 * i) % 16 for i in range(B)])
    pulse_frame = int(round(a.pulse_at * 1000 / FRAME_MS))
    if a.background > 0:
        fb.stimulate(idx_epg, a.background, a.seconds * 1000 + 100)
    started = time.perf_counter()
    for k in range(count):
        if a.pulse_hz > 0 and k == pulse_frame:
            hz = np.zeros((B, len(idx_epg)), np.float32)
            for i in range(B):
                on = np.isin(wedge_of, [(start_wedge[i] + j) % 16 for j in range(BLOCK)])
                hz[i, on] = a.background + a.pulse_hz
            fb.stimulate(idx_epg, hz, a.pulse_s * 1000)
            print(f"t={sim.fb.t / 1000:.2f}s pulse: {a.pulse_hz} Hz on {BLOCK} wedges for {a.pulse_s} s; start wedges {start_wedge.tolist()}", flush=True)
        sim.step()
        r = fb.brain.rates(all_idx)                      # (B, n_all), one device sync
        for j, name in enumerate(names):
            blk = r[:, bounds[j]:bounds[j + 1]]
            if name == "epg":
                rec["epg"][k] = blk.astype(np.float16)
            else:
                rec[name][k] = blk.mean(1) if blk.shape[1] else 0.0
        rec["rest"][k] = fb.brain.mean_rate(others)
        for i, fly in enumerate(sim.flies):
            rec["heading"][k, i] = fly.heading; rec["yaw"][k, i] = fly.yaw_rate; rec["speed"][k, i] = fly.speed
            rec["x"][k, i] = fly.x; rec["y"][k, i] = fly.y; rec["airborne"][k, i] = fly.airborne
            rec["energy"][k, i] = sim.metabolisms[i].energy
            cmd = sim.commands[i]
            mode = "feeding" if sim.feeding[i] else cmd.get("mode", "plain")
            rec["mode"][k, i] = modes.setdefault(mode, len(modes))
            if "steer err x10" in cmd.get("rates", {}):
                rec["prog_err"][k, i] = cmd["rates"]["steer err x10"] / 10.0
        if (k + 1) % int(round(a.log_every * 1000 / FRAME_MS)) == 0 or k + 1 == count:
            e = rec["epg"][k].astype(float)
            bf = bump_frames(e, wedge_of)
            print(f"t={sim.fb.t / 1000:.1f}s EPG mean {e.mean():.1f} Hz; confined {int(bf['confined'].sum())}/{B}; vs {np.round(bf['vs'], 2).tolist()}; "
                  f"centre {np.round(bf['centre'], 1).tolist()}; bump Hz {np.round(bf['bump_hz'], 0).tolist()}; "
                  f"PEN {(rec['pen_L'][k].mean() + rec['pen_R'][k].mean()) / 2:.1f} D7 {(rec['d7_L'][k].mean() + rec['d7_R'][k].mean()) / 2:.1f} "
                  f"ring {rec['ring'][k].mean():.2f} PFL3 {(rec['pfl3_L'][k].mean() + rec['pfl3_R'][k].mean()) / 2:.2f} "
                  f"DNa02 {(rec['dna02_L'][k].mean() + rec['dna02_R'][k].mean()) / 2:.2f} rest {rec['rest'][k].mean():.3f}; "
                  f"airborne {int(rec['airborne'][k].sum())}; {(time.perf_counter() - started) / (k + 1) * 1000:.0f} ms/frame", flush=True)
    wall = time.perf_counter() - started
    rec.update(wedge_of=wedge_of, epg_label=np.array(epg["label"]), start_wedge=start_wedge, env_seed=np.array(env_seeds),
               pulse_hz=a.pulse_hz, pulse_at=a.pulse_at, pulse_s=a.pulse_s, background=a.background, seconds=a.seconds,
               mode_names=np.array(list(modes)))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    npz = out.with_suffix(".npz")
    np.savez_compressed(npz, **rec)
    flies = [fly_metrics(rec, i, a.pulse_at, a.pulse_s, a.seconds) for i in range(B)]
    summary = condition_summary(flies)
    receptor_info = dict(model=lp.receptor_model, net_rule=lp.receptor_net_rule if lp.receptor_model else None,
                         fast_sign_changed_entries=int((fb.receptor.fast_sign != np.sign(c.W.data)).sum()) if fb.receptor is not None else 0)
    result = dict(config=dict(gE=a.gE, gD=a.gD, program=a.program, seed=a.seed, env_seeds=env_seeds, batch=B, seconds=a.seconds,
                              pulse_hz=a.pulse_hz, pulse_at=a.pulse_at, pulse_s=a.pulse_s, background=a.background, energy=a.energy,
                              start=start, fruit=a.fruit, fence=not a.no_fence, receptor_model=a.receptor_model,
                              type_path_gain=[list(x) for x in (tpg or brain.DEFAULT_TYPE_PATH_GAIN)], adapt_by_type=adapt,
                              adapt_default=brain.DEFAULT_ADAPT_BY_TYPE, thresh_hz=THRESH_HZ, vs_min=VS_MIN, in_frac=IN_FRAC, out_max=OUT_MAX, block=BLOCK),
                  device=dev, torch=torch.__version__, cuda_kernels=bool(fb.brain.cuda), event_driven=bool(fb.brain.event_driven), cuda_graphs=bool(fb.cuda_graphs),
                  receptor=receptor_info, sum_abs_W=float(abs(c.W).sum()), n_cells={k: int(len(v)) for k, v in groups.items()}, n_epg=int(len(idx_epg)),
                  frames=count, wall_s=wall, ms_per_frame=wall / count * 1000, setup_s=started - t0 + 0.0,
                  body=[dict(row=i, energy=float(m.energy), meals=int(m.meals), hops=int(sim.hops_escape[i] + sim.hops_voluntary[i])) for i, m in enumerate(sim.metabolisms)],
                  mode_names=list(modes), flies=[{k: v for k, v in f.items() if not k.startswith("_")} for f in flies], summary=summary, npz=str(npz))
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print_condition(result)
    print(f"-> {out}, {npz}; {wall:.0f} s for {count} frames ({wall / count * 1000:.0f} ms/frame), setup {started - t0:.0f} s", flush=True)


def print_condition(result: dict):
    s = result["summary"]; cfg = result["config"]

    def f(k, d=2):
        v = s.get(k)
        return f"{v['mean']:.{d}f} +- {v['sd']:.{d}f} [{v['min']:.{d}f}, {v['max']:.{d}f}] (n {v['n']})" if v and np.isfinite(v["mean"]) else "nan"
    print(f"\n== gE {cfg['gE']} gD {cfg['gD']} program {cfg['program']} seed {cfg['seed']} receptor {result['receptor']['model']} ==")
    print(f"confined fraction: pre {f('frac_confined_pre')}; during {f('frac_confined_during')}; post {f('frac_confined_post')}; last 30 s {f('frac_confined_late')}")
    print(f"active (any EPG > 22 Hz): pre {f('frac_active_pre')}; post {f('frac_active_post')}; EPG mean pre {f('epg_mean_pre', 1)} post {f('epg_mean_post', 1)}")
    print(f"survival after pulse (s): {f('survival_s', 1)}; bump Hz post {f('bump_hz_post', 0)}; out Hz {f('out_hz_post', 1)}; vs {f('vs_post')}; width half-max {f('width_half_post', 1)} wedges")
    print(f"drift |d|/s {f('drift_abs_wedges_per_s')} wedges/s; circ sd {f('centre_circ_sd_wedges')}; dist from pulse block mean {f('dist_from_pulse_block_mean')} final {f('dist_from_pulse_block_final')}; within 2.5 {f('frac_within_2p5_of_pulse_block')}; jumps {f('jump_frames_0p5s', 0)}")
    print(f"PEN L {f('pen_L_post', 1)} R {f('pen_R_post', 1)}; Delta7 {f('d7_L_post', 1)} / {f('d7_R_post', 1)}; ring {f('ring_post')}; GLNO {f('glno_post', 1)}; PFN {f('pfn_post')}; hDelta {f('hdelta_post')}; rest {f('rest_post', 3)}")
    print(f"heading: circ corr(centre, heading) {f('circ_corr_centre_heading')}; r(bump vel, yaw) {f('r_bumpvel_yaw')}; slope {f('slope_bumpvel_per_yaw')}; |bump vel| {f('bumpvel_abs_mean_rad_s')} rad/s vs |yaw| {f('yaw_abs_mean_conf_rad_s')} rad/s; "
          f"circ sd(centre - heading) {f('circ_sd_centre_minus_heading_wedges')} vs sd(centre) {f('circ_sd_centre_wedges_conf')} wedges; heading circ sd {f('heading_circ_sd_deg_post', 0)} deg")
    print(f"PFL3 L-R: mean {f('pfl3_LR_mean_post')} sd {f('pfl3_LR_sd_post')} Hz; amp on ring {f('pfl3_amp_ring')} (null95 {f('pfl3_amp_ring_null95')}) r2 {f('pfl3_r2_ring')}; amp rel heading {f('pfl3_amp_rel_heading')}")
    print(f"DNa02 L-R: mean {f('dna02_LR_mean_post')} sd {f('dna02_LR_sd_post')} Hz; amp on ring {f('dna02_amp_ring')} (null95 {f('dna02_amp_ring_null95')}) r2 {f('dna02_r2_ring')}; amp rel heading {f('dna02_amp_rel_heading')}; yaw amp {f('yaw_amp_ring', 3)} rad/s")
    if "pooled_fit" in s:
        p = s["pooled_fit"]
        print(f"pooled over flies ({p['n']} confined frames): PFL3 amp {p['pfl3']['amp']:.3f} phase {p['pfl3']['phase_wedge']:.1f} r2 {p['pfl3']['r2']:.3f}; "
              f"DNa02 amp {p['dna02']['amp']:.3f} phase {p['dna02']['phase_wedge']:.1f} r2 {p['dna02']['r2']:.3f}; yaw amp {p['yaw']['amp']:.3f}")
    if "across_fly_fit" in s:
        p, q = s["across_fly_fit"], s["across_fly_fit_all"]
        print(f"across flies (per-fly mean vs pulsed tile; confined frames | all post frames): PFL3 amp {p['pfl3']['amp']:.3f} r2 {p['pfl3']['r2']:.3f} | {q['pfl3']['amp']:.3f} r2 {q['pfl3']['r2']:.3f}; "
              f"DNa02 amp {p['dna02']['amp']:.3f} r2 {p['dna02']['r2']:.3f} | {q['dna02']['amp']:.3f} r2 {q['dna02']['r2']:.3f}; yaw mean amp {q['yaw_mean']['amp']:.3f} rad/s r2 {q['yaw_mean']['r2']:.3f}")
    bw = s["by_wedge"]
    print("by centre wedge (n; PFL3 L-R; DNa02 L-R): " + " ".join(f"{w}:{bw['n'][w]}/{bw['pfl3_LR'][w] if bw['pfl3_LR'][w] is None else round(bw['pfl3_LR'][w], 2)}/{bw['dna02_LR'][w] if bw['dna02_LR'][w] is None else round(bw['dna02_LR'][w], 2)}" for w in range(16)))
    print("body: " + " ".join(f"{b['row']}:{b['energy']:.2f}/{b['meals']}/{b['hops']}" for b in result["body"]))


# ------------------------------------------------------------------------------------------------ report (CPU)
def report(files, table: str):
    import pandas as pd
    paths = []
    for pat in files:
        paths += sorted(Path().glob(pat)) if any(ch in pat for ch in "*?[") else [Path(pat)]
    rows, conds = [], []
    for p in paths:
        if not p.exists():
            print(f"missing {p}"); continue
        res = json.loads(p.read_text(encoding="utf-8"))
        cfg = res["config"]
        npz = Path(res["npz"]) if Path(res["npz"]).exists() else p.with_suffix(".npz")
        if npz.exists():                                     # recompute from the record (the JSON's numbers are the same code)
            rec = dict(np.load(npz, allow_pickle=False))
            rec["pulse_hz"] = float(rec["pulse_hz"])
            flies = [fly_metrics(rec, i, float(rec["pulse_at"]), float(rec["pulse_s"]), float(rec["seconds"])) for i in range(rec["epg"].shape[1])]
            res["flies"] = [{k: v for k, v in f.items() if not k.startswith("_")} for f in flies]
            res["summary"] = condition_summary(flies)
        print_condition(res)
        key = dict(gE=cfg["gE"], gD=cfg["gD"], program=cfg["program"], seed=cfg["seed"], receptor=res["receptor"]["model"], file=str(p))
        for f in res["flies"]:
            rows.append({**key, **f})
        conds.append((key, res))
    if not rows:
        print("no rows"); return
    df = pd.DataFrame(rows)
    out = Path(table)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_name(out.name + "_flies.csv"), index=False)
    cols = [("frac_confined_pre", 2), ("frac_confined_post", 2), ("frac_confined_late", 2), ("survival_s", 1), ("bump_hz_post", 0), ("out_hz_post", 1),
            ("vs_post", 2), ("width_half_post", 1), ("drift_abs_wedges_per_s", 2), ("drift_1s_abs_wedges", 2), ("centre_circ_sd_wedges", 2), ("dist_from_pulse_block_final", 1),
            ("jump_frames_0p5s", 0), ("circ_corr_centre_heading", 2), ("r_bumpvel_yaw", 2), ("bumpvel_abs_mean_rad_s", 3), ("yaw_abs_mean_conf_rad_s", 2),
            ("pfl3_LR_mean_post", 2), ("pfl3_LR_sd_post", 2), ("pfl3_amp_ring", 2), ("pfl3_amp_ring_null95", 2), ("pfl3_r2_ring", 2),
            ("dna02_LR_mean_post", 2), ("dna02_LR_sd_post", 2), ("dna02_amp_ring", 2), ("dna02_amp_ring_null95", 2), ("dna02_r2_ring", 2),
            ("pen_L_post", 1), ("pen_R_post", 1), ("d7_L_post", 1), ("d7_R_post", 1), ("ring_post", 2), ("pfn_post", 2), ("hdelta_post", 2), ("rest_post", 3)]
    hdr = ["gE", "gD", "program", "seed", "receptor", "flies"] + [c for c, _ in cols]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for key, res in conds:
        s = res["summary"]
        cells = [str(key["gE"]), str(key["gD"]), key["program"], str(key["seed"]), str(key["receptor"]), str(len(res["flies"]))]
        for c, d in cols:
            v = s.get(c)
            cells.append(f"{v['mean']:.{d}f} +- {v['sd']:.{d}f} ({v['min']:.{d}f}-{v['max']:.{d}f}; n {v['n']})" if v and np.isfinite(v["mean"]) else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    md = "## Per condition (mean +- sd over flies (min-max; n flies with a value); rates in Hz; post = after the pulse)\n\n" + "\n".join(lines) + "\n"
    # per-fly table
    fcols = ["gE", "gD", "program", "seed", "row", "env_seed", "start_wedge", "frac_confined_pre", "frac_confined_post", "survival_s", "bump_hz_post", "vs_post",
             "drift_abs_wedges_per_s", "dist_from_pulse_block_final", "circ_corr_centre_heading", "r_bumpvel_yaw", "pfl3_LR_mean_post_conf", "pfl3_amp_ring",
             "pfl3_amp_ring_null95", "dna02_LR_mean_post_conf", "dna02_amp_ring", "dna02_amp_ring_null95", "airborne_frac", "path_m"]
    fcols = [c for c in fcols if c in df.columns]
    lines = ["| " + " | ".join(fcols) + " |", "|" + "---|" * len(fcols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{r[c]:.3g}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in fcols) + " |")
    md += "\n## Per fly\n\n" + "\n".join(lines) + "\n"
    # by-wedge pooled table per condition
    md += "\n## Steering asymmetry by bump centre wedge (confined post-pulse frames pooled over the condition's flies; n frames / PFL3 L-R / DNa02 L-R)\n\n"
    md += "| condition | " + " | ".join(str(w) for w in range(16)) + " |\n|---|" + "---|" * 16 + "\n"
    for key, res in conds:
        bw = res["summary"]["by_wedge"]
        md += f"| gE {key['gE']} gD {key['gD']} {key['program']} s{key['seed']} | " + " | ".join(
            f"{bw['n'][w]}/{'-' if bw['pfl3_LR'][w] is None else round(bw['pfl3_LR'][w], 2)}/{'-' if bw['dna02_LR'][w] is None else round(bw['dna02_LR'][w], 2)}" for w in range(16)) + " |\n"
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"\n-> {out.with_suffix('.md')}, {out.with_name(out.name + '_flies.csv')}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gE", type=float, default=None, help="EPG <-> PEN / PEG gain (omit = the shipped default, no compass gains)")
    ap.add_argument("--gD", type=float, default=None, help="Delta7 -> EPG gain (with --gE)")
    ap.add_argument("--receptor-model", choices=("default", "off"), default="default")
    ap.add_argument("--seed", type=int, default=0, help="batched brain RNG seed; environment seeds are seed*batch .. +batch-1 unless --seeds")
    ap.add_argument("--seeds", default="", help="explicit comma-separated environment seeds (length = batch)")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--program", default="none", help="'none' | 'cx' | ... (programs.make_program)")
    ap.add_argument("--out", default="out/cxroom/cxroom.json")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pulse-hz", type=float, default=40.0, help="EPG pulse on 4 wedges per fly (row i from wedge 2i mod 16); 0 = none")
    ap.add_argument("--pulse-at", type=float, default=20.0, help="s; before it the ring gets no drive of any kind")
    ap.add_argument("--pulse-s", type=float, default=2.0)
    ap.add_argument("--background", type=float, default=0.0, help="held Poisson background on every EPG (Hz; cx_wedge's protocol used 10; default 0 = senses only)")
    ap.add_argument("--energy", type=float, default=0.9)
    ap.add_argument("--start", default="-0.15,0.15", help="x,y[,z] shared initial position (batch_sustain's default)")
    ap.add_argument("--fruit", choices=("all", "apple"), default="apple")
    ap.add_argument("--no-fence", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--log-every", type=float, default=5.0)
    ap.add_argument("--report", action="store_true", help="CPU: tables from --files")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--table", default="out/cxroom/compass_room")
    a = ap.parse_args()
    if a.report:
        report(a.files or ["out/cxroom/cxroom_*.json"], a.table)
        return
    if (a.gE is None) != (a.gD is None):
        ap.error("--gE and --gD go together")
    run(a)


if __name__ == "__main__":
    main()
