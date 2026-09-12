"""Small moving object: does the lobula's small-field pathway (T2 / T3 / Tm -> LC11 / LC10) carry it?

The decisive small-object test of docs/NOTES.md session 9 ('Relative motion: the small-object pathway is absent, not
untested'), written as a script so it can be run under the receptor model (docs/NT_INTEGRATION.md, round 2).

Protocol (pinned fly, as scripts/probe_figure_ground.py / screen_object.py):
  * the fly stands on the fenced single-apple table with the apple removed (empty table), at (-0.20, 0.10, 0.75) facing
    -y, so that above the table's horizon it sees only the plain wall at y = -2 (the door on the +x wall begins at
    57 deg to its left, beyond the sweep); wind off;
  * a 1 cm (diameter) black ball rests on the table 5 cm ahead of the eye and sweeps 12 cm laterally (+-6 cm, i.e.
    +-50 deg of azimuth, ~46 deg/s at the centre, ~11 deg across) in 3 s, left -> right then right -> left, repeatedly;
  * versus the same timeline with the ball parked out of the scene (no ball).
Both conditions settle for `--settle` s (ball hidden) before the `--seconds` s sweep window is scored.

Recorded over the sweep window, per cell: for the spiking visual projection neurons (LC11, LC10a, LC10b, LC16, LPLC2,
LC4) the optic-lobe drive they receive (mV; brain.drive, the optic lobe's injected current) and their spike rate (Hz,
from brain.spike_counts); for the optic-lobe rate units (T2, T3, Tm5Y, TmY21, TmY13, TmY5a, and Mi4 / Tm3 / Mi1 as the
medulla reference) the deviation from the operating point (rate units, optic.rates() - r0).

Pass criterion (docs/audits/object_sweep.md): LC11 or LC10a peak per-cell drive > 7 mV or rate > 1 Hz with the ball
and not without.

    python scripts/probe_object_sweep.py --receptor-model {off,sign} --receptor-net-rule {class,abs,nonmda}
                                         [--seed 0] [--seconds 12] [--rectify-t2t3] [--cache-dir DIR] [--out out/obj.json]

--rectify-t2t3 sets the T2 / T3 baseline to 0 in optic.DEFAULT_BASELINE_BY_TYPE before the model is built (ReLU units
like T4 / T5): the critic's ON/OFF-cancellation hypothesis (NOTES session 9, 'Critic's follow-ups').
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402

pygame.init(); pygame.display.set_mode((64, 64))
sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402

import room_demo as rd  # noqa: E402
from flyverse import brain, connectome, optic  # noqa: E402

SPIKING = ["LC11", "LC10a", "LC10b", "LC16", "LPLC2", "LC4"]
RATE = ["T2", "T3", "Tm5Y", "TmY21", "TmY13", "TmY5a", "Mi4", "Tm3", "Mi1"]
POS = (-0.20, 0.10, 0.75)          # on the table top, 45 cm from the (removed) apple's spot
HEADING = -np.pi / 2               # facing -y: plain wall behind the table edge
BALL_R = 0.005                     # 1 cm diameter
AHEAD = 0.05                       # m ahead of the eye
HALF_SWEEP = 0.06                  # +-6 cm = 12 cm lateral sweep
SWEEP_S = 3.0                      # s per one-way sweep
FRAME_S = rd.FRAME_MS / 1000.0
PASS_DRIVE_MV = 7.0
PASS_RATE_HZ = 1.0


def patch_receptor(model, net_rule):
    """Make every brain.LIFParams built from here on (room_demo.Sim's included) carry the receptor model
    (LIFParams.receptor_model; docs/NT_INTEGRATION.md). 'off' leaves the class untouched."""
    if model in (None, "off"):
        return
    L = brain.LIFParams

    def make(**kw):
        p = L(**kw); p.receptor_model = model; p.receptor_net_rule = net_rule
        return p
    brain.LIFParams = make


def patch_cache(cache_dir):
    """Point connectome.load at another cache directory (e.g. one built with a different TYPE_NT_OVERRIDE)."""
    if not cache_dir:
        return
    orig = connectome.load

    def load(cache_dir_=None, **kw):
        return orig(cache_dir=Path(cache_dir), **kw)
    connectome.load = load


def ball_offset(t):
    """Lateral offset (m, +left) of the ball at sweep time t: a triangle wave, left -> right first."""
    phase = (t / SWEEP_S) % 2.0
    frac = phase if phase < 1.0 else 2.0 - phase          # 0 -> 1 -> 0
    return HALF_SWEEP - 2 * HALF_SWEEP * frac              # +6 cm (left) -> -6 cm (right) -> +6 cm


class Rec:
    """Per-cell online statistics over the sweep window for one group of cells."""

    def __init__(self, n):
        self.n = 0; self.sum = np.zeros(n); self.sum_abs = np.zeros(n); self.max = np.full(n, -np.inf); self.min = np.full(n, np.inf)
        self.trace = []                                    # per-frame population mean (signed) and mean |x|

    def add(self, x):
        self.n += 1; self.sum += x; self.sum_abs += np.abs(x); np.maximum(self.max, x, out=self.max); np.minimum(self.min, x, out=self.min)
        self.trace.append((float(x.mean()), float(np.abs(x).mean())))

    def mean(self):
        return self.sum / max(self.n, 1)


def run(with_ball, args, seed):
    sim = rd.Sim(seed, start=POS, trail_seconds=0.0, fruit_set="apple", fence=True, wind_speed=0.0,
                 cuda_kernels=True, cuda_graphs=False, event_driven=True, cuda_sparse="warp")
    dev = sim.fb.brain.device
    print(f"[{'ball' if with_ball else 'none'}] sim ready; device {dev}; cuda available {torch.cuda.is_available()}", flush=True)
    assert dev.type == "cuda", f"device {dev}: not CUDA (node race; resubmit)"
    for i, s in enumerate(sim.world.spheres):               # empty table: the apple goes out of the scene
        if s.material == "apple":
            sim.world.move_sphere(i, (9, 9, 9))
    sim.world.move_sphere(sim.loom_idx, (9, 9, 9), (BALL_R, BALL_R, BALL_R))      # the ball, hidden
    c = sim.c; o = sim.fb.optic; b = sim.fb.brain
    types = c.neurons.type.fillna("").to_numpy()
    ridx = np.asarray(o.rate_idx); inv = -np.ones(c.n, np.int64); inv[ridx] = np.arange(len(ridx))
    spk_cells = {t: np.flatnonzero(types == t) for t in SPIKING}
    rate_cells = {t: inv[np.flatnonzero(types == t)] for t in RATE}
    for t, ix in rate_cells.items():
        assert (ix >= 0).all(), f"{t}: not all cells are optic rate units"
    fly = sim.fly
    fly.place(*POS, heading=HEADING)
    eye0 = fly.eye_pos.copy(); fwd = fly.forward.copy(); left = fly.left.copy()
    top_z = sim.info["table_top_z"]
    ball_z = top_z + BALL_R                                  # resting on the table
    print(f"  eye {np.round(eye0, 4).tolist()} forward {np.round(fwd, 3).tolist()} left {np.round(left, 3).tolist()}; "
          f"ball centre z {ball_z:.4f} ({(ball_z - eye0[2]) * 1000:+.1f} mm above the eye, elevation {np.degrees(np.arctan2(ball_z - eye0[2], AHEAD)):+.1f} deg at azimuth 0)", flush=True)
    n_settle = int(round(args.settle / FRAME_S)); n_sweep = int(round(args.seconds / FRAME_S))
    rec_drive = {t: Rec(len(ix)) for t, ix in spk_cells.items()}
    rec_rate = {t: Rec(len(ix)) for t, ix in rate_cells.items()}
    drive_frames = {t: np.zeros((n_sweep, len(ix)), np.float32) for t, ix in spk_cells.items()}   # per-frame per-cell drive, mV
    spk_frames = {t: np.zeros((n_sweep, len(ix)), np.float32) for t, ix in spk_cells.items()}     # per-frame per-cell spikes
    counts0 = None; prev_counts = None; ball_pos = []; radiance_diag = []
    t0 = time.time()
    for k in range(n_settle + n_sweep):
        fly.place(*POS, heading=HEADING)
        if k >= n_settle and with_ball:
            s = ball_offset((k - n_settle) * FRAME_S)
            centre = eye0 + AHEAD * fwd + s * left; centre[2] = ball_z
            sim.world.move_sphere(sim.loom_idx, centre)
            ball_pos.append(float(s))
        elif k >= n_settle:
            ball_pos.append(float("nan"))
        sim.step()
        if with_ball and k >= n_settle and (k - n_settle) in (0, 75, 150, 225):
            # is the ball in the picture?  column radiance with the ball vs the same pose with the ball hidden (two fresh
            # traces at the pinned pose: sim.step() lets the body walk after its own trace, so the pose is re-pinned first)
            fly.place(*POS, heading=HEADING)
            rad_b = sim.column_radiance().detach().cpu().numpy().sum(1)
            sim.world.move_sphere(sim.loom_idx, (9, 9, 9)); rad_n = sim.column_radiance().detach().cpu().numpy().sum(1)
            sim.world.move_sphere(sim.loom_idx, centre)
            rel = (rad_b - rad_n) / (rad_n + 1e-9); hit = np.flatnonzero(np.abs(rel) > 0.05)
            cd = np.asarray(sim.fb.retina.col_dir); az = np.degrees(np.arctan2(cd[:, 1], cd[:, 0])); el = np.degrees(np.arcsin(np.clip(cd[:, 2], -1, 1)))
            radiance_diag.append({"t_s": (k - n_settle) * FRAME_S, "ball_offset_m": float(s), "columns_changed_5pct": int(len(hit)),
                                  "columns_darkened_50pct": int((rel < -0.5).sum()),
                                  "min_rel_radiance": float(rel.min()), "az_deg": [float(v) for v in az[hit]], "el_deg": [float(v) for v in el[hit]]})
            print(f"  ball at {s:+.3f} m: {len(hit)} of {len(rel)} columns change radiance by > 5 % (min ratio {1 + rel.min():.2f}); "
                  f"azimuth {np.round(az[hit], 0).tolist()} elevation {np.round(el[hit], 0).tolist()}", flush=True)
        if k == n_settle - 1:                                # spike counts at the end of the settle: the window's origin
            counts0 = b.spike_counts[0].detach().cpu().numpy().ravel().copy(); prev_counts = counts0
        if k < n_settle:
            continue
        j = k - n_settle
        drive = b.drive[0].detach().cpu().numpy().ravel()
        counts = b.spike_counts[0].detach().cpu().numpy().ravel()
        if counts0 is None:                                  # --settle 0
            counts0 = counts.copy(); prev_counts = counts0
        dr = (o.rates()[0] - o.r0[0]).detach().cpu().numpy().ravel()
        for t, ix in spk_cells.items():
            rec_drive[t].add(drive[ix]); drive_frames[t][j] = drive[ix]; spk_frames[t][j] = counts[ix] - prev_counts[ix]
        for t, ix in rate_cells.items():
            rec_rate[t].add(dr[ix])
        prev_counts = counts
        if j % 300 == 0:
            print(f"  t {j * FRAME_S:5.1f} s  ball {ball_pos[-1]:+.3f} m  LC11 drive max {drive[spk_cells['LC11']].max():+.2f} mV  "
                  f"LC10a {drive[spk_cells['LC10a']].max():+.2f}  LPLC2 {drive[spk_cells['LPLC2']].max():+.2f}  ({time.time() - t0:.0f} s wall)", flush=True)
    window_s = n_sweep * FRAME_S
    rates_hz = {t: (prev_counts[ix] - counts0[ix]) / window_s for t, ix in spk_cells.items()}   # per cell over the window
    return sim, dict(drive=rec_drive, rate=rec_rate, drive_frames=drive_frames, spk_frames=spk_frames, rates_hz=rates_hz,
                     ball=np.array(ball_pos), window_s=window_s, spk_cells=spk_cells, rate_cells=rate_cells, radiance=radiance_diag)


N_BINS = 24                                                 # 0.5 cm bins of the ball's lateral offset


def sweep_bins(n_frames):
    """Bin index per frame by the ball's lateral offset on the shared timeline (the no-ball run uses the same bins)."""
    off = np.array([ball_offset(j * FRAME_S) for j in range(n_frames)])
    return np.clip(((off + HALF_SWEEP) / (2 * HALF_SWEEP) * N_BINS).astype(int), 0, N_BINS - 1)


def tuning(frames, bins):
    """(n_frames, n_cells) -> (N_BINS, n_cells) mean per bin (sweep-locked average over the repeated sweeps)."""
    out = np.zeros((N_BINS, frames.shape[1])); cnt = np.bincount(bins, minlength=N_BINS)
    np.add.at(out, bins, frames)
    return out / np.maximum(cnt, 1)[:, None], cnt


def smooth_peak(frames, w=10):
    """Peak over cells and time of the per-cell drive smoothed with a `w`-frame boxcar (100 ms at w = 10)."""
    if frames.shape[0] < w:
        return float(frames.max())
    cs = np.cumsum(frames, axis=0, dtype=np.float64); sm = (cs[w:] - cs[:-w]) / w
    return float(sm.max())


def summarize(res, other=None):
    """Per type numbers for one condition; `other` (the no-ball run) gives the per-frame difference statistics."""
    out = {}
    for t, r in res["drive"].items():
        m = r.mean(); hz = res["rates_hz"][t]; frames = res["drive_frames"][t]
        best = int(np.argmax(m)); bins = sweep_bins(frames.shape[0])
        tun, cnt = tuning(frames, bins); tun_rate, _ = tuning(res["spk_frames"][t], bins)
        tun_rate = tun_rate / FRAME_S                                                   # spikes per frame -> Hz per bin
        d = {"n_cells": int(len(m)), "kind": "spiking",
             "drive_mean_mv": float(m.mean()), "drive_best_cell_mean_mv": float(m[best]), "drive_p90_cell_mean_mv": float(np.percentile(m, 90)),
             "drive_peak_mv": float(frames.max()), "drive_peak_cell": int(np.argmax(frames.max(0))), "drive_peak_t_s": float(np.argmax(frames.max(1)) * FRAME_S),
             "drive_peak_100ms_mv": smooth_peak(frames),
             "tuning_range_best_cell_mv": float((tun.max(0) - tun.min(0)).max()),        # sweep-locked: best cell's (max bin - min bin)
             "tuning_peak_mv": float(tun.max()), "tuning_rate_peak_hz": float(tun_rate.max()), "tuning_bins_frames": [int(v) for v in cnt],
             "drive_peak_cells_over_7mv": int((frames.max(0) > PASS_DRIVE_MV).sum()),
             "drive_frames_over_7mv_frac": float((frames > PASS_DRIVE_MV).mean()),
             "drive_min_mv": float(frames.min()),
             "rate_hz_mean": float(hz.mean()), "rate_hz_max_cell": float(hz.max()), "cells_over_1hz": int((hz > PASS_RATE_HZ).sum()),
             "rate_hz_per_sweep_max": float(max((res["spk_frames"][t][i:i + int(SWEEP_S / FRAME_S)].sum(0) / SWEEP_S).max()
                                                for i in range(0, frames.shape[0], int(SWEEP_S / FRAME_S))))}
        if other is not None:
            fo = other["drive_frames"][t]; n = min(len(frames), len(fo)); diff = frames[:n] - fo[:n]
            md = m - other["drive"][t].mean()
            tun_o, _ = tuning(fo[:n], bins[:n]); tun_ro, _ = tuning(other["spk_frames"][t][:n], bins[:n]); tun_ro = tun_ro / FRAME_S
            d.update({"diff_best_cell_mean_mv": float(md.max()), "diff_peak_mv": float(diff.max()), "diff_peak_cells_over_7mv": int((diff.max(0) > PASS_DRIVE_MV).sum()),
                      "diff_peak_100ms_mv": smooth_peak(diff),
                      "diff_tuning_peak_mv": float((tun - tun_o).max()),                  # best (cell, ball-position bin) of the sweep-locked difference
                      "diff_tuning_rate_peak_hz": float((tun_rate - tun_ro).max()),
                      "diff_rate_hz_max_cell": float((hz - other["rates_hz"][t]).max()),
                      "diff_rate_hz_mean": float((hz - other["rates_hz"][t]).mean())})
        out[t] = d
    for t, r in res["rate"].items():
        m = r.mean(); ma = r.sum_abs / max(r.n, 1)
        d = {"n_cells": int(len(m)), "kind": "rate",
             "dev_mean": float(m.mean()), "dev_abs_mean": float(ma.mean()), "dev_abs_best_cell_mean": float(ma.max()),
             "dev_max": float(r.max.max()), "dev_min": float(r.min.min()), "dev_abs_p90_cell_mean": float(np.percentile(ma, 90))}
        if other is not None:
            mo = other["rate"][t].mean(); mao = other["rate"][t].sum_abs / max(other["rate"][t].n, 1)
            d.update({"diff_abs_mean": float((ma - mao).mean()), "diff_abs_best_cell_mean": float((ma - mao).max()),
                      "diff_signed_mean": float((m - mo).mean()), "diff_signed_best_cell": float(np.abs(m - mo).max())})
        out[t] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receptor-model", default="off", choices=["off", "sign"], help="LIFParams.receptor_model (off = the presynaptic NT_SIGN rule)")
    ap.add_argument("--receptor-net-rule", default="class", choices=["class", "abs", "nonmda"])
    ap.add_argument("--seed", type=int, default=0, help="room_demo.Sim seed (Brain RNG; the empty-table scene does not depend on it)")
    ap.add_argument("--seconds", type=float, default=12.0, help="scored sweep window (4 one-way sweeps at the default)")
    ap.add_argument("--settle", type=float, default=3.0, help="settling time before the window, ball hidden")
    ap.add_argument("--rectify-t2t3", action="store_true", help="T2 / T3 baseline 0 (ReLU units) via optic.DEFAULT_BASELINE_BY_TYPE")
    ap.add_argument("--cache-dir", default=os.environ.get("FLYVERSE_CACHE") or None, help="connectome cache directory (default cache/ or $FLYVERSE_CACHE)")
    ap.add_argument("--out", default="out/object_sweep.json")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA is not available (node race; resubmit)"
    patch_receptor(args.receptor_model, args.receptor_net_rule)
    patch_cache(args.cache_dir)
    if args.rectify_t2t3:
        optic.DEFAULT_BASELINE_BY_TYPE.update({"T2": 0.0, "T3": 0.0})
    mode = "off" if args.receptor_model == "off" else f"sign-{args.receptor_net_rule}"
    print(f"object sweep: mode {mode}{' rectified T2/T3' if args.rectify_t2t3 else ''}; seed {args.seed}; window {args.seconds} s after {args.settle} s settle; "
          f"cache {args.cache_dir or connectome.CACHE_DIR}; torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)
    sim_b, with_ball = run(True, args, args.seed)
    if sim_b.fb.receptor is not None:
        cov = sim_b.fb.receptor.coverage(sim_b.c.W)
        print(f"receptor model {args.receptor_model} ({args.receptor_net_rule}); fast sign changed on "
              f"{int((sim_b.fb.receptor.fast_sign != np.sign(sim_b.c.W.data)).sum()):,} of {sim_b.c.W.nnz:,} entries; coverage by tier:")
        print(cov.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    del sim_b; torch.cuda.empty_cache()
    sim_n, no_ball = run(False, args, args.seed)
    del sim_n; torch.cuda.empty_cache()
    S_b = summarize(with_ball, no_ball); S_n = summarize(no_ball)
    # verdict
    verdict = {}
    for t in ("LC11", "LC10a"):
        b_ = S_b[t]; n_ = S_n[t]
        drive_pass = b_["drive_peak_mv"] > PASS_DRIVE_MV and not n_["drive_peak_mv"] > PASS_DRIVE_MV
        rate_pass = b_["rate_hz_max_cell"] > PASS_RATE_HZ and not n_["rate_hz_max_cell"] > PASS_RATE_HZ
        verdict[t] = {"drive_pass": bool(drive_pass), "rate_pass": bool(rate_pass), "pass": bool(drive_pass or rate_pass)}
    verdict["pass"] = bool(verdict["LC11"]["pass"] or verdict["LC10a"]["pass"])
    # table
    print(f"\nmode {mode}{' rectified T2/T3' if args.rectify_t2t3 else ''}, seed {args.seed}: {args.seconds:.0f} s window, ball vs none")
    print(f"{'type':7s} {'cells':>5s} | {'drive mean mV':>14s} {'best cell mean':>15s} {'peak mV':>15s} {'peak 100ms':>15s} {'>7mV cells':>10s} | "
          f"{'rate Hz mean':>14s} {'max cell Hz':>15s} {'>1Hz':>9s} | {'tuning best mV':>15s} {'tuning Hz':>13s} | diff best mean/peak/tuning")
    for t in SPIKING:
        b_ = S_b[t]; n_ = S_n[t]
        print(f"{t:7s} {b_['n_cells']:5d} | {b_['drive_mean_mv']:+6.2f}/{n_['drive_mean_mv']:+6.2f} {b_['drive_best_cell_mean_mv']:+6.2f}/{n_['drive_best_cell_mean_mv']:+6.2f}   "
              f"{b_['drive_peak_mv']:+6.2f}/{n_['drive_peak_mv']:+6.2f} {b_['drive_peak_100ms_mv']:+6.2f}/{n_['drive_peak_100ms_mv']:+6.2f} "
              f"{b_['drive_peak_cells_over_7mv']:4d}/{n_['drive_peak_cells_over_7mv']:<4d} | "
              f"{b_['rate_hz_mean']:6.3f}/{n_['rate_hz_mean']:6.3f} {b_['rate_hz_max_cell']:6.2f}/{n_['rate_hz_max_cell']:6.2f} {b_['cells_over_1hz']:3d}/{n_['cells_over_1hz']:<3d} | "
              f"{b_['tuning_range_best_cell_mv']:6.2f}/{n_['tuning_range_best_cell_mv']:6.2f} {b_['tuning_rate_peak_hz']:5.2f}/{n_['tuning_rate_peak_hz']:5.2f} | "
              f"{b_['diff_best_cell_mean_mv']:+.2f}/{b_['diff_peak_mv']:+.2f}/{b_['diff_tuning_peak_mv']:+.2f} mV")
    print(f"{'type':7s} {'cells':>5s} | {'|dev| mean':>14s} {'best cell |dev|':>15s} {'dev max/min':>15s} | signed mean | diff |dev| mean / best")
    for t in RATE:
        b_ = S_b[t]; n_ = S_n[t]
        print(f"{t:7s} {b_['n_cells']:5d} | {b_['dev_abs_mean']:6.4f}/{n_['dev_abs_mean']:6.4f} {b_['dev_abs_best_cell_mean']:6.3f}/{n_['dev_abs_best_cell_mean']:6.3f}   "
              f"{b_['dev_max']:+5.2f}/{b_['dev_min']:+5.2f} vs {n_['dev_max']:+5.2f}/{n_['dev_min']:+5.2f} | {b_['dev_mean']:+7.4f}/{n_['dev_mean']:+7.4f} | "
              f"{b_['diff_abs_mean']:+.4f} / {b_['diff_abs_best_cell_mean']:+.4f}")
    print(f"verdict: LC11 {verdict['LC11']}  LC10a {verdict['LC10a']}  -> {'PASS' if verdict['pass'] else 'FAIL'} "
          f"(criterion: peak per-cell drive > {PASS_DRIVE_MV:.0f} mV or a cell > {PASS_RATE_HZ:.0f} Hz with the ball and not without)")
    # per-sweep time course of the population mean drive (ball run) for the two detectors and LPLC2
    n_per = int(SWEEP_S / FRAME_S)
    tc = {}
    for t in ("LC11", "LC10a", "LPLC2"):
        tr = np.array(with_ball["drive"][t].trace)[:, 1]; trn = np.array(no_ball["drive"][t].trace)[:, 1]
        mx = with_ball["drive_frames"][t].max(1); mxn = no_ball["drive_frames"][t].max(1)
        tc[t] = {"ball_mean_abs_drive_per_sweep": [float(tr[i:i + n_per].mean()) for i in range(0, len(tr), n_per)],
                 "none_mean_abs_drive_per_sweep": [float(trn[i:i + n_per].mean()) for i in range(0, len(trn), n_per)],
                 "ball_max_cell_drive_per_sweep": [float(mx[i:i + n_per].max()) for i in range(0, len(mx), n_per)],
                 "none_max_cell_drive_per_sweep": [float(mxn[i:i + n_per].max()) for i in range(0, len(mxn), n_per)],
                 "ball_max_cell_drive_per_500ms": [float(mx[i:i + 50].max()) for i in range(0, len(mx), 50)],
                 "none_max_cell_drive_per_500ms": [float(mxn[i:i + 50].max()) for i in range(0, len(mxn), 50)]}
    print("population mean |drive| per one-way sweep (ball / none): " + "; ".join(
        f"{t} " + " ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(v["ball_mean_abs_drive_per_sweep"], v["none_mean_abs_drive_per_sweep"])) for t, v in tc.items()))
    out = {"config": {"mode": mode, "receptor_model": args.receptor_model, "receptor_net_rule": args.receptor_net_rule, "rectify_t2t3": bool(args.rectify_t2t3),
                      "seed": args.seed, "seconds": args.seconds, "settle": args.settle, "cache_dir": str(args.cache_dir or connectome.CACHE_DIR),
                      "pos": list(POS), "heading_rad": float(HEADING), "ball_radius_m": BALL_R, "ahead_m": AHEAD, "half_sweep_m": HALF_SWEEP, "sweep_s": SWEEP_S,
                      "pass_drive_mv": PASS_DRIVE_MV, "pass_rate_hz": PASS_RATE_HZ, "device": torch.cuda.get_device_name(0), "torch": torch.__version__},
           "ball": S_b, "none": S_n, "verdict": verdict, "time_course": tc, "radiance_check": with_ball["radiance"]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
