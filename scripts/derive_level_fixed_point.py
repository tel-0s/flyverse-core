"""The THREE-CHANNEL-MATCHED level control's three sense parameters (arm L3 of family level4, docs/audits/level_fixed_point.md),
derived TOGETHER as one fixed point of the afferent -> VNC -> leg-MN -> afferent loop, on CPU, with the calibration-pair method
of rounds 4 / 4b (out/vncd4/mn_ref_derivation.json, out/vncd5/derive_channel_match.py) iterated on all three parameters.

THE LAWS (senses.Proprioception.rates, unchanged; the round-2 MN-rate form under the existing 'unsided' token, i.e. every leg cell
reads the side-mean leg-MN rate m(t) = (legMN_L + legMN_R) / 2):
    drive per cell     d(t) = clip(m(t) / mn_ref_hz, 0, 1) * ground(t)          -- the SAME number on every chordotonal and hair-plate cell
    chordotonal        rate = 10 + 140 * d(t)                                    (10 Hz airborne)
    hair plate         rate = 5 + (hp_max - 5) * d(t)                            (5 Hz airborne)
    leg campaniform    rate = load_hz * ground(t)                                (0 airborne)
so with D = the window mean of d(t) over frames and flies:  chordotonal = 10 + 140 D,  hair plate = 5 + (hp_max - 5) D,
campaniform = load_hz x ground fraction. Two of the three parameters are therefore ALGEBRA once the targets are named --
hp_max = 5 + (hair* - 5) x 140 / (chordotonal* - 10) exactly (both channels read one drive), load = campaniform* / ground --
and the one loop parameter is mn_ref: D depends on the leg-MN distribution m(t), which the VNC produces FROM the afferent input
the three parameters set (the loop). Under the SIDED law (round 4's L) the leg-MN side bias puts +13.5 Hz of chordotonal and
+9.1 Hz of hair plate between the sides by construction, so no value of the three parameters lands a sided arm inside a
per-side tolerance; 'unsided' (round 4b's arm U, a labelled control token) is what makes "both sides" satisfiable.

WHY THE FIXED POINT NEEDS THE GPU RECORDINGS AND NOT ONLY SHORT CPU RUNS (the transfer defect of rounds 4 / 4b, measured here):
the leg-MN rate is not constant over the 60-s room protocol -- every arm shows a dip at 24-34 s and the flies leave the table
after ~45 s -- so the 5-60 s window mean sits BELOW a 2-12 s calibration window by a WINDOW FACTOR W = (chord_full - 10) /
(chord_early - 10) of 0.947-0.971 on the steady GPU arms (K / U / L). A 12-s CPU run realises the early-window level (cal_L
86.97 against GPU L's 88.3 early / 86.0 full; cal_K2 80.99 against K's 81.1 early / 77.4 full), which is where round 4b's
-3.6 Hz K transfer error came from. The fixed point is therefore solved on the FULL-WINDOW per-frame leg-MN distribution of the
GPU anchor arm (out/vncd5 arm U: 'all+unsided' at mn_ref 8.84, hp_max 100, load 50 -- the same law and token as L3, whose
D(8.84) reproduces its recorded 84.84 Hz exactly), scaled by the LOOP GAIN g the CPU measures: g_k = m_L3,k / m_U, the ratio of
the L3 candidate's leg-MN rate to the anchor's on the SAME CPU protocol (a calibration PAIR: same brain seed, env seeds, flies,
window). The CPU pair carries the loop response to the three parameters together; the GPU anchor carries the protocol's window.

  step 0 (targets)      the realised window-mean commanded rates of the cycle arm C, per side, from its recordings
  step 1 (algebra)      hp_max and load from the targets; mn_ref_0 from D_full(mn_ref; g = 1) = D* on the anchor's distribution
  step 2 (CPU pair k)   one CPU room run of L3 at (mn_ref_k, hp_max, load) beside the anchor's CPU run -> g_k; the secant on the
                        measured (mn_ref_k, g_k) points solves D_full(mn_ref; g(mn_ref)) = D* for mn_ref_{k+1}
  step 3 (chosen)       the converged values rounded to two decimals, the predicted realised means and their uncertainty
  step 4 (verify)       a LONGER CPU pair at the chosen values (L3 beside the anchor) checks the loop gain over more of the
                        window and predicts the pair's AN04B003 / DNa02; the GPU realisation is checked from the batch
                        recordings against the predeclared tolerance (verify_runs.py), and nothing is re-fitted after it.

Nothing about behaviour enters; the three values exist only to match three channel LEVELS and are LABELLED CONTROL parameters
passed on the job line (--mn-ref-hz / --hair-plate-max-hz / --campaniform-load-hz; probe_vnc_drive.ARM_SENSE_KW['level4']['L3']).
The sense's defaults (30 / 100 / 50) and flyverse/senses.py are untouched.

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python scripts/derive_level_fixed_point.py            # reads whatever cal runs exist, prints the next command
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python scripts/derive_level_fixed_point.py --chosen 8.62,81.09,25.10 --verify cal_L3_v_r0,cal_U_v_r0
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CH_TONIC, CH_MAX = 10.0, 150.0
HP_TONIC = 5.0
ANCHOR = {"mn_ref_hz": 8.84, "hair_plate_max_hz": 100.0, "campaniform_load_hz": 50.0}     # the GPU anchor arm U of out/vncd5 (family level2)
CHANNELS = ("chordotonal", "hair_plate", "campaniform", "haltere")


def _runs(d, arm):
    return [p for p in sorted(glob.glob(f"{d}/room_{arm}_r*.json")) if re.fullmatch(rf"room_{arm}_r\d+\.json", Path(p).name)]


def gpu_levels(d, arm, early=(200, 1200)):
    """Per-run realised window-mean commanded channel rates (post-skip), per side, the ground / walking fractions, the leg-MN
    rates, and the same over the EARLY window (frames 200-1200 = 2-12 s, the CPU calibration window)."""
    rows = []
    for p in _runs(d, arm):
        j = json.loads(Path(p).read_text(encoding="utf-8")); z = np.load(p[:-5] + "_body.npz")
        sk = int(round(j.get("skip_s", 5.0) * 100)); w = slice(sk, None); e = slice(*early)
        air = z["airborne"]; m = 0.5 * (z["cmd__legMN_L"] + z["cmd__legMN_R"])
        r = {"file": Path(p).name, "seed": int(j["seed"]), "device": j.get("device"), "device_name": j.get("device_name"),
             "ground_frac": float((~air[w]).mean()), "ground_frac_early": float((~air[e]).mean()),
             "walking_frac": float((z["speed_cmd"][w] > 1e-4).mean()),
             "legMN_L": float(z["cmd__legMN_L"][w].mean()), "legMN_R": float(z["cmd__legMN_R"][w].mean()), "legMN_mean": float(m[w].mean()), "legMN_mean_early": float(m[e].mean())}
        for ch in CHANNELS:
            k = f"commanded__{ch}"
            if k not in z:
                continue
            r[ch] = float(z[k][w].mean()); r[f"{ch}_early"] = float(z[k][e].mean())
            for s in ("L", "R"):
                g = f"commandedg__{ch}:{s}"
                if g in z:
                    r[f"{ch}_{s}"] = float(z[g][w].mean())
        rows.append(r)
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    summ = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0, "runs": [round(r[k], 4) for r in rows]} for k in keys}
    summ["n_runs"] = len(rows); summ["files"] = [r["file"] for r in rows]; summ["device_name"] = sorted({str(r["device_name"]) for r in rows})
    return summ


def anchor_distribution(d, arm):
    """The GPU anchor's per-frame side-mean leg-MN rate and ground flag over the FULL post-skip window (every run, every fly, every
    frame) and over the early window: the distribution the fixed point is evaluated on."""
    full_m, full_g, early_m, early_g = [], [], [], []
    for p in _runs(d, arm):
        j = json.loads(Path(p).read_text(encoding="utf-8")); z = np.load(p[:-5] + "_body.npz")
        sk = int(round(j.get("skip_s", 5.0) * 100))
        m = 0.5 * (z["cmd__legMN_L"] + z["cmd__legMN_R"]); g = (~z["airborne"]).astype(float)
        full_m.append(m[sk:].ravel()); full_g.append(g[sk:].ravel()); early_m.append(m[200:1200].ravel()); early_g.append(g[200:1200].ravel())
    return (np.concatenate(full_m), np.concatenate(full_g)), (np.concatenate(early_m), np.concatenate(early_g))


def D_of(dist, mn_ref, g=1.0):
    m, gr = dist
    return float((np.clip(g * m / mn_ref, 0.0, 1.0) * gr).mean())


def solve_mn(dist, target_D, g_of_mn, lo=4.0, hi=20.0):
    """Bisection on mn_ref for D_full(mn_ref; g(mn_ref)) = target_D (D is decreasing in mn_ref for a fixed g; with the loop's
    g(mn_ref) rising as mn_ref falls the product is still monotone at the loop gains measured here -- checked by the bracket)."""
    f = lambda mn: D_of(dist, mn, g_of_mn(mn)) - target_D
    flo, fhi = f(lo), f(hi)
    if not (flo > 0 > fhi):
        raise SystemExit(f"no bracket: D({lo}) - D* = {flo:+.4f}, D({hi}) - D* = {fhi:+.4f}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def g_model(pts):
    """The loop gain as a function of mn_ref from the measured (mn_ref, g) points of the CPU pairs: one point -> held; two ->
    the secant; three or more -> the least-squares line through ALL of them (each g carries the CPU pair's fly-level noise,
    ~+-0.02 at 8 flies, so the last-two secant would chase that noise while the loop's slope is one number)."""
    if len(pts) == 1:
        g0 = pts[0][1]
        return (lambda mn, g0=g0: g0), f"g held at the measured {g0:.4f}"
    x = np.array([m for m, _ in pts]); y = np.array([g for _, g in pts])
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / max(((x - x.mean()) ** 2).sum(), 1e-12)); x0, y0 = float(x.mean()), float(y.mean())
    kind = "secant through" if len(pts) == 2 else "least-squares line through all"
    return (lambda mn, x0=x0, y0=y0, slope=slope: y0 + slope * (mn - x0)), f"{kind} {len(pts)} measured points {[(m, round(g, 4)) for m, g in pts]}: dg/dmn_ref = {slope:+.4f} per Hz"


def cal_point(path):
    """A CPU calibration room run: its sense parameters, its realised window means (the run's own window: skip..seconds) and
    the leg-MN rates -- pooled and per side -- plus the per-frame side-mean MN over the window (for the time-course check)."""
    j = json.loads(Path(path).read_text(encoding="utf-8")); r = j["run"]; sen = j["sense"]; z = np.load(str(path)[:-5] + "_body.npz")
    sk = int(round(j.get("skip_s", 2.0) * 100)); m = 0.5 * (z["cmd__legMN_L"] + z["cmd__legMN_R"])
    out = {"file": str(path).replace("\\", "/"), "arm": j["arm"], "family": j.get("family"), "spec": j.get("proprioception"), "device": j.get("device"), "batch": j["batch"], "seconds": j["seconds"], "skip_s": j["skip_s"],
           "wall_s": j.get("wall_s"), "started_utc": j.get("started_utc"), "brain_seed": j["seed"], "env_seeds": j.get("env_seeds"),
           "mn_ref_hz": float(sen["mn_ref_hz"]), "hair_plate_max_hz": float(sen["hair_plate_max_hz"]), "campaniform_load_hz": float(sen["campaniform_load_hz"]),
           "tokens": [k for k, v in sen["tokens"].items() if v],
           "chordotonal_hz": float(r["commanded_chordotonal_hz"]), "chordotonal_L_hz": float(r["commanded_chordotonal:L_hz"]), "chordotonal_R_hz": float(r["commanded_chordotonal:R_hz"]),
           "chordotonal_fly_sd": float(r["commanded_chordotonal_hz_fly_sd"]),
           "hair_plate_hz": float(r["commanded_hair_plate_hz"]), "hair_plate_L_hz": float(r["commanded_hair_plate:L_hz"]), "hair_plate_R_hz": float(r["commanded_hair_plate:R_hz"]),
           "campaniform_hz": float(r["commanded_campaniform_hz"]), "haltere_hz": float(r["commanded_haltere_hz"]),
           "ground_fraction": 1.0 - float(r["airborne_frac"]), "leg_mn_L_R_hz": [float(r["leg_L_hz"]), float(r["leg_R_hz"])], "leg_mn_mean_hz": float(m[sk:].mean()),
           "leg_mn_mean_by_second": [round(float(m[a:a + 100].mean()), 4) for a in range(0, m.shape[0], 100)],
           "AN04B003_L_R_hz": [float(r["AN04B003_L_hz"]), float(r["AN04B003_R_hz"])], "DNa02_L_R_hz": [float(r["DNa02_L_hz"]), float(r["DNa02_R_hz"])],
           "IN13B001_L_R_hz": [float(r.get("IN13B001_L_hz", np.nan)), float(r.get("IN13B001_R_hz", np.nan))], "straightness": float(r["straightness"])}
    out["D_realised"] = (out["chordotonal_hz"] - CH_TONIC) / (CH_MAX - CH_TONIC)
    out["hair_over_chord_identity"] = {"(hair - 5) / (hp_max - 5)": (out["hair_plate_hz"] - HP_TONIC) / (out["hair_plate_max_hz"] - HP_TONIC), "(chord - 10) / 140": out["D_realised"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets-dir", default="out/vncd6"); ap.add_argument("--targets-arm", default="C")
    ap.add_argument("--context-dir", default="out/vncd5", help="the previous batch's cycle arm, reported beside the targets (not used)")
    ap.add_argument("--anchor-dir", default="out/vncd5"); ap.add_argument("--anchor-arm", default="U")
    ap.add_argument("--steady", default="out/vncd5:L,out/vncd5:U,out/vncd5:K,out/vncd6:L", help="the steady GPU arms whose ground fraction and window factor are pooled")
    ap.add_argument("--cal", default="out/vncd7/cal"); ap.add_argument("--anchor-cal", default="cal_U_r0", help="the anchor's CPU run (stem under --cal)")
    ap.add_argument("--iterates", default=None, help="comma list of L3 CPU iterate stems in order (default: cal_L3_i*_r0 sorted)")
    ap.add_argument("--chosen", default=None, help="mn_ref,hp_max,load: freeze the chosen values (else the next iterate is proposed)")
    ap.add_argument("--verify", default=None, help="comma list of verification stems (L3 first, then the anchor) at the chosen values")
    ap.add_argument("--out", default="out/vncd7/fixed_point_derivation.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    # ---- step 0: the targets
    tg = gpu_levels(a.targets_dir, a.targets_arm); ctx = gpu_levels(a.context_dir, a.targets_arm) if Path(a.context_dir).exists() else None
    ch_t, hp_t, cp_t = tg["chordotonal"]["mean"], tg["hair_plate"]["mean"], tg["campaniform"]["mean"]
    print(f"targets ({a.targets_dir} arm {a.targets_arm}, {tg['n_runs']} runs, {tg['device_name']}): chordotonal {ch_t:.3f} +- {tg['chordotonal']['sd']:.3f} (L {tg['chordotonal_L']['mean']:.3f} / R {tg['chordotonal_R']['mean']:.3f}), "
          f"hair plate {hp_t:.3f} +- {tg['hair_plate']['sd']:.3f} (L {tg['hair_plate_L']['mean']:.3f} / R {tg['hair_plate_R']['mean']:.3f}), campaniform {cp_t:.3f} +- {tg['campaniform']['sd']:.3f}; "
          f"ground {tg['ground_frac']['mean']:.4f}, walking {tg['walking_frac']['mean']:.4f}, leg MN {tg['legMN_L']['mean']:.3f} / {tg['legMN_R']['mean']:.3f}")
    if ctx:
        print(f"context ({a.context_dir} arm {a.targets_arm}, {ctx['n_runs']} runs): chordotonal {ctx['chordotonal']['mean']:.3f}, hair {ctx['hair_plate']['mean']:.3f}, campaniform {ctx['campaniform']['mean']:.3f} (not used: the targets are the most recent, six-run batch)")
    # ---- the steady GPU arms: ground fraction and window factor
    steady = {}
    for item in a.steady.split(","):
        d, arm = item.split(":"); s = gpu_levels(d, arm)
        steady[item] = {"ground_frac": s["ground_frac"]["mean"], "chordotonal_full": s["chordotonal"]["mean"], "chordotonal_early": s["chordotonal_early"]["mean"],
                        "window_factor": (s["chordotonal"]["mean"] - CH_TONIC) / (s["chordotonal_early"]["mean"] - CH_TONIC),
                        "legMN_mean_full": s["legMN_mean"]["mean"], "legMN_mean_early": s["legMN_mean_early"]["mean"], "n_runs": s["n_runs"]}
    ground_steady = float(np.mean([v["ground_frac"] for v in steady.values()])); W = np.array([v["window_factor"] for v in steady.values()])
    print("steady GPU arms (ground fraction; chordotonal full 5-60 s / early 2-12 s -> window factor W):")
    for k, v in steady.items():
        print(f"   {k}: ground {v['ground_frac']:.4f}; chord {v['chordotonal_full']:.3f} / {v['chordotonal_early']:.3f} -> W {v['window_factor']:.4f}; leg MN {v['legMN_mean_full']:.3f} / {v['legMN_mean_early']:.3f}")
    print(f"   pooled ground fraction {ground_steady:.4f} (SD {np.std([v['ground_frac'] for v in steady.values()], ddof=1):.4f}); W {W.mean():.4f} +- {W.std(ddof=1):.4f}")
    # ---- step 1: the algebra and the open-loop mn_ref on the anchor's full-window distribution
    D_t = (ch_t - CH_TONIC) / (CH_MAX - CH_TONIC)
    hp_max = HP_TONIC + (hp_t - HP_TONIC) / D_t
    load = cp_t / ground_steady
    an = gpu_levels(a.anchor_dir, a.anchor_arm); (full, early) = anchor_distribution(a.anchor_dir, a.anchor_arm)
    D_an = D_of(full, ANCHOR["mn_ref_hz"]); chord_an_pred = CH_TONIC + 140.0 * D_an
    mn_0 = solve_mn(full, D_t, lambda mn: 1.0)
    print(f"\nSTEP 1 (algebra, exact under 'unsided'): D* = (chord* - 10) / 140 = {D_t:.6f}; hair_plate_max_hz = 5 + (hair* - 5) / D* = {hp_max:.4f}; campaniform_load_hz = camp* / ground_steady = {load:.4f}")
    print(f"anchor ({a.anchor_dir} arm {a.anchor_arm}, GPU, {an['n_runs']} runs at mn_ref {ANCHOR['mn_ref_hz']} / hp_max {ANCHOR['hair_plate_max_hz']} / load {ANCHOR['campaniform_load_hz']}): recorded chordotonal {an['chordotonal']['mean']:.3f} "
          f"(L {an['chordotonal_L']['mean']:.3f} / R {an['chordotonal_R']['mean']:.3f}), hair {an['hair_plate']['mean']:.3f}, camp {an['campaniform']['mean']:.3f}; leg MN full {an['legMN_mean']['mean']:.3f}, early {an['legMN_mean_early']['mean']:.3f}; "
          f"D_full(8.84) on its recorded per-frame distribution = {D_an:.6f} -> {chord_an_pred:.3f} Hz (the law reproduces the recording: the check); clip binds on {(full[0] / ANCHOR['mn_ref_hz'] > 1).mean() * 100:.2f} % of frames")
    print(f"open loop (g = 1): mn_ref_0 = {mn_0:.4f} Hz (D_full = D* on the anchor's distribution); early-window D at that mn_ref {D_of(early, mn_0):.6f} -> {CH_TONIC + 140 * D_of(early, mn_0):.3f} Hz (what a 2-12 s CPU run should read if the loop gain were 1)")
    rec = {"schema": "flyverse.derivation/2", "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "quantity": "senses.Proprioception mn_ref_hz, hair_plate_max_hz and campaniform_load_hz of the THREE-CHANNEL-MATCHED level control (arm L3, spec 'all+unsided', family level4), derived together as one fixed point of the afferent -> leg-MN loop",
           "law": "unsided round-2 MN-rate law: d = clip((legMN_L + legMN_R) / 2 / mn_ref_hz, 0, 1) * ground on every leg cell; chordotonal 10 + 140 d; hair plate 5 + (hp_max - 5) d; campaniform load_hz * ground; senses.Proprioception.rates unchanged, token 'unsided' existing (round 4b arm U)",
           "why_unsided": "the per-side match: under the sided law the leg-MN side bias (5.25 / 4.40 Hz at mn_ref 8.84) puts +13.5 Hz of chordotonal and +9.1 Hz of hair plate between the sides by construction (out/vncd6 L: 92.59 / 79.12 against C's 87.23 / 87.59), so no value of the three parameters lands a sided arm inside a 3-Hz per-side tolerance; with every leg cell reading the side-mean rate both channels are side-symmetric and one drive fraction sets both",
           "targets": {"source": f"{a.targets_dir} arm {a.targets_arm}: the realised window-mean commanded rates over its {tg['n_runs']} runs (16 flies x 55 s, post-skip frames), recomputed from room_C_r*_body.npz",
                       "why_this_batch": "the most recent cycle-arm batch, six runs (the previous had five), run on the same body / senses code this batch's C runs on; its C is +1.3 Hz of chordotonal above out/vncd5's C (87.40 vs 86.11), inside the between-batch scatter of the same arm, and the batch's own C is what L3 is compared with -- the match is verified from the new recordings (precondition P1) whichever batch the targets came from",
                       "chordotonal_hz": ch_t, "chordotonal_L_hz": tg["chordotonal_L"]["mean"], "chordotonal_R_hz": tg["chordotonal_R"]["mean"],
                       "hair_plate_hz": hp_t, "hair_plate_L_hz": tg["hair_plate_L"]["mean"], "hair_plate_R_hz": tg["hair_plate_R"]["mean"], "campaniform_hz": cp_t,
                       "target_arm_levels": tg, "context_previous_batch": ctx},
           "steady_gpu_arms": {"note": "the ground fraction (campaniform = load x ground) and the WINDOW FACTOR W = (chord_full - 10) / (chord_early - 10): the 5-60 s window mean sits below the 2-12 s calibration window because the leg-MN rate dips at 24-34 s in every arm and the flies leave the table after ~45 s; a 12-s CPU run realises the early-window level, which is where round 4b's K transfer error (-3.6 Hz) came from",
                               "arms": steady, "ground_fraction_pooled": ground_steady, "window_factor_mean": float(W.mean()), "window_factor_sd": float(W.std(ddof=1))},
           "step_1_algebra": {"D_star": D_t, "hair_plate_max_hz": hp_max, "campaniform_load_hz": load,
                              "hair_plate_max_note": "exact: under 'unsided' the chordotonal and hair-plate cells read the same drive, so (hair - 5) / (hp_max - 5) = (chord - 10) / 140 on every frame; hp_max is set by the target RATIO and does not depend on the loop",
                              "campaniform_note": "load x ground: the loop enters only through the ground (airborne) fraction, a behavioural quantity taken from the steady GPU arms (the CPU calibration runs never leave the table in 12 s: ground 1.000)",
                              "anchor": {"dir": a.anchor_dir, "arm": a.anchor_arm, "params": ANCHOR, "levels": an, "D_full_at_anchor": D_an, "chordotonal_reproduced_hz": chord_an_pred,
                                         "clip_fraction_at_anchor": float((full[0] / ANCHOR["mn_ref_hz"] > 1).mean()), "n_frames_full": int(full[0].size), "n_frames_early": int(early[0].size)},
                              "mn_ref_0_open_loop": mn_0, "early_window_chordotonal_at_mn_ref_0_if_g_1": CH_TONIC + 140 * D_of(early, mn_0)},
           "classification": "LABELLED CONTROL parameters: chosen to match three channel LEVELS, never defaults; the sense's defaults (mn_ref_hz 30, hair_plate_max_hz 100, campaniform_load_hz 50) are untouched (probe_vnc_drive --mn-ref-hz / --hair-plate-max-hz / --campaniform-load-hz, the family table ARM_SENSE_KW['level4']['L3'])",
           "generator": "scripts/derive_level_fixed_point.py", "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    # ---- step 2: the CPU calibration pairs
    anchor_cal = Path(a.cal, a.anchor_cal + ".json")
    iters = [Path(a.cal, s.strip() + ".json") for s in a.iterates.split(",")] if a.iterates else sorted(Path(a.cal).glob("cal_L3_i*_r0.json"), key=lambda p: int(re.search(r"_i(\d+)_", p.name).group(1)))
    rec["step_2_cpu_calibration_pairs"] = {"protocol": "CPU (CUDA_VISIBLE_DEVICES=-1), probe_vnc_drive room --device cpu, brain seed 0, 8 flies (env seeds 0..7), 12 s, window 2-12 s; the anchor arm U (family level2, 8.84 / 100 / 50) and each L3 iterate (family level4) on the SAME protocol -- the pair's leg-MN ratio g = m_L3 / m_U is the loop gain the GPU anchor distribution is scaled by",
                                           "anchor_cal": None, "iterates": []}
    if not anchor_cal.exists():
        print(f"\n(no anchor CPU run yet: {anchor_cal}); next: python scripts/probe_vnc_drive.py room --family level2 --arm U --seed 0 --batch 8 --seconds 12 --skip 2 --device cpu --block cal --out {a.cal}/{a.anchor_cal}")
    else:
        U = cal_point(anchor_cal); rec["step_2_cpu_calibration_pairs"]["anchor_cal"] = U
        print(f"\nSTEP 2: anchor on CPU ({U['file']}, {U['batch']} flies x {U['seconds']} s, wall {U['wall_s']} s): chordotonal {U['chordotonal_hz']:.3f} (L {U['chordotonal_L_hz']:.3f} / R {U['chordotonal_R_hz']:.3f}; fly SD {U['chordotonal_fly_sd']:.2f}), hair {U['hair_plate_hz']:.3f}, camp {U['campaniform_hz']:.3f}; "
              f"leg MN {U['leg_mn_L_R_hz'][0]:.3f} / {U['leg_mn_L_R_hz'][1]:.3f} (mean {U['leg_mn_mean_hz']:.3f}); GPU anchor early window: chordotonal {an['chordotonal_early']['mean']:.3f}, leg MN {an['legMN_mean_early']['mean']:.3f} "
              f"(CPU - GPU early: {U['chordotonal_hz'] - an['chordotonal_early']['mean']:+.3f} Hz, {U['leg_mn_mean_hz'] - an['legMN_mean_early']['mean']:+.3f} Hz of leg MN)")
        rec["step_2_cpu_calibration_pairs"]["cpu_vs_gpu_anchor_early_window"] = {"chordotonal_cpu_minus_gpu_hz": U["chordotonal_hz"] - an["chordotonal_early"]["mean"], "legMN_cpu_minus_gpu_hz": U["leg_mn_mean_hz"] - an["legMN_mean_early"]["mean"]}
        pts = []
        for k, p in enumerate(iters, 1):
            if not p.exists():
                break
            L3 = cal_point(p)
            g = L3["leg_mn_mean_hz"] / U["leg_mn_mean_hz"]
            mn_k = L3["mn_ref_hz"]
            D_full_k = D_of(full, mn_k, g); pred_full = {"chordotonal_hz": CH_TONIC + 140 * D_full_k, "hair_plate_hz": HP_TONIC + (L3["hair_plate_max_hz"] - HP_TONIC) * D_full_k, "campaniform_hz": L3["campaniform_load_hz"] * ground_steady}
            D_early_k = D_of(early, mn_k, g); pred_early_chord = CH_TONIC + 140 * D_early_k
            # the identity the algebra rests on, checked on the realised CPU means
            ident = L3["hair_over_chord_identity"]
            pts.append((mn_k, g))
            row = {"k": k, "params": {"mn_ref_hz": mn_k, "hair_plate_max_hz": L3["hair_plate_max_hz"], "campaniform_load_hz": L3["campaniform_load_hz"]}, "cal": L3,
                   "loop_gain_g": g, "legMN_L3_over_anchor": [L3["leg_mn_mean_hz"], U["leg_mn_mean_hz"]],
                   "realised_cpu_means": {"chordotonal_hz": L3["chordotonal_hz"], "chordotonal_L_hz": L3["chordotonal_L_hz"], "chordotonal_R_hz": L3["chordotonal_R_hz"], "hair_plate_hz": L3["hair_plate_hz"], "campaniform_hz": L3["campaniform_hz"], "haltere_hz": L3["haltere_hz"]},
                   "predicted_gpu_full_window": pred_full, "predicted_cpu_early_window_chordotonal_hz": pred_early_chord, "early_window_prediction_error_hz": L3["chordotonal_hz"] - pred_early_chord,
                   "hair_over_chord_identity_on_cpu": ident}
            print(f"  iterate {k} ({p.name}): mn_ref {mn_k} hp_max {L3['hair_plate_max_hz']} load {L3['campaniform_load_hz']} -> CPU chordotonal {L3['chordotonal_hz']:.3f} (L {L3['chordotonal_L_hz']:.3f} / R {L3['chordotonal_R_hz']:.3f}; fly SD {L3['chordotonal_fly_sd']:.2f}), "
                  f"hair {L3['hair_plate_hz']:.3f}, camp {L3['campaniform_hz']:.3f}, haltere {L3['haltere_hz']:.3f}; leg MN {L3['leg_mn_L_R_hz'][0]:.3f} / {L3['leg_mn_L_R_hz'][1]:.3f} (mean {L3['leg_mn_mean_hz']:.3f}) -> g = {g:.4f}; "
                  f"AN04B003 {L3['AN04B003_L_R_hz'][0]:.2f} / {L3['AN04B003_L_R_hz'][1]:.2f}, DNa02 {L3['DNa02_L_R_hz'][0]:.3f} / {L3['DNa02_L_R_hz'][1]:.3f}")
            print(f"     identity (hair - 5)/(hp_max - 5) = {ident['(hair - 5) / (hp_max - 5)']:.6f} vs (chord - 10)/140 = {ident['(chord - 10) / 140']:.6f}; early-window prediction on the anchor distribution x g: {pred_early_chord:.3f} (realised {L3['chordotonal_hz']:.3f}, error {L3['chordotonal_hz'] - pred_early_chord:+.3f}); "
                  f"PREDICTED GPU full window: chordotonal {pred_full['chordotonal_hz']:.3f}, hair {pred_full['hair_plate_hz']:.3f}, camp {pred_full['campaniform_hz']:.3f} (targets {ch_t:.3f} / {hp_t:.3f} / {cp_t:.3f})")
            # the next iterate: g(mn_ref) from the measured points (constant after one, secant after two or more)
            g_of, model = g_model(pts)
            mn_next = solve_mn(full, D_t, g_of)
            row["next"] = {"mn_ref_hz": mn_next, "g_model": model, "g_at_next": g_of(mn_next), "predicted_full_window_chordotonal_at_next_if_model_holds": CH_TONIC + 140 * D_of(full, mn_next, g_of(mn_next))}
            print(f"     NEXT: mn_ref = {mn_next:.4f} ({model}; g at next {g_of(mn_next):.4f}); command: python scripts/probe_vnc_drive.py room --family level4 --arm L3 --mn-ref-hz {mn_next:.2f} --hair-plate-max-hz {hp_max:.2f} --campaniform-load-hz {load:.2f} --seed 0 --batch 8 --seconds 12 --skip 2 --device cpu --block cal --out {a.cal}/cal_L3_i{k + 1}_r0")
            rec["step_2_cpu_calibration_pairs"]["iterates"].append(row)
        # the loop slope for the record: leg MN per Hz of chordotonal across the anchor and the iterates (the same CPU protocol)
        if pts:
            xs = [U["chordotonal_hz"]] + [it["cal"]["chordotonal_hz"] for it in rec["step_2_cpu_calibration_pairs"]["iterates"]]
            ys = [U["leg_mn_mean_hz"]] + [it["cal"]["leg_mn_mean_hz"] for it in rec["step_2_cpu_calibration_pairs"]["iterates"]]
            hs = [U["hair_plate_hz"]] + [it["cal"]["hair_plate_hz"] for it in rec["step_2_cpu_calibration_pairs"]["iterates"]]
            rec["step_2_cpu_calibration_pairs"]["loop_slope_for_the_record"] = {"points_chordotonal_hz": xs, "points_hair_plate_hz": hs, "points_legMN_mean_hz": ys,
                                                                                "note": "the anchor and the iterates differ in all three parameters at once (the hair plate falls 10 Hz and the campaniform 25 Hz with them), so this is the joint response, not a chordotonal-only slope"}
    # ---- step 3: the chosen values and their predicted realisation
    if a.chosen:
        mn_c, hp_c, ld_c = [float(x) for x in a.chosen.split(",")]
        its = rec["step_2_cpu_calibration_pairs"]["iterates"]
        pts = [(it["params"]["mn_ref_hz"], it["loop_gain_g"]) for it in its]
        if pts:
            g_of, g_desc = g_model(pts); g_c = g_of(mn_c)
        else:
            g_c = 1.0; g_desc = "no CPU pair: g = 1 (open loop)"
        D_c = D_of(full, mn_c, g_c)
        pred = {"chordotonal_hz": CH_TONIC + 140 * D_c, "hair_plate_hz": HP_TONIC + (hp_c - HP_TONIC) * D_c, "campaniform_hz": ld_c * ground_steady}
        # uncertainty: the window factor's spread across the steady arms (W enters as (chord - 10) x W_arm / W_anchor), the CPU pair's fly-level noise on g, the ground fraction's spread on the campaniform
        W_rel = float(W.std(ddof=1) / W.mean())
        g_se = 0.0
        if its:
            L3 = its[-1]["cal"]; U = rec["step_2_cpu_calibration_pairs"]["anchor_cal"]
            se_l3 = L3["chordotonal_fly_sd"] / np.sqrt(L3["batch"]); se_u = U["chordotonal_fly_sd"] / np.sqrt(U["batch"])
            g_se = float(np.sqrt((se_l3 / max(L3["chordotonal_hz"] - CH_TONIC, 1e-9)) ** 2 + (se_u / max(U["chordotonal_hz"] - CH_TONIC, 1e-9)) ** 2))     # relative SE of the pair's drive ratio
        unc_ch = (pred["chordotonal_hz"] - CH_TONIC) * float(np.sqrt(W_rel ** 2 + g_se ** 2)); unc_hp = (pred["hair_plate_hz"] - HP_TONIC) * float(np.sqrt(W_rel ** 2 + g_se ** 2))
        unc_cp = ld_c * float(np.std([v["ground_frac"] for v in steady.values()], ddof=1))
        rec["step_3_fixed_point"] = {"chosen": {"mn_ref_hz": mn_c, "hair_plate_max_hz": hp_c, "campaniform_load_hz": ld_c}, "loop_gain_at_chosen": g_c, "g_model": g_desc, "D_full_at_chosen": D_c,
                                     "predicted_realised_gpu_full_window": pred, "predicted_minus_target_hz": {"chordotonal": pred["chordotonal_hz"] - ch_t, "hair_plate": pred["hair_plate_hz"] - hp_t, "campaniform": pred["campaniform_hz"] - cp_t},
                                     "uncertainty_hz": {"chordotonal": unc_ch, "hair_plate": unc_hp, "campaniform": unc_cp,
                                                        "note": f"window-factor spread across the steady GPU arms (relative {W_rel:.4f}) and the CPU pair's fly-level SE on the drive ratio (relative {g_se:.4f}), propagated on (rate - tonic); campaniform: load x SD of the steady arms' ground fraction; the predeclared tolerances are 3 / 3 / 1.5 Hz per side"},
                                     "sensitivity": {"d_chordotonal_d_mn_ref_hz_per_hz": (CH_TONIC + 140 * D_of(full, mn_c + 0.05, g_c) - (CH_TONIC + 140 * D_of(full, mn_c - 0.05, g_c))) / 0.1,
                                                     "note": "at a fixed loop gain (open loop); the loop makes the realised slope smaller in magnitude by the factor 1 / (1 + loop gain)"},
                                     "per_side": "equal by construction under 'unsided' (the targets' own sides differ by %.3f / %.3f Hz of chordotonal / hair plate)" % (tg["chordotonal_L"]["mean"] - tg["chordotonal_R"]["mean"], tg["hair_plate_L"]["mean"] - tg["hair_plate_R"]["mean"])}
        print(f"\nSTEP 3 (chosen): mn_ref_hz {mn_c}, hair_plate_max_hz {hp_c}, campaniform_load_hz {ld_c}; g at chosen {g_c:.4f} ({g_desc}); D_full {D_c:.6f}")
        print(f"   PREDICTED GPU realisation (5-60 s window, 16 flies): chordotonal {pred['chordotonal_hz']:.3f} +- {unc_ch:.2f} (target {ch_t:.3f}, {pred['chordotonal_hz'] - ch_t:+.3f}), hair plate {pred['hair_plate_hz']:.3f} +- {unc_hp:.2f} (target {hp_t:.3f}, {pred['hair_plate_hz'] - hp_t:+.3f}), "
              f"campaniform {pred['campaniform_hz']:.3f} +- {unc_cp:.2f} (target {cp_t:.3f}, {pred['campaniform_hz'] - cp_t:+.3f}); d chord / d mn_ref {rec['step_3_fixed_point']['sensitivity']['d_chordotonal_d_mn_ref_hz_per_hz']:+.2f} Hz/Hz open loop")
    # ---- step 4: the verification pair (longer CPU runs at the chosen values)
    if a.verify:
        stems = [s.strip() for s in a.verify.split(",")]
        V = [cal_point(Path(a.cal, s + ".json")) for s in stems if Path(a.cal, s + ".json").exists()]
        if len(V) >= 1:
            v3 = V[0]; vu = V[1] if len(V) > 1 else None
            ver = {"L3": v3, "anchor": vu}
            if vu:
                g_v = v3["leg_mn_mean_hz"] / vu["leg_mn_mean_hz"]
                # per-second ratio of the two arms' leg-MN rates over the longer window: is the loop gain stable in time?
                r = np.array(v3["leg_mn_mean_by_second"]) / np.maximum(np.array(vu["leg_mn_mean_by_second"]), 1e-9)
                sk = int(round(v3["skip_s"]))
                ver["loop_gain_over_window"] = {"g_window": g_v, "g_by_second": [round(float(x), 4) for x in r], "g_first_10s": float(r[sk:sk + 10].mean()), "g_last_10s": float(r[-10:].mean()),
                                                "note": "the fixed point scales the GPU anchor's whole-window leg-MN distribution by one g; a g that drifts over the longer CPU window would say the scaling is not uniform in time"}
                mn_c = v3["mn_ref_hz"]; D_v = D_of(full, mn_c, g_v)
                ver["predicted_gpu_full_window_with_verification_g"] = {"chordotonal_hz": CH_TONIC + 140 * D_v, "hair_plate_hz": HP_TONIC + (v3["hair_plate_max_hz"] - HP_TONIC) * D_v, "campaniform_hz": v3["campaniform_load_hz"] * ground_steady}
                # the early-window prediction at the chosen values against the longer run's own 2-12 s window (from its per-second MN: the realised early window)
                ver["pair_cpu"] = {"chordotonal_L3_minus_anchor_hz": v3["chordotonal_hz"] - vu["chordotonal_hz"], "hair_L3_minus_anchor_hz": v3["hair_plate_hz"] - vu["hair_plate_hz"], "camp_L3_minus_anchor_hz": v3["campaniform_hz"] - vu["campaniform_hz"],
                                   "AN04B003_L3": v3["AN04B003_L_R_hz"], "AN04B003_anchor": vu["AN04B003_L_R_hz"], "DNa02_L3": v3["DNa02_L_R_hz"], "DNa02_anchor": vu["DNa02_L_R_hz"]}
                print(f"\nSTEP 4 (verification, {v3['batch']} flies x {v3['seconds']} s, window {v3['skip_s']}-{v3['seconds']} s): L3 chordotonal {v3['chordotonal_hz']:.3f} (L {v3['chordotonal_L_hz']:.3f} / R {v3['chordotonal_R_hz']:.3f}), hair {v3['hair_plate_hz']:.3f}, camp {v3['campaniform_hz']:.3f}, leg MN {v3['leg_mn_mean_hz']:.3f}; "
                      f"anchor chordotonal {vu['chordotonal_hz']:.3f}, leg MN {vu['leg_mn_mean_hz']:.3f}; g over the window {g_v:.4f} (first 10 s {ver['loop_gain_over_window']['g_first_10s']:.4f}, last 10 s {ver['loop_gain_over_window']['g_last_10s']:.4f}); "
                      f"predicted GPU full window with this g: chordotonal {ver['predicted_gpu_full_window_with_verification_g']['chordotonal_hz']:.3f}, hair {ver['predicted_gpu_full_window_with_verification_g']['hair_plate_hz']:.3f}")
                print(f"   L3 v anchor on CPU: AN04B003 {v3['AN04B003_L_R_hz'][0]:.2f} / {v3['AN04B003_L_R_hz'][1]:.2f} vs {vu['AN04B003_L_R_hz'][0]:.2f} / {vu['AN04B003_L_R_hz'][1]:.2f}; DNa02 {v3['DNa02_L_R_hz'][0]:.3f} / {v3['DNa02_L_R_hz'][1]:.3f} vs {vu['DNa02_L_R_hz'][0]:.3f} / {vu['DNa02_L_R_hz'][1]:.3f}")
            for extra in V[2:]:
                ver.setdefault("others", []).append(extra)
                print(f"   {extra['arm']} ({extra['file']}): chordotonal {extra['chordotonal_hz']:.3f}, hair {extra['hair_plate_hz']:.3f}, camp {extra['campaniform_hz']:.3f}, leg MN {extra['leg_mn_mean_hz']:.3f}, AN04B003 {extra['AN04B003_L_R_hz'][0]:.2f} / {extra['AN04B003_L_R_hz'][1]:.2f}, DNa02 {extra['DNa02_L_R_hz'][0]:.3f} / {extra['DNa02_L_R_hz'][1]:.3f}")
            rec["step_4_verification"] = ver
            rec["step_4_verification"]["rule"] = "nothing is re-fitted after the verification; the GPU realisation is checked from the batch recordings against the predeclared tolerances (out/vncd7/verify_runs.py, precondition P1)"
    rec["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
