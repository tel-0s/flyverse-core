"""Direction selectivity probe: drift a sine grating across the retina in 4 directions and measure the
optic-lobe response of T4a-d / T5a-d (and downstream LPi, HS/VS, LC/LPLC, DNs).

Drosophila: T4a/T5a prefer front-to-back, b back-to-front, c upward, d downward (layer-wise in the
lobula plate). The grating is synthesised directly as per-column radiance (no ray tracing).

    python scripts/probe_motion.py [--speed 60] [--period 30] [--contrast 0.5]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import probe_vision_common as vc

from flyverse import brain, optic

DIRS = {"front->back": 0, "back->front": 1, "up": 2, "down": 3}


def grating(r, t_s, direction, speed_deg_s, period_deg, contrast, mean=0.3):
    az, el = r.col_az_el[:, 0], r.col_az_el[:, 1]
    # front->back on each eye = increasing |azimuth|
    if direction in ("front->back", "back->front"):
        x = np.abs(az)
        sgn = 1 if direction == "front->back" else -1
    else:
        x = el
        sgn = 1 if direction == "up" else -1
    phase = 2 * np.pi * (x - sgn * speed_deg_s * t_s) / period_deg
    lum = mean * (1 + contrast * np.sin(phase))
    rad = np.stack([0.5 * lum, lum, lum, lum], axis=1).astype(np.float32)
    return torch.from_numpy(rad)


def main():
    ap = argparse.ArgumentParser()
    vc.add_arguments(ap)
    ap.add_argument("--speed", type=float, default=60.0)
    ap.add_argument("--period", type=float, default=30.0)
    ap.add_argument("--contrast", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--slow-tau", type=float, default=0, help="override the slow-cell time constant (Mi4/Mi9/CT1/Tm9)")
    ap.add_argument("--t4-baseline", type=float, default=-1, help="operating point of T4/T5 units (default 0.5)")
    ap.add_argument("--inh-gain", type=float, default=1.0, help="gain on Mi4/Mi9/CT1/C3 -> T4 and Tm4/Tm9/CT1 -> T5 weights")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.seconds < .03:
        ap.error("--seconds must leave at least three frames")
    c, r = vc.load(args)
    types = c.neurons.type.fillna("").to_numpy()
    params = optic.OpticParams()
    if args.t4_baseline >= 0:
        params.baseline_by_type = {t: args.t4_baseline for t in ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]}
    if args.inh_gain != 1.0:
        params.pair_gain = [(r"^(Mi4|Mi9|CT1|C3)$", r"^T4[abcd]$", args.inh_gain), (r"^(Tm4|Tm9|CT1|TmY15)$", r"^T5[abcd]$", args.inh_gain)]
    if args.slow_tau > 0:
        params.tau_by_type = dict(optic.DEFAULT_TAU_BY_TYPE, Mi4=args.slow_tau, Mi9=args.slow_tau, CT1=args.slow_tau, Tm9=args.slow_tau)
    ol = optic.OpticLobe(c, r, params, device=args.device)
    ol.relax()
    rt = types[ol.rate_idx]
    b = brain.Brain(c, device=args.device)
    b.freeze(ol.rate_idx)
    probe_ol = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9", "LPi34", "LPi43", "LPi12", "LPi21"]
    probe_spk = ["HSE", "HSN", "HSS", "VS", "H2", "LPLC2", "LC4", "LPC1", "LLPC1", "DNa02", "DNp09", "MDN", "DNa01"]
    results = {}
    included = (c.neurons.somaSide.to_numpy()[ol.rate_idx] == "R") if args.eye == "right" else np.ones(len(rt), dtype=bool)
    for name in DIRS:
        ol.reset(); b = brain.Brain(c, device=args.device); b.freeze(ol.rate_idx)
        n_frames = int(args.seconds * 100)
        acc = {}
        for k in range(n_frames):
            rad = grating(r, k * 0.01, name, args.speed, args.period, args.contrast)
            b.drive = ol.step_frame(rad, b.rate, 10.0)
            b.step(20)
            if k >= n_frames // 3:   # average after onset transient
                dr = ol.last["dr"][0].cpu().numpy()
                for t in probe_ol:
                    values = dr[(rt == t) & included]
                    acc.setdefault(t, []).append(np.maximum(values, 0).mean() if len(values) else float("nan"))
        spk = b.rate[0].cpu().numpy()
        results[name] = ({t: float(np.mean(v)) for t, v in acc.items()}, {t: float(spk[c.select(type=t)].mean()) for t in probe_spk})
        if not args.quiet:
            print(f"[{name:12s}] OL mean dr: " + " ".join(f"{t}={results[name][0][t]:+.3f}" for t in probe_ol[:8]))
            print("               spiking Hz: " + " ".join(f"{t}={results[name][1][t]:.1f}" for t in probe_spk))
    print("\nDirection selectivity (mean rectified dr per direction; each T4/T5 subtype should peak in a different direction):")
    rows = []
    for t in ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9", "LPi34", "LPi43"]:
        vals = [results[d][0][t] for d in DIRS]
        best = list(DIRS)[int(np.argmax(vals))]
        dsi = (max(vals) - min(vals)) / (abs(max(vals)) + abs(min(vals)) + 1e-6)
        print(f"  {t:6s} " + " ".join(f"{d}={v:+.3f}" for d, v in zip(DIRS, vals)) + f"   best={best:12s} DSI={dsi:.2f}")
        if t in probe_ol[:8]:
            expected = list(DIRS)["abcd".index(t[-1])]
            valid = bool(np.isfinite(vals).all() and np.sum(np.asarray(vals) == max(vals)) == 1 and max(vals) > 0)
            rows.append({"type": t, "n": int(((rt == t) & included).sum()), "responses": dict(zip(DIRS, vals)),
                         "preferred": best, "expected": expected, "dsi": dsi, "pass": valid and best == expected})
    passed = all(row["pass"] for row in rows) and len(rows) == 8
    vc.result(args, c, r, b, params, "probe_motion", rows=rows,
              summary={"correct_subtypes": sum(row["pass"] for row in rows), "all_eight_correct": passed},
              status="pass" if passed else "fail", criterion="finite, positive, unique preferred direction correct for all T4a-d and T5a-d")


if __name__ == "__main__":
    main()
