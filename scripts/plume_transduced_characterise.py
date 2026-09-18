"""CPU-only sensory signal/noise characterization; no brain or room rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from flyverse import air, body, connectome, world
from flyverse.navigation import PlumeNavigation, bilateral_orn_groups
from flyverse.olfaction import ODOURS
from flyverse.senses import Smell


def source_hashes():
    files = [
        "flyverse/navigation.py",
        "flyverse/instruments.py",
        "flyverse/senses.py",
        "flyverse/brain.py",
        "flyverse/air.py",
        "flyverse/olfaction.py",
        "flyverse/world.py",
        "scripts/plume_transduced_characterise.py",
    ]
    return {
        p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in files
    }


def prediction(smell, groups, weights, left, right):
    """Independent input spike model: weighted means, delta-method contrast noise."""
    rates = smell.rates(left, right)[0]
    positions = {int(ix): j for j, ix in enumerate(smell.orn_idx)}
    means, intensity, bernoulli = [], [], []
    for side in ("L", "R"):
        r = rates[[positions[int(ix)] for ix in groups[side]]]
        w = weights[side].ravel().astype(float)
        means.append(float(w @ r))
        intensity.append(float((w * w) @ r))
        bernoulli.append(float((w * w) @ (r * (1 - r * 0.0005))))
    l, r = means
    contrast = (l - r) / (l + r)
    jac = np.array([2 * r, -2 * l]) / (l + r) ** 2
    # Exponential filters with tau1=.1 and tau2=.25 have integrated squared
    # impulse response 1/[2*(tau1+tau2)]. This linearization is not exact for a ratio.
    sd = math.sqrt(sum(intensity) / 0.25)
    sd_filtered = math.sqrt(sum(intensity) / 0.7)
    sd_contrast = math.sqrt(float(jac**2 @ np.array(intensity)) / 0.7)
    raw = [float(rates[smell.side == s].mean()) for s in (1, -1)]
    return {
        "left_hz": l,
        "right_hz": r,
        "difference_hz": l - r,
        "contrast": contrast,
        "sd_difference_box_250ms_hz": sd,
        "snr_box_250ms": abs(l - r) / sd,
        "sd_difference_cascade_hz": sd_filtered,
        "snr_cascade": abs(l - r) / sd_filtered,
        "sd_contrast_cascade_linearized": sd_contrast,
        "sd_difference_bernoulli_cascade_hz": math.sqrt(sum(bernoulli) / 0.7),
        "naive_unmatched_difference_hz": raw[0] - raw[1],
    }


def monte_carlo(table, profile, total, contrast, *, seeds=range(8), seconds=30):
    """Exact grouped Bernoulli sensory spikes at .5 ms, rate EMA at .1 s,
    then normalized contrast at the 10 ms frame boundary and .25 s EMA.
    This has no synaptic input, no circuit noise and no feedback.
    """
    ng = len(table)
    counts = np.array([[r["left"], r["right"]] for r in table]).T
    masses = np.array([r["mass"] for r in table])
    masses /= masses.sum()
    conc = total * np.array([profile.get(r["glomerulus"], 0.0) for r in table])
    lam = 1 + 150 * (conc[None] * np.array([1 + contrast, 1 - contrast])[:, None]) / (
        0.5 + conc[None] * np.array([1 + contrast, 1 - contrast])[:, None]
    )
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rate = np.zeros((2, ng))
        filtered = 0.0
        samples = []
        ar = math.exp(-0.0005 / 0.1)
        ac = math.exp(-0.01 / 0.25)
        for step in range(int(seconds / 0.0005)):
            k = rng.binomial(counts, lam * 0.0005)
            rate = ar * rate + (1 - ar) * k / counts / 0.0005
            if step % 20 == 19:
                l, r = rate @ masses
                filtered = ac * filtered + (1 - ac) * (l - r) / (l + r)
                if step >= 4000:
                    samples.append((l - r, filtered))
        a = np.asarray(samples)
        rows.append(
            {
                "seed": seed,
                "mean_difference_hz": float(a[:, 0].mean()),
                "sd_rate_difference_hz": float(a[:, 0].std(ddof=1)),
                "mean_filtered_contrast": float(a[:, 1].mean()),
                "sd_filtered_contrast": float(a[:, 1].std(ddof=1)),
                "fraction_correct_sign": float((a[:, 1] > 0).mean()),
                "mean_abs_goal_offset_deg": float(
                    np.rad2deg(
                        np.abs(
                            np.arctan(200 * np.arctanh(np.clip(a[:, 1], -0.999, 0.999)))
                        )
                    ).mean()
                ),
            }
        )
    return {
        "seconds": seconds,
        "burn_in_s": 2,
        "dt_ms": 0.5,
        "frame_ms": 10,
        "seeds": list(seeds),
        "rows": rows,
    }


def characterize(history):
    c = connectome.load()
    smell = Smell(c)
    groups, weights, table = bilateral_orn_groups(c)
    old = json.loads(history.read_text(encoding="utf-8"))
    data = np.asarray(old["samples"])
    j = old["columns"].index("antenna_L")
    total = data[:, :, j : j + 2].mean(2)
    totals = np.quantile(total, [0.05, 0.5, 0.95])
    profiles = {}
    for name in ("apple", "lime", "banana"):
        profiles[name] = {
            g: v / sum(ODOURS[name].values()) for g, v in ODOURS[name].items()
        }
    profiles["all_fruit_equal_source"] = {}
    for name in ("apple", "orange", "banana", "lime", "grape", "blueberry"):
        for g, v in ODOURS[name].items():
            profiles["all_fruit_equal_source"][g] = (
                profiles["all_fruit_equal_source"].get(g, 0) + v
            )
    p = profiles["all_fruit_equal_source"]
    profiles["all_fruit_equal_source"] = {g: v / sum(p.values()) for g, v in p.items()}
    scenarios = []
    for name, profile in profiles.items():
        for q, concentration in zip((0.05, 0.5, 0.95), totals):
            for contrast in (0.0, 0.0026, 0.0057, 0.007, 0.011):
                left = {
                    g: concentration * v * (1 + contrast) for g, v in profile.items()
                }
                right = {
                    g: concentration * v * (1 - contrast) for g, v in profile.items()
                }
                scenarios.append(
                    dict(
                        profile=name,
                        total_quantile=q,
                        total=float(concentration),
                        physical_contrast=contrast,
                        **prediction(smell, groups, weights, left, right),
                    )
                )
    # Actual static sensory fields from the room generator, not a neural room run.
    stationary = []
    for seed in range(6):
        _, info = world.make_room(seed)
        field = air.Air([(n, p, r / 0.02, r) for n, p, r in info["fruit"]], seed=seed)
        fly = body.FlyState(
            x=-0.5, y=0.05, z=info["table_top_z"], heading=np.deg2rad(5)
        )
        for t in (0.0, 2.0, 4.0, 6.0):
            field.t = t
            left, right = field.antennae(fly.eye_pos, fly.left, fly.forward)
            stationary.append(
                dict(
                    env_seed=seed,
                    time_s=t,
                    concentration_sum_left=float(sum(left.values())[0]),
                    concentration_sum_right=float(sum(right.values())[0]),
                    **prediction(smell, groups, weights, left, right),
                )
            )
    mc = monte_carlo(
        table, profiles["all_fruit_equal_source"], float(totals[1]), 0.0057
    )
    inst = PlumeNavigation(c, bilateral="orn")
    return {
        "provenance": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256_lf": source_hashes(),
            "dataset": c.dataset,
            "release": c.release,
            "cache_md5": {
                name: hashlib.md5(
                    (connectome.CACHE_DIR / name).read_bytes()
                ).hexdigest()
                for name in ("neurons.parquet", "W_post_pre.npz", "sign0_counts.npz")
            },
            "preset": "instrumented",
            "instruments": [inst.describe()],
            "execution": "CPU analytical and isolated sensory-spike Monte Carlo; no room or full-brain rollout",
        },
        "history": {
            "file": "out/plume_validation_v1/compass.json",
            "sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
            "samples": int(total.size),
            "total_quantiles": totals.tolist(),
            "limitation": "only antenna sums were logged; odor profiles below are explicit scenarios, not recovered historical glomerular mixtures or measured brain.rate",
        },
        "counts": {
            "total": len(smell.orn_idx),
            "left": int((smell.side == 1).sum()),
            "right": int((smell.side == -1).sum()),
            "unsided": int((smell.side == 0).sum()),
            "matched_glomeruli": len(table),
            "effective_left": float(1 / np.square(weights["L"]).sum()),
            "effective_right": float(1 / np.square(weights["R"]).sum()),
        },
        "glomeruli": table,
        "scenarios": scenarios,
        "stationary_room_sensors": stationary,
        "monte_carlo": mc,
        "monte_carlo_null": monte_carlo(
            table, profiles["all_fruit_equal_source"], float(totals[1]), 0.0
        ),
        "assumptions": [
            "independent sensory Poisson spikes; Bernoulli correction uses the shipped .5 ms step",
            "r(c)=1+150c/(c+.5) is forcing, not a measurement of full-brain ORN or PN rates",
            "250 ms box SD and 100+250 ms cascade SD are distinct; cascade contrast SD is a delta-method approximation",
            "stationary field samples are sensor-only CPU evaluations, not behavioral evidence",
            "full-brain rates, correlations and functional food finding remain unmeasured in this experiment",
        ],
    }


def report(result):
    lines = [
        "# CPU sensory characterization",
        "",
        "Independent-input model; no room rollout or behavioral claim.",
        "",
        "| Profile (median historical total) | Physical contrast % | L-R Hz | SD, 250 ms box Hz | SNR box | SNR cascade |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["scenarios"]:
        if r["total_quantile"] == 0.5 and r["physical_contrast"]:
            lines.append(
                f"| {r['profile']} | {r['physical_contrast'] * 100:.2f} | {r['difference_hz']:.3f} | {r['sd_difference_box_250ms_hz']:.3f} | {r['snr_box_250ms']:.3f} | {r['snr_cascade']:.3f} |"
            )
    lines += [
        "",
        "Generated by scripts/plume_transduced_characterise.py. Exact model values and all conditions are in characterisation.json.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--history", type=Path, default=ROOT / "out/plume_validation_v1/compass.json"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.out / "characterisation.json").exists():
        ap.error("output exists; use a fresh directory")
    result = characterize(args.history)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "characterisation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out / "characterisation.md").write_text(
        report(result), encoding="utf-8", newline="\n"
    )
    print(report(result))
