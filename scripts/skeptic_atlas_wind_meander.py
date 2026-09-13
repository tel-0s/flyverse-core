"""SKEPTIC SCRATCH (delete after use): does the room's wind meander explain the atlas' higher DNp18 flip?

`flyverse/interp/atlas.py::wind_deflections` holds the nominal wind direction (WindParams.direction_deg 180), but
`flyverse.air.Air.direction` adds `meander_deg 20` sin-wandering with an 8 s period, and `scripts/benchmark.py::sec_wind`
averages 10 s of that.  The meander period is 8 s against millisecond neural time constants, so a quasi-static sweep
is a fair emulation: present the JO drive at N phases of one meander cycle, flip at each phase, average the flips.

Writes an AtlasRun exactly like `scripts/interp_atlas.py run` so `interp_atlas.py analyse` reads it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import connectome as cn                                    # noqa: E402
from flyverse.air import Air, WindParams                                 # noqa: E402
from flyverse.interp import atlas as A                                   # noqa: E402
from flyverse.interp import common                                       # noqa: E402


def phase_deflections(heading_deg: float, n: int, meander: bool) -> list[tuple[float, float, float]]:
    """[(t_s, dL, dR)] over one meander period at `heading_deg`, from the room's own Air."""
    air = Air(sources=[], wind=WindParams(meander_deg=(20.0 if meander else 0.0)))
    r = np.deg2rad(heading_deg)
    fwd = np.array([[np.cos(r), np.sin(r), 0.0]])
    left = np.array([[-np.sin(r), np.cos(r), 0.0]])
    out = []
    period = air.wind.meander_period_s
    for k in range(n):
        air.t = period * k / n
        dL, dR = air.deflections(fwd, left)
        out.append((air.t, float(dL[0]), float(dR[0])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    common.add_common_args(ap)
    ap.add_argument("--phases", type=int, default=8)
    ap.add_argument("--ms", type=float, default=3000.0)
    ap.add_argument("--settle-ms", type=float, default=200.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    c = cn.load(verbose=False)
    lif, optic = common.params_from_args(args)
    entries = []
    for side, hd in (("left", -90.0), ("right", 90.0)):
        for k, (t, dL, dR) in enumerate(phase_deflections(hd, args.phases, True)):
            w = A.jo_wind_rates(c, dL, dR)
            entries.append({"label": f"JO_meander_{side}_p{k}", "ms": args.ms, "idx": w["idx"], "hz": w["hz"],
                            "spec": f"senses.Wind.rates(dL={dL:+.4f}, dR={dR:+.4f})  # Air t={t:.2f}s, meander 20deg"})
    # the fixed-direction arms the atlas itself used, as the in-batch reference
    for side, hd in (("left", -90.0), ("right", 90.0)):
        dL, dR = A.wind_deflections(hd)
        w = A.jo_wind_rates(c, dL, dR)
        entries.append({"label": f"JO_fixed_{side}", "ms": args.ms, "idx": w["idx"], "hz": w["hz"],
                        "spec": f"senses.Wind.rates(dL={dL:+.4f}, dR={dR:+.4f})  # no meander"})
    pops = A.make_populations(c, entries, by_side=False, split=None, ms=args.ms)
    run = A.run_once(c, pops, ms=args.ms, settle_ms=args.settle_ms, batch=args.batch,
                     pattern=A.VALIDATION_PATTERN, params=lif, optic_params=optic, device=args.device,
                     seed=args.seed, quiet=args.quiet)
    out = Path(args.out)
    run.meta["file"] = str(out.with_suffix(".npz"))
    run.save(out)
    df = run.frame("mean")
    print(f"device {run.meta['provenance']['execution']['device']}   {len(pops)} populations   {run.meta['wall_s']}s")
    for t in ("DNp18", "DNp33", "WED080", "DNge016"):
        ro = f"type.{t}_LR"
        if ro not in df.columns:
            continue
        fl = [df.loc[f"JO_meander_left_p{k}", ro] - df.loc[f"JO_meander_right_p{k}", ro] for k in range(args.phases)]
        fixed = df.loc["JO_fixed_left", ro] - df.loc["JO_fixed_right", ro]
        print(f"{t:8s} meander-mean flip {np.mean(fl):+7.2f}  (phases " +
              " ".join(f"{v:+.1f}" for v in fl) + f")   fixed-direction flip {fixed:+7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
