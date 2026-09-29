"""Brain-free rule riders for TRON: how does the fly's decoded cycle compare with a one-line distance rule?

The recorded runs rank the fly against riders that turn at random times (games/tron.py's `summary.baselines`). This
script adds the comparison a caption skeptic asked for: a cyan rider that turns when the free run straight ahead is at
most `look` metres, choosing its side by one of three GAME rules (`room`: the side with the longer free run; `right`:
always right; `coin`: a fair coin, 50 draws per seed). Like the logged baselines it is a replay: games/tron.py's
brain-free game (`play_game_only`) on the same seed, with the orange FAFB cycle's recorded turns replayed open loop by
(round, race tick). No brain runs, so this is CPU-only and deterministic.

    python games/tron_rules.py                    # reads out/games/tron/rec2/seed{0,1,2}.json
    python games/tron_rules.py --json out/games/tron/rule_baselines.json

Every rider here is GAME: it knows the arena geometry exactly, which the fly does not.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import tron as T  # noqa: E402


class DistanceRule:
    """Turn 90 deg when the free run ahead is <= look_m (re-arm as the fly's decoder: T.REARM_S); side by `mode`."""

    kind = "never"   # the HUD / summary treat it as a brain-free cyan rider

    def __init__(self, look_m, mode, seed=0, draw=0):
        self.look, self.mode = float(look_m), mode
        self.rng = np.random.default_rng([0x7209, seed, draw])
        self.last = -1e9

    def step(self, game, me, t, live):
        if not live or t - self.last < T.REARM_S - 1e-9:
            return None, None
        others = [o for o in game.cycles if o is not me]
        if T.free_run(me, others, me.heading) > self.look:
            return None, None
        if self.mode == "room":
            fl = T.free_run(me, others, T.turn_heading(me.heading, "L"))
            fr = T.free_run(me, others, T.turn_heading(me.heading, "R"))
            side = "L" if fl >= fr else "R"
        elif self.mode == "right":
            side = "R"
        else:
            side = "L" if self.rng.random() < 0.5 else "R"
        self.last = t
        return side, side


def run(seed, log, look, mode, draw=0):
    turns = [e for e in log["events"] if e["kind"] == "turn" and e["player"] == "B"]
    g = T.play_game_only(seed, log["summary"]["ticks"], DistanceRule(look, mode, seed, draw), T.ReplayPilot(T.turn_schedule(turns)))
    o = T.outcome_of(g.log.summary)
    return {k: o[k] for k in ("crashes_A", "riding_s_A", "crashes_per_min_A", "won_A", "won_B", "draws", "turns_A")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", default="out/games/tron/rec2/seed{seed}.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--looks", default="1.1,1.5", help="look-ahead distances (m)")
    ap.add_argument("--coin-draws", type=int, default=50)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {"what": __doc__.split("\n\n")[1].replace("\n", " "), "seeds": {}}
    for s in (int(v) for v in args.seeds.split(",")):
        log = json.loads(Path(args.logs.format(seed=s)).read_text(encoding="utf-8"))
        fly = log["summary"]["baselines"]["fly"]
        row = {"fly": {"crashes_per_min_A": fly["crashes_per_min_A"], "won_A": fly["won_A"]}}
        for look in (float(v) for v in args.looks.split(",")):
            for mode in ("room", "right"):
                row[f"look{look}_{mode}"] = run(s, log, look, mode)
            coins = [run(s, log, look, "coin", d) for d in range(args.coin_draws)]
            cpm = np.array([c["crashes_per_min_A"] for c in coins])
            row[f"look{look}_coin"] = {"draws": len(coins), "crashes_per_min_A_median": float(np.median(cpm)),
                                       "crashes_per_min_A_range": [float(cpm.min()), float(cpm.max())],
                                       "draws_fewer_crashes_than_fly": int((cpm < fly["crashes_per_min_A"]).sum()),
                                       "draws_equal": int((cpm == fly["crashes_per_min_A"]).sum()),
                                       "won_A_median": float(np.median([c["won_A"] for c in coins]))}
        out["seeds"][s] = row
        print(f"seed {s}: fly {row['fly']}")
        for k, v in row.items():
            if k != "fly":
                print(f"   {k}: {v}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
