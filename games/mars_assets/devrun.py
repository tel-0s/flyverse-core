"""Dev runner for games/mars.py (dev seeds >= 100 only; on the cluster's GPUs, never on the desktop): the game's own
`MarsRover`, ticked headless without the camera or the HUD, so a 40 s run costs about a third of a clip. Writes the
run log (meta, the game's events, the summary with its 10 Hz series) to --out.

    python games/mars_assets/devrun.py --seed 101 --control on --seconds 40 --out out/games/mars/dev/on_s101.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import games.common as gc  # noqa: E402
from games import mars as M  # noqa: E402


def main():
    ap = M.add_args(gc.standard_args("mars dev runner", seconds=40.0))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.seed >= 100, "dev seeds only"
    g = M.MarsRover(args)
    t0 = time.time()
    n = int(round(args.seconds * 100))
    for k in range(n):
        g.tick()
        if k % 500 == 0:
            print(f"  t {g.t_s:5.1f} s  x {g.x:6.1f}  y {g.y:+6.2f}  wall {time.time() - t0:6.1f} s", flush=True)
        if g.finished():
            break
    g.log.meta["wall_s"] = round(time.time() - t0, 1)
    g.log.meta["brain_s"] = round(g.t_s, 3)
    g.log.summary = M.run_summary(g)
    p = g.log.save(args.out, g.brains())
    s = g.log.summary
    print(json.dumps({k: v for k, v in s.items() if not isinstance(v, list) or len(v) < 20}, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
