"""interp atlas -- what every motor readout does when population X is stimulated (docs/INTERP.md 4.5).

GPU half (`run`, one job per replicate on the cluster) and CPU half (`analyse`, on the desktop):

    # the round's batch: the validation arms and every descending-neuron type + sensory class, 3 runs each, ONE call
    python scripts/cluster_run.py --name atlas --minutes 90 \
      "mkdir -p out/atlas && python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_atlas.py run --preset validation --seed 0 --out out/atlas/val_r0 > out/atlas/val_r0.txt 2>&1; cat out/atlas/val_r0.txt" \
      "... --preset validation --seed 1 --out out/atlas/val_r1 ..." "... --seed 2 ..." \
      "... --preset dn+sensory --seed 0 --out out/atlas/dn_r0 ..." "... --seed 1 ..." "... --seed 2 ..." \
      --fetch out/atlas/ 2>&1 | tee out/atlas_cluster.log
    PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/val_r*" --top 8  --json out/interp/atlas/validation.json
    PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/dn_r*"  --top 20 --json out/interp/atlas/dn_sensory.json

`--preset validation` also records the wind / steering DN types by side (VALIDATION_PATTERN) so the DNp18 / DNp33
left-right flips can be read off; `docs/audits/interp_atlas.md` is the round this ran.

`record` is an alias of `run`. Every JSON carries the provenance block with the REALISED device; `Result.check()`
is printed when it is not empty.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import connectome as cn                                    # noqa: E402
from flyverse.interp import common                                       # noqa: E402
from flyverse.interp import atlas as atlas_mod                           # noqa: E402


def _load(args):
    return cn.load(cache_dir=Path(args.cache_dir) if args.cache_dir else cn.CACHE_DIR, verbose=not args.quiet)


def _pops(c, args, lif_hz, ms):
    if args.preset:
        return atlas_mod.preset_populations(c, args.preset, by_side=not args.no_by_side, hz=lif_hz, ms=ms,
                                            min_cells=args.min_cells)
    if not args.populations:
        raise SystemExit("give --preset or --populations")
    specs = args.populations if len(args.populations) > 1 else args.populations[0]
    return atlas_mod.make_populations(c, specs, by_side=not args.no_by_side,
                                      split=(None if args.split == "none" else args.split), hz=lif_hz, ms=ms,
                                      min_cells=args.min_cells)


def cmd_run(args) -> int:
    c = _load(args)
    lif, optic = common.params_from_args(args)
    pops = _pops(c, args, args.hz, args.ms)
    pattern = args.pattern if args.pattern is not None else (atlas_mod.VALIDATION_PATTERN if args.preset == "validation" else None)
    if not args.quiet:
        print(f"atlas run: {len(pops)} populations, batch {args.batch} ({args.n_null} null rows), "
              f"{args.hz} Hz for {args.ms} ms after {args.settle_ms} ms, context {args.context}, seed {args.seed}")
    run = atlas_mod.run_once(c, pops, hz=args.hz, ms=args.ms, settle_ms=args.settle_ms, batch=args.batch,
                             readouts=tuple(args.readouts.split(",")), pattern=pattern, by_side=not args.no_by_side,
                             n_null=(args.n_null if not args.no_null else 0), params=lif, optic_params=optic,
                             device=args.device, context=args.context, seed=args.seed,
                             modules=(args.modules.split(",") if args.modules else None),
                             per_body=not args.no_per_body, include_pn=args.include_pn, quiet=args.quiet)
    out = Path(args.out or (common.ROOT / "out" / "atlas" / f"run_seed{args.seed}"))
    run.meta["file"] = str(out.with_suffix(".npz"))
    run.save(out)
    dev = run.meta["provenance"]["execution"]["device"]
    print(f"device {dev}   {run.meta['n_populations']} populations x {len(run.readouts)} readouts   "
          f"{len(run.null_ids)} null rows   {run.meta['wall_s']}s   -> {out.with_suffix('.npz')}")
    if not args.quiet:
        df = run.frame("mean")
        cols = [x for x in ("leg_LR", "leg_L", "leg_R", "power", "gf", "turn_LR", "proboscis") if x in df.columns]
        if cols:
            common.print_table(df[cols].reindex(df[cols].abs().max(axis=1).sort_values(ascending=False).index)
                               .head(15).reset_index().rename(columns={"index": "population"}), max_rows=15)
    return 0


def cmd_analyse(args) -> int:
    files: list[str] = []
    for pat in args.runs:
        hits = sorted(glob.glob(pat)) or sorted(glob.glob(pat + ".json")) or sorted(glob.glob(pat + "*.json"))
        files += [h for h in hits if h.endswith(".json") or h.endswith(".npz")]
    files = sorted({str(Path(f).with_suffix("")) for f in files})
    if not files:
        raise SystemExit(f"no run files matched {args.runs}")
    runs = []
    for f in files:
        r = atlas_mod.AtlasRun.load(f)
        r.meta.setdefault("file", f + ".npz")
        runs.append(r)
    c = _load(args) if not args.no_connectome else None
    res = atlas_mod.analyse_runs(runs, c=c, top=args.top, summary=args.summary, sd_floor=args.sd_floor,
                                 per_body_for=(args.per_body_for.split(",") if args.per_body_for else None),
                                 generator="python scripts/interp_atlas.py " + " ".join(sys.argv[1:]))
    path = Path(args.json) if args.json else common.default_json_path("atlas", res.run_id)
    res.save(path)
    movers = res.table("movers")
    if not len(movers):
        movers = pd.DataFrame(columns=["readout"])
    show = args.readout.split(",") if args.readout else sorted(movers.readout.unique())
    for ro in show:
        g = movers[movers.readout == ro]
        if not len(g):
            continue
        print(f"\n=== {ro} ===  (null {g.iloc[0].null_mean:+.3f} Hz over {len(runs)} runs)")
        cols = [x for x in ("rank", "population", "n_cells", "stim_mean", "null_mean", "diff", "z_floor", "p",
                            "verdict", "self_drive", "stim_sd", "n_runs") if x in g.columns]
        common.print_table(g[cols], max_rows=args.top)
    if res.validation.get("measured"):
        print("\nvalidation (" + res.validation["status"] + "):")
        print(json.dumps(res.validation["measured"], indent=1)[:4000])
    problems = res.check()
    if problems:
        print("\nResult.check():", problems)
    print(f"\n{len(runs)} runs, {res.summary['n_populations']} populations x {res.summary['n_readouts']} readouts "
          f"-> {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "record"):
        p = sub.add_parser(name, help="one independent run (the GPU half)")
        common.add_common_args(p)
        p.add_argument("--preset", default=None, choices=["dn", "sensory", "dn+sensory", "validation"])
        p.add_argument("--populations", action="append", default=[], help="a common.resolve spec (repeatable)")
        p.add_argument("--split", default="type", choices=["type", "none"], help="split each spec into its types")
        p.add_argument("--hz", type=float, default=150.0)
        p.add_argument("--ms", type=float, default=400.0)
        p.add_argument("--settle-ms", type=float, default=200.0)
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--n-null", type=int, default=4, help="unstimulated rows per batch")
        p.add_argument("--no-null", action="store_true")
        p.add_argument("--readouts", default="motor")
        p.add_argument("--pattern", default=None, help="regex: also record every matching type (screen.TypeRecorder)")
        p.add_argument("--by-side", action="store_true", help="split populations and readouts by soma side (the default; docs/INTERP.md section 7)")
        p.add_argument("--no-by-side", action="store_true", help="pool the sides")
        p.add_argument("--min-cells", type=int, default=1)
        p.add_argument("--include-pn", action="store_true", help="one readout per PN glomerulus")
        p.add_argument("--no-per-body", action="store_true")
        p.add_argument("--modules", default=None, help="regions.MODULES subset, comma separated")
        p.add_argument("--context", default=None, help=f"one of {sorted(atlas_mod.CONTEXTS)}")
        p.add_argument("--out", default=None, help="run file prefix (writes .npz + .json)")
        p.set_defaults(fn=cmd_run)
    p = sub.add_parser("analyse", help="combine runs into the Result (the CPU half)")
    common.add_common_args(p)
    p.add_argument("--runs", action="append", required=True, help="run file prefix or glob (repeatable)")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--summary", default="mean", choices=list(atlas_mod.SUMMARIES))
    p.add_argument("--sd-floor", type=float, default=atlas_mod.SD_FLOOR_HZ)
    p.add_argument("--readout", default=None, help="print only these readouts (comma separated)")
    p.add_argument("--per-body-for", default=None, help="populations whose readout_per_body rows to write")
    p.add_argument("--no-connectome", action="store_true", help="skip loading the connectome (no per-body table)")
    p.set_defaults(fn=cmd_analyse)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
