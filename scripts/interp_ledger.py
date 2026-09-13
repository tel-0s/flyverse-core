"""CLI of the expectation ledger (flyverse/interp/ledger.py; docs/INTERP.md 4.7): score every finished probe output
against `flyverse/data/expected_responses.csv` -- the curated table of what each population should do under each
stimulus, with the literature citation and the file that measured it in this model.

CPU only. The ledger simulates nothing: `record` / `run` exist so every `scripts/interp_<tool>.py` takes the same
words, and both say so and stop. `analyse` (the default) is the tool; `validate` reproduces VALIDATION['ledger'];
`table` prints the expectation table itself.

    # everything the project has produced, with the object-sweep nulls as the null arm
    PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py --results "out/interp/*/*.json" out/benchmark_suite.json \
        "out/r3obj/ball_*.json" "out/loom2_*.txt" "out/bitter*.txt" "out/fg*.csv" "out/optic_audit/baseline/stages_s*.json" \
        --null "out/r3obj/null_*.json" --json out/interp/ledger/all.json
    # only the rows that are not PASS, one arm at a time
    PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py --results out/benchmark_suite.json --status FAIL,KNOWN GAP,MISSING
    # the validation target (docs/INTERP.md 6): benchmark.py's own JSON + the rotation screen
    PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py validate --json out/interp/ledger/validate.json --csv out/interp/ledger/validate.csv
    # the table itself
    PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py table --only compass

Sources are paths or globs; the kind is sniffed, not declared (`flyverse.interp.result/1` JSONs from any tool,
`scripts/benchmark.py` JSONs, `probe_object_sweep.py` / `probe_compass_room.py` / `batch_sustain.py` /
`probe_figure_stages.py` JSONs, `r5_attr_taste_cpu.py`'s arm x seed table, loom / bitter console logs, rotation /
figure-ground CSVs). An unreadable or unrecognised file is reported in the `sources` table, never fatal.

`--null` takes the none-vs-none runs (it is the common `--null` flag with values; `probe_object_sweep.py` JSONs that
carry `config.null` and the `CB` block of a stage JSON are picked up as nulls on their own). `--replicates` sets the
scatter rule's minimum (default 3; below it a difference reads `underpowered` whatever the numbers). `--seed` /
`--device` / `--receptor-*` are accepted for uniformity and recorded in the provenance: the ledger's realised device
is the CPU that scored, and each scored run's own realised device is in the `sources` table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyverse import connectome as cn  # noqa: E402
from flyverse.interp import common  # noqa: E402
from flyverse.interp import ledger as L  # noqa: E402

ACTIONS = ["analyse", "validate", "table", "record", "run"]


def _filter(df: pd.DataFrame, only, statuses, arms) -> pd.DataFrame:
    if not len(df):
        return df
    if only:
        df = df[df.row_id.str.startswith(tuple(only))]
    if statuses and "status" in df.columns:
        df = df[df.status.isin(statuses)]
    if arms and "arm" in df.columns:
        df = df[df.arm.isin(arms)]
    return df


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    action = argv.pop(0) if (argv and argv[0] in ACTIONS) else "analyse"   # a leading verb, else straight to analyse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 conflict_handler="resolve")
    ap.add_argument("sources", nargs="*", default=[], help="result / probe files or globs (same as --results)")
    ap.add_argument("--results", nargs="+", default=[], metavar="PATH_OR_GLOB",
                    help="result / probe files or globs to score")
    ap.add_argument("--table", default=None, help="expectation table CSV (default flyverse/data/expected_responses.csv)")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="relax every bound by this fraction (default 0 = benchmark.py's own criteria)")
    ap.add_argument("--strict", action="store_true", help="a MISSING row counts as a failure in summary.ok")
    ap.add_argument("--group-by", default="arm", choices=["arm", "none"],
                    help="'arm' (default): one row per receptor / gain / hold arm; 'none': pool every run")
    ap.add_argument("--only", action="append", default=[], metavar="PREFIX",
                    help="keep rows whose row_id starts with this (repeatable)")
    ap.add_argument("--status", default=None,
                    help=f"comma-separated statuses to print (of {', '.join(L.STATUSES)})")
    ap.add_argument("--arm", action="append", default=[], help="keep only these arms in the printed table (repeatable)")
    ap.add_argument("--csv", default=None, help="also write the ledger table here as CSV")
    ap.add_argument("--max-rows", type=int, default=200)
    ap.add_argument("--no-connectome", action="store_true",
                    help="do not load the cache: skip the population resolution (scoring itself needs no connectome)")
    ap.add_argument("--exit-nonzero", action="store_true", help="exit 1 when summary.ok is false (for CI)")
    common.add_common_args(ap)
    # the common --null is a flag; for the ledger it names the none-vs-none runs (docs/audits/object_sweep.md 8.4)
    ap.add_argument("--null", nargs="*", default=[], metavar="PATH_OR_GLOB",
                    help="none-vs-none runs that form the null arm")
    args = ap.parse_args(argv)
    args.action = action
    log = (lambda *a, **k: None) if args.quiet else print

    if args.action in ("record", "run"):
        print("the ledger has no GPU part: it scores finished files. Use `analyse` (the default) or `validate`.")
        return 2

    sources = list(args.results) + list(args.sources)
    if args.action == "table":
        tab = L.load_table(args.table)
        tab = tab[tab.row_id.str.startswith(tuple(args.only))] if args.only else tab
        if not args.quiet:
            common.print_table(tab[["row_id", "population", "label", "stimulus", "quantity", "expected", "op", "bound",
                                    "unit", "gap", "check_key", "requires"]], max_rows=args.max_rows)
        log(f"\n{len(tab)} expectation rows from {L.TABLE_PATH if args.table is None else args.table}")
        return 0

    lif, optic = common.params_from_args(args)
    cache_dir = Path(args.cache_dir) if args.cache_dir else cn.CACHE_DIR
    c = None if args.no_connectome else cn.load(cache_dir=cache_dir, verbose=not args.quiet)

    if args.action == "validate":
        res, side = L.validate(sources or None, table=args.table, c=c, cache_dir=cache_dir, null=args.null, log=log)
        if not args.quiet:
            common.print_table(side.drop(columns=["sources"]), max_rows=args.max_rows)
    else:
        if not sources:
            ap.error("no sources: pass files or globs with --results (or as positional arguments)")
        res = L.ledger(sources, table=args.table, tolerance=args.tolerance, strict=args.strict, null=args.null, c=c,
                       group_by=None if args.group_by == "none" else "arm", min_replicates=args.replicates,
                       cache_dir=cache_dir, lif=lif, optic=optic,
                       generator="scripts/interp_ledger.py " + " ".join(sys.argv[1:]))
        led = res.table("ledger")
        statuses = [s.strip() for s in args.status.split(",")] if args.status else None
        if not args.quiet:
            common.print_table(_filter(led, args.only, statuses, args.arm)[L.PRINT_COLUMNS], max_rows=args.max_rows)

    s, srcs = res.summary, res.table("sources")
    if len(srcs):
        bad = srcs[srcs.error != ""]
        log(f"\nsources: {len(srcs)} read, {len(srcs) - len(bad)} yielded observations"
            f"{'' if not len(bad) else '; problems: ' + '; '.join(f'{r.source}: {r.error}' for r in bad.itertuples())}")
        log(f"devices of the scored runs: {s.get('sources', {}).get('devices') or ['(none recorded)']}")
    if s:
        log(f"status counts: {s['status_counts']}")
        log(f"battery rows {s['battery_rows']}, agreeing {s['battery_agree']}, "
            f"disagreeing {len(s['battery_disagree'])}, stale criterion {len(s['battery_criterion_mismatch'])}")
        for d in s["battery_disagree"]:
            log(f"  DISAGREE {d['row_id']} [{d['arm']}]: ledger {d['status']} vs benchmark.py {d['battery_status']}")
        for d in s["battery_criterion_mismatch"]:
            log(f"  STALE CRITERION {d['row_id']} [{d['arm']}]: table {d['op']} {d['bound']} vs stored "
                f"'{d['battery_criterion']}' (ledger {d['status']}, file {d['battery_status']})")
        up = s.get("underpowered_rows") or []
        if up:
            log(f"under {args.replicates} runs (no difference is a result there): {len(up)} rows, "
                f"{', '.join(up[:10])}{' ...' if len(up) > 10 else ''}")
    log(f"validation: {res.validation.get('status')}")

    path = Path(args.json) if args.json else common.default_json_path("ledger", res.run_id)
    res.save(path)
    log(f"wrote {path}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        res.table("validation" if args.action == "validate" else "ledger").to_csv(args.csv, index=False)
        log(f"wrote {args.csv}")
    problems = res.check()
    if problems:
        log("Result.check(): " + "; ".join(problems))
    if args.exit_nonzero and not res.summary.get("ok", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
