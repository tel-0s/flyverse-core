"""The toolkit's front door (flyverse/interp/deficit.py; docs/INTERP.md 10.1 and 11.2 item 14): name a ledger row that
fails, get steps 1-2 of the procedure run on the CPU and steps 3-7 written out. It runs no simulation and needs no GPU;
it calls the shipped `paths` and `decompose`, unchanged, and never edits the model.

    # a row by its ledger id, or by the scripts/benchmark.py check it mirrors
    PYTHONIOENCODING=utf-8 python scripts/interp_deficit.py compass.wedge_cells_persisting
    PYTHONIOENCODING=utf-8 python scripts/interp_deficit.py rotation.DNa02.rate_hz --out-dir out/interp/deficit/turning
    # score the row's status from finished results first (anything scripts/interp_ledger.py reads)
    PYTHONIOENCODING=utf-8 python scripts/interp_deficit.py taste.MN9_hz --results out/benchmark_suite.json
    # a stimulus with no declared sensory entry, or a different question: name the sources / the readout
    PYTHONIOENCODING=utf-8 python scripts/interp_deficit.py lit.KC.dan_tone_mv --source "DANs=~^(PAM|PPL1)"
    # which rows can be named (gap rows first; with --results, their statuses)
    PYTHONIOENCODING=utf-8 python scripts/interp_deficit.py --list
    # the validation target: GLNO -> PEN (sign 0) into the compass, DNa02's net-excitatory wiring, from the rows alone
    PYTHONIOENCODING=utf-8 python scripts/interp_deficit.py validate

Writes, under --out-dir (default out/interp/deficit/<row>): paths_<i>_<source>.json and decompose_static.json (each a
`flyverse.interp.result/1` JSON, as the tools' own wrappers write them), report.json (`flyverse.interp.deficit/1`) and
report.md, which is also printed. The common flags (--receptor-model, --lif, --cache-dir, ...) select the model the
structure is read from; the default is the shipped one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyverse import connectome as cn  # noqa: E402
from flyverse.interp import common  # noqa: E402
from flyverse.interp import deficit as DF  # noqa: E402
from flyverse.interp import ledger as L  # noqa: E402


# The population grammar's own keys (docs/INTERP.md 2.1: any neurons column, plus module / body / index): 'class=olfactory'
# is a spec, never the label 'class'.
GRAMMAR_KEYS = {*cn.KEEP_COLS, "module", "body", "bodyId", "index", "nt", "sign", "hex1", "hex2", "hex_side", "hex_source"}


def _source(s: str) -> tuple[str, str]:
    """'[LABEL=]SPEC' -> (label, spec). A LABEL is a plain word that is not a grammar key; anything else is the spec."""
    label, sep, spec = s.partition("=")
    label = label.strip()
    if sep and label and label not in GRAMMAR_KEYS and not any(ch in label for ch in "~&:|^$()"):
        return label, spec.strip()
    return s, s


def list_rows(table=None, results=(), null=()) -> pd.DataFrame:
    t = L.load_table(table)
    t = t.assign(sources=[", ".join(lbl for lbl, _ in DF.STIMULUS_SOURCES.get(s, ())) or ("the stimulated population" if s.endswith(DF.STIM_SUFFIX) else "--")
                          for s in t.stimulus])
    cols = ["row_id", "check_key", "gap", "population", "stimulus", "sources"]
    if results:
        led = L.ledger(list(results), table=table, null=list(null)).table("ledger")
        st = led.groupby("row_id").status.agg(lambda s: "/".join(sorted(set(s))))
        t = t.assign(status=t.row_id.map(st).fillna("MISSING"))
        cols.append("status")
    return t.sort_values(["gap", "row_id"], ascending=[False, True])[cols]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    action = argv.pop(0) if (argv and argv[0] == "validate") else "diagnose"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("row", nargs="?", help="a ledger row_id or the scripts/benchmark.py check key it mirrors")
    ap.add_argument("--list", action="store_true", help="print the rows that can be named and exit")
    ap.add_argument("--table", default=None, help="expectation table CSV (default flyverse/data/expected_responses.csv)")
    ap.add_argument("--results", nargs="+", default=[], metavar="PATH_OR_GLOB", help="finished results to score the row's status from")
    ap.add_argument("--source", action="append", default=[], metavar="[LABEL=]SPEC",
                    help="a sensory entry point for `paths` (repeatable; replaces the stimulus's declared sources)")
    ap.add_argument("--readout", default=None, help="the readout cells (a population spec), replacing the row's own")
    ap.add_argument("--k", type=int, default=3, help="maximum path length in synapses (default 3)")
    ap.add_argument("--top", type=int, default=5, help="walks per k and rows per table in the report")
    ap.add_argument("--out-dir", default=None, help="where the Result JSONs and the report go (default out/interp/deficit/<row>)")
    ap.add_argument("--name", default=None, help="the batch name the step-3 templates use (default: the row id)")
    common.add_common_args(ap)
    args = ap.parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    if args.list:
        with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60):
            print(list_rows(args.table, args.results, args.null_runs or []).to_string(index=False))
        return 0
    lif, _optic = common.params_from_args(args)
    cache_dir = Path(args.cache_dir) if args.cache_dir else cn.CACHE_DIR
    c = cn.load(cache_dir=cache_dir, verbose=not args.quiet)

    if action == "validate":
        out_dir = Path(args.out_dir or common.ROOT / "out" / "interp" / "deficit" / "validate")
        v = DF.validate(c, out_dir=out_dir, log=log)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validation.json").write_text(json.dumps(common.to_jsonable(v), indent=1), encoding="utf-8")
        print(json.dumps(common.to_jsonable(v["measured"]), indent=1))
        print(f"validation: {v['status']} -> {out_dir / 'validation.json'}")
        return 0 if v["status"] == "reproduced" else 1

    if not args.row:
        ap.error("name a row (a row_id or a check key), or pass --list")
    row = DF.find_row(args.row, args.table)
    out_dir = Path(args.out_dir or common.ROOT / "out" / "interp" / "deficit" / DF._slug(row.row_id))
    rep = DF.diagnose(c, args.row, table=args.table, results=args.results, null=args.null_runs or [],
                      sources=[_source(s) for s in args.source] or None, readout=args.readout, k_max=args.k, top=args.top,
                      params=lif, out_dir=out_dir, cache_dir=cache_dir, name=args.name, log=log)
    print()
    print(DF.render(rep))
    print(f"-> {rep['files']['report']}  {rep['files']['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
