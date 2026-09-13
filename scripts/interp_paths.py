"""CLI of the paths tool (flyverse/interp/paths.py; docs/INTERP.md 4.3): effective k-step signed gains A -> B through
the shaped weights, silent links flagged. CPU only -- structural, no simulation, no GPU part -- so there is one action,
`analyse` (the default), plus `validate`, which reproduces VALIDATION['paths'] (GLNO as the silent rotation -> PEN link,
the cx_wedge one-step ring weights, the ExR -> EPG two-step loop) and writes the Result JSONs with `validation.measured`.

    # the rotation inputs of PEN (type level, k <= 3)
    PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a "LNO1|LNO2|LNOa|SpsP|PS196_b|~^LAL" --b "PEN_a|PEN_b" --k 3 --json out/interp/paths/rot_pen.json
    # the ring loop at the cell level, aggregated by PB wedge as cx_wedge.py does
    PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a EPG --b EPG --k 2 --level cell --wedge --json out/interp/paths/epg_loop.json
    # never_firing flags from a recording; frozen units from the optic superclass rule (default) or a FlyBrain's rate_idx npz
    PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a T4a --b DNp15 --recording out/cxroom/rec_r0 --frozen static
    # the validation targets
    PYTHONIOENCODING=utf-8 python scripts/interp_paths.py validate --json out/interp/paths/validate.json

`--a` / `--b` take the population grammar of docs/INTERP.md 2.1; a '|'-joined string with '~regex' tokens is split into
an OR list (`paths.spec_from_cli`); both flags are repeatable (repeats OR). `--null-runs` / `--replicates` / `--seed` /
`--device` are accepted for uniformity: the tool is deterministic (there is nothing to replicate; the JSON records the
host CPU as the realised device); the only rollout-dependent flag is `never_firing`, judged on the named `--recording`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyverse import connectome as cn  # noqa: E402
from flyverse.interp import common  # noqa: E402
from flyverse.interp import paths as P  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="analyse", choices=["analyse", "validate"])
    ap.add_argument("--a", action="append", default=[], help="source population spec (repeatable = OR)")
    ap.add_argument("--b", action="append", default=[], help="target population spec (repeatable = OR)")
    ap.add_argument("--k", type=int, default=3, help="maximum path length in synapses (default 3)")
    ap.add_argument("--level", default="type", choices=["type", "cell"])
    ap.add_argument("--top", type=int, default=20, help="walks per k and kind (signed / silent) to list")
    ap.add_argument("--min-abs-mv", type=float, default=0.0, help="drop links weaker than this (if-signed mV per post volley)")
    ap.add_argument("--exclude", action="append", default=[], help="type (or ~regex) to drop from the intermediates (repeatable)")
    ap.add_argument("--recording", default=None, help="a Recording (npz/json path) whose max rates give the never_firing flag")
    ap.add_argument("--frozen", default="static", help="'static' (superclass ol_intrinsic rate units), 'none', or an npz with rate_idx")
    ap.add_argument("--rates-min-hz", type=float, default=common.NEVER_FIRING_HZ)
    ap.add_argument("--wedge", action="store_true", help="cell level, k >= 2: aggregate the two-step matrices by PB wedge (cx_wedge)")
    ap.add_argument("--wedge-group", action="append", default=[], metavar="NAME=REGEX", help="a wedge-profile group (default cx_wedge's PEN / PEG / Delta7 / Ring)")
    ap.add_argument("--contributions-per-link", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=40)
    common.add_common_args(ap)
    args = ap.parse_args(argv)
    lif, _optic = common.params_from_args(args)
    cache_dir = Path(args.cache_dir) if args.cache_dir else cn.CACHE_DIR
    c = cn.load(cache_dir=cache_dir, verbose=not args.quiet)
    log = (lambda *a, **k: None) if args.quiet else print

    if args.action == "validate":
        out_dir = Path(args.json).parent if args.json else common.default_json_path("paths", "validate").parent
        v = P.validate(c, params=lif, out_dir=out_dir, log=log)
        path = Path(args.json) if args.json else out_dir / "validate.json"
        summary = {"status": v["status"], "n_ok": v["n_ok"], "n": v["n"], "checks": v["checks"], "measured": v["measured"],
                   "results": {k: str(out_dir / f"validate_{k.replace('->', '_to_')}.json") for k in v["results"]},
                   "generator": "scripts/interp_paths.py " + " ".join(argv or sys.argv[1:])}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            import json
            json.dump(common.to_jsonable(summary), f, indent=1)
        common.print_table(pd.DataFrame(v["checks"])[["check", "measured", "reference", "ok"]], max_rows=200)
        print(f"validation {v['status']} ({v['n_ok']}/{v['n']} checks) -> {path}")
        return 0 if v["status"] == "reproduced" else 1

    if not args.a or not args.b:
        ap.error("--a and --b are required for analyse")
    a = [P.spec_from_cli(s) for s in args.a]; b = [P.spec_from_cli(s) for s in args.b]
    a = a[0] if len(a) == 1 else a; b = b[0] if len(b) == 1 else b
    groups = dict(kv.split("=", 1) for kv in args.wedge_group) or None
    res = P.paths(c, a, b, params=lif, k_max=args.k, top=args.top, min_abs_mv=args.min_abs_mv, recording=args.recording,
                  frozen=None if args.frozen == "none" else args.frozen, level=args.level, exclude=tuple(args.exclude),
                  rates_min_hz=args.rates_min_hz, wedge=args.wedge, wedge_groups=groups,
                  contributions_per_link=args.contributions_per_link, cache_dir=str(cache_dir))
    res.files["generator"] = "scripts/interp_paths.py " + " ".join(argv or sys.argv[1:])
    path = Path(args.json) if args.json else common.default_json_path("paths", res.run_id)
    res.save(path)
    if not args.quiet:
        pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
        pt = res.table("paths")
        if len(pt):
            print(f"\nwalks {a!r} -> {b!r}, level {args.level}, k <= {args.k} (gain in mV^k per {'volley' if args.level == 'type' else 'spike'}; "
                  f"'silent' walks contain a link that is an explicit zero, ranked by the weight it would carry if signed)")
            common.print_table(pt[["k", "kind", "rank", "path", "gain", "gain_if_signed", "signs", "silent_links", "raw_counts"]], max_rows=args.max_rows)
        lt = res.table("links")
        if len(lt):
            print("\nlinks on the listed walks (mV per pair over non-zero pairs; per post volley = mean over post cells of the summed input)")
            common.print_table(lt[["pre", "post", "pairs", "raw_count", "share_of_post_input", "mv_per_pair", "mv_per_pair_if_signed",
                                   "mv_per_post_volley", "mv_per_post_volley_if_signed", "sign", "silent", "silent_partial"]], max_rows=args.max_rows)
        bi = res.table("b_inputs")
        if len(bi):
            print(f"\npresynaptic types of b pooled (raw input {res.summary['b_raw_input_total']:.0f} synapses; silent share "
                  f"{100 * res.summary['b_silent_input_share']:.1f} %, sign-0 share {100 * res.summary['b_sign0_input_share']:.1f} %)")
            common.print_table(bi[["rank", "pre_type", "n_pre_cells", "entries", "raw_count", "share_of_b_input", "nt", "sign", "mv_per_pair",
                                   "mv_per_pair_if_signed", "mv_per_post_volley", "mv_per_post_volley_if_signed", "silent", "silent_partial"]], max_rows=args.max_rows)
        if "two_step_by_type" in res.tables and res.tables["two_step_by_type"]:
            print("\ntwo-step a -> type -> b, total per post cell (mV^2 per spike)")
            common.print_table(res.table("two_step_by_type")[["type", "cells", "nt", "a_to_type_mv_per_pair", "type_to_b_mv_per_pair",
                                                               "two_step_total_per_post_mv2", "two_step_total_per_post_mv2_if_signed", "silent"]], max_rows=args.max_rows)
        if "wedge_profile" in res.tables and res.tables["wedge_profile"]:
            print("\nring-distance profiles (16-wedge level, mean over bands; distance in wedges of 22.5 deg)")
            common.print_table(res.table("wedge_profile"), floatfmt="{:+.1f}")
        print("\nstrongest silent link per k:")
        for k, v in res.summary["strongest_silent_link_per_k"].items():
            print(f"  k={k}: " + ("none" if v is None else f"{v['pre']} -> {v['post']} [{v['silent']}] {v['mv_per_post_volley_if_signed']:+.2f} mV per post volley if signed "
                                                            f"({v['mv_per_pair_if_signed']:+.2f} per pair, {v['raw_count']:.0f} raw synapses, {v['pairs']} entries, "
                                                            f"{100 * v['share_of_post_input']:.1f} % of the post node's raw input) on {v['on_path']}"))
    problems = res.check()
    if problems:
        print("Result.check():", problems)
    print(f"json: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
