"""CLI for the interpretability toolkit's lesion tool (flyverse/interp/lesion.py; docs/INTERP.md section 4.4).

    # 1. plan: resolve the manifest on the connectome (CPU, local) and write the ONE cluster batch line
    PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py plan --manifest holds --out out/les_holds \
        --replicates 2 --minutes 25 --name les-holds
    sh out/les_holds/batch.sh                     # the single scripts/cluster_run.py call; ships out/les-holds_cluster.log

    # 2. one job (this is what every cluster job runs)
    python scripts/interp_lesion.py run --manifest holds --one holdBrain --replicate 0 --out out/les_holds

    # 3. the CPU protocol: every arm x replicate serially, in this process, on the CPU (no optic lobe in taste/smell)
    PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py run --manifest holds_cpu --device cpu \
        --out out/les_cpu --replicates 3 --sections taste,smell

    # 4. analyse: the check x lesion matrix, the sensitivity table and the double dissociations
    PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py analyse --out out/les_holds --json out/interp/lesion/holds.json

`--manifest` takes a builtin name ('holds' = the round-4 / round-5 hold arms of
docs/audits/receptor_integration.md E.1-E.2, 'holds_cpu' = the E.4 double dissociation), a JSON / YAML path, or a
literal JSON string (what `plan` embeds in the cluster commands, since out/ is git-ignored and is not shipped).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from flyverse.interp import common                                  # noqa: E402
from flyverse.interp import lesion as L                             # noqa: E402


def _manifest(args) -> dict:
    man = L.load_manifest(args.manifest)
    lif = common.parse_kv(args.lif)
    optic = common.parse_kv(args.optic)
    if args.receptor_table:
        lif["receptor_table"] = args.receptor_table
    if lif or optic:                       # --lif / --optic reach EVERY arm (benchmark.py's own CLI-override semantics)
        g = man.setdefault("global", {})
        g.setdefault("lif", {}).update(lif)
        g.setdefault("optic", {}).update(optic)
    if args.receptor_model != "default":
        man["baseline"]["receptor_model"] = args.receptor_model
        man["baseline"]["receptor_net_rule"] = args.receptor_net_rule
    if getattr(args, "probe", None):
        for spec in args.probe:
            d = json.loads(spec)
            man.setdefault("probes", []).append(d)
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plan", "run", "analyse"])
    ap.add_argument("--manifest", default="holds", help="builtin name (holds, holds_cpu), a JSON / YAML path, or a JSON string")
    ap.add_argument("--out", required=True, help="the run directory: job JSONs, batch.sh, manifest.resolved.json")
    ap.add_argument("--sections", default=None, help="scripts/benchmark.py sections (default: the manifest's)")
    ap.add_argument("--one", default=None, help="run: this lesion id only (one cluster job)")
    ap.add_argument("--replicate", type=int, default=None, help="run: the replicate index of this job")
    ap.add_argument("--brain-seed", type=int, default=None, help="run: the LIF Poisson seed (default: the manifest's brain_seeds[replicate])")
    ap.add_argument("--seeds", default="0,1,2", help="benchmark.py --seeds (the demo sections only)")
    ap.add_argument("--baseline", default="baseline", help="the arm every delta is measured against")
    ap.add_argument("--checks", default="all", help="analyse: comma-separated check keys (default all)")
    ap.add_argument("--tolerance", type=float, default=0.0, help="analyse: |delta| at or below this is 'not moved' (0 = bit-identity)")
    ap.add_argument("--rel-tolerance", type=float, default=0.0,
                    help="analyse: the same as a fraction of the baseline value (0.05 ignores moves under 5 %%)")
    ap.add_argument("--minutes", type=int, default=30, help="plan: the cluster_run.py estimate")
    ap.add_argument("--name", default=None, help="plan: the cluster job name (default les-<manifest>)")
    ap.add_argument("--cluster", action="store_true", help="plan: also submit the batch it wrote")
    ap.add_argument("--eager", action="store_true", help="benchmark.py --eager (torch path for the demo sections)")
    ap.add_argument("--probe", action="append", default=[], metavar="JSON",
                    help='an extra probe, e.g. \'{"id":"object","cmd":"python scripts/probe_object_sweep.py --null --seed {seed} --out {out}","read":"summary.LC11.diff_max_over_cells_mean_mv"}\'')
    common.add_common_args(ap)
    args = ap.parse_args()
    man = _manifest(args)
    res = L.lesion(man, out_dir=args.out, mode=args.mode, replicates=args.replicates, checks=args.checks,
                   seeds=args.seeds, cluster=args.cluster, minutes=args.minutes, baseline=args.baseline,
                   one=args.one, replicate=args.replicate, device=args.device, sections=args.sections,
                   tolerance=args.tolerance, rel_tolerance=args.rel_tolerance, brain_seed=args.brain_seed,
                   eager=args.eager, cache_dir=args.cache_dir, name=args.name, quiet=args.quiet)

    if args.mode == "plan" and not args.quiet:
        print("\n-- lesions resolved --")
        keep = ["id", "kind", "spec", "entries_changed_vs_sign_W", "n_bodies", "n_entries_removed", "receptor_table_md5"]
        df = res.table("lesions")
        common.print_table(df[[k for k in keep if k in df.columns]], floatfmt="{:.0f}")
        print(f"\n{res.summary['n_jobs']} jobs -> {res.summary['batch']}  (console log {res.summary['cluster_log']})")
    elif args.mode == "run" and not args.quiet:
        df = res.table("checks")
        if len(df):
            common.print_table(df[["lesion_id", "replicate", "key", "measured", "status"]], floatfmt="{:.4f}", max_rows=200)
    elif args.mode == "analyse":
        mat = res.table("matrix")
        if not args.quiet and len(mat):
            print("\n-- value per (check, lesion) --")
            common.print_table(L.pivot(mat, "value"), floatfmt="{:.4f}", max_rows=80)
            print("\n-- delta vs the baseline --")
            common.print_table(L.pivot(mat, "delta"), floatfmt="{:+.4f}", max_rows=80)
            print("\n-- replicate scatter (sd; nan = one run) --")
            common.print_table(L.pivot(mat, "replicate_sd"), floatfmt="{:.4f}", max_rows=80)
            dis = res.table("dissociations")
            print(f"\n-- {len(dis)} double dissociation(s) --")
            if len(dis):
                common.print_table(dis, floatfmt="{:+.4f}", max_rows=40)
            v = res.validation
            print(f"\n-- validation ({v.get('arm')}, {v.get('devices')}): {v['status']} "
                  f"({v['measured']['n_reproduced']}/{v['measured']['n_compared']} rows) --")
            common.print_table(pd_rows(v["measured"]["checks"]), floatfmt="{:.4f}", max_rows=80)
            if v["measured"]["entries_changed"]:
                common.print_table(pd_rows(v["measured"]["entries_changed"]), floatfmt="{:.0f}")

    path = args.json or common.default_json_path("lesion", res.run_id)
    res.save(path)
    problems = res.check()
    if problems:
        print("Result.check():", problems)
    print(f"wrote {path}")
    return 0


def pd_rows(rows):
    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.exit(main())
