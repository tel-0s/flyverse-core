"""Skeptic's independent re-derivation of the round-4 GLNO seed-matched grid (docs/audits/cx_glno.md section 5).

Reads the round-3 + skeptic + round-4 cx_glno JSONs directly (NOT through cx_glno.seed_table) and re-computes,
from the raw t5.0_* fields:
  * the de-duplicated (config, gE, gD, seed) grid and the determinism check on repeated keys
  * the shipped persistence rule (t5.0_in_above >= 8 of n_in AND t5.0_out_above <= 3 of n_out), the `confined`
    count (in_above >= 8) and the `boundary` count (confined, out_above == 4)
  * the per-seed outside-count table over the confined runs
  * per-point x condition persist / boundary / confined counts and the rate means +- sd
  * receptor_model / receptor_net_rule / nt_override / cache_dir recorded in every row
CPU only.  Usage:
  python scripts/verify_cx_glno_r4.py [--files <glob> ...] [--point 2:15 2.25:25 2.5:25]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

DEFAULT_FILES = [
    "out/cx_glno_gaba_gE*.json", "out/cx_glno_base*.json", "out/sk_base_*.json",
    "out/sk_gaba_*.json", "out/r4_base_*.json", "out/r4_gaba_*.json",
]
SKIP_FIELDS = {"wall_s", "cache_dir", "glno_structure"}


def load(files):
    paths = []
    for pat in files:
        paths += sorted(Path(ROOT).glob(pat)) if any(c in pat for c in "*?[") else [Path(ROOT) / pat]
    rows, dups = {}, []
    for p in paths:
        if not p.exists():
            print(f"missing {p}")
            continue
        for r in json.load(open(p)):
            key = (r["config"], r["gE"], r["gD"], r["seed"])
            if key in rows:
                prev, prevp = rows[key]
                diff = {k: (prev.get(k), r.get(k)) for k in set(prev) | set(r)
                        if k not in SKIP_FIELDS and prev.get(k) != r.get(k)}
                dups.append((key, prevp, p.name, diff))
            else:
                rows[key] = (r, p.name)
    return rows, dups, paths


def mean_sd(v):
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--point", nargs="*", default=["2.0:15.0", "2.25:25.0", "2.5:25.0"])
    a = ap.parse_args()
    rows, dups, paths = load(a.files)
    print(f"{len(rows)} unique (config, gE, gD, seed) keys from {len(paths)} files; {len(dups)} repeated keys")
    bad = [d for d in dups if d[3]]
    for key, p1, p2, diff in dups:
        tag = "IDENTICAL" if not diff else f"DIFFERS in {len(diff)} fields: {sorted(diff)[:6]}"
        print(f"  repeat {key}: {p1} vs {p2} -- {tag}")
    print(f"determinism: {len(dups) - len(bad)} of {len(dups)} repeated keys identical in every field "
          f"(skipping {sorted(SKIP_FIELDS)})")

    # per-row derived quantities
    recs = []
    for (cfg, gE, gD, seed), (r, src) in rows.items():
        in_n, out_n = r["t5.0_in_above"], r["t5.0_out_above"]
        recs.append(dict(config=cfg, gE=gE, gD=gD, seed=seed, src=src,
                         n_in=r["n_in"], n_out=r["n_out"], in_n=in_n, out_n=out_n,
                         persist=(in_n >= 8 and out_n <= 3), confined=(in_n >= 8),
                         boundary=(in_n >= 8 and out_n == 4),
                         bump=r["t5.0_in_mean"], out_hz=r["t5.0_out_mean"], vs=r["t5.0_vector_strength"],
                         pen=r["t5.0_pen"], delta7=r["t5.0_delta7"], glno=r["t5.0_glno"],
                         rest=r["t5.0_rest"], centre=r["t5.0_centre_wedge"],
                         pre_in=r["pre_in_mean"], pre_out=r["pre_out_mean"],
                         pre_in_n=r["pre_in_above"], pre_out_n=r["pre_out_above"],
                         receptor_model=r.get("receptor_model"), net_rule=r.get("receptor_net_rule"),
                         nt_override=json.dumps(r.get("nt_override")), cache=r.get("cache_dir"),
                         glno_nt=",".join(sorted(set(r.get("glno_nt", [])))),
                         glno_w=r.get("glno_structure", {}).get("glno_to_pen_W_sum")))

    # recorded receptor / override / cache identity
    print("\nrecorded per-row identity (config -> receptor_model / net_rule / nt_override / glno_nt / GLNO->PEN Wsum / caches):")
    for cfg in sorted({x["config"] for x in recs}):
        sub = [x for x in recs if x["config"] == cfg]
        print(f"  {cfg}: n={len(sub)} receptor_model={sorted({str(x['receptor_model']) for x in sub})} "
              f"net_rule={sorted({str(x['net_rule']) for x in sub})} nt_override={sorted({x['nt_override'] for x in sub})} "
              f"glno_nt={sorted({x['glno_nt'] for x in sub})} W={sorted({x['glno_w'] for x in sub})} "
              f"caches={sorted({str(x['cache']) for x in sub})}")

    # the 3/35 half vs the seed, over the confined runs
    conf = [x for x in recs if x["confined"]]
    print(f"\nconfined runs (in_above >= 8): {len(conf)} of {len(recs)}")
    print("seed | n runs | out_n min-max | out_hz mean | in_n min-max")
    const = True
    for s in sorted({x["seed"] for x in conf}):
        sub = [x for x in conf if x["seed"] == s]
        lo, hi = min(x["out_n"] for x in sub), max(x["out_n"] for x in sub)
        const &= lo == hi
        print(f"{s:4d} | {len(sub):6d} | {lo}-{hi} | {sum(x['out_hz'] for x in sub) / len(sub):.1f} | "
              f"{min(x['in_n'] for x in sub)}-{max(x['in_n'] for x in sub)}")
    print(f"outside count constant within each seed: {const}; seeds with out_n > 3: "
          f"{sorted({x['seed'] for x in conf if x['out_n'] > 3})}")

    # per point x condition
    print("\npoint x condition: persist(3/35) / boundary / confined / n, per-seed in_n and out_n")
    keys = sorted({(x["gE"], x["gD"]) for x in recs})
    for gE, gD in keys:
        for cfg in sorted({x["config"] for x in recs if (x["gE"], x["gD"]) == (gE, gD)}):
            sub = sorted([x for x in recs if (x["gE"], x["gD"], x["config"]) == (gE, gD, cfg)], key=lambda x: x["seed"])
            print(f"  gE {gE} gD {gD:.0f} {cfg:10s} seeds {'/'.join(str(x['seed']) for x in sub)}: "
                  f"persist {sum(x['persist'] for x in sub)}/{len(sub)} boundary {sum(x['boundary'] for x in sub)} "
                  f"confined {sum(x['confined'] for x in sub)}/{len(sub)}  in {'/'.join(str(x['in_n']) for x in sub)} "
                  f"out {'/'.join(str(x['out_n']) for x in sub)} rho {gD / gE ** 2:.2f}")

    # rates at the named points
    print("\nrates (confined runs only), mean +- sd [min-max]:")
    for pt in a.point:
        gE, gD = (float(v) for v in pt.split(":"))
        for cfg in ("base", "gaba"):
            sub = [x for x in recs if (x["gE"], x["gD"], x["config"]) == (gE, gD, cfg) and x["confined"]]
            if not sub:
                print(f"  gE {gE} gD {gD:.0f} {cfg}: no confined runs")
                continue
            parts = []
            for f in ("bump", "pen", "delta7", "glno", "vs", "rest", "out_hz"):
                m, sd = mean_sd([x[f] for x in sub])
                parts.append(f"{f} {m:.3g}+-{sd:.3g} [{min(x[f] for x in sub):.3g}-{max(x[f] for x in sub):.3g}]")
            print(f"  gE {gE} gD {gD:.0f} {cfg:5s} n={len(sub)} seeds {'/'.join(str(x['seed']) for x in sub)}: " + "; ".join(parts))

    print("\nrows with an explicit receptor model:",
          sorted({(x['config'], str(x['receptor_model']), str(x['net_rule'])) for x in recs if x['receptor_model']}))


if __name__ == "__main__":
    main()
