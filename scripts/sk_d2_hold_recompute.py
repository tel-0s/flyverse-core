"""Independent skeptic recomputation of the d2_hold take-off class split (does NOT import the report script)."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(r"D:\Projects\flyverse")
sys.path.insert(0, str(ROOT))
D = ROOT / "out" / "d2_hold"
ARMS = ["shipped", "off", "holdOpticHis", "holdOpticGlu", "holdOpticRandom"]
EXP = {"shipped": 48295, "off": 0, "holdOpticHis": 17256, "holdOpticGlu": 27207, "holdOpticRandom": 3832}
SEEDS = [0, 1, 2, 3]

runs = {}
for a in ARMS:
    runs[a] = []
    for s in SEEDS:
        p = D / f"{a}_{s}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_md5"] = hashlib.md5(p.read_bytes()).hexdigest()
        d["_f"] = p.name
        runs[a].append(d)

print("== per-run integrity ==")
prob = 0
lifref = None
for a in ARMS:
    for d in runs[a]:
        errs = []
        if d["device"] != "cuda": errs.append("device=" + str(d["device"]))
        if d["receptor"]["fast_sign_changed_entries"] != EXP[a]:
            errs.append("entries=%s" % d["receptor"]["fast_sign_changed_entries"])
        if d["frames"] != 30000: errs.append("frames")
        if d["batch"] != 16: errs.append("batch")
        if abs(d["flight"]["gf_hz"] - 33.0) > 1e-9: errs.append("gf_hz")
        if abs(d["fly_s"] - 4800.0) > 1e-6: errs.append("fly_s")
        if len(d["rows"]) != 16: errs.append("nrows")
        # hops == escape + voluntary per row and totals
        h = sum(r["hops"] for r in d["rows"]); e = sum(r["hops_escape"] for r in d["rows"]); v = sum(r["hops_voluntary"] for r in d["rows"])
        if h != e + v: errs.append("row hops != e+v")
        if (h, e, v) != (d["hops_total"], d["hops_escape_total"], d["hops_voluntary_total"]): errs.append("totals mismatch")
        # per-row escape+voluntary
        for r in d["rows"]:
            if r["hops"] != r["hops_escape"] + r["hops_voluntary"]: errs.append("row%d" % r["row"])
        # env seeds
        exp_seeds = list(range(d["brain_seed"] * 16, d["brain_seed"] * 16 + 16))
        if [r["environment_seed"] for r in d["rows"]] != exp_seeds: errs.append("env seeds %s" % [r["environment_seed"] for r in d["rows"]][:3])
        if d["brain_seed"] != int(d["_f"].split("_")[-1].split(".")[0]): errs.append("brain_seed")
        # GF median / rows>=33 recomputed
        gf = np.array([r["gf_max_walk_hz"] for r in d["rows"]])
        if abs(np.median(gf) - d["gf_max_walk_median_hz"]) > 1e-6: errs.append("gf median")
        if int((gf >= 33.0).sum()) != d["rows_gf_at_threshold"]: errs.append("rows_ge_33")
        # LIF identical apart from receptor fields
        lif = {k: v for k, v in d["lif"].items() if not k.startswith("receptor")}
        if lifref is None: lifref = lif
        elif lif != lifref: errs.append("LIF differs: %s" % [k for k in lif if lif[k] != lifref.get(k)])
        # cache fingerprint
        if abs(d["cache_sum_abs_W"] - 121460584.0) > 1: errs.append("cache")
        if errs: prob += 1; print("  PROBLEM", d["_f"], errs)
print("  problems:", prob, "of 20")

print()
print("== pooled per arm (independent) ==")
pool = {}
for a in ARMS:
    rows = [r for d in runs[a] for r in d["rows"]]
    h = sum(r["hops"] for r in rows); e = sum(r["hops_escape"] for r in rows); v = sum(r["hops_voluntary"] for r in rows)
    gf = np.array([r["gf_max_walk_hz"] for r in rows])
    pool[a] = dict(h=h, e=e, v=v, gf=gf, n=len(rows), ge33=int((gf >= 33).sum()),
                   minE=float(np.median([r["min_energy"] for r in rows])),
                   rows_with_escape=int(sum(1 for r in rows if r["hops_escape"] > 0)))
    print(f"  {a:16s} hops {h:3d} = esc {e:3d} + vol {v:3d} | GF med {np.median(gf):6.2f} | >=33 {pool[a]['ge33']:2d}/64 | "
          f"rate/1000 {h/19.2:.3f} | minE med {pool[a]['minE']:.3f} | rows w/ esc {pool[a]['rows_with_escape']}")

def jeff(ka, kb):
    n = ka + kb
    lo, hi = stats.beta.ppf([0.025, 0.975], ka + .5, kb + .5)
    r = lambda p: p / (1 - p) if p < 1 else float("inf")
    return r(lo), r(hi), stats.binomtest(ka, n, 0.5).pvalue if n else float("nan")

def pois(k, exp):
    lo = 0.0 if k == 0 else stats.chi2.ppf(0.025, 2 * k) / 2
    hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return lo / exp, hi / exp

print()
print("== ratios vs shipped / off (independent) ==")
for a in ["holdOpticHis", "holdOpticGlu", "holdOpticRandom", "off"]:
    for m, key in [("hops", "h"), ("escape", "e"), ("voluntary", "v")]:
        ka = pool[a][key]; ks = pool["shipped"][key]; ko = pool["off"][key]
        lo, hi, p = jeff(ka, ks)
        lo2, hi2, p2 = jeff(ka, ko)
        share = (ka - ko) / (ks - ko) * 100 if ks != ko else float("nan")
        print(f"  {a:16s} {m:10s} {ka:3d} | vs shipped {ks:3d}: {ka/ks if ks else float('nan'):.3f}x [{lo:.3f},{hi:.3f}] p {p:.3g}"
              f" | vs off {ko:3d}: p {p2:.3g} | share {share:6.1f} %")
    gfmed = np.median(pool[a]["gf"]); o = np.median(pool["off"]["gf"]); s = np.median(pool["shipped"]["gf"])
    print(f"  {a:16s} gf med {gfmed:.2f} (off {o:.2f} shipped {s:.2f}) -> {(gfmed-o)/(s-o)*100:.1f} % of shift; >=33 {pool[a]['ge33']}/64")

print()
print("== poisson CIs per 1000 fly-s ==")
for a in ARMS:
    for m, key in [("hops", "h"), ("escape", "e"), ("voluntary", "v")]:
        lo, hi = pois(pool[a][key], 19.2)
        print(f"  {a:16s} {m:10s} {pool[a][key]/19.2:.3f} [{lo:.3f},{hi:.3f}]")

print()
print("== additivity ==")
for m, key in [("hops", "h"), ("escape", "e"), ("voluntary", "v")]:
    o = pool["off"][key]
    print(f"  {m:10s} His excess {pool['holdOpticHis'][key]-o} + Glu excess {pool['holdOpticGlu'][key]-o} = "
          f"{pool['holdOpticHis'][key]+pool['holdOpticGlu'][key]-2*o} vs shipped excess {pool['shipped'][key]-o}")

print()
print("== compare (flyverse.interp.common) over the 4 runs, RUN as replicate ==")
from flyverse.interp.common import compare
def runvals(a, key):
    return [round(sum(r[key] for r in d["rows"]) / 4.8, 2) for d in runs[a]]
pairs = [("holdOpticHis", "shipped"), ("holdOpticHis", "off"), ("holdOpticGlu", "off"), ("holdOpticGlu", "shipped"),
         ("holdOpticRandom", "off"), ("holdOpticRandom", "shipped"), ("off", "shipped"),
         ("holdOpticHis", "holdOpticRandom")]
for a, b in pairs:
    for key in ["hops", "hops_escape", "hops_voluntary"]:
        A = runvals(a, key); B = runvals(b, key)
        c = compare(A, B)
        print(f"  {a:16s} vs {b:16s} {key:15s} a {A} b {B} -> {c['verdict']} p {c['p']:.4g} z {round(c['z'],2)} U {c['U']} p_floor {c['p_floor']:.4g}")
    A = [round(d["gf_max_walk_median_hz"], 2) for d in runs[a]]; B = [round(d["gf_max_walk_median_hz"], 2) for d in runs[b]]
    c = compare(A, B)
    print(f"  {a:16s} vs {b:16s} {'gf_median':15s} a {A} b {B} -> {c['verdict']} p {c['p']:.4g} z {round(c['z'],2)} U {c['U']} p_floor {c['p_floor']:.4g}")
    A = [float(d["rows_gf_at_threshold"]) for d in runs[a]]; B = [float(d["rows_gf_at_threshold"]) for d in runs[b]]
    c = compare(A, B)
    print(f"  {a:16s} vs {b:16s} {'rows_ge_33':15s} a {A} b {B} -> {c['verdict']} p {c['p']:.4g} z {round(c['z'],2)} U {c['U']} p_floor {c['p_floor']:.4g}")

print()
print("== Mann-Whitney over 64 rows (selected) ==")
for a, b in [("holdOpticRandom", "off"), ("holdOpticHis", "off"), ("holdOpticHis", "shipped"), ("holdOpticGlu", "off")]:
    ra = [r for d in runs[a] for r in d["rows"]]; rb = [r for d in runs[b] for r in d["rows"]]
    for key in ["hops", "hops_escape", "hops_voluntary", "gf_max_walk_hz"]:
        u = stats.mannwhitneyu([r[key] for r in ra], [r[key] for r in rb], alternative="two-sided")
        print(f"  {a:16s} vs {b:16s} {key:15s} U {u.statistic:.1f} p {u.pvalue:.3g}")

print()
print("== md5 of fetched JSONs ==")
for a in ARMS:
    for d in runs[a]:
        print("  %s %s" % (d["_md5"], d["_f"]))
