"""Skeptic replication report: fresh brain seeds 4/5/6 (env 64-111), five arms, same protocol."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(r"D:\Projects\flyverse"); sys.path.insert(0, str(ROOT))
from flyverse.interp.common import compare
D = ROOT / "out" / "sk_d2_hold"
ARMS = ["shipped", "off", "holdOpticHis", "holdOpticGlu", "holdOpticRandom"]
EXP = {"shipped": 48295, "off": 0, "holdOpticHis": 17256, "holdOpticGlu": 27207, "holdOpticRandom": 3832}
SEEDS = [4, 5, 6]

runs = {}
for a in ARMS:
    runs[a] = []
    for s in SEEDS:
        p = D / f"{a}_{s}.json"
        d = json.loads(p.read_text(encoding="utf-8")); d["_f"] = p.name
        d["_md5"] = hashlib.md5(p.read_bytes()).hexdigest()
        runs[a].append(d)

print("== integrity (15 fresh runs) ==")
lifref = None; prob = 0
for a in ARMS:
    for d in runs[a]:
        e = []
        if d["device"] != "cuda": e.append("device " + str(d["device"]))
        if d["receptor"]["fast_sign_changed_entries"] != EXP[a]: e.append("entries %s" % d["receptor"]["fast_sign_changed_entries"])
        if d["frames"] != 30000 or d["batch"] != 16 or abs(d["flight"]["gf_hz"] - 33) > 1e-9: e.append("protocol")
        if [r["environment_seed"] for r in d["rows"]] != list(range(d["brain_seed"] * 16, d["brain_seed"] * 16 + 16)): e.append("env seeds")
        for r in d["rows"]:
            if r["hops"] != r["hops_escape"] + r["hops_voluntary"]: e.append("route split")
        lif = {k: v for k, v in d["lif"].items() if not k.startswith("receptor")}
        if lifref is None: lifref = lif
        elif lif != lifref: e.append("LIF")
        if abs(d["cache_sum_abs_W"] - 121460584.0) > 1: e.append("cache")
        if e: prob += 1; print("  PROBLEM", d["_f"], e)
print("  problems", prob, "of 15")

print()
print("== per run and pooled (3 runs x 16 flies x 300 s = 14,400 fly-s per arm) ==")
pool = {}
for a in ARMS:
    rows = [r for d in runs[a] for r in d["rows"]]
    h = sum(r["hops"] for r in rows); e = sum(r["hops_escape"] for r in rows); v = sum(r["hops_voluntary"] for r in rows)
    gf = np.array([r["gf_max_walk_hz"] for r in rows])
    pool[a] = dict(h=h, e=e, v=v, gf=gf, ge33=int((gf >= 33).sum()))
    per = " / ".join(f"{d['hops_total']}({d['hops_escape_total']}+{d['hops_voluntary_total']})" for d in runs[a])
    gfs = " / ".join(f"{d['gf_max_walk_median_hz']:.2f}" for d in runs[a])
    print(f"  {a:16s} {per:28s} | pooled {h:3d} = {e:3d} + {v:3d} | GF med {np.median(gf):6.2f} ({gfs}) | >=33 {pool[a]['ge33']:2d}/48")

def jeff(ka, kb):
    n = ka + kb
    if n == 0: return float('nan'), float('nan'), float('nan')
    lo, hi = stats.beta.ppf([0.025, 0.975], ka + .5, kb + .5)
    r = lambda p: p / (1 - p) if p < 1 else float("inf")
    return r(lo), r(hi), stats.binomtest(ka, n, 0.5).pvalue

print()
print("== ratios (fresh seeds) ==")
for a in ["holdOpticHis", "holdOpticGlu", "holdOpticRandom", "off"]:
    for m, k in [("hops", "h"), ("escape", "e"), ("voluntary", "v")]:
        ka, ks, ko = pool[a][k], pool["shipped"][k], pool["off"][k]
        lo, hi, p = jeff(ka, ks); _, _, p2 = jeff(ka, ko)
        share = (ka - ko) / (ks - ko) * 100 if ks != ko else float("nan")
        print(f"  {a:16s} {m:10s} {ka:3d} | vs shipped {ks:3d}: {ka/ks if ks else float('nan'):.3f}x [{lo:.3f},{hi:.3f}] p {p:.3g} | vs off {ko:3d} p {p2:.3g} | share {share:6.1f} %")
    print(f"  {a:16s} gf med {np.median(pool[a]['gf']):.2f} (off {np.median(pool['off']['gf']):.2f} shipped {np.median(pool['shipped']['gf']):.2f}); >=33 {pool[a]['ge33']}/48")

print()
print("== compare over the 3 fresh runs (p_floor 3v3 = 0.10 -> 'underpowered' by design; z and draws are the reading) ==")
def rv(a, key): return [round(sum(r[key] for r in d["rows"]) / 4.8, 2) for d in runs[a]]
for a, b in [("holdOpticHis", "shipped"), ("holdOpticHis", "off"), ("holdOpticGlu", "off"), ("holdOpticGlu", "shipped"),
             ("holdOpticRandom", "off"), ("holdOpticRandom", "shipped"), ("off", "shipped")]:
    for key in ["hops", "hops_escape", "hops_voluntary"]:
        c = compare(rv(a, key), rv(b, key))
        print(f"  {a:16s} vs {b:16s} {key:15s} a {rv(a,key)} b {rv(b,key)} diff {c['diff']:+7.3f} z {c['z']:+7.2f} U {c['U']:4.1f} p {c['p']:.3g} -> {c['verdict']}")
    A = [round(d["gf_max_walk_median_hz"], 2) for d in runs[a]]; B = [round(d["gf_max_walk_median_hz"], 2) for d in runs[b]]
    c = compare(A, B)
    print(f"  {a:16s} vs {b:16s} {'gf_median':15s} a {A} b {B} diff {c['diff']:+7.3f} z {c['z']:+7.2f} U {c['U']:4.1f} p {c['p']:.3g} -> {c['verdict']}")

print()
print("== POOLED with the reported batch: 7 runs per arm, 33,600 fly-s ==")
D0 = ROOT / "out" / "d2_hold"
allruns = {a: [json.loads((D0 / f"{a}_{s}.json").read_text(encoding="utf-8")) for s in range(4)] +
              [json.loads((D / f"{a}_{s}.json").read_text(encoding="utf-8")) for s in SEEDS] for a in ARMS}
def rva(a, key): return [round(sum(r[key] for r in d["rows"]) / 4.8, 2) for d in allruns[a]]
tot = {}
for a in ARMS:
    rows = [r for d in allruns[a] for r in d["rows"]]
    tot[a] = dict(h=sum(r["hops"] for r in rows), e=sum(r["hops_escape"] for r in rows), v=sum(r["hops_voluntary"] for r in rows),
                  gf=np.median([r["gf_max_walk_hz"] for r in rows]), ge33=int(sum(1 for r in rows if r["gf_max_walk_hz"] >= 33)), n=len(rows))
    print(f"  {a:16s} hops {tot[a]['h']:3d} = {tot[a]['e']:3d} + {tot[a]['v']:3d} over {len(rows)*300:,} fly-s | GF med {tot[a]['gf']:.2f} | >=33 {tot[a]['ge33']}/{len(rows)}")
print()
for a, b in [("holdOpticHis", "shipped"), ("holdOpticHis", "off"), ("holdOpticGlu", "off"), ("holdOpticGlu", "shipped"),
             ("holdOpticRandom", "off"), ("off", "shipped")]:
    for key in ["hops", "hops_escape", "hops_voluntary"]:
        c = compare(rva(a, key), rva(b, key))
        print(f"  7v7 {a:16s} vs {b:16s} {key:15s} diff {c['diff']:+7.3f} z {c['z']:+7.2f} U {c['U']:5.1f} p {c['p']:.4g} p_floor {c['p_floor']:.4g} -> {c['verdict']}")
    A = [round(d["gf_max_walk_median_hz"], 2) for d in allruns[a]]; B = [round(d["gf_max_walk_median_hz"], 2) for d in allruns[b]]
    c = compare(A, B)
    print(f"  7v7 {a:16s} vs {b:16s} {'gf_median':15s} diff {c['diff']:+7.3f} z {c['z']:+7.2f} U {c['U']:5.1f} p {c['p']:.4g} -> {c['verdict']}")
    ka, kb = tot[a]["h"], tot[b]["h"]; lo, hi, p = jeff(ka, kb)
    print(f"      pooled hops ratio {ka}/{kb} = {ka/kb:.3f}x [{lo:.3f}, {hi:.3f}] p {p:.3g}")
