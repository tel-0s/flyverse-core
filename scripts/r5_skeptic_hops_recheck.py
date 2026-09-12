"""Skeptic recount of the round-5 room take-off batches, with an independent Mann-Whitney."""
import json, math, pathlib
OUT = pathlib.Path(r"D:\Projects\flyverse\out")
def load(n): return json.loads((OUT/n).read_text(encoding="utf-8"))

ARMS = {
 "edited":  ["r5_adopt_sustain_live_1.json","r5_adopt_sustain_live_2.json","r5_adopt_sustain_live_3.json"],
 "pre":     ["r5_sustain_default_live_1.json","r5_sustain_default_live_2.json","r5_sustain_default_live_3.json"],
 "off":     ["r5_sustain_off_live_1.json","r5_sustain_off_live_2.json","r5_sustain_off_live_3.json"],
 "r4def":   ["r4_sustain_default_1.json","r4_sustain_default_2.json","r4_sustain_default_3.json"],
}

def rank(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0]*len(xs); i = 0
    while i < len(idx):
        j = i
        while j+1 < len(idx) and xs[idx[j+1]] == xs[idx[i]]: j += 1
        avg = (i+j)/2.0 + 1.0
        for k in range(i, j+1): r[idx[k]] = avg
        i = j+1
    return r

def mwu(a, b):
    """U for a vs b (U = #(a>b) + 0.5#ties), asymptotic p with tie correction."""
    n1, n2 = len(a), len(b); xs = list(a)+list(b); r = rank(xs)
    R1 = sum(r[:n1]); U1 = R1 - n1*(n1+1)/2.0
    mu = n1*n2/2.0
    from collections import Counter
    N = n1+n2; tie = sum(t**3 - t for t in Counter(xs).values())
    sd = math.sqrt(n1*n2/12.0 * ((N+1) - tie/(N*(N-1))))
    if sd == 0: return U1, 1.0, 1.0, 1.0
    z = (U1 - mu)/sd
    ncdf = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
    return U1, 2*min(ncdf(z), 1-ncdf(z)), ncdf(z), 1-ncdf(z)   # p2, p(a<b), p(a>b)

per = {}
for arm, files in ARMS.items():
    rows = []; hdr = []
    for f in files:
        j = load(f)
        rs = j["rows"]
        h = dict(hops=sum(r["hops"] for r in rs),
                 esc=sum(r.get("hops_escape", 0) for r in rs) if "hops_escape" in rs[0] else None,
                 vol=sum(r.get("hops_voluntary", 0) for r in rs) if "hops_voluntary" in rs[0] else None,
                 fly_s=j.get("fly_s", j["batch"]*j["simulated_s"]), batch=j["batch"], seed=j["brain_seed"], sim=j["simulated_s"],
                 hdr_hops=j.get("hops_total"), hdr_esc=j.get("hops_escape_total"), hdr_vol=j.get("hops_voluntary_total"),
                 gfmed=j.get("gf_max_walk_median_hz"), thr=j.get("rows_gf_at_threshold"),
                 rec=(j.get("receptor") or {}), opts={k: j["options"].get(k) for k in ("gf_hz","escape_gating","energy","program","seeds","receptor_model","receptor_net_rule","device","cuda_sparse","fence","fruit","start","load")},
                 flight=j.get("flight"), file=f, nrows=len(rs))
        hdr.append(h); rows += rs
    per[arm] = (hdr, rows)

for arm, (hdr, rows) in per.items():
    print("="*110); print(arm)
    for h in hdr:
        print(f"  {h['file']:34s} rows={h['nrows']} seed={h['seed']} batch={h['batch']} sim={h['sim']} fly_s={h['fly_s']} "
              f"hops(rows)={h['hops']} hdr={h['hdr_hops']} esc={h['esc']}/{h['hdr_esc']} vol={h['vol']}/{h['hdr_vol']} "
              f"gfmed={h['gfmed']!r} thr={h['thr']}")
        print(f"      receptor={h['rec']} flight={h['flight']}")
        print(f"      opts={h['opts']}")
    tot_h = sum(h["hops"] for h in hdr); fs = sum(h["fly_s"] for h in hdr)
    e = sum(h["esc"] for h in hdr) if hdr[0]["esc"] is not None else None
    v = sum(h["vol"] for h in hdr) if hdr[0]["vol"] is not None else None
    print(f"  TOTAL hops={tot_h} escape={e} voluntary={v} fly_s={fs} -> per 1000 fly-s: "
          f"all {1000*tot_h/fs:.3f}" + (f" escape {1000*e/fs:.3f} voluntary {1000*v/fs:.3f}" if e is not None else ""))
    if e is not None: print(f"  consistency: escape+voluntary = {e+v} vs hops {tot_h} -> {'OK' if e+v==tot_h else 'MISMATCH'}")
    print(f"  per-batch hops {[h['hops'] for h in hdr]} escape {[h['esc'] for h in hdr]} voluntary {[h['vol'] for h in hdr]}")

def col(arm, k): return [r.get(k, 0) for r in per[arm][1]]
def gfcol(arm): return [r.get("gf_max_walk_hz") for r in per[arm][1]]

print("\n" + "="*110)
for pair in (("edited","pre"), ("edited","off"), ("pre","off"), ("edited","r4def")):
    a, b = pair
    print(f"--- {a} vs {b} (n={len(per[a][1])} v {len(per[b][1])}) ---")
    for k in ("hops","hops_escape","hops_voluntary"):
        if k != "hops" and (k not in per[b][1][0]):
            print(f"   {k:15s} not in {b}"); continue
        U, p2, plt_, pgt = mwu(col(a,k), col(b,k))
        print(f"   {k:15s} U={U:7.1f} p2={p2:.4g}  p({a}<{b})={plt_:.4g}  p({a}>{b})={pgt:.4g}  sums {sum(col(a,k))} v {sum(col(b,k))}")
    U, p2, plt_, pgt = mwu(gfcol(a), gfcol(b))
    import statistics as st
    ga, gb = gfcol(a), gfcol(b)
    print(f"   gf_max_walk     U={U:7.1f} p2={p2:.4g}  p({a}<{b})={plt_:.4g} p({a}>{b})={pgt:.4g} "
          f" median {st.median(ga):.2f} v {st.median(gb):.2f}  >=33Hz {sum(x>=33 for x in ga)}/{len(ga)} v {sum(x>=33 for x in gb)}/{len(gb)}")
