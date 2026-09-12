"""Skeptic: my independent seed-0 rerun of the edited default's room batch vs the adopt task's seed-0 batch,
and the pooled contrast with my batch substituted / added."""
import json, pathlib
from scipy.stats import mannwhitneyu
import numpy as np
OUT = pathlib.Path(r"D:\Projects\flyverse\out")
L = lambda n: json.loads((OUT/n).read_text(encoding="utf-8"))

mine = L("r5_skeptic_sustain_live_4.json"); theirs = L("r5_adopt_sustain_live_1.json")
for nm, j in (("skeptic rerun (seed 0)", mine), ("adopt batch 1 (seed 0)", theirs)):
    o = j["options"]
    print(f"{nm}: seed={j['brain_seed']} env={o['seeds']} fly_s={j['fly_s']} gf_hz_opt={o['gf_hz']} energy={o['energy']} "
          f"program={o['program']} receptor={j['receptor']} flight={j['flight']}")
    print(f"   hops {j['hops_total']} = escape {j['hops_escape_total']} + voluntary {j['hops_voluntary_total']}; "
          f"per 1000 fly-s {1000*j['hops_total']/j['fly_s']:.2f} / {j['hops_escape_per_1000_fly_s']:.2f} / {j['hops_voluntary_per_1000_fly_s']:.2f}; "
          f"GF median {j['gf_max_walk_median_hz']:.2f}, rows>=33 {j['rows_gf_at_threshold']}/16  wall {j['wall_s']:.0f}s")
    print(f"   hops per fly {[r['hops'] for r in j['rows']]}")
    print(f"   escape      {[r['hops_escape'] for r in j['rows']]}")
    print(f"   voluntary   {[r['hops_voluntary'] for r in j['rows']]}")

print("\n--- identical-seed rerun scatter, edited default, seed 0 ---")
for k in ("hops", "hops_escape", "hops_voluntary"):
    a = [r[k] for r in mine["rows"]]; b = [r[k] for r in theirs["rows"]]
    u = mannwhitneyu(a, b, alternative="two-sided")
    print(f"  {k:15s} mine {sum(a):3d}  theirs {sum(b):3d}   U {u.statistic:6.1f} p2 {u.pvalue:.3f}")
ga = [r["gf_max_walk_hz"] for r in mine["rows"]]; gb = [r["gf_max_walk_hz"] for r in theirs["rows"]]
u = mannwhitneyu(ga, gb, alternative="two-sided")
print(f"  gf_max_walk     mine median {np.median(ga):.2f} ({sum(x>=33 for x in ga)}/16)  theirs {np.median(gb):.2f} ({sum(x>=33 for x in gb)}/16)  U {u.statistic:.1f} p2 {u.pvalue:.3f}")

rows = lambda fs: [r for f in fs for r in L(f)["rows"]]
E3 = rows([f"r5_adopt_sustain_live_{i}.json" for i in (1, 2, 3)])
PRE = rows([f"r5_sustain_default_live_{i}.json" for i in (1, 2, 3)])
OFFr = rows([f"r5_sustain_off_live_{i}.json" for i in (1, 2, 3)])
M = mine["rows"]
E_sub = M + rows([f"r5_adopt_sustain_live_{i}.json" for i in (2, 3)])          # my batch REPLACING their seed-0 batch
E_add = E3 + M                                                                  # my batch ADDED (64 flies)

print("\n--- how much the 27-32 % 'shrink' moves when one batch is swapped ---")
for lbl, E in (("as reported (3 batches, 56 hops)", E3), ("seed-0 batch = my rerun (48 hops)", E_sub), ("4 batches (67 hops)", E_add)):
    for k, off_tot in (("hops", 9), ("hops_escape", 9), ("hops_voluntary", 0)):
        e = sum(r[k] for r in E); p = sum(r[k] for r in PRE)
        fe = len(E) * 300.0; fp = len(PRE) * 300.0
        re_, rp, ro = 1000*e/fe, 1000*p/fp, 1000*off_tot/(len(OFFr)*300.0)
        shrink = 1 - (re_-ro)/(rp-ro) if rp != ro else float("nan")
        u = mannwhitneyu([r[k] for r in E], [r[k] for r in PRE], alternative="two-sided")
        ug = mannwhitneyu([r[k] for r in E], [r[k] for r in PRE], alternative="greater")
        print(f"  {lbl:36s} {k:15s} edited {re_:.2f} vs pre {rp:.2f} vs off {ro:.2f} per 1000 -> "
              f"excess removed {100*shrink:5.1f} %   U {u.statistic:7.1f} p2 {u.pvalue:.3f}  p(edited>pre) {ug.pvalue:.3f}")
    print()
