"""Round-3 structural coverage of the object-sweep types under the CURRENT receptors_by_type.csv (CPU).

    PYTHONIOENCODING=utf-8 python out/r3obj/coverage_r3.py out/r3obj/coverage_r3.json

Per type: cells, |W| input synapses, and the input synapses whose fast sign the 'sign' model changes under the
default net rule 'abs' (flipped / silenced), plus the global change counts.  Same measurement as round 2's
out/obj/obj_coverage.py, on the round-3 contested-flip table.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Projects\flyverse")
from flyverse import connectome  # noqa: E402

TYPES = ["LC11", "LC10a", "LC10b", "LC16", "LPLC2", "LC4", "T2", "T3", "Tm5Y", "TmY21", "TmY13", "TmY5a", "Mi4", "Tm3", "Mi1"]
c = connectome.load(verbose=False)
W = c.W.tocsr(); absW = np.abs(W.data)
types = c.neurons.type.fillna("").to_numpy()
base = np.sign(W.data)
post = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
out = {"total_cells": int(c.n), "total_absW": float(absW.sum()), "nnz": int(W.nnz), "types": {}}
print(f"cells {c.n:,} entries {W.nnz:,} sum|W| {absW.sum():,.0f}", flush=True)
for rule in ("abs", "class"):
    rs = connectome.receptor_signs(c, net_rule=rule)
    ch = rs.fast_sign != base
    out[f"global_{rule}"] = {"changed_entries": int(ch.sum()), "changed_syn": float(absW[ch].sum()),
                             "flipped_entries": int((rs.fast_sign * base < 0).sum()),
                             "silenced_entries": int(((rs.fast_sign == 0) & (base != 0)).sum()),
                             "matched_entries": int(rs.matched.sum())}
    print(rule, out[f"global_{rule}"], flush=True)
    if rule != "abs":
        continue
    for t in TYPES:
        cells = np.flatnonzero(types == t)
        m = np.isin(post, cells); tot = float(absW[m].sum())
        chm = m & ch
        pre = W.indices[chm]
        s = pd.Series(absW[chm]).groupby(types[pre]).sum().sort_values(ascending=False) if chm.any() else pd.Series(dtype=float)
        r = {"cells": int(len(cells)), "in_syn_absW": tot, "entries": int(m.sum()),
             "abs_changed_syn": float(absW[chm].sum()), "abs_changed_syn_frac": float(absW[chm].sum() / tot) if tot else 0.0,
             "abs_flipped_syn": float(absW[m & (rs.fast_sign * base < 0)].sum()),
             "abs_silenced_syn": float(absW[m & (rs.fast_sign == 0) & (base != 0)].sum()),
             "abs_changed_top_pre": {str(k): float(v) for k, v in s.head(5).items()}}
        out["types"][t] = r
        print(t, {k: v for k, v in r.items() if k != "abs_changed_top_pre"}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1])
