"""Round-4 skeptic check of the hold-table attribution (CPU, local cache; no simulation).

    PYTHONIOENCODING=utf-8 python scripts/skeptic_r4_rescore_check.py

For the shipped table and each of out/receptors_hold{KC,DN1}.csv it prints, from connectome.receptor_signs
(net_rule='abs') against sign(c.W.data): the number of changed entries and their |W|, the entries the default
changes that the hold table does not (with the postsynaptic-type split in entries and |W|), any entry the hold
table changes that the default does not, and -- the point the round-3 record's 'KCg-m 3,906 syn' turns on -- the
entries the default still changes onto the held types after the hold (i.e. whether the hold is complete for its
own target group). Verifies the tables' md5 as read.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from flyverse import connectome as cn  # noqa: E402

GROUPS = {
    "KC": ["KCg-m", "KCg-s1", "KCg-s2", "KCg-s3", "KCg-s4", "KCa'b'-ap1", "KCa'b'-ap2", "KCg"],
    "DN1": ["DN1a", "DN1pA", "DN1pB"],
}


def main() -> int:
    c = cn.load(verbose=False)
    base = np.sign(c.W.data)
    absW = np.abs(c.W.data)
    coo = c.W.tocoo()
    post, pre = coo.row, coo.col
    types = c.neurons.type.to_numpy()
    post_t, pre_t = types[post], types[pre]
    print(f"sum|W| {absW.sum():.0f}; entries {c.W.data.size}")

    ref = cn.receptor_signs(c, net_rule="abs")
    d_ref = ref.fast_sign != base
    print(f"shipped table {os.path.basename(ref.table_path)}: changed {int(d_ref.sum())} entries, |W| {absW[d_ref].sum():.0f}")

    for g, names in GROUPS.items():
        p = os.path.join(ROOT, "out", f"receptors_hold{g}.csv")
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        r = cn.receptor_signs(c, table_path=p, net_rule="abs")
        d = r.fast_sign != base
        held = d_ref & ~d
        extra = d & ~d_ref
        print(f"\nhold{g} ({os.path.basename(p)} md5 {md5}): changed {int(d.sum())} entries "
              f"(default {int(d_ref.sum())} - {int(held.sum())} held + {int(extra.sum())} new); held |W| {absW[held].sum():.0f}")
        ent = pd.Series(post_t[held]).value_counts()
        syn = pd.Series(absW[held]).groupby(pd.Series(post_t[held])).sum().sort_values(ascending=False)
        print("  held entries by postsynaptic type: " + ", ".join(f"{k} {v}" for k, v in ent.items()))
        print("  held |W| by postsynaptic type:     " + ", ".join(f"{k} {v:.0f}" for k, v in syn.items()))
        # group-level completeness: what the default changes onto the group's types, and what survives the hold
        in_g = np.isin(post_t, names)
        for label, mask in (("default changes onto the group", d_ref & in_g), ("hold changes onto the group (leftover)", d & in_g)):
            s = pd.Series(absW[mask]).groupby(pd.Series(post_t[mask])).sum().sort_values(ascending=False)
            e = pd.Series(post_t[mask]).value_counts()
            print(f"  {label}: {int(mask.sum())} entries, |W| {absW[mask].sum():.0f}"
                  + ("; " + ", ".join(f"{k} {e[k]}e/{s[k]:.0f}syn" for k in s.index) if int(mask.sum()) else ""))
        if int((d & in_g).sum()):
            left = d & in_g
            pt = pd.Series(pre_t[left]).value_counts().head(12)
            print("    leftover by presynaptic type: " + ", ".join(f"{k} {v}" for k, v in pt.items()))
            print("    leftover sign change: " + ", ".join(
                f"{int(b)}->{int(a)} x{int(((base[left] == b) & (r.fast_sign[left] == a)).sum())}"
                for b in (-1.0, 1.0) for a in (-1.0, 0.0, 1.0)
                if int(((base[left] == b) & (r.fast_sign[left] == a)).sum())))
    # ---- where the rest of the default's change lands: the residual the two hold tables do NOT cover ----------
    scl = c.neurons.superclass.fillna("").to_numpy()[post]
    sp = pd.Series(absW[d_ref]).groupby(pd.Series(scl[d_ref])).sum().sort_values(ascending=False)
    se = pd.Series(scl[d_ref]).value_counts()
    print("\ndefault changed entries by postsynaptic superclass: "
          + ", ".join(f"{k} {se[k]}e/{sp[k]:.0f}syn" for k in sp.index))
    non_ol = d_ref & (scl != "ol_intrinsic")
    resid = non_ol & ~np.isin(post_t, GROUPS["KC"] + GROUPS["DN1"])
    print(f"non-ol_intrinsic changed: {int(non_ol.sum())} entries / {absW[non_ol].sum():.0f} syn; "
          f"residual after holding KC and DN1: {int(resid.sum())} entries / {absW[resid].sum():.0f} syn")
    e = pd.Series(post_t[resid]).value_counts()
    s = pd.Series(absW[resid]).groupby(pd.Series(post_t[resid])).sum().sort_values(ascending=False)
    print("  residual by postsynaptic type: " + ", ".join(f"{k} {e[k]}e/{s[k]:.0f}syn" for k in s.index[:20]))
    pe = pd.Series(pre_t[resid]).value_counts()
    print("  residual by presynaptic type: " + ", ".join(f"{k} {v}e" for k, v in pe.head(12).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
