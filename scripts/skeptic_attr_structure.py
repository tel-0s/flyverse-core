"""Skeptic structural recheck of the round-5 Brain-side / optic-side hold tables (task key: attribute).

    PYTHONIOENCODING=utf-8 python scripts/skeptic_attr_structure.py

Independent of scripts/build_hold_tables.py --verify: recomputes the entry / |W| partition in int64 (the
reported row-sum arithmetic is float32 and cannot resolve single synapses), splits the changed entries into
glutamate flips vs histamine silencings, reports the transmitter composition of the held ROWS of each table,
and rebuilds all six tables twice to check the md5s are stable.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from flyverse import connectome as cn                                   # noqa: E402
import build_hold_tables as bht                                        # noqa: E402

GROUPS = ["Brain", "Optic", "BrainGlu", "BrainHis", "KC", "DN1"]


def main() -> int:
    scratch = os.path.join(ROOT, "out", "sk_tables")
    os.makedirs(scratch, exist_ok=True)
    md5 = {}
    for run in (1, 2):
        for g in GROUPS:
            header, t, n = bht.hold_table(g)
            text = header + t.to_csv(index=False, lineterminator="\n")
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            md5.setdefault(g, []).append((n, h))
            if run == 1:
                bht.write_atomic(os.path.join(scratch, f"receptors_hold{g}.csv"), text)
    print("-- builder determinism (two in-process rebuilds; md5 of the exact bytes written) --")
    for g, v in md5.items():
        same = "SAME" if v[0] == v[1] else "DIFFER"
        print(f"  hold{g:9s} rows held {v[0][0]:4d}  md5 {v[0][1]}  {same}")
    for g in GROUPS:
        p = os.path.join(ROOT, "out", f"receptors_hold{g}.csv")
        if os.path.isfile(p):
            h = hashlib.md5(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
            print(f"  reported out/receptors_hold{g}.csv md5 {h} "
                  f"{'== rebuild' if h == md5[g][0][1] else '!= rebuild'}")

    print("\n-- transmitter composition of the HELD rows --")
    t0 = pd.read_csv(bht.TABLE, comment="#")
    prior = t0.transmitter.map(bht.NT_SIGN).astype(float)
    optic = t0.superclass.isin(bht.OPTIC_SUPERCLASSES)
    synthetic = t0.malecns_type.astype(str).str.startswith("<")
    diff = t0.fast_sign_abs.astype(float) != prior
    for name, side in (("Brain(non-optic)", ~optic & ~synthetic), ("Optic", optic),
                       ("synthetic(<...)", synthetic & ~optic)):
        sel = side & diff
        print(f"  {name:18s} rows held {int(sel.sum()):4d}  by transmitter "
              f"{t0.loc[sel].groupby('transmitter').size().to_dict()}")

    print("\n-- entry / |W| partition, int64 --")
    c = cn.load(verbose=False)
    base = np.sign(c.W.data)
    wabs = np.abs(c.W.data)
    wint = np.rint(wabs).astype(np.int64)
    print(f"  connectome nnz {c.W.nnz:,}  sum|W| float {float(wabs.sum()):,.0f}  sum|W| int64 {int(wint.sum()):,}")
    post = c.W.tocoo().row
    sc = c.neurons.superclass.fillna("").to_numpy()
    is_optic_post = np.isin(sc[post], list(bht.OPTIC_SUPERCLASSES))
    tabs = {"default": None}
    for g in GROUPS:
        tabs[f"hold{g}"] = os.path.join(scratch, f"receptors_hold{g}.csv")
    for name, path in tabs.items():
        r = cn.receptor_signs(c, table_path=path, net_rule="abs")
        d = r.fast_sign != base
        sil = d & (r.fast_sign == 0)
        flip = d & (r.fast_sign != 0)
        print(f"  {name:13s} changed {int(d.sum()):6d} entries / |W| {int(wint[d].sum()):7,}"
              f" | silenced {int(sil.sum()):6d} / {int(wint[sil].sum()):7,}"
              f" | sign-flips {int(flip.sum()):6d} / {int(wint[flip].sum()):7,}"
              f" | optic-post {int((d & is_optic_post).sum()):6d} / {int(wint[d & is_optic_post].sum()):7,}"
              f" | other-post {int((d & ~is_optic_post).sum()):6d} / {int(wint[d & ~is_optic_post].sum()):7,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
