"""Round-4 Brain-side isolation tables: the shipped receptor table with one group of glutamate flips held at NT_SIGN.

    python scripts/build_hold_tables.py                # writes out/receptors_holdKC.csv and out/receptors_holdDN1.csv
    python scripts/build_hold_tables.py --verify       # + counts the entries each table changes vs sign(W.data) (CPU, local cache)

Each table is flyverse/data/receptors_by_type.csv (its '#' header kept) with the glutamate rows of one type group set to
fast_sign_abs = -1, fast_gain_class_abs = none, fast_net_abs = 'held' -- i.e. those targets keep the presynaptic sign
under receptor_model='sign' / net_rule='abs' while every other entry of the default stays. connectome.receptor_signs reads
only the fast_sign_abs / fast_gain_class_abs columns, so 'held' in fast_net_abs is a label for the record.

Groups (docs/audits/receptor_verification.md, round-3 critic follow-up 2): KC = KCg-m, KCg-s1..s4, KCa'b'-ap1, KCa'b'-ap2,
KCg (GluRIB-led +1 rows, tier fuzzy, fca2022); DN1 = DN1a, DN1pA, DN1pB (clock, davie2018).
The files are written atomically (temp + os.replace) so concurrent cluster jobs can regenerate them in one run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLE = os.path.join(ROOT, "flyverse", "data", "receptors_by_type.csv")
GROUPS = {
    "KC": ["KCg-m", "KCg-s1", "KCg-s2", "KCg-s3", "KCg-s4", "KCa'b'-ap1", "KCa'b'-ap2", "KCg"],
    "DN1": ["DN1a", "DN1pA", "DN1pB"],
}


def hold_table(group: str) -> tuple[str, pd.DataFrame, int]:
    header = "".join(l for l in open(TABLE, encoding="utf-8") if l.startswith("#"))
    t = pd.read_csv(TABLE, comment="#")
    sel = t.malecns_type.isin(GROUPS[group]) & (t.transmitter == "glutamate")
    t.loc[sel, "fast_sign_abs"] = -1
    t.loc[sel, "fast_gain_class_abs"] = "none"
    t.loc[sel, "fast_net_abs"] = "held"
    header += f"# scripts/build_hold_tables.py: {int(sel.sum())} glutamate rows of {group} ({', '.join(GROUPS[group])}) held at NT_SIGN (fast_sign_abs -1, gain none, fast_net_abs held)\n"
    return header, t, int(sel.sum())


def write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--verify", action="store_true", help="count changed entries with connectome.receptor_signs (loads the local cache)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {}
    for g in GROUPS:
        header, t, n = hold_table(g)
        p = os.path.join(a.out_dir, f"receptors_hold{g}.csv")
        write_atomic(p, header + t.to_csv(index=False, lineterminator="\n"))
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        print(f"{p}: {n} rows held; {len(t)} rows; md5 {md5}")
        paths[g] = p
    if a.verify:
        import numpy as np
        sys.path.insert(0, ROOT)
        from flyverse import connectome as cn
        c = cn.load(verbose=False)
        base = np.sign(c.W.data)
        ref = cn.receptor_signs(c, net_rule="abs")
        d_ref = ref.fast_sign != base
        print(f"shipped table: {int(d_ref.sum())} entries changed vs sign(W.data), |W| {float(np.abs(c.W.data[d_ref]).sum()):.0f}")
        post = c.W.tocoo().row
        types = c.neurons.type.to_numpy()
        for g, p in paths.items():
            r = cn.receptor_signs(c, table_path=p, net_rule="abs")
            d = r.fast_sign != base
            held = d_ref & ~d                                  # entries the default changes and this table does not
            extra = d & ~d_ref
            ht = pd.Series(types[post[held]]).value_counts()
            print(f"hold{g}: {int(d.sum())} entries changed (default - {int(held.sum())}; new {int(extra.sum())}); held |W| "
                  f"{float(np.abs(c.W.data[held]).sum()):.0f}; held entries by postsynaptic type: {ht.to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
