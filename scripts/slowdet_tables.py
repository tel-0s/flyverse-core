"""Structural half of the round-4 `slowdet` task (CPU only): what exactly separates the `--dopamine-lead dop1r1` arm
from the shipped (DopEcR-led) arm, so that a walk.power_max difference between them can be attributed.

For the shipped `flyverse/data/receptors_by_type.csv` and a dop1r1 rebuild (default `out/r4_sd_dop1r1_table.csv`),
under `net_rule='abs'` and `--receptor-gain 1,1,1` (every gain class 1.0), it reports
  * whether the FAST signs are identical entry for entry (they must be: then the two arms differ only in the slow term),
  * the monoamine slow entries / syn-equivalents and their sign split,
  * per postsynaptic superclass, the cells carrying a monoamine tone and the sign of their tone row sum (syn-eq),
  * the descending neurons with a tone under each table (the only spiking route from the tone to wing power in
    benchmark.py's `walk` section, whose optic lobe is built without `slow=`).
Writes out/r4_slowdet_tables.json.

    python scripts/slowdet_tables.py [--dop1r1 out/r4_sd_dop1r1_table.csv]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from flyverse import brain as br, connectome as cn  # noqa: E402

GAIN = {"none": 1.0, "low": 1.0, "mid": 1.0, "high": 1.0}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dop1r1", default="out/r4_sd_dop1r1_table.csv")
    ap.add_argument("--out", default="out/r4_slowdet_tables.json")
    a = ap.parse_args()
    dop = os.path.join(ROOT, a.dop1r1)
    if not os.path.isfile(dop):
        raise SystemExit(f"no dop1r1 table at {dop} (copy the one benchmark.py --dopamine-lead dop1r1 wrote)")
    c = cn.load(verbose=False)
    sup = c.neurons.superclass.fillna("").to_numpy()
    typ = c.neurons.type.fillna("").to_numpy()
    shipped = os.path.join(ROOT, "flyverse", "data", "receptors_by_type.csv")
    out = {"tables": {"shipped": {"path": shipped, "md5": md5(shipped)},
                      "dop1r1": {"path": dop, "md5": md5(dop)}},
           "connectome": {"n": int(c.n), "nnz": int(c.W.nnz), "sum_abs_W": float(np.abs(c.W.data).sum())},
           "receptor_gain": GAIN, "net_rule": "abs"}
    p0 = br.LIFParams(receptor_model="full", receptor_net_rule="abs", receptor_gain=GAIN,
                      slow_mode="threshold", slow_gain_by_class={"monoamine": 0.01})
    arms = {}
    for name, path in (("shipped", None), ("dop1r1", dop)):
        p = br.LIFParams(**{**p0.__dict__, "receptor_table": path})
        rs = cn.receptor_signs(c, table_path=path, net_rule="abs", with_counts=True)
        S = br._slow_weights(c, p, rs, br._slow_spec(p))["monoamine"]
        row = np.asarray(S.sum(axis=1)).ravel()                      # syn-eq tone load per postsynaptic cell
        m = rs.slow_class == 2
        arms[name] = {
            "fast_sign_changed_vs_presyn": int((rs.fast_sign != np.sign(c.W.data)).sum()),
            "fast_sign": rs.fast_sign,
            "monoamine_entries": int(m.sum()),
            "monoamine_syn_eq": float((rs.count[m] * np.abs(rs.slow_sign[m])).sum()),
            "slow_sign_split": {str(int(v)): int((rs.slow_sign[m] == v).sum()) for v in (-1, 0, 1)},
            "slow_matrix_nnz": int(S.nnz),
            "cells_with_tone": int((row != 0).sum()),
            "tone_negative_cells": int((row < 0).sum()), "tone_positive_cells": int((row > 0).sum()),
            "row": row,
        }
    same_fast = bool(np.array_equal(arms["shipped"]["fast_sign"], arms["dop1r1"]["fast_sign"]))
    out["fast_signs_identical"] = same_fast
    out["fast_sign_diff_entries"] = int((arms["shipped"]["fast_sign"] != arms["dop1r1"]["fast_sign"]).sum())
    for name in arms:
        d = {k: v for k, v in arms[name].items() if k not in ("row", "fast_sign")}
        row = arms[name]["row"]
        by_sup = {}
        for s in sorted(set(sup)):
            k = (sup == s) & (row != 0)
            if k.sum():
                by_sup[s or "<none>"] = {"cells": int(k.sum()), "neg": int((row[k] < 0).sum()),
                                         "pos": int((row[k] > 0).sum()), "syn_eq_sum": float(row[k].sum())}
        d["by_superclass"] = by_sup
        dn = (sup == "descending_neuron") & (row != 0)
        d["descending_with_tone"] = sorted({str(t) for t in typ[dn]})
        d["descending_tone_by_type"] = {str(t): round(float(row[dn][typ[dn] == t].sum()), 3)
                                        for t in sorted(set(typ[dn]))}
        out[name] = d
    rs_diff = arms["shipped"]["row"] - arms["dop1r1"]["row"]
    out["tone_row_diff"] = {"cells_differing": int((rs_diff != 0).sum()),
                            "max_abs": float(np.abs(rs_diff).max()),
                            "by_superclass": {s or "<none>": int(((sup == s) & (rs_diff != 0)).sum())
                                              for s in sorted(set(sup)) if ((sup == s) & (rs_diff != 0)).sum()}}
    path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "connectome"}, indent=1, default=float)[:4000])
    print("wrote", path)


if __name__ == "__main__":
    main()
