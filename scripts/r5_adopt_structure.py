"""Receptor round 5, GF-damping adoption: structural check of the edited default (CPU, no simulation).

    PYTHONIOENCODING=utf-8 python scripts/r5_adopt_structure.py [--json out/r5_adopt_structure.json]

Prints, on the local connectome cache: the md5 of brain._shaped_weights (sorted CSR data + indices + indptr) under the new
default LIFParams(), under LIFParams(type_path_gain=brain.GF_DAMPED_TYPE_PATH_GAIN) (the previous default: must equal the
round-3 pinned hash f0d145d1bb81b446ebc51f89ded7bd4b on the adopted TYPE_NT_OVERRIDE cache), under receptor_model=None with
both lists (the previous None hash is 2e276b30b6117c1f62688b01775eda6b); the entries that differ between the new and the
previous default (count, the presynaptic types, the sum of |W| restored, per NT of the presynaptic cell) and DNp01's input
before / after.  The same hashes are pinned in tests/test_receptor_model.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import brain, connectome as cn                                  # noqa: E402


def md5(W):
    W = W.tocsr(); W.sort_indices()
    m = hashlib.md5(); m.update(W.data.tobytes()); m.update(W.indices.tobytes()); m.update(W.indptr.tobytes())
    return m.hexdigest(), W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/r5_adopt_structure.json")
    a = ap.parse_args()
    c = cn.load(verbose=False)
    out = {"cache": {"nnz": int(c.W.nnz), "sum_abs_W": int(np.abs(c.W.data).sum()),
                     "glutamate_cells": int((c.neurons.nt == "glutamate").sum())},
           "default_type_path_gain": brain.DEFAULT_TYPE_PATH_GAIN, "gf_damped_type_path_gain": brain.GF_DAMPED_TYPE_PATH_GAIN}
    print("cache:", out["cache"])
    print("DEFAULT_TYPE_PATH_GAIN:", brain.DEFAULT_TYPE_PATH_GAIN)
    print("GF_DAMPED_TYPE_PATH_GAIN:", brain.GF_DAMPED_TYPE_PATH_GAIN)
    cfgs = {"new_default": brain.LIFParams(),
            "previous_default": brain.LIFParams(type_path_gain=brain.GF_DAMPED_TYPE_PATH_GAIN),
            "new_none": brain.LIFParams(receptor_model=None),
            "previous_none": brain.LIFParams(receptor_model=None, type_path_gain=brain.GF_DAMPED_TYPE_PATH_GAIN)}
    Ws = {}
    for k, p in cfgs.items():
        h, W = md5(brain._shaped_weights(c, p)); Ws[k] = W
        out[f"md5_{k}"] = h
        print(f"md5 {k:17s} {h}   receptor_model={p.receptor_model!r} type_path_gain={p.type_path_gain}")
    ty = c.neurons.type.fillna("").to_numpy(); nt = c.neurons.nt.fillna("").to_numpy()
    for pair in (("new_default", "previous_default"), ("new_none", "previous_none")):
        A, B = Ws[pair[0]], Ws[pair[1]]
        assert np.array_equal(A.indices, B.indices) and np.array_equal(A.indptr, B.indptr)
        d = np.flatnonzero(A.data != B.data)
        Ac = A.tocoo(); rows, cols = Ac.row[d], Ac.col[d]
        assert np.all(ty[rows] == "DNp01")
        pre = {}
        for t, n_, r, w_new, w_old in zip(ty[cols], nt[cols], rows, A.data[d], B.data[d]):
            e = pre.setdefault(t, {"nt": n_, "edges": 0, "abs_w_new": 0.0, "abs_w_previous": 0.0})
            e["edges"] += 1; e["abs_w_new"] += float(abs(w_new)); e["abs_w_previous"] += float(abs(w_old))
        ratio = sorted(set(np.round(np.abs(B.data[d]) / np.abs(A.data[d]), 6).tolist()))
        print(f"{pair[0]} vs {pair[1]}: {len(d)} entries differ, all onto DNp01; previous/new |W| ratio {ratio}; by presynaptic type:")
        for t, e in sorted(pre.items()):
            print(f"   {t:8s} {e['nt']:14s} edges {e['edges']:3d}  |W| new {e['abs_w_new']:9.1f}  previous {e['abs_w_previous']:9.1f}")
        out[f"diff_{pair[0]}_vs_{pair[1]}"] = {"entries": int(len(d)), "ratio_previous_over_new": ratio, "by_pre_type": pre,
                                                "abs_w_new_total": float(np.abs(A.data[d]).sum()),
                                                "abs_w_previous_total": float(np.abs(B.data[d]).sum())}
    dn = np.flatnonzero(ty == "DNp01")
    for k in ("new_default", "previous_default"):
        W = Ws[k].tocsr(); tot = float(np.abs(W[dn]).sum()); inh = float(np.abs(W[dn].data[W[dn].data < 0]).sum())
        out[f"dnp01_input_{k}"] = {"abs_w": tot, "inhibitory_abs_w": inh, "edges": int(W[dn].nnz)}
        print(f"DNp01 shaped input {k}: {W[dn].nnz} edges, |W| {tot:.1f}, inhibitory |W| {inh:.1f} ({100*inh/tot:.2f} %)")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1)
    print("wrote", a.json)


if __name__ == "__main__":
    main()
