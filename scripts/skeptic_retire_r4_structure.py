"""Skeptic re-count of the round-4 retire study's structural numbers (CPU, local cache, no GPU).

Reproduces, from flyverse.connectome.load() alone, every structural figure the exp:retire report states in its
KEY CLAIMS 7 / 8 / 13 and in docs/audits/anti_runaway.md's "Round 4" section: the LPi34|LPi43 -> LPLC2 pair, the
T4/T5 -> LPLC2 pair, LPLC2's whole input, the five damped DNp01 inputs, LC4 / LPLC2 -> DNp01, and the AL LN
UNKNOWN_NT_OVERRIDE_REGEX (cells, types, output edges / |W| by post class, ORN inputs, the sum|W| and unknown-NT
cell count with the regex emptied).  c.W is W[post, pre].

    PYTHONIOENCODING=utf-8 python scripts/skeptic_retire_r4_structure.py > out/sk4_retire_structure.txt
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flyverse import connectome as cn  # noqa: E402


def pair(c, pre_re, post_re, types):
    W = c.W.tocoo()
    pre_m = np.array([bool(re.match(pre_re, t)) for t in types])
    post_m = np.array([bool(re.match(post_re, t)) for t in types])
    sel = pre_m[W.col] & post_m[W.row]
    return {"edges": int(sel.sum()), "absW": float(np.abs(W.data[sel]).sum()),
            "pre_cells": int(len(np.unique(W.col[sel]))), "post_cells": int(len(np.unique(W.row[sel])))}


def total_input(c, post_re, types):
    W = c.W.tocoo()
    post_m = np.array([bool(re.match(post_re, t)) for t in types])
    sel = post_m[W.row]
    return {"edges": int(sel.sum()), "absW": float(np.abs(W.data[sel]).sum())}


def main():
    c = cn.load(verbose=False)
    types = c.neurons.type.fillna("").to_numpy()
    out = {"sum_absW": float(np.abs(c.W.data).sum()), "n": int(c.n), "nnz": int(c.W.nnz)}

    # --- LPi -> LPLC2 and the scan ---
    lpi = pair(c, r"^LPi(34|43)$", r"^LPLC2$", types)
    t45 = pair(c, r"^T[45][abcd]$", r"^LPLC2$", types)
    tot = total_input(c, r"^LPLC2$", types)
    out["lpi_to_lplc2"] = lpi
    out["t4t5_to_lplc2"] = t45
    out["lplc2_total_input"] = tot
    out["scan"] = {}
    for f in (1.0, 2.0, 3.0, 4.0):
        g = lpi["absW"] * f
        out["scan"][f"x{int(f)}"] = {
            "lpi_over_t4t5_with_gains": g / (t45["absW"] * 2.0),                       # T4/T5 carry the x2 pair gain
            "share_report_rule": g / (tot["absW"] - lpi["absW"] + g),                  # LPi gain only (the report's rule)
            "share_all_gains": g / (tot["absW"] - lpi["absW"] - t45["absW"] + g + t45["absW"] * 2.0),
        }

    # --- the five damped DNp01 inputs ---
    gf = {}
    for t in ("SAD073", "GNG300", "DNp70", "CL367", "PVLP010", "LC4", "LPLC2"):
        d = pair(c, rf"^{t}$", r"^DNp01$", types)
        nts = sorted(set(c.neurons.nt.fillna("").to_numpy()[np.unique(c.W.tocoo().col[
            np.array([bool(re.match(rf"^{t}$", x)) for x in types])[c.W.tocoo().col] &
            np.array([bool(re.match(r"^DNp01$", x)) for x in types])[c.W.tocoo().row]])]))
        gf[t] = {"edges": d["edges"], "absW": d["absW"], "nt": nts}
    gf["DNp01_total_input"] = total_input(c, r"^DNp01$", types)
    five = sum(gf[t]["absW"] for t in ("SAD073", "GNG300", "DNp70", "CL367", "PVLP010"))
    five_e = sum(gf[t]["edges"] for t in ("SAD073", "GNG300", "DNp70", "CL367", "PVLP010"))
    gf["five_damped"] = {"edges": five_e, "absW": five, "share_of_DNp01_input": five / gf["DNp01_total_input"]["absW"]}
    gf["dnp70_only"] = {"absW": gf["DNp70"]["absW"], "share_of_five": gf["DNp70"]["absW"] / five,
                        "share_of_DNp01_input": gf["DNp70"]["absW"] / gf["DNp01_total_input"]["absW"]}
    out["gf"] = gf

    # --- the AL LN unknown-NT override ---
    pat = list(cn.UNKNOWN_NT_OVERRIDE_REGEX)[0]
    W = c.W.tocoo()
    cls = c.neurons["class"].fillna("").to_numpy() if "class" in c.neurons.columns else np.array([""] * c.n)
    # the cells the regex relabels: type matches AND the raw consensus was 'unknown' -- the compiled cache already
    # carries the override, so identify them by recompiling the neuron table is not possible here; use the label the
    # cache has (gaba) plus the type pattern, and cross-check the count against the compile log rule.
    m_type = np.array([bool(re.match(pat, t)) for t in types])
    al = {"regex": pat, "cells_matching_type": int(m_type.sum()),
          "gaba_cells_matching_type": int((m_type & (c.neurons.nt.fillna("").to_numpy() == "gaba")).sum()),
          "types": sorted(set(types[m_type & (c.neurons.nt.fillna("").to_numpy() == "gaba")]))}
    sel = m_type[W.col]
    al["out_edges"] = int(sel.sum()); al["out_absW"] = float(np.abs(W.data[sel]).sum())
    by_cls = {}
    for k in sorted(set(cls[W.row[sel]])):
        s2 = sel & (cls[W.row] == k)
        by_cls[k or "(none)"] = {"edges": int(s2.sum()), "absW": float(np.abs(W.data[s2]).sum())}
    al["out_by_post_class"] = by_cls
    orn = np.array([t.startswith("ORN_") for t in types])
    s3 = orn[W.col] & m_type[W.row]
    al["orn_into_them"] = {"edges": int(s3.sum()), "absW": float(np.abs(W.data[s3]).sum())}
    pn = np.array([("PN" in t) for t in types])
    s4 = m_type[W.col] & pn[W.row]
    al["onto_PN_named"] = {"edges": int(s4.sum()), "absW": float(np.abs(W.data[s4]).sum())}
    out["al_ln"] = al

    # --- the same brain with the regex emptied (compiles into a scratch cache dir, never the shared one) ---
    scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "sk4_cache_no_al_ln")
    old = dict(cn.UNKNOWN_NT_OVERRIDE_REGEX)
    try:
        cn.UNKNOWN_NT_OVERRIDE_REGEX.clear()
        c2 = cn.load(verbose=False, cache_dir=scratch)
    finally:
        cn.UNKNOWN_NT_OVERRIDE_REGEX.update(old)
    out["no_al_ln_override"] = {
        "sum_absW": float(np.abs(c2.W.data).sum()),
        "unknown_nt_cells": int((c2.neurons.nt.fillna("") == "unknown").sum()),
        "unknown_nt_cells_default": int((c.neurons.nt.fillna("") == "unknown").sum()),
        "sum_absW_default": out["sum_absW"],
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
