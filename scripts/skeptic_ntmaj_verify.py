"""Independent re-derivation of the round-4 type-majority NT rule (skeptic pass on exp:ntmaj).

Written from scratch against flyverse.connectome / flyverse.regions and the raw MaleCNS weight table; it does
not import scripts/audit_nt.py, so its numbers are an independent check of docs/audits/nt_type_majority.md,
out/r4_type_majority.csv and out/r4_type_majority_proposed.json.

    PYTHONIOENCODING=utf-8 python scripts/skeptic_ntmaj_verify.py [--cache-dir DIR] [--min-labelled 4] [--min-share 0.8]
    PYTHONIOENCODING=utf-8 python scripts/skeptic_ntmaj_verify.py --scores out/r4_ntmaj_*.json --baselines out/r3_default_?.json

--scores/--baselines: per-check cross-tab of benchmark JSONs (measured + status per run, the status set per group,
and the checks whose status set differs) plus the config.receptor / cache_dir / nt_counts header of every run --
the adoption criterion re-evaluated outside the reporting script that produced the doc. CPU only, no GPU.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as pf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flyverse import connectome as cn      # noqa: E402
from flyverse import regions               # noqa: E402
from flyverse.brain import LIFParams       # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def raw_out_syn(neurons: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Raw output synapses / edges per model cell, from the raw weight table (weight >= 1)."""
    idx = pd.Series(np.arange(len(neurons)), index=neurons.bodyId.to_numpy())
    w = pf.read_table(cn.DATA_DIR / cn.WEIGHTS_FILE).to_pandas()
    pre = idx.reindex(w.body_pre.to_numpy()).to_numpy()
    post = idx.reindex(w.body_post.to_numpy()).to_numpy()
    keep = ~np.isnan(pre) & ~np.isnan(post) & (w.weight.to_numpy() >= 1)
    pre = pre[keep].astype(np.int64)
    cnt = w.weight.to_numpy()[keep].astype(np.int64)
    n = len(neurons)
    return (np.bincount(pre, weights=cnt, minlength=n).astype(np.int64),
            np.bincount(pre, minlength=n).astype(np.int64), int(cnt.sum()))


def rule_table(cache_dir, min_lab, min_share):
    c = cn.load(verbose=False) if cache_dir is None else cn.load(cache_dir=cache_dir, verbose=False)
    n = c.neurons
    nt = n.nt.to_numpy()
    ty = n.type.fillna("").to_numpy()
    mod = regions.labels(c)
    out_syn, out_edges, total = raw_out_syn(n)
    unknown = nt == "unknown"
    print(f"cache {cache_dir}: {c.n:,} cells, nnz {c.W.nnz:,}, sum|W| {int(np.abs(c.W.data).sum()):,}, raw syn {total:,}")
    print(f"unknown {int(unknown.sum()):,} cells, {int(out_syn[unknown].sum()):,} raw out syn "
          f"({100 * out_syn[unknown].sum() / total:.3f} %), presynaptic {int((unknown & (out_syn > 0)).sum()):,}; "
          f"untyped unknown {int((unknown & (ty == '')).sum()):,} ({int(out_syn[unknown & (ty == '')].sum()):,} syn)")

    tr = pd.read_csv(os.path.join(ROOT, "flyverse", "data", "nt_by_type_transcriptome.csv"), comment="#",
                     dtype=str, keep_default_na=False).set_index("malecns_type")
    nern = pd.read_csv(os.path.join(ROOT, "flyverse", "data", "type_map_nern2025.csv"), comment="#",
                       dtype=str, keep_default_na=False)
    nern = nern[nern.tier.isin(["exact", "class"])].drop_duplicates("malecns_type").set_index("malecns_type")

    rows = []
    for t in sorted(set(ty[unknown & (ty != "")])):
        m = ty == t
        u = m & unknown
        lab = m & ~unknown
        vc = pd.Series(nt[lab]).value_counts()
        maj, maj_n = (str(vc.index[0]), int(vc.iloc[0])) if len(vc) else ("", 0)
        n_lab = int(lab.sum())
        share = maj_n / n_lab if n_lab else 0.0
        tr_nt, tr_n, mixed = "", 0, False
        if t in tr.index:
            tr_nt = tr.at[t, "nt_transcriptome"]
            tr_n = len([x for x in tr.at[t, "sources_agreeing"].split(";") if x])
            mixed = tr.at[t, "pool_mixed"] == "True"
        nern_nt = tr.at[t, "nern2025_prediction"] if (t in tr.index and tr.at[t, "nern2025_prediction"]) else \
            (nern.at[t, "nern_nt"] if t in nern.index else "")
        rows.append({"type": t, "module": pd.Series(mod[m]).value_counts().index[0], "n_cells": int(m.sum()),
                     "n_unknown": int(u.sum()), "n_unknown_pre": int((u & (out_syn > 0)).sum()),
                     "unknown_out_syn": int(out_syn[u].sum()), "unknown_out_edges": int(out_edges[u].sum()),
                     "n_labelled": n_lab, "majority": maj, "majority_n": maj_n, "majority_share": share,
                     "transcriptome": tr_nt, "transcriptome_sources": tr_n, "pool_mixed": mixed,
                     "nern2025": nern_nt,
                     "src_ok": bool(((not mixed) and tr_n >= 1 and tr_nt == maj and maj != "") or (nern_nt and nern_nt == maj))})
    tab = pd.DataFrame(rows)
    tab["rule"] = (tab.n_labelled >= min_lab) & (tab.majority_share >= min_share) & (tab.majority != "")
    tab["proposed"] = tab.rule & tab.src_ok
    wl = tab[tab.n_labelled > 0]
    print(f"typed unknown cells {int(tab.n_unknown.sum()):,} in {len(tab)} types; with labelled members {len(wl)} types, "
          f"{int(wl.n_unknown.sum())} cells ({int(wl.n_unknown_pre.sum())} presynaptic), {int(wl.unknown_out_syn.sum()):,} syn")
    print("  by majority transmitter: " + ", ".join(f"{k} {int(v):,}" for k, v in
          wl.groupby("majority").unknown_out_syn.sum().sort_values(ascending=False).items()))
    nl = tab[tab.n_labelled == 0]
    print(f"  types with NO labelled member: {len(nl)} ({int(nl.n_unknown.sum())} cells, {int(nl.unknown_out_syn.sum()):,} syn); "
          "largest: " + ", ".join(f"{r.type} {r.unknown_out_syn:,}" for r in
                                  nl.sort_values('unknown_out_syn', ascending=False).head(7).itertuples()))
    ns = wl[(wl.transcriptome == "") & (wl.nern2025 == "")]
    print(f"  with labelled members but no transcriptome / Nern call: {len(ns)} types, {int(ns.n_unknown.sum())} cells, "
          f"{int(ns.unknown_out_syn.sum()):,} syn")
    p = tab[tab.proposed]
    print(f"rule alone ({min_lab} labelled, share {min_share:g}): {int(tab.rule.sum())} types / {int(tab.n_unknown[tab.rule].sum())} "
          f"cells / {int(tab.unknown_out_syn[tab.rule].sum()):,} syn")
    print(f"PROPOSED: {len(p)} types / {int(p.n_unknown.sum())} cells / {int(p.n_unknown_pre.sum())} presynaptic / "
          f"{int(p.unknown_out_syn.sum()):,} raw out syn / {int(p.unknown_out_edges.sum())} edges = "
          + ", ".join(f"{r.type} -> {r.majority} ({r.n_unknown} cells, {r.n_labelled} labelled, share {r.majority_share:.2f}, "
                      f"{r.module})" for r in p.itertuples()))
    for min_l, sh in [(1, 0.5), (1, 1.0), (2, 1.0), (3, 0.8), (4, 0.8), (4, 1.0), (8, 0.8), (20, 0.8)]:
        r = (tab.n_labelled >= min_l) & (tab.majority_share >= sh) & (tab.majority != "")
        b = r & tab.src_ok
        print(f"  sens ({min_l}, {sh:g}): rule {int(r.sum())}/{int(tab.n_unknown[r].sum())}/{int(tab.unknown_out_syn[r].sum()):,} | "
              f"+source {int(b.sum())}/{int(tab.n_unknown[b].sum())}/{int(tab.unknown_out_syn[b].sum()):,}")
    return c, tab


def compare_w(c, other):
    c2 = cn.load(cache_dir=other, verbose=False)
    A, B = c.W.tocsr(), c2.W.tocsr()
    A.sort_indices(); B.sort_indices()
    h = lambda x: hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
    same = A.shape == B.shape and A.nnz == B.nnz and all(h(getattr(A, k)) == h(getattr(B, k)) for k in ("data", "indices", "indptr"))
    print(f"W identical {same}: nnz {A.nnz:,}/{B.nnz:,}, sum|W| {int(np.abs(A.data).sum()):,}/{int(np.abs(B.data).sum()):,}, "
          f"data md5 {h(A.data)}/{h(B.data)}, indices {h(A.indices)[:8]}/{h(B.indices)[:8]}, indptr {h(A.indptr)[:8]}/{h(B.indptr)[:8]}")
    n1, n2 = c.neurons, c2.neurons
    d = n1.nt.to_numpy() != n2.nt.to_numpy()
    nnz_out = np.diff(A.tocsc().indptr)
    print(f"nt differs on {int(d.sum())} cells (sign differs {int((n1.sign.to_numpy() != n2.sign.to_numpy()).sum())}); "
          f"stored W output entries of those cells {int(nnz_out[d].sum())}; by type "
          + ", ".join(f"{k} {int(v)}" for k, v in pd.Series(n1.type.fillna('').to_numpy()[d]).value_counts().items()))
    lp = LIFParams()
    r1 = cn.receptor_signs(c, table_path=lp.receptor_table, net_rule=lp.receptor_net_rule,
                           nt_class_fallback=lp.receptor_nt_class_fallback)
    r2 = cn.receptor_signs(c2, table_path=lp.receptor_table, net_rule=lp.receptor_net_rule,
                           nt_class_fallback=lp.receptor_nt_class_fallback)
    print(f"receptor {lp.receptor_model}/{lp.receptor_net_rule}: fast_sign md5 {h(r1.fast_sign)}/{h(r2.fast_sign)} "
          f"identical {h(r1.fast_sign) == h(r2.fast_sign)}; changed vs presynaptic sign "
          f"{int((r1.fast_sign != np.sign(c.W.data)).sum()):,}/{int((r2.fast_sign != np.sign(c2.W.data)).sum()):,}")


def cross_tab(scores, baselines):
    def expand(fs):
        out = []
        for f in fs:
            out += sorted(glob.glob(f)) or [f]
        return out
    runs = []
    for grp, fs in (("cand", expand(scores)), ("base", expand(baselines))):
        for f in fs:
            d = json.load(open(f, encoding="utf-8"))
            cfg = d.get("config", {})
            r = dict(cfg.get("receptor", {}))
            r.pop("coverage", None)
            runs.append({"group": grp, "file": os.path.basename(f), "checks": {c["key"]: c for c in d["checks"]},
                         "hdr": f"{d.get('date')} {cfg.get('device')} cache={cfg.get('cache_dir')} receptor={r} "
                                f"unknown={(cfg.get('nt_counts') or {}).get('unknown')}"})
    for r in runs:
        tally = {s: sum(1 for c in r["checks"].values() if c["status"] == s) for s in ("PASS", "FAIL", "KNOWN GAP")}
        print(f"{r['group']} {r['file']}: {tally['PASS']}/{tally['FAIL']}/{tally['KNOWN GAP']}  {r['hdr']}")
    keys = list(dict.fromkeys(k for r in runs for k in r["checks"]))
    moved = []
    for k in keys:
        cells, st = [], {"cand": set(), "base": set()}
        for r in runs:
            c = r["checks"].get(k)
            if c is None:
                cells.append("-"); continue
            m = c["measured"]
            cells.append((f"{m:.2f}" if isinstance(m, (int, float)) and m is not None else "--") + " " + c["status"][:1])
            st[r["group"]].add(c["status"])
        flag = ""
        if st["cand"] and st["base"] and st["cand"] != st["base"]:
            moved.append(f"{k}: cand {sorted(st['cand'])} vs base {sorted(st['base'])}"); flag = "  <== MOVED"
        print(f"  {k:38s} " + " | ".join(f"{x:>9s}" for x in cells) + flag)
    print("status moved: " + ("; ".join(moved) if moved else "none") +
          f"  => criterion (no check changes status) {'NOT MET' if moved else 'MET'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--compare-cache", default=None)
    ap.add_argument("--min-labelled", type=int, default=4)
    ap.add_argument("--min-share", type=float, default=0.8)
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--csv-out", default=None)
    a = ap.parse_args()
    if a.scores:
        cross_tab(a.scores, a.baselines)
        return
    c, tab = rule_table(a.cache_dir, a.min_labelled, a.min_share)
    if a.csv_out:
        tab.to_csv(a.csv_out, index=False)
        print(f"wrote {a.csv_out}")
    if a.compare_cache:
        compare_w(c, a.compare_cache)


if __name__ == "__main__":
    main()
