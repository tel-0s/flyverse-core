"""Hold tables: the shipped receptor table with one group of its contested rows held at the NT_SIGN prior.

    python scripts/build_hold_tables.py                            # out/receptors_hold{KC,DN1,Brain,Optic,BrainGlu,BrainHis}.csv
    python scripts/build_hold_tables.py --groups Brain,Optic        # only those two
    python scripts/build_hold_tables.py --verify                    # + counts the entries each table changes vs sign(W.data)
    python scripts/build_hold_tables.py --fanin                     # round-5 fan-in probe (CPU, no simulation)

Each table is flyverse/data/receptors_by_type.csv (its '#' header kept) with one group of rows set back to the
presynaptic prior (`connectome.NT_SIGN`): fast_sign_abs = NT_SIGN[transmitter] (glutamate / histamine -1, i.e. no +1
flip and no silencing), fast_gain_class_abs = none, fast_net_abs = 'held'. Every other entry of the default stays.
connectome.receptor_signs reads only the fast_sign_abs / fast_gain_class_abs columns, so 'held' in fast_net_abs is a
label for the record.

Groups:
  round 4 (docs/audits/receptor_verification.md, round-3 critic follow-up 2) -- glutamate rows of one type group:
    KC   = KCg-m, KCg-s1..s4, KCa'b'-ap1, KCa'b'-ap2, KCg (GluRIB-led +1 rows, tier fuzzy, fca2022)
    DN1  = DN1a, DN1pA, DN1pB (clock, davie2018)
  round 5 (Brain-side vs optic-side attribution) -- every row of one SIDE that differs from the prior:
    Brain = the non-optic types (table column `superclass` not in ol_intrinsic / ol_sensory): holding them leaves the
            optic-side changes only, so a difference from the default is carried by the Brain side
    Optic = the complement (ol_intrinsic / ol_sensory types): holding them leaves the Brain-side changes only
    BrainGlu / BrainHis = the Brain side restricted to one transmitter. The Brain side's glutamate rows are exactly
            the KC and DN1 groups (3,709 entries / 8,551 syn), so holdBrainGlu is round 4's holdKC + holdDN1 applied
            together; its histamine rows are the 123 entries / 282 syn round 4 was left with, so holdBrainHis holds
            exactly that residual. The pair dissociates taste (the histamine group) from smell (the glutamate group).
  The 28 synthetic `<nt=...>` / `<superclass=...>` rows (superclass NaN) are in neither side: receptor_signs excludes
  them from the per-type lookup and reads them only under nt_class_fallback, which is off in every run here.

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
TYPE_GROUPS = {
    "KC": ["KCg-m", "KCg-s1", "KCg-s2", "KCg-s3", "KCg-s4", "KCa'b'-ap1", "KCa'b'-ap2", "KCg"],
    "DN1": ["DN1a", "DN1pA", "DN1pB"],
}
OPTIC_SUPERCLASSES = ("ol_intrinsic", "ol_sensory")
# {group: (side, transmitter or None)} -- 'brain' = the types whose table superclass is NOT ol_intrinsic / ol_sensory.
SIDE_GROUPS = {"Brain": ("brain", None), "Optic": ("optic", None),
               "BrainGlu": ("brain", "glutamate"), "BrainHis": ("brain", "histamine")}
GROUPS = (*TYPE_GROUPS, *SIDE_GROUPS)
# connectome.NT_SIGN, duplicated so the table can be built without importing torch / loading the cache.
NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
           "dopamine": 0.0, "octopamine": 0.0, "serotonin": 0.0}


def hold_table(group: str) -> tuple[str, pd.DataFrame, int]:
    header = "".join(l for l in open(TABLE, encoding="utf-8") if l.startswith("#"))
    t = pd.read_csv(TABLE, comment="#")
    if group in TYPE_GROUPS:
        sel = t.malecns_type.isin(TYPE_GROUPS[group]) & (t.transmitter == "glutamate")
        t.loc[sel, "fast_sign_abs"] = -1
        t.loc[sel, "fast_gain_class_abs"] = "none"
        t.loc[sel, "fast_net_abs"] = "held"
        header += (f"# scripts/build_hold_tables.py: {int(sel.sum())} glutamate rows of {group} "
                   f"({', '.join(TYPE_GROUPS[group])}) held at NT_SIGN (fast_sign_abs -1, gain none, fast_net_abs held)\n")
        return header, t, int(sel.sum())
    if group not in SIDE_GROUPS:
        raise ValueError(f"unknown group {group!r}; known: {', '.join(GROUPS)}")
    which, nt = SIDE_GROUPS[group]
    optic = t.superclass.isin(OPTIC_SUPERCLASSES)
    synthetic = t.malecns_type.astype(str).str.startswith("<")
    side = optic if which == "optic" else (~optic & ~synthetic)
    if nt is not None:
        side = side & (t.transmitter == nt)
    prior = t.transmitter.map(NT_SIGN).astype(float)
    sel = side & (t.fast_sign_abs.astype(float) != prior)          # only the rows that differ from the prior need holding
    t.loc[sel, "fast_sign_abs"] = prior[sel].astype(int)
    t.loc[sel, "fast_gain_class_abs"] = "none"
    t.loc[sel, "fast_net_abs"] = "held"
    by = t.loc[sel].groupby("transmitter").size().to_dict()
    what = ("the ol_intrinsic / ol_sensory types" if which == "optic" else "the non-(ol_intrinsic / ol_sensory) types") + \
           ("" if nt is None else f", {nt} rows only")
    header += (f"# scripts/build_hold_tables.py: {int(sel.sum())} rows of {what} held at NT_SIGN "
               f"(fast_sign_abs = NT_SIGN[transmitter], gain none, fast_net_abs held); by transmitter {by}\n")
    return header, t, int(sel.sum())


def write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def verify(paths: dict) -> None:
    """Count the entries each table changes vs sign(W.data) and split them by postsynaptic superclass."""
    import numpy as np
    sys.path.insert(0, ROOT)
    from flyverse import connectome as cn
    c = cn.load(verbose=False)
    base = np.sign(c.W.data)
    ref = cn.receptor_signs(c, net_rule="abs")
    d_ref = ref.fast_sign != base
    post = c.W.tocoo().row
    types = c.neurons.type.to_numpy()
    sc = c.neurons.superclass.fillna("").to_numpy()
    wabs = np.abs(c.W.data)

    def split(mask):
        s = pd.Series(sc[post[mask]]).value_counts()
        w = pd.Series(wabs[mask]).groupby(pd.Series(sc[post[mask]])).sum().round().astype(int)
        return {k: (int(v), int(w[k])) for k, v in s.items()}

    print(f"connectome: {c.W.nnz:,} stored entries, sum|W| {float(wabs.sum()):,.0f}")
    print(f"shipped table: {int(d_ref.sum())} entries changed vs sign(W.data), |W| {float(wabs[d_ref].sum()):.0f}")
    print(f"  by postsynaptic superclass (entries, |W|): {split(d_ref)}")
    optic_post = np.isin(sc, OPTIC_SUPERCLASSES)[post]
    print(f"  optic-side (post superclass in {OPTIC_SUPERCLASSES}): {int((d_ref & optic_post).sum())} entries / "
          f"{float(wabs[d_ref & optic_post].sum()):.0f} |W|; Brain-side {int((d_ref & ~optic_post).sum())} / "
          f"{float(wabs[d_ref & ~optic_post].sum()):.0f}")
    for g, p in paths.items():
        r = cn.receptor_signs(c, table_path=p, net_rule="abs")
        d = r.fast_sign != base
        held = d_ref & ~d                                  # entries the default changes and this table does not
        extra = d & ~d_ref
        print(f"hold{g}: {int(d.sum())} entries changed (default - {int(held.sum())}; new {int(extra.sum())}); "
              f"held |W| {float(wabs[held].sum()):.0f}; remaining |W| {float(wabs[d].sum()):.0f}")
        if g in TYPE_GROUPS:
            ht = pd.Series(types[post[held]]).value_counts()
            print(f"  held entries by postsynaptic type: {ht.to_dict()}")
        else:
            print(f"  remaining changes by postsynaptic superclass (entries, |W|): {split(d)}")
            print(f"  held entries by postsynaptic superclass (entries, |W|): {split(held)}")
            exp = {"Brain": 44463, "Optic": 3832, "BrainGlu": 44586, "BrainHis": 48172}[g]
            print(f"  EXPECTED {exp} entries changed -> {'OK' if int(d.sum()) == exp else 'MISMATCH'}")


def fanin(paths: dict, out_dir: str) -> None:
    """Round 5: the fan-in-normalisation hypothesis, directly.

    brain._shaped_weights multiplies abs(W.data) by receptor.fast_factor(), so under receptor_model='sign' (gain None)
    the magnitude of an entry changes only where fast_sign == 0 -- a silencing. A sign FLIP leaves abs(W) untouched.
    The fan-in scale is Brain.__init__'s
        tot = abs(shaped W).sum(axis=1);  input_scale = clip((input_norm_ref / max(tot, 1)) ** input_norm_alpha, 0.02, 1)
    so this reproduces tot / input_scale per cell under each table and reports every cell whose scale moves, plus the
    per-type numbers for MN9, MN9's presynaptic partners and the sweet second-order cells of benchmark.sec_taste.
    """
    import json
    import numpy as np
    import scipy.sparse as sp
    sys.path.insert(0, ROOT)
    from flyverse import brain, connectome as cn
    c = cn.load(verbose=False)
    n = c.neurons
    p0 = brain.LIFParams()
    print(f"LIFParams: receptor_model {p0.receptor_model!r} net_rule {p0.receptor_net_rule!r} "
          f"input_norm_alpha {p0.input_norm_alpha} input_norm_ref {p0.input_norm_ref} conn_cap {p0.conn_cap}")

    def scale_of(table_path=None, model="sign"):
        p = brain.LIFParams(receptor_model=model, receptor_net_rule="abs", receptor_table=table_path)
        W = brain._shaped_weights(c, p)
        tot = np.asarray(abs(W).sum(axis=1)).ravel()
        s = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
        return tot, s

    conds = {"off": (None, None), "default": (None, "sign")}
    for g in ("Brain", "Optic"):
        if g in paths:
            conds[f"hold{g}"] = (paths[g], "sign")
    tots, scales = {}, {}
    for name, (tp, model) in conds.items():
        tots[name], scales[name] = scale_of(tp, model)
        print(f"{name:10s} sum tot {tots[name].sum():,.0f}; cells with tot > 0 {int((tots[name] > 0).sum()):,}; "
              f"input_scale mean {scales[name].mean():.4f}, at the 0.02 floor {int((scales[name] <= 0.02).sum()):,}")

    # who moves at all
    print("\n-- cells whose fan-in scale differs from 'off' --")
    rows = []
    for name in conds:
        if name == "off":
            continue
        d = scales[name] != scales["off"]
        dt = tots[name] != tots["off"]
        by = pd.Series(n.superclass.fillna("").to_numpy()[d]).value_counts().to_dict()
        print(f"{name:10s} tot moves on {int(dt.sum()):,} cells; input_scale moves on {int(d.sum()):,} cells; "
              f"by superclass {by}")
        if int(d.sum()):
            tp = pd.Series(n.type.fillna("").to_numpy()[d]).value_counts()
            print(f"           top types: {tp.head(15).to_dict()}")
        rows.append({"condition": name, "cells_tot_moved": int(dt.sum()), "cells_scale_moved": int(d.sum()),
                     "by_superclass": by})

    # the taste pathway of benchmark.sec_taste
    taste = pd.read_csv(os.path.join(ROOT, "flyverse", "data", "taste_grns.csv"))
    sweet = c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
    sweet = sweet[np.isin(n.subclass.to_numpy()[sweet], ["labellar bristle", "taste peg"])]
    mn9 = c.select(type="MN9")
    W = c.W.tocsr()
    second = np.unique(W[:, sweet].tocoo().row)                      # postsynaptic partners of the driven sweet GRNs
    mn9_in = np.unique(W[mn9, :].tocoo().col)                        # presynaptic partners of MN9
    groups = {"sweet_GRN_driven": sweet, "sweet_second_order": second, "MN9": mn9, "MN9_inputs": mn9_in}
    print(f"\n-- the sec_taste pathway: sweet GRNs {len(sweet)} cells, second order {len(second)}, "
          f"MN9 {len(mn9)}, MN9 inputs {len(mn9_in)} --")
    per_group = {}
    for gname, idx in groups.items():
        moved = {k: int((scales[k][idx] != scales["off"][idx]).sum()) for k in conds if k != "off"}
        line = {k: [float(tots[k][idx].mean()), float(scales[k][idx].mean()), float(scales[k][idx].min()),
                    float(scales[k][idx].max())] for k in conds}
        per_group[gname] = {"n_cells": int(len(idx)), "cells_with_moved_scale": moved,
                            "mean_tot__mean_scale__min__max": line}
        print(f"{gname:20s} n {len(idx):5d}  cells whose scale moves vs off: {moved}")
        for k in conds:
            print(f"    {k:10s} mean tot {tots[k][idx].mean():12,.1f}  input_scale mean {scales[k][idx].mean():.6f} "
                  f"[{scales[k][idx].min():.6f}, {scales[k][idx].max():.6f}]")

    # per cell type of the two named groups, with the sign changes on their own input
    base = np.sign(c.W.data)
    coo = c.W.tocoo()
    post, pre = coo.row, coo.col
    chg = {}
    for name, (tp, model) in conds.items():
        if model is None:
            continue
        r = cn.receptor_signs(c, table_path=tp, net_rule="abs")
        chg[name] = r.fast_sign != base

    # every cell whose input_scale moves, in full
    moved_any = np.zeros(c.n, dtype=bool)
    for k in conds:
        if k != "off":
            moved_any |= scales[k] != scales["off"]
    print(f"\n-- the {int(moved_any.sum())} cells whose input_scale moves at all --")
    for i in np.flatnonzero(moved_any):
        m = post == i
        ch = {k: int((cm & m).sum()) for k, cm in chg.items()}
        sil = {k: int(((cn.receptor_signs(c, table_path=conds[k][0], net_rule='abs').fast_sign == 0) & m & (base != 0)).sum())
               for k in ("default",)}
        print(f"  {n.bodyId.to_numpy()[i]} {n.type.to_numpy()[i]} ({n.superclass.to_numpy()[i]}): "
              f"tot " + " ".join(f"{k} {tots[k][i]:,.0f}" for k in conds) + "; scale " +
              " ".join(f"{k} {scales[k][i]:.6f}" for k in conds) +
              f"; changed input entries {ch}; silenced (default) {sil['default']}")

    # changed input entries per group, over every cell of the group (not only the top types)
    print("\n-- changed input entries of the sec_taste pathway groups (all cells of the group) --")
    for gname, idx in groups.items():
        m = np.isin(post, idx)
        line = {k: (int((cm & m).sum()), float(np.abs(c.W.data[cm & m]).sum())) for k, cm in chg.items()}
        print(f"  {gname:20s} input entries {int(m.sum()):9,d}  changed (entries, |W|) {line}")

    # the Brain-side changed entries, by postsynaptic type, with the taste-pathway membership of each type
    print("\n-- the Brain-side changed entries (holdOptic's 3,832) by postsynaptic type --")
    m = chg["holdOptic"]
    ty = n.type.fillna("").to_numpy()
    pre_nt = n.nt.fillna("unknown").to_numpy() if "nt" in n.columns else np.array(["?"] * c.n)
    dd = pd.DataFrame({"type": ty[post[m]], "nt": pre_nt[pre[m]], "w": np.abs(c.W.data[m])})
    agg = dd.groupby(["type", "nt"]).agg(entries=("w", "size"), syn=("w", "sum")).reset_index()
    agg = agg.sort_values("entries", ascending=False)
    in_taste = {t for t in ty[np.concatenate([groups["sweet_second_order"], groups["MN9_inputs"], groups["MN9"]])]}
    agg["in_taste_pathway"] = agg.type.isin(in_taste)
    print(agg.to_string(index=False))
    print(f"  types in the sec_taste pathway among them: "
          f"{agg[agg.in_taste_pathway].groupby('type').entries.sum().to_dict()}")

    print("\n-- per postsynaptic type (the cells of MN9 / MN9_inputs / sweet_second_order): tot, input_scale, "
          "changed input entries --")
    tbl = []
    for gname in ("MN9", "MN9_inputs", "sweet_second_order"):
        idx = groups[gname]
        ty = n.type.fillna("").to_numpy()
        keep = np.isin(post, idx)
        for t in pd.Series(ty[idx]).value_counts().head(25).index:
            cells = idx[ty[idx] == t]
            m = keep & np.isin(post, cells)
            row = {"group": gname, "type": t, "cells": int(len(cells)),
                   "in_entries": int(m.sum()), "in_syn": float(np.abs(c.W.data[m]).sum())}
            for k in conds:
                row[f"tot_{k}"] = float(tots[k][cells].mean())
                row[f"scale_{k}"] = float(scales[k][cells].mean())
            for k, cm in chg.items():
                row[f"changed_in_{k}"] = int((cm & m).sum())
            tbl.append(row)
    df = pd.DataFrame(tbl)
    cols = ["group", "type", "cells", "in_entries", "in_syn"] + [f"tot_{k}" for k in conds] + \
           [f"scale_{k}" for k in conds] + [f"changed_in_{k}" for k in chg]
    df = df[cols]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, "r5_fanin_types.csv")
    df.to_csv(csv, index=False, lineterminator="\n")
    js = os.path.join(out_dir, "r5_fanin.json")
    with open(js, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"lif": {"input_norm_alpha": p0.input_norm_alpha, "input_norm_ref": p0.input_norm_ref,
                           "conn_cap": p0.conn_cap, "receptor_model": p0.receptor_model,
                           "receptor_net_rule": p0.receptor_net_rule},
                   "moved": rows, "groups": per_group,
                   "sum_tot": {k: float(v.sum()) for k, v in tots.items()}}, f, indent=1)
    print(f"\nwrote {csv} and {js}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--groups", default=",".join(GROUPS), help=f"comma-separated subset of {', '.join(GROUPS)}")
    ap.add_argument("--verify", action="store_true", help="count changed entries with connectome.receptor_signs (loads the local cache)")
    ap.add_argument("--fanin", action="store_true", help="fan-in normalisation probe: per-cell input_scale under each table (CPU)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {}
    for g in [x.strip() for x in a.groups.split(",") if x.strip()]:
        header, t, n = hold_table(g)
        p = os.path.join(a.out_dir, f"receptors_hold{g}.csv")
        write_atomic(p, header + t.to_csv(index=False, lineterminator="\n"))
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        print(f"{p}: {n} rows held; {len(t)} rows; md5 {md5}")
        paths[g] = p
    if a.verify:
        verify(paths)
    if a.fanin:
        fanin(paths, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
