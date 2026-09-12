"""Independent CPU recomputation of the takeoff-hold report's mechanism claims (G.0):
per-type landing of each side, the flip / silencing split, and whether ANY changed entry lands on the
take-off pathway (DNp01, MN9, LC4, LPLC2 and the five formerly damped DNp01 inputs)."""
import numpy as np
import pandas as pd

from flyverse import connectome as C

c = C.load(verbose=False)
W = c.W.tocoo()
print(f"connectome: {c.W.nnz:,} stored entries, sum|W| {np.abs(c.W.data).sum():,.0f}")
base = np.sign(c.W.data)

TAB = {
    "default":   None,
    "holdBrain": r"C:/Users/ethee/AppData/Local/Temp/claude/D--Projects-flyverse/d280c0e1-e89c-49ce-943c-279a614bcc17/scratchpad/holdtab/receptors_holdBrain.csv",
    "holdOptic": r"C:/Users/ethee/AppData/Local/Temp/claude/D--Projects-flyverse/d280c0e1-e89c-49ce-943c-279a614bcc17/scratchpad/holdtab/receptors_holdOptic.csv",
}
masks, signs = {}, {}
for name, path in TAB.items():
    r = C.receptor_signs(c, table_path=path, net_rule="abs")
    fs = r.fast_sign
    masks[name] = fs != base
    signs[name] = fs
    print(f"{name:10s} changed {int(masks[name].sum()):,} |W| {np.abs(c.W.data)[masks[name]].sum():,.0f}")

n = c.neurons
post_type = n["malecns_type"].to_numpy() if "malecns_type" in n else n["type"].to_numpy()
cols = list(n.columns)
print("neuron columns:", cols[:20])
tcol = "malecns_type" if "malecns_type" in cols else ("type" if "type" in cols else None)
scol = "superclass" if "superclass" in cols else None
types = n[tcol].astype(str).to_numpy()
rowsc = n[scol].astype(str).to_numpy() if scol else None

wabs = np.abs(c.W.data)
df = pd.DataFrame({"post": types[W.row], "pre": types[W.col], "w": wabs,
                   "sc": rowsc[W.row] if rowsc is not None else "?",
                   "old": base, "newD": signs["default"], "newHB": signs["holdBrain"], "newHO": signs["holdOptic"],
                   "mD": masks["default"], "mHB": masks["holdBrain"], "mHO": masks["holdOptic"]})

print("\n== partition ==")
print("  both:", int((df.mHB & df.mHO).sum()), " neither-but-default:",
      int((df.mD & ~df.mHB & ~df.mHO).sum()), " union==default:", bool((df.mD == (df.mHB | df.mHO)).all()))

print("\n== per postsynaptic type, optic side (holdBrain arm) ==")
g = df[df.mHB].groupby("post").agg(entries=("w", "size"), W=("w", "sum")).sort_values("W", ascending=False)
print(g.head(15).to_string())
print("\n== per postsynaptic type, Brain side (holdOptic arm) ==")
g2 = df[df.mHO].groupby("post").agg(entries=("w", "size"), W=("w", "sum")).sort_values("W", ascending=False)
print(g2.head(15).to_string())

print("\n== flip / silencing split ==")
for side, m, key in (("optic", df.mHB, "newHB"), ("brain", df.mHO, "newHO")):
    sub = df[m]
    for newv in (0.0, 1.0):
        s = sub[sub[key] == newv]
        print(f"  {side:6s} -1 -> {newv:.0f}: {len(s):6d} entries  {s.w.sum():9.0f} |W|")

print("\n== the take-off pathway: input entries of each type and how many the DEFAULT changes ==")
for t in ("DNp01", "MN9", "LC4", "LPLC2", "DNp70", "SAD073", "GNG300", "CL367", "PVLP010"):
    cells = int((types == t).sum())
    sub = df[df.post == t]
    print(f"  {t:9s} cells {cells:4d}  input entries {len(sub):7d}  changed default {int(sub.mD.sum()):4d} "
          f"optic {int(sub.mHB.sum()):4d} brain {int(sub.mHO.sum()):4d}")

print("\n== every changed entry whose postsynaptic superclass is descending_neuron or motor ==")
sub = df[df.mD & df.sc.isin(["descending_neuron", "motor_neuron", "vnc_neuron", "efferent"])]
print(sub.groupby(["sc", "post"]).agg(entries=("w", "size"), W=("w", "sum")).to_string())
print("  totals:", len(sub), "entries", sub.w.sum(), "|W|; on the Brain side:", int(sub.mHO.sum()),
      "on the optic side:", int(sub.mHB.sum()))

print("\n== any changed entry PREsynaptic to DNp01 / MN9 (i.e. on a cell that drives them)? ==")
for t in ("DNp01", "MN9"):
    pre_of = df[(df.post == t)]
    drivers = sorted({str(v) for v in pre_of.pre})
    chg = df[df.mD & df.pre.isin(drivers)]
    print(f"  {t}: {len(drivers)} presynaptic types; changed entries anywhere onto those driver types: "
          f"{int(df[df.mD & df.post.isin(drivers)].w.size)} ({df[df.mD & df.post.isin(drivers)].w.sum():.0f} |W|)")
    top = df[df.mD & df.post.isin(drivers)].groupby("post").agg(e=("w", "size"), W=("w", "sum")).sort_values("W", ascending=False)
    print(top.head(12).to_string())
