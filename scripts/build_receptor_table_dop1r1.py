"""CPU check: the --dopamine-lead dop1r1 in-memory rebuild (working-tree build_receptor_table.py, flip rule 'any')
reproduces the shipped flyverse/data/receptors_by_type.csv on every fast column (the abs fast weights the cluster runs
use) and differs only in the dopamine slow columns. Writes out/receptors_r3_dop1r1.csv for the per-cell analysis."""
import os, sys, time, hashlib
import numpy as np, pandas as pd
ROOT = r"D:\Projects\flyverse"
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import flyverse.connectome as cn
import build_receptor_table as brt

t0 = time.time()
c = cn.load(verbose=False)
print(f"connectome {c.n:,} neurons, {c.W.nnz:,} entries, sum|W| {int(np.abs(c.W.data).sum()):,} ({time.time()-t0:.1f}s)")
shipped_path = os.path.join(ROOT, "flyverse", "data", "receptors_by_type.csv")
print("shipped table md5", hashlib.md5(open(shipped_path, "rb").read()).hexdigest())
old = pd.read_csv(shipped_path, comment="#", low_memory=False)
print("shipped rows", len(old), "flip rule line:", [l for l in open(shipped_path, encoding="utf-8") if "flip-rule" in l][:1])

# 1. unpatched rebuild (no raw weights) == shipped on sign/net/class columns?
_, rec0, *_ = brt.build_tables(c, None, log=lambda *a, **k: None)
print(f"unpatched rebuild {len(rec0)} rows ({time.time()-t0:.1f}s)")
# 2. patched rebuild (dop1r1)
brt.RECEPTOR_GROUPS["dopamine"]["slow"] = [(+1, "Dop1R", ["Dop1R1", "Dop1R2"]), (-1, "Dop2R", ["Dop2R"])]
_, rec1, *_ = brt.build_tables(c, None, log=lambda *a, **k: None)
print(f"dop1r1 rebuild {len(rec1)} rows ({time.time()-t0:.1f}s)")

def cmp(a, b, cols, label):
    a = a.sort_values(["malecns_type", "transmitter"]).reset_index(drop=True)
    b = b.sort_values(["malecns_type", "transmitter"]).reset_index(drop=True)
    assert len(a) == len(b), (len(a), len(b))
    assert (a.malecns_type.astype(str).to_numpy() == b.malecns_type.astype(str).to_numpy()).all()
    out = {}
    for col in cols:
        if col not in a.columns or col not in b.columns:
            out[col] = "missing"; continue
        x = a[col].fillna("").astype(str).to_numpy(); y = b[col].fillna("").astype(str).to_numpy()
        try:
            xf = pd.to_numeric(a[col], errors="coerce"); yf = pd.to_numeric(b[col], errors="coerce")
            if xf.notna().sum() == a[col].notna().sum() and yf.notna().sum() == b[col].notna().sum():
                d = ~np.isclose(xf.to_numpy(float), yf.to_numpy(float), equal_nan=True)
                out[col] = int(d.sum()); continue
        except Exception:
            pass
        out[col] = int((x != y).sum())
    print(label, out)
    return out

fast_cols = [c_ for c_ in old.columns if c_.startswith("fast_")] + ["tier", "source", "n_sources"]
slow_cols = [c_ for c_ in old.columns if c_.startswith("slow_")]
cmp(rec0, old, fast_cols + slow_cols, "unpatched rebuild vs shipped:")
r = cmp(rec1, old, fast_cols + slow_cols, "dop1r1 rebuild vs shipped:")
d = rec1[(rec1.transmitter == "dopamine") & ~rec1.malecns_type.astype(str).str.startswith("<")]
print("dop1r1 dopamine rows", len(d), "slow_net", d.slow_net.value_counts().to_dict(), "+lead", d.slow_pos_lead.fillna("").value_counts().to_dict())
d0 = old[(old.transmitter == "dopamine") & ~old.malecns_type.astype(str).str.startswith("<")]
print("shipped dopamine rows", len(d0), "slow_net", d0.slow_net.value_counts().to_dict(), "+lead", d0.slow_pos_lead.fillna("").value_counts().to_dict())
# abs flips in the shipped table
g = old[(old.transmitter == "glutamate") & ~old.malecns_type.astype(str).str.startswith("<")]
print("shipped glutamate rows", len(g), "fast_net_abs", g.fast_net_abs.value_counts().to_dict())
print("shipped glutamate fast_sign_abs +1 rows:", sorted(g.loc[g.fast_sign_abs == 1, "malecns_type"].tolist()))
out = os.path.join(ROOT, "out", "receptors_r3_dop1r1.csv")
with open(out, "w", encoding="utf-8") as f:
    f.write("# receptors_by_type.csv rebuilt by scripts/build_receptor_table_dop1r1.py: dopamine slow + group = Dop1R1 / Dop1R2 only; working-tree builder (flip rule any)\n")
    rec1.to_csv(f, index=False)
print("wrote", out, f"({time.time()-t0:.1f}s)")
