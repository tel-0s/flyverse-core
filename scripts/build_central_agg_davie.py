"""Per-cluster expression from the Davie et al. 2018 57k-cell 10x matrix (GEO GSE107451).

Writes <scratch>/davie_cluster_expr.parquet: rows = (annotation, metric), columns = every gene symbol;
metrics: mean_cp10k (mean of counts / cell_total * 1e4), mean_log1p_cp10k, frac_expr (fraction of cells
with >= 1 UMI). Cluster = the `annotation` column of the GEO metadata (paper's final annotation; numeric
labels are unannotated res.2 clusters). All ages pooled.
"""
import sys, time, numpy as np, pandas as pd, scipy.sparse as sp, scipy.io

root = r"D:\Projects\flyverse\data\external\central\davie2018_57k_mex"
out = sys.argv[1]
t = time.time()
M = scipy.io.mmread(root + r"\matrix.mtx").tocsc().astype(np.float32)  # genes x cells
print("mmread", M.shape, M.nnz, f"{time.time()-t:.0f}s", flush=True)
genes = pd.read_csv(root + r"\genes.tsv", sep="\t", header=None, names=["fbgn", "symbol"])
bc = pd.read_csv(root + r"\barcodes.tsv", header=None)[0]
meta = pd.read_csv(r"D:\Projects\flyverse\data\external\central\GSE107451_DGRP-551_w1118_WholeBrain_57k_Metadata.tsv.gz", sep="\t")
meta = meta.set_index("new_barcode").reindex(bc)
assert meta.annotation.notna().all(), meta.annotation.isna().sum()
ann = meta.annotation.astype(str).to_numpy()
tot = np.asarray(M.sum(axis=0)).ravel()
Mn = M.multiply(1e4 / tot[None, :]).tocsc()  # cp10k
Ml = Mn.copy(); Ml.data = np.log1p(Ml.data)
Mb = M.copy(); Mb.data = np.ones_like(Mb.data)
rows = []
labels = pd.unique(ann)
for lab in labels:
    cols = np.flatnonzero(ann == lab)
    n = len(cols)
    for name, X in (("mean_cp10k", Mn), ("mean_log1p_cp10k", Ml), ("frac_expr", Mb)):
        v = np.asarray(X[:, cols].sum(axis=1)).ravel() / n
        rows.append(pd.Series(v, index=genes.symbol.to_numpy(), name=(lab, name, n)))
df = pd.DataFrame(rows)
df.index = pd.MultiIndex.from_tuples(df.index, names=["annotation", "metric", "n_cells"])
df = df.reset_index()
df.to_parquet(out)
print("wrote", out, df.shape, f"{time.time()-t:.0f}s")
