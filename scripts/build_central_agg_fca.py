"""Per-annotation expression from the Fly Cell Atlas head 10x stringent loom (s_fca_biohub_head_10x.loom).

/matrix is raw UMI counts (13,056 filtered genes x 100,527 nuclei). Writes <scratch>/fca_cluster_expr.parquet:
rows = (annotation, metric, n_cells, sex_subset), columns = genes; metrics: mean_cp10k, mean_log1p_cp10k,
frac_expr. cp10k normalises by the per-nucleus total over the 13,056 retained genes (not the full transcriptome).
sex_subset = 'all' (pooled) and 'male' (col_attrs/sex == 'male').
"""
import sys, time, numpy as np, pandas as pd, scipy.sparse as sp, h5py

path = r"D:\Projects\flyverse\data\external\central\s_fca_biohub_head_10x.loom"
out = sys.argv[1]
t = time.time()
f = h5py.File(path, "r")
m = f["matrix"]
G, N = m.shape
genes = f["row_attrs/Gene"][:].astype(str)
ann = f["col_attrs/annotation"][:].astype(str)
sex = f["col_attrs/sex"][:].astype(str)
labels = pd.unique(ann)
lab_idx = {l: i for i, l in enumerate(labels)}
li = np.array([lab_idx[a] for a in ann])
L = len(labels)
sums = {k: np.zeros((2, L, G), np.float64) for k in ("cp10k", "log1p", "bin")}
counts = np.zeros((2, L), np.int64)
B = 4096
for a in range(0, N, B):
    b = min(N, a + B)
    X = sp.csc_matrix(m[:, a:b])  # genes x cells
    tot = np.asarray(X.sum(axis=0)).ravel()
    tot[tot == 0] = 1
    Xn = X.multiply(1e4 / tot[None, :]).tocsc()
    Xl = Xn.copy(); Xl.data = np.log1p(Xl.data)
    Xb = X.copy(); Xb.data = np.ones_like(Xb.data)
    lab = li[a:b]
    male = sex[a:b] == "male"
    # one-hot cells x labels, then genes x labels via sparse product
    for s, mask in ((0, np.ones(b - a, bool)), (1, male)):
        if not mask.any():
            continue
        H = sp.csr_matrix((np.ones(mask.sum(), np.float32), (np.flatnonzero(mask), lab[mask])), shape=(b - a, L))
        for k, Xk in (("cp10k", Xn), ("log1p", Xl), ("bin", Xb)):
            sums[k][s] += (Xk @ H).T.toarray()
        counts[s] += np.bincount(lab[mask], minlength=L)
    if (a // B) % 5 == 0:
        print(f"{b}/{N} {time.time()-t:.0f}s", flush=True)
rows = []
for s, sname in ((0, "all"), (1, "male")):
    for j, lab in enumerate(labels):
        n = counts[s, j]
        if n == 0:
            continue
        for k, mname in (("cp10k", "mean_cp10k"), ("log1p", "mean_log1p_cp10k"), ("bin", "frac_expr")):
            rows.append(pd.Series(sums[k][s, j] / n, index=genes, name=(lab, mname, int(n), sname)))
df = pd.DataFrame(rows)
df.index = pd.MultiIndex.from_tuples(df.index, names=["annotation", "metric", "n_cells", "sex_subset"])
df = df.reset_index()
df.to_parquet(out)
print("wrote", out, df.shape, f"{time.time()-t:.0f}s")
