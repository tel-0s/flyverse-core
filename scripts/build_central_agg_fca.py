"""Per-annotation expression from the Fly Cell Atlas 2022 head 10x stringent loom (s_fca_biohub_head_10x.loom).

Step 2 of 3 of the central-brain expression build (run order, all CPU, from the repo root):
    PYTHONIOENCODING=utf-8 python scripts/build_central_agg_davie.py   # -> data/external/central/derived/davie_cluster_expr.parquet
    PYTHONIOENCODING=utf-8 python scripts/build_central_agg_fca.py     # -> data/external/central/derived/fca_cluster_expr.parquet
    PYTHONIOENCODING=utf-8 python scripts/build_central_map.py         # -> flyverse/data/type_map_central.csv, expression_central.csv
Input (data/external/central/, git-ignored; `python scripts/fetch_data.py --external central`):
    s_fca_biohub_head_10x.loom  (/matrix = raw UMI counts, 13,056 filtered genes x 100,527 nuclei; col_attrs/annotation, col_attrs/sex)
Output: one parquet, rows = (annotation, metric, n_cells, sex_subset), columns = genes; metrics mean_cp10k, mean_log1p_cp10k,
frac_expr (as in the Davie aggregator). cp10k normalises by the per-nucleus total over the 13,056 retained genes, not the full
transcriptome, so FCA levels are inflated relative to Davie and comparable within source only.
sex_subset = 'all' (pooled) and 'male' (col_attrs/sex == 'male'). Streams the matrix in 4,096-nucleus blocks (~1 GB peak).
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "central"
LOOM = EXT / "s_fca_biohub_head_10x.loom"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else EXT / "derived" / "fca_cluster_expr.parquet"
BLOCK = 4096


def main() -> None:
    if not LOOM.exists():
        sys.exit(f"missing {LOOM}; run: python scripts/fetch_data.py --external central")
    t = time.time()
    f = h5py.File(LOOM, "r")
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
    for a in range(0, N, BLOCK):
        b = min(N, a + BLOCK)
        X = sp.csc_matrix(m[:, a:b])  # genes x cells
        tot = np.asarray(X.sum(axis=0)).ravel()
        tot[tot == 0] = 1
        Xn = X.multiply(1e4 / tot[None, :]).tocsc()
        Xl = Xn.copy()
        Xl.data = np.log1p(Xl.data)
        Xb = X.copy()
        Xb.data = np.ones_like(Xb.data)
        lab = li[a:b]
        male = sex[a:b] == "male"
        # one-hot cells x labels, then genes x labels via a sparse product
        for s, mask in ((0, np.ones(b - a, bool)), (1, male)):
            if not mask.any():
                continue
            H = sp.csr_matrix((np.ones(mask.sum(), np.float32), (np.flatnonzero(mask), lab[mask])), shape=(b - a, L))
            for k, Xk in (("cp10k", Xn), ("log1p", Xl), ("bin", Xb)):
                sums[k][s] += (Xk @ H).T.toarray()
            counts[s] += np.bincount(lab[mask], minlength=L)
        if (a // BLOCK) % 5 == 0:
            print(f"{b}/{N} {time.time() - t:.0f}s", flush=True)
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print("wrote", OUT, df.shape, f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
