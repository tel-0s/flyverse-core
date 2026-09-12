"""Per-cluster expression from the Davie et al. 2018 57k-cell whole-brain 10x matrix (GEO GSE107451).

Step 1 of 3 of the central-brain expression build (run order, all CPU, from the repo root):
    PYTHONIOENCODING=utf-8 python scripts/build_central_agg_davie.py   # -> data/external/central/derived/davie_cluster_expr.parquet
    PYTHONIOENCODING=utf-8 python scripts/build_central_agg_fca.py     # -> data/external/central/derived/fca_cluster_expr.parquet
    PYTHONIOENCODING=utf-8 python scripts/build_central_map.py         # -> flyverse/data/type_map_central.csv, expression_central.csv,
                                                                       #    out/coverage_central.md, out/nt_crosscheck_central.md
Inputs (data/external/central/, git-ignored; `python scripts/fetch_data.py --external central` downloads and unpacks them):
    davie2018_57k_mex/{matrix.mtx, genes.tsv, barcodes.tsv}   (17,473 genes x 56,902 cells, integer UMI, from the GEO MEX tarball)
    GSE107451_DGRP-551_w1118_WholeBrain_57k_Metadata.tsv.gz  (per-cell `annotation` = the paper's final cluster labels)
Output: one parquet, rows = (annotation, metric, n_cells), columns = every gene symbol; metrics
    mean_cp10k       = mean over cells of counts / cell_total * 1e4 (cell_total over all 17,473 genes)
    mean_log1p_cp10k = mean over cells of log1p(cp10k)
    frac_expr        = fraction of cells with >= 1 UMI
Cluster = the `annotation` column (75 named labels + 41 numeric labels = res.2 clusters left 'Unannotated' in Table S2).
All ages (0-50 d), both sexes and both genotypes pooled. Peak memory ~4 GB (four float32 copies of the 70.8 M-nonzero matrix).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "central"
MEX = EXT / "davie2018_57k_mex"
META = EXT / "GSE107451_DGRP-551_w1118_WholeBrain_57k_Metadata.tsv.gz"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else EXT / "derived" / "davie_cluster_expr.parquet"


def main() -> None:
    for p in (MEX / "matrix.mtx", MEX / "genes.tsv", MEX / "barcodes.tsv", META):
        if not p.exists():
            sys.exit(f"missing {p}; run: python scripts/fetch_data.py --external central")
    t = time.time()
    M = scipy.io.mmread(MEX / "matrix.mtx").tocsc().astype(np.float32)  # genes x cells
    print("mmread", M.shape, M.nnz, f"{time.time() - t:.0f}s", flush=True)
    genes = pd.read_csv(MEX / "genes.tsv", sep="\t", header=None, names=["fbgn", "symbol"])
    bc = pd.read_csv(MEX / "barcodes.tsv", header=None)[0]
    meta = pd.read_csv(META, sep="\t").set_index("new_barcode").reindex(bc)
    assert meta.annotation.notna().all(), meta.annotation.isna().sum()
    ann = meta.annotation.astype(str).to_numpy()
    tot = np.asarray(M.sum(axis=0)).ravel()
    Mn = M.multiply(1e4 / tot[None, :]).tocsc()  # cp10k
    Ml = Mn.copy()
    Ml.data = np.log1p(Ml.data)
    Mb = M.copy()
    Mb.data = np.ones_like(Mb.data)
    rows = []
    for lab in pd.unique(ann):
        cols = np.flatnonzero(ann == lab)
        n = len(cols)
        for name, X in (("mean_cp10k", Mn), ("mean_log1p_cp10k", Ml), ("frac_expr", Mb)):
            v = np.asarray(X[:, cols].sum(axis=1)).ravel() / n
            rows.append(pd.Series(v, index=genes.symbol.to_numpy(), name=(lab, name, n)))
    df = pd.DataFrame(rows)
    df.index = pd.MultiIndex.from_tuples(df.index, names=["annotation", "metric", "n_cells"])
    df = df.reset_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print("wrote", OUT, df.shape, f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
