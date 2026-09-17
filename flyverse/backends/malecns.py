"""MaleCNS v1.0 source reader; selection and row order preserved from f9e9fea."""
import pandas as pd
import pyarrow.feather as pf
from ..connectome import ANNOT_FILE, NT_FILE, WEIGHTS_FILE, KEEP_COLS, PHOTORECEPTOR_TYPES

def nt_table(data_dir) -> pd.DataFrame:
    nt = pd.read_feather(data_dir / NT_FILE)
    # best available label: consensus -> cell-type prediction -> per-body prediction
    best = nt.consensus_nt.where(nt.consensus_nt != "unclear")
    best = best.fillna(nt.celltype_predicted_nt.where(nt.celltype_predicted_nt != "unclear"))
    best = best.fillna(nt.predicted_nt.where(nt.predicted_nt != "unclear"))
    return pd.DataFrame({"bodyId": nt.body, "nt": best.fillna("unknown")})

def read(data_dir, *, edges="threshold", nt_threshold=0.5, log=print):
    ann = pd.read_feather(data_dir / ANNOT_FILE)
    is_pr = ann.type.isin(PHOTORECEPTOR_TYPES)
    keep = (ann.status == "Traced") | is_pr
    neurons = ann.loc[keep, KEEP_COLS].reset_index(drop=True)
    log(f"nodes: {len(neurons)} (traced {int((ann.status == 'Traced').sum())}, "
        f"+untraced photoreceptors {int((is_pr & (ann.status != 'Traced')).sum())})")
    neurons = neurons.merge(nt_table(data_dir), on="bodyId", how="left")
    log("loading weights ...")
    return neurons, pf.read_table(data_dir / WEIGHTS_FILE).to_pandas()
