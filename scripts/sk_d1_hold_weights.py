"""Independent recomputation of the four shaped-weight md5s the takeoff-hold report quotes
(out/d1_hold_weights.log), with the same recipe tests/test_receptor_model.py uses."""
import hashlib

import numpy as np

from flyverse import connectome as C
from flyverse.brain import LIFParams, _shaped_weights

SC = r"C:/Users/ethee/AppData/Local/Temp/claude/D--Projects-flyverse/d280c0e1-e89c-49ce-943c-279a614bcc17/scratchpad/holdtab"


def md5(W):
    W = W.tocsr(); W.sort_indices()
    m = hashlib.md5()
    m.update(W.data.tobytes()); m.update(W.indices.tobytes()); m.update(W.indptr.tobytes())
    return m.hexdigest()


c = C.load(verbose=False)
print(f"cache nnz {c.W.nnz:,} sum|W| {np.abs(c.W.data).sum():,.0f} glutamate cells {int((c.neurons.nt=='glutamate').sum()):,}")
arms = {
    "none":             LIFParams(receptor_model=None),
    "sign_abs_shipped": LIFParams(),
    "holdBrain":        LIFParams(receptor_table=SC + "/receptors_holdBrain.csv"),
    "holdOptic":        LIFParams(receptor_table=SC + "/receptors_holdOptic.csv"),
}
hs = {}
for name, p in arms.items():
    W = _shaped_weights(c, p)
    hs[name] = md5(W)
    print(f"{name:18s} md5 {hs[name]} nnz {W.nnz:,} sum|W| {np.abs(W.tocsr().data).sum():,.1f}")

EXPECT = {"none": "fcb5bec2a6c492196a622e31cdb24fc6", "sign_abs_shipped": "0e30e4a80cb607d4a168d1b08ebd6a40",
          "holdBrain": "022894e72a0a702ceb6e0c9ccad48df1", "holdOptic": "40fd50c7aa887a62dca585ebaca3c0dd"}
for k, v in EXPECT.items():
    print(f"  {k:18s} report {v}  mine {hs[k]}  {'MATCH' if hs[k]==v else 'MISMATCH'}")

# the partition on the SHAPED weights, as d1_hold_weights.log reports it
base = _shaped_weights(c, arms["none"]).tocsr(); base.sort_indices()
res = {}
for name in ("sign_abs_shipped", "holdBrain", "holdOptic"):
    Wn = _shaped_weights(c, arms[name]).tocsr(); Wn.sort_indices()
    d = Wn.data != base.data
    res[name] = d
    print(f"shaped-weight diff vs none: {name:18s} entries {int(d.sum()):6d}  |W| {np.abs(base.data)[d].sum():9.0f}")
print("overlap holdBrain & holdOptic on shaped weights:", int((res['holdBrain'] & res['holdOptic']).sum()))
print("union == default:", bool(((res['holdBrain'] | res['holdOptic']) == res['sign_abs_shipped']).all()))
