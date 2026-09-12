"""Hashes of brain._shaped_weights on the cached connectome under explicit receptor settings (default-independent)."""
import hashlib, sys, time
sys.path.insert(0, r"D:\Projects\flyverse")
import numpy as np
from flyverse import connectome as cn
from flyverse.brain import LIFParams, _shaped_weights

t0 = time.time()
c = cn.load(verbose=False)
print("W content md5", hashlib.md5(c.W.data.tobytes()).hexdigest(), "nnz", c.W.nnz, "sum|W|", int(np.abs(c.W.data).sum()),
      "glutamate cells", int((c.neurons.nt == "glutamate").sum()))


def h(W):
    W = W.tocsr(); W.sort_indices()
    m = hashlib.md5(); m.update(W.data.tobytes()); m.update(W.indices.tobytes()); m.update(W.indptr.tobytes())
    return m.hexdigest(), W


for name, p in [("none", LIFParams(receptor_model=None)),
                ("sign_abs", LIFParams(receptor_model="sign", receptor_net_rule="abs")),
                ("default_now", LIFParams())]:
    t = time.time(); hx, W = h(_shaped_weights(c, p))
    print(f"{name:12s} md5 {hx} nnz {W.nnz} sum|W| {float(np.abs(W.data).sum()):.1f} dtype {W.data.dtype} ({time.time() - t:.1f} s)")
    if name == "none":
        W_none = W
    if name == "sign_abs":
        d = W.data != W_none.data
        print("  entries differing from none:", int(d.sum()), " sign flips:", int((np.sign(W.data) * np.sign(W_none.data) < 0).sum()),
              " zeroed:", int(((W.data == 0) & (W_none.data != 0)).sum()))
print("total", round(time.time() - t0, 1), "s")
