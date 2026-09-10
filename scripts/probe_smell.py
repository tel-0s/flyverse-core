"""Diagnostic: olfactory input (spontaneous ORN rate, then fruit odours) -> antennal lobe -> mushroom
body / lateral horn -> descending neurons. Checks for runaway in the antennal-lobe local neurons.

    python scripts/probe_smell.py [--alpha 1.0] [--base 8]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import brain, connectome, olfaction  # noqa: E402

PROBE = ["DM1_lPN", "DM2_lPN", "VA2_adPN", "DM4_adPN", "DL5_adPN", "lLN1_bc", "lLN2F_b", "v2LN30", "lLN2P_b", "MBON01",
         "KCab-m", "KCg-m", "PAM01", "DNa02", "MDN", "DNp09", "MN9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--base", type=float, default=8.0)
    ap.add_argument("--eln-zero", action="store_true", help="cholinergic antennal-lobe LNs: outputs sign 0")
    args = ap.parse_args()
    c = connectome.load(verbose=False)
    n = c.neurons
    olf = olfaction.Olfaction(c, [("apple", (0.25, 0.15, 0.79), 1.0), ("banana", (0.05, -0.22, 0.77), 1.0)],
                              olfaction.OlfactionParams(base_hz=args.base))
    b = brain.Brain(c, brain.LIFParams(input_norm_alpha=args.alpha))
    if args.eln_zero:
        import numpy as np, scipy.sparse as sp, torch
        m = n.type.fillna("").str.match(r"^(lLN|v2LN|v3LN|il3LN|l2LN)") & (n.nt == "acetylcholine")
        print("zeroing outputs of", int(m.sum()), "cholinergic AL LNs")
        W = (c.W @ sp.diags((~m.to_numpy()).astype(np.float32))).tocsr()
        b.W = torch.sparse_csr_tensor(torch.from_numpy(W.indptr.astype(np.int64)), torch.from_numpy(W.indices.astype(np.int64)),
                                      torch.from_numpy(W.data * np.float32(b.p.w_syn)), size=W.shape).to(b.device)
    print(f"alpha={args.alpha} base={args.base} Hz, {len(olf.orn_idx)} ORNs")

    def rep(lab):
        rt = b.rate.cpu().numpy()
        g = n.assign(rate=rt).groupby("type").rate.agg(["mean", "size"])
        print(f"[{lab}] spikes/step {b.total_spikes():.0f} frac>1Hz {(rt > 1).mean():.3f} | "
              + " ".join(f"{t}={rt[c.select(type=t)].mean():.0f}" for t in PROBE))
        print("    top:", g[g["size"] >= 4].sort_values("mean", ascending=False).head(8)["mean"].round(0).to_dict())

    for pos, lab in [((-0.5, 0.05, 0.75), "far"), ((0.19, 0.15, 0.75), "next to apple"), ((0.05, -0.15, 0.75), "next to banana")]:
        olf.apply(b, pos)
        b.run_ms(800)
        print("odour:", olf.summary())
        rep(lab)


if __name__ == "__main__":
    main()
