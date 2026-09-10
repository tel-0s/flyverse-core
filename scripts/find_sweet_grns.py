"""Identify putative sugar-sensing GRNs by their connectivity to the known sweet second-order neurons.

Known sweet-pathway interneurons (Shiu et al. 2022/2024, Sterne et al. 2021) appear in the annotations
under their hemibrain names in `synonyms`: G2N-1 = GNG232, Rattle = GNG132, Usnea = GNG175, Phantom =
GNG229, Bract = DNge173/174, Clavicle = ANXXX462a, Zorro = GNG215, Fudog = DNg67, Roundup = GNG108.
Sugar GRNs (Gr64f/Gr5a) are their dominant GRN inputs; bitter GRNs (Gr66a) target other cells
(e.g. Bitter = DNg28, Scapula = GNG087). We score every GRN by synapses onto sweet vs bitter targets.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import connectome  # noqa: E402

SWEET = ["GNG232", "GNG132", "GNG175", "GNG229", "DNge173", "DNge174", "ANXXX462a", "GNG215", "DNg67", "GNG108"]
BITTER = ["DNg28", "GNG087"]


def main():
    c = connectome.load(verbose=False)
    n = c.neurons
    types = n.type.fillna("").to_numpy()
    grn = np.flatnonzero((n["class"] == "gustatory").to_numpy())
    Wabs = abs(c.W)
    sweet_t = c.select(type=SWEET)
    bitter_t = c.select(type=BITTER)
    print("sweet targets:", n.iloc[sweet_t].groupby("type").size().to_dict())
    print("bitter targets:", n.iloc[bitter_t].groupby("type").size().to_dict())
    s = np.asarray(Wabs[sweet_t][:, grn].sum(axis=0)).ravel()
    b = np.asarray(Wabs[bitter_t][:, grn].sum(axis=0)).ravel()
    out = np.asarray(Wabs[:, grn].sum(axis=0)).ravel()
    df = pd.DataFrame({"idx": grn, "type": types[grn], "subclass": n.subclass.to_numpy()[grn], "nerve": n.entryNerve.to_numpy()[grn],
                       "side": n.somaSide.to_numpy()[grn], "sweet": s, "bitter": b, "out": out})
    print("\nGRN types by synapses onto sweet targets:")
    g = df.groupby(["type", "subclass"]).agg(n=("idx", "size"), sweet=("sweet", "sum"), bitter=("bitter", "sum"), out=("out", "sum"))
    g["sweet_frac"] = g.sweet / g.out.clip(lower=1)
    print(g.sort_values("sweet", ascending=False).head(25).round(3).to_string())
    # per-cell: sweet GRNs = at least 3 synapses onto sweet targets and sweet > 2*bitter
    sw = df[(df.sweet >= 3) & (df.sweet > 2 * df.bitter)]
    print(f"\nputative sweet GRNs: {len(sw)} cells: " + str(sw.groupby(["subclass", "nerve"]).size().to_dict()))
    bi = df[(df.bitter >= 3) & (df.bitter > 2 * df.sweet)]
    print(f"putative bitter GRNs: {len(bi)} cells: " + str(bi.groupby(["subclass", "nerve"]).size().to_dict()))
    os.makedirs("cache", exist_ok=True)
    sw_ids = n.bodyId.to_numpy()[sw.idx.to_numpy()]
    bi_ids = n.bodyId.to_numpy()[bi.idx.to_numpy()]
    pd.DataFrame({"bodyId": np.r_[sw_ids, bi_ids], "taste": ["sweet"] * len(sw_ids) + ["bitter"] * len(bi_ids)}).to_csv("flyverse/data/taste_grns.csv", index=False)
    print("wrote flyverse/data/taste_grns.csv")


if __name__ == "__main__":
    os.makedirs("flyverse/data", exist_ok=True)
    main()
