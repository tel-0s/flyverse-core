"""Load the MaleCNS v1.0 flat connectome tables and compile them into a simulation-ready graph.

Output (cached in cache/): a neuron table (one row per simulated body, index 0..N-1) and a CSR matrix
with rows = POSTsynaptic neuron, cols = PREsynaptic neuron, value = signed synapse count. Multiplying
it by a spike vector gives the synaptic input to every neuron in one shot.

Node set = every body with status "Traced", plus every photoreceptor body (R1-R6, R7*, R8*) regardless
of status (many photoreceptor axon fragments are untraced but carry real synapses onto the lamina).

Retinotopy: photoreceptors have no assignedOlHex1/2 in the annotations. The hex-column grid is annotated
on the columnar lamina/medulla cells (L1/L2/L3/L5, C2/C3, Mi1/Mi4/Mi9, Tm1/Tm2/Tm4/Tm9/Tm20, T1), so each
photoreceptor is assigned the column of its synapse-count-weighted strongest hexed postsynaptic partner.
Empirically the top-3 hexed partners of a photoreceptor agree on the column 94-99% of the time.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import scipy.sparse as sp

DATA_DIR = Path(os.environ.get("FLYVERSE_DATA", r"D:\Datasets\male-cns-connectome-v1.0\flat-connectome"))
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

ANNOT_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Sign convention (Shiu et al. 2024; stonkfly/doomfly): ACh + monoamines excitatory; GABA, glutamate
# (GluCl in fly) and histamine (photoreceptor -> lamina via HisCl) inhibitory.
# Departure from Shiu: neurons with NO usable NT prediction get sign 0 (no fast synaptic effect) instead
# of +1. The 2,494 "unknown" neurons include huge antennal-lobe local neurons (v2LN30, lLN2F_a, lLN2T_d;
# 25-50k output synapses each, almost certainly GABAergic); treating them as excitatory turned the whole
# central brain into a 300 Hz runaway (KCs, PAM DANs, PEN) from any sustained input.
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
    # Monoamines act through GPCRs on 100s-of-ms timescales; as fast +1 synapses (Shiu's convention) the
    # KC<->PAM dopamine loop (KCs synapse directly onto DAN axons) is a positive-feedback runaway.
    "dopamine": 0.0, "octopamine": 0.0, "serotonin": 0.0, "unknown": 0.0,
}
# Type-name overrides applied when the prediction is unknown: antennal-lobe local neurons are GABAergic
# as a class (a minority are glutamatergic/cholinergic, which the prediction would normally have caught).
UNKNOWN_NT_OVERRIDE_REGEX = {r"^(lLN|v2LN|v3LN|il3LN|l2LN|vLN|LN)": "gaba"}

PHOTORECEPTOR_TYPES = ["R1-R6", "R7y", "R7p", "R7d", "R7_unclear",
                       "R8y", "R8p", "R8d", "R8_unclear", "R7R8_unclear"]

KEEP_COLS = ["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "somaNeuromere",
             "status", "entryNerve", "exitNerve", "flywireType", "hemibrainType", "mancType",
             "assignedOlHex1", "assignedOlHex2"]


@dataclass
class Connectome:
    neurons: pd.DataFrame      # index 0..N-1; KEEP_COLS + nt, sign, hex1, hex2, hex_side, hex_source
    W: sp.csr_matrix           # (N, N), W[post, pre] = signed synapse count (float32)
    body_to_index: pd.Series   # bodyId -> row index

    @property
    def n(self) -> int:
        return len(self.neurons)

    def index_of(self, body_ids) -> np.ndarray:
        return self.body_to_index.loc[np.asarray(body_ids)].to_numpy()

    def select(self, **criteria) -> np.ndarray:
        """Row indices of neurons matching all criteria. Values may be scalars, lists, or regex strings
        prefixed with '~'  e.g. select(type='~^DNa0[12]', somaSide='L')."""
        m = np.ones(self.n, dtype=bool)
        for col, val in criteria.items():
            s = self.neurons[col]
            if isinstance(val, str) and val.startswith("~"):
                m &= s.fillna("").astype(str).str.contains(val[1:], regex=True).to_numpy()
            elif isinstance(val, (list, tuple, set, np.ndarray)):
                m &= s.isin(list(val)).to_numpy()
            else:
                m &= (s == val).to_numpy()
        return np.flatnonzero(m)

    def describe(self, idx) -> pd.DataFrame:
        return self.neurons.iloc[np.asarray(idx)][["bodyId", "type", "instance", "superclass", "somaSide", "nt"]]


def _nt_table() -> pd.DataFrame:
    nt = pd.read_feather(DATA_DIR / NT_FILE)
    # best available label: consensus -> cell-type prediction -> per-body prediction
    best = nt.consensus_nt.where(nt.consensus_nt != "unclear")
    best = best.fillna(nt.celltype_predicted_nt.where(nt.celltype_predicted_nt != "unclear"))
    best = best.fillna(nt.predicted_nt.where(nt.predicted_nt != "unclear"))
    return pd.DataFrame({"bodyId": nt.body, "nt": best.fillna("unknown")})


def compile_connectome(min_weight: int = 1, verbose: bool = True) -> Connectome:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    ann = pd.read_feather(DATA_DIR / ANNOT_FILE)
    is_pr = ann.type.isin(PHOTORECEPTOR_TYPES)
    keep = (ann.status == "Traced") | is_pr
    neurons = ann.loc[keep, KEEP_COLS].reset_index(drop=True)
    log(f"nodes: {len(neurons)} (traced {int((ann.status == 'Traced').sum())}, "
        f"+untraced photoreceptors {int((is_pr & (ann.status != 'Traced')).sum())})")

    neurons = neurons.merge(_nt_table(), on="bodyId", how="left")
    neurons["nt"] = neurons.nt.fillna("unknown")
    neurons.loc[neurons.type.isin(PHOTORECEPTOR_TYPES), "nt"] = "histamine"  # photoreceptors are histaminergic
    for pat, nt in UNKNOWN_NT_OVERRIDE_REGEX.items():
        m = (neurons.nt == "unknown") & neurons.type.fillna("").str.match(pat)
        neurons.loc[m, "nt"] = nt
        log(f"unknown-NT override {pat} -> {nt}: {int(m.sum())} neurons")
    neurons["sign"] = neurons.nt.map(NT_SIGN).fillna(0.0).astype(np.float32)
    log("nt counts:", neurons.nt.value_counts().to_dict())

    body_to_index = pd.Series(np.arange(len(neurons)), index=neurons.bodyId.to_numpy())

    log("loading weights ...")
    w = pf.read_table(DATA_DIR / WEIGHTS_FILE).to_pandas()
    log(f"  {len(w):,} rows in {time.time() - t0:.1f}s")
    pre = body_to_index.reindex(w.body_pre.to_numpy()).to_numpy()
    post = body_to_index.reindex(w.body_post.to_numpy()).to_numpy()
    m = ~np.isnan(pre) & ~np.isnan(post) & (w.weight.to_numpy() >= min_weight)
    pre = pre[m].astype(np.int64)
    post = post[m].astype(np.int64)
    cnt = w.weight.to_numpy()[m].astype(np.float32)
    del w
    log(f"edges kept: {len(pre):,}  synapses: {int(cnt.sum()):,}")

    sign = neurons.sign.to_numpy()
    W = sp.csr_matrix((cnt * sign[pre], (post, pre)), shape=(len(neurons), len(neurons)), dtype=np.float32)
    W.sum_duplicates()

    _assign_photoreceptor_columns(neurons, W, log)

    log(f"compiled in {time.time() - t0:.1f}s")
    return Connectome(neurons=neurons, W=W, body_to_index=body_to_index)


def _assign_photoreceptor_columns(neurons: pd.DataFrame, W: sp.csr_matrix, log) -> None:
    """Fill hex1/hex2/hex_side for every neuron; photoreceptors inherit the column of their strongest
    hexed postsynaptic partner (synapse counts summed over partners sharing a column)."""
    neurons["hex1"] = neurons.assignedOlHex1
    neurons["hex2"] = neurons.assignedOlHex2
    neurons["hex_side"] = neurons.somaSide.where(neurons.assignedOlHex1.notna())
    neurons["hex_source"] = np.where(neurons.assignedOlHex1.notna(), "annotation", "")

    hexed = neurons.assignedOlHex1.notna().to_numpy()
    pr_idx = np.flatnonzero(neurons.type.isin(PHOTORECEPTOR_TYPES).to_numpy())
    Wc = W.tocsc()  # columns = presynaptic
    h1 = neurons.hex1.to_numpy()
    h2 = neurons.hex2.to_numpy()
    hs = neurons.hex_side.to_numpy(dtype=object)
    n_ok = 0
    for i in pr_idx:
        lo, hi = Wc.indptr[i], Wc.indptr[i + 1]
        posts = Wc.indices[lo:hi]
        wts = np.abs(Wc.data[lo:hi])
        sel = hexed[posts]
        if not sel.any():
            continue
        posts, wts = posts[sel], wts[sel]
        votes = pd.DataFrame({"s": hs[posts], "a": h1[posts], "b": h2[posts], "w": wts}).groupby(["s", "a", "b"]).w.sum()
        s, a, b = votes.idxmax()
        neurons.at[i, "hex1"] = a
        neurons.at[i, "hex2"] = b
        neurons.at[i, "hex_side"] = s
        neurons.at[i, "hex_source"] = "partner"
        n_ok += 1
    log(f"photoreceptors assigned to hex columns: {n_ok}/{len(pr_idx)}")


def save(c: Connectome, cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    c.neurons.to_parquet(cache_dir / "neurons.parquet")
    sp.save_npz(cache_dir / "W_post_pre.npz", c.W, compressed=False)


def load(cache_dir: Path = CACHE_DIR, rebuild: bool = False, verbose: bool = True) -> Connectome:
    if rebuild or not (cache_dir / "W_post_pre.npz").exists():
        c = compile_connectome(verbose=verbose)
        save(c, cache_dir)
        return c
    neurons = pd.read_parquet(cache_dir / "neurons.parquet")
    W = sp.load_npz(cache_dir / "W_post_pre.npz").tocsr()
    return Connectome(neurons=neurons, W=W,
                      body_to_index=pd.Series(np.arange(len(neurons)), index=neurons.bodyId.to_numpy()))


if __name__ == "__main__":
    c = load(rebuild=True)
    print(c.neurons.head())
    print("N =", c.n, " nnz =", c.W.nnz)
