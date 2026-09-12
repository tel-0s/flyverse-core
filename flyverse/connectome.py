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
from dataclasses import dataclass, field
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
# Type-level transmitter overrides applied AFTER the MaleCNS consensus (and after the unknown-NT regex above) to the
# cells of the type that have NO usable fast label -- consensus 'unknown' or a monoamine, i.e. sign 0 under NT_SIGN;
# cells with a classical consensus (ACh / GABA / glutamate / histamine) keep it. From the receptor-integration
# cross-check (docs/audits/receptor_nt_disagreements.md section 4, docs/audits/receptor_sources_nern2025.md;
# docs/NT_INTEGRATION.md round 2). Each entry names its evidence. Applied by compile_connectome(); whether it is part
# of the default cache is decided by a benchmark run (the round-2 rule: adopt only if no suite check changes status)
# -- see docs/audits/nt_audit.md and the TYPE_NT_OVERRIDE_DEFAULT switch below.
TYPE_NT_OVERRIDE = {
    # 477 cells: 209 glutamate, 177 acetylcholine, 91 'unknown' (sign 0; 26,208 raw output synapses) in the consensus.
    # Transcriptome: Davie 2018 cluster 'TmY14' VGlut fraction 0.98, FCA 2022 and Özel 2021 TmY14 clusters VGlut on;
    # Nern 2025 Sup. Table 1 TmY14 = glutamate. Three transcriptome sources + the EM classifier agree. Only the 91
    # unknown cells are relabelled; the 177 cells with an acetylcholine consensus keep it (relabelling them would flip
    # the sign of 177 cells' output and is a separate decision, not taken here).
    "TmY14": "glutamate",
    # 12 cells, all 'unknown' (24,996 raw output synapses; FlyWire label dopamine). Nern 2025 predicts serotonin and
    # validated it by EASI-FISH (Sup. Table 1, validated_nt). Serotonin is sign 0 under NT_SIGN, so the weights do not
    # change until a working slow term exists; the label is set so the receptor model's monoamine columns apply.
    "Mi19": "serotonin",
    # 4 cells, 3 'unknown' + 1 serotonin (7,238 raw output synapses). Nern 2025 predicts acetylcholine (aMe8 in the
    # optic-lobe classifier's scope); no transcriptome profile exists for the type.
    "aMe8": "acetylcholine",
    # NOT overridden: T1 (1,777 cells) keeps the MaleCNS consensus 'histamine' (confidence 0.51) although the label is
    # marker-silent in all four transcriptome sources (Davis 2020 T1 Hdc 0.61 / ChAT 3.04 / VGlut 2.48 / Gad1 1.59 TPM;
    # Özel 2021 P(on) 0 for ChAT / VAChT / Gad1 / VGlut / Hdc; Davie 2018 and FCA 2022 fractions < 0.2) and Nern 2025
    # says 'unclear' -- no source supports an alternative label, so there is nothing to replace it with.
}
# Whether compile_connectome() applies TYPE_NT_OVERRIDE when no explicit table is passed (the default cache). Decided
# in round 2 (docs/audits/nt_audit.md): adopted. The override relabels 107 cells (91 TmY14 + 12 Mi19 + 4 aMe8), changes
# the sign of 95 cells' output (33,446 raw synapses; sign-0 share 2.203 % -> 2.176 %), and the cluster benchmark
# (--seeds 0,1,2, out/rm_ntov.json vs out/rm_off.json) shows every check at the same status as the reference off run
# (26 PASS / 1 FAIL walk.power_max / 2 KNOWN GAP); the only status difference against a concurrent baseline run
# (out/rm_base_r2.json) is loom_escape, whose GF peak straddles its 33 Hz threshold run-to-run in both conditions
# (out/lo_base_*.json, lo_ov_*.json). A scratch cache without the table: load(cache_dir=..., rebuild=True,
# type_nt_override={}). Existing caches (local cache/, the cluster's shared cache) carry the table only once rebuilt.
TYPE_NT_OVERRIDE_DEFAULT = True

PHOTORECEPTOR_TYPES = ["R1-R6", "R7y", "R7p", "R7d", "R7_unclear",
                       "R8y", "R8p", "R8d", "R8_unclear", "R7R8_unclear"]

# ---------------------------------------------------------------------------------------------- receptor model
# Optional per-(postsynaptic type, presynaptic transmitter) response classes from the transcriptomic receptor table
# (flyverse/data/receptors_by_type.csv, built by scripts/build_receptor_table.py; rules and coverage in
# docs/audits/receptor_rules.md). Off unless LIFParams.receptor_model is set; see receptor_signs().
RECEPTOR_TABLE = Path(__file__).resolve().parent / "data" / "receptors_by_type.csv"
TRANSMITTERS = ["acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "octopamine", "serotonin"]
RECEPTOR_TIERS = ["fallback", "pre_unknown", "nt_class", "class", "fuzzy", "alias", "exact"]   # int8 codes 0..6
GAIN_CLASSES = ["none", "low", "mid", "high"]                                                   # int8 codes 0..3
RECEPTOR_NET_RULES = ("class", "abs", "nonmda")
# Slow-term classes (int8 codes 0..2), decided by the PRESYNAPTIC transmitter of the entry: the classical transmitters'
# metabotropic receptors (mAChR-A/-B, GABA-B, mGluR) and the monoamine receptors (Dop1R/DopEcR/Dop2R, Oamb/Octbeta/
# Octalpha2R, 5-HT1/2/7) get separate scales and time constants in LIFParams (docs/audits/slow_term.md); 'none' = no
# slow sign. Histamine has no slow group, so it never carries a class.
SLOW_CLASSES = ["none", "metabotropic_classical", "monoamine"]
SLOW_CLASS_OF_TRANSMITTER = {"acetylcholine": 1, "gaba": 1, "glutamate": 1, "histamine": 1,
                             "dopamine": 2, "octopamine": 2, "serotonin": 2}
SIGN0_COUNTS_FILE = "sign0_counts.npz"


@dataclass
class ReceptorSigns:
    """Per stored entry of a connectome's W (CSR order, explicit zeros included): the receptor model's
    fast sign, slow sign, gain classes and match tier. Produced by receptor_signs(); consumed by
    brain._shaped_weights, brain._slow_weights and optic.OpticLobe.

    fast_sign: +1 / -1 / 0 (float32). Matched entries (post type x pre transmitter has a table row) take the
        table's fast sign; every other entry keeps NT_SIGN of the presynaptic cell, so that
        abs(W.data) * fast_sign == W.data there exactly.
    slow_sign: +1 / -1 / 0; 0 for unmatched entries.
    fast_gain, slow_gain: int8 codes into GAIN_CLASSES (0 = none, also for unmatched entries).
    tier: int8 codes into RECEPTOR_TIERS. matched = the sign came from the table (tier nt_class or better).
    count: synapse count per entry (float32): abs(W.data), plus the raw count for explicit-zero entries when
        sign-0 counts were available (see sign0_counts()); None when receptor_signs(with_counts=False).
    slow_class: int8 codes into SLOW_CLASSES (0 wherever slow_sign == 0; else 1 for a classical presynaptic
        transmitter, 2 for a monoamine).
    """
    fast_sign: np.ndarray
    slow_sign: np.ndarray
    fast_gain: np.ndarray
    slow_gain: np.ndarray
    tier: np.ndarray
    count: np.ndarray | None
    net_rule: str
    nt_class_fallback: bool
    table_path: str
    slow_class: np.ndarray | None = None

    @property
    def matched(self) -> np.ndarray:
        return self.tier >= RECEPTOR_TIERS.index("nt_class")

    def fast_factor(self, gain: dict | None = None) -> np.ndarray:
        """Per-entry multiplier for abs(W.data): fast_sign, times the gain-class factor when `gain`
        ({class_name: factor}) is given (classes absent from the dict get 1)."""
        f = self.fast_sign.astype(np.float32)
        if gain:
            g = np.array([float(gain.get(k, 1.0)) for k in GAIN_CLASSES], dtype=np.float32)
            f = f * g[self.fast_gain]
        return f

    def slow_factor(self, gain: dict | None = None, slow_class: str | None = None) -> np.ndarray:
        """Per-entry slow multiplier: slow_sign (x the gain-class factor when `gain` is given), restricted to the
        entries of one SLOW_CLASSES name when `slow_class` is given (the others become 0)."""
        f = self.slow_sign.astype(np.float32)
        if gain:
            g = np.array([float(gain.get(k, 1.0)) for k in GAIN_CLASSES], dtype=np.float32)
            f = f * g[self.slow_gain]
        if slow_class is not None:
            if self.slow_class is None:
                raise ValueError("this ReceptorSigns carries no slow_class codes")
            f = f * (self.slow_class == SLOW_CLASSES.index(slow_class)).astype(np.float32)
        return f

    def coverage(self, W: sp.csr_matrix) -> pd.DataFrame:
        """Edges and abs(W) synapses per tier (fractions of the whole matrix)."""
        wabs = np.abs(W.tocsr().data).astype(np.float64)
        rows = []
        for i, name in enumerate(RECEPTOR_TIERS):
            m = self.tier == i
            rows.append({"tier": name, "edges": int(m.sum()), "edges_frac": float(m.mean()) if len(m) else 0.0,
                         "syn_W": float(wabs[m].sum()), "syn_W_frac": float(wabs[m].sum() / max(wabs.sum(), 1.0))})
        m = self.matched
        rows.append({"tier": "matched", "edges": int(m.sum()), "edges_frac": float(m.mean()) if len(m) else 0.0,
                     "syn_W": float(wabs[m].sum()), "syn_W_frac": float(wabs[m].sum() / max(wabs.sum(), 1.0))})
        return pd.DataFrame(rows)


def read_receptor_table(path=None) -> pd.DataFrame:
    return pd.read_csv(RECEPTOR_TABLE if path is None else path, comment="#")


def receptor_signs(c: "Connectome", table_path=None, net_rule: str = "class", nt_class_fallback: bool = False,
                   W: sp.csr_matrix | None = None, with_counts: bool = False, counts: np.ndarray | None = None,
                   table: pd.DataFrame | None = None) -> ReceptorSigns:
    """Look every stored entry of `W` (default c.W, CSR order) up in the receptor table.

    net_rule: 'class' (sign of the larger gain class; ties keep the NT_SIGN prior), 'abs' (larger summed expression
    with a 2-fold margin) or 'nonmda' (class rule with the glutamate fast + group restricted to KaiR1D / GluRIA /
    GluRIB) -- the three column sets of receptors_by_type.csv.
    nt_class_fallback: unprofiled postsynaptic types take the Davis 2020 ChAT / Gad1 / VGlut whole-class baseline
    of their own transmitter (tier 'nt_class'); off by default.
    with_counts: also fill `count` (abs(W.data), with explicit-zero entries -- sign-0 presynaptic cells -- rescued
    from sign0_counts(c), needed by the slow term); `counts` supplies that array directly (aligned with W.data).
    Same logic as scripts/build_receptor_table.edge_lookup (which is the audited reference)."""
    if net_rule not in RECEPTOR_NET_RULES:
        raise ValueError(f"net_rule must be one of {RECEPTOR_NET_RULES}")
    W = c.W.tocsr() if W is None else W.tocsr()
    rt = read_receptor_table(table_path) if table is None else table
    sfx = {"class": "", "abs": "_abs", "nonmda": "_nonmda"}[net_rule]
    ssfx = "_abs" if net_rule == "abs" else ""
    coo = W.tocoo()                                   # csr -> coo keeps the stored order
    post, pre = coo.row, coo.col
    n = c.neurons
    # A graph without transmitter / type labels (the synthetic graphs of the tests) has nothing to look up: every
    # entry keeps the presynaptic sign at tier 'fallback', i.e. the weights of the model before the receptor block
    # (round 3 made the model the default, so such graphs reach this function through LIFParams()).
    labelled = "nt" in n.columns and "type" in n.columns
    nt_cats = TRANSMITTERS + ["unknown"]
    nt_code = pd.Categorical(n.nt if labelled else pd.Series(["unknown"] * len(n), index=n.index),
                             categories=nt_cats).codes.astype(np.int16)
    nt_code[nt_code < 0] = len(TRANSMITTERS)
    type_cat = pd.Categorical(n.type.fillna("") if labelled else pd.Series([""] * len(n), index=n.index))
    type_code = type_cat.codes.astype(np.int32)
    type_index = {t: i for i, t in enumerate(type_cat.categories)}
    gain_code = {g: i for i, g in enumerate(GAIN_CLASSES)}
    tier_code = {t: i for i, t in enumerate(RECEPTOR_TIERS)}

    rows = rt[~rt.malecns_type.astype(str).str.startswith("<")]
    L = np.full((len(type_cat.categories), len(nt_cats)), -1, dtype=np.int64)
    ti = np.array([type_index.get(t, -1) for t in rows.malecns_type], dtype=np.int64)
    ni = np.array([TRANSMITTERS.index(x) for x in rows.transmitter], dtype=np.int64)
    ok = ti >= 0
    L[ti[ok], ni[ok]] = np.arange(len(rows))[ok]
    fs = rows["fast_sign" + sfx].to_numpy(np.float32); ss = rows["slow_sign" + ssfx].to_numpy(np.float32)
    fg = rows["fast_gain_class" + sfx].map(gain_code).fillna(0).to_numpy(np.int8)
    sg = rows["slow_gain_class" + ssfx].map(gain_code).fillna(0).to_numpy(np.int8)
    tr = rows.tier.map(tier_code).fillna(tier_code["class"]).to_numpy(np.int8)

    pre_nt = nt_code[pre]
    idx = L[type_code[post], pre_nt]
    matched = idx >= 0
    i2 = np.where(matched, idx, 0)
    pre_sign = n.sign.to_numpy(np.float32)[pre] if "sign" in n.columns else np.sign(W.data).astype(np.float32)
    fast_sign = np.where(matched, fs[i2], pre_sign).astype(np.float32)
    slow_sign = np.where(matched, ss[i2], np.float32(0)).astype(np.float32)
    fast_gain = np.where(matched, fg[i2], 0).astype(np.int8)
    slow_gain = np.where(matched, sg[i2], 0).astype(np.int8)
    unknown_pre = (pre_nt == len(TRANSMITTERS)) if labelled else np.zeros(len(pre), dtype=bool)
    tier = np.where(matched, tr[i2], np.where(unknown_pre, tier_code["pre_unknown"], 0)).astype(np.int8)

    if nt_class_fallback:
        sel = rt[rt.malecns_type.astype(str).str.startswith("<nt=")].reset_index(drop=True)
        L2 = np.full((len(nt_cats), len(nt_cats)), -1, dtype=np.int64)
        for j, r in sel.iterrows():
            post_nt = r.malecns_type[len("<nt="):-1]
            if post_nt in TRANSMITTERS:
                L2[TRANSMITTERS.index(post_nt), TRANSMITTERS.index(r.transmitter)] = j
        idx2 = L2[nt_code[post], pre_nt]
        use = (~matched) & (idx2 >= 0)
        j2 = idx2[use]
        fast_sign[use] = sel["fast_sign" + sfx].to_numpy(np.float32)[j2]
        slow_sign[use] = sel["slow_sign" + ssfx].to_numpy(np.float32)[j2]
        fast_gain[use] = sel["fast_gain_class" + sfx].map(gain_code).fillna(0).to_numpy(np.int8)[j2]
        slow_gain[use] = sel["slow_gain_class" + ssfx].map(gain_code).fillna(0).to_numpy(np.int8)[j2]
        tier[use] = tier_code["nt_class"]

    count = None
    if counts is not None:
        count = np.asarray(counts, dtype=np.float32)
        if count.shape != W.data.shape:
            raise ValueError("counts must be aligned with W.data")
    elif with_counts:
        count = np.abs(W.data).astype(np.float32)
        zero = W.data == 0
        if zero.any():
            rescued = sign0_counts(c, W=W)
            if rescued is not None:
                count[zero] = rescued[zero]
    # slow class per entry from the presynaptic transmitter (0 where there is no slow sign)
    class_of_nt = np.array([SLOW_CLASS_OF_TRANSMITTER[t] for t in TRANSMITTERS] + [0], dtype=np.int8)
    slow_class = np.where(slow_sign != 0, class_of_nt[pre_nt], 0).astype(np.int8)
    return ReceptorSigns(fast_sign, slow_sign, fast_gain, slow_gain, tier, count, net_rule, nt_class_fallback,
                         str(RECEPTOR_TABLE if table_path is None else table_path), slow_class)


def build_sign0_counts(ref: "Connectome", cache_dir: Path = CACHE_DIR, verbose: bool = True) -> Path:
    """Raw synapse counts of the reference graph's explicit-zero entries (sign-0 presynaptic cells: monoamines
    and unknown transmitter), keyed by (post, pre) row index of the reference graph, from the raw weights table.
    Written once to cache/sign0_counts.npz; the cache's W stores 0 for these synapses, so the slow term needs it."""
    log = print if verbose else (lambda *a, **k: None)
    if ref.reference is not ref:
        raise ValueError("build_sign0_counts needs the reference (full) graph")
    t0 = time.time()
    w = pf.read_table(DATA_DIR / WEIGHTS_FILE, columns=["body_pre", "body_post", "weight"]).to_pandas()
    pre = ref.body_to_index.reindex(w.body_pre.to_numpy()).to_numpy()
    post = ref.body_to_index.reindex(w.body_post.to_numpy()).to_numpy()
    m = ~np.isnan(pre) & ~np.isnan(post) & (w.weight.to_numpy() >= 1)
    R = sp.csr_matrix((w.weight.to_numpy()[m].astype(np.float32), (post[m].astype(np.int64), pre[m].astype(np.int64))),
                      shape=ref.W.shape)
    R.sum_duplicates()
    del w
    Wr = ref.W.tocsr()
    Rc, Wc = R.tocoo(), Wr.tocoo()
    zero = Wc.data == 0
    key_w = Wc.row[zero].astype(np.int64) * ref.n + Wc.col[zero]
    key_r = Rc.row.astype(np.int64) * ref.n + Rc.col
    order = np.argsort(key_r, kind="stable")
    pos = np.searchsorted(key_r, key_w, sorter=order)
    pos = np.minimum(pos, len(key_r) - 1)
    hit = key_r[order[pos]] == key_w
    cnt = np.where(hit, Rc.data[order[pos]], 0.0).astype(np.float32)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / SIGN0_COUNTS_FILE
    np.savez(out, key=np.sort(key_w), count=cnt[np.argsort(key_w, kind="stable")], n=np.int64(ref.n))
    log(f"sign-0 counts: {int(zero.sum()):,} zero entries, {int(hit.sum()):,} found in the raw table, "
        f"{int(cnt.sum()):,} synapses ({time.time() - t0:.1f}s) -> {out}")
    return out


def sign0_counts(c: "Connectome", W: sp.csr_matrix | None = None, cache_dir: Path = CACHE_DIR,
                 build: bool = True) -> np.ndarray | None:
    """Raw synapse counts aligned with the stored entries of `W` (default c.W), non-zero only where W.data == 0 and the
    raw table has the edge. Reads cache/sign0_counts.npz (built from the raw weights table if absent and the table
    is available); None with a warning when neither exists, in which case the slow term skips sign-0 edges."""
    import warnings
    path = Path(cache_dir) / SIGN0_COUNTS_FILE
    if not path.exists():
        if not build or not (DATA_DIR / WEIGHTS_FILE).exists():
            warnings.warn(f"{path} is absent and the raw weights table is not available at {DATA_DIR}: "
                          "sign-0 (monoamine / unknown) synapses carry no slow term")
            return None
        build_sign0_counts(c.reference, cache_dir)
    z = np.load(path)
    key, cnt, n_ref = z["key"], z["count"], int(z["n"])
    if n_ref != c.reference.n:
        warnings.warn(f"{path} was built for a graph of {n_ref} neurons, this reference has {c.reference.n}; ignored")
        return None
    W = c.W.tocsr() if W is None else W.tocsr()
    coo = W.tocoo()
    ref_idx = c.reference.index_of(c.neurons.bodyId.to_numpy()) if c.reference is not c else np.arange(c.n)
    out = np.zeros(W.nnz, dtype=np.float32)
    zero = coo.data == 0
    if not zero.any():
        return out
    q = ref_idx[coo.row[zero]].astype(np.int64) * n_ref + ref_idx[coo.col[zero]]
    pos = np.minimum(np.searchsorted(key, q), len(key) - 1)
    hit = key[pos] == q
    vals = np.where(hit, cnt[pos], 0.0).astype(np.float32)
    out[np.flatnonzero(zero)] = vals
    return out

KEEP_COLS = ["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "somaNeuromere",
             "status", "entryNerve", "exitNerve", "flywireType", "hemibrainType", "mancType",
             "assignedOlHex1", "assignedOlHex2"]


@dataclass
class Connectome:
    neurons: pd.DataFrame      # index 0..N-1; KEEP_COLS + nt, sign, hex1, hex2, hex_side, hex_source
    W: sp.csr_matrix           # (N, N), W[post, pre] = signed synapse count (float32)
    body_to_index: pd.Series   # bodyId -> row index
    _reference: Connectome | None = field(default=None, repr=False)
    _norm_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Fast synapse magnitudes; sign-zero neuromodulatory contacts contribute zero.
        # Old caches acquire these columns in memory without rewriting the cache.
        if "in_syn" not in self.neurons:
            self.neurons["in_syn"] = np.asarray(abs(self.W).sum(axis=1)).ravel()
        if "in_syn_l2" not in self.neurons:
            self.neurons["in_syn_l2"] = np.sqrt(np.asarray(self.W.multiply(self.W).sum(axis=1)).ravel())

    @property
    def reference(self) -> Connectome:
        """Original graph, retained on CPU for parameter-dependent normalization and laterality."""
        return self if self._reference is None else self._reference

    def indices(self, selection) -> np.ndarray:
        """Validate a row-index selection, boolean mask, or select() criteria dictionary."""
        if isinstance(selection, dict):
            return self.select(**selection)
        idx = np.asarray(selection)
        if idx.dtype == bool:
            if idx.shape != (self.n,):
                raise ValueError(f"mask must have shape ({self.n},)")
            return np.flatnonzero(idx)
        if idx.ndim != 1 or (idx.size and not np.issubdtype(idx.dtype, np.integer)):
            raise ValueError("selection must be a one-dimensional integer index array")
        idx = idx.astype(np.int64)
        if np.any((idx < 0) | (idx >= self.n)) or len(np.unique(idx)) != len(idx):
            raise ValueError("indices must be unique and within the connectome")
        return idx

    def subset(self, selection) -> Connectome:
        """Induced subgraph in selection order, with fresh indices and original normalization.

        Selectors refer to this graph's row indices; body IDs stay stable across subsets.
        The original CPU graph is shared, never copied, including for nested subsets.
        """
        idx = self.indices(selection)
        neurons = self.neurons.iloc[idx].copy().reset_index(drop=True)
        return Connectome(neurons, self.W[idx][:, idx].tocsr(),
                          pd.Series(np.arange(len(idx)), index=neurons.bodyId.to_numpy()), self.reference)

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


def compile_connectome(min_weight: int = 1, verbose: bool = True, type_nt_override: dict | None = None) -> Connectome:
    """type_nt_override: {type: nt} applied to the type's sign-0 cells after the consensus (None = TYPE_NT_OVERRIDE if
    TYPE_NT_OVERRIDE_DEFAULT else nothing; pass {} for none, TYPE_NT_OVERRIDE to force it)."""
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
    if type_nt_override is None:
        type_nt_override = TYPE_NT_OVERRIDE if TYPE_NT_OVERRIDE_DEFAULT else {}
    no_fast_label = neurons.nt.map(NT_SIGN).fillna(0.0) == 0.0          # unknown or monoamine: sign 0
    for t, nt in type_nt_override.items():
        m = (neurons.type == t) & no_fast_label & (neurons.nt != nt)
        was = neurons.loc[m, "nt"].value_counts().to_dict()
        neurons.loc[m, "nt"] = nt
        log(f"type-NT override {t} -> {nt}: {int(m.sum())} of {int((neurons.type == t).sum())} neurons relabelled (from {was}); "
            f"kept: {neurons.loc[(neurons.type == t) & ~m, 'nt'].value_counts().to_dict()}")
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
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    c.neurons.to_parquet(cache_dir / "neurons.parquet")
    sp.save_npz(cache_dir / "W_post_pre.npz", c.W, compressed=False)
    # A saved subset must retain normalization under custom LIF parameters as well.
    if c.reference is not c:
        c.reference.neurons.to_parquet(cache_dir / "reference_neurons.parquet")
        sp.save_npz(cache_dir / "reference_W.npz", c.reference.W, compressed=False)
    else:
        for name in ("reference_neurons.parquet", "reference_W.npz"):
            (cache_dir / name).unlink(missing_ok=True)


def load(cache_dir: Path = CACHE_DIR, rebuild: bool = False, verbose: bool = True,
         type_nt_override: dict | None = None) -> Connectome:
    cache_dir = Path(cache_dir)
    if rebuild or not (cache_dir / "W_post_pre.npz").exists():
        c = compile_connectome(verbose=verbose, type_nt_override=type_nt_override)
        save(c, cache_dir)
        return c
    neurons = pd.read_parquet(cache_dir / "neurons.parquet")
    W = sp.load_npz(cache_dir / "W_post_pre.npz").tocsr()
    reference = None
    if (cache_dir / "reference_W.npz").exists():
        rn = pd.read_parquet(cache_dir / "reference_neurons.parquet")
        reference = Connectome(rn, sp.load_npz(cache_dir / "reference_W.npz").tocsr(),
                              pd.Series(np.arange(len(rn)), index=rn.bodyId.to_numpy()))
    return Connectome(neurons=neurons, W=W,
                      body_to_index=pd.Series(np.arange(len(neurons)), index=neurons.bodyId.to_numpy()),
                      _reference=reference)


if __name__ == "__main__":
    c = load(rebuild=True)
    print(c.neurons.head())
    print("N =", c.n, " nnz =", c.W.nnz)
