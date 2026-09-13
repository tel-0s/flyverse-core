"""paths -- effective k-step signed gains from population A to population B through the shaped weights, silent links flagged.

The question (docs/INTERP.md 4.3): along which types / cells does A reach B within k <= k_max synapses under the model's
own synaptic rules, how strong is each route (the product of the effective weights along it), and which of the links
on those routes are SILENT -- explicit zeros in the compiled matrix (a sign-0 presynaptic transmitter: GLNO, the
monoamines, unknown NT), frozen / pruned optic rate units, or cells that never fired in a named rollout. Silent links
are the finding, not a footnote: the summary is the strongest silent link per k, quoted with the weight it WOULD carry
if it were signed (the same cap / gains / fan-in scale / w_syn applied to its raw synapse count).

Two levels.

* `level='type'`: nodes are one source node per type in `a` ('a:<type>', source only: a volley of ONE type is the unit
  in which walk gains are comparable, and a pooled heterogeneous source would cancel signs inside its own volley), the
  pooled target `b` (target only: its statistic is a mean over post cells, well defined for any set), and every type
  with a cell on some path of length <= k_max between them (forward depth from `a` + backward depth from `b`, over the
  stored structure of `c.W`, explicit zeros included so that silent links are visible; an intermediate node holds every
  cell of its type except those in a / b). The link weight is `EffectiveWeights.type_matrix`'s statistic, M[post, pre] =
  mean over post cells of the summed input from every pre cell (mV per post cell per presynaptic volley; cx_wedge's
  'total per post'), computed here as D^-1 P^T A P. A walk's gain is the product of its M entries (mV^k per volley).
  The `top` walks per k by |gain| are found exactly with a k-best dynamic programme over (node, has-silent-link) states
  (a product of non-negative magnitudes is monotone, so the K best walks ending at a node extend the K best walks
  ending at its predecessors). `exclude` drops types from the intermediate set; `min_abs_mv` drops links weaker than
  that (if-signed magnitude). The direct link a -> b over the whole of `a` is always reported (summary 'direct'), and
  the table 'b_inputs' lists every presynaptic type of `b` pooled -- raw share, sign, flags -- so the dominant silent
  input of the target (GLNO: 19.4 % of PEN's raw input, sign 0) is read off directly, walks or no walks.
* `level='cell'`: nodes are cells; a walk's gain is the product of A entries (mV^k per spike). With k_max >= 2 the tool
  also lists, per intermediate type, the two-step matrix A[b, t] @ A[t, a] summed per post cell (cx_wedge.md section 2:
  ExR6 -15,792 mV^2 ...), and with `wedge=True` aggregates the two-step matrices by PB wedge exactly as
  scripts/cx_wedge.py does (its RING16 order, glomerulus regex and `aggregate`), giving the ring-distance profiles of
  cx_wedge.md section 3 (PEN +2,465 on-wedge, Delta7 -433 opposite, ER/ExR -2,760..-3,210 flat).

Nothing here re-implements a rule of brain.py: the matrix is `common.effective_weights` (brain._shaped_weights x the
fan-in scale x w_syn), the raw counts are |W| plus cache/sign0_counts.npz for the explicit zeros (`raw_counts` here), the
silence flags are `common.silent_flags`, and the if-signed magnitude of a silent entry is w_syn x scale[post] x
min(count, conn_cap) x the path / type / same-type gain factors that brain._shaped_weights applies to that (pre, post)
pair (obtained by running brain._shaped_weights on the graph's structure with unit counts and the cap off).

CPU only (numpy / scipy); no torch. Validation targets: `common.VALIDATION['paths']`, run by `validate()` and
`scripts/interp_paths.py validate`, reported in docs/audits/interp_paths.md.
"""
from __future__ import annotations

import dataclasses
import platform
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import common
from .. import connectome as cn

TOOL = "paths"
TOOL_VERSION = "0.1"
SILENT_FLAGS = ("sign0", "frozen", "pruned", "never_firing")
UNTYPED = "(untyped)"
# the groups cx_wedge.md section 3 profiles the EPG x EPG two-step through (regex on the intermediate type)
WEDGE_GROUPS = {"PEN": r"^PEN_", "PEG": r"^PEG$", "Delta7": r"^Delta7$", "Ring": r"^(ER|ExR)"}


# ---------------------------------------------------------------------------------------------- helpers
def _cx_wedge():
    """scripts/cx_wedge.py (the generator of docs/audits/cx_wedge.json), imported from scripts/ so that the wedge order,
    the glomerulus regex and the wedge aggregation are the audit's own code, not a copy."""
    scripts = common.ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import cx_wedge  # noqa: WPS433
    return cx_wedge


def spec_from_cli(s):
    """A CLI population spec: 'LNO1|LNO2|~^LAL' (types or '~regex' tokens joined by '|', none of them a key=value clause)
    becomes the list ['LNO1', 'LNO2', '~^LAL'] (= OR in common.resolve); everything else is passed through unchanged
    (a plain 'LC10a|LC11' resolves identically either way; a leading '~' keeps the whole string as one regex)."""
    if isinstance(s, str) and "|" in s and not s.startswith("~") and "&" not in s and not re.match(r"^[A-Za-z_]\w*\s*[=:~]", s):
        toks = [t.strip() for t in s.split("|") if t.strip()]
        if any(t.startswith("~") for t in toks):
            return toks
    return s


def _label_of(spec) -> str:
    return common.spec_repr(spec)


def resolve_loose(c: cn.Connectome, spec) -> np.ndarray:
    """common.resolve, with one convenience for the MaleCNS type strings that carry a parenthesised synonym
    ('PEN_a(PEN1)', 'PEN_b(PEN2)' -- the only two in the release): a plain type token that selects nothing is retried
    as the regex '^<token>(?:\\(|$)', so 'PEN_a|PEN_b' and ['PEN_a', 'PEN_b'] select the PEN cells. Everything that
    resolves as written is returned unchanged."""
    idx = common.resolve(c, spec)
    if len(idx) or not isinstance(spec, (str, list, tuple)):
        return idx
    toks = spec if isinstance(spec, (list, tuple)) else ([spec] if ("=" in spec or spec.startswith(("~", "body:", "index:", "type:"))) else spec.split("|"))
    out = []
    for t in toks:
        if isinstance(t, str) and re.match(r"^[A-Za-z0-9_\-+.]+$", t) and len(common.resolve(c, t)) == 0:
            out.append(f"~^{re.escape(t)}(?:\\(|$)")
        else:
            out.append(t)
    return common.resolve(c, out)


def raw_counts(c: cn.Connectome, with_sign0: bool = True) -> tuple[sp.csr_matrix, bool]:
    """`common.raw_counts`: |W| merged with cache/sign0_counts.npz on the explicit-zero (sign-0) entries.

    The private copy this tool carried while the shared accessor SUBSTITUTED the sign-0 array for the whole count
    vector is gone (docs/INTERP.md 11, defect 1, closed): the merge is `common.raw_counts`'s own."""
    return common.raw_counts(c, with_sign0)


def structure_matrix(counts: sp.csr_matrix) -> sp.csr_matrix:
    """S[post, pre] = 1 for every stored entry of the compiled matrix, explicit zeros included."""
    S = counts.tocsr().copy()
    S.data = np.ones_like(S.data, dtype=np.float32)
    return S


def depths(S: sp.csr_matrix, src_mask: np.ndarray, k_max: int, backward: bool = False) -> np.ndarray:
    """Shortest number of synapses from the source set to every cell (forward: over S[post, pre]; backward: to the set),
    -1 when farther than k_max. The source cells are at depth 0."""
    n = S.shape[0]
    T = S.T.tocsr() if backward else S
    d = np.full(n, -1, np.int64)
    frontier = np.asarray(src_mask, bool).copy()
    d[frontier] = 0
    for step in range(1, int(k_max) + 1):
        nxt = np.asarray(T @ frontier.astype(np.float32)).ravel() > 0
        nxt &= d < 0
        if not nxt.any():
            break
        d[nxt] = step
        frontier = nxt
    return d


def if_signed_magnitudes(c: cn.Connectome, ew: common.EffectiveWeights, counts: sp.csr_matrix) -> sp.csr_matrix:
    """A_if[post, pre] = |A| where A is non-zero, and for the silent stored entries (A == 0, raw count > 0) the magnitude
    the entry would carry if it were signed: w_syn x scale[post] x min(count, conn_cap) x the path / type / same-type
    gain factors brain._shaped_weights applies to that (pre, post) pair. Same structure as `counts` (explicit zeros
    where the raw count is 0 too)."""
    from .. import brain
    p = ew.params
    C = counts.tocsr()
    W1 = C.copy(); W1.data = np.ones_like(W1.data, dtype=np.float32)
    ones = cn.Connectome(c.neurons, W1, c.body_to_index, c._reference)
    p1 = dataclasses.replace(p, receptor_model=None, conn_cap=0.0)
    F = brain._shaped_weights(ones, p1).tocsr()                     # the gain factors per stored entry (unit counts, no cap)
    F.sort_indices(); C = C.copy(); C.sort_indices()
    if F.nnz != C.nnz or not np.array_equal(F.indices, C.indices) or not np.array_equal(F.indptr, C.indptr):
        raise RuntimeError("gain-factor matrix is not aligned with the count matrix (brain._shaped_weights changed the structure)")
    A = ew.A.tocsr().copy(); A.sort_indices()
    cap = np.float32(p.conn_cap) if p.conn_cap > 0 else np.float32(np.inf)
    coo = C.tocoo()
    mag = np.minimum(coo.data, cap) * np.abs(F.data) * ew.scale[coo.row] * np.float32(p.w_syn)
    Aif = sp.csr_matrix((mag.astype(np.float32), (coo.row, coo.col)), shape=C.shape)
    Aif.sort_indices()
    # where A is non-zero keep |A| (it already carries the receptor factor / gains); elsewhere the if-signed magnitude
    Aabs = abs(A)
    nz = structure_matrix(Aabs)
    Aif = Aif - Aif.multiply(nz) + Aabs
    Aif.eliminate_zeros()
    return Aif.tocsr()


def rates_from_recording(recording) -> dict | None:
    """{model index: max rate over the rollout (and the batch)} from a Recording (or its path); None when no recording."""
    if recording is None:
        return None
    if isinstance(recording, (str, Path)):
        recording = common.Recording.load(recording)
    if "rate_hz" not in recording.quantities:
        raise ValueError("the recording carries no rate_hz")
    r = np.asarray(recording.quantities["rate_hz"], dtype=np.float64)
    mx = r.reshape(-1, r.shape[-1]).max(axis=0) if r.size else np.full(r.shape[-1], np.nan)
    return {int(i): float(m) for i, m in zip(recording.idx, mx)}


def frozen_from(frozen, c: cn.Connectome):
    """`frozen`: None, an index array, a FlyBrain (its optic.rate_idx), or 'static' (superclass ol_intrinsic)."""
    if frozen is None:
        return None
    if isinstance(frozen, str):
        if frozen == "static":
            return np.flatnonzero(common.unit_kinds(c) == "graded")
        if frozen == "none":
            return None
        z = np.load(frozen)
        return np.asarray(z["rate_idx"] if "rate_idx" in z.files else z[z.files[0]], dtype=np.int64)
    fr = common.frozen_indices(frozen)
    return fr if fr is not None else np.asarray(frozen, dtype=np.int64)


# ---------------------------------------------------------------------------------------------- the node graph
@dataclass
class NodeGraph:
    """The k-best search graph: node labels (index 0.. ; the endpoint populations are 'a' and 'b' at the type level),
    the cell membership per node (P: cells x nodes), the signed link matrix M[post node, pre node], the if-signed
    magnitude matrix Mif, the raw-count matrix Craw, the stored-entry count Pairs, per-flag raw-count shares, the
    source / target / intermediate node index sets and the per-node cell counts."""
    labels: list
    P: sp.csr_matrix
    M: sp.csr_matrix
    Mif: sp.csr_matrix
    Craw: sp.csr_matrix
    Pairs: sp.csr_matrix              # stored entries with a raw count > 0 per node pair
    PairsNz: sp.csr_matrix            # entries that are non-zero in A per node pair (cx_wedge's 'pairs')
    silent_share: dict
    src: np.ndarray
    dst: np.ndarray
    mid: np.ndarray
    n_cells: np.ndarray
    raw_input: np.ndarray            # total raw input of the node's cells (sum over all presynaptic cells)
    level: str
    node_type: list = field(default_factory=list)
    node_cells: list = field(default_factory=list)

    def link(self, v: int, u, label: str | None = None) -> dict:
        """One link pre node u -> post node v: mV per post volley (M) and per pair (mean over the non-zero pairs, as
        cx_wedge's 'mean_per_pair'), the if-signed magnitudes, raw count, share of the post node's raw input, sign and
        flags. `u` may be a list of pre nodes, which are pooled (a volley of all their cells; `label` names the pool)."""
        us = np.atleast_1d(np.asarray(u, dtype=np.int64))
        m, mif, cr, pr, pnz = (float(X[v, us].sum()) for X in (self.M, self.Mif, self.Craw, self.Pairs, self.PairsNz))
        n_post = float(self.n_cells[v])
        s = np.sign(m) if m != 0 else 1.0
        pre_label = label if label is not None else (self.labels[int(us[0])] if len(us) == 1 else "+".join(self.labels[int(i)] for i in us))
        pre_type = self.node_type[int(us[0])] if len(us) == 1 else pre_label
        d = {"pre": pre_label, "post": self.labels[v], "pre_type": pre_type, "post_type": self.node_type[v],
             "n_pre_cells": int(self.n_cells[us].sum()), "n_post_cells": int(self.n_cells[v]), "pairs": int(round(pr)), "pairs_nonzero": int(round(pnz)),
             "raw_count": cr, "share_of_post_input": (cr / self.raw_input[v]) if self.raw_input[v] > 0 else np.nan,
             "mv_per_post_volley": m, "mv_per_post_volley_if_signed": s * mif,
             "mv_per_pair": (m * n_post / pnz) if pnz > 0 else np.nan,
             "mv_per_pair_if_signed": (s * mif * n_post / pr) if pr > 0 else np.nan,
             "sign": int(np.sign(m))}
        flags, partial = [], []
        for f, S in self.silent_share.items():
            share = float(S[v, us].sum()) / cr if cr > 0 else 0.0
            d[f"{f}_frac"] = share
            if share >= 0.999:
                flags.append(f)
            elif share > 0:
                partial.append(f"{f}:{share:.2f}")
        if m == 0 and cr > 0 and not flags:
            flags.append("zeroed")             # zero in A without a structural flag (a receptor-table zero, or no rates given)
        d["silent"] = "|".join(flags)
        d["silent_partial"] = "|".join(partial)
        return d


def _flag_vectors(c: cn.Connectome, frozen_idx, rates, prune_frozen: bool, min_hz: float) -> dict:
    fl = common.silent_flags(c, np.arange(c.n), frozen_idx=frozen_idx, prune_frozen=prune_frozen, rates=rates, min_hz=min_hz)
    out = {}
    for f in SILENT_FLAGS:
        v = fl[f].to_numpy()
        v = np.where(pd.isna(v), False, v).astype(np.float32)
        if v.any():
            out[f] = v
    return out


def build_graph(c: cn.Connectome, ew: common.EffectiveWeights, a_idx, b_idx, *, counts=None, k_max: int = 3, level: str = "type",
                exclude=(), frozen_idx=None, rates=None, prune_frozen: bool = True, rates_min_hz: float = common.NEVER_FIRING_HZ,
                min_abs_mv: float = 0.0, Aif=None) -> NodeGraph:
    """The search graph of `paths` (see the module docstring for the node sets at each level)."""
    a_idx, b_idx = np.asarray(a_idx, np.int64), np.asarray(b_idx, np.int64)
    if counts is None:
        counts, _ = raw_counts(c)
    C = counts.tocsr()
    S = structure_matrix(C)
    A = ew.A.tocsr()
    if Aif is None:
        Aif = if_signed_magnitudes(c, ew, C)
    N = c.n
    ma = np.zeros(N, bool); ma[a_idx] = True
    mb = np.zeros(N, bool); mb[b_idx] = True
    da = depths(S, ma, k_max)                     # from a (post side)
    db = depths(S, mb, k_max, backward=True)      # to b
    cand = (da >= 1) & (db >= 1) & (da + db <= k_max) & ~ma & ~mb
    ty = c.neurons.type.fillna("").to_numpy().astype(object)
    ty = np.where(ty == "", UNTYPED, ty)
    excl = set()
    for e in exclude or ():
        if isinstance(e, str) and e.startswith("~"):
            excl |= {t for t in np.unique(ty) if re.search(e[1:], t)}
        else:
            excl.add(str(e))
    flags = _flag_vectors(c, frozen_idx, rates, prune_frozen, rates_min_hz)
    if level == "type":
        # source nodes: one per type in `a` ('a:<type>', its cells of that type -- a volley of one type is the unit the
        # walk gains are comparable in); target node 'b' pooled (its statistic is a mean over post cells, which is well
        # defined for any set); intermediates: every type with a cell on some path, all its cells except those in a / b
        ty_a = ty[a_idx]
        a_types = [t for t in np.unique(ty_a)]
        cand_types = [t for t in np.unique(ty[cand]) if t not in excl]
        labels = [f"a:{t}" for t in a_types] + ["b", *cand_types]
        node_type = [*a_types, "b", *cand_types]
        n_a = len(a_types)
        rows = [a_idx, b_idx]
        cols = [np.array([a_types.index(t) for t in ty_a], dtype=np.int64), np.full(len(b_idx), n_a, np.int64)]
        pos = {t: i + n_a + 1 for i, t in enumerate(cand_types)}
        member = np.array([pos.get(t, -1) for t in ty], dtype=np.int64)
        ok = (member >= 0) & ~ma & ~mb
        rows.append(np.flatnonzero(ok)); cols.append(member[ok])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        node_cells = [a_idx[ty_a == t] for t in a_types] + [b_idx] + [np.flatnonzero((ty == t) & ~ma & ~mb) for t in cand_types]
        src, dst, mid = np.arange(n_a), np.array([n_a]), np.arange(n_a + 1, len(labels))
    elif level == "cell":
        mid_cells = np.flatnonzero(cand & ~np.isin(ty, list(excl)))
        cells = np.concatenate([a_idx, b_idx[~ma[b_idx]], mid_cells])   # a cell in both a and b appears once
        bid = c.neurons.bodyId.to_numpy()
        labels = [f"{ty[i]}#{int(bid[i])}" for i in cells]
        node_type = [str(ty[i]) for i in cells]
        node_of = -np.ones(N, np.int64); node_of[cells] = np.arange(len(cells))
        rows, cols = cells, np.arange(len(cells))
        node_cells = [np.array([i]) for i in cells]
        src = node_of[a_idx]; dst = node_of[b_idx]
        mid = node_of[mid_cells]
    else:
        raise ValueError("level must be 'type' or 'cell'")
    G = len(labels)
    P = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, G))
    n_cells = np.asarray(P.sum(axis=0)).ravel()
    Dinv = sp.diags(1.0 / np.maximum(n_cells, 1.0))
    PT = P.T.tocsr()
    M = (Dinv @ (PT @ A @ P)).tocsr()
    Mif = (Dinv @ (PT @ Aif @ P)).tocsr()
    Craw = (PT @ C @ P).tocsr()
    Cpos = C.copy(); Cpos.data = (Cpos.data > 0).astype(np.float32); Cpos.eliminate_zeros()
    Pairs = (PT @ Cpos @ P).tocsr()
    Anz = A.copy(); Anz.data = (Anz.data != 0).astype(np.float32); Anz.eliminate_zeros()
    PairsNz = (PT @ Anz @ P).tocsr()
    silent_share = {f: (PT @ C @ sp.diags(v) @ P).tocsr() for f, v in flags.items()}
    raw_input = np.asarray(PT @ np.asarray(C.sum(axis=1)).ravel()).ravel()
    if min_abs_mv > 0:
        keep = Mif.copy(); keep.data = (keep.data >= min_abs_mv).astype(np.float32); keep.eliminate_zeros()
        Mif = Mif.multiply(keep).tocsr(); M = M.multiply(keep).tocsr()
    for X in (M, Mif, Craw, Pairs, PairsNz):
        X.eliminate_zeros(); X.sort_indices()
    return NodeGraph(labels, P, M, Mif, Craw, Pairs, PairsNz, silent_share, src, dst, mid, n_cells, raw_input, level, node_type, node_cells)


# ---------------------------------------------------------------------------------------------- k-best walks
def _topk_per_key(keys: np.ndarray, vals: np.ndarray, K: int) -> np.ndarray:
    """Indices of the K largest `vals` per distinct key (ties by order)."""
    order = np.lexsort((-vals, keys))
    ks = keys[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
    return order[(np.arange(len(ks)) - start) < K]


def edges(g: NodeGraph, full: float = 0.999) -> tuple:
    """The link list of a NodeGraph: (post node v, pre node u, if-signed magnitude, sign of M, silent) per stored link
    of g.Mif. A link is silent when M is zero there (sign 0 / receptor zero) or when a silence flag (frozen, pruned,
    never_firing, sign0) covers at least `full` of its raw synapses -- the same rule NodeGraph.link uses for 'silent'."""
    Mif = g.Mif.tocoo()
    v, u, w = Mif.row, Mif.col, Mif.data.astype(np.float64)
    sgn = np.sign(np.asarray(g.M.tocsr()[v, u]).ravel()).astype(np.int8)
    sil = sgn == 0
    if len(v):
        craw = np.asarray(g.Craw.tocsr()[v, u]).ravel().astype(np.float64)
        for S in g.silent_share.values():
            share = np.asarray(S.tocsr()[v, u]).ravel() / np.maximum(craw, 1e-9)
            sil |= share >= full
    return v, u, w, sgn, sil.astype(np.int8)


def kbest_walks(g: NodeGraph, k_max: int, K: int, chunk: int = 250_000) -> dict:
    """Exact top-K walks by |gain| of every length 1..k_max from g.src to g.dst through g.mid, split by whether the walk
    contains a silent link (state 1: a link that is zero in M, or fully covered by a silence flag -- see `edges`) or
    not (state 0). Returns {k: [walk dict, ...]} with the walk's node list, |gain| (the product of the if-signed link
    magnitudes), the product of the signed links' signs, the per-link signed values and flags."""
    n = len(g.labels)
    v_all, u_all, w_all, sgn_all, sil_all = edges(g)
    is_src = np.zeros(n, bool); is_src[g.src] = True
    is_dst = np.zeros(n, bool); is_dst[g.dst] = True
    is_mid = np.zeros(n, bool); is_mid[g.mid] = True
    ok_v = is_mid[v_all] | is_dst[v_all]
    e_src = ok_v & is_src[u_all]
    e_mid = ok_v & is_mid[u_all]
    # per step: arrays (n, 2, K)
    best_abs, best_sgn, par_node, par_state, par_rank = [], [], [], [], []

    def _select(cands, step_abs, step_sgn, step_pn, step_ps, step_pr):
        v, s, ab, sg, pn, ps, pr = cands
        # merge with the current running best of this step
        cur = step_abs > 0
        if cur.any():
            cv, cs, ck = np.nonzero(cur)
            v = np.concatenate([v, cv]); s = np.concatenate([s, cs]); ab = np.concatenate([ab, step_abs[cv, cs, ck]])
            sg = np.concatenate([sg, step_sgn[cv, cs, ck]]); pn = np.concatenate([pn, step_pn[cv, cs, ck]])
            ps = np.concatenate([ps, step_ps[cv, cs, ck]]); pr = np.concatenate([pr, step_pr[cv, cs, ck]])
        keep = _topk_per_key(v * 2 + s, ab, K)
        step_abs[:] = 0; step_sgn[:] = 0; step_pn[:] = -1; step_ps[:] = 0; step_pr[:] = -1
        kv, ks_ = v[keep], s[keep]
        # rank within (v, s): the keep order is sorted by key then -abs
        key = kv * 2 + ks_
        first = np.r_[True, key[1:] != key[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
        rank = np.arange(len(key)) - start
        step_abs[kv, ks_, rank] = ab[keep]; step_sgn[kv, ks_, rank] = sg[keep]
        step_pn[kv, ks_, rank] = pn[keep]; step_ps[kv, ks_, rank] = ps[keep]; step_pr[kv, ks_, rank] = pr[keep]

    for j in range(1, int(k_max) + 1):
        sa = np.zeros((n, 2, K)); ss = np.zeros((n, 2, K), np.int8)
        pn = -np.ones((n, 2, K), np.int64); ps = np.zeros((n, 2, K), np.int8); pr = -np.ones((n, 2, K), np.int64)
        if j == 1:
            e = np.flatnonzero(e_src)
            cands = (v_all[e], sil_all[e].astype(np.int64), w_all[e], np.where(sgn_all[e] == 0, 1, sgn_all[e]).astype(np.int8),
                     u_all[e], np.zeros(len(e), np.int8), np.zeros(len(e), np.int64))
            _select(cands, sa, ss, pn, ps, pr)
        else:
            prev_abs, prev_sgn = best_abs[-1], best_sgn[-1]
            active = np.zeros(n, bool); active[g.mid] = (prev_abs[g.mid] > 0).any(axis=(1, 2))
            e_all = np.flatnonzero(e_mid & active[u_all])
            for c0 in range(0, len(e_all), chunk):
                e = e_all[c0:c0 + chunk]
                u, v, w, sg, si = u_all[e], v_all[e], w_all[e], sgn_all[e], sil_all[e]
                pa = prev_abs[u]                                    # (m, 2, K)
                valid = pa > 0
                if not valid.any():
                    continue
                ei, s_i, r_i = np.nonzero(valid)
                ab = pa[ei, s_i, r_i] * w[ei]
                sgn_new = (prev_sgn[u[ei], s_i, r_i] * np.where(sg[ei] == 0, 1, sg[ei])).astype(np.int8)
                s_new = (s_i | si[ei]).astype(np.int64)
                cands = (v[ei], s_new, ab, sgn_new, u[ei], s_i.astype(np.int8), r_i.astype(np.int64))
                _select(cands, sa, ss, pn, ps, pr)
        best_abs.append(sa); best_sgn.append(ss); par_node.append(pn); par_state.append(ps); par_rank.append(pr)

    out = {}
    for j in range(1, int(k_max) + 1):
        walks = []
        sa = best_abs[j - 1]
        for state in (0, 1):
            dv, dr = np.nonzero(sa[g.dst, state, :] > 0)
            if len(dv) == 0:
                continue
            vals = sa[g.dst[dv], state, dr]
            order = np.argsort(-vals, kind="stable")[:K]
            for o in order:
                v, r, s = int(g.dst[dv[o]]), int(dr[o]), state
                nodes = [v]
                for step in range(j, 0, -1):
                    u = int(par_node[step - 1][v, s, r]); s2 = int(par_state[step - 1][v, s, r]); r2 = int(par_rank[step - 1][v, s, r])
                    nodes.append(u); v, s, r = u, s2, r2
                nodes = nodes[::-1]
                links = [g.link(nodes[i + 1], nodes[i]) for i in range(j)]
                walks.append({"k": j, "kind": "silent" if state else "signed", "nodes": nodes, "abs_gain": float(vals[o]),
                              "sign_prod": int(best_sgn[j - 1][g.dst[dv[o]], state, dr[o]]), "links": links})
        out[j] = walks
    return out


# ---------------------------------------------------------------------------------------------- the target's inputs
def inputs_of(c: cn.Connectome, ew: common.EffectiveWeights, b_idx, *, counts=None, Aif=None, flags=None, frozen_idx=None,
              rates=None, prune_frozen: bool = True, min_hz: float = common.NEVER_FIRING_HZ) -> pd.DataFrame:
    """Every presynaptic type of the cell set `b_idx`, pooled (cx_shift.md section 1 as a function): the raw synapse
    count and its share of b's raw input, the stored entries, the cells of the type that project onto b, the mean
    input per b cell per volley of the type (signed and if-signed), the mean per pair, the transmitter, the sign and the
    silence flags / shares of the type's entries onto b. Sorted by raw count; the silent rows are where the target's
    input is lost in the compiled model."""
    A = ew.A.tocsr()
    b_idx = np.asarray(b_idx, np.int64)
    if counts is None:
        counts, _ = raw_counts(c)
    C = counts.tocsr()
    if Aif is None:
        Aif = if_signed_magnitudes(c, ew, C)
    if flags is None:
        flags = _flag_vectors(c, frozen_idx, rates, prune_frozen, min_hz)
    Cb, Ab, Bb = C[b_idx], A[b_idx], Aif[b_idx]
    ty = c.neurons.type.fillna("").to_numpy().astype(object)
    ty = np.where(ty == "", UNTYPED, ty)
    nt = c.neurons["nt"].fillna("unknown").to_numpy() if "nt" in c.neurons else np.array(["unknown"] * c.n)
    pre_cells = np.unique(Cb.tocoo().col)
    types = list(np.unique(ty[pre_cells]))
    if not types:
        return pd.DataFrame()
    pos = {t: i for i, t in enumerate(types)}
    member = np.array([pos.get(t, -1) for t in ty], dtype=np.int64)
    ok = member >= 0
    Pt = sp.csr_matrix((np.ones(ok.sum(), np.float32), (np.flatnonzero(ok), member[ok])), shape=(c.n, len(types)))
    nb = len(b_idx)
    raw = np.asarray((Cb @ Pt).sum(axis=0)).ravel()
    Cpos = Cb.copy(); Cpos.data = (Cpos.data > 0).astype(np.float32); Cpos.eliminate_zeros()
    entries = np.asarray((Cpos @ Pt).sum(axis=0)).ravel()
    Anz = Ab.copy(); Anz.data = (Anz.data != 0).astype(np.float32); Anz.eliminate_zeros()
    entries_nz = np.asarray((Anz @ Pt).sum(axis=0)).ravel()
    m = np.asarray((Ab @ Pt).sum(axis=0)).ravel() / nb
    mif = np.asarray((Bb @ Pt).sum(axis=0)).ravel() / nb
    proj = np.zeros(len(types)); np.add.at(proj, member[pre_cells], 1)
    total = float(raw.sum())
    df = pd.DataFrame({"pre_type": types, "n_pre_cells": proj.astype(int), "entries": entries.astype(int), "entries_nonzero": entries_nz.astype(int),
                       "raw_count": raw, "share_of_b_input": raw / total if total > 0 else np.nan,
                       "nt": [str(pd.Series(nt[np.flatnonzero((ty == t))]).mode().iloc[0]) for t in types],
                       "mv_per_post_volley": m, "mv_per_post_volley_if_signed": np.where(m != 0, np.sign(m), 1.0) * mif,
                       "mv_per_pair": np.where(entries_nz > 0, m * nb / np.maximum(entries_nz, 1), np.nan),
                       "mv_per_pair_if_signed": np.where(entries > 0, np.where(m != 0, np.sign(m), 1.0) * mif * nb / np.maximum(entries, 1), np.nan),
                       "sign": np.sign(m).astype(int)})
    sil = [[] for _ in types]; partial = [[] for _ in types]
    for f, v in flags.items():
        share = np.asarray((Cb @ sp.diags(v) @ Pt).sum(axis=0)).ravel() / np.maximum(raw, 1e-9)
        df[f"{f}_frac"] = share
        for i, s in enumerate(share):
            if s >= 0.999:
                sil[i].append(f)
            elif s > 0:
                partial[i].append(f"{f}:{s:.2f}")
    for i in range(len(types)):
        if m[i] == 0 and raw[i] > 0 and not sil[i]:
            sil[i].append("zeroed")
    df["silent"] = ["|".join(s) for s in sil]; df["silent_partial"] = ["|".join(s) for s in partial]
    df = df.sort_values("raw_count", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------------------------- cell-level two-step
def two_step_by_type(c: cn.Connectome, ew: common.EffectiveWeights, a_idx, b_idx, Aif=None, counts=None, flags=None) -> pd.DataFrame:
    """Per intermediate type t (a cell postsynaptic to `a` and presynaptic to `b`): cells, nt, the mean per-pair weights
    a -> t and t -> b, and the two-step total per post cell, mean over b of the row sums of A[b, t] @ A[t, a] (mV^2 per
    spike -- cx_wedge.json 'ring_types'.two_step_epg_total_mV2 when a = b = EPG), plus the same through the if-signed
    magnitudes (silent links counted at the weight they would carry) and the link flags."""
    A = ew.A.tocsr()
    a_idx, b_idx = np.asarray(a_idx), np.asarray(b_idx)
    if counts is None:
        counts, _ = raw_counts(c)
    S = structure_matrix(counts)
    post_of_a = np.asarray(S[:, a_idx].sum(axis=1)).ravel() > 0
    pre_of_b = np.asarray(S[b_idx].sum(axis=0)).ravel() > 0
    X = np.flatnonzero(post_of_a & pre_of_b)
    ty = c.neurons.type.fillna(UNTYPED).to_numpy().astype(object)
    nt = c.neurons["nt"].fillna("unknown").to_numpy() if "nt" in c.neurons else np.array(["unknown"] * c.n)
    Ab, Aa = A[b_idx][:, X].toarray().astype(np.float64), A[X][:, a_idx].toarray().astype(np.float64)
    if Aif is None:
        Aif = if_signed_magnitudes(c, ew, counts)
    Bb, Ba = Aif[b_idx][:, X].toarray().astype(np.float64), Aif[X][:, a_idx].toarray().astype(np.float64)
    Ca, Cb = counts[X][:, a_idx].toarray().astype(np.float64), counts[b_idx][:, X].toarray().astype(np.float64)
    rows = []
    for t in np.unique(ty[X]):
        sel = ty[X] == t
        Kt = Ab[:, sel] @ Aa[sel]
        Kif = Bb[:, sel] @ Ba[sel]
        in_nz, out_nz = Aa[sel] != 0, Ab[:, sel] != 0
        cells = X[sel]
        r = {"type": str(t), "cells": int(sel.sum()), "nt": str(pd.Series(nt[cells]).mode().iloc[0]),
             "a_to_type_mv_per_pair": float(Aa[sel][in_nz].mean()) if in_nz.any() else 0.0,
             "type_to_b_mv_per_pair": float(Ab[:, sel][out_nz].mean()) if out_nz.any() else 0.0,
             "a_to_type_raw": float(Ca[sel].sum()), "type_to_b_raw": float(Cb[:, sel].sum()),
             "two_step_total_per_post_mv2": float(Kt.sum(axis=1).mean()),
             "two_step_total_per_post_mv2_if_signed": float(Kif.sum(axis=1).mean()),
             "a_to_type_pairs": int((Ca[sel] > 0).sum()), "type_to_b_pairs": int((Cb[:, sel] > 0).sum())}
        if flags is not None:
            fl = flags.set_index("index").reindex(cells)
            for f in SILENT_FLAGS:
                if f in fl:
                    v = pd.to_numeric(fl[f], errors="coerce").to_numpy(dtype=float)
                    r[f"{f}_frac"] = float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else 0.0
        r["silent"] = "|".join(f for f in SILENT_FLAGS if r.get(f"{f}_frac", 0) >= 0.999)
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.reindex(df.two_step_total_per_post_mv2.abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df


def wedge_profiles(c: cn.Connectome, ew: common.EffectiveWeights, a_idx, b_idx, groups=None) -> dict:
    """cx_wedge.md section 3 as a function: for each group of intermediate types (regex; default WEDGE_GROUPS) the
    cell-level two-step matrix A[b, X_g] @ A[X_g, a] and the direct A[b, a], aggregated by PB wedge (scripts/cx_wedge.py's
    RING16 order, glomerulus() and aggregate()) into a 16-wedge matrix (rows post wedge, cols pre wedge) and the mean over
    ring-distance bands 0..8 (wedges of 22.5 deg). Cells of a / b without a PB glomerulus in their instance name are
    dropped from the aggregation (counted in 'dropped')."""
    cxw = _cx_wedge()
    groups = dict(groups or WEDGE_GROUPS)
    A = ew.A.tocsr()
    a_idx, b_idx = np.asarray(a_idx), np.asarray(b_idx)
    inst = c.neurons.instance.fillna("").to_numpy()
    ty = c.neurons.type.fillna("").to_numpy()

    def positions(idx):
        gl = [cxw.glomerulus(inst[i]) for i in idx]
        ok = np.array([g in cxw.POS16 for g in gl])
        pos = np.array([cxw.POS16[g] if g in cxw.POS16 else -1 for g in gl], dtype=float)
        return pos, ok

    pa, oka = positions(a_idx); pb, okb = positions(b_idx)
    a2, b2 = a_idx[oka], b_idx[okb]
    bin16 = lambda x: np.asarray(np.round(x), int) % 16
    d = cxw.ring_dist(np.arange(16), np.arange(16))
    S = structure_matrix(abs(A))
    post_of_a = np.asarray(S[:, a2].sum(axis=1)).ravel() > 0
    pre_of_b = np.asarray(S[b2].sum(axis=0)).ravel() > 0
    X = np.flatnonzero(post_of_a & pre_of_b)
    Ab, Aa = A[b2][:, X].toarray().astype(np.float64), A[X][:, a2].toarray().astype(np.float64)
    out = {"ring16": list(cxw.RING16), "distance_wedges": list(range(9)), "dropped": {"a": int((~oka).sum()), "b": int((~okb).sum())},
           "groups": {}, "M16": {}, "profile": {}}
    K = {"direct": A[b2][:, a2].toarray().astype(np.float64)}
    for name, pat in groups.items():
        sel = np.array([bool(re.match(pat, t)) for t in ty[X]])
        K[name] = Ab[:, sel] @ Aa[sel]
        out["groups"][name] = {"regex": pat, "cells": int(sel.sum()), "types": sorted(set(ty[X][sel]))}
    for name, Kk in K.items():
        M16 = cxw.aggregate(Kk, pb[okb], pa[oka], 16, bin16)
        out["M16"][name] = np.round(M16, 3).tolist()
        out["profile"][name] = [float(M16[d == j].mean()) for j in range(9)]
    return out


# ---------------------------------------------------------------------------------------------- the tool
def _cpu_provenance(c, params, cache_dir=None, stimulus=None) -> dict:
    prov = common.provenance(c, params, None, fb=None, device="cpu", cache_dir=cache_dir, stimulus=stimulus, replicate_unit="n/a (structural)")
    ex = prov["execution"]
    ex["device"] = "cpu"                                  # the realised device of a structural tool: numpy / scipy on the host
    ex["device_name"] = platform.processor() or platform.machine()
    ex["backend"].update({"structural": True, "torch": False, "event_driven": False, "cuda_kernels": False,
                          "cuda_sparse": None, "metal": False, "cuda_graphs": False})
    ex["dt"]["lif_ms"] = float(params.dt)
    return prov


def paths(c, a, b, *, params=None, receptor=None, k_max=3, top=20, min_abs_mv=0.0, recording=None, frozen=None,
          level="type", exclude=(), rates_min_hz=common.NEVER_FIRING_HZ, counts=None, ew=None, aif=None, wedge=False,
          wedge_groups=None, contributions_per_link=200, cache_dir=None) -> common.Result:
    """Effective k-step signed gains from population `a` to population `b` (k <= k_max), ranked, silent links flagged.

    Level 'type': the type-level matrix M[post, pre] = mean over post cells of the summed effective input from all pre
    cells (mV per post cell per pre volley; `EffectiveWeights.type_matrix`), restricted to the types on any path of
    length <= k_max between a and b (the source nodes are 'a:<type>', one per type in a, source only; the target is
    the pooled node 'b', target only); a k-step walk a:t0 -> t1 -> ... -> b has gain prod M along it (mV^k per volley) and the tool lists the `top`
    walks by |gain| per k, separately for fully signed walks ('signed') and walks containing at least one silent link
    ('silent', ranked by the if-signed magnitude), with every link's sign, raw count and silence flags: 'sign0' (the
    presynaptic type carries no sign: its entries are explicit zeros; raw count from cache/sign0_counts.npz),
    'frozen' / 'pruned' (a rate unit of the optic lobe -- pass `frozen` = fb.optic.rate_idx, a FlyBrain, or 'static'
    for the superclass rule), 'never_firing' (max rate over `recording` below `rates_min_hz`). Level 'cell' keeps
    cells (endpoint cells are not intermediates); with k_max >= 2 it adds the per-type two-step table and, with
    `wedge=True`, cx_wedge's wedge matrices / ring-distance profiles (`wedge_groups`: {name: type regex}). `exclude`
    drops types from the intermediate set ('~regex' allowed). `counts` / `ew` accept precomputed paths.raw_counts /
    common.effective_weights. Validation: VALIDATION['paths'] (`validate()`).

    Tables: 'paths' (k, kind, rank, path, gain, gain_if_signed, signs, silent_links, raw_counts, pairs, link_mv),
    'links' (every distinct link on a listed walk, with mv per pair / per post volley, raw count, share of the post
    node's raw input, sign, silence flags and shares), 'contributions' (the cell-level edges of those links in Neurome
    fields, at most `contributions_per_link` per link), 'b_inputs' (every presynaptic type of b pooled: raw count and
    share of b's input, entries, mV per post volley / per pair signed and if-signed, nt, sign, silence flags --
    cx_shift.md section 1), 'two_step_by_type' and 'wedge_profile' (cell level).
    Summary: the strongest silent link per k (largest |if-signed mV per post volley| on a listed silent walk), the top
    signed and silent walk per k, the direct link a -> b, the dominant silent input of b (largest raw share among the
    silent presynaptic types), b's raw input total and its silent / sign-0 shares.
    """
    from .. import brain
    p = params or brain.LIFParams()
    k_max, top = int(k_max), int(top)
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    ia, ib = resolve_loose(c, a), resolve_loose(c, b)
    pop_a = common.Population("a", a, ia, c.neurons.bodyId.to_numpy()[ia])
    pop_b = common.Population("b", b, ib, c.neurons.bodyId.to_numpy()[ib])
    if pop_a.n == 0 or pop_b.n == 0:
        raise ValueError(f"empty population: a {pop_a.n} cells, b {pop_b.n} cells")
    if ew is None:
        ew = common.effective_weights(c, p, receptor)
    if counts is None:
        counts, sign0_available = raw_counts(c)
    else:
        sign0_available = None
    counts = counts.tocsr()
    Aif = if_signed_magnitudes(c, ew, counts) if aif is None else aif
    frozen_idx = frozen_from(frozen, c)
    rates = rates_from_recording(recording)
    g = build_graph(c, ew, pop_a.idx, pop_b.idx, counts=counts, k_max=k_max, level=level, exclude=exclude, frozen_idx=frozen_idx,
                    rates=rates, prune_frozen=bool(getattr(p, "prune_frozen", True)), rates_min_hz=rates_min_hz, min_abs_mv=min_abs_mv, Aif=Aif)
    walks = kbest_walks(g, k_max, top)

    # ---- tables
    path_rows, link_keys = [], {}
    for k in range(1, k_max + 1):
        for kind in ("signed", "silent"):
            ws = [w for w in walks.get(k, []) if w["kind"] == kind]
            for rank, w in enumerate(ws):
                labels = [g.labels[i] for i in w["nodes"]]
                gain_if = w["sign_prod"] * w["abs_gain"]
                gain = 0.0 if kind == "silent" else gain_if
                silent_links = [f"{L['pre']}->{L['post']}:{L['silent'] or L['silent_partial']}" for L in w["links"] if L["silent"] or L["silent_partial"]]
                path_rows.append({"k": k, "kind": kind, "rank": rank, "path": " -> ".join(labels), "gain": gain, "gain_if_signed": gain_if,
                                  "abs_gain_if_signed": w["abs_gain"],
                                  "signs": ",".join({1: "+", -1: "-", 0: "0"}[L["sign"]] for L in w["links"]),
                                  "silent_links": "; ".join(silent_links), "n_silent": sum(1 for L in w["links"] if L["silent"]),
                                  "raw_counts": ",".join(f"{L['raw_count']:.0f}" for L in w["links"]),
                                  "pairs": ",".join(str(L["pairs"]) for L in w["links"]),
                                  "link_mv": ",".join(f"{L['mv_per_post_volley_if_signed']:+.3f}" for L in w["links"]),
                                  "unit": f"mV^{k} per {'volley' if level == 'type' else 'spike'}"})
                for i in range(k):
                    link_keys.setdefault((w["nodes"][i + 1], w["nodes"][i]), w["links"][i])
    # the direct link a -> b (the whole population a pooled), always reported, even when it is not among the top walks
    direct = g.link(int(g.dst[0]), g.src, label="a") if level == "type" else None
    link_rows = []
    for (v, u), L in link_keys.items():
        link_rows.append(dict(L, pre_node=int(u), post_node=int(v), pre_nt=_node_nt(c, g, u)))
    links_df = pd.DataFrame(link_rows)
    if len(links_df):
        links_df = links_df.sort_values("mv_per_post_volley_if_signed", key=np.abs, ascending=False).reset_index(drop=True)
    contrib_df = _contributions(c, g, ew, link_keys, receptor, counts, frozen_idx, rates, p, rates_min_hz, contributions_per_link, Aif)
    # every presynaptic type of b, pooled (cx_shift.md section 1): where the target's input is silent
    b_in = inputs_of(c, ew, pop_b.idx, counts=counts, Aif=Aif, frozen_idx=frozen_idx, rates=rates,
                     prune_frozen=bool(getattr(p, "prune_frozen", True)), min_hz=rates_min_hz)

    # ---- result
    prov = _cpu_provenance(c, p, cache_dir, stimulus={"protocol": "structural", "params": {"a": _label_of(a), "b": _label_of(b), "k_max": k_max,
                                                                                            "level": level, "top": top, "min_abs_mv": min_abs_mv,
                                                                                            "exclude": list(exclude or ()), "rates_min_hz": rates_min_hz,
                                                                                            "recording": str(recording) if isinstance(recording, (str, Path)) else (None if recording is None else "<Recording>"),
                                                                                            "frozen": "static" if isinstance(frozen, str) else (None if frozen_idx is None else int(len(frozen_idx)))},
                                                   "control": None})
    res = common.Result.new(TOOL, prov)
    res.tool_version = TOOL_VERSION
    kinds = common.unit_kinds(c)
    res.add_population(pop_a, unit_kind=_kind_label(kinds[pop_a.idx]), keep_ids=pop_a.n <= 10_000)
    res.add_population(pop_b, unit_kind=_kind_label(kinds[pop_b.idx]), keep_ids=pop_b.n <= 10_000)
    res.add_table("paths", pd.DataFrame(path_rows))
    res.add_table("links", links_df)
    res.add_table("contributions", contrib_df)
    res.add_table("b_inputs", b_in)
    b_total = float(b_in.raw_count.sum()) if len(b_in) else 0.0
    silent_rows = b_in[b_in.silent != ""] if len(b_in) else b_in
    summary = {"level": level, "k_max": k_max, "top": top, "n_nodes": len(g.labels), "n_source_nodes": int(len(g.src)), "n_intermediate": int(len(g.mid)),
               "n_links": int(g.Mif.nnz), "sign0_counts_available": sign0_available, "a_cells": pop_a.n, "b_cells": pop_b.n,
               "a_types": sorted(set(np.asarray(c.neurons.type.fillna(UNTYPED).to_numpy())[pop_a.idx].tolist())),
               "b_raw_input_total": b_total,
               "b_silent_input_share": float(silent_rows.raw_count.sum() / b_total) if b_total > 0 else np.nan,
               "b_sign0_input_share": float(b_in.raw_count.mul(b_in["sign0_frac"]).sum() / b_total) if (b_total > 0 and "sign0_frac" in b_in) else 0.0,
               "dominant_silent_input_of_b": (silent_rows.iloc[0].to_dict() if len(silent_rows) else None),
               "direct": direct, "strongest_silent_link_per_k": {}, "top_walk_per_k": {}, "top_silent_walk_per_k": {},
               "n_walks": {k: len(v) for k, v in walks.items()}}
    for k in range(1, k_max + 1):
        signed = [r for r in path_rows if r["k"] == k and r["kind"] == "signed"]
        silent = [r for r in path_rows if r["k"] == k and r["kind"] == "silent"]
        summary["top_walk_per_k"][k] = signed[0] if signed else None
        summary["top_silent_walk_per_k"][k] = silent[0] if silent else None
        best = None
        for r in silent:
            w = next(w for w in walks[k] if w["kind"] == "silent" and " -> ".join(g.labels[i] for i in w["nodes"]) == r["path"])
            for L in w["links"]:
                if L["silent"] and (best is None or abs(L["mv_per_post_volley_if_signed"]) > abs(best["mv_per_post_volley_if_signed"])):
                    best = dict(L, on_path=r["path"], path_gain_if_signed=r["gain_if_signed"])
        summary["strongest_silent_link_per_k"][k] = best
    if level == "cell" and k_max >= 2:
        flags = common.silent_flags(c, np.arange(c.n), frozen_idx=frozen_idx, prune_frozen=bool(getattr(p, "prune_frozen", True)), rates=rates, min_hz=rates_min_hz)
        ts = two_step_by_type(c, ew, pop_a.idx, pop_b.idx, Aif=Aif, counts=counts, flags=flags)
        res.add_table("two_step_by_type", ts)
        if wedge:
            wp = wedge_profiles(c, ew, pop_a.idx, pop_b.idx, wedge_groups)
            res.add_table("wedge_profile", pd.DataFrame([{"group": name, **{f"d{j}": v[j] for j in range(9)}} for name, v in wp["profile"].items()]))
            summary["wedge"] = {k: wp[k] for k in ("ring16", "dropped", "groups")}
            res.files["M16"] = wp["M16"]
    res.summary = summary
    res.files.update({"generator": "flyverse.interp.paths.paths (scripts/interp_paths.py)", "reference": "docs/audits/cx_wedge.json",
                      "effective_weights_md5": ew.md5, "shaped_md5": ew.shaped_md5})
    return res


def _kind_label(kinds) -> str:
    u = sorted(set(np.asarray(kinds).tolist()))
    return u[0] if len(u) == 1 else "|".join(u)


def _node_nt(c, g: NodeGraph, u: int) -> str:
    cells = g.node_cells[u]
    if "nt" not in c.neurons or len(cells) == 0:
        return "unknown"
    return str(pd.Series(c.neurons["nt"].fillna("unknown").to_numpy()[cells]).mode().iloc[0])


def _contributions(c, g: NodeGraph, ew, link_keys, receptor, counts, frozen_idx, rates, p, min_hz, per_link, Aif) -> pd.DataFrame:
    """Cell-level rows (Neurome fields) of every link on a listed walk: the strongest `per_link` entries by if-signed
    magnitude; every sign-0 entry counts as an entry (value 0, value_if_signed given)."""
    rows = []
    fp = common.connectome_fingerprint(c)["md5"]
    norm = f"input_norm alpha {p.input_norm_alpha} ref {p.input_norm_ref}"
    for (v, u), L in link_keys.items():
        pre_cells, post_cells = g.node_cells[u], g.node_cells[v]
        if len(pre_cells) * len(post_cells) == 0:
            continue
        flags = common.silent_flags(c, pre_cells, frozen_idx=frozen_idx, prune_frozen=bool(getattr(p, "prune_frozen", True)), rates=rates, min_hz=min_hz)
        df = common.links(c, ew, pre_cells, post_cells, receptor=receptor, counts=counts, flags=flags)
        if len(df) == 0:
            continue
        # `silent` comes from common.links: its flags are booleans and a not-evaluated never_firing is False
        # (docs/INTERP.md 11, defect 2, closed -- this tool used to rebuild the string to undo bool(NaN) == True).
        mag = np.asarray(Aif[df.post_index.to_numpy(), df.pre_index.to_numpy()]).ravel()
        df["value_if_signed"] = np.where(df.effective_mv != 0, df.effective_mv, mag)
        df = df.reindex(df.value_if_signed.abs().sort_values(ascending=False).index).head(int(per_link))
        for r in df.itertuples(index=False):
            rows.append({"body_pre": r.body_pre, "body_post": r.body_post, "pre_type": r.pre_type, "post_type": r.post_type,
                         "value": float(r.effective_mv), "value_if_signed": float(r.value_if_signed), "kind": "effective_weight_mV",
                         "sign_rule": r.sign_rule, "gain_rule": r.gain_rule, "normalisation": norm, "reference_graph": fp, "window": None,
                         "synaptic_pair_count": float(r.synaptic_pair_count), "silent": r.silent, "link": f"{L['pre']}->{L['post']}",
                         "pre_index": int(r.pre_index), "post_index": int(r.post_index)})
    return pd.DataFrame(rows, columns=common.EXPORT_TABLES["contributions"] + ["value_if_signed", "silent", "link", "pre_index", "post_index"])


# ---------------------------------------------------------------------------------------------- validation
ROTATION_SOURCES = ["LNO1", "LNO2", "LNOa", "SpsP", "PS196_b", "~^LAL"]
PEN = ["~^PEN_a", "~^PEN_b"]          # the MaleCNS type strings are 'PEN_a(PEN1)' / 'PEN_b(PEN2)': a literal 'PEN_a|PEN_b' selects nothing


def _close(x, ref, rel=5e-3, abs_tol=0.0) -> bool:
    try:
        return bool(np.isfinite(x) and abs(float(x) - float(ref)) <= max(rel * abs(float(ref)), abs_tol))
    except (TypeError, ValueError):
        return False


def validate(c, params=None, out_dir=None, log=print) -> dict:
    """VALIDATION['paths'] against the cache: (1) rotation sources -> PEN at the type level, k = 3: GLNO -> PEN 84
    entries / 16,371 raw synapses / 19.4 % of PEN's raw input / sign 0 / 16.5 mV per pair if signed, and the LNO /
    SpsP links at 0-13 synapses (cx_glno.md 1, cx_shift.md 1); (2) the one-step ring weights EPG -> PEN +5.05 per pair
    / +79.9 per PEN volley, PEN -> EPG +7.98, Delta7 -> PEN -4.70, Delta7 -> EPG -2.46 (cx_wedge.md 2); (3) the cell-level
    EPG -> EPG two-step through the ring: per-type totals (ExR6 -15,792, ER4m -14,097, ExR4 -3,946, ER6 -3,608, ExR5
    -3,276 mV^2) and the wedge profiles (PEN +2,465 at distance 0, Delta7 -433 at 8, ER/ExR -2,760..-3,210 at every
    distance). Returns {'measured', 'status', 'checks', 'results' (the Result objects, saved to out_dir when given)}."""
    import json
    from .. import brain
    p = params or brain.LIFParams()
    ew = common.effective_weights(c, p)
    counts, ok = raw_counts(c)
    if not ok:
        warnings.warn("cache/sign0_counts.npz is not available: sign-0 links carry a raw count of 0 and the GLNO checks cannot pass")
    aif = if_signed_magnitudes(c, ew, counts)
    with open(common.ROOT / "docs" / "audits" / "cx_wedge.json", encoding="utf-8") as f:
        ref = json.load(f)
    measured, checks, results = {}, [], {}
    kw = dict(params=p, ew=ew, counts=counts, aif=aif, frozen="static")

    def check(name, value, reference, rel=5e-3, abs_tol=0.0):
        good = _close(value, reference, rel, abs_tol)
        checks.append({"check": name, "measured": None if value is None else float(value), "reference": float(reference), "ok": good})
        shown = "None" if value is None else f"{float(value):+.4g}"
        log(f"  {'ok ' if good else 'NO '} {name}: measured {shown} vs reference {float(reference):+.4g}")
        return good

    log("[1] rotation sources -> PEN, type level, k = 3")
    r1 = paths(c, ROTATION_SOURCES, PEN, k_max=3, top=20, **kw)
    results["rot_pen"] = r1
    b_in = r1.table("b_inputs").set_index("pre_type")
    m1 = {}
    if "GLNO" in b_in.index:
        L = b_in.loc["GLNO"]
        m1 = {"entries": int(L.entries), "raw_synapses": float(L.raw_count), "share_of_PEN_input": float(L.share_of_b_input), "sign": int(L.sign),
              "mv_per_pair_if_signed": float(abs(L.mv_per_pair_if_signed)), "silent": str(L.silent), "mv_per_post_volley_if_signed": float(abs(L.mv_per_post_volley_if_signed)),
              "rank_among_PEN_inputs": int(L["rank"])}
        check("GLNO->PEN entries", m1["entries"], 84, rel=0)
        check("GLNO->PEN raw synapses", m1["raw_synapses"], 16371, rel=0)
        check("GLNO->PEN share of PEN raw input", m1["share_of_PEN_input"], 0.194, rel=0, abs_tol=0.0005)
        check("GLNO->PEN sign", m1["sign"], 0, rel=0)
        check("GLNO->PEN mV per pair if signed", m1["mv_per_pair_if_signed"], 16.5, rel=5e-3)
        check("GLNO->PEN mV per PEN per GLNO volley if signed", m1["mv_per_post_volley_if_signed"], 33.0, rel=5e-3)
        checks.append({"check": "GLNO->PEN flagged sign0", "measured": m1["silent"], "reference": "sign0", "ok": "sign0" in m1["silent"]})
        log(f"  {'ok ' if 'sign0' in m1['silent'] else 'NO '} GLNO->PEN silent flags: {m1['silent']!r}")
        check("GLNO is PEN's largest input by raw synapses (rank)", m1["rank_among_PEN_inputs"], 1, rel=0)
    else:
        checks.append({"check": "GLNO->PEN link present", "measured": None, "reference": 1, "ok": False}); log("  NO  GLNO is not among PEN's presynaptic types")
    dom = r1.summary["dominant_silent_input_of_b"]
    good = dom is not None and dom["pre_type"] == "GLNO"
    checks.append({"check": "dominant silent input of PEN is GLNO", "measured": None if dom is None else dom["pre_type"], "reference": "GLNO", "ok": good})
    log(f"  {'ok ' if good else 'NO '} dominant silent input of PEN: {None if dom is None else dom['pre_type']}")
    # cx_shift.md section 1: PEN raw input 84,572; 21.9 % from sign-0 cells; the top-10 presynaptic types by raw synapses
    check("PEN raw input total", r1.summary["b_raw_input_total"], 84572, rel=0)
    check("PEN input share from sign-0 cells", r1.summary["b_sign0_input_share"], 0.219, rel=0, abs_tol=0.0005)
    top10_ref = {"GLNO": 16371, "EPG": 12330, "PEN_b": 9291, "PEN_a": 8625, "Delta7": 7239, "ExR4": 7122, "ExR6": 5271, "LPsP": 2725, "IbSpsP": 2637, "PEG": 2424}
    top10 = {}
    for t, ref_raw in top10_ref.items():
        key = next((i for i in b_in.index if i == t or i.startswith(t + "(")), None)      # PEN_a(PEN1) / PEN_b(PEN2) in MaleCNS
        top10[t] = None if key is None else float(b_in.loc[key, "raw_count"])
        check(f"PEN input from {t} (raw synapses)", top10[t], ref_raw, rel=0)
    m1["top10_inputs_raw"] = top10
    m1["b_silent_input_share"] = r1.summary["b_silent_input_share"]
    # the silent walks: GLNO -> PEN must be on a listed silent walk at k = 2 (PS196_b / LAL -> GLNO -> PEN), and it is the
    # strongest silent link ONTO PEN on any listed walk; the strongest silent link anywhere on the walks is reported as data
    pt = r1.table("paths")
    via_glno = pt[(pt.kind == "silent") & pt.path.str.contains("-> GLNO -> b")]
    good = len(via_glno) > 0
    checks.append({"check": "a -> GLNO -> PEN is a listed silent walk", "measured": via_glno.path.tolist()[:5], "reference": "a:<type> -> GLNO -> b", "ok": good})
    log(f"  {'ok ' if good else 'NO '} silent walks through GLNO: {via_glno.path.tolist()[:5]}")
    m1["silent_walks_via_GLNO"] = via_glno[["k", "rank", "path", "gain_if_signed"]].to_dict("records")
    links = r1.table("links")
    onto_b = links[(links.post == "b") & (links.silent != "")] if len(links) else links
    best_onto_b = None if len(onto_b) == 0 else onto_b.iloc[int(np.argmax(np.abs(onto_b.mv_per_post_volley_if_signed.to_numpy())))]
    good = best_onto_b is not None and best_onto_b.pre == "GLNO"
    checks.append({"check": "strongest silent link onto PEN on the listed walks is GLNO", "measured": None if best_onto_b is None else str(best_onto_b.pre), "reference": "GLNO", "ok": good})
    log(f"  {'ok ' if good else 'NO '} strongest silent link onto PEN on the listed walks: {None if best_onto_b is None else best_onto_b.pre}")
    strongest = r1.summary["strongest_silent_link_per_k"]
    m1["strongest_silent_link_per_k"] = {k: (None if v is None else f"{v['pre']}->{v['post']} [{v['silent']}] {v['mv_per_post_volley_if_signed']:+.1f} mV per post volley if signed, on {v['on_path']}") for k, v in strongest.items()}
    for k, v in strongest.items():
        log(f"      strongest silent link k={k}: {m1['strongest_silent_link_per_k'][k]}")
    m1["top_silent_walk_per_k"] = {k: (None if v is None else v["path"]) for k, v in r1.summary["top_silent_walk_per_k"].items()}
    m1["top_walk_per_k"] = {k: (None if v is None else f"{v['path']} {v['gain']:+.4g}") for k, v in r1.summary["top_walk_per_k"].items()}
    # LNO / SpsP / PS196_b -> PEN at 0-13 raw synapses (cx_shift.md 1: 1 / 6 / 0 / 13 / 0)
    ref_direct = {"LNO1": 1, "LNO2": 6, "LNOa": 0, "SpsP": 13, "PS196_b": 0}
    g_direct = {}
    for t in ref_direct:
        idx = common.resolve(c, t)
        g_direct[t] = float(counts[common.resolve(c, PEN)][:, idx].sum()) if len(idx) else float("nan")
        check(f"{t} -> PEN raw synapses", g_direct[t], ref_direct[t], rel=0)
    m1["raw_syn_onto_PEN_by_source_type"] = g_direct
    d = r1.summary["direct"]
    m1["direct_a_to_PEN"] = {"raw_count": d["raw_count"], "pairs": d["pairs"], "mv_per_post_volley": d["mv_per_post_volley"]}
    measured["rotation_to_PEN"] = m1

    log("[2] one-step ring weights (mV per pair / per post volley)")
    m2 = {}
    for name, (a, b, key) in {"EPG->PEN": ("EPG", PEN, "EPG->PEN"), "PEN->EPG": (PEN, "EPG", "PEN->EPG"), "Delta7->PEN": ("Delta7", PEN, "Delta7->PEN"),
                              "Delta7->EPG": ("Delta7", "EPG", "Delta7->EPG")}.items():
        r = paths(c, a, b, k_max=1, top=5, **kw)
        results[f"one_{key}"] = r
        d = r.summary["direct"]
        m2[name] = {"pairs": d["pairs"], "mv_per_pair": d["mv_per_pair"], "mv_per_post_volley": d["mv_per_post_volley"]}
        o = ref["one_step"][key]
        check(f"{name} pairs", d["pairs"], o["pairs"], rel=0)
        check(f"{name} mV per pair", d["mv_per_pair"], o["mean_per_pair"])
        check(f"{name} mV per post volley", d["mv_per_post_volley"], o["total_per_post"])
    measured["one_step"] = m2

    log("[3] EPG -> EPG two-step at the cell level, aggregated by wedge")
    r3 = paths(c, "EPG", "EPG", k_max=2, top=20, level="cell", wedge=True, **kw)
    results["epg_loop"] = r3
    ts = r3.table("two_step_by_type").set_index("type")
    m3 = {"two_step_by_type": {}, "profile": {}}
    for t in ["ExR6", "ER4m", "ExR4", "ER6", "ExR5", "ER2_c", "ER4d"]:
        v = float(ts.loc[t, "two_step_total_per_post_mv2"]) if t in ts.index else float("nan")
        m3["two_step_by_type"][t] = v
        check(f"EPG->{t}->EPG two-step total per EPG (mV^2)", v, ref["ring_types"][t]["two_step_epg_total_mV2"])
    wp = r3.table("wedge_profile").set_index("group")
    prof_ref = ref["profile16_by_distance"]
    for grp in ("PEN", "PEG", "Delta7", "Ring", "direct"):
        vals = [float(wp.loc[grp, f"d{j}"]) for j in range(9)]
        m3["profile"][grp] = vals
        for j in (0, 4, 8):
            check(f"{grp} profile at distance {j}", vals[j], prof_ref[grp][j], rel=5e-3, abs_tol=0.05)
    ring = m3["profile"]["Ring"]
    good = min(ring) >= -3215 and max(ring) <= -2755
    checks.append({"check": "ER/ExR loop -3,210..-2,760 mV^2 at every distance", "measured": [round(x) for x in ring], "reference": [-3210, -2760], "ok": good})
    log(f"  {'ok ' if good else 'NO '} ER/ExR profile: {[round(x) for x in ring]}")
    measured["epg_loop"] = m3

    n_ok = sum(1 for x in checks if x["ok"]); n = len(checks)
    status = "reproduced" if n_ok == n else "not reproduced"
    log(f"validation: {n_ok}/{n} checks ok -> {status}")
    for key, r in results.items():
        r.validation = dict(r.validation, measured=measured, status=status, checks=checks,
                            files={"validated_by": "flyverse.interp.paths.validate", "reference_file": "docs/audits/cx_wedge.json"})
        if out_dir is not None:
            r.save(Path(out_dir) / f"validate_{key.replace('->', '_to_')}.json")
    return {"measured": measured, "status": status, "checks": checks, "n_ok": n_ok, "n": n, "results": results}
