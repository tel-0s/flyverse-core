"""trace -- where along the synaptic depth from a sensory population is a stimulus lost? (docs/INTERP.md 4.2)

The tool takes three arms of independent runs (stimulus, matched control, and control-again = the null), each a
`common.Recording` of per-cell time-means over the scored window (the `record` subcommand of scripts/interp_trace.py
writes them on the cluster), scores every cell type by one statistic per run, compares the (stimulus - control) draws
against the (null - control) draws with `common.compare` (z on the null SD, Welch, exact U / p, the scatter rule),
orders the types by their synaptic depth from the source population, names the first stage at which the stimulus is
lost, and decomposes that stage's input into the carrying and non-carrying presynaptic types (the reading
docs/audits/optic_measures.md 5.3 made by hand for T3: ON and OFF carriers arriving with opposite figures through
excitatory synapses).

Statistics (`stat`), every one the statistic of an existing probe:
  * 'best_cell' -- max over the type's cells of the per-cell (A - B) time-mean: probe_object_sweep's
    `diff_abs_best_cell_mean` (rate units, |deviation|) / `diff_max_over_cells_mean_mv` (spiking, drive); the
    statistic of docs/audits/object_sweep.md 8.4 / 8.7 and the trace validation target;
  * 'mean' -- the population mean of the per-cell (A - B) (Hz for spiking types: the LH odour gate);
  * 'figure_z' -- probe_figure_stages' retinotopic figure: (A - B) in the object columns minus (A - B) in the
    background columns, z against the background scatter (needs the column map in the recording's meta);
  * 'dprime' -- screen.rank's d' between the stimulus and control condition over the per-frame pooled series
    (the `<run>_series` recording the CLI writes alongside).

Depth. Over |A| > 0 literally, the photoreceptors touch nearly every medulla type and the whole optic lobe sits at
depth 2 (measured on the cache: Mi1, T2, T3, LC11, LPLC2, DNp01 all at 2), so the depth is computed on the type-level
input-share graph: type P -> type Q is an edge when the mean input a Q cell receives from all P cells is at least
`min_share` of its total |input| (`EffectiveWeights`-derived; `min_share = 0` is the literal contract rule). At the
default 0.02 the photoreceptor trace reads L 1 / Mi-Tm 2 / T2-T3-T4-LC 3 / DNp01 4 and the ORN trace ORN 0 / PN 1 /
LH 1-2 / DN 2-4. `stage_table` ('family', a {type: stage} dict, or the Nern 2025 table path) adds
probe_figure_stages' stage labels for the optic lobe.

First stage lost. Two rules, both reported: the contract's depth rule (`first_lost_depth`: the smallest depth at
which no type has verdict 'result' while a type one step shallower has) and the input rule (`lost_types`: a
non-carrying type that receives >= `min_share` of its input from a carrying type; the first lost stage is the
smallest depth among them). `decompose_at = 'first_lost'` decomposes the lost types at that depth (the input rule,
falling back to the depth rule); an int selects a depth, a list of type names selects them directly. The
decomposition is the type-level input table of the lost types (share, sign, mV per volley, raw synapses, the input's
own verdict and figure, and the signed-share x input-figure term of optic_measures 5.3) plus, when
flyverse/interp/decompose.py exists, its static / dynamic tables prefixed 'lost_'.

CPU only. torch is never imported here; the GPU half is scripts/interp_trace.py record.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import common
from .common import (Recording, Result, compare, effective_weights, resolve, population, body_str, to_jsonable,
                     MIN_REPLICATES)
from .. import connectome as cn
from ..screen import TypeRecorder, rank

STATS = ("best_cell", "mean", "figure_z", "dprime")
DEFAULT_MIN_SHARE = 0.02
CELL_QUANTITIES = ("rate_hz", "drive_mv", "drive_mv_abs", "optic_dr", "optic_dr_abs")
UNITS_OF = {"rate_hz": "Hz", "drive_mv": "mV", "drive_mv_abs": "mV", "optic_dr": "rate units", "optic_dr_abs": "rate units"}
NEUROME_QUANTITY = {"rate_hz": "output_Hz", "drive_mv": "upstream_drive_mV", "drive_mv_abs": "upstream_drive_abs_mV",
                    "optic_dr": "rate_deviation", "optic_dr_abs": "rate_deviation_abs"}
TWO_ROW_TYPES = ("LC11", "LC10a")          # never pooled: an upstream_drive_mV and an output_Hz row per body

# ---- stages: scripts/probe_figure_stages.py's STAGES / FAMILY_STAGE / NERN_GROUP_STAGE (copied verbatim so the CPU
# analysis does not import that script, which initialises pygame and torch; the test checks the copy against the script)
STAGES = {1: "1 lamina (L1-L5, C2/C3, Lawf, T1)", 2: "2a medulla intrinsic (Mi, Dm, Pm, Cm, Sm)",
          3: "2b medulla -> lobula projection (Tm, TmY, MeLo)", 4: "3 T cells + lobula-plate connecting (T2/T3, T4/T5, CT1, Tlp, Y)",
          5: "4 lobula / lobula-plate intrinsic (Li, LPi)", 6: "5 visual projection neurons (spiking; drive)",
          7: "6 visual centrifugal (spiking; drive)"}
NERN_GROUP_STAGE = {1: 1, 5: 2, 6: 2, 7: 2, 8: 2, 2: 3, 3: 3, 4: 4, 9: 5, 10: 5, 11: 5, 12: 5}
FAMILY_STAGE = [(r"^(L[1-5]|C[23]|Lawf[12]|T1|Lai)$", 1), (r"^(Mi|Dm|Pm|Cm|Sm)", 2), (r"^(Tm(?!23|24|Y)|TmY(?!20)|MeLo)", 3),
                (r"^(T2a?|T3|T4[abcd]|T5[abcd]|T4_unclear|T5a_unclear|CT1|TmY20|Am1|LOLP1|Tlp|Y\d|Y_unclear)", 4),
                (r"^(Li|LPi|Tm23|Tm24|LT33|aMe6b|HBeyelet)", 5)]


def stage_of(t: str, superclass: str, table: dict | None = None):
    """probe_figure_stages.stage_of: the Nern figure group when the table has it, else the type-name family rule."""
    if superclass == "visual_projection":
        return 6
    if superclass == "visual_centrifugal":
        return 7
    if table and t in table and table[t] is not None:
        return table[t]
    for pat, s in FAMILY_STAGE:
        if re.match(pat, t):
            return s
    return None


def load_stage_table(spec) -> dict | None:
    """None / 'none' -> no stages; 'family' -> {} (the family rule); a dict -> itself; a path -> the Nern 2025 table
    read as probe_figure_stages.stage_table does (figure_group -> stage, VPN / VCN by main_groups)."""
    if spec is None or spec == "none":
        return None
    if spec == "family":
        return {}
    if isinstance(spec, dict):
        return dict(spec)
    x = pd.read_excel(spec)
    out = {}
    for t, g, mg in zip(x["type"], x["figure_group"], x["main_groups"]):
        try:
            out[str(t)] = NERN_GROUP_STAGE.get(int(g))
        except (TypeError, ValueError):
            out[str(t)] = 6 if str(mg) == "VPN" else 7 if str(mg) == "VCN" else None
    return out


# ---------------------------------------------------------------------------------------------- type-level graph
def full_raw_counts(c: cn.Connectome) -> tuple[sp.csr_matrix, bool]:
    """Raw, unsigned, uncapped synapse count per stored entry of c.W (post x pre) -- `common.raw_counts`.

    |W.data| on the signed entries and, on the explicit-zero entries (sign-0 presynaptic cells), the count from
    cache/sign0_counts.npz. The private merge this tool carried while the shared accessor substituted the sign-0
    array for the whole vector (every lost_inputs.raw_synapses_per_post read 0.0) is gone: docs/INTERP.md 11,
    defect 1, closed. Returns (counts, sign0_available)."""
    return common.raw_counts(c)


class TypeGraph:
    """The type-level input matrix of the shaped weights: M[post_type, pre_type] = mean over post cells of the summed
    signed effective input from all pre cells (mV per post cell per presynaptic volley; EffectiveWeights.type_matrix
    for every type pair at once), `tot` = the same over |A| summed over pre types, `share` = |M| / tot, and the raw
    synapse counts aggregated the same way (`raw`, synapses per post cell from the pre type)."""

    def __init__(self, c: cn.Connectome, ew: common.EffectiveWeights, counts: sp.csr_matrix | None = None):
        self.c = c
        ty = c.neurons.type.fillna("").to_numpy()
        self.keys, inv = np.unique(ty, return_inverse=True)
        self.inv = inv
        nT = len(self.keys)
        self.n_cells = np.bincount(inv, minlength=nT)
        P = sp.csr_matrix((np.ones(c.n, np.float64), (np.arange(c.n), inv)), shape=(c.n, nT))
        Pn = sp.csr_matrix((1.0 / self.n_cells[inv], (inv, np.arange(c.n))), shape=(nT, c.n))
        A = ew.A.astype(np.float64)
        self.M = (Pn @ (A @ P)).tocsr()
        Mabs = (Pn @ (abs(A) @ P)).tocsr()
        self.tot = np.asarray(Mabs.sum(axis=1)).ravel()
        self.share = (sp.diags(1.0 / np.maximum(self.tot, 1e-12)) @ Mabs).tocsr()
        if counts is None:
            counts, _ = full_raw_counts(c)
        self.raw = (Pn @ (counts.astype(np.float64) @ P)).tocsr()
        self.raw_tot = np.asarray(self.raw.sum(axis=1)).ravel()
        self.pos = {k: i for i, k in enumerate(self.keys)}
        self.untyped = self.pos.get("")                          # the pooled untyped cells: never a node (they would short-circuit every depth)

    def depths(self, source_idx, min_share: float = DEFAULT_MIN_SHARE, depth_max: int = 6) -> np.ndarray:
        """(n_types,) shortest path length from the source types over edges with share >= min_share; -1 unreached."""
        S = self.share.tolil() if self.untyped is not None else self.share.copy()
        if self.untyped is not None:
            S[self.untyped, :] = 0.0; S[:, self.untyped] = 0.0
            S = S.tocsr()
        if min_share > 0:
            S.data[S.data < min_share] = 0.0
        S.eliminate_zeros()
        ST = S.T.tocsr()                                              # ST[pre_type, post_type]
        d = np.full(len(self.keys), -1, np.int64)
        src = np.unique(self.inv[np.asarray(source_idx)])
        src = src[src != self.untyped] if self.untyped is not None else src
        d[src] = 0
        front = src
        for k in range(1, int(depth_max) + 1):
            if not len(front):
                break
            nxt = np.unique(ST[front].indices)
            nxt = nxt[d[nxt] < 0]
            d[nxt] = k
            front = nxt
        return d

    def inputs(self, post_type: str, min_share: float = 0.01, top: int = 12) -> pd.DataFrame:
        """The presynaptic types of `post_type` with share >= min_share (top by share): pre_type, share, sign,
        mv_per_volley, raw_synapses_per_post."""
        i = self.pos[post_type]
        row = self.share[i]
        cols, sh = row.indices, row.data
        keep = (sh >= min_share) & (cols != self.untyped)
        cols, sh = cols[keep], sh[keep]
        order = np.argsort(-sh)[:top]
        cols, sh = cols[order], sh[order]
        mv = np.asarray(self.M[i, cols].todense()).ravel() if len(cols) else np.zeros(0)
        raw = np.asarray(self.raw[i, cols].todense()).ravel() if len(cols) else np.zeros(0)
        return pd.DataFrame({"pre_type": self.keys[cols], "share": sh, "sign": np.sign(mv).astype(int), "mv_per_volley": mv,
                             "raw_synapses_per_post": raw})


# ---------------------------------------------------------------------------------------------- per-cell statistics
def cell_means(rec: Recording, quantity: str, window=None) -> np.ndarray:
    """(n,) per-cell time-mean of `quantity` over the recording (or `window` = (start_s, end_s)); a '<q>_abs' quantity
    absent from a per-frame recording is the time-mean of |<q>|; NaN where the quantity is absent."""
    r = rec.window(*window) if (window is not None and rec.n_frames > 1) else rec
    if quantity in r.quantities:
        x = r.quantities[quantity]
    elif quantity.endswith("_abs") and quantity[:-4] in r.quantities and r.n_frames > 1:
        x = np.abs(r.quantities[quantity[:-4]])
    else:
        return np.full(len(rec.idx), np.nan)
    if x.ndim == 3:
        raise ValueError("trace works on single-fly recordings; use Recording.row(b) for a batch member")
    return np.asarray(x, dtype=np.float64).mean(axis=0)


def figure_stats(xA, xB, obj, bg) -> dict:
    """probe_figure_stages.figure_stats: (A - B) in the object cells minus (A - B) in the background cells; z against
    the background scatter (population sd + 1e-9)."""
    d = xA - xB
    d_obj = d[obj].mean(); d_bg = d[bg].mean(); sd = d[bg].std() + 1e-9
    return {"figure": float(d_obj - d_bg), "z": float((d_obj - d_bg) / sd), "sd_bg": float(sd),
            "resp_obj_A": float(xA[obj].mean()), "resp_obj_B": float(xB[obj].mean()), "bg_level_B": float(xB[bg].mean())}


def _group_stat(d: np.ndarray, inv: np.ndarray, n_types: int, stat: str, obj=None, bg=None, min_obj=5, min_bg=20):
    """Per-type statistic of a per-cell difference vector d (NaN -> excluded). Returns (value, aux) arrays."""
    ok = np.isfinite(d)
    if stat == "best_cell":
        out = np.full(n_types, -np.inf)
        dd = np.where(ok, d, -np.inf)
        np.maximum.at(out, inv, dd)
        out[~np.isfinite(out)] = np.nan
        return out, None
    if stat == "mean":
        s = np.bincount(inv[ok], weights=d[ok], minlength=n_types); n = np.bincount(inv[ok], minlength=n_types)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan), None
    if stat == "figure_z":
        o = ok & obj; b = ok & bg
        no = np.bincount(inv[o], minlength=n_types); nb = np.bincount(inv[b], minlength=n_types)
        so = np.bincount(inv[o], weights=d[o], minlength=n_types); sb = np.bincount(inv[b], weights=d[b], minlength=n_types)
        sb2 = np.bincount(inv[b], weights=d[b] ** 2, minlength=n_types)
        mo = so / np.maximum(no, 1); mb = sb / np.maximum(nb, 1)
        var = sb2 / np.maximum(nb, 1) - mb ** 2
        sd = np.sqrt(np.maximum(var, 0.0)) + 1e-9
        valid = (no >= min_obj) & (nb >= min_bg)
        fig = np.where(valid, mo - mb, np.nan); z = np.where(valid, (mo - mb) / sd, np.nan)
        return fig, z
    raise ValueError(f"unknown stat {stat!r}; choose from {STATS}")


#: The verdicts that make a type a CARRIER of the stimulus here: `common.compare`'s 'result', plus 'undetermined' --
#: a deterministic null (SD 0: the ORNs under a fixed plume are bit-identical across runs) leaves z undefined, so the
#: shared comparison declines to call it, and this tool's own rule is that a separation the exact rank test supports
#: at a magnitude (+83 Hz for ORN_DM1) carries. The private 'override the verdict to result' that used to do this
#: inside the per-type loop is gone (docs/INTERP.md 11, defect 4, closed); the verdict column stays compare's.
CARRIER_VERDICTS = ("result", "undetermined")


def carries(verdict) -> bool:
    """Does this per-type verdict mean 'the stimulus is still here'? (CARRIER_VERDICTS)"""
    return str(verdict) in CARRIER_VERDICTS


#: The smallest two-sided exact Mann-Whitney p two arms of n_a and n_b runs can reach (3 v 3 floors at 0.10, 4 v 4 at
#: 0.029, 5 v 5 at 0.0079). This lived here while `common.compare` ignored it; it is `common.p_floor` now and
#: `compare` itself says 'underpowered' while the floor exceeds alpha (docs/INTERP.md 11, defect 3, closed).
p_floor = common.p_floor


def _quantity_for(kind: str, stat: str, quantity, meta: dict) -> str:
    """The recorded quantity a type of unit kind `kind` is scored on."""
    if isinstance(quantity, dict):
        if kind in quantity:
            return quantity[kind]
    elif quantity is not None and kind == "spiking":
        return quantity
    if kind == "graded" or kind == "photoreceptor":
        return "optic_dr" if stat in ("figure_z", "mean") else "optic_dr_abs"
    q = (meta.get("default_quantity") or {}).get("spiking", "rate_hz")
    if stat == "best_cell" and q == "rate_hz" and "drive_mv" in meta.get("quantities", []) and meta.get("protocol") in ("object", "apple", "ball"):
        return "drive_mv"
    return q


# ---------------------------------------------------------------------------------------------- loading runs
def load_run(path) -> Recording:
    """A recording written by `scripts/interp_trace.py record` (<path>.npz + .json); the per-frame pooled series
    <path>_series.npz, when present, is attached as `.series`."""
    p = Path(str(path))
    if p.suffix in (".npz", ".json"):
        p = p.with_suffix("")
    rec = Recording.load(p)
    s = Path(str(p) + "_series")
    rec.series = Recording.load(s) if s.with_suffix(".npz").exists() else None
    rec.path = str(p)
    return rec


def load_runs(patterns) -> list:
    """Recordings for one or more glob patterns / paths (each matching '<run>.json' or '<run>.npz'; '_series' files
    are skipped), sorted by path."""
    if patterns is None:
        return []
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    files = []
    for pat in patterns:
        pat = str(pat)
        hits = sorted(glob.glob(pat)) or sorted(glob.glob(pat + ".json")) or sorted(glob.glob(pat + "*.json"))
        for h in hits:
            hp = Path(h)
            if hp.stem.endswith("_series") or hp.suffix not in (".json", ".npz"):     # the console .txt next to a run is not a run
                continue
            hp = hp.with_suffix("")
            if str(hp) not in files:
                files.append(str(hp))
    return [load_run(f) for f in files]


def _as_recordings(x) -> list:
    if x is None:
        return []
    if isinstance(x, Recording):
        return [x]
    if isinstance(x, (str, Path)):
        return load_runs(x)
    out = []
    for item in x:
        out.extend(_as_recordings(item))
    return out


# ---------------------------------------------------------------------------------------------- the tool
def trace(c, source, *, stimulus, control, null=None, params=None, optic_params=None, stat="best_cell", depth_max=6,
          stage_table=None, decompose_at="first_lost", quantity=None, min_cells=3, fb=None,
          min_share=DEFAULT_MIN_SHARE, ew=None, window=None, per_body="lost", top_inputs=12, input_min_share=0.01,
          max_lost=8, positive=None, negative=None, min_obj=5, min_bg=20, lost_min_carrier_share=0.2) -> Result:
    """Where along the synaptic depth from `source` is a stimulus lost? (module docstring; docs/INTERP.md 4.2)

    Parameters of the contract: `c` a Connectome; `source` a population spec (common.resolve grammar); `stimulus` /
    `control` / `null` lists of Recordings (or paths / globs; >= 3 independent runs each, paired by position: run r
    is (stimulus[r] - control[r]) against (null[r] - control[r]); without `null` the null draws are every ordered
    pair of distinct control runs, flagged `null_source = 'control_pairs'`); `params` / `optic_params` the model
    parameters the depth graph is built with (None = the recording's provenance, else the defaults); `stat` one of
    STATS; `depth_max` the deepest stage listed; `stage_table` None / 'family' / dict / path; `decompose_at`
    'first_lost' | int depth | list of types | None; `quantity` the recorded quantity spiking types are scored on
    (None = the recording's default; graded types always use optic_dr / optic_dr_abs); `min_cells` the smallest
    type scored; `fb` an optional FlyBrain (provenance when the recordings carry none).

    Extra keyword parameters (defaults keep the contract): `min_share` the type-level input share that makes an
    edge of the depth graph (0 = the literal |A| > 0 rule); `ew` a precomputed common.effective_weights; `window`
    (start_s, end_s) for per-frame recordings; `per_body` 'lost' | 'all' | 'none' rows of readout_per_body;
    `top_inputs` / `input_min_share` the lost-stage input table; `max_lost` the most lost types decomposed;
    `positive` / `negative` the arm names of the d' statistic (default the stimulus vs the control arm); `min_obj` /
    `min_bg` the smallest object / background cell counts a type needs for the figure statistic
    (probe_figure_stages' MIN_OBJ_CELLS 5 / MIN_BG_CELLS 20); `lost_min_carrier_share` the share of a non-carrying
    type's |input| that must come from carriers for it to count as lost (the input rule; 0.2 = a fifth).

    Returns a common.Result with tables 'per_type' (type, depth, stage, unit_kind, n_cells, quantity, stat, per-run
    values of both arms, stim_mean, ctrl_level, null_mean, diff, z, welch, U, p, verdict [, figure z per run and
    carry_runs]), 'lost_inputs' (the presynaptic decomposition of the lost types), 'readout_per_body' (Neurome
    fields; LC11 / LC10a two rows per body), and the decompose tool's tables prefixed 'lost_' when it is available.
    """
    stim = _as_recordings(stimulus); ctrl = _as_recordings(control); nul = _as_recordings(null)
    if not stim or not ctrl:
        raise ValueError("trace needs at least one stimulus and one control recording")
    if stat not in STATS:
        raise ValueError(f"stat {stat!r} not in {STATS}")
    n_pairs = min(len(stim), len(ctrl))
    meta0 = stim[0].meta
    # ---- the model the depth graph is built with: the recording's provenance when it carries one
    prov_rec = meta0.get("provenance")
    if params is None and prov_rec:
        params = params_from_provenance(prov_rec)
    if ew is None:
        ew = effective_weights(c, params)
    counts, sign0_available = full_raw_counts(c)
    tg = TypeGraph(c, ew, counts)
    src_idx = resolve(c, source)
    if len(src_idx) == 0:
        raise ValueError(f"source {source!r} selects no cells")
    depth = tg.depths(src_idx, min_share=min_share, depth_max=depth_max)
    kinds = common.unit_kinds(c, fb)
    sc = c.neurons.superclass.fillna("").to_numpy()
    stages = load_stage_table(stage_table)
    # ---- the recorded cells -> model indices; the type of each recorded cell
    idx0 = np.asarray(stim[0].idx)
    for r in stim[1:] + ctrl + nul:
        if len(r.idx) != len(idx0) or not np.array_equal(np.asarray(r.idx), idx0):
            raise ValueError("every recording must hold the same cells (same Recorder selection)")
    inv = tg.inv[idx0]                                                 # type id per recorded cell
    nT = len(tg.keys)
    n_rec = np.bincount(inv, minlength=nT)
    kind_of_type = np.array([kinds[idx0[inv == i]][0] if n_rec[i] else "spiking" for i in range(nT)], dtype=object)
    # ---- the quantity per type (graded vs spiking)
    q_of_type = np.array([_quantity_for(kind_of_type[i], stat, quantity, meta0) for i in range(nT)], dtype=object)
    q_names = sorted(set(q_of_type[n_rec > 0]))
    # ---- column masks for figure_z
    obj = bg = None
    col = np.asarray(meta0.get("column", []), dtype=np.int64)
    has_columns = len(col) == len(idx0) and "columns_obj" in meta0
    if stat == "figure_z" and not has_columns:
        raise ValueError("stat 'figure_z' needs meta['column'] (per recorded cell) and meta['columns_obj'] / ['columns_bg'] "
                         "in the stimulus recording (scripts/interp_trace.py record writes them for the object / apple protocols)")
    if has_columns:                                    # the retinotopic figure serves every stat's signed-figure reading of the lost stage
        in_obj = np.zeros(int(max(col.max(), 0)) + 1, bool); in_obj[np.asarray(meta0["columns_obj"], dtype=np.int64)] = True
        in_bg = np.zeros_like(in_obj); in_bg[np.asarray(meta0["columns_bg"], dtype=np.int64)] = True
        has = col >= 0
        obj = has & in_obj[np.clip(col, 0, len(in_obj) - 1)]; bg = has & in_bg[np.clip(col, 0, len(in_bg) - 1)]

    # ---- per-run per-type statistics of each arm
    def arm_values(a_recs, b_recs, pairs):
        vals = np.full((len(pairs), nT), np.nan); aux = np.full((len(pairs), nT), np.nan)
        for k, (ia, ib) in enumerate(pairs):
            if stat == "dprime":
                v = _dprime(a_recs[ia], b_recs[ib], tg, positive, negative)
                vals[k] = v
                continue
            for q in q_names:
                m_q = q_of_type[inv] == q
                xa = cell_means(a_recs[ia], q, window); xb = cell_means(b_recs[ib], q, window)
                d = np.where(m_q, xa - xb, np.nan)
                v, z = _group_stat(d, inv, nT, stat, obj, bg, min_obj, min_bg)
                sel = q_of_type == q
                vals[k, sel] = v[sel]
                if z is not None:
                    aux[k, sel] = z[sel]
        return vals, aux

    stim_pairs = [(r, r) for r in range(n_pairs)]
    if nul:
        null_pairs = [(r, r) for r in range(min(len(nul), len(ctrl)))]
        null_src = "null_runs"
        null_vals, null_aux = arm_values(nul, ctrl, null_pairs)
    else:
        null_pairs = [(i, j) for i in range(len(ctrl)) for j in range(len(ctrl)) if i != j]
        null_src = "control_pairs"
        null_vals, null_aux = arm_values(ctrl, ctrl, null_pairs)
    stim_vals, stim_aux = arm_values(stim, ctrl, stim_pairs)
    # levels: the mean over the stimulus / control runs of the pooled per-cell time-mean (context for the table)
    levels = {}
    for q in q_names:
        m_q = q_of_type[inv] == q
        xa = np.nanmean([cell_means(r, q, window) for r in stim[:n_pairs]], axis=0)
        xb = np.nanmean([cell_means(r, q, window) for r in ctrl[:n_pairs]], axis=0)
        for name, x in (("stim_level", xa), ("ctrl_level", xb)):
            ok = np.isfinite(x) & m_q
            s = np.bincount(inv[ok], weights=x[ok], minlength=nT); n = np.bincount(inv[ok], minlength=nT)
            lv = levels.setdefault(name, np.full(nT, np.nan))
            sel = (q_of_type == q) & (n > 0)
            lv[sel] = (s / np.maximum(n, 1))[sel]

    # ---- the per-type table
    rows = []
    for i in range(nT):
        if n_rec[i] < min_cells or not tg.keys[i]:
            continue
        if depth[i] < 0 or depth[i] > depth_max:
            continue
        sv = stim_vals[:, i]; nv = null_vals[:, i]
        if not np.isfinite(sv).any():
            continue
        cmp = compare(sv, nv)
        # every null draw identical (a deterministic input stage: the ORNs under a fixed plume): z is undefined (NaN in
        # the table; JSON has no inf) and `compare` says 'undetermined' -- read `diff` and `p`. The private override to
        # 'result' this tool used to apply here is gone (docs/INTERP.md 11, defect 4, closed).
        note = "null_sd_zero" if cmp.get("null_sd_zero") else None
        t = tg.keys[i]
        first_cell = idx0[inv == i][0]
        row = {"type": t, "depth": int(depth[i]), "stage": stage_of(t, sc[first_cell], stages) if stages is not None else None,
               "unit_kind": kind_of_type[i], "n_cells": int(n_rec[i]), "n_cells_type": int(tg.n_cells[i]), "quantity": q_of_type[i],
               "stat": stat, "stim_values": [float(v) for v in sv], "null_values": [float(v) for v in nv],
               "stim_mean": cmp["stim"]["mean"], "stim_sd": cmp["stim"]["sd"], "null_mean": cmp["null"]["mean"], "null_sd": cmp["null"]["sd"],
               "stim_level": float(levels.get("stim_level", np.full(nT, np.nan))[i]), "ctrl_level": float(levels.get("ctrl_level", np.full(nT, np.nan))[i]),
               "diff": cmp["diff"], "z": cmp["z"], "welch": cmp["welch"], "U": cmp["U"], "p": cmp["p"], "verdict": cmp["verdict"],
               "p_floor": p_floor(int(np.isfinite(sv).sum()), int(np.isfinite(nv).sum())), "note": note}
        if stat == "figure_z":
            zs = stim_aux[:, i]; zn = null_aux[:, i]
            row["figure_z_runs"] = [float(v) for v in zs]; row["null_figure_z_runs"] = [float(v) for v in zn]
            fin = zs[np.isfinite(zs)]
            row["carry_runs"] = int(((np.abs(fin) >= common.Z_RESULT) & (np.sign(fin) == np.sign(fin[0]) if len(fin) else False)).sum()) if len(fin) else 0
            row["null_carry_runs"] = int((np.abs(zn[np.isfinite(zn)]) >= common.Z_RESULT).sum())
        rows.append(row)
    per_type = pd.DataFrame(rows)
    if len(per_type):
        per_type = per_type.sort_values(["depth", "z"], ascending=[True, False], key=lambda s: s if s.name == "depth" else s.abs()).reset_index(drop=True)
    verdict_of = dict(zip(per_type.type, per_type.verdict)) if len(per_type) else {}
    depth_of = dict(zip(per_type.type, per_type.depth)) if len(per_type) else {}
    diff_of = dict(zip(per_type.type, per_type["diff"])) if len(per_type) else {}
    z_of = dict(zip(per_type.type, per_type.z)) if len(per_type) else {}

    # ---- first stage lost: the depth rule and the input rule
    carriers = [t for t, v in verdict_of.items() if carries(v)]
    first_lost_depth = None
    for d in range(1, depth_max + 1):
        here = [t for t in verdict_of if depth_of[t] == d]; before = [t for t in verdict_of if depth_of[t] == d - 1]
        if here and before and any(carries(verdict_of[t]) for t in before) and not any(carries(verdict_of[t]) for t in here):
            first_lost_depth = d
            break
    lost = []
    carrier_set = set(carriers)
    for t in verdict_of:
        if carries(verdict_of[t]) or depth_of[t] == 0:
            continue
        inp = tg.inputs(t, min_share=min_share, top=50)
        inp = inp[inp.pre_type.isin(carrier_set)]
        if len(inp) and float(inp.share.sum()) >= lost_min_carrier_share:
            lost.append({"type": t, "depth": depth_of[t], "verdict": verdict_of[t], "carrier_input_share": float(inp.share.sum()),
                         "carriers": list(inp.pre_type), "z": z_of[t]})
    lost_df = pd.DataFrame(lost)
    lost_depth = int(lost_df.depth.min()) if len(lost_df) else None
    if len(lost_df):
        lost_df = lost_df.sort_values(["depth", "carrier_input_share"], ascending=[True, False]).reset_index(drop=True)
    lost_top_by_share = list(lost_df.sort_values("carrier_input_share", ascending=False).type[:max_lost]) if len(lost_df) else []
    counts_by_depth = {int(d): {"carriers": int(g.verdict.isin(CARRIER_VERDICTS).sum()), "scored": int(len(g))} for d, g in per_type.groupby("depth")} if len(per_type) else {}
    counts_by_stage = ({str(st): {"carriers": int(g.verdict.isin(CARRIER_VERDICTS).sum()), "scored": int(len(g))} for st, g in per_type.groupby("stage")}
                       if len(per_type) and stages is not None and per_type.stage.notna().any() else {})
    # ---- what to decompose
    if decompose_at is None:
        targets = []
    elif decompose_at == "first_lost":
        if lost_depth is not None:
            targets = list(lost_df[lost_df.depth == lost_depth].type)[:max_lost]
        elif first_lost_depth is not None:
            targets = [t for t in verdict_of if depth_of[t] == first_lost_depth][:max_lost]
        else:
            targets = []
    elif isinstance(decompose_at, (int, np.integer)):
        targets = [t for t in verdict_of if depth_of[t] == int(decompose_at)][:max_lost]
    else:
        targets = [t for t in (decompose_at if isinstance(decompose_at, (list, tuple)) else [decompose_at]) if t in tg.pos]
    # ---- the lost-stage input table (the optic_measures 5.3 reading as a table)
    signed_fig = _signed_figure(stim, ctrl, n_pairs, inv, nT, q_of_type, kind_of_type, obj, bg, window, min_obj, min_bg)
    lost_rows = []
    for t in targets:
        inp = tg.inputs(t, min_share=input_min_share, top=top_inputs)
        for _, r in inp.iterrows():
            p = r.pre_type
            pre_fig = float(signed_fig[tg.pos[p]]) if p in tg.pos else float("nan")
            lost_rows.append({"target_type": t, "target_depth": depth_of.get(t), "target_verdict": verdict_of.get(t), "target_z": z_of.get(t),
                              "pre_type": p, "pre_depth": depth_of.get(p, int(depth[tg.pos[p]]) if depth[tg.pos[p]] >= 0 else None),
                              "share": float(r.share), "sign": int(r.sign), "mv_per_volley": float(r.mv_per_volley),
                              "raw_synapses_per_post": float(r.raw_synapses_per_post), "pre_verdict": verdict_of.get(p, "unscored"),
                              "pre_z": z_of.get(p, float("nan")), "pre_diff": diff_of.get(p, float("nan")), "pre_signed_figure": pre_fig,
                              "term": float(r.sign * r.share * pre_fig) if np.isfinite(pre_fig) else float("nan")})
    lost_inputs = pd.DataFrame(lost_rows)
    cancellation = []
    if len(lost_inputs):
        for t, g in lost_inputs.groupby("target_type", sort=False):
            gc = g[g.pre_verdict.isin(CARRIER_VERDICTS)]
            pos_t = float(gc.term[gc.term > 0].sum()) if len(gc) else 0.0; neg_t = float(gc.term[gc.term < 0].sum()) if len(gc) else 0.0
            own = float(signed_fig[tg.pos[t]])
            cancellation.append({"target_type": t, "carrier_inputs": list(gc.pre_type), "carrier_share": float(gc.share.sum()),
                                 "carriers_raising": list(gc.pre_type[gc.term > 0]), "carriers_lowering": list(gc.pre_type[gc.term < 0]),
                                 "sum_positive_terms": pos_t, "sum_negative_terms": neg_t, "linear_estimate": pos_t + neg_t,
                                 "cancellation_fraction": float(1 - abs(pos_t + neg_t) / max(abs(pos_t) + abs(neg_t), 1e-12)) if (pos_t or neg_t) else float("nan"),
                                 "own_signed_figure": own, "excitatory_share_of_carriers": float(gc.share[gc.sign > 0].sum())})

    # ---- provenance and the Result
    prov = _provenance(c, stim, ctrl, nul, params, optic_params, fb)
    res = Result.new("trace", prov)
    res.add_population(population(c, source, "source"), unit_kind=str(kinds[src_idx[0]]), keep_ids=len(src_idx) <= 10_000)
    for t in targets:
        res.add_population(population(c, t, f"lost:{t}"), unit_kind=str(kind_of_type[tg.pos[t]]), keep_ids=True)
    res.replicates = {"n": int(n_pairs), "unit": "runs", "min_replicates": MIN_REPLICATES,
                      "runs": [_run_record(r, k) for k, r in enumerate(stim[:n_pairs])],
                      "control": [_run_record(r, k) for k, r in enumerate(ctrl)],
                      "null": {"source": null_src, "n": int(len(null_pairs)), "runs": [_run_record(r, k) for k, r in enumerate(nul)]}}
    res.add_table("per_type", per_type)
    res.add_table("lost_candidates", lost_df)
    res.add_table("lost_inputs", lost_inputs)
    res.add_table("lost_cancellation", cancellation)
    res.add_table("depth_edges", _depth_edges(tg, depth, per_type, min_share))
    if per_body != "none":
        body_types = set(targets) | set(TWO_ROW_TYPES) | set(carriers[:10]) if per_body == "lost" else (set(per_type.type) if len(per_type) else set())
        res.add_table("readout_per_body", _readout_per_body(c, stim[:n_pairs], ctrl[:n_pairs], idx0, inv, tg, body_types, q_of_type, kinds, window, meta0))
    res.summary = {"stat": stat, "source": common.spec_repr(source), "n_source_cells": int(len(src_idx)), "min_share": min_share,
                   "min_obj": int(min_obj), "min_bg": int(min_bg),
                   "depth_max": depth_max, "types_scored": int(len(per_type)), "carriers": carriers,
                   "carriers_by_depth": {int(d): list(per_type[(per_type.depth == d) & per_type.verdict.isin(CARRIER_VERDICTS)].type) for d in sorted(per_type.depth.unique())} if len(per_type) else {},
                   "counts_by_depth": counts_by_depth, "counts_by_stage": counts_by_stage, "lost_min_carrier_share": lost_min_carrier_share,
                   "first_lost_depth": first_lost_depth, "lost_depth": lost_depth, "lost_types": list(lost_df.type[:max_lost]) if len(lost_df) else [],
                   "lost_top_by_share": lost_top_by_share, "n_lost_candidates": int(len(lost_df)),
                   "decomposed": targets, "null_source": null_src, "n_stim_runs": int(n_pairs), "n_null_draws": int(len(null_pairs)),
                   "p_floor": p_floor(n_pairs, len(null_pairs)),
                   "p_floor_note": ("the exact U p of %d v %d runs floors at %.3f > 0.05: no type can reach verdict 'result' at this n; "
                                    "add runs (4 v 4 floors at 0.029, 5 v 5 at 0.0079)" % (n_pairs, len(null_pairs), p_floor(n_pairs, len(null_pairs))))
                   if p_floor(n_pairs, len(null_pairs)) > 0.05 else None,
                   "cancellation": cancellation, "sign0_counts_available": bool(sign0_available),
                   "effective_weights_md5": ew.md5, "stage_source": ("none" if stages is None else "Nern 2025 table" if stages else "type-name family rule")}
    res.files = {"recordings": {"stimulus": [getattr(r, "path", None) for r in stim], "control": [getattr(r, "path", None) for r in ctrl],
                                "null": [getattr(r, "path", None) for r in nul]}, "generator": "scripts/interp_trace.py analyse / flyverse.interp.trace.trace"}
    # ---- the decompose tool at the lost stage, when it exists
    if optic_params is None and prov_rec:
        optic_params = optic_params_from_provenance(prov_rec)
    res.summary["decompose"] = _call_decompose(res, c, targets, params, optic_params, stim[:n_pairs], ctrl[:n_pairs], window)
    _validate(res, meta0.get("protocol"))
    return res


def _dprime(a: Recording, b: Recording, tg: TypeGraph, positive=None, negative=None) -> np.ndarray:
    """screen.rank's d' per type between the pooled series of two runs (a = the positive condition)."""
    sa = getattr(a, "series", None); sb = getattr(b, "series", None)
    if sa is None or sb is None:
        raise ValueError("stat 'dprime' needs the per-frame pooled series (<run>_series.npz) next to every recording")
    keys = sa.types.astype(str)
    rec = TypeRecorder(keys=keys, idx=np.arange(len(keys)), inv=np.arange(len(keys)), n_cells=np.asarray(sa.meta.get("n_cells", np.ones(len(keys)))))
    runs = {"stim": sa.quantities["pooled"], "ctrl": sb.quantities["pooled"]}
    hz = float(sa.meta.get("hz") or 100.0)                       # the series' sample rate (series_every frames of 10 ms)
    df = rank(runs, positive or ["stim"], negative or ["ctrl"], rec, min_cells=1, min_hz=-np.inf, hz=hz, skip_s=0.0)   # the series is already the scored window
    out = np.full(len(tg.keys), np.nan)
    pos = {k: i for i, k in enumerate(tg.keys)}
    for k, v in zip(df.key, df.d_prime):
        if k in pos:
            out[pos[k]] = v
    return out


def _signed_figure(stim, ctrl, n_pairs, inv, nT, q_of_type, kind_of_type, obj, bg, window, min_obj=5, min_bg=20) -> np.ndarray:
    """Per type the signed (stimulus - control) figure, mean over runs: the retinotopic figure when columns exist,
    else the population-mean signed difference of the signed quantity (optic_dr for graded, the spiking quantity)."""
    out = np.full(nT, np.nan)
    for kind in ("graded", "photoreceptor", "spiking"):
        sel = kind_of_type == kind
        if not sel.any():
            continue
        q = "optic_dr" if kind != "spiking" else str(q_of_type[sel][0]).replace("_abs", "")
        m_q = sel[inv]
        acc = []
        for r in range(n_pairs):
            xa = cell_means(stim[r], q, window); xb = cell_means(ctrl[r], q, window)
            d = np.where(m_q, xa - xb, np.nan)
            if obj is not None:
                v, _ = _group_stat(d, inv, nT, "figure_z", obj, bg, min_obj, min_bg)
            else:
                v, _ = _group_stat(d, inv, nT, "mean")
            acc.append(v)
        v = np.nanmean(np.stack(acc), axis=0) if acc else np.full(nT, np.nan)
        out[sel] = v[sel]
    return out


def _depth_edges(tg: TypeGraph, depth, per_type, min_share) -> pd.DataFrame:
    """For every scored type its strongest input from the previous depth (the edge that placed it)."""
    rows = []
    for t in (per_type.type if len(per_type) else []):
        i = tg.pos[t]
        if depth[i] <= 0:
            continue
        inp = tg.inputs(t, min_share=min_share, top=50)
        inp = inp[[depth[tg.pos[p]] == depth[i] - 1 for p in inp.pre_type]]
        if len(inp):
            r = inp.iloc[0]
            rows.append({"type": t, "depth": int(depth[i]), "from_type": r.pre_type, "share": float(r.share), "sign": int(r.sign),
                         "mv_per_volley": float(r.mv_per_volley), "raw_synapses_per_post": float(r.raw_synapses_per_post)})
    return pd.DataFrame(rows)


def _run_record(r: Recording, k: int) -> dict:
    m = r.meta
    ex = (m.get("provenance") or {}).get("execution", {})
    return {"run_index": k, "seed": m.get("seed"), "arm": m.get("arm"), "file": getattr(r, "path", None), "device": ex.get("device"),
            "window_s": m.get("window_s"), "frames": m.get("frames")}


def _readout_per_body(c, stim, ctrl, idx0, inv, tg, body_types, q_of_type, kinds, window, meta0) -> pd.DataFrame:
    """Neurome readout_per_body rows for the cells of `body_types`: the scored quantity per body; LC11 / LC10a get an
    upstream_drive_mV and an output_Hz row each."""
    n = c.neurons
    bid = n.bodyId.to_numpy(); ty = n.type.fillna("").to_numpy()
    win = meta0.get("window_s") or [None, None]
    ctrl_ids = [str(getattr(r, "path", None) or r.meta.get("run_id") or k) for k, r in enumerate(ctrl)]
    rows = []
    sel_types = [t for t in body_types if t in tg.pos]
    cell_mask = np.isin(inv, [tg.pos[t] for t in sel_types])
    cells = np.flatnonzero(cell_mask)
    if not len(cells):
        return pd.DataFrame(columns=common.EXPORT_TABLES["readout_per_body"])
    quantities = sorted(set(q_of_type[inv[cells]]) | ({"drive_mv", "rate_hz"} if set(sel_types) & set(TWO_ROW_TYPES) else set()))
    for q in quantities:
        xa = np.stack([cell_means(r, q, window) for r in stim]); xb = np.stack([cell_means(r, q, window) for r in ctrl])
        for j in cells:
            t = ty[idx0[j]]
            want = (q_of_type[inv[j]] == q) or (t in TWO_ROW_TYPES and q in ("drive_mv", "rate_hz"))
            if not want or not np.isfinite(xa[:, j]).any():
                continue
            d = xa[:, j] - xb[:, j]
            rows.append({"bodyId": str(int(bid[idx0[j]])), "model_index": int(idx0[j]), "type": t, "unit_kind": str(kinds[idx0[j]]),
                         "quantity": NEUROME_QUANTITY.get(q, q), "window_start_s": win[0], "window_end_s": win[1],
                         "stimulus_value": float(np.nanmean(xa[:, j])), "control_value": float(np.nanmean(xb[:, j])),
                         "stimulus_minus_control": float(np.nanmean(d)), "unit": UNITS_OF.get(q, ""), "n_trials": int(np.isfinite(d).sum()),
                         "trial_sd": float(np.nanstd(d, ddof=1)) if np.isfinite(d).sum() > 1 else float("nan"), "control_ids": ctrl_ids})
    return pd.DataFrame(rows, columns=common.EXPORT_TABLES["readout_per_body"])


def params_from_provenance(prov: dict):
    """LIFParams rebuilt from a provenance block's model.lif (resolved fields; unknown keys dropped); None on failure."""
    try:
        import dataclasses
        from .. import brain
        lif = dict((prov.get("model") or {}).get("lif") or {})
        names = {f.name for f in dataclasses.fields(brain.LIFParams)}
        kw = {k: v for k, v in lif.items() if k in names}
        for k in ("path_gain", "type_path_gain"):
            if isinstance(kw.get(k), list):
                kw[k] = [tuple(x) if isinstance(x, list) else x for x in kw[k]]
        return brain.LIFParams(**kw)
    except Exception:  # noqa: BLE001 -- an old or foreign provenance: the defaults
        return None


def optic_params_from_provenance(prov: dict):
    """OpticParams rebuilt from a provenance block's model.optic (resolved fields; pair_gain rows back to tuples); None
    on failure -- the optic decomposition of a graded lost stage needs the OpticParams the recording was made with."""
    try:
        import dataclasses
        from .. import optic
        op = dict((prov.get("model") or {}).get("optic") or {})
        names = {f.name for f in dataclasses.fields(optic.OpticParams)}
        kw = {k: v for k, v in op.items() if k in names}
        if isinstance(kw.get("pair_gain"), list):
            kw["pair_gain"] = [tuple(x) if isinstance(x, list) else x for x in kw["pair_gain"]]
        return optic.OpticParams(**kw)
    except Exception:  # noqa: BLE001
        return None


def _provenance(c, stim, ctrl, nul, params, optic_params, fb) -> dict:
    """The recording's provenance (written on the GPU with the realised device) with the analysis' git state and the
    arms' files added; built fresh from `fb` / the parameters when the recordings carry none."""
    rec = stim[0].meta.get("provenance")
    if rec:
        prov = dict(rec)
    else:
        prov = common.provenance(c, params, optic_params, fb=fb, seeds=[r.meta.get("seed") for r in stim],
                                 stimulus=stim[0].meta.get("stimulus") or {"protocol": stim[0].meta.get("protocol"), "params": {}, "control": ctrl[0].meta.get("arm")})
    prov["analysis"] = {"flyverse_commit": common.git_state(), "arms": {"stimulus": [r.meta.get("arm") for r in stim], "control": [r.meta.get("arm") for r in ctrl],
                                                                       "null": [r.meta.get("arm") for r in nul]},
                        "seeds": {"stimulus": [r.meta.get("seed") for r in stim], "control": [r.meta.get("seed") for r in ctrl], "null": [r.meta.get("seed") for r in nul]},
                        "devices": sorted({str(((r.meta.get("provenance") or {}).get("execution") or {}).get("device")) for r in stim + ctrl + nul})}
    return prov


def _call_decompose(res: Result, c, targets, params, optic_params, stim_runs, ctrl_runs, window) -> dict:
    """Compose with flyverse.interp.decompose at the lost stage when that module exists (static, and dynamic over the
    paired stimulus runs against the control runs -- the contract's stimulus[0] and the other runs with it, so the
    dynamic table carries replicate scatter); never fail the trace."""
    if not targets:
        return {"status": "no lost stage to decompose"}
    try:
        from .. import interp as pkg
        fn = getattr(pkg, "decompose")
        if not callable(fn):                       # flyverse.interp.decompose is the submodule once imported: take its function
            fn = getattr(fn, "decompose")
    except Exception as e:  # noqa: BLE001
        return {"status": f"unavailable: {e!r}"}
    out = {"targets": list(targets)}
    spec = list(targets)
    try:
        static = fn(c, spec, params=params, optic_params=optic_params, tiers=True, top=40)
        for name, tab in static.tables.items():
            res.tables[f"lost_static_{name}"] = tab
        out["static"] = "ok"; out["static_summary"] = to_jsonable(static.summary)
    except NotImplementedError as e:
        return {"status": f"stub: {e}"}
    except Exception as e:  # noqa: BLE001
        out["static"] = f"failed: {e!r}"
    try:
        dyn = fn(c, spec, recording={"stim": list(stim_runs)}, null_recording={"ctrl": list(ctrl_runs)}, params=params, optic_params=optic_params, window=window, top=40)
        for name, tab in dyn.tables.items():
            res.tables[f"lost_dynamic_{name}"] = tab
        out["dynamic"] = "ok"; out["dynamic_summary"] = to_jsonable(dyn.summary)
    except Exception as e:  # noqa: BLE001
        out["dynamic"] = f"failed: {e!r}"
    out["status"] = "ok"
    return out


# ---------------------------------------------------------------------------------------------- validation
def _validate(res: Result, protocol) -> None:
    """Fill validation.measured / status against VALIDATION['trace'] from the per_type table: the object stage (Mi4 /
    Mi1 / Tm3 result, T2 / T3 / Tm5Y / TmY21 / LC11 / LC10a not) or the LH odour gate (LHPD4d1 result at depth <= 2
    from the ORNs) -- whichever the scored types allow; 'not run' otherwise."""
    pt = res.table("per_type")
    if not len(pt):
        res.validation["status"] = "not run"; return
    row = {t: r for t, r in zip(pt.type, pt.to_dict("records"))}
    ref = res.validation["reference"]
    measured = {}
    status_obj = status_lh = None
    carriers_ref = ["Mi4", "Mi1", "Tm3"]; null_ref = list(ref["at_null"])
    if any(t in row for t in carriers_ref + null_ref):
        for t in carriers_ref + null_ref:
            if t in row:
                measured[t] = {"z": row[t]["z"], "verdict": row[t]["verdict"], "depth": row[t]["depth"], "stim_mean": row[t]["stim_mean"], "null_mean": row[t]["null_mean"]}
        measured["first_lost_depth"] = res.summary.get("first_lost_depth"); measured["lost_types"] = res.summary.get("lost_types")
        measured["decomposed"] = res.summary.get("decomposed")
        ok_c = [t for t in carriers_ref if t in row and carries(row[t]["verdict"])]
        ok_n = [t for t in null_ref if t in row and not carries(row[t]["verdict"])]
        present_c = [t for t in carriers_ref if t in row]; present_n = [t for t in null_ref if t in row]
        under = any(row[t]["verdict"] == "underpowered" for t in present_c + present_n)
        if not (present_c and present_n):
            status_obj = "not run"                             # the object stage needs both the carriers and the at-null types scored
        else:
            status_obj = "reproduced" if (ok_c == present_c and ok_n == present_n) else "underpowered" if under else "not reproduced"
        measured["object_stage"] = {"carriers_result": ok_c, "carriers_expected": present_c, "at_null": ok_n, "at_null_expected": present_n, "status": status_obj}
    lh = [t for t in ("LHPD4d1", "LHAV4a1_a", "LHAV4a1_b", "LHCENT12_a", "LHPD2a1") if t in row]
    if lh:
        for t in lh:
            measured[t] = {"stim_level": row[t]["stim_level"], "ctrl_level": row[t]["ctrl_level"], "diff": row[t]["diff"], "z": row[t]["z"],
                           "verdict": row[t]["verdict"], "depth": row[t]["depth"]}
        measured["lh_first_lost_depth"] = res.summary.get("first_lost_depth"); measured["lh_lost_types"] = res.summary.get("lost_types")
        if "LHPD4d1" in row:
            r = row["LHPD4d1"]
            status_lh = "reproduced" if (carries(r["verdict"]) and r["depth"] <= 2 and r["diff"] > 0) else ("underpowered" if r["verdict"] == "underpowered" else "not reproduced")
        else:
            status_lh = "not run"
        measured["lh_gate"] = {"status": status_lh}
    # the block that matches the protocol decides; else whichever ran
    if protocol in ("object", "apple", "ball"):
        status = status_obj or "not run"
    elif protocol in ("odour", "smell", "apple8_into_wind"):
        status = status_lh or "not run"
    else:
        status = next((x for x in (status_obj, status_lh) if x and x != "not run"), "not run")
    res.validation["measured"] = to_jsonable(measured); res.validation["status"] = status
    res.validation["protocol"] = protocol


# ---------------------------------------------------------------------------------------------- recording (GPU half)
class ArmAccumulator:
    """Per-cell time-means of one arm over the scored window, and the per-frame pooled series per type -- the
    recording the trace analyses. torch-free: reads numpy copies of the FlyBrain tensors each scored frame
    (`brain.rate_np()`, `brain.drive`, `brain.spike_counts`, `optic.rates()` / `optic.r0`), so the same code runs
    against a fake in the tests. Frames before `skip` are not scored; `spike_counts` at frame skip - 1 is the origin
    of the window's rate.

        acc = ArmAccumulator(fb); for k in range(n): place(k); sim.step(); acc.add(fb, k, skip)
        cells, series = acc.finish(meta)                      # two Recordings: per-cell means, pooled series
    """

    def __init__(self, fb, keep_series: bool = True, series_every: int = 1):
        self.c = fb.c
        self.series_every = max(int(series_every), 1)
        self.N = fb.c.n
        b = fb.brain
        o = getattr(fb, "optic", None)
        self.rate_idx = np.asarray(o.rate_idx) if o is not None else np.zeros(0, np.int64)
        self.n = 0
        self.drv_s = np.zeros(self.N); self.drv_a = np.zeros(self.N)
        self.dr_s = np.zeros(len(self.rate_idx)); self.dr_a = np.zeros(len(self.rate_idx))
        self.counts0 = None; self.counts_last = None
        self.keep_series = keep_series
        ty = fb.c.neurons.type.fillna("").to_numpy()
        self.tr = TypeRecorder.build(fb.c)
        self.series = []
        self.t_ms = []
        self.types = ty
        self._b = b; self._o = o

    @staticmethod
    def _np(x, row=0):
        a = common._np(x)
        return np.asarray(a[row] if a.ndim > 1 else a, dtype=np.float64).reshape(-1)

    def add(self, fb, k: int, skip: int) -> None:
        b = fb.brain; o = getattr(fb, "optic", None)
        if k == skip - 1 or (skip == 0 and k == 0 and self.counts0 is None):
            self.counts0 = self._np(b.spike_counts).copy()
        if k < skip:
            return
        if self.counts0 is None:
            self.counts0 = self._np(b.spike_counts).copy()
        self.n += 1
        d = self._np(b.drive); self.drv_s += d; self.drv_a += np.abs(d)
        rate = np.asarray(b.rate_np(), dtype=np.float64).reshape(-1) if hasattr(b, "rate_np") else self._np(b.rate)
        if o is not None:
            dr = self._np(o.rates()) - self._np(o.r0)
            self.dr_s += dr; self.dr_a += np.abs(dr)
        self.counts_last = self._np(b.spike_counts)
        if self.keep_series and (self.n - 1) % self.series_every == 0:
            self.t_ms.append(float(getattr(b, "t", k * common.FRAME_MS)))
            x = rate.copy()
            if o is not None:
                x[self.rate_idx] = dr
            self.series.append(self.tr.snapshot(x).astype(np.float32))

    def finish(self, meta: dict | None = None, frame_ms: float = common.FRAME_MS) -> tuple[Recording, Recording | None]:
        n = max(self.n, 1)
        window_s = self.n * frame_ms / 1000.0
        rate_hz = (self.counts_last - self.counts0) / max(window_s, 1e-9) if self.counts_last is not None else np.full(self.N, np.nan)
        rate_hz[self.rate_idx] = np.nan                                                          # rate units never spike
        dr = np.full(self.N, np.nan); dra = np.full(self.N, np.nan)
        dr[self.rate_idx] = self.dr_s / n; dra[self.rate_idx] = self.dr_a / n
        q = {"rate_hz": rate_hz[None].astype(np.float32), "drive_mv": (self.drv_s / n)[None].astype(np.float32),
             "drive_mv_abs": (self.drv_a / n)[None].astype(np.float32), "optic_dr": dr[None].astype(np.float32), "optic_dr_abs": dra[None].astype(np.float32)}
        m = dict(meta or {})
        m.update({"accumulated": True, "frames": int(self.n), "window_frames_ms": frame_ms, "quantities": list(q),
                  "unit_kind_note": "rate_hz from spike_counts over the window (NaN on rate units); optic_dr = rates() - r0 time-mean; "
                                    "drive_mv = brain.drive time-mean (the optic / sensory injected current)"})
        n_ = self.c.neurons
        cells = Recording(np.asarray(self.t_ms[-1:] or [0.0]), np.arange(self.N), n_.bodyId.to_numpy(), self.types.astype(str), q, {}, m)
        m["series_every"] = self.series_every
        series = None
        if self.keep_series and self.series:
            keys = self.tr.keys
            series = Recording(np.asarray(self.t_ms), np.arange(len(keys)), -np.ones(len(keys), np.int64), keys.astype(str),
                               {"pooled": np.stack(self.series)}, {},
                               {"pooled": True, "n_cells": self.tr.n_cells.tolist(), "arm": m.get("arm"), "seed": m.get("seed"), "protocol": m.get("protocol"),
                                "series_every": self.series_every, "hz": 1000.0 / (frame_ms * self.series_every),
                                "unit": "Hz (spiking types: brain.rate) / rate-unit deviation from r0 (graded types)"})
        return cells, series


def save_recording(rec: Recording, path, compress: bool = True) -> Path:
    """Recording.save with np.savez_compressed: the same npz / json schema (Recording.load reads both), a quarter of the
    size for the full-N per-cell means (167k type strings, NaN on every rate unit) and the pooled series."""
    import json as _json
    path = Path(str(path))
    if not compress:
        return rec.save(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"t_ms": rec.t_ms, "idx": rec.idx, "body_ids": rec.body_ids, "types": rec.types.astype(str)}
    arrays.update({f"q__{k}": v for k, v in rec.quantities.items()})
    arrays.update({f"m__{k}": v for k, v in rec.motor.items()})
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        _json.dump({"schema": common.RECORDING_SCHEMA, "meta": to_jsonable(rec.meta), "quantities": list(rec.quantities),
                    "motor": list(rec.motor), "n_frames": rec.n_frames, "n_cells": int(len(rec.idx)),
                    "units": {k: common.QUANTITIES.get(k, "") for k in rec.quantities}}, f, indent=1)
    return path.with_suffix(".npz")


def figure_columns(col_dir, ang_deg: np.ndarray, radius_deg: float, bg_margin_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """probe_figure_stages' column sets: object columns within `radius_deg` of the object's direction, background
    columns farther than radius + margin (`ang_deg` = per-column angle to the object)."""
    return np.flatnonzero(ang_deg <= radius_deg), np.flatnonzero(ang_deg >= radius_deg + bg_margin_deg)


def column_of_cells(c: cn.Connectome, retina, rate_idx) -> tuple[np.ndarray, int]:
    """probe_figure_stages.column_of_cells, torch-free: column index per cell (-1 = none) -- rate cells from their hex
    annotation, three propagation passes over the strongest rate input partner, spiking cells from their strongest
    rate input partner."""
    key = {(str(sd), int(h[0]), int(h[1])): i for i, (sd, h) in enumerate(zip(retina.col_side, retina.col_hex))}
    ridx = np.asarray(rate_idx)
    nr = c.neurons.iloc[ridx]
    h1 = nr["hex1"].to_numpy(); h2 = nr["hex2"].to_numpy(); sd_ = nr["hex_side"].fillna("").to_numpy()
    col_r = np.array([key.get((str(s_), int(a_), int(b_)), -1) if (a_ == a_ and b_ == b_) else -1 for s_, a_, b_ in zip(sd_, h1, h2)], dtype=np.int64)
    n_ann = int((col_r >= 0).sum())
    Wabs = sp.csr_matrix(abs(c.W)); Wr = Wabs[ridx][:, ridx].tocsr()
    for _ in range(3):
        for i in np.flatnonzero(col_r < 0):
            s0, s1 = Wr.indptr[i], Wr.indptr[i + 1]; pre = Wr.indices[s0:s1]; w = Wr.data[s0:s1]
            ok = col_r[pre] >= 0
            if ok.any():
                col_r[i] = col_r[pre[ok][np.argmax(w[ok])]]
    col = -np.ones(c.n, np.int64); col[ridx] = col_r
    Wsr = Wabs[:, ridx].tocsr()
    for i in np.flatnonzero(col < 0):
        s0, s1 = Wsr.indptr[i], Wsr.indptr[i + 1]
        if s1 == s0:
            continue
        pre = Wsr.indices[s0:s1]; w = Wsr.data[s0:s1]; ok = col_r[pre] >= 0
        if ok.any():
            col[i] = col_r[pre[ok][np.argmax(w[ok])]]
    return col, n_ann


def retina_record(retina, c: cn.Connectome) -> dict:
    """The provenance 'retina' block: column count and the column -> photoreceptor-body map."""
    if retina is None:
        return {"file": None, "n_columns": None, "column_to_bodies": None}
    bid = c.neurons.bodyId.to_numpy()
    col_to = {}
    for pr, col in zip(np.asarray(retina.pr_index), np.asarray(retina.pr_column)):
        col_to.setdefault(int(col), []).append(str(int(bid[pr])))
    return {"file": None, "n_columns": int(retina.n_columns), "column_to_bodies": col_to,
            "col_side": [str(s) for s in retina.col_side], "col_hex": np.asarray(retina.col_hex).tolist()}


def print_trace(res: Result, max_rows: int = 80) -> str:
    """The printed table: per type in depth order, then the lost-stage input table."""
    pt = res.table("per_type")
    cols = [k for k in ["type", "depth", "stage", "unit_kind", "n_cells", "quantity", "stim_mean", "null_mean", "null_sd", "z", "welch", "U", "p", "verdict", "carry_runs"] if k in pt.columns]
    s = res.summary
    print(f"trace: source {s.get('source')} ({s.get('n_source_cells')} cells), stat {s.get('stat')}, {s.get('n_stim_runs')} stimulus runs vs "
          f"{s.get('n_null_draws')} null draws ({s.get('null_source')}); {s.get('types_scored')} types scored; carriers by depth {s.get('carriers_by_depth')}")
    print(f"first lost (depth rule) {s.get('first_lost_depth')}; lost types (input rule, depth {s.get('lost_depth')}): {s.get('lost_types')}; decomposed: {s.get('decomposed')}")
    out = common.print_table(pt[cols], max_rows=max_rows) if len(pt) else ""
    li = res.table("lost_inputs")
    if len(li):
        print("\nlost-stage inputs (share of the target's |input|, sign, the input's own verdict / signed figure, term = sign x share x figure):")
        out += "\n" + common.print_table(li[["target_type", "pre_type", "pre_depth", "share", "sign", "mv_per_volley", "raw_synapses_per_post", "pre_verdict", "pre_z", "pre_signed_figure", "term"]], max_rows=max_rows)
        for cz in res.summary.get("cancellation", []):
            print(f"  {cz['target_type']}: carriers raising {cz['carriers_raising']} (+{cz['sum_positive_terms']:.4g}) vs lowering {cz['carriers_lowering']} ({cz['sum_negative_terms']:.4g}); "
                  f"linear estimate {cz['linear_estimate']:.4g} vs own figure {cz['own_signed_figure']:.4g}; cancellation {cz['cancellation_fraction']:.2f}; excitatory carrier share {cz['excitatory_share_of_carriers']:.3f}")
    print(f"validation: {res.validation.get('status')}  {res.validation.get('measured')}")
    return out
