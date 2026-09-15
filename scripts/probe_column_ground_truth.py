"""Ground truth for `flyverse.interp.trace.column_of_cells` (TODO.md F.4; docs/audits/column_ground_truth.md).

The optic-lobe column annotations of the two releases are the truth: FAFB v783 `column_assignment` (45,528 cells of 31
types, `hex_source == "annotation"` after the affine map of docs/audits/connectome_backends.md) and MaleCNS
`assignedOlHex1/2` (23,720 cells of 15 types). `column_of_cells` is the inference the object rounds used to place the
anatomical LC windows (object_baseline_r2.md: 405 of 418 LC windows; object_localizer_r3.md 1.2): a rate cell without an
annotation takes the column of its strongest |W| rate input partner, three passes; a spiking cell the column of its
strongest rate input partner. Three measurements, CPU only:

1. leave-one-type-out: withhold one annotated type's annotation, run the SAME propagation, compare with the withheld
   truth (fraction assigned, exact, median / p90 error in columns and in degrees through the retina's col_az_el),
   per type and per side, on both datasets; on FAFB also with the annotation reduced to MaleCNS's 15 annotated types
   (`condition = loo_malecns_set`: the propagation depth the MaleCNS LC windows relied on, measured where the truth
   exists);
2. the LC case: LC11 / LC10a / LC4 / LPLC2 (no annotation in either release) -- the inferred column's spread, the
   |W|-weighted input centroid and its distance to the inferred column, cross-dataset as distributions; on FAFB the
   per-cell distance between the columns inferred under the full and the MaleCNS-like annotation sets;
3. the chance level: the inferred columns permuted within type and side (`n_shuffles` draws), every statistic again.

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python scripts/probe_column_ground_truth.py --out out/colgt

Writes out/colgt/column_ground_truth.json (a Result, tool `trace`, tables loo_per_type / lc_summary / lc_cross /
fafb_annotation_depth / lc_strongest_partner / loo_failure_partner) and the per-cell CSVs it summarises (loo_per_cell_<dataset>.csv, lc_per_cell.csv).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyverse import connectome as cn, retina as rt          # noqa: E402
from flyverse.interp import common                           # noqa: E402
from flyverse.interp import trace as tr                      # noqa: E402

LC_TYPES = ("LC11", "LC10a", "LC4", "LPLC2")
LC_INPUT_TYPES = ("T2", "T3", "Tm5Y", "TmY21")               # the object rounds' LC-input stage (trace VALIDATION at_null)
INTEROMMATIDIAL_DEG = rt.EyeGeometry().interommatidial_deg   # 4.6
DEG_TOL = 1e-6                                               # the modal one-step error IS 4.6 deg and the modal
                                                             # centroid-to-column distance IS r50, so `<=` on a
                                                             # floating-point arccos splits the mode arbitrarily
HEX_COLS = ("hex1", "hex2", "hex_side", "hex_source")


# ------------------------------------------------------------------------------------------------------ the machinery
def rate_index(c: cn.Connectome) -> np.ndarray:
    """OpticLobe.rate_idx without an OpticLobe: ol_intrinsic cells that are not photoreceptors."""
    n = c.neurons
    return np.flatnonzero((n.superclass == "ol_intrinsic").to_numpy() & ~n.type.isin(cn.PHOTORECEPTOR_TYPES).to_numpy())


def withhold(c: cn.Connectome, types) -> cn.Connectome:
    """The same graph with the column annotation of `types` removed (hex1 / hex2 / hex_side NaN, hex_source '')."""
    n = c.neurons.copy()
    m = n.type.isin(list(types)).to_numpy()
    n.loc[m, "hex1"] = np.nan; n.loc[m, "hex2"] = np.nan; n.loc[m, "hex_side"] = None; n.loc[m, "hex_source"] = ""
    return cn.Connectome(neurons=n, W=c.W, body_to_index=c.body_to_index, dataset=c.dataset, release=c.release)


def infer_columns(c: cn.Connectome, retina, ridx=None) -> np.ndarray:
    """trace.column_of_cells over the optic lobe's rate units: column index per cell, -1 = none."""
    col, _ = tr.column_of_cells(c, retina, rate_index(c) if ridx is None else ridx)
    return col


def hex_xy(h1, h2) -> np.ndarray:
    """retina.py's Cartesian embedding of the hex lattice (column spacing 1): X = h1 - h2 / 2, Y = h2 sqrt(3) / 2."""
    h1 = np.asarray(h1, float); h2 = np.asarray(h2, float)
    return np.c_[h1 - 0.5 * h2, h2 * np.sqrt(3) / 2]


def angular_distance_deg(az1, el1, az2, el2) -> np.ndarray:
    a1, e1, a2, e2 = map(np.deg2rad, (az1, el1, az2, el2))
    cosd = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


def retina_key(retina) -> dict:
    return {(str(s), int(h[0]), int(h[1])): i for i, (s, h) in enumerate(zip(retina.col_side, retina.col_hex))}


def truth_table(c: cn.Connectome, retina, idx) -> pd.DataFrame:
    """Per cell of `idx`: the annotated (side, hex1, hex2) and the retina column it names (-1 when the retina has no such
    column -- MaleCNS's retina holds only photoreceptor columns)."""
    n = c.neurons.iloc[np.asarray(idx)]
    key = retina_key(retina)
    h1 = n.hex1.to_numpy(float); h2 = n.hex2.to_numpy(float); sd = n.hex_side.fillna("").astype(str).to_numpy()
    tcol = np.array([key.get((s, int(a), int(b)), -1) if np.isfinite(a) and np.isfinite(b) else -1 for s, a, b in zip(sd, h1, h2)], dtype=np.int64)
    return pd.DataFrame({"model_index": np.asarray(idx), "bodyId": n.bodyId.to_numpy(), "type": n.type.to_numpy(),
                         "truth_side": sd, "truth_hex1": h1, "truth_hex2": h2, "truth_col": tcol})


def score_cells(truth: pd.DataFrame, inferred_col: np.ndarray, retina) -> pd.DataFrame:
    """Per cell: the inferred column, whether it exists / is on the truth side / is exact, the lattice distance in
    columns (same side only) and the great-circle distance in degrees (both columns in the retina)."""
    ic = np.asarray(inferred_col)[truth.model_index.to_numpy()]
    out = truth.copy(); out["inferred_col"] = ic
    ok = ic >= 0
    side = np.where(ok, np.asarray(retina.col_side, str)[np.maximum(ic, 0)], "")
    hexes = np.asarray(retina.col_hex, float)[np.maximum(ic, 0)]
    out["inferred_side"] = side; out["inferred_hex1"] = np.where(ok, hexes[:, 0], np.nan); out["inferred_hex2"] = np.where(ok, hexes[:, 1], np.nan)
    out["assigned"] = ok
    out["same_side"] = ok & (side == out.truth_side.to_numpy())
    d = np.linalg.norm(hex_xy(out.truth_hex1, out.truth_hex2) - hex_xy(out.inferred_hex1, out.inferred_hex2), axis=1)
    out["col_err"] = np.where(out.same_side.to_numpy(), d, np.nan)
    out["exact"] = out.same_side.to_numpy() & (np.nan_to_num(d, nan=np.inf) < 1e-9)
    cae = np.asarray(retina.col_az_el, float); tc = out.truth_col.to_numpy()
    both = ok & (tc >= 0)
    deg = angular_distance_deg(cae[np.maximum(ic, 0), 0], cae[np.maximum(ic, 0), 1], cae[np.maximum(tc, 0), 0], cae[np.maximum(tc, 0), 1])
    out["deg_err"] = np.where(both, np.where(ic == tc, 0.0, deg), np.nan)     # an identical column is exactly 0, not arccos(1 - eps)
    return out


def _stats(s: pd.DataFrame) -> dict:
    n = len(s); a = s.assigned.to_numpy(); ss = s.same_side.to_numpy()
    ce = s.col_err.to_numpy(float); de = s.deg_err.to_numpy(float)
    fin_c = np.isfinite(ce); fin_d = np.isfinite(de); in_ret = (s.truth_col.to_numpy() >= 0)
    return {"n_truth": int(n), "n_truth_in_retina": int(in_ret.sum()), "n_assigned": int(a.sum()), "frac_assigned": float(a.mean()) if n else np.nan,
            "frac_exact_in_retina": float(s.exact.to_numpy()[in_ret].mean()) if in_ret.any() else np.nan,
            "frac_wrong_side": float((a & ~ss).sum() / max(a.sum(), 1)),
            "frac_exact": float(s.exact.mean()) if n else np.nan,
            "frac_within_1col": float((fin_c & (ce <= 1.0 + 1e-9)).sum() / n) if n else np.nan,
            "median_col_err": float(np.median(ce[fin_c])) if fin_c.any() else np.nan,
            "p90_col_err": float(np.percentile(ce[fin_c], 90)) if fin_c.any() else np.nan,
            "n_with_deg": int(fin_d.sum()),
            "median_deg_err": float(np.median(de[fin_d])) if fin_d.any() else np.nan,
            "p90_deg_err": float(np.percentile(de[fin_d], 90)) if fin_d.any() else np.nan,
            "frac_within_io_deg": float((fin_d & (de <= INTEROMMATIDIAL_DEG + DEG_TOL)).sum() / n) if n else np.nan}


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (type, side) and per type pooled ('both')."""
    rows = []
    for t, g in scored.groupby("type", sort=True):
        for sd, gg in list(g.groupby("truth_side", sort=True)) + [("both", g)]:
            rows.append({"type": t, "side": sd, **_stats(gg)})
    return pd.DataFrame(rows)


def shuffle_within(inferred_col: np.ndarray, truth: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """The chance control: the inferred columns of the truth cells permuted among the cells of the same type and truth
    side (a random assignment drawn from the inference's own column distribution; the side is trivially known)."""
    out = np.array(inferred_col, copy=True)
    for _, g in truth.groupby(["type", "truth_side"], sort=False):
        idx = g.model_index.to_numpy()
        out[idx] = inferred_col[rng.permutation(idx)]
    return out


def failure_partners(held: cn.Connectome, col: np.ndarray, scored: pd.DataFrame, bad_deg: float = 10.0) -> pd.DataFrame:
    """Why a withheld type fails: per type, the type of the strongest rate input partner (with a column, in the
    withheld graph) over all its cells and over the cells more than `bad_deg` off, and whether those partners carry
    an annotated or a propagated column."""
    n = held.neurons; tn = n.type.fillna("").to_numpy(); ann = n.hex1.notna().to_numpy()
    W = sp.csr_matrix(abs(held.W)); ridx = rate_index(held); is_rate = np.zeros(held.n, bool); is_rate[ridx] = True
    part = np.full(len(scored), -1, np.int64)
    for k, i in enumerate(scored.model_index.to_numpy()):
        s0, s1 = W.indptr[i], W.indptr[i + 1]; pre = W.indices[s0:s1]; w = W.data[s0:s1]; ok = is_rate[pre] & (col[pre] >= 0)
        if ok.any():
            part[k] = pre[ok][np.argmax(w[ok])]
    rows = []
    for t, g in scored.groupby("type", sort=True):
        pk = part[g.index.to_numpy()]; has = pk >= 0
        bad = has & (np.nan_to_num(g.deg_err.to_numpy(float), nan=np.inf) > bad_deg) & g.assigned.to_numpy()
        vc_all = pd.Series(tn[pk[has]]).value_counts(); vc_bad = pd.Series(tn[pk[bad]]).value_counts()
        rows.append({"type": t, "n_cells": int(len(g)), "n_bad": int(bad.sum()), "bad_deg": bad_deg,
                     "partner_types_top5": {str(k): int(v) for k, v in vc_all.head(5).items()},
                     "bad_partner_types_top5": {str(k): int(v) for k, v in vc_bad.head(5).items()},
                     "frac_partner_annotated": float(ann[pk[has]].mean()) if has.any() else np.nan,
                     "frac_bad_partner_annotated": float(ann[pk[bad]].mean()) if bad.any() else np.nan})
    return pd.DataFrame(rows)


def leave_one_out(c: cn.Connectome, retina, groups: dict, rng: np.random.Generator, n_shuffles: int = 5, ridx=None,
                  log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`groups`: {condition_label: [types withheld together]}; every withheld type is scored against its own truth.
    Returns (per_type with chance_* columns from `n_shuffles` within-type-and-side permutations, per_cell, failure_partners)."""
    ridx = rate_index(c) if ridx is None else ridx
    per_type, per_cell, partners = [], [], []
    for label, types in groups.items():
        t0 = time.time()
        held = withhold(c, types)
        col = infer_columns(held, retina, ridx)
        idx = np.flatnonzero(c.neurons.type.isin(list(types)).to_numpy() & c.neurons.hex1.notna().to_numpy())
        truth = truth_table(c, retina, idx)
        scored = score_cells(truth, col, retina); scored.insert(0, "condition", label)
        fp = failure_partners(held, col, scored); fp.insert(0, "condition", label); partners.append(fp)
        summ = summarize(scored); summ.insert(0, "condition", label)
        chance = []
        for k in range(n_shuffles):
            sh = summarize(score_cells(truth, shuffle_within(col, truth, rng), retina)); sh["shuffle"] = k; chance.append(sh)
        ch = pd.concat(chance).groupby(["type", "side"], sort=False).mean(numeric_only=True).drop(columns=["shuffle"]).add_prefix("chance_").reset_index()
        summ = summ.merge(ch, on=["type", "side"], how="left")
        summ["n_withheld_types"] = len(types)
        per_type.append(summ); per_cell.append(scored)
        b = summ[summ.side == "both"]
        log(f"  [{c.dataset}] {label}: withheld {len(types)} type(s), {len(truth)} truth cells, {time.time() - t0:.1f} s; "
            + "; ".join(f"{r.type} exact {r.frac_exact:.2f} med {r.median_deg_err:.1f} deg (chance {r.chance_median_deg_err:.1f})" for r in b.itertuples()))
    return pd.concat(per_type, ignore_index=True), pd.concat(per_cell, ignore_index=True), pd.concat(partners, ignore_index=True)


# ------------------------------------------------------------------------------------------------------- the LC case
def input_centroids(c: cn.Connectome, col: np.ndarray, retina, idx) -> pd.DataFrame:
    """object_round3_localizer.anatomical_prior's input set: per cell the |W|-weighted mean direction of the columns of
    its rate inputs, the 50 % / 80 % weight radii, the distinct input columns, and the distance from that centroid to
    the single inferred column."""
    cae = np.asarray(retina.col_az_el, float); a, e = np.deg2rad(cae[:, 0]), np.deg2rad(cae[:, 1])
    U = np.c_[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
    W = sp.csr_matrix(abs(c.W))
    ridx = rate_index(c); is_rate = np.zeros(c.n, bool); is_rate[ridx] = True
    rows = []
    for i in np.asarray(idx):
        cc = int(col[i])
        row = {"model_index": int(i), "inferred_col": cc, "az_deg": float(cae[cc, 0]) if cc >= 0 else np.nan, "el_deg": float(cae[cc, 1]) if cc >= 0 else np.nan,
               "centroid_az_deg": np.nan, "centroid_el_deg": np.nan, "centroid_to_col_deg": np.nan, "r50_deg": np.nan, "r80_deg": np.nan,
               "n_input_columns": 0, "n_rate_inputs": 0, "input_weight_with_column": 0.0}
        s0, s1 = W.indptr[i], W.indptr[i + 1]; pre = W.indices[s0:s1]; w = W.data[s0:s1]
        keep = is_rate[pre]; pre, w = pre[keep], w[keep]
        pc = col[pre] if len(pre) else np.zeros(0, np.int64); ok = pc >= 0
        row["n_rate_inputs"] = int(len(pre))
        if ok.any() and w.sum() > 0:
            ww, cols = w[ok], pc[ok]
            v = (ww[:, None] * U[cols]).sum(0); v /= max(np.linalg.norm(v), 1e-12)
            az = float(np.degrees(np.arctan2(v[1], v[0]))); el = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
            d = angular_distance_deg(cae[cols, 0], cae[cols, 1], az, el); o = np.argsort(d); cw = np.cumsum(ww[o]) / ww.sum()
            row.update({"centroid_az_deg": az, "centroid_el_deg": el, "n_input_columns": int(len(np.unique(cols))),
                        "r50_deg": float(d[o][min(np.searchsorted(cw, 0.5), len(d) - 1)]), "r80_deg": float(d[o][min(np.searchsorted(cw, 0.8), len(d) - 1)]),
                        "input_weight_with_column": float(ww.sum() / w.sum())})
            if cc >= 0:
                row["centroid_to_col_deg"] = float(angular_distance_deg(cae[cc, 0], cae[cc, 1], az, el))
        rows.append(row)
    return pd.DataFrame(rows)


def lc_case(c: cn.Connectome, retina, col: np.ndarray, types, rng: np.random.Generator, n_shuffles: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per cell and per type (per side and pooled) for `types` under the inferred columns `col`; the chance column is
    the centroid-to-column distance with the inferred columns permuted within type and soma side."""
    n = c.neurons
    idx = np.flatnonzero(n.type.isin(list(types)).to_numpy())
    pc = input_centroids(c, col, retina, idx)
    pc.insert(1, "bodyId", n.bodyId.to_numpy()[idx]); pc.insert(2, "type", n.type.to_numpy()[idx]); pc.insert(3, "side", n.somaSide.fillna("").astype(str).to_numpy()[idx])
    pc.insert(0, "dataset", c.dataset)
    # chance: permute the single column within (type, side) and remeasure its distance to the (unchanged) centroid
    cae = np.asarray(retina.col_az_el, float)
    ch = np.full((n_shuffles, len(pc)), np.nan)
    for k in range(n_shuffles):
        colk = pc.inferred_col.to_numpy().copy()
        for _, g in pc.groupby(["type", "side"], sort=False):
            ii = g.index.to_numpy(); colk[ii] = pc.inferred_col.to_numpy()[rng.permutation(ii)]
        ok = (colk >= 0) & np.isfinite(pc.centroid_az_deg.to_numpy())
        ch[k, ok] = angular_distance_deg(cae[colk[ok], 0], cae[colk[ok], 1], pc.centroid_az_deg.to_numpy()[ok], pc.centroid_el_deg.to_numpy()[ok])
    pc["chance_centroid_to_col_deg"] = np.nanmean(ch, axis=0)
    rows = []
    for t, g in pc.groupby("type", sort=False):
        for sd, gg in list(g.groupby("side", sort=True)) + [("both", g)]:
            cc = gg.inferred_col.to_numpy(); ok = cc >= 0
            vc = pd.Series(cc[ok]).value_counts()
            rows.append({"dataset": c.dataset, "type": t, "side": sd, "n_cells": int(len(gg)), "frac_no_column": float((~ok).mean()),
                         "n_distinct_columns": int(len(vc)), "largest_column_share": float(vc.iloc[0] / max(ok.sum(), 1)) if len(vc) else np.nan,
                         "top4_column_cells": vc.iloc[:4].tolist(),
                         "median_n_input_columns": float(gg.n_input_columns.median()), "median_r50_deg": float(gg.r50_deg.median()),
                         "median_r80_deg": float(gg.r80_deg.median()),
                         "median_centroid_to_col_deg": float(gg.centroid_to_col_deg.median()), "p90_centroid_to_col_deg": float(gg.centroid_to_col_deg.quantile(0.9)),
                         "frac_col_within_io_of_centroid": float((gg.centroid_to_col_deg <= INTEROMMATIDIAL_DEG + DEG_TOL).mean()),
                         "frac_col_within_r50": float((gg.centroid_to_col_deg <= gg.r50_deg + DEG_TOL).mean()),
                         "chance_median_centroid_to_col_deg": float(gg.chance_centroid_to_col_deg.median()),
                         "median_input_weight_with_column": float(gg.input_weight_with_column.median())})
    return pc, pd.DataFrame(rows)


def strongest_partner(c: cn.Connectome, col: np.ndarray, types) -> pd.DataFrame:
    """The mechanism: per type, the type of the rate input partner whose column the cell inherits (the strongest |W|
    rate input with a column), how many distinct partner cells carry the whole type, and whether those partners are
    annotated or themselves propagated."""
    n = c.neurons; tn = n.type.fillna("").to_numpy(); ann = (n.hex_source == "annotation").to_numpy()
    W = sp.csr_matrix(abs(c.W)); ridx = rate_index(c); is_rate = np.zeros(c.n, bool); is_rate[ridx] = True
    rows = []
    for t in types:
        idx = np.flatnonzero(tn == t); partner, share = [], []
        for i in idx:
            s0, s1 = W.indptr[i], W.indptr[i + 1]; pre = W.indices[s0:s1]; w = W.data[s0:s1]; k = is_rate[pre] & (col[pre] >= 0)
            if k.any():
                j = pre[k][np.argmax(w[k])]; partner.append(int(j)); share.append(float(w[k].max() / max(w[k].sum(), 1e-9)))
        pt = pd.Series([tn[j] for j in partner]); vc = pt.value_counts()
        rows.append({"dataset": c.dataset, "type": t, "n_cells": int(len(idx)), "n_with_partner": int(len(partner)),
                     "partner_types_top5": {str(k): int(v) for k, v in vc.head(5).items()}, "n_distinct_partner_cells": int(len(set(partner))),
                     "frac_partner_annotated": float(np.mean([ann[j] for j in partner])) if partner else np.nan,
                     "median_partner_share_of_input": float(np.median(share)) if share else np.nan})
    return pd.DataFrame(rows)


def cross_dataset(lc: pd.DataFrame, datasets=("malecns", "fafb")) -> pd.DataFrame:
    """Homologous types across animals compared as distributions only: quantiles and the two-sample KS statistic of the
    per-cell centroid-to-column distance, r50 and the input-column count (descriptive; two animals, one each)."""
    from scipy.stats import ks_2samp
    rows = []
    for t, g in lc.groupby("type", sort=False):
        a = g[g.dataset == datasets[0]]; b = g[g.dataset == datasets[1]]
        if not len(a) or not len(b):
            continue
        row = {"type": t, f"n_{datasets[0]}": int(len(a)), f"n_{datasets[1]}": int(len(b))}
        for q in ("centroid_to_col_deg", "r50_deg", "n_input_columns", "centroid_az_deg", "centroid_el_deg"):
            x = a[q].to_numpy(float); y = b[q].to_numpy(float); x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
            for name, v in ((datasets[0], x), (datasets[1], y)):
                row[f"{q}_q25_{name}"], row[f"{q}_median_{name}"], row[f"{q}_q75_{name}"] = (float(np.percentile(v, p)) for p in (25, 50, 75)) if len(v) else (np.nan,) * 3
            ks = ks_2samp(x, y) if len(x) and len(y) else None
            row[f"{q}_ks"] = float(ks.statistic) if ks else np.nan; row[f"{q}_ks_p"] = float(ks.pvalue) if ks else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------- main
def annotated_types(c: cn.Connectome, min_cells: int) -> list[str]:
    """Rate-unit types with >= min_cells annotated cells (photoreceptors excluded: the retina assigns them by a different
    rule, connectome._assign_photoreceptor_columns)."""
    n = c.neurons
    m = (n.hex_source == "annotation").to_numpy() & ~n.type.isin(cn.PHOTORECEPTOR_TYPES).to_numpy() & (n.superclass == "ol_intrinsic").to_numpy()
    vc = n[m].type.value_counts()
    return sorted(vc[vc >= min_cells].index.tolist())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "colgt")
    ap.add_argument("--datasets", nargs="+", default=["fafb", "malecns"])
    ap.add_argument("--min-cells", type=int, default=100)
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    graphs = {ds: cn.load(dataset=ds, verbose=False) for ds in args.datasets}
    retinas = {ds: rt.build_retina(c) for ds, c in graphs.items()}
    ann = {ds: annotated_types(c, args.min_cells) for ds, c in graphs.items()}
    print("annotated rate-unit types:", {ds: len(v) for ds, v in ann.items()})

    # baseline: the full inference each dataset's rounds used, and how much of the truth the retina can even express
    base_cols, baseline = {}, {}
    for ds, c in graphs.items():
        col = infer_columns(c, retinas[ds]); base_cols[ds] = col
        n = c.neurons; annm = (n.hex_source == "annotation").to_numpy()
        tt = truth_table(c, retinas[ds], np.flatnonzero(annm))
        ridx = rate_index(c)
        baseline[ds] = {"n_cells": int(c.n), "n_rate_units": int(len(ridx)), "n_annotated": int(annm.sum()),
                        "annotated_types": ann[ds], "n_annotated_types_scored": len(ann[ds]),
                        "n_annotated_columns_absent_from_retina": int((tt.truth_col < 0).sum()),
                        "retina_columns": int(retinas[ds].n_columns), "rate_units_assigned": int((col[ridx] >= 0).sum()),
                        "rate_units_unassigned": int((col[ridx] < 0).sum())}
        print(f"[{ds}] N {c.n}, rate units {len(ridx)}, annotated {annm.sum()} ({baseline[ds]['n_annotated_columns_absent_from_retina']} in columns the retina lacks), "
              f"retina {retinas[ds].n_columns} columns, rate units assigned {baseline[ds]['rate_units_assigned']}")

    # 1 + 3: leave-one-type-out with chance
    per_type, per_cell, loo_partners = [], {}, []
    for ds, c in graphs.items():
        groups = {f"loo:{t}": [t] for t in ann[ds]}
        if ds == "fafb" and "malecns" in ann:
            extra = [t for t in ann["fafb"] if t not in set(ann["malecns"])]
            if extra:
                groups["loo_malecns_set"] = extra
        pt, pc, fp = leave_one_out(c, retinas[ds], groups, rng, args.shuffles)
        pt.insert(0, "dataset", ds); pt["condition"] = pt.condition.where(pt.condition == "loo_malecns_set", "loo")
        pc.insert(0, "dataset", ds); pc["condition"] = pc.condition.where(pc.condition == "loo_malecns_set", "loo")
        fp.insert(0, "dataset", ds); fp["condition"] = fp.condition.where(fp.condition == "loo_malecns_set", "loo")
        per_type.append(pt); per_cell[ds] = pc; loo_partners.append(fp)
        pc.to_csv(out / f"loo_per_cell_{ds}.csv", index=False)
    per_type = pd.concat(per_type, ignore_index=True); loo_partners = pd.concat(loo_partners, ignore_index=True)
    per_type.to_csv(out / "loo_per_type.csv", index=False); loo_partners.to_csv(out / "loo_failure_partner.csv", index=False)

    # 2: the LC case (and the LC-input stage as the small-field control), both datasets, plus the FAFB depth comparison
    lc_cells, lc_summ = [], []
    for ds, c in graphs.items():
        pcs, summ = lc_case(c, retinas[ds], base_cols[ds], LC_TYPES + LC_INPUT_TYPES, rng, args.shuffles)
        lc_cells.append(pcs); lc_summ.append(summ)
    lc_cells = pd.concat(lc_cells, ignore_index=True); lc_summ = pd.concat(lc_summ, ignore_index=True)
    partners = pd.concat([strongest_partner(c, base_cols[ds], LC_TYPES + LC_INPUT_TYPES) for ds, c in graphs.items()], ignore_index=True)
    partners.to_csv(out / "lc_strongest_partner.csv", index=False)
    depth = pd.DataFrame()
    if "fafb" in graphs and "malecns" in ann:
        c = graphs["fafb"]; r = retinas["fafb"]
        extra = [t for t in ann["fafb"] if t not in set(ann["malecns"])]
        col_m = infer_columns(withhold(c, extra), r)
        cae = np.asarray(r.col_az_el, float)
        m = lc_cells.dataset == "fafb"; ii = lc_cells.loc[m, "model_index"].to_numpy(); cm = col_m[ii]; cf = lc_cells.loc[m, "inferred_col"].to_numpy()
        both = (cm >= 0) & (cf >= 0)
        lc_cells.loc[m, "inferred_col_malecns_set"] = cm
        d_fm = angular_distance_deg(cae[np.maximum(cm, 0), 0], cae[np.maximum(cm, 0), 1], cae[np.maximum(cf, 0), 0], cae[np.maximum(cf, 0), 1])
        lc_cells.loc[m, "full_vs_malecns_set_deg"] = np.where(both, np.where(cm == cf, 0.0, d_fm), np.nan)   # identical column = exactly 0, as in score_cells
        pcs_m, summ_m = lc_case(c, r, col_m, LC_TYPES + LC_INPUT_TYPES, rng, args.shuffles)
        rows = []
        for t in LC_TYPES + LC_INPUT_TYPES:
            g = lc_cells[m & (lc_cells.type == t)]; sf = lc_summ[(lc_summ.dataset == "fafb") & (lc_summ.type == t) & (lc_summ.side == "both")].iloc[0]
            sm = summ_m[(summ_m.type == t) & (summ_m.side == "both")].iloc[0]
            d = g.full_vs_malecns_set_deg.to_numpy(float); fin = np.isfinite(d)
            # the identity test is on the column INDEX: arccos of a dot product returns up to ~1.2e-6 deg for two
            # identical columns, which `d < 1e-9` would read as a different column (score_cells shortcuts the same way)
            same = fin & (g.inferred_col.to_numpy(float) == g.inferred_col_malecns_set.to_numpy(float))
            rows.append({"type": t, "n_cells": int(len(g)), "n_distinct_columns_full": int(sf.n_distinct_columns), "n_distinct_columns_malecns_set": int(sm.n_distinct_columns),
                         "largest_column_share_full": float(sf.largest_column_share), "largest_column_share_malecns_set": float(sm.largest_column_share),
                         "frac_no_column_malecns_set": float(sm.frac_no_column),
                         "frac_same_column": float(same.sum() / max(len(g), 1)), "frac_within_io_deg": float((fin & (d <= INTEROMMATIDIAL_DEG + DEG_TOL)).sum() / max(len(g), 1)),
                         "median_full_vs_malecns_set_deg": float(np.median(d[fin])) if fin.any() else np.nan, "p90_full_vs_malecns_set_deg": float(np.percentile(d[fin], 90)) if fin.any() else np.nan,
                         "median_centroid_to_col_deg_full": float(sf.median_centroid_to_col_deg), "median_centroid_to_col_deg_malecns_set": float(sm.median_centroid_to_col_deg),
                         "chance_median_centroid_to_col_deg": float(sf.chance_median_centroid_to_col_deg)})
        depth = pd.DataFrame(rows)
    lc_cells.to_csv(out / "lc_per_cell.csv", index=False); lc_summ.to_csv(out / "lc_summary.csv", index=False)
    cross = cross_dataset(lc_cells) if len(graphs) > 1 else pd.DataFrame()
    cross.to_csv(out / "lc_cross.csv", index=False); depth.to_csv(out / "fafb_annotation_depth.csv", index=False)

    # the verdict inputs: the LC-input-like types where the truth exists
    b = per_type[per_type.side == "both"]
    verdict_rows = b[b.type.isin(["T2", "T2a", "T3", "Tm3", "Tm4", "Tm9", "Tm20", "Tm1", "Tm2", "Tm6", "Mi1"])][
        ["dataset", "condition", "type", "n_truth", "frac_assigned", "frac_exact", "median_deg_err", "p90_deg_err", "frac_within_io_deg", "chance_frac_exact", "chance_median_deg_err"]]
    loo_all = b[b.condition == "loo"].groupby("dataset").agg(n_types=("type", "size"), n_truth=("n_truth", "sum"), frac_exact_min=("frac_exact", "min"),
                                                            frac_exact_median=("frac_exact", "median"), median_deg_err_max=("median_deg_err", "max"),
                                                            p90_deg_err_max=("p90_deg_err", "max"), chance_frac_exact_max=("chance_frac_exact", "max"),
                                                            chance_median_deg_err_min=("chance_median_deg_err", "min"))
    lcb = lc_summ[lc_summ.side == "both"]

    prov = {ds: common.provenance(c, device="cpu", retina=tr.retina_record(retinas[ds], c),
                                  stimulus={"protocol": "column_ground_truth", "params": {"min_cells": args.min_cells, "shuffles": args.shuffles, "seed": args.seed}, "control": "within-type-and-side permutation of the inferred columns"})
            for ds, c in graphs.items()}
    res = common.Result.new("trace", prov["fafb" if "fafb" in prov else args.datasets[0]])
    res.tool_version = "probe_column_ground_truth/1"
    res.add_table("loo_per_type", per_type); res.add_table("lc_summary", lc_summ); res.add_table("lc_cross", cross); res.add_table("fafb_annotation_depth", depth); res.add_table("lc_strongest_partner", partners); res.add_table("loo_failure_partner", loo_partners)
    res.add_table("verdict_inputs", verdict_rows); res.add_table("loo_overall", loo_all.reset_index())
    res.summary = {"baseline": baseline, "provenance_by_dataset": prov, "datasets": list(graphs), "lc_types": list(LC_TYPES), "lc_input_types": list(LC_INPUT_TYPES),
                   "interommatidial_deg": INTEROMMATIDIAL_DEG, "shuffles": args.shuffles, "seed": args.seed,
                   "method": {"inference": "flyverse.interp.trace.column_of_cells over OpticLobe.rate_idx (ol_intrinsic minus photoreceptors): annotation, three passes of strongest-|W|-rate-input propagation, spiking cells from their strongest rate input",
                              "truth": "hex_source == 'annotation' (FAFB column_assignment through hex1 = q + 18, hex2 = p + 20; MaleCNS assignedOlHex1/2); the retina is built from the FULL annotation and never withheld",
                              "loo": "withhold one type (condition loo) or every FAFB type MaleCNS does not annotate (loo_malecns_set); score each withheld type against its own truth",
                              "col_err": "Euclidean distance in retina.py's hex embedding (X = h1 - h2/2, Y = h2 sqrt(3)/2), same side only; deg_err = great circle between retina.col_az_el of the inferred and the truth column (both in the retina)",
                              "chance": "inferred columns permuted within (type, truth side), mean of the statistics over `shuffles` draws"},
                   "elapsed_s": time.time() - t_start,
                   "lc_headline": {f"{r.dataset}:{r.type}": {"n": int(r.n_cells), "distinct_columns": int(r.n_distinct_columns), "largest_share": round(float(r.largest_column_share), 3),
                                                             "frac_no_column": round(float(r.frac_no_column), 3), "median_centroid_to_col_deg": round(float(r.median_centroid_to_col_deg), 1),
                                                             "chance_median_centroid_to_col_deg": round(float(r.chance_median_centroid_to_col_deg), 1), "median_r50_deg": round(float(r.median_r50_deg), 1)}
                                   for r in lcb.itertuples()}}
    res.validation = {"name": "column_of_cells against the two releases' column annotations (leave-one-type-out) and the LC single-column prior",
                      "reference": {"interommatidial_deg": INTEROMMATIDIAL_DEG, "object_localizer_r3_1_2": {"LC11_distinct_columns_malecns": 13, "LC10a_distinct_columns_malecns": 88, "LC11_centroid_to_col_median_deg": 64.5, "LC10a_centroid_to_col_median_deg": 47.1}},
                      "measured": {"loo_overall": loo_all.reset_index().to_dict("records"), "verdict_inputs": verdict_rows.to_dict("records"), "lc": res.summary["lc_headline"],
                                   "fafb_annotation_depth": depth.to_dict("records"), "lc_strongest_partner": partners.to_dict("records")},
                      "status": "measured", "source": "docs/audits/column_ground_truth.md"}
    res.files = {"generator": "scripts/probe_column_ground_truth.py", "per_cell": [str(out / f"loo_per_cell_{ds}.csv") for ds in graphs] + [str(out / "lc_per_cell.csv")],
                 "tables": [str(out / f) for f in ("loo_per_type.csv", "loo_failure_partner.csv", "lc_summary.csv", "lc_cross.csv", "fafb_annotation_depth.csv", "lc_strongest_partner.csv")]}
    path = res.save(out / "column_ground_truth.json")
    print("\nleave-one-type-out, pooled sides:")
    common.print_table(b[["dataset", "condition", "type", "n_truth", "frac_assigned", "frac_exact", "median_col_err", "median_deg_err", "p90_deg_err", "frac_within_io_deg", "chance_frac_exact", "chance_median_deg_err"]], max_rows=120)
    print("\nLC case, pooled sides:")
    common.print_table(lcb[["dataset", "type", "n_cells", "frac_no_column", "n_distinct_columns", "largest_column_share", "median_n_input_columns", "median_r50_deg", "median_centroid_to_col_deg", "chance_median_centroid_to_col_deg"]])
    print("\nwhy a withheld type fails (strongest partner over all cells / over cells > 10 deg off):")
    common.print_table(loo_partners[loo_partners.n_bad >= 30][["dataset", "condition", "type", "n_cells", "n_bad", "partner_types_top5", "bad_partner_types_top5", "frac_partner_annotated", "frac_bad_partner_annotated"]])
    print("\nthe partner whose column an LC / LC-input cell inherits:")
    common.print_table(partners)
    if len(depth):
        print("\nFAFB: full annotation vs the MaleCNS-like annotation set:")
        common.print_table(depth)
    print(f"\nwrote {path}  ({time.time() - t_start:.0f} s); problems: {res.check()}")
    return res


if __name__ == "__main__":
    main()
