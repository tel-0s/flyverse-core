"""Recover a CANDIDATE BANC lattice and report the gates before R1-R6 integration.

CPU only; never edits a Connectome, cache, capability, or synapse. Outputs are
diagnostic CSV/JSON files, NOT an annotation consumed by load(). In particular,
an integer, collision-free drawing is not evidence that its columns are correct.

    python scripts/recover_banc_columns.py --fafb-control --plot --out out/banc_columns

Exit 2 means the report was written but the anatomical validation gate did not
pass. T4a/b choose one of twelve lattice orientations; T4c/d are held out from
that choice. All four unordered T4 populations contribute to the neighbour graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import lsqr
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import connectome as cn
from flyverse.interp import common

T4 = ("T4a", "T4b", "T4c", "T4d")
STEPS = np.array([[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]])
CARTESIAN = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2]])
# Same-animal, one-to-one assignments; no cross-animal body ID/type seam.
# Zero-score matches are discarded. T4 home columns are especially uncertain;
# the offset validation uses their INPUT centroids, not their assigned homes.
ANCHORS = (
    ("L1", ("Mi1",)),
    ("L5", ("Mi1",)),
    ("L2", ("L5",)),
    ("L3", ("Mi1",)),
    ("Mi4", ("Mi1", "L1", "L5", "L2", "L3")),
    ("Mi9", ("Mi1", "L1", "L5", "L2", "L3")),
    ("R8_unclear", ("Mi1", "L1", "L3", "Mi4", "Mi9")),
    ("R7_unclear", ("R8_unclear", "L3")),
    *((t, ("Mi1", "L1", "L5")) for t in T4),
)
PARAMETERS = {
    "neighbours": 6,
    "shared_partner_weight": "raw count dot product",
    "embedding": "classical MDS of unweighted graph distances",
    "robust_iterations": 12,
    "residual_floor": 0.01,
    "assignment_radius": 2,
    "rim_depth_rows": 2,
    "t4_cosine_min": 0.95,
    "right_column_range": [750, 950],
}


def symmetries():
    """Twelve integer isometries of the 120-degree axial lattice."""
    return [
        np.array([STEPS[k], STEPS[(k + sign * 2) % 6]])
        for k in range(6)
        for sign in (-1, 1)
    ]


def distances(delta):
    return np.sqrt(
        delta[..., 0] ** 2 + delta[..., 1] ** 2 - delta[..., 0] * delta[..., 1]
    )


def selected(c, cell_type, side):
    idx = c.select(type=cell_type, somaSide=side)
    return idx[np.argsort(c.neurons.bodyId.to_numpy()[idx], kind="stable")]


def reconstruct(inputs, *, neighbours=6):
    """Shared-partner graph -> largest component -> approximate integer lattice.

    inputs[partner, Mi1] contains nonnegative same-animal synapse counts. No
    subtype names, geometry, DRA labels or reference lattice enter this function.
    Disconnected cells remain unassigned. Injectivity is an imposed constraint:
    report every displacement from the unconstrained rounded solution.
    """
    w = np.asarray(inputs, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] < 3 or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError(
            "need finite nonnegative partner-by-cell counts for at least three cells"
        )
    if not isinstance(neighbours, int) or neighbours < 1:
        raise ValueError("neighbours must be a positive integer")
    affinity = w.T @ w
    np.fill_diagonal(affinity, 0)
    order = np.argsort(-affinity, axis=1, kind="stable")[:, :neighbours]
    graph = np.zeros_like(affinity)
    np.put_along_axis(graph, order, np.take_along_axis(affinity, order, axis=1), axis=1)
    graph = np.minimum(graph, graph.T)
    _, labels = connected_components(sp.csr_matrix(graph))
    sizes = np.bincount(labels)
    keep = np.flatnonzero(labels == sizes.argmax())
    if len(keep) < 3:
        raise ValueError("no connected component with at least three Mi1 cells")
    graph = graph[np.ix_(keep, keep)]
    d = shortest_path(sp.csr_matrix(graph > 0), directed=False)
    centering = np.eye(len(d)) - 1 / len(d)
    gram = -0.5 * centering @ (d * d) @ centering
    values, vectors = eigh(gram, subset_by_index=[len(d) - 2, len(d) - 1])
    if values.min() <= 0:
        raise ValueError("shared-partner graph has no two-dimensional embedding")
    xy = vectors * np.sqrt(values)
    i, j = np.nonzero(np.triu(graph, 1))
    delta = xy[j] - xy[i]
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    angle = np.angle(np.mean(np.exp(6j * angles))) / 6
    directions = np.rint((angles - angle) / (np.pi / 3)).astype(int) % 6
    steps = STEPS[directions]
    incidence = sp.csr_matrix(
        (
            np.tile([-1.0, 1.0], len(i)),
            (np.repeat(np.arange(len(i)), 2), np.c_[i, j].ravel()),
        ),
        shape=(len(i), len(keep)),
    )
    weights = np.sqrt(graph[i, j])
    weights /= np.median(weights)
    for _ in range(PARAMETERS["robust_iterations"]):
        system = sp.diags(weights) @ incidence
        h = np.column_stack(
            [
                lsqr(system, steps[:, axis] * weights, atol=1e-10, btol=1e-10)[0]
                for axis in range(2)
            ]
        )
        residual = np.linalg.norm(incidence @ h - steps, axis=1)
        weights = np.sqrt(graph[i, j]) / np.sqrt(
            np.maximum(residual, PARAMETERS["residual_floor"])
        )
    h -= np.angle(np.mean(np.exp(2j * np.pi * h), axis=0)) / (2 * np.pi)
    rounded = np.rint(h).astype(np.int64)
    radius = PARAMETERS["assignment_radius"]
    offsets = np.array(
        [[a, b] for a in range(-radius, radius + 1) for b in range(-radius, radius + 1)]
    )
    sites = np.unique((rounded[:, None] + offsets[None, :]).reshape(-1, 2), axis=0)
    cost = cdist(h @ CARTESIAN, sites @ CARTESIAN) ** 2
    rows, cols = linear_sum_assignment(cost)
    assigned = sites[cols]
    moved = np.any(rounded != assigned, axis=1)
    return {
        "keep": keep,
        "hex": assigned,
        "continuous": h,
        "xy": xy,
        "moved": moved,
        "assignment_distance": np.sqrt(cost[rows, cols]),
        "diagnostics": {
            "n_seed_cells": w.shape[1],
            "n_columns": len(keep),
            "component_sizes": sorted(sizes.tolist(), reverse=True),
            "n_neighbour_edges": len(i),
            "rounded_collisions": int(len(rounded) - len(np.unique(rounded, axis=0))),
            "injectivity_displacements": int(moved.sum()),
            "max_assignment_distance": float(np.sqrt(cost[rows, cols]).max()),
            "edge_step_agreement": float(
                np.all(incidence @ assigned == steps, axis=1).mean()
            ),
            "caveat": "injective coordinates are a candidate, not validated column annotations",
        },
    }


def assign_cells(c, side, seed_indices, fit):
    """Transfer candidate coordinates using positive, within-type bijective matches."""
    n = c.neurons
    seed_hex = np.full((len(seed_indices), 2), np.nan)
    seed_hex[fit["keep"]] = fit["hex"]
    maps = {"Mi1": np.arange(len(seed_indices))}
    populations = {"Mi1": seed_indices}
    diagnostics = []
    for cell_type, anchors in ANCHORS:
        idx = selected(c, cell_type, side)
        score = np.zeros((len(idx), len(seed_indices)))
        for anchor in anchors:
            valid = maps[anchor] >= 0
            home = populations[anchor][valid]
            columns = maps[anchor][valid]
            # W[post, pre]: count both directions without cancellations of signs.
            score[:, columns] += (
                abs(c.W[idx][:, home]).toarray() + abs(c.W[home][:, idx]).toarray().T
            )
        rows, cols = linear_sum_assignment(score, maximize=True)
        positive = score[rows, cols] > 0
        mapping = np.full(len(idx), -1, dtype=np.int64)
        mapping[rows[positive]] = cols[positive]
        maps[cell_type], populations[cell_type] = mapping, idx
        matched = mapping >= 0
        diagnostics.append(
            {
                "type": cell_type,
                "n": len(idx),
                "positive_matches": int(matched.sum()),
                "with_coordinates": int(
                    np.isfinite(seed_hex[mapping[matched]]).all(axis=1).sum()
                ),
                "matched_synapse_fraction": float(
                    score[rows, cols].sum() / max(score.sum(), 1)
                ),
            }
        )
    records = []
    for cell_type, idx in populations.items():
        for index, column in zip(idx, maps[cell_type]):
            h = seed_hex[column] if column >= 0 else [np.nan, np.nan]
            records.append(
                {
                    "bodyId": int(n.bodyId.iloc[index]),
                    "type": cell_type,
                    "side": side,
                    "seed_bodyId": str(int(n.bodyId.iloc[seed_indices[column]]))
                    if column >= 0
                    else "",
                    "hex1": h[0],
                    "hex2": h[1],
                    "hex_source": "connectivity_candidate"
                    if np.isfinite(h).all()
                    else "unassigned",
                }
            )
    return pd.DataFrame(records), diagnostics


def t4_offsets(c, side, table=None):
    """Mi4-minus-Mi1 input centroids, matching the existing spec-2.5 check."""
    coordinates = (
        c.neurons.set_index("bodyId")[["hex1", "hex2"]]
        if table is None
        else table.set_index("bodyId")[["hex1", "hex2"]]
    )
    result = {}
    for cell_type in T4:
        post = selected(c, cell_type, side)
        means, valid = {}, np.ones(len(post), dtype=bool)
        for pre_type in ("Mi1", "Mi4"):
            pre = selected(c, pre_type, side)
            h = coordinates.reindex(c.neurons.bodyId.to_numpy()[pre]).to_numpy()
            mask = np.isfinite(h).all(axis=1)
            w = abs(c.W[post][:, pre[mask]])
            total = np.asarray(w.sum(axis=1)).ravel()
            means[pre_type] = w @ h[mask] / np.maximum(total[:, None], 1)
            valid &= total > 0
        offset = (
            (means["Mi4"] - means["Mi1"])[valid].mean(axis=0) if valid.any() else None
        )
        result[cell_type] = {
            "n": int(valid.sum()),
            "offset": None if offset is None else offset.tolist(),
        }
    return result


def cosine(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a), np.asarray(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 0 and np.isfinite(norm) else None


def orient(c, male, side, table):
    raw, target = t4_offsets(c, side, table), t4_offsets(male, side)
    fits = []
    for matrix in symmetries():
        scores = [
            cosine(
                None
                if raw[t]["offset"] is None
                else np.array(raw[t]["offset"]) @ matrix,
                target[t]["offset"],
            )
            for t in T4[:2]
        ]
        if all(v is not None for v in scores):
            fits.append((sum(scores), matrix))
    if not fits:
        return table.copy(), {
            "status": "unavailable",
            "reason": "no nonzero T4a/b orientation anchors",
            "offsets": raw,
        }
    _, matrix = max(fits, key=lambda pair: pair[0])
    oriented = table.copy()
    oriented[["hex1", "hex2"]] = table[["hex1", "hex2"]].to_numpy() @ matrix
    offsets = t4_offsets(c, side, oriented)
    for t, row in offsets.items():
        row.update(
            reference_offset=target[t]["offset"],
            cosine=cosine(row["offset"], target[t]["offset"]),
            role="orientation_fit" if t in T4[:2] else "held_out_from_orientation",
        )
    passed = all(
        row["cosine"] is not None and row["cosine"] > PARAMETERS["t4_cosine_min"]
        for row in offsets.values()
    )
    return oriented, {
        "status": "pass" if passed else "fail",
        "matrix": matrix.tolist(),
        "offsets": offsets,
        "caveat": "T4a/b fit orientation; c/d hold out subtype direction, but all T4s contributed unordered shared partners",
    }


def dra_check(c, side, table):
    """A held-out connectivity proxy, NOT invented DRA photoreceptor labels.

    Use only the literature pairs R7->Dm-DRA1 and R8->Dm-DRA2. The rim and
    orientation were recovered without these edges. Unassigned proxy cells stay
    in the denominator; no empty or partially mapped set can pass the strict gate.
    """
    records = []
    coordinates = table.set_index("bodyId")
    grid = table.loc[table.type.eq("Mi1"), ["hex1", "hex2"]].dropna().drop_duplicates()
    envelope = (
        pd.DataFrame({"ap": grid.hex1 - grid.hex2, "height": grid.hex1 + grid.hex2})
        .groupby("ap")
        .height.max()
    )
    boundary = float(np.quantile(grid.sum(axis=1), 0.8)) if len(grid) else None
    for pre_type, post_type in (("R7_unclear", "Dm-DRA1"), ("R8_unclear", "Dm-DRA2")):
        pre, post = selected(c, pre_type, side), selected(c, post_type, side)
        counts = np.asarray(abs(c.W[post][:, pre]).sum(axis=0)).ravel()
        for index, count in zip(pre[counts > 0], counts[counts > 0]):
            body = int(c.neurons.bodyId.iloc[index])
            h = coordinates.reindex([body])[["hex1", "hex2"]].to_numpy()[0]
            mapped = bool(np.isfinite(h).all())
            depth = (float(envelope.loc[h[0] - h[1]] - h.sum()) / 2) if mapped else None
            records.append(
                {
                    "bodyId": str(body),
                    "type": pre_type,
                    "target": post_type,
                    "synapses": float(count),
                    "mapped": mapped,
                    "hex": None if not mapped else h.tolist(),
                    "depth_rows": depth,
                    "in_dorsal_band": bool(mapped and h.sum() >= boundary),
                    "within_two_rows": bool(
                        mapped and depth <= PARAMETERS["rim_depth_rows"]
                    ),
                }
            )
    mapped = sum(row["mapped"] for row in records)
    rim = sum(row["within_two_rows"] for row in records)
    return {
        "status": "pass"
        if len(records) > 0 and rim == len(records)
        else "fail"
        if records
        else "unavailable",
        "source": "same-animal R7->Dm-DRA1 and R8->Dm-DRA2 connectivity proxy; not direct DRA labels",
        "n_candidates": len(records),
        "mapped": mapped,
        "within_two_rows": rim,
        "in_dorsal_band": sum(row["in_dorsal_band"] for row in records),
        "dorsal_boundary_q80": boundary,
        "cells": records,
    }


def compare_truth(table, c):
    """Align lattice symmetry and integer origin ONLY for external evaluation."""
    truth = (
        c.neurons.set_index("bodyId")[["hex1", "hex2"]].reindex(table.bodyId).to_numpy()
    )
    estimate = table[["hex1", "hex2"]].to_numpy()
    seeds = (
        table.type.eq("Mi1").to_numpy()
        & np.isfinite(truth).all(axis=1)
        & np.isfinite(estimate).all(axis=1)
    )
    if not seeds.any():
        return {
            "status": "unavailable",
            "reason": "no annotated Mi1 control cells",
        }, table.copy()
    choices = []
    for matrix in symmetries():
        delta = truth[seeds] - estimate[seeds] @ matrix
        shifts, counts = np.unique(delta, axis=0, return_counts=True)
        choices.append((int(counts.max()), matrix, shifts[counts.argmax()]))
    _, matrix, shift = max(choices, key=lambda row: row[0])
    aligned = estimate @ matrix + shift
    rows = []
    for cell_type in table.type.unique():
        population = table.type.eq(cell_type).to_numpy()
        valid = (
            population
            & np.isfinite(truth).all(axis=1)
            & np.isfinite(aligned).all(axis=1)
        )
        error = distances(truth[valid] - aligned[valid])
        rows.append(
            {
                "type": cell_type,
                "n": int(population.sum()),
                "evaluated": len(error),
                "exact": int((error == 0).sum()),
                "within_one_column": int((error <= 1).sum()),
                "max_error_columns": float(error.max()) if len(error) else None,
            }
        )
    copy = table.copy()
    copy["control_error_columns"] = distances(truth - aligned)
    return {
        "status": "measured",
        "alignment_matrix": matrix.tolist(),
        "alignment_translation": shift.tolist(),
        "caveat": "control annotations enter only this evaluation, never reconstruction or same-animal assignment",
        "by_type": rows,
    }, copy


def gate(checks):
    required = ("i_dra_rim", "ii_lr_mirror", "iii_t4_direction", "iv_column_count")
    return all(checks.get(name, {}).get("status") == "pass" for name in required)


def recover_eye(c, male, side):
    seed = selected(c, "Mi1", side)
    post = selected(c, list(T4), side)
    fit = reconstruct(
        abs(c.W[post][:, seed]).toarray(), neighbours=PARAMETERS["neighbours"]
    )
    table, assignment = assign_cells(c, side, seed, fit)
    table, orientation = orient(c, male, side, table)
    report = {
        "side": side,
        "reconstruction": fit["diagnostics"],
        "type_assignments": assignment,
        "orientation": orientation,
        "dra": dra_check(c, side, table),
    }
    report["unassigned_seed_body_ids"] = [
        str(int(b))
        for b in c.neurons.bodyId.to_numpy()[
            seed[np.setdiff1d(np.arange(len(seed)), fit["keep"])]
        ]
    ]
    uncertainty = dict(
        zip(
            c.neurons.bodyId.to_numpy()[seed[fit["keep"]]],
            zip(fit["moved"], fit["assignment_distance"]),
        )
    )
    table["seed_displaced_for_injectivity"] = [
        bool(uncertainty.get(int(b), (False, np.nan))[0]) if b else False
        for b in table.seed_bodyId
    ]
    table["seed_assignment_distance"] = [
        float(uncertainty.get(int(b), (False, np.nan))[1]) if b else np.nan
        for b in table.seed_bodyId
    ]
    return table, report, fit


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def direct_dra_labels():
    """Record the actual BANC label availability, rather than infer it from aliases."""
    path = cn.data_directory("banc") / "neurons.csv.gz"
    if not path.exists():
        return {"status": "unavailable", "reason": "raw BANC neuron table is absent"}
    raw = pd.read_csv(
        path, usecols=["Root ID", "Primary Cell Type", "Class", "Community labels"]
    )
    photoreceptor = raw.Class.eq("photoreceptor_neuron") | raw[
        "Primary Cell Type"
    ].isin(["R7", "R8", "R7d", "R8d", "R7_DRA", "R8_DRA"])
    dra = raw["Primary Cell Type"].isin(["R7d", "R8d", "R7_DRA", "R8_DRA"]) | raw[
        "Community labels"
    ].str.contains("DRA|dorsal rim", case=False, na=False)
    ids = raw.loc[photoreceptor & dra, "Root ID"]
    return {
        "status": "absent" if ids.empty else "requires_validation",
        "source_file": path.name,
        "sha256": file_hash(path),
        "n_labels": len(ids),
        "body_ids": [str(int(b)) for b in ids],
    }


def plot_candidate(path, table, fit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].scatter(*fit["xy"].T, s=8, color="#4683a8")
    axes[0].set_title("BANC shared-partner embedding")
    seeds = table[table.type.eq("Mi1") & table.hex1.notna()]
    xy = seeds[["hex1", "hex2"]].to_numpy() @ CARTESIAN
    axes[1].scatter(
        *xy.T,
        s=10,
        c=np.where(seeds.seed_displaced_for_injectivity, "#b13f35", "#4683a8"),
    )
    axes[1].set_title("Candidate lattice (red: injectivity displaced seed)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("lattice units (arbitrary origin)")
    fig.suptitle("DIAGNOSTIC ONLY — anatomical gate has not passed")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fafb-control", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    # Decline accidental overwrites of previous evidence, including cache paths.
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("--out must be an empty directory for a new diagnostic run")
    male, banc = cn.load(verbose=False), cn.load(dataset="banc", verbose=False)
    snapshots = {
        name: common.connectome_fingerprint(c)
        for name, c in (("malecns", male), ("banc", banc))
    }
    report = {
        "provenance": common.provenance(
            banc,
            device="cpu",
            stimulus={
                "protocol": "banc_column_reconstruction",
                "params": PARAMETERS,
                "control": "FAFB published map" if args.fafb_control else None,
            },
        ),
        "generator": {
            "path": "scripts/recover_banc_columns.py",
            "sha256": file_hash(__file__),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "parameters": PARAMETERS,
        "reference_provenance": common.provenance(male, device="cpu"),
        "direct_dra_labels": direct_dra_labels(),
        "eyes": {},
    }
    artifacts = []
    for side in ("R", "L"):
        try:
            table, eye, fit = recover_eye(banc, male, side)
        except ValueError as error:
            report["eyes"][side] = {"status": "unavailable", "reason": str(error)}
            continue
        report["eyes"][side] = eye
        path = args.out / f"banc_{side}_candidate.csv"
        table.to_csv(path, index=False, float_format="%.9g")
        artifacts.append(path)
        if side == "R" and args.plot:
            path = args.out / "banc_R_candidate.png"
            plot_candidate(path, table, fit)
            artifacts.append(path)
        print(
            f"BANC {side}: {eye['reconstruction']['n_columns']}/{eye['reconstruction']['n_seed_cells']} Mi1 seeds; "
            f"{len(eye['reconstruction']['component_sizes'])} components; DRA {eye['dra']['status']}",
            flush=True,
        )
    right = report["eyes"].get("R", {})
    count = right.get("reconstruction", {}).get("n_columns", 0)
    dra = right.get("dra", {"status": "unavailable"})
    # A successful proxy would still need corroboration of the literal spec check.
    # Do not quietly substitute a new anatomical criterion for the original gate.
    dra_gate = {
        "status": "fail" if dra["status"] == "fail" else "unavailable",
        "proxy": dra,
        "reason": "DRA connectivity proxy is diagnostic; direct photoreceptor DRA labels have not been validated",
    }
    checks = {
        "i_dra_rim": dra_gate,
        "ii_lr_mirror": {
            "status": "unavailable",
            "reason": "independent left candidate has no validated column correspondence/origin; reflecting an eye by construction is not an anatomical mirror test",
        },
        "iii_t4_direction": right.get("orientation", {"status": "unavailable"}),
        "iv_column_count": {
            "status": "pass" if 750 <= count <= 950 else "fail",
            "right_columns": count,
            "interval": [750, 950],
            "caveat": "population sanity check, not validation of individual positions",
        },
    }
    report.update(
        checks=checks,
        integration_ready=gate(checks),
        synthetic_nodes_added=0,
        synthetic_edges_added=0,
    )
    if args.fafb_control:
        fafb = cn.load(dataset="fafb", verbose=False)
        table, eye, fit = recover_eye(fafb, male, "R")
        control, table = compare_truth(table, fafb)
        report["fafb_control"] = {
            "provenance": common.provenance(fafb, device="cpu"),
            "reconstruction": eye,
            "comparison": control,
        }
        path = args.out / "fafb_R_control.csv"
        table.to_csv(path, index=False, float_format="%.9g")
        artifacts.append(path)
    report["unchanged_fingerprints"] = {
        name: common.connectome_fingerprint(c) == snapshots[name]
        for name, c in (("malecns", male), ("banc", banc))
    }
    assert all(report["unchanged_fingerprints"].values())
    report["malecns_cache_md5"] = {
        name: file_hash(male.cache_dir / name, "md5")
        for name in ("neurons.parquet", "W_post_pre.npz", "sign0_counts.npz")
    }
    report["artifacts"] = {p.name: file_hash(p) for p in artifacts}
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("Integration gate:", {k: v["status"] for k, v in checks.items()}, flush=True)
    return 0 if report["integration_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
