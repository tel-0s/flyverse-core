"""Explicit, unvalidated BANC right-eye candidate; never used by the default loader."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).with_name("data")
QUALIFICATION = "synthetic input layer on a candidate lattice"


def annotation_sha256(c):
    """Platform-independent identity of the biological rows used by this map."""
    columns = ["bodyId", "type", "somaSide", "nt", "superclass"]
    records = c.neurons[columns].fillna("").itertuples(index=False, name=None)
    h = hashlib.sha256()
    for body, *labels in records:
        h.update(
            (
                json.dumps(
                    [int(body), *labels], ensure_ascii=True, separators=(",", ":")
                )
                + "\n"
            ).encode()
        )
    return h.hexdigest()


def read_candidate():
    meta = json.loads((DATA / "banc_candidate_vision.json").read_text(encoding="utf-8"))
    path = DATA / "banc_candidate_columns.csv"
    # Git's Windows checkout may use CRLF; the asset identity is defined over LF.
    content = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(content).hexdigest() != meta["columns_sha256"]:
        raise ValueError("BANC candidate column asset fingerprint mismatch")
    return pd.read_csv(path), meta


def _validate_base(c, meta):
    from .interp.common import _md5_csr

    if c.dataset != "banc" or c.reference is not c or c._extension is not None:
        raise ValueError("candidate vision requires a full, unextended BANC graph")
    if (
        c.release != meta["release"]
        or _md5_csr(c.W) != meta["base_csr_md5"]
        or annotation_sha256(c) != meta["base_annotations_sha256"]
        or c.neurons[["hex1", "hex2"]].notna().any().any()
    ):
        raise ValueError(
            "BANC candidate was reconstructed for a different biological graph/annotation table"
        )


def _candidate_tables(c, columns, meta):
    if columns.bodyId.duplicated().any() or not columns.side.eq("R").all():
        raise ValueError(
            "candidate asset must have unique biological IDs and only right-eye columns"
        )
    columns = columns.dropna(subset=["hex1", "hex2"]).copy()
    idx = c.index_of(columns.bodyId.to_numpy())
    if not np.array_equal(c.neurons.type.to_numpy()[idx], columns.type.to_numpy()):
        raise ValueError("candidate type identity mismatch")
    n = c.neurons.copy(deep=True)
    for key in ("hex_side", "hex_source"):
        n[key] = n[key].astype(object)
    n.loc[idx, ["hex1", "hex2"]] = columns[["hex1", "hex2"]].to_numpy()
    n.loc[idx, "hex_side"] = "R"
    n.loc[idx, "hex_source"] = "connectivity_candidate"
    seeds = columns[columns.type.eq("Mi1")].sort_values(["hex1", "hex2"])
    if seeds.duplicated(["hex1", "hex2"]).any():
        raise ValueError("candidate cartridge coordinates must be unique")
    cartridges = seeds[["hex1", "hex2"]].reset_index(drop=True)
    nodes = cartridges.loc[cartridges.index.repeat(6)].reset_index(drop=True)
    nodes["bodyId"] = -np.arange(1, len(nodes) + 1, dtype=np.int64)
    nodes["type"] = "R1-R6"
    nodes["instance"] = [
        f"candidate_R1-R6_{i // 6}_{i % 6 + 1}" for i in range(len(nodes))
    ]
    nodes["dataset"] = "synthetic"
    nodes["release"] = meta["candidate_id"]
    nodes["nt"] = "histamine"
    nodes["superclass"] = "ol_sensory"
    nodes["somaSide"] = nodes["hex_side"] = "R"
    nodes["hex_source"] = "synthetic_candidate"
    targets = columns[columns.type.isin(["L1", "L2", "L3"])]
    pairs = nodes[["bodyId", "hex1", "hex2"]].merge(
        targets, on=["hex1", "hex2"], suffixes=("_pre", "_post")
    )
    edges = pairs.rename(
        columns={"bodyId_pre": "body_pre", "bodyId_post": "body_post"}
    )[["body_pre", "body_post"]]
    edges["weight"] = pairs.type.map(meta["input_layer"]["weight_by_target"])
    return n, nodes, edges


def validate_candidate_cache(c):
    """Check persisted candidates against the packaged model, not just their label.

    This is read-only; it also accepts the original acceptance caches whose
    metadata predates the subsequently attached functional evidence.
    """
    from .interp.common import _md5_csr

    if not c.has_optic_columns or c.dataset != "banc" or c.reference is not c:
        raise ValueError("persisted graph is not a full BANC candidate")
    columns, meta = read_candidate()
    for key in (
        "candidate_id",
        "release",
        "columns_sha256",
        "base_csr_md5",
        "base_annotations_sha256",
        "input_layer",
        "checks",
        "accuracy_estimate",
    ):
        if c.vision.get(key) != meta[key]:
            raise ValueError(f"persisted candidate metadata mismatch: {key}")
    base = c._extension_base
    if base is None:
        raise ValueError("persisted candidate lacks its biological base")
    _validate_base(base, meta)
    n, nodes, edges = _candidate_tables(base, columns, meta)
    expected = pd.concat([n, nodes], ignore_index=True)
    # extend() fills these fields on synthetic nodes; native values remain intact.
    expected.loc[len(n) :, "sign"] = -1.0
    fields = [
        "bodyId",
        "type",
        "nt",
        "sign",
        "superclass",
        "somaSide",
        "hex1",
        "hex2",
        "hex_side",
        "hex_source",
    ]
    if (
        c.n != len(expected)
        or not expected[fields]
        .astype(object)
        .fillna("")
        .equals(c.neurons[fields].astype(object).fillna(""))
        or not c.neurons.dataset.iloc[len(n) :].eq("synthetic").all()
    ):
        raise ValueError(
            "persisted candidate neurons/columns differ from the pinned model"
        )
    mapping = pd.Series(np.arange(len(expected)), index=expected.bodyId)
    addition = sp.csr_matrix(
        (
            -edges.weight.to_numpy(np.float32),
            (
                mapping.loc[edges.body_post].to_numpy(),
                mapping.loc[edges.body_pre].to_numpy() - base.n,
            ),
        ),
        shape=(base.n, len(nodes)),
        dtype=base.W.dtype,
    )
    expected_w = sp.vstack(
        [
            sp.hstack([base.W, addition], format="csr"),
            sp.csr_matrix((len(nodes), len(expected)), dtype=base.W.dtype),
        ],
        format="csr",
    )
    if c.W.shape != expected_w.shape or _md5_csr(c.W) != _md5_csr(expected_w):
        raise ValueError("persisted candidate synapses differ from the pinned model")


def extend_candidate(c, *, cache_dir=None):
    """Append the predeclared model in a scratch cache, leaving native edges intact.

    Removing its synthetic nodes restores the original graph, including its lack
    of optical coordinates/capability.
    """
    from .connectome import save

    columns, meta = read_candidate()
    _validate_base(c, meta)
    if cache_dir is not None and c.cache_dir is not None:
        path, base = Path(cache_dir).resolve(), Path(c.cache_dir).resolve()
        if path == base or base in path.parents or path in base.parents:
            raise ValueError(
                "candidate scratch must be separate from the biological cache"
            )
    n, nodes, edges = _candidate_tables(c, columns, meta)
    annotated = replace(c, neurons=n, _norm_cache={})
    out = annotated.extend(nodes, edges, cache_dir=cache_dir)
    # extend() saved an annotated copy as its base. Restore the actual biological
    # base so prune() and a save/load round trip undo the candidate coordinates too.
    out._extension_base = c
    out._extension["vision"] = copy.deepcopy(meta)
    out._extension["vision"]["realised"] = {
        "cartridges": len(nodes) // 6,
        "synthetic_nodes": len(nodes),
        "synthetic_edges": len(edges),
        "target_cells": n.loc[n.type.isin(["L1", "L2", "L3"]) & n.hex1.notna(), "type"]
        .value_counts()
        .to_dict(),
        "empty_cartridges": len(
            nodes[["hex1", "hex2"]]
            .drop_duplicates()
            .merge(
                n.loc[
                    n.type.isin(["L1", "L2", "L3"]) & n.hex1.notna(), ["hex1", "hex2"]
                ].drop_duplicates(),
                how="left",
                indicator=True,
            )
            .query('_merge == "left_only"')
        ),
    }
    save(out, out.cache_dir)
    return out
