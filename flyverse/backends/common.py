"""Shared FlyWire table mechanics; no MaleCNS transmitter overrides."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

ALIASES = Path(__file__).resolve().parents[1] / "data" / "type_aliases.csv"
NT_NAMES = {"ACH": "acetylcholine", "GABA": "gaba", "GLUT": "glutamate", "HIST": "histamine",
            "DA": "dopamine", "OCT": "octopamine", "SER": "serotonin", "TYR": "tyramine"}
CLASSICAL = ("acetylcholine", "gaba", "glutamate", "histamine")
MONOAMINES = ("dopamine", "octopamine", "serotonin", "tyramine")
SIDES = {"left": "L", "right": "R", "center": "M", "L": "L", "R": "R", "M": "M"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_types(types, path=None):
    """Exact names win; ambiguous alias-only joins remain unresolved, never last-row-wins.

    The CSV documents flags that make an alias unusable. Retain those exclusions here,
    and record every unresolved source name in the compiler manifest for review.
    """
    a = pd.read_csv(path or ALIASES, comment="#").fillna("")
    a = a[a.system.isin(["flywire", "banc"]) & a.tier.isin(["exact", "alias"])]
    bad = {"unresolvable", "unmatched", "conflict_nern7_vs_malecns", "conflict_nern7_vs_malecns_notation"}
    a = a[~a.flag.map(lambda s: bool(set(s.split(";")) & bad))]
    mapping, ambiguous = {}, {}
    present = set(types.dropna())
    for alias, rows in a.groupby("alias", sort=True):
        exact = rows.loc[rows.tier == "exact", "malecns_type"].unique()
        primary = rows.loc[rows.flag.map(lambda s: "backend_primary" in s.split(";")), "malecns_type"].unique()
        candidates = exact if len(exact) else primary if len(primary) else rows.malecns_type.unique()
        if len(candidates) == 1:
            mapping[alias] = candidates[0]
        elif alias in present:
            ambiguous[alias] = sorted(candidates.tolist())
    result = types.map(mapping).fillna(types)
    return result, {"changed_cells": int((result.fillna("") != types.fillna("")).sum()),
                    "matched_cells": int(types.isin(mapping).sum()), "ambiguous": ambiguous}


def canonical_nt(value):
    if pd.isna(value):
        return "unknown"
    return NT_NAMES.get(value, value if value in CLASSICAL + MONOAMINES else "unknown")


def verified_nt(verified, predicted):
    if pd.isna(verified) or not str(verified).strip():
        return canonical_nt(predicted)
    parts = [canonical_nt(v.strip()) for v in str(verified).split(",")]
    return next((v for v in parts if v in CLASSICAL),
                next((v for v in parts if v in MONOAMINES), "unknown"))


def pairs(path):
    """Sum neuropils BEFORE the compiler's min_weight filter. IDs never pass through float."""
    w = pd.read_csv(path, usecols=["pre_root_id", "post_root_id", "syn_count"],
                    dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int64"})
    w = w.rename(columns={"pre_root_id": "body_pre", "post_root_id": "body_post", "syn_count": "weight"})
    return w.groupby(["body_pre", "body_post"], as_index=False, sort=False).weight.sum()


def finish(n, dataset, release, paths, **metadata):
    from ..connectome import KEEP_COLS
    n["type"], coverage = normalize_types(n.flywireType)
    for col in KEEP_COLS:
        if col not in n:
            n[col] = np.nan
    n["bodyId"] = n.bodyId.astype(np.int64)
    if n.bodyId.duplicated().any():
        raise ValueError(f"{dataset} has duplicate body IDs")
    n["status"] = "Traced"
    n["instance"] = n.type.fillna("unknown") + "_" + n.somaSide.fillna("?")
    n.attrs["manifest"] = dict(dataset=dataset, release=release,
        sources=[{"path": p.name, "sha256": sha256(p)} for p in paths],
        alias_sha256=sha256(ALIASES), aliases=coverage, **metadata)
    return n
