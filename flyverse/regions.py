"""Named, exhaustive modules and directed reachability on a Connectome (W[post, pre])."""
from __future__ import annotations

import numpy as np

from .connectome import Connectome, PHOTORECEPTOR_TYPES

MODULES = ("optic", "visual_projection", "antennal_lobe", "mushroom_body", "gustatory",
           "mechanosensory", "central", "descending", "vnc")
SWEET_TYPES = ("GNG232", "GNG132", "GNG175", "GNG229", "GNG215", "GNG108",
               "DNge173", "DNge174", "ANXXX462a", "DNg67")


def labels(c: Connectome) -> np.ndarray:
    """One module per neuron; unclassified cells remain in central, never silently dropped."""
    n = c.neurons
    sc, cl, ty = (n[k].fillna("") for k in ("superclass", "class", "type"))
    out = np.full(c.n, "central", dtype=object)
    out[(sc.str.startswith("vnc_") | sc.str.contains("ascending")).to_numpy()] = "vnc"
    out[(cl.isin(["olfactory", "ALLN", "ALPN", "ALIN", "ALON"]) |
         ty.str.match(r"^(ORN_|lLN|v2LN|v3LN|il3LN|l2LN|vLN)")).to_numpy()] = "antennal_lobe"
    out[(cl.isin(["Kenyon_Cell", "MBON", "DAN"]) | ty.str.match(r"^(KC|MBON|APL$|DPM$)")).to_numpy()] = "mushroom_body"
    out[(cl.str.startswith("mechanosensory") | ty.str.match(r"^(JO-|AMMC|WED)")).to_numpy()] = "mechanosensory"
    out[(cl.eq("gustatory") | ty.isin(SWEET_TYPES) | sc.eq("cb_motor")).to_numpy()] = "gustatory"
    out[sc.eq("descending_neuron").to_numpy()] = "descending"
    out[sc.isin(["visual_projection", "visual_projection_tbc", "visual_centrifugal"]).to_numpy()] = "visual_projection"
    out[(sc.eq("ol_intrinsic") | ty.isin(PHOTORECEPTOR_TYPES)).to_numpy()] = "optic"
    if "dataset" in n:
        synthetic = n.dataset.eq("synthetic").to_numpy()
        out[synthetic] = sc.replace("", "synthetic").to_numpy()[synthetic]
    return out


def select(c: Connectome, modules=None) -> np.ndarray:
    if modules is None:
        return np.arange(c.n)
    modules = [modules] if isinstance(modules, str) else list(modules)
    available = set(MODULES) | set(labels(c))
    unknown = set(modules) - available
    if unknown:
        raise ValueError(f"unknown modules: {sorted(unknown)}; choose from {sorted(available)}")
    return np.flatnonzero(np.isin(labels(c), modules))


def subset(c: Connectome, modules=None) -> Connectome:
    return c if modules is None else c.subset(select(c, modules))


def pare(c: Connectome, sources, sinks, max_hops: int | None = None) -> Connectome:
    """Keep neurons on directed source-to-sink walks, optionally of at most max_hops edges.

    Selectors are row indices, boolean masks or select() dictionaries. Zero-weight contacts
    are excluded. With cycles, reachability means walks, not enumeration of simple paths.
    """
    if max_hops is not None and (not isinstance(max_hops, int) or max_hops < 0):
        raise ValueError("max_hops must be a nonnegative integer or None")
    adjacency = c.W.copy()
    adjacency.eliminate_zeros()

    def distances(graph, start):
        dist = np.full(c.n, np.inf)
        front = c.indices(start)
        dist[front] = 0
        depth = 0
        while front.size and (max_hops is None or depth < max_hops):
            reached = np.unique(graph[front].indices)
            front = reached[np.isinf(dist[reached])]
            depth += 1
            dist[front] = depth
        return dist

    forward = distances(adjacency.T.tocsr(), sources)
    backward = distances(adjacency.tocsr(), sinks)
    keep = np.isfinite(forward) & np.isfinite(backward)
    if max_hops is not None:
        keep &= forward + backward <= max_hops
    return c.subset(keep)
