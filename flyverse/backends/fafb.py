"""FAFB v783: complete female brain, both optic lobes, no VNC."""
import numpy as np
import pandas as pd

from .common import SIDES, canonical_nt, finish, pairs

SUPERCLASS = {"optic": "ol_intrinsic", "central": "cb_intrinsic", "sensory": "cb_sensory",
    "visual_projection": "visual_projection", "visual_centrifugal": "visual_centrifugal",
    "ascending": "ascending_neuron", "descending": "descending_neuron",
    "sensory_ascending": "sensory_ascending", "motor": "cb_motor", "endocrine": "cb_endocrine"}
FILES = ("neurons.csv.gz", "classification.csv.gz", "consolidated_cell_types.csv.gz", "column_assignment.csv.gz")
NT_RULE = "nt_type when nt_type_score >= nt_threshold; photoreceptors histamine; no overrides"
# The published p/q grid has dorsal = p+q and anterior = q-p in both hemispheres.
# Translations keep positive indices; retina centres each eye independently.
HEX_TRANSFORMS = {"left": ((0, 1), (1, 0), (18, 20)), "right": ((0, 1), (1, 0), (18, 20))}


def hex_coordinates(p, q, hemisphere):
    p, q, hemisphere = np.broadcast_arrays(p, q, hemisphere)
    if not np.isin(hemisphere, list(HEX_TRANSFORMS)).all():
        raise ValueError("column hemisphere must be left or right")
    out = np.empty(p.shape + (2,), dtype=np.float64)
    for side, (row1, row2, shift) in HEX_TRANSFORMS.items():
        m = hemisphere == side
        out[m, 0] = row1[0] * p[m] + row1[1] * q[m] + shift[0]
        out[m, 1] = row2[0] * p[m] + row2[1] * q[m] + shift[1]
    return out


def read(data_dir, *, edges="threshold", nt_threshold=0.5, log=print):
    raw = pd.read_csv(data_dir / FILES[0])
    cls = pd.read_csv(data_dir / FILES[1])
    types = pd.read_csv(data_dir / FILES[2])
    cols = pd.read_csv(data_dir / FILES[3])
    d = raw.merge(cls, on="root_id", validate="one_to_one").merge(types, on="root_id", how="left", validate="one_to_one")
    n = pd.DataFrame({"bodyId": d.root_id, "flywireType": d.primary_type, "somaSide": d.side.map(SIDES),
        "superclass": d.super_class.map(SUPERCLASS), "class": d["class"], "subclass": d.sub_class,
        "entryNerve": d.nerve, "exitNerve": d.nerve.where(d.super_class.isin(["motor", "endocrine"])),
        "nt": d.nt_type.where(d.nt_type_score >= nt_threshold).map(canonical_nt)})
    pr = d.sub_class.eq("photo_receptor") | d.primary_type.isin(["R1-6", "R7", "R8"])
    n.loc[pr | d.sub_class.eq("eye_bristle"), "superclass"] = "ol_sensory"
    n.loc[pr, "nt"] = "histamine"
    xy = hex_coordinates(cols.p.to_numpy(), cols.q.to_numpy(), cols.hemisphere.to_numpy())
    cols["assignedOlHex1"], cols["assignedOlHex2"] = xy[:, 0], xy[:, 1]
    n = n.merge(cols[["root_id", "assignedOlHex1", "assignedOlHex2"]], left_on="bodyId", right_on="root_id", how="left", validate="one_to_one").drop(columns="root_id")
    # Column hemisphere is the eye, including cells with a contralateral soma.
    col_side = pd.Series(cols.hemisphere.map(SIDES).to_numpy(), index=cols.root_id)
    n["hex_side"] = n.bodyId.map(col_side)
    edgefile = "connections_princeton_no_threshold.csv.gz" if edges == "no_threshold" else "connections_princeton.csv.gz"
    n = finish(n, "fafb", "v783", [data_dir / f for f in (*FILES, edgefile)],
               pair_threshold=1 if edges == "no_threshold" else 5, edges=edges, nt_rule=NT_RULE,
               nt_threshold=nt_threshold, hex_transforms=HEX_TRANSFORMS)
    n.attrs["nt_scores"] = raw.rename(columns={"root_id": "bodyId"}).drop(columns="group")
    return n, pairs(data_dir / edgefile)
