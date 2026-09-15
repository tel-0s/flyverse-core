"""BANC v888: female brain and VNC. No optic column annotations."""
import numpy as np
import pandas as pd

from .common import SIDES, finish, pairs, verified_nt

FILES = ("neurons.csv.gz", "connections_princeton.csv.gz")
NT_RULE = "verified-first; first classical else first monoamine; unsupported verified NT -> unknown; photoreceptors histamine"
DROP_CLASSES = ("glia", "not_a_neuron", "trachea")
SUPERCLASS = {"optic_lobe_intrinsic": "ol_intrinsic", "central_brain_intrinsic": "cb_intrinsic",
    "ventral_nerve_cord_intrinsic": "vnc_intrinsic", "ascending": "ascending_neuron",
    "descending": "descending_neuron", "visual_projection": "visual_projection",
    "visual_centrifugal": "visual_centrifugal", "sensory_ascending": "sensory_ascending",
    "sensory_descending": "sensory_descending", "ascending_visceral_circulatory": "vnc_endocrine",
    "sensory": "cb_sensory", "motor": "cb_motor", "visceral_circulatory": "cb_endocrine"}
CLASS = {"olfactory_receptor_neuron": "olfactory", "antennal_lobe_projection_neuron": "ALPN",
    "antennal_lobe_local_neuron": "ALLN", "kenyon_cell": "Kenyon_Cell",
    "mushroom_body_output_neuron": "MBON", "taste_bristle_gustatory_neuron": "gustatory",
    "taste_peg_gustatory_neuron": "gustatory", "gustatory_neuron": "gustatory",
    "bristle_neuron": "mechanosensory", "chordotonal_organ_neuron": "mechanosensory",
    "hair_plate_neuron": "mechanosensory_proprioceptive",
    "campaniform_sensillum_neuron": "mechanosensory_proprioceptive"}
NERVES = {"antennal_nerve": "AN", "maxillary-labial_nerve": "MxLbN", "eye_nerve": "EyeN",
    "ocellar_nerve": "OCN", "occipital_nerve": "ON", "pharyngeal_nerve": "PhN",
    "accessory_pharyngeal_nerve": "aPhN", "corpus_cardiacum_nerve": "NCC", "stomodeal_nerve": "SN",
    "pharyngeal_nerve_or_accessory_pharyngeal_nerve": "PhN/aPhN",
    "prothoracic_leg_nerve": "ProLN", "mesothoracic_leg_nerve": "MesoLN", "metathoracic_leg_nerve": "MetaLN",
    "prothoracic_accessory_nerve": "ProAN", "ventral_prothoracic_nerve": "VProN",
    "dorsal_prothoracic_nerve": "DProN", "prothoracic_chordotonal_nerve": "ProCN",
    "anterior_dorsal_mesothoracic_nerve": "ADMN", "posterior_dorsal_mesothoracic_nerve": "PDMN",
    "mesothoracic_accessory_nerve": "MesoAN", "dorsal_metathoracic_nerve": "DMetaN",
    "abdominal_nerve_trunk": "AbN", "first_abdominal_nerve": "AbN1", "second_abdominal_nerve": "AbN2",
    "third_abdominal_nerve": "AbN3", "fourth_abdominal_nerve": "AbN4", "prosternal_nerve": "ProSN",
    "cervical_nerve": "CV", "ventral_cervical_nerve": "VCV"}
BRAIN_NERVES = {"AN", "MxLbN", "EyeN", "OCN", "ON", "PhN", "aPhN", "NCC", "SN", "PhN/aPhN"}
VNC_PARTS = {"front_leg", "middle_leg", "hind_leg", "haltere", "wing", "wing_base", "wing_margin",
    "wing_tegula", "abdomen", "abdominal_wall", "thorax", "thoracic_abdominal", "reproductive_tract", "uterus"}
MOTOR_SUBCLASS = {"front_leg_motor_neuron": "fl", "middle_leg_motor_neuron": "ml", "hind_leg_motor_neuron": "hl",
    "wing_power_motor_neuron": "wm", "wing_steering_motor_neuron": "wm", "wing_tension_motor_neuron": "wm",
    "haltere_steering_neuron": "hm", "haltere_power_neuron": "hm"}
MOTOR_PART = {"front_leg": "fl", "middle_leg": "ml", "hind_leg": "hl", "wing": "wm", "haltere": "hm"}
PROPRIO_CLASS = {"chordotonal_organ_neuron": "chordotonal organ", "hair_plate_neuron": "hair plate",
                 "campaniform_sensillum_neuron": "campaniform sensilla"}


def nerve(value):
    if pd.isna(value):
        return np.nan, np.nan, False
    codes, sides = [], set()
    for part in value.split("+"):
        for prefix, side in (("left_", "L"), ("right_", "R")):
            if part.startswith(prefix):
                sides.add(side)
                part = part[len(prefix):]
                break
        codes.append(NERVES.get(part, part))
    return "+".join(dict.fromkeys(codes)), next(iter(sides)) if len(sides) == 1 else np.nan, any(c not in BRAIN_NERVES for c in codes)


def adapt(d):
    d = d[~d["Super Class"].isin(DROP_CLASSES)].reset_index(drop=True)
    nv = [nerve(v) for v in d.Nerve]
    code = pd.Series([v[0] for v in nv]); ns = pd.Series([v[1] for v in nv])
    parts = d["Body Part"].fillna("").map(lambda s: set(s.split(",")))
    vnc = pd.Series([v[2] for v in nv]) | parts.map(lambda s: bool(s & VNC_PARTS))
    n = pd.DataFrame({"bodyId": d["Root ID"], "flywireType": d["Primary Cell Type"],
        "hemibrainType": d["Alternative Cell Type(s)"], "mancType": d["Alternative Cell Type(s)"],
        "somaSide": d["Soma side"].map(SIDES).fillna(ns), "superclass": d["Super Class"].map(SUPERCLASS),
        "class": d.Class.map(CLASS).fillna(d.Class), "subclass": d["Sub Class"],
        "entryNerve": code.where(d["Super Class"].isin(["sensory", "sensory_ascending", "sensory_descending"])),
        "exitNerve": code.where(d["Super Class"].isin(["motor", "visceral_circulatory", "ascending_visceral_circulatory"])),
        "nt": [verified_nt(v, p) for v, p in zip(d["Verified NT type"], d["Predicted NT type"])],
        "nt_verified": d["Verified NT type"]})
    for src, dst in (("sensory", "vnc_sensory"), ("motor", "vnc_motor"), ("visceral_circulatory", "vnc_endocrine")):
        n.loc[vnc & d["Super Class"].eq(src), "superclass"] = dst
    pr = d.Class.eq("photoreceptor_neuron") | d["Sub Class"].eq("retina_photoreceptor_neuron")
    n.loc[pr | parts.map(lambda s: bool(s & {"retina", "interommatidial"})), "superclass"] = "ol_sensory"
    n.loc[pr, "nt"] = "histamine"
    sensory = d["Super Class"].isin(["sensory", "sensory_ascending", "sensory_descending"])
    prop = sensory & vnc & (d.Class.isin(PROPRIO_CLASS) | d.Function.fillna("").map(lambda s: "proprioception" in s.split(",")))
    n.loc[prop, "class"] = "mechanosensory_proprioceptive"
    n.loc[prop, "subclass"] = d.loc[prop, "Class"].map(PROPRIO_CLASS).fillna("leg")
    n.loc[prop & parts.map(lambda s: "haltere" in s), "subclass"] = "haltere"
    motor = d["Super Class"].eq("motor")
    n.loc[motor, "subclass"] = d.loc[motor, "Sub Class"].map(MOTOR_SUBCLASS).fillna(d.loc[motor, "Body Part"].map(MOTOR_PART))
    return n


def read(data_dir, *, edges="threshold", nt_threshold=0.5, log=print):
    d = pd.read_csv(data_dir / FILES[0], low_memory=False)
    n = finish(adapt(d), "banc", "v888", [data_dir / f for f in FILES], pair_threshold=3, edges=edges,
               nt_rule=NT_RULE, excluded_rows=int(d["Super Class"].isin(DROP_CLASSES).sum()),
               unclassified_rows=int(d["Super Class"].isna().sum()))
    return n, pairs(data_dir / FILES[1])
