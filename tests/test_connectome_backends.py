"""Release contracts on tiny tables; no external data or accelerator required."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyverse import connectome as cn
from flyverse.backends import banc, fafb
from flyverse.backends.common import normalize_types, verified_nt
from flyverse.interp import common


@pytest.fixture
def tmp_path():
    # Avoid the shared pytest numbered directory used by concurrent local agents.
    with tempfile.TemporaryDirectory(prefix="flyverse-backends-") as path:
        yield Path(path)


def female_files(path):
    ids = np.array([720575940000000001, 720575940000000002, 720575940000000003], dtype=np.int64)
    pd.DataFrame(dict(root_id=ids, group=["a", "b", "c"], nt_type=["ACH", "DA", "GLUT"],
                      nt_type_score=[0.9, 0.9, 0.2])).to_csv(path / "neurons.csv.gz", index=False)
    pd.DataFrame(dict(root_id=ids, side=["left"] * 3, super_class=["central"] * 3,
                      **{"class": ["CX"] * 3}, sub_class=[""] * 3, nerve=[""] * 3)).to_csv(path / "classification.csv.gz", index=False)
    pd.DataFrame(dict(root_id=ids, primary_type=["PS196a", "PEN_a/PEN1", "unmapped"])).to_csv(path / "consolidated_cell_types.csv.gz", index=False)
    pd.DataFrame(columns=["root_id", "hemisphere", "type", "column_id", "x", "y", "p", "q"]).to_csv(path / "column_assignment.csv.gz", index=False)
    pd.DataFrame(dict(pre_root_id=[ids[0], ids[0], ids[1], ids[2]], post_root_id=[ids[1], ids[1], ids[2], ids[0]],
                      syn_count=[2, 2, 7, 1], neuropil=["A", "B", "A", "A"])).to_csv(path / "connections_princeton.csv.gz", index=False)
    return ids


def test_pair_sum_orientation_explicit_zeros_and_no_male_override(tmp_path):
    ids = female_files(tmp_path)
    c = cn.compile_connectome(dataset="fafb", data_dir=tmp_path, min_weight=3, verbose=False)
    assert c.neurons.bodyId.tolist() == ids.tolist()
    assert c.neurons.type.tolist() == ["PS196_a", "PEN_a(PEN1)", "unmapped"]
    assert c.W[1, 0] == 4 and c.W[0, 1] == 0 and c.W.nnz == 2
    assert c.W.data.tolist() == [4, 0]
    assert cn.sign0_counts(c).tolist() == [0, 7]
    assert c.neurons.nt.tolist() == ["acetylcholine", "dopamine", "unknown"]
    with pytest.raises(ValueError, match="overrides"):
        cn.compile_connectome(dataset="fafb", data_dir=tmp_path, type_nt_override={"unmapped": "gaba"})


def test_cache_subset_extension_namespace_and_counts(tmp_path):
    source = tmp_path / "source"; source.mkdir(); female_files(source)
    c = cn.load(tmp_path / "cache", dataset="fafb", data_dir=source, verbose=False)
    loaded = cn.load(tmp_path / "cache", verbose=False)
    assert (loaded.dataset, loaded.release) == ("fafb", "v783")
    assert np.array_equal(cn.sign0_counts(c), cn.sign0_counts(loaded))
    sub = loaded.subset([2, 1]).subset([1, 0])
    cn.save(sub, tmp_path / "subset")
    sub = cn.load(tmp_path / "subset", verbose=False)
    assert sub.reference.n == 3 and sub.dataset == "fafb"
    assert cn.sign0_counts(sub).tolist() == [7]
    new_id = -1
    ex = loaded.extend(pd.DataFrame([dict(bodyId=new_id, nt="tyramine")]),
                       pd.DataFrame([dict(body_pre=new_id, body_post=int(loaded.neurons.bodyId[0]), weight=9)]),
                       cache_dir=tmp_path / "extension")
    ex = cn.load(tmp_path / "extension", verbose=False)
    assert ex.dataset == "fafb" and cn.sign0_counts(ex).sum() == 17  # 7 dopamine + 1 unknown + 9 synthetic
    assert ex.neurons.dataset.iloc[0] == "fafb"
    prov = common.provenance(ex, device="cpu")
    assert prov["dataset_release"]["name"] == "fafb"
    assert prov["model"]["dataset"] == "fafb"
    assert prov["compiled_connectome"]["type_nt_override"] == {}
    with pytest.raises(ValueError, match="belongs"):
        cn.load(tmp_path / "cache", dataset="banc", rebuild=True)
    with pytest.raises(ValueError, match="cached"):
        cn.load(tmp_path / "cache", dataset="fafb", nt_threshold=0.8)
    with pytest.raises(ValueError, match="overrides"):
        cn.load(tmp_path / "cache", dataset="fafb", type_nt_override={"x": "gaba"})
    assert (tmp_path / "subset" / "nt_scores.parquet").exists()


def test_save_rejects_cross_dataset_before_writing(tmp_path):
    female_files(tmp_path)
    female = cn.compile_connectome(dataset="fafb", data_dir=tmp_path, verbose=False)
    male = cn.Connectome(female.neurons.copy(), female.W.copy(), female.body_to_index.copy())
    path = tmp_path / "male"
    cn.save(male, path)
    old = (path / "neurons.parquet").read_bytes()
    with pytest.raises(ValueError, match="another dataset"):
        cn.save(female, path)
    with pytest.raises(ValueError, match="another dataset"):
        cn.build_sign0_counts(female, path)
    assert (path / "neurons.parquet").read_bytes() == old
    cn.save(female, tmp_path / "female")
    with pytest.raises(ValueError, match="another dataset"):
        cn.save(male, tmp_path / "female")


def test_dataset_variant_directories_and_tyramine_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYVERSE_CACHE", str(tmp_path))
    assert cn.default_cache_directory("fafb") == tmp_path / "fafb"
    assert cn.default_cache_directory("fafb", "no_threshold") == tmp_path / "fafb" / "no_threshold"
    female_files(tmp_path)
    c = cn.compile_connectome(dataset="fafb", data_dir=tmp_path, verbose=False)
    c.neurons.loc[1, "nt"] = "tyramine"
    r = cn.receptor_signs(c, with_counts=True)
    assert r.fast_sign[c.W.indices == 1].tolist() == [0]
    assert r.slow_sign[c.W.indices == 1].tolist() == [0]
    assert cn.sign0_counts(c).sum() == 8


@pytest.mark.parametrize("v,p,expected", [
    (None, "TYR", "tyramine"), ("glutamate,serotonin", "ACH", "glutamate"),
    ("dopamine,gaba", "ACH", "gaba"), ("tyramine,serotonin", "ACH", "tyramine"),
    ("nitric_oxide", "ACH", "unknown"), ("glycine", "GABA", "unknown"),
    ("acetylcholine,histamine", "GLUT", "acetylcholine"), (None, None, "unknown")])
def test_verified_first(v, p, expected):
    assert verified_nt(v, p) == expected


def test_vocabulary_and_missing_superclass():
    rows = []
    for i, (sc, cls, sub, part, nerve) in enumerate([
        ("sensory", "chordotonal_organ_neuron", "", "front_leg", "left_prothoracic_leg_nerve"),
        ("sensory_ascending", "campaniform_sensillum_neuron", "", "haltere", "right_dorsal_metathoracic_nerve"),
        ("motor", "wing_motor_neuron", "wing_power_motor_neuron", "wing", "left_anterior_dorsal_mesothoracic_nerve"),
        ("motor", "haltere_motor_neuron", "haltere_power_neuron", "haltere", "right_dorsal_metathoracic_nerve"),
        ("motor", "leg_motor_neuron", "front_leg_motor_neuron", "front_leg", "left_prothoracic_leg_nerve"),
        (None, None, None, None, None), ("glia", None, None, None, None)]):
        rows.append({"Root ID": i, "Super Class": sc, "Class": cls, "Sub Class": sub, "Body Part": part,
            "Nerve": nerve, "Primary Cell Type": "x", "Alternative Cell Type(s)": None, "Soma side": None,
            "Function": "proprioception", "Verified NT type": None, "Predicted NT type": "ACH"})
    n = banc.adapt(pd.DataFrame(rows))
    assert len(n) == 6
    assert n.subclass[:5].tolist() == ["chordotonal organ", "haltere", "wm", "hm", "fl"]
    assert n.somaSide[:5].tolist() == ["L", "R", "L", "R", "L"]
    assert n.entryNerve[:2].tolist() == ["ProLN", "DMetaN"]
    assert n.superclass[:5].tolist() == ["vnc_sensory", "sensory_ascending", "vnc_motor", "vnc_motor", "vnc_motor"]


def test_aliases_never_choose_an_ambiguous_last_row(tmp_path):
    p = tmp_path / "aliases.csv"
    pd.DataFrame([
        ("a", "source", "flywire", "alias", ""), ("b", "source", "flywire", "alias", ""),
        ("c", "same", "flywire", "exact", ""), ("d", "same", "flywire", "alias", ""),
        ("e", "bad", "flywire", "alias", "unresolvable")],
        columns=["malecns_type", "alias", "system", "tier", "flag"]).to_csv(p, index=False)
    out, audit = normalize_types(pd.Series(["source", "same", "bad", None]), p)
    assert out[:3].tolist() == ["source", "c", "bad"]
    assert audit["ambiguous"] == {"source": ["a", "b"]}


def test_hex_axes_and_mirror():
    p, q = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
    left = fafb.hex_coordinates(p, q, "left")
    right = fafb.hex_coordinates(p, q, "right")
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left.sum(axis=1) - 38, p + q)
    np.testing.assert_array_equal(left[:, 0] - left[:, 1] + 2, q - p)
    with pytest.raises(ValueError, match="hemisphere"):
        fafb.hex_coordinates(p, q, "unknown")


def test_unavailable_capabilities_are_explicit(tmp_path):
    from flyverse.retina import build_retina
    from flyverse.senses import Proprioception
    from flyverse.motor import motor_groups, wing_groups
    female_files(tmp_path)
    c = cn.compile_connectome(dataset="fafb", data_dir=tmp_path, verbose=False)
    for fn in (Proprioception, motor_groups, wing_groups):
        with pytest.raises(cn.NotAvailable, match="no VNC"):
            fn(c)
    c.dataset = "banc"
    with pytest.raises(cn.NotAvailable, match="column"):
        build_retina(c)
    from flyverse.optic import OpticLobe
    with pytest.raises(cn.NotAvailable, match="column"):
        OpticLobe(c, None, device="cpu")
