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


def test_release_capabilities_validate_names_and_survive_subsets(tmp_path):
    from dataclasses import replace
    female_files(tmp_path)
    c = cn.compile_connectome(dataset="fafb", data_dir=tmp_path, verbose=False)
    assert c.has_optic_columns and not c.has_vnc
    assert c.subset([0]).has_optic_columns   # release availability, not a census of selected cells
    with pytest.raises(ValueError, match="unknown dataset"):
        replace(c, dataset="unregistered")
    with pytest.raises(ValueError, match="unknown connectome capability"):
        c.require("typo")
    c.dataset = "unregistered"
    for attr in ("has_vnc", "has_optic_columns"):
        with pytest.raises(ValueError, match="unknown dataset"):
            getattr(c, attr)


def test_fafb_nerve_roles_are_distinct(tmp_path):
    female_files(tmp_path)
    p = tmp_path / "classification.csv.gz"
    cls = pd.read_csv(p)
    cls["super_class"] = ["central", "motor", "sensory_ascending"]
    cls["nerve"] = ["test_central", "test_motor", "test_sensory"]
    cls.to_csv(p, index=False)
    n, _ = fafb.read(tmp_path)
    assert n.entryNerve.isna().tolist() == [True, True, False]
    assert n.exitNerve.isna().tolist() == [True, False, True]
    assert n.entryNerve.iloc[2] == "test_sensory" and n.exitNerve.iloc[1] == "test_motor"


def test_retina_reports_missing_photoreceptors_in_summary_and_provenance(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from scipy import sparse
    from flyverse.retina import build_retina, summarize
    female_files(tmp_path)
    base = cn.compile_connectome(dataset="fafb", data_dir=tmp_path, verbose=False)
    n = base.neurons.iloc[[0, 0, 1, 2]].copy().reset_index(drop=True)
    n["bodyId"] = np.arange(1, 5, dtype=np.int64)
    n["type"] = ["R7_unclear", "R7_unclear", "Mi1", "Mi1"]
    n["hex1"], n["hex2"] = [1, 1, 2, 1], [1, 1, 1, 1]
    n["hex_side"] = ["L", "L", "L", "R"]
    c = cn.Connectome(n, sparse.csr_matrix((4, 4), dtype=np.float32),
                     pd.Series(np.arange(4), index=n.bodyId), dataset="fafb", release="v783")
    r = build_retina(c)
    coverage = r.coverage()
    assert coverage["n_columns"] == 3 and coverage["with_photoreceptors"] == 1
    assert coverage["without_photoreceptors_indices"] == [1, 2]
    assert coverage["by_side"] == {"L": {"n_columns": 2, "without_photoreceptors": 1},
                                    "R": {"n_columns": 1, "without_photoreceptors": 1}}
    assert "2 columns without photoreceptor input" in summarize(r, c)
    record = {"mode": "test"}
    p = common.provenance(c, fb=SimpleNamespace(retina=r), device="cpu", retina=record)
    assert p["retina"]["coverage"] == coverage and record == {"mode": "test"}
    empty = replace(r, pr_index=np.array([], dtype=int), pr_column=np.array([]), pr_sens=np.empty((0, 4)),
                    col_side=r.col_side[:2], col_hex=r.col_hex[:2], col_dir=r.col_dir[:2], col_az_el=r.col_az_el[:2])
    assert empty.coverage()["without_photoreceptors"] == 2
    assert "R: no columns" in summarize(empty, c)


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


def wing_motor_graph(types):
    """A minimal VNC graph whose only cells are wing motor neurons carrying `types`."""
    import scipy.sparse as sp
    n = pd.DataFrame(dict(bodyId=np.arange(10, 10 + len(types), dtype=np.int64), type=list(types),
                          instance=[f"{t}_L" for t in types], superclass=["vnc_motor"] * len(types),
                          **{"class": [""] * len(types)}, subclass=["wm"] * len(types),
                          somaSide=["L"] * len(types), nt=["acetylcholine"] * len(types)))
    w = sp.csr_matrix((len(n), len(n)), dtype=np.float32)
    return cn.Connectome(n, w, pd.Series(np.arange(len(n)), index=n.bodyId.to_numpy()), dataset="banc")


def test_empty_wing_group_raises_instead_of_reading_zero():
    """B1 / spec 2.3: wing motor neurons the steering (or power) selection does not recognise mean an unaliased type
    vocabulary, not an absent muscle. Naming them the MaleCNS way makes both groups resolve."""
    from flyverse.motor import wing_groups
    unaliased = wing_motor_graph(["b1", "b2", "DLM1-4", "DVM1a-c"])          # BANC v888 notation, no aliases applied
    with pytest.raises(cn.NotAvailable, match="dataset banc has no steering wing motor neurons"):
        wing_groups(unaliased)
    with pytest.raises(cn.NotAvailable, match=r"DLM1-4, DVM1a-c, b1, b2"):
        wing_groups(unaliased, allow_missing_vnc=True)
    with pytest.raises(cn.NotAvailable, match="dataset banc has no power wing motor neurons"):
        wing_groups(wing_motor_graph(["b1 MN", "b2 MN", "DLM1-4", "DVM1a-c"]))
    named = wing_groups(wing_motor_graph(["b1 MN", "b2 MN", "DLMn c-f", "DVMn 1a-c"]))
    assert (len(named.steer_L), len(named.steer_R), len(named.power)) == (2, 0, 2)
    # A graph with no wing motor neurons at all is a capability question, not a vocabulary one: it stays quiet.
    quiet = wing_groups(wing_motor_graph([]))
    assert (len(quiet.steer_L), len(quiet.power)) == (0, 0)


def test_flyverse_cache_never_moves_the_default_malecns_cache(tmp_path, monkeypatch):
    """B2: $FLYVERSE_CACHE is the parent of the non-MaleCNS caches only. The shipped MaleCNS default stays pinned to
    CACHE_DIR for load() and, symmetrically, for save()."""
    repo, env = tmp_path / "repo_cache", tmp_path / "env_cache"
    monkeypatch.setattr(cn, "CACHE_DIR", repo)
    monkeypatch.setenv("FLYVERSE_CACHE", str(env))
    assert cn.default_cache_directory() == repo and cn.default_cache_directory("malecns") == repo
    assert cn.default_cache_directory("banc") == env / "banc"
    assert cn.default_cache_directory("fafb", "no_threshold") == env / "fafb" / "no_threshold"
    female_files(tmp_path)
    f = cn.compile_connectome(dataset="fafb", data_dir=tmp_path, verbose=False)
    male = cn.Connectome(f.neurons.copy(), f.W.copy(), f.body_to_index.copy())
    cn.save(male)                                       # no cache_dir: the pinned default, never the env root
    assert (repo / "W_post_pre.npz").exists() and not (env / "W_post_pre.npz").exists()
    assert cn.load(verbose=False).cache_dir == repo.resolve()
