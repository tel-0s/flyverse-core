"""The candidate opt-in must never alter the biological graph or hide its uncertainty."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import banc_vision as bv
from flyverse import connectome as cn
from flyverse import retina
from flyverse.interp import common


@pytest.fixture
def candidate(monkeypatch):
    # One complete cartridge, one with only L2, plus a left-eye biological cell
    # and an explicit sign-zero edge that must survive extension byte-for-byte.
    n = pd.DataFrame(
        {
            "bodyId": np.arange(1, 9, dtype=np.int64),
            "type": ["Mi1", "L1", "L2", "L3", "Mi1", "L2", "L1", "DNa02"],
            "somaSide": ["R"] * 6 + ["L", "R"],
            "nt": ["acetylcholine"] * 7 + ["unknown"],
            "superclass": ["ol_intrinsic"] * 7 + ["descending"],
            "sign": [1.0] * 7 + [0.0],
            "hex1": np.nan,
            "hex2": np.nan,
            "hex_side": np.nan,
            "hex_source": np.nan,
        }
    )
    w = sp.csr_matrix(
        (np.array([3.0, 0.0, 5.0], np.float32), ([0, 1, 6], [1, 7, 2])), shape=(8, 8)
    )
    c = cn.Connectome(
        n,
        w,
        pd.Series(np.arange(8), index=n.bodyId),
        dataset="banc",
        release="v888",
        _sign0={
            "key": np.array([15]),
            "count": np.array([7.0], np.float32),
            "n": np.array(8),
        },
    )
    columns = n.iloc[:6][["bodyId", "type"]].copy()
    columns["hex1"], columns["hex2"], columns["side"] = [0, 0, 0, 0, 1, 1], 0, "R"
    _, meta = bv.read_candidate()
    meta.update(
        base_csr_md5=common.connectome_fingerprint(c)["md5"],
        base_annotations_sha256=bv.annotation_sha256(c),
    )
    monkeypatch.setattr(bv, "read_candidate", lambda: (columns, meta))
    return c


def identical_csr(a, b):
    assert a.shape == b.shape
    for key in ("data", "indices", "indptr"):
        assert getattr(a, key).tobytes() == getattr(b, key).tobytes()


def test_candidate_cartridges_roundtrip_prune_and_provenance(candidate, tmp_path):
    c = candidate
    before = c.neurons.copy(deep=True)
    x = bv.extend_candidate(c, cache_dir=tmp_path / "candidate")
    assert not c.has_optic_columns and x.has_optic_columns
    pd.testing.assert_frame_equal(c.neurons, before)
    identical_csr(x.W[: c.n, : c.n].tocsr(), c.W)
    synth = x.neurons[x.neurons.bodyId < 0]
    assert len(synth) == 12 and synth.dataset.eq("synthetic").all()
    assert synth.nt.eq("histamine").all() and synth.hex_side.eq("R").all()
    assert np.isnan(x.neurons.loc[6, "hex1"])
    assert x.W[: c.n, c.n :].nnz == 24
    coo = x.W[: c.n, c.n :].tocoo()
    assert (coo.data < 0).all()
    assert (
        x.neurons.hex1.to_numpy()[coo.row].tolist()
        == synth.hex1.to_numpy()[coo.col].tolist()
    )
    r = retina.build_retina(x)
    assert r.n_columns == 2 and set(r.col_side) == {"R"}
    y = cn.load(x.cache_dir, verbose=False)
    identical_csr(y.W, x.W)
    np.testing.assert_array_equal(cn.sign0_counts(y)[:2], cn.sign0_counts(x)[:2])
    restored = y.prune(y.neurons.bodyId.lt(0).to_numpy())
    assert not restored.has_optic_columns
    pd.testing.assert_frame_equal(restored.neurons, before)
    identical_csr(restored.W, c.W)
    for graph in (x, y, y.subset(np.arange(9))):
        p = common.provenance(graph, device="cpu")
        assert p["model"]["vision"]["qualification"] == bv.QUALIFICATION
        assert p["model"]["vision"]["accuracy_estimate"]["exact_fraction"] == [
            0.8,
            0.85,
        ]
        assert p["model"]["vision"]["checks"]["i_dra_rim"]["status"] == "fail"
    assert "vision" not in common.provenance(c, device="cpu")["model"]


def test_candidate_guards(candidate, tmp_path):
    with pytest.raises(ValueError, match="full, unextended"):
        bv.extend_candidate(candidate.subset([0, 1]), cache_dir=tmp_path / "subset")
    candidate._cache_dir = tmp_path / "base"
    with pytest.raises(ValueError, match="separate"):
        bv.extend_candidate(candidate, cache_dir=candidate.cache_dir / "nested")
    candidate.neurons.loc[0, "type"] = "changed"
    with pytest.raises(ValueError, match="different biological"):
        bv.extend_candidate(candidate, cache_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="vision must"):
        cn.load(vision="validated")
    with pytest.raises(ValueError, match="requires"):
        cn.load(vision_cache_dir=tmp_path / "orphan")


def test_pinned_asset():
    columns, meta = bv.read_candidate()
    assert columns.side.eq("R").all()
    assert len(columns[columns.type.eq("Mi1")].dropna(subset=["hex1", "hex2"])) == 877
    assert len(meta["right_eye"]["unassigned_seed_body_ids"]) == 1


def test_right_eye_selection_preserves_directions():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from probe_vision_common import select_eye

    r = retina.Retina(
        np.array([8, 9, 10]),
        np.array([0, 1, 2]),
        np.ones((3, 4)),
        np.array(["L", "R", "R"]),
        np.array([[0, 0], [0, 0], [1, 0]]),
        np.eye(3),
        np.array([[55, 0], [-55, 0], [-59, 4]]),
    )
    selected = select_eye(r, "right")
    assert selected.pr_index.tolist() == [9, 10]
    assert selected.pr_column.tolist() == [0, 1]
    np.testing.assert_array_equal(selected.col_az_el, r.col_az_el[1:])
