"""CPU, synthetic anatomy: reconstruction correctness and fail-closed reporting."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyverse import connectome as cn
from scripts.recover_banc_columns import (
    assign_cells,
    distances,
    dra_check,
    gate,
    orient,
    reconstruct,
    t4_offsets,
)


def ideal_tiling():
    h = np.array(
        [
            [a, b]
            for a in range(-3, 4)
            for b in range(-3, 4)
            if max(abs(a), abs(b), abs(a - b)) <= 3
        ]
    )
    i, j = np.nonzero(np.triu(distances(h[:, None] - h[None, :]) == 1, 1))
    # Each synthetic shared partner contacts exactly two neighbouring columns.
    inputs = np.zeros((len(i), len(h)))
    inputs[np.arange(len(i)), i] = 1
    inputs[np.arange(len(i)), j] = 1
    return h, inputs


@pytest.mark.parametrize("permute", [False, True])
def test_ideal_lattice_preserves_every_pairwise_distance(permute):
    h, inputs = ideal_tiling()
    if permute:
        order = np.random.default_rng(42).permutation(len(h))
        h, inputs = h[order], inputs[:, order]
    result = reconstruct(inputs)
    assert len(result["keep"]) == len(h)
    actual = result["hex"]
    expected = h[result["keep"]]
    np.testing.assert_allclose(
        distances(actual[:, None] - actual[None, :]),
        distances(expected[:, None] - expected[None, :]),
    )
    assert result["diagnostics"]["rounded_collisions"] == 0
    assert result["diagnostics"]["injectivity_displacements"] == 0


def test_orphan_has_no_invented_position():
    h, inputs = ideal_tiling()
    result = reconstruct(np.pad(inputs, ((0, 0), (0, 1))))
    np.testing.assert_array_equal(result["keep"], np.arange(len(h)))
    assert result["diagnostics"]["component_sizes"] == [len(h), 1]
    with pytest.raises(ValueError, match="no connected component"):
        reconstruct(np.zeros((4, 5)))


def graph():
    n = pd.DataFrame(
        [
            {"bodyId": 1, "type": "Mi1", "somaSide": "R", "nt": "acetylcholine"},
            {"bodyId": 2, "type": "Mi1", "somaSide": "R", "nt": "acetylcholine"},
            {"bodyId": 3, "type": "L1", "somaSide": "R", "nt": "glutamate"},
            {"bodyId": 4, "type": "L1", "somaSide": "R", "nt": "glutamate"},
            {"bodyId": 5, "type": "R7_unclear", "somaSide": "R", "nt": "histamine"},
            {"bodyId": 6, "type": "R7_unclear", "somaSide": "R", "nt": "histamine"},
            {"bodyId": 7, "type": "Dm-DRA1", "somaSide": "R", "nt": "gaba"},
        ]
    )
    n["hex1"], n["hex2"] = np.nan, np.nan
    # L1 body 3 -> Mi1 body 2 is inhibitory; absolute counts must still match.
    # Both R7s project to Dm-DRA1; only one will have a mapped test column.
    w = sp.csr_matrix(
        ([-7.0, -5.0, -3.0], ([1, 6, 6], [2, 4, 5])),
        shape=(len(n), len(n)),
        dtype=np.float32,
    )
    return cn.Connectome(
        n,
        w,
        pd.Series(np.arange(len(n)), index=n.bodyId),
        dataset="banc",
        release="v888",
    )


def test_cell_assignment_keeps_zero_matches_unassigned_and_graph_immutable():
    c = graph()
    before, weights = c.neurons.copy(deep=True), c.W.copy()
    table, _ = assign_cells(
        c,
        "R",
        np.array([0, 1]),
        {"keep": np.array([0, 1]), "hex": np.array([[0, 0], [1, 1]])},
    )
    cells = table.set_index("bodyId")
    assert cells.loc[3, "seed_bodyId"] == "2"
    assert cells.loc[4, "hex_source"] == "unassigned"
    assert np.isnan(cells.loc[4, "hex1"])
    assert not c.has_optic_columns
    pd.testing.assert_frame_equal(c.neurons, before)
    np.testing.assert_array_equal(c.W.data, weights.data)
    np.testing.assert_array_equal(c.W.indices, weights.indices)
    np.testing.assert_array_equal(c.W.indptr, weights.indptr)


def test_dra_gate_keeps_unmapped_photoreceptor_in_denominator():
    c = graph()
    table = pd.DataFrame(
        {
            "bodyId": [1, 2, 5, 6],
            "type": ["Mi1", "Mi1", "R7_unclear", "R7_unclear"],
            "hex1": [0.0, 1.0, 1.0, np.nan],
            "hex2": [0.0, 1.0, 1.0, np.nan],
        }
    )
    check = dra_check(c, "R", table)
    assert check["n_candidates"] == 2
    assert check["mapped"] == check["within_two_rows"] == 1
    assert check["status"] == "fail"
    assert dra_check(c, "L", table)["status"] == "unavailable"


def test_no_t4_evidence_is_unavailable_not_a_pass():
    c = graph()
    table = c.neurons[["bodyId", "type", "hex1", "hex2"]].copy()
    assert all(row["offset"] is None for row in t4_offsets(c, "R", table).values())
    _, orientation = orient(c, c, "R", table)
    assert orientation["status"] == "unavailable"


def test_integration_requires_all_four_gates():
    names = ("i_dra_rim", "ii_lr_mirror", "iii_t4_direction", "iv_column_count")
    checks = {name: {"status": "pass"} for name in names}
    assert gate(checks)
    for name in names:
        for state in ("fail", "unavailable", "measured"):
            assert not gate({**checks, name: {"status": state}})
        assert not gate({key: value for key, value in checks.items() if key != name})


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_invalid_counts_rejected(value):
    _, inputs = ideal_tiling()
    inputs[0, 0] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        reconstruct(inputs)
