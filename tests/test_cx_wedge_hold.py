"""CPU tests of scripts/cx_wedge.py's `--hold-edges` flag (thread 6A, docs/audits/compass_dc_balance.md).

The hold is an `edges`-kind LABELLED COUNTERFACTUAL installed through the existing `LIFParams.type_path_gain` stage of
`brain._shaped_weights`. What is tested here: the parser, that the hold zeroes exactly the intended block of the shaped
weight matrix and leaves every other entry BIT-IDENTICAL, that the flag is off by default, and that the resolved-count
record a run writes into its JSON is the truth about what was silenced. No connectome and no GPU: a synthetic graph.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

cw = pytest.importorskip("cx_wedge")
from flyverse import brain  # noqa: E402


def synthetic():
    """6 cells: 2 ExR6, 1 ER6, 1 EPG, 1 EPGt, 1 PEN_a. Every pair wired with a distinct weight so a zeroed block is
    visible and a surviving one is checkable entry by entry."""
    types = ["ExR6", "ExR6", "ER6", "EPG", "EPGt", "PEN_a(PEN1)"]
    n = len(types)
    W = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i, j] = (i + 1) * 10 + (j + 1)          # post i, pre j
    neurons = pd.DataFrame(dict(type=types, superclass=["central"] * n, bodyId=np.arange(n), nt=["glutamate"] * n,
                                sign=[-1.0] * n, instance=[f"{t}_L1" for t in types]))
    return SimpleNamespace(neurons=neurons, W=sp.csr_matrix(W), n=n)


def shaped(c, holds=None):
    tpg = list(brain.DEFAULT_TYPE_PATH_GAIN) + [(p, q, float(f)) for p, q, f in (holds or [])]
    p = brain.LIFParams(receptor_model=None, type_path_gain=tpg)
    return np.asarray(brain._shaped_weights(c, p).todense())


# ------------------------------------------------------------------------------------------------ the parser
def test_parse_hold_edges_forms():
    assert cw.parse_hold_edges(None) == []
    assert cw.parse_hold_edges([]) == []
    out = cw.parse_hold_edges([r"^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)"])
    assert out == [(r"^(ExR6|ER6|ER4m)$", r"^(PEN_|EPG$)", 0.0)]
    assert len(cw.parse_hold_edges([r"^ExR6$:^EPG$", r"^ER6$:^PEN_"])) == 2
    for bad in ["nocolon", ":^EPG$", r"^ExR6$:", r"^ExR6($:^EPG$"]:
        with pytest.raises(SystemExit):
            cw.parse_hold_edges([bad])


def test_post_regex_excludes_EPGt_and_matches_both_PEN_types():
    """The 6A hold is onto PEN and EPG; EPGt is a different type and must NOT be held."""
    import re
    pre, post, _ = cw.parse_hold_edges([r"^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)"])[0]
    matched = [t for t in ("EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7") if re.match(post, t)]
    assert matched == ["EPG", "PEN_a(PEN1)", "PEN_b(PEN2)"]
    assert [t for t in ("ExR6", "ER6", "ER4m", "ER4d", "ExR4", "ER6b") if re.match(pre, t)] == ["ExR6", "ER6", "ER4m"]


# ------------------------------------------------------------------------------------------------ the hold itself
def test_hold_zeroes_exactly_its_block_and_nothing_else():
    c = synthetic()
    base = shaped(c)
    held = shaped(c, cw.parse_hold_edges([r"^(ExR6|ER6)$:^(PEN_|EPG$)"]))
    ty = c.neurons.type.to_numpy()
    pre = np.isin(ty, ["ExR6", "ER6"]); post = np.isin(ty, ["EPG", "PEN_a(PEN1)"])
    blk = np.outer(post, pre)
    assert np.all(held[blk] == 0.0)                                   # the class is silenced
    assert np.array_equal(held[~blk], base[~blk])                     # every other entry bit-identical
    assert np.any(base[blk] != 0.0)                                   # the block was not already empty
    # EPGt is not a post target of the hold
    epgt = ty == "EPGt"
    assert np.array_equal(held[np.outer(epgt, pre)], base[np.outer(epgt, pre)])


def test_hold_off_is_bit_identical():
    c = synthetic()
    assert np.array_equal(shaped(c), shaped(c, cw.parse_hold_edges(None)))
    assert np.array_equal(shaped(c), shaped(c, []))


def test_hold_edge_counts_reports_what_was_silenced():
    c = synthetic()
    holds = cw.parse_hold_edges([r"^(ExR6|ER6)$:^(PEN_|EPG$)"])
    rec = cw.hold_edge_counts(c, holds)
    assert len(rec) == 1
    r = rec[0]
    assert r["n_pre_cells"] == 3 and r["n_post_cells"] == 2          # 2 ExR6 + 1 ER6 -> EPG + PEN_a
    assert r["pre_types"] == ["ER6", "ExR6"] and r["post_types"] == ["EPG", "PEN_a(PEN1)"]
    assert r["n_entries"] == 6 and r["factor"] == 0.0
    ty = c.neurons.type.to_numpy()
    W = np.asarray(c.W.todense())
    want = np.abs(W[np.ix_(np.isin(ty, ["EPG", "PEN_a(PEN1)"]), np.isin(ty, ["ExR6", "ER6"]))]).sum()
    assert r["synapses"] == pytest.approx(want)
    assert cw.hold_edge_counts(c, None) == []


def test_hold_is_order_free_with_the_path_gains_it_rides_on():
    """A factor-0 hold commutes with the gE / gD path gains simulate() installs before it."""
    c = synthetic()
    holds = cw.parse_hold_edges([r"^(ExR6|ER6)$:^(PEN_|EPG$)"])
    gains = [(r"^EPG$", r"^PEN_", 2.0), (r"^ExR6$", r"^EPG$", 3.0)]
    p_first = brain.LIFParams(receptor_model=None, type_path_gain=list(brain.DEFAULT_TYPE_PATH_GAIN) + [(a, b, f) for a, b, f in holds] + gains)
    p_last = brain.LIFParams(receptor_model=None, type_path_gain=list(brain.DEFAULT_TYPE_PATH_GAIN) + gains + [(a, b, f) for a, b, f in holds])
    assert np.array_equal(np.asarray(brain._shaped_weights(c, p_first).todense()),
                          np.asarray(brain._shaped_weights(c, p_last).todense()))
