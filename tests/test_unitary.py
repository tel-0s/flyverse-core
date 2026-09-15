"""LIFParams.w_syn_by_nt (per-transmitter unitary strength; docs/audits/unitary_strength.md): None / {} / all-ones are
byte-identical to the shipped shaped weights, the multiplier lands on exactly the entries whose PRESYNAPTIC cell
carries the transmitter, before the connection cap, and a Brain built with it on the CPU carries the same matrix.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_unitary.py -q

CPU, synthetic graphs (tests/test_receptor_model.py's); one test counts the entries per transmitter on the cached
connectome when it is present (skipped otherwise)."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")        # a CPU test never touches this machine's GPU

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from flyverse import connectome as cn                      # noqa: E402
from flyverse.brain import Brain, LIFParams, _shaped_weights, _nt_factor   # noqa: E402
from test_receptor_model import graph                      # noqa: E402  (6-cell graph: ACh, Glu, GABA, DA, ACh, unknown)


def _bytes(W):
    W = W.tocsr()
    if not W.has_sorted_indices:
        W = W.copy(); W.sort_indices()
    return W.data.tobytes(), W.indices.tobytes(), W.indptr.tobytes()


class NoneIsBitIdentical(unittest.TestCase):
    def setUp(self):
        self.c, self.cnt = graph()

    def test_none_empty_and_ones_are_byte_identical(self):
        base = _shaped_weights(self.c, LIFParams(receptor_model=None))
        for over in (None, {}, {k: 1.0 for k in cn.NT_SIGN}, {"acetylcholine": 1.0}):
            p = LIFParams(receptor_model=None, w_syn_by_nt=over)
            self.assertEqual(_bytes(_shaped_weights(self.c, p)), _bytes(base), f"w_syn_by_nt={over!r} moved the weights")

    def test_default_field_is_none(self):
        self.assertIsNone(LIFParams().w_syn_by_nt)

    def test_brain_matrix_identical_under_none(self):
        b0 = Brain(self.c, LIFParams(receptor_model=None), device="cpu")
        b1 = Brain(self.c, LIFParams(receptor_model=None, w_syn_by_nt=None), device="cpu")
        self.assertEqual(_bytes(b0._W_cpu), _bytes(b1._W_cpu))
        # and one CPU step sequence with a stimulus gives the same state
        torch.manual_seed(0)
        for b in (b0, b1):
            b.drive[:, 0] = 9.0
            b.step(200)
        self.assertTrue(torch.equal(b0.v, b1.v) and torch.equal(b0.spike_counts, b1.spike_counts))


class MultiplierLandsOnTheRightEdges(unittest.TestCase):
    def setUp(self):
        self.c, self.cnt = graph()
        self.nt = self.c.neurons.nt.to_numpy()

    def test_factor_per_cell(self):
        f = _nt_factor(self.c, {"gaba": 0.5, "glutamate": 0.25})
        np.testing.assert_array_equal(f, [1, 0.25, 0.5, 1, 1, 1])
        with self.assertRaises(ValueError):
            _nt_factor(self.c, {"not-a-transmitter": 1.0})
        with self.assertRaises(ValueError):
            _nt_factor(self.c, {"gaba": -1.0})

    def test_exact_entries_before_the_cap(self):
        over = {"acetylcholine": 0.5, "gaba": 0.25, "glutamate": 2.0}
        cap = 60.0
        p = LIFParams(receptor_model=None, w_syn_by_nt=over, conn_cap=cap, same_type_gain=1.0, path_gain=[], type_path_gain=[])
        W = _shaped_weights(self.c, p).tocoo()
        sign = self.c.neurons.sign.to_numpy()
        for r, q, v in zip(W.row, W.col, W.data):
            f = over.get(self.nt[q], 1.0)
            expect = np.float32(sign[q]) * np.float32(min(np.float32(self.cnt[r, q]) * np.float32(f), np.float32(cap)))
            self.assertEqual(np.float32(v), expect, f"entry post {r} <- pre {q} ({self.nt[q]})")
        # the cap applies to the SCALED count: glutamate cell 1 -> cell 4 has 90 synapses x2 = 180 -> capped at 60
        W = W.tocsr()
        self.assertEqual(float(W[4, 1]), -60.0)
        # ... and with a factor of 0.5 the 90-synapse ACh cell 0 -> cell 1 (70) becomes 35, not min(70, 60) x 0.5
        self.assertEqual(float(W[1, 0]), 35.0)
        # sign-0 presynaptic cells (dopamine, unknown) stay explicit zeros whatever their factor
        p2 = LIFParams(receptor_model=None, w_syn_by_nt={"dopamine": 5.0, "unknown": 5.0})
        W2 = _shaped_weights(self.c, p2).tocsr()
        self.assertEqual(float(abs(W2[:, 3]).sum()), 0.0)
        self.assertEqual(float(abs(W2[:, 5]).sum()), 0.0)

    def test_entries_touched_equal_transmitter_membership(self):
        over = {"gaba": 0.5}
        base = _shaped_weights(self.c, LIFParams(receptor_model=None)).tocoo()
        W = _shaped_weights(self.c, LIFParams(receptor_model=None, w_syn_by_nt=over)).tocoo()
        moved = base.data != W.data
        is_gaba_pre = self.nt[base.col] == "gaba"
        nonzero = base.data != 0
        np.testing.assert_array_equal(moved, is_gaba_pre & nonzero)

    def test_fan_in_scale_recomputed(self):
        # a per-transmitter scale changes the shaped totals, so the fan-in factor must not come from a stale cache
        c = self.c
        c._norm_cache.clear()
        b0 = Brain(c, LIFParams(receptor_model=None, input_norm_ref=20.0), device="cpu")
        b1 = Brain(c, LIFParams(receptor_model=None, input_norm_ref=20.0, w_syn_by_nt={"acetylcholine": 0.1}), device="cpu")
        self.assertFalse(np.allclose(b0.input_scale, b1.input_scale))
        self.assertEqual(len(c._norm_cache), 2)


class CachedConnectomeCounts(unittest.TestCase):
    """The multiplier lands on exactly the entries of the transmitter, counted from the cache's nt column."""

    def test_entries_per_transmitter(self):
        if not (cn.CACHE_DIR / "W_post_pre.npz").exists():
            self.skipTest("no cached connectome")
        c = cn.load(verbose=False)
        nt = c.neurons.nt.fillna("unknown").to_numpy()
        base = _shaped_weights(c, LIFParams())
        raw = c.W.tocsr()
        raw.sort_indices()
        np.testing.assert_array_equal(base.indices, raw.indices)           # same entry order: raw counts align
        cnt = np.abs(raw.data)
        cap = LIFParams().conn_cap
        for t in ("acetylcholine", "gaba", "glutamate", "histamine"):
            W = _shaped_weights(c, LIFParams(w_syn_by_nt={t: 0.5}))
            moved = base.data != W.data
            pre = nt[base.indices]
            # every non-zero entry of the transmitter moves unless the cap saturates both the raw and the scaled
            # count (count >= 2 x cap): x0.5 lands on exactly the transmitter's entries, before the cap
            expect = (pre == t) & (base.data != 0) & (np.minimum(cnt * 0.5, cap) != np.minimum(cnt, cap))
            self.assertEqual(int(moved.sum()), int(expect.sum()))
            np.testing.assert_array_equal(moved, expect)
            self.assertTrue(np.all(moved[(pre == t) & (base.data != 0) & (cnt < 2 * cap)]))
            self.assertFalse(np.any(moved[pre != t]))
        # None on the real graph: byte-identical
        self.assertEqual(_bytes(_shaped_weights(c, LIFParams(w_syn_by_nt=None))), _bytes(base))


if __name__ == "__main__":
    unittest.main()
