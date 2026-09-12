"""The optional receptor model (LIFParams.receptor_model): off is byte-identical, 'sign' touches only matched
edges, the slow term integrates with its time constant. CPU, synthetic graphs; one test uses the cached
connectome when it is present (skipped otherwise)."""
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from flyverse import connectome as cn
from flyverse.connectome import Connectome, receptor_signs, RECEPTOR_TIERS, GAIN_CLASSES
from flyverse.brain import Brain, LIFParams, DEFAULT_RECEPTOR_GAIN, _shaped_weights

TABLE_COLUMNS = ["malecns_type", "transmitter", "fast_sign", "fast_gain_class", "fast_sign_abs", "fast_gain_class_abs",
                 "fast_sign_nonmda", "fast_gain_class_nonmda", "slow_sign", "slow_gain_class", "slow_sign_abs",
                 "slow_gain_class_abs", "tier", "source", "pool_mixed"]


def table_row(t, nt, fs, fg, ss, sg, tier="exact"):
    return dict(malecns_type=t, transmitter=nt, fast_sign=fs, fast_gain_class=fg, fast_sign_abs=fs, fast_gain_class_abs=fg,
                fast_sign_nonmda=fs, fast_gain_class_nonmda=fg, slow_sign=ss, slow_gain_class=sg, slow_sign_abs=ss,
                slow_gain_class_abs=sg, tier=tier, source="davis2020", pool_mixed=False)


def small_table() -> pd.DataFrame:
    """Post type 'TA': glutamate excitatory (iGluR, high), ACh + (mid) with a slow - (mAChR-B, low), dopamine slow + (mid);
    'TB': glutamate no fast receptor (sign 0), GABA - (low); 'TC' has no row (unmatched)."""
    rows = [table_row("TA", "glutamate", 1, "high", 0, "none"),
            table_row("TA", "acetylcholine", 1, "mid", -1, "low"),
            table_row("TA", "dopamine", 0, "none", 1, "mid"),
            table_row("TB", "glutamate", 0, "none", -1, "high", tier="fuzzy"),
            table_row("TB", "gaba", -1, "low", 0, "none", tier="fuzzy"),
            table_row("<nt=acetylcholine>", "gaba", -1, "mid", -1, "mid", tier="nt_class")]
    return pd.DataFrame(rows)[TABLE_COLUMNS]


def graph() -> Connectome:
    """6 neurons: 0 ACh, 1 glutamate, 2 GABA, 3 dopamine (sign 0), 4 ACh, 5 unknown; types TA, TB, TC, TC, TA, TB."""
    nts = ["acetylcholine", "glutamate", "gaba", "dopamine", "acetylcholine", "unknown"]
    n = pd.DataFrame({"bodyId": np.arange(100, 106), "type": ["TA", "TB", "TC", "TC", "TA", "TB"],
                      "superclass": ["cb_intrinsic"] * 6, "class": [""] * 6, "subclass": [""] * 6, "somaSide": ["L"] * 6,
                      "nt": nts, "sign": np.array([cn.NT_SIGN[x] for x in nts], dtype=np.float32)})
    # counts (post, pre); the dopamine and unknown cells' outputs are explicit zeros as in the cache
    cnt = np.array([[0, 30, 10, 20, 5, 4],
                    [70, 0, 15, 0, 0, 0],
                    [8, 12, 0, 6, 9, 0],
                    [3, 0, 0, 0, 0, 0],
                    [0, 90, 20, 50, 0, 0],
                    [0, 0, 0, 0, 25, 0]], dtype=np.float32)
    sign = n.sign.to_numpy()
    coo = sp.coo_matrix(cnt)
    W = sp.csr_matrix((coo.data * sign[coo.col], (coo.row, coo.col)), shape=(6, 6), dtype=np.float32)
    W.sort_indices()
    return Connectome(n, W, pd.Series(np.arange(6), index=n.bodyId)), cnt


class ReceptorLookupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = Path(self.tmp.name) / "receptors.csv"
        with open(self.table, "w", encoding="utf-8") as f:
            f.write("# test table\n")
            small_table().to_csv(f, index=False)
        self.c, self.cnt = graph()

    def tearDown(self):
        self.tmp.cleanup()

    def test_fallback_reproduces_presynaptic_sign(self):
        rs = receptor_signs(self.c, table_path=self.table)
        W = self.c.W
        self.assertEqual(len(rs.fast_sign), W.nnz)
        # unmatched entries: abs(W) * fast_sign == W exactly; explicit zeros stay zero
        un = ~rs.matched
        np.testing.assert_array_equal(np.abs(W.data[un]) * rs.fast_sign[un], W.data[un])
        self.assertTrue(np.all(rs.slow_sign[un] == 0))
        coo = W.tocoo(); types = self.c.neurons.type.to_numpy(); nt = self.c.neurons.nt.to_numpy()
        expect_matched = np.array([(types[r], t) in {("TA", "glutamate"), ("TA", "acetylcholine"), ("TA", "dopamine"),
                                                     ("TB", "glutamate"), ("TB", "gaba")}
                                   for r, t in zip(coo.row, nt[coo.col])])
        np.testing.assert_array_equal(rs.matched, expect_matched)
        tiers = np.array(RECEPTOR_TIERS)[rs.tier]
        self.assertTrue(np.all(tiers[(nt[coo.col] == "unknown")] == "pre_unknown"))
        self.assertTrue(np.all(np.isin(tiers[expect_matched], ["exact", "fuzzy"])))
        # glutamate onto TA flips to +1 (high), onto TB becomes 0 (no fast receptor)
        glu_TA = (nt[coo.col] == "glutamate") & (types[coo.row] == "TA")
        glu_TB = (nt[coo.col] == "glutamate") & (types[coo.row] == "TB")
        self.assertTrue(np.all(rs.fast_sign[glu_TA] == 1) and np.all(rs.fast_gain[glu_TA] == GAIN_CLASSES.index("high")))
        self.assertTrue(np.all(rs.fast_sign[glu_TB] == 0))
        cov = rs.coverage(W)
        self.assertAlmostEqual(float(cov[cov.tier == "matched"].edges_frac.iloc[0]), expect_matched.mean())

    def test_nt_class_fallback(self):
        rs0 = receptor_signs(self.c, table_path=self.table)
        rs = receptor_signs(self.c, table_path=self.table, nt_class_fallback=True)
        coo = self.c.W.tocoo(); nt = self.c.neurons.nt.to_numpy(); types = self.c.neurons.type.to_numpy()
        # GABA onto the cholinergic TA cells has no (TA, gaba) row: with the fallback it takes the <nt=acetylcholine>
        # gaba baseline (tier nt_class, slow -1); without it, plain fallback (tier fallback, slow 0)
        sel = (nt[coo.col] == "gaba") & (nt[coo.row] == "acetylcholine")
        self.assertEqual(int(sel.sum()), 2)
        self.assertTrue(np.all(np.array(RECEPTOR_TIERS)[rs.tier[sel]] == "nt_class"))
        self.assertTrue(np.all(rs.slow_sign[sel] == -1) and np.all(rs.fast_sign[sel] == -1))
        self.assertTrue(np.all(np.array(RECEPTOR_TIERS)[rs0.tier[sel]] == "fallback"))
        self.assertTrue(np.all(rs0.slow_sign[sel] == 0))
        other = ~sel
        np.testing.assert_array_equal(rs.fast_sign[other], rs0.fast_sign[other])
        np.testing.assert_array_equal(rs.tier[other], rs0.tier[other])

    def test_off_is_byte_identical(self):
        base = _shaped_weights(self.c, LIFParams())
        off = _shaped_weights(self.c, LIFParams(receptor_model=None, receptor_table=str(self.table)))
        np.testing.assert_array_equal(base.data, off.data)
        np.testing.assert_array_equal(base.indices, off.indices); np.testing.assert_array_equal(base.indptr, off.indptr)
        b0 = Brain(self.c, LIFParams(event_driven=False), device="cpu")
        b1 = Brain(self.c, LIFParams(event_driven=False, receptor_table=str(self.table)), device="cpu")
        torch.testing.assert_close(b0.W.to_dense(), b1.W.to_dense(), rtol=0, atol=0)
        self.assertIsNone(b1.receptor); self.assertIsNone(b1.W_slow); self.assertFalse(b1._slow_on)

    def test_sign_changes_only_matched_edges(self):
        p0 = LIFParams(event_driven=False, same_type_gain=1.0)          # (TA -> TA would otherwise be damped x0.1 in both)
        p1 = LIFParams(event_driven=False, same_type_gain=1.0, receptor_model="sign", receptor_table=str(self.table))
        W0 = _shaped_weights(self.c, p0).tocsr(); W1 = _shaped_weights(self.c, p1).tocsr()
        rs = receptor_signs(self.c, table_path=self.table)
        raw = self.c.W.tocsr()
        # same sparsity structure; unmatched entries identical; matched entries = |count capped| * table sign
        np.testing.assert_array_equal(W0.indices, W1.indices); np.testing.assert_array_equal(W0.indptr, W1.indptr)
        np.testing.assert_array_equal(W0.data[~rs.matched], W1.data[~rs.matched])
        capped = np.minimum(np.abs(raw.data), 60.0)
        np.testing.assert_array_equal(W1.data[rs.matched], capped[rs.matched] * rs.fast_sign[rs.matched])
        changed = W0.data != W1.data
        self.assertTrue(changed.any() and np.all(rs.matched[changed]))
        b0 = Brain(self.c, p0, device="cpu"); b1 = Brain(self.c, p1, device="cpu")
        D0, D1 = b0.W.to_dense().numpy(), b1.W.to_dense().numpy()
        types = self.c.neurons.type.to_numpy(); nt = self.c.neurons.nt.to_numpy()
        for post in range(6):
            for pre in range(6):
                if types[post] == "TC" or (types[post], nt[pre]) not in {("TA", "glutamate"), ("TA", "acetylcholine"),
                                                                        ("TA", "dopamine"), ("TB", "glutamate"), ("TB", "gaba")}:
                    self.assertEqual(D0[post, pre], D1[post, pre], (post, pre))
        # glutamate (pre 1) -> TA (post 0): excitatory now, same magnitude
        self.assertLess(D0[0, 1], 0); self.assertEqual(D1[0, 1], -D0[0, 1])
        # glutamate -> TB (post 5 has no glutamate input; post 1 is TB with 0 count) -- TB post 1 <- pre 1 absent

    def test_gain_classes_before_cap(self):
        p = LIFParams(event_driven=False, receptor_model="sign+gain", receptor_table=str(self.table), conn_cap=60.0,
                      input_norm_alpha=0.0, same_type_gain=1.0, path_gain=[], type_path_gain=[])
        W = _shaped_weights(self.c, p).toarray()
        # ACh (pre 0) -> TA (post 4 count 0; post 0 count 0) ... use GABA (pre 2) -> TB (post 1, count 15, class low)
        self.assertAlmostEqual(W[1, 2], -15 * 0.5, places=5)
        # glutamate (pre 1) -> TA (post 4, count 90, class high): 90 * 1.5 = 135 -> capped at 60
        self.assertAlmostEqual(W[4, 1], 60.0, places=5)
        # glutamate (pre 1) -> TA (post 0, count 30, high): 45, under the cap
        self.assertAlmostEqual(W[0, 1], 45.0, places=5)
        custom = LIFParams(**{**p.__dict__, "receptor_gain": {"low": 0.25, "mid": 1.0, "high": 2.0}})
        Wc = _shaped_weights(self.c, custom).toarray()
        self.assertAlmostEqual(Wc[1, 2], -15 * 0.25, places=5); self.assertAlmostEqual(Wc[0, 1], 60.0, places=5)

    def test_slow_term_time_constant(self):
        """Two neurons: dopamine cell 0 -> TA cell 1 (slow +, mid). The fast synapse is 0; g_slow jumps by
        w_syn * slow_gain * count and decays with slow_tau_ms; v follows it."""
        nts = ["dopamine", "acetylcholine"]
        n = pd.DataFrame({"bodyId": [1, 2], "type": ["TC", "TA"], "superclass": ["cb_intrinsic"] * 2, "class": ["", ""],
                          "subclass": ["", ""], "somaSide": ["L", "L"], "nt": nts,
                          "sign": np.array([cn.NT_SIGN[x] for x in nts], dtype=np.float32)})
        W = sp.csr_matrix((np.array([0.0], np.float32), (np.array([1]), np.array([0]))), shape=(2, 2))   # explicit zero
        c = Connectome(n, W, pd.Series([0, 1], index=[1, 2]))
        counts = np.array([20.0], np.float32)
        rs = receptor_signs(c, table_path=self.table, counts=counts)
        self.assertEqual(rs.count[0], 20.0); self.assertEqual(rs.slow_sign[0], 1.0)
        p = LIFParams(event_driven=False, receptor_model="full", receptor_table=str(self.table), slow_tau_ms=150.0,
                      slow_gain=0.1, input_norm_alpha=0.0, path_gain=[], type_path_gain=[], adapt_jump=0.0)
        b = Brain(c, p, device="cpu", receptor=rs)
        self.assertTrue(b._slow_on); self.assertIsNotNone(b.W_slow)
        self.assertEqual(float(b.W.to_dense()[1, 0]), 0.0)                         # no fast synapse
        jump = p.w_syn * p.slow_gain * 20.0 * DEFAULT_RECEPTOR_GAIN["mid"]
        self.assertAlmostEqual(float(b.W_slow.to_dense()[1, 0]), jump, places=6)
        b.spike_buf[b.buf_pos, 0, 0] = 1.0                                        # a transmitted dopamine spike
        b.step(1)
        self.assertAlmostEqual(float(b.g_slow[0, 1]), jump, places=6)
        self.assertAlmostEqual(float(b.g[0, 1]), 0.0, places=6)
        k = 40
        b.step(k)
        self.assertAlmostEqual(float(b.g_slow[0, 1]), jump * math.exp(-k * p.dt / p.slow_tau_ms), places=6)
        self.assertGreater(float(b.v[0, 1]), p.v_rest)                            # depolarised by the slow term
        # exponential-Euler membrane: after the step with g_slow = jump the target was v_rest + jump
        b.reset(); b.spike_buf[b.buf_pos, 0, 0] = 1.0; b.step(1)
        self.assertAlmostEqual(float(b.v[0, 1]), p.v_rest + jump * (1 - math.exp(-p.dt / p.tau_m)), places=5)   # float32 state
        # the same graph with the model off: nothing happens
        b0 = Brain(c, LIFParams(event_driven=False, adapt_jump=0.0), device="cpu")
        b0.spike_buf[b0.buf_pos, 0, 0] = 1.0; b0.step(k + 1)
        self.assertEqual(float(b0.v[0, 1]), p.v_rest); self.assertEqual(float(b0.g_slow[0, 1]), 0.0)

    def test_full_forces_torch_path(self):
        p = LIFParams(event_driven=False, receptor_model="full", receptor_table=str(self.table))
        rs = receptor_signs(self.c, table_path=self.table, counts=np.abs(self.c.W.data) + 1.0)
        b = Brain(self.c, p, device="cpu", receptor=rs)
        self.assertFalse(b.cuda); self.assertFalse(b.metal)
        with self.assertRaises(ValueError):
            Brain(self.c, LIFParams(receptor_model="nonsense"), device="cpu")


@unittest.skipUnless((cn.CACHE_DIR / "W_post_pre.npz").exists(), "connectome cache absent")
class CachedConnectomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = cn.load(verbose=False)
        cls.sub = cls.c.subset(cls.c.select(type=["Mi4", "Mi1", "L1", "Tm5Y", "T4a", "PAM08", "KCg-m", "MBON01", "DNp01"]))

    def test_off_identical_and_sign_changes_only_matched(self):
        sub = self.sub
        p0 = LIFParams(event_driven=False); p1 = LIFParams(event_driven=False, receptor_model="sign")
        b0 = Brain(sub, p0, device="cpu"); b1 = Brain(sub, p1, device="cpu")
        boff = Brain(sub, LIFParams(event_driven=False, receptor_model=None), device="cpu")
        np.testing.assert_array_equal(b0._W_cpu.data, boff._W_cpu.data)
        rs = b1.receptor
        self.assertEqual(len(rs.fast_sign), sub.W.nnz)
        # subset lookup == full-graph lookup restricted to the subset (tier, signs)
        full = receptor_signs(self.c)
        idx = self.c.index_of(sub.neurons.bodyId)
        # map the subset's entries onto the full graph's CSR order through (post, pre) keys
        cf = self.c.W.tocoo(); cs = sub.W.tocoo()
        key_full = cf.row.astype(np.int64) * self.c.n + cf.col
        key_sub = idx[cs.row].astype(np.int64) * self.c.n + idx[cs.col]
        order = np.argsort(key_full); pos = np.searchsorted(key_full, key_sub, sorter=order)
        self.assertTrue(np.array_equal(key_full[order[pos]], key_sub))
        np.testing.assert_array_equal(full.fast_sign[order[pos]], rs.fast_sign)
        np.testing.assert_array_equal(full.tier[order[pos]], rs.tier)
        # unmatched entries untouched; matched entries only may differ (before fan-in scaling: compare _shaped_weights)
        W0 = _shaped_weights(sub, p0).tocsr(); W1 = _shaped_weights(sub, p1).tocsr()
        np.testing.assert_array_equal(W0.data[~rs.matched], W1.data[~rs.matched])
        changed = W0.data != W1.data
        self.assertTrue(np.all(rs.matched[changed]))
        self.assertGreater(int(changed.sum()), 0)                       # glutamate -> Mi4 (iGluR) flips
        # the full-graph fallback reproduces c.W exactly
        un = ~full.matched
        np.testing.assert_array_equal(np.abs(self.c.W.data[un]) * full.fast_sign[un], self.c.W.data[un])


if __name__ == "__main__":
    unittest.main()
