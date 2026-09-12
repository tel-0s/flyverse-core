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
from flyverse.brain import Brain, LIFParams, DEFAULT_RECEPTOR_GAIN, RECEPTOR_MODELS, _shaped_weights, _receptor_key

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
        # receptor_model=None: the table is never read, whatever path is set (round 3: None is no longer the default)
        base = _shaped_weights(self.c, LIFParams(receptor_model=None))
        off = _shaped_weights(self.c, LIFParams(receptor_model=None, receptor_table=str(self.table)))
        np.testing.assert_array_equal(base.data, off.data)
        np.testing.assert_array_equal(base.indices, off.indices); np.testing.assert_array_equal(base.indptr, off.indptr)
        b0 = Brain(self.c, LIFParams(event_driven=False, receptor_model=None), device="cpu")
        b1 = Brain(self.c, LIFParams(event_driven=False, receptor_model=None, receptor_table=str(self.table)), device="cpu")
        torch.testing.assert_close(b0.W.to_dense(), b1.W.to_dense(), rtol=0, atol=0)
        self.assertIsNone(b1.receptor); self.assertIsNone(b1.W_slow); self.assertFalse(b1._slow_on)
        # the default model on a graph whose types have no row in the shipped table changes nothing either
        bd = Brain(self.c, LIFParams(event_driven=False), device="cpu")
        self.assertIsNotNone(bd.receptor); self.assertEqual(int(bd.receptor.matched.sum()), 0)
        torch.testing.assert_close(b0.W.to_dense(), bd.W.to_dense(), rtol=0, atol=0)

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
        self.assertEqual(b.slow_classes, ["monoamine"])                             # round 2: one matrix per active class
        self.assertAlmostEqual(float(b.W_slow[0].to_dense()[1, 0]), jump, places=6)
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
        # explicit None (the pre-round-3 default) against the class rule this test was written for
        p0 = LIFParams(event_driven=False, receptor_model=None); p1 = LIFParams(event_driven=False, receptor_model="sign", receptor_net_rule="class")
        b0 = Brain(sub, p0, device="cpu"); b1 = Brain(sub, p1, device="cpu")
        boff = Brain(sub, LIFParams(event_driven=False, receptor_model=None, receptor_table=str(cn.RECEPTOR_TABLE)), device="cpu")
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

    def test_type_nt_override_targets_exist(self):
        """TYPE_NT_OVERRIDE names real MaleCNS types and NT_SIGN transmitters; the shipped cache states whether it was applied."""
        n = self.c.neurons
        for t, nt in cn.TYPE_NT_OVERRIDE.items():
            self.assertIn(nt, cn.NT_SIGN)
            self.assertGreater(int((n.type == t).sum()), 0, t)
        # a cache built with the override has every cell of each type relabelled; one built without keeps its consensus
        applied = [bool((n.nt[n.type == t] == nt).all()) for t, nt in cn.TYPE_NT_OVERRIDE.items()]
        self.assertIn(all(applied), (True, False))

    # ---- round-3 adoption (docs/audits/receptor_integration.md "Round 3: adoption") ---------------------------------
    # Hashes of brain._shaped_weights (data + indices + indptr of the sorted CSR) on the adopted TYPE_NT_OVERRIDE cache
    # (nnz 25,578,600, sum|W| 121,460,584, glutamate cells 29,707), computed with EXPLICIT receptor settings before the
    # default changed (scratch hash_weights.py, 2026-09-12): receptor_model=None must keep giving the previous weights.
    ADOPTED_CACHE = {"nnz": 25_578_600, "sum_abs_W": 121_460_584, "glutamate_cells": 29_707}
    WEIGHTS_MD5_NONE = "2e276b30b6117c1f62688b01775eda6b"          # receptor_model=None: the pre-round-3 default weights
    WEIGHTS_MD5_SIGN_ABS = "f0d145d1bb81b446ebc51f89ded7bd4b"      # receptor_model='sign', receptor_net_rule='abs' (shipped table)
    SIGN_ABS_VS_NONE = {"differing": 48_295, "flipped": 30_916, "zeroed": 17_379}

    @staticmethod
    def _weights_md5(W):
        import hashlib
        W = W.tocsr(); W.sort_indices()
        m = hashlib.md5(); m.update(W.data.tobytes()); m.update(W.indices.tobytes()); m.update(W.indptr.tobytes())
        return m.hexdigest(), W

    def test_default_is_sign_abs_and_none_is_selectable(self):
        p = LIFParams()
        self.assertEqual(p.receptor_model, "sign"); self.assertEqual(p.receptor_net_rule, "abs")
        self.assertFalse(p.receptor_nt_class_fallback); self.assertIsNone(p.receptor_table)
        self.assertIn(None, RECEPTOR_MODELS)
        p0 = LIFParams(receptor_model=None)
        self.assertIsNone(p0.receptor_model); self.assertIsNone(_receptor_key(p0))
        self.assertEqual(_receptor_key(LIFParams()), ("sign", "abs", False, None, None))
        # a Brain under None carries no lookup; under the default it carries one aligned with W
        b0 = Brain(self.sub, LIFParams(event_driven=False, receptor_model=None), device="cpu")
        bd = Brain(self.sub, LIFParams(event_driven=False), device="cpu")
        self.assertIsNone(b0.receptor); self.assertEqual(len(bd.receptor.fast_sign), self.sub.W.nnz)

    def test_none_reproduces_previous_weights_byte_for_byte(self):
        c = self.c
        on_adopted_cache = (c.W.nnz == self.ADOPTED_CACHE["nnz"] and int(np.abs(c.W.data).sum()) == self.ADOPTED_CACHE["sum_abs_W"]
                            and int((c.neurons.nt == "glutamate").sum()) == self.ADOPTED_CACHE["glutamate_cells"])
        h_none, W_none = self._weights_md5(_shaped_weights(c, LIFParams(receptor_model=None)))
        h_def, W_def = self._weights_md5(_shaped_weights(c, LIFParams()))
        h_abs, W_abs = self._weights_md5(_shaped_weights(c, LIFParams(receptor_model="sign", receptor_net_rule="abs")))
        self.assertEqual(h_def, h_abs)                                   # the default IS sign / abs on the shipped table
        self.assertNotEqual(h_def, h_none)
        d = W_def.data != W_none.data
        self.assertTrue(np.all(W_def.indices == W_none.indices)); self.assertTrue(np.all(W_def.indptr == W_none.indptr))
        self.assertEqual(int(((W_def.data == 0) & (W_none.data != 0)).sum()) + int((np.sign(W_def.data) * np.sign(W_none.data) < 0).sum()), int(d.sum()))
        if not on_adopted_cache:
            self.skipTest("cache is not the adopted TYPE_NT_OVERRIDE cache; the pinned hashes do not apply")
        self.assertEqual(h_none, self.WEIGHTS_MD5_NONE)                  # receptor_model=None: the previous weights
        self.assertEqual(h_def, self.WEIGHTS_MD5_SIGN_ABS)               # the round-3 default on the shipped table
        self.assertEqual(int(d.sum()), self.SIGN_ABS_VS_NONE["differing"])
        self.assertEqual(int((np.sign(W_def.data) * np.sign(W_none.data) < 0).sum()), self.SIGN_ABS_VS_NONE["flipped"])
        self.assertEqual(int(((W_def.data == 0) & (W_none.data != 0)).sum()), self.SIGN_ABS_VS_NONE["zeroed"])


# ---- round-2 slow term: class split, zero-cost off, the multiplicative variants, the optic-lobe term ----------------
from flyverse.brain import SlowSpec, _slow_spec, _slow_gains, _slow_taus, DEFAULT_SLOW_TAU_BY_CLASS  # noqa: E402
from flyverse.connectome import SLOW_CLASSES  # noqa: E402
from flyverse import optic as optic_mod  # noqa: E402
from flyverse.retina import Retina, EyeGeometry  # noqa: E402


def two_neuron_graph(pre_nt="dopamine", post_type="TA"):
    """cell 0 (pre_nt, type TC) -> cell 1 (post_type, acetylcholine); the edge is an explicit zero when the presynaptic
    cell is a monoamine (as in the cache), so its count comes from `counts`."""
    nts = [pre_nt, "acetylcholine"]
    n = pd.DataFrame({"bodyId": [1, 2], "type": ["TC", post_type], "superclass": ["cb_intrinsic"] * 2, "class": ["", ""],
                      "subclass": ["", ""], "somaSide": ["L", "L"], "nt": nts,
                      "sign": np.array([cn.NT_SIGN[x] for x in nts], dtype=np.float32)})
    W = sp.csr_matrix((np.array([cn.NT_SIGN[pre_nt] * 20.0], np.float32), (np.array([1]), np.array([0]))), shape=(2, 2))
    return Connectome(n, W, pd.Series([0, 1], index=[1, 2]))


class SlowTermRound2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = Path(self.tmp.name) / "receptors.csv"
        with open(self.table, "w", encoding="utf-8") as f:
            f.write("# test table\n")
            small_table().to_csv(f, index=False)
        self.c, self.cnt = graph()

    def tearDown(self):
        self.tmp.cleanup()

    def _full(self, **kw):
        base = dict(event_driven=False, receptor_model="full", receptor_table=str(self.table), input_norm_alpha=0.0,
                    path_gain=[], type_path_gain=[], adapt_jump=0.0, same_type_gain=1.0)
        base.update(kw)
        return LIFParams(**base)

    def test_slow_class_per_entry(self):
        rs = receptor_signs(self.c, table_path=self.table, counts=self.cnt[self.c.W.tocoo().row, self.c.W.tocoo().col])
        coo = self.c.W.tocoo(); nt = self.c.neurons.nt.to_numpy()[coo.col]; types = self.c.neurons.type.to_numpy()[coo.row]
        names = np.array(SLOW_CLASSES)[rs.slow_class]
        # ACh -> TA carries mAChR-B (slow -1): classical; dopamine -> TA slow +1: monoamine; glutamate -> TB slow -1 (mGluR): classical
        self.assertTrue(np.all(names[(nt == "acetylcholine") & (types == "TA")] == "metabotropic_classical"))
        self.assertTrue(np.all(names[(nt == "dopamine") & (types == "TA")] == "monoamine"))
        self.assertTrue(np.all(names[(nt == "glutamate") & (types == "TB")] == "metabotropic_classical"))
        self.assertTrue(np.all(names[rs.slow_sign == 0] == "none"))
        self.assertTrue(np.all((rs.slow_sign != 0) == (rs.slow_class != 0)))
        # slow_factor restricted to a class zeroes the other class's entries
        f_mono = rs.slow_factor(slow_class="monoamine"); f_cls = rs.slow_factor(slow_class="metabotropic_classical")
        self.assertTrue(np.all(f_mono[names != "monoamine"] == 0) and np.all(f_mono[names == "monoamine"] == rs.slow_sign[names == "monoamine"]))
        self.assertTrue(np.all(f_cls[names != "metabotropic_classical"] == 0))
        np.testing.assert_array_equal(f_mono + f_cls, rs.slow_factor())
        with self.assertRaises(ValueError):
            rs.slow_factor(slow_class="nonsense")

    def test_class_split_builds_one_matrix_per_active_class(self):
        counts = self.cnt[self.c.W.tocoo().row, self.c.W.tocoo().col]
        rs = receptor_signs(self.c, table_path=self.table, counts=counts)
        # defaults: classical 0 (off), monoamine slow_gain -> only the monoamine matrix (the dopamine -> TA entries)
        b = Brain(self.c, self._full(), device="cpu", receptor=rs)
        self.assertEqual(b.slow_classes, ["monoamine"]); self.assertEqual(len(b.W_slow), 1)
        M = b.W_slow[0].to_dense().numpy()
        nt = self.c.neurons.nt.to_numpy(); types = self.c.neurons.type.to_numpy()
        for post in range(6):
            for pre in range(6):
                expect = 0.0
                if nt[pre] == "dopamine" and types[post] == "TA":
                    expect = b.p.w_syn * b.p.slow_gain * min(self.cnt[post, pre], 60.0) * DEFAULT_RECEPTOR_GAIN["mid"]
                self.assertAlmostEqual(M[post, pre], expect, places=6, msg=(post, pre))
        # classical on, monoamine off: only the mAChR-B / mGluR entries, with the classical scale
        p2 = self._full(slow_gain_by_class={"metabotropic_classical": 0.05, "monoamine": 0.0})
        b2 = Brain(self.c, p2, device="cpu", receptor=rs)
        self.assertEqual(b2.slow_classes, ["metabotropic_classical"])
        M2 = b2.W_slow[0].to_dense().numpy()
        self.assertTrue(np.all(M2[:, nt == "dopamine"] == 0))
        ach_TA = (M2[0, 4], self.cnt[0, 4])          # ACh cell 4 -> TA cell 0: slow -1 (mAChR-B, low)
        self.assertAlmostEqual(ach_TA[0], -0.275 * 0.05 * ach_TA[1] * DEFAULT_RECEPTOR_GAIN["low"], places=6)
        # both on: two matrices, two time constants, in SLOW_CLASSES order
        p3 = self._full(slow_gain_by_class={"metabotropic_classical": 0.05, "monoamine": 0.02},
                        slow_tau_by_class={"metabotropic_classical": 80.0}, slow_tau_ms=300.0)
        b3 = Brain(self.c, p3, device="cpu", receptor=rs)
        self.assertEqual(b3.slow_classes, ["metabotropic_classical", "monoamine"])
        self.assertEqual(b3.slow.tau, {"metabotropic_classical": 80.0, "monoamine": 300.0})
        self.assertEqual(b3.g_slow_cls.shape, (2, 1, 6))
        with self.assertRaises(ValueError):
            Brain(self.c, self._full(slow_gain_by_class={"peptide": 1.0}), device="cpu", receptor=rs)
        with self.assertRaises(ValueError):
            Brain(self.c, self._full(slow_mode="nonsense"), device="cpu", receptor=rs)
        self.assertEqual(_slow_gains(LIFParams())["metabotropic_classical"], 0.0)
        self.assertEqual(_slow_taus(LIFParams())["metabotropic_classical"], DEFAULT_SLOW_TAU_BY_CLASS["metabotropic_classical"])

    def test_zero_gains_cost_nothing_and_equal_sign_gain(self):
        """'full' with every class scale 0: no slow matrix, no slow state, the Torch path kept, and the dynamics
        byte-identical to 'sign+gain' on the same path (and receptor_model=None still byte-identical to the base)."""
        counts = self.cnt[self.c.W.tocoo().row, self.c.W.tocoo().col]
        rs = receptor_signs(self.c, table_path=self.table, counts=counts)
        p_off = self._full(slow_gain=0.0)
        self.assertIsNone(_slow_spec(p_off))
        b = Brain(self.c, p_off, device="cpu", receptor=rs)
        self.assertTrue(b._slow_on); self.assertFalse(b._slow_active); self.assertIsNone(b.W_slow)
        self.assertEqual(b.g_slow_cls.shape, (0, 1, 6)); self.assertEqual(b.slow_classes, [])
        self.assertFalse(b.cuda); self.assertFalse(b.metal)
        p_sg = LIFParams(**{**p_off.__dict__, "receptor_model": "sign+gain"})
        b_sg = Brain(self.c, p_sg, device="cpu", receptor=receptor_signs(self.c, table_path=self.table))
        torch.testing.assert_close(b.W.to_dense(), b_sg.W.to_dense(), rtol=0, atol=0)
        for bb in (b, b_sg):
            bb.set_poisson([0, 1], 200.0); bb.step(200)
        torch.testing.assert_close(b.v, b_sg.v, rtol=0, atol=0); torch.testing.assert_close(b.rate, b_sg.rate, rtol=0, atol=0)
        self.assertEqual(float(b.g_slow.abs().sum()), 0.0)
        # the model off is still byte-identical to the base weights
        base = _shaped_weights(self.c, LIFParams(receptor_model=None)); off = _shaped_weights(self.c, LIFParams(receptor_model=None, slow_mode="gain"))
        np.testing.assert_array_equal(base.data, off.data)

    def _tone(self, mode, **kw):
        """A dopamine spike into TA (slow +, mid, count 20) under `mode`; returns the brain and the tone jump."""
        c = two_neuron_graph()
        rs = receptor_signs(c, table_path=self.table, counts=np.array([20.0], np.float32))
        p = self._full(slow_mode=mode, slow_gain=0.1, slow_tau_ms=1e9, **kw)    # tau -> inf: the tone holds
        b = Brain(c, p, device="cpu", receptor=rs)
        jump = p.w_syn * 0.1 * 20.0 * DEFAULT_RECEPTOR_GAIN["mid"]               # 0.55 mV
        b.spike_buf[b.buf_pos, 0, 0] = 1.0; b.step(1)
        self.assertAlmostEqual(float(b.g_slow[0, 1]), jump, places=6)
        self.assertAlmostEqual(float(b.g_slow_cls[0, 0, 1]), jump, places=6)
        return b, jump

    def test_gain_mode_scales_the_fast_input(self):
        b, jump = self._tone("gain")
        p = b.p
        self.assertEqual(b.slow.norm_mv, p.v_th - p.v_rest)
        # no fast input yet: the tone alone does nothing to the membrane (unlike 'additive')
        self.assertAlmostEqual(float(b.v[0, 1]), p.v_rest, places=6)
        ba, _ = self._tone("additive")
        self.assertGreater(float(ba.v[0, 1]), p.v_rest)
        # a fast conductance g under the tone is scaled by 1 + jump / 7 mV
        for bb in (b, ba):
            bb.g[0, 1] = 2.0
        b.step(1); ba.step(1)
        a_m = math.exp(-p.dt / p.tau_m); a_s = math.exp(-p.dt / p.tau_syn)
        g_now = 2.0 * a_s                                                       # g decays before the membrane update
        f = 1.0 + jump / (p.v_th - p.v_rest)
        self.assertAlmostEqual(float(b.v[0, 1]), p.v_rest + g_now * f * (1 - a_m), places=5)
        # 'additive' adds the tone instead: target v_rest + g + jump
        v_add_1 = p.v_rest + jump * (1 - a_m)                                    # after the first step
        self.assertAlmostEqual(float(ba.v[0, 1]), (p.v_rest + g_now + jump) + (v_add_1 - (p.v_rest + g_now + jump)) * a_m, places=5)
        # the factor is clamped: a huge negative tone silences the input (factor 0), never inverts it
        b.g_slow_cls[0, 0, 1] = -100.0; b.g[0, 1] = 2.0; v0 = float(b.v[0, 1]); b.step(1)
        self.assertLessEqual(float(b.v[0, 1]), max(v0, p.v_rest) + 1e-6)

    def test_threshold_mode_shifts_the_threshold(self):
        b, jump = self._tone("threshold")
        p = b.p
        self.assertAlmostEqual(float(b.v[0, 1]), p.v_rest, places=6)              # the tone is not a current
        th = b._threshold()                                                          # float32 state: 5 places
        self.assertAlmostEqual(float(th[0, 1]), p.v_th - jump, places=5); self.assertAlmostEqual(float(th[0, 0]), p.v_th, places=5)
        # a membrane between the shifted and the nominal threshold fires only under the tone
        b.v[0, 1] = p.v_th - jump / 2; b.step(1)
        self.assertEqual(float(b.spikes[0, 1]), 1.0)
        ba, _ = self._tone("additive"); ba.g_slow_cls.zero_(); ba.g_slow.zero_()
        ba.v[0, 1] = p.v_th - jump / 2; ba.step(1)
        self.assertEqual(float(ba.spikes[0, 1]), 0.0)
        # the shift is capped at 0.9 of the rest-threshold gap; a negative tone raises the threshold without bound
        b.g_slow_cls[0, 0, 1] = 100.0; b.g_slow.copy_(b.g_slow_cls.sum(0)); th = b._threshold()
        self.assertAlmostEqual(float(th[0, 1]), p.v_th - 0.9 * (p.v_th - p.v_rest), places=5)
        b.g_slow_cls[0, 0, 1] = -100.0
        b.g_slow.copy_(b.g_slow_cls.sum(0)); th = b._threshold()
        self.assertAlmostEqual(float(th[0, 1]), p.v_th + 100.0, places=5)

    def test_state_reset_and_batch(self):
        c = two_neuron_graph()
        rs = receptor_signs(c, table_path=self.table, counts=np.array([20.0], np.float32))
        b = Brain(c, self._full(slow_mode="gain"), device="cpu", batch=3, receptor=rs)
        self.assertEqual(b.g_slow_cls.shape, (1, 3, 2)); self.assertEqual(b.g_slow.shape, (3, 2))
        b.spike_buf[b.buf_pos, 1, 0] = 1.0; b.step(1)
        self.assertGreater(float(b.g_slow[1, 1]), 0); self.assertEqual(float(b.g_slow[0, 1]), 0.0)
        b.reset([1])
        self.assertEqual(float(b.g_slow.abs().sum()), 0.0); self.assertEqual(float(b.g_slow_cls.abs().sum()), 0.0)

    # ---- the optic-lobe term ----------------------------------------------------------------------------------------
    def _optic_graph(self):
        """4 cells: 0 photoreceptor R1-R6 (histamine), 1 rate unit Mi1 (ol_intrinsic, ACh), 2 spiking octopamine cell
        (visual_centrifugal), 3 spiking LC4 (visual_projection). Edges: 0 -> 1 (10), 2 -> 1 (20, explicit zero), 1 -> 3 (30)."""
        nts = ["histamine", "acetylcholine", "octopamine", "acetylcholine"]
        n = pd.DataFrame({"bodyId": [1, 2, 3, 4], "type": ["R1-R6", "Mi1", "OAVC", "LC4"],
                          "superclass": ["ol_sensory", "ol_intrinsic", "visual_centrifugal", "visual_projection"],
                          "class": [""] * 4, "subclass": [""] * 4, "somaSide": ["L"] * 4, "nt": nts,
                          "sign": np.array([cn.NT_SIGN[x] for x in nts], dtype=np.float32)})
        W = sp.csr_matrix((np.array([-10.0, 0.0, 30.0], np.float32), (np.array([1, 1, 3]), np.array([0, 2, 1]))), shape=(4, 4))
        W.sort_indices()
        c = Connectome(n, W, pd.Series(np.arange(4), index=n.bodyId))
        counts = np.array([10.0, 20.0, 30.0], np.float32)                          # CSR order: (1,0), (1,2), (3,1)
        rt = pd.DataFrame([table_row("Mi1", "histamine", -1, "mid", 0, "none"),
                           table_row("Mi1", "octopamine", 0, "none", 1, "mid")])[TABLE_COLUMNS]
        path = Path(self.tmp.name) / "optic_table.csv"
        rt.to_csv(path, index=False)
        rs = receptor_signs(c, table_path=path, counts=counts)
        retina = Retina(pr_index=np.array([0]), pr_column=np.array([0]), pr_sens=np.ones((1, 4), np.float32) / 4,
                        col_side=np.array(["L"]), col_hex=np.zeros((1, 2), int), col_dir=np.array([[1.0, 0.0, 0.0]]),
                        col_az_el=np.zeros((1, 2)), geometry=EyeGeometry())
        return c, rs, retina

    def test_optic_slow_term(self):
        c, rs, retina = self._optic_graph()
        # adapt_gain 0: the rate units' slow adaptation would otherwise halve any tonic offset at steady state
        op = optic_mod.OpticParams(pair_gain=[], tau_by_type={}, baseline_by_type={}, adapt_gain=0.0)
        gain, tau, tau_syn = 0.02, 50.0, 5.0
        spec = SlowSpec("additive", {"monoamine": gain}, {"monoamine": tau}, tau_syn, 7.0)
        ol0 = optic_mod.OpticLobe(c, retina, op, device="cpu", receptor=rs, receptor_gain=DEFAULT_RECEPTOR_GAIN)
        ol = optic_mod.OpticLobe(c, retina, op, device="cpu", receptor=rs, receptor_gain=DEFAULT_RECEPTOR_GAIN, slow=spec)
        self.assertEqual(ol.slow_classes, ["monoamine"]); self.assertEqual(ol0.slow_classes, [])
        self.assertEqual(ol.slow_entries["monoamine"]["spiking_to_rate"], 1); self.assertEqual(ol.slow_entries["monoamine"]["rate_to_rate"], 0)
        self.assertFalse(ol.cuda); self.assertFalse(ol.metal)
        # the fast octopamine edge is a zero (fast sign 0) in both; the slow matrix carries count / l2-norm x sign x gain factor
        self.assertEqual(float(ol.W_rs.to_dense()[0, 0]), 0.0)                                # Mi1 <- OAVC fast
        denom = float(c.neurons.in_syn_l2.iloc[1])                                            # |W| l2 of Mi1's inputs (10)
        self.assertAlmostEqual(float(ol.W_slow_rs[0].to_dense()[0, 0]), 20.0 * DEFAULT_RECEPTOR_GAIN["mid"] / denom, places=6)
        # octopamine cell at 100 Hz (s = 1), dark: the Mi1 tone relaxes to scale x gain_fb x w with scale = gain tau / tau_syn
        rad = torch.zeros(1, 1, 4); rates = torch.zeros(1, 4); rates[0, 2] = 100.0
        for _ in range(1000):                                 # 1 s = 20 tau: within 1e-8 of the steady state
            drive = ol.step_frame(rad, rates, 1.0)
        w = 20.0 * DEFAULT_RECEPTOR_GAIN["mid"] / denom
        g_inf = gain * tau / tau_syn * op.gain_fb * w
        self.assertAlmostEqual(float(ol.g_slow[0, 0]), g_inf, places=5)
        self.assertAlmostEqual(float(ol.rates()[0, 0]), op.baseline + g_inf, places=4)         # additive: r = b + tone
        self.assertGreater(float(drive[0, 3]), 0.0)                                            # ... which drives LC4
        # without the spec (the round-1 optic model) the same input does nothing
        for _ in range(1000):
            drive0 = ol0.step_frame(rad, rates, 1.0)
        self.assertEqual(float(ol0.rates()[0, 0]), op.baseline); self.assertEqual(float(drive0[0, 3]), 0.0)
        # 'gain' scales the synaptic input (none here: no rate change), 'threshold' shifts the output like 'additive'
        olg = optic_mod.OpticLobe(c, retina, op, device="cpu", receptor=rs, receptor_gain=DEFAULT_RECEPTOR_GAIN,
                                  slow=SlowSpec("gain", {"monoamine": gain}, {"monoamine": tau}, tau_syn, 7.0))
        olt = optic_mod.OpticLobe(c, retina, op, device="cpu", receptor=rs, receptor_gain=DEFAULT_RECEPTOR_GAIN,
                                  slow=SlowSpec("threshold", {"monoamine": gain}, {"monoamine": tau}, tau_syn, 7.0))
        for _ in range(1000):
            olg.step_frame(rad, rates, 1.0); olt.step_frame(rad, rates, 1.0)
        self.assertAlmostEqual(float(olg.g_slow[0, 0]), g_inf, places=5); self.assertEqual(float(olg.rates()[0, 0]), op.baseline)
        self.assertAlmostEqual(float(olt.rates()[0, 0]), op.baseline + g_inf, places=4); self.assertAlmostEqual(float(olt.v[0, 0]), 0.0, places=5)
        # a lit photoreceptor under the 'gain' tone: the (inhibitory) histamine input is scaled by 1 + tone / (1 - b)
        lit = torch.full((1, 1, 4), 1.0)
        for o in (olg, ol0):
            o.reset(); o.relax()
        for _ in range(200):
            olg.step_frame(lit, rates, 1.0); ol0.step_frame(lit, torch.zeros(1, 4), 1.0)
        # after the onset transient both are at their operating points again (contrast adapts) but the tone persists in olg
        self.assertGreater(float(olg.g_slow[0, 0]), 0.0)
        olg.reset(); olg.relax(); ol0.reset(); ol0.relax()
        ol.reset(); self.assertEqual(float(ol.g_slow.abs().sum()), 0.0); self.assertEqual(float(ol.g_slow_cls.abs().sum()), 0.0)


# ---- round-2 profile-selection rule (scripts/build_receptor_table.select_profile) --------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_receptor_table as brt  # noqa: E402


def _cands(*pairs):
    return [{"source": s, "net": n} for s, n in pairs]


class SelectProfileTests(unittest.TestCase):
    """A silencing ('none') stands only if every source agrees; a single-nucleus 'none' never outranks a whole-cell 'on'."""

    def test_primary_with_group_on_decides(self):
        self.assertEqual(brt.select_profile(_cands(("fca2022", "-1"), ("davis2020", "none"))), (0, "primary"))
        self.assertEqual(brt.select_profile(_cands(("davis2020", "+1"), ("ozel2021", "none"), ("davie2018", "-1"))), (0, "primary"))

    def test_all_none_is_silenced_only_with_two_sources(self):
        self.assertEqual(brt.select_profile(_cands(("davis2020", "none"), ("ozel2021", "none"), ("fca2022", "none"))), (0, "primary"))
        self.assertEqual(brt.select_profile(_cands(("davis2020", "none"), ("fca2022", "none"))), (0, "primary"))
        # one profile's 'none' (Özel cluster 163 -> Pm5 / Pm6, Rdl P(on) 0.37) is not enough to silence
        self.assertEqual(brt.select_profile(_cands(("davie2018", "none"))), (0, brt.NONE_SINGLE))
        self.assertEqual(brt.select_profile(_cands(("ozel2021", "none"))), (0, brt.NONE_SINGLE))
        self.assertEqual(brt.select_profile(_cands(("ozel2021", "none")), min_sources=1), (0, "primary"))
        self.assertEqual(brt.MIN_SOURCES_TO_SILENCE, 2)

    def test_single_nucleus_none_yields_to_whole_cell_on(self):
        # Tm5a / Tm5b: Davie 'Tm5ab' (fuzzy, HisCl off) vs Özel 'PR'-independent class row with ort on
        self.assertEqual(brt.select_profile(_cands(("davie2018", "none"), ("ozel2021", "-1"))), (1, "group_on_override"))
        # the FIRST whole-cell source with the group on is taken, single-nucleus 'on' rows are skipped
        self.assertEqual(brt.select_profile(_cands(("fca2022", "none"), ("davie2018", "mixed"), ("kurmangaliyev2020", "+1"),
                                                   ("ozel2021", "-1"))), (2, "group_on_override"))

    def test_contested_none_falls_back_to_prior(self):
        # whole-cell 'none' against any 'on' (R7y glutamate: Davis R7 off vs Davie 'Photoreceptors' mixed)
        self.assertEqual(brt.select_profile(_cands(("davis2020", "none"), ("davie2018", "mixed"))), (0, brt.NONE_CONTESTED))
        self.assertEqual(brt.select_profile(_cands(("davis2020", "none"), ("ozel2021", "-1"))), (0, brt.NONE_CONTESTED))
        # single-nucleus 'none' against a single-nucleus 'on' only
        self.assertEqual(brt.select_profile(_cands(("fca2022", "none"), ("davie2018", "-1"))), (0, brt.NONE_CONTESTED))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            brt.select_profile([])

    def test_source_classes_partition_the_priority_list(self):
        srcs = set(brt.SOURCE_PRIORITY)
        self.assertEqual(brt.WHOLE_CELL_SOURCES | brt.SINGLE_NUCLEUS_SOURCES, srcs)
        self.assertFalse(brt.WHOLE_CELL_SOURCES & brt.SINGLE_NUCLEUS_SOURCES)


class ContestFlipTests(unittest.TestCase):
    """Round-3 symmetric rule (scripts/build_receptor_table.contest_flip): a flip contradicted by another profiled
    source at the prior's sign falls back to NT_SIGN ('any'); 'majority' only when the contradictors outnumber the
    sources agreeing with the flip; 'off' never."""

    TM9 = _cands(("davis2020", "+1"), ("kurmangaliyev2020", "-1"), ("fca2022", "-1"), ("davie2018", "-1"))

    def test_any_contests_on_one_contradictor(self):
        # Tm9 glutamate under abs: Davis +1 (KaiR1D 107 TPM, GluClalpha 0) vs three exact sources at -1
        self.assertEqual(brt.contest_flip(self.TM9, 0, -1), (brt.FLIP_CONTESTED, ["kurmangaliyev2020", "fca2022", "davie2018"]))
        # L1: Davis +1 vs Davie -1 alone
        self.assertEqual(brt.contest_flip(_cands(("davis2020", "+1"), ("davie2018", "-1")), 0, -1), (brt.FLIP_CONTESTED, ["davie2018"]))
        self.assertEqual(brt.FLIP_RULE_DEFAULT, "any")

    def test_majority_needs_more_contradictors_than_agreers(self):
        self.assertEqual(brt.contest_flip(self.TM9, 0, -1, "majority"), (brt.FLIP_CONTESTED_MAJORITY, ["kurmangaliyev2020", "fca2022", "davie2018"]))
        one_each = _cands(("davis2020", "+1"), ("ozel2021", "+1"), ("davie2018", "-1"))
        self.assertEqual(brt.contest_flip(one_each, 0, -1, "majority"), (None, ["davie2018"]))   # 1 vs 1: the flip stands
        self.assertEqual(brt.contest_flip(one_each, 0, -1, "any"), (brt.FLIP_CONTESTED, ["davie2018"]))
        two_v_one = _cands(("davis2020", "+1"), ("ozel2021", "-1"), ("davie2018", "-1"))
        self.assertEqual(brt.contest_flip(two_v_one, 0, -1, "majority"), (brt.FLIP_CONTESTED_MAJORITY, ["ozel2021", "davie2018"]))

    def test_uncontested_mixed_none_and_prior_are_left_alone(self):
        # T1 glutamate under abs: Kurmangaliyev +1, the other three 'none' -> the flip stands
        self.assertEqual(brt.contest_flip(_cands(("kurmangaliyev2020", "+1"), ("ozel2021", "none"), ("fca2022", "none")), 0, -1), (None, []))
        self.assertEqual(brt.contest_flip(_cands(("davis2020", "+1"), ("ozel2021", "mixed")), 0, -1), (None, []))
        self.assertEqual(brt.contest_flip(_cands(("davis2020", "-1"), ("ozel2021", "+1")), 0, -1), (None, []))   # not a flip
        self.assertEqual(brt.contest_flip(_cands(("davis2020", "mixed"), ("ozel2021", "-1")), 0, -1), (None, []))
        self.assertEqual(brt.contest_flip(_cands(("davis2020", "none"), ("ozel2021", "-1")), 0, -1), (None, []))
        self.assertEqual(brt.contest_flip(_cands(("davis2020", "+1")), 0, 0), (None, []))                       # monoamine prior 0

    def test_symmetric_case_and_selected_index(self):
        # a -1 on a +1 transmitter is the mirror image
        self.assertEqual(brt.contest_flip(_cands(("davis2020", "-1"), ("ozel2021", "+1")), 0, +1), (brt.FLIP_CONTESTED, ["ozel2021"]))
        # the selected candidate need not be the first (group_on_override); the skipped 'none' does not count
        self.assertEqual(brt.contest_flip(_cands(("davie2018", "none"), ("ozel2021", "+1"), ("fca2022", "-1")), 1, -1),
                         (brt.FLIP_CONTESTED, ["fca2022"]))

    def test_off_and_bad_rule(self):
        self.assertEqual(brt.contest_flip(self.TM9, 0, -1, "off"), (None, []))
        with self.assertRaises(ValueError):
            brt.contest_flip(self.TM9, 0, -1, "sometimes")
        self.assertEqual(brt.FAST_FALLBACK, brt.NONE_FALLBACK + brt.FLIP_FALLBACK)
        self.assertEqual(set(brt.FLIP_FALLBACK), {"flip_contested", "flip_contested_majority"})


@unittest.skipUnless(cn.RECEPTOR_TABLE.exists(), "receptor table absent")
class ShippedTableRuleTests(unittest.TestCase):
    """The shipped receptors_by_type.csv obeys the round-2 rule."""

    @classmethod
    def setUpClass(cls):
        cls.rt = cn.read_receptor_table()
        cls.rt = cls.rt[~cls.rt.malecns_type.astype(str).str.startswith("<")]

    def _row(self, t, nt):
        r = self.rt[(self.rt.malecns_type == t) & (self.rt.transmitter == nt)]
        self.assertEqual(len(r), 1, (t, nt))
        return r.iloc[0]

    def test_silenced_rows_have_every_source_at_none(self):
        classical = self.rt[self.rt.transmitter.isin(brt.CLASSICAL)]
        z = classical[classical.fast_sign == 0]
        self.assertGreater(len(z), 0)
        for _, r in z.iterrows():
            self.assertEqual(r.fast_net, "none", (r.malecns_type, r.transmitter))
            alts = [a for a in str(r.alt_sources).split(";") if a]
            for a in alts:
                self.assertIn("fast=none", a, (r.malecns_type, r.transmitter, a))
            self.assertGreaterEqual(len(alts) + 1, brt.MIN_SOURCES_TO_SILENCE, (r.malecns_type, r.transmitter))
        # and a contested or single-source none never silences
        nc = classical[classical.fast_net.isin(brt.NONE_FALLBACK)]
        self.assertGreater(len(nc), 0)
        self.assertTrue((nc.fast_sign != 0).all())
        for _, r in nc.iterrows():
            self.assertEqual(int(r.fast_sign), int(cn.NT_SIGN[r.transmitter]))
            self.assertEqual(r.fast_gain_class, "none")
        single = classical[classical.fast_net == brt.NONE_SINGLE]
        self.assertTrue((single.n_sources == 1).all())

    def test_named_pairs_of_the_verification_record(self):
        for t in ("Tm5a", "Tm5b"):                       # R7 -> Tm5a/b: Özel class profile with ort on replaces Davie 'Tm5ab'
            r = self._row(t, "histamine")
            self.assertEqual(int(r.fast_sign), -1); self.assertTrue(str(r.fast_selection).startswith("group_on_override"))
        r = self._row("Mi1", "histamine")                # R8 -> Mi1: every source has HisCl / ort off -> stays silenced
        self.assertEqual(int(r.fast_sign), 0); self.assertEqual(r.fast_selection, "primary")
        for t in ("R7y", "R7p", "R8y", "R8p"):           # glutamate / GABA onto photoreceptors: contested -> NT_SIGN
            for nt in ("glutamate", "gaba"):
                self.assertEqual(int(self._row(t, nt).fast_sign), -1, (t, nt))
        for t in ("Pm5", "Pm6"):                         # Özel class row is the only source: single-source none -> NT_SIGN
            r = self._row(t, "gaba")
            self.assertEqual(int(r.fast_sign), -1); self.assertEqual(r.fast_net, brt.NONE_SINGLE)

    def test_kaird1_is_not_the_old_alias_and_kurmangaliyev_is_present(self):
        self.assertIn("kurmangaliyev2020", set(self.rt.source))
        for t in ("T4a", "T5a"):
            self.assertEqual(self._row(t, "acetylcholine").source, "kurmangaliyev2020")

    @staticmethod
    def _alt_nets(row, field):
        """The other candidates' net calls for one variant field ('fast', 'abs', 'nonmda', 'slow') from alt_sources."""
        out = []
        for a in str(row.alt_sources).split(";"):
            if not a or a == "nan":
                continue
            kv = dict(kv.split("=") for kv in a.split(":", 1)[1].split(","))
            out.append(kv[field])
        return out

    def test_contested_flips_fall_back_to_prior(self):
        """Round-3 rule on the shipped table (built with --flip-rule any): every flip-fallback row carries the prior with
        gain class none, and no surviving flip has another source at the prior's sign under the same variant."""
        classical = self.rt[self.rt.transmitter.isin(brt.CLASSICAL)]
        self.assertIn("flip_contested", self.rt.columns)
        seen = 0
        for col, sgn, gcls, field in [("fast_net", "fast_sign", "fast_gain_class", "fast"),
                                      ("fast_net_abs", "fast_sign_abs", "fast_gain_class_abs", "abs"),
                                      ("fast_net_nonmda", "fast_sign_nonmda", "fast_gain_class_nonmda", "nonmda")]:
            fb = classical[classical[col].isin(brt.FLIP_FALLBACK)]
            self.assertGreater(len(fb), 0, col)
            seen += len(fb)
            for _, r in fb.iterrows():
                prior = int(cn.NT_SIGN[r.transmitter])
                self.assertEqual(int(r[sgn]), prior, (r.malecns_type, r.transmitter, col))
                self.assertEqual(r[gcls], "none", (r.malecns_type, r.transmitter, col))
                self.assertIn(f"{field if field != 'fast' else 'class'}:", str(r.flip_contested), (r.malecns_type, r.transmitter, col))
                self.assertIn(f"{prior:+d}", self._alt_nets(r, field), (r.malecns_type, r.transmitter, col))
            # surviving flips: no other source at the prior's sign under this variant
            for _, r in classical[classical[col].isin(["+1", "-1"])].iterrows():
                prior = int(cn.NT_SIGN[r.transmitter])
                if int(r[col]) != prior:
                    self.assertNotIn(f"{prior:+d}", self._alt_nets(r, field), (r.malecns_type, r.transmitter, col))
        self.assertGreater(seen, 0)
        # a row not contested in any variant has an empty flip_contested
        empty = classical[~classical.fast_net.isin(brt.FLIP_FALLBACK) & ~classical.fast_net_abs.isin(brt.FLIP_FALLBACK)
                          & ~classical.fast_net_nonmda.isin(brt.FLIP_FALLBACK)]
        self.assertTrue(empty.flip_contested.fillna("").eq("").all())

    def test_named_contested_flips_of_round_3(self):
        r = self._row("Tm9", "glutamate")                # Davis KaiR1D 107 TPM vs GluClalpha 0; Kurmangaliyev / FCA / Davie exact at -1
        self.assertEqual(r.source, "davis2020")
        for col, sgn in [("fast_net", "fast_sign"), ("fast_net_abs", "fast_sign_abs"), ("fast_net_nonmda", "fast_sign_nonmda")]:
            self.assertEqual(r[col], brt.FLIP_CONTESTED, col); self.assertEqual(int(r[sgn]), -1, col)
        self.assertEqual(set(str(r.flip_contested).split(";")[1].split(":")[1].split("+")), {"kurmangaliyev2020", "fca2022", "davie2018"})
        r = self._row("L1", "glutamate")                 # Davis +1 vs Davie -1 (class), Kurmangaliyev + Davie -1 (abs)
        self.assertEqual(r.fast_net_abs, brt.FLIP_CONTESTED); self.assertEqual(int(r.fast_sign_abs), -1)
        per_variant = dict(kv.split(":") for kv in str(r.flip_contested).split(";"))
        self.assertEqual(set(per_variant["abs"].split("+")), {"kurmangaliyev2020", "davie2018"})
        r = self._row("T1", "glutamate")                 # Kurmangaliyev +1, the others 'none': an uncontested flip survives
        self.assertEqual(r.fast_net_abs, "+1"); self.assertEqual(int(r.fast_sign_abs), +1); self.assertTrue(pd.isna(r.flip_contested) or r.flip_contested == "")
        for t in ("Tm5a", "Tm5b", "Mi1", "Pm5", "Pm6"):  # the round-2 named pairs are untouched
            for nt in ("histamine", "gaba"):
                self.assertNotIn(self._row(t, nt).fast_net, brt.FLIP_FALLBACK, (t, nt))


if __name__ == "__main__":
    unittest.main()
