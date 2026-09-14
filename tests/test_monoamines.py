"""Thread monoamines (docs/audits/monoamine_slow_term.md): the arm table of scripts/probe_monoamines.py leaves the shipped
path untouched and puts every treatment on the same fast weights; the steady-tone formula of the magnitude bracket
(scripts/build_monoamine_tables.py) is the one the Brain integrates; the sign rule the coverage table cites is the
receptor table builder's. CPU, synthetic graphs (tests/test_receptor_model.py's), no cache."""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))

from flyverse import brain as br                                                       # noqa: E402
from flyverse.brain import Brain, LIFParams, _shaped_weights, _slow_spec, _slow_weights  # noqa: E402
from flyverse.connectome import receptor_signs                                       # noqa: E402
from test_receptor_model import graph, small_table, two_neuron_graph                 # noqa: E402
import probe_monoamines as pm                                                        # noqa: E402
import build_monoamine_tables as bmt                                                 # noqa: E402


class ArmTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = Path(self.tmp.name) / "receptors.csv"
        with open(self.table, "w", encoding="utf-8") as f:
            f.write("# test table\n")
            small_table().to_csv(f, index=False)
        self.c, self.cnt = graph()
        self.counts = self.cnt[self.c.W.tocoo().row, self.c.W.tocoo().col]

    def tearDown(self):
        self.tmp.cleanup()

    def test_off_arm_is_the_shipped_default_and_no_default_moved(self):
        """The reference arm is LIFParams() itself, and this thread changed no LIFParams default."""
        self.assertEqual(pm.ARMS["off"]["lif"], {})
        self.assertEqual(pm.lif_for("off"), LIFParams())
        d = LIFParams()
        self.assertEqual(d.receptor_model, "sign"); self.assertEqual(d.receptor_net_rule, "abs")
        self.assertEqual(d.slow_gain, 0.02); self.assertEqual(d.slow_tau_ms, 200.0); self.assertEqual(d.slow_mode, "additive")
        self.assertIsNone(_slow_spec(d))                                              # no slow term on the shipped path
        # patch_lif({}) (the off arm's job) builds the very same LIFParams
        L0 = pm.patch_lif({})
        try:
            self.assertEqual(br.LIFParams(), d)
        finally:
            br.LIFParams = L0

    def test_treatment_arms_share_the_fast_weights_and_differ_only_in_the_slow_matrix(self):
        """Every treatment arm's fast weights equal the off arm's entry for entry (receptor_gain 1/1/1 under 'full' =
        the 'sign' weights); the monoamine slow matrix scales with the arm's gain; the off arm has none."""
        rs_off = receptor_signs(self.c, table_path=self.table, net_rule="abs")
        p_off = LIFParams(receptor_table=str(self.table), event_driven=False)
        W_off = _shaped_weights(self.c, p_off, rs_off)
        for arm, a in pm.ARMS.items():
            if arm == "off":
                continue
            p = LIFParams(receptor_table=str(self.table), event_driven=False, **a["lif"])
            rs = receptor_signs(self.c, table_path=self.table, net_rule="abs", counts=self.counts)
            W = _shaped_weights(self.c, p, rs)
            np.testing.assert_array_equal(W.indices, W_off.indices); np.testing.assert_array_equal(W.data, W_off.data)
            spec = _slow_spec(p)
            self.assertIsNotNone(spec); self.assertEqual(spec.mode, a["mode"]); self.assertEqual(spec.gain, {"monoamine": float(a["gain"])})
            S = _slow_weights(self.c, p, rs, spec)["monoamine"]
            self.assertGreater(S.nnz, 0)                                              # dopamine -> TA carries a slow +
            b = Brain(self.c, p, device="cpu", receptor=rs)
            self.assertTrue(b._slow_active); self.assertFalse(b.cuda)
            # per synapse per spike the tone jumps by gain x w_syn (x fan-in scale): Brain's slow matrix = S x scale x w_syn x gain
            Sb = b._W_slow_cpu["monoamine"]
            expect = S.multiply(b.input_scale[:, None]).tocsr(); expect.data = expect.data * np.float32(p.w_syn * a["gain"])
            np.testing.assert_allclose(Sb.data, expect.data, rtol=1e-6)
            # the benchmark flags name the same mode and scale
            self.assertIn(a["mode"], a["bench"]); self.assertIn(str(a["gain"]), a["bench"]); self.assertIn("1,1,1", a["bench"])
        b_off = Brain(self.c, p_off, device="cpu")
        self.assertIsNone(b_off.W_slow)

    def test_bracket_formula_is_what_the_brain_integrates(self):
        """The steady tone used by build_monoamine_tables' bracket -- slow_gain x w_syn x load x R x tau -- is the mean g_slow
        the Brain reaches with the presynaptic monoamine cell at R Hz (dopamine -> TA, count 20, mid gain class = 1 here)."""
        c = two_neuron_graph()
        rs = receptor_signs(c, table_path=self.table, counts=np.array([20.0], np.float32))
        gain = 0.2; tau = 200.0; R = 100.0
        p = LIFParams(receptor_model="full", receptor_table=str(self.table), receptor_gain=dict(pm.UNIT_GAIN), slow_mode="additive", slow_gain=gain,
                      slow_tau_ms=tau, event_driven=False, input_norm_alpha=0.0, path_gain=[], type_path_gain=[], adapt_jump=0.0, same_type_gain=1.0)
        b = Brain(c, p, device="cpu", receptor=rs)
        torch.manual_seed(0)
        b.set_poisson([0], R)
        b.step(int(round(4 * tau / p.dt)))                                               # 800 ms: the tone has relaxed
        tones = []
        for _ in range(int(round(4000.0 / p.dt))):                                      # 4 s of samples
            b.step(1); tones.append(float(b.g_slow[0, 1]))
        expect = gain * p.w_syn * 20.0 * R * tau / 1000.0                               # 0.2 x 0.275 x 20 x 100 Hz x 0.2 s = 22 mV
        self.assertAlmostEqual(expect, 22.0, places=6)
        self.assertAlmostEqual(float(np.mean(tones)) / expect, 1.0, delta=0.15)         # Poisson scatter over 400 spikes


class GuardAndSignRuleTests(unittest.TestCase):
    def test_rest_guard_is_ten_times_the_rest_bound(self):
        self.assertEqual(pm.REST_BOUND, 5.0)
        import argparse
        ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
        # the record parser's default abort_rest is 10 x the bound (read through main()'s parser)
        argv = sys.argv; sys.argv = ["probe_monoamines.py", "record", "--arm", "off", "--out", "x"]
        try:
            import inspect
            src = inspect.getsource(pm.main)
            self.assertIn("default=10 * REST_BOUND", src)
        finally:
            sys.argv = argv

    def test_group_sign_rule_matches_the_table_builder(self):
        """Every receptor gene of GROUP_SIGN (the audit's sign rule) is in build_receptor_table.RECEPTOR_GROUPS' slow group
        of that transmitter with the same sign, and no builder gene is missing."""
        import build_receptor_table as brt
        for nt, genes in bmt.GROUP_SIGN.items():
            builder = {g: sign for sign, _name, gs in brt.RECEPTOR_GROUPS[nt]["slow"] for g in gs}
            self.assertEqual(set(builder), set(genes), nt)
            for g, (sign, _cite) in genes.items():
                self.assertEqual(builder[g], sign, f"{nt} {g}")
            self.assertEqual(brt.RECEPTOR_GROUPS[nt]["fast"], [])                     # monoamines have no fast group

    def test_taste_replay_is_registered_and_cpu_by_default(self):
        """The CPU replay of the benchmark's taste section (probe_monoamines.py taste) exists, defaults to the CPU and to
        the benchmark's own 600 ms / seed 0, and names MN9 (the section's readout) among its reported types."""
        import inspect
        src = inspect.getsource(pm.main)
        self.assertIn('sub.add_parser("taste")', src)
        self.assertIn('t.add_argument("--device", default="cpu")', src)
        self.assertIn('t.add_argument("--ms", type=float, default=600.0)', src)
        self.assertIn("MN9", pm.TASTE_NAMED); self.assertIn("GNG175", pm.TASTE_NAMED)
        self.assertEqual(set(pm.MONO_NTS), set(bmt.MONO))

    def test_anchor_brackets_are_ordered_and_named(self):
        for a in bmt.ANCHORS:
            self.assertLessEqual(abs(a["tone_mv_lo"]), abs(a["tone_mv_hi"]))
            self.assertIn(a["kind"], br.SLOW_MODES); self.assertIn(a["transmitter"], bmt.MONO)
            self.assertTrue(a["source"])
        gains = [pm.ARMS[k]["gain"] for k in ("add_low", "add_mid", "add_high")]
        self.assertEqual(gains, sorted(gains)); self.assertEqual(gains[0], LIFParams().slow_gain)


if __name__ == "__main__":
    unittest.main()
