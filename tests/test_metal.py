"""The Metal kernels (flyverse/metal.py) against the torch implementations they replace. Runs on a Mac
with MPS; skipped elsewhere."""
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from flyverse import metal
from flyverse.brain import Brain, LIFParams
from test_control import graph

HAVE = torch.backends.mps.is_available() and metal.available()


@unittest.skipUnless(HAVE, "Metal kernels need an MPS device")
class SparseTests(unittest.TestCase):
    def test_matvec_matches_scipy(self):
        rng = np.random.default_rng(0)
        for rows, cols, density in ((300, 200, 0.05), (1, 50, 0.5), (64, 64, 0.0), (1000, 10, 0.3)):
            M = sp.random(rows, cols, density=density, format="csr", dtype=np.float32, random_state=rng)
            m = metal.MetalCSR(M, "mps")
            for B in (1, 4):
                x = torch.rand(B, cols)
                y = m.matvec(x.to("mps")).cpu()
                ref = torch.from_numpy(np.asarray(M @ x.numpy().T).T.astype(np.float32))
                torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-6)
            x1 = torch.rand(cols)
            torch.testing.assert_close(m.matvec(x1.to("mps")).cpu(), torch.from_numpy(np.asarray(M @ x1.numpy()).astype(np.float32)), rtol=1e-5, atol=1e-6)

    def test_matvec_rejects_bad_input(self):
        m = metal.MetalCSR(sp.eye(8, format="csr", dtype=np.float32), "mps")
        with self.assertRaises(ValueError):
            m.matvec(torch.zeros(2, 16, device="mps")[:, :8])   # non-contiguous
        with self.assertRaises(ValueError):
            m.matvec(torch.zeros(2, 5, device="mps"))            # wrong width


@unittest.skipUnless(HAVE, "Metal kernels need an MPS device")
class BrainTests(unittest.TestCase):
    def pair(self, batch=1, params=None, seed=3):
        c = graph()
        return (Brain(c, params, device="mps", seed=seed, batch=batch, metal_kernels=True),
                Brain(c, params, device="mps", seed=seed, batch=batch, metal_kernels=False))

    def assert_same(self, bm, bt, steps):
        for b in (bm, bt):
            b.record_activity = True
            b.step(steps)
        for name in ("spikes", "spike_counts", "refrac"):
            torch.testing.assert_close(getattr(bm, name), getattr(bt, name), rtol=0, atol=0)
        for name in ("v", "g", "rate", "adapt", "res", "spike_buf"):     # spike_buf = spikes * res: rounding
            torch.testing.assert_close(getattr(bm, name), getattr(bt, name), rtol=1e-5, atol=1e-4)
        self.assertGreater(float(bt.spike_counts.sum()), 0)

    def test_drive_and_depression(self):
        bm, bt = self.pair()                    # ORN_DM1 has depressing synapses by default
        for b in (bm, bt):
            b.set_drive([0], 12.0)
        self.assert_same(bm, bt, 400)

    def test_poisson_shares_the_generator(self):
        bm, bt = self.pair(batch=3)
        for b in (bm, bt):
            b.set_poisson([0, 1], np.array([[200.0, 50.0], [0.0, 0.0], [400.0, 0.0]]))
        self.assert_same(bm, bt, 300)
        self.assertEqual(float(bm.spike_counts[1].sum()), 0.0)

    def test_frozen_and_no_adaptation(self):
        bm, bt = self.pair(params=LIFParams(adapt_jump=0.0, std_u_by_type={}))
        for b in (bm, bt):
            b.freeze([2]); b.set_drive([0, 3], 15.0)
        self.assert_same(bm, bt, 200)
        self.assertEqual(float(bm.spike_counts[0, 2]), 0.0)

    def test_reset_and_readouts(self):
        bm, _ = self.pair()
        bm.set_drive([0], 12.0); bm.step(100)
        self.assertGreater(bm.mean_rate([0]), 0.0)
        bm.reset()
        self.assertEqual(float(bm.rate.abs().sum()), 0.0)
        self.assertEqual(bm.mean_rate([0]), 0.0)

    def test_explicit_request_needs_event_driven(self):
        with self.assertRaises(ValueError):
            Brain(graph(), LIFParams(event_driven=False), device="mps", metal_kernels=True)
        with self.assertRaises(ValueError):
            Brain(graph(), device="cpu", metal_kernels=True)
        self.assertFalse(Brain(graph(), device="cpu").metal)


@unittest.skipUnless(HAVE, "Metal kernels need an MPS device")
class OpticKernelTests(unittest.TestCase):
    def test_substep_matches_torch_formula(self):
        torch.manual_seed(0)
        B, n = 2, 500
        v = torch.randn(B, n, device="mps") * 0.3; adapt = torch.rand(B, n, device="mps") * 0.1
        y = torch.randn(B, n, device="mps"); pr = torch.randn(B, n, device="mps"); sk = torch.randn(B, n, device="mps") * 0.1
        a = torch.rand(n, device="mps") * 0.9; bvec = (torch.rand(n, device="mps") > 0.3).float() * 0.5
        gain_rr, adapt_gain, a_ad = 1.0, 1.0, 0.9975
        dr0 = (v + bvec).clamp(0, 1) - bvec
        inp = gain_rr * y + pr - adapt_gain * adapt; inp = inp + sk
        v_ref = inp + (v - inp) * a[None]; adapt_ref = dr0 + (adapt - dr0) * a_ad
        dr_ref = (v_ref + bvec).clamp(0, 1) - bvec
        dr = torch.empty_like(v)
        metal.optic_dr(v, bvec, dr)
        torch.testing.assert_close(dr, dr0, rtol=0, atol=1e-6)
        vm, am = v.clone(), adapt.clone()
        metal.optic_substep(vm, am, dr, y, pr, sk, a, bvec, gain_rr, adapt_gain, a_ad)
        torch.testing.assert_close(vm, v_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(am, adapt_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(dr, dr_ref, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
