"""Short-window gradients and backend guards, entirely on CPU.

The differentiable paths (`Brain._step_surrogate`, `OpticLobe._step_frame_grad`) are a second spelling of the
shipped update, so they are pinned against the inference paths bit for bit with the optic per-stream hooks off,
and the optic grad substep is required to route through the SAME `OpticLobe._recurrent` / `_output` as inference
so that an active hook cannot silently vanish from the differentiable forward model (B2 of
docs/audits/extensibility_review.md)."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_interp import graph
from test_optic_hooks import hex_graph, plain_params
from flyverse import optic as optic_mod, retina as retina_mod
from flyverse.brain import Brain, LIFParams
from flyverse.fly import FlyBrain
from flyverse.modules import TorchModule


def params(**kwargs):
    return LIFParams(receptor_model=None, surrogate_grad=True, **kwargs)


def optic_graph():
    c = graph()
    c.neurons["hex1"] = np.nan; c.neurons["hex2"] = np.nan; c.neurons["hex_side"] = ""
    c.neurons.loc[0, ["hex1", "hex2", "hex_side"]] = [1., 1., "L"]
    return c


class SurrogateTests(unittest.TestCase):
    def test_drive_gradient_through_delayed_spikes_and_reset(self):
        b = Brain(graph().subset([3, 4, 5]), params(), device="cpu", batch=2)
        b.record_activity = True
        drive = torch.full((2, 1), 30., requires_grad=True)
        b.set_drive([0], drive)
        b.step(50)
        loss = b.rate[:, 1:].sum() + b.rate[:, 0].sum()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(drive.grad).all()))
        self.assertGreater(float(drive.grad.abs().sum()), 0.)
        b.reset([0])
        self.assertEqual(float(b.rate[0].sum()), 0.)
        self.assertFalse(b.rate.requires_grad)

    def test_torch_module_trains_through_flybrain(self):
        fb = FlyBrain(graph().subset([3, 4, 5]), device="cpu", lif_params=params())
        net = torch.nn.Linear(1, 1)
        with torch.no_grad():
            net.weight.fill_(0); net.bias.fill_(30)
        fb.attach(TorchModule({"rate": [0]}, {"drive": [0]}, net, "drive_mv"))
        fb.step(20)
        fb.brain.rate.sum().backward()
        self.assertGreater(float(net.bias.grad.abs().sum()), 0.)
        fb.detach_state()
        self.assertFalse(fb.brain.rate.requires_grad)
        saved = fb.state_dict()
        fb.step()
        fb.load_state_dict(saved)
        fb.step()

    def test_fast_sigmoid_forward_matches_hard_spiking(self):
        c = graph().subset([3, 4, 5])
        p = params(event_driven=False)
        a, b = Brain(c, p, device="cpu"), Brain(c, replace(p, surrogate_grad=False), device="cpu")
        for brain in (a, b):
            brain.set_drive([0], 30.)
            brain.record_activity = True
        for _ in range(60):
            a.step(); b.step()
            torch.testing.assert_close(a.spikes, b.spikes, rtol=0, atol=0)
        torch.testing.assert_close(a.rate, b.rate, rtol=0, atol=0)

    def test_clocked_gradients(self):
        b = Brain(graph().subset([3, 4, 5]), params(), device="cpu")
        b.set_clocks([1, 2, 2])
        drive = torch.tensor([[30.]], requires_grad=True)
        b.set_drive([0], drive); b.step(30)
        b.rate.sum().backward()
        self.assertGreater(float(drive.grad.abs().sum()), 0)

    def test_backend_guards(self):
        c = graph().subset([3, 4, 5])
        for kw in ({"cuda_kernels": True}, {"metal_kernels": True}, {"cuda_sparse": "warp"}):
            with self.assertRaisesRegex(ValueError, "surrogate_grad"):
                Brain(c, params(), device="cpu", **kw)
        with self.assertRaisesRegex(ValueError, "surrogate_grad"):
            FlyBrain(c, lif_params=params(), cuda_graphs=True, device="cpu")

    def test_optic_gradient_and_encoder(self):
        fb = FlyBrain(optic_graph(), device="cpu", lif_params=params())
        net = torch.nn.Sequential(torch.nn.Linear(4, 1), torch.nn.Softplus())
        encoder = fb.attach(TorchModule({}, {}, net, name="retina", kind="sensor", boundary="vision", output_sizes={"intensity": 1}))
        fb.vision(torch.ones(1, 4)); fb.step()
        fb.vision(torch.tensor([[.2, .3, .4, .5]])); fb.step()
        fb.optic.v.sum().backward()
        self.assertGreater(float(net[0].weight.grad.abs().sum()), 0)
        self.assertEqual(encoder.outputs["intensity"].shape, (1, 1))
        fb.reset(); fb.vision(torch.ones(1, 4)); fb.step()


class SurrogateMatchesInferenceTests(unittest.TestCase):
    """`_step_surrogate` / `_step_frame_grad` against `_step_inference` / `_step_frame_inference`, bit for bit.

    Both differentiable paths duplicate an update that lives elsewhere, so nothing but a test keeps them in step.
    A difference here is a real divergence of the forward model, not a tolerance question: assert equality.
    """
    def brains(self, clocks=None, **kwargs):
        p = LIFParams(receptor_model=None, surrogate_grad=True, event_driven=False, **kwargs)
        c = graph().subset([2, 3, 4, 5, 6, 7])
        pair = (Brain(c, p, device="cpu", batch=2, seed=5),
                Brain(c, replace(p, surrogate_grad=False), device="cpu", batch=2, seed=5))
        for b in pair:
            if clocks is not None:
                b.set_clocks(clocks)
            b.record_activity = True
            b.set_drive([0, 1], 30.)
            b.set_poisson([2], 120.)
        return pair

    def assert_same_state(self, a, b):
        for name in FlyBrain.BRAIN_TENSORS:
            x, y = getattr(a, name).detach(), getattr(b, name).detach()
            self.assertTrue(torch.equal(x, y), f"{name} differs between the surrogate and inference LIF")
        for k, v in a._acc.items():
            self.assertTrue(torch.equal(v.detach(), b._acc[k]), f"clock accumulator {k} differs")
        self.assertTrue(bool((a.gen.get_state() == b.gen.get_state()).all()), "the two paths consumed different RNG")

    def test_surrogate_lif_matches_inference_bitwise(self):
        # spike-frequency adaptation and short-term depression on, so the branches of the update are exercised
        a, b = self.brains(adapt_jump=1.0, std_u=0.2)
        for _ in range(80):
            a.step(); b.step()
        self.assertGreater(float(a.spike_counts.sum()), 0)
        self.assert_same_state(a, b)

    def test_surrogate_clocked_lif_matches_inference_bitwise(self):
        a, b = self.brains(clocks=[1, 2, 2, 1, 2, 2], adapt_jump=1.0, std_u=0.2)
        for _ in range(80):
            a.step(); b.step()
        self.assertGreater(float(a.spike_counts.sum()), 0)
        self.assert_same_state(a, b)

    # ---------------------------------------------------------------- the optic substep (B2)
    OPTIC_STATE = ("v", "adapt", "I_lp", "I_mean", "contrast", "delta_rate", "stream_adapt_state")

    @classmethod
    def setUpClass(cls):
        cls.c_hex = hex_graph()
        cls.r_hex = retina_mod.build_retina(cls.c_hex)

    def optic_run(self, params, *, surrogate, frames=6):
        """One deterministic radiance sequence through a fresh lobe; the per-frame drive and the final state."""
        lobe = optic_mod.OpticLobe(self.c_hex, self.r_hex, params, device="cpu", surrogate_grad=surrogate)
        lobe.relax()
        spikes = torch.zeros(1, self.c_hex.n)
        rng = np.random.default_rng(3)
        drives = []
        for _ in range(frames):
            radiance = torch.as_tensor(rng.uniform(.1, 1., (self.r_hex.n_columns, 4)), dtype=torch.float32)
            drives.append(lobe.step_frame(radiance, spikes, 10.).detach().clone())
        return lobe, torch.stack(drives)

    def assert_same_optic(self, params):
        grad, d_grad = self.optic_run(params, surrogate=True)
        ref, d_ref = self.optic_run(params, surrogate=False)
        self.assertTrue(torch.equal(d_grad, d_ref), "the differentiable optic frame writes a different drive")
        for name in self.OPTIC_STATE:
            x, y = getattr(grad, name).detach(), getattr(ref, name).detach()
            self.assertTrue(torch.equal(x, y), f"optic {name} differs between the grad and inference substep")
        return d_ref

    def test_optic_grad_substep_matches_inference_bitwise(self):
        """Hooks off: the two substeps must evaluate the identical expressions (the review measured maxdiff 0.0)."""
        for params in (plain_params(), optic_mod.OpticParams()):
            with self.subTest(params=params.pair_gain is not None):
                self.assertGreater(float(self.assert_same_optic(params).abs().max()), 0)

    def test_optic_stream_hooks_reach_the_grad_path(self):
        """B2: the grad substep used to be a hand copy that ignored the opt-in per-stream hooks -- an active hook
        moved the inference model and left the differentiable one at exactly 0, with no error. Both paths now go
        through `_recurrent` / `_output`, so the hooks act identically, and the hooked model really does differ
        from the unhooked one (otherwise this test would pass on a lobe that ignores the hooks twice over)."""
        hooked = plain_params(stream_rectify=[("^Mi1$", ".*", "pos")],
                              stream_adapt=[("^Mi1$", ".*", 50., 1.)],
                              spatial_suppress=[("^Mi1$", .5, 5.)])
        self.assertTrue(optic_mod.OpticLobe(self.c_hex, self.r_hex, hooked, device="cpu")._hooks)
        with_hooks = self.assert_same_optic(hooked)
        without_hooks = self.optic_run(plain_params(), surrogate=False)[1]
        self.assertGreater(float((with_hooks - without_hooks).abs().max()), 1.0)

    def test_optic_gradients_flow_with_stream_hooks(self):
        """The functional adaptation-state update of the hooked substep must stay differentiable."""
        hooked = plain_params(stream_rectify=[("^Mi1$", ".*", "pos")], stream_adapt=[("^Mi1$", ".*", 50., 1.)],
                              spatial_suppress=[("^Mi1$", .5, 5.)])
        lobe = optic_mod.OpticLobe(self.c_hex, self.r_hex, hooked, device="cpu", surrogate_grad=True)
        lobe.relax()
        # A CONSTANT radiance on a relaxed lobe is a degenerate gradient test: the lobe sits exactly at its operating
        # point (dr = 0 everywhere), the `pos` rectifier's gate has zero gradient at exactly 0, and on an exact CPU
        # build the whole gradient is 0.0 (it was only non-zero locally through CUDA-build float noise). Modulate the
        # radiance across frames so the contrast stage and the rectified stream are both away from their kinks.
        base = torch.full((self.r_hex.n_columns, 4), .5, requires_grad=True)
        spikes = torch.zeros(1, self.c_hex.n)
        for k in range(4):
            radiance = base * (1. + .5 * float(k % 2))          # 0.5 / 0.75 alternating: a non-zero contrast step
            drive = lobe.step_frame(radiance, spikes, 10.)
        drive.sum().backward()
        self.assertTrue(bool(torch.isfinite(base.grad).all()))
        self.assertGreater(float(base.grad.abs().sum()), 0.)
        lobe.reset()
        self.assertFalse(lobe.v.requires_grad)
        self.assertFalse(lobe.stream_adapt_state.requires_grad)


if __name__ == "__main__":
    unittest.main()
