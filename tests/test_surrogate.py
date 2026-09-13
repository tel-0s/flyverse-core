"""Short-window gradients and backend guards, entirely on CPU."""
import sys
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_interp import graph
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


if __name__ == "__main__":
    unittest.main()
