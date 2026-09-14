"""Opt-in CUDA correctness checks; no connectome dataset or timing claims."""
import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_interp import graph
from test_surrogate import optic_graph
from flyverse.brain import LIFParams
from flyverse.fly import FlyBrain
from flyverse.modules import FunctionModule, TorchModule


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("FLYVERSE_EXTENSION_CUDA_TESTS") == "1",
                     "set FLYVERSE_EXTENSION_CUDA_TESTS=1 on a CUDA test worker")
class ExtensionCudaTests(unittest.TestCase):
    def test_hooks_and_modules_outside_capture(self):
        for native in (False, True):
            with self.subTest(native=native):
                p = LIFParams(receptor_model=None, adapt_by_type={}, event_driven=native)
                brains = [FlyBrain(graph().subset([3, 4, 5]), device="cuda", batch=2, seed=42, lif_params=p,
                                   cuda_graphs=capture, cuda_kernels=native) for capture in (False, True)]
                counts = [[], []]
                for i, fb in enumerate(brains):
                    def hook(f, t, calls=counts[i]):
                        calls.append(t)
                        f.brain.set_poisson([0], torch.full((2, 1), 300., device=f.device))
                    fb.add_hook(hook, name="forcing")
                    fb.attach(FunctionModule({}, {"out": [1]}, lambda dt, inputs: {"out": torch.tensor([[20.], [-5.]], device="cuda")},
                                             name="drive", channel_out="drive_mv"))
                for _ in range(10):
                    for fb in brains:
                        fb.step()
                    for name in FlyBrain.BRAIN_TENSORS:
                        torch.testing.assert_close(getattr(brains[0].brain, name), getattr(brains[1].brain, name), rtol=1e-6, atol=1e-5)
                self.assertEqual(counts[0], counts[1])
                self.assertEqual(len(counts[1]), 10)
                self.assertTrue(brains[1]._graphs)
                saved = brains[1].state_dict()
                brains[1].step(); expected = brains[1].brain.rate.clone()
                brains[1].load_state_dict(saved); brains[1].step()
                torch.testing.assert_close(brains[1].brain.rate, expected, rtol=0, atol=0)

    def test_vision_encoder_updates_captured_input(self):
        p = LIFParams(receptor_model=None, event_driven=False)
        brains = [FlyBrain(optic_graph(), device="cuda", batch=2, seed=4, lif_params=p, cuda_graphs=capture,
                           cuda_kernels=False) for capture in (False, True)]
        for fb in brains:
            net = torch.nn.Linear(4, 1)
            with torch.no_grad():
                net.weight.fill_(.25); net.bias.zero_()
            fb.attach(TorchModule({}, {}, net, name="encoder", kind="sensor", boundary="vision", output_sizes={"intensity": 1}))
        for k in range(6):
            for fb in brains:
                fb.vision(torch.full((2, 1, 4), .1 if k % 2 else .8, device="cuda")); fb.step()
            torch.testing.assert_close(brains[0].optic.v, brains[1].optic.v, rtol=1e-6, atol=1e-6)

    def test_surrogate_module_gradient(self):
        fb = FlyBrain(graph().subset([3, 4, 5]), device="cuda", batch=2,
                      lif_params=LIFParams(receptor_model=None, surrogate_grad=True))
        net = torch.nn.Linear(1, 1)
        with torch.no_grad():
            net.weight.zero_(); net.bias.fill_(30.)
        fb.attach(TorchModule({"in": [0]}, {"out": [0]}, net, "drive_mv"))
        fb.step(20)
        fb.brain.rate.sum().backward()
        self.assertGreater(float(net.bias.grad.abs().sum()), 0.)
        fb.detach_state()


if __name__ == "__main__":
    unittest.main()
