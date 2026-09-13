"""Extension contracts on tiny CPU graphs; no dataset or accelerator required."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_interp import graph
from flyverse.fly import FlyBrain
from flyverse.brain import LIFParams
from flyverse.modules import FunctionModule, TorchModule, ReadoutModule
from flyverse.nt_readout import NTChannel


def controller(batch=1, **kwargs):
    c = graph().subset([4, 5, 7])
    c.W.data[:] = 0
    return FlyBrain(c, batch=batch, device="cpu", seed=19,
                    lif_params=LIFParams(receptor_model=None), **kwargs)


def force(fb, t):
    fb.brain.set_poisson([0], 200.)


class ModuleTests(unittest.TestCase):
    def test_hook_isolation_and_remove_identity(self):
        fb, bare = controller(), controller()
        fb.add_hook(force, name="force")
        for _ in range(100):
            fb.step(); bare.step()
        self.assertGreater(float(fb.brain.spike_counts[0, 0]), 100)
        torch.testing.assert_close(fb.brain.rate[:, 1:], bare.brain.rate[:, 1:], rtol=0, atol=0)
        fb.remove_hook("force")
        pristine = controller().state_dict()
        fb.load_state_dict(pristine); bare.load_state_dict(pristine)
        for _ in range(100):
            fb.step(); bare.step()
        for name in FlyBrain.BRAIN_TENSORS:
            torch.testing.assert_close(getattr(fb.brain, name), getattr(bare.brain, name), rtol=0, atol=0)

    def test_hook_cadence_error_and_fraction(self):
        fb = controller(batch=2)
        calls = []
        fb.add_hook(lambda f, t: calls.append(("pre", t, f.B)), name="before")
        fb.add_hook(lambda f, t: calls.append(("post", t, f.B)), name="after", when="post")
        fb.step(30)
        self.assertEqual(calls, [("pre", 0, 2), ("post", 30, 2)])
        fb.step(.1)
        self.assertEqual(len(calls), 4)
        def fail(f, t):
            raise ValueError("bad input")
        fb.add_hook(fail, name="broken")
        with self.assertRaisesRegex(RuntimeError, "broken.*bad input"):
            fb.step()
        self.assertEqual(fb.t, 30)

    def test_previous_frame_native_batch_and_drive(self):
        fb = controller(batch=2)
        seen = []
        def fn(dt, inputs):
            seen.append((dt, inputs["in"].clone()))
            return {"out": torch.tensor([[200.], [0.]])}
        fb.attach(FunctionModule({"in": [0]}, {"out": [0]}, fn, name="input"))
        fb.step(30)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0][0], 10)
        self.assertEqual(seen[0][1].shape, (2, 1))
        self.assertEqual(float(seen[0][1].sum()), 0)
        self.assertGreater(float(seen[-1][1][0]), 0)
        self.assertEqual(float(fb.brain.rate[1].sum()), 0)
        fb.attach(FunctionModule({}, {"inhibit": [0]}, lambda dt, i: {"inhibit": torch.full((2, 1), -7.)},
                                 name="negative", channel_out="drive_mv"))
        fb.step()
        torch.testing.assert_close(fb.brain.drive[:, 0], torch.full((2,), -7.))

    def test_overlap_and_negative_poisson(self):
        fb = controller()
        m = lambda name: FunctionModule({}, {"out": [0]}, lambda dt, i: {"out": torch.tensor([[-10.]])}, name=name)
        fb.attach(m("a"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            fb.attach(m("b"))
        fb.step()
        self.assertEqual(float(fb.brain.poisson_p.sum()), 0)
        fb.detach("a")
        self.assertIsNone(fb._extensions)

    def test_pulse_expiry_preserves_module_forcing(self):
        fb = controller()
        fb.attach(FunctionModule({}, {"out": [0]}, lambda dt, i: {"out": torch.tensor([[100.]])}, name="held"))
        fb.stimulate([0], 200., 3)
        fb.step()
        self.assertAlmostEqual(float(fb.brain.poisson_p[0, 0]), .05)
        fb.detach("held")
        self.assertEqual(float(fb.brain.poisson_p.sum()), 0)

    def test_torch_module_checkpoint_and_provenance(self):
        from flyverse.interp.common import provenance
        fb = controller(batch=2)
        net = torch.nn.Linear(1, 1)
        with torch.no_grad():
            net.weight.fill_(0); net.bias.fill_(25)
        module = fb.attach(TorchModule({"in": [0]}, {"out": [1]}, net, "drive_mv", name="net", kind="mechanism"))
        fb.add_hook(force, name="force")
        fb.step()
        saved = fb.state_dict()
        prov = provenance(fb.c, fb=fb)
        self.assertEqual(prov["model"]["hooks"][0]["name"], "force")
        self.assertEqual(prov["model"]["modules"][0]["checkpoint_hash"], module.describe()["checkpoint_hash"])
        fb.step(30)
        expected = fb.brain.rate.clone()
        with torch.no_grad():
            net.bias.zero_()
        fb.load_state_dict(saved)
        fb.step(30)
        torch.testing.assert_close(fb.brain.rate, expected, rtol=0, atol=0)
        fb.reset([0])
        self.assertEqual(float(fb.brain.rate[0].sum()), 0)
        self.assertGreater(float(fb.brain.rate[1].sum()), 0)

    def test_readout(self):
        fb = controller(batch=2)
        m = fb.attach(ReadoutModule({"cells": [0, 1]}, lambda dt, i: {"health": i["cells"] + 1},
                                   channels=[NTChannel("health", "score", 0, 10)]))
        self.assertIsNone(m.readout())
        fb.step()
        snap = m.readout(batch_index=1)
        self.assertEqual(snap.levels.shape, (2, 1))
        np.testing.assert_array_equal(snap.levels, 1)
        self.assertIsNone(fb.neurotransmitters())

    def test_recorder_decompose_and_trace_input_classes(self):
        from flyverse.interp.common import Recorder
        from flyverse.interp.decompose import decompose
        from flyverse.interp.trace import trace, ArmAccumulator
        fb = FlyBrain(graph().subset([3, 4, 5]), device="cpu", lif_params=LIFParams(receptor_model=None))
        fb.attach(FunctionModule({}, {"out": [1]}, lambda dt, i: {"out": torch.tensor([[7.]])},
                                 name="extra", channel_out="drive_mv"))
        rec = Recorder(fb.c, np.arange(fb.c.n), quantities=("rate_hz", "drive_mv"))
        acc = ArmAccumulator(fb)
        for k in range(5):
            fb.step(); rec.capture(fb); acc.add(fb, k, 0)
        recorded = rec.finish()
        self.assertIn("module:extra:drive_mv", recorded.quantities)
        result = decompose(fb.c, "DNa02", recording=recorded, fb=fb, params=fb.brain.p)
        self.assertIn("module:extra", result.table("per_type").pre_group.tolist())
        self.assertEqual(result.table("module_inputs").input_class.iloc[0], "module:extra")
        traced = trace(fb.c, "DNp01", stimulus=[recorded], control=[recorded], fb=fb, decompose_at=None, min_cells=1)
        self.assertEqual(traced.table("module_inputs").input_class.iloc[0], "module:extra")
        accumulated, _ = acc.finish()
        self.assertIn("module:extra:drive_mv", accumulated.quantities)

    def test_motor_decoder_and_environment_factory(self):
        from unittest.mock import patch
        from flyverse.env import EnvParams, FlyRoomEnv
        from flyverse.modules import MotorDecoder
        fb = controller(batch=2)
        net = torch.nn.Linear(4, 2)
        with torch.no_grad():
            net.weight.zero_(); net.bias[:] = torch.tensor([.5, -.25])
        decoder = MotorDecoder(net)
        params = EnvParams(modules_attached=lambda env: [decoder], motor_decoder=decoder.name)
        with patch("flyverse.env.FlyBrain", return_value=fb):
            env = FlyRoomEnv(batch=2, params=params, device="cpu")
        env.reset()
        before = env.heading.copy()
        obs, reward, done, info = env.step()
        np.testing.assert_allclose(env.heading - before, -.25 * params.max_yaw * .01)
        self.assertEqual(obs.shape, (2, 2))
        self.assertEqual(env.provenance()["model"]["modules"][0]["kind"], "decoder")
        env.p.action_adapter = lambda e, a: torch.zeros((2, 2)) + a
        before = env.heading.copy()
        env.step(.1)
        np.testing.assert_allclose(env.heading - before, .1 * params.max_yaw * .01, rtol=1e-6)
        from flyverse.programs import Composite
        cmd = Composite([decoder]).apply(fb.motor().row(0), {"speed": 0., "yaw": 0., "proboscis": .7}, None, None, .01)
        self.assertAlmostEqual(cmd["speed"], .5 * decoder.max_speed)
        self.assertAlmostEqual(cmd["yaw"], -.25 * decoder.max_yaw)
        self.assertEqual(cmd["proboscis"], .7)

    def test_hook_drive_survives_optic_update_and_detach(self):
        from test_surrogate import optic_graph
        fb = FlyBrain(optic_graph(), device="cpu", lif_params=LIFParams(receptor_model=None))
        fb.vision(torch.ones(1, 4))
        fb.add_hook(lambda f, t: f.brain.set_drive([3], -8.), name="inhibit")
        fb.step()
        self.assertAlmostEqual(float(fb.brain.drive[0, 3]), -8.)
        fb.remove_hook("inhibit")
        fb.step()
        self.assertAlmostEqual(float(fb.brain.drive[0, 3]), 0.)

    def test_late_readout_does_not_hold_old_senses(self):
        from test_control import graph as smell_graph
        p = LIFParams(receptor_model=None)
        a, b = [FlyBrain(smell_graph(), device="cpu", lif_params=p, seed=9) for _ in range(2)]
        for f in (a, b):
            f.smell({"DM1": 1.}, {"DM1": 1.}); f.step()
        a.attach(FunctionModule({"observe": [0]}, {}, lambda dt, i: {}, name="observe", kind="analysis"))
        for f in (a, b):
            f.smell({"DM1": 0.}, {"DM1": 0.}); f.step()
        torch.testing.assert_close(a.brain.poisson_p, b.brain.poisson_p, rtol=0, atol=0)
        torch.testing.assert_close(a.brain.rate, b.brain.rate, rtol=0, atol=0)
        from test_surrogate import optic_graph
        a, b = [FlyBrain(optic_graph(), device="cpu", lif_params=p) for _ in range(2)]
        for f in (a, b):
            f.vision(torch.ones(1, 4)); f.step()
            f.vision(torch.full((1, 4), .1)); f.step()
        a.attach(FunctionModule({"observe": [3]}, {}, lambda dt, i: {}, name="observe", kind="analysis"))
        for f in (a, b):
            f.step()
        torch.testing.assert_close(a.brain.drive, b.brain.drive, rtol=0, atol=0)
        a.detach("observe")
        torch.testing.assert_close(a.brain.drive, b.brain.drive, rtol=0, atol=0)

    def test_module_poisson_is_not_added_to_current_units(self):
        from flyverse.interp.common import Recorder
        from flyverse.interp.decompose import decompose
        fb = FlyBrain(graph().subset([3, 4, 5]), device="cpu", lif_params=LIFParams(receptor_model=None))
        fb.attach(FunctionModule({}, {"out": [1]}, lambda dt, i: {"out": torch.tensor([[123.]])}, name="forced"))
        rec = Recorder(fb.c, np.arange(fb.c.n), quantities=("rate_hz", "drive_mv"))
        for _ in range(3):
            fb.step(); rec.capture(fb)
        result = decompose(fb.c, "DNa02", recording=rec.finish(), params=fb.brain.p)
        rates = result.table("module_poisson")
        self.assertEqual(rates.unit.iloc[0], "Hz")
        self.assertIn("module:forced", rates.pre_group.tolist())
        self.assertNotIn("module:forced", result.table("per_type").pre_group.tolist())
        self.assertEqual(result.provenance["model"]["modules"][0]["name"], "forced")

    def test_unannotated_readout_checkpoint(self):
        fb = controller()
        module = fb.attach(ReadoutModule({"cells": [0]}, lambda dt, i: {"score": i["cells"] + 1}))
        fb.step(); saved = fb.state_dict()
        fb.reset(); fb.load_state_dict(saved)
        self.assertEqual(module.readout().channels[0].unit, "a.u.")

    def test_decoder_rl_example_on_small_cpu_brain(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from examples.learned_motor_decoder import make_decoder, reinforcement
        fb = controller(batch=2)
        weights = fb.brain._W_cpu.data.copy()
        with patch("flyverse.env.FlyBrain", return_value=fb):
            report = reinforcement(make_decoder(), SimpleNamespace(batch=2, epochs=1, device="cpu", seed=2))
        self.assertEqual(len(report["episode_rewards"]), 1)
        np.testing.assert_array_equal(weights, fb.brain._W_cpu.data)

    def test_post_hook_inputs_are_recorded_in_the_frame_they_advance(self):
        fb = controller()
        fb.add_hook(lambda f, t: f.brain.set_drive([0], t), name="after", when="post")
        fb.step()
        self.assertNotIn("hook:after:drive_mv", fb.module_inputs())
        fb.step()
        self.assertEqual(float(fb.module_inputs()["hook:after:drive_mv"][0, 0]), 10.)
        self.assertEqual(float(fb.brain.drive[0, 0]), 10.)


if __name__ == "__main__":
    unittest.main()
