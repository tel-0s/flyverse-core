"""Opt-in full-connectome checks: FLYVERSE_INTEGRATION=1 python -m unittest discover -s tests."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from flyverse import FlyBrain, connectome, retina, senses


@unittest.skipUnless(os.environ.get("FLYVERSE_INTEGRATION") == "1" and torch.cuda.is_available(),
                     "set FLYVERSE_INTEGRATION=1 with a local connectome cache and CUDA")
class FullConnectomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = connectome.load(verbose=False)

    def test_retina_and_sensory_laterality_survive_pruning(self):
        full = retina.build_retina(self.c)
        selected = full.pr_index[:40]
        sub = self.c.subset(selected[::-1])
        reduced = retina.build_retina(sub)
        columns = np.unique(full.pr_column[:40])
        np.testing.assert_array_equal(reduced.col_dir, full.col_dir[columns])
        smell = senses.Smell(self.c)
        small = senses.Smell(self.c.subset(smell.orn_idx))
        np.testing.assert_array_equal(smell.side, small.side)
        wind = senses.Wind(self.c)
        small = senses.Wind(self.c.subset(np.r_[wind.joC, wind.joE]))
        np.testing.assert_array_equal(wind.sideC, small.sideC)
        np.testing.assert_array_equal(wind.sideE, small.sideE)

    def test_full_hybrid_graph(self):
        a = FlyBrain(self.c, device="cuda", seed=5)
        b = FlyBrain(self.c, device="cuda", seed=5, cuda_graphs=True)
        for fb in (a, b):
            fb.smell({"DM1": 1.}, {"DM1": .2})
        for i, ms in enumerate((10., 10., 1.5, 1.5, 10.)):
            radiance = np.full((a.retina.n_columns, 4), .2 + i * .03, dtype=np.float32)
            a.vision(radiance); b.vision(radiance)
            a.step(ms); b.step(ms)
            # Sparse reduction order can perturb subthreshold values; spike histories must agree.
            for name in a.BRAIN_TENSORS:
                tol = dict(rtol=1e-5, atol=2e-4) if name in ("v", "g", "drive") else dict(rtol=0, atol=0)
                torch.testing.assert_close(getattr(a.brain, name), getattr(b.brain, name), **tol)
            torch.testing.assert_close(a.optic.delta_rate, b.optic.last["dr"], rtol=1e-5, atol=2e-5)
        state = b.state_dict()
        b.step(10)
        expected = b.state_dict()
        b.load_state_dict(state); b.step(10)
        for name in b.BRAIN_TENSORS:
            torch.testing.assert_close(getattr(b.brain, name).cpu(), expected["brain"][name], rtol=1e-5, atol=2e-4)

    def test_demo_resume_preserves_body_timers(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from room_demo import Sim
        sim = Sim(seed=7)
        for _ in range(10): sim.step()
        sim.stimulate_wing_dns(100)
        sim.loco._cast_from = sim.loco._t
        sim.loco._last_hit = sim.loco._t
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "state.pt")
            sim.save_state(path)
            for _ in range(5): sim.step()
            expected = sim.fb.state_dict()
            pose, command = vars(sim.fly).copy(), sim.cmd.copy()
            sim.load_state(path)
            saved = torch.load(path, weights_only=False)["controller"]
            for name in sim.fb.BRAIN_TENSORS:
                torch.testing.assert_close(getattr(sim.brain, name).cpu(), saved["brain"][name], rtol=0, atol=0)
            for _ in range(5): sim.step()
        for name in sim.fb.BRAIN_TENSORS:
            tol = dict(rtol=1e-5, atol=2e-4) if name in ("v", "g", "drive") else dict(rtol=0, atol=0)
            torch.testing.assert_close(getattr(sim.brain, name).cpu(), expected["brain"][name], **tol)
        self.assertEqual(vars(sim.fly), pose)
        self.assertEqual(sim.cmd, command)
