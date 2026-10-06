"""games/common.py: the eye sampling conventions and the read-only decoder contract, on CPU with no dataset."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "games"))
import common as gc  # noqa: E402
from test_interp import graph  # noqa: E402
from flyverse.fly import FlyBrain  # noqa: E402
from flyverse.brain import LIFParams  # noqa: E402


def controller():
    c = graph().subset([4, 5, 7])
    return FlyBrain(c, device="cpu", seed=19, lif_params=LIFParams(receptor_model=None))


def fake_eyes(dirs):
    """An Eyes over a fake retina whose columns look along `dirs` (n, 3) with one ray each."""
    dirs = np.asarray(dirs, np.float32)
    retina = SimpleNamespace(ray_directions=lambda: (dirs[:, None, :], np.ones(1, np.float32)),
                             col_az_el=np.zeros((len(dirs), 2), np.float32), col_side=np.array(["L"] * len(dirs)))
    return gc.Eyes(SimpleNamespace(retina=retina, device=torch.device("cpu")))


class RadianceTests(unittest.TestCase):
    def test_rgb_to_radiance_uv_is_half_blue(self):
        rad = gc.rgb_to_radiance(np.array([[0.2, 0.4, 0.8]], np.float32))
        np.testing.assert_allclose(rad, [[0.4, 0.8, 0.4, 0.2]])
        t = gc.rgb_to_radiance(torch.tensor([[0.2, 0.4, 0.8]]))
        torch.testing.assert_close(t, torch.tensor([[0.4, 0.8, 0.4, 0.2]]))

    def test_pinhole_centre_left_and_behind(self):
        img = np.zeros((3, 3, 3), np.float32)
        img[1, 1] = (1.0, 0.0, 0.0)           # centre pixel red
        img[1, 0] = (0.0, 1.0, 0.0)           # left-middle pixel green
        eyes = fake_eyes([[1, 0, 0], [1, np.tan(np.deg2rad(40)), 0], [-1, 0, 0]])
        rad = eyes.from_pinhole(img, hfov_deg=90, outside=(9, 9, 9, 9)).numpy()
        np.testing.assert_allclose(rad[0], [0, 0, 0, 1])                  # straight ahead -> centre
        np.testing.assert_allclose(rad[1], [0, 0, 1, 0])                  # 40 deg left -> the left column
        np.testing.assert_allclose(rad[2], [9, 9, 9, 9])                  # behind the camera -> outside

    def test_world_dirs_rotate_body_frame(self):
        eyes = fake_eyes([[1, 0, 0], [0, 1, 0]])
        d = eyes.world_dirs(forward=(0, 1, 0), left=(-1, 0, 0), up=(0, 0, 1))
        np.testing.assert_allclose(d, [[0, 1, 0], [-1, 0, 0]], atol=1e-6)

    def test_equirect_longitude_convention(self):
        img = np.zeros((2, 4, 3), np.float32)
        img[:, 2] = (1, 1, 1)                                            # u = 2 is longitude 0 (+x)
        eyes = fake_eyes([[1, 0, 0], [-1, 0, 0]])
        rad = eyes.from_equirect(img).numpy()
        self.assertGreater(rad[0, 1], 0.9)
        self.assertLess(rad[1, 1], 0.1)


class DecoderTests(unittest.TestCase):
    def test_read_decoder_is_numerically_neutral_and_recorded(self):
        fb, bare = controller(), controller()
        seen = []
        dec = gc.ReadDecoder("probe", {"a": [0], "b": [1, 2]}, lambda dt, x: seen.append(dt) or float(x["b"].sum()),
                             law="sum of two rates, for the test")
        fb.attach(dec)
        for f in (fb, bare):
            f.brain.set_poisson([0], 200.0)
        for _ in range(30):
            fb.step(10.0); bare.step(10.0)
        self.assertEqual(len(seen), 30)                                  # stepped once per step() call
        self.assertIsInstance(dec.value, float)
        for name in FlyBrain.BRAIN_TENSORS:
            torch.testing.assert_close(getattr(fb.brain, name), getattr(bare.brain, name), rtol=0, atol=0)
        rec = fb.module_records()[0]
        self.assertEqual((rec["kind"], rec["law"], rec["writes"]), ("decoder", "sum of two rates, for the test", {}))

    def test_declare_rejects_unknown_kind_and_logs(self):
        log = gc.RunLog("test")
        gc.declare(log, "autopilot", "game", "forward at 1 m/s", speed=1.0)
        self.assertEqual(log.meta["declared"][0]["parameters"], {"speed": 1.0})
        with self.assertRaises(ValueError):
            gc.declare(log, "x", "mechanism", "no")


if __name__ == "__main__":
    unittest.main()
