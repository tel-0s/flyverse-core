import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_interp import graph
from flyverse import connectome, regions
from flyverse.brain import Brain, LIFParams
from flyverse.fly import FlyBrain
from flyverse.modules import SNNModule
from flyverse.interp.common import connectome_fingerprint


def nodes():
    return pd.DataFrame({"bodyId": np.array([-1, -2], np.int64), "type": ["relay", "inhibitor"],
                         "superclass": ["auxiliary", "auxiliary"], "side": ["L", "R"],
                         "nt": ["acetylcholine", "gaba"]})


def edges():
    return pd.DataFrame({"body_pre": [1005, -1, -2], "body_post": [-1, -2, 1006],
                         "weight": [20, 30, 40], "sign": [1, 1, -1]})


class ExtendTests(unittest.TestCase):
    def test_extend_spike_prune_and_cache(self):
        c = graph().subset([4, 5, 6, 7])
        original = c.W.copy()
        with tempfile.TemporaryDirectory() as td:
            c2 = c.extend(nodes(), edges(), cache_dir=Path(td) / "extended")
            block = c2.W[:c.n, :c.n].tocsr()
            for part in ("data", "indices", "indptr"):
                self.assertEqual(getattr(block, part).tobytes(), getattr(original, part).tobytes())
            self.assertEqual(c2.W[1, -1], -40)
            self.assertEqual(c2.neurons.bodyId.tolist(), c.neurons.bodyId.tolist() + [-1, -2])
            np.testing.assert_array_equal(regions.select(c2, "auxiliary"), [4, 5])
            b = Brain(c2, LIFParams(receptor_model=None), device="cpu")
            b.set_poisson([4, 5], 1000)
            b.step(20)
            self.assertTrue(bool((b.rate[:, 4:] > 0).all()))
            fp = connectome_fingerprint(c2)
            self.assertNotEqual(fp["md5"], connectome_fingerprint(c)["md5"])
            self.assertEqual((fp["extension"]["n_nodes"], fp["extension"]["n_edges"]), (2, 3))
            self.assertEqual(fp["extension"]["base"]["md5"], connectome_fingerprint(c)["md5"])
            loaded = connectome.load(c2.cache_dir, verbose=False)
            self.assertEqual(connectome_fingerprint(loaded)["extension"], fp["extension"])
            for extended in (c2, loaded):
                restored = extended.prune(extended.select(dataset="synthetic"))
                self.assertEqual(restored.W.data.tobytes(), c.W.data.tobytes())
                self.assertEqual(connectome_fingerprint(restored)["md5"], connectome_fingerprint(c)["md5"])
            with self.assertRaisesRegex(ValueError, "scratch"):
                connectome.save(c2)
            # a plain save must not silently strip the extension cache it lands on
            with self.assertRaisesRegex(ValueError, "clear_extension"):
                connectome.save(c, c2.cache_dir)
            self.assertTrue((Path(c2.cache_dir) / "extension.json").exists())
            connectome.save(c, c2.cache_dir, clear_extension=True)
            self.assertFalse((Path(c2.cache_dir) / "extension.json").exists())

    def test_validation_and_nt_sign(self):
        c = graph().subset([4, 5])
        with tempfile.TemporaryDirectory() as td:
            c2 = c.extend(nodes(), edges().drop(columns="sign"), cache_dir=Path(td) / "valid")
            self.assertEqual(c2.W[1, -1], -40)
            bad = nodes(); bad.loc[0, "bodyId"] = 1005
            with self.assertRaisesRegex(ValueError, "negative"):
                c.extend(bad, edges(), cache_dir=Path(td) / "bad")
            bad_edges = edges(); bad_edges.loc[0, "body_post"] = 1006
            with self.assertRaisesRegex(ValueError, "immutable"):
                c.extend(nodes(), bad_edges, cache_dir=Path(td) / "bad")

    def test_secondary_snn_and_checkpoint(self):
        c = graph().subset([4, 5])
        with tempfile.TemporaryDirectory() as td:
            aux = c.extend(nodes(), edges(), cache_dir=td).subset([2, 3])
            fb = FlyBrain(c, batch=2, device="cpu", lif_params=LIFParams(receptor_model=None))
            module = fb.attach(SNNModule({"in": [0]}, {"out": [1]}, aux,
                                         {"in": [0]}, {"out": [0]}, params=LIFParams(receptor_model=None)))
            W = fb.brain._W_cpu.copy()
            fb.stimulate([0], 1000, 100)
            fb.step(30)
            self.assertGreater(float(module.brain.rate.sum()), 0)
            self.assertGreater(float(fb.brain.poisson_p[:, 1].sum()), 0)
            saved = fb.state_dict()
            fb.step(20); expected = fb.brain.rate.clone()
            fb.load_state_dict(saved); fb.step(20)
            torch.testing.assert_close(fb.brain.rate, expected, rtol=0, atol=0)
            np.testing.assert_array_equal(fb.brain._W_cpu.data, W.data)

    def test_export_endpoint_namespaces_and_sign_zero_counts(self):
        from flyverse.interp import common, export
        from unittest.mock import patch
        c = graph().subset([4, 5])
        e = edges(); e.loc[2, "sign"] = 0
        with tempfile.TemporaryDirectory() as td:
            c2 = c.extend(nodes(), e, cache_dir=td)
            with patch("flyverse.connectome.build_sign0_counts", side_effect=AssertionError("shared cache write")):
                raw = connectome.sign0_counts(c2, cache_dir=Path(td) / "absent", build=True)
            at = c2.W.tocoo()
            self.assertEqual(float(raw[(at.row == 1) & (at.col == 3)][0]), 40.)
            prov = common.provenance(c2)
            table = export._dataset_columns(pd.DataFrame({"body_pre": [1005, -2], "body_post": [-1, 1006]}), prov)
            self.assertEqual(table.dataset_pre.tolist(), [common.DATASET_NAME, "synthetic"])
            self.assertEqual(table.dataset_post.tolist(), ["synthetic", common.DATASET_NAME])

    def test_existing_cache_is_protected_and_fingerprint_checks_identity(self):
        from flyverse.interp import common
        c = graph().subset([4, 5])
        with tempfile.TemporaryDirectory() as td:
            c2 = c.extend(nodes(), edges(), cache_dir=Path(td) / "new")
            biological = Path(td) / "biological"
            connectome.save(c, biological)
            before = (biological / "W_post_pre.npz").read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                connectome.save(c2, biological)
            self.assertEqual((biological / "W_post_pre.npz").read_bytes(), before)
            expected = connectome_fingerprint(c2)["md5"]
            connectome_fingerprint(c)
            # Reproduce an ID-cache collision without depending on the allocator.
            common._FP_CACHE[id(c2.reference)] = common._FP_CACHE[id(c.reference)]
            self.assertEqual(connectome_fingerprint(c2)["md5"], expected)


if __name__ == "__main__":
    unittest.main()
