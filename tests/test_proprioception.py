"""The opt-in proprioception transducer (flyverse.senses.Proprioception; docs/audits/proprioception_transducer.md).

Synthetic-graph tests need no dataset: (a) the shipped path is bit-identical with the sense absent or attached-but-unfed,
(b) the channel selection and the side rule, (c) the rate laws (ceilings, airborne / ground, sides, the stop-gap Coriolis
term off by default) and the BatchSim per-frame call. The cache-backed class pins the real cell counts per channel and
side (SApp 148 ...) and repeats (a) on a Connectome.subset of the shipped cache; it is skipped without the cache.
"""
import os
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from flyverse.connectome import Connectome, CACHE_DIR
from flyverse.fly import FlyBrain
from flyverse.senses import Proprioception
from flyverse.batch_sim import BatchSim

PRO = "mechanosensory_proprioceptive"


def graph():
    """22 cells: an ORN -> PN pair (a smell input), DNa02 L / R, VNC interneurons L / R (the laterality targets), leg MNs
    L / R, a haltere MN, and the proprioceptors: chordotonal (SNpp39 x2 sided by their output, an unsided SNppxx, a
    sensory_ascending SApp23_L), hair plates, leg campaniform (SNpp53_L / _R by instance; _R wired to the LEFT target to
    check the instance wins), a wing campaniform (ADMN: excluded), haltere (SApp_L / _R by instance, SNpp34 by output),
    and a bristle (excluded)."""
    rows = [
        # bodyId, type, instance, superclass, class, subclass, somaSide, entryNerve
        (10, "ORN_DM1", "ORN_DM1", "cb_sensory", "olfactory", "", None, None),
        (11, "DM1_lPN", "DM1_lPN_L", "cb_intrinsic", "ALPN", "", "L", None),
        (12, "DNa02", "DNa02_L", "descending_neuron", "", "", "L", None),
        (13, "DNa02", "DNa02_R", "descending_neuron", "", "", "R", None),
        (14, "IN19A003", "IN19A003_L", "vnc_intrinsic", "", "", "L", None),
        (15, "IN19A003", "IN19A003_R", "vnc_intrinsic", "", "", "R", None),
        (16, "Fe reductor MN", "", "vnc_motor", "", "fl", "L", None),
        (17, "Fe reductor MN", "", "vnc_motor", "", "fl", "R", None),
        (18, "hDVM MN", "", "vnc_motor", "", "hm", "L", None),
        (19, "SNpp39", "", "vnc_sensory", PRO, "chordotonal organ", None, "MesoLN"),    # -> IN_L : side L
        (20, "SNpp39", "", "vnc_sensory", PRO, "chordotonal organ", None, "MesoLN"),    # -> IN_R : side R
        (21, "SNppxx", "", "vnc_sensory", PRO, "leg", None, "MetaLN"),                  # -> both : unsided
        (22, "SNpp45", "", "vnc_sensory", PRO, "hair plate", None, "MesoLN"),           # -> IN_L
        (23, "SNpp45", "", "vnc_sensory", PRO, "hair plate", None, "MesoLN"),           # -> IN_R
        (24, "SNpp53", "SNpp53_L", "vnc_sensory", PRO, "campaniform sensilla", None, "ProLN"),
        (25, "SNpp53", "SNpp53_R", "vnc_sensory", PRO, "campaniform sensilla", None, "ProLN"),   # wired to IN_L: instance wins
        (26, "SNpp07", "", "vnc_sensory", PRO, "campaniform sensilla", None, "ADMN"),   # wing: excluded
        (27, "SApp", "SApp_L", "sensory_ascending", PRO, "haltere", None, "DMetaN"),
        (28, "SApp", "SApp_R", "sensory_ascending", PRO, "haltere", None, "DMetaN"),
        (29, "SNpp34", "", "vnc_sensory", PRO, "haltere", None, "DMetaN"),              # -> IN_R
        (30, "SApp23", "SApp23_L", "sensory_ascending", PRO, "chordotonal organ", None, "ProLN"),
        (31, "SNta01", "", "vnc_sensory", "mechanosensory_tactile", "bristle", None, "ProLN"),  # excluded
    ]
    n = pd.DataFrame(rows, columns=["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "entryNerve"])
    N = len(n); w = np.zeros((N, N), np.float32)
    w[1, 0] = 400; w[2, 1] = 100; w[3, 1] = 100; w[6, 2] = 100; w[7, 3] = 100; w[8, 2] = 40; w[8, 3] = 40
    w[2, 4] = -60; w[3, 5] = -60                                     # the VNC inhibitors onto DNa02
    for pre, post in ((9, 4), (10, 5), (12, 4), (13, 5), (15, 4), (19, 5), (20, 4), (21, 4), (11, 4), (11, 5)):
        w[post, pre] = 50
    w[4, 14] = 50; w[5, 14] = 50; w[4, 16] = 50; w[4, 17] = 50; w[5, 18] = 50; w[4, 21] = 50
    w[6, 9] = 30; w[7, 10] = 30                                      # afferents onto the leg MNs
    return Connectome(n, sp.csr_matrix(w), pd.Series(np.arange(N), index=n.bodyId))


def snapshot(fb):
    return {k: getattr(fb.brain, k).clone() for k in ("v", "g", "rate", "poisson_p", "spike_counts")}


class SelectionTests(unittest.TestCase):
    def test_channels_sides_and_exclusions(self):
        s = Proprioception(graph(), "all")
        self.assertEqual(s.channels, Proprioception.CHANNELS); self.assertFalse(s.haltere_coriolis); self.assertEqual(s.spec, "all")
        np.testing.assert_array_equal(s.idx["chordotonal"], [9, 10, 11, 20]); np.testing.assert_array_equal(s.side["chordotonal"], [1, -1, 0, 1])
        np.testing.assert_array_equal(s.idx["hair_plate"], [12, 13]); np.testing.assert_array_equal(s.side["hair_plate"], [1, -1])
        np.testing.assert_array_equal(s.idx["campaniform"], [14, 15]); np.testing.assert_array_equal(s.side["campaniform"], [1, -1])  # instance beats the wiring
        np.testing.assert_array_equal(s.idx["haltere"], [17, 18, 19]); np.testing.assert_array_equal(s.side["haltere"], [1, -1, -1])
        c = s.counts()
        self.assertEqual(c["chordotonal"], dict(n=4, L=2, R=1, both=1, vnc_sensory=3, sensory_ascending=1, side_source={"instance": 1, "laterality": 3}))
        self.assertEqual(c["haltere"]["side_source"], {"instance": 2, "laterality": 1})
        for excluded in (16, 21):                                    # wing campaniform, bristle
            self.assertFalse(any(excluded in idx for idx in s.idx.values()))

    def test_spec_grammar(self):
        P = Proprioception
        self.assertEqual(P.parse_spec("all"), (P.CHANNELS, False))
        self.assertEqual(P.parse_spec("all+haltere_coriolis"), (P.CHANNELS, True))
        self.assertEqual(P.parse_spec("haltere, chordotonal"), (("chordotonal", "haltere"), False))
        self.assertEqual(P.parse_spec("haltere_coriolis"), (("haltere",), True))
        self.assertEqual(P.parse_spec(["hair_plate", "campaniform"]), (("hair_plate", "campaniform"), False))
        for bad in ("", "legs", "all,typo", 3):
            with self.assertRaises(ValueError): P.parse_spec(bad)
        s = Proprioception(graph(), "chordotonal")
        self.assertEqual(tuple(s.idx), ("chordotonal",)); self.assertEqual(s.spec, "chordotonal")
        self.assertEqual(Proprioception(graph(), "haltere_coriolis").spec, "haltere+haltere_coriolis")
        with self.assertRaises(ValueError): Proprioception(graph(), "all", chordotonal_tonic_hz=200, chordotonal_max_hz=150)
        with self.assertRaises(ValueError): Proprioception(graph(), "all", mn_ref_hz=0)
        with self.assertRaises(ValueError): Proprioception(graph(), "all", haltere_k=-1)


class RateLawTests(unittest.TestCase):
    def rates(self, s, **kw):
        args = dict(leg_L=0., leg_R=0., haltere=0., airborne=False, yaw_rate=0., batch=1); args.update(kw)
        return {ch: hz for ch, _, hz in s.rates(**args)}

    def test_ground_laws_sides_and_ceilings(self):
        s = Proprioception(graph(), "all", mn_ref_hz=30., chordotonal_tonic_hz=10., chordotonal_max_hz=150.,
                           hair_plate_tonic_hz=5., hair_plate_max_hz=100., campaniform_load_hz=50., haltere_k=1., haltere_max_hz=250.)
        r = self.rates(s)                                                     # standing still on the ground
        np.testing.assert_allclose(r["chordotonal"], [[10, 10, 10, 10]]); np.testing.assert_allclose(r["hair_plate"], [[5, 5]])
        np.testing.assert_allclose(r["campaniform"], [[50, 50]]); np.testing.assert_allclose(r["haltere"], [[0, 0, 0]])
        r = self.rates(s, leg_L=15., leg_R=0.)                                # left legs at half the reference rate
        np.testing.assert_allclose(r["chordotonal"], [[80, 10, 45, 80]])      # L, R, both (mean), L (SApp23_L)
        np.testing.assert_allclose(r["hair_plate"], [[52.5, 5]])
        r = self.rates(s, leg_L=1e6, leg_R=1e6, haltere=1e6)                  # every channel clamps to its ceiling
        np.testing.assert_allclose(r["chordotonal"], [[150] * 4]); np.testing.assert_allclose(r["hair_plate"], [[100, 100]])
        np.testing.assert_allclose(r["haltere"], [[250] * 3]); np.testing.assert_allclose(r["campaniform"], [[50, 50]])
        r = self.rates(s, haltere=180.)
        np.testing.assert_allclose(r["haltere"], [[180] * 3])                 # one afferent spike per haltere MN spike
        with self.assertRaises(ValueError): self.rates(s, leg_L=-1.)

    def test_airborne_logic(self):
        s = Proprioception(graph(), "all")
        r = self.rates(s, leg_L=30., leg_R=30., haltere=200., airborne=True)
        np.testing.assert_allclose(r["chordotonal"], [[10] * 4])              # tonic only
        np.testing.assert_allclose(r["hair_plate"], [[5, 5]])
        np.testing.assert_allclose(r["campaniform"], [[0, 0]])                # no load
        np.testing.assert_allclose(r["haltere"], [[200] * 3])                 # wingbeat-driven, unchanged
        r = self.rates(s, leg_L=[30., 30.], leg_R=[0., 0.], haltere=[100., 100.], airborne=[False, True], batch=2)
        np.testing.assert_allclose(r["chordotonal"], [[150, 10, 80, 150], [10, 10, 10, 10]])
        np.testing.assert_allclose(r["campaniform"], [[50, 50], [0, 0]])

    def test_coriolis_is_a_labelled_option_off_by_default(self):
        default = Proprioception(graph(), "all")
        self.assertEqual(default.params["haltere"]["coriolis_gain_per_rad_s"], 0.0)
        a = self.rates(default, haltere=100., yaw_rate=0.); b = self.rates(default, haltere=100., yaw_rate=2.)
        np.testing.assert_array_equal(a["haltere"], b["haltere"])            # the yaw scalar is not read
        arm = Proprioception(graph(), "all+haltere_coriolis", coriolis_gain_per_rad_s=1.0)
        self.assertTrue(arm.haltere_coriolis)
        np.testing.assert_allclose(self.rates(arm, haltere=100., yaw_rate=-0.5)["haltere"], [[150] * 3])
        np.testing.assert_allclose(self.rates(arm, haltere=100., yaw_rate=10.)["haltere"], [[250] * 3])   # clamped


class ShippedPathTests(unittest.TestCase):
    """The sense is inert unless attached and fed: the shipped path is bit-identical with or without it."""

    def run_frames(self, fb, frames=8):
        for k in range(frames):
            fb.smell({"DM1": 1.0 + 0.1 * k}, {"DM1": 0.5}); fb.step(10.)
        return snapshot(fb)

    def test_flybrain_bit_identity_and_available_senses(self):
        a = FlyBrain(graph(), batch=2, device="cpu", seed=4)
        b = FlyBrain(graph(), batch=2, device="cpu", seed=4)
        self.assertEqual(a.available_senses, ("smell",))
        with self.assertRaises(ValueError): a.proprioception(0, 0, 0, False)
        b.proprioception_sense = Proprioception(b.c, "all")
        self.assertEqual(b.available_senses, ("smell", "proprioception"))
        sa, sb = self.run_frames(a), self.run_frames(b)
        for k in sa: torch.testing.assert_close(sa[k], sb[k], rtol=0, atol=0)
        b.proprioception([15., 0.], [0., 30.], [100., 0.], [False, True])
        dt = b.brain.p.dt / 1000
        np.testing.assert_allclose(b.brain.poisson_p[:, [9, 10, 11, 20]].numpy(), np.array([[80, 10, 45, 80], [10, 10, 10, 10]]) * dt, rtol=1e-5)  # row 1 airborne: tonic
        np.testing.assert_allclose(b.brain.poisson_p[:, [14, 15]].numpy(), np.array([[50, 50], [0, 0]]) * dt, rtol=1e-5)
        np.testing.assert_allclose(b.brain.poisson_p[:, [17, 18, 19]].numpy(), np.array([[100] * 3, [0] * 3]) * dt, rtol=1e-5)
        torch.testing.assert_close(b.brain.poisson_p[:, :9], a.brain.poisson_p[:, :9], rtol=0, atol=0)     # nothing else touched
        self.assertTrue(b.brain._poisson_on)

    def test_batch_sim_default_is_untouched_and_opt_in_feeds_the_body_state(self):
        base = BatchSim(2, c=graph(), device="cpu", seed=1)
        same = BatchSim(2, c=graph(), device="cpu", seed=1, proprioception=None)
        self.assertIsNone(base.proprioception); self.assertNotIn("proprioception", base.fb.available_senses)
        for _ in range(5): base.step(); same.step()
        for k, v in snapshot(base.fb).items(): torch.testing.assert_close(v, snapshot(same.fb)[k], rtol=0, atol=0)
        self.assertEqual(base.brain.poisson_p[:, 9:].abs().sum().item(), 0.0)
        on = BatchSim(2, c=graph(), device="cpu", seed=1, proprioception="all")
        self.assertEqual(on.proprioception, "all"); self.assertIn("proprioception", on.fb.available_senses)
        on.step()                                                       # first frame: no motor snapshot yet -> tonic / load only
        dt = on.brain.p.dt / 1000
        np.testing.assert_allclose(on.brain.poisson_p[:, [9, 10, 11, 20]].numpy(), 10 * dt, rtol=1e-5)
        np.testing.assert_allclose(on.brain.poisson_p[:, [14, 15]].numpy(), 50 * dt, rtol=1e-5)
        on.flies[1].airborne = True
        state = on.body.proprio_state(on.motor)
        self.assertEqual(state["airborne"].tolist(), [False, True]); self.assertEqual(state["leg_L"].shape, (2,))
        on.step()                                                       # the sense reads the flag before the body step lands the fly
        np.testing.assert_allclose(on.brain.poisson_p[1, [14, 15]].numpy(), 0.0)
        np.testing.assert_allclose(on.brain.poisson_p[0, [14, 15]].numpy(), 50 * dt, rtol=1e-5)
        with self.assertRaises(ValueError): BatchSim(2, c=graph(), device="cpu", proprioception="typo")
        st = on.state_dict(); self.assertEqual(st["proprioception"], "all")
        with self.assertRaises(ValueError): base.load_state_dict(st)


@unittest.skipUnless((Path(CACHE_DIR) / "W_post_pre.npz").exists(), "compiled connectome cache not available")
class CacheCountsTests(unittest.TestCase):
    """The shipped cache's cell sets (docs/audits/proprioception_transducer.md section 3) and the bit-identity of the
    shipped path on a Connectome.subset around the afferents."""
    @classmethod
    def setUpClass(cls):
        from flyverse import connectome
        cls.c = connectome.load(verbose=False)
        cls.sense = Proprioception(cls.c, "all")

    def test_cell_counts_per_channel_and_side(self):
        counts = self.sense.counts(); n = self.c.neurons
        self.assertEqual(counts["chordotonal"], dict(n=615, L=271, R=259, both=85, vnc_sensory=593, sensory_ascending=22, side_source={"instance": 22, "laterality": 593}))
        self.assertEqual(counts["hair_plate"], dict(n=113, L=57, R=54, both=2, vnc_sensory=112, sensory_ascending=1, side_source={"instance": 1, "laterality": 112}))
        self.assertEqual(counts["campaniform"], dict(n=12, L=6, R=6, both=0, vnc_sensory=12, sensory_ascending=0, side_source={"instance": 12, "laterality": 0}))
        self.assertEqual(counts["haltere"], dict(n=201, L=99, R=95, both=7, vnc_sensory=53, sensory_ascending=148, side_source={"instance": 148, "laterality": 53}))
        types = n.type.fillna("").to_numpy()
        h = self.sense.idx["haltere"]; self.assertEqual(int((types[h] == "SApp").sum()), 148)
        ch = self.sense.idx["chordotonal"]
        for t, k in (("SNpp39", 39), ("SNpp50", 62), ("SNpp60", 41), ("SNpp52", 17), ("SNppxx", 78), ("SApp23", 22)):
            self.assertEqual(int((types[ch] == t).sum()), k, t)
        self.assertEqual(sorted(set(n.entryNerve.to_numpy()[self.sense.idx["campaniform"]])), ["MesoLN", "MetaLN", "ProLN"])
        # sides on the sensory_ascending cells come from the instance suffix and agree with the output laterality
        self.assertEqual(int((self.sense.side["haltere"][types[h] == "SApp"] != 0).sum()), 148)

    def test_bit_identity_on_a_cache_subset(self):
        c = self.c
        idx = np.unique(np.concatenate([v for v in self.sense.idx.values()] + [c.select(type=t) for t in ("AN04B003", "AN07B035", "AN07B037_a", "AN06A026", "PS196_b", "IN12B014", "IN19A003", "DNa02")]))
        sub = c.subset(idx)
        a = FlyBrain(sub, batch=1, device="cpu", seed=0); b = FlyBrain(sub, batch=1, device="cpu", seed=0)
        self.assertNotIn("proprioception", a.available_senses)
        b.proprioception_sense = Proprioception(sub, "all")
        self.assertIn("proprioception", b.available_senses)
        self.assertEqual(sum(len(v) for v in b.proprioception_sense.idx.values()), 615 + 113 + 12 + 201)
        for _ in range(5): a.step(10.); b.step(10.)
        sa, sb = snapshot(a), snapshot(b)
        for k in sa: torch.testing.assert_close(sa[k], sb[k], rtol=0, atol=0)
        self.assertEqual(float(b.brain.poisson_p.abs().sum()), 0.0)
        b.proprioception(2.5, 2.5, 0., False)
        touched = np.flatnonzero(b.brain.poisson_p[0].numpy() > 0)
        expected = np.unique(np.concatenate([b.proprioception_sense.idx[ch] for ch in ("chordotonal", "hair_plate", "campaniform")]))
        np.testing.assert_array_equal(touched, expected)               # haltere at 0 Hz (no wingbeat), nothing else


if __name__ == "__main__":
    unittest.main()
