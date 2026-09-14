"""The opt-in stance / swing leg cycle (flyverse.body.LegCycle) and the side-split haltere readout
(flyverse.motor.read_haltere_sides); docs/audits/body_sided_state.md.

CPU only. (a) The shipped path is bit-identical with the cycle attached and the sense off: the walk never reads the leg
state. (b) The timing laws reproduce the cited curves: stance duration ~ v^-1.025 (DeAngelis et al. 2019 Fig 1E), swing
constant, step frequency ~16 Hz at Mendes et al. 2013's ~28 mm/s plateau, stance fraction falling with speed and floored
at the tripod's 1/2, standing below 0.5 mm/s. (c) The two tripods are 180 deg apart at every frame. (d) Under an imposed
yaw the outer legs' amplitude exceeds the inner legs' by 2 * yaw * half_width * tau_stance / step_ref, and the sense's
left and right cells of one segment then differ while its unsided cells read the mean; yaw 0 gives equal sides, the
opposite yaw flips them. (e) The batched and scalar cycles agree; the leg state is checkpointed with the FlyState.
(f) The side-split haltere readout equals the per-side means of the same rate tensor MotorRates.haltere averages.
"""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

from flyverse import body
from flyverse.batch_sim import BatchSim
from flyverse.motor import haltere_side_groups, read_haltere_sides, read_motor
from flyverse.senses import Proprioception
from flyverse.fly import FlyBrain
from tests.test_proprioception import graph, snapshot


LEGS = body.LegCycle.LEGS


def legs_state(phase, stance, amp, beta):
    return dict(phase=np.asarray(phase, float)[None], stance=np.asarray(stance, bool)[None], amp=np.asarray(amp, float)[None], beta=np.array([beta]))


class TimingLawTests(unittest.TestCase):
    def test_stance_duration_power_law_and_constant_swing(self):
        cyc = body.LegCycle()
        v = np.array([0.005, 0.010, 0.020, 0.028, 0.030])
        tau, f, beta = cyc.timing(v)
        np.testing.assert_allclose(tau, 0.9328 * (v / 0.001) ** -1.025, rtol=1e-12)          # DeAngelis 2019: 932.8 ms * v^-1.025
        np.testing.assert_allclose(1.0 / f - tau, cyc.swing_s)                                 # swing constant
        self.assertTrue(np.all(np.diff(f) > 0)); self.assertTrue(np.all(np.diff(beta) <= 0))
        self.assertAlmostEqual(float(f[3]), 16.0, delta=1.0)                                   # Mendes 2013: ~16 Hz at >= 30 mm/s (28 mm/s representative)
        t5 = 0.9328 * 5 ** -1.025                                                              # 5 mm/s: 179 ms stance, 30 ms swing
        self.assertAlmostEqual(float(beta[0]), t5 / (t5 + 0.030), places=9); self.assertAlmostEqual(t5, 0.179, delta=0.001)
        self.assertGreaterEqual(float(beta.min()), cyc.stance_min)                             # the tripod never leaves 3 legs down

    def test_standing_below_the_walking_threshold(self):
        cyc = body.LegCycle()
        tau, f, beta = cyc.timing([0.0, 0.0004, 0.0005])
        self.assertEqual(f[:2].tolist(), [0.0, 0.0]); self.assertEqual(beta[:2].tolist(), [1.0, 1.0]); self.assertTrue(np.isinf(tau[:2]).all())
        self.assertGreater(f[2], 0.0)
        k = cyc.advance(cyc.initial_phase(1), [0.0], [0.0], [False], 0.01)
        self.assertTrue(k["stance"].all()); np.testing.assert_allclose(k["amp"], 0.0); self.assertEqual(k["n_stance"].tolist(), [6])
        np.testing.assert_allclose([k["load_L"][0], k["load_R"][0]], [0.5, 0.5])
        np.testing.assert_allclose(k["phase"], cyc.initial_phase(1))                           # no advance while standing

    def test_airborne_suspends_the_cycle(self):
        cyc = body.LegCycle()
        k = cyc.advance(cyc.initial_phase(1), [0.02], [0.0], [True], 0.01)
        np.testing.assert_allclose(k["phase"], cyc.initial_phase(1))
        self.assertFalse(k["stance"].any()); np.testing.assert_allclose(k["amp"], 0.0); self.assertEqual(float(k["load_L"][0] + k["load_R"][0]), 0.0)


class TripodTests(unittest.TestCase):
    def test_tripods_stay_half_a_cycle_apart_and_the_load_alternates(self):
        cyc = body.LegCycle()
        phase = cyc.initial_phase(3)
        speeds = np.array([0.008, 0.015, 0.028])
        loads = []
        for _ in range(400):
            k = cyc.advance(phase, speeds, [0.0, 0.0, 0.0], [False] * 3, 0.01)
            phase = k["phase"]
            d = (phase[:, [1, 2, 5]] - phase[:, [0, 3, 4]]) % 1.0                            # R1 L2 R3 minus L1 R2 L3
            np.testing.assert_allclose(d, 0.5, atol=1e-9)
            np.testing.assert_allclose(phase[:, 0], phase[:, 3]); np.testing.assert_allclose(phase[:, 3], phase[:, 4])
            self.assertTrue(np.all(k["n_stance"] >= 3))
            loads.append(np.stack([k["load_L"], k["load_R"]], 1))
        loads = np.array(loads)                                                                # (T, 3, 2)
        self.assertAlmostEqual(float(loads[:, 2].mean()), 0.5, delta=0.02)                     # symmetric on average ...
        self.assertGreater(float(np.abs(loads[:, 2, 0] - loads[:, 2, 1]).max()), 0.3)          # ... but sided within the tripod
        self.assertTrue(np.all(np.abs(loads.sum(2) - 1.0) < 1e-9))


class TurnAsymmetryTests(unittest.TestCase):
    def test_outer_legs_take_longer_steps(self):
        cyc = body.LegCycle()
        for yaw, sign in ((2.0, +1), (-2.0, -1), (0.0, 0)):
            k = cyc.advance(cyc.initial_phase(1), [0.01], [yaw], [False], 0.01)
            side = np.asarray(cyc.SIDE, float)
            left, right = k["amp"][0][side > 0], k["amp"][0][side < 0]
            tau = cyc.timing([0.01])[0][0]
            expect = 2 * abs(yaw) * cyc.half_width_m * tau / cyc.step_ref_m
            if sign > 0:                                                                       # left turn: right legs outer
                self.assertTrue(np.all(right > left)); np.testing.assert_allclose(right - left, expect, rtol=1e-9)
            elif sign < 0:
                self.assertTrue(np.all(left > right)); np.testing.assert_allclose(left - right, expect, rtol=1e-9)
            else:
                np.testing.assert_allclose(left, right)
            np.testing.assert_allclose(0.5 * (left + right), 0.01 * tau / cyc.step_ref_m, rtol=1e-9)

    def test_sided_channels_differ_between_sides_under_an_imposed_yaw(self):
        c = graph()
        s = Proprioception(c, "all+leg_cycle+haltere_sided")
        cyc = body.LegCycle()
        cells = {ch: hz for ch, _, hz in s.rates(0., 0., 0., False, 0., 1, legs=dict(cyc.advance(cyc.initial_phase(1), [0.01], [2.0], [False], 0.01)), haltere_L=10., haltere_R=10.)}
        # chordotonal cells 9 (L2), 10 (R2) share a leg phase (R1 L2 R3 are one tripod; R2 the other) -- compare L1-side vs R1-side
        # through the hair plates 12 (L2) / 13 (R2): same segment, opposite sides, opposite tripods, so compare the amplitude
        # itself through cells at the same phase: campaniform 14 (L1) and 15 (R1) carry no amplitude; use a synthetic state instead
        k = cyc.advance(cyc.initial_phase(1), [0.01], [2.0], [False], 0.01)
        same = legs_state(np.full(6, 0.9), np.zeros(6, bool), k["amp"][0], 0.6)                # every leg in swing at the same phase
        r = {ch: hz[0] for ch, _, hz in s.rates(0., 0., 0., False, 0., 1, legs=same, haltere_L=10., haltere_R=10.)}
        L2, R2, both3, L1 = r["chordotonal"]
        self.assertLess(L2, R2)                                                                # left turn: the right (outer) leg's burst is larger
        self.assertAlmostEqual(both3, 10 + 140 * min(0.5 * (k["amp"][0][4] + k["amp"][0][5]), 1.0))   # the unsided cell reads the mean of L3 / R3
        hL, hR = r["hair_plate"]
        self.assertLess(hL, hR)
        opp = dict(same, amp=cyc.advance(cyc.initial_phase(1), [0.01], [-2.0], [False], 0.01)["amp"])
        r2 = {ch: hz[0] for ch, _, hz in s.rates(0., 0., 0., False, 0., 1, legs=opp, haltere_L=10., haltere_R=10.)}
        self.assertGreater(r2["chordotonal"][0], r2["chordotonal"][1]); self.assertAlmostEqual(r2["chordotonal"][0], R2); self.assertAlmostEqual(r2["chordotonal"][1], L2)
        straight = dict(same, amp=cyc.advance(cyc.initial_phase(1), [0.01], [0.0], [False], 0.01)["amp"])
        r0 = {ch: hz[0] for ch, _, hz in s.rates(0., 0., 0., False, 0., 1, legs=straight, haltere_L=10., haltere_R=10.)}
        self.assertAlmostEqual(r0["chordotonal"][0], r0["chordotonal"][1]); self.assertAlmostEqual(r0["hair_plate"][0], r0["hair_plate"][1])
        del cells


class BatchAndScalarTests(unittest.TestCase):
    def test_shipped_path_is_bit_identical_with_the_cycle_attached_and_the_sense_off(self):
        a = BatchSim(2, c=graph(), device="cpu", seed=3)
        b = BatchSim(2, c=graph(), device="cpu", seed=3); b.body.leg_cycle = body.LegCycle()
        for _ in range(30): a.step(); b.step()
        for k, v in snapshot(a.fb).items(): torch.testing.assert_close(v, snapshot(b.fb)[k], rtol=0, atol=0)
        for fa, fb_ in zip(a.flies, b.flies):
            for name in ("x", "y", "z", "speed", "yaw_rate", "_heading"):
                self.assertEqual(getattr(fa, name), getattr(fb_, name))
            np.testing.assert_array_equal(fa.fwd, fb_.fwd)
        np.testing.assert_array_equal([f.leg_phase for f in a.flies], body.LegCycle().initial_phase(2))   # a: static defaults
        self.assertTrue(np.any(np.asarray([f.leg_phase for f in b.flies]) != body.LegCycle().initial_phase(2)))   # b: the cycle ran
        np.testing.assert_array_equal([f.stance_frac for f in a.flies], 1.0)

    def test_batched_cycle_matches_the_scalar_cycle(self):
        cyc = body.LegCycle()
        fly = body.FlyState(); fly.speed, fly.yaw_rate = 0.012, 0.7
        fly.leg_phase = cyc.initial_phase(1)[0]
        phase = cyc.initial_phase(1)
        for _ in range(50):
            cyc.apply(fly, 0.01)
            k = cyc.advance(phase, [0.012], [0.7], [False], 0.01); phase = k["phase"]
        np.testing.assert_allclose(fly.leg_phase, phase[0]); np.testing.assert_array_equal(fly.leg_stance, k["stance"][0])
        np.testing.assert_allclose(fly.leg_amp, k["amp"][0]); self.assertEqual(fly.stance_frac, float(k["beta"][0]))
        self.assertEqual((fly.stance_load_L, fly.stance_load_R), (float(k["load_L"][0]), float(k["load_R"][0])))

    def test_batch_sim_feeds_per_leg_rates_and_checkpoints_the_leg_state(self):
        sim = BatchSim(2, c=graph(), device="cpu", seed=1, proprioception="all+leg_cycle+haltere_sided")
        sim.body.leg_cycle = body.LegCycle()
        self.assertEqual(sim.fb.proprioception_sense.spec, "all+leg_cycle+haltere_sided")
        for _ in range(6): sim.step()
        st = body.LegCycle.state(sim.flies)
        self.assertTrue(np.any(st["phase"] != body.LegCycle().initial_phase(2)))
        self.assertTrue(np.all(st["beta"] < 1.0))
        # the previous frame's cycle state is what the transducer injected this frame
        sense = sim.fb.proprioception_sense
        state = sim.body.proprio_state(sim.motor, haltere_sides=sense.haltere_sides(sim.brain))
        self.assertIn("legs", state); self.assertIn("haltere_L", state)
        dt = sim.brain.p.dt / 1000
        sim.step()
        expect = {ch: hz for ch, _, hz in sense.rates(**state, batch=2)}
        np.testing.assert_allclose(sim.brain.poisson_p[:, [9, 10, 11, 20]].numpy(), expect["chordotonal"] * dt, rtol=1e-5)
        np.testing.assert_allclose(sim.brain.poisson_p[:, [14, 15]].numpy(), expect["campaniform"] * dt, rtol=1e-5)
        diff = []
        for _ in range(40):                                                                    # L1 vs R1 sit in opposite tripods: at
            sim.step(); diff.append(float(np.abs(sim.brain.poisson_p[:, 14] - sim.brain.poisson_p[:, 15]).max()))   # some frame one is in swing
        self.assertGreater(max(diff), 0.0); self.assertEqual(min(diff), 0.0)                    # ... and at some frame both are down (beta > 1/2)
        # checkpoint: the leg state rides with the FlyState; a resumed sim (cycle re-attached) injects the same afferents
        chk = sim.state_dict()
        for _ in range(2): sim.step()
        other = BatchSim(2, c=graph(), device="cpu", seed=1, proprioception="all+leg_cycle+haltere_sided"); other.body.leg_cycle = body.LegCycle()
        other.load_state_dict(chk)
        for _ in range(2): other.step()
        torch.testing.assert_close(other.brain.poisson_p, sim.brain.poisson_p, rtol=0, atol=0)
        with self.assertRaises(ValueError):                                                  # the token without the body state is loud
            sense.rates(0., 0., 0., False, 0., 2)

    def test_side_split_haltere_readout(self):
        c = graph()
        g = haltere_side_groups(c)
        self.assertEqual((len(g["L"]), len(g["R"]), len(g["unsided"])), (1, 0, 0))              # one hDVM MN_L on the synthetic graph
        fb = FlyBrain(c, batch=2, device="cpu", seed=2)
        fb.stimulate([8], 300., 50.)
        for _ in range(5): fb.step(10.)
        hL, hR = read_haltere_sides(fb.brain, g)
        m = read_motor(fb.brain)
        np.testing.assert_allclose(hL, m.haltere, rtol=1e-6); np.testing.assert_allclose(hR, 0.0)
        self.assertGreater(float(np.max(hL)), 0.0)


if __name__ == "__main__":
    unittest.main()
