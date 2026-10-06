"""games/hairdryer.py: the pure logic (jet law, wind geometry, the declared decoder, the scripted holder, the contact
rule, the camera, the banner and HUD helpers, the wind streaks) on CPU, with no dataset and no brain."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "games"))
import common as gc
import hairdryer as hd

from flyverse import air

DEG = math.pi / 180


class JetAndWind(unittest.TestCase):
    def test_round_jet_law(self):
        self.assertAlmostEqual(hd.jet_speed(hd.JET_D_REF), hd.JET_V_REF)
        self.assertAlmostEqual(hd.jet_speed(2 * hd.JET_D_REF), hd.JET_V_REF / 2)       # 1/d
        self.assertAlmostEqual(hd.jet_speed(1e-6), hd.JET_V_MAX)                        # capped

    def test_wind_blows_from_the_nozzle_to_the_fly(self):
        self.assertAlmostEqual(hd.wind_towards_deg((0.0, 0.0), (1.0, 0.0)), 0.0)
        self.assertAlmostEqual(hd.wind_towards_deg((0.0, 1.0), (0.0, 0.0)), -90.0)

    def test_wind_from_relative(self):
        # fly heading +x; nozzle on its left (+y): the wind blows towards -y and comes from the left (+90 deg)
        towards = hd.wind_towards_deg((0.0, 0.2), (0.0, 0.0))
        self.assertAlmostEqual(hd.wind_from_relative(towards, 0.0), 90 * DEG)
        towards = hd.wind_towards_deg((0.2, 0.0), (0.0, 0.0))                          # from dead ahead
        self.assertAlmostEqual(hd.wind_from_relative(towards, 0.0), 0.0, places=9)

    def test_left_wind_deflects_the_left_antenna_back(self):
        # the sign convention the decoder relies on: flyverse.air, wind from the fly's left
        a = air.Air([], air.WindParams(speed=0.5, direction_deg=hd.wind_towards_deg((0.0, 0.2), (0.0, 0.0)),
                                       meander_deg=0.0))
        dL, dR = a.deflections(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        self.assertGreater(float(dL[0]), 0.0)
        self.assertLess(float(dR[0]), 0.0)


class Decoder(unittest.TestCase):
    def rates(self, p18L=10.0, p18R=10.0, p33L=5.0, p33R=5.0):
        return {"DNp18_L": p18L, "DNp18_R": p18R, "DNp33_L": p33L, "DNp33_R": p33R}

    def test_index_signs(self):
        self.assertGreater(hd.WindTurnDecoder.index(60, 10, 0, 30), 0)     # wind from the left: DNp18 L, DNp33 R
        self.assertLess(hd.WindTurnDecoder.index(10, 60, 30, 0), 0)        # wind from the right
        self.assertEqual(hd.WindTurnDecoder.index(20, 20, 7, 7), 0)

    def test_turns_into_the_wind_and_smooths(self):
        dec = hd.WindTurnDecoder(gain_deg_s_per_hz=3.0, tau_s=0.15, yaw_max_deg_s=200.0)
        y1 = dec.update(0.01, self.rates(p18L=60, p18R=10, p33L=0, p33R=30))    # u = 40 Hz
        self.assertGreater(y1, 0.0)                                            # a left turn, towards the wind
        for _ in range(300):
            y = dec.update(0.01, self.rates(p18L=60, p18R=10, p33L=0, p33R=30))
        self.assertAlmostEqual(dec.u_s, 40.0, places=3)
        self.assertAlmostEqual(y, 120 * DEG, places=4)                          # 3 deg/s per Hz x 40 Hz
        self.assertLess(y1, y)                                                  # the EMA lags

    def test_default_cap(self):
        dec = hd.WindTurnDecoder()                         # 60 deg/s: faster self-turns near the apple fire the GF
        for _ in range(300):
            y = dec.update(0.01, self.rates(p18L=80, p18R=0, p33L=0, p33R=60))
        self.assertAlmostEqual(y, 60 * DEG)

    def test_clipped(self):
        dec = hd.WindTurnDecoder(gain_deg_s_per_hz=10.0, tau_s=0.01, yaw_max_deg_s=200)
        for _ in range(50):
            y = dec.update(0.01, self.rates(p18L=200, p18R=0, p33L=0, p33R=200))
        self.assertAlmostEqual(y, 200 * DEG)

    def test_as_read_decoder(self):
        dec = hd.WindTurnDecoder()
        rd = gc.ReadDecoder("wind_turn", hd.WindTurnDecoder.READS, dec, law=dec.law, parameters=dec.parameters())
        self.assertEqual(rd.kind, "decoder")
        self.assertEqual(rd.writes, {})
        x = {"DNp18_L": torch.tensor([[50.0]]), "DNp18_R": torch.tensor([[10.0]]),
             "DNp33_L": torch.tensor([[0.0]]), "DNp33_R": torch.tensor([[20.0]])}
        self.assertEqual(rd.step(10.0, x), {})
        self.assertGreater(rd.value, 0.0)
        self.assertEqual(dec.rates["DNp18_L"], 50.0)
        d = rd.describe()
        self.assertEqual(d["kind"], "decoder")
        self.assertIn("DNp18", d["law"])
        self.assertEqual(d["callable"], "WindTurnDecoder")


class Holder(unittest.TestCase):
    def test_nozzle_beyond_the_apple(self):
        h = hd.FanHolder(0.3, 0.0, 0.0, lead=0.10, min_range=0.15)
        tb, tr, err, clamped = h.target((0.0, 0.0), 0.0, (0.2, 0.0))        # apple dead ahead, 20 cm
        self.assertAlmostEqual(tb, 0.0)
        self.assertAlmostEqual(tr, 0.30)
        self.assertAlmostEqual(err, 0.0)
        self.assertFalse(clamped)

    def test_rear_dead_zone_is_avoided(self):
        h = hd.FanHolder(0.3, 0.0, 0.0, max_rel_deg=100.0, min_range=0.15)
        tb, tr, _, clamped = h.target((0.0, 0.0), 0.0, (-0.1, 0.05))        # apple behind-left
        self.assertTrue(clamped)
        self.assertAlmostEqual(tb, 100 * DEG)
        self.assertAlmostEqual(tr, 0.15)
        tb, _, _, _ = h.target((0.0, 0.0), 0.0, (-0.1, -0.001))               # nearly astern: keep the side in use
        self.assertAlmostEqual(tb, 100 * DEG)

    def test_integral_cancels_a_steady_heading_error(self):
        h = hd.FanHolder(0.3, 0.0, 0.0, k_i=1.0)
        for _ in range(100):                                                   # fly heads 15 deg left of the apple
            h.update(0.01, (0.0, 0.0), 15 * DEG, (0.3, 0.0))
        self.assertLess(h.offset, 0.0)                                         # so the nozzle moves to the right

    def test_hand_speed_is_limited(self):
        h = hd.FanHolder(0.2, 0.0, 0.0, max_speed=0.2)
        before = h.nozzle((0.0, 0.0))
        after = h.update(0.01, (0.0, 0.0), 0.0, (0.0, 0.5))                   # target 90 deg away
        self.assertLessEqual(np.hypot(*(after - before)), 0.2 * 0.01 * 1.01 + 0.10 * 0.01)

    def test_settles_near_the_apple(self):
        h = hd.FanHolder(0.2, 0.3, 0.0, settle=0.075)
        b0, r0 = h.bearing, h.range
        h.update(0.01, (0.0, 0.0), 1.0, (0.05, 0.0))
        self.assertTrue(h.settled)
        self.assertEqual((h.bearing, h.range), (b0, r0))


class GeometryAndCamera(unittest.TestCase):
    def test_contact_rule(self):
        apple, r = np.array([0.0, 0.0, 0.79]), 0.04
        edge = math.sqrt((r + hd.CONTACT_REACH) ** 2 - 0.04 ** 2)
        self.assertTrue(hd.in_contact((edge - 1e-4, 0.0, 0.75), apple, r, False))
        self.assertFalse(hd.in_contact((edge + 1e-4, 0.0, 0.75), apple, r, False))
        self.assertFalse(hd.in_contact((0.0, 0.0, 0.75), apple, r, True))           # airborne

    def test_hairdryer_parts(self):
        parts = hd.hairdryer_parts((0.0, 0.0, 0.75 + hd.NOZZLE_HEIGHT), (-1.0, 0.0, -0.3))
        self.assertEqual(len(parts), len(hd.hairdryer_parts((1, 2, 3), (0, 1, 0))))
        self.assertEqual(parts[0][2], "black")                                       # the nozzle comes first
        for c, rad, _ in parts:
            self.assertGreater(c[2] - rad, 0.75)                                    # nothing sinks into the table

    def test_camera_is_not_mirrored_and_round_trips(self):
        cam = hd.Cam((0.0, -1.0, 1.0), (0.0, 0.0, 0.0), 800, 600, 40.0)
        right = cam.project((0.2, 0.0, 0.0))                  # camera looks +y, so +x is on its right
        self.assertGreater(right[0], 400)
        p = cam.unproject_to_z(*cam.project((0.1, 0.05, 0.0)), 0.0)
        np.testing.assert_allclose(p, (0.1, 0.05, 0.0), atol=1e-9)
        np.testing.assert_allclose(cam.project_many([(0.1, 0.05, 0.0)])[0], cam.project((0.1, 0.05, 0.0)))
        self.assertGreater(cam.depth([(0.0, 0.0, 0.0)])[0], 0)
        self.assertLess(cam.depth([(0.0, -2.0, 2.0)])[0], 0)                        # behind the camera


class Display(unittest.TestCase):
    def test_turn_towards_wind(self):
        self.assertTrue(hd.turn_towards_wind(50.0, 60 * DEG))                         # left turn, wind from the left
        self.assertTrue(hd.turn_towards_wind(-50.0, -60 * DEG))
        self.assertFalse(hd.turn_towards_wind(50.0, -15 * DEG))                       # the frontal null's case
        self.assertIsNone(hd.turn_towards_wind(50.0, 1 * DEG))                        # dead ahead
        self.assertIsNone(hd.turn_towards_wind(50.0, 179.5 * DEG))                    # dead astern
        self.assertIs(type(hd.turn_towards_wind(50.0, np.float64(0.5))), bool)        # JSON-safe with numpy input

    def test_turn_banner_names_the_decoder_and_never_overclaims(self):
        head, sub, towards = hd.turn_banner(31.0, 48.2, -14.5 * DEG)                  # dev.json t = 8.92 s
        self.assertEqual(head, "DECODER: TURN LEFT")
        self.assertFalse(towards)
        self.assertNotIn("towards the wind", sub)
        self.assertIn("deg right", sub)                                              # says where the wind is
        head, sub, towards = hd.turn_banner(50.0, 52.0, 100 * DEG)
        self.assertTrue(towards)
        self.assertIn("towards the wind", sub)
        self.assertIn("+50 Hz", sub)

    def test_banner_queue_holds_each_banner(self):
        q = hd.BannerQueue(min_s=1.2)
        q.show(1.50, {"text": "ON", "kind": "fan"})
        q.show(1.62, {"text": "TURN", "kind": "turn"})                              # too soon: waits
        self.assertEqual(q.update(1.62)["text"], "ON")
        self.assertEqual(q.update(2.69)["text"], "ON")
        cur = q.update(2.70)
        self.assertEqual(cur["text"], "TURN")
        self.assertAlmostEqual(cur["t0"], 2.70)
        self.assertAlmostEqual(cur["t_event"], 1.62)                                # the event keeps its time
        q.show(3.0, {"text": "JUMP", "kind": "jump"})                              # urgent: at once
        self.assertEqual(q.update(3.0)["text"], "JUMP")

    def test_banner_queue_keeps_contact_and_feeding_drops_stale_turns(self):
        q = hd.BannerQueue(min_s=1.2)                                              # the game calls update() per tick
        q.show(20.5, {"text": "TURN", "kind": "turn"})
        q.show(21.03, {"text": "CONTACT", "kind": "contact"})
        q.show(21.2, {"text": "TURN2", "kind": "turn"})                            # dropped: others are waiting
        self.assertEqual([b["text"] for b in q.pending], ["CONTACT"])
        self.assertEqual(q.update(21.71)["text"], "CONTACT")
        q.show(22.02, {"text": "FEEDING", "kind": "feeding"})
        self.assertEqual(q.update(22.5)["text"], "CONTACT")
        self.assertEqual(q.update(22.92)["text"], "FEEDING")
        q2 = hd.BannerQueue(min_s=1.2)                                             # never out of order, even
        q2.show(0.0, {"text": "A", "kind": "fan"})                                 # without update() in between
        q2.show(0.5, {"text": "B", "kind": "contact"})
        q2.show(2.0, {"text": "C", "kind": "feeding"})
        self.assertEqual((q2.current["text"], [b["text"] for b in q2.pending]), ("B", ["C"]))

    def test_trace_axis_is_fixed(self):
        x = hd.trace_x(5, 801, 100, 801)                                           # 1 px per sample
        np.testing.assert_allclose(x, [896, 897, 898, 899, 900])                   # newest at the right edge
        self.assertAlmostEqual(hd.trace_x(801, 801, 100, 801)[0], 100)             # a full history spans the rect

    def test_streaks_flow_round_the_apple(self):
        c, r = np.array([0.0, 0.0, 0.79]), 0.04
        p = np.array([[-0.02, 0.035, 0.79, 0.5, 0.0, 0.0, 0.1],                    # inside, a glancing hit
                      [-0.035, 0.0, 0.79, 0.5, 0.0, 0.0, 0.1],                     # head-on
                      [-0.10, 0.0, 0.79, 0.5, 0.0, 0.0, 0.1]])                     # outside: untouched
        before = p.copy()
        hd.deflect_round_sphere(p, c, r)
        d0 = np.linalg.norm(p[0, :3] - c)
        self.assertGreaterEqual(d0, r)
        n = (p[0, :3] - c) / d0
        self.assertGreaterEqual(float(p[0, 3:6] @ n), -1e-12)                      # no longer moving inwards
        self.assertGreater(p[0, 3], 0.0)                                            # still flowing downstream
        self.assertTrue(0.35 * 0.5 < np.linalg.norm(p[0, 3:6]) < 0.5)               # tangential part only
        self.assertLess(p[0, 6], 1.0)                                               # glancing: flows on
        self.assertGreater(p[1, 6], 1.0)                                            # head-on: retired (no burst)
        np.testing.assert_array_equal(p[2], before[2])
        q = before.copy()
        hd.deflect_round_sphere(q, c, r, age_after_hit=0.62)                        # the game's setting: fade out
        self.assertAlmostEqual(q[0, 6], 0.62)
        np.testing.assert_array_equal(q[2], before[2])

    def test_interactive_power_keeps_the_cap(self):
        class G:
            power = 2.4
        self.assertAlmostEqual(hd.HairdryerGame._jet(G(), 0.05), hd.JET_V_MAX)
        G.power = 1.0
        self.assertAlmostEqual(hd.HairdryerGame._jet(G(), 0.4), hd.jet_speed(0.4))  # record mode: the law as is


if __name__ == "__main__":
    unittest.main()
