"""games/pong.py: court physics, the arena mapping and the ball decoder on synthetic tensors (no brain, no data)."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "games"))
import pong  # noqa: E402


class ArenaMappingTests(unittest.TestCase):
    def test_each_fly_sees_its_own_end_at_the_bottom(self):
        self.assertEqual(pong.view_of("A", 0.0, 0.0), (0.0, -pong.COURT_LEN / 2))
        self.assertEqual(pong.view_of("B", pong.COURT_LEN, 0.0), (0.0, -pong.COURT_LEN / 2))
        az_a, el_a = pong.view_of("A", 20.0, 30.0)
        az_b, el_b = pong.view_of("B", 20.0, 30.0)
        self.assertEqual((az_a, az_b), (30.0, -30.0))       # facing each other: left and right swap
        self.assertEqual(el_a, -el_b)

    def test_decoded_azimuth_maps_back_to_court_y(self):
        for side in ("A", "B"):
            az, _ = pong.view_of(side, 35.0, -17.0)
            self.assertEqual(pong.court_y_of(side, az), -17.0)

    def test_arena_draws_the_ball_where_the_fly_looks(self):
        # rays: straight ahead, 20 deg left, 20 deg left and 20 up, behind
        dirs = []
        for az, el in ((0, 0), (20, 0), (20, 20), (180, 0)):
            a, e = math.radians(az), math.radians(el)
            dirs.append((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
        arena = pong.Arena(np.array(dirs, np.float32))
        base = arena.radiance(None)
        np.testing.assert_allclose(base[0], pong.RAD_LINE)       # the net line crosses straight ahead (dashed at az 0)
        np.testing.assert_allclose(base[2], pong.RAD_COURT)
        np.testing.assert_allclose(base[3], pong.RAD_OUTSIDE)    # behind the fly: not the court
        rad = arena.radiance((20.0, 20.0))
        np.testing.assert_allclose(rad[2], pong.RAD_BALL)
        np.testing.assert_allclose(rad[1], base[1])              # 20 deg away from the ball: untouched


class CourtTests(unittest.TestCase):
    def serve_straight(self, receiver, y0=0.0):
        c = pong.Court(serves=[(receiver, 0.0, y0)])
        ev = c.step(pong.FIRST_SERVE, 0.01)
        self.assertEqual(ev[0]["kind"], "serve")
        return c

    def run_until(self, c, kinds, t0=1.0, dt=0.01, n=2000):
        t = t0
        for _ in range(n):
            t += dt
            for e in c.step(t, dt):
                if e["kind"] in kinds:
                    return e, t
        self.fail(f"no {kinds} event")

    def test_serve_starts_at_the_servers_end_toward_the_receiver(self):
        c = self.serve_straight("A", y0=5.0)
        self.assertAlmostEqual(c.X, pong.face_x("B"))
        self.assertLess(c.vX, 0)
        self.assertEqual(c.Y, 5.0)

    def test_a_paddle_on_the_ball_returns_it_faster(self):
        c = self.serve_straight("A", y0=4.0)
        c.paddle["A"] = 4.0
        e, _ = self.run_until(c, {"return", "miss"})
        self.assertEqual((e["kind"], e["side"]), ("return", "A"))
        self.assertGreater(c.vX, 0)
        self.assertAlmostEqual(c.speed, pong.V0 * pong.SPEEDUP)

    def test_english_follows_the_hit_offset(self):
        c = self.serve_straight("A", y0=4.0)
        c.paddle["A"] = 4.0 - 0.8 * (c.pad_half + pong.BALL_RY)    # the ball meets the upper part of the paddle
        e, _ = self.run_until(c, {"return", "miss"})
        self.assertAlmostEqual(e["offset"], 0.8, places=2)
        self.assertGreater(0.8 * pong.ENGLISH, pong.MIN_ANGLE)
        self.assertFalse(e["steepened"])
        self.assertAlmostEqual(math.degrees(math.atan2(c.vY, c.vX)), 0.8 * pong.ENGLISH, places=4)   # straight in: no angle to keep

    def test_a_flat_return_is_steepened_to_the_minimum_angle(self):
        c = pong.Court()
        for a_in, off, want in ((0.0, 0.1, pong.MIN_ANGLE), (-4.0, 0.0, -pong.MIN_ANGLE), (30.0, 0.0, pong.MIN_ANGLE),
                                (-30.0, -0.5, -(15.0 + 0.5 * pong.ENGLISH))):
            a, steep = c.return_angle(a_in, off)
            self.assertAlmostEqual(a, want, places=6)
            self.assertEqual(steep, abs(pong.ANGLE_KEEP * a_in + pong.ENGLISH * off) < pong.MIN_ANGLE)
        c.Y = 40.0                                           # exactly flat near the top wall: away from it
        self.assertEqual(c.return_angle(0.0, 0.0), (-pong.MIN_ANGLE, True))
        self.assertEqual(c.return_angle(80.0, 1.0)[0], pong.MAX_ANGLE)

    def tracked_rally(self, min_angle, seconds=30.0):
        """Paddles that follow the ball perfectly (clamped to the court, 80 deg/s), the ball served flat near the top
        wall: the reviewer's seed-111 attractor. Returns (ball Y per tick, |vY| after each return at the speed cap)."""
        c = pong.Court(serves=[("A", 2.0, 40.0)])
        c.min_angle = min_angle
        c.step(pong.FIRST_SERVE, 0.01)
        t, ys, vy_cap = pong.FIRST_SERVE, [], []
        for _ in range(int(seconds / 0.01)):
            t += 0.01
            for e in c.step(t, 0.01):
                if e["kind"] == "return" and e["speed_deg_s"] >= pong.VMAX - 1e-6:
                    vy_cap.append(abs(c.vY))
            for side in ("A", "B"):
                c.move_paddle(side, c.Y, 0.01)
            ys.append(c.Y)
        self.assertTrue(c.live)                              # perfect paddles never miss
        return np.asarray(ys), np.asarray(vy_cap)

    def test_the_flat_rally_attractor_is_gone(self):
        # without a minimum angle, returns off centred paddles flatten to 0 deg: the ball shuttles straight near the
        # top wall at the speed cap and the paddles need not move (review of seed 111)
        ys, vy = self.tracked_rally(0.0)
        self.assertLess(np.median(vy), 1.0)
        self.assertGreater(ys.min(), 20.0)
        # with it, every return at the cap moves sideways at >= 200 sin(20 deg) and the ball crosses the court
        ys, vy = self.tracked_rally(pong.MIN_ANGLE)
        self.assertGreaterEqual(vy.min(), pong.VMAX * math.sin(math.radians(pong.MIN_ANGLE)) - 1e-6)
        self.assertLess(ys.min(), -20.0)

    def test_a_paddle_elsewhere_misses_and_the_other_side_scores(self):
        c = self.serve_straight("B", y0=-30.0)
        c.paddle["B"] = 30.0
        e, _ = self.run_until(c, {"return", "miss"})
        self.assertEqual((e["kind"], e["side"]), ("miss", "B"))
        self.assertAlmostEqual(e["by_deg"], 60.0 - c.pad_half - pong.BALL_RY, places=6)
        e, t = self.run_until(c, {"point"}, t0=1.5)
        self.assertEqual(e["scorer"], "A")
        self.assertEqual(c.score, {"A": 1, "B": 0})
        self.assertFalse(c.live)
        self.assertAlmostEqual(c.next_serve_t, t + pong.SERVE_PAUSE)

    def test_side_walls_reflect(self):
        c = pong.Court(serves=[("A", 60.0, pong.HALF_W - pong.BALL_RY - 0.1)])
        c.step(pong.FIRST_SERVE, 0.01)
        self.run_until(c, {"wall"})
        self.assertLess(c.vY, 0)
        self.assertLessEqual(abs(c.Y), pong.HALF_W - pong.BALL_RY)

    def test_the_paddle_that_missed_stops_until_the_point(self):
        c = self.serve_straight("B", y0=-30.0)
        c.paddle["B"] = 30.0
        e, t = self.run_until(c, {"miss"})
        self.assertEqual(c.missed, "B")
        c.move_paddle("B", -30.0, 0.5)                       # would reach the ball; the point is already decided
        self.assertEqual(c.paddle["B"], 30.0)
        c.move_paddle("A", 10.0, 0.5)                        # the other paddle still moves
        self.assertGreater(c.paddle["A"], 0.0)
        e, t = self.run_until(c, {"point", "return"}, t0=t)
        self.assertEqual(e["kind"], "point")
        e, _ = self.run_until(c, {"serve"}, t0=t)
        self.assertIsNone(c.missed)

    def test_no_serves_once_the_game_is_decided(self):
        c = pong.Court()
        c.serving = False
        self.assertEqual(c.step(pong.FIRST_SERVE + 5, 0.01), [])
        self.assertFalse(c.live)

    def test_game_floor_is_deterministic_and_counts_every_contact(self):
        a, b = pong.game_floor(20.0), pong.game_floor(20.0)
        self.assertEqual(a, b)
        n_contacts = sum(a["returns"].values()) + sum(a["misses"].values())
        self.assertGreater(n_contacts, 0)
        self.assertEqual(sum(a["score"].values()), len(a["returns_per_point"]))
        self.assertEqual(sum(a["misses"].values()) - sum(a["score"].values()) in (0, 1), True)

    def test_paddle_speed_is_limited(self):
        c = pong.Court()
        c.move_paddle("A", 40.0, 0.01)
        self.assertAlmostEqual(c.paddle["A"], pong.PAD_SPEED * 0.01)
        for _ in range(200):
            c.move_paddle("A", 99.0, 0.01)
        self.assertAlmostEqual(c.paddle["A"], pong.HALF_W - c.pad_half)

    def test_the_serve_schedule_does_not_depend_on_anything_but_its_index(self):
        a, b = pong.Court(), pong.Court()
        for c in (a, b):
            c.paddle["A"], c.paddle["B"] = 17.0, -3.0
        self.assertEqual(a.serve(), b.serve())
        self.assertEqual(a.serves, pong.SERVES)


class DecoderTests(unittest.TestCase):
    def grid(self):
        az, el = np.meshgrid(np.arange(-50, 51, 5.0), np.arange(-35, 36, 5.0))
        return az.ravel(), el.ravel()

    def test_decoder_finds_a_transient_bump(self):
        az, el = self.grid()
        n = len(az)
        dec = pong.BallDecoder(np.arange(n), az, el, gate=0.1, target_tau_s=1e-6)
        dec.update(torch.zeros(n), 0.01)                     # the running mean starts at rest
        bump = torch.as_tensor(np.exp(-((az - 20) ** 2 + (el + 10) ** 2) / (2 * 6.0 ** 2)), dtype=torch.float32)
        noise = torch.zeros(n)
        noise[np.argmin(np.hypot(az + 45, el - 30))] = 0.45   # one strong outlier far away: not the densest cluster
        dec.update(bump + noise, 0.01)
        self.assertTrue(dec.valid)
        self.assertAlmostEqual(dec.az_hat, 20.0, delta=2.0)
        self.assertAlmostEqual(dec.el_hat, -10.0, delta=2.0)
        self.assertAlmostEqual(dec.target_az, dec.az_hat, places=3)

    def test_decoder_holds_below_the_gate(self):
        az, el = self.grid()
        n = len(az)
        dec = pong.BallDecoder(np.arange(n), az, el, gate=0.5)
        dec.update(torch.zeros(n), 0.01)
        dec.target_az = 7.0
        x = torch.zeros(n)
        x[0] = 0.2
        dec.update(x, 0.01)
        self.assertFalse(dec.valid)
        self.assertEqual(dec.target_az, 7.0)

    def test_steady_activity_is_high_passed_away(self):
        az, el = self.grid()
        n = len(az)
        dec = pong.BallDecoder(np.arange(n), az, el, gate=0.05, highpass_tau_s=0.05)
        x = torch.full((n,), 0.3)
        x[5] = 0.8
        dec.update(x, 0.01)
        for _ in range(200):
            dec.update(x, 0.01)
        self.assertLess(dec.conf, 1e-3)

    def test_decoder_reads_only_its_cells(self):
        dec = pong.BallDecoder([3, 1], [10.0, -10.0], [0.0, 0.0], gate=0.1)
        dr = torch.zeros(6)
        dec.update(dr, 0.01)
        dr[0] = 5.0                                          # not a decoder cell
        dr[1] = 0.6                                          # the cell looking at az -10
        dec.update(dr, 0.01)
        self.assertAlmostEqual(dec.az_hat, -10.0, places=4)


class HelperTests(unittest.TestCase):
    def test_near_misses_never_read_zero(self):
        self.assertEqual(pong.fmt_deg(0.43), "0.4°")
        self.assertEqual(pong.fmt_deg(1.96), "2.0°")
        self.assertEqual(pong.fmt_deg(26.2), "26°")

    def test_trail_is_cut_to_its_length_from_the_newest_point(self):
        pts = [(0.0, 0.0), (100.0, 0.0), (300.0, 0.0), (400.0, 0.0)]
        out = pong.truncate_polyline(pts, 250.0)
        np.testing.assert_allclose(out[-1], (400.0, 0.0))
        np.testing.assert_allclose(out[0], (150.0, 0.0))
        self.assertEqual(len(pong.truncate_polyline(pts[:1], 10.0)), 1)

    def test_error_stats_ignore_nan_and_respect_the_mask(self):
        e = [1.0, 2.0, float("nan"), 10.0]
        self.assertEqual(pong.err_stats(e)["n"], 3)
        self.assertEqual(pong.err_stats(e, [True, True, True, False])["median"], 1.5)
        self.assertEqual(pong.err_stats([], None), {"median": None, "p90": None, "n": 0})

    def test_photoreceptor_bins_cover_every_count_once(self):
        for n in (0, 1, 3, 4, 8, 9, 20, 21, 500):
            hits = [lo <= n <= hi for lo, hi in pong.PR_BINS]
            self.assertEqual(sum(hits), 1, n)
        self.assertEqual([pong.pr_bin_label(*b) for b in pong.PR_BINS], ["0", "1-3", "4-8", "9-20", "21+"])


if __name__ == "__main__":
    unittest.main()
