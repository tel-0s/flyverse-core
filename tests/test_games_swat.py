"""games/swat.py: the swatter's geometry and state machine, the one-clock referee and the outcome rule, the display
cameras, the oriented-ellipsoid ray tracer and the camera convention -- on CPU, with no dataset and no brain."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "games"))
import swat  # noqa: E402
from flyverse import world as W  # noqa: E402

DT = 0.01


class GeometryTests(unittest.TestCase):
    def test_approach_directions_in_the_fly_frame(self):
        np.testing.assert_allclose(swat.approach_dir(0.3, 0, 90), [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(swat.approach_dir(0.0, 90, 0), [0, 1, 0], atol=1e-12)        # + azimuth = left
        np.testing.assert_allclose(swat.approach_dir(math.pi / 2, 0, 0), [0, 1, 0], atol=1e-12)  # ahead of the fly
        u = swat.approach_dir(1.0, -90, 35)
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0)
        self.assertAlmostEqual(math.degrees(math.asin(u[2])), 35.0)

    def test_contact_distance_is_the_4mm_gap(self):
        R = np.eye(3)
        self.assertAlmostEqual(swat.contact_distance([0, 0, 1], R), 0.008)       # 4 mm half-thickness + 4 mm gap
        self.assertAlmostEqual(swat.contact_distance([1, 0, 0], R), 0.064)       # along the long axis
        self.assertAlmostEqual(swat.contact_distance([1, 0, 0], swat.rot_z(math.pi / 2)), 0.049)  # short axis now
        c = np.array([0.1, 0.2, 0.3])
        self.assertAlmostEqual(swat.inflated_metric(c + [0, 0, 0.008], c, R, swat.PADDLE_RADII, 0.004), 1.0)
        self.assertLess(swat.inflated_metric(c + [0.02, 0, 0.007], c, R, swat.PADDLE_RADII, 0.004), 1.0)

    def test_slerp_and_clamp(self):
        a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        np.testing.assert_allclose(swat.slerp(a, b, 0), a, atol=1e-9)
        np.testing.assert_allclose(swat.slerp(a, b, 1), b, atol=1e-9)
        m = swat.slerp(a, -a, 0.5)                                                # opposite: over the top
        np.testing.assert_allclose(m, [0, 0, 1], atol=1e-9)
        u = swat.clamp_elevation([1.0, 0, -0.5], 20, 90)
        self.assertAlmostEqual(math.degrees(math.asin(u[2])), 20.0)
        self.assertAlmostEqual(swat.windup_speed(0.0), swat.V_MIN)
        self.assertAlmostEqual(swat.windup_speed(10.0), swat.V_MAX)

    def test_aim_point_on_the_sphere(self):
        c = np.zeros(3)
        p = swat.aim_point((2.0, 0, 0), (-1, 0, 0), c, 0.45)                       # outside: the near crossing
        np.testing.assert_allclose(p, [0.45, 0, 0], atol=1e-12)
        p = swat.aim_point((0.1, 0, 0), (1, 0, 0), c, 0.45)                        # inside: the exit
        np.testing.assert_allclose(p, [0.45, 0, 0], atol=1e-12)
        p = swat.aim_point((2.0, 0, 1.0), (-1, 0, 0), c, 0.45)                     # a miss: the nearest point
        np.testing.assert_allclose(p, [0, 0, 0.45], atol=1e-12)

    def test_handle_starts_beyond_the_rim(self):
        c, R = swat.handle_pose(np.zeros(3), 0.0)
        self.assertGreater(c[0], swat.PADDLE_RADII[0] + swat.HANDLE_RADII[0] * 0.9)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)

    def test_fly_model_is_fly_sized(self):
        parts = swat.fly_parts(np.array([0.0, 0.0, 0.75]), [1, 0, 0], [0, 1, 0], [0, 0, 1])
        self.assertEqual(len(parts), swat.N_FLY_PARTS)
        for c, r, R, mat in parts:
            self.assertLess(np.linalg.norm(c - [0, 0, 0.75]), 0.003)
            self.assertLess(max(r), 0.0012)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
            self.assertTrue(mat in swat.MATERIALS or mat in W.MATERIALS)


class SwatterTests(unittest.TestCase):
    def swing(self, v, eye_path, u=(0, 0, 1)):
        eye0 = np.array([0.0, 0.0, 0.7512])
        s = swat.Swatter(u, eye0, 0.0)
        s.start_swing(v, eye0)
        for k in range(1, 400):
            eye, airborne = eye_path(k * DT)
            if s.step(DT, eye, airborne) == "contact":
                return k * DT, s
        self.fail("no contact")

    def test_grounded_contact_time(self):
        for v in (0.5, 1.0, 2.5):
            t, s = self.swing(v, lambda t: (np.array([0.0, 0.0, 0.7512]), False))
            nominal = (swat.START_DIST - 0.008) / v
            self.assertLessEqual(abs(t - nominal), DT + 1e-9)
            self.assertAlmostEqual(s.dist, 0.008)

    def test_swing_tracks_a_walking_fly_until_takeoff(self):
        def walk(t):
            return np.array([0.015 * t, 0.0, 0.7512]), False
        t, s = self.swing(1.0, walk)
        self.assertAlmostEqual(s.anchor[0], 0.015 * t, places=6)

    def test_after_contact_the_swatter_lifts_then_moves_round(self):
        eye = np.array([0.0, 0.0, 0.7512])
        s = swat.Swatter([0, 0, 1], eye, 0.0)
        s.start_swing(2.0, eye)
        while s.step(DT, eye, False) != "contact":
            pass
        s.set_phase("pin")
        phases = []
        for _ in range(200):
            s.step(DT, eye, False)
            phases.append(s.phase)
            if s.phase == "hold":
                s.start_move(swat.approach_dir(0.0, 90, 35))
        self.assertIn("lift", phases)
        self.assertIn("move", phases)
        self.assertEqual(phases[-1], "aim")
        np.testing.assert_allclose(s.u, swat.approach_dir(0.0, 90, 35), atol=1e-9)
        self.assertAlmostEqual(s.dist, swat.START_DIST)

    def test_outcome_rule(self):
        self.assertEqual(swat.swat_outcome(None), ("splat", None))         # no takeoff before the paddle arrived
        self.assertEqual(swat.swat_outcome(0), ("splat", None))            # a takeoff on the arrival tick: a tie
        self.assertEqual(swat.swat_outcome(1), ("in_time", 10))
        self.assertEqual(swat.swat_outcome(8), ("in_time", 80))

    def test_ticks_until_arrival(self):
        self.assertEqual(swat.ticks_until_arrival(0.0, 1.0, DT), 0)
        self.assertEqual(swat.ticks_until_arrival(0.01, 1.0, DT), 1)       # exactly one tick of travel
        self.assertEqual(swat.ticks_until_arrival(0.0101, 1.0, DT), 2)
        self.assertEqual(swat.ticks_until_arrival(0.442, 1.5, DT), 30)     # 29.47 ticks: arrives on the 30th

    def test_schedule_is_fixed_and_rising(self):
        speeds = [row[3] for row in swat.SCHEDULE]
        self.assertEqual(speeds, sorted(speeds))
        self.assertTrue(all(20 <= row[2] <= 90 for row in swat.SCHEDULE))


class ClockTests(unittest.TestCase):
    """`referee_step` is the order SwatGame.tick uses every tick, after the brain has stepped. A stand-in body stands
    still, takes off on a chosen tick (the shipped launch: +3 mm, 0.6 m/s at 45 deg forward, reduced gravity 3 m/s^2)
    and flies ballistically."""
    EYE0 = np.array([0.0, 0.0, 0.7512])

    def arrival_tick(self, v, u=(0, 0, 1)):
        s = swat.Swatter(u, self.EYE0, 0.0)
        return swat.ticks_until_arrival(swat.START_DIST - s.d_contact, v, DT)

    def play(self, v, k_takeoff=None, u=(0, 0, 1)):
        s = swat.Swatter(u, self.EYE0, 0.0)
        s.start_swing(v, self.EYE0)                                  # at the end of tick 0, as SwatGame._swing
        st = {"eye": self.EYE0.copy(), "air": False, "vel": None}
        took = None
        for k in range(1, 400):
            def body(blocked, k=k):
                if st["air"]:
                    st["vel"][2] -= 3.0 * DT
                    st["eye"] = st["eye"] + st["vel"] * DT
                    return False, st["eye"], True
                if k == k_takeoff and not blocked:
                    st["air"] = True
                    st["eye"] = st["eye"] + np.array([0.0, 0.0, 0.003])
                    st["vel"] = 0.6 * np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4)])
                    return True, st["eye"], True
                return False, st["eye"], False
            out = swat.referee_step(s, DT, st["eye"], st["air"], body)
            if out["took"]:
                took = (k, out)
            if out["arrived"] or out["air"]:
                self.last_eye = st["eye"].copy()
                return k, out, took, s
        self.fail("the swing never ended")

    def test_a_standing_fly_is_reached_on_the_arrival_tick(self):
        for v in (0.5, 1.0, 1.5, 2.0, 2.5):
            for u in ((0, 0, 1), swat.approach_dir(0.0, 90, 35), swat.approach_dir(0.0, 0, 45)):
                k, out, took, _ = self.play(v, None, u)
                self.assertEqual(k, self.arrival_tick(v, u))
                self.assertEqual(out["arrived"], "table")
                self.assertIsNone(took)

    def test_a_takeoff_on_the_arrival_tick_is_a_splat(self):
        for v in (0.5, 1.5, 2.5):
            K = self.arrival_tick(v)
            k, out, took, _ = self.play(v, K)
            self.assertEqual((k, out["arrived"]), (K, "table"))
            self.assertIsNone(took)                                  # the body was blocked: a tie goes to the paddle

    def test_one_tick_early_is_in_time_by_10_ms(self):
        for v in (0.5, 1.5, 2.5):
            K = self.arrival_tick(v)
            _, _, took, _ = self.play(v, K - 1)
            kt, ref = took
            self.assertEqual((kt, ref["ticks_to_arrival"]), (K - 1, 1))
            self.assertEqual(swat.swat_outcome(ref["ticks_to_arrival"]), ("in_time", 10))
            self.assertGreater(ref["ttc_ms"], 0.0)
            self.assertLessEqual(ref["ttc_ms"], 10.0 + 1e-9)

    def test_the_margin_reads_the_paddle_at_the_takeoff_tick(self):
        for v, u in ((0.5, (0, 0, 1)), (1.5, (0, 0, 1)), (2.0, swat.approach_dir(0.0, 90, 35))):
            s0 = swat.Swatter(u, self.EYE0, 0.0)
            t_nominal = (swat.START_DIST - s0.d_contact) / v              # s after the swing started (tick 0)
            K = self.arrival_tick(v, u)
            for j in (K - 9, K - 4, K - 2):
                _, _, took, s = self.play(v, j, u)
                kt, ref = took
                self.assertEqual(kt, j)
                self.assertAlmostEqual(ref["ttc_ms"], (t_nominal - j * DT) * 1000, places=6)
                self.assertEqual(ref["ticks_to_arrival"], K - j)
                _, margin = swat.swat_outcome(ref["ticks_to_arrival"])
                self.assertEqual(margin, (K - j) * 10)
                self.assertTrue(0 <= margin - ref["ttc_ms"] < 10)       # whole ticks, never a tick generous
                self.assertTrue(s.took_off)

    def test_the_hop_meets_a_paddle_from_above_in_the_air(self):
        v = 1.0
        K = self.arrival_tick(v)
        k, out, took, s = self.play(v, K - 8)
        self.assertTrue(out["air"])
        self.assertIsNone(out["arrived"])
        self.assertLess(k, K)
        self.assertEqual(took[1]["ticks_to_arrival"], 8)
        np.testing.assert_allclose(s.anchor, self.EYE0)              # the swing stopped tracking at takeoff

    def test_a_fast_paddle_cannot_pass_through_the_hop(self):
        for v in (1.5, 2.0, 2.5):
            K = self.arrival_tick(v)
            for lead in (1, 2, 3, 4):
                k, out, took, s = self.play(v, K - lead)
                self.assertTrue(out["air"], (v, lead))                  # met in the air, not 'reached the spot'
                self.assertIsNone(out["arrived"])
                eye = self.last_eye
                self.assertAlmostEqual(swat.inflated_metric(eye, s.center, s.rot, swat.PADDLE_RADII, swat.CONTACT_GAP),
                                       1.0, places=6)                   # shown touching the shell ...
                self.assertLess(float((eye - s.center) @ s.u), 0.0)    # ... from the anchor's side, not through it

    def test_swept_touch(self):
        R = np.eye(3)
        self.assertTrue(swat.swept_touch([0, 0, -0.02], [0, 0, 0.02], R))      # straight through in one tick
        self.assertFalse(swat.swept_touch([0.2, 0, -0.02], [0.2, 0, 0.02], R))  # beside it
        self.assertTrue(swat.swept_touch([0, 0, -0.007], [0, 0, -0.0075], R))  # inside the 4 mm shell
        self.assertAlmostEqual(swat.touch_dist([0, 0, 0.0], [0, 0, 1], R), 0.008)

    def test_hero_cameras_stay_under_the_paddle_and_frame_the_fly(self):
        eye = np.array([0.0, 0.0, 0.7512])
        for cam, rows in ((swat.HERO_CAM, 968), (swat.REPLAY_CAM, 968)):
            tan = math.tan(math.radians(cam["fov"]) / 2)
            for ph in (0.004, 0.01, 0.03, 0.3):
                tgt, dist, el = swat.hero_goal(eye, False, eye[2], eye[2] + ph, cam)
                z = tgt[2] + dist * math.sin(math.radians(el))
                self.assertLess(z, eye[2] + ph)                        # below the paddle's underside
                self.assertGreater(z, eye[2])
                D = dist * math.cos(math.radians(el))
                self.assertGreaterEqual(0.0026 * rows / (2 * D * tan), 150)   # a 2.6 mm fly >= 150 px tall frame
        self.assertGreaterEqual(0.0026 * 290 / (2 * swat.FLYCAM_D * math.tan(math.radians(swat.FLYCAM_FOV) / 2)), 120)


class TracerTests(unittest.TestCase):
    def world(self, rot):
        base = W.World(device="cpu", detail=0.0)
        w = swat.make_world(base, [((1.0, 0.0, 0.0), (0.5, 0.05, 0.05), rot, "black")], device="cpu")
        w._pack()
        return w

    def hit(self, w, o, d):
        t, n, mid = w._intersect(torch.tensor([o], dtype=torch.float32), torch.tensor([d], dtype=torch.float32))
        return float(t[0]), n[0].numpy(), int(mid[0])

    def test_unrotated_matches_world(self):
        t, n, mid = self.hit(self.world(None), (0, 0, 0), (1, 0, 0))
        self.assertAlmostEqual(t, 0.5, places=5)
        np.testing.assert_allclose(n, [-1, 0, 0], atol=1e-5)
        self.assertEqual(mid, 0)

    def test_rotated_ellipsoid(self):
        w = self.world(swat.rot_z(math.pi / 2))                   # long axis now along y: thin along x
        t, n, _ = self.hit(w, (0, 0, 0), (1, 0, 0))
        self.assertAlmostEqual(t, 0.95, places=5)
        np.testing.assert_allclose(n, [-1, 0, 0], atol=1e-5)
        t, n, _ = self.hit(w, (1.0, -2.0, 0.0), (0, 1, 0))       # along the long axis
        self.assertAlmostEqual(t, 1.5, places=5)
        np.testing.assert_allclose(n, [0, -1, 0], atol=1e-5)
        w.move(0, (1.0, 0.0, 1.0))                                 # moved out of the way
        t, _, mid = self.hit(w, (0, 0, 0), (1, 0, 0))
        self.assertGreater(t, 1e8)

    def test_render_shape_and_camera_convention(self):
        w = self.world(None)
        cam = swat.Camera((0, 0, 0), (1, 0, 0), 32, 24, 60)
        img = w.render(cam, chunk=100)
        self.assertEqual(tuple(img.shape), (24, 32, 4))
        cx, cy = cam.project((1, 0, 0))
        self.assertAlmostEqual(cx, 15.5); self.assertAlmostEqual(cy, 11.5)
        self.assertLess(cam.project((1, 0.2, 0))[0], 15.5)       # the world's +y (left) is on the image's left
        self.assertLess(cam.project((1, 0, 0.2))[1], 11.5)       # up is up
        d = cam.ray_through(0, 0)
        self.assertGreater(d[1], 0); self.assertGreater(d[2], 0)
        rays = cam.rays("cpu").reshape(24, 32, 3).numpy()
        np.testing.assert_allclose(rays[0, 0], d, atol=1e-5)


class ArgsTests(unittest.TestCase):
    def test_parse_args(self):
        a = swat.parse_args(["--seed", "100", "--record", "x.mp4", "--control"])
        self.assertEqual(a.seed, 100)
        self.assertTrue(a.control)
        self.assertEqual(a.seconds, 40.0)
        self.assertEqual(tuple(a.start), (0.5, 0.0, 180.0))
        self.assertEqual(a.fruit, "apple")
        self.assertEqual(a.from_swat, 1)


if __name__ == "__main__":
    unittest.main()
