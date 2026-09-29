"""games/spacecraft.py: rigid-body physics, the decoder laws on synthetic rates, the seed-drawn scenario, the rock
shape and contact model, and the ray-object intersections -- on CPU, with no dataset and no brain."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "games"))
import spacecraft as sc  # noqa: E402

DT = 0.01


def means(**kw):
    """Mean rates for every population the attitude / dodge laws read, zero unless given."""
    out = {f"{t}_{s}": 0.0 for t in list(sc.YAW_CELLS) + list(sc.ROLL_CELLS) + list(sc.LOOM_CELLS) for s in "LR"}
    out.update(kw)
    return out


def ball_rock(pos, vel, radius=2.0, name="r"):
    """A rock whose lobes are all the same sphere (so its shape is exact and easy to reason about)."""
    lobe_c = np.zeros((sc.N_LOBES, 3))
    lobe_r = np.full((sc.N_LOBES, 3), float(radius))
    return sc.Rock(name, np.asarray(pos, float), np.asarray(vel, float), lobe_c, lobe_r, np.array([0, 0, 1.0]), 0.0,
                   sc.ROCK_ALBEDO, 1.0, threat=True)


def still_ship():
    return sc.ShipState(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3))


class MathTests(unittest.TestCase):
    def test_quaternion_rotates_x_to_y_about_z(self):
        R = sc.quat_to_matrix(sc.quat_from_axis_angle((0, 0, 1), math.pi / 2))
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_look_basis_is_right_handed(self):
        f, left, up = sc.look_basis((1, 1, 0.2))
        np.testing.assert_allclose(np.cross(f, left), up, atol=1e-12)
        self.assertAlmostEqual(float(np.dot(f, up)), 0.0, places=12)


class PhysicsTests(unittest.TestCase):
    def test_torque_free_tumble_conserves_momentum_and_energy(self):
        s = sc.ShipState(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.deg2rad([40.0, 18.0, -45.0]))
        L0 = np.linalg.norm(sc.angular_momentum_world(s))
        E0 = 0.5 * float(np.sum(sc.INERTIA * s.w ** 2))
        for _ in range(2000):
            s = sc.rigid_step(s, np.zeros(3), np.zeros(3), DT)
        self.assertAlmostEqual(np.linalg.norm(sc.angular_momentum_world(s)) / L0, 1.0, places=2)
        self.assertAlmostEqual(0.5 * float(np.sum(sc.INERTIA * s.w ** 2)) / E0, 1.0, places=2)
        self.assertAlmostEqual(float(np.linalg.norm(s.q)), 1.0, places=9)

    def test_damping_torque_on_all_axes_stops_the_tumble(self):
        s = sc.ShipState(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.deg2rad([40.0, 18.0, -45.0]))
        for _ in range(1000):
            s = sc.rigid_step(s, -2.0 * s.w, np.zeros(3), DT)
        self.assertLess(np.linalg.norm(s.w), np.deg2rad(1.0))

    def test_linear_motion_and_autopilot(self):
        s = sc.ShipState(np.zeros(3), np.array([10.0, 0, 0]), np.array([1.0, 0, 0, 0]), np.zeros(3))
        a = sc.cruise_autopilot(s.vel)
        self.assertLessEqual(np.linalg.norm(a), sc.CRUISE_AMAX + 1e-12)
        self.assertLess(a[0], 0)                                          # slows toward the cruise speed
        s2 = sc.rigid_step(s, np.zeros(3), np.zeros(3), 0.5)
        np.testing.assert_allclose(s2.pos, [5.0, 0, 0])

    def test_predict_position_is_the_autopilot_integrated(self):
        s = sc.ShipState(np.array([1.0, 2.0, 3.0]), np.array([9.0, -4.0, 1.0]), np.array([1.0, 0, 0, 0]), np.zeros(3))
        p = sc.predict_position(s.pos, s.vel, 3.0)
        for _ in range(300):
            s = sc.rigid_step(s, np.zeros(3), sc.cruise_autopilot(s.vel), DT)
        np.testing.assert_allclose(p, s.pos, atol=1e-9)


class DecoderTests(unittest.TestCase):
    def test_weighted_lr_zero_activity_is_zero(self):
        self.assertEqual(sc.weighted_lr(means(), sc.YAW_CELLS, sc.BETA["yaw"]), 0.0)
        self.assertEqual(sc.weighted_lr(means(), sc.ROLL_CELLS, sc.BETA["roll"]), 0.0)

    def test_side_equalisation(self):
        # equal left and right activity reads as -beta * (L + R), not zero: the right side's excess is expected
        m = means(VST2_L=10.0, VST2_R=10.0)
        self.assertAlmostEqual(sc.weighted_lr(m, sc.ROLL_CELLS, -0.31), 0.31 * 20.0)
        m = means(H2_L=4.0, H2_R=1.0)
        self.assertAlmostEqual(sc.weighted_lr(m, sc.YAW_CELLS), 3.0)

    def test_attitude_law_opposes_the_signal_and_never_pitches(self):
        law = sc.AttitudeLaw(g_roll=5.0, g_yaw=4.0, tau_s=0.0)
        out = law.from_means(means(VST2_L=8.0, H2_L=6.0), DT)
        self.assertGreater(out["roll_sig"], 0)
        self.assertLess(out["alpha"][0], 0)                                # positive roll signal -> negative roll torque
        self.assertLess(out["alpha"][2], 0)
        self.assertEqual(out["alpha"][1], 0.0)                              # pitch is not built
        law0 = sc.AttitudeLaw(tau_s=0.0)
        np.testing.assert_allclose(law0.from_means(means(), DT)["alpha"], 0.0)

    def test_attitude_low_pass(self):
        law = sc.AttitudeLaw(g_roll=5.0, g_yaw=4.0, tau_s=0.5)
        first = law.from_means(means(VST2_L=10.0), DT)["roll_sig"]
        for _ in range(1000):
            last = law.from_means(means(VST2_L=10.0), DT)["roll_sig"]
        self.assertLess(first, 0.05 * last)
        self.assertAlmostEqual(last, 10.0 - sc.BETA["roll"] * 10.0, places=3)

    def test_attitude_clip_saturates_the_signal(self):
        law = sc.AttitudeLaw(g_roll=3.0, g_yaw=4.0, tau_s=0.0, clip_hz=3.0)
        out = law.from_means(means(VST2_L=20.0), DT)
        self.assertAlmostEqual(out["roll_sig"], 3.0)
        self.assertAlmostEqual(out["alpha"][0], -9.0 * sc.DEG)
        self.assertEqual(sc.AttitudeLaw().g_roll, 3.0)                     # the shipped gains
        self.assertEqual(sc.AttitudeLaw().g_yaw, 4.0)

    def test_dodge_fires_away_from_the_looming_side(self):
        law = sc.DodgeLaw(threshold_hz=5.0, tau_s=0.0)
        self.assertEqual(law.from_means(means(LPLC2_L=12.0, LC4_L=2.0), DT)["fire"], 1)
        self.assertEqual(law.from_means(means(LPLC2_R=12.0, LC4_R=2.0), DT)["fire"], -1)
        self.assertEqual(law.from_means(means(LPLC2_L=6.0, LC4_L=2.0), DT)["fire"], 0)   # side 4 Hz < 5

    def test_escape_law_is_the_body_threshold_on_the_mean(self):
        law = sc.EscapeLaw(33.0)
        self.assertTrue(law.from_means(20.0, 46.0)["fire"])
        self.assertFalse(law.from_means(10.0, 52.0)["fire"])                # one hot cell is not enough
        self.assertAlmostEqual(law.from_means(20.0, 46.0)["gf"], 33.0)


class ScenarioTests(unittest.TestCase):
    def test_the_seed_draws_the_scenario_reproducibly(self):
        a, b, c = sc.make_scenario(0), sc.make_scenario(0), sc.make_scenario(1)
        np.testing.assert_array_equal(a.w0_deg, b.w0_deg)
        self.assertEqual([t.describe() for t in a.threats], [t.describe() for t in b.threats])
        self.assertFalse(np.allclose(a.w0_deg, c.w0_deg))
        self.assertNotEqual([t.az for t in a.threats], [t.az for t in c.threats])

    def test_scenario_ranges(self):
        for seed in range(20):
            s = sc.make_scenario(seed)
            self.assertTrue(55.0 <= np.linalg.norm(s.w0_deg) <= 70.0)
            self.assertEqual([t.t_arrive for t in s.threats], [slot[0] for slot in sc.THREAT_SLOTS])
            for th, (_, speed, _, kind) in zip(s.threats, sc.THREAT_SLOTS):
                self.assertEqual(th.speed, speed)
                rel = th.az - (sc.CAM_AZ0 + sc.CAM_AZ_RATE * th.t_arrive)
                self.assertTrue(-26.0 <= rel <= 6.0)                         # near the camera's line of sight
                lo, hi = (0.8, 2.2) if kind == "hit" else (5.5, 7.5)
                self.assertTrue(lo <= th.offset <= hi)

    def test_still_scenario(self):
        s = sc.make_scenario(3, still=True)
        np.testing.assert_array_equal(s.w0_deg, 0.0)
        self.assertEqual(s.threats, [])

    def test_threat_passes_its_offset_from_the_predicted_position_on_time(self):
        th = sc.make_scenario(5).threats[1]
        pos, vel, t_now = np.array([1.0, 2.0, 3.0]), np.array([6.0, 1.0, 0.0]), th.t_arrive - sc.LAUNCH_LEAD
        r = sc.make_threat("rock 2", th, pos, vel, t_now)
        at = r.pos + r.vel * sc.LAUNCH_LEAD
        pred = sc.predict_position(pos, vel, sc.LAUNCH_LEAD)
        self.assertAlmostEqual(float(np.linalg.norm(at - pred)), th.offset, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(r.vel)), th.speed)
        # it comes from beyond the ship, as seen from the chase camera
        self.assertGreater(float(np.dot(r.pos - pred, sc.cam_look(th.t_arrive))), 0.0)


class RockTests(unittest.TestCase):
    def test_rock_shape_and_bound(self):
        lobe_c, lobe_r = sc.rock_shape(np.random.default_rng(0), 2.6)
        self.assertEqual(lobe_c.shape, (sc.N_LOBES, 3))
        np.testing.assert_array_equal(lobe_c[0], 0.0)
        r = sc.Rock("r", np.zeros(3), np.zeros(3), lobe_c, lobe_r, np.array([0, 0, 1.0]), 0.1, sc.ROCK_ALBEDO, 0.0)
        # every lobe's surface lies inside the bound
        for c, rad in zip(lobe_c, lobe_r):
            self.assertLessEqual(np.linalg.norm(c) + rad.max(), r.bound + 1e-12)
        self.assertEqual(len(sc.SpaceRenderer.pack_rock(r.render_spec())), sc.SpaceRenderer.ROW)

    def test_ship_collision_points(self):
        self.assertEqual(len(sc.SHIP_POINTS), len(sc.SHIP_POINT_PART))
        self.assertTrue(4.3 < sc.SHIP_BOUND < 5.5)
        parts = set(sc.SHIP_POINT_PART)
        self.assertTrue({"fuselage", "left wing", "right wing", "canopy", "tail fin"} <= parts)
        self.assertGreater(sc.SHIP_POINTS[:, 1].max(), 3.0)                  # the wings reach out sideways

    def test_proximity_far_near_and_touching(self):
        ship = still_ship()
        far = ball_rock((60.0, 0, 0), (-20.0, 0, 0))
        gap, contact = sc.rock_proximity(ship.pos, ship.R, far)
        self.assertIsNone(contact)
        self.assertAlmostEqual(gap, 60.0 - sc.SHIP_BOUND - 2.0)
        near = ball_rock((4.3 + 2.0 + 1.0, 0, 0), (-20.0, 0, 0))            # 1 m off the nose
        gap, contact = sc.rock_proximity(ship.pos, ship.R, near)
        self.assertAlmostEqual(gap, 1.0, delta=0.1)
        self.assertEqual(contact[2], "fuselage")
        touching = ball_rock((4.3 + 2.0 - 0.2, 0, 0), (-20.0, 0, 0))
        gap, contact = sc.rock_proximity(ship.pos, ship.R, touching)
        self.assertLess(gap, 0)
        np.testing.assert_allclose(contact[1], [-1.0, 0, 0], atol=0.05)     # normal from the rock toward the ship

    def test_collision_bounces_and_kicks_about_the_lever(self):
        ship = still_ship()
        rock = ball_rock((-0.8, 0.0, 1.67), (0.0, 0.0, -20.0), radius=1.0)  # lands on the fuselage behind the middle
        hit = sc.collide(ship, rock)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["part"], "fuselage")
        self.assertLess(ship.vel[2], 0)                                     # pushed away from the rock
        self.assertGreater(float(np.dot(rock.vel - ship.vel, rock.pos - ship.pos)), 0)   # now separating
        self.assertGreater(abs(ship.w[1]), 5 * max(abs(ship.w[0]), abs(ship.w[2])))      # a blow off the middle pitches
        self.assertGreater(abs(ship.w[1]), 0.1)
        self.assertIsNone(sc.collide(still_ship(), ball_rock((50.0, 0, 0), (-20.0, 0, 0))))
        # separating rocks do not bounce
        self.assertIsNone(sc.collide(still_ship(), ball_rock((-0.8, 0.0, 1.67), (0.0, 0.0, 20.0), radius=1.0)))

    def test_a_blow_through_the_centre_barely_spins(self):
        ship = still_ship()
        hit = sc.collide(ship, ball_rock((4.3 + 2.0 - 0.1, 0, 0), (-20.0, 0, 0)))
        self.assertIsNotNone(hit)
        self.assertLess(np.linalg.norm(ship.w), 0.06)                       # only the sampled tip's small lever


class HistoryTests(unittest.TestCase):
    def test_series_grows_and_views(self):
        s = sc.Series(3, cap=4)
        for k in range(10):
            s.append([k, 2 * k, 3 * k])
        self.assertEqual(len(s), 10)
        np.testing.assert_array_equal(s.tail(2), [[8, 16, 24], [9, 18, 27]])
        self.assertEqual(s.all().shape, (10, 3))


class RayTests(unittest.TestCase):
    def test_ellipsoid_hits_agree_numpy_and_torch(self):
        o = np.zeros(3)
        d = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0.1, 0.0]])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c, radii, rot = np.array([10.0, 0, 0]), np.array([2.0, 2.0, 2.0]), sc.rot_axis((0, 0, 1), 0.4)
        t_np = sc.EyeScene.ellipsoid(o, d, c, radii, rot)
        self.assertAlmostEqual(t_np[0], 8.0, places=9)
        self.assertTrue(np.isinf(t_np[1]))
        t_t = sc.SpaceRenderer._ellipsoid_hit(torch.zeros(3, dtype=torch.float64), torch.tensor(d),
                                               torch.tensor(c), torch.tensor(radii), torch.tensor(rot))
        np.testing.assert_allclose(t_t.numpy(), t_np, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
