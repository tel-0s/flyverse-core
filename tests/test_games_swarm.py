"""games/swarm.py: the swatter's geometry and kinematics, the aim, the outcome rule and the camera -- pure logic, no
brain, no dataset, no GPU."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "games"))
import swarm as S  # noqa: E402


class GeometryTests(unittest.TestCase):
    def test_footprint_is_the_head_ellipse_with_a_margin(self):
        rx, ry, _ = S.head_radii("-x")
        self.assertEqual((rx, ry), (S.HEAD_ALONG, S.HEAD_ACROSS))           # long axis along the handle
        self.assertEqual(S.head_radii("+y")[:2], (S.HEAD_ACROSS, S.HEAD_ALONG))
        aim = (0.02, -0.01)
        pts = [aim, (aim[0] + rx - 0.001, aim[1]), (aim[0] + rx + 0.001, aim[1]), (aim[0] + rx + 0.002, aim[1]),
               (aim[0], aim[1] + ry + 0.01)]
        np.testing.assert_array_equal(S.footprint(pts, aim, "-x"), [True, True, True, False, False])
        np.testing.assert_allclose(S.rho([(aim[0] + rx, aim[1]), (aim[0], aim[1] - ry)], aim, "-x"), [1, 1])

    def test_handle_rides_above_the_rim_when_the_head_rests(self):
        for h in S.HANDLE_DIRS:
            off = S.handle_offset(h)
            rest_z = S.HEAD_THICK                                          # head centre above the tray when resting
            bottom = rest_z + off[2] - S.handle_radii(h)[2]
            self.assertGreater(bottom, S.RIM_H)
            d = np.array(S.HANDLE_DIRS[h])
            self.assertGreater(float(off[:2] @ d), S.HEAD_ALONG)           # the rod reaches out past the head

    def test_bearing_convention(self):
        xy = np.zeros((4, 2))
        heading = np.array([0.0, 0.0, 0.0, np.pi / 2])
        b = S.bearing_deg(xy, heading, (1.0, 0.0))
        self.assertAlmostEqual(b[0], 0.0)                                  # ahead
        b = S.bearing_deg(xy[:1], heading[:1], (0.0, 1.0))
        self.assertAlmostEqual(b[0], 90.0)                                 # +y is the fly's left when it faces +x
        self.assertAlmostEqual(abs(S.bearing_deg(xy[:1], [0.0], (-1.0, 0.0))[0]), 180.0)
        self.assertAlmostEqual(S.bearing_deg(xy[3:], heading[3:], (0.0, 1.0))[0], 0.0)

    def test_segment_occlusion(self):
        o = (0.0, 0.0, 1.0)
        c, r = (0.0, 0.0, 0.5), (0.1, 0.1, 0.01)
        hits = S.segment_hits_ellipsoid(o, [(0, 0, 0), (0.5, 0, 0), (0, 0, 0.7)], c, r)
        np.testing.assert_array_equal(hits, [True, False, False])          # through, beside, in front of it


class AimTests(unittest.TestCase):
    def test_aim_covers_the_cluster_and_stays_inside_the_rim(self):
        rng = np.random.default_rng(3)
        cluster = rng.normal((0.06, 0.03), 0.01, (20, 2))
        stray = np.array([[-0.12, -0.12], [0.13, -0.13]])
        xy = np.r_[cluster, stray]
        aim, n = S.aim_point(xy, np.ones(len(xy), bool), "-x")
        self.assertGreaterEqual(n, 20)
        self.assertTrue(S.footprint(cluster, aim, "-x").all())
        lx, ly = S.aim_limits("-x")
        self.assertLessEqual(abs(aim[0]), lx + 1e-9)
        self.assertLessEqual(abs(aim[1]), ly + 1e-9)
        rx, _, _ = S.head_radii("-x")
        self.assertLessEqual(lx + rx, S.TRAY_HALF - S.RIM_W)               # the head lands inside the rim

    def test_aim_ignores_ineligible_flies_and_breaks_ties_to_the_centre(self):
        xy = np.array([[0.05, 0.0], [-0.05, 0.0]])
        aim, n = S.aim_point(xy, np.array([True, False]), "-x")
        self.assertEqual(n, 1)
        self.assertTrue(S.footprint(xy[:1], aim, "-x")[0])
        self.assertLess(np.hypot(*aim), 0.01)                              # the centre already covers it
        self.assertEqual(S.aim_point(xy, np.array([False, False]), "-x"), ((0.0, 0.0), 0))


class RuleTests(unittest.TestCase):
    def test_outcome_rule(self):
        #            inside0 inside_c splat  caught launched  expected
        cases = [(True, True, True, False, False, "splat"),
                 (True, True, True, False, True, "splat"),                  # hopped, but landed back under it
                 (True, True, False, True, True, "caught"),                 # hopped; the head met it in the air
                 (True, False, False, True, True, "caught"),
                 (False, False, False, True, True, "caught"),               # hopped into the head's path from outside
                 (True, False, False, False, True, "clear"),                # hopped out, never touched
                 (True, False, False, False, False, "clear"),               # walked out
                 (False, True, False, False, False, "clear"),               # airborne from before the swat, not met
                 (False, False, False, False, True, "startled"),
                 (False, False, False, False, False, "safe")]
        a = np.array([c[:5] for c in cases]).T
        out = S.classify(*a)
        np.testing.assert_array_equal(out, [c[5] for c in cases])
        jumped = S.jumped_in_time(out, a[4])
        np.testing.assert_array_equal(jumped, [False, True, True, True, True, True, False, False, False, False])

    def test_path_meets_ellipsoid(self):
        r = (0.10, 0.085, 0.006)
        c = np.array([0.0, 0.0, 0.76])
        still = [c, c]
        pts0 = np.array([[0.0, 0.0, 0.80], [0.0, 0.0, 0.80], [0.2, 0.0, 0.76], [0.0, 0.0, 0.7661]])
        pts1 = np.array([[0.0, 0.0, 0.70], [0.0, 0.0, 0.79], [0.3, 0.0, 0.76], [0.0, 0.0, 0.7661]])
        hit = S.path_meets_ellipsoid(pts0, pts1, *still, r)
        np.testing.assert_array_equal(hit, [True, False, False, False])     # through it; above; beside; just above
        # a fly hovering still, a fast head sweeping past it within one step: caught although neither end is inside
        fly = np.array([[0.0, 0.0, 0.76]])
        self.assertTrue(S.path_meets_ellipsoid(fly, fly, c + (0, 0, 0.02), c - (0, 0, 0.02), r)[0])
        self.assertFalse(S.path_meets_ellipsoid(fly, fly, c + (0.3, 0, 0.02), c + (0.3, 0, -0.02), r)[0])
        # the margin: a point 1 mm above the top is met only with a margin larger than 1 mm
        top = np.array([[0.0, 0.0, 0.76 + 0.006 + 0.001]])
        self.assertFalse(S.path_meets_ellipsoid(top, top, *still, r)[0])
        self.assertTrue(S.path_meets_ellipsoid(top, top, *still, np.array(r) + S.FLY_R)[0])

    def test_sweep_catches_only_flies_in_the_air(self):
        """In the last step before contact the head's underside passes within the margin of the tray: a walking fly
        under its centre is left for the footprint rule (SPLAT), the same point in the air is caught."""
        class Log:
            def __init__(self):
                self.events = []

            def event(self, t, kind, **kw):
                self.events.append((t, kind, kw))
        g = S.Swarm.__new__(S.Swarm)
        g.z_s, g.B, g.focus, g.fx, g.log = 0.754, 2, -1, [], Log()
        g.dead, g.death_xy = np.zeros(2, bool), np.zeros((2, 2))
        g.death_t, g.death_kind = np.full(2, np.nan), np.full(2, "", dtype=object)
        sw = S.Swat(0.0, 0.5, "-x", aim=(0.0, 0.0))
        sw.rec = dict(alive0=np.ones(2, bool), caught_t=np.full(2, np.nan), caught_phase=np.full(2, "", dtype=object),
                      caught_z=np.full(2, np.nan), launch_t=np.full(2, np.nan), gf_peak=np.zeros(2))
        t = sw.tc - 0.013                                                  # the step [t, t + 10 ms] ends 3 ms before contact
        self.assertLess(S.Swat.height(sw, t + 0.01), S.FLY_R)
        pos = np.array([[0.0, 0.0, g.z_s], [0.0, 0.0, g.z_s]])
        g._sweep(sw, t, t + 0.01, pos, pos, np.array([False, True]))
        np.testing.assert_array_equal(g.dead, [False, True])
        self.assertEqual([k for _, k, _ in g.log.events], ["caught"])
        self.assertEqual(g.death_kind[1], "caught")

    def test_gf_colour_ramp(self):
        c = S.gf_color([0, 10, 22, 33, 50, 80, 200]).astype(int)
        self.assertEqual(c.shape, (7, 3))
        np.testing.assert_array_equal(c[3], (232, 179, 104))               # the threshold is amber
        np.testing.assert_array_equal(c[-1], c[-2])                        # clamps above the top stop
        self.assertLess(c[0].sum(), c[2].sum())


class SwatTests(unittest.TestCase):
    def test_kinematics(self):
        sw = S.Swat(t0=2.0, v=1.0, handle="-x", aim=(0.01, 0.02))
        z_s = 0.754
        self.assertAlmostEqual(sw.tc, 2.0 + S.H0 / np.sin(np.deg2rad(S.APPROACH_DEG)))
        self.assertIsNone(sw.head_centre(1.99, z_s))
        start = sw.head_centre(2.0, z_s)
        self.assertAlmostEqual(start[2] - S.HEAD_THICK - z_s, S.H0)        # starts H0 above the tray
        self.assertLess(start[0], 0.01)                                    # comes in from the handle's side (-x)
        rest = sw.head_centre(sw.tc + 0.01, z_s)
        np.testing.assert_allclose(rest, (0.01, 0.02, z_s + S.HEAD_THICK))  # the head's underside on the tray
        self.assertAlmostEqual(sw.height(sw.tc + 0.05), 0.0)
        self.assertGreater(sw.height(sw.tc + S.DWELL_S + 0.1), 0.0)        # lifting
        self.assertIsNone(sw.head_centre(sw.t_end + 1e-6, z_s))
        mid = sw.head_centre(sw.t0 + 0.5 * (sw.tc - sw.t0), z_s)
        self.assertAlmostEqual(np.linalg.norm(mid - rest), 0.5 * sw.path_len, places=9)

    def test_round_plan(self):
        plan = S.round_plan()
        self.assertEqual(len(plan), len(S.ROUNDS))
        self.assertIsNone(plan[0][1])                                      # the run starts fresh
        self.assertAlmostEqual(plan[0][2], S.FIRST_SWAT_S)
        for k, reset, t0, v, h in plan[1:]:
            self.assertAlmostEqual(reset, k * S.ROUND_S)
            self.assertAlmostEqual(t0, reset + S.SETTLE_S)
        for (k, _, t0, v, h), nxt in zip(plan, plan[1:]):
            self.assertLess(S.Swat(t0, v, h).t_end, nxt[1])                # each swat is over before the next reset
        self.assertLess(S.Swat(plan[-1][2], plan[-1][3], plan[-1][4]).t_end + 0.8, S.CLIP_SECONDS)
        later = S.round_plan(start=3.0)
        self.assertAlmostEqual(later[0][1], 3.0)                           # started mid-run: round 1 resets too

    def test_round_label_counts_from_the_schedule(self):
        g = S.Swarm.__new__(S.Swarm)
        g.round, g.round_base, g.n_rounds = 0, 0, 5                         # the recorded schedule
        self.assertEqual(g._round_label(), "ROUND 1 / 5")
        self.assertEqual(g._round_label(4), "ROUND 5 / 5")
        g.round, g.round_base = 2, 3                                        # 'A' pressed in round 3 of free play
        self.assertEqual(g._round_label(), "ROUND 3")                       # until the schedule's first reset
        self.assertEqual(g._round_label(3), "ROUND 1 / 5")
        self.assertEqual(g._round_label(7), "ROUND 5 / 5")
        g.n_rounds = 0
        self.assertEqual(g._round_label(8), "ROUND 9")

    def test_crowd_layout(self):
        xy, h = S.crowd_layout(64, 0.024, 0.004, np.random.default_rng([100, 0]))
        self.assertEqual(xy.shape, (64, 2))
        self.assertTrue((np.abs(xy) <= S.FENCE_HALF).all())
        xy2, h2 = S.crowd_layout(64, 0.024, 0.004, np.random.default_rng([100, 0]))
        np.testing.assert_array_equal(xy, xy2)
        np.testing.assert_array_equal(h, h2)


class CameraTests(unittest.TestCase):
    def test_projection_round_trip_and_handedness(self):
        cam = S.orbit_camera(3.0, 24.0, 320, 240, (0.0, 0.0, 0.754))
        p = np.array([[0.03, -0.02, 0.754]])
        px, depth = cam.project(p)
        self.assertGreater(depth[0], 0)
        back = cam.unproject_to_plane(px[0, 0], px[0, 1], 0.754)
        np.testing.assert_allclose(back, p[0], atol=1e-9)
        c0, _ = cam.project(cam.pos + cam.f)
        right, _ = cam.project(cam.pos + cam.f + 0.1 * cam.r)
        up, _ = cam.project(cam.pos + cam.f + 0.1 * cam.u)
        self.assertGreater(right[0, 0], c0[0, 0])                          # scene right is pixel right (not mirrored)
        self.assertLess(up[0, 1], c0[0, 1])

    def test_fit_camera_holds_the_points_tightly(self):
        W, H, box = 1344, 967, (40, 30, 1300, 800)
        az, el = S.orbit_pose(5.0, 24.0)
        sw = S.Swat(5.0, 1.0, "+y")
        pts = S.swat_frame_points(sw, 0.754)
        cam = S.fit_camera(az, el, W, H, pts, box)
        p, z = cam.project(pts)
        self.assertTrue((z > 0).all())
        self.assertTrue((p[:, 0] >= box[0] - 1e-6).all() and (p[:, 0] <= box[2] + 1e-6).all())
        self.assertTrue((p[:, 1] >= box[1] - 1e-6).all() and (p[:, 1] <= box[3] + 1e-6).all())
        # tight: some point touches a vertical or a horizontal pair of box edges
        touch_x = np.isclose(p[:, 0].min(), box[0], atol=0.5) and np.isclose(p[:, 0].max(), box[2], atol=0.5)
        touch_y = np.isclose(p[:, 1].min(), box[1], atol=0.5) and np.isclose(p[:, 1].max(), box[3], atol=0.5)
        self.assertTrue(touch_x or touch_y)
        np.testing.assert_allclose(cam.f, S.orbit_camera(5.0, 24.0, W, H, (0, 0, 0.754)).f, atol=1e-12)

    def test_swat_camera_keeps_the_head_in_view(self):
        """The main view pulls back before each swat and holds the whole head in frame from 0.1 s after it begins
        (earlier for the fast swats), at every speed, for aims anywhere the rule can put it."""
        g = S.Swarm.__new__(S.Swarm)                                       # no brain: only the camera's state
        g.z_s, g.seconds, g.shake_until, g.queue = 0.754, 24.0, -1.0, []
        W, H = 1344, 967
        rect = type("R", (), {"w": W, "h": H})()
        for k, _, t0, v, h in S.round_plan():
            for aim in ((0.0, 0.0), S.aim_limits(h), (-S.aim_limits(h)[0], S.aim_limits(h)[1])):
                sw = S.Swat(t0, v, h, aim=tuple(aim))
                g.swats, g.queue = [], [sw]
                g.t_s = t0 - S.CAM_PREROLL_S - 0.1
                far = g._camera(rect).pos
                np.testing.assert_allclose(far, S.orbit_camera(g.t_s, 24.0, W, H, (0, 0, g.z_s)).pos)
                sw.rec = {"started": True}
                g.swats, g.queue = [sw], []
                for t in np.arange(t0 + min(0.1, 0.2 * (sw.tc - t0)), sw.tc, 0.01):
                    g.t_s = t
                    cam = g._camera(rect)
                    c = sw.head_centre(t, g.z_s)
                    rx, ry, _ = S.head_radii(h)
                    a = np.linspace(0, 2 * np.pi, 32)
                    p, z = cam.project(np.c_[c[0] + rx * np.cos(a), c[1] + ry * np.sin(a), np.full(32, c[2])])
                    inside = (z > 0) & (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
                    self.assertTrue(inside.all(), (v, h, aim, round(t - t0, 3)))

    def test_rays_match_projection(self):
        import torch
        cam = S.Camera((0.4, -0.4, 1.1), (0.0, 0.0, 0.75), 64, 48, 37.0)
        o, d = cam.rays("cpu")
        d = d.reshape(48, 64, 3).numpy()
        for (j, i) in ((0, 0), (20, 50), (47, 63)):
            px, _ = cam.project(cam.pos + d[j, i])
            np.testing.assert_allclose(px[0], (i + 0.5, j + 0.5), atol=1e-4)
        self.assertEqual(tuple(o.shape), (48 * 64, 3))
        self.assertTrue(torch.allclose(o[0], torch.tensor(cam.pos, dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
