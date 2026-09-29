"""games/mars.py: the decoder laws (wind steer, hazard stop), the GAME arbitration between the wind steer and the
autopilot, the dust devils' wind, the rover's speed schedule, kinematics and contact geometry, the hazard tally, the
fixed boulder layout, the texture sampler, the track map, the camera convention and the ray tracer (terrain, boulders,
rover, dust devils) -- on CPU, with no dataset and no brain."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import mars  # noqa: E402

DT = 0.01


class LawTests(unittest.TestCase):
    def test_stop_threshold_is_the_body_models(self):
        from flyverse.body import Flight
        self.assertEqual(mars.GF_HZ, Flight().gf_hz)
        law = mars.StopLaw()
        self.assertFalse(law(10, {"gf": torch.tensor([32.9, 32.9])})["fire"])
        self.assertTrue(law(10, {"gf": torch.tensor([30.0, 36.0])})["fire"])     # the mean of both cells

    def test_wind_index_and_turn_into_the_wind(self):
        self.assertAlmostEqual(mars.wind_index(50.0, 8.0, 3.0, 40.0), 0.5 * (42.0 + 37.0))    # wind from the left
        law = mars.WindSteerLaw(tau_s=0.0)
        left = law.from_rates(50.0, 8.0, 3.0, 40.0, DT)
        self.assertTrue(left["active"])
        self.assertGreater(left["yaw_rate"], 0)                  # + = a left turn: into a wind from the left
        u = 0.5 * (42.0 + 37.0)
        self.assertAlmostEqual(left["yaw_rate"], min(mars.WIND_GAIN * (u - mars.WIND_DEAD), mars.WIND_MAX))
        right = mars.WindSteerLaw(tau_s=0.0).from_rates(8.0, 50.0, 40.0, 3.0, DT)
        self.assertLess(right["yaw_rate"], 0)

    def test_wind_dead_band_limit_and_lowpass(self):
        law = mars.WindSteerLaw(tau_s=0.0)
        quiet = law.from_rates(10.0 + 1.8 * mars.WIND_DEAD * 0.9, 10.0, 10.0, 10.0, DT)   # u = 0.9 x dead band
        self.assertEqual(quiet["yaw_rate"], 0.0)
        self.assertFalse(quiet["active"])
        huge = mars.WindSteerLaw(tau_s=0.0).from_rates(500.0, 0.0, 0.0, 500.0, DT)
        self.assertAlmostEqual(huge["yaw_rate"], mars.WIND_MAX)
        lp = mars.WindSteerLaw(tau_s=0.2)
        lp.from_rates(0.0, 0.0, 0.0, 0.0, DT)
        s = lp.from_rates(20.0, 0.0, 0.0, 0.0, DT)["u_s"]
        self.assertAlmostEqual(s, 10.0 * (1 - math.exp(-DT / 0.2)), places=6)

    def test_wind_decoder_reads_its_four_cells(self):
        law = mars.WindSteerLaw(tau_s=0.0)
        x = {"DNp18_L": torch.tensor([30.0]), "DNp18_R": torch.tensor([10.0]), "DNp33_L": torch.tensor([0.0]),
             "DNp33_R": torch.tensor([20.0])}
        out = law(10, x)
        self.assertAlmostEqual(out["u"], 0.5 * (20.0 + 20.0))
        self.assertEqual(set(mars.WIND_CELLS), set(x))


class WindTests(unittest.TestCase):
    def devil(self, **kw):
        d = {"name": "T", "xy0": (0.0, 0.0), "vel": (0.0, 0.0), "w0": 4.0, "flare": 0.2, "height": 100.0,
             "opacity": 0.5, "vt": 20.0, "rc": 7.0, "decay_m": 20.0, "spin": -1, "inflow_deg": 35.0}
        d.update(kw)
        spec = dict(d)
        spec["pos"] = lambda t: d["xy0"]
        return [spec]

    def test_vortex_profile(self):
        dv = self.devil(inflow_deg=0.0)
        w_core = mars.devil_wind((7.0, 0.0), dv, 0.0)
        self.assertAlmostEqual(float(np.hypot(*w_core)), 20.0, places=6)          # peak at the core radius
        self.assertAlmostEqual(float(np.hypot(*mars.devil_wind((3.5, 0.0), dv, 0.0))), 10.0, places=6)
        far = float(np.hypot(*mars.devil_wind((27.0, 0.0), dv, 0.0)))
        self.assertAlmostEqual(far, 20.0 * 7.0 / 27.0 * math.exp(-1.0), places=6)
        # clockwise seen from above: east of the centre the wind blows south (-y)
        self.assertLess(w_core[1], 0)
        self.assertEqual(float(np.hypot(*mars.devil_wind((0.0, 0.0), dv, 0.0))), 0.0)

    def test_inflow_points_at_the_devil(self):
        dv = self.devil()
        w = mars.devil_wind((20.0, 0.0), dv, 0.0)
        self.assertLess(w[0], 0)                                  # the radial part blows towards the centre
        self.assertAlmostEqual(-w[0] / abs(w[1]), math.tan(35.0 * mars.DEG), places=6)

    def test_rover_passing_a_clockwise_devil_on_its_left_feels_it_from_the_right(self):
        dv = self.devil(xy0=(0.0, 21.0))
        w = mars.devil_wind((0.0, 0.0), dv, 0.0)                  # abeam, heading +x
        a = mars.wind_from_deg(w, 0.0)
        self.assertLess(a, -20.0)
        self.assertGreater(a, -60.0)                              # from ahead-right (about -35 deg)

    def test_the_game_devil_meets_the_rover_from_the_left(self):
        d3 = [d for d in mars.devil_specs() if d["name"] == "D3"][0]
        self.assertLess(d3["xy0"][1], 0)                          # right of the route
        self.assertEqual(d3["spin"], 1)                           # counter-clockwise seen from above
        cx, cy = d3["pos"](0.0)
        for dx in (-30.0, -15.0, 0.0):                            # approaching and abeam: wind from the left
            a = mars.wind_from_deg(mars.devil_wind((cx + dx, 0.0), [d3], 0.0), 0.0)
            self.assertGreater(a, 20.0)
            self.assertLess(a, 110.0)

    def test_many_points_match_the_single_point_wind(self):
        devils = mars.devil_specs()
        pts = np.array([[10.0, 2.0], [80.0, -3.0], [74.0, -16.0 + 3.0], [300.0, 100.0]])
        many = mars.devil_wind_many(pts, devils, 12.5)
        for q, w in zip(pts, many):
            np.testing.assert_allclose(w, mars.devil_wind(q, devils, 12.5), rtol=1e-9, atol=1e-12)

    def test_antenna_deflections_are_the_shipped_models(self):
        from flyverse.air import Air, WindParams
        for w_xy, yaw in (((3.0, -1.0), 0.3), ((-2.0, 0.5), -1.2), ((0.0, 0.0), 0.0)):
            dL, dR = mars.antenna_deflections(w_xy, yaw)
            we = np.asarray(w_xy) * mars.WIND_EQUIV
            air = Air([], WindParams(speed=float(np.hypot(*we)), direction_deg=math.degrees(math.atan2(we[1], we[0])),
                                     meander_deg=0.0))
            fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            eL, eR = air.deflections(fwd, left, mars.JO_FULL_SPEED)
            self.assertAlmostEqual(dL, float(eL[0]), places=9)
            self.assertAlmostEqual(dR, float(eR[0]), places=9)
        # a wind from the left deflects the left antenna back (> 0) and the right one forward (< 0)
        dL, dR = mars.antenna_deflections((0.0, -5.0), 0.0)
        self.assertGreater(dL, 0)
        self.assertLess(dR, 0)
        self.assertAlmostEqual(mars.WIND_EQUIV, math.sqrt(0.020 / 1.225))


class ArbitrationTests(unittest.TestCase):
    def test_the_autopilot_keeps_the_last_word(self):
        # the decoder's cap is below the autopilot's limit, so their sum can always turn the rover back
        self.assertLess(mars.WIND_MAX, mars.AP_MAX)
        worst = mars.autopilot_yaw_rate(0.0, 50.0, 0.0) + mars.WIND_MAX         # far left, decoder turning left
        self.assertLess(worst, 0)

    def test_held_while_recovering_from_a_collision(self):
        self.assertEqual(mars.steer_applied(0.2, False), (0.2, False))
        self.assertEqual(mars.steer_applied(-0.2, True), (0.0, True))
        self.assertEqual(mars.steer_applied(0.0, True), (0.0, False))            # nothing commanded, nothing held


class RoverTests(unittest.TestCase):
    def test_autopilot_steers_back_to_the_line(self):
        self.assertLess(mars.autopilot_yaw_rate(0.0, 3.0, 0.0), 0)     # left of the line -> turn right
        self.assertGreater(mars.autopilot_yaw_rate(0.0, -3.0, 0.0), 0)
        self.assertEqual(mars.autopilot_yaw_rate(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(abs(mars.autopilot_yaw_rate(0.0, 50.0, 0.0)), mars.AP_MAX)

    def test_wrap(self):
        self.assertAlmostEqual(mars.wrap(3 * math.pi / 2), -math.pi / 2)
        self.assertAlmostEqual(mars.wrap(-3 * math.pi / 2), math.pi / 2)

    def test_kinematics(self):
        x, y, yaw = mars.rover_step(0.0, 0.0, 0.0, 5.0, 0.0, 1.0)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 0.0)
        x, y, yaw = mars.rover_step(0.0, 0.0, 0.0, 1.0, math.pi / 2, 1.0)
        self.assertAlmostEqual(yaw, math.pi / 2)

    def test_speed_schedule(self):
        d = mars.DriveState()
        t = 0.0
        while t < mars.WARM_S - 1e-9:
            t += DT
            self.assertEqual(d.update(DT, t - DT), 0.0)
        for _ in range(1000):
            t += DT
            d.update(DT, t)
        self.assertEqual(d.state, "drive")
        self.assertAlmostEqual(d.v, mars.V_CRUISE)
        self.assertTrue(d.hazard_stop())
        self.assertFalse(d.hazard_stop())                  # only from 'drive'
        n = 0
        while d.state == "stop":
            d.update(DT, t)
            n += 1
        self.assertAlmostEqual(n * DT, mars.V_CRUISE / mars.A_BRAKE, delta=2 * DT)
        self.assertEqual(d.state, "reverse")
        self.assertEqual(d.update(DT, t), -mars.V_REVERSE)
        for _ in range(int(mars.REVERSE_S / DT) + 2):
            d.update(DT, t)
        self.assertEqual(d.state, "drive")
        d.collide()
        self.assertEqual((d.state, d.v), ("detour", 0.0))
        d.v = 1.7
        self.assertEqual(d.update(DT, t), 1.7)             # a Detour sets the speed, the schedule keeps it

    def test_detour_goes_round_in_one_contact(self):
        """The GAME detour from a grid of approaches (offsets and headings): one contact, then round the rock and
        past it, as the game's tick does it (autopilot on the route line, contact = footprint overlaps the rock)."""
        rock = {"name": "H", "x": 150.0, "y": 0.8, "r": 1.45}
        for y0 in (-2.0, -1.0, 0.0, 0.8, 1.6, 2.6):
            for h0 in (-12.0, 0.0, 12.0):
                x, y, yaw, v = 136.0, y0, h0 * mars.DEG, mars.V_CRUISE
                detour, contacts = None, 0
                for _ in range(4000):
                    gap = mars.rect_circle_clearance(x, y, yaw, mars.ROVER_L / 2, mars.ROVER_W / 2, rock["x"],
                                                     rock["y"], rock["r"])
                    if detour is not None:
                        v, w, done = detour.step(DT, x, y, yaw, v, gap)
                        if done:
                            detour = None
                    if detour is None:
                        v, w = min(mars.V_CRUISE, v + mars.A_DRIVE * DT), mars.autopilot_yaw_rate(x, y, yaw)
                    nx, ny, nyaw = mars.rover_step(x, y, yaw, v, w, DT)
                    if mars.rect_circle_clearance(nx, ny, nyaw, mars.ROVER_L / 2, mars.ROVER_W / 2, rock["x"],
                                                  rock["y"], rock["r"]) < 0:
                        if v < 0:
                            detour.t = mars.BACK_MAX_S
                        else:
                            contacts += 1
                            v, detour = 0.0, mars.Detour(rock, x, y, yaw)
                        continue
                    x, y, yaw = nx, ny, nyaw
                    if x > 158:
                        break
                self.assertEqual(contacts, 1, (y0, h0))
                self.assertGreater(x, 158, (y0, h0))

    def test_rectangle_circle_clearance(self):
        c = mars.rect_circle_clearance
        self.assertAlmostEqual(c(0, 0, 0, 1.5, 1.0, 3.0, 0.0, 0.5), 1.0)     # ahead: 3 - 1.5 - 0.5
        self.assertAlmostEqual(c(0, 0, 0, 1.5, 1.0, 0.0, 2.0, 0.5), 0.5)     # beside
        self.assertLess(c(0, 0, 0, 1.5, 1.0, 1.8, 0.0, 0.5), 0)              # overlap
        self.assertAlmostEqual(c(0, 0, math.pi / 2, 1.5, 1.0, 3.0, 0.0, 0.5), 1.5)   # rotated: its side faces +x
        self.assertAlmostEqual(c(0, 0, 0, 1.5, 1.0, 1.5 + 3, 1.0 + 4, 0.0), 5.0)    # corner


class TallyTests(unittest.TestCase):
    def rocks(self):
        rng = np.random.default_rng(0)
        return [mars.make_rock(rng, (20.0, 0.0, 0.0), 1.0, hazard=True, name="H1"),
                mars.make_rock(rng, (40.0, 0.0, 0.0), 1.0, hazard=True, name="H2"),
                mars.make_rock(rng, (30.0, 9.0, 0.0), 1.0)]

    def test_clean_pass_and_collision(self):
        tally = mars.HazardTally(self.rocks())
        self.assertEqual(len(tally.rows), 2)
        for x in np.arange(0, 30, 0.1):
            tally.observe(x / 5, x, 4.0, 0.0)                              # drives by 4 m to the side
        h1 = tally.rows[0]
        self.assertEqual(h1["outcome"], "clean")
        self.assertGreater(h1["min_clear_m"], 0)
        tally.rows[1]["contacts"] = 1
        for x in np.arange(30, 50, 0.1):
            tally.observe(x / 5, x, 4.0, 0.0)
        s = tally.summary()
        self.assertEqual((s["passed"], s["passed_clean"], s["passed_after_collision"]), (2, 1, 1))
        self.assertEqual(s["not_reached"], [])

    def test_stop_credit_window(self):
        tally = mars.HazardTally(self.rocks(), stop_window_m=12.0)
        self.assertIsNone(tally.credit_stop(0.0))                          # H1's face is ~19 m ahead
        h = tally.credit_stop(10.0)
        self.assertEqual(h["name"], "H1")
        self.assertEqual(tally.summary()["hazard_stops_credited"], 1)


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = mars.MarsScene("cpu", flat=True)

    def test_texture_sampler(self):
        data = torch.arange(16, dtype=torch.float32).reshape(4, 4)           # rows = y, columns = x
        tex = mars.Tex2D(data, 0.0, 4.0)
        v = tex(torch.tensor([0.5, 2.5, 1.0]), torch.tensor([0.5, 1.5, 0.5]))
        np.testing.assert_allclose(v.numpy(), [0.0, 6.0, 0.5], atol=1e-5)
        wrap = mars.Tex2D(data, 0.0, 4.0, wrap=True)
        np.testing.assert_allclose(wrap(torch.tensor([4.5]), torch.tensor([8.5])).numpy(), [0.0], atol=1e-5)

    def test_layout_is_fixed_and_the_lane_is_clear(self):
        a = mars.build_rocks(self.scene)
        b = mars.build_rocks(self.scene)
        self.assertEqual([r.pos.tolist() for r in a], [r.pos.tolist() for r in b])
        hz = [r for r in a if r.hazard]
        self.assertEqual(len(hz), len(mars.HAZARDS))
        for r in a:
            if not r.hazard and -5 < r.pos[0] < mars.ROUTE_END + 10:
                self.assertGreaterEqual(abs(r.pos[1]), mars.LANE_HALF)
        xs = sorted(r.pos[0] for r in hz)                 # an open stretch for the dust devil between H2 and H3
        self.assertGreater(xs[2] - xs[1], 100.0)
        d3 = [d for d in mars.devil_specs() if d["name"] == "D3"][0]
        self.assertTrue(xs[1] < d3["xy0"][0] < xs[2])
        self.assertLess(xs[-1] + max(r.r_ground for r in hz) + mars.ROVER_L / 2, mars.ROUTE_END)
        for r in hz:                                 # every hazard blocks the rover's path on the route line
            self.assertLess(abs(r.pos[1]) - r.r_ground, mars.ROVER_W / 2)

    def test_camera_convention(self):
        d = mars.camera_rays((1, 0, 0), (0, 0, 1), 3, 3, 90, "cpu").reshape(3, 3, 3)
        np.testing.assert_allclose(d[1, 1].numpy(), [1, 0, 0], atol=1e-6)
        self.assertGreater(float(d[1, 0, 1]), 0)          # column 0 looks left (+y): not mirrored
        self.assertGreater(float(d[0, 1, 2]), 0)          # row 0 looks up

    def test_trace_ground_rock_sky(self):
        sc = self.scene
        rng = np.random.default_rng(1)
        sc.set_rocks([mars.make_rock(rng, (10.0, 0.0, 0.0), 1.0)])
        o = torch.tensor([0.0, 0.0, 0.5])
        d = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.0, -0.8]])
        d = d / d.norm(dim=1, keepdim=True)
        rgb, depth = sc.trace(o, d)
        self.assertTrue(torch.isfinite(rgb).all() and (rgb >= 0).all())
        self.assertLess(float(depth[0]), 10.0)             # the boulder ahead
        self.assertTrue(math.isinf(float(depth[1])))        # the sky
        self.assertAlmostEqual(float(depth[2]), 0.5 / 0.8, delta=0.08)     # the ground (flat, +- detail relief)
        sc.set_rocks([])

    def test_dust_devil_layer(self):
        sc = self.scene
        dv = [d for d in mars.devil_specs() if d["name"] == "D3"]
        sc.devils = dv
        cx, cy = dv[0]["pos"](0.0)
        o = torch.tensor([cx, cy + 40.0, 2.0])
        d = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.5]])
        d = d / d.norm(dim=1, keepdim=True)
        a, col = sc.devil_layer(o, d, torch.full((3,), float("inf")), 0.0)
        self.assertGreater(float(a[0]), 0.3)                  # looking through the column's foot
        self.assertLess(float(a[1]), 1e-3)                    # looking away
        self.assertTrue(bool(((a >= 0) & (a <= 1)).all()) and bool(torch.isfinite(col).all()))
        a2, _ = sc.devil_layer(o, d[:1], torch.full((1,), 10.0), 0.0)
        self.assertEqual(float(a2[0]), 0.0)                   # a surface in front of the devil hides it
        sc.devils = []

    def test_tracks(self):
        tr = mars.Tracks("cpu", lo=(0.0, 0.0), size=(10.0, 10.0), res=0.05)
        tr.stamp([(5.0, 5.0)])
        v = tr.sample(torch.tensor([5.0, 5.1, 7.0]), torch.tensor([5.0, 5.0, 7.0]))
        self.assertGreater(float(v[0]), 0.9)
        self.assertGreater(float(v[1]), 0.5)                  # within the 13 cm wheel patch
        self.assertEqual(float(v[2]), 0.0)
        tr.stamp([(-50.0, 500.0)])                            # off the map: clamped, no error

    def test_rover_is_hit_and_casts_a_shadow(self):
        rv = mars.Rover("cpu")
        rv.set_pose(np.zeros(3), mars.rot_zyx(0.0, 0.0, 0.0), 0.0)
        o = torch.tensor([10.0, 0.0, 1.0])
        d = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        for dense in (False, True):
            t, n, col, mask = rv.hit(o, d, torch.full((2,), float("inf")), dense=dense)
            self.assertTrue(bool(mask[0]))
            self.assertFalse(bool(mask[1]))
            self.assertAlmostEqual(float(t[0]), 10.0 - 1.30, delta=0.02)          # the body's front face
            np.testing.assert_allclose(n[0].numpy(), [1, 0, 0], atol=1e-5)
        sun = torch.tensor([0.0, 0.0, 1.0])
        lit = rv.shadow(torch.tensor([[0.0, 0.0, 0.01], [8.0, 0.0, 0.01]]), sun)
        np.testing.assert_allclose(lit.numpy(), [0.0, 1.0])

    def test_eye_sits_on_a_post_above_the_head(self):
        boxes, caps, wheels = mars.rover_parts()
        head = boxes[2]
        head_top = head[0][2] + head[1][2]
        self.assertEqual(mars.EYE_MOUNT, "mast_post")
        self.assertAlmostEqual(mars.EYE_LOCAL[2], head_top + mars.POST_H, places=6)
        self.assertEqual(len(wheels), 6)
        rv = mars.Rover("cpu")
        rv.set_pose(np.zeros(3), np.eye(3), 0.0)
        # the eye is outside every part: a ray from it straight up hits nothing, straight down hits the post first
        o = torch.tensor(mars.EYE_LOCAL, dtype=torch.float32)
        d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        t, n, col, mask = rv.hit(o, d, torch.full((3,), float("inf")), dense=True)
        self.assertEqual(mask.tolist(), [False, True, False])
        self.assertAlmostEqual(float(t[1]), 0.03 - 0.012, delta=1e-3)          # the post's rounded top


if __name__ == "__main__":
    unittest.main()
