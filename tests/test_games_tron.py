"""games/tron.py: headings and turns, jet-wall boxes, the swept collision test, the cycles' moves and crashes, the turn
decoder's law (upward crossing, re-arm, side), the random and bot players, and the ray tracer's conventions -- on
CPU, with no dataset and no brain."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "games"))
import tron  # noqa: E402


class HeadingTests(unittest.TestCase):
    def test_turns_are_90_degree_steps(self):
        self.assertEqual(tron.turn_heading(0, "L"), 1)          # +x -> +y (left, z up)
        self.assertEqual(tron.turn_heading(0, "R"), 3)          # +x -> -y
        self.assertEqual(tron.turn_heading(3, "L"), 0)
        for h in range(4):
            f, lft = tron.heading_vec(h), tron.left_vec(h)
            self.assertAlmostEqual(float(np.cross(np.r_[f, 0], np.r_[lft, 0])[2]), 1.0)   # left is +90 deg
            np.testing.assert_allclose(tron.heading_vec(tron.turn_heading(h, "L")), lft)

    def test_segment_box_is_thickened_all_round(self):
        lo, hi = tron.segment_box((0, 0), (10, 0))
        np.testing.assert_allclose(lo, [-tron.TRAIL_HALF_W, -tron.TRAIL_HALF_W, 0])
        np.testing.assert_allclose(hi, [10 + tron.TRAIL_HALF_W, tron.TRAIL_HALF_W, tron.TRAIL_H])

    def test_cycle_box_follows_heading(self):
        lo, hi = tron.cycle_box((0, 0), 1)                       # heading +y: long along y
        self.assertAlmostEqual(hi[1] - lo[1], tron.CYCLE_LEN)
        self.assertAlmostEqual(hi[0] - lo[0], tron.CYCLE_W)


class CollisionTests(unittest.TestCase):
    def test_sweep_hits_the_first_box(self):
        boxes = [tron.segment_box((5, -3), (5, 3)), tron.segment_box((3, -3), (3, 3))]
        s, k = tron.sweep_hit((0, 0), (10, 0), boxes, 0.0)
        self.assertEqual(k, 1)
        self.assertAlmostEqual(s, (3 - tron.TRAIL_HALF_W) / 10)
        self.assertEqual(tron.sweep_hit((0, 5), (10, 5), boxes, 0.0), (None, None))

    def test_a_thin_wall_cannot_be_tunnelled(self):
        box = [tron.segment_box((0.5, -3), (0.5, 3))]
        s, _ = tron.sweep_hit((0, 0), (1.0, 0), box, 0.0)      # the wall lies wholly inside one step
        self.assertIsNotNone(s)

    def test_move_stops_at_the_arena_wall(self):
        c = tron.Cycle("A", tron.CYAN, (tron.ARENA_HALF - tron.FRONT - 0.05, 0), 0)
        hit = tron.move_cycle(c, [], 1.0)
        self.assertEqual(hit, "arena")
        self.assertFalse(c.alive)
        self.assertAlmostEqual(c.nose[0], tron.ARENA_HALF, places=6)

    def test_move_hits_the_other_trail_and_own_old_trail(self):
        a = tron.Cycle("A", tron.CYAN, (0, 0), 0)
        b = tron.Cycle("B", tron.ORANGE, (4, -10), 1)
        b.pos = np.array([4.0, 10.0])                            # B's live wall now runs along x = 4
        self.assertEqual(tron.move_cycle(a, [b], 5.0), "opponent")
        # own finished segment: ride +x, turn L twice (a U-turn 1 m over), then ride back into... nothing (parallel);
        # then turn L again and cross the first segment
        c = tron.Cycle("C", tron.CYAN, (0, 0), 0)
        c.pos = np.array([20.0, 0])
        c.turn("L")
        c.pos = np.array([20.0, 6.0])
        c.turn("L")
        c.pos = np.array([10.0, 6.0])
        c.turn("L")                                              # heading -y, toward the first segment at y = 0
        self.assertIsNone(tron.move_cycle(c, [], 1.0))
        self.assertEqual(tron.move_cycle(c, [], 10.0), "own trail")

    def test_own_live_segment_is_not_an_obstacle(self):
        c = tron.Cycle("A", tron.CYAN, (0, 0), 0)
        for _ in range(50):
            self.assertIsNone(tron.move_cycle(c, [], 0.15))
        self.assertTrue(c.alive)
        self.assertAlmostEqual(c.pos[0], 7.5)

    def test_free_run_and_bot(self):
        c = tron.Cycle("A", tron.CYAN, (0, 0), 0)
        self.assertAlmostEqual(tron.free_run(c, [], 0), tron.ARENA_HALF - tron.FRONT)
        self.assertAlmostEqual(tron.free_run(c, [], 1), tron.ARENA_HALF - tron.FRONT)
        self.assertIsNone(tron.bot_choice(c, [], 10.0))
        near = tron.Cycle("A", tron.CYAN, (tron.ARENA_HALF - 5, tron.ARENA_HALF - 8), 0)   # wall ahead, room to the right
        self.assertEqual(tron.bot_choice(near, [], 10.0), "R")


class LawTests(unittest.TestCase):
    def test_turn_on_upward_crossing_only(self):
        law = tron.TurnLaw(33.0, rearm_s=0.3, window_s=0.1)
        out = [law.update(i * 0.01, g, -1.0) for i, g in enumerate([10, 20, 40, 50, 60, 20, 10])]
        self.assertEqual(out, [None, None, "L", None, None, None, None])

    def test_rearm_interval(self):
        law2 = tron.TurnLaw(33.0, rearm_s=0.3, window_s=0.1)
        seq = [(0.00, 10), (0.01, 40), (0.02, 10), (0.03, 40), (0.40, 10), (0.41, 40)]
        res = [law2.update(t, g, 1.0) for t, g in seq]
        self.assertEqual(res, ["R" if i in (1, 5) else None for i in range(6)])

    def test_first_tick_crossing_counts(self):
        law = tron.TurnLaw(33.0)
        self.assertEqual(law.update(0.0, 40, 2.0), "R")

    def test_side_is_away_from_the_louder_eye(self):
        law = tron.TurnLaw(33.0, rearm_s=0.0, window_s=0.05)
        for i, s in enumerate([-5, -5, -5, -5, 3]):
            law.update(i * 0.01, 0.0, s)
        self.assertAlmostEqual(law.side_signal(), np.mean([-5, -5, -5, -5, 3]))   # the last 50 ms: 5 ticks
        self.assertEqual(law.update(0.05, 50.0, -1.0), "L")     # mean s < 0: the right eye looms more -> left
        law.update(0.06, 0.0, 9.0)
        for i in range(7, 12):
            law.update(i * 0.01, 0.0, 9.0)
        self.assertEqual(law.update(0.12, 50.0, 9.0), "R")

    def test_side_offset_is_subtracted(self):
        law = tron.TurnLaw(33.0, rearm_s=0.0, window_s=0.05, side_offset=-2.0)
        for i in range(5):
            law.update(i * 0.01, 0.0, -1.5)                     # less negative than the resting -2: left looms
        self.assertAlmostEqual(law.side_signal(), 0.5)
        self.assertEqual(law.update(0.05, 40.0, -1.5), "R")
        self.assertEqual(tron.SIDE_OFFSET_HZ["malecns"], -2.1)

    def test_not_live_never_fires(self):
        law = tron.TurnLaw(33.0)
        self.assertIsNone(law.update(0.0, 60.0, 1.0, live=False))
        self.assertIsNone(law.update(0.01, 70.0, 1.0, live=True))       # no new crossing: still above threshold

    def test_random_rate(self):
        r = tron.RandomTurns(2.0, np.random.default_rng(0), rearm_s=0.0)
        n = sum(r.update(i * 0.01) is not None for i in range(100000))
        self.assertAlmostEqual(n / 1000.0, 2.0, delta=0.15)

    def test_turn_inputs(self):
        m = {"DNp01_L": 30.0, "DNp01_R": 40.0, "LPLC2_L": 2.0, "LPLC2_R": 1.0, "LC4_L": 3.0, "LC4_R": 1.0}
        gf, side, L, R = tron.turn_inputs(m)
        self.assertEqual((gf, side, L, R), (35.0, 3.0, 5.0, 2.0))

    def test_population_means(self):
        x = {"a": torch.tensor([[1.0, 3.0]]), "b": torch.zeros(1, 0)}
        self.assertEqual(tron.population_means(x), {"a": 2.0, "b": 0.0})


class GameLoopTests(unittest.TestCase):
    def test_brain_free_arms_play_rounds(self):
        import argparse
        args = argparse.Namespace(seed=100, device="cpu", record=None, control=False, player="random", opponent="bot",
                                  random_rate=0.8, cam_scale=0.2, style=tron.STYLE, seconds=10.0)
        g = tron.Tron(args)
        self.assertEqual(g.brains(), {})
        for _ in range(1500):                                   # 15 s of game time
            g.tick()
            g.t_s = round(g.t_s, 6)
            for c in g.cycles:
                self.assertLessEqual(abs(c.nose[0]), tron.ARENA_HALF + 1e-6)
                self.assertLessEqual(abs(c.nose[1]), tron.ARENA_HALF + 1e-6)
        kinds = [e["kind"] for e in g.log.events]
        self.assertIn("round_start", kinds)
        self.assertIn("crash", kinds)
        self.assertGreaterEqual(g.round, 1)
        s = g.log.summary
        self.assertEqual(s["rounds"], len(g.rounds))
        self.assertEqual(sum(s["score"].values()), s["rounds"])
        self.assertEqual({d["name"] for d in g.log.meta["declared"]} >= {"arena", "schedule", "eye_mount"}, True)

    def test_never_player_rides_into_the_first_wall(self):
        import argparse
        args = argparse.Namespace(seed=100, device="cpu", record=None, control=False, player="never", opponent="bot",
                                  random_rate=0.5, cam_scale=0.2, style=tron.STYLE, seconds=10.0)
        g = tron.Tron(args)
        for _ in range(900):
            g.tick()
        crash = [e for e in g.log.events if e["kind"] == "crash" and e["player"] == "A"][0]
        a = tron.SCHEDULE[0][0]
        run = tron.ARENA_HALF - tron.FRONT - a[0]                # straight along +x from the first start
        self.assertAlmostEqual(crash["survived_s"], run / tron.SPEED, delta=0.02)
        self.assertEqual(crash["into"], "arena")
        self.assertEqual(crash["turns"], 0)


def _brainfree_args(**kw):
    import argparse
    a = dict(seed=100, device="cpu", record=None, control=False, player="random", opponent="bot", random_rate=0.8,
             cam_scale=0.2, style=tron.STYLE, seconds=10.0, baseline_draws=0)
    a.update(kw)
    return argparse.Namespace(**a)


class CrossingLogTests(unittest.TestCase):
    def test_every_crossing_has_a_status(self):
        law = tron.TurnLaw(33.0, rearm_s=0.3)
        law.update(0.00, 40.0, 0.0, live=False)
        self.assertEqual(law.last_cross["status"], "not_live")
        law.update(0.01, 10.0, 0.0)
        self.assertIsNone(law.last_cross)
        self.assertIsNotNone(law.update(0.02, 40.0, 0.0))
        self.assertEqual(law.last_cross["status"], "turn")
        law.update(0.03, 10.0, 0.0)
        self.assertIsNone(law.update(0.04, 40.0, 0.0))
        self.assertEqual(law.last_cross["status"], "blocked_rearm")
        self.assertAlmostEqual(law.last_cross["since_last_turn_s"], 0.02)


class BaselineTests(unittest.TestCase):
    def test_replay_reproduces_a_run_and_draws_rank_it(self):
        g = tron.Tron(_brainfree_args())
        for _ in range(1500):
            g.tick()
        g._summary()
        s = g.log.summary
        self.assertEqual(s["ticks"], 1500)
        self.assertGreater(s["turns_A"], 0)
        rep = tron.play_game_only(100, 1500, tron.ReplayPilot(tron.turn_schedule(g.turn_log["A"])),
                                  tron.ReplayPilot(tron.turn_schedule(g.turn_log["B"])))
        self.assertEqual(rep.log.summary["round_records"], s["round_records"])
        self.assertEqual(tron.outcome_of(rep.log.summary), tron.outcome_of(s))
        b = tron.run_baselines(100, 1500, g.turn_log, s, draws=4)
        self.assertTrue(b["replay_matches_fly"])
        r = b["rank_of_fly"]
        self.assertEqual(r["draws"], 4)
        self.assertEqual(r["draws_fewer_crashes_per_min"] + r["draws_equal"] + r["draws_more_crashes_per_min"], 4)
        self.assertAlmostEqual(b["random"]["rate_hz"], s["turns_A"] / s["riding_s_A"], places=3)
        self.assertEqual(b["never"]["turns_A"], 0)

    def test_crash_rate_metric(self):
        s = {"crashes_A": 2, "riding_s_A": 30.0, "crashes_per_min_A": 4.0, "rounds": 3, "turns_A": 5, "crashes_B": 1,
             "score": {"A": 1, "B": 2, "draw": 0}}
        o = tron.outcome_of(s)
        self.assertEqual((o["won_A"], o["won_B"], o["crashes_per_min_A"]), (1, 2, 4.0))


class CameraTests(unittest.TestCase):
    def test_cached_grid_matches_the_pinhole_formula(self):
        W, H, fov = 16, 9, 72.0
        f = np.array([0.3, 0.9, -0.2]); up = np.array([0, 0, 1.0])
        d, pix = tron.camera_rays(np.zeros(3), f, up, W, H, fov, "cpu")
        f = f / np.linalg.norm(f); left = np.cross(up, f); left /= np.linalg.norm(left); u = np.cross(f, left)
        tan = math.tan(math.radians(fov) / 2)
        X, Y = np.meshgrid((2 * (np.arange(W) + 0.5) / W - 1) * tan, (1 - 2 * (np.arange(H) + 0.5) / H) * tan * H / W)
        ref = f[None, None] - X[..., None] * left[None, None] + Y[..., None] * u[None, None]
        ref = (ref / np.linalg.norm(ref, axis=-1, keepdims=True)).reshape(-1, 3)
        np.testing.assert_allclose(d.numpy(), ref, atol=1e-5)
        self.assertAlmostEqual(pix, 2 * tan / W)

    def test_camera_stays_inside_the_arena_at_every_round_start(self):
        g = tron.Tron(_brainfree_args(player="never"))
        for k in range(len(tron.SCHEDULE)):
            g.round = k - 1
            g._new_round(g.t_s)
            cam, fwd = g.camera_pose()
            self.assertLessEqual(max(abs(cam[0]), abs(cam[1])), tron.ARENA_HALF - 1.0 + 1e-9)
            a = g.cycles[0]
            to_cycle = np.array([a.pos[0], a.pos[1], 0.6]) - cam
            cosang = to_cycle @ fwd / np.linalg.norm(to_cycle) / np.linalg.norm(fwd)
            self.assertGreater(cosang, math.cos(math.radians(30)))   # the cycle is well inside the 72 deg view


class LegendTests(unittest.TestCase):
    def test_footer_states_the_offset_and_the_arm(self):
        g = tron.Tron(_brainfree_args(player="random", random_rate=0.42))
        laws = g.footer_laws()
        self.assertIn("cyan: Poisson turns 0.42/s", laws[-1])
        self.assertIn("orange: GAME bot", laws[-1])
        self.assertFalse(any("side" in x for x in laws))               # no fly in this arm: no decoder laws
        fly = tron.FlyPilot.__new__(tron.FlyPilot)
        fly.law = tron.TurnLaw(side_offset=-2.1)
        g.pa = fly
        laws = g.footer_laws()
        self.assertIn("> -2.1 Hz, else L", laws[1])
        for control in (False, True):                                    # fits the 1920 px footer in a wide mono font
            g.args.control = control
            for kind, pilot in (("fly", fly), ("random", tron.RandomPilot(0.31, 0)), ("never", tron.NeverPilot())):
                g.player_kind, g.pa = kind, pilot
                self.assertLessEqual(len("   ".join(g.footer_laws())), 180)

    def test_banners_carry_a_chip_tab(self):
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        pygame.init()
        import common
        g = tron.Tron(_brainfree_args())
        g.hud = common.Hud()
        plain = tron.Banner(0.0, "GO", chip=None)
        tabbed = tron.Banner(0.0, "GIANT FIBRE 40 Hz  →  TURN LEFT", "MaleCNS · 0.9 m to the wall ahead", chip="decoder")
        self.assertGreater(g.banner_surface(tabbed).get_height(), g.banner_surface(plain).get_height() + 30)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.r = tron.Renderer("cpu", tron.STYLE)

    def test_floor_sky_and_box(self):
        boxes = tron.Renderer.pack([(np.array([10.0, -5, 0]), np.array([10.2, 5, 2.2]), tron.KIND_TRAIL, tron.ORANGE)]
                                   + [(lo, hi, tron.KIND_ARENA, tron.WALL_RGB) for lo, hi in tron.arena_boxes(500.0)])
        d = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0.6, 0, -0.8]])
        rgb = self.r.trace((0, 0, 1.0), d, boxes, math.radians(1.5))
        self.assertEqual(tuple(rgb.shape), (3, 3))
        self.assertTrue(torch.isfinite(rgb).all())
        sky = self.r._sky(d[1:2])[0]
        torch.testing.assert_close(rgb[1], sky)
        self.assertGreater(float(rgb[0, 0]), float(rgb[0, 2]))  # the orange wall ahead: red over blue

    def test_camera_is_not_mirrored(self):
        # a bright wall on the camera's left must land in the left half of the image
        boxes = tron.Renderer.pack([(np.array([5.0, 0.5, 0]), np.array([5.2, 6, 3]), tron.KIND_TRAIL, tron.CYAN)]
                                   + [(lo, hi, tron.KIND_ARENA, tron.WALL_RGB) for lo, hi in tron.arena_boxes(500.0)])
        img = tron.render_camera(tron.Renderer("cpu", "glow"), boxes, np.array([0.0, 0, 1.5]), (1, 0, 0), (0, 0, 1), 64, 36, 90)
        g = img[..., 1].mean(0)
        self.assertGreater(float(g[:32].mean()), float(g[32:].mean()))

    def test_bloom_tonemap_range(self):
        img = torch.rand(20, 30, 3) * 3
        u8 = tron.bloom_tonemap(img)
        self.assertEqual(u8.shape, (20, 30, 3))
        self.assertEqual(u8.dtype, np.uint8)

    def test_scene_hides_the_riders_own_body(self):
        a, b = tron.Cycle("A", tron.CYAN, (0, 0), 0), tron.Cycle("B", tron.ORANGE, (10, 10), 2)
        full = tron.scene_boxes([a, b])
        hidden = tron.scene_boxes([a, b], hide=a)
        self.assertEqual(len(full) - len(hidden), 1)


if __name__ == "__main__":
    unittest.main()
