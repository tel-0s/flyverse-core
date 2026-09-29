"""games/minecraft.py: the pure pieces -- the vectorised voxel ray caster against a reference 3D-DDA, the frame and
yaw conventions shared with mineflayer, the walk decoder's law, Auto-Jump, the bridge's region encoding and the mob
boxes -- on CPU with no dataset, no GPU, no server, no Node and no pygame."""
import base64
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from games import minecraft as mc  # noqa: E402


def flat_table(n_rows=4):
    """A texture table whose row r is a flat colour (r / n_rows, 0.5, 0.25) on every face: no atlas needed."""
    t = torch.zeros((n_rows, 3, 16, 16, 3))
    for r in range(1, n_rows):
        t[r, ..., 0] = r / n_rows
        t[r, ..., 1] = 0.5
        t[r, ..., 2] = 0.25
    return SimpleNamespace(tensor=t)


def reference_dda(grid, o, d, max_t=80.0):
    """Amanatides & Woo: the first opaque voxel along the ray from o (grid coords) in direction d -> (t, axis)."""
    ix = [int(math.floor(o[i])) for i in range(3)]
    step = [1 if d[i] > 0 else -1 for i in range(3)]
    t_max, t_delta = [], []
    for i in range(3):
        if abs(d[i]) < 1e-12:
            t_max.append(math.inf)
            t_delta.append(math.inf)
        else:
            nxt = ix[i] + (1 if d[i] > 0 else 0)
            t_max.append((nxt - o[i]) / d[i])
            t_delta.append(abs(1.0 / d[i]))
    while True:
        a = int(np.argmin(t_max))
        t = t_max[a]
        if t >= max_t:
            return math.inf, -1
        ix[a] += step[a]
        t_max[a] += t_delta[a]
        if not all(0 <= ix[i] < grid.shape[i] for i in range(3)):
            return math.inf, -1
        if grid[ix[0], ix[1], ix[2]] > 0:
            return t, a


class TestCaster(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        g = np.zeros((20, 14, 18), np.int64)
        g[:, :4, :] = 1                                    # ground
        g[rng.random(g.shape) < 0.04] = 2                  # scattered blocks
        g[12, 4:9, 3:10] = 3                               # a wall
        g[9:11, 5:7, 8:10] = 0                             # a hole
        self.grid = g
        self.caster = mc.VoxelCaster(flat_table(), mc.Sky(), "cpu", graphs=False)
        self.caster.set_region((100, 60, -40), g)

    def test_matches_reference_dda(self):
        rng = np.random.default_rng(11)
        o = np.array([7.3, 6.6, 8.45])
        g = self.grid.copy()
        g[int(o[0]), int(o[1]), int(o[2])] = 0             # the eye is in air
        self.caster.set_region((100, 60, -40), g)
        d = rng.normal(size=(400, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        _, info = self.caster.cast(o + [100, 60, -40], d)
        t = info["t"].numpy()
        for i in range(len(d)):
            t_ref, _ = reference_dda(g, o, d[i])
            if math.isinf(t_ref):
                self.assertTrue(math.isinf(t[i]), f"ray {i}: caster hit at {t[i]}, reference none")
            else:
                self.assertAlmostEqual(float(t[i]), t_ref, places=4, msg=f"ray {i}")

    def test_ground_sky_and_entity(self):
        o = np.array([100 + 5.5, 60 + 4 + 1.62, -40 + 5.5])   # standing on the ground (top at y index 4)
        down = np.array([[0.0, -1.0, 0.0]])
        up = np.array([[0.0, 1.0, 0.0]])
        rad, info = self.caster.cast(o, down)
        self.assertAlmostEqual(float(info["t"][0]), 1.62, places=4)
        self.assertEqual(rad.shape, (1, 4))
        # UV is half the blue channel (the declared RGB -> radiance convention)
        self.assertAlmostEqual(float(rad[0, 0]), 0.5 * float(rad[0, 1]), places=6)
        g = self.grid.copy()
        g[:, 7:, :] = 0
        self.caster.set_region((100, 60, -40), g)
        _, info = self.caster.cast(o, up)
        self.assertFalse(bool(info["hit"][0]))
        # an entity box 2 m ahead is nearer than anything behind it
        fwd = np.array([[1.0, 0.0, 0.0]])
        ent = [{"name": "zombie", "x": o[0] + 2.3, "y": o[1] - 1.62, "z": o[2], "width": 0.6, "height": 1.95}]
        _, info = self.caster.cast(o, fwd, ent)
        self.assertTrue(bool(info["entity"][0]))
        self.assertAlmostEqual(float(info["t"][0]), 2.0, places=4)

    def test_entity_arrays_pad(self):
        lo, hi, cols, cuts = mc.entity_arrays([{"name": "zombie", "x": 1, "y": 2, "z": 3}], origin=(1, 2, 3), k_max=4)
        self.assertEqual(lo.shape, (4, 3))
        np.testing.assert_allclose(lo[0], [-0.3, 0, -0.3], atol=1e-6)
        np.testing.assert_allclose(hi[0], [0.3, 1.8, 0.3], atol=1e-6)
        self.assertTrue((lo[1:] > 1e5).all())               # empty slots sit far away
        np.testing.assert_allclose(cuts[0], [0.38, 0.74], atol=1e-6)


class TestConventions(unittest.TestCase):
    def test_yaw_heading_round_trip(self):
        for a in (0.0, 45.0, 90.0, 157.5, 200.0, 300.0):
            yaw = mc.mc_heading_to_yaw(a)
            np.testing.assert_allclose(mc.yaw_forward(yaw), [math.cos(math.radians(a)), 0, math.sin(math.radians(a))],
                                       atol=1e-9)
            # the fly's body forward (z-up frame) is the same direction in Minecraft axes
            f, left, up = mc.body_axes(mc.heading_from_mineflayer_yaw(yaw))
            np.testing.assert_allclose(mc.zup_to_mc(np.array(f)), mc.yaw_forward(yaw), atol=1e-6)
            # left is counter-clockwise from forward seen from above (+y up in Minecraft)
            cross = np.cross(mc.zup_to_mc(np.array(f)), mc.zup_to_mc(np.array(left)))
            np.testing.assert_allclose(cross, [0, 1, 0], atol=1e-6)
        self.assertAlmostEqual(mc.mineflayer_yaw_from_heading(mc.heading_from_mineflayer_yaw(0.7)), 0.7)
        # mineflayer yaw 0 faces north (-z); notchian yaw 180 is north
        np.testing.assert_allclose(mc.yaw_forward(0.0), [0, 0, -1], atol=1e-12)
        self.assertAlmostEqual(mc.notch_yaw_deg(0.0), 180.0)

    def test_frames_round_trip(self):
        v = np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]])
        np.testing.assert_allclose(mc.mc_to_zup(mc.zup_to_mc(v)), v, atol=1e-6)

    def test_sun(self):
        # time 0: just risen in the east (vanilla's eased celestial angle puts it ~12 deg up); 6000: the zenith
        x, y, z = mc.Sky.sun_direction(0)
        self.assertGreater(x, 0.95)
        self.assertTrue(0.0 < y < 0.25)
        self.assertAlmostEqual(z, 0.0)
        np.testing.assert_allclose(mc.Sky.sun_direction(6000), [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(mc.Sky.sun_direction(12000), [-mc.Sky.sun_direction(0)[0], mc.Sky.sun_direction(0)[1], 0],
                                   atol=1e-6)                                            # symmetric sunset in the west
        x, y, _ = mc.Sky.sun_direction(mc.TIME_OF_DAY)
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)


class TestWalkAndJump(unittest.TestCase):
    def test_walk_decoder_law(self):
        c = mc.player_command(0.02)                        # a brisk fly walk -> Minecraft's walk
        self.assertAlmostEqual(c["v_player"], 4.317, places=6)
        self.assertAlmostEqual(c["speed"], 0.1, places=6)  # the vanilla movement_speed attribute
        self.assertTrue(c["forward"] and not c["back"])
        c = mc.player_command(-0.005)
        self.assertTrue(c["back"] and not c["forward"])
        self.assertGreater(c["speed"], 0)
        c = mc.player_command(0.0001)                      # inside the deadband
        self.assertFalse(c["forward"] or c["back"])

    def test_autojump_and_ground(self):
        g = np.zeros((10, 8, 10), np.int64)
        g[:, :3, :] = 1                                    # ground top at y index 2 -> feet at 3
        origin = np.array([0, 60, 0])
        feet = np.array([4.5, 63.0, 4.5])
        fwd = np.array([1.0, 0.0, 0.0])
        self.assertFalse(mc.autojump_wanted(g, origin, feet, fwd, True))
        g[5, 3, :] = 1                                     # a 1-block step ahead
        self.assertTrue(mc.autojump_wanted(g, origin, feet, fwd, True))
        self.assertFalse(mc.autojump_wanted(g, origin, feet, fwd, False))   # only while walking
        g[5, 4, :] = 1                                     # now a 2-block wall
        self.assertFalse(mc.autojump_wanted(g, origin, feet, fwd, True))
        self.assertEqual(mc.ground_y(g, origin, 2.5, 2.5, 63.0), 63.0)
        self.assertEqual(mc.ground_y(g, origin, 5.5, 2.5, 63.0), 65.0)
        self.assertIsNone(mc.ground_y(g, origin, 50.0, 2.5, 63.0))


class TestBridgeEncoding(unittest.TestCase):
    def test_decode_region_order(self):
        dx, dy, dz = 3, 4, 5
        grid = np.arange(dx * dy * dz, dtype=np.uint16).reshape(dx, dy, dz)
        flat = np.zeros(dx * dy * dz, np.uint16)
        for y in range(dy):                                # bridge.js: index ((y * dz) + z) * dx + x
            for z in range(dz):
                for x in range(dx):
                    flat[(y * dz + z) * dx + x] = grid[x, y, z]
        reg = {"origin": [7, 8, 9], "dims": [dx, dy, dz], "palette": ["air"],
               "data": base64.b64encode(flat.tobytes()).decode()}
        origin, idx, pal = mc.decode_region(reg)
        np.testing.assert_array_equal(idx, grid)
        self.assertEqual(origin.tolist(), [7, 8, 9])
        self.assertEqual(pal, ["air"])


class TestMarker(unittest.TestCase):
    def test_fly_view_of(self):
        yaw = mc.mc_heading_to_yaw(157.5)
        psi = mc.heading_from_mineflayer_yaw(yaw)
        fwd = mc.yaw_forward(yaw)
        eye = np.array([3.0, 65.62, -7.0])
        az, el, r, d = mc.fly_view_of(eye + 2 * fwd, eye, psi, 0.5)
        self.assertAlmostEqual(az, 0.0, places=4)
        self.assertAlmostEqual(el, 0.0, places=4)
        self.assertAlmostEqual(d, 2.0, places=5)
        self.assertAlmostEqual(r, math.degrees(math.asin(0.25)), places=4)
        left = np.cross([0.0, 1.0, 0.0], fwd)             # Minecraft axes: up x forward = the player's left
        self.assertAlmostEqual(mc.fly_view_of(eye + 3 * left, eye, psi, 0.1)[0], 90.0, places=4)
        self.assertAlmostEqual(mc.fly_view_of(eye + [0, 2, 0], eye, psi, 0.1)[1], 90.0, places=4)

    def test_mosaic_points_follow_the_eye_layout(self):
        # a toy two-eye lattice: left eye azimuth -20..120, right eye -120..20, elevation -60..60
        az = np.concatenate([np.linspace(-20, 120, 50), np.linspace(-120, 20, 50)])
        el = np.tile(np.linspace(-60, 60, 50), 2)
        side = np.array(["L"] * 50 + ["R"] * 50)
        rect = (0, 0, 600, 300)
        left = mc.mosaic_points(np.c_[az, el], side, rect, 90.0, 0.0)
        right = mc.mosaic_points(np.c_[az, el], side, rect, -90.0, 0.0)
        ahead = mc.mosaic_points(np.c_[az, el], side, rect, 0.0, 0.0)
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertEqual(len(ahead), 2)                    # the binocular strip is in both eyes
        self.assertLess(left[0][0], 300)                   # the left eye is drawn on the left
        self.assertGreater(right[0][0], 300)
        self.assertAlmostEqual(left[0][1], 150.0)          # elevation 0 is the vertical centre


def jump_heights(n_frames, frames_per_tick=5):
    """Eye height gain of a vanilla jump, sampled every brain frame (0.42 blocks/tick up, then -0.08 and x0.98 drag
    per tick), linearly interpolated between Minecraft ticks as the game does."""
    y, vy, ticks = 0.0, 0.42, [0.0]
    for _ in range(n_frames // frames_per_tick + 2):
        y += vy
        vy = (vy - 0.08) * 0.98
        ticks.append(max(y, 0.0))
    return [ticks[f // frames_per_tick] + (ticks[f // frames_per_tick + 1] - ticks[f // frames_per_tick])
            * ((f % frames_per_tick) + 1) / frames_per_tick for f in range(n_frames)]


class TestDive(unittest.TestCase):
    def run_dive(self, jump_at_turn, heading=(-0.924, 0.383), height=11.0, dist=22.0, gap=0.10):
        """A frontal dive at 10 m/s (0.1 m per 10 ms frame) at a player walking towards it at 1.9 m/s; optionally the
        player jumps 20 ms after the turn (the GF latency)."""
        h = np.array([heading[0], 0.0, heading[1]])
        eye0 = np.array([0.0, 1.62, 0.0])
        c = eye0 + h * dist + np.array([0.0, height + 0.25, 0.0])
        mode, pass_dir, step = "dive", None, 0.1
        gaps, turned_at, eye = [], None, eye0
        jumps = jump_heights(400)
        for k in range(400):
            eye = eye0 + h * 0.019 * k
            if turned_at is not None and jump_at_turn and k >= turned_at + 2:
                eye = eye + np.array([0.0, jumps[k - turned_at - 2], 0.0])
            c, mode, pass_dir, direction, turned = mc.dive_step(c, eye, mode, pass_dir, step, gap)
            if turned:
                turned_at = k
                self.assertAlmostEqual(mc.box_gap(c, mc.PHANTOM_HALF, eye), gap, places=6)
            gaps.append(mc.box_gap(c, mc.PHANTOM_HALF, eye))
        return np.array(gaps), turned_at, c, eye

    def test_dive_turns_at_the_gap_and_the_eye_never_gets_closer(self):
        for jump in (False, True):
            for heading in ((-0.924, 0.383), (0.707, 0.707), (1.0, 0.0)):
                gaps, turned_at, c, eye = self.run_dive(jump, heading)
                self.assertIsNotNone(turned_at)
                self.assertLess(turned_at, 260)            # ~24.6 m at 0.1 m per frame: no hover at the head
                self.assertGreaterEqual(gaps.min(), 0.10 - 1e-6, f"jump={jump} heading={heading}")
                self.assertGreater(c[1] - eye[1], 3.0)     # gone up and over by the end

    def test_pass_moves_along_the_pass_direction_when_clear(self):
        c = np.array([5.0, 8.0, 0.0])
        eye = np.array([0.0, 1.62, 0.0])
        pass_dir = np.array([0.78, 0.62, 0.0])
        c2, mode, _, d2, turned = mc.dive_step(c, eye, "pass", pass_dir, 0.1, 0.1)
        self.assertFalse(turned)
        self.assertEqual(mode, "pass")
        np.testing.assert_allclose(c2 - c, pass_dir * 0.1)

    def test_keep_clear_and_box_gap(self):
        half = (0.45, 0.25, 0.45)
        self.assertEqual(mc.box_gap((0, 0, 0), half, (0.1, 0.1, 0.1)), 0.0)
        self.assertTrue(mc.inside_box((0, 0, 0), half, (0.1, 0.1, 0.1)))
        self.assertAlmostEqual(mc.box_gap((0, 0, 0), half, (0.0, 1.25, 0.0)), 1.0)
        c = mc.keep_clear(np.array([0.0, 0.3, 0.0]), half, np.array([0.1, 0.0, 0.0]), 0.1)   # above an eye inside
        self.assertAlmostEqual(c[1], 0.35)
        self.assertAlmostEqual(mc.box_gap(c, half, (0.1, 0.0, 0.0)), 0.1)
        c = np.array([2.0, 0.0, 0.0])
        np.testing.assert_allclose(mc.keep_clear(c, half, np.zeros(3), 0.1), c)                   # already clear


class TestCamera(unittest.TestCase):
    def test_orbit_is_rate_limited_and_takes_the_short_way(self):
        phi = mc.orbit_step(0.1, 0.1 + np.pi - 0.01, 0.03, 1.0)
        self.assertAlmostEqual(phi, 0.13)
        phi = mc.orbit_step(3.1, -3.1, 1.0, 1.0)             # across the +-pi seam: 0.08 rad the short way
        self.assertAlmostEqual((phi - 3.1 + np.pi) % (2 * np.pi) - np.pi, 2 * np.pi - 6.2, places=6)


class TestEncounters(unittest.TestCase):
    def game(self, encs, control="none"):
        g = SimpleNamespace(encounters=encs, args=SimpleNamespace(control=control))
        g.scripted_ids = lambda: mc.MinecraftGame.scripted_ids(g)
        return g

    def test_key_summoned_mobs_get_the_full_timeout(self):
        encs = [mc.Encounter(1.5, 0.0, 12.0, n=1), mc.Encounter(7.5, 35.0, 10.0, n=2),
                mc.Encounter(0.3, 0.0, 12.0, n=3, scripted=False)]
        g = self.game(encs)
        self.assertAlmostEqual(mc.MinecraftGame.timeout_for(g, encs[0]), 7.5 - 1.5 - 0.4)
        self.assertAlmostEqual(mc.MinecraftGame.timeout_for(g, encs[2]), mc.ENCOUNTER_TIMEOUT_S)

    def test_eye_entities(self):
        eye = np.array([0.0, 65.62, 0.0])
        ents = [{"id": 1, "name": "zombie", "type": "hostile", "x": 3.0, "y": 64.0, "z": 0.0, "width": 0.6, "height": 1.95},
                {"id": 2, "name": "sheep", "type": "animal", "x": 6.0, "y": 64.0, "z": 0.0, "width": 0.9, "height": 1.3},
                {"id": 3, "name": "falling_block", "type": "other", "x": 2.0, "y": 64.0, "z": 0.0},
                {"id": 4, "name": "zombie", "type": "hostile", "x": 0.1, "y": 64.0, "z": 0.1, "width": 0.6, "height": 1.95},
                {"id": -5, "name": "phantom", "type": "mob", "x": 1.0, "y": 66.0, "z": 0.0, "width": 0.9, "height": 0.5}]
        enc = mc.Encounter(1.5, 0.0, 12.0, n=1)
        enc.entity_id = 1
        g = self.game([enc, mc.Encounter(19.0, 0.0, 22.0, mob="phantom", n=5)])
        out, inside = mc.MinecraftGame.eye_entities(g, ents, eye)
        self.assertEqual([e["id"] for e in out], [-5, 1, 2])  # nearest first; 'other' dropped
        self.assertEqual([e["id"] for e in inside], [4])      # the box around the eye is culled
        g = self.game(g.encounters, control="blind-to-mobs")
        out, _ = mc.MinecraftGame.eye_entities(g, ents, eye)
        self.assertEqual([e["id"] for e in out], [2])         # the control hides only the scripted mobs


class TestScript(unittest.TestCase):
    def test_script_is_fixed_and_ordered(self):
        times = [row[0] for row in mc.SCRIPT]
        self.assertEqual(times, sorted(times))
        self.assertTrue(all(row[1] in ("zombie", "phantom") for row in mc.SCRIPT))
        self.assertLess(times[-1], 30.0)


if __name__ == "__main__":
    unittest.main()
