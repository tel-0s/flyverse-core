"""games/doom.py: the pure pieces -- the trigger decoder's law, the brain-to-Doom clock, the Doom session's timeline and
scoring (on a scripted stand-in for ViZDoom), the baseline triggers, the screen geometry, the frame conversion, the
no-muzzle-light PWAD and the Freedoom sprite decoding -- on CPU with no dataset, no GPU, no ViZDoom and no pygame."""
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from games import doom  # noqa: E402
from games import common as gc  # noqa: E402


class ClockTests(unittest.TestCase):
    def test_35_tics_per_brain_second_in_10_ms_frames(self):
        clock = doom.TicClock()
        due = [clock.advance(10.0) for _ in range(100)]
        self.assertEqual(sum(due), 35)
        self.assertEqual(due[:3], [0, 0, 1])                 # the first tic lands at 30 ms (one every 28.57 ms)
        self.assertTrue(set(due) <= {0, 1})

    def test_long_run_has_no_drift(self):
        clock = doom.TicClock()
        self.assertEqual(sum(clock.advance(10.0) for _ in range(3000)), 35 * 30)


class TriggerTests(unittest.TestCase):
    def test_fires_on_any_frame_at_threshold_within_a_tic(self):
        trig = doom.GFTrigger(33.0)
        for v in (5.0, 33.0, 12.0):
            trig.observe(v)
        self.assertEqual(trig.fire_for_tic(), (True, 33.0))

    def test_below_threshold_never_fires_and_peak_resets(self):
        trig = doom.GFTrigger(33.0)
        trig.observe(32.9)
        self.assertFalse(trig.fire_for_tic()[0])
        self.assertEqual(trig.fire_for_tic(), (False, 0.0))
        trig.observe(50.0)
        trig.idle()                                            # not live: the peak is forgotten
        self.assertEqual(trig.fire_for_tic(), (False, 0.0))

    def test_threshold_is_the_body_models(self):
        from flyverse.body import Flight
        self.assertEqual(doom.GF_THRESHOLD_HZ, Flight().gf_hz)

    def test_upward_crossings(self):
        self.assertEqual(doom.upward_crossings([0, 40, 50, 10, 33, 0, 34], 33.0), [1, 4, 6])
        self.assertEqual(doom.upward_crossings([35, 36], 33.0), [0])

    def test_decoder_reads_the_mean_rate(self):
        self.assertAlmostEqual(doom._mean_rate(10.0, {"gf": torch.tensor([[20.0, 50.0]])}), 35.0)

    def test_turn_keys_sign(self):
        # ViZDoom's TURN_LEFT_RIGHT_DELTA turns right for positive values: LEFT must send a negative delta
        self.assertEqual(doom.turn_delta(True, False), -doom.TURN_DEG_PER_TIC)
        self.assertEqual(doom.turn_delta(False, True), doom.TURN_DEG_PER_TIC)
        self.assertEqual(doom.turn_delta(True, True), 0.0)
        self.assertEqual(doom.turn_delta(False, False), 0.0)


class BaselineTriggerTests(unittest.TestCase):
    def test_fixed_and_replay(self):
        self.assertEqual(doom.FixedTrigger(True).fire_for_tic(7), (True, 0.0))
        self.assertEqual(doom.FixedTrigger(False).fire_for_tic(7), (False, 0.0))
        rep = doom.ReplayTrigger([3, 4, 9])
        self.assertEqual([rep.fire_for_tic(t)[0] for t in range(1, 11)],
                         [False, False, True, True, False, False, False, False, True, False])

    def test_random_trigger_rate_and_holds(self):
        trig = doom.RandomTrigger(0.05, [3, 5], np.random.default_rng(0))
        seq = [trig.fire_for_tic()[0] for _ in range(20000)]
        runs, n = [], 0
        for f in seq + [False]:
            if f:
                n += 1
            elif n:
                runs.append(n)
                n = 0
        self.assertTrue(set(runs) <= {3, 5})                  # every hold has a drawn length: holds never merge
        free = len(seq) - sum(seq) - len(runs)                 # tics where a hold could start
        self.assertAlmostEqual(len(runs) / free, 0.05, delta=0.01)

    def test_fly_rank(self):
        runs = [{"kills": k, "deaths": d, "triggers": 1, "shots": 1} for k, d in ((1, 3), (5, 2), (6, 1), (2, 2))]
        r = doom.fly_rank({"kills": 5, "deaths": 2}, runs)
        self.assertEqual(r["draws_with_at_least_as_many_kills"], 2)
        self.assertEqual(r["draws_with_more_kills"], 1)
        self.assertEqual(r["draws_with_at_most_as_many_deaths"], 3)
        self.assertEqual(r["draws_with_fewer_deaths"], 1)
        self.assertEqual(r["kills_min_median_max"], [1, 3.5, 6])


class FakeWorld:
    """A scripted stand-in for DoomWorld: `script[tic]` is that tic's result (hits, misses, kills, hurt, dead)."""

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.seed, self.episode, self.tic = 7, 0, 0
        self.steps = []                                        # (tic, fire)
        self.new_episode()

    def new_episode(self):
        self.episode += 1
        self.doom_seed = doom.episode_seed(self.seed, self.episode)
        self.vars = {"HEALTH": 100.0}

    def restart(self):
        self.episode, self.tic = 0, 0
        self.new_episode()

    def step(self, fire, turn=0.0):
        self.tic += 1
        self.steps.append((self.tic, bool(fire)))
        out = {"hits": 0, "misses": 0, "kills": 0, "hurt": 0, "dead": False, "health": 100.0}
        out.update(self.script.get(self.tic, {}))
        return out

    def close_recent(self):
        return False

    def nearest(self, n=3):
        return []

    def closest_recent(self):
        return None


def _run(session, trigger, seconds):
    t, recs = 0.0, []
    for _ in range(round(seconds * 100)):
        t += gc.TICK_MS / 1000.0
        recs.append((t, session.frame(t, trigger)))
    return t, recs


class SessionTests(unittest.TestCase):
    def test_get_ready_then_35_tics_per_second(self):
        w = FakeWorld()
        s = doom.DoomSession(w)
        t, _ = _run(s, doom.FixedTrigger(False), 3.0)
        self.assertEqual(w.tic, 70)                            # nothing in the first WARM_S, then 35 per second
        self.assertEqual(s.summary(t)["episodes"][0]["ended"], "clip end")

    def test_death_freezes_then_new_episode_and_nothing_fires_while_frozen(self):
        w = FakeWorld({10: {"dead": True, "health": 0}})
        s = doom.DoomSession(w)
        t, recs = _run(s, doom.FixedTrigger(True), 3.0)
        # tic 10 is played at brain 1.0 + 10 / 35 s (the frame that contains it); the freeze lasts 1.2 s
        died = next(tt for tt, rr in recs if any(r.get("dead") for r in rr))
        respawn = next(tt for tt, rr in recs if {"marker": "respawn"} in rr)
        resume = next(tt for tt, rr in recs if {"marker": "resume"} in rr)
        self.assertAlmostEqual(respawn - died, doom.DEATH_FREEZE_S, places=6)
        self.assertAlmostEqual(resume - died, doom.DEATH_FREEZE_S + doom.READY_S, places=6)
        self.assertEqual(w.episode, 2)
        self.assertFalse(any(died < tt < resume and rr and "marker" not in rr[0] for tt, rr in recs))
        st = s.summary(t)
        self.assertEqual(st["deaths"], 1)
        self.assertEqual([e["ended"] for e in st["episodes"]], ["death", "clip end"])
        self.assertAlmostEqual(st["episodes"][0]["survived_s"], died - doom.WARM_S, places=3)
        # always-fire: one hold before the death, a new rising edge (trigger) after the resume
        self.assertEqual(st["triggers"], 2)
        self.assertEqual(st["holds"][0], [1, 10])
        self.assertEqual(st["trigger_tics"], sum(n for _, n in st["holds"]))

    def test_gf_above_threshold_across_the_freeze_counts_a_new_trigger(self):
        w = FakeWorld({5: {"dead": True, "health": 0}})
        s = doom.DoomSession(w)
        trig = doom.GFTrigger(33.0)
        t = 0.0
        for _ in range(300):
            t += 0.01
            trig.observe(50.0)                                 # the GF never drops
            s.frame(t, trig)
        self.assertEqual(s.stats["triggers"], 2)

    def test_kill_credit_window(self):
        w = FakeWorld({3: {"hits": 1}, 5: {"kills": 1}, 20: {"kills": 1}})
        s = doom.DoomSession(w)
        t, _ = _run(s, doom.FixedTrigger(False), 2.0)
        self.assertEqual((s.stats["kills"], s.stats["infighting_kills"]), (1, 1))   # tic 5 is within 3 tics of the hit

    def test_control_off_counts_commands_and_never_fires(self):
        w = FakeWorld()
        s = doom.DoomSession(w, connected=False)
        _run(s, doom.ReplayTrigger(range(5, 10)), 2.0)
        self.assertEqual(s.stats["decoder_commands"], 1)
        self.assertEqual((s.stats["triggers"], s.stats["trigger_tics"]), (0, 0))
        self.assertFalse(any(f for _, f in w.steps))

    def test_after_own_shot_window(self):
        w = FakeWorld({4: {"misses": 1}})
        s = doom.DoomSession(w)
        t, recs = _run(s, doom.FixedTrigger(False), 1.5)
        shot_t = next(tt for tt, rr in recs if any(r.get("misses") for r in rr))
        self.assertEqual(s.last_shot_t, shot_t)
        lo, hi = doom.OWN_SHOT_WINDOW_S
        self.assertEqual((lo, hi), (0.1, 0.7))
        self.assertFalse(s.after_own_shot(shot_t + lo - 0.05))
        self.assertTrue(s.after_own_shot(shot_t + 0.5))
        self.assertFalse(s.after_own_shot(shot_t + hi + 0.1))

    def test_replay_of_a_session_reproduces_it(self):
        script = {12: {"hits": 1, "kills": 1}, 30: {"dead": True, "health": 0}}
        w = FakeWorld(script)
        s = doom.DoomSession(w)
        fire = doom.ReplayTrigger(list(range(8, 14)) + list(range(50, 60)))
        t, _ = _run(s, fire, 4.0)
        fly = s.summary(t)
        w2 = FakeWorld(script)
        again = doom.play_doom_only(w2, round(t * 100), doom.ReplayTrigger(
            [a + k for a, n in fly["holds"] for k in range(n)]))
        self.assertEqual(doom.outcome(again), doom.outcome(fly))


class ScreenTests(unittest.TestCase):
    def test_geometry(self):
        self.assertAlmostEqual(doom.screen_distance_ratio(90.0), 1.0)
        self.assertAlmostEqual(doom.screen_distance_ratio(160.0), 1 / math.tan(math.radians(80)))
        self.assertAlmostEqual(doom.screen_az_deg(1.0, 160.0), 80.0)
        self.assertAlmostEqual(doom.screen_az_deg(0.0, 160.0), 0.0)
        # the centre is magnified: a point 10 % of the half-width off centre sits ~30 deg off the fly's midline
        self.assertGreater(doom.screen_az_deg(0.1, 160.0), 29.0)

    def test_frame_to_linear_is_srgb_then_box(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (4, 6, 3), dtype=np.uint8)
        out = doom.frame_to_linear(frame, 2)
        self.assertEqual(out.shape, (2, 3, 3))
        ref = gc.srgb_to_linear(frame).reshape(2, 2, 3, 2, 3).mean((1, 3))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(doom.frame_to_linear(frame, 1), gc.srgb_to_linear(frame), rtol=1e-5, atol=1e-6)

    def test_closest_recent_object_window(self):
        w = doom.DoomWorld.__new__(doom.DoomWorld)             # no ViZDoom: only the bookkeeping
        w.tic = 20
        w.recent = [(10, ("Demon", 50.0, 0.0, 90)), (15, ("DoomImpBall", 300.0, 0.2, 8)), (20, None)]
        self.assertEqual(w.closest_recent(11), ["Demon", 50.0, 0.0, 90, 10])
        self.assertEqual(w.closest_recent(), ["Demon", 50.0, 0.0, 90, 10])
        self.assertEqual(w.closest_recent(6), ["DoomImpBall", 300.0, 0.2, 8, 5])
        self.assertIsNone(w.closest_recent(1))
        self.assertTrue(w.close_recent())

    def test_episode_seed_formula(self):
        # the formula only; whether two Doom seeds give different monster schedules is Doom's business (see the
        # run log's schedules_no_fire)
        self.assertEqual(doom.episode_seed(0, 1), 1)
        self.assertEqual(doom.episode_seed(101, 2), 101002)
        seeds = {doom.episode_seed(s, e) for s in (0, 1, 2) for e in range(1, 50)}
        self.assertEqual(len(seeds), 3 * 49)


class ModTests(unittest.TestCase):
    def test_pwad_round_trip(self):
        blob = doom.build_pwad([("DECORATE", b"abc"), ("MAPINFO", b"defg")])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.wad"
            p.write_bytes(blob)
            lumps = doom.read_wad(p)
        self.assertEqual(lumps, {"DECORATE": b"abc", "MAPINFO": b"defg"})
        with self.assertRaises(ValueError):
            doom.build_pwad([("NINECHARS", b"")])

    def test_the_pistol_has_no_light_action(self):
        dec = doom.NO_GUN_LIGHT_DECORATE
        self.assertNotIn("A_Light", dec)
        self.assertIn("ACTOR FlyversePistol : Pistol", dec)
        self.assertIn("Flash:", dec)
        self.assertIn('Player.StartItem "FlyversePistol"', dec)
        self.assertIn('playerclasses = "FlyversePlayer"', doom.NO_GUN_LIGHT_MAPINFO)
        self.assertIn("FlyversePlayer", doom.PLAYER_CLASSES)


def _patch(width, height, left, top, columns):
    """Build a Doom patch lump: `columns` is a list of [(topdelta, [palette indices]), ...] per column."""
    body, offsets = b"", []
    head = 8 + 4 * width
    for posts in columns:
        offsets.append(head + len(body))
        for delta, data in posts:
            body += bytes([delta, len(data), 0]) + bytes(data) + b"\0"
        body += b"\xff"
    return struct.pack("<hhhh", width, height, left, top) + struct.pack(f"<{width}I", *offsets) + body


class SpriteTests(unittest.TestCase):
    def test_decode_patch_posts_and_transparency(self):
        pal = np.stack([np.arange(256), 2 * np.arange(256) % 256, 3 * np.arange(256) % 256], 1).astype(np.uint8)
        lump = _patch(2, 3, -5, -7, [[(0, [1, 2])], [(1, [3])]])
        rgba, left, top = doom.decode_patch(lump, pal)
        self.assertEqual((left, top), (-5, -7))
        self.assertEqual(rgba.shape, (3, 2, 4))
        np.testing.assert_array_equal(rgba[:, 0, 3], [255, 255, 0])
        np.testing.assert_array_equal(rgba[:, 1, 3], [0, 255, 0])
        np.testing.assert_array_equal(rgba[1, 0, :3], pal[2])
        np.testing.assert_array_equal(rgba[1, 1, :3], pal[3])

    def test_tall_patch_relative_delta(self):
        pal = np.zeros((256, 3), np.uint8)
        pal[9] = (9, 9, 9)
        # second post's topdelta (2) is not above the first (5): DeePsea convention, it is relative -> row 7
        lump = _patch(1, 10, 0, 0, [[(5, [9]), (2, [9])]])
        rgba, _, _ = doom.decode_patch(lump, pal)
        self.assertEqual(np.flatnonzero(rgba[:, 0, 3]).tolist(), [5, 7])

    def test_read_wad(self):
        lumps = [(b"PLAYPAL", bytes(768)), (b"PISGA0", b"abc"), (b"PISGA0", b"second")]
        data, directory = b"", b""
        for name, blob in lumps:
            directory += struct.pack("<ii8s", 12 + len(data), len(blob), name)
            data += blob
        wad = struct.pack("<4sii", b"IWAD", len(lumps), 12 + len(data)) + data + directory
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.wad"
            p.write_bytes(wad)
            out = doom.read_wad(p)
        self.assertEqual(out["PISGA0"], b"abc")                  # the first lump of a name wins
        self.assertEqual(len(out["PLAYPAL"]), 768)

    def test_weapon_animation(self):
        anim = doom.WeaponAnim()
        self.assertEqual(anim.frame(5), ("PISGA0", False))
        anim.shot(10)
        seq = [anim.frame(t) for t in range(9, 27)]
        self.assertEqual(seq[0], ("PISGA0", False))
        self.assertEqual(seq[1], ("PISGB0", True))
        self.assertEqual(sum(f for _, f in seq), 7)             # the flash lasts 7 tics
        self.assertEqual(seq[7][0], "PISGC0")
        self.assertEqual(seq[-1], ("PISGA0", False))


if __name__ == "__main__":
    unittest.main()
