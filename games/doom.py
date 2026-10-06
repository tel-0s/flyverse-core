"""doom -- yes, it runs Doom.

The MaleCNS v1.0 brain (preset 'raw', nothing instrumented) watches Freedoom through its compound eye and holds the
trigger. The ViZDoom scenario `defend_the_line` sends demons down a corridor at the player while imps throw fireballs
from the far wall. Each Doom frame is shown to the fly as a flat screen spanning HFOV_DEG of its horizontal field
(`Eyes.from_pinhole`), i.e. the fly stands a few centimetres in front of a monitor. The only thing the brain controls
is ATTACK, through one declared decoder:

    gf_trigger (DECODER)   ATTACK is held for the next Doom tic when the mean rate of the two giant fibres (DNp01 L/R)
                           reached >= 33 Hz during the current tic -- the body model's escape-takeoff threshold
                           (flyverse/body.py, Flight.gf_hz). The trigger reads the connectome; it writes nothing.

Everything else is GAME (games/captions/doom.md has the full table):
* Doom itself: Freedoom (BSD) on ViZDoom, 35 tics per brain second, a 1 s GET READY, and a freeze and a new episode
  after a death. The trigger is not live while Doom waits.
* The player's pistol is Freedoom's with its muzzle-flash *lighting* removed (a DECORATE player class loaded as a
  PWAD, NO_GUN_LIGHT_DECORATE). Doom's A_Light1 brightens the whole 3D view by about 10 % for 7 tics per shot, which by
  itself drives the giant fibre to about 40 Hz half a second later: with it, the gun re-triggers itself. The shot's
  effects in the world (bullet puffs, a hit monster vanishing) remain. `--gun-light` restores it (ablation).
* The pistol sprite is drawn over the *human* view only. The fly sees the 3D view without it.
* The view never turns in recordings. In interactive mode a human turns it.
* After a recorded run, Doom-only baselines on the same Doom seeds and the same timeline (never fire, always fire,
  random triggers matched to the fly's rate and hold lengths, and a replay of the fly's own ATTACK tics as a
  determinism check) are written into the run log's summary.

    python games/doom.py                                   # interactive: arrows / A-D turn the view (GAME, human)
    python games/doom.py --seed 101 --seconds 30 --record out/games/doom/dev.mp4 --screenshot out/games/doom/dev.png
    python games/doom.py --seed 101 --seconds 30 --control off --record out/games/doom/dev_control.mp4
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games import common as gc  # noqa: E402

# ---------------------------------------------------------------------------------------------------- constants
TICRATE = 35                    # Doom tics per second; GAME: one Doom second = one brain second
HFOV_DEG = 160.0                # the Doom frame's width as seen by the fly (flat screen, pinhole); tuned on seeds >= 100
GF_THRESHOLD_HZ = 33.0          # flyverse.body.Flight.gf_hz: the shipped escape-takeoff threshold (checked at start-up)
SCENARIO = "defend_the_line"
SKILL = 3                       # the scenario's own default
DOOM_RES = (640, 480)           # ViZDoom render; the fly gets a 2x2 box-filtered 320x240 copy
BASELINE_RES = (160, 120)       # Doom-only baselines and schedule checks (Doom's game logic does not depend on the render)
FLY_DOWNSAMPLE = 2
WARM_S = 1.0                    # GAME: the fly sees the first frame for this long before Doom starts ("GET READY")
DEATH_FREEZE_S = 0.6            # GAME: after a death the last frame holds this long, then a new episode is loaded ...
READY_S = 0.6                   # ... and its first frame is shown this long before Doom resumes (the gun is not live
                                # during either: a scene cut is not something the fly should be able to shoot at)
CONTEXT_UNITS = 128.0           # a GF crossing / trigger "has something close" when a labelled object was nearer (map units)
CONTEXT_LOOKBACK_TICS = 18      # ... at any tic in the last ~500 ms
OWN_SHOT_WINDOW_S = (0.1, 0.7)  # a trigger or GF crossing this long after the player's last shot may be the shot's doing
BANNER_S = 0.8                  # event banners fade over this long
KILL_WINDOW_TICS = 3            # a KILLCOUNT step within this many tics of a player hit is the player's kill
TURN_DEG_PER_TIC = 3.0          # GAME, interactive only: a human's turn rate
BASELINE_DRAWS = 50             # random-trigger draws per recorded run (Doom only, ~0.3 s each)
PLAYER_CLASSES = ("DoomPlayer", "FlyversePlayer")

# Doom map units: the player is 56 units tall (~1.7 m), so 1 unit ~ 3 cm.

# GAME: the pistol without Doom's muzzle-flash lighting. Freedoom's own Pistol (vizdoom.pk3 actors/doom/doomweapons.txt)
# has `Flash: PISF A 7 Bright A_Light1`; A_Light1 sets the player's extralight, which brightens every sector in the
# rendered 3D view. This subclass keeps the flash sprite and removes the light; a player class starts with it.
NO_GUN_LIGHT_DECORATE = """\
ACTOR FlyversePistol : Pistol
{
  States
  {
  Flash:
    PISF A 7 Bright
    Goto LightDone
    PISF A 7 Bright
    Goto LightDone
  }
}

ACTOR FlyversePlayer : DoomPlayer
{
  Player.StartItem "FlyversePistol"
  Player.StartItem "Fist"
  Player.StartItem "Clip", 50
  Player.WeaponSlot 2, FlyversePistol
}
"""
NO_GUN_LIGHT_MAPINFO = """\
gameinfo
{
  playerclasses = "FlyversePlayer"
}
"""


# ---------------------------------------------------------------------------------------------------- pure pieces
_SRGB_LUT = gc.srgb_to_linear(np.arange(256, dtype=np.uint8)).astype(np.float32)


def frame_to_linear(frame_u8: np.ndarray, factor: int = FLY_DOWNSAMPLE) -> np.ndarray:
    """(H, W, 3) sRGB uint8 -> (H/f, W/f, 3) linear RGB float32, box-filtered f x f (an anti-alias step before the
    ommatidial rays sample it; the screen's pixels are finer than the fly's 7 rays per column)."""
    lin = _SRGB_LUT[frame_u8]
    if factor <= 1:
        return lin
    H, W = lin.shape[:2]
    H2, W2 = H // factor, W // factor
    return lin[:H2 * factor, :W2 * factor].reshape(H2, factor, W2, factor, 3).mean((1, 3))


def screen_distance_ratio(hfov_deg: float) -> float:
    """Distance from the eye to a flat screen, in units of the screen's half-width, for the screen to span
    `hfov_deg` horizontally: d / (w/2) = 1 / tan(hfov / 2)."""
    return 1.0 / math.tan(math.radians(hfov_deg) / 2)


def screen_az_deg(x_norm: float, hfov_deg: float = HFOV_DEG) -> float:
    """Azimuth (deg, + = right of centre here) at which the fly sees a point at normalised screen x (-1 .. 1)."""
    return math.degrees(math.atan(x_norm * math.tan(math.radians(hfov_deg) / 2)))


def turn_delta(left: bool, right: bool, deg_per_tic: float = TURN_DEG_PER_TIC) -> float:
    """The TURN_LEFT_RIGHT_DELTA value for held keys. ViZDoom turns *right* for a positive delta (Doom's ANGLE falls)."""
    return float(deg_per_tic) * (float(bool(right)) - float(bool(left)))


class TicClock:
    """Brain milliseconds -> whole Doom tics at TICRATE per brain second (one tic every 28.57 ms)."""

    def __init__(self, ticrate: int = TICRATE):
        self.ticrate, self.ms, self.tics = ticrate, 0.0, 0

    def advance(self, ms: float) -> int:
        self.ms += ms
        due = int(math.floor(self.ms * self.ticrate / 1000.0 + 1e-9)) - self.tics
        self.tics += due
        return due


class GFTrigger:
    """The decoder's law, as pure logic: `observe(gf_hz)` every brain frame; `fire_for_tic()` once per Doom tic returns
    (hold ATTACK, peak) -- hold if the giant-fibre rate reached the threshold at any frame since the previous tic.
    `idle()` is called on every brain frame while the gun is not live (it forgets the running peak)."""

    def __init__(self, threshold_hz: float = GF_THRESHOLD_HZ):
        self.threshold = float(threshold_hz)
        self.peak = 0.0            # max GF since the last tic

    def observe(self, gf_hz: float):
        self.peak = max(self.peak, float(gf_hz))

    def fire_for_tic(self, tic=None) -> tuple[bool, float]:
        peak, self.peak = self.peak, 0.0
        return peak >= self.threshold, peak

    def idle(self):
        self.peak = 0.0


class FixedTrigger:
    """GAME baseline: always (or never) hold ATTACK while the gun is live."""

    def __init__(self, fire: bool):
        self.fire = bool(fire)

    def fire_for_tic(self, tic=None):
        return self.fire, 0.0

    def idle(self):
        pass


class ReplayTrigger:
    """GAME check: hold ATTACK exactly on the given Doom tics (a fly run's own, from its log)."""

    def __init__(self, tics):
        self.tics = set(int(t) for t in tics)

    def fire_for_tic(self, tic=None):
        return tic in self.tics, 0.0

    def idle(self):
        pass


class RandomTrigger:
    """GAME baseline: on each live tic that is not part of a hold (and does not directly follow one), start a hold
    with probability `p`; hold lengths are drawn from `holds` (the fly's own). Rate- and duration-matched chance."""

    def __init__(self, p: float, holds, rng):
        self.p, self.holds, self.rng = float(p), [max(1, int(h)) for h in holds] or [1], rng
        self.left, self.gap = 0, False

    def fire_for_tic(self, tic=None):
        if self.left > 0:
            self.left -= 1
            self.gap = self.left == 0
            return True, 0.0
        if self.gap:                                   # one released tic between holds, so each hold is its own trigger
            self.gap = False
            return False, 0.0
        if self.rng.random() < self.p:
            self.left = int(self.rng.choice(self.holds)) - 1
            self.gap = self.left == 0
            return True, 0.0
        return False, 0.0

    def idle(self):
        self.left, self.gap = 0, False


def episode_seed(run_seed: int, episode: int) -> int:
    """The Doom seed of episode `episode` (1-based) of a run: 1000 * seed + episode. Distinct seeds do not guarantee
    distinct games: in defend_the_line many Doom seeds replay one monster schedule (the run log's `schedules_no_fire`
    says which schedule each episode had)."""
    return 1000 * int(run_seed) + int(episode)


def upward_crossings(values, threshold: float) -> list[int]:
    """Indices i where values[i] >= threshold > values[i-1] (the first sample counts if it is already above)."""
    v = np.asarray(values, np.float64)
    above = v >= threshold
    prev = np.r_[False, above[:-1]]
    return np.flatnonzero(above & ~prev).tolist()


def build_pwad(lumps) -> bytes:
    """[(name, bytes), ...] -> a PWAD file (header, lump data, directory)."""
    data, directory = b"", b""
    for name, blob in lumps:
        if len(name) > 8:
            raise ValueError(f"lump name too long: {name!r}")
        directory += struct.pack("<ii8s", 12 + len(data), len(blob), name.encode("ascii"))
        data += blob
    return struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(data)) + data + directory


def no_gun_light_wad() -> tuple[str, str]:
    """Write the no-muzzle-light PWAD (deterministic bytes) to the temp dir; returns (path, sha256[:16])."""
    blob = build_pwad([("DECORATE", NO_GUN_LIGHT_DECORATE.encode("ascii")),
                       ("MAPINFO", NO_GUN_LIGHT_MAPINFO.encode("ascii"))])
    sha = hashlib.sha256(blob).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"flyverse_doom_no_gun_light_{sha}.wad")
    if " " in path:
        raise SystemExit(f"doom: the temp dir path has a space, which ViZDoom's -file cannot take: {path}")
    if not os.path.isfile(path):
        with open(path, "wb") as f:
            f.write(blob)
    return path, sha


def outcome(s: dict) -> dict:
    """The comparable part of a run summary (fly or baseline)."""
    keys = ("triggers", "trigger_tics", "shots", "hits", "misses", "kills", "infighting_kills", "deaths", "hurt",
            "doom_tics")
    return {**{k: int(s[k]) for k in keys}, "survival_s": list(s["survival_s"])}


def fly_rank(fly: dict, runs: list) -> dict:
    """Where a fly run's outcome sits among random-trigger draws."""
    def mmm(key):
        v = sorted(r[key] for r in runs)
        return [v[0], float(np.median(v)), v[-1]] if v else None
    return {"draws": len(runs),
            "draws_with_at_least_as_many_kills": sum(r["kills"] >= fly["kills"] for r in runs),
            "draws_with_more_kills": sum(r["kills"] > fly["kills"] for r in runs),
            "draws_with_at_most_as_many_deaths": sum(r["deaths"] <= fly["deaths"] for r in runs),
            "draws_with_fewer_deaths": sum(r["deaths"] < fly["deaths"] for r in runs),
            "kills_min_median_max": mmm("kills"), "deaths_min_median_max": mmm("deaths"),
            "triggers_min_median_max": mmm("triggers"), "shots_min_median_max": mmm("shots")}


# ------------------------------------------------------------------ Freedoom pistol overlay (human view only, GAME)
def read_wad(path) -> dict:
    """{lump name: bytes} of a Doom WAD (first lump of each name)."""
    b = Path(path).read_bytes()
    ident, n, off = struct.unpack_from("<4sii", b, 0)
    if ident not in (b"IWAD", b"PWAD"):
        raise ValueError(f"{path}: not a WAD")
    lumps = {}
    for i in range(n):
        o, s, nm = struct.unpack_from("<ii8s", b, off + 16 * i)
        lumps.setdefault(nm.rstrip(b"\0").decode("latin1"), b[o:o + s])
    return lumps


def decode_patch(lump: bytes, palette: np.ndarray):
    """A Doom picture ("patch") lump -> ((h, w, 4) RGBA uint8, leftoffset, topoffset). Column posts; transparent
    where no post covers a pixel. Handles the DeePsea tall-patch convention (a non-increasing topdelta is relative)."""
    w, h, left, top = struct.unpack_from("<hhhh", lump, 0)
    cols = struct.unpack_from(f"<{w}I", lump, 8)
    out = np.zeros((h, w, 4), np.uint8)
    for x, p in enumerate(cols):
        ytop = -1
        while lump[p] != 0xFF:
            delta, length = lump[p], lump[p + 1]
            ytop = delta if delta > ytop else ytop + delta
            idx = np.frombuffer(lump, np.uint8, count=length, offset=p + 3)
            ys = ytop + np.arange(length)
            ok = ys < h
            out[ys[ok], x, :3] = palette[idx[ok]]
            out[ys[ok], x, 3] = 255
            p += length + 4
    return out, left, top


class WeaponAnim:
    """Which pistol frame to draw, from the tic a shot was detected (Doom's pistol: PISG B 6 tics with the PISF flash
    7 tics, PISG C 4, PISG B 5, then ready PISG A). Cosmetic: only the human view shows it."""

    SEQ = ["PISGB0"] * 6 + ["PISGC0"] * 4 + ["PISGB0"] * 5

    def __init__(self):
        self.shot_tic = None

    def shot(self, tic: int):
        self.shot_tic = tic

    def frame(self, tic: int) -> tuple[str, bool]:
        if self.shot_tic is None or tic < self.shot_tic or tic - self.shot_tic >= len(self.SEQ):
            return "PISGA0", False
        k = tic - self.shot_tic
        return self.SEQ[k], k < 7


# ---------------------------------------------------------------------------------------------------- Doom (GAME)
class DoomWorld:
    """ViZDoom in synchronous player mode, one tic per `step`. Renders the 3D view without the weapon sprite (the
    fly's screen); keeps the counters a run log needs. Everything here is GAME."""

    VARS = ("KILLCOUNT", "HITCOUNT", "HEALTH", "HITS_TAKEN", "POSITION_X", "POSITION_Y", "ANGLE")

    def __init__(self, seed: int, scenario: str = SCENARIO, skill: int = SKILL, *, weapon_for_fly: bool = False,
                 gun_light: bool = False, res=DOOM_RES, labels: bool = True):
        import vizdoom as vzd
        self.vzd = vzd
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, scenario + ".cfg"))
        g.set_window_visible(False)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_screen_resolution(getattr(vzd.ScreenResolution, "RES_%dX%d" % tuple(res)))
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_render_hud(False)
        g.set_render_weapon(bool(weapon_for_fly))
        g.set_render_crosshair(False)
        g.set_render_screen_flashes(False)            # the scenario's own setting, restated
        g.set_labels_buffer_enabled(bool(labels))
        g.set_objects_info_enabled(True)
        g.set_sound_enabled(False)
        g.set_episode_timeout(TICRATE * 3600)
        g.set_doom_skill(skill)
        g.set_seed(int(seed))
        g.set_available_buttons([vzd.Button.ATTACK, vzd.Button.TURN_LEFT_RIGHT_DELTA])
        g.set_available_game_variables([getattr(vzd.GameVariable, v) for v in self.VARS])
        self.mod = None
        if not gun_light:
            path, sha = no_gun_light_wad()
            g.add_game_args("-file " + path)
            self.mod = {"pwad": Path(path).name, "sha256_16": sha, "what": "pistol without A_Light1 (no muzzle light)"}
        g.init()
        self.g, self.scenario, self.skill, self.seed = g, scenario, skill, seed
        self.version = getattr(vzd, "__version__", "?")
        self.iwad = Path(g.get_doom_game_path() or "freedoom2.wad").name
        self.restart()

    def restart(self):
        """Back to episode 1 of this run (the baselines replay a run's Doom seeds from the start)."""
        self.episode = 0
        self.tic = 0                                   # tics played in total
        self.new_episode()

    def new_episode(self):
        self.doom_seed = episode_seed(self.seed, self.episode + 1)
        self.g.set_seed(self.doom_seed)
        self.g.new_episode()
        self.episode += 1
        self.ep_tic = 0
        self._seen_puffs = set()
        st = self.g.get_state()
        self.vars = dict(zip(self.VARS, st.game_variables.tolist()))
        self.frame = st.screen_buffer.copy()
        self.context = self._context(st)
        self.recent = [(self.tic, self.context[0] if self.context else None)]

    def closest_recent(self, n_tics: int = CONTEXT_LOOKBACK_TICS):
        """The closest labelled object over the last `n_tics` tics of this episode: [name, units, screen x, width px,
        tics ago], or None."""
        best = None
        for tic, obj in self.recent:
            if obj is not None and self.tic - tic < n_tics and (best is None or obj[1] < best[1]):
                best = list(obj) + [self.tic - tic]
        return best

    def close_recent(self) -> bool:
        best = self.closest_recent()
        return best is not None and best[1] <= CONTEXT_UNITS

    def nearest(self, n: int = 3) -> list:
        return [list(z) for z in self.context[:n]]

    def _context(self, st):
        """Labelled (visible) objects sorted by distance: (name, map units, screen x in -1..1, width px)."""
        if st.labels is None:
            return []
        px, py = self.vars["POSITION_X"], self.vars["POSITION_Y"]
        W = st.screen_buffer.shape[1]
        out = []
        for lab in st.labels:
            if lab.object_name in PLAYER_CLASSES:
                continue
            d = float(np.hypot(lab.object_position_x - px, lab.object_position_y - py))
            out.append((lab.object_name, round(d, 1), round((lab.x + lab.width / 2) / (W / 2) - 1, 3), int(lab.width)))
        return sorted(out, key=lambda z: z[1])

    def step(self, fire: bool, turn_deg: float = 0.0) -> dict:
        """Play one tic. Returns {'hits', 'misses', 'kills', 'hurt', 'dead', 'health'} for this tic."""
        g = self.g
        g.make_action([1.0 if fire else 0.0, float(turn_deg)], 1)
        self.tic += 1
        self.ep_tic += 1
        out = {"hits": 0, "misses": 0, "kills": 0, "hurt": 0, "dead": False, "health": self.vars["HEALTH"]}
        if g.is_episode_finished():
            dead = bool(g.is_player_dead())
            out.update(dead=True, timeout=not dead, health=0 if dead else self.vars["HEALTH"])   # a timeout ends it too
            return out
        st = g.get_state()
        v = dict(zip(self.VARS, st.game_variables.tolist()))
        out["hits"] = int(max(0, v["HITCOUNT"] - self.vars["HITCOUNT"]))
        out["kills"] = int(max(0, v["KILLCOUNT"] - self.vars["KILLCOUNT"]))
        out["hurt"] = int(max(0, v["HITS_TAKEN"] - self.vars["HITS_TAKEN"]))
        out["health"] = v["HEALTH"]
        for o in st.objects:
            if o.name == "BulletPuff" and o.id not in self._seen_puffs:
                self._seen_puffs.add(o.id)
                out["misses"] += 1
        self.vars = v
        self.frame = st.screen_buffer.copy()
        self.context = self._context(st)
        self.recent = self.recent[-(CONTEXT_LOOKBACK_TICS - 1):] + [(self.tic, self.context[0] if self.context else None)]
        return out

    def close(self):
        try:
            self.g.close()
        except Exception:
            pass


class DoomSession:
    """GAME: the Doom side of a run, one 10 ms brain frame at a time -- the 1 s GET READY, 35 tics per brain second, the
    freeze and new episode after a death, and the score. `frame(t_s, trigger)` advances the frame that ends at brain
    time `t_s`; `trigger` has `fire_for_tic(tic) -> (hold ATTACK, peak)` and `idle()`. The brain game and the Doom-only
    baselines both run through this class, so their timelines and scoring are the same code."""

    def __init__(self, world, *, connected: bool = True, log=None):
        self.world, self.connected, self.log = world, bool(connected), log
        self.clock = TicClock()
        self.frozen_until = None
        self.respawned = True
        self.fire_now = False
        self._cmd_prev = False
        self.stats = {"kills": 0, "hits": 0, "misses": 0, "deaths": 0, "timeouts": 0, "hurt": 0, "infighting_kills": 0,
                      "decoder_commands": 0, "triggers": 0, "trigger_tics": 0, "triggers_close": 0,
                      "triggers_after_own_shot": 0}
        self.last_hit_tic = -999
        self.last_shot_t = None
        self.ep_start_t = WARM_S
        self.ep_kills = 0
        self.episodes = []
        self.holds = []                   # [first Doom tic, number of tics] of every hold of ATTACK
        self.health = float(world.vars["HEALTH"])

    def live(self, t_s: float) -> bool:
        """Is the gun live (Doom running) in the frame ending at `t_s`?"""
        return t_s >= WARM_S - 1e-9 and self.frozen_until is None

    def since_last_shot(self, t_s: float):
        return None if self.last_shot_t is None else round(t_s - self.last_shot_t, 3)

    def after_own_shot(self, t_s: float) -> bool:
        s = self.since_last_shot(t_s)
        return s is not None and OWN_SHOT_WINDOW_S[0] <= s <= OWN_SHOT_WINDOW_S[1]

    def _event(self, t_s, kind, **detail):
        if self.log is not None:
            self.log.event(t_s, kind, **detail)

    def _idle(self, trigger):
        trigger.idle()
        self._cmd_prev = False
        self.fire_now = False

    def frame(self, t_s: float, trigger, turn: float = 0.0) -> list:
        """Advance one brain frame. Returns what happened: one record per Doom tic played (see `_tic`) and
        {'marker': 'respawn' | 'resume'} when a new episode is loaded / resumes."""
        if t_s < WARM_S - 1e-9:
            self._idle(trigger)
            return []
        if self.frozen_until is not None:
            self._idle(trigger)
            out = []
            if not self.respawned and t_s >= self.frozen_until - READY_S - 1e-9:
                self.respawned = True
                self.world.new_episode()
                self.health = float(self.world.vars["HEALTH"])
                self._event(t_s, "respawn", episode=self.world.episode, doom_seed=self.world.doom_seed)
                out.append({"marker": "respawn"})
            if t_s >= self.frozen_until - 1e-9:
                self.frozen_until = None
                self.ep_start_t = t_s
                self.ep_kills = 0
                self._event(t_s, "resume", episode=self.world.episode)
                out.append({"marker": "resume"})
            return out
        out = []
        for _ in range(self.clock.advance(gc.TICK_MS)):
            out.append(self._tic(t_s, trigger, turn))
            if self.frozen_until is not None:
                break
        return out

    def _tic(self, t_s, trigger, turn):
        w = self.world
        cmd, peak = trigger.fire_for_tic(w.tic + 1)
        cmd = bool(cmd)
        cmd_rising = cmd and not self._cmd_prev
        self._cmd_prev = cmd
        fire, rising = cmd and self.connected, cmd_rising and self.connected
        self.fire_now = fire
        self.stats["decoder_commands"] += int(cmd_rising)
        if fire:
            self.stats["trigger_tics"] += 1
            if rising:
                self.holds.append([w.tic + 1, 0])
            self.holds[-1][1] += 1
        if rising:
            close, own = w.close_recent(), self.after_own_shot(t_s)
            self.stats["triggers"] += 1
            self.stats["triggers_close"] += int(close)
            self.stats["triggers_after_own_shot"] += int(own)
            self._event(t_s, "trigger", gf_peak_hz=round(float(peak), 1), since_last_shot_s=self.since_last_shot(t_s),
                        after_own_shot=own, nearest=w.nearest(), closest_500ms=w.closest_recent())
        res = w.step(fire, turn)
        tic = w.tic
        rec = {"fire": fire, "rising": rising, "peak_hz": float(peak), "mine": False, **res}
        if res["hits"] + res["misses"]:
            self.last_shot_t = t_s
        if res["hits"]:
            self.stats["hits"] += res["hits"]
            self.last_hit_tic = tic
            self._event(t_s, "shot", result="hit", n=res["hits"])
        if res["misses"]:
            self.stats["misses"] += res["misses"]
            self._event(t_s, "shot", result="miss", n=res["misses"])
        if res["kills"]:
            # Doom's KILLCOUNT also counts monsters killed by other monsters (an imp's fireball can kill a demon). A kill
            # is credited to the player's gun only when a player hit landed in this tic or the KILL_WINDOW before it.
            mine = tic - self.last_hit_tic <= KILL_WINDOW_TICS
            key = "kills" if mine else "infighting_kills"
            self.stats[key] += res["kills"]
            if mine:
                self.ep_kills += res["kills"]
            rec["mine"] = mine
            self._event(t_s, "kill", by="player" if mine else "another monster", n=res["kills"],
                        total=self.stats[key], nearest=w.nearest(2))
        if res["hurt"]:
            self.stats["hurt"] += res["hurt"]
            self._event(t_s, "hurt", health=res["health"])
        self.health = float(res["health"])
        if res["dead"]:
            timeout = bool(res.get("timeout", False))
            self.stats["timeouts" if timeout else "deaths"] += 1
            ep = {"episode": w.episode, "doom_seed": w.doom_seed, "start_s": round(self.ep_start_t, 3),
                  "end_s": round(t_s, 3), "survived_s": round(t_s - self.ep_start_t, 3), "kills": self.ep_kills,
                  "ended": "timeout" if timeout else "death"}
            self.episodes.append(ep)
            self._event(t_s, "timeout" if timeout else "death", episode=w.episode, survived_s=ep["survived_s"])
            self.frozen_until = t_s + DEATH_FREEZE_S + READY_S
            self.respawned = False
            self.clock = TicClock()
            self._cmd_prev = False
            self.fire_now = False
        return rec

    def summary(self, t_s: float) -> dict:
        """Counts, episodes (the one running at the end is 'clip end') and every hold of ATTACK. Does not mutate."""
        s = dict(self.stats)
        eps = [dict(e) for e in self.episodes]
        if self.frozen_until is None and t_s >= WARM_S - 1e-9:
            eps.append({"episode": self.world.episode, "doom_seed": self.world.doom_seed,
                        "start_s": round(self.ep_start_t, 3), "end_s": round(t_s, 3),
                        "survived_s": round(t_s - self.ep_start_t, 3), "kills": self.ep_kills, "ended": "clip end"})
        s.update(shots=s["hits"] + s["misses"], doom_tics=int(self.world.tic), episodes=eps,
                 survival_s=[e["survived_s"] for e in eps if e["ended"] == "death"],
                 holds=[list(h) for h in self.holds])
        return s


def play_doom_only(world, n_frames: int, trigger) -> dict:
    """GAME: replay a run's Doom (same seeds, same brain-frame timeline) with `trigger` in place of the fly."""
    world.restart()
    sess = DoomSession(world)
    t = 0.0
    for _ in range(int(n_frames)):
        t += gc.TICK_MS / 1000.0                  # the same float accumulation as the game's t_s
        sess.frame(t, trigger)
    return sess.summary(t)


def run_baselines(seed: int, n_frames: int, fly: dict, *, draws: int = BASELINE_DRAWS, scenario: str = SCENARIO,
                  skill: int = SKILL, gun_light: bool = False) -> dict:
    """GAME baselines on the Doom seeds and timeline of a fly run (no brain): never fire, always fire, a replay of the
    fly's own ATTACK tics (a determinism check: it must reproduce the fly's Doom outcome), and `draws` random triggers
    matched to the fly's trigger rate and hold lengths."""
    world = DoomWorld(seed, scenario, skill, gun_light=gun_light, res=BASELINE_RES, labels=False)
    try:
        fly_o = outcome(fly)
        out = {"what": "Doom only, same Doom seeds and brain-frame timeline as the fly run; kills/deaths by the same rules",
               "frames": int(n_frames), "fly": fly_o,
               "never": outcome(play_doom_only(world, n_frames, FixedTrigger(False))),
               "always": outcome(play_doom_only(world, n_frames, FixedTrigger(True)))}
        tics = [first + k for first, n in fly["holds"] for k in range(n)]
        out["replay"] = outcome(play_doom_only(world, n_frames, ReplayTrigger(tics)))
        out["replay_matches_fly"] = all(out["replay"][k] == fly_o[k] for k in ("hits", "misses", "kills", "deaths", "doom_tics"))
        if fly["triggers"] > 0 and draws > 0:
            holds = [n for _, n in fly["holds"]]
            p = fly["triggers"] / max(1, fly["doom_tics"] - fly["trigger_tics"])
            runs = [outcome(play_doom_only(world, n_frames, RandomTrigger(p, holds, np.random.default_rng([int(seed), d]))))
                    for d in range(int(draws))]
            out["random"] = {"p_start_per_free_tic": round(p, 5), "hold_tics_drawn_from": holds,
                             "rng": "numpy default_rng([seed, draw])", "rank_of_fly": fly_rank(fly_o, runs), "runs": runs}
        return out
    finally:
        world.close()


def schedule_fingerprints(seed: int, episodes, *, scenario: str = SCENARIO, skill: int = SKILL,
                          max_tics: int = 600) -> list:
    """GAME: what Doom does with nobody firing in each listed episode of a run (its Doom seed): the episode tic at which
    the player dies and the first three tics it is hurt. Equal fingerprints mean the same monster schedule."""
    world = DoomWorld(seed, scenario, skill, res=BASELINE_RES, labels=False)
    try:
        out = []
        for e in episodes:
            world.episode = int(e) - 1
            world.new_episode()
            hurt, death = [], None
            for t in range(1, max_tics + 1):
                r = world.step(False)
                if r["dead"]:
                    death = t
                    break
                if r["hurt"]:
                    hurt.append(t)
            out.append({"episode": int(e), "doom_seed": episode_seed(seed, e), "no_fire_death_tic": death,
                        "no_fire_first_hurt_tics": hurt[:3]})
        return out
    finally:
        world.close()


# ---------------------------------------------------------------------------------------------------- the game
class DoomFly(gc.Game):
    title = "doom"
    subtitle = "yes, it runs Doom · the giant fibre pulls the trigger"

    def __init__(self, args):
        super().__init__(args)
        import torch
        from flyverse.body import Flight
        self.torch = torch
        if abs(Flight().gf_hz - GF_THRESHOLD_HZ) > 1e-9:
            raise SystemExit(f"flyverse.body.Flight.gf_hz is {Flight().gf_hz}, this game says {GF_THRESHOLD_HZ}")
        self.hfov = float(args.hfov)
        self.control = args.control
        self.human = args.record is None
        self.fb = gc.build_brain(args)                                 # preset 'raw', nothing attached yet
        self.eyes = gc.Eyes(self.fb)
        c = self.fb.c
        # the one decoder: attached, read-only, recorded in module_records()
        self.trigger = GFTrigger(GF_THRESHOLD_HZ)
        law = (f"hold ATTACK for the next Doom tic when the mean rate of DNp01 L+R (giant fibres) reached >= "
               f"{GF_THRESHOLD_HZ:g} Hz during the current tic")
        self.gf_dec = gc.ReadDecoder("gf_trigger", {"gf": {"type": "DNp01"}}, _mean_rate, law=law,
                                     parameters={"threshold_hz": GF_THRESHOLD_HZ, "source": "flyverse.body.Flight.gf_hz",
                                                 "window": "max over the brain frames of one Doom tic (28.6 ms)",
                                                 "latency_ms": "one brain frame (10) + up to one tic (28.6)",
                                                 "connected": self.control == "on"})
        self.fb.attach(self.gf_dec)
        # CONNECTOME readouts for the HUD (display only; not controls)
        side = c.neurons.somaSide.fillna("").to_numpy()
        groups = {}
        for t in ("LPLC2", "LC4"):
            idx = c.select(type=t)
            for s in "LR":
                groups[f"{t} {s}"] = idx[side[idx] == s]
        groups["GF"] = c.select(type="DNp01")
        groups["TTMn"] = c.select(type="TTMn")
        self.read_names = list(groups)
        flat = np.concatenate([groups[k] for k in self.read_names])
        self.read_seg = np.cumsum([0] + [len(groups[k]) for k in self.read_names])
        self.read_idx = torch.as_tensor(flat, device=self.fb.device)
        self.rates = {k: 0.0 for k in self.read_names}
        # Doom
        self.world = DoomWorld(args.seed, args.scenario, args.skill, weapon_for_fly=args.weapon_for_fly,
                               gun_light=args.gun_light)
        self.session = DoomSession(self.world, connected=self.control == "on", log=self.log)
        self.rad = self._eye(self.world.frame)
        self.sprites = _load_pistol_sprites(self.world)
        self.anim = WeaponAnim()
        # state for HUD and log
        n_hist = 400
        self.hist = {"gf": [0.0] * n_hist, "lplc2": [0.0] * n_hist, "lc4": [0.0] * n_hist}
        self.gf_prev = 0.0
        self.turn_input = 0.0
        self.lanes = {}                   # fixed banner slots: lane -> (t_start, text, colour, size, chip)
        self.toasts = []                  # GAME notes, lower left: (t_start, slot, text, colour)
        self.hurt_t = -9.0
        self.gstats = {"gf_crossings": 0, "gf_crossings_live": 0, "gf_crossings_close": 0,
                       "gf_crossings_after_own_shot": 0}
        self.gf_max = 0.0
        self.gf_above_frames = 0
        self.frames_played = 0
        self.peak_rates = {k: 0.0 for k in self.read_names}
        self._full = None                 # a 1920x1080 canvas when --size differs (the layout is in its pixels)
        self._declare()
        self.log.event(0.0, "start", scenario=args.scenario, skill=args.skill, hfov_deg=self.hfov, control=self.control,
                       warm_s=WARM_S, gun_light=bool(args.gun_light), doom_seed=self.world.doom_seed)

    # ------------------------------------------------------------------ provenance
    def _declare(self):
        a, w = self.args, self.world
        m = self.log.meta
        m["doom"] = {"engine": f"ViZDoom {w.version}", "iwad": w.iwad, "assets": "Freedoom (BSD-3-Clause)",
                     "scenario": a.scenario, "skill": a.skill, "render": "%dx%d" % DOOM_RES, "ticrate_per_brain_s": TICRATE,
                     "doom_seed": "1000 * seed + episode (episode 1, 2, ...)", "hfov_deg": self.hfov, "control": self.control,
                     "screen_distance_over_half_width": round(screen_distance_ratio(self.hfov), 4),
                     "mod": w.mod, "gun_light": bool(a.gun_light)}
        gc.declare(self.log, "eyes: Doom screen", "game",
                   f"each tic's 3D view ({'WITH the weapon sprite (ablation)' if a.weapon_for_fly else 'no weapon sprite'}, "
                   "no HUD), sRGB -> linear, 2x2 box to 320x240, shown to the "
                   f"fly as a flat screen spanning {self.hfov:g} deg of azimuth (Eyes.from_pinhole); radiance 0 outside "
                   "the screen; UV = 0.5 B", reads="ViZDoom screen buffer", hfov_deg=self.hfov, outside=[0, 0, 0, 0],
                   weapon_sprite_shown_to_fly=bool(a.weapon_for_fly))
        gc.declare(self.log, "pistol without muzzle light", "game",
                   ("VANILLA (ablation): Doom's A_Light1 muzzle light brightens the whole 3D view ~10 % for 7 tics per shot"
                    if a.gun_light else
                    "a DECORATE player class (FlyversePlayer) starts with FlyversePistol, Freedoom's Pistol whose Flash "
                    "state has no A_Light1, so a shot does not brighten the fly's screen; bullets, damage and the flash "
                    "sprite are unchanged"), mod=w.mod)
        gc.declare(self.log, "doom", "game",
                   f"Freedoom via ViZDoom, scenario {a.scenario} at skill {a.skill}; {TICRATE} Doom tics per brain second; "
                   f"Doom starts after {WARM_S:g} s; after a death the frame freezes {DEATH_FREEZE_S:g} s, then the new "
                   f"episode's first frame (Doom seed 1000 * seed + episode) shows {READY_S:g} s before Doom resumes; the "
                   "trigger is not live while Doom waits", scenario=a.scenario, skill=a.skill,
                   ticrate=TICRATE, warm_s=WARM_S, death_freeze_s=DEATH_FREEZE_S, ready_s=READY_S)
        gc.declare(self.log, "view direction", "game",
                   "recordings: the player never turns (no decoder turns it); interactive: arrow keys / A-D turn it "
                   "(a human, GAME)", turn_deg_per_tic_human=TURN_DEG_PER_TIC if self.human else 0.0)
        gc.declare(self.log, "score", "game",
                   "Doom's counters: a shot is a HITCOUNT step (hit) or a new BulletPuff object (miss); a KILLCOUNT step "
                   f"is the gun's kill only if a player hit landed within {KILL_WINDOW_TICS} tics, else infighting",
                   kill_window_tics=KILL_WINDOW_TICS, close_units=CONTEXT_UNITS, lookback_tics=CONTEXT_LOOKBACK_TICS,
                   own_shot_window_s=list(OWN_SHOT_WINDOW_S))
        gc.declare(self.log, "weapon overlay", "game",
                   "human view only: Freedoom pistol sprites drawn over the frame from the tic a shot is detected")
        if self.args.record and self.control == "on" and self.args.baselines > 0:
            gc.declare(self.log, "baselines", "game",
                       "after the run, Doom only (no brain): never fire, always fire, a replay of the fly's ATTACK tics, "
                       f"and {self.args.baselines} random triggers matched to the fly's rate and hold lengths, on the same "
                       "Doom seeds and timeline", draws=self.args.baselines)
        if self.control != "on":
            gc.declare(self.log, "control: trigger disconnected", "game",
                       "the gf_trigger decoder still reads DNp01 but its ATTACK command is discarded (the gun never fires)")

    def brains(self):
        return {"brain": self.fb}

    # ------------------------------------------------------------------ helpers
    def _eye(self, frame):
        return self.eyes.from_pinhole(frame_to_linear(frame), self.hfov)

    def _toast(self, text, color):
        live = [z for z in self.toasts if self.t_s - z[0] < BANNER_S]
        used = {z[1] for z in live}
        free = [k for k in range(4) if k not in used]
        if not free:                                        # all four slots busy: reuse the oldest one's
            oldest = min(live, key=lambda z: z[0])
            live.remove(oldest)
            free = [oldest[1]]
        self.toasts = live + [(self.t_s, free[0], text, color)]

    def _read_rates(self):
        r = self.fb.brain.rate[0, self.read_idx].float().cpu().numpy()
        for i, k in enumerate(self.read_names):
            seg = r[self.read_seg[i]:self.read_seg[i + 1]]
            self.rates[k] = float(seg.mean()) if len(seg) else 0.0
            self.peak_rates[k] = max(self.peak_rates[k], self.rates[k])

    # ------------------------------------------------------------------ loop
    def tick(self):
        fb = self.fb
        fb.vision(self.rad)
        fb.step(gc.TICK_MS)
        self.t_s += gc.TICK_MS / 1000.0
        gf = float(self.gf_dec.value) if self.gf_dec.value is not None else 0.0
        self._read_rates()
        for key, val in (("gf", gf), ("lplc2", 0.5 * (self.rates["LPLC2 L"] + self.rates["LPLC2 R"])),
                         ("lc4", 0.5 * (self.rates["LC4 L"] + self.rates["LC4 R"]))):
            h = self.hist[key]
            h.append(val)
            del h[0]
        self.gf_max = max(self.gf_max, gf)
        self.frames_played += 1
        self.gf_above_frames += gf >= GF_THRESHOLD_HZ
        if gf >= GF_THRESHOLD_HZ > self.gf_prev:
            live, own, close = self.session.live(self.t_s), self.session.after_own_shot(self.t_s), self.world.close_recent()
            self.gstats["gf_crossings"] += 1
            self.gstats["gf_crossings_live"] += int(live)
            self.gstats["gf_crossings_close"] += int(close)
            self.gstats["gf_crossings_after_own_shot"] += int(own)
            self.log.event(self.t_s, "gf_cross", gf_hz=round(gf, 1), doom_running=live,
                           since_last_shot_s=self.session.since_last_shot(self.t_s), after_own_shot=own,
                           nearest=self.world.nearest(), closest_500ms=self.world.closest_recent())
        self.gf_prev = gf
        self.trigger.observe(gf)
        recs = self.session.frame(self.t_s, self.trigger, self.turn_input if self.human else 0.0)
        for rec in recs:
            self._show(rec)
        if recs:
            self.rad = self._eye(self.world.frame)

    def _show(self, rec):
        """Banners and the pistol animation for one tic record (display only)."""
        if "marker" in rec:
            return
        if rec["rising"]:
            self.lanes["fire"] = (self.t_s, f"GIANT FIBRE {rec['peak_hz']:.0f} Hz → FIRE", gc.RED, 58, "decoder")
        if rec["hits"] + rec["misses"]:
            self.anim.shot(self.world.tic)
        if rec["misses"]:
            self._toast("MISS", gc.TEXT)
        if rec["kills"]:
            if rec["mine"]:
                self.lanes["kill"] = (self.t_s, f"KILL #{self.session.stats['kills']}", gc.SAGE, 64, "game")
            else:
                self._toast("MONSTER INFIGHTING KILL", gc.AMBER)
        if rec["hurt"]:
            self.hurt_t = self.t_s

    def handle(self, event, canvas_pos=None):
        import pygame
        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
            keys = pygame.key.get_pressed()
            self.turn_input = turn_delta(keys[pygame.K_LEFT] or keys[pygame.K_a], keys[pygame.K_RIGHT] or keys[pygame.K_d])

    def finish(self):
        """Summary for the run log (called once after the loop), then the Doom-only checks."""
        s = self.session.summary(self.t_s)
        s.update(self.gstats)
        for kind, key in (("trigger", "triggers"), ("gf_cross", "gf_crossings")):
            by = {}
            for e in self.log.events:
                if e["kind"] == kind:
                    c = e.get("closest_500ms")
                    name = c[0] if c is not None and c[1] <= CONTEXT_UNITS else f"nothing within {CONTEXT_UNITS:g} units"
                    by[name] = by.get(name, 0) + 1
            s[f"{key}_by_closest_object_500ms"] = by
        s.update({"gf_max_hz": round(self.gf_max, 2), "gf_threshold_hz": GF_THRESHOLD_HZ,
                  "gf_frac_frames_above": round(self.gf_above_frames / max(1, self.frames_played), 4),
                  "peak_rates_hz": {k: round(v, 2) for k, v in self.peak_rates.items()},
                  "close_units": CONTEXT_UNITS, "own_shot_window_s": list(OWN_SHOT_WINDOW_S), "hfov_deg": self.hfov,
                  "control": self.control, "gun_light": bool(self.args.gun_light), "brain_s": round(self.t_s, 3)})
        self.world.close()
        a = self.args
        t0 = time.time()
        try:
            s["schedules_no_fire"] = schedule_fingerprints(a.seed, [e["episode"] for e in s["episodes"]],
                                                           scenario=a.scenario, skill=a.skill)
        except Exception as exc:                                    # a check, not the run: never lose the log over it
            s["schedules_no_fire"] = f"failed: {exc!r}"
        if a.record and self.control == "on" and a.baselines > 0:
            try:
                s["baselines"] = run_baselines(a.seed, round(self.t_s * 1000 / gc.TICK_MS), s, draws=a.baselines,
                                               scenario=a.scenario, skill=a.skill, gun_light=a.gun_light)
            except Exception as exc:
                s["baselines"] = f"failed: {exc!r}"
        print(f"doom: Doom-only checks in {time.time() - t0:.1f} s", flush=True)
        self.log.summary = s
        return s

    # ------------------------------------------------------------------ drawing
    def draw(self, surface):
        import pygame
        if surface.get_size() != gc.DEFAULT_SIZE:                   # the layout is in 1920x1080 canvas pixels
            if self._full is None:
                self._full = pygame.Surface(gc.DEFAULT_SIZE)
            self._draw(self._full)
            pygame.transform.smoothscale(self._full, surface.get_size(), surface)
            return
        self._draw(surface)

    def _draw(self, surface):
        import pygame
        h = self.hud
        surface.fill(gc.BG)
        sub = ("CONTROL RUN  ·  trigger disconnected, the gun never fires" if self.control != "on"
               else "yes, it runs Doom  ·  the giant fibre pulls the trigger")
        h.header(surface, "doom", sub, gc.model_line(self.fb))
        view = pygame.Rect(12, 68, 1312, 962)                   # 60.9 % of the canvas, between header and footer
        self._draw_doom(surface, view)
        x0 = view.right + 16
        w = surface.get_width() - x0 - 16
        y = view.y
        # what the fly sees
        r = h.panel(surface, (x0, y, w, 300), "what the fly sees")
        h.text(surface, f"Doom frame on a {self.hfov:.0f}° flat screen", (r.right, y + 11), 14, gc.DIM,
               anchor="topright")
        h.mosaic(surface, (r.x, r.y, r.w, r.h - 2), self.eyes, gc.eye_colors(self.rad))
        y += 300 + 10
        # giant fibre
        r = h.panel(surface, (x0, y, w, 262), "giant fibre  DNp01", "connectome")
        gf = self.hist["gf"][-1]
        col = gc.RED if gf >= GF_THRESHOLD_HZ else gc.SAGE
        h.text(surface, f"{gf:5.1f}", (r.x + 2, r.y - 4), 60, col, bold=True, display=True)
        h.text(surface, "Hz", (r.x + 176, r.y + 30), 22, gc.MUTED)
        h.text(surface, f"threshold {GF_THRESHOLD_HZ:.0f} Hz", (r.right, r.y + 2), 17, gc.RED, anchor="topright")
        h.text(surface, f"jump MN TTMn {self.rates['TTMn']:4.0f} Hz", (r.right, r.y + 26), 15, gc.MUTED, anchor="topright")
        tr = pygame.Rect(r.x, r.y + 68, r.w, r.h - 86)
        h.trace(surface, tr, self.hist["gf"], 0, 100, gc.SAGE, threshold=GF_THRESHOLD_HZ)
        _overdraw_above(surface, tr, self.hist["gf"], 0, 100, GF_THRESHOLD_HZ, gc.RED)
        h.text(surface, "last 4 s", (tr.x + 6, tr.bottom + 2), 13, gc.DIM)
        y += 262 + 10
        # LPLC2 / LC4
        r = h.panel(surface, (x0, y, w, 170), "LPLC2 / LC4  mean rate", "connectome")
        bw = (r.w - 16) // 2
        for i, t in enumerate(("LPLC2", "LC4")):
            for j, s in enumerate("LR"):
                v = self.rates[f"{t} {s}"]
                h.bar(surface, (r.x + j * (bw + 16), r.y + i * 64, bw, 50), v, 20, f"{t} {'left' if s == 'L' else 'right'}",
                      gc.TEAL if t == "LPLC2" else gc.LILAC)
        y += 170 + 10
        # trigger + score
        r = h.panel(surface, (x0, y, w, view.bottom - y), "trigger", "decoder")
        live = self.session.live(self.t_s)
        lit = live and self.session.fire_now
        if self.control != "on":
            state, scol = "DISCONNECTED", gc.MUTED
        elif lit:
            state, scol = "FIRE", gc.RED
        elif not live:
            state, scol = "WAIT", gc.DIM
        else:
            state, scol = "armed", gc.MUTED
        pygame.draw.circle(surface, gc.RED if lit else gc.INSET, (r.x + 26, r.y + 24), 22)
        pygame.draw.circle(surface, gc.RED if live and self.control == "on" else gc.LINE, (r.x + 26, r.y + 24), 22, 2)
        h.text(surface, state, (r.x + 60, r.y + 4), 30, scol, bold=True, display=True)
        note = (f"ATTACK while GF ≥ {GF_THRESHOLD_HZ:.0f} Hz" if live or self.control != "on"
                else "not live while Doom waits")
        h.text(surface, note, (r.x + 60, r.y + 40), 15, gc.TEAL)
        sy = r.y + 70
        pygame.draw.line(surface, gc.LINE, (r.x, sy - 8), (r.right, sy - 8))
        h.text(surface, "SCORE", (r.x, sy + 6), 14, gc.MUTED, bold=True)
        h.chip(surface, "game", (r.x, sy + 30), 11)
        st = self.session.stats
        cells = (("KILLS", st["kills"], gc.SAGE), ("SHOTS", st["hits"] + st["misses"], gc.TEXT),
                 ("MISSES", st["misses"], gc.MUTED), ("DEATHS", st["deaths"], gc.RED),
                 ("HEALTH", max(0, int(round(self.session.health))), _health_color(self.session.health)))
        cx = r.x + 84
        cw = (r.right - cx) // len(cells)
        for i, (lab, val, c) in enumerate(cells):
            h.text(surface, str(val), (cx + i * cw + cw // 2, sy), 44, c, bold=True, display=True, anchor="midtop")
            h.text(surface, lab, (cx + i * cw + cw // 2, sy + 50), 14, gc.DIM, anchor="midtop")
        dec = f"DECODER gf_trigger: ATTACK next tic if DNp01 mean ≥ {GF_THRESHOLD_HZ:g} Hz"
        if self.control != "on":
            dec += " (disconnected: control run)"
        h.footer(surface, [dec, f"GAME: Freedoom {self.args.scenario}, skill {self.args.skill}, "
                                f"{'vanilla muzzle light' if self.args.gun_light else 'no muzzle light'}, 35 tics/brain-s, "
                                f"{'human turns' if self.human else 'fixed view'}, {self.hfov:.0f}° screen"])

    def _draw_doom(self, surface, view):
        import pygame
        fr = self.world.frame
        img = pygame.image.frombuffer(np.ascontiguousarray(fr).tobytes(), (fr.shape[1], fr.shape[0]), "RGB")
        img = pygame.transform.scale(img, view.size)
        if not self.args.weapon_for_fly and self.sprites:
            name, flash = self.anim.frame(self.world.tic)
            sx, sy = view.w / 320.0, view.h / 200.0
            for nm in ([name, "PISFA0"] if flash else [name]):
                spr = self.sprites.get(nm)
                if spr is None:
                    continue
                s_img, left, top = spr
                s2 = pygame.transform.scale(s_img, (round(s_img.get_width() * sx), round(s_img.get_height() * sy)))
                img.blit(s2, (round((1 - left) * sx), round((32 - top) * sy)))
        if self.session.frozen_until is not None and not self.session.respawned:    # death freeze: tint red
            tint = pygame.Surface(view.size, pygame.SRCALPHA)
            tint.fill((140, 0, 0, 90))
            img.blit(tint, (0, 0))
        dt_h = self.t_s - self.hurt_t
        if dt_h < 0.4:
            a = int(160 * (1 - dt_h / 0.4))
            vign = pygame.Surface(view.size, pygame.SRCALPHA)
            for k in range(6):
                pygame.draw.rect(vign, (220, 20, 10, max(0, a - 25 * k)), vign.get_rect().inflate(-k * 16, -k * 16), 8)
            img.blit(vign, (0, 0))
        surface.blit(img, view.topleft)
        pygame.draw.rect(surface, gc.LINE, view, 1)
        # corner labels
        h = self.hud
        box = pygame.Surface((520, 34), pygame.SRCALPHA)
        box.fill((9, 12, 14, 170))
        surface.blit(box, (view.x + 10, view.y + 10))
        c = h.chip(surface, "game", (view.x + 18, view.y + 16), 12)
        h.text(surface, f"FREEDOOM · ViZDoom {self.args.scenario} · skill {self.args.skill}", (c.right + 10, view.y + 17),
               15, gc.TEXT)
        tb = pygame.Surface((260, 34), pygame.SRCALPHA)
        tb.fill((9, 12, 14, 170))
        surface.blit(tb, (view.right - 270, view.y + 10))
        h.text(surface, f"brain t {self.t_s:6.2f} s  ep {self.world.episode}", (view.right - 20, view.y + 17), 17, gc.TEXT,
               anchor="topright")
        if self.control != "on":
            cb = pygame.Surface((400, 34), pygame.SRCALPHA)
            cb.fill((9, 12, 14, 190))
            surface.blit(cb, (view.x + 10, view.y + 48))
            h.chip(surface, "game", (view.x + 18, view.y + 54), 12)
            h.text(surface, "CONTROL: trigger disconnected", (view.x + 84, view.y + 55), 17, gc.AMBER, bold=True)
        self._draw_banners(surface, view)

    def _banner_surface(self, text, color, size, chip=None, pad=18):
        """A banner on a translucent box, with its provenance chip as a tab above the box's left edge."""
        import pygame
        h = self.hud
        img = _banner_text(h.font(size, True, True), text, color)
        tab = 24 if chip else 0
        bw, bh = img.get_width() + 2 * pad, img.get_height() + pad
        s = pygame.Surface((bw, bh + tab), pygame.SRCALPHA)
        pygame.draw.rect(s, (9, 12, 14, 184), (0, tab, bw, bh))
        s.blit(img, img.get_rect(center=(bw // 2, tab + bh // 2)))
        if chip:
            h.chip(s, chip, (0, 0), 11)
        return s

    def _draw_banners(self, surface, view):
        """Fixed slots, no reflow: the decoder's FIRE and the gun's KILL in the upper centre, the Doom state (YOU DIED,
        GET READY) on the floor below the horizon, GAME notes (misses, infighting) as small toasts in the lower left."""
        now = self.t_s
        slots = {"fire": view.y + 96, "kill": view.y + 206}
        for lane in list(self.lanes):
            t0, txt, col, size, chip = self.lanes[lane]
            age = now - t0
            if not 0 <= age < BANNER_S:
                del self.lanes[lane]
                continue
            s = self._banner_surface(txt, col, size, chip)
            s.set_alpha(int(255 * (1 - (age / BANNER_S) ** 2)))
            surface.blit(s, s.get_rect(midtop=(view.centerx, slots[lane])))
        ses = self.session
        state = None
        if now < WARM_S - 1e-9 or (ses.frozen_until is not None and ses.respawned):
            state = ("GET READY", gc.MUTED, 64)
        elif ses.frozen_until is not None:
            state = ("YOU DIED", gc.RED, 96)
        if state:                                           # on the floor, below the corridor's horizon and its monsters
            s = self._banner_surface(*state, "game")
            surface.blit(s, s.get_rect(center=(view.centerx, view.y + 0.70 * view.h)))
        self.toasts = [z for z in self.toasts if 0 <= now - z[0] < BANNER_S]
        for t0, slot, txt, col in self.toasts:
            s = self._banner_surface(txt, col, 28, "game", pad=12)
            s.set_alpha(int(255 * (1 - ((now - t0) / BANNER_S) ** 2)))
            surface.blit(s, s.get_rect(bottomleft=(view.x + 16, view.bottom - 16 - slot * 78)))


def _health_color(hp):
    return gc.SAGE if hp > 50 else (gc.AMBER if hp > 25 else gc.RED)


def _banner_text(font, text, color):
    """Render a banner line; a '→' is drawn as an arrow shape (display fonts often lack the glyph)."""
    import pygame
    parts = text.split("→")
    imgs = [font.render(p.strip(), True, color) for p in parts]
    if len(imgs) == 1:
        return imgs[0]
    hgt = max(i.get_height() for i in imgs)
    aw = int(hgt * 0.9)
    out = pygame.Surface((sum(i.get_width() for i in imgs) + aw * (len(imgs) - 1), hgt), pygame.SRCALPHA)
    x = 0
    for k, img in enumerate(imgs):
        out.blit(img, (x, (hgt - img.get_height()) // 2))
        x += img.get_width()
        if k < len(imgs) - 1:
            cy, th = hgt // 2, max(3, hgt // 11)
            pygame.draw.rect(out, color, (x + aw * 0.18, cy - th // 2, aw * 0.45, th))
            pygame.draw.polygon(out, color, [(x + aw * 0.55, cy - hgt * 0.22), (x + aw * 0.86, cy), (x + aw * 0.55, cy + hgt * 0.22)])
            x += aw
    return out


def _overdraw_above(surface, rect, values, vmin, vmax, threshold, color):
    """Redraw the parts of a Hud.trace line at or above `threshold` in `color`."""
    import pygame
    v = np.asarray(values, np.float64)
    if len(v) < 2 or not (v >= threshold).any():
        return
    x = rect.x + np.linspace(0, rect.w - 1, len(v))
    y = rect.bottom - 1 - np.clip((v - vmin) / (vmax - vmin), 0, 1) * (rect.h - 2)
    ythr = rect.bottom - 1 - (threshold - vmin) / (vmax - vmin) * (rect.h - 2)
    above = v >= threshold
    i = 0
    while i < len(v):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(v) and above[j + 1]:
            j += 1
        pts = [(x[i], ythr)] + [(x[k], y[k]) for k in range(i, j + 1)] + [(x[j], ythr)]
        pygame.draw.polygon(surface, tuple(int(c * 0.35 + b * 0.65) for c, b in zip(color, gc.INSET)), pts)
        if j > i:
            pygame.draw.lines(surface, color, False, [(x[k], y[k]) for k in range(i, j + 1)], 2)
        i = j + 1


def _mean_rate(dt_ms, inputs):
    return float(inputs["gf"].float().mean().item())


def _load_pistol_sprites(world):
    """{name: (pygame RGBA surface, leftoffset, topoffset)} for the Freedoom pistol, or {} if the WAD is unreadable."""
    import pygame
    try:
        wad = Path(world.vzd.__file__).parent / world.iwad
        lumps = read_wad(wad)
        pal = np.frombuffer(lumps["PLAYPAL"][:768], np.uint8).reshape(256, 3)
        out = {}
        for nm in ("PISGA0", "PISGB0", "PISGC0", "PISFA0"):
            rgba, left, top = decode_patch(lumps[nm], pal)
            surf = pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), rgba.shape[1::-1], "RGBA").copy()
            out[nm] = (surf, left, top)
        return out
    except Exception as exc:                                       # cosmetic only: never fail a run over it
        print(f"doom: no weapon overlay ({exc})", file=sys.stderr)
        return {}


def main(argv=None):
    ap = gc.standard_args(__doc__.splitlines()[0], seconds=30.0)
    ap.add_argument("--hfov", type=float, default=HFOV_DEG, help="the Doom frame's horizontal extent in the fly's view, deg")
    ap.add_argument("--scenario", default=SCENARIO, help="ViZDoom scenario (.cfg name)")
    ap.add_argument("--skill", type=int, default=SKILL, help="Doom skill 1-5")
    ap.add_argument("--control", default="on", choices=("on", "off"),
                    help="'off' disconnects the trigger (the decoder still reads the GF; the gun never fires)")
    ap.add_argument("--baselines", type=int, default=BASELINE_DRAWS, metavar="N",
                    help="after a recorded run, Doom-only baselines with N random-trigger draws (0: none)")
    ap.add_argument("--weapon-for-fly", action="store_true",
                    help="ablation: render the weapon sprite into the fly's frame too (default: the fly sees the 3D view only)")
    ap.add_argument("--gun-light", action="store_true",
                    help="ablation: Doom's vanilla pistol, whose muzzle light brightens the whole 3D view on every shot")
    args = ap.parse_args(argv)
    try:
        import vizdoom  # noqa: F401
    except ImportError:
        raise SystemExit("doom needs ViZDoom: pip install --no-deps vizdoom==1.3.1 "
                         "(a plain `pip install vizdoom` also pulls pygame-ce, which replaces the pygame package)")
    t0 = time.time()
    game = DoomFly(args)
    print(f"doom: brain + Doom ready in {time.time() - t0:.1f} s (seed {args.seed}, hfov {args.hfov:g}, control {args.control})",
          flush=True)
    _run_with_summary(game, args)


def _run_with_summary(game, args):
    """common.run, with the game's summary filled in before the run log is written."""
    orig_save = game.log.save

    def save(path, fbs=None):
        game.finish()
        return orig_save(path, fbs)
    game.log.save = save
    try:
        path = gc.run(game, args)
    finally:
        if not game.log.summary:
            game.finish()
    s = game.log.summary
    print("doom: kills {kills}, shots {shots} (hits {hits}, misses {misses}), deaths {deaths}, triggers {triggers} "
          "({triggers_close} with an object within {close_units:g} units, {triggers_after_own_shot} 0.1-0.7 s after a shot), "
          "GF crossings {gf_crossings}, GF max {gf_max_hz} Hz".format(**s))
    b = s.get("baselines")
    if isinstance(b, dict):
        print(f"doom: baselines never {b['never']['kills']} kills / {b['never']['deaths']} deaths, always "
              f"{b['always']['kills']} / {b['always']['deaths']}, replay matches fly: {b['replay_matches_fly']}", flush=True)
        if "random" in b:
            print(f"doom: random triggers {b['random']['rank_of_fly']}", flush=True)
    return path


if __name__ == "__main__":
    main()
