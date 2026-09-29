# doom: yes, it runs Doom

**Yes, it runs Doom: 167,106 neurons, one button, and the gun fires when the fly's giant fibre does, usually half a
second after an imp's fireball bursts in its face.**

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the fly) | MaleCNS v1.0 as shipped: 167,106 neurons, preset `raw`, nothing instrumented, trained or tuned for this game. Everything between the light reaching the eye and the giant fibre's rate is the model: photoreceptors, optic lobe, LPLC2 / LC4, DNp01. That includes the shipped stop-gap gain on LC4 / LPLC2 -> DNp01 that the README's ledger declares (x3, x6 with the visual-projection gain), which was set for the loom escape margin, not for Doom. The HUD's GF, LPLC2, LC4 and TTMn numbers are its rates. |
| **DECODER** (written for the game, declared) | One rule, `gf_trigger`: hold ATTACK for the next Doom tic when the mean DNp01 rate reached 33 Hz during this tic. 33 Hz is the shipped body model's takeoff threshold, not a tuned value. The decoder only reads; it writes nothing into the brain. |
| **GAME** (not the fly) | Doom itself (Freedoom on ViZDoom, `defend_the_line`, skill 3): the monsters, their fireballs, the pistol (its muzzle light removed), the aim (fixed down the corridor), the score, and the timing of deaths and GET READY. The 160° screen placement, which magnifies the centre of Doom's view 5.7x so that the giant fibre can reach threshold. The pistol sprite and the banners in the human view. The baselines. |
| **not claimed** | The fly does not aim, turn, move, dodge or hunt. The trigger's timing is not shown to beat random timing (below). What drives the giant fibre here is mostly a sudden, large brightness change (a fireball bursting), not specifically a looming monster. |

## Design

The shipped fly (MaleCNS v1.0, `preset="raw"`, no instruments) watches Freedoom through its compound eye and holds
one button: ATTACK. The ViZDoom scenario `defend_the_line` sends three demons down a corridor at the player while
three imps throw fireballs from the far wall. A monster that dies is removed at once and a tougher one comes back at
the far end 90 tics later. Every Doom tic's 3D view is shown to the fly as a flat screen. The gun fires when the
fly's giant fibre (DNp01, the escape command neuron) crosses the body model's takeoff threshold. The fly does not aim
or move, and in the recordings the view never turns. Headline-safe wording: "the gun fires when the fly's giant fibre
does". Do not say the fly defends itself, dodges, or survives longer (see "Against chance" below).

| what | kind | law and parameters |
|---|---|---|
| the brain | CONNECTOME | `FlyBrain(seed=...)`, preset `raw`, nothing instrumented; one read-only decoder attached |
| the fly's eyes | GAME | Each tic's 640x480 frame, **3D view only** (no weapon sprite, no HUD). sRGB -> linear, box-filtered 2x2 to 320x240, handed to `Eyes.from_pinhole` with `hfov_deg = 160`: a flat screen spanning 160 deg of azimuth and 154 deg of elevation. The eye is 0.176 screen half-widths from the glass, about 3.5 cm from a 40 cm-wide monitor. The frame centre is magnified 5.7x relative to Doom's own 90 deg view. Radiance 0 outside the screen (a dark room). UV = 0.5 B. |
| pistol without muzzle light | GAME | Doom's pistol flash calls `A_Light1`, which brightens every sector of the rendered 3D view by about 10 % for 7 tics per shot. By itself, that drove the giant fibre over threshold half a second later, so the gun re-triggered itself (below). The game loads a small PWAD, built at start-up from text in `games/doom.py`: a DECORATE player class `FlyversePlayer` that starts with `FlyversePistol`, which is Freedoom's `Pistol` with that one action removed. Bullets, damage, refire timing and the flash sprite are unchanged. The fly still sees what the shot does in the world: puffs, and a hit monster vanishing. `--gun-light` restores the vanilla pistol (ablation). |
| `gf_trigger` | DECODER | An attached `games.common.ReadDecoder` (`kind="decoder"`, reads `rate_hz` of DNp01 L and R, writes nothing): **ATTACK is held for the next Doom tic when the mean DNp01 rate reached >= 33 Hz at any 10 ms brain frame during the current tic.** 33 Hz is `flyverse.body.Flight.gf_hz`, the shipped escape-takeoff threshold, asserted at start-up. Nothing in this decoder was tuned. Latency: one brain frame (the decoder sees the previous frame) plus up to one tic (28.6 ms). The pistol fires about 4 tics after ATTACK goes down and refires every 14 tics while it is held. |
| Doom | GAME | Freedoom (BSD) assets on the ViZDoom 1.3.1 engine, `defend_the_line.cfg` at its own skill 3, pistol (one shot kills a first-wave monster; replacements are tougher). 35 Doom tics per brain second (one Doom second = one brain second). For the first 1.0 s the fly sees the first frame ("GET READY") and Doom waits. After a death the last frame holds 0.6 s ("YOU DIED"), then the next episode's first frame shows for 0.6 s ("GET READY") before Doom resumes. The trigger is not live while Doom waits, so a scene cut cannot fire the gun. A GF that is still above threshold when Doom resumes counts as a new trigger. Episode *e* of run seed *s* uses Doom seed 1000 *s* + *e*. |
| view direction | GAME | Recordings: fixed down the corridor; nothing turns it. Interactive: a human turns it with LEFT / A and RIGHT / D (3 deg per tic). |
| score | GAME | Doom's own counters. A shot is a player hit (a `HITCOUNT` step) or a new `BulletPuff` object (a miss). `KILLCOUNT` also counts monsters killed by other monsters (imp fireballs kill demons), so a kill is credited to the gun only when a player hit landed in the same tic or up to 3 tics before. Otherwise it is logged as infighting. The rule can also credit the gun with a monster that another monster killed within 3 tics of a hit (seed 1, 13.54 s, below). |
| baselines | GAME | After each recorded run with the trigger connected, and with no brain, Doom is replayed on the same Doom seeds and the same brain-frame timeline (GET READY, freezes, respawns), scored by the same code (`DoomSession`). The arms: **never fire**; **always fire**; **50 random triggers** that start a hold with the fly's own per-tic rate and draw hold lengths from the fly's own holds (numpy `default_rng([seed, draw])`); and a **replay** of the fly's exact ATTACK tics. The replay is a determinism check: it must reproduce the fly's hits, misses, kills, deaths and tic count, and did in all three connected dev runs below. Written to the run log's `summary.baselines`, with `rank_of_fly`. |
| schedule fingerprint | GAME | For each episode of a run, Doom is played from that episode's seed with nobody firing, and the log's `summary.schedules_no_fire` records the tic of death and of the first three hurts. Equal fingerprints mean the same monster schedule. |
| pistol on screen | GAME (display) | The Freedoom pistol sprites (PISG A/B/C and the flash PISF A), drawn over the human view from the tic a shot is detected. The fly never sees them. |
| HUD | mixed | CONNECTOME: the DNp01 mean rate (the decoder's input) against the 33 Hz line, LPLC2 and LC4 mean rates per side, TTMn (the jump motor neuron). DECODER: the trigger lamp (FIRE / armed / WAIT while Doom waits / DISCONNECTED in the control) and the banner "GIANT FIBRE *n* Hz → FIRE". GAME: the score row (kills, shots, misses, deaths, health), KILL #*n*, YOU DIED / GET READY, and small lower-left notes for misses and infighting kills. Banners keep fixed slots and fade over 0.8 s. |
| control run | GAME | `--control off`: same seed. The decoder still reads DNp01 but its ATTACK is discarded, so the gun never fires. The log counts `decoder_commands`, the times the decoder would have fired. |

Each run log records every GF upward crossing and every trigger with the following fields:
* `since_last_shot_s` and `after_own_shot` (true when the player's last shot was 0.1-0.7 s earlier);
* `nearest`, the labelled objects on screen at that moment;
* `closest_500ms`, the closest labelled object of the preceding 18 tics (about 0.5 s), as name, map units, screen x,
  width in pixels and tics ago.

The summary counts triggers and crossings by that closest object (anything nearer than 128 units) and by
`after_own_shot`. These fields describe what was on screen; they do not establish the cause, and there is always
*some* object. The player is 56 units tall, so one map unit is about 3 cm.

### What was measured first (dev seeds >= 100)

Provenance. The screen-width table, the closed-loop prototype, the weapon-sprite check and the turning test below are
the builder's pre-freeze runs. Their arrays are in the build session's scratch directory, not in `out/games/doom`;
the caption skeptic recomputed every number quoted from them, and all matched. The open-loop muzzle-light table is
`out/games/doom/fix/flash_mod.log`. The honesty review's numbers were reported by that review and its outputs were not
saved, so they cannot be re-checked from files here: the eight forced shots, the 10.2 %, the synthetic brightness
steps and the presynaptic shares. The same is true of the builder's "6 of 6".

**Doom's own geometry does not reach the giant fibre; the screen had to come closer.** The builder played one recorded
Doom episode (no firing; demons reach melee range, fireballs hit) open loop to brain seed 100 at four screen widths:

| screen width (hfov) | GF max | crossings of 33 Hz | LPLC2 max (side mean) | LC4 max (side mean) |
|---|---|---|---|---|
| 90 deg (Doom's own view, true angles) | 12.6 Hz | 0 | 1.7 Hz | 0.8 Hz |
| 150 deg | 28.8 Hz | 0 | 3.5 Hz | 2.7 Hz |
| **160 deg (used)** | 53.0 Hz | 2 | 6.9 Hz | 2.8 Hz |
| 170 deg | 80.7 Hz | 9 | 10.0 Hz | 3.5 Hz |

At true angles a demon at melee range subtends about 33 x 53 deg and stops there. It never enters the explosive last
phase of a collision loom (the README's 1 m/s ball passes 90 deg). A closed-loop prototype on seeds 102 and 103 (10 s
each) crossed at 160, 165 and 170 deg alike (GF max 79-97 Hz; 3-8 crossings). 160 deg is the least magnified width
that crossed, and it keeps the corridor recognisable in the fly's-eye mosaic.

**The player's own shot reached the fly's eye twice over. Both paths are now removed.**
* *The weapon sprite.* It sits at the magnified centre-bottom of the screen. With it in the fly's frame (170 deg, seed
  100, 10 s, muzzle light still on), 5 of 10 GF crossings had nothing labelled within 150 units. It has been kept out
  of the fly's frame since.
* *Doom's muzzle light.* The honesty review showed that `A_Light1` brightens the whole 3D view by 10.2 % for 7 tics,
  even with the sprite off, and that this alone drives the GF over threshold. The review's eight forced shots (misses
  and far kills alike, brain seeds 110 and 111) were followed about 0.5 s later by 31-44 Hz; 7 of 8 crossed 33 Hz. A
  synthetic uniform +10 % brightening gave 45 Hz. In the builder's surviving pre-freeze logs
  (`out/games/doom/pre_fix_builder/`), a trigger followed one of the gun's own hits by 0.29-0.34 s three times: once in
  `dev102.json`, and twice in `dev_weapon101.json`, which also had the weapon sprite in the fly's frame. The builder's
  finding, "a shot that killed a far demon at the centre is followed by a 46-57 Hz GF peak in 6 of 6 cases", is
  explained by this light, not by the kill.

With the `FlyversePistol` the light is gone. Doom seed 100001, 160 deg, open loop; every condition was replayed from
the same brain snapshot (`out/games/doom/fix/flash_mod.py`, `flash_mod.log`):

| forced shot | frame mean luminance | GF max in the 0.8 s after the shot, brain seed 110 / 111 |
|---|---|---|
| none (whole 2.9 s) | - | 9.5 / 9.5 Hz |
| miss at tic 20, vanilla pistol | +10.5 % for 7 tics | **39.8 / 40.7 Hz** at +470-480 ms (7 / 8 frames >= 33 Hz) |
| miss at tic 20, no muzzle light | +0.0 % | 9.5 / 9.5 Hz (the no-shot level) |
| far kill at tic 35, vanilla pistol | +10.1 % | **41.3 / 37.0 Hz** at +470-480 ms |
| far kill at tic 35, no muzzle light | +0.0 % | 4.9 / 0.3 Hz (no shot, same window: 4.8 / 0.3) |
| miss at tic 8 / far kill at tic 50, no muzzle light | +0.0 % | 9.6 / 9.6 and 5.8 / 8.7 Hz (no shot: 9.5 and 4.8) |

In closed loop the light's effect on one 30 s run is small. Seed 101 with `--gun-light` (frozen code,
`dev_gunlight.json`): 7 triggers and 10 shots. A GF crossing 0.26 s after a kill shot stretched that hold from 14 to 17
tics, into an extra refire (a miss). Without the light, the same seed gave 7 triggers and 9 shots. A pre-freeze repeat
of the ablation gave 8 triggers, one of them 0.29 s after a kill shot; its log was overwritten. The CUDA brain is not
bit-reproducible, so the open-loop table is the clean comparison.

**Turning drives the GF too.** Imposed sinusoidal turning with the trigger disconnected (170 deg, seed 102):
* a peak of 1 deg/tic (35 deg/s) kept the GF under 21 Hz;
* 3 deg/tic (105 deg/s) crossed 33 Hz three times in 3 s, with nothing closer than 169 units.

The recordings never turn. A human turning in interactive mode can fire the gun.

**What the trigger responds to: a sudden brightness change at the magnified screen, here a fireball bursting.** In
the honesty review (brain seed 112, 160 deg), a 0.2 s change of whole-screen brightness drove the GF over threshold
with no loom at all:
* +10 %: 45.1 Hz; -10 %: 50.8 Hz. The same frames with no change: 9.5 Hz.
* Across DNp01's 1,261 presynaptic cells, LPLC2 carried 80-90 % of the net extra drive and LC4 14-16 %.

So the route is LPLC2 / LC4 -> giant fibre, but the stimulus need not be a loom. In the frozen-code dev runs, 12 of
13 triggers came **0.40-0.57 s after an imp fireball burst**:
* 9 burst on the player, 0.40-0.55 s after a `hurt` event;
* 3 burst on a demon about 86 units ahead, 0.57 s after that infighting kill;
* the 13th (seed 103, 29.52 s) had a demon 55 units away as its closest object, 0.69 s after an infighting kill.

The closest object in the preceding 0.5 s was a fireball 23-86 units away for all 7 triggers on seed 101 and 5 of 6
on seed 103. So the trigger comes after the damage is done. The pistol then fires straight down the centre line, and
it kills when a charging demon happens to be in that line. Not every fireball burst crosses the threshold: seed 101
had 12 hurts and 7 triggers.

**Doom seeds and schedules.** ViZDoom is deterministic for a given Doom seed and action sequence, and in
`defend_the_line` many Doom seeds replay one monster schedule. With nobody firing, Doom seeds 101001, 101002, 102001,
102002, 104001, 104002 and 103002 all end at episode tic 248 with hurts at tics 128, 156 and 224. The sources are the
`schedules_no_fire` field of `dev.json` and `dev103.json` for 101001, 101002 and 103002, and the honesty review's
`review_honesty/doom_seeds.log` for 102001-104002; that log counts tics from 0 (247, and 127, 155, 223). That is why
the builder's first episodes on brain seeds 101, 102 and 104 looked alike: the Doom was identical, and different
brains answered the same strong stimuli alike. Seed 103's first episode (Doom seed
103001: death at tic 413 with nobody firing, first hurt to health 92) is a different Doom game from its first tic.
Its later first trigger (8.31 s) comes from that schedule, not from a brain that "did not cross". The recorded seeds
0-2 start on schedules that differ from the dev runs' and from each other; each run log lists its own.

### 30 s dev runs, frozen code (source hash `games/doom.py 3fa9241469623dfc`)

| run | triggers (held tics) | shots (hits / misses) | kills by the gun | infighting kills | deaths | completed episodes lasted |
|---|---|---|---|---|---|---|
| **seed 101, trigger connected** (`dev.json`) | 7 (100) | 9 (5 / 4) | 5 | 5 | 2 | 10.77, 10.78 s |
| seed 101, control, `--control off` (`dev_control.json`) | 0; the decoder would have fired 4 times | 0 | 0 | 8 | 3 | 7.08, 7.09, 5.20 s |
| seed 101, vanilla muzzle light, `--gun-light` (`dev_gunlight.json`) | 7 (97) | 10 (5 / 5) | 5 | 5 | 2 | 10.77, 10.78 s |
| **seed 103, trigger connected** (`dev103.json`) | 6 (77) | 8 (4 / 4) | 4 | 7 | 2 | 12.05, 10.78 s |

On seed 101, 0 of 7 triggers and 1 of 9 GF crossings came 0.1-0.7 s after a shot. That crossing, 0.17 s after a
kill with blood at 59 units, fell inside a hold. On seed 103, 1 of 6 triggers did: 0.35 s after a kill shot, with a
fireball burst 28 units away 0.43 s earlier (and a hurt 0.46 s earlier). 2 of its 7 GF crossings did.

With the trigger disconnected, the GF still crossed 33 Hz 5 times on seed 101 (closest object: fireball 3, demon 2):
the crossings are the fly's; the gun only listens. Seed 101 connected: GF max 94.7 Hz, LPLC2 side means up to
13.0 Hz, LC4 up to 3.6 Hz, TTMn up to 60 Hz.

**Against chance** (Doom only, same Doom seeds and timeline, `summary.baselines` of each run):

| seed | never fire | always fire | 50 random triggers at the fly's rate: kills / deaths (median, range) | draws with at least the fly's kills (more) | draws with at most the fly's deaths (fewer) | the fly |
|---|---|---|---|---|---|---|
| 101 | 0 kills, 3 deaths (7.08, 7.09, 5.20 s) | 20 kills from 70 shots, 1 death (15.54 s) | 5.5 (0-9) / 2 (1-3) | 33 (25) | 39 (15) | 5 kills, 2 deaths |
| 103 | 0 kills, 3 deaths (11.79, 7.09, 5.60 s) | 18 kills from 69 shots, 1 death (15.97 s) | 5.5 (2-9) / 2 (1-3) | 42 (34) | 45 (18) | 4 kills, 2 deaths |

The control run and the never-fire baseline agree to the hundredth of a second: with the gun disconnected, nothing
the brain does reaches Doom. The fly's kills and deaths sit inside the random-timing range on both dev seeds. On seed 101, 25 of 50
random draws killed more and 15 died less; on seed 103, 34 killed more. **The GF trigger's timing is not shown to
beat chance.** Holding the trigger down beats both. The disconnected control shows only that the GF crosses whether
or not the gun listens. Keep "survives longer" out of the trailer and the thread.

### Limits

* The fly does not aim, turn or hunt. The gun fires down a fixed line, and a demon at melee range is hit even 15 deg
  off centre (Doom's demon radius is 30 units), so kills say little about aim. Every fireball that reaches the
  player hurts; firing more kills more (always-fire above).
* The giant fibre reaches threshold only because the screen is close (160 deg). At Doom's own 90 deg it stayed under
  13 Hz.
* The effective stimulus is a large, sudden brightness change at the magnified screen, not specifically a loom (see
  above). Keep "loom" out of the banners and the headline. The HUD shows LPLC2 and LC4 by name.
* The shot's consequences in the world remain visible to the fly. A monster killed at close range vanishes at once
  (the scenario removes it), and blood sprays. A crossing within 0.1-0.7 s of a shot is flagged, not removed.
* LC4 stays low throughout. Its side-mean peaks are at most 4.05 Hz in the dev runs, 4.49 Hz in the recordings and
  5.14 Hz in control seed 0. LPLC2 reaches about 13 Hz (14.5 Hz in the recordings).
* The CUDA brain is not bit-reproducible. Doom is: each run's `replay` baseline reproduces its Doom outcome exactly.
* Dependency: `pip install --no-deps vizdoom==1.3.1`. A plain `pip install vizdoom` also pulls pygame-ce, which
  replaces the `pygame` package in place.
* Wall time: 120-290 s per 30 s clip on the shared RTX 4090 in dev (4-10x real time). The recordings took 295-326 s
  (10-11x, two runs at once), plus 11-69 s for the Doom-only baselines.

    python games/doom.py --seed 0 --seconds 30 --record out/games/doom/seed0.mp4
    python games/doom.py --seed 0 --seconds 30 --control off --record out/games/doom/control_seed0.mp4

## Clips

Recorded on 2026-09-28 on this desktop: `cuda`, NVIDIA GeForce RTX 4090, torch 2.10.0+cu128, Python 3.13.2, ViZDoom
1.3.1. All six run logs carry the same provenance:
* commit `959f2e927b1c2806bc526fd5fc51cb3f5cc3e4b1` with `dirty: true` (games/ is not committed yet);
* `games/doom.py` sha256 `3fa9241469623dfc` and `games/common.py` `b110142562a927a1` (first 16 hex). These are the
  frozen code of the dev runs above.
* Preset `raw`, no instruments, and one attached module: `gf_trigger`, kind `decoder`.

Each command ran exactly once, from the repo root, as `PYTHONIOENCODING=utf-8 <miniconda>/python.exe`
followed by the arguments below. All six clips are kept. No run crashed, and `games/doom.py` was not changed. Each
clip is 30.0 s of brain time (1,500 frames, 1920x1080, 50 fps). Each connected run and its control ran side by side
on the shared GPU. The wall times exclude the Doom-only checks that run afterwards (10.7, 11.7 and 68.6 s for the
connected runs' baselines; about 1 s for a control's schedule fingerprints). Doom seed = 1000 k + episode.

| clip | command arguments | started (UTC) | brain / wall |
|---|---|---|---|
| `out/games/doom/seed0.mp4` | `games/doom.py --seed 0 --seconds 30 --record out/games/doom/seed0.mp4` | 21:01:33 | 30.0 / 298.0 s |
| `out/games/doom/control_seed0.mp4` | `games/doom.py --seed 0 --seconds 30 --control off --record out/games/doom/control_seed0.mp4` | 21:01:33 | 30.0 / 297.7 s |
| `out/games/doom/seed1.mp4` | `games/doom.py --seed 1 --seconds 30 --record out/games/doom/seed1.mp4` | 21:07:15 | 30.0 / 325.5 s |
| `out/games/doom/control_seed1.mp4` | `games/doom.py --seed 1 --seconds 30 --control off --record out/games/doom/control_seed1.mp4` | 21:07:15 | 30.0 / 321.3 s |
| `out/games/doom/seed2.mp4` | `games/doom.py --seed 2 --seconds 30 --record out/games/doom/seed2.mp4` | 21:13:21 | 30.0 / 294.6 s |
| `out/games/doom/control_seed2.mp4` | `games/doom.py --seed 2 --seconds 30 --control off --record out/games/doom/control_seed2.mp4` | 21:13:21 | 30.0 / 295.0 s |

What the run logs count:

| clip | triggers (held tics) | shots (hits / misses) | kills by the gun | infighting kills | hurts | deaths | episodes lasted | GF crossings | GF max |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 7 (71) | 9 (4 / 5) | 4 | 9 | 8 | 1 | 14.59 s; 13.21 s at clip end | 9 | 97.0 Hz |
| control 0 | 0; the decoder would have fired 5 times | 0 | 0 | 8 | 11 | 2 | 11.74, 11.43 s; 3.43 s at clip end | 6 | 102.5 Hz |
| seed 1 | 11 (134) | 14 (8 / 6) | 7 | 4 | 12 | 2 | 5.42, 18.46 s; 2.72 s at clip end | 14 | 103.3 Hz |
| control 1 | 0; would have fired 9 times | 0 | 0 | 9 | 10 | 2 | 5.42, 18.72 s; 2.46 s at clip end | 14 | 86.7 Hz |
| seed 2 | 15 (162) | 18 (6 / 12) | 5 | 7 | 12 | 1 | 18.62 s; 9.18 s at clip end | 20 | 90.7 Hz |
| control 2 | 0; would have fired 8 times | 0 | 0 | 11 | 10 | 2 | 20.05, 7.09 s | 16 | 90.6 Hz |

Over the three connected runs (90 s of brain time), the counts are:
* 33 triggers, 41 shots (18 hits, 23 misses), 16 kills by the gun and 4 deaths. The 16 are credited by the 3-tic
  rule and came from 15 killing hits; one of them is most likely infighting (seed 1, 13.54 s);
* peak side-mean rates of LPLC2 10.1-14.5 Hz, LC4 2.0-4.5 Hz and TTMn 57.6-69.4 Hz;
* the GF at or above 33 Hz in 6.3 %, 12.0 % and 14.1 % of 10 ms frames.

Three of the 18 hits did not kill at once: 18.02 and 22.02 s on seed 1, and 17.65 s on seed 2. A killed monster's
replacement is tougher (Design).

**What each trigger followed.** For each trigger, the log records the closest labelled object in the preceding 0.5 s:
* 27 of 33: an imp fireball 23-28 units away, bursting on the player. These came 0.32-0.55 s after the hurt that
  burst caused, except one inside a barrage, 0.17 s after a hurt.
* 5: a fireball bursting on a demon 63-91 units ahead, 0.52-0.60 s after that infighting kill.
* 1: nothing within 128 units (seed 1, 15.60 s).

Of the 43 GF crossings, 41 had a fireball as that closest object and 2 had nothing within 128 units. Seven triggers
and 12 crossings fell 0.1-0.7 s after the fly's own shot; they are flagged, not removed. The pistol still fires only
down the centre line, and it hits whatever demon happens to be there.

**Same Doom, same moment, different brains.** In seven stretches, a connected run and its control played identical
Doom: the same Doom seed, up to the connected run's first shot of that episode, timed from the episode's start. In
five of them both brains crossed 33 Hz, and their first crossings fell within 0.00-0.09 s of each other:
* seed 0 episode 1: 7.65 s (connected; again at 7.73 s) and 7.74 s (control);
* seed 0 episode 2: 4.41 s into the episode in both (21.20 and 18.35 s);
* seed 1 episode 2: 12.16 and 12.13 s;
* seed 2 episode 1: 5.16 and 5.16 s;
* seed 2 episode 2: 5.54 and 5.55 s into the episode (26.36 and 27.80 s).

In the other two, seed 1 episodes 1 and 3, neither brain crossed. Seed 2's episode 2 uses Doom seed 2002, which plays
the common schedule of the dev runs (nobody firing: death at tic 248), and dev seed 101 crossed 5.53 s into that
schedule. What is on the screen sets the crossings; the brain seed moves them by at most 0.09 s here.

**Seed 0** (Doom seeds 1 and 2). Episode 1:
* Before any crossing, the player is hurt three times (health 56 by 7.28 s), and monsters kill each other twice.
* The GF first crosses at 7.65 s (banner: 34 Hz), 0.37 s after a fireball bursts on the player. That first hold lasts
  2 tics, and its press starts the pistol, which fires 4 tics later: it kills a demon about 400 units down the centre
  line at 7.77 s (KILL #1).
* A second crossing 0.08 s later (banner: 45 Hz, at 7.74 s) holds ATTACK for 16 tics. It does not fire the kill shot.
  It keeps ATTACK down into the pistol's refire, which misses (8.17 s).
* Two more triggers (8.71 and 10.28 s) each fire one shot, and both miss. The 8.71 s trigger came 0.54 s after the
  fly's own miss and 0.23 s after a hurt.
* At 15.34 s (47 Hz), 0.52 s after an imp fireball killed a demon 91 units ahead, the gun kills a demon about
  200 units ahead (KILL #2, 15.45 s).
* A fireball already in flight kills the player 0.14 s later (15.59 s), after 14.59 s of play.

Episode 2 (from 16.79 s):
* A trigger at 21.22 s, 0.40 s after a hurt, fires two shots; both miss.
* At 24.74 s (36 Hz), one 15-tic hold kills two demons: one about 350 units down the centre line (KILL #3,
  24.85 s), then one that had walked in from the left to about 80 units (KILL #4, 25.25 s).
* The clip ends with health 82.

Display note (the human view only): this death came 5 tics (0.14 s) after a shot. Doom's tic count stops while Doom
waits, so the pistol overlay keeps its muzzle-flash frame through YOU DIED and GET READY (15.59-16.79 s). The fly's frame never
contains the overlay, and the log is unaffected. It is cosmetic and was not fixed: the code is frozen.

**Control 0.** The Doom is identical until 7.77 s. This brain's GF first crosses at 7.74 s; the connected run's
crossed at 7.65 s. With the gun disconnected, a demon reaches melee range and fills the right half of the screen. The
GF crosses at 12.53 s (38.2 Hz; that demon, 50 units away, is the closest object) and again at 12.71 s. That the
demon drove it is not established. The demon already fills the right half at 12.30 s, when the GF reads 0.2 Hz. The
rise comes 0.51 s after an imp fireball bursts on a demon 86 units ahead (the 12.02 s infighting kill), which is the
latency of the fireball-driven triggers. The two candidate stimuli are not separated. The player dies at 12.74 s, and
again at 25.37 s. Two of this run's 6 crossings had a demon as the closest object; the rest had a fireball.

**Seed 1** (Doom seeds 1001-1003). Episode 1 is a non-event for the trigger:
* A demon at melee range fills the centre of the screen and bites the player three times (health 100 -> 76 -> 40 ->
  32, 4.31-5.74 s).
* The GF never crosses. Sampled every 0.1 s from 4.2 to 6.9 s, the on-screen readout stays between 0.1 and 12.1 Hz.
  Nothing fires.
* The player dies at 6.42 s, after 5.42 s, as with nobody firing.

Episode 2 (from 7.62 s) has all 11 triggers and all 7 of the gun's kills:
* KILL #1 at 12.28 s: trigger at 12.17 s, 39 Hz, 0.43 s after a hurt.
* KILL #2 and #3 in one tic at 13.54 s: one hit, and both kills credited by the 3-tic rule. A pistol bullet stops at
  the first monster it hits, so at most one of the two is the bullet's. In the clip, two demons vanish together: one
  near the centre line, and one right of centre beside an imp fireball. The second is most likely an infighting kill,
  so this run's 7 gun kills probably include one that the gun did not make.
* KILL #4 at 17.62 s, #5 at 19.00 s, #6 at 22.42 s and #7 at 24.51 s.
* Between 14.54 and 15.60 s, after fireballs hit the player at 14.11 and 14.54 s (health 88 -> 79 -> 61), five
  triggers in 1.06 s fired two shots, and both missed.
* The last of those five (15.60 s) is the one trigger across the three recordings with nothing within 128 units in the
  preceding 0.5 s. It came 0.40 s after the fly's own miss.
* The player dies at 26.08 s, after 18.46 s of play; nobody firing on the same Doom seed lasted 18.72 s.

Of the 11 triggers, 10 had a fireball bursting on the player as the closest object. Four triggers and 6 of 14
crossings fell 0.1-0.7 s after the fly's own shot.

**Control 1.** The Doom is identical until 12.28 s, because neither run fired in episode 1 and both resumed at
7.62 s. The GF crosses at 12.13 s. On screen at 12.22 s it reads 48.5 Hz while the trigger shows DISCONNECTED, and the
demon that the connected run killed keeps coming. Of the 14 crossings, 12 had a fireball as the closest object. The
other 2 had a demon 60 units away at the screen's edge (19.85 and 19.88 s).

**Seed 2** (Doom seeds 2001 and 2002). This run has the most triggers (15), the most shots (18) and the most misses
(12). Episode 1:
* The first trigger (5.17 s) comes 0.52 s after an imp fireball kills a demon 86 units ahead, before any hurt.
* The first five triggers (5.17-10.94 s) fire six shots, and all miss.
* KILL #1 at 12.65 s: trigger at 12.54 s, 0.60 s after another infighting kill 88 units ahead.
* KILL #2 at 15.05 s.
* At 17.54 s (53 Hz, health 4), a demon coming down the centre line takes a hit at 17.65 s and dies on the refire
  at 18.05 s, about 140 units away (KILL #3).
* Another demon reaches melee range and kills the player at 19.62 s. Over the last 0.7 s the on-screen GF reads
  1.2-10.7 Hz (7.2 Hz at 19.60 s). The episode lasted 18.62 s. Nobody firing on Doom seed 2001 lasts 20.05 s: on this
  seed, the gun's shots ended the episode 1.43 s sooner.

Episode 2 (Doom seed 2002, from 20.82 s) plays the dev runs' common schedule:
* The first trigger comes 5.55 s into the episode (26.37 s, 46 Hz), 0.57 s after a fireball kills a demon 86 units
  ahead. The shot kills the next demon at 59 units (KILL #4, 26.48 s).
* KILL #5 at 29.05 s.
* The clip ends on a trigger at 29.97 s: 40 Hz, health 6, a fireball bursting on the player and a demon 54 units
  away. The pistol fires about 4 tics after ATTACK goes down, so the shot falls after the clip ends.

Of the 15 triggers, 12 had a fireball bursting on the player as the closest object and 3 had one bursting on a demon.
Two triggers and 4 of 20 crossings fell 0.1-0.7 s after the fly's own shot.

**Control 2.** All 16 crossings had a fireball as the closest object. The player dies at 21.05 and 29.34 s. The GF
crosses at 27.80 s, 5.55 s into episode 2: the same schedule moment as the connected run's KILL #4. On screen at
27.86 s it reads 56.6 Hz while the trigger shows DISCONNECTED and a demon closes in.

**Controls against the never-fire baseline.** Each control reproduces its seed's Doom-only never-fire baseline
exactly: the same episode lengths to 0.01 s, the same infighting kills (8, 9, 11) and the same hurts (11, 10, 10).
With the trigger disconnected, nothing the brain does reaches Doom. The brain still crosses 33 Hz (6, 14 and 16 times),
so the crossings are the fly's own; the gun only listens. Each connected run's `replay` baseline reproduced its Doom
outcome exactly (`replay_matches_fly: true` on all three).

**Against chance, on the recorded seeds** (Doom only, the same Doom seeds and timeline, `summary.baselines`):

| seed | never fire | always fire | 50 random triggers at the fly's rate: kills / deaths, median (range) | draws with at least the fly's kills (more) | draws with at most the fly's deaths (fewer) | the fly |
|---|---|---|---|---|---|---|
| 0 | 0 kills, 2 deaths (11.74, 11.43 s) | 17 kills from 69 shots, 1 death (16.39 s) | 5 (1-9) / 1 (1-2) | 43 (29) | 35 (0) | 4 kills, 1 death |
| 1 | 0 kills, 2 deaths (5.42, 18.72 s) | 18 kills from 69 shots, 2 deaths (15.71, 11.95 s) | 8 (1-14) / 1 (1-3) | 39 (32) | 48 (30) | 7 kills, 2 deaths |
| 2 | 0 kills, 2 deaths (20.05, 7.09 s) | 18 kills from 66 shots, 2 deaths (9.71, 15.55 s) | 9 (4-14) / 2 (1-3) | 49 (44) | 17 (0) | 5 kills, 1 death |

Pooled over the 150 random draws:
* 105 killed more than the fly did, and 19 killed fewer.
* 30 died fewer times than the fly, 50 died more often, and 70 died as often.

The GF trigger fires about half a second after a fireball burst, whether or not a demon is in the line of fire. On
kills, it does worse than most random timings; on deaths, the result is mixed. **The trigger's timing is not shown
to beat chance.** Holding the trigger down kills far more: 17-18, where no random draw exceeds 14. It dies no less
often than the fly, with 1, 2 and 2 deaths against the fly's 1, 2 and 1. Keep "survives longer" and "defends itself"
out of the trailer and the thread.

## Corrections (caption skeptic)

Checked against all ten run logs, `fix/flash_mod.log`, the pre-freeze logs, the builder's scratch arrays, frames
extracted from all six clips, and `games/doom.py`. Source hashes match the logs (`3fa9241469623dfc`,
`b110142562a927a1`). Every clip is 1,500 frames at 1920x1080 and 50 fps.
* Headline: "the fly's giant fibre presses it" became "the gun fires when the fly's giant fibre does". A declared
  decoder presses ATTACK; this is the caption's own headline-safe wording.
* CONNECTOME row: "nothing ... tuned" became "nothing ... tuned for this game". The row now names the shipped stop-gap
  gain on LC4 / LPLC2 -> DNp01 (x3; x6 in total), which the whole effect rides on.
* Doom row: "pistol, one shot kills" became "one shot kills a first-wave monster; replacements are tougher". Three
  of the 18 hits did not kill.
* Score row and the totals: added that the 3-tic rule can credit the gun with an infighting kill. At seed 1, 13.54 s,
  one hit is credited with two kills, and two demons vanish together, one beside a fireball. "16 kills by the gun"
  now says it came from 15 killing hits.
* Seed 0, KILL #1: the kill shot at 7.77 s was started by the 7.65 s press. That trigger's banner read "34 Hz", and
  the shot came 4 tics later. The "45 Hz" trigger at 7.74 s only held ATTACK into the refire that missed. The earlier
  text put the 45 Hz crossing just before the kill.
* "Same Doom, same moment": "In four stretches ... within 0.00-0.03 s ... seed 0 episode 1: 7.73 and 7.74 s" was
  selective. 7.73 s was the connected run's second crossing; its first was 7.65 s. There are seven identical stretches,
  including seed 0 episode 2 (4.41 s in both) and seed 1 episode 3. The range is now 0.00-0.09 s.
* Control 0, 12.53 s: added that the GF was at 0.2 Hz with the demon already at melee range. The rise followed a
  fireball burst at 12.02 s by 0.51 s, so the demon is not shown to be what drove the crossing.
* Seed 1 episode 1: "2-4 Hz on screen at 5.8-6.5 s" became 0.1-12.1 Hz. The readout was 0.7 Hz at 6.2 s and 12.1 Hz
  at 5.2 s.
* Seed 2 death: "with the GF at about 2 Hz" became 1.2-10.7 Hz over the last 0.7 s.
* Recorded "Against chance": "Holding the trigger down (17-18 kills) beats both" was false on deaths. Always-fire
  died 1, 2 and 2 times against the fly's 1, 2 and 1.
* Display note: "4 tics after a shot" became 5 tics. The shot was at tic 506 and the death at tic 511.
* LC4 limits: "under ... 5.1 Hz in control seed 0" was wrong, because that peak is 5.14 Hz. The limits now give the
  logged peaks.
* Pre-freeze section: added provenance for its figures.
  * The screen-width table, the prototype (79-97 Hz, 3-8 crossings), the weapon check (5 of 10) and the turning test
    (under 21 Hz; 3 crossings, nothing within 169 units) were recomputed from the builder's scratch arrays, and all
    matched. Those arrays are not in `out/`.
  * The honesty review's figures and the builder's "6 of 6" have no saved output.
  * "0.32-0.35 s three times" became 0.29-0.34 s, the value in the surviving `pre_fix_builder` logs.
  * The schedule list now cites the logs it actually comes from.
