# tron: light cycles on the escape pathway

**Two connectomes on light cycles. A declared decoder turns the MaleCNS cycle only when its giant fibre crosses
33 Hz, and 19 of its 20 turns came about a metre from a wall. Over three recorded seeds it took 7 rounds to FlyWire
FAFB's 3, with 4 draws; each of the 7 ended with FAFB's own decoder turning the orange cycle toward a wall it then
hit. It crashed
less often per minute than 140 of 150 brain-free riders turning at random times at its own rate; a one-line rule
that turns 1.1 m before any wall and flips a coin for the side did as well or better (median 2.24 / 7.56 / 2.24
crashes per minute against the fly's 2.24 / 7.56 / 6.93). It picks left or right about as well as a coin.**

Seeds 0, 1 and 2, 40 s each, on the cluster's NVIDIA B200s (Clips, below). The MaleCNS cycle won 3 : 1, 2 : 1 (2
draws) and 2 : 1 (2 draws). In 3 x 40 s the decoder turned it before every wall it was heading for; it never rode
straight into one. Its 7 crashes were 6 meetings with the orange cycle and 1 turn into a corner. Every round it won
ended 0.03-0.31 s after FAFB's decoder turned the orange cycle toward a wall or trail 0.3-3.1 m away; in one of
them (seed 1, round 4) the MaleCNS cycle never turned. Between the connectome and the handlebars there is only a
declared decoder: a 33 Hz threshold on DNp01 and a side rule. What the decoder beats is random timing. Against a
brain-free rule that turns 1.1 m before any wall (added after the recordings, below), it does as well on two seeds
and worse on the third when that rule flips a coin for the side, and worse on every seed when the rule turns towards
the roomier side.

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the flies) | Cyan: MaleCNS v1.0 as shipped (167,106 neurons, preset `raw`, nothing instrumented). Orange: FlyWire FAFB v783 as shipped (139,255 neurons, `FlyBrain(dataset="fafb")`, preset `raw`). Everything between the light reaching each eye and the rates of DNp01, LPLC2 and LC4 is the model, including the shipped stop-gap gain on LC4 / LPLC2 -> DNp01 that the README's ledger declares. The HUD's DNp01 and LPLC2 + LC4 numbers are their rates. |
| **DECODER** (written for the game, declared) | One rule per fly, `turn_A` / `turn_B` (an attached `games.common.ReadDecoder`, writes nothing). **Trigger:** the cycle turns 90 degrees on each upward crossing of mean(DNp01 L, R) through 33 Hz, no sooner than 0.30 s after its last turn. **Side:** s = (LPLC2_L + LC4_L) - (LPLC2_R + LC4_R); s' = the mean of s over the last 100 ms minus -2.1 Hz (the brain's resting offset); **s' > 0 -> turn right, else left.** Because of the offset, the cycle turns right whenever the left-minus-right difference is above -2.1 Hz, which can be a turn toward the eye whose LPLC2 + LC4 is momentarily larger (4 of the 10 MaleCNS decoder turns in the three seed-102 dev runs, below). In the recordings, 14 of the 20 MaleCNS turns went right, and 12 of those 14 went toward the eye whose LPLC2 + LC4 was larger on the turn's tick (`loom_L`, `loom_R` in each turn event). The HUD prints s' beside each decoder arrow. 33 Hz is the shipped body model's takeoff threshold (`flyverse.body.Flight.gf_hz`), asserted at start-up, not tuned. |
| **GAME** (not the fly) | The arena, the constant speed (10 m/s), the 90-degree turn itself, collisions and scoring, the round schedule (the same six start positions for every seed), the 1.2 s GET READY hold, the scene's look, the eye mount, the chase camera, minimap, bloom and banners. In the baseline arms: the random and never-turning players, and the brain-free replays behind the ranking. |
| **not claimed** | The flies do not steer, aim, chase or avoid each other. Nothing here turns on its own: a turn happens only when the decoder's threshold is crossed. The MaleCNS giant fibre crosses 33 Hz late. In the open-loop probe with the game's 3.5 m walls that was 0.48-1.4 m (40-130 ms) before a wall. In the recordings, 19 of its 20 turns came 0.5-1.5 m (50-150 ms) before one; the other came with 48.7 m clear. The side choice does not beat a coin: 10 turns went to the side with more room, 8 to the side with less, and 2 had equal room. The pathway did not catch the other cycle coming from the side or nearly head-on. In five side-on meetings, the MaleCNS DNp01 stayed at or below 8.5 Hz over the last second; in the one near head-on meeting (seed 1, round 3) it peaked at 29.7 Hz. Many of FAFB's crossings are spontaneous: 36 of its 78 turns (46 %) came with more than 13 m clear. This is not a comparison of the two animals or sexes. |

## Design

A TRON arena, 60 x 60 m with 6 m walls and a 5 m floor grid. Two light cycles ride at a constant 10 m/s and turn
only in 90-degree steps; each leaves a jet wall 3.5 m tall and 0.2 m thick along its path. A cycle whose nose meets
an arena wall, a jet wall or the other cycle is out ("derezzed"); the last cycle riding takes the round; both out on
the same 10 ms tick is a draw; a round still running after 30 s is a draw. Each round opens with the cycles standing
still for 1.2 s (GET READY; the decoders' turns are not live), and after a crash the arena holds for 1.8 s while the
wreck's walls fade, then the next round starts from the schedule. The clip runs a fixed time (40 s of brain time),
not a fixed number of rounds.

Each fly's eyes ride its cycle at the nose (1.3 m ahead of its centre, the point that collides), 1.0 m above the
floor, looking along the heading. Each brain's own ommatidial columns (MaleCNS 1,466, FAFB 1,581) x 7 rays are traced
on the GPU through the same scene the chase camera shows: dark-glass jet walls with bright top and base edges, dark
arena walls with a lit grid, a glossy floor whose grid lines are filtered to each ray's footprint (1.5 deg, so the 7
rays per column do not alias them), one mirror bounce off the floor, and a sky with a horizon glow. UV = 0.5 B. The
fly's own cycle body is not in its view. The radiance goes to `fb.vision` untouched; only the chase camera is
exposed (x 0.32), bloomed and tone mapped, which is display.

| what | kind | law and parameters |
|---|---|---|
| the brains | CONNECTOME | `FlyBrain(seed=s)` (MaleCNS) and `FlyBrain(dataset="fafb", seed=s)`, preset `raw`, one read-only decoder attached to each. Two separate brains stepped one after the other each 10 ms tick. |
| `turn_A`, `turn_B` | DECODER | reads `rate_hz` of DNp01 L / R, LPLC2 L / R and LC4 L / R (soma side); population means (MaleCNS 1 / 1, 94 / 91, 71 / 55 cells; FAFB 1 / 1, 108 / 102, 54 / 50). **Trigger:** gf = mean(DNp01 L, DNp01 R); a turn fires on the tick gf goes from < 33 Hz to >= 33 Hz, if >= 0.30 s since the last turn and the round is live. **Side:** s = (LPLC2_L + LC4_L) - (LPLC2_R + LC4_R); s' = mean of s over the last 100 ms minus the resting offset (-2.1 Hz for both brains); s' > 0 -> turn right, else left. Latency: the decoder reads the previous 10 ms frame, and the turn is applied before the next move. |
| resting side offset | DECODER (set on dev seed 100) | MaleCNS: its mean s while riding open floor and passing a wall 3 m to the left (probe, dark look) was -2.10 and -2.14 Hz. FAFB: -2.05 Hz on the wall pass in the dark + light-bands look (-2.4 to -2.6 in the glow look); FAFB had not been probed in the game's own dark look when the offset was set. Probed afterwards in that look (3.5 m walls, seed 100, `out/games/tron/dev4/probe_fafb_dark_game_s100.json`), its mean s was -2.37 Hz on open floor and -2.36 Hz on the wall pass, so with -2.1 Hz subtracted FAFB's s' rests near -0.3 Hz: a lean to the left. The offset was left at -2.1 Hz (the code was being frozen); this is reported, not corrected. Uncorrected, nearly every turn would go left. The offset is a constant, -2.1 Hz for both, declared in `SIDE_OFFSET_HZ`. |
| arena, speed, turns, collisions | GAME | as above; a swept test of the nose's path against every box each tick, so a thin wall cannot be skipped. |
| scene look | GAME | chosen on dev seed 100 for what the MaleCNS eye responds to (below): `STYLE = "dark"`. |
| round schedule | GAME | six (cyan, orange) start poses, cycled; the same for every seed. |
| baselines (clips) | GAME | `--player random`: the cyan cycle has no brain and turns at Poisson times at a given rate, side 50/50, the same 0.30 s re-arm (numpy `default_rng([0x7209, seed])`); for the recordings the rate is the matching fly run's own `summary.turn_rate_A_hz`, passed with `--random-rate`. `--player never`: the cyan cycle never turns. The orange FAFB fly is closed loop and unchanged in both, so what it does also differs between arms. |
| baselines (ranking) | GAME | After each connected fly run, with no brain, the game is replayed on the same seed, arena, schedule and tick timeline, scored by the same code; the orange cycle repeats its recorded turns at the same tick of each round, open loop (after they run out it rides straight). Arms: the fly's own turns (a determinism check: it must reproduce the run's rounds; it did for dev102, 15 of 15 events), a never-turning cyan cycle, and 50 random cyan players at the fly run's own turn rate (turns per second of riding), numpy `default_rng([0x7209, seed, 1 + draw])`. **Metric:** cyan crashes per minute of riding (riding = race time with the cyan cycle alive; lower is better), with rounds won alongside. Written to the run log's `summary.baselines`, with `rank_of_fly`. `python games/tron.py --baselines-from <log>` recomputes it on a CPU. |
| control | GAME | `--control`: the MaleCNS decoder is attached and read, its turns are logged (`turn_not_applied`) and not applied: the cyan cycle never turns. Its path is the never-turning player's; the difference is that the brain runs and the log shows what the decoder would have done. |
| HUD | mixed | CONNECTOME: both flies' DNp01 traces against the 33 Hz line, LPLC2 + LC4 per eye. DECODER: the side each decoder would choose now (the arrow) with its s', the turn tally per fly, the banner "GIANT FIBRE *n* Hz -> TURN LEFT / RIGHT" (a DECODER chip tab) with the free run ahead at that moment on its subline (a GAME chip). GAME: score, round clock, minimap, ROUND / GO / DEREZZED / DRAW banners (GAME chip tabs), the baseline players' panel. The footer states the decoder's law with the offset and names what each cycle is in this arm. What each fly sees is the exact radiance handed to `fb.vision` (`eye_colors`, 'human', exposure 2). |
| chase camera | GAME (display) | 10 m behind, 3 m left of and 8 m above the cyan cycle, yaw smoothed with a 0.22 s time constant of brain time, its floor position kept within +-29 m (inside the arena), looking at a point ahead of the cycle that comes nearer when the camera is clamped. On a brain-free replay of dev102, the line of sight to the top of the cyan cycle's nose was blocked by a wall in 8 % of frames and the camera was never outside the arena (the earlier camera: 92 % and 21 %). |

The run log records every turn (tick, brain, side, gf, s', the instantaneous LPLC2 + LC4 per side, the free run ahead
along the old heading, the free run along the new one, the tick in the round), every upward crossing of 33 Hz with
what became of it (`gf_crossing`: turn / turn_not_applied / blocked_rearm / not_live, and the time since the last
turn), every crash (what was hit, the round's time, the cycle's turns, and for a fly the max gf in the last second),
every round, and, in the summary, the turn rate, riding time, crashes per minute of riding, the turns that came with
a wall within 1 s ahead, the turns into a wall within 1 s, each fly's crossings by phase and by outcome, and the
baselines above.

### What was measured first (dev seeds >= 100)

**1. Open loop: what reaches the giant fibre, and when.** `python games/tron.py --probe`: the eye rides straight at
a wall (a jet wall spanning the whole view, a half wall on one side, the arena wall) or past one (3 m to the left) or
over open floor, at a fixed speed, after 1.2 s of the first frame held still. It stops 0.3 m before the wall. The
trials run one after the other in one brain, so each trial starts from the state the previous one left. One run per
look and brain, seed 100 (MaleCNS glow look also on seed 101: the first trial's rows were identical, later trials
differed by a few Hz, GPU nondeterminism; the raw brain has no Poisson input). "Moving" excludes the frames after the
eye stopped. Files: `out/games/tron/probe*.json`. Three trial lists were used:

* the first probe (`probe_*_s100.json`, `probe_malecns_s101.json`; glow look, 8-25 m/s, 2.2 and 4 m walls) came from
  an earlier trial list that is no longer in `tron.py` (its source, sha256 `730d3846...`, was fetched back from the
  cluster run directory by the honesty review);
* `probe2_*.json` (2.2 m walls, plus two 3.5 m rows) is reproduced by `--probe-set probe2`;
* `--probe-set game` is the same list with the game's own 3.5 m walls. It was run once on the current code (MaleCNS,
  dark, seed 100, `out/games/tron/review_honesty/probe_now_malecns_dark_s100.json`) and once for FAFB in the dark
  look (`out/games/tron/dev4/probe_fafb_dark_game_s100.json`). New probe files record the trial list and the source
  hashes.

| look | brain | wall approaches (v, height) | moving GF max | crossings of 33 Hz while moving | first crossing |
|---|---|---|---|---|---|
| glow (bright walls, black arena) | MaleCNS | 8-25 m/s, 2.2 and 4 m, arena wall, half walls | 8.1-13.2 Hz | **0 of 7** | none. GF reached 17.9-32.7 Hz only after the eye had stopped at the wall. |
| glow | MaleCNS | open floor, wall pass | 8.4, 12.3 Hz | 0 | |
| glow + 1 m light bands | MaleCNS | 10-20 m/s, 2.2 / 3.5 m, arena, half walls | 8.7-13.4 Hz | 0 of 8 | none |
| dark glass walls, lit floor and horizon (`probe2`) | MaleCNS | 10 m/s, 2.2 m | 44.6 Hz | 1 | 0.50 m from the wall (50 ms) |
| dark (`probe2`) | MaleCNS | 15 / 20 m/s, 2.2 m | 18.2, 18.8 Hz | 0 | |
| dark (`probe2`) | MaleCNS | 10 m/s, 3.5 m | 70.4 Hz | 1 | 0.80 m (80 ms) |
| dark (`probe2`) | MaleCNS | 15 m/s, 3.5 m; arena wall 6 m at 15 m/s | 46.8, 46.8 Hz | 1, 1 | 0.75 m (50 ms), 0.75 m (50 ms) |
| dark (`probe2`) | MaleCNS | half walls 2.2 m at 12 m/s | 25.9, 26.9 Hz | 0 | |
| dark (`probe2`) | MaleCNS | open floor, wall pass | 11.8, 9.1 Hz | 0 | |
| **dark, 3.5 m walls (`game`)** | MaleCNS | 10, 15, 20 m/s; 3.5 m at 10 / 15 again; arena 15 m/s; half walls 12 m/s | 57-103 Hz | **8 of 8** | 1.3, 0.9, 1.4, 1.0, 0.6 m; arena 0.9 m; half walls 0.48, 0.48 m (40-130 ms) |
| dark, 3.5 m (`game`) | MaleCNS | open floor, wall pass | 12.4, 7.4 Hz | 0 | |
| dark + light bands | MaleCNS | 10-20 m/s, 2.2 / 3.5 m, half walls | 8.8-27.4 Hz | 0 of 7 | only the arena wall crossed, 1.35 m |
| glow | FAFB | every run, including open floor | 37.6-49.0 Hz | 1-4 per run | 20-68 m from any wall |
| dark + light bands | FAFB | every run, including open floor and pass | 32.0-76.1 Hz | 0-5 per run | mostly 12-43 m from the wall |
| dark, 3.5 m (`game`) | FAFB | 10-20 m/s, arena, half walls; open floor, wall pass | 43.2-96.5 Hz | 2-6 per run, every run | open floor and wall pass: 2 each, 12.6-28 m out. Wall trials: the first crossing 14-54 m from the wall in 7 of 8 (3.6 m in the other), and every wall trial crossed again 0.36-4.05 m out |

So: in the MaleCNS brain, bright walls never reached the giant fibre before contact at any speed tried. Dark walls in
front of a lit background (an OFF loom) did. With 2.2 m walls they did so only at 10 m/s, 0.5 m out; at 15 and 20 m/s
the crossing came only after the eye had stopped. With the game's 3.5 m walls every approach crossed while moving,
0.48-1.4 m (40-130 ms) out, at 10, 15 and 20 m/s and on both half walls. The two 3.5 m rows that appear in both lists
(10 and 15 m/s) crossed at 0.80 / 0.75 m in `probe2` and 1.0 / 0.6 m in `game`: the same trial differs by a few tenths
of a metre depending on what ran before it. The game uses the dark look, 3.5 m walls and 10 m/s, and puts the eye at
the nose so that "1 m from the wall" is 1 m before the collision. In the game's own look the FAFB giant fibre crossed
33 Hz two to six times per 3 s run, open floor included, whether or not a wall was ahead: many of its turns are not
wall-driven.

The scene cut at the start of each probe trial drove the GF to 5-103 Hz in the MaleCNS dark runs (up to 114 Hz in the
glow runs). On the hold's last tick (1.2 s after the cut) it was 0.0-2.5 Hz in the MaleCNS dark `probe2` run (max over
the hold's last 100 ms 0.1-5.3 Hz), 0.0-0.2 Hz in the FAFB dark + bands run, and up to 11.0 Hz in the MaleCNS glow runs
(max over the last 100 ms up to 27 Hz). That is why each round opens with a 1.2 s GET READY during which no turn is
live. It does not always absorb the cut: in `control102` the MaleCNS GF crossed 33 Hz once during GET READY (34.0 Hz,
t = 7.03 s); in `dev103` neither brain crossed during GET READY (FAFB crossed three times during an END hold, `not_live`). Each recorded log reports its GET READY crossings (`gf_crossings_*_by_phase`).

A reviewer also tested whether the game's instant 90-degree turn drives the GF by itself (scripted turns on open
floor with more than 13 m clear, `out/games/tron/review_honesty/turnprobe_*.json`): the MaleCNS GF stayed at or below
5.2 Hz after every turn (8.3 Hz riding straight), and FAFB's 15-27 Hz with no crossing after a turn; FAFB crossed 33 Hz
0.40-0.46 s after motion onset on open floor in every condition.

**2. The side signal.** LPLC2 + LC4 left minus right averaged between -1.5 and -2.5 Hz over every MaleCNS run (all
looks, walls or not), so the uncorrected decoder would almost always turn left. With the -2.1 Hz offset subtracted,
the `probe2` dark half-wall runs gave the right sign in the last 300 ms before arrival (wall on the left: s' = +0.41
Hz -> turn right; wall on the right: -0.20 Hz -> turn left), one run each; but the head-on walls, which have no
correct side, gave s' from -0.12 to +1.81 Hz, a spread larger than the half-wall difference. In the `game` probe the
half walls came out with the right side at the crossing (s' = +4.3 and -1.3 Hz). In closed loop, 8 of the 10
MaleCNS decoder turns in the three seed-102 runs below went right, and in dev102 3 of its 4 turns went to the side
with less free room (a brain-free replay of the run: e.g. t = 4.22 s, R with 23.7 m free against 33.7 m on the left).
In the glow look both FAFB half-wall sides came out inverted (`probe_fafb_s100.json`). The side is not reliable, and
the caption claims only what each log records.

**3. Closed loop, dev clips.** `out/games/tron/dev3/dev102.mp4` + `.json` (seed 102, 16 s of brain time) was made
with the final decoder, scene and camera of that time; afterwards the banner wording, chips and legend, the chase
camera pose, and the default random rate changed (display and GAME only). The honesty review reran the same seed
and command on the cluster (`review_honesty/rerun102`, and `control102`). Two rounds, 1 : 1 in dev102.

| | MaleCNS (cyan) | FAFB (orange) |
|---|---|---|
| turns (race time 10.4 s) | 4 (0.38 per s) | 7 (0.67 per s) |
| turns with the wall ahead within 1 s | 3: at 0.90, 1.10 and 1.00 m (90-110 ms before impact) | 3: at 4.1, 1.9 and 4.0 m |
| turns with nothing within 13 m ahead | 1 (17.3 m ahead, GF 33.8 Hz, 0.32 s after the previous turn) | 4 (13.7-44.2 m ahead) |
| side chosen (s') | R (+2.34), L (-0.89), L (-1.86), R (+2.31) | 5 L, 2 R |
| 33 Hz crossings during GET READY | 0 | 0 |
| crashes | round 1, 5.77 s in: into FAFB's jet wall, 0.17 s after the far-from-any-wall turn left it 1.6 m of free run | round 2, 4.64 s in: into MaleCNS's jet wall |

A rerun of the same seed and command gives a different game. rerun102 used the same game and brain code as dev102
(only the banner wording and the default random rate differed); the runs diverged at the first event (FAFB's first
turn at 2.35 vs 2.36 s, GF 36.8 vs 43.7 Hz). rerun102 finished one round (FAFB won; a second was running at the
end). All four of its MaleCNS turns came 0.9-1.1 m before a wall and none far from one; 3 of FAFB's 6 turns came with
more than 13 m clear. control102 (decoder read, not applied) logged MaleCNS decoder turns at 1.4 and 1.1 m before a
wall, and 2 of FAFB's 4 turns with more than 13 m clear. What reproduced across the three: MaleCNS turns about a metre
before a wall, and FAFB turning about half the time with more than 13 m clear.

**dev103** (`out/games/tron/dev4/dev103.mp4` + `.json`, seed 103, 14 s, the fixed code: camera, chips, legend,
crossing log and baselines; the frozen `tron.py` differs from it only in the footer's wording, the banners' height on
screen and one declaration's text). One finished round, a draw: both cycles out on the same tick at 4.91 s into the
round, each logged as hitting the other, 0.04 s after a MaleCNS turn; a second round was running at the end.

| | MaleCNS (cyan) | FAFB (orange) |
|---|---|---|
| turns (riding 9.8 s) | 3 (0.31 per s), all R: at 1.5, 0.9 and 1.1 m before a wall (s' +1.22, +1.93, +0.35) | 7: 2 at 1.9-2.2 m from a wall ahead, 1 at 9.1 m, 4 with 34-43 m clear |
| 33 Hz crossings | 3, all turns | 10: 7 turns, 3 during an END hold |
| crashes | 1 (the draw) | 1 (the draw) |

Its baselines (`summary.baselines`, brain-free, orange replayed): the replay of the fly's turns reproduced the run
(`replay_matches_fly` true). The never-turning cyan cycle crashed twice in 7.8 s of riding (15.4 per minute). The 50
random cyan players at 0.306 turns per second of riding had 6.1-17.6 crashes per minute, median 15.4; the fly run's
6.1 equalled 12 draws and was below 38, and no draw was lower. The fly run won no round (one draw); 5 of the 50 draws
won one. One 14 s dev run is too short to rank anything; the recorded 40 s runs are. `control103` (6 s): the
decoder's turn at 4.14 s (R, GF 33.6 Hz, 1.6 m before FAFB's wall) was logged, not applied, and the cyan cycle hit
that wall 0.16 s later.

So in closed loop the MaleCNS cycle turned about a metre from the wall, as the probe predicted, and survived most of
those turns. The FAFB cycle turned more often, about half the time with nothing close ahead. Dev clips are not a
result; the recorded seeds and their baselines are.

A longer dev batch (seeds 100 and 101, 40 s: fly, control, random) was started first; the cluster restarted each of
its jobs while the first processes kept running (Limits), so its files had two writers and are not used here.

### Limits

* The MaleCNS giant fibre crosses 33 Hz late: open loop with the game's 3.5 m walls, 0.48-1.4 m (40-130 ms) before a
  wall at 10-20 m/s. In the recordings at 10 m/s it crossed 0.5-1.5 m (50-150 ms) before one. With 2.2 m walls it
  crossed while moving only at 10 m/s (0.5 m). The speed, wall height, eye mount and look are GAME choices made so
  that it can cross in time.
* Bright (emissive) walls, the iconic TRON look, did not drive this model's escape pathway before contact at any speed
  tried. The game's walls are dark glass with bright edges, in front of a lit floor grid and horizon (an OFF loom).
* **The side law.** s' is the 100 ms mean of (LPLC2 + LC4) left minus right, minus a constant resting offset of
  -2.1 Hz set on dev seed 100; s' > 0 turns right. The side rests on a few-tenths-of-a-Hz difference left after that
  constant is removed. For FAFB the offset is about 0.3 Hz off: its resting s' sits near -0.3 Hz, a lean to the left
  (Design). In the recordings, 14 of the 20 MaleCNS turns went right. A brain-free replay of each fly run gives the
  free run on each side at every turn; each replay reproduced its run's rounds. By that measure, 10 turns went to the
  side with more room, 8 to the side with less, and 2 had equal room. In all three seeds, the first turn of round 1
  and the first turn of round 2 both went right, toward less room. The side is not shown to beat a coin.
* The FAFB giant fibre crosses 33 Hz on open floor, so many of the orange cycle's turns are spontaneous, not replies to
  walls. In the recordings, 36 of its 78 turns (46 %) came with more than 13 m clear, including its first turn in
  every run: 2.36 s, 43.7 Hz, 39.1 m clear, the same in all twelve runs. The other 42 came with a wall within 10 m.
  Some of its crashes followed turns made with more than 13 m clear. On seed 0,
  round 4, it turned left with 49.1 m clear into 1.0 m and hit the arena wall 0.10 s later. This describes the two
  models as shipped in this scene; it is not a comparison of the animals.
* The escape pathway answered walls ahead, not a cycle arriving from the side. Round 5 of the schedule sends the
  cycles on crossing paths. In the two fly runs that finished it and in all three controls, the cycles met side-on,
  and the MaleCNS DNp01 peaked at 4.6-8.5 Hz in the last second. Two of the fly's four draws were such meetings. In a
  third (seed 1, round 3) the cycles were passing nearly head-on, 0.8 m apart centre to centre, and the MaleCNS
  DNp01 peaked at 29.7 Hz, under the threshold.
* **One run per arm per seed.** The comparison with the closed-loop arms rests on single runs. The brain-free ranking
  rests on 50 random draws per seed against one fly run. Seed 2's control and never arms come from recording batch 1
  (Clips). They used the same code and device type but a different submission, and started 49 minutes before seed 2's
  fly run.
* **A distance rule does as well.** The decoded cycle beats random turning at its own rate. A brain-free GAME rider
  that turns 1.1 m (or 1.5 m) before any wall does as well or better (the table under "Against a distance rule"). Its
  advantage over random is that of a wall-proximity trigger; the connectome is not shown to add anything beyond one.
* A rerun of a seed is a different game. The raw brains have no Poisson input, the seed changes nothing in them, and
  the round schedule is the same for every seed. Differences between runs, including two runs of one seed, are GPU
  floating-point nondeterminism amplified by the game (dev102 vs rerun102 above). In the recordings, FAFB's first
  2.36 s played out identically in all twelve runs. The MaleCNS's first crossing fell at 4.14-4.20 s in its six runs.
  The recorded seeds are single draws of a nondeterministic run, not reproducible outputs.
* **The opponent differs between arms.** The ranking's orange cycle is an open-loop replay of FAFB's recorded turns.
  It does not react to the random or never-turning cyan cycle, whereas in the fly run it saw the MaleCNS cycle's walls.
  The closed-loop baseline clips keep the FAFB brain, so there the orange cycle differs between arms. Against the
  fly, FAFB turned 22-28 times per clip and crashed 3-4 times. In the other nine clips it turned 8-25 times and
  crashed at most once; there the cyan cycle crashed in 44 of the 45 finished rounds (the exception: the random arm,
  seed 1, round 3). Each is one reading of "the same opponent".
* A recording's log has `device` NVIDIA B200 and `commit` null: the cluster run directory is not a git checkout. The
  code is identified by the `sources` hashes in the log (Clips).
* **Cost.** Each tick steps two brains, and each frame ray-traces a 1440 x 968 chase camera. Alone on a B200, the dev
  clips took about 14-20 s of wall per brain second. The camera's ray grid is now built once per size; it used to be
  rebuilt on the CPU every frame. After that change, dev103 took 162 s of wall for 14 s of brain. The recordings took
  189-1367 s of wall per 40 s clip, with up to nine of these jobs running at once (batch 1 ran all nine together;
  Clips). The brain-free baselines add 13-14 s per fly run.
* **Cluster note.** In the dev batch `tron-dev1` and in recording batch 1, the job manager marked jobs `preempted`
  and started a second attempt while the first process kept running. pygame / SDL turned the SIGTERM into a quit
  event that the headless loop never read. Both attempts wrote the same clip, console and log paths. The batch-1
  consoles fetched into `rec1_preempted_discarded/` hold the second attempt's opening lines and the first attempt's
  later ones, with NUL bytes between; none reached "wrote" or wrote a run log. (An ffmpeg failure at close, "ffmpeg
  failed writing ...", is not in any fetched file.) After batch 2 had been shipped, `games/common.run` gained `SDL_NO_SIGNAL_HANDLERS=1` so that SIGTERM terminates.
  Batch 2 ran the older `common.py` and relied on a per-clip ownership guard instead (Clips). Before trusting a
  recording, check the job's attempts and that one process wrote it.

### Commands

```
# on the cluster (scripts/cluster_run.py; games/PLAN.md "Where GPU work runs"), from the run directory,
# after `source .venv/bin/activate`; each console to <clip>.console.txt
python games/tron.py --seed {seed} --seconds 40 --record out/games/tron/rec2/seed{seed}.mp4        # writes summary.baselines
# same job, right after: its own rate, summary.turn_rate_A_hz of rec2/seed{seed}.json (0.373, 0.168, 0.231)
python games/tron.py --seed {seed} --seconds 40 --player random --random-rate {rate} --record out/games/tron/rec2/random_seed{seed}.mp4
python games/tron.py --seed {seed} --seconds 40 --control --record out/games/tron/rec2/control_seed{seed}.mp4
python games/tron.py --seed {seed} --seconds 40 --player never --record out/games/tron/rec2/never_seed{seed}.mp4
# seed 2's control and never arms (batch 1) wrote out/games/tron/{control,never}_seed2.mp4; kept in batch1_single/
# CPU, no brain: recompute a fly run's baselines (writes <log>_baselines.json)
python games/tron.py --baselines-from out/games/tron/rec2/seed{seed}.json
python games/tron.py --probe out/games/tron/probe.json --seed 100 [--probe-set game|probe2] [--style dark|glow|...] [--dataset fafb]
```

## Clips

Recorded on 2026-09-29 (UTC) on the cluster through `scripts/cluster_run.py`: `cuda`, NVIDIA B200, torch
2.11.0+cu128, Python 3.12.3. All twelve run logs carry the same provenance:
* `games/tron.py` sha256 `5806571a46569d4b` and `games/common.py` `e06078fb69d1275a` (first 16 hex). `commit` and
  `dirty` are null.
  * `tron.py` is the frozen file in the working tree.
  * The working tree's `common.py` has changed since (now `01e0fb9cf3178c67`). Its only difference from the copy in
    the batch-2 run directory is the three-line `SDL_NO_SIGNAL_HANDLERS` change to `run` (Limits).
* Preset `raw`, no instruments. Each brain has one attached module, kind `decoder`: `turn_A` on MaleCNS and `turn_B`
  on FAFB. Their parameters are 33 Hz, 0.30 s re-arm, a 0.10 s side window and a -2.1 Hz side offset. The random
  and never arms have no MaleCNS brain, only `turn_B`.
* Each clip is 40.0 s of brain time: 2,000 frames, 1920x1080, 50 fps (ffprobe). Each console shows one process
  (one start, 40 progress lines) ending in "wrote ... (2000 frames, 40.00 s)".

**How the clips were made, in two batches.**
* **Batch 1** (`tron-rec-05d172`, nine jobs, started 02:57 UTC: for each seed, one job that recorded the fly run and
  then the random arm, one control and one never) was corrupted. The job manager preempted and requeued seven of the
  nine jobs while their first processes kept running (Limits, cluster note). The retries started second writers on the same files. With the owner's approval, the lead killed those
  seven: three cancelled, four exit 137. Their clips are not used. Files from them that were fetched earlier are
  kept apart in `rec1_preempted_discarded/`.
* Two batch-1 jobs, `-7` (control, seed 2) and `-8` (never, seed 2), ran exactly once and finished with exit 0. Each
  has one writer, a complete log and 2,000 frames. They are seed 2's control and never arms, fetched read-only into
  `out/games/tron/batch1_single/` (byte-identical to the earlier copies at `out/games/tron/{control,never}_seed2.*`).
* **Batch 2** (`tron-rec2-1a957b`, seven jobs, started 03:45 UTC) recorded everything else into
  `out/games/tron/rec2/`:
  * the three fly runs, each followed in the same job by the random arm at that run's own turn rate;
  * control and never for seeds 0 and 1.

  Each job first claimed its clip with `mkdir .own_<arm><seed>`. A retried job would then have waited for the first
  writer's log instead of writing. No job was requeued, and the batch ended "7 job(s), 0 failed".
* Both batches ran the same `tron.py` and `common.py`. Each arm was recorded to completion once per seed. Batch 2
  re-ran the seven killed batch-1 jobs (their random arms had not started). None of those had finished or written a
  run log: their consoles stop at 15-31 s of brain time. The partial clips fetched from them are not used.

| clip | arguments after `python games/tron.py` | started (UTC) | brain / wall |
|---|---|---|---|
| `rec2/seed0.mp4` | `--seed 0 --seconds 40 --record out/games/tron/rec2/seed0.mp4` | 03:45:47 | 40.0 / 1367.4 s |
| `rec2/control_seed0.mp4` | `--seed 0 --seconds 40 --control --record out/games/tron/rec2/control_seed0.mp4` | 03:45:47 | 40.0 / 1360.8 s |
| `rec2/never_seed0.mp4` | `--seed 0 --seconds 40 --player never --record out/games/tron/rec2/never_seed0.mp4` | 03:45:47 | 40.0 / 1133.8 s |
| `rec2/random_seed0.mp4` | `--seed 0 --seconds 40 --player random --random-rate 0.373 --record out/games/tron/rec2/random_seed0.mp4` | 04:09:08 | 40.0 / 323.0 s |
| `rec2/seed1.mp4` | `--seed 1 --seconds 40 --record out/games/tron/rec2/seed1.mp4` | 03:45:48 | 40.0 / 1363.5 s |
| `rec2/control_seed1.mp4` | `--seed 1 --seconds 40 --control --record out/games/tron/rec2/control_seed1.mp4` | 03:45:49 | 40.0 / 1358.8 s |
| `rec2/never_seed1.mp4` | `--seed 1 --seconds 40 --player never --record out/games/tron/rec2/never_seed1.mp4` | 03:45:51 | 40.0 / 737.5 s |
| `rec2/random_seed1.mp4` | `--seed 1 --seconds 40 --player random --random-rate 0.168 --record out/games/tron/rec2/random_seed1.mp4` | 04:09:06 | 40.0 / 318.1 s |
| `rec2/seed2.mp4` | `--seed 2 --seconds 40 --record out/games/tron/rec2/seed2.mp4` | 03:45:51 | 40.0 / 847.5 s |
| `batch1_single/control_seed2.mp4` | `--seed 2 --seconds 40 --control --record out/games/tron/control_seed2.mp4` | 02:57:08 | 40.0 / 379.0 s |
| `batch1_single/never_seed2.mp4` | `--seed 2 --seconds 40 --player never --record out/games/tron/never_seed2.mp4` | 02:57:08 | 40.0 / 326.4 s |
| `rec2/random_seed2.mp4` | `--seed 2 --seconds 40 --player random --random-rate 0.231 --record out/games/tron/rec2/random_seed2.mp4` | 04:00:33 | 40.0 / 188.7 s |

Each random rate is `summary.turn_rate_A_hz` of that seed's fly log, which is also in `rec2/random_seed<n>.rate.txt`.
Wall time depends on how many jobs shared a GPU and is not a speed measurement. The fly runs' brain-free baselines
took 13-14 s more (console). Clip time = brain time - 0.02 s: the first frame is drawn after two 10 ms ticks.

**What the fly runs' logs count.** Wall within 1 s means the free run ahead was at most 10 m (1 s at 10 m/s) when the
cycle turned. Into a wall within 1 s means the free run on the new heading was at most 10 m. Crashes count draws;
riding is race time with that cycle alive.

| seed | rounds finished | MaleCNS won / FAFB won / draws | MaleCNS turns (per s of racing) | wall ahead within 1 s | into a wall within 1 s | MaleCNS crashes (per min of riding) | FAFB turns (per s) | wall ahead within 1 s | more than 13 m clear | into a wall within 1 s | FAFB crashes (per min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 (round 5 riding at the end) | 3 / 1 / 0 | 10 (0.373) | 9 | 3 | 1 (2.24) | 28 (1.045) | 17 | 11 | 11 | 3 (6.72) |
| 1 | 5 (round 6 riding) | 2 / 1 / 2 | 4 (0.168) | 4 | 0 | 3 (7.56) | 22 (0.924) | 10 | 12 | 6 | 4 (10.08) |
| 2 | 5 (END hold at the end) | 2 / 1 / 2 | 6 (0.231) | 6 | 1 | 3 (6.93) | 28 (1.078) | 15 | 13 | 10 | 4 (9.24) |
| all | 14 | **7 / 3 / 4** | 20 | 19 | 4 | 7 in 76.6 s (5.49) | 78 | 42 | 36 | 27 | 11 (8.62) |

The riding times were 26.8, 23.8 and 25.97 s, the same for both cycles.

The MaleCNS giant fibre crossed 33 Hz 14, 7 and 7 times. That was 20 turns, 5 crossings inside the 0.30 s re-arm,
and 3 during an END hold after a crash. None came during GET READY. At a turn its rate was 33.0-37.9 Hz.
* 19 of its 20 turns came with a wall 0.5-1.5 m ahead (50-150 ms, median 1.1 m).
* The exception is seed 0 at 25.44 s: 33.8 Hz, 0.73 s after its previous turn, 48.7 m clear. It turned left into
  6.3 m.

FAFB crossed 43, 38 and 41 times. None of those came during GET READY either.

**The decoder turned the MaleCNS cycle before every wall it was heading for; it never rode straight into one.** Its 7
crashes:
* 2 into the FAFB cycle's body, 0.07 and 0.11 s after a turn: round 1 of seeds 0 and 1;
* 4 draws in which the two cycles met. One came 0.04 s after a MaleCNS turn (seed 2, round 1). One came as the
  cycles passed nearly head-on and FAFB's turn swung its body across the MaleCNS cycle, with the MaleCNS DNp01
  peaking at 29.7 Hz (seed 1, round 3). Two were side-on (round 5 of seeds 1 and 2), with DNp01 at or below
  5.3 Hz over the last second;
* 1 into the arena corner, 0.06 s after a left turn into 0.6 m (seed 2, round 2).

FAFB's 11 crashes:
* 3 into the MaleCNS cycle's jet wall;
* 1 into its own trail;
* 3 into the arena wall, each 0.03-0.10 s after a turn into 0.3-1.0 m;
* the 4 draws.

**Seed 0 (MaleCNS 3 : 1).**
* **Round 1** is head to head, offset, and opens the same way on every seed. FAFB turned left at 2.36 s with 39.1 m
  clear, and its jet wall crossed the MaleCNS cycle's path.
  * At 4.17 s the MaleCNS giant fibre crossed (34.1 Hz), 1.3 m (130 ms) before that wall. The cycle turned right,
    with s' +1.06 Hz. That was the side with less room: 23.7 m against 33.7 m on the left.
  * At 6.08 s (33.3 Hz), 1.0 m before FAFB's next wall, it turned right into 36.4 m of free run.
  * 0.07 s later it met the FAFB cycle, which had turned at 6.01 s and was crossing the new path. MaleCNS derezzed at
    6.15 s, and round 1 went to FAFB (4.95 s).
* **Round 2** (4.67 s).
  * The MaleCNS cycle rode straight for 4.56 s. At 13.71 s it turned right, 1.1 m before the arena wall, into 10.7 m;
    the left had 46.7 m.
  * FAFB had turned left at 9.41 s with 44 m clear and again at 12.67 s. At 13.61 s it turned right into 2.0 m.
  * 0.21 s after that turn, at 13.82 s, FAFB hit the MaleCNS cycle's jet wall, 3.8 m from that cycle.
    MaleCNS took the round.
* **Round 3** (9.95 s, the longest round of the three runs; MaleCNS 6 turns, FAFB 17).
  * MaleCNS turned at 20.56, 23.29 and 24.71 s, each 0.7-1.4 m before a wall.
  * At 25.44 s came its one turn far from any wall (above).
  * It turned again at 26.02 s (0.5 m before FAFB's jet wall, into 5.9 m) and 26.54 s (0.7 m before its own, into
    7.0 m), in a pocket of its own and FAFB's jet walls near the arena's corner.
  * FAFB hit its own trail at 26.77 s, 24.9 m from the MaleCNS cycle. MaleCNS took the round.
* **Round 4.** MaleCNS turned once: at 34.33 s, 1.1 m before a wall, right into 40.7 m. FAFB turned left at 35.98 s
  with 49.1 m clear, into 1.0 m, and hit the arena wall at 36.08 s. MaleCNS took the round.
* Round 5 had been racing for 0.92 s when the clip ended.

**Seed 1 (MaleCNS 2 : 1, 2 draws).**
* **Round 1** played out like seed 0's. MaleCNS turned right at 4.20 s (1.0 m ahead, into 23.7 m) and again at
  6.10 s (1.1 m ahead, into 36.7 m). It met the FAFB cycle 0.11 s later, and round 1 went to FAFB.
* **Round 2.** MaleCNS turned right at 13.75 s (1.3 m before the arena wall) and at 14.70 s (1.2 m, into 56.1 m).
  FAFB turned seven times and hit the MaleCNS cycle's jet wall at 16.61 s.
* **Round 3** was a draw at 22.36 s, 2.75 s in. The cycles were passing nearly head-on, 0.8 m apart centre to
  centre, when FAFB's decoder turned the orange cycle right (32.5 m clear ahead) and its turned body overlapped the
  MaleCNS cycle on the same tick. The MaleCNS DNp01 peaked at 29.7 Hz and did not cross 33 Hz before the crash (it
  crossed at 22.37 s, in the END hold).
* **Round 4.** MaleCNS never turned. FAFB turned seven times. At 29.33 s it turned right into 0.6 m, and it hit the
  arena wall at 29.39 s.
* **Round 5** was a draw at 35.59 s: the orange cycle came in from the MaleCNS cycle's left and they met. In the frame
  at clip time 35.56 s (brain 35.58 s) the orange cycle fills part of the MaleCNS left eye, while its DNp01 reads
  0.0 Hz (at most 5.3 Hz over the last second).
* Round 6 was racing at the end.

**Seed 2 (MaleCNS 2 : 1, 2 draws).**
* **Round 1.** MaleCNS turned right at 4.15 s (1.5 m ahead) and 6.07 s (0.9 m ahead, into 36.2 m). The cycles met
  0.04 s later: a draw at 6.11 s.
* **Round 2.** At 13.72 s MaleCNS turned right, 0.6 m before the arena wall, into 10.7 m. At 14.66 s its giant fibre
  crossed again (36.6 Hz, s' -0.03 Hz), 1.3 m before the side wall. It turned left into the corner, 0.6 m, where the
  right had 56.8 m. It hit the arena wall 0.06 s later, and round 2 went to FAFB.
* **Round 3.** MaleCNS turned left at 22.87 s (1.2 m ahead; 28.7 m on either side). FAFB turned right at 23.44 s
  with 45.9 m clear, into 2.6 m, and hit the MaleCNS cycle's jet wall at 23.71 s.
* **Round 4.** MaleCNS turned once: at 31.28 s, 1.0 m before a wall, right into 31.25 m. FAFB turned 15 times and hit
  the arena wall at 32.95 s, 0.03 s after a right turn into 0.3 m.
* **Round 5** was a draw at 39.17 s: side-on again, with DNp01 at 0.1 Hz (at most 4.6 Hz over the last second). The
  clip ended in the END hold.

**Controls: the decoder read, not applied.**
* The cyan cycle rides the never-turning player's path. On all three seeds it won 0 rounds, lost 4 and drew 1. That
  is 5 crashes in 23.8 s of riding, 12.6 per minute.
* The decoder would have turned 4, 5 and 4 times: 13 would-be turns, all 0.8-4.3 m before a wall, 12 of them to
  the right.
* In rounds 1-4 of each control, 12 rounds in all, the cyan cycle hit the wall ahead. The last would-be turn came
  0.08-0.16 s (0.8-1.6 m) before each impact.
* Round 5 (crossing paths) of each control was the side-on meeting, with DNp01 at or below 5.1, 5.3 and 8.5 Hz over
  the last second.
* Control seed 2 (batch 1) is the only MaleCNS run whose giant fibre crossed during GET READY. It did so three times
  (14.67, 22.08 and 22.55 s; 34.4, 43.1 and 33.8 Hz). These crossings are logged `not_live`.

**Never and random arms (no brain on the cyan cycle; FAFB closed loop).**
* Never: 0 rounds won in all three. 5 crashes in 23.8 s each (12.6 per minute).
* Random at 0.373, 0.168 and 0.231 turns per second:
  * 7, 3 and 6 turns;
  * 0, 1 and 0 rounds won;
  * 5, 4 and 5 crashes (12.6, 10.0 and 12.6 per minute).

**Ranking, closed-loop clips** (one run per arm, one device type; lower crashes per minute is better):

| seed | fly: crashes / min, won-lost-drawn | random | never | control (decoder read, not applied) |
|---|---|---|---|---|
| 0 | **2.24, 3-1-0** | 12.61, 0-5-0 | 12.61, 0-4-1 | 12.61, 0-4-1 |
| 1 | **7.56, 2-1-2** | 10.03, 1-4-0 | 12.61, 0-5-0 | 12.61, 0-4-1 |
| 2 | **6.93, 2-1-2** | 12.61, 0-5-0 | 12.61, 0-5-0 (batch 1) | 12.61, 0-4-1 (batch 1) |

On every seed the fly run had the fewest crashes per minute of riding and won the most rounds. The orange FAFB is
closed loop in these clips, so its game differs between arms (Limits).

**Ranking against 50 brain-free random players** (`summary.baselines`: the same seed, arena, schedule and tick
timeline, with the orange cycle replaying its recorded turns open loop):

| seed | the fly: crashes / min (rounds won) | never | 50 random at the fly's rate: crashes / min min, median, max | draws with fewer / equal / more crashes per min | draws' rounds won: range (median) | draws winning more / equal / fewer rounds | the fly's place among 52 (fly, never, 50 random) on crashes / min |
|---|---|---|---|---|---|---|---|
| 0 | 2.24 (3) | 10.08 (1) | 5.04, 10.08, 14.42 | 0 / 0 / 50 | 0-3 (1) | 0 / 1 / 49 | 1st |
| 1 | 7.56 (2) | 13.09 (1) | 4.48, 10.49, 17.31 | 3 / 5 / 42 | 0-3 (1) | 2 / 19 / 29 | 4th, tied with 5 draws |
| 2 | 6.93 (2) | 12.61 (0) | 6.72, 12.61, 17.31 | 2 / 0 / 48 | 0-2 (0) | 0 / 2 / 48 | 3rd |

Pooled over the 150 random draws:
* 5 crashed less per minute than the fly, 5 as often, and 140 more.
* 2 won more rounds than the fly, 22 as many, and 126 fewer.

The never-turning player crashed more often than the fly on every seed. On each seed, the replay of the fly's own
turns reproduced the run's rounds (`replay_matches_fly: true`). A CPU recompute with `--baselines-from` reproduced
each logged `summary.baselines` exactly.

**What this shows.** A turn fired by the giant fibre's crossing comes about a metre before a wall. With that timing
the cycle crashed less often per minute of riding than the closed-loop random arm on every seed, and than 140 of the
150 brain-free random draws. Its round wins came from FAFB's decoder turning the orange cycle into walls. The side it
picks does not beat a coin (Limits), and the pathway did not see the other cycle coming from the side. The
comparison that says whether the connectome adds anything beyond a range finder was run after the recordings
(below): a one-line rule that turns 1.1 m before any wall matches the decoded cycle on two seeds and beats it on the
third when it flips a coin for the side, and beats it on every seed when it turns towards the roomier side. The trailer and thread may say "the cycle turns when its giant
fibre fires, about a metre from the wall", "crashed less than riders turning at random times" and "no better than a
rule that turns 1.1 m before a wall and flips a coin". They may not say "dodges", "outsmarts", "chooses the open
side" or "beats a simple distance rule".

**Against a distance rule (added after the recordings; brain-free, CPU).** `games/tron_rules.py` puts a GAME rider in
the cyan seat that turns when the free run ahead is at most 1.1 m (or 1.5 m), with the fly decoder's 0.3 s re-arm,
and chooses its side by rule: towards the longer free run (`room`), always right (`right`), or a fair coin (`coin`, 50
draws per seed). The orange cycle's recorded turns are replayed open loop, exactly as for the logged random baselines.
The rider knows the arena's geometry, which the fly does not. Output: `out/games/tron/rule_baselines.json`. Crashes
per minute of riding (rounds won in brackets):

| seed | decoded fly | 1.1 m, room | 1.1 m, right | 1.1 m, coin (median of 50; draws with fewer crashes than the fly) | 1.5 m, room | 1.5 m, coin (median; fewer than the fly) |
|---|---|---|---|---|---|---|
| 0 | 2.24 (3) | 0.00 (4) | 2.24 (3) | 2.24 (3); 21 fewer, 14 equal | 0.00 (4) | 2.17 (3); 26 fewer, 16 equal |
| 1 | 7.56 (2) | 5.04 (3) | 7.56 (2) | 7.56 (2); 13 fewer, 30 equal | 5.04 (3) | 6.30 (2.5); 25 fewer, 22 equal |
| 2 | 6.93 (2) | 0.00 (4) | 2.24 (3) | 2.24 (3); 47 fewer, 0 equal | 0.00 (4) | 2.15 (3); 47 fewer, 0 equal |

This comparison was proposed by the caption skeptic after the recordings and run by the lead; it did not exist when
the code was frozen, so it is post-hoc, and it replays the recorded orange cycle rather than a closed-loop FAFB brain.

**Trailer cuts, checked against frames of `rec2/seed0.mp4`** (clip time; brain time = clip time + 0.02 s; frames
in `out/games/tron/rec2_frames/`):
* **2.9-4.5 s: accurate.**
  * The MaleCNS banner "GIANT FIBRE 34 Hz → TURN RIGHT · MaleCNS · 1.3 m to the wall ahead" fades in at 4.16 s. It is
    opaque by 4.24 s and still up at 4.5 s. The wall ahead is FAFB's jet wall.
  * At 2.9 s FAFB's earlier banner ("44 Hz → TURN LEFT · 39 m to the wall ahead") is still fully up. It fades
    from 2.94 s and is gone by 3.74 s. That FAFB turn came with 39 m clear: no wall loomed.
  * The HUD's live DNp01 readout shows 31.5 Hz at 4.24 s. The banner gives the crossing value.
  * The turn went toward less room (23.7 m against 33.7 m).
* **9.4-14.1 s: accurate.**
  * 9.4 s shows GO and FAFB's "44 m" turn banner fading in.
  * At 13.76 s both turn banners are up: FAFB "36 Hz → TURN RIGHT · 3.9 m" and MaleCNS "33 Hz → TURN RIGHT · 1.1 m".
  * By 14.1 s, "FAFB DEREZZED · MaleCNS takes round 2" (1 : 1) is up, with orange sparks beside the cyan wall.
  * FAFB's crash followed its own right turn into 2.0 m. The MaleCNS turn, 1.1 m before the arena wall and toward less
    room, did not cause it; FAFB hit the wall the MaleCNS cycle had laid on its straight run.
* **26.3-27.2 s: accurate.**
  * At 26.3 s: "MaleCNS 37 Hz → TURN LEFT · 0.5 m" and "FAFB 39 Hz → TURN RIGHT · 0.5 m".
  * At 26.6 s: MaleCNS "35 Hz → TURN RIGHT · 0.7 m", under FAFB's "36 Hz → TURN RIGHT · 0.6 m".
  * From 26.76 s: "FAFB DEREZZED · MaleCNS takes round 3" (2 : 1).
  * FAFB hit its own trail 24.9 m away, off camera. Only the minimap's cross and the banner show it. The MaleCNS turn
    did not cause it.
* **6.1-6.5 s: accurate; it shows a decoder turn followed 0.07 s later by the MaleCNS cycle's own crash.**
  * MaleCNS derezzed at 6.15 s, 0.07 s after its right turn (1.0 m before FAFB's wall), by meeting the FAFB cycle.
    The turn took it into the path of the FAFB cycle, 2.9 m away and crossing.
  * At 6.1 s FAFB's banner "35 Hz → TURN RIGHT · 34 m to the wall ahead" (its 6.01 s turn, no wall near) sits above
    the MaleCNS "33 Hz → TURN RIGHT · 1.0 m" banner.
  * "MaleCNS DEREZZED · FAFB takes round 1" (0 : 1) fades in from 6.14 s.
  * After the crash, the MaleCNS DNp01 climbs to 52 Hz (6.5 s frame) during the END hold, which is not live. It is not
    a reaction to anything the cycle could still do.
* **The lower third "a wall looms → the fly's escape pathway → a declared decoder turns the cycle"** fits the MaleCNS
  turns in these shots: 4.17 s (1.3 m), 13.71 s (1.1 m), 26.02 and 26.54 s (0.5 and 0.7 m) and 6.08 s (1.0 m). It
  does not fit the FAFB banners in the same shots: 39 m clear at 2.9-3.7 s, 44 m at 9.4-10.8 s and 34 m at 6.1 s.
  Over 6.1-6.5 s the turn it describes is followed 0.07 s later by the MaleCNS cycle's own derezz, so the shot shows
  the decoder's turn failing, not an escape. In 9.4-14.1 and 26.3-27.2 s the round is won by FAFB's own crash, which
  the MaleCNS turn did not cause.

## Corrections (caption skeptic)

Checked against all twelve run logs (summaries and every event), the cluster logs and command files of both batches,
every console (including the discarded batch-1 ones), ffprobe of all twelve clips, frames of `rec2/seed0.mp4` at
2.90-4.50, 6.04-6.50, 9.40-14.10 and 26.30-27.20 s and of `rec2/seed1.mp4` at 22.34 and 35.56 s (clip time), and
`games/tron.py`. Brain-free CPU replays (`CUDA_VISIBLE_DEVICES=-1`) of the three fly runs, the three controls and two
never arms reproduced every logged round and gave the geometry of every MaleCNS turn and crash. A CPU
`--baselines-from` recompute of the three `summary.baselines` matched them exactly. Source hashes match the logs;
the working tree's `common.py` without its three `SDL_NO_SIGNAL_HANDLERS` lines hashes to the logged
`e06078fb69d1275a`.
* Headline: "The MaleCNS cycle turns only when its giant fibre crosses 33 Hz" became "A declared decoder turns the
  MaleCNS cycle only when its giant fibre crosses 33 Hz". A decoder turns the cycle (ground rules 2 and 6).
* Headline: "it took 7 rounds to FlyWire FAFB's 3" left out the 4 draws and why the rounds were won. It now adds
  "with 4 draws; each of the 7 ended with FAFB's own decoder turning the orange cycle toward a wall it then hit". Each
  of the 7 came 0.03-0.31 s after an FAFB turn into 0.3-3.1 m of free run. In one of them (seed 1, round 4) the
  MaleCNS cycle never turned. The same fact is now in the opening paragraph and "What this shows".
* Headline: "140 of 150 brain-free random riders" became "140 of 150 brain-free riders turning at random times at its
  own rate; no rider that turns a metre before every wall was run". The ranking's real limit now appears in the
  headline and not only in the body.
* Opening paragraph and the bold line above the crash list: "it never rode into a wall it was heading for without
  turning first" became "the decoder turned it before every wall it was heading for; it never rode straight into
  one".
* DECODER row: the caption gave only the dev figure for turns toward the momentarily more-looming eye (4 of 10). It
  now adds the recorded one: 12 of the 14 right turns went toward the eye whose LPLC2 + LC4 was larger on the turn's
  tick.
* "not claimed" row and Limits: "FAFB's crossings are mostly spontaneous: 36 of its 78 turns came with more than
  13 m clear" became "Many ... 36 of its 78 turns (46 %)". 36 of 78 is under half; the other 42 came with a wall
  within 10 m.
* "not claimed" row, Limits, the crash list and seed 1, round 3: the one near head-on meeting was left out. The
  cycles were passing 0.8 m apart centre to centre when FAFB's decoder turned (32.5 m clear ahead), and its turned
  body overlapped the MaleCNS cycle on the same tick. The MaleCNS DNp01 peaked at 29.7 Hz. "and never crossed" became
  "did not cross 33 Hz before the crash (it crossed at 22.37 s, in the END hold)". "stayed at or below 8.5 Hz" became
  "... over the last second" (`gf_max_last_1s`).
* Probe summary: "The FAFB giant fibre crossed 33 Hz one to five times per 3 s run ... its turns are mostly not
  wall-driven" became "In the game's own look ... two to six times per 3 s run ... many of its turns are not
  wall-driven". The table row for that look says 2-6, and no look gives 1-5.
* Seed 0, round 3: "in a pocket of its own jet walls" became "of its own and FAFB's jet walls". At 26.02 s the wall
  0.5 m ahead was FAFB's (replay).
* Limits, opponent: "because there the cyan cycle went out early in every round" was false. The random cyan won
  seed 1, round 3, and random seed 2's round 3 lasted 10.37 s, longer than any fly-run round. The sentence now reads
  "there the cyan cycle crashed in 44 of the 45 finished rounds (the exception: the random arm, seed 1, round 3)".
* Limits, cost: "with up to seven jobs at once" became "up to nine of these jobs running at once (batch 1 ran all
  nine together)". `rec_cluster.log` shows all nine running from 620 s.
* Cluster note: "The first attempts' ffmpeg failed at close ("ffmpeg failed writing ..."), and the console files hold
  both attempts' lines". The ffmpeg message is in no fetched file. The note now says what the fetched batch-1
  consoles show: the second attempt's opening lines, then NUL bytes, then the first attempt's later lines. None of
  them reached "wrote".
* Clips, batch 1: "nine jobs, one per arm and seed" became "for each seed, one job that recorded the fly run and then
  the random arm, one control and one never". Nine jobs covered twelve arm-seeds (`rec_cluster.commands.txt`).
* Clips: "Each arm ran once per seed, and nothing was rerun" was false as written, because batch 2 re-ran the seven
  killed batch-1 jobs. It now says so. None of those jobs had finished or written a run log: their consoles stop at
  15-31 s of brain time, and their random arms never started.
* Seed 1, round 5: "In the frame at 35.56 s" became "at clip time 35.56 s (brain 35.58 s)". The narrative's other
  times are brain times.
* "What this shows": "That timing beat the closed-loop random arm ... and 140 of the 150" now names the metric
  (crashes per minute of riding). It adds "The decoded cycle is shown to beat random timing only". Allowed thread
  wording "beat random turning" became "crashed less than riders turning at random times". "turns when" became "the
  cycle turns when". "beats a simple distance rule" joined the forbidden phrases. Limits, "No distance-rule
  baseline": "The fly beats random turning" became "The decoded cycle beats random turning at its own rate, and that
  is all it is shown to beat".
* Trailer, 2.9-4.5 s: "At 2.9 s FAFB's earlier banner ... is still fading out" was wrong. At 2.90 s it is fully up;
  the banner holds 0.6 s after 2.36 s brain time. It fades from 2.94 s and is gone by 3.74 s. The caption now adds
  that this FAFB turn came with 39 m clear.
* Trailer, 26.6 s: added FAFB's "36 Hz → TURN RIGHT · 0.6 m" banner, which is on screen beside the MaleCNS one.
* Trailer, 6.1-6.5 s: added that the turn took the cycle into the path of the FAFB cycle, 2.9 m away, and that FAFB's
  "34 m" banner is up at 6.1 s. The heading now says the shot shows a decoder turn followed by the MaleCNS cycle's
  own crash.
* Trailer: added the lower-third check ("a wall looms → the fly's escape pathway → a declared decoder turns the
  cycle"). It fits the MaleCNS turns in all four shots. It does not fit the FAFB banners shown with them (39, 44 and
  34 m clear). Over 6.1-6.5 s it would caption a failed turn.
