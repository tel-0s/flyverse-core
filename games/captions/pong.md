# pong: the first game of Pong between two connectomes

**The first game of Pong between two connectomes. Each paddle follows a decoder's reading of its own fly's L2 and
Mi1 columns. Rallies reach 12 returns and 200°/s. The reconstruction with 3.7x the photoreceptors facing the court
takes 19 of the 20 points on an RTX 4090. On a B200 it took 17 of 18 from the same end and 13 of 15 with the ends
swapped.**

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the flies) | MaleCNS v1.0 (167,106 neurons) and FlyWire FAFB v783 (139,255 neurons), both as shipped: preset `raw`, nothing instrumented, attached, trained or tuned. The model is everything between the arena's light reaching the photoreceptors and the L2 / Mi1 rate deviations the decoders read. The HUD's photoreceptor counts come from each reconstruction's retina, and "L2 mean Δr" is the model's rate. |
| **DECODER** (written for the game, declared) | `ball_decoder_A` / `_B`, one law and one set of parameters for both flies. For each L2 and Mi1 cell it takes \|dr − EMA(dr, 0.15 s)\| and keeps the 12 strongest cells. The activity-weighted mean azimuth of their densest 12° cluster is the reading. The paddle's target follows the reading (EMA 50 ms) and holds while the top cell is below 0.40. It is read-only (`fb.optic.last["dr"]`) and writes nothing. The high-pass, the gate, the cluster and the target EMA were chosen on dev seeds 101-108. The top 12 and the L2 / Mi1 population come from the PLAN's prototype (Design, "Which parameters were chosen where"). |
| **GAME** (not the fly) | The court and the LED arena each fly is shown: the ball is a 12° disc, and the paddles are not shown to the flies. The ball physics: x1.15 per return up to a 200°/s cap, a 20° minimum return angle, and english. The serve schedule (the same for every seed), the paddles' 80°/s speed limit, the score, the banners and the blind control. |
| **not claimed** | Neither fly plays, aims, tracks, moves, learns or is rewarded. No motor neuron is read (FAFB has no VNC). The raw model does not pass small objects to LC10a / LC11: locating the ball is the decoder's work, on cells one or two synapses from the photoreceptors. Not a sex comparison: the difference in misses is consistent with how many photoreceptors each reconstruction placed facing the court (MaleCNS 747, FAFB 2,753). With the ends swapped the misses stayed with MaleCNS: on one NVIDIA B200 node it made 13 of the 15 misses from the right end, and 17 of the 18 from the left in a same-ends replicate run beside it. The swap does not separate the brain from its photoreceptor coverage, and three runs per end are not a significance test (Limits). |

## Design

Two connectomes, one court. On the left, MaleCNS v1.0 (a male CNS, `FlyBrain(seed=s)`, 167,106 neurons); on the
right, FlyWire FAFB v783 (a female brain with the complete optic lobe, `FlyBrain(dataset="fafb", seed=s)`, 139,255
neurons). These are the default ends; `--left fafb --right malecns` swaps them (Clips, "Sides swapped"). Both
`preset="raw"`, nothing instrumented, nothing attached. Only the eyes are used: FAFB has no VNC and
nothing here reads a motor neuron. Each paddle follows a decoder's reading of its own fly's L2 (lamina) and Mi1
(medulla) columns. Neither fly plays, aims or moves anything; the flies' only link to the game is what their eyes are
shown, which includes where the ball goes after the paddles meet or miss it.

**Not a sex comparison.** One male and one female reconstruction, which differ in how many photoreceptors were
reconstructed facing the court (MaleCNS 747, FAFB 2,753; below), in proofreading and in naming. The main view says so
on every frame: under each name it prints the brain's neurons and its photoreceptors facing the court, with the line
"not a sex comparison: one male and one female reconstruction, with different photoreceptor coverage". The title card
repeats it with the two counts.

| what | kind | law and parameters |
|---|---|---|
| the brains | CONNECTOME | both stepped 10 ms per game tick (`fb.step(10)`); the brain seed is the run's `--seed` for both. Before the clip each fly watches the empty court for 1.0 s (warm-up, logged), so the photoreceptors have adapted when the first frame is drawn. |
| the arena (what each fly sees) | GAME | the court as an LED arena in the fly's own field: court degrees (X along the court 0-70, Y across it +-50) map to (az, el) = (Y, X - 35) for the left fly and (-Y, 35 - X) for the right one, so each fly's own end is at el -35, the far end at el +35, and the court fills az +-50 x el +-35. Radiance [UV, B, G, R] given directly: court (0.05, 0.12, 0.16, 0.12), its edges and a dashed net line (0.16, 0.30, 0.34, 0.30), outside (0.015, 0.025, 0.03, 0.03), the ball (0.8, 1, 1, 1), a disc of 6 deg radius (about three ommatidia across) with a 1 deg soft edge. 7 rays per column, `Eyes.pool`. **The paddles are not shown to the flies** (measured below). |
| `ball_decoder_A` / `_B` | DECODER | read-only, from `fb.optic.last["dr"]` (the optic lobe's rate-unit deviations), declared with `common.declare` rather than attached. Cells: every L2 and Mi1 rate unit whose retinotopic column (`flyverse.interp.trace.column_of_cells`, on that fly's own connectome and retina) lies within az +-55 x el +-40: MaleCNS 1,089 cells, FAFB 1,078. Per cell `hp = \|dr - EMA(dr, 0.15 s)\|`; the 12 cells with the largest `hp`; among them the densest cluster (for each of the 12, the `hp`-weighted count of the 12 within 12 deg of it; the best one's neighbourhood); the `hp`-weighted mean azimuth and elevation of that cluster's columns is this tick's **reading**. The **target** follows the reading (EMA 50 ms) and holds while the strongest cell's `hp` is below 0.40. The paddle follows the target. The same law and parameters for both flies. |
| paddle | GAME | moves toward its target (its court Y = the target azimuth, sign-flipped for the right fly) at up to 80 deg/s; half-length 12 deg; centre clamped to +-38 deg (the court edge). After a miss the paddle that missed stops where it was until the ball has left the court (the point is already decided), so an escaping ball is never drawn meeting a paddle that has moved into it. |
| court | GAME | constant-speed ball, reflecting side walls. When the ball reaches a paddle line it is returned if \|ball Y - paddle Y\| <= 12 + 6 deg (paddle half + ball radius), otherwise it is a miss and the other side scores when the ball leaves the court. A return keeps half the incoming angle, adds english = 35 deg x the hit offset (-1 .. 1 along the paddle), caps the angle at 55 deg, and **never leaves flatter than 20 deg**: a flatter return is steepened to 20 deg in the direction it was going (the offset's if exactly flat, then away from the nearer side wall). It multiplies the speed by 1.15, up to 200 deg/s. Serve speed 52 deg/s. All speeds are in degrees of the flies' arenas. The collision radius along the court (1.8 deg of X) and across it (6 deg of Y) is the ellipse the main view draws. |
| serves | GAME | a fixed schedule of 16 (receiver, angle, start Y) entries, cycled, the same for every seed; the ball is put in play at the server's end line, 1.5 s into the clip (the title card; the flies watch the empty court) and 1.0 s after each point (the ball is hidden in between). |
| score, rally, banners | GAME | from the court's events: POINT (score and the miss distance), RALLY at 5, 10, 15 ... returns (showing the live count and ball speed while the rally lasts), GAME when `--points` ends a game. Every serve, return (hit offset, speed, outgoing angle, whether it was steepened, paddle - ball, target - ball), miss (distance, paddle - ball, target - ball, speed, sideways speed, photoreceptors and L2 cells under the ball) and point is in the run log. |
| ribbons beside the court | DECODER (display) | the decoder's input: max `hp` of its L2 + Mi1 cells per 4-deg azimuth bin, the last 3.2 s, rows aligned with the court's Y; white dots: the ball's Y; teal line: the decoder's target. |
| "photoreceptors under the ball", "L2 mean Δr" | CONNECTOME (display) | for the columns within 6 deg of the ball's true direction: how many reconstructed photoreceptors look there (a count from the reconstruction, `retina.pr_column`), and the mean `dr` of the L2 cells whose column looks there (L2 hyperpolarises under the bright ball; bar full at -0.5, the rate floor for L2's 0.5 baseline). Drives nothing. |
| "photoreceptors face the court" | CONNECTOME (display) | reconstructed photoreceptors whose column lies in the court window az +-50 x el +-35: MaleCNS 747 (281 columns, median 2 per column), FAFB 2,753 (428 columns, 9 without a photoreceptor, median 8). Also in `meta.court_coverage`. |
| \|target − ball\| | DECODER (display) | median over the last 3 s of live frames of the decoder target's azimuth error, the quantity the paddle follows. The run log also reports the reading's error (before the 50 ms EMA and the gate). |
| WHAT THE FLY SEES | GAME (display) | the frontal +-55 x +-40 deg of the exact radiance handed to `fb.vision` (`eye_colors`, human view, exposure 1.2), one hexagon per column at its own azimuth and elevation, both eyes; columns without a reconstructed photoreceptor are dark (FAFB, 51 of 1,581); MaleCNS's 1,466 columns are only those with photoreceptors, so its gaps are black. Teal ring: the decoder's target; bar: the paddle, which the fly does not see. |
| control arms | GAME | `--control blind`: the ball is never drawn in either arena (the flies see the empty court); decoders and paddles run unchanged. `--control frozen`: decoders disconnected, both paddles held at the centre. Every run log also carries `summary.game_floor_frozen_paddles`, the frozen-paddle game computed without a brain for the same duration (deterministic). |

Main view: the court is 1,564 x 700 px of the 1920 x 1080 canvas (52.8 %), 63.5 % with the two decoder ribbons that
share its rows (`meta.display.layout_px`).

### Run log (what the seeds' captions are written from)

`summary` is keyed by side and brain (`A_MaleCNS`, `B_FAFB`): the score, returns per point, the longest rally, the
median outgoing return angle and how many returns were steepened, the GAME floor, and per fly: returns and misses;
the decoder **target** error (`decoder_target_abs_err_*`) and **reading** error (`decoder_reading_abs_err_*`), median
and 90th percentile, overall and with the ball in the fly's own half; the error broken down by the number of
photoreceptors under the ball (`decoder_err_by_photoreceptors_under_ball`: 0, 1-3, 4-8, 9-20, 21+); the fraction of
live time the target was held, the ball was over no L2 column, and the paddle sat at the court edge.

### What was measured first (dev seeds >= 100)

The builder's development measurements (scratch scripts on dev seeds; exploratory, not in `out/`):

* **The two eyes are not the same instrument.** Within the court window MaleCNS v1.0 has 281 columns carrying 747
  reconstructed photoreceptors (median 2 per column), and its central 40 x 30 deg has 16 columns with 17
  photoreceptors; FAFB has 428 columns (9 without a photoreceptor) carrying 2,753 (median 8 per column), 97 columns
  and 655 photoreceptors in the same centre (the review re-derived these from the retina). MaleCNS's frontal gap is the
  black hole in its mosaic. This is the reconstructions' photoreceptor coverage, not a property of either animal.
* **Signal to noise, per cell type** (seed 101, a 9-deg ball bouncing open-loop through the arena): the strongest
  high-passed cell within 7 deg of the ball against the 99th percentile of cells 25 deg or more away. FAFB L2 2.6x
  (the ball above that noise in 100 % of frames), Mi1 2.7x (100 %); MaleCNS L2 1.5x (66 %), Mi1 1.7x (72 %). (The
  population centroid of T4 / T5 does not localise the ball: correlation 0.03 in the PLAN's prototype, not
  re-measured.)
* **The decoder on each connectome.** Open loop, the PLAN's first law (top-12 L2 cells, weighted mean, EMA 0.3 s):
  FAFB azimuth r = 0.98, median error 3.9 deg (seed 100); MaleCNS r = 0.77-0.90, median 6.9-11.4 deg (seed 100, two
  trajectories). Adding Mi1 to L2 helped MaleCNS (r 0.92 and 0.94). The cluster step (the densest 12-deg cluster of
  the 12 instead of their mean) cut MaleCNS's median error in its own half from 17.6 to 6.7 deg on seed 101 but
  lengthened the tail (90th percentile 30.7 -> 36.7 deg).
* **Paddles visible or not.** Closed loop on seed 101 with both paddles drawn in the arenas: FAFB's median error with
  the ball in its own half rose from 5.9 to 9.5 deg and its elevation correlation fell from 0.91 to 0.33; MaleCNS's
  elevation correlation fell from 0.63 to 0.22. The moving paddles become the peak the decoder reads, and each
  paddle would partly follow itself. The flies therefore see only the court and the ball.
* **Polarity.** Under the ball, L2 hyperpolarises and Mi1 depolarises (mean `dr - EMA(0.3 s)` of the cells within 5
  deg: L2 -0.32 to -0.33 FAFB, -0.11 to -0.12 MaleCNS; Mi1 +0.06 to +0.09; seeds 104, 108). Reading only those signs
  sharpened FAFB but did not help MaleCNS (its share of errors > 16 deg near its end line 0.11-0.26 against 0.13).
  Both flies keep the unsigned law.
* **Which parameters were chosen where.** `top_k` 12 and the L2 / Mi1 population were carried over from the PLAN's
  seed-0 prototype and were not re-tuned. The high-pass EMA (0.3 -> 0.15 s), the gate (0.40, about MaleCNS's 20th
  percentile of the top cell's `hp`; FAFB's is 0.71, so FAFB almost never holds), the 12-deg cluster and the 50 ms
  target EMA were chosen on seeds 101-108 (offline on the decoder cells' `dr` recorded through two 30-s games, seeds
  104 and 108, and closed loop on 102 and 103).

The review of the builder's version (seed 111, `out/games/pong/dev.json`, score 0-7) found that its headline
21-return rally at the 200 deg/s cap was a GAME attractor, not tracking: with no minimum angle, returns off a paddle
parked at the court edge flattened to about 7 deg (25 deg/s sideways at the cap), the ball skimmed the top wall, and
both paddles sat at their +38 deg clamp (49 % and 59 % of that rally's frames). The 20-deg minimum return angle
(declared above) removes it. A unit test (`test_the_flat_rally_attractor_is_gone`: paddles that follow the ball
perfectly, a flat serve near the top wall) shows both arms: without the rule the returns flatten to 0 deg at the cap
and the ball never leaves the top band; with it every return at the cap moves at least 68 deg/s sideways
(200 sin 20 deg; up to 164 deg/s at the 55-deg cap, against the paddles' 80 deg/s limit) and the ball crosses the
court. Offline (GAME only, no brain), with the decoder errors the review recorded on seed 111 replayed onto lagged
paddles, the 20-deg rule kept rallies of up to 9-12 returns and 4-5 points per 40 s over eight replays.

**With the pre-freeze code (`games/pong.py` sha256 `8a77ce3f2ff54bb0`; the recorded code differs from it only in
display, see Code provenance), seed 111, 40 s** (`out/games/pong/dev3.mp4`, `dev3.json`): MaleCNS 0 - FAFB 5, returns per
point 6, 3, 10, 9, 2 (3 in progress at the end), longest rally 10 (one return at the 200 deg/s cap). MaleCNS 15 returns
/ 5 misses, FAFB 18 / 0. Median outgoing angle 22.8 deg; 9 of 33 returns steepened to 20 deg. The paddles sat at the
court edge for 11 % (MaleCNS) and 12 % (FAFB) of live time. Decoder error, median (90th percentile), overall: target
MaleCNS 6.9 (19.3) deg, FAFB 5.3 (9.4) deg; reading MaleCNS 6.0 (31.5) deg, FAFB 4.0 (7.3) deg. The target reads a few
degrees behind a moving ball (the 50 ms EMA), so its median is above the reading's; the gate and EMA shorten
MaleCNS's tail. MaleCNS held its target 18 % of live time, FAFB 1 %; the ball was over no MaleCNS L2 column 7 % of the
time. **Error against the photoreceptors under the ball** (target median, reading median, live ticks): MaleCNS 0: 11.7,
16.6 deg (246); 1-3: 9.2, 11.5 (676); 4-8: 8.7, 8.5 (937); 9-20: 5.5, 4.2 (1,190); 21+: 4.4, 3.5 (301). FAFB (the
ball is almost always over 21+): 5.3, 4.0 (2,940). Where MaleCNS has FAFB's coverage, its error is at or below FAFB's.
**MaleCNS's five misses:** at 7.41, 12.57, 30.58 and 35.43 s the decoder's target was within the paddle's reach
(18 deg) of the ball at the moment of the miss (target - ball -3.8, -9.6, -13.7, +1.9 deg; paddle - ball -22.9, -21.5,
-21.3, +26.9 deg): the target had moved late and the 80 deg/s paddle had not caught up. At 21.0 s (200 deg/s) the
target itself was 28 deg off.

**Blind control, seed 111, 40 s** (`out/games/pong/dev3_blind.mp4`, `dev3_blind.json`; an independent draw, see
Reproducibility): the ball is never drawn in either arena. 6 - 6 over 12 points, returns per point 0, 0, 3, 0, 0, 1,
0, 1, 1, 2, 0, 0, so 8 returns (4 each) against 12 misses (6 each), against 33 returns and 5 misses with the ball
shown. Decoder target error, median (90th percentile): MaleCNS 19.5 (41.2) deg, FAFB 17.6 (37.6) deg; reading
27.5 (57.4) and 19.6 (44.5) deg. Held 52 % (MaleCNS) and 76 % (FAFB) of live time. Without the ball the error no longer
falls with the photoreceptors under it (MaleCNS target medians 22.8, 14.3, 17.6, 23.6, 35.6 deg across the five
bins). The blind arm is close to the GAME floor below (6 returns / 13 misses).

**GAME floor** (paddles frozen at the centre, no brain, `pong.game_floor(40)`, deterministic): 6 returns (MaleCNS's
side 2, FAFB's 4) against 13 misses in 40 s, 4-9. Return counts on screen sit on this floor.

**Code provenance.** `dev3` and `dev3_blind` were recorded from `games/pong.py` sha256 `8a77ce3f2ff54bb0`. The frozen
code (`610cc4c40745732f`) differs only in display: the not-a-sex-comparison line is drawn only when the two brains
differ (it is unchanged for MaleCNS vs FAFB), and the miss label's backdrop stays opaque while the label fades. A
10-s check with the frozen code (`dev4_final.mp4` / `.json`, seed 121: 0-2, MaleCNS 1 return / 2 misses, FAFB 2 / 0)
includes a 1.0-deg MaleCNS miss at 8.55 s: the paddle stays put and dimmed, the ball passes it with a visible gap, and
the label reads "missed by 1.0°" clear of the ball.

**Reproducibility.** On CUDA a seed does not reproduce a run (docs/REPRODUCIBILITY.md 4.1 and the GPU notes after
it). The review measured it here: two runs of the same code on seed 111 differ from the first tick after warm-up. A
control arm on the same seed is therefore an independent draw, not a paired replay; the deterministic GAME floor above
is the paired baseline. The recorded seeds 0-2 cannot be re-run exactly.

### Limits

* **Not a sex comparison** (see Design). The difference in misses is consistent with how many photoreceptors each
  reconstruction placed in the frontal field: within MaleCNS alone, the decoder's error is lower with 9 or more
  photoreceptors under the ball than with 0-8 (above), and with 21 or more under the ball it is at or below FAFB's.
* **Sides were swapped on one GPU model, with three seeds per end.** By default MaleCNS plays the left end (side A,
  which receives the first serve) and FAFB the right. The fixed serve schedule is not side-neutral for paddles that do
  not track the ball: with both paddles frozen at the centre (GAME floor) side A misses 9 times to side B's 4, and in
  the three blind controls (RTX 4090) A missed 23 to B's 14. The swapped arm (FAFB left, MaleCNS right) and a
  same-ends replicate ran in one submission on one NVIDIA B200 node (Clips, "Sides swapped"). MaleCNS made 13 of the
  15 misses from the right end and 17 of the 18 from the left. Side A's misses went from 17 (MaleCNS there) to 2
  (FAFB there), and side B's from 1 to 13. In these runs the misses went with the brain rather than the end. Each
  brain did miss more often at the left end than at the right (MaleCNS 17 against 13, FAFB 2 against 1), which is
  the direction of the GAME floor's bias. Three runs per arm cannot tell that apart from run-to-run variation. What
  the swap does not do: it does not separate the brain from its photoreceptor coverage (or from proofreading and
  naming). It is not a significance test (three independent CUDA draws per arm). And no blind control was recorded with the ends swapped or on the B200.
* **The flies do not play, and the model does not track the ball.** The raw model does not pass small objects to
  LC10a / LC11 (README, "What the raw model does unprompted"). Locating the ball is the decoder's work, read from L2
  and Mi1 columns one or two synapses from the photoreceptors: the connectome contributes a retinotopic image, not a
  tracking computation. The flies have no motor output here, learn nothing, and are not rewarded.
* The score depends on GAME choices (speed-up, paddle size and speed, the angle rules). The decoder's law does not;
  its errors depend on them only through the speeds and paths the ball takes.
  A long rally needs steady localisation at a sideways speed the 80 deg/s paddle can follow; returns near the 20-deg
  floor at the cap move 68 deg/s sideways.
* The arena is a flat angular mapping (as in a tethered-fly LED arena), not a 3D court: the ball has a fixed angular
  size and does not loom.
* Both flies use the same decoder law and parameters (chosen as listed above).
* The decoders read `fb.optic.last["dr"]` directly (declared in the run log with `common.declare`), not through an
  attached `ReadDecoder`: gathering 1,000+ optic units per tick through the module path is slow. They write nothing,
  so `fb.module_records()` is empty and `meta.declared` carries both decoders.

### Commands

    python games/pong.py --record out/games/pong/seed{k}.mp4 --seed {k} --seconds 40
    python games/pong.py --record out/games/pong/control_seed{k}.mp4 --seed {k} --seconds 40 --control blind
    python games/pong.py --record out/games/pong/swapped_seed{k}.mp4 --seed {k} --seconds 40 --left fafb --right malecns
    python games/pong.py --record out/games/pong/b200_seed{k}.mp4 --seed {k} --seconds 40

40 s of brain time (2,000 frames at 50 fps), 8-13 min of wall time per clip on the shared RTX 4090 (486 s for
`dev3`, 709 s for `dev3_blind`, 669-752 s for the six recorded clips), and 396-480 s per clip for the six B200
clips (six at once on one node). Interactive: `python games/pong.py` (fly vs fly) or
`python games/pong.py --left human` (W / S or the mouse against FAFB); `--points N` ends the game when a side reaches
N.

## Clips

Two devices, twelve clips. Every comparison between arms (connected and blind, same ends and swapped) is made within
one device. The B200 same-ends replicate appears next to the RTX 4090 runs only as a description. The six clips in
this first part (seeds 0-2 and their blind controls) were recorded on the desktop's RTX 4090. Six more were recorded later on NVIDIA
B200s: the side swap and a same-ends replicate ("Sides swapped", below). Their provenance differs and is given there.

Recorded on 2026-09-28 on this desktop: `cuda`, NVIDIA GeForce RTX 4090, torch 2.10.0+cu128, Python 3.13.2. All six
run logs carry the same provenance:
* commit `959f2e927b1c2806bc526fd5fc51cb3f5cc3e4b1` with `dirty: true` (games/ is not committed yet);
* `games/pong.py` sha256 `610cc4c40745732f` and `games/common.py` `b110142562a927a1` (first 16 hex). This is the frozen
  code described above, the same `games/pong.py` as `dev4_final`.
* Both brains: preset `raw`, no instruments, no attached modules. `meta.declared` lists the four GAME rules (court,
  serves, arena, paddle) and the two decoders; the controls add `control_blind`.

Each command ran exactly once, from the repo root, as `PYTHONIOENCODING=utf-8 <miniconda>/python.exe`
followed by the arguments below. All six clips are kept. No run crashed or ran out of CUDA memory, and
`games/pong.py` was not changed.

Each clip is 40.0 s of brain time (2,000 frames, 1920x1080, 50 fps). The 1.0 s warm-up on the empty court comes
before the first frame. Each connected run and its blind control ran side by side on the shared GPU. "Wall" is the
log's `wall_s`: the game loop and encoding. It leaves out process start-up, building the two brains and the warm-up,
which took another 36-52 s per run (the recording script's console timestamps, less `wall_s`).

| clip | command arguments | started (UTC) | brain / wall |
|---|---|---|---|
| `out/games/pong/seed0.mp4` | `games/pong.py --record out/games/pong/seed0.mp4 --seed 0 --seconds 40` | 21:23:56 | 40.0 / 694.7 s |
| `out/games/pong/control_seed0.mp4` | `games/pong.py --record out/games/pong/control_seed0.mp4 --seed 0 --seconds 40 --control blind` | 21:25:44 | 40.0 / 668.8 s |
| `out/games/pong/seed1.mp4` | `games/pong.py --record out/games/pong/seed1.mp4 --seed 1 --seconds 40` | 21:36:22 | 40.0 / 752.1 s |
| `out/games/pong/control_seed1.mp4` | `games/pong.py --record out/games/pong/control_seed1.mp4 --seed 1 --seconds 40 --control blind` | 21:37:51 | 40.0 / 731.9 s |
| `out/games/pong/seed2.mp4` | `games/pong.py --record out/games/pong/seed2.mp4 --seed 2 --seconds 40` | 21:49:44 | 40.0 / 742.3 s |
| `out/games/pong/control_seed2.mp4` | `games/pong.py --record out/games/pong/control_seed2.mp4 --seed 2 --seconds 40 --control blind` | 21:50:48 | 40.0 / 737.8 s |

What the run logs count. Where a cell has two numbers they are MaleCNS / FAFB, and the score is MaleCNS - FAFB. Errors
are in degrees of azimuth, as median (90th percentile) over live ticks. "Held" is the fraction of live time the target
was held below the gate. "Edge" is the fraction of live time the paddle sat at the court edge.

| clip | score | points | returns | misses | longest rally | fastest return | target error | reading error | held | edge |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 0 - 6 | 6 | 15 / 18 | 6 / 0 | 10 | 200°/s (1 at the cap) | 6.6 (18.0) / 5.3 (9.1) | 5.6 (30.7) / 4.0 (6.9) | 22 / 1 % | 12 / 12 % |
| control 0 | 5 - 7 | 12 | 4 / 5 | 7 / 5 | 3 | 79°/s | 17.6 (42.5) / 17.4 (33.6) | 26.8 (57.8) / 18.5 (42.2) | 50 / 76 % | 0 / 0 % |
| seed 1 | 1 - 6 | 7 | 10 / 13 | 6 / 1 | 7 | 138°/s | 7.0 (19.5) / 5.0 (8.1) | 5.7 (36.8) / 3.9 (6.5) | 23 / 1 % | 9 / 10 % |
| control 1 | 4 - 9 | 13 | 2 / 4 | 9 / 4 | 3 | 79°/s | 18.2 (42.4) / 17.0 (36.1) | 26.6 (57.1) / 19.0 (43.8) | 47 / 76 % | 0 / 0 % |
| seed 2 | 0 - 7 | 7 | 13 / 17 | 7 / 0 | 12 | 200°/s (3 at the cap) | 6.6 (19.2) / 5.2 (8.6) | 5.7 (33.4) / 3.9 (6.6) | 22 / 1 % | 10 / 10 % |
| control 2 | 5 - 7 | 12 | 4 / 4 | 7 / 5 | 3 | 79°/s | 20.5 (39.0) / 16.9 (32.9) | 26.6 (54.6) / 18.1 (39.8) | 53 / 76 % | 1 / 0 % |
| GAME floor, 40 s | 4 - 9 | 13 | 2 / 4 | 9 / 4 | 2 | 69°/s | – | – | – | – |

**Over the three connected runs** (120 s of brain time):
* FAFB scored 19 points and MaleCNS 1.
* There were 86 returns (MaleCNS 38, FAFB 48) and 20 misses (MaleCNS 19, FAFB 1).
* The median outgoing return angle was 21.9-22.9°, and 30 of the 86 returns were steepened to the 20° floor.

In the three blind controls the score was 14 - 23, with 23 returns (10 / 13) and 37 misses (23 / 14). Three times
the GAME floor comes to 18 returns and 39 misses. With the ball shown there were 3.7 times as many returns as in the
blind arm (86 against 23), and 0.54 times as many misses (20 against 37).

**Where the misses came from** (connected runs). At the moment of each miss the log gives the decoder target's and the
paddle's distance from the ball. The paddle reaches the ball if that distance is at most 18° (the paddle half plus the
ball radius).
* Of MaleCNS's 19 misses, 14 had the target within reach at the moment of the miss: the paddle, limited to 80°/s
  (GAME), had not caught up with its target. The log gives both distances only at that moment, not when the target
  got there: in the missed serve at seed 2 35.47 s (target 1.2° off, paddle 18.5°) the clip shows the target on the
  ball's line at about 34.7 s, back near the middle of the court from about 34.9 to 35.2 s (the paddle following it),
  and on the ball again only at the miss. In the other 5 the target itself was more than 18° off:
  * seed 1: 46.1° at 11.71 s and 19.8° at 34.27 s;
  * seed 2: 43.7° at 2.72 s, 30.9° at 9.54 s and 39.3° at 29.58 s.
* Five misses were by less than 1°:
  * seed 0: 0.18° at 8.94 s;
  * seed 1: 0.60° at 19.19 s (FAFB) and 0.61° at 24.40 s;
  * seed 2: 0.33° at 32.97 s and 0.54° at 35.47 s.
* In four misses the ball was moving sideways faster than the paddle's 80°/s limit (GAME): seed 0 8.94 s (88°/s),
  seed 1 19.19 s (93°/s), seed 2 18.53 s (88°/s) and 27.22 s (128°/s).
* In the blind controls, 34 of the 37 misses had the target more than 18° from the ball. The controls' misses were by
  4.4-38.1°; the connected runs' by 0.18-28.1°.

**The decoder against the photoreceptors under the ball** (connected runs; target median error, the three seeds). For
MaleCNS the error is 7.3-11.5° with 0-8 reconstructed photoreceptors under the ball (not monotonic across those three
bins) and falls to 5.3-6.1° with 9-20 and 3.5-4.1° with 21 or more:

| photoreceptors under the ball | MaleCNS target median error | live ticks per run |
|---|---|---|
| none | 9.4, 11.5, 7.3° | 199-257 |
| 1-3 | 8.3, 8.6, 7.7° | |
| 4-8 | 8.4, 8.5, 7.8° | |
| 9-20 | 5.4, 5.3, 6.1° | |
| 21+ | 3.5, 4.1, 3.7° | 251-307 |

FAFB's error is 5.0-5.3° overall, with 21 or more photoreceptors under the ball for about 90 % of live time. Its
error in that same 21+ bin is 5.0-5.4°, so there MaleCNS's (3.5-4.1°, over 8-9 % of its live ticks) is at or below
FAFB's. The ball was over no MaleCNS L2 column for 6-8 % of live time. In the blind
controls MaleCNS's error does not fall with the photoreceptors under the hidden ball: in each control the 21+ bin is
the worst (31-38°; control 0 runs 22.6, 13.1, 15.8, 22.5, 36.1° across the five bins).

**Seed 0** (MaleCNS 0 - FAFB 6):
* The first serve (1.50 s) opens a 10-return rally. The ball speeds up x1.15 per return, from 52°/s to the 200°/s cap,
  which it reaches on FAFB's return at 8.61 s (the RALLY 10 banner).
* At 8.94 s MaleCNS's paddle misses by 0.18°. The ball was moving 88°/s sideways; the target was 15.4° from it (within
  reach) and the paddle 18.2°.
* Points 2-4 end after 5, 8 and 7 returns, with MaleCNS misses at 15.73, 23.43 and 31.47 s (by 1.9, 3.3 and 1.0°).
* Point 5 is a missed serve (33.84 s, by 3.7°). Point 6 ends after 3 returns (39.28 s, by 2.8°).
* All six misses are MaleCNS's, by 0.18-3.65°. At each, the target was 5.3-17.5° from the ball and the paddle 18.2-21.7°.
* FAFB returned all 18 balls that reached it. The clip ends between points.

**Seed 1** (MaleCNS 1 - FAFB 6):
* Point 1: MaleCNS misses at 4.93 s (by 3.1°) after 2 returns.
* Point 2: after 5 returns, at 11.71 s MaleCNS's target is 46° from the ball and the paddle misses by 28°. This is
  the largest miss of the connected runs.
* Point 3 is a 7-return rally (13.97-18.62 s, up to 138°/s). MaleCNS's return at 18.62 s is hit near the paddle's tip
  (offset -0.76) and leaves at -42°, 93°/s sideways. At 19.19 s FAFB's paddle misses by 0.6°: its target was 13.6°
  from the ball and the paddle 18.6°. The result is POINT MaleCNS, 1 - 2: FAFB's only miss and MaleCNS's only point in
  the three connected runs.
* Points 4-7 are MaleCNS misses at 24.40 s (0.6°), 29.20 s (4.5°) and 34.27 s (1.8°), then a missed serve at 36.73 s
  (5.0°). At the moment of that last one the target was 0.4° from the ball and the paddle 23° away.
* The clip ends 1 return into point 8.

**Seed 2** (MaleCNS 0 - FAFB 7):
* MaleCNS misses the first serve (2.72 s) by 22.6°; its target was 43.7° off.
* Point 2: after 5 returns, a 22.7° miss at 9.54 s (target 30.9° off).
* Point 3 is the longest rally of the three runs: 12 returns from 11.81 to 18.20 s, the last three at the 200°/s cap
  (RALLY 10 at 17.56 s). At 18.53 s MaleCNS's paddle misses by 11.8°. The ball was moving 88°/s sideways; the target
  was 16.2° from it and the paddle 29.8°.
* Point 4 runs 9 returns (20.90-26.77 s, up to 183°/s). FAFB's return at 26.77 s leaves at -44° (128°/s sideways),
  and MaleCNS misses by 14.1° at 27.22 s (target 9.2° off).
* Point 5: MaleCNS misses a serve by 10.4° (29.58 s; target 39.3° off).
* Point 6: after one FAFB return, MaleCNS misses by 0.33° at 32.97 s. Point 7: it misses a serve by 0.54° at 35.47 s.
  At the moment of these two misses the target was 6.9° and 1.2° from the ball, and the paddle 18.3° and 18.5°.
* FAFB returned all 17 balls that reached it. The clip ends 3 returns into point 8.

**Blind controls** (`--control blind`: the ball is never drawn in either arena; decoders, paddles, court and serves
are unchanged). These are the same seeds but, on CUDA, independent draws, not paired replays (Reproducibility, above).
* Scores were 5 - 7, 4 - 9 and 5 - 7. There were 9, 6 and 8 returns against 12, 13 and 12 misses.
* The longest rally in each was 3, and no return was faster than 79°/s.
* Control 1's totals equal the GAME floor's exactly: MaleCNS 2 returns / 9 misses, FAFB 4 / 4, score 4 - 9. The points
  differ, though. Control 1's returns per point were 0, 0, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 0; the floor's are
  0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2.
* The target's median error was 17.6-20.5° (MaleCNS) and 16.9-17.4° (FAFB), against 6.6-7.0° and 5.0-5.3° with the
  ball shown.
* Targets were held 47-53 % (MaleCNS) and 76 % (FAFB) of live time, against 22-23 % and 1 %.
* On screen the ball stays on the main court, labelled "hidden from both flies". Both mosaics show the empty court,
  and the decoder ribbons' activity carries no ball track (their white dots still mark the hidden ball's Y, which the
  display takes from the court, not from the flies).

The dev run on seed 111 (`dev3`, above) ended 0 - 5 with 33 returns, 5 misses and a longest rally of 10. The three
recorded seeds ended 0 - 6, 1 - 6 and 0 - 7, with 33, 23 and 30 returns, 6, 7 and 7 misses, and longest rallies of
10, 7 and 12.

**Trailer windows** (brain seconds, as in the logs; in the clip file each frame sits 0.02 s earlier, clip t = brain
t − 0.02, because frame f is drawn after ticks 2f+1 and 2f+2):
* seed 2, 15.1-18.9 s: RALLY 5 to RALLY 12 at the 200°/s cap, the miss, then POINT FAFB 0 - 3 and "LONGEST RALLY 12".
* seed 1, 17.3-20.0 s: FAFB's only miss (0.6°), then POINT MaleCNS.
* seed 0, 6.2-9.4 s: RALLY 10 at 200°/s, then a 0.18° miss.
* control 0, 11.5-13.4 s: the ball hidden from both flies; FAFB's paddle misses by 32°.

### Sides swapped, and a same-ends replicate (NVIDIA B200)

The caption skeptic found that the ends were never swapped. In every clip above MaleCNS played the left end (side A,
which receives the first serve), so the side and the brain went together. This arm puts FAFB on the left and
MaleCNS on the right, for seeds 0, 1 and 2. The RTX 4090 clips above could not be extended: brain runs had moved off
the desktop. So the original orientation was recorded again beside the swap, in the same submission on the same GPU
model, and the side comparison is made there. The RTX 4090 clips and their blind controls stay as they are. This
subsection was written after the caption skeptic's pass. A second pass then checked it (Corrections (caption
skeptic, sides swapped), at the end).

**Provenance.** The six clips came from one submission of six concurrent jobs to a GPU cluster, all on one node's
NVIDIA B200s. The run logs do not record the node or GPU, so this comes from the job manager's records: five jobs ran
on `<cluster-node>`'s GPU 3, which two other jobs also used until 00:42:47 UTC, and `b200_seed2` ran on GPU 0. The jobs started
2026-09-29 00:37:02-00:37:05 UTC, which is 17:37 on 2026-09-28 local time. Each job ran in a fresh copy of the cluster's
own checkout, with this working tree's modified and untracked files copied over it (`scripts/cluster_run.py`; the
log lists 41 files shipped):

    source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available(), "no CUDA"; print(torch.cuda.get_device_name(0))' \
        && mkdir -p out/games/pong && python games/pong.py <arguments below> > out/games/pong/<clip>.console.txt 2>&1

Each job printed "NVIDIA B200", exited 0 and wrote 2,000 frames (40.00 s, 1920x1080, 50 fps). The batch's last line
was "6 job(s), 0 failed" (`out/games/pong/swap_cluster.log`). Each command ran exactly once, and all six clips are
kept. All six run logs carry:
* `device` cuda, `device_name` NVIDIA B200, torch 2.11.0+cu128, Python 3.12.3.
* `commit` null and `dirty` null. The run directory is not a git checkout, so the game code is identified by the
  `sources` hashes. The model code is not hashed in the logs. The second caption skeptic compared the run directory's
  50 `flyverse/*.py` files on the cluster, blob by blob, with commit `959f2e9` (the RTX 4090 clips' commit), and
  all 50 are identical.
* `games/pong.py` sha256 `610cc4c40745732f`. This is the same frozen file as in the RTX 4090 clips.
* `games/common.py` `e06078fb69d1275a`, which **differs** from the RTX 4090 clips' `b110142562a927a1`. The lead's
  change note says the only change in between is `find_ffmpeg`: it now falls back to the ffmpeg that the
  `imageio-ffmpeg` package ships, because the cluster has no ffmpeg on PATH. A backup of the earlier file (sha256
  `b110142562a927a1`, from the session's edit history) was diffed against it and confirms this. The only changes are
  the new `find_ffmpeg`, which `Recorder` now calls, and the wording of `Recorder`'s missing-ffmpeg error. The mp4s'
  muxer tags show the other encoder: `Lavf61.1.100` here, `Lavf62.3.100` in the RTX 4090 clips.
* Both brains: preset `raw`, no instruments, no attached modules. `meta.declared` lists the same four GAME rules and
  two decoders. `summary.game_floor_frozen_paddles` is unchanged: side A 2 returns / 9 misses, side B 4 / 4. The
  swapped logs record `mode.left` fafb and `mode.right` malecns, and key the summary `A_FAFB` / `B_MaleCNS`.

| clip | command arguments | started (UTC) | brain / wall |
|---|---|---|---|
| `out/games/pong/swapped_seed0.mp4` | `games/pong.py --record out/games/pong/swapped_seed0.mp4 --seed 0 --seconds 40 --left fafb --right malecns` | 00:37:02 | 40.0 / 479.8 s |
| `out/games/pong/swapped_seed1.mp4` | `games/pong.py --record out/games/pong/swapped_seed1.mp4 --seed 1 --seconds 40 --left fafb --right malecns` | 00:37:03 | 40.0 / 480.0 s |
| `out/games/pong/swapped_seed2.mp4` | `games/pong.py --record out/games/pong/swapped_seed2.mp4 --seed 2 --seconds 40 --left fafb --right malecns` | 00:37:03 | 40.0 / 479.1 s |
| `out/games/pong/b200_seed0.mp4` | `games/pong.py --record out/games/pong/b200_seed0.mp4 --seed 0 --seconds 40` | 00:37:04 | 40.0 / 477.9 s |
| `out/games/pong/b200_seed1.mp4` | `games/pong.py --record out/games/pong/b200_seed1.mp4 --seed 1 --seconds 40` | 00:37:04 | 40.0 / 475.8 s |
| `out/games/pong/b200_seed2.mp4` | `games/pong.py --record out/games/pong/b200_seed2.mp4 --seed 2 --seconds 40` | 00:37:05 | 40.0 / 395.8 s |

**An earlier, stopped attempt.** A first run of the swapped seed 0 was started on the desktop's RTX 4090. Its video
file was created at 00:16:30 UTC and last written at 00:28:04 UTC, when brain runs were stopped on the desktop. It
left no run log, and its mp4 was never finalised, so neither a player nor ffprobe can open it. It is kept as
`out/games/pong/killed_rtx4090_swapped_seed0.partial.mp4`, and there is no outcome to report. The B200 runs are the
only complete recordings of the swapped arm.

What the run logs count. In every two-number cell the order is MaleCNS / FAFB, whichever end each played. The score is
MaleCNS - FAFB; on screen, the left score comes first. Errors are in degrees, median (90th percentile), as logged.
"Held" and "edge" are percentages of live time, from the logged fractions.

| clip | MaleCNS's end | score | points | returns | misses | longest rally | fastest return | target error | reading error | held | edge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B200 seed 0 | left | 0 - 6 | 6 | 16 / 19 | 6 / 0 | 12 | 200°/s (3 at the cap) | 6.81 (19.01) / 5.59 (10.06) | 5.87 (32.05) / 4.21 (7.61) | 19.2 / 1.1 % | 10.6 / 10.5 % |
| B200 seed 1 | left | 0 - 6 | 6 | 15 / 18 | 6 / 0 | 10 | 200°/s (1 at the cap) | 6.59 (20.19) / 5.03 (7.96) | 5.28 (31.69) / 3.89 (6.43) | 19.9 / 0.9 % | 11.2 / 11.9 % |
| B200 seed 2 | left | 1 - 5 | 6 | 14 / 15 | 5 / 1 | 7 | 138°/s | 6.87 (18.08) / 5.00 (8.53) | 5.83 (31.16) / 3.95 (6.93) | 24.0 / 0.8 % | 10.5 / 9.8 % |
| swapped seed 0 | right | 1 - 4 | 5 | 17 / 18 | 4 / 1 | 10 | 200°/s (1 at the cap) | 5.36 (17.08) / 4.80 (7.53) | 5.25 (30.65) / 3.59 (6.14) | 19.5 / 1.3 % | 12.7 / 12.7 % |
| swapped seed 1 | right | 0 - 5 | 5 | 16 / 19 | 5 / 0 | 10 | 200°/s (1 at the cap) | 5.91 (18.06) / 4.91 (7.82) | 5.29 (31.23) / 3.72 (6.29) | 17.8 / 1.1 % | 14.3 / 11.9 % |
| swapped seed 2 | right | 1 - 4 | 5 | 18 / 20 | 4 / 1 | 11 | 200°/s (4 at the cap) | 6.46 (17.47) / 5.23 (8.52) | 5.65 (28.95) / 3.93 (6.77) | 18.9 / 1.5 % | 12.6 / 11.3 % |

**The two arms on the B200** (three runs each, 120 s of brain time per arm):

| | same ends (MaleCNS left) | swapped (MaleCNS right) |
|---|---|---|
| points, MaleCNS - FAFB | 1 - 17 | 2 - 13 |
| returns, MaleCNS / FAFB | 45 / 52 | 51 / 57 |
| misses, MaleCNS / FAFB | 17 / 1 | 13 / 2 |
| misses at the left end (side A, first serve) | 17 (MaleCNS) | 2 (FAFB) |
| misses at the right end | 1 (FAFB) | 13 (MaleCNS) |
| decoder target error, median per run, MaleCNS | 6.59-6.87° | 5.36-6.46° |
| decoder target error, median per run, FAFB | 5.00-5.59° | 4.80-5.23° |
| target held, MaleCNS / FAFB | 19.2-24.0 / 0.8-1.1 % | 17.8-19.5 / 1.1-1.5 % |
| longest rally per run | 12, 10, 7 | 10, 10, 11 |
| returns at the 200°/s cap | 4 | 6 |
| returns steepened to the 20° floor | 32 of 97 | 38 of 108 |
| median outgoing return angle per run | 28.5, 23.3, 26.7° | 22.4, 24.9, 25.9° |

* **The misses stayed with MaleCNS at either end.** With MaleCNS on the left, the left end missed 17 times and the
  right end once. With FAFB on the left, the left end missed twice and the right end 13 times. The GAME floor's side
  bias (frozen paddles: side A 9 misses, side B 4) points the other way from the swapped result. Each brain did miss
  more often at the left end than at the right (MaleCNS 17 against 13, FAFB 2 against 1), the direction of that bias.
  Three runs per arm cannot tell this apart from run-to-run variation.
* MaleCNS's median target error was lower in each swapped run (5.36-6.46°) than in each same-ends run (6.59-6.87°).
  FAFB's ranges overlap (4.80-5.23° swapped, 5.00-5.59° same ends). The logs do not say why. Two things differ between
  the arms by design. A fly at the right end sees the court mirrored left to right (az = −Y; Design, the arena row).
  And the serve schedule is fixed by end, so each brain received different serves in the two arms. The rallies also
  differ from run to run.
* **MaleCNS's error is lower with 9 or more photoreceptors under the ball than with 0-8, at either end.** Target
  medians, three runs each:

  | photoreceptors under the ball | same ends | swapped |
  |---|---|---|
  | none | 10.59, 12.37, 10.93° | 8.62, 12.36, 9.32° |
  | 1-3 | 8.82, 9.13, 8.87° | 7.74, 8.89, 9.04° |
  | 4-8 | 8.52, 7.64, 7.81° | 7.55, 6.45, 9.29° |
  | 9-20 | 5.64, 5.57, 5.35° | 3.99, 4.39, 4.70° |
  | 21+ | 4.76, 3.79, 4.20° | 3.06, 4.26, 3.60° |

  In every one of the six runs the 9-20 and 21+ bins are below every 0-8 bin. In five of the six runs the error falls
  across all five bins. The exception is swapped seed 2, where the 4-8 bin (9.29°) is above the 1-3 bin (9.04°). In
  every run MaleCNS's 21+ median is also below FAFB's own 21+ median: same ends 4.76 / 5.57, 3.79 / 4.95
  and 4.20 / 5.00°, swapped 3.06 / 4.74, 4.26 / 4.86 and 3.60 / 5.17°. FAFB had 21 or more photoreceptors under the
  ball for 88-90 % of live ticks. The ball was over no MaleCNS L2 column for 5.0-6.6 % of live time.
* **Where the misses came from.** The measure is the decoder target's distance from the ball at the moment of the
  miss. "Within reach" means at most 18° (the paddle half plus the ball radius).
  * Same ends: 12 of MaleCNS's 17 misses had the target within reach. In the other 5 the target was more than 18°
    off: seed 0 at 4.99 s (30.3°), 14.01 s (22.6°), 28.54 s (21.6°) and 30.97 s (31.6°, a missed serve), and seed 1
    at 29.75 s (18.8°).
  * Same ends, sizes: MaleCNS's misses were by 0.27-39.42°. The 39.42° miss (seed 1, 29.75 s) is the largest of the
    twelve clips. Four misses were by less than 1°: MaleCNS 0.53° (seed 1, 20.96 s), 0.27° (seed 2, 16.14 s) and
    0.63° (seed 2, 21.80 s), and FAFB's only miss, 0.47° (seed 2, 8.50 s).
  * Same ends, speed: in 7 of MaleCNS's misses and in FAFB's one, the ball was moving sideways faster than the
    paddle's 80°/s limit (MaleCNS 86.7-132.5°/s, FAFB 92.6°/s).
  * Swapped: 11 of MaleCNS's 13 misses had the target within reach. In the other 2 it was more than 18° off (seed 0
    at 32.77 s, 23.7°; seed 2 at 20.93 s, 22.9°). The misses were by 0.38-18.41°, two of them under 1° (seed 1:
    0.38° at 17.31 s, 0.45° at 25.48 s). In 5 of them the ball was moving sideways faster than 80°/s (85.7-136.1°/s).
  * Swapped, FAFB: both of FAFB's misses came at the 200°/s cap, at the end of 10-return rallies, with the ball moving
    131.1 and 130.4°/s sideways. The target was within reach (16.4 and 16.6°) and the paddle 31.5 and 22.9° away.
    They were misses by 13.45° (seed 0, 8.87 s) and 4.91° (seed 2, 8.92 s), and they are MaleCNS's two points in
    this arm.
* The B200 same-ends replicate is not a paired replay of the RTX 4090 clips. It ran on a different GPU with torch
  2.11 vs 2.10 and Python 3.12 vs 3.13, and a CUDA run does not reproduce its seed anyway (Reproducibility, above).
  It ended 1 - 17 over 18 points, with 17 MaleCNS misses and 1 FAFB miss. The RTX 4090 runs ended 1 - 19 over 20,
  with 19 and 1.

**B200 seed 0, same ends** (MaleCNS 0 - FAFB 6; returns per point 2, 9, 12, 3, 0, 9):
* MaleCNS misses at 4.99 s (by 6.3°) and at 14.01 s (3.1°, after 9 returns).
* Point 3 is a 12-return rally (15.07-22.66 s) whose last three returns are at the 200°/s cap (RALLY 10 at 21.99 s).
  MaleCNS misses it by 19.7° at 23.02 s. The target was 12.1° from the ball and the paddle 37.7°, and the ball was
  moving 121°/s sideways.
* Then come misses at 28.54 s (3.6°), a missed serve at 30.97 s (7.4°), and 39.57 s (3.0°, after 9 returns).
* The clip ends between points. FAFB returned all 19 balls that reached it.

**B200 seed 1, same ends** (MaleCNS 0 - FAFB 6; returns per point 6, 3, 10, 9, 4, 1):
* MaleCNS misses at 7.32 s (2.95°) and 12.44 s (1.1°).
* At 20.96 s it misses by 0.53°, after a 10-return rally that reached the cap at 20.63 s.
* At 29.75 s it misses by 39.4°: the target was 18.8° off and the ball was moving 129°/s sideways.
* It misses again at 36.11 s (4.4°) and 39.44 s (9.5°).
* The clip ends between points. FAFB returned all 18 balls that reached it.

**B200 seed 2, same ends** (MaleCNS 1 - FAFB 5; returns per point 7, 7, 4, 3, 0, 7):
* Point 1: after 7 returns (up to 138°/s), FAFB's paddle misses by 0.47° at 8.50 s. The target was 13.3° from the
  ball and the paddle 18.5°, with the ball moving 93°/s sideways. POINT MaleCNS 1 - 0 at 8.58 s is MaleCNS's only
  point in the three B200 same-ends runs.
* MaleCNS then misses at 16.14 s (0.27°), 21.80 s (0.63°) and 27.03 s (2.3°), misses a serve at 29.44 s (2.1°), and
  misses at 37.31 s (15.4°).
* No return reached the cap (the fastest was 138°/s), and the longest rally was 7. The clip ends 1 return into
  point 7.

**Swapped seed 0** (MaleCNS 1 - FAFB 4, shown 4 - 1; returns per point 10, 6, 1, 8, 3):
* The first serve (1.50 s) goes to FAFB, now at the left end. A 10-return rally reaches the cap on MaleCNS's return
  at 8.49 s (RALLY 10), which leaves at -41° (offset -0.765).
* At 8.87 s FAFB's paddle misses by 13.4°. The ball was moving 131°/s sideways, the target was 16.4° from it (within
  reach), and the paddle was 31.5° away. POINT MaleCNS (0 - 1 on screen) follows at 8.92 s.
* MaleCNS then misses at 15.81 s (6.9°, after 6 returns), 19.09 s (1.1°), 27.51 s (2.6°, after 8 returns, ball
  96°/s sideways) and 32.77 s (16.5°, target 23.7° off).
* The clip ends 7 returns into point 6 (up to 138°/s).

**Swapped seed 1** (MaleCNS 0 - FAFB 5; returns per point 3, 10, 1, 8, 7):
* MaleCNS misses at 5.46 s (4.8°).
* Point 2 is a 10-return rally that reaches the cap on FAFB's return at 13.66 s, which leaves at +41°. MaleCNS
  misses by 7.3° at 14.05 s: the ball was moving 131°/s sideways, the target was 5.1° off and the paddle 25.3°.
* It misses again at 17.31 s (0.38°), at 25.48 s (0.45°, after 8 returns) and at 33.29 s (13.6°, after 7 returns,
  ball 92°/s sideways).
* FAFB returned all 19 balls that reached it. The clip ends 6 returns into point 6.

**Swapped seed 2** (MaleCNS 1 - FAFB 4, shown 4 - 1; returns per point 10, 10, 1, 6, 11):
* Point 1: 10 returns. MaleCNS's return at 8.53 s is at the cap and leaves at -41° (offset -0.88). FAFB's paddle
  misses by 4.9° at 8.92 s: the ball was moving 130°/s sideways, the target was 16.6° off and the paddle 22.9°. The
  result is POINT MaleCNS (0 - 1 on screen).
* Point 2: 10 returns. FAFB's return at 17.27 s is at the cap and leaves at +43°. MaleCNS misses by 18.4° at 17.67 s:
  the ball was moving 136°/s sideways, the target was 17.3° off and the paddle 36.4°.
* Point 3: MaleCNS misses by 8.3° at 20.93 s (target 22.9° off). Point 4: after 6 returns it misses by 1.5° at
  28.65 s (target 1.3°, paddle 19.5°).
* Point 5 is the longest rally of the swapped runs: 11 returns (29.72-37.49 s), the last two at the cap (RALLY 10
  at 37.18 s). MaleCNS misses by 1.6° at 37.82 s (ball 86°/s sideways, target 10.8°, paddle 19.6°). The banners
  read POINT FAFB 4 - 1 and "LONGEST RALLY 11".
* A serve (38.86 s) is in flight when the clip ends.

**On screen, the colours follow the ends, not the brains.** In the swapped clips FAFB is sage (left) and MaleCNS is
lilac (right), the reverse of every other clip. The title card reads "FlyWire FAFB v783 vs MaleCNS v1.0" and keeps
the not-a-sex-comparison line and both photoreceptor counts. Frames checked (brain seconds):
* swapped seed 0: 1.00 (title card), 5.00, 8.50, 8.88 and 8.94;
* swapped seed 1: 13.68, 14.06 and 14.14;
* swapped seed 2: 8.90, 8.96, 17.30, 17.68, 35.90, 37.20, 37.50, 37.84, 37.90 and 38.20;
* B200 seed 0: 21.60, 22.68, 23.04 and 23.40;
* B200 seed 2: 8.52 and 8.60.

The scores, banners and miss labels ("missed by 13°", "5°", "7°", "18°", "1.6°", "20°", "0.5°") match the logs.

**Trailer windows, B200 clips** (brain seconds; clip seconds in brackets, brain − 0.02):
* swapped seed 2, 36.0-38.2 s (35.98-38.18): RALLY 10 and 11 at the cap, MaleCNS's 1.6° miss, POINT FAFB 4 - 1,
  LONGEST RALLY 11.
* B200 seed 0, 21.6-23.4 s (21.58-23.38): RALLY 10 to 12 at the cap, a 20° miss, POINT FAFB 0 - 3, LONGEST RALLY 12.
* swapped seed 0, 7.2-9.4 s (7.18-9.38): RALLY 10 at 200°/s, FAFB's paddle misses by 13°, POINT MaleCNS.
* B200 seed 2, 7.9-8.9 s (7.88-8.88): FAFB's paddle misses by 0.5°, POINT MaleCNS 1 - 0.

### Corrections (caption skeptic)

Every number in this caption was re-checked against the six run logs (`out/games/pong/{seed,control_seed}{0,1,2}.json`),
`dev.json`, `dev3.json`, `dev3_blind.json`, `dev4_final.json`, `pong.game_floor(40)` (re-run), the code hashes, the
recording script's console logs (one attempt per clip, all exit 0), the retina (court and central coverage re-derived
on CPU: 747 / 281, 2,753 / 428 / 419, 16 / 17, 97 / 655, 51 of 1,581) and frames of the clips at the stated moments.
They hold. What changed:

* "Not a sex comparison: the difference in misses is consistent with ..." gained "Sides were not swapped: MaleCNS
  always played the left end (Limits)", and Limits gained the bullet **Sides were not swapped** (GAME floor 9 - 4 and
  blind controls 23 - 14 in misses against side A). The caption had left out that the side and the serve schedule are
  fixed and not side-neutral.
* "For MaleCNS the error falls as more reconstructed photoreceptors lie under the ball" became "the error is 7.3-11.5°
  with 0-8 ... (not monotonic across those three bins) and falls to 5.3-6.1° with 9-20 and 3.5-4.1° with 21 or more":
  seed 2's 0-photoreceptor bin (7.3°) is below its 1-3 and 4-8 bins (7.7, 7.8°), and seed 0's 4-8 bin is above its
  1-3 bin.
* Limits: "the decoder's error falls as the photoreceptors under the ball rise ... MaleCNS's error is close to FAFB's"
  became "lower with 9 or more photoreceptors under the ball than with 0-8 ... at or below FAFB's"; the 21+ bin is
  3.5-4.1° for MaleCNS against 5.0-5.4° for FAFB (added after the table), and 4.4 against 5.3° in `dev3` ("close to"
  became "at or below" there too).
* "14 had the target within reach: the paddle was following a target it had not caught up with" became "within reach
  at the moment of the miss ... The log gives both distances only at that moment ...", with the seed 2 35.47 s serve as
  the example (frames 34.50-35.47 s: the target marker is on the ball's line at 34.70 s, jumps back to within a few
  degrees of the court's centre line by 34.95 s, partly grey (held), stays there to about 35.2 s, and reaches the ball
  only at the miss; the ball is about 21-34° away over that stretch, from the serve's constant sideways speed). "At the moment of" was also added to the seed 1 36.73 s and seed 2 32.97 / 35.47 s
  distances.
* "the decoder ribbons carry no ball track" (blind controls) became "the decoder ribbons' activity carries no ball
  track (their white dots still mark the hidden ball's Y ...)": the frames show the white dots in both ribbons.
* "With the final code, seed 111" became "With the pre-freeze code (`8a77ce3f2ff54bb0` ...)": `dev3.json` records that
  hash, not the frozen `610cc4c40745732f`. That the two differ only in display is the builder's statement; no copy of
  the pre-freeze file was found to diff.
* "The three recorded seeds fall in the same range" became the three runs' scores, returns, misses and longest rallies.
* "8-12 min of wall time per clip" became "8-13 min ... 669-752 s for the six recorded clips" (seed 1: 752.1 s).
* "It leaves out building the two brains and the warm-up, which took another 36-52 s" became "process start-up,
  building the two brains and the warm-up ... (the recording script's console timestamps, less `wall_s`)": 36-52 s is
  launch-to-exit minus `wall_s`, which includes interpreter start-up.

Checked and not changed: the headline (19 of 20 points, 12 returns, 200°/s, 2,753 / 747 = 3.7x); every per-seed
bullet; the table rows, the totals (86 / 20, 23 / 37, 3.7x and 0.54x) and the GAME floor (69°/s = 68.8°/s); the miss
breakdowns (14 / 5, the five sub-1° misses, the four over 80°/s, 34 of 37, 4.4-38.1° and 0.18-28.1°); the provenance
(commit, hashes, argv, start times, 2,000 frames at 1920x1080 and 50 fps); the four trailer windows above. The dev-seed
measurements under "What was measured first" (signal to noise, open-loop r, paddles visible, polarity, offline replays)
come from scratch scripts, as the caption says, and were not re-run.

### Corrections (caption skeptic, sides swapped)

This pass re-checked every sentence that refers to the B200 runs against the six run logs
(`out/games/pong/{b200,swapped}_seed{0,1,2}.json`: meta, summary and every event). It also checked the six console
logs, `swap_cluster.log`, the job manager's records for the six job ids, `ffprobe` on the twelve mp4s and on the
partial file, and the partial file's timestamps. It diffed the two `games/common.py` versions and compared the
run directory's `flyverse/` blob by blob with `959f2e9`. The frames at the stated moments were extracted by frame
index (f = brain t / 0.02 − 1): swapped seed 0 at 1.00, 8.88, 8.94 and 9.40 s; swapped seed 1 at 14.14 s; swapped
seed 2 at 8.96, 17.68, 37.20, 37.84 and 38.20 s; B200 seed 0 at 23.04 and 23.40 s; B200 seed 2 at 8.60 s. The
within-B200 totals hold: same ends MaleCNS 1 - FAFB 17 (returns 45 / 52, misses 17 / 1); swapped MaleCNS 2 - FAFB 13
(returns 51 / 57, misses 13 / 2). What changed:

* Headline. Was: "takes 19 of the 20 points, and 13 of 15 with the ends swapped (on a B200, where the same-ends
  replicate went 17 of 18)". Now: "takes 19 of the 20 points on an RTX 4090. On a B200 it took 17 of 18 from the same
  end and 13 of 15 with the ends swapped." The old sentence set the RTX 4090's 19 of 20 directly against the B200's
  13 of 15 and left the within-device baseline (17 of 18) in brackets. The side comparison belongs within the B200.
* Not claimed. Was: "Swapping the ends did not move the miss split". The split did move, from 17 of 18 to 13 of 15,
  and FAFB's misses went from 1 to 2. Now: "With the ends swapped the misses stayed with MaleCNS ... and 17 of the
  18 from the left in a same-ends replicate run beside it".
* Limits and the first "Sides swapped" bullet. Was: "In these runs the split follows the brain, not the end" and
  "**The miss split follows the brain, not the end.**" Both brains missed more at the left end (side A) than at the
  right: MaleCNS 17 against 13, FAFB 2 against 1. That is the direction of the GAME floor's bias, so "not the end" is
  more than three runs per arm can show. Now: "the misses went with the brain rather than the end", "**The misses
  stayed with MaleCNS at either end.**", and both places add the per-brain counts and the variation caveat.
* Was: "Both decoders' median target errors were lower in the swapped runs than in the same-ends runs". FAFB's were
  not: swapped seed 2 (5.23°) is above same-ends seeds 1 and 2 (5.03, 5.00°). Only MaleCNS's ranges separate
  (5.36-6.46° against 6.59-6.87°). Was also: "so the ball paths fall on the other half of each fly's field". The
  rallies cover both halves of the court, so this was imprecise. Now: the court is mirrored left to right for the
  right-end fly (az = −Y), and the serve schedule is fixed by end.
* Was: "**MaleCNS's error still falls with the photoreceptors under the ball**" and "Across 0-8 the error is not
  monotonic". Five of the six runs fall monotonically across all five bins. Only swapped seed 2 does not (1-3 at
  9.04°, 4-8 at 9.29°). Now the header matches the first skeptic's wording for the RTX 4090 runs ("lower with 9 or
  more ... than with 0-8") and names the one exception.
* Provenance. Was: "(five shared one GPU; `b200_seed2` ran on another)", with no source. The run logs carry no node
  or GPU index. The job manager's records confirm the placement: all six on `<cluster-node>`, `pong-swap-5fab86-0` to `-4`
  on GPU 3 (00:36:56-00:45:54 UTC) and `-5` (`b200_seed2`) on GPU 0 (00:36:57-00:44:04). They also show two other
  jobs of the same user on GPU 3 until 00:42:47. Now cited.
* Provenance. Was: "Each job ran in a fresh copy of this working tree" and "The run directory is a copy of the working
  tree". `scripts/cluster_run.py` copies the files that differ from `origin/main` (41 shipped, per the log) over a
  fresh copy of the cluster's own checkout. That checkout is at `959f2e9` with no tracked changes. The run
  directory's 50 `flyverse/*.py` files are blob-identical to `959f2e9`, so the model code matches the RTX 4090 clips'
  commit. Now says so.
* Provenance. Was: "No copy of the earlier file was found to diff, so that rests on the note." A backup of
  `games/common.py` at `b110142562a927a1` was found in the session's edit history. Diffed against `e06078fb69d1275a`
  (the current file), it differs only in `find_ffmpeg` (new, with the `imageio-ffmpeg` fallback), `Recorder` calling
  it, and the error message. The lead's note holds. Now says so.
* "This subsection ... has not been through it" now points to this list.
* Clips intro. Was: "every comparison below is made within one device". The subsection itself sets the B200
  replicate beside the RTX 4090 runs (1 - 17 against 1 - 19). Now: "Every comparison between arms ... is made within
  one device. The B200 same-ends replicate appears next to the RTX 4090 runs only as a description."
* B200 seed 1: "7.32 s (2.9°)" became "(2.95°)" (the log's `by_deg`, which rounds to 3.0, not 2.9).

Checked and not changed: both B200 tables (every cell), the two-arm totals and ranges, the steepened counts (32 of 97,
38 of 108) and median angles, and the photoreceptor-bin table. Also unchanged: the 21+ comparisons with FAFB,
88-90 % and 5.0-6.6 %, and every miss breakdown (12 / 5 and 11 / 2 within reach, the listed times and sizes,
0.27-39.42° and 0.38-18.41°, the four and two sub-1° misses, the 7 + 1 and 5 misses above 80°/s with their ranges,
and FAFB's two swapped misses). So are all six per-seed narratives (times, sizes, speeds, returns per point, clip
endings) and the provenance (device, torch, Python, both hashes, argv, start times, wall 396-480 s, 2,000 frames at
1920x1080 and 50 fps, Lavf61.1.100 against Lavf62.3.100, "6 job(s), 0 failed", 17:37 PDT). The partial file checks
out too: created 00:16:30 and last written 00:28:04 UTC, and ffprobe reports "moov atom not found". Finally, the
title card, the colours (sage left, lilac right, by end), the miss labels ("13°", "5°", "7°", "18°", "1.6°", "20°",
"0.5°", matching `fmt_deg`) and the banners (POINT MaleCNS 0 - 1, POINT FAFB 4 - 1 with LONGEST RALLY 11, POINT FAFB
0 - 3 with LONGEST RALLY 12, POINT MaleCNS 1 - 0) all match the frames. Not verifiable from any log, and left as the
lead's account: that the RTX 4090 runs could not be extended because brain runs had moved off the desktop, and that
the partial file came from the desktop's RTX 4090.
