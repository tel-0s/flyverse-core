# swat: a fly swatter vs the giant fibre

**A fly swatter vs 167,106 neurons: in 18 of 30 swings the giant fibre crossed the body's 33 Hz takeoff line and the
jump launched before the paddle arrived, with the paddle still 5.6 to 101 ms away. In all 18 the fixed forward hop then
met the paddle in the air. From behind, and from 1.5 m/s up, the paddle arrived first every time (12 of 12).**

Three recorded runs (seeds 0, 1, 2) of the same ten scripted swings, 0.5 to 2.5 m/s. Each run scored 6 IN TIME and
4 SPLAT, and it was the same six swings every time: 0.5-1.2 m/s from above, the front and the sides. The swing from
behind at 1.0 m/s and every swing at 1.5, 2.0 and 2.5 m/s ended in a SPLAT. In 4 of the 12 splats the giant fibre
crossed 33 Hz on the paddle's arrival tick itself (a tie, which the referee gives to the paddle), and in the other 8 it crossed 10-30 ms after contact. There is no
decoder: the takeoff is the shipped body model's own rule (DNp01 >= 33 Hz). The jump always launches as the same 45 deg
forward hop, and in all 18 in-time swings it met the paddle in the air 10-60 ms later. With the swatter removed
from the fly's world (the control), the fly never took off during a swing.

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the fly) | MaleCNS v1.0 (167,106 neurons), as shipped: preset `raw`, nothing instrumented, attached, trained or tuned (`fb.module_records()` is empty in every log). The model is everything from the light on the 1,466 ommatidia to the giant fibre's (DNp01's) rate: the loom detectors (LPLC2, LC4), the giant fibre and the jump motor neurons (TTMn). The HUD's DNp01, LPLC2, LC4 and TTMn numbers are the model's rates. |
| **DECODER** | **None.** No brain quantity is mapped to a game control. The takeoff is the shipped body model (`flyverse/body.py`, `Flight.maybe_takeoff`): DNp01's mean rate >= 33 Hz, with the fly on the ground for >= 1.0 s since its last landing, launches a 0.6 m/s hop at 45 deg, forward. The walk is the shipped `Locomotion` readout, and the hop is flown by the shipped `Flight.step` (thrust and lift from the wing-power motor neurons, yaw from the steering motor neurons' L-R asymmetry). All of these are part of flyverse, not written for this game. |
| **GAME** (not the fly) | The room and table (room_demo's scene, one apple), the wind, odour and taste inputs, the swatter, its ten-swing schedule (the same for every seed), the referee (one clock; a takeoff on the arrival tick counts for the paddle), IN TIME / SPLAT, the 0.4 s pin after a splat, the cameras, the drawn fly body, the replays, the banners, the scoreboard and the end card. |
| **not claimed** | The fly does not dodge, aim or fly. The hop always launches 45 deg forward (never away from the threat), and the raw model has no flight state, so it came down 0.25-0.36 s later (`Flight.step` adds thrust, lift and yaw from the wing motor neurons while it is in the air, but cannot sustain it). It met the paddle in the air in every in-time swing. The game then stops the paddle there and lifts it, and the hop carries on and lands: that meeting is not scored, as a hit or as an escape. The game scores only whether the jump was launched before the paddle arrived. The fly does not turn on its own when walking (README): its walk drifted by at most 15 deg in 30 s (the controls, which have no hops). The larger changes of heading came in the hops (`Flight.step`'s in-air yaw, read from the steering motor neurons) and at the table's fence (Clips). The same seed run twice is not the same run: the CUDA brain is not bit-reproducible (Design). |

## Design

The shipped fly (MaleCNS v1.0, `preset="raw"`, no instruments, **no decoders**) walks on the table of the room
demo's scene while a fly swatter swings at it ten times, faster each time. No game control reads the brain: the
swatter reads only where the fly is, and the brain reaches the game only through the shipped body model. The HUD and the
log read its rates (the log also times a splatted fly's DNp01 crossing, which picks the replays). The escape is the body model's own readout (`flyverse/body.py`): when the giant fibre (DNp01, the escape command
neuron) reaches 33 Hz, `Flight.maybe_takeoff` launches the jump. The swat is scored **IN TIME** (PLAN.md's
"escaped") if that takeoff comes on a 10 ms tick before the paddle reaches the spot where the fly stands, and
**SPLAT** otherwise. What is scored is the reflex's timing, not the hop: the hop is the body model's fixed 45 deg
forward jump, and in all 18 in-time swats of the recorded runs it met the paddle in the air. The screen says "JUMPED IN TIME", not "escaped", and
says when the hop met the paddle.

| what | kind | law and parameters |
|---|---|---|
| the brain | CONNECTOME | `FlyBrain(seed=...)` via `games.common.build_brain`, preset `raw`, nothing instrumented, **no module attached** (`fb.module_records()` is empty). No decoder: no brain quantity is mapped to any game control. |
| the fly's eyes | GAME (scene) | `scripts/room_demo.py`'s room with its single-fruit table (`world.make_room(seed, fruit_set="apple")`: a 4 x 4 x 2.6 m room, a 1.2 x 0.8 m table, a red-white cloth, one apple and a lamp) plus the swatter. It is ray traced from the eye (1.2 mm above the feet) along the 1,466 x 7 ommatidial rays (`games.common.Eyes`) every 10 ms. The tracer is spectral ([UV, B, G, R] reflectances), so UV is each material's own value, not 0.5 B. The fly's eyes never see the fly's body. |
| other senses | GAME (scene) | as in room_demo's `Sim.step`: wind at 0.3 m/s from 180 deg on the Johnston's organ, the apple's puffing odour plume on both antennae, and sugar taste when the legs are within 1.5 cm of the apple |
| walking, jumping, landing | body model, shipped | `Locomotion.readout / step` on `fb.motor()` (fenced table, as room_demo `--fence`). **`Flight.maybe_takeoff`: an escape jump (0.6 m/s at 45 deg, forward) when the smoothed DNp01 mean rate >= `Flight.gf_hz` = 33 Hz and the fly has stood >= 1.0 s since its last landing**, or a voluntary takeoff (0.2 m/s at 30 deg) when the wing power motor neurons hold >= 50 Hz for 0.3 s. Each takeoff is logged with the rule that fired. `Flight.step` flies the hop, adding thrust and lift from the wing-power motor neurons' rate (logged at 3.5-69.4 Hz at the 30 recorded takeoffs) and yaw from the steering motor neurons' L-R asymmetry, so the hop is not purely ballistic. The raw model has no flight state, so every recorded hop came down after 0.25-0.36 s, about 9-13 cm on (less where the table ended). The fly starts at (0.5, 0.0) m facing -x, walking away from the apple. |
| the swatter | GAME | a black paddle, an ellipsoid with radii 0.06 x 0.045 x 0.004 m lying face down (long axis towards the handle), plus a handle (0.19 x 0.006 x 0.004 m) rising 20 deg from its rim. It hovers with the paddle centre 0.45 m from the fly's eye, then swings in a straight line at the scheduled speed towards the eye. It tracks the eye while the fly stands (the fly walks about 1 cm/s) and freezes on the spot where it took off. The swing ends on the tick the paddle **reaches that spot** (the eye there within 4 mm of the paddle: inside the paddle grown by 4 mm on every semi-axis; from straight above, the underside is 4 mm above the eye), or when **the eye of a fly in the air comes within 4 mm of the paddle**. That test is swept over the tick, because a fast paddle and the hop can close 20-30 mm in 10 ms, more than the paddle's 16 mm shell. In the air case the paddle is drawn stopping where it touches the eye's 4 mm shell. Then the swatter lifts. |
| the referee: one clock | GAME | `referee_step`, the order every tick follows: (1) the brain steps 10 ms; (2) the paddle moves to that tick's time and is tested against the fly where it stands; (3) the body acts. It is blocked on the tick the paddle reaches the standing fly, so **a takeoff on the arrival tick is a SPLAT (a tie goes to the paddle)**. (4) A takeoff reads the paddle at that same tick; (5) an airborne eye is tested against the paddle over that same tick. The arrival tick is fixed when the swing starts, by the paddle's path length and speed. |
| outcome | GAME | **IN TIME if the body took off on a tick before the paddle's arrival tick; SPLAT otherwise.** The **margin** is the number of whole 10 ms ticks from the takeoff tick to the arrival tick, so the smallest in-time margin is +10 ms. The log also keeps the exact time the paddle still needed at the takeoff tick (`paddle_ttc_at_takeoff_ms`, which is 0-10 ms less than the margin). A splat's **lateness** is the number of whole ticks from contact to the first tick with DNp01 >= 33 Hz (0 = the same tick). Each swat logs where it ended (`contact_where`): `table` (a splat), `air` (the paddle met the hop in the air; `air_contact_after_takeoff_ms`) `takeoff spot` (the paddle reached the spot the fly had left) or `landed` (it came down on a fly that had jumped in time and landed there again; only a very slow player swing can do this). **The hop is not scored.** |
| pin and respawn | GAME | after a SPLAT the body is held where it is (no walking, no takeoff) for 0.4 s under the paddle. Then the swatter lifts at 1.0 m/s and the fly carries on from the same spot. The brain is never reset. |
| timing | GAME | no swing in the first 1.5 s (the optic lobe starts from rest). A swing starts after >= 0.7 s of aiming, with the fly on the table for >= 1.2 s (the body's own landing refractory is 1.0 s) and no replay pending. Between swings the swatter lifts back to 0.45 m and swings round to the next direction in 1.0 s, following the fly with a 0.25 s lag. |
| schedule | GAME | the same ten swats for every seed, in the fly's frame (azimuth 0 = ahead, + = left): above 0.5 m/s; front 45 deg up 0.6; left 35 deg up 0.7; above 0.8; right 35 deg up 0.9; behind 45 deg up 1.0; front 45 deg up 1.2; above 1.5; left 35 deg up 2.0; above 2.5 |
| HUD traces | CONNECTOME | DNp01 mean rate (the value the body model thresholds; `fb.motor().gf`, scale 0-120 Hz) with the 33 Hz line and swing / takeoff / contact ticks; LPLC2 (185 cells) and LC4 (126 cells) population mean rates (read-only `fb.brain.rate`); TTMn (the jump-muscle motor neurons) mean rate. **WHAT THE FLY SEES** is the exact radiance handed to `fb.vision`. The scoreboard, the swat box, the replay's timing and the end card carry GAME chips; the replay's GF carries a CONNECTOME chip. |
| cameras, fly model, replay, end card | GAME (display) | The main view is a human camera. WIDE shows the swatter over the ringed fly. It cuts to CLOSE-UP (a 30 deg lens 2.4-3 cm from the fly, 2-6 mm above its eye, below the paddle, following the hop; the 2.6 mm fly is about 150 px long) when the swinging paddle is within 6 cm or 80 ms of the fly, or when the fly takes off. The picture-in-picture shows a tight fly-cam under WIDE (the fly about 120 px long) and WIDE under the close-up. Banners go below the fly (or above it when the fly is low in the frame). The fly's body (13 ellipsoids, 2.6 mm, wings folded, legs swung in a display-only tripod gait) exists only in the camera's scene. After a SPLAT a splat mark is drawn on the table. Photo finishes (margin <= 30 ms, or a GF crossing <= 30 ms after contact) are replayed at 0.25x, 0.9 s after the arrival tick. Each replay covers -250 to +120 ms around the arrival tick and shows a profile close-up (stored at 0.7x resolution) with the WIDE view of the same instant inset and the side column as it was at that instant (framed in amber, "REPLAY: t = ..."). The brain runs on meanwhile. After the last swat an end card sums up the run, and a recording ends 3 s later (or at `--seconds`, 40). |
| control | GAME | `--control`: the swatter is removed from the fly's world (the camera still draws it); same seed, schedule and everything else |

### What was measured first (dev seeds >= 100)

* **The loom reaches the giant fibre.** Prototype (seed 100; a bare copy of room_demo's loop, the PLAN's axis-aligned
  paddle without a handle, the full fruit set; orientation only, not re-checked for the clock error below, and no run
  log of it is kept in out/games/swat): takeoff
  before contact by 134 ms (from above, 0.5 m/s), 2 ms (above, 1.0), 30 ms (front 45 deg up, 1.0), 56 ms (left 30 deg
  up, 0.5) and 28 ms (right 30 deg up, 1.0); too late by 29 ms (above, 2.0), 12 ms (left, 1.0) and 20 ms (behind
  45 deg up, 1.0). DNp01 peaked at 51-79 Hz, the LPLC2 population mean at 6-9 Hz, LC4 at 1-2 Hz and TTMn at 34-48 Hz.
* **The shipped hop cannot clear a 12 cm paddle at these margins.** A kinematic check (CPU; `Flight.launch` +
  `Flight.step` without wing power, with the paddle's path as in the game; no log kept) found the lead the fly needs to never touch
  the paddle's volume: 190-205 ms from above, 235-280 ms from the front, and 85-145 ms from the side or behind (25-45
  deg up, 0.5-1.0 m/s). The jump always goes 45 deg forward-up, never away from the threat. So the game scores
  PLAN.md's rule (takeoff before the paddle arrives) and reports the hop separately (`contact_where`).
* **A clock error, found in review and fixed.** The first build read the paddle's position from the tick before the
  takeoff. That added 10 ms to every margin and scored a takeoff on the arrival tick as an escape, which produced its
  "photo finishes" of +4.7, +6.8 and +8.1 ms. The fixed referee is described above. It has a CPU test that drives it
  tick by tick (`tests/test_games_swat.py`, `ClockTests`). The brain's inputs are identical between the two orders up
  to each takeoff, so the first build's logged runs can be re-read exactly. An old margin of 10 ms or less becomes a
  SPLAT with the GF crossing on the arrival tick (0 ms late); any other old margin becomes ceil((m - 10) / 10) ticks.
  Re-read that way, its seven 32 s dev runs (seeds 100-105 and the first dev clip; builds `606f3f89` / `61a151a6` /
  `ccc62a5d` / `c697a14d` of games/swat.py; seeds 100-103 started at (-0.5, -0.25) m facing +x, seeds 104-105 and the dev
  clip at (0.5, 0) facing -x; the camera was at 12 % resolution except in the dev clip) give:

  | # | swat | speed | s100 | s101 | s102 | s103 | s104 | s105 | dev (s100) |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | from above | 0.5 m/s | +90 | +60 | +90 | +80 | +100 | +100 | +50 |
  | 2 | from the front, 45 deg up | 0.6 m/s | +120 | +100 | +80 | +100 | +110 | +120 | +80 |
  | 3 | from the left, 35 deg up | 0.7 m/s | +40 | +60 | +40 | +50 | +40 | +40 | +100 |
  | 4 | from above | 0.8 m/s | +30 | +40 | +40 | +50 | +40 | +30 | +40 |
  | 5 | from the right, 35 deg up | 0.9 m/s | +20 | +30 | +30 | +30 | +30 | +50 | +60 |
  | 6 | from behind, 45 deg up | 1.0 m/s | SPLAT 10 | +10 | SPLAT 20 | SPLAT 10 | SPLAT 10 | SPLAT 0 | SPLAT 20 |
  | 7 | from the front, 45 deg up | 1.2 m/s | +10 | +20 | +30 | +40 | +20 | +30 | +30 |
  | 8 | from above | 1.5 m/s | SPLAT 10 | SPLAT 10 | SPLAT 10 | SPLAT 0 | SPLAT 10 | SPLAT 0 | SPLAT 0 |
  | 9 | from the left, 35 deg up | 2.0 m/s | SPLAT 0 | SPLAT 30 | SPLAT 10 | SPLAT 20 | SPLAT 10 | SPLAT 20 | SPLAT 20 |
  | 10 | from above | 2.5 m/s | SPLAT 20 | SPLAT 0 | SPLAT 20 | SPLAT 10 | SPLAT 20 | SPLAT 20 | SPLAT 0 |

  (+n = IN TIME with an n ms margin; SPLAT n = the giant fibre crossed 33 Hz n ms after contact.) That is 43 of 70 in
  time (6, 7, 6, 6, 6, 6 and 6 per run). Swats 1-5 and 7 were in time in all 42 of their runs, with margins of 10-120 ms.
  The swing from behind at 1.0 m/s was in time once in 7. **No swing at 1.5-2.5 m/s was in time (0 of 21)**: in every
  one the giant fibre crossed 33 Hz 0-30 ms after the paddle arrived. After the 0.4 s pin was released, the body jumped.
* **Control** (`--control`, seeds 100 and 101, build `61a151a6`): with the swatter removed from the fly's world there were
  **0 takeoffs in 20 swings** and 20 splats, and DNp01 peaked at 0-28 Hz during the swings (walking). A review rerun of
  build `c697a14d` (seed 100, 2 swings) also gave 0 takeoffs, with GF peaks of 23 and 10 Hz. The takeoffs are the fly
  seeing the swatter.
* **The giant fibre while walking.** On the ground between swings (>= 0.5 s after a landing), DNp01 had a 99th
  percentile of 16-21 Hz and a maximum of 22-35 Hz across eight runs (seeds 100-105 and controls 100-101, builds
  `606f3f89` / `61a151a6` / `ccc62a5d`). The one value above 33 Hz there (seed 104) fell within the body's 1.0 s landing
  refractory and launched nothing. An earlier build's seed-101 run also logged a 33.7 Hz ground maximum, which launched
  nothing either. No escape jump happened outside a swing. One voluntary
  takeoff did (seed 104, 0.3 s after a landing; wing power >= 50 Hz for 0.3 s, DNp01 0.6 Hz), and the log and the
  banner name the rule that fired.
* **Fruit.** With room_demo's full fruit set, the hopping fly landed inside the apple on seed 101 and was swatted with
  its eye inside the fruit: the fenced walk does not treat fruit as obstacles. The game therefore uses room_demo's
  single-apple table and starts the fly at (0.5, 0.0) facing -x, so the apple stays behind it. In the seven re-read dev
  runs below the heading turned left, by 33-65 deg before the fly reached the table's end, in steps between swats (mostly
  the hops' in-air yaw, which `Flight.step` reads from the steering motor neurons; the controls show the walk itself
  drifting by up to about 15 deg in 30 s). The dev clip on the fixed referee stayed within 12 deg of 180 until the
  fence, and recorded seed 1 turned right (Clips). The log records the eye's closest approach to the fruit and any entry.
* **The same seed twice is not the same run.** Two identical runs of seed 106 agreed through the first swat, landed
  4 mm apart after the first hop, and then diverged (swat 2: margins 10 ms apart). The CUDA brain is not
  bit-reproducible and the network amplifies the difference. The camera does not feed the brain.

* **The dev clip on the fixed referee** (`out/games/swat/dev_fix.mp4` + `.json`, seed 100, 1920 x 1080, 32.3 s;
  games/swat.py `dfa198b0`). The frozen code (`58003979`) differs from it only in the replay camera following the hop
  more closely (display), the interactive A key, and a `landed` label that the schedule cannot reach. Those three
  changes were checked on CPU with a stand-in brain. The run gave **6 JUMPED IN TIME, 4 SPLAT**:

  | # | swat | speed | outcome | notes |
  |---|---|---|---|---|
  | 1 | from above | 0.5 m/s | IN TIME +70 ms | paddle 64.0 ms away at the takeoff tick; hop met it in the air 40 ms later |
  | 2 | from the front, 45 deg up | 0.6 m/s | IN TIME +100 ms | met in the air 50 ms after takeoff |
  | 3 | from the left, 35 deg up | 0.7 m/s | IN TIME +90 ms | met in the air 50 ms after takeoff |
  | 4 | from above | 0.8 m/s | IN TIME +20 ms | met in the air 10 ms after takeoff; replayed |
  | 5 | from the right, 35 deg up | 0.9 m/s | IN TIME +60 ms | met in the air 40 ms after takeoff |
  | 6 | from behind, 45 deg up | 1.0 m/s | SPLAT | DNp01 reached 34.9 Hz on the arrival tick itself: a tie, 0 ms late; replayed |
  | 7 | from the front, 45 deg up | 1.2 m/s | IN TIME +20 ms | met in the air 10 ms after takeoff; replayed |
  | 8 | from above | 1.5 m/s | SPLAT | GF 29.6 Hz by contact, crossed 33 Hz 10 ms after; replayed |
  | 9 | from the left, 35 deg up | 2.0 m/s | SPLAT | GF 21.3 Hz by contact, crossed 20 ms after; replayed |
  | 10 | from above | 2.5 m/s | SPLAT | GF 5.2 Hz by contact, crossed 20 ms after; replayed |

  All six in-time hops met the paddle in the air, 10-50 ms after takeoff: the fixed 45 deg forward hop goes into the
  swing. In the in-time swings DNp01 peaked at 41-61 Hz, the LPLC2 mean at 4.1-5.7 Hz, LC4 at 1.1-1.5 Hz and TTMn at
  16-42 Hz. Each splatted fly jumped as soon as the pin was released (DNp01 68-83 Hz). There were no takeoffs outside a
  swing or a pin release. On the ground between swings, DNp01 had a 99th percentile of 15.9 Hz and a maximum of 23.7 Hz.
  The eye stayed >= 10.9 cm from the apple. The six replays ended by 29.3 s, the end card appeared at 29.28 s, and the
  clip ended 3 s later. It took 26 min of wall time on the shared RTX 4090 (about 50 s per brain-second under
  contention; about 22 s at half resolution when the card was quieter).
* **Control on the final code**: not re-run on the GPU here. With the swatter invisible, no takeoff happens, every
  swing ends on the table, and the referee's change (which only matters at a takeoff) cannot act. The control is
  recorded with seeds 0-2 (command below).

### Limits

* It is the body model's reflex, not a dodge. The fly does not aim its jump: the escape is always 0.6 m/s, 45 deg
  forward-up, and the raw model has no flight state, so the hop came down after 0.25-0.36 s, about 9-13 cm on
  (`Flight.step` shapes it with thrust, lift and yaw from the wing motor neurons, but cannot sustain it). A real fly
  directs its takeoff away from the threat and flies (Card & Dickinson 2008). The game scores whether the jump was
  triggered before the paddle arrived, and it logs and shows that the hop then met the paddle in the air (18 of 18 in
  the recorded runs).
* The swatter is a loom with a swatter's size and speed. It stops 4 mm from the eye (or where it meets the hop) and
  never touches the table.
* The fly does not turn on its own when walking (README). Its heading changed mostly in the air (the in-air yaw that
  `Flight.step` reads from the steering motor neurons) and at the fence's edge. The walk itself drifted by up to about
  15 deg in 30 s (the controls, Clips).
* Margins and lateness are counted in whole 10 ms ticks. A margin of +n ms means that at the takeoff tick the paddle
  still needed more than n - 10 and at most n ms (the exact value is logged as `paddle_ttc_at_takeoff_ms`). The
  brain steps 10 ms at a time, so the GF crossing itself happened somewhere within that takeoff tick. The 100 ms rate
  estimate behind the 33 Hz threshold is the body model's, unchanged.
* The splat and the respawn are GAME: the brain is never told and never reset. A splatted fly's giant fibre typically
  crosses 33 Hz 0-30 ms after contact, so it jumps as soon as the pin is released.
* The LPLC2 and LC4 traces are population means over all cells of the type. The loom drives only the columns it
  covers.

    python games/swat.py --seed 0 --record out/games/swat/seed0.mp4 --seconds 40
    python games/swat.py --seed 0 --record out/games/swat/control_seed0.mp4 --seconds 40 --control --render-scale 0.5

(`--seconds 40` is a cap: a recording ends 3 s after the end card, 31.9-33.1 s in. The control renders the camera at
half resolution because only its log is compared; the camera never feeds the brain.)

## Clips

Recorded on 2026-09-28 on this desktop: `cuda`, NVIDIA GeForce RTX 4090, torch 2.10.0+cu128, Python 3.13.2. All six
run logs carry the same provenance:
* commit `959f2e927b1c2806bc526fd5fc51cb3f5cc3e4b1` with `dirty: true` (games/ is not committed yet);
* `games/swat.py` sha256 `580039795d149441` and `games/common.py` `b110142562a927a1` (first 16 hex). This is the frozen
  code (`58003979`) described above;
* the brain: MaleCNS v1.0, 167,106 neurons, preset `raw`, `instruments: []`, `modules: []`. `meta.decoders` says "none:
  no ReadDecoder is attached and no brain quantity is mapped to a game control". `meta.declared` lists the six GAME
  rules (swatter, outcome, pin and respawn, timing, schedule, camera / fly model / replay). The controls add a seventh,
  `control` ("the swatter is removed from the fly's world (the camera still draws it); everything else is identical").

Each command ran exactly once, from the repo root, as `PYTHONIOENCODING=utf-8 <miniconda>/python.exe`
followed by the arguments below. All six clips are kept. No run crashed or ran out of CUDA memory, and
`games/swat.py` was not changed. The three seeds ran side by side, then the three controls did (control 0 started when seed 0 finished, about a
minute before seeds 1 and 2 did), on a GPU and CPU
shared with about ten other recordings. Hence 39-41 s of wall per brain second (32-33 s for the half-resolution
controls). The brain steps in 10 ms ticks either way, so wall time does not change a run. "Wall" is the log's
`wall_s`: the game loop and encoding, without building the brain.

| clip | command arguments | started (UTC) | brain / wall | frames |
|---|---|---|---|---|
| `out/games/swat/seed0.mp4` | `games/swat.py --seed 0 --record out/games/swat/seed0.mp4 --seconds 40` | 23:10:38 | 32.66 / 1284.0 s | 1,633 |
| `out/games/swat/seed1.mp4` | `games/swat.py --seed 1 --record out/games/swat/seed1.mp4 --seconds 40` | 23:11:37 | 31.94 / 1317.8 s | 1,597 |
| `out/games/swat/seed2.mp4` | `games/swat.py --seed 2 --record out/games/swat/seed2.mp4 --seconds 40` | 23:11:37 | 32.34 / 1310.8 s | 1,617 |
| `out/games/swat/control_seed0.mp4` | `games/swat.py --seed 0 --record out/games/swat/control_seed0.mp4 --seconds 40 --control --render-scale 0.5` | 23:33:11 | 33.06 / 1057.4 s | 1,653 |
| `out/games/swat/control_seed1.mp4` | `games/swat.py --seed 1 --record out/games/swat/control_seed1.mp4 --seconds 40 --control --render-scale 0.5` | 23:34:34 | 33.06 / 1077.4 s | 1,653 |
| `out/games/swat/control_seed2.mp4` | `games/swat.py --seed 2 --record out/games/swat/control_seed2.mp4 --seconds 40 --control --render-scale 0.5` | 23:34:57 | 33.06 / 1103.2 s | 1,653 |

Every clip is 1920 x 1080 at 50 fps (H.264, yuv420p); in the controls the camera view inside it is rendered at half
resolution. Each clip ended 3.0 s after its end card (29.66, 28.94 and 29.33 s for seeds 0, 1 and 2; 30.06 s for
every control). Clip time = brain time - 0.02 s: frame *f* shows the state after tick 2(*f* + 1), and the HUD's
"brain t" is the log's clock. An event on an even 10 ms tick is first on screen at its brain time - 0.02 s, one on an odd
tick at its brain time - 0.01 s (seed 0's swing-1 takeoff at 2.29 s is first on screen at 2.28 s in the clip). Times
below are brain times.

### What happened, swing by swing

| # | swing | speed | seed 0 | seed 1 | seed 2 | control 0 / 1 / 2 |
|---|---|---|---|---|---|---|
| 1 | from above | 0.5 m/s | +100 | +90 | +90 | SPLAT, no GF crossing |
| 2 | from the front, 45 deg up | 0.6 m/s | +90 | +110 | +100 | SPLAT, no GF crossing |
| 3 | from the left, 35 deg up | 0.7 m/s | +30 | +80 | +30 | SPLAT, no GF crossing |
| 4 | from above | 0.8 m/s | +50 | +30 | +40 | SPLAT, no GF crossing |
| 5 | from the right, 35 deg up | 0.9 m/s | +30 | +60 | +40 | SPLAT, no GF crossing |
| 6 | from behind, 45 deg up | 1.0 m/s | SPLAT 0 | SPLAT 0 | SPLAT 10 | SPLAT, no GF crossing |
| 7 | from the front, 45 deg up | 1.2 m/s | +10 | +50 | +30 | SPLAT, no GF crossing |
| 8 | from above | 1.5 m/s | SPLAT 10 | SPLAT 0 | SPLAT 0 | SPLAT, no GF crossing |
| 9 | from the left, 35 deg up | 2.0 m/s | SPLAT 20 | SPLAT 10 | SPLAT 30 | SPLAT, no GF crossing |
| 10 | from above | 2.5 m/s | SPLAT 20 | SPLAT 20 | SPLAT 20 | SPLAT, no GF crossing |
| | **score** | | **6 - 4** | **6 - 4** | **6 - 4** | **0 - 10** each |

(+n = IN TIME: the body took off n ms, in whole 10 ms ticks, before the paddle's arrival tick. SPLAT n = DNp01
crossed 33 Hz n ms after contact; 0 = on the arrival tick itself, which the referee scores for the paddle.)

* **In time: 18 of 30, the same six swings in every run.** Swings 1-5 and 7 (0.5-1.2 m/s, from above, the front and
  the sides) were in time in all three runs, with margins of 10-110 ms. When the body took off, the paddle still needed
  5.6-101.3 ms to reach the spot (`paddle_ttc_at_takeoff_ms`). The closest call was seed 0, swing 7 (1.2 m/s from
  the front): DNp01 crossed 33 Hz (36.9 Hz) on the tick when the paddle was 5.6 ms from the spot, a +10 ms margin.
  The widest was seed 1, swing 2 (0.6 m/s from the front): +110 ms, with the paddle 101.3 ms away.
* **Then the hop met the paddle in the air, 18 of 18**, 10-60 ms after takeoff (`contact_where: air`). The paddle never
  reached an empty spot. The 45 deg forward hop is not scored (Design).
* **Splat: 12 of 30, the same four swings in every run.** They were swing 6 (from behind, 1.0 m/s) and swings 8-10
  (1.5, 2.0 and 2.5 m/s). Four were ties, where DNp01 crossed 33 Hz on the arrival tick itself: seed 0 swing 6 (35.5 Hz),
  seed 1 swings 6 (40.2 Hz) and 8 (33.9 Hz), and seed 2 swing 8 (34.1 Hz). In the other eight, DNp01 had reached
  8.7-27.5 Hz by contact and crossed 33 Hz 10-30 ms later. At 2.5 m/s it had reached 9.5, 8.7 and 18.8 Hz by contact.
  No swing at 1.5 m/s or faster was in time (0 of 9), and neither was any swing from behind (0 of 3).
* **After each splat** the pin (GAME, 0.4 s) held the body. When it was released, the body jumped by the escape rule,
  12 times in 12, with DNp01 at 57.2-85.8 Hz.
* **The neurons** (the per-swing peaks the log keeps). In the 18 in-time swings (from the swing's start to the tick the
  hop met the paddle, 10-60 ms after takeoff), DNp01 peaked at 37.3-67.9 Hz, the
  LPLC2 population mean at 3.9-6.0 Hz, LC4 at 0.4-1.8 Hz and TTMn at 10.7-43.2 Hz. In the 12 splat swings, up to and
  including contact, LPLC2 peaked at 2.1-5.0 Hz, LC4 at 0.3-1.2 Hz and TTMn at 4.9-30.4 Hz.
* **Nothing else took off.** No run had a takeoff outside a swing or a pin release (`takeoffs_unprompted`: 0 escape,
  0 voluntary). Each run had 10 takeoffs and 10 landings, and no `clock_mismatch` event. On the ground between
  swings (>= 0.5 s after a landing), DNp01 had a median of 3.30 / 2.72 / 3.38 Hz, a 99th percentile of
  17.74 / 16.95 / 18.00 Hz and a maximum of 23.91 / 27.52 / 23.07 Hz (seeds 0 / 1 / 2, over 1,853 / 1,809 / 1,826 ticks).
* **Replays** (photo finishes at 0.25x): 7 in seed 0 (swings 3, 5, 6, 7, 8, 9, 10), 5 in seed 1 (4, 6, 8, 9, 10) and 6
  in seed 2 (3, 6, 7, 8, 9, 10).
* **Where the fly went.** Each fly crossed the table, mostly in its ten hops, from (0.50, 0.00) m to the far end
  (x = -0.58 m; y = -0.33, +0.33 and -0.08 m). The log's heading at each swing start, from 180 deg:
  * Seed 0 turned left, to -129.7 deg by swing 6. The release hop after swing 6 landed at the table's side fence
    (y = -0.388 m), which turned it back to within 6.5 deg of 180 (173.5 to -174.0 deg) for swings 7-10.
  * Seed 1 turned **right**, to 146.9 deg by swing 5, and ended at 162.3 deg. So the dev runs' leftward turn (Design)
    held in seeds 0 and 2 but not here.
  * Seed 2 turned left, to -152.3 deg by swing 8. At the table's far end the fence then turned it to 89.8 deg before
    swing 10.

  The eye stayed at least 14.5, 9.4 and 13.6 cm from the apple, and there was no eye-inside-fruit event.
* **Against the dev runs** (Design) the pattern is the same. Across the eight dev runs, swings 1-5 and 7 were always
  in time and 8-10 never were. Swing 6 was in time once (seed 101, +10 ms), and here it was not in time in any of the
  three runs.

Per clip, in order:

* **Seed 0** (`seed0.mp4`): 6 in time (+100, +90, +30, +50, +30, +10 ms), 4 splats. Swing 7 is the closest call of all
  30 swings (above), and its replay starts at 20.2 s. Swing 6 was a tie: DNp01 reached 35.5 Hz on the arrival tick,
  and the body jumped at 16.79 s when the pin was released (57.2 Hz). Swings 8-10 crossed 10, 20 and 20 ms late.
* **Seed 1** (`seed1.mp4`): 6 in time (+90, +110, +80, +30, +60, +50 ms), 4 splats. It had the widest margin (swing 2,
  +110 ms) and two ties (swings 6 and 8). Swings 9 and 10 crossed 10 and 20 ms late. This is the one run whose heading
  drifted right.
* **Seed 2** (`seed2.mp4`): 6 in time (+90, +100, +30, +40, +40, +30 ms), 4 splats. Swing 6 crossed 10 ms late and swing
  8 was a tie. Swing 9 (2.0 m/s from the left) was the latest crossing of the 12 splats: DNp01 was at 16.8 Hz by
  contact and reached 43.0 Hz 30 ms later. The body jumped at 24.87 s, when the pin was released (74.8 Hz).

### The control: the swatter removed from the fly's world

Same seeds, the same schedule and everything else, but the fly's eyes do not see the swatter. The camera still draws
it, and the screen says "CONTROL RUN: THE FLY CANNOT SEE THE SWATTER". The result was **0 takeoffs in 30 swings**:
30 splats (10 per run), 0 takeoffs after a pin, 0 unprompted takeoffs, 0 replays. During the swings **DNp01 peaked at
1.6-24.0 Hz** (at most 14.8, 18.0 and 24.0 Hz in controls 0, 1 and 2) and never reached 33 Hz. In the recorded runs
it peaked at 37.3-67.9 Hz in the in-time swings. Here is the same seed and swing at the same instant, with and
without the swatter visible, read off the HUD at brain time 2.28 s (2.26 s in both clips; swing 1, the paddle 104 ms away). Seed 0 had DNp01 at 28 Hz,
LPLC2 at 2.3 Hz and TTMn at 20 Hz, and the paddle was dark across the top of both eyes in the mosaic. Its control had
DNp01 at 6 Hz, LPLC2 at 0.9 Hz and TTMn at 1 Hz, and no paddle in the mosaic.

Across the 30 control swings, the swing peaks were 1.2-4.2 Hz for LPLC2 and 0.08-0.84 Hz for LC4. These overlap the
in-time ranges (3.9-6.0 and 0.4-1.8 Hz) at their edges: two control swings reached 4.1-4.2 Hz of LPLC2, and 12 of the
30 were above the in-time minimum of LC4 (0.37 Hz). DNp01 does not overlap (at most 24.0 Hz here; the in-time swings crossed 33 Hz by
definition and peaked at 37.3-67.9 Hz). For TTMn they were 3.8-39.2 Hz, which overlaps most of the in-time range (10.7-43.2 Hz). In control 0's swing 7, TTMn peaked at 39.2 Hz while DNp01 peaked at
only 14.7 Hz, so TTMn is not a clean readout of the loom in this model. On the ground between swings, DNp01 had a median of
2.57 / 2.36 / 2.08 Hz, a 99th percentile of 17.19 / 16.39 / 15.81 Hz and a maximum of 22.49 / 23.52 / 22.16 Hz
(1,980 ticks each).

Without hops, the control flies only walked. They ended at x = 0.25, 0.25 and 0.24 m, about 25 cm from the start. Their
heading at the swing starts still drifted: from -179.0 to -164.3 deg in control 0 (15 deg to the left), from -179.2 to
166.5 deg in control 1 (14 deg to the right), and within 4.1 deg of 180 in control 2. The flies were nowhere near a
fence, so the walk itself is not perfectly straight. It drifts by up to about 15 deg in 30 s. That is small next to the hops' changes
of heading (Limits).

## Corrections (caption skeptic)

Checked against the six run logs (`out/games/swat/{seed,control_seed}{0,1,2}.json`), the dev logs (`stats2/`, `stats3/`,
`dev.json`, `dev_fix.json`, `det/`, `review_honesty/`), frames pulled from the clips, `games/swat.py`, `games/common.py`
and `flyverse/body.py`. Every per-swing number, provenance hash, frame count and dev-table cell matched. What changed:

1. Headline. Was: "the giant fibre launched the jump before the paddle landed in 18 of 30 swings, with 10 to 110 ms to
   spare. From 1.5 m/s up, the swatter won all nine." It left out that the hop met the paddle in the air in all 18. The
   "10 to 110 ms" are whole-tick margins, and the logged time the paddle still needed was 5.6-101.3 ms. It also left out
   the three splats from behind at 1.0 m/s. Now: in 18 of 30 swings the jump launched "with the paddle still 5.6 to 101 ms
   away. In all 18 the fixed forward hop then met the paddle in the air. From behind, and from 1.5 m/s up, the paddle
   arrived first every time (12 of 12)."
2. "Ballistic" (the not-claimed row, the Design table's walking row and Limits). Was: "the raw model has no sustained wing
   power, so the hop is ballistic" and "the hop is ballistic and lands about 10 cm on". The shipped `Flight.step` adds
   thrust and lift from the wing-power motor neurons (logged at 3.5-69.4 Hz at the 30 takeoffs) and yaw from the
   steering motor neurons. Now: not purely ballistic, and it came down after 0.25-0.36 s, about 9-13 cm on (landing
   minus takeoff events in the logs).
3. Heading. Was: "Its heading changes only in the air (the ballistic hop's yaw) and at the fence's edge" (Limits), and "The
   heading turns left by 25-65 deg over a run ... (the hops' in-air yaw; the walking fly goes straight)" (Design, Fruit).
   The in-air yaw is `Flight.step`'s reading of the steering motor neurons. The controls' walk drifted by up to 15 deg.
   The dev clip on the fixed referee (`dev_fix`) did not turn left, and neither did recorded seed 1. Now each place says
   where the turns came from, and the seven re-read dev runs turned left by 33-65 deg.
4. Design: "Nothing in the game reads the brain." The HUD and the log read its rates, and the log's DNp01 crossing after a
   splat decides the replays. Now: "No game control reads the brain", with the rest listed.
5. Design: "in most swats it meets the paddle in the air". Now: all 18 in-time swats. The not-claimed row now also says
   that the game stops the paddle at that meeting, the hop carries on and lands, and the meeting is not scored.
6. Control: "For TTMn they were 3.8-39.2 Hz, which overlaps the in-time range" suggested that LPLC2 and LC4 did not overlap.
   They do at the edges: LPLC2 reached 4.1-4.2 Hz in two control swings (in-time minimum 3.9), and 12 of 30 control LC4
   peaks were above the in-time minimum (0.37 Hz). DNp01 does not overlap.
7. The first paragraph now says that the 4 arrival-tick crossings are ties which the referee gives to the paddle (GAME).
8. Timing. "about 32-35 s in" is now 31.9-33.1 s (the logs' `brain_s`). "read off the HUD at 2.28 s" is now brain time 2.28 s,
   2.26 s in the clips (frames checked: seed 0 DNp01 28 / LPLC2 2.3 / TTMn 20, control 6 / 0.9 / 1). Added a note that
   events on odd ticks first show 0.01 s, not 0.02 s, before their brain time. "then the three controls did" now says
   that control 0 started about a minute before seeds 1 and 2 finished (file times).
9. Dev notes. The eight walking-GF runs are now named. An earlier build's seed-101 run, with a 33.7 Hz ground maximum and
   no launch, is mentioned. The prototype and the kinematic check are marked as having no kept log.
10. The in-time neuron peaks now state their window: from swing start to the air contact, 10-60 ms after takeoff.
