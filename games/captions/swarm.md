# swarm: 64 flies, one GPU

**64 flies, 64 brains, one GPU, one swatter. Over each clip's five swats the giant fibre fired in time for about two
thirds of the flies in the head's path (162 of 253, 170 of 254 and 163 of 258 in seeds 0, 1 and 2), and for none of
the 738 in its path in the blind controls, where the swatter was left out of the flies' scenes. The shipped escape hop is
not aimed: it goes forward along the fly's heading. 438 of the 495 flies that jumped in time were caught in mid-air,
among them 85 of the 110 that faced away from where the head came down.**

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the flies) | 64 copies of MaleCNS v1.0 as shipped (167,106 neurons each, preset `raw`), stepped as one batch: nothing instrumented, attached, trained or tuned (`instruments: []`, `modules: []` in every run log). Each fly's own ray-traced eyes feed its own brain, and nothing but its own giant fibre (DNp01) launches its escape hop. The discs under the flies, the 64-row raster and the focus fly's traces show DNp01, LPLC2 and LC4 rates. |
| **BODY MODEL** (shipped readout, not a game decoder) | `flyverse/body.py` via `BatchBody`: DNp01 >= 33 Hz launches the escape hop (0.6 m/s, 45 deg up, along the fly's heading), the power MNs drive the rare voluntary take-off, and `Locomotion.readout` drives walking. These are the shipped defaults, not tuned for the game. |
| **DECODER** (written for the game) | None. No mapping from neural activity to a game control was written for this game. |
| **GAME** (not the fly) | The swatter: its shape, path, speeds, schedule and aim. Also the tray and fence, the lamp moved below its globe, the crowd's start grid, the fresh crowd each round, and the outcome rule (splat / caught / clear). The camera, the 2.5x fly sprites, the banners and the blind control are the game's too. |
| **not claimed** | The flies do not aim their jumps: the hop goes where the fly faces. Most flies that jumped in time were caught in the air whichever way they faced (seeds 0-2 pooled, bearing to the head's landing point when the swat began: 198 of 207 facing it, 155 of 178 side-on, 85 of 110 facing away), and the hop is short beside a 20 x 17 cm head (What was measured first). The flies do not see each other and do not settle after an escape (Design). |

## Design

Sixty-four copies of the shipped fly (MaleCNS v1.0, `preset="raw"`, no instruments, 167,106 neurons each) run as one
batched brain, `flyverse.BatchSim(64)`: one `FlyBrain(batch=64)` stepped in lockstep, 10.7 million simulated neurons,
64 pairs of compound eyes ray-traced as one batch, 64 bodies. The flies walk on a white tray on the room's table. A
giant swatter comes down on the crowd, once per round, five rounds, faster each round (0.5, 1.0, 1.5, 2.0,
3.0 m/s). **The game has no decoders.** Nothing but a fly's own giant fibre launches its escape hop: the shipped body
model launches it when DNp01 reaches 33 Hz. The swatter, its aim and the scoring are the game's.

The game scores two questions separately, because their answers differ:

1. **Did the giant fibre fire in time?** An at-risk fly *jumped in time* if its DNp01 launched the escape hop before
   the head reached the tray. This is the connectome's answer.
2. **Did the hop carry it clear?** A fly *got clear* if the head never touched it. This is the answer of the body
   model's hop, which goes forward along the fly's heading at 45 deg. A fly's hop can start in time and still carry it
   into the swatter's path. Most of the flies that jumped in time were caught in the air.

Each round starts a fresh crowd: all 64 brains and bodies are reset and laid out on the start grid again. That is a
game device, and it is there because the model does not settle after an escape. Landings drive the giant fibre
again: in the builder's seed-101 chain test (one 0.5 m/s swat, no reset), each of the 32 landings made after the
swatter had gone was followed within 0.5 s by a DNp01 rate above 40 Hz (up to 114 Hz; median peak 0.2 s after
touchdown, with LPLC2 and LC4 also above their pre-hop rates). Which part of the scene drives it was not isolated. So
flies that have hopped hop again, 1.3-2.4 s apart (10th-90th percentile of the 14 re-hop intervals; median 1.4 s, longest 3.5 s), past the body's 1 s landing refractory. Without resets, later swats would score flies that were hopping anyway. The hops after each swat stay in
the clip and are counted in the run log.

| what | kind | law and parameters |
|---|---|---|
| the brains | CONNECTOME | `BatchSim(64, seed=s, fruit_set="apple", fence=True, cuda_graphs=True, cuda_kernels=True, event_driven=True)`: one `FlyBrain(batch=64)`, preset `raw`, nothing attached (`module_records()` is empty). Batch rows share one RNG stream (docs/BATCH_SIM.md). |
| the eyes | physics | each fly's 1,466 columns x 7 rays traced through its own copy of the scene (`BatchWorld`): room, table, apple, tray, rim, swatter. The tracer is spectral (UV, B, G, R); no RGB-to-UV approximation. **The flies do not see each other.** |
| escape hop | body model (shipped readout, not a game decoder) | `Flight` via `BatchBody`: DNp01 mean rate >= `Flight.gf_hz` = 33 Hz launches the escape hop at 0.6 m/s, 45 deg up, **along the fly's heading** (`Flight.launch`). In the air the body model's flight step adds thrust and lift from the power MNs; the raw model has no sustained flight, so the fly comes down again (dev seed 101: apex 24-55 mm, median 38; air time 0.25-0.47 s, median 0.36; 10th-90th percentiles of the 357 of 359 hops that landed within the run; the honesty review's replay, What was measured first). No new escape within 1.0 s of landing (`landing_refractory_s`). |
| voluntary take-off | body model (shipped readout) | power MNs >= 50 Hz held for 0.3 s launch a 0.2 m/s take-off. Logged as its own event (`takeoff_voluntary`). A fly that gets clear this way is counted as `clear_by_voluntary_takeoff`, not as a hop. |
| walking | body model (shipped readout) | `Locomotion.readout`: speed = a constant 0.8 cm/s (`baseline_speed`) plus forward-DN and leg-MN terms, capped at 3 cm/s; the constant dominates (games/README.md, finding 4). The raw model walks straight. |
| room air | GAME (shipped defaults) | the shipped `BatchSim` room, unchanged: 0.3 m/s wind blowing from +x to -x with a 20 deg, 8 s meander, and the apple's odour plume. Every fly smells and feels it through `fb.smell` and `fb.wind`. The apple sits outside the tray, so no fly tastes it. The same in the control run. |
| tray and fence | GAME | a 34 x 34 cm white tray (`plate`) with a 12 mm dark wooden rim, on the room's table. The body model's fence keeps the flies within +-15.8 cm (at the fence a fly turns towards the centre at 90 deg/s). |
| room light | GAME | the room's point light moved 1 cm below the lamp's globe. In `world.make_room` it sits at the globe's centre, so every shadow ray hits the globe and the room is lit by ambient light alone. Here the lamp lights the tray and the swatter casts a shadow (the flies see it too). `--room-light shipped` keeps the original. |
| the crowd | GAME | 64 flies on an 8 x 8 grid, 2.4 cm apart, jittered by 4 mm (SD), uniformly random headings; numpy seed `[seed, r - 1]` for round r. |
| rounds | GAME | round *k* (k >= 2) begins at 4.5 (k-1) s with `BatchSim.reset` of all 64 rows: each row's brain (`FlyBrain.reset`, the shared clock kept) and body back to the start grid. The swat comes 1.5 s later (round 1: at 2.0 s). A splat or caught fly is out for the rest of its round. |
| the swatter | GAME | a black flattened ellipsoid head, semi-axes 10 x 8.5 x 0.6 cm (20 x 17 cm), with a flat 26 cm rod handle that rides 13 mm above the tray when the head rests, clear of the rim. It comes in along a straight line at 65 deg above horizontal from the handle's side, starting 45 cm above the tray, at the round's constant speed. It rests on the tray 0.2 s, then lifts back at 1.2 m/s. Handles alternate -x / +y. |
| aim | GAME | when a swat begins: the head position (1 cm grid, head inside the rim) covering the most live flies on the tray; ties nearest the tray's centre. |
| outcome | GAME | **at risk**: under the head's footprint (+1.5 mm) when the swat begins or at contact, or caught. **Jumped in time**: at risk, and the giant fibre launched the escape hop in a 10 ms step that ended before contact. **SPLAT**: on the tray under the footprint at contact, scored on the last state before the head reaches the tray (at most 10 ms earlier). **CAUGHT**: the head or the handle (radii + 1.5 mm) met the fly's path while it was in the air (airborne at either end of a 10 ms step), tested every step with both moving in straight lines over the step (a swept test, so a fast head cannot skip a fly), from the swat's start to the end of the head's 0.2 s rest on the tray. A fly walking on the tray is scored by the footprint at contact instead. The body model has no swatter surface, so a fly that meets the resting swatter counts as caught, not as landed on it; the log counts these separately as `caught_on_resting_head`, a name that also covers the resting handle and late hops into the head's edge. **CLEAR**: at risk, neither splat nor caught: the head never touched it; the log says how (`clear_by_hop`, `clear_walked`, `clear_by_voluntary_takeoff`, `clear_airborne_before`). **STARTLED**: not at risk, hopped during the descent. **LATE**: an escape hop launched in the contact step or during the rest (counted, not scored). |
| HUD | CONNECTOME | the disc under each fly, the 64-row raster and the focus fly's trace: DNp01 mean rate, from `fb.motor().gf` (what the body reads), with the 33 Hz line. LPLC2 (185 cells) and LC4 (126) mean rates of the focus fly: read-only, they control nothing. |
| outcomes panel and final card | GAME | per round: jumped in time / at risk, got clear, hit (caught + splat). The polar map puts each at-risk fly's view of the swatter's **landing point** (its aim) at the moment the swat began: bearing in the fly's own frame, distance in head radii. Colour = outcome, filled dot = jumped in time. |
| what the fly sees | display | the focus fly's radiance exactly as handed to `fb.vision`, as a 1,466-hexagon mosaic (exposure 1.2). The focus is the at-risk fly nearest the aim; the mosaic freezes with SPLAT or CAUGHT when it is hit. |
| main view | display | the same geometry and light, ray-traced for a human camera slowly orbiting at 43 deg elevation, with texture noise 0.5 (the flies' scenes keep the room's 1.4). The camera pulls back over 0.45 s before each swat so that the whole head is in frame, then pushes back in as the head comes down (the least pull-back that keeps the head in frame, and no faster than over 1 s, so the fast swats are seen wide). The impact shake moves the camera by up to 3 mm. Flies are drawn as sprites at 2.5x life size (a 2.6 mm fly is about 6 px there) and never smaller than 12 px, so larger than 2.5x in the wide framing; they are hidden where the swatter covers them. A splat fly and a caught fly are both drawn as the same red splat mark on the tray, at the point below where they were hit, so the marks count hits (caught + splat), not splats. The main-view disc colour stops at red (50 Hz); the raster's goes on to white. The on-screen wall rate times the simulation step only (eyes, 64 brains, bodies), on a GPU shared with other jobs. |
| control run | GAME | `--control blind`: same seed; the swatter (head, handle, shadow) is left out of all 64 flies' scenes. The main view still draws it and the same rule scores the run. |

### What was measured first (dev seeds >= 100)

Numbers from the game's own code unless a line says it was a prototype. Two kept sources are earlier versions of
that code: the builder's seed-101 chain arrays (`builder_scratch/chain_*_101.npz`; the harness sets
`swarm.SCHEDULE`, which the frozen file does not have, and in that version only splat flies left the game, so flies the
frozen rule would score as caught stay in their hop counts) and the honesty review's replay
(`review_honesty/rerun_101_none.*`, which scored "escaped" with no mid-air test). The brain and the body model they ran
are the shipped ones. The frozen file is `games/swarm.py` sha256 `3a653c1d9fc0d4bf`. `--measure` runs the rounds without drawing and writes the same run log a clip does; the seed-101
logs (`out/games/swarm/measure_seed101.json`, `measure_seed101_blind.json`) were written by `ea8f41f1d05873d5`,
which by the builder's account differs from the frozen file in drawing only (the final card's layout and two banners'
durations). That version's source was not kept, so the diff cannot be re-checked; its seed-100 render
(`superseded/dev_ea8f41f1.json`) and the frozen file's (`dev.json`) agree to within 2 flies per round in jumped in
time, and to within 3 in every count (got clear in round 1: 14 against 11). Prototype
lines marked "no kept log" below come from the builder's session; their runs left no log or array in
`out/games/swarm/` (the arrays that were kept are in `builder_scratch/` and `review_honesty/`).

* **Batch 64 fits.** (Memory and quiet-GPU timings: no kept log; the build times in the recorded logs are 11.3-14.9 s
  on the shared RTX 4090 and 12.0-12.1 s on the B200.) `BatchSim(64)` builds in about 7-9 s and peaks at 1.76 GB allocated with the default backend (160 ms
  per 10 ms frame on the shared RTX 4090), or 1.95 GB with CUDA graphs + native kernels + event-driven LIF (the
  flags docs/BATCH_SIM.md uses), at 100-140 ms per frame. On a quiet GPU the simulation (64 brains, eyes and bodies)
  took 13.7 s of wall per brain second and the whole game about 20 s with the 1344 x 967 render and the HUD. With the
  GPU shared by other jobs the same step took 42 s (seed 101 measure runs). The on-screen figure is this simulation
  step, whatever the GPU's load was during the run.
* **The loom reaches the giant fibre in a crowd** (prototype: seed 100, vertical swatter from 60 cm, shipped room
  light, no tray; `builder_scratch/proto1_64_*.npz`, re-checked). At 0.8 m/s, 42 of 64 flies launched before contact. The flies under the centre launched 70-110 ms
  before it. At 1.6 m/s the centre flies still launched 20-40 ms before contact. Walking giant fibres before the
  swat: per-fly maxima at median 15 Hz, 99th percentile 24.5 Hz. A freshly built brain peaked at 22.8 Hz in its first
  0.5 s, and no recorded run had a hop in the first 0.5 s of a round. Fresh crowds do make false starts later in the
  walking time: 1 in seed 0 (fly 8 at 2.00 s, 33.9 Hz) and 2 in seed 1 (flies 29 and 28, 1.04 and 0.99 s after a
  reset; its blind control logged the same two), none in seed 2 or seed 101.
* **The crowd disperses, so the swatter aims.** Four swats at one fixed spot on one crowd (seed 101) covered 38,
  10, 0 and 0 flies (no kept log). Hence the aim rule.
* **The shipped room has no direct light.** Looking down at the table, the shipped room gives radiance 0.014-0.113 (G);
  with the globe out of the way, 0.22-1.70, about 15x (no kept log; the finding itself is confirmed in
  captions/sunrise.md and games/README.md). Moving the light 1 cm below the globe barely changed the giant fibre's
  response. On seed 100 (one swat, 0.5 m/s): 40 of 48 at-risk flies launched in time vs 38 of 46 with the shipped
  light, and pre-swat giant-fibre maxima were 26.1 vs 33.6 Hz, with one spontaneous hop under the shipped light (no
  kept log; the kept seed-101 chain arrays below show pre-swat maxima of 25.0 vs 33.7 Hz).
* **The model does not settle after an escape.** Five swats on one crowd (seed 100, 25 s, no resets): afterwards
  there were 111 hops outside the swat windows, and the giant fibre's 99th percentile there was 75 Hz, against 17 Hz
  before the first swat (no kept log). One swat (seed 101, 0.5 m/s) was followed by 40 hops from 26 flies in the next
  5 s. With the shipped light the figure was 26 from 17, so the new light is not the cause
  (`builder_scratch/chain_{globe,shipped}_101.npz`: 41 take-offs from 27 surviving flies, which include any voluntary
  take-off, and 26 from 17). In the same arrays, every one of the 32 landings made after the swatter had gone was
  followed within 0.5 s by a giant-fibre rate above 40 Hz (up to 114 Hz), and 14 of 18 with the shipped light (up to
  85 Hz). This is why each round resets the crowd.
* **The hop goes forward, and it is short.** Over the 357 of 359 hops that landed within one seed-101 run (the
  honesty review's per-tick replay) the hop's apex was 24-55 mm (10th-90th percentile; median 38, maximum 89) and the fly was in the air
  0.25-0.47 s (median 0.36). The head is 20 x 17 cm. A fly under it that launches in time is still over the head's footprint, or in its path, when the head
  comes down, unless it started near the rim and faced outward.
* **Rounds, seed 101** (`measure_seed101.json`). *At risk* counts the flies under the head (see the outcome rule).
  *Jumped in time* is the giant fibre's answer, *got clear* the hop's:

  | round | speed | at risk | jumped in time | got clear (by the hop) | caught in the air | splat | median lead of the hop | median GF peak before impact, at risk (blind control) |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 0.5 m/s | 53 | 46 (87 %) | 16 (12) | 31 | 6 | 178 ms | 66 Hz (14 Hz) |
  | 2 | 1.0 m/s | 49 | 42 (86 %) | 5 (4) | 38 | 6 | 102 ms | 52 Hz (9 Hz) |
  | 3 | 1.5 m/s | 52 | 39 (75 %) | 1 (1) | 38 | 13 | 41 ms | 42 Hz (9 Hz) |
  | 4 | 2.0 m/s | 47 | 26 (55 %) | 1 (1) | 25 | 21 | 33 ms | 34 Hz (9 Hz) |
  | 5 | 3.0 m/s | 48 | 11 (23 %) | 0 | 11 | 37 | 16 ms | 24 Hz (6 Hz) |

  In total 164 of 249 at-risk flies jumped in time, and 23 got clear: 18 by the hop, 4 walked out, 1 by a voluntary
  take-off. Of the 164 that jumped in time, 143 were caught in the air (141 during the descent, 2 during the head's
  rest: fly 48 came down onto the head 9.5 mm up, and fly 7 met the resting handle 23.2 mm up, beyond the head's rim)
  and 3 landed back under it. So the giant fibre fired in time for most flies up to 1.5 m/s, and
  the forward hop carried few of them clear. 5 escape hops were late (launched in the contact step or during the
  rest), 11 flies outside the footprint hopped during a descent (startled), and 2 flies took off voluntarily. No fly
  hopped in the walking time before any swat (giant fibre 99th percentile 17.7 Hz, maximum 32.1 Hz). The hops after
  contact, until the next reset: 3, 4, 0, 3 and 5.
* **Heading, pooled over the five rounds, seed 101.** The bearing is to the swatter's landing point, in the fly's
  own frame, when the swat began. At the rim (rho >= 0.8), the giant fibre fired in time for 35 of 39 flies facing
  the landing point (|bearing| < 60 deg), 19 of 30 side-on, and 9 of 31 facing away (|bearing| >= 120 deg). The
  eyes cover azimuth -121 to +115 deg, so a looming edge behind the fly falls in the blind sector (prototype, seed
  101, 0.8 m/s: rim flies facing the swatter launched 9 of 9, facing away 0 of 9; no kept log, and
  `builder_scratch/proto2_101.npz` keeps rates only). **The physical outcome runs the
  other way:** the flies facing the landing point jumped towards it, and 0 of those 39 got clear; 7 of the 31 facing
  away got clear. Over all at-risk flies the giant fibre fired in time for 74 of 89 facing the landing point, 53 of
  77 side-on and 37 of 83 facing away; 1, 13 and 9 of them got clear. Distance mattered less for the giant fibre:
  inner half of the head 40 of 54, middle 61 of 95, rim 54 of 91 and beyond it 9 of 9 (1, 8, 14 and 0 got clear).
* **Control, seed 101, `--control blind`** (`measure_seed101_blind.json`; the swatter not in the flies' scenes): 0
  of 245 at-risk flies jumped. 240 were splat, 5 walked clear and none was caught. The at-risk flies' median
  giant-fibre peak before impact was 6-14 Hz. One fly outside the footprint hopped during a descent. No at-risk fly
  hopped in the control, so the main run's hops in time depend on the swatter being in the flies' scenes.
* **Run to run.** Two runs of the same simulation on seed 101 (the scoring changed between them, not the model):
  jumped in time 45/52, 41/48, 39/52, 25/47, 12/49 and 46/53, 42/49, 39/52, 26/47, 11/48. Only the second run's log
  is kept (`measure_seed101.json`); the first set is the builder's record. The two logged seed-100 renders (next
  bullet) are a kept run-to-run pair.
* **Dev clip, seed 100** (`out/games/swarm/dev.mp4`, rendered from the frozen file): jumped in time
  41/48, 37/50, 37/49, 22/50 and 23/47 at 0.5 to 3.0 m/s (median leads 203, 77, 51, 28 and 16 ms). Got clear: 11 (9
  by the hop), 1, 3, 2 (1) and 0. Caught in the air: 31, 36, 34, 21 and 23. Splat: 6, 13, 12, 27 and 24. In total 160
  of 244 at-risk flies jumped in time, 17 got clear, 145 were caught in the air (1 on the resting head) and 82 were
  splat. At the rim, the giant fibre fired in time for 25 of 27 flies facing the landing point, and none of them got
  clear; for 5 of 37 facing away, and 5 of those 37 got clear. On this seed the 3.0 m/s round (23 of 47) came out
  above the 2.0 m/s round (22 of 50). An earlier render of the same simulation (only drawing changed since) gave
  41/48, 39/50, 37/49, 22/50 and 21/47 (`superseded/dev_ea8f41f1.json`). Other jobs were loading the GPU: the
  on-screen simulation rate ran between about 12 and 100 s of wall per brain second in this render (frames every 2 s),
  and the 24 s clip took 23 minutes.
* Earlier versions of this caption reported "escaped" (the giant fibre launched the hop before contact, with airborne
  flies never tested against the swatter). That counted what is now *jumped in time*, and it was not an escape: the
  reviewers' per-tick replay of seed 101 found that the head met the path of 144 of the 161 flies then scored
  "escaped". The superseded logs and clip are in `out/games/swarm/superseded/`.

### Limits

* **The hop's direction is the body model's.** `Flight.launch` sends every escape hop forward along the fly's
  heading at 45 deg. Real flies aim the escape jump away from the looming object (Card and Dickinson 2008); the
  shipped body does not. The giant fibre's timing is the model's answer here, the jump's direction is not, and the
  heading result above is a result about the eyes and the giant fibre (which flies fired in time), not about
  escape strategy.
* **Caught is a geometric test on a point.** A fly is a point with a 1.5 mm margin, tested against the head's and
  the handle's ellipsoids; the fly's own body is not modelled. A caught fly leaves the round: being knocked aside,
  stunned or surviving is not modelled. The body model has no swatter surface, so a fly that meets the resting
  swatter counts as caught, although it could have landed on it (on seed 101 one fly came down onto the head and one
  onto the handle).
* The flies do not see each other, and nothing carries the swatter's air ahead of it (a real fly feels the bow wave).
  The swatter moves in a straight line at constant speed; a real swat accelerates and swings about the wrist.
* The rounds and resets are the game's. Within a round the hops after contact are the model's (driven by the
  landings; Design). The body model's 1 s refractory does not stop them.
* "GF peak before impact" is a maximum over the descent, which shortens from 0.99 s to 0.17 s as the speed rises.
  The blind control's column shows how much of the fall comes from the shorter window alone (14 to 6 Hz); with the
  swatter visible the peak falls from 66 to 24 Hz.
* The outcome map and the heading bins use the bearing to the swatter's landing point (its aim) when the swat began,
  when the head is still 45 cm up and about 21 cm towards the handle's side. The raw model walks straight, so
  headings change little during a 0.17-1.0 s descent.
* Batch rows share one RNG stream, and the native event kernels are not bit-reproducible run to run, so a re-run of
  a seed gives similar but not identical numbers. Jumped in time moved by up to 1 fly per round on seed 101 (only the
  second run's log is kept) and by up to 2 on the two logged seed-100 renders, whose got-clear counts differ by up to
  3. Seed 2's stopped desktop attempt against its B200 recording, a different device as well as a re-run, moved
  jumped in time by up to 2, caught in the air by up to 5 and got clear by up to 7 (round 1: 19 against 12; Clips).
  Got clear, a few flies per round after round 1, moves the most for its size. Changing `--batch` changes every row's
  random draws.
* The room light, the tray, the camera's lighter texture, the camera's pull-back around each swat, the impact shake
  and the 2.5x fly sprites are presentation and scene choices, declared above. The fly's-eye mosaic shows what the
  focus fly's `fb.vision` received.

    python games/swarm.py --seed 0 --record out/games/swarm/seed0.mp4 --screenshot out/games/swarm/seed0.png
    python games/swarm.py --seed 0 --control blind --record out/games/swarm/control_seed0.mp4
    python games/swarm.py --seed 101 --seconds 22.5 --measure --log out/games/swarm/measure_seed101.json   # the rounds, no drawing
    python games/swarm.py --batch 16                      # interactive: click the tray to swat, 1-5 speed, R new crowd, A rounds

## Clips

Six clips, from two machines. **Seeds 0 and 1 and their blind controls** were recorded on 2026-09-28 on this desktop:
`cuda`, NVIDIA GeForce RTX 4090, torch 2.10.0+cu128, Python 3.13.2. **Seed 2 and its blind control** were recorded on
the project's compute cluster, where all GPU work moved once the desktop was needed: `cuda`, NVIDIA B200, torch
2.11.0+cu128, Python 3.12.3. Both seed-2 jobs were submitted together in one `scripts/cluster_run.py` call and ran at
the same time, on one GPU. The run logs record only the GPU model (NVIDIA B200). The cluster job manager's records
(jobs `e0888766c5e0` and `e93fa46402e1`) place both on `<cluster-node>`'s GPU 3, from 00:35:54 to 00:42:47 UTC. Seed 2 had been started on the desktop first and was killed unfinished (below).

**Every comparison below is made within one device type:** a seed's connected run against its own blind control, which
ran beside it on the same kind of GPU. Seed 2 differs from seeds 0 and 1 in both its seed and its machine (GPU, torch 2.11 vs
2.10, Python 3.12 vs 3.13; both torch builds are cu128). Its numbers stand beside theirs for reading, not as a test of either. Even on one GPU, a re-run of a seed does
not repeat exactly (the native kernels are not bit-reproducible; Limits).

Provenance, from the run logs:
* `games/swarm.py` sha256 `3a653c1d9fc0d4bf` (first 16 hex) in all six. This is the frozen file named above, the same
  one that rendered the seed-100 dev clip. It was not changed during the recordings.
* `games/common.py` is `b110142562a927a1` in the four desktop logs and `e06078fb69d1275a` in the two cluster logs. The
  lead changed common.py once in between, and only in its ffmpeg lookup. `find_ffmpeg` now falls back to the ffmpeg
  that the `imageio-ffmpeg` package bundles, because the cluster has no ffmpeg on PATH, and `Recorder` calls it. A
  byte-code comparison with a cached compile of the earlier file found no other change. The encoder settings are the
  same (H.264 High, CRF 16, yuv420p, 1920x1080, 50 fps); the ffmpeg builds differ (the file tag reads Lavf62.3.100 on
  the desktop, Lavf61.1.100 on the cluster).
* Commit: the desktop logs record `959f2e927b1c2806bc526fd5fc51cb3f5cc3e4b1` with `dirty: true` (games/ is not
  committed yet). The cluster logs record `commit: null` and `dirty: null`. A cluster job runs in a fresh copy of the
  cluster's own checkout with this working tree's changes (the files that differ from `origin/main`) laid over it, and
  that copy is not a git checkout. The checkout's reflog shows it at `959f2e9` (= `origin/main`) since 2026-09-22, with
  no tracked changes. The `sources` hashes above identify the game code.
* One `FlyBrain(batch=64)` in each clip: batch 64 fit on both GPUs, so no smaller batch was needed. Preset `raw`,
  `instruments: []`, `modules: []`, backend CUDA graphs + native kernels + event-driven LIF. `meta.declared` lists the
  eight GAME rules (room air, tray, room light, crowd, rounds, swatter, aim, outcome rule), and the controls add
  `blind control`. There is no decoder.

The commands (argv as logged, identical on both machines apart from the seed):

    python games/swarm.py --seed {k} --seconds 24 --record out/games/swarm/seed{k}.mp4 --screenshot out/games/swarm/seed{k}.png
    python games/swarm.py --seed {k} --seconds 24 --control blind --record out/games/swarm/control_seed{k}.mp4

On the desktop (k = 0, 1) each ran from the repo root as `PYTHONIOENCODING=utf-8 <miniconda>/python.exe`
followed by the arguments. The connected runs and the controls ran in two lanes side by side, on a GPU shared
throughout with the other environments' final recordings (spacecraft, hairdryer, minecraft, pong and swat ran at the
same time) and some of their dev runs. On the cluster (k = 2) each was one job of a two-job batch, with the cluster's `.venv` activated and
`SDL_VIDEODRIVER=dummy`, `PYTHONIOENCODING=utf-8`. Each job first ran a check that asserts CUDA is available and prints
the device name ("NVIDIA B200" in both), then the command, with its console redirected to
`out/games/swarm/seed2.console.txt` and `control_seed2.console.txt`. The call's own console is
`out/games/swarm/seed2_cluster.log`, and its final line begins `2 job(s), 0 failed` (7.6 min for the batch). The
seed-2 GPU was shared too, with other cluster jobs. The job manager's records put five of pong's six cluster jobs
(`pong-swap-5fab86-0` to `-4`; the sixth ran on GPU 0) on the same GPU 3 from 00:36:56 UTC to 00:45:54. What the run
logs show: pong's six cluster
runs (`out/games/pong/b200_seed*.json` and `swapped_seed*.json`, one `cluster_run.py` call whose console notes "2
job(s) of tel0s already pending/queued/running", the size of the swarm batch) started at 00:37:02-00:37:05 UTC and ran 396-480 s of wall, over the same minutes, and the
swarm pair slowed at the same time, from about 10 s of wall per brain second to 17-21 s after brain second 6.

Each of the six commands ran to the end exactly once, and all six clips are kept. None crashed or ran out of CUDA
memory. Each clip is 24.0 s of brain time (1,200 frames, 1920x1080, 50 fps). The connected runs also saved their last
frame (`seed{k}.png`: the final card). "Wall" is the log's `wall_s`: the game loop and encoding, after the brain was
built. It swung with the GPU's load and with the device. Recording is offline, so the load changed only the wall
time, not the time step, the inputs or the rules. The device is another matter: a different GPU and torch build can
change the numerics (as a re-run on one GPU can; Limits), which is why each seed is read against its own control.
The HUD's sim-step line names the GPU: "RTX 4090, shared" in the desktop clips,
"NVIDIA B200, shared" in seed 2's. The seed-2 clips also look different: the cluster has neither of the HUD's
first-choice fonts in `games/common.py` (Cascadia Mono, Bahnschrift), so their text is set in the DejaVu fallbacks.
Only the typeface changes; the panels, banners and the numbers' sources are the same.

**Seed 2's first attempt, killed.** Seed 2 and its control were first started on the desktop, as the third run of each
lane (lane logs: control at 23:39:28Z, seed 2 at 23:48:11Z, 2026-09-28 UTC). The lead stopped both at 00:28Z on
2026-09-29, when GPU work moved off the desktop; the lane logs record `exit 1`. By their last progress lines they had
reached 14.0 s (seed 2) and 12.0 s (control) of the 24 s. Neither wrote a run log, and their console logs print
progress only. Neither video opens as a file (ffprobe: "moov atom not found"; the encoder was never closed), but the
H.264 stream inside each is intact up to where the encoder stopped. With the stream headers (SPS / PPS) of `seed0.mp4`
put in front (its x264 settings string is byte-identical to the one inside both streams), they decode: 623 frames
(12.46 s) of seed 2 and 532 frames (10.64 s) of its control. The caption skeptic recovered them after the recordings,
into a scratch directory; nothing in the kept files shows that anyone had seen these frames before the restart. Their
result banners, drawn by the same scoring that writes the logs, read:

| desktop attempt, RTX 4090 (killed) | at risk | jumped in time | got clear | caught in the air | splat |
|---|---|---|---|---|---|
| seed 2, round 1, 0.5 m/s | 53 | 46 | 19 | 30 | 4 |
| seed 2, round 2, 1.0 m/s | 48 | 39 | 6 | 34 | 8 |
| seed 2, round 3, 1.5 m/s | 55 | 43 | 3 | 40 | 12 |
| control, round 1 | 51 | 0 | 5 | 0 | 46 |
| control, round 2 | 47 | 0 | 2 | 0 | 45 |

The recorded B200 run of the same seed and code gave 46 of 53, 39 of 51 and 41 of 54 jumped in time in those rounds,
with 12, 6 and 2 clear (its control: 0 of 50 and 0 of 49, 2 and 1 clear). The attempt was stopped for the move off
the desktop, not on an outcome, and on the fly's own measures the recording did no better than it. The pair also shows how far
one seed moves between two runs that differ in device as well as in the kernels' run-to-run noise: jumped in time by
up to 2 flies per round, at risk by up to 3, got clear by up to 7 (round 1: 19 against 12). The focus fly and its
banner differ too (round 1: fly 27, "35 Hz ... 143 ms" on the desktop against "113 ms" on the B200; round 2: fly 35
against fly 27). These partial numbers are read from frames, not from a log, and are in no total above or below. The
two .mp4 files and their console logs are kept, unchanged, in `out/games/swarm/killed_partial/`. The cluster runs are
the seed-2 recordings.

The six recordings:

| clip | device | started (UTC) | wall | at risk | jumped in time | got clear (by the hop) | caught in the air | splat | escape hops in 24 s |
|---|---|---|---|---|---|---|---|---|---|
| `seed0.mp4` | RTX 4090 | 2026-09-28 21:48 | 2,573 s | 253 | 162 (64 %) | 29 (23) | 140 | 84 | 204 |
| `control_seed0.mp4` | RTX 4090 | 2026-09-28 21:57 | 1,976 s | 246 | 0 | 5 (0) | 0 | 241 | 0 |
| `seed1.mp4` | RTX 4090 | 2026-09-28 22:31 | 4,589 s | 254 | 170 (67 %) | 21 (13) | 156 | 77 | 206 |
| `control_seed1.mp4` | RTX 4090 | 2026-09-28 22:30 | 4,098 s | 247 | 0 | 9 (0) | 0 | 238 | 3 |
| `seed2.mp4` | NVIDIA B200 | 2026-09-29 00:36 | 389 s | 258 | 163 (63 %) | 20 (15) | 148 | 90 | 187 |
| `control_seed2.mp4` | NVIDIA B200 | 2026-09-29 00:36 | 385 s | 245 | 0 | 3 (0) | 0 | 242 | 0 |

Totals over the five rounds, from each log's `summary.totals` and `launches_total`. The seed-1 control's 3 hops all
came outside the descents (seed 1, below). Read each connected row against the control row under it, which ran on the
same GPU. The wall times differ by device and by load, and say nothing about the simulation.

The schedule is the game's and the same in every clip. The swats begin at 2.00, 6.00, 10.50, 15.00 and 19.50 s, at
0.5, 1.0, 1.5, 2.0 and 3.0 m/s, and the head reaches the tray at 2.99, 6.50, 10.83, 15.25 and 19.67 s. The handle
comes from -x in rounds 1, 3 and 5 and from +y in rounds 2 and 4. A fresh crowd is laid out at 4.5, 9.0, 13.5 and
18.0 s. The final card fades in over 21.08-21.58 s and stays to the end. The aim (the game's, chosen when each swat
begins) came to within 2 cm of the tray's centre in every round of seeds 0 and 1, and within 2.3 cm in seed 2 (round
4 aimed at (-2, +1) cm). A control shares its seed's aims, crowds and schedule: its logged aim points and start
positions are the same, on both devices.

The banners the trailer can cut on come from the run log: "64 FLIES · 64 BRAINS · ONE GPU" (0-2.0 s); "ROUND k / 5 ·
v m/s · 64 FRESH FLIES" at each reset; "GIANT FIBRE n Hz -> JUMP" when the focus fly (the at-risk fly nearest the aim,
shown in the eye panel) launches during a descent; the result, "n JUMPED IN TIME · n GOT CLEAR", 0.2 s after
contact, held 1.6 s and then faded over 0.8 s. Its second line, "n under it at v m/s", is the at-risk count (the
outcome rule), while the round panel's "n flies under it" during the descent counts only the flies under the footprint
when the swat began (the log's `under_at_start`). The two can differ within one round: in seed 0 the panel reads 50,
48, 50, 48 and 47 and the banner 52, 49, 53, 49 and 50. Each reset clears the banners, so round 1's result is gone at 4.5 s,
1.3 s after it appears (and round 2's at 9.0 s, while fading). The fly's-eye mosaic freezes on CAUGHT or SPLAT when the
focus fly is hit.

### Seed 0: `out/games/swarm/seed0.mp4`

| round | speed | at risk | jumped in time | got clear (by the hop) | caught in the air | splat | median lead of the hop | median GF peak before impact, at risk | blind control: at risk, jumped, clear, splat; GF peak |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 m/s | 52 | 43 (83 %) | 23 (18) | 24 | 5 | 223 ms | 66 Hz | 52, 0, 4, 48; 14 Hz |
| 2 | 1.0 m/s | 49 | 39 (80 %) | 5 (4) | 35 | 9 | 76.5 ms | 45.8 Hz | 48, 0, 1, 47; 11.8 Hz |
| 3 | 1.5 m/s | 53 | 42 (79 %) | 0 (0) | 43 (1 on the resting handle) | 10 | 56 ms | 47.9 Hz | 50, 0, 0, 50; 10 Hz |
| 4 | 2.0 m/s | 49 | 25 (51 %) | 1 (1) | 25 (1 on the resting head) | 23 | 28.3 ms | 33.3 Hz | 48, 0, 0, 48; 7.8 Hz |
| 5 | 3.0 m/s | 50 | 13 (26 %) | 0 (0) | 13 | 37 | 5.5 ms | 26.8 Hz | 48, 0, 0, 48; 6 Hz |

* **The giant fibre fired in time for 162 of 253 at-risk flies (64 %).** That was 79-83 % from 0.5 to 1.5 m/s, 51 %
  at 2.0 and 26 % at 3.0 m/s. The median lead of the hop over contact fell from 223 ms to 5.5 ms.
* **29 got clear:** 23 by the hop, 5 walked out and 1 by a voluntary take-off. 23 of the 29 were in round 1
  (0.5 m/s). Rounds 3 and 5 had none.
* **140 were caught in the air.** They include 138 of the 162 that jumped in time. The other 2 launched late, after
  contact, from just outside the head's outline, and were caught during the head's rest: fly 6 (round 3) met the
  resting handle 23.3 mm up, and fly 3 (round 4) the head's edge 6.8 mm up. One fly that jumped in time (fly 54,
  783 ms before contact) landed back under the head and was splat (round 1).
* **84 were splat**, from 5 at 0.5 m/s to 37 at 3.0 m/s.
* **Blind control, same seed: 0 of 246 at-risk flies jumped.** 241 were splat and 5 walked clear. No fly in the
  control launched an escape hop at any time in its 24 s. The at-risk flies' median giant-fibre peak before impact was
  6-14 Hz, against 27-66 Hz with the swatter visible.
* **Heading** (the bearing to the landing point when the swat began, pooled over the five rounds). The giant fibre
  fired in time for 66 of 79 flies facing the landing point, 58 of 85 side-on and 38 of 89 facing away; 0, 11 and 18
  of them got clear. At the rim it fired for 23 of 30 facing it (0 got clear) and 7 of 39 facing away (9 got clear).
* **Distance** (head radii from the landing point). In time: 42 of 55 in the inner half, 63 of 91 in the middle, 50
  of 96 at the rim, and 7 of 11 beyond it (outside the head's outline when the swat began; of these 11, 8 were
  caught in the air, 2 were splat and 1 got clear). Got clear: 4, 9, 15 and 1.
* **Other events.** Fly 8 launched an escape hop at 2.00 s, in the last 10 ms step before swat 1 began, with DNp01 at
  33.9 Hz. It was the only hop in walking time before any swat in this run (pre-swat 99th percentile 17.3 Hz), and fly
  8 was not under the head. The blind control's scenes were the same up to that step, because the swatter is parked
  out of every scene between swats. It had no such hop (pre-swat maximum 30.6 Hz): the two runs differ only by the
  native kernels' run-to-run variation (Limits), which each fly's own walking then carries into what it sees (seed 1's
  pair already differed by one fly under the head when swat 1 began, 51 against 52). 17 flies outside the footprint hopped during a descent (startled). 6 escape hops were
  late (launched in the contact step or during the rest). 4 voluntary take-offs came during round 1's descent. The
  hops after contact, until the next reset, numbered 7, 5, 5, 4 and 3.
* **The focus fly.** In rounds 1-4 the focus flies (43, 35, 36, 35) each launched in time. The JUMP banner came at
  2.80, 6.32, 10.78 and 15.22 s (193, 176.5, 51 and 28.3 ms before impact), and each fly was caught in the air (the
  mosaic reads CAUGHT). In round 5, fly 27 did not launch (peak 26.8 Hz) and was splat, so that round has no JUMP
  banner.
* **Wall.** The run took 2,573 s for 24.0 s of brain time. The first frame took 102 s and the first brain second
  614 s. Brain seconds 12-14 took 780 and 371 s, under other jobs' load, and the rest 19-81 s each. The on-screen
  sim-step figure (the mean of the last 200 steps) read 4,968 s in the first frame, 2,897 s in the second, 1,416 s at
  0.3 s, 253 s at 2.0 s, 38 s at 3.0 s and 354 s at 14.0 s. (The draft trailer's full-frame shot of round 1,
  `out/trailer/draft1.mp4` at 52.18-55.16 s, opens on the second frame, so these figures fall from 2,897 s to 38.4 s on
  screen there. The current cut has no swarm shot; fifth pass, at the end.) In frames sampled every 2 s from 16 s to the end it read 13.7-18.3 s (17.5 s in the last frame). The control
  took 1,976 s (brain second 12-13 took 796 s). Its on-screen figure read 120 s at 0.3 s, 375 s at 14 s and 15-19 s
  from 18 s on.

### Seed 1: `out/games/swarm/seed1.mp4`

| round | speed | at risk | jumped in time | got clear (by the hop) | caught in the air | splat | median lead of the hop | median GF peak before impact, at risk | blind control: at risk, jumped, clear, splat; GF peak |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 m/s | 52 | 44 (85 %) | 16 (11) | 31 | 5 | 193 ms | 71.6 Hz | 53, 0, 6, 47; 13.1 Hz |
| 2 | 1.0 m/s | 52 | 46 (88 %) | 2 (1) | 45 | 5 | 81.5 ms | 54.7 Hz | 48, 0, 1, 47; 12.5 Hz |
| 3 | 1.5 m/s | 51 | 43 (84 %) | 2 (1) | 42 | 7 | 71 ms | 51.4 Hz | 50, 0, 1, 49; 9 Hz |
| 4 | 2.0 m/s | 49 | 28 (57 %) | 1 (0) | 28 | 20 | 28.3 ms | 35.3 Hz | 47, 0, 1, 46; 5.8 Hz |
| 5 | 3.0 m/s | 50 | 9 (18 %) | 0 (0) | 10 (1 on the resting head) | 40 | 15.5 ms | 25.3 Hz | 49, 0, 0, 49; 5.8 Hz |

* **The giant fibre fired in time for 170 of 254 at-risk flies (67 %).** That was 84-88 % from 0.5 to 1.5 m/s, 57 %
  at 2.0 and 18 % at 3.0 m/s. The median lead fell from 193 ms to 15.5 ms.
* **21 got clear:** 13 by the hop and 8 walked out. 16 of them were in round 1.
* **156 were caught in the air.** They include 155 of the 170 that jumped in time. The other one, fly 47 in round 5,
  launched late, while the head rested on the tray, and was caught on it 6.8 mm up. 2 flies that jumped in time in
  round 1 (flies 9 and 16, 293 ms before contact) landed back under the head and were splat.
* **77 were splat**, from 5 at 0.5 and 1.0 m/s to 40 at 3.0 m/s.
* **Blind control, same seed: 0 of 247 at-risk flies jumped.** 238 were splat and 9 walked clear. The control logged 3
  escape hops, none in a descent. Two are the same pre-swat hops as the connected run (below). The third, fly 56 at
  8.79 s, came in round 2's after-contact window with no swatter in its scene. The median giant-fibre peak before impact
  was 5.8-13.1 Hz, against 25-72 Hz with the swatter visible.
* **Heading.** In time: 77 of 92 facing the landing point, 60 of 78 side-on, 33 of 84 facing away; 1, 7 and 13 got
  clear. At the rim: 34 of 39 facing it (1 got clear) and 6 of 36 facing away (11 got clear).
* **Distance.** In time: 43 of 57 in the inner half, 66 of 92 in the middle, 47 of 89 at the rim and 14 of 16 beyond
  it. Got clear: 0, 4, 15 and 2.
* **Other events.** Two flies hopped in walking time before a swat: fly 29 at 5.54 s (33.3 Hz, round 2) and fly 28 at
  14.49 s (35.2 Hz, round 4). This run's pre-swat 99th percentile was 18.2 Hz and its maximum 36.5 Hz. The blind
  control logged the same two hops, by the same flies at the same times, so they are not a response to the swatter.
  15 flies were startled, 6 hops were late and there were no voluntary take-offs. The hops after contact numbered 2,
  2, 4, 2 and 9.
* **The focus fly.** In rounds 1, 2, 3 and 5 the focus flies (28, 28, 36, 35) launched in time. The JUMP banner came
  at 2.83, 6.39, 10.74 and 19.66 s (163, 106.5, 91 and 5.5 ms before impact; the round-5 banner reads "37 Hz ... 6 ms
  before impact"), and all four were caught in the air, the last one (fly 35) 7.1 mm above the tray. In round 4, fly
  27 did not launch and was splat.
* **Wall.** The run took 4,589 s (the control 4,098 s), because the other environments' seeds were recording at the
  same time. The first four brain seconds took 24-30 s each and the rest 40-508 s. In frames sampled every 2 s the
  on-screen figure read 18-21 s up to 4 s and 44-259 s from 6 s on (187 s in the last frame). The control's read
  17-21 s up to 6 s and 40-314 s after.

### Seed 2: `out/games/swarm/seed2.mp4` (NVIDIA B200)

Recorded on the cluster, with its control beside it on the same GPU (above). Read it against that control, not against
seeds 0 and 1, which ran on another GPU. The first, stopped attempt at this seed on the desktop reached round 3, and
its recovered frames are reported under Clips; they differ from this run by up to 2 flies jumped in time and 7 got
clear per round.

| round | speed | at risk | jumped in time | got clear (by the hop) | caught in the air | splat | median lead of the hop | median GF peak before impact, at risk | blind control: at risk, jumped, clear, splat; GF peak |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 m/s | 53 | 46 (87 %) | 12 (8) | 35 | 6 | 158 ms | 64.3 Hz | 50, 0, 2, 48; 15.4 Hz |
| 2 | 1.0 m/s | 51 | 39 (76 %) | 6 (5) | 34 | 11 | 76.5 ms | 49.6 Hz | 49, 0, 1, 48; 10.8 Hz |
| 3 | 1.5 m/s | 54 | 41 (76 %) | 2 (2) | 39 | 13 | 61 ms | 44.3 Hz | 50, 0, 0, 50; 9.5 Hz |
| 4 | 2.0 m/s | 50 | 22 (44 %) | 0 (0) | 23 (1 on the resting head) | 27 | 48.3 ms | 31.1 Hz | 48, 0, 0, 48; 7.9 Hz |
| 5 | 3.0 m/s | 50 | 15 (30 %) | 0 (0) | 17 (2 on the resting head) | 33 | 15.5 ms | 24.1 Hz | 48, 0, 0, 48; 7.7 Hz |

* **The giant fibre fired in time for 163 of 258 at-risk flies (63 %).** That was 76-87 % from 0.5 to 1.5 m/s, 44 %
  at 2.0 and 30 % at 3.0 m/s. The median lead of the hop over contact fell from 158 ms to 15.5 ms.
* **20 got clear:** 15 by the hop, 4 walked out and 1 by a voluntary take-off (fly 18, at 2.80 s, during round 1's
  descent). 12 of the 20 were in round 1 (0.5 m/s). Rounds 4 and 5 had none.
* **148 were caught in the air.** They include 145 of the 163 that jumped in time, all during the descent: 144 met
  the head, and fly 2 (round 1, facing away), which hopped out towards the handle's side, met the handle 19.2 mm up in
  the contact step.
  The other 3 launched late, in the contact step or during the rest (fly 0 in round 4, flies 55 and 8 in round 5), and
  were caught on the resting head 6.8-7.1 mm up. 3 flies that jumped in time in round 1 (flies 15, 51 and 63, 283-673 ms
  before contact) landed back under the head and were splat.
* **90 were splat**, from 6 at 0.5 m/s to 33 at 3.0 m/s.
* **Blind control, same seed, same GPU: 0 of 245 at-risk flies jumped.** 242 were splat and 3 walked clear. No fly in
  the control launched an escape hop at any time in its 24 s, and none took off. The at-risk flies' median giant-fibre
  peak before impact was 7.7-15.4 Hz, against 24-64 Hz with the swatter visible.
* **Heading.** In time: 64 of 78 facing the landing point, 60 of 85 side-on, 39 of 95 facing away; 2, 7 and 11 got
  clear. At the rim: 34 of 42 facing it (1 got clear) and 12 of 37 facing away (7 got clear).
* **Distance.** In time: 40 of 54 in the inner half, 51 of 88 in the middle, 57 of 96 at the rim and 15 of 20 beyond
  it. Got clear: 1, 4, 12 and 3.
* **Other events.** No fly hopped in walking time before a swat, in this run or in its control. The pre-swat 99th
  percentile was 17.6 Hz (control 17.5 Hz) and the maximum 31.8 Hz in both. 11 flies were startled, 6 hops were late
  and there was 1 voluntary take-off (fly 18, above). The hops after contact, until the next reset, numbered 2, 2, 1,
  2 and 6.
* **The focus fly.** In rounds 1-3 the focus fly (fly 27 in each round's fresh crowd) launched in time. The JUMP
  banner came at 2.88, 6.31 and 10.75 s (113, 186.5 and 81 ms before impact; the banners read "35 Hz ... 113 ms",
  "34 Hz ... 187 ms" and "33 Hz ... 81 ms"), and each time fly 27 was caught in the air. In round 4, fly 20 (peak
  24.9 Hz) and in round 5, fly 36 (peak 27.4 Hz) did not launch and were splat, so those rounds have no JUMP banner.
* **Wall.** The run took 389 s for 24.0 s of brain time (the control 385 s), after 12.1 s to build the batch. The
  first frame took 0.8 s. Brain seconds 0-6 took 9.9-10.6 s of wall each, and 6-24 took 17.3-21.2 s each. Five pong jobs
  shared the pair's GPU from 00:36:56 UTC (the job manager's records, Clips), and from brain second 6 a brain second's
  wall rose from about 10 s to 17-21 s. The on-screen sim-step
  figure (frames every 0.5 s) read 34.9 s in the first frame, 8.5 s at 0.3 s and 6.3-7.0 s from 1 s to the end of
  round 1. In round 2 it read 6.4-6.6 s up to 6 s, then rose to 11.8 s at 8 s as the slower brain seconds entered its
  200-step mean. In round 5 it read 11.2-11.6 s, and 10.9 s in the last frame (the control's last frame: 11.2 s). These
  are B200 wall times, not comparable with the desktop's.

## Corrections (caption skeptic)

I checked the caption against the six recorded run logs (`out/games/swarm/{seed,control_seed}{0,1,2}.json`), `dev.json`,
`measure_seed101{,_blind}.json`, the superseded logs, the lane logs, the render and console logs, `seed2_cluster.log`,
the builder's kept arrays (`builder_scratch/proto1_64_*.npz`, `chain_*_101.npz`), the honesty review's replay
(`review_honesty/rerun_101_none.*`; I re-ran `collide.py` on CPU and got 117 + 27 = 144 of 161), `games/swarm.py`
(sha256 `3a653c1d9fc0d4bf`, the logged hash), `flyverse/body.py`, `flyverse/air.py`, `flyverse/world.py` and
`scripts/cluster_run.py`. I also looked at frames that ffmpeg pulled from all six clips and the dev clip. Every
per-round table cell, total, heading and distance bin, focus-fly banner, schedule time, result banner, final card and
provenance field matched, except for the changes below. So did every swarm window the trailer uses (seed 0 at 0.02-4.79,
6.0-11.19 and 15.1-15.4 s, and seed 1 at 0.3-1.95 s). What changed:

1. Design. Was: "Every landing drives the giant fibre again (the ground looms as the fly falls back), so many flies hop
   again every 1.5-2 s". "Every landing" and the looming mechanism were not measured. Now the kept chain arrays give the
   numbers: after the swatter had gone, 32 of 32 landings were followed within 0.5 s by DNp01 above 40 Hz (up to
   114 Hz), with LPLC2 and LC4 above their pre-hop rates. The mechanism is marked as not isolated. The re-hop interval
   is now 1.3-2.4 s (median 1.4 s) from the same arrays. Limits' "(the landing loom)" now reads "(driven by the
   landings; Design)".
2. Measured first. Was: "In 21 traced landings the giant fibre passed 40 Hz within 0.5 s in 9 (up to 85 Hz)". No
   kept log holds that sample. It is replaced by the chain arrays' counts (32 of 32 with the game's light, 14 of 18
   with the shipped light, up to 85 Hz). The same arrays confirm "26 from 17" and give 41 take-offs from 27 flies where
   the text says "40 hops from 26" (their count includes any voluntary take-off).
3. Measured first. The following have no log or array anywhere under `out/games/swarm/` and are now marked "no kept
   log": the memory and quiet-GPU timings, the four fixed-spot swats (38, 10, 0, 0), the radiance figures, the seed-100
   light comparison (40/48 vs 38/46, 26.1 vs 33.6 Hz; the kept seed-101 arrays give 25.0 vs 33.7 Hz) and the seed-100
   no-reset run (111 hops, 75 vs 17 Hz). Re-checked and now cited: the 0.8 / 1.6 m/s prototype (42 of 64; centre leads
   70-110 and 20-40 ms; walking maxima median 15.0, 99th percentile 24.5 Hz; 22.8 Hz in the first 0.5 s). The recorded
   logs' build times (11.3-14.9 s shared RTX 4090, 12.0-12.1 s B200) now stand beside the "7-9 s".
4. Measured first. "which differs from the frozen file in drawing only" is now credited to the builder, because the
   `ea8f41f1` source was not kept. The two seed-100 renders (one from each version) agree to within 2 flies per round.
5. Run to run. The first seed-101 set (45/52, 41/48, 39/52, 25/47, 12/49) is in no kept log. It is now marked as the
   builder's record, and the logged seed-100 pair is named as the kept comparison (also in Limits).
6. Seed-101 control. Was: "Every hop in time in the main run was a response to the swatter", which claims too much
   for each individual hop. Now: no at-risk fly hopped in the control, so the main run's hops in time depend on the
   swatter being in the flies' scenes.
7. Resting catches. `caught_on_resting_head` counts every catch during the head's 0.2 s rest, including the handle and
   late hops into the head's edge. Was, on seed 101: "2 coming down on the resting head". Fly 7 met the resting handle
   23.2 mm up, beyond the head's rim, and only fly 48 came down onto the head (9.5 mm). Was, on seed 0: "2 of the 140
   came down on the resting head (rounds 3 and 4)". Both launched late, after contact, from outside the head's
   outline. Fly 6 met the handle 23.3 mm up (the head's top is 12 mm; table cell now "on the resting handle"), and fly
   3 met the head's edge 6.8 mm up. The outcome row, the seed-101 bullet, the seed-0 bullet and Limits now say
   "the resting swatter". "141 by the descending head" now reads "141 during the descent".
8. Seed 2. Was: "145 of the 163 that jumped in time, all caught by the descending head". Fly 2 (round 1) met the
   handle 19.2 mm up in the contact step. Now: 144 met the head, 1 the handle.
9. Seed 0 distance. Was: "7 of 11 beyond it (at risk at contact only)". Of those 11, 8 were at risk because the head
   met their hop in the air, 2 were splat and 1 got clear. Now says that.
10. Seed 1 wall. Was: "188 s in the last frame". The last frame (1,199) reads 187.2 s, and the log's
    `wall_s_per_brain_s_sim` is 187.19. Now 187 s.
11. Dev clip. Was: the on-screen rate "between about 30 and 100 s". Frames every 2 s read 11.8-100.0 s. Now "about
    12 and 100 s".
12. Clips. Was: "It swung with the GPU's load and with the device, not with the simulation. Recording is offline, so
    neither changed anything but the wall time." A different GPU and torch build can change the numerics. Now the load
    is said to change only the wall time, the device may also change the numerics, and each seed is read against its
    own control. "Every comparison ... within one device" now reads "one device type". "(GPU, torch and CUDA build)"
    now reads "(GPU, torch 2.11 vs 2.10, Python 3.12 vs 3.13; both torch builds are cu128)": both logs say cu128.
13. Clips. "ran at the same time on the same GPU", "The seed-2 GPU was shared too" and "five pong jobs were running on
    the same GPU" are not in any run log, which records only the GPU model. Each is now attributed to the cluster status
    check made during the runs. Pong's cluster logs, which started at 00:37 UTC, are cited as the overlap that the logs
    do show.
14. Design table. Walking. Was: "speed from the forward DNs and leg MNs (0.8-3 cm/s)". `Locomotion.readout` adds a
    constant 0.8 cm/s `baseline_speed`, which dominates (games/README.md, finding 4). Now says so. The crowd's numpy
    seed is `[seed, r - 1]` for round r, not `[seed, round]`.
15. Main view. The code never draws a sprite below 12 px, so flies are larger than 2.5x in the wide framing. Caught
    flies are drawn with the same red splat mark as splat ones, so the marks on the tray count hits. Both are now
    declared.
16. The measure command now carries `--seconds 22.5`, as logged. With the default 24 s, round 5's after-contact
    window and its hop count would differ.

Not changed, but checked. The desktop attempt at seed 2 left no playable video (ffprobe: "moov atom not found") and
no run log. The seed-2 re-recording on the B200 is reported as such above. The trailer's swarm shots all come from
seeds 0 and 1 (RTX 4090). Its "64 brains × 167,106 neurons, one GPU" label is accurate for those runs. Its cropped
main-view shots (seed 0 at 2.85, 6.3 and 15.1 s) cut off the eye panel and the JUMP banner (except in the 2.85 s
shot's last two frames; second pass), and show the red rings and
splat marks. Per item 15, those marks are hits, caught and splat together.

### Corrections (caption skeptic, second pass: the trailer's swarm and the seed-2 material)

A second skeptic checked the swarm moments the trailer uses and the seed-2 / cluster material against the six run
logs, the seed-2 consoles and cluster logs, pong's cluster logs, `out/games/swarm/killed_partial/`, the lane logs,
`games/trailer_shots.py`, `games/trailer_edl.json`, `games/trailer.py` and frames pulled locally with ffmpeg. The
trailer's swarm windows matched the logs (seed 0 at 0.02-2.99 s, 2.85-3.22 s, 2.87-4.05 s, 2.99-4.79 s,
6.0-11.19 s, 6.3-6.7 s, 10.4-11.18 s and 15.1-15.4 s, and seed 1 at 0.3-1.95 s). The list above stands. What changed:

A. Clips, seed 2's first attempt. Was: "Neither video can be opened (ffprobe: "moov atom not found"; the encoder was
   never closed), and their console logs print progress only. So they left no outcome to report, and the restart
   could not have been a rerun for a better one." The files do not open, but their H.264 streams do decode with the
   stream headers of `seed0.mp4` (same x264 settings string, byte for byte): 12.46 s of seed 2 and 10.64 s of its
   control, with the result banners of rounds 1-3 and 1-2. Those outcomes are now reported in a table, beside the B200
   recording's, with the size of the difference (jumped in time within 2 per round, got clear up to 7). "Could not have
   been a rerun for a better one" now reads: stopped for the move off the desktop, not on an outcome (the lane logs'
   00:28Z stop and the PLAN's move of GPU work), and the recording did no better on the fly's measures. The first
   pass's "left no playable video" holds for the files as files.
B. Limits, run to run. Was: "a re-run of a seed gives similar but not identical numbers (above: up to 1 fly per round
   on seed 101 ... and up to 2 on the two logged seed-100 renders)". Those figures are for jumped in time only. The
   seed-100 renders' got-clear counts differ by up to 3, and the seed-2 desktop / B200 pair's by up to 7 (caught in
   the air by up to 5). Now says so, and that got clear is the least stable for its size. The seed-2 section now
   points to the stopped attempt.
C. Clips. Was: "pong's cluster runs, `out/games/pong/b200_seed*.json`, started at 00:37 UTC". Pong's cluster call ran
   six jobs, `b200_seed{0,1,2}` and `swapped_seed{0,1,2}`, all started 00:37:02-00:37:05 UTC and ran 396-480 s. Now
   names all six, quotes their call's note of 2 jobs already pending, queued or running (it does not name them), and
   adds what the swarm consoles show: about 10 s of wall per brain second up to brain second 6, then 17-21 s.
D. Seed 2, Wall. Was: "five pong jobs were running on the same GPU as the pair". Kept as the status check's figure,
   now beside the logs' six running pong jobs (no log names a GPU).
E. Seed 2, Wall. Was: "The on-screen sim-step figure read 8.5 s at 0.3 s, 6.3-6.4 s in round 1, 8.1-9.8 s in round 2,
   11.2-11.4 s in round 5". Frames every 0.5 s read 34.9 s in the first frame, 7.0 s at 1.0 s, 6.3-6.5 s from 2 s to
   the end of round 1; 6.4-6.6 s in round 2 up to 6 s, then 7.9, 9.6, 10.6 and 11.8 s at 6.4, 7, 7.5 and 8 s (it
   passes through 8.1-9.8 s only once, on the way up); and 11.2-11.6 s in round 5. Now says so.
F. Clips, banners. Was: the result banner comes "0.2 s after contact, held 1.6 s". `games/swarm.py` clears every
   banner at a reset, so round 1's result (from 3.19 s) is gone at 4.5 s, after 1.3 s. Now says so. The trailer's
   2.99-4.79 s shot of seed 0 therefore ends on the "ROUND 2 / 5 · 1.0 m/s · 64 FRESH FLIES" banner and a fresh
   crowd for its last 0.29 s, not on the result.
G. Seed 0, Wall. Added the first two frames' on-screen figures (4,968 s and 2,897 s), because the trailer's full-frame
   round-1 shot opens on the second frame and shows the figure falling from about 2,900 s.

Not changed, but checked against frames. Seed 1 at 0.3-1.95 s: the "64 FLIES · 64 BRAINS · ONE GPU" banner fading
out, "RTX 4090, shared" on the HUD, and the trailer's "64 brains × 167,106 neurons, one GPU" is true of that run.
Seed 0 round 1: "GIANT FIBRE 33 Hz -> JUMP · fly 43 · 193 ms before impact" from 2.80 s, the head on the tray at
2.99 s with fly 43's mosaic reading CAUGHT, then "43 JUMPED IN TIME · 23 GOT CLEAR · 52 under it at 0.5 m/s · 24 caught
in the air · 5 splat on the tray" from 3.19 s, as logged. Round 2: fly 35's banner (33 Hz, 177 ms) and "39 JUMPED IN
TIME · 5 GOT CLEAR · 49 under it · 35 caught · 9 splat". The reset banner "ROUND 3 / 5 · 1.5 m/s · 64 FRESH FLIES" at
9.0 s. Round 3: fly 36's banner (37 Hz, 51 ms) and "42 JUMPED IN TIME · 0 GOT CLEAR · 53 under it · 43 caught · 10
splat". Round 4: fly 35's banner (39 Hz, 28 ms), the head down at 15.25 s. In the cropped 2.85-3.22 s shot the JUMP
banner is cut off except in its last two frames, where the result banner pushes it up into the crop. Seed 2's three
JUMP banners and its round-5 result ("15 JUMPED IN TIME · 0 GOT CLEAR · 50 under it · 17 caught · 33 splat") match
its log. The trailer's line "every number on screen: CONNECTOME · DECODER · GAME" (laid over a Doom shot at about
62-63 s, in the draft and in the current cut; not the end card) is broader than these frames: the sim-step
figure, the "64 BRAINS × 167,106" panel and the JUMP and result banners carry no chip of their own (the round panel,
the outcomes panel and the traces do).

### Corrections (caption skeptic, third pass)

A third check, of the sentences the two passes above left standing, against the six run logs' per-fly `result` rows
and `launch` events, `measure_seed101{,_blind}.json`, `builder_scratch/chain_{globe,shipped}_101.npz`,
`flyverse/body.py` (`Flight.launch`) and `flyverse/batch_sim.py` (`escape_gating=False`, so the threshold stays at
33 Hz), and frames pulled locally with ffmpeg from `seed0.mp4`, `seed1.mp4` and `seed2.mp4` in every trailer window.
What changed:

1. Headline. Was: "The shipped escape hop goes straight ahead, so 438 of the 495 flies that jumped in time were caught
   in mid-air." The "so" credits the catches to the hop's direction. The per-fly rows say that direction explains only
   part of it. Of the flies that jumped in time, 198 of 207 facing the head's landing point were caught, but so were
   155 of 178 side-on and 85 of 110 facing away (seed 0: 65/66, 48/58, 25/38; seed 1: 74/77, 53/60, 28/33; seed 2:
   59/64, 54/60, 32/39). Now: "The shipped escape hop is not aimed: it goes forward along the fly's heading. 438 of the
   495 flies that jumped in time were caught in mid-air, among them 85 of the 110 that faced away from where the head
   came down."
2. "not claimed" row. Was: "the hop goes where the fly faces, which is why most flies that jumped in time were caught
   in the air". Same reason. Now it gives the three bearing bins above and points to the hop's short reach (What was
   measured first) without claiming a cause.
3. Measured first. Was: "A freshly built brain peaked at 22.8 Hz in its first 0.5 s, so a fresh crowd produces no
   false starts." The recordings contradict the conclusion. Seed 0's fresh crowd had fly 8 hop at 2.00 s (33.9 Hz),
   and seed 1's had flies 29 and 28 hop 1.04 and 0.99 s after a reset. What holds is narrower: no run (the six clips,
   seed 101's pair) had a hop in the first 0.5 s of a round. Now says that and lists the false starts.
4. Design. Was: "1.3-2.4 s apart (median 1.4 s)". In `chain_globe_101.npz` those are the 10th-90th percentiles of the 14
   intervals between the surviving flies' hops after contact. The longest is 3.45 s. Now labelled.
5. Clips. Added: the seed-2 clips' HUD is set in the DejaVu fallback fonts (the cluster has neither Cascadia Mono nor
   Bahnschrift from `games/common.py`'s lists), so those clips look different. The numbers do not change.
6. First pass, closing note. Was: "cut off the JUMP banner and the eye panel". The second pass found the exception (the
   2.85 s shot's last two frames). The note now carries it, so the two passes no longer disagree.

Trailer moments checked against frames (clip seconds; each frame shows brain time 0.02 s later). Everything below
matched the logs.
* `seed1.mp4` 0.30-1.95: the "64 FLIES · 64 BRAINS · ONE GPU" banner and "RTX 4090, shared". The trailer's label is
  true of this run.
* `seed0.mp4` 0.02-2.99, the round-1 descent. One amber ring appears as the head does, at 2.00 s. It is fly 8's false
  start, launched in the step before the swat began, while the head was still parked out of every scene. The shot does
  not show that this hop is not a response to the swatter. Then "GIANT FIBRE 33 Hz -> JUMP · fly 43 · 193 ms before
  impact" from 2.80 s.
* `seed0.mp4` 2.99-4.79 and 2.87-4.05: the impact, fly 43's mosaic reading CAUGHT, and the result "43 JUMPED IN TIME ·
  23 GOT CLEAR" (52 under it, 24 caught in the air, 5 splat). It is accurate, and it is seed 0's most favourable round:
  23 of the run's 29 clears came in it. The grid shot and the 10.40-11.18 shot also show round 2 (5 got clear) and
  round 3 (0 got clear).
* `seed0.mp4` 2.85-3.22, 6.30-6.70 and 15.10-15.40, cropped: the head landing at 0.5, 1.0 and 2.0 m/s. The 6.0-11.19
  grid cell and the 10.40-11.18 shot: round 2's and round 3's JUMP banners (33 Hz, fly 35, 177 ms; 37 Hz, fly 36,
  51 ms), their results and the round-3 reset banner.
The trailer uses no seed-2 or control footage. The controls' result (0 of 738 jumped) is in the caption only.

### Corrections (caption skeptic, fourth pass: the swarm in the trailer)

A fourth check tested the sentences the three passes above left standing. It used the six run logs' per-fly `result`
rows, `measure_seed101{,_blind}.json`, `dev.json` against `superseded/dev_ea8f41f1.json`, the kept arrays
(`builder_scratch/chain_*_101.npz`, `proto1_64_*.npz`, `proto2_101.npz`, `review_honesty/rerun_101_none.npz`; I re-ran
`collide.py` on CPU and got 117 + 27 = 144 of 161), `builder_scratch/chain_check.py` and `games/swarm.py`
(`3a653c1d9fc0d4bf`, the logged hash). It also used the cluster job manager's records for the seed-2 jobs and the
cluster checkout's reflog, both read-only, and frames pulled locally with ffmpeg from every swarm window the trailer
uses. What changed:

1. Headline. Was: "for about two thirds of the flies under the head ... and for none of the 738 under it". *At risk*
   also counts a fly whose hop the head met in the air (the outcome rule). In seeds 0, 1 and 2, 8, 10 and 11 of the
   at-risk flies were outside the head's outline when the swat began (rho > 1.02) and were caught in the air, and 6, 9
   and 8 of them count as jumped in time. Now: "the flies in the head's path" and "the 738 in its path".
2. What was measured first, opening paragraph. Added that two kept sources come from earlier versions of
   `games/swarm.py`, not from the frozen file. The first is the chain arrays: `chain_check.py` sets `swarm.SCHEDULE`,
   which the frozen file lacks, and the arrays' take-offs cluster at 1.5-2.0 s (one swat at 1.0 s, contact 1.99 s).
   Only 5 and 7 flies are out, the splats. So their "41 take-offs from 27 surviving flies" include flies that the
   frozen rule would score as caught. The second is the honesty review's replay, which scored "escaped" with no
   mid-air test. Both ran the shipped brain and body.
3. Same paragraph. Was: the two seed-100 renders "agree to within 2 flies per round". That holds for jumped in time
   only. Round 1's got clear is 14 against 11. Now: "to within 2 flies per round in jumped in time, and to within 3 in
   every count". The second pass fixed the same figure in Limits.
4. The hop. Was: "Over 359 hops" and "10th-90th percentiles of 359 hops". 357 of the 359 landed within the run
   (`rerun_101_none.npz`: apex 24.1 / 37.8 / 54.8 mm at the 10th / 50th / 90th percentile, maximum 88.5; air time
   0.25 / 0.36 / 0.47 s). Now "the 357 of 359 hops that landed within the run", in the bullet and in the table row,
   which now also names its source.
5. Seed-101 heading. The prototype's "rim flies facing the swatter launched 9 of 9, facing away 0 of 9" now says "no
   kept log". `proto2.py` printed those counts, and `proto2_101.npz` keeps only the GF, LC4 and LPLC2 rates.
6. Seed-101 distance. Was: three bins, "inner half ... 40 of 54, middle 61 of 95, rim 54 of 91". The log's fourth bin
   was missing. Now adds "beyond it 9 of 9", with 0 got clear.
7. Seed 0, other events. Was: "The blind control's inputs were identical up to that step ... It had no such hop ...,
   because the native kernels are not bit-reproducible". The scenes were identical, but the inputs were not. What a
   fly sees depends on where its own walking took it, and that carries the kernels' run-to-run variation. Seed 1's
   pair already differed by one fly under the head (51 against 52) when swat 1 began. Now says so.
8. Seed 1, control. Was: "with nothing visible to it". The fly saw the room. Now: "with no swatter in its scene".
9. Clips and seed-2 Wall. Was: "on one GPU by a cluster status check made during the runs", "(the same status check)",
   and "At a cluster status check near 14 s of brain time (not in the run logs), five pong jobs were running on the
   same GPU as the pair". The job manager's records confirm them. Jobs `e0888766c5e0` and `e93fa46402e1` both ran on
   `<cluster-node>`'s GPU 3, 00:35:54-00:42:47 UTC. `pong-swap-5fab86-0` to `-4` ran on GPU 3 from 00:36:56 to 00:45:54, and
   `-5` ran on GPU 0. Now cites them.
10. Clips, commit. Was: "a fresh copy of the repository at `origin/main` (which is `959f2e9`)". `cluster_run.py` lays
    the local diff over a fresh copy of the target's own checkout. Its docstring warns that a stale checkout runs
    stale files. The house checkout's `git rev-parse HEAD` and `git reflog` show `959f2e9` since 2026-09-22 17:49
    -0400, with no tracked changes. Now says so.
11. Clips, desktop. Was: the GPU was shared with "the other environments' final recordings". Their dev runs shared it
    too (spacecraft `dev` / `dev_control` 21:46-22:28, minecraft `dev_final_control` 21:46-22:00, hairdryer `dev`
    22:28-22:50, swat `dev_fix` 22:33-22:59 UTC, from their run logs). Added.

Not changed, but checked. The third pass's headline and "not claimed" numbers (198 of 207, 155 of 178, 85 of 110 over
the three seeds) match the per-fly rows. Its false-start list matches the `launch` events. The chain arrays hold its
re-hop percentiles (10th, 50th and 90th: 1.30, 1.41 and 2.43 s; 14 intervals). Flies in the chain test that never hopped stayed at or
below 26.4 Hz after the swatter had gone, which fits "landings drive the giant fibre" without isolating it. At exact
frames, the seed-2 sim-step figure reads 8.5 s at frame 14 and 8.4 s at frame 15 ("at 0.3 s" depends on the 0.02 s
frame offset). Every trailer window matched its run log. Details are in `verified_moments` of this pass's report. The
clips themselves were not changed. Two things in the frozen clips remain for the owner. The final card juxtaposes "the
shipped hop goes forward" with "138 of the 162 that jumped in time were caught in the air", the same implied cause as
the old headline. games/README.md says the clips were made on an RTX 4090, but swarm's seed 2 and its control were made
on a B200.

### Corrections (caption skeptic, fifth pass: the swarm in the trailer)

A fifth check followed the swarm into the trailer itself. It read `games/trailer_shots.py` and `games/trailer_edl.json`
(both rewritten at 18:14 PDT on 2026-09-28), `games/trailer.py`, `games/trailer_cut.py` and `out/trailer/thread.py`.
It decoded every frame of `out/trailer/draft1.mp4` (16:44 PDT, the latest render) and matched each one against all
1,200 frames of `seed0.mp4`, `seed1.mp4`, `seed2.mp4` and the two desktop controls, full frame and cropped. It then
read the matched frames at full size against the run logs. What changed:

1. **The current cut has no swarm shot.** Neither `trailer_shots.py` nor `trailer_edl.json` names any
   `out/games/swarm/` clip. The slots the swarm held in the earlier cut now hold TRON, Mars and hairdryer: the
   timpani stroke at 27.0 s and the grid cell (TRON), 44.96 s (Mars), the call-3 stretch from 50.54 to 56.96 s (TRON,
   then Mars from 55.15 s), 67.2 s (Mars) and 70.28 s (TRON) in the build, and 77.38-78.56 s in the climax (hairdryer
   to 77.8 s, then Mars). The trailer to be published (`out/trailer/flyverse_trailer.mp4`, the EDL's `out`) has
   not been rendered yet. The X thread has no swarm post either. Every "trailer" window in the four passes above
   belongs to the earlier cut, and `draft1.mp4` is the only render that holds it. No earlier pass says so, although
   this file's last save before this pass, the fourth pass's, was at 18:20 PDT, after the re-cut. The seed-0 Wall sentence was
   "The trailer's full-frame shot of round 1 opens on the second frame, so these figures fall from about 2,900 s to
   38 s on screen there." Now: "The draft trailer's full-frame shot of round 1, `out/trailer/draft1.mp4` at
   52.18-55.16 s, opens on the second frame, so these figures fall from 2,897 s to 38.4 s on screen there. The current
   cut has no swarm shot" (the draft's first and last frames of that shot read 2,897.4 s and 38.4 s).
2. **In the draft the seed-1 shot is black.** From 50.54 s to 52.18 s, `draft1.mp4` shows only the "SWARM · 64 brains
   × 167,106 neurons, one GPU" label over black (mean luminance 0-1 of 255). The source window, `seed1.mp4` 0.30-1.95 s,
   is correct in the clip: the second and third passes checked it there. But the clip was still being recorded when
   the draft was rendered. The run started at 22:31:27Z and took 4,589 s of wall. `seed1.mp4` was finalised at
   23:48:09Z and `draft1.mp4` was written at 23:44:04Z. `trailer.py` draws a source it cannot read as dark. So the only
   rendered trailer never showed that window.
3. Clips, banners. The caption did not say that "under it" means two different counts on screen. Added: "Its second
   line, "n under it at v m/s", is the at-risk count (the outcome rule), while the round panel's "n flies under it"
   during the descent counts only the flies under the footprint when the swat began (the log's `under_at_start`). The
   two can differ within one round: in seed 0 the panel reads 50, 48, 50, 48 and 47 and the banner 52, 49, 53, 49 and
   50." Both are on screen in the draft's swarm shots. Round 1 reads "50 flies under it"
   at 55.06 s and "52 under it at 0.5 m/s" at 55.40 s. Round 3 reads 50 at 67.30 s and 53 at 67.96 s, within one
   0.78 s shot.

Not changed, but checked. The headline's counts (162 of 253, 170 of 254, 163 of 258, 0 of 738, 438 of 495) and every
per-round cell of the three seed tables and their control columns match `summary.rounds` in the six logs. The pooled
counts span both GPU types (seeds 0 and 1 on the RTX 4090, seed 2 on the B200), but the same way on both sides of the
headline's comparison: each seed's connected run and its blind control ran on the same GPU type.
Apart from item 2, every swarm frame in `draft1.mp4` is a frame of `seed0.mp4`, at 1x and in order. These are the
windows (draft time → clip time):

* 27.00-27.36 → 2.86-3.22, cropped.
* 37.46-42.65 → 6.0-11.19, the grid cell.
* 44.96-45.36 → 6.30-6.70, cropped.
* 52.18-55.16 → 0.02-3.00.
* 55.16-56.96 → 3.00-4.80. A white flash covers its first 5 frames, the head's first 0.1 s on the tray.
* 67.20-67.98 → 10.40-11.18.
* 70.28-70.58 → 15.10-15.40, cropped.
* 77.38-78.56 → 2.88-4.06.

Every number they show matches the log: the focus flies' JUMP banners (fly 43 at 33 Hz, 193 ms; fly 35 at 33 Hz,
177 ms; fly 36 at 37 Hz, 51 ms), the three result banners, the cumulative outcomes panel (82 / 28 / 73 of 101 after
round 2, and 124 / 28 / 126 of 154 after round 3), "35 of 64 alive" after round 1, and the ROUND 2 reset banner at
56.94 s. No draft shot uses seed 2 or a control.

### Corrections (caption skeptic, sixth pass: the swarm in the trailer, re-checked)

A sixth check went over the same ground independently. It ran its own frame match of `draft1.mp4` (25-80 s,
downscaled grey, against every frame of `seed0.mp4` and `seed1.mp4`, full frame, the 1344 x 756 crop and the grid
cell's crop). It read the draft's frames at full size at every logged JUMP, contact, result and reset in those
windows. It also read the draft's end card, the shots that follow its call-3 and climax swarm slots, the EDL as rewritten again
at 19:06 PDT, and `out/trailer/thread.py`. What changed:

1. Fifth pass, item 1. Was: "The slots the swarm held in the earlier cut are now swat, TRON and Mars: the timpani
   stroke at 27.0 s, the grid cell, 44.95 s, the call-3 stretch from 50.5 to 57.9 s, 67.2 and 70.28 s in the build,
   and 77.8 s in the climax." No former swarm slot holds swat. The draft's grid had swat in the next cell, and it
   still does. The draft's swarm call-3 stretch ends at 56.96 s: from 56.96 to 57.86 s the draft shows hairdryer
   (CONTACT, then FEEDING: MN9 5.7 Hz). Its climax swarm shot is 77.38-78.56 s. Now: TRON, Mars and hairdryer, with
   each slot's exact span and what fills it in `trailer_edl.json`.
2. Second pass, closing note. Was: "The trailer's end card ("every number on screen is labelled") is broader than
   these frames". No end card says that. The draft's end card reads "every clip: seeds 0-2, recorded once, captioned
   in games/ · nothing tuned into the connectome". The current EDL's reads "every clip is seed 0, 1 or 2 · every game
   control is declared · the captions are in games/". The words on screen are "every number on screen: CONNECTOME ·
   DECODER · GAME", a line laid over a Doom seed-1 shot at about 62-63 s in both cuts. Now quotes that line and says
   where it sits. The point stands: the swarm's "64 BRAINS × 167,106" panel, its sim-step figure and its JUMP and
   result banners carry no chip.

Not changed, but checked. The fifth pass's eight windows, and its black 50.54-52.18 s seed-1 window, are what this
pass found, frame for frame (mean squared error 1-3, in grey levels squared, against 35-500 for the same moment of
`seed1.mp4`, the next-best candidate). Its
timings hold: `seed1.json` started 22:31:27Z with 4,589.2 s of wall, `seed1.mp4` was closed at 23:48:09Z and
`draft1.mp4` at 23:44:04Z. The EDL was rewritten again at 19:06:16 PDT and still names no swarm clip. The X thread has no
swarm post, but its post 5 says "So every game labels every number on screen", the same claim that item 2 bounds. The
swarm's own frames do not bear it out.
