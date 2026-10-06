# hairdryer: steer the fly to the apple with a fan

**Blow-dried to lunch: in all three recorded seeds a scripted hand that sees the fly and the apple aimed a hairdryer,
the fly's wind neurons reported the jet, one declared decoder turned the fly into it, and its walk carried it within
reach of the apple, where a 1.5 cm rule switched on its sugar neurons and they drove MN9. On seed 1 its giant fibre fired half a millimetre
short of reach and it jumped, through the apple (the game's flight step has no apple in it); the hand carried the
hairdryer round, and the fly got back within reach with 0.08 s of the clip to spare. Disconnect the decoder and the
wind neurons still report the jet for as long as it blows, but the fly walks straight on: 0 of 3 controls came closer
to the apple than their start, against 3 of 3 decoder-on runs, all within reach, in the same B200 batch.**

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the fly) | MaleCNS v1.0 as shipped: 167,106 neurons, preset `raw`, nothing instrumented, trained or tuned for this game. Everything from the antennae to the wind descending neurons is the model: `Air.deflections` -> `fb.wind` -> Johnston's organ -> DNp18 / DNp33. So is everything from the eyes to the giant fibre (DNp01), and from the sugar GRNs to MN9. The HUD's DNp18 / DNp33, GF and MN9 numbers are its rates. The walking (`Locomotion`) and the takeoffs (`Flight`: GF >= 33 Hz, or wing power MNs >= 50 Hz for 0.3 s) are the shipped body model, unmodified. |
| **DECODER** (written for the game, declared) | One rule, `wind_turn`: yaw = 3 deg/s per Hz x EMA(0.15 s) of 1/2 [(DNp18 L - R) - (DNp33 L - R)], capped at +-60 deg/s. It turns the fly towards the side the index reports and is added to the body readout's own yaw. It reads the four DN rates and writes nothing into the brain. Apart from the body readout's own slow drift (a net 0.8-1.3 deg/s in the controls) and the shipped flight model's own yaw during the two takeoffs (9.3 deg between the 10 Hz samples either side of seed 1's jump, 3.4 deg across the B200 seed 0 hop), every turn in these clips is this decoder's. `proboscis_drawing` only draws the proboscis from MN9 and controls nothing. |
| **GAME** (not the fly) | The room and the apple. The hairdryer and its jet law (0.55 m/s x 0.20 m / d, capped at 1.0 m/s). The scripted holder: it sees the fly's pose and the apple, aims the jet, cancels the index's ~15 deg bias and keeps out of the rear dead zone. The 1.5 cm reach rule that switches the sugar GRNs on (`fb.taste(1)`). The feeding hold that stops the walk. The start pose. The surface the airborne fly lands on: a flat table top with no apple in it (`Flight.step` gets a table-height function). The camera, the 6x fly sprite, the wind streaks and the banners. |
| **not claimed** | The raw model does not turn on its own, and it is not shown to turn upwind or to seek food. The holder, not the brain, knows where the apple is: it points the jet along the fly's line to it (Limits). Contact is a reach rule on the cloth, not a landing on the apple. The flight step has no apple in it (Clips, seed 1). |

## Design

The shipped fly (MaleCNS v1.0, `preset="raw"`, no instruments) walks on the table of `scripts/room_demo.py` with one
apple on it and an invisible fence 2 cm inside the table edge. A hairdryer blows at it. The fly's Johnston's organs
feel the wind and its wind descending neurons lateralise. **A declared decoder turns the fly towards the side those
neurons report**: upwind for lateral winds. If the hairdryer is held beyond the apple, the decoder keeps the fly facing
up the jet and its straight walk carries it to the apple.
Near the apple the sugar receptors fire and drive MN9, the proboscis motor neuron. The raw model does not turn on its
own (DNa02 is held below threshold). Every turn in this game is the decoder's, added to the shipped body readout. In
record mode a scripted holder (GAME) aims the hairdryer. It knows the fly's pose and the apple's position, which the
brain does not, and it, not the brain, supplies the apple's direction (Limits). The screen says so: the hairdryer's label reads
"scripted holder: sees the fly's pose + the apple" with a GAME chip, and the footer prints the holder's law.

| what | kind | law and parameters |
|---|---|---|
| the brain | CONNECTOME | `FlyBrain(seed=...)`, preset `raw`, nothing instrumented; one read-only decoder attached (`fb.attach`, `kind="decoder"`, writes nothing) |
| the room | GAME | `world.make_room(seed, "apple")`: table, checked cloth, one apple (radius 4 cm, centre (0.25, 0.15, 0.79) m), the lamp. Fence 2 cm inside the table edge (room_demo `--fence`); `body.Locomotion`'s edge rule turns the fly inward there. The room's seed only places the grapes and blueberries, which the apple set drops, so the scene is the same for every seed |
| the fly's eyes | shipped | 1,466 ommatidial columns x 7 rays, ray-traced from the eye (1.2 mm above the cloth) into the same scene the main view shows, **including the hairdryer**; the mosaic is the exact radiance handed to `fb.vision` |
| the hairdryer | GAME | 12 spheres (nozzle ring, barrel, intake, handle), nozzle 7.5 cm above the cloth, always pointed at the fly |
| wind at the fly | GAME | blows from the nozzle to the fly (a uniform horizontal field in `flyverse.air.Air`, meander off); speed = 0.55 m/s x 0.20 m / d, d = the nozzle's horizontal distance to the fly (the 1/d centreline decay of a round jet), capped at 1.0 m/s. The apple's odour plume follows it. The jet has no width and the apple does not shadow it |
| antennae -> Johnston's organ | shipped sense | `Air.deflections` (antennae at +-45 deg, full deflection at 0.5 m/s) -> `fb.wind`: JO-E / JO-C rates per side |
| wind DNs | CONNECTOME | DNp18 L / R and DNp33 L / R (one cell each), the model's own rates. For lateral winds DNp18 fires on the wind side and DNp33 on the other; the frontal code is left-biased (the index's null is ~15 deg right of dead ahead) and the rear code is poor (below). The panel says so, and the decoder dial marks the null |
| `wind_turn` | DECODER | `u = 1/2 [(DNp18_L - DNp18_R) - (DNp33_L - DNp33_R)]` Hz of the brain's `rate_hz` (flyverse/brain.py's running rate estimate, 100 ms time constant; the previous 10 ms frame, via a `games.common.ReadDecoder`), `u_s = EMA(u, 0.15 s)`, **yaw = 3.0 deg/s per Hz x u_s**, clipped at **+-60 deg/s** (below), + = a left turn, i.e. **towards the side the index reports**. Added to `Locomotion.readout`'s yaw before `Locomotion.step`, whose 80 ms motor smoothing still applies. The whole chain from spikes to heading: 100 ms rate estimate -> 150 ms EMA -> 80 ms body smoothing |
| walking | shipped body | `flyverse.body.Locomotion` readout and step, unmodified: about 0.9 cm/s (mostly the body's 8 mm/s baseline drive); its own yaw is small (a net heading drift of 0.6 deg/s in the dev control and 0.8-1.3 deg/s in the three recorded controls, below) |
| takeoff | shipped body | `flyverse.body.Flight`, unmodified: GF (DNp01) >= 33 Hz -> escape jump; wing power MNs >= 50 Hz for 0.3 s -> a short voluntary hop. Both are logged and bannered with their cause |
| taste | GAME -> shipped sense | on its feet within 1.5 cm of the apple's surface (room_demo's rule; 3.77 cm from the apple's centre on the cloth) -> `fb.taste(1)`: the 165 sweet GRNs fire at 120 Hz. The fly is then on the cloth under the apple's bulge, 1.5 cm from its surface, not on it: the HUD says "within reach" / "in reach" and "1.5 cm rule", not "touching" |
| MN9 | CONNECTOME | mean rate of the two MN9 cells. The FEEDING banner comes 1 s after contact and quotes MN9's mean over that first second (or says MN9 SILENT); the panel then shows the mean since contact. The first MN9 rate above 2 Hz after contact is logged as its own event |
| `proboscis_drawing` | DECODER (display only) | the sprite's proboscis is drawn out while MN9 > 1 Hz, length 0.3 + 0.9 x min(1, MN9 / 20 Hz) mm (x the sprite's 6x). It controls nothing; declared in the run log |
| feeding hold | GAME | while the fly is within reach its walk is held (speed 0, yaw 0), as room_demo holds a feeding fly; the hairdryer switches off at first contact |
| the holder (record mode) | GAME | a scripted hand that knows the fly's pose and the apple's position (not the brain). Nozzle on the fly's line to the apple, 10 cm past the apple's centre and >= 15 cm from the fly; kept within 100 deg of the fly's heading (wind from behind barely turns the decoder, below); its bearing offset by the integral of the fly's heading error to the apple (k_i 0.6 /s, +-40 deg, only while the error is < 45 deg) to cancel the decoder's 15 deg bias (below); stops re-aiming within 7.5 cm of the apple's centre; moves round the fly at <= 0.20 m/s (0.10 m/s radially). Switched on at 1.5 s. On screen: the hairdryer's label ("scripted holder: sees the fly's pose + the apple", GAME chip) and the footer |
| the player (interactive) | GAME | the mouse places the hairdryer (6-60 cm from the fly, the hand at <= 0.6 m/s; it points at the fly); hold the left button to blow, or latch it with SPACE; the wheel scales the jet law by 0.3-2.4 but the wind stays capped at 1.0 m/s; A hands over to the scripted holder (its own timer; the brain-time clock is untouched), D disconnects the decoder, R restarts the round (the run log keeps each earlier round's summary) |
| start | GAME | fly at (0.105, 0.035) m heading -100 deg, 18.5 cm from the apple's centre, the apple behind it to its left; hairdryer off, arriving from 30 cm. The same for every seed |
| main view | GAME (display) | a camera side-on to the fly's line to the apple, on the side away from where the holder brings the hairdryer (so it never fills the foreground), 50 deg above the cloth, at the nearest distance that frames the fly and most of the apple, plus the hairdryer until 3.5 s after it switches on (always, in interactive mode). After contact it closes in over about 2 s on the fly and the near side of the apple. The fly is drawn 6x life size. None of this reaches the brain |
| wind streaks | GAME (display) | a picture of the jet, not its data: particles from the nozzle towards the fly that slide along the cloth; where they hit the apple they wrap round it and fade within ~0.3 s (head-on ones stop), and they are depth-tested so none crosses its face. The wind model itself has no apple in it (the wind at the fly is the unshadowed jet) |
| banners | GAME (display) | each carries the provenance chips of what it reports: HAIRDRYER ON / OFF (GAME), DECODER: TURN LEFT / RIGHT with the index u (CONNECTOME + DECODER), CONTACT (GAME), FEEDING: MN9 (CONNECTOME), GIANT FIBRE -> JUMP and WING MNs -> HOP (CONNECTOME, body model). A banner stays at full opacity for at least 1.2 s before a newer one replaces it (a jump replaces at once); the log keeps every event at its true time. A turn banner fires when the decoder's yaw passes 0.8 of its cap (at most every 3.5 s, before contact) **and** the turn goes towards the side the wind comes from; every such crossing is logged as a `turn` event with `towards_wind` true / false / null (null = wind from dead ahead or dead astern) |
| HUD traces | display | DNp18 L / R, the decoder's yaw and MN9 on a fixed 8 s axis (the newest sample at the right edge) |
| control run | GAME | `--decoder off`: same seed, the decoder is attached and logged but its yaw is not applied; the status box reads STEERING "OFF: control" and the decoder panel greys its turn and says "not applied (control)" |

Each run log (`<clip>.json`) holds every event (hairdryer on / off, decoder turn crossings, takeoffs with their cause,
contact, MN9's first rate above 2 Hz after contact, the 1 s feeding mean) and a 10 Hz sample of pose, distance, wind, the four
DN rates, `u`, the decoder's yaw, MN9 and the GF.

### What was measured first (dev seeds >= 100)

Provenance: the seed-100 30 s runs below have run logs (`out/games/hairdryer/dev*.json`, `prev_dev*.json`,
`fix/superseded_ba25/*.json`). The other dev numbers (seeds 101-111: the wind sweeps, closed-loop batches, MN9,
GF-rotation and walk-into-apple probes) come from probe scripts whose printed output was not kept as a run log, so
they are orientation, not results.

* **The wind code is clean from the sides, biased at the front and poor from behind.** Standing fly, one batched brain
  with a row per wind direction (seed 100, 1 s of wind at 0.4 m/s): `u` = +15 Hz for wind from dead ahead, +39 / +43 /
  +36 at 30 / 60 / 90 deg left, +13 at 120, -7 at 150, -15 behind, -22 / -27 / -32 / -30 at 210 / 240 / 270 / 300, -13
  at 330. DNp18 fires on the wind side (52.8 vs 8.4 Hz for wind from the left) and DNp33 on the other; but for wind
  from anywhere behind (150-240 deg) DNp33-L fires (13-50 Hz) whichever side it comes from, and DNp18 stays near
  baseline. Finer sweep (seed 101, 0.5 m/s): `u` crosses zero between 340 deg (-5.1 Hz) and 350 deg (+4.9 Hz), i.e.
  **about 15 deg right of dead ahead**, with a slope of about 1.2 Hz/deg; the rear crossing is near 135 deg and
  shallow (a reviewer's rerun on seed 101: -5.4 Hz at -20 deg, +1.0 Hz at -15 deg). `u` grows with speed (DNp18 L-R at
  90 deg: 9 / 19 / 33 / 44 / 63 / 68 Hz at 0.1 / 0.2 / 0.3 / 0.4 / 0.6 / 0.8 m/s) and rises within 50-100 ms of wind
  onset. Raw `u` fluctuates by 5-8 Hz SD; smoothed over 0.2 s, 1-3 Hz. Consequence: for a wind up to ~15 deg right of
  dead ahead the index reads "left", so the decoder turns the fly away from that wind. In the first dev clip, 32 of 97
  fan-on 10 Hz samples with a turn over 15 deg/s were such turns. The decoder panel labels each turn "towards the
  wind" or "wind on the other side", and no turn banner is shown for the latter.
* **Hence a turn towards the reported side (upwind).** Its fixed point near the front is steep and stable (the fly
  settles with the wind ~15 deg to its right). A turn-away decoder would settle near 135-140 deg, where the slope is
  shallow. So the hairdryer here is a lure, not a pusher. The DNp18 + DNp33 pair and the five-type groups
  `MotorRates.wind_ipsi / contra` gave the same signal-to-noise; the pair is the README's named, strongest
  lateralised signal.
* **Closed loop, no corrections** (seed 102, 10 s, six batched flies): walking speed 0.85-0.91 cm/s; the fly holds a
  12-21 deg heading offset (the null above); a fly whose fan stood at ~143 deg (behind-left) did not turn at gain 2 and
  took 5-8 s at gain 3.5. That is why the holder offsets its bearing and keeps the nozzle within 100 deg of the heading.
* **MN9** (seed 103, fly standing at the apple, sugar from 0.5 s): MN9 > 1 Hz within about 0.1-0.2 s of contact (a
  reviewer's rerun of the same seed: 0.19 / 0.13 / 0.15 s in three rows); 6-11 Hz per 0.5 s window over 4.5 s, a mean
  of 6.5-7.6 Hz per row, and 0 Hz without sugar.
* **The first full-length dev run lost the apple to the giant fibre.** Seed 100, 30 s, an earlier holder (jet
  0.6 m/s x 0.2 m / d, capped 1.2; hand 0.3 m/s; no settling): the fly turned and walked to the apple, but near it the
  fan swung and the wind was ~0.8 m/s, the heading oscillated by +-50 deg/s, and the GF crossed 33 Hz at 4.3 cm from the
  apple's centre (0.5 cm short of contact): it jumped over the apple, came back, and crossed again at 3.8 cm. Two short
  hops at 5.2 and 7.2 s came from the wing power MNs (>= 50 Hz for 0.3 s), not the GF. Controls: walking straight into
  the apple with no wind and no decoder (seed 104, six approach directions) peaked the GF at 19-30 Hz and 5 of 6 flies
  touched it; with the hairdryer standing (off) beyond the apple, 1 of 6 crossed 33 Hz (38.6 Hz at 5.2 cm). Wind alone
  does not drive the GF or the wing MNs (seed 105, standing fly, 0-1.2 m/s from the front or side: GF max < 10.1 Hz,
  power MN mean 14-20 Hz, no takeoff). The holder was then changed to what the table above says (stop re-aiming within
  7.5 cm, jet 0.55 m/s at 20 cm capped at 1.0, >= 15 cm away, hand 0.20 m/s).
* **With that holder** (seed 106, 28 s, six flies in one batched brain, three each at gains 2 and 3, yaw cap still
  200 deg/s): 6 of 6 reached the apple, at 19.7-20.4 s; no takeoffs; GF max before contact 20-26 Hz; mean |heading
  error to the apple| after 3 s 3-5 deg; mean |decoder yaw| 9-16 deg/s. Gain 3 was kept (a quicker first turn).
* **Fast turns next to the apple drive the giant fibre.** A fly started 7.8 cm from the apple with the hairdryer
  coming round from behind (seed 107) was being turned left at 100-120 deg/s when the GF rose from 2 to 31 Hz in 0.3 s
  and crossed 33 Hz at 6.6 cm from the apple; it jumped. Measured directly (a standing fly rotated in place for 2.5 s,
  no wind): **7 cm from the apple**, seed 108 peaked the GF at 27 Hz at 45 deg/s and 32.2 Hz at 70 deg/s, and 90 /
  110 / 140 deg/s crossed 33 Hz 3 / 8 / 5 times (peaks 36-41 Hz); a reviewer's rerun on seed 111 peaked at 31.4 Hz at
  45 deg/s and crossed 33 Hz twice at 70 deg/s (max 34.1 Hz). **18 cm away**, every rate up to 140 deg/s stayed under
  25 Hz (seed 111: under 18.7 Hz). So the decoder's yaw is capped at 60 deg/s (the turn banner threshold is 48 deg/s,
  0.8 of the cap). The cap reduces turn-evoked escapes near the apple; it does not remove them. Why a self-turn near
  the apple drives the GF (e.g. the apple sweeping across the eye acting like a loom on LC4 / LPLC2) is a hypothesis;
  LC4 / LPLC2 were not measured here.
* **With the 60 deg/s cap** (seed 109, 28 s, six flies in one batched brain, gain 3): from the standard start 3 of 3
  reached the apple, at 20.7-21.3 s, with no takeoffs and the GF at most 23-30 Hz before contact. From a start 7.8 cm
  from the apple with the hairdryer coming round from behind (`--start 0.19,0.10,20`), 3 of 3 jumped on the giant
  fibre at 2.4-2.8 s, 5.4-5.8 cm from the apple, while being turned towards the wind, then walked back and touched it at
  12.4-14.1 s. The recorded start keeps the big turn 18 cm from the apple, where turning does not reach the GF, and the
  holder's settling keeps the last few centimetres straight.
* **Earlier dev pairs** (seed 100, 30 s, same brain-facing logic, older display code): contact at 21.03 s (sha256
  `8cb8f1c741a2d11b`, control from `e4d53dc5e59a7031`: 40.9 cm from the apple at the end) and 20.76 s (`ba2532865498c996`,
  control 40.7 cm); a run on an even earlier revision touched at 20.79 s (no run log kept). The CUDA run is not bit-reproducible, so
  the same seed and logic differ by a fraction of a second.
* **The dev pair with the frozen code** (`games/hairdryer.py` sha256 `c292d4b115a04db4`, both clips; seed 100, 30 s;
  `out/games/hairdryer/dev.mp4` and `dev_control.mp4`). **Decoder on:** hairdryer on at 1.5 s, 15 cm away, wind
  0.73 m/s from 100 deg left. At 1.62 s DNp18 L 49.7 vs R 1.0 Hz and DNp33 R 56.9 Hz; the smoothed index u = +16.9
  Hz gave a 50.8 deg/s left turn, towards the wind (logged and bannered). The heading went from -100 deg to +57 deg
  by 4.1 s. A second crossing at 13.43 s turned the fly left at 48.1 deg/s with the wind 7.1 deg to its *right* (the
  frontal null). It was logged with `towards_wind: false` and not bannered. Of 101 fan-on 10 Hz samples with
  |decoder yaw| >= 15 deg/s, 37 turned away from the wind side. Contact came at 20.90 s (19.40 s after switch-on),
  after 18.96 cm of walking. There were no takeoffs of either kind; GF max 25.9 Hz; mean |heading error to the apple|
  with the hairdryer on 12.9 deg (including the first turn). MN9 rose above 2 Hz 0.02 s after contact (a single run;
  the standing-fly latency above is 0.1-0.2 s), averaged 7.4 Hz over the first second (the FEEDING banner) and
  6.6 Hz over 9.1 s within reach (max 33.3 Hz). **Control** (`--decoder off`, same seed and code): the holder ran the
  same law. With the fly never turning towards the apple, the holder stayed clamped 100 deg to the fly's left, 15 cm
  away, for the whole run (wind 0.73 m/s from 100 deg left). The wind DNs lateralised hard throughout (mean |u| 48.9
  Hz; DNp18 L 68.0 vs R 10.1 Hz with the fan on). Only the shipped `Locomotion.readout`'s own small yaw turned the
  fly: the heading drifted from -100.0 deg to at most -73.3 deg and ended at -82.5 deg, under 1 deg/s against the
  decoder's 60 deg/s cap (the earlier control drifted 28.5 deg: -100.0 to -71.5). It walked 27.1 cm, never came
  closer than its 18.5 cm start, and ended 40.2 cm from the apple's centre; GF max 25.2 Hz, no takeoff.

### Limits

* The turn is the game's decoder, not a behaviour of the raw model; a real fly in a hairdryer's wind would not
  necessarily turn upwind without odour.
* The holder supplies all the knowledge of where the apple is: it sees the fly and the apple, points the jet along
  the fly's line to the apple, cancels the decoder's 15 deg bias and avoids the rear dead zone. The brain's part is
  to report which side the wind comes from; the decoder turns the fly towards that side. How the work splits between
  them was not measured. The holder is labelled on screen (GAME).
* Turning near the apple drives the giant fibre (above): within ~8 cm of the apple even the capped decoder's turns
  can make the fly jump. A player who swings the hairdryer round a fly next to the apple will see it escape. The
  recordings report every takeoff with its cause.
* The shipped body's voluntary takeoff (wing power MNs >= 50 Hz for 0.3 s) fires now and then while the fly walks: a
  short hop of about 2 cm (0-2 per 30 s dev run on seed 100). It is the body model's, not the game's.
* Walking is slow (about 0.9 cm/s, mostly the body's baseline drive), so the herd takes about 20 s for 15 cm.
* MN9 has a known low draw (calibrated 3.93-5.52 Hz in the README's ledger); here it averaged 5.8-6.8 Hz within
  reach of the apple over 8-11 s (five runs; RTX 4090 seed 1 was within reach for only 0.09 s). Two cells: the instantaneous rate jumps between 0 and ~30 Hz, which is why the banner and the
  panel quote means.
* "Contact" is room_demo's reach rule: on the cloth within 1.5 cm of the apple's surface, not on the apple.
* `flyverse.world.render_camera` returns a left-right mirrored image (column 0 looks to the camera's right;
  `room_demo`'s own `Camera.project` mirrors to match). This game flips the render and projects correctly; the fly's
  eyes use `World.trace` directly and are unaffected.
* Labels (APPLE, HAIRDRYER) and the wind streaks are drawn over the hairdryer regardless of depth; the streaks and the
  contact ring are depth-tested against the apple only.

Record command (the three recorded seeds and their controls; from the repo root, `PYTHONIOENCODING=utf-8`; 6-52 min
of wall per clip on the shared 4090, depending on contention, and 7 min on the cluster's NVIDIA B200s with six clips
running at once):

    python games/hairdryer.py --seed {k} --seconds 30 --record out/games/hairdryer/seed{k}.mp4
    python games/hairdryer.py --seed {k} --seconds 30 --decoder off --record out/games/hairdryer/control_seed{k}.mp4

30 s of brain time: on dev seeds the fly touches the apple at about 20-21 s, leaving about 9 s of feeding.

## Clips

Nine clips, recorded on two GPUs from one game file: `games/hairdryer.py` sha256 `c292d4b115a04db4` (first 16 hex),
the frozen code of the dev pair above. **Comparisons between the arms are made within one device.** Each seed's
decoder-on and control runs were recorded side by side, in one batch on the house cluster's NVIDIA B200s, and are
compared there. The logs name the device (NVIDIA B200) but not which of the cluster's B200s ran each job. The RTX 4090 set is
the recording-policy set of three decoder-on seeds (the trailer uses seed 0). Its controls were stopped before they
finished (below), so it has no control of its own, and it is set beside the B200 set only as a replicate of the same
command.

| clips | device | recorded | torch, Python | commit | `games/common.py` | wall per clip (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|---|
| `seed{0,1,2}.mp4`: decoder on | NVIDIA GeForce RTX 4090, this desktop (Windows 11) | 2026-09-28, started 22:53-22:56 UTC, side by side on a GPU that other games' recordings were also using | 2.10.0+cu128, 3.13.2 | `959f2e927b1c2806bc526fd5fc51cb3f5cc3e4b1`, `dirty: true` (games/ is not committed yet) | `b110142562a927a1` | 2,864 / 3,063 / 3,100 s |
| `control_seed{0,1,2}.mp4`: `--decoder off` | NVIDIA B200, house cluster: all six B200 clips in one batch (the logs do not record the GPU index) | 2026-09-29, started 00:42:58-00:43:03 UTC, all six side by side | 2.11.0+cu128, 3.12.3 | `null`: the cluster runs a copy of the working tree, not a git checkout; `sources` identifies the code | `e06078fb69d1275a` | 441 / 441 / 441 s |
| `b200_seed{0,1,2}.mp4`: decoder on, a same-device replicate | NVIDIA B200, in the same batch | as above | as above | as above | as above | 404 / 420 / 415 s |

`games/common.py` changed once between the two sets, in its ffmpeg lookup: `find_ffmpeg` now falls back to the binary
that the `imageio-ffmpeg` package bundles, because the cluster has no ffmpeg on PATH. That is why its hash differs
between the RTX 4090 and the B200 logs. The earlier file was not kept, so the two versions are not diffed here. The
game file is the same in all nine logs. Every log records MaleCNS v1.0, 167,106 neurons, preset `raw`, no instruments
and one attached module, `wind_turn`, kind `decoder`. The controls attach it too but do not apply its yaw
(`decoder_connected: false`). `meta.declared` lists the two decoders (`wind_turn`, `proboscis_drawing`) and seven GAME
rules (turn banner, jet, holder, contact taste, feeding hold, fence, start). Each clip is 30.0 s of brain time (1,500
frames, 1920x1080, 50 fps). "Wall" is the log's `wall_s`. On the RTX 4090 it swung with the load that other
recordings put on the shared GPU (2.4-2.6 times the seed-100 dev clip's 20 min). Recording is offline, so the load
changed only the wall time.

Each decoder-on command ran exactly once; each control command completed once, on the B200, after a first attempt
on the RTX 4090 was stopped (below). All ran from the repo root. On the desktop the interpreter was
`PYTHONIOENCODING=utf-8 <miniconda>/python.exe`. On the cluster each job activated the cluster's venv,
asserted `torch.cuda.is_available()` (it printed `NVIDIA B200`), then ran `python` with the arguments below and wrote
its console to `out/games/hairdryer/<clip>.console.txt`. The batch (`scripts/cluster_run.py`, run
`games-hairdryer-rec-8f3556`) ended with "6 job(s), 0 failed".

    python games/hairdryer.py --seed {k} --seconds 30 --record out/games/hairdryer/seed{k}.mp4                        # RTX 4090
    python games/hairdryer.py --seed {k} --seconds 30 --decoder off --record out/games/hairdryer/control_seed{k}.mp4  # B200
    python games/hairdryer.py --seed {k} --seconds 30 --record out/games/hairdryer/b200_seed{k}.mp4                   # B200

**The controls' first attempt was stopped.** The three control commands first ran on the RTX 4090 after the
decoder-on runs, starting at 23:41-23:48 UTC. They were stopped at 00:28:15 UTC, when game recording was moved off the
desktop's GPU. By then they had reached at least 28, 16 and 22 s of their 30 s of brain time (seeds 0 / 1 / 2: the
last progress lines in `out/games/hairdryer/rec/control_seed{k}_stdout.txt`). They wrote no run log, and their .mp4 files
have no index (ffprobe: "moov atom not found"), so they have no outcome to report. They are kept in
`out/games/hairdryer/killed_partial/`. The same three commands then ran once each on the B200, with the decoder-on
replicate beside them so that both arms were recorded on one device.

What the run logs say, at a glance (times in brain seconds; "GF max" is the run's maximum of the 10 ms GF rate;
distances are from the apple's centre, and contact is at 3.77 cm):

**RTX 4090, decoder on** (the trailer's clips):

| clip | contact | after switch-on | walked | takeoffs | GF max | mean abs. heading error, fan on | MN9 > 2 Hz after contact | FEEDING banner (1 s mean) | MN9 within reach |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 20.48 s | 18.98 s | 18.61 cm | none | 20.69 Hz | 12.8 deg | +0.17 s (4.58 Hz) | 5.7 Hz (5.73) | 6.84 Hz mean over 9.53 s, max 31.96 |
| seed 1 | 29.92 s | 28.42 s | 38.45 cm (incl. the hop) | 1 giant fibre, at 21.42 s | 69.09 Hz (in the hop; 34.23 at takeoff) | 22.9 deg | +0.02 s (4.94 Hz) | none: the clip ends 0.08 s after contact | 7.52 Hz mean over 0.09 s |
| seed 2 | 21.25 s | 19.75 s | 19.01 cm | none | 26.91 Hz | 13.9 deg | +0.23 s (4.91 Hz) | 5.8 Hz (5.84) | 6.51 Hz mean over 8.76 s, max 28.81 |

**NVIDIA B200, both arms** (one batch, all six side by side):

| clip | contact | after switch-on | walked | takeoffs | GF max | mean abs. heading error, fan on | MN9 > 2 Hz after contact | FEEDING banner (1 s mean) | MN9 within reach |
|---|---|---|---|---|---|---|---|---|---|
| b200 seed 0 | 18.87 s | 17.37 s | 18.64 cm (incl. the hop) | 1 wing power MNs, at 5.99 s | 24.64 Hz | 13.7 deg | +0.03 s (4.70 Hz) | 8.5 Hz (8.48) | 6.67 Hz mean over 11.14 s, max 28.43 |
| b200 seed 1 | 21.83 s | 20.33 s | 19.16 cm | none | 29.78 Hz | 14.2 deg | +0.03 s (9.71 Hz) | 6.0 Hz (6.02) | 5.79 Hz mean over 8.18 s, max 27.42 |
| b200 seed 2 | 21.02 s | 19.52 s | 18.95 cm | none | 26.61 Hz | 13.7 deg | +0.03 s (4.72 Hz) | 8.6 Hz (8.57) | 6.65 Hz mean over 8.99 s, max 26.47 |
| control seed 0 | none: never closer than its 18.51 cm start; 40.09 cm at the end | - | 27.41 cm | none | 27.98 Hz | 142.1 deg | - | - | - |
| control seed 1 | none (18.51 cm at the start, 41.05 cm at the end) | - | 26.70 cm | none | 28.25 Hz | 148.0 deg | - | - | - |
| control seed 2 | none (18.51 cm at the start, 40.08 cm at the end) | - | 26.91 cm | none | 25.11 Hz | 143.6 deg | - | - | - |

All six decoder-on flies reached the apple; none of the three controls did. The dev pair's fly (seed 100, RTX 4090)
reached it at 20.90 s. No RTX 4090 run had a wing-power-MN hop; on the B200, seed 0 had one.

### RTX 4090, decoder on

**The first turn is the same in all three.** At 1.50 s the holder switches the hairdryer on, 15.0 cm from the fly;
the wind at the fly is 0.73 m/s from 100 deg left. In the 120 ms that follow, the wind DNs lateralise. The smoothed
index then passes 0.8 of the yaw cap at 1.62 s in every seed. Each crossing is logged as a left turn towards the
wind and bannered ("DECODER: TURN LEFT"). The banner shows from about 2.70 s, once "HAIRDRYER ON" has had its 1.2 s:

| seed | DNp18 L / R | DNp33 L / R | index (smoothed) | decoder yaw |
|---|---|---|---|---|
| 0 | 46.1 / 0.2 Hz | 0.0 / 48.7 Hz | +17.1 Hz | 51.3 deg/s left |
| 1 | 34.8 / 18.9 Hz | 0.0 / 55.0 Hz | +16.2 Hz | 48.5 deg/s left |
| 2 | 57.2 / 10.8 Hz | 0.1 / 55.1 Hz | +17.7 Hz | 53.2 deg/s left |

In the 10 Hz samples, the decoder's yaw sat at its 60 deg/s cap from 1.7 s to 4.1-4.2 s. The heading swung from
-92.0 / -104.7 / -96.8 deg (seeds 0 / 1 / 2) at switch-on through the apple's bearing to +74.4-74.7 deg at 4.6-4.8 s.
It overshot: the index reads "left" until the wind is about 15 deg to the right (Design), and the chain from spikes to
heading lags (100 ms rate estimate, 150 ms EMA, 80 ms body smoothing); at 4.4 s on seed 0 the wind was 21 deg to the
right and the decoder still turned the fly left at 14.5 deg/s. It then came back.
The heading error to the apple first fell under 20 deg at 3.6 / 3.8 / 3.7 s. In frames from about 4.2 s, the decoder
panel reads "wind on the other side" while the earlier turn banner is still up.

**Seed 0** (`out/games/hairdryer/seed0.mp4`). Steered by the decoder, the fly walked up the jet. From 4.6 s to contact
(10 Hz samples), the mean abs. heading error to the apple was 4.2 deg and the mean abs. decoder yaw was 14.6 deg/s.
The wind at the fly rose from 0.39 to 0.63 m/s as the hairdryer's distance shrank. A second threshold crossing came at 15.52 s: 48.4 deg/s left, index
+16.1 Hz, the wind 2.6 deg to the right. It was logged with `towards_wind: null` (wind from dead ahead) and was not
bannered. Contact came at 20.48 s after 18.61 cm, and the hairdryer switched off ("CONTACT" banner). MN9 rose above
2 Hz 0.17 s later. "FEEDING: MN9 5.7 Hz" (the 1 s mean, 5.73 Hz) shows from about 21.68 s. Within reach, MN9
averaged 6.84 Hz over 9.53 s, with a maximum of 31.96 Hz. The fly never took off, and the GF peaked at 20.69 Hz. Of
93 fan-on 10 Hz samples with abs. decoder yaw >= 15 deg/s, 30 turned the fly away from the side the wind came from
(the frontal null).

**Seed 2** (`out/games/hairdryer/seed2.mp4`) went the same way. From 4.6 s to contact, the mean abs. heading error
was 5.4 deg and the mean abs. yaw 15.4 deg/s. A second crossing came at 16.40 s: 50.1 deg/s left, index +16.7 Hz,
the wind 8.8 deg to the right. It was logged `towards_wind: false` and not bannered. Contact came at 21.25 s after
19.01 cm. MN9 rose above 2 Hz 0.23 s later. "FEEDING: MN9 5.8 Hz" (5.84 Hz) shows from about 22.45 s. Within reach,
MN9 averaged 6.51 Hz over 8.76 s (max 28.81 Hz). There were no takeoffs, and the GF peaked at 26.91 Hz. Of 101
samples with abs. yaw >= 15 deg/s, 34 turned away from the wind's side.

**Seed 1** (`out/games/hairdryer/seed1.mp4`): the giant fibre fires at the apple. After the same first turn, the decoder
kept the fly walking up the jet. At 20.0 s it was 5.12 cm from the apple's centre (seeds 0 and 2: 4.20 and 4.87 cm).
* **The jump.** From 20.4 s it walked nearly straight at the apple. In the 10 Hz samples the heading error was 0-11
  deg and the decoder's yaw ran from -17 to +35 deg/s. The GF rose from 0.4 Hz (20.6 s) through 12.9 Hz (21.2 s) to
  25.6 Hz (21.4 s). At 21.42 s it crossed 33 Hz: takeoff, cause giant fibre, GF 34.23 Hz, wing power MNs 21.76 Hz.
  The fly was 3.82 cm from the apple's centre, 0.5 mm outside the 3.77 cm reach radius. Banner: "GIANT FIBRE
  34 Hz -> JUMP". The dev runs above found the same: walking straight at the apple peaked the GF at 19-30 Hz, and 1
  of 6 approaches crossed 33 Hz. LC4 / LPLC2 were not logged here.
* **The hop went through the apple.** This is a limit of the game's physics, not an event. The game hands
  `Flight.step` a flat table top with no apple in it, so for the 0.33 s of the hop the fly moved through the apple's
  volume. In the 10 Hz sample at 21.5 s it is 0.89 cm from the apple's centre, where the apple is solid from 1 mm
  above the cloth. The run log does not record height, but the shipped escape jump (0.6 m/s, 45 deg up) rises about
  2.5 cm on its launch alone (the shipped `Flight` constants, gravity 3 m/s^2 and drag 2.5 /s, integrated at 10 ms
  with the wing MNs silent), and the apple is 8 cm tall. At 21.52 s the eye mosaic is all apple, because
  the eyes were inside it. The GF's run maximum, 69.09 Hz (66.5 Hz in the 21.5 s sample), came during this pass, so
  it is a response to an impossible view. Everything that follows starts from a landing a real apple would not
  allow. The sprite is drawn over the apple, so on screen it reads as a jump over the apple. It was not one.
* **The way back.** The fly landed at 21.75 s at (0.293, 0.217) m, about 8 cm from the apple's centre on its far
  side, facing away (heading 60.6 deg in the 21.8 s sample). The main camera loses it from about 21.7 s; it is back at the frame's edge at 23.2 s and fully in view by 23.5 s.
  Because the fly was now more than 7.5 cm out, the holder (GAME) aimed again and carried the hairdryer round to the
  other side of the apple. At 21.89 s the decoder crossed its threshold to the right: 50.3 deg/s, index -16.8 Hz,
  DNp18 R 59.0 vs L 13.5 Hz, the wind 41.8 deg to the right. That was towards the wind, so it was bannered ("DECODER:
  TURN RIGHT", shown from about 22.60 s). The yaw sat at the -60 deg/s cap in every 10 Hz sample from 22.0 to 25.0 s. The heading went
  from 60.6 to -149.8 deg, a 210 deg right turn, by 25.6 s. The fly was 7.7-9.2 cm from the apple's centre, the GF
  stayed at or below 26.0 Hz in the 10 Hz samples, and there was no second takeoff. The heading error fell under
  20 deg at 24.8 s. One more crossing came at 25.82 s: 48.9 deg/s left, the wind 6.8 deg right, `towards_wind: false`,
  not bannered. It then walked the last 4.0 cm to the apple (10 Hz samples from 25.8 s).
* **Contact.** It touched at 29.92 s, 0.08 s before the end of the clip and 28.42 s after switch-on. It walked
  38.45 cm in all (the log's path length): about 18.9 cm up to the jump and 7.6 cm from the 21.8 s sample to the end
  (10 Hz samples), so about 12 cm airborne. MN9 rose above 2 Hz
  0.02 s after contact (4.94 Hz). The clip ends before the 1 s FEEDING banner: the fly was within reach for 0.09 s,
  with an MN9 mean of 7.52 Hz. The last frame shows "CONTACT" with MN9 7.5 Hz. The mean abs. heading error with the
  fan on was 22.9 deg, including the U-turn after landing. Of 143 samples with abs. yaw >= 15 deg/s, 42 turned away
  from the wind's side.

### NVIDIA B200: the same seeds with the decoder on and off

**Controls** (`out/games/hairdryer/control_seed{0,1,2}.mp4`, `--decoder off`). The holder ran the same law as in every
run. The fly never turned towards the apple, so the holder stayed clamped 100 deg to the fly's left and 15 cm away
from switch-on to the end: every fan-on 10 Hz sample of all three runs has the wind at 0.73 m/s from 100 deg left.
The wind DNs lateralised for the whole 28.5 s: DNp18 L exceeded R in every fan-on sample. DNp18 L averaged 67.9 /
68.5 / 68.8 Hz against R 9.9 / 11.0 / 10.1 Hz, and DNp33 R 39.7 / 39.0 / 39.8 Hz against L 0.0 Hz (seeds 0 / 1 / 2). The decoder, attached but
not applied, computed a left turn at its +60 deg/s cap in 284 of the 285 fan-on samples of each run (mean smoothed
index 48.2 / 48.4 / 49.0 Hz). The panel says "not applied (control)", the status box says "OFF: control", and no turn
was logged or bannered. Only the shipped `Locomotion.readout`'s own yaw turned the fly. From switch-on to the end its
heading drifted left by 35.8 / 24.6 / 23.8 deg (1.26 / 0.86 / 0.84 deg/s), ending at -63.6 / -80.0 / -75.6 deg. It
walked 27.41 / 26.70 / 26.91 cm at about 0.9 cm/s. It never came closer to the apple's centre than its 18.51 cm
start, and it ended 40.09 / 41.05 / 40.08 cm from it, about 15 cm inside the fence. In the last frame the HUD reads
36.3 / 37.3 / 36.3 cm "to the apple" (to its reach radius). There were no takeoffs, the GF peaked at 27.98 / 28.25 /
25.11 Hz, and MN9 was 0 Hz in every sample.

**Decoder on** (`out/games/hairdryer/b200_seed{0,1,2}.mp4`). The first crossing came at 1.62 s in all three. It was
logged as a left turn towards the wind and bannered. For seeds 0 and 2 it logged the same DNp18, DNp33, index and yaw
values as on the RTX 4090 (table above). For seed 1 it logged DNp18 L / R 24.7 / 8.9 Hz, DNp33 0.0 / 54.7 Hz, index
+16.1 Hz and 48.4 deg/s left. The yaw sat at its cap from 1.7 s to 4.1-4.2 s (and on seed 1 again in the 12.2 s
sample, after its second crossing). The heading swung from the same -92.0 /
-104.7 / -96.8 deg at switch-on to +74.7 / +74.0 / +76.1 deg at 4.5-4.7 s. The heading error first fell under 20 deg
at 3.6 / 3.7 / 3.7 s.
* **Seed 0.** At 5.99 s the wing power MNs (59.31 Hz, held 0.3 s; GF 2.19 Hz) launched the body model's voluntary
  hop, 17.01 cm from the apple's centre. The banner read "WING MNs 59 Hz -> HOP". The fly landed at 6.10 s, about
  1.9 cm nearer the apple (17.01 cm at takeoff, 15.09 cm at the logged landing point). From 4.6 s to contact the mean abs.
  heading error was 4.5 deg and the mean abs. yaw 15.7 deg/s. There was no second crossing. Contact came at 18.87 s,
  after 18.64 cm including the hop. MN9 rose above 2 Hz 0.03 s later (4.70 Hz). "FEEDING: MN9 8.5 Hz" (8.48 Hz) shows
  from about 20.1 s. Within reach, MN9 averaged 6.67 Hz over 11.14 s (max 28.43 Hz). The GF peaked at 24.64 Hz. Of 91
  fan-on samples with abs. yaw >= 15 deg/s, 29 turned away from the wind's side.
* **Seed 1.** A second crossing came at 12.08 s: 48.4 deg/s left, index +16.1 Hz, the wind 1.3 deg left of dead
  ahead. It was logged `towards_wind: null` and not bannered. From 4.6 s to contact the mean abs. heading error was
  5.1 deg and the mean abs. yaw 14.9 deg/s. There was no takeoff. The GF's run maximum was 29.78 Hz, before contact;
  the 10 Hz samples read 24.4 Hz at 21.8 s, 3.80 cm from the apple's centre. Contact came at 21.83 s after 19.16 cm.
  MN9 rose above 2 Hz 0.03 s later (9.71 Hz). "FEEDING: MN9 6.0 Hz" (6.02 Hz) shows from about 23.0 s. Within reach,
  MN9 averaged 5.79 Hz over 8.18 s (max 27.42 Hz). 34 of 101 samples with abs. yaw >= 15 deg/s turned away from the
  wind's side.
* **Seed 2.** There was no second crossing. From 4.6 s to contact the mean abs. heading error was 5.1 deg and the mean
  abs. yaw 13.5 deg/s. There was no takeoff, and the GF peaked at 26.61 Hz. Contact came at 21.02 s after 18.95 cm. MN9
  rose above 2 Hz 0.03 s later (4.72 Hz). "FEEDING: MN9 8.6 Hz" (8.57 Hz) shows from about 22.2 s. Within reach, MN9
  averaged 6.65 Hz over 8.99 s (max 26.47 Hz). 28 of 87 samples with abs. yaw >= 15 deg/s turned away from the wind's
  side.

**On and off, same seed, same device, same batch.** With the decoder connected the fly reached the apple at 18.87 / 21.83 / 21.02 s.
With it disconnected the fly never came closer than its start and ended about 40 cm away. In both arms the wind DNs
reported the jet on the fly's left as soon as it came on (DNp18 L above R, DNp33 R above L). The difference is whether
the decoder's turn is applied; the holder runs the same law in both, and where it puts the hairdryer then follows
from the fly's pose. The arms differ from the first or second 10 Hz sample (0.1 s; 0.2 s on seed 1), before the
hairdryer comes on,
because the connected decoder already turns the fly a little on the wind DNs' baseline fluctuations. The headings at
switch-on were -92.0 / -104.7 / -96.8 deg with the decoder and -99.4 / -104.6 / -99.4 deg without it.

### The same command on two GPUs (decoder on)

`seed{k}.mp4` (RTX 4090) and `b200_seed{k}.mp4` (B200) ran the same command on the same game file. This is a
replicate, not a comparison of arms. The simulation is not bit-reproducible on CUDA, and the two devices also differ
in torch and Python. The logged 10 Hz samples agree to their printed precision up to 4.3 s (seed 0), 1.5 s (seed 1)
and 1.8 s (seed 2), then drift apart. Seed 1's first turn crossing already differs (above). All six reached the
apple: seed 0 at 20.48 vs 18.87 s, seed 1 at 29.92 vs 21.83 s, seed 2 at 21.25 vs 21.02 s (RTX 4090 vs B200). Each of
the two large differences comes with a takeoff that happened on one GPU and not the other:
* On the B200, seed 0's voluntary hop carried the fly about 1.9 cm towards the apple, about 2 s of walking at
  0.9 cm/s. Its contact came 1.61 s before the RTX 4090's.
* On the RTX 4090, seed 1's giant fibre crossed 33 Hz 3.82 cm from the apple's centre and the fly jumped (above). On
  the B200 the same seed's GF peaked at 29.78 Hz, and the fly walked into reach.

On the approach the GF can come close to its 33 Hz threshold on either device. Its run maxima for seeds 0 and 2
were 20.69 and 26.91 Hz on the RTX 4090 and 24.64 and 26.61 Hz on the B200. On seed 1 it reached 34.23 Hz at the RTX
4090's takeoff and peaked at 29.78 Hz on the B200, so the same seed crossed on one device and not on the other. The FEEDING banner's
1 s MN9 mean was 5.7 / none / 5.8 Hz on the RTX 4090 and 8.5 / 6.0 / 8.6 Hz on the B200. Within reach, MN9 averaged
6.84 / 7.52 (over 0.09 s) / 6.51 Hz and 6.67 / 5.79 / 6.65 Hz.

## Corrections (caption skeptic)

Checked against all nine run logs (`out/games/hairdryer/{seed,b200_seed,control_seed}{0,1,2}.json`), the dev logs,
`cluster_rec.log`, the console and attempt files, frames from `seed0.mp4`, `seed1.mp4`, the B200 clips and the
controls, and `games/hairdryer.py` / `flyverse/body.py`. No fly brain was run. Changes:

1. Headline: "a hairdryer's jet, the fly's wind neurons and one declared turning decoder got it to the apple, where its
   sugar neurons drove MN9" -> "a scripted hand that sees the fly and the apple aimed a hairdryer, the fly's wind
   neurons reported the jet, one declared decoder turned the fly into it, and its walk carried it within reach of the
   apple, where a 1.5 cm rule switched on its sugar neurons and they drove MN9". The headline left out the GAME holder
   that supplies the apple's direction, and "to the apple" hid the reach rule.
2. Headline: "its giant fibre fired half a millimetre short and it jumped; it got back to the apple" -> "... short of
   reach and it jumped, through the apple (the game's flight step has no apple in it); the hand carried the hairdryer
   round, and the fly got back within reach". The return depended on a physically impossible pass through the apple
   and on the holder re-aiming.
3. Headline: "report the jet for the whole run, but the fly walks away" -> "for as long as it blows, but the fly walks
   straight on" (the jet is off for the first 1.5 s; the raw model walks straight).
4. Headline and Clips: "3 of 3 decoder-on replicates on the same GPU", "all six B200 clips on one GPU (index 1)", "on
   one NVIDIA B200", "(one GPU, all six side by side)", "same seed, same GPU", "the same GPU, in the same batch", "7 min
   on an NVIDIA B200 with six clips sharing it" -> "in the same B200 batch" / "in one batch (the logs do not record the
   GPU index)" and matching wording. No log, console file or `cluster_rec.log` records a GPU index or that the six
   budgeted jobs shared one GPU; they record the device (NVIDIA B200) and the batch.
5. DECODER row: "It reads and writes nothing into the brain" -> "It reads the four DN rates and writes nothing into the
   brain".
6. DECODER row: "Apart from the body readout's own slow drift ..., every turn in these clips is this decoder's" -> adds
   "and the shipped flight model's own yaw during the two takeoffs (9.3 deg between the 10 Hz samples either side of
   seed 1's jump, 3.4 deg across the B200 seed 0 hop)". `Flight.step` turns the airborne fly from the steering MNs.
7. "not claimed" row, Design: "The holder does part of the steering" / "it does part of the steering" -> "The holder,
   not the brain, knows where the apple is: it points the jet along the fly's line to it" / "it, not the brain,
   supplies the apple's direction".
8. Limits: "The holder does about half the work ... The brain only has to turn towards the wind it reports" -> "The
   holder supplies all the knowledge of where the apple is ... How the work splits between them was not measured."
   "About half" was not measured.
9. Design, seeds 0 and 1: "the fly walks up the jet to it", "The fly walked up the jet", "the fly walked up the jet" ->
   "the decoder keeps the fly facing up the jet and its straight walk carries it to the apple", "Steered by the
   decoder, the fly walked up the jet", "the decoder kept the fly walking up the jet".
10. Seed 1: "the shipped escape jump (0.6 m/s, 45 deg up) rises about 1 cm without lift" -> "about 2.5 cm on its launch
    alone" (integrating the shipped `Flight` constants at 10 ms: 2.5 cm with the wing MNs silent, about 3.1 cm for a
    hop of the logged 0.33 s). The conclusion (the fly passed through the 8 cm apple) stands.
11. Seed 1: "It walked 38.45 cm in all: 18.88 cm up to the jump, about 12 cm airborne, and 7.63 cm after landing" ->
    "38.45 cm in all (the log's path length): about 18.9 cm up to the jump and 7.6 cm from the 21.8 s sample to the end
    (10 Hz samples), so about 12 cm airborne". 18.88 is in no log; the split comes from the 10 Hz samples.
12. Seed 1: "The fly then walked 7.6 cm back" (after the 25.82 s crossing) -> "It then walked the last 4.0 cm to the
    apple (10 Hz samples from 25.8 s)". 7.6 cm is the whole post-landing walk, 3.5 cm of it during the U-turn.
13. Seed 1: "The yaw sat at the -60 deg/s cap for most of 22.0-25.0 s" -> "in every 10 Hz sample from 22.0 to
    25.0 s"; "(heading 60.6 deg)" -> "(heading 60.6 deg in the 21.8 s sample)".
14. First turn: "It overshot because the index reads 'left' until the wind is about 15 deg to the right" -> adds the
    lag of the chain (100 ms rate, 150 ms EMA, 80 ms body): at 4.4 s on seed 0 the wind was 21.1 deg right, outside
    the null, and the decoder still turned left at 14.5 deg/s.
15. Seeds 0 and 2, B200 seeds 0-2: "From 4.5 s to contact" -> "From 4.6 s to contact". The quoted means (4.2 / 5.4 /
    4.5 / 5.1 / 5.1 deg; 14.6 / 15.4 / 15.7 / 14.9 / 13.5 deg/s) are the samples after 4.5 s; with the 4.5 s sample they
    are 4.4 / 5.5 / 4.7 / 5.3 / 5.3 deg.
16. Seed 0: "The wind at the fly rose from 0.38 to 0.63 m/s" -> "from 0.39" (0.389 m/s at 4.5 s, the minimum after it).
17. B200 seed 0 and the replicate section: "about 1.8 cm nearer the apple (16.84 cm in the 6.0 s sample, 15.09 cm at
    6.1 s)" -> "about 1.9 cm nearer (17.01 cm at takeoff, 15.09 cm at the logged landing point)"; the 6.0 s sample is
    already airborne.
18. "On and off": "The arms differ from the first 10 Hz sample (0.1 s)" -> "from the first or second 10 Hz sample
    (0.1 s; 0.2 s on seed 1)"; seed 1's two arms are identical at 0.1 s. Added that the holder runs the same law in
    both arms and its placement follows the fly's pose.
19. Limits: MN9 "averages about 6-7 Hz within reach" -> "averaged 5.8-6.8 Hz within reach over 8-11 s (five runs; RTX
    4090 seed 1 was within reach for only 0.09 s)"; B200 seed 1 was 5.79 Hz.
20. Clips: "Each command ran exactly once" -> "Each decoder-on command ran exactly once; each control command completed
    once, on the B200, after a first attempt on the RTX 4090 was stopped" (the caption's own next paragraph).
21. Dev section: added a provenance line (only the seed-100 30 s runs have run logs; the seed 101-111 probe numbers
    were not kept as logs); "a run on an even earlier revision touched at 20.79 s" -> adds "(no run log kept)"; the dev
    pair's "Of 102 fan-on 10 Hz samples ..., 37" -> "Of 101" (`dev.json`: 101 samples after switch-on and before
    contact, 37 of them away; 102 and 38 if the switch-on and contact samples are included).

Checked and left as written: every summary and event number in both tables and the per-seed text (contact times,
paths, GF maxima, turn events, MN9 latencies and means, heading errors, turned-away counts, control drifts, DN means,
284 of 285 capped samples, final distances and the HUD's 36.3 / 37.3 / 36.3 cm), provenance (start times, walls,
torch / Python, commit, source hashes, "6 job(s), 0 failed", the stopped controls at 28 / 16 / 22 s and their
missing moov atoms), and these frames: seed 0 CONTACT at 20.50 s, FEEDING: MN9 5.7 Hz first on screen at 21.68 s
(logged 21.47 s; CONTACT holds its 1.2 s), DECODER: TURN LEFT from 2.72 s, "wind on the other side" at 4.20 s; seed 1
GIANT FIBRE 34 Hz -> JUMP at 21.44 s, an all-apple eye mosaic at 21.52 s, the fly out of the main view at 21.80 s and
back at its lower edge at 23.16-23.50 s, DECODER: TURN RIGHT at 22.62 s, CONTACT with MN9 7.5 Hz in the last frame;
B200 WING MNs 59 Hz -> HOP at 6.00 s and FEEDING 8.5 / 6.0 / 8.6 Hz; the controls' "OFF: control" and "not applied
(control)".
