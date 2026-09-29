# spacecraft: declared decoders read a fly connectome's motion cells to steady a tumbling spaceship

**A fly connectome in a tumbling spaceship. One declared decoder reads its H2 / HS and VST2 cells and fires the roll
and yaw thrusters; two more read LPLC2 / LC4 (a dodge, which never fired) and the giant fibre (an escape burn). On one
NVIDIA B200, seeds 0, 1 and 2 began tumbling at 60-67 deg/s. Over 2-30 s they turned at a mean of 15.2, 21.5 and
29.8 deg/s with the decoders on, and at 58.3, 54.0 and 65.5 deg/s with the thrusters off. The escape burn fired mostly
while the ship was tumbling, and it pushes along a fixed body direction, aimed at nothing. Its burns moved the ship off
the paths the rocks were aimed at, so some rocks aimed to hit missed and some aimed to pass hit: on the B200 the
decoders-on runs took 8 hits in 15 rocks, the controls 9 in 15.**

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the fly) | MaleCNS v1.0 as shipped: 167,106 neurons, preset `raw`, no instruments, nothing trained or tuned for this game. Its only input is vision: the radiance along its 1,466 ommatidial columns, handed to `fb.vision` every 10 ms. Everything between that light and the rates of H2, HSN, HSE, VST2, LPLC2, LC4 and the giant fibre DNp01 is the model. That includes the LC4 / LPLC2 -> DNp01 x3 gain that the README's ledger lists (mechanism, magnitude a stop-gap). The HUD's CONNECTOME numbers are those rates. |
| **DECODER** (written for this game, declared) | Three read-only modules, attached with `kind="decoder"`. They write nothing into the brain. **Attitude hold:** yaw torque from H2 / HSN / HSE left minus right, roll torque from VST2 left minus right. Both are side-equalised and low-passed 0.5 s, with gains of 4 (yaw) and 3 (roll) deg/s^2 per Hz. The weights were fitted and the gains tuned on dev seeds 100-105. **Dodge:** LPLC2 + LC4 left minus right >= 5 Hz. **Escape burn:** mean DNp01 >= 33 Hz, the body model's takeoff line. Pitch has no decoder. |
| **GAME** (not the fly) | The ship, its rigid-body physics and thruster sizes, and the refractories. The cruise autopilot. The scene (planet, sun, moon, nebulae, station, rocks) and where the fly's eye sits in it. The seed-drawn tumble and rock paths, the collisions and knocks, and the torque-free reference. The control arm, the chase camera and the banners. |
| **not claimed** | The fly does not fly, steer or turn on its own. No motor neuron and no descending turning neuron is read (the README: DNa02 is held below threshold). The attitude hold is these decoders' mapping of the lobula plate, fitted for this scene. It is not a general readout of self-rotation, and it holds no pitch. The dodge never fired in the nine recorded runs; the seed-0 control would have fired it once, with the nearest rock 36 m away. The escape burn fires mostly from the tumble (measured 5 below). It can change which rocks hit, in both directions, because each rock is aimed at where the autopilot alone would take the ship, and a burn after the rock's launch moves the ship off that path (Clips). That is the game's aiming rule meeting a burn, not avoidance. Of the 24 burns fired in the six decoders-on runs, none fired with the rock that then ended the other way closer than 10 m, and the one that followed a rock's pass within 1 m came 0.14 s after that pass. |

## Design

The shipped fly (MaleCNS v1.0, `preset="raw"`, no instruments) sits in the cockpit of a rigid spaceship with six
degrees of freedom. The ship starts tumbling at 55-70 deg/s in low orbit, beside a rotating ring station, while five
rocks are sent at it. The fly's 1,466 ommatidial columns look out along the ship's axes. Three read-only decoders turn
named populations into thruster commands:
- **attitude hold:** horizontal-system cells and H2 for yaw, VST2 for roll. Pitch is not built.
- **dodge:** LPLC2 + LC4, left minus right.
- **escape burn:** the giant fibre, at the body model's 33 Hz takeoff line.

Everything else is GAME. The comparison is the **control**: the same seed and scenario, with the same decoders attached
and read, and their outputs not applied (thrusters off).

**The seed draws the scenario.** The initial tumble and each rock's direction, aim and shape come from the seed, on an
RNG stream of their own. The raw brain here has no Poisson input, so the seed alone would change nothing but GPU
floating-point noise (measured 11). The decoder gains were tuned on dev seeds 100-105. Seeds 0, 1 and 2 are therefore
tumbles and rock passes the gains never saw.

What the ship does here is what these decoders make of the fly's lobula-plate, loom and giant-fibre cells. The fly does
not fly, turn or steer on its own (README, "where it stops": DNa02 is held below threshold), and nothing here reads
DNa02 or a motor neuron.

| what | kind | law and parameters |
|---|---|---|
| the brain | CONNECTOME | `FlyBrain(seed=...)` via `games.common.build_brain`, preset `raw`. Three `games.common.ReadDecoder`s are attached (`kind="decoder"`). They write nothing, so the brain is numerically unaffected, and they are listed in the run log's `modules`. The only sensory input is vision: no odour, wind or taste, so the raw brain has no Poisson drive. |
| the fly's eyes | GAME (scene) | The eye is at body (2.18, 0, 0.82) m in the canopy, looking along the ship's axes (x forward, y left, z up). The hull is not in its view. Its 1,466 x 7 rays are traced every 10 ms on the CPU and pooled with `games.common.Eyes`. The sky at infinity is sampled from a 2048 x 1024 texture of the camera's own sky function. Rocks (their lobes) and the station are intersected in 3D with plain Lambert shading. The camera adds surface bump detail that the 4.5 deg ommatidia would average away. The scene is RGB, so UV = 0.5 B. |
| scene | GAME | **At infinity, so a pure rotation gives a purely rotational image flow:** a day-lit planet fills everything more than 24 deg below the horizon (its angular radius is 66 deg). The sun is 6 deg above the horizon ahead, a moon is behind, and there are stars and **nebulae over the whole sky** (radiance about 0.05-0.3, the planet's day side 0.3-0.6). The nebulae are there for the fly (measured 1). **Local:** a ring station (R 45 m, spinning at 0.1 rad/s) 420 m away, 7 large scenery rocks 140-320 m away, and the threats. The chase camera lights local objects with a dimmer sun than the eye scene (0.62 vs 2.2) so the sunlit hull does not burn out. That is art only: the fly's view uses 2.2. |
| ship | GAME | A rigid body with Euler's equations: principal moments (roll, pitch, yaw) = (1.0, 1.7, 2.0) normalised, RK4, quaternion attitude. Commanded angular accelerations are clipped at 90 deg/s^2 per axis and applied as torque = I x alpha. |
| scenario | GAME | `default_rng([0x5CAF, seed])`. **Initial tumble:** 55-70 deg/s about an axis made by adding a normal of sd 0.35 per component to the unit vector of (roll, pitch, yaw) = (40, 18, -45) and normalising. **Rocks:** five, with fixed arrival times, speeds and sizes: 9.0 s, 24 m/s, 2.6 m, hit; 14.0 s, 26 m/s, 2.8 m, close; 19.0 s, 24 m/s, 3.0 m, hit; 24.0 s, 22 m/s, 2.6 m, close; 28.5 s, 28 m/s, 2.8 m, hit. The seed draws each one's direction: from beyond the ship, near the chase camera's line of sight at arrival (4 deg right to 6 deg left of it, or 12-26 deg to its right; 8 deg below to 4 deg above). It also draws the aim offset (hit: the rock's centre passes 0.8-2.2 m from the ship's predicted centre; close: 5.5-7.5 m, and the wings can still catch it) and the lumpy shape. |
| **yaw hold** | DECODER | `yaw accel = -G_yaw * s_yaw`, G_yaw = 4 deg/s^2 per Hz. `s_yaw = sum_t w_t (L_t - R_t) - beta_yaw sum_t abs(w_t) (L_t + R_t)`, where L_t / R_t are the population mean rates (Hz) on each side. w = H2 1.0, HSN -0.47, HSE -1.15 (each 1 cell per side) and beta_yaw = -0.20, low-passed with tau 0.5 s. |
| **roll hold** | DECODER | `roll accel = -G_roll * s_roll`, **G_roll = 3** deg/s^2 per Hz (retuned from 5 on dev seeds 100-105, measured 8). `s_roll = (VST2_L - VST2_R) - beta_roll (VST2_L + VST2_R)`, VST2 3 cells left and 4 right, beta_roll = -0.31, low-passed with tau 0.5 s. |
| pitch | not built | No lobula-plate population carried it (measured 3). The HUD says so. |
| side equalisation (in both holds) | DECODER | beta = mean(L - R) / mean(L + R) over the dev tumbles. During rotation the right-side cells fire more than the left, so a plain -G (L - R) is a steady torque that spins the ship up (measured 4). Beta cancels that average bias. It does **not** make a still ship's resting activity torque-free (measured 9). |
| **dodge** | DECODER | `side = mean(LPLC2_L, LC4_L) - mean(LPLC2_R, LC4_R)` (population mean rates; LPLC2 94 left / 91 right, LC4 71 / 55), low-passed 50 ms. `abs(side) >= 5 Hz` fires. The thruster (GAME) adds 4 m/s along body -y if the loom is left and +y if it is right, with a 1.0 s refractory. |
| **escape burn** | DECODER | Giant fibre DNp01 (1 cell per side), the mean of both cells (the body model's `motor.gf`), `>= 33 Hz` (`flyverse/body.py` `Flight.gf_hz`). The thruster (GAME) gives a 6 m/s burn over 0.4 s along body (forward + up)/sqrt(2), with a 1.5 s refractory. That refractory is the game's; the body model's own landing refractory is 1.0 s. |
| cruise | GAME | A translation-only autopilot: world velocity relaxes toward (2, 0, 0) m/s with tau 2.5 s, capped at 1.5 m/s^2. It never touches attitude. |
| rocks | GAME | A rock is the union of 4 ellipsoids: a core and three lobes. Each is launched 5 s before its arrival time at where the autopilot alone would take the ship, plus its aim offset (sideways, on the far side from the camera). **Contact** is when one of 1,779 points on the hull, wings, pods, fin and canopy enters the rock. It is resolved as a momentum-conserving bounce along the contact normal (rock 1.5x the ship's mass, restitution 0.4). The ship also gets an angular kick of min(0.45 rad/s, 0.016 x impulse) x min(1, lever / 1.5 m) about the blow's lever axis, scaled per body axis by mean inertia / axis inertia. A blow through the centre barely spins it. A miss is logged with its closest surface clearance. Each impact or miss logs the peak LPLC2+LC4 rate per side and the peak GF while the rock was within 30 m. |
| torque-free reference | GAME | A copy of the ship's rotation integrated with no torque and no impacts, from the same start. It is the dashed line in the TUMBLE panel, and the one-shot ATTITUDE HOLD banner quotes it. |
| control | GAME | `--control`: identical, except the decoder outputs are not applied (no torque, no dodge, no burn). The HUD shows what they would have done, greyed: "BURN NOT APPLIED" / "DODGE NOT APPLIED" banners, grey thruster lamps, "not applied" under the thrust bars, an amber frame round the view and the tag CONTROL · THRUSTERS OFF. |
| still | GAME | `--still`: the ship starts at rest and no rocks come. It is a check of what the decoders do to a still ship (measured 9). |
| HUD | | **CONNECTOME:** the per-type rates the attitude decoder reads (H2 / HSN / HSE left minus right, VST2 left and right; 0.5 s running means for display), LPLC2+LC4 left and right (0.1 s means) and the giant fibre (per 10 ms tick, with its 33 Hz line). **WHAT THE FLY SEES** is the exact radiance handed to `fb.vision`. **DECODER:** the decoded signals s_yaw / s_roll (teal traces and values: weighted, side-equalised, low-passed), the thrust bars and the thruster lamps. The footer prints both hold laws in full. **GAME:** the ship's angular speed and per-axis rates (the physics), the torque-free line, the true rate overlaid on each decoder trace (amber, ±80 deg/s full scale) and the rock tally. |

### What was measured first (dev seeds >= 100 only)

**1. The scene has to be visible to the fly's photoreceptors.** They code temporal contrast, (I - mean) / (mean + 0.02),
per column. A black starfield (about 0.001) is far below that floor.

The first screen used a sparse sky: stars, a faint Milky Way and a limb sunrise. It ran seed 100 with an interleaved
rotation protocol: +-60 deg/s about each axis, 1.2 s on, 3 repeats each way. Responses were dominated by where the sun
and the planet's limb happened to sweep:
- HS L-R did not flip canonically with yaw: the (yaw left) - (yaw right) difference was HSE +2.65 +- 0.80 Hz and HSN
  +0.57 +- 0.91, where front-to-back-preferring HS cells should give a negative value.
- VS L-R flipped more with pitch (-36.9 Hz) than with roll (-13.7).

The scene was therefore rebuilt: a day-lit planet below and nebulae across the sky. Single-axis screens in the new sky
were still scene-dependent (HSE +1.5 +- 1.2 Hz, HSN -1.4 +- 2.4), so the decoders were built from random tumbles
instead (2).

The lobula-plate cells used here have low resting rates (0-2 Hz, not always 0: measured 9). They report motion mostly
by firing more, not by dropping below a baseline, so a single cell cannot report its null direction well. That is why
the decoders use left minus right, and why pitch (symmetric between the eyes) fails.

**2. Random-tumble datasets.** Two runs of 45 s each: random body angular-velocity targets of 20-100 deg/s about random
axes, 20 % rest, 0.15 s ramps, changes every 0.8-2 s, from a random start attitude. Brain seeds 100 and 101 with two
different tumble schedules; the analysis ran on both datasets and was repeated in the final scene.
- **Yaw:** a least-squares fit, with no intercept, of the body yaw rate on HSN / HSE / HSS / H2 L-R. It gave the same
  signs on both tumbles (HSN -1.0 / -2.0, HSE -3.0 / -4.5, H2 +3.1 / +3.8 deg/s per Hz; HSS about 0, dropped). That is
  canonical: H2 prefers back-to-front motion and HS front-to-back.
  - Weights fitted on one tumble predicted the other at r = 0.41-0.70.
  - The shipped signal (weights scaled to H2, beta, 0.5 s low-pass) correlates with the true yaw rate at r = 0.33-0.41
    and 0.59-0.65 on the two tumbles, at 200-400 ms neural lag.
  - Its slope is 0.035-0.087 Hz per deg/s.
- **Roll:** VST2 L-R was the one vertical-system population whose weight kept its sign on both tumbles; VS, VSm and
  VST1 flipped sign. The shipped signal gives r = 0.49-0.57 and 0.35-0.38, with a slope of 0.065-0.100 Hz per deg/s.
- The mean HS / VST2 firing rates are low: HSN 1.5-1.8 Hz, HSE 0.3-0.5 Hz, H2 1.0-1.2 Hz, VST2 6-7 Hz.

**3. Pitch.** No pooled population tracked it reliably.
- The best held-out fit of the VS family gave r <= 0.31, with a prediction slope of 0.09.
- The best single population reached |r| 0.26 on one tumble and 0.51 (VST1 L+R) on the other.
- A left-plus-right sum rises for any rotation, so it could only ever report one direction.

Not built.

**4. Side bias.** VST2 L - R averaged -6.05 / -5.66 Hz while tumbling, correlated with L + R at -0.58 / -0.65.

The first closed loop, with G 25 / 20 deg/s^2 per Hz and no beta (seed 102), spun the roll up to 525 deg/s. The
constant bias turned into constant torque.

With beta = mean(L-R) / mean(L+R) = -0.31 (the final scene gave -0.33 / -0.29) and -0.20 for yaw (-0.26 / -0.15), the
loop damps.

**5. The giant fibre fires from rotation.** While tumbling on the builder's tumbles, the mean DNp01 rate was >= 33 Hz in
7.2 % / 9.7 % of 10 ms ticks. That is 11-14 % above 40 deg/s and 0.0-0.3 % below 15 deg/s. Most of it comes from the
right cell (DNp01_R mean 18.1-18.7 Hz, DNp01_L 10.3-11.0).

Removing the sun did not change this: DNp01_R was above 33 Hz 43 % of the time during 60 deg/s rotations with and
without it. What drives it is the rotating scene; only the sun was ablated.

In the final code's dev runs (six decoders-on runs at the final gains and two controls, seeds 100-105), the rate of
ticks with GF >= 33 Hz rose with the tumble: 0.1 % below 15 deg/s, 2.6 % at 15-40, 15.1 % at 40-80 and 63 % above 80.
Of the 60 escape burns (fired or, in the controls, would-be) in those runs, 34 came with no rock within 50 m.

So the "escape burn" fires during the tumble. With the game's 1.5 s refractory, a 60 deg/s tumble triggers one about
every 1.5-2 s.

**6. Rocks do not reach the loom pathway before contact.** The builder tested approaches (single-ellipsoid rocks) from
the left, right, front, above and below:
- a sunlit 3 m rock at 25 m/s;
- a dark (albedo 0.04) 2.4 m rock at 60 m/s (r/v 40 ms, like the shipped 30 ms black-ball loom);
- a dark 12 m rock at 10 m/s.

All gave LPLC2 <= 3.6 Hz, LC4 <= 2.6 Hz and mean GF <= 15 Hz until the rock touched.

A dark rock held 0.3 m from the eye for 0.3 s, as the shipped loom probe holds its ball, did lateralise: LPLC2 on the
loom side reached 4.9 / 7.5 Hz, and GF_L 26.7 Hz (loom left) / GF_R 52 Hz (loom right). That comes only at contact.

Tumbling alone drives the LPLC2 + LC4 side signal to a p99 of 4.3-5.9 Hz, the same size. The dodge threshold of 5 Hz
therefore sits at about the tumble's 99th percentile, and a dodge, when one fires, is not evidence of a rock.

The final code's dev clips (seed 106, lumpy rocks) log the peak rates while each rock was within 30 m
(`approach_peak`):
- LPLC2+LC4 peaked at 0.4-3.6 Hz per side with the decoders on, and 2.1-4.7 Hz in the control.
- No dodge fired near a rock in either run. The control's two would-be dodges came with the nearest rock 44 and 50 m
  away.
- The mean giant fibre peaked at 10.7-37.7 Hz with the decoders on and 23.6-60.0 Hz in the control, whose tumble never
  slowed. It followed the tumble, not the rock.

**7. First closed-loop tuning** (the builder's, on one fixed tumble (40, 18, -45), seeds 102-105, 20-30 s; mean |w| =
mean angular speed; the control keeps 60-63 deg/s):

| gains G_roll / G_yaw, low-pass | result |
|---|---|
| 10 / 8, 0.2 s | yaw runaway to 470 deg/s |
| 5 / 4, 0.2 s | about 22 deg/s after 6 s, with pitch (not held) about -19 |
| 10 / 8, 0.5 s | 13 deg/s on 102; excursions to 72 deg/s on 103 |
| 8 / 6, 0.5 s | oscillates, 57 deg/s at 10 s |
| 5 / 4, 0.5 s | 67 -> 10-15 deg/s by 6-18 s on every seed tried (102-105) |

A 4 s high-pass on the signals (tested to remove slow bias drift) slowed the first damping and was left off.

**8. Retuning on seed-drawn tumbles** (dev seeds 100-105, final scenario code, 30 s, no camera). Once the seed drew the
tumble and the rocks, 5 / 4 still damped two tumbles but spun two others up **in roll**: seed 100 reached 108 deg/s
near 19 s and seed 101 125 deg/s near 28 s, against 62-77 deg/s in their controls. The roll readout has the right sign
on average but is weak and noisy. Across these runs the decoded s_roll moved by only about 1.5-4 Hz over 50-100 deg/s
of roll, against swings of +-6-9 Hz, so at G_roll 5 the noise drives a random walk in roll, the lightest axis.

Mean |w| over 2-30 s (deg/s):

| seed | 5 / 4 (first tuning) | 5 / 4, signal clipped at +-3 Hz | **3 / 4 (shipped)** | 2 / 4 | control |
|---|---|---|---|---|---|
| 100 | 53.5 (max 108) | 52.2 | 42.8 (max 86) | 27.5 | 69.0 |
| 101 | 53.8 (max 125) | 42.8 | 34.0 (max 70) | 39.0 | 70.2 |
| 102 | 20.5 | | 15.4 | | |
| 103 | 35.9 | | 23.2 | | |
| 104 | | 29.7 | 24.2 | 22.3 | |
| 105 | | | 21.0 | | |

G_roll 3 was better than 5 on every seed where both ran, and it averaged 26.8 deg/s over the six. G_roll 2 was mixed
(better on 100, worse on 101), and the clip mostly trimmed peaks. The shipped gains are 3 / 4 with the 0.5 s low-pass
and no clip (`--att-clip` remains as an option, off).

**9. A still ship.** The builder's claim that these cells sit at 0 Hz in a static scene holds only for the first
half-second. With the ship at rest (reviewer probe, seed 105, first gains 5 / 4, no rocks), the first 0.5 s was 0 Hz.
From 1-2 s, HSN_R fired at 1.6-2.0 Hz with HSN_L at 0, HSE_R reached 1.7 with HSE_L at 0, H2_L 1.9 with H2_R at 0, and
VST2 1.3-2.3 Hz on both sides. The decoded yaw signal at rest was +2.5 Hz, about -10 deg/s^2 of command. The side
equalisation adds to it: at rest, +0.31 (L + R) is not balanced by any rotation.

The final code, dev seed 107, gains 3 / 4, `--still` (`out/games/spacecraft/dev_still.mp4` and
`dev_still_control.mp4`, 15 s each):
- With the decoders on, the still ship was spun up to 6.8 deg/s at 4-5 s, 7.1 at 9-10 s and 13.6 at 14-15 s. The
  0.5 s samples peaked at 16.3 deg/s at 12.0 s, mostly roll.
- The control stayed at 0.0 deg/s for all 15 s.
- The giant fibre never reached 33 Hz in either.

**10. Dev clips, final code** (`out/games/spacecraft/dev.mp4`, `dev_control.mp4`, seed 106: not used in any tuning).

| window (s) | 0-1 | 4-5 | 9-10 | 14-15 | 19-20 | 24-25 | last 2 |
|---|---|---|---|---|---|---|---|
| decoders connected | 68.7 | 20.1 | 22.1 | 18.8 | 28.7 | 28.9 | 22.7 |
| control | 66.1 | 65.4 | 66.2 | 68.4 | 54.5 | 59.9 | 62.1 |
| torque-free reference (both runs) | 66.1 | 65.4 | 63.9 | 65.6 | 65.0 | 63.9 | 65.6 |

Mean |w| in the windows above (deg/s). Over 2-30 s (0.5 s samples), the decoders gave 24.9 deg/s and the control 63.1.
The seed drew a 65.7 deg/s tumble, (roll, pitch, yaw) = (+34.2, +39.8, -39.6).

- **Decoders connected:**
  - The tumble dropped below 15 deg/s at 5.64 s (the ATTITUDE HOLD banner).
  - Rock 1 hit the left wing at 8.82 s (12.3 -> 19.8 deg/s). Rock 3 hit the right wingtip at 18.60 s (19.6 -> 32.7).
    Rock 5 hit the belly at 28.35 s (21.3 -> 21.0). Rocks 2 and 4 missed by 2.15 and 11.44 m.
  - Over the last 5 s the per-axis rms was roll 12.0, pitch 18.7 and yaw 8.6 deg/s. Pitch, which is not held,
    dominates what is left.
  - The escape burn fired 5 times: at 0.17 s and 2.46 s, before any rock was launched; at 16.86 s (rock 3 43.5 m
    away); at 19.53 s (rock 4 105 m away); and at 21.79 s (rock 4 45.5 m away). None fired with a rock within 30 m. No
    dodge fired.
- **Control:**
  - The tumble stayed at 53.8-69.3 deg/s in the 0.5 s samples for all 30 s (window means 54.5-68.4). The torque-free
    reference ran at 63.9-67.0, and the difference is the knocks.
  - Rock 1 hit the left wing (8.86 s), rock 3 the belly (18.82 s) and rock 4 the right wingtip (23.93 s). Rock 2
    missed by 2.09 m.
  - Rock 5 was still unresolved at 30 s (nearest 23.3 m at 29.72 s). The rock-4 impact had pushed the ship off the
    path it was aimed at.
  - The burn would have fired 15 times: 12 with no rock within 50 m and 3 with a rock 11.6-25.7 m away. The dodge would
    have fired twice, with the nearest rock 50 and 44 m away.

These dev clips were recorded from the code one revision before the frozen file: their logs carry
`games/spacecraft.py` `6e34970afe63f18e`, the recordings `793eddf416eaa6bb`. The builder reports the only change since as
a draw-only fix that keeps the "torque-free" label off the TUMBLE panel's axis label; the earlier file is not kept, so
that was not re-checked.

The chase camera's hull no longer burns out. In brain-free renders of the first seconds (seeds 100, 106 and 108 at
0.5, 2 and 4 s), the hull's median grey level was 141-190 and its 95th percentile 193-216, against 241 and 254 before.
The 25th-75th percentile spread was 30-115 levels, 4 of 9 frames at 40 or more. The narrow ones are front-lit
(sun behind the camera) smooth hulls. A dimmer camera sun or stronger plate variation darkened them without widening
the spread, so both were left as they are.

**11. The seed and the brain.** With no Poisson input, the raw brain's only randomness is GPU floating-point
nondeterminism. With the builder's fixed scenario, runs on seeds 102-105 were identical for the first seconds (48.1
deg/s at 2 s, 26.5 at 4 s) and diverged later. That is why the scenario is drawn from the seed.

### Limits

- **Pitch is not held.**
  - A pure pitch tumble cannot be removed by roll and yaw torques: its angular momentum only moves into roll and yaw
    through the gyroscopic coupling. A pitch-heavy tumble or knock can therefore stay at 10-30 deg/s.
  - The ship's pitch axis is the intermediate one, where rotation is unstable, which helps a little.
- **The holds are noisy.** In the six recorded decoders-on runs they took a 60-67 deg/s tumble to a mean of 15-30 deg/s
  over 2-30 s (B200 15.2-29.8, RTX 4090 17.5-28.9), with window means of 5-50 deg/s after 2 s, not to zero. That floor
  is not leftover tumble alone: rock knocks raise it (b200_seed2 went back up to 51.2 deg/s after rock 2's hit), and
  the loop makes motion of its own: a still ship driven by the decoders goes to 7-16 deg/s within 15 s (measured 9).
  - The decoded signals follow the true rate at r 0.35-0.65, about 300 ms late.
  - The loop can spin the ship up for a while. In every recorded decoders-on run it did so in the first second: at
    1.0 s the B200 runs turned at 65.4, 67.1 and 74.3 deg/s, against 60.7, 65.6 and 67.3 with the thrusters off. At
    the shipped gains the dev maximum was 86 deg/s (seed 100), against a torque-free 62-64 deg/s. Higher gains
    oscillated or ran away (measured 7, 8).
  - The side-equalisation beta is a measured average and can drift with the scene.
- **The dodge cannot avoid a rock.** The loom pathway does not see one before contact in this scene (measured 6), and
  tumbling produces a side signal of the same size. Any dodge that fires is the tumble's.
- **The escape burn is mostly the tumble's.** The rotating scene drives the giant fibre (mostly the right cell) over
  the body's 33 Hz line. When a rock is near at the time, that is mostly coincidence (measured 5).
- The decoder weights were fitted on dev seeds 100/101, in this scene, for this game. They are not a general readout
  of the fly's self-rotation.
- The eye sees rocks and the station flat-shaded. The camera lights local objects with a dimmer sun than the eye's.
- There is no orbital mechanics, and the sun, planet and moon are fixed.
- UV is 0.5 x blue.

### Commands

    python games/spacecraft.py --seed {seed} --seconds 30 --record out/games/spacecraft/seed{seed}.mp4
    python games/spacecraft.py --seed {seed} --seconds 30 --control --record out/games/spacecraft/control_seed{seed}.mp4
    python games/spacecraft.py --seed {seed} --seconds 15 --still --record out/games/spacecraft/still_seed{seed}.mp4   # optional

Each run writes its log next to the clip (`.json`). On the shared RTX 4090 the 30 s dev clips took 40-42 min of wall
time while about 20 other processes used the GPU (the builder saw 5-8 min on a quieter one). On the cluster, six
30 s recordings running at once on `<cluster-node>`'s NVIDIA B200s took 297.5-302.9 s each. The cluster form of the recording
commands is under Clips.

## Clips

Nine 30 s recordings, all from one frozen `games/spacecraft.py` (sha256 `793eddf416eaa6bb`, first 16 hex, in every
log): the three decoders-on seeds on this desktop's RTX 4090, then the three controls and a replicate of each
decoders-on seed, all six in one batch on the cluster's NVIDIA B200s (`<cluster-node>`). The comparison between the arms is made
**on the B200 only**, where both arms ran on one GPU model in one batch. The 4090's `seed0.mp4` is the trailer's main
source (its split screen pairs the two B200 arms), and the 4090 clips are reported in full below. No 4090
control exists (next paragraph), so no 4090 clip is compared with a control. The B200 replicates are not reruns for a
better outcome: all nine clips are kept and reported, and the trailer keeps the 4090 ones.

**The interrupted desktop controls.** The controls were first started on the 4090 with the frozen command. None of
them finished. The lead stopped them to free the desktop GPU, and no run log was written for any of them:
* seed 0 twice, both times between 12 and 13 s of brain time;
* seed 1 once, after 9.58 s;
* seed 2 never started.

The partial files are kept in `out/games/spacecraft/killed_partial/` with their console output:
`control_seed0_attempt2.mp4` has no moov atom and does not play, and `control_seed1_attempt1.mp4` plays 479 frames
(9.58 s). None of them is used here. The controls were then recorded on the cluster (below), with the same arguments.

### Provenance

**RTX 4090** (`seed0.mp4`, `seed1.mp4`, `seed2.mp4`; started together at 22:33 UTC on 2026-09-28). Each ran once, from
the repo root, as `PYTHONIOENCODING=utf-8 <miniconda>/python.exe` followed by the arguments in the table.
The logs record:
* `cuda`, NVIDIA GeForce RTX 4090, torch 2.10.0+cu128, Python 3.13.2;
* commit `959f2e927b1c2806bc526fd5fc51cb3f5cc3e4b1` with `dirty: true` (games/ is not committed);
* `games/spacecraft.py` `793eddf416eaa6bb` and `games/common.py` `b110142562a927a1`.

The three shared the GPU with other agents' recordings, hence 53-67 min of wall time each.

**NVIDIA B200** (the six other clips). One `scripts/cluster_run.py` call on the house cluster, `<scheduler>` run
`games-spacecraft-rec-443f80`. All six jobs were sent to `<cluster-node>` (`--node <cluster-node>`; no `--gpu-ids` pin), and they ran at
once from 00:37 UTC on 2026-09-29. The saved console output and logs name the GPU model, not which of `<cluster-node>`'s GPUs ran
each job. The console log ends `6 job(s), 0 failed  (7.3 min)`. The logs record:
* `cuda`, NVIDIA B200, torch 2.11.0+cu128, Python 3.12.3;
* `commit: null` and `dirty: null`: the run directory is a copy of the tree, not a git checkout, so the code is
  identified by its source hashes;
* `games/spacecraft.py` `793eddf416eaa6bb`, the same file as the 4090 runs;
* `games/common.py` `e06078fb69d1275a`. This differs from the 4090 logs' `b110142562a927a1` by one change only: the
  recorder now finds ffmpeg through `find_ffmpeg()` (ffmpeg on PATH, else imageio-ffmpeg's bundled binary), because the
  cluster has no ffmpeg on PATH. This was checked byte for byte. Deleting `find_ffmpeg()` from the current file and
  restoring the recorder's `ffmpeg = shutil.which("ffmpeg")` and its old error message ("--record needs ffmpeg on PATH")
  reproduces `b110142562a927a1` exactly. Nothing that touches the brain, the game or the drawing changed.

The command, run from the repo root on this desktop (`PYTHONIOENCODING=utf-8 <miniconda>/python.exe`):

    python scripts/cluster_run.py --name games-spacecraft-rec --target house --node <cluster-node> --minutes 90 \
        --fetch=out/games/spacecraft/ "<job 0>" "<job 1>" ... "<job 5>"

Jobs 0-5 are, in order, `control_seed0`, `b200_seed0`, `control_seed1`, `b200_seed1`, `control_seed2` and
`b200_seed2`. Each runs this command, with seed k = 0, 1 or 2 and {stem} = `control_seed{k}` (adding `--control`) or
`b200_seed{k}`:

    python -c 'import torch; assert torch.cuda.is_available()' && source .venv/bin/activate && mkdir -p out/games/spacecraft
      && python games/spacecraft.py --seed {k} --seconds 30 [--control] --record out/games/spacecraft/{stem}.mp4
      > out/games/spacecraft/{stem}.console.txt 2>&1 && tail -n 4 out/games/spacecraft/{stem}.console.txt

`cluster_run.py` puts `source .venv/bin/activate && ` in front of each job. The six job strings are saved verbatim in
`out/games/spacecraft/cluster_run.commands.txt`, and the whole console in `cluster_run.console.txt`.

In every log: MaleCNS v1.0, 167,106 neurons, preset `raw`, no instruments, and three attached modules of kind
`decoder` (`attitude_hold`, `dodge`, `escape_burn`). The controls attach and read them too, and apply nothing. Every
clip is 30.0 s of brain time: 1,500 frames at 1920x1080 and 50 fps. All nine are kept.

| clip | GPU | command arguments (after `python`) | started (UTC) | brain / wall |
|---|---|---|---|---|
| `seed0.mp4` | 4090 | `games/spacecraft.py --seed 0 --seconds 30 --record out/games/spacecraft/seed0.mp4` | 2026-09-28 22:33:33 | 30.0 / 3184.9 s |
| `seed1.mp4` | 4090 | `games/spacecraft.py --seed 1 --seconds 30 --record out/games/spacecraft/seed1.mp4` | 2026-09-28 22:33:34 | 30.0 / 3446.8 s |
| `seed2.mp4` | 4090 | `games/spacecraft.py --seed 2 --seconds 30 --record out/games/spacecraft/seed2.mp4` | 2026-09-28 22:33:34 | 30.0 / 4011.4 s |
| `b200_seed0.mp4` | B200 | `games/spacecraft.py --seed 0 --seconds 30 --record out/games/spacecraft/b200_seed0.mp4` | 2026-09-29 00:37:34 | 30.0 / 298.6 s |
| `control_seed0.mp4` | B200 | `games/spacecraft.py --seed 0 --seconds 30 --control --record out/games/spacecraft/control_seed0.mp4` | 2026-09-29 00:37:34 | 30.0 / 302.9 s |
| `b200_seed1.mp4` | B200 | `games/spacecraft.py --seed 1 --seconds 30 --record out/games/spacecraft/b200_seed1.mp4` | 2026-09-29 00:37:35 | 30.0 / 297.5 s |
| `control_seed1.mp4` | B200 | `games/spacecraft.py --seed 1 --seconds 30 --control --record out/games/spacecraft/control_seed1.mp4` | 2026-09-29 00:37:35 | 30.0 / 302.3 s |
| `b200_seed2.mp4` | B200 | `games/spacecraft.py --seed 2 --seconds 30 --record out/games/spacecraft/b200_seed2.mp4` | 2026-09-29 00:37:35 | 30.0 / 302.5 s |
| `control_seed2.mp4` | B200 | `games/spacecraft.py --seed 2 --seconds 30 --control --record out/games/spacecraft/control_seed2.mp4` | 2026-09-29 00:37:35 | 30.0 / 302.4 s |

**The B200 clips look slightly different.** The cluster has neither of the HUD's first-choice fonts (Cascadia Mono,
Bahnschrift). It resolves `games/common.py`'s font lists to DejaVu Sans Mono and DejaVu Sans Condensed, so the text is
set in a different face. Two labels run past their panel's edge: the end of the attitude legend ("LP 0.5 s") and
the TUMBLE panel's "torque-free" tag near its right edge. The numbers and the scene are unaffected.

### What the run logs say

All speeds are the ship's angular speed |w| in deg/s, and all times are in brain seconds. The 2-30 s means (and their
torque-free means) and the maxima after 5 s are from the log's 0.5 s samples; "2-30 s" is the 56 samples at
2.5-30.0 s, the log's own (a, b] window convention (with the 2.0 s sample the means would be 0.4-0.6 deg/s higher with the decoders on, 0.0-0.2 in the controls). The
window means, the last-5-s rms and "first below 15" are the log's own per-tick summary values. "Torque-free" is the reference copy of the ship that no torque or knock touches. "GF >= 33 Hz" is the share of
10 ms ticks with the mean giant-fibre rate at or above the burn line. A control's burns and dodges are the ones its
decoders would have fired.

| clip | GPU | tumble at t = 0 | mean \|w\|, 2-30 s (torque-free) | first below 15 | 0-1 / 4-5 / 9-10 / 14-15 / 19-20 / 24-25 s / last 2 s | max after 5 s | rms roll / pitch / yaw, last 5 s | GF >= 33 Hz | burns | dodges | impacts / misses |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed0 | 4090 | 60.0 | **17.5** (59.0) | 4.26 | 62.8 / 12.2 / 16.4 / 12.0 / 16.4 / 25.5 / 27.3 | 30.4 at 29.5 s | 14.3 / 12.6 / 15.4 | 2.3 % | 4 | 0 | 2 / 3 |
| seed1 | 4090 | 65.3 | **21.4** (64.0) | 11.1 | 66.4 / 29.3 / 37.9 / 7.6 / 20.7 / 18.9 / 27.9 | 40.0 at 9.0 s | 16.5 / 14.1 / 4.8 | 2.0 % | 3 | 0 | 4 / 1 |
| seed2 | 4090 | 67.0 | **28.9** (66.8) | 25.91 | 70.5 / 26.9 / 31.5 / 32.0 / 41.1 / 18.8 / 27.9 | 43.0 at 18.5 s | 14.8 / 13.3 / 11.8 | 4.3 % | 4 | 0 | 4 / 1 |
| b200_seed0 | B200 | 60.0 | **15.2** (59.0) | 4.34 | 62.8 / 14.0 / 14.1 / 12.2 / 20.4 / 16.3 / 5.2 | 31.9 at 16.5 s | 2.6 / 3.5 / 4.0 | 2.4 % | 4 | 0 | 2 / 3 |
| control_seed0 | B200 | 60.0 | **58.3** (59.0) | never | 60.4 / 59.7 / 58.4 / 60.0 / 59.7 / 58.6 / 47.0 | 61.0 at 16.5 s | 15.9 / 43.8 / 28.7 | 11.9 % | 0 (would have: 11) | 0 (would have: 1) | 3 / 2 |
| b200_seed1 | B200 | 65.3 | **21.5** (64.0) | 20.48 | 66.4 / 28.1 / 28.6 / 25.4 / 20.9 / 7.0 / 19.3 | 30.2 at 17.5 s | 6.0 / 9.1 / 8.9 | 2.0 % | 4 | 0 | 3 / 2 |
| control_seed1 | B200 | 65.3 | **54.0** (64.0) | never | 65.5 / 63.5 / 56.2 / 58.2 / 43.7 / 43.7 / 53.6 | 63.5 at 8.5 s | 22.3 / 20.1 / 37.8 | 9.8 % | 0 (would have: 12) | 0 | 3 / 2 |
| b200_seed2 | B200 | 67.0 | **29.8** (66.8) | 21.78 | 70.5 / 25.4 / 32.4 / 39.9 / 49.7 / 20.3 / 22.2 | 51.2 at 17.0 s | 4.7 / 16.1 / 7.3 | 4.8 % | 5 | 0 | 3 / 2 |
| control_seed2 | B200 | 67.0 | **65.5** (66.8) | never | 67.2 / 66.3 / 66.3 / 66.6 / 61.0 / 61.2 / 75.7 | 78.1 at 30.0 s | 56.2 / 31.8 / 19.0 | 24.4 % | 0 (would have: 14) | 0 | 3 / 2 |

The seed draws the tumble, as (roll, pitch, yaw) deg/s:
* seed 0: (+33.5, +33.2, -37.1);
* seed 1: (+37.8, +19.4, -49.6);
* seed 2: (+59.4, +20.5, -23.2), the most roll-heavy.

### Decoders connected vs control (B200)

**The attitude hold beat the control on every seed over 2-30 s** (not in the first second, below). Over 2-30 s, the
decoders-on ship turned at a mean of 15.2, 21.5 and 29.8 deg/s (seeds 0, 1, 2). The controls turned at 58.3, 54.0 and 65.5 deg/s, against a torque-free 59.0,
64.0 and 66.8. With the decoders on, |w| first fell below 15 deg/s at 4.34, 20.48 and 21.78 s; no control got there.
Over the last 5 s, the per-axis rms was 2.6-16.1 deg/s with the decoders on and 15.9-56.2 in the controls. The largest
remaining axis with the decoders on was pitch on seed 2 (16.1), and pitch is the axis nothing holds.

The hold is not monotone. In the first second it made every seed's tumble faster than the control's: at 1.0 s the
decoders-on ship turned at 65.4, 67.1 and 74.3 deg/s against 60.7, 65.6 and 67.3 with the thrusters off (0-1 s means
62.8, 66.4 and 70.5 against 60.4, 65.5 and 67.2). On seed 2 the ship was back up to 51.2 deg/s at 17.0 s, after
rock 2's belly hit at 14.00 s (30.7 -> 39.3 deg/s) and a burn at 14.97 s, before it came down below 15 at 21.78 s. After 5 s the decoders-on B200
runs peaked at 31.9, 30.2 and 51.2 deg/s, below their torque-free means of 59.0, 64.0 and 66.8.

**With the thrusters off, the ship's rotation is physics alone.** In the three controls, |w| equals the torque-free
reference in every 0.5 s sample (difference 0.000 deg/s) until the first rock hits, at 8.78-8.90 s. After that only
the knocks move it. The largest changes:
* seed 1: 63.7 -> 56.1 deg/s at rock 1's left-wing hit (8.80 s), and 56.3 -> 43.9 at rock 3's belly hit (18.87 s);
* seed 2: 61.4 -> 77.4 deg/s at rock 5's left-wing hit (28.27 s);
* seed 0: 60.1 -> 44.9 deg/s at rock 5's left-wing hit (28.28 s).

**Rocks.** Each rock is launched 5 s before its arrival time and aimed at where the cruise autopilot alone would take
the ship (`make_threat`). After launch, only a burn, a dodge or a knock from another rock moves the ship off that path.
The ship's attitude also decides which part a rock meets, and whether a wing or a pod reaches a rock aimed to pass the
centre at 5.5-7.5 m.

| clip | rock 1 (aimed to hit) | rock 2 (aimed to pass) | rock 3 (aimed to hit) | rock 4 (aimed to pass) | rock 5 (aimed to hit) |
|---|---|---|---|---|---|
| seed0 (4090) | hit left wing, 8.85 s | missed by 0.58 m, 13.95 s | missed by 6.20 m, 18.85 s *(after burn 14.09 s)* | missed by 2.02 m, 23.96 s | hit fuselage, 28.30 s |
| seed1 (4090) | hit left wing, 8.85 s | hit right engine pod, 13.94 s | hit canopy, 18.90 s | missed by 4.05 m, 23.08 s *(a)* | hit right engine pod, 28.27 s |
| seed2 (4090) | hit fuselage, 8.79 s | hit fuselage, 13.81 s *(after burn 13.15 s)* | hit main engine, 18.60 s | missed by 4.84 m, 24.21 s *(after burn 21.51 s)* | hit fuselage, 28.31 s |
| b200_seed0 (B200) | hit left wing, 8.85 s | missed by 1.26 m, 14.00 s | missed by 2.01 m, 18.96 s *(after burn 16.79 s)* | hit left engine pod, 23.90 s *(after burn 22.09 s)* | missed by 10.88 m, 28.80 s *(after rock 4's knock 23.90 s)* |
| control_seed0 (B200) | hit fuselage, 8.78 s | missed by 2.04 m, 14.24 s | hit belly, 18.91 s | missed by 2.01 m, 23.92 s | hit left wing, 28.28 s |
| b200_seed1 (B200) | hit left wing, 8.78 s | missed by 7.54 m, 13.51 s *(after burns 11.61 and 13.21 s)* | hit right engine pod, 18.84 s | missed by 2.09 m, 24.15 s | hit left engine pod, 28.30 s |
| control_seed1 (B200) | hit left wing, 8.80 s | missed by 0.38 m, 14.00 s | hit belly, 18.87 s | missed by 2.07 m, 23.28 s | hit fuselage, 28.25 s |
| b200_seed2 (B200) | hit fuselage, 8.82 s | hit belly, 14.00 s *(after burn 12.93 s)* | missed by 57.93 m, 23.12 s *(after rock 2's knock 14.00 s and burns 14.97, 18.73 s)* | missed by 3.92 m, 24.06 s *(a)* | hit belly, 28.35 s |
| control_seed2 (B200) | hit belly, 8.90 s | missed by 0.72 m, 14.23 s | hit belly, 18.86 s | missed by 1.66 m, 23.60 s | hit left wing, 28.27 s |

"After" lists every burn and knock between the rock's launch and its closest approach. Clearances are closest surface
clearances; times are the moment of contact or of closest approach. (a) A burn that began 0.09 s (seed1, 18.91 s) or
0.27 s (b200_seed2, 18.73 s) before rock 4's launch was still thrusting after it. Both rocks passed anyway.

What the table shows:
* **In the three controls every rock did what it was aimed to do.** Rocks 1, 3 and 5 hit, 9 of 9. Rocks 2 and 4
  passed at 0.38-2.07 m, 6 of 6.
* **With the decoders on, 8 of 30 rocks ended the other way: 3 on the 4090 and 5 on the B200.** Seven of the eight
  followed a move of the ship after the rock's launch:
  * after a burn alone, two rocks aimed to hit missed (rock 3 on seed 0, on both GPUs), and three rocks aimed to pass
    hit (rock 2 on seed 2, on both GPUs, and rock 4 on b200_seed0);
  * after a knock from another rock, two rocks aimed to hit missed (rock 5 on b200_seed0; rock 3 on b200_seed2, which
    also had two burns and passed 57.93 m away).
  * The eighth, rock 2 on the 4090's seed 1, was aimed to pass 6.01 m off and hit the right engine pod, with nothing
    moving the ship after its launch. The ship's attitude put the pod in its way; its path was the aimed one.
* **On the B200, the decoders-on runs took 8 hits in 15 rocks and the controls 9 in 15.** On the 4090 the decoders-on
  runs took 10 in 15.
* **The burns do not aim.** Their direction is fixed in the ship's frame, (forward + up) / sqrt 2, while the ship
  turns. When a burn fired, the rock that later ended the other way was 10.5 m away (4090 seed 2, rock 2), 15.0 m (b200_seed2,
  rock 2) or 37.5-84 m (b200_seed0 rocks 3 and 4, b200_seed2 rock 3). On the 4090's seed 0, rock 3 had been launched
  0.09 s before the burn; the HUD labels it 120 m away at 14.12 s.
* **Seed 0, rock 3, on the 4090.** By construction, had nothing moved the ship after its launch, its centre would have
  passed 2.15 m from the ship's centre, closer than its core's largest radius (2.70 m). The only thing that moved the
  ship was the 14.09 s burn. The rock passed 6.20 m clear.

**Escape burns follow the tumble** (as measured 5 found). The GF was at or above 33 Hz in 2.0-4.8 % of ticks with the
decoders on and 9.8-24.4 % in the controls, whose tumble never slowed.
* **Decoders on (B200):** 13 burns. 8 fired with no rock within 50 m, 3 with a rock 30-50 m away, and 2 with a rock
  10-15 m away: b200_seed1 at 13.21 s (rock 2 at 10.2 m) and b200_seed2 at 12.93 s (rock 2 at 15.0 m).
* **Controls:** 37 would-be burns. 17 with no rock within 50 m, 10 at 30-50 m, 10 within 30 m.
* The controls logged 11-14 would-be burns in 30 s, one every 2.1-2.7 s on average, with or without a rock near. At
  their 43-67 deg/s, the log cannot separate a near rock's contribution from the tumble's.

**Rocks within 1 m.** Three rocks passed within 1 m of the hull in the nine runs: rock 2 on the 4090's seed 0 (0.58 m),
on control seed 1 (0.38 m) and on control seed 2 (0.72 m). A burn event marks the first tick at or above 33 Hz once the
1.5 s refractory has run out, so it dates a crossing only when the refractory had long expired:
* **seed 0, 4090:** the burn at 14.09 s, 0.14 s after the pass, came 11.65 s after the previous one, so this is when
  the giant fibre crossed 33 Hz. Mean 36.3 Hz: the left cell at 71.8 Hz, the right at 0.7 Hz. The tumble was
  4.8 deg/s, the rock 2.1 m away. It was fired, after the pass, so it could not have moved the ship out of that rock's
  way.
* **control seed 1:** a would-be burn at 14.03 s, 0.03 s after the pass: mean 49.0 Hz, with the right cell at
  89.9 Hz. It came on the first tick after the refractory of the 12.53 s would-be burn, so the log does not say when
  the giant fibre crossed; it may have been above 33 Hz before the pass. It was not applied.
* **control seed 2:** no would-be burn between 11.33 s (rock 2 then 36.1 m away) and 15.19 s. The approach peak shows
  the mean GF reached 44.6 Hz while rock 2 was within 30 m, so it crossed 33 Hz during the 11.33 s burn's refractory
  (which ran out at 12.83 s), more than 1.4 s before the pass at 14.23 s, with no event logged.

**The dodge never fired.** In all six decoders-on runs, the low-passed LPLC2 + LC4 side signal never reached 5 Hz. The
seed-0 control would have dodged once, at 7.06 s: side -5.03 Hz (L 1.22, R 6.49), tumbling 58.4 deg/s, with rock 1
36.4 m away. A tumble drives this signal to about that size on its own (measured 6).

### Same seed, two GPUs

The 4090 run and the B200 replicate of each seed ran the same `games/spacecraft.py`. They are not a comparison of arms;
they show how far one arm moves between GPUs. Their 0.5 s samples (per-axis rates, speed and torque-free, to two
decimals) are identical through 2.5 s (seed 0), 3.0 s (seed 1) and 2.0 s (seed 2). They then drift apart: GPU
floating-point noise, amplified by the closed loop (measured 11).
* The first two burns came at the same ticks on both GPUs: seed 0 at 0.17 and 2.44 s, seed 1 at 0.19 and 1.69 s,
  seed 2 at 0.16 and 1.74 s.
* The means over 2-30 s differ by 2.2, 0.1 and 0.9 deg/s (4090: 17.46, 21.40, 28.86; B200: 15.24, 21.46, 29.78).
* Hit or miss differs for 4 of the 15 rocks: seed 0 rocks 4 and 5, seed 1 rock 2 and seed 2 rock 3. The part hit or
  the clearance differs for most of the others.

### The 4090 clips, one by one (the trailer's source)

**Seed 0** (`seed0.mp4`; tumble 60.0 deg/s).
* Burns at 0.17 s (GF 33.8 Hz: left 12.6, right 54.9) and 2.44 s (35.1 Hz), tumbling at 60.2 and 35.5 deg/s, before
  any rock was launched.
* The hold first sped the tumble up (65.4 deg/s at 1.0 s; the HUD reads 65 with roll +54), then brought it down:
  |w| fell below 15 deg/s at 4.26 s. Banner: "ATTITUDE HOLD 60 -> 15 deg/s IN 4.3 s", under it "decoders -> roll /
  yaw thrusters · torque-free it would spin at 60 deg/s".
* Rock 1 hit the left wing at 8.85 s (5.0 -> 18.4 deg/s).
* **Rock 2 passed 0.58 m from the hull at 13.95 s.** The giant fibre crossed 33 Hz 0.14 s later, at 14.09 s: mean
  36.3 Hz, left cell 71.8 Hz, right 0.7 Hz, with the ship turning at 4.8 deg/s. The burn fired (banner "GIANT FIBRE
  36 Hz -> ESCAPE BURN · tumbling 5 deg/s · rock 2 2 m away"). "MISS 0.6 m" follows at 14.16 s, and the two banners
  share the screen at 14.5 s. The burn came after the pass. It did not make that miss.
* Rock 3, aimed to hit, then passed 6.20 m clear at 18.85 s. The 14.09 s burn is the only thing that moved the ship
  after rock 3's launch (What the table shows).
* Rock 4 passed at 2.02 m (23.96 s).
* Rock 5 hit the fuselage at 28.30 s (16.4 -> 27.1 deg/s). A burn at 28.45 s followed; its banner says "no rock
  within 50 m", because the nearest-rock lookup skips rocks already resolved, rock 5 included.
* The last 2 s averaged 27.3 deg/s against a torque-free 58.1.

**Seed 1** (`seed1.mp4`; tumble 65.3 deg/s).
* Burns at 0.19 s and 1.69 s (34.4 Hz each, the right cell at 58.5 and 63.5 Hz), tumbling at 65.5 and 44.7 deg/s.
* Rock 1 hit the left wing at 8.85 s and raised the tumble from 26.7 to 41.5 deg/s. |w| fell below 15 at 11.1 s
  (banner "65 -> 15 deg/s IN 11.1 s").
* Rock 2, aimed to pass 6.01 m off, hit the right engine pod at 13.94 s (8.0 -> 8.9 deg/s).
* **Rock 3 hit the canopy, where the fly's eye sits, at 18.90 s** (19.0 -> 18.3 deg/s). The giant fibre crossed
  33 Hz one tick later (18.91 s: mean 34.2, left 35.9, right 32.5 Hz). While rock 3 was within 30 m, the mean GF had
  peaked at 29.9 Hz. The burn's banner says "no rock within 50 m" for the same reason as seed 0's last one: rock 3 had
  just been resolved.
* Rock 4 passed at 4.05 m (23.08 s). Rock 5 hit the right engine pod at 28.27 s (18.3 -> 28.4 deg/s).

**Seed 2** (`seed2.mp4`; tumble 67.0 deg/s, roll-heavy). The hold was slowest here.
* Burns at 0.16 and 1.74 s, tumbling at 67.2 and 63.1 deg/s.
* Rock 1 hit the fuselage at 8.79 s (26.6 -> 27.6 deg/s).
* A burn at 13.15 s (33.6 Hz; rock 2 10.5 m away, tumbling 24.3 deg/s). Rock 2, aimed to pass 6.64 m off, then hit
  the fuselage at 13.81 s (23.3 -> 32.0 deg/s); in the clip the flash is at the nose.
* Rock 3 hit the main engine at 18.60 s. The 19-20 s window averaged 41.1 deg/s.
* A burn at 21.51 s (rock 4 35.0 m away). Rock 4 passed at 4.84 m (24.21 s).
* |w| first fell below 15 at 25.91 s. Rock 5 hit the fuselage at 28.31 s (22.1 -> 25.6 deg/s).

### The B200 clips, briefly

* **b200_seed0** ends calmest of all nine: 5.2 deg/s over the last 2 s, per-axis rms 2.6-4.0 over the last 5 s, and
  "MISS 10.9 m" for rock 5 at 29.20 s. Rock 4 hit the left engine pod at 23.90 s after a burn at 22.09 s. That knock
  came 0.40 s after rock 5's launch and moved the ship off rock 5's path.
* **control_seed0.** Tumbling at 58-61 deg/s until rock 5's hit at 28.28 s: "CONTROL · THRUSTERS OFF", an amber frame,
  and grey "BURN NOT APPLIED" banners (11) and one "DODGE NOT APPLIED" (7.06 s). A would-be burn at 28.61 s also reads
  "no rock within 50 m" just after rock 5's hit.
* **b200_seed1.** Below 15 deg/s at 20.48 s; rock 2 passed 7.54 m clear after burns at 11.61 and 13.21 s.
* **control_seed1.** Rock 2 passed 0.38 m from the hull at 14.00 s, and a would-be burn followed at 14.03 s, the first
  tick after the 12.53 s one's refractory ("GIANT FIBRE 49 Hz -> BURN NOT APPLIED · rock 2 0 m away").
* **b200_seed2.** Rock 2 hit the belly at 14.00 s after a burn at 12.93 s. Its knock (a velocity change of
  11.95 m/s, in the tick rock 3 was launched) and burns at 14.97 and 18.73 s moved the ship off rock 3's path, and
  rock 3 passed 57.93 m away. |w| peaked
  at 51.2 deg/s at 17.0 s and first fell below 15 at 21.78 s.
* **control_seed2.** The fastest tumble; rock 5's left-wing hit at 28.27 s raised it to 77.4 deg/s, and the last 2 s
  averaged 75.7. The GF was at or above 33 Hz in 24.4 % of ticks, and 14 burns would have fired.

### For the trailer

The trailer's spacecraft shots come from the 4090's `seed0.mp4`, except the split screen, which pairs
`b200_seed0.mp4` ("DECODERS ON") with `control_seed0.mp4` ("CONTROL: THRUSTERS OFF") from 1.6 s
(`games/trailer_shots.py`). Both are B200 clips from one batch, so the only on-screen comparison of the arms stays on
one GPU model, and both HUDs are set in the same DejaVu fonts. `seed0.mp4` must not stand in for the decoders-on side:
it is a 4090 run, and although its samples match `b200_seed0.mp4` through 2.5 s (both HUDs read 47 deg/s at 1.6 s and
22 at 3.5 s), the two drift apart after that. What the split shows, per window:
* **1.6-3.5 s:** decoders on 47 -> 22 deg/s (0.5 s samples 41.7, 34.4, 27.8, 22.5 at 2.0-3.5 s) beside the control at
  61 -> 60 deg/s (61.0, 60.9, 60.7, 60.4). Both sides carry giant-fibre banners, neither an escape from anything: on
  the left "GIANT FIBRE 35 Hz -> ESCAPE BURN · tumbling 35 deg/s · no rock within 50 m" from 2.42 s (the 0.17 s one
  still fading at 1.6 s), on the right "GIANT FIBRE 34 Hz -> BURN NOT APPLIED" (1.77 s). Neither side falls below
  15 deg/s in this window.
* **to 4.5 s:** `b200_seed0.mp4` falls below 15 deg/s at 4.32 s and its banner reads "ATTITUDE HOLD 60 -> 15 deg/s IN
  4.3 s"; the control is at 59.7 deg/s.

The control side does not depend on the GPU for its motion: with the thrusters off, nothing the brain does moves the
ship (|w| equals the torque-free reference exactly until the first knock), and the killed 4090 control of seed 1 and
the B200 one show the same ship at 9.2 s: 56 deg/s, roll -4 / pitch -48 / yaw -29, rock 2 89 m away, one impact. Its
neural numbers (GF, LPLC2 / LC4, the decoded signals) are the B200 brain's: at that moment the two show different GF
banners (37 Hz, rock 1 16 m away, against 36 Hz, 17 m). The pair's numbers over 2-30 s are 15.2 against 58.3 deg/s.

The first shot, `seed0.mp4` from 0.0 s, shows the ship tumbling beside the ring station. Cut at 4.0 s, it ends with
the HUD at 17 deg/s: |w| falls below 15 at 4.24 s and the ATTITUDE HOLD banner fades in from there, so neither is in
a 0.0-4.0 s cut. On the way the tumble first rises from 60 to 65 deg/s (about 1 s), and two escape-burn banners show
(0.15 and 2.42 s, "no rock within 50 m"; no rock has been launched).

In `seed0.mp4` around 14 s, "MISS 0.6 m" and "GIANT FIBRE 36 Hz -> ESCAPE BURN" share the screen. The burn came
0.14 s after rock 2's closest pass, so a cut there must not say that the burn made the miss.

Trailer windows, in clip seconds (clip time = brain time - 0.02 s):

| clip | window (s) | what is on screen |
|---|---|---|
| `seed0.mp4` | 0.0-5.5 | the 60 deg/s tumble beside the ring station, rising to 65 at about 1 s before it falls; the roll and yaw thrusters fire; burn banners at 0.15 and 2.42 (no rock launched); "ATTITUDE HOLD 60 -> 15 deg/s IN 4.3 s" fades in from 4.24 (a cut ending at 4.0 stops at 17 deg/s) |
| `b200_seed0.mp4` + `control_seed0.mp4` | 1.6-3.5 (to 4.5 for the banner) | split screen, both B200: the hold, 47 -> 22 deg/s (below 15 at 4.32, banner "ATTITUDE HOLD 60 -> 15 deg/s IN 4.3 s"), beside the control at 61 -> 60 deg/s; GF banners on both sides |
| `seed1.mp4` | 18.6-19.8 | rock 3 hits the canopy (18.88); sparks; "IMPACT" and "GIANT FIBRE 34 Hz -> ESCAPE BURN", the burn one tick after the hit |
| `seed2.mp4` | 13.1-14.3 | "GIANT FIBRE 34 Hz -> ESCAPE BURN" (13.13); rock 2, aimed to pass, hits the nose in a flash (13.79) |
| `seed0.mp4` | 8.4-9.4 | rock 1 hits the left wing (8.83; the rock is mostly behind the hull, the flash in front), the tumble jumps from 5 to 18 deg/s, "IMPACT · rock 1 hit the left wing" |
| `seed0.mp4` | 13.4-14.8 | rock 2 passes 0.58 m from the hull (13.93; on screen it is large beside the nose), then, with the rock already behind, the burn (14.07) and "MISS 0.6 m" (14.14); both banners on screen from about 14.3 |
| `seed0.mp4` | 27.9-29.3 | rock 5 hits the fuselage (28.28), then a burn (28.43) |
| `b200_seed0.mp4` | 28.6-30.0 | the calmest ending: 5 deg/s, "MISS 10.9 m" (29.18); B200 fonts |
| `control_seed1.mp4` | 13.6-14.6 | control: rock 2 passes 0.38 m (13.98), "GIANT FIBRE 49 Hz -> BURN NOT APPLIED" (14.01) |


## Corrections (caption skeptic)

Checked against all nine run logs (summaries, events, 0.5 s samples), frames extracted from `seed0.mp4`,
`b200_seed0.mp4`, `control_seed0.mp4`, `seed1.mp4`, `seed2.mp4`, `control_seed1.mp4` and the killed
`control_seed1_attempt1.mp4`, `games/spacecraft.py`, `games/common.py`, `scripts/cluster_run.py`,
`games/trailer_shots.py` and the cluster console output. Every table number in "What the run logs say", the rock table,
the burn and dodge counts, the source hashes (including the `b110142562a927a1` reconstruction of `common.py`) and the
frame and clip facts checked out except as listed. The dev-seed measurements 1-8 were not re-derived.

1. Title: "a fly flies a spaceship" -> "declared decoders read a fly connectome's motion cells to steady a tumbling
   spaceship". The fly does not fly; the decoders and the game do.
2. Headline: "Three declared decoders read its H2 / HS and VST2 cells and fire the roll and yaw thrusters" -> "One
   declared decoder reads its H2 / HS and VST2 cells and fires the roll and yaw thrusters; two more read LPLC2 / LC4
   (a dodge, which never fired) and the giant fibre (an escape burn)". Only `attitude_hold` reads those cells.
3. Headline: "In all three recorded seeds, a 60-67 deg/s tumble fell to a mean of 15-30 deg/s over 2-30 s. With the
   thrusters off, the same tumbles stayed at 54-66 deg/s" -> the B200 pairs by name and number ("On one NVIDIA B200 ...
   15.2, 21.5 and 29.8 deg/s with the decoders on, and at 58.3, 54.0 and 65.5 deg/s with the thrusters off"). The
   "recorded seeds" are the 4090 clips, which have no control, and "stayed at" hid control samples of 43-78 deg/s.
4. Headline: "After its burns, three rocks aimed to pass the ship hit it, and two aimed to hit it missed" -> "some
   rocks aimed to hit missed and some aimed to pass hit: on the B200 the decoders-on runs took 8 hits in 15 rocks, the
   controls 9 in 15". The old count mixed the two GPUs, counted seed 0 rock 3 and seed 2 rock 2 once per GPU, and left
   out b200_seed2 rock 3, which also missed after two burns (and a knock).
5. Design: "What the fly does here is what these decoders make of its lobula plate ... DNa02 stays silent" -> "What
   the ship does here is what these decoders make of the fly's lobula-plate, loom and giant-fibre cells ... (README:
   DNa02 is held below threshold), and nothing here reads DNa02". The fly does nothing here; the escape burn reads the
   giant fibre, not the lobula plate; and no run log measures DNa02.
6. "What the run logs say": "All speeds are ... from the log's 0.5 s samples" -> only the 2-30 s means and the maxima
   come from the samples (the 56 at 2.5-30.0 s); the window means, rms and "first below 15" are per-tick summary
   values.
7. Decoders vs control (B200): added "In the first second it made every seed's tumble faster than the control's: at
   1.0 s ... 65.4, 67.1 and 74.3 deg/s against 60.7, 65.6 and 67.3". This outcome had been left out. The bold claim now
   reads "beat the control on every seed over 2-30 s".
8. "Rocks within 1 m": "control seed 1: 14.03 s ... the giant fibre crossed 33 Hz after the pass" -> the 14.03 s
   would-be burn is the first tick after the 12.53 s one's 1.5 s refractory, so the log does not date that crossing;
   the GF may have been above 33 Hz before the pass. "No crossing was logged near the third (control seed 2)" -> its
   approach peak shows the mean GF at 44.6 Hz with rock 2 within 30 m, a crossing during the 11.33 s burn's
   refractory, more than 1.4 s before the pass, with no event logged. The same fix is made in "The B200 clips,
   briefly" ("the giant fibre crossed at 14.03 s" -> "a would-be burn followed at 14.03 s, the first tick after the
   12.53 s one's refractory").
9. Same seed, two GPUs: "differ by 2.3, 0.1 and 0.9 deg/s" -> "2.2, 0.1 and 0.9" (17.46 - 15.24 = 2.22).
10. Provenance: "The scheduler placed all six jobs on one GPU (`<cluster-node>`, GPU 0)" -> "All six jobs were sent to `<cluster-node>`
    (`--node <cluster-node>`; no `--gpu-ids` pin) ... The saved console output and logs name the GPU model, not which of
    `<cluster-node>`'s GPUs ran each job". Nothing saved shows a GPU index, and without `--gpu-ids`, `cluster_run.py` does not pin
    one. For the same reason, "a same-GPU replicate ... on one NVIDIA B200" -> "all six in one batch on the cluster's
    NVIDIA B200s (`<cluster-node>`)", and "on one NVIDIA B200 took 297.5-302.9 s" -> "on `<cluster-node>`'s NVIDIA B200s".
11. Provenance: "Job k runs this command ... for k = 0, 1, 2" -> the job order (jobs 0-5) spelled out, and k named as
    the seed. Job indices run 0-5; seeds run 0-2.
12. For the trailer: the split screen "of `seed0.mp4` ('DECODERS ON') beside `control_seed0.mp4`" (4090 beside B200)
    -> `b200_seed0.mp4` beside `control_seed0.mp4`, as `games/trailer_shots.py` now cuts it, with what each side shows
    at 1.6-3.5 s (47 -> 22 against 61 -> 60 deg/s, GF banners on both sides) and at 4.32 s. The table row "`seed0.mp4`
    + `control_seed0.mp4` | 1.6-6.0 | ... below 15 at 4.24" is replaced the same way (below 15 at 4.32 on the B200).
13. For the trailer: added that a 0.0-4.0 s cut of `seed0.mp4` ends at 17 deg/s, before the drop below 15 (4.24 s clip)
    and the ATTITUDE HOLD banner. On the way the tumble first rises to 65 deg/s, and two escape-burn banners show with
    no rock launched. The same rise was added to the seed-0 clip notes, and the burn banners to the `seed0.mp4`
    0.0-5.5 row.
14. Trailer table: rock 1's row now says the rock is mostly hidden behind the hull at the hit; rock 2's row (window
    widened to 13.4-14.8) says the burn banner comes with the rock already behind the ship.
15. "The 4090 clips are the trailer's source" -> "`seed0.mp4` is the trailer's main source (its split screen pairs the
    two B200 arms)".
16. Limits: "They reduce a 55-70 deg/s tumble to about 10-30 deg/s ... That floor is largely the loop's own
    noise-driven motion" -> the recorded 15-30 deg/s means (B200 15.2-29.8, 4090 17.5-28.9, window means 5-50), and
    "not leftover tumble alone: rock knocks raise it (b200_seed2 went back up to 51.2 deg/s) ...". Added the
    first-second spin-up to "The loop can spin the ship up".
17. Dev clips (measured 10): "These dev clips were recorded from the final code" -> their logs carry
    `spacecraft.py` `6e34970afe63f18e`, not the frozen `793eddf416eaa6bb`; the draw-only difference is the builder's
    account and could not be re-checked (the earlier file is not kept). "The tumble stayed at 54-68 deg/s ... The
    torque-free reference ran at 64-66" -> "53.8-69.3 deg/s in the 0.5 s samples (window means 54.5-68.4) ... 63.9-67.0".
18. Wording: "the rock it changed" -> "the rock that (then / later) ended the other way" (twice); "sent rock 3 past
    57.93 m away" -> "moved the ship off rock 3's path, and rock 3 passed 57.93 m away"; "inside its own 2.70 m core" ->
    "closer than its core's largest radius (2.70 m)". The core is an ellipsoid, and a pass-aimed rock can hit without any
    move (4090 seed 1 rock 2), so a burn is not shown to have changed the outcome.
