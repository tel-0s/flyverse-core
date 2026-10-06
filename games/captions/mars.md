# mars: a fly connectome rides a Mars rover

**Fly-by-wire on Mars: a dust devil blows on a fruit-fly connectome's antennae, its wind neurons report which
side the wind comes from, and a declared decoder veers the rover into the wind, which the game set up to be away from
the devil.** In all three recorded seeds the devil's wind reached the rover from the left on the approach. The decoder
began turning the rover left at 29.9-31.3 s (seed 0 also by 0.05 deg at 26.8 s), and its command passed the log's
veer mark (0.6 of its 18 deg/s cap) at 31.95-32.29 s, when DNp18 fired 24-35 Hz on the left against 0-7 Hz on the
right. The rover's heading swung to about 11 deg and it held about 4 m off the route line while the devil passed,
19.8-19.9 m away. That offset is set by GAME numbers: the decoder sat at its 18 deg/s cap, and the autopilot's pursuit
cancels 18 deg/s at 12 m x tan 18 deg = 3.9 m off the line. In the six controls on the same seeds (no wind to the antennae, or the
decoders disconnected while the wind neurons reported the devil's wind as strongly) the rover drove the line and the
devil came within 16.0 m. The escape pathway never flagged a boulder: the blind GAME autopilot drove into all four in
every clip (GAME rules took it round each in one contact), and the giant fibre peaked at 15.7-21.8 Hz, under its 33 Hz
stop line (0-5.3 Hz at the moments of contact).

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the fly) | MaleCNS v1.0 as shipped: 167,106 neurons, preset `raw`, nothing instrumented, trained or tuned for this game. Two senses: vision (the radiance along its 1,466 ommatidial columns, 7 rays each, ray-traced from a post above the rover's mast and handed to `fb.vision` every 10 ms) and the antennae (the dust devils' wind through the shipped antenna model, handed to `fb.wind` every 10 ms). Everything from that light and that antennal deflection to the rates of DNp18, DNp33, the giant fibre DNp01, LPLC2, LC4 and MDN is the model, including the shipped LC4 / LPLC2 -> DNp01 stop-gap gain that the README's ledger declares. The HUD's DN, GF and loom numbers are its rates. |
| **DECODER** (written for this game, declared) | Two read-only modules attached with `kind="decoder"`; they write nothing into the brain. **wind_steer:** a yaw rate from the wind descending neurons' left-right index, towards the side the index reports (into the wind), capped at 18 deg/s (below the autopilot's 24 deg/s limit) and added to the autopilot's while the rover drives forward (a GAME rule holds it during collision recoveries). **hazard_stop:** the giant fibre's mean rate >= 33 Hz (the shipped body model's takeoff line) brakes the rover; it never fired. |
| **GAME** (not the fly) | The planet: terrain, crater rim, dunes, boulders, sky, sun, haze, the dust devils and their wind. The rover: its shape, speeds and the autopilot that drives the route. The rule that holds the wind steer's turn while the rover recovers from a collision, and the manoeuvre round a rock it hit. The scoring, the cameras, the wheel dust, the tracks, the wind-borne dust streaks, the banners. |
| **not claimed** | The fly does not drive, detect hazards, avoid boulders or flee dust devils. The autopilot drives; the wind-steer decoder adds a turn when the wind DNs lateralise; the stop decoder never fired. The rover's yaw has three sources in the code: the GAME autopilot, the GAME manoeuvre round a rock it hit, and the decoder's applied command. Nothing else from the brain reaches it (the fly's own motor output is not read, and the raw model does not turn on its own anyway: README). "Away from the dust devil" holds because the GAME devil's wind reached the rover from the side away from the devil on the approach. The fly does not know where the devil is. |

## Design

A six-wheeled rover about Perseverance's size (3.1 x 2.7 m footprint) crosses the floor of a Jezero-like crater (a
procedurally generated GAME scene, not survey data) toward a waypoint 170 m down a straight route. Four boulders sit
on the route line, two before and two after a 110 m open stretch, and a dust devil drifts along the open stretch 16 m
to the right of the route (the same scene for every seed). The shipped fly (MaleCNS v1.0, `preset="raw"`, no
instruments) rides on a 35 cm post above the mast's camera head, 2.52 m above the ground, facing the rover's heading.
A GAME autopilot drives the route line at 5 m/s, about 40 times a real rover's top speed. The brain touches the rover
only through two declared decoders:

- **wind steer.** The dust devil's wind reaches the fly's antennae through the shipped antenna model and `fb.wind`. The
  wind descending neurons DNp18 and DNp33 lateralise (the README's strongest lateralised signal; games/hairdryer.py
  measured them), and a decoder turns the rover towards the side they report, i.e. into the wind. The devil turns
  counter-clockwise and its surface wind spirals in, so the rover meets its wind from the left while it approaches
  and passes: that turn is a turn away from the devil. The log scores every veer for both (into the wind? away from
  the devil?).
- **hazard stop.** The giant fibre DNp01 at the shipped 33 Hz takeoff line brakes the rover. It is the test PLAN.md
  asks for (does the escape pathway see a boulder at rover speed?). It never fired in any run (below).
  The first build's loom-steer decoder (LPLC2 + LC4 left minus right) was removed: that signal was biased to the right
  whichever side the boulder was on (PLAN.md: build only what responds).

The wind steer and the autopilot share the rover's yaw by a declared GAME rule: **the decoder's command is added in
full while the rover drives forward, and held (not applied) while it backs up or goes round a rock it hit; the
autopilot's limit (24 deg/s) is above the decoder's cap (18 deg/s), so the autopilot can always bring the rover back to
the route.** An earlier build (a clockwise devil on the left of the route) had the decoder's cap (30 deg/s) above the
autopilot's limit (14 deg/s) and no hold: after the devil had passed, its wind turned the rover left harder than the
autopilot could turn it back, and on dev seeds 102 and 103 the rover hit the last boulder 9 times each (14 contacts in
each run) and never finished (`out/games/mars/fix2/lo_on_s10{2,3}.json`). The log counts every held tick and every held
episode.

| what | kind | law and parameters |
|---|---|---|
| the brain | CONNECTOME | `FlyBrain(seed=...)` via `games.common.build_brain`, preset `raw`, nothing instrumented; two read-only decoders attached (`module_records()` in the run log) |
| the fly's eyes | GAME | 1,466 columns x 7 rays (`fb.retina.ray_directions()`, retina weights) from `EYE_LOCAL` = (1.08, 0, 2.52) m in the rover frame: on a 35 cm post (radius 1.2 cm) above the mast's camera head, oriented with the rover (x forward, y left, z up; the rover pitches and rolls with the ground under its six wheels). The rays are traced every 10 ms through the camera's scene: the same terrain, rocks, rover, dust devils and sky, with the sun's halo but not its 0.55 deg disc, marched coarser (170 steps growing 5.5 % of the distance, 24 shadow steps; the camera 300 / 1.2 % / 40), and without the wheel dust, the tracks and the wind-borne dust streaks (display only). Linear RGB -> radiance [UV, B, G, R] with **UV = 0.5 B** (`games.common.rgb_to_radiance`). On CUDA the eye's trace is replayed from a CUDA graph (the same kernels on the same inputs), checked against the eager trace once at the start of every run (`meta.eye_cuda_graph` in the log) |
| what the fly sees of the rover | GAME (geometry) | At the start pose 71 of the 1,466 columns get at least half their rays from the rover itself (89 get at least one ray): the mast head below the eye, the post and the deck, all at elevations -75 to -51 deg, in the lower field. (The first build's eye, 9 cm above the head, had 421.) CPU geometry, no brain |
| what a boulder fills | GAME (geometry) | Boulder H1 (top 1.29 m, ground radius 1.26 m) as the rover drives the line at it, in columns with at least half their rays on it: 1 at 12 m and 8 m from the footprint's front, 7 at 5 m, 16 at 3 m, 30 at 2 m, 63 at 1 m, 118 at contact, all below the eye (elevation -8 to -66 deg). CPU geometry from the game's eye and rover, no brain |
| the scene | GAME | A 4.1 x 4.1 km heightfield (2048^2, 2 m texels; spectral fractal noise, a crater rim of 120 m at 1.45 km from (160, 60) m, dark transverse dunes north of the route, a smoother track along the route) plus a 64 m tiled detail field (4.5 cm SD). 74 boulders (unions of three ellipsoids, sunk a third into the ground): 4 hazards on the route line (H1-H4 at x = 20, 33, 150, 163 m) and 70 scenery rocks kept at least 8 m off it. Mars sky: butterscotch horizon, darker zenith, a pale blue-white glow round a small sun (azimuth 58 deg, elevation 24 deg), hard sun shadows from terrain, boulders and the rover, aerial haze (2.6 km). Layout seed `0x3A25`: the scene is identical for every `--seed` |
| dust devils | GAME | Three. D1 and D2 drift 290-540 m away (scenery; their wind at the rover is negligible). **D3** starts at (84, -16) m, 16 m right of the route line, and drifts along it at 0.55 m/s. Each is drawn (for the eye and the camera) as a column of dust whose width grows with height, with twisting streaks, a dusty skirt at its foot and a sunlit and a shaded side, and has a surface wind: a vortex whose tangential speed rises linearly to `vt` at the core radius `rc` and falls outside as `vt (rc / r) exp(-(r - rc) / 20 m)`, spiralling inward at 35 deg (wind = v_t (e_theta - tan 35 deg e_r)). D3: `vt` 20 m/s, `rc` 7 m, **counter-clockwise** seen from above (dust devils turn either way). No ambient wind |
| antennae | GAME -> shipped sense | The devils' wind at the rover's position, as the Earth wind of the same dynamic pressure (x 0.128 = sqrt(0.020 / 1.225 kg/m^3), Mars' surface air density over Earth's), through the shipped antenna model `flyverse.air.Air.deflections` (antennae at +-45 deg, full deflection at 0.5 m/s) for the rover's heading, handed to `fb.wind(dL, dR)` every 10 ms. **The rover's own motion is not delivered as wind** (GAME): at a real rover's few cm/s it would be negligible, and at the game's 5 m/s it would saturate both antennae |
| the rover | GAME | Boxes, capsules and six finite-cylinder wheels (radius 0.27 m), ray-traced analytically; its pose follows the heightfield under its wheels. Kinematics: a skid-steered point on the plane, `yaw += (autopilot + applied wind steer) dt`, `x += v cos(yaw) dt`. Speed schedule: waits 1.0 s (the brain settles on the scene), accelerates at 1.5 m/s^2 to 5.0 m/s |
| autopilot | GAME | Pure pursuit of the route line y = 0: `yaw rate = clip(1.0 /s x wrap(atan2(-y, 12 m) - yaw), +-24 deg/s)` |
| `wind_steer` | DECODER | An attached `games.common.ReadDecoder` reading `rate_hz` (the brain's 100 ms running rate estimate) of DNp18 L / R and DNp33 L / R (one cell each). `u = 1/2 [(DNp18_L - DNp18_R) - (DNp33_L - DNp33_R)]` (games/hairdryer.py's index; > 0 when the wind comes from the left), `u_s = EMA(u, 0.2 s)`, **yaw rate = 1.5 deg/s per Hz x (u_s beyond an 8 Hz dead band)**, clipped at **+-18 deg/s**, + = a left turn: towards the side u reports. Set on dev seeds >= 100 (below) |
| wind steer arbitration | GAME | The decoder's command is added to the autopilot's in full while the rover drives forward; while it backs up (after a hazard stop or a collision) or goes round a rock it hit, the command is held (not applied). Logged: `steer_held_ticks`, `steer_held_episodes` (collision recoveries in which a turn was held), `steer_held_abs_deg` (the turn withheld), and `steer_cap_ticks` (ticks on which the decoder's own cap clipped its command) |
| `hazard_stop` | DECODER | An attached `ReadDecoder` reading `rate_hz` of DNp01 (both cells): **stop when the mean rate >= 33 Hz** (`flyverse.body.Flight.gf_hz`, the shipped takeoff threshold; a test asserts they agree). Re-armed only after the rate falls below 33 Hz. Nothing in this decoder was tuned |
| stop mechanics | GAME | On the stop decoder's trigger, while driving above 0.3 m/s: brake at 6 m/s^2 to rest, back up 1.0 s at 1.2 m/s, drive on |
| collision | GAME | Contact = the 3.1 x 2.7 m footprint (an oriented rectangle) overlaps a boulder's ground circle (the largest horizontal extent of its three parts). Driving forward, the rover stops dead and goes round that rock the way a rover that turns in place would: it backs up straight at 1.2 m/s until the footprint is 1.2 m clear of the rock (at most 3 s), pivots in place at 45 deg/s to 55 deg off the route towards the side its centre is on, drives out at up to 2 m/s until the footprint will clear the rock by 0.8 m, pivots back parallel to the route, and drives past (pursuing that lane, 6 m look-ahead, 1.5 /s per rad, 14 deg/s) until its rear is past the rock; then the autopilot rejoins the route. Backing into a rock ends the back-up there (`reverse_blocked`, not a collision). The same in every arm. (A pure-pursuit detour of an earlier build needed up to 5 contacts to get round a rock met at an angle in stub runs without a brain; tests/test_games_mars.py checks one contact from a grid of approaches) |
| scoring | GAME | A hazard is **passed** when the rover's rear clears its far side; **clean** if the footprint never touched it, else **after a collision**. The log keeps each hazard's minimum footprint clearance over the poses the rover took (a pose that would overlap a rock is rejected, so this is >= 0 even for a rock it hit; `collision_gap_m` keeps the rejected pose's overlap), every GF crossing (applied or not) with the nearest hazard and its gap, every wind-steer onset and every **veer** (the decoder's command passing 0.6 of its cap after >= 2 s below it: direction, DN rates, index, the wind's direction, D3's distance and bearing, whether the turn was into the wind and away from D3, applied or held), every collision and held episode, every devil's closest approach to the rover's centre, the D3 encounter (while D3 is within 45 m: the rover's lateral excursion and heading range and the decoder's applied turn), and a 10 Hz series of the pose, the four wind DNs, u, u_s, the decoder's command and applied yaw, the autopilot's yaw, the wind, the deflections, GF, LPLC2 + LC4, MDN and D3's distance and bearing |
| MDN | CONNECTOME (log only) | The moonwalker DNs (the shipped `Locomotion` reads them as "back up" above 15 Hz) are logged, not shown, and drive nothing |
| HUD | | **CONNECTOME:** DNp18 / DNp33 L and R with the L-R index u (grey) and u_s (white) and the shaded dead band; DNp01 with its 33 Hz line; LPLC2 + LC4 left and right (population means, auto-scaled with the scale printed). **DECODER:** the wind steer's command and whether it is applied, held or off; the counts of hazard stops and of applied veers (WIND VEERS; the controls show them as "not applied"). **GAME:** the wind at the antennae (Mars m/s, its Earth equivalent, where it comes from relative to the heading, the two antennal deflections handed to `fb.wind`), the tally, speed, distance to the waypoint, the route minimap with the devils, the dust-devil label, the scene title. **WHAT THE FLY SEES** is the exact radiance handed to `fb.vision` (human colours, UV not shown). Banners carry the chips of what they report: WIND DNs -> VEER (CONNECTOME + DECODER, with the DN rates, the index and the commanded turn, and "away from the dust devil" only when the log says so), GIANT FIBRE -> HAZARD STOP (CONNECTOME + DECODER), COLLISION and PASSED (GAME) |
| cameras | GAME (display) | 0-3.2 s: a crane shot from behind-right of the rover that dollies in and rises, the sun in the upper left and the rover whole in frame, the title plate in the lower left (the rover starts rolling at 1.0 s); by 6.0 s a chase camera behind-left of the rover that follows a smoothed heading (1 s) and holds the route's direction while the rover goes round a rock it hit, so its pivots show as the rover turning. While D3 is within ~50 m and not yet well behind the rover, the camera sits behind-left of the rover in the route's frame and aims between the rover and the devil, never more than 20 deg off the rover; it does not turn with the rover there, so the veer shows as the rover turning in frame. A "the fly's eyes" callout points at the eye at 4.0-5.9 s. Nothing the cameras show reaches the brain except through the eye's own trace |
| controls | GAME | Same seed, scene and code. `--control off`: both decoders still read their cells and the log records every would-be stop, onset and veer, but neither is applied. `--control nowind`: `fb.wind(0, 0)` every tick (still air), decoders applied. `--control hidden` (dev only): the hazard boulders and their shadows are removed from the scene the fly's eyes trace (the camera, the collision rule and the tally keep them). Each control's view carries an amber frame and its name on a dark plate. `--apply steer` / `--apply stop` are one-decoder ablations |

### What was measured first (dev seeds >= 100)

All on the house cluster's NVIDIA B200s (torch 2.11.0+cu128), dev seeds only, 10 ms ticks, headless without the
camera (`--log-only`, the same tick as the clips). The GPU rollout is not bit-reproducible (docs/REPRODUCIBILITY.md),
so a same-seed repeat is not the same run. Logs in `out/games/mars/fix1..fix4/` (earlier builds), `own1/` (this
build's rules with the devil 10 m nearer the start and six boulders), `own2/early/` (the recorded rules and layout; the
devil drawn a little fainter, the older encounter camera), and the dev clips in `own3/` and `own4/` (below).

* **The boulders do not drive the escape pathway to the stop line.** Earlier build (eight boulders on the route, the
  autopilot driving into each): decoders off, seeds 100-103, 8 of 8 collisions in every run and the giant fibre's
  maximum 16.0 / 16.0 / 21.5 / 20.6 Hz (`fix1/lo_off_s10{0..3}.json`); with the boulders hidden from the fly's eye
  (`--control hidden`, in the next fix pass's build: `fix2/lo_hidden_s10{0,1}.json`), 18.4 and 19.9 Hz (seeds 100,
  101), so the boulders add little to the giant fibre's usual fluctuations. This build (seed 100): GF max 18.2 Hz with the boulders hidden, 18.2 / 21.4 / 17.9 / 25.0 Hz decoders
  on (seeds 100-103), 22.3 Hz decoders off, 19.7 Hz without wind; no crossing of 33 Hz in any run. The loom detectors
  (LPLC2 + LC4, population means) stayed at or below 2.4 Hz. MDN (the moonwalkers) peaked at 51-64 Hz in every run,
  wind or not, boulders or not, and drives nothing here.
* **Still air keeps the wind decoder quiet.** Without wind (`--control nowind`) the smoothed index u_s stayed within
  -3.4 to +4.1 Hz for the whole run (seed 100; the earlier build's seed 101: -3.0 to +3.8), half the 8 Hz dead band:
  no onset, no veer.
* **The devil's wind lateralises the wind DNs and the decoder veers the rover away from it.** Seeds 100-103, decoders
  on: the wind reached the rover from the left (from 90-94 deg left at the veer), DNp18 L fired 31-35 Hz against R
  0.1-3.5 Hz and DNp33 R 4-16 Hz against L 0-3 Hz, and the decoder's command passed 0.6 of its cap at 31.7-32.2 s: one
  veer per run, left, into the wind and away from D3, applied. From the route line (heading 2-4 deg at the veer,
  already turned by the decoder since its onset at 29.6-31.1 s; the controls' heading was about 0 deg then) the
  heading rose to 11.0-11.5 deg and the rover held about 3.95 m left of the line while D3 passed;
  u_s peaked at 60-62 Hz and the command sat at its 18 deg/s cap for 833-859 ticks. D3's closest approach to the
  rover's centre: **19.8 m in all four runs, 16.0 m decoders off and 16.0 m without wind** (same seed 100). The
  decoder-off run's wind DNs reported the same wind (a would-be veer at 32.41 s, logged, not applied), and its rover
  drove the line.
* **The hold rule hardly bites now.** In each decoder-on run the decoder commanded a turn during one collision
  recovery (at H3, after the devil had passed): 71-151 held ticks withholding 1.6-3.8 deg of turn in all. The earlier
  build, with the decoder's cap above the autopilot's limit and no hold, ground the rover into its last boulder on
  seeds 102 and 103 (Design).
* **The boulder tally does not depend on the fly here.** All eight of own2's seed-100-103 runs (decoders on, off,
  no wind, boulders hidden) passed the four boulders after one collision each (4 collisions, 0 clean): the rover is
  back on the line before H3. `own1`'s layout (a third boulder at 46 m and the post-devil boulders at 120-146 m) let
  the veer's offset carry the decoder-on rover past a boulder clean (2 clean vs 1 in both controls), a side effect of
  the veer, not avoidance; the recorded layout moves the boulders clear of it.
* **The rover-style manoeuvre gets round a rock in one contact.** Every collision in own2 was a single contact. With
  the previous pure-pursuit detour, stub runs (no brain) needed up to 5 contacts for a rock met at an angle, and a
  dev layout deadlocked against one rock (2,270 contacts in a stub run) when backing up while turning.
* **The eye's CUDA graph.** Checked in every GPU run against the eager trace at the first tick: max abs difference
  0.0 (`meta.eye_cuda_graph`). Same seed, graph vs eager: 803 vs 1,073 s of wall for 65.6 s of brain on a shared
  B200, with the same outcomes (a veer at 32.15 vs 32.08 s, D3 at 19.82 vs 19.83 m).
* **Dev clips.** `own3/dev_s101.mp4` (full resolution, decoders on; with `dev_nowind_s102_lowres.mp4` and
  `dev_off_s103_lowres.mp4`) had the same outcomes (a veer at 31.78 s, D3 at 19.84 m; the controls 16.0 m), but its
  frames showed the encounter camera losing the rover once the devil was behind it and the chase camera swinging with
  the rover's pivots. The cameras were reworked (the rows above) and `own4/dev_s101.mp4` (`games/mars.py` sha256
  `7fdbfdc8e73d61ee`; with `dev_off_s102_lowres.mp4`) kept the rover whole and in frame throughout (a veer at
  31.75 s, D3 at 19.8 m; decoder off 16.0 m). The recorded build (`49d2e06d3e20c38b`) differs from own4's only in the
  HUD: the WIND VEERS tile counts applied veers (it counted every small wind-steer onset) and the index label got a
  backing plate.

### Limits

- **The fly's escape pathway does not answer the boulders enough to stop the rover.** Nothing is rescaled: from 2.5 m
  up the fly sees a boulder 2.5-2.7 m across and 1.1-1.3 m tall as a slow loom in its lower field (one column at 8 m, 30
  at 2 m), far from the small, fast looms its escape pathway answers in `swat`. The giant fibre never reached the stop line, so every hazard collision
  in these clips is the blind autopilot's, and the manoeuvre round the rock is GAME intelligence.
- **The veer is the decoder's, and its size is set by GAME rules.** The raw model turns nothing. The decoder turns the
  rover towards the side its wind DNs report, capped at 18 deg/s, and the autopilot, whose limit is higher, pulls back
  towards the route; together they hold the rover a few metres off the line while the wind lasts. In the recordings the
  command sat at its cap from about 32.6 s to 41-41.6 s, so the offset is the GAME balance point, where the
  autopilot's pursuit (1.0 /s per rad towards a point 12 m ahead on the line) cancels 18 deg/s: 12 m x tan 18 deg =
  3.9 m (3.95 m logged in all three seeds). Once the index is more than 12 Hz past the dead band, larger DN rates do
  not change it. The fly's DNs set the veer's side, when it starts and how long it lasts. The autopilot took back
  almost all of the decoder's 181-185 deg of applied turn (the heading never passed +11.4 deg). A stronger cap or a
  weaker autopilot would swing it further, and a cap above the autopilot's limit let the wind spin the rover into a
  boulder (Design). The rule that holds the decoder during a collision recovery is also GAME, and every tick it held
  is counted in the log.
- **"Away from the dust devil" follows from GAME choices.** The devil is on the right and turns counter-clockwise, so
  its inward-spiralling surface wind reaches the rover from the left while it approaches and passes, and "into the
  wind" is "away". With the devil on the left and turning clockwise (an earlier build), the wind after the pass came
  from ahead-left, the index's frontal bias (it reads "left" for a wind from dead ahead; hairdryer: null ~15 deg right
  of ahead) added to it, and the same decoder turned the rover towards the devil's side for a long stretch. This
  layout was chosen on dev seeds for a clean veer and is the same for every seed and arm.
- **The wind is a GAME model, and so is its scale.** The vortex, its spin and its inward spiral are chosen, not
  measured on Mars. The antennae get the Earth wind of the same dynamic pressure, which assumes that the shipped
  antenna model's deflection scales with dynamic pressure; they do not get the rover's own motion.
- **The rover is 40x too fast** for a real rover (Perseverance's top speed is about 0.12 m/s), to make a clip.
- **The GPU rollout is not bit-reproducible** (docs/REPRODUCIBILITY.md): the same seed and code give slightly different
  rates, so a seed's control is a different sample of the same brain, not the same run with one input removed. (The
  six controls' rover paths are nevertheless identical to the log's precision, because nothing from the brain reached
  the rover in them.)
- The run log's `commit` is null on the cluster (the run directory is not a git checkout); the code is identified by
  the `sources` hashes, and the device by `device_name`.
- Build history: the first build put the eye 9 cm above the camera head (421 columns saw the rover) and had a
  loom-steer decoder (LPLC2 + LC4 left minus right, biased to the right whichever side the boulder was on) and eight
  boulders; the fix passes moved the eye onto the post, removed that decoder, added the dust devil's wind and the
  wind-steer decoder, and this pass (the recorded build) moved the devil to the right with a counter-clockwise spin,
  put the decoder's cap below the autopilot's limit, added the hold during collision recoveries and the rover-style
  manoeuvre round a rock, cut the route's boulders to four, and reworked the dust devil's look, the cameras and the
  HUD.

## Clips

Nine clips from one game file and one batch: the three recorded seeds (0, 1, 2), each in three arms, all on the
house cluster's NVIDIA B200s. **Comparisons between the arms are made within this one batch and device model.**

| | |
|---|---|
| clips | `out/games/mars/seed{0,1,2}.mp4` (decoders on), `nowind_seed{0,1,2}.mp4` (`--control nowind`), `off_seed{0,1,2}.mp4` (`--control off`), each with its run log `<clip>.json` and console `<clip>.console.txt`; 1920x1080, 50 fps, 3,279-3,283 frames (65.58-65.66 s: every run reached the waypoint and ended there) |
| code | `games/mars.py` sha256 `49d2e06d3e20c38b`, `games/common.py` `01e0fb9cf3178c67` (first 16 hex, the logs' `sources`; the same in all nine), `games/mars_assets/rec_lock.sh` `2f1ea5ec703e490f`. `commit` is null: the cluster runs a copy of the working tree, not a git checkout. After the recordings two code comments in `games/mars.py` were corrected (the columns that see the rover: 71, as measured above, not 57; D3's open stretch: between H2 and H3, not H3 and H4). That is a comment-only change (the parsed code is identical), so the file's sha256 is now `3e010fae0791358f`; the recorded runs used `49d2e06d3e20c38b` |
| device | NVIDIA B200 (`device_name` in every log), torch 2.11.0+cu128, Python 3.12.3; the eye's CUDA graph checked against the eager trace at the first tick of every run: max abs difference 0.0 |
| model | every log: MaleCNS v1.0, 167,106 neurons, preset `raw`, no instruments, two attached modules (`wind_steer`, `hazard_stop`, both kind `decoder`); `meta.declared` lists the GAME rules (9 in the decoder-on arm, 10 in each control: the control's own rule) |
| recorded | 2026-09-29, started 06:37:35-06:37:37 UTC, all nine side by side; wall 4,126-4,166 s for seven clips and 1,183-1,184 s for `nowind_seed2` and `off_seed2`. The logs record the device (NVIDIA B200) but not the GPU index or its load, and every lock's `owner.txt` names the same node (`<cluster-node>`). No file here explains the 3.5x spread in wall time. The owner reports, from the scheduler's job records (not in the repository), that the seven slower jobs shared one GPU with another project's model server and the two faster ones ran on another. Recording is offline, so the load changed only the wall time |
| batch | one `scripts/cluster_run.py` call (run `games-mars-rec-261b3d`, target `house`, the only enabled one, so no `--arm-block` was needed), nine jobs, console log `out/games/mars/rec_cluster.log`: "9 job(s), 0 failed (74.0 min)". Commands in `out/games/mars/rec_cluster.commands.txt` |
| once each | each clip ran under an ownership lock (`games/mars_assets/rec_lock.sh`: `mkdir out/games/mars/.own_<clip>`; a retry of a preempted job waits for the owner's run log instead of writing a second copy, and takes over only if the owner's console has been silent for 20 min). Every lock's `owner.txt` has one line: each clip was recorded exactly once, no job was preempted, and nothing was re-recorded |

Each job activated the cluster's venv, asserted `torch.cuda.is_available()`, and ran from the repo root:

    bash games/mars_assets/rec_lock.sh out/games/mars seed{k}        --seed {k} --seconds 100
    bash games/mars_assets/rec_lock.sh out/games/mars nowind_seed{k} --seed {k} --control nowind --seconds 100
    bash games/mars_assets/rec_lock.sh out/games/mars off_seed{k}    --seed {k} --control off --seconds 100

which runs `python games/mars.py --seed {k} [--control ...] --seconds 100 --record out/games/mars/<clip>.mp4` with its
console in `out/games/mars/<clip>.console.txt`. `--seconds 100` is a ceiling; the run ends when the rover passes the
waypoint.

What the run logs say (times in brain seconds; `games/mars_assets/summarize.py --table` prints these rows from the
logs, with the arm labels and a few cells shortened here; "max heading and left offset" is over the stretch from 30 s,
when every rover is within 0.06 m of the route line after H2, to its H3 collision, and the controls' +2.0 deg is their
heading at 30 s as they close on the line; "D3 closest" is the rover's centre to the devil's centre; "held" counts the ticks, collision
recoveries and degrees of turn that the arbitration rule withheld; "cap ticks" are ticks on which the decoder's own
18 deg/s cap clipped its command):

| clip | arm | brain / wall s | finished | boulders passed | collisions | veers | DN rates at the first veer (L / R, Hz) | max heading and left offset, 30 s to the H3 collision | D3 closest | held | cap ticks | GF max, crossings | u_s range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed0 | decoders on | 65.6 / 4,126 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | 32.26 s left, away from D3 | DNp18 35 / 7, DNp33 0 / 26 | +11.4 deg, +3.95 m | 19.9 m at 38.1 s (rover y +3.84 m) | 162 ticks, 1, 3.7 deg | 883 | 18.8 Hz, 0 | -12.4 to 57.8 Hz |
| seed1 | decoders on | 65.6 / 4,157 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | 32.29 s left, away from D3 | DNp18 24 / 2, DNp33 0 / 12 | +11.2 deg, +3.95 m | 19.8 m at 38.1 s (y +3.83 m) | 72 ticks, 1, 1.0 deg | 843 | 17.3 Hz, 0 | -10.2 to 58.1 Hz |
| seed2 | decoders on | 65.6 / 4,166 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | 31.95 s left, away from D3 | DNp18 34 / 0, DNp33 0 / 20 | +11.2 deg, +3.95 m | 19.9 m at 38.1 s (y +3.86 m) | 138 ticks, 1, 2.6 deg | 853 | 15.8 Hz, 0 | -11.6 to 62.0 Hz |
| nowind_seed0 | no wind | 65.7 / 4,165 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | none | - | +2.0 deg, +0.08 m | 16.0 m at 38.2 s (y 0.00 m) | 0 | 0 | 15.9 Hz, 0 | -3.4 to 4.2 Hz |
| nowind_seed1 | no wind | 65.7 / 4,156 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | none | - | +2.0 deg, +0.08 m | 16.0 m at 38.2 s (y 0.00 m) | 0 | 0 | 18.6 Hz, 0 | -2.2 to 3.9 Hz |
| nowind_seed2 | no wind | 65.7 / 1,184 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | none | - | +2.0 deg, +0.08 m | 16.0 m at 38.2 s (y 0.00 m) | 0 | 0 | 18.0 Hz, 0 | -2.5 to 4.0 Hz |
| off_seed0 | decoders off | 65.7 / 4,157 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | 31.80 s left, not applied, away from D3; 50.57 s left, not applied, away from D3 | DNp18 21 / 0, DNp33 0 / 25 | +2.0 deg, +0.08 m | 16.0 m at 38.2 s (y 0.00 m) | 0 | 827 | 15.7 Hz, 0 | -7.5 to 60.0 Hz |
| off_seed1 | decoders off | 65.7 / 4,162 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | 32.35 s left, not applied, away from D3 | DNp18 37 / 9, DNp33 0 / 16 | +2.0 deg, +0.08 m | 16.0 m at 38.2 s (y 0.00 m) | 0 | 822 | 15.9 Hz, 0 | -8.1 to 59.5 Hz |
| off_seed2 | decoders off | 65.7 / 1,183 | yes | 4 of 4, 0 clean, 4 after a collision | 4 | 32.10 s left, not applied, away from D3 | DNp18 25 / 6, DNp33 0 / 20 | +2.0 deg, +0.08 m | 16.0 m at 38.2 s (y 0.00 m) | 0 | 823 | 21.8 Hz, 0 | -8.0 to 61.3 Hz |

**Decoders on (seeds 0, 1, 2).** The rover's path is the same in every clip and every arm, to the log's precision,
until the devil's wind first moved the decoder: at 26.84 s in seed 0 (0.05 deg of turn, the devil 52 m away), at
29.92 / 31.26 s in seeds 1 and 2. Nothing else the brain did reached the rover. The rover waited 1.0 s, set off, hit H1 at 6.10 s (5.0 m/s; giant
fibre 0.3 / 4.9 / 0.6 Hz), backed off, pivoted, went round and hit H2 at 16.14 s coming back to the line (GF 0.2 /
2.1 / 5.3 Hz), and passed it at 24.22 s. From 28.7 s D3 was within 45 m. Its wind reached the rover from the left
(89-91 deg left at the veer, 1.5-1.7 m/s, i.e. 0.19-0.21 m/s Earth-equivalent) and the wind DNs lateralised: at the
veer (the decoder's command passing 11 deg/s) DNp18 L 35.2 / 23.9 / 34.3 Hz against R 6.8 / 2.4 / 0.5 Hz and DNp33 R
26.0 / 11.5 / 19.8 Hz against L 0 Hz, u_s +15.4 / +15.2 / +15.6 Hz. One veer per run, at 32.26 / 32.29 / 31.95 s:
left, into the wind, away from D3 (then 31.0 / 30.9 / 32.2 m away, 36 / 36 / 34 deg to the right), applied, and
bannered. The decoder had been turning the rover left since its onset at 30.49 / 29.92 / 31.26 s, so the rover's heading
went from +4.7 / +4.2 / +3.8 deg at the veer (the controls': -0.1 to +0.1 deg) to +10.6 / +10.6 / +11.0 deg 2 s later
(peak +11.4 / +11.2 / +11.2 deg), and the rover moved left of the route line, holding about 3.95 m off it with the
decoder at its cap while the devil passed: the command sat at 18 deg/s for 843-883 ticks and u_s peaked at 57.8 /
58.1 / 62.0 Hz. At D3's closest approach (38.1 s, 19.9 / 19.8 / 19.9 m) the wind at the rover was 4.5 m/s from 35-36
deg left and DNp18 L fired 90 / 84 / 70 Hz against R 13 / 10 / 4 Hz. After the pass the wind came from ahead and then
ahead-right, the command fell back under the dead band (by about 42.3 s), and the autopilot brought the rover back
towards the line before H3 (its heading down to -11.4 / -10.9 / -11.1 deg on the way). The decoder's applied turn totalled 183.5 / 181.1 /
184.6 deg, all of it to the left and all but at most 0.05 deg of it while D3 was within 45 m; the autopilot took
nearly all of it back. The rover then hit H3 at 46.72 s (0.4 m left of the line, so the GAME manoeuvre took it round
on the left, where the controls went round on the right) and H4 at 57.18 / 57.21 / 57.20 s, passed H4 at 64.70-64.73 s and reached the waypoint at
65.58-65.60 s. The hold rule withheld one small turn per run, during the H3 recovery (a right turn of 0.4 / 0.3 /
0.0 deg/s when the hold began; 3.7 / 1.0 / 2.6 deg in all over 162 / 72 / 138 ticks). No hazard stop: the giant fibre
peaked at 18.8 / 17.3 / 15.8 Hz, the loom detectors (LPLC2 + LC4 means) at 2.5 Hz or less. The HUD's WIND VEERS tile
ends at 1 in each.

**No wind (the same seeds).** `fb.wind(0, 0)` every tick; the devil's wind still blew at the rover in the GAME (up
to 6.8 m/s at its closest approach) but did not reach the antennae. The wind DNs stayed near baseline (at the moment
of closest approach DNp18 3 / 3 / 7 Hz on the left), u_s stayed within -3.4 to +4.2 Hz, under the 8 Hz dead band, and
the decoder never moved: no onset, no veer. The rover drove the route line (from 32 s to its H3 collision its heading
stayed within -0.26 to +0.03 deg and its centre within 0.08 m of the line), D3 came within 16.0 m at 38.2 s, and the rover hit H3 at 46.61 s and
H4 at 56.90 s and reached the waypoint at 65.66 s. GF max 15.9 / 18.6 / 18.0 Hz, no crossing.

**Decoders off (the same seeds).** The wind reached the antennae as in the decoder-on runs until those began to turn
(26.8-31.3 s; from then on the controls, nearer the devil, got more of it), and the wind DNs reported it: a would-be veer, left, away from D3, at 31.80 / 32.35 / 32.10 s (DNp18 L 21 / 37 / 25 Hz
against R 0 / 9 / 6 Hz), logged and bannered "NOT APPLIED". With nothing applied the rover drove the line, D3 came
within 16.0 m, the wind at the rover reached 6.8 m/s (from 35 deg left) and DNp18 L fired 83 / 77 / 79 Hz against R
18 / 8 / 10 Hz at that moment. Seed 0 logged a second would-be veer at 50.57 s, left, while the rover was going round
H3 (the wind then 22 deg left of its heading); not applied. Same collisions and finish as without wind; GF max
15.7 / 15.9 / 21.8 Hz, no crossing.

**The comparison, within this batch.** In all three seeds the decoder-on rover passed D3 3.84-3.86 m further away
(19.84-19.86 m against 16.00 m in all six controls) and 2.3 m/s less wind reached it at the closest approach (4.5 against
6.8 m/s). The size of that margin is the GAME balance of the capped command against the autopilot (Limits): the
fly's wind DNs supplied the side, the start and the duration, and the three seeds agree to 0.02 m because the command
was clipped. The decoder-off controls show that the wind DNs reported the devil's wind as strongly without the decoder
(the same veer, logged, within 0.5 s of the decoder-on runs'); the no-wind controls show that without the wind at the
antennae those cells, and so the decoder, stay quiet. The boulder tally (4 of 4 passed after one collision each, 0
clean) and the finish (65.6-65.7 s) did not depend on the fly in this layout, and the giant fibre stopped nothing in
any arm. In all nine clips MDN (logged only; it drives nothing) peaked at 53.9-66.6 Hz, far above the shipped
`Locomotion`'s 15 Hz "back up" line, and the loom detectors (LPLC2 + LC4 means) at 2.6 Hz or less.

**Trailer windows** (clip seconds = brain seconds - 0.02; seed 0 unless named):

- `seed0.mp4` 0.0-3.2 s: the opening crane shot, dollying in and rising: the pale sun upper left, a distant dust
  devil (D1, about 550 m away) against the crater rim, the rover whole in frame starting to roll at 1.0 s, the title plate "A JEZERO-LIKE
  CRATER, MARS · a GAME scene, not survey data" lower left (fades out 3.0-3.8 s). Strong (the F-major tutti).
- `seed0.mp4` 6.0-12.0 s: COLLISION · H1 ("giant fibre 0 Hz: below the 33 Hz stop line"), then the rover backs off,
  pivots in place beside the boulder and drives round it; the camera holds steady. At 6.0 s the rover is already at
  the rock (contact at 6.10 s brain); the approach is in the 4.0-5.9 s "the fly's eyes" callout. Medium: an honest
  non-event of the escape pathway, good for a quick cut.
- `seed0.mp4` 31.2-36.2 s: the dust devil's column ahead-right, the wind arrow swinging in from the left, DNp18 L
  climbing on the wind-DN panel, the banner "WIND DNs -> VEER LEFT · DNp18 L 35 · R 7 Hz ... away from the dust
  devil" (CONNECTOME + DECODER chips) at 32.24 s, and the rover's nose turning left in a camera that does not turn
  with it. The WIND STEER panel already reads "L 4 deg/s" at 31.2 s (the decoder's onset was at 30.49 s), and the
  "DUST DEVIL · 28 m" label overlaps the fading banner at about 33.2 s. Strongest: the fly's beat.
- `seed0.mp4` 36.5-39.5 s with `off_seed0.mp4` at the same clip seconds (split screen): abeam of the devil, the
  decoder-on rover 3.8 m left of the line with D3 at 19.9 m and DNp18 L near 90 Hz, the control driving the line at
  16.0 m with its wind DNs just as lateralised and its panel reading "not applied". Medium: the numbers carry the
  comparison (the minimap trails, the wind tile's 4.1-4.5 against 5.7-6.8 m/s, "1" against "1 not applied"), but in
  frame the two rovers sit in nearly the same place and the devil is at or past the right edge (the encounter camera
  keeps within 20 deg of the rover).
- `seed0.mp4` 63.5-65.5 s: beside and then past H4 (the tally turns to "HAZARDS PASSED 4 / 4" at 64.68 s; 3 / 4
  before), 8 m to 0 m to the waypoint. Weak; an end beat.

Frames were extracted locally from all nine clips at 0.3, 2.5 s, each clip's first collision (+0.2 and +3.5 s), its
first veer (+0.1, +2, +4 s), D3's closest approach, the H3 collision and the last half second, and looked at: the
rover is whole and in frame in every one, the controls carry their amber frame and plate, and the banners, chips and
panels read as described.

## Corrections (caption skeptic)

Checked against all nine run logs (`out/games/mars/{seed,nowind_seed,off_seed}{0,1,2}.json`: meta, summaries, events
and the 10 Hz series), their consoles, `rec_cluster.log`, `rec_cluster.commands.txt`, the lock files, the dev logs in
`fix1`-`fix4` and `own1`-`own4`, `games/mars.py` (sha256 prefix `49d2e06d3e20c38b`, matching the logs) and
`games/mars_assets/summarize.py`, and frames from `seed0.mp4`, `off_seed0.mp4` and `nowind_seed0.mp4`. No fly brain
was run. The geometry rows were recomputed on the CPU from the connectome's retina and the game's scene and rover:
71 / 89 columns on the rover and 421 for the first build's eye reproduce exactly; the H1 fill came out
1 / 1 / 7 / 17 / 29 / 64 / 121 columns against the caption's 1 / 1 / 7 / 16 / 30 / 63 / 118 (a different reading of
"from the footprint's front"; left as written). Changes:

1. Headline: "a declared decoder swerves the rover away from the devil" -> "a declared decoder veers the rover into the
   wind, which the game set up to be away from the devil". The decoder turns into the wind. "Away" follows from the GAME
   devil's side, spin and inflow (Limits), and the turn is a veer of about 7 deg and 4 m, not a swerve.
2. Headline paragraph: "the devil's wind reached the rover from the left, DNp18 fired 24-35 Hz ..., and at 31.95-32.29 s
   the decoder turned the rover left" -> "... from the left on the approach. The decoder began turning the rover left at
   29.9-31.3 s (seed 0 also by 0.05 deg at 26.8 s), and its command passed the log's veer mark (0.6 of its 18 deg/s cap)
   at 31.95-32.29 s, when DNp18 fired ...". The logged veer is a threshold crossing 0.7-2.4 s after the decoder's
   onsets (`wind_steer_on` at 30.49 / 29.92 / 31.26 s). The 24-35 / 0-7 Hz are the rates at that moment. After the pass
   the wind came from ahead-right.
3. Headline paragraph, added: "That offset is set by GAME numbers: the decoder sat at its 18 deg/s cap, and the
   autopilot's pursuit cancels 18 deg/s at 12 m x tan 18 deg = 3.9 m off the line." The command was clipped for 843-883
   ticks, and the logged 3.95 m (all three seeds) is that balance point.
4. Headline paragraph: "the decoders disconnected while the wind neurons reported the same wind" -> "... reported the
   devil's wind as strongly". The controls' rovers stayed nearer the devil and got more wind (6.8 against 4.5 m/s at
   the closest approach).
5. Headline paragraph: "The escape pathway did not see the boulders coming: the blind GAME autopilot drove into all four
   in every clip, and the giant fibre peaked at 15.7-21.8 Hz" -> "The escape pathway never flagged a boulder: ... in every
   clip (GAME rules took it round each in one contact), and the giant fibre peaked at 15.7-21.8 Hz, under its 33 Hz stop
   line (0-5.3 Hz at the moments of contact)". The fly's eyes do get the boulders (up to about 120 columns at contact).
   The run peaks are not at the contacts.
6. DECODER row: "added to the autopilot's" -> "capped at 18 deg/s (below the autopilot's 24 deg/s limit) and added to
   the autopilot's while the rover drives forward (a GAME rule holds it during collision recoveries)". Added to
   hazard_stop: "it never fired".
7. "not claimed" row: "see hazards" -> "detect hazards". "The raw model does not turn on its own (README), so every turn
   beyond the autopilot's is the decoder's" -> "The rover's yaw has three sources in the code: the GAME autopilot, the
   GAME manoeuvre round a rock it hit, and the decoder's applied command. Nothing else from the brain reaches it (the
   fly's own motor output is not read, ...)". The old wording left out the GAME manoeuvre's pivots and cited the wrong
   reason. "'Away from the dust devil' is where the devil's GAME wind came from on the approach" -> "'Away from the dust
   devil' holds because the GAME devil's wind reached the rover from the side away from the devil on the approach".
8. Design: "It never fired on a boulder in any run" -> "It never fired in any run" (`gf_crossings` 0 in all nine logs
   and in every dev log).
9. Design: "(14 contacts in all)" -> "(14 contacts in each run)". `fix2/lo_on_s10{2,3}.json` log 14 contacts each, 9 of
   them on H8.
10. Measured first: "with the boulders hidden from the fly's eye (`--control hidden`), 18.4 and 19.9 Hz" -> adds "in the
    next fix pass's build: `fix2/lo_hidden_s10{0,1}.json`". Those logs are not `fix1`'s build (`bf1ae1620399e815` vs
    `23a7a78c601f72df`).
11. Measured first: "From the route line (heading 3-5 deg, returning from the H2 detour)" -> "From the route line (heading
    2-4 deg at the veer, already turned by the decoder since its onset at 29.6-31.1 s; the controls' heading was about 0
    deg then)". own2's series: 1.8-4.3 deg at the veer, against -0.1 deg in its controls.
12. Limits: "The fly does not see the boulders coming, at least not enough. ... sees a boulder a metre across" -> "The
    fly's escape pathway does not answer the boulders enough to stop the rover. ... sees a boulder 2.5-2.7 m across and
    1.1-1.3 m tall". The logged hazards have ground radii of 1.26-1.36 m and tops 1.13-1.29 m above the ground.
13. Limits, veer bullet, added: the command sat at its cap from about 32.6 s to 41-41.6 s. "12 m x tan 18 deg = 3.9 m
    (3.95 m logged in all three seeds)". Larger DN rates do not change it once the index is more than 12 Hz past the
    dead band. The DNs set the side, the start and the duration. The autopilot took back almost all of the decoder's
    181-185 deg of applied turn.
14. Limits, reproducibility bullet, added: "(The six controls' rover paths are nevertheless identical to the log's
    precision, because nothing from the brain reached the rover in them.)"
15. Clips, recorded row: "the scheduler's job records put the seven slower jobs on one GPU shared with another
    project's model server and the two faster ones on another" -> "The logs record the device (NVIDIA B200) but not the
    GPU index or its load, and every lock's `owner.txt` names the same node (`<cluster-node>`). No file here explains the 3.5x
    spread in wall time. The owner reports, from the scheduler's job records (not in the repository), ...". No log,
    console or `rec_cluster.log` records a GPU assignment.
16. Clips, table preamble: "`summarize.py --table` prints this table" -> "prints these rows ..., with the arm labels and
    a few cells shortened here". "30 s, when every rover is back on the route line after H2" -> "when every rover is
    within 0.06 m of the route line after H2, ... and the controls' +2.0 deg is their heading at 30 s as they close on
    the line".
17. Clips table, `off_seed0`: "50.57 s left, not applied" -> "50.57 s left, not applied, away from D3", as the log and
    the script have it.
18. Decoders on: "The rover's path over the first 28 s is the same in every clip and every arm, because nothing the brain
    did reached the rover before the devil's wind" -> "The rover's path is the same in every clip and every arm, to the
    log's precision, until the devil's wind first moved the decoder: at 26.84 s in seed 0 (0.05 deg of turn, the devil
    52 m away), at 29.92 / 31.26 s in seeds 1 and 2". Seed 0's path leaves the controls' at 26.9 s.
19. Decoders on: "The rover's heading went from +4.7 / +4.2 / +3.8 deg at the veer" -> prefixed "The decoder had been
    turning the rover left since its onset at 30.49 / 29.92 / 31.26 s, so ..." and added "(the controls': -0.1 to +0.1
    deg)".
20. Decoders on: "the command fell back under the dead band, and the autopilot brought the rover back to the line
    before H3" -> "... under the dead band (by about 42.3 s), and the autopilot brought the rover back towards the line
    before H3". Added "the autopilot took nearly all of it back" and, at H3, "(0.4 m left of the line, so the GAME
    manoeuvre took it round on the left, where the controls went round on the right)". H3's `detour_done` is at y +3.71
    m in the decoder-on runs and -3.31 m in the controls.
21. Decoders off: "The wind reached the antennae exactly as in the decoder-on runs up to the veer" -> "... as in the
    decoder-on runs until those began to turn (26.8-31.3 s; from then on the controls, nearer the devil, got more of
    it)".
22. Comparison: "(the same veer, logged, at the same moment)" -> "(the same veer, logged, within 0.5 s of the decoder-on
    runs')". Those veers were at 31.80 / 32.35 / 32.10 s against 32.26 / 32.29 / 31.95 s. Added the GAME share of the
    3.85 m margin, and an outcome left out before: "In all nine clips MDN (logged only; it drives nothing) peaked at
    53.9-66.6 Hz, far above the shipped `Locomotion`'s 15 Hz 'back up' line, and the loom detectors (LPLC2 + LC4 means)
    at 2.6 Hz or less".
23. Trailer, opening: "a distant dust devil (D1) over the crater rim" -> "(D1, about 550 m away) against the crater rim".
    The column in the frame lies at azimuth 39 deg from the camera, and D1 is at 38 deg. D1 is about 1 km inside the
    rim.
24. Trailer, H1 window: added "At 6.0 s the rover is already at the rock (contact at 6.10 s brain); the approach is in
    the 4.0-5.9 s 'the fly's eyes' callout."
25. Trailer, veer window: added "The WIND STEER panel already reads 'L 4 deg/s' at 31.2 s (the decoder's onset was at
    30.49 s), and the 'DUST DEVIL · 28 m' label overlaps the fading banner at about 33.2 s."
26. Trailer, split screen: "Strong for the comparison" -> "Medium: the numbers carry the comparison (...), but in frame
    the two rovers sit in nearly the same place and the devil is at or past the right edge".
27. Trailer, end beat: "63.5-65.5 s: past H4, 'HAZARDS PASSED 4 / 4', 2 m to the waypoint" -> "beside and then past H4
    (the tally turns to 'HAZARDS PASSED 4 / 4' at 64.68 s; 3 / 4 before), 8 m to 0 m to the waypoint". The frame at
    63.5 s shows 3 / 4 and 8 m.
