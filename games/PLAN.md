# games/ -- the build plan for the 0.2.0 announcement

Nine environments that put the shipped fly (MaleCNS v1.0, `preset="raw"`) somewhere it has never been. Each shows
off a different part of the control surface (docs/CONTROL_SURFACE.md, docs/EXTENSIBILITY.md). Their clips cut together
into a trailer on Kevin MacLeod's "Also Sprach Zarathustra (Sonnenaufgang)" (CC BY 3.0) and an X thread. This file
is the spec every builder, reviewer and recorder works from. `games/common.py` is the shared plumbing: read it
first. Treat it as frozen. If you need a change, put the helper in your own file and name it under
`common_change_requests` in your result.

## Ground rules (the repository's own, applied to games)

1. **The model is used as shipped.** Do not edit anything under `flyverse/`, `scripts/`, `tests/` (except adding a
   `tests/test_games_<name>.py`), `cache/`, the defaults or the weights. Talk to the brain only through the public
   surface: `fb.vision / smell / wind / taste / stimulate / step`, `fb.motor()`, attached modules, and read-only
   access to state (`fb.brain.rate`, `fb.optic.rates()` / `fb.optic.last["dr"]`). `preset="raw"` everywhere: no
   instruments.
2. **Every mapping from neural activity to a game control is a declared decoder.** Use `games.common.ReadDecoder`
   (attached with `fb.attach`, `kind="decoder"`, writes nothing, so the brain is numerically unaffected) or, when
   you read state directly for speed, `games.common.declare(log, name, "decoder", law, reads, **params)`. Every
   non-fly influence is declared `kind="game"`: autopilot, scripted opponents, physics, a `stimulate()` pulse.
   The HUD labels every on-screen number with its chip: CONNECTOME (a rate read from named cells), DECODER, or
   GAME.
3. **Know where the raw model stops** (README, "What the raw model does unprompted, and where it stops"). It escapes
   looms (LC4 / LPLC2 -> giant fibre DNp01, 33 Hz body threshold). Its T4/T5 are direction selective. Its wind DNs
   lateralise hard (DNp18 about +45 Hz on the wind side, DNp33 about -50 Hz on the other). Sugar drives MN9 and
   bitter shuts it off. It walks, straight. It does **not** turn on its own (DNa02 is held below threshold), has no
   flight state without stimulation, does not pass small objects to LC10a / LC11, holds no compass bump and does
   no directed food search. A game may give the fly a decoder that turns or steers, and must say that it did. It
   may not imply the fly does these things natively.
4. **Tune on dev seeds, never on the recorded ones.** Decoder gains, thresholds and smoothing may be tuned for
   playability using seeds >= 100 only. Their final values are declared (in the decoder's `parameters` and the
   caption).
5. **Recording policy.** Once the code is frozen, record **seeds 0, 1 and 2** exactly once each, all with the same
   command line. Keep all three. A caption reports what each run's log says happened, including misses, splats,
   deaths and non-events. There are no reruns for a better outcome. If a run exposes a software bug, fix it,
   re-record all three, and say so. Where a cheap control exists (decoders disconnected, same seed), record it too
   and report the comparison. The trailer may use any recorded seed; games/README.md will say which.
6. **Language.** Say "the paddle follows the decoder's reading of the fly's L2 columns", not "the fly plays". Headlines
   may be playful ("the first game of Pong between two connectomes"). The caption under a headline is exact:
   numbers from the run log, the decoder's law, what is GAME.

## Practicalities

* **Python:** `<miniconda>/python.exe` (torch 2.10 + cu128, RTX 4090 24 GB). The repo's `.venv` has CPU
  torch; don't use it for GPU runs. Run from the repo root with `PYTHONIOENCODING=utf-8`.
  `games/common.py` puts the repo on `sys.path`. The shell tool is Git Bash on Windows; PowerShell also exists.
* **GPU etiquette:** the one GPU is shared by up to about 10 agents at once. A full brain is about 1-2 GB. Keep dev runs
  short (<= 10 s of brain time), run in the foreground with timeouts, and kill only processes you started. On CUDA OOM,
  wait about 60 s and retry. Building a `FlyBrain` takes about 7 s. Full brain alone: about 8 ms of wall per 10 ms of
  brain. A 1080p game loop with ray tracing: roughly 2-3x slower than real time. Headless recording is offline, so
  speed only affects wall time, never fidelity.
* **Reusable pieces:** `scripts/room_demo.py`'s `Sim` class (room, table, fruit, wind / odour `Air`, body `Locomotion`
  and `Flight`, loom) imports fine with `sys.path.insert(0, "scripts")`. So do `flyverse.world` (a torch spectral ray
  tracer: spheres / ellipsoids, boxes, planes, `render_camera`), `flyverse.body`, `flyverse.air`,
  `flyverse.batch_sim.BatchSim` (batched rooms, docs/BATCH_SIM.md) and `flyverse.interp.trace.column_of_cells`
  (retinotopic column of each optic cell).
* **Files:** your game is `games/<name>.py`. Put any sub-assets in `games/<name>_assets/` (Node code for Minecraft:
  `games/minecraft_bridge/`, with `node_modules/` git-ignored). Put the caption in `games/captions/<name>.md`, dev clips
  in `out/games/<name>/dev*.mp4` and final clips in `out/games/<name>/seed{0,1,2}.mp4` (+ `.json` run logs; `out/`
  is git-ignored). No git commands that change state (no commit, checkout, stash or reset). Don't edit
  `pyproject.toml`; list new dependencies in your result instead.
* **Lint:** `<miniconda>/python.exe -m ruff check games tests --select E9,F63,F7,F82` must be clean
  (CI runs this over the whole repo). An optional `tests/test_games_<name>.py` must pass with no data, no GPU and no
  network: test pure logic (physics, decoders on synthetic tensors), mark nothing, and import the game module
  without constructing a brain.
* **Canvas:** 1920x1080 at 50 fps (`common.run` handles pacing: 2 ticks of 10 ms per frame). Use `Hud.header` / `Hud.footer`
  (the footer prints each declared law). Show at least: the main view (>= 60 % of the canvas, dramatic, legible at
  phone size), **WHAT THE FLY SEES** (`Hud.mosaic` + `eye_colors` from the exact radiance handed to `fb.vision`), and
  1-3 live neural traces or bars with chips. Add big event banners the trailer can cut on (e.g. `GIANT FIBRE 41 Hz ->
  JUMP`), fading over about 0.8 s. Key numbers >= 22 px.

## The environments

Prototype measurements below are from one exploratory run each on this desktop (seed 0). They are orientation, not
results.

### 1. `swat` -- dodge the swatter (raw model, zero decoders)
The fly on the table (room_demo `Sim(fence=True)`). A swatter (a flattened black ellipsoid, radii about
(0.06, 0.045, 0.004) m, `world.Sphere(..., "black")`) approaches the fly's eye along a direction at speed v until
4 mm away. The fly escapes if the shipped body takes off (GF >= 33 Hz -> `Flight.maybe_takeoff`) before contact;
otherwise SPLAT (GAME: the trial ends, the fly respawns). Prototype: from above, 0.5 m/s escaped (GF crossed at
0.70 s of 0.79), 1.0 m/s escaped **6 ms** before contact, 2.0 m/s hit. From the left, 0.5 escaped and 1.0 hit by 14 ms.
Front-above at 1.0 escaped with 86 ms to spare. Interactive: click to aim (direction from the camera), hold to wind
up speed, release to swing; a score. Record mode: a scripted sequence of swats of rising speed and varied direction
(GAME), the same for every seed. Show a big orbit / third-person camera (`world.render_camera`), the fly's-eye
mosaic, the GF trace with its 33 Hz line, LC4 / LPLC2 mean rates, TTMn, and a result banner with the margin in ms.
Optional slow-motion replay of the last 400 ms before contact. Here the body model's takeoff *is* the shipped
readout, not a game decoder: say "body model (flyverse/body.py): GF >= 33 Hz -> takeoff".

### 2. `pong` -- the first game of Pong between two connectomes
MaleCNS v1.0 (male CNS) vs FlyWire FAFB v783 (female brain; complete optic lobe): `FlyBrain()` and
`FlyBrain(dataset="fafb")`. FAFB has no VNC, so `fb.motor()` raises; Pong needs only the eyes. Each fly faces the
court from its own end. The court spans about +-50 deg azimuth and +-35 deg elevation of its field (the eyes cover
azimuth -121 to +115, elevation +-75, with a frontal binocular strip). Decoder (prototype on MaleCNS): paddle target =
the court position of the peak of **high-passed L2 (lamina) and/or Mi1 (medulla)** activity. Per cell: |dr - EMA(dr,
tau 0.3 s)|; top-12 cells weighted; each cell's retinotopic column from `interp.trace.column_of_cells`, and that
column's `col_az_el`. Correlation with the ball's true elevation was 0.82 for L2 and 0.88 for Mi1's azimuth, median
error about 9 deg (two ommatidia). The whole-population centroid of T4/T5 did **not** work (corr 0.03; activity
spreads eye-wide). Smooth the target; the paddle moves toward it at a GAME max speed. Decide, test and declare
whether the flies see the paddles (their own motion contaminates the decoder) or only the ball and court. Validate
the decoder on FAFB separately; its column map is its own. Interactive: human (W/S or mouse) vs a fly, or fly vs
fly. Record: fly vs fly, first to N points or fixed time, same serve schedule (GAME) for every seed. Show the court
big, both flies' eye mosaics with the decoded ball position overlaid, the score, rally length. Caption: descriptive;
**not** a sex comparison: one male and one female reconstruction, different coverage and naming.

### 3. `spacecraft` -- 6DOF: a fly flies a spaceship
A rigid body with 6 degrees of freedom (position, velocity, orientation quaternion, angular velocity) and thrusters.
Space scene rendered for the fly's eyes and a human camera: a starfield at infinity, a planet with a sunrise (a
2001 homage fits the music), a rotating ring station, drifting asteroids. Brain -> thrusters, all declared decoders,
each reading a population that genuinely carries that signal:
* **Attitude hold:** horizontal-system cells (HSN / HSE / HSS, H2) L-R -> yaw torque; vertical-system cells (VS /
  VSm / VST, left vs right, and combinations) -> roll and pitch torque. A starfield at infinity makes the optic flow
  purely rotational, which is exactly where these cells are clean. Measure first: the sign and size of each
  population's L-R response to imposed roll, pitch and yaw at 30-120 deg/s. Build only the axes that respond.
* **Dodge:** LPLC2 / LC4 activity split by eye side (and dorsal / ventral via their columns, if it resolves) -> a
  translation impulse away from the looming side.
* **Escape burn:** giant fibre DNp01 >= 33 Hz -> a big forward / up burn (the jump, in space).
* Cruise is a GAME autopilot, unless a DN channel honestly carries it.
Scenario: the ship starts tumbling slowly near the station while asteroids drift in. **Control run (required):** same
seed, decoders disconnected (thrusters off). Report the angular-rate decay and the asteroid outcomes in both. If
attitude hold does not beat the control, the caption says so. Show an external chase camera (main view), the
fly's-eye mosaic, HS / VS L-R traces, LPLC2 L / R, GF, and thruster indicators.

### 4. `minecraft` -- the fly in real Minecraft (owner approved: local server, eula=true)
A local **offline-mode** Minecraft Java server on localhost. Pick the newest version that mineflayer and
prismarine-viewer both support, and download the jar from Mojang's version manifest (Java 25 is installed). The
server directory lives under `out/minecraft/server/` (git-ignored). Write `eula=true` there; the owner approved it.
The fly is a **mineflayer bot**. Put the Node code in `games/minecraft_bridge/`: package.json, the bot, and a
line-delimited JSON socket or stdio bridge. `games/minecraft.py` orchestrates it:
* **Eyes:** Python fetches the blocks around the bot (e.g. a 64^3 region, refreshed as it moves) and nearby entities
  (mobs as boxes) through the bridge. It ray-marches the 1,466 x 7 ommatidial rays through the **real block grid** on
  the GPU (DDA) from the player's eye (1.62 m). Block colours come from the block's texture mean or a palette; there
  is a sky with a sun. This gives radiance with UV = 0.5 B. A declared scale: the fly's eye becomes a player's eye.
* **Body:** the shipped `Locomotion.readout` / `Flight.readout` on `fb.motor()`. Speed -> forward control (declared
  mapping; the fly's cm/s become the player's walk). Yaw rate -> `bot.look` (declared gain). GF >= 33 Hz -> jump
  (connectome -> body threshold; GAME: the jump itself is Minecraft's).
* **Lockstep:** freeze the server tick (`/tick freeze`; `/tick step 1` per 50 ms of brain; 1.20.3+) so a brain slower
  than real time stays in sync, and runs are as reproducible as the brain allows. Run commands through the server
  console or rcon.
* **Scenario:** a scenic spawn (fixed world seed), the fly walks, a mob (zombie / creeper, summoned by the GAME) walks
  at it. Does the loom reach the giant fibre? Measure it; if it does not, the caption says so. Other honest events
  (walking off a ledge, bumping a tree) are fine.
* **Footage:** real Minecraft rendering of the bot (prismarine-viewer: first person and / or third person),
  captured to frames (headless node-canvas / WebGL, or the browser viewer driven by puppeteer / playwright), and
  composited with the HUD (fly's-eye mosaic from our raycast, GF trace, speed / yaw). If no route to real Minecraft
  rendering works on this machine, fall back to our own render of the same blocks and **label it** ("rendered from
  the server's block data"). Say which one you used. Test the whole chain with short runs before the full scenario.

### 5. `doom` -- yes, it runs Doom
`pip install vizdoom` into the miniconda Python; Freedoom ships with it (BSD). A scenario where monsters approach
(`defend_the_center`, or `basic` / `deadly_corridor` if better). The fly sees the Doom frame through
`Eyes.from_pinhole` (declare the field of view). The trigger is the giant fibre: DNp01 >= 33 Hz -> ATTACK, a
connectome signal through a declared decoder. Aiming or turning, if any, is a declared decoder, e.g. LPLC2 / LC4
left-minus-right -> turn toward the looming side. Movement not driven by the brain is GAME (rails or autopilot).
Measure what the monsters' approach does to LC4 / LPLC2 / GF. Report kills, misses and deaths. Show the Doom frame
big, the fly's-eye mosaic, GF with its threshold, and a "GF -> FIRE" banner.

### 6. `hairdryer` -- herd the fly to the apple with a fan
The room (`Sim`, one apple, `--fence`). The player holds a fan. The GAME sets the `Air` wind direction and speed
from the fan's position relative to the fly. The Johnston's organ -> `fb.wind` -> wind DNs (DNp18 / DNp33, the
strongest lateralised signal in the model) -> a **declared turning decoder**, e.g. yaw = k * (wind_ipsi_L -
wind_ipsi_R). Choose and declare whether it turns up- or downwind; it is a decoder, not the shipped body readout,
which gates wind turning by odour. The goal: steer the fly onto the apple. Contact then drives the sugar GRNs ->
MN9 proboscis, a connectome signal. Show a "FEEDING: MN9 x Hz" banner when MN9 fires, and report MN9 honestly (it
has a known low draw). Record mode: a scripted fan-holder (a GAME heuristic) herds the fly. Show the room from
above, the fan and wind arrow, the fly's-eye mosaic, DNp18 L / R, the decoder's yaw, and MN9.

### 7. `swarm` -- 64 flies, one GPU
One `FlyBrain(batch=64)` (or `flyverse.batch_sim.BatchSim`). 64 flies on a big table, each with its own eyes
(batched ray tracing) and its own body. A giant swatter comes down on the crowd (GAME). Each fly's own giant fibre
decides whether it takes off. Count escapes vs splats, and report how the outcome depends on distance and angle.
Show a top-down or orbit view of the whole crowd (the main view), a grid of small GF bars or a raster of 64 GF
traces, the tally, and the brain-time vs wall-time rate ("64 brains x 167,106 neurons"). Keep batch 64 if it runs;
otherwise use the largest batch that fits, and say so.

### 8. `mars` -- the fly drives a Mars rover
A six-wheeled rover on a Martian plain (Jezero-like: butterscotch sky, a small pale sun, red regolith, dunes, a crater
rim, scattered boulders; a dust devil or two is welcome). The fly's eyes sit on the rover's mast; its 1,466 columns
see the ray-marched terrain, rendered on the GPU (a heightfield plus rocks as ellipsoids is enough). A GAME autopilot
drives toward a waypoint at rover speed. The brain supplies only what it honestly carries, as declared decoders:
* **Hazard avoidance.** A boulder looming in the path drives LPLC2 / LC4. Their left-minus-right steers away. The
  giant fibre (DNp01 >= 33 Hz) is the rover's **hazard stop** (a brake, then a reverse). MDN (the moonwalker DNs,
  which the shipped `Locomotion` already reads as "back up" above 15 Hz) is a candidate reverse channel: measure
  whether it responds before using it.
* **Heading hold on rough ground.** Optional: HS / H2 left-minus-right against terrain-induced yaw, if it measurably
  helps, as in `spacecraft`.
* **Wind.** Optional: a dust devil's wind reaches the Johnston's organ through `fb.wind` (DNp18 / DNp33), if it adds
  something honest.
Measure each mechanic first on dev seeds and build only what responds. Scenario: a fixed boulder field on the route.
Control run (required): decoders disconnected, same seed, the rover drives the autopilot line. Report hazards passed,
hazard stops, and collisions in both arms. Make it gorgeous: a wide Mars vista (the trailer's F-major tutti lands on
it), a chase camera with dust, the fly's-eye mosaic, and LPLC2 / LC4 L/R and GF traces.

### 9. `tron` -- light cycles
A TRON arena: a black floor with a glowing grid, and two light cycles that leave glowing jet walls. The fly's eyes
ride one cycle; the arena and walls are ray-traced (emissive boxes and ribbons, high contrast, ideal for a fly's
photoreceptors). Cycles move at constant speed (GAME) and turn in 90-degree steps. The fly's only control is a
declared decoder built on its escape pathway: a wall looming ahead drives LPLC2 / LC4 and the giant fibre, and when
the decoder's trigger crosses (e.g. GF >= 33 Hz, or a declared LPLC2 threshold) the cycle turns 90 degrees away from
the more-looming side (LPLC2+LC4 left-minus-right). Measure what actually crosses and when, before choosing. The
opponent: a second fly (a second MaleCNS brain on another seed, or FlyWire FAFB via `FlyBrain(dataset="fafb")`,
which has an optic lobe and DNp01; check it), or a declared GAME bot. Choose, declare and justify. Baselines (Doom
style, required): the same arena with a random-turn player at the fly's own turn rate, and with a never-turning
player; report where the fly ranks. Control: the decoder disconnected. Rounds: best of N or fixed time, with the same
arena schedule for every seed. Make it look like TRON: a chase camera low behind the cycle, bloom, the cyan-vs-orange
trails, a big "GIANT FIBRE -> TURN" banner.

## Where GPU work runs (added 2026-09-28, evening)

**Do not run fly brains on this desktop's GPU any more.** The owner needs the machine. Every GPU run (prototypes,
dev clips, recordings) goes to the cluster through `scripts/cluster_run.py`, which ships the working tree, untracked
files included, to a fresh run directory on the house B200s, submits one `<scheduler>` job per command, waits, prints the
logs and fetches outputs back. Read `docs/CLUSTER.md` sections 4, 5, 8, 13 and 14 first (it is git-ignored and local).
The recipe:

    python scripts/cluster_run.py --name <game>-<what> --target house --minutes 30 --fetch=out/games/<game>/ \
        "python -c 'import torch; assert torch.cuda.is_available()' && source .venv/bin/activate && mkdir -p out/games/<game> && python games/<game>.py --seed 0 --record out/games/<game>/seed0.mp4 ... > out/games/<game>/seed0.console.txt 2>&1"

* One command string per job; several strings in one call run as a concurrent batch. Always fetch a **named**
  directory (`out/games/<game>/`), never bare `out/`.
* The cluster venv is Python 3.12 with torch 2.11 + cu128, pygame and imageio-ffmpeg. `games/common.Recorder` falls
  back to imageio-ffmpeg's bundled ffmpeg, because there is none on PATH there. The MaleCNS cache and FlyWire FAFB's
  cache (`cache/fafb`) are present; BANC is not. There is no Java, Node or Chrome, so Minecraft does not run there.
* Read the console log's final `<n> job(s), <k> failed` line before trusting anything.
* Local work is CPU-only: tests, pure-logic checks, drawing code without a brain (`CUDA_VISIBLE_DEVICES=-1`). No local
  FlyBrain on CUDA.
* A recording's log records its device (`NVIDIA B200`) and has `commit` null or dirty (the run dir is not a git
  checkout; it is identified by `sources` hashes). Say so in the caption.

## Trailer (after the clips)

Music: `out/trailer/music/zarathustra_macleod.ogg` (Kevin MacLeod, "Also Sprach Zarathustra (Sonnenaufgang)",
CC BY 3.0, 1:26, via Wikimedia Commons). A beat map from onset analysis gives the cut points: the three rising
trumpet figures (C-G-C), the major / minor chord hits, the timpani strokes and the final C-major climax. The edit
goes: low C pedal = darkness and the connectome's numbers; the first fanfare = the first reveals; the timpani = rapid
cuts; the climax = the best moment of each environment; the end card = the repo, the command line, the credit line.
