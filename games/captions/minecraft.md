# minecraft: the fly in real Minecraft

**A simulated fly connectome (MaleCNS, 167,106 neurons) behind a Minecraft player's eyes: a scripted phantom skims the head, the giant fibre crosses its 33 Hz takeoff threshold, and the player jumps: 5 of 6 dives in seeds 0-2 (the sixth crossed too, but inside the body model's 1 s landing refractory), and 0 of 6 when the mobs are left out of the fly's eyes.**

### What is the fly and what is not

| | |
|---|---|
| **CONNECTOME** (the fly) | MaleCNS v1.0 as shipped: 167,106 neurons, preset `raw`, no instruments, nothing attached, trained or tuned for this game. Everything from the radiance handed to `fb.vision` to the rates of the giant fibre (DNp01), LPLC2, LC4, MDN and the descending and motor neurons that the body readout reads is the model. That includes the shipped gains that the README's ledger declares: LC4 / LPLC2 -> DNp01 x3 (a stop-gap magnitude for the loom escape margin), and visual projection -> descending x2 and descending -> VNC x3 (stop-gaps for per-cell-type synaptic strengths). None was set for Minecraft. The HUD's GF, LPLC2, LC4 and MDN numbers are its rates. |
| **DECODER** (declared mappings to the game's controls) | `escape_jump`: the shipped body model's takeoff rule (`Flight.maybe_takeoff`: DNp01 >= 33 Hz, and no new escape within 1 s of landing) presses Minecraft's jump key. `walk`: player speed = 215.85 x the shipped body readout's walking speed; about 92 % of that speed is the readout's constant 0.8 cm/s drive, not the brain. `turn`: the body's yaw rate, 1:1. The raw model does not turn, so the player walks almost straight. The decoders only read; they write nothing into the brain. |
| **GAME** (not the fly) | Minecraft 1.21.4 itself: the world (level seed `flyverse`), the zombies' AI, hits and knockback, the jump's physics and the Auto-Jump option. The five scripted mobs: three zombies, and two phantom dives scripted to turn 0.10 m from the eye and skim the head. The ray cast that turns the server's blocks into the fly's 1,466-column view (mobs as coloured hitbox boxes). The tick lockstep, the heals, the chase camera and the footage (prismarine-viewer). |
| **not claimed** | The fly does not play, steer, aim or dodge. The giant fibre crossed only with a scripted mob's hitbox already 0.13-0.38 m from the eye (six phantom dives at the head, and one zombie just after its second hit). The jump followed within one Minecraft tick, except once, when the body model's 1 s landing refractory, started by the landing of a GAME Auto-Jump, blocked it (no jump followed). Most of the walk is a constant. The fly sees mobs as boxes. |

## Design

The shipped fly (MaleCNS v1.0, 167,106 neurons, `preset="raw"`, no instruments, nothing attached) is the body and
eyes of a mineflayer bot on a local, offline-mode vanilla **Minecraft 1.21.4** server. The player walks at a steady
stroll, but most of that speed is the body readout's constant intrinsic drive, not the brain (see `walk`). The GAME
sends five mobs at it on a fixed schedule: three zombies under Minecraft's own AI and two scripted phantom dives.
The player jumps when the fly's giant fibre (DNp01) crosses the shipped body's takeoff threshold. Headline-safe
wording: "the player jumps when the fly's giant fibre does". Not "the fly plays Minecraft", and not "the fly
dodges": nothing steers, and the jump comes when the phantom is already at the head.

1.21.4 is the newest version both halves support: mineflayer 4.39.0 is tested up to 26.1, but prismarine-viewer
1.33.0 ships textures and block states only up to 1.21.4. The server jar comes from Mojang's version manifest
(sha1 `4707d00e...`), runs on Java 25 (Temurin 25.0.3) under `out/minecraft/server/` with `eula=true` (owner
approved), and every run deletes and regenerates the world from level seed `flyverse`, so the terrain is identical
across runs.

| what | kind | law and parameters |
|---|---|---|
| the brain | CONNECTOME | `FlyBrain(seed=...)`, preset `raw`, default execution path (see "CUDA graphs" below); nothing attached, so `fb.module_records()` is empty |
| the fly's eyes | GAME | The 1,466 x 7 ommatidial rays (`games.common.Eyes`, pooled with the retina's weights) are cast from the player's eye (feet + 1.62 m, Minecraft's standing eye height) through the **server's real block grid**: a 96 x 64 x 96 block region around the player, fetched through the bridge and fetched again when the player moves more than 20 blocks horizontally or 10 vertically. Exact voxel traversal: the tests check it against a reference 3D-DDA on 400 random rays. It runs as a CUDA graph, bit-identical to the eager path. Block faces take their 16 x 16 texel from prismarine-viewer's 1.21.4 texture atlas (plains biome tint for grass, leaves and water), with vanilla face shading (top 1.0, bottom 0.5, north/south 0.8, east/west 0.6). Leaves are opaque, their cut-out texels filled with 0.75 x the mean opaque texel. Blocks smaller than half a cube (grass tufts, flowers, torches) are **see-through** for the fly. The sky runs from zenith (0.18, 0.36, 0.95) to horizon (0.62, 0.78, 1.00) linear RGB, with a 5 deg sun disc at the world's time of day (1000: 28 deg up in the east, vanilla celestial angle). Fog runs from 28 to 60 blocks, and rays that leave the region below the horizon see distant grass. A white texel in full sun has radiance 0.42. **Living mobs and players** (minecraft-data types hostile, mob, animal, passive, water_creature, ambient, player; the 8 nearest) are **boxes of their hitbox size**, in three colour bands taken from the mean opaque texels of the textures the viewer draws (zombie legs (75, 65, 151), shirt (12, 148, 158), head (77, 111, 60); phantom (83, 82, 98)). Falling blocks, boats, projectiles and other non-living entities are left out. A box that contains the eye is not drawn (the vanilla client culls a model the camera is inside); each such episode is logged as `eye_in_mob_box` (none happened in any run with this code). RGB -> [UV, B, G, R] with UV = 0.5 B. The eye sees the world the way the vanilla client draws it: the player interpolated between its last two ticks, mobs between the last two server ticks, the scripted phantom at its scripted position (below). A declared scale: the fly's eye becomes a player's eye. The run log lists every entity name that entered the cast. |
| `walk` | DECODER | **player ground speed = 215.85 x the body model's walking speed** (m/s -> m/s: a brisk 2 cm/s fly walk = Minecraft's 4.317 m/s walk). The body speed is the shipped `Locomotion.readout` on `fb.motor()` after its 80 ms smoothing, averaged over the 5 brain frames of each Minecraft tick. That readout is **a constant 0.8 cm/s intrinsic drive** (`Locomotion.baseline_speed`, "flies walk spontaneously") **plus** forward-DN (0.05 cm/s per Hz), leg-MN (0.033 cm/s per Hz) and MDN-backing (-0.05 cm/s per Hz above 15 Hz) terms. **About 92% of the walk is the constant** (91-93% in every run): over the final dev clip the body speed had median 0.861 cm/s, of which 0.80 is the constant and 0.061 the neural terms. So the player walks at a constant 1.73 m/s plus about 0.13 m/s (median) from the brain. The brain's visible contribution is a speed-up of up to about 0.7 cm/s as a phantom passes (body speed peaked at 1.47-1.54 cm/s in the two seed-104 dev runs, 1.47-1.62 cm/s over all six dev runs of the dive table below, and 1.46-1.58 cm/s during the phantom passes of the three recorded connected runs, against at most 1.03-1.11 cm/s in the three recorded controls) and brief MDN dips to just below zero (below). Minecraft applies the speed through its own movement code: the forward (or back) key, with the `movement_speed` attribute = speed / 43.17 (vanilla: 0.1 -> 4.317 m/s); deadband 0.05 m/s. The HUD shows the body speed with a DECODER chip and splits it into "constant" and "from DN/MN rates". |
| `turn` | DECODER | **player yaw rate = the shipped body's yaw rate, 1:1 rad/s** (`Locomotion.readout`'s yaw: the high-passed optomotor L-R, DNa02 L-R and leg MN L-R terms; angles are scale-free). The raw model does not turn (README), so the player walks almost straight. |
| `escape_jump` | DECODER (the shipped body threshold) | **body model (`flyverse/body.py` `Flight.maybe_takeoff`): DNp01 mean rate >= 33 Hz, no new escape within 1 s of landing -> the jump key for the next Minecraft tick.** The jump itself is Minecraft's physics (0.42 blocks/tick upward, about 1.25 blocks high). Latency: the next Minecraft tick (at most 50 ms) when the player is on the ground; during knockback the jump waits for the ground. Nothing was tuned. The same call's voluntary branch (power MNs >= 50 Hz for 0.3 s) would also jump; the game checks the GF itself and would log and label that case separately (`voluntary_takeoff`, "WING POWER MNs -> TAKEOFF JUMP"). It never fired. |
| lockstep | GAME | The server is frozen (`/tick freeze`). Every 50 ms of brain: one client physics tick of the bot (prismarine-physics, mineflayer's port of vanilla player movement, with mineflayer's own timer disabled), then `/tick step 1` over RCON, then a wait for the game time to advance and for a chat marker that proves the bot has received that tick. Brain time is the integer tick count x 10 ms, so the script's times do not drift. The logs carry `minecraft_ticks_stepped` and `server_ticks_elapsed`. |
| zombies | GAME | Summoned at t = 1.5 s (12 m ahead), 7.5 s (10 m, 35 deg left) and 13.0 s (10 m, 35 deg right), bearings relative to the player's heading at that moment, on the ground there. They walk at the player under **Minecraft's own AI** (easy difficulty: 2.5 HP a hit). Each wears a leather helmet so it does not burn in daylight; the viewer does not draw armour. Each is removed (`/kill`, no loot) 0.6 s after its first hit (checked once per Minecraft tick, so 0.60-0.65 s in the logs), or at min(9 s, next summon - 0.4 s). |
| phantom dives | GAME | At t = 19.0 s (22 m ahead, 11 m above the eye) and 24.5 s (20 m out, 60 deg left, 9 m up). The dive is **scripted in Python and stepped every 10 ms brain frame against the eye's position in that frame** (`dive_step`), so the geometry the fly sees is exactly the script's. The phantom flies **straight at the eye at 10 m/s** until its hitbox (0.9 x 0.5 x 0.9 m) is **0.10 m from the eye**. Then it turns into a pass up and over the head (the horizontal direction + 0.8 up). During the pass it is lifted where needed so the eye stays at least 0.10 m from the hitbox (`keep_clear`): it pulls up over a jumping head instead of swallowing it. A phantom's real attack reaches the head; that is why the gap is 0.10 m. The fly's eye and the footage both take the phantom's position from the script. A NoAI, fire-resistant phantom on the server is `/tp`'d there every Minecraft tick. It never attacks and is removed 1.5 s after the pass. |
| rules | GAME | Fixed start: the grass block at (-296, 63, -294) (the log's `start`; the player's feet at y = 64), heading 157.5 deg from +x towards +z (west-south-west) along a flat lakeshore runway. Time 1000, clear weather, no daylight or weather cycle, no natural mob spawning, natural regeneration off. The player is healed to full (instant health) whenever a mob is summoned. Minecraft's **Auto-Jump** option is on: walking into a 1-block step with two blocks of air above jumps. |
| footage | GAME (display) | **Real Minecraft rendering**: prismarine-viewer 1.33.0's own three.js client (Minecraft's block models, the 1.21.4 texture atlas, entity models and skins) in headless Chrome for Testing 154 (puppeteer 25.12, WebGL through ANGLE/Direct3D 11 on the RTX 4090). It is patched at serve time only to expose the viewer and render on demand. Added: a sky dome whose lowest ~8 deg equal the fog colour, Minecraft's own sun texture, and distance fog (64-120 blocks). `games/minecraft_bridge/web/fv.js` corrects three things prismarine-viewer draws wrongly. (1) **Limb pivots**: the viewer nests bones whose positions are absolute pivots, so a posed limb swings about a point 36 px above its joint; fv.js sets each posed bone's matrix so it turns about its true pivot (checked numerically against three r128). (2) **Texture mapping**: zombie and husk geometry declares a 64 x 32 texture for 64 x 64 skins, so the torso, arms and legs sampled transparent texels and vanished; sheep the reverse. fv.js rescales v. (3) **Phantom offset**: vanilla's PhantomRenderer draws the model 1.3125 blocks lower and 0.1875 further back than prismarine-viewer does, so the phantom used to float 1.3 m above its hitbox. fv.js applies vanilla's offset and pitches the model along its flight. The bot is drawn as the viewer's player model (Steve); limb swing (and the zombies' raised arms) are ours. Main view: a chase camera that **orbits** the player, turning at most 90 deg/s towards the nearest mob and holding its bearing while a phantom passes overhead, never closer than 3 m. Inset: the player's first-person view at the eye (80 deg). The fly never sees either. If the real renderer fails to start during `--record`, the game stops instead of recording the fallback. |
| HUD | mixed, chipped | DNp01 (`fb.motor().gf`, the escape_jump input) against the 33 Hz line on a 0-120 Hz trace, and the LPLC2 (185 cells) / LC4 (126 cells) mean rates per soma side and MDN (`back_dn`) at 22 px: CONNECTOME. The body's walking speed (with its constant / neural split), the player speed, the yaw and the jump count: DECODER. The fly's-eye mosaic (the GAME's ray cast; the exact radiance handed to `fb.vision`), hearts, the mob's eye-to-hitbox distance, the amber ring on the mosaic (where the nearest mob is and its angular size) and the event list (a coloured dot gives each line's class): GAME. Banners carry one chip per class they report ("GIANT FIBRE 38 Hz -> JUMP": CONNECTOME + DECODER). In interactive mode keys 1-4 summon a zombie 12 m ahead / 35 deg left / 35 deg right / behind; those zombies are not part of the script and stay up to 9 s. |
| control run | GAME | `--control blind-to-mobs`: same seed, same script, same world, but **the scripted mobs** (the three zombies and two phantoms) are left out of the fly's ray cast. Other entities (the world's sheep and pigs) stay. The footage still shows the scripted mobs; the header and the mosaic say CONTROL. |

### Per-encounter numbers, as the run log defines them

* `gf_peak_near`: the GF maximum while the mob was the nearest live mob **and within 3 m** (zombie: horizontal
  centre-to-centre distance to the player; phantom: hitbox centre to the eye). `gf_max_window` is the maximum at any
  distance while it was the nearest; it is logged but not quoted as a response to the mob.
* Phantoms: the peak is split at the turn (`gf_peak_dive`, `gf_peak_pass`). `first_crossing` gives the GF at the
  first 33 Hz crossing, the distance and **eye-to-hitbox gap** then, and the phase (dive / pass).
* `mob_gap` (trace and events): the distance from the eye to the nearest mob's hitbox; 0 would mean the eye is
  inside it.

### What was measured (dev seeds >= 100)

**A block-world prototype says only a very close loom reaches the giant fibre.** Before any Minecraft, the same
caster ran on a synthetic grass plain with trees (seed 100, builder's notes; no run log kept, so the numbers in
this paragraph cannot be checked against a file and are context only). Walking with the eye
at 1.62 m gave GF max 9.9 Hz at 2 m/s and 14.3 Hz at 4.3 m/s. A zombie-sized box flying at the eye crossed 33 Hz
only at 0.55-0.57 m from the eye; stopping at 0.8 m it peaked at 10-12 Hz.

**The final dev clip** (seed 104, the frozen code: `games/minecraft.py` sha256 prefix `2db7dbf5db01c497`,
`out/games/minecraft/dev_final.mp4` + `.json`, 546 s wall on the shared GPU):

* Zombie from ahead (12 m): came within 0.93 m (hitbox 0.61 m from the eye), hit twice (4.8 and 5.2 s). GF peak
  within 3 m 15.2 Hz. No jump.
* Zombie from 35 deg left: closest 1.49 m, no hit. GF peak within 3 m 8.5 Hz.
* Zombie from 35 deg right: closest 1.86 m, one hit at 17.75 s. GF peak within 3 m 13.2 Hz.
* Phantom from ahead: **GF crossed at 38.5 Hz at 21.09 s, during the dive, with the phantom's hitbox 0.25 m from
  the eye** (centre 0.70 m). The player jumped at 21.10 s (the next Minecraft tick), the same tick the dive turned
  into its pass at 0.10 m. GF peaked at 93.0 Hz during the pass over the head (the dive's own peak was 44.6 Hz).
  The eye never entered the hitbox.
* Phantom from 60 deg left: it turned at 0.10 m at 26.51 s. **GF crossed at 38.6 Hz at 26.57 s, 60 ms after that
  closest approach**, with the hitbox 0.40 m away and receding. The player jumped at 26.60 s. Peak 51.5 Hz during
  the pass (dive part 21.5 Hz).
* MDN (moonwalker) reached 54.6 Hz at 27.0 s, 0.4 s after the second jump; the body's speed went just negative and
  the player drifted back at up to 0.17 m/s for about 0.1 s (logged `backing_up`). The player walked 55.7 m, at a
  median 1.85 m/s. No deaths, no bumps, no Auto-Jump.
* Lockstep: 600 Minecraft ticks stepped = 600 server ticks elapsed.

**The control arm** (seed 104, `--control blind-to-mobs`, same code, `dev_final_control.mp4` + `.json`):
**0 jumps, GF max 12.5 Hz over the whole run.** The scripted geometry was the same: both phantoms still turned
0.10 m from the eye (21.11 and 26.51 s), but with the mobs left out of the cast their GF peaks within 3 m were 5.0
and 4.6 Hz (against 93.0 and 51.5 Hz, and crossings at 38.5 and 38.6 Hz, with the mobs in the cast). The zombies
(Minecraft's side differs between runs: 2, 0 and 1 hits) peaked at 0.0, 5.6 and 9.7 Hz within 3 m. The walking
speed was unchanged (median 0.861 cm/s), MDN max 56.3 Hz, no backing up, 600 = 600 ticks. The header and the mosaic
both say CONTROL. The earlier control clip (`dev_c_control`, seed 103) predates these fixes and is not used.

**The phantom dive, with this code, on six dev runs** (`tally_ph_101/102/103/105.json`: the two dives alone, at
t = 1.0 s and 5.5 s; `dev_fixed.json` and `dev_final.json`: seed 104, full script). Every dive made the player jump.
The GF crossed at the head, never earlier:

| dive | runs | jumped | GF at the crossing (Hz) | hitbox gap then (m) | when | peak (Hz, all during the pass) |
|---|---|---|---|---|---|---|
| from ahead | 6 (seeds 101, 102, 103, 105, 104 x2) | 6 | 33.8, 37.5, 37.2, 42.2, 39.0, 38.5 | 0.22, 0.24, 0.25, 0.28, 0.16, 0.25 | 10-20 ms before the turn, except one 20 ms after | 101.5, 96.8, 100.3, 111.1, 59.8, 93.0 |
| 60 deg left | 6 (same) | 6 | 36.7, 34.2, 34.0, 38.1, 42.7, 38.6 | 0.10, 0.14, 0.10, 0.34, 0.16, 0.40 | 10-60 ms after the turn | 83.9, 90.5, 90.1, 74.5, 96.3, 51.5 |

The dive's own GF peak (before the turn) was 30.9-57.1 Hz from ahead and 18.7-32.3 Hz from 60 deg left, so the
side dive crosses only as its closest pass sweeps over the head. The peaks of 51-111 Hz come 70-190 ms after the
turn (50 Hz trace; the longest, 190 ms, is `dev_final`'s 60 deg dive), during the pass, with the phantom's hitbox 0.10-0.57 m from the rising (jumping) head: they are the response to
a phantom skimming the head, not to the approach. Zero frames with the eye inside a hitbox in any of these runs.

Sources: `tally_ph_*` and `dev_fixed` ran the same stimulus, eye and body code as the frozen version, before three
HUD / log-only changes (banner chip fading, banner rounding, an extra baseline field); `dev_final` is the frozen code.

**Zombies rarely reach the giant fibre.** They stop at melee range, and their hits knock the player back:

| zombie | encounters (logged) | GF >= 33 Hz (jump) | GF peak within 3 m (Hz) |
|---|---|---|---|
| from ahead, 12 m, this code | 2 (seed 104 x2) | 1 (`dev_fixed`: 33.5 Hz at 5.43 s, hitbox 0.36 m, 0.5 s after a hit) | 37.6, 15.2 |
| from ahead, 12 m, earlier revisions (same zombie script and eye; recomputed from the logs' 50 Hz traces) | 4 (`dev_b` seed 102, `dev` / `dev_seed104_hoverbug` / `dev_seed104_pass1.1` seed 104) | 1 (seed 102, after a hit) | 35.5, 5.0, 11.7, 31.7 |
| from 35 deg left / right, this code | 4 (seed 104 x2) | 0 | 9.4, 12.7, 8.5, 13.2 |
| from 35 deg left / right, earlier revisions | 6 (seed 104 x3) | 0 | 2.8, 11.4, 15.5, 12.9, 7.7, 7.6 |
| from ahead / 35 deg left at 6-7 m, intermediate revisions of this fix (same zombie and eye code; `fix_a` seed 106, `fix_b` seed 107) | 3 | 0 | 0.2, 5.9, 5.0 |

The side zombies' peaks (2.8-15.5 Hz) are within the walking baseline below. Zombies summoned 45 and 70 deg to the
side (seed 102, earlier revision) never caught the walking player, which is why the script uses 35 deg.

**Baseline.** With no mob within 8 m, and leaving out the 1.5 s after each GF crossing, phantom pass and removal,
GF p99 was 6.2-9.3 Hz and max 8.9-15.4 Hz (seed 104, `dev_fixed` and `dev_final`). In the second after a phantom's
pass and the landing of the jump, the GF still reached 21.8-24.2 Hz with the phantom 8-12 m away; the run log's
`gf_no_mob_within_8m` includes those tails, `gf_outside_encounters` does not. The body's walking speed had median 0.86-0.88 cm/s
in every final-code run (0.80 of it the constant drive), so the player strolled at a median 1.85-1.89 m/s.

**A check of the interactive mode** (seed 108, SDL dummy driver, key 1 at 0.30 s): the key's zombie
was summoned 12 m ahead and was still approaching (6.0 m) when the run quit at 2.4 s. Before the fix, the key's
zombie was removed after 0.8 s.

### What the reviews found, and what changed

Two independent reviews of the first version found real problems. The numbers above come only from runs of the
fixed code, except where a table row says "earlier revisions".

* **The eye went inside the phantom.** The old dive was aimed once per Minecraft tick at an eye position two ticks
  stale, and pulled up 0.6 m (box centre) from it. After the jump the eye rose into the 0.9 x 0.5 x 0.9 m hitbox,
  every ray saw flat phantom grey, and the old "GF peak 85-114 Hz" figures were measured there. The dive is now
  stepped every brain frame against the current eye, turns at a 0.10 m eye-to-hitbox gap, and keeps that gap on the
  pass. A box around the eye is culled and logged. The old phantom figures and the old "the 60 deg dive never
  crosses" result are withdrawn: with the lag gone, the side dive reaches the head and does cross.
* **The footage drew the phantom 1.3 m above its hitbox**, and **drew zombies as floating lumps** (a limb-pivot bug
  and a texture-size bug in prismarine-viewer). All three are corrected in fv.js; the phantom is now drawn at the
  player's head exactly when the fly's eye has it there.
* **Most of the walk is a constant.** The HUD used to chip the body speed CONNECTOME; it is now DECODER, and the walk
  law states the constant.
* **"GF peak" per encounter** used to be the maximum at any distance (a zombie's "18.8 Hz" came 4 m away). It is now
  taken within 3 m, and the control's old "knockback" explanation is withdrawn.
* Also fixed: the chase camera whip-panned through the player after the phantom pass (it now orbits and holds its
  bearing); key-summoned zombies were removed after 0.8 s; non-living entities could enter the eye's cast; script
  times drifted by a tick; banners' chips were clipped and did not fade; the GF trace flat-topped at 80 Hz; the
  recorder could silently fall back to our own render.

**CUDA graphs were not used for the brain.** `FlyBrain(cuda_graphs=True)` ran 2.4x faster on this machine but did
not reproduce the eager path's spikes on the same seed and input (seed 101, 300 frames of a fixed radiance
sequence; a builder's measurement, no log kept). The game keeps the default path. Only our own ray caster is graph-captured (bit-identical).

### Limits

* **The jump is not a dodge.** The giant fibre crosses when the zombie is at arm's length after a hit, or when the
  phantom's hitbox is 0.1-0.4 m from the eye (from ahead just before the closest approach, from the side just after
  it). The Minecraft jump starts at most 50 ms later.
* **The phantom's pass is designed to skim the head.** The 0.10 m gap is a GAME choice (a phantom's attack reaches
  the head), made on dev seeds. The large GF peaks come from that pass, and the pass keeps its gap by lifting over
  a jumping head.
* **The eye map has a hole in front.** MaleCNS's reconstructed columns are sparse in the frontal strip (azimuth
  about -20 to +15 deg, most of the elevation range; `out/retina_map.png`). A head-on zombie and the last metre of a
  phantom dive partly fall where there are no columns. The mosaic shows those gaps in black and says so.
* The fly sees mobs as coloured hitbox boxes. The phantom's wings and the zombie's arms are drawn in the footage
  but not in the fly's eye. Grass tufts and flowers are drawn in the footage and are invisible to the fly. One biome
  tint everywhere; leaves opaque.
* **Minecraft's side is not seeded by the brain seed.** Zombie pathing, attack timing and knockback use the
  server's own randomness, so two runs of one seed differ on the Minecraft side (seed 104's zombie from ahead:
  crossed after a hit in one run, 15.2 Hz in the next). The phantom dives are fully scripted.
* `/tick freeze` does not freeze players. The player's server-side entity ticks on wall-clock time (this is why
  natural regeneration is off). Some logs show two 2.5 HP hits 0.1-0.45 s of brain time apart from one zombie,
  closer than a zombie's attack cooldown, and hits logged while the zombie was never closer than 1.85-1.86 m
  (horizontal centre distance, as the bot sees it). They are reported as logged.
* The footage renderer is prismarine-viewer, not the vanilla client: no clouds, no armour or fire on mobs, simpler
  lighting. Walk animation and the zombies' raised arms are ours. Distant fogged mountains show as pale haze.
* Speed: on the shared GPU, depending on what else runs on it, 8-28 s of wall time per brain second in the dev
  runs with the real renderer and 10.7-85.2 s in the six recordings: 5.3-42.6 minutes for a 30 s clip, plus 31-50 s of server, world and
  viewer start-up (Clips, below).

### Recording

Clip length 30 s of brain time (1,500 frames at 50 fps): the five scripted mobs arrive at 1.5, 7.5, 13.0, 19.0 and
24.5 s, and the second phantom is removed at about 28 s. From the repository root, with no Minecraft server running
(the game starts a fresh one and stops it):

    <miniconda>/python.exe games/minecraft.py --seed {seed} --record out/games/minecraft/seed{seed}.mp4 --seconds 30

Control, same seed:

    <miniconda>/python.exe games/minecraft.py --seed {seed} --control blind-to-mobs --record out/games/minecraft/control_seed{seed}.mp4 --seconds 30

Record one run at a time: the server (ports 25631 / RCON 25632) and the viewer (3031) use fixed local ports, and the
game refuses to start if a server already answers on the RCON port. If the real renderer cannot start, a recording
stops with an error rather than recording the fallback (`--no-viewer` records our own ray-cast render of the
server's blocks on purpose, labelled "OUR RENDER OF THE SERVER'S BLOCK DATA"; tested). The run log (`.json` next to
the clip) carries every summon, hit, GF crossing (with the mob, its distance, the eye-to-hitbox gap and the dive
phase), escape, jump, landing, phantom pass and removal, a per-encounter summary, 50 Hz traces of GF, LPLC2, LC4,
MDN, speeds, mob distance and gap, the walking-speed split, two GF baselines, the entity names in the eye's cast,
the Minecraft and server tick counts, and sha256 prefixes of the Python and Node sources.

## Clips

Recorded on 2026-09-28 on this desktop: `cuda`, NVIDIA GeForce RTX 4090, torch 2.10.0+cu128, Python 3.13.2. The
footage comes from prismarine-viewer in headless Chrome (WebGL 2.0 through ANGLE / Direct3D 11 on the same GPU).
All six run logs carry the same provenance:
* commit `959f2e927b1c2806bc526fd5fc51cb3f5cc3e4b1` with `dirty: true` (games/ is not committed yet);
* sha256 (first 16 hex): `games/minecraft.py` `2db7dbf5db01c497` (the frozen code of `dev_final` above),
  `games/common.py` `b110142562a927a1`, `games/minecraft_bridge/bridge.js` `fa618c5d4f135896`,
  `games/minecraft_bridge/web/fv.js` `1eef8b762b27f1f7`, `package.json` `134bcf874e53777e`, `package-lock.json`
  `859b8d032fda5ae4`;
* preset `raw`, no instruments, no attached modules (the decoders are declared in the log, not attached);
* Minecraft 1.21.4, level seed `flyverse`, the same five-mob script. 600 Minecraft ticks stepped = 600 server ticks
  elapsed, and no frame had the eye inside a mob's hitbox (no `eye_in_mob_box` events).

`games/common.py` has changed since the recordings: its sha256 prefix is now `e06078fb69d1275a`, not the logged
`b110142562a927a1`. `games/minecraft.py`, `bridge.js`, `fv.js` and the two package files still match the logs. A
re-run from today's tree therefore does not use exactly the recorded code.

Each command ran exactly once, one at a time (the ports are fixed), from the repository root, as
`PYTHONIOENCODING=utf-8 <miniconda>/python.exe` followed by the arguments below. All six clips are
kept. No run crashed, and `games/minecraft.py` was not changed. Each clip is 30.0 s of brain time (1,500 frames,
1920x1080, 50 fps). Wall times vary eightfold because about ten other agents' runs shared the GPU and at times
oversubscribed its memory. Seed 0 spent 603 s of wall time on brain second 18-19, its main thread waiting in a
device-to-host copy (sampled with py-spy). The server is frozen between steps, so such a stall only stretches wall
time. The exception is the player's own server-side entity, which ticks on wall time (Limits).

| clip | command arguments | started (UTC) | brain / wall |
|---|---|---|---|
| `out/games/minecraft/seed0.mp4` | `games/minecraft.py --seed 0 --record out/games/minecraft/seed0.mp4 --seconds 30` | 22:03:34 | 30.0 / 1385.4 s |
| `out/games/minecraft/control_seed0.mp4` | `games/minecraft.py --seed 0 --control blind-to-mobs --record out/games/minecraft/control_seed0.mp4 --seconds 30` | 22:27:20 | 30.0 / 320.2 s |
| `out/games/minecraft/seed1.mp4` | `games/minecraft.py --seed 1 --record out/games/minecraft/seed1.mp4 --seconds 30` | 22:33:21 | 30.0 / 942.8 s |
| `out/games/minecraft/control_seed1.mp4` | `games/minecraft.py --seed 1 --control blind-to-mobs --record out/games/minecraft/control_seed1.mp4 --seconds 30` | 22:49:50 | 30.0 / 2556.6 s |
| `out/games/minecraft/seed2.mp4` | `games/minecraft.py --seed 2 --record out/games/minecraft/seed2.mp4 --seconds 30` | 23:33:17 | 30.0 / 1214.8 s |
| `out/games/minecraft/control_seed2.mp4` | `games/minecraft.py --seed 2 --control blind-to-mobs --record out/games/minecraft/control_seed2.mp4 --seconds 30` | 23:54:29 | 30.0 / 1381.2 s |

Setup took 31-50 s of wall time before brain time 0 (console). Of that, 15.7-18.6 s (the log's `setup_wall_s`) came
before the viewer started: the server start, world generation and the bot joining.

### What the run logs count

| clip | GF jumps | GF crossings without a jump | Auto-Jumps / bumps | hits / deaths | GF max | path | body speed, median (the constant's share) | MDN max |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 2: zombie #1 (5.45 s), phantom #2 (26.55 s) | 1: phantom #1 (21.04 s), inside the body's landing refractory | 2 / 3 | 2 / 0 | 90.8 Hz | 53.3 m | 0.858 cm/s (93.2 %) | 61.8 Hz |
| control 0 | 0 | 0 | 0 / 0 | 3 / 0 | 13.6 Hz | 54.3 m | 0.858 cm/s (93.3 %) | 56.3 Hz |
| seed 1 | 2: phantom #1 (21.10 s), phantom #2 (26.60 s) | 0 | 0 / 0 | 4 / 0 | 85.2 Hz | 56.8 m | 0.864 cm/s (92.6 %) | 54.8 Hz |
| control 1 | 0 | 0 | 0 / 0 | 4 / 0 | 19.1 Hz | 56.0 m | 0.865 cm/s (92.5 %) | 53.6 Hz |
| seed 2 | 2: phantom #1 (21.10 s), phantom #2 (26.55 s) | 0 | 0 / 0 | 3 / 0 | 94.8 Hz | 56.3 m | 0.872 cm/s (91.7 %) | 58.6 Hz |
| control 2 | 0 | 0 | 0 / 0 | 4 / 0 | 13.8 Hz | 55.1 m | 0.862 cm/s (92.9 %) | 60.6 Hz |

No run had a voluntary (wing power) takeoff or a death. One run backed up: control 2 at 21.51 s (below). In the
other five, MDN peaked at 53.6-61.8 Hz without reversing the walk. Every run ended at 20 / 20 HP: the player is
healed at each summon, and nothing hit it after 19 s. The player strolled at a median 1.85-1.88 m/s. The neural part of the body speed had
a median of 0.058-0.072 cm/s on top of the 0.80 cm/s constant (`walk`). The fly's cast saw a phantom, zombies, pigs
and sheep in the three connected runs, and only pigs and sheep in the controls.

**GF peak within 3 m of each scripted mob** (Hz; the definition is above). "cross" gives the GF at the first 33 Hz
crossing and the eye-to-hitbox gap at that moment.

| encounter | seed 0 | control 0 | seed 1 | control 1 | seed 2 | control 2 |
|---|---|---|---|---|---|---|
| zombie #1, ahead, 12 m | **47.5** (cross 36.0 Hz at 5.42 s, 0.24 m) -> jump | 0.1 | 7.7 | 3.4 | 5.0 | 0.0 |
| zombie #2, 35 deg left | 5.5 | 9.3 | 5.6 | 8.1 | 5.0 | 8.5 |
| zombie #3, 35 deg right | 8.7 | 13.6 | 11.2 | 9.4 | 20.0 | 12.8 |
| phantom #1, ahead, 22 m out, 11 m up | **88.0** (cross 37.7 Hz at 21.04 s, 0.38 m, dive) -> no jump | 5.0 | **85.2** (cross 41.1 Hz at 21.09 s, 0.22 m, dive) -> jump | 8.3 | **94.8** (cross 43.5 Hz at 21.09 s, 0.21 m, dive) -> jump | 3.3 |
| phantom #2, 60 deg left, 20 m out, 9 m up | **90.8** (cross 38.2 Hz at 26.54 s, 0.13 m, pass) -> jump | 4.9 | **42.5** (cross 38.4 Hz at 26.56 s, 0.34 m, pass) -> jump | 0.0 | **90.2** (cross 34.1 Hz at 26.53 s, 0.14 m, pass) -> jump | 4.2 |

### The phantom dives (6 in the connected runs)

* **Every dive crossed 33 Hz, and all of them at the head:** 34.1-43.5 Hz, with the phantom's hitbox 0.13-0.38 m
  from the eye. The frontal dives crossed 10-30 ms before the dive turned into its pass at 0.10 m; the 60 deg dives
  crossed 20-50 ms after that turn. The dive's own peaks (before the turn) were 42.0-54.1 Hz from ahead and
  24.2-30.9 Hz from the side. The within-3 m peaks of 42.5-94.8 Hz all came during the pass over the head.
* **Five of the six crossings made the player jump**, 10-40 ms later (the next Minecraft tick). No jump came before
  the dive turned at 0.10 m: the phantom was already at the head.
* **The sixth (seed 0, phantom #1) did not.** Minecraft's Auto-Jump (GAME) had lifted the player onto a one-block
  rise at 20.20 s, and it landed at 20.60 s. The GF crossed at 21.04 s and stayed above 33 Hz until about 21.22 s
  (50 Hz trace), 0.44-0.62 s after that landing. That is inside the shipped body model's 1 s landing refractory
  (`Flight.landing_refractory_s`: "a jump-land-jump chain is not fly behaviour"). The body did not take off, and
  the log records `gf_cross` without `escape`. The dive turned 0.10 m from the eye at 21.07 s, and the GF peaked
  at 88.0 Hz at 21.12 s (the hitbox then 0.38 m away and receding), with the player on the ground. On screen, the AUTO-JUMP banner at 21.0-21.2 s is left over from 20.20 s (banners stay up
  2 s). The player Auto-Jumped again at 21.35 s.
* **Controls (the scripted mobs left out of the fly's cast):** the same dives turned 0.10 m from the eye at
  21.09-21.10 s and 26.51-26.52 s, and none crossed. The GF peaked at 0.0-8.3 Hz within 3 m, and no dive made
  the player jump.

### The zombies (9 in the connected runs)

One zombie crossed the threshold. In seed 0, zombie #1 hit the player at 4.90 and 5.40 s. At 5.42 s, 20 ms after
the second hit, the GF crossed at 36.0 Hz, with the zombie 0.58 m away (horizontal centre distance) and its hitbox
0.24 m from the eye. The player jumped at 5.45 s, and the zombie was removed at 5.55 s. It was also the zombie
that came closest to the eye: its hitbox reached 0.15 m (`min_gap`), against 0.42-2.31 m for the other eight
connected-run zombies. The other eight stayed at 5.0-20.0 Hz within 3 m. The highest of these, seed 2's zombie #3 at 20.0 Hz (about 18.0 s, 0.4 s after its hit),
came as the player passed a tree on its left. The mosaic at 18.04 s shows the tree across the upper left eye
(left LPLC2 3.3 Hz, right 1.6 Hz) and no zombie. The Minecraft side differs between runs of the same seed (Limits), so the zombies' paths, hits and closest
distances differ between each run and its control:

| zombie | seed 0 | control 0 | seed 1 | control 1 | seed 2 | control 2 |
|---|---|---|---|---|---|---|
| #1: closest (hits) | 0.48 m (2) | 0.68 m (2) | 0.87 m (2) | 1.02 m (2) | 0.73 m (2) | 0.35 m (2) |
| #2: closest (hits) | 1.74 m (0) | 1.71 m (0) | 1.87 m (1) | 1.80 m (1) | 1.84 m (0) | 1.42 m (1) |
| #3: closest (hits) | 2.70 m (0) | 1.53 m (1) | 1.60 m (1) | 1.50 m (1) | 1.83 m (1) | 1.76 m (1) |

As in the dev runs, some hits were logged while the zombie was never closer than 1.76-1.87 m (horizontal centre
distance): seed 1 #2 and control 1 #2 at 10.75 s, seed 2 #3 at 17.65 s, and control 2 #3 at 17.70 s. They are
reported as logged.

### Baseline, and a tree

With no mob within 8 m, the GF had p99 7.1-13.4 Hz and max 8.9-18.2 Hz in the connected runs, and p99 7.3-14.1 Hz
and max 11.7-18.3 Hz in the controls. The two highest of these values (18.2 and 18.3 Hz) came from the terrain,
not from a mob. Seed 0's 14.3 Hz did not: it is the decaying tail of the zombie crossing, at 5.56 s, 10 ms after
that zombie was removed (5.55 s), which is the only reason it counts as "no mob within 8 m". Around
22.1 s (seed 1) and 22.7 s (control 1), the player walked past an oak close on its left. The left eye's mosaic
fills with trunk and leaves, left LPLC2 rises to 3.8-5.4 Hz, and the GF reaches 18.2 Hz (seed 1, the phantom
11.8 m away) and 18.3 Hz (control 1, 50 Hz trace). In seed 1 the oak came 1.0 s after phantom #1's pass and
0.45 s after the jump landed, so a post-pass tail (see the dev baseline) may add to its 18.2 Hz; control 1 reached
18.3 Hz at the same oak with neither. Control 1's maximum, 19.1 Hz, is a 100 Hz value, and not at the same moment:
the log puts it in phantom #1's window (`gf_max_window_hz` 19.1), so it came before that phantom (unseen by the fly
in the control) was removed at 22.6 s. The 50 Hz trace's highest sample in that window is 17.3 Hz at 22.06 s, as
the oak came into the left eye (left LPLC2 3.6 Hz), with the phantom 10.9 m away. Control 0 passed the same trees farther off at 22.66 s (12.2 Hz). No run's GF crossed 33 Hz except with a
scripted mob's hitbox within 0.4 m of the eye.

### Per clip

* **seed 0.** Zombie #1 hit twice (4.90 and 5.40 s), and then came the only zombie crossing in the three seeds:
  36.0 Hz at 5.42 s -> jump at 5.45 s, peak 47.5 Hz. Zombies #2 and #3 never hit; #3 never came within 2.70 m.
  The path then met the terrain: bumps at 19.3, 21.3 and 23.65 s, and Auto-Jumps at 20.20 and 21.35 s, the only
  ones in the six runs. Phantom #1 crossed at 37.7 Hz (21.04 s, gap 0.38 m) inside the landing refractory: no
  jump, peak 88.0 Hz. Phantom #2 crossed at 38.2 Hz (26.54 s, gap 0.13 m) -> jump at 26.55 s, peak 90.8 Hz, the
  run's maximum.
* **control 0.** No jump of any kind. GF max 13.6 Hz at 17.72 s, with zombie #3 within 3 m but outside the fly's
  cast (it hit the player at 18.15 s). Zombie #1 hit twice and came within 0.68 m while the GF stayed at 0.1 Hz
  within 3 m. The phantoms passed at 0.10 m with 5.0 and 4.9 Hz.
* **seed 1.** No zombie crossing: hits 2, 1 and 1, and within-3 m peaks of 7.7, 5.6 and 11.2 Hz. At 5.2 s zombie #1
  stood at the player's shoulder, its face filling the player's-eye view, with the GF at 3.7 Hz. Phantom #1:
  41.1 Hz at 21.09 s (gap 0.22 m) -> jump at 21.10 s, the tick the dive turned; peak 85.2 Hz. Phantom #2: 38.4 Hz
  at 26.56 s (gap 0.34 m, 50 ms after the turn) -> jump at 26.60 s; peak 42.5 Hz, the lowest pass peak of the six.
* **control 1.** No jump, GF max 19.1 Hz (the oak, above). Hits 2, 1 and 1. The phantoms passed at 0.10 m with
  8.3 and 0.0 Hz within 3 m.
* **seed 2.** No zombie crossing: hits 2, 0 and 1, and within-3 m peaks of 5.0, 5.0 and 20.0 Hz (the last during
  the tree, above). Phantom #1: 43.5 Hz at 21.09 s (gap 0.21 m) -> jump at 21.10 s; peak 94.8 Hz, the highest of
  the six clips. Phantom #2: 34.1 Hz at 26.53 s (gap 0.14 m) -> jump at 26.55 s; peak 90.2 Hz.
* **control 2.** No jump, GF max 13.8 Hz (15.94 s; zombie #3 was the nearest mob, not in the fly's cast). Zombie
  #1 came within 0.35 m, its hitbox 0.05 m from the eye, and hit twice (4.85 and 5.40 s). At 5.40 s its body fills
  the player's-eye view while the mosaic shows open grass and the GF reads 0.0 Hz. Hits 2, 1 and 1. The phantoms
  passed at 0.10 m with 3.3 and 4.2 Hz within 3 m. At 21.51 s, 0.4 s after phantom #1's pass (unseen by the fly),
  MDN burst to 59.5 Hz (50 Hz trace; 53.8 Hz when logged). The body's walking speed dipped to -0.09 cm/s, and the
  player drifted back at 0.16 m/s for about 0.1 s (`backing_up`, MOONWALKER DN 54 Hz -> BACKING UP). It is the
  only backing up in the six runs.

**Over the three connected runs (90 s of brain time):** 6 GF jumps (5 of the 6 phantom dives and 1 of the 9
zombies), one more crossing withheld by the landing refractory, 2 Auto-Jumps (GAME), 9 hits and no deaths. **Over the
three controls:** no jump of any kind and no GF crossing (GF max 13.6, 19.1 and 13.8 Hz). The phantom peaks
within 3 m were 0.0-8.3 Hz, against 42.5-94.8 Hz with the mobs in the cast. 11 hits, no deaths, and one backing
up.

### Trailer windows, checked against frames

* `out/games/minecraft/seed2.mp4` 20.4-21.7 s: accurate. The frontal phantom dives at the player, the "GIANT FIBRE
  44 Hz -> JUMP" banner appears at 21.10 s (log: crossing 43.5 Hz at 21.09 s, jump 21.10 s), the player is in the
  air at 21.2-21.4 s with the phantom passing overhead, and the banner is still up at 21.7 s.
* `out/games/minecraft/seed0.mp4` 3.5-5.6 s: accurate on the HUD, weak in the main view. The HIT banners come at
  4.90 and 5.40 s and "GIANT FIBRE 36 Hz -> JUMP" from about 5.44 s, but the chase camera sits behind the player,
  so the zombie is mostly hidden behind Steve; its face shows only in the player's-eye inset (and as a box in the
  mosaic), and the Minecraft jump is small.
* `out/games/minecraft/seed0.mp4` around 12.0 s is not a plain walk: zombie #2 is at the player's shoulder
  (hitbox 1.5-1.8 m from the eye at 11.8-12.3 s), the camera faces the player, and the "ZOMBIE #2 ... NO JUMP"
  banner arrives at 12.6 s. For a walk with nothing else in it, use seed 0 at 0.2-1.4 s (before the first summon at
  1.5 s): lakeshore meadow, camera behind the player, GF 0.0 Hz.

### Corrections (caption skeptic)

Every sentence was checked against the six recorded run logs, the dev logs it cites, the clips (frames extracted
with ffmpeg), `games/minecraft.py` and `flyverse/body.py`. What held: every per-seed and per-encounter number in the
recorded-run tables, the dev tables (`tally_ph_*`, `dev_fixed`, `dev_final`, `dev_final_control`, the
earlier-revision zombie peaks recomputed from the 50 Hz traces), the provenance fields, the console timings, the
decoder laws against the code, and the LPLC2 (185) / LC4 (126) cell counts against `cache/neurons.parquet`. Changed:

1. Headline. Was: "167,106 neurons behind a Minecraft player's eyes: a phantom skims the head, the fly's giant fibre
   fires, and the player jumps (5 of 6 dives; 0 of 6 when the fly cannot see the mobs)." Now it says the connectome
   is simulated, that the phantom is scripted, that "fires" means crossing the 33 Hz takeoff threshold (the GF fires
   at 7-14 Hz at baseline too), and that the sixth dive crossed as well but fell inside the landing refractory.
2. "not claimed" row. Was: "except once, when the body's 1 s landing refractory held it". Now: "blocked it (no
   jump followed)", and it says the refractory was started by a GAME Auto-Jump's landing (`games/minecraft.py`
   resets `fly.ground_time` on every landing, Auto-Jumps included).
3. `walk`. Was: "a speed-up of up to 0.7 cm/s as a phantom passes (body speed up to 1.47-1.54 cm/s)". The 1.47-1.54
   covers only the two seed-104 dev runs. Added: 1.47-1.62 cm/s over the six dev dive runs, 1.46-1.58 cm/s in the
   recorded connected runs, and at most 1.03-1.11 cm/s in the recorded controls.
4. `turn`. Was: "(DNa02 L-R, leg MN L-R; ...)". `Locomotion.readout`'s yaw also has a high-passed optomotor L-R term;
   it is now listed.
5. zombies. Was: "removed ... 0.6 s after its first hit". The logs show 0.60-0.65 s (checked once per Minecraft
   tick); said so.
6. rules. Was: "Fixed start: block (-296, 64, -294)". The code's and the log's `start` is (-296, 63, -294), the
   grass-top block; the player's feet are at y = 64. Corrected.
7. Block-world prototype and CUDA-graph figures: they have no run log. Both are now marked as unlogged builder's
   numbers.
8. Dev dive peaks. Was: "The peaks of 51-111 Hz come 70-160 ms after the turn". `dev_final`'s 60 deg dive peaked
   190 ms after its turn (50 Hz trace); now "70-190 ms".
9. Limits, speed. Was: "13-28 s of wall time per brain second ... A 30 s clip took 7-14 minutes plus about 30-40 s".
   That was dev data and is contradicted by the recordings: 10.7-85.2 s per brain second, 320-2,557 s (5.3-42.6 min)
   per clip, 31-50 s of set-up. Dev runs with the viewer: 7.8-27.8 s per brain second. Corrected.
10. Clips, provenance. Added: `games/common.py` has changed since the recordings (sha256 prefix now
    `e06078fb69d1275a`, logged `b110142562a927a1`), so today's tree is not exactly the recorded code.
11. Seed 0, phantom #1. Was: "The phantom skimmed the head at 0.10 m with the GF at 88.0 Hz and the player on the
    ground." The turn at 0.10 m (21.07 s) and the 88.0 Hz peak (21.12 s, hitbox 0.38 m and receding) are different
    moments; now stated separately.
12. Zombies. Added what the caption left out: the one zombie that crossed was also the one that came closest (hitbox
    0.15 m from the eye, against 0.42-2.31 m for the other eight).
13. Seed 2, zombie #3. Was: "the tree filling the left eye". In the frame at 18.04 s the tree spans the upper part of
    the left eye; now "across the upper left eye", with the LPLC2 rates.
14. Baseline. Was: "The highest of these values came from the terrain, not from a mob." Seed 0's no-mob maximum
    (14.3 Hz) is the decaying tail of the zombie crossing, 10 ms after that zombie's removal. Now only the two
    highest (18.2, 18.3 Hz) are credited to terrain, and seed 1's oak is flagged as coming 1.0 s after a phantom pass
    and 0.45 s after a landing.
15. Control 1. Was: "19.1 Hz, that run's maximum, is the 100 Hz value at the same moment". It is not: the log puts it
    in phantom #1's window, before 22.6 s, and the HUD already shows "max 19.1 Hz" at 22.06 s. The 18.3 Hz 50 Hz
    sample is at 22.70 s. Corrected.
16. Added "Trailer windows, checked against frames" (above).
17. CONNECTOME row. Was: "(LC4 / LPLC2 -> DNp01 x3; visual projection -> descending x2), which were set for the loom
    escape margin". The README's ledger gives the loom escape margin as the reason only for the x3 on LC4 / LPLC2 ->
    DNp01. It calls visual projection -> descending x2 a stop-gap for per-cell-type synaptic strengths, and it lists
    a third gain, descending -> VNC x3, which the row left out. Corrected.

Not changed, and still weak: the dev baseline "p99 6.2-9.3 Hz" does not reproduce exactly from the stated exclusion
rule (6.0 or 6.4 Hz for `dev_final`, depending on the percentile method); the py-spy stall diagnosis, "about ten
other agents", and the interactive-mode "removed after 0.8 s before the fix" have no file behind them.
