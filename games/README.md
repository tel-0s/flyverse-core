# games: strange places to put a fly connectome

The shipped fly (MaleCNS v1.0, 167,106 neurons, `preset="raw"`) in ten places it has never been: Pong, real
Minecraft, Doom, a spaceship, a TRON arena, a rover on Mars, a swatter, a hairdryer, a crowd of 64, and a dark room
whose light comes on. None of
them changes the model. Each talks to the brain only through the public control surface
([docs/CONTROL_SURFACE.md](../docs/CONTROL_SURFACE.md), [docs/EXTENSIBILITY.md](../docs/EXTENSIBILITY.md)). Senses
go in; named rates come out through read-only decoders, which are modules that write nothing. Together the games are
a tour of that surface, and a demonstration of what the raw model does and does not do.

## How to read them

Every number on screen carries one of three chips, and every caption sorts the game the same way:

| chip | means |
|---|---|
| **CONNECTOME** | a rate read straight from named cells: the model's own output |
| **DECODER** | a mapping from neural activity to a game control, written for the game and declared with its law and parameters (`games.common.ReadDecoder`, `kind="decoder"`, writes nothing) |
| **GAME** | everything that is not the fly: physics, opponents, scripts, autopilots, the rules |

The README's list of where the raw model stops applies here unchanged. The fly does not turn on its own, has no flight
state without stimulation, and does no goal-directed search. Where a game needs any of that, a declared decoder or the
game supplies it, and the caption says which.

**Recording policy** ([PLAN.md](PLAN.md), rule 5). After each game's code was frozen, seeds 0, 1 and 2 were recorded
exactly once each, with one command line. Where a cheap control exists (decoders disconnected, stimulus hidden from
the fly), it was recorded for the same seeds. No run was repeated for a better outcome. Every caption reports all
three seeds, including the misses, splats, deaths and non-events. Each game went through the same pipeline: a
builder, then two independent adversarial reviews (scientific honesty; engineering and visuals), then a fix pass,
then the recordings, then a caption skeptic that checked every sentence against the run logs and frames. Each
caption ends with the skeptic's corrections.

## Run them

```
python scripts/fetch_data.py --malecns --fafb     # once: the two connectomes, hash-verified (see ../README.md)
python games/pong.py                     # interactive; W/S to play against a fly with --left human
python games/doom.py                     # needs vizdoom:  pip install --no-deps vizdoom==1.3.1
python games/minecraft.py                # needs Java 21+, Node 18+, and `npm install` in games/minecraft_bridge
python games/<name>.py --record out/games/<name>/seed0.mp4 --seed 0     # headless, the way the clips were made
```

Every game shares `--seed`, `--record`, `--seconds`, `--fps` and `--size` ([common.py](common.py)). A GPU is
assumed: the full brain costs about 8 ms per 10 ms of brain time on an RTX 4090, and the games' rendering and ray
casting add to that. Headless recording is offline, so a slower machine changes only the wall time, never the result.
The clips were made on two kinds of GPU, and each caption names the device of every clip: an RTX 4090 desktop (Windows 11, torch 2.10.0+cu128), with several games recording at once, and NVIDIA B200s on a cluster (Linux, torch 2.11.0+cu128). Comparisons between arms, such as decoders on vs off, are always made within one device, because the GPU rollout is not bit-reproducible across devices (docs/REPRODUCIBILITY.md).

## The environments

| game | what the fly's brain does in it | the result, from the recorded seeds (details and every outcome in the caption) |
|---|---|---|
| [sunrise](sunrise.py) | the room's light switches on; nothing else is done | the optic lobe's median response triples within 80 ms and adapts back; every soma's glow is the model's own activity ([caption](captions/sunrise.md)) |
| [pong](pong.py) | MaleCNS v1.0 vs FlyWire FAFB v783; each paddle follows a decoder reading where the ball excites that fly's L2 / Mi1 columns | FAFB took 19 of 20 points on the RTX 4090; on one B200, 17 of 18 from the same end and 13 of 15 with the ends swapped, consistent with its 3.7x photoreceptors facing the court; not a sex comparison ([caption](captions/pong.md)) |
| [minecraft](minecraft.py) | a real 1.21.4 server, stepped one tick per 50 ms of brain; the fly's 1,466 ommatidia are rays through the server's blocks; the giant fibre (DNp01 >= 33 Hz) jumps | a scripted phantom skimming the head made the player jump in 5 of 6 dives, 0 of 6 when the mobs are left out of the fly's eyes ([caption](captions/minecraft.md)) |
| [doom](doom.py) | ViZDoom / Freedoom on a 160 deg screen; the gun fires when the giant fibre crosses 33 Hz | the crossings are locked to the screen (usually a fireball bursting across it); random triggers at the same rate usually kill more ([caption](captions/doom.md)) |
| [spacecraft](spacecraft.py) | a 6-DOF ship; a declared decoder reads H2 / HS and VST2 left-minus-right and fires roll and yaw thrusters | on one B200, a 60-70 deg/s tumble averaged 15-30 deg/s with the decoder on and 54-66 deg/s with the thrusters off; pitch is not held ([caption](captions/spacecraft.md)) |
| [swat](swat.py) | no decoders at all: LC4 / LPLC2 -> giant fibre -> the body model's escape jump | the jump launched before the paddle arrived in 18 of 30 swings (0 of 30 when the fly cannot see the swatter), and in all 18 the fixed forward hop then met the paddle in the air ([caption](captions/swat.md)) |
| [hairdryer](hairdryer.py) | a scripted hand that knows where the apple is aims a jet; the wind DNs (DNp18 / DNp33) feed a declared turn-into-the-wind decoder; apple contact drives the sugar neurons -> MN9 | on one B200, 3 of 3 decoder-on runs reached the apple and 0 of 3 controls came closer than their start ([caption](captions/hairdryer.md)) |
| [swarm](swarm.py) | 64 flies, 64 brains, one GPU, one swatter; no decoders | the giant fibre fired in time for about two thirds of the flies in the head's path, and for none in the blind controls; the unaimed hop sent most of them into the swatter ([caption](captions/swarm.md)) |
| [tron](tron.py) | MaleCNS vs FlyWire FAFB on light cycles; a declared decoder turns each cycle 90 deg when its giant fibre crosses 33 Hz, side from LPLC2 + LC4 left-minus-right | 19 of MaleCNS's 20 turns came about a metre from a wall and it took 7 rounds to FAFB's 3; it crashed less than 140 of 150 random-time riders, and no less than a one-line rule that turns 1.1 m before a wall and flips a coin ([caption](captions/tron.md)) |
| [mars](mars.py) | a GAME rover on a Jezero-like plain; a dust devil's wind reaches the Johnston's organ, and a declared decoder reads DNp18 / DNp33 left-minus-right and veers the rover into the wind (capped below the autopilot, held during collision manoeuvres) | in all three seeds one veer of about 7 deg and 4 m, into the wind and so away from the devil (closest pass 19.8-19.9 m against 16.0 m in the no-wind and decoder-off controls); the giant fibre never flagged a boulder, and every arm hit all four ([caption](captions/mars.md)) |


## The trailer

[`docs/media/trailer.mp4`](../docs/media/trailer.mp4) (86 s, 720p; the 1080p50 master is rebuilt below) is cut from
the recorded clips above to Kevin MacLeod's "Also Sprach Zarathustra". Every cut lands on a beat from an onset
analysis of the track ([`trailer_assets/beatmap.json`](trailer_assets/beatmap.json), measured by
[`trailer_assets/analyze_track.py`](trailer_assets/analyze_track.py)). The track is not committed: `trailer_cut.py`'s
docstring gives the download, its SHA-256 and the decode. Each shot is placed so that a logged event lands on its
beat: a takeoff at the climax, a kill, a veer. Every shot plays at 1x; the only slow motion is swat's own labelled
0.25x replays. The trailer adds cuts, fades, flashes, cards and lower-thirds, and never retimes or edits a clip.

    python games/trailer_cut.py            # games/trailer_shots.py + the beat map -> games/trailer_edl.json
    python games/trailer.py games/trailer_edl.json          # -> out/trailer/flyverse_trailer.mp4

The source clips, all among the recorded seeds above. Their captions describe what happens in each:

| clip | device | shots |
|---|---|---|
| sunrise seed 0 | RTX 4090 | 5 |
| swat seed 0 | RTX 4090 | 10 |
| spacecraft seed 0 | RTX 4090 | 6 |
| spacecraft b200_seed0 + control_seed0, side by side | B200 (both) | 1 |
| pong seeds 0 and 2 | RTX 4090 | 8 |
| minecraft seeds 0 and 2 | RTX 4090 | 7 |
| doom seeds 0, 1, 2 | RTX 4090 | 10 |
| hairdryer seed 0 | RTX 4090 | 5 |
| tron rec2/seed0 | B200 | 5 |
| mars seed 0 | B200 | 4 |

The one on-screen comparison (spacecraft: decoders on vs thrusters off) uses the two same-seed B200 runs.
`games/trailer_edl.json` records every shot's source window.

## What the games found about flyverse itself (not changed for this release)

These surfaced while building the games. None is fixed here, because each touches the shipped model's inputs or the
shipped room, and changing those is an owner decision that needs the affected suite rows re-run. They are tracked in
[TODO.md](../TODO.md), section G.

1. **The room's point light never reaches a surface.** `world.make_room` puts `World.light_pos` at the centre of the
   8 cm lamp globe, and `World._trace`'s shadow ray always hits the globe first. The shipped room is therefore lit by
   the ambient term alone, flat and without shadows. At the fly's start pose, per-column radiance is identical with
   the point light off (max 0.211, mean 0.095). Found by the swarm builder; confirmed in
   [captions/sunrise.md](captions/sunrise.md). swarm moves its light 1 cm below the globe (declared GAME); the other
   games use the shipped room.
2. **`world.render_camera` returns a left-right mirrored image** (column 0 looks to the camera's right).
   `room_demo.Camera.project` mirrors to match, so the console is consistent. Games that project points into a
   render must flip it; hairdryer and swat do. Found by the hairdryer builder.
3. **`world.value_noise` is constant along z.** `_hash3`'s z coefficient (2147483647) is 0 modulo its own modulus.
   This is harmless on the room's surfaces. Found by the spacecraft builder.
4. **About 92 % of the walking speed is a constant.** `Locomotion.baseline_speed` (0.8 cm/s, "intrinsic walking
   drive") dominates the body's forward speed in every game that walks, with DN / leg-MN rates adding the rest.
   Walking speed is therefore not a connectome readout in these games, and minecraft's HUD labels it DECODER. Found by
   the minecraft fixer.

## Credits

The connectome: MaleCNS v1.0 (Janelia FlyEM and Google Research) and FlyWire FAFB v783 (Pong's second player), as
cited in [CITATION.cff](../CITATION.cff). Doom: [Freedoom](https://freedoom.github.io/) (BSD) on
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) (MIT). Minecraft: a local, offline vanilla 1.21.4 server,
Minecraft's own assets, [mineflayer](https://github.com/PrismarineJS/mineflayer) and
[prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) (MIT). Minecraft is a trademark of Mojang;
this project is not affiliated with Mojang or Microsoft. Trailer music: "Also Sprach Zarathustra" by Kevin MacLeod
(incompetech.com), licensed under Creative Commons: By Attribution 3.0
(http://creativecommons.org/licenses/by/3.0/), via Wikimedia Commons.
