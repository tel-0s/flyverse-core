# sunrise: the lights come on for 141,781 somata

## Design

The trailer's opening shot. The shipped fly (MaleCNS v1.0, `preset="raw"`, nothing attached) stands on the table in
the room of `scripts/room_demo.py` (`Sim(fence=True)`). The camera shows the whole CNS as a point cloud of every soma
with a recorded position, 141,781 of 167,106 (`somaLocation` in the MaleCNS body annotations; the rest, mostly VNC and
sensory cells, have none and are not drawn), orbiting slowly.

| what | kind | law |
|---|---|---|
| the room's light | GAME | the lamp's point light and the ambient light are both x0 (black) until `--lamp-on-at` seconds, then the shipped values. The switch is the only thing done to the fly. What actually changes the fly's view is the ambient term. In the shipped room the point light sits at the centre of its own 8 cm lamp globe, so every shadow ray is blocked by the globe and the point light reaches no surface (a flyverse finding, below). Measured at the start pose: with the point light off and the ambient term at its shipped value, the per-column radiance is identical to the shipped room (max 0.211, mean 0.095); with the ambient term off it is 0 in every column. |
| each soma's glow | GAME (display) | spiking cells: rate / 30 Hz; graded optic-lobe units: (\|r - baseline\| - 0.12) / 0.3; both clipped to [0, 1]. 0.12 sits just above the optic lobe's median \|r - baseline\| in total darkness, so the dark phase reads as a dim flicker and the light's arrival reads as a surge. |
| the activity itself | CONNECTOME | the model's own rates, every frame |
| camera orbit, bloom, colours | GAME (display) | decoration |
| WHAT THE FLY SEES inset | CONNECTOME input | the exact radiance handed to `fb.vision` |

Measured on a dev seed (101) before the recordings, CPU-side from `fb.optic.last["dr"]`: in total darkness the optic
lobe is never silent: median \|dr\| 0.06-0.10, 90th percentile about 0.3, flickering frame to frame. Within about 60 ms
of the lamp coming on, the median rises about 2.6-fold, to about 0.21, then adapts back over roughly a second.

    python games/sunrise.py --record out/games/sunrise/seed0.mp4 --seconds 22 --lamp-on-at 12.02 --seed 0 --clean

`--lamp-on-at 12.02` is the first trumpet note of the soundtrack (out/trailer/music/beatmap.json, `call1_C4`), so
the lamp switches on on the beat. This shot uses a light switch for its drama; it is not a behaviour.

## Clips

Recorded on this desktop (NVIDIA GeForce RTX 4090, torch 2.10.0+cu128) from the working tree on top of commit
`959f2e9`, with `games/sunrise.py` sha256 `bfa997e30948acf7` and `games/common.py` sha256 `b110142562a927a1`
(the first 16 hex, in each run log). 22 s of brain time each, about 140-160 s of wall time. Command, for k = 0, 1, 2:

    python games/sunrise.py --record out/games/sunrise/seed{k}.mp4 --seconds 22 --lamp-on-at 12.02 --seed {k} --clean

What the three run logs say (the optic lobe's |r - baseline| sampled every 100 ms, over every graded unit; "dark" is
2-12 s):

| seed | dark median \|dr\| (range) | dark 90th pct | peak median after the lamp | lit, 16-22 s | spiking mean rate, dark -> 12.1-12.5 s | giant fibre max |
|---|---|---|---|---|---|---|
| 0 | 0.075 (0.053-0.093) | 0.333 | 0.239 at 12.10 s | 0.100 | 0.58 -> 0.81 Hz | 18.4 Hz (12.8 s) |
| 1 | 0.075 (0.057-0.098) | 0.332 | 0.234 at 12.10 s | 0.101 | 0.59 -> 0.78 Hz | 14.7 Hz (20.8 s) |
| 2 | 0.075 (0.058-0.093) | 0.334 | 0.225 at 12.10 s | 0.100 | 0.58 -> 0.68 Hz | 18.6 Hz (16.2 s) |

In every seed, the optic lobe's median response triples within 80 ms of the lamp and adapts to about 1.3x its dark
level within a few seconds. The giant fibre stays under the body's 33 Hz threshold, so no seed takes off (the lamp is
not a loom). The trailer uses seed 0.

One change before these recordings, disclosed. A first render of all three seeds is kept in
`out/games/sunrise/old_framing/`:
* seeds 0 and 1 used `games/sunrise.py` sha256 `04b0229aa0c488db`, which framed the orbit on the whole CNS, so the
  brain drifted off-centre;
* seed 2 had started after the camera was re-centred on the brain (sha256 `98283b5e80a90e92`).

All three still logged only every 0.5 s. The log was then changed to sample the model's raw optic statistics every
100 ms, and all three seeds were recorded again. The fly's run is the same command and seed.

## A flyverse finding surfaced here (not changed)

`flyverse/world.py`: `make_room` places the point light (`World.light_pos`) at the centre of the lamp globe
(`Sphere(w.light_pos, 0.08, "lamp")`). `_trace` casts each shadow ray from the hit point toward the light and counts
the point as lit only if nothing is closer than the light. The globe always is, at `dist - 0.08`. So in the shipped
room, direct light never reaches any surface: every image the model has seen in the room is lit by the ambient term
alone, flat and without shadows. It was found independently by the swarm builder (games/captions/swarm.md) and
confirmed here. Fixing it (exclude emissive spheres from the shadow test, or move the light out of the globe) changes
the model's visual input in every room run. So it is an owner decision that needs the room-based suite rows re-run.
It is **not** changed for this release.

After the recordings, the module docstring of `games/sunrise.py` was corrected to say that only the ambient term
reaches the fly. That is a comment-only change, so the file's sha256 is now `ac277e1ad4d59bd8`; the recorded runs used
`bfa997e30948acf7`.
