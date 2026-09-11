# flyverse

Simulations, games and other strange places to put the **MaleCNS v1.0** fly connectome (HHMI Janelia
FlyEM + Google Research, Sept 2026: 167k neurons, 25.6 M connections, 124 M synapses, brain **and**
ventral nerve cord). The fly gets colour vision, smell, taste, wind sense, a body that walks, jumps and
flies, and a room with a table with fruit on it. The connectome is never trained or edited; everything
the fly does comes from the wiring plus a small, documented set of physiological assumptions.

![demo](docs/demo_ui.png)
*The demo with `--brain-map`: fly's-eye camera, orbiting scene view with the fly's trail and the wind
arrow, the two eyes' photoreceptor mosaics, brain-activity panels, and every soma of the CNS with
activity as highlights (brain left, VNC right).*

## Run it

```
python -m flyverse.connectome              # one-off: compile the graph cache (15 s, needs the 3 feather files)
python scripts/room_demo.py                # live window: fly on the table
python scripts/room_demo.py --brain-map --trail-seconds 30 --window 1920x1080
python scripts/room_demo.py --start floor  # begin on the floor (it has to fly to get back up)
python scripts/room_demo.py --headless --seconds 20 --loom-at 4 --gif out/room.gif
```

Keys: `SPACE` pause · `R` reset · `T` teleport to the apple (taste) · `L` loom a black ball at the fly
(escape jump) · `F` fire the giant fibre · `W` stimulate the flight DNs (DNg02_a/DNa08: a 2-3 s powered
flight) · arrows / mouse drag orbit the scene camera, `+`/`-` or wheel zoom, `C` follow the fly, `HOME`
reset · `F5`/`F9` quick-save / quick-load, `S` timestamped save · `ESC` quit.

Options: `--brain-map` (all 140k located somata in dorsal and lateral view, activity as highlights;
`--map-every N`, `--map-no-blur`), `--trail-seconds` (decaying trail in the scene view), `--start x,y[,z]`
| `floor`, `--wind-speed`, `--wind-dir`, `--load state.pt`, `--window WxH` (the window is resizable, the
UI scales to fit), `--fast` (preset for Macs / slower GPUs; `--brain-dt`, `--optic-dt`, `--cam-scale`).
Save states are ~10 MB and resume bit-exactly.

Data location defaults to `D:\Datasets\male-cns-connectome-v1.0\flat-connectome` (override with
`FLYVERSE_DATA`); only `body-annotations`, `body-neurotransmitters` and `connectome-weights` are used
(1.1 GB). Torch backend: CUDA, else Apple MPS, else CPU (`FLYVERSE_DEVICE`); on MPS/CPU the synaptic
input is an event-driven gather instead of the sparse matmul (`flyverse/device.py`).

## What is simulated

| stage | neurons | model | file |
|---|---|---|---|
| photoreceptors R1-R6, R7 (UV), R8 (blue / green) | 5,895 placed on 1,466 real hex columns of the two eyes | ray-traced spectral radiance [UV,B,G,R] per ommatidium, low-pass + contrast adaptation | `retina.py`, `world.py` |
| optic lobe (lamina, medulla, lobula, lobula plate) | 89,390 `ol_intrinsic` | graded rate units on the signed, input-normalised connectome (flyvis-style); T4/T5 rectify with strong delayed inhibition and are **direction selective** with the correct preferred direction for all eight subtypes | `optic.py` |
| everything else: visual projection neurons, central brain, VNC, motor neurons | 71,625 | Shiu et al. 2024 leaky integrate-and-fire on the GPU, plus adaptation, a per-connection saturation cap, a fan-in cap for giant neurons, same-type synapse damping, antennal-lobe-only depression, and two pathway gains (DN -> VNC x3, visual projection -> DN x2) | `brain.py` |
| taste | 165 labellar sugar GRNs, found by connectivity to the known sweet interneurons | Poisson while touching fruit -> Usnea / Rattle / Phantom / G2N-1 -> proboscis MN9 | `scripts/find_sweet_grns.py`, `flyverse/data/taste_grns.csv` |
| smell | 2,639 ORNs in 53 glomeruli, sided to the left / right antenna by their PN targets | every fruit is a **wind-blown plume** (Gaussian, puffing) sampled by two antennae 1 mm apart | `air.py` |
| wind | Johnston's organ C / E neurons, sided by their AMMC/WED targets | antennal deflection per side from the wind vector in the body frame | `air.py` |
| body | walking, escape jumps, flight with pitch and roll | named readout from measured groups: DNp04 + LPT27/30 (optomotor), DNp18 / DNp33 (wind direction, gated by the odour signal), DNa02 (goal steering), MDN (back up), MN9 (proboscis); giant fibre -> escape jump; DLMn / DVMn -> wingbeat; wing steering MNs -> yaw and roll | `body.py` |

Speed: ~0.6x real time for one fly on an RTX 4090 (10 ms of brain per 16.6 ms frame). Batched brains
(`Brain(c, batch=64)`) run 64 flies at 1.8 ms per fly-frame -- one sparse matmul serves them all.

## What the connectome does, unprompted

These are measured behaviours of the wiring under the model, each with a probe script that reproduces it:

* **Direction selectivity** (`probe_motion.py`): T4a/T5a prefer front-to-back, b back-to-front, c up,
  d down, from the one-column offsets of their Mi4 / Mi9 inputs (DSI 0.16-0.26, speed-tuned).
* **Looming escape** (`probe_loom.py`): a black ball approaching at 1 m/s drives LC4 / LPLC2 -> the
  giant fibre, which spikes at ~3.5 cm range; TTMn (the jump muscle) follows it 1:1. Walking through a
  textured room leaves the giant fibre silent.
* **Sugar -> proboscis** (`probe_taste.py`): the labellar sweet GRNs drive the known second-order
  neurons (Usnea 67 Hz, Rattle 49, Phantom 47, G2N-1 28) and MN9 -- Shiu et al.'s result on MaleCNS.
* **Wind direction** (`probe_wind.py`): the JO -> AMMC / WED -> DN pathway is the strongest lateralised
  signal in the model: DNp18 fires on the side the wind comes from (+60 Hz over its partner), DNp33 on
  the opposite side (+64), then DNge016, DNg99, DNge175, DNg05_a, DNp19 and DNpe017.
* **Rotation** (`benchmark.py`): DNp04, LPT27/30 and DNp20 flip their left-right asymmetry with the
  direction of yaw rotation -- the optomotor signal the body reads. DNa02, by contrast, takes 80% of
  its input from the central-complex steering output (PFL3 / LAL) and is read as goal steering.
* **Measured motor map** (`screen_dns.py`, every DN type stimulated in its own batched brain copy):
  DNg02 -> wing power muscles at 146 Hz with no leg drive (Namiki's wingbeat-amplitude cluster), DNa08
  and DNp31 likewise; giant fibre -> TTMn 47 Hz; DNa02_L -> ipsilateral leg motor neurons; MDN -> the
  backward-walking premotor set; DNge080 / DNge062 -> proboscis.
* **Odour code** (`probe_smell.py`): glomerulus-specific PN responses (apple: DM1 / VA2 / DM4; banana:
  DM2 / VM2), sparse Kenyon cells, MBONs at 1-4 Hz.

`scripts/benchmark.py` scores any parameter set on all of these at once (~3 min); every default in
`brain.py` / `optic.py` was chosen with it. Runs are chaotic -- repeat before trusting a 20% change.

## What we had to add to the point model, and why

Shiu et al.'s LIF with one 0.275 mV synapse per contact reproduces sugar -> proboscis but, on the whole
CNS with sensory input, it has two regimes: silent or epileptic. The notes document each finding; the
short version: neurons without a neurotransmitter prediction and monoamines are treated as non-additive
(they were the first 300 Hz loops); giant connections are capped at 60 synapse-equivalents; same-type
synapses (gap-junction-coupled populations) are damped; adaptation is 1.5 mV/spike; depression lives
only in the antennal lobe, where it is documented -- globally it blocked every descending command. The
optic lobe up to T4/T5 is graded in the animal and is a rate model here; a spiking lamina/medulla does
not propagate at all. Two pathway gains (descending -> VNC x3, visual projection -> descending x2)
stand in for per-cell-type synaptic strengths; they were the difference between motor commands that
reach the legs and ones that do not.

## Food-finding (in progress)

Flies find fruit by odour-gated anemotaxis: surge upwind on an odour hit, cast crosswind when it is
lost. The room now has wind and plumes, the fly has two antennae and a wind sense, and the brain's own
wind-direction DNs steer it upwind with a gain gated by the antennal lobe's odour signal (the one
behavioural assumption the wiring does not supply itself). `scripts/probe_foraging.py` starts the fly
downwind of the apple and measures whether it gets there; see `docs/NOTES.md` for where that stands.

## RL on top of the brain

`flyverse/env.py` `FlyRoomEnv(batch=B)` is a vectorised environment (PufferLib convention; PufferLib
itself does not build on Windows/py3.13): observations are brain activity (descending-neuron rates, or
the per-glomerulus odour code), actions are walking commands, reward is progress to fruit + tasting.
`scripts/train_decoder.py` runs evolution strategies over a linear decoder. Results so far are negative
and documented: a hand-written klinotaxis policy on the odour code beats chance (93 vs 47 tasted
frames), ES did not find it in 40 generations.

## Layout

```
flyverse/connectome.py   load + compile the three feather files -> signed CSR (cached in cache/)
flyverse/retina.py       photoreceptor -> hex column -> viewing direction; spectral sensitivities
flyverse/world.py        torch ray tracer: room, table, fruit, 4-channel light, fly-scale textures
flyverse/optic.py        graded optic lobe + photoreceptor contrast stage + interface to the LIF
flyverse/brain.py        whole-CNS LIF (torch sparse, batched)
flyverse/air.py          wind, plumes, bilateral olfaction, Johnston's organ wind sense
flyverse/body.py         fly pose (3-D), walking / flight, motor readout from named neuron groups
flyverse/brainmap.py     soma projections for the --brain-map panel
flyverse/env.py          vectorised RL environment
scripts/room_demo.py     the interactive demo
scripts/benchmark.py     one-shot calibration harness;  scripts/probe_*.py  one behaviour each
scripts/screen_dns.py    the DN activation screen;  scripts/find_sweet_grns.py  sugar GRNs
docs/NOTES.md            everything learned, session by session, with numbers
```

![retina](docs/retina_map.png)
*The two eyes reconstructed from the connectome: 1,466 hex columns with R1-R6 (yellow), R7 (purple),
R8 (green) and dorsal-rim (red) photoreceptors placed by their strongest lamina/medulla partner.*

Data: CC-BY 4.0, https://male-cns.janelia.org. Inspirations: stonkfly, doomfly (nftechie), fly-escape (dzhng).
