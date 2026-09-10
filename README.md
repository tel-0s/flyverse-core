# flyverse

Simulations, games and other strange places to put the **MaleCNS v1.0** fly connectome (HHMI Janelia
FlyEM + Google Research, Sept 2026: 167k neurons, 25.6 M connections, 124 M synapses, brain **and** ventral
nerve cord). The fly gets colour vision, a body, and a room with a table with fruit on it.

```
python -m flyverse.connectome          # one-off: compile the graph cache (15 s, needs the 3 feather files)
python scripts/room_demo.py            # live pygame window: fly on the table, brain panels
python scripts/room_demo.py --headless --seconds 20 --gif out/room.gif
python scripts/probe_vision.py         # diagnostic: what responds to rotation / walking, stage by stage
```

Data location defaults to `D:\Datasets\male-cns-connectome-v1.0\flat-connectome` (override with
`FLYVERSE_DATA`). Only three files are used: `body-annotations`, `body-neurotransmitters`,
`connectome-weights` (1.1 GB total). See `docs/NOTES.md` for everything learned about the data and the
model, and for the PufferLib note if you want to put RL on top.

## What is simulated

| stage | neurons | model | file |
|---|---|---|---|
| photoreceptors R1-R6, R7 (UV), R8 (blue/green) | 5,895 placed on 1,466 real hex columns (two eyes) | ray-traced spectral radiance [UV,B,G,R] per ommatidium -> low-pass + contrast adaptation | `retina.py`, `world.py`, `optic.py` |
| optic lobe (lamina, medulla, lobula, lobula plate) | 89,390 `ol_intrinsic` | graded rate units on the signed, input-normalised connectome (flyvis-style) | `optic.py` |
| everything else: visual projection neurons, central brain, VNC, motor neurons | 71,625 | Shiu et al. 2024 leaky integrate-and-fire (+ adaptation, synaptic depression) on the GPU | `brain.py` |
| body | walking on the table | named readout: DNa02 L/R -> turning, DNp09/DNa01/... -> forward, MDN -> backward, leg MNs, MN9 -> proboscis | `body.py` |

Whole thing runs at ~0.6x real time on an RTX 4090 (10 ms of brain per 16.6 ms frame; slower with the
window open). Rotating the fly in the textured room modulates every optic-lobe layer and drives the
DNa02 turning neuron on one side at ~28 Hz vs ~0 on the other, with under 2% of the CNS active.

The motor readout is a *hypothesis* (which descending neurons mean what), not a trained decoder; the
connectome is not modified. Whether the fly walks towards the fruit is an experiment, not a promise.

![demo](docs/demo_frame.png)

![retina](docs/retina_map.png)
*The two eyes reconstructed from the connectome: 1,466 hex columns with R1-R6 (yellow), R7 (purple),
R8 (green) and dorsal-rim (red) photoreceptors placed by their strongest lamina/medulla partner.*

## Layout

```
flyverse/connectome.py   load + compile the three feather files -> signed CSR (cached in cache/)
flyverse/retina.py       photoreceptor -> hex column -> viewing direction; spectral sensitivities
flyverse/world.py        tiny torch ray tracer: room, table, fruit, 4-channel light, textures
flyverse/optic.py        graded optic lobe + photoreceptor contrast stage + interface to the LIF
flyverse/brain.py        whole-CNS LIF (torch sparse, GPU)
flyverse/body.py         fly pose/kinematics + motor readout from named neuron groups
scripts/room_demo.py     the interactive demo
scripts/probe_vision.py  stage-by-stage diagnostic
docs/NOTES.md            findings, parameters, references, RL notes
```

Data: CC-BY 4.0, https://male-cns.janelia.org. Inspirations: stonkfly, doomfly (nftechie), fly-escape (dzhng).
