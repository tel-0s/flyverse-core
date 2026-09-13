# Extensibility spec: gluing things onto the fly

Status: proposal (2026-09-13), for implementation by the Neurome side (Astra) against this repo. The flyverse
owner reviews the API; the project rule (`README.md`, `docs/INTERP.md` §10) applies: nothing here changes the
shipped model's defaults or behaviour; every extension is opt-in, named, and recorded in provenance so the
interpretability toolkit can see it.

## 0. What exists today (the surface to extend, not replace)

| layer | file | what it gives you |
|---|---|---|
| brain | `flyverse/fly.py` `FlyBrain` | senses in physical units (`vision`, `smell`, `wind`, `taste`; `proprioception` incoming), `stimulate(selection, hz, ms)`, `step(ms)`, `motor() -> MotorRates`, `state_dict()`, `reset(rows)`, `step_budget`, `AsyncFlyBrain` |
| primitives | `flyverse/brain.py` `Brain` | `set_drive(idx, mv)`, `set_poisson(idx, hz)`, `rates(idx)`, `rate_np()`, `freeze`, `prune`, `set_weights(W)`, `set_clocks` |
| selection | `Connectome.select(**criteria)`, `regions.select/subset/pare`, `flyverse.interp.common.resolve(c, spec)` | one grammar for "which cells": type / regex / superclass / module / bodyId / mask |
| body | `flyverse/body.py`, `batch_body.py` | `FlyState`, `Locomotion`, `Flight`, `Metabolism`: rates -> motion |
| programs | `flyverse/programs.py` | body-side modules with `apply(motor, cmd, fly, metabolism, dt_s) -> cmd`, `state()/load_state()`, `Composite` |
| batch | `flyverse/batch_sim.py` `BatchSim` | B environments around one batched brain |
| RL | `flyverse/env.py` `FlyRoomEnv` | vectorised gym: obs = DN rates, action = (forward, yaw) |
| readouts | `flyverse/nt_readout.py` (`docs/NT_READOUT.md`) | optional per-module readout sources for the observatory |
| provenance | `flyverse/interp/common.py` `provenance()` / `Result` | every experiment JSON records cache fingerprint, params, device, seeds |

Everything below is additive to that table.

## 1. Step hooks (small; do first)

```python
fb.add_hook(fn, when="pre" | "post", name="...")     # fn(fb, t_ms) -> None
fb.remove_hook(name)
```

- `pre` hooks run once per `step()` call before the LIF/optic advance, after the senses have written their
  inputs; `post` hooks run after the advance, before `step()` returns. Hooks see the batched brain (`fb.B`).
- Inside a hook the allowed writes are the existing primitives: `fb.brain.set_drive(idx, mv)` (mV per LIF
  step, added to the sensory drive), `fb.brain.set_poisson(idx, hz)` (background Poisson rate),
  `fb.stimulate(...)`. Reads: `fb.brain.rates(idx)`, `fb.brain.rate_np()`, `fb.motor()`, `fb.optic.rates()`.
- Invariants: a hook that raises aborts the step with the hook's name in the error; hooks are listed in
  `fb.hooks` and in `provenance()["model"]["hooks"]` (name, when, and a stable identifier — module:qualname —
  so a JSON says which code was attached). `state_dict()` does not serialise hook code; it records their names.
- Tests: a pre-hook that writes `set_poisson` on a selection changes that selection's rate and nothing else's
  mean over 1 s on a `Connectome.subset`; removing the hook restores bit-identical output at the same seed.

## 2. Module protocol (the general case)

A module is anything that reads some populations each frame and writes some populations (or motor commands)
each frame: a second SNN, a torch network, a trainable encoder/decoder, a plain function.

```python
class Module(Protocol):
    name: str
    reads: dict[str, Selection]          # label -> selection (cells whose rates it receives), may be empty
    writes: dict[str, Selection]         # label -> selection (cells it drives), may be empty
    quantity_in: str = "rate_hz"         # rate_hz | drive_mv | spike_count | optic_rate
    channel_out: str = "poisson_hz"      # poisson_hz | drive_mv
    def reset(self, B: int, device) -> None: ...
    def step(self, dt_ms: float, inputs: dict[str, Tensor]) -> dict[str, Tensor]:  # (B, n_cells) in, (B, n_cells) out
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...
    def describe(self) -> dict: ...      # provenance: class, parameters, checkpoint hash, trainable flag
```

`fb.attach(module)` / `fb.detach(name)`; `FlyBrain` calls every attached module's `step` once per frame
(10 ms) after the senses and before the LIF advance, delivering `inputs` gathered from the previous frame's
state and applying `outputs` through the channel named (`set_poisson` or `set_drive`). Two modules may not
write the same cell through the same channel in the same frame (raise at attach time). Modules run on the
brain's device and are batched by construction.

Reference implementations to ship with the protocol (each a few dozen lines, each with a CPU test):

- `FunctionModule(reads, writes, fn)` — a plain callable.
- `TorchModule(reads, writes, net, channel_out)` — an `nn.Module` from (B, n_in) to (B, n_out); `describe()`
  hashes its state_dict; trainable via §4.
- `SNNModule` — a second spiking graph run beside the connectome: a small `Brain` built from a synthetic
  `Connectome` (§3 gives it a bodyId namespace), whose input cells are driven from `reads` and whose output
  cells drive `writes`. This is "glue an SNN on" without touching the main CSR.
- `ReadoutModule(reads, fn)` — writes nothing; feeds the observatory through the `NT_READOUT` source
  interface (a `health`-style panel of anything you compute).

Programs (`flyverse/programs.py`) stay the body-side counterpart: they see `MotorRates` and write commands.
A module that wants to act on the body rather than the brain implements the program interface instead; the
two are composed in the same frame loop (`Composite`).

## 3. Graph extension: synthetic cells wired into the connectome

```python
c2 = c.extend(nodes, edges)      # nodes: DataFrame(bodyId, type, superclass, side, nt, dataset="synthetic")
                                 # edges: DataFrame(body_pre, body_post, weight, sign | nt)
```

- Synthetic bodies live under `dataset = "synthetic"` with their own bodyId space (negative int64 or a
  reserved high range; the interchange key `(dataset, release, bodyId)` already exists, so exports and Neurome
  joins stay unambiguous). MaleCNS cells are never renumbered.
- `extend` rebuilds the CSR (`W[post, pre]`) with the new rows/cols appended, re-runs the sign rule for the
  new edges (explicit `sign`, or `nt` through `NT_SIGN`), leaves every existing entry byte-identical, and
  writes a scratch cache (`cache_dir=<scratch>`, the existing pattern) — the shared cache is never modified.
- `Brain` / `FlyBrain` need nothing new: they take the Connectome. `prune` is the inverse. `regions.labels`
  assigns synthetic cells to the module named in `nodes.superclass` (or `"synthetic"`).
- The CUDA graph and native kernels re-capture on a new Connectome (they already do on `subset`).
- Provenance: `connectome_fingerprint()` records the base cache fingerprint plus the extension's own hash and
  node/edge counts, so a Result says "MaleCNS v1.0 + 120 synthetic cells / 4,000 edges (sha …)".
- Tests: extend by two cells and three edges on a subset; the original block of W is identical; the new
  cells spike when driven; `prune` returns the original; the fingerprint changes and says so.

## 4. Trainable pieces

- Sense-side encoders and motor-side decoders are `TorchModule`s at the boundary: e.g. a learned
  `radiance -> photoreceptor drive` in place of `Retina.photoreceptor_intensity`, or a learned
  `MotorRates -> (forward, yaw)` decoder in place of `Locomotion.readout`. They are trained by RL through
  `FlyRoomEnv` (already vectorised; add an `EnvParams.modules_attached` hook so the policy's action can be the
  decoder's parameters or its output) or by supervision against recorded rollouts (`common.Recorder` npz).
- Gradients through the brain itself: opt-in `LIFParams.surrogate_grad = True` on the Torch path only
  (spike = Heaviside with a fast-sigmoid surrogate; the CUDA/Metal kernels and CUDA graphs do not backprop and
  raise if requested). Feasible for `modules=` subsets and short windows; document as slow and experimental;
  never the default. The rate optic lobe is already a differentiable torch graph.
- Any trained artefact attached to a run is a `describe()`d module with a checkpoint hash in provenance.

## 5. Rules that make extensions safe to publish

1. Off by default: a fresh `FlyBrain()` has no hooks, no modules, no extension; bit-identical output to today.
2. Named and recorded: everything attached appears in `provenance()`, `state_dict()`, and the observatory's
   module list. The toolkit's `decompose` / `trace` treat module writes as an input class ("module:<name>").
3. Sign-safe: a module writing `drive_mv` may be negative (inhibition); `poisson_hz` is clamped ≥ 0. Neither
   channel may rewrite W.
4. Batched: every interface takes (B, …) tensors; scalar convenience wrappers call the batched path.
5. Tested on CPU with `Connectome.subset` or synthetic graphs (`tests/test_interp.py`'s 8-neuron `graph()`
   helper is the template); no test may need the 3 GB data.
6. Classified: a module that implements a hand-designed behaviour is a stop-gap and is labelled as such in its
   `describe()` (`kind = "stop-gap" | "mechanism" | "sensor" | "decoder" | "analysis"`), exactly like
   `--program` modules today.

## 6. Order of work and where files go

1. `fly.py`: hooks (§1) + `attach/detach` + the per-frame module loop; `flyverse/modules.py`: the protocol
   and the four reference modules; tests `tests/test_modules.py`.
2. `connectome.py`: `Connectome.extend`; `regions.py` label rule; `interp/common.py` fingerprint field;
   tests `tests/test_extend.py`.
3. `env.py`: the attach hook for trainable boundary modules; one worked example under `examples/`
   (a learned motor decoder that leaves the brain untouched) with a short doc `docs/EXTENSIBILITY.md`.
4. Surrogate gradients (§4) last, and only on the Torch path.

Open questions for the owner: whether `poisson_hz` writes should add to or replace the sensory forcing
(today `stimulate` combines by maximum); whether synthetic bodyIds are negative or a high reserved range
(Neurome's join contract prefers an explicit `dataset` field, so either works); and whether module `step`
should see the current frame's sensory inputs (pre) or the previous frame's rates (post) — the spec says
previous frame to keep the loop explicit and one frame long.
