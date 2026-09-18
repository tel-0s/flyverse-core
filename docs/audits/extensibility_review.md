# Review of `feat/extensibility` (4f56dcb) against `docs/EXTENSIBILITY_SPEC.md`

Independent read-only review (Opus, 2026-09-13) of Astra's implementation branch (one commit on `b5463af`, 21 files,
+2,242), run in its worktree `flyverse-refactor`; nothing edited. **Verdict: merge with fixes** (B1-B3), after the
round-2 working tree lands, because the surrogate-gradient optic substep must be rewritten against main's final
`_recurrent`.

## 1. Coverage against the spec

| spec item | status | where |
|---|---|---|
| §1 `add_hook` / `remove_hook`, pre / post | implemented | `fly.py:127,135`; `modules.py:354` |
| §1 allowed writes = existing primitives (setters re-routed) | implemented | `brain.py:671,682` -> `modules.py:337` |
| §1 raise aborts the step with the hook's name | implemented | `modules.py:360` |
| §1 hooks listed in `fb.hooks` and `provenance()["model"]["hooks"]`; `state_dict` records names only | implemented | `fly.py:110,549`; `interp/common.py:944`; `modules.py:529` |
| §1 required isolation test | implemented | `tests/test_modules.py::test_hook_isolation_and_remove_identity` |
| §2 Module protocol (reads / writes / quantity_in / channel_out / reset / step / state_dict / describe) | implemented | `modules.py:50-62` |
| §2 selections through `interp.common.resolve` | implemented | `modules.py:370` |
| §2 write-collision raise at attach (own labels and cross-module, same channel) | implemented | `modules.py:379-386` |
| §2 `FunctionModule` / `TorchModule` / `SNNModule` / `ReadoutModule`, each CPU-tested | implemented | `modules.py:64 / 94 / 188 / 257` |
| §2 per-frame loop after the senses, before the LIF; previous-frame inputs | implemented | `fly.py:275`; `modules.py:418-432` |
| §3 `Connectome.extend`, negative-int64 synthetic namespace | implemented | `connectome.py:431` |
| §3 CSR rebuild with the existing block byte-identical (tested) | implemented | `tests/test_extend.py:39-43` |
| §3 sign rule (explicit `sign`, else `nt` -> `NT_SIGN`) | implemented | `connectome.py:~485` |
| §3 scratch cache; the shared cache never modified (guarded both ways) | implemented | `connectome.py:653,659` |
| §3 `prune` inverse; fingerprint changes and says so; `regions.labels` synthetic rule | implemented | `connectome.py:417`; `common.py:853`; `regions.py:29` |
| §3 CUDA-graph / kernel re-capture on a new Connectome | inherited, not verified (no GPU in the review) | -- |
| §4 boundary `TorchModule` (vision encoder / motor decoder); `EnvParams.modules_attached` + `action_adapter` | implemented | `modules.py:94-186,374`; `env.py:55-57,110,198` |
| §4 worked example + `docs/EXTENSIBILITY.md` | implemented | `examples/learned_motor_decoder.py` |
| §4 supervision from `Recorder` npz | partial: needs `teacher_forward` / `teacher_yaw` motor fields nothing in the repo produces | `examples/learned_motor_decoder.py:88-92` |
| §4 `surrogate_grad`, Torch path only, raises on kernels / graphs | implemented | `brain.py:149,360,712`; `fly.py:47` |
| §5.1 off by default, bit-identical | verified (section 2) | -- |
| §5.2 named / recorded; `decompose` / `trace` input class `module:<name>`; observatory list | implemented | `common.py:646`; `decompose.py:_module_rows`; `trace.py:ArmAccumulator`; `room_ui.py:466` |
| §5.3 `poisson_hz` clamped >= 0, `drive_mv` may be negative, W never rewritten | implemented, tested | `modules.py:350,483`; `tests/test_extend.py:88` |
| §5.4 batched (B, ...) with per-output validation | implemented | `modules.py:459` |
| §5.5 CPU tests on subsets / synthetic graphs only | implemented (no test needs the data) | -- |
| §5.6 `kind` classification, `stop-gap` default, validated at attach | implemented | `modules.py:19,375` |
| public surface: `docs/CONTROL_SURFACE.md` and README 'Use the brain in your own simulation' | missing: CONTROL_SURFACE untouched; README gains a 4-line pointer only | -- |

## 2. Bit-identity: verified, but no test in the repo

The branch asserts bit-identity only in prose (`docs/EXTENSIBILITY.md`). The reviewer built a harness (three trees via
`git archive`, each run in its own process): 14 default-path cases (four senses + `stimulate` + `set_drive` /
`set_poisson` at B = 1 / 2 / 4, clocked, `receptor_model` None / sign / full; a 19-cell 4-column optic graph through
`W_rp` / `W_rr` / `W_rs` / `W_sr` and the LIF behind them; fractional-ms carry-over; `state_dict` round trip; full and
per-row `reset`; `Brain.prune`; STD; custom normalisation), dumping every brain / optic tensor, `_acc`, RNG state,
`base_poisson`, W and every `MotorRates` field. **5,254 arrays, 1,670 non-trivially non-zero, 0 differing** between
`main` and `feat/extensibility`, and between main's working tree (proprioception + optic stream hooks) and the branch.
The duplicated surrogate paths also match the inference paths bit for bit today (`_step_surrogate` vs
`_step_inference` over 80 steps incl. Poisson forcing; `_step_frame_grad` vs `_step_frame_inference`, maxdiff 0.0).

## 3. Tests

| run | result |
|---|---|
| branch full CPU suite (minus `test_cuda` / `test_metal`) | 211 passed, 21 skipped, 200 subtests |
| main's existing tests on the branch (`test_control`, `test_batch_sim`, `test_interp`, `test_nt_readout`) | 112 passed, 7 skipped, 0 regressions |
| branch-only suites (`test_modules`, `test_extend`, `test_surrogate`, `test_extensions_cuda`, `test_room_ui`) | 33 passed, 3 skipped (CUDA gated) |
| simulated merge: main HEAD + main working tree + branch, full CPU suite | 249 passed, 19 skipped, 0 failures (incl. `test_proprioception`, `test_optic_hooks`) |

## 4. Merge-conflict map

| file | main (uncommitted working tree) | branch | verdict |
|---|---|---|---|
| `flyverse/optic.py` | per-stream hooks, substep rewritten via `_recurrent()` | `surrogate_grad`, `intensity=` kwarg, duplicated `_step_frame_grad` | one textual conflict in `OpticLobe.__init__` (both insert a kernel guard), trivial; substantive: B2 |
| `flyverse/fly.py` | `available_senses` + `proprioception()` | hooks / attach / step split / `state_dict` | auto-merges; separable |
| `flyverse/interp/common.py` | `parse_kv` / `_parse_tuple_list` | `Recorder`, `connectome_fingerprint`, `provenance` | auto-merges; disjoint |
| `batch_sim.py`, `batch_body.py`, `senses.py` | proprioception | untouched | the per-frame `fb.proprioception(...)` call and the module loop compose (verified in the merge sandbox) |
| `interp/export.py`, `tests/test_interp.py`, `pyproject.toml` | unchanged by main's working tree | branch-only | none |

Recommended order: land main's working tree first, then merge `feat/extensibility`, resolve the one `optic.py` hunk,
fix B2 against main's final substep.

## 5. Blocking issues

- **B1 -- default-path resource regression.** `fly.py:100` sets `self.brain._track_external_drive = self.optic is not
  None` unconditionally, so `brain.py:677-678` clones and appends `(idx, value)` on every `set_drive` even with nothing
  attached; the list is cleared only in `_vision_frame` when `_radiance is not None` (`fly.py:334,346`). An
  optic-capable `FlyBrain` that never calls `vision()` accumulates one entry per frame, unbounded, and `state_dict()`
  serialises the list (`fly.py:552`). Numerically identical, but a per-call cost and a leak on the shipped path. Gate
  both on `self._extensions is not None`.
- **B2 -- merge hazard: `surrogate_grad` silently drops main's optic stream hooks.** `optic.py:589 _step_frame_grad`
  is a hand-copied substep using `p.gain_rr * _mv(self.W_rr, dr)`; main's `_step_frame_inference` routes through
  `self._recurrent(dr, update=True)` (streams, `G_supp`, `stream_adapt_state`). Measured on the merged tree with
  `stream_rectify` + `stream_adapt` on: the inference model moves by 0.0733, the grad model by exactly 0.0 -- a
  silently different forward model, no error. The same structural risk applies to `brain.py:725 _step_surrogate` for
  any future LIF change. Either implement the hooks in the grad path or raise when both are requested.
- **B3 -- a writes-nothing `analysis` module perturbs the model.** `fly.py:282` splits `step(ms)` into <= 10 ms frames
  whenever anything is attached; a reads-only `kind="analysis"` `FunctionModule` changes `step(50)`'s optic `v` by
  0.37. A `ReadoutModule` / observatory panel is advertised as observation and should be numerically neutral: split
  only when a module writes (or when `quantity_in` needs per-frame sampling).

## 6. Nits

1. `common.py:939-942` -- `provenance()` now derives `lif` / `optic` from `fb` when not passed (fixes a real bug:
   `provenance(c, fb=fb)` used to record LIFParams defaults) but silently changes recorded model records; only
   `env.py:227` is affected internally; changelog it.
2. `common.py:849` -- `connectome_fingerprint` prefers `c.cache_dir` (set by `load()`) over `FLYVERSE_CACHE`;
   `subset()` (`connectome.py:414`) does not propagate `_cache_dir`, so a graph and its subset report different dirs.
3. Every Result JSON gains `model.hooks: []`, `model.modules: []`, `model.lif.surrogate_grad: false` -- a schema
   change on the default path (harmless; consumers should know).
4. Per-frame allocation: `modules.py:429` rebuilds a dict over `optic.rate_idx` every frame per `optic_rate` module
   (~70k entries on the real graph); `:431` re-converts read indices to device tensors every frame; `:349` clones a
   full `(B, N)` field per write. Precompute at attach.
5. `modules.py:410-416` and `SNNModule.describe()` dump every read / write bodyId into provenance and every
   checkpoint with no `keep_ids <= 10000`-style guard.
6. `connectome.py:669-672` -- `extend` saves the full base graph and its reference into the scratch cache
   uncompressed; `:313-331` builds a dict over every sign-0 edge of the base. Exercised only on 8-cell graphs.
7. `connectome.py:675` -- `save()` of a non-extended connectome unlinks `extension.json` in the target dir; a plain
   save can strip an extension cache (the reverse direction is guarded).
8. `modules.py:337-346` -- with extensions attached, a plain `fb.brain.set_drive()` becomes a held, additive input
   instead of a one-frame value overwritten by the optic frame; documented, but the same call means two things.
9. `fly.py:76` / `regions.py:38-41` -- `dt_by_module` and `regions.select` accept any live label, so a typo matching
   a superclass string passes silently (needed for synthetic labels; note the loosening).
10. The spec's three open questions are decided (Poisson combines by maximum; synthetic bodyIds negative int64;
    module `step` sees the previous frame) in `docs/EXTENSIBILITY.md` but not flagged as owner decisions.
11. `modules.py:283` -- `ReadoutModule.step` reads `self.body_ids`, set only by `attach`; a standalone instance
    raises `AttributeError`.
12. `fly.py:127-133` -- `add_hook` on a live runtime does not clear `self._graphs` (harmless today; inconsistent).
13. `regions.py:30` -- a synthetic cell with `superclass="descending_neuron"` gets the label `descending_neuron`, not
    the canonical `descending`.

## 7. Owner decisions on the open questions

Accepted as decided on the branch: Poisson writes combine with sensory forcing by maximum; synthetic bodyIds are
negative int64 under `dataset = "synthetic"`; a module's `step` sees the previous frame's rates. To add before the
merge lands: `docs/CONTROL_SURFACE.md` coverage of hooks / attach / extend, and a cross-version bit-identity test in
`tests/` so the section-2 claim lives in code rather than prose.
