# Reproducibility statement

What the shipped model is, exactly; what identifies the data it runs on; what is reproducible and
what is not; and which number in `README.md` was measured at which commit.

Everything in sections 1-3 was recomputed on **2026-09-14 at commit `28e862f`** with the commands each
section quotes. Sections 4-8 record facts established earlier and later -- the latest are round 8 and
session 14, 2026-09-18 -- and each names the audit, the batch and the file it comes from. Rounds 3
through 8 and session 14 adopted nothing and moved no default (`docs/audits/determinism_gate.md` 1,
`compass_sign_control.md` 4, `instrumented_suite.md` 4, `plume_goal_only.md` 4, `plume_transduced.md` 0;
`docs/NOTES.md` sessions 11-14), so the shipped model sections 1-3 describe is the model at the current
tree. How a round that produces any of these numbers is run is [`PROCESS.md`](PROCESS.md); what is owed
before the repository is made public is [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).

---

## 1. The shipped default model

The default is `LIFParams()` + `OpticParams()` + the module-level `DEFAULT_*` tables, with no
overrides, no program, no attached module and no hook. Read them back with:

```
python -c "import dataclasses; from flyverse.brain import LIFParams; print(dataclasses.asdict(LIFParams()))"
python -c "import dataclasses; from flyverse.optic import OpticParams; print(dataclasses.asdict(OpticParams()))"
```

### 1.1 `LIFParams` (`flyverse/brain.py`)

| field | value | unit / meaning |
|---|---|---|
| `v_rest` | -52.0 | mV |
| `v_reset` | -52.0 | mV |
| `v_th` | -45.0 | mV (a 7.0 mV gap from rest) |
| `tau_m` | 20.0 | ms |
| `tau_syn` | 5.0 | ms |
| `t_ref` | 2.2 | ms |
| `delay` | 1.8 | ms |
| `w_syn` | 0.275 | mV per synapse (Shiu et al. 2024; a cholinergic calibration) |
| `dt` | 0.5 | ms |
| `rate_tau` | 100.0 | ms, running firing-rate estimate |
| `adapt_jump` | 1.5 | mV per spike (**added**, not in Shiu et al.) |
| `adapt_tau` | 200.0 | ms |
| `adapt_by_type` | `None` -> `DEFAULT_ADAPT_BY_TYPE = {}` | uniform `adapt_jump` |
| `std_u` | 0.0 | short-term depression off globally (**stop-gap**: on, it blocked every descending command) |
| `std_tau` | 300.0 | ms |
| `std_u_by_type` | `None` -> `DEFAULT_STD_U_BY_TYPE` | `{'^ORN_': 0.2, '^(lLN\|v2LN\|v3LN\|il3LN\|l2LN\|vLN)': 0.2}` (antennal lobe only) |
| `input_norm_alpha` | 1.0 | fan-in normalisation exponent (**added**) |
| `input_norm_ref` | 5000.0 | synapses; above this the unitary scales by `(ref/total)^alpha`, floor 0.02 |
| `same_type_gain` | 0.1 | within-type synapse damping (**added**) |
| `conn_cap` | 60.0 | synapse-equivalents per connection (**added**) |
| `path_gain` | `None` -> `DEFAULT_PATH_GAIN` | see 1.3 (**stop-gap**) |
| `type_path_gain` | `None` -> `DEFAULT_TYPE_PATH_GAIN` | see 1.3 (**stop-gap**) |
| `event_driven` | `None` | matmul on CUDA, event gather elsewhere |
| `weight_dtype` | `'float32'` | |
| `dt_by_module` | `None` | one clock for every module |
| `prune_frozen` | `True` | drop frozen rate-unit rows/cols from the LIF matrix (exact) |
| `receptor_model` | `'sign'` | round-3 default; `None` restores the presynaptic-sign rule |
| `receptor_net_rule` | `'abs'` | which column set decides the sign |
| `receptor_nt_class_fallback` | `False` | |
| `receptor_table` | `None` -> `flyverse/data/receptors_by_type.csv` | |
| `receptor_gain` | `None` | no gain classes under `'sign'`; `DEFAULT_RECEPTOR_GAIN = {'none': 1.0, 'low': 0.5, 'mid': 1.0, 'high': 1.5}` applies only under `'sign+gain'` / `'full'` |
| `slow_tau_ms` | 200.0 | ms, monoamine class (inert: `receptor_model` is not `'full'`) |
| `slow_gain` | 0.02 | monoamine scale (inert at this `receptor_model`) |
| `slow_gain_by_class` | `None` -> `{'metabotropic_classical': 0.0, 'monoamine': 0.02}` | |
| `slow_tau_by_class` | `None` -> `{'metabotropic_classical': 100.0, 'monoamine': 200.0}` | |
| `slow_mode` | `'additive'` | |
| `slow_gain_clip` | `(0.0, 4.0)` | |
| `w_syn_by_nt` | `None` | per-transmitter unitary: opt-in instrument, executes nothing (`tests/test_unitary.py`) |
| `surrogate_grad` | `False` | |

The receptor model at `'sign'` / `'abs'` changes **48,295 of 25,578,600 entries** (30,916 glutamate
sign flips, 17,379 histamine silencings), 0.15 % of `|W|`; `scripts/hash_weights.py` prints the md5
of the shaped weights under each rule.

### 1.2 `OpticParams` (`flyverse/optic.py`)

| field | value | unit / meaning |
|---|---|---|
| `tau_ms` | 10.0 | ms, rate-unit membrane |
| `dt_ms` | 1.0 | ms, optic substep |
| `baseline` | 0.5 | operating point in [0,1] |
| `gain_rr` | 1.0 | recurrent optic-lobe gain |
| `gain_in` | 3.0 | photoreceptor contrast -> lamina |
| `norm` | `'l2'` | input normalisation of the optic-lobe weights |
| `gain_fb` | 0.5 | spiking neurons -> optic lobe (rate / 100 Hz) |
| `adapt_tau_ms` | 400.0 | ms |
| `adapt_gain` | 1.0 | |
| `tau_by_type` | `None` -> `DEFAULT_TAU_BY_TYPE` | Mi4/Mi9/CT1/Tm9 150, L3 40, Mi1/Tm3/Tm1/Tm2/Tm4 8, L1/L2 6, T4a-d and T5a-d 10 (ms) |
| `baseline_by_type` | `None` -> `DEFAULT_BASELINE_BY_TYPE` | T4a-d, T5a-d at 0.0 (rectifying) |
| `pair_gain` | `None` -> `DEFAULT_PAIR_GAIN` | see 1.3 |
| `gain_out_mv` | 100.0 | optic delta-rate -> injected current in spiking targets |
| `out_norm` | `'l1'` | |
| `drive_clip_mv` | 35.0 | mV (retirement tested and closed NOT adoptable, `docs/audits/anti_runaway.md` round 7) |
| `tau_lp_ms` | 10.0 | photoreceptor low-pass |
| `tau_adapt_ms` | 300.0 | photoreceptor contrast adaptation |
| `contrast_clip` | 2.0 | |
| `eps` | 0.02 | |
| `stream_rectify`, `stream_adapt`, `spatial_suppress`, `fb_hold` | all `None` | the four opt-in per-stream hooks; off is bit-identical (`tests/test_optic_hooks.py`) |

### 1.3 The gains, and which are stop-gaps

| table | entries | what it is |
|---|---|---|
| `DEFAULT_PATH_GAIN` | `(^descending_neuron$, ^vnc_, 3.0)`, `(^visual_projection$, ^descending_neuron$, 2.0)` | **stop-gap** for per-cell-type synaptic strengths: the difference between motor commands that reach the legs and ones that do not |
| `DEFAULT_TYPE_PATH_GAIN` | `(^(LC4\|LPLC2)$, ^DNp01$, 3.0)` | loom escape margin (Ache et al. 2019); x6 in total with the pathway gain above |
| `DEFAULT_PAIR_GAIN` | `(^(Mi4\|Mi9\|CT1\|C3)$, ^T4[abcd]$, 5.0)` | T4 delayed-inhibition arm (a mechanism) |
| | `(^(Tm4\|Tm9\|CT1\|TmY15)$, ^T5[abcd]$, 5.0)` | **stop-gap**: 78 % of the 101,619 edges it multiplies are cholinergic Tm9 / Tm4, so it is mostly a **drive** gain (`docs/audits/optic_measures.md`) |
| | `(^LPi(34\|43)$, ^LPLC2$, 4.0)` | LPi -> LPLC2 expansion selectivity; x1 tested and closed NOT adoptable (`anti_runaway.md` round 7) |
| | `(^T[45][abcd]$, .*, 2.0)` | T4/T5 output gain |

The remaining added mechanisms (adaptation 1.5 mV/spike, `conn_cap` 60, `same_type_gain` 0.1, the
fan-in normalisation, antennal-lobe-only depression, the graded optic lobe) are argued in
`docs/OVERVIEW.md` section 4 and `docs/NOTES.md`.

### 1.4 What is opt-in and therefore not part of the default

`senses.Proprioception`, `body.LegCycle`, `motor.read_haltere_sides`, `LIFParams.w_syn_by_nt`, the
monoamine slow class (needs `receptor_model='full'`), the four `OpticParams` stream hooks,
`fb.add_hook` / `fb.attach`, `Connectome.extend`, `surrogate_grad`, every
`flyverse/programs.py` / `flyverse/cx.py` module, and **every instrument** (the `instrumented` preset,
1.5). All default to off/`None`/empty; with nothing attached the simulation is bit-identical to the
default path on CPU (`tests/test_bit_identity.py`, `tests/test_body_cycle.py`,
`tests/test_optic_hooks.py`, `tests/test_unitary.py`).

### 1.5 The presets, and what a JSON says about them

`raw` is the default and is the plain model every audit refers to: the connectome, the LIF, the
receptor table, the senses and the motor readout exactly as shipped, no module attached, no held edge,
no relabel beyond the sourced `TYPE_NT_OVERRIDE` rows (`docs/PRESETS_SPEC.md` 1). `instrumented` is
`raw` plus a **named list of instruments**, and an empty list on the original connectome and LIF
parameters reproduces `raw` -- which `tests/test_bit_identity.py` asserts against the recorded golden
(4.4 below).

Every JSON says which it ran under. `flyverse.interp.common.provenance()` carries two keys beside
`compiled_connectome` (`flyverse/interp/common.py`, the `provenance` builder; `flyverse/instruments.py`
module docstring):

| key | value |
|---|---|
| `preset` | `'raw'` or `'instrumented'` |
| `instruments` | the list of each installed instrument's `describe()`; `[]` under `raw` (`flyverse/fly.py`, `instrument_records()`) |

`describe()` is the instrument's own label, and `docs/PRESETS_SPEC.md` section 2 requires every field
of it to be fillable before the instrument is admitted: `name`, `class`, `kind`
(`stop-gap` / `mechanism` / `edges` / `relabel` -- the four `flyverse.instruments.INSTRUMENT_KINDS`
enforces, and `_check_instrument` rejects anything else), `law` (a source, or the word
`unverified`), `law_text`, `parameters`, `replaces`, `gap` (which physiology it stands in for, pointing
at the audit that established the gap), `source`, `removal` (what result would retire it) and `audits`.
`replaces` is `"input"`, `"computation"` or `"configuration"`, and
`flyverse.instruments._check_instrument` **requires** it, so an instrument that does not declare what it
replaces cannot attach (`docs/PRESETS_SPEC.md` 5, the 2026-09-17 owner decision). There is no `program`
kind: a program-shaped stand-in -- one that replaces a *computation* rather than supplying a missing
*input* -- is admitted under that same section-5 extension by declaring `replaces: "computation"`, and
still carries one of the four kinds above.

Nothing is adopted into `raw` by any of this. An instrument that moves a suite row its declared gap
does not cover is rejected, in either direction (`docs/PRESETS_SPEC.md` 2 item 5; the worked run is 6.4
below).

---

## 2. The compiled connectome: cache fingerprint

Recomputed at `28e862f` with

```
python -c "from flyverse import connectome; from flyverse.interp import common; import json; print(json.dumps({k: v for k, v in common.connectome_fingerprint(connectome.load()).items() if k != 'nt_counts'}, indent=2, default=str))"
python scripts/hash_weights.py
```

| quantity | value |
|---|---|
| `sum_abs_W` | **121,460,584** |
| `nnz` | 25,578,600 |
| `n_neurons` | 167,106 |
| CSR fingerprint `md5` | **`ef23cc27bea13be7f6a96f3c04fd3737`** |
| `md5_data` | `e015d9d4007c4d2e71b604036c5e72e9` |
| `md5_indices` | `d898bcfbe222a506f82e835bafc4df8a` |
| `md5_indptr` | `718a6a97e25439ab1dc2285c39e064ad` |

Cache file md5s (`cache/`, git-ignored, ~200 MB):

| file | md5 |
|---|---|
| `neurons.parquet` | `c50c598a708b5b373cbaffca7d6a9d82` |
| `W_post_pre.npz` | `ac131529cebf98decde58d0c227b7954` |
| `sign0_counts.npz` | `bf01d724acf2a1fec8fdb60ef8a9e066` |

Receptor table `flyverse/data/receptors_by_type.csv`: md5 **`0381a446107e6050e75cc87b16d7f830`**.

These three md5s are the identity a reader can check: the cache is not shipped, so "the model I ran is
the model you ran" is `python -m flyverse.connectome` reproducing them from the manifest's files
(section 7.1). They are re-checked by hand in the audits that depend on them -- for example
`docs/audits/connectome_backends.md` section 1 and `docs/audits/compass_standin.md` (skeptic claim 1),
both of which confirm all three plus the compiled-CSR md5 live.

Transmitter overrides in force (they are part of the fingerprint):
`TYPE_NT_OVERRIDE = {TmY14: glutamate, Mi19: serotonin, aMe8: acetylcholine}`,
`UNKNOWN_NT_OVERRIDE_REGEX = {'^(lLN|v2LN|v3LN|il3LN|l2LN|vLN|LN)': gaba}`.
Cell counts by transmitter: acetylcholine 104,044, glutamate 29,707, gaba 22,135, histamine 7,908,
unknown 2,361, serotonin 415, dopamine 395, octopamine 141.

The rebuild is deterministic: `python -m flyverse.connectome` recompiled from the feather files at
`28e862f` in 15.1 s and every one of the three cache md5s above was unchanged.

**One recorded exception, and it is a scoping rule, not a discrepancy.** The transduced-plume v5 rooms
ran against a reconstructed **frozen** cache held in isolation, not against the cluster's shared cache:
their `sign0_counts.npz` differ -- the frozen 916,626 entries against the shared 926,233 -- so those
rooms are reproducible against the frozen cache only, and the audit says so
(`docs/audits/plume_transduced.md` 7; `docs/NOTES.md` "Session 14, continued"). A run that names a cache
md5 other than the three above is reproducible only against that cache.

### 2.1 The female graphs

| dataset | release | neurons | stored pairs | CSR md5 | `sum|W|` |
|---|---|---|---|---|---|
| `fafb` | v783 (pairs >= 5) | 139,255 | 3,732,460 | `31cffad8895690b83a3b02437b37bbcc` | 44,957,232 |
| `banc` | v888 (pairs >= 3) | 157,789 | 3,036,600 | `27e330891b62641686f64fd7a1b66138` | 22,084,940 |

Their fingerprints carry `dataset`, `release` and the compile manifest (source sha256s, pair
threshold, NT rule, alias hash) alongside the CSR hashes; the MaleCNS fingerprint dictionary is
unchanged, including its key set (`docs/CONNECTOME_BACKENDS_SPEC.md` 4.1).

---

## 3. Dataset versions

### 3.1 MaleCNS v1.0 (minconf 0.5)

Janelia FlyEM flat-connectome release, <https://male-cns.janelia.org/download/>; paper Cell 2026,
S0092-8674(26)00942-6. Licence: Janelia FlyEM data terms (CC BY 4.0 at the time of writing).
`flyverse/data/manifest.json` holds the URL, sha256 and size of each file.

| file | sha256 | used by |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` | cache rebuild, brainmap |
| `body-neurotransmitters-male-cns-v1.0.feather` | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` | cache rebuild |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` | cache rebuild |
| `tbar-neurotransmitters-male-cns-v1.0.feather` | `bade84c9eab431dd537ff644aaf3d203d639a819c739ecedb338e7d109064f4d` | `scripts/audit_nt.py` only |

### 3.2 FlyWire FAFB v783 (female brain)

Codex public release, <https://codex.flywire.ai/faq>; CC BY 4.0.

| file | sha256 |
|---|---|
| `neurons.csv.gz` | `6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719` |
| `classification.csv.gz` | `e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663` |
| `consolidated_cell_types.csv.gz` | `8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d` |
| `column_assignment.csv.gz` | `bdf4ce7f62cc63493d53eefad3816ff2dfd08b190e97b35a492e0e453df2f0f6` |
| `connections_princeton.csv.gz` | `445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b` |
| `connections_princeton_no_threshold.csv.gz` (optional) | `62c2e562a7470bfd32cbb98e36af3daa607ba0822b37223b34f2aa45250a920e` |
| `labels.csv.gz` | `bdd4eafab2bfe30540256c84ea1513e4b1877c0c4cf03f919204b4eafae5868e` |

### 3.3 FlyWire BANC v888 (female brain + VNC)

Codex public release, <https://codex.flywire.ai/faq>; CC BY 4.0.

| file | sha256 |
|---|---|
| `neurons.csv.gz` | `40a2201554a8c34d2c4b07c8322543a07c1b3faf5acafbac363fd1a3d0fa617f` |
| `connections_princeton.csv.gz` | `8772298eb69455759c7b4a23ee1d1843f5b389965e312524e652415f8a09b312` |

### 3.4 External expression tables

Six transcriptomic sources feed `flyverse/data/receptors_by_type.csv` (Ozel 2021, Davis 2020,
Kurmangaliyev 2020, Nern 2025, Fly Cell Atlas / Davie 2018, and the typing tables). They are **not
redistributed**: `data/external/` is git-ignored, and `flyverse/data/manifest.json` records the URL,
sha256, citation and licence of all 55 files. The built CSVs are committed, so running the model does
not need them. `python scripts/fetch_data.py --external all` fetches them for a rebuild.

---

## 4. What is and is not reproducible

### 4.1 Which execution paths repeat, and which do not (the determinism gate)

The governing measurement is the **deterministic-kernel gate**, round 8 item 1, batch `det1`
(`docs/audits/determinism_gate.md`). Each protocol ran **twice, sequentially, in one job on one pinned
GPU**, and every saved array and numeric metric was compared exactly under a rule frozen before
submission -- `numpy.array_equal(equal_nan=True)` per array after a shape check, `==` per numeric JSON
leaf, bookkeeping keys excluded, **no tolerance and no partial credit** (`determinism_gate.md` 2).

| pair | protocol | path | verdict | equal |
|---|---|---|---|---|
| cxS | `cx_wedge` round-7 S arm, seed 0, B=1, no world, no optic lobe | torch (`cuda_kernels` off, `event_driven` False, `cuda_sparse torch`) | **repeats exactly** | 31 / 31 arrays, 546 / 546 metrics, max abs diff 0 |
| plume_native | the plume room, 10 s, B=6, seed 0 | native (`cuda_kernels`, `event_driven`, graphs) | **one draw** | 1 / 15 arrays, 992 / 1026 metrics; first differing frame **30** |
| plume_torch | the same room | torch | **one draw** | 1 / 15 arrays, 992 / 1026 metrics; first differing frame **60** |
| raw_native | the plain room, 10 s, B=6, seed 0 | native | **one draw** | 1 / 14 arrays, 142 / 178 metrics; first differing frame **4** |
| raw_torch | the same room | torch | **one draw** | 1 / 14 arrays, 142 / 178 metrics; first differing frame **86** |

(`determinism_gate.md` 3. The one equal array in every room pair is `final__g_slow`, all zero.)

**What repeats.** The `cx_wedge` protocol repeats exactly on one B200, and that is the **third** exact
repeat of the same protocol: `cx8 == cx8r` across two submissions twelve minutes apart on one node, and
`cx9 == cx8r` **across two nodes** -- 76 of 76 (arm, key) metric rows, extended by the independent
skeptic to the ledger arrays, which are bit-equal too at **384 / 384** over the twelve S / V runs
(`determinism_gate.md` 4.1 and skeptic claim 6; `docs/audits/compass_sign_control.md` 3). By
`docs/INTERP.md` 10.4 rule 16 this is written "reproduced exactly in n draws on the torch path", never
"bit-identical".

**What does not.** Neither room path repeats. The native path diverges within the first 4 (raw) or 30
(plume) frames, the torch path at 86 (raw) or 60 (plume) -- and once it does, the two runs are as
different as two seeds: final per-cell rates up to 127-146 Hz apart, final voltages up to 56-111 mV
apart (`determinism_gate.md` 3).

**How far it is localised, and how far it is not.** The first difference in every room pair is exactly
**+-1 spike in exactly one of the six rows** (plume_native row 0 at frame 30, plume_torch row 4 at 60,
raw_native row 5 at 4, raw_torch row 0 at 86); that row's body diverges 1-51 frames later and its
antenna / plume columns later still, while the other five rows stay bit-identical for a further 215-329
frames -- so the body integrator and the air / odour field are **not** the source. The Python layer is
excluded too: the same B=6 raw room on the torch path repeats bit-exactly across two separate CPU
processes for 120 frames, with Python hash randomisation on. On the native path the event-scatter CUDA
kernel accumulates with a float `atomicAdd` (`flyverse/kernels/neural.cu:122`), which is
order-nondeterministic by construction and is a **sufficient mechanism for that path**. Which stage of
the room stack does not repeat on the torch path -- batched cuSPARSE SpMM, the optic lobe, the ray
tracer or an atomic in the readout -- is **not established**, and is the open question the gate leaves
(`determinism_gate.md` 4.2 and skeptic claim 2).

**An existing, unused knob.** `scripts/benchmark.py --deterministic` already sets
`torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; no round has run the
gate under it (`determinism_gate.md` 4.2, `open_questions`).

### 4.2 The earlier finding: the GPU rollout is not seed-reproducible

**Two identical runs of one source tree at one seed on one GPU diverge.** The round-1 finding, which the
gate above generalises and supersedes as the governing measurement: on a B200,
`scripts/skeptic_proprio_bitid.py` ran a 6,000-frame, 16-fly room rollout three times from the same tree
at the same seeds. From `out/proprio_bitid/compare.txt`:

```
reference out/proprio_bitid/head.json: tree /root/runs/propbitid-dbee52/_skepthead kwarg False device NVIDIA B200 frames 6000

out/proprio_bitid/work.json: tree /root/runs/propbitid-dbee52 kwarg False device NVIDIA B200
  every checkpoint hash equal: False (first difference at frame 500)
  per-row table identical:     False
  hops_total               5 vs 7   <-- DIFFERS
  hops_escape_total        5 vs 6   <-- DIFFERS
  hops_voluntary_total     0 vs 1   <-- DIFFERS
  gf_max_walk_median_hz    29.913812279701233 vs 31.24690580368042   <-- DIFFERS
  cache_sum_abs_W          121460584.0 vs 121460584.0
  ...
  VERDICT: NOT identical
```

All three comparisons (`work.json`, `work2.json`, `work_kwarg.json`) read *first difference at frame
500* and *VERDICT: NOT identical*, while `cache_sum_abs_W` is the same 121,460,584 in every one: the
model is identical, the rollout is not.

### 4.3 How a room number may be quoted

The rule frozen in `out/det1/predeclared.json` before submission, and binding on every room number in
this repository (`docs/audits/determinism_gate.md` 2 and 4.3):

* **Runs are the replicate unit**, not seeds. `flyverse.interp.common.compare` returns `underpowered`
  below four runs in either arm; `docs/INTERP.md` 10.1 step 3 requires `min(n_a, n_b) >= 4`, five when
  the effect is small.
* **Rooms run at >= 6 draws, and no room number is quoted to more than its across-draw SD.**
* **Every room number quoted before the gate stays one draw** -- explicitly, the `plume_steering.md`
  first-contact times (32.0 ... 21.9 s) and the 0.841211 end energy, and the `compass_standin` room
  repeats, which stay observations.
* **Two same-code batches with the same seeds are not the same draws.** Re-running a batch is a new set
  of replicates, never a check of the old one.
* A difference between two GPU runs of the same tree is not evidence of a model change; a difference
  between two *arms* needs the replicate count above and `common.compare`'s verdict.

One limit of the gate itself, stated in its own self-review: it tested **B=6** rooms at 10 s. B=1
room-level determinism is untested there and untested anywhere, so no rerun of a B=1 room is claimed to
reproduce (`determinism_gate.md` 6; `docs/audits/receptor_verification.md`, the plume-rooms NOT CHECKED
list).

### 4.4 Bit-identity is a CPU claim, and this is what pins it

`tests/test_bit_identity.py` pins the default path with a recorded golden: one deterministic CPU
scenario over a synthetic 19-cell, 4-column graph that touches all four senses, `stimulate`,
`set_drive`, fractional-ms carry-over, a `state_dict` round trip, a per-row `reset` and a full
`reset`, hashed after each stage. The five digests cover every brain and optic state tensor, the clock
accumulators, the RNG state, the held sensory Poisson field and every `MotorRates` field. It is run as

```
CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_bit_identity.py -q
```

**The golden's provenance.** It was recorded on **2026-09-13**, on the merge of `feat/extensibility`
into `main`, and verified identical on the pre-merge tree (`10ad8cc`, which has neither
`flyverse/modules.py` nor the hooks / `attach` / `surrogate_grad` surface) by running the same file
there: five stages, five equal digests. That equality **is** the cross-version claim of
`docs/audits/extensibility_review.md` 2 -- 5,254 arrays, 0 differing, across three source trees with an
external harness -- moved out of prose and into code (the file's own module docstring). The independent
skeptic of the instrument branch confirmed the golden has been byte-identical since `96cfdaf`
(2026-09-13), which predates that branch (`docs/audits/compass_standin.md`, skeptic claim 1).

**Regenerating that golden is an owner decision, not a test fix**: a failure means the simulated model
moved, and the printed block is pasted over `GOLDEN` in the same commit as the change that caused it.
Never regenerate to make a red test green.

The same file also pins the presets (`docs/PRESETS_SPEC.md` 1): `preset="raw"` reproduces all five
digests, `preset="instrumented", instruments=[]` reproduces the whole golden dictionary, and the
installed weights' md5 and the spike train are identical between `FlyBrain(...)` and
`FlyBrain(..., preset="raw")`. The same CPU-only rule applies to `tests/test_body_cycle.py` (atol 0 on
every brain tensor with a `LegCycle` attached and the sense off), `tests/test_optic_hooks.py`,
`tests/test_unitary.py` and `tests/test_receptor_model.py`.

Nothing in this repository claims bit-identity of a GPU rollout; `docs/INTERP.md` 10.4 item 2 states it
as a rule ("bit-identity is a CPU / `gain_fb=0` property") and item 16 restates it after three
violations in one round.

On GPU, three further sources of non-identity are documented and expected, beside the gate of 4.1: CUDA
graphs match the eager spike train for ~130 frames and then diverge by a spike (chaos);
`--weight-dtype float16` and a coarser module clock change spikes from the start; and the same code on
two GPU *models* can cross a verdict gate differently (`docs/audits/object_samedevice_r3.md`: the
reference arm crosses the effect gate on a B200 and not on an H200).

---

## 5. Commits, and the 2026-09-17 history rewrite

### 5.1 The commits

| commit | date | what it is |
|---|---|---|
| `bffbb0a` | 2026-09-18 | **The current tree.** The merge of the transduced-plume v5 rooms and their skeptic pass. Nothing adopted; `raw` untouched (`docs/audits/plume_transduced.md` 0; `docs/NOTES.md` "Session 14, continued"). |
| `03ddeb2` | 2026-09-17 | Round 8 recorded: the determinism gate, the V- sign control, the instrumented suite column and the goal-only plume arm, with both skeptic passes applied. Nothing adopted; no default moved (`docs/NOTES.md` "Session 14, round 8"). |
| `f9e9fea` | 2026-09-14 | **The round-3 reproducibility anchor.** The whole round-3 working tree in one commit, so every batch's uncommitted cross-task dependencies have a history. `provenance.source_fingerprint` covers **44 files** at this tree (`flyverse/*.py`, `flyverse/interp/*.py`, `flyverse/data/receptors_by_type.csv`, `scripts/probe_object_sweep.py`, `scripts/interp_export.py`; verified by enumerating that commit against its own `export.SOURCE_PATTERNS`). Compiled-W md5 `ef23cc27bea13be7f6a96f3c04fd3737`, receptor-table md5 `0381a446107e6050e75cc87b16d7f830`. **Nothing was adopted in round 3 and no default moved**, so the shipped model this document describes is unchanged by it. |
| `7937f01`, `cfaa694`, `28e862f` | 2026-09-14 | The FAFB / BANC connectome backends and their review. **The tree sections 1-3 were recomputed at.** MaleCNS identity preserved exactly: the three cache md5s and the CSR fingerprint are unchanged, the `test_bit_identity.py` golden is unchanged, and the legacy fingerprint dictionary keeps its key set (`docs/audits/connectome_backends.md`). `source_fingerprint` covers **51 files** here (the five `flyverse/backends/*.py`, `flyverse/data/type_aliases.csv` and `flyverse/data/manifest.json` were added to `SOURCE_PATTERNS`). |
| `10ad8cc` | 2026-09-13 | Round 2 (object + dynamics): the matched object assay, the optic stream hooks, the proprioceptive transducer, `walk.power_max` de-scored. |
| `560aaf3` | 2026-09-12 | Receptor round 5: the GF x0.3 input damping retired. **A default model change.** |
| `79769c3` | 2026-09-12 | Receptor round 3: `receptor_model='sign'` / `'abs'` adopted as the default. **A default model change.** |
| `069deb0` | 2026-09-11 | Session-9 audit: the 14-section benchmark suite, the compass ring attractor, the NT-sign audit. |

### 5.2 The rewrite, the commit map, and how a recorded id is read

On **2026-09-17** the whole history was rewritten, not the tip: `git filter-repo` over all **161**
commits with a replacement map -- hostnames, the cluster user, the shared-filesystem root, the
scheduler's name, three rented-box IPs and the workstation home path replaced by `<cluster-host>`,
`<cluster-node>`, `<cluster-node-2>`, `<cluster-user>`, `<cluster-fs>`, `<scheduler>`,
`<rented-box-ip>`, `<workstation-home>` -- applied to blobs **and** commit messages, verified by a
full-history grep returning nothing. Every commit id changed, the ten remote branches were force-pushed,
and a pre-rewrite bundle is kept outside the repository (`docs/NOTES.md`, "Session 13, continued: the
history rewrite").

Three consequences, and they are `docs/INTERP.md` 10.4 rule 30:

1. **Content identity survived; id identity did not.** Blob hashes are unchanged except in the scrubbed
   files, so every frozen source-hash set still ties to its (renamed) commit -- content identity is the
   durable record (rule 6, and 4.4 above). What went stale is every audit sentence, reproduction command
   and analyser default that *named* a commit.
2. **The map is committed**, as
   [`audits/commit_map_2026-09-17.json`](audits/commit_map_2026-09-17.json): **161 entries, 93 of which
   are renamed and 68 unchanged.** The 68 are exactly the oldest 68 commits, in history order: an id
   survives only while neither the commit nor any ancestor was touched, so the map splits the history at
   the first scrubbed commit and everything after it moved. Every tracked text file was swept and
   remapped, so **the ids in section 5.1 are
   post-rewrite ids** -- `f9e9fea`'s pre-rewrite id is `eac71e0`, `28e862f`'s is `4db2e8c`, `d2abf3c`'s
   is `6ec2de1`, and `069deb0` is one of the 68 that did not move.
3. **Frozen records are NOT edited**, whether under ignored `out/` directories or tracked, so they keep
   their pre-rewrite ids. An analyser that `git show`s a recorded commit resolves it through the map
   with `flyverse.interp.common.resolve_commit`, which returns the id unchanged when it still resolves,
   chases old -> new across successive rewrites, and **raises** rather than silently reading the wrong
   tree on an ambiguous abbreviation or an id no map knows (`flyverse/interp/common.py`;
   `tests/test_commit_map.py` covers both, plus `commit_equivalent` and map composition in date order).
   `scripts/compass_driver_analyse.py`, `scripts/cx8_transfer_verify.py` and
   `scripts/navigation_analyse.py` all read recorded commits through it.

The quoting rule that follows: an audit writes "`<new id>` (pre-rewrite `<old id>`)" **only** where the
old id is needed to find a frozen record, and the new id alone everywhere else. After any future
rewrite, sweep every tracked text file for quoted ids, remap them from that rewrite's own map, commit it
as `docs/audits/commit_map_<date>.json`, and keep the pre-rewrite repository as a bundle outside the
repository.

---

## 6. The benchmark tables, and which commit each was produced at

There are two, and they are not the same table.

### 6.1 The 14-section per-check suite (`docs/audits/benchmark_suite.md`)

Its printed table is **2026-09-11 14:07 on an RTX 4090**, `out/benchmark_suite.json`, at the
session-9 audit commit **`069deb0`**. **That table predates the round-3 anchor and does not describe
the shipped model.** The JSON's own `config` block proves it:

* `type_path_gain` still carries the GF damping `(^(SAD073|GNG300|DNp70|CL367|PVLP010)$, ^DNp01$, 0.3)`,
  which was **retired at `560aaf3`** (receptor round 5) and is now only available as
  `brain.GF_DAMPED_TYPE_PATH_GAIN`;
* `gf_hz = 38.0`, where `body.Flight.gf_hz` has been 33 since session 9;
* the receptor model was not yet the default (adopted at `79769c3`).

So none of `walk.power_max_hz 79.06 FAIL`, `loom_escape.GF_peak_hz 37.12 FAIL`,
`loom_escape.escapes 0 FAIL` in that table is a statement about the shipped model. The skeptic's
corrections at the foot of that audit already restate three of them at HEAD.

### 6.2 The 29-check suite at the shipped defaults (`docs/audits/guard_suites_r3.md`)

The current, authoritative per-check numbers. Cluster batch **`guard7-97ce35`** (2026-09-14, rented
H200 boxes, **27 jobs, 0 failed**), three independent draws per arm of
`scripts/retire_measures.py --sections all --seeds 0,1,2 --timeout 60`, every run recording NVIDIA
H200, torch 2.11.0+cu128 and compiled-connectome md5 `ef23cc27...`.

**Tally, every draw of the shipped arm: 27 PASS / 0 FAIL / 2 KNOWN GAP / 0 MISSING** (the two gaps
are `object.LC10a_flip_hz` and `compass.wedge_cells_persisting`). `walk.power_max_hz` is reported,
not scored (`scripts/benchmark.py:85`, the `notnone` form, de-scored from the data in
`docs/audits/anti_runaway.md` round 6), and `loom.escape_cm` is the other `notnone` row; both count
as PASS whenever measured, so a 27/0/2 tally contains them.

The source tree of that batch was HEAD **`d2abf3c`** plus an inert `body.LegCycle` (recorded in
`out/guard_r3/submit_tree.txt`; every other file md5-equal to `git show HEAD:<file>`). Round 3 was
then committed as `f9e9fea` and changed no default, so the shipped model of `d2abf3c + inert
LegCycle` and of `f9e9fea` is the same model.

Note, against 4.1: that batch's "0 failed" line is a job-count line, and the per-check numbers come
from three draws per arm on a path that does not repeat run to run. Read the suite rows with their draw
spread, not as pinned values -- 6.4 measures exactly which of the 29 rows repeated under identical code
in the later `suite-inst` batch.

### 6.3 The behaviour-status table (`docs/BENCHMARK_BATTERY.md`)

Assay-level statuses (`pass` / `partial` / `program` / `gap` / `fail` / `untested`), not a per-check
run. Its prose and numbers were last updated at **`f9e9fea`** (round 3). Its machine-readable form is
`flyverse/data/expected_responses.csv`, scored by `scripts/interp_ledger.py`.

### 6.4 The `instrumented` column beside `raw` (`docs/audits/instrumented_suite.md`)

Round 8 item 3, batches `suite-inst` and `suite-inst-room`, 2026-09-18. Three draws per preset,
`benchmark.py --sections all` at draw seeds 0, 1, 2, six jobs in one submission, each pinned to one GPU
of a four-id pool. The instrument list is round 7's: `sided_turn_afferent:k=0.5`, `ring_dc_hold`,
`glno_sign` (`instrumented_suite.md` 2).

| arm | pass / fail / gap / missing, by draw seed |
|---|---|
| raw s0 / s1 / s2 | 27 / 0 / 2 / 0; 26 / 1 / 2 / 0; 27 / 0 / 2 / 0 |
| instrumented s0 / s1 / s2 | 27 / 0 / 2 / 0; 26 / 1 / 2 / 0; 27 / 0 / 2 / 0 |

**No row changes status in any draw.** The only row whose statuses are not uniform is `taste.MN9_hz`
(FAIL in draw 1 under both presets, 1.69046 Hz against `> 2`): raw's own instability, identical under
the instruments, reported and not counted. The list is therefore **admissible** under
`docs/PRESETS_SPEC.md` section 2 item 5 -- and admissible is all it is: nothing adopted, `raw` stays the
default, the afferent's law is still `unverified` (`instrumented_suite.md` 4).

That batch also measures the reproducibility of the suite itself, which is what makes it useful here:
the raw column **reproduces exactly on 18 of the 29 rows and differs on 11**, both against
`compass_standin`'s raw column and against this batch's own first submission -- a same-code repeat of
the raw arm. The 11 that do not repeat are `loom.GF_peak`, `rotate.DNp20`, `motion.min_dsi`,
`loom_escape.GF_peak`, `walk_gf.p99`, `rotation.group_flip`, `object.LC10a`, `wind.DNp18`, `wind.DNp33`
and the two odour rows (`instrumented_suite.md` 2).

---

## 7. How to regenerate

### 7.1 The cache

```
python scripts/fetch_data.py --malecns                                  # ~3.7 GB, hash-verified
python -c "from flyverse import connectome; connectome.load(rebuild=True)"
python -m flyverse.connectome                                           # the same, then prints N and nnz
python scripts/hash_weights.py                                          # compiled-W md5 and sum|W|
python -c "from flyverse import connectome; from flyverse.interp import common; print(common.connectome_fingerprint(connectome.load())['md5'])"
```

Expect `N = 167106  nnz = 25578600`, `sum|W| 121460584`, md5 `ef23cc27bea13be7f6a96f3c04fd3737`.
The female caches:

```
python scripts/fetch_data.py --fafb --banc
python -c "from flyverse import connectome; connectome.load(dataset='fafb'); connectome.load(dataset='banc')"
python scripts/check_connectome_backends.py
```

### 7.2 The benchmark tables (GPU)

```
python scripts/benchmark.py --sections all --seeds 0,1 --json out/benchmark_suite.json     # 14 sections, ~4 min on a 4090
python scripts/retire_measures.py --sections all --seeds 0,1,2 --timeout 60                # the 29-check suite, one draw
bash scripts/guard_suites.sh                                                               # the three-draw guard batch (needs .cluster.json, which is git-ignored)
bash scripts/guard_suites.sh --report                                                      # aggregate to out/guard_r3/guard_report.md (CPU)
```

Repeat any table **three times or more** before believing a 20 % change, and read section 4.1 first: in
the `suite-inst` batch 11 of the 29 rows did not repeat under identical code (6.4). Score the ledger on
whatever was produced, on CPU:

```
PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py --results "out/interp/*/*.json" out/benchmark_suite.json --json out/interp/ledger/all.json
```

### 7.3 The CPU test subset (no data, no cache, no GPU)

```
python -m pytest -m "not gpu and not data and not cluster" -q
```

### 7.4 The bit-identity golden and the determinism gate

```
CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_bit_identity.py -q      # 4.4; a failure means the model moved
CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_commit_map.py -q        # 5.2, resolve_commit against a throwaway repo
PYTHONIOENCODING=utf-8 python scripts/determinism_gate.py analyse --runs out/det1 --out out/det1/analysis
```

The third command is section 4.1's, and it is the one command in this section that **a reader of the
repository cannot run to completion**. The gate's per-run JSON / NPZ records, consoles and client log
are git-ignored under `out/det1/`, because they carry host paths, so `analyse` reproduces the analysis
only where the fetched batch still exists; run against a clean checkout it reports one `missing run
JSON(s)` problem for each of the five pairs of 4.1 and decides **UNDETERMINED** under the frozen rule.

What a reader of the repository can check is what is committed: `analysis/{pairs,arrays,runs}.csv`,
`analysis.md` and `summary.json`. Where the batch is present, re-running `analyse` reproduces the four
files it writes -- `pairs.csv`, `arrays.csv`, `runs.csv` and `analysis.md` -- byte-identically with 0
problems, with one recorded exception, `summary.json`'s own `analysis_sha256`, which is the hash of the
script and moved when `plan --ship` was added. The other three committed files, `batch.sh`, `jobs.json`
and `predeclared.json`, are plan artefacts that `analyse` does not touch: `batch.sh` was regenerated by
`plan --ship flyverse` after the runs, and `predeclared.json` is **not** re-stamped
(`docs/audits/determinism_gate.md` 5).

---

## 8. Which README number is quoted at which commit

Two suite batches are quoted across the release documents and they are **not** interchangeable:
`README.md`'s ledger and its behaviour sentences quote the **round-8 `raw` column** of
`docs/audits/instrumented_suite.md` section 2 (batch `suite-inst`, 2026-09-18), while `docs/RESULTS.md`
quotes the **round-3 guard batch** `docs/audits/guard_suites_r3.md` section 1 (batch `guard7-97ce35`).
The two differ on exactly the rows 6.4 identifies: the raw column reproduces on 18 of the 29 rows and
differs on 11, under identical code. Every row below therefore names its batch.

| number in `README.md` | measured at | source |
|---|---|---|
| 167,106 neurons / 25,578,600 stored pairs / `sum|W|` 121,460,584 / md5 `ef23cc27...` | `28e862f` (recomputed 2026-09-14) | section 2 above |
| FAFB 139,255 cells, BANC 157,789 cells | `28e862f` | section 2.1; `docs/audits/connectome_backends.md` |
| 29-check suite tally **27/0/2, 26/1/2, 27/0/2** over three raw draws (README's ledger caption). `docs/RESULTS.md` quotes the round-3 batch instead: 27 PASS / 0 FAIL / 2 KNOWN GAP in every draw of the shipped arm | round 8, batch `suite-inst` (2026-09-18); round 3 at `d2abf3c` + inert `LegCycle` = `f9e9fea`'s defaults | `docs/audits/instrumented_suite.md` 2 and `guard_suites_r3.md` 1 (batch `guard7-97ce35`); 6.4 |
| Behaviour statuses (optomotor `partial`, escape `pass`, landing `fail`, ...) | `f9e9fea` | `docs/BENCHMARK_BATTERY.md` |
| `motion.min_dsi` **0.241-0.246** with the correct preferred direction 8/8 in three raw draws (README); 0.241 and 8/8 in 9 of 9 round-3 runs (RESULTS) | round 8, batch `suite-inst`; round 3 as above | `docs/audits/instrumented_suite.md` 2; `guard_suites_r3.md` 1; 6.4 |
| Loom `loom_escape.GF_peak_hz` **48.1 / 55.3 / 48.8 Hz** with an escape in all three raw draws, walking `walk_gf.p99_hz` **13.5-22.5 Hz** against the 33 Hz threshold (README); 46.7-49.5 Hz, escape in 9 of 9, p99 17-25 Hz in the round-3 batch (RESULTS) | round 8, batch `suite-inst`; round 3 as above | `docs/audits/instrumented_suite.md` 2; `guard_suites_r3.md` 1; 6.4 |
| GF spikes at ~3.5 cm range (`loom.escape_cm`, a `notnone` reported row) | session 4 reference, still the reference in `scripts/benchmark.py` | `docs/audits/benchmark_suite.md`, `probe_loom.py` |
| Sugar -> MN9: calibrated MN9 **3.93-5.52 Hz** and 0 Hz with bitter added over three raw draws (README); Shiu rules 139.9 -> 0.8 Hz, calibrated 5.5 -> 0 Hz and `taste.MN9_hz` 10.93 identical in all nine round-3 runs (RESULTS). `taste.MN9_hz` is the one row whose status is not uniform in round 8: FAIL at 1.69046 Hz in draw 1 under both presets | round 8, batch `suite-inst`; round 3 as above | `docs/audits/instrumented_suite.md` 2; `guard_suites_r3.md` 1; 6.4 |
| Wind: `DNp18_flip_hz` **+44.8 to +47.0**, `DNp33_flip_hz` **-50.3 to -50.7** (README); +44.7 to +45.8 / -49.4 to -50.0 in the round-3 batch (RESULTS) | round 8, batch `suite-inst`; round 3 as above | `docs/audits/instrumented_suite.md` 2; `guard_suites_r3.md` 1; 6.4 |
| Rotation: `rotation.group_flip_hz` -8.7 to -10.3 (round-3 batch) | `d2abf3c` + inert `LegCycle` = `f9e9fea`'s defaults | `docs/audits/guard_suites_r3.md` 1 |
| Odour: apple channel 17.3-17.5 Hz at 8 cm vs 4.3-4.9 Hz plume-free (round-3 batch) | `d2abf3c` + inert `LegCycle` = `f9e9fea`'s defaults | `docs/audits/guard_suites_r3.md` 1 |
| Take-off: 3.0 voluntary + 2.0 escape per 1,000 fly-s at the default | receptor round 5 / `560aaf3`, replicated round 3 | `docs/BENCHMARK_BATTERY.md`, `docs/audits/receptor_integration.md` G.5-G.6 |
| Straight walking: clean-frame yaw SD 2.56 deg/s (2.45-2.57 with no fence), straightness 0.995 +/- 0.001; DNa02 tonic inhibition -1.6 / -2.0 mV against a 7 mV gap | `4d55f96` (localization), `f9e9fea` (round-3 replication) | `docs/audits/deficit_turning.md`, `body_sided_state.md` |
| Leg cycle + sided haltere: yaw SD -> 7.7-7.9 deg/s, DNa02 0.54 / 0.38 Hz, no frame above 100 deg/s, fixed left drift | `f9e9fea` | `docs/audits/body_sided_state.md`, `round3_integration.md` |
| Compass: 0/48 bumps at shipped gains under every per-transmitter unitary bracket; bump drift <= 0.005 wedges/s vs 4.0 ideal | `f9e9fea` | `docs/audits/unitary_strength.md` 4, `body_sided_state.md` 6 |
| Small object: LC11 12/12 `null`, LC10a fails Holm (p_holm 0.104); no passing mechanism in 8 arms | `10ad8cc` (round 2) | `docs/audits/object_matched_assay.md`, `object_compare_r2.md` |
| Object round 3: localizer negative, 40/40 rectangle verdicts `null`, same-device H200 `REPRODUCES` / B200 `PARTIAL` | `f9e9fea` | `docs/audits/object_localizer_r3.md`, `object_rectangles_r3.md`, `object_samedevice_r3.md` |
| Cross-connectome walking: MaleCNS 2.641 +/- 0.146 vs BANC 0.275 +/- 0.004 deg/s clean yaw SD, leg-cycle arm 7.837 vs 2.651 | `cfaa694` (merged at `28e862f`) | `docs/audits/connectome_backends.md`, batch `cbwalk-384afd` |
| Anatomy scale MaleCNS : FAFB : BANC = 1 : 0.59215 : 0.28127 | `cfaa694` | `docs/audits/connectome_backends.md` |
| `cx_wedge` repeats exactly on one GPU (31/31 arrays, 546/546 metrics); no B=6 room repeats on either path (first difference at frame 4 / 30 / 60 / 86) | round 8, batch `det1` (2026-09-18) | `docs/audits/determinism_gate.md` 3-4; section 4.1 |
| `cx9` reproduces `cx8r` across two nodes in 76 of 76 metric rows, and 384 / 384 ledger arrays are bit-equal | round 8, batch `cx9` (2026-09-18) | `docs/audits/compass_sign_control.md` 3; `determinism_gate.md` skeptic claim 6 |
| The `instrumented` column changes no suite row's status (27/0/2, 26/1/2, 27/0/2 under both presets) | round 8, batches `suite-inst` / `suite-inst-room` (2026-09-18) | `docs/audits/instrumented_suite.md` 2-3; section 6.4 |
| Plume rooms, six 60 s seed-drawn starts per arm: full **6/6**, transduced **1/6**, goal_only **3/6** fed for >= 1 s | session 14, batch `plume_transduced_v5-6ebfff` (2026-09-18) | `docs/audits/plume_transduced.md` 0 and 7.2 |
| Demo speed 0.5x / 0.8x / 1.3x real time on a 4090 | session 10 | `docs/NOTES.md`, `docs/PERFORMANCE.md` |
| CPU test subset green | `bffbb0a` (2026-09-19, the release pass, this desktop, cache present) | `520 passed, 34 skipped, 30 deselected, 220 subtests passed in 203.55 s` (`RELEASE_CHECKLIST.md` 4). The earlier `364 passed, 7 skipped, 29 deselected ... in 144.62s` at `28e862f` is superseded: the suite has grown |

### 8.1 The two milestone sentences, and their licensed wording

Both are behavioural claims about the **instrumented** preset, not about `raw`, and both are quoted with
the scope their audits license.

1. **Turning and finding food on the physical antennal contrast.** Under `instrumented` the fly turns
   and finds food in a 60 s room with no oracle in the loop and `raw` untouched: the shipped `plume`
   instrument, which reads the **physical** odour-concentration difference between the antennae, fed in
   **6 of 6** rooms, while the opt-in `plume:bilateral=orn` variant, which substitutes the model's own
   ORN population rates at the same gain, fed in **1 of 6** and steered on a cue **uncorrelated with the
   true lateral contrast** (`docs/NOTES.md` "Session 13, the instrumented preset" for the 6/6 milestone
   and the no-oracle scope; `docs/audits/plume_transduced.md` 0 and 7.2 for the three-arm comparison;
   `docs/audits/plume_steering.md` for the original rooms). This is a **descriptive room observation** --
   not a significance test, not an SNR measurement, not a claim about flies -- and in particular **the
   rooms do not confirm the CPU SNR finding and are not offered as confirming it**: the in-room ORN L-R
   and its temporal SD are different quantities from the CPU's fixed-contrast signal and 250 ms
   counting-window noise. What the rooms add is that the measured in-room cue noise is **1.3-2.6x** the
   analytical estimate and that the cue's sign was right in only **0.52 +/- 0.10** of samples
   (`docs/audits/plume_transduced.md` 7.3, and its rooms skeptic claim 7 -- the licensed wording).
2. **It never flies and feeds in the same episode.** "No run in these audits shows artificially powered
   flight and feeding in the same episode: the only powered-flight rooms are the superseded navigation
   flight arm (59.67 s airborne, zero feeding in all six rows), and `powered_s` is 0.00 in all 12
   flight-priority room rows and all 12 plume room rows" (`docs/audits/flight_foraging_priority.md`,
   repeated verbatim in `docs/audits/navigation_instruments.md` and `docs/audits/plume_steering.md`;
   required by the independent skeptic, item 13 in `docs/audits/receptor_verification.md`).

A number not in this table and not in a `docs/audits/*.md` file with a batch line is not a result.
