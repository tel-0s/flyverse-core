# The interpretability toolkit (`flyverse/interp/`)

**Purpose.** Localize a behavioural deficit of the connectome model to populations, synapses and links -- instead of
tuning the behaviour into the model. The project rule stands over every tool: the full plain model is not hand-tuned
toward behaviour; hand-designed systems exist only as swappable modules or documented stop-gaps; a default changes only
when the data or a mechanism the connectome implies says so. **The toolkit reads the model; it never changes it.** No
tool edits `flyverse/brain.py`, `optic.py`, `body.py`, `programs.py`, `room_demo.py` or `room_ui.py`; a tool that needs
a counterfactual (a lesion, a hold table, a gain) applies it as a `LIFParams` / `OpticParams` override, a
`--receptor-table`, or an in-process weight mask, exactly as `scripts/retire_measures.py` and
`scripts/build_hold_tables.py` already do, and records what it applied.

**Status (2026-09-12, design task; build round closed 2026-09-13).** This document, `flyverse/interp/__init__.py` (the eight signatures as stubs),
`flyverse/interp/common.py` (the shared machinery, real code) and `tests/test_interp.py` (the common tests, 11 passing
on the CPU) are the contract. Seven implementers build the tools against it in parallel; the file each owns is in
section 7. The build round shipped all eight tools (`tests/test_interp.py` 65 passed on the CPU), their validation
records (`docs/audits/interp_<tool>.md`) and three applications (`docs/audits/deficit_{turning,object,rotation}.md`);
section 10 is the procedure that came out of it, and section 11 lists the contract defects the round found in the
shared files, which are still open. Inputs this design takes as given are the dynamics-round-1 results listed in section 6 (compass, object,
take-off, feeding, `walk.power_max`), the round-4/5 attribution (`docs/audits/receptor_integration.md` E.4 / G.4) and
the structural ring analysis (`docs/audits/cx_wedge.md`, `cx_glno.md`).

Read with: `docs/NEUROME_INTERFACE.md` (the export this toolkit serves), `docs/NT_READOUT.md` (the readout-source hook
the health tool implements), `docs/BATCH_SIM.md` (rollouts), `docs/BENCHMARK_BATTERY.md` + `scripts/benchmark.py` (the
checks the lesion tool scores), `docs/audits/anti_runaway.md` (the lesion / retirement procedure), `object_sweep.md`
(the null-referenced assay every stochastic measurement copies).

---

## 1. The eight tools in one table

| tool | question it answers | inputs | GPU part (`scripts/interp_<tool>.py record` / `run`) | CPU part (`analyse`) | reuses |
|---|---|---|---|---|---|
| **decompose** | what drives this cell set, per frame, by presynaptic type / transmitter / receptor tier | a target spec; a `Recording` of its presynaptic cells (or none: structural) | record presynaptic rates over a protocol | `A[target, :] @ r(t)` split by pre group | `brain._shaped_weights` via `common.effective_weights`; `connectome.receptor_signs` for tiers |
| **trace** | where along the depth from a sensory population is a stimulus lost | stimulus / control / null `Recording`s (>= 3 runs each); a source spec | record all optic + VPN cells (or a stage list) under stimulus, control, control again | per-type z / d' vs the null, depth order, first stage lost, then `decompose` there | `probe_figure_stages` (figure z, stage table), `screen.rank` (d'), `probe_object_sweep --null` (the null) |
| **paths** | the effective k-step signed gains A -> B, and which links are silent | two specs, k <= 3; optional rollout rates / frozen set | none (optionally a rollout for `never_firing`) | type-level matrix products on `effective_weights`; flags from `silent_flags` | `scripts/cx_wedge.py structure()` (the two-step matrices), `cache/sign0_counts.npz` |
| **lesion** | check x lesion delta matrix with scatter; the double dissociations | a manifest of lesion sets, suite sections, probe CLIs | one `cluster_run.py` batch: one job per lesion x replicate (+ baseline), each `benchmark.py` sections and probes in-process | the matrix, replicate sd / n, dissociation extraction | `benchmark.py` (sections via `retire_measures.make_context`), `build_hold_tables.py`, `r5_attr_taste_cpu.py` |
| **atlas** | what every motor readout does when population X is stimulated | a population list; hz / ms; context | one `FlyBrain(batch=B)`, row r stimulates population r, null rows unstimulated; >= 3 runs | pulse-window means vs null rows, `compare` | `screen_dns.py` (protocol), `screen.TypeRecorder`, `FlyBrain.stimulate`, `motor.read_motor` |
| **health** | per-type operating point of a rollout; live channels for the map | a `Recording` with v / adapt / refrac (or a live FlyBrain for the readout source) | record the quantities | silent / at-threshold / refractory-load / E-I / fan-in / adaptation per type | `nt_readout.NTSource` (no UI edits), `effective_weights.scale`, `silent_flags` |
| **ledger** | does each probe's per-type output match the curated expectations | Result JSONs (+ benchmark JSONs); `flyverse/data/expected_responses.csv` | none | PASS / FAIL / MISSING per row | `benchmark.REFERENCES` (the first rows of the CSV) |
| **export** | the Neurome read-only probe export | any Result JSON | none | manifest + tables, SHA-256 | `docs/NEUROME_INTERFACE.md` section 1 |

Composition: **trace calls decompose** at the lost stage; **lesion calls the suite** (`scripts/benchmark.py` sections,
in-process through `retire_measures.make_context` so a weight mask can be installed) **and the probes through their
CLIs** (`probe_object_sweep.py --null`, `probe_figure_stages.py`, `probe_walk_straightness.py`, ... with `{out}` /
`{seed}` placeholders); **atlas reuses screen.py**; **health is also an NTSource**; **ledger reads every other tool's
Result**; **export serializes any Result**. Nothing is computed twice: the export is a serializer because the Result
schema (section 3) already carries everything `docs/NEUROME_INTERFACE.md` asks for.

---

## 2. The common module (`flyverse/interp/common.py`, implemented)

### 2.1 Population selection -- one grammar everywhere

```python
from flyverse.interp import resolve, population, populations
resolve(c, "DNa02")                       # exact type            -> sorted unique model indices
resolve(c, "~^LC1(0a|1)$")                # regex on type (Connectome.select's '~')
resolve(c, "LC10a|LC11")                  # types a or b
resolve(c, "type=PEN_a&somaSide=L")       # AND of clauses: key=value, key~regex, key:a|b|c
resolve(c, "module=optic&nt=histamine")   # module = regions.MODULES; any neurons column is a key
resolve(c, "body:12104|14881")            # bodyIds; "index:0|5" model indices
resolve(c, ["EPG", "PEN_a", "PEN_b"])     # a list = OR
resolve(c, {"type": "~^DNa"})             # Connectome.select criteria; index arrays / masks also accepted
population(c, spec, label) -> Population(label, spec, idx, body_ids); .record() gives the JSON block
by_type(c, idx, by_side=False) -> {type or type_side: idx}     # TypeRecorder key order
```

Every tool's `--population`, `--source`, `--target`, `--a/--b` and every lesion-manifest `spec` is this grammar, so a
lesion set, a trace source and an atlas entry name cells identically and the Result records the resolved bodyIds.

### 2.2 The shaped-weight accessor

```python
ew = effective_weights(c, params=LIFParams(), receptor=None)   # EffectiveWeights
ew.A          # csr (N, N), A[post, pre] in mV per presynaptic spike -- exactly Brain._W_cpu (tested bit-for-bit
              # on a synthetic graph under three parameter sets): brain._shaped_weights x fan-in scale x w_syn;
              # subsets normalise on the reference graph as Brain does
ew.scale, ew.tot                   # the fan-in factor and shaped input total per postsynaptic cell
ew.type_matrix(c, post_types, pre_types)   # M[post, pre] = mean over post cells of the summed input of the pre type
                                           # (mV per post cell per presynaptic volley: cx_wedge's 'total_per_post')
ew.md5, ew.shaped_md5              # hashes of A and of the shaped matrix before normalisation (scripts/hash_weights.py's)
links(c, ew, pre, post, receptor=None, counts=None, flags=None) -> DataFrame   # the edge table, Neurome names:
   # pre_index, post_index, body_pre, body_post, pre_type, post_type, pre_nt, synaptic_pair_count (raw, unsigned,
   # uncapped; from cache/sign0_counts.npz when available so sign-0 entries carry their real count), effective_mv,
   # sign, sign_rule ('nt_sign' | 'receptor:<tier>'), gain_rule (the factors in force), fanin_scale_post, silent
silent_flags(c, pre_idx, frozen_idx=fb.optic.rate_idx, prune_frozen=True, rates=recording.quantities['rate_hz'])
   # sign0 | frozen | pruned | never_firing (max rate over the rollout < 0.5 Hz) -- all four are BOOLEAN columns, and
   # with no `rates` the never_firing question was not asked, so it is False (not NaN, which every bool(flag) reader
   # turned into 'every cell never fires' and stamped on every structural table)
raw_counts(c, with_sign0=True, dtype=np.float32) -> (counts csr, sign0_available)
   # the TRUE raw count of every stored entry: C.data = maximum(|W.data|, connectome.sign0_counts) -- merged, never
   # substituted (sign0_counts is non-zero ONLY on the explicit zeros). with_sign0=False leaves those entries at 0,
   # i.e. counts only what the LIF can carry; dtype=np.float64 where an exact whole-model total is wanted.
unit_kinds(c, fb=None) -> 'spiking' | 'graded' | 'photoreceptor' per cell     # docs/NEUROME_INTERFACE.md section 2
```

`silent` is the toolkit's definition of a link that cannot carry anything in a given run: **sign 0** (the presynaptic
transmitter carries no sign, so its entries are explicit zeros in `W` -- GLNO, the monoamines, unknown NT), **frozen** (a
rate unit of the optic lobe: never spikes in the LIF; its effect on spiking cells arrives through the optic drive
instead), **pruned** (frozen and dropped from the LIF matrix under `prune_frozen`), **never_firing** (silent in the
rollout it is judged on). The first three are structural facts of the compiled model; the fourth is a property of one
rollout and must name it.

### 2.3 Recording -- the GPU / CPU seam

```python
rec = Recorder(fb.c, ["DNp01", "~^LC4$", "module=descending"], quantities=("rate_hz", "drive_mv", "v_mv", "adapt_mv", "refrac", "spike_count", "optic_dr"))
for _ in range(frames):
    sim.step(); rec.capture(sim.fb, motor=sim.fb.motor())        # B = 1 uses brain.rate_np() (the demo's cached copy)
recording = rec.finish(meta={"protocol": ..., "stimulus": ...})   # Recording; .save(path) -> path.npz + path.json
recording.per_type("rate_hz", by_side=False)  -> (keys, (T, n_keys))    # pooling = screen.TypeRecorder
recording.window(start_s, end_s); recording.row(b)   # a batch member as a single-fly recording
record_frames(step, rec, frames, every=1)            # the loop
```

Quantities: `rate_hz` (Brain.rate), `drive_mv` (the injected optic / sensory current), `v_mv`, `adapt_mv`, `refrac`
(0/1), `spike_count` (cumulative; diff for spikes per frame), `optic_dr` / `optic_rate` (OpticLobe.delta_rate /
rates() for recorded cells that are rate units, NaN elsewhere); motor fields of `MotorRates` as `motor[name]`
(`lh_odour.<channel>` per channel). A batched recording is `(T, B, n)`; **B rows of one BatchSim are one draw layout,
not replicates** (`docs/BATCH_SIM.md`): the replicate unit of every tool is `runs` = independent jobs / processes with
their own brain seed, `>= 3` per arm, submitted in ONE cluster batch.

The npz schema (`flyverse.interp.recording/1`): `t_ms (T,)`, `idx (n,)`, `body_ids (n,)`, `types (n,)`, `q__<quantity>`,
`m__<motor field>`; the json sidecar holds `meta` (protocol, stimulus, sides, window, and -- written by the CLI -- the
provenance block of section 3 so a recording is self-describing).

### 2.4 Null and replicate helpers -- the scatter rule, in code

```python
compare(stim_values, null_values, z_min=3.0, min_n=3, alpha=0.05)   # the call rule is max(min_n, CALL_REPLICATES=4) per arm
  -> {"stim": ArmStats, "null": ArmStats, "diff", "z", "welch", "U", "p", "p_floor", "null_sd_zero", "verdict", ...}
p_floor(n_a, n_b)   # 2 / C(n_a + n_b, n_a): 3 v 3 -> 0.10, 4 v 4 -> 0.029, 5 v 5 -> 0.0079
```

`z = (mean stim - mean null) / SD(null)`, Welch = the difference over the standard error of the two means, exact
Mann-Whitney U and p for small arms (`docs/audits/object_sweep.md` 8.4; the test reproduces LPLC2 sign-abs z +5.4 /
Welch +8.6 / U 25 / p 0.0079 and LC11 off z -0.1 / U 12 from the audit's numbers). The verdict, in this order:

* **`underpowered`** -- an arm has fewer than `min_n` runs, **or** `p_floor > alpha`: at this many runs the exact rank
  test cannot reach alpha however large the effect. `MIN_REPLICATES` stays 3 (three runs buy the scatter) but four per
  arm is the smallest that can be *called*, five when the effect is small.
* **`undetermined`** -- the null arm is deterministic (SD 0 up to float noise: bit-identical draws, an all-silent
  readout, `gain_fb=0`), the arms differ, and the rank test does not settle it as null. z is the criterion and z is
  undefined there: read `diff` and `p`. A tool may rank such rows by a declared effect size of its own -- the atlas'
  `z_floor` (diff over the null SD floored at 0.05 Hz, `atlas.called`), trace's `CARRIER_VERDICTS` -- but the verdict
  column is `compare`'s in every tool.
* **`result`** -- |z| >= `z_min` and p <= `alpha` (when a p exists).
* **`null`** -- otherwise, a deterministic null with no difference at all included.

Every tool that reports a difference reports this dict, never a bare z. `replicate_seeds(n, seed0)` names the runs.

Why 'runs' and not 'seeds': the figure-stage pipeline is not reproducible at a fixed seed on the GPU (median |dz| 0.4-0.5,
max 5-7; the object sweep's LC10b ball moved 0.084 -> 0.251 at the same seed), so a per-type z carries +-1.5 and the
draw is the job, not the seed value. `gain_fb = 0` (spiking feedback off) makes the optic lobe deterministic and is the
exact null arm for optic-only questions (`docs/audits/optic_measures.md` section 6).

### 2.5 Provenance -- what every JSON must carry

```python
prov = provenance(c, lif, optic, fb=sim.fb, device=args.device, seeds=[...], env_seeds=[...], batch=B,
                  stimulus={"protocol": ..., "params": {...}, "control": {...}}, retina={"file": ..., "n_columns": ..., "column_to_bodies": ...}, cache_dir=...)
```

| block | content | source |
|---|---|---|
| `flyverse_commit` | `commit`, `dirty`, `modified_files` | `git rev-parse HEAD`, `git status --porcelain` |
| `source_fingerprint` | the code's identity when git cannot give it: `{computed: false, commit}` when `git_state()` resolved the commit, else the SHA-256 of every simulation / probe source under ROOT **and** of the modules the process actually imported (`files`, `files_lf`, `files_loaded`), which `export.match_sources` matches against a checkout by content | `common.source_fingerprint` -> `export.source_fingerprint(include_loaded=True)` |
| `dataset_release` | name `male-cns`, release `v1.0 flat-connectome`, the four file names + SHA-256 | `flyverse/data/manifest.json` |
| `compiled_connectome` | `md5_data / md5_indices / md5_indptr / md5` of the reference `W` (= `cache/W_post_pre.npz` when the cache is what was loaded), `sum_abs_W`, `nnz`, `n_neurons`, `nt_counts`, `cache_dir`, `type_nt_override`, `unknown_nt_override_regex`, `subset` | in memory, memoised |
| `model` | `lif` = every `LIFParams` field with defaults RESOLVED (`path_gain`, `type_path_gain`, `adapt_by_type`, `std_u_by_type`, receptor gain classes, slow classes), `receptor_table` + md5, `optic` = every `OpticParams` field resolved (`pair_gain`, `tau_by_type`, `baseline_by_type`), `body` = `gf_hz 33`, `takeoff_power_hz 50`, `takeoff_hold_s 0.3`, `mdn_threshold_hz 15`, `k_opto` | `brain.py`, `optic.py`, `body.py` defaults (read, not edited) |
| `execution` | `device_requested`, **`device` = `fb.brain.device` (the realised one)**, `device_name`, backend flags (`event_driven`, `cuda_kernels`, `cuda_sparse`, `cuda_graphs`, `metal`), `dt` (LIF 0.5 / optic 1 / frame 10 ms), `seeds` (brain, env), `batch`, `replicate_unit`, `host`, `torch` | the FlyBrain |
| `stimulus` | protocol name and every parameter; the matched control | the tool |
| `retina` | file, `n_columns`, the column -> photoreceptor-body map (or where it is) | `retina.build_retina` (the tool records it when a retina was in the loop) |
| `units` | the unit-handling table (`common.UNITS`) | fixed |

A JSON whose `execution.device` is None, or whose provenance lacks a block, fails `Result.check()` and the export refuses
it. A cluster log that says `device cpu` means the job is resubmitted, not reported.

### 2.6 The CLI flags every wrapper accepts (`add_common_args`)

`--json PATH` (default `out/interp/<tool>/<run_id>.json`), `--replicates N` (default 3), `--seed S`,
**`--null-runs GLOB [GLOB ...]`**, `--device`, `--cache-dir`, `--receptor-model default|off|sign|sign+gain|full`,
`--receptor-net-rule`, `--receptor-table PATH`, `--lif KEY=VALUE` (repeatable, JSON / literal values),
`--optic KEY=VALUE`, `--quiet`. `params_from_args(args) -> (LIFParams, OpticParams)`. Every wrapper ends with
`print_table(...)` and the JSON path.

**One null convention.** `--null-runs` names the finished control-vs-control runs on an `analyse` subcommand and
reaches the tool as `args.null_runs` (a list of globs; the wrappers that label arms take `LABEL=GLOB`, e.g.
`--null-runs "off=out/dec/off_r*.npz"`). `--null-arm` and `--null-recordings` are hidden aliases of it, as is the
ledger's `--null`, so the round's command lines still run. A subcommand that has to **generate** the null arm
declares its own bare `--null` switch (`interp_export.py record --null`); `add_common_args` no longer does, which is
what made `analyse --null-arm X --null` silently drop the control.

### 2.7 The RF-map file (`flyverse.interp.rfmap/1`; generator `scripts/probe_synthetic_stimuli.py rfmap`)

A per-body receptive-field map measured with the synthetic localizer (a 4.5-deg dark square flashed for 200 ms at every
node of a 10-deg azimuth x elevation grid on the radiance path, `docs/audits/object_synthetic_stimuli.md`). It fixes
each body's measurement region for the matched assays (`probe_object_matched.py --rf-map`, `probe_synthetic_stimuli.py
analyse --rf-map`) and is the stimulus-driven alternative to the anatomical column of `trace.column_of_cells`. Two
files, the same rows:

* **`<name>.csv`** -- one row per recorded body (every body of the recorded types, fitted or not). The first seven
  columns are the interface every consumer reads, in this order:

  | column | meaning |
  |---|---|
  | `bodyId` | decimal string (MaleCNS body id) |
  | `type` | the cell type |
  | `az_deg`, `el_deg` | the RF centre (azimuth + left, elevation + up, degrees; `Retina.col_az_el`'s frame); **NaN when not fitted** |
  | `width_deg` | the equivalent-disc FWHM: `2 sqrt(n spacing^2 / pi)` over the nodes at or above half-maximum, at least one grid spacing; NaN when not fitted |
  | `peak` | the peak per-node response (`on` window minus the pre-onset `base`; `drive_mv` for spiking cells, `optic_dr` for rate units), signed |
  | `n_nodes_above_threshold` | the nodes at or above half-maximum of the same sign (0 when not fitted) |

  then the fit's bookkeeping: `sign` (+1 / -1), `noise_mad` (1.4826 x MAD over nodes), `z_peak` (|peak| / noise),
  **`fitted`** (|peak| >= `z_min` x noise; `z_min` 5 by default), `peak_node_az_deg` / `peak_node_el_deg`, `quantity`,
  `window` (`on` | `off` | `onoff`), `model_index`, `spikes_on_minus_base_hz_peak`; over several runs `n_runs_fitted`,
  `az_sd_runs` / `el_sd_runs`, `centre_spread_deg` (the largest pairwise great-circle distance between the runs'
  centres -- the reproducibility of the map, on file per body); and the anatomical comparison `anat_column`,
  `anat_az_deg` / `anat_el_deg` (the column of `trace.column_of_cells`), `hex_annotated`, `anat_distance_deg`
  (great-circle, fitted centre vs anatomical column). A consumer selects `fitted == True` (equivalently a finite
  `az_deg`); a body without a row, or with `fitted` false, has no localizer and its windowed statistics are NaN,
  never 0. `height_deg` is absent: the fit is isotropic and consumers default it to `width_deg`.
* **`<name>.json`** -- a `Result` (tool `trace`) whose `tables.rf_map` holds the same rows and `tables.rf_per_type` the
  per-type summary (`coverage` = the fraction of bodies fitted, `n_fitted_all_runs`, `width_median_deg`,
  `anat_distance_median_deg`, `anat_within_10deg` / `_15deg`, `centre_spread_median_deg`, and
  `false_fit_rate_blank_arm_mean` -- the same fit applied to the localizer's **blank arm**, i.e. the fraction of
  bodies the rule 'fits' from noise alone); `summary.schema` = `flyverse.interp.rfmap/1`, `summary.fit_rule` (the
  docstring of `fit_rf`), `summary.optic_overrides` (`{}` = the shipped lobe, `{'gain_fb': 0}` = the deterministic
  lobe), and the full provenance of the first run (resolved `LIFParams` / `OpticParams`, realised device, cache
  fingerprint, source fingerprint). `replicates.runs` lists the localizer runs the map averages over (several runs
  = the map is the mean centre of the bodies fitted in EVERY run) and `replicates.null.blank_arm_files` the blank
  arms behind the false-fit rate.

The per-run inputs are `<run>_nodes.npz` (stimulus arm) / `<run>_nodes_blank.npz` (blank arm): `node_az_deg`,
`node_el_deg` (n_nodes), `idx` / `body_ids` / `types` (n_cells), and per window `on__` / `off__` / `base__` x
`drive_mv` / `optic_dr` / `spikes_per_frame` as (n_nodes, n_cells) means, with `n_on` / `n_off` / `n_base` frame
counts (20 / 10 / 10 at 10 ms). The map is one localizer's magnitudes: coverage and agreement are quoted with the
centre scatter over >= 3 runs (`centre_spread_deg`) and the blank-arm false-fit rate, never alone. The `window`
column names the node window the response was read in (`on` = the flash frames, `off` = the frames after the offset)
-- the same ON / OFF role split the family analysis reports as its `transition = on | off` rows
(`probe_synthetic_stimuli.py analyse`, `object_synthetic_stimuli.md` 6.1); it is not a bright-vs-dark label.

**What is on file, and what a consumer must carry with it** (object round 2, 2026-09-13; the `verify:synthetic`
corrections). Five maps exist and only one has a replicate: `out/synth/rfmap_shipped.csv` (4.5 deg, 1 pass, **3
runs**, so `centre_spread_deg` is a real number) against `out/synth/rfmap_fb0.csv`, `out/synth2/rfmap_088_fb0_p3.csv`,
`rfmap_088_shipped_p3.csv` and `rfmap_150_fb0_p3.csv`, which are **one run each** (`n_runs_fitted` 1,
`centre_spread_deg` NaN -- no scatter, against the round rule of >= 3 runs of one localizer in one submission).
**No map on file carries a fitted LC11 row** at `z_min` 5 (0 of 143 bodies at 4.5, 8.8 and 15 deg, peak node at
chance), and LC10a rows exist only in `rfmap_150_fb0_p3.csv` (13 of 275 = 4.7 %, ONE run, `gain_fb=0`; the 15-deg
localizer was never run on the shipped lobe). So an LC body's window is normally the anatomical fallback -- the
round-2 sphere ladder windows 405 of its 418 LC bodies that way -- and any consumer that windows an LC statistic
states which map it used and that LC11 has no fitted centre in any of them (`object_synthetic_stimuli.md` 4.1 / 4.2 /
8). The maps measured with `gain_fb=0` describe the deterministic lobe, not the shipped one; `summary.optic_overrides`
is the field to read.

---

## 3. The result schema (`flyverse.interp.result/1`)

One JSON for all eight tools (`common.Result`; `Result.new(tool, provenance)`, `.add_table`, `.add_population`,
`.save`, `.load`, `.check`):

```json
{
 "schema": "flyverse.interp.result/1", "tool": "trace", "tool_version": "0.1",
 "run_id": "trace-20260912T153000Z-1a2b3c4d", "created_utc": "...",
 "provenance": { "flyverse_commit": {...}, "dataset_release": {...}, "compiled_connectome": {...}, "model": {...},
                 "execution": {...}, "stimulus": {...}, "retina": {...}, "units": [...] },
 "populations": [ {"label": "source", "spec": "type:R1-R6|R7y|R7p|R7d|R8y|R8p", "n_cells": 5895, "body_ids": ["..."], "unit_kind": "photoreceptor"} ],
 "replicates": { "n": 3, "unit": "runs", "runs": [{"run_index": 0, "seed": 0, "file": "out/.../stim_r0.npz", "device": "cuda:0"}], "null": {...} },
 "tables": {
   "per_type":          [ {"type": "Mi4", "depth": 2, "stage": "2a", "n_cells": 1374, "stim_mean": ..., "ctrl_mean": ..., "null_mean": ..., "z": 22.3, "welch": ..., "U": 25, "p": 0.0079, "verdict": "result"} ],
   "readout_per_body":  [ {"bodyId": "12104", "model_index": 88123, "type": "GLNO", "unit_kind": "spiking", "quantity": "output_Hz", "window_start_s": 3.0, "window_end_s": 15.0, "stimulus_value": ..., "control_value": ..., "stimulus_minus_control": ..., "unit": "Hz", "n_trials": 3, "trial_sd": ..., "control_ids": ["..."]} ],
   "contributions":     [ {"body_pre": "...", "body_post": "...", "pre_type": ..., "post_type": ..., "value": ..., "kind": "current", "sign_rule": "receptor:exact", "gain_rule": "...", "normalisation": "input_norm alpha 1 ref 5000", "reference_graph": "<md5>", "window": [3.0, 15.0], "synaptic_pair_count": 182} ],
   "sensitivity":       [ {"lesion_id": "holdBrainHis", "lesion_kind": "hold_table", "lesion_spec": "out/receptors_holdBrainHis.csv", "bodies": "...", "check": "taste.MN9_hz", "baseline": 5.0909, "value": 1.5548, "delta": -3.536, "replicate_sd": 0.0, "n_replicates": 3, "replicate_values": [...]} ],
   "<tool-specific>":   [ ... ]
 },
 "summary": { ... the tool's scalars ... },
 "validation": { "name": "...", "reference": {...}, "source": "...", "measured": {...}, "status": "reproduced | not reproduced | not run" },
 "files": { "recordings": [...], "log": "...", "generator": "scripts/interp_trace.py ..." }
}
```

Rules. (1) `tables` are lists of records; the three Neurome tables use the column names of `common.EXPORT_TABLES`
(`docs/NEUROME_INTERFACE.md` section 1) and `bodyId` / `body_pre` / `body_post` are **decimal strings**;
`synaptic_pair_count` is raw, unsigned, uncapped; coordinates, when a tool adds them, carry `space`, `axis_order`, `unit`.
(2) LC11 and LC10a get two `readout_per_body` rows per body, `quantity` = `upstream_drive_mV` and `output_Hz`, never a
pooled one. (3) `validation` is prefilled from `common.VALIDATION[tool]` (section 6) and the tool fills `measured` and
`status`. (4) `populations[].body_ids` is kept unless the population is > 10,000 cells (then `n_cells` + spec and the
bodies live in `readout_per_body`). (5) numbers come out of `to_jsonable` (numpy -> plain; NaN -> null).
(6) `Result.check()` lists what would make the export refuse the file; every wrapper prints it when non-empty.

---

## 4. The tools (contract = `flyverse/interp/__init__.py::stubs`; every parameter there is kept by the implementation)

**Two namespaces.** `flyverse.interp.<tool>` is the **module** (`from flyverse.interp import trace as tr`, what every
wrapper does) and `flyverse.interp.tools.<tool>` / `flyverse.interp.tool('<tool>')` is the **function** -- the
implementation when `flyverse/interp/<tool>.py` defines it, `stubs.<tool>` (NotImplementedError naming the file to
write) until then. Neither depends on what has been imported already, which is what the contract test now enforces.

### 4.1 decompose -- `flyverse/interp/decompose.py`, `scripts/interp_decompose.py`

```python
decompose(c, target, *, recording=None, params=None, optic_params=None, receptor=None, by=("type",), tiers=True,
          window=None, kind="current", null_recording=None, fb=None, top=40) -> Result
```

*Static* (no recording): per (target type, pre type) the effective input a volley of the pre type would deliver --
`ew.type_matrix` in mV per post cell per pre volley, the raw count and share of the target's raw input, the sign rule --
i.e. `docs/audits/cx_glno.md` section 1 as a function. *Dynamic*: per frame `I_i(t) = sum_j A[i, j] r_j(t)` (mV/s of
synaptic input per post cell; `kind='current'`), split by the presynaptic grouping `by` ⊆ {type, transmitter, tier,
sign, module, side}; for graded targets the optic matrices (`OpticLobe.W_rr / W_rp / W_rs`, with `optic_dr` and
photoreceptor activity) replace `A`. `null_recording` gives the matched control and the arm comparison. The recording
must hold `rate_hz` of every presynaptic cell with a non-zero entry onto the target (`links(...).pre_index`); the CLI's
`record` subcommand builds that Recorder from the target spec and a protocol name (`walk`, `taste`, `smell`, `object`,
`apple`, `compass`, `room`) so the GPU job is one line.

Tables `per_type`, `contributions` (body-level rows for the `top` groups), `readout_per_body`; summary: total E / I per
target type, the largest cancelling pair (the reading `optic_measures.md` 5.3 made by hand for T3: Mi1 + Tm3 lowered vs
Tm1 + Tm4 raised, through excitatory synapses of equal weight).

Validation (`VALIDATION['decompose']`): (a) the **walk.GF_max cancellation** -- decompose DNp01 over the pinned `walk`
protocol under the four arms off / default / `holdBrain` / `holdOptic` (`walk.GF_max_hz` 4.964 / 4.629 / 12.517 /
13.311, `receptor_integration.md` G.4): the per-type input table must show which presynaptic groups' rate-weighted input
rises in each single-side arm and cancels in the default; (b) the **taste dependence on 282 histamine synapses** --
static decompose of OA-AL2i3 / TmY14 / DNge138 / DNge149 / DNge150 under default vs `holdBrainHis` shows exactly the
R8p / R8_unclear / R8y / HBeyelet histamine terms going 0 -> -1 (123 entries, 282 synapses; E.4). (b) is CPU-only.

### 4.2 trace -- `flyverse/interp/trace.py`, `scripts/interp_trace.py`

```python
trace(c, source, *, stimulus, control, null=None, params=None, optic_params=None, stat="figure_z", depth_max=6,
      stage_table=None, decompose_at="first_lost", quantity=None, min_cells=3, fb=None) -> Result
```

`stimulus` / `control` / `null`: lists of Recordings (>= 3 runs each; null = control vs control). Per type: `stat`
`figure_z` (probe_figure_stages' signed figure statistic when the recording's meta carries the column map and object /
background column sets) or `dprime` (`screen.rank` between the two conditions), and always `compare(stimulus-control
per run, null per run)`. Depth = shortest path from `source` over |A| > 0 (photoreceptor -> lamina -> ... ; graded and
spiking alike) or the Nern figure stages (`probe_figure_stages.stage_table`) when `stage_table` is given. `first_lost` =
the smallest depth at which no type has verdict `result` while a type one step shallower has; trace then calls
`decompose` (static, and dynamic over `stimulus[0]`) on the types at that depth and prefixes those tables `lost_`.

Validation: the **object stage** -- Mi4 z +22.3 / +28.6, Mi1 +7.8 / +27.9, Tm3 +7.8 / +15.6 above the null; T2 (+0.5 /
+1.2), T3 (0.0 / +0.7), Tm5Y (+0.4 / +2.6), TmY21, TmY13, LC11 (-0.1 / +0.4), LC10a (-0.1 / +0.3) at it
(`object_sweep.md` 8.7, off / sign-abs) -- and the **LH odour gate**: LHPD4d1 20.6 Hz at 8 cm, 12.5 at 40 cm, 3.4
plume-free, d' 4.51, at depth 2 from the ORNs (`NOTES.md` session 8). Every per-type z is +-1.5 (section 2.4); the
target is the ordering and the verdicts, not the digits.

### 4.3 paths -- `flyverse/interp/paths.py`, `scripts/interp_paths.py`

```python
paths(c, a, b, *, params=None, receptor=None, k_max=3, top=20, min_abs_mv=0.0, recording=None, frozen=None,
      level="type", exclude=(), rates_min_hz=0.5) -> Result
```

Type-level matrix `M` (section 2.2) over the types on any path of length <= `k_max` from `a` to `b`; a path's gain is the
product of its links (mV^k per volley); the `top` paths per k with each link's sign, raw count and silence flags.
`level='cell'` keeps cells (cx_wedge's wedge matrices are this level aggregated by wedge; `cx_wedge.json` is the
reference file). Silent links are the finding, not a footnote: the tool's summary is "the strongest silent link per k".

Validation: **GLNO on rotation -> PEN** -- from `LNO1|LNO2|LNOa|SpsP|PS196_b|~^LAL` to `PEN_a|PEN_b`: GLNO -> PEN is 84
entries / 16,371 raw synapses / 19.4 % of PEN's raw input, sign 0, 16.5 mV per pair if signed; EPG -> PEN +5.05 mV per
pair (+79.9 mV per PEN per EPG volley); Delta7 -> PEN -4.70 per pair; LNO / SpsP -> PEN 0-13 synapses (`cx_glno.md`
section 1, `cx_shift.md` section 1). **The ExR4/5/6 -> EPG loop**: EPG -> ExR6 / ExR4 / ER6 / ER4m -> EPG, PEN two-step
-2,760 to -3,210 mV^2 per presynaptic wedge vs on-wedge PEN +2,465 and peak Delta7 -433 (`cx_wedge.md` section 3; the
cell level aggregated by wedge reproduces `cx_wedge_matrices.npz`).

### 4.4 lesion -- `flyverse/interp/lesion.py`, `scripts/interp_lesion.py`

```python
lesion(manifest, *, out_dir, mode="plan", replicates=3, checks="all", probes=(), seeds=None, cluster=False,
       minutes=60, baseline="baseline") -> Result
```

Manifest (JSON / YAML / dict):

```json
{"sections": "rest,taste,smell,walk,bitter,motion",
 "lesions": [
   {"id": "holdBrainHis", "kind": "hold_table", "spec": "out/receptors_holdBrainHis.csv"},
   {"id": "no_KC", "kind": "population", "spec": "~^KC"},
   {"id": "no_gaba_LN", "kind": "transmitter", "spec": "gaba", "within": "module=antennal_lobe"},
   {"id": "tier_fuzzy_held", "kind": "receptor_tier", "spec": "fuzzy"},
   {"id": "no_adapt_cx", "kind": "lif", "spec": {"adapt_by_type": {"^(EPG|PEN|PEG|Delta7)": 0}}},
   {"id": "lpi_x1", "kind": "pair_gain", "spec": ["^LPi(34|43)$", "^LPLC2$", 1.0]}],
 "probes": [{"id": "object", "cmd": "python scripts/probe_object_sweep.py --null --seed {seed} --out {out}", "read": "summary.LC11.diff_max_over_cells_mean_mv"}]}
```

Kinds and how each is applied (all as overrides; nothing in `flyverse/` edited): `population` / `types` / `module` /
`transmitter` zero the presynaptic columns of the shaped weights in-process (the `retire_measures` hook pattern on
`brain._shaped_weights`; functionally `screen.ablate` -- no output -- and additionally recorded as a `links` table of what
was removed); `receptor_tier` holds the entries the lookup decided at that tier at NT_SIGN (a temporary receptor table
written the way `build_hold_tables.py` writes one); `hold_table` = `--receptor-table`; `lif` / `optic` / `pair_gain` =
parameter overrides. `plan` writes `out_dir/batch.sh` (one `cluster_run.py` call, one job per lesion x replicate plus
`replicates` baseline jobs, `--fetch out_dir/`) and `manifest.resolved.json` (bodies per lesion, entries per hold);
`run` executes one job (`--one <lesion> --replicate k`); `analyse` builds `matrix` (check x lesion: baseline, value,
delta, replicate values / sd / n, status change), `sensitivity` (Neurome fields), `dissociations` (L1 moves c1 and not
c2, L2 moves c2 and not c1 -- bit-identity for deterministic sections, otherwise |delta| > 2 x pooled replicate sd in
both directions).

Validation: the **round-5 double dissociation** on the CPU protocol of `r5_attr_taste_cpu.py` (taste / smell, seeds 0-2,
device cpu): `holdBrain` = off and `holdOptic` = default to every digit on `taste.MN9_hz` and `smell.KC_active`;
`holdBrainGlu` = default on taste and = off on smell; `holdBrainHis` = off on taste and = default on smell; `holdKC`
1225 / 464 / 1006 and `holdDN1` 525 / 433 / 540 on `KC_active` (`receptor_integration.md` E.4). The tool must extract
(holdBrainGlu, holdBrainHis) x (smell.KC_active, taste.MN9_hz) as a dissociation automatically.

### 4.5 atlas -- `flyverse/interp/atlas.py`, `scripts/interp_atlas.py`

```python
atlas(c, populations, *, hz=150.0, ms=400.0, settle_ms=200.0, batch=64, readouts=("motor",), pattern=None, by_side=True,
      null=True, replicates=3, params=None, optic_params=None, device=None, context=None, seed=0) -> Result
```

`screen_dns.py` generalised: one `FlyBrain(batch)`, row r stimulates population r (`FlyBrain.stimulate`), null rows get
no pulse in the same batch, `replicates` independent runs give the scatter; readouts = every `MotorRates` field plus
any `pattern` through `screen.TypeRecorder` (by side). `context` = none (default), `blueberry_site` / `apple_site` /
`walking` (a callable driving the senses each frame). Per (population, readout): `compare` of the pulse-window means
against the null rows over replicates.

Validation: **PFL3-left at 80 Hz -> DNa02-right 22.6 Hz, DNa02-left 0.0** (NOTES session 8, `FlyBrain.stimulate` at the
blueberry site); **DNa02_L at 150 Hz -> left leg MNs 3.1 Hz vs right 0.1** (`benchmark.py dn`); and, with the wind
context (JO-C/E driven for wind-left vs wind-right as `screen_steering.py` does), the **DNp18 +45 / DNp33 -49 Hz L-R
flips**, identical with and without odour (`wind.*` references).

### 4.6 health -- `flyverse/interp/health.py`, `scripts/interp_health.py`

```python
health(recording, *, c=None, params=None, window=None, by="type", ew=None, thresholds=None, fb=None) -> Result
class HealthReadout(NTSource): __init__(fb, channels=("rate_hz", "v_margin_mv", "adapt_mv", "refractory", "fanin_scale")); readout(*, batch_index=0) -> NTSnapshot
```

Per type over the window: `rate_mean`, `rate_max`, `silent_frac` (max rate < 0.5 Hz), `at_threshold_frac` (mean v within
1 mV of v_th while not refractory), `refractory_load` = rate x t_ref (fraction of time refractory; 'refractory-limited'
above 0.4), `ei_balance` (positive vs negative rate-weighted input from `ew` and the recorded presynaptic rates, as
decompose's `current`), `fanin_scale` min / median / max (`ew.scale`), `adapt_load` = mean adapt / (v_th - v_rest),
`sign0_in_share` / `sign0_out_share` / frozen shares from `silent_flags`. The readout source samples the same
quantities live from `fb.brain` (`rate`, `v`, `adapt`, `refrac`, `input_scale`) into an `NTSnapshot` keyed by bodyId
(`docs/NT_READOUT.md`: owned CPU copies, NaN for unsampled units, explicit units and display ranges); attach with
`fb.nt_source = HealthReadout(fb)` -- no UI edit.

Validation: the **200 Hz refractory-limited bump** -- EPG 180-260 Hz at t_ref 2.2 ms = refractory load 0.40-0.57, PEN
40-65 Hz, Delta7 90-112 Hz (`cx_wedge.md` 7, `cx_glno.md` 5) -- and the **sign-0 / silent populations of the NT audit**:
3,312 sign-0 bodies, 2,683 of them presynaptic (2,701,289 of 124,161,873 synapses = 2.2 %), the mushroom body losing
9.8 % of input / 12.5 % of output, the octopaminergic visual centrifugal cells 17.6 % of output (`nt_audit.md`; the
tool reproduces all of it from the shipped cache -- `interp_health.py structure --by module`).

### 4.7 ledger -- `flyverse/interp/ledger.py`, `scripts/interp_ledger.py`, `flyverse/data/expected_responses.csv`

```python
ledger(results, *, table=None, tolerance=0.0, strict=False) -> Result
```

The CSV (owned by the ledger implementer): `population` (section 2.1 grammar), `stimulus` (a protocol id: `motion`,
`loom_left`, `rotation_ccw`, `sugar`, `sugar+bitter`, `wind_left`, `apple_8cm`, `object_sweep`, ...), `quantity`
(`rate_hz`, `dsi`, `flip_hz`, `dprime`, `drive_mv`, ...), `expected`, `op` (`>`, `<`, `>=`, `<=`, `sign`, `range`,
`abs>=`), `unit`, `source` (the literature citation), `model_reference` (the session / audit file that measured it in
this model), `notes`. First rows = `benchmark.REFERENCES` re-expressed per population, then the audits' per-type numbers
(object_sweep 8.7, optic_measures 5.1, cx_glno 1). The checker scores every Result whose `populations` x `stimulus`
match a row (and `benchmark.py` JSONs through their `checks`) as PASS / FAIL / MISSING with the measured value,
replicate sd / n and the citation.

Validation: the existing **T4/T5 DS** (min DSI 0.16, preferred direction per subtype), **loom** (GF 34-56 Hz), **optomotor**
(HSN d' 4.2, DNp20 3.1, HSE 2.2 flips) and **sugar / bitter** (MN9 4.6 -> 0.0 calibrated; 123.5 -> 2.1 Shiu) rows
reproduce the benchmark's statuses on its own JSON.

### 4.8 export -- `flyverse/interp/export.py`, `scripts/interp_export.py`

```python
export(result, *, out_root="out/export", run_id=None, retina=None, parquet_rows=1_000_000,
       control_ids=None, paired_control_ids=None, null_reference_ids=None,
       retina_blank=None, retina_in_loop=False, retina_geometry=None) -> Path
```

A serializer: `manifest.json` (every field of `docs/NEUROME_INTERFACE.md` section 1 from `provenance`; `tables` with row
counts, columns, units, SHA-256) and `readout_per_body.csv` / `contributions.csv` / `sensitivity.csv` (Parquet above
`parquet_rows`) from `Result.tables`, `dataset` / `release` columns prepended, bodyIds as decimal strings, model index and
type alongside. Refuses a Result whose `check()` is non-empty. Validation: a round trip of a Result through the export
and back reproduces every number; LC11 / LC10a bodies have two rows.

**Revision 2** (`manifest.schema` `flyverse.neurome.export/2`, `export_revision: 2`; Neurome's intake of the size
ladder, `docs/audits/interp_export.md` section 14):

* **two controls, named apart.** `paired_control_ids` names the record **and arm** each `control_value` came from
  (arm b of the same recording, suffix `#arm_b`); `null_reference_ids` names the independent blank/blank runs behind
  `null_mean` / `z_vs_null` / the verdict. `control_ids` stays for one revision as a **deprecated alias of
  `null_reference_ids`**, written equal to it and documented in `manifest.conventions.control_ids`; a caller that
  passes only `control_ids` is read as passing `null_reference_ids`. `verify()` checks all three rules.
* **`manifest.statistic_definitions`**: one sentence per `quantity` / `statistic` the tables use, so a drive figure
  ("max over cells within each run of the time-mean object-minus-blank optic drive, against the same statistic in
  blank/blank runs") cannot be read as an absolute membrane voltage.
* **retina**: `manifest.retina.mode` is `geometry_replay` | `in_loop_capture` | `unknown`, beside `pinned_pose` and a
  statement of what was replayed -- a replay is never labelled a capture, and `verify()` refuses radiance without a
  mode. `retina_blank` writes the matched blank arm as `retina_radiance_blank` in the identical schema, and
  `retina_geometry` (`ball_radius_m`, `ahead_m`, `eye_above_table_m`) writes `retina_object_track`: the ball's
  azimuth, centre elevation and angular diameter from the eye per frame, plus the columns it dims against the blank.
* **tie-aware rank test**: `mann_whitney` / `compare_tie_aware` use the exact U only when the pooled arms have no
  ties and the tie-corrected asymptotic test otherwise; every per-type row carries `p_method` and `n_tied_values`.
  `holm` / `family_labels` / `add_family_columns` add an optional `family` / `p_holm` pair within a family the
  **caller predeclares** (`FAMILY_SPECS`, a JSON spec, or nothing -- the default); no verdict rests on `p_holm`.
* **`scripts/interp_export.py ladder`** re-exports a recorded size ladder (`--runs-csv <summary>/runs.csv` or
  `--dir <recordings>`) into one NEW run directory per size plus a summary, on the CPU, from the recordings alone.

---

## 5. Where work runs, and the process rules

**GPU work happens only through a tool's CLI on the cluster; the analysis is CPU.** Each wrapper has `record` (or `run`
for atlas / lesion jobs) and `analyse`; `record` writes Recordings + provenance, `analyse` reads them and writes the
Result. The desktop runs nothing on a GPU. Submission is always

```
python scripts/cluster_run.py --name <short> --minutes <est> \
  "python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_trace.py record --protocol object --arm stim --seed 0 --out out/tr/stim_r0 > out/tr/stim_r0.txt; cat out/tr/stim_r0.txt" \
  "... --arm stim --seed 1 ..." "... --arm ctrl ..." "... --arm null ..." --fetch out/tr/
```

-- several commands in ONE call (they run concurrently; the replicate unit is that job), `--fetch` a NAMED subdirectory
(the bare `--fetch out/` form fails when `out/` holds a scratch `cache_*`), the console output shipped as
`out/<name>_cluster.log`, the `'<n> job(s), 0 failed'` line read before any status is written, a log saying `device cpu`
resubmitted, an ssh / HTTP failure retried once after 60 s, a failed fetch replaced by `scp` -- never a finished batch
reported as pending. Room sweeps use `BatchSim` (`--cuda-sparse torch`); pinned single-fly probes stay scalar.
CPU-only work (structural: paths, static decompose, health on a saved recording, ledger, export, the lesion analysis,
the CPU taste / smell protocol) runs locally with `PYTHONIOENCODING=utf-8 python ...`; headless pygame needs
`SDL_VIDEODRIVER=dummy`.

**Numbers.** Raw data, every number from a named file or run; scatter over >= 3 replicates quoted before a difference is
called a result (`compare` enforces it); `walk.power_max`'s 50 Hz bound is never a verdict and nothing is tuned to it;
per-type figure z carries +-1.5. Every generator of every quoted number lives in `scripts/` or `flyverse/interp/`.

---

## 6. Validation targets (what each tool must reproduce before its numbers are trusted)

`common.VALIDATION[tool]` carries these with their source files; the Result's `validation` block is prefilled from it.

| tool | known localization | numbers to reproduce | file |
|---|---|---|---|
| decompose | walk.GF_max cancellation; taste's 282 histamine synapses | GF_max 4.964 off / 4.629 default / 12.517 holdBrain / 13.311 holdOptic; 123 entries onto OA-AL2i3 62, TmY14 25, DNge138 5, ... from R8p / R8_unclear / R8y / HBeyelet | receptor_integration.md G.4, E.4 |
| trace | the object stage; the LH odour gate | Mi4 z +22.3 / +28.6, Mi1 +7.8 / +27.9, Tm3 +7.8 / +15.6; T2 / T3 / Tm5Y / TmY21 / LC11 / LC10a |z| <= 2.6; LHPD4d1 20.6 / 12.5 / 3.4 Hz, d' 4.51 | object_sweep.md 8.7; NOTES session 8 |
| paths | GLNO silent on rotation -> PEN; the ExR -> EPG loop | GLNO -> PEN 84 entries / 16,371 syn / 19.4 % / sign 0 / 16.5 mV per pair if signed; EPG -> PEN +5.05 per pair, +79.9 per PEN volley; Delta7 -> PEN -4.70; ExR loop -2,760..-3,210 mV^2 vs PEN +2,465, Delta7 -433 | cx_glno.md 1; cx_wedge.md 2-3 |
| lesion | the hold double dissociation | holdBrain = off, holdOptic = default (bit-exact, 3 seeds); holdBrainGlu / holdBrainHis mirror on taste vs smell; holdKC 1225 / 464 / 1006, holdDN1 525 / 433 / 540 | receptor_integration.md E.4 |
| atlas | wind DN flips; PFL3 -> DNa02; DNa02 -> legs | DNp18 +45, DNp33 -49 Hz; PFL3_L 80 Hz -> DNa02_R 22.6 / DNa02_L 0.0; DNa02_L 150 Hz -> leg L 3.1 / R 0.1 | NOTES session 8; benchmark wind.*, dn.* |
| health | the refractory-limited bump; the sign-0 populations | EPG 180-260 Hz = load 0.40-0.57 at t_ref 2.2; PEN 40-65, Delta7 90-112; 3,312 sign-0 bodies (2,683 presynaptic; `common.VALIDATION` carries 3,312 since this revision, with a note that 3,407 was the pre-`TYPE_NT_OVERRIDE` cache's count -- interp_health.md 2; `docs/NEUROME_INTERFACE.md` line 73 still says 3,407), MB 9.8 % input silenced | cx_wedge.md 7, cx_glno.md 5; nt_audit.md |
| ledger | the suite's own references | motion.min_dsi 0.16; loom GF 34-56; HSN 4.2 / DNp20 3.1 / HSE 2.2; MN9 4.6 -> 0.0, Shiu 123.5 -> 2.1 | benchmark.py REFERENCES |
| export | a round trip; LC11 / LC10a two rows per body | every number back; key (dataset, release, bodyId) | NEUROME_INTERFACE.md 1 |

The dynamics-round-1 inputs the tools take as given (not re-derived here): the ring bump at gE 2/15, 2.25/25, 2.5/25
persists 38 s in 112/112 flies at 219-260 Hz, does not track heading (|circ corr| <= 0.11), does not steer (yaw first
harmonic 0.3-1.2 deg/s; PFL3 L-R 2-3 Hz fixed gradient, DNa02 L-R 0.08-0.24 Hz), is pinned to 5-7 attractor sites; PEN
L-R +-20-35 Hz is an output asymmetry; visual rotation at 90 deg/s moves the bump 0.00 +- 0.01 wedges/s (the efferent
PS196_b / LAL -> GLNO route untested); GLNO=gaba gives +0.30 vs +0.04 wedge deflection (6/6, p 0.031). The object
figure is carried by lamina / medulla / T4c,d / LPLC2 and lost at the LC10 / LC11 inputs by sign-correct ON / OFF
cancellation through excitatory convergence and l1 pooling (LC11 0.046 mV, LC10a 0.080 vs LPLC2 0.54); none of 16
optic ablations makes them carry; the T5 pair gain x5 is 78 % ACh edges. The optic half of the receptor signs carries
75-100 % of the room take-off cost, the Brain half none, with 0 changed entries on DNp01 / LC4 / LPLC2 / motor types.
Meals == contacts; the mechanism is the cx program's terminal approach. (`compass_room.md` and `cx_shift.md` on disk
still read PENDING at the time of this design; the numbers above are the round's reported results and the first thing
each implementer checks is that the fetched files agree with them.)

---

## 7. Worked CLI examples

```bash
# decompose: DNp01's input during the pinned walk, four arms, three runs each -- one cluster batch
python scripts/cluster_run.py --name dec-gf --minutes 40 \
  "python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_decompose.py record --target DNp01 --protocol walk --seed 0 --out out/dec/default_r0 > out/dec/default_r0.txt; cat out/dec/default_r0.txt" \
  "... --receptor-table out/receptors_holdBrain.csv --out out/dec/holdBrain_r0 ..." "..." --fetch out/dec/
PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py analyse --target DNp01 --recordings "default=out/dec/default_r*.npz" --null-runs "off=out/dec/off_r*.npz" --by type,transmitter,tier --window 0.5,1.5 --json out/interp/decompose/gf_walk.json
# static, CPU (runs as written on the shipped cache): the taste carrier
PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py analyse --target "OA-AL2i3|TmY14|DNge138|DNge149|DNge150" --static --receptor-table out/receptors_holdBrainHis.csv --json out/interp/decompose/taste_static.json

# trace: the moving ball from the photoreceptors, 3 stimulus + 3 control + 3 null runs in ONE batch, then CPU
python scripts/cluster_run.py --name tr-ball --minutes 30 $(for a in stim ctrl null; do for s in 0 1 2; do echo "\"python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_trace.py record --protocol object --arm $a --seed $s --out out/tr/${a}_r$s > out/tr/${a}_r$s.txt; cat out/tr/${a}_r$s.txt\""; done; done) --fetch out/tr/
PYTHONIOENCODING=utf-8 python scripts/interp_trace.py analyse --source "type:R1-R6|R7y|R7p|R7d|R7_unclear|R8y|R8p|R8_unclear" --stimulus "out/tr/stim_r*.npz" --control "out/tr/ctrl_r*.npz" --null-runs "out/tr/null_r*.npz" --stat best_cell --stage-table family --decompose-at T3,T2,Tm5Y,TmY21 --json out/interp/trace/ball.json
# (--stat figure_z is the signed retinotopic figure, which the sweeping ball does not survive: it reads 'not reproduced'
#  on the object target; best_cell is the object sweep's statistic and the validation's -- interp_trace.md 2.1)

# paths: CPU, the rotation inputs of PEN and the ring loop
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a "LNO1|LNO2|LNOa|SpsP|PS196_b|~^LAL" --b "PEN_a|PEN_b" --k 3 --json out/interp/paths/rot_pen.json
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a EPG --b EPG --k 2 --level cell --wedge --json out/interp/paths/epg_loop.json

# lesion: plan -> one batch -> analyse (`--manifest` takes a builtin name -- 'holds', 'holds_cpu',
# `lesion.BUILTIN_MANIFESTS` -- a JSON / YAML path or a JSON string)
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py plan --manifest holds --out out/les_holds --replicates 4
bash out/les_holds/batch.sh            # the single cluster_run.py call plan wrote; ships out/les_holds_cluster.log
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py analyse --out out/les_holds --json out/interp/lesion/holds.json
# the CPU validation (taste / smell, no optic lobe): --device cpu runs every job locally in-process
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py run --manifest holds_cpu --sections taste,smell --device cpu --out out/les_cpu --replicates 3

# atlas: 64 populations per batch, three runs, on the cluster; then CPU
python scripts/cluster_run.py --name atlas-dn --minutes 45 "python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_atlas.py run --populations superclass=descending_neuron --by-side --hz 150 --ms 400 --seed 0 --out out/atlas/dn_r0 > out/atlas/dn_r0.txt; cat out/atlas/dn_r0.txt" "... --seed 1 ..." "... --seed 2 ..." --fetch out/atlas/
PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/dn_r*" --json out/interp/atlas/dn.json

# health: from a compass-room recording; and live in the observatory
PYTHONIOENCODING=utf-8 python scripts/interp_health.py analyse --recordings "out/health/compass_r*" --window 5.5,8 --groups meta --json out/interp/health/compass.json
PYTHONIOENCODING=utf-8 python scripts/interp_health.py analyse --recordings "out/health/walk_default_r*" --null-runs "out/health/walk_off_r*" --window 0.5,1.5 --json out/interp/health/walk.json
# CPU, no rollout: the sign-0 / frozen shares of the shipped cache (reproduces the NT audit: 3,312 sign-0 bodies)
PYTHONIOENCODING=utf-8 python scripts/interp_health.py structure --by module --json out/interp/health/structure_module.json
python -c "from flyverse.interp import HealthReadout; ...; fb.nt_source = HealthReadout(fb)"     # then N in the map

# ledger and export (both CPU; the ledger scores whatever it is pointed at, the export serializes any Result)
PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py --results "out/interp/*/*.json" out/benchmark_suite.json --null-runs "out/r3obj/null_*.json" --json out/interp/ledger/all.json
PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py validate --json out/interp/ledger/validate.json      # VALIDATION['ledger']
PYTHONIOENCODING=utf-8 python scripts/interp_export.py analyse --result out/interp/paths/rot_pen.json --out out/export
# `analyse` / `run` / `static-decompose` take `--paired LC11,LC10a` (default): the types whose every body must carry
# BOTH readout quantities (upstream_drive_mV and output_Hz) in readout_per_body -- the LC pooling rule of
# docs/NEUROME_INTERFACE.md section 1, checked by the export's own verify pass. `--paired ""` turns the check off for
# a Result that has no per-body readouts (a structural one).
PYTHONIOENCODING=utf-8 python scripts/interp_export.py static-decompose --target "LC11|LC10a" --json out/interp/export/lc11_lc10a_static.json --out out/export
PYTHONIOENCODING=utf-8 python scripts/interp_export.py verify --run-dir out/export/<run_id> --paired LC11,LC10a
```

Every CPU line above runs as written wherever its inputs are in the tree -- the cache (paths, static decompose,
`health structure`, `export static-decompose`), `out/dec`, `out/health`, `out/atlas`, `out/les_cpu`, `out/interp`
(ledger, export) -- and the `analyse` lines that read a batch need that batch fetched first: the trace pair above
writes and reads `out/tr/`, while the round's own odour batch is `out/trv/`, which runs today:

```bash
PYTHONIOENCODING=utf-8 python scripts/interp_trace.py analyse --source "class=olfactory" --stimulus "out/trv/od_stim_r*" \
  --control "out/trv/od_ctrl_r*" --null-runs "out/trv/od_null_r*" --stat mean --min-cells 2 --json out/interp/trace/odour_mean.json
```

`--help` works on every wrapper and every subcommand.

---

## 8. Implementer checklist (one per tool; the shared files are not yours)

Files you own: `flyverse/interp/<tool>.py`, `scripts/interp_<tool>.py`, your test class in `tests/test_interp.py` (append;
the `graph()` there has a photoreceptor, a graded unit, a VPN, the GF, DNa02 L/R, a sign-0 GLNO and a PEN), and for the
ledger `flyverse/data/expected_responses.csv`. Do not edit `common.py` or `__init__.py`; if the contract needs a change,
say so in your report and keep the stub's parameters.

1. The function keeps every parameter of `stubs.<tool>` (the test `StubTests` enforces it) and returns `common.Result`.
2. Every stochastic measurement has `--null-runs` and `--replicates` (default 3; >= 4 per arm to reach a `result`,
   5 for a small effect) and reports `compare`'s dict -- its `verdict`, not one the tool invents.
3. The CLI has `record` / `run` (GPU, cluster) and `analyse` (CPU) subcommands where the tool has a GPU part; the JSON
   carries the provenance block with the REALISED device; `Result.check()` is empty.
4. The Neurome tables use `common.EXPORT_TABLES` columns; bodyIds are decimal strings; LC11 / LC10a are never pooled.
5. The validation target of section 6 is run first, its numbers written into `validation.measured` with the file that
   produced them, and `status` set honestly (`not reproduced` is a finding).
6. The test class runs on the CPU on `graph()` or a `Connectome.subset` in seconds.
7. Every number in your report names its file; every generator is in `scripts/` or `flyverse/interp/`.

---

## 9. Open questions for the round

* `compass_room.md` / `cx_shift.md` are PENDING on disk while the brief reports their results; the health / paths
  implementers should fetch the run directories first and confirm the 112/112 persistence and the 0.00 wedges/s rotation
  numbers from the files before quoting them.
* `receptor_tier` lesions need a hold-table writer per tier; `build_hold_tables.py --groups` covers Brain / Optic / KC /
  DN1 / transmitter groups, not tiers -- the lesion implementer adds a `--tier` group there only if that file is theirs,
  else writes the temporary table inside `lesion.py`.
* Dynamic decompose of graded targets uses the OpticLobe matrices; the exact per-frame identity holds for the LIF (linear
  synaptic input) but the rate units' recurrent term makes the optic decomposition an input decomposition, not an
  output attribution -- state it in the table's `kind`.
* The atlas' wind context reproduces `screen_steering.py`'s sites; whether `stimulate` of JO-C/E alone (no room) gives
  the same DNp18 / DNp33 flips is itself a finding worth one line. (Answered by the build round: yes to within the
  room's wind meander -- `interp_atlas.md` 3.1 and its skeptic's meander job, +47.5 vs +51.3 fixed vs the room's +45.3.)

---

## 10. How to use it when a behaviour fails

A behaviour "fails" when a ledger row reads `FAIL` / `KNOWN GAP` (`flyverse/data/expected_responses.csv`, scored by
`scripts/interp_ledger.py`) or a `scripts/benchmark.py` check does. The toolkit's output is a **diagnosis** -- which
population, synapse class or link, and which kind of fact -- never a fix. Under the project rule the only things a
diagnosis licenses are: a measurement, a mechanism the connectome data imply (a missing afferent, a transmitter
call above the 0.5 confidence the project accepts), or a documented swappable module. A gain, a bias, a threshold, a
readout change or a sign the data cannot see is hand-crafting, and the diagnosis must say so by name (the
`deficit_*.md` audits end with exactly that split; copy it).

### 10.1 The procedure (seven steps; the first four are always run, in this order)

1. **Name the readout and the expectation (CPU, minutes).** The ledger row with its citation and the model reference
   (`model_reference`), or write the row first -- `label`, `population` in the section 2.1 grammar, `op`, `bound`,
   `source`. Without a row there is nothing to fail. Read the row's `requires` (a bump rate without a bump is
   `NOT_APPLICABLE`, not a pass). Never use `walk.power_max` as the failing check.
2. **Structure first (CPU, minutes).** `interp_paths.py --a <source> --b <readout population> --k 3` for every
   sensory / central source that could carry the signal, and `interp_decompose.py analyse --static --target <readout>`;
   `interp_health.py structure` when the readout's module has sign-0 or frozen input. Read three things: the
   strongest **silent** link per k (`sign0`, `frozen`, `pruned`), the readout's E / I per volley and its largest
   cancelling pair, and whether any route into the readout is signed at every link. This is where GLNO -> PEN
   (sign 0, 19.4 % of PEN's input) and LLPC1 / PFL3 / AOTU015 -> DNa02 were found before any GPU job ran.
3. **One cluster batch (GPU), replicates as runs.** Record stimulus / matched control / control-again (**the null**)
   under the shipped model, **`min(n_a, n_b) >= 4` runs, 5 when the effect is small**. The rule is about the SMALLER
   arm, not about the symmetric exact-U floor: 3 v 3 floors at p 0.10, but 3 v 5 floors at 0.036 and 3 v 6 at 0.024,
   both under alpha, so a big null arm used to buy a three-run stimulus arm a `result`. `common.compare` applies both
   (`CALL_REPLICATES`, and `p_floor > alpha`) and returns `underpowered` below four runs in either arm whatever the
   floor says; `MIN_REPLICATES` stays 3 because three runs still buy the scatter. Add the
   deterministic arm when one exists (`--optic gain_fb=0` for optic-lobe questions: its null is exactly 0 and it is
   read as magnitudes, not verdicts). One `cluster_run.py` call, `mkdir -p` in every job line, `--fetch` a NAMED
   subdirectory, one client per directory, and a job line that preserves python's exit code (`&& tail` /
   `st=$?; tail; exit $st`, 10.4 item 4); read `'<n> job(s), 0 failed'`, then count the expected artefacts and check
   every run block's `device` and arm spec before analysing; `interp_apply_rotation.py verify-batch <dir>` (or the
   tool's console-vs-meta check) before analysis.
   Two same-code batches with the same seeds are not the same draws (the GPU rollout is not seed-reproducible):
   the replicate unit is the job.
4. **Trace, then decompose at the lost stage (CPU).** `interp_trace.py analyse --source <sensory spec> --stimulus
   ... --control ... --null-runs ... --stat best_cell` (or `mean` for firing rates; `figure_z` only with the column
   map and only for a static figure), `--min-cells 2` when the chain has 2-cell types. Read the carriers per depth
   and the **named chain**, not the contract's `first_lost_depth` (it is 5 / None for any stimulus a wide pathway
   carries). `--decompose-at <the first non-carrier types on the chain>`: the `lost_inputs` table gives every input's
   share, sign, tier, its own verdict and signed figure, and the cancellation fraction of the carriers; the
   `lost_static_*` / `lost_dynamic_*` tables are decompose's. Feed the recording back to `paths --recording` so every
   link on the structural routes carries a `never_firing` flag from the same rollout.
5. **Classify the loss with counterfactual arms (one more batch), and only the arms that decide a kind:**

   | kind of fact | the arm that decides it | tool | reads as |
   |---|---|---|---|
   | **wiring** (convergence / pooling) | hold one presynaptic class onto the lost type at 0 (`edges` kind: `interp_apply_object.py record --lesion t3_off_held`; to be promoted into `lesion.py`) | lesion / apply_object | the type reaches `result` with either class held and its run scatter collapses to the carriers' |
   | **sign / receptor** | a hold table or receptor tier (`--receptor-table`, `kind: receptor_tier`), `decompose contrast --arms a,b` for the entries that differ; a counterfactual flip is a question, never a proposal | lesion / decompose | the type reaches `result` only under the flip **and** the data (tier, confidence) allow it |
   | **rate-model artefact** | `OpticParams` overrides: `norm`, `out_norm`, `baseline_by_type`, pair gains | lesion (`kind: optic`) | the stage moves with the override and the carriers do not |
   | **dynamics** | the deterministic arm (`gain_fb=0`) vs the shipped feedback, per cell: is the deterministic figure present at the same cells under the feedback? (`interp_apply_object.py perrun`, table `projection`) | trace / apply_object | present for a direct response, absent for a residual of opposing sums |
   | **missing input** | `never_firing` at the source of every wired route (`paths --recording`), plus the atlas showing the route works when driven (`interp_atlas.py run --populations <source>`) | paths + atlas | the route is signed and functional at 150 Hz and its source is at 0 Hz in the room |

   The three deficits of this round each needed two of the five (object: wiring + dynamics; rotation: missing input
   + a sign-0 link; turning: missing input on three routes + an operating point). State the composition, not one
   label.
6. **Attribute with the lesion matrix when a suite check is the failure.** `interp_lesion.py plan --manifest <holds
   or your own> --replicates 3` for bit-identical checks (taste, smell, bitter, `walk.GF_max` under the pinned walk,
   `rest`) -- bit-identity is a CPU / `gain_fb=0` property (10.4 item 2) and those three draws are ONE effective
   replicate, read as magnitudes (`undetermined`), never as a called difference -- **>= 4-5** for any check with run scatter (`loom.GF_peak_hz`, `rotate.DNp20_flip_hz`, the walk power pair,
   anything from the room); read `dissociations` only for rows called by bit-identity or by seed pairing, and quote a
   scatter-rule dissociation with its draw count (the same `holds` manifest gave 1 / 8 / 17 dissociations over three
   batches of 4 / 4 / 3 draws; only the (holdDN1, holdKC) x (Shiu sugar, Shiu bitter) pair survived all three).
7. **Write it down and export it.** `docs/audits/deficit_<name>.md` in the shape of the three shipped ones: the
   answer first (where; which kind; the data-driven route if one exists, the hand-crafting explicitly named and not
   done), every number from a named file or run with its scatter over >= 3 runs, the batch line, the defects found in
   the tools. `interp_export.py analyse --result <the Result>` for every JSON a partner may read (the export refuses a
   Result whose `check()` is non-empty; `verify` must print `problems: none`). Add a ledger row for the new
   localization so the next round scores it. Two separate obligations about `flyverse/`, and they are not the same
   one:

   * **(a) Authorship.** Nothing in `flyverse/` outside `interp/` is AUTHORED by the task -- no edit of `brain.py`,
     `optic.py`, `body.py`, `programs.py`, `room_demo.py`, `room_ui.py` lands from this work, and a counterfactual is
     a `LIFParams` / `OpticParams` override, a `--receptor-table` or an in-process mask (section 10's opening).
   * **(b) Dependency.** Every `flyverse/` file the task DEPENDS on that is **not at `origin/main`** is listed in the
     audit header with its diffstat (`git diff --stat origin/main -- <file>`) and the task that owns it; the run's
     `provenance.source_fingerprint` is quoted as the code identity of the numbers (10.4 item 6); and a cross-task
     dependency is **committed before the batch is submitted** -- `cluster_run.py` ships the working tree, so an
     uncommitted file from another task is in the run and in nobody's history, and the audit's numbers then have no
     code they can be reproduced from.

### 10.2 Reading rules (the ones this round had to learn)

* A `result` needs |z| >= 3 **and** p <= 0.05 **and** `min(n_a, n_b) >= 4` (5 for a small effect). The run count is a
  rule about the smaller arm, not about the exact-U floor: the floor is symmetric (3 v 3 -> 0.10, 4 v 4 -> 0.029,
  5 v 5 -> 0.0079), so pairing three runs against five (0.036) or against a pile of null rows (3 v 6 -> 0.024) clears
  it while the claim still rests on three draws. `common.compare` reports the floor as `p_floor`, the applied run
  floor as `min_n` and the smaller arm as `n_min`, and returns `underpowered` when either bites: a 3-run comparison
  is z and the values, never a result, whatever the other arm's size.
  A deterministic null (SD 0) makes z NaN or astronomical; `compare` returns
  `undetermined` (`null_sd_zero` true) and the reading is the magnitude `diff` with `p`, plus whatever effect size
  the tool declares (atlas `z_floor`). Neither verdict is a tool's own invention any more -- both come from
  `compare`, in every JSON.
* A `result` whose stimulus value did not move from the baseline arm's is the null's scatter (LPLC2 / L5 under the
  T3 arms; Tm5Y at z +0.9 / +2.2 / +2.6 / +3.65 over four batches of the same protocol): read `delta_vs_base` and the
  per-run draws, not the per-arm verdict. Per-body `verdict` columns (`readout_per_body.z_vs_null`) are an
  uncorrected screen (5-11 % of bodies at chance), never a finding.
* Two same-code batches disagree on sub-Hz cells and on single-run ring jumps; a claim that rests on one batch is
  quoted as such. Three batches of the same manifest are the replication of an attribution, not one.
* The atlas measures excitatory movers only (a silent readout's null is 0.000 Hz), at one uncontrolled dose
  (150 Hz x 400 ms on 1-6,365 cells), in darkness (no optic drive); `self_drive` rows are trivial; a photoreceptor
  or optic-lobe population reads 0.000 because its output is pruned with the rate units.
* The static per-volley tables and the room's rate-weighted input are different quantities (DNa02: +466 mV per
  volley wired, -326 / -392 mV/s arriving); convert mV/s to a steady conductance (x tau_syn = 0.005) before calling
  a dose "N x too small".
* `first_lost_depth` is informative only for a single-pathway stimulus; `lost_top_by_share` and the named chain are
  what to read. `figure_z` (signed) is not survived by a moving object; `best_cell` is the object sweep's statistic.
* Every cluster JSON says `flyverse_commit.commit: unknown` (the run copy has no `.git`); the identity of the code is
  `provenance.compiled_connectome.md5` + `files.effective_weights_md5` + `provenance.source_fingerprint` (the loaded
  files, matched by content -- written by `provenance()` itself since this revision, no longer only by the export);
  quote those, not the commit.
* **A max-over-cells statistic never travels alone.** A statistic defined as an extremum over a population
  (`diff_signed_best_cell`, `diff_max_over_cells_mean_mv`, `drive_max`) has a null that is itself an extremum over
  thousands of cells, and an arm can multiply it 10x while leaving the typical cell where it was: in object round 2,
  per-stream rectification raised T3's `diff_signed_best_cell` 6-10x over base at every small rung while the per-body
  RF-windowed median |drive| read `null` against base in all 48 arm x type x rung rows. **Report the per-body median
  (or mean) of the same quantity in the same table, and say in the sentence that the headline is a within-run
  population maximum.** Neurome's round-1 correction is a standing rule, not a one-off.
* **A family member whose statistic is constant by construction is not a test.** If a member's value is known in
  advance to be identical in every run of both arms (an all-zero spike median, a saturated count), it contributes no
  information, returns p = 1.0 through ties, and still inflates the Holm denominator -- in object round 2 six of each
  twelve-member family were all-zero spike medians, which is the whole reason the arm size had to rise from 5 to 6
  runs. **Declare such quantities as reported magnitudes outside the family**, and size the family (and therefore the
  arm count) from the members that can move.
* **Read a zero-SD null's p as a structural constant, never as evidence.** `compare` returns `undetermined` and the
  exact-U p collapses to `p_floor` whenever an arm has SD 0, so a deterministic arm (`gain_fb = 0`) produces
  `p = 0.00216` at 6 v 6 and `p_holm = 0.026` on every member regardless of sign or size: in object round 2's
  `primary:fb0` families all six drive rows read exactly that while their diffs alternate -,+,+,-,+,-. Quote the
  magnitude with its scatter; never tabulate such a row in the same column as a Holm-surviving call.
* **A stimulus label is a claim about the stimulus, and must be checked against its generator.** A periodic
  square-wave flash contains both an ON and an OFF transition in every run, so a whole-window time mean of it
  separates bright from dark, not ON from OFF -- object round 2's `flashon` / `flashoff` battery was labelled
  "isolated ON / OFF transitions" and measured neither. Before a specificity rule is written, read the generator and
  the recorded `params`.

### 10.4 Process rules the round learned (rules, not advice)

1. **Replicates are jobs, and four is the floor -- in the SMALLER arm.** The replicate unit is the cluster job, not
   the seed and not the batch row. The rule is `min(n_a, n_b) >= 4`, 5 when the effect is small, and it is not the
   exact-U floor: the floor is symmetric (`common.p_floor`: 0.10 at 3 v 3, 0.036 at 3 v 5, 0.024 at 3 v 6), so a
   three-run arm against a large one clears it and still rests on three draws. `common.compare` enforces the run
   count itself (`CALL_REPLICATES`), so three runs read `underpowered` whatever the other arm is. Three runs still
   buy the scatter and are right for a bit-identical check (taste, smell, the pinned walk) -- read as magnitudes.
2. **The GPU rollout is not seed-reproducible, and 'bit-identical' is a CPU claim.** Two same-code batches at the
   same seeds are different draws (same-seed leg L-R +0.190 vs +0.208; ring histories differ). Never quote a seed as
   if it pinned a number, never compare a new batch to an old one row by row, and quote a one-batch claim as a
   one-batch claim. **`bit-identical` is a property of a CPU arm, or of an arm that is deterministic by construction
   (`gain_fb=0`), and never of the shipped GPU path**: at a fixed seed on one box it diverges run to run
   (`out/proprio_bitid/compare.txt`: identical to frame 500 of 6,000, then not), so a plan that says 'three runs,
   bit-identical' is a plan for the CPU. Record the **realised device per job** (`provenance.execution.device`,
   `device_name`; 10.4 item 5) and say which arm was which. And N bit-identical draws are **one effective
   replicate**, not N: `compare` sees SD 0, returns `null_sd_zero` / `undetermined`, and the reading is the magnitude
   `diff` with the tool's declared effect size -- never a verdict, and never an n of N in a power statement.
3. **Never point two clients at one `--fetch` directory.** Two `cluster_run.py` calls fetching into the same
   directory interleave and silently mix runs of different arms. One batch, one NAMED subdirectory, `mkdir -p` in
   every job line.
4. **A job line must preserve the python exit code, and `'0 failed'` is never sufficient.** A line that ends
   `... > file 2>&1; tail -4 file` exits with **tail's** status: the python can die mid-write and the job still
   reports `completed exit 0`. Write `... > file 2>&1 && tail -4 file` (the tail runs only on success and the failure
   survives) or `...; st=$?; tail -4 file; exit $st`. `cluster_run.py` warns on the `;` form (stderr and the console
   log) and submits the command unchanged.

   `'<n> job(s), 0 failed'` is **necessary and never sufficient**. Before any analysis: count the expected artefacts
   (`ls out/<name>/ | wc -l` against arms x runs) and open **each** run's block in the console log -- its
   `device cuda` line (a job that fell back to the CPU analyses fine and means nothing), the device it really used,
   and the arm spec it echoed (a job that ran the wrong arm is the failure mode `'0 failed'` cannot see). Then run
   the batch check (`interp_apply_rotation.py verify-batch <dir>`, or the tool's own console-vs-meta check).
5. **A CPU smoke on Windows needs `CUDA_VISIBLE_DEVICES=-1`, set before torch is imported.** The empty string
   (`""`) is what `--device cpu` used to export, and Windows *unsets* an empty environment variable, so the "CPU"
   smoke runs on the desktop's GPU and writes `device_requested cpu` / `device cuda` into the provenance. `-1` is
   the value that works on both platforms.
6. **Identify the code by content, not by the commit.** Quote the compiled-W md5
   (`provenance.compiled_connectome.md5`, `ef23cc27...` for the shipped cache), the effective-weights md5
   (`files.effective_weights_md5`, `ed1df661...`) and `provenance.source_fingerprint` when
   `flyverse_commit.commit` is `unknown`; `export.match_sources(fp)` turns that fingerprint back into a local
   commit when every file matches.
7. **The analysis is regenerated on the complete fetch, before any number is written.** No table is quoted from a
   staged or partial analysis directory -- re-run the analysis on the fetched batch as a whole and read the numbers
   off that run. The audit **pastes the analysis console's run-count line verbatim, per table, with the directory it
   came from** (`n_runs 5 (out/od/stim_r*.npz)`), so a reader can see which draws each table rests on. An audit that
   still contains an unfilled `ALL_CAPS_NOTE` / `_TABLE` placeholder does not ship: grep for `[A-Z_]{6,}` in step 7,
   before the commit.
8. **A new arm family is opt-in, and a default invocation stays cheap.** A new arm / group family (a hold-table
   group, a lesion set, a context) goes behind an **explicit flag or an explicit name**, never into a tool's bare
   default: `scripts/build_hold_tables.py`'s three ALONE groups sat in the `--groups` default and made a bare
   table build load the connectome cache. **A tool's default invocation must not acquire a new heavy dependency**
   (the cache, a GPU, a network fetch) -- if the new work needs one, the work is named on the command line.
   And a provenance defect is fixed in `flyverse/interp/common.py`, where every tool inherits the fix; a per-tool
   workaround for a shared-layer bug is the thing section 11 exists to retire.
9. **An experimental factor is never the unit of scheduling.** `cluster_run.py` places one job per command on the
   least-loaded target, so "one job per arm" makes arm collinear with box -- and the fleet mixes GPU models. In object
   round 2 `base` ran on a B200 while `rectify` and `suppress` ran on H200s, and `base` itself sat on a different GPU
   model for the sphere than for the specificity and bench sections, so the primary arm-vs-base question compared
   arm-on-box-X with base-on-box-Y. **Block the family: put every arm of a comparison family on one box (see
   `--arm-block` below), or replicate the reference arm on every box that hosted a treatment arm, in the same
   submission.** Record the realised `device_name` per run AND add a verifier problem when an arm and its reference
   differ; a `result` on a cross-device comparison is reported as device-crossed until a same-device pair confirms it.

   `python scripts/cluster_run.py --arm-block KEY[,KEY...]` keeps every job whose command carries the same
   `<KEY>_<value>` on ONE target: the KEY names the **block** that must stay together (the comparison family), not the
   factor, so every arm inside a block is compared on one box. Blocks are dealt round-robin over the available
   targets, largest first (`--balance-blocks`, the default under `--arm-block`; `--no-balance-blocks` resolves each
   block with the least-loaded rule), each block's target is charged with the block's whole size at assignment, and a
   block whose target drops out falls back to per-job placement with a printed line. `--arm-block-map file.json` (job
   index -> block name) blocks job lines that carry no usable key. Every job line then records its block and its box
   (`job 7f3a queued  @rent-a  objr2c-ab12-3 [block sphere]: python ...`), and a batch with more jobs than targets and
   no `--arm-block` is warned about at submit time. `object_round2_compare.py verify` raises one problem per arm whose
   realised `device_name` set differs from `base`'s, and every arm-vs-base row of the Result carries
   `same_device_as_reference` (false also when either device is unknown), which the console prints as
   `DEVICE-CROSSED`.
10. **The predeclaration is the stamped JSON, never the prose audit.** The audit doc is written around the stamp and
    is routinely edited after submission (object round 2: `predeclared.json` 15:04:04Z, submit 15:06:37Z, audit doc
    15:07:27Z). Cite the JSON path and its `stamped_utc`; never cite an audit section as the predeclaration.
11. **The Result records the analysis code that produced it, and a post-hoc fix obliges a re-emit.** "Stamped before
    submission" covers the reading rules, not the reducer. In object round 2 the `spearman_perm` NaN floor was fixed
    in the script 1h41m AFTER the Result was written, and the shipped Result and the exported `preference.csv` still
    carry `p = 5e-05` with an undefined rho on two PRIMARY preference rows. **`common.provenance` must carry the
    sha256 of the analysis script and its imported reducers, and any change to them after a Result is emitted requires
    re-running the analysis on the same fetched batch (CPU, cheap) and re-emitting every artifact derived from it.**
12. **The fetch is part of the experiment: slim on the box and pull the consoles first.** A 12-s matched-sphere run
    writes a 114 MB npz of which 106 MB is per-frame graded arrays no consumer reads; object round 2's two batches
    cost 1.7 GB (slimmed) and 36 GB (not). Slim on the box before the transfer, and **pull `*.json` and `*.txt` before
    the `*.npz`** -- only 14 of 84 consoles of the baseline batch survived its hand fetch, which left the
    console-vs-JSON device cross-check of item 4 unavailable for 70 runs. A tool's `verify` must count consoles
    against runs and raise a problem when they do not match.

13. **A verdict-agreement statement is produced by a script, not by eye.** `docs/audits/body_sided_state.md`
    section 4 said the five-run and four-run tables "differ in NO verdict (0 rows ... checked by diffing the
    verdict columns)". Diffing `out/vncd3/analysis` against `analysis_4runs` gives **15 of 765 pairwise and 7 of
    548 room-table verdict cells** (reproduced independently). No headline row flips, so no conclusion moves --
    but the five-run tables span two submissions (seed 2 from `vncd3-f3bb50`, seeds 0/1/3/4 from `vncd3b`) while
    the four-run table is the single-submission one this document asks for. **RULE: a verdict-agreement statement
    is produced by a script that diffs the two CSVs and prints the count and the flipped keys, pasted verbatim
    into the audit (this extends item 7); and when a family spans two submissions, the single-submission table is
    the headline and the pooled one is the supplement.**
14. **A suite claim is device-scoped, and the CPU path is the reference.** `add_low`'s "suite 12/0/0 in 5/5" holds
    on H200 and on B200 -- the seed-locked values are bit-identical across the two GPUs (`taste.MN9_hz`
    10.93417739868164 / 2.4795215129852295 on all nine runs) -- and **FAILS on the CPU**: 1.9669914 against a
    criterion of `> 2` (with the term off, CPU 5.0909, PASS). A shipped value under an opt-in mode therefore fails
    a scored check on the path the project's bit-identity rule uses. **RULE: any suite claim about a candidate
    default (or about a shipped value under an opt-in mode) is reported on the CPU path as well as the GPU;
    "seed-locked" sections are named with the device they are locked on; and a 2x CPU-vs-CUDA difference on a
    scored check (MN9 5.09 vs 10.93 -- a residual of +-1.7 V/s inputs) is itself a defect of the check and is
    recorded as one.**
15. **`--arm-block` must block the family, and a one-job block is a bug.** In unitary batch 1 `--arm-block fam`
    resolved to **one block per FILE**: 16 one-job blocks, dealt round-robin over two H200s. It happened to be
    balanced (seeds 0/2 -> box b, 1/3 -> box a for every arm) and the skeptic reproduced the family in one block
    on a B200, so nothing is confounded -- but as executed it was a departure from item 9's rule, and the ANSWER's
    "one submission, both boxes H200, balanced per arm" did not say so. **RULE: `cluster_run.py` refuses (or at
    minimum prints a red line and requires `--confirm`) when `--arm-block` yields more blocks than jobs/2, or any
    block of size 1 in a batch larger than the target count; job commands carry a family token that is the SAME
    string on every arm (`fam_<family>`), with `--arm-block-map` as the fallback; and the audit prints the
    block-to-box table verbatim.**
16. **"Bit for bit" is never said of a GPU number** (item 2, restated because this round broke it three times).
    The walk triple 48.48052978515625 / 20.109053071339925 / 4.629162311553955 was called "bit for bit rounds
    4/5/6/7" in `guard_suites_r3.md`, `anti_runaway.md` round 7 and `round3_integration.md` section 1, while
    (a) the attribution of the candidate triples to rounds 4/5 is partly wrong (round 4's `lpi_x1` `walk.GF_max`
    was 9.7681, not 9.7951), (b) the same walk section is **not** seed-locked under the unitary-high arm on the
    eager path (90.47 vs 44.70 between two draws on one box), and (c) the CPU reads **57.38 / 26.66 / 9.76** for
    the shipped default. **RULE: "bit-identical" is a CPU claim; a GPU value that repeats is "reproduced the
    recorded value exactly in n draws on backend X", and the CPU value is quoted beside it whenever the number is
    used as a reference.**
17. **The adopt-alone rate-half is a two-sample exact Poisson, stated in one direction.** The guards audit
    attributed "inside the baseline's exact Poisson 95 % CI" to rounds 4-6; round 5 read it one-sided ("not worse
    in either route") and no round stated CI containment. Containment of the candidate's point estimate in the
    baseline's CI ignores the candidate's own sampling error -- **14.9-15.6 % false failure for an identical true
    rate at 3-6 batches per arm, and it does not improve with n** -- and it was applied asymmetrically: the
    transducer arm, also outside the CI but *below* it, was passed. **RULE: the rate-half is a two-sample exact
    Poisson (conditional binomial) at equal exposure, quoted beside the run-level `common.compare`; the direction
    is stated once for every arm; and the prescribed replication is derived from a power calculation on the
    observed contrast (here >= 6 runs per arm: power 0.85 at n 6, 0.50 at n 4 for 15.0 vs 24.3 hops per run), not
    a fixed ">= 4".**
18. **Compare on unrounded values; round only at print time.** `scripts/probe_unitary.py` `_stats()` rounds to
    4 dp **before** `common.compare` (compass `mid` `rest_mean_post` z 4.50 rounded against 7.12 unrounded; `low`
    -3.50 against -7.29), and the integration's pairwise table prints z to the nearest integer, so "z-3 p 0.032
    null" appears beside a `|z| >= 3` criterion and reads as a contradiction (the actual values are -2.88 and
    -2.77). **RULE: compare on unrounded values, round only at print time, and print z to one decimal in any table
    whose criterion is `|z| >= 3`.**
19. **"Verified" needs a receipt, or the word is "size-verified".** "md5-verified against the box" (unitary batch
    1) rests on a size compare (`FETCHED.txt`) -- **no tool in `cluster_run.py` or `fetch_run.py` computes a
    digest** -- and the box has since been destroyed, so it cannot be re-checked; "the 155 raw job files ... with
    their md5" (monoamines) is 40 files. **RULE: `fetch_run.py` writes a per-file sha256 receipt (the remote hash
    computed on the box before transfer, compared locally), and an audit says "md5/sha256-verified" only with the
    receipt path; otherwise it says "size-verified".**
20. **A shared dependency is committed before the batch is submitted.** Every batch of this round shipped
    uncommitted cross-task files (item 10.1(7b) violated by all five threads), and the guards batch ran an
    **intermediate `body.py`** (md5 46e3c10a, neither HEAD nor the present tree), so its "transducer on" arms are
    the previous round's sense and the audit had to read the tree back over ssh. The never-edit list was breached
    once -- `flyverse/brain.py`, thread unitary, through the "a new `LIFParams` field is unavoidable" carve-out
    (default `None` plus a CPU bit-identity test, so the letter was kept). **RULE: a shared dependency is
    committed (on a branch if need be) before the batch is submitted, or the batch is explicitly a working-tree
    batch and the audit header carries `submit_tree.txt` plus the md5 of every `flyverse/` file that differs from
    HEAD; and the never-edit list names the carve-out's two conditions AND requires a line in the hand-off.**
21. **The scheduler's receipt is the record of "n jobs, 0 failed", not the client's console.** Every predecessor
    agent died at a session limit; the rented boxes were stopped ~5 h for funds; four of six batches lost their
    `cluster_run.py` client and were pulled by hand (`scripts/box_status.py` and `scripts/fetch_run.py` were
    written for it); unitary batch 1's log ends in "connection refused" with no summary line. **RULE: the
    scheduler's own completion receipt (job counts and exit codes) is the record of "<n> job(s), <m> failed",
    fetched as a file into `out/<dir>/`, never the client console; `cluster_run.py` gets an `--attach <run>` mode
    so a new client can resume the wait / fetch of an existing run; and rented-box batches are kept inside the
    session budget.**
22. **One table, one frame mask.** Inherited from `probe_vnc_drive`'s reducer: `DNa02_L` / `DNa02_R` include
    airborne frames while `DNa02 L-R` is the non-airborne window, so the AC row reads 2.014 / 0.940 beside an
    L-R of +1.124 (the okf-masked pair, 2.061 / 0.937, does subtract). Separately, `sided_frames()` aligns the
    recorded samples one body frame early (`scripts/probe_vnc_drive.py` ~1126), understating the AN04B003
    sidedness by ~13 %. **RULE: a reducer declares the frame mask per key in the CSV header, rows that are
    arithmetically related share one mask, and a lag scan is part of the validation of any cross-signal
    correlation.**
23. **When arms differ in clean fraction, quote the window-matched statistic and call the verdict on it.** Under
    the unitary-high arms the clean frames are 12-17 % of the window (every fly off the table in 7.7-11 s)
    against 94 % under the transducer alone, so every AC-vs-A row is selection-confounded; window-matched
    re-scoring shrinks AC v A yaw SD from +5.09 to **+4.55** (5-16 s) and **+3.78** (5-10 s) -- the ordering
    holds, the magnitude is inflated by 15-26 %. **RULE: when arms differ in clean fraction by more than ~2x, the
    window-matched statistic is quoted beside the full-window one and the verdict is called on the matched one.**
24. **"Underpowered by rule" is not a verdict to plan for.** Both the body-state and the integration batch ran
    **3 seeds per compass arm** (3 v 6 per phase, 3 v 3 for the flip rows), so every compass row was
    `underpowered` by item 1 before the first job was submitted, and the audits quoted per-seed lists instead.
    **RULE: no arm is planned below 4 runs unless the audit predeclares that its rows are magnitudes only.**
25. **Every bracket edge names its source, or is labelled "headroom".** This round: Strauss & Heisenberg 1990
    read as "tripod at every walking speed" (the paper says at the fastest walking); a "stance amplitude ...
    largely kept constant" quotation not found in DeAngelis 2019 (the number is self-derived and sound);
    `lit.HS.oa_response_gain`'s upper edge 2.3 with no source; `unitary.IoverE.chloride_driving_force` citing no
    E_Cl source at all (larval / embryonic values give an I/E of 0.0-0.15, below the row's own bracket); Kazama &
    Wilson 2008's primary EPSP possibly 7 mV rather than the 5 mV gloss; and the one insect unitary I/E on record
    (0.28, Periplaneta, J Neurosci 34:13039) missed entirely. **RULE: every bracket edge of a `lit.*` / `unitary.*`
    row names its source verbatim or is labelled "headroom" (as the 250 Hz haltere ceiling is), and quotations are
    checked against the full text before they are typed as quotations.**
26. **The REPORT / ANSWER block is generated from the audit's tables, never retyped.** The header text drifted
    from the body three times in one round: the integration's "undetermined/structural" against the body's "null
    with a zero-SD null"; guards' "21 wrapper runs" against 18 and "wrong in direction" against "not what
    happened"; monoamines' "zero scatter in the abort time" when frame 51 is the guard's earliest possible abort.
    **RULE: the REPORT / ANSWER block is generated from the audit's tables, or the audit's section 0 IS the ANSWER
    verbatim -- it is never retyped by hand.**
27. **A round with two workflows names one owner per file and one closing critic.** Round 3 ran a behaviour
    workflow and an object workflow over the same house cluster and the same working tree, and an external agent
    (Astra) took the object tasks mid-round after API 529s; the object half's skeptic passes were the author's own
    separately implemented checks, not a second agent, and it had no NOTES entry until the owner wrote one.
    **RULE: one hand-off file per round names every file's owner (done: `docs/HANDOFF_ROUND3_ASTRA.md`,
    git-ignored), and the closing critic covers both workflows or says plainly which one it does not.**

### 10.3 The minimal command sequence (an odour-to-DN question, as an example)

```bash
PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py --results "out/interp/*/*.json" out/benchmark_suite.json --status "FAIL,KNOWN GAP" --json out/interp/ledger/fails.json
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a "class=olfactory" --b "DNa02" --k 3 --json out/interp/paths/orn_dna02.json
PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py analyse --static --target DNa02 --json out/interp/decompose/dna02_static.json
# `st=$?; tail ...; exit $st` keeps PYTHON's exit code as the job's (a bare `; tail` makes it tail's: 10.4 item 4).
# The \$ below is escaped so THIS shell leaves it for the job's shell; ${a} / $s are expanded here, on purpose.
cmds=(); for a in stim ctrl null; do for s in 0 1 2 3 4; do cmds+=("mkdir -p out/od && python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_trace.py record --protocol odour --arm $a --seed $s --series-every 5 --out out/od/${a}_r$s > out/od/${a}_r$s.txt 2>&1; st=\$?; tail -4 out/od/${a}_r$s.txt; exit \$st"); done; done
python scripts/cluster_run.py --name od --minutes 40 "${cmds[@]}" --fetch out/od/ 2>&1 | tee out/od_cluster.log      # read: 15 job(s), 0 failed; then 15 .npz in out/od/, and each run block's device cuda + its --arm
ls out/od/*.npz | wc -l; grep -c "device cuda" out/od_cluster.log     # necessary, not sufficient: 15 and 15
PYTHONIOENCODING=utf-8 python scripts/interp_trace.py analyse --source "class=olfactory" --stimulus "out/od/stim_r*.npz" --control "out/od/ctrl_r*.npz" --null-runs "out/od/null_r*.npz" --stat mean --min-cells 2 --decompose-at DNa02,DNg56 --json out/interp/trace/odour_dna02.json
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a "class=olfactory" --b DNa02 --k 3 --recording out/od/stim_r0.npz --json out/interp/paths/orn_dna02_nf.json
PYTHONIOENCODING=utf-8 python scripts/interp_export.py analyse --result out/interp/trace/odour_dna02.json --out out/export
```

---

## 11. Contract defects -- closed in this revision / still open

The build round found ten defects in the shared files (`common.py`, `__init__.py`); each had been worked around
inside a tool. **Eight are closed here**, in the shared layer, and the private workarounds are gone with them. What a
tool must do now is stated with each.

### 11.1 Closed

1. **`raw_counts` merges instead of substituting.** `C.data = maximum(|W.data|, connectome.sign0_counts)`, so every
   signed edge keeps its own count and the sign-0 entries gain theirs; `with_sign0=False` means *exclude the sign-0
   entries* (they stay 0), and `dtype=np.float64` gives the exact whole-model total. The five private copies
   (`paths.raw_counts`, `decompose.counts_matrix`, `health.full_counts`, `trace.full_raw_counts`,
   `export.raw_counts`) are one-line delegations to it and agree entry for entry (tested). Effect: the mandatory
   Neurome column `synaptic_pair_count` is no longer 0 on every signed edge -- `trace`'s `depth_edges` read 0.0 on
   11,090 of 11,094 rows and its `lost_inputs.raw_synapses_per_post` on 96 of 96 in the shipped
   `out/interp/trace/odour_mean.json`, and both now carry the counts (2,619 / 6,546 / 47.5 ...).
2. **`silent_flags` returns booleans.** With no `rates` the never_firing question was not asked and the column is
   `False`, and `links` reads a missing or unevaluated flag as False rather than `bool(NaN) == True`. Effect: a
   structural table no longer marks every row `never_firing` (it did on 5,000 of 5,000 rows of
   `out/interp/apply_turning/static_legMN.json`, whose `silent_entries` equalled `n_entries` on all 21,905 per-type
   rows). The private handling in `paths._contributions` and `export.silent_flags` is gone.
3. **`compare` reports `p_floor` and calls three runs `underpowered`.** `MIN_REPLICATES` stays 3 for the scatter
   rule, but the verdict now tests p against the floor the arm sizes allow (`common.p_floor`, 2 / C(n_a+n_b, n_a)):
   0.10 at 3 v 3, so no z can make a 3-run difference a result, and the JSON says so instead of saying `null`.
   Effect: at 3 runs per arm a verdict column reads `underpowered` where it read `null` (13,999 rows of
   `out/interp/decompose/taste_chain_by_type.json`); every z / p / value is unchanged.
4. **A deterministic null is `undetermined`, not a tool's own answer.** SD(null) == 0 (up to float noise) leaves z
   undefined; `compare` returns `undetermined` with `null_sd_zero`, unless the rank test settles it as null
   (p > alpha), in which case it is `null`. The three private answers are gone: `trace` no longer overrides the
   verdict to `result` (it declares `CARRIER_VERDICTS = ('result', 'undetermined')` -- a carrier is a carrier, and
   the ORNs' +83 Hz under a bit-identical null is one), the atlas keeps `z_floor` as its **declared effect size**
   with `atlas.called()` naming the rows it reports, and `decompose` no longer lets z stand as a verdict on a
   near-zero group. Effect: verdict columns only. `out/interp/trace/odour_mean.json` moves 13 of 11,147 rows from
   `result` to `undetermined` (carriers, lost stage, decomposition and `validation` identical); the atlas' shipped
   tables move 3,212 -> 0 `result` and 0 -> 4,140 `undetermined` with **zero** change to the movers, the summary,
   the per-body table or the validation.
5. **One null flag.** `add_common_args` declares `--null-runs GLOB [GLOB ...]` (section 2.6) with `--null-arm` /
   `--null-recordings` as hidden aliases, and no longer declares a bare `--null`; a subcommand that must GENERATE
   the null arm declares its own (`interp_export.py record --null`). The ledger keeps `--null` as a hidden alias of
   `--null-runs`. Section 7's examples all run.
6. **The two namespaces are separate.** `flyverse.interp.<tool>` is the module, `flyverse.interp.tools.<tool>` /
   `interp.tool('<tool>')` is the function (section 4). Nothing depends on import order, and every wrapper's
   `from flyverse.interp import atlas as atlas_mod` keeps working. The contract test imports all eight submodules
   first, on purpose, and then checks both namespaces.
7. **`VALIDATION['health']` says 3,312**, the shipped cache's sign-0 body count (2,683 of them presynaptic), with a
   note that 3,407 was the pre-`TYPE_NT_OVERRIDE` cache's. `interp_health.py structure --by module` reproduces
   3,312 / 2,683 / 2,701,289 of 124,161,873 synapses from the cache.
8. **`provenance()` carries `source_fingerprint`.** When `git_state()` cannot resolve the commit (a cluster copy has
   no `.git`), `common.source_fingerprint` calls `export.source_fingerprint(include_loaded=True)`, so every Result
   -- not only the export's -- can be matched to a checkout by content (`export.match_sources`). When git answers,
   the block records that instead of hashing anything, so nothing pays for it twice. `commit: unknown` is now the
   last resort, not the only answer.

### 11.2 Still open (owner in brackets)

9. **`--device cpu` sets `CUDA_VISIBLE_DEVICES=""` in `scripts/interp_trace.py`** [trace]. Windows unsets an empty
   variable, so a "CPU smoke" runs on the desktop's GPU with `device_requested cpu` / `device cuda` in the
   provenance. `-1`, set before torch is imported, is the fix (`interp_apply_rotation.py` has it); the guard belongs
   in whatever `add_common_args` grows for `--device`, but the wrapper is the tool's file. Rule 5 of section 10.4
   until then.
10. **Advertised flags that are no-ops** [lesion, health, paths]: `interp_lesion.py` takes `--null-runs` / `--seed`
    and reads neither; `interp_health.py record` takes `--replicates` / `--null-runs` and writes one stem;
    `interp_paths.py` accepts the replicate flags for uniformity (documented -- the tool is deterministic). Wire
    them or drop them per wrapper.
11. **One validation semantics** [all eight]. `validation.status = 'reproduced'` still means eight different things
    (paths: 63 numeric checks; trace: verdict-level, and qualitative for the odour gate; the atlas' `validate()`
    bypasses the replicate floor; lesion's `_close` pads a published range by +-50 % and compares means, not draws;
    the ledger's analyse-mode status compares nothing numerically). Every status should be a numeric comparison
    against `VALIDATION[tool]['reference']` with its tolerance stated. The verdict half of this is closed
    (11.1.3, 11.1.4); the validation half is not.
12. **`lesion` cannot address one presynaptic class onto one postsynaptic class** [lesion]: the `edges` kind lives
    only in `scripts/interp_apply_object.py`. Promote it, make the scatter rule use `max(sd_arm, sd_baseline)`, and
    compare draws rather than means to a published range.
13. **The atlas has no live null and no vision context** [atlas]: the shipped atlas' null is bit-identical 0.000 in
    100 % of 26,487 rows, which is why 11.1.4 shows up there as 4,140 `undetermined` rows. Inhibitory movers are
    invisible and the optic drive is absent. A context arm exists and was never used.
14. **No front door** [design]: nothing runs section 10's procedure end to end from a failing ledger row
    (`scripts/interp_deficit.py`, or `interp_ledger --explain <row>`).
15. **Neurome-facing gaps** [export, trace]: graded units carry no received drive in mV; the per-body `verdict`
    column is an uncorrected screen (5-11 % of bodies at chance) and should be labelled or gated as one.
16. **`docs/NEUROME_INTERFACE.md` line 73 still says 3,407 sign-0 presynaptic bodies** [that document's owner], and
    `scripts/interp_health.py`'s validate note still describes `VALIDATION['health']` as carrying 3,407, which it no
    longer does [health].
