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
   # sign0 | frozen | pruned | never_firing (max rate over the rollout < 0.5 Hz; NaN without rates)
raw_counts(c) -> (counts csr, sign0_available)
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
compare(stim_values, null_values, z_min=3.0, min_n=3) -> {"stim": ArmStats, "null": ArmStats, "diff", "z", "welch", "U", "p", "verdict"}
```

`z = (mean stim - mean null) / SD(null)`, Welch = the difference over the standard error of the two means, exact
Mann-Whitney U and p for small arms (`docs/audits/object_sweep.md` 8.4; the test reproduces LPLC2 sign-abs z +5.4 /
Welch +8.6 / U 25 / p 0.0079 and LC11 off z -0.1 / U 12 from the audit's numbers). **Verdict `underpowered` whenever an
arm has fewer than three runs, whatever the numbers**; `result` needs |z| >= 3 and p <= 0.05; else `null`. Every tool
that reports a difference reports this dict, never a bare z. `replicate_seeds(n, seed0)` names the runs.

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

`--json PATH` (default `out/interp/<tool>/<run_id>.json`), `--replicates N` (default 3), `--seed S`, `--null`,
`--device`, `--cache-dir`, `--receptor-model default|off|sign|sign+gain|full`, `--receptor-net-rule`,
`--receptor-table PATH`, `--lif KEY=VALUE` (repeatable, JSON / literal values), `--optic KEY=VALUE`, `--quiet`.
`params_from_args(args) -> (LIFParams, OpticParams)`. Every wrapper ends with `print_table(...)` and the JSON path.

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
3,407 sign-0 presynaptic bodies (2.2 % of synapses), the mushroom body losing 9.8 % of input / 12.5 % of output, the
octopaminergic visual centrifugal cells 17.6 % of output (`nt_audit.md`).

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
export(result, *, out_root="out/export", run_id=None, retina=None, parquet_rows=1_000_000, control_ids=None) -> Path
```

A serializer: `manifest.json` (every field of `docs/NEUROME_INTERFACE.md` section 1 from `provenance`; `tables` with row
counts, columns, units, SHA-256) and `readout_per_body.csv` / `contributions.csv` / `sensitivity.csv` (Parquet above
`parquet_rows`) from `Result.tables`, `dataset` / `release` columns prepended, bodyIds as decimal strings, model index and
type alongside. Refuses a Result whose `check()` is non-empty. Validation: a round trip of a Result through the export
and back reproduces every number; LC11 / LC10a bodies have two rows.

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
| health | the refractory-limited bump; the sign-0 populations | EPG 180-260 Hz = load 0.40-0.57 at t_ref 2.2; PEN 40-65, Delta7 90-112; 3,312 sign-0 bodies (2,683 presynaptic; `common.VALIDATION` still carries 3,407 = the pre-`TYPE_NT_OVERRIDE` cache's count, interp_health.md 2), MB 9.8 % input silenced | cx_wedge.md 7, cx_glno.md 5; nt_audit.md |
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
PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py analyse --target DNp01 --recordings "default=out/dec/default_r*.npz" --null-arm "off=out/dec/off_r*.npz" --by type,transmitter,tier --window 0.5,1.5 --json out/interp/decompose/gf_walk.json
# static, CPU: the taste carrier
PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py analyse --target "OA-AL2i3|TmY14|DNge138|DNge149|DNge150" --static --receptor-table out/receptors_holdBrainHis.csv --json out/interp/decompose/taste_static.json

# trace: the moving ball from the photoreceptors, 3 stimulus + 3 control + 3 null runs in ONE batch, then CPU
python scripts/cluster_run.py --name tr-ball --minutes 30 $(for a in stim ctrl null; do for s in 0 1 2; do echo "\"python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_trace.py record --protocol object --arm $a --seed $s --out out/tr/${a}_r$s > out/tr/${a}_r$s.txt; cat out/tr/${a}_r$s.txt\""; done; done) --fetch out/tr/
PYTHONIOENCODING=utf-8 python scripts/interp_trace.py analyse --source "type:R1-R6|R7y|R7p|R7d|R7_unclear|R8y|R8p|R8_unclear" --stimulus "out/tr/stim_r*.npz" --control "out/tr/ctrl_r*.npz" --null-runs "out/tr/null_r*.npz" --stat best_cell --stage-table family --decompose-at T3,T2,Tm5Y,TmY21 --json out/interp/trace/ball.json
# (--stat figure_z is the signed retinotopic figure, which the sweeping ball does not survive: it reads 'not reproduced'
#  on the object target; best_cell is the object sweep's statistic and the validation's -- interp_trace.md 2.1)

# paths: CPU, the rotation inputs of PEN and the ring loop
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a "LNO1|LNO2|LNOa|SpsP|PS196_b|~^LAL" --b "PEN_a|PEN_b" --k 3 --json out/interp/paths/rot_pen.json
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a EPG --b EPG --k 2 --level cell --wedge --json out/interp/paths/epg_loop.json

# lesion: plan -> one batch -> analyse
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py plan --manifest docs/interp/lesions_holds.json --out out/les_holds --replicates 3
bash out/les_holds/batch.sh            # the single cluster_run.py call plan wrote; ships out/les_holds_cluster.log
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py analyse --out out/les_holds --json out/interp/lesion/holds.json
# the CPU validation (taste / smell, no optic lobe): --device cpu runs every job locally in-process
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py run --manifest docs/interp/lesions_holds.json --sections taste,smell --device cpu --out out/les_cpu --replicates 3

# atlas: 64 populations per batch, three runs, on the cluster; then CPU
python scripts/cluster_run.py --name atlas-dn --minutes 45 "python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_atlas.py run --populations superclass=descending_neuron --by-side --hz 150 --ms 400 --seed 0 --out out/atlas/dn_r0 > out/atlas/dn_r0.txt; cat out/atlas/dn_r0.txt" "... --seed 1 ..." "... --seed 2 ..." --fetch out/atlas/
PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/dn_r*" --json out/interp/atlas/dn.json

# health: from a compass-room recording; and live in the observatory
PYTHONIOENCODING=utf-8 python scripts/interp_health.py analyse --recordings "out/health/compass_r*" --window 5.5,8 --groups meta --json out/interp/health/compass.json
PYTHONIOENCODING=utf-8 python scripts/interp_health.py analyse --recordings "out/health/walk_default_r*" --null-recordings "out/health/walk_off_r*" --window 0.5,1.5 --json out/interp/health/walk.json
python -c "from flyverse.interp import HealthReadout; ...; fb.nt_source = HealthReadout(fb)"     # then N in the map

# ledger and export
PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py --results "out/interp/*/*.json" out/bench.json --json out/interp/ledger/all.json
PYTHONIOENCODING=utf-8 python scripts/interp_export.py --result out/interp/trace/ball.json --retina out/tr/retina.npz --out out/export
```

---

## 8. Implementer checklist (one per tool; the shared files are not yours)

Files you own: `flyverse/interp/<tool>.py`, `scripts/interp_<tool>.py`, your test class in `tests/test_interp.py` (append;
the `graph()` there has a photoreceptor, a graded unit, a VPN, the GF, DNa02 L/R, a sign-0 GLNO and a PEN), and for the
ledger `flyverse/data/expected_responses.csv`. Do not edit `common.py` or `__init__.py`; if the contract needs a change,
say so in your report and keep the stub's parameters.

1. The function keeps every parameter of `stubs.<tool>` (the test `StubTests` enforces it) and returns `common.Result`.
2. Every stochastic measurement has `--null` or `--replicates` (default 3) and reports `compare`'s dict.
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
   under the shipped model, **>= 4 runs per arm, 5 when the effect is small** -- three runs cannot give a `result`
   through `common.compare` (exact U floor p = 0.10 at 3 v 3; 0.029 at 4 v 4; 0.0079 at 5 v 5). Add the
   deterministic arm when one exists (`--optic gain_fb=0` for optic-lobe questions: its null is exactly 0 and it is
   read as magnitudes, not verdicts). One `cluster_run.py` call, `mkdir -p` in every job line, `--fetch` a NAMED
   subdirectory, one client per directory; read `'<n> job(s), 0 failed'` and every job's `device cuda` before
   analysing; `interp_apply_rotation.py verify-batch <dir>` (or the tool's console-vs-meta check) before analysis.
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
   `rest`), **>= 4-5** for any check with run scatter (`loom.GF_peak_hz`, `rotate.DNp20_flip_hz`, the walk power pair,
   anything from the room); read `dissociations` only for rows called by bit-identity or by seed pairing, and quote a
   scatter-rule dissociation with its draw count (the same `holds` manifest gave 1 / 8 / 17 dissociations over three
   batches of 4 / 4 / 3 draws; only the (holdDN1, holdKC) x (Shiu sugar, Shiu bitter) pair survived all three).
7. **Write it down and export it.** `docs/audits/deficit_<name>.md` in the shape of the three shipped ones: the
   answer first (where; which kind; the data-driven route if one exists, the hand-crafting explicitly named and not
   done), every number from a named file or run with its scatter over >= 3 runs, the batch line, the defects found in
   the tools. `interp_export.py analyse --result <the Result>` for every JSON a partner may read (the export refuses a
   Result whose `check()` is non-empty; `verify` must print `problems: none`). Add a ledger row for the new
   localization so the next round scores it. Nothing in `flyverse/` outside `interp/` changes.

### 10.2 Reading rules (the ones this round had to learn)

* A `result` needs |z| >= 3 **and** p <= 0.05 over >= 3 runs; at 3 v 3 that is unreachable, so a 3-run comparison
  reports z and the values and is not a result. A deterministic null (SD 0) makes z NaN / astronomical: read the
  magnitude, never the verdict (`note null_sd_zero`).
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
  `provenance.compiled_connectome.md5` + `files.effective_weights_md5` + the export's `source_fingerprint` (28 loaded
  files, matched by content); quote those, not the commit.

### 10.3 The minimal command sequence (an odour-to-DN question, as an example)

```bash
PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py --results "out/interp/*/*.json" out/benchmark_suite.json --status "FAIL,KNOWN GAP" --json out/interp/ledger/fails.json
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a "class=olfactory" --b "DNa02" --k 3 --json out/interp/paths/orn_dna02.json
PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py analyse --static --target DNa02 --json out/interp/decompose/dna02_static.json
cmds=(); for a in stim ctrl null; do for s in 0 1 2 3 4; do cmds+=("mkdir -p out/od && python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_trace.py record --protocol odour --arm $a --seed $s --series-every 5 --out out/od/${a}_r$s > out/od/${a}_r$s.txt 2>&1; tail -4 out/od/${a}_r$s.txt"); done; done
python scripts/cluster_run.py --name od --minutes 40 "${cmds[@]}" --fetch out/od/ 2>&1 | tee out/od_cluster.log      # read: 15 job(s), 0 failed; device cuda x15
PYTHONIOENCODING=utf-8 python scripts/interp_trace.py analyse --source "class=olfactory" --stimulus "out/od/stim_r*.npz" --control "out/od/ctrl_r*.npz" --null-runs "out/od/null_r*.npz" --stat mean --min-cells 2 --decompose-at DNa02,DNg56 --json out/interp/trace/odour_dna02.json
PYTHONIOENCODING=utf-8 python scripts/interp_paths.py --a "class=olfactory" --b DNa02 --k 3 --recording out/od/stim_r0.npz --json out/interp/paths/orn_dna02_nf.json
PYTHONIOENCODING=utf-8 python scripts/interp_export.py analyse --result out/interp/trace/odour_dna02.json --out out/export
```

---

## 11. Contract defects the build round found in the shared files (open; owner: the design task)

Each was worked around inside a tool and reported by its implementer and skeptic; none is fixed in `common.py` /
`__init__.py`, so a new tool built on the contract inherits it.

1. `common.raw_counts(c)` replaces the whole count vector with `connectome.sign0_counts` (non-zero only on explicit
   zeros): every signed edge reads 0 synapses, `common.links` writes `synaptic_pair_count 0` -- a mandatory Neurome
   column. Five private copies of the fix exist (`paths.raw_counts`, `decompose.counts_matrix`, `health.full_counts`,
   `trace.full_raw_counts`, `export.raw_counts`). Fix: `C.data = np.maximum(np.abs(c.W.data), cnt)`.
2. `common.silent_flags(rates=None)` returns `never_firing = NaN` and `links` tests `bool(flag)`, so every structural
   table reads `never_firing`. Fix: `False` when no rollout, or `flag is True`.
3. `common.compare` requires p <= 0.05 at `MIN_REPLICATES = 3`, where the exact two-sided U floor is 0.10: three runs
   is simultaneously the floor for not being `underpowered` and a guaranteed `null`. Fix: `MIN_REPLICATES = 4`, or
   test p against its own floor and report `p_floor` (trace does).
4. `compare` on a deterministic null (SD 0) gives z NaN and verdict `null` (+83 Hz called null) or, with p reachable,
   `result` on nothing; trace overrides, atlas floors the SD at 0.05 Hz, decompose does neither (z 1e18-1e41 on
   near-zero groups). Fix: a declared `undetermined` verdict with the magnitude when SD(null) == 0.
5. `add_common_args` declares `--null` as `store_true`; four wrappers need a value and each chose a different name
   (`--null-arm`, `--null-runs`, `--null-recordings`, the ledger's resolved `--null PATH...`), section 7 of this
   document was wrong on two lines until this revision, and `decompose analyse ... --null-arm X --null` silently drops
   the control. Fix: one `--null-runs GLOB` in `add_common_args`; the bare flag only on `record`.
6. `flyverse.interp.<tool>` is shadowed by the submodule once it is imported (`__getattr__` never runs again), so
   `StubTests` fails under `python -m unittest` (alphabetical order) and passes under pytest; `globals()[name] = obj`
   would break every wrapper's `from flyverse.interp import atlas as atlas_mod`. Fix: keep the two namespaces apart
   (the tests import the function from the submodule).
7. `common.VALIDATION['health'].sign0_presynaptic_bodies = 3407` is the pre-override cache's count; the adopted
   model's is 3,312 (`docs/NEUROME_INTERFACE.md` line 73 carries the same 3,407).
8. `common.provenance` records no source fingerprint, so every cluster JSON says `commit: unknown`;
   `export.source_fingerprint` / `match_sources` exist and should be called from `provenance()` whenever
   `git_state()` fails.
9. `--device cpu` in `scripts/interp_trace.py` sets `CUDA_VISIBLE_DEVICES=""`, which Windows unsets: a CPU smoke
   runs on this desktop's GPU with corrupt provenance (`device_requested cpu`, `device cuda`). `-1` before torch is
   imported is the fix (`interp_apply_rotation.py` has it); the same guard belongs in `add_common_args`.
10. `scripts/interp_lesion.py` advertises `--null` and `--seed` and never reads them; `scripts/interp_health.py
    record` advertises `--replicates` / `--null` and writes one stem. Either wire them or drop them per wrapper.
