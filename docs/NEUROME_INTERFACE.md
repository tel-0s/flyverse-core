# Flyverse <-> Neurome: the evidence / probe interchange

Reply to Neurome's request (`D:\Projects\neurome\docs\flyverse-evidence-handoff.md`, 2026-09-12). Neurome owns
source anatomy, reconstruction evidence and molecular correspondence; flyverse owns the model, its physiological
transforms, simulation and probes. First joint case: the small-object pathway (T2, T3, Tm5Y, TmY21, TmY13, TmY5a,
LC11, LC10a, LC10b) -- localized on our side to the medulla -> lobula stage (`docs/audits/object_sweep.md`; the
medulla carries a moving ball at z +22 to +29, T2 / T3 / Tm5Y / TmY21 and the LC types sit at the none-vs-none null).
Receptor-expression coverage within that scope (corrected per Neurome's reading of `receptors_by_type.csv`, which
matches ours): T2, T3, TmY5a, LC10a and LC10b have profiles; **Tm5Y, TmY21, TmY13 and LC11 do not**, in any of six
sources. Of LC11's larger upstream set (below) only T2a is profiled. A profile is a cross-dataset expression
interpretation, not a measured conductance of a MaleCNS cell.

**Edge-level agreement.** Neurome's exported input distributions reproduce our compiled cache to the synapse: LC11 <-
T3 69,205 / T2 36,917 / Tm6 24,736 / T2a 24,572 / Tm12 19,721 / TmY18 15,593 / LC11 15,491 / Li15 13,918; LC10a <-
LC10a 25,244 / TuTuA_2 18,097 / AOTU042 14,492 / Tm5Y 14,221 / LC9 9,661 / LC10c-1 8,823 / TmY21 7,406 / LC10c-2 6,981
(our fractions are 0.5-1 pp higher only because our denominator excludes sign-0 entries). Tm6, T2a, Tm12, TmY18 and
Li15 -- 29 % of LC11's input outside the original probe list -- join the populations the interpretability tools trace.

## 1. What flyverse will export (read-only probe export)

Implemented by the interpretability toolkit (`flyverse/interp/export.py`, CLI `scripts/interp_export.py`; see
`docs/INTERP.md` once written). Location: `out/export/<run_id>/` with `manifest.json` plus CSV / Parquet tables,
one directory per probe run; a run is a directory, never a mutable file. The interchange key is
**(dataset.name, dataset.release, bodyId)** with matching source fingerprints -- `bodyId` as a decimal string (int64 in
our cache, 167,106 unique); model indices and type names travel alongside. Cross-release joins require an explicit
correspondence (unchanged / merge / split / retired / uncertain), never an unchanged number alone (Neurome's
`evidence-bundle-v1.md`). Field conventions follow that contract where they overlap: `body_pre` / `body_post`,
anatomical counts as `synaptic_pair_count` (uncapped, unsigned, pre -> post), coordinates with dataset space, axis
order and physical units.

`manifest.json` (provenance, all fields mandatory):

| field | source |
|---|---|
| `run_id` | `<probe>-<utc timestamp>-<8 hex>` |
| `flyverse_commit` | `git rev-parse HEAD` (+ `dirty` flag) |
| `dataset_release` | `male-cns v1.0 flat-connectome`, the four file names + SHA-256 from `flyverse/data/manifest.json` |
| `compiled_connectome` | `cache/` fingerprint: md5 of `W_post_pre.npz` data / indices / indptr, `sum_abs_W`, `nnz`, `n_neurons`, the `TYPE_NT_OVERRIDE` and `UNKNOWN_NT_OVERRIDE_REGEX` in force |
| `model` | `LIFParams` (every field incl. `receptor_model` / `receptor_net_rule` / `receptor_table` md5, `type_path_gain`, `path_gain`), `OpticParams` (pair gains, `gain_out_mv`, `out_norm`, `drive_clip_mv`, baselines), the flight / locomotion thresholds if a body is in the loop |
| `execution` | device, backend flags (native kernels / graphs / event-driven / sparse), dt (LIF 0.5 ms, optic 1 ms, frame 10 ms), seeds (brain RNG, environment), batch size |
| `stimulus` | protocol name and every parameter: object angular size (deg, from the eye), speed (deg/s), sweep geometry, contrast (object vs background radiance in the four channels), background, pinned / free, program; matched control spec |
| `retina` | the sampled retinal drive actually presented: per-column radiance [UV, B, G, R] per frame at the probe's own sampling, plus the column -> body map (which photoreceptor bodies feed each of the 1,466 hex columns; 5,895 photoreceptors) |
| `units` | the unit-handling table (section 2) |
| `tables` | file list with row counts, columns, units and SHA-256 |

Tables (CSV, Parquet when > 1e6 rows):

- `readout_per_body.csv`: `bodyId`, `model_index`, `type`, `unit_kind` (spiking | graded | photoreceptor),
  `window_start_s`, `window_end_s`, `stimulus_value`, `control_value`, `stimulus_minus_control`, `unit` (Hz for
  spiking units; rate units [0-1] and mV of received drive for graded units), `n_trials`, `trial_sd`, `control_ids`
  (the matched control run ids). For LC11 and LC10a: **two rows per body**, `quantity` = `upstream_drive_mV` and
  `output_Hz`, populations never pooled.
- `contributions.csv` (when the decomposition tool ran): `body_pre`, `body_post` (decimal strings; or
  `pre_bodies` / `post_bodies` explicit lists for group rows), `value`, `kind` (`anatomical_count` |
  `effective_weight_mV` | `current` | `voltage` | `activity`), `sign_rule` (presynaptic NT_SIGN | receptor table
  tier), `gain_rule` (cap 60, same-type 0.1, fan-in scale, path / type gains, pair gains: the exact factors applied),
  `normalisation` (input_norm alpha / ref; optic out_norm), `reference_graph` (cache fingerprint), `window`.
- `sensitivity.csv` (lesion / hold results): `lesion_id`, the body list or rule, the check / readout, `delta`,
  `replicate_sd`.

## 2. Unit handling the export must declare (facts of the compiled model)

- **Node set**: every MaleCNS body with status `Traced` plus every photoreceptor body regardless of status
  (167,106 units). Bodies outside it are absent, not merged.
- **Graded units**: the 89,390 `ol_intrinsic` cells run as rate units (`optic.py`); their LIF rows are frozen and
  pruned from the spiking matrix (`Brain.freeze` / `prune`); exported as `unit_kind = graded` with rate [0-1] and
  received drive in mV.
- **Photoreceptors**: 5,895 bodies driven by the ray tracer, assigned to hex columns by their strongest hexed
  postsynaptic partner (94-99 % agreement among the top-3 partners); the column assignment is in the export.
- **Substituted signs**: `TYPE_NT_OVERRIDE` (TmY14 glutamate, Mi19 serotonin, aMe8 ACh, applied to those types'
  unknown-NT cells: 107 bodies) and `UNKNOWN_NT_OVERRIDE_REGEX` (27 antennal-lobe LN bodies -> GABA); the receptor
  model's per-edge sign changes (48,295 entries; listed by `body_pre`, `body_post` on request). Sign-0 (silenced)
  presynaptic bodies: 3,312 on the shipped cache (3,407 before `TYPE_NT_OVERRIDE`; unknown NT, dopamine, octopamine, serotonin).
- **Nothing is combined**: no cells are merged; per-connection synapse counts are capped at 60 in the effective
  weight (the raw count is kept in the cache and can be exported).

## 3. The first experiment (Neurome's request) -- delivered

Size tuning of **LC11 and LC10a, separately**: objects of ~4.5, 11, 20 and 30 degrees with matched blank controls,
upstream T2 / T3 responses, and the retinal sampling (a **replay** of the scene geometry at a pinned pose -- see
3b). A diagnostic, not a pass criterion, and -- after Neurome's intake -- **a scene baseline rather than a
controlled size-tuning assay**, because size and retinal position change together along this ladder.

**Export location** (local, git-ignored; `run_id` directories, never mutable files; each `manifest.json` carries the
commit `0d32fd6` + a per-file source fingerprint verified against the cluster run, the cache fingerprint
(sum|W| 121,460,584), the resolved `LIFParams` / `OpticParams`, device `cuda` / NVIDIA B200, seeds, stimulus geometry
and the unit table):

| size (nominal / from the eye) | ball radius | run directory (under `D:\Projects\flyverse\`) |
|---|---|---|
| 4.5 / 4.50 deg | 1.965 mm | `out/export/objsize-d045-20260913T014556Z-47ed1383/` |
| 11.4 / 11.42 deg | 4.991 mm | `out/export/objsize-d114-20260913T014606Z-7a4eadbd/` |
| 20 / 20.08 deg | 8.816 mm | `out/export/objsize-d200-20260913T014616Z-5ece62d6/` |
| 30 / 30.18 deg | 13.397 mm | `out/export/objsize-d300-20260913T014626Z-d5048f74/` |
| ladder summary | -- | `out/export/export-20260913T014635Z-db22e3ea/` (`size_tuning.csv` 288 rows, `runs.csv` 40 runs) |

Protocol: `scripts/probe_object_sweep.py::run` through `scripts/interp_export.py record` -- a black ball 5 cm ahead
of the eye sweeping 50 deg of azimuth, 12 s per run (3 s settle), 5 stimulus runs + 5 matched blank runs per size
(replicate unit = independent cluster jobs; batch `apobj-lad-287a43`, 40 jobs, 0 failed). Each per-size directory
holds `readout_per_body.csv` (26,482 rows: every LC11 / LC10a / LC10b / LPLC2 / LC4 body with `upstream_drive_mV` and
`output_Hz` as separate rows, plus T2 / T3 / Tm5Y / TmY21 / Mi4 graded rows), `retina_columns.csv` (1,466),
`retina_bodies.csv` (5,895 photoreceptor bodies -> column), `retina_radiance.csv` (1,759,200 rows: per-column
[UV, B, G, R] per frame -- **the ball arm only; the matched blank radiance was omitted from this delivery**, and the
table is a geometry replay, not an in-loop capture) and `per_type.csv`. Generator:
`scripts/interp_apply_object.py ladder` (batch script in `out/apply_object/ladder/`).

**What the drive statistic is.** Every "received drive" number below is the **maximum, over cells within one run, of
the per-cell time-mean of (object-arm drive - blank-arm drive)**; each run's maximum may come from a different cell.
The comparator is the *same* maximum statistic computed in independent blank/blank runs. It is therefore neither an
absolute membrane voltage nor the tuning curve of one identified neuron, and **comparing it against the 7 mV
threshold gap cannot establish that no spikes occurred** (corrected per Neurome's intake, section 3b).

**Result** (stimulus minus blank, verdict = |z| >= 3 on the blank-run SD and exact U p <= 0.05, 5 v 5 runs):

| type | statistic | 4.5 deg | 11.4 deg | 20 deg | 30 deg |
|---|---|---|---|---|---|
| LC11 | drive statistic, mV (object/blank vs blank/blank) | 0.0657 / 0.0784, z -0.9 | 0.0664 / 0.0476, +0.6 | 0.0685 / 0.0478, +1.6 | **0.1710 / 0.0539, +4.8** |
| LC11 | excess above the independent null (mV) | -0.0126 | +0.0188 | +0.0206 | **+0.1172** |
| LC10a | drive statistic, mV | z -0.1 | +0.7 | +2.8 | **+8.6** (0.208 vs 0.076) |
| LC10a | excess above the independent null (mV) | -0.0023 | +0.0137 | +0.0225 | **+0.1321** |
| LPLC2 | same (loom chain, for scale) | -0.5 | +2.6 | **+13.5** | **+98.7** (2.03 vs 0.12) |
| LC11 / LC10a | population firing-rate comparisons | null | null | null | null (**16/16 null**: no *detected* object effect; not zero firing) |
| T2 / T3 / Tm5Y / TmY21 | `diff_signed_best_cell` | null | null | **result** (z +7.5 / +11.9 / +6.5 / +3.4) | **result** (+56.6 / +36.5 / +27.5 / +17.5) |
| T2 / T3 / Tm5Y / TmY21 | `diff_abs_best_cell_mean` | null | null | null | **result** (z +10.2 / +4.4 / +38.5 / +8.9) |
| T2 / T3 / Tm5Y / TmY21 | population `diff_signed_mean` | null | null | null | null |

Reading, as corrected with Neurome (section 3b):

* **The two LC types have different biological targets and only LC11's is a small-object target.** LC11's published
  preference is a vertical extent of **8.8 deg** with a width optimum near **4.4 deg** (Keles & Frye 2017, Fig 3D/E,
  calcium); LC10a's is **15-30 deg** in width and height (Schretter et al. 2024, Nature, Fig 3a). A 30-deg LC10a
  drive response is therefore **not** inverted tuning, and the LC11 optimum must not be imposed on LC10a. What
  remains a biological concern is LC11's lack of a detected small-object response.
* **The cells are not literally silent.** Exported object-arm firing is nonzero in both populations: 6/143 LC11 and
  19/275 LC10a bodies have nonzero mean firing at 4.5 deg (LC11 body `24647` 0.1833 Hz object vs 0.1167 Hz on its
  paired blank at 11.4 deg; LC10a body `69463` 0.8333 vs 0.7500 Hz at 4.5 deg). Those are examples of firing, not
  individually established object responses. **The supported statement is "no detected object effect in the exported
  population firing-rate comparisons (16/16 null)"**, not "LC11 / LC10a never spike".
* **Size and retinal position change together, so this is not yet a controlled size-tuning assay.** Ball centre
  elevation rises **0.88 / 4.34 / 8.66 / 13.71 deg** across the ladder and the mean per-frame count of columns dimmed
  > 50 % rises **0.14 / 2.26 / 7.64 / 19.73**; lateral translation at constant world speed also changes angular speed
  and apparent diameter along each sweep. The retinal contrast footprint may contribute to the drive ordering and its
  causal contribution has not been isolated. **The ladder is a scene baseline, not a size-tuning measurement.**
* **The retina tables are a geometry REPLAY at a pinned pose, not an in-loop capture of the input the brain received**
  (the manifest says so), and the per-size tables **omit the matched blank radiance** -- Neurome could reconstruct the
  footprint only because the referenced blank NPZs exist in this checkout. Both are fixed in the next export (3b).
* **Keep the 30-deg statistics exploratory.** Both 30-deg comparisons have complete rank separation, unadjusted
  p = 0.0079365, z 4.85 / 8.65. Treating the eight LC drive comparisons as one family, **Holm-adjusted p = 0.0635** --
  a post-hoc sensitivity calculation, not a new pass rule, and enough reason not to call them established. The exact
  U also meets ties in **25 of the 288** summary statistics; reproducing it does not validate its use with tied data.
  Cells are not independent replicates; runs are the replicate unit.
* **The two upstream statistics are distinct and must not be collapsed.** At 20 deg T2 / T3 / Tm5Y / TmY21 reach the
  `result` criterion on `diff_signed_best_cell` while their `diff_abs_best_cell_mean` and population
  `diff_signed_mean` stay `null` -- different 20-deg verdicts, so "the stage is active / silent" is not a statement
  this export supports.

The localization behind the model-side loss is in `docs/audits/deficit_object.md`: at T3 the ON and
OFF carriers converge through excitatory exact-tier synapses whose figures for a sweeping small object have opposite
sign (holding either class restores T3; a size-matched control hold does not), the residual is then scrambled by the
spiking feedback into the rate lobe, and what survives is pooled away at LC11 / LC10a (94 / 32 columns per cell under
`out_norm l1`). Caveats from the Opus skeptic: the 'lost at T3 / Tm5Y' verdict at 11.4 deg is a threshold call
(Tm5Y reads z +0.9 / +2.2 / +2.6 / +3.7 across four batches of the same model); the GPU pipeline is not seed-reproducible,
so the replicate unit is runs and every z is +-1.5; LPLC2's 11.4 deg verdict flips between batches.

What Neurome could decide, in order of leverage on this pathway: (1) the receptor tiers of the unprofiled types --
Tm5Y, TmY21, TmY13 and LC11 run on the presynaptic-sign fallback for 100 % of their input (`contributions.csv`
`sign_rule` column); (2) whether the Mi1 / Tm3 (ON) and Tm1 / Tm4 (OFF) inputs to T3 are, in the animal, rectified
per class before summation (the model sums them linearly; a per-input-class rectification arm is the next
physiology-parameter test, `deficit_object.md` section 6); (3) per-body LC11 / LC10a recordings at these four sizes
against `readout_per_body.csv` (`bodyId` decimal strings; `n_trials` 5). **Field defect in this delivery:**
`control_ids` names the independent blank/blank runs used for `null_mean`, while `control_value` is the mean of
**arm b of the stimulus recording** -- two different references under one identifier (for LC11 `24647` at 11.4 deg
the paired blank is 0.1167 Hz while the independent null arms average 0.1333 / 0.0500 Hz). The underlying values
reproduce; the identifiers are split in the next export (3b).

**Re-export under revision 2 of the exporter (2026-09-13, same recordings, numbers bit-identical):**
`out/export/objsize-d045-20260913T065428Z-07e3f4dd/`, `objsize-d114-20260913T065428Z-d1e4eae6/`,
`objsize-d200-20260913T065428Z-23a951ae/`, `objsize-d300-20260913T065428Z-05fd4a45/`, summary
`export-20260913T065509Z-5dc2aa41/` (schema `flyverse.neurome.export/2`): `paired_control_ids` / `null_reference_ids`
(`control_ids` kept as a deprecated alias of the latter for one revision), `manifest.statistic_definitions`,
`retina.mode = geometry_replay` stated with the pinned pose, the matched blank radiance as `retina_radiance_blank`,
`retina_object_track` (per-frame azimuth / elevation / diameter / dimmed columns), tie-aware p with `p_method`, and
`family = lc_drive` with `p_holm` (30 deg: 0.0635, as Neurome computed). Generator: `scripts/interp_export.py ladder`;
record `docs/audits/interp_export.md` section 14.

## 3b. Neurome intake (2026-09-13) -- corrections accepted and next export

Neurome consumed the four `objsize-d*` exports and the ladder summary read-only and replied. Both documents:
`D:\Projects\neurome\docs\flyverse-size-tuning-reply.md` (the reply) and
`D:\Projects\neurome\reports\flyverse-size-tuning-intake.md` (the full intake, with the checked hashes, the 3,344-row
anatomy/readout join, the recomputed statistics and the literature), plus the machine-readable
`D:\Projects\neurome\reports\flyverse-size-tuning-biological-constraints.json`. **Their corrections are accepted**
and section 3 above is corrected in place. What they verified, for the record: 26 table hashes/schemas, 105,928
body-keyed readout rows, all 288 summary statistics, the compiled-matrix fingerprint (`sum_abs_W` 121,460,584,
167,106 neurons, 25,578,600 entries), an anatomy/readout join for all 143 LC11 + 275 LC10a bodies, and reductions of
the 40 recording NPZs (20 object/blank + 20 independent blank/blank pairs). The exported numbers reproduce.

**The corrections, in their order of consequence:**

1. **LC10a has its own size target (15-30 deg, Schretter et al. 2024, Nature, Fig 3a + Ext. Data Fig 2).** A 30-deg
   LC10a drive response is not an inversion of biological tuning. Only **LC11** (vertical extent 8.8 deg, width
   optimum ~4.4 deg, Keles & Frye 2017, Fig 3D/E, calcium imaging of rectangles) is the small-object concern.
2. **The cells are not literally silent** (6/143 LC11 and 19/275 LC10a bodies fire at 4.5 deg). The supported claim is
   "no detected object effect in the exported population firing-rate comparisons, 16/16 null". The headline drive
   number is a within-run maximum over cells of a time-mean difference, compared with the same statistic in
   blank/blank runs -- not an absolute voltage, and not a basis for asserting the absence of spikes.
3. **Size and retinal position co-vary** (centre elevation 0.88 / 4.34 / 8.66 / 13.71 deg; columns dimmed > 50 %:
   0.14 / 2.26 / 7.64 / 19.73 per frame), the retina tables are a **replay of scene geometry at a pinned pose**, and
   the matched blank radiance was omitted. The ladder is a useful **scene baseline**, not a size-tuning assay.
4. **Control-identifier defect**: `control_ids` names the independent blank/blank runs while `control_value` comes
   from **arm b of the stimulus recording**. Paired-control and independent-null references must travel as separate
   identifiers.
5. **Statistics**: keep the 30-deg results exploratory (Holm across the eight LC drive comparisons gives p 0.0635);
   the exact U meets ties in 25/288 rows; `diff_signed_best_cell` and `diff_abs_best_cell_mean` stay distinct
   statistics (they give different 20-deg verdicts); cells are not replicates, runs are.
6. **Literature constraints, not parameters** (also as ledger rows, `flyverse/data/expected_responses.csv`, op
   `report`): Keles et al. 2020, Cell Reports -- LC11 expresses **Rdl** and **nAChR alpha1 / alpha6 / alpha7**;
   LC11-specific Rdl disruption reduces small-dark-object responses by **~40 %** *without* releasing bar/grating
   responses (unlike bath pharmacology); **T2 / T3 respond to both ON and OFF transitions**; **T3 -> LC11 is
   functionally excitatory**. These constrain *output behaviour*; they are not per-synapse rectification evidence,
   give no gains and do not resolve an LC11 glutamate sign. Tanaka & Clark 2020, Current Biology -- an LC11 model
   pooling tightly size-tuned, fast-adapting inputs: a **competing hypothesis** (adaptation + spatial suppression),
   not parameters to copy.
7. **No graph correction follows from this ladder.** The simulation's failure to respond does not establish a missing
   connection, and nothing in the intake justifies merging a fragment.

**Field change for the next export** -- *implemented in the `export.py` revision; see `docs/audits/interp_export.md`*:
`readout_per_body.csv` carries **`paired_control_ids`** (the record + arm the `control_value` actually came from) and
**`null_reference_ids`** (the independent blank/blank runs behind `null_mean` / `null_sd` / `z_vs_null` / `verdict`)
as separate columns, with `control_ids` kept for one revision as a deprecated alias of `null_reference_ids`; the
per-size export adds the matched **`retina_radiance_blank`** table alongside `retina_radiance`; and `manifest.json`
gains **`retina.mode`** (`in_loop_capture` = radiance recorded during the rollout, `geometry_replay` = the scene
re-rendered at a pinned pose), so a replay is never silently labelled as recorded input.

**The agreed next experiment (we run it; Neurome supplies constraints and body-keyed anatomy).** Numbered as agreed:

1. **A matched visual assay.** Object centre elevation, angular trajectory, angular speed, contrast and background
   held constant across sizes. **First a fixed-centre square ladder** (one changing variable, its more limited
   interpretation stated up front), **then separate height and width ladders** to match the published rectangle
   experiments. An independent **per-body receptive-field localizer** fixes each body's measurement region before the
   size runs. **Blank and object radiance recorded in-loop** during execution (`retina.mode = in_loop_capture`,
   `retina_radiance` + `retina_radiance_blank`), with the
   present ladder kept as the scene baseline. **Per-body time courses and their uncertainty are reported before any
   population maximum.** Predeclared, type-specific primary statistics: **LC11 = ?**, **LC10a = ?** (each the biological
   target for that type: LC11 small-object, LC10a 15-30 deg), with the T2 / T3 upstream readouts kept as the two
   distinct statistics (`diff_signed_best_cell` vs population mean), never collapsed. **>= 5 runs per arm, all arms in
   one submission**, runs as the replicate unit, Holm within the predeclared family.
2. **A physiological model comparison on fixed anatomy** -- no graph change. Three arms: (a) the existing linear
   **sum**; (b) **rectification per presynaptic stream before summation**, synaptic signs preserved (rectification
   must never turn inhibitory input into excitation, and a rectifier after the final sum cannot recover components
   that already cancelled); (c) **adaptation + spatial suppression** (the Tanaka & Clark hypothesis, implemented as
   our own arm, not copied parameters). Controls: **`gain_fb = 0`** (the deterministic lobe) and the existing
   **feedback-hold** arms, to separate recurrent variability from the candidate computation. Specificity battery, so
   an apparent rescue can also fail: **bright and dark objects, isolated ON and OFF flashes, stationary flicker, bars
   and gratings**. Global type-mean cancellation alone does not locate a cellular mechanism.
3. **Transfer to the LC output.** Keep the arm's parameter set **fixed** across sizes and across **held-out stimuli**,
   test downstream pooling / output dynamics, and score the whole thing on the existing benchmark suite
   (`docs/BENCHMARK_BATTERY.md`, `flyverse/data/expected_responses.csv`). A calcium observation model is an explicit,
   declared hypothesis, not a free scale factor. **LC10a state gating is a separate follow-up**, opened only once
   LC10a's matched visual response can be measured.

## 3c. Round 2 (2026-09-13) -- the matched assay delivered, the model comparison run

**Export location** (schema `flyverse.neurome.export/2`, revision 2; thirteen run directories,
194 tables, 42,750,310 rows, `export.verify()` `problems: none` in every one; index
`out/export/objr2_index.json`, flat SHA-256 list `out/export/objr2_tables.csv`). These are the
**2026-09-14 re-emit** from the corrected Result; the identically-shaped 2026-09-13 directories
(`objr2-*-20260913T2329*Z-*`, ladder `objr2-ladder-20260913T232912Z-72041020`) remain on disk as
the superseded delivery:

| rung | lobe `ship` | lobe `fb0` (`gain_fb = 0`, deterministic control) |
|---|---|---|
| 4.5 deg | `out/export/objr2-ship-d045-20260914T024747Z-5f117772/` | `objr2-fb0-d045-20260914T024629Z-8f991830/` |
| 8.8 deg | `objr2-ship-d088-20260914T024801Z-e6b2d4fb/` | `objr2-fb0-d088-20260914T024642Z-c1a2c537/` |
| 11 deg | `objr2-ship-d110-20260914T024814Z-28adcffb/` | `objr2-fb0-d110-20260914T024655Z-7469fef6/` |
| 15 deg | `objr2-ship-d150-20260914T024827Z-6d201251/` | `objr2-fb0-d150-20260914T024708Z-2cb077e8/` |
| 20 deg | `objr2-ship-d200-20260914T024840Z-3d77182b/` | `objr2-fb0-d200-20260914T024722Z-67a7460e/` |
| 30 deg | `objr2-ship-d300-20260914T024853Z-820b11d5/` | `objr2-fb0-d300-20260914T024735Z-7e9c07f0/` |
| ladder summary | `objr2-ladder-20260914T024906Z-035363c0/` (both lobes, all six rungs, the predeclared Holm columns, the old ladder beside it) | |

One provenance cost of the re-emit, stated up front: the 2026-09-14 directories were written from
a working tree that had drifted since the batch ran, so their `flyverse_commit` reads `unknown`
with `source_match.verified false` (19 of 29 loaded source files identical, 30 of 43 glob). The
2026-09-13 directories carry commit `653179b4...` **verified 29/29 loaded and 43/43 glob**. The
tables are otherwise identical apart from the four preference cells named at the end of this
section; if you need the content-pinned provenance, read it from the 2026-09-13 pair.

**Experiment 1 (matched visual assay) -- delivered, with two items short of the request.**
Elevation, distance, angular diameter and angular speed are held per frame (realised deviation
0.0 deg / 1.4e-14 deg / 40.000 deg/s), against the old ladder's 0.88-13.71 deg elevation and
2.4x speed change. Object AND blank radiance are captured **in the loop** on both arms
(`retina.mode = in_loop_capture`); recomputing the footprint from the two Parquet tables alone
reproduces the probe's numbers to 2.2e-16. Not delivered: (a) **effective retinal contrast is not
matched** across the sphere ladder -- the median per-frame extreme relative luminance change is
-0.504 / -0.869 / -0.876 / -0.899 / -0.927 / -0.948 across 4.5 -> 30 deg, because a 4.5-deg ball
only partially fills a 4.6-deg column; the contrast-matched families are the synthetic rectangle
ladders, which are recorded on the cluster but not yet fetched or exported. (b) **The per-body RF
localizer does not localize LC11.** At the predeclared `z_min` 5 not one of 143 LC11 bodies is
fitted in either lobe's 15-deg three-pass map (LC10a 13 of 275 = 4.7 %), so **405 of 418 LC
windows are the anatomical fallback** (`window_source = anat`; `rf_map` says per body which clause
it took). Please do not read these as measured receptive fields. The effective per-rung body
counts are in `sphere_per_run.n_bodies_windowed`: LC11 55 / 99 / 99 / 99 / 103 / 103 of 143 and
LC10a 73 / 76 / 76 / 76 / 79 / 90 of 275. Ground truth for the localizer is Mi1, which it places
at its hex-annotated column to a median 2.2-3.5 deg with a 0-0.1 % false-fit rate on the blank arm.

**Answers, to the predeclared rules** (`out/objr2/predeclared.json`, stamped 2026-09-13T21:45:39Z,
before the 21:46:06Z submission; 6 runs per arm, one submission; 12 members per LC type; Holm
inside the family; a preference is CALLED only on `result` AND `p_holm <= 0.05`):

* **LC11: 12 of 12 members `null`** (smallest p 0.180, smallest `p_holm` 1.000). Excess over null
  +0.103 / +0.042 / +0.051 / +0.009 / +0.015 / +0.005 mV across 4.5 -> 30 deg -- largest at the
  smallest rung, the direction Keles & Frye 2017 predicts -- but Spearman rho -0.216 (p_perm
  0.206) and the small-minus-large contrast +0.029 mV (p_perm 0.277). **No size preference called.**
* **LC10a: 11 `null`, one `result`** -- 30 deg drive median +0.0155 +- 0.0132 mV against a null of
  -0.0046 +- 0.0060, z +3.37, p 0.00866 -- **which does not survive Holm (`p_holm` 0.1039)**. The
  member sits inside the 15-30 deg range you gave from Schretter et al. 2024; it is reported as
  exploratory. **No size preference called.**
* `spikes_median` is exactly 0.000 +- 0.000 in every arm of every rung, object and null alike:
  the spike half of each family is uninformative, not negative. The populations are not silent.
* **Upstream** (exploratory, surviving its own family's Holm): `diff_signed_best_cell` reaches
  `result` at T2 and Tm5Y from 11 deg and at T3 and TmY21 from 20 deg, rising monotonically with
  size (T2 +0.0141 -> +0.0442 against +0.0081 +- 0.0016, z +3.8 -> +23.0). The population
  `diff_signed_mean` stays `null` at every rung for every type, and `diff_abs_best_cell_mean`
  gives a third verdict: the three statistics are shipped as separate rows and are never collapsed.
  On the old headline max-over-cells statistic the LC types move only at 30 deg (LC11 +0.159 vs a
  null of +0.053, z +10.6) -- the large end, and exactly the statistic your intake warned about.

**Experiment 2 (fixed-anatomy model comparison) -- run, not exported.** Eight arms on the same
matched ladder (5 runs each), a seven-stimulus specificity battery (4 runs each) and three
benchmark draws per arm, all in one submission. **No mechanism class passes.** Per-stream
rectification is the only arm that carries the T3 and T2 carrier figure (6.0 / 10.5 / 9.0x base at
4.5 / 8.8 / 11 deg), and it **releases bar, grating and flicker responses at LC11** -- the grating
0.611 -> 3.202 mV and 0.042 -> 2.92 Hz, the flicker 0.145 -> 2.994 mV and 4.71 -> 10.67 Hz -- which
is the Keles 2020 constraint failing. Its figure also grows with object size (Spearman rho +0.937
on the LC11 max-over-cells excess), lives entirely in the extremum over cells (the per-body
RF-windowed T3/T2 medians read `null` against base in all 48 rows), and comes with an
operating-point shift (T3 blank-arm mean deviation +0.00002 -> +0.0284; LC11 blank-arm drive
0.093 -> 0.348 mV) that the spatially uniform flicker reproduces. Adaptation (100 and 300 ms) is
inert. Spatial suppression produces no figure and costs the escape benchmark
(`loom_escape.GF_peak_hz` 50.0 -> 31.2 Hz, escapes 1.0 -> 0.33). **LC11 output follows in no arm.**
**Nothing is adopted; every hook stays `None` by default and is bit-identical off.** Two caveats
we owe you: (i) the scheduler put each arm on the least-loaded box, so `rectify` and `suppress`
ran on a different GPU model from `base` -- the rectification magnitude is confirmed on base's own
GPU model by the two combination arms, and a same-device re-run of `base` is owed; (ii) the flash
stimulus is a periodic square, so the `flashon` / `flashoff` rows of the batch separate a
**bright** flash from a **dark** one and pool both transition polarities inside each.

**The ON/OFF split you asked for now exists**, re-derived on CPU from the 2,048 stored specificity
recordings (`out/interp/objr2c/spec_transitions.json`; 0.3 s window after each ON and each OFF
edge, truncated at the next edge, the blank/blank arm scored on each family's own edge schedule as
the floor). **Both transitions survive in every arm**: no ON or OFF window at T2 or T3, in either
flash family, falls below 0.56x base's, so the "T2 and T3 respond to both ON and OFF transitions"
constraint holds for all eight arms. What rectification changes is the asymmetry -- base's two
windows sit within 6-30 % of each other (T3 bright 0.063 / 0.057 mV; T3 dark 0.083 / 0.064), while
`rectify` tilts the dark flash's T3 toward ON by 3.1x (0.142 / 0.045) and its T2 toward OFF by
1.5x, and `rect_supp` does the same at T3 (2.1x); adaptation and spatial suppression leave the
ratios near base's. Read these as 4-run magnitudes of a max-over-cells statistic with no
null-referenced verdict, and note the accompanying limit: measured against the matched blank/blank
floor on the same edge schedule, almost every 0.3 s window sits at its own null (0.68-1.71x), and
only `rectify` and `rect_supp` on the dark flash's ON window clear it (2.36x and 2.38x). A 0.3 s
window is 30 frames, and a maximum over 1,940 cells has a high noise floor there.

**Field changes in this export, against revision 2's list:** `paired_control_ids` (the record and
arm the `control_value` came from, `#arm_b` suffixed) and `null_reference_ids` (the independent
blank/blank runs), disjoint in every directory, with `control_ids` kept for one revision as a
deprecated alias of the latter -- the labelling defect you found is closed. `retina_radiance_blank`
ships beside `retina_radiance` (1,759,200 rows each, identical schema). `manifest.retina.mode` is
`in_loop_capture` with the probe's own sampling sentence; nothing in this export is a replay.
`manifest.statistic_definitions` (13 entries) plus a per-row `statistic_definition` column.
`retina_object_track` carries azimuth / centre elevation / angular diameter / distance / angular
speed **per frame as measured in the loop**. Two rank tests per row (`p`, the predeclared tie-aware
exact permutation U over all C(12,6) assignments, and `p_mannwhitney` with `p_method` in
{exact, asymptotic_tie_corrected}; they agree to 1.1e-16 on the 129 untied rows of 288, and the
159 tied rows read `asymptotic_tie_corrected`, never a false `exact`). Two multiplicity corrections
that do not overwrite each other: `family` / `p_holm` (the predeclared family) and
`analysis_family` / `analysis_p_holm` (the analysis' own), with `role` per row.

**One defect found on review, now closed.** The 2026-09-13 ladder summary's `preference.csv`
carried `spearman_p_perm = 5.0e-05` with an empty `spearman_rho` on four rows (both lobes x both
LC types, statistic `spikes_median`), **all four `role = primary`**. The Spearman is undefined
there because every spike median is exactly 0.0; the correct value is NaN. The permutation code
was fixed, the analysis re-run on the already-fetched batch (CPU) and the ladder re-exported: in
`objr2-ladder-20260914T024906Z-035363c0/preference.csv` those four rows now carry **empty**
`spearman_rho` and `spearman_p_perm`, and no other number in the delivery changed. If you are
holding the 2026-09-13 ladder summary, treat those four cells as missing, not as p = 5e-05, and
prefer the 2026-09-14 directory.

**What we ask Neurome for next, in order of leverage:**
1. **Receptor tiers for Tm5Y, TmY21, TmY13 and LC11.** All four still run on the presynaptic-sign
   fallback for 100 % of their input, and all four are load-bearing in this pathway (T3 and T2 are
   LC11's two largest inputs at 69,205 and 36,917 predicted pairs; Tm5Y and TmY21 are LC10a's at
   14,221 and 7,406). This is the single change that would replace a modelling assumption with data.
2. **Per-body LC11 / LC10a recordings at these six angular sizes** (4.5 / 8.8 / 11 / 15 / 20 /
   30 deg) under matched geometry, keyed by `bodyId` decimal string against `readout_per_body.csv`
   (`n_trials` 6, both `upstream_drive_mV` and `output_Hz` rows per body, plus the RF-windowed
   columns). We can compare shape under an explicit observation model; we cannot yet compare
   absolute Hz or mV.
3. **The Keles 2020 Rdl constraint as a test, not a citation.** LC11-specific Rdl disruption
   reduced small-dark-object responses by ~40 % *without* releasing bar or grating responses. That
   is a two-sided criterion our rectification arm fails on the second half. If you can supply the
   quantitative form -- the effect size on the small-object response and the bound on the
   bar/grating release -- we will add it to `flyverse/data/expected_responses.csv` as a scored row
   and run every future arm against it, instead of the magnitude rule we used this round.
4. A note on what would help most on the measurement side: our LC11 localizer finds no
   stimulus-driven receptive field at any square size we ran (4.5, 8.8, 15 deg). If the animal's
   LC11 receptive fields are known per cell type at a comparable resolution, that would let us
   replace 143 anatomical boxes with measured regions.

## 3d. Round 3 (2026-09-14) -- the arm/device confound closed, the rectangle ladders run, the localizer answered

**Export location** (schema `flyverse.neurome.export/2`, revision 2; **95 run directories, 1,712 tables,
`problems` empty in every one and in every source-Result round trip**). Start at the combined index
`out/export/objr3_index.json`; it names its four component indices and, separately, the replication evidence:

| index | contents |
|---|---|
| `out/export/objr3_r2compare_index.json` | 41 directories -- the round-2 fixed-anatomy comparison arms (section 3c's "run, not exported" item, now exported), with the ON/OFF transition and specificity/benchmark sidecars |
| `out/export/objr3_samedevice_index.json` | 16 directories -- the same-device sphere batch (base / rectify / suppress on one GPU model) |
| `out/export/objr3_rectangles_index.json` | 36 directories -- the height and width rectangle ladders, with the per-rung effective-contrast and window-coverage tables |
| `out/export/objr3_rfmap_index.json` | 2 directories -- the stimulus-driven RF maps, both lobes |
| `out/export/objr3_house_results/index.json` | **linked from `objr3_index.json` under `replication_evidence`** (sha256 `466c4fb2...`, `n_results` 4): the four **native fresh-seed B200 replication Results** -- house same-device, house rectangles, and the two house RF maps -- kept as native Results rather than folded into the 95 original directories |

`objr3_tables.json` lists every one of the 1,712 tables; `objr3_skeptic.json` records the independent delivery
checks. Every one of the 91 sphere / rectangle rungs carries **all 418 LC bodies at all 1,200 frames** with run
means and sample SDs. Per-section reference devices are explicit fields (round-2 sphere base B200; specificity and
benchmark base H200), and the literal verdict text `null` survives the CSV round trip -- it is a scientific
verdict, not a missing value.

**The device caveat, restated.** "Same-device" in round 3 means **one GPU model per batch** -- NVIDIA H200 for the
original batches, NVIDIA B200 for the replications -- with the reference arm and its treatments in the same
submission and the same block on the same physical host. It does not mean the two batches ran on the same GPU
model as each other, and no causal GPU-model attribution is claimed: the two round-3 sphere batches have
**identical source fingerprints** (all 44 fingerprinted files match) and differ in GPU model, box, seeds and
execution history together.

**Round-2 question 1 (a same-device `base` re-run, to de-confound arm from GPU model) -- answered, and the answer
is PARTIAL on the second box.** Rectify exceeds base at every small T3 / T2 maximum in both batches (H200 ratios
T3 6.281 / 9.528 / 13.063 and T2 2.356 / 3.369 / 3.068 at 4.5 / 8.8 / 11 deg, Holm p .023810 on all six rows;
B200 T3 6.131 / 9.028 / 12.598, T2 3.522 / 3.421 / 3.225). The predeclared reproduction rule requires T3 **and**
T2 at all three small rungs: H200 `REPRODUCES`; B200 is **`PARTIAL`**, because rectify T2 at 4.5 deg reads
z **+2.66825** against its own blank/blank null and is therefore `null` under `common.compare`'s z >= 3 gate even
at Holm p .023810. The separation at that rung is complete (all five rectify runs above all five null runs,
U 25/25, p at the exact 5 v 5 floor .0079365); the gate is missed because a single null run inflates the null SD
to .005302 against the H200 null's .001147. Three corrections you should carry with that verdict:

* The **large-rung companion effect is a three-batch effect, not a device-specific one**. Rectify's per-body
  RF-windowed T3 companion exceeds base at 20 deg (.006050 +/- .000709 vs .004445 +/- .000414, z +3.88) and 30 deg
  (.007377 +/- .000775 vs .004259 +/- .000400, z +7.80), and its T2 companion at 30 deg (z +3.56), all
  Holm .039683, on **H200 as well as B200** -- and round 2 already carried the T3-at-30 row (z +9.795, ratio
  1.827). `companion_all_null_all_rungs` is false in both round-3 Results. The round-2 statement that the
  rectification figure lives entirely in the extremum holds **only at the three small rungs**.
* **The reference arm is not identical across the two batches.** `base`'s own T2 maximum versus its own
  blank/blank null crosses the effect gate on the **B200** (z +6.12652 at 8.8 deg, +9.29988 at 11 deg, both
  Holm .023810, `result`) and not on the **H200** (z +0.35 / +1.02 / +1.96, `null` at every small rung). The
  shipped model's own small-object figure separates from blank on one GPU model and not the other.
* **Suppress reproduces as a null**, and there is no LC11 rescue in either batch (`both_null` on T3, T2 and LC11;
  `lc11_follows` false in every arm of both batches). Nothing is adopted and no default moved; rectify and
  suppress remain hand-set, opt-in control arms.

**Round-2 question 2 (the contrast-matched rectangle ladders, height and width separately, as Keles & Frye did) --
run in both batches; 40 of 40 primary verdicts `null`, and the contrast is matched only at width >= 8.8 deg.**
Each batch is 216 object runs (9 shapes x 2 contrasts x 6 seeds x 2 lobes) plus 108 blank/blank null runs
(9 shapes x 6 seeds x 2 lobes) = 324 paired runs. Nulls are blank/blank and carry no contrast, so a shape's six
nulls serve **both** its dark and its bright family -- those two families are not independent of each other,
which matters if you re-test across them. **No LC11 or LC10a animal-shape preference is called in either batch.**
Three things to read beside the nulls:

* **Object contrast is fixed at Weber +/- .995; effective retinal contrast is not.** Fractional coverage of the
  4.5-deg column acceptance caps the per-column change wherever a rectangle is narrower than a column: peak
  per-column effective contrast reaches +/- .995 **only at width >= 8.8 deg -- three of the nine rectangles, all
  of them in the width ladder.** The whole height ladder runs at width 4.4 and is capped at max coverage .88530 /
  extreme -.65262, so its 8.8 / 15 / 30 rungs are matched to each other but not to the width ladder, and its 2.2
  and 4.4 rungs (.42649 / -.22826 and .65590 / -.53849) are matched to nothing. At the smallest rungs the stimulus
  is also intermittent: 40 % of frames at 4.4 x 2.2 and 25 % at 2.2 x 8.8 change no column by more than 5 %.
* **The one sub-.05 row points both ways.** B200 dark LC10a at width 15 deg reads diff **+.055055 mV**,
  z **+2.17788**, Holm p .021645 -- `null` under the z >= 3 gate -- and the H200 batch has the **opposite sign**
  at the same type, ladder, contrast and rung (-.015022 mV, z -.45490, Holm 1). We report it as open in both
  directions, not as weak positive evidence.
* **The upstream size-monotonicity call is a two-directional knife-edge, so no conclusion leaves the round.** Of
  the 16 declared (ladder x contrast x type) joint tests, exactly **one passes in each batch and it is a different
  one** (B200 `hlad:dark:T2`, H200 `hlad:bright:Tm5Y`); the two batches agree in **sign** on both statistics, and
  both calls flip under a different permutation seed (H200 median p .049248 -> .053047; B200 `hlad:bright:T2`
  .050797 -> .047048). "Does not replicate" would be the wrong word for a threshold crossing.

**Round-2 question 3 (a stimulus-driven LC11 / LC10a localizer, after 0 of 143 fits at `z_min` 5) -- run, and
neither LC population localizes under this probe, on either lobe, in either batch.** LC11 has **zero fits at the
fixed z = 5** on both lobes in both batches. In the B200 `fb0` map the blank-selected threshold is z* = 4, giving
4 of 143 LC11 fits against 1 of 143 blank fits, which still **fails both the population-coverage and the
spatial-enrichment criteria**; shipped B200 LC10a shows free-peak enrichment above 2x chance but insufficient
fitted coverage. Four limits define what this negative is:

* **The positive control is on a different quantity.** Mi1 pooled coverage is ~53 % in both batches, but Mi1, T2,
  T3, Tm5Y and TmY21 are all optic-lobe rate units fitted on **`optic_dr`**, while the only `drive_mv` rows in the
  map are the 418 LC bodies that produced the negative. The control shows the grid, dwell, pooling and fitter work
  on rates; it does not show that a `drive_mv` receptive field of the same strength would have been detected.
* **The probe was 4.5 deg and static**, not the 2-4 deg we asked for and not the moving square Keles & Frye used
  for their RF measurement; the presented grid was 1,466 of 1,787 reachable nodes (82 % of the eye), with the
  per-cell box restriction applied at fit time only.
* **This is not a statement about firing.** The fitted quantity is received drive; the LC populations emit
  essentially no spikes in this protocol (137 of 143 LC11 bodies emit exactly zero over the 1,466.5 s recording).
  Please do not read the result as "these cells cannot be localized by any stimulus".
* **The round's ladder windows were not retrofitted.** All 143 LC11 windows remain the anatomical fallback, and
  using any new RF map in inference would need a separately stamped analysis.

**What we ask Neurome for next, in order of leverage** (items 1 and 3 of section 3c stand unchanged; these are the
round-3 additions):

1. **Receptor tiers for LC11, T2 and T3** (with Tm5Y / TmY21 / TmY13 from section 3c). This is now the top of the
   list rather than one item on it: T3 and T2 are LC11's two largest inputs (69,205 and 36,917 predicted pairs)
   and all of them still run on the presynaptic-sign fallback for 100 % of their input. Round 3 tested the
   physiological alternatives that do not need data -- per-stream rectification, adaptation, spatial suppression,
   a per-transmitter unitary scale -- and none of them passes; the remaining lever on this pathway is the receptor
   tier, not another hand-set mechanism.
2. **The contrast at which LC11 should be probed.** Keles & Frye's Figure 3B shows that **maximum contrast is not
   LC11's optimum**: reducing OFF-object Weber contrast from 100 % to 30 % "nearly doubled the amplitude of the
   calcium response". Both round-3 ladders ran at |Weber| .995, i.e. at the contrast the animal responds to
   *least* strongly. If you can give the response-versus-contrast curve in quantitative form, the next ladder
   runs at the animal's optimum instead of at ours, and the caveat stops travelling with every null.
3. **Whether you need localizer frame chronology.** The long RF recording retained only node / role-window means,
   so the delivery carries **node response tables, explicitly labelled as such** -- we did not reconstruct a
   per-cell frame series from means. Per-cell 10 ms traces over the RF grid are obtainable, but only from a **new
   recording**; tell us if the analysis needs them before we cost another long run.
4. **The moving-probe RF assay, if you want the localizer negative closed.** We will declare it separately, with
   blank controls and a smaller, level-matched probe, before making any absence claim under other stimuli. Whether
   that is worth the runs, and whether more LC10a runs under a new declaration are worth them, are the two open
   protocol decisions on our side.

## 4. Neurome evidence bundles

Accepted initially as **read-only observations**: predicted anatomical counts, source annotations, reconstruction
candidates, evidence references. Unreviewed candidates never modify the simulator graph; the compiler
(`flyverse/connectome.py`) reads only the released MaleCNS tables, and every model-side change so far went through
the audit-and-verify procedure of `docs/audits/`. When anatomy is independently supported, a revision format can
feed the compiler into an isolated cache (`connectome.load(cache_dir=<scratch>, rebuild=True, ...)` is the
existing pattern -- scratch caches per experiment, the shared cache untouched), scored against the benchmark
suite before adoption.

## 5. Open items for Neurome

- The schema and a checksummed example bundle (their side): received (`small_object_v2`); our export uses the `evidence-bundle-v1` field
  names where they overlap (`body_pre` / `body_post`, `synaptic_pair_count`).
- Delivered: the size-tuning export (section 3), **consumed and replied to** (section 3b). Field changes needed on
  the compiler side: none. Field changes needed on the **export** side: `paired_control_ids` / `null_reference_ids`
  split, the matched `retina_radiance_blank` table, and `manifest.json` `retina.mode`
  (`in_loop_capture` | `geometry_replay`) -- implemented in the export.py revision, `docs/audits/interp_export.md`.
- **Done in round 2** (section 3c): the **matched visual assay** (sphere ladder at six rungs, elevation / distance /
  angular diameter / angular speed held per frame, in-loop blank and object radiance on both arms, per-body RF
  localizer, predeclared per-type primary statistics with Holm inside a 12-member family, 6 runs per arm in one
  submission) -- delivered and exported; the **fixed-anatomy model comparison** (base, `gain_fb 0`, per-stream
  rectification with signs preserved, 100 / 300 ms adaptation, spatial suppression and the two combinations, with the
  bright/dark, flash, flicker, bar and grating specificity battery and three benchmark draws per arm) -- run, **not
  exported**. Nothing adopted; no default moved.
- Open on our side, in order: (a) the **contrast-matched synthetic rectangle ladders** (height and width separately,
  as Keles & Frye did) -- recorded on the cluster, never fetched (~10 GB, 150 expected outputs missing); (b) the
  **export of the compare arms** (`object_round2_export.py compare --out out/objr2c --baseline
  out/interp/objr2c/compare.json`); (c) a **same-device `base` re-run** to de-confound arm from GPU model; (d) a
  **stimulus-driven LC11 localizer** (0 of 143 bodies fit at `z_min` 5); then the **transfer test** with fixed
  parameters and held-out stimuli. LC10a state gating is deferred. **Done since the round closed:** the **ON/OFF
  transition split**, re-derived on CPU from the stored specificity recordings
  (`out/interp/objr2c/spec_transitions.json`; section 3c). **All four closed in round 3 (section 3d):** (a) both
  rectangle ladders ran in two batches, 40/40 primary verdicts `null`, with the effective-contrast limit measured
  (matched only at width >= 8.8 deg); (b) the compare arms are exported (41 of the 95 directories); (c) the
  same-device re-run is done on two GPU models -- H200 `REPRODUCES`, B200 `PARTIAL`; (d) the stimulus-driven
  localizer ran and neither LC population localizes under a static 4.5-deg probe. The **transfer test** with fixed
  parameters and held-out stimuli is still open, and there is nothing to transfer until a mechanism passes.
- Open for Neurome: per-body LC11 / LC10a physiology at matched geometry when it exists (the matched sizes now exist
  and are exported, section 3c); the receptor tiers of the unprofiled types (Tm5Y, TmY21, TmY13, LC11 -- 100 %
  fallback-tier input); and the **quantitative form of the Keles 2020 Rdl constraint** (the effect size on the
  small-object response and the bound on the bar / grating release) so it can become a scored ledger row rather than
  the magnitude rule round 2 used. Their Keles 2020 receptor / perturbation constraints are today recorded as
  literature rows in `flyverse/data/expected_responses.csv` (op `report`), not adopted as gains or signs.
- **Delivery correction carried forward:** the 2026-09-13 ladder summary's `preference.csv` shipped four
  `spikes_median` preference rows (all `role = primary`) with `spearman_p_perm = 5.0e-05` against an undefined rho.
  Fixed and re-emitted 2026-09-14 (`objr2-ladder-20260914T024906Z-035363c0`); the 2026-09-13 directories stay on disk
  as the superseded delivery.
- **Coming: a cross-connectome anatomy table.** `docs/audits/flywire_banc_survey.md` (2026-09-14) surveys the two
  public Princeton / FlyWire female releases against the MaleCNS v1.0 graph we ship on -- FAFB v783 (brain with both
  optic lobes, 139,255 cells) and BANC v888 (brain **and** VNC, 158,262 cells, a *verified* transmitter for 65,369
  of them). Exact type-name overlap covers 59 % (FAFB) and 72 % (BANC) of MaleCNS cells and `type_aliases.csv`
  already bridges the rest, so the next deliverable on this side is a **side-by-side MaleCNS / FAFB / BANC table for
  every anatomical claim we have made** (`scripts/cross_connectome.py`; spec in `docs/CONNECTOME_BACKENDS_SPEC.md`),
  with a per-release synapse scale -- raw counts run ~1 : 0.6 : 0.3 and are not comparable unscaled, and BANC's
  optic lobes are under-proofread (T2 853 vs FAFB 1,466 vs MaleCNS 1,630). It is read-only and changes no model.
  Relevant to this interchange: FAFB's `column_assignment` (45,528 cells, 31 types, hex coordinates) is an
  independent column map for `trace.column_of_cells` and the LC anatomical windows -- though LC11 and LC10a
  themselves have no column assignment there, which is the same gap the round-3 localizer hit.
- Whether `bodyId` alone is stable across MaleCNS releases; we key on it and record the release.
- Cluster paths: agreed -- Neurome's namespace is `$NEUROME_NS/{runs,evidence,exports}/`;
  flyverse keeps `$CLUSTER_FLYVERSE/` and `$CLUSTER_RUNS/`.
- Neurome's first bundle (`D:\Projects\neurome\data\male_cns\small_object_v2\`, 680 MiB; `manifest.json`,
  `observed_edges.parquet` with 1,967,026 both-Traced edges incident to the 7,149 selected bodies, 91,727 endpoint
  bodies, a receptor-table excerpt, a 24-case fragment review packet) is accepted as read-only observations; no
  anatomical judgment in it is reviewed and none enters the simulator graph.
