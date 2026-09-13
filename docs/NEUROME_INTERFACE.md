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
- Open on our side, in order: the **matched visual assay** of section 3b (fixed-centre square ladder, then separate
  height and width ladders; per-body RF localizer; in-loop blank and object radiance; predeclared per-type primary
  statistics and Holm family; >= 5 runs per arm in one submission), then the **fixed-anatomy model comparison**
  (sum / per-presynaptic-stream rectification with signs preserved / adaptation + spatial suppression, with
  `gain_fb 0` and feedback-hold controls and the bright/dark, ON/OFF, flicker, bar and grating specificity battery),
  then the **transfer test** with fixed parameters and held-out stimuli. LC10a state gating is deferred.
- Open for Neurome: per-body LC11 / LC10a physiology at matched geometry when it exists; the receptor tiers of the
  unprofiled types (Tm5Y, TmY21, TmY13, LC11 -- 100 % fallback-tier input). Their Keles 2020 receptor / perturbation
  constraints are recorded as literature rows in `flyverse/data/expected_responses.csv` (op `report`), not adopted as
  gains or signs.
- Whether `bodyId` alone is stable across MaleCNS releases; we key on it and record the release.
- Cluster paths: agreed -- Neurome's namespace is `$NEUROME_NS/{runs,evidence,exports}/`;
  flyverse keeps `$CLUSTER_FLYVERSE/` and `$CLUSTER_RUNS/`.
- Neurome's first bundle (`D:\Projects\neurome\data\male_cns\small_object_v2\`, 680 MiB; `manifest.json`,
  `observed_edges.parquet` with 1,967,026 both-Traced edges incident to the 7,149 selected bodies, 91,727 endpoint
  bodies, a receptor-table excerpt, a 24-case fragment review packet) is accepted as read-only observations; no
  anatomical judgment in it is reviewed and none enters the simulator graph.
