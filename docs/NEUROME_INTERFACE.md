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
upstream T2 / T3 responses, and the actual retinal sampling. A diagnostic, not a pass criterion.

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
[UV, B, G, R] per frame) and `per_type.csv`. Generator: `scripts/interp_apply_object.py ladder` (batch script in
`out/apply_object/ladder/`).

**Result** (stimulus minus blank, verdict = |z| >= 3 on the blank-run SD and exact U p <= 0.05, 5 v 5 runs):

| type | statistic | 4.5 deg | 11.4 deg | 20 deg | 30 deg |
|---|---|---|---|---|---|
| LC11 | received drive, max over cells (mV) | z -0.9 | +0.6 | +1.6 | **+4.8** (0.171 vs 0.054) |
| LC10a | same | -0.1 | +0.7 | +2.8 | **+8.6** (0.208 vs 0.076) |
| LPLC2 | same (loom chain, for scale) | -0.5 | +2.6 | **+13.5** | **+98.7** (2.03 vs 0.12) |
| LC11 / LC10a | output rate (Hz) | 0.00 | 0.00 | 0.00 | 0.00 (no body fires for the object at any size) |
| T2 / T3 / Tm5Y / TmY21 | best-cell signed figure | null | null | **result** (z +7.5 / +11.9 / +6.5 / +3.4) | **result** (+56.6 / +36.5 / +27.5 / +17.5) |
| T2 / T3 / Tm5Y / TmY21 | population signed mean | null | null | null | null |

Reading: in the model the small-field stage and the LC10 / LC11 drive respond only to the two largest objects and
the loom chain (LPLC2) from 20 deg -- the animal's size ordering (LC11 preferring 5-10 deg objects, Keles & Frye
2017) is inverted -- and LC11 / LC10a never spike for the object at any size (their received drive stays 0.05-0.2 mV
against a 7 mV threshold gap). The localization behind that is in `docs/audits/deficit_object.md`: at T3 the ON and
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
against `readout_per_body.csv` (`bodyId` decimal strings; `n_trials` 5; `control_ids` = the blank run ids).

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
- Delivered: the size-tuning export (section 3). Field changes needed on the compiler side: none so far.
- Whether `bodyId` alone is stable across MaleCNS releases; we key on it and record the release.
- Cluster paths: agreed -- Neurome's namespace is `<cluster-fs>/neurome/neurome-reconstruction/{runs,evidence,exports}/`;
  flyverse keeps `<cluster-fs>/neurome/flyverse/` and `<cluster-fs>/neurome/runs/`.
- Neurome's first bundle (`D:\Projects\neurome\data\male_cns\small_object_v2\`, 680 MiB; `manifest.json`,
  `observed_edges.parquet` with 1,967,026 both-Traced edges incident to the 7,149 selected bodies, 91,727 endpoint
  bodies, a receptor-table excerpt, a 24-case fragment review packet) is accepted as read-only observations; no
  anatomical judgment in it is reviewed and none enters the simulator graph.
