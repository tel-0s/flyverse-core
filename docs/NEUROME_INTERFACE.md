# Flyverse <-> Neurome: the evidence / probe interchange

Reply to Neurome's request (`D:\Projects\neurome\docs\flyverse-evidence-handoff.md`, 2026-09-12). Neurome owns
source anatomy, reconstruction evidence and molecular correspondence; flyverse owns the model, its physiological
transforms, simulation and probes. First joint case: the small-object pathway (T2, T3, Tm5Y, TmY21, TmY13, TmY5a,
LC11, LC10a, LC10b) -- localized on our side to the medulla -> lobula stage (`docs/audits/object_sweep.md`; the
medulla carries a moving ball at z +22 to +29, T2 / T3 / Tm5Y / TmY21 and the LC types sit at the none-vs-none null)
and out of reach of the receptor-expression data (none of those types has a profile in any of six sources,
`docs/NT_INTEGRATION.md` section 7).

## 1. What flyverse will export (read-only probe export)

Implemented by the interpretability toolkit (`flyverse/interp/export.py`, CLI `scripts/interp_export.py`; see
`docs/INTERP.md` once written). Location: `out/export/<run_id>/` with `manifest.json` plus CSV / Parquet tables,
one directory per probe run; a run is a directory, never a mutable file. The interchange key is the **MaleCNS
`bodyId` as a decimal string** (int64 in our cache, 167,106 unique); model indices and type names travel alongside.

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
  presynaptic bodies: 3,407 (unknown NT, dopamine, octopamine, serotonin).
- **Nothing is combined**: no cells are merged; per-connection synapse counts are capped at 60 in the effective
  weight (the raw count is kept in the cache and can be exported).

## 3. The first experiment (Neurome's request)

Size tuning of **LC11 and LC10a, separately**: objects of ~4.5, 11, 20 and 30 degrees with matched blank controls,
upstream T2 / T3 responses, and the actual retinal sampling. This is a diagnostic, not a pass criterion. Our
`scripts/probe_object_sweep.py` already has `--ball-radius / --ahead / --null` (the round-4 skeptic ran a
11 / 22 / 28 / 43 degree ladder: LPLC2 rose +0.11 / +0.97 / +1.37 / +3.67 mV, LC11 +0.05 / +0.12 / +0.12 / +0.21);
the interp workflow's `apply:object` task runs the requested ladder through the export, with >= 3 replicates per
size and matched nulls, and posts the run directory here.

## 4. Neurome evidence bundles

Accepted initially as **read-only observations**: predicted anatomical counts, source annotations, reconstruction
candidates, evidence references. Unreviewed candidates never modify the simulator graph; the compiler
(`flyverse/connectome.py`) reads only the released MaleCNS tables, and every model-side change so far went through
the audit-and-verify procedure of `docs/audits/`. When anatomy is independently supported, a revision format can
feed the compiler into an isolated cache (`connectome.load(cache_dir=<scratch>, rebuild=True, ...)` is the
existing pattern -- scratch caches per experiment, the shared cache untouched), scored against the benchmark
suite before adoption.

## 5. Open items for Neurome

- The schema and a checksummed example bundle (their side); we will adapt field names to the compiler's columns.
- Whether `bodyId` alone is stable across MaleCNS releases; we key on it and record the release.
- Cluster paths: flyverse's checkout and run directories live under `<cluster-fs>/neurome/` (flyverse/, runs/); keep
  the two projects' run directories apart there.
