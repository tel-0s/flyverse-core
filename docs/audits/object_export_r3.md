# Object round 3: Neurome revision 2 delivery

The delivery is indexed by `out/export/objr3_index.json`; its flat table inventory
is `out/export/objr3_tables.json`. `scripts/object_round3_export.py` uses the
existing revision 2 exporter and verifier. It does not change the interchange
schema, model defaults or recorded inference families.

## 1. Contents

| Experiment | Directories | Source |
|---|---:|---|
| Round-2 comparison: 8 arms x 5 sphere rungs, plus family | 41 | `out/interp/objr2c/compare.json`, `out/objr2c/` |
| Round-3 same-device H200: 3 arms x 5 rungs, plus family | 16 | `out/objr3sd/samedevice.json`, `out/objr3sd/` |
| Round-3 rectangles: 2 lobes x 9 distinct shapes x 2 contrasts | 36 | `out/interp/objr3rect/rectangles.json`, `out/objr3rect/` |
| Round-3 pooled RF maps: shipped and fb0 | 2 | `out/interp/objr3rf/rfmap_r3_{ship,fb0}.json` |
| **Total** | **95** | **1,712 tables** |

Each sphere/rectangle directory includes LC received drive and output readouts,
resolved model parameters, source fingerprints, per-run scatter and effective
contrast. Round-2 comparison directories carry operating levels, specificity
comparisons and benchmark results; the family directory also carries the full
ON/OFF transition tables and 24 benchmark provenance sidecars. The battery is the
recorded round-2 battery, not a newly run full benchmark suite.

The comparison-arm names occupy the lobe/group slot. Round-2 optic hook diagnostics
are copied from the same arm's recorded synthetic-stimulus summary with its source
path and hash. Same-device sphere recordings did not record those diagnostics;
their resolved OpticParams are preserved, without inventing live hook counts.

## 2. Every LC body, with the right time resolution

All **91 sphere/rectangle rung directories** contain `lc_per_body_time_course`:
143 LC11 plus 275 LC10a bodies, 1,200 frames per body at 10 ms. These are
**45,645,600 rows** in total, including bodies whose analysis window misses the
trajectory. For received drive and per-frame spikes, each row carries object,
paired blank and within-run difference means plus sample SD over runs. Means
are taken after within-run subtraction; cells and frames are not replicates.

The rectangle lattice reduction preserves window means exactly but cannot recover
frame order. `pack-rect` therefore selected LC columns from all 324 original paired
recordings on the original box, using CPU only, and retained their hashes and
provenance in `out/objr3rect_lc/`. `verify-lc` reports 324 archives, zero problems.
No GPU re-recording or statistical reconstruction was used to fill those traces.

The RF recorder retained **node/role-window responses**, not per-body frame arrays.
Each map therefore has `lc_per_body_node_responses`: 418 bodies x 1,466 nodes,
with stimulus/blank means and run SDs for `on`, `on1`, `off` and `base`. Across
the two lobes this is **1,225,576 rows**. The shuffled presentation orders are
pooled by spatial node. This is explicitly labeled as a node response table;
the unavailable 10-ms LC localizer traces are not manufactured.

## 3. Windows, contrast, devices and inference

`window_coverage` gives `n_bodies_windowed` by type and `window_source` per rung.
An anatomical assignment does not imply that the trajectory entered the window:
for example, the height ladder windows 55/143 LC11 bodies at height 2.2 degrees,
99/143 at height 8.8, and 103/143 at height 30. All assigned LC11 windows are
anatomical; none is relabeled as a fitted RF. Actual per-body frame counts remain
in the readout table. The effective-contrast table accompanies the retinal capture;
fixed object Weber contrast does not imply matched retinal contrast for small shapes.

Every table has `device_name` and `same_device_as_reference`. Comparison rows with
an explicit `against` field use that reference: own-arm nulls are on the same
device; vs-base rows compare with that experiment's base. Other comparison-delivery
rows describe the arm's device relative to its batch's base; rectangle/RF rows
use the shipped-lobe reference. Pipe-separated device names denote aggregate
tables spanning GPU models. These fields do not claim identical hardware
instances, source trees, random trajectories or numerical results.
For static retinal anatomy and reused window-definition tables, the label names
the associated experiment, not a device on which anatomical coordinates were
measured. Their own source-map paths remain in the provenance.

This distinction matters in round 2: the sphere reference ran on B200, whereas
the specificity and benchmark reference ran on H200. Finalization reads the
actual per-run provenance for each section, including every ON/OFF source run;
it does not copy the family directory's representative device into all rows.

Original stamped `family`, tie-aware p and Holm values are retained. The export
does not form new testing families from a subset of rungs, reclassify nulls as
equivalence, or promote exploratory bodies to population-level results. Round-2
and round-3 Results remain distinct datasets. House replications are reported
separately in the three experiment audits and their own Results.
The combined index also links `out/export/objr3_house_results/index.json`, which
bundles **four native Result JSONs** for the fresh B200 sphere, rectangle and two
RF analyses, with all tables, full provenance and hashes. They are separate
replication evidence, not additional revision-2 rung directories or pooled runs.

## 4. Reproduce the delivery

All commands below are CPU work. On Windows use `PYTHONIOENCODING=utf-8` and
`CUDA_VISIBLE_DEVICES=-1` for the process.

```text
python scripts/object_round3_export.py compare
python scripts/object_round3_export.py compare --out out/objr3sd --result out/objr3sd/samedevice.json --prefix objr3-samedevice --index out/export/objr3_samedevice_index.json --no-transitions
python scripts/object_round3_export.py rectangles
python scripts/object_round3_export.py rfmap --results out/interp/objr3rf/rfmap_r3_ship.json out/interp/objr3rf/rfmap_r3_fb0.json
python scripts/object_round3_export.py finalize --indices out/export/objr3_r2compare_index.json out/export/objr3_samedevice_index.json out/export/objr3_rectangles_index.json out/export/objr3_rfmap_index.json
python scripts/object_round3_export.py link-replications --results out/objr3sd_house/samedevice.json out/interp/objr3rect_house/rectangles.json out/interp/objr3rf_house/rfmap_r3_ship.json out/interp/objr3rf_house/rfmap_r3_fb0.json
python scripts/object_round3_export.py check-delivery
```

`finalize` is part of the delivery workflow: it completes row provenance,
replicate metadata and the combined index, updates hashes, and checks the final
artifacts. It preserves verdict text `null` using the exporter's reader. The
analysis Results remain the source for restored text and numerical columns;
they are not regenerated from CSV type inference. The finalizer snapshot is
`out/export/object_round3_export_snapshot.py`, with its hash in the manifests.

`pack-rect --raw DIR --out DIR` is the lossless CPU packing step. `fetch --target
house --run RUN --subdir NAME --out DIR` resumes a named output directory with a
compressed tar stream, comparing every local/remote file size and SHA-256, including partial
files. It writes a receipt; it neither submits GPU work nor changes remote data.
`receipts --target TARGET --log LOG --out DIR` recovers scheduler states by the
job IDs in an existing submission log. Recovered status is labeled as such.

## 5. Verification and skeptic corrections

Every final directory is checked with `verify(..., neurons=True)`: table hashes,
body IDs, expected LC counts, received-drive/output pairing and required
provenance. The finalizer also requires the source-Result round trip to preserve
IDs and text, with numerical differences below 1e-6. `checks.json` travels in
each directory; the combined index lists its problems.

The independent delivery checker scans every table for row provenance and every
LC frame table for the complete body/frame lattice. It recomputes representative
paired mean and sample-SD traces directly from raw NPZs, without the export
time-course reducer. Its detailed scope and residuals are in
`out/export/objr3_skeptic.json`.

The skeptic corrected three delivery defects before handoff: transition/battery
device labels must come from their own source runs; the inherited sphere export's
empty replicate record must carry the actual run count; and pandas' default NA
parser must not turn a scientific verdict of `null` into a missing field. These
are export/metadata corrections, not revisions to the experimental statistics.

Final validation: **95 directories, 1,712 tables, zero verifier or round-trip
problems**. The independent checker finds **zero problems** across those tables
and the 91 LC frame lattices. Nine representative rung checks (three from each
of round 2, same-device and rectangles) independently reproduce every paired
received-drive mean and run SD for all 418 bodies x 1,200 frames. This is nine
raw-trace checks, not a claim of independently recomputing every exported trace.

## Report

```yaml
summary: >-
  Delivered the round-2 comparison and original round-3 object experiments through
  revision 2, with row provenance, LC traces, window coverage, contrast and held-out
  battery sidecars. All 95 directories and 1712 tables verify without problems.
key_claims:
  - All 91 sphere/rectangle rungs contain all 418 LC bodies at all 1200 frames, with run means and sample SDs.
  - Both RF maps retain all LC node/role responses and scatter, explicitly distinct from unavailable frame traces.
  - Original inference families and Holm values are retained; per-section reference devices are explicit.
  - Source Results round-trip with IDs and literal verdict text preserved.
files_written:
  - scripts/object_round3_export.py
  - docs/audits/object_export_r3.md
  - out/export/objr3_index.json
  - out/export/objr3_tables.json
  - out/export/objr3_skeptic.json
  - out/export/object_round3_export_snapshot.py
  - out/export/objr3_house_results/index.json and its four verified native Results
  - out/export/objr3_r2compare_index.json and its 41 listed directories
  - out/export/objr3_samedevice_index.json and its 16 listed directories
  - out/export/objr3_rectangles_index.json and its 36 listed directories
  - out/export/objr3_rfmap_index.json and its 2 listed directories
  - out/objr3rect_lc/
  - out/round3_astra/ export logs, smoke artifacts and source-review evidence
api:
  - compare defaults to out/objr2c and out/interp/objr2c/compare.json; optional --group and --sizes filter rungs.
  - compare --transitions defaults to out/interp/objr2c/spec_transitions.json; --no-transitions omits those sidecars.
  - rectangles defaults to out/objr3rect, its analyzed Result and out/objr3rect_lc; optional --group selects one shape/lobe/contrast.
  - rfmap requires --results; default export root is out/export and prefix is objr3-rfmap.
  - finalize requires --indices; defaults to out/export/objr3_index.json and --spec-root out/objr2c/spec.
  - check-delivery defaults to out/export/objr3_index.json and out/export/objr3_skeptic.json.
  - pack-rect requires --raw and --out; retains every LC body and both arms at 10 ms.
  - verify-lc requires --out and defaults to --expected 324.
  - fetch defaults to --target house; --run, --subdir and --out are required; resume checks SHA-256 and size.
  - receipts requires --target, --log and --out; reads scheduler states of existing job IDs only.
  - check-stats requires --result and --json; optional --out enables scheduler/stamp validation.
  - check-rf requires --result and --json; checks raw pooled and per-run fits independently.
  - check-preference requires --result and --json; checks rectangle rank and permutation statistics independently.
  - check-rect-raw defaults to the original rectangle batch, Result and lossless LC archive.
  - check-contrast defaults to the original rectangle batch and Result, writing out/objr3rect/skeptic_contrast.json.
  - link-replications requires --results; defaults to objr3_index.json and out/export/objr3_house_results, preserving native Results separately.
validation: >-
  Revision-2 verify with neuron validation and source-Result round trips passed
  for every directory. The independent scan checked 1712 tables, all 91 LC frame
  lattices and nine representative complete paired received-drive traces with
  their sample SDs across five or six runs, with zero problems. Rectangle LC
  archive verification passed for 324 recordings. No model adoption or new
  benchmark pass is implied by exporting the existing evidence.
recommendations:
  - Use objr3_index.json as the delivery entry point and the device-reference fields when comparing arms.
  - Carry the audits' contrast, anatomical-window and static-localizer limitations into Neurome documentation.
  - If 10-ms per-cell localizer traces are required, specify a new recording protocol before another long run.
open_questions:
  - Does Neurome need localizer frame chronology in addition to the delivered node responses and run-order provenance?
refuted:
  - A representative family device is sufficient for all transition and benchmark rows.
  - Per-body localizer frame series can be reconstructed from the retained node means.
confirmed:
  - All original comparison arms, rectangle shapes and RF maps are present in the revision-2 delivery.
  - Final verifier, round-trip and independent delivery checks report zero problems.
corrections:
  - Joined transition and battery labels to their actual source-run provenance.
  - Populated inherited empty sphere replicate metadata with actual stimulus/null counts.
  - Preserved literal null verdicts through CSV completion and restored source-Result text before verification.
  - Added SHA-256 to resumed transfers after detecting a partial file with a matching size.
verdict: mostly sound
```
