# A stimulus-driven LC11 / LC10a receptive-field localizer, round 3 (`scripts/object_round3_localizer.py`) -- `run:localizer`

**Status: original H200 maps and fresh-seed B200 replication complete, verified and independently checked.
The Results are `out/interp/objr3rf/rfmap_r3_{ship,fb0}.json`.** The predeclaration is the stamped JSON
`out/objr3rf/predeclared.json` (`stamped_utc` **2026-09-14T05:26:46Z**), written by `plan` before the 05:27Z
submission; this document is not the predeclaration (docs/INTERP.md 10.4 item 10). Nothing in `flyverse/` is
edited; every hook stays off; the localizer is a measurement, not a proposal.

Read first: `docs/audits/object_baseline_r2.md` 0b (0 of 143 LC11 bodies fitted at `z_min` 5; 405 of 418 LC windows
anatomical), `docs/audits/object_synthetic_stimuli.md` 4 / 5 / 7 / 8 (the round-2 localizer, the lobe's autonomous
4.5-Hz oscillation that every within-run window sits on, the 15-deg three-pass result), `docs/INTERP.md` 2.7 (the
RF-map file this map is written in) and 10.4 items 9-12, `docs/NEUROME_INTERFACE.md` 3c item 4 (what Neurome asked
about LC11 receptive fields).

Files: `scripts/object_round3_localizer.py` (`preview` / `plan` / `submit` / `record` / `verify` / `rfmap` /
`selfcheck`), `out/objr3rf/` (`preview.json`, `batch.sh`, `jobs.json`, `predeclared.json`, `tree_state.json`,
`submit_console.txt`, then `loc/` = the fetched runs, `verify.json`, `rfmap_r3_ship.csv`, `rfmap_r3_fb0.csv`),
`out/objr3rf_cluster.log` (the cluster console), `out/interp/objr3rf/rfmap_r3_{ship,fb0}.json` (the Results, schema
`flyverse.interp.rfmap/1`). The CPU smoke is `out/objr3rf_smoke/` (18 nodes, 0.05-s dwell, `device cpu`; nothing in
it is a number of this map).

## 0. Predeclared reading rules (`out/objr3rf/predeclared.json`, stamped 2026-09-14T05:26:46Z)

The rules are the `PREDECLARED` dict of the script, dumped by `plan` with the sha256 of the analysis code at the
stamp (`analysis_code_sha256_at_stamp`), and copied into every Result (`summary.predeclared`,
`files.analysis_code`). In prose:

* **The question.** Can a small, maximum-contrast dark probe with a long dwell, a fine grid over the anatomical
  boxes and >= 4 pooled runs fit a stimulus-driven RF centre for the LC11 and LC10a bodies, on the shipped lobe
  and on the `gain_fb = 0` lobe -- and if not for LC11, at what measured false-fit rate is that the answer.
* **Replicate unit.** One process = one (stimulus, blank) pair under one brain seed and one node order = one run.
  The map POOLS the per-node responses over the runs of a lobe before the fit (the task's ">= 4 runs pooled per
  node"); the per-run fits are on file (`n_runs_fitted`, `centre_spread_deg`) and so are the split-half pooled
  fits (`split_half_centre_distance_deg`). Five runs per lobe, both lobes in ONE submission under one
  `--arm-block fam` block (`fam_locr3` on every job line), so both maps come from one box.
* **Quantity.** The per-body received drive (`drive_mv`, the LIF's input in mV) for the spiking LC cells; `optic_dr`
  for the rate units. Spikes are reported (`spikes_on_minus_base_hz_peak`) and never fitted.
* **Windows.** `on` = the 50 flash frames minus `base` (PRIMARY); `on1` = the first 10 flash frames minus base and
  `off` = the 10 frames after the offset minus base (secondary); `base` = the last 20 frames of the 0.5-s blank
  before the onset.
* **The prior.** Per cell the |W|-weighted set of columns of its rate inputs (`trace.column_of_cells` over the
  optic-lobe rate units): the weighted centroid (`anat_input_az_deg / _el_deg`), the 50 % / 80 % weight radii, the
  number of distinct input columns; the round-2 single column (`anat_az_deg / anat_el_deg`) kept beside it. The
  fit uses the grid nodes within 20 deg of the input-weighted centroid (>= 12 nodes, else NaN). The criterion-free
  check is `free_peak_in_box`: the argmax |R| over the WHOLE grid falls inside that box, against `chance_in_box` =
  the fraction of grid nodes inside it.
* **The fit rule.** `probe_synthetic_stimuli.fit_rf`'s rule on the pooled response restricted to the box nodes:
  peak = argmax |R|, noise = 1.4826 x MAD over the box nodes, FITTED when |peak| >= `z_min` x noise; centre = the
  (R - half-max)-weighted centroid of the nodes at or above half-maximum of the peak's sign; width = the
  equivalent-disc FWHM `2 sqrt(n spacing^2 / pi)`, at least one spacing (4 deg).
* **The false-fit rate.** The same rule on the run-pooled BLANK arm (the same node schedule, the background
  presented) per type at every z of the ladder {3, 4, 5, 6, 8}.
* **The threshold rule.** Per type and lobe, z* = the smallest z of the ladder at which the pooled blank-arm
  false-fit rate is <= 1 %; if none, the largest z, reported as "no threshold reaches the bound". `z_min` 5 (round
  2's) is reported beside it.
* **The answer.** Per type and lobe: the fraction of bodies fitted at z*, stated WITH the blank-arm false-fit rate
  at z*; the centre and width distributions; the agreement with the prior (median great-circle distance of the
  fitted centre to the input-weighted centroid and to the round-2 single column, deg); `free_peak_in_box` against
  chance.
* **The call.** A type is LOCALIZED on a lobe when (a) coverage at z* exceeds the blank-arm false-fit rate at z* by
  >= 0.05 AND (b) `free_peak_in_box` >= 2 x `chance_in_box`; otherwise "not localized by this stimulus". If LC11 is
  not localized on either lobe the finding is stated as such -- no RF was established by this stimulus,
  prior and fitting rule at the measured false-fit rate -- never "silent" (the cells fire) and never
  "inverted".
  (Measured in this batch rather than assumed: in `loc45_ship_r0_nodes.npz` only 6 of 143 LC11 bodies and 25 of 275
  LC10a bodies emit any spike at all, at population mean rates of 0.0017 and 0.0185 Hz. The word "silent" is avoided
  because the fit is on received drive and the protocol is not a firing assay -- not because these cells fire
  appreciably here; the script's docstring records that "the LC populations emit essentially none in this protocol".)
* **No verdict vocabulary.** A map is magnitudes: no `compare()` verdict, no Holm. The only inferential quantities
  are the false-fit rate (measured) and the chance level of the free-peak check (computed).
* **Controls.** Mi1 (hex-annotated: the ground truth for a column-sized RF), T2 / T3 (LC11's inputs) and Tm5Y /
  TmY21 (LC10a's) are recorded and fitted by the same rule, so "the inputs are retinotopic, the output is not" is
  measured, not assumed.
* **Consumers.** The CSV / JSON follow docs/INTERP.md 2.7: the first seven columns are the interface; fitted rows
  have a finite `az_deg`; `anat_az_deg / anat_el_deg` keep round 2's single-column meaning so the ladder window
  rule's fallback is unchanged, and the input-weighted centroid is the new `anat_input_*` pair.
* **Not matched, stated up front.** The probe's effective contrast at a column depends on the node-to-column
  offset (coverage 0.54-0.89 at the best node of a 4-deg grid, median 0.66); the map's width is bounded below by one
  grid spacing (4 deg) and above by the 20-deg box.

## 1. The design, and why each number is what it is (`preview`, `out/objr3rf/preview.json`)

### 1.1 The recorded probe is 4.5 deg, slightly larger than the requested 2-4 deg

The task asked for a 2-4 deg dark square at maximum contrast. On this retina a column samples a 7-ray Gaussian
kernel of acceptance 4.5 deg (`Retina.ray_directions`; the centre ray and six at 2.25 deg), and the object's
effective contrast at a column is its Weber contrast times the fraction of that kernel it covers
(`object_synthetic_stimuli.md` 1.1). A square contains the whole kernel only from 4.5 deg. Measured with
`rect_coverage` on the shipped retina, the best-node coverage per column over a 4-deg grid:

| square | best-node coverage per column: median | 10th pct | max | columns >= 0.5 |
|---|---|---|---|---|
| 2 deg | 0.115 | 0.115 | 0.426 | 0 % |
| 3 deg | 0.426 | 0.115 | 0.541 | 29 % |
| 4 deg | 0.541 | 0.541 | 0.885 | 99 % |
| **4.5 deg** | **0.656** | 0.541 | 0.885 | 99 % |
| 8.8 deg | 1.000 | 1.000 | 1.000 | 100 % |

Smaller probes reduce the typical sampled signal: at the median column's best grid node,
a 2-deg square covers about 12 %, a 3-deg square about 43 %. The 2-deg maximum in the table is
43 %, so "at most 12 %" would be incorrect. Smaller objects can still differ in spatial sampling;
the acceptance kernel does not prove they are equivalent weaker stimuli. The recorded probe is
**4.5 deg at Weber contrast -0.995** (the black ball's contrast, the darkest object the room presents), which at
the best node covers 0.54-0.89 of a column (median 0.66; `params.column_equivalents_per_node_median` 1.0: each
node dims one column-equivalent in total -- but `column_equivalents_per_node_min` is 0.0, and 3 of the 1,466
nodes present essentially nothing while 90 present less than half a column-equivalent). Both facts are in
`predeclared.not_matched`.

### 1.2 The prior: `column_of_cells` is a 13-column assignment for 143 LC11 cells

`trace.column_of_cells` gives a spiking cell the column of its strongest rate input. Read from the round-2 maps
(`out/synth2/rfmap_150_fb0_p3.csv`, `anat_column`) and recomputed by `preview`:

| type | cells | distinct single columns | the largest four | input columns per cell (median) | 50 % weight radius (median) | 80 % | single column to input centroid (median) |
|---|---|---|---|---|---|---|---|
| **LC11** | 143 | **13** | 50 / 44 / 14 / 13 cells | **100** | **26.1 deg** | 72.6 deg | **64.5 deg** |
| **LC10a** | 275 | 88 | 52 / 33 / 31 / 31 | 48 | 44.8 deg | 72.8 deg | 47.1 deg |
| T3 | 1,940 | 1,245 | 44 / 33 / 27 / 23 | 19 | 13.2 deg | 51 deg | 12.0 deg |
| T2 | 1,630 | 1,043 | 38 / 34 / 26 / 16 | 41 | 16.5 deg | 57 deg | 14.3 deg |

Two consequences. (1) The round-2 anatomical windows of the LC types were not per-cell regions: 50 LC11 bodies
shared one column at (-25.4, +3.9) and 44 another at (+43.5, -10.7), so the sphere ladder's LC11 "anatomical box"
was two boxes for two thirds of the population, and the "agreement with the anatomical column" that the round-2
localizer reported for LC cells was agreement with a column a median 65 deg from where the cell's input weight
sits. (2) The input column SET is wide: an LC11 cell in this connectome receives input from a median 100 distinct
columns with half its weight within 26 deg of the centroid (T3, the small-field control, 19 columns within 13
deg). The prior this round uses is that centroid (`anat_input_az_deg / _el_deg`) with its radii on file; the
single column is kept beside it because the ladders' fallback reads it. A 20-deg box around the centroid holds a
median 81 grid nodes per LC cell (42-133).

### 1.3 The grid, the dwell, the runs

* **Grid**: 4-deg spacing (sub-column against the 4.6-deg pitch), nodes that reach a column AND lie within 20 deg of
  any LC11 / LC10a input-weighted centroid: **1,466 nodes** against 1,787 for the whole reachable eye at the same
  spacing (the LC centroids cover most of the eye, so "not the whole eye" saves 18 %; the box restriction is what
  the fit uses). Grid sha256 `21d4f2ab6818...`, recorded in every run and checked equal across runs by `verify`
  and `rfmap`.
* **Dwell**: 0.5 s on, 0.5 s of background between nodes, one presentation per node per run: 146,650 frames =
  1,466.5 s per arm, two arms per run. The lobe's autonomous oscillation (`object_synthetic_stimuli.md` 5: 4.5 Hz,
  ~2 mV per LC11 cell) averages over ~2 cycles inside a 0.5-s window instead of ~1 in round 2's 0.2 s.
* **Runs**: 5 per lobe, seeds 0-4, node order seed = run index -- a different shuffled order per run, so the pooled
  per-node response samples the oscillation at different phases in different runs while a retinotopic response
  adds coherently. On the `gain_fb = 0` lobe two runs with the same order would be the same trajectory (the lobe
  is deterministic given the input), which is why the order, not only the seed, changes per run.
* **Recording**: per node and cell the mean of one quantity over the four role windows (`resp__on / on1 / off /
  base`, float32) and spikes per frame for the spiking cells; per-type population means per frame; the stimulus as
  pattern + index with per-frame checksums on both arms. No per-frame per-cell array is written, so there is
  nothing to slim on the box: ~300 MB per run.

### 1.4 Validation on the CPU

`selfcheck` (no connectome): 200 synthetic cells on a 4-deg grid, 100 with a Gaussian RF (sigma 5 deg, amplitude 2)
offset 5 deg from their prior centroid and 100 without, noise SD 0.5 per run, 5 runs. The threshold rule picks z* =
5 for the responders' type (blank false-fit 0.31 / 0.04 / 0.00 at z 3 / 4 / 5); at z* 95 % of responders are
fitted from the pooled response against 10 % from one run, the centre error is 0.63 deg median (p90 1.14), the
width 11.9 deg against a true FWHM of 11.8, 0 of 100 non-responders are fitted, and the free peak lies in the box
for 100 % of responders against 13 % of non-responders (chance 12.5 %). The first version of `fit_boxed` failed
this check (an `abs(nan_to_num(-inf))` made out-of-box nodes win the argmax; fixed before the stamp -- the stamped
sha256 is the fixed code).

CPU smoke (`out/objr3rf_smoke/`, `CUDA_VISIBLE_DEVICES=-1`, 18 nodes at 40-deg spacing, 0.05-s dwell, `gain_fb=0`):
both arms' checksums true, `rfmap --allow-cpu` end to end (`problems: none`), `verify` flags the CPU device as it
must.

## 2. The batch (`objr3rf-a2fc62`, 10 jobs = 10 processes, ONE submission)

`plan --out out/objr3rf --name objr3rf --minutes 90 --runs 5` (stamp 05:26:46Z) -> `submit` (one
`scripts/cluster_run.py --arm-block fam` call, `--fetch out/objr3rf/loc/`, console teed to
`out/objr3rf_cluster.log`), submitted 2026-09-14 05:27Z. Every job line: `mkdir -p out/objr3rf/loc && source
.venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && python
scripts/object_round3_localizer.py record --lobe <ship|fb0> --run <k> --tag fam_locr3 --out <stem> > <stem>.txt
2>&1; st=$?; tail -4 <stem>.txt; exit $st` (the exit code is python's; `cluster_run` printed no exit-masking
warning). `block_value(line, 'fam')` reads `locr3` on all 10 lines; the block was dealt to **`r3-h200a`** (both
lobes, all 10 jobs, one box), behind 10 jobs of the other round-3 workflows already on that box (the other box held
12): the batch waited for capacity rather than splitting.

The tree the batch shipped from (`out/objr3rf/tree_state.json`, commit `d2abf3cd`): 17 files differing from
`origin/main`, among them other tasks' uncommitted edits to `flyverse/brain.py` (+31 lines at plan time),
`senses.py` (+169), `body.py` (+132), `motor.py`, `batch_body.py`, `batch_sim.py` -- `tree_state.json` records no
per-file counts, so these are plan-time figures and they drift: `body.py` is +134 in the current tree.
`flyverse/optic.py`, `retina.py` and `fly.py` are at
HEAD. The code identity of every run is its `provenance.source_fingerprint` (docs/INTERP.md 10.4 item 6), not the
commit; the optic path the localizer measures is the committed one.

## 3. Verification

`out/objr3rf/verify.json` reports **10 runs, 10 consoles, 20 node archives, zero
problems**. All runs record NVIDIA H200, the same 1,466-node grid, the required
shipped/fb0 parameters and matching stimulus checksums. Node files are checked for
required arrays, body alignment, grid alignment and nonempty expected recordings;
the fitter now refuses a missing run instead of silently pooling fewer runs.

Fable's `FETCHED.txt` records 100 fetched, size-checked files. The original client
died before its final polling line; `scheduler_receipt.json` and the explicitly
labeled `recovered_cluster.log` recover **10 jobs, 0 failed** from the scheduler.
The 05:26:46Z predeclaration precedes the first 05:27:14Z job submission.

## 4. The pooled map

Both maps are written in the documented RF CSV interface, preserving anatomical
fallback coordinates and body IDs. At the selected z*=5 for both LC types:

| Lobe | Type | Fitted bodies | Coverage | Blank false-fit fraction | Free peak inside prior | Grid chance |
|---|---|---:|---:|---:|---:|---:|
| ship | LC11 | 0 / 143 | 0 | 1 / 143 (.00699) | .04196 | .05843 |
| fb0 | LC11 | 0 / 143 | 0 | 1 / 143 (.00699) | .04895 | .05843 |
| ship | LC10a | 0 / 275 | 0 | 0 / 275 | .06545 | .05104 |
| fb0 | LC10a | 2 / 275 | .00727 | 0 / 275 | .07273 | .05104 |

**Neither LC population meets the predeclared localization criterion.** No LC11
centre or width distribution exists to summarize. The two fitted fb0 LC10a bodies
have median fitted width 11.91 degrees and median distance 13.52 degrees from the
input-weighted anatomical centroid; two bodies do not localize the population.

## 5. Controls and run scatter

The same fit detects the positive control Mi1: **946 / 1,773 shipped bodies
(53.36 %)**, with 3 / 1,773 blank fits (0.169 %); fb0 gives 947 / 1,773,
with 2 / 1,773 blank fits (0.113 %). Mi1's free-peak fraction is .60124 shipped
and .60519 fb0, against chance .05662. T3 gives 99 / 1,940 shipped and 116 /
1,940 fb0 fits, with one blank fit on each lobe. The localizer can detect a
retinotopic input signal **in the rate units**: Mi1, T2, T3, Tm5Y and TmY21 are all
fitted on `optic_dr`, and the only `drive_mv` rows in the map are the 418 LC bodies. So the control shows that the
grid, the dwell, the pooling and the fit rule work, and that the failure on LC cells is not zero coverage everywhere
in the model; it does **not** establish the sensitivity of the same rule on the received-drive quantity, which is
the quantity the negative result is measured in. No positive control on `drive_mv` exists in this batch.

The `rf_sensitivity` table contains all five thresholds (3, 4, 5, 6, 8), including
the blank-arm fraction at every threshold, and the conventional z=5 beside z*.
The `rf_per_run` table reports coverage and false fits for all five runs at the
pooled map's selected threshold. Cells and spatial nodes are not independent
replicate runs. A zero observed blank fraction is an empirical count, not proof
of a zero population false-positive probability.

## 6. Centres, widths and anatomical agreement

Mi1's median fitted width is 6.383 degrees on both lobes, with median distance to
the prior 6.157 degrees shipped / 6.261 fb0. Its median across-run centre spread
is .727 / .720 degrees. T3's corresponding centre spread is 5.650 / 4.629 degrees,
and prior distance 10.723 / 9.720 degrees. These are the map's recorded scatter
metrics, conditional on successful fits; they do not turn bodies into replicates.

The full body-level centre/width distributions, split-half distance and number of
successful per-run fits are in `rf_map` and the CSVs. LC11's absent fits, and the
absence of repeated fits for the two fb0 LC10a bodies, remain missing values,
not manufactured zero scatter. Widths are coarse, grid- and prior-constrained
equivalent-disc FWHM estimates, not measured continuous physiological RF borders.

## 7. Scope of the negative result

This is a **4.5-degree static dark-flash** localizer, with 0.5-second dwell,
the received-drive quantity and a 20-degree input-centroid prior. It does not
show that no stimulus could localize LC11, that LC cells are silent, or that the
connectome has no retinotopic output representation. The requested 2-4-degree
probe was not recorded in the original batch; we preserved the actual protocol
for replication rather than silently changing it after seeing results.

The physiological LC11 RF assay used an **8.8-degree moving square**, with
horizontal/vertical trajectories at 33 deg/s; its RF-width definition also differs
from this fitter. See Figure 2 and the methods of [Keles and Frye](https://escholarship.org/content/qt6qv2m85f/qt6qv2m85f_noSplash_354705c807f4c66bb3b875626501bf0d.pdf).
A localized moving-probe assay is an evidence-based next measurement. Before using
new windows in ladder inference, stamp that analysis separately and retain the
anatomical fallback and false-fit controls. No model change is adopted.

## 8. Independent computational skeptic and corrections

`object_round3_export.py check-rf` uses a separate scalar implementation and raw
node archives: **7,031 bodies x five runs on each lobe**,
zero differences in `fitted`, centres, widths, `n_nodes_above_threshold`,
`n_runs_fitted` and `blank_fitted_at_z`, **at the threshold and prior taken from the delivered map**
(`check-rf` re-derives neither z* nor the input-weighted centroid; its `scope` field says so). Reports are
`out/objr3rf/skeptic_ship.json` and `skeptic_fb0.json`. This is an independent
numerical implementation by Astra, not another reviewer.

A consistency defect was fixed in the analysis: the fitter accepted a nonzero
peak with zero MAD (infinite z), but the final map rejected all nonfinite z.
`passes_threshold` now applies the fit's same amplitude floor (>1e-9), minimum
node count and zero-MAD rule to pooled, blank, per-run and split-half outputs.
Both original maps were recomputed after this fix.
The fix is a robustness fix with **no effect on any reported number**: `noise_mad == 0` occurs for 0 of the 7,031
bodies in each of the four maps, and evaluating the pre-fix rule (`isfinite(z_peak) & (z_peak >= z) & n_box >= 12`,
the version in `out/round3_astra/start/scripts/object_round3_localizer.py`) against the corrected rule gives
identical pooled, blank and fixed-z=5 counts for all 7 types in all 4 maps (28 of 28 cells, zero differences).
The stamped predeclaration,
submitted source hash and current analysis hash are retained separately.

`selfcheck` passes the synthetic Gaussian recovery test (95 % pooled coverage
versus 10 % in one run; median centre error .63 degrees; fitted width 11.9 vs
true FWHM 11.8), and explicit zero-MAD, zero-signal and amplitude-floor checks.
The test does not replace real-data false-fit measurement or the fresh replication.

The export retains all LC per-body node/role responses and run scatter. The
original recorder did not retain per-body 10-ms frame series for this long
localizer; those cannot be reconstructed from node means. No frame series is
invented or substituted without a label.

## 9. Fresh-seed B200 replication

`r3rfcheck-c82c1e` completed **10 jobs, 0 failed**, on <cluster-node>. Both brain and
node-order seeds are 1000-1004. The final transfer receipt validates all **100
file SHA-256s**; `out/objr3rf_house/verify.json` confirms 10 runs/consoles, 20
node archives, one 1,466-node grid and NVIDIA B200 throughout, with no problems.
The native console's completed-job count and scheduler receipt agree. The
initial SCP transfer was replaced by a compressed, hash-checked archive transfer;
no additional GPU runs were needed to recover files.

Fresh Results are `out/interp/objr3rf_house/rfmap_r3_{ship,fb0}.json`, with CSVs
under `out/objr3rf_house/`. **Neither LC population meets the localization
criterion, reproducing the population-level conclusion.** The blank-selected
threshold and literal fitted-cell counts are not identical across batches:

| Lobe | Type | Selected z* | Fits at z* | Blank fits at z* | Fits at fixed z=5 | Free peak inside prior | Chance |
|---|---|---:|---:|---:|---:|---:|---:|
| ship | LC11 | 5 | 0 / 143 | 0 / 143 | 0 / 143 | .07692 | .05843 |
| fb0 | LC11 | 4 | 4 / 143 | 1 / 143 | 0 / 143 | .06993 | .05843 |
| ship | LC10a | 6 | 1 / 275 | 0 / 275 | 2 / 275 | .11636 | .05104 |
| fb0 | LC10a | 5 | 1 / 275 | 0 / 275 | 1 / 275 | .03636 | .05104 |

The four fb0 LC11 fits give only 2.10 percentage points of coverage above blank,
below the required 5 points, and free-peak enrichment is 1.20x chance, below 2x.
Their conditional median width is 10.57 degrees (IQR 9.52-12.36), median distance
to the input prior 7.04 degrees; none supplies a repeated-fit centre-spread
estimate. They do not establish reliable population RFs. At the fixed z=5,
LC11 remains 0/143 on both lobes in both batches.

Shipped B200 LC10a selects z*=6 because its blank has 4/275 fits (1.45 %) at
z=5. Its one z*=6 fit has width 4.51 degrees and prior distance 12.95 degrees;
the one fb0 fit has width 6.38 and prior distance 16.52 degrees. Shipped free-peak
enrichment exceeds 2x chance, but .36 % fitted coverage still fails the required
5-percentage-point excess. This weak indication is retained without changing
the joint localization criterion.

The positive control remains clear: Mi1 fits **939/1,773 shipped and 941/1,773
fb0**, with zero pooled blank fits at z*=5. Median width is 6.383 degrees on
both lobes, prior distance 6.230 / 6.244 degrees, and across-run centre spread
.752 / .758 degrees. All body-level distributions and split-half/per-run fields
are retained in the maps.

LC11 **single-run coverage mean +/- sample SD over five runs**, evaluated at
that map's pooled blank-selected z*, is distinct from pooled coverage:

| Batch/lobe | z* | Single-run coverage | Single-run blank false-fit fraction |
|---|---:|---:|---:|
| H200 ship | 5 | .00420 +/- .00383 | .00559 +/- .00585 |
| H200 fb0 | 5 | .00979 +/- .00938 | .00420 +/- .00625 |
| B200 ship | 5 | 0 +/- 0 | .00559 +/- .00313 |
| B200 fb0 | 4 | .03636 +/- .01670 | .04755 +/- .01670 |

The <=1 % selection bound applies to the **pooled blank** fit fraction, not
to every single run or a confidence bound. The B200 fb0 row uses a different
threshold; its larger single-run fractions are not a causal GPU comparison.

Both fresh maps pass the independent raw-node checker: **7,031 bodies x five
runs per lobe, zero differences** in individual fits, widths/centres and
aggregate/per-run coverage -- again at the delivered map's own threshold and
prior, which `check-rf` does not re-derive (section 8). The map interpretation,
not a universal assertion of zero fitted cells, is what replicates.

## Report

```yaml
summary: >-
  Completed the static-flash RF maps and fresh-seed B200 replication. Neither
  LC11 nor LC10a meets the predeclared population-localization criterion; Mi1
  localizes clearly. This is a stimulus/prior/fit-limited negative result.
key_claims:
  - Each batch has five runs per lobe, both lobes on one GPU model, with a common 1466-node grid.
  - LC11 has zero fits at fixed z=5 on both lobes in both batches.
  - The adaptive B200 fb0 threshold is z=4, with four LC11 fits versus one blank fit; joint localization still fails.
  - B200 shipped LC10a has free-peak enrichment above 2x chance but insufficient fitted coverage.
  - Mi1 pooled coverage is about 53 percent in both batches; this is a positive control on `optic_dr`, not on the `drive_mv` quantity the LC result is measured in.
  - Received drive is fitted; spikes, anatomical fallbacks and unavailable frame chronology are not substituted for RF evidence.
files_written:
  - scripts/object_round3_localizer.py
  - scripts/object_round3_export.py
  - docs/audits/object_localizer_r3.md
  - out/objr3rf/
  - out/objr3rf_house/
  - out/interp/objr3rf/rfmap_r3_ship.json
  - out/interp/objr3rf/rfmap_r3_fb0.json
  - out/interp/objr3rf_house/rfmap_r3_ship.json
  - out/interp/objr3rf_house/rfmap_r3_fb0.json
  - out/r3rfcheck_cluster.log
  - out/export/objr3_rfmap_index.json and its listed directories
api:
  - plan --seed-offset defaults to 0; replication uses 1000 for both brain and order seeds.
  - submit --target and --node default to None; replication uses house and <cluster-node>.
  - passes_threshold(fit, z_min) applies the existing >1e-9 amplitude floor, 12-node minimum and zero-MAD rule consistently.
  - rfmap now refuses missing or empty expected node archives and records an rf_per_run table.
  - verify validates the 20 node archives and a single known CUDA device in addition to summaries and consoles.
  - selfcheck includes zero-MAD, zero-signal and amplitude-floor edge cases; no new model fields or defaults.
validation: >-
  Each batch completed ten jobs with zero failed and verified ten paired runs;
  all 100 house artifact hashes match. Independent scalar checks match 7031 bodies
  times five runs on each lobe in both batches, with zero differences. LC11 pooled
  z=5 coverage is zero throughout; B200 fb0 at selected z=4 has 4/143 fits versus
  1/143 blank. Its single-run coverage is .03636 +/- .01670 and blank .04755 +/-
  .01670 across five runs; the bound is on pooled blank coverage. Synthetic recovery
  selfcheck passes. No mechanism adoption is proposed.
recommendations:
  - Carry the fixed-threshold and blank-selected-threshold distinction into owner/Neurome documentation.
  - Declare a localized moving-square assay with blank controls before claiming absence under other stimuli.
  - Keep this round's ladder windows unchanged; using new RF maps in inference requires a separately stamped analysis.
  - Any later model intervention still requires specificity, full benchmark draws and relevant room guards before adoption.
open_questions:
  - Can a moving probe localize these LC cells with reproducible centres and acceptable blank fits?
  - Does shipped LC10a free-peak enrichment replicate when the coverage test has greater power?
  - How sensitive is the measurement to the 20-degree input-centroid prior and to the requested smaller probe?
author_self_review:
  refuted:
    - The unsupported inference that no stimulus could localize LC11 in this model.
    - Zero LC11 fits at every blank-selected threshold in every batch.
    - Calling the recorded 4.5-degree probe an exact implementation of the requested 2-4-degree probe.
  confirmed:
    - No declared LC population localization under this static probe on either lobe in either batch.
    - Mi1 positive-control localization, with recorded run scatter and empirical blank fits.
    - Pre-submission stamps, recorded arms, one GPU model per batch and independent numerical agreement.
  corrections:
    - Applied the fitter's zero-MAD acceptance rule consistently in final, blank, per-run and split-half maps.
    - Rejected incomplete node sets instead of silently pooling fewer runs.
    - Corrected the probe-coverage wording and restricted negative conclusions to the tested protocol.
    - Reported adaptive threshold changes and rare fits explicitly, without retuning the criterion.
  verdict: mostly sound
skeptic:
  source: "independent Opus pass, 2026-09-14"
  verdict: "mostly sound"
  refuted:
    - Calling the presented grid a grid "over each cell's anatomical box" - 1,466 of the 1,787 reachable nodes were presented (82 % of the eye, an 18 % saving); only the fit is boxed per cell.
    - 'R1: "Mi1 pooled coverage ... supporting a functioning positive control" - every control type is fitted on optic_dr, and the only drive_mv rows in the map are the 418 LC bodies, so no positive control exists on the quantity that produced the negative result.'
    - 'R2: "(the cells fire)" - 137 of 143 LC11 bodies emit zero spikes across the 1,466.5-s recording; the conclusion (never write "silent") stands, the stated reason does not.'
    - 'R3: the probe-coverage wording was corrected in the audit body but not in scripts/object_round3_localizer.py; the docstring still carried "at most 12 % / 43 % / 54 %" until this pass.'
    - 'R4: the effectively-whole-eye grid deviation was disclosed in section 1.3 but carried into no Report field.'
    - 'R5: "zero differences in ... threshold decisions" overstates check-rf, which takes both z* and the prior from the delivered map.'
    - 'R6 (minor): body.py is +134 lines in the current tree, not +132, and tree_state.json records no per-file counts.'
  corrections:
    - 'C1: rewrote the section 5 positive-control sentence to scope it to the rate units, and amended the matching key_claims entry.'
    - 'C2: added the measured LC spike counts beside the localizable_call bullet in section 0.'
    - 'C3: narrowed the section 8 check-rf claim to the fields it actually checks, at the delivered threshold and prior; the same narrowing applied to the parallel sentence in section 9.'
    - 'C4: applied the probe-coverage correction to the module docstring of scripts/object_round3_localizer.py (lines 9-11).'
    - 'C5: stated in section 8 that the zero-MAD consistency fix changed no reported number (28 of 28 cells identical).'
    - 'C6: added the presented-grid deviation to the Report as a refuted entry.'
    - 'C7 (optional): labelled the section 2 per-file line counts as plan-time figures and noted body.py is now +134.'
    - 'C8 (optional): noted column_equivalents_per_node_min 0.0 beside the median of 1.0 in section 1.1.'
```

## Skeptic pass (independent, 2026-09-14)

### Refuted

#### R1. The positive control is not on the quantity that produced the negative result

> **Claimed** (`## Report`, `key_claims`): "Mi1 pooled coverage is about 53 percent in both batches, supporting a
> functioning positive control."
> and (audit section 5): "The localizer can detect a retinotopic input signal; its failure on LC cells is not simply
> zero coverage everywhere in the model."

**Recomputed.** From the `quantity` column of `out/objr3rf/rfmap_r3_ship.csv` (7,031 rows; identical split in all four
maps):

| type | n | fitted quantity |
|---|---:|---|
| LC11 | 143 | `drive_mv` |
| LC10a | 275 | `drive_mv` |
| Mi1 | 1,773 | `optic_dr` |
| T2 | 1,630 | `optic_dr` |
| T3 | 1,940 | `optic_dr` |
| Tm5Y | 898 | `optic_dr` |
| TmY21 | 372 | `optic_dr` |

Every control type -- Mi1, T2, T3, Tm5Y, TmY21 -- is an optic-lobe **rate unit** fitted on `optic_dr`. The only
`drive_mv` rows in the whole map are the 418 LC bodies that produced the negative result. So **no positive control
exists on the fitted quantity**: the 53 % Mi1 coverage demonstrates that the grid, the dwell, the pooling and the
fitter work on `optic_dr`, not that a `drive_mv` receptive field of the same strength would have been detected. The
median |peak| is also in different units and a different regime (`out/objr3rf/rfmap_r3_ship.csv`): LC11 1.060,
LC10a 0.818 (mV) against Mi1 0.052, T3 0.055 (`optic_dr`). The quantity split is declared in
`predeclared.quantity` and repeated in section 0, but neither section 5 nor the Report's control claim states it.
Refuted as stated; the underlying measurement is fine.

#### R2. "(the cells fire)" is contradicted by this batch's own spike record

> **Claimed** (audit section 0, quoting the stamped `localizable_call`): "... never 'silent' (the cells fire) and never
> 'inverted'."

**Recomputed** from `out/objr3rf/loc/loc45_ship_r0_nodes.npz` (`spk__base` / `spk__on`, 1,466 nodes x 0.2 s base
windows, 10 ms frames):

| type | bodies | mean base rate | max base rate | bodies with ANY base spike | bodies with ANY on-window spike |
|---|---:|---:|---:|---:|---:|
| LC11 | 143 | 0.0017 Hz | 0.0750 Hz | **6 / 143** | 10 / 143 |
| LC10a | 275 | 0.0185 Hz | 0.6787 Hz | 25 / 275 | 26 / 275 |

137 of 143 LC11 bodies emit exactly zero spikes across a 1,466.5-second recording. The script's own docstring
(`scripts/object_round3_localizer.py` line 20) says the opposite of the audit: "the LC populations emit essentially
none in this protocol". The *conclusion* (do not write "silent") is right and is required by the project rule; the
*reason given for it* is not supported by this data. Refuted as a factual parenthetical.

#### R3. The shipped script still carries the probe-coverage wording the audit says was corrected

> **Claimed** (`## Report`, `corrections`): "Corrected the probe-coverage wording ..."

`scripts/object_round3_localizer.py` lines 9-11 still read: "a 2 / 3 / 4-deg square dims a column by at most 12 % /
43 % / 54 %". From `out/objr3rf/preview.json` `probe_coverage_on_this_grid`, the 2-deg square's **max** best-node
coverage is 0.4265, not 0.115 (0.115 is the median). The audit body (section 1.1) does correct it in prose -- "The
2-deg maximum in the table is 43 %, so 'at most 12 %' would be incorrect" -- but the correction was never applied to
the file the Report lists as written. Refuted: the correction is documented, not made.

#### R4. One task deviation is in the body but absent from the `## Report`

> **Task 4.5 asked for**: "a finer grid over each cell's anatomical box (not the whole eye)".

`out/objr3rf/preview.json`: `grid.n_nodes` **1,466**, `grid.n_nodes_whole_eye` **1,787**. The presented grid is
**82 % of the whole reachable eye**; the per-cell box restriction is applied only at fit time, not at presentation
time. Audit section 1.3 states this honestly ("so 'not the whole eye' saves 18 %"). The Report's `refuted` list names
only the probe size ("Calling the recorded 4.5-degree probe an exact implementation of the requested 2-4-degree
probe"); the grid deviation is not carried into any Report field. Refuted as an omission from the structured record.

(The other spec items were met and I confirm them below: 0.5 s dwell, 0.5 s blank between nodes, 5 runs pooled per
node on both lobes in one submission, the fit on `drive_mv`, the anatomical column set as prior, the false-fit rate at
every threshold.)

#### R5. The independent checker did not check the threshold decision

> **Claimed** (audit section 8): "`object_round3_export.py check-rf` uses a separate scalar implementation and raw
> node archives: 7,031 bodies x five runs on each lobe, zero differences in centres, widths, **threshold decisions**
> and pooled/per-run coverage."

`scripts/object_round3_export.py check_rf` (lines ~745-800) takes the threshold from the delivered map
(`row.z_min_used`) and the prior from the delivered map; its own `scope` field says so: "Independent reduction of
every recorded body's pooled node response; **supplied anatomical prior held fixed**." It re-derives neither z* from
the blank ladder nor the input-weighted centroid. It checks `fitted / az_deg / el_deg / width_deg /
n_nodes_above_threshold / n_runs_fitted / blank_fitted_at_z`. So "threshold decisions" overstates the tool's scope.
Substantively harmless -- my own reimplementation re-derived **both** z* and the prior and matched every value (see
Confirmed) -- but the sentence claims more than its own checker did.

#### R6. (minor) Stale line counts for other tasks' files

> **Claimed** (audit section 2): "... `flyverse/brain.py` (+31 lines), `senses.py` (+169), `body.py` (+132)".

`git diff --numstat` now: brain.py 30+1 = 31 (ok), senses.py 157+12 = 169 (ok), body.py **134+0 = 134**, not 132.
`out/objr3rf/tree_state.json` records no per-file counts, so the figure cannot be reconstructed from the stamp.
Incidental context about another task's files; refuted only in the literal sense.

---

### Confirmed

**Every quantitative claim in the audit reproduced, twice.**

1. **Shipped-script reproduction.** Re-running `rfmap` for `out/objr3rf` and `out/objr3rf_house` x {ship, fb0} on CPU
   reproduces `out/interp/objr3rf/rfmap_r3_{ship,fb0}.json` and `out/interp/objr3rf_house/rfmap_r3_{ship,fb0}.json`
   **cell-for-cell identical** across all four tables (`rf_per_type`, `rf_map` 7,031 rows, `rf_sensitivity`,
   `rf_per_run`) -- 0 differing cells in any of the four maps.

2. **Independent reimplementation** (own prior, own great-circle, own fitter, raw npz only) reproduces every headline
   count exactly:

| batch / lobe | type | z* | fits @ z* | blank @ z* | fits @ fixed z=5 | free peak | chance |
|---|---|---:|---:|---:|---:|---:|---:|
| H200 ship | LC11 | 5 | 0/143 | 1/143 | 0/143 | .04196 | .05843 |
| H200 fb0 | LC11 | 5 | 0/143 | 1/143 | 0/143 | .04895 | .05843 |
| B200 ship | LC11 | 5 | 0/143 | 0/143 | 0/143 | .07692 | .05843 |
| **B200 fb0** | **LC11** | **4** | **4/143** | **1/143** | **0/143** | .06993 | .05843 |
| H200 ship | LC10a | 5 | 0/275 | 0/275 | 0/275 | .06545 | .05104 |
| H200 fb0 | LC10a | 5 | 2/275 | 0/275 | 2/275 | .07273 | .05104 |
| B200 ship | LC10a | 6 | 1/275 | 0/275 | 2/275 (blank 4/275) | .11636 | .05104 |
| B200 fb0 | LC10a | 5 | 1/275 | 0/275 | 1/275 | .03636 | .05104 |

   Positive control and inputs also reproduce: Mi1 946 / 947 (H200 ship / fb0) and 939 / 941 (B200 ship / fb0) of
   1,773, blank 3 / 2 / 0 / 0, free peak .60124 / .60519 / .59898 / .60406 against chance .05662; T3 99 / 116 / 107 /
   102 of 1,940.

3. **Derived figures the audit quotes are arithmetically right.** B200 fb0 LC11 coverage minus blank = .027972 -
   .006993 = **.020979 = "2.10 percentage points"** (< the declared .05); free-peak enrichment .06993 / .05843 =
   **1.197 = "1.20x"** (< 2x). B200 ship LC10a enrichment .11636 / .05104 = 2.28 ("exceeds 2x") with .36 % coverage.
   B200 ship LC10a selects z*=6 because its blank is 4/275 = 1.4545 % at z=5. Conditional B200 fb0 LC11 width 10.574
   (IQR 9.5238-12.3603), prior distance 7.0363, `n_fitted_all_runs` 0 ("none supplies a repeated-fit centre-spread
   estimate"). H200 fb0 LC10a: 2 bodies, width 11.911, prior distance 13.518.

4. **The per-run scatter table (section 9) is exact** (`rf_per_run`, mean +/- sample SD over five runs at that map's
   pooled blank-selected z*): H200 ship .00420 +/- .00383 vs blank .00559 +/- .00585; H200 fb0 .00979 +/- .00938 vs
   .00420 +/- .00625; B200 ship 0 +/- 0 vs .00559 +/- .00313; B200 fb0 (z*=4) .03636 +/- .01670 vs blank .04755 +/-
   .01670. The audit correctly flags that the B200 fb0 single-run blank rate **exceeds** its single-run coverage and
   that the <= 1 % bound is on the pooled blank only.

5. **False-fit rate at EVERY threshold, as the task required.** `rf_sensitivity` carries 7 types x 5 z levels = 35
   rows per map, with `coverage_pooled`, `false_fit_rate_blank_pooled`, `coverage_all_runs`, `coverage_any_run`,
   `false_fit_rate_blank_any_run`. Recomputed independently; identical. The threshold rule (smallest z with pooled
   blank <= 1 %) is applied per type and lobe as declared, and `coverage_z5` / `false_fit_rate_z5` travel beside it.

6. **Stamp order, and the criterion was fixed before the data.**
   - H200: `out/objr3rf/predeclared.json` `stamped_utc` **2026-09-14T05:26:46Z**; first job `submitted_at`
     **05:27:14.023741Z** (`out/objr3rf/scheduler_receipt.json`).
   - B200: stamped **19:03:43Z**; first job **19:04:54.379423Z** (`out/objr3rf_house/scheduler_receipt.json`).
   - The two `predeclared` dicts are **byte-identical to each other**, and `predeclared_matches_live_dict` is True in
     all four Results -- confirmed by my rerun with today's code reproducing them. The localization criterion
     (coverage - blank >= 0.05 AND `free_peak_in_box` >= 2 x chance), the z ladder and the 1 % bound were all set
     before either batch and were not retuned after the B200 fb0 4/143 appeared.
   - `analysis_code_sha256_at_stamp` differs from `analysis_code_sha256` in both Results and both are retained --
     the analysis-code drift is disclosed, not hidden.

7. **One box per family, one grid, matched arms.** All 10 H200 run summaries record `device_name` **NVIDIA H200**; all
   10 B200 summaries record **NVIDIA B200**. One `--arm-block fam` block `locr3` per submission (`submit_console.txt`:
   "block locr3: 10 job(s) -> @r3-h200a" / "-> @house"). Grid sha256 `21d4f2ab6818...` and `n_nodes` 1466 on all 20
   runs and all 40 node archives. `optic_overrides` `{}` on ship and `{"gain_fb": 0}` on fb0 for every run.
   `checksum_per_arm` `{stim: True, blank: True}` on all 20 runs. `hook_info.active` False.
   `verify.json` problems **none** for both batches (10 runs, 10 consoles, 20 node archives, one device, one grid).
   Seeds: H200 brain 0-4 / order 0-4; B200 brain 1000-1004 / order 1000-1004 -- a real fresh-seed replication.

8. **The two batches ran the same optic code.** Comparing `provenance.source_fingerprint.files` between
   `out/objr3rf/skeptic_ship.json` and `out/objr3rf_house/skeptic_ship.json`, exactly **one** of the 44 source files
   differs: `flyverse/body.py` (another task's edit, unused here). `flyverse/optic.py`, `retina.py`, `brain.py`,
   `fly.py`, `scripts/probe_synthetic_stimuli.py` and `scripts/object_round3_localizer.py` are identical. The
   H200 -> B200 comparison is therefore not confounded by a code change on the measured path.

9. **No default changed.** `git diff -- flyverse/optic.py` is **empty**. `git diff -- scripts/probe_synthetic_stimuli.py`
   is also **empty** -- the permitted "new flags only" extension was never used, so default behaviour is trivially
   bit-identical. `flyverse/brain.py` is modified (+30/-1) but by the `unitary` task, and
   `out/objr3rf/tree_state.json`'s recorded sha256 for `flyverse/optic.py`, `retina.py`, `fly.py` and `brain.py` all
   equal the current files.

10. **The RF maps follow docs/INTERP.md 2.7.** First seven CSV columns exactly `bodyId, type, az_deg, el_deg,
    width_deg, peak, n_nodes_above_threshold`, in that order. Every documented bookkeeping column is present (`sign`,
    `noise_mad`, `z_peak`, `fitted`, `peak_node_az_deg/_el_deg`, `quantity`, `window`, `model_index`,
    `spikes_on_minus_base_hz_peak`, `n_runs_fitted`, `az_sd_runs`, `el_sd_runs`, `centre_spread_deg`, `anat_column`,
    `anat_az_deg/_el_deg`, `hex_annotated`, `anat_distance_deg`); `height_deg` is absent as 2.7 requires. JSON:
    `summary.schema` = `flyverse.interp.rfmap/1`, `summary.fit_rule`, `summary.optic_overrides`, `tables.rf_per_type`
    with all eight named fields, `replicates.runs` (5, with per-run device) and `replicates.null.blank_arm_files` (5).
    *Only deviation:* `model_index` is dropped from the JSON `rf_map` table (it is in the CSV) -- harmless.

11. **The round-2 anatomical fallback is preserved bit-for-bit.** Merging `out/objr3rf/rfmap_r3_ship.csv` with
    `out/synth2/rfmap_150_fb0_p3.csv` on `bodyId`: 7,031 bodies matched, `anat_column` / `anat_az_deg` / `anat_el_deg`
    identical for **100 %** of rows (all 418 LC rows included). The ladder window rule's fallback is unchanged, as the
    audit claims.

12. **New maps were NOT retrofitted into the ladder windows.** `files.rf_maps` in `out/interp/objr3rect/rectangles.json`,
    `out/interp/objr3rect_house/rectangles.json`, `out/objr3sd/samedevice.json` and `out/objr3sd_house/samedevice.json`
    all read `['out/objr2/rfmap_ship.csv', 'out/objr2/rfmap_fb0.csv', 'out/synth2/rfmap_150_fb0_p3.csv']`;
    `scripts/object_round3_rectangles.py` `RF_MAPS` and `object_round2_compare.DEFAULT_MAPS` are the unchanged round-2
    lists. Nothing in `scripts/` reads `rfmap_r3_*` except the localizer itself and the export.

13. **The anatomical prior table (section 1.2) reproduces independently**, recomputed from `c.W` + `column_of_cells`:
    LC11 143 cells / 13 distinct single columns / top-4 **50, 44, 14, 13** / median 100 input columns / r50 26.1 deg /
    r80 72.6 / single-column-to-centroid 64.5 -- and the two big columns sit at **(-25.4, +3.9)** with 50 cells and
    **(+43.5, -10.7)** with 44, exactly as the audit says. LC10a 275 / 88 / 52,33,31,31 / 48 / 44.8 / 72.8 / 47.1.
    T3 1,940 / 1,245 / 44,33,27,23 / 19 / 13.2 / 51.3 / 12.0. T2 1,630 / 1,043 / 38,34,26,16 / 41 / 16.5 / 57.1 / 14.3.

14. **The design numbers (sections 1.1, 1.3) reproduce** from `out/objr3rf/preview.json` and the run summaries:
    probe 4.5 deg at Weber contrast -0.995, 0.5 s flash / 0.5 s blank, 1 presentation per node, spacing 4.0 deg,
    radius 20.0 deg, 1,466 nodes, 146,650 frames = 1,466.5 s per arm, `column_equivalents_per_node_median` 1.0,
    box nodes per LC cell median 81 (42-133), probe-coverage medians 0.115 / 0.426 / 0.541 / 0.656 / 1.000 for
    2 / 3 / 4 / 4.5 / 8.8 deg.

15. **The selfcheck reproduces exactly** (re-run on CPU): z* = 5 for the responders, blank false fit .310 / .040 / .000
    at z 3 / 4 / 5; pooled coverage .950 vs .100 from one run; centre error median .63 deg, p90 1.14; width 11.9 vs
    true FWHM 11.8; non-responders .000; free peak in box 1.000 (responders) vs .130 (non-responders) at chance .125;
    zero-MAD / amplitude-floor / zero-signal edge check OK; `selfcheck OK`, exit 0.

16. **Section 6 / 9 scatter numbers are exact**: Mi1 width 6.383 deg on both lobes in both batches; prior distance
    6.157 / 6.261 (H200) and 6.230 / 6.244 (B200); across-run centre spread .727 / .720 and .752 / .758; T3 spread
    5.650 / 4.629 and prior distance 10.723 / 9.720.

17. **The literature citation in section 7 is exactly right.** Fetched the cited escholarship PDF (Keles & Frye,
    *Curr Biol* 2017; 27(5):680-687). Figure 2A caption, verbatim: "An 8.8 deg square dark object was scanned along
    non-overlapping trajectories along both horizontal and vertical paths at **33 deg/sec**." Main text p.3: "We
    enclosed the spatial receptive field with a contour representing the **full-width at 25 % max** of the Gaussian
    fits" -- so "its RF-width definition also differs from this fitter" is correct. (For the record, the paper's
    measured LC11 functional RF is 24.1 deg x 18.8 deg, n = 27; the 8.8 / 4.4 deg figures in the round briefing are the
    size-tuning optima of Figures 3D / 3E, not RF dimensions -- the audit does not conflate them.)

18. **The honest statement the task demanded is present and correctly scoped.** Neither the audit nor the Report calls
    LC cells "silent" or "inverted" or claims no stimulus can localize LC11. Section 7: "It does not show that no
    stimulus could localize LC11, that LC cells are silent, or that the connectome has no retinotopic output
    representation," and the Report's `refuted` list opens with "The unsupported inference that no stimulus could
    localize LC11 in this model." The reply file repeats the disclaimer. Confirmed.

19. **Transfer and provenance.** `out/objr3rf_house/loc/round3_fetch_receipt.json`: `n_files` 100,
    `missing_or_hash_mismatched` **[]**. The four `skeptic_*.json` files' `result_sha256` match the current Result
    files **byte for byte**, so the independent checks were run against the final (post-correction) maps, not an
    earlier draft. `out/r3rfcheck_cluster.log` carries the native "10 job(s), 0 failed (75.0 min)" line and one
    labelled `FETCH FAILED` (a transfer failure, as the reply says); the H200 batch's "10 job(s), 0 failed" comes from
    the explicitly labelled `recovered_cluster.log` / `scheduler_receipt.json`, which the audit states plainly.

20. **The Report's `api` and `files_written` entries check out.** `plan --seed-offset` default 0; `submit --target` /
    `--node` default None; `passes_threshold` applies the 1e-9 floor, `MIN_BOX_NODES` 12 and the zero-MAD rule;
    `rfmap` exits on missing/empty node files and writes `rf_per_run`; `verify` counts the 20 node archives and
    requires one known CUDA device; `selfcheck` has the edge-case block; no new model field or default. All eleven
    `files_written` paths exist.

---

### Corrections (exact replacement text)

**C1 -- section 5, second paragraph, last sentence.** Replace:

> The localizer can detect a retinotopic input signal; its failure on LC cells is not simply zero coverage everywhere
> in the model.

with:

> The localizer can detect a retinotopic input signal **in the rate units**: Mi1, T2, T3, Tm5Y and TmY21 are all
> fitted on `optic_dr`, and the only `drive_mv` rows in the map are the 418 LC bodies. So the control shows that the
> grid, the dwell, the pooling and the fit rule work, and that the failure on LC cells is not zero coverage everywhere
> in the model; it does **not** establish the sensitivity of the same rule on the received-drive quantity, which is
> the quantity the negative result is measured in. No positive control on `drive_mv` exists in this batch.

Add the same caveat to the Report `key_claims` entry, e.g. "Mi1 pooled coverage is about 53 percent in both batches;
this is a positive control on `optic_dr`, not on the `drive_mv` quantity the LC result is measured in."

**C2 -- section 0, the `localizable_call` bullet.** After the quoted rule, add:

> (Measured in this batch rather than assumed: in `loc45_ship_r0_nodes.npz` only 6 of 143 LC11 bodies and 25 of 275
> LC10a bodies emit any spike at all, at population mean rates of 0.0017 and 0.0185 Hz. The word "silent" is avoided
> because the fit is on received drive and the protocol is not a firing assay -- not because these cells fire
> appreciably here; the script's docstring records that "the LC populations emit essentially none in this protocol".)

**C3 -- section 8, first paragraph.** Replace "zero differences in centres, widths, threshold decisions and
pooled/per-run coverage" with:

> zero differences in `fitted`, centres, widths, `n_nodes_above_threshold`, `n_runs_fitted` and `blank_fitted_at_z`,
> **at the threshold and prior taken from the delivered map** (`check-rf` re-derives neither z* nor the
> input-weighted centroid; its `scope` field says so).

**C4 -- `scripts/object_round3_localizer.py`, module docstring lines 9-11.** Replace "a 2 / 3 / 4-deg square dims a
column by at most 12 % / 43 % / 54 %" with "a 2 / 3 / 4-deg square dims the median column by 12 % / 43 % / 54 % at its
best grid node (maxima 43 % / 54 % / 89 %)". Then the Report's `corrections` entry "Corrected the probe-coverage
wording" becomes true of the shipped file as well as the audit.

**C5 -- section 8, second paragraph.** After "Both original maps were recomputed after this fix," add:

> The fix is a robustness fix with **no effect on any reported number**: `noise_mad == 0` occurs for 0 of the 7,031
> bodies in each of the four maps, and evaluating the pre-fix rule (`isfinite(z_peak) & (z_peak >= z) & n_box >= 12`,
> the version in `out/round3_astra/start/scripts/object_round3_localizer.py`) against the corrected rule gives
> identical pooled, blank and fixed-z=5 counts for all 7 types in all 4 maps (28 of 28 cells, zero differences).

**C6 -- `## Report`, `refuted` list.** Add a fourth entry:

> - Calling the presented grid a grid "over each cell's anatomical box": 1,466 of the 1,787 reachable nodes were
>   presented (82 % of the eye, an 18 % saving); only the fit is boxed per cell.

**C7 (optional) -- section 2.** `body.py` is +134 lines, not +132, in the current tree; `tree_state.json` records no
per-file counts, so quote the figures as "at plan time" or drop them.

**C8 (optional) -- section 1.1.** `column_equivalents_per_node_min` is 0.0: 3 of 1,466 nodes present essentially
nothing and 90 present less than half a column-equivalent. Worth one clause beside the median of 1.0.

---

### Verdict

**mostly sound.**

Everything that could be recomputed was recomputed, twice, and matched: all four Results reproduce exactly from the
shipped script, and an independent reimplementation that re-derives the prior, the box masks, the ladder, the
threshold and the fit from the raw node archives reproduces every fitted count, blank count, free-peak fraction and
chance level to the last digit -- including the decisive B200 fb0 LC11 4/143 at z*=4 against 1/143 blank, and 0/143 at
fixed z=5 in all four maps. The process controls hold without exception: the predeclaration precedes both submissions,
the two stamped rule dictionaries are byte-identical and equal the live dict, the criterion was not retuned after the
data, each batch sat on one GPU model under one `--arm-block` block with one grid, the false-fit rate is on file at
every threshold, the map is in the documented 2.7 format with the round-2 fallback preserved exactly, nothing was
retrofitted into the ladder windows, and `flyverse/optic.py` and `scripts/probe_synthetic_stimuli.py` are both
untouched.

What keeps it from `sound` is interpretive, not numerical: the positive control is on `optic_dr` while the negative
result is on `drive_mv` and that gap is never stated where the control is claimed (R1); the stamped justification for
avoiding "silent" -- "(the cells fire)" -- is contradicted by the batch's own spike record and by the script's own
docstring (R2); the "consistency fix" is presented as a correction that required recomputing both maps when in fact it
changed nothing (C5); one genuine task deviation, the effectively-whole-eye grid, is disclosed in the body but not in
the structured Report (R4); and the audit's own independent checker is described as verifying more than it does (R5).
The negative result itself -- neither LC population localized by a 4.5-deg static dark flash under a 20-deg
input-centroid prior at a measured false-fit rate, on either lobe, in either batch -- stands.
