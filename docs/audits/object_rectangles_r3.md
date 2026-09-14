# The rectangle ladders, round 3 (`scripts/object_round3_rectangles.py`)

**Status: original H200 batch and fresh-seed B200 replication complete, verified and analyzed.
Sections 0-3 retain the submitted design; the results below correct its claim of matched retinal contrast.
The predeclaration is `out/objr3rect/predeclared.json`, stamped 2026-09-14T05:54:45Z,
before the first scheduler submission at 05:55:08Z.** Nothing is adopted; no default changes.

Read first: `docs/audits/object_baseline_r2.md` (the sphere ladder: LC11 12/12 null, LC10a's 30-deg member fails
Holm, 405 of 418 LC windows are anatomical boxes, the sphere's effective contrast NOT matched), `object_compare_r2.md`
3-9 (the rectify carrier figure is size-monotone and lives in the extremum over cells; the arm/box confound),
`object_synthetic_stimuli.md` 1 / 6.2 / 8 (the rectangle families, the elevation-0 retinal hole, the localizer's
false-fit rates), `docs/INTERP.md` 2.4 and 10.2 / 10.4 (the round-2 rules this batch is built on).

Files: `scripts/object_round3_rectangles.py` (`plan` / `submit` / `run-job` / `reduce` / `check-job` / `verify` /
`analyse` / `selftest`), `out/objr3rect/` (`batch.sh`, `jobs.json`, `predeclared.json`, `tree_state.json`,
`plan_console.txt`, `submit_console.txt`, then `syn/`, `verify.json`, `per_body.csv`), `out/objr3rect_cluster.log`,
`out/interp/objr3rect/rectangles.json` (the Result). The CPU smokes are `out/objr3rect_smoke/` (section 3); nothing in
them is a number of this experiment.

## 0. Predeclared reading rules (`out/objr3rect/predeclared.json`, stamped before submission)

The rules are the `PREDECLARED` dict of the script, dumped by `plan`; `analyse` copies the stamped file into the
Result, checks it against the live dict (`summary.predeclared_matches_live_dict`) and records the sha256 of the
analysis code at stamp time and at analysis time (`files.analysis_code_sha256*`, docs/INTERP.md 10.4 item 11).
Plainly: the reading rules are the stamped dict (unchanged, `predeclared_matches_live_dict` true); the analysis
code of the original batch was written after the stamp and its sha256 therefore differs
(`files.analysis_code_matches_stamp` is **false** for `out/objr3rect/` -- `16750e7b...` at stamp against
`cb77c4b4...` at analysis -- and true for the house batch), which is why both hashes are recorded. In
prose:

* **The question.** With retinal contrast matched by construction and height / width varied separately, does any LC
  population show the animal's size preference -- LC11: peak near height 8.8 deg at width 4.4 and near width 4.4 deg
  at height 8.8 (Keles & Frye 2017); LC10a: 15-30 deg (Schretter et al. 2024)? Is the upstream T2 / T3 figure still
  size-monotone? How does it compare with the sphere ladder?
* **Replicate unit.** One process = one (A, B) pair under one brain seed = one run; a cluster job is a container of
  sequential processes. Runs, not cells, not seeds.
* **Design.** Height ladder at width 4.4 deg (heights 2.2 / 4.4 / 8.8 / 15 / 30) and width ladder at height 8.8 deg
  (widths 2.2 / 4.4 / 8.8 / 15 / 30); dark (Weber -0.995) and bright (+0.995); the sphere ladder's arc (elevation 0,
  +-50 deg at 40 deg/s, 12 s after a 3-s settle); two lobes, `ship` (the shipped OpticParams) and `fb0`
  (`--optic gain_fb=0`, the deterministic control). **Six object runs per rectangle x contrast and six blank/blank
  nulls PER RECTANGLE** (their own seeds, 100 (i + 1) + k), so the members of a family share no null run (round 2's
  caveat 4 closed). The 4.4 x 8.8 rectangle is height 8.8 of the height ladder AND width 4.4 of the width ladder: one
  set of runs, a member of both families -- 9 distinct rectangles, 324 processes, ONE submission with
  `cluster_run.py --arm-block fam` (every job carries the token `fam_rect`, so both lobes and every rung sit on one
  box).
* **Why six.** The primary family has 5 members (the rungs, drive only): Holm's smallest threshold is 0.05 / 5 = 0.01;
  the exact-U floor is 0.0079 at 5 v 5 and 0.00216 at 6 v 6, so six keeps a fully separated member clear of the
  floor with margin and matches round 2's arm size.
* **Window rule.** The round-2 comparison's (`object_round2_compare.load_windows` over `out/objr2/rfmap_ship.csv`,
  `out/objr2/rfmap_fb0.csv`, `out/synth2/rfmap_150_fb0_p3.csv`): a body's window is its fitted box in the first map
  that fits it, else the anatomical column with the type's median fitted width, else the whole window; the window
  frames are those on which the rectangle overlaps the box (`|d az| <= (w + width)/2 and |d el| <= (h + height)/2`);
  a null run is windowed with the rectangle's size on its own identical track. **No map fits an LC11 body (0 of
  143): every LC11 window is an anatomical box, not a measured receptive field.**
* **Primary.** Per run the population MEDIAN over the type's windowed bodies of the per-body RF-windowed time-mean
  received drive (A - B, mV), `drive_median`. One family per (ladder x contrast x LC type) on the shipped lobe: 8
  families of 5 members, Holm within (m = 5); each member = the six object runs at that rung vs the six nulls of that
  rectangle; tie-aware exact permutation U (two-sided, every C(12, 6) = 924 assignment); verdict from
  `common.compare`. **`spikes_median` is a reported magnitude OUTSIDE the family** (it was exactly 0.000 in every
  run of round 2; a member constant by construction is not a test, docs/INTERP.md 10.2). Expectations: LC11 excess
  largest at height 8.8 (width ladder: width 4.4), falling by 15-30; LC10a largest at 15-30 on both ladders.
  Preference tests per family and statistic: Spearman rho of the per-run statistic against the varied dimension over
  the 30 object runs (20,000 label permutations, seed 0); small-minus-large = mean at {2.2, 4.4, 8.8} minus mean at
  {15, 30}; peak contrast = mean at the type's target rungs (LC11 hlad {8.8}, wlad {4.4}; LC10a {15, 30}) minus the
  rest, permutation p; the nulls' own Spearman against the rung beside each. **A size preference is CALLED only if at
  least one member reads `result` AND survives Holm; the animal's SHAPE is shown only if, in addition, the peak
  contrast is positive with p_perm <= 0.05.** `fb0` rows are magnitudes (its null has SD ~0), never tabulated beside
  a Holm call.
* **Secondary, exploratory.** T2 / T3 / Tm5Y / TmY21: `diff_signed_best_cell` (a within-run population MAXIMUM),
  `diff_signed_mean` and `diff_abs_best_cell_mean`, whole-window, kept apart; Holm within each 5-rung set. **Beside
  every max-over-cells row, in the same table: `abs_drive_median`** (per-body RF-windowed median of |window mean of
  dr_A - dr_B| over the type's windowed bodies) and `abs_drive_median_rf` (fitted bodies only). "Size-monotone" =
  Spearman rho > 0 with p_perm <= 0.05 on `diff_signed_best_cell` AND on `abs_drive_median`. LC11 / LC10a
  `drive_mean`, `drive_max`, the old headline `diff_max_over_cells_mean_mv` and `diff_rate_hz_max_cell`, Holm within
  each 5-rung set, with the per-body median always in the same table.
* **Effective contrast.** Per rectangle from the synthesised radiance (the array handed to `FlyBrain.vision`): the
  object's Weber contrast (fixed +-0.995), the per-frame extreme relative luminance change (median over frames), the
  mean relative change over the changed set, the maximum column coverage, the column-equivalents per frame, the
  fraction of frames in which no column is covered > 5 %, the columns dimmed / brightened > 50 % per frame. Stated
  up front: contrast-matched in the object's Weber contrast; the per-COLUMN extreme is capped by coverage where a
  rectangle is narrower than a column's 4.5-deg acceptance (a 4.4-deg width never covers a whole column, max
  coverage 0.885), and the elevation-0 arc runs through the sparsest patch of this retina.
* **Sphere comparison.** Magnitudes only, at the matching sizes (4.5 ~ 4.4 x 4.4; 8.8 ~ 8.8 x 8.8; 15; 30); two
  batches on different boxes are never compared row by row.
* **Devices.** The realised `device_name` of every run is recorded; the verifier raises a problem if the two lobes'
  device sets differ. **Nothing is adopted.**

## 1. What was built (`scripts/object_round3_rectangles.py`)

* `plan` writes `batch.sh`, `jobs.json`, the stamped `predeclared.json` and `tree_state.json` (commit, porcelain
  status, sha256 of every simulation source AND of the analysis sources). 12 jobs = one per (lobe, seed): the 9
  rectangles x 2 contrasts at that seed plus the 9 nulls of that seed index = 27 processes; every job line is
  `mkdir -p ... && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' &&
  python scripts/object_round3_rectangles.py run-job --fam fam_rect --job <lobe>_s<k> ...`, so the job's exit code
  is `check-job`'s (every expected output present, every console with `device cuda` and a `reduced` line, no
  Traceback). `cluster_run.py` prints no exit-masking warning for these lines.
* `submit` = one `scripts/cluster_run.py --name objr3rect --minutes 60 --arm-block fam <12 lines> --fetch
  out/objr3rect/` call, console teed to `out/objr3rect_cluster.log`.
* **`reduce` (on the box, after every `record`).** A 12-s run's two per-cell recordings are ~64 MB and never leave
  the box (docs/INTERP.md 10.4 item 12; round 2's 10 GB of unfetched ladders is why this family was never analysed).
  `reduce` writes `<stem>_reduced.npz`: the per-cell whole-window means of both arms (`family_stats`' inputs), the
  per-type population means per frame, and the **sweep-lattice sums**: the rectangle's azimuth is a triangle wave in
  angle at 40 deg/s sampled every 10 ms, so every frame sits on one of P = 251 lattice azimuths 0.4 deg apart
  (+-50 deg), and the sum and count of every cell's value at each lattice azimuth reproduce the time-mean over ANY
  azimuth-interval window exactly -- and the round-2 window rule IS an azimuth interval (x a constant elevation
  test) on this arc. LC drive and spikes are kept per arm in float64 (418 cells), the upstream `optic_dr` as the A - B
  sum in float32 (6,613 cells); ~5-7 MB per run against 64. Every `reduce` checks the identity on 40 random boxes
  against the raw frames of the same run and stores the deviation (`check`); `verify` refuses a run above 1e-6.
* `verify` (CPU): the cluster log's `0 failed` line, every expected output of `jobs.json`, per run the summary /
  provenance / console device, both checksums, arms vs the null flag, seed / rectangle / contrast / arc vs the file
  name, optic overrides and the resolved `gain_fb` vs the lobe, `hook_info.active` false, the reduced file's params
  and frame count, consoles counted against runs, and the device set per lobe (one known GPU each, equal across
  lobes).
* `analyse` (CPU): `load_windows` -> per-body windowed statistics from the lattice (vectorised over bodies) ->
  `pop_stats` -> the families, Holm, the preference tests, the effective-contrast table, the sphere comparison; the
  Result records the stamped predeclaration, whether it matches the live dict, and the analysis code's sha256.
  `tie_aware_exact_u`, `contrast_perm` and `spearman_perm` are the round-2 definitions vectorised (identical U / rho
  / p on random inputs with and without ties, checked in section 3).
* `selftest --raw <stem>`: the lattice reconstruction against `object_round2_baseline.windowed_body_stats` on a raw
  run under the real window rule, every LC and upstream body.

## 2. The batch (`objr3rect-e9a8f3`, 12 jobs = 324 processes, ONE submission, ONE block)

`out/objr3rect/submit_console.txt`: `block rect: 12 job(s) -> @r3-h200a (round-robin, largest first)`; all 12
jobs queued on **r3-h200a** (NVIDIA H200) behind 62 jobs of the other workflow already on that box (r3-h200b held 46,
r3-h200d 7, r3-h200c 0 at submission). The block rule places a block on a target regardless of its load, so the
batch waited in the box's queue rather than being resubmitted (the task's cluster rule). Tree: commit `6ec2de1a`
plus other tasks' uncommitted edits (`flyverse/body.py`, `brain.py`, `senses.py`, `motor.py`, `batch_*.py`,
`flyverse/data/expected_responses.csv`, ...; `out/objr3rect/tree_state.json` lists them with sha256), so
`provenance.source_fingerprint`, not the commit, identifies the code of every run. **CPU bit-identity of the shipped
tree against a clean `git archive HEAD` checkout on this stimulus** (4.4 x 8.8 dark, seed 0, 50 frames after a 0.2-s
settle, `--allow-cpu`): `drive_mv`, `optic_dr` and `spike_count` max |diff| **0.0** in both arms, NaN patterns
identical -- the other tasks' edits do not touch this path on the CPU (the CUDA path inherits the usual caveat that
bit-identity is a CPU claim).

## 3. Validation before the batch (CPU, `out/objr3rect_smoke/`)

* **The exact process command** (`_record_cmd` with `--allow-cpu`, 0.5 s / 0.2-s settle) for one object run
  (`ship_w044_h088_dark_s0`) and one fb0 null (`fb0_w044_h088_null_s100`): both `record` and `reduce` exit 0,
  checksums true in both arms, `check-job --allow-cpu` 4 expected / 0 problems (`out/objr3rect_smoke/cpu/`).
* **`selftest` on a stored GPU run** (`out/synth/rect_dark_h088_w044`, 300 frames, +-30 deg): lattice 151 azimuths,
  0.4 deg apart, 1-2 frames each; lattice-reconstructed vs per-frame windows over all 418 LC bodies:
  `drive_diff_win` max |dev| **0.0**, `n_frames_win` 0, `spikes_diff_win` 0.0, `drive_a_win` 0.0; over 4,840
  upstream bodies `drive_diff_win` 2.1e-09 (float32 sums), `n_frames_win` 0. **PASSED.**
* **The vectorised statistics against round 2's**: `compare_row` on seven cases (continuous, tied, separated,
  all-zero, constant null, 3 v 6, empty arm) -- identical `U_tie`, `p`, `n_tied`, `z`, `p_scipy_exact` and verdict;
  `contrast_perm` identical p on three cases; `spearman_perm` identical rho and p on four cases including ties, NaN /
  NaN on a constant input (never a floor p). The vectorised `per_body_from_reduced` takes 0.2 s per run against 80 s
  for the per-body loop; a full `analyse` on 30 runs runs in 17 s.
* **A fake mini-batch** (`out/objr3rect_smoke/fake/`: `out/synth`'s 4.4 x 8.8 dark / bright, 8.8 x 8.8 dark and the
  blank/blank null copied under six seeds each): `verify` flags exactly what a copy should fail (seed vs name,
  half-span 30 vs 50, the null copied under the wrong rectangle) and passes the rest; `analyse --no-verify` runs end
  to end, every populated member `undetermined` (SD-0 copies) or `underpowered` (absent rungs), `Result.check()`
  none, `predeclared_matches_live_dict` false (no stamp in that directory), the effective-contrast rows reproduce
  `object_synthetic_stimuli.md`'s 0.885 max coverage for the 4.4-deg width with `*_sd_runs` 0.

## 4. Completed batch and artifact verification

`objr3rect-e9a8f3` completed **12 jobs, 324 paired runs** on NVIDIA H200.
`out/objr3rect/verify.json` checks all 324 consoles, summaries, provenance records,
reduced arrays and radiance captures, with no missing outputs or other problems.
The largest recorded lattice reconstruction error is **5.42e-9**, below 1e-6.
The dead polling client's status was recovered by actual job ID into
`scheduler_receipt.json` and explicitly labeled `recovered_cluster.log`:
**12 jobs, 0 failed**. The original machine-outage retries do not count as extra runs.

A lossless LC column selection, `out/objr3rect_lc/`, was packed **CPU-only, with no GPU work**, from the complete
per-frame originals in `out/objr3rect_raw/` (`FETCHED.txt`: all 972 npz are the originals; the set was already
local and size-verified): 324 paired frame archives, all 143 LC11 and 275 LC10a bodies, each with the sha256 of
both source recordings (`index.json` generator line:
`object_round3_export.py pack-rect --raw out/objr3rect_raw --out out/objr3rect_lc`). `verify-lc` reports zero
problems. The lattice reduction discarded nothing: it is a reduction of data that remains available in
`out/objr3rect_raw/`, and the per-body statistics reproduce from those frames directly.
The lattice remains the exact input to the predeclared analysis.

The fresh replication is `r3rectcheck`, one 12-job submission on <cluster-node>, with
object seeds 2000-2005 and disjoint per-rectangle null seeds 2100-2905.
`out/objr3rect_house/` retains its own predeclaration and source snapshot.

## 5. LC preference and upstream companions

`out/interp/objr3rect/rectangles.json` contains all comparisons, per-run values,
preference tests and sphere comparisons; `out/objr3rect/per_body.csv` retains the
body-level window measurements. **All 40 shipped-lobe primary LC comparisons are
null. None of the eight families shows the declared animal-like preference.**
Each comparison has six object runs and six independent nulls; the 4.4 x 8.8
recording belongs to both ladders but is not counted twice as an independent run.

| Ladder | Contrast | LC11 minimum Holm p | LC10a minimum Holm p |
|---|---|---:|---:|
| height, width 4.4 | dark | .66017 | 1 |
| height, width 4.4 | bright | 1 | 1 |
| width, height 8.8 | dark | 1 | .89827 |
| width, height 8.8 | bright | .66017 | 1 |

An isolated negative target contrast in the dark width ladder does not establish
an inverse preference: there is no positive primary family call. LC10a's target
remains 15-30 degrees, not the smallest object. Spike medians stay descriptive;
the constant/small-null control is not evidence for a precise biological effect.

Neither T2 nor T3 meets the predeclared size-monotonicity criterion, which requires
a significant positive rank relation for **both** the maximum and the per-body
windowed median. The shipped dark height ladder gives T2 rho .850 / .316 and
T3 .528 / .071 (maximum / median); the dark width ladder gives T2 .920 / -.523
and T3 .359 / -.591. Maxima alone would give a misleadingly simple answer.
All run-level maxima and companions, including Tm5Y/TmY21 and bright stimuli,
travel together in the Result and export. The round-2 sphere and this rectangle
experiment both lack an established LC population preference, under different
stimulus geometry; a direct causal attribution to contrast is not supported.

Beside the sphere ladder (`out/interp/objr2/baseline.json`, 36 object runs, shipped lobe, recorded across
**two** device models — B200 and H200 — so magnitudes only): LC11 Spearman rho **-.21607** (p_perm .20614),
LC10a **+.02975** (p_perm .87181). Neither ladder produces a called preference on either statistic; the
rectangles add that the failure survives separating height from width and holding the object's Weber contrast,
and that at LC11's published shape the target contrast is if anything negative (H200 `wlad:dark` peak contrast
-.092474). Two batches on different boxes are not compared row by row.

## 6. Object contrast is fixed; retinal contrast is not fully matched

The predeclaration's `question` field, and the task text it came from, use the shorthand "retinal contrast matched
by construction". The radiance capture refutes that shorthand — but not the stamped design, whose
`effective_contrast` clause already declared the cap and its value before submission ("the per-COLUMN extreme
change is capped by coverage where a rectangle is narrower than a column's 4.5-deg acceptance (a 4.4-deg width
never covers a whole column: max coverage 0.885)"). The capture confirms that clause to three decimals.
Object Weber contrast is exactly
-0.995 or +0.995; fractional coverage of the retinal acceptance kernel changes
the effective signal for small rectangles. These are measured dark-stimulus
values, pooled over the identical stimulus captures from both lobes:

| Width × height (deg) | Max column coverage | Median extreme retinal Weber change | Median column equivalents | Frames with no column changed > 5 % |
|---|---:|---:|---:|---:|
| 2.2 × 8.8 | .54119 | -.34238 | .54119 | .2500 |
| 4.4 × 2.2 | .42649 | -.22826 | .17205 | .4000 |
| 4.4 × 4.4 | .65590 | -.53849 | .65590 | .2958 |
| 4.4 × 8.8 | .88530 | -.65262 | .88530 | .1542 |
| 4.4 × 15 | .88530 | -.65262 | 1.26172 | .0833 |
| 4.4 × 30 | .88530 | -.65262 | 4.11470 | .0000 |
| 8.8 × 8.8 | 1 | -.995 | 1.94265 | .0542 |
| 15 × 8.8 | 1 | -.995 | 3.75444 | .0000 |
| 30 × 8.8 | 1 | -.995 | 7.54119 | .0000 |

**Which rungs are actually matched.** Peak per-column effective contrast reaches the object's ±.995 only where the
width is ≥ 8.8 deg — that is, at three of the nine rectangles, all of them in the width ladder (8.8, 15, 30).
The whole height ladder runs at width 4.4 and is capped at coverage .88530 / extreme -.65262, so its three upper
rungs (8.8, 15, 30) are matched **to each other** but not to the width ladder, while its 2.2 rung (.42649 /
-.22826) and 4.4 rung (.65590 / -.53849) are matched to nothing. In the width ladder the 2.2 rung (.54119 /
-.34238) and the 4.4 rung (.88530 / -.65262) are each at their own level. Total retinal drive is not matched
anywhere and is not meant to be: column-equivalents rise 24× along the height ladder (.17205 → 4.11470) and 14×
along the width ladder (.54119 → 7.54119). At the smallest rungs the stimulus is also intermittent: 40 % of frames
at 4.4 × 2.2 and 25 % at 2.2 × 8.8 change no column by more than 5 %.

The effective-contrast table and per-run captures include bright stimuli and run
scatter. Saturated peak column contrast at large sizes does not equal constant
total retinal drive: covered area intentionally increases. The observed nulls
are valid for these stimuli; a strictly retinal-contrast-matched ladder was not
performed. Preserve the original stamped wording as a record, with this correction.

All 143 LC11 windows use the anatomical fallback. LC10a has 262 anatomical and
13 fitted windows under the inherited maps. Coverage by rung and window source
is exported explicitly; it must accompany the population statistics.
The arc enters only 55 of those LC11 boxes at height 2.2, 99 at height 8.8,
and 103 at height 30. Assignment coverage and actually windowed bodies are
different counts; all 418 LC bodies remain in the exported frame traces.

## 7. Literature and interpretation checked by the skeptic

Keles and Frye report LC11 peaks near vertical extent 8.8 degrees and width 4.4
degrees. Their Figure 3D height sweep held **width at 30 degrees**, however;
the requested width-4.4 height sweep here is an adapted assay, not an exact
reproduction of that protocol. Their RF measurement used a moving 8.8-degree
square. These distinctions are explicit in the [author-hosted primary paper](https://escholarship.org/content/qt6qv2m85f/qt6qv2m85f_noSplash_354705c807f4c66bb3b875626501bf0d.pdf).

The width ladder here is much closer to the published protocol than the height ladder: Figure 3E varied width at a
**fixed height of 8.8 deg**, which is exactly this ladder's fixed height (the tested widths differ only at the top
— 2.2 / 4.4 / 8.8 / 15 / 30 here against 2.2 / 4.4 / 8.8 / 18 / 35 / 70 / 210 there). Figure 3D varied height
2.2 / 4.4 / 8.8 / 18 / 35 / 73.2 at a fixed width of 30 deg; the width-4.4 height sweep run here has no published
counterpart, so LC11's "peak at height 8.8 with width 4.4" is an extrapolation across two separate experiments,
not a measured expectation. Separately, **Figure 3B shows that maximum contrast is not LC11's optimum**: reducing
OFF-object Weber contrast from 100 % to 30 % "nearly doubled the amplitude of the calcium response". Both ladders
here run at |Weber| .995, i.e. at the contrast the animal responds to *least* strongly — a protocol caveat that
belongs beside every null in §5.

LC10a's 15-30-degree width and height preference is supported by Figure 3a and
Extended Data Figure 2 of [Schretter et al.](https://www.nature.com/articles/s41586-024-08255-6).
Failure to show those shapes under this assay is a model/protocol finding; it
does not establish that these cells lack stimulus-driven responses generally.
No gain, threshold, fallback window or model parameter was tuned to the result.

## 8. Independent computational checks

The skeptic pass uses separate numerical implementations, not a second reviewer.
`out/objr3rect/skeptic_stats.json` checks **1,280 comparisons and 236 scored Holm
families, zero differences**. There are 256 family labels in total; descriptive
or entirely undefined families do not become scored tests. A first checker draft
under-counted declared family membership when p was undefined; that checker was
corrected to retain the declared m, with no change to the analysis or its p-values.
**That last claim is not independently checkable**: no draft and no diff of the
first checker is retained under `out/`, so a reader cannot verify it. The substantive
fact behind it -- 236 of the 256 family labels carry a defined Holm p -- is checkable
and reproduces. The predeclaration-before-submission check passes.

`skeptic_raw.json` checks **3,888 run-level values, zero differences**, reconstructing
LC medians directly from the lossless frame archives and upstream companions from
the verified lattice arrays. Source provenance and all 324 artifact checks support
the stated arms, run counts, common GPU model and unchanged optic defaults.

## 9. Fresh-seed B200 replication and final verdict

`r3rectcheck-6f0cbc`: **12 jobs, 0 failed; 324 runs and consoles**, all NVIDIA B200
on <cluster-node>. The standard SCP fetch was interrupted locally to use a compressed
archive. One size-matched partial file failed NPZ verification; the hash-checked
resume replaced it. The final receipt verifies **1,620 / 1,620 file SHA-256s**,
and `out/objr3rect_house/verify.json` reports zero problems (largest lattice check
error 6.77e-9). These were transfer retries, not simulation reruns.

The fresh Result is `out/interp/objr3rect_house/rectangles.json`.
**All 40 primary verdicts are again null; no declared animal-shape call.**
This does not mean every adjusted p exceeds .05: dark LC10a at width 15 degrees
has difference +.055055 mV, z = 2.17788 and Holm p = .021645. It fails
`common.compare`'s z ≥ 3 gate (`Z_RESULT`, declared by reference in the stamped
`primary.families` clause), hence remains `null` under that rule.
The same comparison in the H200 batch has the opposite sign: object -.02034 ± .03658 vs null -.00532 ± .03302,
diff **-.015022 mV**, z -.45490, p .699134, Holm 1. Across the two batches this rung carries no consistent
direction, which is the first thing a new declaration would have to survive.
That lower-threshold statistical evidence is retained, not hidden or promoted
after the fact. Further precision would require a separately declared experiment.

The joint size-monotonicity call is not stable across batches, in either direction, and neither batch's call
should be read as replicating or refuting the other (`docs/INTERP.md` 10.4 item 2: one-batch claims, no row-by-row
comparison across batches). Exactly one of the 16 declared families passes in each batch, and it is a different
family: on B200, `hlad:dark:T2` (max rho .67535, p .000150; median rho .57187, p .001350); on H200,
`hlad:bright:Tm5Y` (max rho .61544, p .000550; median rho .36218, p .049248), which fails on B200
(median rho .07625, p .691265). For `hlad:dark:T2` the two batches agree in sign on both statistics
(median rho +.31589 on H200, +.57187 on B200); what differs is whether the permutation p crosses .05
(.095095 vs .001350). Both borderline calls flip under a different permutation RNG seed (H200
`hlad:bright:Tm5Y` median p .049248 → .053047; B200 `hlad:bright:T2` .050797 → .047048 at seed 12345), so the
criterion is knife-edge at 30 runs and no upstream size-monotonicity conclusion is carried out of this round.
This upstream response still supplies no declared LC population shape.

Representative primary values below are **object and null run means +/- sample
SD, n=6 in each arm**, in mV. Each run's value is its median over windowed bodies.

| Cell/type and shape | H200 object | H200 null | B200 object | B200 null |
|---|---:|---:|---:|---:|
| LC11, 4.4 x 8.8 dark | -.08125 +/- .07873 | -.00329 +/- .10614 | -.03374 +/- .11344 | -.01990 +/- .05963 |
| LC11, 4.4 x 8.8 bright | .00659 +/- .08113 | -.00329 +/- .10614 | -.05344 +/- .14133 | -.01990 +/- .05963 |
| LC10a, 4.4 x 30 dark | -.00431 +/- .03301 | -.00760 +/- .02843 | -.02701 +/- .03024 | -.01150 +/- .02437 |
| LC10a, 30 x 8.8 dark | .00183 +/- .02988 | .01756 +/- .02243 | -.00711 +/- .02527 | -.00285 +/- .02423 |

The independent B200 checks also reproduce **1,280 comparisons / 236 scored
families and 1,728 preference statistics with zero differences**. Original and
fresh results remain separate, not pooled across devices or source snapshots.
`skeptic_contrast.json` independently checks 756 radiance-derived statistics in
each batch, with zero discrepancies. Its luminance metric retains the specified
denominator floor and exclusion of frames without a >5% changed column.
The original raw-body checker and per-run lattice identity checks support the
reduction; the fresh run-vector checks and rerun independently test the claims.

## Report

```yaml
summary: >-
  Completed and exported both rectangle ladders. All 40 shipped primary verdicts
  are null on H200 and on the fresh B200 replication under the predeclared rule;
  retinal contrast is not fully matched for the small shapes.
key_claims:
  - Six object runs per distinct shape x contrast x lobe (216 runs) and six blank/blank null runs per shape x lobe (108 runs), 324 paired runs per batch. A shape's six nulls serve both its dark and its bright family, so the dark and bright families of a ladder share their null arm and are not independent of each other.
  - No declared LC11 or LC10a animal-shape preference in either batch.
  - B200 LC10a dark width 15 has Holm p .021645 but z 2.17788, below common.compare's z >= 3 gate; the same comparison on H200 has the OPPOSITE sign (diff -.015022 mV, z -.45490, Holm 1).
  - H200 T2 and T3 maxima do not establish size-monotonicity in their windowed-median companions.
  - The joint size-monotonicity call is unstable in both directions at 30 runs - exactly one of the 16 declared families passes in each batch and it is a different one (B200 hlad:dark:T2, H200 hlad:bright:Tm5Y), both flip under a different permutation seed, and no upstream size-monotonicity conclusion is carried out of this round.
  - Every LC body has a frame trace in the original-batch export; actual window coverage is explicit.
files_written:
  - scripts/object_round3_rectangles.py
  - scripts/object_round3_export.py
  - docs/audits/object_rectangles_r3.md
  - out/objr3rect/
  - out/objr3rect_lc/
  - out/objr3rect_house/
  - out/interp/objr3rect/rectangles.json
  - out/interp/objr3rect_house/rectangles.json
  - out/r3rectcheck_cluster.log
  - out/export/objr3_rectangles_index.json and its listed directories
api:
  - plan and run-job --seed-offset defaults to 0; replication uses 2000.
  - submit --target and --node default to None; explicit house/<cluster-node> routing is supported.
  - Exporter check-rect-raw defaults to the original rectangle batch and lossless LC archive.
  - Exporter check-preference requires --result and --json; fixed 20000 object and 2000 null permutations.
  - Exporter check-contrast defaults to the original rectangle batch and its Result; all three paths are overridable.
validation: >-
  Original and house batches each completed 12 jobs with 0 failed and verified
  324 paired runs. Original raw checks covered 3888 values; each batch's independent
  statistics covered 1280 comparisons, 236 scored families and 1728 preference
  statistics, with zero discrepancies. LC11 4.4 x 8.8 dark object mean +/- run SD
  is -.08125 +/- .07873 mV (H200) and -.03374 +/- .11344 mV (B200), n=6 each;
  matched nulls are -.00329 +/- .10614 and -.01990 +/- .05963. No adoption is proposed.
recommendations:
  - Correct owner documentation to distinguish fixed object contrast from effective retinal contrast.
  - Preserve the predeclared effect gate and report the B200 LC10a adjusted-p/effect-size distinction.
  - Before any mechanism adoption, require separately declared specificity, ON/OFF and full benchmark draws plus relevant room guards.
open_questions:
  - Does a moving-probe RF map improve the anatomical-window coverage of these ladders?
  - Does the B200 LC10a 15-degree comparison persist with more independent runs under a new declaration? The H200 row at the same type, ladder, contrast and rung points the other way, so the question is open in both directions.
author_self_review:
  refuted:
    - The submitted assertion that retinal contrast is matched across every rectangle rung.
    - Treating the width-4.4 height ladder as the literal Keles-Frye Figure 3D protocol.
    - Inferring T2/T3 size-monotonicity from the maximum alone on the original batch.
  confirmed:
    - Predeclarations precede submissions; both lobes share one GPU model in each batch.
    - Null primary verdict pattern under the full declared rule reproduces at fresh seeds on <cluster-node>.
    - Original raw and both independent run-vector checks agree with the analyzed Results.
  corrections:
    - Documented the contrast and literature-protocol limitations without altering the stamped design.
    - The independent Holm checker now counts undefined members in the declared family size.
    - SHA-256 transfer checks replaced one partial file that a size-only check had missed.
  verdict: mostly sound
skeptic:
  source: "independent Opus pass, 2026-09-14"
  verdict: "mostly sound"
  refuted:
    - 'R1: "This CPU-only packing on the original box recovered frame traces omitted by the lattice reduction" - no frame trace was lost, the complete per-frame originals were already local, and no artefact records the packing host.'
    - 'R2: "B200 T2 in the dark height ladder does satisfy the declared joint size-monotonicity test ... the original H200 failure is not a stable negative finding" - the two batches agree in sign, the reverse non-replication (H200 hlad:bright:Tm5Y) is unreported, both calls flip under a different permutation seed, and the declared rule forbids the cross-batch row comparison.'
    - 'R3: "The radiance capture refutes that wording" - the stamped effective_contrast clause had already declared the coverage cap and its 0.885 value before submission; the capture confirms it to three decimals.'
    - 'R4: "Six object and six null runs per distinct shape, contrast and lobe" - nulls are blank/blank and carry no contrast: six nulls per shape x lobe (108), shared by the dark and bright families of that shape.'
    - 'R5: the B200 LC10a width-15 evidence was carried forward without its H200 counterpart, which has the opposite sign.'
  corrections:
    - 'C1: rewrote the section 4 LC-archive paragraph - CPU-only repack of complete local originals, no recovery and no host claim.'
    - 'C2: replaced the section 9 batch-sensitivity paragraph with the two-directional instability, the seed-12345 flips and the one-batch-claim rule.'
    - 'C3: rewrote the section 6 refutation as a refutation of the shorthand, not of the stamped design.'
    - 'C4: corrected the Report key_claims null-design line and stated that dark and bright families share their null arm.'
    - 'C5: inserted the H200 counterpart row in section 9 and named the gate as common.compare''s z >= 3 (Z_RESULT).'
    - 'C6: added the no-changed-column fraction column to the section 6 table and a paragraph saying which rungs are actually matched.'
    - 'C7: added the Figure 3E fixed-height point, the absence of a published width-4.4 height sweep, and Figure 3B''s contrast optimum to section 7.'
    - 'C8: delivered the predeclared sphere-vs-rectangle magnitudes in section 5 (LC11 rho -.21607, LC10a +.02975).'
    - 'C9: disclosed in section 0 that analysis_code_matches_stamp is false for the original batch and why.'
    - 'C10: marked the first-checker-draft sentence in section 8 as not independently checkable (no draft retained under out/).'
```

## Skeptic pass (independent, 2026-09-14)

### Refuted

#### R1. "This CPU-only packing on the original box recovered frame traces omitted by the lattice reduction" (§4)

**Quoted claim** (§4): "The full original recordings also supplied a lossless LC column selection,
`out/objr3rect_lc/` … **This CPU-only packing on the original box recovered frame traces omitted by the lattice
reduction**; no new GPU work ran there."

**What the files say.** `out/objr3rect_lc/index.json` records
`"generator": "scripts/object_round3_export.py pack-rect --raw out/objr3rect_raw --out out/objr3rect_lc"`, and every
`*_lc.json` names its sources as `out/objr3rect_raw/<stem>_{stim,blank}.npz`. `out/objr3rect_raw/FETCHED.txt` reads
"All 2268 remote files local with matching size; **the 972 npz are the ORIGINALS** (the slim step found nothing to
drop)". The local raw set was being written from 10:58 local; the LC archives were written 12:10–12:27. I verified
two of the recorded source hashes against the local files
(`ship_w044_h088_dark_s0_stim.npz` → `d80d5846…`, `_blank.npz` → `f3ff3060…`, archive → `cbce15b4…`: all match).

Nothing in `out/objr3rect_lc/` (no `FETCHED.txt`, no host line in `pack.txt`, no box path in `index.json`) supports
"on the original box", and the premise is wrong either way: no frame trace was lost. The complete per-frame
originals were already local, which is exactly what `indep_raw.py` used to reproduce the per-body statistics.
The lattice is a *reduction* of data that remained available, not a discard.

#### R2. "The B200 T2 dark-height companion is size-monotone; that secondary negative does not replicate" (Report `key_claims`; §9)

**Quoted claim** (§9): "One upstream conclusion is batch-sensitive: **B200 T2 in the dark height ladder does satisfy
the declared joint size-monotonicity test** … **Thus the original H200 failure of the T2 companion test is not a
stable negative finding.**"

**Recomputed** (`indep_secondary.py`; 0 mismatches against either Result, so the arithmetic is not in dispute):

| `hlad:dark:T2` | max rho | p(seed 0) | median rho | p(seed 0) | joint call |
|---|---:|---:|---:|---:|---|
| H200 `out/interp/objr3rect/rectangles.json` | .84963 | .000050 | **+.31589** | **.095095** | False |
| B200 `out/interp/objr3rect_house/rectangles.json` | .67535 | .000150 | **+.57187** | **.001350** | True |

Three reasons the claim does not stand as written:

1. **The two batches agree in sign on both statistics.** The median companion is *positive* in both
   (+.316 and +.572); what differs is whether a Monte-Carlo p crosses .05. "Does not replicate" describes a
   sign reversal or a disappearance; this is a threshold crossing.
2. **The reverse non-replication is not reported.** Of the 16 declared (ladder × contrast × type) size-monotone
   tests, **exactly one passes in each batch, and it is a different one**: H200 `hlad:bright:Tm5Y`
   (max rho .61544 p .000550, median rho .36218 p .049248 → True) fails on B200 (median rho .07625, p .691265).
   The audit reports only the family that gained the call, not the one that lost it. The honest summary is that
   the joint criterion is unstable in *both* directions at 30 runs.
3. **The declared rule forbids the cross-batch row comparison the claim rests on.** The stamped predeclaration cites
   `docs/INTERP.md` 10.4 item 2, which reads "never compare a new batch to an old one row by row, and quote a
   one-batch claim as a one-batch claim". The two batches differ in device *and* seeds *and* source snapshot.
   No declared rule resolves a disagreement between them; the declared rule says to quote each as a one-batch claim.

Further: the call is knife-edge under the permutation RNG itself. Re-drawing the 20,000 label permutations at
seed 12345 moves H200 `hlad:bright:Tm5Y`'s median p from **.049248 → .053047** (the True call becomes False) and
B200 `hlad:bright:T2`'s from **.050797 → .047048** (False becomes True). A binary declared call resting on a
Monte-Carlo p within ±.004 of .05 cannot carry the weight "not a stable negative finding" puts on it.

#### R3. "The radiance capture refutes that wording" (§6)

**Quoted claim** (§6): "The submitted predeclaration says 'retinal contrast matched by construction.'
**The radiance capture refutes that wording.**"

**What the stamped file says.** `out/objr3rect/predeclared.json` (stamped 2026-09-14T05:54:45Z) already declared the
cap, with the number, *before* submission — `predeclared.effective_contrast`:

> "Stated up front: the family is contrast-matched in the object's Weber contrast; the per-COLUMN extreme change is
> **capped by coverage where a rectangle is narrower than a column's 4.5-deg acceptance (a 4.4-deg width never covers
> a whole column: max coverage 0.885)**, and the elevation-0 arc runs through the sparsest patch of this retina …
> so the column-equivalents per rung are the number to read beside every small-rung null"

My independent recomputation from the radiance returns max coverage **.88530** for every width-4.4 rung — i.e. the
capture *confirms* the stamped effective-contrast clause to three decimals. The loose phrase is the shorthand in the
predeclaration's `question` field and in the handoff's task text ("this family IS contrast-matched"), not the
design's contrast declaration. Presenting the result as a refutation of the predeclaration overstates a
discrepancy between two fields of the same stamped file, and quietly takes credit for a caveat the design already
carried.

#### R4. "Six object and six null runs per distinct shape, contrast and lobe" (Report `key_claims`)

**Refuted by the design and by the data.** Nulls are blank/blank and therefore have **no contrast**: there are six
null runs per **shape × lobe**, shared by the dark and the bright family of that shape. In
`out/objr3rect/per_body.csv` the null rows carry `contrast_name` = NaN, and the run counts are 216 object runs
(9 shapes × 2 contrasts × 6 seeds × 2 lobes) + **108** null runs (9 shapes × 6 seeds × 2 lobes) = 324.
The audit's own §9 table shows it: LC11 4.4 × 8.8 *dark* and *bright* quote the identical H200 null
`-.00329 ± .10614`. The stamped design says this correctly ("six blank/blank nulls **PER RECTANGLE**"); only the
Report's summary line mis-states it. It is not cosmetic — it means the dark and bright families of a ladder are
**not independent of each other**, which no part of the audit states.

#### R5. The B200 LC10a width-15 evidence is presented without its H200 counterpart (§9, Report `key_claims`)

**Quoted claim** (§9): "dark LC10a at width 15 degrees has difference +.055055 mV, z = 2.17788 and Holm p = .021645
… **That lower-threshold statistical evidence is retained, not hidden or promoted after the fact.**"

The numbers are exactly right (see C2). What is missing is the same comparison in the other batch, which the audit
has in hand and does not quote. Recomputed from `per_body.csv`, ship lobe, wlad, dark, LC10a, rung 15.0:

| batch | object mean ± SD | null mean ± SD | diff | z | p (exact U) | Holm |
|---|---:|---:|---:|---:|---:|---:|
| H200 | -.02034 ± .03658 | -.00532 ± .03302 | **-.015022** | -.45490 | .699134 | 1.000000 |
| B200 | +.03708 ± .02645 | -.01797 ± .02528 | **+.055055** | +2.17788 | .004329 | .021645 |

The H200 batch has the **opposite sign** at the same type, ladder, contrast and rung. Retaining the B200 p as
"lower-threshold statistical evidence" while omitting a same-protocol batch that points the other way is selective;
and `open_questions` ("Does the B200 LC10a 15-degree comparison persist with more independent runs under a new
declaration?") reads as if the question were open in one direction only. The H200 row is the strongest available
evidence that it will not persist.

---

### Confirmed

**C1. 40/40 primary verdicts null, in both batches — reproduced independently.**
`indep_primary.py`, from `per_body.csv` with my own median / exact-U / Holm / verdict code, reproduces all 40
primary members in each batch: max |deviation| over `diff`, `z`, `p`, `p_holm`, `stim_mean`, `null_mean`,
`stim_sd`, `null_sd`, `U_tie` = **1.332e-15** (H200) and **1.665e-15** (B200); **0** verdict mismatches; all 40
`null` in each. The §5 table of family-minimum Holm p reproduces exactly:

| ladder | contrast | LC11 | LC10a | (audit) |
|---|---|---:|---:|---|
| height, w 4.4 | dark | .660173 | 1.000000 | .66017 / 1 ✓ |
| height, w 4.4 | bright | 1.000000 | 1.000000 | 1 / 1 ✓ |
| width, h 8.8 | dark | 1.000000 | .898268 | 1 / .89827 ✓ |
| width, h 8.8 | bright | .660173 | 1.000000 | .66017 / 1 ✓ |

`size_preference_called` and `animal_shape_shown` are False for all 8 families in both batches.

**C2. B200 dark LC10a, width 15 deg: diff +.055055 mV, z 2.17788, Holm p .021645, verdict `null`.** Exact, from
`per_body.csv` (`out/objr3rect_house/per_body.csv`) and from `out/interp/objr3rect_house/rectangles.json`.
The gate is real and declared by reference: the stamped `predeclared.primary.families` says "verdict =
common.compare (z on the null SD, exact U, p_floor)", and `flyverse/interp/common.py:61` sets `Z_RESULT = 3.0`.
(The numeral 3 appears in the code, not in the stamped JSON — worth saying "common.compare's z ≥ 3 gate" rather
than "the predeclared 3-null-SD effect gate".)

**C3. Every size-monotone number in §5 and §9.** `indep_secondary.py` rebuilt `diff_signed_best_cell` from the
per-cell reduced arrays and `abs_drive_median` from `per_body.csv`, then computed its own Spearman and permutation
p: **0 mismatches** against either Result over all 16 families × 4 numbers per batch. Specifically
H200 `hlad:dark` T2 .84963 / .31589 and T3 .52829 / .07080 (audit: ".850 / .316" and ".528 / .071");
H200 `wlad:dark` T2 .92043 / -.52285 and T3 .35946 / -.59093 (audit: ".920 / -.523", ".359 / -.591");
B200 `hlad:dark:T2` .67535 (p .000150) / .57187 (p .001350) (audit: ".67535 (p = .000150)" / ".57187 (p = .001350)").
The claim that neither T2 nor T3 meets the joint criterion on H200 is correct as literally stated.

**C4. All preference statistics.** `indep_pref.py` reproduces every object-arm preference number for all 8 primary
families in both batches exactly (Spearman rho and p, small-minus-large and p, peak contrast and p, null Spearman
rho). The 8 apparent mismatches per batch were all `null_spearman_p_perm`, and are explained: the analysis draws the
**null** arm's permutation p from 2,000 labels, not 20,000 (`scripts/object_round3_rectangles.py:865`). Recomputing
`hlad:dark:LC11` with 2,000 draws gives rho -0.106203573904659, p **0.5647176411794103** — the recorded value to the
last digit. (Minor: the 10× smaller Monte-Carlo budget on the null arm is in the code, not in the stamped design.)
The negative LC11 target contrast the audit flags in §5 is real: H200 `wlad:dark:LC11` peak contrast
**-0.092474, p_perm .002800**; it does not reproduce on B200 (-0.057892, p .121294), and the audit is right to
decline to read it as an inverted preference.

**C5. The §6 effective-contrast table.** `indep_contrast.py`, from `syn/*_radiance.npz` only: all 18
rectangle × contrast rows × 7 statistics reproduce, max |deviation| **1.776e-15**. Every dark row of §6 is exact
(.54119 / -.34238 / .54119; .42649 / -.22826 / .17205; .65590 / -.53849 / .65590; .88530 / -.65262 / .88530;
.88530 / -.65262 / 1.26172; .88530 / -.65262 / 4.11470; 1 / -.995 / 1.94265; 1 / -.995 / 3.75444;
1 / -.995 / 7.54119). Across the 12 captures per rectangle (6 seeds × 2 lobes) the run SD of `max_coverage` is
≤ 2.3e-16 — the stimulus is seed- and lobe-independent, as the audit says.

**C6. Predeclarations precede submissions, from the scheduler's own record.**
`out/objr3rect/predeclared.json` stamped **2026-09-14T05:54:45Z**; earliest `submitted_at` in
`out/objr3rect/scheduler_receipt.json` is **2026-09-14T05:55:08.355875Z** (23 s later; all 12 jobs submitted
05:55:08–05:55:17). House: stamped **19:03:45Z**, earliest `submitted_at` **19:05:13.918542Z** (88 s later; 12 jobs
19:05:13–19:05:20). `summary.predeclared_matches_live_dict` is True in both Results.

**C7. One box per family, from the per-run device.** All 324 `syn/*_prov.json` of the original batch report
`device_name` **NVIDIA H200** on host **42d57a7efe61** (162 ship + 162 fb0); all 324 of the house batch report
**NVIDIA B200** on host **<cluster-node>**; torch 2.11.0+cu128 in both. `optic.gain_fb` is 0.5 in every `ship` run and 0 in
every `fb0` run; `optic_hook_info.active` is false in all 648. `out/objr3rect/submit_console.txt` shows one
`--arm-block` block ("block rect: 12 job(s) -> @r3-h200a"), and the house console one block "-> @house".

**C8. Max-over-cells statistics travel with their per-body medians.** In both Results: 160 `diff_signed_best_cell`
rows, and **0** of them lack `abs_drive_median` / `abs_drive_median_rf` at the same
(lobe, ladder, contrast, type, rung) key; 80 `diff_max_over_cells_mean_mv` rows, **0** without `drive_median`.
Roles: 40 primary, 40 control_fb0, 80 reported_magnitude, 320 exploratory, 800 secondary_exploratory = 1280.

**C9. No shipped default changed.** `git diff -- flyverse/optic.py` is **empty** and the file is not in
`git status`. (`flyverse/brain.py` is modified, but only by the *unitary* thread: an opt-in
`LIFParams.w_syn_by_nt: dict | None = None` that executes nothing when None. Not this task's file.)

**C10. The reduction is faithful to the raw frames.** `indep_raw.py` rebuilt the per-body window from the
per-frame recordings (`out/objr3rect_raw/*_{stim,blank}.npz`) and the stimulus track for four runs spanning both
lobes and both contrasts: `n_frames_win` identical for all 418 LC bodies in every run, max |Δ drive_diff_win|
**2.2e-16**, and the population medians match `tables.per_run.drive_median` to ≤ 7e-18 (e.g.
`ship_w044_h088_dark_s0` LC11 +0.034196216 raw vs +0.034196216 Result; `ship_w044_h300_bright_s4` LC11
-0.218352374 both). The audit's §4 lattice-error figures also check out: `verify.json`
`reduce_check_max_dev` = **5.418604e-9** (H200, quoted 5.42e-9) and **6.773255e-9** (B200, quoted 6.77e-9),
`problems: []`, 324 runs / 324 consoles, `expected_missing: []` in both.

**C11. Transfer and batch bookkeeping.** `out/objr3rect_house/round3_fetch_receipt.json`: **1620** files,
`missing_or_hash_mismatched: []` — the "1,620 / 1,620 file SHA-256s" claim holds. Both
`scheduler_receipt.json` files show 12 jobs, status `completed`, exit code 0. `out/objr3rect/syn` and
`out/objr3rect_house/syn` each hold 1620 files (648 json + 648 npz + 324 txt).

**C12. Window coverage.** From `per_body.csv`: LC11 **143 / 143 anatomical**, 0 fitted; LC10a **262 anatomical
+ 13 fitted** (7 `rf:rfmap_ship`, 6 `rf:rfmap_fb0`) — exactly §6's "262 anatomical and 13 fitted". Windowed-body
counts per rung reproduce (LC11 55 at height 2.2, 99 at 8.8, 103 at 30; audit §6 the same).

**C13. The Keleş & Frye 2017 claims in §7, checked against the primary source the audit links.** The Figure 3
legend of the author manuscript (escholarship PDF, p. 15) reads: "D and D′) LC11 is vertically size tuned.
**A 30° wide object was moved on the same horizontal trajectory, with varied vertical heights: 2.2°, 4.4°, 8.8°,
18°, 35°, 73.2°**"; "E and E′) … **An object of fixed height (8.8°) and varied width: 2.2°, 4.4°, 8.8°, 18°, 35°,
70°, 210°**". The Figure 2 legend (p. 12): "**An 8.8° square dark object was scanned along non-overlapping
trajectories along both horizontal and vertical paths at 33 °/sec.**" Body text (p. 5): "The optimum LC11 response
occurs for a vertical extent of 8.8°"; "For objects of increasing width, the response amplitude peaked near 4.4°".
So **all three of §7's factual claims are correct**, and consistent with the ledger rows
`lit.LC11.preferred_height_deg` / `lit.LC11.preferred_width_deg` in `flyverse/data/expected_responses.csv`
(which state "width held fixed" without the value). Two things follow that the audit does not say — see
Corrections 6 and 7.

**C14. Descriptive claims that hold.** `spikes_median` is exactly **0.0** in all 648 per-run rows of both batches
(so "spike medians stay descriptive" is right, and keeping them outside the family is right). Family accounting:
**256** family labels, **236** with a defined Holm p — matching `skeptic_stats.json`'s "1,280 rows_checked /
236 families_checked / 256 family_labels_seen, 0 differences", which I reproduce. `skeptic_raw.json` (3,888 values),
`skeptic_preference.json` (1,728) and `skeptic_contrast.json` (756, both batches) all record 0 differences, and my
own passes agree with them. The `api` block checks out (`--seed-offset` default 0 on both `plan` and `run-job`;
`submit --target` / `--node` have no default).

---

### Corrections (exact replacement text)

**1. §4, replace** "The full original recordings also supplied a lossless LC column selection, `out/objr3rect_lc/`:
324 paired frame archives … This CPU-only packing on the original box recovered frame traces omitted by the lattice
reduction; no new GPU work ran there." **with:**

> A lossless LC column selection, `out/objr3rect_lc/`, was packed **CPU-only, with no GPU work**, from the complete
> per-frame originals in `out/objr3rect_raw/` (`FETCHED.txt`: all 972 npz are the originals; the set was already
> local and size-verified): 324 paired frame archives, all 143 LC11 and 275 LC10a bodies, each with the sha256 of
> both source recordings (`index.json` generator line:
> `object_round3_export.py pack-rect --raw out/objr3rect_raw --out out/objr3rect_lc`). `verify-lc` reports zero
> problems. The lattice reduction discarded nothing: it is a reduction of data that remains available in
> `out/objr3rect_raw/`, and the per-body statistics reproduce from those frames directly.

(The host on which `pack-rect` ran is not recorded in any artefact — `pack.txt` has no host line, `index.json` and
every `*_lc.json` name only local relative paths — so "on the original box" cannot be checked and should not be
asserted; nothing turns on it, since the archive's recorded source hashes match the local raw files.)

**2. §9, replace** "One upstream conclusion is batch-sensitive: **B200 T2 in the dark height ladder does satisfy the
declared joint size-monotonicity test**, with maximum rho .67535 (p = .000150) and windowed-median rho .57187
(p = .001350), across 30 object runs. Thus the original H200 failure of the T2 companion test is not a stable
negative finding." **with:**

> The joint size-monotonicity call is not stable across batches, in either direction, and neither batch's call
> should be read as replicating or refuting the other (`docs/INTERP.md` 10.4 item 2: one-batch claims, no row-by-row
> comparison across batches). Exactly one of the 16 declared families passes in each batch, and it is a different
> family: on B200, `hlad:dark:T2` (max rho .67535, p .000150; median rho .57187, p .001350); on H200,
> `hlad:bright:Tm5Y` (max rho .61544, p .000550; median rho .36218, p .049248), which fails on B200
> (median rho .07625, p .691265). For `hlad:dark:T2` the two batches agree in sign on both statistics
> (median rho +.31589 on H200, +.57187 on B200); what differs is whether the permutation p crosses .05
> (.095095 vs .001350). Both borderline calls flip under a different permutation RNG seed (H200
> `hlad:bright:Tm5Y` median p .049248 → .053047; B200 `hlad:bright:T2` .050797 → .047048 at seed 12345), so the
> criterion is knife-edge at 30 runs and no upstream size-monotonicity conclusion is carried out of this round.

**3. §6, replace** "The submitted predeclaration says 'retinal contrast matched by construction.' **The radiance
capture refutes that wording.**" **with:**

> The predeclaration's `question` field, and the task text it came from, use the shorthand "retinal contrast matched
> by construction". The radiance capture refutes that shorthand — but not the stamped design, whose
> `effective_contrast` clause already declared the cap and its value before submission ("the per-COLUMN extreme
> change is capped by coverage where a rectangle is narrower than a column's 4.5-deg acceptance (a 4.4-deg width
> never covers a whole column: max coverage 0.885)"). The capture confirms that clause to three decimals.

**4. Report `key_claims`, replace** "Six object and six null runs per distinct shape, contrast and lobe; 324 paired
runs per batch." **with:**

> Six object runs per distinct shape × contrast × lobe (216 runs) and six blank/blank null runs per shape × lobe
> (108 runs), 324 paired runs per batch. A shape's six nulls serve both its dark and its bright family, so the dark
> and bright families of a ladder share their null arm and are not independent of each other.

**5. §9, after** "It fails the predeclared 3-null-SD effect gate, hence remains `null` under that rule." **insert:**

> The same comparison in the H200 batch has the opposite sign: object -.02034 ± .03658 vs null -.00532 ± .03302,
> diff **-.015022 mV**, z -.45490, p .699134, Holm 1. Across the two batches this rung carries no consistent
> direction, which is the first thing a new declaration would have to survive.

Also replace "the predeclared 3-null-SD effect gate" with "`common.compare`'s z ≥ 3 gate (`Z_RESULT`, declared by
reference in the stamped `primary.families` clause)".

**6. §6, add the two predeclared readouts the table omits, and state which rungs are matched.** Add a column and a
sentence:

> | Width × height (deg) | Max column coverage | Median extreme retinal Weber change | Median column equivalents | Frames with no column changed > 5 % |
> |---|---:|---:|---:|---:|
> | 2.2 × 8.8 | .54119 | -.34238 | .54119 | .2500 |
> | 4.4 × 2.2 | .42649 | -.22826 | .17205 | .4000 |
> | 4.4 × 4.4 | .65590 | -.53849 | .65590 | .2958 |
> | 4.4 × 8.8 | .88530 | -.65262 | .88530 | .1542 |
> | 4.4 × 15 | .88530 | -.65262 | 1.26172 | .0833 |
> | 4.4 × 30 | .88530 | -.65262 | 4.11470 | .0000 |
> | 8.8 × 8.8 | 1 | -.995 | 1.94265 | .0542 |
> | 15 × 8.8 | 1 | -.995 | 3.75444 | .0000 |
> | 30 × 8.8 | 1 | -.995 | 7.54119 | .0000 |
>
> **Which rungs are actually matched.** Peak per-column effective contrast reaches the object's ±.995 only where the
> width is ≥ 8.8 deg — that is, at three of the nine rectangles, all of them in the width ladder (8.8, 15, 30).
> The whole height ladder runs at width 4.4 and is capped at coverage .88530 / extreme -.65262, so its three upper
> rungs (8.8, 15, 30) are matched **to each other** but not to the width ladder, while its 2.2 rung (.42649 /
> -.22826) and 4.4 rung (.65590 / -.53849) are matched to nothing. In the width ladder the 2.2 rung (.54119 /
> -.34238) and the 4.4 rung (.88530 / -.65262) are each at their own level. Total retinal drive is not matched
> anywhere and is not meant to be: column-equivalents rise 24× along the height ladder (.17205 → 4.11470) and 14×
> along the width ladder (.54119 → 7.54119). At the smallest rungs the stimulus is also intermittent: 40 % of frames
> at 4.4 × 2.2 and 25 % at 2.2 × 8.8 change no column by more than 5 %.

**7. §7, add the two literature points the section omits.** After "Their RF measurement used a moving 8.8-degree
square", add:

> The width ladder here is much closer to the published protocol than the height ladder: Figure 3E varied width at a
> **fixed height of 8.8 deg**, which is exactly this ladder's fixed height (the tested widths differ only at the top
> — 2.2 / 4.4 / 8.8 / 15 / 30 here against 2.2 / 4.4 / 8.8 / 18 / 35 / 70 / 210 there). Figure 3D varied height
> 2.2 / 4.4 / 8.8 / 18 / 35 / 73.2 at a fixed width of 30 deg; the width-4.4 height sweep run here has no published
> counterpart, so LC11's "peak at height 8.8 with width 4.4" is an extrapolation across two separate experiments,
> not a measured expectation. Separately, **Figure 3B shows that maximum contrast is not LC11's optimum**: reducing
> OFF-object Weber contrast from 100 % to 30 % "nearly doubled the amplitude of the calcium response". Both ladders
> here run at |Weber| .995, i.e. at the contrast the animal responds to *least* strongly — a protocol caveat that
> belongs beside every null in §5.

**8. §5, deliver the predeclared sphere comparison.** The predeclaration promised "the round-2 sphere ladder …
beside these rectangles at the matching sizes, magnitudes only", and `tables.sphere_vs_rectangles` (72 rows) and
`summary.sphere` carry it, but §5 answers the third question with one qualitative sentence and no number. Add:

> Beside the sphere ladder (`out/interp/objr2/baseline.json`, 36 object runs, shipped lobe, recorded across
> **two** device models — B200 and H200 — so magnitudes only): LC11 Spearman rho **-.21607** (p_perm .20614),
> LC10a **+.02975** (p_perm .87181). Neither ladder produces a called preference on either statistic; the
> rectangles add that the failure survives separating height from width and holding the object's Weber contrast,
> and that at LC11's published shape the target contrast is if anything negative (H200 `wlad:dark` peak contrast
> -.092474). Two batches on different boxes are not compared row by row.

**9. §0 / §8, disclose the analysis-code hash difference.** `files.analysis_code_matches_stamp` is **False** for the
original batch (`scripts/object_round3_rectangles.py` sha256 `16750e7b…` at stamp vs `cb77c4b4…` at analysis) and
True for the house batch. This is expected — the predecessor agent stamped the plan and the analysis was written
afterwards — and `predeclared_matches_live_dict` is True in both, but §0 states only that both hashes are recorded.
Say plainly: "the reading rules are the stamped dict (unchanged, `predeclared_matches_live_dict` true); the analysis
code of the original batch was written after the stamp and its sha256 therefore differs, which is why both hashes
are recorded."

**10. §8, mark one claim unverifiable.** "A first checker draft under-counted declared family membership when p was
undefined; that checker was corrected to retain the declared m, with no change to the analysis or its p-values."
No draft and no diff is retained under `out/`, so a reader cannot check it. Either ship the draft's output or drop
the sentence; the substantive fact (236 of 256 family labels carry a defined Holm p) I confirm independently.

---

### Verdict

**mostly sound.**

Every number I could reach reproduces, and it reproduces at the deepest level available: the 40 + 40 primary
verdicts, the 64 + 64 size-monotone statistics, the 48 + 48 object-arm preference statistics and the 18 × 7
effective-contrast rows all come back to ≤ 2e-15 from `per_body.csv`, the reduced per-cell arrays and the raw
radiance, and four spot-checked runs reconstruct from the per-frame originals to ≤ 2.2e-16. Predeclaration before
submission is confirmed from the scheduler's own receipts; one GPU model and one host per batch is confirmed from
all 648 run provenances; `flyverse/optic.py` is untouched; every max-over-cells row carries its per-body median;
the Keleş & Frye claims in §7 are correct against the paper itself. The headline — 40/40 null, no animal-shape
call, in both batches — stands.

What keeps it from `sound` is framing, in five places where the audit is more generous to itself than the data
allows: a "recovery on the original box" that was a local repack of data that was never lost (R1); a "does not
replicate" built on one of 16 knife-edge exploratory tests, reported in the direction that gained a call and not in
the direction that lost one, against a declared rule that forbids the cross-batch comparison (R2); a "refutation" of
a predeclaration clause that in fact predicted the number (R3); a Report line that mis-describes the null design and
hides that the dark and bright families share their nulls (R4); and the B200 LC10a width-15 result carried forward
without the H200 row that points the other way (R5). Plus two predeclared readouts that never reach the prose: the
no-changed-column fraction and the sphere-vs-rectangle magnitudes.

None of these change a verdict, an adoption or a number. All ten corrections are text.
