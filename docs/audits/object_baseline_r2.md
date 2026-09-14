# The matched object baseline, round 2 (`scripts/object_round2_baseline.py`) -- Neurome intake, experiment 1, RUN

**Status: RESULTS PENDING -- sections 2-7 are written when the batch is analysed; section 0 was written and stamped
before submission.**

## 0. Predeclared reading rules (written before the batch was submitted; `out/objr2/predeclared.json`, stamped 2026-09-13T21:45:39Z)

The rules are the `PREDECLARED` dict of `scripts/object_round2_baseline.py`, dumped by `plan` at 21:45:39Z on
2026-09-13 and copied into the Result unchanged by `analyse`. In prose:

* **Replicate unit.** One process = one (A, B) pair under one brain seed = one run. A cluster job is a container of
  sequential processes (14-25 per job); the job is not the replicate.
* **Six runs per arm, not five.** The primary family has 12 members per LC type (6 rungs x {drive, spikes}) with Holm
  inside it. Holm needs the smallest member at p <= 0.05 / 12 = 0.00417; the exact Mann-Whitney floor
  (`common.p_floor`) is 0.0079 at 5 v 5 -- above that threshold, so at five runs no member of a 12-member family could
  survive Holm whatever the effect -- and 0.00216 at 6 v 6. Six is the smallest count at which the family is
  decidable. Every arm of every comparison (object runs at every rung, blank/blank nulls, both lobes, the rectangle
  ladders, the localizer) is in ONE submission.
* **The window rule (per body, in this order).** (1) fitted in the shipped localizer map of this submission (15-deg
  dark square, 3 passes, fitted in all 3 runs at `z_min` 5); (2) else fitted in the fb0 localizer map (1 run); (3)
  else the anatomical column of `trace.column_of_cells` (the rfmap's `anat_az_deg / anat_el_deg`), box = the type's
  median fitted width over both maps, else 15 deg; (4) else the whole window. The window frames are those on which
  the object overlaps the box: sphere `|d az| <= (w + diam)/2 and |d el| <= (h + diam)/2`
  (`probe_object_matched.rf_frames`); rectangle `|d az| <= (w + width)/2 and |d el| <= (h + height)/2`. A null run
  is windowed with the rung's diameter on its own track (the same frames, no object). `window_source` is recorded per
  body.
* **LC11 primary.** Per run: the population MEDIAN over the 143 LC11 bodies of the per-body windowed time-mean
  received drive (A - B, mV), and the median of the per-body windowed spike-count difference (A - B, spikes in the
  window). Family = {4.5, 8.8, 11, 15, 20, 30} x {drive, spikes} = 12 members, Holm within; each member = the six
  object runs at that rung vs the six blank/blank runs of the same lobe, windowed with that rung's diameter; the
  tie-aware exact permutation U (two-sided, every C(12, 6) = 924 assignment enumerated, average ranks for ties) is
  the p; the verdict is `common.compare`'s (z on the null SD, its exact U, `p_floor`). Expectation: a preference for
  small objects (4-9 deg; Keles & Frye 2017 Fig 3D/E) -- the excess over the null largest at 4.5-8.8 deg, falling by
  20-30 deg. Preference test: Spearman rho of the per-run statistic against the diameter over the 36 object runs
  (permutation p, 20,000 shuffles, seed 0) and the small-vs-large contrast (mean over runs at {4.5, 8.8, 11} minus at
  {20, 30}, permutation p over run labels).
* **LC10a primary.** The same two statistics over the 275 LC10a bodies, the same 12-member family; expectation a
  15-30 deg preference (Schretter et al. 2024 Fig 3a), i.e. a positive Spearman sign.
* **The call.** A size preference is CALLED only if at least one member reads `result` from `common.compare` AND
  survives Holm (`p_holm <= 0.05`); the preference tests then say which way. Otherwise the family is exploratory and
  the answer is "no detected size preference at 6 v 6 on this statistic".
* **Secondary, exploratory.** T2 / T3 / Tm5Y / TmY21: `diff_signed_best_cell` AND the population `diff_signed_mean`
  kept as two statistics (plus `diff_abs_best_cell_mean` as a third), whole-window, `probe_object_matched`'s
  definitions, per rung vs the nulls, unadjusted p and Holm both reported. LC: population mean and best-cell of the
  windowed per-body drive, the whole-window `diff_max_over_cells_mean_mv` (the old headline statistic, for
  continuity), `diff_rate_hz_max_cell`. The fb0 lobe: its blank/blank null has SD ~0 for the optic quantities, so
  `compare` returns `undetermined`; it is read as magnitudes with the LIF's own spike scatter. The rectangle ladders:
  the same windowed medians, Holm within each (family x type) = 12, reported as the matched-rectangle families; the
  sphere is the primary family.
* **Footprint per rung, from the captured radiance**: columns dimmed > 50 % and changed > 5 % per frame, centroid
  elevation (mean, band), and the effective-contrast readout: the per-frame extreme relative luminance change (median
  over frames) and the mean relative change over the changed set. Stated up front: **the effective retinal contrast
  is NOT matched across the sphere ladder** (a 4.5-deg ball only partially fills a 4.6-deg column); the synthetic
  ladders are contrast-matched by construction (Weber +-0.995 per covered fraction), so the bright synthetic square
  ladder is the contrast-matched bright control, and the sphere `lamp` ball (not contrast-matched) is not run here.

## 0b. Verification

**The baseline skeptic** (`verify:baseline`, Opus, verdict **mostly sound**) read the design and the queued batch
before any of it was analysable; the verdict is recorded verbatim in `docs/audits/receptor_verification.md`
("Object round 2"). Its three refutations:

1. **`spearman_perm` returns the smallest attainable permutation p whenever rho is undefined.**
   `cnt += abs(rho_perm) >= abs(rho) - 1e-12` is False for every shuffle when rho is NaN, so `cnt = 0` and
   p = 1/20001 = 4.99975e-05. The skeptic predicted the exposure exactly: every LC population spike median is
   exactly 0.0 in every run of both arms, so at 6 v 6 the `spikes_median` rho is NaN across all 36 object runs and
   **a predeclared PRIMARY preference test would print p = 5e-05**. It did. See "The spearman floor" below.
2. **No `<n> job(s), 0 failed` line will ever exist for this batch** -- `out/objr2_cluster.log` is 0 bytes and the
   client was killed. Correct: the analysis was run with `--force` and the Result carries the problem string; the
   replacement check is `probe_object_matched.py verify out/objr2/sph` -> `out/objr2/verify_sph.json`, **problems
   none over all 84 runs**.
3. The status line "all 16 jobs were QUEUED, no process had started" was true at the 21:46Z stamp and false by the
   time the skeptic ran. The batch completed.

Its corrections that bear on the numbers in this file: the analysis script was itself edited after the
predeclaration stamp (below); the 12 members of each primary family share the **same six blank/blank runs**, so
they are strongly dependent (Holm is still valid under arbitrary dependence, but they are not twelve independent
tests); six of each twelve are the structurally uninformative `spikes_median` members, which is the whole reason
the arm count had to rise from five to six (a drive-only six-member family needs p <= 0.00833, reachable at 5 v 5's
floor of 0.00794) -- and changing that now would be post-hoc, so it is predeclared for the next round, not applied
here; and the specificity battery is **not in this batch** by design (it belongs to the model comparison,
`object_compare_r2.md`), which should be stated rather than implied.

**The round-2 critic's corrections**, recomputed from `out/interp/objr2/baseline.json` and the run files:

* **The spearman floor was live in the shipped Result and in the export.** The fix landed in
  `scripts/object_round2_baseline.py` (mtime 2026-09-14T00:58:40Z) **1h41m after** `baseline.json` was written
  (2026-09-13T23:17:09Z), so the delivered Result and the 2026-09-13 ladder export's `preference.csv` carried
  `spearman_p_perm = 4.99975e-05` with an empty `spearman_rho` on **four** `spikes_median` preference rows
  (ship x LC11, ship x LC10a, fb0 x LC11, fb0 x LC10a) -- **all four `role = primary`** -- with
  `null_spearman_p_perm = 4.9975e-04` beside them. **Closed 2026-09-14**: `analyse` was re-run on the
  already-fetched batch (CPU, no GPU), `out/interp/objr2/baseline.json` now carries `None` / NaN on all four rows
  and on their null counterparts, **every other number is unchanged and the primary families are identical**, and
  the ladder was re-exported to `out/export/objr2-ladder-20260914T024906Z-035363c0` whose `preference.csv` leaves
  those cells empty. The 2026-09-13 directories remain on disk as the superseded delivery.
* **Only 14 of 84 consoles arrived.** `ls out/objr2/sph/*.txt` = **14** against 84 run JSONs, and
  `verify_sph.json` records `console_device_cuda: None` for **70 of 84**. The device claim survives because all 84
  run JSONs read `execution.device = cuda`, but the console-vs-JSON cross-check `docs/INTERP.md` 10.4 item 4 makes
  mandatory -- and which was supposed to *replace* the missing `0 failed` line -- exists for 14 runs, not 84. The
  compare batch has 240/240 and is the model to copy.
* **The analysis code was edited after the predeclaration stamp.** `predeclared.json` 21:45:39Z, `submit_console.txt`
  21:46:06Z, and `scripts/object_round2_baseline.py` edited at 21:46:53Z and again at 2026-09-14T00:58:40Z. The box
  copy differs from the local one and `tree_state.json` records the box hash. The skeptic diffed the first edit (15
  lines, analysis-only: `compare_row`'s empty-arm branch and a duplicate `contrast_readout` key; `build_jobs`,
  `cmd_run_job` and the process command strings byte-identical), so the running batch is unaffected -- but
  "stamped before submission" covers the reading rules, not the reducer, and the eventual analysis was run by a
  script the boxes never saw.
* **The `spikes_median` family members are uninformative, not negative.** All six are exactly
  **0.000 +- 0.000 in every arm of every rung** for both LC types and both lobes; U = 18 and p = 1.000 by ties, and
  `p_holm` 1.000. They are 6 of each 12-member family. The LC populations are **not silent** -- `bodies_firing_a/b`
  is nonzero in most runs -- they simply emit essentially no spikes in this protocol, so the spike half of each
  family carries no information.
* **Effective contrast is not matched, and the mismatch is large.** `sphere_footprint.extreme_rel_change_median`
  runs **-0.504 / -0.869 / -0.876 / -0.899 / -0.927 / -0.948** across 4.5 -> 30 deg and
  `mean_rel_change_over_changed_set` **-0.266 -> -0.665**. This is declared in `predeclared.not_matched` above, so
  it is disclosed rather than hidden, but it is the one item of Neurome's five requirements ("trajectory, speed,
  **contrast** and background controlled") that the sphere ladder only partly meets. The contrast-matched families
  are the synthetic rectangle ladders, which were never fetched.
* **The RF map covers 0 of 143 LC11 bodies.** `summary.windows.sources`: LC11 **143/143 `anat`**, LC10a 262 `anat`
  + 7 `rf_ship` + 6 `rf_fb0` = 13/275 = 4.7 % fitted -- so **405 of 418 LC windows are anatomical boxes**, not
  measured receptive fields. Two coverage numbers that belong beside the primary statistic and were not in this
  file: `sphere_per_run.n_bodies_windowed` gives LC11 **55 / 99 / 99 / 99 / 103 / 103** of 143 and LC10a
  **73 / 76 / 76 / 76 / 79 / 90** of 275 across 4.5 -> 30 deg. The 4.5-deg rung -- the biologically decisive one --
  is a median over **55 anatomically-placed boxes**.
