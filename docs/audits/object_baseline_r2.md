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
