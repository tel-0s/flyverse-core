# apply:object -- the moving ball between the medulla and the lobula: where it is lost and what kind of fact the loss is

Task `apply:object` of the interpretability workflow (docs/INTERP.md). Generator of every number here:
`scripts/interp_apply_object.py` (subcommands `trace`, `stage`, `plan` / `record` / `analyse`, `perrun`, `ladder-plan` /
`ladder`, `selftest`), composing the validated tools -- trace (docs/audits/interp_trace.md), decompose
(interp_decompose.md), the lesion pattern (interp_lesion.md) and export (interp_export.md). Results:
`out/interp/apply_object/*.json` (consoles `*_console.txt`); recordings `out/apply_object/{les,ladder,smoke}/`;
cluster logs `out/apobj-{smoke,les,lad}_cluster.log`; exports `out/export/<run_id>/` (section 7). Nothing under
`flyverse/` was edited by this task (the model is read, not changed); every counterfactual is an `OpticParams`
override or an in-process edit of the loaded graph / receptor signs recorded in the run's meta (section 4.1).

Inputs taken as given (dynamics round 1): `object_sweep.md` 8.4-8.7 (the null-referenced assay; the medulla carries
the ball, T2 / T3 / Tm5Y / TmY21 / LC11 / LC10a do not), `optic_measures.md` 5-7 (ON / OFF cancellation at T3, l1
pooling at the LC cells, `gain_fb = 0` the deterministic null arm, no hand-set measure on the small-field edges),
`docs/NEUROME_INTERFACE.md` (LC11 / LC10a per-type input tables; the size-tuning request of section 3).

**Status: complete.** Cluster runs `apobj-smoke-84bc5f` (3 jobs, 0 failed, 2.4 min), **`apobj-les-e38d76` (96 jobs, 0
failed, 22.5 min)** and **`apobj-lad-287a43` (40 jobs, 0 failed, 11.6 min)**, every job `device cuda` (NVIDIA B200,
torch 2.11.0+cu128), fetched into named subdirectories; the CPU analyses below were re-run after the trace tool's
raw-count fix (section 7.2) and the exports re-made.

---

## 0. The answer

**Where.** From the photoreceptors the ball (1 cm at 5 cm: 11.4 deg) is carried at depth 1 (L1 / L2 / L3 z +13.3 /
+13.9 / +11.3 above the none-vs-none null), depth 2 (Mi1 +17.2, Mi9 +14.2, Tm2 +11.0, Tm3 +10.5, C3 +7.4, Mi4 +6.6,
Tm1 +4.4, Tm4 +4.3, Tm20 +4.3, Tm5a +4.7, Dm8a +4.5, L5 +3.8) and along the motion pathway at depth 3-4 (T4c +14.5,
T4d +15.6, T5a +6.9, LPi34 +5.9). **Every input class of LC11 and of LC10a is at the null**: LC11 <- T3 +0.2, T2 -0.2,
Tm6 -0.2, T2a -0.1, Tm12 -0.3, TmY18 +0.0, Li15 -0.2; LC10a <- Tm5Y +0.9, TmY21 -0.3, LC9 +0.7, LC10c-1 +1.2, LC10c-2
+0.5, AOTU042 +0.5, TuTuA_2 silent. The first stage at the null is the **T-cell / small-field Tm stage** on the LC11
chain (depth 3) and the **Tm5Y / TmY21 stage** on the LC10a chain (depth 2); LC11 (+0.1) and LC10a (+0.0) then have
nothing to pool (section 1).

**"At the null" at the small-field stage is a threshold call, not a measured absence.** The same quantity on the
same model and stimulus, measured in four independent batches, reads **Tm5Y z +0.86** (`trace_obj.json`, 5 v 5),
**+2.19** (`lesions.json` base arm, 4 v 4), **+2.65** (`ladder.json` d114, 5 v 5) and **+3.65**
(`skeptic_lesions.json` base arm, 4 v 4: 0.0470 +- 0.0031 vs its own null 0.0290 +- 0.0049, U 16/16, p 0.0286,
verdict `result`); **T3** reads +0.20, +1.39, +2.15 (d114, U 25/25, p 0.0079 -- complete rank separation of ball
over null) and +0.34. The small-field stage carries an **attenuated but non-zero** figure that crosses the tool's
`|z| >= 3` rule from batch to batch. LC11 and LC10a are at or below their nulls in all four, so only the
small-field half of "lost" is a threshold call.

**Which kind of fact** -- the four candidates the task named, decided by the eight lesion arms of section 4 (each
recorded under the identical protocol with its own none-vs-none null, 4 independent runs per arm), the per-run
scatter of section 4.4 and the deterministic projection of section 4.5:

1. **Wiring -- yes, at T3.** T3 sums **carriers whose measured figures have opposite sign** through excitatory,
   exact-tier synapses: Mi1 22.5 %, Tm3 8.2 % and Tm2 1.8 % raise, Tm1 16.0 % and Tm4 6.7 % lower. The split is
   *not* strictly ON vs OFF -- `stage.json`'s own `carriers_raising` is `[Mi1, Tm3, Tm2]` against
   `carriers_lowering` `[Tm1, Tm4]`, i.e. `pathways_raising` `[OFF, ON]`: Tm2, an OFF type, raises with the ON
   carriers. "ON / OFF convergence" is shorthand for the two hold arms, which held Tm1|Tm4 and Mi1|Tm3.
   In the deterministic lobe the carrier terms cancel to **3.6 % of their sum** (cancellation 0.964, `perrun` on
   the fb0 arm; 0.92 in the pooled shipped-model trace, 0.84 +- 0.20 per run). **The cancellation fraction is a
   descriptive ratio of the two opposing sums in the type-mean linear decomposition; it does not predict the
   target's measured figure in sign or size** -- on the deterministic fb0 arm `perrun`'s own
   `cancellation_summary` gives T3 `linear_estimate` +4.20e-6 against `own_signed_figure` -2.25e-4 (54x too small
   and the wrong sign), T2 -1.91e-5 vs +1.26e-3, Tm5Y +6.28e-6 vs +9.52e-4, TmY21 -3.04e-6 vs -1.88e-4, LC10a
   +1.57e-6 vs +6.28e-3. The wiring conclusion rests on the lesion arms below, not on this arithmetic.
   Silencing either channel restores T3
   (`t3_off_held` 0.062 vs its null 0.026 +- 0.010, z +3.5; `t3_on_held` 0.072 vs 0.025 +- 0.010, z +4.8; both U 16/16,
   p 0.029 = the 4 v 4 floor) and makes T3's figure a direct response instead of a residual (per-run cancellation
   0.20 / 0.24) whose run-to-run scatter collapses to the medulla's (T3 0.062 +- 0.002 and 0.072 +- 0.002 across 4
   stochastic runs, against 0.024 +- 0.005 in the shipped model and Mi1's 0.073 +- 0.001).
2. **Dynamics -- yes, and it is what removes the residual.** With the spiking -> rate feedback off (`fb0`,
   `gain_fb = 0`; the 4 runs are one deterministic draw and its null is exactly 0) the small-field stage passes a
   per-cell figure of the medulla's size: T3 0.041, T2 0.066, Tm5Y 0.072, TmY21 0.062, TmY5a 0.084 against Mi1 0.068,
   Mi4 0.047, Tm3 0.104 -- 4.5-5.5 SD above the shipped model's own null floor (z vs the base null: T3 +4.5, T2 +5.5,
   Tm5Y +4.7, TmY21 +5.2). Under the shipped feedback (`gain_fb 0.5`) **those very cells carry nothing**: at fb0's best
   T3 cell the shipped model's figure is 0.003 +- 0.009 (4 runs; T2 0.017 +- 0.018 of 0.066, Tm5Y 0.018 +- 0.012 of
   0.072, TmY21 0.013 +- 0.014 of 0.062), while the medulla's best cells keep theirs to the digit (Mi1 0.073 +- 0.001
   of 0.068, Tm3 0.106 +- 0.003 of 0.104, Mi4 0.045 +- 0.002 of 0.047, Tm1 0.070 +- 0.004 of 0.074). The feedback does
   not add a floor on top of the figure (the T3 |dr| level is 0.084 with and 0.085 without it) and it does not enter
   the small-field types preferentially (spiking units are 2.8 / 3.1 / 5.4 % of T3 / T2 / Tm5Y's shaped-weight input
   against 3.6 / 5.5 / 4.0 % of Mi1 / Mi4 / Tm3): it perturbs the two channels independently, and a figure that is
   the 4-30 % residual of two opposing sums is scrambled while a direct response is not. Point 1's edge arms show the
   same thing from the other side: once T3's figure is direct it survives the same feedback.
3. **Rate-model artefact -- no.** `norm l1` collapses every figure 5-100x (Mi1 0.016, T3 0.0010, LC11 0.0003 mV) and
   silences the loop (its null is exactly 0: the lobe no longer drives a spike); `out_norm l2` scales the LC drives
   30x (LC11 1.90, LC10a 1.36 mV) together with their nulls (z -0.3 / -0.2) and leaves T3 / T2 where they were (z
   +0.7 / +0.6); T2 / T3 at operating point 0 (`rect`) lowers them (T3 0.010, T2 0.023; z +1.0 / +2.8). No
   normalisation, rectification or gain arm restores anything.
4. **Sign / receptor fact -- no at T3; untested at the unprofiled types.** The counterfactual flip of Tm1 | Tm4 -> T3
   restores T3 (0.094 vs 0.025 +- 0.012, z +5.8) -- but silencing either channel does the same (point 1), so no sign
   the data cannot see is needed to explain the loss, and the data say + (exact tier, 0 receptor changes on T3's
   inputs). Tm5Y, TmY21 and LC11 are fallback-tier on 100 % of their input (unprofiled); no sign arm was run on them.
   Their deterministic figures exist (point 2) and vanish under the feedback exactly as T3's does, so their loss in
   the shipped model is the same dynamics fact on top of a smaller cancellation (0.68 / 0.81 deterministic).
5. **LC11 / LC10a -- pooling; the structural half is measured, the experimental half is an underpowered
   non-detection.** The structural statement is sound: T3 is 21 % of LC11's input, an LC11 cell pools 94 retinal
   columns under `out_norm l1`, LC10a 32. The experimental statement is **"not detectable at 4 v 4 with this
   null"**, not "lost in every arm": LC11's null mean ranges 0.045-0.082 mV with SD **0.020-0.045** across arms, so
   point estimates that did move are invisible -- `t3_off_flip` LC11 0.0765 vs the base's 0.0451 (**+70 %**), and
   in the replicate batch `t3_on_held` 0.0722 vs base 0.0532 (**+36 %**) and `t3_nc_held` 0.0682 (**+28 %**), all
   `null`, all `not moved`, only because the pooled scatter is 0.03-0.05 mV. The deterministic LC figures
   themselves (LC11 0.046 mV, LC10a 0.080 mV) likewise lie inside the shipped model's null (0.057
   +- 0.027 / 0.083 +- 0.037, 5 draws). The mechanism -- wiring (94 / 32 columns per cell) times the output
   normalisation, as optic_measures.md 6 read it -- is upstream-independent; the assay cannot yet resolve a 30-70 %
   change at the LC cells.
6. **All of it is a small-object fact.** The Neurome size ladder (section 5): at 4.5 / 11.4 / 20 deg the two LC types
   and their small-field inputs sit at the null; at 30 deg T2 / T3 / Tm5Y / TmY21 / TmY13 / TmY5a and LC11 (+4.8) /
   LC10a (+8.6) all reach 'result' in 5 v 5 runs, behind the loom chain, whose threshold lies **below** 20 deg
   (LPLC2 / LC16 / LC4 reach `result` at 20 deg here, and LPLC2's verdict at 11.4 deg **flips between batches** --
   z +2.05 / +1.51 / +2.60 / +2.82 over four batches of the same stimulus, so this file's own 11.4 deg point does
   not reproduce object_sweep.md 8.7's +5.4; section 5). The model's
   size ordering is the opposite of the animal's (LC11 ~5-10 deg objects; LPLC2 expansion).

So: **the object is lost by the convergence of opposite-figure carriers at the small-field stage (a wiring fact under a linear sum),
whose residual the stochastic spiking feedback then destroys (a dynamics fact), and what survives is pooled away at
the LC cells (a wiring x output-normalisation fact); it is not a normalisation / rectification / gain artefact and
not a sign fact.** Nothing was tuned; the edge arms are diagnoses of the linear reading, not proposals.

---

## 1. The trace with the null, LC11 / LC10a and their inputs included (`trace`)

`PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py trace --dir out/trv --prefix obj --json out/interp/apply_object/trace_obj.json`
on the 15 recordings of interp_trace.md section 1 (cluster run `trv-dee1e7`, 5 stim + 5 ctrl + 5 null jobs, every
job `device cuda` NVIDIA B200, cache md5 `ef23cc27bea13be7f6a96f3c04fd3737`, effective weights
`ed1df661716d240b0f9289607f95320c`, the shipped model: receptor `sign` / `abs`, OpticParams defaults: `norm l2`,
`gain_in 3`, `gain_fb 0.5`, `out_norm l1`, `gain_out 100 mV`, clip 35 mV, baseline 0.5, T4 / T5 at 0). Statistic
`best_cell` = the object sweep's max over cells of the per-cell (stim - ctrl) time-mean (rate units: the time-mean of
|deviation from r0|; spiking cells: the optic drive in mV), five draws against five none-vs-none draws; `min_cells
2` so that TuTuA_2 (2 cells) is scored; decomposed at T3, T2, Tm5Y, TmY21, LC11, LC10a. A second pass scores every
spiking type on its firing rate (`trace_obj_rate.json`) for the central-brain inputs of LC10a, which receive no
optic drive. Re-run on the fixed trace tool (section 7.2): every statistic identical to the first run.

Validation (`validation.status`): **reproduced, and that word is verdict-level only.** `trace.py:786-801` checks
only that Mi4 / Mi1 / Tm3 read `result` and the six named types do not; the **z bands** of docs/INTERP.md section 6
(Mi4 +22.3 / +28.6, Mi1 +7.8 / +27.9, Tm3 +7.8 / +15.6) are **not tested and do not reproduce here** -- Mi4's
measured z is **+6.6**, 3.4x below its band, because this batch's 5-draw null SD is ~3x the reference's. What
reproduces is the carrier / at-null grouping (Mi4 / Mi1 / Tm3 `result` at +6.6 / +17.2 / +10.5, T2 / T3 / Tm5Y /
TmY21 / LC11 / LC10a `null`) and the stimulus-arm levels; the z magnitudes do not, and under the contract's own
default statistic (`figure_z`) the target reads `not reproduced`.

| type | depth | stage | kind | n | quantity | stim mean | null mean +- SD | z | U | p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 1 | 1 | graded | 1776 | optic_dr_abs | 0.1256 | 0.0191 +- 0.0080 | +13.3 | 25 | 0.0079 | result |
| L2 | 1 | 1 | graded | 1779 | optic_dr_abs | 0.1009 | 0.0205 +- 0.0058 | +13.9 | 25 | 0.0079 | result |
| L3 | 1 | 1 | graded | 1772 | optic_dr_abs | 0.0797 | 0.0186 +- 0.0054 | +11.3 | 25 | 0.0079 | result |
| L5 | 2 | 1 | graded | 1787 | optic_dr_abs | 0.0373 | 0.0154 +- 0.0058 | +3.8 | 25 | 0.0079 | result |
| C3 | 2 | 1 | graded | 1779 | optic_dr_abs | 0.0669 | 0.0181 +- 0.0066 | +7.4 | 25 | 0.0079 | result |
| Mi1 | 2 | 2 | graded | 1773 | optic_dr_abs | 0.0724 | 0.0149 +- 0.0034 | +17.2 | 25 | 0.0079 | result |
| Mi4 | 2 | 2 | graded | 1772 | optic_dr_abs | 0.0478 | 0.0112 +- 0.0055 | +6.6 | 25 | 0.0079 | result |
| Mi9 | 2 | 2 | graded | 1775 | optic_dr_abs | 0.0743 | 0.0103 +- 0.0045 | +14.2 | 25 | 0.0079 | result |
| Tm3 | 2 | 3 | graded | 2054 | optic_dr_abs | 0.1081 | 0.0239 +- 0.0080 | +10.5 | 25 | 0.0079 | result |
| Tm1 | 2 | 3 | graded | 1777 | optic_dr_abs | 0.0716 | 0.0249 +- 0.0106 | +4.4 | 25 | 0.0079 | result |
| Tm2 | 2 | 3 | graded | 1766 | optic_dr_abs | 0.0840 | 0.0247 +- 0.0054 | +11.0 | 25 | 0.0079 | result |
| Tm4 | 2 | 3 | graded | 1670 | optic_dr_abs | 0.1035 | 0.0354 +- 0.0157 | +4.3 | 25 | 0.0079 | result |
| Tm9 | 2 | 3 | graded | 1771 | optic_dr_abs | 0.0667 | 0.0324 +- 0.0118 | +2.9 | 25 | 0.0079 | null (z < 3) |
| Tm20 | 2 | 3 | graded | 1762 | optic_dr_abs | 0.1007 | 0.0343 +- 0.0155 | +4.3 | 25 | 0.0079 | result |
| Tm5a | 1 | 3 | graded | 624 | optic_dr_abs | 0.0593 | 0.0261 +- 0.0070 | +4.7 | 25 | 0.0079 | result |
| Dm8a | 1 | 2 | graded | 572 | optic_dr_abs | 0.0346 | 0.0134 +- 0.0047 | +4.5 | 25 | 0.0079 | result |
| **T3** | 3 | 4 | graded | 1940 | optic_dr_abs | 0.0278 | 0.0258 +- 0.0101 | **+0.2** | 14 | 0.84 | null |
| **T2** | 3 | 4 | graded | 1630 | optic_dr_abs | 0.0409 | 0.0439 +- 0.0196 | **-0.2** | 13 | 1.00 | null |
| T2a | 3 | 4 | graded | 1872 | optic_dr_abs | 0.0301 | 0.0317 +- 0.0139 | -0.1 | 13 | 1.00 | null |
| Tm6 | 2 | 3 | graded | 1526 | optic_dr_abs | 0.0406 | 0.0447 +- 0.0178 | -0.2 | 10 | 0.69 | null |
| Tm12 | 2 | 3 | graded | 967 | optic_dr_abs | 0.0307 | 0.0352 +- 0.0166 | -0.3 | 12 | 1.00 | null |
| TmY18 | 3 | 3 | graded | 1367 | optic_dr_abs | 0.0367 | 0.0366 +- 0.0189 | +0.0 | 14 | 0.84 | null |
| Li15 | 3 | 5 | graded | 41 | optic_dr_abs | 0.0414 | 0.0459 +- 0.0246 | -0.2 | 11 | 0.84 | null |
| **Tm5Y** | 2 | 3 | graded | 898 | optic_dr_abs | 0.0512 | 0.0390 +- 0.0142 | **+0.9** | 20 | 0.15 | null |
| **TmY21** | 2 | 3 | graded | 372 | optic_dr_abs | 0.0399 | 0.0448 +- 0.0185 | **-0.3** | 9 | 0.55 | null |
| TmY13 | 3 | 3 | graded | 432 | optic_dr_abs | 0.0224 | 0.0263 +- 0.0086 | -0.5 | 9 | 0.55 | null |
| TmY5a | 3 | 3 | graded | 1364 | optic_dr_abs | 0.0414 | 0.0430 +- 0.0138 | -0.1 | 12 | 1.00 | null |
| T4c | 3 | 4 | graded | 1778 | optic_dr_abs | 0.0517 | 0.0103 +- 0.0029 | +14.5 | 25 | 0.0079 | result |
| T4d | 3 | 4 | graded | 1709 | optic_dr_abs | 0.0502 | 0.0090 +- 0.0026 | +15.6 | 25 | 0.0079 | result |
| T5a | 3 | 4 | graded | 1664 | optic_dr_abs | 0.0856 | 0.0345 +- 0.0074 | +6.9 | 25 | 0.0079 | result |
| LPi34 | 4 | 5 | graded | 119 | optic_dr_abs | 0.0709 | 0.0215 +- 0.0084 | +5.9 | 25 | 0.0079 | result |
| LC9 | 3 | 6 | spiking | 219 | drive_mv | 0.0463 | 0.0332 +- 0.0178 | +0.7 | 18 | 0.31 | null |
| LC10c-1 | 2 | 6 | spiking | 130 | drive_mv | 0.0733 | 0.0522 +- 0.0182 | +1.2 | 14 | 0.84 | null |
| LC10c-2 | 2 | 6 | spiking | 125 | drive_mv | 0.0594 | 0.0514 +- 0.0174 | +0.5 | 9 | 0.55 | null |
| **LC11** (mV) | 3 | 6 | spiking | 143 | drive_mv | 0.0592 | 0.0567 +- 0.0271 | **+0.1** | 13 | 1.00 | null |
| **LC10a** (mV) | 2 | 6 | spiking | 275 | drive_mv | 0.0844 | 0.0828 +- 0.0365 | **+0.0** | 12 | 1.00 | null |
| LPLC2 (mV) | 3 | 6 | spiking | 185 | drive_mv | 0.3010 | 0.1653 +- 0.0661 | +2.1 | 25 | 0.0079 | null (z < 3) |
| LC10b (mV) | 3 | 6 | spiking | 95 | drive_mv | 0.1633 | 0.1319 +- 0.0859 | +0.4 | 15 | 0.69 | null |
| LC16 (mV) | 3 | 6 | spiking | 182 | drive_mv | 0.0901 | 0.0743 +- 0.0127 | +1.2 | 14 | 0.84 | null |
| LC4 (mV) | 3 | 6 | spiking | 126 | drive_mv | 0.0900 | 0.0653 +- 0.0248 | +1.0 | 19 | 0.22 | null |

Per-run draws of the two LC types (stim / null, mV): LC11 .0471 .0911 .0527 .0491 .0561 / .0793 .0721 .0770 .0343
.0209; LC10a .0530 .1209 .0530 .0878 .1076 / .0766 .1301 .1081 .0594 .0397 -- each arm's scatter is the size of its
mean. The central-brain inputs of LC10a on their firing rate (`trace_obj_rate.json`, Hz over the 12 s window):
TuTuA_2 (2 cells) 0.000 in every stim, ctrl and null run -- silent in this protocol, so its 7.3 % of LC10a's LIF
input carries nothing; AOTU042 (4 cells) 0.46 vs 0.40 Hz, best-cell difference +0.27 vs null +0.17 +- 0.20, z +0.5,
U 17, p 0.42; LC9 0.006 Hz, LC10c-1 / -2 0.003 Hz (all null). Carriers per depth (result / scored): 1 7/30, 2
16/298, 3 16/1404, 4 5/3995, 5 0/4101, 6 0/1358; the contract's depth rule returns 5 (the motion pathway carries to
depth 4) and the input rule lists 240 non-carriers fed >= 20 % by carriers, among them T3 (carrier share 0.53 from
Mi1 / Tm1 / Tm3 / Tm4), TmY13 0.49, TmY18 0.42, Tm6 0.40, T2a 0.35, T2 0.34, Tm12 0.34, TmY5a 0.26 -- every named
input of LC11 is a non-carrier whose own input is a third to a half carriers.

The raw synapse counts behind the LIF shares (`lost_inputs.raw_synapses_per_post`, now the cache's counts -- section
7.2): onto one T3 cell Mi1 53.5, Tm1 38.2, Tm3 19.3, Tm4 15.7, Pm5 13.0, Pm1 12.3 synapses; onto one LC11 cell T3
484.0, T2 258.2, Tm6 173.0, T2a 171.8, Tm12 137.9, TmY18 109.0, Li15 97.3 -- the per-type tables of
docs/NEUROME_INTERFACE.md, matched to the synapse.

Composition note: `trace` decomposes at the six targets; the static tables came back (`lost_static_*`, 6 targets,
2,618 / 5,000 / 5,258 rows in the export) and no dynamic table was produced (the earlier run's `decompose.arm_params`
rejection of the recording's `meta['arm'] = 'stim'` -- a defect for the decompose / trace owners, not this task's
files; the retinotopic `lost_inputs` table, the reading, is unaffected).

## 2. What is in force at the first null stage (`stage`, and its per-run scatter, `perrun`)

`PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py stage --trace out/interp/apply_object/trace_obj.json --json out/interp/apply_object/stage.json`
-- the rate lobe's own input table rebuilt on the CPU exactly as `optic.OpticLobe.__init__` builds it (|count| x
the receptor fast sign, / `in_syn_l2` for rate targets and / `in_syn` for spiking targets under `out_norm l1`, the
`DEFAULT_PAIR_GAIN` factors per (pre, post) type pair; the OpticParams are the recording's own), joined with the
trace's verdict and signed figure per input, the receptor tier that decided the entries' sign, tau, operating point.
`share_abs` = the pre type's share of the post cells' |input| in that normalisation; `share_lif` = the trace's
shaped-weight share (the LIF's `A`); they agree to 1-2 pp for the rate targets.

**T3** (verdict null, z +0.2; E 0.58 / I 0.42; carriers 0.56 of the input; tau 10 ms, baseline 0.5, norm l2,
denominator 52.2; all inputs exact-tier, 0 entries changed vs NT_SIGN):

| input | pathway | share (optic l2) | sign | pair gain | pre tau | tier | pre verdict (z) | signed figure | term |
|---|---|---|---|---|---|---|---|---|---|
| Mi1 | ON | 0.225 | + | 1 | 8 ms | exact | result (+17.2) | +2.15e-4 | +4.8e-5 |
| Tm1 | OFF | 0.160 | + | 1 | 8 | exact | result (+4.4) | -2.18e-4 | -3.5e-5 |
| Tm3 | ON | 0.082 | + | 1 | 8 | exact | result (+10.5) | +1.9e-5 | +1.5e-6 |
| Tm4 | OFF | 0.067 | + | 1 | 8 | exact | result (+4.3) | -1.27e-4 | -8.5e-6 |
| Pm5 | medulla inhibitory | 0.056 | - | 1 | 10 | exact | null (+0.6) | -7.1e-4 | +4.0e-5 |
| Pm1 | medulla inhibitory | 0.053 | - | 1 | 10 | exact | null (+1.5) | -8.6e-5 | +4.6e-6 |
| Mi2 | medulla inhibitory | 0.034 | - | 1 | 10 | exact | null (-0.0) | +5.3e-4 | -1.8e-5 |
| Li26, Pm3, Li25, Pm10, Pm2b, Li17, Pm2a | inhibitory | 0.19 together | - | 1 | 10 | exact | null | | |
| Tm2 | OFF | 0.018 | + | 1 | 8 | exact | result (+11.0) | +1.0e-4 | +1.8e-6 |

Carriers raising T3 {Mi1, Tm3, Tm2} +5.1e-5 vs lowering {Tm1, Tm4} -4.3e-5: **cancellation fraction 0.92 through
excitatory synapses only** (excitatory carrier share 0.55 of 0.56). Note the split is by measured figure, not by
pathway: `stage.json` records `pathways_raising ['OFF', 'ON']` because Tm2 (OFF) raises alongside Mi1 / Tm3, so the
accurate statement at T3 is "carriers whose measured figures have opposite sign converge through same-sign
(excitatory, exact-tier) synapses" -- which is exactly what the two hold arms of section 4 test. The Pm / Li inhibition (42 % of the input) is
not a carrier (Pm5 z +0.6, Pm1 +1.5, Mi2 -0.0) -- not tuned to the object in this model, as optic_measures.md 5.3
read by hand. **T2** (z -0.2; E 0.52 / I 0.48; carriers 0.39): Tm2 0.113 + / L5 0.083 + / Tm3 0.070 + / Mi1 0.034 +
raising (+2.0e-5) vs C3 0.024 - lowering (-1.9e-5), cancellation 0.99, exact-tier throughout; the OFF carrier Tm2 and
the ON carriers Tm3 / Mi1 arrive with the SAME measured sign at T2 (the signed figures of Tm2 +1.0e-4, Tm3 +1.9e-5,
Mi1 +2.2e-4 are all positive for the sweeping ball -- the time-mean of a moving dark object is not the static apple's
ON-lowered / OFF-raised pattern), and the one carrier with the opposite sign, C3, enters through an inhibitory
synapse and cancels them. **Tm5Y** (z +0.9; carriers 0.32; fallback tier on 100 % -- unprofiled): Tm20 0.140 +
(result +4.3, the largest input and a carrier), Li19 0.093 -, Y3 0.075 +, Tm5a 0.034 + (carrier, lowering), Dm8a
0.027 - (carrier), Mi4 0.018 - (tau 150 ms, carrier), TmY20 0.017 + (carrier, lowering), L3 0.016 + (carrier, tau 40
ms): raising +3.2e-5 vs lowering -2.9e-5, cancellation 0.96. **TmY21** (z -0.3; carriers 0.23, fallback tier): TmY5a
0.097 - (null), TmY13 0.076 + (null), Tm20 0.058 + (carrier), Tm5a 0.051 + (carrier, lowering), Dm3a 0.034 -
(carrier), TmY20 0.021 + (carrier): raising +8.0e-6 vs lowering -2.8e-5, cancellation 0.45 -- at TmY21 the carriers
are a quarter of the input and they mostly lower; the loss is dilution by non-carriers (its two largest inputs TmY5a
/ TmY13 are themselves non-carriers). **LC11** (fallback tier on 100 %, E 0.68 / I 0.32): T3 0.233, T2 0.122, Tm6
0.082, T2a 0.082, Tm12 0.065, TmY18 0.052, Li15 0.046 (-), MeLo10 0.040 (-) -- carrier share **0.012** (nothing
above the null reaches it). **LC10a** (exact tier, E 0.58 / I 0.42): Tm5Y 0.121, TmY21 0.066, Tm5a 0.050 (carrier),
Li22 0.049 (-), Tm3 0.038 (carrier), ... carrier share 0.14, Tm3 raising vs Tm5a lowering, cancellation 0.34 -- and on
the LIF side TuTuA_2 7.3 % (silent), AOTU042 6.1 % (0.46 Hz, null), LC9 4.3 %, LC10c-1 / -2 3.9 / 3.1 % (null): no
input of either LC type carries the ball at any level.

Pair gains: no `DEFAULT_PAIR_GAIN` entry touches the carrier edges of T3 / T2 / Tm5Y / TmY21; the only gained edges
onto them are T4 / T5 -> T3 / T2 / Tm5Y / TmY21 at x2 with shares < 2 % (T5b -> T2 0.021 the largest), and T5a / c /
d -> LC11 / LC10a x2 at < 1 %. Time constants: the carriers into T3 / T2 are the fast 8 ms types (Mi1, Tm3, Tm1, Tm2,
Tm4); into Tm5Y the slow Mi4 (150 ms) and L3 (40 ms) enter at 1.8 / 1.6 %. Operating points: 0.5 everywhere on this
stage (linear units); T4 / T5 at 0 are one to two steps upstream of nothing here.

LC pooling (`lc_pooling`, generated from `trace.column_of_cells` on a CPU retina): LC11 143 / 143 cells with
columns, median 94 columns per cell (p10 71, p90 110), best-three share 0.304, median 454 rate-unit inputs; LC10a
275 / 275, 32 columns (19-44), 0.482, 128 inputs. `out_norm l1`, mean `in_syn` 2,398 (LC11) / 910 (LC10a), `gain_out`
100 mV, clip 35 mV.

**The cancellation fractions with their scatter** (`perrun`, table `cancellation_per_run`: run r's signed figure is
(stimulus[r] - control[r]) alone, the carriers / signs / shares are the pooled trace's; the pooled number is the
same arithmetic on the mean-over-runs figure):

| target | pooled (trace_obj, 5 runs) | per run, trace_obj (5) | per run, base arm (4) | fb0 arm (deterministic) |
|---|---|---|---|---|
| T3 | 0.921 | 0.73 +- 0.25 (0.53 .88 .42 .98 .85) | 0.84 +- 0.20 (.96 .97 .55 .88) | **0.964** (+6.07e-5 / -5.65e-5) |
| T2 | 0.988 | 0.52 +- 0.23 | 0.86 +- 0.07 | 0.683 |
| Tm5Y | 0.956 | 0.20 +- 0.12 | 0.47 +- 0.41 | 0.784 |
| TmY21 | 0.445 | 0.29 +- 0.30 | 0.29 +- 0.15 | 0.808 |

The single-run values are biased low (a noisy term never cancels exactly) and scatter by +-0.2-0.4, so the pooled
0.92 / 0.99 / 0.96 are not what one run shows; the clean number is the deterministic arm's: **T3 0.96, T2 0.68, Tm5Y
0.78, TmY21 0.81**.

**What the fraction is and is not.** It is a *descriptive* ratio of the two opposing sums in the type-mean linear
decomposition -- how much of the raising sum the lowering sum removes. It is **not** a prediction of the target's
measured figure, in sign or in size, and the generating table says so: on the deterministic fb0 arm
(`perrun.json`, table `cancellation_summary`) the residual `linear_estimate` and the measured
`own_signed_figure` are T3 **+4.20e-6 vs -2.25e-4** (54x too small, opposite sign), T2 -1.91e-5 vs +1.26e-3, Tm5Y
+6.28e-6 vs +9.52e-4, TmY21 -3.04e-6 vs -1.88e-4, LC10a +1.57e-6 vs +6.28e-3. So read the fraction as a statement
about the *input decomposition* -- the carriers' terms very nearly annihilate at all four types -- and read the
claim that the small-field figure is a residual off the **lesion arms** of section 4 (where holding either class
raises T3 2.6-3.9x and collapses its run scatter), not off this arithmetic. Wherever the "4-32 % residual" phrase
appears below it is that qualitative statement.

## 3. Reading of sections 1-2 before the counterfactuals

* The loss happens **one synapse after the last carriers**, at every input class of both LC types at once, and it
  is not a threshold or a gain: T3 / T2 are linear units at 0.5 with no pair gain and the same l2 normalisation as
  the carriers that feed them. Their input is 40-56 % carriers; what the carriers deliver sums to a small residual
  because the raised and the lowered figures arrive with the same synaptic sign (excitatory ACh on T3 / T2, exact
  tier), and the inhibitory Pm / Li / C3 arms carry either no object or the opposite one.
* On the LC10a chain the mechanism is dilution plus the same cancellation: Tm5Y / TmY21 take 23-32 % of their
  input from carriers whose signs cancel (0.78 / 0.81 deterministic) and the rest from non-carriers; both types are
  unprofiled (fallback = NT_SIGN), so the receptor tables neither confirm nor change a sign there.
* Whether this is a wiring fact, a rate-model fact, a sign fact or a dynamics fact is what section 4 separates.

## 4. The lesion arms

### 4.1 Manifest and mechanics

`PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py plan --out out/apply_object/les --runs 4 --name apobj-les`
-> `out/apply_object/les/batch.sh` (ONE `cluster_run.py` call, 96 jobs = 8 arms x {stim, ctrl, null} x 4 runs,
`--fetch out/apply_object/les/`) and `manifest.resolved.json`; **run `apobj-les-e38d76`: 96 job(s), 0 failed, 22.5
min** (`out/apobj-les_cluster.log`; every job `device cuda`, the lesion stamped into each recording's meta and
provenance). Every job is `scripts/interp_apply_object.py record --lesion <id> --arm <stim|ctrl|null> --seed <k>`:
`scripts/interp_trace.py record`'s object protocol verbatim (12 s scored after 3 s settle; the same 167,106-cell
recordings, so `analyse` runs the same trace with the same statistic and each arm's own none-vs-none null), with
the lesion installed first. Four runs per arm: the exact U p floor of 4 v 4 is 0.029, so a 'result' verdict is
reachable (3 v 3 floors at 0.10 and never is). Smoke test `apobj-smoke-84bc5f` (3 jobs, 0 failed, `--quick`)
preceded it.

| id | kind | what is applied | mechanism | question |
|---|---|---|---|---|
| base | none | the shipped model | -- | the same-batch baseline |
| fb0 | optic | `gain_fb = 0` | OpticParams override (interp_trace's `install_overrides`) | dynamics: the deterministic lobe; magnitudes, not verdicts, are read (its null is exactly 0) |
| inl1 | optic | `norm = 'l1'` | override | rate-model: the l2 input amplification |
| outl2 | optic | `out_norm = 'l2'` | override | rate-model: the LC pooling normalisation |
| rect | optic | `baseline_by_type` = T4/T5 0 + T2 0, T3 0 | override | rate-model: rectification at the lost stage (round 2's flag, now with a null) |
| t3_off_held | edges | Tm1\|Tm4 -> T3 x 0 | c.W entries x 0 (11,174 entries, 104,478 synapses, 22.4 % of T3's raw input; 3,447 pre / 1,940 post cells) + receptor fast_sign 0 | wiring: does the ON figure pass T3 alone? |
| t3_on_held | edges | Mi1\|Tm3 -> T3 x 0 | 12,279 entries, 141,213 synapses, 30.3 % | wiring: does the OFF figure pass T3 alone? |
| t3_off_flip | edges | Tm1\|Tm4 -> T3 x -1 | entries negated + fast_sign negated | COUNTERFACTUAL sign: the data say + (exact tier, nAChR); asks, does not propose |

The `edges` kind (this script's addition to the lesion tool's kinds, needed because the lesion tool's population
mask cannot address one post-synaptic class and its probes run as subprocesses outside its in-process mask): the
loaded graph's stored entries of the block are multiplied in place, the sparsity pattern is kept (explicit zeros),
`connectome.load` is made to return the edited graph to `room_demo.Sim` / `FlyBrain`, and `brain._receptor` is
wrapped so the receptor model's per-entry fast sign follows the factor (a matched entry's table sign would
otherwise override a flip). The rate lobe's normalisation denominators come from the neuron table and are untouched
(interp_lesion.md section 2's fan-in caveat does not arise here). `selftest` pins the arithmetic on the synthetic
graph.

What the linear reading of section 2 predicted, so that the recordings could refute it: with the OFF carriers held,
T3's carrier terms no longer cancel and the figure grows (ON sign); with the ON carriers held, the lowering sum alone
(OFF sign); with the OFF carriers flipped, the two add -- ordering flip > held > shipped. If T3 reaches 'result' in
the first two arms the loss at T3 is the convergence itself (a wiring fact under a linear sum); if only the flip
restores it, a sign fact; if none does while `fb0` shows a per-cell signal of the medulla's size, a dynamics fact; if
`inl1` / `outl2` / `rect` move T3 / T2 / LC11 and the edge arms do not, a rate-model fact.

### 4.2 Results -- the type x arm matrix (`analyse` -> `out/interp/apply_object/lesions.json`, consoles in `lesions_console.txt`; per-arm traces `trace_<arm>.json`)

`PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py analyse --dir out/apply_object/les` -- each arm's four
stimulus runs against its own four none-vs-none draws (z on the null SD, exact U, verdict 'result' = |z| >= 3 and p
<= 0.05). Stim mean [z vs the arm's own null]; `*` result, `.` null; mV for the LC types, rate units otherwise:

| type | base | fb0 (det.) | inl1 (det.) | outl2 | rect | t3_off_held | t3_on_held | t3_off_flip |
|---|---|---|---|---|---|---|---|---|
| Mi1 | .0729 [+24.9]* | .0679 | .0164 | .0749 [+2.4]. | .0733 [+4.9]* | .0741 [+3.9]* | .0739 [+16.6]* | .0769 [+7.2]* |
| Mi4 | .0452 [+13.1]* | .0467 | .0031 | .0413 [+1.0]. | .0485 [+18.3]* | .0488 [+17.1]* | .0491 [+40.5]* | .0499 [+28.0]* |
| Tm3 | .1064 [+10.5]* | .1037 | .0134 | .1007 [+3.6]* | .0988 [+4.1]* | .1079 [+4.4]* | .1038 [+14.5]* | .1138 [+11.1]* |
| Tm1 | .0722 [+4.1]* | .0745 | .0237 | .0805 [+4.3]* | .0722 [+4.5]* | .0752 [+2.9]. | .0777 [+1.8]. | .0743 [+5.8]* |
| **T3** | .0240 [+1.4]. | **.0409** | .0010 | .0341 [+0.7]. | .0100 [+1.0]. | **.0623 [+3.5]\*** | **.0718 [+4.8]\*** | **.0939 [+5.8]\*** |
| **T2** | .0391 [+1.8]. | **.0664** | .0012 | .0495 [+0.6]. | .0233 [+2.8]. | .0453 [+0.3]. | .0461 [+3.0]. (p 0.2) | .0418 [+1.2]. |
| T2a | .0227 [+1.0]. | .0727 | .0009 | .0587 [+0.3]. | .0337 [+0.2]. | .0317 [-0.4]. | .0265 [+1.9]. | .0274 [+0.1]. |
| Tm6 | .0337 [+0.9]. | .0753 | .0015 | .0369 [-0.1]. | .0383 [+1.8]. | .0359 [-0.3]. | .0314 [-0.6]. | .0362 [+0.3]. |
| Tm12 | .0292 [+0.5]. | .0587 | .0022 | .0406 [+0.6]. | .0324 [+0.0]. | .0292 [-0.2]. | .0244 [+0.3]. | .0249 [-0.1]. |
| TmY18 | .0312 [+2.8]. | .0455 | .0022 | .0402 [+0.2]. | .0365 [+1.8]. | .0374 [-0.0]. | .0329 [+0.6]. | .0341 [+0.2]. |
| **Tm5Y** | .0477 [+2.2]. | **.0722** | .0016 | .0674 [+0.9]. | .0465 [+1.4]. | .0483 [+0.8]. | .0513 [+3.0]. (z 2.97) | .0507 [+1.2]. |
| **TmY21** | .0268 [-0.1]. | **.0624** | .0003 | .0511 [+0.3]. | .0335 [-0.5]. | .0355 [-0.5]. | .0289 [-0.1]. | .0301 [-0.2]. |
| TmY13 | .0248 [+1.1]. | .0345 | .0003 | .0405 [+0.2]. | .0284 [-0.0]. | .0202 [-0.6]. | .0211 [-0.4]. | .0190 [-0.5]. |
| TmY5a | .0351 [+0.4]. | .0839 | .0005 | .0702 [+0.0]. | .0393 [+0.4]. | .0368 [-0.4]. | .0419 [+1.4]. | .0344 [+0.0]. |
| T4c | .0533 [+15.4]* | .0505 | .0017 | .0483 [+1.8]. | .0533 [+4.8]* | .0541 [+8.9]* | .0542 [+12.0]* | .0532 [+8.4]* |
| LPLC2 (mV) | .3441 [+1.5]. | .5395 | .0210 | 2.938 [+0.2]. | .3059 [+2.4]. | .3336 [+3.3]* | .3527 [+8.7]* | .3330 [+2.4]. |
| **LC11** (mV) | .0451 [-0.8]. | **.0456** | .0003 | 1.902 [-0.3]. | .0993 [+1.2]. | .0661 [+0.7]. | .0302 [-0.8]. | .0765 [+1.2]. |
| **LC10a** (mV) | .0540 [-0.7]. | **.0797** | .0023 | 1.364 [-0.2]. | .0898 [+0.8]. | .0763 [-0.3]. | .0879 [+0.3]. | .0847 [+0.8]. |

The **fb0 and inl1 arms are deterministic**: their four runs are identical to four digits and their none-vs-none
null is exactly 0.0000 in every type, so the tool's z is nan / 1e7-1e9 and its verdict 'result' on p 0.029 is
degenerate (`note null_sd_zero`). The JSON's `restores` list holds **48 rows, of which 41 are that degeneracy**
(fb0 21, inl1 20) and **7 are not** (T3 x3, L5 x2, LPLC2 x2) -- a trace-tool defect noted in 7.4. Those two columns are read as magnitudes; against the base arm's
own 4-draw null the fb0 magnitudes sit at z **T3 +4.5, T2 +5.5, Tm5Y +4.7, TmY21 +5.2, Mi1 +22.9, LC11 -0.8, LC10a
-0.4** (`z_vs_base_null` in `matrix`).

Per-run values of the six stage types (stim runs 0-3 / null draws 0-3):

| arm | T3 | T2 | Tm5Y | TmY21 | LC11 (mV) | LC10a (mV) |
|---|---|---|---|---|---|---|
| base | .022 .032 .020 .022 / .009 .021 .017 .019 | .046 .035 .040 .035 / .018 .033 .021 .031 | .049 .053 .042 .047 / .013 .028 .029 .036 | .037 .019 .021 .030 / .018 .028 .033 .031 | .041 .078 .035 .028 / .036 .107 .054 .133 | .047 .059 .056 .054 / .046 .111 .064 .219 |
| fb0 | .0409 x4 / 0 x4 | .0664 x4 / 0 | .0722 x4 / 0 | .0624 x4 / 0 | .0456 x4 / 0 | .0797 x4 / 0 |
| outl2 | .054 .020 .045 .017 / .019 .045 .019 .019 | .065 .029 .068 .037 / .032 .065 .026 .037 | .130 .042 .056 .042 / .038 .083 .026 .028 | .104 .026 .054 .020 / .034 .083 .028 .032 | 2.91 1.99 0.60 2.10 / 1.12 4.61 1.23 2.28 | 3.08 0.75 0.85 0.77 / 1.17 3.42 0.79 1.27 |
| rect | .009 .010 .008 .013 / .006 .010 .010 .006 | .023 .022 .020 .028 / .009 .016 .016 .009 | .041 .052 .048 .045 / .023 .038 .044 .027 | .022 .039 .025 .049 / .025 .045 .044 .037 | .092 .110 .072 .123 / .078 .073 .022 .085 | .086 .103 .067 .103 / .082 .065 .069 .098 |
| t3_off_held | **.062 .065 .061 .062** / .028 .038 .027 .013 | .033 .049 .042 .058 / .048 .058 .032 .028 | .050 .047 .049 .048 / .044 .053 .030 .019 | .027 .031 .036 .048 / .049 .065 .027 .033 | .076 .072 .048 .069 / .002 .073 .045 .061 | .072 .083 .084 .066 / .040 .195 .083 .080 |
| t3_on_held | **.071 .070 .074 .072** / .021 .040 .020 .019 | .034 .043 .053 .054 / .032 .039 .038 .036 | .046 .053 .052 .054 / .030 .040 .035 .025 | .027 .029 .037 .023 / .026 .047 .024 .025 | .042 .023 .041 .015 / .021 .044 .058 .068 | .106 .067 .092 .086 / .091 .048 .054 .123 |
| t3_off_flip | **.093 .096 .093 .093** / .043 .020 .017 .020 | .038 .060 .035 .035 / .044 .027 .017 .027 | .054 .044 .050 .055 / .056 .019 .015 .017 | .028 .045 .030 .018 / .050 .033 .024 .024 | .119 .096 .067 .023 / .048 .080 .045 .031 | .153 .066 .066 .053 / .097 .071 .057 .057 |

**Restores** (verdict 'result' where the base's is not, excluding the degenerate fb0 / inl1 rows): **T3 in
`t3_off_held` (z +3.5, U 16/16, p 0.029), `t3_on_held` (+4.8) and `t3_off_flip` (+5.8)** -- and four scatter items:
L5 in `rect` (+3.5) and `t3_off_flip` (+4.9), LPLC2 in `t3_off_held` (+3.3) / `t3_on_held` (+8.7), whose stim values
did not move from the base's (LPLC2 0.334 / 0.353 vs 0.344; L5 0.034 / 0.035 vs 0.030) and whose null means did not
either (LPLC2 0.168 / 0.175 vs the base's 0.178; L5 0.015 / 0.011 vs 0.015): only the null's SD happened to be
small in those arms (LPLC2 0.050 / 0.021 vs 0.110; L5 0.006 / 0.005 vs 0.009) -- the holdOptic caveat of
interp_lesion.md in another guise: with 4 v 4 draws a 'result' whose stim value is unchanged is the null's
scatter. **Loses** (10 rows): `outl2` Mi1 / Mi4 / T4c / T4d / Tm20 (their nulls double under the l2 output stage's
larger feedback), `t3_off_held` Tm1 / Tm2 and `t3_on_held` Tm1 / Tm2 / T5a (z 1.8-2.9 with stim values unchanged)
-- scatter again.

**State this as a general rule of the assay, not as a footnote to two rows: a per-arm verdict of `result` is not
by itself a restoration.** A size-matched control run in a replicate batch shows the mechanism cleanly:
`t3_nc_held` (the non-carrier hold) reaches T3 `result` at **z +3.56, U 16/16, p 0.0286** purely because its own
null floor fell to 0.0193 +- 0.0031 from the base arm's 0.0233 +- 0.0052, while its stimulus value did not move
(0.0303 vs 0.0250, `delta_vs_base` +0.005, `moved_vs_base` `not moved`;
`out/interp/apply_object/skeptic_lesions.json`). **Read `delta_vs_base` and `z_vs_base_null`, never the per-arm
verdict alone.** By that rule the three T3 edge arms stand (delta +0.038 / +0.049, `moved`) and the L5 / LPLC2
rows do not.

### 4.3 What the edge arms say

T3's figure under the three edge arms: **0.062 +- 0.002, 0.072 +- 0.002, 0.094 +- 0.001** (4 stochastic runs each)
against the shipped model's 0.024 +- 0.005 and the deterministic 0.041 -- 2.6x / 3.0x / 3.9x the shipped value, in the
predicted order (flip > ON held > OFF held > shipped), and with a run-to-run scatter that has dropped to the
medulla's (Mi1 +- 0.001) from the shipped model's +- 0.005. The per-run cancellation at T3 under the arms is 0.20 +-
0.08 / 0.24 +- 0.32 / 0.14 +- 0.10 (section 2's arithmetic on the base carriers minus the held class; `perrun`): the
figure is now a direct response, not a residual. LC11 does not follow (0.066 / 0.030 / 0.077 mV, z +0.7 / -0.8 / +1.2
against nulls of 0.045 / 0.048 / 0.051): T3 is 21 % of LC11's LIF input (484 of ~2,300 synapses per LC11 cell), an
LC11 cell pools 94 columns under `out_norm l1`, and a 0.06-0.09 rate-unit figure in the few columns the ball
crosses is 0.02-0.03 mV of drive. T2 / Tm5Y / TmY21 are untouched by the T3 arms, as they should be (their carriers
are not Tm1 / Tm4 / Mi1 / Tm3 -> T3).

### 4.4 The rate-model arms

`inl1` divides the lobe's input by the l1 sum instead of the l2 norm: every figure shrinks 5-100x (L1 0.098 vs 0.126,
Mi1 0.016 vs 0.073, T3 0.0010, LC11 0.0003 mV) and the whole simulation becomes deterministic (null exactly 0 in
every type: the lobe's drive no longer produces a spike anywhere, so no feedback returns) -- optic_measures.md 7's
'~110x quieter linear lobe'. `outl2` multiplies the LC drives 30x (LC11 1.90 +- 0.96 mV, LC10a 1.36 +- 1.14, LPLC2
2.94; nulls 2.31 / 1.66 / 2.56) and changes nothing upstream (T3 0.034 vs null 0.026, z +0.7; T2 z +0.6) -- the
pooling normalisation scales signal and null alike. `rect` (T2 / T3 at operating point 0) halves T3 / T2 (0.010 /
0.023; z +1.0 / +2.8) and leaves the LC types at the null (LC11 0.099 vs 0.065 +- 0.028, z +1.2). None of the three
restores a signal at any stage.

### 4.5 The deterministic projection (`perrun`, table `projection`) -- is the fb0 figure present under the feedback?

Each stochastic arm's per-cell figure (the `best_cell` statistic before its max) regressed, type by type and run by
run, on the fb0 arm's: the slope <f_fb0, f_run> / <f_fb0, f_fb0>, Pearson r, and the run's value at the cell that is
fb0's best (mean +- sd over the arm's 4 runs):

| type (n cells) | fb0 figure at its best cell | base: value at that cell | base: own best cell | base slope on fb0 | r |
|---|---|---|---|---|---|
| Mi1 (1773) | 0.068 | **0.073 +- 0.001** | 0.073 +- 0.001 | 0.53 +- 0.17 | 0.70 +- 0.18 |
| Mi4 (1772) | 0.047 | **0.045 +- 0.002** | 0.045 +- 0.002 | 0.42 +- 0.23 | 0.57 +- 0.24 |
| Tm3 (2054) | 0.104 | **0.106 +- 0.003** | 0.106 +- 0.003 | 0.48 +- 0.18 | 0.64 +- 0.23 |
| Tm1 (1777) | 0.074 | **0.070 +- 0.004** | 0.072 +- 0.003 | 0.42 +- 0.22 | 0.61 +- 0.28 |
| **T3** (1940) | 0.041 | **0.003 +- 0.009** | 0.024 +- 0.005 | 0.24 +- 0.28 | 0.40 +- 0.46 |
| **T2** (1630) | 0.066 | **0.017 +- 0.018** | 0.039 +- 0.005 | 0.25 +- 0.26 | 0.45 +- 0.44 |
| **Tm5Y** (898) | 0.072 | **0.018 +- 0.012** | 0.048 +- 0.005 | 0.28 +- 0.24 | 0.46 +- 0.39 |
| **TmY21** (372) | 0.062 | **0.013 +- 0.014** | 0.027 +- 0.008 | 0.20 +- 0.24 | 0.33 +- 0.41 |
| TmY5a (1364) | 0.084 | 0.016 +- 0.014 | 0.035 +- 0.006 | 0.18 +- 0.23 | 0.34 +- 0.46 |
| LC11 (143, mV) | 0.046 | 0.007 +- 0.022 | 0.045 +- 0.022 | 0.04 +- 0.47 | -0.03 +- 0.24 |
| LC10a (275, mV) | 0.080 | -0.019 +- 0.019 | 0.054 +- 0.005 | -0.10 +- 0.12 | -0.12 +- 0.16 |
| LPLC2 (185, mV) | 0.539 | 0.024 +- 0.096 | 0.344 +- 0.011 | 0.34 +- 0.17 | 0.52 +- 0.24 |

The medulla's deterministic best-cell figures are present to the digit under the shipped feedback (the slope < 1
there is the noise on the other 1,700 cells); the small-field stage's are **absent at the same cells** (T3 7 %, T2
26 %, Tm5Y 25 %, TmY21 21 % of the deterministic value, each within its own sd of 0), and the arms' own best cells
(0.024-0.048) are the maxima of noise (they equal the arms' null floors). The same holds in every stochastic arm
that leaves the carriers alone (outl2 T3 0.004 +- 0.015, rect -0.003 +- 0.007). The entry points of the feedback do
not explain it (table `feedback_share`, structural: spiking units supply 2.8 % of T3's, 3.1 % of T2's, 5.4 % of
Tm5Y's, 17.0 % of TmY21's shaped-weight |input| against 3.6 % of Mi1's, 5.5 % of Mi4's, 4.0 % of Tm3's -- LoVC16 the
largest at T3 / T2 with 2.2-2.4 %), and the |dr| level does not (T3 0.084 with, 0.085 without feedback): what
distinguishes the two stages is that the medulla's figure is a direct response and the small-field stage's is the
residual of two opposing sums (section 2: 0.96 / 0.68 / 0.78 / 0.81 cancelled), which independent perturbation of
the two channels re-signs from run to run. That the T3 edge arms give a robust figure (4.3) under the same feedback
closes the argument.

## 5. The size ladder for Neurome (docs/NEUROME_INTERFACE.md section 3) -- run, exported

`PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py ladder-plan --out out/apply_object/ladder --runs 5 --name apobj-lad`
-> `out/apply_object/ladder/batch.sh` (ONE call, 40 jobs = 4 sizes x {ball, none-vs-none null} x 5 runs of
`scripts/interp_export.py record`, the retina captured at seed 0 of both arms of every size) and `ladder.json`.
Cluster run **`apobj-lad-287a43`: 40 job(s), 0 failed, 11.6 min** (`out/apobj-lad_cluster.log`; every job `device
cuda` NVIDIA B200, torch 2.11.0+cu128, the shipped model `sign` / `abs`, 12 s window after 3 s settle; recordings
`out/apply_object/ladder/d{045,114,200,300}_{stim,null}_s{0..4}.{json,_cells.npz,_prov.json,.txt}` + the two
`_retina.npz` per size). Then `PYTHONIOENCODING=utf-8 python scripts/interp_apply_object.py ladder --dir
out/apply_object/ladder --out out/export --json out/interp/apply_object/ladder.json` (CPU; console
`ladder_console.txt`): one export per size and one summary export, every one `verify: problems none`, LC11 143 /
LC10a 275 bodies with both quantities, 167,106 known bodies, commit `0d32fd6e` verified by source hash on all 28
loaded files.

**Run directories for hand-off** (the interchange of NEUROME_INTERFACE section 1; `manifest.json` +
`readout_per_body.csv` (26,482 rows: LC11 / LC10a x {upstream_drive_mV, output_Hz}, LC10b / LC16 / LC4 / LPLC2, and
the graded T2 / T3 / Tm5Y / TmY21 / TmY13 / TmY5a / Mi1 / Mi4 / Tm3 per body as rate_deviation / abs_rate_deviation
with null_mean / null_sd / z_vs_null per body), `per_type.csv` (72 rows: every statistic x type against the size's
own null), `retina_columns.csv` (1,466) / `retina_bodies.csv` (5,895) / `retina_radiance.parquet` (1,759,200 rows =
1,200 frames x 1,466 columns x [UV, B, G, R]), `result.json`, `checks.json`):

| size | run directory |
|---|---|
| 4.5 deg | `out/export/objsize-d045-20260913T014556Z-47ed1383/` |
| 11.4 deg | `out/export/objsize-d114-20260913T014606Z-7a4eadbd/` |
| 20 deg | `out/export/objsize-d200-20260913T014616Z-5ece62d6/` |
| 30 deg | `out/export/objsize-d300-20260913T014626Z-d5048f74/` |
| summary (`size_tuning` 288 rows, `retina_footprint` 4, `runs` 40) | `out/export/export-20260913T014635Z-db22e3ea/` |

Geometry: the ball rests on the table 5 cm ahead of the eye and sweeps +-6 cm (+-50.2 deg of azimuth) in 3 s per
pass, 12 s scored after 3 s settle -- the object-sweep protocol with only the radius changed:

| id | nominal | radius (m) | angular diameter, probe convention 2 atan(r/d) | from the eye (1.2 mm above the table; centre r above it) | centre elevation |
|---|---|---|---|---|---|
| d045 | 4.5 deg | 0.001966 | 4.50 deg | 4.50 deg | +0.9 deg |
| d114 | 11.4 deg | 0.005 | 11.40 | 11.42 | +4.3 |
| d200 | 20 deg | 0.008816 | 20.00 | 20.08 | +8.7 |
| d300 | 30 deg | 0.013397 | 30.00 | 30.18 | +13.7 |

**The retinal sampling actually presented** (`retina_footprint`, from the replayed radiance of the seed-0 ball run
against the seed-0 blank run, which is identical over its 1,200 frames): columns dimmed by > 5 % at any frame of
the sweep, per-frame mean of columns dimmed > 5 % and > 50 %, the darkest column's radiance relative to the blank,
and the azimuth / elevation extent of the dimmed columns.

| size | columns dimmed (any frame) | per frame > 5 % | per frame > 50 % | darkest ratio | azimuth | elevation |
|---|---|---|---|---|---|---|
| 4.5 deg | 26 | 1.5 | 0.14 | 0.38 | -53 .. +51 deg | -3.0 .. +3.9 deg |
| 11.4 deg | 45 | 4.8 | 2.3 | 0.01 | -53 .. +55 | -3.0 .. +10.0 |
| 20 deg | 85 | 12.4 | 7.6 | 0.01 | -57 .. +55 | -3.0 .. +20.0 |
| 30 deg | 133 | 26.8 | 19.7 | 0.00 | -61 .. +59 | -3.0 .. +29.3 |

So the 4.5 deg ball (median inter-column angle 3.95 deg) sits inside one column and never darkens any column by
more than 62 %: it is a contrast step of one column, not a silhouette; from 11.4 deg up the ball blacks out 2-20
columns per frame.

**Size tuning** (`size_tuning`; the primary statistics: spiking `diff_max_over_cells_mean_mv` (mV of optic drive),
rate units `diff_abs_best_cell_mean` (rate units); 5 ball runs vs 5 none-vs-none runs per size, exact U p floor
0.0079; z on the null SD; the full 288-row table including the firing-rate statistics is in the summary export):

| type | 4.5 deg: stim / null / z / verdict | 11.4 deg | 20 deg | 30 deg |
|---|---|---|---|---|
| **LC11** (mV) | .0657 / .0784 +- .0134 / **-0.9** / null | .0664 / .0476 +- .0299 / +0.6 / null | .0685 / .0478 +- .0126 / +1.6 / null | **.1710 / .0539 +- .0242 / +4.8 / result** (U 25, p 0.0079; Welch +6.2) |
| **LC10a** (mV) | .0853 / .0876 +- .0211 / -0.1 / null | .0873 / .0736 +- .0205 / +0.7 / null | .0972 / .0747 +- .0080 / +2.8 / null (U 21, p 0.095) | **.2084 / .0764 +- .0153 / +8.6 / result** (U 25; Welch +8.0) |
| T2 | .0355 / .0257 / +1.5 / null | .0496 / .0325 / +1.7 / null | .0665 / .0445 / +1.5 / null (U 24, p 0.016) | **.0974 / .0288 / +10.2 / result** |
| T3 | .0237 / .0190 / +1.0 / null | .0286 / .0192 / +2.1 / null (U 25, p 0.0079) | .0309 / .0257 / +1.5 / null | **.0443 / .0230 / +4.4 / result** |
| Tm5Y | .0294 / .0275 / +0.2 / null | .0624 / .0307 / +2.6 / null (U 23) | .0831 / .0452 / +2.2 / null (U 25) | **.1338 / .0268 / +38.5 / result** |
| TmY21 | .0317 / .0319 / -0.0 / null | .0433 / .0300 / +1.9 / null | .0324 / .0404 / -0.6 / null | **.0526 / .0271 / +8.9 / result** |
| TmY13 | +1.1 null | +1.0 null | -0.7 null | **+6.7 result** |
| TmY5a | +0.8 null | +2.5 null | **+3.5 result** | **+17.2 result** |
| Mi1 | **+4.5 result** (.0259 / .0098) | +23.5 result | +10.1 result | +31.6 result |
| Mi4 | **+5.1 result** (.0157 / .0077) | +70.7 result | +67.3 result | +39.5 result |
| Tm3 | +0.8 null (.0266 / .0197) | +21.5 result | +18.4 result | +27.5 result |
| LPLC2 (mV) | -0.5 null | +2.6 null (U 24) | **+13.5 result** (.951 / .216) | **+98.7 result** (2.028 / .115; best-cell rate +3.3 Hz vs +0.6) |
| LC16 (mV) | +0.0 null | +0.3 null | **+10.7 result** (.352 / .086) | **+23.1 result** (.673 / .073) |
| LC4 (mV) | +0.4 null | +0.1 null | **+4.7 result** (.156 / .072) | **+28.7 result** (.490 / .063) |
| LC10b (mV) | -0.2 null | +1.1 null | -0.6 null | +1.8 null (U 22, p 0.056) |

Per-run draws of the two LC types at 30 deg (stim / null, mV): LC11 .192 .112 .175 .178 .199 / .072 .013 .050 .066
.068; LC10a .169 .218 .181 .221 .254 / .077 .051 .077 .090 .087 -- every ball draw above every null draw in both.
The LC firing rates never move at any size (LC11 `diff_rate_hz_max_cell` 0.07-0.13 Hz vs null 0.08-0.17; LC10a
0.28-0.42 vs 0.35-0.48): the 30 deg signal is +0.12 / +0.13 mV of drive, 50x below the 7 mV pass level, on cells
that fire 0.002 / 0.02 Hz.

**Reading.** (1) The medulla carries every size including the one-column 4.5 deg step (Mi1 +4.5, Mi4 +5.1; Tm3 not
at 4.5 deg, +21.5 from 11.4 deg). (2) The small-field stage and the two LC types carry **only the 30 deg ball** --
and T2 / T3 / Tm5Y / TmY21 / TmY13 / TmY5a all switch on at the same size, with LC11 (+4.8) and LC10a (+8.6)
behind them: the loss of sections 1-4 is a loss of *small* objects specifically, and what passes at 30 deg is a
silhouette that darkens ~20 columns at once, the regime in which cancellation between neighbouring ON / OFF
carriers no longer removes the mean and a residual survives the feedback. (3) The loom chain (LPLC2, LC16, LC4)
reaches `result` at 20 deg with z 5-100 -- its threshold is below 20 deg, not at it (point 5) -- and LPLC2's best
cell fires (+3.3 Hz at 30 deg): the size ordering of the model is
loom detectors < LC10a < LC11 in threshold and the opposite of the animal's (LC11 prefers ~5-10 deg objects, LPLC2
expansion). (4) The 20 deg point is where LC10a (+2.8, U 21) and T2 (U 24, p 0.016) begin to separate from their
nulls without reaching the |z| >= 3 rule; at 4 sizes x 5 runs the threshold lies between 20 and 30 deg for both LC
types. (5) **The loom chain's threshold lies *below* 20 deg, and this ladder's own 11.4 deg point does not
reproduce object_sweep.md 8.7.** There LPLC2 at the 11.4 deg ball is stim +0.351 +- 0.038 vs null +0.139 +- 0.040,
z **+5.4**, "the one type at z > 3" (8.7, `sign-abs` row). The stimulus value reproduces here and in every
batch -- 0.301 (`trace_obj`), 0.344 (`lesions` base), 0.363 (ladder d114), 0.349 (`skeptic_lesions` base) -- but
the nulls are 0.165 +- 0.066, 0.178 +- 0.110, 0.176 +- 0.072 and 0.165 +- 0.065, so z reads **+2.05 / +1.51 /
+2.60 / +2.82, all `null`**. LPLC2's verdict at 11.4 deg is a **null-draw difference**, not a size threshold, and
flips between batches; quote the difference over the null, not the z. This is the diagnostic Neurome asked for, with the per-body tables and the retinal sampling in the run
directories above; it is not a pass criterion and nothing was tuned.

## 6. What follows (diagnoses, not proposals)

* The **two facts compose**: a wiring fact (the ON / OFF convergence leaves a 4-30 % residual) and a dynamics fact
  (the stochastic feedback re-signs the residual). Either alone would leave a signal: the deterministic lobe passes
  the residual at 4.5-5.5 SD above the shipped floor, and a direct T3 response survives the feedback. A model that
  kept the connectome's convergence and had a deterministic lobe would carry the ball to T3 / T2 / Tm5Y / TmY21 at a
  per-cell level -- and still not to LC11 / LC10a, whose loss is the pooling.
* Whether the animal's T3 / T2 do something the linear sum cannot (ON / OFF rectification before summation, a
  nonlinearity the rate unit lacks) is a question about the unit model, not about the data: `rect` (a ReLU at the
  T-cell output) is not it; a rectification of each input class before the sum was not tested and is the natural
  next arm (an `edges`-kind lesion cannot express it; it needs an OpticParams hook).
* The unprofiled types (Tm5Y, TmY21, LC11: fallback tier on 100 %) were not sign-tested. Their deterministic figures
  exist at the medulla's size, so a receptor profile would change their sign fact only if it changed a carrier
  edge's sign; the receptor data for those types is the open item, not a sign choice.
* The LC pooling (94 / 32 columns, `out_norm l1`, gain_out 100 mV) is the one place where no data in the tree
  decides (optic_measures.md 6): the size ladder gives Neurome the per-body numbers to compare against LC11 / LC10a
  recordings at four sizes.

## 7. Files, defects, provenance

### 7.1 Generators and results

`scripts/interp_apply_object.py` (all subcommands; `selftest` pins the edge lesion, the size geometry, the batch
lines, the per-run cancellation and the projection on the synthetic graph of tests/test_interp.py);
`out/interp/apply_object/`: `trace_obj.json` (+ `_rate.json`, `_console.txt`), `stage.json`, `trace_<arm>.json` (8
arms), `lesions.json` (+ console), `perrun.json` (+ console), `ladder.json`, `ladder_<size>.json`;
`out/apply_object/les/{batch.sh,manifest.resolved.json}` + 96 x 5 recording files, `out/apply_object/ladder/` + 40
runs, `out/apply_object/smoke/`; cluster logs **`out/apobj-smoke2_cluster.log`** (the 3-job `apobj-smoke-84bc5f`
run), `out/apobj-les_cluster.log`, `out/apobj-lad_cluster.log`. `out/apobj-smoke_cluster.log` is **not** that run:
it is a 3-line ssh failure (`Could not resolve hostname <cluster-host>`) for an earlier run id,
`apobj-smoke-15b44e`; it is kept as evidence of the retry, not cited. Every JSON carries the resolved LIFParams / OpticParams, `type_path_gain`, the realised
device and the cache fingerprint (`ef23cc27bea13be7f6a96f3c04fd3737`). **Where the device evidence lives:** the
`apobj-les` and `apobj-smoke2` consoles print a device line per job; the `apobj-lad` console does not, so "every
job `device cuda` (NVIDIA B200)" for the ladder batch is read from each recording's own
`provenance.execution.device` / `device_name` (which do say `cuda` / NVIDIA B200 in all 40), not from the log.

Exports (`scripts/interp_export.py analyse --result ... --out out/export`; every one `problems: none`):

| Result | run directory |
|---|---|
| trace_obj (section 1, re-run with the raw-count fix) | `out/export/trace-20260913T024240Z-62a53da3/` (readout_per_body 16,628 rows; LC11 143 / LC10a 275 bodies) |
| lesions (section 4) | `out/export/lesion-20260913T025105Z-1d9bce84/` (matrix 280, sensitivity 245) |
| perrun (sections 2, 4.3-4.5) | `out/export/trace-20260913T025220Z-757f8262/` (cancellation_per_run 222, projection 640, feedback_share 33) |
| size ladder (section 5) | the five directories of section 5 |

Superseded (kept, not cited): `out/export/trace-20260913T005309Z-874ad771/` (trace_obj before the raw-count fix,
`raw_synapses_per_post` 0.0) and `out/export/lesion-20260913T020919Z-a0ff458f/` (the same lesion matrix, made
before the re-analysis; numbers identical).

### 7.2 The trace defect (build skeptic, trace 1) and its fix

`lost_inputs.raw_synapses_per_post` was 0.0 in every row because `trace.TypeGraph` and `trace.trace` took
`common.raw_counts(c)` with its default `with_sign0=True`, which returns `connectome.sign0_counts` -- non-zero only
where `W.data == 0` -- as the whole data vector. When this task began the fix was **already in
`flyverse/interp/trace.py`** (`full_raw_counts`, lines 111-132: |W| on the signed entries merged with the sign-0
counts on the explicit zeros, used at both call sites; file modified 19:40 local, by a concurrent task of the same
workflow). This task did not edit `trace.py`; it verified the fix (the 7 trace tests of `tests/test_interp.py`
pass; on the real cache `sign0_counts_available` is true and the T3 / LC11 rows above carry the cache's counts) and
re-ran `trace` and `analyse` on it -- every statistic, verdict and z is identical to the pre-fix run (the raw count
column is informational; the shares and figures come from the shaped weights).

The cancellation ratios the skeptic named as single-draw-set numbers are now quoted with their scatter (section 2):
T3 0.92 pooled / 0.73 +- 0.25 and 0.84 +- 0.20 per run / 0.964 deterministic; T2 0.99 / 0.52 +- 0.23 / 0.86 +- 0.07 /
0.683; Tm5Y 0.96 / 0.20 +- 0.12 / 0.47 +- 0.41 / 0.784; TmY21 0.45 / 0.29 +- 0.30 / 0.29 +- 0.15 / 0.808.

### 7.3 Defects in this script, fixed in this revision

* **`analyse` wrote its per-arm traces to a fixed path.** `cmd_analyse` built
  `OUT_JSON / f"trace_{lid}.json"` regardless of `--dir` and `--json`, so analysing a second recording set with
  its own `--json` silently overwrote the first set's per-arm traces (a replicate batch overwrote
  `trace_base.json` / `trace_t3_off_held.json` / `trace_t3_on_held.json`; they were restored by re-running
  `analyse --dir out/apply_object/les --lesions base,t3_off_held,t3_on_held`, which reproduced them to every digit
  -- T3 0.0240 z +1.39, 0.0623 z +3.51, 0.0718 z +4.81, only `run_id` and `created_utc` differing -- and the
  replicate's own traces are kept as `skeptic_trace_*.json`). The path is now `trace_dir_for(args)`: `--trace-dir`
  if given, else `--json`'s directory, else the tool's output directory. `perrun` reads the same location through
  a matching `--arm-traces`.
* **No CPU test covered this script.** `selftest` pinned the edge-lesion arithmetic, the size geometry and the
  projection but was never collected by pytest. `tests/test_interp.py` now carries an `ApplyObjectTests` class
  that runs those assertions on the 8-neuron synthetic graph plus the new trace-path derivation.

### 7.4 Defects found in the tools used (for their owners; not edited here)

* **trace**: with a degenerate null (SD exactly 0: the fb0 / inl1 arms) `compare` returns z nan / ~1e8 and verdict
  'result' on p 0.029 (`note null_sd_zero`); the lesion matrix's `restores` list therefore carries **41** spurious
  rows of its 48 (fb0 21, inl1 20), leaving 7 real ones (T3 x3, L5 x2, LPLC2 x2).
  A null of SD 0 should yield verdict 'undetermined' (magnitude only), not 'result'.
* **decompose / trace composition**: the dynamic decomposition at the lost stage produced no table on these
  recordings (`decompose.arm_params` reads `meta['arm']` = 'stim' / 'ctrl' as a receptor arm name); the static tables
  are complete.
* **lesion tool**: its kinds cannot address one presynaptic class onto one postsynaptic class (`t3_off_held` needed
  the `edges` kind added in this script); worth promoting into `flyverse/interp/lesion.py` with its receptor-sign
  wrapper.

### 7.5 Provenance

Shipped model throughout (receptor `sign` / `abs`, `type_nt_override` TmY14 glutamate / Mi19 serotonin / aMe8
acetylcholine, OpticParams as listed in section 1); dataset MaleCNS v1.0 flat connectome (body annotations sha256
`2177e246...`, weights `e35da783...`), 167,106 neurons, 25,578,600 entries, 121,460,584 |W|; cluster host `<cluster-node>`,
Linux 6.8, NVIDIA B200, torch 2.11.0+cu128, `event_driven` / `cuda_kernels` / `cuda_sparse warp`; the trace
recordings of interp_trace.md (`trv-dee1e7`); dt LIF 0.5 ms / optic 1 ms / frame 10 ms; brain seed = run index.
