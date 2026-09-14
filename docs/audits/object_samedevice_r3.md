# The arm/box confound, closed: base / rectify / suppress on one GPU model (`scripts/object_round3_samedevice.py`) -- object round 3, task same-device

**Status: original H200 batch and fresh-seed B200 replication complete, verified and analyzed.
Sections 0-2 retain the design around `out/objr3sd/predeclared.json` (2026-09-14T05:28:42Z,
before submission); sections 3 onward report the fetched data.** Nothing is adopted; every hook stays off by
default (the project rule); the arms are diagnoses on fixed anatomy.

Read first: `docs/audits/object_compare_r2.md` 3-9 (what round 2 found and its section 8 item 1: "a re-run of `base`
on the box that hosted `rectify` and `suppress` ... is owed before any vs-base row is quoted without the caveat"),
`docs/INTERP.md` 10.2 (a max-over-cells statistic never travels alone) and 10.4 items 2, 9, 11, 12.

Files: `scripts/object_round3_samedevice.py` (`plan` / `submit` / `run-job` / `slim` / `assemble` / `verify` /
`analyse`), `out/objr3sd/` (`batch.sh`, `jobs.json`, `predeclared.json`, `tree_state.json`, `submit_console.txt`,
then `meta/`, `sph_slim/`, `sph/`, `verify.json`, `per_body_sphere.csv`, `side_by_side_round2.csv`,
`samedevice.json` = the Result), `out/objr3sd_cluster.log` (the cluster console). The CPU smoke of the analysis on
hardlinked copies of the round-2 runs is `out/objr3sd/smoke/` (`analyse_console.txt`, `samedevice_smoke.json`;
nothing in it is a number of this batch).

## 0. Predeclared reading rules (`out/objr3sd/predeclared.json`, stamped 2026-09-14T05:28:42Z)

The rules are the `PREDECLARED` dict of the script, dumped by `plan` and copied into the Result unchanged by
`analyse` (which also checks that the live dict still equals the stamp). In prose:

* **The question.** Do round 2's rectify-vs-base results and suppress's nulls on the matched sphere ladder reproduce
  when `base`, `rectify` and `suppress` run on ONE GPU model in ONE submission -- the confound of
  `object_compare_r2.md` section 8 item 1 closed. A diagnosis on fixed anatomy; nothing is adopted.
* **The arms.** `base` (A, no override), `rectify` (C, `stream_rectify` = the four T3 / T2 stream entries) and
  `suppress` (E, `spatial_suppress` k 0.5 within 10 deg on Mi1 / Tm1 / Tm2 / Tm3 / Tm4) -- the `--optic` flags are
  produced by `object_round2_compare.optic_flags` from the same `ARMS` dict, so they are byte-identical to round 2's
  (`object_compare_r2.md` section 1). `rectify` because it is the one arm that carried the carrier figure (T3
  6.0 / 10.5 / 9.0x base); `suppress` because its sphere rows were all null while it cost the escape benchmark.
* **Replicate unit and counts.** One process = one (A, B) pair under one brain seed = one run. 5 object runs per
  rung x 5 rungs (4.5 / 8.8 / 11 / 20 / 30 deg) + 5 blank/blank nulls per arm = 30 runs per arm, 90 in all, the
  round-2 protocol verbatim (elevation 0, 5 cm, +-50 deg at 40 deg/s, dark ball, eye 0.15 m, headlamp, 12 s after
  3 s settle).
* **One submission, one box.** One `cluster_run.py` call; every job line carries the block token `fam_samedev` and the
  call passes `--arm-block fam` (`--no-balance-blocks`: the one block resolves to the least-loaded box), so every
  job of every arm is placed on one target. Jobs are blocked by SEED, not by arm: job `seed<s>` runs base, rectify
  and suppress in turn at every rung and the null (18 processes), so the arms share the box and its load history.
* **Device criterion.** The batch answers the question only if every run of every arm records the same
  `provenance.execution.device_name`, `object_round2_compare.verify_all` raises no device problem, and every
  arm-vs-base comparison row carries `same_device_as_reference = True`. Otherwise the batch is reported as not having
  closed the confound.
* **GPU model.** The task named a B200. At plan time the two B200 boxes of round 2 (`vast-a` / `vast-b`) were gone
  (ssh refused; disabled in `.cluster.json`) and the two live boxes are both NVIDIA H200 -- the model that hosted
  `rectify` and `suppress` in round 2. `base` therefore lands on the treatment arms' GPU model, which is exactly what
  `object_compare_r2.md` section 8 item 1 asked for. The realised `device_name` is read from every run, never assumed.
* **Window rule, P1, P2, families, Holm, verdicts:** round 2's, verbatim (`object_round2_compare.PREDECLARED`): per
  arm the three small rungs {4.5, 8.8, 11} = 3 members, Holm within, question (i) arm object runs vs the arm's own
  blank/blank runs, question (ii) arm object runs vs the base arm's object runs at the same rung; T3
  `diff_signed_best_cell` (P1), T2 the second carrier, LC11 windowed `drive_median` (P2); the baseline's window rule
  on the maps on disk (`out/objr2/rfmap_ship.csv`, `rfmap_fb0.csv`, `out/synth2/rfmap_150_fb0_p3.csv`).
* **The companion.** Beside every P1 / T2 row, the per-body RF-windowed median |drive| of the same type and rung
  (`abs_drive_median`, role `upstream_windowed`, the same two questions). The headline is a within-run maximum over
  1,940 T3 / 1,630 T2 cells and never travels without it.
* **Reproduction rules.**
  * `rectify` vs base **REPRODUCES** if, on this batch, rectify reads `result` with `p_holm <= 0.05` and diff > 0 vs
    base AND `result` vs its own null at every small rung on T3 and on T2 (round 2: `result` x3 on both questions at
    both types, `carries_small_field = true`); **PARTIAL** if at least one small rung on T3 satisfies both; otherwise
    **NOT REPRODUCED**. The ratio to base per rung is quoted beside round 2's 6.01 / 10.49 / 8.99 (T3) and
    3.75 / 3.27 / 3.13 (T2) as magnitudes -- two batches are two draws and are never tested row by row
    (`docs/INTERP.md` 10.4 item 2).
  * `suppress`'s nulls **REPRODUCE** if suppress reads `null` on both questions at every small rung on T3, T2 and
    LC11 `drive_median` (round 2: null x3 on both questions at all three). A `null` is an absence at 5 v 5 and is
    reported with its z and p, never as evidence of equality.
  * **LC11 follows in neither arm** is reproduced if the P2 family reads no rung with `result` on both questions for
    either arm (round 2: false 8/8).
  * **The companion null** (rectification leaves the typical cell where it was) is reproduced if every
    `abs_drive_median` vs-base row of rectify at T3 and T2 (small rungs; large rungs beside) reads `null` (round 2:
    null in all 48 arm x type x rung rows, ratios 0.73-1.27).
  * **Levels**: the blank-arm operating point per arm (T3 / T2 `dev_mean_b`, LC11 `drive_mean_mv_b`), a magnitude
    beside round 2's section 5 table.
* **Cross-batch, exploratory, no call.** `base` (this batch, one H200) vs `base` (round 2, B200): per type x
  statistic x rung, the 5 object runs of each batch through `compare_row`, and the 5 nulls likewise (role
  `cross_batch_base`). The two batches differ in more than the GPU model (submission time, concurrent load, the
  shipped tree -- section 2), so a difference here is a box-or-batch effect and an agreement is one draw's worth of
  evidence that the reference arm is stable across models.
* **Analysis code.** The sha256 of every analysis source (this script, `object_round2_compare.py`,
  `object_round2_baseline.py`, `probe_object_matched.py`, `flyverse/interp/common.py`) is in the stamp and again in
  the Result; a difference is reported, and a change to the reducers after the Result is written obliges a re-run on
  the same fetched batch (`docs/INTERP.md` 10.4 item 11).
* **Classification.** The rectify modes and the suppression k / radius are hand-set numbers, not data. NOTHING IS
  ADOPTED; every hook stays off by default and is bit-identical off on the CPU.

## 1. The batch (`objr3sd`, 5 jobs = 90 processes, ONE submission, ONE block)

`python scripts/object_round3_samedevice.py plan --out out/objr3sd --name objr3sd --minutes 240` ->
`python scripts/object_round3_samedevice.py submit --out out/objr3sd` (one `scripts/cluster_run.py --arm-block fam
--no-balance-blocks` call, `--fetch out/objr3sd/meta/ out/objr3sd/sph_slim/`, console teed to
`out/objr3sd_cluster.log`).

Every job line is `mkdir -p out/objr3sd/sph out/objr3sd/sph_slim out/objr3sd/meta && source .venv/bin/activate &&
python -c 'import torch; assert torch.cuda.is_available()' && python scripts/object_round3_samedevice.py run-job --job
seed<s> --fam fam_samedev --out out/objr3sd --runs 5 --seconds 12 --settle 3; st=$?; echo "run-job seed<s> exit $st";
exit $st`. `run-job` regenerates its process list on the box from the same arguments, runs `probe_object_matched.py
run` 18 times in sequence (base, rectify, suppress at 4.5 deg; then 8.8; 11; 20; 30; then the three nulls), runs
`object_round2_compare.cmd_check_job` (every expected JSON present, every console says `device cuda`, no Traceback),
then slims its own 18 npz into `sph_slim/` and copies the JSON + console into `meta/`, and exits with the check's code.

**The slim (`docs/INTERP.md` 10.4 item 12, without losing the companion statistic).** A 12-s run npz is 114 MB, of
which 117 MB uncompressed is the two (1200, 12235) graded arrays `a__optic_dr` / `b__optic_dr`. The round-2 export's
`slim` drops them outright -- but the round-2 COMPARE analysis reads them: `abs_drive_median`, the per-body
RF-windowed median that must travel beside P1, is computed from exactly those arrays
(`object_round2_compare.analyse_sphere` -> `windowed_body_stats` over the UPSTREAM types). So this script's slim
keeps the COLUMNS of the four types the analysis reads (T3 1,940 + T2 1,630 + Tm5Y 898 + TmY21 372 = 4,840 of
12,235) and drops the other five types' columns (Tm3, Mi1, Mi4, TmY5a, TmY13 -- read by nothing in this analysis),
with `rate_idx` / `rate_body_ids` / `rate_types` subset in step and four `slim_*` arrays recording what was kept.
114 -> 50 MB per run; every other array is byte-identical. Verified on CPU on two round-2 runs
(`rectify_d110_obj_s0`, `base_null_s3`): `windowed_body_stats` on the slimmed file equals the full file's row for
row over all 4,840 bodies (`pd.testing.assert_frame_equal`), and the 34 / 29 other arrays compare equal (the
remainder are the subset arrays themselves). `assemble` copies `meta/` and `sph_slim/` into `sph/` locally so every
round-2 reader (`verify_all`, `analyse_sphere`, `probe_object_matched.verify_runs`) sees the layout it expects.

## 2. The CPU smoke, and the tree the batch shipped from

**The smoke** (`out/objr3sd/smoke/`): the 90 round-2 runs of `base`, `rectify` and `suppress` hardlinked from
`out/objr2c/sph/` and pushed through `analyse --force` (the round-2 devices trip the device check, as they should).
Every one of the 264 side-by-side rows (`side_by_side_round2.csv`) reads `verdict_agrees = True` with
max |z_r3 - z_r2| = 0.0 and max |p_holm_r3 - p_holm_r2| = 0.0 -- the analysis reused here IS round 2's, and the
Holm families are per arm so restricting to three arms changes no adjusted p. On that data the script reads
`closed: false`, `one_model: false`, `same-device vs-base rows 140/420`, prints `DEVICE-CROSSED` on every rectify /
suppress vs-base line, `rectify: REPRODUCES` (trivially, against itself), `suppress: REPRODUCE`, and the
cross-batch rows z 0.00 / p 1.000 throughout (the two "batches" are the same files). `Result.check()` none.

**The tree.** `out/objr3sd/tree_state.json`: HEAD `6ec2de1` plus other tasks' uncommitted edits (`flyverse/body.py`,
`brain.py`, `senses.py`, `motor.py`, `batch_body.py`, `batch_sim.py`; `git diff --stat` 9 files, +499 / -27), which
`cluster_run.py` ships with everything else. Against round 2's shipped tree (`out/objr2c/tree_state.json`, commit
`653179b4`), the sha256 of `flyverse/optic.py`, `brain.py`, `body.py`, `senses.py`, `fly.py`, `connectome.py` and
`interp/common.py` DIFFER (the merges since `653179b` and the edits above; e.g. `LIFParams.w_syn_by_nt`, default
`None`, byte-identical off); `scripts/probe_object_matched.py` and `flyverse/retina.py` are the SAME. Every arm of
THIS batch runs on this one tree, so the within-batch questions are untouched; the cross-batch base rows of section 5
carry a code difference as well as a box difference, which is why they were predeclared exploratory.
`provenance.source_fingerprint`, not the commit, identifies the code of every run (`docs/INTERP.md` 10.4 item 6).

## 3. What ran, and how the data got here

All 5 jobs in `objr3sd-d6f06a` completed: 90 paired recordings, 90 JSONs and 90 consoles,
all recording **NVIDIA H200**. `out/objr3sd/verify.json` reports no problems. Fable's
`FETCHED.txt` records the completed pull; `assemble` restored the reader's `sph/` layout
from `meta/` plus the 90 companion-preserving slim NPZs. The 90 raw full-optic NPZs
(114 MB each, 11 GB) were never pulled, and `out/objr3sd/FETCHED.txt` records that the
box was destroyed after that marker, so they are gone rather than merely remote; every
array needed for the declared LC and upstream comparisons is present in the slim.

The original polling client died. Its unfinished console is not evidence of unfinished
GPU work: `out/objr3sd/scheduler_receipt.json` records the actual Heimdall job states;
`recovered_cluster.log` explicitly labels the recovered **5 jobs, 0 failed** status.
The first scheduler submission was 05:29:08Z, after the 05:28:42Z stamp. Retries after
the earlier machine outage are part of the original batch, not extra independent runs.
No timing inference is made from these jobs or the shared house machine.

The independent B200 replication uses `out/objr3sd_house/`, seeds 1000-1004,
the same 90-process design and one submission named `r3sdcheck`, pinned to <cluster-node>.
Its predeclaration and tree snapshot were written before submission. Results are kept
separate from H200; neither the seeds nor GPU arithmetic imply bit identity.

## 4. P1 / T2 / P2 on one GPU model, beside round 2

The reference arm is not identical across the two batches. On H200, `base` T2 vs its own blank/blank null is
`null` at every small rung (z +0.35 / +1.02 / +1.96). On B200 it reads `result` at 8.8 deg (z +6.12652, Holm
.023810) and 11 deg (z +9.29988, Holm .023810). T3 base is `null` on both. Every within-batch comparison is
still same-device, but the shipped model's own T2 maximum separates from blank on one GPU model and not the
other, which is itself part of what this batch reproduced.

`out/objr3sd/samedevice.json` is the analyzed Result; `side_by_side_round2.csv`
and `per_body_sphere.csv` retain the complete comparisons and body-level inputs.
The declared same-device criterion is met. **Rectify REPRODUCES** on both questions
at all three small rungs for T3 and T2. **Suppress's primary null pattern REPRODUCES**;
LC11 follows in neither treatment. A null is not an equivalence claim.

The carrier maxima below are mean +/- sample SD over **five runs**, in optic-drive
units. The companion is the median over bodies of the absolute RF-windowed drive
difference, computed within each run and then summarized over those same five runs.

| Type | Diameter | Base maximum | Rectify maximum | Round-2 rectify/base | H200 rectify/base | Base companion | Rectify companion |
|---|---:|---:|---:|---:|---:|---:|---:|
| T3 | 4.5 | .003425 +/- .001004 | .021512 +/- .001608 | 6.015 | 6.281 | .006865 +/- .000417 | .006705 +/- .000705 |
| T3 | 8.8 | .003881 +/- .001077 | .036979 +/- .001796 | 10.494 | 9.528 | .005286 +/- .001140 | .006377 +/- .000437 |
| T3 | 11 | .003418 +/- .000705 | .044644 +/- .000625 | 8.988 | 13.063 | .004503 +/- .000515 | .005747 +/- .000520 |
| T2 | 4.5 | .009808 +/- .003177 | .023106 +/- .001736 | 3.745 | 2.356 | .010391 +/- .000863 | .010339 +/- .000886 |
| T2 | 8.8 | .012152 +/- .001197 | .040946 +/- .002798 | 3.268 | 3.369 | .011776 +/- .002531 | .013045 +/- .001376 |
| T2 | 11 | .015418 +/- .000772 | .047300 +/- .000553 | 3.131 | 3.068 | .011006 +/- .001488 | .011603 +/- .001957 |

Every rectify-vs-base carrier maximum above has Holm p = .02381. Its companion
comparisons are all null: T3 Holm p = 1, .30159, .09524; T2 Holm p = 1, .66667, 1 — all three `null`.
The two p's are not adjusted within the same family: the primary `P1_carrier` /
`T2_carrier` / `P2_lc11` families are the three small rungs (m = 3), while the
`upstream_windowed` companion rows are adjusted over all five rungs (m = 5) and the
`large_rungs` rows over two. The Result's `summary.reading` string describes every
`p_holm` as "Holm within the round-2 family (3 small rungs per arm x type x question)",
which is accurate only for those three primaries.
The raw maxima need not exceed the differently windowed medians: they are different
statistics, on different time windows. This is evidence for an extremal carrier
change, without an established typical-cell improvement.

The large rungs are not null in this batch either: rectify's T3 companion exceeds base at 20 deg
(.006050 +/- .000709 vs .004445 +/- .000414, z +3.88, Holm .039683) and 30 deg (.007377 +/- .000775 vs
.004259 +/- .000400, z +7.80, Holm .039683), and its T2 companion at 30 deg (.012665 +/- .001896 vs
.009673 +/- .000841, z +3.56, Holm .039683). Round 2 already carried the T3-at-30 row (z +9.80, ratio 1.827,
`result`), so this is a three-batch effect, not a device-specific one, and `companion_all_null_all_rungs` is
false in both round-3 Results. The typical-cell claim holds only at the small rungs.

LC11's signed population-median received-drive difference (mV), mean +/- run SD:

| Diameter | Base | Rectify | Suppress |
|---|---:|---:|---:|
| 4.5 | -.03668 +/- .11569 | -.00360 +/- .16645 | .03445 +/- .12508 |
| 8.8 | .03485 +/- .06044 | .05487 +/- .02300 | .03853 +/- .05756 |
| 11 | -.04669 +/- .05863 | .03049 +/- .07431 | .00053 +/- .05700 |

All small-rung P2 comparisons are null. Every LC11 window remains anatomical;
these are not measured RFs. The suppress companion at T3/4.5 has an unadjusted
`result` (z = -3.075), but Holm p = .12698: it is not a family-level finding.

## 5. Levels, the large rungs, and the cross-batch base rows

The Result's `levels_runs` and `levels` retain the blank-arm operating points with
run scatter; `sphere_comparisons` includes the 20/30-degree rungs and their medians.
The export carries these alongside each rung. Larger-rung and cross-batch rows do
not enlarge the predeclared three-member small-rung primary family.

`cross_batch_base` is exploratory. Round 2 ran on a different tree (18 of 44 fingerprinted files differ, `optic.py`, `brain.py` and
`interp/common.py` among them). The H200 and B200 round-3 batches, by contrast, have **identical** source
fingerprints: they differ only in GPU model, box, brain/env seeds (0-4 vs 1000-1004) and execution history.
That still does not isolate a causal GPU-model effect — seeds and box are confounded with model — but the code
is not a confound between them, and no comparison row between the two round-3 base arms was computed.
These batches are not pooled for inference.
Likewise this sphere-only replication supplies no new specificity or benchmark
pass: the round-2 evidence that rectification releases other stimulus responses and
suppression costs loom escape still matters to any adoption decision.

## 6. Reading, and what this batch cannot say

The same-box experiment supports the round-2 carrier diagnosis and closes that
experiment's arm/device confound. It does not establish small-object selectivity,
an LC11 rescue, behavioral benefit, or equality of suppression and the baseline.
Rectification modes and suppression strength/radius remain hand-set control arms.
Adoption would require a physiological rationale, same-device specificity and ON/OFF
controls, the full benchmark suite with independent draws, and the appropriate room
take-off/anti-runaway checks. No model default was changed here.

## 7. Independent computational skeptic pass

This is a separately implemented computational check by Astra, not a second human
or agent review. `check-raw` reconstructs time windows directly from fetched frame
arrays, without the analysis window reducer: **1,860 run-level values, zero
differences**. `object_round3_export.py check-stats` independently recomputes means,
sample SDs, z, tie-aware exact U p-values, verdicts and Holm from the recorded run
vectors: **928 comparison rows, 204 scored families, zero differences**. Evidence:
`out/objr3sd/skeptic_raw.json` and `skeptic_stats.json`; the latter also verifies
the stamp precedes the scheduler submission.

The inherited same-device reducer and predeclared dictionary were retained. Changes
add seed offsets, explicit cluster routing and a raw checker; their source hashes
therefore differ from the submitted version. Both stamped and analysis hashes remain
on file. The H200 Result was not itself re-run under the final script version (it
records analysis-source sha256 d13b6b60, the file is now bc9c3af9); re-running
`analyse` on both fetched batches with the current script reproduces the shipped
Results byte for byte, which is what `docs/INTERP.md` 10.4 item 11 obliges and is
stronger evidence than the function-level source review (verified in the independent
skeptic pass appended below). The fresh-seed replication below tests the substantive
claims independently.

## 8. Fresh-seed B200 replication

`r3sdcheck-00126d` completed **5 jobs, 0 failed**, on <cluster-node>. The final transfer
receipts validate SHA-256 for all 180 metadata/console files and 90 slim NPZs;
one partial metadata file was replaced after the interrupted SCP transfer.
`out/objr3sd_house/verify.json` verifies **90 paired runs, 90 consoles, all B200**,
with no missing expected recordings or problems. `samedevice.json` in that
directory is the fresh Result; its stamp predates the scheduler submission.

The exact declared answer is **rectify PARTIAL, suppress REPRODUCE, LC11 follows
in no arm**. T3 passes both questions at all three small rungs. T2 passes vs-base
at all three, and vs-null at 8.8/11; at 4.5 its maximum is .024922 +/- .000927
against null .010775 +/- .005302 (five runs each), z = **2.66825**, Holm p =
.023810. The effect gate is 3 null SDs, so that member is `null` despite its
adjusted p.
The five rectify runs (.025383 .024782 .023364 .025453 .025629) all exceed all five of its blank/blank runs
(.007654 .019080 .012175 .005235 .009728): U = 25/25 and p is at the exact 5 v 5 floor .0079365, Welch 5.88.
The z gate is missed because one null run (.019080) roughly quadruples the null SD to .005302 against the H200
null's .001147. The verdict is the declared rule applied honestly; the separation is complete.
The full two-question T2/T3 reproduction claim therefore fails here.
Note also that on this batch the reference arm itself reads `result` vs its own
blank/blank null at T2 8.8 and 11 deg, where it reads `null` on H200 (section 4).

B200 maxima and companions, mean +/- sample run SD, n=5:

| Type | Diameter | Base maximum | Rectify maximum | Rectify/base | Base companion | Rectify companion |
|---|---:|---:|---:|---:|---:|---:|
| T3 | 4.5 | .003415 +/- .000618 | .020939 +/- .000275 | 6.131 | .005821 +/- .000800 | .007101 +/- .001351 |
| T3 | 8.8 | .004042 +/- .000784 | .036491 +/- .000764 | 9.028 | .005358 +/- .000481 | .006078 +/- .000747 |
| T3 | 11 | .003580 +/- .000542 | .045106 +/- .001193 | 12.598 | .004585 +/- .000443 | .005660 +/- .000473 |
| T2 | 4.5 | .007076 +/- .001030 | .024922 +/- .000927 | 3.522 | .009656 +/- .002072 | .009355 +/- .001216 |
| T2 | 8.8 | .012011 +/- .002143 | .041091 +/- .001729 | 3.421 | .011537 +/- .001427 | .011363 +/- .001597 |
| T2 | 11 | .015113 +/- .001794 | .048738 +/- .001715 | 3.225 | .011519 +/- .001822 | .011636 +/- .002355 |

All six vs-base maximum tests have Holm p = .023810. The small-rung companion
verdicts remain null. T3's 11-degree companion has Holm p = .047619 but z =
2.42722, again below the declared effect gate; T2 companion Holm p = 1 throughout.
Thus "null" here is the full declared verdict, not shorthand for p > .05.

The large-rung companions are not all null here either — the same three rows
(rectify T3 at 20/30, T2 at 30) exceed base with Holm p = .039683, as they do on H200
and, for T3 at 30 deg, in round 2. For T3 the medians are .006247 +/- .000484
and .007283 +/- .000510 versus base .004292 +/- .000387 and .004288 +/- .000103;
T2 at 30 is .014109 +/- .001640 versus .010885 +/- .000700. These are secondary
large-rung findings, not evidence of a small-field preference.

The fresh independent checks reproduce **1,860 raw-derived values and 928
comparisons / 204 scored families with zero differences**. They confirm the
partial verdict rather than forcing agreement with the original H200 draw.
The source review finds the predeclared dictionary and statistical reducers
unchanged, and `brain.py`/`optic.py` byte-identical to the handoff state. The
three batches remain separate; these results do not isolate a causal GPU effect.

## Report

```yaml
summary: >-
  Closed the arm/device confound within both same-box batches. The H200 batch
  fully reproduces the declared rectify carrier result; fresh B200 replication
  is partial because T2 at 4.5 degrees misses the vs-null effect gate. Suppress's
  primary null pattern and lack of an LC11 rescue reproduce.
key_claims:
  - Each batch has 90 paired runs, five per object rung/null arm, on one GPU model in one submission.
  - Rectify exceeds base at every small T3/T2 maximum on H200 and B200.
  - B200 T2 at 4.5 degrees has vs-null z 2.66825 and Holm p .023810, so the full declared reproduction is partial.
  - Small-rung windowed-median companion verdicts remain null; the same three large-rung companion effects (rectify T3 at 20/30 degrees, T2 at 30 degrees) are present in BOTH round-3 batches, and the T3 30-degree row also in round 2.
  - The reference arm is not identical across batches - base T2 versus its own null is null at every small rung on H200 but result at 8.8 and 11 degrees on B200.
  - No default, specificity improvement or behavioral rescue is adopted or established by this sphere assay.
files_written:
  - scripts/object_round3_samedevice.py
  - scripts/object_round3_export.py
  - docs/audits/object_samedevice_r3.md
  - out/objr3sd/
  - out/objr3sd_house/
  - out/r3sdcheck_cluster.log
  - out/export/objr3_samedevice_index.json and its listed directories
  - out/round3_astra/source_review.json
api:
  - plan and run-job --seed-offset defaults to 0; replication uses 1000.
  - submit --target and --node default to None; replication routes explicitly to house/<cluster-node>.
  - check-raw defaults to out/objr3sd, its samedevice.json and skeptic_raw.json; each path is overridable.
  - Exporter check-stats requires --result and --json; optional --out enables stamp/scheduler verification.
validation: >-
  Original H200 and fresh B200 batches each completed five jobs with zero failed
  and verified all 90 recordings. Each independent pass matched 1860 raw-derived
  values and 928 comparison rows in 204 scored families. At 4.5 degrees, B200 T3
  maximum is .020939 +/- .000275 under rectify versus .003415 +/- .000618 under
  base; companions are .007101 +/- .001351 versus .005821 +/- .000800, n=5 each.
  H200 full reproduction becomes partial on B200 under the unchanged effect gate.
recommendations:
  - Carry the exact partial B200 reproduction verdict and companion statistics into owner documentation.
  - Retain rectify/suppress as opt-in hand-set control arms.
  - Before any adoption, require same-device specificity/ON-OFF evidence, full benchmark draws and relevant room/anti-runaway guards.
open_questions:
  - How stable is the T2 4.5-degree vs-null effect with more independently declared runs?
  - Can a data-supported mechanism carry a small-field response to LC11 without the existing specificity costs?
author_self_review:
  refuted:
    - Full two-question T2/T3 reproduction in every fresh batch.
    - The stronger claim that all typical-cell medians are unchanged at every object size.
  confirmed:
    - All within-batch arm-vs-base rows share a GPU model and the source predeclarations precede submission.
    - Strong small-rung rectify maxima versus base, accompanied by the actual per-body medians.
    - Suppress's primary null pattern and lack of an LC11 rescue at fresh seeds.
  corrections:
    - Reported the B200 T2 effect-gate miss and large-rung median effects without changing the declared tests.
    - Recovered dead-client status from scheduler job IDs and repaired partial fetches by hash.
    - Distinguished inherited model edits from this task using handoff source hashes; this task changes no model defaults.
  verdict: mostly sound
skeptic:
  source: "independent Opus pass, 2026-09-14"
  verdict: "mostly sound"
  refuted:
    - 'R1: the quoted companion Holm p "T2 p = 1 at all three rungs"; the values are 1, .66667, 1 (all three still null).'
    - 'R2: "the original H200 batch and the new B200 batch have distinct source fingerprints" - the two round-3 batches have IDENTICAL fingerprints; only round 2 differs in code.'
    - 'R3: presenting the large-rung companion effect as a B200 observation - the same three rows read result on H200, and the T3 30-degree row reproduces round 2.'
    - 'R4: the reference arm was unreported - base T2 versus its own null reads result at 8.8 and 11 degrees on B200 and null at every small rung on H200.'
    - 'R5: the Report refuted/confirmed/corrections/verdict fields were self-authored rather than an independent pass; accurate, but now filed under author_self_review.'
  corrections:
    - 'C1: corrected the T2 companion Holm p in section 4.'
    - 'C2: rewrote the section 5 cross-batch fingerprint claim - the round-3 batches share code, and no round-3 base-versus-base row was computed.'
    - 'C3: added the H200 large-rung companion rows to section 4 and removed the B200-only framing in section 8.'
    - 'C4: added the reference arm''s own versus-null rows at the head of section 4, with a pointer in section 8.'
    - 'C5: added the shape of the B200 T2 4.5-degree gate miss - complete separation, U 25/25 at the exact floor, null SD inflated by one run.'
    - 'C6: stated that companion p_holm are adjusted over five rungs (large_rungs over two), not the primaries'' three.'
    - 'C7: stated that re-running analyse on both fetched batches with the current script reproduces the shipped Results byte for byte.'
    - 'C8: resolved the "raw arrays remain remote" wording against FETCHED.txt, and printed the LC11 suppress SD at 5 dp.'
```

## Skeptic pass (independent, 2026-09-14)

### Refuted

#### R1. "T2 p = 1 at all three rungs" (audit §4)

> "Every rectify-vs-base carrier maximum above has Holm p = .02381. Its companion comparisons are all null: T3 Holm
> p = 1, .30159, .09524; **T2 p = 1 at all three rungs**."

Recomputed. H200 rectify `abs_drive_median` vs base (role `upstream_windowed`), Holm within the 5-rung family:

| type | 4.5 | 8.8 | 11 |
|---|---|---|---|
| T3 | 1.00000 | 0.30159 | 0.09524 | (audit correct)
| T2 | 1.00000 | **0.66667** | 1.00000 | (audit says 1 at all three)

Sources: `out/objr3sd/samedevice.json` `tables.sphere_comparisons` (role `upstream_windowed`, arm `rectify`,
against `base`, type `T2`), and independently my own windowed medians over `out/objr3sd/sph/*.npz` +
`tables.windows` (z −0.06031 / +0.50135 / +0.40100; p 0.690476 / 0.222222 / 0.841270). All three verdicts are
`null`, so the *claim* ("all null") survives; the *number* does not.

#### R2. "the original H200 batch and the new B200 batch have distinct source fingerprints" (audit §5)

> "Round 2, the original H200 batch and the new B200 batch have **distinct source fingerprints** as well as
> different execution histories. They cannot identify a causal GPU-model effect..."

Recomputed from `provenance.source_fingerprint` of `out/objr3sd/sph/base_d045_obj_s0.json` and
`out/objr3sd_house/sph/base_d045_obj_s1000.json`: the two round-3 batches have **identical** fingerprints — all 44
entries of `files` / `files_lf` match and all 29 entries of `files_loaded` match; the set of differing files is
**empty**. (Round 2, `out/objr2c/sph/base_d045_obj_s0.json`, does differ: 18 of 44 files, including `optic.py`,
`brain.py`, `interp/common.py`.) The H200 and B200 batches therefore differ in GPU model, box, brain/env seeds
(0-4 vs 1000-1004) and execution history — but **not in code**. The audit's conclusion (no causal GPU attribution)
still stands on seeds + box alone; the stated *reason* is false, and it understates what this pair of batches can
support. Related gap: neither Result contains a statistical row comparing H200 `base` with B200 `base` — the only
pair in the whole round with identical code, identical protocol and a different GPU model. `cross_batch_base` in
both Results compares against round 2 only.

#### R3. The large-rung companion effect is presented as a B200 observation; it is a three-batch effect, and the H200 numbers are never given

> §8: "The large-rung companions are not all null: **B200** rectify T3 at 20/30 and T2 at 30 exceed base with Holm
> p = .039683."

True of B200, but the identical three rows are `result` in the **H200** batch as well, and §4/§5 report none of them:

| batch | row | arm median-of-medians | base | z | Holm p | verdict |
|---|---|---|---|---|---|---|
| H200 | T3 20 deg | .006050 +/- .000709 | .004445 +/- .000414 | +3.88098 | .039683 | result |
| H200 | T3 30 deg | .007377 +/- .000775 | .004259 +/- .000400 | +7.80038 | .039683 | result |
| H200 | T2 30 deg | .012665 +/- .001896 | .009673 +/- .000841 | +3.55642 | .039683 | result |

(`out/objr3sd/samedevice.json` `sphere_comparisons`, role `upstream_windowed`, against `base`; reproduced by my own
numpy over the npz.) Both Results record `reproduction.rectify.companion_all_null_all_rungs = false`. Round 2 had
it too: `out/interp/objr2c/compare.json` rectify T3 at 30 deg, z +9.795, Holm .039683, ratio 1.827, verdict
`result`. So this is a **reproduced** round-2 large-rung companion effect, not a fresh B200 finding — and §4's
"This is evidence for an extremal carrier change, without an established typical-cell improvement" reads as a
general statement when it is only true of the three small rungs.

#### R4. Unreported: the reference arm itself crosses the effect gate on B200 and not on H200

Nowhere in the audit are the `base` arm's own vs-null rows reported. Recomputed (both Results,
`sphere_comparisons`, role `T2_carrier`, arm `base`, against `null`; confirmed independently from the run JSONs):

| batch | base T2 8.8 | base T2 11 |
|---|---|---|
| H200 | z +1.02481, Holm .301587, **null** | z +1.95986, Holm .023810, **null** |
| B200 | z +6.12652, Holm .023810, **result** | z +9.29988, Holm .023810, **result** |

On the fresh B200 batch the *shipped* model's T2 maximum is already distinguishable from its own blank/blank null
at two of the three small rungs; on H200 it is not. For a batch whose entire purpose is "does the round-2 picture
reproduce on one GPU model", a reference arm that behaves differently across the two same-device batches is
material and is absent from the audit, from the `## Report` `key_claims`, and from the handoff reply.

#### R5. The `## Report`'s `refuted` / `confirmed` / `corrections` / `verdict` are the author's own, not a skeptic's

§7 is honest about this ("a separately implemented computational check by Astra, not a second human or agent
review"), but the Report fills the four fields that handoff §5 reserves for an independent pass with self-review.
The fields as written are *accurate* (I confirmed each one), but they are not what the workflow asked for. This
document is the actual independent pass.

---

### Confirmed

**Every headline statistic.** Recomputed twice (my own numpy over the per-run JSONs / per-body arrays, and the
shipped script re-run). Audit §4 (H200) and §8 (B200) tables match to every printed digit:

* H200 maxima (`diff_signed_best_cell`, mean +/- sample SD, n=5): T3 base .003425/.003881/.003418, rectify
  .021512/.036979/.044644, ratios 6.281/9.528/13.063; T2 base .009808/.012152/.015418, rectify
  .023106/.040946/.047300, ratios 2.356/3.369/3.068. All six rectify-vs-base rows: z +18.02/+30.72/+58.48 (T3),
  +4.19/+24.06/+41.31 (T2), exact-U p .0079365 (the floor at 5 v 5), **Holm p .023810** — the audit's ".02381".
* H200 companions (per-body RF-windowed median |drive|): base .006865/.005286/.004503 (T3),
  .010391/.011776/.011006 (T2); rectify .006705/.006377/.005747 (T3), .010339/.013045/.011603 (T2). Every
  small-rung vs-base verdict `null`.
* H200 LC11 windowed `drive_median` table (all nine cells): base −.036680/.034847/−.046689, rectify
  −.003597/.054873/.030485, suppress .034451/.038534/.000527, with the quoted SDs. All small-rung P2 rows `null`.
* Suppress companion at T3/4.5: z **−3.07536**, unadjusted `result`, **Holm p .126984**, exactly as stated.
* B200 maxima and companions: all twelve table cells match, including rectify T3 .020939 +/- .000275 at 4.5 and
  T2 .024922 +/- .000927; ratios 6.131/9.028/12.598 (T3), 3.522/3.421/3.225 (T2). All six vs-base Holm p .023810.
* B200 T3 11-deg companion Holm p **.047619** with z **+2.42722**; T2 companion Holm p 1.0 at all three small
  rungs. Both as stated.

**The PARTIAL call is exactly the declared rule, and the H200 "full reproduction" survives that same rule.**
`common.compare` calls `result` only on |z| >= `Z_RESULT` = 3.0 **and** p <= 0.05
(`flyverse/interp/common.py:703-757`). B200 rectify T2 at 4.5: stim .024922 +/- .000927 vs its own null
.010775 +/- .005302, z **+2.66825** (< 3), p .0079365, Holm **.023810** -> verdict `null`, so `both_result` fails
there. `reproduction()` (`scripts/object_round3_samedevice.py:452-455`) gives `REPRODUCES` only if T3 *and* T2 pass
at all three small rungs, else `PARTIAL` if any T3 rung passes — matching the stamped text verbatim. My hand-rolled
rule gives B200 T3 `[True, True, True]`, T2 `[False, True, True]` -> **PARTIAL**; H200 T3 `[True, True, True]`,
T2 `[True, True, True]` -> **REPRODUCES**. Suppress: `both_null` True x3 on T3, T2 and LC11 in both batches ->
REPRODUCE. `lc11_follows` False in every arm of both batches.

**Predeclaration stamps precede submission, in both batches.**
`out/objr3sd/predeclared.json` `stamped_utc` **2026-09-14T05:28:42Z**; `out/objr3sd/scheduler_receipt.json` first
`submitted_at` **2026-09-14T05:29:08.552021Z** (and `out/objr3sd_cluster.log` header "submitted 05:28:52Z").
`out/objr3sd_house/predeclared.json` **2026-09-14T19:03:44Z**; house receipt first submission
**2026-09-14T19:05:02.141174Z**. The two `predeclared` rule dicts are byte-equal to each other, and both Results
record `predeclared_live_equals_stamp: true`.

**One box per family, reference and treatments together — stronger than the audit claims.** Read off every run
JSON independently: `out/objr3sd` 90/90 `NVIDIA H200`, all on host `e9a16d3dec07` (30 per arm); `out/objr3sd_house`
90/90 `NVIDIA B200`, all on host `<cluster-node>` (30 per arm). Both Results: `same_device_rows 420/420`,
`all_vs_base_rows_same_device true`, `one_model true`, `closed true`, `verify.problems []`. The cluster logs show a
single `cluster_run.py --arm-block fam --no-balance-blocks` call each, one block `samedev` placed on one target
(`-> @r3-h200b` / `-> @house` with `--target house --node <cluster-node>`), 5 jobs, `0 failed`, and jobs blocked by *seed*
(each job runs base/rectify/suppress in turn at every rung), so the arm contrast is balanced inside each job — which
also neutralises the fact that H200 seeds 3 and 4 restarted ~10 h later after the box outage.

**Arms are the round-2 arms.** Every rectify run JSON carries exactly
`stream_rectify = [["^(Mi1|Tm3|Tm2)$","^T3$","pos"],["^(Tm1|Tm4)$","^T3$","neg"],["^(Tm2|L5|Tm3|Mi1)$","^T2$","pos"],["^C3$","^T2$","neg"]]`
and no `spatial_suppress`; every suppress run carries `[["^(Mi1|Tm1|Tm2|Tm3|Tm4)$", 0.5, 10.0]]` and no rectify;
every base run carries neither — byte-identical to the stamped `ARMS` and to `object_compare_r2.md` §1.
`verify.json` `hook_liveness` shows the hook actually fired in 30/30 rectify and 30/30 suppress runs, 0/30 base.

**Max-over-cells statistics travel with their per-body median.** Both quoted tables carry the companion columns
beside the maxima, §4 and §8 state the definitions, and the `## Report` `validation` field pairs the B200 T3 4.5
maximum with its companions. `sphere_comparisons` carries a `upstream_windowed` row for every
`P1_carrier`/`T2_carrier` row. No orphan maximum found.

**Holm families and m match the declaration.** The primary `P1_carrier` / `T2_carrier` / `P2_lc11` families are the
three small rungs, adjusted separately per question (m=3) — my own Holm over the three raw p reproduced every
`p_holm`. The companion `upstream_windowed` family runs over all five rungs (m=5) and the `large_rungs` role over
two (m=2), consistent with the stamp's "Holm within each arm x type x statistic set"; my Holm over those family
sizes reproduced every reported companion `p_holm` too. (Minor inaccuracy, see Corrections: the Result's `reading`
string describes *all* p_holm as "within the round-2 family (3 small rungs …)", which is only true of the primaries.)

**Round-2 numbers quoted beside round-3 ones are the actual round-2 numbers.** From
`out/interp/objr2c/compare.json` (run_id `objr2c-compare`, 2026-09-14T02:09:44Z), rectify vs base:
T3 ratios 6.0150 / 10.4938 / 8.9877 -> audit "6.015 / 10.494 / 8.988" and stamp "6.01 / 10.49 / 8.99"; T2
3.7450 / 3.2678 / 3.1308 -> "3.745 / 3.268 / 3.131" and "3.75 / 3.27 / 3.13". The stamp's companion claim
("round 2: null in all 48 arm x type x rung rows, ratios 0.73-1.27") is exactly the 8 arms x {T3,T2} x 3 small
rungs slice: 48 rows, 48 `null`, ratio range **0.7297-1.2729**. Round-2 devices (`base` B200, `rectify`/`suppress`
H200) are as the audit says.

**`git diff -- flyverse/optic.py flyverse/brain.py`.** `optic.py`: **empty**. `brain.py`: 30 insertions / 1
deletion, all of it the **unitary thread's** opt-in field and nothing from this task — `LIFParams.w_syn_by_nt:
dict | None = None` (per-transmitter unitary factor, docs/audits/unitary_strength.md), the `_nt_factor()` helper,
one `if p.w_syn_by_nt:` branch in `_shaped_weights` before the connection cap, and the field appended to the
fan-in normalisation cache key. Default `None` executes nothing. All 180 run JSONs record
`lif.w_syn_by_nt = null` and `lif.w_syn = 0.275`. Raw sha256 of both files equals the handoff snapshot in
`out/round3_astra/start/state.json` (`brain.py` 7fadb8d6bb1a…, `optic.py` afb78cfe0365…) and equals what both
batches shipped, so §7's "brain.py/optic.py byte-identical to the handoff state" holds.

**Analysis-code drift is real but harmless — verified, not taken on trust.** `analysis_code_drift` in both Results
is `["scripts/object_round3_samedevice.py"]` (LF-normalised sha256 426742da -> d13b6b60 -> bc9c3af9; the other four
sources, `object_round2_compare.py`, `object_round2_baseline.py`, `probe_object_matched.py`,
`flyverse/interp/common.py`, are unchanged from plan to analysis to now). `out/round3_astra/source_review.json`
attributes the drift to `cmd_check_raw` (new) and `_job_line` / `_plan_args` / `build_jobs` / `build_parser` /
`cmd_submit` (changed). I did not take that on trust: re-running the *current* script's `analyse` over both fetched
batches reproduced both shipped Results exactly, so the stamp's "a change to the reducers obliges a re-run"
obligation is discharged.

**The author's own computational checks are real.** `out/objr3sd/skeptic_raw.json` and
`out/objr3sd_house/skeptic_raw.json`: `values_checked` 1860, `differences` 0 each. `skeptic_stats.json` (both):
`rows_checked` 928, `families_checked` 204, `differences` 0, `predeclaration_before_submission` true, with the
correct `stamped_utc` / `first_submission_utc` pairs. Both pin `result_sha256`, and both sha256 match the shipped
`samedevice.json` bytes today (69d26323…, 9ff05e67…). My own raw check (item 2 above) agrees.

**Batch mechanics and the slim.** 90 JSON + 90 npz + 90 txt in each `sph/`; `meta/` 180 files; `verify.json`
`expected_missing []` in both. Slim: `slim_n_columns_before` 12235 -> `slim_n_columns_after` 4840, kept
`T3, T2, Tm5Y, TmY21` (npz counts 1940 + 1630 + 898 + 372 = 4840, exactly the audit's arithmetic), dropped
`Mi1, Mi4, Tm3, TmY13, TmY5a` — none of which `object_round2_compare.UPSTREAM`/`LC_TYPES` reads, so the companion
statistic is recoverable (I recomputed it from the slim files). Cluster logs show 114 -> ~50 MB per run.
Fetch receipts: `out/round3_astra/fetch_sd_meta_house_hashes.txt` "180 remote files; 1 missing or hash-mismatched
/ hash verification: 180 files, 0 problems" and `fetch_sd_slim_house_hashes.txt` "90 files, 0 problems" — exactly
§8's "180 metadata/console files and 90 slim NPZs … one partial metadata file was replaced".

**The CPU smoke (§2) and the tree state (§2).** `out/objr3sd/smoke/samedevice_smoke.json`: 264 side-by-side rows,
264 `verdict_agrees`, max |z_r3 − z_r2| = 0.0, max |p_holm_r3 − p_holm_r2| = 0.0; `closed false`, `one_model false`,
`same_device_rows 140/420`; all 88 cross-batch rows z = 0.0, p = 1.0. `out/objr3sd/tree_state.json`: commit
`6ec2de1a8c52…`, `diff_stat` "9 files changed, 499 insertions(+), 27 deletions(-)"; round 2's
`out/objr2c/tree_state.json` commit `653179b4b0e2…`; `probe_object_matched.py` (adb77d08…) and `retina.py`
(a8e60aac…) identical between round 2 and round 3, as §2 says.

**`## Report` field-by-field.** `summary`, all five `key_claims`, the `api` lines (verified against
`build_parser()`: `plan`/`run-job` `--seed-offset` default 0; `submit --target`/`--node` default None; `check-raw`
defaults `out/objr3sd`, `out/objr3sd/samedevice.json`, `out/objr3sd/skeptic_raw.json`; `object_round3_export.py
check-stats` has `--result` and `--json` required, `--out` optional), the `validation` numbers, and all
`files_written` paths (`out/export/objr3_samedevice_index.json` exists and pins `baseline_sha256` = the shipped
Result; `out/round3_astra/source_review.json` exists) check out. Nothing adopted; the classification of the rectify
modes and suppression k/radius as hand-set numbers is correct and is stated.

---

### Corrections

1. **§4, replace** "T2 p = 1 at all three rungs" **with**:
   "T2 Holm p = 1, .66667, 1 — all three `null`."

2. **§5, replace** "Round 2, the original H200 batch and the new B200 batch have distinct source fingerprints as
   well as different execution histories." **with**:
   "Round 2 ran on a different tree (18 of 44 fingerprinted files differ, `optic.py`, `brain.py` and
   `interp/common.py` among them). The H200 and B200 round-3 batches, by contrast, have **identical** source
   fingerprints: they differ only in GPU model, box, brain/env seeds (0-4 vs 1000-1004) and execution history.
   That still does not isolate a causal GPU-model effect — seeds and box are confounded with model — but the code
   is not a confound between them, and no comparison row between the two round-3 base arms was computed."

3. **§4, after the companion sentence, add** (and drop the B200-only framing from §8):
   "The large rungs are not null in this batch either: rectify's T3 companion exceeds base at 20 deg
   (.006050 +/- .000709 vs .004445 +/- .000414, z +3.88, Holm .039683) and 30 deg (.007377 +/- .000775 vs
   .004259 +/- .000400, z +7.80, Holm .039683), and its T2 companion at 30 deg (.012665 +/- .001896 vs
   .009673 +/- .000841, z +3.56, Holm .039683). Round 2 already carried the T3-at-30 row (z +9.80, ratio 1.827,
   `result`), so this is a three-batch effect, not a device-specific one, and `companion_all_null_all_rungs` is
   false in both round-3 Results. The typical-cell claim holds only at the small rungs."
   **§8, replace** "The large-rung companions are not all null: B200 rectify T3 at 20/30 and T2 at 30 exceed base
   with Holm p = .039683." **with** "The large-rung companions are not all null here either — the same three rows
   (rectify T3 at 20/30, T2 at 30) exceed base with Holm p = .039683, as they do on H200 and, for T3 at 30 deg,
   in round 2."

4. **§4 and §8, add the reference arm's own vs-null rows** (new short paragraph, e.g. at the head of §4):
   "The reference arm is not identical across the two batches. On H200, `base` T2 vs its own blank/blank null is
   `null` at every small rung (z +0.35 / +1.02 / +1.96). On B200 it reads `result` at 8.8 deg (z +6.12652, Holm
   .023810) and 11 deg (z +9.29988, Holm .023810). T3 base is `null` on both. Every within-batch comparison is
   still same-device, but the shipped model's own T2 maximum separates from blank on one GPU model and not the
   other, which is itself part of what this batch reproduced."

5. **§8, add the shape of the T2/4.5 gate miss** after "…so that member is `null` despite its adjusted p":
   "The five rectify runs (.025383 .024782 .023364 .025453 .025629) all exceed all five of its blank/blank runs
   (.007654 .019080 .012175 .005235 .009728): U = 25/25 and p is at the exact 5 v 5 floor .0079365, Welch 5.88.
   The z gate is missed because one null run (.019080) roughly quadruples the null SD to .005302 against the H200
   null's .001147. The verdict is the declared rule applied honestly; the separation is complete."

6. **Result `reading` string** (`summary.reading`, written by `cmd_analyse`) says every `p_holm` is "Holm within the
   round-2 family (3 small rungs per arm x type x question)". True of `P1_carrier` / `T2_carrier` / `P2_lc11`;
   the `upstream_windowed` companion rows are adjusted over five rungs and the `large_rungs` rows over two. Worth
   saying, because §4 prints an m=3 primary Holm p (.02381) next to m=5 companion Holm p (1, .30159, .09524).

7. **§7, add one line**: the reducers were *not* re-run after the H200 Result was written under the final script
   version (the Result records d13b6b60, the file is now bc9c3af9). Re-running `analyse` on the fetched batch with
   the current script reproduces the shipped Result byte-for-byte (both batches), which is what the stamp's
   `docs/INTERP.md` 10.4 item 11 clause requires; say so rather than relying on the function-level source review.

8. **Minor**: §3 "The raw full optic arrays remain remote" sits oddly against `out/objr3sd/FETCHED.txt`
   ("the box is destroyed after this marker") and against the reply's "No box was destroyed." Pick one; no number
   depends on it. Also §4's LC11 table prints ".056997" (6 dp) where every other cell is 5 dp.

---

### Verdict

**mostly sound**

Every statistic the audit quotes is reproducible — I reproduced all of them twice, once with the shipped script and
once with an independent numpy/scipy implementation that never imports the project's reducers, plus a raw-array
check of the max-over-cells primary. The design holds: predeclaration before submission in both batches, one
submission and one physical host per batch with the reference arm in the same block as its treatments, byte-correct
arms, correct Holm families, the max-over-cells primary always travelling with its per-body median, round-2 numbers
quoted correctly, `optic.py` untouched and `brain.py` carrying only the unitary thread's opt-in `w_syn_by_nt`
(default `None`, no-op, recorded `null` in all 180 runs). The PARTIAL call on B200 is exactly what the stamped rule
produces, and the H200 batch genuinely survives that same rule — the audit did not soften the gate to rescue either
batch. The defects are one wrong quoted Holm p (R1, conclusion unaffected), one false provenance claim that happens
to understate the audit's own evidence (R2), a large-rung companion effect reported only for B200 when it is
present in all three batches (R3), an unreported cross-batch difference in the reference arm itself (R4), and the
Report's skeptic fields being self-authored (R5). None of these overturns the answer — rectify's carrier figure is
real, large and same-device on two GPU models; suppress's nulls and the absence of an LC11 rescue reproduce;
nothing is adopted — but R3 and R4 are omissions that a reader of the `## Report` alone would not recover, so this
is not `sound`.
