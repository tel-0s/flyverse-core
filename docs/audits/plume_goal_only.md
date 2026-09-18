# The goal-only plume arm (round 8, item 4; batch `plume-go`)

Question (TODO.md section B, follow-up 1 of the Session-13 skeptics): [plume_steering.md](plume_steering.md)'s
correction bundled a bilateral walking goal with a DNa02 L-R integral feedback onto PFL3, and the skeptic inferred
that the feedback is the load-bearing part (mean |demand| did not rise between the diagnostic and the validation
while mean |yaw| tripled). No arm separated the two. This batch does: `full` (as shipped), `goal-only` (the walking
goal with the integral gain at 0, i.e. the diagnostic's one-way `clip(80 * turn)` PFL3 bridge) and `feedback-only`
(the feedback with the pre-correction upwind / entry-memory goal law), on the shipped six rooms and on six different
starts and headings drawn from a seed. Runs are the replicate unit. No admission claim; nothing adopted; no default
changed; the `plume` instrument's shipped law is untouched (the two variants are opt-in keywords recorded in
`describe()`).

## 1. The variants (code)

`flyverse/navigation.py`: `PlumeNavigation(c, *, feedback_gain_per_s=None, walking_goal=True)`. `feedback_gain_per_s`
None keeps the shipped 5.0 /s; 0 keeps the integral at zero, so the PFL3 input is `clip(80 * turn, -80, 80)` exactly as
in the diagnostic. `walking_goal` False skips the bilateral local goal (the bilateral state is still tracked and
checkpointed) so the goal is the pre-correction law. `describe()['parameters']` carries
`steering_integral_gain_per_s`, `walking_goal_enabled` and `variant` (`full` / `goal-only` / `feedback-only`).
`flyverse/instruments.py` accepts `plume:feedback=0` and `plume:walking_goal=0`; bare `plume` is byte-for-byte the
shipped law (`tests/test_navigation_instruments.py::test_plume_variants_are_opt_in_and_recorded`).

## 2. Design (frozen: `out/plume-go/predeclared.json`, stamped 2026-09-18T00:14:43Z; submitted as runs
`plume-go-c574f6` (shipped starts) and `plume-go-1c49fb` (drawn starts) at ~00:16Z)

| factor | levels |
|---|---|
| arm | full (`compass plume hunger flight`), goal-only (`plume:feedback=0`), feedback-only (`plume:walking_goal=0`) |
| starts | shipped: default start (-0.5, 0.05), headings 5 / 90 / -90 / 185 / 5 / 5 deg (the plume_steering.md rooms); drawn: six (x, y, heading) from `numpy.random.default_rng(8)`, x in [-0.55, 0.55], y in [-0.35, 0.35], heading in [-180, 180) deg: (-0.190, -0.043, -113.5), (0.536, -0.089, -110.2), (-0.199, -0.275, 113.0), (0.317, -0.015, -27.7), (0.407, -0.181, -87.9), (-0.120, -0.170, 32.7) |
| runs | six per (arm, start set), brain seeds 31-36 (r0 with the shipped starts is the plume_steering.md configuration); one B=6 room per job, env seeds 0-5, all fruit, no fence, energy 0.1, 60 s, the shipped native path |

36 jobs in two `cluster_run.py` calls of 18, every job pinned to one GPU of the pool 4-7 of the second node.
Carried forward from [plume_steering.md](plume_steering.md) (skeptic item 5): the six shipped rooms are
near-repeats -- `world.make_room` places apple, orange, banana and lime at seed-independent coordinates, only the
grapes and blueberries jitter, the start is identical, and three rows share the 5 deg heading.
Measures per run: `fed_rows` (rows of six feeding >= 1 s, the plume_steering.md criterion), `mean_first_contact_s`
(over the fed rows; descriptive), `mean_abs_dna_lr_hz` (|DNa02 L - R| over rows and 0.1 s samples), `mean_path_m`,
and the instrument's own |turn|, |PFL3 input|, |integral| and the body's |yaw|. One Holm family, m = 6 (6 v 6 floor
0.0021645 x 6 = 0.013):

| test | key | contrast | starts | predicted |
|---|---|---|---|---|
| 1 | fed_rows | full - goal-only | shipped | positive |
| 2 | fed_rows | full - feedback-only | shipped | none |
| 3 | mean_abs_dna_lr_hz | full - goal-only | shipped | positive |
| 4 | fed_rows | full - goal-only | drawn | positive |
| 5 | fed_rows | full - feedback-only | drawn | none |
| 6 | mean_abs_dna_lr_hz | full - goal-only | drawn | positive |

Verdicts from `common.compare` (result / null / underpowered / undetermined); [determinism_gate.md](determinism_gate.md):
no room repeats, so every summary number below is a mean +/- SD over runs; the per-room-row values in section 3.3
and the single-draw times quoted in section 4 are one draw each, pasted from `rows.csv` under INTERP 10.4 rule 28
and labelled as such.

**Withdrawn:** "every number below is a mean +/- SD over runs and nothing is quoted beyond that" -- section 3.3
quotes 36 x 6 single-draw values and section 4 items 1, 2 and 4 quote single draws.

## 3. Results (`out/plume-go/analysis/`; 36 runs, 0 problems)

Every run: `NVIDIA B200`, `device cuda` in 36/36 consoles, no traceback, backend native (`cuda_kernels`,
`event_driven`, `cuda_sparse torch`), preset instrumented with the four instruments, the plume record's `variant` equal
to the arm's (gain 5.0 / 0.0 / 5.0; walking goal True / True / False), brain seed 31 + r, the six starts equal to the
frozen list, every loaded source at the predeclared hash; GPUs 4 / 5 / 6 / 7 took 10 / 10 / 8 / 8 jobs; wall
110.7-538.8 s. Scheduler: 36 completed, 0 failed.

### 3.1 The family (`compare.csv`)

| test | key | stim | null | starts | predicted | n | verdict | verdict (Holm, sign) | diff | z (ref SD) | Welch t | p | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fed_rows | full | goal-only | shipped | positive | 6 v 6 | null | null | +1.3333 | +2.58 | +6.32 | 0.0022 | 0.0130 |
| 2 | fed_rows | full | feedback-only | shipped | none | 6 v 6 | result | **result** | +5.8333 | +14.29 | +35.00 | 0.0022 | 0.0130 |
| 3 | mean_abs_dna_lr_hz | full | goal-only | shipped | positive | 6 v 6 | result | **result** | +1.1870 | +12.26 | +13.35 | 0.0022 | 0.0130 |
| 4 | fed_rows | full | goal-only | drawn | positive | 6 v 6 | null | null | +1.6667 | +2.04 | +5.00 | 0.0022 | 0.0130 |
| 5 | fed_rows | full | feedback-only | drawn | none | 6 v 6 | undetermined | undetermined | +2.0000 | NA | NA | 0.0022 | 0.0130 |
| 6 | mean_abs_dna_lr_hz | full | goal-only | drawn | positive | 6 v 6 | null | null | -0.0905 | -0.39 | -0.90 | 0.5887 | 0.5887 |

Tests 1 and 4 are `null` under the declared rule because `compare` divides by the reference arm's SD (goal-only's
0.52 / 0.82 rows) and needs |z| >= 3; the samples separate completely (full 6,6,6,6,6,6 against 5,4,4,5,5,5 and
5,5,4,4,5,3; Welch t 6.3 / 5.0; exact p 0.0022). The verdict stands as declared and the reading is the magnitude:
the feedback adds 1.3-1.7 fed rows out of six. Test 5 is `undetermined` because feedback-only fed exactly four rows in
all six drawn runs (SD 0): the magnitude is +2.0 rows. Test 6 is a plain null: with the drawn starts the feedback does
not raise the 60 s mean |DNa02 L-R| at all.

### 3.2 Per (arm, start set): mean +/- SD over the six runs, with the six run values (`descriptive.csv`)

| arm | starts | fed rows | first contact s (fed rows) | mean abs DNa02 L-R Hz | path m | abs turn | abs PFL3 input Hz | abs integral Hz | abs yaw rad/s |
|---|---|---|---|---|---|---|---|---|---|
| full | shipped | 6.000 +/- 0.000 [6,6,6,6,6,6] | 27.819 +/- 1.878 [24.59,28.71,27.21,29.59,29.47,27.35] | 1.810 +/- 0.195 [1.48,1.96,1.71,1.84,2.03,1.84] | 0.651 +/- 0.054 | 0.054 +/- 0.005 | 8.533 +/- 0.708 | 4.746 +/- 0.337 | 0.163 +/- 0.017 |
| goal-only | shipped | 4.667 +/- 0.516 [5,4,4,5,5,5] | 31.845 +/- 3.662 [29.37,27.59,37.92,33.78,30.38,32.03] | 0.623 +/- 0.097 [0.65,0.46,0.76,0.61,0.65,0.60] | 0.988 +/- 0.061 | 0.054 +/- 0.008 | 4.358 +/- 0.600 | 0.000 +/- 0.000 | 0.065 +/- 0.007 |
| feedback-only | shipped | 0.167 +/- 0.408 [0,0,0,1,0,0] | 58.370 (one fed row in one run) | 1.972 +/- 0.076 [1.96,2.06,1.84,1.99,1.97,2.02] | 0.929 +/- 0.048 | 0.063 +/- 0.002 | 12.205 +/- 0.324 | 8.133 +/- 0.231 | 0.176 +/- 0.005 |
| full | drawn | 6.000 +/- 0.000 [6,6,6,6,6,6] | 9.874 +/- 1.257 [11.09,9.17,11.84,9.05,9.17,8.94] | 0.475 +/- 0.075 [0.46,0.48,0.60,0.50,0.39,0.41] | 0.798 +/- 0.108 | 0.013 +/- 0.002 | 2.672 +/- 0.458 | 1.746 +/- 0.308 | 0.054 +/- 0.006 |
| goal-only | drawn | 4.333 +/- 0.816 [5,5,4,4,5,3] | 17.189 +/- 4.667 [24.25,15.94,21.64,15.00,13.44,12.85] | 0.565 +/- 0.234 [0.51,0.31,0.77,0.60,0.32,0.88] | 1.045 +/- 0.057 | 0.052 +/- 0.019 | 4.133 +/- 1.531 | 0.000 +/- 0.000 | 0.060 +/- 0.019 |
| feedback-only | drawn | 4.000 +/- 0.000 [4,4,4,4,4,4] | 6.161 +/- 1.165 [5.65,8.54,5.61,5.73,5.69,5.75] | 0.594 +/- 0.028 [0.56,0.63,0.59,0.61,0.56,0.61] | 1.034 +/- 0.070 | 0.017 +/- 0.001 | 4.053 +/- 0.307 | 2.884 +/- 0.266 | 0.066 +/- 0.002 |

The full six-value lists of every column are in `descriptive.csv` (`*_runs`); the table shortens the last six to
mean +/- SD.

### 3.3 Per room row (`rows.csv`; feeding s of rows 0-5 | first contact s of rows 0-5; pasted, rule 28)

| arm | starts | run | brain seed | feeding s | first contact s |
|---|---|---|---|---|---|
| full | shipped | 0 | 31 | 15.00,15.00,15.00,15.00,15.00,15.00 | 20.7,23.1,30.0,32.7,19.7,21.4 |
| goal-only | shipped | 0 | 31 | 15.00,15.00,15.00,0.00,14.88,15.00 | 36.6,23.2,35.6,none,16.9,34.6 |
| feedback-only | shipped | 0 | 31 | 0.00,0.00,0.00,0.00,0.00,0.00 | none x 6 |
| full | shipped | 1 | 32 | 15.00,15.00,15.00,15.00,15.00,15.00 | 18.5,42.1,29.6,40.7,19.9,21.6 |
| goal-only | shipped | 1 | 32 | 15.00,0.00,15.00,0.00,15.00,14.89 | 36.5,none,38.2,none,18.6,17.0 |
| feedback-only | shipped | 1 | 32 | 0.00 x 6 | none x 6 |
| full | shipped | 2 | 33 | 15.00,15.00,15.00,15.00,15.00,15.00 | 21.0,39.1,28.7,33.2,20.1,21.2 |
| goal-only | shipped | 2 | 33 | 15.00,0.00,15.00,0.00,15.00,15.00 | 44.0,none,32.7,none,38.0,37.0 |
| feedback-only | shipped | 2 | 33 | 0.00 x 6 | none x 6 |
| full | shipped | 3 | 34 | 15.00,15.00,15.00,10.00,15.00,15.00 | 21.2,35.6,28.6,50.0,20.2,21.9 |
| goal-only | shipped | 3 | 34 | 15.00,14.26,7.62,0.00,14.86,14.98 | 36.1,45.8,52.4,none,16.7,18.0 |
| feedback-only | shipped | 3 | 34 | 0.00,0.00,0.00,1.64,0.00,0.00 | none,none,none,58.4,none,none |
| full | shipped | 4 | 35 | 15.00,15.00,15.00,14.60,15.00,15.00 | 20.9,40.0,29.2,45.4,19.9,21.4 |
| goal-only | shipped | 4 | 35 | 14.85,15.00,15.00,0.00,15.00,15.00 | 16.6,42.0,36.9,none,20.3,36.1 |
| feedback-only | shipped | 4 | 35 | 0.00 x 6 | none x 6 |
| full | shipped | 5 | 36 | 15.00,15.00,15.00,15.00,15.00,15.00 | 20.7,39.3,29.4,31.8,21.0,21.9 |
| goal-only | shipped | 5 | 36 | 15.00,14.74,15.00,0.00,14.87,14.89 | 39.0,45.3,42.0,none,16.8,17.1 |
| feedback-only | shipped | 5 | 36 | 0.00 x 6 | none x 6 |
| full | drawn | 0 | 31 | 13.57,14.52,15.00,13.45,15.00,14.02 | 1.9,12.8,23.9,0.4,20.5,7.0 |
| goal-only | drawn | 0 | 31 | 13.58,15.00,7.21,13.45,0.00,15.00 | 2.0,32.0,52.8,0.4,none,34.1 |
| feedback-only | drawn | 0 | 31 | 13.57,0.00,14.56,13.45,0.00,14.04 | 1.8,none,13.0,0.4,none,7.3 |
| full | drawn | 1 | 32 | 15.06,14.62,14.36,13.45,15.00,14.02 | 1.9,13.8,11.0,0.4,20.9,7.0 |
| goal-only | drawn | 1 | 32 | 13.57,15.00,15.00,13.45,0.00,14.49 | 1.9,31.4,33.4,0.4,none,12.6 |
| feedback-only | drawn | 1 | 32 | 14.56,0.00,14.58,13.45,0.00,14.04 | 13.1,none,13.4,0.4,none,7.2 |
| full | drawn | 2 | 33 | 14.95,14.52,15.00,13.45,15.00,14.02 | 1.7,12.9,28.7,0.4,20.4,6.8 |
| goal-only | drawn | 2 | 33 | 13.57,0.00,8.83,13.45,0.00,15.00 | 1.8,none,51.2,0.4,none,33.2 |
| feedback-only | drawn | 2 | 33 | 14.51,0.00,14.57,13.45,0.00,14.04 | 1.7,none,13.1,0.4,none,7.2 |
| full | drawn | 3 | 34 | 15.05,14.62,14.30,13.45,15.00,14.02 | 1.8,13.9,10.2,0.5,20.8,7.1 |
| goal-only | drawn | 3 | 34 | 13.57,0.00,15.00,13.45,0.00,15.00 | 1.9,none,24.3,0.5,none,33.3 |
| feedback-only | drawn | 3 | 34 | 14.53,0.00,14.57,13.45,0.00,14.05 | 1.8,none,13.2,0.5,none,7.4 |
| full | drawn | 4 | 35 | 13.57,14.63,14.39,13.45,15.00,14.02 | 1.8,14.0,11.3,0.4,20.5,7.0 |
| goal-only | drawn | 4 | 35 | 13.57,15.00,14.35,13.45,0.00,14.29 | 1.8,44.0,10.9,0.4,none,10.1 |
| feedback-only | drawn | 4 | 35 | 14.60,0.00,14.58,13.45,0.00,14.04 | 1.7,none,13.4,0.4,none,7.2 |
| full | drawn | 5 | 36 | 13.57,14.60,14.32,13.45,15.00,14.01 | 1.8,13.7,10.4,0.5,20.2,7.0 |
| goal-only | drawn | 5 | 36 | 13.57,0.00,0.00,13.45,0.00,15.00 | 1.8,none,none,0.5,none,36.3 |
| feedback-only | drawn | 5 | 36 | 13.57,0.00,14.59,13.45,0.00,14.04 | 1.8,none,13.5,0.4,none,7.3 |

## 4. Reading

1. **The walking goal is the load-bearing part, not the feedback.** With the shipped starts, feedback-only -- the
   pre-correction upwind goal closed on DNa02 -- feeds 0.17 +/- 0.41 rows of six (one row, once, at 58.4 s) while its
   DNa02 |L-R| (1.97 Hz), |yaw| (0.176 rad/s) and integral (8.1 Hz) are the largest of the three arms: the loop turns
   the fly hard toward a goal that does not point at food. Goal-only, the diagnostic's one-way bridge with the
   bilateral goal, feeds 4.67 +/- 0.52 rows with a DNa02 |L-R| of 0.62 Hz -- inside the 0.28-0.69 Hz range the skeptic
   quoted as evidence the bridge could not steer. The diagnostic's own 2 of 6 (>= 1 s) became 4.7 of 6 by the goal
   alone.
2. **The feedback adds the last rows and is what turns the fly facing away.** Full feeds 6/6 in every run; the row
   goal-only never feeds (0 of 6 runs) is row 3, the 185 deg heading, which full feeds in 6 of 6 runs at 31.8-50.0 s.
   Row 1 (90 deg) also fails in 2 of the 6 goal-only runs, so row 3 is 1.00 of the +1.33 rows and row 1 the
   remaining 0.33. The shipped set has one room per heading, so heading and room are not separable here; with the
   drawn starts the row goal-only never feeds is row 4 (heading -87.9 deg), not a downwind start.
   The 80 Hz one-way input is too weak to recruit a large DNa02 asymmetry (goal-only mean |PFL3 input| 4.4 Hz), so a
   fly that starts facing downwind is not turned round; the integral (4.7 Hz mean, up to 80) is. That is the
   +1.33 rows of test 1, the +1.19 Hz of test 3 and the shorter path of full (0.65 m: the flies arrive and stop).
3. **The drawn starts are less discriminating than intended.** Two of the six drawn rows start just outside a fruit
   -- row 3 at 0.0163 m and row 0 at 0.0318 m from the nearest fruit surface, against the 0.015 m feeding radius --
   and are fed in every arm and run (first contact 0.4-1.9 s, once 13.1 s). They are counted in every drawn
   `fed_rows` figure, so full 6, goal-only 4.33 and feedback-only 4.00 each include two rows no arm has to steer to.
   Over the other four, full feeds 4/4 in every run,
   goal-only 2.33 +/- 0.82 (row 4 never, rows 1 and 2 sometimes) and feedback-only exactly 2 (rows 2 and 5 always, 1
   and 4 never). The drawn set therefore says the same thing with less range, and its 60 s mean |DNa02 L-R| (test 6,
   null) is dominated by the post-feeding phase in every arm. The draw had no exclusion zone around the fruit; a
   reader wanting a clean drawn set needs one.
4. **The shipped-configuration run is one draw beside plume_steering.md's.** Brain seed 31 with the shipped starts
   fed 6/6 here at 20.7 / 23.1 / 30.0 / 32.7 / 19.7 / 21.4 s against the audit's 32.0 / 37.3 / 27.4 / 32.9 / 20.1 / 21.9 s:
   the same outcome, different times, as determinism_gate.md says it must be. Over six runs the shipped-start first
   contact is 27.8 +/- 1.9 s (per-run means 24.6-29.6).
5. What this does not say: nothing about the model's own nose (the 200x pre-transduction gradient is still the
   instrument's, follow-up 2 of the skeptics, not run here), nothing about hunger. The shipped-start rooms still
   deplete before every first contact under `full`, but that is no longer general: over the 36 runs 143 of 216 rows
   reach energy 0 (17.8-19.6 s) and 74 of 151 first contacts precede it -- 28 of 36 under `full` with the drawn
   starts, all 24 under feedback-only drawn, and 7 of 28 under goal-only with the shipped starts. The drawn set
   therefore does exercise the hunger gain below 1.0, which the shipped set did not. No admission, no default.

**Withdrawn:** "energy reaches zero before every first contact as before" -- it holds for the shipped-start `full`
rooms only; 74 of 151 first contacts in this batch precede energy zero.

**Withdrawn:** "Two of the six drawn rows (0 and 3) begin within a fruit's reach" -- neither row starts inside the
feeding shell (0.0163 m and 0.0318 m against a 0.015 m feeding radius); they are counted in every drawn figure, not
excluded.

## 5. Reproduction

```sh
PYTHONIOENCODING=utf-8 python scripts/plume_goal_only.py analyse --runs out/plume-go --out out/plume-go/analysis
```

Committed: `out/plume-go/{batch.sh,jobs.json,predeclared.json}` and `out/plume-go/analysis/{runs,rows,compare,descriptive}.csv`,
`analysis.md`, `summary.json`. Run JSONs, NPZs and consoles stay ignored (host paths).

## 6. Author self-review (the independent skeptic pass is the section after the Report block below)

- The family was frozen with signs; tests 1 and 4 are nulls under the declared rule although the samples separate
  fully -- the reference-SD z is the rule the round chose and the Welch t is printed beside it; the reader may weigh
  them, the audit does not upgrade them.
- `fed_rows` is a count of six with a 15 s satiety ceiling (`feeding_s` 15.00 means "reached the meal length"), so
  ties and SD-0 arms are expected; test 5's `undetermined` is that, not a missing measurement.
- Two drawn rows start just outside a fruit -- row 3 at 0.0163 m and row 0 at 0.0318 m from the nearest fruit
  surface, against the 0.015 m feeding radius -- and are fed in every arm and run (item 3 above); the drawn half is
  weaker than designed and I say so rather than re-drawing after the fact.
- The three arms are three cells of a 2 x 2: (bilateral goal, gain 5), (bilateral goal, gain 0) and (pre-correction
  goal, gain 5). The fourth -- pre-correction goal with gain 0 -- is not in this batch; it exists only as
  plume_steering.md's diagnostic (2 of 6, one draw, a different batch). So +5.83 and +1.33 are simple effects, not
  main effects, and the +5.83 includes whatever the feedback costs when it chases a goal that does not point at
  food. "Load-bearing" is a comparison of those two simple effects.
- Six runs per arm are six draws of a non-repeating room; the brain seeds keep the draws honest, they do not
  identify anything.
- The feedback-only arm uses the pre-correction goal law with the shipped 5 /s gain, the walking-goal arm the shipped
  gradient law with gain 0: the two variants are the shipped code with one term removed each, not new laws; the
  `variant` field in every provenance record says which.
- "Load-bearing" is a statement about these rooms with 19 odour sources and the 200x gradient; it is not a claim about
  a fly's olfactory steering.

This is the author's review, not the independent skeptic pass. That pass ran on 2026-09-18 (Opus, verdict
mostly sound for this audit); its verdict line and all ten of its claim lines are quoted verbatim in
[Skeptic pass (independent, Opus, 2026-09-18)](#skeptic-pass-independent-opus-2026-09-18), after the Report
block, and the corrections it required are applied above.

## Report

```yaml
summary: |-
  The bilateral walking goal, not the DNa02 feedback, is the load-bearing part of the plume correction. With the
  shipped six rooms, feedback-only (the pre-correction upwind goal closed on DNa02) feeds 0.17 +/- 0.41 rows of six
  while showing the largest DNa02 |L-R| (1.97 Hz) and |yaw| of the three arms; goal-only (the walking goal with the
  diagnostic's one-way PFL3 bridge, integral gain 0) feeds 4.67 +/- 0.52 rows at a DNa02 |L-R| of 0.62 Hz; full feeds
  6/6 in every run (first contact 27.8 +/- 1.9 s). Full minus feedback-only is +5.83 rows (result, Holm p 0.013);
  full minus goal-only is +1.33 rows (null under the reference-SD rule, though all six runs separate) and +1.19 Hz
  of DNa02 |L-R| (result) -- the feedback's contribution is the fly that starts facing away (row 3, 185 deg: goal-only
  0/6 runs, full 6/6; row 1 at 90 deg also fails in 2 of 6). Six drawn starts say the same with less range (full 6,
  goal-only 4.33, feedback-only 4.00 rows; two drawn rows start just outside a fruit -- 0.0163 and 0.0318 m from the
  nearest fruit surface against the 0.015 m feeding radius -- and are fed in every arm and run, so every drawn
  fed_rows figure includes two rows no arm has to steer to). Every summary number is a mean over six non-repeating
  runs; the per-row values are one draw each. No admission claim; the plume law is unchanged; the variants are opt-in
  keywords recorded in describe(). The three arms are three cells of a 2 x 2 -- the (pre-correction goal, gain 0)
  cell is not in this batch -- so these are simple effects, not main effects.
key_claims:
- feedback-only feeds 0.17 +/- 0.41 rows of six with the shipped starts; full minus feedback-only +5.83 rows, result.
- goal-only feeds 4.67 +/- 0.52 rows at DNa02 |L-R| 0.62 Hz, inside the range the skeptic read as non-steering.
- full minus goal-only +1.33 rows (null under the declared rule; all runs separate) and +1.19 Hz DNa02 |L-R| (result); the difference is 1.00 row at 185 deg and 0.33 at 90 deg.
- With the drawn starts full 6, goal-only 4.33, feedback-only 4.00 rows; two drawn rows start just outside a fruit (0.0163 / 0.0318 m from the surface against the 0.015 m feeding radius) and are fed in every arm and run.
validation:
- 36 runs, 36/36 consoles device cuda, NVIDIA B200 x 36, backend native, variant per arm verified in every provenance record, starts equal to the frozen list, 0 analysis problems.
- Frozen before submission (out/plume-go/predeclared.json 2026-09-18T00:14:43Z; runs plume-go-c574f6 and plume-go-1c49fb).
- Nothing adopted; bare `plume` is byte-for-byte the shipped law (test).
recommendations:
- Any further plume arm draws its starts with an exclusion zone around every fruit.
- The transduced-contrast law (the skeptics' follow-up 2) is the next question; the goal is where the steering is.
open_questions:
- Whether a walking goal read from ORN rates (not the physical field) still carries 4-5 rows of six.
```


## Skeptic pass (independent, Opus, 2026-09-18)

An independent skeptic pass ran on 2026-09-18 (Opus, CPU only, no cluster job, nothing adopted). Its verdict line
and its claim lines are quoted verbatim below. The CORRECTIONS REQUIRED list is applied in place in the sections
above; where a correction replaced a sentence that stated a finding, the original sentence stays in the record
marked **Withdrawn:** (INTERP 10.4 rule 29 iii). The pass's NOT CHECKED list is recorded verbatim with this
round's entry in [receptor_verification.md](receptor_verification.md).

One pass covered round-8 items 3 and 4 -- this audit and [instrumented_suite.md](instrumented_suite.md) -- and carried one verdict
line per audit; the line for this audit is quoted below, and all ten claims are quoted in each of the two.

### Verdict

```text
VERDICT
- plume_goal_only: mostly sound

All three analysis scripts reproduce their committed CSVs byte-for-byte on the CPU (suite_rows/runs/suite_table, room_runs/room_tests, plume compare/descriptive/rows/runs -- all diff-identical, 0 problems). Branch tests: 514 passed / 15 skipped / 220 subtests, 1 failure (test_connectome_data.py::test_loading_malecns_never_rewrites_cache) that is an artefact of the cache junction, not a branch defect; test_bit_identity.py and test_plume_variants_are_opt_in_and_recorded pass.
```

### Claims

```text
CLAIMS 1-10
1. Suite half + instruments live -- confirmed, and NOT vacuous. Rebuilt the 29 x 6 status table from the JSONs: 27/0/2, 26/1/2, 27/0/2 under both presets, 0 status changes, taste.MN9_hz the only non-uniform row and bit-identical between presets at every seed (10.93417739868164 / 1.6904562711715698 / 3.1295931339263916) -- same seed (1) flips in both arms. Instruments provably active: (a) all 19 instrumented controllers carry the three records and the hold in lif.type_path_gain, cache md5 7a10d93b vs ef23cc27, nt_counts glutamate 29707->29711 / unknown 2361->2357 (the 4 GLNO cells); (b) the compass section moves hard -- during.PEN_hz raw 2.01/1.78/1.08 -> inst 7.72/7.66/7.22, rest_EPG_hz 0.000x3 -> 1.36/2.12/3.36, wedge_hz 61-67 -> 69-80, non-overlapping 3 v 3, exactly the direction of releasing the ExR6/ER6/ER4m hold, with wedge_cells_persisting still 0; (c) on rows that repeat bit-exactly under raw, raw vs instrumented still differ (bitter.shiu_sugar_MN9_hz all 3 seeds incl. 138.934->149.314; walk.power_max/sustained at s2). The audit reports none of (b) or (c). Adapter fix verified: attempt1 lost exactly the 8 named checks, attempt2 has 0 MISSING, the fix and its regression test are on the branch and pass. Re-freeze honest: the diff changes only stamped_utc and two script hashes; rule/jobs_sha256/batch_sha256 unchanged.
2. Apples to apples on seeds/suite/check list -- yes; on instrument scope -- no. Same 29 keys in the same order, same draw seeds 0,1,2; the REFS criteria table untouched; raw column bit-identical on 18 of 29 rows including taste.MN9_hz s1 = 1.6904562711715698, the exact row the stand-in was rejected on. But compass wrote 50 Hz into 46 EPG cells in every brain (it perturbed even the B=1 probes: taste 10.934->2.479 at s0), whereas the round-8 list is bit-identical to raw on 8 of 29 rows by construction -- the afferent is body-less in the legacy probes and the hold/relabel are confined to the CX. Part of why it survives is that it is inert where the stand-in was not; the audit does not say this.
3. Room rate-half -- reproduces exactly; one omission. Exposure 28,800 fly-s/arm verified; hops = escape + voluntary in all 12; 0 problems. The declared test is the two-sample exact Poisson = conditional binomial binomtest(K_inst, K_inst+K_raw, 0.5) one-sided (INTERP 10.4 rule 17) -- recomputed: 110 vs 101, 3.8194/3.5069 per 1,000, one-sided p 0.2910, two-sided 0.5819, run-level compare diff +1.500 z +0.437 p 0.9372 null; escape 42/39 p 0.4122; voluntary 68/62 p 0.3306. The pooled Poisson ignores run clustering, which makes the PASS conservative. Swept every row field: gf_max_hz is the one measure with a rank-test p below 0.05 (inst 31.4/31.6/30.8/33.6/33.2/35.8 vs raw 36.7/36.7/33.9/38.2/34.2/34.9; diff -3.01, z -1.79, Welch -2.94, MW p 0.0152; null under the z>=3 rule). The audit reports the walking-GF median instead and singles out meals (p 0.394) while never mentioning gf_max_hz. Everything else flat. Yaw SD and DNa02 rates are not in the room records at all.
4. The gate is satisfied as written and the audit does not over-claim (one wording defect). PRESETS_SPEC 2.5: 3 draws, 0 changes, 6 runs, p 0.291. The declared touch set is a single row (conservative). Answer == Report.summary word-for-word; self-review labelled; "nothing adopted / raw stays default / law still unverified" all present. Only "this list, unlike the compass stand-in, moves no suite row" over-reaches: values move on 20 of 29 rows, statuses on none.
5. Plume arm means and Holm family -- reproduce exactly. fed_rows recomputed from feeding_s >= 1.0 matches metrics.fed_rows in all 36 runs. feedback-only shipped 0.167+-0.408 at DNa02 1.972+-0.076; goal-only 4.667+-0.516 at 0.623+-0.097; full 6/6. t1 +1.3333 z +2.582 Welch +6.325 null; t2 +5.8333 z +14.289 result; t3 +1.1870 z +12.264 result; t4 +1.6667 z +2.041 null; t5 +2.0000 SD-0 undetermined; t6 -0.0905 p 0.5887 null; Holm 0.012987 (t1-t5) / 0.588745 (t6). Drawn starts reproduce exactly from numpy.random.default_rng(8). Orientation caveat correctly stated by the audit. Sound.
6. "the feedback's share is the fly starting at 185 deg" -- true but incomplete. Goal-only shipped loses 8 row-instances, not 6: row 3 (185 deg) fails 6/6 and row 1 (90 deg) fails 2/6 (runs r1, r2). So row 3 is ~75% of the +1.333. Row 3 is confounded with env seed 3. The single feedback-only success is also row 3 (1.64 s, contact 58.4 s). With the drawn starts the never-fed goal-only row is row 4 (heading -87.9 deg, not "facing away"), so the mechanism does not carry over. The earlier skeptic's finding that the six shipped rooms are near-repeats is not carried forward.
7. "two drawn rows start on a fruit" -- refuted as worded; counted, not excluded. fruit_distance = nearest-fruit-centre distance minus that fruit's radius; feeding is < 0.015 m. At t=0 the six drawn rows sit at 0.03184, 0.17550, 0.14324, 0.01633, 0.13898, 0.07716 m (identical in all 18 drawn runs; shipped rows all 0.226 m). Row 3 is 1.3 mm outside the feeding shell, row 0 is 2.1x the shell -- neither starts on a fruit. They are fed in all 18 drawn runs and are included in 6 / 4.333 / 4.000; the over-the-other-four figures (4/4, 2.33+-0.82, exactly 2) check out. One exception to "0.4-1.9 s in every arm and run": feedback-only drawn r1 row 0 first contacts at 13.11 s.
8. The refutation is earned; the headline is a simple-effect claim, not a main effect. The three arms are the (new goal, gain 5), (new goal, gain 0) and (old goal, gain 5) cells of a 2x2; the (old goal, gain 0) cell is absent, so the interaction is unidentified. Goal-only keeps everything and sets steering_integral_gain_per_s to 0 so steer_input = clip(80*turn, +-80); feedback-only keeps gain 5 and skips exactly one line (the local goal copy), falling back to the pre-correction upwind/entry-memory goal. So +5.83 is "swap the goal given gain 5" and contains whatever harm the servo does when chasing a bad goal -- an upper bound on the goal's share. Nonetheless the core refutation stands: the earlier skeptic predicted a goal-only arm "would probably still show 0.28-0.69 Hz DNa02 differences" and inferred the feedback was load-bearing; goal-only shows 0.623+-0.097 Hz and still feeds 4.67 of 6, so the prediction was right and the inference is refuted.
9. Determinism-gate compliance -- two defects. (a) Section 2 declares "every number below is a mean +/- SD over runs and nothing is quoted beyond that", but section 3.3 quotes 36x6 single-draw values and section 4 items 1/2/4 quote single draws. Legitimate under rule 28 (pasted from rows.csv) and item 4 labels its numbers "one draw" -- but the blanket sentence is false. (b) "energy reaches zero before every first contact as before" is FALSE here. zero_energy_s spans 17.81-19.57 s and only 143 of 216 rows reach 0 at all; 74 of 151 first contacts precede energy zero (full drawn 28/36 at 0.4-14.0 s; goal-only shipped 7/28 at 16.6-18.0 s; goal-only drawn 15/26; feedback-only drawn 24/24). Only shipped-start full and feedback-only satisfy the old statement.
10. Code -- clean. PlumeNavigation(c, *, feedback_gain_per_s=None, walking_goal=True); the diff from the merge base is two kwargs, three parameters entries and one if self._walking_goal: guard. The integral reads self.parameters["steering_integral_gain_per_s"], literally 5.0 for bare plume, so bare plume is arithmetically unchanged; test_bit_identity.py passes and test_plume_variants_are_opt_in_and_recorded asserts the shipped parameters. Grammar plume:feedback=0 / plume:walking_goal=0 with the three ValueError cases tested. describe() records variant / walking_goal_enabled / steering_integral_gain_per_s -- verified in all 36 plume records and the suite/room records. Stamps precede runs in all three batches. No infrastructure identifiers in the branch diff. Answer == Report.summary in instrumented_suite.md; plume_goal_only.md uses "## 4. Reading" with no Answer section, matching determinism_gate.md, and its summary's claims all check out. Self-review labelled in both. Integration note: feat/round8 branched at a455bcf and main has since merged astra/plume-transduced, which gives PlumeNavigation.__init__ an incompatible signature (bilateral=, feedback=) -- flyverse/navigation.py and flyverse/instruments.py will conflict on merge and the two feedback semantics need reconciling.
```
