# The level control matched on all three leg channels: the cycle's structure term as a direct matched difference (thread level controls, round 4d)

Generator: `scripts/probe_vnc_drive.py --family level4` (`plan` / `analyse` / `pairs` CPU; `room` GPU), the CPU derivation
`scripts/derive_level_fixed_point.py` (-> `out/vncd7/fixed_point_derivation.json`, console
`out/vncd7/fixed_point_derivation_console.txt`), the CPU checks `out/vncd7/verify_runs.py`, `out/vncd7/level_model.py`
(declared secondary) and `out/vncd7/per_seed_lists.py`. Data: `out/vncd7/` (ONE submission, `out/vncd7/batch.sh` -> run
**vncd7-9bdd12** on the house cluster, B200, console `out/vncd7_cluster.log`: **24 job(s), 0 failed (18.5 min)**); analysis
`out/vncd7/analysis_console.txt`, `out/vncd7/analysis/pairs_console.txt`, `out/vncd7/verification_console.txt`,
`out/vncd7/level_model_console.txt`. Predeclaration `out/vncd7/predeclared.json`, stamped **2026-09-15T19:22:14Z** (archived
unchanged as `out/vncd7/predeclared_archive/predeclared_2026-09-15T192214Z.json`) against the earliest run's own
`started_utc` **19:22:56Z**; submission receipt `out/vncd7/submitted_at.txt` (19:22:25Z). Working-tree record
`out/vncd7/tree_state.json` (sha256 of every shipped file; `tree_state_check_astra.json` at closeout; stamp limits in section 10).
Every number below is a run mean +- SD over the SIX runs of an arm (16 flies x 55 s window each; runs are the replicate
unit); verdicts are `flyverse.interp.common.compare` on runs (6 v 6, exact-U floor p 0.0021645). **Every per-seed list in
this document is pasted from `out/vncd7/analysis/per_seed_lists.txt`, which `per_seed_lists.py` emits from
`out/vncd7/analysis/per_seed.csv` (columns `key, arm, seed, file, value`, written by `pairs`) and from
`analysis/level_model_r2_residual_runs.csv`; the file and column are named beside each list** (docs/INTERP.md 10.4 item
28). Nothing in `out/vncd7` is compared row by row with `out/vncd6` or `out/vncd5`; their numbers appear as labelled context,
as the derivation's targets and anchor, and as the level model's calibration.

## 0. Answer

**The fixed point landed, and the cycle retains a matched structure term.** All three leg channels
meet the declared per-side tolerances. C minus L3 raises AN04B003 by **+2.6687 Hz left / +3.8782 Hz right**
(both `result`, Holm p 0.0152), a direct pooled difference of **+3.2734 Hz**. DNa02-left is `result`
(+0.1342 Hz, z 4.996); DNa02-right (+0.0504 Hz, z 1.794), clean yaw SD (+0.5918 deg/s, z 2.284),
straightness and DNa02 L-R are `null`. A positive mean difference alone does not pass the declared rule.

The modulation-only M2 arm also exceeds L3 at the relay (**+2.8521 / +4.0290 Hz**, pooled **+3.4406 Hz**),
DNa02-left (+0.1260 Hz) and clean yaw SD (+0.7880 deg/s); those four primaries are `result`.
**M2 versus C is `null` on all seven primaries**, repeating round 4c's outcome: no extra drive from the
amplitude/turn law is detected at this level. This is a null test, not an equivalence bound. The declared
sided secondary still separates their DNa02 response: tripod-conditioned L-R swing **-0.3543 Hz C /
-0.0884 Hz M2**, with L3 +0.0074 (numerical residue of a nominally unsided input, not a physiological signal).

The direct matched term is smaller than the earlier slope-corrected, cross-batch +4.49 / +4.60 / +4.72 Hz
context. The old level model leaves **+1.2028 / +1.2344 Hz** residual in the modulation-free L3 control
at the cycle's level point, showing the limitation of its extrapolation. Correcting the small remaining
C-L3 level mismatch gives **+2.7790 / +3.9415 Hz**, pooled **+3.3602 Hz**, as a declared secondary;
the primary conclusion uses the direct matched difference, not that correction.

This closes round 4d's matching question on **24 CUDA runs, six per arm, one house submission**.
All preconditions pass; runtime, defaults and MaleCNS cache are unchanged, and nothing is adopted.
**Independent skeptic pass (Opus, 2026-09-17): mostly sound** -- all seven claims reproduced, the corrections
it required are applied here, and no conclusion moved ("Skeptic pass" below). The handoff overstated the saved prose:
sections 4-11 and Report were also placeholders at ad2efc0. This closeout reconstructs them from the
completed recordings and regenerated analysis. Source-stamp limitations and a misleading descriptive
rate column are disclosed in sections 8 and 10; neither is used to infer a primary result.

## 1. The question and the design

Three batches (rounds 4, 4b, 4c: `docs/audits/level_matched_control.md`, `level_controls.md`, `level_controls_r2.md`)
measured the cycle's excess over a level control at the ascending relay AN04B003, and every one of them read it against a
control that was NOT matched on all three leg channels: L (the round-2 law at `mn_ref_hz` 8.84) matches the chordotonal
level and carries the law's own hair plate (+9.5 Hz above the cycle arm's) and campaniform (+24.7 Hz), so the +4.5 / +4.6 /
+4.7 Hz "structure terms" of those rounds are slope-CORRECTED numbers, at slopes fitted on three steady arms that lie
near one line in the (chordotonal, hair-plate) plane and extrapolated ~9.5 Hz of hair plate off it (round 4c section 7;
its skeptic pass R8 / R6); K (round 4b) matched the hair plate and the campaniform at a fixed `mn_ref` and lost 8.7 Hz of
chordotonal through the afferent -> leg-MN loop, so C v K was an upper bound. Round 4c's item 2 and round 4b's item 3
both named the arm that closes this: the three sense parameters derived TOGETHER as one fixed point of the loop, so that
ONE steady arm sits at the cycle's window means on every leg channel, and C v that arm is a direct matched difference.

**Design (four arms x 6 brain seeds = 24 room jobs, blocks `fam_r<seed>`; no compass).**

| arm | spec | classification |
|---|---|---|
| A | shipped (no sense) | the reference |
| **L3** | `'all+unsided'`, `mn_ref_hz` **8.23**, `hair_plate_max_hz` **81.09**, `campaniform_load_hz` **25.10** | LABELLED CONTROL: the round-2 transducer under the existing `unsided` token (round 4b's arm U) with its three parameters derived together (section 3) so that its realised chordotonal / hair-plate / campaniform window means equal the cycle arm's on both sides; the sense's defaults (30 / 100 / 50) are untouched, `senses.py` is unchanged |
| M2 | `'all+leg_cycle+leg_cycle_flat'`, `--flat-amplitude-value 0.948` | LABELLED CONTROL of the body-model mechanism (round 4c's modulation-only cycle at the cycle's realised amplitude), re-run inside this batch |
| C | `'all+leg_cycle'` | the body-model mechanism (rounds 3, 4, 4b, 4c's arm C): the treatment |

Primaries per pair (7 keys, Holm m = 7 within each family, 6 runs per arm): `DNa02_L_hz`, `DNa02_R_hz` (on the same
clean mask as `DNa02_LR_hz`), `yaw_sd_clean_deg_s`, `straightness`, `DNa02_LR_hz`, `chain_AN04B003_L_hz` /
`chain_AN04B003_R_hz`. Four predeclared families: **F1 C v L3** (the matched structure term), **F2 M2 v L3** (the
modulation alone, matched), **F3 M2 v C** (round 4c's F1 replicated), **F4 C v A** (the fraction denominators). Each family's
decision rule is quoted verbatim in section 6. Two further pairs (L3 v A, M2 v A) are bookkeeping only. At 6 v 6 the floor
is 2 / C(12, 6) = 0.0021645 and m = 7 gives a smallest adjusted p of 0.0152 (docs/INTERP.md 10.2, 10.4 item 1).

**Why `unsided`.** The task asks for the match on BOTH sides, and under the sided MN-rate law it cannot be had at any value
of the three parameters: the leg-MN side bias (5.25 / 4.40 Hz at `mn_ref` 8.84) puts +13.5 Hz of chordotonal and +9.1 Hz of
hair plate between the sides by construction (round 4c's L: 92.59 / 79.12 against C's 87.23 / 87.59), five to eight Hz
outside a 3-Hz per-side tolerance. With every leg cell reading the side-mean leg-MN rate (the `unsided` token, four lines
of `senses.py` that already exist as round 4b's labelled control U) both channels are side-symmetric, and -- the fact the
derivation rests on -- ONE drive fraction `d = mean[clip(legMN_mean / mn_ref, 0, 1) x ground]` sets both: chordotonal
= 10 + 140 d and hair plate = 5 + (hp_max - 5) d exactly, on every cell and every frame. Round 4b found that U v L moves
the relay's side balance (-1.01 / +0.85 Hz) and none of its pooled level (-0.08), none of DNa02's rate, and owns the
straightness / drift rows, so the pooled structure term is comparable with the earlier rounds' and the straightness row is
read against U's, not L's (the predeclared rule says so).

## 2. No new mechanism; the tests

No `flyverse/` file changes for this round. The three values travel on the job line (`--mn-ref-hz 8.23
--hair-plate-max-hz 81.09 --campaniform-load-hz 25.1`, the family table `ARM_SENSE_KW['level4']['L3']`, which for this arm
carries `mn_ref_hz` beside the two channel parameters; `mn_ref_of` reads it there) into `senses.Proprioception`'s own
constructor keywords, and the run JSON's `sense` block records them as `overrides` with `defaults_used` false. The probe
gains the `level4` family (`ARMS_LEVEL4`, `PAIRS_BY_FAMILY['level4']` with C v L3 first, `ARM_ORDER` with L3, `watch_of`, the
plan name `vncd7`) and nothing else; `flat_amplitude_value` 0.948 for M2 is round 4c's `ARM_CYCLE_KW` entry repeated for the
new family.

**Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`).** `tests/test_proprioception.py::ThreeChannelMatchedControlTests` pins the algebra
the derivation rests on -- under `'all+unsided'` with (8.6, 81.09, 25.10) every chordotonal cell reads 10 + 140 d and every
hair-plate cell 5 + 76.09 d with the SAME d on the synthetic graph's L / R / unsided / sensory_ascending cells, so
(hair - 5) / (hp_max - 5) == (chord - 10) / 140 to 1e-12 at five (leg_L, leg_R) pairs including the clip and the mirror,
the campaniform is `load` on the ground and 0 airborne, and a bare `'all+unsided'` still gets the sense's defaults (30 /
100 / 50) -- and the family tables (L3's three keys, the other arms carrying none, the earlier families untouched, an
explicit flag winning over the table). The shipped golden and every earlier test pass unchanged:

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_proprioception.py tests/test_body_cycle.py tests/test_bit_identity.py -q
    38 passed, 5 subtests passed                     (out/vncd7/smoke/pytest_cpu.txt)

A CPU smoke of the L3 arm through the family table (2 flies x 0.5 s, `out/vncd7/smoke/room_L3_r0.json`): family `level4`,
spec `all+unsided`, `sense.mn_ref_hz 8.23`, `hair_plate_max_hz 81.09`, `campaniform_load_hz 25.1`, overrides all three,
`tokens.unsided` true.

## 3. The joint derivation (CPU; `scripts/derive_level_fixed_point.py` -> `out/vncd7/fixed_point_derivation.json`, stamped 19:21:59Z)

### 3.1 The targets, and why `out/vncd6`'s C

The realised window-mean commanded rates of the cycle arm C of `out/vncd6` (round 4c: six runs, 16 flies x 55 s, B200),
recomputed from `room_C_r*_body.npz` on the post-skip frames: **chordotonal 87.404 +- 1.616** (L 87.229 / R 87.587; per run
[86.5014 85.2857 88.3181 87.7931 89.8892 86.6377]), **hair plate 47.067 +- 0.879** (L 46.968 / R 47.171), **campaniform
24.864 +- 0.053**; ground fraction 0.9946, leg MN 4.783 / 4.647. `out/vncd5`'s C (five runs: 86.109 / 46.363 / 24.907) is
reported as context and not used: the vncd6 batch is the most recent, has six runs rather than five, and ran on the same
`body.py` / `senses.py` this batch's C runs on; its C is +1.3 Hz of chordotonal above vncd5's, inside the same arm's
between-batch scatter, and whichever batch the targets come from, the match is verified against THIS batch's own C from
the new recordings (precondition P1), which is the C that L3 is compared with.

### 3.2 The algebra: two of the three parameters are exact

Under `unsided` (section 1) the window means obey chordotonal = 10 + 140 D, hair plate = 5 + (hp_max - 5) D, campaniform
= load x ground, with one D. So D* = (87.404 - 10) / 140 = **0.552887**, **`hair_plate_max_hz` = 5 + (47.067 - 5) / D* =
81.086 -> 81.09** (set by the target RATIO, independent of the loop), and **`campaniform_load_hz` = 24.864 / 0.9908 = 25.096
-> 25.10**, where 0.9908 +- 0.0012 is the pooled ground fraction of the four steady GPU arms on record (vncd5 L / U / K,
vncd6 L; the CPU calibration runs never leave the table in 12-30 s and read ground 1.000, so the fraction has to come from
the GPU protocol). The one loop parameter is `mn_ref`: D depends on the leg-MN distribution the VNC produces FROM the
afferent input the three parameters set.

### 3.3 What a short CPU run cannot see: the window factor (the transfer defect of rounds 4 / 4b, measured)

Round 4b's K realised 77.4 Hz on the GPU against 81.0 on its CPU calibration (-3.6 Hz), round 4's L 86.2 against a
predicted 88.1. The reason is in the recordings: the leg-MN rate is not constant over the 60-s protocol. Every arm shows a
dip at 24-34 s (some flies' side-mean leg-MN rate falls to 3.0-3.5 Hz for several seconds; e.g. vncd6 L 5.0 -> 4.5 Hz, vncd5
U 5.0 -> 4.1 Hz per-second means) and the flies leave the table after ~45 s (on-table fraction 0.59-0.84 at 45-55 s), so the
5-60 s window mean sits BELOW the 2-12 s window a 12-s calibration run realises by a **window factor W = (chord_full -
10) / (chord_early - 10)** of **0.9648 (vncd5 L), 0.9550 (U), 0.9469 (K), 0.9710 (vncd6 L): 0.9594 +- 0.0106**
(`fixed_point_derivation.json` `steady_gpu_arms`). The CPU calibration window matches the GPU's EARLY window, not its
full one: vncd5's cal_L 86.97 sits against L's 88.8 early / 86.1 full, cal_K2 80.99 against K's 81.1 early / 77.4 full.

### 3.4 The fixed point: the GPU anchor's full-window distribution scaled by the CPU-measured loop gain

The fixed point is therefore solved on the FULL-WINDOW per-frame side-mean leg-MN distribution of a GPU anchor arm with
the same law and token as L3 -- `out/vncd5` arm U (`'all+unsided'` at 8.84 / 100 / 50; 5 runs x 16 flies x 5,500 frames =
440,000 (fly, frame) samples) -- whose `D_full(8.84)` on that distribution is 0.534555 -> **84.838 Hz, the recorded 84.838**
(the law reproduces the recording; the clip binds on 0.78 % of frames), scaled by the LOOP GAIN g the CPU measures:

    D_full(mn_ref; g) = mean over the anchor's (fly, frame) samples of clip(g x m(t) / mn_ref, 0, 1) x ground(t),  solve D_full = D*

with **g_k = m_L3,k / m_U**, the ratio of the L3 candidate's leg-MN rate to the anchor's on the SAME CPU protocol (a
calibration PAIR: brain seed 0, env seeds 0..7, 8 flies, 12 s, window 2-12 s, `--device cpu`, `CUDA_VISIBLE_DEVICES=-1`,
770-825 s wall each). The pair carries the loop's response to the three parameters together (the hair plate falls 10-12 Hz
and the campaniform 25 Hz between the anchor and every iterate, not only the chordotonal); the GPU anchor carries the
protocol's window. The CPU anchor `cal_U_r0` (8.84 / 100 / 50) realised chordotonal **88.468** (fly SD 2.51), hair 58.246,
campaniform 50.000, leg MN 5.185 / 4.734 (mean **4.959**) against the GPU anchor's early window 88.361 / 4.952: **CPU - GPU
+0.107 Hz of chordotonal, +0.007 Hz of leg MN** -- the CPU proxy is faithful for the early window.

**The iterations** (`step_2_cpu_calibration_pairs.iterates`; hp_max 81.09 and load 25.10 on every one; the "early-window
prediction" is D on the anchor's 2-12 s distribution x g, the check that the scaled distribution predicts the CPU run itself):

| k | mn_ref | CPU chordotonal (L = R) | hair | camp | leg MN L / R (mean) | **g** | early-window prediction (error) | predicted GPU full window chord / hair | next mn_ref (g model) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 8.5416 | open loop: D_full(mn; g = 1) = D* on the anchor | | | | 1 (assumed) | 91.06 if g were 1 | 87.40 / 47.07 by construction | 8.54 |
| 1 | 8.54 | **86.623** (fly SD 2.51) | 46.645 | 25.100 | 4.835 / 4.522 (4.678) | **0.9433** | 86.537 (+0.086) | 83.10 / 44.73 | 8.058 (g held) |
| 2 | 8.06 | **94.771** (3.83) | 51.073 | 25.100 | 5.059 / 4.716 (4.888) | **0.9855** | 94.561 (+0.210) | 90.74 / 48.88 | 8.264 (secant) |
| 3 | 8.26 | **89.438** (1.95) | 48.174 | 25.100 | 4.864 / 4.535 (4.699) | **0.9476** | 89.454 (-0.016) | 85.88 / 46.24 | **8.230** (least squares over all three) |

The identity (hair - 5) / (hp_max - 5) = (chord - 10) / 140 holds on every CPU run to six decimals (0.547308 / 0.605504 /
0.567413), as the algebra says it must. The loop gain rises as `mn_ref` falls (more afferent drive, more leg-MN rate:
dg / dmn_ref = -0.083 per Hz over the three points, i.e. dm / d(chordotonal) ~ +0.026 Hz per Hz between iterates 1 and 2, the
size of round 4's CPU loop slope 0.032), and each measured g carries the pair's fly-level noise (~+-0.02 at 8 flies: iterate 3
at 8.26 read 0.948 where the secant through 1 and 2 predicted 0.968), so the chosen value uses the least-squares line
through ALL three (mn_ref, g) points rather than the last-two secant, which would chase that noise. **Chosen: `mn_ref_hz`
8.23, `hair_plate_max_hz` 81.09, `campaniform_load_hz` 25.10**; g at the chosen value 0.9635; D_full 0.552892; **predicted
GPU realisation 87.405 +- 1.40 / 47.070 +- 0.76 / 24.869 +- 0.03 Hz** (targets 87.404 / 47.067 / 24.864; the +- is the
window factor's spread across the steady arms (relative 0.011) and the CPU pair's fly-level SE on the drive ratio (relative
0.014), propagated on (rate - tonic); d chord / d mn_ref -9.2 Hz per Hz open loop). Nothing about behaviour enters; the
three numbers are a level match.

### 3.5 The verification pair (longer CPU runs at the chosen values; nothing re-fitted after it)

L3 at (8.23, 81.09, 25.10) and the anchor U, 8 flies x 30 s (window 2-30 s; 1,583 / 1,866 s wall): L3 chordotonal
**89.178** (L = R), hair 48.033, campaniform 25.100, haltere 17.35, leg MN 4.807 / 4.517 (mean 4.662); U 86.558, leg MN 4.839;
**g over the 28-s window 0.9634 against the model's 0.9635** at the chosen value, and the prediction with this g is
87.394 / 47.064 Hz. Per second the ratio drifts (first 10 s 0.948, last 10 s 1.022; the anchor's leg-MN rate dips more in
the 24-34 s window than L3's), so the one-g scaling is an average over the window, not a constant in time; its window mean
is what the fixed point uses and it is what the pair reproduces. On the same CPU protocol the matched pair already
separates the relay: **AN04B003 20.45 / 19.91 (L3) against 18.41 / 17.58 (U)** at 89.2 vs 86.6 Hz of chordotonal, and the
CPU cycle arm `cal_C_r0` (8 x 12 s) reads 24.45 / 25.23 at 92.7 Hz -- the CPU prediction of the matched structure term is
~+4-5 Hz per side, the size of the slope-corrected ones.

## 4. The realised match and the preconditions

`out/vncd7/arm_levels.csv`, columns `cmd_<channel>_<side>_mean`; window 5-60 s.
These are mean commanded rates, not a claim that the cycle and level-control time series match.

| arm | chord L / R (Hz) | hair L / R | camp L / R |
|---|---|---|---|
| L3 | 87.444 / 87.444 | 47.091 / 47.091 | 24.978 / 24.978 |
| M2 | 87.438 / 87.407 | 47.078 / 47.070 | 24.863 / 24.877 |
| C | 86.703 / 87.023 | 46.680 / 46.866 | 24.878 / 24.887 |

P1 passes: L3-C per-side gaps are chord +0.741 / +0.421 Hz (tolerance 3), hair +0.411 / +0.225 (3),
camp +0.100 / +0.091 (1.5). L3's pooled realised means are 87.444 / 47.091 / 24.978 Hz, only
+0.039 / +0.021 / +0.109 from the derivation. P2 passes: M2-C pooled gaps +0.563 / +0.304 / -0.013 Hz;
M2 amplitude is exactly 0.948 while walking and its amplitude L-R is zero. P3 passes: L3 chord/hair L-R
zero. P4 brackets, P5 configurations and P6 predeclaration ordering pass. Detailed checks are in
`out/vncd7/precondition_checks.csv` and the re-run verifier log `out/compass7/4d_verify.log`, which is
not retained in the repository. The preconditions use six-run arm means, not a requirement that every stochastic
run or instantaneous rate lies inside the matching tolerance.


## 5. Primary measurements and the four families

`analysis/decision_table.csv`, unrounded inputs; runs are replicates, six versus six. Each family has
Holm m=7, with minimum adjusted p 0.0151515. The correction is within each family, not round-wide.
`called` requires both `common.compare` result (|z| >= 3, p <= 0.05) and Holm <= 0.05.
DNa02 L, R and L-R share post-skip non-airborne frames; clean yaw uses the stricter walking/table mask.
AN04B003 uses the all-window per-fly recording. These windows are not interchangeable.

| family | key | treatment - reference | z | p | Holm p | verdict / called |
|---|---|---|---|---|---|---|
| F1 | DNa02_L_hz | +0.1342 | +4.9959 | 0.002165 | 0.015152 | result / True |
| F1 | DNa02_R_hz | +0.0504 | +1.7945 | 0.015152 | 0.030303 | null / False |
| F1 | yaw_sd_clean_deg_s | +0.5918 | +2.2837 | 0.002165 | 0.015152 | null / False |
| F1 | straightness | +0.0097 | +0.3902 | 0.818182 | 0.818182 | null / False |
| F1 | DNa02_LR_hz | +0.0838 | +2.9530 | 0.004329 | 0.015152 | null / False |
| F1 | chain_AN04B003_L_hz | +2.6687 | +29.7355 | 0.002165 | 0.015152 | result / True |
| F1 | chain_AN04B003_R_hz | +3.8782 | +27.0592 | 0.002165 | 0.015152 | result / True |
| F2 | DNa02_L_hz | +0.1260 | +4.6901 | 0.002165 | 0.015152 | result / True |
| F2 | DNa02_R_hz | +0.0558 | +1.9864 | 0.008658 | 0.017316 | null / False |
| F2 | yaw_sd_clean_deg_s | +0.7880 | +3.0406 | 0.002165 | 0.015152 | result / True |
| F2 | straightness | +0.0090 | +0.3651 | 0.937229 | 0.937229 | null / False |
| F2 | DNa02_LR_hz | +0.0702 | +2.4734 | 0.004329 | 0.015152 | null / False |
| F2 | chain_AN04B003_L_hz | +2.8521 | +31.7797 | 0.002165 | 0.015152 | result / True |
| F2 | chain_AN04B003_R_hz | +4.0290 | +28.1119 | 0.002165 | 0.015152 | result / True |
| F3 | DNa02_L_hz | -0.0082 | -0.3766 | 0.309524 | 1.000000 | null / False |
| F3 | DNa02_R_hz | +0.0054 | +0.2513 | 0.588745 | 1.000000 | null / False |
| F3 | yaw_sd_clean_deg_s | +0.1962 | +1.1322 | 0.132035 | 0.924242 | null / False |
| F3 | straightness | -0.0006 | -0.0150 | 1.000000 | 1.000000 | null / False |
| F3 | DNa02_LR_hz | -0.0136 | -0.3878 | 0.309524 | 1.000000 | null / False |
| F3 | chain_AN04B003_L_hz | +0.1835 | +0.8697 | 0.937229 | 1.000000 | null / False |
| F3 | chain_AN04B003_R_hz | +0.1509 | +0.6792 | 1.000000 | 1.000000 | null / False |
| F4 | DNa02_L_hz | +0.5411 | +201.5005 | 0.002165 | 0.015152 | result / True |
| F4 | DNa02_R_hz | +0.3079 | +35.4336 | 0.002165 | 0.015152 | result / True |
| F4 | yaw_sd_clean_deg_s | +5.1387 | +35.4273 | 0.002165 | 0.015152 | result / True |
| F4 | straightness | -0.1540 | -169.8761 | 0.002165 | 0.015152 | result / True |
| F4 | DNa02_LR_hz | +0.2332 | +32.1454 | 0.002165 | 0.015152 | result / True |
| F4 | chain_AN04B003_L_hz | +22.3682 | +2493.9286 | 0.002165 | 0.015152 | result / True |
| F4 | chain_AN04B003_R_hz | +23.1260 | +4174.4503 | 0.002165 | 0.015152 | result / True |

F1 calls 3/7, F2 4/7, F3 0/7 and F4 7/7. In particular, F1 DNa02 L-R has z **2.9530**, below 3
despite its small p; rounding it to 3 would change the decision incorrectly. F2 clean yaw has z 3.0406;
F1 yaw has z 2.2837. F1's clean-yaw p is at the exact-U floor because the six seed pairs separate completely
(C min 7.7038 > L3 max 7.5874); the null is the |z| >= 3 criterion, not an absence of a consistent difference.
No precision or significance is borrowed from the 16 flies within a run.


## 6. The predeclared decision rules, applied

The following rules are copied from `out/vncd7/predeclared.json`, `primaries.families`.

### F1_C_v_L3

> With precondition P1 met (L3's three leg channels at C's, per side, within 3 / 3 / 1.5 Hz): if C v L3 is RESULT POSITIVE on chain_AN04B003_L / _R, the per-leg / per-phase STRUCTURE of the cycle raises the ascending relay above a steady transducer of the SAME three-channel level, and the pooled AN04B003 difference IS the structure term, a direct matched measurement with no slope correction; its size is reported per side and pooled beside the three slope-corrected terms, and the cross-batch level model's residual for L3 (a modulation-free arm at the cycle arms' own level point) is reported as what the earlier extrapolation was worth. If C v L3 is RESULT on DNa02_L / DNa02_R as well, the structure reaches DNa02 at a matched level; if C v L3 is NULL on AN04B003, the structure term of the earlier rounds was the unmatched channels and the audit says so. If P1 FAILS on any channel / side, the pair is LEVEL-MISMATCHED on that channel: the raw rows are reported, the gap and its direction are stated, a positive C v L3 is read as a BOUND (upper if L3 sits below C on the chordotonal, lower if above) and not as the size, and the level-corrected reading of the secondaries is quoted with its propagated slope uncertainty. The three behavioural / sided rows (clean yaw SD, straightness, DNa02 L-R) are reported and do not decide the drive question; straightness and the drift are known sidedness rows of the round-2 law that the unsided L3 removes (round 4b's U), so a straightness difference here is read against U's, not L's.

**Reading:** P1 passes; both relay sides are positive results. The pooled matched term is +3.2734 Hz. The DNa02 result is left-only; right and all three behavioural/sided primaries are null.

### F2_M2_v_L3

> With P1 and P2 met: if M2 v L3 is RESULT POSITIVE on chain_AN04B003_L / _R, the per-phase modulation alone, at the cycle's level on all three channels, raises the relay above the steady transducer of that level -- round 4c's +4.60 Hz (slope-corrected) is confirmed as a matched measurement and its matched size is this pair's pooled AN04B003 difference. If M2 v L3 is NULL on AN04B003, the modulation-only arm does not exceed a fully matched level control and round 4c's structure term was carried by the unmatched channels. DNa02_L / DNa02_R are read the same way for DNa02. The sided / behavioural rows are reported, not attributed.

**Reading:** P1/P2 pass; both relay sides and DNa02-left are positive results. The pooled modulation-only matched term is +3.4406 Hz; right DNa02 is null. The yaw result is reported without assigning a mechanism to it.

### F3_M2_v_C

> With P2 met: if M2 v C is NULL on chain_AN04B003_L / _R and DNa02_L / DNa02_R, round 4c's closure of F4 replicates (the amplitude / turn law adds no drive beyond the modulation at the same level). If M2 v C is RESULT on any of those four rows, round 4c's null did not replicate and the audit reports the row, its sign and size against C's own structure term. The per-frame sided DNa02 signal (E[DNa02 L-R | chord L-R > 0] - E[. | < 0]) is a declared secondary and is expected to separate the two arms as in round 4c (-0.338 C vs -0.091 M2); it is reported for all four arms.

**Reading:** P2 passes; all seven primaries are null, including all four named rate rows. Round 4c's null replicates under the declared rule; this does not establish zero effect or equivalence.

### F4_C_v_A

> C v A RESULT on DNa02_L / DNa02_R / the clean yaw SD / AN04B003 means the cycle fires DNa02 and the relay and raises the yaw SD over the shipped path, as in rounds 3, 4, 4b and 4c; the fractions for L3 and M2 are then reported with their per-seed scatter.

**Reading:** All seven are results. Fractions are descriptive decompositions of this within-batch treatment difference, not new tests.



## 7. The level model: declared secondary, cross-batch calibration

`out/vncd7/level_model.py` uses the unchanged vncd5 L/U/K six arm-side calibration points, not a
within-batch refit (L3 alone supplies two points for four parameters). AN04B003 slopes are
chord +0.18485 +- 0.03548 and hair -0.06475 +- 0.02851 Hz/Hz. The fit extrapolates off its narrow
calibration line. L3's residual +1.2028 left / +1.2344 right is a direct check of that extrapolation;
it cannot be called modulation because L3 has no cycle. The nominal fit uncertainty does not cover
this model discrepancy.

From `analysis/level_model_r2_pairs.csv`, AN04B003 only, `raw_diff`, `level_correction`,
`level_correction_unc`, `residual_diff`. Uncertainties are propagated calibration slope uncertainty,
not uncertainty on a new physiological parameter. Secondary verdicts in that file are unadjusted;
the four primary families above decide the round.

| pair | side | raw difference | level correction +- uncertainty | residual difference |
|---|---|---|---|---|
| M2vC | L | +0.1835 | +0.1100 +- 0.0284 | +0.0735 |
| M2vC | R | +0.1509 | +0.0578 +- 0.0148 | +0.0931 |
| M2vC | pooled | +0.1672 | +0.0839 +- 0.0216 | +0.0833 |
| CvL3 | L | +2.6687 | -0.1103 +- 0.0288 | +2.7790 |
| CvL3 | R | +3.8782 | -0.0633 +- 0.0163 | +3.9415 |
| CvL3 | pooled | +3.2734 | -0.0868 +- 0.0225 | +3.3602 |
| M2vL3 | L | +2.8521 | -0.0003 +- 0.0004 | +2.8525 |
| M2vL3 | R | +4.0290 | -0.0055 +- 0.0014 | +4.0346 |
| M2vL3 | pooled | +3.4406 | -0.0029 +- 0.0009 | +3.4435 |

The same file also carries the DNa02 residual differences, where pooled C-L3 is +0.0941 Hz (z +4.3, `result`);
those are unadjusted secondary verdicts from `level_model_r2_pairs.csv`, not one of the four primary families.

Labelled cross-batch context only: earlier corrected terms +4.49 Hz (vncd5), +4.60 M2 / +4.72 C
(vncd6, uncertainty 0.31 / 0.32) exceed this batch's direct 3.27 / 3.44 Hz. No old run is compared
row by row with a new run. The unmatched channels and extrapolation affected the earlier size;
the positive matched difference establishes that they did not explain away the entire structure term.


## 8. Declared descriptive rows and limitations

`analysis/sided_frames.csv`, `<arm>_mean`: the per-frame DNa02 L-R swing conditioned on chord L-R
is -0.3543 Hz C, -0.0884 M2, +0.0074 L3, unavailable in A. AN04B003's corresponding swing is
-3.6192 / -3.6288 / -0.0509 Hz. L3 is nominally unsided; correlations or sign-conditioned values on
its floating-point L-R residue are not interpreted as biological sidedness. C's extra DNa02 swing
does not translate into an F3 rate or behavioural primary result.

`analysis/dna02_decompose_summary.csv` gives descriptive rate-weighted input, not a causal intervention:
C-L3 AN04B003 contribution is +41.222 / +58.649 mV/s per DNa02 cell, with SNpp45 about 128.2 vs 129.3
left and 59.1 vs 59.4 right. These are population means from the all-window input decomposition,
not the non-airborne DNa02 command rate used for primaries. **Do not read this CSV's `rate_hz` as the
postsynaptic DNa02 rate**: `dna02_decompose` takes the first presynaptic row's rate (AN04B003 here).
The correctly named input totals and primary DNa02 rates are separate. This existing descriptive
column defect is recorded without changing a reducer during this documentation closeout.

`analysis/room_table.csv`, `<arm>_mean` / `<arm>_run_sd`, descriptive means +- SD:

| key | A | L3 | M2 | C |
|---|---|---|---|---|
| clean_frac | 0.9981 +- 0.0016 | 0.9223 +- 0.0208 | 0.9247 +- 0.0365 | 0.9314 +- 0.0262 |
| yaw_median_abs_clean_deg_s | 0.7381 +- 0.0101 | 1.5284 +- 0.0696 | 1.6073 +- 0.0895 | 1.5678 +- 0.0538 |
| yaw_sd_deg_s | 2.8741 +- 0.2654 | 99.7399 +- 10.8932 | 107.0523 +- 10.9172 | 95.3981 +- 22.7174 |
| hops | 0.0938 +- 0.0765 | 0.4375 +- 0.0968 | 0.5104 +- 0.2834 | 0.4375 +- 0.1896 |
| step_hz | nan +- nan | nan +- nan | 7.7856 +- 0.1588 | 7.7319 +- 0.0694 |
| stance_frac | nan +- nan | nan +- nan | 0.7664 +- 0.0048 | 0.7680 +- 0.0021 |

The large all-walking yaw SD includes edge/fall artifacts; the predeclared clean statistic is the
behavioural primary. Clean fractions differ modestly among active arms, not by the ~2x threshold
of INTERP 10.4 rule 23. These are the same 5-60 s windows; no all-window neural rate is substituted
for a masked command rate. No food-seeking claim follows from this controlled room comparison.

Fractions `(X-A)/(C-A)` from `analysis/fractions.csv`, mean +- SD over the six seed-paired ratios:

| primary | L3 | M2 |
|---|---|---|
| DNa02_L_hz | 0.7534 +- 0.0639 | 0.9867 +- 0.0550 |
| DNa02_R_hz | 0.8393 +- 0.1192 | 1.0227 +- 0.1144 |
| yaw_sd_clean_deg_s | 0.8859 +- 0.0542 | 1.0416 +- 0.0764 |
| straightness | 1.1568 +- 0.4997 | 1.1269 +- 0.6209 |
| DNa02_LR_hz | 0.6648 +- 0.2375 | 0.9572 +- 0.1647 |
| chain_AN04B003_L_hz | 0.8808 +- 0.0093 | 1.0083 +- 0.0237 |
| chain_AN04B003_R_hz | 0.8324 +- 0.0099 | 1.0066 +- 0.0246 |

Fractions above one and large straightness scatter are reported without clipping; they are not adoption gates.


## 9. Per-seed record

Verbatim generator output `out/vncd7/analysis/per_seed_lists.txt`. Primary and descriptive lists use
`analysis/per_seed.csv`, columns `key,arm,seed,file,value`; residuals use
`analysis/level_model_r2_residual_runs.csv`, `residual`; fraction lists use `analysis/fractions.csv`,
`per_seed`. Seeds are r0-r5 in that order; they are not sorted by value.
`commanded_<ch>_absLR_hz` is computed on the post-skip NON-AIRBORNE mask, while the pooled `commanded_<ch>_hz` and
the signed `commanded_<ch>_LR_hz` keys use post-skip frames only; the three do not share one window
(INTERP 10.4 rule 22).

```text
# per-seed lists pasted into docs/audits/level_fixed_point.md; source out\vncd7\analysis\per_seed.csv (columns key, arm, seed, file, value), seeds r0..r5 in seed order
# format: <key> <arm> [v(r0) v(r1) v(r2) v(r3) v(r4) v(r5)]  mean +- SD
DNa02_L_hz A [0.0158 0.0194 0.0136 0.0159 0.0113 0.0149]  0.0152 +- 0.0027
DNa02_L_hz L3 [0.3720 0.4450 0.4291 0.4125 0.4380 0.4355]  0.4220 +- 0.0269
DNa02_L_hz M2 [0.5449 0.5466 0.5574 0.5696 0.5289 0.5408]  0.5480 +- 0.0140
DNa02_L_hz C [0.5597 0.5176 0.5618 0.5519 0.5844 0.5621]  0.5563 +- 0.0218
DNa02_R_hz A [0.0831 0.0923 0.0761 0.0850 0.0778 0.0669]  0.0802 +- 0.0087
DNa02_R_hz L3 [0.3069 0.3184 0.3179 0.3456 0.3574 0.3799]  0.3377 +- 0.0281
DNa02_R_hz M2 [0.4286 0.4116 0.3837 0.3975 0.3498 0.3900]  0.3935 +- 0.0268
DNa02_R_hz C [0.3712 0.4012 0.4240 0.3710 0.3719 0.3895]  0.3881 +- 0.0215
yaw_sd_clean_deg_s A [2.7706 2.9058 2.6042 2.8826 2.6217 2.5823]  2.7279 +- 0.1450
yaw_sd_clean_deg_s L3 [6.9211 7.4744 7.0241 7.2656 7.3759 7.5874]  7.2747 +- 0.2592
yaw_sd_clean_deg_s M2 [8.3294 8.2215 7.9413 8.2813 7.7075 7.8955]  8.0627 +- 0.2502
yaw_sd_clean_deg_s C [7.7038 7.7065 8.1056 7.7977 7.8324 8.0534]  7.8666 +- 0.1733
straightness A [0.9955 0.9932 0.9945 0.9949 0.9951 0.9936]  0.9945 +- 0.0009
straightness L3 [0.8313 0.8297 0.8167 0.8689 0.7953 0.8432]  0.8309 +- 0.0247
straightness M2 [0.8792 0.8385 0.8067 0.8132 0.8097 0.8920]  0.8399 +- 0.0374
straightness C [0.8121 0.8370 0.9126 0.8540 0.7922 0.8352]  0.8405 +- 0.0414
DNa02_LR_hz A [-0.0673 -0.0729 -0.0625 -0.0691 -0.0665 -0.0520]  -0.0651 +- 0.0073
DNa02_LR_hz L3 [0.0651 0.1266 0.1112 0.0668 0.0806 0.0556]  0.0843 +- 0.0284
DNa02_LR_hz M2 [0.1163 0.1350 0.1737 0.1721 0.1791 0.1509]  0.1545 +- 0.0251
DNa02_LR_hz C [0.1886 0.1164 0.1378 0.1809 0.2124 0.1726]  0.1681 +- 0.0351
chain_AN04B003_L_hz A [0.5902 0.5947 0.5837 0.6106 0.5943 0.5917]  0.5942 +- 0.0090
chain_AN04B003_L_hz L3 [20.1663 20.2424 20.2742 20.2909 20.4140 20.3746]  20.2937 +- 0.0897
chain_AN04B003_L_hz M2 [23.7462 22.9625 22.8955 23.0985 22.5386 23.6341]  23.1459 +- 0.4618
chain_AN04B003_L_hz C [23.0030 23.0599 23.1288 22.5443 23.0538 22.9848]  22.9624 +- 0.2109
chain_AN04B003_R_hz A [0.3148 0.3205 0.3064 0.3201 0.3136 0.3098]  0.3142 +- 0.0055
chain_AN04B003_R_hz L3 [19.3027 19.5299 19.5451 19.6254 19.6879 19.6814]  19.5621 +- 0.1433
chain_AN04B003_R_hz M2 [24.1826 23.3985 23.3617 23.5659 22.9375 24.1004]  23.5911 +- 0.4747
chain_AN04B003_R_hz C [23.4011 23.4750 23.6288 23.0189 23.5970 23.5205]  23.4402 +- 0.2222
dna02LR_given_chordLR_pos_minus_neg L3 [0.0189 0.0159 0.0070 -0.0004 -0.0146 0.0175]  0.0074 +- 0.0131
dna02LR_given_chordLR_pos_minus_neg M2 [-0.1158 -0.0926 -0.0912 -0.0883 -0.0605 -0.0823]  -0.0884 +- 0.0179
dna02LR_given_chordLR_pos_minus_neg C [-0.3370 -0.3361 -0.3893 -0.3351 -0.3593 -0.3690]  -0.3543 +- 0.0222
corr_dna02LR_chordLR L3 [0.0147 0.0118 0.0070 0.0116 -0.0010 0.0042]  0.0081 +- 0.0058
corr_dna02LR_chordLR M2 [-0.0323 -0.0328 -0.0315 -0.0301 -0.0280 -0.0275]  -0.0303 +- 0.0022
corr_dna02LR_chordLR C [-0.1421 -0.1414 -0.1497 -0.1429 -0.1444 -0.1488]  -0.1449 +- 0.0036
an04LR_given_chordLR_pos_minus_neg L3 [-0.0043 -0.0675 -0.0943 -0.1190 -0.1046 0.0841]  -0.0509 +- 0.0776
an04LR_given_chordLR_pos_minus_neg M2 [-3.6880 -3.6230 -3.5954 -3.6305 -3.5583 -3.6777]  -3.6288 +- 0.0490
an04LR_given_chordLR_pos_minus_neg C [-3.6110 -3.5853 -3.6472 -3.6071 -3.6493 -3.6151]  -3.6192 +- 0.0248
corr_an04LR_chordLR L3 [-0.0061 -0.0084 -0.0080 -0.0137 -0.0180 -0.0037]  -0.0097 +- 0.0053
corr_an04LR_chordLR M2 [-0.3597 -0.3555 -0.3526 -0.3600 -0.3528 -0.3587]  -0.3565 +- 0.0034
corr_an04LR_chordLR C [-0.3482 -0.3419 -0.3494 -0.3480 -0.3472 -0.3440]  -0.3464 +- 0.0029
yaw_signed_mean_deg_s A [0.1792 0.1478 0.1929 0.1416 0.1832 0.2414]  0.1810 +- 0.0359
yaw_signed_mean_deg_s L3 [1.1238 1.4104 1.2474 1.1864 1.2861 1.1377]  1.2320 +- 0.1074
yaw_signed_mean_deg_s M2 [0.9782 1.0982 1.2760 1.2467 1.1047 1.2085]  1.1521 +- 0.1122
yaw_signed_mean_deg_s C [1.2311 1.0128 1.0699 1.2476 1.4531 1.2532]  1.2113 +- 0.1558
commanded_chordotonal_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_chordotonal_hz L3 [86.6075 87.2632 87.3789 87.6323 87.9101 87.8734]  87.4442 +- 0.4845
commanded_chordotonal_hz M2 [89.6747 86.8375 86.5028 87.3416 85.0577 89.1203]  87.4224 +- 1.7171
commanded_chordotonal_hz C [86.9160 87.1411 87.5521 85.4219 87.1482 86.9784]  86.8596 +- 0.7384
commanded_hair_plate_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_hair_plate_hz L3 [46.6362 46.9925 47.0554 47.1931 47.3441 47.3242]  47.0909 +- 0.2633
commanded_hair_plate_hz M2 [48.2886 46.7554 46.5900 47.0385 45.7823 47.9898]  47.0741 +- 0.9295
commanded_hair_plate_hz C [46.7999 46.9316 47.1411 45.9915 46.9360 46.8228]  46.7705 +- 0.4003
commanded_campaniform_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_campaniform_hz L3 [24.9802 24.9582 24.9311 24.9813 25.0110 25.0056]  24.9779 +- 0.0298
commanded_campaniform_hz M2 [24.9270 24.8264 24.8514 24.8151 24.8259 24.9747]  24.8701 +- 0.0655
commanded_campaniform_hz C [24.8474 24.9514 24.9298 24.8457 24.9230 24.7983]  24.8826 +- 0.0605
commanded_chordotonal_absLR_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_chordotonal_absLR_hz L3 [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_chordotonal_absLR_hz M2 [12.6193 12.2207 12.1584 12.2975 11.9364 12.5039]  12.2893 +- 0.2455
commanded_chordotonal_absLR_hz C [12.2846 12.2728 12.3545 12.0541 12.2991 12.3114]  12.2628 +- 0.1060
commanded_chordotonal_LR_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_chordotonal_LR_hz L3 [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_chordotonal_LR_hz M2 [0.0438 0.0296 0.0176 0.0373 0.0311 0.0261]  0.0309 +- 0.0090
commanded_chordotonal_LR_hz C [-0.3487 -0.2330 -0.2961 -0.3276 -0.3815 -0.3304]  -0.3195 +- 0.0508
chain_IN13B001_L_hz A [10.6587 10.6341 10.5773 10.6394 10.7220 10.6504]  10.6470 +- 0.0466
chain_IN13B001_L_hz L3 [65.9996 66.5027 66.6038 66.8557 66.9489 66.8693]  66.6300 +- 0.3533
chain_IN13B001_L_hz M2 [67.7955 65.7329 65.5223 66.0447 64.3928 67.2398]  66.1213 +- 1.2294
chain_IN13B001_L_hz C [65.8928 66.0349 66.3731 64.9034 66.0879 65.8538]  65.8576 +- 0.5024
chain_IN13B001_R_hz A [6.5023 6.5561 6.5205 6.5083 6.5152 6.4333]  6.5059 +- 0.0402
chain_IN13B001_R_hz L3 [63.5777 64.4026 64.4242 64.5477 64.6727 64.7693]  64.3991 +- 0.4265
chain_IN13B001_R_hz M2 [66.3098 63.9261 63.6364 64.3447 62.3159 65.6102]  64.3572 +- 1.4322
chain_IN13B001_R_hz C [63.8648 64.0386 64.3735 62.5883 63.9307 63.8580]  63.7756 +- 0.6124
chain_SNpp45_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
chain_SNpp45_hz L3 [45.7621 46.1259 46.1952 46.2948 46.4615 46.3440]  46.1973 +- 0.2430
chain_SNpp45_hz M2 [47.3898 45.8692 45.6895 46.1549 44.9042 47.0152]  46.1704 +- 0.9084
chain_SNpp45_hz C [45.9141 46.0416 46.2256 45.1035 46.0472 45.8365]  45.8614 +- 0.3943
commanded_chordotonal:L_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_chordotonal:L_hz L3 [86.6075 87.2632 87.3789 87.6323 87.9101 87.8734]  87.4442 +- 0.4845
commanded_chordotonal:L_hz M2 [89.6966 86.8524 86.5116 87.3602 85.0733 89.1334]  87.4379 +- 1.7190
commanded_chordotonal:L_hz C [86.7455 87.0272 87.4073 85.2616 86.9616 86.8170]  86.7034 +- 0.7430
commanded_chordotonal:R_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_chordotonal:R_hz L3 [86.6075 87.2632 87.3789 87.6323 87.9101 87.8734]  87.4442 +- 0.4845
commanded_chordotonal:R_hz M2 [89.6528 86.8227 86.4941 87.3230 85.0422 89.1073]  87.4070 +- 1.7152
commanded_chordotonal:R_hz C [87.0942 87.2602 87.7034 85.5892 87.3431 87.1474]  87.0229 +- 0.7346
commanded_hair_plate:L_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_hair_plate:L_hz L3 [46.6362 46.9925 47.0554 47.1931 47.3442 47.3242]  47.0909 +- 0.2633
commanded_hair_plate:L_hz M2 [48.2920 46.7613 46.5896 47.0404 45.7857 47.9979]  47.0778 +- 0.9306
commanded_hair_plate:L_hz C [46.6968 46.8626 47.0618 45.9002 46.8276 46.7298]  46.6798 +- 0.4029
commanded_hair_plate:R_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_hair_plate:R_hz L3 [46.6362 46.9925 47.0554 47.1931 47.3441 47.3242]  47.0909 +- 0.2633
commanded_hair_plate:R_hz M2 [48.2849 46.7490 46.5904 47.0365 45.7786 47.9812]  47.0701 +- 0.9282
commanded_hair_plate:R_hz C [46.9086 47.0041 47.2244 46.0875 47.0501 46.9207]  46.8659 +- 0.3981
commanded_hair_plate_LR_hz A [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_hair_plate_LR_hz L3 [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
commanded_hair_plate_LR_hz M2 [0.0071 0.0123 -0.0008 0.0038 0.0071 0.0167]  0.0077 +- 0.0062
commanded_hair_plate_LR_hz C [-0.2117 -0.1415 -0.1626 -0.1873 -0.2225 -0.1909]  -0.1861 +- 0.0302
leg_L_hz A [2.5442 2.5427 2.5319 2.5523 2.5501 2.5280]  2.5415 +- 0.0097
leg_L_hz L3 [4.6745 4.7119 4.7255 4.7333 4.7450 4.7456]  4.7226 +- 0.0268
leg_L_hz M2 [4.8899 4.7680 4.7717 4.8034 4.6845 4.8435]  4.7935 +- 0.0706
leg_L_hz C [4.7788 4.7835 4.7978 4.7263 4.7798 4.7677]  4.7723 +- 0.0245
leg_R_hz A [2.3881 2.3920 2.3792 2.4061 2.3937 2.3743]  2.3889 +- 0.0113
leg_R_hz L3 [4.3879 4.4382 4.4512 4.4527 4.4644 4.4601]  4.4424 +- 0.0282
leg_R_hz M2 [4.7430 4.6183 4.6105 4.6475 4.5432 4.6988]  4.6436 +- 0.0704
leg_R_hz C [4.6368 4.6428 4.6501 4.5851 4.6408 4.6290]  4.6308 +- 0.0234
leg_LR_hz A [0.1561 0.1507 0.1526 0.1462 0.1564 0.1537]  0.1526 +- 0.0038
leg_LR_hz L3 [0.2865 0.2737 0.2743 0.2806 0.2805 0.2855]  0.2802 +- 0.0054
leg_LR_hz M2 [0.1469 0.1497 0.1612 0.1559 0.1413 0.1447]  0.1500 +- 0.0074
leg_LR_hz C [0.1420 0.1407 0.1477 0.1412 0.1391 0.1388]  0.1416 +- 0.0032
n_left_table A [0.0000 0.0000 0.0000 0.0000 1.0000 0.0000]  0.1667 +- 0.4082
n_left_table L3 [9.0000 13.0000 10.0000 8.0000 10.0000 7.0000]  9.5000 +- 2.0736
n_left_table M2 [8.0000 8.0000 12.0000 11.0000 9.0000 7.0000]  9.1667 +- 1.9408
n_left_table C [9.0000 9.0000 8.0000 8.0000 8.0000 7.0000]  8.1667 +- 0.7528
hops A [0.0625 0.1875 0.0625 0.1875 0.0625 0.0000]  0.0938 +- 0.0765
hops L3 [0.5000 0.5000 0.5625 0.3750 0.3750 0.3125]  0.4375 +- 0.0968
hops M2 [0.2500 0.6875 0.6250 0.7500 0.6875 0.0625]  0.5104 +- 0.2834
hops C [0.5625 0.2500 0.3125 0.5625 0.2500 0.6875]  0.4375 +- 0.1896
clean_frac A [0.9987 0.9962 0.9987 0.9961 0.9986 1.0000]  0.9981 +- 0.0016
clean_frac L3 [0.9134 0.9081 0.9240 0.9574 0.8988 0.9318]  0.9223 +- 0.0208
clean_frac M2 [0.9657 0.9240 0.8981 0.8885 0.8988 0.9728]  0.9247 +- 0.0365
clean_frac C [0.9048 0.9254 0.9812 0.9347 0.9220 0.9203]  0.9314 +- 0.0262
gf_max_hz A [24.2491 25.1630 25.2024 26.4610 25.9629 24.5500]  25.2647 +- 0.8337
gf_max_hz L3 [32.9348 31.6056 33.4093 28.4088 31.1982 28.6741]  31.0385 +- 2.1009
gf_max_hz M2 [26.8320 33.1530 33.1486 32.4652 33.2029 25.2694]  30.6785 +- 3.6289
gf_max_hz C [33.9134 26.5294 24.2420 29.8627 30.4398 30.0766]  29.1773 +- 3.3667
abs_amp_LR M2 [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]  0.0000 +- 0.0000
abs_amp_LR C [0.0161 0.0161 0.0168 0.0160 0.0166 0.0163]  0.0163 +- 0.0003
step_hz M2 [7.9994 7.7409 7.6930 7.7895 7.5631 7.9278]  7.7856 +- 0.1588
step_hz C [7.7488 7.7355 7.7915 7.5965 7.7457 7.7736]  7.7319 +- 0.0694
stance_frac M2 [0.7600 0.7678 0.7692 0.7663 0.7731 0.7622]  0.7664 +- 0.0048
stance_frac C [0.7675 0.7679 0.7663 0.7721 0.7676 0.7668]  0.7680 +- 0.0021
DNa02_active_frac A [0.0069 0.0079 0.0063 0.0072 0.0062 0.0058]  0.0067 +- 0.0008
DNa02_active_frac L3 [0.0481 0.0546 0.0529 0.0541 0.0563 0.0580]  0.0540 +- 0.0034
DNa02_active_frac M2 [0.0691 0.0684 0.0665 0.0690 0.0624 0.0661]  0.0669 +- 0.0025
DNa02_active_frac C [0.0662 0.0652 0.0703 0.0653 0.0677 0.0679]  0.0671 +- 0.0019
yaw_median_abs_clean_deg_s A [0.7495 0.7461 0.7239 0.7401 0.7409 0.7281]  0.7381 +- 0.0101
yaw_median_abs_clean_deg_s L3 [1.4042 1.5239 1.5240 1.5662 1.5382 1.6137]  1.5284 +- 0.0696
yaw_median_abs_clean_deg_s M2 [1.6775 1.6860 1.6218 1.5931 1.4395 1.6260]  1.6073 +- 0.0895
yaw_median_abs_clean_deg_s C [1.5309 1.5196 1.6250 1.5077 1.6090 1.6145]  1.5678 +- 0.0538
# level-model per-run residuals; source out\vncd7\analysis\level_model_r2_residual_runs.csv (column residual), seeds in seed order
residual_AN04B003 L3L [+1.2006 +1.1786 +1.1931 +1.1719 +1.2534 +1.2195]  +1.2028 +- 0.0299
residual_AN04B003 L3R [+1.1002 +1.2293 +1.2272 +1.2696 +1.2905 +1.2895]  +1.2344 +- 0.0714
residual_AN04B003 M2L [+4.3167 +3.9597 +3.9445 +4.0198 +3.8015 +4.2897]  +4.0553 +- 0.2051
residual_AN04B003 M2R [+5.5240 +5.1635 +5.1773 +5.2571 +4.9689 +5.5229]  +5.2689 +- 0.2187
residual_AN04B003 CL [+4.0158 +4.0312 +4.0428 +3.7798 +4.0350 +3.9865]  +3.9819 +- 0.1010
residual_AN04B003 CR [+5.1263 +5.1757 +5.2618 +4.9692 +5.2853 +5.2366]  +5.1758 +- 0.1168
residual_DNa02_cmd L3L [-0.0034 +0.0675 +0.0513 +0.0339 +0.0586 +0.0562]  +0.0440 +- 0.0258
residual_DNa02_cmd L3R [+0.0266 +0.0361 +0.0353 +0.0622 +0.0731 +0.0957]  +0.0548 +- 0.0268
residual_DNa02_cmd M2L [+0.1599 +0.1704 +0.1823 +0.1918 +0.1583 +0.1576]  +0.1701 +- 0.0143
residual_DNa02_cmd M2R [+0.1389 +0.1307 +0.1037 +0.1150 +0.0744 +0.1020]  +0.1108 +- 0.0230
residual_DNa02_cmd CL [+0.1839 +0.1409 +0.1839 +0.1806 +0.2079 +0.1861]  +0.1805 +- 0.0218
residual_DNa02_cmd CR [+0.0894 +0.1189 +0.1404 +0.0939 +0.0894 +0.1076]  +0.1066 +- 0.0202
fraction_DNa02_L_hz L3 [0.6549 0.8542 0.7579 0.7399 0.7446 0.7686]  0.7534 +- 0.0639
fraction_DNa02_L_hz M2 [0.9727 1.0582 0.9920 1.0330 0.9032 0.9611]  0.9867 +- 0.0550
fraction_DNa02_R_hz L3 [0.7769 0.7319 0.6950 0.9113 0.9507 0.9703]  0.8393 +- 0.1192
fraction_DNa02_R_hz M2 [1.1995 1.0338 0.8841 1.0925 0.9248 1.0014]  1.0227 +- 0.1144
fraction_yaw_sd_clean_deg_s L3 [0.8413 0.9516 0.8034 0.8917 0.9124 0.9148]  0.8859 +- 0.0542
fraction_yaw_sd_clean_deg_s M2 [1.1268 1.1073 0.9701 1.0984 0.9760 0.9711]  1.0416 +- 0.0764
fraction_straightness L3 [0.8953 1.0467 2.1700 0.8948 0.9844 0.9494]  1.1568 +- 0.4997
fraction_straightness M2 [0.6343 0.9903 2.2924 1.2898 0.9138 0.6410]  1.1269 +- 0.6209
fraction_DNa02_LR_hz L3 [0.5175 1.0538 0.8673 0.5438 0.5273 0.4789]  0.6648 +- 0.2375
fraction_DNa02_LR_hz M2 [0.7174 1.0979 1.1795 0.9649 0.8803 0.9033]  0.9572 +- 0.1647
fraction_chain_AN04B003_L_hz L3 [0.8734 0.8746 0.8734 0.8973 0.8825 0.8834]  0.8808 +- 0.0093
fraction_chain_AN04B003_L_hz M2 [1.0332 0.9957 0.9897 1.0253 0.9771 1.0290]  1.0083 +- 0.0237
fraction_chain_AN04B003_R_hz L3 [0.8225 0.8296 0.8249 0.8505 0.8321 0.8346]  0.8324 +- 0.0099
fraction_chain_AN04B003_R_hz M2 [1.0338 0.9967 0.9885 1.0241 0.9717 1.0250]  1.0066 +- 0.0246
fraction_DNa02_active_frac L3 [0.6947 0.8144 0.7284 0.8076 0.8153 0.8406]  0.7835 +- 0.0579
fraction_DNa02_active_frac M2 [1.0478 1.0555 0.9415 1.0642 0.9150 0.9715]  0.9993 +- 0.0647
```


## 10. Files, provenance and reproduction

24 run JSONs and their recorded companion artifacts, all CUDA / NVIDIA B200, family level4,
six seeds per A/L3/M2/C, blocks fam_r0-r5. The source fingerprint's 52-file maps agree in every run;
compiled MaleCNS MD5 is `ef23cc27bea13be7f6a96f3c04fd3737`. Run headers have unknown Git commit in the
cluster copy, so the file map, not that field, identifies the simulation source. The 52-file map covers
`flyverse/**` plus `scripts/interp_export.py` and `scripts/probe_object_sweep.py`; it does NOT include
`scripts/probe_vnc_drive.py`, the script every job ran, so the driver is identified by the verbatim job lines in
`out/vncd7_cluster.log` and by the arm / spec / sense block each run JSON records, not by the fingerprint.
One submission, 24 completed jobs / zero failed (`out/vncd7_cluster.log`); the record is the client console, and
all 24 jobs are logged completed exit None, so no exit code survives; the 24 run JSONs with their `finished_utc`,
five artefacts each and `device cuda` lines are what establishes completion (INTERP 10.4 items 4 and 21).
First run starts 19:22:56Z and last finishes
19:34:40Z. Predeclaration 19:22:14Z and submission receipt 19:22:25Z precede them. The archived
predeclaration is preserved; no arms, tolerances or decisions are changed at closeout.

**Source-stamp limit.** The original header promised `tree_state_check.json`, but that file did not
exist. Closeout writes `out/vncd7/tree_state_check_astra.json` instead of fabricating an earlier check.
The source/analysis differences against the 19:22:14Z tree stamp are:

| file | stamped SHA-256 | closeout SHA-256 |
|---|---|---|
| scripts/probe_vnc_drive.py | a3530afd49c9b5521bec6a25c4805e70c783510a711ff09b67e2566c8d01d8d6 | 4202f34b07459bd69741e154b03d2a47904fdd62a08e1839891403f0ab12658b |
| scripts/derive_level_fixed_point.py | f2b615832aa6b5b6722f6f3974165001847b8d07cc08db07c179eef50ec12ccd | 8071f2db8e46da3a869ea3e3203cbe1b037ca1d383bb8980091ef5aadb70b10c |
| out/vncd7/batch.sh | 58611cc0ba8073d4eb5b898a00fb132aef93d5908f2d2746f108ab944f0509f0 | 3990a045f777b2984a85fd4c0e68b8bbe023b12cb50ac656596da1998f056519 |

Checked at f732f75. Two of the three digests differ only by line endings: the stamp hashes raw working-tree
bytes (CRLF), and the sha256 of the CRLF rendering of `scripts/probe_vnc_drive.py` and
`scripts/derive_level_fixed_point.py` as committed is exactly the stamped `a3530afd...` / `f2b61583...`, so their
CONTENT is unchanged since the stamp (`git diff ad2efc0 HEAD` on both is empty). Only `out/vncd7/batch.sh` is a
content difference, and its 24 quoted job lines are identical to the 24 commands the scheduler recorded in
`out/vncd7_cluster.log`, so nothing about what ran is in doubt. On main e59f5a4 two further stamped files now
differ -- `flyverse/senses.py` and `flyverse/interp/common.py`, both predeclared reducers -- because of later
rounds; the simulated `senses.py` is the one committed at ad2efc0 (`c140067f...`, the run fingerprint's own hash).
**Withdrawn:** "This does not certify when the changed files were edited." Two of the three digests are
line-ending artefacts rather than edits, and the third is checkable against the scheduler's own job lines, so the
sentence understated what the stamp establishes. Its companion sentence stands: the rest of the 14-entry stamp
matches.
The complete fetched batch was reanalysed with main f732f75: `probe_vnc_drive.py analyse`, `pairs
--family level4 --predeclared ...`, the recorded `verify_runs.py`, `level_model.py`, and `per_seed_lists.py`.
Logs are `out/compass7/4d_{verify,analyse,pairs,level_model,per_seed}.log`, not retained in the repository.
The committed primary generator is `scripts/probe_vnc_drive.py`; the three auxiliary generators remain
with the local batch artifacts as named in the original predeclaration. Analysis outputs were re-emitted
in full. The current reducer hashes above describe this reconstruction; they are not substituted into
the frozen predeclaration. The reconstruction is CPU-only; it does not re-run or alter the GPU experiment.


## 11. Validation and self-review

CPU-only full suite on the unchanged runtime: **441 passed / 19 skipped, 215 subtests**, including
`tests/test_bit_identity.py`. MaleCNS cache MD5s unchanged: neurons `c50c598a708b5b373cbaffca7d6a9d82`,
W `ac131529cebf98decde58d0c227b7954`, sign-0 counts `bf01d724acf2a1fec8fdb60ef8a9e066`.
The original 38-test focused validation in section 2 remains recorded separately.

**Astra self-review, not the independent pass:** re-opened all 24 runs, re-ran the configuration and
matching checks and all analysis stages; tables and per-seed lists above are emitted from their CSVs.
Checked the masks in the primary reducer, satisfiable per-family Holm correction, and the matched
versus cross-batch distinction. Disclosed the incomplete WIP text, source-stamp mismatch and misleading
descriptive rate column. No parameter, mechanism, ledger expectation or default is changed or adopted.

**Independent skeptic pass (Opus, 2026-09-17): mostly sound.** All seven claims reproduced; the corrections it
required are applied above and no conclusion moved. Its verdict line and claim lines are quoted verbatim in
"Skeptic pass (independent, Opus, 2026-09-17)" at the end of this audit.


## Report

```yaml
summary: |-
  **The fixed point landed, and the cycle retains a matched structure term.** All three leg channels
  meet the declared per-side tolerances. C minus L3 raises AN04B003 by **+2.6687 Hz left / +3.8782 Hz right**
  (both `result`, Holm p 0.0152), a direct pooled difference of **+3.2734 Hz**. DNa02-left is `result`
  (+0.1342 Hz, z 4.996); DNa02-right (+0.0504 Hz, z 1.794), clean yaw SD (+0.5918 deg/s, z 2.284),
  straightness and DNa02 L-R are `null`. A positive mean difference alone does not pass the declared rule.

  The modulation-only M2 arm also exceeds L3 at the relay (**+2.8521 / +4.0290 Hz**, pooled **+3.4406 Hz**),
  DNa02-left (+0.1260 Hz) and clean yaw SD (+0.7880 deg/s); those four primaries are `result`.
  **M2 versus C is `null` on all seven primaries**, repeating round 4c's outcome: no extra drive from the
  amplitude/turn law is detected at this level. This is a null test, not an equivalence bound. The declared
  sided secondary still separates their DNa02 response: tripod-conditioned L-R swing **-0.3543 Hz C /
  -0.0884 Hz M2**, with L3 +0.0074 (numerical residue of a nominally unsided input, not a physiological signal).

  The direct matched term is smaller than the earlier slope-corrected, cross-batch +4.49 / +4.60 / +4.72 Hz
  context. The old level model leaves **+1.2028 / +1.2344 Hz** residual in the modulation-free L3 control
  at the cycle's level point, showing the limitation of its extrapolation. Correcting the small remaining
  C-L3 level mismatch gives **+2.7790 / +3.9415 Hz**, pooled **+3.3602 Hz**, as a declared secondary;
  the primary conclusion uses the direct matched difference, not that correction.

  This closes round 4d's matching question on **24 CUDA runs, six per arm, one house submission**.
  All preconditions pass; runtime, defaults and MaleCNS cache are unchanged, and nothing is adopted.
  **Independent skeptic pass (Opus, 2026-09-17): mostly sound** -- all seven claims reproduced, the corrections
  it required are applied here, and no conclusion moved ("Skeptic pass" below). The handoff overstated the saved prose:
  sections 4-11 and Report were also placeholders at ad2efc0. This closeout reconstructs them from the
  completed recordings and regenerated analysis. Source-stamp limitations and a misleading descriptive
  rate column are disclosed in sections 8 and 10; neither is used to infer a primary result.
key_claims:
- P1 passes; both relay sides are positive results. The pooled matched term is +3.2734 Hz. The DNa02 result is left-only;
  right and all three behavioural/sided primaries are null.
- P1/P2 pass; both relay sides and DNa02-left are positive results. The pooled modulation-only matched term is +3.4406
  Hz; right DNa02 is null. The yaw result is reported without assigning a mechanism to it.
- P2 passes; all seven primaries are null, including all four named rate rows. Round 4c's null replicates under
  the declared rule; this does not establish zero effect or equivalence.
- All seven are results. Fractions are descriptive decompositions of this within-batch treatment difference, not
  new tests.
validation:
- 24 CUDA runs, all declared preconditions pass; four primary families m=7 at six versus six runs.
- Full CPU suite 441 passed, 19 skipped, 215 subtests; golden and cache hashes unchanged.
- Independent skeptic pass (Opus, 2026-09-17): mostly sound; all seven claims reproduced, corrections applied,
  no conclusion moved.
recommendations:
- Adopt nothing from these labelled controls.
- Use the direct matched difference for the structure term; preserve the extrapolation failure as a limit on the
  earlier model.
- Keep DNa02-right and C-L3 yaw as null; larger reference-arm precision requires a new predeclared experiment.
- Fix the descriptive dna02_decompose rate_hz label separately, regenerating its affected outputs.
open_questions:
- The independent skeptic pass is complete (mostly sound, 2026-09-17); what stays unchecked is its NOT CHECKED list.
- Biological calibration and suite/room adoption gates for the leg-cycle mechanism remain open.
```


## Skeptic pass (independent, Opus, 2026-09-17)

An independent skeptic pass ran on 2026-09-17 (Opus, CPU only, no cluster job, nothing adopted). Its verdict line
and its claim lines are quoted verbatim below. The CORRECTIONS REQUIRED list is applied in place in the sections
above; where a correction replaced a sentence that stated a finding, the original sentence stays in the record
marked **Withdrawn:** (INTERP 10.4 rule 29 iii). The pass's NOT CHECKED list is recorded verbatim with this
round's entry in [receptor_verification.md](receptor_verification.md).

### Verdict

```text
VERDICT: mostly sound
```

### Claims

```text
CLAIMS:
1. Fixed-point derivation -- REPRODUCED. Targets recomputed from out/vncd6/room_C_r*_body.npz: chord 87.40418 (L 87.22939 / R 87.58664), hair 47.06689, camp 24.86421 -- identical to out/vncd7/fixed_point_derivation.json targets. Algebra: D* 0.5528870, hair_plate_max_hz = 5+(47.06689-5)/D* = 81.08587 -> 81.09, campaniform_load_hz = 24.86421/0.9907822 = 25.09553 -> 25.10. Re-solved the loop on out/vncd5 arm U's 440,000-sample full-window leg-MN distribution: LS slope dg/dmn_ref -0.0832071, g(8.23)=0.9635293, brentq gives mn_ref = 8.2300430 -> 8.23. Realised match from the 24 runs: L3 87.4442/47.0909/24.9779 vs C 86.7034-87.0229/46.6798-46.8659/24.8783-24.8870, per-side gaps +0.7408/+0.4213, +0.4111/+0.2250, +0.0996/+0.0909 against tolerances 3/3/1.5 -- P1 passes. Stamp order holds: all seven cal runs are device: cpu, finish by 19:21:47Z, derivation stamped 19:21:59Z, predeclaration 19:22:14Z, earliest room run 19:22:56Z.
2. Primaries -- REPRODUCED, all 28 rows exactly (analysis/decision_table.csv). compare 6 v 6, floor 0.0021645022, Holm m=7 per declared family. C-L3 AN04B003 +2.6687 (z 29.7355) / +3.8782 (z 27.0592), pooled +3.2734; DNa02_L +0.1342 z 4.9959 result; DNa02_R +0.0504 z 1.7945 null; yaw +0.5918 z 2.2837 null; straightness +0.0097 null; DNa02_LR +0.0838 z 2.9530 null. M2-L3 +2.8521/+4.0290, pooled +3.4406, DNa02_L +0.1260, yaw +0.7880 -> 4/7 called. M2 v C null on all seven. F4 7/7. m as predeclared (7 keys, m=7 in all four families) and satisfiable (7x0.0021645 = 0.0151515 <= 0.05).
3. Per-seed lists -- REPRODUCED. The audit's text block is a zero-line diff against out/vncd7/analysis/per_seed_lists.txt (169 lines). 126 lists recomputed from the run files themselves: zero mismatches. The 12 residual_* lists match level_model_console.txt line for line.
4. Predeclaration / stamps -- REPRODUCED in substance, but the audit's provenance prose is wrong in both directions. Archive byte-identical (8a3d2f46628a3be2e423d27aa18235d2c527866321128f457cc7cd5c0e2eff20); stamp 19:22:14Z precedes the earliest started_utc 19:22:56Z; no measure, key, family, m, tolerance or decision rule differs from predeclared.json. But: (a) the header's submission receipt time 19:22:20Z is wrong -- the file says 19:22:25Z; (b) two of the three "source-stamp" differences are line-ending artifacts, not edits -- sha256 of the CRLF rendering of today's scripts/probe_vnc_drive.py is exactly the stamped a3530afd..., ditto derive_level_fixed_point.py/f2b61583..., and git diff ad2efc0 HEAD on both is empty; only batch.sh is a real content difference, and its 24 quoted job lines are set-identical to the 24 commands in out/vncd7_cluster.log; (c) two MORE stamped files now differ and the audit does not say so -- flyverse/senses.py (6b134bfa->6718e72a) and flyverse/interp/common.py (9e75c838->18e8eff9), both predeclared reducers, senses.py on the simulated path; they matched at f732f75 and were changed by the later round-7/navigation work; (d) scripts/probe_vnc_drive.py -- the driver of all 24 jobs -- is not one of the 52 files in provenance.source_fingerprint; (e) the logs cited in sections 4 and 10 (out/compass7/4d_*.log) are not in the repository.
5. dna02_decompose_summary.csv rate_hz -- REPRODUCED. Its values are exactly chain_AN04B003_L_hz / _R_hz, not DNa02's rates. No conclusion depends on it: the only two things section 8 takes from the file are the AN04B003 input totals (+41.222, +58.649) and SNpp45 (128.2 vs 129.3 left, 59.1 vs 59.4 right), both verified; every primary, fraction and level-model number comes from per_seed.csv / the npz.
6. Section 0 / Report consistency -- REPRODUCED. Section 0 Answer == Report.summary verbatim; key_claims 1-4 == section 6's four "Reading:" paragraphs verbatim. Every prose number traces to a table. Vocabulary clean. No finding asserted that the tables do not support. No unfilled placeholder remains.
7. Astra's reconstruction changed no number -- REPRODUCED. All 28 decision rows, all slopes (b +0.18485+-0.03548, c -0.06475+-0.02851, RMSE 0.1279), the L3 residuals +1.2028/+1.2344, every row of section 7's correction table, and all 23 precondition lines are already in Fable's consoles with the audit's exact values. git diff ad2efc0 HEAD on the audit removes only the five PENDING_* placeholders and two header lines. The commit ad2efc0 really did carry PENDING_SECTION_0, PENDING_SECTIONS_4_TO_9, PENDING_SECTION_10, PENDING_SECTION_11, PENDING_REPORT, so that commit's message ("the Report block is complete") was false and it breached rule 7's placeholder gate.
```
