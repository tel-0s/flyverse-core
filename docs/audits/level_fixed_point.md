# The level control matched on all three leg channels: the cycle's structure term as a direct matched difference (thread level controls, round 4d)

Generator: `scripts/probe_vnc_drive.py --family level4` (`plan` / `analyse` / `pairs` CPU; `room` GPU), the CPU derivation
`scripts/derive_level_fixed_point.py` (-> `out/vncd7/fixed_point_derivation.json`, console
`out/vncd7/fixed_point_derivation_console.txt`), the CPU checks `out/vncd7/verify_runs.py`, `out/vncd7/level_model.py`
(declared secondary) and `out/vncd7/per_seed_lists.py`. Data: `out/vncd7/` (ONE submission, `out/vncd7/batch.sh` -> run
**vncd7-9bdd12** on the house cluster, B200, console `out/vncd7_cluster.log`: **24 job(s), 0 failed (18.5 min)**); analysis
`out/vncd7/analysis_console.txt`, `out/vncd7/analysis/pairs_console.txt`, `out/vncd7/verification_console.txt`,
`out/vncd7/level_model_console.txt`. Predeclaration `out/vncd7/predeclared.json`, stamped **2026-09-15T19:22:14Z** (archived
unchanged as `out/vncd7/predeclared_archive/predeclared_2026-09-15T192214Z.json`) against the earliest run's own
`started_utc` **19:22:56Z**; submission receipt `out/vncd7/submitted_at.txt` (19:22:20Z). Working-tree record
`out/vncd7/tree_state.json` (sha256 of every shipped file; `tree_state_check.json` after the analysis, section 10).
Every number below is a run mean +- SD over the SIX runs of an arm (16 flies x 55 s window each; runs are the replicate
unit); verdicts are `flyverse.interp.common.compare` on runs (6 v 6, exact-U floor p 0.0021645). **Every per-seed list in
this document is pasted from `out/vncd7/analysis/per_seed_lists.txt`, which `per_seed_lists.py` emits from
`out/vncd7/analysis/per_seed.csv` (columns `key, arm, seed, file, value`, written by `pairs`) and from
`analysis/level_model_r2_residual_runs.csv`; the file and column are named beside each list** (docs/INTERP.md 10.4 item
28). Nothing in `out/vncd7` is compared row by row with `out/vncd6` or `out/vncd5`; their numbers appear as labelled context,
as the derivation's targets and anchor, and as the level model's calibration.

PENDING_SECTION_0

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
U 5.0 -> 4.1 Hz per-second means) and the flies leave the table after ~45 s (on-table fraction 0.59-0.82 at 45-55 s), so the
5-60 s window mean sits BELOW the 2-12 s window a 12-s calibration run realises by a **window factor W = (chord_full -
10) / (chord_early - 10)** of **0.9648 (vncd5 L), 0.9550 (U), 0.9469 (K), 0.9710 (vncd6 L): 0.9594 +- 0.0106**
(`fixed_point_derivation.json` `steady_gpu_arms`). The CPU calibration window matches the GPU's EARLY window, not its
full one: vncd5's cal_L 86.97 sits against L's 88.3 early / 86.0 full, cal_K2 80.99 against K's 81.1 early / 77.4 full.

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

PENDING_SECTIONS_4_TO_9

## 10. Files, provenance and reproduction

PENDING_SECTION_10

## 11. Validation (CPU, `CUDA_VISIBLE_DEVICES=-1`; this desktop)

PENDING_SECTION_11

## Report

PENDING_REPORT
