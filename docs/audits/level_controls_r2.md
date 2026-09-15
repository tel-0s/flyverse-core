# The modulation-only arm at the cycle's realised amplitude: does the amplitude / turn law add drive? (thread level controls, round 4c)

Generator: `scripts/probe_vnc_drive.py --family level3` (`plan` / `analyse` / `pairs` CPU; `room` GPU), the CPU derivation
`out/vncd6/derive_flat_amplitude.py`, the CPU checks `out/vncd6/verify_runs.py`, `out/vncd6/level_model.py` (declared
secondary) and `out/vncd6/per_seed_lists.py`. Data: `out/vncd6/` (ONE submission, `out/vncd6/batch.sh` -> run
**vncd6-2ad71d** on the house cluster, B200, console `out/vncd6_cluster.log`: **24 job(s), 0 failed (21.5 min)**);
analysis `out/vncd6/analysis_console.txt`, `out/vncd6/analysis/pairs_console.txt`, `out/vncd6/verification_console.txt`,
`out/vncd6/level_model_console.txt`. Predeclaration `out/vncd6/predeclared.json`, stamped **2026-09-15T07:47:38Z**
(archived unchanged as `out/vncd6/predeclared_archive/predeclared_2026-09-15T074738Z.json`) against the earliest run's own
`started_utc` **07:48:15Z**; submission receipt `out/vncd6/submitted_at.txt` (07:47:47Z). Working-tree record
`out/vncd6/tree_state.json` (sha256 of every shipped file; `tree_state_check.json`: UNCHANGED after the analysis).
Every number below is a run mean +- SD over the SIX runs of an arm (16 flies x 55 s window each; runs are the replicate
unit); verdicts are `flyverse.interp.common.compare` on runs (6 v 6, exact-U floor p 0.0021645). **Every per-seed list in
this document is pasted from `out/vncd6/analysis/per_seed_lists.txt`, which `per_seed_lists.py` emits from
`out/vncd6/analysis/per_seed.csv` (columns `key, arm, seed, file, value`, written by `pairs`) and from
`analysis/level_model_r2_residual_runs.csv`; the file and column are named beside each list** (docs/INTERP.md 10.4 item
28). Nothing in `out/vncd6` is compared row by row with `out/vncd5`; round 4b's numbers appear as labelled context and as
the level model's calibration.

## 0. Answer

**The amplitude / turn law adds no drive at AN04B003 or DNa02 beyond the per-phase modulation: F4 is closed.** The
modulation-only arm at the cycle's own realised amplitude (M2: `LegCycle(flat_amplitude=True, flat_amplitude_value=0.948)`)
sits at the cycle arm's afferent level -- chordotonal **86.65 +- 0.81 Hz against C's 87.40 +- 1.62 (-0.76 Hz)**, hair plate
46.66 vs 47.07 (-0.41), campaniform 24.89 vs 24.86 (+0.03), amplitude exactly 0.948000 on every walking frame with
|amp L-R| 0.000000 against C's 0.9478 / 0.0175 -- and **M2 v C is `null` on all seven primaries**: AN04B003_L -0.20 Hz
(z -0.5), AN04B003_R -0.27 (z -0.6), DNa02_L -0.037 (z -2.0, p 0.026), DNa02_R +0.000 (z +0.0), clean yaw SD +0.07
(z +0.4), straightness +0.031 (z +0.8), DNa02 L-R -0.037 (z -1.6). Round 4b's F4 was `result` (M over C by +1.62 /
+1.65 Hz of AN04B003 and +0.093 of DNa02_L) because M's flat amplitude of 1.000 bought it +6.0 Hz of chordotonal; at 0.948
the excess is gone, and with it the +0.66 / +0.75 Hz "structure residual" difference the round-4b skeptic (R2) found
between M and C: on the level model the residuals of M2 and C are **+3.99 / +5.15 (M2, L / R) against +4.10 / +5.28 (C)**,
difference **-0.12 / -0.13 Hz, z -0.6 / -0.5, `null`**, and on DNa02_L -0.036 (z -2.1, `null`).

**What the cycle does to the relay and to DNa02 is therefore the per-phase modulation, measured here with an arm that
cannot carry a turn term.** F2 (M2 v L, the modulation-only arm against the level reference at the same chordotonal level,
+0.66 Hz): **CALLED** on AN04B003_L **+3.92 Hz** (z +30.0, Holm 0.015), AN04B003_R **+6.85** (z +67.2), DNa02_L **+0.111**
(z +5.0) and straightness +0.339 (z +15.2); DNa02_R +0.094 (z +2.9, `null` on z alone, p at the floor), clean yaw SD
+0.66 (z +3.0 -- 2.96 unrounded, `null`), DNa02 L-R +0.017 (`null`). F3 (C v L, round 4's pair at 6 v 6) is CALLED on the
same four rows with the same sizes (AN04B003 +4.12 / +7.12, DNa02_L +0.149, straightness +0.308) -- the third batch in
which the cycle arm exceeds the level control on the relay by 4-7 Hz per side. On the level model (calibrated on round
4b's three steady arms, arm-level SEs, and transferring to this batch's L to -0.10 / +0.03 Hz) the structure term of the
modulation-only arm is **+4.09 / +5.12 Hz (M2, L / R side; pooled +4.60 +- 0.31 after the level correction)** and the
cycle's is +4.20 / +5.25 (pooled +4.72 +- 0.32): the same number. (The +- are full-covariance propagations over the
arm-level fit, where `corr(b, c)` is -0.951; section 7.)

**The one row the amplitude / turn term owns is the per-frame SIDED signal at DNa02, and it survives the level match.**
The tripod-conditioned DNa02 L-R swing `E[DNa02 L-R | chord L-R > 0] - E[. | < 0]` is **-0.338 +- 0.031 Hz under C against
-0.091 +- 0.012 under M2** (M2 v C +0.247, z +7.9, `result`; L -0.007), and `corr(DNa02 L-R, chord L-R)` is -0.142 (C)
against -0.032 (M2), while the AN04B003 antiphase is the same in both (-3.60 C, -3.63 M2) and the alternation SIZE is the
same (mean per-frame |chordotonal L-R| 12.36 C, 12.16 M2) -- but the afferent WAVEFORM is not. Split at one step cycle,
the two arms share the fast tripod term (SD 14.02 C, 13.99 M2) and differ 2.6-fold in the slow one (mean |slow| 1.65 C,
0.63 M2), which is yaw-locked only under C (corr(yaw, slow) -0.779 vs +0.009); DNa02's sidedness follows the slow term
alone and the relay follows the fast one (section 8.3). The amplitude / turn law does not make DNa02 read a matched
input differently: it CREATES a slow, yaw-locked sided afferent signal that an arm with |amp L-R| exactly 0 cannot have.
Round 4b saw this with M at the wrong level (-0.334 vs -0.107); at the right level the numbers are -0.338 vs
-0.091. It is a sided signal, not a rate: it does not move DNa02's
window rate, the yaw SD or the straightness (F1 `null`), and the raw DNa02 L-R differs by -0.037 (z -1.6, `null`).

**The Holm family was satisfiable this time and it called rows.** Six runs per arm, m = 7: the smallest attainable
adjusted p is 0.0021645 x 7 = **0.0152 <= 0.05**. F1 calls nothing (every row `null`, which is the answer); F2 calls four
of seven, F3 four, F4 seven.

**Under the project rule:** nothing is adopted. M2 is a LABELLED CONTROL; `LegCycle.flat_amplitude_value` is opt-in
(default 1.0, read only under `flat_amplitude`), and the shipped CPU path and round 4b's flat cycle are bit-identical with
it at its default (section 2). What an adoption of the leg cycle would still require is in section 9.

## 1. The question and the design

Round 4b (`docs/audits/level_controls.md`) left F4 open: its modulation-only arm M (`flat_amplitude=True`, amplitude 1.000)
fired MORE than the cycle arm C on AN04B003 (+1.62 / +1.65 Hz) and DNa02_L (+0.093), carried +6.0 Hz more chordotonal
because the default law realises 0.948 in straight walking, and -- after the level model took 60 % of that excess back as
level -- still had a structure residual 0.66 / 0.75 Hz LARGER than C's (its skeptic pass R2: "this batch cannot say that
the amplitude law adds no drive"). The confound is the level. The arm that settles it is M at the cycle's own realised
amplitude, so that the only difference between it and C is the amplitude law (the `half_width_m` yaw term and the
|amp L-R| turn term) and the per-frame variation of the amplitude around its mean.

**Design (four arms x 6 brain seeds = 24 room jobs, blocks `fam_r<seed>`; no compass).**

| arm | spec | classification |
|---|---|---|
| A | shipped (no sense) | the reference |
| L | `'all'`, `mn_ref_hz` **8.84** | LABELLED CONTROL: round 4's level-matched control, the level reference, re-run inside this batch |
| **M2** | `'all+leg_cycle+leg_cycle_flat'`, **`--flat-amplitude-value 0.948`** | LABELLED CONTROL of the body-model mechanism: `body.LegCycle(flat_amplitude=True, flat_amplitude_value=0.948)` holds every walking leg at the default law's own realised walking-mean amplitude; the phase / stance-swing modulation is kept, the amplitude law is removed, and the afferent LEVEL is the cycle's |
| C | `'all+leg_cycle'` | the body-model mechanism (rounds 3, 4, 4b's arm C): the treatment |

Primaries per pair (7 keys, Holm m = 7 within each family, 6 runs per arm): `DNa02_L_hz`, `DNa02_R_hz` (on the same
clean mask as `DNa02_LR_hz`), `yaw_sd_clean_deg_s`, `straightness`, `DNa02_LR_hz`, `chain_AN04B003_L_hz` /
`chain_AN04B003_R_hz`. Four predeclared families: **F1 M2 v C** (the question), **F2 M2 v L** and **F3 C v L** (structure
over level, with and without the amplitude law), **F4 C v A** (the fraction denominators). Each family's decision rule is
quoted verbatim in section 6. Two further pairs (L v A, M2 v A) are bookkeeping only.

**Why six runs.** Round 4b declared m = 7 at 5 v 5, where the exact-U floor is 0.0079 and 0.0079 x 7 = 0.056: nothing
could be called. At 6 v 6 the floor is 2 / C(12, 6) = 0.0021645 and m <= 23 is satisfiable; m = 7 gives a smallest
adjusted p of 0.0152 (docs/INTERP.md 10.2, 10.4 item 1).

## 2. The one new parameter and its bit-identity test

**`LegCycle.flat_amplitude_value` (`flyverse/body.py`, default `1.0`).** The flat branch of `advance()` was
`amp = where(air | ~isfinite(tau_st), 0, ones_like(v_leg))`; it is now `where(..., 0, full_like(v_leg,
flat_amplitude_value))`. The parameter is read by nothing unless `flat_amplitude` is True; with it at 1.0 the two forms
are the same float, bit for bit. The default amplitude law (`flat_amplitude=False`) is not touched. `senses.py` is
untouched: the spec token stays `leg_cycle_flat`, and the value travels on the job line (`--flat-amplitude-value`,
`probe_vnc_drive.ARM_CYCLE_KW` / `cycle_kwargs_of` / `attach_cycle`, refused without the flat token) and into the run JSON
(`sense.cycle` -- the whole `vars(LegCycle)` -- plus `sense.cycle_overrides` and the existing `leg_cycle_params`), so a
run is self-describing: M2's JSON reads `flat_amplitude True, flat_amplitude_value 0.948, overrides
flat_amplitude_value=0.948`; C's `flat_amplitude False, flat_amplitude_value 1.0`.

**Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`).** `tests/test_body_cycle.py::FlatAmplitudeValueTests` pins that the default is
1.0 and `vars(LegCycle()) == vars(LegCycle(flat_amplitude_value=1.0))`; that with the flag OFF a cycle carrying
`flat_amplitude_value=0.948` advances byte-for-byte as the default cycle at yaw +2 / -2 / 0 (the value is unread); that
`LegCycle(flat_amplitude=True)` and `LegCycle(flat_amplitude=True, flat_amplitude_value=1.0)` advance byte-for-byte (round
4b's M is the value 1.0); that with the value the amplitude is exactly 0.948 on every walking leg at any yaw while phase /
stance / freq / beta / load_L / load_R / n_stance equal the default's, and 0 standing and airborne; and that in `BatchSim`
(2 flies, 12 steps, `torch.testing.assert_close(rtol=0, atol=0)`) the default cycle with and without the value, and the
flat cycle with and without `=1.0`, are bit-identical, while the 0.948 cycle feeds the sense a chordotonal command that
equals the phase law at amplitude 0.948. `tests/test_bit_identity.py`'s shipped golden passes unchanged.

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_body_cycle.py tests/test_proprioception.py tests/test_bit_identity.py -q
    36 passed, 5 subtests passed                     (out/vncd6/smoke/pytest_cpu.txt)

`scripts/probe_vnc_drive.py` also gains the `level3` family, `ARM_ORDER` as a tuple (the two-character arm name), the
`per_seed.csv` emission in `pairs`, and a dedupe of `pairs`' key list (section 8, the one post-stamp reducer change).

## 3. The derivation of 0.948 (CPU; `out/vncd6/flat_amplitude_derivation.json`, console `flat_amplitude_derivation_console.txt`)

The law: `amp_i = max(v - s_i * yaw * half_width_m, 0) * tau_stance(v) / step_ref_m` for a walking leg on the ground, 0
standing and airborne. The value is the run mean of the per-leg amplitude of round 4b's cycle arm C (`out/vncd5/
room_C_r*_body.npz`, 5 runs, `cyc__amp_L` / `cyc__amp_R`) on WALKING frames -- post-skip, on the ground, step frequency
> 0, the frames on which the flat law is read -- rounded to three decimals. Per run (`flat_amplitude_derivation.json`
`runs[*].amp_mean_walking`): **[0.947665 0.947714 0.947744 0.947786 0.947730]**, run mean **0.947728 +- 0.000044**,
|amp L-R| 0.0170; the law at the recorded mean walking speed (9.794 mm/s, yaw 0) gives 0.947396, so the recorded mean is
the law, not an artefact of the mask. On the spread: the percentiles and the above-1.0 fraction in
`flat_amplitude_derivation.json` are of the per-frame L/R MEAN amplitude (5th / 50th / 95th 0.9451 / 0.9471 / 0.9523;
0.1 % above 1.0); of the PER-LEG amplitude the flat law replaces they are **0.9193 / 0.9473 / 0.9753 with 0.63 % of
leg-frames above 1.0** (per-run max 1.36-1.55), the round-4b skeptic pass's C4 numbers. The mean is the same on either
array, so the chosen value is unchanged; the distribution the constant replaces is three times wider than the L/R-mean
percentiles suggest. **Chosen `flat_amplitude_value` = 0.948.** Nothing about behaviour enters; it is a summary statistic of the
recordings. A CPU calibration pair (`out/vncd6/cal/`, 4 flies x 12 s, brain seed 0, one run each) predicted the M2 - C
level gap before submission: chordotonal 92.82 vs 92.73 (+0.09 Hz), hair plate 50.03 vs 50.01, campaniform 25.00 vs
25.00, |amp L-R| 0.0000 vs 0.0188 (recorded in `predeclared.json` `preconditions.P1_M2_level_match.cpu_prediction`).

## 4. The batch, and what the four arms realised (`out/vncd6/verification_console.txt`, `run_verification.csv`, `arm_levels.csv`)

ONE submission, `out/vncd6/batch.sh` (one `cluster_run.py` call, `--arm-block fam`, house target; every command
`mkdir -p out/vncd6 && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && ...;
st=$?; tail -4 ...; exit $st`), run **vncd6-2ad71d**: 24 room jobs in 6 blocks of 4 (`fam_r0`..`fam_r5`, the four arms of
one seed on one node), **24 job(s), 0 failed (21.5 min)**, `--fetch out/vncd6/`. Artefacts: 24 x (`.json`, `_body.npz`,
`_rec.npz`, `_flies.npz`, `_max.npz`) + 24 consoles. Every run: `device cuda`, `device_name NVIDIA B200`, `family level3`,
its arm's spec, its `sense` block (with the cycle block for M2 and C), its block key, one compiled-connectome md5, one
52-file source fingerprint; wall 143-716 s (the jobs of one block share the node). Sense parameters per arm, from the
JSONs: A none; L `all` at mn_ref **8.84** (hair 100 / load 50, the defaults); M2 `all+leg_cycle+leg_cycle_flat`, tokens
`leg_cycle,leg_cycle_flat`, cycle `flat_amplitude True / flat_amplitude_value 0.948`; C `all+leg_cycle`, cycle
`flat_amplitude False / 1.0`.

### 4.1 The realised levels (recomputed from `*_body.npz` on the post-skip frames, run mean +- SD over 6 runs)

| arm | chordotonal (L / R; L-R) | hair plate (L-R) | campaniform | haltere | leg MN L / R | amp on walking frames (run mean of the per-run min / max; \|L-R\|) |
|---|---|---|---|---|---|---|
| A | -- (no sense) | -- | -- | -- | 2.545 / 2.391 | -- |
| L | **85.99 +- 0.61** (92.59 / 79.12; **+13.47**) | 56.59 +- 0.41 (+9.14) | 49.57 | 18.86 | 5.250 / 4.397 | -- |
| **M2** | **86.65 +- 0.81** (86.66 / 86.64; +0.02) | **46.66 +- 0.44** (+0.00) | **24.89** | 16.56 | 4.758 / 4.604 | **0.948000** (0.9480 / 0.9480, one distinct value; **0.000000**) |
| C | **87.40 +- 1.62** (87.23 / 87.59; -0.36) | 47.07 +- 0.88 (-0.20) | 24.86 | 16.63 | 4.783 / 4.647 | 0.947750 (0.452 / 1.574, the mean of the per-run extrema -- pooled over the six runs the per-leg amplitude spans 0.060 .. 1.979; 0.017509) |

Step frequency 8.28 (M2) / 8.29 Hz (C) on walking frames, stance fraction 0.751 / 0.751; the walking fraction 0.925 / 0.935.

### 4.2 The predeclared preconditions (`precondition_checks.csv`; every one checked from the recordings before any verdict was read)

| precondition | result |
|---|---|
| P1 M2's chordotonal / hair plate / campaniform against C (+-3 / +-3 / +-1.5 Hz) | **PASS**: **-0.757 / -0.409 / +0.028 Hz** (the +0.028 Hz campaniform gap is the two arms' differing airborne fraction, not a channel difference: on non-airborne frames both arms read 24.998 Hz, M2 - C +0.0005) |
| P2 M2's amplitude 0.948 on walking frames (+-0.01), \|amp L-R\| <= 1e-6 | **PASS**: 0.948000 (min 0.948000, max 0.948000, 1 distinct value) and 0.000e+00, against C's 0.947750 / 0.01751 |
| P3 L's chordotonal against C (+-10) | **PASS**: -1.418 Hz |
| level gaps of every pair (reported) | M2 v C: chord -0.757, hair -0.409, camp +0.028, haltere -0.075; M2 v L: +0.662 / **-9.936** / **-24.681** / -2.300; C v L: +1.418 / -9.527 / -24.709 / -2.225 |
| P4 every channel inside its ledger bracket | **PASS** |
| P5 run configuration (24 runs, 6 per arm, cuda / B200, family, specs, cycle blocks, blocks r0..r5, 5 artefacts, one md5, one fingerprint) | **PASS** |
| P6 the predeclaration precedes the batch | **PASS**: stamped 07:47:38Z, earliest `started_utc` 07:48:15Z, latest `finished_utc` 08:00:47Z |

M2 v C is level-matched on all three leg channels to under 0.8 Hz; the pair the question rests on is a clean contrast.
M2 v L and C v L carry the same hair-plate (-9.9 / -9.5 Hz) and campaniform (-24.7) mismatches round 4b's L did, and are
read with them (section 6, section 7).

## 5. The primaries, per predeclared family (`analysis/decision_table.csv`; 6 runs per arm, floor p 0.0021645, Holm m = 7)

Run mean +- SD; `diff` = treatment - reference; `verdict` is `compare`'s; `holm` the Holm-adjusted p; **CALLED** = `result`
AND holm <= 0.05.

| family | key | reference | treatment | diff | z | p | holm | verdict |
|---|---|---|---|---|---|---|---|---|
| **F1 M2 v C** | DNa02_L | C 0.5715 +- 0.0184 | M2 0.5343 +- 0.0233 | -0.0372 | -2.0 | 0.026 | 0.182 | null |
| | DNa02_R | 0.3800 +- 0.0136 | 0.3801 +- 0.0145 | +0.0001 | +0.0 | 0.937 | 1.000 | null |
| | yaw SD clean | 7.827 +- 0.200 | 7.902 +- 0.113 | +0.074 | +0.4 | 0.699 | 1.000 | null |
| | straightness | 0.8120 +- 0.0380 | 0.8426 +- 0.0304 | +0.0306 | +0.8 | 0.240 | 1.000 | null |
| | DNa02 L-R | 0.1915 +- 0.0238 | 0.1542 +- 0.0305 | -0.0373 | -1.6 | 0.065 | 0.390 | null |
| | AN04B003_L | 23.162 +- 0.421 | 22.960 +- 0.230 | -0.202 | -0.5 | 0.240 | 1.000 | null |
| | AN04B003_R | 23.628 +- 0.479 | 23.361 +- 0.268 | -0.267 | -0.6 | 0.240 | 1.000 | null |
| **F2 M2 v L** | DNa02_L | L 0.4228 +- 0.0223 | 0.5343 | **+0.1115** | +5.0 | 0.0022 | 0.015 | result **CALLED** |
| | DNa02_R | 0.2858 +- 0.0326 | 0.3801 | +0.0942 | +2.9 | 0.0022 | 0.015 | null |
| | yaw SD clean | 7.239 +- 0.224 | 7.902 | +0.662 | +3.0 (2.96) | 0.0022 | 0.015 | null |
| | straightness | 0.5036 +- 0.0224 | 0.8426 | +0.3390 | +15.2 | 0.0022 | 0.015 | result **CALLED** |
| | DNa02 L-R | 0.1370 +- 0.0226 | 0.1542 | +0.0172 | +0.8 | 0.394 | 0.394 | null |
| | AN04B003_L | 19.040 +- 0.131 | 22.960 | **+3.921** | +30.0 | 0.0022 | 0.015 | result **CALLED** |
| | AN04B003_R | 16.510 +- 0.102 | 23.361 | **+6.851** | +67.2 | 0.0022 | 0.015 | result **CALLED** |
| **F3 C v L** | DNa02_L | 0.4228 | 0.5715 | **+0.1487** | +6.7 | 0.0022 | 0.015 | result **CALLED** |
| | DNa02_R | 0.2858 | 0.3800 | +0.0941 | +2.9 | 0.0022 | 0.015 | null |
| | yaw SD clean | 7.239 | 7.827 | +0.588 | +2.6 | 0.0087 | 0.015 | null |
| | straightness | 0.5036 | 0.8120 | +0.3083 | +13.8 | 0.0022 | 0.015 | result **CALLED** |
| | DNa02 L-R | 0.1370 | 0.1915 | +0.0546 | +2.4 | 0.0043 | 0.015 | null |
| | AN04B003_L | 19.040 | 23.162 | **+4.122** | +31.6 | 0.0022 | 0.015 | result **CALLED** |
| | AN04B003_R | 16.510 | 23.628 | **+7.118** | +69.8 | 0.0022 | 0.015 | result **CALLED** |
| **F4 C v A** | DNa02_L | A 0.0161 +- 0.0066 | 0.5715 | +0.5555 | +83.9 | 0.0022 | 0.015 | result **CALLED** |
| | DNa02_R | 0.0826 +- 0.0128 | 0.3800 | +0.2973 | +23.2 | 0.0022 | 0.015 | result **CALLED** |
| | yaw SD clean | 2.772 +- 0.238 | 7.827 | +5.056 | +21.2 | 0.0022 | 0.015 | result **CALLED** |
| | straightness | 0.9957 +- 0.0009 | 0.8120 | -0.1837 | -202.8 | 0.0022 | 0.015 | result **CALLED** |
| | DNa02 L-R | -0.0666 +- 0.0099 | 0.1915 | +0.2581 | +26.2 | 0.0022 | 0.015 | result **CALLED** |
| | AN04B003_L | 0.596 +- 0.019 | 23.162 | +22.566 | +1216.1 | 0.0022 | 0.015 | result **CALLED** |
| | AN04B003_R | 0.314 +- 0.009 | 23.628 | +23.314 | +2545.7 | 0.0022 | 0.015 | result **CALLED** |

**Per-seed scatter (r0..r5; `out/vncd6/analysis/per_seed_lists.txt`, from `per_seed.csv` column `value`):**

    DNa02_L_hz A [0.0147 0.0206 0.0158 0.0078 0.0112 0.0263]  0.0161 +- 0.0066
    DNa02_L_hz L [0.4630 0.4020 0.4309 0.4056 0.4151 0.4204]  0.4228 +- 0.0223
    DNa02_L_hz M2 [0.5198 0.5285 0.5283 0.5521 0.5063 0.5707]  0.5343 +- 0.0233
    DNa02_L_hz C [0.5845 0.5681 0.5577 0.5593 0.6026 0.5569]  0.5715 +- 0.0184
    DNa02_R_hz A [0.0813 0.0790 0.0857 0.0840 0.0630 0.1028]  0.0826 +- 0.0128
    DNa02_R_hz L [0.3369 0.2584 0.2821 0.3019 0.2451 0.2907]  0.2858 +- 0.0326
    DNa02_R_hz M2 [0.3575 0.3979 0.3860 0.3792 0.3894 0.3704]  0.3801 +- 0.0145
    DNa02_R_hz C [0.3849 0.3568 0.3814 0.3989 0.3792 0.3785]  0.3800 +- 0.0136
    yaw_sd_clean_deg_s A [2.7325 2.7874 2.8582 2.6335 2.4541 3.1639]  2.7716 +- 0.2379
    yaw_sd_clean_deg_s L [7.6227 7.1008 7.2577 7.3262 6.9754 7.1524]  7.2392 +- 0.2242
    yaw_sd_clean_deg_s M2 [7.7915 7.9465 7.9185 7.9656 7.7447 8.0434]  7.9017 +- 0.1125
    yaw_sd_clean_deg_s C [8.0077 7.6200 7.6077 8.0236 7.9860 7.7189]  7.8273 +- 0.1996
    straightness A [0.9953 0.9956 0.9959 0.9962 0.9942 0.9968]  0.9957 +- 0.0009
    straightness L [0.4858 0.4878 0.4994 0.5403 0.4873 0.5212]  0.5036 +- 0.0224
    straightness M2 [0.8249 0.8394 0.8399 0.8608 0.8897 0.8010]  0.8426 +- 0.0304
    straightness C [0.8259 0.7437 0.8104 0.8511 0.8016 0.8391]  0.8120 +- 0.0380
    DNa02_LR_hz A [-0.0666 -0.0585 -0.0700 -0.0762 -0.0518 -0.0765]  -0.0666 +- 0.0099
    DNa02_LR_hz L [0.1261 0.1436 0.1487 0.1037 0.1700 0.1297]  0.1370 +- 0.0226
    DNa02_LR_hz M2 [0.1623 0.1305 0.1423 0.1729 0.1169 0.2004]  0.1542 +- 0.0305
    DNa02_LR_hz C [0.1995 0.2112 0.1763 0.1603 0.2233 0.1784]  0.1915 +- 0.0238
    chain_AN04B003_L_hz A [0.6015 0.5871 0.5727 0.6072 0.6231 0.5814]  0.5955 +- 0.0186
    chain_AN04B003_L_hz L [19.2034 18.9473 19.1402 18.8409 19.0568 19.0492]  19.0396 +- 0.1306
    chain_AN04B003_L_hz M2 [22.9295 22.7526 22.8311 23.2708 22.7591 23.2178]  22.9602 +- 0.2298
    chain_AN04B003_L_hz C [22.9417 22.6076 23.4170 23.2818 23.7913 22.9314]  23.1618 +- 0.4206
    chain_AN04B003_R_hz A [0.3155 0.3189 0.3163 0.3125 0.3250 0.2977]  0.3143 +- 0.0092
    chain_AN04B003_R_hz L [16.6129 16.3867 16.6402 16.4201 16.4777 16.5227]  16.5100 +- 0.1020
    chain_AN04B003_R_hz M2 [23.3318 23.0958 23.2114 23.6792 23.1455 23.7027]  23.3610 +- 0.2676
    chain_AN04B003_R_hz C [23.3527 23.0417 23.8345 23.7693 24.4015 23.3701]  23.6283 +- 0.4790

Reading the M2 and C lists side by side: on AN04B003 the two arms interleave (M2's r3 23.27 above C's r0, r1, r5), on
DNa02_L C's run sits 0.007-0.096 above M2's in five of six seeds (M2's r5 0.5707 is above C's r5 0.5569), on DNa02_R
the lists are indistinguishable. Two rows that were `result` in round 4's C v L are `null` here on z alone with the exact p at or near
the floor: DNa02_R (+0.094, z +2.9 in both F2 and F3, every treatment run above every L run: p 0.0022) and the clean yaw
SD (F3 +0.59, z +2.6, p 0.0087 -- C's r2 7.6077 sits below L's r0 7.6227, so the separation is not complete this time; F2
+0.66, z 2.96, p 0.0022). Both are the reference arm's scatter in the denominator, as in round 4b (its section 9), and
both are reported as `null`.

**The fractions (X - A) / (C - A), paired by seed (`analysis/fractions.csv`, `per_seed_lists.txt` `fraction_*`):**
DNa02_L L 0.73 +- 0.04 / **M2 0.94 +- 0.07**; DNa02_R 0.68 / **1.00**; clean yaw SD 0.88 / **1.02**; DNa02-active fraction
0.71 / **0.96**; AN04B003_L 0.82 / **0.99**; AN04B003_R 0.69 / **0.99**; straightness 2.75 / 0.86. The modulation-only arm
is the cycle arm on every rate row; the level control is 70-88 % of it.

## 6. Each pair's decision, in its predeclared words

Quoted from `out/vncd6/predeclared.json` (`primaries.families.<name>.decision_rule_in_words`), then applied.

**F1 M2 v C.** *"With the preconditions P1 (M2's three leg channels at C's levels) and P2 (M2's amplitude at 0.948,
|amp L-R| 0) met: if M2 v C is NULL on chain_AN04B003_L, chain_AN04B003_R, DNa02_L and DNa02_R, the amplitude law -- and
with it the half_width_m yaw term and the |amp L-R| turn term -- adds NO drive at the relay or at DNa02 beyond the
per-leg / per-phase modulation at the same level: the whole cycle effect on those cells is the modulation, and round 4b's
M-over-C excess was the level. If M2 v C is RESULT on AN04B003 or on DNa02 (either side), the amplitude law contributes
drive of that sign ... The three behavioural / sided rows (clean yaw SD, straightness, DNa02 L-R) are read separately and
do not decide the drive question ..."*
-> **P1 and P2 met (section 4.2); `null` on AN04B003_L (-0.20, z -0.5), AN04B003_R (-0.27, z -0.6), DNa02_L (-0.037,
z -2.0) and DNa02_R (+0.000); `null` on the three behavioural rows too. The first branch fires: the amplitude law adds no
drive at the relay or at DNa02 beyond the modulation, and round 4b's M-over-C excess was its level.** The one trend in the
family is DNa02_L, where M2 is 0.037 Hz BELOW C (p 0.026, five of six seeds); it has the sign opposite to "the amplitude
law adds drive", it is inside the family's null, and the level-corrected residual difference is -0.036 (z -2.1, `null`).

What F1 could have seen: at 6 v 6 a row is called only at |z| >= 3, i.e. at three times C's own between-run SD --
1.26 Hz (AN04B003_L), 1.44 (AN04B003_R), 0.055 (DNa02_L), 0.041 (DNa02_R), 0.60 deg/s (clean yaw SD), 0.114
(straightness), 0.071 (DNa02 L-R). Round 4b's M-over-C excess (+1.62 / +1.65 Hz of AN04B003, +0.093 of DNa02_L)
clears every one of them, so the null is a null against the effect the arm was built to exclude, not an absence of
power.

**F2 M2 v L.** *"With P1 and P3 met: if M2 v L is RESULT POSITIVE on chain_AN04B003_L / _R (and on DNa02_L / DNa02_R), the
per-phase modulation alone -- no amplitude law, no turn term, at the cycle's afferent level -- raises the relay (and
DNa02) over the round-2 transducer of the same chordotonal level: round 4b's ~4.5 Hz structure term IS the modulation ...
If M2 v L is NULL on AN04B003 and DNa02, the modulation-only arm does not exceed the level control ... (the hair-plate /
campaniform mismatch between M2 and L, ~-10 / -25 Hz by the laws, is reported beside the pair as in round 4b, and the
audit says the pair carries it). Straightness and the drift are sidedness rows (round 4b F1) and are reported, not
attributed."*
-> **`result` POSITIVE and CALLED on AN04B003_L (+3.92) and AN04B003_R (+6.85) and on DNa02_L (+0.111); DNa02_R +0.094
(z +2.9, `null`). The first branch fires: the modulation alone, with no amplitude law and no turn term, raises the relay
by 4-7 Hz per side and DNa02_L by 0.11 Hz over the level control at the same chordotonal level (+0.66 Hz).** The pair
carries the -9.94 Hz hair-plate and -24.68 Hz campaniform mismatch; at the level model's arm-level slope the hair plate
is worth +0.64 Hz of the pooled +5.39 Hz AN04B003 difference and the corrected structure term is +4.60 +- 0.31 (section
7). Straightness (+0.339, CALLED) and the drift (+1.19 vs +3.37 deg/s) are the round-2 law's sidedness rows, reported.

**F3 C v L.** *"If C v L is RESULT on DNa02_L, DNa02_R and AN04B003_L / _R, round 4's and round 4b's within-batch finding
reproduces a third time; the clean-yaw-SD row is reported with its z whatever its verdict ..."*
-> **CALLED on DNa02_L (+0.149), AN04B003_L (+4.12), AN04B003_R (+7.12) and straightness (+0.308); DNa02_R +0.094 (z +2.9,
`null` on z, p at the floor); clean yaw SD +0.59 (z +2.6, p 0.0087, `null`).** The relay and DNa02_L reproduce for the
third batch (round 4: +0.127 / +3.7-6.7; round 4b: +0.092 / +3.73 / +6.72; here +0.149 / +4.12 / +7.12). DNa02_R is
`result` in rounds 4 and 4b and `null` on z here at the same +0.09-0.10 Hz size; the yaw SD is `null` in 4b and here.
The rule's first branch names DNa02_R as well, and DNa02_R is `null` here, so this is the rule's MIXED case, not its
first branch: the pair reproduces on DNa02_L and on both AN04B003 sides for a third batch and does not reproduce as a
verdict on DNa02_R at 6 v 6, at an unchanged size (+0.094 here against +0.103 in round 4 and +0.107 in round 4b).

**F4 C v A.** -> **CALLED on all seven** (DNa02_L +0.556, DNa02_R +0.297, yaw SD +5.06, straightness -0.184, DNa02 L-R
+0.258, AN04B003 +22.57 / +23.31): the cycle fires DNa02 and the relay and raises the yaw SD over the shipped path, as in
rounds 3, 4 and 4b; the fractions are in section 5.

## 7. The level model: the structure term with and without the amplitude law (declared SECONDARY, labelled CROSS-BATCH; `out/vncd6/analysis/level_model_r2*.csv/json`, console `out/vncd6/level_model_console.txt`)

`AN04B003_side = a_side + b x chordotonal_side + c x hair_plate_side`, fitted on round 4b's three steady arms (L, U, K of
`out/vncd5`) on their **six (arm, side) means** -- the arm-level fit its skeptic pass asked for (R9), 2 residual df --
and evaluated at THIS batch's arms' own realised per-side levels. Slopes **b +0.1848 +- 0.0355 per Hz chordotonal,
c -0.0648 +- 0.0285 per Hz hair plate** (a_L +5.976, a_R +5.213, residual standard error 0.128 (sqrt(RSS / 2 df); the
RMSE over the six fitted points is 0.074); the run-level fit is b +0.1921 / c -0.0697 with pseudo-replicated SEs, for the
record).

A within-batch calibration is not available: L is the only steady-input arm in this batch, two (arm, side) means
against the model's four parameters, so the cross-batch application is the only option rather than a preference. It
is also a JOINT extrapolation, as round 4b's skeptic pass found (R8) and as this batch repeats: the calibration's six
arm-side points lie near a line, `corr(chord, hair) = +0.925`, `hair = 0.979 x chord - 28.73` with 2.65 Hz of scatter
about it, and the cycle arms sit **-9.4 (M2) to -9.8 (C) Hz of hair plate off that line** while this batch's L sits on
it (-0.84 / +3.20). The residual DIFFERENCE M2 - C is nearly free of this (the pair's own gaps are -0.57 / -0.95 of
chordotonal and -0.31 / -0.51 of hair plate, worth -0.09 / -0.14 Hz), which is why F1's level correction is safe; the
+4.60 / +4.72 structure terms over L are not, and neither corrects for the -24.7 Hz campaniform mismatch (round 4b
C8: on the steady arms the campaniform is a near-binary arm indicator and is not separable from the hair plate).

Three readings:

1. **Transfer.** This batch's L under the round-4b calibration: residual **-0.099 +- 0.041 (L side), +0.033 +- 0.041 (R
   side)** -- the intercepts transfer between batches to 0.1 Hz. The residual differences below are anchoring-free (a
   per-side constant cancels).
2. **The structure terms.** Per side (run mean +- SD over 6 runs; `level_model_r2_predictions.csv`), and per run
   (`per_seed_lists.txt` `residual_AN04B003 *`, from `level_model_r2_residual_runs.csv` column `residual`):

       residual_AN04B003 M2L [+3.9633 +3.8925 +3.9313 +4.1780 +3.8840 +4.0711]  +3.9867 +- 0.1156
       residual_AN04B003 M2R [+5.1330 +5.0025 +5.0800 +5.3527 +5.0371 +5.3217]  +5.1545 +- 0.1484
       residual_AN04B003 CL [+4.0178 +3.8689 +4.2196 +4.1631 +4.3620 +3.9843]  +4.1026 +- 0.1791
       residual_AN04B003 CR [+5.1381 +5.0068 +5.3502 +5.3637 +5.6794 +5.1387]  +5.2795 +- 0.2393

   | pair | side | raw diff (z) | level gap chord / hair (Hz) | level correction +- slope unc. | residual diff (z, p) | verdict |
   |---|---|---|---|---|---|---|
   | **M2 v C** | L | -0.202 (-0.5) | -0.57 / -0.31 | -0.086 +- 0.022 | **-0.116** (-0.6, 0.31) | null |
   | | R | -0.267 (-0.6) | -0.95 / -0.51 | -0.142 +- 0.037 | **-0.125** (-0.5, 0.24) | null |
   | | pooled | -0.234 (-0.5) | -0.76 / -0.41 | -0.114 +- 0.029 | **-0.121** (-0.6, 0.31) | null |
   | M2 v L | L | +3.921 (+30.0) | -5.93 / -14.38 | -0.165 +- 0.461 | +4.086 (+99.3, 0.0022) | result |
   | | R | +6.851 (+67.2) | +7.52 / -5.25 | +1.729 +- 0.306 | +5.122 (+125.8, 0.0022) | result |
   | | pooled | +5.386 (+47.7) | +0.79 / -9.81 | +0.782 +- 0.281 | **+4.604** (+129.3, 0.0022) | result |
   | C v L | L | +4.122 (+31.6) | -5.36 / -14.08 | -0.079 +- 0.444 | +4.202 (+102.1, 0.0022) | result |
   | | R | +7.118 (+69.8) | +8.47 / -4.73 | +1.872 +- 0.329 | +5.247 (+128.9, 0.0022) | result |
   | | pooled | +5.620 (+49.8) | +1.55 / -9.40 | +0.896 +- 0.274 | **+4.724** (+132.7, 0.0022) | result |

   Round 4b's skeptic R2 found M's residual 0.66 / 0.75 Hz ABOVE C's (z +3.8 / +3.5) at the wrong level; at the right
   level **M2's residual is 0.12 / 0.13 Hz BELOW C's and `null`**. The level correction of M2 v C is -0.114 Hz
   (M2 sits 0.76 Hz of chordotonal under C), and the raw pair and the corrected pair say the same thing. The structure
   term of the cycle over the level control, pooled over sides, is **+4.60 Hz without the amplitude law and +4.72 with
   it**, over the -9.4 .. -9.8 Hz hair-plate gap.

   On the uncertainty of those corrections: the level correction of a pair is b x dChord + c x dHair over the pair's
   realised level gaps, and its uncertainty is `sqrt([dChord dHair] Cov(b, c) [dChord dHair]^T)` with the ARM-LEVEL
   covariance. The two slopes are strongly anti-correlated on this design (`corr(b, c) = -0.951`), so the
   independent-SE form `sqrt((dChord se_b)^2 + (dHair se_c)^2)` is not the right one: it OVERstates the L-side
   corrections (+-0.46 -> +-0.22, +-0.44 -> +-0.23) and UNDERstates the R-side and the pooled ones (+-0.31 -> +-0.41,
   +-0.33 -> +-0.43; pooled +-0.28 -> +-0.31 for M2 v L and +-0.27 -> +-0.32 for C v L). **Pooled structure terms:
   M2 +4.60 +- 0.31, C +4.72 +- 0.32.** The M2 v C correction is small enough that the distinction does not matter
   there (-0.114 +- 0.016). The `level correction +- slope unc.` column of the table above prints the independent-SE
   form; the full-covariance values are the ones just listed.

   The decomposition round 4b gave for C v L (+0.01 chordotonal, +0.72 hair plate, +4.49 modulation of +5.22) reads
   here **+0.29 + 0.61 + 4.72 = 5.62** (C v L) and **+0.15 + 0.64 + 4.60 = 5.39** (M2 v L).
3. **DNa02 (the model round 4b found not robust; reported, not interpreted).** Slopes b +0.0020 +- 0.0068, c +0.0020 +-
   0.0055 (neither separated from zero); transfer +0.007 / +0.010; residuals M2 +0.159 / +0.100, C +0.194 / +0.097
   (`residual_DNa02_cmd *` lists in `per_seed_lists.txt`); M2 v C residual difference -0.036 (L, z -2.1, `null`; the
   raw DNa02_L trend of F1) and +0.003 (R, inside the calibration error). M2 v L and C v L: +0.152 / +0.188 (L side,
   `result`), +0.090 / +0.087 (R side, z +2.8 / +2.7, `null`).

## 8. The decomposition, the sided signal per frame, the route and the behaviour

### 8.1 DNa02's input per arm (`analysis/dna02_decompose_summary.csv`, rate-weighted mV/s per post cell; `rate_hz` is AN04B003's)

| DNa02_L input | A | L | M2 | C |
|---|---|---|---|---|
| E total / I total / **net** | +742 / -1046 / **-304** | +1335 / -1273 / **+61** | +1356 / -1259 / **+97** | +1365 / -1259 / **+106** |
| **AN04B003/L** | +9.2 | +294.1 | **+354.7** | **+357.8** |
| **SNpp45/?** (hair plate, direct) | 0.0 | +167.6 | +128.2 | +129.0 |
| PS059/L (haltere afferents) | -90.6 | -209.1 | -204.3 | -205.6 |
| IN12B014/R | -90.8 | -64.0 | -65.4 | -65.1 |
| IN19A003/L | -43.0 | -18.1 | -20.9 | -20.2 |
| LT51/L | -57.8 | -48.8 | -45.5 | -45.6 |

| DNa02_R input | A | L | M2 | C |
|---|---|---|---|---|
| E total / I total / **net** | +808 / -1170 / **-362** | +1228 / -1498 / **-270** | +1331 / -1497 / **-166** | +1338 / -1500 / **-162** |
| **AN04B003/R** | +4.8 | +249.7 | **+353.3** | **+357.3** |
| SNpp45/? | 0.0 | +65.4 | +58.9 | +59.5 |
| PS059/R | -49.7 | -170.3 | -167.8 | -168.4 |
| IN12B014/L | -126.0 | -94.3 | -98.6 | -98.2 |
| IN19A003/R | -60.2 | -49.0 | -57.1 | -55.4 |
| LT51/R | -68.3 | -71.6 | -67.1 | -67.8 |

M2 and C put the same AN04B003 drive on DNa02 (+355 / +353 vs +358 / +357 mV/s), the same direct hair-plate term
(+128 vs +129) and the same haltere cancellation; every row of the two cycle arms agrees to within 1 %, and both differ
from L where rounds 4 and 4b said they do -- on the ascending relay.

### 8.2 The hair-plate route per arm (`pairwise.csv`, `chain_*` keys: every cell of the type, per side)

| type (Hz) | A | L | M2 | C |
|---|---|---|---|---|
| SNpp45 | 0.00 | 55.65 | 45.77 | 46.16 |
| IN13B001 (L / R) | 10.64 / 6.50 | 70.81 / 81.62 | 65.57 / 63.71 | 66.24 / 64.23 |
| AN04B003 (L / R) | 0.60 / 0.31 | 19.04 / 16.51 | **22.96 / 23.36** | **23.16 / 23.63** |
| PS059 (L / R) | 8.80 / 4.93 | 20.30 / 16.89 | 19.84 / 16.65 | 19.97 / 16.71 |
| PS196_b (L) | 0.21 | 7.68 | 8.47 | 8.51 |
| GLNO | 0.05 | 4.44 | 4.23 | 4.27 |

M2 v C is `null` on every one of them (SNpp45 -0.40, z -0.5; IN13B001 -0.67 / -0.52; PS059 -0.13 / -0.06; PS196_b_L
-0.03; GLNO -0.04). Every cell of these tables fires in all three sense arms. In the shipped arm A the hair-plate
afferent SNpp45 is silent (0.00 Hz: no sense is attached), and every other row sits at its A-arm floor (AN04B003
0.60 / 0.31, GLNO 0.05, PS196_b 0.21, IN13B001 10.64 / 6.50, PS059 8.80 / 4.93 Hz).

### 8.3 Sidedness per frame (`analysis/sided_frames.csv`; clean walking frames, per fly then the run mean; rec rows at lag 0)

(the `mean per-frame |chordotonal L-R|` row is `commanded_chordotonal_absLR_hz` from `analysis/pairwise.csv`, a room
key on the post-skip non-airborne mask, not a `sided_frames.csv` clean-frame statistic; on the clean-frame mask it
reads 12.07 M2 / 12.27 C)

| statistic | A | L | M2 | C | M2 v C |
|---|---|---|---|---|---|
| corr(yaw, chordotonal cmd L-R) | -- | +0.270 | +0.015 | **-0.147** | +0.162, z +29.0, result |
| corr(yaw, leg MN L-R) | +0.250 | +0.255 | +0.132 | +0.040 | +0.091, z +7.7, result |
| corr(AN04B003 L-R, chordotonal L-R) | -- | +0.191 | **-0.360** | **-0.345** | -0.015, z -6.4, result |
| E[AN04B003 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) | -- | +1.96 | **-3.63** | **-3.60** | -0.03, z -1.0, null |
| corr(DNa02 L-R, chordotonal L-R) | -- | +0.004 | -0.032 | **-0.142** | +0.110, z +18.5, result |
| **E[DNa02 L-R \| chord L-R > 0] - E[. \| < 0] (Hz)** | -- | -0.007 | **-0.091** | **-0.338** | **+0.247, z +7.9, result** |
| corr(AN04B003 L-R, yaw) | +0.025 | +0.082 | +0.009 | -0.059 | +0.068, z +8.8, result |
| mean per-frame \|chordotonal L-R\| (Hz) | 0 | 16.06 | 12.16 | 12.36 | -0.20, z -0.9, null |
| mean signed yaw, clean (deg/s; + left) | +0.18 | **+3.37** | +1.19 | +1.36 | -0.17, z -1.2, null |

Per seed (`per_seed_lists.txt`, from `per_seed.csv`):

    dna02LR_given_chordLR_pos_minus_neg L [-0.0377 -0.0086 0.0160 0.0424 -0.0330 -0.0222]  -0.0072 +- 0.0311
    dna02LR_given_chordLR_pos_minus_neg M2 [-0.0672 -0.0920 -0.0958 -0.0907 -0.1005 -0.1002]  -0.0911 +- 0.0124
    dna02LR_given_chordLR_pos_minus_neg C [-0.3594 -0.2937 -0.3028 -0.3616 -0.3638 -0.3440]  -0.3376 +- 0.0314
    corr_dna02LR_chordLR M2 [-0.0300 -0.0334 -0.0321 -0.0289 -0.0324 -0.0342]  -0.0318 +- 0.0020
    corr_dna02LR_chordLR C [-0.1494 -0.1400 -0.1317 -0.1453 -0.1425 -0.1417]  -0.1418 +- 0.0059
    an04LR_given_chordLR_pos_minus_neg M2 [-3.6055 -3.6471 -3.6426 -3.7003 -3.5733 -3.6314]  -3.6334 +- 0.0428
    an04LR_given_chordLR_pos_minus_neg C [-3.6175 -3.5598 -3.6316 -3.6358 -3.6006 -3.5597]  -3.6008 +- 0.0341

Three things read off it. (i) **The tripod-locked antiphase at AN04B003 is the per-phase structure's**: -3.63 Hz under M2,
-3.60 under C, +1.96 (the round-2 law's DC bias) under L. (ii) **The per-frame sided signal at DNa02 belongs to the
amplitude / turn term, and the attribution survives the level match**: -0.338 Hz under C against -0.091 under M2 (six
non-overlapping runs each; round 4b: -0.334 vs -0.107 with M at the wrong level), with the same antiphase at the relay
(-3.63 M2, -3.60 C) and the same alternation SIZE (mean per-frame |chordotonal L-R| 12.16 vs 12.36 Hz,
`commanded_chordotonal_absLR_hz` in `analysis/pairwise.csv`, post-skip non-airborne frames; 12.07 vs 12.27 on the
clean-frame mask) -- but **not the same afferent waveform**. Splitting the commanded chordotonal L-R at one step cycle
(a 12-frame centred boxcar) leaves the two arms with the same FAST tripod term (SD 13.99 M2 vs 14.02 C; mean |.| 11.95
vs 12.05) and a SLOW term that differs 2.6-fold (mean |slow| 0.63 vs 1.65; SD 1.55 vs 2.78, z -13.2) and is yaw-locked
only under C (corr(yaw, slow) +0.009 M2 vs -0.779 C). DNa02's sidedness follows the slow term alone:
E[DNa02 L-R | slow > 0] - E[. | < 0] is +0.066 (M2) against -1.552 (C), while on the fast term the two arms agree
(-0.123 vs -0.159); the relay is the mirror image, identical on the fast term (-3.756 vs -3.750) and small on the slow
one. **The amplitude / turn law does not change how DNa02 reads a matched afferent input: it CREATES a slow,
yaw-locked sided afferent signal that an arm with |amp L-R| exactly 0 cannot have, and DNa02 -- not the relay --
integrates it.** That is what "the turn term owns this row" means here. (iii) The realised yaw is coupled to the
afferent L-R only under C (corr -0.147; M2 +0.015): that coupling is the amplitude law reading the body's own yaw into
the afferents (`amp_LR_yaw_corr` -0.42 under C, undefined under M2 where |amp L-R| is exactly 0), and it is what the
sided DNa02 signal follows. It changes no window rate and no behavioural primary.

### 8.4 Behaviour (run mean +- SD; `analysis/room_table.csv`, `pairwise.csv`)

| key | A | L | M2 | C | M2 v C |
|---|---|---|---|---|---|
| clean yaw SD (deg/s) | 2.77 +- 0.24 | 7.24 +- 0.22 | 7.90 +- 0.11 | 7.83 +- 0.20 | +0.07, null |
| median \|yaw\|, clean | 0.74 | **3.50** | 1.57 | 1.57 | -0.00, null |
| mean signed yaw (+ left) | +0.18 | **+3.37** | +1.19 | +1.36 | -0.17, null |
| straightness | 0.996 | **0.504** | 0.843 | 0.812 | +0.03, null |
| flies (of 16) off the table | 0.2 | **11.8** | 7.8 | 9.2 | -1.3, null |
| hops per fly | 0.02 | 0.72 | 0.39 | 0.50 | -0.11, null |
| clean fraction | 0.999 | 0.882 | 0.934 | 0.919 | +0.015, null |
| net heading change (turns / fly) | 0.033 | 0.265 | 0.211 | 0.247 | -0.036, null |
| DNa02-active frames | 0.0070 | 0.0501 | 0.0648 | 0.0674 | -0.003, null |
| leg MN L-R (Hz) | +0.154 | **+0.860** | +0.155 | +0.137 | +0.018, z +3.4, result |
| haltere MN L-R (Hz) | -1.77 | -1.60 | -2.57 | -2.66 | +0.10, null |
| GF per-frame max (Hz) | 25.2 | 36.3 | 29.1 | 31.9 | -2.8, null |
| step frequency (Hz) / stance fraction | | | 7.70 / 0.769 | 7.79 / 0.766 | null / null |

The clean fraction spans 1.13x (0.882 L .. 0.999 A), inside the 2x that would oblige a window-matched re-scoring
(docs/INTERP.md 10.4 item 23). The one M2 v C `result` outside the sided rows is the leg-MN L-R (+0.018 Hz, z +3.4, on a
reference SD of 0.0055 Hz): a 0.4 % difference of the leg-MN side bias, reported as the magnitude it is.

## 9. What an adoption of the leg cycle would still require (nothing is adopted)

1. **The turn term is now sized and located, and it is not a drive.** Its whole measurable footprint in this batch is
   the per-frame sided DNa02 signal (-0.338 vs -0.091 Hz) and the yaw <-> afferent coupling of 8.3; it changes no window
   rate. `half_width_m` (1.0 mm, unmeasured) scales exactly that row, so a ledger value for it -- and for
   `lit.walk.outer_leg_step_ratio_in_turn` (still empty) -- is what would make the sided signal a measured quantity
   rather than a constant's consequence.
2. **A channel-matched control that also holds the chordotonal level** (round 4b's item 3): the structure term of +4.6 /
   +4.7 Hz over L still carries a -9.5 .. -9.9 Hz hair-plate and -24.7 Hz campaniform mismatch, corrected here at a slope
   with an arm-level SE of 0.03 Hz/Hz (+0.6 +- 0.3 Hz). A three-parameter fixed point (mn_ref, hair_plate_max,
   campaniform_load) would turn the correction into a matched pair.
3. **The suite with the cycle on** (`scripts/benchmark.py` and the room ledger rows under `all+leg_cycle`, 5 runs, one
   block): unchanged from rounds 4 and 4b -- the cycle fly leaves the table (9.2 of 16), hops (0.50 / fly) and raises GF
   (31.9 vs 25.2 Hz); the object / loom / feeding rows have never run with the cycle.
4. **A free-walking compass room under the cycle** (`cx_shift.py` room mode) -- unchanged.
5. **The FeCO walking-mean rate as a ledger number.** Still only the 10-150 Hz bracket exists; 86-87 Hz and the cycle's
   realised amplitude 0.948 are both consequences of the laws, not measurements, and the modulation-only control's value
   is now derived from the former rather than assumed.
6. **DNa02_R and the clean yaw SD at 6 v 6.** Both C v L rows sit at z +2.6-2.9 with the exact p at or near the floor:
   the differences (+0.094 Hz, +0.59 deg/s) replicate across three batches, the z does not, and the reference arm's
   between-run SD is the denominator each time. A claim on either row needs more runs of L, not more of C.

## 10. Files, provenance and reproduction

* Code (this task): `flyverse/body.py` (`LegCycle.flat_amplitude_value`, default 1.0), `scripts/probe_vnc_drive.py` (the
  `level3` family, `ARM_CYCLE_KW` / `cycle_kwargs_of`, `--flat-amplitude-value` on `room` / `compass`, `attach_cycle`'s
  cycle keywords, the `sense.cycle` / `sense.cycle_overrides` block, `plan`'s cycle flags and its refusal for an underived
  cycle parameter, `ARM_ORDER` as a tuple, the two-character arm regex in `load_room`, `per_seed.csv` in `pairs`, and the
  key-list dedupe), `tests/test_body_cycle.py` (`FlatAmplitudeValueTests`), `out/vncd6/` (`derive_flat_amplitude.py`,
  `predeclare.py`, `tree_state.py`, `verify_runs.py`, `level_model.py`, `per_seed_lists.py`), this audit.
  `flyverse/senses.py`, `brain.py`, `optic.py`, `motor.py`, `batch_sim.py`, `batch_body.py` and
  `tests/test_proprioception.py` are untouched.
* Data: `out/vncd6/` -- `batch.sh`, `predeclared.json` (stamped 07:47:38Z) and `predeclared_archive/`, `tree_state.json`
  / `tree_state_check.json`, `flat_amplitude_derivation.json`, `cal/` (2 CPU calibration runs), `smoke/` (the CPU M2 smoke
  and `pytest_cpu.txt`), `submitted_at.txt`, 24 room runs x 5 files, `run_verification.csv`, `arm_levels.csv`,
  `precondition_checks.csv`, `analysis/` (`decision_table.csv`, `pairwise.csv`, `per_seed.csv`, `per_seed_lists.txt`,
  `fractions.csv`, `sided_frames.csv`, `room_table.csv`, `dna02_decompose_summary.csv`, `level_model_r2*`,
  `trace_{C,M2,L}_vs_A.json`, `decompose_*`, `paths_*`), `post_analysis_sha.json`; console `out/vncd6_cluster.log`
  (`vncd6-2ad71d`: 24 job(s), 0 failed, 21.5 min). Every room JSON carries `provenance` (resolved params, the compiled-W
  md5, the 52-file `source_fingerprint`, `execution.device` / `device_name`, seeds, the block key), the `sense` block with
  its cycle block, and `started_utc` / `finished_utc`.
* Reducer identity (`post_analysis_sha.json`): five of the six predeclared reducers are byte-identical to their stamped
  sha256. **One changed after the first `pairs` run: `scripts/probe_vnc_drive.py`, a one-line dedupe of the key list in
  `cmd_pairs`** (`DNa02_L_hz` / `DNa02_R_hz` were listed as a room key and again as a watch key, one run value, and
  `per_seed.csv` carried 48 duplicate rows). The change alters no value: `decision_table.csv` before and after the change
  are identical (`analysis/decision_table_before_dedupe.csv` is kept) and `pairwise.csv` after equals the before-file
  with its duplicate rows dropped; the room-job code path is untouched, so every run was produced by the stamped reducer.
  The shipped `decision_table.csv`, `pairwise.csv` and `per_seed.csv` were produced by the POST-change reducer, sha256
  `5a288184a371a30adaf7e623e0ed1f5a2a3f3e6bda8627399530d750484e3ee4` (`post_analysis_sha.json` `post_analysis`), the
  whole `pairs` step having been re-run on the same fetch and every derived artefact re-emitted, as docs/INTERP.md
  10.4 item 11 requires; the stamped hash `f636d330...` is the one the 24 room jobs ran, which every run JSON confirms
  independently through its 52-file `source_fingerprint` (`flyverse/body.py` `1f873d32...` = the stamped hash).
* Reproduce: `python scripts/probe_vnc_drive.py plan --family level3 --dir out/vncd6 --runs 6 --compass-seeds 0 --draws 0
  --minutes 120 --name vncd6` (writes the 24-job `batch.sh`); `bash out/vncd6/batch.sh`; then on CPU
  `python out/vncd6/verify_runs.py`, `python scripts/probe_vnc_drive.py analyse --dir out/vncd6 --out out/vncd6/analysis
  --trace-arms C,M2,L` (~10 min), `python scripts/probe_vnc_drive.py pairs --dir out/vncd6 --out out/vncd6/analysis
  --analysis out/vncd6/analysis --predeclared out/vncd6/predeclared.json`, `python out/vncd6/level_model.py`,
  `python out/vncd6/per_seed_lists.py`. `level_model.py` run on `out/vncd5` as both calibration and target reproduces the
  round-4b skeptic's arm-level fit exactly (b +0.1848 +- 0.0355, c -0.0648 +- 0.0285; C's residual +3.916 / +5.073).

## 11. Validation (CPU, `CUDA_VISIBLE_DEVICES=-1`; this desktop)

* `pytest tests/test_body_cycle.py tests/test_proprioception.py tests/test_bit_identity.py -q`: **36 passed, 5 subtests
  passed** (`out/vncd6/smoke/pytest_cpu.txt`): the three new bit-identity tests and the unchanged shipped golden.
* CPU smoke of the M2 arm (2 flies x 0.5 s): exit 0, the JSON's `sense.cycle` reads `flat_amplitude True,
  flat_amplitude_value 0.948`, `sense.cycle_overrides {flat_amplitude_value: 0.948}`.
* Two CPU calibration runs (4 flies x 12 s): M2 - C chordotonal +0.09 Hz, hair +0.02, campaniform 0.00, |amp L-R| 0.0000
  vs 0.0188 -- the prediction the GPU batch then realised at -0.76 / -0.41 / +0.03 Hz.
* Batch verification before any verdict (`verify_runs.py`): 24 runs, 6 per arm, cuda / B200, family level3, one spec per
  arm, the cycle block per arm, blocks r0..r5, 5 artefacts each, one md5, one fingerprint, the stamp 37 s before the
  first `started_utc`; preconditions P1-P6 all PASS.
* The level-model script was validated on the previous batch first (section 10); the Holm arithmetic is checkable from
  the table (0.0021645 x 7 = 0.01515).
* `tree_state.py --check` after the analysis: every shipped file's sha256 UNCHANGED except `scripts/probe_vnc_drive.py`
  (the dedupe, section 10); the room-job path is untouched by it.

## Report

```yaml
summary: >
  Round 4c ran the modulation-only leg-cycle arm AT THE CYCLE'S REALISED AMPLITUDE (M2: LegCycle(flat_amplitude=True,
  flat_amplitude_value=0.948), a new opt-in LegCycle parameter defaulting to 1.0 and read only under flat_amplitude; the
  value derived on CPU from round 4b's cycle-arm recordings, walking-frame run mean 0.947728 +- 0.000044 rounded to
  0.948) against the cycle arm C, the level reference L (mn_ref 8.84) and the shipped path A, in ONE house submission
  (vncd6-2ad71d, 24 jobs, 0 failed, 21.5 min, B200; 4 arms x 6 seeds so that the m = 7 Holm family is satisfiable:
  6 v 6 floor 0.0021645 x 7 = 0.0152). Predeclared 07:47:38Z (archived) against the earliest run's own started_utc
  07:48:15Z; every precondition PASSED from the recordings: M2 sits at C's level on all three leg channels (chordotonal
  86.65 vs 87.40, -0.76 Hz; hair plate -0.41; campaniform +0.03, and that campaniform gap is the two arms' differing
  airborne fraction rather than a channel difference -- on non-airborne frames both read 24.998 Hz, M2 - C +0.0005)
  with its amplitude exactly 0.948000 on every walking frame and |amp L-R| 0.000000 (C: 0.9478 / 0.0175). RESULT.
  (1) F4 IS CLOSED: M2 v C is null on all seven primaries (AN04B003_L -0.20 Hz z -0.5, AN04B003_R -0.27 z -0.6,
  DNa02_L -0.037 z -2.0, DNa02_R +0.000, clean yaw SD +0.07, straightness +0.03, DNa02 L-R -0.037 z -1.6), and it is a
  null against the effect the arm was built to exclude rather than an absence of power: at 6 v 6 a row is called only
  at |z| >= 3, i.e. at three times C's own between-run SD -- 1.26 Hz (AN04B003_L), 1.44 (AN04B003_R), 0.055 (DNa02_L),
  0.041 (DNa02_R), 0.60 deg/s (clean yaw SD), 0.114 (straightness), 0.071 (DNa02 L-R) -- and round 4b's M-over-C excess
  (+1.62 / +1.65 Hz AN04B003, +0.093 DNa02_L) clears every one of them. The amplitude / turn law therefore adds no
  drive at the relay or at DNa02 beyond the per-phase modulation; round 4b's excess was its +6.0 Hz of level, and the
  +0.66 / +0.75 Hz structure-residual excess its skeptic found (R2) is -0.12 / -0.13 Hz and null at the right level
  (M2 residual +3.99 / +5.15 vs C +4.10 / +5.28 on a level model calibrated on round 4b's three steady arms with
  arm-level SEs b +0.1848 +- 0.0355, c -0.0648 +- 0.0285, transferring to this batch's L to -0.10 / +0.03 Hz). A
  WITHIN-BATCH calibration is not available: L is the only steady-input arm in this batch, two (arm, side) means
  against the model's four parameters, so the cross-batch application is the only option rather than a preference, and
  it is a joint extrapolation -- the calibration's six arm-side points lie near a line (corr(chord, hair) +0.925) and
  the cycle arms sit -9.4 (M2) to -9.8 (C) Hz of hair plate off it while this batch's L sits on it.
  (2) THE STRUCTURE TERM IS THE MODULATION: M2 v L is CALLED (Holm 0.015) on AN04B003_L +3.92 (z +30.0), AN04B003_R
  +6.85 (z +67.2), DNa02_L +0.111 (z +5.0) and straightness +0.339, with an arm that cannot carry a turn term, at the
  same chordotonal level (+0.66 Hz); the level-corrected pooled structure term is +4.60 +- 0.31 Hz without the
  amplitude law and +4.72 +- 0.32 with it (full-covariance propagation over the arm-level fit, where corr(b, c) is
  -0.951; the independent-SE form printed in section 7's table overstates the L-side corrections and understates the
  R-side and pooled ones), C v L being CALLED on the same four rows (+4.12 / +7.12 / +0.149 / +0.308) for the third
  batch. M2 is the cycle arm on every rate fraction ((X - A) / (C - A): AN04B003 0.99 / 0.99, DNa02_L 0.94, DNa02_R
  1.00, yaw SD 1.02, DNa02-active 0.96) where L is 0.69-0.88. (3) THE TURN TERM OWNS THE SIDED DNa02 SIGNAL, AND AS A
  SIDEDNESS RATHER THAN A RATE: E[DNa02 L-R | chord L-R > 0] - E[. | < 0] is -0.338 +- 0.031 Hz under C against
  -0.091 +- 0.012 under M2 (z +7.9, result; L -0.007) and corr(DNa02 L-R, chord L-R) -0.142 vs -0.032, with the same
  AN04B003 antiphase (-3.60 vs -3.63). The two arms do NOT see the same afferent waveform: the alternation SIZE is the
  same (mean per-frame |chord L-R| 12.36 vs 12.16 Hz, null) and so is the fast tripod term (SD 14.02 vs 13.99), but the
  SLOW component (longer than one step cycle) differs 2.6-fold (mean |slow| 1.65 vs 0.63 Hz; SD 2.78 vs 1.55) and is
  yaw-locked only under C (corr(yaw, slow) -0.779 vs +0.009). DNa02's sidedness follows the slow term alone
  (E[DNa02 L-R | slow > 0] - E[. | < 0] -1.552 C vs +0.066 M2; on the fast term -0.159 vs -0.123) while the relay
  follows the fast term and is identical in both arms (-3.750 vs -3.756): the amplitude / turn law CREATES a slow,
  yaw-locked sided afferent signal that an arm with |amp L-R| exactly 0 cannot have, and DNa02 -- not the relay --
  integrates it. It moves no window rate and no behavioural primary. (4) BOOKKEEPING: DNa02_R and the clean yaw SD
  over L reproduce as DIFFERENCES in all three batches (DNa02_R +0.103 / +0.107 / +0.094; yaw SD +0.572 / +0.490 /
  +0.588) and as verdicts in some -- DNa02_R is result in rounds 4 and 4b (z +16.6, +7.8) and null here (z +2.9), the
  clean yaw SD is result only in round 4 (z +7.1) and null in 4b and 4c (z +2.8, +2.6) -- with L's own between-run SD
  the denominator every time (0.081 -> 0.172 -> 0.224 deg/s); C v A is CALLED on all seven. INTEGRITY: every per-seed
  list is pasted from a script-emitted file (analysis/per_seed_lists.txt <- per_seed.csv,
  level_model_r2_residual_runs.csv), and the independent skeptic pass checked all 39 of them against those two sources
  with zero mismatches; one predeclared reducer changed after the first pairs run (a one-line key-list dedupe in
  probe_vnc_drive.py cmd_pairs), the whole pairs step was re-run on the same fetch and every derived artefact
  re-emitted as docs/INTERP.md 10.4 item 11 requires (decision_table.csv byte-identical before and after, the
  before-file kept), so the shipped analysis artefacts carry the post-change hash 5a288184... while the 24 room jobs
  ran the stamped f636d330... Nothing is adopted and no default moved.
skeptic:
  source: "independent Opus pass, 2026-09-15"
  verdict: "mostly sound"
  integrity: "all 39 per-seed lists verified against per_seed.csv / level_model_r2_residual_runs.csv"
key_claims:
  - "Batch: ONE submission vncd6-2ad71d on house, `24 job(s), 0 failed (21.5 min)`, 24 room JSON + 96 npz + 24 consoles; every run device cuda / NVIDIA B200 / family level3 / its spec / its sense block with the LegCycle block (M2 flat_amplitude True, flat_amplitude_value 0.948; C False, 1.0) / blocks fam_r0..fam_r5; one compiled md5, one 52-file fingerprint. Predeclared 2026-09-15T07:47:38Z (archived copy predeclared_archive/predeclared_2026-09-15T074738Z.json, byte-identical to predeclared.json) against the earliest started_utc 07:48:15Z; submitted_at 07:47:47Z."
  - "The parameter: body.LegCycle.flat_amplitude_value, default 1.0, read only under flat_amplitude; passed as --flat-amplitude-value (ARM_CYCLE_KW / cycle_kwargs_of), recorded in the run JSON's sense.cycle / sense.cycle_overrides / leg_cycle_params. Tests (36 passed, 5 subtests, CPU): the default cycle with the value unread and round 4b's flat cycle with =1.0 are bit-identical over 12 BatchSim steps (rtol=atol=0); the value is the constant on every walking leg at any yaw, 0 standing / airborne; the shipped golden unchanged. senses.py untouched."
  - "Derivation (CPU, out/vncd6/flat_amplitude_derivation.json): per-run walking-frame amplitude of out/vncd5 arm C [0.947665 0.947714 0.947744 0.947786 0.947730], run mean 0.947728 +- 0.000044, the law at the recorded 9.794 mm/s gives 0.947396; chosen 0.948. The p05 / p50 / p95 0.9451 / 0.9471 / 0.9523 and the 0.1 % above 1.0 in that file are of the per-frame L/R MEAN amplitude; of the PER-LEG amplitude the flat law replaces they are 0.9193 / 0.9473 / 0.9753 with 0.63 % of leg-frames above 1.0 (round 4b's skeptic pass's C4 numbers), so the chosen value is unchanged but the distribution the constant replaces is three times wider than the L/R-mean percentiles suggest. CPU calibration pair predicted M2 - C chordotonal +0.09 Hz; the batch realised -0.76."
  - "Preconditions from the recordings, all PASS: P1 M2 v C chordotonal -0.757 / hair -0.409 / campaniform +0.028 Hz (tolerances 3 / 3 / 1.5; the campaniform gap is the arms' differing airborne fraction, not a channel difference -- on non-airborne frames both read 24.998 Hz, M2 - C +0.0005); P2 M2 amplitude 0.948000 (one distinct value) and |amp L-R| 0.000e+00 against C's 0.947750 / 0.01751; P3 L v C chordotonal -1.418; P4 brackets; P5 configuration; P6 ordering. Level gaps M2 v L / C v L: hair plate -9.94 / -9.53, campaniform -24.68 / -24.71 Hz (the round-4b mismatch, carried and corrected)."
  - "F1 M2 v C (the question), all null: DNa02_L -0.0372 (z -2.0, p 0.026, holm 0.18), DNa02_R +0.0001 (z +0.0), clean yaw SD +0.074 (z +0.4), straightness +0.031 (z +0.8), DNa02 L-R -0.037 (z -1.6), AN04B003_L -0.202 (z -0.5), AN04B003_R -0.267 (z -0.6). In the predeclared words: the amplitude law adds NO drive at the relay or at DNa02 beyond the modulation; round 4b's M-over-C excess was the level. What the family could have seen: at 6 v 6 a row is called only at |z| >= 3, three times C's own between-run SD -- 1.26 Hz (AN04B003_L), 1.44 (AN04B003_R), 0.055 (DNa02_L), 0.041 (DNa02_R), 0.60 deg/s (clean yaw SD), 0.114 (straightness), 0.071 (DNa02 L-R) -- and round 4b's M-over-C excess clears every one of them."
  - "F2 M2 v L: CALLED on AN04B003_L +3.921 (z +30.0), AN04B003_R +6.851 (z +67.2), DNa02_L +0.1115 (z +5.0), straightness +0.339 (z +15.2); DNa02_R +0.094 (z +2.9, null), yaw SD +0.66 (z 2.96, null), DNa02 L-R +0.017 (null). F3 C v L: CALLED on DNa02_L +0.149 (z +6.7), straightness +0.308, AN04B003 +4.122 / +7.118; DNa02_R +0.094 (z +2.9, null), yaw SD +0.588 (z +2.6, p 0.0087, null), DNa02 L-R +0.055 (z +2.4, null). F3's predeclared first branch names DNa02_R as well, so F3 is the rule's MIXED case, not its first branch: the pair reproduces on DNa02_L and both AN04B003 sides for a third batch and does not reproduce as a verdict on DNa02_R at 6 v 6, at an unchanged size (+0.094 here against +0.103 in round 4 and +0.107 in round 4b; `result` there at z +16.6 and +7.8, `null` here at z +2.9). F4 C v A: CALLED on all seven."
  - "Level model (declared secondary, cross-batch calibration on vncd5's L/U/K six arm-side means; arm-level SEs b +0.1848 +- 0.0355, c -0.0648 +- 0.0285, residual standard error 0.128 = sqrt(RSS / 2 df), the RMSE over the six fitted points 0.074; this batch's L residual -0.099 / +0.033 = the transfer): AN04B003 residuals M2 +3.987 / +5.155, C +4.103 / +5.280 (per-run lists in per_seed_lists.txt); M2 - C residual difference -0.116 (z -0.6) / -0.125 (z -0.5), null, level correction -0.086 / -0.142 (pooled -0.114 +- 0.016). Pooled structure over L: M2 +4.60 +- 0.31, C +4.72 +- 0.32 under full-covariance propagation (corr(b, c) = -0.951 on the six-point design, so the independent-SE form printed in the section-7 table overstates the L-side corrections, +-0.46 -> +-0.22 and +-0.44 -> +-0.23, and understates the R-side and pooled ones, +-0.31 -> +-0.41, +-0.33 -> +-0.43, +-0.28 -> +-0.31, +-0.27 -> +-0.32); decompositions +0.15 + 0.64 + 4.60 = 5.39 and +0.29 + 0.61 + 4.72 = 5.62. A WITHIN-BATCH calibration is impossible here: L is the only steady-input arm, two (arm, side) means against four parameters, so the cross-batch application is the only option rather than a preference, and it remains a joint extrapolation (the six calibration points near a line, corr(chord, hair) +0.925, the cycle arms -9.4 / -9.8 Hz of hair plate off it) that corrects for no part of the -24.7 Hz campaniform mismatch. DNa02 model (not robust, reported only): M2 - C -0.036 (L, z -2.1, null), +0.003 (R, inside the calibration error)."
  - "Sided rows: E[DNa02 L-R | chord L-R > 0] - E[. | < 0] = -0.3376 +- 0.0314 (C) vs -0.0911 +- 0.0124 (M2) vs -0.0072 (L), M2 v C +0.247 z +7.9 result (per seed C [-0.3594 -0.2937 -0.3028 -0.3616 -0.3638 -0.3440], M2 [-0.0672 -0.0920 -0.0958 -0.0907 -0.1005 -0.1002]); corr(DNa02 L-R, chord L-R) -0.142 vs -0.032; AN04B003 antiphase -3.601 vs -3.633 (null); corr(yaw, chord L-R) -0.147 (C) vs +0.015 (M2) vs +0.270 (L). The two arms have the same alternation size (mean per-frame |chord L-R| 12.36 vs 12.16 Hz, null; a pairwise.csv room key on the post-skip non-airborne mask, 12.27 vs 12.07 on the clean-frame mask) but a slow, yaw-locked afferent L-R component 2.6x larger under C (mean |slow chord L-R| 1.65 vs 0.63 Hz, SD 2.78 vs 1.55, corr with yaw -0.779 vs +0.009), which is the component DNa02's sidedness follows (E[DNa02 L-R | slow > 0] - E[. | < 0] -1.552 vs +0.066, against -0.159 vs -0.123 on the fast tripod term the relay follows and shares). The turn term owns the sided DNa02 signal at the matched level by CREATING that slow signal, and it is a sidedness, not a rate."
  - "Decomposition: AN04B003's drive onto DNa02_L +354.7 (M2) / +357.8 (C) / +294.1 (L) mV/s, DNa02_R +353.3 / +357.3 / +249.7; direct SNpp45 +128.2 / +129.0 / +167.6; PS059 -204 / -206 / -209; every row of M2 and C within 1 %. Chain rates M2 v C all null (SNpp45 -0.40, IN13B001 -0.67 / -0.52, PS059, PS196_b, GLNO); in the shipped arm A the hair-plate afferent SNpp45 is silent (0.00 Hz, no sense attached) and every other row sits at its A-arm floor."
  - "Behaviour: M2 v C null on every row (yaw SD 7.90 vs 7.83, median |yaw| 1.57 vs 1.57, drift +1.19 vs +1.36, straightness 0.843 vs 0.812, flies off 7.8 vs 9.2, hops 0.39 vs 0.50, GF 29.1 vs 31.9, step 7.70 vs 7.79 Hz) except the leg-MN L-R (+0.018 Hz, z +3.4 on a 0.0055 Hz reference SD, a 0.4 % magnitude). Fractions (X - A) / (C - A), paired by seed: DNa02_L L 0.73 / M2 0.94; DNa02_R 0.68 / 1.00; yaw SD 0.88 / 1.02; AN04B003_L 0.82 / 0.99; AN04B003_R 0.69 / 0.99; DNa02-active 0.71 / 0.96; straightness 2.75 / 0.86."
  - "Reducer identity: five of six predeclared reducers byte-identical after the analysis; scripts/probe_vnc_drive.py changed by a one-line dedupe of cmd_pairs' key list after the first pairs run (DNa02_L_hz / DNa02_R_hz listed twice, 48 duplicate rows in per_seed.csv); decision_table.csv identical before and after (before-file kept), pairwise.csv equal after dropping the duplicates, the room-job path untouched. tree_state --check: every other shipped file unchanged. The SHIPPED decision_table.csv, pairwise.csv and per_seed.csv were produced by the POST-change reducer, sha256 5a288184a371a30adaf7e623e0ed1f5a2a3f3e6bda8627399530d750484e3ee4 (post_analysis_sha.json post_analysis), the whole pairs step having been re-run on the same fetch and every derived artefact re-emitted as docs/INTERP.md 10.4 item 11 requires; the stamped hash f636d330... is the one the 24 room jobs ran, which every run JSON confirms independently through its 52-file source_fingerprint (flyverse/body.py 1f873d32... = the stamped hash)."
  - "What rounds 4, 4b and 4c jointly say (the skeptic pass's closing paragraph, the sanctioned reading): three batches of the same protocol agree on an attribution and on its size -- against a level-matched round-2 transducer at the same chordotonal level the leg cycle raises AN04B003 by +3.7 to +7.1 Hz per side (result, CALLED in the one batch whose Holm family was satisfiable) and DNa02_L by +0.09 to +0.15 Hz (result in all three). Round 4c splits that term in two: the RATE part is the per-phase modulation (M2 reproduces C on every rate row, fractions 0.94-1.00 against L's 0.69-0.88, null on all seven primaries, structure term +4.60 +- 0.31 against +4.72 +- 0.32), and the SIDED part is the amplitude / turn law, which is a sidedness and not a rate. Two rows stay unsettled across all three batches: DNa02_R (+0.09-0.11 Hz) and the clean yaw SD (+0.49-0.59 deg/s) over L, because the denominator is the control arm's own between-run scatter. Nothing is adopted."
files_written:
  - flyverse/body.py (LegCycle.flat_amplitude_value, default 1.0; the flat branch reads it; docstring)
  - scripts/probe_vnc_drive.py (family level3, LEVEL3_FLAT_AMPLITUDE, ARM_CYCLE_KW / cycle_kwargs_of / CYCLE_KW_FLAGS, --flat-amplitude-value on room and compass, attach_cycle's cycle keywords and refusals, sense.cycle / sense.cycle_overrides, plan's cycle flags, ARM_ORDER as a tuple, the two-character arm regex, per_seed.csv, the pairs key-list dedupe)
  - tests/test_body_cycle.py (FlatAmplitudeValueTests, 3 tests)
  - docs/audits/level_controls_r2.md
  - out/vncd6/ (derive_flat_amplitude.py, flat_amplitude_derivation.json, predeclare.py, predeclared.json, predeclared_archive/, tree_state.py, tree_state.json, tree_state_check.json, verify_runs.py, level_model.py, per_seed_lists.py, batch.sh, submitted_at.txt, cal/, smoke/, 24 room runs x 5 files, run_verification.csv, arm_levels.csv, precondition_checks.csv, analysis/, post_analysis_sha.json, *_console.txt)
  - out/vncd6_cluster.log
api:
  - "body.LegCycle(flat_amplitude=True, flat_amplitude_value=v): amp_i = v for every walking leg (0 standing and airborne); default v 1.0 = round 4b's M; unread when flat_amplitude is False."
  - "probe_vnc_drive.py room --family level3 --arm A|L|M2|C [--flat-amplitude-value V]: V defaults to ARM_CYCLE_KW (M2: 0.948) else LegCycle's default; refused without the leg_cycle_flat token. plan refuses an underived cycle parameter."
  - "run JSON: sense.cycle = vars(LegCycle) (flat_amplitude, flat_amplitude_value, ...), sense.cycle_overrides; pairs writes analysis/per_seed.csv (key, arm, seed, file, value) beside pairwise.csv; PAIRS_BY_FAMILY['level3'] = M2vC, M2vL, CvL, CvA, LvA, M2vA."
  - "out/vncd6/level_model.py --cal <steady batch> --dir <batch>: the arm-level level model (six arm-side means) evaluated on another batch's arms, with transfer, per-run residuals, residual-difference compare and propagated slope uncertainty."
validation:
  - "pytest tests/test_body_cycle.py tests/test_proprioception.py tests/test_bit_identity.py: 36 passed, 5 subtests (CPU); the golden unchanged"
  - "CPU smoke of M2 (exit 0, cycle block recorded); two CPU calibration runs predicting the M2 - C level gap (+0.09 Hz) before submission"
  - "batch: 24 job(s), 0 failed (21.5 min); 24 JSON + 96 npz + 24 consoles; verify_runs.py: every run cuda / B200 / level3 / correct spec, sense and cycle block / blocks r0..r5 / 5 artefacts / one md5 / one fingerprint; preconditions P1-P6 PASS from the recordings before any verdict; the stamp precedes the first run by 37 s"
  - "level_model.py reproduces the round-4b skeptic's arm-level fit exactly when run on out/vncd5 (b +0.1848 +- 0.0355, c -0.0648 +- 0.0285, C residual +3.916 / +5.073) and transfers to this batch's L to -0.10 / +0.03 Hz"
  - "the Holm family is satisfiable (0.0021645 x 7 = 0.0152) and CALLED rows in F2, F3, F4; F1 calls none, which is the answer"
  - "every per-seed list is pasted from analysis/per_seed_lists.txt (script-emitted from per_seed.csv / level_model_r2_residual_runs.csv); post_analysis_sha.json and tree_state_check.json record the reducer state"
recommendations:
  - "Close F4 in NOTES: the amplitude / turn law adds no drive at AN04B003 or DNa02; the cycle's rate effect is the per-phase modulation (+4.6 Hz at the relay over the level control, +0.11-0.15 Hz of DNa02_L), and the turn term's only footprint is the per-frame sided DNa02 signal."
  - "Give half_width_m and lit.walk.outer_leg_step_ratio_in_turn ledger values before reading the sided DNa02 signal as a mechanism: it is the only row the amplitude law owns and it scales with an unmeasured constant."
  - "Derive mn_ref_hz, hair_plate_max_hz and campaniform_load_hz together (round 4b item 3) so the structure term over L (+4.6 / +4.7 Hz) is a matched measurement rather than a slope-corrected one (+0.6 +- 0.3 Hz of correction)."
  - "Rows at z 2.6-2.9 with p at the floor (DNa02_R, the clean yaw SD of C v L) need more runs of the REFERENCE arm; a 6 v 6 batch did not move them off the z threshold because L's between-run SD is the denominator."
  - "The suite under all+leg_cycle (5 runs, one block) before any default discussion; the object / loom / feeding rows have still never run with the cycle."
open_questions:
  - "What closes the loop the slow afferent term implies? The amplitude law writes a slow (longer than one step cycle), yaw-locked L-R component into the afferents -- mean |slow chord L-R| 1.65 Hz under C against 0.63 under M2, corr with yaw -0.779 against +0.009 -- and DNa02's sidedness follows that component alone (-1.552 vs +0.066 Hz of conditioned L-R swing) while the relay follows the fast tripod term and sees none of it. The loop (yaw -> amp -> slow afferent L-R -> DNa02 L-R -> yaw) is shown here and not opened, and its gain scales with the unmeasured half_width_m."
  - "DNa02_R at +0.09-0.11 Hz and the clean yaw SD at +0.49-0.59 deg/s over L reproduce as DIFFERENCES in all three batches (DNa02_R +0.103 / +0.107 / +0.094; yaw SD +0.572 / +0.490 / +0.588) and as verdicts in some: DNa02_R is `result` in rounds 4 and 4b (z +16.6, +7.8) and `null` here (z +2.9); the clean yaw SD is `result` only in round 4 (z +7.1) and `null` in 4b and 4c (z +2.8, +2.6). The denominator is L's own between-run SD every time (0.081 -> 0.172 -> 0.224 deg/s); what sets it is not known."
  - "M2 sits 0.037 Hz BELOW C on DNa02_L (p 0.026, five of six seeds) with the relay equal: a sign opposite to a drive, inside the null, and possibly the sided signal's rectification at the window mean; not separable at 6 v 6."
  - "The structure term over the level control still carries the hair-plate / campaniform mismatch of the round-2 law, and no within-batch calibration can remove it: L is the only steady-input arm in this batch, two (arm, side) means against the model's four parameters, so the correction is cross-batch and jointly extrapolated (the six calibration points near a line, corr(chord, hair) +0.925; the cycle arms -9.4 / -9.8 Hz of hair plate off it) by necessity rather than by preference. The campaniform's contribution is known only from a 752-cell single cell (round 4b: null) and no room arm has matched it at a fixed chordotonal level."
```

## Skeptic pass (independent, 2026-09-15)

Everything below was recomputed on this desktop (CPU, `CUDA_VISIBLE_DEVICES=-1`) from `out/vncd6/room_*_body.npz`,
`*_flies.npz`, `*_rec.npz` and `room_*.json`, and from `out/vncd4` / `out/vncd5` for the cross-batch rows, with my own
reducer (my own clean-frame rule, same-mask DNa02, chain rates from `_flies.npz`, my own exact Mann-Whitney / z / Holm),
and by re-running the three test modules. Nothing was imported from `scripts/probe_vnc_drive.py`. No cluster job was
submitted; nothing needed one. Scratch scripts: `scratchpad/recompute2.py` (primaries, channels, amplitude),
`sided.py` (per-frame sided rows + a slow/fast split of the afferent waveform), `levelfit.py` (the level model refit);
outputs `recomputed.json`, `sided.json`, `levelrows.json`.

---

### Refuted

**R1. "The afferent alternation is the same" is not an argument for the sided-DNa02 attribution -- the afferent
waveforms are NOT matched.** Section 8.3 reading (ii) and the Report's `key_claims` sided-rows bullet support
"the turn term owns the per-frame sided DNa02 signal" with "the same AN04B003 antiphase (-3.60 vs -3.63) and the same
afferent alternation (mean per-frame |chordotonal L-R| 12.36 C, 12.16 M2)", i.e. *same input, different DNa02
sidedness*. That reading does not hold. Splitting the commanded chordotonal L-R at one step cycle (a 12-frame centred
boxcar, 8.3 Hz stepping at 100 Hz frames) on clean walking frames, run mean over the six runs of each arm
(`scratchpad/sided.py`):

| statistic of commanded chordotonal L-R | L | M2 | C | M2 - C (z) |
|---|---|---|---|---|
| SD, total | 14.14 | 14.05 | 14.28 | -0.24 (-1.4) |
| SD, FAST (tripod) | 5.07 | **13.99** | **14.02** | -0.03 (-0.2) |
| SD, SLOW (> 1 step cycle) | 12.25 | **1.55** | **2.78** | **-1.23 (-13.2)** |
| mean \|SLOW\| (Hz) | 15.31 | **0.63** | **1.65** | **-1.02 (-18.5)** |
| corr(yaw, SLOW) | +0.290 | **+0.009** | **-0.779** | +0.788 (+74) |
| corr(DNa02 L-R, SLOW) | -0.000 | **+0.009** | **-0.533** | +0.542 (+54) |
| corr(DNa02 L-R, FAST) | +0.012 | -0.033 | -0.039 | +0.006 (+2.1) |
| E[DNa02 L-R \| SLOW > 0] - E[. \| < 0] | -0.022 | **+0.066** | **-1.552** | +1.619 (+24.5) |
| E[DNa02 L-R \| FAST > 0] - E[. \| < 0] | +0.027 | **-0.123** | **-0.159** | +0.035 (+2.2) |
| E[AN04B003 L-R \| FAST > 0] - E[. \| < 0] | +0.061 | **-3.756** | **-3.750** | -0.006 (-0.2) |
| E[AN04B003 L-R \| SLOW > 0] - E[. \| < 0] | +2.320 | +0.994 | +0.589 | +0.405 (+4.9) |

The mean |L-R| is dominated by the fast tripod term, which the two arms share to 0.8 %. The whole difference lives in
the slow term, which C has (mean |slow| 1.65 Hz, yaw-locked at -0.78) and M2 cannot have (|amp L-R| exactly 0). DNa02's
sidedness follows the slow term and only the slow term; the relay follows the fast term and is identical in both arms --
which is exactly why "the antiphase is the same" is true and irrelevant to the DNa02 row. **The mechanism the audit
states is right; the supporting sentence is wrong.** The amplitude/turn law *creates a slow yaw-locked sided afferent
signal*, and DNa02 (not the relay) integrates it. It is not that DNa02 reads matched afferents differently.

**R2. Section 8.2: "No cell in these tables is silent in any arm" is contradicted by its own table.** `chain_SNpp45_hz`
in arm A is **exactly 0.0000 Hz** (`analysis/pairwise.csv`, and the 8.2 table prints `SNpp45 | 0.00 | 55.65 | 45.77 |
46.16`). A has no sense attached, so the hair-plate afferents are undriven. Every other row of 8.2 has a non-zero A
value (AN04B003 0.60 / 0.31, GLNO 0.05, PS196_b 0.21, IN13B001 10.64 / 6.50, PS059 8.80 / 4.93).

**R3. Section 3's spread statistics are of the wrong array (and the same sentence mixes two arrays).** The sentence
names "the per-leg amplitude ... (`cyc__amp_L` / `cyc__amp_R`)" and then quotes "5th / 50th / 95th percentile
0.9451 / 0.9471 / 0.9523 ... 0.1 % of frames above 1.0". `out/vncd6/derive_flat_amplitude.py` line 37 sets
`a = 0.5 * (aL[walk] + aR[walk])`, so those three percentiles and `frac_above_1` are of the **per-frame L/R mean**,
while `amp_max` in the same record is taken over the per-leg arrays. Recomputed over the per-leg amplitudes (my own
mask, post-skip, on the ground, `step_hz > 0`, `out/vncd5/room_C_r*_body.npz`): **p05 / p50 / p95 = 0.9193 / 0.9473 /
0.9753 and 0.63 % of leg-frames above 1.0** -- three times the quoted spread and six times the quoted tail, and exactly
the round-4b skeptic pass's independently checked C4 numbers ("0.919-0.975, max 1.547, 0.6 % of frames above 1.0").
The run mean (0.947728 +- 0.000044), the per-run maxima (1.356 / 1.436 / 1.515 / 1.547 / 1.359) and |amp L-R| 0.0170
all reproduce exactly, so **the chosen value 0.948 is unaffected**; what is misstated is how wide the distribution is
that the constant replaces.

**R4. The Report's open question gets the cross-batch verdict history wrong for DNa02_R.** `open_questions` says
DNa02_R and the clean yaw SD "reproduce as differences in three batches and as verdicts in none since round 4".
Recomputed C v L with one reducer from each batch's own `room_table.csv` `*_runs`:

| key | vncd4 (5 v 5) | vncd5 (5 v 5) | vncd6 (6 v 6) |
|---|---|---|---|
| DNa02_R | +0.1030, z +16.6, p 0.0079 -> **result** | +0.1068, z +7.8, p 0.0079 -> **result** | +0.0941, z +2.9, p 0.0022 -> null |
| clean yaw SD | +0.5718, z +7.1, p 0.0079 -> **result** | +0.4896, z +2.8, p 0.0079 -> null | +0.5881, z +2.6, p 0.0087 -> null |

DNa02_R **was** a verdict in round 4b (z +7.8) -- as section 6 F3 of this very audit says correctly ("DNa02_R is
`result` in rounds 4 and 4b"). The Report contradicts section 6. Only the yaw SD matches the "verdicts in none since
round 4" wording.

**R5. Section 4.1's C amplitude "min / max" are means of the per-run minima and maxima, not a min and a max.** The
cell reads `0.947750 (0.452 / 1.574; 0.017509)` under the header "amp on walking frames (min / max; |L-R|)".
`arm_levels.csv` carries these as `amp_min_mean` / `amp_max_mean`, i.e. the run mean of each run's extremum.
Recomputed per run, C's per-leg minima are `[0.4956 0.3023 0.5599 0.5440 0.0603 0.7520]` and maxima
`[1.5388 1.7267 1.4686 1.4886 1.9787 1.2420]`: **the pooled range is 0.060 .. 1.979**, four times wider on the low
side than the printed "0.452". (M2's column is exact: one distinct value 0.948000 in every run.)

**R6. Section 7's slope-uncertainty formula drops a -0.95 slope correlation and therefore understates two of the three
pooled corrections.** The audit propagates `sqrt((dChord x se_b)^2 + (dHair x se_c)^2)`. On the six-point arm-level design
`corr(b-hat, c-hat) = -0.951` (my refit: `cov(b,c) = -9.6e-4`, se_b 0.0355, se_c 0.0285), so the independent-SE form is not the
right one. Full-covariance propagation `sqrt(v^T Cov v)`:

| pair / side | correction | audit +- | full-cov +- |
|---|---|---|---|
| M2 v C, pooled | -0.114 | 0.029 | 0.016 |
| M2 v L, L | -0.165 | 0.461 | **0.220** |
| M2 v L, R | +1.729 | 0.306 | **0.412** |
| M2 v L, pooled | +0.782 | 0.281 | **0.307** |
| C v L, L | -0.079 | 0.444 | **0.228** |
| C v L, R | +1.872 | 0.329 | **0.431** |
| C v L, pooled | +0.896 | 0.274 | **0.321** |

The pooled structure terms become **+4.60 +- 0.31 (M2) and +4.72 +- 0.32 (C)**. No reading changes; the stated error does.

**R7. "Point RMSE 0.128" is the residual standard error, not an RMSE.** On the six arm-side means with 2 residual df,
`sqrt(RSS/dof) = 0.1279` (the audit's number) but `sqrt(mean resid^2) = 0.0739`. Both are legitimate; only one is an
RMSE, and round 4b's audit quoted the six-point fit's RMSE (0.1059 for its run-level fit) in the same role.

**R8. Section 8.3's table attributes a row to a file that does not carry it, and to the wrong mask (rule 28's
neighbourhood).** The table is headed `analysis/sided_frames.csv`; `sided_frames.csv` holds 22 keys and
`mean per-frame |chordotonal L-R|` is not among them. That row is `commanded_chordotonal_absLR_hz` from
`analysis/pairwise.csv`, a room key computed on the **post-skip non-airborne** mask, not on the clean-walking-frame
mask the table's caption declares. On the clean-frame mask it reads **12.07 (M2) / 12.27 (C)**, not 12.16 / 12.36.
(Values and verdict are otherwise right: diff -0.2046, z -0.88, null.)

---

### Confirmed

**C1. Stamps, archive, one submission, provenance -- clean (a, b).** `predeclared.json` and
`predeclared_archive/predeclared_2026-09-15T074738Z.json` are **byte-identical** (sha256
`02998712042c5aac549ea8d203fb1be488f6d74bbfba73a1b7614d26e58d8f72` for both). Ordering from the run JSONs' own
absolute timestamps: derivation 07:31:46Z < **stamp 07:47:38Z** < submit 07:47:47Z < earliest `started_utc`
**07:48:15Z** (37 s) < latest `finished_utc` 08:00:47Z; zero runs start before the stamp. `batch.sh` contains
**one** `cluster_run.py` call with 24 `room` commands, each preserving the python exit code (`st=$?; tail -4; exit $st`,
INTERP 10.4 item 4); the receipt in `out/vncd6_cluster.log` is `24 job(s), 0 failed (21.5 min) run dir
/mnt/beegfs/neurome/runs/vncd6-2ad71d`; 24 room JSON + 24 consoles + 96 npz. Every run: `device cuda`,
`device_name NVIDIA B200`, `family level3`, its arm's spec (`A` none / `L` `all` at `mn_ref_hz` **8.84**, hair 100,
load 50 / `M2` `all+leg_cycle+leg_cycle_flat`, tokens `leg_cycle,leg_cycle_flat`, `sense.cycle.flat_amplitude True`,
`flat_amplitude_value 0.948`, `cycle_overrides {flat_amplitude_value: 0.948}` / `C` `all+leg_cycle`,
`flat_amplitude False`, `flat_amplitude_value 1.0`), block `fam_r0..fam_r5`, **one** compiled-connectome md5
`ef23cc27bea13be7f6a96f3c04fd3737`, **one** 52-file `source_fingerprint` (identical in all 24), wall 143-716 s.
The fingerprint's `flyverse/body.py` sha256 is `1f873d32af435e18e31bd3e50606c8a2ab1ea63b30e2acb2ed80a8e53337d8ba` --
the stamped reducer hash -- so the runs demonstrably ran the stamped body code.

**C2. Every one of the 28 decision-table rows reproduces exactly (c).** My own reducer -- post-skip non-airborne mask
per fly for `DNa02_L/_R/_LR`; my own clean-frame rule (post-skip, on the table top before and after, no airborne frame
within 50 frames, |yaw| <= 720 deg/s) for the yaw SD; net/path per fly over the whole run for straightness; every cell
of the type per side from `_flies.npz` for AN04B003 -- reproduces every per-run value in `analysis/per_seed.csv` to
**max |Delta| = 0.000e+00** on all seven keys x 24 runs, and every mean, SD, diff, z, exact p, Holm p and verdict of
`analysis/decision_table.csv`: F1 all seven `null` (AN04B003_L -0.2016 z -0.48, _R -0.2672 z -0.56, DNa02_L -0.0372
z -2.02 p 0.0260 holm 0.1818, DNa02_R +0.0001, yaw SD +0.0744, straightness +0.0306, DNa02 L-R -0.0373 z -1.57);
F2 four CALLED (DNa02_L +0.1115 z +5.01, straightness +0.3390 z +15.16, AN04B003_L +3.9205 z +30.01, _R +6.8510
z +67.17) with DNa02_R +0.0942 z +2.89 and yaw SD +0.6625 z **2.96** `null`; F3 four CALLED (+0.1487 / +0.3083 /
+4.1222 / +7.1182) with DNa02_R z +2.89 and yaw SD +0.5881 z +2.62 p 0.0087 `null`; F4 seven CALLED. `p_floor(6,6) =
2/C(12,6) = 0.00216450`, `x 7 = 0.015152 <= 0.05`; my independent Holm gives 0.0152 on every floor row. The
predeclared JSON declares `m = 7` and the same seven keys for all four families.

**C3. THE INTEGRITY CHECK PASSES -- all 39 per-seed lists are the data (d).** I machine-extracted every bracketed
per-seed list in the audit's prose and tables (28 in section 5, 4 in section 7, 7 in section 8.3) and compared each
value, mean and SD against its named source: 33 against `analysis/per_seed.csv` (column `value`) and 6 against
`analysis/level_model_r2_residual_runs.csv` (column `residual`). **Zero mismatches** at the printed precision.
`per_seed_lists.txt` itself matches both sources. The round-4b failure (R1 there: four fabricated arm lists built to
carry the published mean) does **not** recur; the audit's rule-28 claim is true. The prose readings off those lists
are also right: M2's r3 AN04B003_L 23.2708 is above C's r0 / r1 / r5 (22.9417 / 22.6076 / 22.9314); C's DNa02_L sits
+0.0072 .. +0.0963 above M2's in five of six seeds and -0.0138 below in r5; C's yaw-SD r2 7.6077 sits below L's r0
7.6227 (so F3's yaw-SD separation is incomplete and p is 0.0087, not the floor), while every M2 and every C DNa02_R run
is above every L run (p at the floor).

**C4. The preconditions reproduce exactly from the recordings under the reducer's declared mask (e).** Flat post-skip
mean per channel, run mean +- SD over 6 (`verify_runs.py`'s convention, which the audit states as "post-skip frames"):
L 85.986 +- 0.607 (92.589 / 79.119; L-R **+13.470**), M2 **86.647 +- 0.808** (86.658 / 86.637; +0.021), C **87.404 +-
1.616** (87.229 / 87.587; -0.357); hair plate 56.594 / **46.658** / 47.067 (L-R +9.140 / +0.003 / -0.204); campaniform
49.573 / 24.893 / 24.864; haltere 18.858 / 16.557 / 16.633. **P1 M2 - C = -0.757 / -0.409 / +0.028 Hz** against +-3 /
+-3 / +-1.5 -- PASS. **P2**: M2's per-leg amplitude on my own walking mask is **0.948000, one distinct value, min = max =
0.948000, |amp L-R| = 0.000000e+00** in all six runs; C's is 0.947750 with |amp L-R| **0.017509** -- PASS.
**P3** C - L chordotonal +1.418 (<= 10) -- PASS. Level gaps: M2 v L **+0.662 / -9.936 / -24.681 / -2.300**;
C v L **+1.418 / -9.527 / -24.709 / -2.225**. Walking fraction 0.9251 (M2) / 0.9345 (C), step 8.284 / 8.292 Hz,
stance 0.7515 / 0.7512. Every published number in 4.1, 4.2 and the "level gaps" row is exact.
*Refinement, not a defect:* the +0.028 Hz campaniform gap is the arms' differing airborne fraction, not a channel
difference -- on non-airborne frames both arms read 24.998 Hz (M2 - C = +0.0005). On that mask the chordotonal gap is
slightly larger (-0.846) and the hair-plate gap -0.457; P1 passes on either mask with room to spare.

**C5. The level model refits exactly, including the transfer and the vncd5 self-check (f).** My own OLS on the six
(arm, side) means of `out/vncd5`'s L, U, K: **a_L +5.9763, a_R +5.2131, b +0.1848 +- 0.0355, c -0.0648 +- 0.0285**,
2 residual df; run-level (pseudo-replicated, 30 rows) b +0.1921 +- 0.0128, c -0.0697 +- 0.0103 -- all as printed.
Transfer to this batch's L: residual **-0.0989 +- 0.0412 (L side), +0.0328 +- 0.0407 (R side)**. Per-run AN04B003
residuals M2 **+3.9867 +- 0.1156 / +5.1545 +- 0.1484**, C **+4.1026 +- 0.1791 / +5.2795 +- 0.2393**; residual differences
M2 - C **-0.1159 (z -0.6, p 0.31) / -0.1250 (z -0.5, p 0.24) / pooled -0.1205 (z -0.6)**, all `null`; pooled structure
terms **M2 +4.6037, C +4.7241** (z +129.3 / +132.7). Level corrections -0.086 / -0.142 (M2 v C), -0.165 / +1.729
(M2 v L), -0.079 / +1.872 (C v L). Decompositions check: +0.147 + 0.636 + 4.604 = 5.386 (M2 v L, raw +5.386) and
+0.287 + 0.609 + 4.724 = 5.620 (C v L, raw +5.620). DNa02 model: b +0.0020 +- 0.0068, c +0.0020 +- 0.0055, transfer
+0.0066 / +0.0101, residuals M2 +0.1587 / +0.0997, C +0.1942 / +0.0967, M2 - C -0.0355 (z -2.1, `null`) / +0.0030.
Run on `out/vncd5` as its own target, **C's residual is +3.9162 / +5.0732** -- the round-4b skeptic's arm-mean-fit
numbers (+3.916 / +5.073), as claimed. *A within-batch refit is genuinely unavailable: L is the only steady-input arm
here, two arm-side means against four parameters. The cross-batch application is the only option, not a preference --
but the audit never says so.*

**C6. The per-frame sided rows reproduce exactly (g).** My own recomputation (my clean-frame mask, lag 0, per fly then
run mean) matches `analysis/sided_frames.csv`'s `*_runs` to the CSV's 4-dp rounding on every key I checked:
`dna02LR_given_chordLR_pos_minus_neg` L -0.0072 +- 0.0311 / **M2 -0.0911 +- 0.0124 / C -0.3376 +- 0.0314**
(M2 v C +0.2465, z **+7.9**, p 0.0022, `result`); `corr_dna02LR_chordLR` +0.0041 / -0.0318 / -0.1418;
`an04LR_given_chordLR_pos_minus_neg` +1.9646 / **-3.6334 / -3.6008** (M2 v C -0.0325, z -1.0, `null`);
`corr_an04LR_chordLR` +0.1912 / -0.3600 / -0.3447; `corr_yaw_chordLR` +0.2700 / +0.0150 / **-0.1471**;
`yaw_signed_mean_deg_s` +3.3742 / +1.1902 / +1.3553. `amp_LR_yaw_corr` is **-0.4189 under C and undefined (NaN) under
M2**, as stated. **The substantive claim -- the amplitude/turn term owns the sided DNa02 signal and the attribution
survives the level match -- is confirmed and, by R1's decomposition, localised.**

**C7. Fractions, the DNa02 decomposition, the chain rates and the behaviour table all reproduce (h).** Paired by seed,
`(X - A)/(C - A)` from my own per-run values equals `analysis/fractions.csv` to <= 3e-16 on all 14 rows: DNa02_L
L 0.7327 +- 0.0401 / M2 0.9350 +- 0.0679; DNa02_R 0.6834 / 1.0022; yaw SD 0.8849 / 1.0181; straightness 2.7481 / 0.8641;
DNa02 L-R 0.7895 / 0.8625; AN04B003_L 0.8176 / 0.9914; _R 0.6949 / 0.9889; DNa02-active 0.7128 / 0.9574. Every
section-8.1 cell matches `dna02_decompose_summary.csv` (AN04B003 onto DNa02_L +9.2 / +294.1 / **+354.7** / **+357.8**;
onto DNa02_R +4.8 / +249.7 / +353.3 / +357.3; SNpp45 0 / +167.6 / +128.2 / +129.0; nets -304 / +61 / +97 / +106 and
-362 / -270 / -166 / -162). Section 8.2's chain rates and section 8.4's behaviour rows match `pairwise.csv` exactly,
including the one M2 v C `result` outside the sided rows (`leg_LR_hz` +0.01830, z **+3.36**, on C's SD 0.005456).

**C8. The new parameter is opt-in, keyword-only, and off the default path (j).** The `body.py` diff adds one dataclass
field (`flat_amplitude_value: float = 1.0`, last in the field list) and changes one line inside
`if self.flat_amplitude:` (`np.ones_like(v_leg)` -> `np.full_like(v_leg, float(self.flat_amplitude_value))`); the
`else` branch (the amplitude law) and every timing/tripod/load line are untouched, and `full_like(x, 1.0)` is the same
float array as `ones_like(x)`. No call site constructs `LegCycle` positionally (grep over the repo: every construction
is keyword-only), so appending the field is safe. `PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest
tests/test_body_cycle.py tests/test_proprioception.py tests/test_bit_identity.py -q` -> **36 passed, 5 subtests passed**
on this desktop, including the shipped golden and the three new bit-identity tests (`rtol=0, atol=0` over 12 `BatchSim`
steps for default-with-value-unread and flat-with-1.0).

**C9. The post-stamp reducer edit is handled the way INTERP 10.4 item 11 requires, and I verified the no-effect claim
myself (a).** `analysis/decision_table.csv` and `decision_table_before_dedupe.csv` are **byte-identical** (both sha256
`3acf6f02c41e7258fc7974a75c15e3873ad742aeb83af62e6c7a85f5468c2864`), and `pairwise.csv` (195 rows) equals
`pairwise_before_dedupe.csv` (197 rows) with its two duplicate rows dropped -- order preserved, header identical, exact
row-for-row equality. I recomputed all six predeclared reducer hashes: five are byte-identical to the stamp;
`scripts/probe_vnc_drive.py` is now `5a288184a371a30adaf7e623e0ed1f5a2a3f3e6bda8627399530d750484e3ee4` against the
stamped `f636d3307edb33ee830260f72eb4b36e265c679e19886bd1cb2c82a5dab6af2a`, exactly as `post_analysis_sha.json`
records. **On the rule:** the governing rule is 10.4 item 11, not "the hashes are recorded at the stamp" -- the stamp
covers the *reading rules*, and item 11 says a change to the analysis code after a Result obliges re-running the
analysis on the same fetch and re-emitting every derived artifact. That is what happened (the edit landed after the
first `pairs` run, `pairs` was re-run, every `analysis/` artifact downstream of it carries the 01:22 re-emit, and the
before-files are kept). So a post-data reducer edit **is** acceptable here; the right handling is the one taken, plus
one thing the audit omits -- naming the hash that produced the SHIPPED artifacts (the post-change `5a28818...`), because
under item 11 the Result records the analysis code that produced it, not the stamped one. The room-job path is
independently covered: every run JSON's own 52-file fingerprint carries the stamped `body.py` hash.

**C10. The F1 null is informative at the size it was built to exclude -- the audit under-sells it.** At 6 v 6 a row is
called only at |z| >= 3, i.e. at 3 x the reference arm's between-run SD: **1.26 Hz (AN04B003_L), 1.44 (AN04B003_R),
0.055 (DNa02_L), 0.041 (DNa02_R), 0.60 deg/s (clean yaw SD), 0.114 (straightness), 0.071 (DNa02 L-R)**. Round 4b's
M-over-C excess (+1.62 / +1.65 Hz AN04B003, +0.093 DNa02_L) clears every one of those thresholds. The design could
therefore have seen the effect it set out to kill, and did not.

---

### Corrections (exact replacement text)

**1. Section 8.3, reading (ii) -- replace the clause after "round 4b: -0.334 vs -0.107 with M at the wrong level)":**

> with the same antiphase at the relay (-3.63 M2, -3.60 C) and the same alternation SIZE (mean per-frame
> |chordotonal L-R| 12.16 vs 12.36 Hz, `commanded_chordotonal_absLR_hz` in `analysis/pairwise.csv`, post-skip
> non-airborne frames; 12.07 vs 12.27 on the clean-frame mask) -- but **not the same afferent waveform**. Splitting the
> commanded chordotonal L-R at one step cycle (a 12-frame centred boxcar) leaves the two arms with the same FAST tripod
> term (SD 13.99 M2 vs 14.02 C; mean |.| 11.95 vs 12.05) and a SLOW term that differs 2.6-fold (mean |slow| 0.63 vs
> 1.65; SD 1.55 vs 2.78, z -13.2) and is yaw-locked only under C (corr(yaw, slow) +0.009 M2 vs -0.779 C). DNa02's
> sidedness follows the slow term alone: E[DNa02 L-R | slow > 0] - E[. | < 0] is +0.066 (M2) against -1.552 (C), while
> on the fast term the two arms agree (-0.123 vs -0.159); the relay is the mirror image, identical on the fast term
> (-3.756 vs -3.750) and small on the slow one. **The amplitude / turn law does not change how DNa02 reads a matched
> afferent input: it CREATES a slow, yaw-locked sided afferent signal that an arm with |amp L-R| exactly 0 cannot
> have, and DNa02 -- not the relay -- integrates it.** That is what "the turn term owns this row" means here.

(The Report's sided-rows `key_claims` bullet needs the same fix: replace "and the same afferent alternation
(12.36 vs 12.16 Hz)" with "the same alternation size (12.36 vs 12.16 Hz) but a slow, yaw-locked afferent L-R component
2.6x larger under C (mean |slow chord L-R| 1.65 vs 0.63 Hz), which is the component DNa02's sidedness follows".)

**2. Section 8.2, last sentence -- replace:**

> Every cell of these tables fires in all three sense arms. In the shipped arm A the hair-plate afferent SNpp45 is
> silent (0.00 Hz: no sense is attached), and every other row sits at its A-arm floor (AN04B003 0.60 / 0.31, GLNO 0.05,
> PS196_b 0.21, IN13B001 10.64 / 6.50, PS059 8.80 / 4.93 Hz).

**3. Section 3 -- replace "5th / 50th / 95th percentile 0.9451 / 0.9471 / 0.9523, max 1.36-1.55, 0.1 % of frames above
1.0" with:**

> the percentiles and the above-1.0 fraction in `flat_amplitude_derivation.json` are of the per-frame L/R MEAN
> amplitude (5th / 50th / 95th 0.9451 / 0.9471 / 0.9523; 0.1 % above 1.0); of the PER-LEG amplitude the flat law
> replaces they are **0.9193 / 0.9473 / 0.9753 with 0.63 % of leg-frames above 1.0** (per-run max 1.36-1.55), the
> round-4b skeptic pass's C4 numbers. The mean is the same on either array, so the chosen value is unchanged; the
> distribution the constant replaces is three times wider than the L/R-mean percentiles suggest.

**4. Report `open_questions`, second entry -- replace the first clause with:**

> DNa02_R at +0.09-0.11 Hz and the clean yaw SD at +0.49-0.59 deg/s over L reproduce as DIFFERENCES in all three
> batches (DNa02_R +0.103 / +0.107 / +0.094; yaw SD +0.572 / +0.490 / +0.588) and as verdicts in some: DNa02_R is
> `result` in rounds 4 and 4b (z +16.6, +7.8) and `null` here (z +2.9); the clean yaw SD is `result` only in round 4
> (z +7.1) and `null` in 4b and 4c (z +2.8, +2.6). The denominator is L's own between-run SD every time (0.081 ->
> 0.172 -> 0.224 deg/s); what sets it is not known.

**5. Section 4.1, the amplitude column -- replace the header and add one clause:**

> | ... | amp on walking frames (run mean of the per-run min / max; \|L-R\|) |
> ... C: 0.947750 (0.452 / 1.574, the mean of the per-run extrema -- pooled over the six runs the per-leg amplitude
> spans 0.060 .. 1.979; 0.017509)

**6. Section 7, reading 3 -- replace the uncertainty sentence:**

> the level correction of a pair is b x dChord + c x dHair over the pair's realised level gaps, and its uncertainty is
> `sqrt([dChord dHair] Cov(b, c) [dChord dHair]^T)` with the ARM-LEVEL covariance. The two slopes are strongly
> anti-correlated on this design (`corr(b, c) = -0.951`), so the independent-SE form
> `sqrt((dChord se_b)^2 + (dHair se_c)^2)` is not the right one: it OVERstates the L-side corrections (+-0.46 ->
> +-0.22, +-0.44 -> +-0.23) and UNDERstates the R-side and the pooled ones (+-0.31 -> +-0.41, +-0.33 -> +-0.43;
> pooled +-0.28 -> +-0.31 for M2 v L and +-0.27 -> +-0.32 for C v L). **Pooled structure terms: M2 +4.60 +- 0.31,
> C +4.72 +- 0.32.** The M2 v C correction is small enough that the distinction does not matter there
> (-0.114 +- 0.016).

**7. Section 7, the fit line -- replace "point RMSE 0.128" with:** `residual standard error 0.128 (sqrt(RSS / 2 df);
the RMSE over the six fitted points is 0.074)`.

**8. Section 8.3, table caption -- add:**

> (the `mean per-frame |chordotonal L-R|` row is `commanded_chordotonal_absLR_hz` from `analysis/pairwise.csv`, a room
> key on the post-skip non-airborne mask, not a `sided_frames.csv` clean-frame statistic; on the clean-frame mask it
> reads 12.07 M2 / 12.27 C)

**9. Section 7, opening paragraph -- add (the cross-batch justification and round 4b's R8, which this round inherits
unchanged):**

> A within-batch calibration is not available: L is the only steady-input arm in this batch, two (arm, side) means
> against the model's four parameters, so the cross-batch application is the only option rather than a preference. It
> is also a JOINT extrapolation, as round 4b's skeptic pass found (R8) and as this batch repeats: the calibration's six
> arm-side points lie near a line, `corr(chord, hair) = +0.925`, `hair = 0.979 x chord - 28.73` with 2.65 Hz of scatter
> about it, and the cycle arms sit **-9.4 (M2) to -9.8 (C) Hz of hair plate off that line** while this batch's L sits on
> it (-0.84 / +3.20). The residual DIFFERENCE M2 - C is nearly free of this (the pair's own gaps are -0.57 / -0.95 of
> chordotonal and -0.31 / -0.51 of hair plate, worth -0.09 / -0.14 Hz), which is why F1's level correction is safe; the
> +4.60 / +4.72 structure terms over L are not, and neither corrects for the -24.7 Hz campaniform mismatch (round 4b
> C8: on the steady arms the campaniform is a near-binary arm indicator and is not separable from the hair plate).

**10. Section 6, F3 -- add after the arrow line:**

> The rule's first branch names DNa02_R as well, and DNa02_R is `null` here, so this is the rule's MIXED case, not its
> first branch: the pair reproduces on DNa02_L and on both AN04B003 sides for a third batch and does not reproduce as a
> verdict on DNa02_R at 6 v 6, at an unchanged size (+0.094 here against +0.103 in round 4 and +0.107 in round 4b).

**11. Section 5 or 6, F1 -- add (it makes the null much stronger than it currently reads):**

> What F1 could have seen: at 6 v 6 a row is called only at |z| >= 3, i.e. at three times C's own between-run SD --
> 1.26 Hz (AN04B003_L), 1.44 (AN04B003_R), 0.055 (DNa02_L), 0.041 (DNa02_R), 0.60 deg/s (clean yaw SD), 0.114
> (straightness), 0.071 (DNa02 L-R). Round 4b's M-over-C excess (+1.62 / +1.65 Hz of AN04B003, +0.093 of DNa02_L)
> clears every one of them, so the null is a null against the effect the arm was built to exclude, not an absence of
> power.

**12. Section 10, the reducer-identity bullet -- add one sentence:**

> The shipped `decision_table.csv`, `pairwise.csv` and `per_seed.csv` were produced by the POST-change reducer, sha256
> `5a288184a371a30adaf7e623e0ed1f5a2a3f3e6bda8627399530d750484e3ee4` (`post_analysis_sha.json` `post_analysis`), the
> whole `pairs` step having been re-run on the same fetch and every derived artefact re-emitted, as docs/INTERP.md
> 10.4 item 11 requires; the stamped hash `f636d330...` is the one the 24 room jobs ran, which every run JSON confirms
> independently through its 52-file `source_fingerprint` (`flyverse/body.py` `1f873d32...` = the stamped hash).

**13. Section 4.2, P1 row -- add one clause:**

> (the +0.028 Hz campaniform gap is the two arms' differing airborne fraction, not a channel difference: on non-airborne
> frames both arms read 24.998 Hz, M2 - C +0.0005)

---

### Verdict

**MOSTLY SOUND.** Every number this audit's conclusions rest on reproduces from the recordings with an independent
reducer: the 28 decision-table rows exactly, the preconditions exactly, the level model exactly (slopes, intercepts,
transfer, residuals, differences, corrections and the vncd5 self-check), the sided rows exactly, the fractions, the
decomposition, the chain rates and the behaviour table exactly. The stamps are clean, the archive is byte-identical to
`predeclared.json`, there was one submission, and -- the thing round 4b failed -- **all 39 per-seed lists quoted in prose
are the data**. The post-stamp reducer edit is documented, provably outcome-free (byte-identical decision tables), and
handled as INTERP 10.4 item 11 requires. The three headline claims stand: F4 is closed (M2 v C `null` on all seven, at
a design that could have seen round 4b's excess); the structure term is the modulation (+4.6 vs +4.7 Hz, the same
number with and without the amplitude law); and the turn term owns the per-frame sided DNa02 signal. What is refuted
is supporting material, not conclusions: the "same afferent alternation" sentence is the wrong argument for the right
claim (the slow, yaw-locked afferent component differs 2.6-fold and is where the whole DNa02 signal lives); one
sentence in 8.2 contradicts its own table; section 3's dispersion statistics are of the L/R mean, not the per-leg
amplitude; the Report's open question mis-states DNa02_R's verdict history; a "min / max" is a mean of extrema; the
slope-uncertainty formula drops a -0.95 correlation; and section 7 drops round 4b's R8 extrapolation caveat, which this
batch inherits unchanged. None of that moves a verdict, and all of it is fixable in prose.

**What rounds 4, 4b and 4c jointly say about the leg cycle.** Three batches of the same protocol now agree on an
attribution and on its size: against a level-matched round-2 transducer at the same chordotonal level, the leg cycle
raises the ascending relay AN04B003 by **+3.7 to +7.1 Hz per side** (`result`, CALLED in the one batch whose Holm
family was satisfiable) and DNa02_L by **+0.09 to +0.15 Hz** (`result` in all three) -- a replicated attribution, not a
single-batch finding. Round 4c splits that term cleanly in two. The rate part is the **per-phase modulation**: an arm
that keeps the phase/stance structure and holds the amplitude at the cycle's own realised 0.948 reproduces the cycle
arm on every rate row (fractions 0.94-1.00 against the level control's 0.69-0.88) and is `null` against it on all seven
primaries, with a level-corrected structure term of +4.60 +- 0.31 Hz against the cycle's +4.72 +- 0.32 -- so round 4b's
M-over-C excess was its +6.0 Hz of level, and the amplitude / turn law adds **no drive** at the relay or at DNa02. The
sided part is the **amplitude / turn law**, and it is a sidedness, not a rate: the law writes a slow, yaw-locked L-R
asymmetry into the afferents (mean |slow chord L-R| 1.65 Hz against the flat arm's 0.63, corr with yaw -0.78 against
+0.01) which DNa02 integrates into a -0.34 Hz tripod-conditioned L-R swing (against -0.09), while the relay, which
follows the fast tripod term, sees nothing of it -- and no window rate, yaw SD, straightness or behavioural primary
moves. Two rows stay unsettled across all three batches: DNa02_R (+0.09-0.11 Hz) and the clean yaw SD (+0.49-0.59
deg/s) over the level control reproduce as differences every time and as verdicts only sometimes, because the
denominator is the control arm's own between-run scatter. Nothing is adopted; `LegCycle.flat_amplitude_value` is an
opt-in labelled-control parameter whose default path is bit-identical to the shipped one, and the structure term still
carries a -9.5 Hz hair-plate and -24.7 Hz campaniform mismatch that is corrected at slopes extrapolated ~9.5 Hz off
their calibration manifold, and an unmeasured `half_width_m` that scales the one row the turn term owns.
