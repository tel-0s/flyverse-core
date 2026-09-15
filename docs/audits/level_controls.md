# The three controls that split "structure": per-phase modulation, the unmatched channels, and the DC sidedness (thread level controls, round 5)

Generator: `scripts/probe_vnc_drive.py --family level2` (`plan` / `analyse` / `pairs` CPU; `room` GPU), the CPU
derivation `out/vncd5/derive_channel_match.py`, the CPU checks `out/vncd5/verify_runs.py` and `out/vncd5/level_model.py`,
and the single-cell probe `scripts/probe_an04b003_single_cell.py`. Data: `out/vncd5/` (ONE submission,
`out/vncd5/batch.sh` -> run `vncd5-2ffbc3` on the house cluster, B200, console `out/vncd5_cluster.log`:
**30 job(s), 0 failed (22.6 min)**); analysis `out/vncd5/analysis_console.txt`, `out/vncd5/analysis/pairs_console.txt`,
`out/vncd5/verification_console.txt`, `out/vncd5/level_model_console.txt`. Predeclaration
`out/vncd5/predeclared.json`, stamped **2026-09-15T04:54:59Z** against the earliest run's own
`started_utc` **04:56:11Z** -- an absolute timestamp inside the run JSONs, so round 4's "the ordering rests on file
mtimes" caveat is closed. Working-tree record `out/vncd5/tree_state.json`; derivation
`out/vncd5/channel_match_derivation.json` with its CPU calibration runs `out/vncd5/cal/`; CPU smokes and the CPU test
run `out/vncd5/smoke/`. Every number below is a run mean +- SD over the five runs of an arm (16 flies x 55 s window
each; runs are the replicate unit); verdicts are `flyverse.interp.common.compare` on runs (5 v 5, floor p 0.0079).
Nothing in `out/vncd5` is compared row by row with `out/vncd4` or `out/vncd3`; their numbers appear as context and are
labelled so.

## 0. Answer

**The three differences are separated, and they are not the same size.** Round 4 could not say whether the C-over-L
excess was the per-phase modulation, the unmatched hair plate / campaniform, or the round-2 law's DC sidedness. With
one control per difference:

1. **The DC sidedness is the level control's DRIFT, not its DNa02.** The UNSIDED control U (every leg cell on the
   side-mean leg-MN rate; realised chordotonal L-R **+0.000 Hz** against L's **+13.564**, hair-plate L-R 0.000 against
   +9.204, at 84.8 vs 86.1 Hz of chordotonal) leaves DNa02 where it was -- DNa02_L -0.049 (z -1.8, `null`), DNa02_R
   +0.006 (`null`), clean yaw SD -0.49 (z -2.8, `null`) -- while it halves the fly's left drift (+3.48 -> +1.74 deg/s),
   raises straightness 0.483 -> 0.714 (z +9.7), symmetrises the relay (AN04B003_L -1.01, AN04B003_R +0.85, both
   `result`, pooled -0.08) and cuts DNa02 L-R by a third (0.162 -> 0.108). Round 4's skeptic R2 -- "the straightness
   row is a sidedness row, not a structure row" -- now has its arm: **the round-2 law's DC bias owns the straightness
   and the drift and owns none of the DNa02 rate.**
2. **The unmatched hair plate is real, small and NEGATIVE; the campaniform is nothing.** The CHANNEL-MATCHED control K
   (`hair_plate_max_hz` 86.71, `campaniform_load_hz` 25.05, derived on CPU in section 3) realises hair plate
   **44.34** and campaniform **24.78** against C's 46.36 / 24.91 -- matched to -2.02 / -0.13 Hz -- and equalises the
   direct hair-plate term on DNa02_L (SNpp45/? **+128.5** mV/s against C's +127.1, where L has +167.9) while IN13B001
   falls 76.3 -> 61.6 Hz. But K v L is `result` NEGATIVE on AN04B003 (-0.94 / -0.61), the OPPOSITE of the
   disinhibition the predeclaration predicted, because matching those two channels costs K **8.7 Hz of chordotonal**
   through the afferent -> leg-MN loop (the predeclared side effect P1b). The size of the hair-plate route is
   therefore taken from the level model and the single cell, not from the pair: **-0.070 Hz of AN04B003 per Hz of hair
   plate** in the room (**-0.149** on the isolated cell). L's +10.28 Hz hair-plate excess **suppresses L's own
   AN04B003 by -0.72 Hz**, which therefore **accounts for +0.72 Hz of the +5.22 Hz pooled C-over-L difference
   (14 % of it)**; the chordotonal term is +0.01 (the channels were matched), and the remaining **+4.49 Hz** is what
   the levels do not explain. (+0.01 + 0.72 + 4.49 = 5.22.) The sign is the one round 4's skeptic argued for, and the
   route is not enough to explain the difference. The campaniform contributes nothing measurable (single cell: -0.055 Hz,
   z -0.3, `null`, for the whole 24.9 -> 49.7 Hz step).
3. **The per-phase modulation is what is left, and it is the large term.** A LEVEL MODEL fitted on the three
   steady-input arms (L, U, K; 3 arms x 2 sides x 5 runs = 30 side-observations spanning chordotonal 71.6-93.5 Hz and
   hair plate 41.0-61.6 Hz, no modulation anywhere in them) gives
   `AN04B003_side = a_side + 0.192 x chordotonal - 0.070 x hair_plate` and predicts **every steady arm to within
   0.11 Hz** (residuals -0.11 .. +0.08, every one `null`). Evaluated at the CYCLE arms' own realised levels it
   under-predicts them by **+3.88 / +5.00 Hz (C, L / R side)** and **+4.54 / +5.75 (M)** -- 37-54 x the fit's RMSE,
   and +3.6 .. +4.1 (C L-side) under a leave-one-arm-out refit. **At its own afferent level the cycle's ascending
   neuron fires ~4-6 Hz more than any steady transducer of that level does**, and the single-cell check names the
   mechanism directly: at the SAME per-cell chordotonal mean with IN13B001's rate clamped, modulating the input raises
   AN04B003 by **+1.63 Hz** (z +7.2, `result`).
4. **MOST of M's excess over C is level; whether the amplitude law adds drive is OPEN, and what the turn term owns is
   the sided signal.** The MODULATION-ONLY arm M (per-leg amplitude held at 1, |amp L-R| exactly 0.000 against C's
   0.0157) fires MORE than C (DNa02_L +0.093, AN04B003 +1.62 / +1.65), and it also carries +6.0 Hz more chordotonal,
   because amplitude 1.000 is above the law's realised 0.948. On the level model **about 60 % of M's AN04B003 excess
   over C is its extra 6.0 Hz of chordotonal** (predicted +0.956 / +0.904 of the observed +1.615 / +1.649). The rest
   is not: M's structure residual **exceeds** C's by +0.659 (L, z +3.8, `result`) and +0.745 (R, z +3.5, `result`),
   6-7 x the fit's RMSE, and on DNa02_L by +0.069 (z +4.0, `result`), which is essentially the whole +0.093 M-over-C
   DNa02_L difference. **This batch therefore cannot say that the amplitude law adds no drive**; what it can say is
   that most of M's excess is level, and that a flat amplitude is not only "no turn term" but also a different
   modulation waveform. The row the turn term does own remains the per-frame sided DNa02 signal (-0.334 C vs -0.107
   M), and it is not a bigger afferent alternation under C: mean per-frame |chordotonal L-R| is 12.13 Hz (C) against
   13.02 (M). The tripod-conditioned DNa02 L-R swing is **-0.334 Hz under C against -0.107 under M** (and 0.00 under
   every steady arm), and C is the straighter fly (0.857 vs 0.794).

**The decision rules, applied in their predeclared words, are in section 6.** Under `compare` the pattern above is
unambiguous; under the predeclared HOLM step **nothing is CALLED, and the reason is arithmetic, not evidence**: at
5 v 5 the exact Mann-Whitney floor is p 0.0079 and the declared family size is m = 7, so the smallest attainable
adjusted p is 0.0079 x 7 = **0.0556 > 0.05** and no member of any family can survive. That is a defect of THIS
round's predeclaration (round 4 used m = 5, where 0.0079 x 5 = 0.0397 passes), it was fixed in neither direction after
the fact, and every table below therefore reports the `compare` verdict, the z, the per-run scatter AND the
Holm-adjusted p, with "not called under the declared m = 7" stated wherever a row would otherwise read as called.
**This is not a new rule discovered here**: `docs/INTERP.md` 10.2 already requires the family to be sized from the
members that can move, and records object round 2 raising its arms from 5 to 6 runs for exactly that reason -- the
predeclaration step failed to apply a standing house rule. Section 10 item 1 restates it.

**What the round can now say about round 4's question.** With the chordotonal channel matched (C v L: 86.11 vs
86.05 Hz), C v L reproduces inside this batch on DNa02_L (+0.092, z +3.4), DNa02_R (+0.107, z +7.8), straightness
(+0.374) and AN04B003 (+3.73 / +6.72) and does NOT reproduce on the clean yaw SD (+0.49, z +2.8, `null`, against
round 4's +0.57 at z +7.1 -- the verdict flips, but the difference itself replicated to 86 % with complete rank
separation in both batches; section 9). Of that excess: **the DC sidedness owns the straightness row and none of the
DNa02 rate (U); the unmatched hair plate accounts for +0.72 Hz of the +5.22 Hz AN04B003 difference (the level
model's slope, sign-negative as the route predicts: it suppresses L's own rate); the per-phase modulation owns the
remaining ~4.49 Hz** -- so "the per-leg / per-phase STRUCTURE contributes beyond the LEVEL" is supported for the first
time, on the relay, with a number. On DNa02 itself the same model is too weak to carry the claim (its slopes are
0.0032 +- 0.0027 per Hz and the leave-one-arm-out residuals for C swing +0.197 .. +0.049): the DNa02 statement rests
on the pairs, where C exceeds every steady control (C v K +0.178, C v U +0.141, C v L +0.092, all `result` by
compare, none Holm-called).

**Under the project rule:** nothing is adopted. U, K and M are labelled controls; `unsided` and
`LegCycle.flat_amplitude` are opt-in and default OFF, with the shipped path bit-identical on CPU (section 2); the leg
cycle stays an opt-in body-model mechanism. What its adoption would still require is in section 10.

## 1. The question, and the three differences it has to separate

Round 4 (`docs/audits/level_matched_control.md`) ran the level-matched control L -- the round-2 transducer at
`mn_ref_hz` 8.84, chosen only so that its window-mean commanded chordotonal rate equals the leg-cycle arm C's -- and
found C v L `result` on DNa02_L (+0.127 Hz), DNa02_R (+0.103) and the clean yaw SD (+0.57 deg/s). Its own skeptic pass
(`## Skeptic pass`, R1-R2, and the corrected section 7 item 1) then showed that the decision could not be read as
"the per-leg / per-phase STRUCTURE contributes beyond the LEVEL", because **L differs from C in three ways at once**:

1. **the per-phase modulation** of the chordotonal input (under C a leg's command sweeps 20 -> 142 Hz within every
   128 ms step, per-leg SD 46.4 Hz with 0.81 of its variance in the 5-12 Hz band; under L it is steady, SD 23.9, 0.07);
2. **the unmatched hair plate (+9.7 Hz) and campaniform (+24.8 Hz)**, whose dominant route into DNa02 is
   sign-NEGATIVE -- `SNpp45 -> IN13B001 -| AN04B003 -> DNa02`, signs `+,-,+`, `gain_if_signed` -3.0e+04, the strongest
   three-step afferent walk in `paths_summary.csv`, with SNpp45 a hair-plate afferent (53 of the channel's 113 cells);
3. **the +13.6 Hz DC chordotonal L-R** (and +9.2 Hz hair-plate L-R) that the round-2 law derives from the leg-MN side
   bias at `mn_ref` 8.84, which the cycle arm does not have.

This round runs the three controls that separate them (round 4's section 7 item 1(a)(b)(c)), in ONE submission, with
the decision pairs and their reading rules stamped before submission
(`out/vncd5/predeclared.json`, stamped 2026-09-15T04:54:59Z), plus the single-cell mechanism check its item 6 asked for.

**Design (six arms x 5 brain seeds = 30 room jobs, blocks `fam_r<seed>`; no compass -- rounds 2-4 settled that
question: nothing moves the bump).**

| arm | spec | classification |
|---|---|---|
| A | shipped (no sense) | the reference |
| L | `'all'`, `mn_ref_hz` **8.84** | LABELLED CONTROL: round 4's level-matched control, re-run here so every pair is within-batch |
| **U** | `'all+unsided'`, `mn_ref_hz` 8.84 | LABELLED CONTROL (new token): every leg cell reads the SIDE-MEAN leg-MN rate, so the +13.6 Hz DC chordotonal L-R and the +9.2 Hz hair-plate L-R are gone at the same channel means |
| **K** | `'all'`, `mn_ref_hz` 8.84, `hair_plate_max_hz` **86.71**, `campaniform_load_hz` **25.05** | LABELLED CONTROL (two further hand-set parameters, derived in section 3): the realised hair-plate and campaniform means match the cycle arm's 47.06 / 24.89 Hz |
| **M** | `'all+leg_cycle+leg_cycle_flat'` | LABELLED CONTROL of the body-model mechanism (new flag): `body.LegCycle(flat_amplitude=True)` holds the per-leg amplitude at 1 while walking, so the phase / stance-swing modulation is kept and the amplitude law -- with it the `half_width_m` yaw term and the \|amp L-R\| turn term -- is removed |
| C | `'all+leg_cycle'` | the body-model mechanism (round 3's and round 4's arm C): the treatment |

Primaries per pair (7 keys, Holm m = 7 within each family): `DNa02_L_hz`, `DNa02_R_hz` (on the SAME clean frame mask
as `DNa02_LR_hz`), `yaw_sd_clean_deg_s`, `straightness`, `DNa02_LR_hz`, and `chain_AN04B003_L_hz` / `chain_AN04B003_R_hz`
-- the relay the whole argument turns on, read per side from the per-fly recordings of every cell of the type.
The seven predeclared families are **U v L** (is it the sidedness?), **K v L** (is it the unmatched channels?),
**C v K** (with all three matched, does structure still win?), **M v C** (does the amplitude / turn term matter?),
**C v U** and **M v K** (the two cleanest structure-vs-level contrasts) and **L v A** (the fraction bookkeeping).
Each family's decision rule is quoted verbatim in section 6.

## 2. The two new opt-in mechanisms, and their bit-identity tests

Both default OFF, both are LABELLED CONTROL constructions (they exist to separate the parts of a comparison; neither is
a candidate for a default), and the shipped path is bit-identical on CPU with both absent.

**(a) `unsided` (`flyverse/senses.py`, a token in `Proprioception.FLAGS`).** Under the round-2 MN-rate law the leg
channels read `clip(legMN_side / mn_ref_hz, 0, 1)`: the 271 left chordotonal cells read `leg_L`, the 259 right cells
`leg_R`, the 85 unsided cells the mean. With `unsided` in the spec, `rates()` replaces both side rates by their mean
`(leg_L + leg_R) / 2` before the law is applied, so EVERY leg cell reads the same drive. Nothing else changes: the
campaniform and haltere channels, every ceiling, every tonic and the cell sets are untouched. The token is refused
together with `leg_cycle` (under the cycle the leg channels do not read the MN rate at all) and refused without an
MN-rate leg channel. Four lines of code, one of them the law.

**(b) `LegCycle.flat_amplitude` (`flyverse/body.py`) and the matching spec token `leg_cycle_flat`.** The cycle's
`advance()` sets `amp_i = max(v - s_i * yaw * half_width_m, 0) * tau_stance / step_ref_m` -- ~0.95 in straight walking
and the only place the turn enters the afferents. With `flat_amplitude=True` the walking legs get `amp_i = 1` (standing
and airborne rows keep 0, as before) and nothing else moves: the phase, the stance flag, the step frequency, the stance
fraction and the per-side loads are the default's, bit for bit. The sense-side token `leg_cycle_flat` selects nothing by
itself -- it tells `probe_vnc_drive.attach_cycle` to build `LegCycle(flat_amplitude=True)` and puts the fact in the
spec, so a run JSON is self-describing. It is refused without `leg_cycle`.

**Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`).** `tests/test_proprioception.py::UnsidedTokenTests` pins that with the token
ABSENT the round-2 law is unchanged (the sided values 80 / 10 / 45 / 80 Hz of the existing law), that two `BatchSim`s
under `'all'` -- one built by the constructor, one with the sense rebuilt as the probe rebuilds it -- are bit-identical
over four frames and one fed frame (`torch.testing.assert_close(..., rtol=0, atol=0)`), that with the token every leg
cell reads the side mean (38 Hz on both sides at leg_L 12 / leg_R 0 against the sided 66 / 10), that the campaniform and
haltere Poisson fields are untouched, and the grammar (refusals). `tests/test_body_cycle.py::FlatAmplitudeTests` pins
that the default flag is off and `vars(LegCycle()) == vars(LegCycle(flat_amplitude=False))`, that the default amplitude
law is exactly the documented formula, that with the flag the amplitude is 1 for every walking leg at any yaw while
`phase / stance / freq / beta / load_L / load_R / n_stance` are byte-equal to the default cycle's, that standing and
airborne rows still give 0, and that a `BatchSim` with `LegCycle()` and one with `LegCycle(flat_amplitude=False)` are
bit-identical over 12 steps. `tests/test_bit_identity.py`'s golden (the shipped default path) passes unchanged.

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_proprioception.py tests/test_body_cycle.py tests/test_bit_identity.py -q
    33 passed, 5 subtests passed                     (out/vncd5/smoke/pytest_cpu.txt)

**Provenance fix (round 4's defect R9).** Every run JSON now carries a `sense` block -- the spec, every token of
`Proprioception.FLAGS`, `mn_ref_hz`, `hair_plate_max_hz`, `campaniform_load_hz`, the explicit overrides and which
parameters are the sense's own defaults -- plus `started_utc` / `finished_utc`, at the top level and inside
`provenance.stimulus.params`. Round 4's compass JSONs recorded `mn_ref_hz: null`; a level2 run cannot.

## 3. The channel-matched control's two parameters (CPU; `out/vncd5/channel_match_derivation.json`)

**The laws** (`senses.Proprioception.rates`, unchanged): hair plate `rate = 5 + (hp_max - 5) * clip(legMN_side /
mn_ref_hz, 0, 1)` on the ground (5 Hz airborne); leg campaniform `rate = load_hz` on the ground, 0 airborne. The window
mean over cells and frames is therefore LINEAR in `hp_max` at a fixed drive distribution, `mean = 5 + (hp_max - 5) d`
with `d = mean[clip(legMN_side / 8.84, 0, 1) * ground]`, and `load_hz x ground` for the campaniform. Both parameters sit
inside the same closed loop the round-4 `mn_ref` derivation had to solve (afferents -> VNC -> leg MNs -> afferents), so
lowering them lowers the leg-MN rate and `d` falls with it: the realised hair-plate mean drops MORE than the open-loop
algebra says. The derivation is the same fixed-point method, with the same kind of CPU calibration runs
(`out/vncd5/cal/`, `CUDA_VISIBLE_DEVICES=-1`, 4 flies x 12 s, window 2-12 s, brain seed 0; 340-400 s wall each):

| step | what | result |
|---|---|---|
| targets | the realised window-mean commanded rates of arm C in `out/vncd4` (5 runs x 16 flies), recomputed here from `room_C_r*_body.npz` | hair plate **47.065 +- 0.562**, campaniform **24.887 +- 0.009** Hz |
| 1 open loop | `cal_L` at the shipped `hp_max` 100 / `load` 50 and `mn_ref` 8.84 realises hair **57.263** Hz -> `d` = **0.550133**; chordotonal 86.970, leg MN 5.297 / 4.418, ground 1.000 | `hp_max` = 5 + 42.065 / 0.550133 = **81.46**; `load` = 24.887 / 0.99342 (the GPU L arm's ground fraction) = **25.05** |
| 2 calibration | `cal_K1` at (81.46, 25.05) realises hair **43.294** -> `d` = **0.500841**; chordotonal 80.087, leg MN 4.694 / 4.150 | the loop cost 3.77 Hz of hair plate: `d(d)/d(hp_max)` = **+0.00266 per Hz** |
| 3 fixed point | solve `5 + (x - 5)(d0 + s(x - 100)) = 47.065` | x = **86.7105** -> **chosen `hair_plate_max_hz` 86.71**, **`campaniform_load_hz` 25.05**; predicted realised hair 47.06 |
| 4 verification | `cal_K2` at (86.71, 25.05) realises hair **46.449** (error **-0.62** Hz), campaniform **25.050** (+0.16), chordotonal **80.986**, leg MN 4.768 / 4.197 | inside the predeclared +-5 / +-2.5 Hz tolerances; **nothing was re-fitted** after the verification |

**The declared side effect, predeclared as precondition P1b.** Matching the two channels at a FIXED `mn_ref` 8.84 also
lowers the chordotonal channel, because the same leg-MN rate feeds it: on CPU, 86.97 Hz (arm L) -> **80.99** (arm K),
a -6.0 Hz deficit at the same `mn_ref`. That is inside the predeclared 10 Hz level tolerance but not zero, and the
predeclaration therefore fixed the reading in advance: *whenever K's realised chordotonal sits more than 3 Hz below C's,
a positive C v K (or M v K) on DNa02 or AN04B003 is reported as an UPPER BOUND on any structure term, never as its
size.* The alternative -- re-deriving `mn_ref` for K as well -- would have made K a two-parameter refit of a control
whose only job is to match two channels, and was not done.

Both values are LABELLED CONTROL parameters passed on the job line (`--hair-plate-max-hz 86.71
--campaniform-load-hz 25.05`, the family table `ARM_SENSE_KW`); the sense's defaults (100 / 50) are untouched, and
`senses.Proprioception` is built with its own constructor keywords.

## 4. The batch, and what the six arms actually realised (`out/vncd5/verification_console.txt`, `run_verification.csv`, `arm_levels.csv`)

ONE submission, `out/vncd5/batch.sh` -> `cluster_run.py --name vncd5 --minutes 120 --arm-block fam`, run
**vncd5-2ffbc3** on the house cluster: 30 room jobs in 5 blocks of 6 (`fam_r0` .. `fam_r4`, the six arms of one brain
seed on one node), **30 job(s), 0 failed (22.6 min)**, `--fetch out/vncd5/`. Artefacts: 30 x (`.json`, `_body.npz`,
`_rec.npz`, `_flies.npz`, `_max.npz`) = 30 JSON + 120 npz + 30 consoles. Every run reports `device cuda`,
`device_name NVIDIA B200`, `family level2`, its arm's spec, its `sense` block and its block key; the per-run table is
`run_verification.csv` and is reproduced in the verification console. The sense parameters, per arm, straight out of
the run JSONs: A spec `None`; L `all` at mn_ref **8.84** (hair 100 / load 50, the sense's defaults); U `all+unsided`
at 8.84, token `unsided`; K `all` at 8.84 with hair **86.71** / load **25.05** (`overrides mn_ref_hz=8.84,
hair_plate_max_hz=86.71, campaniform_load_hz=25.05`); M `all+leg_cycle+leg_cycle_flat`, tokens
`leg_cycle,leg_cycle_flat`, every rate parameter at its default; C `all+leg_cycle`, token `leg_cycle`.

### 4.1 The realised levels (recomputed from `*_body.npz` on the post-skip frames, run mean +- SD over 5 runs)

| arm | chordotonal (L / R; L-R) | hair plate (L-R) | campaniform | haltere | leg MN L / R | amp (walking) |
|---|---|---|---|---|---|---|
| A | -- (no sense attached) | -- | -- | -- | 2.540 / 2.389 | -- |
| L | **86.05 +- 0.70** (92.70 / 79.14; **+13.56**) | 56.64 +- 0.47 (**+9.20**) | 49.54 | 18.89 | 5.260 / 4.401 | -- |
| U | **84.84 +- 0.87** (84.84 / 84.84; **+0.000**) | 55.78 +- 0.59 (**+0.000**) | 49.59 | 18.66 | 4.967 / 4.548 | -- |
| K | **77.37 +- 0.81** (81.61 / 72.97; +8.64) | **44.34 +- 0.47** (+5.04) | **24.78** | 16.37 | 4.560 / 4.014 | -- |
| M | **92.12 +- 1.53** (92.13 / 92.10; +0.03) | 49.57 +- 0.83 (+0.00) | 24.87 | 17.22 | 4.939 / 4.794 | **1.000000** (\|L-R\| **0.000000**) |
| C | **86.11 +- 1.67** (85.96 / 86.26; -0.30) | 46.36 +- 0.91 (-0.17) | 24.91 | 16.47 | 4.735 / 4.591 | 0.9477 (\|L-R\| 0.01702) |

### 4.2 The predeclared preconditions (`precondition_checks.csv`; every one checked from the recordings, before any verdict was read)

| precondition | result |
|---|---|
| P1 chordotonal L / U / K against C (+-10 Hz) | **PASS**: -0.06 / -1.27 / **-8.74** Hz |
| P1b K's chordotonal deficit (> 3 Hz makes C v K and M v K upper bounds) | **triggered**: -8.74 Hz, so the declared upper-bound reading applies to F3 and F6 |
| P2 K's hair plate (+-5) and campaniform (+-2.5) against C | **PASS**: -2.02 and -0.13 Hz |
| P3 U's chordotonal and hair-plate L-R ~ 0 (+-1 Hz) | **PASS**: +0.0000 and +0.0000 against L's +13.5643 / +9.2043 |
| P4 M's amplitude 1 while walking (+-0.02), \|amp L-R\| 0 (+-1e-6) | **PASS**: 1.000000 (max 1.000000, a single distinct value) and 0.000e+00, against C's 0.947728 and 0.01702 |
| P5 every channel inside its ledger bracket | **PASS** |
| P6 run configuration (30 runs, cuda, family, specs, sense block, blocks, 5 artefacts each) | **PASS**; one B200 device name, one compiled-connectome md5, one 51-file fingerprint |
| the predeclaration precedes the batch | **PASS**: stamped 04:54:59Z, earliest run `started_utc` 04:56:11Z |

**The one check that fails is one this audit added after the fact, and it matters**: the realised chordotonal gap of
every decision pair, reported for all six pairs. Five are inside the 10 Hz level tolerance (U v L -1.21, K v L -8.68,
C v K +8.74, M v C +6.01, C v U +1.27) and **M v K is +14.74 Hz, OUTSIDE it** -- so F6, which the predeclaration
called one of the two "cleanest structure-vs-level contrasts", is a LEVEL-MISMATCHED pair and no attribution is made
from it (section 6). Two of the gaps are structural, not accidental, and both were predictable from the laws:
matching K's two channels costs it 8.7 Hz of chordotonal through the loop (P1b, declared), and holding M's amplitude
at 1 BUYS it 6.0 Hz over C, because the cycle law's realised amplitude is 0.948, not 1. The level model of section 7
exists because of these two facts.

## 5. The primaries, per predeclared family (`analysis/decision_table.csv`, `pairwise.csv`; 5 runs per arm, floor p 0.0079)

Run mean +- SD over the five runs; `diff` = treatment - reference; `verdict` is `compare`'s; `holm` is the
Holm-adjusted p at the declared m = 7. **No cell of this table is CALLED**, because 0.0079 x 7 = 0.0556 (section 0).

| family | key | reference | treatment | diff | z | p | holm | verdict |
|---|---|---|---|---|---|---|---|---|
| **F1 U v L** | DNa02_L | 0.4357 +- 0.0269 | 0.3870 +- 0.0089 | -0.0487 | -1.8 | 0.008 | 0.056 | null |
| | DNa02_R | 0.2734 +- 0.0137 | 0.2793 +- 0.0270 | +0.0059 | +0.4 | 1.000 | 1.000 | null |
| | yaw SD clean | 7.235 +- 0.172 | 6.748 +- 0.131 | -0.487 | -2.8 | 0.008 | 0.056 | null |
| | straightness | 0.4831 +- 0.0239 | 0.7141 +- 0.0365 | +0.2309 | +9.7 | 0.008 | 0.056 | result |
| | DNa02 L-R | 0.1623 +- 0.0147 | 0.1078 +- 0.0350 | -0.0545 | -3.7 | 0.008 | 0.056 | result |
| | AN04B003_L | 19.059 +- 0.129 | 18.054 +- 0.124 | -1.005 | -7.8 | 0.008 | 0.056 | result |
| | AN04B003_R | 16.506 +- 0.193 | 17.356 +- 0.212 | +0.850 | +4.4 | 0.008 | 0.056 | result |
| **F2 K v L** | DNa02_L | 0.4357 | 0.3498 +- 0.0140 | -0.0859 | -3.2 | 0.008 | 0.056 | result |
| | DNa02_R | 0.2734 | 0.2612 +- 0.0108 | -0.0122 | -0.9 | 0.095 | 0.095 | null |
| | yaw SD clean | 7.235 | 6.781 +- 0.085 | -0.455 | -2.6 | 0.008 | 0.056 | null |
| | straightness | 0.4831 | 0.6782 +- 0.0201 | +0.1951 | +8.2 | 0.008 | 0.056 | result |
| | DNa02 L-R | 0.1623 | 0.0885 +- 0.0108 | -0.0738 | -5.0 | 0.008 | 0.056 | result |
| | AN04B003_L | 19.059 | 18.119 +- 0.132 | -0.940 | -7.3 | 0.008 | 0.056 | result |
| | AN04B003_R | 16.506 | 15.898 +- 0.223 | -0.608 | -3.2 | 0.008 | 0.056 | result |
| **F3 C v K** | DNa02_L | 0.3498 | 0.5280 +- 0.0136 | **+0.1783** | +12.8 | 0.008 | 0.056 | result |
| | DNa02_R | 0.2612 | 0.3802 +- 0.0233 | +0.1189 | +11.0 | 0.008 | 0.056 | result |
| | yaw SD clean | 6.781 | 7.725 +- 0.134 | +0.944 | +11.1 | 0.008 | 0.056 | result |
| | straightness | 0.6782 | 0.8570 +- 0.0346 | +0.1788 | +8.9 | 0.008 | 0.056 | result |
| | DNa02 L-R | 0.0885 | 0.1478 +- 0.0337 | +0.0593 | +5.5 | 0.008 | 0.056 | result |
| | AN04B003_L | 18.119 | 22.786 +- 0.428 | **+4.666** | +35.5 | 0.008 | 0.056 | result |
| | AN04B003_R | 15.898 | 23.224 +- 0.464 | **+7.326** | +32.9 | 0.008 | 0.056 | result |
| **F4 M v C** | DNa02_L | 0.5280 | 0.6209 +- 0.0073 | +0.0929 | +6.8 | 0.008 | 0.056 | result |
| | DNa02_R | 0.3802 | 0.4067 +- 0.0205 | +0.0265 | +1.1 | 0.151 | 0.151 | null |
| | yaw SD clean | 7.725 | 8.354 +- 0.181 | +0.629 | +4.7 | 0.008 | 0.056 | result |
| | straightness | 0.8570 | 0.7939 +- 0.0143 | -0.0631 | -1.8 | 0.008 | 0.056 | null |
| | DNa02 L-R | 0.1478 | 0.2143 +- 0.0154 | +0.0664 | +2.0 | 0.016 | 0.056 | null |
| | AN04B003_L | 22.786 | 24.401 +- 0.418 | +1.615 | +3.8 | 0.008 | 0.056 | result |
| | AN04B003_R | 23.224 | 24.873 +- 0.432 | +1.649 | +3.6 | 0.008 | 0.056 | result |
| **F5 C v U** | DNa02_L | 0.3870 | 0.5280 | +0.1410 | +15.9 | 0.008 | 0.056 | result |
| | DNa02_R | 0.2793 | 0.3802 | +0.1009 | +3.7 | 0.008 | 0.056 | result |
| | yaw SD clean | 6.748 | 7.725 | +0.977 | +7.5 | 0.008 | 0.056 | result |
| | straightness | 0.7141 | 0.8570 | +0.1430 | +3.9 | 0.008 | 0.056 | result |
| | DNa02 L-R | 0.1078 | 0.1478 | +0.0401 | +1.1 | 0.222 | 0.222 | null |
| | AN04B003_L | 18.054 | 22.786 | **+4.732** | +38.1 | 0.008 | 0.056 | result |
| | AN04B003_R | 17.356 | 23.224 | **+5.868** | +27.7 | 0.008 | 0.056 | result |
| **F6 M v K** (LEVEL-MISMATCHED, +14.7 Hz) | DNa02_L | 0.3498 | 0.6209 | +0.2712 | +19.4 | 0.008 | 0.056 | result |
| | DNa02_R | 0.2612 | 0.4067 | +0.1454 | +13.4 | 0.008 | 0.056 | result |
| | yaw SD clean | 6.781 | 8.354 | +1.573 | +18.5 | 0.008 | 0.056 | result |
| | straightness | 0.6782 | 0.7939 | +0.1156 | +5.8 | 0.008 | 0.056 | result |
| | DNa02 L-R | 0.0885 | 0.2143 | +0.1258 | +11.6 | 0.008 | 0.056 | result |
| | AN04B003_L | 18.119 | 24.401 | +6.282 | +47.7 | 0.008 | 0.056 | result |
| | AN04B003_R | 15.898 | 24.873 | +8.975 | +40.3 | 0.008 | 0.056 | result |
| **F7 L v A** | DNa02_L | 0.0152 +- 0.0040 | 0.4357 | +0.4204 | +106.3 | 0.008 | 0.056 | result |
| | DNa02_R | 0.0766 +- 0.0080 | 0.2734 | +0.1968 | +24.6 | 0.008 | 0.056 | result |
| | yaw SD clean | 2.651 +- 0.104 | 7.235 | +4.585 | +44.3 | 0.008 | 0.056 | result |
| | straightness | 0.9951 +- 0.0007 | 0.4831 | -0.5120 | -689.8 | 0.008 | 0.056 | result |
| | DNa02 L-R | -0.0614 +- 0.0091 | 0.1623 | +0.2236 | +24.5 | 0.008 | 0.056 | result |
| | AN04B003_L | 0.592 +- 0.016 | 19.059 | +18.467 | +1191.0 | 0.008 | 0.056 | result |
| | AN04B003_R | 0.314 +- 0.006 | 16.506 | +16.192 | +2873.5 | 0.008 | 0.056 | result |

Per-seed values behind the arms (r0..r4; `analysis/room_table.csv` `*_runs`, reproduced in
`analysis/pairs_console.txt`'s per-arm block): DNa02_L A [0.0113 0.0113 0.0205 0.0171 0.0161], L [0.4005 0.4251
0.4333 0.4459 0.4736], U [0.3917 0.3950 0.3720 0.3873 0.3892], K [0.3519 0.3632 0.3529 0.3261 0.3548], M [0.6187
0.6327 0.6227 0.6141 0.6165], C [0.5155 0.5343 0.5290 0.5144 0.5470]; clean yaw SD A [2.6841 2.5047 2.7246 2.5861
2.7544], L [7.0034 7.1886 7.2284 7.2737 7.4825], U [6.8111 6.6278 6.9263 6.7598 6.6147]; every other key's five
per-run values are in `room_table.csv` and `pairs_console.txt` (`pairwise.csv` carries the run mean, SD and n
only).

**Integrity note, added after the independent skeptic pass (2026-09-15).** The previous version of the sentence
above printed per-seed DNa02_L lists for U, K, M and C, and a clean-yaw-SD list for U, whose values are in no file
in `out/vncd5`: they had been made to carry the published means instead of being transcribed from
`analysis/room_table.csv`, and the skeptic pass caught it (R1). The lists above are pasted from that file's `*_runs`
columns and were re-checked against them. Every mean, SD, diff, z, p and verdict in the table above is unaffected
and reproduces exactly; only the prose scatter was wrong, and it is replaced rather than deleted.

**Within-batch context, C v L (round 4's decision pair, the 8th pair of the family table):** DNa02_L +0.092
(z +3.4, `result`), DNa02_R +0.107 (z +7.8, `result`), straightness +0.374 (z +15.6, `result`), AN04B003_L +3.73 /
_R +6.72 (`result`), **clean yaw SD +0.49 (z +2.8, `null`)** and DNa02 L-R -0.014 (`null`). Round 4 called the yaw-SD
row at z +7.1; in a second batch of the same two arms the verdict is `null` while the difference itself replicated to
86 % (+0.572 -> +0.490) with complete rank separation in both batches -- what moved is L's own between-run SD
(0.081 -> 0.172), which is the z denominator (section 9). The DNa02 and AN04B003 rows replicate.

**The fractions (X - A) / (C - A), bookkeeping with the per-seed scatter:** DNa02_L L 0.82 / U 0.73 / K 0.65 /
M 1.18; DNa02_R 0.65 / 0.67 / 0.61 / 1.09; clean yaw SD 0.90 / 0.81 / 0.81 / 1.12; DNa02-active fraction 0.75 / 0.70 /
0.63 / 1.14; AN04B003_L 0.83 / 0.79 / 0.79 / 1.07. On straightness the same fractions are 3.71 / 2.04 / 2.30 / 1.46 --
every steady control overshoots, as round 4's skeptic said of L alone.

## 6. Each pair's decision, in its predeclared words

Quoted from `out/vncd5/predeclared.json` (`primaries.families.<name>.decision_rule_in_words`), then applied. The
rules are phrased on the `compare` verdict; the Holm layer (section 0) is reported beside every row and calls none.

**F1 U v L.** *"If U v L is null on DNa02_L, DNa02_R, the clean yaw SD and AN04B003_L/_R, removing the round-2 law's
DC chordotonal L-R (+13.6 Hz) and hair-plate L-R (+9.2 Hz) at the same channel MEANS leaves the level control where it
was: the sidedness is not what separates L from C ... If U v L is result on DNa02 or on AN04B003, the sidedness
contributes to the level control's own rates in the direction of the sign, and the structure-vs-level contrast is then
read on C v U (and M v K) rather than on C v L."*
-> **`null` on DNa02_L, DNa02_R and the yaw SD; `result` on AN04B003_L (-1.01) and AN04B003_R (+0.85), in OPPOSITE
directions** (pooled -0.08 Hz). The second branch fires, so the structure-vs-level contrast is read on **C v U**
(F5); and what the sidedness moves at the relay is its SIDE BALANCE, not its level. It moves the behaviour: drift
+3.48 -> +1.74 deg/s, straightness 0.483 -> 0.714, DNa02 L-R 0.162 -> 0.108.

**F2 K v L.** *"... If K v L is result POSITIVE on AN04B003_L/_R and on DNa02_L / DNa02_R, the unmatched channels were
holding the level control's relay down ... If K v L is null on AN04B003 and DNa02, the unmatched hair plate /
campaniform are NOT what separated L from C ..."*
-> **Neither branch: `result` NEGATIVE on AN04B003 (-0.94 / -0.61) and on DNa02_L (-0.086).** The predicted
disinhibition is not visible in the pair because P1b's declared side effect swamps it (K loses 8.7 Hz of chordotonal,
worth -1.67 Hz of AN04B003 at the level model's slope, against the hair plate's +0.86 Hz of disinhibition at K's
-12.3 Hz hair step). **The pair therefore settles nothing on its own and the audit says so**; the hair-plate route is
sized in section 7 (room slope -0.070 Hz/Hz) and section 13 (isolated cell -0.149 Hz/Hz), both sign-negative, both
small. The channel match itself worked: the direct SNpp45 term on DNa02_L is +128.5 mV/s (K) against +127.1 (C) and
+167.9 (L), and IN13B001 76.3 -> 61.6 Hz (C 64.3).

**F3 C v K.** *"If C v K is result on DNa02_L or DNa02_R or the clean yaw SD with the three commanded channel means
matched (preconditions P1-P2 met), something other than the three channel LEVELS separates the leg cycle from the
round-2 transducer, in the direction of the sign ... whenever the realised deficit exceeds 3 Hz, a positive C v K ...
is reported as an UPPER BOUND on any structure term, never as its size."*
-> **`result` on all seven, positive** (DNa02_L +0.178, DNa02_R +0.119, yaw SD +0.94, AN04B003 +4.67 / +7.33), with
P1-P2 met and **P1b triggered (-8.74 Hz)**: so something other than the three channel levels separates the cycle from
the round-2 transducer, **and these numbers are upper bounds**. How loose: at the level model's slopes K's -8.735 Hz
chordotonal deficit is worth +1.68 Hz of the +4.67 / +7.33 AN04B003 difference, and only **+0.028 Hz of the +0.178
DNa02_L difference** -- so the DNa02 upper bound is loose by about a sixth, not by an unknown amount. The level model
removes the bound by using K as one
calibration point among three instead of as a matched partner: the structure term on AN04B003 is then +3.88 / +5.00,
not +4.67 / +7.33.

**F4 M v C.** *"If M v C is null on every primary, the amplitude law -- and with it the half_width_m yaw term and the
|amp L-R| turn term -- contributes nothing beyond the per-leg / per-phase modulation ... If M v C is result, the
amplitude / turn term contributes, in the direction of the sign, and the audit reports which rows."*
-> **`result` on DNa02_L (+0.093), the clean yaw SD (+0.63) and AN04B003 (+1.62 / +1.65); `null` on DNa02_R,
straightness and DNa02 L-R.** The direction is "more without the amplitude law", which is the opposite of a turn term
that adds drive -- and it is mostly the level: M carries +6.0 Hz more chordotonal than C because amplitude 1.000
exceeds the law's realised 0.948. On the level model **about 60 % of M's AN04B003 excess over C is its extra 6.0 Hz
of chordotonal** (predicted +0.956 / +0.904 of the observed +1.615 / +1.649). The rest is not: M's structure residual
**exceeds** C's by +0.659 (L, z +3.8, `result`) and +0.745 (R, z +3.5, `result`), 6-7 x the fit's RMSE, and on
DNa02_L by +0.069 (z +4.0, `result`), which is essentially the whole +0.093 M-over-C DNa02_L difference. **This batch
therefore cannot say that the amplitude law adds no drive**; what it can say is that most of M's excess is level, and
that a flat amplitude is not only "no turn term" but also a different modulation waveform. The row the turn term does
own remains the per-frame sided DNa02 signal (-0.334 C vs -0.107 M), and it is not a bigger afferent alternation
under C: mean per-frame |chordotonal L-R| is 12.13 Hz (C) against 13.02 (M). Where else the turn term shows: the
`corr(DNa02 L-R, chordotonal L-R)` (-0.143 C vs -0.034 M) and straightness (0.857 vs 0.794). **F4 stays open.**

**F5 C v U** and **F6 M v K.** *"A primary called in BOTH C v U and M v K is structure beyond level under either
control; a primary called in neither is not supported; a primary called in exactly one is reported as conditional on
the confound that control leaves open."*
-> **F6 is VOID as a level contrast**: its realised chordotonal gap is +14.74 Hz, outside the declared 10 Hz
tolerance (section 4.2), so no attribution is made from it. That leaves **F5 alone**: `result` on DNa02_L (+0.141),
DNa02_R (+0.101), the yaw SD (+0.98), straightness (+0.143) and AN04B003 (+4.73 / +5.87), `null` on DNa02 L-R -- with
the chordotonal matched to +1.27 Hz and the sidedness removed, but the hair plate still +9.42 Hz higher in U (which,
at the measured sign-negative slope, ACCOUNTS FOR -0.66 Hz of AN04B003 in U, i.e. works in C's favour and inflates
the C v U excess by that much). C v U also leaves a **-24.7 Hz campaniform mismatch** (C 24.91, U 49.59). The level
model does not correct for it because the campaniform cannot be separated from the hair plate across three steady
arms; the only evidence that it does not matter is the single cell's campaniform null (-0.055 Hz, z -0.3), on a
752-cell subset. F5's attribution is conditional on that. The two-control rule cannot be applied as predeclared; the
level model of section 7 is what replaces it, and it uses all three steady arms at once.

**F7 L v A.** *"L v A result on DNa02_L / DNa02_R / the clean yaw SD means the afferent level as delivered by the
round-2 law (its sidedness included) fires DNa02 and raises the yaw SD over the shipped path, as in round 4."*
-> **`result` on all seven** (DNa02_L +0.420, DNa02_R +0.197, yaw SD +4.58, straightness -0.512, DNa02 L-R +0.224,
AN04B003 +18.47 / +16.19). The level alone fires DNa02 and turns the fly, exactly as round 4 found, and the fractions
above are the bookkeeping.

## 7. The level model: what the LEVELS predict, and what the cycle arms do (`out/vncd5/level_model.json`, POST HOC)

This analysis is **not predeclared**; it is labelled so, and no decision of section 6 rests on it. It exists because
every predeclared pair moves more than one thing at once (section 4.2). The three steady-input arms have no per-phase
modulation anywhere in them and between them span chordotonal **71.6-93.5 Hz** and hair plate **41.0-61.6 Hz** per
side -- so they calibrate what a steady transducer of a given level does to a given cell, and the cycle arms'
distance above that surface is the part their levels do not explain. OLS on 3 arms x 2 sides x 5 runs
(`y_side = a_side + b x chordotonal_side + c x hair_plate_side`; the campaniform is not a predictor -- it is
collinear with K's hair plate and the single cell finds no campaniform route into AN04B003):

| response | b (per Hz chordotonal) | c (per Hz hair plate) | RMSE | steady-arm residuals | **C (L / R)** | **M (L / R)** |
|---|---|---|---|---|---|---|
| **AN04B003** | **+0.1921 +- 0.0128** | **-0.0697 +- 0.0103** | **0.106 Hz** | -0.108 .. +0.084, every one `null` | **+3.878 / +5.004** (+37 / +47 x RMSE) | **+4.537 / +5.749** |
| DNa02 (per-fly rate) | +0.0032 +- 0.0027 | +0.0012 +- 0.0022 | 0.022 Hz | -0.017 .. +0.019 | +0.149 / +0.090 | +0.218 / +0.094 |
| IN13B001 | -2.648 +- 0.250 | +2.976 +- 0.203 | 2.08 Hz | -1.53 .. +2.59 | +20.3 / +19.2 | +30.3 / +29.3 |

**What the error bars and the "span" do and do not mean** (added after the independent skeptic pass). The design has
only **six distinct (arm, side) level points**; the 30 side-observations are five runs of each, so the OLS SEs above
are pseudo-replicated. Refitting on the six arm-side means (2 residual df) gives **b +0.1848 +- 0.0355, c -0.0648 +-
0.0285** (C's residual +3.92 / +5.07) -- the hair-plate slope is 2.3 sigma from zero, not 6.8, and `-0.070 Hz/Hz`
should be read as "small, negative, order 0.07", not as two significant figures. The three steady arms also lie close
to one line in the (chordotonal, hair-plate) plane (corr 0.925 on the six means, hair = 0.974 x chord - 28.35):
**C sits 9.1 Hz and M 11.8 Hz of hair plate off that line**, so the cycle predictions are joint extrapolations even
though both marginal ranges contain them. Two checks bound the risk: fitting on two steady arms and predicting the
**held-out steady arm** costs only -0.33 .. +0.25 Hz (the honest out-of-sample bar, against which the cycle residual
is 11-20 x, not 37-54 x); and **no hair-plate slope that removes the cycle residual is compatible with the steady
arms** -- `c = -0.55` would be needed and it degrades the steady-arm RMSE 9 x. The residual also survives adding the
campaniform and the haltere (C +2.50 / +3.57), though at 5 and 6 parameters those fits are at or past saturation for
a six-point design.

**How much of this rests on any one control arm (leave-one-arm-out).** Dropping L: b +0.233, c -0.093, C's residual
+3.60 / +4.66. Dropping U: b +0.141, c -0.037, C's residual +4.08 / +5.57. **Dropping K the fit collapses**
(b -5713, c +8419): K is the only arm that breaks the chordotonal / hair-plate collinearity the afferent -> leg-MN
loop imposes, which is exactly what a channel-matched control is for, and it means the two slopes are identified by
one arm's displacement. The AN04B003 conclusion survives both admissible refits (+3.6 .. +4.1 on C's left side);
**the DNa02 row does not** (dropping U gives C +0.049 / -0.176), so the DNa02 statement is left to the pairs of
section 5, and **the IN13B001 model is ill-conditioned** (b negative, c positive, wild under leave-one-out) and its
cycle residuals are not interpreted.

Reading the AN04B003 fit: the chordotonal slope **+0.192 Hz/Hz** and the hair-plate slope **-0.070 Hz/Hz** have the
signs and the rough ratio the anatomy predicts (`SNpp45 -> IN13B001 -| AN04B003` is sign-negative; the single cell
gives -0.149 Hz/Hz in isolation, the room's loop damps it). Applied to **this batch's** C-over-L difference
(+5.22 Hz pooled; round 4's own was +5.55 Hz, recomputed from `out/vncd4/room_{C,L}_r*_flies.npz` -- a labelled
cross-batch context number, not a compared row): L's +10.28 Hz hair-plate excess **suppresses L's own AN04B003 by
-0.72 Hz**, which therefore **accounts for +0.72 Hz of the +5.22 Hz pooled C-over-L difference (14 % of it)**; the
chordotonal term is +0.01 (the channels were matched), and the remaining **+4.49 Hz** is what the levels do not
explain. (+0.01 + 0.72 + 4.49 = 5.22.) That is the number round 4 asked for.

## 8. The decomposition, the sided signal per frame, and the behaviour

### 8.1 DNa02's input per arm (`analysis/dna02_decompose_summary.csv`, rate-weighted mV/s per post cell; the `rate_hz` column is AN04B003's, as round 4's R7 records)

| DNa02_L input | A | L | U | K | M | C |
|---|---|---|---|---|---|---|
| E total / I total / **net** | +743 / -1048 / **-305** | +1338 / -1276 / **+62** | +1308 / -1275 / **+33** | +1255 / -1241 / **+14** | +1408 / -1284 / **+124** | +1350 / -1256 / **+95** |
| **AN04B003/L** (leg afferents) | +9.1 | **+294.4** | +278.9 | +279.9 | **+376.9** | **+352.0** |
| **SNpp45/?** (hair plate, direct) | 0.0 | **+167.9** | +153.2 | **+128.5** | +136.1 | **+127.1** |
| PS059/L (haltere afferents) | -91.0 | -209.3 | -207.7 | -188.2 | -214.2 | -203.7 |
| PS049/L | -19.9 | -14.3 | -12.8 | -14.3 | -9.9 | -9.8 |
| IN12B014/R | -90.7 | -64.0 | -64.7 | -67.7 | -63.5 | -65.3 |
| IN19A003/L | -43.0 | -18.3 | -21.9 | -24.8 | -19.4 | -21.2 |
| LT51/L | -57.8 | -48.9 | -47.1 | -49.0 | -45.3 | -45.1 |

| DNa02_R input | A | L | U | K | M | C |
|---|---|---|---|---|---|---|
| E total / I total / **net** | +807 / -1172 / **-365** | +1227 / -1501 / **-273** | +1240 / -1506 / **-267** | +1185 / -1459 / **-273** | +1376 / -1529 / **-153** | +1328 / -1496 / **-168** |
| **AN04B003/R** | +4.8 | +249.6 | +262.5 | +240.4 | **+376.1** | **+351.2** |
| SNpp45/? | 0.0 | +65.5 | +70.3 | +52.7 | +62.5 | +58.6 |
| PS059/R | -49.7 | -170.3 | -170.5 | -151.6 | -176.6 | -166.9 |
| IN12B014/L | -125.7 | -94.2 | -93.9 | -97.7 | -96.4 | -98.5 |
| IN19A003/R | -60.6 | -49.0 | -57.7 | -61.2 | -54.8 | -57.8 |
| LT51/R | -68.8 | -71.9 | -69.4 | -70.4 | -68.0 | -66.9 |

(IN13B001 is **not** a direct input to DNa02 -- it acts on AN04B003 -- so it has no row here; its window rates per arm
are in 8.2. Every row is the run mean over 5 runs.) Reading it: the three steady arms put +279 .. +294 mV/s of
AN04B003 onto DNa02_L and the two cycle arms +352 .. +377, while the haltere cancellation (PS059) and the tonic VNC
inhibitors sit within 10 % of each other across all five sensed arms -- **the arms differ where round 3 and round 4
said they differ, on the ascending relay, and the channel-matched control is the first one whose DIRECT hair-plate
term (+128.5) equals the cycle's (+127.1)**.

### 8.2 The window rates of the hair-plate route, per arm (`pairwise.csv`, `chain_*` keys: every cell of the type, per side, from the per-fly recordings)

| type (Hz) | A | L | U | K | M | C |
|---|---|---|---|---|---|---|
| SNpp45 (hair-plate afferent) | 0.00 | 55.72 | 54.73 | **43.57** | 48.62 | **45.48** |
| IN13B001 (L / R) | 10.67 / 6.45 | 70.80 / 81.76 | 75.59 / 74.53 | **59.44 / 63.74** | 68.84 / 67.19 | **65.32 / 63.20** |
| AN04B003 (L / R) | 0.59 / 0.31 | 19.06 / 16.51 | 18.05 / 17.36 | 18.12 / 15.90 | **24.40 / 24.87** | **22.79 / 23.22** |
| PS059 (L / R) | 8.84 / 4.93 | 20.33 / 16.89 | 20.17 / 16.92 | 18.27 / 15.03 | 20.80 / 17.52 | 19.78 / 16.55 |
| IN12B014 (L / R) | 15.15 / 12.70 | 11.13 / 8.85 | 10.92 / 8.82 | 11.30 / 9.00 | 10.88 / 8.71 | 10.98 / 8.79 |
| PS049 (L / R) | 3.87 / 1.64 | 2.78 / 0.62 | 2.48 / 0.54 | 2.79 / 0.66 | 1.91 / 0.41 | 1.90 / 0.45 |
| IN14B003 | 1.25 | 2.58 | 2.38 | 1.64 | 3.31 | 3.04 |
| GLNO | 0.05 | 4.47 | 4.66 | 3.83 | 4.79 | 4.55 |

The route moves as its signs predict in every arm: the hair plate (SNpp45) and IN13B001 rise and fall together
(L 55.7 / 76.3, U 54.7 / 75.1, K 43.6 / 61.6, C 45.5 / 64.3), and AN04B003 is the one row where the cycle arms depart
from the steady ones regardless of where their hair plate sits. **No cell in these tables is called "silent": every
one has a nonzero rate in every arm, arm A included.**

### 8.3 Sidedness per frame (`analysis/sided_frames.csv`; clean walking frames, per fly then the run mean; recorder rows at lag 0)

| statistic | A | L | U | K | M | C |
|---|---|---|---|---|---|---|
| corr(yaw, chordotonal cmd L-R) | -- | **+0.265** | **+0.003** | +0.254 | +0.013 | **-0.147** |
| corr(yaw, leg MN L-R) | +0.257 | +0.251 | +0.138 | +0.239 | +0.128 | +0.031 |
| corr(AN04B003 L-R, chordotonal cmd L-R) | -- | +0.188 | **-0.008** | +0.249 | **-0.368** | **-0.345** |
| E[AN04B003 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) | -- | +1.98 | **+0.02** | +2.53 | **-3.80** | **-3.58** |
| corr(DNa02 L-R, chordotonal cmd L-R) | -- | +0.005 | +0.007 | +0.017 | -0.034 | **-0.143** |
| E[DNa02 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) | -- | -0.021 | +0.003 | +0.008 | **-0.107** | **-0.334** |
| corr(AN04B003 L-R, yaw) | +0.014 | +0.076 | +0.014 | +0.091 | +0.009 | -0.062 |
| corr(PS196_b L-R, yaw) | +0.014 | +0.059 | +0.074 | +0.068 | +0.080 | +0.071 |
| mean signed yaw, clean (deg/s; + left) | +0.194 | **+3.481** | +1.740 | +2.046 | +1.459 | **+1.123** |

Three things read straight off it. (i) **The unsided token does exactly what it claims**: under U the afferent L-R is
gone (corr(yaw, chord L-R) +0.003, the AN04B003 tripod term +0.02 Hz) while under L and K the round-2 law's bias is
the dominant sided signal (+0.265 / +0.254). (ii) **The tripod-locked antiphase at AN04B003 belongs to the per-phase
structure, not to the turn term**: it is -3.80 Hz under M and -3.58 under C, and absent from every steady arm. (iii)
**The per-frame sided signal at DNa02 is the one row the amplitude / turn term owns**: -0.334 Hz under C against
-0.107 under M, and ~0 in all three steady arms.

### 8.4 Behaviour (run mean +- SD; `analysis/room_table.csv`, `pairwise.csv`)

| key | A | L | U | K | M | C |
|---|---|---|---|---|---|---|
| clean yaw SD (deg/s) | 2.65 +- 0.10 | 7.24 +- 0.17 | 6.75 +- 0.13 | 6.78 +- 0.09 | **8.35 +- 0.18** | 7.72 +- 0.13 |
| median \|yaw\|, clean | 0.73 | **3.50** | 1.74 | 2.30 | 1.79 | 1.50 |
| mean signed yaw (+ left) | +0.19 | **+3.48** | +1.74 | +2.05 | +1.46 | +1.12 |
| straightness | 0.995 | **0.483** | 0.714 | 0.678 | 0.794 | **0.857** |
| flies (of 16) off the table | 0.2 | 12.6 | **14.0** | **14.0** | 9.8 | **7.4** |
| hops per fly | 0.05 | 0.76 | 0.74 | 0.89 | 0.45 | 0.35 |
| clean fraction | 0.999 | 0.866 | 0.849 | 0.865 | 0.913 | 0.933 |
| net heading change (turns/fly) | 0.032 | 0.246 | 0.292 | 0.289 | 0.254 | 0.211 |
| DNa02-active frames | 0.0064 | 0.0502 | 0.0471 | 0.0433 | **0.0728** | 0.0645 |
| leg MN L-R (Hz) | +0.152 | **+0.866** | +0.422 | +0.551 | +0.145 | +0.145 |
| haltere MN L-R (Hz) | -1.75 | -1.59 | -2.92 | -2.10 | -2.74 | -2.55 |
| GF per-frame max (Hz) | 24.9 | 37.8 | 38.2 | 40.4 | 29.4 | 28.6 |
| step frequency (Hz) / stance fraction | | | | | 7.86 / 0.764 | 7.65 / 0.770 |

The clean fraction differs between the arms by at most 1.18x (0.849 U vs 0.999 A; 0.866 L vs 0.933 C), well inside
the 2x that would oblige a window-matched re-scoring (docs/INTERP.md 10.4 item 23). Note that removing the round-2
law's sidedness does not make the fly stay on the table -- U and K lose 14 of 16 flies against L's 12.6 -- because
what puts a fly off the table here is the yaw SD, not the drift.

## 9. What this batch does NOT show

* **No row is Holm-called** (section 0). Read the diffs, the z and the per-run scatter; the family-wise control was
  unsatisfiable at m = 7 with 5 runs per arm.
* **M v K is level-mismatched** (+14.7 Hz of chordotonal) and carries no attribution.
* **C v K and M v K are upper bounds** on any structure term by the predeclared P1b rule (K's -8.74 Hz deficit).
* **The DNa02 structure residual is not robust** to the level model's leave-one-arm-out (section 7); only AN04B003's is.
* **The clean-yaw-SD row of C v L is `result` in round 4 (+0.572, z +7.1) and `null` here (+0.490, z +2.8)** -- but
  the **difference** replicated to 86 % and the rank separation is complete in **both** batches (exact p on the 5 v 5
  floor, 0.0079, every C run above every L run). What moved is L's own between-run SD (0.081 -> 0.172), which is the
  z denominator. Pooled over the two batches (10 v 10, a labelled cross-batch statement, not a compared row):
  +0.531 Hz, p 1.1e-05. **Neither batch is an outlier; the verdict flipped, the effect did not.**
* **That the amplitude law adds no drive.** M's structure residual EXCEEDS C's by +0.659 / +0.745 Hz on AN04B003
  (z +3.8 / +3.5) and by +0.069 on DNa02_L (z +4.0), so F4 stays open: most of M's excess over C is level, the rest
  is not, and this batch cannot separate the amplitude law from the fact that a flat amplitude is also a different
  modulation waveform.
* **Nothing here is about the compass**: no compass job was submitted (rounds 2-4 settled it), so this batch says
  nothing new about whether the self-turn reaches PS196_b, GLNO or the ring.
* **The single-cell check is a subset brain** (752 cells): AN04B003's absolute rate there is not the room's, and the
  numbers are within-protocol contrasts only.

## 10. What an adoption of the leg cycle would still require (nothing is adopted)

1. **A predeclaration whose Holm family can be satisfied.** At n v n runs the exact-U floor is `p_floor(n, n)`, and a
   family of m members can only be called if `p_floor x m <= alpha`: at 5 v 5 (0.0079) that is **m <= 6**, at 6 v 6
   (0.0021645) **m <= 23**. This round declared m = 7 at 5 runs and called nothing. The fix is arithmetic and belongs
   in the predeclaration step, not in the reading: **size the family against the floor, or raise the runs per arm.**
   This is not a new rule: `docs/INTERP.md` 10.2 already requires the family to be sized from the members that can
   move, and records object round 2 raising its arms from 5 to 6 runs for exactly this reason. Round 5's
   predeclaration did not apply it.
2. **A level-matched modulation-only arm.** M was meant to isolate the phase modulation and instead bought +6.0 Hz of
   chordotonal, because amplitude 1.000 is above the law's realised 0.948. The arm to run is
   `flat_amplitude` with the amplitude set to the cycle's own realised mean (or `mn_ref`-style compensation), so that
   M and C sit at the same level; then F4 and F6 become the clean contrasts they were meant to be.
3. **A channel-matched control that also holds the chordotonal level.** K's two parameters cost it 8.7 Hz of
   chordotonal through the loop. Deriving `mn_ref_hz` and the two channel parameters TOGETHER (a three-parameter fixed
   point, the same method as section 3) would give a control matched on all three channels at once, and would turn
   C v K from an upper bound into a measurement.
4. **The suite with the cycle on** (`scripts/benchmark.py` and the room ledger rows under `all+leg_cycle`, 5 runs, one
   block): unchanged from round 4 -- the cycle fly leaves the table (7.4 of 16), hops (0.35 / fly) and raises GF
   (28.6 vs 24.9 Hz); the object / loom / feeding rows have still never been run with the cycle.
5. **Ledger values** for `lit.walk.outer_leg_step_ratio_in_turn` (still empty) and `half_width_m` (1.0 mm,
   unmeasured). This round adds a reason to care about the second: the turn term is what owns the per-frame sided
   DNa02 signal (8.3), so `half_width_m` scales the only row the amplitude law is responsible for.
6. **A free-walking compass room under the cycle** (`cx_shift.py` room mode) -- unchanged from round 4 item 5.
7. **The FeCO walking-mean rate as a ledger number.** Still only the 10-150 Hz bracket exists, so 86-92 Hz is a
   consequence of the laws, not a measurement -- and this round adds a second unmeasured quantity of the same kind:
   the cycle's realised per-leg amplitude (0.948), which M's flat 1.000 replaced.

## 11. Files, provenance and reproduction

* Code (this task): `flyverse/senses.py` (the `unsided` and `leg_cycle_flat` tokens in `FLAGS`, their grammar
  refusals, the four-line side-mean in `rates`; every default unchanged), `flyverse/body.py`
  (`LegCycle.flat_amplitude`, default `False`), `scripts/probe_vnc_drive.py` (the `level2` family, `ARM_SENSE_KW`,
  `sense_kwargs_of` / `sense_record`, `--hair-plate-max-hz` / `--campaniform-load-hz`, the `sense` + `started_utc` /
  `finished_utc` provenance block, `chain_rates`, `holm_families` / `pairs --predeclared`, `WATCH_LEVEL2` with
  IN13B001, `attach_cycle`'s flat cycle, `plan`'s refusal to plan an underived control), `tests/test_proprioception.py`
  (`UnsidedTokenTests`), `tests/test_body_cycle.py` (`FlatAmplitudeTests`), `scripts/probe_an04b003_single_cell.py`
  (new), `out/vncd5/` (`derive_channel_match.py`, `verify_runs.py`, `level_model.py`), this audit.
  `flyverse/brain.py`, `optic.py`, `motor.py`, `batch_sim.py`, `batch_body.py` are untouched.
* Data: `out/vncd5/` -- `batch.sh`, `predeclared.json` (stamped 04:54:59Z), `tree_state.json`,
  `channel_match_derivation.json`, `cal/` (3 CPU calibration runs), `smoke/` (the CPU smokes and `pytest_cpu.txt`),
  30 room runs x 5 files, `analysis/`, `single_cell/`, `run_verification.csv`, `arm_levels.csv`,
  `precondition_checks.csv`, `level_model*.csv/json`, `submitted_at.txt`; console `out/vncd5_cluster.log`
  (`vncd5-2ffbc3`: 30 job(s), 0 failed, 22.6 min). Every room JSON carries `provenance` (resolved `LIFParams` /
  `OpticParams`, the compiled-W md5, the 51-file `source_fingerprint`, `execution.device` / `device_name`, seeds, the
  block key) **and the new `sense` block plus `started_utc` / `finished_utc`** -- the defect round 4's skeptic R9
  recorded (compass JSONs with `mn_ref_hz: null`) cannot occur in this family.
* Reducer identity: the five predeclared reducers (`scripts/probe_vnc_drive.py`, `scripts/probe_an04b003_single_cell.py`,
  `flyverse/interp/common.py`, `flyverse/senses.py`, `flyverse/body.py`, `out/vncd5/derive_channel_match.py`) are
  **byte-identical to their predeclared sha256** at the time this audit was written (`out/vncd5/post_analysis_sha.json`);
  `verify_runs.py` and `level_model.py` were written after the stamp, are not predeclared reducers, and their hashes
  are recorded in the same file.
* Reproduce: `PYTHONIOENCODING=utf-8 python scripts/probe_vnc_drive.py plan --family level2 --dir out/vncd5 --runs 5
  --compass-seeds 0 --draws 0 --minutes 120 --name vncd5` (writes the 30-job `batch.sh`); `bash out/vncd5/batch.sh`
  (one `cluster_run.py` call, `--arm-block fam`, house target); then, on CPU,
  `python out/vncd5/verify_runs.py --dir out/vncd5 --predeclared out/vncd5/predeclared.json`,
  `python scripts/probe_vnc_drive.py analyse --dir out/vncd5 --out out/vncd5/analysis --trace-arms L,K,C` (~11 min),
  `python scripts/probe_vnc_drive.py pairs --dir out/vncd5 --out out/vncd5/analysis --analysis out/vncd5/analysis
  --predeclared out/vncd5/predeclared.json` (~4 min), `python out/vncd5/level_model.py --dir out/vncd5`, and
  `python scripts/probe_an04b003_single_cell.py --runs 5 --out out/vncd5/single_cell` (~7 min, deterministic on CPU:
  the second run of it reproduced every row exactly). An independent rerun of
  `scripts/probe_an04b003_single_cell.py --runs 5` on a different machine reproduced every row of `rows.csv` to
  0.000e+00. A GPU rerun is a replication, not a bit-for-bit check.

## 12. Validation (CPU, `CUDA_VISIBLE_DEVICES=-1`; this desktop)

* `PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_proprioception.py tests/test_body_cycle.py
  tests/test_bit_identity.py -q`: **33 passed, 5 subtests passed** (`out/vncd5/smoke/pytest_cpu.txt`), including the
  two new bit-identity tests of section 2 and the unchanged shipped-path golden.
* CPU smokes, one short room step per new arm (U / K / M, 2 flies x 0.5 s, `--device cpu`): every one `exit 0`, and
  each one's JSON carries the right `sense` block (`out/vncd5/smoke/room_{U,K,M}_r0.json`). No number in this audit
  comes from a smoke.
* Three CPU calibration runs (`out/vncd5/cal/`, 4 flies x 12 s each) for the derivation of section 3; the fourth-step
  verification is the evidence for the chosen values and nothing was re-fitted after it.
* Batch verification (`verify_runs.py`, before any verdict was read): 30 runs, `device cuda` / `NVIDIA B200` in every
  one, one family, one spec per arm, the right `sense` block per arm, 5 artefacts per run, blocks `fam_r0..fam_r4`,
  and the predeclaration's stamp (04:54:59Z) earlier than the earliest run's `started_utc` (04:56:11Z).
* The predeclared preconditions P1-P6 all PASS (section 4.2); the one FAIL is the post-hoc level-gap check on M v K,
  which is reported as such and removes F6 from the decision set.
* `pairs`' new reducers were validated on the PREVIOUS batch before this one was analysed: run on `out/vncd4` with
  round 4's own `predeclared.json`, `holm_families` reproduces round 4's published calls exactly (C v L DNa02_L
  +0.1272 z +9.2 Holm 0.0397 CALLED, and the rest of its three families), and `chain_rates` reproduces its skeptic
  pass's AN04B003 (19.10 / 16.54 vs 23.13 / 23.62) and IN13B001 (76.36 vs 65.24) numbers.

## 13. The single-cell check: what AN04B003 actually responds to (CPU; `out/vncd5/single_cell/`)

Round 4's section 7 item 6, as corrected by its skeptic pass: the room runs cannot separate the modulation from the
hair-plate level, because in a room every channel moves with every other. This does, on a **subset brain**
(`Connectome.subset`): the three leg-afferent channels (chordotonal 615, hair plate 113, leg campaniform 12 cells) +
**AN04B003** (6) + **IN13B001** (6), 752 cells, and nothing else -- so AN04B003 receives only its direct afferent input
and the disynaptic hair-plate route. Its absolute rate is therefore NOT the room's and every number below is a
within-protocol contrast. Generator `scripts/probe_an04b003_single_cell.py`; 5 runs (brain seeds 0-4), 2 s settle +
20 s window, rows of one `FlyBrain` batch per clamp state, verdicts `flyverse.interp.common.compare` over runs
(5 v 5, floor p 0.0079).

The afferents are driven exactly as the room drives them (Poisson at a commanded rate per cell per 10 ms frame) with
either the cycle arm's own per-leg / per-phase law (`body.LegCycle` at the room's realised 9 mm/s, yaw 0: step 7.81 Hz,
stance fraction 0.766, amplitude 0.949; per-cell SD over time 38.6 Hz chordotonal) or **each cell at its own time-mean
of that same sequence** -- the same per-cell mean by construction (92.02 Hz chordotonal, 50.10 hair plate, 25.00
campaniform over the subset's cells). IN13B001 is either FREE or **CLAMPED**: its 177 incoming synapses are zeroed in
the subset's `W` and the cell is driven by Poisson at its rate in the room's cycle arm (65.2 Hz), so its inhibition of
AN04B003 is the same train in every clamped row.

| row (AN04B003 Hz, run mean +- SD over 5 runs) | AN04B003 | L / R | IN13B001 |
|---|---|---|---|
| steady chordotonal, IN13B001 clamped | 21.717 +- 0.227 | 18.01 / 25.43 | 65.30 |
| **8 Hz-modulated** chordotonal, IN13B001 clamped | **23.345 +- 0.204** | 19.32 / 27.37 | 64.90 |
| steady, IN13B001 free | 16.533 +- 0.170 | 12.87 / 20.20 | 91.83 |
| modulated, IN13B001 free | 17.932 +- 0.067 | 14.04 / 21.82 | 90.77 |
| all three channels modulated, IN13B001 free | 19.987 +- 0.202 | 16.30 / 23.67 | 88.03 |
| steady + hair plate raised to the level control's 56.8 Hz | 15.533 +- 0.096 | 11.82 / 19.24 | **99.64** |
| steady + campaniform raised to the level control's 49.7 Hz | 16.478 +- 0.164 | 12.84 / 20.11 | 91.95 |
| steady + both at the level control's values | 15.673 +- 0.146 | 12.04 / 19.31 | 99.89 |

**(i) The modulation term is real and it is positive.** At the SAME per-cell chordotonal mean, with IN13B001's rate
clamped, modulating the input raises AN04B003 by **+1.63 Hz** (21.72 -> 23.35; z +7.2, p 0.0079, `result`; L +1.31,
R +1.94). With IN13B001 free the same contrast is **+1.40 Hz** (z +8.2, `result`). A threshold unit fed a modulated
Poisson input fires more than one fed a steady input of the same mean: round 4 PROPOSED that mechanism for the
residual, and here it is measured. Modulating the hair plate and campaniform as well adds a further **+2.06 Hz**
(z +30.7) -- the cycle's full afferent pattern is worth +3.45 Hz of AN04B003 over the matched steady input.

**(ii) The hair-plate level term is real and it is negative, through IN13B001, exactly as the route predicts.**
Raising the hair plate alone from the cycle law's own per-cell mean (50.10 Hz) to the level control's realised 56.80 Hz
-- a +6.70 Hz step -- raises IN13B001 by **+7.80 Hz** (91.83 -> 99.64) and LOWERS AN04B003 by **-1.00 Hz**
(z -5.9, `result`), i.e. **-0.149 Hz of AN04B003 per Hz of hair plate**, with `dIN13B001/dHair` = 1.16 (round 4's
pooled room estimate: 1.15). **The campaniform does nothing**: 24.9 -> 49.7 Hz alone gives -0.055 Hz, z -0.3, `null`
(and IN13B001 +0.12) -- so round 4's second unmatched channel, the -24.8 Hz one, is not a route into AN04B003 at all,
and only the hair plate of the two needs matching. Both together: -0.86 Hz.

**What the two terms predict for the room.** The room's L-vs-C hair-plate mismatch is +9.69 Hz, which at the measured
slope is **-1.44 Hz of AN04B003 in L**; the modulation is worth **+1.63 Hz in C**; together **+3.07 Hz** of the pooled
C-over-L AN04B003 difference that round 4 measured as +5.55 Hz (23.36 vs 17.82). So on this subset both mechanisms
exist, the modulation is the larger of the two, and neither alone accounts for the room's difference -- the room adds
the closed loop (the cycle arm's own leg MNs and the rest of the VNC), which is what the batch of sections 4-6
measures.

## Report

```yaml
summary: >
  The three controls round 4's skeptic pass demanded were run in ONE house submission (vncd5-2ffbc3, 30 jobs, 0
  failed, 22.6 min, B200; six arms x 5 brain seeds, no compass), with two new opt-in mechanisms that default OFF and
  leave the shipped CPU path bit-identical: the `unsided` spec token (every leg cell reads the side-mean leg-MN rate)
  and `LegCycle.flat_amplitude` (the per-leg amplitude held at 1). The channel-matched control's two parameters
  (hair_plate_max_hz 86.71, campaniform_load_hz 25.05) were derived on CPU as a fixed point of the same afferent ->
  leg-MN loop round 4 solved for mn_ref_hz, and verified by a fourth CPU run before submission. Every predeclared
  precondition passed (K's hair plate -2.02 Hz and campaniform -0.13 Hz from C's; U's chordotonal and hair-plate L-R
  exactly 0.000 against L's +13.564 / +9.204; M's walking amplitude exactly 1.000000 with |amp L-R| 0.000000 against
  C's 0.948 / 0.0170; the predeclaration stamped 04:54:59Z against the earliest run's own started_utc 04:56:11Z).
  RESULT. (1) The round-2 law's DC sidedness owns the level control's DRIFT and none of its DNa02: U v L is null on
  DNa02_L, DNa02_R and the clean yaw SD while it halves the drift (+3.48 -> +1.74 deg/s), raises straightness 0.483 ->
  0.714 and symmetrises AN04B003 (-1.01 L, +0.85 R). (2) The unmatched hair plate is real, sign-negative and small:
  -0.070 Hz of AN04B003 per Hz of hair plate in the room (-0.149 on an isolated cell), so L's +10.28 Hz hair-plate
  excess SUPPRESSES L's own AN04B003 by -0.72 Hz and therefore ACCOUNTS FOR +0.72 Hz -- 14 % -- of the +5.22 Hz
  pooled C-over-L difference (+0.01 chordotonal + 0.72 hair plate + 4.49 residual = 5.22); the campaniform
  contributes nothing (single cell -0.055 Hz, null, over the whole 24.9 -> 49.7 Hz step). (3) The per-phase
  modulation is the large term and is the best-supported thing in the round: a level model fitted on the three steady
  arms predicts every steady arm's AN04B003 to within 0.11 Hz RMSE, predicts a HELD-OUT steady arm to 0.33 Hz, and
  under-predicts the two CYCLE arms by +3.9 / +5.0 (C) and +4.5 / +5.7 Hz (M) -- 11-20 x that held-out bar, surviving
  twelve specifications (C's left-side residual +2.50 .. +5.02, never near zero) and a slope profile in which no
  admissible hair-plate slope removes it; the single-cell check gives the mechanism (+1.63 Hz at the same per-cell
  mean with IN13B001 clamped, z +7.2, reproduced to 0.000e+00 on an independent rerun). (4) MOST of M's excess over
  C is level -- about 60 % of its AN04B003 excess is its extra 6.0 Hz of chordotonal (+0.956 / +0.904 predicted of
  +1.615 / +1.649 observed) -- but "the amplitude law adds no drive" is NOT supported: M's structure residual EXCEEDS
  C's by +0.659 (z +3.8) and +0.745 (z +3.5) and on DNa02_L by +0.069 (z +4.0, essentially the whole M-over-C
  difference), so F4 stays open; what the turn term does own is the per-frame sided DNa02 signal (-0.334 Hz under C
  vs -0.107 under M) and the straighter walk, and it is not a bigger afferent alternation under C (mean per-frame
  |chordotonal L-R| 12.13 C vs 13.02 M). CAVEAT THAT LIMITS EVERYTHING: the predeclared Holm family size m = 7 is
  unsatisfiable at 5 v 5 (floor p 0.0079 x 7 = 0.0556 > 0.05), so NO row is CALLED; every verdict quoted is
  `compare`'s, with the z and the per-run scatter. That defect is a failure to apply docs/INTERP.md 10.2, a standing
  rule from object round 2, not a rule discovered here. M v K is level-mismatched (+14.7 Hz) and carries no
  attribution; C v K and M v K are upper bounds by the predeclared P1b rule (K loses 8.7 Hz of chordotonal to the
  loop), and the DNa02 upper bound is loose by about a sixth. The level model's published slope SEs are
  pseudo-replicated over a design with only six distinct (arm, side) points -- on the six arm-side means they are
  b +0.1848 +- 0.0355, c -0.0648 +- 0.0285, so the hair-plate slope is 2.3 sigma from zero, not 6.8 -- and the cycle
  arms sit 9.1 (C) and 11.8 (M) Hz of hair plate off the steady arms' near-line in the joint plane, so the cycle
  predictions are joint extrapolations. INTEGRITY: an independent Opus skeptic pass (verdict MOSTLY SOUND) found
  that this audit's section-5 per-seed lists for U, K, M and C were NOT the data -- they had been made to carry the
  published means; they are replaced from analysis/room_table.csv `*_runs` and the substitution is stated in
  section 5. Nothing is adopted and no default moved.
skeptic:
  source: "independent Opus pass, 2026-09-15"
  verdict: "mostly sound"
  integrity_note: "section 5 per-seed lists for U/K/M/C were not the data; replaced"
key_claims:
  - "INTEGRITY DEFECT, found by the skeptic pass and corrected here: the section-5 sentence 'Per-seed values behind the arms' printed DNa02_L per-run lists for U, K, M and C (and a clean-yaw-SD list for U) that occur nowhere in out/vncd5 -- they had been constructed to carry the published means rather than transcribed. They are replaced from analysis/room_table.csv `*_runs` (U [0.3917 0.3950 0.3720 0.3873 0.3892], K [0.3519 0.3632 0.3529 0.3261 0.3548], M [0.6187 0.6327 0.6227 0.6141 0.6165], C [0.5155 0.5343 0.5290 0.5144 0.5470]; U yaw SD [6.8111 6.6278 6.9263 6.7598 6.6147]), and section 5 says so in the document. Every mean, SD, diff, z, p and verdict in the same table reproduced exactly under the skeptic's independent reducer (all 49 rows)."
  - "Batch: ONE submission vncd5-2ffbc3 on house, `30 job(s), 0 failed (22.6 min)`, 30 room JSON + 120 npz, every run device cuda / NVIDIA B200 / family level2 / its own `sense` block / blocks fam_r0..fam_r4. Predeclared 2026-09-15T04:54:59Z against the earliest run's started_utc 04:56:11Z (an absolute timestamp in the run JSON: round 4's file-mtime caveat is closed). The skeptic pass reproduced the provenance, the single submission, the one compiled-connectome md5 ef23cc27bea13be7f6a96f3c04fd3737, the one 51-file fingerprint and all seven predeclared reducer hashes independently."
  - "Realised levels (run mean +- SD over 5 runs, recomputed from the recordings): chordotonal L 86.05 +- 0.70, U 84.84 +- 0.87, K 77.37 +- 0.81, M 92.12 +- 1.53, C 86.11 +- 1.67; hair plate 56.64 / 55.78 / 44.34 / 49.57 / 46.36; campaniform 49.54 / 49.59 / 24.78 / 24.87 / 24.91. Chordotonal L-R: L +13.564, U +0.000, K +8.640, M +0.032, C -0.304."
  - "F1 U v L (the sidedness): DNa02_L -0.0487 (z -1.8, null), DNa02_R +0.0059 (null), clean yaw SD -0.487 (z -2.8, null); straightness +0.2309 (z +9.7, result), DNa02 L-R -0.0545 (z -3.7, result), AN04B003_L -1.005 (z -7.8) and AN04B003_R +0.850 (z +4.4), pooled -0.08. The DC bias owns the drift and the straightness, not the DNa02 level -- round 4 skeptic R2 confirmed with an arm."
  - "F2 K v L (the channels): result NEGATIVE, not the predicted positive -- AN04B003 -0.940 / -0.608, DNa02_L -0.0859 -- because matching the two channels costs K 8.7 Hz of chordotonal through the loop (predeclared P1b; the loss is exactly the round-2 law, 15.84 Hz of chordotonal per Hz of leg MN, predicting -11.09 / -6.13 against -11.09 / -6.17 observed). The match itself worked: SNpp45's direct term on DNa02_L is +128.5 mV/s (K) vs +127.1 (C) vs +167.9 (L), and IN13B001 76.3 -> 61.6 Hz."
  - "F3 C v K: result on all seven, positive (DNa02_L +0.178, DNa02_R +0.119, yaw SD +0.944, AN04B003 +4.67 / +7.33) with P1-P2 met -- but P1b triggered, so these are UPPER BOUNDS, as predeclared. How loose: K's -8.735 Hz chordotonal deficit is worth +1.68 Hz of the AN04B003 difference and only +0.028 Hz of the +0.178 DNa02_L difference, so the DNa02 bound is loose by about a sixth."
  - "F4 M v C (the amplitude / turn term): result on DNa02_L +0.093, yaw SD +0.629, AN04B003 +1.615 / +1.649; null on DNa02_R, straightness and DNa02 L-R. M carries +6.0 Hz more chordotonal (amplitude 1.000 vs the law's realised 0.948), and on the level model about 60 % of its AN04B003 excess is that level (+0.956 / +0.904 of +1.615 / +1.649). THE REST IS NOT: M's structure residual EXCEEDS C's by +0.659 (z +3.8, result) and +0.745 (z +3.5, result), 6-7 x the fit's RMSE, and on DNa02_L by +0.069 (z +4.0, result) -- essentially the whole M-over-C DNa02_L difference. So the claim 'the amplitude law adds no drive' is NOT supported and F4 stays open; this batch cannot separate the amplitude law from the fact that a flat amplitude is also a different modulation waveform. What the turn term does own: E[DNa02 L-R | chord L-R>0] - E[.|<0] = -0.334 Hz (C) vs -0.107 (M), corr(DNa02 L-R, chord L-R) -0.143 vs -0.034, straightness 0.857 vs 0.794 -- and it is not a bigger afferent alternation under C (mean per-frame |chordotonal L-R| 12.13 C vs 13.02 M)."
  - "F5 C v U: result on DNa02_L +0.141, DNa02_R +0.101, yaw SD +0.977, straightness +0.143, AN04B003 +4.73 / +5.87 (null on DNa02 L-R), with the chordotonal matched to +1.27 Hz and the sidedness removed -- the hair plate is still +9.42 Hz higher in U, which at the measured slope inflates the AN04B003 excess by 0.66 Hz. F5 also leaves a -24.7 Hz campaniform mismatch (C 24.91, U 49.59) that the level model cannot correct for (the campaniform is not separable from the hair plate across three steady arms); the only evidence it does not matter is the single cell's campaniform null on a 752-cell subset, so F5's attribution is conditional on that."
  - "F6 M v K is VOID: realised chordotonal gap +14.74 Hz, outside the 10 Hz level tolerance. F7 L v A: result on all seven (DNa02_L +0.420, yaw SD +4.58, AN04B003 +18.47 / +16.19) -- the level alone fires DNa02 and turns the fly, as in round 4."
  - "NO ROW IS CALLED: the predeclared Holm family m = 7 cannot be satisfied at 5 v 5, where the exact Mann-Whitney floor is p 0.0079 and 0.0079 x 7 = 0.0556 > 0.05. This is a defect of this round's predeclaration (round 4 used m = 5 -> 0.0397), reported rather than repaired after the fact -- and it is a failure to apply a STANDING rule, docs/INTERP.md 10.2 from object round 2 ('declare quantities that are constant by construction as reported magnitudes outside the family, and size the family from the members that can move'), not a rule this round discovered. At 6 v 6 the floor is 0.0021645 and m <= 23."
  - "Level model (POST HOC, labelled): AN04B003_side = a_side + 0.1921 (+- 0.0128) x chordotonal - 0.0697 (+- 0.0103) x hair_plate, fitted on L / U / K (30 side-observations), RMSE 0.106 Hz, every steady-arm residual null (-0.108 .. +0.084). Cycle residuals: C +3.878 / +5.004, M +4.537 / +5.749. THE SEs ARE PSEUDO-REPLICATED: the design has only six distinct (arm, side) level points and the 30 rows are five runs of each; refitting on the six arm-side means (2 residual df) gives b +0.1848 +- 0.0355 and c -0.0648 +- 0.0285 (C's residual +3.92 / +5.07), so the hair-plate slope is 2.3 sigma from zero, not 6.8, and -0.070 Hz/Hz should be read as 'small, negative, order 0.07'. The steady arms lie close to one line in the joint (chordotonal, hair-plate) plane (corr 0.925 on the six means) and the cycle arms sit 9.1 (C) / 11.8 (M) Hz of hair plate off it, so the cycle predictions are joint extrapolations even though both marginal ranges contain them. What bounds the risk: a HELD-OUT steady arm is predicted to -0.33 .. +0.25 Hz (the honest out-of-sample bar -- the cycle residual is 11-20 x it, not 37-54 x an in-sample RMSE), the residual survives twelve specifications (C L-side +2.50 .. +5.02), and no hair-plate slope that removes it is compatible with the steady arms (c ~ -0.55 would be needed and degrades the steady-arm RMSE 9 x). Leave-one-arm-out: without L +3.60 / +4.66 (C), without U +4.08 / +5.57; without K the fit collapses. The DNa02 version of the model is NOT robust (without U, C's residual +0.049 / -0.176) and the IN13B001 version is ill-conditioned; neither is interpreted."
  - "Decomposition of the +5.22 Hz pooled C-over-L AN04B003 difference at the model's slopes, with the sign as the route requires: chordotonal +0.01 (matched); L's +10.28 Hz hair-plate excess SUPPRESSES L's own rate by -0.72 Hz and therefore ACCOUNTS FOR +0.72 Hz of the C-over-L difference (14 %); per-phase modulation +4.49. (+0.01 + 0.72 + 4.49 = 5.22.) The +5.22 Hz is THIS batch's pooled C-over-L difference; round 4's own was +5.55 Hz, a labelled cross-batch context number."
  - "Single cell (CPU subset brain, 752 cells, 5 runs): at the SAME per-cell chordotonal mean with IN13B001 clamped at 65.2 Hz, an 8 Hz-modulated input raises AN04B003 +1.628 Hz (21.717 -> 23.345, z +7.2, result); with IN13B001 free +1.398 (z +8.2); modulating the hair plate and campaniform too adds +2.055 (z +30.7). Raising the hair plate alone by +6.70 Hz raises IN13B001 +7.80 and lowers AN04B003 -1.000 Hz (z -5.9, result) = -0.149 Hz/Hz; the campaniform alone (24.9 -> 49.7 Hz) gives -0.055 Hz (z -0.3, null). An independent rerun on a different machine reproduced every column of rows.csv to 0.000e+00, and the clamp is effectively complete (the -0.40 Hz residual IN13B001 drift is worth +0.05 Hz of AN04B003, 3 % of the +1.628)."
  - "Both new mechanisms default OFF and the shipped path is bit-identical on CPU: pytest tests/test_proprioception.py tests/test_body_cycle.py tests/test_bit_identity.py = 33 passed, 5 subtests, the golden unchanged (reproduced independently). New tests pin that 'all' with the token absent is byte-equal to the shipped construction over a fed frame, and that LegCycle() and LegCycle(flat_amplitude=False) are identical over 12 BatchSim steps. One harmless asymmetry: the flat branch is where(air | ~isfinite(tau_st), 0, 1) while the default is where(air, 0, v_leg * tau_fin / step_ref), so the two differ when v_leg clips to 0 (|yaw| above ~515 deg/s at 9 mm/s); M's realised amplitude has exactly two distinct values {0, 1}, so it never bit in this batch."
  - "Provenance defect R9 of round 4 is closed: every run JSON carries a `sense` block (spec, every token, mn_ref_hz, hair_plate_max_hz, campaniform_load_hz, the explicit overrides, which parameters are defaults) plus started_utc / finished_utc."
  - "Round 4's decision pair re-run inside this batch: C v L result on DNa02_L (+0.092), DNa02_R (+0.107), straightness (+0.374), AN04B003 (+3.73 / +6.72); the clean-yaw-SD VERDICT flips (result in round 4 at +0.572 / z +7.1, null here at +0.490 / z +2.8) but the DIFFERENCE replicated to 86 % and the rank separation is complete in BOTH batches (exact p on the 5 v 5 floor, every C run above every L run). What moved is L's own between-run SD, 0.081 -> 0.172, which is the z denominator. Pooled 10 v 10 (a labelled cross-batch statement, not a compared row): +0.531 Hz, p 1.1e-05. Neither batch is an outlier."
files_written:
  - flyverse/senses.py (the `unsided` and `leg_cycle_flat` tokens; both default absent, every default unchanged)
  - flyverse/body.py (LegCycle.flat_amplitude, default False)
  - scripts/probe_vnc_drive.py (family level2, ARM_SENSE_KW / sense_kwargs_of / sense_record, --hair-plate-max-hz / --campaniform-load-hz, the `sense` + started_utc / finished_utc provenance block, chain_rates, holm_families and `pairs --predeclared`, WATCH_LEVEL2, the flat cycle in attach_cycle, plan's refusal for an underived control)
  - scripts/probe_an04b003_single_cell.py (new)
  - tests/test_proprioception.py (UnsidedTokenTests), tests/test_body_cycle.py (FlatAmplitudeTests)
  - docs/audits/level_controls.md
  - out/vncd5/ (batch.sh, predeclared.json, tree_state.json, channel_match_derivation.json, derive_channel_match.py, verify_runs.py, level_model.py, cal/, smoke/, single_cell/, 30 room runs, analysis/, run_verification.csv, arm_levels.csv, precondition_checks.csv, level_model*.csv/json, post_analysis_sha.json)
  - out/vncd5_cluster.log
api:
  - "senses.Proprioception spec tokens: 'unsided' (leg channels read the side-mean leg-MN rate; refused with 'leg_cycle' and without an MN-rate leg channel) and 'leg_cycle_flat' (requires 'leg_cycle'; tells the caller to build LegCycle(flat_amplitude=True) and records the fact in .spec). Both in FLAGS, both absent by default; .unsided / .leg_cycle_flat are the attributes."
  - "body.LegCycle(flat_amplitude=True): amp_i = 1 for every walking leg (0 standing and airborne); phase, stance, freq, beta, load_L/R, n_stance unchanged. Default False."
  - "probe_vnc_drive.py room --family level2 --arm A|L|U|K|M|C [--mn-ref-hz X --hair-plate-max-hz Y --campaniform-load-hz Z]: the values default to ARM_MN_REF / ARM_SENSE_KW for the family's labelled controls, else the sense's own defaults; plan refuses a family whose control parameters are underived."
  - "probe_vnc_drive.py pairs --predeclared <stamped json>: Holm-calls the stamped primaries.families against pairwise.csv into decision_table.csv (CALLED = compare result AND p_holm <= alpha)."
  - "run JSON keys: `sense` {spec, tokens, mn_ref_hz, hair_plate_max_hz, campaniform_load_hz, params, overrides, defaults_used}, `started_utc`, `finished_utc`; pairwise keys `chain_<type>_<L|R|?>_hz` from the per-fly recordings. Per-run scatter lives in analysis/room_table.csv `*_runs` and analysis/pairs_console.txt; pairwise.csv carries mean, sd and n only."
validation:
  - "pytest tests/test_proprioception.py tests/test_body_cycle.py tests/test_bit_identity.py: 33 passed, 5 subtests (CPU); the bit-identity golden unchanged; reproduced independently on a second machine"
  - "CPU smokes of the three new arms (U / K / M, 2 flies x 0.5 s): exit 0, each with the right sense block; three CPU calibration runs for the K derivation, the fourth-step verification (hair 46.449 against the 47.065 target, campaniform 25.050 against 24.887) accepted without re-fitting"
  - "batch: 30 job(s), 0 failed (22.6 min); 30 room JSON + 120 npz + 30 consoles; every run cuda / NVIDIA B200 / family level2 / correct spec and sense block / blocks fam_r0..fam_r4 / one compiled-connectome md5 / 51-file fingerprint"
  - "predeclared preconditions P1-P6 all PASS, checked from the recordings before any verdict (verify_runs.py, precondition_checks.csv); the added level-gap check fails only for M v K (+14.74 Hz), which is therefore void"
  - "the new reducers were validated against the PREVIOUS batch first: on out/vncd4 with round 4's predeclared.json, holm_families reproduces round 4's published calls exactly and chain_rates reproduces its skeptic pass's AN04B003 / IN13B001 numbers"
  - "the single-cell probe is deterministic on CPU: a second run at the same seeds reproduced every row and every contrast exactly, and an INDEPENDENT rerun on a different machine reproduced every column of rows.csv to 0.000e+00"
  - "an independent Opus skeptic pass recomputed the batch from the raw recordings with its own reducer: all 49 decision-table rows, the realised levels, the 8.1 / 8.2 / 8.3 rows, the six preconditions, the channel-match derivation and the Holm arithmetic reproduce exactly; the four section-5 per-seed lists did not, and are corrected here"
recommendations:
  - "Size the Holm family against the exact-U floor BEFORE submission: at n v n the smallest attainable adjusted p is p_floor(n,n) x m, so 5 v 5 admits m <= 6 and 6 v 6 admits m <= 23, and this round's m = 7 at 5 runs called nothing. Either drop a primary, or run 6 runs per arm. docs/INTERP.md 10.2 already required this."
  - "Re-run the modulation-only arm at the cycle's own realised amplitude (0.948, not 1.000) so that M and C sit at one level; as run, M buys +6.0 Hz of chordotonal, F4 / F6 cannot be read as level contrasts, and M's structure residual exceeds C's by 0.66-0.75 Hz."
  - "Derive mn_ref_hz, hair_plate_max_hz and campaniform_load_hz TOGETHER as one three-parameter fixed point, so the channel-matched control keeps the chordotonal level too; that turns C v K from an upper bound into a measurement."
  - "Use the level model as a designed arm set, not a post-hoc fit: steady arms that deliberately span the (chordotonal, hair-plate) PLANE rather than a near-line -- the three here have corr 0.925 and the cycle arms sit 9-12 Hz of hair plate off that line -- plus the cycle arms, predeclared as such, and quote arm-level SEs over the distinct level points rather than OLS SEs over the runs."
  - "Emit every per-run / per-seed list the prose quotes into a named file from the analysis script and paste it from there; this round's section-5 lists were retyped and four of six were wrong (docs/INTERP.md 10.4, the rule this round added)."
  - "The suite under all+leg_cycle (5 runs, one block) before any default discussion; the object / loom / feeding rows have still never run with the cycle."
open_questions:
  - "Is the ~4.5 Hz modulation term at AN04B003 enough to matter downstream? DNa02's own level model is too weak to carry a residual, and the pairs put C above every steady control by 0.09-0.18 Hz -- real, but not yet separable from the level with this batch's arms. A consistency check the audit did not run: the steady arms' own relay gain dDNa02/dAN04B003 = +0.03425 Hz/Hz turns C's 4.441 Hz mean structure residual into +0.152 Hz of DNa02 against an observed C-over-U +0.121 -- correlational, not a proof."
  - "Does the amplitude law add drive? F4 is OPEN: M's structure residual exceeds C's by +0.659 / +0.745 Hz (z +3.8 / +3.5) and by +0.069 on DNa02_L (z +4.0), and this batch cannot separate the amplitude law from the fact that a flat amplitude is also a different modulation waveform. The arm that would settle it is M at the cycle's own realised amplitude 0.948."
  - "Why does the tripod-locked DNa02 L-R swing need the amplitude law (-0.334 C vs -0.107 M) when AN04B003's antiphase does not (-3.58 vs -3.80)? The turn term is 1.7 % of the alternation at the afferents and yet it is what makes the DNa02 sided signal -- and it is not a bigger afferent alternation under C (mean per-frame |chordotonal L-R| 12.13 C vs 13.02 M)."
  - "The clean-yaw-SD row of C v L flipped from result (round 4, +0.572, z +7.1) to null (here, +0.490, z +2.8) because the CONTROL arm's between-run SD doubled (0.081 -> 0.172), not because the effect moved: the difference replicated to 86 %, the rank separation is complete in both batches and the pooled 10 v 10 difference is +0.531 Hz at p 1.1e-05. Neither batch is an outlier; what is open is why L's scatter doubled between two batches of the same arm."
  - "The cycle's realised per-leg amplitude (0.948) is now a second unmeasured quantity of the same kind as half_width_m: the modulation-only arm had to choose a value for it, and 1.000 was the wrong choice."
```

## Skeptic pass (independent, 2026-09-15)

Everything below was recomputed from `out/vncd5/*_body.npz` / `*_flies.npz` / `room_*.json` with my own reducer
(clean-frame rule, same-mask DNa02, chain rates, `compare`, Holm), from `out/vncd4/*_body.npz` / `*_flies.npz` for the
cross-batch rows, and by re-running the single-cell probe and the two bit-identity test classes on this desktop
(CPU, `CUDA_VISIBLE_DEVICES=-1`). No house job was submitted; nothing was needed that recomputation could not settle.
Scratch scripts: `.../scratchpad/s1_prov.py`, `s2_prim.py`, `s3_cmp.py`, `s4_vncd4.py`, `s5_level.py`, `s6_loo.py`;
outputs `my_primaries.csv`, `my_decision_table.csv`, `my_level_rows.csv`, `sc_repro/`.

---

### Refuted

**R1. The per-seed lists in section 5 are not the data (four of six arms wrong).** Section 5 line 302-306 prints
"Per-seed values behind the arms (r0..r4)". A and L reproduce exactly. U, K, M and C do not:

| arm | audit prints | the data (`room_table.csv` `DNa02_L_hz` `*_runs`, `pairs_console.txt` line 190, the run JSONs' `run.DNa02_L_hz`, and my own recomputation from `cmd__DNa02_L` on the post-skip non-airborne mask -- all three agree) |
|---|---|---|
| U | `[0.3917 0.3950 0.3720 0.3859 0.3903]` | `[0.3917 0.3950 0.3720 0.3873 0.3892]` (r3, r4 wrong) |
| K | `[0.3316 0.3416 0.3549 0.3561 0.3648]` | `[0.3519 0.3632 0.3529 0.3261 0.3548]` (all five wrong) |
| M | `[0.6104 0.6252 0.6199 0.6224 0.6265]` | `[0.6187 0.6327 0.6227 0.6141 0.6165]` (all five wrong) |
| C | `[0.5099 0.5307 0.5338 0.5338 0.5316]` | `[0.5155 0.5343 0.5290 0.5144 0.5470]` (all five wrong) |

The same sentence's clean-yaw-SD list is wrong for U in two places too: audit `[6.811 6.628 6.926 6.756 6.619]`
against `[6.8111 6.6278 6.9263 6.7598 6.6147]` (`room_table.csv`, `pairs_console.txt` line 202, my recomputation).
None of the printed U/K/M/C values occurs anywhere in `out/vncd5` as a `DNa02_L_hz` per-run value. Each fabricated
list has been made to carry the published mean to four decimals (K 0.3498, M 0.6209, C 0.5280) and an SD close to but
not equal to the published one (K: printed list gives 0.0131 against the file's 0.0140). **The means, the SDs, the
diffs, the z and every verdict in the section-5 table are correct -- I reproduced all 49 rows exactly -- but the
per-seed scatter quoted in prose is not the data and must be replaced or deleted.** This is the one finding in this
pass that is an integrity problem rather than a reading problem.

**R2. "M and C carry the SAME structure residual" is refuted by the audit's own file.** Section 0 item 4 and F4
conclude "the amplitude law contributes no extra drive to DNa02 beyond its own level" from
"M's structure residual is C's (+4.54 / +5.75 against +3.88 / +5.00)". Those are not the same number. Taking the
per-run residuals straight out of `out/vncd5/level_model_predictions.csv` and applying the project's own `compare`:

* AN04B003 L side: M +4.537 vs C +3.878, **diff +0.659, z +3.8, p 0.0079, `result`**
* AN04B003 R side: M +5.749 vs C +5.004, **diff +0.745, z +3.5, p 0.0079, `result`**
* DNa02 L side: M +0.2181 vs C +0.1488, **diff +0.0693, z +4.0, p 0.0079, `result`**

The difference is 6-7 x the model's own RMSE (0.106 Hz). Decomposed: of M's +1.615 Hz AN04B003 excess over C on the
L side, the level model attributes +0.956 (0.1921 x 6.174 Hz of chordotonal - 0.0697 x 3.293 Hz of hair plate) and
leaves **+0.659 Hz unexplained**; of the +0.093 Hz M-over-C DNa02_L difference the level accounts for only +0.020
(0.0032 x 6.17) and the residual difference is +0.069 -- i.e. essentially the whole of it. **The honest F4 reading is
"most (about 60 %) of M's excess over C is its extra 6.0 Hz of chordotonal; the rest is not, and this batch cannot say
whether the remainder is the amplitude law or the fact that a flat amplitude is also a different modulation
waveform."** Not "no extra drive."

**R3. The section-7 decomposition does not add up, and the hair-plate term has the wrong sign.** Section 7 last
paragraph: "the chordotonal term is +0.01 ..., the hair-plate term is **-0.72** ..., and the residual **+4.4** Hz is the
modulation", against an observed +5.22. `0.01 + (-0.72) + 4.4 = 3.69`, not 5.22. The arithmetic that produces 4.4 uses
**+0.72**: `5.222 - 0.1921 x (+0.057) - (-0.0697) x (-10.276) = 5.222 - 0.011 - 0.716 = +4.495`. L's hair-plate excess
*suppresses L's own rate* by -0.72 Hz, which *contributes +0.72 Hz to the C-over-L difference*. Section 6 F5 states the
identical logic correctly ("the hair plate is still +9.42 Hz higher in U, which ... ACCOUNTS FOR -0.66 Hz of AN04B003 in
U, i.e. works in C's favour and **inflates** the C v U excess by that much") and section 13 adds the two magnitudes
(`1.44 + 1.63 = 3.07`). So this is a sign slip in two places (section 0 claim 2 and section 7), not a wrong number.

**R4. `pairwise.csv` does not carry per-run values.** Section 5: "the rest are in `pairwise.csv` (every key carries its
five per-run values)". `out/vncd5/analysis/pairwise.csv` has 51 columns -- `key`, six arms x (`mean`, `sd`, `n`), and
eight pairs x (`diff`, `z`, `p`, `verdict`). There is no `*_runs` column. The per-run values are in
`analysis/room_table.csv` (`*_runs`) and in `analysis/pairs_console.txt`'s per-arm block.

**R5. "The clean-yaw-SD row of C v L did not replicate" mischaracterises the two batches.** Recomputing both batches
with one reducer (`s4_vncd4.py`):

| batch | L (5 runs) | C (5 runs) | diff | z (= diff / SD(L)) | exact MWU p | verdict |
|---|---|---|---|---|---|---|
| `out/vncd4` | 7.300 +- 0.081 `[7.392 7.172 7.313 7.296 7.329]` | 7.872 +- 0.160 `[8.055 7.952 7.836 7.892 7.626]` | **+0.572** | +7.1 | 0.0079 (floor) | `result` |
| `out/vncd5` | 7.235 +- 0.172 `[7.003 7.189 7.228 7.274 7.483]` | 7.725 +- 0.134 `[7.832 7.781 7.674 7.822 7.514]` | **+0.490** | +2.8 | 0.0079 (floor) | `null` |

The **effect replicated to 86 %** and the rank separation is **perfect in both batches** (every C run above every L
run, so the exact p sits on the 5 v 5 floor in both). What changed is the *denominator*: L's own between-run SD went
0.081 -> 0.172. Pooling the two batches, 10 v 10: **diff +0.531, p 1.08e-05 (the 10 v 10 floor), min(C) 7.514 >
max(L) 7.483.** So the audit's open question "which of the two batches is the outlier is unknown" has an answer:
**neither** -- the two batches agree on the difference and disagree only on one arm's scatter, and jointly they support
C > L on the clean yaw SD. The verdict flip is real under `compare` and should be reported as such, but "a row that
does not survive a second batch" overstates it.

**R6. "+5.22 Hz" is this batch's number, not round 4's.** Section 7: "Applied to round 4's unexplained C-over-L
difference (+5.22 Hz pooled)". Recomputed pooled AN04B003 C-over-L: `out/vncd4` **+5.548**, `out/vncd5` **+5.222**.
Section 13 uses round 4's +5.55 correctly; section 7's label is wrong (the number is right for `vncd5`).

**R7. Section 10 item 1's 6 v 6 family bound is off by one.** `p_floor(6,6) = 2/C(12,6) = 0.0021645`;
`0.0021645 x 23 = 0.04978 <= 0.05`, so **m <= 23**, not 22. (The 5 v 5 bound, m <= 6, is right:
`0.0079365 x 6 = 0.047619`.)

**R8. The level model's "span" is a marginal statement that hides a joint extrapolation.** Section 7 justifies the
fit with "between them span chordotonal 71.6-93.5 Hz and hair plate 41.0-61.6 Hz per side" and `level_model.json`'s
caveat says the cycle arms are extrapolations "only in so far as their levels fall outside the fitted range". Both
marginal ranges contain C and M. But the three steady arms lie on a near-line in the joint plane -- `corr(chord, hair)
= 0.925` over the 30 side-observations **and 0.925 over the six (arm, side) means**, `hair = 0.974 x chord - 28.35`,
R^2 0.855 -- and the cycle arms sit far off it:

| point | chord | hair | hair the steady manifold has at that chord | off-manifold | Mahalanobis^2 from the steady cloud |
|---|---|---|---|---|---|
| steady points (6) | 72.97 - 92.70 | 41.75 - 61.12 | -- | -- | 0.49 - 3.13 |
| C L / R | 85.96 / 86.26 | 46.28 / 46.45 | 55.38 / 55.68 | **-9.11 / -9.23 Hz** | **13.84 / 14.27** |
| M L / R | 92.13 / 92.10 | 49.57 / 49.57 | 61.40 / 61.37 | **-11.83 / -11.80 Hz** | **25.22 / 25.09** |

The prediction for each cycle arm therefore runs ~9-12 Hz of hair plate *perpendicular to the data*. It does **not**
destroy the conclusion (see C9), but the "spans" sentence should not be read as "the cycle arms are interpolated".

**R9. The published slope SEs are pseudo-replicated.** `b = +0.1921 +- 0.0128`, `c = -0.0697 +- 0.0103` are OLS SEs on
30 rows with 4 parameters -- but the design has only **six distinct (arm, side) level points**; the 30 rows are five
runs each of the same six points, and the run-to-run scatter within an arm is far smaller than the between-arm
spacing. Refitting on the six (arm, side) means (2 residual df) gives **`b = +0.1848 +- 0.0355`, `c = -0.0648 +-
0.0285`** -- SEs ~2.8 x larger, and the hair-plate slope is then **2.3 sigma from zero, not 6.8**. The residuals are
unaffected (+3.916 / +5.073 for C). The slope numbers should carry the arm-level SE, and `-0.070 Hz/Hz` should not be
quoted to two significant figures as if measured.

**R10. The Holm defect is a violation of a rule that was already written down.** Section 0 and section 10 item 1
present `m = 7 x p_floor(5,5) = 0.0556` as this round's discovery ("the rule for the next round is in section 10 item
1"). `docs/INTERP.md` 10.2 already says, from object round 2: *"A family member whose statistic is constant by
construction is not a test ... still inflates the Holm denominator ... **Declare such quantities as reported magnitudes
outside the family**, and size the family (and therefore the arm count) from the members that can move"* -- and records
that this is exactly why that round's arms went from 5 to 6 runs. The predeclaration step failed to apply a standing
house rule. That is worth saying plainly, because "the rule for the next round" implies it did not exist.

**R11. Section 0 item 4 leans on a model sections 7 and 9 disqualify.** "The amplitude / turn term is not what raises
DNa02 ... the amplitude law contributes no extra drive to DNa02 beyond its own level" is a DNa02 level-model statement.
Section 7 says the DNa02 fit "is not robust" (dropping U flips C's R residual to -0.176) and section 9 lists it among
what the batch does not show. A claim cannot rest on a model the same document declares uninterpretable. (Combined
with R2, the DNa02 residuals in fact separate M from C at z +4.0 on the L side.)

---

### Confirmed

**C1. Provenance, stamps, one submission -- all clean (a).** 30 room JSON + 120 npz + 30 consoles; every run
`device cuda`, `device_name NVIDIA B200`, `family level2`, correct `proprioception` spec, correct `sense` block
(`mn_ref_hz` / `hair_plate_max_hz` / `campaniform_load_hz` / tokens / overrides present in every one -- round 4's
`mn_ref_hz: null` defect cannot occur), blocks `fam_r0..fam_r4`, batch 16, 60 s, skip 5 s; **one** compiled-connectome
md5 `ef23cc27bea13be7f6a96f3c04fd3737`, **one** 51-file source fingerprint (identical SHA over the file map in all 30),
**one** `cluster_run.py` invocation in `batch.sh` (`grep -c` = 1), console `30 job(s), 0 failed (22.6 min)`.
Ordering: derivation 04:52:11Z < **predeclaration 04:54:59Z** < tree state 04:55:14Z < submitted 04:55:38Z <
earliest `started_utc` **04:56:11Z** (72 s after the stamp) < latest finish 05:08:06Z. Zero runs start before the
stamp. The six predeclared reducer SHA-256s are byte-identical to the stamp now (`post_analysis_sha.json` `changed:
[]` -- I recomputed all seven hashes independently and they match). `verify_runs.py` and `level_model.py` are correctly
declared non-predeclared. Nothing was changed after the data were seen.

**C2. Every one of the 49 decision-table rows reproduces exactly (b).** My own reducer (post-skip non-airborne mask
per fly for DNa02_L/_R/_LR; my own clean-frame rule -- window, on the table top before and after, no airborne frame
within 50 frames, |yaw| <= 720 deg/s -- for the yaw SD; net/path per fly for straightness; every cell of the type per
side from `_flies.npz` for AN04B003) reproduces every mean, SD, diff, z, p and verdict in section 5 to the printed
precision, including the four `null` cells (U v L DNa02_R, K v L DNa02_R, M v C DNa02_R / straightness / DNa02 L-R,
C v U DNa02 L-R) and the C v L context row. Realised levels (4.1) reproduce to 3 decimals for all six arms and both
sides, as do the 8.1 decomposition rows, the 8.2 chain rates and the 8.3 sidedness rows.

**C3. The Holm arithmetic is exactly as stated (b).** `p_floor(5,5) = 2/C(10,5) = 0.0079365`; the smallest attainable
adjusted p at m = 7 is `0.0555556 > 0.05`; **0 rows called** in my independent Holm implementation, matching
`decision_table.csv`. The declared decision rule was **unsatisfiable before a single job ran**. `m <= 6` at 5 v 5 is
right. Round 4's m = 5 at `0.0397` is right, and I confirmed the reducer-validation claim: on `out/vncd4` the round-4
C v L DNa02_L row reproduces at **+0.1272, z +9.2** (Holm 0.0397, CALLED).

**C4. All six predeclared preconditions reproduce from the recordings (c).** P1 chordotonal L / U / K vs C
**-0.057 / -1.271 / -8.735** Hz; P1b triggered (-8.735 > 3); P2 K's hair plate **-2.022**, campaniform **-0.132** Hz;
P3 U's chordotonal L-R **+0.000003**, hair-plate L-R **0.000000** against L's **+13.5643 / +9.2043**; P4 M's
per-leg amplitude on walking frames **1.000000** (two distinct values, {0, 1}; max 1.000000) and `|amp L-R|`
**0.000000** against C's **0.947728** (5th-95th percentile 0.919-0.975, max 1.547, 0.6 % of frames above 1.0) and
0.017019; P5 every channel in bracket; P6 as C1. The added post-hoc level-gap check fails only for M v K
(**+14.744 Hz**), exactly as reported. Per-side chordotonal means: L 92.701 / 79.137, U 84.838 / 84.838,
K 81.610 / 72.970, M 92.134 / 92.102, C 85.960 / 86.264.

**C5. The channel-match derivation is arithmetically correct step by step (f).** `d = (57.2626-5)/95 = 0.550133`;
`hp = 5 + 42.0646/0.550133 = 81.463`; `load = 24.8868/0.99342 = 25.052`; `cal_K1` `d = (43.2943-5)/76.46 = 0.500841`;
secant slope `+0.0026587 / Hz`; fixed point `x = 86.7105`; `cal_K2` realises 46.449 (error -0.616) and 25.050 --
inside the predeclared +-5 / +-2.5, nothing re-fitted afterwards.

**C6. K's 8.7 Hz chordotonal loss is exactly the round-2 law, independently derived (f).** The law is
`chord = 10 + 140 x clip(legMN/8.84, 0, 1)`, so `d(chord)/d(legMN) = 15.84 Hz/Hz`. K's realised leg-MN rates fall
0.7004 (L) / 0.3873 (R) below L's, predicting **-11.09 / -6.13 Hz** of chordotonal; observed **-11.09 / -6.17**. The
loop attribution in section 3 and section 6 F2 is right, and the mechanism is fully accounted for. **"Upper bound" is
the right reading of C v K, and it is tight**: at the level model's chordotonal slope the -8.735 Hz deficit is worth
+1.68 Hz of the +4.67 / +7.33 AN04B003 difference and, at the DNa02 slope, only **+0.028 Hz of the +0.178 DNa02_L
difference (16 %)** -- so the DNa02 upper bound is loose by about a sixth, which the audit could say.

**C7. The single-cell check reproduces bit-for-bit on an independent rerun, and the clamp is a real clamp (e).**
I re-ran `scripts/probe_an04b003_single_cell.py --runs 5` to a scratch directory: **max |original - rerun| = 0.000e+00
on every column of `rows.csv`** (AN04B003, per side, IN13B001, all three commanded channels). The determinism claim in
section 11 is confirmed by a second machine-run, not just asserted.
* The clamp zeroes **all** of IN13B001's incoming synapses in the subset `W` (`W2[in13_sub, :] = 0.0`) and drives the
  cell by Poisson at 65.2 Hz every frame. Realised IN13B001 in the two clamped rows: 65.30 +- 0.36 vs 64.90 +- 0.77 --
  a -0.40 Hz drift that is `null` under `compare` and worth, at the isolated cell's own `dAN04/dIN13` = -0.128 Hz/Hz,
  **+0.05 Hz of AN04B003 -- 3 % of the +1.628**. The clamp is effectively complete.
* The per-cell mean is matched by construction and in fact: commanded chordotonal **92.0247 (steady) vs 92.0244
  (modulated)**, a 3e-4 Hz difference; hair plate and campaniform identical to 4 decimals.
* `z = +7.2` comes from a **real** reference SD (0.2269 over five runs, not a degenerate one): `1.628 / 0.227 = 7.17`,
  p at the 5 v 5 floor. Same for the hair-plate row (-1.000, SD 0.170, z -5.9) and the campaniform null
  (-0.055, z -0.3, p 0.548).
* The per-Hz slopes check: `-1.000 / 6.70 = -0.1493 Hz/Hz`; `dIN13B001/dHair = 7.80 / 6.70 = 1.164`;
  `9.69 x 0.1493 = 1.447`; `1.44 + 1.63 = 3.07`. All as printed.

**C8. The level model refits exactly, and the AN04B003 residual survives every specification I could construct (d).**
Base fit reproduced to 4 decimals: `b = +0.1921`, `c = -0.0697`, RMSE 0.1059, residuals C +3.878 / +5.004,
M +4.537 / +5.749. It is fitted on **L, U, K only** (arm A is excluded -- it has no sense and no channels) and C and M
are genuinely out of sample. C's L-side residual under every alternative I tried:

| specification | C L / R residual |
|---|---|
| author (chord + hair) | +3.878 / +5.004 |
| chordotonal only | +4.421 / +5.833 |
| hair plate only | +5.021 / +6.900 |
| + campaniform | +2.503 / +3.565 |
| + haltere | +3.542 / +4.678 |
| + campaniform + haltere | +2.505 / +3.570 |
| + chord x hair interaction | +3.652 / +4.710 |
| leave-out L | +3.596 / +4.664 |
| leave-out U | +4.082 / +5.567 |
| leave-out K | collapses (b -5713, c +8419) |
| per-side fits (no shared slopes) | +3.855 / +4.933 |
| arm-mean fit (6 points, no pseudo-replication) | +3.916 / +5.073 |

**Range +2.50 .. +5.02 (L) and +3.57 .. +6.90 (R); it never approaches zero.** Two further tests the audit did not run
and should have, both of which *strengthen* it:
* **Held-out steady arm.** Fit on two steady arms, predict the *third steady arm* at its own levels: the error is
  **-0.330 / +0.116 (L held out) and -0.141 / +0.245 (U held out)**. That, not the in-sample RMSE, is the model's
  honest out-of-sample bar -- so the cycle residual clears it by **11-20 x**, not the "37-54 x RMSE" the audit quotes
  against an in-sample yardstick. The conclusion is unchanged and the honest multiple is still decisive.
* **Slope profile.** Fixing `c` and refitting: to drive C's L-side residual to zero you need `c ~ -0.55` (8 x the
  fitted value) and the steady-arm RMSE degrades 0.104 -> 0.953 (**9 x**); to zero M's you need `c ~ -0.445` and RMSE
  0.747 (**7 x**). **No hair-plate slope that eats the residual is compatible with the steady arms.** So R8's
  extrapolation is real but cannot explain the residual away.

The `+campaniform` fit's very different slopes (`c = -0.445`) are **not** an admissible refutation: the steady design
has six (arm, side) points and the base model already spends four of them; adding campaniform leaves 1 residual df and
adding haltere as well leaves **0** (saturated). Among the steady arms the campaniform is a near-binary
L/U-vs-K indicator, so it cannot be separated from the hair plate at all. The audit's stated reason for excluding it
(collinearity + the single-cell campaniform null) is the right call -- but it means the **-0.070 slope is identified by
one arm's displacement and is not separable from a campaniform term**; that risk should be named, because F5
(C v U, the only surviving structure-vs-level contrast) leaves a **-24.7 Hz campaniform mismatch** as well as the
-9.4 Hz hair-plate one, and the audit corrects only for the latter.

**C9. "Not robust and not interpreted" is a fair stop for the DNa02 model, not an evasion (d).** Across my
specifications C's DNa02 residual runs **-0.176 .. +0.485** and changes sign under leave-out-U on the R side; the
IN13B001 fit is worse still (adding the haltere drops its RMSE 2.08 -> 0.70 and its cycle residuals +20/+19 -> +4.7/+4.1,
i.e. the IN13B001 "residual" is mostly an unmodelled haltere term). Stopping there is correct. **But a cleaner route
exists that the audit did not take and that supports its DNa02 sentence:** regressing DNa02 on AN04B003 with side
intercepts across the three steady arms gives a relay gain **dDNa02/dAN04B003 = +0.03425 Hz/Hz (RMSE 0.025)**. C's
mean AN04B003 structure residual (4.441 Hz) then predicts **+0.152 Hz** of DNa02, against an observed C-over-U DNa02
of **+0.121 Hz** (C-over-K +0.149, C-over-L +0.101). The DNa02 excess is what the AN04B003 residual is worth through
the steady arms' own relay gain. This is correlational and AN04B003 is itself collinear with the levels, so it is a
consistency check, not a proof -- but it is a better argument than "it rests on the pairs".

**C10. Both new mechanisms are opt-in, default OFF, and bit-identical on the shipped path (i).** I ran
`PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_proprioception.py tests/test_body_cycle.py
tests/test_bit_identity.py -q` -> **33 passed, 5 subtests passed**, matching `out/vncd5/smoke/pytest_cpu.txt`
exactly, golden included. Reading the diffs: `unsided` is four lines (`FLAGS`, the attribute, two grammar refusals,
and `if self.unsided: lL = lR = 0.5 * (lL + lR)` before the law), refused with `leg_cycle` and without an MN-rate leg
channel, and touches nothing else; `LegCycle.flat_amplitude` defaults `False` and only swaps the `amp` line. One
harmless asymmetry worth knowing: the flat branch is `where(air | ~isfinite(tau_st), 0, 1)` while the default is
`where(air, 0, v_leg * tau_fin / step_ref)`, so the two differ when `v_leg` clips to 0 (|yaw| above ~515 deg/s at
9 mm/s); M's realised amplitude has exactly two distinct values {0, 1}, so it never bit in this batch.

**C11. Wording (i).** "Silent" occurs once, in section 8.2, and is used correctly to say **no** cell is called silent
(I confirmed every arm including A has a nonzero rate for every tabulated type). "Inverted" does not occur. The
descriptive verbs check out: U halves the drift (1.740 / 3.481 = 0.4999), cuts DNa02 L-R by a third (0.108 / 0.162 =
0.667), the clean fraction spans 1.177 x (0.999 / 0.849, inside the 2 x rule of `docs/INTERP.md` 10.4). The fractions
`(X - A) / (C - A)` reproduce (DNa02_L 0.820 / 0.725 / 0.653 / 1.181; straightness L 3.707).

**C12. What claims (1)-(4) get right.** (1) is confirmed: U v L is `null` on DNa02_L (-0.049, z -1.8), DNa02_R
(+0.006), the yaw SD (-0.487, z -2.8) while straightness goes +0.231 (z +9.7), the drift +3.481 -> +1.740 deg/s and the
relay symmetrises (-1.005 L / +0.850 R, pooled -0.078). (2) is confirmed in sign and order of magnitude, subject to R3
and R9. (3) is confirmed and is the strongest thing in the round (C8). (4) is half right: the per-frame sided DNa02
signal is the turn term's (-0.3336 C vs -0.1069 M, five non-overlapping runs each), and it is **not** explained by a
larger afferent alternation under C -- I recomputed mean per-frame `|chordotonal L-R|` and M's is *larger*
(13.02 vs 12.13 Hz) -- so open question 2 is a real puzzle honestly stated. The "adds no drive" half is R2.

---

### Corrections (exact replacement text)

1. **Section 5, the "Per-seed values" sentence (lines 302-306).** Replace the whole sentence with, verbatim from
   `room_table.csv` / `pairs_console.txt`:
   > Per-seed values behind the arms (r0..r4; `analysis/room_table.csv` `*_runs`, reproduced in
   > `analysis/pairs_console.txt`'s per-arm block): DNa02_L A [0.0113 0.0113 0.0205 0.0171 0.0161], L [0.4005 0.4251
   > 0.4333 0.4459 0.4736], U [0.3917 0.3950 0.3720 0.3873 0.3892], K [0.3519 0.3632 0.3529 0.3261 0.3548], M [0.6187
   > 0.6327 0.6227 0.6141 0.6165], C [0.5155 0.5343 0.5290 0.5144 0.5470]; clean yaw SD A [2.6841 2.5047 2.7246 2.5861
   > 2.7544], L [7.0034 7.1886 7.2284 7.2737 7.4825], U [6.8111 6.6278 6.9263 6.7598 6.6147]; every other key's five
   > per-run values are in `room_table.csv` and `pairs_console.txt` (`pairwise.csv` carries the run mean, SD and n
   > only).

2. **Section 0 claim 2, and section 7 last paragraph (the sign).** Replace
   *"L's +10.28 Hz hair-plate excess over C is worth -0.72 Hz of the +5.22 Hz pooled C-over-L AN04B003 difference"*
   and *"the hair-plate term is -0.72 (L's +10.28 Hz excess), and the residual +4.4 Hz is the modulation"* with:
   > L's +10.28 Hz hair-plate excess **suppresses L's own AN04B003 by -0.72 Hz**, which therefore **accounts for
   > +0.72 Hz of the +5.22 Hz pooled C-over-L difference (14 % of it)**; the chordotonal term is +0.01 (the channels
   > were matched), and the remaining **+4.49 Hz** is what the levels do not explain. (+0.01 + 0.72 + 4.49 = 5.22.)

3. **Section 7 attribution, and the Report.** Replace *"Applied to round 4's unexplained C-over-L difference (+5.22 Hz
   pooled)"* with:
   > Applied to **this batch's** C-over-L difference (+5.22 Hz pooled; round 4's own was +5.55 Hz, recomputed from
   > `out/vncd4/room_{C,L}_r*_flies.npz` -- a labelled cross-batch context number, not a compared row)

4. **Section 0 claim 4, section 6 F4, and the Report's claim 4.** Replace *"on the level model M's structure residual
   is C's (+4.54 / +5.75 against +3.88 / +5.00), so the amplitude law contributes no extra drive to DNa02 beyond its
   own level"* with:
   > On the level model **about 60 % of M's AN04B003 excess over C is its extra 6.0 Hz of chordotonal** (predicted
   > +0.956 / +0.904 of the observed +1.615 / +1.649). The rest is not: M's structure residual **exceeds** C's by
   > +0.659 (L, z +3.8, `result`) and +0.745 (R, z +3.5, `result`), 6-7 x the fit's RMSE, and on DNa02_L by +0.069
   > (z +4.0, `result`), which is essentially the whole +0.093 M-over-C DNa02_L difference. **This batch therefore
   > cannot say that the amplitude law adds no drive**; what it can say is that most of M's excess is level, and that
   > a flat amplitude is not only "no turn term" but also a different modulation waveform. The row the turn term does
   > own remains the per-frame sided DNa02 signal (-0.334 C vs -0.107 M), and it is not a bigger afferent alternation
   > under C: mean per-frame |chordotonal L-R| is 12.13 Hz (C) against 13.02 (M).

5. **Section 7's model table and the Report.** Add, after the RMSE column:
   > The design has only **six distinct (arm, side) level points**; the 30 side-observations are five runs of each, so
   > the OLS SEs above are pseudo-replicated. Refitting on the six arm-side means (2 residual df) gives
   > **b +0.1848 +- 0.0355, c -0.0648 +- 0.0285** (C's residual +3.92 / +5.07) -- the hair-plate slope is 2.3 sigma
   > from zero, not 6.8, and `-0.070 Hz/Hz` should be read as "small, negative, order 0.07", not as two significant
   > figures. The three steady arms also lie close to one line in the (chordotonal, hair-plate) plane (corr 0.925 on
   > the six means, hair = 0.974 x chord - 28.35): **C sits 9.1 Hz and M 11.8 Hz of hair plate off that line**, so the
   > cycle predictions are joint extrapolations even though both marginal ranges contain them. Two checks bound the
   > risk: fitting on two steady arms and predicting the **held-out steady arm** costs only -0.33 .. +0.25 Hz (the
   > honest out-of-sample bar, against which the cycle residual is 11-20 x, not 37-54 x); and **no hair-plate slope
   > that removes the cycle residual is compatible with the steady arms** -- `c = -0.55` would be needed and it
   > degrades the steady-arm RMSE 9 x. The residual also survives adding the campaniform and the haltere
   > (C +2.50 / +3.57), though at 5 and 6 parameters those fits are at or past saturation for a six-point design.

6. **Section 9, third-from-last bullet, and open question 3.** Replace *"The clean-yaw-SD row of C v L did not
   replicate as a result (z +2.8 here against round 4's +7.1)"* and *"Which of the two batches is the outlier is
   unknown"* with:
   > The clean-yaw-SD row of C v L is `result` in round 4 (+0.572, z +7.1) and `null` here (+0.490, z +2.8) -- but the
   > **difference** replicated to 86 % and the rank separation is complete in **both** batches (exact p on the 5 v 5
   > floor, 0.0079, every C run above every L run). What moved is L's own between-run SD (0.081 -> 0.172), which is
   > the z denominator. Pooled over the two batches (10 v 10, a labelled cross-batch statement, not a compared row):
   > +0.531 Hz, p 1.1e-05. **Neither batch is an outlier; the verdict flipped, the effect did not.**

7. **Section 10 item 1.** Replace *"at 6 v 6 (0.0022) m <= 22"* with *"at 6 v 6 (0.0021645) m <= 23"*, and add:
   > This is not a new rule: `docs/INTERP.md` 10.2 already requires the family to be sized from the members that can
   > move, and records object round 2 raising its arms from 5 to 6 runs for exactly this reason. Round 5's
   > predeclaration did not apply it.

8. **Section 6 F3 (a strengthening, optional).** After "these numbers are upper bounds", add:
   > How loose: at the level model's slopes K's -8.735 Hz chordotonal deficit is worth +1.68 Hz of the +4.67 / +7.33
   > AN04B003 difference, and only **+0.028 Hz of the +0.178 DNa02_L difference** -- so the DNa02 upper bound is loose
   > by about a sixth, not by an unknown amount.

9. **Section 6 F5 (a caveat that is missing).** After the hair-plate correction, add:
   > C v U also leaves a **-24.7 Hz campaniform mismatch** (C 24.91, U 49.59). The level model does not correct for it
   > because the campaniform cannot be separated from the hair plate across three steady arms; the only evidence that
   > it does not matter is the single cell's campaniform null (-0.055 Hz, z -0.3), on a 752-cell subset. F5's
   > attribution is conditional on that.

10. **Section 13 (a strengthening).** Add to the determinism note in section 11/12:
    > An independent rerun of `scripts/probe_an04b003_single_cell.py --runs 5` on a different machine reproduced every
    > row of `rows.csv` to 0.000e+00.

---

### Verdict

**Mostly sound.** The batch itself is the cleanest in the thread: one submission, provenance complete and stamped
before every run, six arms whose realised levels and preconditions I reproduced from the raw recordings, a decision
table of which all 49 rows recompute exactly, a Holm defect that is correctly diagnosed and honestly refused rather
than repaired, a control-parameter derivation whose every step checks out, and a single-cell mechanism check that
reproduces bit-for-bit on an independent rerun. The central scientific claim -- **claim (3), that at its own afferent
level the cycle's ascending relay fires several Hz more than any steady transducer of that level** -- is the best-
supported thing in the round and is, if anything, under-defended: it survives twelve specifications, a held-out
steady arm (0.33 Hz out-of-sample error against a 4-5 Hz residual), and a slope profile that shows no admissible
hair-plate slope can remove it. Claims (1), (5), (6) and the sign and smallness of the hair-plate route in (2) are
confirmed as written. What pulls the verdict down from "sound": one section prints per-seed numbers for four of six
arms that are **not the data** (R1); claim (4)'s "no extra drive" is contradicted at z +3.5 to +4.0 by the author's own
residual file (R2); the headline decomposition has a sign slip that makes it fail to add up (R3); claim (7)'s framing
turns a denominator change into a failed replication (R5); and the level model's slopes are quoted with
pseudo-replicated error bars over a design whose six points do not identify them (R8, R9). None of these is fatal to
the round's conclusion; all five are fixable in text with the numbers above, and R1 must be fixed before this document
is relied on.

**For NOTES -- what rounds 4 and 4b jointly say about the leg cycle.** Round 4 found the leg-cycle arm above a
level-matched steady control on DNa02 and the ascending relay but could not say why; round 4b ran the three controls
that split the difference and, within one 30-job submission whose predeclaration is stamped 72 s before the first run,
separated them: the round-2 law's DC sidedness owns the level control's drift and straightness and **none** of its
DNa02 rate (U v L `null` on DNa02_L, DNa02_R and the clean yaw SD while straightness goes +0.231 and the drift halves);
the unmatched hair plate is a real, sign-negative and **small** route (-0.07 Hz of AN04B003 per Hz in the room, -0.149
on an isolated cell, accounting for +0.72 Hz -- 14 % -- of the +5.22 Hz pooled C-over-L relay difference, with the
campaniform contributing nothing measurable); and the remainder, **~+4.5 Hz, is the per-phase modulation** -- a level
model fitted on three steady arms predicts a held-out steady arm to 0.33 Hz and under-predicts both cycle arms by
+3.9 to +5.7 Hz, a residual that survives every specification I could build and that no admissible hair-plate slope can
remove, with the single cell naming the mechanism directly (+1.63 Hz at a matched per-cell mean with IN13B001 clamped,
z +7.2, reproduced bit-for-bit). Against that, four things are **not** established: no row of either round is
Holm-called in round 4b, because the predeclared family size m = 7 is arithmetically unsatisfiable at 5 runs per arm
(0.0079 x 7 = 0.0556) -- a failure to apply a rule already in `docs/INTERP.md` 10.2; the DNa02 story rests on pairs and
on a level model the audit itself disqualifies, so "the structure raises DNa02" remains a `compare`-level pattern of
+0.09 to +0.18 Hz, not a level-controlled measurement; the amplitude-only control M bought 6.0 Hz of chordotonal and
still carries a structure residual 0.66-0.75 Hz **larger** than C's (z +3.5 to +3.8), so "the amplitude law adds no
drive" is not supported and F4 stays open; and the clean-yaw-SD row of C v L flipped from `result` to `null` between
batches only because the control arm's between-run scatter doubled -- the difference itself replicated (+0.57 then
+0.49, complete rank separation in both, pooled p 1.1e-05), so neither batch is an outlier. Nothing is adopted; the
next round's arms are named in section 10 and the first of them is a family whose Holm denominator can actually be
satisfied.
