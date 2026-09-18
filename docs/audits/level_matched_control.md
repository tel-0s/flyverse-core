# The level-matched control: does the leg cycle's per-leg / per-phase STRUCTURE contribute anything beyond the afferent LEVEL? (thread level-matched control, round 4)

Generator: `scripts/probe_vnc_drive.py --family level` (`plan` / `analyse` / `pairs` CPU; `room` / `compass` GPU). Data:
`out/vncd4/` (ONE submission, `out/vncd4/batch.sh` -> run `vncd4-8dd183` on the house cluster, 8 x B200, console
`out/vncd4_cluster.log`: **24 job(s), 0 failed (32.3 min)**), analysis `out/vncd4/analysis_console.txt` (beside the
directory) and `out/vncd4/analysis/pairs_console.txt`. Predeclaration (stamped 2026-09-15T02:00:02Z
(`predeclared.json`); `out/vncd4_cluster.log` carries no absolute timestamp and no run JSON carries a `submitted_at`,
so the ordering rests on local file mtimes: predeclared.json 02:00:02Z against a first-job start no earlier than
~02:00:55Z (the log's 02:33:12Z mtime less its reported 32.3 min). `batch.sh` was written at 02:00:01Z, one second
before the stamp.), working-tree record `out/vncd4/tree_state.json`, the mn_ref derivation
`out/vncd4/mn_ref_derivation.json`, CPU smokes and the two CPU calibration runs `out/vncd4/smoke/` (no result number
below comes from a smoke). Files authored by this task: `scripts/probe_vnc_drive.py` (the `level` family, `--mn-ref-hz`,
the `sided_frames` lag fix, the DNa02 mask fix, `PAIRS_BY_FAMILY`, `ARM_ORDER`), `flyverse/data/expected_responses.csv`
(+4 `op report` rows), this audit. `flyverse/senses.py`, `body.py`, `brain.py`, `optic.py`, `motor.py` are untouched;
every default is unchanged and the shipped path is bit-identical on CPU (section 6). Every number below is a run mean
+- SD over the five runs of an arm (16 flies x 55 s window each; runs are the replicate unit; per-seed values r0..r4 in
brackets where a call rests on them); verdicts are `flyverse.interp.common.compare` on runs (5 v 5, floor p 0.0079),
Holm within the predeclared families (m = 5). Nothing in `out/vncd4` is compared row by row with `out/vncd3` (H200,
two other submissions); round-3 numbers appear only as context and are marked so.

## 0. Answer

**The level was matched** (the predeclared precondition): the realised window-mean commanded chordotonal rate is
**L 86.2 +- 0.5 Hz** [86.47 86.89 85.95 86.20 85.58] against **C 87.4 +- 1.0** [88.04 88.19 86.76 88.16 85.89]
(measured rates of the same 615 cells 86.2 / 87.4; C v L +1.2 Hz, z +2.4, p 0.15, `null`; |L - C| = 1.2 Hz against the
declared 10 Hz tolerance). The derivation's target was 88.1 (section 2); on the B200 the cycle arm itself realised 87.4.

**Decision, in the predeclared words: C v L is `result` on DNa02_L, DNa02_R and the clean yaw SD with the level
matched, so the structure contributes beyond the level -- in the direction "more DNa02, more yaw" -- but the LEVEL
accounts for most of round 3's effect.** What the predeclared rule licenses and what this batch actually separates
are not the same thing: with the chordotonal channel matched, SOMETHING OTHER THAN THE CHORDOTONAL MEAN separates the
leg cycle from the round-2 transducer, but L differs from C in THREE ways at once -- the per-phase modulation, the
unmatched hair plate / campaniform (whose dominant route into DNa02 is sign-negative), and the +13.6 Hz DC chordotonal
L-R -- and no arm in this batch separates them (the hair-plate paragraph below, section 7 item 1(c), and the skeptic
pass at the end of this file). The decision family F1 (C v L, Holm m 5): DNa02_L **+0.127 Hz** (L 0.431 +-
0.014 [0.445 0.409 0.432 0.430 0.438] -> C 0.558 +- 0.005 [0.556 0.562 0.561 0.562 0.550]; z +9.2, p 0.0079, Holm-adjusted
0.040, CALLED), DNa02_R **+0.103** (0.286 +- 0.006 -> 0.389 +- 0.023; z +16.7, CALLED), clean yaw SD **+0.57 deg/s**
(7.30 +- 0.08 [7.39 7.17 7.31 7.30 7.33] -> 7.87 +- 0.16 [8.06 7.95 7.84 7.89 7.63]; z +7.1, CALLED), straightness
**+0.34** (0.498 +- 0.022 -> 0.833 +- 0.016; z +15.3, CALLED: the cycle fly is the STRAIGHTER one --
but a per-fly regression inside L (straightness = 0.987 - 0.1424 x |signed yaw|, r -0.78, n 80) predicts 0.806 at
C's own drift against C's observed 0.833, so 92 % of this row is the round-2 law's DC sidedness, not the per-leg /
per-phase structure; the reverse fit inside C gives 82 %. Read it as a sidedness row.), DNa02 L-R +0.024
(0.145 -> 0.169; z +1.8, p 0.095, `null`). Family F2 (L v A) is `result` on all five (DNa02_L +0.414, z +106; DNa02_R
+0.207, z +23; yaw SD +4.56, z +33; straightness -0.497; DNa02 L-R +0.208): **the level alone fires DNa02 and turns
the fly.** Put against the shipped path, the level control reaches **76 % of the cycle arm's DNa02_L rise** (0.414 of
0.541 = 0.765), **67 % of its DNa02_R rise** (0.207 of 0.310), **89 % of its clean-yaw-SD rise** (4.56 of 5.13 deg/s) and
**73 % of its DNa02-active fraction** (0.0440 of 0.0605); the per-seed scatter of those fractions is <= +-0.07. "The
level" in these fractions is the round-2 law at a raised gain, so it delivers that law's SIDEDNESS as well as its
level. The residual is the remaining quarter of the DNa02 rate and 0.57 deg/s of yaw SD -- a second-order, called
contribution, and what it is made of is this round's open question -- and its behavioural signature is the opposite
of a stronger turn: under the cycle the fly is straighter (0.83 vs 0.50), its median |yaw| halves (1.59 vs 3.54 deg/s),
its signed drift falls (+1.22 vs +3.44 deg/s left) and fewer flies leave the table (8.8 vs 12.0 of 16).

**What the residual could be made of (sections 4.2-4.4).** THREE differences separate L from C at the same chordotonal
mean -- the two structural ones below, and the unmatched hair-plate / campaniform channels of the paragraph after them
-- and the decomposition puts a number on the route that raises DNa02:
1. *Temporal (per-phase) modulation.* Under the cycle a leg's chordotonal command sweeps 20 -> 142 Hz within every
   step (per-leg SD over time 46.4 Hz, 81 % of its variance in the 5-12 Hz step band; the 29 % first printed here is
   corrected in 4.2); under L it is a steady 63-125 Hz (SD 23.9 Hz, 7 % in the band). AN04B003 -- the leg-afferent ascending neuron that carries the excitation to DNa02 --
   fires **23.1 / 23.6 Hz** (L / R) under C and **19.1 / 16.5** under L at the same mean afferent rate, and its
   rate-weighted input to DNa02_L is **+357 mV/s under C against +295 under L** (DNa02_R: +357 vs +250), while the
   haltere-afferent inhibition through PS059 is the same in both (-207 vs -210 on DNa02_L; -169 vs -171 on DNa02_R).
   DNa02_L's net input is +103 mV/s (C) vs +59 (L); DNa02_R's -158 vs -273. A threshold unit fed a modulated Poisson
   input fires more than one fed a steady input of the same mean; that is the mechanism this round PROPOSES for the
   residual. It is not demonstrated here: the level control's hair plate is +9.7 Hz and its route into DNa02 through
   IN13B001 is sign-negative, so a pure level account of the same AN04B003 difference exists (section 0, hair plate).
   Section 7 item 6 and the channel-matched control are what would settle it.
2. *Sidedness.* The round-2 law reads each side's leg-MN rate, so at mn_ref 8.84 the leg-MN bias (+0.87 Hz L-R under L;
   0.14 under C, where the cycle feeds nothing back) becomes a **+13.6 Hz DC chordotonal L-R** (L1 / L2 / L3 92.9 Hz,
   R1 / R2 / R3 79.3; hair plate 61.3 / 52.0) that the cycle arm does not have (-0.34; every leg 87.2-87.7). That fixed
   asymmetry is the level control's own structure: it makes AN04B003_L > AN04B003_R by 2.6 Hz, gives every one of 80
   fly-runs a left drift (+3.44 +- 0.07 deg/s, 16 / 16 flies per run), and is why L's straightness is 0.50. The cycle's
   per-leg amplitude is the same for all six legs (the turn term |amp L-R| 0.016), so the CYCLE REMOVES the round-2
   law's fixed bias -- a structural effect in the opposite direction to the one the question asked about.
   Consequence for the reading: the level control matches the mean and not the sidedness, so "structure" in the
   decision covers both the per-phase modulation (which raises DNa02) and the absence of the DC bias (which straightens
   the walk). Separating those two needs the controls of section 7.

**The hair-plate and campaniform channels did not match, by construction and as predeclared:** L 56.8 / 49.7 Hz vs C
47.1 / 24.9 (the round-2 laws; C v L `result`, -9.7 / -24.8). **They are a live alternative explanation of the
C-over-L excess, not an excluded one.** The hair plate's dominant route into DNa02 is NEGATIVE: the strongest
three-step afferent walk in `paths_summary.csv` is `SNpp45 -> IN13B001 -> AN04B003 -> DNa02`, signs `+,-,+`
(`gain_if_signed` -3.0e+04), and SNpp45 IS a hair-plate afferent (53 of the channel's 113 cells). It moves as that
route predicts: SNpp45 55.8 Hz (L) vs 46.2 (C), IN13B001 **76.4 vs 65.2**, AN04B003 17.8 vs 23.4
(`trace_{L,C}_vs_A.json`, `tables.per_type`). An additive level model with an inhibitory hair plate
(w_chordotonal +0.53, w_hair_plate -0.51 Hz/Hz) reproduces BOTH the within-L left-vs-right side contrast AND the
whole C-over-L AN04B003 difference with a ZERO structure term. The excess is therefore attributable to the per-phase
modulation OR to the unmatched hair plate (and the unmatched campaniform, whose route was not examined); this batch
does not separate them. The haltere afferents follow the haltere MNs (L 18.9, C 16.7 Hz; PS059's inhibition of DNa02
is within 1-2 % between the arms).

**D v C (family F3, the side-split haltere at the matched level):** DNa02_L **-0.036 Hz** (C 0.558 +- 0.005 -> D
0.522 +- 0.017 [0.509 0.504 0.518 0.534 0.544]; z -6.8, p 0.0079, Holm 0.040, CALLED) -- round 3 had this row at -0.028,
p 0.056, `null`; DNa02_R +0.016 `null`, clean yaw SD -0.04 `null`, straightness +0.03 `null`, DNa02 L-R -0.052 (z -2.4)
`null`. The sided haltere trims DNa02_L by 6 % and changes no behavioural row; as in round 3, it is seen by PS196_b and
DNa02 per frame (corr(DNa02 L-R, haltere L-R) 0.008 -> 0.136, corr(PS196_b L-R, haltere L-R) -0.002 -> 0.145, both
`result`) and carries almost no yaw (corr(yaw, haltere MN L-R) 0.08).

**The compass sees no self-turn under any arm** (4 v 4 per arm, floor p 0.029: the flip rows are callable for the first
time, section 5). PS196_b's L-R does not flip (C +0.46 +- 1.7, L -0.64 +- 1.2, D -1.32 +- 1.7, A -0.18 +- 0.3; all `null`),
GLNO's L-R stays +26.9 to +28.1 Hz at every phase of every arm, PEN_a / PEN_b / EPG flips <= 0.57 Hz (`null`), and the
bump drift is inside -0.0087 .. +0.0094 w/s in every phase of every run of every arm against the ideal +-4.0. The one
signed report of the turn is AN04B003 at depth 1: flip **-10.5 +- 1.1 Hz** (C) / **-11.1 +- 1.2** (D), `result` at
4 v 4 -- and **+0.5 +- 0.5 (`null`) under the level control**: with the round-2 law the ascending neuron's L-R is the
leg-MN bias (2.1-2.6 Hz at every phase), not the turn. The per-leg / per-phase structure is what makes AN04B003's
report of a turn signed; it goes no further than PS047_b (-1.5 / -2.9, `null`).

**Round 3's frame-mask caveat and lag defect are closed** (section 3): DNa02_L / _R and DNa02 L-R are on one mask (L - R
= L-R to 1e-6 in every run), and the sidedness rows are at lag 0 (corr(AN04B003 L-R, chordotonal L-R) **-0.346 +- 0.005**
under C; -0.297 at the round-3 alignment, kept as `_lagm1`).

**Under the project rule:** nothing is adopted. L is a labelled control (one hand-set parameter chosen to match a level);
the leg cycle stays an opt-in body-model mechanism. What its adoption would still require is in section 7.

## 1. The question, and why round 3 could not answer it

Round 3 (`docs/audits/body_sided_state.md` 0(1)-(2), 4.2, 8 item 1; `round3_integration.md` 9 item 1) found that the
leg cycle (`all+leg_cycle`, its arm C) makes DNa02 fire (DNa02_L 0.031 -> 0.540 Hz, DNa02_R 0.104 -> 0.383 on H200) and
raises the clean-frame yaw SD 3.35 -> 7.87 deg/s over the round-2 transducer (`all`, its arm B) -- but the commanded
chordotonal afferent rate rises with it, 23.4 -> 88.1 Hz (hair plate 14.1 -> 47.4, leg campaniform 50.0 -> 24.9), because
the cycle's law averages ~0.6 of the FeCO range over a step while the round-2 law scaled the drive by the side's leg-MN
rate / 30 Hz (~0.1). Round 3's own decomposition (its 4.4) accounted for the whole C-vs-B change in DNa02's input by the
LEVEL of the leg afferents (AN04B003/L +58 -> +360 mV/s against PS059/L -114 -> -207), and it wrote: "Attributing
anything to the per-leg / per-phase structure needs the level-matched control arm ... the round-2 law with `mn_ref_hz`
set so its mean is 88 Hz -- a labelled control, not a default." Nobody ran it. This round runs it, in one submission,
with the two reducer defects the round-3 skeptic found fixed first (section 3).

**Design (predeclared, `out/vncd4/predeclared.json`).** Four arms, five brain seeds (0-4; environment seeds 100 s ..
100 s + 15), 16 flies x 60 s each, the `probe_vnc_drive` room protocol (BatchSim, program none, fruit all, no fence,
window 5-60 s) with the clean-frame statistics, blocks `fam_r<seed>` (the four arms of one seed on one node); the same
four arms on the efferent compass at four seeds (blocks `fam_c<seed>`, one job per seed, the arms sequential in it):

| arm | spec | classification |
|---|---|---|
| A | shipped (no sense) | the reference |
| **L** | `'all'` with `mn_ref_hz` **8.84** (default 30) | **LABELLED CONTROL**: the round-2 transducer with ONE hand-set parameter, chosen only so that its window-mean commanded chordotonal rate equals the cycle arm's 88.1 Hz (section 2). Never a default; the sense's default is untouched. |
| C | `'all+leg_cycle'` | the body-model mechanism of round 3 (its arm C): the same afferents at the same level, with the per-leg / per-phase structure |
| D | `'all+leg_cycle+haltere_sided'` | C + the side-split haltere readout (round 3's arm D) |

Primaries (per pair, on ONE mask: DNa02_L, DNa02_R, clean yaw SD, straightness, DNa02 L-R); the within-batch pairs are
**L v A, C v L, D v C**; the decision pair is **C v L** (family F1, Holm m = 5); F2 = L v A (m 5), F3 = D v C (m 5).
Verdicts are `flyverse.interp.common.compare` on runs (5 v 5, floor p 0.0079; result needs |z| >= 3, p <= 0.05, >= 4
runs per arm in one submission; a zero-SD reference gives a structural p, read as `undetermined`). The decision rule,
verbatim from the predeclaration: *"If C v L is null on DNa02 (DNa02_L_hz and DNa02_R_hz) and on the clean yaw SD, the
leg cycle's contribution is its afferent LEVEL: the per-leg / per-phase structure adds nothing the level control does
not, and round 3's DNa02 effect is attributed to the afferent rate. If C v L is result on DNa02 or on the yaw SD with
the level matched (precondition met), the structure contributes beyond the level, in the direction of the sign of the
difference. If the precondition fails (level mismatch > 10 Hz), no attribution is made; the audit reports the mismatch
and what it implies."* The precondition is the realised chordotonal level of L against C (|L - C| <= 10 Hz); the
hair-plate and campaniform channels follow their own laws in the two arms and are NOT matched by construction
(predicted L ~58 / 50 Hz vs C ~47 / 25) -- reported, not matched. C v A here is a replication ACROSS devices (B200 vs
round 3's H200) of round 3's C-v-A and is labelled so; D v L is a listed secondary pair.

## 2. The mn_ref_hz derivation (CPU; `out/vncd4/mn_ref_derivation.json`)

**The law** (`senses.Proprioception.rates`, unchanged): chordotonal rate per cell = 10 + 140 x clip(legMN_side /
mn_ref_hz, 0, 1) on the ground (10 Hz airborne); the 271 L cells read `leg_L`, the 259 R cells `leg_R`, the 85 unsided
cells the mean. The window mean over cells and frames is what round 3 reported as the commanded chordotonal rate.

**Step 1, open loop.** Solving mean[law(mn_ref)] = 88.1 over the RECORDED per-frame leg-MN rates of `out/vncd3`
(`cmd__legMN_L / _R`, frames >= 5 s, 5 runs x 16 flies) reproduces the recorded levels first (B: f(30) = 23.42 Hz vs
23.4 recorded) and gives mn_ref = **4.99 Hz** with B's MN distribution, **8.45 Hz** with C's, 4.16 with A's. They differ
because the leg-MN rate rises with the afferent level (A 2.46 -> B 2.87 -> C 4.74 Hz side mean): mn_ref sets the level
through a closed loop (afferents -> VNC -> leg MNs -> afferents), so the right value depends on the MN rate the level
arm itself produces. The critique's ~3.5 Hz ignored the clip and the loop and would give 143-150 Hz (the clip
saturates at 150).

**Step 2, CPU calibration** (`out/vncd4/smoke/cal_*`; `CUDA_VISIBLE_DEVICES=-1`, 4 flies x 12 s, window 2-12 s, brain
seed 0, one room step each; 560 s wall each). The round-2 transducer at the default 30 Hz reproduces the GPU level on CPU
(23.32 Hz vs 23.4 on H200; leg MN 3.00 / 2.70 vs 3.03 / 2.72), so the CPU proxy is faithful for the level. The level arm
at the C-distribution value 8.45 Hz realises **95.5 Hz** -- the MNs rose MORE than under the cycle (5.64 / 4.69, side
mean 5.16 vs C's 4.74): the loop slope on CPU is k = d(legMN) / d(level) = 0.032 Hz/Hz, loop gain 140 k / mn_ref = 0.53
(stable).

**Step 3, the fixed point.** level(mn_ref) = mean[law(mn_ref; the cal_L per-frame MN distribution scaled by
m(level) / m_L)], m(level) = m_B + k (level - 23.3); level = 88.1 at mn_ref = **8.839 Hz** (9.26 without the loop
correction). **Chosen: 8.84 Hz.** Sensitivity: d(level) / d(mn_ref) = -17.7 Hz per Hz with the loop, so +-0.3 Hz of
mn_ref is -+5 Hz of level; the predeclared match tolerance (10 Hz) is that uncertainty. **Realised on the B200: 86.2 Hz
(predicted 88.1; the cycle arm itself 87.4)**, with leg MNs 5.28 / 4.41 Hz (the CPU calibration's 5.64 / 4.69 at 8.45
scaled as the loop predicts). Under the law the same mn_ref puts the hair plate at 56.8 Hz and the leg campaniform at
49.7 (the round-2 ground law), against C's 47.1 / 24.9 -- the control matches the chordotonal channel (615 cells, the
channel round 3's attribution rests on: AN04B003 reads the femoral chordotonal afferents) and reports the other two.

The value is passed on the job line (`--mn-ref-hz 8.84`, `ARM_MN_REF['level']['L']`); `senses.Proprioception` is built
with its own `mn_ref_hz` keyword (`fb.proprioception_sense` replaced before the first step) and nothing in `senses.py`
changed. The derivation code is the inline python quoted in `mn_ref_derivation.json` (numpy over `*_body.npz`).

## 3. Fixed before the analysis (the round-3 skeptic's two reducer defects) and the four ledger rows

* **`sided_frames` lag.** Round 3 paired recorder sample s (captured after the step of frame k = 2 s) with body frame
  k - 1 (`fr = round(t_ms / 10) - 2` against the `[1:]`-sliced body arrays). It now pairs it with frame k (lag 0;
  `fr = round(t_ms / 10) - 1`) and keeps the round-3 alignment as labelled secondaries `<key>_lagm1`. Check on
  `out/vncd3/room_C_r0`: corr(AN04B003 L-R, chordotonal L-R) **-0.339** at lag 0 (skeptic: -0.339) vs -0.294 at the
  old alignment (the printed -0.295); the tripod-conditioned swing **-3.49 Hz** vs -3.37. B: 0.128 / 0.127 (unchanged).
  In this batch (5 runs, C): -0.346 +- 0.005 at lag 0 vs -0.297 +- 0.006 at lag -1; the swing -3.60 +- 0.03 vs -3.39.
* **The DNa02 frame mask.** `summarise_room` computed `DNa02_L_hz` / `DNa02_R_hz` from the command readout on the
  post-skip non-airborne mask (the `DNa02_LR_hz` mask) and then let the watch loop OVERWRITE them with the recorder's
  all-window-frames mean (airborne frames included: `round3_integration.md` 4's caveat). The watch loop now leaves them
  and stores its value under `DNa02_{L,R}_allwin_hz`; `robust_room` also recomputes `DNa02_L_hz` / `DNa02_R_hz` /
  `DNa02_LR_hz` from the body arrays on the one mask, so a round-3 JSON tabulates on it too. Check on
  `out/vncd3/room_C_r0`: JSON 0.5578 / 0.3735 (L - R 0.1843) vs recomputed 0.5592 / 0.3752, which subtract to the
  reported L-R 0.18397 exactly. In this batch the all-window value sits 0.001 (L) / 0.003 (C) Hz below the masked one.
* **Ledger rows** (`flyverse/data/expected_responses.csv`, all `op report`, none scored): `lit.walk.step_frequency_hz_at_speed`
  (7.1 Hz at 8 mm/s; 16.5 at 28: the DeAngelis et al. 2019 Fig 1E stance-duration fit with the 30 ms swing plateau --
  the fit's values, not a tabulated number; Mendes et al. 2013 Fig 2 for the direction), `lit.walk.stance_fraction_at_speed`
  (0.79 at 8; 0.51 at 28, the same sources, the same caveat), `lit.walk.swing_duration_ms` (30, bracket 30-50, **UNCERTAIN**;
  Mendes 2013 Fig 2B, DeAngelis 2019 Fig 1E), `lit.walk.outer_leg_step_ratio_in_turn` (**empty value**: DeAngelis 2019
  Fig 6C gives the direction only -- outer legs longer, inner mid / hind shorter -- and no ratio was read off it; a
  placeholder to be filled from a measurement, never invented). `flyverse.interp.ledger.load_table()` validates the
  table with the four rows (142 rows). The realised cycle in this batch: step 7.79 +- 0.11 Hz, stance fraction 0.766
  at the room's ~9 mm/s (C), D 7.76 / 0.767 -- the laws' values, as in round 3.

## 4. The room (20 runs: 4 arms x 5 brain seeds, 16 flies x 60 s, 5-60 s window; `analysis/room_table.csv`, `pairwise.csv`, `sided_frames.csv`)

Every room run reports `device cuda`, `device_name NVIDIA B200`, family `level`, its arm's spec, `mn_ref_hz 8.84` (L)
/ 30.0 (the untouched default, A / C / D), its seed's block `fam_r<seed>` and a 51-file `source_fingerprint`
(`analysis_console.txt` lines 2-6; the verification table is reproduced in section 8). Artefacts: 20 x (`.json`,
`_body.npz`, `_rec.npz`, `_flies.npz`, `_max.npz`).

### 4.1 The primaries and the behaviour (per-seed values r0..r4 in brackets)

| key | A | L (level control) | C (+ cycle) | D (+ sided haltere) | L v A | **C v L** | D v C |
|---|---|---|---|---|---|---|---|
| DNa02_L (Hz, L-R mask) | 0.017 +- 0.004 | 0.431 +- 0.014 [0.445 0.409 0.432 0.430 0.438] | **0.558 +- 0.005** [0.556 0.562 0.561 0.562 0.550] | 0.522 +- 0.017 [0.509 0.504 0.518 0.534 0.544] | +0.414 z+106.4 result | **+0.127 z+9.2 p0.008 result (Holm 0.040)** | -0.036 z-6.8 p0.008 result (Holm 0.040) |
| DNa02_R (Hz) | 0.080 +- 0.009 | 0.286 +- 0.006 [0.285 0.279 0.289 0.295 0.282] | **0.389 +- 0.023** [0.424 0.387 0.394 0.381 0.360] | 0.405 +- 0.007 | +0.207 z+23.4 result | **+0.103 z+16.7 result** | +0.016 z+0.7 null |
| yaw SD, clean frames (deg/s) | 2.74 +- 0.14 | 7.30 +- 0.08 [7.39 7.17 7.31 7.30 7.33] | **7.87 +- 0.16** [8.06 7.95 7.84 7.89 7.63] | 7.83 +- 0.15 | +4.56 z+33.1 result | **+0.57 z+7.1 result** | -0.04 z-0.3 null |
| straightness | 0.995 +- 0.000 | **0.498 +- 0.022** [0.476 0.514 0.523 0.474 0.501] | 0.833 +- 0.016 [0.817 0.818 0.854 0.835 0.843] | 0.865 +- 0.032 | -0.497 z-1009 result | **+0.336 z+15.3 result** | +0.032 z+2.0 null |
| DNa02 L-R (Hz) | -0.063 +- 0.006 | +0.145 +- 0.013 [0.160 0.129 0.143 0.135 0.156] | +0.169 +- 0.022 [0.132 0.175 0.167 0.181 0.189] | +0.117 +- 0.013 | +0.208 z+33.5 result | +0.024 z+1.8 p0.095 null | -0.052 z-2.4 p0.008 null |
| frames with DNa02 > 5 Hz | 0.007 | 0.051 +- 0.001 | 0.067 +- 0.002 | 0.066 | +0.044 result | +0.016 z+13.0 result | null |
| \|DNa02 L-R\| per frame (Hz) | 0.096 | 0.676 | 0.882 | 0.863 | result | +0.21 z+12.2 result | null |
| median \|yaw\|, clean (deg/s) | 0.73 | **3.54 +- 0.08** | 1.59 +- 0.09 | 1.54 | +2.81 result | -1.95 z-25.7 result | null |
| p99 \|yaw\|, clean (deg/s) | 15.4 | 26.2 +- 0.4 | 25.4 +- 0.4 | 25.5 +- 1.1 | +10.8 result | -0.85 z-2.2 null | null |
| mean SIGNED yaw, clean (deg/s; + left) | +0.19 +- 0.03 | **+3.44 +- 0.07** | +1.22 +- 0.09 | +1.00 +- 0.06 | +3.25 z+103 result | -2.22 z-30.8 result | -0.23 z-2.7 null |
| flies (of 16) with mean yaw > 0 | 13.0 | 16.0 +- 0.0 | 15.4 | 15.6 | | | |
| net heading change (turns / fly) | 0.034 | 0.267 +- 0.016 | 0.223 +- 0.021 | 0.196 +- 0.028 | +0.23 result | -0.04 z-2.9 p0.032 null | null |
| flies (of 16) that left the table top | 0.4 +- 0.5 | **12.0 +- 0.7** [13 12 12 11 12] | 8.8 +- 1.6 [6 10 9 9 10] | 8.4 +- 2.5 | +11.6 z+21.2 result | -3.2 z-4.5 result | null |
| hops per fly | 0.05 | 0.51 +- 0.09 | 0.45 +- 0.03 | 0.51 +- 0.14 | +0.46 z+8.8 result | -0.06 null | null |
| clean fraction of window frames | 0.999 | 0.899 +- 0.034 | 0.926 +- 0.010 | 0.933 +- 0.030 | -0.10 result | +0.03 null | null |
| leg MN L / R (Hz) | 2.54 / 2.38 | **5.28 / 4.41** | 4.79 / 4.65 | 4.78 / 4.64 | result | -0.48 / +0.24 result | null |
| leg MN L-R (Hz) | +0.154 +- 0.004 | **+0.867 +- 0.006** | +0.144 +- 0.005 | +0.142 +- 0.007 | +0.71 z+175 result | -0.72 z-119 result | null |
| haltere MN L / R (Hz); L-R | 7.65 / 9.41; -1.76 | 18.11 / 19.69; -1.58 | 15.40 / 18.00; -2.60 | 15.22 / 18.00; -2.78 | result | -2.7 / -1.7 result; -1.03 result | null; -0.17 z-2.3 null |
| wing power MN mean (Hz) | 20.4 | 15.2 | 15.4 | 15.4 | -5.2 result | +0.17 null | null |
| GF per-frame max (Hz) | 25.6 +- 1.0 | 33.6 +- 2.7 | 31.4 +- 1.7 | 31.0 +- 3.9 | +8.0 z+8.2 result | -2.2 z-0.8 null | null |
| step frequency (Hz) / stance fraction | | | 7.79 +- 0.11 / 0.766 | 7.76 / 0.767 | | | null / null |

Holm within F1 (C v L, m 5): the four `result` rows have p 0.0079 each, adjusted 0.040 (CALLED); DNa02 L-R p 0.095
(not called). F2 (L v A): all five CALLED at adjusted 0.040. F3 (D v C): DNa02_L CALLED (adjusted 0.040); the other
four not called. The listed secondary pair D v L reads as C v L on every primary (DNa02_L +0.091 z +6.5, DNa02_R +0.119,
yaw SD +0.53, straightness +0.37, all `result`; DNa02 L-R -0.028 `null`). The labelled cross-device replication C v A
gives the same five calls round 3 made on H200 (DNa02_L +0.541 z +139, DNa02_R +0.310, yaw SD +5.13, straightness
-0.162, DNa02 L-R +0.232, all `result`), and the C-arm numbers sit where round 3's did (DNa02_L 0.558 vs 0.540, yaw SD
7.87 vs 7.87, straightness 0.833 vs 0.826, 8.8 vs 7.4 flies off the table) -- a replication across devices, not a
row-by-row comparison.

**The level control turns MORE persistently, not more variably.** L's yaw SD is 93 % of C's, but its median |yaw| is
2.2x C's and its signed drift 2.8x: the round-2 law's sidedness (section 4.2) turns the leg-MN bias into a constant
left turn (+3.44 deg/s in 16 / 16 flies of every run), so 12 of 16 flies walk off the table within 60 s and the
window's clean fraction is the lowest of the four arms (0.90). The cycle's yaw is larger in SD and smaller in DC.

### 4.2 The afferents per channel, per side and per leg (commanded Hz = the Poisson rate the transducer set; measured = the cells' realised rate)

| channel (n cells; L / R / unsided) | L commanded (measured) | C | D | bracket | C v L |
|---|---|---|---|---|---|
| chordotonal (615; 271 / 259 / 85) | **86.2 +- 0.5** (86.2) | **87.4 +- 1.0** (87.4) | 87.1 +- 1.0 | 10-150 | +1.2 z+2.4 p0.15 **null (matched)** |
| chordotonal L / R | **92.9 / 79.3** | 87.3 / 87.6 | 87.1 / 87.3 | | |
| chordotonal per leg L1 R1 L2 R2 L3 R3 | 92.9 79.3 92.9 79.3 92.9 79.3 | 87.3 87.5 87.2 87.7 87.3 87.5 | | | |
| chordotonal L-R mean / per-frame \|L-R\| | **+13.6 +- 0.1** / 16.1 | -0.34 +- 0.03 / 12.4 | -0.25 / 12.3 | | -14.0 z-126 result / -3.8 result |
| chordotonal per-leg SD over time (Hz); variance in the 5-12 Hz band | 23.9; 0.07 | **46.4; 0.81** | 46.5; not recomputed | | |
| chordotonal per-leg p10 / p90 over time (Hz) | 57.0 / 118.7 | 19.6 / 142.5 | 19.6 / 142.5 | | |
| hair plate (113; 57 / 54 / 2) | 56.8 +- 0.3 | 47.1 +- 0.6 | 46.9 | 5-100 | -9.7 z-28.7 result |
| hair plate L / R; L-R | 61.3 / 52.0; +9.2 | 47.0 / 47.2; -0.19 | | | |
| leg campaniform (12; 6 / 6 / 0) | 49.7 | 24.9 | 24.9 | 0-100 | -24.8 z-543 result |
| haltere (201; 99 / 95 / 7) | 18.9 +- 0.1 | 16.7 +- 0.2 | 16.6 | 0-250 | -2.2 z-28 result |
| haltere L / R; L-R | 18.9 / 18.9; 0 | 16.7 / 16.7; 0 | 15.2 / 18.0; **-2.78** | | |

(The per-frame temporal statistics are computed from `room_<arm>_r<s>_body.npz` `commandedg__chordotonal:<leg>` on
window non-airborne frames, per fly then run mean; run SD over seeds 0.2 / 0.6 Hz on the per-leg SD. The band
fraction is a periodogram of the mean-removed per-leg series on the same frames, per leg per fly then run mean:
**L 0.076 +- 0.002, C 0.814 +- 0.030** (Welch nperseg 256 gives 0.086 / 0.942; including the DC bin gives 0.098).
The 0.29 printed for C at first analysis is not reproducible under any estimator next to an L value of 0.07 and is
corrected to 0.81; D's band fraction was not recomputed. The per-leg SD, the p90 and the qualitative contrast all
reproduce; L's p10 / p90 is corrected from 62.6 / 125.4 to 57.0 / 118.7.) Every channel stays inside its ledger
bracket in every arm. The level control matches the mean of the chordotonal channel and nothing else: its rates are
steady within a leg (the leg-MN rate's own fluctuation), sided (the leg-MN bias x 140 / 8.84), and its hair plate and
campaniform sit 10 and 25 Hz above the cycle's. The cycle's rates
are equal across legs and sides, unsteady within a leg (20 -> 142 Hz within each 128 ms step: the stance sweep
|2p - 1| and the swing burst), and alternate between the tripods (per-frame |L-R| 12.4 Hz with a -0.34 Hz DC from the
turn kinematics).

### 4.3 Sidedness per frame (`sided_frames.csv`; clean walking frames, per fly, then the run mean; SD over 5 runs; recorder rows at lag 0)

| statistic | L | C | D | C v L |
|---|---|---|---|---|
| corr(realised yaw, chordotonal cmd L-R) | **+0.272 +- 0.013** | **-0.150 +- 0.003** | -0.149 | -0.42 z-32 result |
| corr(realised yaw, leg MN L-R) | +0.257 +- 0.013 | +0.029 +- 0.012 | +0.036 | -0.23 result |
| corr(realised yaw, haltere MN L-R) | +0.192 | +0.053 | +0.078 | result |
| corr(AN04B003 L-R, chordotonal cmd L-R) [lag -1] | +0.204 +- 0.007 [+0.215] | **-0.346 +- 0.005** [-0.297] | -0.344 [-0.294] | -0.55 z-80 result |
| E[AN04B003 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) [lag -1] | +2.10 +- 0.11 [+2.25] | **-3.60 +- 0.03** [-3.39] | -3.59 [-3.37] | -5.70 z-54 result |
| corr(DNa02 L-R, chordotonal cmd L-R) | +0.013 +- 0.006 | **-0.146 +- 0.004** | -0.141 | -0.16 z-25 result |
| E[DNa02 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) | -0.02 +- 0.04 | **-0.35 +- 0.03** | -0.35 +- 0.02 | -0.33 z-9.4 result |
| corr(AN04B003 L-R, realised yaw) | +0.081 +- 0.014 | -0.062 +- 0.009 | -0.056 | -0.14 result |
| corr(DNa02 L-R, haltere cmd L-R) | +0.006 | +0.008 | **+0.136 +- 0.009** | null (D v C +0.13 z+53 result) |
| corr(PS196_b L-R, haltere cmd L-R) | -0.014 | -0.002 | **+0.145 +- 0.010** | null (D v C +0.15 z+73 result) |
| corr(PS196_b L-R, realised yaw) | +0.059 | +0.068 | +0.089 | null |

Reading it: under the level control every sided correlation has the round-2 sign -- the afferent L-R IS the leg-MN
bias, the yaw follows it (+0.27), AN04B003's L-R follows it (+0.20), and DNa02's L-R does not follow the afferent L-R
per frame at all (+0.01). Under the cycle the kinematic sign appears (a left turn shortens the left steps, -0.15), and
AN04B003 and DNa02 carry the tripod alternation in antiphase to the side sum (-0.35 / -0.15; the wiring reason is
round 3's 4.3: AN04B003_L reads L1 + L2, which swing with opposite tripods). The tripod-locked swing of DNa02's L-R
(-0.35 Hz) is the same as round 3's, and it is absent under the level control (-0.02): a per-frame sided signal on
DNa02 exists ONLY with the per-leg / per-phase structure -- and, as in round 3, it steers nothing (corr(AN04B003 L-R,
yaw) -0.06, corr(PS196_b L-R, yaw) 0.07). `corr(PS059 L-R, haltere cmd L-R)` is empty in this batch: the level family
ran with the round-2 watch list (PS059 was not recorded per frame; `watch_of` now includes it for this family, so a
rerun records it); PS059's window means are in the decomposition below.

### 4.4 DNa02's input per arm (`decompose_DNa02_L / _R_per_type.csv`, rate-weighted input in mV/s per post cell over the window; `dna02_decompose_summary.csv`; run SD 1-5 mV/s on the named rows)

| DNa02_L input (mV/s) | A | L (level) | C (cycle) | D |
|---|---|---|---|---|
| E total / I total / **net** | +741 / -1048 / -306 | +1337 / -1278 / **+59** | +1363 / -1260 / **+103** | +1354 / -1257 / +97 |
| AN04B003 rate carried in the CSV's `rate_hz` column (Hz) | 0.60 | 19.09 | 23.12 | 23.06 |
| (DNa02_L's own window rate is in 4.1: 0.017 / 0.431 / 0.558 / 0.522) | | | | |
| AN04B003/L (leg afferents, k1) | +9.2 | **+294.9 +- 1.4** | **+357.1 +- 4.0** | +356.3 +- 4.4 |
| SNpp45/? (unsided afferent, direct) | 0 | +168.2 | +129.0 | +128.6 |
| GNG580/L, PS013/L, AN10B021/R, AN10B018/R | 0 / +2 / 0 / +2 | +55 / +44 / +32 / +29 | +49 / +52 / +39 / +32 | +45 / +50 / +39 / +32 |
| PS059/L (haltere afferents, k1) | -90.0 | **-209.6 +- 1.5** | **-206.9 +- 3.9** | -204.1 +- 2.3 |
| IN12B014/R | -90.5 | -64.5 | -65.4 | -65.1 |
| GNG562/L / LT51/L / IN19A003/L | -68.4 / -57.5 / -43.3 | -51.5 / -49.1 / -18.4 | -45.3 / -45.4 / -20.1 | -45.7 / -46.0 / -20.6 |
| LAL046/L, PS077/L, LAL120_a/R, LAL126/R | -11.5 / -0.2 / -32.5 / -1.0 | -45.8 / -37.4 / -41.9 / -31.0 | -47.6 / -44.8 / -42.5 / -33.8 | -45.3 / -43.0 / -42.7 / -35.7 |

| DNa02_R input (mV/s) | A | L (level) | C (cycle) | D |
|---|---|---|---|---|
| E total / I total / **net** | +809 / -1172 / -363 | +1230 / -1503 / **-273** | +1342 / -1500 / **-158** | +1344 / -1500 / -156 |
| AN04B003 rate carried in the CSV's `rate_hz` column (Hz) | 0.31 | 16.54 | 23.61 | 23.50 |
| (DNa02_R's own window rate is in 4.1: 0.080 / 0.286 / 0.389 / 0.405) | | | | |
| AN04B003/R | +4.7 | **+250.2 +- 1.3** | **+357.0 +- 4.2** | +355.3 +- 4.9 |
| SNpp45/?, AN10B021/L, PS013/R, GNG580/R | 0 / +1 / +3 / 0 | +66 / +58 / +39 / +15 | +59 / +54 / +41 / +13 | +59 / +54 / +42 / +13 |
| PS059/R | -49.7 | **-171.0 +- 1.1** | **-168.9 +- 2.0** | -170.6 +- 1.8 |
| IN12B014/L | -125.7 | -95.0 | -98.4 | -98.5 |
| LT51/R / IN19A003/R / GNG562/R | -68.3 / -61.2 / -55.2 | -71.7 / -49.6 / -39.4 | -67.3 / -55.3 / -35.0 | -67.6 / -56.4 / -35.4 |
| LAL126/L, LAL120_a/L, IN09A004/R, LAL046/R | -2.7 / -47.1 / -1.0 / -12.4 | -59.7 / -56.5 / -53.8 / -38.6 | -67.9 / -57.2 / -55.9 / -39.3 | -64.5 / -57.0 / -55.7 / -40.7 |

(The `rate_hz` column of the summary CSV is NOT the post cell's window rate: it is AN04B003's, carried into the table
by the tool's first-row convention -- 0.60 / 19.09 / 23.12 / 23.06 beside DNa02_L and 0.31 / 16.54 / 23.61 / 23.50
beside DNa02_R, in every arm including A. DNa02's own window rates are in 4.1.)

Answers. **Does the level fire DNa02 as the cycle does?** Mostly: at the same chordotonal mean the leg-afferent term on
DNa02_L is +295 (L) against +357 mV/s (C), the haltere-afferent cancellation is the same (-210 vs -207; PS059's rate
follows the haltere MNs, 18.9 vs 16.7 Hz, and its weight on DNa02 is capped), the tonic VNC inhibitors fall the same
way (IN12B014/R -90 -> -65 / -65, IN19A003/L -43 -> -18 / -20), and the LAL loop inhibitors appear the same way
(LAL046, PS077, LAL120_a, LAL126 at -31 to -48 in both). DNa02_L's net input is +59 (L) vs +103 (C). On DNa02_R the
whole difference is AN04B003 (+107 of the +114 mV/s net gap, 94 %). On DNa02_L the +44 mV/s net gap is a
cancellation: AN04B003 +62, SNpp45 -39 (the unmatched hair plate), the other excitatory rows +3 and the inhibitory
total +18 (41 % of the gap). AN04B003's own term is a difference in its RATE at the same afferent mean (19.1 / 16.5 Hz
vs 23.1 / 23.6) -- the input's temporal structure (4.2) OR the unmatched hair plate through the sign-negative
`SNpp45 -> IN13B001 -| AN04B003` route (section 0) -- and, on the R side, the level control's DC sidedness
(AN04B003_R reads R2 + R3 at 79.3 Hz under L against 87.6 under C: an 8.3 Hz per-side deficit that the whole-channel
match hides, so the DNa02_R row mixes whatever structure term there is with a per-side level deficit). The DIRECT
SNpp45 term is LARGER under L (+168 vs +129 on DNa02_L: SNpp45 is a hair-plate cell and the hair plate runs at 57 vs
47 Hz), which works against the C-over-L excess on that route; the hair plate's DOMINANT route into DNa02 is not the
direct one but the disynaptic `SNpp45 -> IN13B001 -| AN04B003`, and that one runs the other way (section 0).
**Why left in both:** the excitatory rows are equalised by
`conn_cap` (AN04B003/L +357 = /R +357 under C; under L the L side gets +295 and the R side +250 -- the DC bias again)
and the inhibitory rows differ by side as round 3 described (IN12B014/L -98 on DNa02_R vs /R -65 on DNa02_L, a
presynaptic-rate asymmetry: IN12B014_L 11.1 Hz vs _R 8.5 under C, 11.2 vs 8.9 under L), so DNa02_L's net is positive
and DNa02_R's negative in both arms, and the level control's fixed afferent L-R adds to it.

The traces (`trace_{L,C,D}_vs_A.json`) carry the same population at every depth (389 / 1295 / 1897 / 1256 / 542 / 227
carriers at depths 1-6 under L; 382 / 1297 / 2025 / 1390 / 499 / 199 under C): the afferents at 86-87 Hz recruit the
same VNC / AVLP / LAL / SMP field whether steady or phase-modulated. The afferent -> DNa02 walks (`paths_summary.csv`)
are round 2's three (SNpp45 direct; SApp -> PS059; SNpp45 -> IN13B001 -> AN04B003), all live under L / C / D.

## 5. The efferent compass (16 runs: 4 arms x 4 seeds; gE 2 / gD 15; DNa02_L / _R at 20 Hz for 10 s; 3 s skipped; `compass_table.csv`, `compass_flip_<arm>.csv`, `compass_flip_chain.csv`, `compass_chain_<arm>.csv`)

Every compass run reports `device cuda`, `NVIDIA B200`, family `level`, its spec and block `fam_c<seed>`; 16 x 8 npz.
Four runs per arm, so the flip statistic is 4 v 4 (floor p 0.029, callable) and the ccw / cw-vs-rests drift comparison
4 v 8. The realised self-turn is +97.6 to +105.2 deg/s (ccw) and -89.4 to -95.3 (cw) across the arms (DNa02 at 19.4-20.1
Hz on the pulsed side); the bump sits at its ~1.5-wedge site, vector strength 0.762-0.766, at every phase of every arm.

| arm | bump drift rest / ccw / rest2 / cw (w/s; per-run range) | ccw v rests / cw v rests | PS196_b L-R rest / ccw / rest2 / cw; flip (ccw - cw) vs null (rest2 - rest) | AN04B003 flip | GLNO L-R |
|---|---|---|---|---|---|
| A | -0.002 / -0.001 / +0.002 / +0.003 (-0.0071 .. +0.0068) | -0.001 null / +0.003 null | -0.07 / -0.18 / -0.04 / 0.00; -0.18 +- 0.32 vs +0.04 +- 0.18, null | -0.27 +- 0.32 (z -3.3, p 0.057) null | +27.8 .. +28.1 |
| L | -0.001 / 0.000 / +0.001 / +0.003 (-0.0087 .. +0.0093) | -0.000 null / +0.003 null | +3.39 / +2.71 / +3.29 / +3.36; **-0.64 +- 1.17** vs -0.11 +- 1.12, null | **+0.50 +- 0.51** vs -0.10 +- 0.97, **null** (L-R +2.1 .. +2.6 at every phase) | +26.9 .. +27.4 |
| C | +0.001 / 0.000 / +0.001 / +0.004 (-0.0049 .. +0.0094) | -0.001 null / +0.003 null | +3.39 / +3.79 / +3.57 / +3.32; **+0.46 +- 1.74** vs +0.18 +- 0.69, null | **-10.55 +- 1.13** vs +0.27 +- 0.89, **result** (rest -1.0, ccw -5.9, rest2 -0.8, cw +4.6) | +26.9 .. +27.2 |
| D | -0.001 / -0.000 / +0.001 / +0.005 (-0.006 .. +0.007) | +0.000 null / +0.005 null | +3.54 / +2.46 / +3.71 / +3.79; **-1.32 +- 1.67** vs +0.18 +- 1.24, null | **-11.11 +- 1.25** vs +0.63 +- 1.13, **result** | +26.9 .. +27.2 |

The chain beyond the ascending neurons: PS047_b flip -1.5 (C) / -2.9 (D) / -2.1 (L) `null`; PEN_a -0.06 / -0.05 / +0.10,
PEN_b -0.04 / 0.00 / +0.06, EPG +0.49 / +0.57 / +0.57, all `null`; PS059 flips +6.0 (C, result) / +4.5 (D, z +4.0
but p 0.057, `null`) / +3.7 (L, result) / +2.8 (A, result) -- it follows the DNa02 stimulus in every arm, including
the shipped one, as round 3 said.
AN06A026 and AN07B035 flip with the CYCLE's sign under C / D (-1.6 / -1.4, result) and with the OPPOSITE sign under the
level control (+2.9 / +2.8, result): under the round-2 law the ascending neurons' L-R follows the pulsed side's leg-MN
rise, under the cycle it follows the kinematics (the inner legs' shorter steps). PS196_b fires at 6.6-7.6 / 3.2-4.2 Hz
(L / R) in every phase of L, C and D (0.0-0.2 under A) and its L-R is +2.5 to +3.8 in every phase: driven, unsigned,
round 2's and round 3's pattern. The level in the pinned protocol: chordotonal 85.5 (L, rest) / 92.1 (ccw) against
92.8 / 90.7 (C) -- the pinned fly's leg MNs sit differently from the walking fly's, so the compass arms are matched to
within 7 Hz at rest, not 1 Hz; nothing in the compass turns on the level, since no arm moves the bump.

So the compass answer of rounds 2 and 3 stands at 4 v 4: the self-turn is reported, signed, at the ascending neurons
under the cycle (AN04B003 -10.5 / -11.1 Hz), unsigned there under the level control, and it does not reach PS196_b,
GLNO or the ring under any arm. The per-leg / per-phase structure is what makes the depth-1 report signed; the
level does not.

## 6. Validation (CPU, `CUDA_VISIBLE_DEVICES=-1`; this desktop)

* `PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_proprioception.py tests/test_body_cycle.py
  tests/test_bit_identity.py -q`: **26 passed, 5 subtests passed** (`out/vncd4/smoke/pytest_cpu.txt`). The bit-identity
  golden passes unchanged: the shipped path is bit-identical on CPU, and no default moved (`senses.py`, `body.py`,
  `brain.py` untouched; `git diff --stat` in `tree_state.json` shows this task's files only, beside other threads'
  docs).
* CPU smokes, one room step per arm (A / C / D: 2 flies x 0.5 s; L: the 4 flies x 12 s calibration run) and one quick
  compass step per arm (A / L / C / D, `--quick --sparse torch`): every one `exit 0` (`out/vncd4/smoke/*.txt`); no
  number is quoted from them except the calibration levels of section 2, which are labelled as CPU.
* Reducer checks on round-3 artefacts (section 3): the lag-0 alignment reproduces the skeptic's numbers to three
  decimals; the same-mask DNa02_L / _R subtract to the reported L-R exactly.
* The level match is verified from the recordings (section 0, 4.2): commanded and measured chordotonal means agree to
  0.03 Hz within an arm; L and C agree to 1.2 Hz.
* Batch verification: `24 job(s), 0 failed (32.3 min)`; 20 room JSON + 80 room npz; 16 compass `_run.json` + 128 npz;
  every run on `cuda` / `NVIDIA B200`, the right family / spec / mn_ref / block, 51-file fingerprint.
* A defect found and fixed after the analysis: `watch_of` returned the round-2 watch list for the new family, so
  PS059 was not recorded per frame in this batch (its window means are in the decomposition; one sided_frames row is
  empty). The fix is a one-line change that affects only future room recordings, not any reducer; the reducers'
  sha256 at predeclaration and after this edit are both recorded (`predeclared.json`, `out/vncd4/post_analysis_sha.json`).

## 7. What an adoption of the leg cycle would still require (nothing is adopted)

1. **THREE more controls to split "structure" itself**, because the level control matched ONE channel's mean and
   neither the sidedness nor the other two channels: (a) an UNSIDED level control -- the round-2 law reading the
   side-MEAN leg-MN rate on every cell at the same mn_ref (removes the +13.6 Hz DC L-R; tests whether the level
   alone, without the round-2 law's fixed bias,
   still makes DNa02_L > DNa02_R and the fly drift left); (b) a MODULATION-ONLY arm -- the cycle's per-phase law with
   the per-leg amplitude held at 1 (no turn kinematics in the afferents), which separates the temporal modulation
   (the part that raises AN04B003 from 19 to 23 Hz) from the amplitude turn term (|amp L-R| 0.016); (c) a
   **CHANNEL-MATCHED level control**: the round-2 law at mn_ref 8.84 with `hair_plate_max_hz` and
   `campaniform_load_hz` set so that the realised hair-plate and campaniform commanded means match C's 47.1 / 24.9 Hz.
   This is the control the C-over-L attribution actually needs: without it the DNa02 excess is attributable either to
   the per-phase modulation or to the unmatched hair plate acting through `SNpp45 -> IN13B001 -| AN04B003`. Neither (a)
   nor (b) matches those two channels. All three are labelled controls in `probe_vnc_drive`'s family table; none
   exists yet.
2. **The suite with the cycle on** (`scripts/benchmark.py` and the room ledger rows under `all+leg_cycle+haltere_sided`,
   5 runs, one block): the cycle fly leaves the table (8.8 of 16), hops (0.45 / fly) and raises GF (31 vs 26 Hz) -- rows
   that assume a straight walk move; the object / loom / feeding rows have not been run with the cycle at all.
3. **Ledger values** for the two placeholders this round added: `lit.walk.outer_leg_step_ratio_in_turn` (empty) and
   `half_width_m` (1.0 mm, unmeasured; it scales the turn asymmetry linearly). A stance-width / per-leg step-length
   extraction from DeAngelis 2019 or Chun et al. 2021 tarsus tracks would fill both.
4. **`MotorRates.haltere_L / haltere_R`** as fields (a golden regeneration; owner decision) -- the sided readout is
   still a function beside `MotorRates`.
5. **A free-walking compass room under the cycle** (`cx_shift.py` room mode): the pinned protocol shows the report is
   gone by PS196_b; the expectation is no drift, and it has not been measured.
6. **A record of what AN04B003 actually responds to**: a single-cell CPU check (`interp_atlas`-style) of AN04B003
   under (i) steady vs 8 Hz-modulated chordotonal input at the same mean with IN13B001's rate clamped, and (ii) the
   hair-plate level varied alone at a fixed chordotonal level -- the two together settle whether the C-over-L
   AN04B003 difference is modulation or hair-plate disinhibition.
7. The FeCO walking-mean rate is still not a ledger number (only the 10-150 Hz bracket exists), so the level 87 Hz is
   a consequence of the law, not a measurement -- the same caveat as round 3.

## 8. Files, provenance and reproduction

* Code (this task): `scripts/probe_vnc_drive.py` -- `LEVEL_MN_REF_HZ`, `ARMS_LEVEL` / `ARM_LABEL_LEVEL` / `ARM_MN_REF`,
  `FAMILIES['level']` (compass arms `ALCD`), `ARM_ORDER`, `PAIRS_BY_FAMILY`, `mn_ref_of`, `--mn-ref-hz` on `room` and
  `compass` (default None = the sense's own default), `plan` writing the flag on the L job lines, `summarise_room`'s
  watch loop (no overwrite of the command-derived DNa02 keys; `_allwin_hz`), `robust_room`'s same-mask DNa02 recompute,
  `sided_frames` at lag 0 with `_lagm1` secondaries (`REC_LAG_KEYS`), `load_room`'s `room_[A-Z]_r` glob, `pairs
  --family`, `watch_of` (level family records PS059 from now on). `flyverse/data/expected_responses.csv`: four `op
  report` rows appended. Nothing else in `flyverse/` changed.
* Data: `out/vncd4/` (`batch.sh`, `predeclared.json`, `tree_state.json`, `mn_ref_derivation.json`, 20 room x 5 files,
  16 compass x 9 files, `smoke/`), console `out/vncd4_cluster.log` (`vncd4-8dd183`: 24 job(s), 0 failed, 32.3 min).
  Every JSON carries `provenance` (resolved `LIFParams`, `OpticParams`, the compiled-W md5, the 51-file
  `source_fingerprint`, `execution.device / device_name`, seeds, the block key, the sense's `mn_ref_hz` and spec) --
  except the compass runs: all 16 `compass_*_run.json` record `mn_ref_hz` null and the 64 phase JSONs carry no arm /
  family / spec, so the compass L arm's `--mn-ref-hz 8.84` is verifiable only from the job line in
  `out/vncd4_cluster.log` (and from its realised pinned level, 85.5-92.1 Hz against the default's ~23).
  The run directory is a snapshot of this working tree: `tree_state.json` records `git diff --stat` at submission
  (HEAD 28e862f; this task's two files plus other threads' `README.md`, `docs/INSTALL.md`, `docs/OVERVIEW.md`,
  `docs/REPRODUCIBILITY.md`, `docs/audits/column_ground_truth.md`, `scripts/probe_column_ground_truth.py`, none
  imported by any job here).
* Analysis: `out/vncd4/analysis_console.txt` (beside the directory, not inside it) and `out/vncd4/analysis/`
  (`room_table.csv/txt`, `summary.json`, `trace_{L,C,D}_vs_A.*`,
  `decompose_{DNa02_L,DNa02_R,PS196_b,GLNO}.*`, `paths_*`, `compass_*`, and from `pairs`: `pairs_console.txt`,
  `pairwise.csv`, `sided_frames.csv`, `dna02_decompose_summary.csv`, `compass_flip_chain.csv`, `pairs_summary.json`).
* Reproduce: `PYTHONIOENCODING=utf-8 python scripts/probe_vnc_drive.py plan --family level --dir out/vncd4 --runs 5
  --compass-seeds 4 --draws 0 --minutes 90` (writes the 24-job `batch.sh`); `bash out/vncd4/batch.sh` (one
  `cluster_run.py` call, `--arm-block fam`, house target); `... analyse --dir out/vncd4 --out out/vncd4/analysis
  --trace-arms L,C,D` (CPU, 620 s); `... pairs --dir out/vncd4 --out out/vncd4/analysis` (CPU, ~3 min). A GPU rerun is
  a replication, not a bit-for-bit check (the GPU rollout is not seed-reproducible).

## Report

```yaml
summary: >
  The level-matched control was run (ONE house submission vncd4-8dd183, 24 jobs, 0 failed, B200; 4 arms x 5 seeds in
  the room, 4 arms x 4 seeds on the efferent compass): the round-2 transducer with mn_ref_hz 8.84 (derived on CPU as
  the fixed point of the afferent -> leg-MN loop; a labelled control) realised a window-mean commanded chordotonal
  rate of 86.2 +- 0.5 Hz against the leg-cycle arm's 87.4 +- 1.0, so the predeclared precondition is met (C v L
  +1.2 Hz, z +2.4, p 0.15, `null`, inside the 10 Hz tolerance). With the CHORDOTONAL channel matched to 1.2 Hz,
  C v L is `result` on DNa02_L (+0.127 Hz, z +9.2), DNa02_R (+0.103, z +16.7) and the clean yaw SD (+0.57 deg/s,
  z +7.1), Holm-called at m 5 -- so SOMETHING OTHER THAN THE CHORDOTONAL MEAN separates the leg cycle from the
  round-2 transducer, in the direction "more DNa02". DNa02_L is the robust row: arm L's left side carries MORE
  chordotonal, hair-plate and campaniform drive than arm C's and still fires DNa02_L less, so no under-drive or
  sidedness account of it survives. But "the per-leg / per-phase STRUCTURE contributes beyond the LEVEL" is NOT yet
  supported: L differs from C in three ways at once -- per-phase modulation, a +9.7 / -24.8 Hz mismatch on the two
  unmatched leg channels whose dominant route into DNa02 is sign-negative (`SNpp45 -> IN13B001 -| AN04B003`), and a
  +13.6 Hz DC chordotonal L-R -- and no arm in this batch separates them. The straightness row is `result` but reads
  as the DC sidedness (82-92 % of the gap is the drift); the DNa02_R row is `result` but is partly a per-side level
  deficit (L's right side sees 79.3 Hz chordotonal against C's 87.6); the DNa02 L-R row is `null`. L v A is `result`
  on all five primaries, and the afferent level as delivered by the round-2 law -- sidedness included -- reaches
  76 % / 67 % / 89 % of the cycle's DNa02_L / DNa02_R / yaw-SD rise over the shipped path with tight per-seed scatter:
  that is the round's solid finding. D v C stands as reported (DNa02_L -0.036 called, no behavioural row). The compass
  answer stands at 4 v 4: no arm moves the bump, and AN04B003's signed report of the turn (-10.5 Hz) exists under the
  cycle and not under the level control. Nothing is adopted, no default changed, and adoption now needs THREE
  controls, not two: the unsided level control, the modulation-only arm, and a channel-matched level control.
skeptic:
  source: "independent Opus pass, 2026-09-15"
  verdict: "mostly sound"
key_claims:
  - "Level matched (the predeclared precondition): L 86.2 +- 0.5 Hz [86.47 86.89 85.95 86.20 85.58] vs C 87.4 +- 1.0 [88.04 88.19 86.76 88.16 85.89] commanded chordotonal (measured identical); C v L +1.2 Hz z +2.4 p 0.15 null; inside the predeclared 10 Hz tolerance."
  - "C v L (decision family, Holm m 5): DNa02_L 0.431 +- 0.014 -> 0.558 +- 0.005 (z +9.2, p 0.0079, adj 0.040, called); DNa02_R 0.286 +- 0.006 -> 0.389 +- 0.023 (z +16.7, called); clean yaw SD 7.30 +- 0.08 -> 7.87 +- 0.16 deg/s (z +7.1, called); straightness 0.498 +- 0.022 -> 0.833 +- 0.016 (z +15.3, called); DNa02 L-R +0.145 -> +0.169 (z +1.8, p 0.095, null). Per-seed values in section 4.1; an independent pass reproduced all fifteen family comparisons from the npz with its own exact Mann-Whitney, its own z and its own Holm, and not one verdict flips."
  - "What the calls DO establish: with the chordotonal channel matched to 1.2 Hz, something other than the chordotonal mean separates the leg cycle from the round-2 transducer, in the direction 'more DNa02'. What they do NOT yet establish is that the per-leg / per-phase STRUCTURE is what it is: L differs from C in three ways at once -- per-phase modulation, a +9.7 / -24.8 Hz hair-plate / campaniform mismatch, and a +13.6 Hz DC chordotonal L-R -- and no arm in this batch separates them."
  - "The hair plate is a live alternative, not an excluded one: its dominant route into DNa02 is NEGATIVE -- `SNpp45 -> IN13B001 -| AN04B003`, signs +,-,+, gain_if_signed -3.0e+04, the strongest three-step afferent walk in paths_summary.csv, and SNpp45 is 53 of the channel's 113 cells. It moves as that route predicts: SNpp45 55.8 (L) vs 46.2 (C), IN13B001 76.4 vs 65.2, AN04B003 17.8 vs 23.4. An additive level model with an inhibitory hair plate (w_chordotonal +0.53, w_hair_plate -0.51 Hz/Hz) reproduces both the within-L side contrast and the whole C-over-L AN04B003 difference with a ZERO structure term."
  - "DNa02_L is the robust row: arm L's LEFT side carries more chordotonal (92.90 vs 87.24 Hz), hair-plate (61.25 vs 46.97) and campaniform (49.67 vs 24.88) drive than arm C's and still fires AN04B003_L lower (19.10 vs 23.13) and DNa02_L lower (0.431 vs 0.558) -- the DC bias works in L's favour there. Within L, DNa02_L per fly rises with the fly's own drift (slope +0.0836, r +0.741), so +0.127 is a lower bound, not an artefact."
  - "Straightness reads as sidedness, not structure: a per-fly regression inside L (straightness = 0.987 - 0.1424 x |signed yaw|, r -0.78, n 80) predicts 0.806 at C's own drift against C's observed 0.833 -- 92 % of the +0.336 gap; the reverse fit inside C gives 82 %. Restricting to on-table post-skip trajectory does not rescue it (L 0.645 +- 0.026 vs C 0.931 +- 0.009). DNa02_R is partly a per-side level deficit: L's right side sees 79.28 Hz chordotonal against C's 87.58 (8.3 Hz), and AN04B003 is 94 % of the DNa02_R gap."
  - "L v A: result on all five primaries (DNa02_L +0.414 z +106; DNa02_R +0.207 z +23; yaw SD +4.56 z +33; straightness -0.497; DNa02 L-R +0.208). The afferent level, as delivered by the round-2 law and including that law's sidedness, reaches 76 % / 67 % / 89 % of the cycle's DNa02_L / DNa02_R / yaw-SD rise and 73 % of its DNa02-active fraction (0.0440 of 0.0605), per-seed scatter <= +-0.07. It is not a general statement: on straightness the same fraction is 3.08 -- the level control overshoots by 3.1x."
  - "Decomposition: at the same chordotonal mean AN04B003 fires 23.1 / 23.6 Hz (C) vs 19.1 / 16.5 (L); its input to DNa02_L +357 vs +295 mV/s, DNa02_R +357 vs +250; PS059 -207 vs -210 / -169 vs -171; DNa02_L net +103 vs +59, DNa02_R -158 vs -273. On DNa02_R AN04B003 is the whole gap (+107 of +114 mV/s, 94 %); on DNa02_L the +44 mV/s gap is a cancellation (AN04B003 +62, SNpp45 -39, other E +3, I total +18 = 41 %). The cycle's per-leg command sweeps 20 -> 142 Hz within a step (SD 46.4 Hz, 0.81 of variance in the 5-12 Hz band on a periodogram of the mean-removed series); the level control's is steady (SD 23.9 Hz, 0.07)."
  - "The level control is sided by the round-2 law: chordotonal L1/L2/L3 92.9 Hz vs R1/R2/R3 79.3 (DC L-R +13.6 Hz; leg MN L-R +0.87 Hz), hair plate 61.3 / 52.0; the cycle's legs are 87.2-87.7 (L-R -0.34). L drifts left +3.44 +- 0.07 deg/s in 16/16 flies, 12.0 of 16 leave the table (C 8.8), median |yaw| 3.54 vs 1.59 deg/s. The DC L-R cannot be regressed out: it is disjoint between the arms (L range [11.55, 15.21], C [-0.79, +0.25]) and its within-arm relation to yaw reverses sign (L +0.272, the cause of the turn; C -0.150, its kinematic consequence)."
  - "D v C: DNa02_L -0.036 Hz (0.558 -> 0.522 +- 0.017; z -6.8, adj 0.040, called; round 3 had -0.028 p 0.056 null); DNa02_R, yaw SD, straightness, DNa02 L-R null. The sided haltere is seen per frame by DNa02 / PS196_b (corr 0.008 -> 0.136 / -0.002 -> 0.145, result) and carries no yaw."
  - "Compass at 4 v 4: PS196_b flip null in every arm (C +0.46 +- 1.74, L -0.64 +- 1.17, D -1.32 +- 1.67); GLNO L-R +26.9..+28.1 fixed; PEN/EPG flips <= 0.57 Hz null; bump drift within -0.0087..+0.0094 w/s in every phase of every run of every arm. AN04B003 flip -10.5 +- 1.1 (C) / -11.1 +- 1.2 (D) result, +0.5 +- 0.5 (L) null. The compass run JSONs do not record mn_ref_hz (see section 8)."
  - "Reducer fixes verified on round-3 artefacts: lag 0 gives corr(AN04B003 L-R, chord L-R) -0.339 / swing -3.49 Hz on room_C_r0 (the round-3 skeptic's numbers); same-mask DNa02_L - DNa02_R = DNa02 L-R to < 1e-6 in all 20 runs. This batch: -0.346 +- 0.005 (lag 0) vs -0.297 (lag -1)."
  - "Cross-device replication (labelled, not row-by-row): C v A on B200 makes the same five calls round 3 made on H200; C's DNa02_L 0.558, yaw SD 7.87, straightness 0.833."
  - "Nothing is adopted and no default changed. Adoption now needs THREE controls, not two: the unsided level control, the modulation-only arm, and a channel-matched level control."
files_written:
  - scripts/probe_vnc_drive.py (level family, --mn-ref-hz, lag-0 sided_frames with _lagm1 secondaries, same-mask DNa02_L/_R, ARM_ORDER, PAIRS_BY_FAMILY, watch_of)
  - flyverse/data/expected_responses.csv (+4 rows: lit.walk.step_frequency_hz_at_speed, lit.walk.stance_fraction_at_speed, lit.walk.swing_duration_ms, lit.walk.outer_leg_step_ratio_in_turn; op report)
  - docs/audits/level_matched_control.md
  - out/vncd4/ (batch.sh, predeclared.json, tree_state.json, mn_ref_derivation.json, post_analysis_sha.json, 20 room + 16 compass runs, analysis/, smoke/)
  - out/vncd4_cluster.log
api:
  - "probe_vnc_drive.py room|compass --family level --arm A|L|C|D [--mn-ref-hz X]: X defaults to ARM_MN_REF['level'][arm] (L: 8.84) else the sense's default 30; senses.Proprioception(c, spec, mn_ref_hz=X) is built with its own keyword, senses.py unchanged"
  - "probe_vnc_drive.py plan --family level (24 jobs; name vncd4); analyse / pairs unchanged in CLI (pairs --family optional; the pair set comes from PAIRS_BY_FAMILY)"
  - "run JSON keys: DNa02_L_hz / DNa02_R_hz on the DNa02_LR_hz mask; DNa02_{L,R}_allwin_hz the old all-window value; sided_frames keys at lag 0 plus <key>_lagm1"
validation:
  - "pytest tests/test_proprioception.py tests/test_body_cycle.py tests/test_bit_identity.py: 26 passed, 5 subtests (CPU); bit-identity golden unchanged"
  - "CPU smokes room A/C/D (2 flies x 0.5 s) and L (4 flies x 12 s), compass A/L/C/D --quick: all exit 0"
  - "batch: 24 job(s), 0 failed (32.3 min); 20 room x 5 files, 16 compass x 9 files; every run cuda / NVIDIA B200 / family level / correct spec, mn_ref, block / 51-file fingerprint"
  - "predeclared.json stamped 2026-09-15T02:00:02Z; out/vncd4_cluster.log carries no absolute timestamp and no run JSON carries a submitted_at, so the ordering rests on local file mtimes: predeclared.json 02:00:02Z against a first-job start no earlier than ~02:00:55Z (the log's 02:33:12Z mtime less its reported 32.3 min); batch.sh was written at 02:00:01Z, one second before the stamp. Reducer sha256 recorded in predeclared.json and, after the post-analysis watch_of edit, in post_analysis_sha.json (only probe_vnc_drive.py differs; the four reducers are byte-identical)"
  - "level match verified from the recordings (commanded and measured chordotonal means, per arm and per seed)"
recommendations:
  - "Run the THREE controls that split 'structure': an UNSIDED level control (round-2 law on the side-mean leg-MN rate at mn_ref 8.84), a MODULATION-ONLY cycle arm (per-leg amplitude held at 1) and a CHANNEL-MATCHED level control (hair_plate_max_hz / campaniform_load_hz set so the realised hair-plate and campaniform means match C's 47.1 / 24.9 Hz at the same chordotonal 86-88 Hz); 5 seeds each, one block with A / L / C. Only the third addresses the hair-plate alternative; neither of the other two matches those channels."
  - "A single-cell CPU check of AN04B003 under (i) steady vs 8 Hz-modulated chordotonal input at the same mean with IN13B001's rate clamped, and (ii) the hair-plate level varied alone at a fixed chordotonal level -- the two together settle whether the C-over-L AN04B003 difference is modulation or hair-plate disinhibition."
  - "The full suite under all+leg_cycle+haltere_sided (5 runs, one block) before any default discussion; the object / loom / feeding rows have never run with the cycle."
  - "Fill lit.walk.outer_leg_step_ratio_in_turn and half_width_m from a tarsus-track measurement; owner decision on MotorRates.haltere_L/_R (golden regeneration)."
  - "scripts/box_status.py assumes a 'port' key and fails on the house target (KeyError); default it to 22 (not this task's file; the poll was done with the same remote query over ssh port 22)."
open_questions:
  - "How much of the C-over-L DNa02 excess is the temporal modulation, how much the unmatched hair plate acting through SNpp45 -> IN13B001 -| AN04B003, and how much the absence of the round-2 law's DC sidedness? This batch separates none of the three (the three controls above)."
  - "Does the level control's fixed left drift (+3.4 deg/s, 16/16 flies) come only from the amplified leg-MN bias, or also from the hair plate's +9.2 Hz L-R? (the unsided control answers both)"
  - "The pinned compass fly's afferent level differs from the walking fly's by 7 Hz at rest under L (85.5 vs 92.8): irrelevant to the bump (nothing moves it) but the compass arms are not level-matched to 1 Hz."
  - "The FeCO walking-mean rate is still not a ledger number; 87 Hz is the law's consequence, not a measurement."
```

## Skeptic pass (independent, 2026-09-15)

Everything below was recomputed on this desktop (CPU, `CUDA_VISIBLE_DEVICES=-1`) from `out/vncd4/*_body.npz`,
`*_rec.npz`, the run JSONs and the analysis CSV/JSONs, with my own clean-frame rule, my own exact Mann-Whitney U
(full enumeration of the C(10,5)=252 / C(8,4)=70 rank splits), my own z = diff / SD(null, ddof 1) and my own Holm.
No cluster job was submitted; nothing under `docs/`, `scripts/`, `flyverse/`, `tests/` was touched.

---

### Refuted

**R1. The hair-plate dismissal is wrong, and it is the load-bearing one.**
Audit section 0 (lines 63-66): *"L has MORE hair-plate and campaniform drive than C and still less DNa02, so neither
can account for the C-over-L excess in the direction found."* That inference needs the hair-plate -> DNa02 transfer to
be net positive. The audit's own paths output says it is net **negative**:

`out/vncd4/analysis/paths_afferents_to_DNa02_nfA.json` -> `summary.top_silent_walk_per_k["3"]`:

```
path   "a:SNpp45 -> IN13B001 -> AN04B003 -> b"
signs  "+,-,+"          link_mv "+119.533,-16.500,+15.285"
gain_if_signed  -30145.98        (the strongest 3-step afferent->DNa02 walk; also row 1 of paths_summary.csv, k3_top_signed)
```

`SNpp45` **is** the hair-plate afferent (`flyverse/senses.py` line 106: `hair_plate subclass 'hair plate'
(SNpp45, SNpp19 ...)`), 53 of the channel's 113 cells. The direct SNpp45->DNa02 link is only +2.003 mV/volley
(`strongest_silent_link_per_k["1"]`); the dominant route is the disynaptic **inhibitory** one through IN13B001 onto
AN04B003 -- the very cell the audit's decomposition says carries the whole C-over-L excess.

And the intermediate moves exactly as that route predicts. From `trace_L_vs_A.json` / `trace_C_vs_A.json`
(`tables.per_type`, `stim_level`, 5 runs each):

| cell | A | **L** | **C** | L - C |
|---|---|---|---|---|
| SNpp45 (hair plate) | 0.00 | **55.83** | **46.17** | **+9.65** |
| IN13B001 | 8.53 | **76.36** | **65.24** | **+11.12** |
| AN04B003 | 0.45 | **17.82** | **23.36** | -5.55 |

(AN04B003 side-split from the room watch: L 19.096 / 16.542, C 23.132 / 23.623 -- the audit's 19.1 / 16.5 vs
23.1 / 23.6, confirmed; the pooled 17.82 / 23.36 is their mean.)

A purely additive **level** model with an inhibitory hair plate fits both constraints with **zero** structure term.
Solve simultaneously (i) the within-L left-vs-right side contrast (Deltachord +13.62, Deltahair +9.24 -> DeltaAN04B003 +2.56) and
(ii) the whole C-over-L pooled gap (Deltachord +1.19, Deltahair -9.69 -> DeltaAN04B003 +5.56):
**w_chord = +0.533, w_hair = -0.508 Hz/Hz.** Both are of the size the IN13B001 route supplies
(dIN13B001/dHair = 11.12/9.65 = 1.15, so it needs only -0.44 Hz/Hz from IN13B001 to AN04B003).

So the sentence at lines 49-51 -- *"A threshold unit fed a modulated Poisson input fires more than one fed a steady
input of the same mean; that is the structure's contribution, and it is the whole of the C-over-L excess on the named
rows"* -- is **not supported by this batch**. The temporal modulation and the unmatched +9.7 Hz hair plate (and the
unmatched -24.8 Hz campaniform, whose route into DNa02 was never examined at all) are not separated by any arm that
was run. The audit's own recommendation 6 (a single-cell modulation check) concedes the mechanism is untested; section
0 nevertheless asserts it as established.

**R2. The straightness row of family F1 is a sidedness row, not a structure row.**
Per fly within arm L (n = 80): `straightness = 0.9872 - 0.14244 x |signed yaw|`, r = -0.776. Extrapolated to arm C's
own mean signed drift (1.270 deg/s) it predicts **0.806** against C's observed **0.833** -- i.e. **92 % of the +0.3357
C-over-L straightness gap** is the drift difference alone. The reverse fit inside C (`1.0656 - 0.18300 x`, r -0.667)
extrapolated to L's drift predicts 0.437 against L's observed 0.498 -- **82 %**. The drift itself is the round-2 law's
+13.6 Hz DC chordotonal L-R, which is the control's construction, not the cycle's structure. Listing straightness
among the CALLED rows that show "the structure contributes beyond the level" (section 0, lines 25-31, and the Report's
`summary` / `key_claims`) overstates it; the audit says the right thing at 4.2 item 2 and then contradicts itself at
the top. (Restricting straightness to on-table, post-skip trajectory does **not** rescue it: L 0.645 +- 0.026 vs C
0.931 +- 0.009, still `result`, so it is not a leaving-the-table artefact either -- it is a drift artefact.)

**R3. "76 % of its DNa02-active fraction (0.044 of 0.060)" (line 36) is 73 %.**
`room_table.csv` key `DNa02_active_frac`: A 0.00679, L 0.05080, C 0.06732 -> (L-A)/(C-A) = 0.04401/0.06052 = **0.727**.
Even the audit's own rounded 0.044/0.060 is 0.733.

**R4. "29 % of its variance in the 5-12 Hz step band" cannot be reproduced.**
Recomputed from `room_<arm>_r<s>_body.npz` `commandedg__chordotonal:<leg>` on window non-airborne frames, per leg per
fly then run mean, 5 seeds: **L 0.076 +- 0.002** (audit 0.07 ok) but **C 0.814 +- 0.030** (audit 0.29). Welch nperseg
256 / 512 gives C 0.94 / 0.93; including the DC bin gives 0.098. No estimator reproduces 0.29 next to an L value of
0.07. The audit names no file and no estimator for this pair, and it appears in the Report's `key_claims`. (The
per-leg SD, p90 and the qualitative contrast all reproduce: L 23.86 +- 0.20 Hz, C 46.53 +- 0.56 Hz vs the audit's
23.9 / 46.4; C p10/p90 19.5 / 142.7 vs 19.6 / 142.5. L's p10/p90 I get 57.0 / 118.7 against the audit's 62.6 / 125.4.)

**R5. Two of the four compass bump-drift ranges in section 5 are wrong.**
From `compass_table.csv` `drift_runs` (4 runs x 4 phases per arm):

| arm | recomputed per-run range | audit |
|---|---|---|
| A | -0.0071 .. +0.0068 | "(all within -0.006 .. +0.006)" -- **wrong** |
| L | -0.0087 .. +0.0093 | "(-0.0087 .. +0.0093)" ok |
| C | -0.0049 .. **+0.0094** | "(-0.006 .. +0.007)" -- **wrong** |
| D | -0.0062 .. +0.0072 | "(-0.006 .. +0.007)" ok (rounded) |

The global bound in section 0 ("inside -0.0087 .. +0.0093 w/s in every phase of every run of every arm") should be
**-0.0087 .. +0.0094** (arm C, rest phase, run 3). Conclusion unaffected; the number is wrong.

**R6. Section 4.1's `|DNa02 L-R| per frame` levels do not match the file.**
Audit row: `0.28 | 0.86 | 1.07 | 1.05`. `room_table.csv` / `pairwise.csv` key `DNa02_abs_LR_hz`:
**A 0.0963, L 0.6759, C 0.8822, D 0.8628**. The quoted difference (+0.21) and z (+12.2) *do* match the file
(0.8822 - 0.6759 = 0.2063; 0.2063/0.0169 = 12.2), so only the four level cells are wrong -- offset by ~+0.19 each.

**R7. Section 4.4's "DNa02_L rate in the decompose window" row is incoherent.**
Audit row: `| 0.6 | 19.1 x 1e-2 (0.44) | 0.56 | 0.52 |`. `dna02_decompose_summary.csv` `rate_hz` is
0.5964 / 19.0919 / 23.1204 / 23.0642 (post DNa02_L) and 0.3080 / 16.5417 / 23.6095 / 23.4963 (post DNa02_R) -- those
are **AN04B003's** rates in every arm including A, as the audit's own parenthetical says. DNa02_L's actual window
rates are 0.017 / 0.431 / 0.558 / 0.522. The row as written takes A and L from AN04B003 and C and D from DNa02, and
"19.1 x 1e-2 (0.44)" is not a number (19.1e-2 = 0.191, not 0.44).

**R8. "the whole difference is AN04B003" is true only on DNa02_R.**
From `dna02_decompose_summary.csv`, C - L:
DNa02_L: net **+44.28** mV/s; AN04B003 **+62.23**; SNpp45 **-39.2**; I_total **+17.98** (41 % of the net gap);
other E rows +3.3. DNa02_R: net **+114.15**; AN04B003 **+106.88** (94 %); SNpp45 -6.2; I_total +2.34.
On DNa02_L the accounting is a +62 against a -39 plus +21 elsewhere, not a single term.

**R9. "Every JSON carries provenance (... the sense's `mn_ref_hz` and spec)" (section 8) is false for the compass.**
All 16 `compass_*_r?_run.json` record `mn_ref_hz: null`, the four L runs included; the 64 phase JSONs
(`compass_*_rest.json` etc.) carry no `arm`, `family`, `proprioception` or `mn_ref_hz` at all. The flag *was* applied
(L's pinned chordotonal is 85.5-92.1 Hz against the ~23 Hz the default 30 would give), but the compass provenance does
not record it, so the compass arm-L configuration is verifiable only from the job line in `out/vncd4_cluster.log`.
(The 20 room JSONs *do* record it correctly -- see Confirmed C2.)

**R10. "stamped ... before the submission" rests on local mtimes only.**
`out/vncd4_cluster.log` contains **no absolute timestamp of any kind** -- only the 24 queued-job lines and relative
`[ 11s]` marks. No room or compass JSON carries a `submitted_at` / `created_utc` / scheduler receipt. The only
evidence is: `predeclared.json` mtime 2026-09-14 19:00:02.355 -0700 = **02:00:02Z** (matching its own
`stamped_utc`), and the log's mtime 02:33:12Z minus its own reported 32.3 min => a first-job start no earlier than
~02:00:55Z. That is consistent with the claim but it is *self-certified*, and `written_before_submission: true` is a
field in the file it certifies. Note also that **`out/vncd4/batch.sh` was written at 02:00:01Z, one second *before*
the predeclaration** -- the plan (arms, 24 jobs, blocks) preceded the stamp.

**R11 (minor). Path error.** Section 8 and line 5 place `analysis_console.txt` in `out/vncd4/analysis/`. It is at
`out/vncd4/analysis_console.txt`; only `pairs_console.txt` is inside `analysis/`.

**R12 (minor). The critique's "~3.5 Hz would give 110-145 Hz" (section 2) is 143-150 Hz.**
Recomputed on the vncd3 C leg-MN distribution: mn_ref 3.0 / 3.5 / 4.0 -> open-loop 146.3 / 145.5 / 143.6 Hz,
self-consistent 150.0 / 149.8 / 149.2. The clip at 150 makes the lower end unreachable. The conclusion stands.

**R13 (minor). PS196_b compass rates.** Audit: "6.7-7.6 / 3.3-4.2 Hz (L / R) in every phase of L, C and D".
`compass_chain_{L,C,D}.csv`: L side 6.57-7.57, R side 3.21-4.18. Also section 5's prose lists
"PS059 flips +6.0 (C, result) / +4.5 (D) / +3.7 (L, result) / +2.8 (A, result)" -- **D is not a result**
(flip +4.464 +- 2.236, z +3.98 but p 0.0571 > 0.05, verdict blank in `compass_flip_chain.csv`); the parallel phrasing
implies it is.

---

### Confirmed

**C1. Every one of the five primaries, in all four pairs, reproduces to the printed precision with my own statistics.**
Per-arm run means +- SD over the 5 seeds, from my own clean-frame rule (post-skip; on the table top at both ends of the
step; no airborne frame within 50 frames; |yaw| <= 720 deg/s) and my own post-skip-non-airborne DNa02 mask:

| key | A | L | C | D |
|---|---|---|---|---|
| DNa02_L | 0.0169 +- 0.0039 | 0.4309 +- 0.0139 | 0.5581 +- 0.0053 | 0.5219 +- 0.0168 |
| DNa02_R | 0.0796 +- 0.0088 | 0.2862 +- 0.0062 | 0.3892 +- 0.0229 | 0.4051 +- 0.0066 |
| yaw SD clean | 2.744 +- 0.138 | 7.300 +- 0.081 | 7.872 +- 0.160 | 7.828 +- 0.145 |
| straightness | 0.9947 +- 0.0005 | 0.4975 +- 0.0219 | 0.8332 +- 0.0160 | 0.8648 +- 0.0319 |
| DNa02 L-R | -0.0628 +- 0.0062 | 0.1447 +- 0.0133 | 0.1690 +- 0.0220 | 0.1168 +- 0.0130 |

Family calls (own exact MWU, own z, own Holm at the declared m = 5):

* **F1 C v L**: DNa02_L +0.1272 z +9.2 p 0.0079 Holm 0.0397 **CALLED**; DNa02_R +0.1030 z +16.7 **CALLED**;
  yaw SD +0.5719 z +7.1 **CALLED**; straightness +0.3357 z +15.3 **CALLED**; DNa02 L-R +0.0243 z +1.8 p 0.0952 `null`.
* **F2 L v A**: all five `result`, all Holm 0.0397 **CALLED** (+0.4140 z +106.4; +0.2065 z +23.4; +4.557 z +33.1;
  -0.4972 z -1009; +0.2075 z +33.5).
* **F3 D v C**: DNa02_L -0.0363 z -6.8 Holm 0.0397 **CALLED**; the other four not called (DNa02 L-R -0.0522 has
  p 0.0079 but |z| 2.4 < 3 -> `null`, as the audit says).
* **C v A** (labelled cross-device): all five `result` (+0.5413 z +139.1; +0.3095 z +35.1; +5.129 z +37.3;
  -0.1615 z -327.8; +0.2317 z +37.4).

Every "called" survives. Not one verdict flips. The JSON `run` values and my from-npz recomputation agree to 1e-4.

**C2. Run configuration (item 2).** All 20 room runs: `device cuda`, `device_name NVIDIA B200`, `family level`,
spec `None`/`all`/`all+leg_cycle`/`all+leg_cycle+haltere_sided` for A/L/C/D, `mn_ref_hz` **8.84 for L and 30.0 for
A/C/D**, blocks `fam_r0..fam_r4`, 51-file `source_fingerprint`. All 16 compass `_run.json`: `cuda`, `NVIDIA B200`,
`family level`, correct spec, blocks `fam_c0..fam_c3` (but `mn_ref_hz` null -- R9). **ONE submission**
(`vncd4-8dd183`, 24 job lines, "24 job(s), 0 failed (32.3 min)"), one `--arm-block fam` call, `--mn-ref-hz 8.84`
present on exactly the five L room job lines and the four L compass sub-commands.

**C3. Reducer sha256 (item 1).** Predeclared vs now: `flyverse/interp/common.py`, `senses.py`, `body.py`,
`expected_responses.csv` are **byte-identical** to their predeclared hashes; only `scripts/probe_vnc_drive.py` differs
(155b6a3f... -> bb70aaf9...), exactly as `post_analysis_sha.json` records. The run JSONs' `source_fingerprint` uses
LF-normalised hashes (senses.py `5cf6d4bf...` = my LF hash of the same file; body.py `dd091f41...` likewise), so the
shipped tree and the predeclared tree are the same files.
**No predeclared primary is affected by the `watch_of` edit.** The five primaries come from the command readout
(`rates_cmd["DNa02_L"/"DNa02_R"]` in `summarise_room`) and from `robust_room`'s recompute over
`cmd__DNa02_{L,R}` in the body npz -- neither touches the watch list. I verified the cells that `watch_of` controls:
`room_C_r0_rec.npz` types contain AN04B003, DNa02, PS196_b but **not PS059**, while `out/vncd3/room_C_r0_rec.npz`
does contain PS059 -- the defect is real, is confined to one empty `sided_frames` row, and changes nothing else.
The analysis ran before the edit (`paths_afferents_to_DNa02_nfA.json` `run_id` 2026-09-15T02:43:24Z;
`post_analysis_sha.json` stamped 03:15:43Z).

**C4. The level match (item 4) -- verified from the recordings, not the summary.** Window means over the post-skip
frames of the `commanded__*` / `commandedg__*` arrays, per fly then run mean, 5 runs:

| channel | L commanded | L measured | C commanded | C measured | C v L |
|---|---|---|---|---|---|
| chordotonal | **86.219 +- 0.498** | 86.224 | **87.407 +- 1.039** | 87.434 | +1.188, z +2.4, p 0.151 `null` |
| hair plate | 56.753 +- 0.338 | 56.767 | 47.065 +- 0.562 | 47.079 | -9.688, z -28.7 `result` |
| leg campaniform | 49.672 +- 0.046 | 49.698 | 24.887 +- 0.009 | 24.900 | -24.785, z -542.7 `result` |
| haltere | 18.893 +- 0.078 | 18.892 | 16.707 +- 0.228 | 16.709 | -2.186, z -28.2 `result` |

|L - C| = 1.19 Hz against the predeclared 10 Hz tolerance: **the chordotonal precondition is met**, and the audit's
per-seed brackets [86.47 86.89 85.95 86.20 85.58] / [88.04 88.19 86.76 88.16 85.89] are exact. Sidedness also exact:
chordotonal L-R **L +13.620 +- 0.111** (L 92.896 / R 79.276, every leg) vs **C -0.336 +- 0.033** (per leg 87.2-87.7);
hair-plate L-R L +9.242 vs C -0.190; leg-MN L-R L +0.867 +- 0.006 vs C +0.144 +- 0.005 vs A +0.154.

**C5. A genuinely sidedness-proof half of the result.** On the **left** side, arm L carries **more of all three leg
channels than arm C** (chordotonal 92.90 vs 87.24, hair plate 61.25 vs 46.97, campaniform 49.67 vs 24.88) and still
fires AN04B003_L lower (19.10 vs 23.13) and DNa02_L lower (0.431 vs 0.558). So the DNa02_L call cannot be an
under-drive artefact of the DC bias -- the bias works *in L's favour* there. The audit never makes this argument and
should. (The DNa02_R row is the opposite case -- see the Corrections.)

**C6. The mn_ref derivation (item 8) reproduces exactly**, from `out/vncd3/*_body.npz` and
`out/vncd4/smoke/cal_*_body.npz`:

| quantity | mine | `mn_ref_derivation.json` |
|---|---|---|
| f(30) on B's recorded MN distribution | 23.422 Hz (recorded 23.423 +- 0.092) | 23.42 vs 23.4 |
| open-loop mn_ref from A / B / C distributions | 4.163 / 4.988 / **8.452** | 4.163 / 4.988 / 8.452 |
| CPU cal B at 30 Hz | 23.318 Hz, leg MN 3.002 / 2.700 | 23.32, 3.002 / 2.700 |
| CPU cal L at 8.45 Hz | 95.479 Hz, leg MN 5.636 / 4.685 | 95.48, 5.636 / 4.685 |
| loop slope k | 0.03201 Hz/Hz | 0.032 |
| loop gain 140k/8.45 | 0.530 | 0.53 |
| fixed point | **8.8396 Hz** | 8.839 (chosen 8.84) |
| without the loop correction | 9.2629 | 9.263 |
| d level / d mn_ref | -17.57 Hz/Hz | -17.7 |

Realised on the B200: 86.22 +- 0.50 vs the predicted 88.1 -- a 1.9 Hz shortfall, inside the declared tolerance.

**C7. Both round-3 reducer defects are closed (item "section 3"), verified with my own code.**
`out/vncd3/room_C_r0`: corr(AN04B003 L-R, chordotonal cmd L-R) **-0.3393** at lag 0 vs **-0.2937** at lag -1;
tripod swing **-3.493** vs **-3.367**. Arm B: 0.1280 / 0.1270. vncd4 arm C, 5 runs: **-0.3457 +- 0.0054** (lag 0) vs
**-0.2974 +- 0.0059**; swing **-3.600 +- 0.031** vs **-3.391 +- 0.039**. The DNa02 same-mask identity
`DNa02_L_hz - DNa02_R_hz == DNa02_LR_hz` holds to < 1e-6 in **all 20 runs**.

**C8. The decomposition magnitudes (item 6) are exactly as printed.** `dna02_decompose_summary.csv`:
DNa02_L E/I/net A +741.2/-1047.5/-306.3, L +1336.8/-1277.7/**+59.1**, C +1363.1/-1259.7/**+103.4**,
D +1353.9/-1256.8/+97.1; AN04B003/L 9.21 / **294.90** / **357.13** / 356.26; PS059/L -90.01 / **-209.58** /
**-206.93** / -204.12. DNa02_R net -362.8 / **-272.6** / **-158.4** / -155.8; AN04B003/R 4.66 / **250.16** /
**357.04** / 355.33; PS059/R -49.71 / **-171.00** / **-168.88** / -170.63. AN04B003's rates 19.10 / 16.54 (L) vs
23.13 / 23.62 (C) confirmed **independently of the summary CSV**, from the room JSON `watch` block.
IN12B014 L/R 11.213 / 8.930 (L) and 11.141 / 8.492 (C) -- the audit's 11.2/8.9 and 11.1/8.5. PS059's inhibition is
within 1.3 % / 1.2 % between the arms, as claimed.

**C9. The compass rows at 4 v 4 (item 9).** `compass_flip_chain.csv`, floor p(4,4) = 2/C(8,4) = **0.02857**:
PS196_b flip A -0.179, **L -0.643 +- 1.169**, **C +0.464 +- 1.735**, **D -1.321 +- 1.672**, every one `null`;
AN04B003 flip **C -10.548 +- 1.130 `result`**, **D -11.107 +- 1.247 `result`**, **L +0.500 +- 0.514 `null`**
(p 0.486), A -0.274 p 0.057 `null`; GLNO L-R 26.86-28.14 across every arm and phase; PEN_a <= 0.100, PEN_b <= 0.058,
EPG <= 0.571 all `null`; AN06A026 / AN07B035 -1.64 / -1.41 (C), -2.29 / -1.75 (D), **+2.91 / +2.79 (L)**, all
`result`. Self-turn realised +97.6..+105.2 (ccw) / -89.4..-95.3 (cw); bump vector strength 0.762-0.766 at every phase
of every arm. The pinned-fly chordotonal levels are L 85.5 (rest) / 92.1 (ccw) vs C 92.8 / 90.7, as stated.

**C10. Behavioural secondaries reproduce.** median |yaw| clean L 3.542 +- 0.076 vs C 1.587 +- 0.086 (-1.955, z -25.7
`result`); mean signed yaw L +3.437 +- 0.072 vs C +1.219 +- 0.085 (-2.218, z -30.8); flies with positive mean yaw
A 13.0, L **16.0 +- 0.0**, C 15.4, D 15.6; flies off the table L **12.0 +- 0.7** vs C **8.8 +- 1.6** (-3.2, z -4.5
`result`); hops 0.513 vs 0.450 `null`; clean fraction 0.899 vs 0.926 `null`; step 7.790 +- 0.107 Hz / stance 0.766;
|amp L-R| 0.0164; GF 33.65 vs 31.45; power MN 15.18 vs 15.35. Every `sided_frames.csv` row in section 4.3 matches,
including the `_lagm1` secondaries, and the PS059 rows are empty exactly as declared.

**C11. The four ledger rows are real, honest and correctly restricted.** Lines 166-169 of
`flyverse/data/expected_responses.csv`, all `op report`, all `scored` 0, all carrying the caveat that the values are
the *fit's* values and not a tabulated number. Both citations exist and are cited for what they support:
**DeAngelis, Zavatone-Veth & Clark 2019, eLife 8:e46409** (Fig 1E stance duration vs forward speed, a power-law fit;
Fig 2 tripod; Fig 6C outer-limb step length rises with yaw rate) and **Mendes, Bartos, Akay, Marka & Mann 2013,
eLife 2:e00231** (Fig 2B stance duration falls with speed while swing stays roughly constant; Fig 4 tripod).
The arithmetic in the notes is right (110/140 = 0.79; 30.7/60.7 = 0.51; 1/0.140 = 7.1 Hz; 1/0.0607 = 16.5 Hz).
`lit.walk.outer_leg_step_ratio_in_turn` has an **empty value** with a note saying no number was read off the figure --
the correct handling, and no ratio is invented anywhere. `lit.walk.swing_duration_ms` is flagged UNCERTAIN with the
30-50 ms bracket. The file is 169 rows; the audit's "142 rows" is `load_table()`'s post-filter count, not a claim
about the file.

**C12. "silent" / "inverted" (item 10).** Neither word appears anywhere in the audit -- nothing to correct. The three
afferent->DNa02 walks the audit names (SNpp45 direct; SApp -> PS059; SNpp45 -> IN13B001 -> AN04B003) are indeed the three
in `paths_summary.csv`, and all three are live under L/C/D (SNpp45 55.8/46.2 Hz, PS059 11.8/11.6, IN13B001 76.4/65.2,
AN04B003 17.8/23.4). What the audit omits is the third walk's **sign** -- see R1.

---

### The level fractions (item 5) and the sidedness confound (item 7)

**Fractions (L-A)/(C-A), recomputed, with the per-seed paired spread (same seed, same block):**

| key | A | L | C | fraction | per-seed r0..r4 |
|---|---|---|---|---|---|
| DNa02_L | 0.0169 | 0.4309 | 0.5581 | **0.765** | 0.794 0.716 0.765 0.759 0.791 |
| DNa02_R | 0.0796 | 0.2862 | 0.3892 | **0.667** | 0.597 0.640 0.668 0.727 0.715 |
| yaw SD clean | 2.744 | 7.300 | 7.872 | **0.888** | 0.874 0.846 0.899 0.888 0.938 |
| DNa02-active | 0.0068 | 0.0508 | 0.0673 | **0.727** | 0.713 0.684 0.728 0.750 0.767 |
| DNa02 L-R | -0.0628 | 0.1447 | 0.1690 | 0.895 | 1.142 0.811 0.895 0.803 0.872 |
| straightness | 0.9947 | 0.4975 | 0.8332 | **3.078** | 2.908 2.720 3.341 3.261 3.257 |

The 77 / 67 / 89 % figures are right (76.5 / 66.7 / 88.8) and the per-seed scatter is tight (<= +-0.07), so
"the level accounts for most of it" is **fair for those three keys**. It is *not* a general statement: on straightness
the level control **overshoots by 3.1x**, and the framing in the Report's `summary` and `key_claims` quotes only the
three keys under 1 and is silent on the one above 1 (section 0 does disclose the behavioural reversal in prose).
The deeper caveat is that "the level" in these fractions is *the round-2 law at a raised gain*, which carries the
+13.6 Hz DC L-R as well as the level -- so "77 % is the level" is really "77 % is the level **plus** the round-2 law's
sidedness".

**Is C v L confounded by sidedness rather than level? Partly -- and the present data cannot regress it out.**

1. **It cannot be regressed out.** The DC chordotonal L-R is effectively constant within an arm and disjoint between
   arms: L 13.620 +- 0.856 per fly, range [11.55, 15.21]; C -0.336 +- 0.203, range [-0.79, +0.25]. A between-arm
   regression would extrapolate a 13.96 Hz gap from a 3.7 Hz within-arm span. Worse, **the within-arm relation
   reverses sign between the arms** -- in L the L-R is the *cause* of the turn (corr(yaw, chord L-R) = +0.272), in C it
   is the kinematic *consequence* of it (-0.150), so the C-side regression (straight ~ chord L-R slope +0.465, r
   +0.677; DNa02 L-R ~ chord L-R slope -0.639, r -0.943) is reverse-causal and not transportable. Regressing on the
   L-R directly is therefore invalid in both directions.
2. **Regressing on its behavioural proxy (the signed drift) does settle straightness** -- see R2: 82-92 % of the
   straightness gap is the drift.
3. **It does not settle the yaw-SD row.** Fit inside L (`yawSD = 6.280 + 0.297|ybar|`, r +0.379) extrapolated to C's
   drift gives 6.66 against C's 7.87 -- the excess *grows* to +1.21. Fit inside C (`8.183 - 0.244|ybar|`, r -0.168)
   extrapolated to L's drift gives 7.34 against L's 7.30 -- the excess *vanishes*. The two directions contradict; the
   yaw-SD row is neither rescued nor refuted by the present data.
4. **It does not weaken DNa02_L; it strengthens it.** Within L, DNa02_L per fly rises with the fly's own drift
   (slope +0.0836, r +0.741), so L's larger drift *inflates* its own DNa02_L. Combined with C5 (L's left side is
   over-driven on all three channels), the +0.127 DNa02_L excess is a lower bound, not an artefact.
5. **DNa02_R is genuinely confounded.** L's right side sees 79.28 Hz chordotonal against C's 87.58 -- an **8.3 Hz
   per-side deficit**, not a match. AN04B003_R at 16.54 Hz is starved for that reason, and the audit's own
   decomposition says AN04B003 is 94 % of the DNa02_R gap. A level model fitted to the two sides of arm L
   (rate = base_A + g x level; g_L 0.199, g_R 0.205 -- reassuringly equal) puts an **unsided** level control's
   AN04B003 at ~17.8 / 18.0 Hz against C's 23.13 / 23.62 -- so the *pooled* AN04B003 excess survives at +5.4 / +5.7
   (observed +4.0 / +7.1), but it is redistributed across the sides. The DNa02_R row as printed mixes structure with
   a per-side level deficit and should be read as such.

**Which control separates it.** Of the two in section 7, **(a) the UNSIDED level control** (round-2 law on the
side-mean leg-MN rate at mn_ref 8.84) is the one that separates the sidedness -- it removes the +13.6 Hz DC L-R and
the +9.2 Hz hair-plate L-R at the same means. **(b) the modulation-only arm does not**, because it keeps the cycle's
law and therefore the cycle's hair-plate/campaniform values; it separates the per-phase modulation from the turn
kinematics, not from the sidedness. **Neither closes R1.** A **third** control is required and is not in the audit's
list: a *channel-matched* level control in which `hair_plate_max_hz` / `campaniform_load_hz` are set so that all
three commanded channel means match C's (47.1 / 24.9), at the same chordotonal 86-88 Hz. Without it the C-over-L
excess on DNa02 remains jointly attributable to (i) per-phase modulation of the chordotonal input and (ii) the
unmatched hair plate acting through the sign-negative SNpp45 -> IN13B001 -> AN04B003 route.

---

### Corrections (exact replacement text)

**1. Section 0, lines 63-66.** Replace:

> **The hair-plate and campaniform channels did not match, by construction and as predeclared:** L 56.8 / 49.7 Hz vs C
> 47.1 / 24.9 (the round-2 laws; C v L `result`, -9.7 / -24.8). L has MORE hair-plate and campaniform drive than C and
> still less DNa02, so neither can account for the C-over-L excess in the direction found; the haltere afferents follow
> the haltere MNs (L 18.9, C 16.7 Hz; PS059's inhibition of DNa02 is within 1-2 % between the arms).

with:

> **The hair-plate and campaniform channels did not match, by construction and as predeclared:** L 56.8 / 49.7 Hz vs C
> 47.1 / 24.9 (the round-2 laws; C v L `result`, -9.7 / -24.8). **They are a live alternative explanation of the
> C-over-L excess, not an excluded one.** The hair plate's dominant route into DNa02 is NEGATIVE: the strongest
> three-step afferent walk in `paths_summary.csv` is `SNpp45 -> IN13B001 -> AN04B003 -> DNa02`, signs `+,-,+`
> (`gain_if_signed` -3.0e+04), and SNpp45 IS a hair-plate afferent (53 of the channel's 113 cells). It moves as that
> route predicts: SNpp45 55.8 Hz (L) vs 46.2 (C), IN13B001 **76.4 vs 65.2**, AN04B003 17.8 vs 23.4
> (`trace_{L,C}_vs_A.json`, `tables.per_type`). An additive level model with an inhibitory hair plate
> (w_chordotonal +0.53, w_hair_plate -0.51 Hz/Hz) reproduces BOTH the within-L left-vs-right side contrast AND the
> whole C-over-L AN04B003 difference with a ZERO structure term. The excess is therefore attributable to the per-phase
> modulation OR to the unmatched hair plate (and the unmatched campaniform, whose route was not examined); this batch
> does not separate them. The haltere afferents follow the haltere MNs (L 18.9, C 16.7 Hz; PS059's inhibition of DNa02
> is within 1-2 % between the arms).

**2. Section 0, lines 49-51.** Replace *"A threshold unit fed a modulated Poisson input fires more than one fed a
steady input of the same mean; that is the structure's contribution, and it is the whole of the C-over-L excess on the
named rows."* with:

> A threshold unit fed a modulated Poisson input fires more than one fed a steady input of the same mean; that is the
> mechanism this round PROPOSES for the residual. It is not demonstrated here: the level control's hair plate is
> +9.7 Hz and its route into DNa02 through IN13B001 is sign-negative, so a pure level account of the same AN04B003
> difference exists (section 0, hair plate). Section 7 item 6 and the channel-matched control are what would settle it.

**3. Section 0, lines 25-31 and the Report's `summary` / `key_claims`.** Where straightness is listed among the rows
that show the structure contributing, add the attribution. Replace *"straightness **+0.34** (0.498 +- 0.022 ->
0.833 +- 0.016; z +15.3, CALLED: the cycle fly is the STRAIGHTER one)"* with:

> straightness **+0.34** (0.498 +- 0.022 -> 0.833 +- 0.016; z +15.3, CALLED: the cycle fly is the STRAIGHTER one --
> but a per-fly regression inside L (straightness = 0.987 - 0.1424 x |signed yaw|, r -0.78, n 80) predicts 0.806 at
> C's own drift against C's observed 0.833, so 92 % of this row is the round-2 law's DC sidedness, not the per-leg /
> per-phase structure; the reverse fit inside C gives 82 %. Read it as a sidedness row.)

**4. Line 36.** "76 % of its DNa02-active fraction (0.044 of 0.060)" -> **"73 % of its DNa02-active fraction
(0.0440 of 0.0605)"**.

**5. Section 4.2 table and Report `key_claims`.** "chordotonal per-leg SD over time (Hz); variance in the 5-12 Hz
band | 23.9; 0.07 | **46.4; 0.29**" -> recompute or drop the band fraction. My recompute (periodogram of the
mean-removed per-leg series on window non-airborne frames, per leg per fly, run mean, 5 seeds) gives
**L 0.076 +- 0.002, C 0.814 +- 0.030**; Welch(256) gives 0.086 / 0.942. Replace "0.29" with **"0.81"** and state the
estimator, or delete the column. Also L's p10/p90 "62.6 / 125.4" -> my **57.0 / 118.7**.

**6. Section 5, table.** Arm A per-run drift range "(all within -0.006 .. +0.006)" -> **"(-0.0071 .. +0.0068)"**;
arm C "(-0.006 .. +0.007)" -> **"(-0.0049 .. +0.0094)"**. Section 0: "inside -0.0087 .. +0.0093 w/s" ->
**"inside -0.0087 .. +0.0094 w/s"**.

**7. Section 5, prose.** "PS059 flips +6.0 (C, result) / +4.5 (D) / +3.7 (L, result) / +2.8 (A, result)" ->
**"+6.0 (C, result) / +4.5 (D, z +4.0 but p 0.057, `null`) / +3.7 (L, result) / +2.8 (A, result)"**.
"PS196_b fires at 6.7-7.6 / 3.3-4.2 Hz" -> **"6.6-7.6 / 3.2-4.2 Hz"**.

**8. Section 4.1, row "|DNa02 L-R| per frame (Hz)".** Replace the four level cells `0.28 | 0.86 | 1.07 | 1.05` with
**`0.096 | 0.676 | 0.882 | 0.863`** (`room_table.csv` key `DNa02_abs_LR_hz`). The diff +0.21 and z +12.2 are correct.

**9. Section 4.4, row "DNa02_L rate in the decompose window (Hz)".** Delete the row and replace with:

> | AN04B003 rate carried in the CSV's `rate_hz` column (Hz) | 0.60 | 19.09 | 23.12 | 23.06 |
> | (DNa02_L's own window rate is in 4.1: 0.017 / 0.431 / 0.558 / 0.522) | | | | |

and the same for DNa02_R (0.31 / 16.54 / 23.61 / 23.50).

**10. Section 4.4, "the whole difference is AN04B003".** Replace with:

> On DNa02_R the whole difference is AN04B003 (+107 of the +114 mV/s net gap, 94 %). On DNa02_L the +44 mV/s net gap
> is a cancellation: AN04B003 +62, SNpp45 -39 (the unmatched hair plate), the other excitatory rows +3 and the
> inhibitory total +18 (41 % of the gap).

**11. Section 8, provenance.** "Every JSON carries `provenance` (... the sense's `mn_ref_hz` and spec)" -> add:
**"except the compass runs: all 16 `compass_*_run.json` record `mn_ref_hz` null and the 64 phase JSONs carry no arm /
family / spec, so the compass L arm's `--mn-ref-hz 8.84` is verifiable only from the job line in
`out/vncd4_cluster.log` (and from its realised pinned level, 85.5-92.1 Hz against the default's ~23)."**

**12. Section 8 and line 5, file path.** `out/vncd4/analysis/` (`analysis_console.txt`, ...) ->
**`out/vncd4/analysis_console.txt` (beside the directory) and `out/vncd4/analysis/pairs_console.txt`**.

**13. Line 6 and the Report's `validation`.** "stamped **2026-09-15T02:00:02Z, before the submission**; run submitted
02:01Z" -> **"stamped 2026-09-15T02:00:02Z (`predeclared.json`); `out/vncd4_cluster.log` carries no absolute
timestamp and no run JSON carries a `submitted_at`, so the ordering rests on local file mtimes: predeclared.json
02:00:02Z against a first-job start no earlier than ~02:00:55Z (the log's 02:33:12Z mtime less its reported 32.3 min).
`batch.sh` was written at 02:00:01Z, one second before the stamp."**

**14. Section 2.** "The critique's ~3.5 Hz ... would give 110-145 Hz" -> **"would give 143-150 Hz (the clip saturates
at 150)"**.

**15. Section 7, add item 1(c).** After the unsided and modulation-only controls:

> (c) a **CHANNEL-MATCHED level control**: the round-2 law at mn_ref 8.84 with `hair_plate_max_hz` and
> `campaniform_load_hz` set so that the realised hair-plate and campaniform commanded means match C's 47.1 / 24.9 Hz.
> This is the control the C-over-L attribution actually needs: without it the DNa02 excess is attributable either to
> the per-phase modulation or to the unmatched hair plate acting through `SNpp45 -> IN13B001 -| AN04B003`. Neither (a)
> nor (b) matches those two channels.

**16. Section 7 item 6 / Report `recommendations`.** Strengthen the single-cell check to include the alternative:
**"AN04B003 under (i) steady vs 8 Hz-modulated chordotonal input at the same mean with IN13B001's rate clamped, and
(ii) the hair-plate level varied alone at a fixed chordotonal level -- the two together settle whether the C-over-L
AN04B003 difference is modulation or hair-plate disinhibition."**

---

### Verdict

**Mostly sound.** The engineering is clean and the arithmetic is honest: one submission, correct provenance on every
room run, a predeclaration whose families and decision rule match what was analysed, two genuine reducer fixes that
reproduce the round-3 skeptic's numbers to three decimals, a derivation (`mn_ref_hz` 8.84) that I reproduced to four,
a level match verified from the recordings (86.22 +- 0.50 vs 87.41 +- 1.04, inside a 10 Hz tolerance), and **every one
of the fifteen family comparisons reproducing to the printed precision with my own exact Mann-Whitney, my own z and
my own Holm -- not one verdict flips**. The audit also self-reports its own two defects (the `watch_of` edit, the
`rate_hz` column) rather than hiding them.

What is unsound is the **attribution**, which is the round's decisive question. The decision rests on the claim that
the C-over-L excess "is the whole" of the per-phase structure, and that claim rests in turn on dismissing the two
unmatched channels because "more drive and less DNa02 cannot produce the excess". The audit's own paths output
contradicts that: the hair plate's dominant route into DNa02 is sign-negative, its intermediate IN13B001 is 11.1 Hz
higher under L exactly as that route predicts, and an additive level model with an inhibitory hair plate reproduces
the whole AN04B003 gap with no structure term at all. A second called row -- straightness -- is 82-92 % explained by
the round-2 law's DC sidedness, and a third -- DNa02_R -- is confounded by an 8.3 Hz per-side chordotonal deficit that
the whole-channel match hides. Six further numbers do not match their own named files. None of this touches the
DNa02_L call, which is the one row that survives every confound I could apply.

**What the round can now say.** With the CHORDOTONAL channel matched to 1.2 Hz (precondition met, `null`), **C v L is
`result` on DNa02_L (+0.127 Hz, z +9.2), DNa02_R (+0.103, z +16.7) and the clean yaw SD (+0.57 deg/s, z +7.1),
Holm-called at m 5** -- so **something other than the chordotonal mean separates the leg cycle from the round-2
transducer**, in the direction "more DNa02". The DNa02_L row is the robust one: arm L's left side carries *more*
chordotonal, hair-plate and campaniform drive than arm C's and still fires DNa02_L less, so no under-drive or
sidedness account of it survives. But **"the per-leg / per-phase STRUCTURE contributes beyond the LEVEL" is not yet
supported**: L differs from C in three ways at once -- per-phase modulation, a +9.7 / -24.8 Hz mismatch on the two
unmatched leg channels whose dominant route into DNa02 is sign-negative, and a +13.6 Hz DC chordotonal L-R -- and no
arm in this batch separates them. The straightness row is `result` but reads as the DC sidedness, not as structure;
the DNa02_R row is `result` but is partly a per-side level deficit; the DNa02 L-R row is `null`. `L v A` is `result`
on all five primaries and the afferent level (as delivered by the round-2 law, sidedness included) reaches 76 % / 67 %
/ 89 % of the cycle's DNa02_L / DNa02_R / yaw-SD rise over the shipped path, with tight per-seed scatter -- that part
of the framing is fair, and it is the round's solid finding. **D v C** stands as reported (DNa02_L -0.036 called, no
behavioural row). The compass answer stands at 4 v 4: no arm moves the bump, and AN04B003's signed report of the turn
(-10.5 Hz) exists under the cycle and not under the level control. Nothing is adopted, and adoption now needs three
controls, not two: the unsided level control, the modulation-only arm, **and a channel-matched level control**.
