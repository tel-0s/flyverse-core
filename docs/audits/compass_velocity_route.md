# Compass round 7: the PS196_b velocity route

Protocol written 2026-09-15 before any job was submitted; valid results are in section 5. Ran on the house
cluster (`--target house`), replacement batch `cx8r` after the invalidated `cx8` declaration. Neither round 6B
(`cx7`) nor 4d (`vncd7`) demonstrated a following compass, so the predeclared condition to run this round was met.

## 1. Question

The connectome gives PEN one non-ring input of size, GLNO, and GLNO one large non-ring input, PS196_b (1,801 syn, 19-21 %
of its input; Wang 2026 finding 3, reproduced in our cache to the synapse). Our body-side rounds reach the same cell and
stop there: the ascending report arrives at PS196_b unsigned (NOTES compass round 2: L-R moves the same way in both turn
directions under the Coriolis stop-gap). **If PS196_b is given a signed turn input, does the bump follow the fly's own
turn -- with the ring's DC brake held and GLNO signed -- and does anything downstream (DNa02 L / R) desynchronise?**

## 2. Arms (6 seeds each, prescribed-turn `cx_wedge` protocol)

| arm | instruments | what it tests |
|---|---|---|
| S | none (`raw`) | reference |
| V | `sided_turn_afferent` only | does a signed PS196_b input reach GLNO / PEN at all at shipped gains (predicted: GLNO L-R moves, no bump: the unheld ring has little activity) |
| HG | `ring_dc_hold` + `glno_sign` (= 6A's H3G) | the near-bump without a velocity input (replicates 6A) |
| HGV | `ring_dc_hold` + `glno_sign` + `sided_turn_afferent` | **the arm**: the bump should move with the turn |
| HGV- | HGV with the afferent sign flipped | the sign control: the bump should move the other way, or the effect is not the afferent |
| HGVp | HGV with the hold restricted to PEN (`^(ExR6|ER6|ER4m)$:^PEN_`; EPG keeps its DC input) | added 2026-09-15 after 6B (`compass_local_recurrence.md`; wording corrected after its skeptic): H3E's hump sits near the driven tile but is ~7 wedges wide, and the hold removes ring inhibition from every EPG, driven and off-block alike, so 6B cannot say which part of it matters; HGVp keeps EPG's DC input to test whether that narrows the hump |

The afferent instrument: Poisson `poisson_hz` on AN07B037_a/_b (and CB0675 / GNG580 / PS047_b in a second variant only
if V is null on the first), rate = `k * max(0, +-yaw_deg_s)` on the side the connectome's contralateral routing
implies, `k` swept over {0.25, 0.5, 1.0} Hz per deg/s as **labelled levels, unverified** (no PS196_b recording exists;
search named in `PRESETS_SPEC.md` section 3). One level (0.5) is the predeclared primary; the other two are
descriptive.

## 3. Predeclared measures and family

Primary (Holm, m = 6, 6 v 6 exact U, floor 0.0022 x 6 = 0.013 -- satisfiable; m was 5 until the HGVp arm was added, before submission):

1. `bump_follow_wedges_per_s` HGV vs HG: the bump's mean angular velocity over the turn, sign-matched to the turn
   (ideal 4.0 w/s at 90 deg/s; 6A measured 0.00 +- 0.01 in every arm).
2. `bump_follow_wedges_per_s` HGV vs HGV-: sign flip.
3. `GLNO_LR_hz` V vs S: the afferent reaches GLNO with a side.
4. `PEN_LR_hz` HGV vs HG: the side reaches PEN.
5. `DNa02_LR_hz` HGV vs HG: anything gets back down.
6. `frac_confined_post` HGVp vs HGV: keeping EPG's DC input confines the bump (added with the HGVp arm, 2026-09-15, before submission).

Descriptive (no verdict): bump survival / rate / width per arm as in 6A, the k-sweep (HGVk025, HGVk1 on HGV only),
per-type ring rates (6B's recorded groups, merged at a41d0f2), PS196_b's own L-R at each k, the afferent / PS196_b /
GLNO / PEN / DNa02 L-R per side. The turn in `cx_wedge` is a prescribed protocol parameter (`--turn 90
--turn-window 0.5:3.5`), not a body: the instrument reads the prescribed yaw, and the body-driven version is the
room run that follows a result. Batch: 8 arms x 6 seeds = 48 jobs in two `cluster_run.py` calls, `out/cx8/batch.sh`,
`set -o pipefail`.

**Eligibility added before submission, 2026-09-15 (instruments review).** Primaries 1 and 2 use only
runs with `bump_follow_confined_frac >= 0.5` over the prescribed turn window. This makes the draft
code's gate explicit before any cx8 result exists. Every excluded run, filename and run id is retained
in the ungated per-seed table; the comparison records the included ids and actual sample sizes.
No eligible measurements is `undetermined`; fewer than four runs in either arm is `underpowered`
under `common.compare`. Holm remains one family of six, including missing p values, applied to the
actual eligible samples. The printed `6 * p_floor` is the first-step bound, not an impossibility
claim for a contrast that falls later in the Holm order. A rejected gate is not a zero-velocity observation.

For the functional outcome, 1 and 2 must both be positive `result`s after Holm, the eligible HGV
mean must be positive, and the eligible HGV- mean negative. A difference between two same-direction
drifts does not pass the sign control. `bump_follow_wedges_per_s` is the least-squares slope of the
unwrapped 16-wedge centre on [3.5, 6.5) s of the full 8 s recording, times sign(prescribed yaw).
The 50% confinement criterion is an experimental eligibility rule, not a physiological rate gate.

The generator freezes `out/cx8/predeclared.json` with `--predeclare out/cx8` after review and before
submission: six contrasts, gate, resolved LIF parameters per arm, eight arm specifications, six seeds,
protocol, source SHA-256s (explicit LF normalization), and byte hashes of the generated wrapper and
arm manifest. Existing declarations cannot be overwritten. Both wrapper calls explicitly select
`--target house`, run sequentially with one fetch directory, and abort if either client fails.

Verdict vocabulary result / null / underpowered / undetermined per INTERP 10.4. Per-seed scatter emitted by the
analysis script to `out/cx8/analysis/per_seed.csv` and pasted (rule 28). Nothing adopted into `raw` under any outcome;
a `result` on 1 + 2 makes `sided_turn_afferent` + `ring_dc_hold` + `glno_sign` the first `instrumented` list and sends
it to the suite (PRESETS_SPEC section 2 item 5) before any room run.

## 4. What each outcome means

- **1 and 2 positive result, with the sign control satisfied:** the route supports turn following under these
  counterfactual instruments. This establishes model sufficiency, not a unique physiological explanation.
  Next: the suite under `instrumented`, then the odour room with DNa02 L / R as the readout.
- **3 result, 1 null:** the afferent changes GLNO L-R, without demonstrated ring following. Test the GLNO -> PEN
  transfer next; the present contrasts alone cannot identify a receptor or kinetic cause.
- **3 null:** no detected GLNO L-R change from AN07B037 at the primary k=0.5 level in the V arm. Try the declared
  alternative afferents; this is not a null on PS196_b itself or a test of every k in V (the k sweep is on HGV).
- **Anything with HGV- not flipped:** the functional sign-control criterion fails; the individual statistical
  verdicts are retained.

The last rule means the sign-control criterion failed; it does not prove the afferent has no effect.
An unavailable or underpowered follow comparison demonstrates no turn-following compass, and retains
its statistical label. In that case a `result` on 3 authorizes the single GLNO-to-PEN diagnostic
follow-up; a `null` on 3 authorizes the alternate-afferent diagnostic. An underpowered or undetermined
3 leaves the location of the failure unresolved. At most one follow-up batch is permitted and its
arms and decisions must be frozen separately before submission. No null is converted into evidence
of equivalence or a unique receptor mechanism.

## 5. Results: valid replacement cx8r

### 5.1 Answer

The signed afferent reaches GLNO, but this round demonstrates no turn-following compass.
V minus S raises GLNO L-R by +2.2183 Hz (result, Holm p 0.0130). No HGV run passes the predeclared
50% turn-window confinement gate: primaries 1 and 2 are undetermined, with 0 versus 2 and 0 versus 1
eligible runs. These are unavailable comparisons, not zero velocity or statistical nulls.

PEN L-R HGV minus HG is null (+0.7813 Hz, z 0.4221), and DNa02 L-R is null (all values zero).
The PEN-only hold gives no confined post-pulse frames in any seed; its contrast with HGV is null
under the full rule despite Holm p 0.0130, because |z|=2.3613 is below the declared threshold of 3.
The low and high k arms also have only one eligible run each. No arm meets the joint ledger
survival/rate/width criteria, and nothing is adopted.

This result uses 48 CUDA runs in cx8r, six per arm, with zero analysis problems. An independent
CPU trace calculation reproduces 972 measurements and 52 source hashes. The first cx8 attempt
is retained as invalid because Astra's frozen receptor rule was wrong; it is not used for inference.
Primary 3's result, with no demonstrated follow, selects the single predeclared GLNO-to-PEN transfer
follow-up. The suite and odour-room milestone are not unlocked. Independent skeptic pending
(Fable, when accounts reset).

### 5.2 The six primaries

Source: `out/cx8r/analysis/compare.csv`; runs are the replicate unit, one Holm family of six.
The first two rows retain the family slots despite having no eligible HGV observations.

| measure / contrast | n | verdict | difference | z | p | Holm p |
|---|---|---|---:|---:|---:|---:|
| 1_bump_follow_HGV_vs_HG | 0 v 2 | undetermined | NA | NA | NA | NA |
| 2_bump_follow_HGV_vs_HGV- | 0 v 1 | undetermined | NA | NA | NA | NA |
| 3_GLNO_LR_V_vs_S | 6 v 6 | result | 2.2183 | 1671.8089 | 0.0022 | 0.0130 |
| 4_PEN_LR_HGV_vs_HG | 6 v 6 | nan | 0.7813 | 0.4221 | 0.5887 | 1.0000 |
| 5_DNa02_LR_HGV_vs_HG | 6 v 6 | nan | 0.0000 | NA | 1.0000 | 1.0000 |
| 6_frac_confined_post_HGVp_vs_HGV | 6 v 6 | nan | -0.2637 | -2.3613 | 0.0022 | 0.0130 |

Primary 3's large z (1671.8) is a +2.2183 Hz effect divided by the nearly zero raw-reference SD
(0.0013269 Hz); it is not a claim of a huge absolute signal. All six V values exceed all six S values.
Primary 6 separates by rank but fails the additional |difference/reference SD| >=3 requirement in
`common.compare`; its verdict remains null. DNa02 L and R turn-window rates are zero in every run,
so the equal-zero contrast is null with undefined z, not a detected steering signal.

### 5.3 Eligibility and the saved traces

Eligibility is evaluated on [3.5,6.5) s, 300 frames. Included ids for the two follow contrasts are
`HG_s0.json#0`, `HG_s4.json#0`, and `HGV-_s4.json#0`; HGV includes none.
The original twelve HGV/HGV- velocity values remain in the ungated table below, never substituted for
eligible observations. The 50% gate does not certify physiological rate or continuous tracking.

![All follow traces and their confinement masks](assets/compass7_follow.png)

Generated by `scripts/cx8_plot.py` from every NPZ. Gray means the run fails eligibility. The raster is
the per-frame confinement mask. Apparent large unwrapped centre movements during loss of confinement
are not following. Even eligible HG/HGV- traces can jump and then park rather than track the ideal slope.

### 5.4 Descriptive magnitudes and k sweep

Means +/- sample SD from `out/cx8r/analysis/per_seed.csv`; no extra verdicts. Bump rate and width use
confined post-pulse frames only; survival is the end of the last confined frame minus pulse end,
not an assertion of uninterrupted confinement. NA means no confined frame.

| arm | eligible / 6 | survival s | bump Hz | width wedges | post confined fraction | turn confined fraction |
|---|---:|---:|---:|---:|---:|---:|
| S | 0 | 0.0000 +/- 0.0000 | NA | NA | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| V | 0 | 0.0000 +/- 0.0000 | NA | NA | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| HG | 2 | 4.7750 +/- 0.2857 | 162.5186 +/- 2.6709 | 3.9419 +/- 0.0739 | 0.2980 +/- 0.1608 | 0.2894 +/- 0.2148 |
| HGV | 0 | 4.8533 +/- 0.1726 | 158.3621 +/- 7.9558 | 3.9217 +/- 0.0686 | 0.2637 +/- 0.1117 | 0.2389 +/- 0.1334 |
| HGV- | 1 | 4.8683 +/- 0.1327 | 161.1477 +/- 3.7156 | 3.9269 +/- 0.0653 | 0.2793 +/- 0.1422 | 0.3000 +/- 0.1843 |
| HGVp | 0 | 0.0000 +/- 0.0000 | NA | NA | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| HGVk025 | 1 | 4.7833 +/- 0.1757 | 158.0102 +/- 4.8380 | 3.8997 +/- 0.1180 | 0.2627 +/- 0.1105 | 0.2311 +/- 0.2125 |
| HGVk1 | 1 | 4.8400 +/- 0.1463 | 159.8255 +/- 4.9640 | 3.9030 +/- 0.0701 | 0.2610 +/- 0.0976 | 0.2456 +/- 0.1396 |

The HGV k sweep (0.25 / 0.5 / 1.0) is descriptive on the held, GLNO-relabelled ring, not a V sweep.
It raises the actual afferent L-R and PS196_b side signal while confinement remains poor. No k is selected.

| arm | afferent L-R Hz | PS196_b L-R Hz | GLNO L-R Hz | PEN L-R Hz | DNa02 L-R Hz |
|---|---:|---:|---:|---:|---:|
| S | NA | 0.0000 +/- 0.0000 | -0.0000 +/- 0.0013 | 0.0168 +/- 0.0200 | 0.0000 +/- 0.0000 |
| V | 42.6139 +/- 2.4907 | -18.1618 +/- 1.7403 | 2.2183 +/- 0.7486 | 0.0201 +/- 0.0453 | 0.0000 +/- 0.0000 |
| HG | NA | 0.0000 +/- 0.0000 | 2.5705 +/- 1.2286 | 0.3259 +/- 1.8511 | 0.0000 +/- 0.0000 |
| HGV | 42.6139 +/- 2.4907 | -15.6582 +/- 1.9647 | 3.7955 +/- 1.8362 | 1.1072 +/- 0.3454 | 0.0000 +/- 0.0000 |
| HGV- | -44.2544 +/- 1.8199 | 17.0148 +/- 0.7345 | 0.6862 +/- 1.9132 | -0.7082 +/- 1.7688 | 0.0000 +/- 0.0000 |
| HGVp | 42.6139 +/- 2.4907 | -18.1863 +/- 1.6740 | 0.9355 +/- 0.9225 | 0.4425 +/- 0.1386 | 0.0000 +/- 0.0000 |
| HGVk025 | 20.0241 +/- 2.0311 | -3.7866 +/- 1.7394 | 2.1845 +/- 0.8605 | 0.7414 +/- 0.9517 | 0.0000 +/- 0.0000 |
| HGVk1 | 85.1863 +/- 2.9019 | -38.9186 +/- 2.2872 | 6.9057 +/- 1.4805 | 1.6210 +/- 0.5402 | 0.0000 +/- 0.0000 |

The afferent sign control reverses the PS196_b side report, but does not establish opposite eligible
compass motion. The k=0.5 law commands 45 Hz Poisson input at +90 deg/s; the measured afferent
spike-rate estimate is about 42.6 Hz. These are different quantities. The law remains unverified.

| arm | ExR6 pre / pulse / post Hz | ER6 pre / pulse / post Hz | ER4m pre / pulse / post Hz |
|---|---|---|---|
| S | 44.8075 / 85.0330 / 42.4232 | 25.1112 / 51.8955 / 23.5573 | 2.7937 / 11.7478 / 2.6888 |
| V | 44.8075 / 85.0330 / 41.9425 | 25.1112 / 51.8955 / 24.0624 | 2.7937 / 11.7478 / 2.7918 |
| HG | 82.3666 / 212.3720 / 199.7528 | 51.8365 / 152.1098 / 135.8932 | 17.8051 / 55.5079 / 56.3856 |
| HGV | 82.3666 / 212.3720 / 174.6682 | 51.8365 / 152.1098 / 118.6363 | 17.8051 / 55.5079 / 47.4391 |
| HGV- | 82.3666 / 212.3720 / 193.0166 | 51.8365 / 152.1098 / 129.8181 | 17.8051 / 55.5079 / 55.6868 |
| HGVp | 45.9326 / 93.0532 / 43.4631 | 25.0387 / 52.7097 / 24.2715 | 3.3498 / 14.6406 / 3.6218 |
| HGVk025 | 82.3666 / 212.3720 / 183.9557 | 51.8365 / 152.1098 / 125.7980 | 17.8051 / 55.5079 / 49.7776 |
| HGVk1 | 82.3666 / 212.3720 / 191.1707 | 51.8365 / 152.1098 / 131.4960 | 17.8051 / 55.5079 / 51.9294 |

These are observed group rates with the specified outgoing edges held; a held edge does not stop
its presynaptic cells firing. Ring-rate changes are descriptive network responses, not receptor evidence.

### 5.5 Per-seed record

Pasted by the report writer from `out/cx8r/analysis/per_seed.csv`, columns `arm,key,seeds,values`.
Order is seed 0,1,2,3,4,5 throughout; values are not sorted. The corresponding file/run-id mapping
is pasted from the same file (`files`, `run_ids`), so every value below has an explicit source.

| arm | files | run ids |
|---|---|---|
| S | S_s0.json,S_s1.json,S_s2.json,S_s3.json,S_s4.json,S_s5.json | S_s0.json#0,S_s1.json#0,S_s2.json#0,S_s3.json#0,S_s4.json#0,S_s5.json#0 |
| V | V_s0.json,V_s1.json,V_s2.json,V_s3.json,V_s4.json,V_s5.json | V_s0.json#0,V_s1.json#0,V_s2.json#0,V_s3.json#0,V_s4.json#0,V_s5.json#0 |
| HG | HG_s0.json,HG_s1.json,HG_s2.json,HG_s3.json,HG_s4.json,HG_s5.json | HG_s0.json#0,HG_s1.json#0,HG_s2.json#0,HG_s3.json#0,HG_s4.json#0,HG_s5.json#0 |
| HGV | HGV_s0.json,HGV_s1.json,HGV_s2.json,HGV_s3.json,HGV_s4.json,HGV_s5.json | HGV_s0.json#0,HGV_s1.json#0,HGV_s2.json#0,HGV_s3.json#0,HGV_s4.json#0,HGV_s5.json#0 |
| HGV- | HGV-_s0.json,HGV-_s1.json,HGV-_s2.json,HGV-_s3.json,HGV-_s4.json,HGV-_s5.json | HGV-_s0.json#0,HGV-_s1.json#0,HGV-_s2.json#0,HGV-_s3.json#0,HGV-_s4.json#0,HGV-_s5.json#0 |
| HGVp | HGVp_s0.json,HGVp_s1.json,HGVp_s2.json,HGVp_s3.json,HGVp_s4.json,HGVp_s5.json | HGVp_s0.json#0,HGVp_s1.json#0,HGVp_s2.json#0,HGVp_s3.json#0,HGVp_s4.json#0,HGVp_s5.json#0 |
| HGVk025 | HGVk025_s0.json,HGVk025_s1.json,HGVk025_s2.json,HGVk025_s3.json,HGVk025_s4.json,HGVk025_s5.json | HGVk025_s0.json#0,HGVk025_s1.json#0,HGVk025_s2.json#0,HGVk025_s3.json#0,HGVk025_s4.json#0,HGVk025_s5.json#0 |
| HGVk1 | HGVk1_s0.json,HGVk1_s1.json,HGVk1_s2.json,HGVk1_s3.json,HGVk1_s4.json,HGVk1_s5.json | HGVk1_s0.json#0,HGVk1_s1.json#0,HGVk1_s2.json#0,HGVk1_s3.json#0,HGVk1_s4.json#0,HGVk1_s5.json#0 |

| arm | key | seeds | values |
|---|---|---|---|
| S | survival_s | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| S | bump_hz_post | 0,1,2,3,4,5 | nan,nan,nan,nan,nan,nan |
| S | width_half_post | 0,1,2,3,4,5 | nan,nan,nan,nan,nan,nan |
| S | frac_confined_post | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| S | bump_follow_confined_frac | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| S | bump_follow_wedges_per_s | 0,1,2,3,4,5 | 6.9645,2.2765,2.2605,-1.5170,2.8487,2.8004 |
| S | AFF_LR_hz | 0,1,2,3,4,5 | nan,nan,nan,nan,nan,nan |
| S | PS196b_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| S | GLNO_LR_hz | 0,1,2,3,4,5 | -0.0018,-0.0014,0.0015,0.0003,0.0002,0.0010 |
| S | PEN_LR_hz | 0,1,2,3,4,5 | 0.0322,0.0002,0.0393,0.0333,0.0001,-0.0041 |
| S | DNa02_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| V | survival_s | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| V | bump_hz_post | 0,1,2,3,4,5 | nan,nan,nan,nan,nan,nan |
| V | width_half_post | 0,1,2,3,4,5 | nan,nan,nan,nan,nan,nan |
| V | frac_confined_post | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| V | bump_follow_confined_frac | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| V | bump_follow_wedges_per_s | 0,1,2,3,4,5 | 6.9784,9.9647,2.2605,-1.5135,2.7974,2.8349 |
| V | AFF_LR_hz | 0,1,2,3,4,5 | 43.3540,43.4948,41.8607,38.0538,45.3305,43.5896 |
| V | PS196b_LR_hz | 0,1,2,3,4,5 | -18.6666,-18.0815,-18.3178,-14.8798,-19.9797,-19.0456 |
| V | GLNO_LR_hz | 0,1,2,3,4,5 | 2.0053,2.6428,2.2497,0.8462,2.6170,2.9488 |
| V | PEN_LR_hz | 0,1,2,3,4,5 | 0.0306,0.0823,0.0379,0.0327,-0.0154,-0.0472 |
| V | DNa02_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HG | survival_s | 0,1,2,3,4,5 | 5.0000,4.7600,4.9100,4.9800,4.7700,4.2300 |
| HG | bump_hz_post | 0,1,2,3,4,5 | 160.7983,163.5748,161.2387,164.1680,158.9786,166.3532 |
| HG | width_half_post | 0,1,2,3,4,5 | 4.0000,3.9600,3.9945,3.8519,4.0000,3.8452 |
| HG | frac_confined_post | 0,1,2,3,4,5 | 0.4880,0.2000,0.3640,0.1080,0.4600,0.1680 |
| HG | bump_follow_confined_frac | 0,1,2,3,4,5 | 0.5067,0.2333,0.1633,0.0833,0.6067,0.1433 |
| HG | bump_follow_wedges_per_s | 0,1,2,3,4,5 | 1.1605,0.0277,2.0798,-2.8925,-0.2563,-0.0145 |
| HG | AFF_LR_hz | 0,1,2,3,4,5 | nan,nan,nan,nan,nan,nan |
| HG | PS196b_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HG | GLNO_LR_hz | 0,1,2,3,4,5 | 4.0459,1.6952,1.9251,1.6793,4.2543,1.8230 |
| HG | PEN_LR_hz | 0,1,2,3,4,5 | 1.8186,0.8305,0.0555,-3.2348,1.5721,0.9135 |
| HG | DNa02_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HGV | survival_s | 0,1,2,3,4,5 | 4.5700,5.0000,4.9900,4.9900,4.7700,4.8000 |
| HGV | bump_hz_post | 0,1,2,3,4,5 | 166.6009,153.8993,161.5561,161.6827,144.4275,162.0063 |
| HGV | width_half_post | 0,1,2,3,4,5 | 3.8939,3.9449,3.9847,4.0000,3.8903,3.8161 |
| HGV | frac_confined_post | 0,1,2,3,4,5 | 0.1320,0.2540,0.2620,0.4500,0.3100,0.1740 |
| HGV | bump_follow_confined_frac | 0,1,2,3,4,5 | 0.1700,0.1833,0.1200,0.4667,0.3333,0.1600 |
| HGV | bump_follow_wedges_per_s | 0,1,2,3,4,5 | 0.0265,-0.0118,-1.3781,1.0026,13.2114,0.4386 |
| HGV | AFF_LR_hz | 0,1,2,3,4,5 | 43.3540,43.4948,41.8607,38.0538,45.3305,43.5896 |
| HGV | PS196b_LR_hz | 0,1,2,3,4,5 | -14.7529,-14.5838,-16.1018,-13.1036,-18.7613,-16.6457 |
| HGV | GLNO_LR_hz | 0,1,2,3,4,5 | 5.2010,4.7047,0.3614,4.9387,3.0920,4.4750 |
| HGV | PEN_LR_hz | 0,1,2,3,4,5 | 0.8673,0.9985,1.7338,1.1978,1.0939,0.7520 |
| HGV | DNa02_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HGV- | survival_s | 0,1,2,3,4,5 | 4.7000,4.9800,5.0000,4.9800,4.7700,4.7800 |
| HGV- | bump_hz_post | 0,1,2,3,4,5 | 165.2961,162.9684,162.5770,155.1195,158.3199,162.6055 |
| HGV- | width_half_post | 0,1,2,3,4,5 | 3.8710,3.9259,4.0000,3.8404,4.0000,3.9244 |
| HGV- | frac_confined_post | 0,1,2,3,4,5 | 0.1240,0.2160,0.4800,0.1880,0.4300,0.2380 |
| HGV- | bump_follow_confined_frac | 0,1,2,3,4,5 | 0.1300,0.2267,0.4400,0.1733,0.6067,0.2233 |
| HGV- | bump_follow_wedges_per_s | 0,1,2,3,4,5 | -3.2726,-0.0293,1.6918,12.0726,-0.1796,-3.5662 |
| HGV- | AFF_LR_hz | 0,1,2,3,4,5 | -44.6523,-41.9164,-42.5673,-44.9742,-47.0013,-44.4152 |
| HGV- | PS196b_LR_hz | 0,1,2,3,4,5 | 16.9972,15.7752,16.9131,17.5792,17.9201,16.9041 |
| HGV- | GLNO_LR_hz | 0,1,2,3,4,5 | 0.6346,-0.2667,1.9787,-2.6218,1.8670,2.5254 |
| HGV- | PEN_LR_hz | 0,1,2,3,4,5 | -3.2168,0.3503,0.6704,-0.4745,0.9802,-2.5586 |
| HGV- | DNa02_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HGVp | survival_s | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HGVp | bump_hz_post | 0,1,2,3,4,5 | nan,nan,nan,nan,nan,nan |
| HGVp | width_half_post | 0,1,2,3,4,5 | nan,nan,nan,nan,nan,nan |
| HGVp | frac_confined_post | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HGVp | bump_follow_confined_frac | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HGVp | bump_follow_wedges_per_s | 0,1,2,3,4,5 | 13.7019,10.0035,2.9416,4.7291,12.1142,2.8142 |
| HGVp | AFF_LR_hz | 0,1,2,3,4,5 | 43.3540,43.4948,41.8607,38.0538,45.3305,43.5896 |
| HGVp | PS196b_LR_hz | 0,1,2,3,4,5 | -18.7740,-18.1010,-18.6128,-14.9199,-19.5479,-19.1623 |
| HGVp | GLNO_LR_hz | 0,1,2,3,4,5 | 0.4205,0.8473,0.2855,-0.0027,1.6580,2.4043 |
| HGVp | PEN_LR_hz | 0,1,2,3,4,5 | 0.7147,0.4006,0.4205,0.4161,0.3160,0.3872 |
| HGVp | DNa02_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HGVk025 | survival_s | 0,1,2,3,4,5 | 5.0000,4.4700,4.8700,4.7700,4.7700,4.8200 |
| HGVk025 | bump_hz_post | 0,1,2,3,4,5 | 151.1527,163.1611,156.3567,163.3758,159.2368,154.7782 |
| HGVk025 | width_half_post | 0,1,2,3,4,5 | 3.9737,3.9043,4.0000,3.7101,4.0000,3.8099 |
| HGVk025 | frac_confined_post | 0,1,2,3,4,5 | 0.2280,0.1880,0.3360,0.1380,0.4440,0.2420 |
| HGVk025 | bump_follow_confined_frac | 0,1,2,3,4,5 | 0.0433,0.2100,0.2500,0.0733,0.6333,0.1767 |
| HGVk025 | bump_follow_wedges_per_s | 0,1,2,3,4,5 | 4.3881,0.0255,-3.4755,-0.0298,-0.2620,-0.0045 |
| HGVk025 | AFF_LR_hz | 0,1,2,3,4,5 | 21.0319,19.2427,19.8771,16.6510,22.7142,20.6277 |
| HGVk025 | PS196b_LR_hz | 0,1,2,3,4,5 | -4.8934,-1.9878,-4.7860,-1.3550,-5.7143,-3.9830 |
| HGVk025 | GLNO_LR_hz | 0,1,2,3,4,5 | 1.4057,2.3191,2.0293,1.4725,3.7781,2.1021 |
| HGVk025 | PEN_LR_hz | 0,1,2,3,4,5 | 0.6346,0.6681,-0.7039,0.7659,2.2973,0.7865 |
| HGVk025 | DNa02_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |
| HGVk1 | survival_s | 0,1,2,3,4,5 | 4.6100,4.9800,4.9200,4.9800,4.7700,4.7800 |
| HGVk1 | bump_hz_post | 0,1,2,3,4,5 | 164.1049,161.1237,162.4651,162.2172,158.6591,150.3830 |
| HGVk1 | width_half_post | 0,1,2,3,4,5 | 3.8077,3.8842,3.8491,3.9504,4.0000,3.9267 |
| HGVk1 | frac_confined_post | 0,1,2,3,4,5 | 0.1560,0.1900,0.2120,0.2820,0.4260,0.3000 |
| HGVk1 | bump_follow_confined_frac | 0,1,2,3,4,5 | 0.1733,0.1933,0.1567,0.1600,0.5167,0.2733 |
| HGVk1 | bump_follow_wedges_per_s | 0,1,2,3,4,5 | -0.0003,-0.0168,0.0537,1.2786,-0.1349,-3.3759 |
| HGVk1 | AFF_LR_hz | 0,1,2,3,4,5 | 87.3898,84.6837,82.0996,81.5266,86.8890,88.5292 |
| HGVk1 | PS196b_LR_hz | 0,1,2,3,4,5 | -39.4605,-37.9786,-36.9032,-36.1891,-41.1423,-41.8380 |
| HGVk1 | GLNO_LR_hz | 0,1,2,3,4,5 | 7.4753,7.4790,5.6954,5.8196,9.3528,5.6122 |
| HGVk1 | PEN_LR_hz | 0,1,2,3,4,5 | 1.4230,1.4109,1.4682,1.4926,2.7067,1.2248 |
| HGVk1 | DNa02_LR_hz | 0,1,2,3,4,5 | 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000 |

### 5.6 Provenance, invalid attempt and validation

Valid source: `2b3df26`; batch generated before submission and declaration frozen after main merge/push.
Freeze: `2026-09-15T21:12:44Z`; submission: `2026-09-15T21:13:05.390018+00:00`.
Frozen JSON SHA-256: `1e220af99ae700c9617bb0ecb538d62d78ba128405820799fdfc733147e214aa`.
Wrapper SHA-256: `4a65f69adb42cc72a170c7d75616e9c84fe130838bbd5d5cc1ec0d42239d3444`; manifest SHA-256: `50a3474d642e03b43bb6bc55f11775484da24ab75a972e3606fab51f7b84f25c`.
Both archived bytes match, all 48 unique identities and complete resolved LIF records match, both clients
report 24 completed / 0 failed and every console says `device cuda`. Metadata checks: zero problems.
Independent trace verifier: 48 runs, 972 metrics checked, 52 source files including loaded probes, zero issues.
Verifier SHA-256: `15c65c60d23fb8a895c834c648801b8167e14f99c416a23257751d647cdf1f76`; primary reducer SHA-256: `cb91cc196d223ea592cd3e780bd652ff7bd96f39f14fca717041a3599fe63786`.
The original cx8 attempt ran the same physical commands but had a class/abs mismatch in Astra's new
declaration. It remains invalid under `out/cx8/`, with its freeze unchanged, 144 metadata issues and
48 model-to-declaration issues in the independent verifier. No inference uses its numbers. All 48
replacement commands match after substituting only the output directory. See `instruments_review.md` 5.

All arms use the common diagnostic protocol, including compass adaptation held at zero and EPG pulse /
background; S is the raw-preset comparator within that assay, not an unperturbed room. Shipped receptor
setting is sign/abs. MaleCNS CSR MD5 is `ef23cc27bea13be7f6a96f3c04fd3737`; the GLNO scratch graph is
`7a10d93ba2086f2c76bcdabdca79b4ec`. Original three cache file MD5s remain unchanged.

CPU suite at the submitted tree: 469 passed / 19 skipped, 220 subtests, including the original golden.
Closeout full CPU suite also passes: 469 passed / 19 skipped, 220 subtests, 166.40 s
(`out/compass7/cx8_closeout_cpu.log`); main and worktree cache hashes unchanged.
Astra self-review: compared the primary family to the freeze, checked all trace masks/means and loaded
sources, inspected every follow trace, and retained the small raw SD and rank/effect-rule caveats.
Independent skeptic pending (Fable, when accounts reset). Nothing adopted.

### 5.7 Outcome

The suite and odour room are conditional on demonstrated following and sign reversal; that condition fails.
The declared branch for primary 3 result and unavailable following therefore selects one GLNO-to-PEN
transfer diagnostic. The present data cannot uniquely assign the failure to a receptor, GLNO amplitude,
PEN recurrence, or confinement. A separate follow-up freeze will state its interventions and decisions.

## Report

```yaml
summary: |-
  The signed afferent reaches GLNO, but this round demonstrates no turn-following compass.
  V minus S raises GLNO L-R by +2.2183 Hz (result, Holm p 0.0130). No HGV run passes the predeclared
  50% turn-window confinement gate: primaries 1 and 2 are undetermined, with 0 versus 2 and 0 versus 1
  eligible runs. These are unavailable comparisons, not zero velocity or statistical nulls.

  PEN L-R HGV minus HG is null (+0.7813 Hz, z 0.4221), and DNa02 L-R is null (all values zero).
  The PEN-only hold gives no confined post-pulse frames in any seed; its contrast with HGV is null
  under the full rule despite Holm p 0.0130, because |z|=2.3613 is below the declared threshold of 3.
  The low and high k arms also have only one eligible run each. No arm meets the joint ledger
  survival/rate/width criteria, and nothing is adopted.

  This result uses 48 CUDA runs in cx8r, six per arm, with zero analysis problems. An independent
  CPU trace calculation reproduces 972 measurements and 52 source hashes. The first cx8 attempt
  is retained as invalid because Astra's frozen receptor rule was wrong; it is not used for inference.
  Primary 3's result, with no demonstrated follow, selects the single predeclared GLNO-to-PEN transfer
  follow-up. The suite and odour-room milestone are not unlocked. Independent skeptic pending
  (Fable, when accounts reset).
key_claims:
- The signed afferent changes GLNO L-R in V; the absolute primary effect is +2.2183 Hz.
- HGV has zero eligible follow runs. Both follow contrasts are undetermined; no eligible sign reversal is demonstrated.
- PEN and DNa02 L-R contrasts are null under the declared rule.
- PEN-only hold removes measured confinement, but its primary contrast fails the z requirement and remains null.
- The k sweep selects no gain; every arm fails the joint ledger compass criteria.
validation:
- 48 valid CUDA runs, six per arm, zero primary-analysis and independent-verifier problems.
- 972 saved-trace measurements and 52 source hashes checked on CPU.
- Submitted-tree CPU suite 469 passed, 19 skipped, 220 subtests; golden and cache unchanged.
- Initial cx8 invalidated for a declaration error, retained separately without inference.
- Independent skeptic pending (Fable, when accounts reset).
recommendations:
- Nothing adopted; raw remains the default.
- Execute the single predeclared GLNO-to-PEN diagnostic; do not run the conditional suite/room milestone yet.
open_questions:
- Whether a directly imposed GLNO side signal transfers into PEN under the held operating state.
- A physiological receptor/kinetic account and a stable following bump remain unresolved.
```
