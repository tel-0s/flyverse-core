# Compass round 7: the PS196_b velocity route (predeclaration)

Written 2026-09-15 before any job was submitted; the results section is empty until the batch returns. Runs on the house
cluster (`--target house`), batch prefix `cx8`. Conditional on rounds 6B (`cx7`) and 4d (`vncd7`) not already answering
the question; if 6B produces a bump that follows the fly's own turn, this round is skipped and the file says so.

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

## 5. Results

Initial attempt `cx8`, source `a20d0ed`, is **invalidated before inference**: the added frozen LIF record
incorrectly declared `sign/class`, whereas `--receptor-model shipped` actually runs the shipped `sign/abs`.
This reviewer-introduced declaration error is recorded in `instruments_review.md` section 5. Original records
and declaration remain unchanged under `out/cx8/`; no statistical or functional decision uses that attempt.
The replacement `cx8r` repeats all 48 original arm/seed commands and the six contrasts, with the declaration
corrected to the actual shipped rule and frozen again before submission. No gain, gate or outcome rule changes.

(valid results pending `cx8r`)
