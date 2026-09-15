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

## 2. Arms (6 seeds each, `cx_wedge` free-turn protocol as in 6A; the fly turns itself at 86-113 deg/s)

| arm | instruments | what it tests |
|---|---|---|
| S | none (`raw`) | reference |
| V | `sided_turn_afferent` only | does a signed PS196_b input reach GLNO / PEN at all at shipped gains (predicted: GLNO L-R moves, no bump: the ring is silent) |
| HG | `ring_dc_hold` + `glno_sign` (= 6A's H3G) | the near-bump without a velocity input (replicates 6A) |
| HGV | `ring_dc_hold` + `glno_sign` + `sided_turn_afferent` | **the arm**: the bump should move with the turn |
| HGV- | HGV with the afferent sign flipped | the sign control: the bump should move the other way, or the effect is not the afferent |

The afferent instrument: Poisson `poisson_hz` on AN07B037_a/_b (and CB0675 / GNG580 / PS047_b in a second variant only
if V is null on the first), rate = `k * max(0, +-yaw_deg_s)` on the side the connectome's contralateral routing
implies, `k` swept over {0.25, 0.5, 1.0} Hz per deg/s as **labelled levels, unverified** (no PS196_b recording exists;
search named in `PRESETS_SPEC.md` section 3). One level (0.5) is the predeclared primary; the other two are
descriptive.

## 3. Predeclared measures and family

Primary (Holm, m = 5, 6 v 6 exact U, floor 0.0022 -- satisfiable):

1. `bump_follow_wedges_per_s` HGV vs HG: the bump's mean angular velocity over the turn, sign-matched to the turn
   (ideal 4.0 w/s at 90 deg/s; 6A measured 0.00 +- 0.01 in every arm).
2. `bump_follow_wedges_per_s` HGV vs HGV-: sign flip.
3. `GLNO_LR_hz` V vs S: the afferent reaches GLNO with a side.
4. `PEN_LR_hz` HGV vs HG: the side reaches PEN.
5. `DNa02_LR_hz` HGV vs HG: anything gets back down.

Descriptive (no verdict): bump survival / rate / width per arm as in 6A, the k-sweep, per-type ring rates (6B's
recorded groups if merged by then), PS196_b's own L-R at each k.

Verdict vocabulary result / null / underpowered / undetermined per INTERP 10.4. Per-seed scatter emitted by the
analysis script to `out/cx8/analysis/per_seed.csv` and pasted (rule 28). Nothing adopted into `raw` under any outcome;
a `result` on 1 + 2 makes `sided_turn_afferent` + `ring_dc_hold` + `glno_sign` the first `instrumented` list and sends
it to the suite (PRESETS_SPEC section 2 item 5) before any room run.

## 4. What each outcome means

- **1 and 2 result:** the velocity route works once it is signed; the compass is a physiology gap (a signed afferent,
  the DC receptors), not an anatomy gap. Next: the suite under `instrumented`, then the odour room with DNa02 L / R
  as the readout.
- **3 result, 1 null:** the afferent reaches GLNO but GLNO does not move the ring -- the GLNO -> PEN transfer (sign,
  receptor kinetics) is the block; back to `glno_relabel.md` with a rate measurement instead of a sign.
- **3 null:** the afferents do not reach PS196_b at the swept gains; the route is not AN07B037 (try the second
  variant) or PS196_b's threshold is above anything an afferent supplies.
- **Anything with HGV- not flipped:** the effect is not the afferent; the round is undetermined on 1.

## 5. Results

(empty until `cx8` returns)
