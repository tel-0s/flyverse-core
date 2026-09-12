# The compass under sensory input: the ring attractor's operating points in the walking fly

Script: `scripts/probe_compass_room.py` (GPU: `--gE --gD --program --seed --seconds --receptor-model --out`; CPU:
`--report --files ... --table ...`). Batch generator: `scripts/cxroom_batch.sh`; console log `out/cxroom_cluster.log`.
Data: `out/cxroom/cxroom_<cond>.{json,npz,txt}` (per condition: config, per-fly metrics, per-condition summary; the
NPZ holds the per-frame per-fly record -- 46 EPG rates in `cx_wedge`'s wedge order, PEN / Delta7 / PFL3 / DNa02 L and
R, PEG, ER/ExR, GLNO, PFN, hDelta and rest-of-brain means, heading, yaw rate, speed, position, airborne, program mode
and steering error); tables `out/cxroom/compass_room.md`, `compass_room_flies.csv`. Replication (skeptic's batch):
`out/cxvfy/rep_*.{json,npz,txt}`, `out/cxvfy/verify_table.{md,csv}`, log `out/cxvfy_cluster.log`. Combined table over
both: `out/cxroom/compass_room_all.{md,csv}` (the command is in section 3).

## 1. Question

Every ring-attractor result on record (`docs/audits/cx_wedge.md`, `cx_glno.md`) was measured with the LIF alone: no
senses, a held 10 Hz Poisson background on the 46 EPG, a 4-wedge pulse, the rest of the brain at 0.03 Hz. The
GLNO-sign-robust operating points from that work are gE 2 / gD 15, gE 2.25 / gD 25 and gE 2.5 / gD 25 (6/6 seeds
confined in both GLNO conditions; `cx_glno.md` section 5). This audit puts two of them (gE 2 / gD 15, gE 2.5 / gD 25)
into the room -- full senses, the fenced single-apple table, the fly walking -- and asks:

(a) does a bump form spontaneously and persist, or does the sensory-driven brain (the ER ring neurons' visual input,
    the ExR feedback) destroy or pin it;
(b) does the bump centre track the fly's heading in any way (the model has no verified rotation input to PEN, so the
    expectation is 'no' -- to be stated precisely);
(c) does a bump produce a steering asymmetry (PFL3 L - R, DNa02 L - R) as a function of its position.

The compass gains are an experiment's gains, applied as `LIFParams` overrides exactly as `cx_glno.py` applies them
(`adapt_by_type {'^(EPG|PEN|PEG|Delta7)': 0}`; `type_path_gain = DEFAULT_TYPE_PATH_GAIN + EPG <-> PEN x gE, EPG <-> PEG
x gE, Delta7 -> EPG x gD`, Delta7 -> PEN x1, ER/ExR x1). Nothing in the default model changes; the control is the
shipped default (no compass gains, uniform adaptation), under which the ring is silent.

## 2. Protocol

* `BatchSim`, 16 flies per job (environment seeds 16 s .. 16 s + 15 for brain seed s), fenced single-apple table,
  start (-0.15, 0.15, table top), initial heading uniform per environment seed, energy 0.9, wind 0.3 m/s from 180 deg
  (the `batch_sustain.py` room), 60 s = 6,000 frames of 10 ms (LIF dt 0.5 ms, optic dt 1 ms).
* Programs: `none` (the plain model: baseline walking, DN drive, optomotor; PFL3 and DNa02 read out passively) and
  `cx` (`flyverse/cx.py CompassSteering`: the stand-in that reads the wind DNs and the LH odour signal and stimulates
  PFL3 L / R and DNp09 every frame -- so under `cx` PFL3 L - R is the program's steering error by construction and the
  bump's contribution has to be read with that error partialled out).
* Ring drive: none for the first 20 s (no background, no pulse: the ring receives only what the brain gives it), then
  at t = 20 s a 2 s pulse of +40 Hz on 4 contiguous wedges (`cx_wedge`'s driven width, two tiles = 90 deg), a
  different block per fly (row i starts at wedge 2 i mod 16: 8 tile positions, two flies each), then 38 s free. The
  0-20 s window answers 'spontaneous'; the 22-60 s window answers 'persists / destroyed / pinned'; the last 30 s
  (30-60 s) is the task's window.
* Backend: one `FlyBrain(batch=16)` on the Torch path in every condition (adapt_by_type disables the native LIF
  kernel, so the control is run on the same path: `cuda_kernels False`, cuSPARSE matmul, CUDA graphs, no event
  traversal); receptor model = the shipped default (`sign` / `abs`; 0 of 27,553 ring-core entries change under it,
  `cx_glno.md` section 2).
* Recorded per frame per fly (see the file list above). Metrics (`probe_compass_room.fly_metrics`, the same code in
  the job and in `--report`):
  - **bump per frame**: vector strength `vs` and centre wedge of the 46-cell EPG rate profile (as `cx_wedge`); the
    4-wedge block best centred on the centre; **confined** = `vs > 0.6` AND >= 8/11 of the block's cells above 22 Hz
    (8/11 = 0.727 of the block's 8-13 cells) AND <= 3 of the cells outside above 22 Hz -- `cx_glno`'s persistence rule
    transported to a block that follows the bump; bump rate = the block's mean; width = wedges above half the peak.
  - **(a)**: fraction of frames confined in the pre-pulse, post-pulse and 30-60 s windows; survival = last confined
    frame minus pulse end; drift = mean |centre step| per s, the mean and max |displacement over 1 s|, the circular
    sd of the centre, the distance of the centre from the pulsed block (mean, final, fraction within 2.5 wedges),
    and jumps (> 2 wedges within 0.5 s).
  - **(b)**: over the post-pulse confined frames, the Jammalamadaka-Sarma circular correlation of the centre angle
    with the heading; the Pearson r and slope of the bump's angular velocity (unwrapped centre, 0.1 s box) against
    the yaw rate (same box); |bump velocity| against |yaw|; and the circular sd of (centre - heading) against that
    of the centre alone -- if the bump were heading-anchored the difference would be constant.
  - **(c)**: over the post-pulse confined frames, PFL3 L - R and DNa02 L - R (and the yaw rate) fitted with a first
    harmonic of the bump angle in the ring coordinate (`y ~ b0 + b1 cos + b2 sin`, plus the `cx` program's steering
    error as a covariate): amplitude, phase, R^2. The ring coordinate is body-fixed anatomy, so the bump's wedge IS
    its position relative to the body axis up to a per-animal offset, which the amplitude does not depend on. Three
    versions: within fly (refused when the centre's circular sd is < 1 wedge -- the fit is then ill-conditioned),
    pooled over the condition's flies, and across flies (each fly's mean asymmetry against the tile it was pulsed at:
    16 flies at 8 tiles). Nulls: the within-fly fit repeated with the bump angle shuffled in 1 s blocks (95th
    percentile of 200 shuffles), and the control condition, whose across-fly fit is what 16 flies' mean asymmetries
    produce against the tile they were pulsed at when no bump is present.
* Batch: `cxroom-<id>`, 12 concurrent jobs = 2 operating points x 2 programs x seeds 0, 1 (8) + the control (no
  compass gains) x 2 programs x seeds 0, 1 (4); the task asked for the control 'x2', the two extra control jobs give
  the null its second seed.

## 3. What landed

Batch `cxroom-7f1fbb`: 12 jobs, 0 failed, 28.9 min, run dir `<cluster-fs>/neurome/runs/cxroom-7f1fbb/`
(`out/cxroom_cluster.log`), fetched to `out/cxroom/cxroom_{g2-15,g2.5-25,ctrl}_{none,cx}_s{0,1}.{json,npz,txt}`;
all 17 `.txt` report `device cuda (NVIDIA B200); torch 2.11.0+cu128`. The skeptic's replication batch `cxvfy-66b0ab` (6 jobs, 0 failed, 17.9 min,
`out/cxvfy_cluster.log`) added five more runs at `out/cxvfy/rep_*.{json,npz,txt}`: `rep_g2-15_none_s0` (a same-config
re-run of a landed job), `rep_g2-15_none_s2`, `rep_g2.5-25_none_s2`, `rep_ctrl_none_s2`, and
`rep_g2.25-25_none_s0` -- the third GLNO-sign-robust operating point (gE 2.25 / gD 25) that the original batch had
left out. Its table is `out/cxvfy/verify_table.{md,csv}`.

17 runs x 16 flies = 272 flies: 12 gain runs (11 distinct configurations; 192 flies) and 5 shipped-default control runs
(80 flies). The combined table used below is

```
PYTHONIOENCODING=utf-8 python scripts/probe_compass_room.py --report \
  --files "out/cxroom/cxroom_*.json" "out/cxvfy/rep_*.json" --table out/cxroom/compass_room_all
```

(`out/cxroom/compass_room_all.{md,csv}`; the 12-run original is `out/cxroom/compass_room.{md,csv}`). The CPU smoke test
(`out/cxroom_smoke/`, 2 flies x 0.3-0.4 s) predicted the mechanics correctly and is superseded by everything below.

## 4. Results

### (a) The senses do not destroy the bump; the shipped default never has one

Once the 2 s pulse lands, the bump is confined for the whole remaining 38 s in **192 / 192 gain-run flies** -- every fly
of every gain run, both operating points plus gE 2.25 / gD 25, both programs, seeds 0-2, originals and replications.
`survival_s` is 38.0 s in all 192 (the run's maximum); `frac_confined_post` is 1.00 to 2 dp in all 192 (per-fly minimum
0.997, i.e. at most 11 of 3,801 frames fall out of the rule and come back); `jump_frames_0p5s` is 0 in all 192; the
fraction of frames within 2.5 wedges of the pulsed block is 1.00 in 191 of 192 flies (the exception, `g2-15_none_s0`
row 2, is the fly that had already formed a spontaneous bump elsewhere before the pulse). Bump rate scales with the
gains and is reproducible to 1-2 Hz across seeds, programs and re-runs:

| operating point | runs | bump Hz post (per-run mean) | PEN L / R Hz | ER/ExR Hz | GLNO Hz | PFN Hz | hDelta Hz |
|---|---|---|---|---|---|---|---|
| gE 2 / gD 15 | 6 | 219-220 | 55-57 / 54-57 | 10.18-10.22 | 137-139 | 0.59-0.61 | 1.47-1.51 |
| gE 2.25 / gD 25 | 1 | 231 | 62 / 65 | 11.16 | 154 | 0.70 | 1.74 |
| gE 2.5 / gD 25 | 5 | 259-261 | 74-77 / 75-78 | 12.79-13.01 | 175-178 | 0.76-0.78 | 1.77-1.86 |
| shipped default (control) | 5 | 35 (never confined) | 0.0 / 0.0 | 0.12-0.14 | 0.0-0.1 | 0.00 | 0.00 |

The control is dead within 0.1 s of pulse end: `frac_confined_post` 0.00 and `survival_s` 0.0-0.1 s in **0 / 80** flies,
PEN 0.0 Hz, ER/ExR 0.12-0.14 Hz. So the room's sensory drive neither sustains the ring on its own nor destroys it when
the gains sustain it -- `cx_wedge.md`'s LIF-alone dichotomy survives full senses, a walking body and 38 s.

**Spontaneous formation, and it is not seed-reproducible.** With no ring drive at all in the first 20 s (no background,
no pulse), a bump forms and satisfies the confinement rule in 1-5 of 16 flies in **9 of the 12 gain runs**
(`frac_confined_pre` >= 0.05; three further flies show 1-2 isolated frames at 0.001-0.002 and are not counted),
never in any of the 5 control runs (0 / 80). The onsets are 3.4-18.9 s in. But the same configuration re-run gives a
different answer: gE 2 / gD 15, program `none`, seed 0 has 2 / 16 flies spontaneous in `cxroom_g2-15_none_s0` (rows 2
and 5, `frac_confined_pre` 0.819 and 0.233) and 0 / 16 in `rep_g2-15_none_s0` (its one flagged fly is a 2-frame
flicker). The two runs are the same command with the same seed; the difference is GPU run-to-run nondeterminism
(cuSPARSE / CUDA-graph reduction order), amplified by a bistable attractor near threshold. Spontaneous formation is
therefore a real property of the gain settings and **not** a per-fly fact that can be reproduced or counted on.

### (b) The bump does not track heading. It is pinned to a handful of attractor sites and frozen there

Two direct tests, both null, per-run means over the 12 gain runs:

| statistic | across gain runs | per fly |
|---|---|---|
| circular corr(bump centre, heading) | **-0.11 to +0.11** | -0.94 to +0.96, mean 0.000 (n 192) |
| Pearson r(bump angular velocity, yaw rate) | **-0.01 to +0.03** | -0.08 to +0.27, mean 0.007 |
| slope, bump velocity per yaw | -0.01 to +0.02 | -0.06 to +0.17 |
| \|bump velocity\| vs \|yaw\| (rad/s) | 0.03-0.05 vs 0.03-0.26 | |

The circular sd of (centre - heading), 0.10-6.65 wedges per fly, is 10-100x the circular sd of the centre alone
(0.006-0.79): all of the variance in the difference is the heading's.

The reason is stronger than "no tracking": **the bump goes to one of a few fixed places and stops.** Taking each fly's
circular-mean centre over the last 30 s of confined frames, the 16 flies of a run land on five or six values, and the
same values recur across seeds, across programs and across the replications:

| operating point | attractor sites (bump centre, wedges) | runs agreeing |
|---|---|---|
| gE 2 / gD 15 | **1.52, 3.93, 8.38, 10.58, 13.24** | 6 / 6 (cx s0, cx s1, none s0, none s1, rep none s0, rep none s2); 96 / 96 flies |
| gE 2.25 / gD 25 | 1.59, 3.98, 8.36, 10.61, 12.97 | 1 / 1; 16 / 16 flies |
| gE 2.5 / gD 25 | **1.58, 3.72, 8.31, 11.07, 13.03, 15.50** | 5 / 5; 71 of 80 flies (the other nine at 3.75-4.04 -- six just off the 3.72 well -- plus 6.70, 9.46, 10.40) |

Eight pulsed tiles map onto five or six sites, so different tiles collapse onto the same site (at gE 2 / gD 15 the
blocks starting at wedge 0 and at wedge 14 both end at 1.52) and the same tile occasionally splits between two sites
across flies (`g2-15_none_s0`, block at wedge 4: one fly to 3.93, the other to 8.38). The distance
from the centre to the tile the fly was actually pulsed at is 0.51-1.03 wedges as a per-run mean and up to 2.88 wedges
for one fly -- the bump slides off the pulse to the nearest well and stays.

Once there it is **frozen**: the circular sd of the centre over 38 s is 0.02-0.09 wedges as a per-run mean
(0.006-0.79 per fly), and there are 0 jumps (> 2 wedges in 0.5 s) in 192 / 192 flies.

**Relabel the drift figure.** `drift_abs_wedges_per_s` reads 0.50-0.59 wedges/s across the gain runs, which looks like
motion and is not: it is the mean per-frame \|centre step\| rescaled to a second (100 frames), i.e. frame-to-frame
jitter of the circular mean of 46 Poisson-ish rates. The same runs' mean \|displacement over 1 s\|
(`drift_1s_abs_wedges`) is **0.006-0.065 wedges**, a factor of 10-80 smaller, and the circular sd over the whole 38 s is
0.02-0.09. Drift is ~0.01 wedges/s; 0.50-0.59 is jitter and should be reported as such.

### (c) A bump does produce a PFL3 asymmetry -- but it is a fixed anatomical gradient the gains scale, not create

The pooled per-frame fit in the report is pseudo-replicated: it fits n ~ 60,800 frames per condition (e.g. 60,793 for
`cxroom_g2-15_none_s0`) whose bump centres are, by (b), 16 essentially constant values. Its r^2 0.62-0.74 measures how
well 5-6 points lie on a cosine, with 3,800 copies of each point. The honest unit is the **bump position**, and since
duplicate tile pairs land on the same site, **n = 8**: per tile, the two flies' mean PFL3 L - R and their circular-mean
centre, then `y ~ b0 + b1 cos + b2 sin` with an F-test on 2 and 5 df.

| run (program `none`) | PFL3 L-R amp (Hz) | phase (wedges) | r^2 | p |
|---|---|---|---|---|
| rep gE 2 / gD 15 s0 | 2.44 | 12.90 | 0.78 | 0.022 |
| rep gE 2 / gD 15 s2 | 2.09 | 12.82 | 0.72 | 0.042 |
| rep gE 2.25 / gD 25 s0 | 2.76 | 12.81 | 0.80 | 0.018 |
| rep gE 2.5 / gD 25 s2 | 3.30 | 12.30 | 0.76 | 0.029 |
| (the four original `none` gain runs) | 2.17-2.90 | 11.96-12.89 | 0.69-0.78 | 0.022-0.053 |
| control, pooled over the 3 `none` control runs | **0.21** | 12.41 | 0.79 | 0.020 |

The control line is the result. With no bump at all -- PEN 0.0 Hz, ER/ExR 0.12 Hz, the ring silent -- the 16 control
flies' mean PFL3 L - R still lies on a first harmonic of the tile they were pulsed at, **at the same phase** (12.4
against 12.3-12.9) and at **one tenth to one sixteenth** of the amplitude. The bump -> PFL3 map is therefore a fixed
anatomical gradient that exists in the connectome without the compass; the gains scale it by 10-16x, they do not create it.
(Per control run the same fit gives 0.09 / 0.29 / 0.25 Hz at phases 12.9 / 11.7 / 13.1; over all post-pulse frames rather
than the controls' 66-77 spurious confined frames it collapses to 0.0003-0.0010 Hz, same phases.)

Per-wedge, the gradient is reproducible to 0.03 Hz and tracks gain monotonically (pooled confined frames, PFL3 L - R in
Hz, `by centre wedge` rows of the report):

| wedge | gE 2 / gD 15 (4 `none` runs) | gE 2.25 / gD 25 | gE 2.5 / gD 25 (3 `none` runs) |
|---|---|---|---|
| 1 | -1.24 / -1.24 / -1.25 / -1.23 | -1.48 | -1.79 / -1.80 / -1.83 |
| 8 | -0.58 / -0.61 / -0.58 / -0.59 | -0.79 | -0.94 / -0.97 / -0.96 |
| 13 | **+3.67 / +3.69 / +3.66 / +3.67** | (+4.17) | +4.75 / +4.76 / +4.74 |

The gE 2.25 / gD 25 entry in the wedge-13 row is bracketed because that run's attractor sits at 12.97, so its wedge-13
bin is a 1,388-frame jitter tail of its wedge-12 bin (+4.55, 6,214 frames); wedges 1 and 8 are its own attractor sites
(15,195 and 11,397 frames) and are comparable. Each other cell is 7,000-19,000 pooled frames.

**Under program `cx` the gains add nothing detectable.** With the steering error partialled out, the pooled fit's r^2 is
**0.000, 0.004, 0.005, 0.007** in the four `cx` gain runs against 0.617-0.736 in the `none` gain runs: the program writes
PFL3 L - R every frame (its within-fly temporal sd is 12-19 Hz as a per-run mean, 3.1-34.6 Hz per fly, against 0.60-0.73
and 0.47-1.93 without the program), and the bump's 2-3 Hz contribution is not recoverable underneath it.

**The asymmetry does not reach the motor output.** DNa02 L - R on the same n = 8 fit, over the four replication runs, is
0.08-0.24 Hz (p 0.004-0.128) -- one cell per side, and the within-fly temporal sd of that same quantity is ~1 Hz
(0.97-1.33 across the `none` gain runs), so the position signal is a fifth of the moment-to-moment noise on the cell that
carries it. The yaw-rate first harmonic over the same four runs is 0.006-0.021 rad/s, which is not a result either: the
control's own yaw harmonic is 0.011 rad/s (pooled over 5 control
runs) on flies with no bump. And the pathway the question named sits at background: **PFN 0.59-0.78 Hz and hDelta
1.47-1.86 Hz** in every gain run (0.00 in the controls), against a rest-of-brain mean of 0.50-0.61 Hz in the same runs,
a ring at 219-261 Hz and PFL3 itself at 1.6-7.1 Hz per side. A stage running at 0.6-1.9 Hz cannot be what carries a
2-3 Hz L - R modulation into PFL3. **PFN -> hDelta -> PFL3 is answered: not in this model** -- the gradient in (c) is
carried by EPG's direct projection onto PFL3.

**One unreported asymmetry, and it points the wrong way.** PEN's *output* L - R is large and locked to the bump's wedge:
per-fly means span -20.1 to +26.5 Hz at gE 2 / gD 15, -23.6 to +20.4 at 2.25 / 25 and -35.5 to +29.6 at 2.5 / 25, against
-0.01 to +0.02 Hz in the controls (whose per-frame band is -2.2 to +5.0 Hz of pure noise). But it is not a rotation
signal: on the n = 8 fit its first harmonic explains r^2 0.03-0.38 in the replication runs (0.003-0.79 across all 12),
i.e. it is a step function of which wedge the bump occupies rather than a smooth sinusoid of bump angle. This is an
**output of position, not an input of velocity** -- the same conclusion `cx_shift.md` reaches from the other side.

## 5. Statistics and what they can bear

* **Unit of replication.** The pooled per-frame fits (n ~ 60,800) are pseudo-replication: by (b) each fly contributes
  one constant centre, so the effective n is the number of distinct bump positions, 5-6, or 8 if tile pairs are kept
  apart. Every headline in (c) is quoted at n = 8 with an F-test; the pooled r^2 values appear only as the `cx`-vs-`none`
  contrast, where they are compared to each other.
* **Seeds.** Two brain seeds (0, 1) per configuration in the original batch; the replication adds seed 2 for three of
  them (gE 2/15 `none`, gE 2.5/25 `none`, control `none`), a new operating point at seed 0 (gE 2.25/25 `none`) and one
  same-seed re-run. The `cx` conditions and gE 2 / gD 15 at seed 1 have two seeds only. The 16 flies of a run are 16
  environment seeds sharing one brain seed, not 16 independent brains. Run-to-run GPU nondeterminism is
  large enough to flip a per-fly spontaneous-formation outcome (a), so any per-fly count in this audit is an estimate of
  a rate, not a reproducible fact.
* **The receptor model is not inert here.** The ring core is untouched (0 of 27,553 entries change under `sign`/`abs`),
  which is why `cx_glno.md` could ignore it -- but the *sensory* half of this audit is not the ring core: **1,487 of the
  49,483 ER/ExR input entries change under `abs`** (`cx_glno.md` section 2; pre ExR5 438, AOTU046 394, WED035 110,
  PLP046 76, ExR6 45 ...), and in the room those ring neurons run at 10.2-13.0 Hz under the gains against 0.12 Hz in the
  control. The bump results here are therefore conditional on the shipped `sign`/`abs` default in a way the LIF-alone
  results were not.
* **Verdict.** The Opus skeptic's pass found this audit **"mostly sound"**; its replication batch `cxvfy-66b0ab`
  (5 runs, `out/cxvfy/verify_table.md`) reproduced (a) and (c) at a third seed, and filled in the missing gE 2.25 / gD 25
  operating point, which behaves like the other two on every measure (bump 231 Hz, confined 16/16, five attractor sites,
  PFL3 amp 2.76 Hz at phase 12.8). The corrections it produced are folded in above: the drift relabelling, the
  pseudo-replication of the pooled fit, and the non-reproducibility of the spontaneous-formation counts.

## 6. What this means for the compass

1. **The compass gains stay experiment overrides.** They do exactly what `cx_wedge.md` and `cx_glno.md` said, now with
   senses and a body: a pulsed bump at 219-261 Hz that survives 38 s in 192/192 flies where the shipped default has
   none. Nothing here argues for making them a default -- they are still a hand-set gain of 2-2.5 on EPG <-> PEN and
   15-25 on Delta7 -> EPG with compass adaptation zeroed, chosen to make a bump, and the bump they make is inert.
2. **The 'compass module as a stop-gap for the plain fly's missing turning' option is closed.** That option assumed a
   bump that (i) tracks heading and (ii) steers. Neither holds: circular corr(centre, heading) -0.11 to +0.11,
   r(bump velocity, yaw) -0.01 to +0.03, and the bump is pinned to 5-6 sites and frozen to within 0.02-0.09 wedges over
   38 s. What the gains buy for steering is a 2-3 Hz PFL3 L - R gradient that is the same shape at a tenth to a
   sixteenth of the size with no bump present, and 0.08-0.24 Hz on DNa02 under ~1 Hz of its own noise. There is nothing
   to steer with.
3. **The missing piece is an input, and it is the same missing piece as `cx_shift.md`'s.** The ring runs, the bump
   persists, the read-out gradient exists -- what does not exist is anything that moves the bump: the model has no
   verified rotation input to PEN (`cx_shift.md` section 1: GLNO is PEN's only nodulus input at 19.4 % and is sign 0;
   nothing else reaches PEN), and here the bump duly ignores a walking fly's yaw exactly as it ignored an imposed 90 deg/s
   visual rotation there. The next question for the compass is that input -- specifically the efferent route
   (PS196_b / LAL -> GLNO), which neither audit has tested -- not more operating points.
