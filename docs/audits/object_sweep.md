# Small-object sweep: the decisive LC11 / LC10 test (receptor integration, round 2)

Script: `scripts/probe_object_sweep.py` (new; the NOTES session-9 'relative motion' protocol that round 1 never ran).
Runs: cluster batch `obj-sweep-035048` (10 jobs, all `device cuda`, NVIDIA B200, torch 2.11.0+cu128, native backend
`--cuda-kernels --event-driven --cuda-sparse warp`, no CUDA graphs; 3.1 min wall for the batch) plus the radiance
diagnostic `obj-diag` (section 5). Raw outputs: `out/obj/obj_{off,class,abs,rect_off,rect_class}_s{0,1}.{json,txt}`,
aggregated by `out/obj/obj_aggregate.py` into `out/obj/obj_aggregate.md`; structural coverage by `out/obj/obj_coverage.py`
into `out/obj/obj_coverage.json` (both scratch scripts, CPU, kept with the outputs; `out/` is git-ignored).
Connectome: the adopted cache with `TYPE_NT_OVERRIDE` applied (the cluster's shared cache lacks it, so every job ran with
`--cache-dir $CLUSTER_RUNS/ntov-r2-e7706e/cache_override`, which equals the local `cache/` -- same NT counts,
nnz 25,578,600, sum W 25,825,116, sum |W| 121,460,584). Receptor tables: the round-2 rebuild (`flyverse/data/receptors_by_type.csv`).

## 1. Protocol and pass criterion

* Fly pinned (re-placed every 10 ms frame) on the fenced single-apple table with the apple moved out of the scene, at
  (-0.20, 0.10, 0.75) facing -y; wind off. Above the table's horizon the fly sees only the plain wall at y = -2 (the door
  on the +x wall begins 57 deg to its left, outside the sweep). The eye is 1.2 mm above the table top.
* Ball: 1 cm diameter, material `black` (reflectance 0.01-0.02), resting on the table (centre 3.8 mm above the eye,
  elevation +4.3 deg at azimuth 0), 5 cm ahead of the eye, sweeping +-6 cm laterally (= 12 cm; +-50 deg of azimuth,
  ~46 deg/s at the centre, ~11 deg across) as a triangle wave with 3 s per one-way sweep: left -> right, right -> left,
  left -> right, right -> left over the 12 s window. Control: the identical timeline with the ball parked at (9, 9, 9).
* 3 s settle with the ball hidden, then 12 s scored (1,200 frames); the two conditions run in separate `room_demo.Sim`
  instances with the same seed. Seeds 0 and 1 (the scene does not depend on the seed; seeds sample the brain RNG and the
  native backend's run-to-run nondeterminism).
* Recorded per cell over the window: for LC11, LC10a, LC10b, LC16, LPLC2, LC4 the optic-lobe drive they receive
  (`brain.drive`, mV, per 10 ms frame) and their spikes (`brain.spike_counts`); for the optic rate units T2, T3, Tm5Y,
  TmY21, TmY13, TmY5a and the medulla reference Mi4 / Tm3 / Mi1 the deviation from the operating point
  (`optic.rates() - r0`, rate units; +-0.5 is the full range at the default baseline 0.5).
* Modes: `off` (NT_SIGN), `sign` with net rule `class`, `sign` with `abs`; then `--rectify-t2t3` (T2 and T3 baseline 0 in
  `optic.DEFAULT_BASELINE_BY_TYPE`, ReLU units as T4 / T5) for `off` and `sign-class`. 2 seeds each = 10 runs.
* Retina geometry (local, `retina.build_retina`): 1,466 columns, median inter-column angle 3.95 deg; 52 columns lie in
  the band the ball crosses (|az| <= 56 deg, el -2..11 deg); the ball covers ~3 columns at any moment.

**Pass criterion** (as set for this round): LC11 or LC10a shows a peak per-cell drive > 7 mV or a cell firing > 1 Hz
**with the ball and not without**. The script evaluates it per run (`verdict` in each JSON): `drive_pass` = peak per-cell
per-frame drive > 7 mV with the ball and <= 7 mV without; `rate_pass` = max cell rate over the window > 1 Hz with and
<= 1 Hz without.

## 2. Verdict: FAIL in 16 of 17 runs of the identical protocol (the per-run verdict is a coin flip; see section 7)

| mode | seed | LC11 peak drive ball / none (mV) | LC11 max cell Hz ball / none | LC10a peak drive | LC10a max cell Hz | verdict |
|---|---|---|---|---|---|---|
| off | 0 | 10.43 / 11.02 | 0.25 / 0.08 | 12.13 / 10.49 | 0.92 / 0.83 | FAIL |
| off | 1 | 10.40 / 9.58 | 0.25 / 0.17 | 11.31 / 10.98 | 0.83 / 0.92 | FAIL |
| sign-class | 0 | 10.64 / 9.87 | 1.00 / 0.67 | 10.03 / 11.28 | 0.50 / 0.58 | FAIL |
| sign-class | 1 | 10.25 / 10.35 | 0.83 / 0.67 | 10.16 / 10.64 | 0.42 / 0.58 | FAIL |
| sign-abs | 0 | 9.13 / 10.59 | 0.08 / 0.17 | 12.63 / 10.80 | 0.92 / 0.92 | FAIL |
| sign-abs | 1 | 9.06 / 9.19 | 0.00 / 0.00 | 11.33 / 12.53 | 0.75 / 0.58 | FAIL |
| rect off | 0 | 9.43 / 8.19 | 0.00 / 0.00 | 12.14 / 13.83 | 0.92 / 0.67 | FAIL |
| rect off | 1 | 9.51 / 8.00 | 0.08 / 0.00 | 11.38 / 10.98 | 1.00 / 1.17 | FAIL |
| rect sign-class | 0 | 9.62 / 9.70 | 0.92 / 0.92 | 10.32 / 10.10 | 0.42 / 0.50 | FAIL |
| rect sign-class | 1 | 9.87 / 9.89 | 0.67 / 0.83 | 10.07 / 10.18 | 0.33 / 0.42 | FAIL |

Reading: the per-frame peak drive exceeds 7 mV in **both** conditions in every run (LC11 9.1-10.6 mV with the ball,
8.0-11.0 without; LC10a 10.0-12.6 vs 10.1-13.8). Those peaks are the model's own frame-to-frame drive fluctuation with
no stimulus at all (0.012-0.425 % of LC11 cell-frames exceed +7 mV over both conditions and all ten runs; the per-frame minimum is -7.56 to -11.49 mV; the next figure was
-11.3 mV; the time of the peak frame is unrelated to the sweep: 0.1-11.7 s), so the criterion's "and not without"
clause fails everywhere, and no cell of either type crosses 1 Hz with the ball while staying under 1 Hz without
(LC11 max cell 0-1.0 Hz with, 0-0.92 without; LC10a 0.33-1.0 vs 0.42-1.17). The ball adds nothing that the noise
floor does not already contain.

## 3. Per-type table (mean of the two seeds; ball / none)

Spiking targets. Columns: time-mean drive of the population; best cell's time-mean drive; best cell's (ball - none)
time-mean; peak per-cell per-frame drive; peak of the 100 ms-smoothed per-cell drive; sweep-locked tuning = the best
cell's (max - min) over 24 bins of the ball's lateral position (0.5 cm), the same bins applied to the no-ball timeline
(a pure-noise reference); peak over (cell, bin) of the sweep-locked ball - none difference; population mean rate; max
cell rate; cells over 1 Hz; ratios ball / none of the peak and of the tuning range. Full per-seed rows in
`out/obj/obj_aggregate.md`.

### off

| type | cells | drive mean | best cell mean | best diff | peak frame | peak 100 ms | tuning best | tuning diff peak | rate mean Hz | max cell Hz | cells > 1 Hz | ratio peak | ratio tuning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC11 | 143 | +0.08 / +0.10 | +0.24 / +0.25 | +0.04 | 10.42 / 10.30 | 7.22 / 6.58 | 5.39 / 5.11 | +2.87 | 0.006 / 0.001 | 0.25 / 0.13 | 0 / 0 | 1.01 | 1.06 |
| LC10a | 275 | -0.00 / +0.00 | +0.40 / +0.47 | +0.05 | 11.72 / 10.74 | 8.56 / 7.33 | 4.95 / 6.38 | +4.36 | 0.020 / 0.017 | 0.88 / 0.88 | 0 / 0 | 1.09 | 0.78 |
| LC10b | 95 | +0.39 / +0.39 | +0.85 / +0.87 | +0.08 | 16.31 / 17.09 | 13.62 / 13.57 | 11.46 / 12.60 | +7.15 | 1.611 / 1.613 | 7.25 / 7.33 | 49.5 / 47.5 | 0.95 | 0.91 |
| LC16 | 182 | -0.03 / -0.04 | +0.17 / +0.16 | +0.08 | 13.93 / 13.47 | 9.41 / 9.03 | 7.06 / 8.36 | +5.27 | 0.125 / 0.117 | 3.17 / 3.08 | 4.5 / 4.5 | 1.03 | 0.84 |
| LPLC2 | 185 | +0.48 / +0.48 | +1.77 / +1.76 | +0.32 | 22.99 / 21.90 | 9.67 / 10.99 | 4.45 / 5.23 | +2.99 | 0.124 / 0.114 | 2.58 / 2.67 | 6.5 / 5.0 | 1.05 | 0.85 |
| LC4 | 126 | -0.07 / -0.07 | +0.35 / +0.33 | +0.06 | 11.68 / 11.83 | 9.27 / 9.40 | 8.01 / 8.80 | +4.22 | 0.080 / 0.088 | 2.33 / 2.25 | 3.0 / 3.0 | 0.99 | 0.91 |

### sign-class

| type | cells | drive mean | best cell mean | best diff | peak frame | peak 100 ms | tuning best | tuning diff peak | rate mean Hz | max cell Hz | cells > 1 Hz | ratio peak | ratio tuning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC11 | 143 | +0.22 / +0.16 | +0.66 / +0.56 | +0.23 | 10.44 / 10.11 | 9.32 / 9.24 | 9.63 / 7.06 | +4.97 | 0.028 / 0.030 | 0.92 / 0.67 | 0 / 0 | 1.03 | 1.36 |
| LC10a | 275 | +0.01 / -0.00 | +0.53 / +0.54 | +0.16 | 10.10 / 10.96 | 8.91 / 8.77 | 7.72 / 7.30 | +4.83 | 0.016 / 0.025 | 0.46 / 0.58 | 0 / 0 | 0.92 | 1.06 |
| LC10b | 95 | +0.23 / +0.27 | +1.12 / +1.22 | +0.23 | 16.69 / 16.90 | 15.15 / 15.41 | 16.21 / 16.91 | +10.04 | 1.545 / 1.586 | 6.71 / 6.92 | 47.0 / 48.0 | 0.99 | 0.96 |
| LC16 | 182 | -0.11 / -0.19 | +0.44 / +0.27 | +0.39 | 17.05 / 17.33 | 15.02 / 16.23 | 18.14 / 15.47 | +10.53 | 0.932 / 0.952 | 4.54 / 4.75 | 58.5 / 62.0 | 0.98 | 1.17 |
| LPLC2 | 185 | +0.85 / +0.85 | +3.19 / +3.09 | +0.45 | 23.65 / 26.68 | 20.08 / 22.10 | 9.99 / 11.13 | +6.07 | 0.448 / 0.510 | 3.71 / 4.33 | 32.0 / 32.0 | 0.89 | 0.90 |
| LC4 | 126 | -0.52 / -0.57 | +0.10 / -0.03 | +0.21 | 10.59 / 11.42 | 9.64 / 9.81 | 9.51 / 9.19 | +6.50 | 0.039 / 0.033 | 0.92 / 0.92 | 0.5 / 0 | 0.93 | 1.03 |

### sign-abs

| type | cells | drive mean | best cell mean | best diff | peak frame | peak 100 ms | tuning best | tuning diff peak | rate mean Hz | max cell Hz | cells > 1 Hz | ratio peak | ratio tuning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC11 | 143 | +0.12 / +0.08 | +0.28 / +0.24 | +0.11 | 9.10 / 9.89 | 6.58 / 6.34 | 5.77 / 5.48 | +3.44 | 0.001 / 0.001 | 0.04 / 0.08 | 0 / 0 | 0.92 | 1.05 |
| LC10a | 275 | +0.01 / -0.00 | +0.49 / +0.46 | +0.10 | 11.98 / 11.66 | 7.59 / 8.06 | 6.98 / 8.01 | +4.89 | 0.015 / 0.017 | 0.83 / 0.75 | 0 / 0 | 1.03 | 0.87 |
| LC10b | 95 | +0.48 / +0.44 | +1.09 / +1.04 | +0.14 | 17.14 / 17.34 | 13.94 / 14.11 | 15.93 / 12.89 | +9.72 | 1.624 / 1.712 | 7.25 / 7.38 | 48.0 / 49.5 | 0.99 | 1.24 |
| LC16 | 182 | -0.07 / -0.04 | +0.18 / +0.19 | +0.04 | 14.11 / 14.22 | 10.46 / 9.19 | 8.93 / 8.48 | +8.13 | 0.128 / 0.114 | 3.00 / 3.04 | 6.5 / 5.0 | 0.99 | 1.05 |
| LPLC2 | 185 | +0.50 / +0.50 | +1.80 / +1.80 | +0.32 | 28.67 / 23.79 | 12.81 / 11.04 | 5.48 / 4.90 | +4.16 | 0.160 / 0.163 | 2.67 / 2.79 | 9.0 / 8.5 | 1.20 | 1.12 |
| LC4 | 126 | -0.09 / -0.09 | +0.30 / +0.35 | +0.07 | 12.41 / 11.88 | 9.77 / 9.78 | 8.89 / 8.80 | +5.42 | 0.092 / 0.088 | 2.46 / 2.29 | 3.0 / 3.0 | 1.04 | 1.01 |

### off, T2 / T3 rectified

| type | cells | drive mean | best cell mean | best diff | peak frame | peak 100 ms | tuning best | tuning diff peak | rate mean Hz | max cell Hz | cells > 1 Hz | ratio peak | ratio tuning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC11 | 143 | +0.20 / +0.16 | +0.62 / +0.62 | +0.11 | 9.47 / 8.09 | 6.01 / 5.44 | 5.80 / 4.51 | +4.09 | 0.000 / 0.000 | 0.04 / 0.00 | 0 / 0 | 1.17 | 1.29 |
| LC10a | 275 | -0.14 / -0.14 | +0.24 / +0.20 | +0.11 | 11.76 / 12.40 | 7.79 / 7.75 | 6.08 / 5.19 | +4.31 | 0.017 / 0.014 | 0.96 / 0.92 | 0 / 0.5 | 0.95 | 1.17 |
| LC10b | 95 | +0.44 / +0.34 | +0.97 / +0.81 | +0.32 | 16.15 / 15.97 | 13.21 / 13.17 | 16.04 / 10.21 | +9.40 | 1.475 / 1.585 | 6.88 / 7.21 | 41.5 / 45.5 | 1.01 | 1.57 |
| LC16 | 182 | -0.16 / -0.13 | +0.14 / +0.18 | +0.10 | 14.37 / 15.08 | 9.10 / 8.90 | 9.54 / 7.21 | +6.85 | 0.114 / 0.114 | 2.88 / 3.17 | 3.0 / 5.5 | 0.95 | 1.32 |
| LPLC2 | 185 | +0.49 / +0.49 | +1.98 / +1.97 | +0.38 | 24.01 / 26.06 | 10.48 / 10.36 | 5.19 / 4.81 | +4.21 | 0.197 / 0.195 | 2.79 / 3.33 | 12.0 / 13.5 | 0.92 | 1.08 |
| LC4 | 126 | +0.18 / +0.21 | +0.89 / +0.89 | +0.15 | 12.39 / 11.92 | 8.34 / 7.90 | 7.38 / 7.04 | +6.27 | 0.075 / 0.047 | 2.62 / 2.50 | 1.0 / 1.0 | 1.04 | 1.05 |

### sign-class, T2 / T3 rectified

| type | cells | drive mean | best cell mean | best diff | peak frame | peak 100 ms | tuning best | tuning diff peak | rate mean Hz | max cell Hz | cells > 1 Hz | ratio peak | ratio tuning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC11 | 143 | +0.36 / +0.36 | +0.87 / +0.90 | +0.16 | 9.75 / 9.80 | 9.03 / 9.08 | 10.29 / 10.75 | +3.14 | 0.023 / 0.025 | 0.79 / 0.88 | 0 / 0 | 0.99 | 0.96 |
| LC10a | 275 | -0.19 / -0.18 | +0.30 / +0.30 | +0.14 | 10.20 / 10.14 | 8.37 / 8.47 | 10.08 / 10.94 | +3.14 | 0.007 / 0.009 | 0.38 / 0.46 | 0 / 0 | 1.01 | 0.92 |
| LC10b | 95 | +0.08 / +0.14 | +1.07 / +1.04 | +0.16 | 16.55 / 16.08 | 14.19 / 14.51 | 16.94 / 16.72 | +7.65 | 1.443 / 1.466 | 5.75 / 5.83 | 44.0 / 45.5 | 1.03 | 1.01 |
| LC16 | 182 | -0.31 / -0.28 | +0.17 / +0.20 | +0.28 | 16.45 / 17.08 | 14.54 / 14.60 | 17.63 / 18.99 | +5.78 | 0.831 / 0.830 | 4.38 / 4.42 | 50.5 / 49.5 | 0.96 | 0.93 |
| LPLC2 | 185 | +0.76 / +0.77 | +2.97 / +3.02 | +0.20 | 22.83 / 22.37 | 19.70 / 20.16 | 12.38 / 12.44 | +4.33 | 0.393 / 0.391 | 3.96 / 4.08 | 27.0 / 27.0 | 1.02 | 1.00 |
| LC4 | 126 | -0.21 / -0.22 | +0.81 / +0.81 | +0.17 | 10.14 / 9.61 | 9.18 / 8.44 | 7.65 / 8.18 | +3.39 | 0.021 / 0.017 | 0.50 / 0.33 | 0 / 0 | 1.06 | 0.94 |

Optic rate units (deviation from the operating point, rate units; mean of the seeds; ball / none). `ratio abs` =
population mean |dev| ball / none; `diff abs best cell` = max over cells of the (ball - none) time-mean |dev|.

| mode | type | cells | mean abs dev | best cell mean abs dev | dev max / min (ball) | signed mean | diff abs mean | diff abs best cell | ratio abs |
|---|---|---|---|---|---|---|---|---|---|
| off | T2 | 1630 | 0.1460 / 0.1446 | 0.379 / 0.371 | +0.50 / -0.50 | +0.0046 / +0.0046 | +0.0014 | +0.0382 | 1.01 |
| off | T3 | 1940 | 0.0852 / 0.0837 | 0.273 / 0.262 | +0.50 / -0.50 | -0.0001 / -0.0000 | +0.0015 | +0.0239 | 1.02 |
| off | Tm5Y | 898 | 0.1637 / 0.1620 | 0.399 / 0.400 | +0.50 / -0.50 | +0.0018 / +0.0018 | +0.0017 | +0.0369 | 1.01 |
| off | TmY21 | 372 | 0.2008 / 0.1996 | 0.363 / 0.368 | +0.50 / -0.50 | -0.0067 / -0.0064 | +0.0012 | +0.0364 | 1.01 |
| off | TmY13 | 432 | 0.1534 / 0.1487 | 0.296 / 0.286 | +0.50 / -0.50 | -0.0076 / -0.0081 | +0.0047 | +0.0319 | 1.03 |
| off | TmY5a | 1364 | 0.1966 / 0.1931 | 0.408 / 0.405 | +0.50 / -0.50 | +0.0034 / +0.0034 | +0.0035 | +0.0350 | 1.02 |
| off | Mi4 | 1772 | 0.0547 / 0.0547 | 0.141 / 0.143 | +0.50 / -0.50 | +0.0020 / +0.0022 | -0.0000 | +0.0749 | 1.00 |
| off | Tm3 | 2054 | 0.0941 / 0.0913 | 0.305 / 0.305 | +0.50 / -0.50 | +0.0009 / +0.0008 | +0.0028 | +0.1021 | 1.03 |
| off | Mi1 | 1773 | 0.0384 / 0.0377 | 0.233 / 0.233 | +0.50 / -0.50 | +0.0017 / +0.0017 | +0.0008 | +0.0476 | 1.02 |
| sign-class | T2 | 1630 | 0.1416 / 0.1451 | 0.294 / 0.303 | +0.50 / -0.50 | +0.0050 / +0.0017 | -0.0034 | +0.0454 | 0.98 |
| sign-class | T3 | 1940 | 0.0915 / 0.0925 | 0.257 / 0.256 | +0.50 / -0.50 | -0.0018 / -0.0033 | -0.0010 | +0.0329 | 0.99 |
| sign-class | Tm5Y | 898 | 0.1741 / 0.1793 | 0.343 / 0.344 | +0.50 / -0.50 | -0.0003 / +0.0002 | -0.0053 | +0.0547 | 0.97 |
| sign-class | TmY21 | 372 | 0.2062 / 0.2106 | 0.401 / 0.391 | +0.50 / -0.50 | -0.0112 / -0.0119 | -0.0045 | +0.0430 | 0.98 |
| sign-class | TmY13 | 432 | 0.4264 / 0.4272 | 0.488 / 0.487 | +0.50 / -0.50 | -0.0187 / -0.0087 | -0.0008 | +0.0415 | 1.00 |
| sign-class | TmY5a | 1364 | 0.2341 / 0.2345 | 0.396 / 0.402 | +0.50 / -0.50 | +0.0012 / +0.0050 | -0.0004 | +0.0487 | 1.00 |
| sign-class | Mi4 | 1772 | 0.1389 / 0.1407 | 0.294 / 0.298 | +0.50 / -0.50 | +0.0122 / +0.0084 | -0.0018 | +0.0308 | 0.99 |
| sign-class | Tm3 | 2054 | 0.0790 / 0.0788 | 0.259 / 0.267 | +0.50 / -0.50 | -0.0032 / -0.0031 | +0.0002 | +0.1108 | 1.00 |
| sign-class | Mi1 | 1773 | 0.0462 / 0.0461 | 0.207 / 0.209 | +0.50 / -0.50 | +0.0015 / +0.0005 | +0.0002 | +0.0694 | 1.00 |
| sign-abs | T2 | 1630 | 0.1469 / 0.1448 | 0.369 / 0.379 | +0.50 / -0.50 | +0.0053 / +0.0045 | +0.0021 | +0.0442 | 1.01 |
| sign-abs | T3 | 1940 | 0.0840 / 0.0845 | 0.266 / 0.271 | +0.50 / -0.50 | -0.0001 / -0.0002 | -0.0005 | +0.0248 | 0.99 |
| sign-abs | Tm5Y | 898 | 0.1606 / 0.1615 | 0.393 / 0.396 | +0.50 / -0.50 | +0.0022 / +0.0020 | -0.0009 | +0.0505 | 0.99 |
| sign-abs | TmY21 | 372 | 0.1975 / 0.1970 | 0.378 / 0.377 | +0.50 / -0.50 | -0.0074 / -0.0073 | +0.0005 | +0.0360 | 1.00 |
| sign-abs | TmY13 | 432 | 0.1455 / 0.1479 | 0.288 / 0.290 | +0.50 / -0.50 | -0.0080 / -0.0076 | -0.0024 | +0.0189 | 0.98 |
| sign-abs | TmY5a | 1364 | 0.1918 / 0.1932 | 0.401 / 0.403 | +0.50 / -0.50 | +0.0058 / +0.0039 | -0.0014 | +0.0438 | 0.99 |
| sign-abs | Mi4 | 1772 | 0.0556 / 0.0558 | 0.141 / 0.146 | +0.46 / -0.41 | +0.0023 / +0.0017 | -0.0002 | +0.0467 | 1.00 |
| sign-abs | Tm3 | 2054 | 0.0950 / 0.0951 | 0.300 / 0.312 | +0.50 / -0.50 | +0.0008 / +0.0006 | -0.0001 | +0.1116 | 1.00 |
| sign-abs | Mi1 | 1773 | 0.0381 / 0.0380 | 0.235 / 0.237 | +0.50 / -0.50 | +0.0017 / +0.0015 | +0.0001 | +0.0736 | 1.00 |
| rect off | T2 | 1630 | 0.0488 / 0.0483 | 0.134 / 0.144 | +0.80 / +0.00 | +0.0488 / +0.0483 | +0.0005 | +0.0282 | 1.01 |
| rect off | T3 | 1940 | 0.0262 / 0.0273 | 0.099 / 0.102 | +0.58 / +0.00 | +0.0262 / +0.0273 | -0.0011 | +0.0127 | 0.96 |
| rect off | Tm5Y | 898 | 0.1620 / 0.1643 | 0.385 / 0.397 | +0.50 / -0.50 | -0.0010 / -0.0019 | -0.0023 | +0.0574 | 0.99 |
| rect off | TmY21 | 372 | 0.1939 / 0.1933 | 0.355 / 0.367 | +0.50 / -0.50 | -0.0158 / -0.0144 | +0.0006 | +0.0511 | 1.00 |
| rect off | TmY13 | 432 | 0.1531 / 0.1571 | 0.294 / 0.304 | +0.50 / -0.50 | -0.0121 / -0.0128 | -0.0040 | +0.0335 | 0.97 |
| rect off | TmY5a | 1364 | 0.1883 / 0.1941 | 0.394 / 0.411 | +0.50 / -0.50 | +0.0130 / +0.0095 | -0.0058 | +0.0598 | 0.97 |
| rect off | Mi4 | 1772 | 0.0543 / 0.0557 | 0.143 / 0.154 | +0.50 / -0.50 | +0.0023 / +0.0016 | -0.0014 | +0.0734 | 0.97 |
| rect off | Tm3 | 2054 | 0.0969 / 0.0968 | 0.313 / 0.334 | +0.50 / -0.50 | -0.0025 / -0.0028 | +0.0002 | +0.0929 | 1.00 |
| rect off | Mi1 | 1773 | 0.0366 / 0.0368 | 0.226 / 0.244 | +0.50 / -0.47 | +0.0020 / +0.0019 | -0.0002 | +0.0539 | 0.99 |
| rect sign-class | T2 | 1630 | 0.0520 / 0.0515 | 0.153 / 0.153 | +0.83 / +0.00 | +0.0520 / +0.0515 | +0.0004 | +0.0149 | 1.01 |
| rect sign-class | T3 | 1940 | 0.0317 / 0.0313 | 0.091 / 0.092 | +0.67 / +0.00 | +0.0317 / +0.0313 | +0.0005 | +0.0087 | 1.02 |
| rect sign-class | Tm5Y | 898 | 0.1677 / 0.1659 | 0.318 / 0.317 | +0.50 / -0.50 | -0.0058 / -0.0050 | +0.0017 | +0.0365 | 1.01 |
| rect sign-class | TmY21 | 372 | 0.1963 / 0.1948 | 0.396 / 0.394 | +0.50 / -0.50 | -0.0181 / -0.0178 | +0.0016 | +0.0403 | 1.01 |
| rect sign-class | TmY13 | 432 | 0.4239 / 0.4235 | 0.489 / 0.489 | +0.50 / -0.50 | -0.0152 / -0.0164 | +0.0004 | +0.0233 | 1.00 |
| rect sign-class | TmY5a | 1364 | 0.2258 / 0.2248 | 0.391 / 0.393 | +0.50 / -0.50 | +0.0032 / +0.0042 | +0.0010 | +0.0381 | 1.00 |
| rect sign-class | Mi4 | 1772 | 0.1379 / 0.1377 | 0.298 / 0.301 | +0.50 / -0.50 | +0.0103 / +0.0109 | +0.0002 | +0.0196 | 1.00 |
| rect sign-class | Tm3 | 2054 | 0.0762 / 0.0749 | 0.246 / 0.244 | +0.50 / -0.50 | -0.0053 / -0.0058 | +0.0013 | +0.1120 | 1.02 |
| rect sign-class | Mi1 | 1773 | 0.0445 / 0.0439 | 0.200 / 0.194 | +0.50 / -0.49 | +0.0015 / +0.0015 | +0.0006 | +0.0761 | 1.01 |

**Coverage of these tables.** The 15 measured types hold 13,241 cells = 7.92 % of the 167,106 MaleCNS cells and receive
6,567,846 |W| input synapses = 5.41 % of the 121,460,584 total (per type in section 4). The measurements are of the
model's activity, so they cover those cells fully in every mode; which of their inputs the receptor table decides is
the tier statement of section 4.

## 4. What the receptor model can and cannot change on this pathway (structural, CPU; `out/obj/obj_coverage.json`)

Per type: MaleCNS cells (fraction of 167,106), |W| input synapses (fraction of 121,460,584), the tier under which the
round-2 table decides the sign of those inputs (the sign model is applied to 100 % of a type's inputs at a single tier
here: `exact` or `fallback` = NT_SIGN), and the input synapses whose fast sign changes vs NT_SIGN under each net rule.
Global: `class` changes 257,578 of 25,578,600 entries, `abs` 63,380, `nonmda` 389,284; matched 7,654,565 entries
(29.93 %) / 31,627,236 |W| syn (26.04 %): exact 19.84 % / 17.14 %, fuzzy 5.72 / 4.70, class 2.69 / 3.00, alias 1.67 / 1.19,
fallback 69.08 / 73.96 (the coverage table each run prints).

| type | cells (% MaleCNS) | in |W| syn (% MaleCNS) | tier of the inputs | ACh / Glu / GABA / His input share | class: changed syn (flipped + silenced) | abs | nonmda |
|---|---|---|---|---|---|---|---|---|
| LC11 | 143 (0.09) | 342,941 (0.28) | fallback 100 % | 65.5 / 12.4 / 22.1 / 0 | 0 | 0 | 0 |
| LC10a | 275 (0.16) | 250,329 (0.21) | exact 100 % | 55.3 / 24.0 / 20.7 / 0 | 0 | 0 | 0 |
| LC10b | 95 (0.06) | 89,524 (0.07) | exact 100 % | 66.6 / 15.7 / 17.7 / 0 | 0 | 0 | 0 |
| LC16 | 182 (0.11) | 97,184 (0.08) | exact 100 % | 67.0 / 21.6 / 11.4 / 0 | 0 | 0 | 0 |
| LPLC2 | 185 (0.11) | 349,724 (0.29) | exact 100 % | 77.2 / 15.1 / 7.8 / 0 | 0 | 0 | 0 |
| LC4 | 126 (0.08) | 305,190 (0.25) | exact 100 % | 70.6 / 7.9 / 21.5 / 0 | 0 | 0 | 0 |
| T2 | 1,630 (0.98) | 914,280 (0.75) | exact 100 % | 49.0 / 15.1 / 35.9 / 0.08 | 712 (0 + 712 His) = 0.08 % | 712 | 712 |
| T3 | 1,940 (1.16) | 465,883 (0.38) | exact 100 % | 57.5 / 10.6 / 32.0 / 0 | 0 | 0 | 0 |
| Tm5Y | 898 (0.54) | 406,050 (0.33) | fallback 100 % | 48.8 / 25.1 / 26.0 / 0.08 | 0 | 0 | 0 |
| TmY21 | 372 (0.22) | 201,184 (0.17) | fallback 100 % | 54.6 / 30.2 / 15.3 / 0.01 | 0 | 0 | 0 |
| TmY13 | 432 (0.26) | 241,195 (0.20) | fallback 100 % | 41.5 / 30.3 / 28.2 / 0.01 | 0 | 0 | 0 |
| TmY5a | 1,364 (0.82) | 935,624 (0.77) | exact 100 % | 70.6 / 10.3 / 19.1 / 0.02 | 216 (0 + 216) = 0.02 % | 216 | 216 |
| Mi4 | 1,772 (1.06) | 535,891 (0.44) | exact 100 % | 50.3 / 36.3 / 9.7 / 3.7 | 214,348 (194,324 + 20,024) = 40.0 % | 20,024 (3.7 %) | 214,348 (40.0 %) |
| Tm3 | 2,054 (1.23) | 749,583 (0.62) | exact 100 % | 48.7 / 34.9 / 16.3 / 0.06 | 476 (0 + 476) = 0.06 % | 476 | 476 |
| Mi1 | 1,773 (1.06) | 683,264 (0.56) | exact 100 % | 30.2 / 37.2 / 30.7 / 1.9 | 13,013 (0 + 13,013 R8 His) = 1.9 % | 13,013 | 267,190 (254,177 + 13,013) = 39.1 % |

So under every net rule the receptor model leaves the inputs of LC11, LC10a/b, LC16, LPLC2, LC4 and T3 exactly as
NT_SIGN has them, and T2, TmY5a and Tm3 lose < 0.1 % (histamine entries silenced). LC11's inputs (T3 20.2 %, T2 10.8 %,
Tm6 7.2 %, T2a 7.2 %, Tm12 5.8 %), T3's (Mi1 22.3 %, Tm1 15.9 %, Tm3 8.0 %, Tm4 6.5 %, Pm5 5.4 %) and T2's (Tm2 11.4 %,
L5 8.4 %, Tm3 7.1 %, T2 4.7 %, Pm2a 4.2 %, Mi1 3.5 %) are unchanged; the only route by which `sign` reaches this
pathway is the medulla (Mi4 40 % of input changed under class / nonmda, 3.7 % under abs; Mi1 39 % under nonmda). That
is what the runs show: `sign-class` raises LC11's time-mean drive from +0.08 to +0.22 mV and its best cell's rate from
0.13 to 0.67-1.0 Hz **in both conditions alike** (Mi4's flipped input changes the medulla's operating point, not the
object signal), and `sign-abs`, which touches Mi4 by 3.7 % only, leaves LC11 where `off` has it (+0.12 / +0.08 mV).

## 5. Is the ball in the picture?  (radiance diagnostic, cluster job `obj-diag`)

`out/obj/obj_diag_off_s0.{json,txt}` (mode off, seed 0, 3 s window after 1 s settle, same geometry; the main batch's
own radiance rows are contaminated because `sim.step()` lets the body walk after its trace and the control trace was
taken from that moved pose -- fixed in the script by re-pinning before both traces, which this rerun uses). At four
moments of the first sweep the column radiance (summed over the four bands) with the ball is compared with a fresh
trace at the same pose with the ball hidden:

| t (s) | ball offset (m) | expected azimuth (deg) | columns changed > 5 % (of 1,466) | darkened > 50 % | darkest ratio | azimuth of changed columns (deg) | elevation (deg) |
|---|---|---|---|---|---|---|---|
| 0.00 | +0.060 (left) | +50.2 | 5 | 2 | 0.020 | 51, 55, 47, 51, 47 | -1, 1, 1, 3, 5 |
| 0.75 | +0.030 | +31.0 | 9 | 4 | 0.012 | 35, 28, 32, 35, 28, 32, 35, 28, 32 | -1, -1, 1, 3, 3, 5, 8, 8, 10 |
| 1.50 | 0.000 | 0 | 2 | 2 | 0.036 | 2, -2 | 6, 9 |
| 2.25 | -0.030 (right) | -31.0 | 8 | 2 | 0.017 | -29, -33, -37, -29, -33, -37, -25, -33 | -3, -1, 2, 2, 4, 6, 4, 9 |

The ball is rendered where the geometry puts it, darkens 2-4 columns to 1-4 % of the wall's radiance (and dims their
neighbours), and moves across the retina at the expected azimuth; the sim's optic input carries it. (The diagnostic
run's own verdict, 3 s window: FAIL, as the batch.)

## 6. Reading the numbers

1. **The object signal at LC11 / LC10a is at the noise floor in every mode.** The most sensitive measure is the
   sweep-locked tuning: averaging the per-cell drive in 24 bins of the ball's position over the four sweeps, the best
   LC11 cell's (max - min) is 4.3-11.0 mV with the ball and 3.0-11.6 mV in the no-ball control, where the same bins can
   only pick up noise; the ratio ball / none is 1.06 (off), 1.36 (sign-class), 1.05 (sign-abs), 1.29 (rect off), 0.96
   (rect sign-class), and for LC10a 0.78 / 1.06 / 0.87 / 1.17 / 0.92. The peak of the sweep-locked (ball - none) difference
   over (cell, bin) is +2.6 to +5.1 mV for LC11 and +3.9 to +5.9 mV for LC10a, the same size as the control's own
   tuning range. The best cell's time-mean (ball - none) drive is +0.04 (off), +0.23 (class), +0.11 (abs), +0.11 (rect
   off), +0.16 mV (rect class) for LC11 and +0.05 / +0.16 / +0.10 / +0.11 / +0.14 mV for LC10a, against the 7 mV
   threshold; NOTES 9 measured +0.09 (LC11) / +0.19 mV (LC10a) under `off` with its own geometry.
2. **The 7 mV / 1 Hz criterion as written cannot pass or fail on its own**: with no stimulus the per-frame drive onto
   LC11 / LC10a already peaks at 8-14 mV (0.1-0.4 % of cell-frames > 7 mV, minima -8 to -11 mV) and the best LC10a cell
   fires 0.4-1.2 Hz spontaneously (LC10b 1.5-1.7 Hz population mean, 42-50 of 95 cells > 1 Hz, in both conditions).
   The criterion is met by "and not without" failing, i.e. the ball is indistinguishable from the model's own
   fluctuation; any future pass must be a sweep-locked (ball - none) measure above the control's scatter.
3. **LPLC2 does not see this ball either** (best cell diff +0.20 to +0.45 mV, rates 0.11-0.51 Hz in both conditions;
   NOTES 9 reported +9.8 mV peak and 0.09 -> 0.20 Hz with an unrecorded geometry). Here the ball rests on the table
   (centre 3.8 mm above the eye) and covers ~3 of 1,466 columns; the radiance check (section 5) confirms it darkens
   those columns to 1-4 % of the background wall as it passes. The NOTES-9 assay probably held the ball higher (the
   loom code puts its ball 1 cm above the eye); the LPLC2 difference is a stimulus-geometry difference, not evidence
   about LC11 / LC10.
4. **T2 / T3 rectification does not rescue the pathway.** Baseline 0 halves T2's and T3's dynamic range (mean |dev|
   0.146 -> 0.049 and 0.085 -> 0.026 rate units; the deviation is >= 0 by construction and reaches +0.80 / +0.58) and
   shifts LC11's time-mean drive (+0.08 -> +0.20 mV under off, +0.22 -> +0.36 under sign-class) and silences its
   spontaneous spikes (0.006 -> 0.000 Hz under off), but the (ball - none) difference stays at +0.11 / +0.16 mV (best
   cell time-mean) and the sweep-locked tuning ratio at 1.29 / 0.96 -- the ON/OFF cancellation the critic proposed is
   not what limits LC11, or rectifying the output of a linear sum does not undo the cancellation at its input.
5. **The medulla carries the ball at the ~0.1 rate-unit level and no further.** The largest per-cell |dev| difference
   is in Tm3 (best cell +0.09 to +0.11 in every mode), then Mi4 (+0.02 to +0.07), Mi1 (+0.05 to +0.08); the lobula
   stage Tm5Y / TmY21 / TmY13 / TmY5a shows +0.02 to +0.06 and T2 / T3 +0.01 to +0.05, all at population ratio
   0.96-1.03. The receptor model cannot alter this (section 4): Tm5Y, TmY21, TmY13 and LC11 have no profile in any
   source (fallback 100 %), T3 has no sign change, T2 0.08 %.

## 7. Verdict

**FAIL in 16 of 17 runs (their 10 + the skeptic's 7 of the identical protocol; 1 PASS on LC10a's spontaneous rate, 13 vs 12 spikes in 12 s), every mode** (`off`, `sign-class`, `sign-abs`, and `off` / `sign-class` with T2 / T3 rectified;
seeds 0 and 1). Neither LC11 nor LC10a carries the 1 cm moving ball above the model's own drive fluctuation; the
receptor model does not change a single input synapse of LC11, LC10a, LC10b, LC16, LPLC2, LC4 or T3 under any net rule,
and its medulla changes (Mi4 40 % under class) move the pathway's operating point without adding an object signal.
Hypothesis (a) of `docs/NT_INTEGRATION.md` (the small-object pathway is a receptor-sign question) is closed as
**outside the receptor route**; the object item remains the medulla -> lobula small-field wiring / dynamics question of
NOTES 9, with this script and `probe_figure_ground.py` as its two benchmarks.

Caveats: 2 seeds per mode (the native backend's scatter is visible in the per-seed rows: LC11 tuning range 4.3-11.0 mV
for the same condition); the `nonmda` rule was not run (its only difference on this pathway is Mi1 39 % flipped;
LC11 / T3 inputs unchanged, so no different outcome is expected, but it is unmeasured); one ball geometry (on the
table, 5 cm ahead), one speed; the per-frame drive fluctuation that sets the noise floor has not been traced to its
source (spiking feedback into the optic lobe vs the LIF noise).

## Corrections (round-2 verification, `verify:score:object`)

* The FAIL verdict is stochastic at the criterion: one of seventeen identical runs passes (LC10a max cell 1.167 Hz with the ball vs 0.833 without = 13 vs 12 spikes at the 1/12 s rate quantum). Only the (ball - none) magnitudes are informative: LC11 best-cell time-mean +0.03..+0.48 mV, LC10a +0.05..+0.16 mV, against 7 mV.
* The sweep-locked tuning ratios and per-mode difference means quoted in sections 3 and 6.1 are two-run estimates with ~100 % run-to-run scatter (sign-class LC11 ratio 1.36 -> 1.16 pooled over 5 runs; off 1.06 -> 1.10 over 6; the no-ball control exceeds the ball in 2 of 5 sign-class runs). The off-vs-class separation of the mean difference survives (+0.076 vs +0.27 mV) but is not a mode ranking.
* `diff_best_cell_mean_mv` is the maximum over cells of the per-cell (ball - none) time-mean, not the difference at the best-driven cell; `diff_peak_mv` / `diff_peak_cells_over_7mv` subtract two independent stochastic runs frame by frame and are noise (to be dropped); the 100 ms boxcar starts at frame 1 (off by one, immaterial).
* Section 6.4's "LC11 spontaneous rate 0.006 -> 0.000 Hz" is the ball condition; no-stimulus values are 0.0015 -> 0.0000. Section 6.4's no-ball LC11 mean drive under sign-class is +0.156 mV (ball +0.221), not +0.22 for both.
* Angular-size ladder (skeptic's positive control, off, seed 0): a static object of 11 / 22 / 28 / 43 deg raises LPLC2's (ball - none) best-cell drive +0.11 / +0.97 / +1.37 / +3.67 mV, LC16 up to +1.20, LC10b +1.65, LC11 only +0.05 / +0.12 / +0.12 / +0.21 -- the loom chain responds to size, the small-object detectors do not; NOTES 9's LPLC2 +9.8 mV was a larger / higher stimulus, not a model difference.
* Two seeds per mode is not enough for any ratio here; round 3 adds a none-vs-none null and five replicates per mode.

## 8. Round 3: the sweep made quantitative -- five replicates per mode, a none-vs-none null, z-scores, the size ladder

Round 2 left the object sweep with a per-run verdict that is a coin flip and (ball - none) magnitudes quoted from two
seeds with no null. Round 3 measures the null directly: the probe can now run the **no-ball condition twice**
(`--null`) and report the identical difference statistics for that pair, so every (ball - none) number has a
distribution to be read against. One cluster batch, 24 jobs, no `--cache-dir`.

### 8.1 Changes to `scripts/probe_object_sweep.py` (this round)

* `diff_best_cell_mean_mv` -> **`diff_max_over_cells_mean_mv`**: it is the max over cells of the per-cell (ball - none)
  time-mean drive, never "the difference at the best-driven cell" (round-2 correction). A companion
  `diff_mean_over_cells_mean_mv` (the population mean of the same per-cell difference) is new.
* `diff_peak_mv` and `diff_peak_cells_over_7mv` are **removed**: a frame-by-frame subtraction of two independent
  stochastic runs is noise. What that noise is worth is now measurable -- `diff_peak_100ms_mv`, which has the same
  defect and is kept only because the null measures it, is **+11.4 mV for LC11 with no ball in either run** (off,
  mean of 5 null runs) against +10.8 mV with the ball. Any "difference peak" statistic on this protocol reads above
  the 7 mV pass criterion in a pair of runs that contain no stimulus at all.
* `smooth_peak` (the 100 ms boxcar) off-by-one fixed: the cumulative sum now carries a leading zero row, so window *i*
  is `frames[i:i+w]`; before, the first window was `frames[1:w+1]` and frame 0 was never scored. Unit check: a single
  +100 mV frame at t = 0 gives +10.0 mV at w = 10 after the fix and +0.0 before.
* `--null`: run the no-ball condition twice (same seed, same code path -- the two runs differ only by the native
  backend's run-to-run nondeterminism) and report the same statistics; `config.null = true`, `condition_a = "none"`.
* `--ball-radius` / `--ahead` / `--half-sweep` expose the geometry, so the round-2 skeptic's angular-size ladder is a
  flag change rather than a wrapper script (his `scripts/skeptic_object_check.py` is no longer in the tree).
* **`--receptor-model off` is now applied, not skipped, and `default` is a third choice.**
  `LIFParams.receptor_model` defaults to `'sign'` / `receptor_net_rule 'abs'` since round 3, so round 2's
  `patch_receptor` (which returned early for `off`) would have run the default receptor model under the label `off` --
  the mislabel `docs/audits/receptor_integration.md` A.1 records for the no-flag runs `out/r3_obj_default_s{0,1}.json`
  ("`config.mode` says `off`"), with the request that this script's owner add a `default` choice. Done:
  `--receptor-model {default,off,sign}`, flag default `default` = leave `LIFParams` alone (so a bare run is the
  shipped model), `off` = `receptor_model = None` applied explicitly. Every run now prints and records the pair it
  actually used (`config.receptor_model` / `config.receptor_net_rule`, with the flag in `config.receptor_model_flag`).
  `--receptor-net-rule` keeps its own default `class` so that round-2 commands reproduce; the LIFParams default is
  `abs`, so the rule has to be spelled out with `--receptor-model sign`.

### 8.2 The batch

Cluster run `r3-obj-0101cb` (24 jobs, one batch, 0 failed, 3.9 min wall; every job `device cuda`, NVIDIA B200, torch
2.11.0+cu128, native backend, no CUDA graphs): `--null` x 5 seeds for each of `off` and `sign`/`abs` (10 jobs) and the
ball sweep x 5 seeds for each of the two modes (10 jobs), plus the four ladder geometries (off, seed 0). Both modes
are given explicitly on the command line, and each mode is compared with its own null. Outputs `out/r3obj/{ball,null}_{off,abs}_s{0..4}.{json,txt}`,
`out/r3obj/lad_{11deg_static,22deg,28deg,43deg}.{json,txt}`, aggregated by `out/r3obj/aggregate.py` into
`out/r3obj/aggregate.md`; structural coverage by `out/r3obj/coverage_r3.py` into `out/r3obj/coverage_r3.json`
(both scratch scripts kept with the outputs; `out/` is git-ignored).

Provenance (job `r3-obj-cache-3add6e`, `out/r3obj/cachecheck.txt`): the cluster's **shared** cache is now the adopted
`TYPE_NT_OVERRIDE` cache -- 167,106 cells, nnz 25,578,600, sum W 25,825,116, sum |W| **121,460,584**, TmY14 glutamate
300 / ACh 177, Mi19 serotonin 12 -- identical to the local `cache/`, so no `--cache-dir` is passed any more. The files
the jobs ran are sha256 `9216bbaa...` `scripts/probe_object_sweep.py`, `ac8f4421...` `flyverse/brain.py`,
`a15dc6e0...` `flyverse/connectome.py`, `0ec5355d...` `flyverse/data/receptors_by_type.csv` (the round-3
contested-flip table), each equal to the local working-tree file at submission time.

*Script provenance.* The 24 jobs ran `probe_object_sweep.py` sha256
`9216bbaae669d7369e3a6aab1301ae7b2126f76af0fdb38f1237dfc6079b5339`; the file shipped in the tree is sha256
`c8ebd337e87103eace2f9dd874bd5d670eaaaff9991f182bb910e8a35b7ea24a`, which adds the `--receptor-model default` choice
and the resolved-receptor recording of 8.1 **after** the batch. The diff is confined to the argument parser, the
header line and the JSON `config` block; no line of the simulation, recording, summary, difference or verdict path
differs, and with the explicit flags used here (`--receptor-model off`, `--receptor-model sign --receptor-net-rule
abs`) both versions build the identical `LIFParams` -- checked by calling `patch_receptor` / `resolved_receptor` on
CPU: `off` -> `(None, 'class')`, `sign abs` -> `('sign', 'abs')`, `default` -> `('sign', 'abs')`. Re-running any
command of 8.2 with the shipped file therefore repeats the same configuration; only the JSONs' `config.receptor_model`
field changes (`"off"` -> `null`, plus the new `receptor_model_flag`). The shipped file was then run end to end on the
cluster (job `r3-obj-smoke-cdc344`, 2 jobs, 1.3 min, `out/r3obj/smoke_{default,off_null}.{json,txt}`, 2 s window): a
bare invocation now prints `mode sign-abs (--receptor-model default)` and `receptor model sign (abs); fast sign
changed on 48,295 of 25,578,600 entries` (the count of 8.3) and records `receptor_model: "sign"`,
`receptor_net_rule: "abs"`, `receptor_model_flag: "default"`; `--receptor-model off --null` records
`receptor_model: null`, `null: true`, `condition_a: "none"`.

### 8.3 What `sign`/`abs` changes on this pathway (round-3 table, CPU, `out/r3obj/coverage_r3.json`)

Model-wide the current table under `abs` changes 48,295 of 25,578,600 entries (30,916 flipped + 17,379 silenced) =
179,944 |W| synapses = 0.148 %. On the measured types it changes **zero** input synapses of LC11, LC10a, LC10b, LC16,
LPLC2, LC4, T3, Tm5Y, TmY21 and TmY13, and only silences histamine input on Mi4 20,024 (3.74 % of its input), Mi1
13,013 (1.90 %), T2 712, Tm3 476, TmY5a 216. (Under `class` the model-wide count is 161,877 entries / 630,436 syn.)
The round-2 statement is unchanged by the contested-flip rebuild: whatever the two modes differ by downstream, it does
not reach these cells through their own inputs.

### 8.4 (ball - none) against the none-vs-none null

Statistic: `diff_max_over_cells_mean_mv` (mV), 12 s window, 5 seeds per cell of the table. `z = (mean(ball-none) -
mean(none-none)) / SD(none-none)`; the Welch column is the same difference over the standard error of the two means
(`sqrt(s_B^2/5 + s_N^2/5)`); `U` is the exact Mann-Whitney statistic of the 5 ball values against the 5 null values
(max 25) with its exact two-sided p (0.0079 is the smallest attainable at n = 5, 5).

| type | cells | mode | (ball - none) per seed 0..4 | mean | SD | (none - none) per seed 0..4 | null mean | null SD | z | Welch | U | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC11 | 143 | off | +0.081 +0.118 +0.040 +0.058 +0.032 | +0.066 | 0.035 | +0.080 +0.067 +0.026 +0.097 +0.071 | +0.068 | 0.026 | **-0.1** | -0.1 | 12 | 1.00 |
| LC11 | 143 | sign-abs | +0.091 +0.062 +0.032 +0.091 +0.096 | +0.074 | 0.027 | +0.093 +0.067 +0.040 +0.065 +0.067 | +0.066 | 0.019 | **+0.4** | +0.5 | 14 | 0.84 |
| LC10a | 275 | off | +0.111 +0.087 +0.065 +0.067 +0.077 | +0.081 | 0.019 | +0.054 +0.104 +0.067 +0.077 +0.079 | +0.076 | 0.019 | **+0.3** | +0.4 | 14 | 0.84 |
| LC10a | 275 | sign-abs | +0.088 +0.058 +0.063 +0.084 +0.116 | +0.082 | 0.023 | +0.127 +0.064 +0.068 +0.102 +0.062 | +0.085 | 0.029 | **-0.1** | -0.2 | 11 | 0.84 |
| LC10b | 95 | off | +0.115 +0.180 +0.219 +0.029 +0.101 | +0.129 | 0.074 | +0.156 +0.036 +0.065 +0.148 +0.021 | +0.085 | 0.063 | **+0.7** | +1.0 | 17 | 0.42 |
| LC10b | 95 | sign-abs | +0.099 +0.146 +0.086 +0.129 +0.166 | +0.125 | 0.033 | +0.026 +0.067 +0.095 +0.082 +0.140 | +0.082 | 0.042 | **+1.0** | +1.8 | 21 | 0.10 |
| LC16 | 182 | off | +0.139 +0.125 +0.049 +0.103 +0.064 | +0.096 | 0.039 | +0.118 +0.055 +0.120 +0.068 +0.146 | +0.102 | 0.038 | **-0.1** | -0.2 | 11 | 0.84 |
| LC16 | 182 | sign-abs | +0.047 +0.121 +0.056 +0.063 +0.052 | +0.068 | 0.031 | +0.219 +0.075 +0.116 +0.138 +0.054 | +0.120 | 0.064 | **-0.8** | -1.7 | 5 | 0.15 |
| LPLC2 | 185 | off | +0.302 +0.288 +0.287 +0.299 +0.267 | +0.289 | 0.014 | +0.153 +0.200 +0.143 +0.266 +0.190 | +0.190 | 0.048 | **+2.0** | +4.4 | 25 | 0.0079 |
| LPLC2 | 185 | sign-abs | +0.369 +0.370 +0.326 +0.392 +0.299 | +0.351 | 0.038 | +0.124 +0.081 +0.178 +0.137 +0.174 | +0.139 | 0.040 | **+5.4** | +8.6 | 25 | 0.0079 |
| LC4 | 126 | off | +0.126 +0.126 +0.023 +0.115 +0.039 | +0.086 | 0.051 | +0.036 +0.050 +0.074 +0.143 +0.142 | +0.089 | 0.051 | **-0.1** | -0.1 | 10 | 0.69 |
| LC4 | 126 | sign-abs | +0.072 +0.104 +0.043 +0.106 +0.046 | +0.074 | 0.030 | +0.110 +0.120 +0.064 +0.100 +0.085 | +0.096 | 0.022 | **-1.0** | -1.3 | 7 | 0.31 |

**The null is not centred on zero.** With no ball in either run the statistic is +0.066 to +0.190 mV, because it is a
maximum over 95-275 cells of a per-cell difference whose per-cell mean is ~0 (`diff_mean_over_cells_mean_mv` under `off`:
+0.004 / +0.000 / -0.016 / +0.004 / +0.001 / +0.008 mV for LC11 / LC10a / LC10b / LC16 / LPLC2 / LC4 against a null
of +0.008 / -0.002 / -0.028 / +0.005 / -0.001 / +0.007 -- every |z| <= 0.2; under `sign`/`abs` the same statistic
spans z -2.2 (LC4) to +1.5 (LC11), also no signal). That positive bias is the whole of the
round-2 "signal" in the two modes replicated here: round 2's LC11 values under `off` (+0.050 +0.034 +0.084 +0.080
+0.131 +0.074 over six runs, mean +0.075; `receptor_verification.md` round-2 record) and LC10a +0.05..+0.16 mV sit
inside this null, and reproduce the off mean of +0.066 +- 0.035 measured here. Round 2's larger `sign-class` values
(LC11 +0.15..+0.48, mean +0.27) are **not** covered: `class` was not replicated in this batch and has no null, and it
is the mode that raises LC11's drive in both conditions (+0.08 -> +0.22 mV time-mean, section 4), so its difference
statistic has to be read against a `class` null, not against this one.

Other difference statistics, same treatment (all in `out/r3obj/aggregate.md`): the sweep-locked tuning peak
`diff_tuning_peak_mv` is null-dominated for every type (LC11 off +3.95 vs null +4.01; LC10a +4.94 vs +4.78; LPLC2
+3.99 vs +4.03; largest |z| 1.4, LC10b sign-abs); `diff_rate_hz_max_cell` likewise (LC11 off +0.13 vs +0.15 Hz;
LPLC2 +0.63 vs +0.58; largest |z| 0.9); `diff_peak_100ms_mv` as in 8.1.

### 8.5 Which types show a (ball - none) signal above the null

**At z > 3 on the primary statistic: LPLC2 under `sign`/`abs` (z = +5.4) and nothing else.** LPLC2 under `off` is
z = +2.0 on the SD of the null but +4.4 on the standard error of the two means, with all five ball runs above all
five null runs (U = 25/25, p = 0.0079) and the tightest ball arm in the table (SD 0.014 mV) -- so LPLC2 carries this
11.4 deg ball in **both** modes, at +0.10 mV (off) / +0.21 mV (sign-abs) above its own null, i.e. **30-70 x below the 7 mV pass
criterion** and invisible in the population mean drive (+0.481 vs +0.479 mV, 8.4).
**LC11, LC10a, LC10b, LC16 and LC4 show no signal above the null in either mode** (every |z| <= 1.0, every U within
chance except LC10b sign-abs: z +1.04, U 21/25, exact p 0.0952; pooled 9 v 9 with the skeptic's seeds Welch p 0.013, U 68/81 p 0.014, but NOT above the skeptic's invisible-ball control (+0.135 vs +0.129 mV, p 0.93)) -- the small-object detectors do not see the object, and the difference between their (ball - none)
numbers and zero is the statistic's bias, not a response.

The per-run 7 mV / 1 Hz verdict remains uninformative and is now bounded: **1 PASS in the 20 replicate runs** (10 ball + 10 null; the PASS is ball off
seed 0, on LC10a's max-cell rate 1.167 vs 0.833 Hz = 14 vs 10 spikes in 12 s), and **0 of 10 no-ball-vs-no-ball runs
pass** -- consistent with round 2's 1 in 17, and with a criterion that is decided by the spontaneous rate.

### 8.6 Angular-size ladder repeated (off, seed 0, one run per size)

Same four geometries as the round-2 skeptic, now as flags (`--ball-radius`, `--ahead`, `--half-sweep`; the 11.4 deg
entry is the static ball, half sweep 1e-9, the other three sweep). `diff_max_over_cells_mean_mv`, mV; the round-2
values are the skeptic's `diff_best_cell_mean_mv` from `out/skobj/sk_{static,r010,near,r020}_off_s0.json`.

| object | r / ahead / half sweep (m) | LC11 | LC10a | LC10b | LC16 | LPLC2 | LC4 | LPLC2 rate mean ball/none (Hz) | LPLC2 max cell ball/none (Hz) |
|---|---|---|---|---|---|---|---|---|---|
| 11.4 deg static | 0.005 / 0.05 / 1e-9 | +0.050 | +0.052 | +0.151 | +0.034 | +0.189 | +0.076 | 0.115 / 0.121 | 2.67 / 2.92 |
| *(round 2)* | | +0.051 | +0.045 | +0.016 | +0.065 | +0.114 | +0.055 | 0.120 / 0.102 | 2.50 / 2.33 |
| 22.6 deg | 0.010 / 0.05 / 0.06 | +0.118 | +0.120 | +0.145 | +0.405 | +0.977 | +0.188 | 0.148 / 0.143 | 2.33 / 2.33 |
| *(round 2)* | | +0.123 | +0.117 | +0.142 | +0.385 | +0.965 | +0.212 | 0.157 / 0.133 | 2.67 / 2.67 |
| 28.1 deg | 0.005 / 0.02 / 0.024 | +0.102 | +0.112 | +0.021 | +0.535 | +1.358 | +0.345 | 0.181 / 0.114 | 2.83 / 2.42 |
| *(round 2)* | | +0.124 | +0.120 | +0.076 | +0.460 | +1.369 | +0.349 | 0.182 / 0.123 | 2.83 / 2.58 |
| 43.6 deg | 0.020 / 0.05 / 0.06 | +0.189 | +0.172 | +1.555 | +1.176 | +3.456 | +0.660 | 0.367 / 0.120 | 7.42 / 2.42 |
| *(round 2)* | | +0.207 | +0.176 | +1.650 | +1.199 | +3.674 | +0.679 | 0.345 / 0.126 | 7.42 / 2.83 |

The ladder reproduces: every entry repeats the round-2 value inside the run-to-run scatter of the section-8.4
replicates except LC10b at 11.4 deg (+0.151 vs +0.016, 2.1 x that type's null SD of 0.063). The null of 8.4 applies to
every row -- neither of its two runs contains an object, so it does not depend on the ball's geometry, only on the
mode and the window. Read against it, the loom types scale with size far outside it (LPLC2 +0.19 -> +0.98 -> +1.36 -> +3.46 mV,
null +0.190 +- 0.048; LC16 +0.03 -> +1.18, null +0.102 +- 0.038; LC10b +0.15 -> +1.56, null +0.085 +- 0.063) and
LPLC2's best cell goes from 2.67 to 7.42 Hz with the largest object while its no-ball control stays at 2.4-2.9 Hz.
LC11 and LC10a move from +0.05 to +0.19 / +0.17 mV over a 4 x size range: the 43.6 deg point is above the 11.4 deg
null (LC11 null +0.068 +- 0.026), but it is one run per size, it is 37 x below the 7 mV criterion, and it is the same
value LPLC2 shows for an object 16 x smaller in solid angle. The small-field detectors have no size tuning here.

### 8.7 The medulla does carry the ball

Same treatment on the optic rate units (`diff_abs_best_cell_mean`, max over cells of the (A - B) time-mean |dev|,
rate units; ball mean of 5 seeds vs the null mean +- SD of 5 seeds):

| type | off ball | off null | z(off) | sign-abs ball | sign-abs null | z(abs) |
|---|---|---|---|---|---|---|
| Mi4 | +0.0754 | +0.0098 +- 0.0023 | **+28.6** | +0.0457 | +0.0077 +- 0.0017 | **+22.3** |
| Mi1 | +0.0510 | +0.0192 +- 0.0041 | **+7.8** | +0.0724 | +0.0153 +- 0.0020 | **+27.9** |
| Tm3 | +0.1010 | +0.0303 +- 0.0091 | **+7.8** | +0.1064 | +0.0252 +- 0.0052 | **+15.6** |
| Tm5Y | +0.0380 | +0.0339 +- 0.0103 | +0.4 | +0.0499 | +0.0292 +- 0.0080 | +2.6 |
| TmY21 | +0.0355 | +0.0300 +- 0.0049 | +1.1 | +0.0343 | +0.0301 +- 0.0073 | +0.6 |
| T2 | +0.0337 | +0.0301 +- 0.0080 | +0.5 | +0.0429 | +0.0302 +- 0.0108 | +1.2 |
| T3 | +0.0224 | +0.0224 +- 0.0073 | -0.0 | +0.0227 | +0.0193 +- 0.0048 | +0.7 |
| TmY13 | +0.0187 | +0.0255 +- 0.0109 | -0.6 | +0.0240 | +0.0208 +- 0.0063 | +0.5 |
| TmY5a | +0.0292 | +0.0354 +- 0.0062 | -1.0 | +0.0365 | +0.0311 +- 0.0045 | +1.2 |

So the ball is in the model's medulla at high confidence (Mi1 / Mi4 / Tm3, z 7.8-28.6 in both modes), is at or below
the null by the time it reaches T2 / T3 and the lobula Tm / TmY stage (|z| <= 1.2, except Tm5Y +2.6 under sign-abs),
and reappears only in LPLC2. This is the same conclusion as section 6.5, now with an error bar: it is not that the
measurement is too noisy to see the object -- the same measurement resolves Mi4's 0.066 rate-unit difference against
a null of 0.010 +- 0.002 -- it is that
the small-field pathway does not pass it on.

### 8.8 Caveats

* 5 seeds per cell of the table; the null and the ball arm are measured under the same mode, so z compares like with
  like, but a z of 3 at n = 5 is a modest claim (the exact rank-sum test cannot go below p = 0.0079 here).
* The null measures the native backend's run-to-run nondeterminism at fixed seed and fixed weights only. It is not a
  null for seed-to-seed variation of the brain RNG, nor for the geometry.
* Two modes only (`off` and the shipped `sign`/`abs`). `class`, `nonmda` and the T2/T3-rectified variants of sections
  3-6 have no null of their own; because the null scales with a mode's drive scatter, their round-2 difference
  numbers must not be compared with the nulls measured here.
* `diff_max_over_cells_mean_mv` is a maximum over cells and is therefore biased upward; it is comparable across
  conditions only because the null is computed the same way, with the same cell count.
* One ball geometry for the replicate table (1 cm, 5 cm ahead, on the table, ~46 deg/s), one window (12 s); the ladder
  is a single run per size, so differences below ~0.1 mV in it are not resolvable.
* LPLC2's signal is +0.10 / +0.21 mV against a 7 mV criterion and does not appear in its population mean drive or its
  firing rate (+0.63 / +0.47 Hz max-cell difference, both inside the null): it is a few cells' drive, not a response
  the downstream escape circuit would act on. The pathway conclusion of sections 6-7 is unchanged.

### Corrections (round-3 verification, `verify:exp:object-null`)

* z > 3 at n = 5 per arm has P = 0.0045 per comparison under a Gaussian null (5.3 % over the twelve comparisons of 8.4); LC16 sign-abs reached z +4.59 against the skeptic's own 4-seed null and -0.8 / -0.21 against the shipped and pooled ones -- one z > 3 at n ~ 5 is not by itself a signal. LPLC2's result does not depend on it (Welch, exact rank-sum, both modes, an independent batch, and an invisible-ball control with 0 of 1,466 columns changed).
* "reads above the 7 mV pass criterion" compares a difference statistic with a threshold the script applies to absolute drive; say "a difference-peak statistic reaches 11 mV in a stimulus-free pair".
* Batch wall 3.9 min is cluster_run's submit-to-fetch wall; the 24 jobs' own span was 3.0 min.
