# Deficit: the plain fly does not turn -- localized with decompose / atlas / paths

Task apply:turning (interpretability toolkit, `docs/INTERP.md`). Script: `scripts/interp_apply_turning.py` (`record` /
`atlas-run` GPU, `plan` / `analyse` CPU); it calls `flyverse.interp.decompose`, `.atlas` and `.paths` unchanged and adds
only the L-R bookkeeping of the turning readout. Data: `out/turn/` (the GPU recordings and atlas runs, cluster batch
`turn-eb210b`, `out/turn_cluster.log`: `6 job(s), 0 failed (11.5 min)`, every job `device cuda`, NVIDIA B200),
`out/interp/apply_turning/` (`turning.json` 12.7 MB, the per-tool Result JSONs, the console `turning_analyse_clean.txt`).
Nothing under `flyverse/` was edited; every run is the shipped default model (`LIFParams()`, `type_path_gain`
`[LC4|LPLC2 -> DNp01 x3]`, receptor model `sign`, no program, no gain, no hold table; `ew.md5 ed1df661...`, compiled
`W` md5 `ef23cc27...`).

## 0. Verdict

**The plain fly walks straight because its one steering neuron, DNa02, sits under net tonic inhibition in the room
while its three lateralised excitatory input classes are silent at their sources -- and the one class that does
deliver a lateralised signal (the wind, through PS230) arrives at ~0.078 mV against a 7.0 mV threshold gap, ~90x
under dose.** In numbers (3 runs x 16 flies x 60 s; run means quoted; see section 2 for which tables drop the 5 s
start-up and which do not):

1. **DNa02 is held below threshold.** Its rate-weighted synaptic input in the room is **-326 / -392 mV/s** per cell
   (L / R; runs -326.4 / -329.3 / -321.5 and -387.5 / -395.3 / -391.9; E +750 / +795, I -1076 / -1186), although the
   wiring is net *excitatory* per volley (static, per cell: DNa02_L E +920.5 / I -454.5 = net +466.1, DNa02_R
   E +911.3 / I -463.7 = net +447.6 mV per volley, recomputed per entry from `common.effective_weights`; the
   grouped `static_dna02_{L,R}.json` report the same nets at E +916.8 / -450.7 and +910.1 / -462.5). **In the
   model's own units** (`brain.py`: `dg/dt = -g/tau_syn`, `v = v_rest + g + drive - adapt`, `tau_syn` 5 ms,
   `v_rest` -52, `v_th` -45) a steady input of I mV/s is a steady `g` of `I x 0.005` mV, so -326 / -392 mV/s is
   **-1.63 / -1.96 mV against a 7.0 mV threshold gap** -- about 25 % of the barrier, not the barrier. Reaching
   threshold needs ~1,400 mV/s *on top of* cancelling the inhibition. The inversion is because the wired excitation
   does not fire and the wired inhibition does:
   the inhibition comes from GABAergic VNC interneurons on DNa02's VNC arbor (IN12B014 -89 / -124 mV/s at 14 Hz,
   IN19A003 -44 / -57 at 12 Hz), GABAergic GNG / PS cells (GNG562 -94 / -80 at 12 Hz, PS059 -95 / -49 at 7 Hz,
   GNG284 -51 / -30), the glutamatergic object-pathway cell LT51 (-57 / -70 at 3.3 Hz) and MBON31 / 32 (-15 / -35,
   -36 / -40) -- every one sign-correct by its transmitter (`nt_sign`). DNa02 fires 0.011 / 0.075 Hz (max 9.9 / 14.4)
   and its L-R is -0.064 Hz (fly sd 0.036) = a 0.3 deg/s right bias through `k_turn`; the yaw command's SD is
   3.5 deg/s (runs 3.48 / 3.58 / 3.56), the realised yaw-rate SD 2.56 deg/s once the fence's 1.2 % of frames is
   removed (8.3 with them) -- the same 2.45-2.57 `probe_walk_straightness.py` measured with no fence.
2. **The three lateralised excitatory classes are silent at source (section 4.3).** Compass: PFL3 **0.000 Hz** in
   every cell of every fly (24 cells, max 0.000), PFL2 0.001, and PFL3 is by two orders of magnitude the largest
   CX -> DNa02 gate (+45-47 Hz per 150 Hz volley) -- but it is **not the only one**: 8 of the 199 non-PFL3 CX
   populations move `turn_LR` with verdict `result`, all by 0.6-1.6 Hz (hDeltaI_R +1.570, ExR8_L +1.516, hDeltaI_L
   +1.299, hDeltaA_L +1.224, vDeltaK_R +0.786, ExR8_R -0.710, PFL2_L +0.703, PEN_b(PEN2)_L -0.623;
   `atlas_turning.json` table `atlas`, readout `turn_LR`, 963 rows). At this file's own `k_turn` = 5 deg/s per Hz
   these are 3-8 deg/s, i.e. the order of the plain fly's whole yaw-command SD (3.54 deg/s) and 1-2.5x DNa02's L-R
   temporal SD (0.645 Hz) -- so "PFL3 is the only CX gate" is an overstatement, not a rounding. (Each of the eight
   fires in 1-2 of the 3 runs and 0.000 in the rest; the verdict is the `sd_floor` 0.05 Hz, section 8.) The
   columnar stages that do read 0.000 in all three runs are EPG, PEG, Delta7, PEN_a, every ER, ExR1-7, every
   FC / FS, PFL1, hDelta B-H / J-M and the other vDeltas. Object: AOTU015 0.000 Hz (max 10, 2 of 8 cells ever), AOTU001 0.000. Motion: LLPC1,
   the strongest DNa02 mover in the model (LLPC1_R at 150 Hz -> DNa02_R +119 Hz, LLPC1_L -> DNa02_L +99; 2.5x PFL3's
   +45-47, 4x AOTU015's +26-28), runs at 0.50 / 1.65 Hz per side with only 1-2 of its 45 left and 9-10 of its 59
   right DNa02-presynaptic cells above 0.5 Hz; 41.3 % of the LLPC1 -> DNa02 synapses (216 of 523) come from cells
   that never fire, while 63.5 % of the 104 cells fire at some point and 36.5 % never do.
   Six of DNa02's 30 strongest wired inputs never fire in the room (PFL3, AOTU001, LAL094, LAL179, PS034, LAL175),
   carrying **6.31 %** of DNa02's absolute static weight; the **12.1 %** figure in
   `summary.lr_top.silent_at_source.share_of_abs_static_weight_never_firing` is a different quantity -- the share
   of *all 794* never-firing presynaptic types over the whole 3,192-type table, not of the top 30 (`turning.json`
   table `lr_DNa02_L|DNa02_R:type`). `paths` never-firing share of DNa02's raw input 10.7 %, silent 11.5 %.
3. **One class carries a time-varying lateralised motion signal into DNa02, and the sum cancels it.** LLPC1/R and
   HSS/R (front-to-back and horizontal motion, cholinergic, +65 / +61 mV/s of within-fly L-R sd) are the largest
   positive contributors to the variance of DNa02's L-R input (var shares +0.124 / +0.128), and LPT22/R (a GABAergic
   lobula-plate tangential, -35 mV/s mean, sd 66) is the largest negative one (-0.113): corr(LPT22/R, LLPC1/R) =
   **-0.852 / -0.859 / -0.861** in the three runs, and the sd of their sum is 34-36 against 63-68 each. What survives
   (total L-R input sd 248 mV/s) does reach DNa02 -- corr(total L-R input, DNa02 L-R) = +0.26 +- 0.09 per fly against
   a 1 s block-shuffle null of 0.09 -- but on a cell whose mean input is -390 mV/s it produces 0.65 Hz of L-R sd and
   no net bias.
4. **The wind is the one lateralised sensory signal that arrives at DNa02, and it fails on dose.** Across the 48
   flies (random headings, wind 0.3 m/s from 180 deg) the fraction of time the wind is on the left predicts DNp18 L-R
   (r -0.73; per run -0.72 / -0.75 / -0.75), DNp33 L-R (+0.65), the PS230 -> DNa02 L-R *input* (r -0.61; -0.59 / -0.56
   / -0.68; slope -15.5 mV/s per unit), the leg driver DNa13's L-R (r +0.44; +0.56 / +0.37 / +0.39; slope +0.26 Hz)
   and the leg-MN L-R (r -0.65; slope -0.066 Hz, i.e. 0.2 deg/s through `k_leg_turn`) -- but not DNa02's rate
   (r -0.05; +0.15 / -0.44 / +0.07), DNae007 (-0.10; signs disagree across runs) or DNge035 (+0.18; disagree)
   (`per_fly_wind_side_stats.csv`). The dose: 15.5 mV/s of wind-lateralised input is **0.078 mV** of steady
   conductance (x `tau_syn` 5 ms) against a **7.0 mV** threshold gap -- **~90x under dose**, not the 25x that
   comparing 15.5 against the -390 mV/s of inhibition gives. That is the localization: the
   JO -> PS230 -> DNa02 route (`paths` +152 mV^2 per volley, no silent link) is live and lateralised, and two
   orders of magnitude under threshold.
5. **The leg readout has drivers that fire, and they carry fixed wiring offsets, not signals.** DNge035 runs at 5.15 /
   0.78 Hz (L-R +4.37, fly sd 0.67), DNae007 7.07 / 4.35 (+2.72), DNge037 1.00 / 3.18 (-2.18), DNa01 2.09 / 0.54,
   DNp18 25.5 / 17.4 (+8.1) under symmetric input -- the same kind of fixed anatomical L-R gradient `compass_room.md`
   found on PFL3 and `interp_atlas.md` 3.1 on the wind DNs. The leg-MN L-R (+0.19 Hz; +2.5 Hz per side) follows
   DNge035 contralaterally within a fly (median r -0.430 / -0.471 / -0.473 over the three runs; generator
   `wind-side --per-fly-corr`, section 8) as the atlas predicts (DNge035_R +7.10 /
   _L -6.42 Hz leg L-R, the strongest leg driver, 3x DNa02), and its summed L-R input over the 63 largest groups does
   not predict it at all (r -0.06, null 0.07): the leg asymmetry is set by the VNC premotor interneurons (IN16B016 /
   IN12A001 / IN19A016, 14-28 Hz) whose L and R cells cancel pairwise (corr -0.95 / -0.96), not by anything
   descending.
6. **What turning would require (section 7).** A lateralised drive that both cancels DNa02's tonic inhibition
   (-1.63 / -1.96 mV) and supplies the 7.0 mV threshold gap: ~1,750 mV/s in all. The
   model has every route wired with the right sign and side -- LLPC1 / HSS (motion, ipsilateral), LC10 -> AOTU015
   (object, ipsilateral), PFL3 (compass, contralateral), JO -> PS230 (wind) -- and each fails for a documented
   upstream reason: no rotation input to the ring (PFL3 at 0 Hz; `deficit_rotation.md`), the object figure lost at
   Tm5Y / TmY21 (`deficit_object.md`), 36.5 % of the DNa02-presynaptic LLPC1 cells never firing (41.3 % of the
   synapses) with the median firing cell at 0 Hz and LPT22 cancelling the
   rest, and a wind dose ~90x too small. **Data-driven** next steps are measurements and calibrations: `trace` from
   T4a / T5a to LLPC1 and LPT22 (why the median LLPC1 cell is silent; whether the LPT22 anti-correlation is the same motion
   signal with the opposite transmitter), `health` on the VNC interneurons that run at 12-37 Hz with every VNC
   sensory neuron at 0 Hz (SNpp39 / 45 / 50, LgLG: the proprioceptive loop is open -- section 6.3), and the JO
   transducer's rate-vs-wind-speed calibration against measured JO responses. **Hand-crafting** would be: any gain on
   PFL3 / LLPC1 / PS230 -> DNa02, a noise or bias term on DNa02, a change to `k_turn` / `k_leg_turn`, or a module that
   writes PFL3 (the `cx` program, already a documented stop-gap). Nothing here argues for changing a default.

## 1. Question and inputs

`body.py`'s yaw is `-k_turn (DNa02_R - DNa02_L) + k_leg_turn (legMN_L - legMN_R)` with `k_opto = 0` since session 9
(`k_turn` = 200 deg/s per 40 Hz = 5 deg/s per Hz; `k_leg_turn` = 100 deg/s per 30 Hz = 3.3 deg/s per Hz).
`scripts/probe_walk_straightness.py` (`out/ws_*.json`, batch `walk-straight-fb28c0`, no fence) measured the plain
fly's yaw-rate SD at 2.45 / 2.57 deg/s (default, seeds 0 / 1, median over 16 flies), 1.6-2.8 in every weight
configuration; straightness 0.99-1.00. So the readout's two inputs are symmetric in every configuration and the
question is where a lateralised signal would have to come from, and why none does. Inputs taken as given:
`compass_room.md` (c) (under the experiment gains PFL3 runs 1.6-7.1 Hz per side with a fixed 2.09-3.31 Hz L-R
gradient of bump position, DNa02 follows at 0.08-0.24 Hz under ~1 Hz of its own noise, yaw first harmonic 0.3-1.2
deg/s; the shipped default has no bump), `object_sweep.md` 8.7 / `deficit_object.md` (LC10a / LC11 carry nothing),
`interp_atlas.md` 3 and 5 (PFL3 -> DNa02 works when driven; the DN motor map; the wind DN flips and their fixed
offset), `cx_shift.md` 1 / `deficit_rotation.md` (GLNO sign 0, the efferent route).

## 2. Protocol

**(a) decompose.** `interp_apply_turning.py record --seed s` runs `BatchSim(16, program='none', fruit_set='all',
fence=True, start=(-0.15, 0.15, table top), energy 0.9, wind 0.3 m/s from 180 deg, initial heading uniform per
environment seed)` for 60 s on the shipped default weights (cuda kernels, event-driven, CUDA graphs, cuSPARSE) and
records, every second 10 ms frame, the rate-weighted synaptic input `I_g(t) = mean over post cells of sum_j A[post, j]
r_j(t)` (mV/s; `A = common.effective_weights = Brain._W_cpu`) of four targets -- DNa02_L, DNa02_R (one cell each),
leg_L (192 MNs: fl / ml / hl, side L), leg_R (189) -- split by presynaptic (type, soma side) group (11,736
presynaptic cells, 5,555 groups), plus the per-fly mean and max rate of every presynaptic cell, the DNa02 optic drive
(0.000 mV: DNa02 gets no direct sensory injection), the by-side rates of 20 watch types (the DN movers, PFL2/3,
DNp18/33, DNa01/03, DNp09, LLPC1, LT51, AOTU015, PS077), the motor readouts, the body's yaw command, heading,
speed, wind angle and airborne flag. The per-fly mean rates are also written as a `common.Recording` whose frames
are flies, so `decompose(c, target, recording={'plain': [3 runs]}, by=('type','side'))` runs unchanged: its window
mean over frames is the mean over the run's 16 flies and the 3 runs (seeds 0-2, one batch) are the replicates (the
`decompose_<target>_<by>.json` Results; `Result.check()` empty on all 8). `analyse` adds the L-R tables (each type's
mean input to the L and the R target, the difference with its scatter over 48 flies and over the 3 run means, and
now the type's own room mean / max rate so 'silent at source' is a column), the cancellation analysis (the union of
the 64 largest groups per pair; the per-fly variance of the total L-R input decomposed into per-group covariances,
the most anti-correlated group pairs, and the correlation of the total L-R input with DNa02 L-R, leg L-R and the yaw
command against a 1 s block-shuffle null, 100 shuffles), plus the two per-fly correlation rows of section 4.4 /
5 (`corr(LLPC1 L-R, DNa02 L-R)` and `corr(DNge035 L-R, leg L-R)`).
**Which tables drop the 5 s start-up:** `--skip 5` reaches `behaviour_table`, `watch_tables`,
`cancellation_tables`, `wind_side_table` and (since this revision) the **totals** row of `lr_tables`. It does
**not** reach `lr_tables`' per-type entries: those come from `group_mean_flies`, a mean the GPU `record` pass
accumulates over all 3,000 frames, so every per-type number in section 4.2 includes the transient and cannot be
re-windowed without re-recording. The totals in the shipped `turning.json` were written before the skip was
added and include it too; recomputed from `out/turn/plain_r{0,1,2}.npz` with the 5 s actually dropped they read
L -327.36 / -330.45 / -322.31 and R -385.59 / -391.89 / -390.57, a change of <= 3.5 mV/s -- the claim survives,
the earlier protocol sentence ("the first 5 s are dropped") did not. Wall time 477-484 s per run.

**(b) atlas.** `interp_apply_turning.py atlas-run --seed s` = `interp_atlas.run_once` on 963 populations: DNa02's
170 presynaptic types with >= 4 stored entries onto the two cells, every LAL type (204), the CX output / columnar
types (PFL1-3, PFR, FC, FS, hDelta, vDelta, EPG, PEN, PEG, Delta7, ExR, ER: 100) and the nine DN movers of
`interp_atlas.md`, split by type and soma side (a type listed by two specs is stimulated twice; the atlas labels the
second copy `#2`, an in-run replicate); 150 Hz for 400 ms after 200 ms settle, batch 64 with 4 null rows per chunk
(125 null rows per run), no sensory context, readouts = every motor group and pair plus the 20 watch types by side
(87 readouts), 3 runs (113 / 133 / 133 s). The 981-population DN + sensory atlas (`out/interp/atlas/dn_sensory.json`)
is read for the same readouts. **Floor caveat:** `turn_LR` *and* `leg_LR` are excitation-only measurements.
`turn_LR`, `turn_L`, `turn_R`, `leg_LR`, `leg_L` and `leg_R` all read exactly 0.00000 with SD exactly 0 in all
375 null rows (`atlas_turning.json`), so a purely inhibitory input cannot register on either -- LT51, PS077,
PS059, LAL083 / 094 / 126 reading 0.000 on `turn_LR` says nothing about them, and the same holds for every
inhibitory leg input behind the `leg_LR` mover table of section 5. With a 0 null SD the verdict reduces to
`|diff| >= 3 x sd_floor` = 0.15 Hz (section 8).

**(c) paths.** `interp_paths.paths(c, source, mover, k_max=3, top=30, recording=plain_r0_max, frozen='static')` from
six sources -- CX, LAL, visual projection neurons, JO, ORNs, VNC sensory -- to DNa02 and to the atlas' leg movers
(DNge035, DNa13, DNge037, DNge049, IN04B074), silent links flagged; `never_firing` is judged on `out/turn/plain_r0_max`
(one frame = per-cell max rate over 16 flies x 60 s, threshold 0.5 Hz). LLPC1 (the top `turn_LR` mover) was not in
the six-mover cap; its own link is the k = 1 entry of `visual_projection -> DNa02`.

Batch: `out/interp/apply_turning/batch.sh` (written by `plan`): 3 `record` + 3 `atlas-run` jobs, `--fetch out/turn/`,
console `out/turn_cluster.log`. Analyse: `PYTHONIOENCODING=utf-8 python scripts/interp_apply_turning.py analyse
--recordings "out/turn/plain_r*" --atlas-runs "out/turn/atlas_r*" --json out/interp/apply_turning/turning.json`
(~45 min CPU, mostly `paths`).

## 3. The plain fly in the room (`turning.json` `summary.behaviour`, 48 flies)

| quantity | mean | fly sd | run means |
|---|---|---|---|
| yaw-rate SD (deg/s, all non-airborne frames) | 8.30 | 4.75 | 8.67 / 8.87 / 7.37 |
| yaw-rate SD excluding |yaw| > 30 deg/s (run 0) | 2.56 | | (these are 1.2 % of frames at median radial position 0.55 m against 0.36 m overall: the fence's turns at the table edge; `probe_walk_straightness` ran with no fence and got 2.45-2.57) |
| median |yaw rate| (deg/s, run 0) | 0.82 | | |
| yaw command SD (deg/s; the brain's own signal, fence excluded by construction) | 3.54 | 0.64 | 3.48 / 3.58 / 3.56 |
| DNa02 L-R (Hz) | -0.064 | 0.036 | -0.063 / -0.059 / -0.070 |
| DNa02 L-R temporal SD (Hz) | 0.645 | 0.139 | 0.635 / 0.648 / 0.653 |
| DNa02_L / DNa02_R (Hz) | 0.011 / 0.075 | 0.014 / 0.032 | |
| leg-MN L-R (Hz; per side 2.52) | +0.191 | 0.037 | 0.193 / 0.192 / 0.188 |
| leg-MN L-R temporal SD (Hz) | 0.419 | 0.017 | |
| airborne fraction | 0.001 | | |

corr(DNa02 L-R, yaw command) per fly is 0.93 (by construction of `body.py`), corr(leg L-R, yaw command) 0.40.

Watch types (mean Hz over flies and runs; `summary.watch_LR`): PFL3_L / R **0.000 / 0.000** (max 0.000 in 24 cells
x 48 flies), PFL2 0.001 / 0.001, AOTU015 0.000 / 0.000 (max 2.2 as a side mean), PS077 0.003 / 0.021, LLPC1 0.504 /
1.647 (temporal sd 0.32 / 1.22; max 3.1 / 11.8), LT51 2.556 / 2.528, DNa02 0.011 / 0.075, DNa01 2.09 / 0.54, DNa03
0.67 / 0.60, DNa13 4.18 / 3.38, DNae007 7.07 / 4.35, DNge035 5.15 / 0.78, DNge037 1.00 / 3.18, DNge049 0.49 / 0.13,
DNge073 1.23 / 0.94, DNpe024 0.54 / 1.56, DNde003 0.21 / 0.21, DNp09 3.74 / 3.22, DNp18 25.5 / 17.4, DNp33 12.3 /
10.1 (run sd <= 0.12 Hz on everything but the wind DNs, 0.4-2.5).

## 4. (a) decompose: what drives DNa02 and the leg MNs

### 4.1 Static: the wiring (unchanged from the CPU round; `static_dna02*.json`)

DNa02's 48,125 raw input synapses come in three lateralised classes, mirror-symmetric to 10 % except LLPC1: (i)
visual, ipsilateral -- LLPC1 +33.1 (45 entries) onto DNa02_L from LLPC1/L, +55.5 (59) onto DNa02_R from LLPC1/R
(T4a +41.8 / T5a +41.5 mV per volley are 32 % of LLPC1's input), LT51 -26.6 / -27.7 (its inputs the small-object
types TmY21 / Tm37 / LC10b / Y3), AOTU015 +20.3 / +18.5 (51 % LC10d + LC10a), HSS +7.1, LPT22 -6.4 / -7.2; (ii)
compass, contralateral -- PFL3 +30.6 / +31.9 (736 synapses, 1.5 % of raw input) plus the LAL two-steps PFL3 ->
LAL126 / LAL083 / LAL040 -> DNa02 at -925 / -702 / -655 mV^2; (iii) LAL / VES / PS / ascending -- LAL304m +15.4,
AN04B003 +15.4 / +15.1, AOTU001 +15.6 / +14.7, PS077 -17.5 / -18.1, LAL179 +11.1, LAL094 -11.1, VES202m -11.1,
LAL083 / LAL126 / PS059 -10, PS230 +10.3. Totals **per side**, not pooled: DNa02_L E +920.5 / I -454.5 (net
+466.1), DNa02_R E +911.3 / I -463.7 (net +447.6) mV per volley (per-entry sums of `common.effective_weights`;
`static_dna02_{L,R}.json`, which group by (type, side) and so report +916.8 / -450.7 and +910.1 / -462.5 with the
same nets). The single pair "E +912 / I -455" is `static_dna02.json`'s two-cell pooled figure and should not be
read as either cell's. Sign-0 input 0.8 % (LAL123 92 synapses, OA-VUMa1 88, DNg34 15).

### 4.2 Room: the input that actually arrives (`lr_DNa02_L|DNa02_R:type`, mV/s per cell, mean over 48 flies, whole rollout)

| pre type | nt | room rate (Hz) | -> DNa02_L | -> DNa02_R | L-R (fly sd) | run means |
|---|---|---|---|---|---|---|
| IN12B014 (VNC 12B) | gaba | 14.0 | -89.5 | -124.1 | +34.6 (3.0) | 3 runs within 1 |
| GNG562 | gaba | 12.5 | -94.1 | -80.0 | -14.1 (1.6) | |
| PS059 | gaba | 7.1 | -95.4 | -48.9 | -46.4 (4.4) | |
| LT51 | glutamate | 3.3 | -57.5 | -69.6 | +12.1 (2.2) | |
| IN19A003 (VNC 19A) | gaba | 11.6 | -43.9 | -57.3 | +13.4 (5.0) | |
| GNG284 | gaba | 7.9 | -50.8 | -30.4 | -20.3 (2.8) | |
| LAL120_a | glutamate | 7.8 | -32.1 | -48.4 | +16.3 (1.3) | |
| PLP029 | glutamate | 7.9 | -41.8 | -38.7 | -3.1 | |
| MBON32 / MBON31 | gaba | 8.6 / 5.3 | -35.6 / -15.1 | -40.0 / -34.8 | +4.4 / +19.7 | |
| AVLP712m | | 5.1 | -23.5 | -48.0 | +24.5 | |
| LPT22 | gaba | 2.9 | -6.1 | -34.7 | +28.5 (2.6) | |
| PLP012 | ACh | 8.6 | +35.9 | +51.9 | -16.1 (2.7) | |
| CB0431 | ACh | 6.5 | +9.4 | +56.1 | -46.7 (1.9) | |
| DNae007 | ACh | 5.7 | +36.6 | +22.0 | +14.6 (1.3) | |
| PVLP141 | ACh | 6.9 | +34.3 | +31.8 | +2.5 | |
| LLPC1 | ACh | 0.27 (0.50 / 1.65 by side) | +0.6 | +31.6 | -30.9 | |
| HSS | ACh | | +0.0 | +26 | | |
| PS230 (wind route) | ACh | 0.64 | +9.0 | +4.0 | +4.9 | |
| AN04B003 (ascending) | ACh | 0.48 | +9.5 | +5.2 | +4.4 | |
| PFL3 | ACh | **0.000** | 0.0 | 0.0 | 0.0 | |
| AOTU015 / AOTU001 | ACh | **0.000 / 0.000** | 0.0 | 0.0 | 0.0 | |
| PS077 | | 0.014 | -0.07 | -0.46 | | |
| **totals** | | | **-325.7** (E +750.3, I -1076.0) | **-391.5** (E +794.6, I -1186.2) | +65.8 (22.5) | -326.4 / -329.3 / -321.5; -387.5 / -395.3 / -391.9 |

The totals row is the whole rollout; with the 5 s start-up dropped it reads L -327.4 / -330.5 / -322.3 and
R -385.6 / -391.9 / -390.6 (section 2). In `g` this is **-1.63 / -1.96 mV** against the 7.0 mV threshold gap.

Every L-R above is a fixed offset (sign consistent in 48 / 48 flies, run sd < 10 % of the mean): the two DNa02 cells
receive different tonic mixtures because the wiring is not mirror-symmetric (PS059 4 cells -95 vs -49; CB0431 +9 vs
+56; LLPC1 +0.6 vs +31.6), and the offsets sum to +66 mV/s in favour of DNa02_L -- yet DNa02_R fires 7x more (0.075
vs 0.011 Hz), so the rate is set by fluctuations, not by the mean. **Caveat on the size of those fluctuations:** the
reconstruction `I = A x brain.rate` filters spikes with `rate_tau` 100 ms, while the real `g` is driven by raw
spikes through a 5 ms synapse, so the mean is exact (a linear functional of mean rates) but the variance is a
smoothed proxy and is a lower bound. Taken literally the reconstruction's temporal sd of 250 / 325 mV/s is
1.25 / 1.63 mV of `g`, which puts -1.6 mV mean 5.5 sd below the +7 mV threshold -- a Gaussian reading of that
cannot produce the observed 0.075 Hz (one spike per 13 s), so the true fluctuations are larger than this table
shows. No verdict here rests on the variance.
`decompose`'s own summary agrees on the mean: `dynamic.plain.DNa02.net -325.8 / -391.5`, largest cancelling pair DNae007 +36.6 vs
PS059 -95.4 (L), CB0431 +56.1 vs IN12B014 -124.1 (R); run sd of every per-type entry <= 1.2 mV/s.

### 4.3 Silent at source (the `room_max_hz` column; `summary.lr_top.*.silent_at_source`)

Of DNa02's 30 strongest wired inputs (by |static weight|), six never exceed 0.5 Hz in any cell of any fly: **PFL3
(+30.6 / +31.9 mV per volley), AOTU001 (+15.6 / +14.7), LAL094 (-11.1), LAL179 (+11.1), PS034 (+10.2), LAL175
(+8.2)** = **6.31 %** of DNa02's absolute static weight (21.4 % of the top 30's). The **12.1 %** that
`summary.lr_top.silent_at_source` reports is a different quantity: the share carried by *all 794* never-firing
presynaptic types of the 3,192-type table (`interp_apply_turning.py` computes it over the whole table, not over
the top 30). Near-silent (mean < 0.02 Hz, a few cells at threshold): AOTU015
(+20.3 / +18.5; 2 of 8 cells ever fire), PS077 (-17.5; 0.014 Hz), PVLP048 (-16.9 / -10.2; 0.000), LAL304m (+15.4 /
+10.6; 0.001), VES202m (-11.1; 0.001), SAD005, VES051 / 052, LAL300m (0.014). LLPC1 (+33.1 / +55.5) is the
in-between case: **63.5 %** of its 104 DNa02-presynaptic cells fire at some point and **36.5 % never do**
(`frac_cells_firing` 0.6346 on the `LLPC1` row; the synapse-weighted never-firing share is a third number, 41.3 %
= 216 of 523), but the per-cell rollout means are 0
for the median cell on both sides and only 1-2 / 45 (L) and 9-10 / 59 (R) exceed 0.5 Hz (runs 0 / 1 / 2 agree). For
the leg MNs the never-firing wired input is the VNC sensory class: SNpp45 (+2.7 / +2.1 mV per volley on 52 cells)
at 0.000 Hz -- **no VNC sensory neuron fires in the room** (10.4 % of the leg MNs' absolute static weight).

### 4.4 Cancellation: is there an asymmetry the sum removes? Yes -- the motion class

Per fly, the total L-R input (sd 248 mV/s) decomposed over the 63 groups every run kept (57 % of its variance
explained, min 47 %; the groups' variances sum to 1.38x the total, so they partly cancel):

| group | var share of total | sd of its L-R (mV/s) | most anti-correlated partner (corr, 3 runs) |
|---|---|---|---|
| HSS/R | +0.128 | 61.3 | LPT22/R (-0.81) |
| LLPC1/R | +0.124 | 65.2 | **LPT22/R (-0.852 / -0.859 / -0.861; sd of the sum 33.8-35.9 against 63-68 each)** |
| IN12B014/L | +0.114 | 85.8 | CB0431/R (-0.83), IN12B014/R (-0.82) |
| LT51/R | +0.053 | 61.9 | |
| PS059/L | -0.058 | 75.9 | |
| LPT22/R | -0.113 | 65.8 | LLPC1/R, HSS/R |
| IN12B014/R | -0.043 | 66.4 | MBON32/L (-0.84) |

So the visual motion class does carry a time-varying lateralised signal into DNa02_R -- LLPC1/R and HSS/R
(cholinergic, +) -- and LPT22/R (GABAergic, -6.4 / -7.2 mV per volley, `nt_sign`) removes 85 % of it, presumably
because LPT22 sees the same front-to-back flow. What survives correlates with DNa02's own L-R (per fly r = +0.259 +-
0.089, |r| null 95th percentile 0.087; with the yaw command +0.280 +- 0.085) -- the readout does follow its input --
but on a cell whose mean input is -326 / -392 mV/s it yields 0.65 Hz of L-R sd and no bias. DNa02 L-R also follows
LLPC1 L-R directly within a fly (median r +0.287 / +0.308 / +0.318 over the three runs, per-fly range +0.044 to
+0.448; generator `interp_apply_turning.py wind-side --per-fly-corr`, section 8). The LLPC1 side asymmetry itself (R 1.65 vs L
0.50 Hz, fly sd 0.07) is **almost all** a fixed offset: regressed on the wind side across the 48 flies it is a
slope of **+0.060 Hz per unit wind side against a fixed -1.14 Hz offset**, i.e. ~6 % of it. (Saying it "does not
follow the wind side" overstates: `per_fly_wind_side_stats.csv` flags `LLPC1_LR` with r = +0.303, p ~ 0.04 and
`consistent_sign_over_runs = True` -- the same acceptance criterion the table uses for DNa13 and DNge037. The
quantitative statement is the defensible one.) DNa02_R also receives 1.7x the LLPC1 weight of DNa02_L (59 vs 45
entries).

For the leg MNs the picture is different: 63 groups explain 84 % of the L-R input variance (sd 58 mV/s), dominated by
VNC premotor interneurons whose L and R cells cancel pairwise (IN12A001/R vs /L corr -0.96, IN16B016 R / L -0.95,
IN16B016/R vs DNge035/L -0.91), and the summed L-R input does **not** predict the leg rate L-R (r -0.059 +- 0.059,
null 0.069) -- the leg asymmetry is set inside the VNC recurrence, not by the sum of its inputs.

## 5. (b) atlas: what produces a DNa02 or leg asymmetry at all

Turning atlas (`atlas_turning.json`, 963 populations x 87 readouts, 3 runs, 125 null rows per run; `diff` in Hz
against the null rows, run sd in brackets; verdict `result` = z_floor >= 3 and p <= 0.05 over 3 runs):

| readout | movers (not the readout itself) |
|---|---|
| `turn_LR` (DNa02 L-R) -- 180 of 963 populations give a result, 148 with |diff| > 1 Hz | **LLPC1_R -119.2 (0.6) / LLPC1_L +99.4 (2.1)** [ipsilateral]; **PFL3_R +45.4 (2.3), +47.1 (4.0, `#2`) / PFL3_L -45.1 (1.8), -46.5 (1.5)** [contralateral]; AOTU015_L +27.5 (2.1) / R -25.9 (1.6); AVLP749m_L +24.2 / R -22.9; LAL304m_R +20.5 (4.3), +16.6; AOTU001_R +18.3 / L -11.8; AN04B003_L +18.0 / R -17.8; LAL179_R +16.5 / L -15.5 (x2); LAL300m_R -13.8 / L +11.0; LAL030_a_L +12.4 / R -10.1; SAD005_R -12.4; LAL301m_R -12.2 / L +11.6; LAL144_L +11.2; PS034_R -10.6; aIPg1_L +9.9; PS230_L +4.2 (0.5) / R -2.5 (0.6); DNae007_L +4.3 / R -4.7; DNpe024_L +3.4 / R -1.8; ExR8_L +1.5; hDeltaI +1.3 / +1.6 (both sides); hDeltaA_L +1.2; PFL2_L +0.7 |
| `turn_LR` = 0.000 (null, all three runs) | EPG, **PEN_a**, PEG, Delta7, every ER, ExR1-7, every FC / FS, PFL1, hDelta B-H / J-M, vDelta A-J / L-M, LT51, PS077, PS059, GNG562, LAL083 / 094 / 126 / 124 / 040 / 121 / 018, DNa13, DNge035 (the inhibitory ones are floor-limited, section 2). **Not** in this row: hDeltaI, hDeltaA_L, vDeltaK_R, ExR8 and PEN_b(PEN2)_L, which reach `result` at 0.6-1.6 Hz (previous row) |
| `leg_LR` -- 138 results, 19 with |diff| > 1 Hz | DNge035_R +7.10 (0.66) / L -6.42 (0.51) [contralateral]; DNa13_L +2.52 / +2.96; DNge037_R +2.57 / L -1.71; IN04B074_L +2.44 / R -2.29; DNa02_L +2.43 / R -1.63; DNge049_R +2.25 / L -1.14; DNge073_R +1.50 / L -1.42; IN04B081 +-1.4; LLPC1_L +1.45 / R -0.80; PFL3_R +0.61 |
| `type.DNge035_LR` | LLPC1_R +9.4, LAL194_L +9.1, LAL021_L +7.4, LAL025_L +7.2, WED072_L +7.1, PFL3_R +6.9, LAL161_L +6.6, DNde003_R +6.5 |
| `type.DNa13_LR` | LAL144 +-19, LAL300m +-18, LLPC1_L +18.3, AN08B026 -18 / +15, LAL303m, LAL021, LAL171 / 172, LAL117 |
| DN + sensory atlas (`dn_sensory.json`) | `turn_LR`: DNae007_L +5.47 / R -4.11, DNpe024_R -4.86 / L +2.94, DNde003 +-3.9, DNg09_a_R +2.43; `channel.JO_wind_left / right` 0.000 / 0.000 on `turn_LR`, +0.03 / -0.08 on `leg_LR` while flipping DNp18 / DNp33 by +-45-49 Hz; every visual class 0.000 (its output is pruned with the optic rate units) |

Reading. The visual motion cell LLPC1 is the strongest DNa02 mover in the whole model and the only strong one that
fires in the room; PFL3 is the dominant CX gate at +45-47 Hz, but not the sole one -- 8 of the 199 non-PFL3 CX
populations reach `result` at 0.6-1.6 Hz (hDeltaI both sides, ExR8 both sides, hDeltaA_L, vDeltaK_R, PFL2_L,
PEN_b(PEN2)_L), which through `k_turn` is 3-8 deg/s, the order of the plain fly's whole yaw-command SD.
AOTU015 / AOTU001 (object) and the LAL / AN04B003 / PS230 interneurons
follow at 4-28 Hz per 150 Hz volley. The same two classes (LLPC1_R +9.4, PFL3_R +6.9) also move the strongest leg
driver, DNge035. The compass' columnar stages are 30-70x weaker than PFL3 at DNa02, so the compass route is
effectively PFL3 or nothing -- but it is a size statement, not a connectivity one.
Both `turn_LR` and `leg_LR` are excitation-only here (null exactly 0.000, section 2): the leg mover table above
is the list of *excitatory* leg drivers, and DNge035 +7.10 / -6.42 should be read as such.

## 6. (c) paths: from the movers back to the CX and the senses, silent links flagged on the room rollout

`interp_paths.paths` k <= 3, type level, mV^k per volley; `never_firing` from `plain_r0_max` (a link is flagged when
its presynaptic cells' max rate over 16 flies x 60 s stays below 0.5 Hz; a fraction = the share of the link's synapses
from such cells).

### 6.1 To DNa02 (never-firing input share 10.7 %, silent 11.5 %, sign-0 0.8 %)

| source | k = 1 | top signed k = 2 / 3 | strongest silent link per k (room) |
|---|---|---|---|
| CX | **PFL3 +31.2 [never_firing]**; ExR6 -0.13 | PFL2 -> DNa03 -> b +432 [PFL2 never_firing 0.33]; ExR7 -> LAL013 -> b +336; PFL2 -> LAL076 -> LAL126 -> b +2.9e4 [never_firing 0.32 / 0.57] | k1 PFL3 -> b (736 syn); k2 / k3 **PFL3 -> LAL121 [never_firing] +201 mV per volley if signed, 2,671 syn = 37.4 % of LAL121's input** on PFL3 -> LAL121 -> (LAL010 ->) b |
| LAL | LAL304m +13.0 [never_firing 0.51]; LAL083 -10.3 | LAL302m -> VES074 -> b +316; LAL302m -> PS049 -> b -261; LAL083 -> PS019 -> PS059 -> b +9,548 | k1 LAL094 -> b [never_firing] -10.9 (257 syn); k2 LAL179 -> PS013 [never_firing] +26.9 (418 syn); k3 PFL3 -> b on LAL152 -> FB5A -> PFL3 -> b |
| visual projection | **LLPC1 +44.3 [never_firing 0.41]**; LT51 -27.2 (live) | LC10a -> AOTU025 -> b +2,009; LPLC4 -> PS306 -> b -1,992; LC10d -> AOTU041 -> AOTU005 / AOTU027 -> b -1.2e5 | k2 **AOTU015 -> b [never_firing] +19.4 (572 syn) on LC10d -> AOTU015 -> b**; k3 AOTU015 -> DNae001 [never_firing] +35.8 |
| JO (wind) | none | **JO-EV5 -> PS230 -> b +152 (no silent flag: live)**; JO-FV -> GNG515 -> b -90; JO-EV3 -> AMMC012 -> PS261 -> b +1.8e4; JO-EV1 -> SAD004 -> CB0540 -> b -1.4e4 | k2 OA-VUMa1 -> b [sign0] +3.7 (88 syn); k3 AOTU015 -> b on JO-EV2 -> pIP1 -> AOTU015 -> b |
| ORN | none | ORN -> (untyped) -> b +1.4; ORN_DA1 -> AL-AST1 -> GNG284 / VES071 -> b -4.2e4 / +4.1e4 | k2 PPM1201 -> b [sign0]; k3 AOTU015 -> b |
| VNC sensory | SNpp45 +2.0 [never_firing] | SNpp39 -> AN04B003 -> b +221 [SNpp39 -> AN04B003 never_firing 0.99]; LgLG6 -> AN03A008 -> b +97 | k2 SNpp45 -> IN19A013 [never_firing] +53 (879 syn); k3 SNpp50 -> IN19B035 [never_firing] +66 (960 syn) |

The CX reaches DNa02 with signed weight only through PFL3, and PFL3 never fires in the plain fly; its largest
downstream target in the LAL, LAL121 (37 % of whose input is PFL3), is therefore also unfed. The object route dies
at AOTU015 (never fires; LC10d -> AOTU015 is the largest two-step gain on the table, +4,645 mV^2, and is inert). The
motion route is live but 41 % of LLPC1 -> DNa02 comes from cells that never fire. The wind route is live at every
link. The VNC sensory route is dead at its source (SNpp39 / 45 / 50 never fire), which silences the ascending
excitatory input AN04B003 (+15 mV per volley; 0.48 Hz in the room).

### 6.2 To the leg drivers (`paths_<source>_to_<DN>.json`; never-firing input shares DNge035 0.4 %, DNa13 2.8 %, DNge037 4.4 %, DNge049 4.1 %, IN04B074 7.8 %)

| mover | CX | LAL | visual projection | JO | VNC sensory |
|---|---|---|---|---|---|
| DNge035 (leg L-R +7.1 / -6.4) | no k = 1; PFL2 -> DNa13 -> b +8.3; PEG -> LAL195 -> b +3.6; ExR7 / EPG -> LAL190 -> CB0609 -> b -2.3e4 / -1.9e4; silent: PFL3 -> LAL018 [never_firing] | LAL018 +0.25, LAL195 +0.14; LAL190 -> CB0609 -> b -558; LAL193 -> VES089 -> b +355 | no k = 1; LC31a -> PVLP137 -> b (round-1 structural); silent GNG702m -> b [sign0] | JO-EV3 -> GNG575 -> b -66 | SNta02 / 09 -> AN17A003 -> b (structural); silent SNpp21 -> AN02A001 [never_firing] +42.5 |
| DNa13 (+2.5) | PFL2 +1.65; PFL3 -> LAL083 -> b -1,185; silent PFL3 -> LAL121 [never_firing] +201 | LAL021 +17.5; silent LAL082 -> b [sign0], LAL060_a -> AOTU025 [never_firing] | LT51 -48.9; LPC1 -> PLP012 -> b +3,126; silent AOTU015 -> b [never_firing] +10.4 | JO-FV -> GNG515 -> b -97; silent OA-VUMa1 -> b [sign0] | SNpp10 -> AN02A002 -> b -1,094; silent SNpp45 -> IN09A004 [never_firing] +44 |
| DNge037 (+2.6) | no k = 1; PFL3 -> DNb01 -> b -448; silent PFL3 -> VES054 [never_firing] +186 | LAL206 -2.8; LAL173 -> VES064 -> b -571; silent LAL027 -> b, CRE014 -> b [never_firing] | LT51 -1.5; LPLC4 -> DNbe007 -> b +1,538; silent CRE014 -> b, AOTU015 -> GNG562 [never_firing] | JO-FV +1.2; JO-FV -> DNge056 -> b +270; silent DNx01 -> b [never_firing] | SNta42 -> ANXXX024 -> b +1,032; silent SNpp21 -> AN02A002, SNpp45 -> ANXXX041 [never_firing] |

The pattern is the one section 6.1 shows at DNa02: the CX reaches every leg driver only through PFL3 / PFL2 (silent
in the room), the object route through AOTU015 (silent), the VNC sensory route through never-firing SN types; the
live inputs are visual (LT51 -49 on DNa13, LPLC4, LPC1 -> PLP012) and the wind's GNG / DNge interneurons.

### 6.3 What the never-firing flags add over the structural round

Three populations that the structural tables ranked as DNa02's strongest routes are now known to be silent in the
plain fly for the whole rollout: PFL3 (24 cells) and with it LAL121 / LAL179 / LAL094 / PS034 / LAL175; AOTU015 and
AOTU001 (the LC10 -> AOTU object route); the VNC sensory neurons (SNpp39 / 45 / 50, LgLG) and through them the
ascending AN04B003 / AN03A008 (49 % / 28 % never-firing shares of their links onto DNa02). The last is a mechanism
gap of the body model, not of the connectome: the VNC interneurons that inhibit DNa02 (IN12B014 14 Hz, IN19A003
12 Hz; and the leg premotor INs at 14-37 Hz) run with every proprioceptor at 0 Hz -- the walking VNC is open-loop.

## 7. What turning would require; data-driven vs hand-crafted

The turning readout is fed by three lateralised excitatory classes and one wind route at DNa02, and by six
descending leg drivers; the localization per class is:

* **DNa02's operating point (new).** Net input -326 / -392 mV/s = **-1.63 / -1.96 mV** of steady `g` against a
  **7.0 mV** threshold gap, from sign-correct GABA / glutamate inputs at 3-14 Hz
  (VNC 12B / 19A interneurons, GNG562 / GNG284, PS059, LT51, MBON31 / 32, LAL120_a, PLP029). Any lateralised signal
  has to beat this. **Data-driven question:** are those tonic rates right? The VNC interneurons run open-loop
  (section 6.3) and the GNG / PS cells' rates have no reference in the ledger; a `health` pass on them and a
  proprioceptive drive for the VNC sensory neurons (a transducer of the body model, like the JO wind model) are the
  mechanism-level items. **Hand-crafting:** a bias, noise term or threshold change on DNa02.
* **Motion class (T4a / T5a -> LLPC1 / HSS -> DNa02, +; LPT22 -> DNa02, -): live, lateralised, cancelled and
  under-driven.** 36.5 % of the DNa02-presynaptic LLPC1 cells never fire (41.3 % of the synapses) and the median
  firing cell sits at 0 Hz; the rest carry a signal that LPT22 removes
  at corr -0.86. **Data-driven:** `trace` from T4a / T5a to LLPC1 and LPT22 in the room (why the median LLPC1 cell
  is silent; whether LPT22's anti-correlation is the same flow signal with a GABA label -- its `nt` is gaba in the
  data, tier `nt_sign`, no receptor row); nothing to tune unless the data on LPT22's transmitter or LLPC1's inputs
  changes. **Hand-crafting:** a gain on LLPC1 -> DNa02 or an ablation of LPT22 as a default.
* **Compass class (PFL3 -> DNa02, PFL3 -> LAL121 / LAL126 / LAL083): wired, functional when driven (+45 Hz at
  150 Hz; 24 Hz at 80 Hz, `interp_atlas.md` 3.3), at 0.000 Hz in the plain fly.** Eight other CX populations also
  move DNa02, by 0.6-1.6 Hz (section 0.2) -- 30-70x weaker than PFL3 but not zero, and each is silent in the room
  for the same reason. The ring is silent without the
  experiment gains and even with them PFL3's L-R is a fixed gradient (`compass_room.md`). **Data-driven:** the
  rotation input to PEN (`deficit_rotation.md`: the GLNO sign, the efferent PS196_b / LAL route). **Hand-crafting:**
  the EPG <-> PEN gains as a default, or a module that writes PFL3.
* **Object class (LC10a / d -> AOTU015 / AOTU001 -> DNa02): wired with the largest two-step gains, silent at
  AOTU015 (0.000 Hz).** Upstream cause `deficit_object.md` (the figure dies at Tm5Y / TmY21 and T2 / T3). No
  data-driven fix in the receptor / NT tables; **hand-crafting** would be an optic object measure on LC10 / AOTU015.
* **Wind class (JO -> PS230 -> DNa02; JO -> GNG515 -> DNa02): live, lateralised, ~90x under dose** (0.078 mV of
  steady `g` against the 7.0 mV threshold gap). The wind side
  predicts the PS230 L-R input to DNa02 across flies (r -0.61, 3 / 3 runs, -15.5 mV/s per unit wind side) and the
  leg L-R (r -0.65; 0.07 Hz = 0.2 deg/s), not DNa02's rate. **Data-driven:** the JO transducer's calibration
  (JO-C / E firing rate vs wind speed against measured responses -- a sensory calibration of the body model that
  `screen_steering.py`'s wind sites already parametrise), and a `trace` from the JO through PS230 / GNG515 with the
  JO rates as source. **Hand-crafting:** the anemotaxis program (exists as a swappable module) or a gain on
  PS230 -> DNa02.
* **Leg class (DNge035, DNa13, DNge037, DNge049, DNge073 -> leg MNs): fire, carry fixed L-R offsets (+4.4, +0.8,
  -2.2 Hz), unread.** `body.py` reads the leg MNs, whose L-R (+0.19 Hz) is set by VNC recurrence and follows DNge035
  (median r -0.43 / -0.47 / -0.47) and the wind side (r -0.65) weakly. Reading these DNs directly would be a readout-definition change of the
  body model (a documented stop-gap either way); giving them a drive would be hand-crafting.

So: turning would require a lateralised drive on LLPC1 / HSS (motion), AOTU015 (object), PFL3 (compass) or PS230
(wind) large enough to cancel ~350 mV/s of tonic inhibition (-1.6 / -2.0 mV) *and* supply the 7.0 mV threshold gap
(~1,400 mV/s more). The model has every route wired with the
right sign and side; it lacks the signal on three of them for documented upstream reasons and has the fourth (wind)
at ~1 % of the required dose (0.078 mV against 7.0). The data-driven programme is three measurements (`trace` T4a -> LLPC1 / LPT22, `health`
of the open-loop VNC, the JO calibration) and one mechanism (proprioceptive drive of the VNC sensory neurons);
nothing here argues for changing a weight, a gain or `body.py`'s readout.

## 8. Statistics, tool defects and caveats

* Replicates are runs (3 seeds, one batch); per-fly numbers carry their scatter over 48 flies and the 3 run means;
  every per-type room input has run sd <= 1.2 mV/s and every L-R offset the same sign in 48 / 48 flies. `compare`
  cannot say `result` at n = 3 (`interp_decompose.md` 5.2); the atlas' `z_floor` verdicts are quoted with their run
  sd. Correlations are per fly against a 1 s block-shuffle null (100 shuffles).
* **The GPU rollout is not bit-reproducible at a fixed seed, and the three seeds are not a determinism claim.**
  Re-running `record --seed 0` verbatim on the cluster gives leg L-R **+0.208** against the original **+0.190**
  (~10 %) and a printed yaw-rate SD of 7.54 against 9.42 (~20 %), while DNa02 L-R moves only -0.064 -> -0.063 and
  the aggregate quantities that carry the verdict move ~0.3 % (net input -326.4 -> -327.0 and -387.5 -> -388.7
  mV/s). Every conclusion here rests on the aggregates; the small L-R readouts should be quoted as ~10 % quantities.
* The atlas' verdict floor: every null arm in `atlas_turning.json` is exactly 0.000 Hz with SD exactly 0, so
  `z_floor = diff / 0.05` and `result` means `|diff| >= 0.15 Hz`, not `moved meaningfully`. The eight sub-2 Hz CX
  movers of section 0.2 and most of the 138 `leg_LR` results sit near that floor; quote the `diff`, not the verdict.
* The `record` rollout uses the fence (the fly stays on the table); its yaw-rate SD (8.3 deg/s) is dominated by the
  fence's turns (1.2 % of frames at the table edge, ~90 deg/s) and is not comparable with `probe_walk_straightness`'s
  2.45-2.57 (no fence) until they are removed (2.56). The yaw command SD (3.5 deg/s) is the brain's own signal.
* Atlas floor: DNa02's *and* the leg MNs' nulls are 0.000 Hz, so inhibitory movers of `turn_LR` and of `leg_LR`
  are equally invisible (section 2). The atlas
  stimulates at 150 Hz for 400 ms, a dose comparison across populations of different size (`interp_atlas.md` 6.3).
* **Defects fixed in this revision** (all in `scripts/interp_apply_turning.py`; no shipped JSON was regenerated, so
  `turning.json` still carries the pre-fix values noted above). (i) `cancellation_tables()` copied DNa02's readout
  SD into the `leg_L|leg_R` summary: `summary.cancellation['leg_L|leg_R'].sd_DNa02_LR_hz_mean` reads 0.6452, which
  is DNa02's number; the leg L-R temporal sd is **0.419** Hz. The summary now reports the pair's own readout as
  `sd_readout_LR_hz_mean` (and names it in `readout_LR`). (ii) `lr_tables()` took no `--skip`: its totals row now
  drops the start-up like every other table (the per-type entries cannot, section 2). (iii) The two per-fly
  correlations quoted in sections 4.4 and 5 -- `corr(LLPC1 rate L-R, DNa02 L-R)` per run and
  `median corr(DNge035 L-R, leg L-R)` -- had no shipped generator, against the round's rule that every quoted
  number ships its generator; `wind-side --per-fly-corr` now emits both as the `per_fly_readout_corr` table
  (`+0.287 / +0.308 / +0.318` and `-0.430 / -0.471 / -0.473`, reproducing the values quoted here from
  `out/turn/plain_r{0,1,2}.npz`). (iv) A CPU test class `ApplyTurningTests` covers the three of them in
  `tests/test_interp.py`.
* **Defects fixed in the previous revision.** (i) `scripts/interp_apply_turning.py`: the mover list built from the atlas broke on
  the atlas' `#n` duplicate labels (`DNa13_L#2` -> an empty population, six wasted `paths` calls in the first
  `analyse`); the suffix is now stripped and a mover must resolve to >= 1 cell; the L-R tables gained the room mean /
  max rate and firing fraction per presynaptic group (`room_mean_hz`, `room_max_hz`, `frac_cells_firing`,
  `silent_at_source`). (ii) Skeptic defect 2 (atlas): `scripts/interp_atlas.py run` now accepts `--by-side` (the
  documented flag of `docs/INTERP.md` 7; a no-op alias of the default, `--no-by-side` pools) -- `flyverse/interp/atlas.py`
  was not touched. The `interp_atlas.md` 3.1 sentence "half of DNp73's flip is the offset" is wrong as the skeptic
  says (a fixed head-on offset cancels in a difference of differences) and is not this task's file; the atlas rows
  used here are differences against the null rows of the same batch, where the offset does cancel.
* Observed, not fixed (not this task's files): `common.links` reports `synaptic_pair_count 0.0` on signed entries
  (e.g. LPT22 -> DNa02) -- skeptic defect 1's root cause in `common.raw_counts`; the `paths` tool uses its own
  `raw_counts` and its counts here are right (PFL3 -> DNa02 736). The GPU JSONs' `flyverse_commit.commit` reads
  `unknown` because the cluster run directory is not a git checkout; the submitting checkout was `e168bd0` plus the
  untracked `flyverse/interp` tree, and the weights are identified by `ew.md5 ed1df661716d240b0f9289607f95320c` /
  compiled `W` `ef23cc27bea13be7f6a96f3c04fd3737` in every file.
* The `static_dna02*.json` `silent_entries` column reads every entry as `never_firing` (the `common.links` NaN-as-True
  bug, `interp_paths.md` 3); the room-judged flags of section 6 supersede it.

## 9. Files

| file | what |
|---|---|
| `scripts/interp_apply_turning.py` | record / atlas-run (GPU), plan / analyse (CPU) |
| `out/interp/apply_turning/batch.sh`, `out/turn_cluster.log` | the one-batch cluster line and its console (`6 job(s), 0 failed`) |
| `out/turn/plain_r{0,1,2}.{npz,json,txt}`, `plain_r*_flies.*`, `plain_r*_max.*` | the three room recordings (16 flies x 60 s each; device cuda) |
| `out/turn/atlas_r{0,1,2}.{npz,json,txt}` | the three turning-atlas runs (963 populations x 87 readouts) |
| `out/interp/apply_turning/turning.json`, `turning_analyse_clean.txt` | `analyse`'s summary + 17 tables, and its printed tables |
| `out/interp/apply_turning/decompose_{DNa02_L,DNa02_R,leg_L,leg_R}_{type,type_side}.json` | the `decompose` Results (frames = flies; `check()` empty) |
| `out/interp/apply_turning/atlas_turning.json` | the `atlas` Result (62 MB) |
| `out/interp/apply_turning/paths_{cx,lal,visual_projection,wind_JO,ORN,vnc_sensory}_to_{DNa02,DNge035,DNa13,DNge037,DNge049,IN04B074}.json` | the `paths` Results with room never-firing flags |
| `out/interp/apply_turning/pre_paths_*.{json,txt}`, `static_dna02{,_L,_R}.json`, `static_legMN.json`, `static_dna02_drivers_inputs.json` | the structural round (CPU, no recording) |
| `out/interp/apply_turning/per_fly_wind_side.csv`, `per_fly_wind_side_stats.csv` | the 48-fly wind-side table of section 0.4 (`interp_apply_turning.py wind-side --recordings "out/turn/plain_r*"`; also `turning.json` tables `per_fly_wind_side` / `wind_side_per_readout` when `analyse` is rerun) |
