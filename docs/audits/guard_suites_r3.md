# Regression guards, behaviour round 3 / anti-runaway round 7: the 29-check suite, the adopt-alone suites and the room take-off protocol on this round's boxes

No model change. One `cluster_run.py` submission (`scripts/guard_suites.sh`; run dir `/root/runs/guard7-97ce35` on the
rented H200 boxes r3-h200a / r3-h200b; console `out/guard_r3_cluster.log`; artefacts `out/guard_r3/`), **27 jobs, 0
failed**, two `--arm-block` blocks: `fam_pinned` (15 jobs, all on r3-h200a) and `fam_room` (12 jobs, all on r3-h200b).
The submitting client died with its session; the boxes were stopped for ~5 h (funds) and restarted, the two jobs that
were running at the stop (`room_proprio_2`, `room_proprio_3`) were retried by the coordinator and finished; every job's
JSON was fetched by hand (`scripts/fetch_run.py`), counted against the plan (15 + 12) and checked for its device.
**Every one of the 27 runs records NVIDIA H200, torch 2.11.0+cu128, compiled-connectome md5 `ef23cc27...`** (the
shipped cache) and, for the **18** runs that go through the wrapper (6 hops + 12 room; the other 9 are the suite jobs,
which call `scripts/retire_measures.py` directly and carry no guard block -- `scripts/guard_suites.sh` lines 228-245
generate the 9 / 6 / 12 split), `guard.problems = []` (the override it was asked for is the one the built simulator
carries). An earlier version of this sentence said 21; the substance holds, the count did not. The aggregate tables are `out/guard_r3/guard_report.md` /
`guard_summary.json` (`bash scripts/guard_suites.sh --report`, CPU).

**What the boxes ran.** `cluster_run.py` ships the working-tree diff at submission (`out/guard_r3/submit_tree.txt`,
2026-09-14 05:19 UTC, HEAD `d2abf3c`): `flyverse/body.py` (+11 lines at that moment: the body-state thread's `LegCycle`
class and five `FlyState` fields with static defaults, opt-in, nothing reads them unless a `LegCycle` is attached) plus
three scripts. Read back over ssh from the run directory on both boxes: `flyverse/senses.py`, `batch_body.py`,
`motor.py`, `batch_sim.py`, `brain.py`, `optic.py`, `fly.py`, `scripts/benchmark.py`, `batch_sustain.py`,
`retire_measures.py` are **HEAD's bytes** (md5 equal to `git show HEAD:<file>`), `body.py` is HEAD + the inert
`LegCycle` (md5 `46e3c10a`, neither HEAD's nor the present tree's -- the body thread has since extended it). So the
"transducer on" arms are the **round-2 `senses.Proprioception('all')`** (chordotonal + hair plate + campaniform +
haltere, `proprioception_channels` in every such JSON), not the round-3 sided version now in the working tree.

## 1. The 29-check suite (`scripts/retire_measures.py --sections all --seeds 0,1,2 --timeout 60`, x3 independent draws per arm)

Arms: `baseline` (the shipped default), `no_drive_clip` (`OpticParams.drive_clip_mv` 35 -> 1e9), `pair_gain_lpi_x1`
(`optic.DEFAULT_PAIR_GAIN`'s LPi34/43 -> LPLC2 factor 4 -> 1, every other pair gain kept). Files
`out/guard_r3/fam_pinned/suite_<arm>_<draw>/<cfg>.json` (+ `.log`, `comparison.json/.md`, the job's `.txt`). Receptor
model `sign` / `abs`, native backend (`cuda_kernels, cuda_graphs, event_driven, warp`), 5.9-6.5 min per run.

**Tally, every arm, every draw: 27 PASS / 0 FAIL / 2 KNOWN GAP / 0 MISSING** (the two gaps are `object.LC10a_flip_hz`
and `compass.wedge_cells_persisting`, the same two in all nine runs). **No check is worse in status than the baseline in
any draw of either candidate** (`guard_summary.json: suite_worse = {noclip: {}, lpi1: {}}`).

**"No check worse in status" is an insensitive criterion, and the margins that shrink belong in the record.** Worst
margin-to-bound over the three draws moves `walk.GF_max` 33.37 -> 24.74 (noclip) / 28.20 (lpi1), `walk.power_sustained`
29.89 -> 24.73 (noclip), `motion.min_dsi` 0.1411 -> 0.1371 (noclip), `walk_gf.p99` 12.62 -> 9.13 (lpi1). None crosses a
bound -- which is exactly the point section 3 makes about the room: a candidate can keep every status while moving the
quantity the closed loop is sensitive to.

`walk.power_max_hz` is **reported, not scored** (`scripts/benchmark.py:85`, `Ref(22, "notnone", ...)`, "REPORTED, NOT
SCORED since session 10", per docs/audits/anti_runaway.md round 6); `loom.escape_cm` is the other `notnone` row. Both
count as PASS whenever measured, so a 27/0/2 tally contains them; under the pre-round-6 `< 50` scoring `pair_gain_lpi_x1`
would read 26/1/2 in 3/3 (51.51 Hz), exactly round 4's verdict.

| check (criterion) | default d1 | d2 | d3 | noclip d1 | d2 | d3 | lpi1 d1 | d2 | d3 |
|---|---|---|---|---|---|---|---|---|---|
| rest.spikes_per_step (< 5) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| taste.MN9_hz (> 2) | 10.93 | 10.93 | 10.93 | 10.93 | 10.93 | 10.93 | 10.93 | 10.93 | 10.93 |
| smell.PN_hz (< 100) | 7.861 x3 | | | 7.861 x3 | | | 7.861 x3 | | |
| smell.KC_active (> 0) | 816 x3 | | | 816 x3 | | | 816 x3 | | |
| dn.DNa02_L_leg_asym_hz (> 0.3) | 2.581 x3 | | | 2.581 x3 | | | 2.581 x3 | | |
| dn.MDN_top_hz (< 250) / dn.DNp09_top_hz (< 250) | 153 / 152 x3 | | | 153 / 152 x3 | | | 153 / 152 x3 | | |
| walk.GF_max_hz (< 38) | **4.629162311553955** x3 | | | **13.259908676147461** x3 | | | **9.795085906982422** x3 | | |
| walk.power_max_hz [reported] | **48.48052978515625** x3 | | | **49.24797821044922** x3 | | | **51.50775146484375** x3 | | |
| walk.power_sustained_hz (< 50) | **20.109053071339925** x3 | | | **25.273166004816694** x3 | | | **20.212885443369547** x3 | | |
| loom.GF_peak_hz (>= 20) | 47.22 | 47.22 | 47.22 | 42.68 | 43.41 | 43.41 | 59.21 | 57.40 | 61.36 |
| rotate.DNp20_flip_hz (< -2) | -31.6 | -32.8 | -38.1 | -39.1 | -45.3 | -45.3 | -43.4 | -32.1 | -34.9 |
| motion.min_dsi (>= 0.1) | 0.24109642 | 0.24109674 | 0.24109651 | 0.23711504 | 0.23711505 | 0.23711499 | 0.2471 | 0.2469 | 0.2464 |
| motion.correct_directions (== 8) | 8 x9 | | | | | | | | |
| loom_escape.GF_peak_hz (>= 33) | 46.72 | 48.68 | 49.50 | 47.86 | 53.83 | 50.46 | 74.39 | 57.55 | 68.09 |
| loom_escape.escapes (>= 1) | 3 x9 | | | | | | | | |
| walk_gf.p99_hz (< 38) | 17.22 | 25.38 | 18.32 | 17.98 | 19.68 | 20.83 | 26.57 | 28.87 | 28.01 |
| rotation.group_flip_hz (<= -3) | -9.00 | -9.40 | -10.34 | -10.57 | -10.21 | -8.74 | -10.23 | -9.74 | -9.65 |
| object.LC10a_flip_hz (abs >= 1) KNOWN GAP | 0.0086 | 0.0023 | 0.0037 | 0.0186 | 0.0228 | -0.0070 | 0.0118 | 0.0070 | 0.0124 |
| bitter.* (4 checks) | 5.518 / 0 / 139.9 / 0.818 in all 9 runs | | | | | | | | |
| wind.DNp18_flip_hz (>= 15) | 45.84 | 44.70 | 45.25 | 46.82 | 45.89 | 44.68 | 45.51 | 45.41 | 45.37 |
| wind.DNp33_flip_hz (<= -15) | -49.67 | -49.42 | -50.01 | -50.24 | -49.69 | -49.66 | -49.51 | -49.66 | -50.17 |
| odour.apple_channel_8cm_hz (>= 10) | 17.44 | 17.44 | 17.25 | 17.44 | 17.40 | 17.54 | 17.39 | 17.51 | 17.53 |
| odour.apple_channel_clean_hz (<= 6) | 4.688 | 4.522 | 4.545 | 4.515 | 4.297 | 4.483 | 4.303 | 4.546 | 4.885 |
| compass.wedge_cells_persisting (>= 6) KNOWN GAP | 0 x9 | | | | | | | | |

The full per-draw table with every status letter is `out/guard_r3/guard_report.md` section 1.

Reproduction of the record (**scoped to the native CUDA backend; "bit-identical" is a CPU claim, so what follows is
"reproduced the recorded value exactly in n draws on this backend", with the CPU value quoted beside it**): the
baseline's three walk values **reproduce the recorded value exactly in all 3 draws here and in every prior native GPU
draw** -- 48.48052978515625 / 20.109053071339925 / 4.629162311553955, on B200 and H200 -- while a CPU / eager run of the
same shipped default on this desktop (`out/guard_sk_cpu/walk_cpu.json`, device cpu, backend "eager torch") reads
**57.3838 / 26.6636 / 9.7632**: the triple is a property of the native backend, not a model constant (and on CPU the
shipped default would fail the retired `< 50` bound, which further supports round 6's de-scoring). The round attribution,
corrected: **the baseline's 48.48052978515625 reproduces rounds 4 (as `no_gf_damping`), 5 and 6; `no_drive_clip`'s
49.2480 / 25.2732 / 13.2599 reproduces round 6 and the optic audit (it appears nowhere in rounds 4 or 5);
`pair_gain_lpi_x1`'s 51.5078 / 20.2129 reproduce round 4 and round 6, and its `walk.GF_max` 9.7951 reproduces round 6
and the optic audit ONLY -- round 4 measured 9.7681 for x1, on the pre-retirement default (its baseline GF_max was
4.6061 against 4.6292 now).** As in round 6, the reproduction claim covers those values and no more:
`motion.min_dsi` takes three distinct values across the three baseline draws (0.24109642-0.24109674), and every
`no_drive_clip` draw (0.237115) sits 0.0040 below every baseline draw on file (round 6's statement, unchanged);
`loom.GF_peak` happens to be identical in the three baseline draws here (47.2162) but spanned 43.6-47.2 in round 6.
`pair_gain_lpi_x1`'s `loom.GF_peak` 57.4-61.4 and `loom_escape.GF_peak` 57.6-74.4 are, as before, *higher* than the
baseline's 47.2 / 46.7-49.5 -- the walking / looming giant-fibre drive that the x4 factor suppresses.

### 1b. The suite with the proprioceptive transducer on: not run, because the pinned sections cannot carry it

`senses.Proprioception` is attached only by `BatchSim(..., proprioception=SPEC)` (`flyverse/batch_sim.py:113-118`,
`fly.py:179-183`: "nothing builds one by default"). Only `sec_hops` builds a `BatchSim` at all
(`scripts/benchmark.py:732-738`); the Brain / OpticLobe protocol sections have no body, and the demo sections
(b-e, g, h) run `scripts/room_demo.py`'s single-fly `Sim`, which never sets `proprioception_sense` (no occurrence of
`proprioception` in `room_demo.py`). **And the point is stronger than "only one section": `hops` is not in the 29-check
suite at all** -- `benchmark.py:785` `OPTIONAL = ['hops']` keeps it out of `--sections all` -- so **no section of the
29-check suite builds a `BatchSim`**. A "transducer on" 29-check suite would therefore be the default suite with the
sense inert in all 29 checks -- bit-identical on CPU, within GPU scatter on a box -- and was not run. The transducer
arm runs where it can act: the `hops` section (section 2) and the room protocol (section 3).

## 2. The `hops` section (`scripts/benchmark.py --sections hops`: 16 rooms x 150 s = 2,400 fly-s, brain seed 0, env seeds 0-15; the take-off reference for this round's boxes)

Both arms through the wrapper (`guard_wrap.py hops [--proprioception all]`), so one harness; the transducer arm is
`BatchSim(proprioception='all')` with the round-2 sense (`proprioception_sense_attached: true`, channels campaniform /
chordotonal / hair_plate / haltere). Bounds at run time: voluntary `< 1.0` per 1,000 fly-s (gap row), escape `< 10.0`,
walking-GF median `< 33.0` (`benchmark.py:121-125`).

| arm | draw | hops = escape + voluntary | escape / voluntary per 1,000 fly-s | walking-GF median (Hz) | section tally | wall (hops, s) |
|---|---|---|---|---|---|---|
| default | 1 | 5 = 3 + 2 | 1.25 / 0.83 | 30.05 | 3 / 0 / 0 (voluntary PASS, gap closed) | 1,864 |
| default | 2 | 9 = 2 + 7 | 0.83 / 2.92 | 29.85 | 2 / 0 / 1 (voluntary KNOWN GAP) | 1,981 |
| default | 3 | 10 = 3 + 7 | 1.25 / 2.92 | 31.60 | 2 / 0 / 1 (voluntary KNOWN GAP) | 2,003 |
| proprio | 1 | 3 = 2 + 1 | 0.83 / 0.42 | 28.21 | 3 / 0 / 0 | 3,126 |
| proprio | 2 | 4 = 2 + 2 | 0.83 / 0.83 | 29.43 | 3 / 0 / 0 | 3,131 |
| proprio | 3 | 1 = 1 + 0 | 0.42 / 0.00 | 29.59 | 3 / 0 / 0 | 3,632 |

Pooled: default **24 take-offs in 7,200 fly-s = 3.33 per 1,000 fly-s (exact Poisson 95 % CI 2.14-4.96)**; proprio 8 in
7,200 = 1.11 (0.48-2.19). Round 5's one B200 draw of this section on the shipped default was 3 hops (0.42 voluntary; the
pre-retirement default's was 7, 1.25), so the section's own scatter spans the gap row's bound (voluntary 0.83 / 2.92 / 2.92 here
against `< 1.0`): **the default passes the voluntary row in 1 of 3 draws**, and the row was marked `gap` for that
reason. The transducer arm is not worse than the default on any check in any draw (3/0/0 x3 vs 2/0/1 x2) -- the
regression guard it was run as passes. Its lower count (`compare` over runs: hops 3 / 4 / 1 vs 5 / 9 / 10, diff -5.3, z
-2.0, exact U 0, p 0.10 = the 3 v 3 floor -> **underpowered**; voluntary 1 / 2 / 0 vs 2 / 7 / 7, p 0.2) is a direction,
not a result: four runs per arm in one submission would be the smallest callable design, and the body-state thread's
audit (docs/audits/body_sided_state.md) owns that question. Cost: the transducer arm takes 1.6-1.8x the wall time
(3,126-3,632 vs 1,864-2,003 s) -- the afferent injection runs on the host every frame.

## 3. The room take-off protocol (`scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence`, live escape route, brain seeds 0 / 1 / 2 x env seeds 0-15 / 16-31 / 32-47, 3 x 4,800 fly-s per arm)

The round-5 adopt-alone protocol, all four arms through `guard_wrap.py sustain` on ONE box (r3-h200b), `--cuda-graphs
--cuda-kernels --event-driven --cuda-sparse torch`, escape at GF >= 33 Hz, voluntary at wing power >= 50 Hz held 0.3 s
(`flight` block in every JSON: `gf_hz 33, takeoff_power_hz 50, takeoff_hold_s 0.3`). Files
`out/guard_r3/fam_room/room_<arm>_<k>.json` (`rows` = the 16 flies), the job's `.txt`, `wrap_room_<arm>_<k>.py`.

| arm | batch (brain seed) | hops = escape + voluntary | all / escape / voluntary per 1,000 fly-s | walking-GF median (rows >= 33 Hz) | per-fly walking-GF min-max | wall s |
|---|---|---|---|---|---|---|
| default | 1 (0) | 18 = 5 + 13 | 3.75 / 1.04 / 2.71 | 31.95 (5/16) | 28.2-36.4 | 4,339 |
| default | 2 (1) | 15 = 6 + 9 | 3.12 / 1.25 / 1.88 | 30.46 (5/16) | 28.0-38.6 | 4,331 |
| default | 3 (2) | 12 = 6 + 6 | 2.50 / 1.25 / 1.25 | 30.24 (5/16) | 27.1-39.2 | 4,331 |
| noclip | 1 (0) | 31 = 4 + 27 | 6.46 / 0.83 / 5.62 | 30.99 (3/16) | 26.3-34.5 | 4,339 |
| noclip | 2 (1) | 15 = 5 + 10 | 3.12 / 1.04 / 2.08 | 30.92 (6/16) | 27.9-43.6 | 4,338 |
| noclip | 3 (2) | 27 = 7 + 20 | 5.62 / 1.46 / 4.17 | 32.10 (6/16) | 27.2-38.3 | 4,280 |
| lpi1 | 1 (0) | 179 = 160 + 19 | 37.29 / 33.33 / 3.96 | 38.50 (16/16) | 36.5-59.4 | 4,311 |
| lpi1 | 2 (1) | 163 = 150 + 13 | 33.96 / 31.25 / 2.71 | 38.28 (16/16) | 35.0-58.2 | 4,314 |
| lpi1 | 3 (2) | 156 = 141 + 15 | 32.50 / 29.38 / 3.12 | 37.74 (16/16) | 34.7-45.0 | 4,314 |
| proprio | 1 (0) | 4 = 3 + 1 | 0.83 / 0.62 / 0.21 | 30.79 (3/16) | 27.5-37.5 | 5,042 |
| proprio | 2 (1) | 11 = 8 + 3 | 2.29 / 1.67 / 0.62 | 31.64 (5/16) | 28.6-39.1 | 3,363 (retried after the restart, alone on the box) |
| proprio | 3 (2) | 10 = 7 + 3 | 2.08 / 1.46 / 0.62 | 31.66 (5/16) | 24.0-37.2 | 3,056 (same) |

Pooled over 48 flies / 14,400 fly-s per arm (exact Poisson 95 % CI on the count):

| arm | hops = escape + voluntary | all per 1,000 fly-s (CI) [per-batch min-max] | escape (CI) | voluntary (CI) | walking-GF median over flies (rows >= 33 Hz) |
|---|---|---|---|---|---|
| **default** | **45 = 17 + 28** | **3.125 (2.279-4.181)** [2.50-3.75] | **1.181 (0.688-1.890)** | **1.944 (1.292-2.810)** | 31.27 (15/48) |
| noclip | 73 = 16 + 57 | 5.069 (3.974-6.374) [3.12-6.46] | 1.111 (0.635-1.804) | 3.958 (2.998-5.128) | 31.16 (15/48) |
| lpi1 | 498 = 451 + 47 | 34.583 (31.612-37.758) [32.50-37.29] | 31.319 (28.495-34.348) | 3.264 (2.398-4.340) | 38.12 (48/48) |
| proprio | 25 = 18 + 7 | 1.736 (1.124-2.563) [0.83-2.29] | 1.250 (0.741-1.976) | 0.486 (0.195-1.002) | 31.29 (13/48) |

**The default's take-off reference for this round's boxes** is therefore 3.125 per 1,000 fly-s (CI 2.28-4.18), escape
1.18, voluntary 1.94, walking-GF median 31.27 Hz with 15 of 48 flies at or above the 33 Hz escape threshold. Against the
round-5 record of the same protocol on the same shipped default (B200: 56 = 24 + 32, 3.89 / 1.67 / 2.22, 31.90 Hz,
19/48) the two batches are inside each other's CIs (3.89 sits in 2.28-4.18; 3.125 sits in round 5's 2.94-5.05): the
device change does not show above the protocol's scatter. **The scatter that sentence is about, quoted so it travels
with it**: round 5 (B200) and round 7 (H200) ran the IDENTICAL seeds and got 19 / 21 / 16 against 18 / 15 / 12 flies at
threshold per batch (diff -1 / -6 / -4), and the skeptic's B200 rerun at fresh seeds gives a default of 3.750 per 1,000
fly-s (CI 2.934-4.723) with walking-GF medians 30.79-33.76 against this batch's 30.24-31.95. The same-arm, same-seed,
cross-device spread is roughly a third of the `no_drive_clip` contrast.

**The B200 replication of the decisive arm (skeptic pass), added to the record.** House cluster, run dir
`guardsk-70c2f8` on <cluster-node>, 8 jobs / 0 failed, all NVIDIA B200 / torch 2.11.0+cu128 / cache md5 `ef23cc27` / receptor md5
`0381a446` / identical `LIFParams` / `guard.problems []` / the same wrapper md5 `d283f486`; 4 runs per arm at fresh
seed-matched seeds (brain 3 / 4 / 5 / 6 x env 48-63 / 64-79 / 80-95 / 96-111), ONE submission, ONE arm-block, one node;
artefacts `out/guard_sk/fam_room2/`. Shipped default 14 / 21 / 20 / 17 = 72 = 36 escape + 36 voluntary in 19,200 fly-s =
**3.750** (CI 2.934-4.723); `no_drive_clip` 35 / 25 / 21 / 24 = 105 = 27 + 78 = **5.469** (4.473-6.620). At 4 v 4 the
run-level exact-U floor is 0.0286, so `interp.common.compare` is callable and returns **`hops_voluntary_total` diff
+10.5, z 3.57, U 16.0, p 0.0286 -> RESULT** (voluntary higher in 4 / 4 seed-matched batches), with `hops_total`
p 0.057 `null`, `hops_escape_total` ratio 0.75 p 0.886 `null`, `gf_max_walk_median_hz` p 1.0 `null`,
`rows_gf_at_threshold` p 1.0 `null`. Two-sample exact Poisson: voluntary 78 vs 36, ratio 2.17, p 1.0e-04; all-route
ratio 1.46, p 0.016; escape p 0.31. Pooled H200 + B200 (33,600 fly-s per arm): default 117 = 3.482 (2.880-4.173),
noclip 178 = 5.298 (4.548-6.136), voluntary 4.018 vs 1.905. **This is the first callable run-level verdict on the clip
and it supports "not adoptable".** Note that the rerun shipped the present working tree (threads A's and C's opt-in
edits, all fields off, the bit-identity golden passing), not HEAD.

Seed-matched per-batch counts and the fly-level Mann-Whitney the round-5 tables used (asymptotic; tied counts; 48 v 48
flies -- descriptive, since runs are the replicate unit):

| arm | metric | candidate per batch | default per batch | candidate lower in n of 3 | U | p |
|---|---|---|---|---|---|---|
| noclip | hops | 31 / 15 / 27 | 18 / 15 / 12 | 0 | 1465.5 | 0.017 |
| noclip | escape | 4 / 5 / 7 | 5 / 6 / 6 | 2 | 1129.0 | 0.84 |
| noclip | voluntary | 27 / 10 / 20 | 13 / 9 / 6 | 0 | 1471.5 | 0.012 |
| noclip | walking-GF max per fly (median) | 30.99 / 30.92 / 32.10 | 31.95 / 30.46 / 30.24 | 1 | 1174.0 | 0.88 |
| lpi1 | hops | 179 / 163 / 156 | 18 / 15 / 12 | 0 | 2304.0 | 1.8e-17 |
| lpi1 | escape | 160 / 150 / 141 | 5 / 6 / 6 | 0 | 2304.0 | 6.1e-18 |
| lpi1 | voluntary | 19 / 13 / 15 | 13 / 9 / 6 | 0 | 1381.0 | 0.068 |
| lpi1 | walking-GF max per fly (median) | 38.50 / 38.28 / 37.74 | 31.95 / 30.46 / 30.24 | 0 | 2197.0 | 2.0e-14 |
| proprio | hops | 4 / 11 / 10 | 18 / 15 / 12 | 3 | 872.0 | 0.026 |
| proprio | escape | 3 / 8 / 7 | 5 / 6 / 6 | 1 | 1122.0 | 0.79 |
| proprio | voluntary | 1 / 3 / 3 | 13 / 9 / 6 | 3 | 754.0 | 3.2e-04 |
| proprio | walking-GF max per fly (median) | 30.79 / 31.64 / 31.66 | 31.95 / 30.46 / 30.24 | 1 | 1164.0 | 0.93 |

`flyverse.interp.common.compare` over runs (`guard_report.md` section 3b): with 3 runs per arm the exact-U floor is p
0.10, so every room contrast is `underpowered` by construction whatever its z (lpi1 hops z 50, noclip hops z 3.1,
noclip voluntary z 2.8). The fly-level p's are the round-5 convention, quoted for continuity.

**The rate-half statistic: a two-sample exact Poisson, not CI containment.** An earlier version of this audit decided
the rate-half by asking whether the candidate's pooled point estimate lies inside the BASELINE's exact Poisson 95 % CI.
That test ignores the candidate's own sampling error and is not a 5 % test: simulating two arms with the identical true
rate 3.125 per 1,000 fly-s, the candidate falls outside the baseline's CI **14.9 % of the time at 3 batches per arm,
15.3 % at 4 and 15.6 % at 6 -- the error does not fall with more batches** (skeptic pass). The correct equal-exposure
test is the **two-sample exact Poisson (conditional binomial)**, which is 5 % by construction and reaches every one of
this round's directional conclusions:

| arm vs default (equal exposure, 14,400 fly-s each) | all routes | voluntary | escape |
|---|---|---|---|
| `no_drive_clip` | 73 vs 45, **p 0.0126** | 57 vs 28, **p 0.0022** | 16 vs 17, p 1.00 |
| `pair_gain_lpi_x1` | 498 vs 45, **p 1.1e-97** | -- | 451 vs 17, **p 1.4e-110** |
| transducer 'all' (a guard, not a candidate) | 25 vs 45, **p 0.0225** (fewer) | 7 vs 28, **p 5.1e-04** (fewer) | 18 vs 17, n.s. |

Both statistics are quoted below: the two-sample exact Poisson on the pooled counts, beside the run-level
`common.compare` (`underpowered` at 3 v 3 on the H200 batch, callable at 4 v 4 on the skeptic's B200 rerun).

## 4. Adopt-alone verdicts

**The rule, stated in one direction and with round 7's own sharpening labelled as such.** Rounds 4-6 asked for two
halves: the candidate's own full 29-check suite x >= 3 draws with no check worse in status than the baseline (round 4,
anti_runaway.md:490-491; round 6, :970-972), and the room take-off protocol against the shipped default -- which round 5
read **one-sided**, as "the take-off rates are not worse in either route", with one-sided p's (:551-553). **No round
stated a CI-containment rate-half; that form is new in round 7, it is two-sided, and it was applied asymmetrically** (it
failed both candidates for being outside the default's CI while the transducer, also outside it but BELOW, was passed
under round 5's "not worse" reading). This round therefore states the rate-half once, for every arm, in the two-sided
form and with a real 5 % level: **the two-sample exact Poisson (conditional binomial) at equal exposure, quoted beside
the run-level `common.compare`** (section 3). Read two-sided, the transducer arm also departs from the default -- in the
other direction (fewer take-offs, p 0.0225 all routes / 5.1e-04 voluntary) -- and that is said plainly below rather than
being passed under a one-sided reading. The owner decision on which form the rule takes is recorded in
docs/audits/anti_runaway.md round 7.

* **`OpticParams.drive_clip_mv` 35 -> no clip (`no_drive_clip`): NOT adoptable by the rule.** Suite half: passes
  (27/0/2 in 3/3, no check worse in status -- though see the shrinking margins in section 1). Room half: fails --
  all-route 5.07 per 1,000 fly-s against the default's 3.125 (two-sample exact Poisson 73 vs 45, **p 0.0126**); the
  excess is entirely on the **voluntary** route (3.96 vs 1.94; 57 vs 28 hops, **p 0.0022**; higher in 2 of 3
  seed-matched batches, equal-ish in the third: 27 / 10 / 20 vs 13 / 9 / 6) while the escape route is unchanged (1.11
  vs 1.18, p 1.00) and so is the walking-GF tail (median 31.16 vs 31.27 Hz, 15/48 rows at 33 Hz in both). The skeptic's
  B200 rerun makes it callable at the run level: `hops_voluntary_total` +10.5, z 3.57, p 0.0286 `result` (section 3).
  Round 6's stated prediction for this run ("the clip's removal raises the room walking-GF tail", called weak there) is
  **not confirmed -- the clip does not move the room walking-GF tail at all** (median 31.16 vs 31.27 Hz with 15/48 in
  both arms on the H200; diff +0.15 Hz, `compare` verdict `null`, 25/64 vs 26/64 on the B200); **it binds on the
  wing-power route instead**, consistent with the pinned `walk.power_sustained` 20.11 -> 25.27 Hz (the room's voluntary
  criterion is 50 Hz held 0.3 s) rather than with `walk.GF_max` 4.63 -> 13.26. ("Wrong in direction" would be too
  strong: the predicted route did not move at all, in either direction.) What an adoption would require: a mechanism
  that bounds the optic -> spiking injected current without a hand-set +-35 mV (the clip's stand-in role in section 1
  of anti_runaway.md), *and* **both halves re-run on it** -- the replacement's own full 29-check suite x >= 3 draws,
  *and* this room protocol at **>= 6 runs per arm**. Six, not four: simulating this round's H200 point estimates (15.0
  vs 24.33 hops per 4,800 fly-s run), the exact-U two-sided test at alpha 0.05 has power 0.00 at n = 3, 0.50 at n = 4,
  0.70 at n = 5 and 0.85 at n = 6 -- and the skeptic's own 4 v 4 rerun landed exactly on the floor (p 0.0286) on the
  voluntary route while returning `null` on the all-route count.
* **LPi34/43 -> LPLC2 x4 -> x1 (`pair_gain_lpi_x1`): NOT adoptable, and by a margin the pinned suite does not see.**
  Suite half: passes by status (27/0/2 in 3/3, since `walk.power_max` is no longer scored; `walk.GF_max` 9.80 and
  `walk_gf.p99` 26.6-28.9 both far under their 38 Hz bound -- with `walk_gf.p99`'s worst margin shrinking 12.62 -> 9.13
  and `walk.GF_max`'s 33.37 -> 28.20). Room half: **34.6 take-offs per 1,000 fly-s against 3.125 (11x; two-sample exact
  Poisson p 1.1e-97), 451 escapes against 17 (26x, p 1.4e-110; 4-16 per fly, every fly), walking-GF median 38.1 Hz with
  48 of 48 flies at or above the 33 Hz escape threshold (the per-fly walking-GF minimum, 34.7 Hz, is above the
  threshold), and 0.4-1.6 % of the time airborne (per-batch medians 0.86-0.98 %, mean 0.94 %; recomputed from the 48
  per-fly rows -- an earlier version said "every one of 48 flies 1-2 % of the time airborne", which is wider than the
  data at the bottom and narrower at the top)**. The escape route fires from the fly's own walking. This is the first room measurement of x1 (no
  earlier audit ran it in the room), and it settles two things the pinned scan could not: (i) round 4's "the size of
  the hand-set factor buys nothing that is scored" was true of the pinned checks and false of the room -- the factor
  is what keeps the walking giant-fibre drive under the escape threshold in the closed loop; (ii) G.4's finding that
  the pinned GF numbers do not predict the room now has its sharpest instance: pinned `walk.GF_max` 4.63 -> 9.80 Hz
  (bound 38) corresponds to a room walking-GF median 31.3 -> 38.1 Hz across the 33 Hz threshold. What an adoption
  would require: not "x1" but a sign-correct LPi -> LPLC2 strength from data (round 3's replacement statement, still
  the honest one), scored through **both halves** -- the replacement's own full 29-check suite x >= 3 draws *and* this
  room protocol at >= 6 runs per arm; any candidate must be run through the room before the pinned suite is taken as
  evidence.
* **The proprioceptive transducer ('all', round-2 sense) as a regression guard on the shipped default**: no check
  worse in status in the `hops` section (3/0/0 x3 vs the default's 2/0/1 in 2 of 3), room take-off rate 1.74 per 1,000 fly-s (25 = 18 + 7) against the default's 3.125 -- **below** the default's CI (2.28-4.18), not above it: the escape route is unchanged (1.25 vs 1.18) and the voluntary route is a quarter of the default's (0.49 vs 1.94; 7 vs 28 hops; 1 / 3 / 3 vs 13 / 9 / 6 per seed-matched batch, lower in 3 of 3, and 1 / 2 / 0 vs 2 / 7 / 7 in the `hops` section, lower in 3 of 3 -- six of six paired comparisons in the same direction, walking-GF median unchanged at 31.29 vs 31.27 Hz).
  **Read under the same two-sided rate-half the candidates are read under, this arm also departs from the default** --
  two-sample exact Poisson 25 vs 45 all routes **p 0.0225**, 7 vs 28 voluntary **p 5.1e-04**, escape n.s. -- in the
  direction of FEWER take-offs. By the status half fewer take-offs is not "worse" on any check; by INTERP's rule the
  run-level contrast is `underpowered` (3 v 3, room voluntary diff -7, z -2.0, exact U 0, p 0.10 = the floor). So it is
  a direction on record for the body-state thread, not a result -- and it is stated here symmetrically with the two
  candidates rather than being passed under a one-sided "not worse" reading while they are failed under a two-sided one.
  Whether the rule's rate-half is one-sided (round 5) or two-sided (this round's sharpening) is an owner decision; the
  transducer arm and both retirement candidates are to be re-read under whichever form is chosen, in the next guard
  round. No number here changes either way.
  It is opt-in and stays off; its adoption question (as a body-model mechanism, not a gain) is the body-state thread's.

**Nothing is adopted here; the owner decides** (docs/audits/anti_runaway.md round 7 carries the decision record).

## 5. Caveats

1. H200 only, both blocks; the GPU path is not run-to-run reproducible at a fixed seed, so **nothing here is a
   bit-identity claim** -- "bit-identical" is a CPU claim under the project rule. The three walk values reproduce the
   recorded value exactly in every native GPU draw on record (B200 and H200) and read 57.3838 / 26.6636 / 9.7632 on the
   CPU: a backend property (section 1). The house rerun (B200) that the skeptic pass ran is a replication across
   devices, not a bit-for-bit check, and is recorded in section 3.
2. Three draws / batches per arm on the H200 batch, four on the B200 rerun. The status-half of the adopt-alone rule is
   a 3/3 criterion and is met by both candidates; the rate-half is the two-sample exact Poisson on pooled counts
   (section 3), **not** CI containment, which has a ~15 % false-failure rate for a candidate identical to the default
   and does not improve with more batches. The run-level `compare` verdicts are `underpowered` at 3 v 3 by construction
   (p floor 0.10) and callable at 4 v 4; the fly-level p's are not the replicate unit. For a callable run-level verdict
   on the clip the prescribed design is **>= 6 runs per arm** (power 0.85 at n 6, 0.50 at n 4 for this contrast), not
   the ">= 4 batches" an earlier version asked for.
3. The `hops` section's voluntary row (`< 1.0` per 1,000 fly-s, gap) passes in 1 of 3 default draws and fails in 2;
   the room protocol (2x the flies, 2x the time, 3 brain seeds) is the measurement the verdicts rest on, the section
   is the same-box reference for it.
4. The 29-check suite with the transducer on is not a measurement (section 1b); the suite's own `hops` section and
   the room protocol are where the sense acts.
5. The batch ran HEAD `d2abf3c` + the inert `LegCycle` in `body.py` (section 0); the other threads' round-3 edits
   (`senses.py`, `motor.py`, `batch_body.py`, `brain.py`, `optic.py`, `fly.py`, `batch_sim.py`,
   `retire_measures.py` in the present tree) were **not** on the boxes. The CPU smoke of the wrapper
   (`bash scripts/guard_suites.sh --smoke`, `out/guard_r3/smoke/`) passes against the present tree (three JSONs,
   `problems []`, the requested clip / LPi / transducer resolved on the built simulator, device cpu).
6. Round 6's bookkeeping defect (`retire_measures.py`'s `provenance.execution.device` read `null`) is gone: every
   suite JSON here reads `execution.device: "cuda"`, `device_name: "NVIDIA H200"`, with the same name in
   `config.device`; the wrapper's `guard` block carries `device_name` from `torch.cuda.get_device_name(0)`.
   **The API fact, stated correctly**: `batch_sustain.py` DOES write a top-level `device: "cuda"` in the room JSONs --
   what it does not write is a top-level **`device_name`**, which is why the aggregator takes the name from the
   wrapper's guard block. (An earlier wording here and the matching comment in `scripts/guard_suites.sh`'s section-3
   loader said "the room JSONs carry no top-level device".) `flyverse_commit` is `unknown` in every cluster JSON (no `.git` in the run copy),
   as in every other cluster Result; the identity is the cache md5 + source fingerprint.

## Files

`scripts/guard_suites.sh` (the generator: the 27 job lines, the embedded `guard_wrap.py`, `--smoke`, `--report`),
`out/guard_r3_cluster.log` (the console until the client died: the 27 queued lines, the first completions),
`out/guard_r3/submit_tree.txt` (the tree at submission), `out/guard_r3/guard_wrap.py` (the wrapper as shipped),
`out/guard_r3/fam_pinned/` (9 suite directories, 6 hops JSONs + `.txt` + `wrap_*.py`), `out/guard_r3/fam_room/` (12
room JSONs + `.txt` + `wrap_*.py`), `out/guard_r3/guard_report.md` / `guard_summary.json` (every table above),
`out/guard_r3/smoke/` (the CPU smoke against the present tree), `out/guard_r3/smoke_predecessor/` (the same smoke at
submission time). Attach / fetch by hand: `scripts/box_status.py`, `scripts/fetch_run.py`.
`out/guard_r3/FETCHED.txt` says "Local file count now: 111"; there are **112** files under `out/guard_r3/`.
Skeptic-pass artefacts, not this thread's: `out/guard_sk/fam_room2/` (the B200 rerun `guardsk-70c2f8`, section 3) and
`out/guard_sk_cpu/walk_cpu.json` (the CPU walk triple, section 1).
