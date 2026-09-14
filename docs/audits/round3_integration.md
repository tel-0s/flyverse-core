# Round 3 integration: the sided transducer, the monoamine slow class and the per-transmitter unitary combined on the plain fly and the two compass protocols (thread integrate, behaviour round 3)

Generator: `scripts/probe_round3_integrate.py` (`arms` / `plan` / `verify` / `analyse` CPU; `room` / `compass` / `wedge` /
`bench` GPU). Data: `out/r3int/` (ONE submission `r3int-9a78e4`, house cluster <cluster-node>, 8 x NVIDIA B200, **51 job(s), 0
failed (67.6 min)**; console `out/r3int_cluster.log`; predeclaration `out/r3int/predeclared.json` stamped
`2026-09-14T20:48:37Z`, submitted `2026-09-14T20:49:36Z`, `out/r3int/submit_tree.txt`), analysis `out/r3int/analysis/`
(`analysis_console.txt` = the re-emitted run after the one reducer change of section 10; `analysis_console_run1.txt` the
first pass, same numbers), verification `out/r3int/verify/verify.json` (`problems: 0`). CPU smokes `out/r3int/smoke/` (no
number below comes from them). No file under `flyverse/` is authored by this task; the script imports every arm's flags
from the thread script that defined it (section 1) and calls the threads' own generators. **Nothing is adopted; every
default is where it was.**

## 0. Answer

Eight arms (the subsets of A = the sided transducer, B = the monoamine slow class at `add_low`, C = the per-transmitter
w_syn at `high`) on the 16 x 60 s plain-fly rollout, 5 runs per arm in ONE submission (runs are the replicate unit;
verdicts `common.compare` 5 v 5, floor p 0.0079); the efferent compass x 3 seeds per arm; the shipped-gain cx_wedge
compass x 4 seeds and the benchmark sections x 2 draws for the four brain-only configurations. Every run `device cuda /
NVIDIA B200`, every arm on the eager Torch path (section 1), every stamped `lif_in_force` equal to the arm's flags
(`verify`: 88 / 88 JSONs, 0 problems).

**Which combination gets closest to emergent turning, and by how much.** None reaches the animal's bracket (saccades
of a few hundred deg/s every ~250 ms with a low residual yaw; section 4): no arm puts a single clean walking frame above
100 deg/s (`yaw_frac_gt100_clean` 0.000 in all 40 runs). Two things move, and they are not the same thing:

* **A (= AB) is the closest legitimate step** -- the only arm whose extra turning is made of walking. Clean-frame yaw SD
  2.75 +- 0.08 -> **7.67 +- 0.11 deg/s** (z +60, `result`), median |yaw| 0.74 -> 1.49, 99th percentile 15.8 -> 24.7,
  straightness 0.995 -> 0.867 +- 0.045, DNa02_L / _R 0.016 / 0.080 -> **0.505 / 0.385 Hz** (`result`), DNa02 above 5 Hz on
  6.3 % of walking frames (0.7 %), speed 8.83 -> 9.71 mm/s, with hops 0.06 -> 0.36 per fly, 6.8 +- 2.2 of 16 flies leaving
  the table top within 60 s (0.2), 94 % of window frames clean, and a fixed left drift of +1.01 +- 0.18 deg/s in 15.2 / 16
  flies (shipped +0.18 in 13.4 / 16). This replicates the body-state thread's arm D (H200, native kernels: 7.73 / 0.876 /
  0.512 / 0.393 / 6.6 leavers / 0.38 hops) on a B200 on the eager path inside the run scatter -- a cross-device,
  cross-backend replication, not a bit check. **B adds nothing to it BEHAVIOURALLY**: AB vs A is `null` on every headline
  key (yaw SD +0.15 z +1 p 0.15; straightness -0.015; DNa02_L +0.018; hops +0.11; leavers +1.4; speed +0.000), as B vs
  shipped is `null` on every behaviour and DNa02 key (yaw SD 2.72 vs 2.75; straightness 0.994; DNa02 0.018 / 0.077; hops
  0.10). It is not `null` on every ROW, and the exceptions matter for how a future B arm is scored: B v shipped has 8
  `result` rows of 86 (leg_L, leg_abs_LR, power_mean, AN07B035_L, IN12B014_L / _R, IN19A003_R, speed -- and the speed
  `result`, -0.0227 mm/s at z -5.0, is 60 % reproduced by an independent re-draw of the SAME configuration, section 4.2),
  and AB v A has 2 (GLNO_L / _R, +0.68 / +0.69 Hz, z +5.0 / +5.7), ABC v AC has 9. **B's footprint is
  background-dependent: null where the VNC is silent, a `result` wherever the transducer has woken it** (section 9 item
  5(a)).
* **AC (= ABC) moves the yaw statistics most and is a bias on a hopping fly.** Clean yaw SD **12.76 +- 0.68** (AC) /
  13.39 (ABC), median |yaw| 7.4 / 8.1 deg/s, p99 39 / 42, straightness **0.385 +- 0.043** / 0.392, DNa02_L / _R **2.01 /
  0.94 Hz** with L-R **+1.12 +- 0.10 Hz of one sign in 16 / 16 flies of every run**, mean signed yaw **+5.2 / +6.2 deg/s (a
  left drift) in 15.8 / 16 flies** -- and 22.3 / 23.3 hops per fly (C alone 43.2), 8.1-8.5 % of the time airborne (C 13.8 %),
  **all 16 flies off the table top by 11.1 +- 2.3 s** (C 7.7 s), so the clean statistics rest on 17 % of the window. AC vs
  C: yaw SD +1.2 (z +2, p 0.016, `null`), straightness -0.47 (`result`), DNa02_L +1.07 (`result`), DNa02 L-R +1.10
  (`result`), hops **-20.9 (z -31, `result`: the transducer halves C's hops)**, airborne -0.057 (`result`), speed +3.3 mm/s
  (`result`); AC vs A: yaw SD +5.1 (z +46), median |yaw| +5.9, straightness -0.48, DNa02_L +1.5 Hz, hops +22, leavers +9.2
  (all `result`). The leg L-R offset is the only one that SHRINKS under AC / ABC (+0.152 -> +0.097 / +0.095 Hz, `result`)
  and the only place in 400 fly-runs of this batch where it changes sign: 2 of 16 flies in AC r2 and 1 of 16 in ABC r3
  (0 / 80 in every other arm, 0 / 80 in every arm of the body thread) -- three fly-runs, not a mechanism. What C does to
  DNa02 is section 4.4: E and I on DNa02 both grow, by x2.6 to x3.5 depending on side and sign (the inhibitory
  interneurons IN12B014 / IN19A003 / PS059 fire 1.5-3x
  under I / E 0.75), the NET input goes more negative on both sides (DNa02_L -304 -> -337, DNa02_R -362 -> -794 mV/s) and
  DNa02 fires from the raised fluctuation; with A on top the left side's net is -34 and the right side's -621 mV/s, which
  is the +1.1 Hz bias -- the connectome's sidedness, 57 % of it through an unsided afferent's unequal delivery (SNpp45/?
  +128 to the left against +61 to the right) and 43 % through the inhibitors' own rates, amplified by the disinhibition
  (section 4.4).
* **C and BC** (no transducer): yaw SD 11.6 / 11.7 on 16 / 13 % clean frames, 43-45 hops per fly, 14 % airborne, every
  fly off the table in 7.7-8.8 s, commanded speed 5.3 mm/s, DNa02 0.95 / 0.92 Hz with no fixed sign (L-R +0.02, L > R in
  8.8 / 16): the unitary thread's room result (`high`: 39-42 hops, 16 / 16 leavers in 3.5-7.9 s) replicated on the eager
  path. BC vs C is `null` on every behavioural key.

**The compass.** No arm forms a bump at the shipped gains: `compass.EPG.bump_survival_s` **0.00 s in 16 / 16 wedge
runs (FAIL x16)**, rate and width `NOT_APPLICABLE` x16, PEN 0.04-0.05 Hz after the pulse in every arm; C / BC raise
Delta7 by +0.96 Hz (z +7.0, p 0.029, `result`) and the ring by +0.06 -- the unitary thread's H200 numbers to 0.01 Hz per
seed (section 6). No arm gives the efferent compass a signed report of the self-turn: bump drift |mean| <= 0.005 w/s in
every phase of every arm (ideal +-4.0), every ccw / cw-vs-rests row `underpowered` (3 v 6) with per-seed values inside
the rest scatter; the signed report exists at depth 1 (AN04B003 flip -10.5 (A) / -10.9 (AB) / -11.5 (AC) / -12.2 (ABC) Hz
against nulls of 1 Hz; -0.03 to -0.4 in the arms without A), reaches PS196_b at depth 2 (-1.1 +- 1.4 (A) -> **-3.00 +-
0.29 under ABC**, per seed -2.71 / -3.29 / -3.00 against a null of -0.57 +- 1.62) and is **gone at GLNO** (flip -0.02 +-
0.08 (ABC), +0.14 (A), +0.50 +- 0.57 (AC)) and at PEN / EPG (|flip| < 0.7 Hz everywhere). C makes the pinned fly turn
left at rest (heading +8.8 (AC) / +6.6 rest, +11.2 rest2 (ABC) deg/s against +1.1 shipped) and turn less in the cw
stimulus (-73 vs -92 deg/s): the same fixed bias, on the pinned body.

**Suite cost.** Shipped and B: 12 / 0 in both draws (B's seed-locked shifts are the monoamine thread's to the digit:
`taste.MN9_hz` 10.93 -> 2.48, `smell.PN_hz` 7.86 -> 6.93, `walk.power_sustained_hz` 20.11 -> 24.52). C: `walk.power_sustained_hz`
**FAIL in 1 of 2 draws (90.47 / 44.70)** -- on the eager path the walk section under C scatters across the bound (the
unitary thread's three native draws were 89-103); `taste.MN9_hz` 10.93 -> 20.08. BC: FAIL in 2 / 2 (103.89, identical
draws), MN9 12.03 -- B's downward tone and C's disinhibition partly cancel on MN9 and add on the wing-power route. A cannot
be scored by the suite (no body); in the room its wing-power and GF rows are at the shipped level (power sustained 65.0 vs
67.3 `null`; GF max 27.9 vs 25.2 `null`), C's are not (136 vs 67, 51 vs 25, both `result`).

**Under the project rule.** A is a body-model mechanism whose room result (DNa02 fires from the leg afferents at
literature-typical rates; the fly meanders and drifts left by the connectome's own asymmetry) stands as the body thread
left it, with that thread's level confound (its section 8 item 1) unresolved by anything here. B at 0.02 is invisible on
behaviour and near-invisible on DNa02's input (E and I totals within 1.5-2.2 %, a few named rows 4-7 %, section 4.4), as
its thread said -- but it is NOT invisible on the VNC relay types once the transducer is on (GLNO +0.68 to +1.34 Hz,
`result`; section 9 item 5(a)). C is a net disinhibition that doubles
every input of DNa02 and pays on the take-off route; combining it with A does not turn the disinhibition into steering,
it turns the connectome's sidedness into a 1.1 Hz DNa02 bias and a 5-6 deg/s left drift. Nothing is adopted; section 9.

## 1. The arms, and where every flag comes from

The eight arms are the subsets of three mechanisms on top of the shipped model (`LIFParams()`, no sense attached):

| mechanism | what is switched on | the thread's flag, imported verbatim | the thread's own verdict on it |
|---|---|---|---|
| **A** sided transducer | `senses.Proprioception('all+leg_cycle+haltere_sided')` attached to `BatchSim` / the scalar `FlySim`, `body.LegCycle` on the body (arm D of `docs/audits/body_sided_state.md`) | `probe_vnc_drive.ARMS_BODY['D']` | a body-model mechanism, opt-in; DNa02 fires (0.54 / 0.38 Hz), yaw SD 2.6 -> 7.7 deg/s, straightness 0.995 -> 0.88, 7 / 16 flies leave the table, a fixed left drift; the compass receives no signed report; level confound unresolved (its section 8 item 1) |
| **B** monoamine slow class | `receptor_model 'full'`, net rule `abs`, gain classes 1/1/1 (= the `sign` fast weights entry for entry), `slow_mode 'additive'`, `slow_gain 0.02` (arm `add_low` of `docs/audits/monoamine_slow_term.md`) | `probe_monoamines.ARMS['add_low']['lif']` (and its `['bench']` flags for `benchmark.py`) | the only bracket that passed the suite 12 / 0 / 0 x 5 with every behaviour key `null`; +3 % spikes; `taste.MN9_hz` 10.9 -> 2.5 seed-locked; 0.2 / 1.0 additive run away in 0.5 s, `gain_mid` hops -- "null on behaviour, regression watch" |
| **C** per-transmitter w_syn | `LIFParams.w_syn_by_nt = {acetylcholine 1.0, gaba 0.75, glutamate 0.75}` (bracket `high` of `docs/audits/unitary_strength.md`: ACh x1, I / E 0.75) | `probe_unitary.BRACKETS['high']` | **no bracket keeps the suite** (`walk.power_sustained_hz` 89-103 Hz vs < 50 in 3 / 3 draws; 11 / 1); in the room every fly hops 39-42 times a minute and leaves the table in 3.5-7.9 s; the ring forms no bump |

The task asked for "C at the suite-safe bracket"; the unitary thread found none (every bracket fails
`walk.power_sustained_hz` in every draw), so C runs at the least-cost bracket the thread itself put in the room, **labelled
as not suite-safe**, rather than being left out -- the combination question ("does A or B change what C does, or C what A
does") is then answered by measurement instead of inference. `probe_round3_integrate.py arms` prints the resolved arms and
asserts, on the CPU, that the shipped arm is `LIFParams()` to the field, that A carries no LIFParams change, that B's dict is
`probe_monoamines.ARMS['add_low']['lif']` and switches the slow term on (`brain._slow_spec` not None, `{'monoamine': 0.02}`,
tau 200 ms), that C's dict is `probe_unitary.BRACKETS['high']`, and that the patch is restored afterwards.

**One backend for every arm of every protocol.** `receptor_model 'full'` is not in the native LIF kernel: `brain.Brain`
turns the CUDA kernels off with a warning and runs the Torch path (`brain.py` 436-445). To keep no arm-vs-arm row
backend-crossed, the script forces the same path on every arm (`BACKEND = cuda_graphs False, cuda_kernels False,
event_driven False (sparse matmul), cuda_sparse torch`; `benchmark.py --eager`), through subclasses of `BatchSim`,
`room_demo.Sim` and `FlyBrain` that override the generators' own flags. The consequence is that the shipped arm here is
the shipped model on the eager path (as the monoamine thread's `off` arm was), not on the native kernels the body-state
and guard threads ran; the shipped arm's own numbers are compared with those threads' only as a cross-backend, cross-device
(H200 -> B200) replication: clean yaw SD 2.75 +- 0.08 here vs 2.64 +- 0.13 (body thread A), DNa02 0.016 / 0.080 vs 0.017 /
0.075 Hz, leg L-R +0.152 vs +0.151, `taste.MN9_hz` 10.934 = 10.93, and the walk triple 48.48052978515625 /
20.109053071339925 / 4.629162311553955 **reproduced the value recorded in rounds 5 / 6 / 7 exactly in both draws here**
(as in the 14 prior native draws on record; the CPU reads 57.3838 / 26.6636 / 9.7632 for the same shipped default).
"Bit for bit" would be the wrong word: bit-identity is a CPU claim under the project rule, never a claim made from a GPU
A / B, and this very batch shows the same walk section is not seed-locked on this path under C (90.47 vs 44.70 between
two draws on one box, section 7). Every room / compass / wedge JSON is stamped by `_stamp` with an
`integrate` block: the arm, its mechanisms, the sense spec, the LIFParams overrides asked for, **the LIFParams fields the
Brain actually carried** (read back from `provenance.model.lif`: `receptor_model`, `receptor_gain`, `slow_mode`,
`slow_gain`, `w_syn_by_nt`, `event_driven`) and the realised backend flags -- and the job exits 3 when they differ from
the arm's. The bench JSON records the same read-back from every `LIFParams` that `benchmark.Context._apply_lif` finalised
(2 per run). `verify` re-reads all of it (section 3).

**What the body-less protocols can distinguish.** The cx_wedge compass (`probe_unitary compass`) is a bare `FlyBrain`
and `benchmark.py`'s sections have no body (`docs/audits/guard_suites_r3.md` 1b: the sense is attached only by `BatchSim`),
so the A mechanism cannot act there: the A-arms equal their brain-only counterpart by construction (A = shipped, AB = B,
AC = C, ABC = BC) and only the four brain configurations were run for those two protocols. The efferent compass is a pinned
`FlySim` with a body (the body-state thread's protocol), so all eight arms ran there.

## 2. Classification under the project rule

| arm | classification | why |
|---|---|---|
| A | body-model mechanism, opt-in (the body thread's own classification; its skeptic's corrections stand: `half_width_m` is a loop gain, the strict tripod at 9 mm/s is an assumption) | a stance / swing phase from the realised speed and a readout that respects the anatomy's sidedness; no number set against behaviour |
| B | a hand-set scalar on a mechanism the connectome implies (the 3,312 sign-0 monoamine bodies), at the value that is invisible on behaviour | the 0.02 is the KC-anchored ceiling, not a measurement (monoamine_slow_term.md 4 item 5); the receptor rows are data, the scale is not |
| C | an argument, not a measurement, and a net disinhibition | the I / E 0.75 rests on the chloride driving-force argument (unitary_strength.md 1; its skeptic: the one measured insect unitary I / E, 0.28, is Periplaneta, and the E_Cl endpoints are uncited) |
| the combinations | nothing new is introduced by combining; no gain, bias or threshold was added to make any combination turn | the script adds no parameter of its own; `arms` pins that |

What would be hand-crafting and is not done: any gain on the leg-afferent -> AN -> DNa02 route; a threshold moved on
DNa02; a scale chosen for B or C by looking at the fly's yaw; a per-side asymmetry anywhere.

## 3. The batch, and its verification

`out/r3int/batch.sh` (`probe_round3_integrate.py plan --dir out/r3int --runs 5 --compass-seeds 3 --wedge-seeds 4 --draws 2
--minutes 180 --name r3int`): ONE `cluster_run.py --name r3int --arm-block fam` call with **51 jobs** -- 40 `room` (8 arms x
brain seeds 0-4, 16 flies x 60 s, env seeds 100 s .. 100 s + 15, `probe_walk_straightness.py`'s plain-fly protocol through
`probe_vnc_drive room --family body`; blocks `fam_r<seed>` = the eight arms of one seed), 3 `compass` jobs (seeds 0-2, the
eight arms sequential inside one job each, each arm its own python process and exit code; blocks `fam_c<seed>`), 4 `wedge`
jobs (shipped / B / C / BC, seeds 0-3 sequential; `fam_wedge`) and 4 `bench` jobs (the same four, 2 draws sequential;
`fam_bench`). Every job line preserves python's exit code (`st=$?; tail; exit $st`; the sequential jobs `exit $((s0 | s1
...))`), `$` is escaped in the batch script so the local shell hands them through, `--fetch out/r3int/`, console
`out/r3int_cluster.log` (`51 job(s), 0 failed (67.6 min) run dir <cluster-fs>/neurome/runs/r3int-9a78e4`). The house cluster is
one target (<cluster-node>, 8 x B200), so every block is trivially on one box and no arm-vs-arm row is device-crossed; all 51 jobs ran
concurrently on <cluster-node> (2 TB RAM; `slots 112`); room jobs took 1,061-2,221 s of wall (the B arms ~1.5x, the slow term is on the
Torch path), a compass arm 20-480 s per phase depending on the load.

**What the batch shipped** (`out/r3int/submit_tree.txt`, HEAD `6ec2de1` = `origin/main`): the whole round-3 working tree,
35 files -- this task's `scripts/probe_round3_integrate.py`; thread body-state's `flyverse/body.py` (+134), `batch_body.py`
(+26 / -4), `batch_sim.py` (+5 / -1), `motor.py` (+29), `senses.py` (+157 / -12), `scripts/probe_vnc_drive.py`; thread
unitary's `flyverse/brain.py` (+30 / -1: `LIFParams.w_syn_by_nt`, default `None`); the 13 `op report` ledger rows of threads
unitary / monoamines in `flyverse/data/expected_responses.csv`; the other threads' scripts, tests and audits. None of it is at
`origin/main` (this task may not commit); the code identity of every number is the JSON's `provenance.source_fingerprint`
plus the compiled-W md5 `ef23cc27...`, as in the other round-3 audits (`flyverse_commit.commit` reads `unknown` in a run copy).

**Verification** (`probe_round3_integrate.py verify --dir out/r3int` -> `out/r3int/verify/verify.json`,
`verify_runs.csv`): 88 planned runs (40 room + 24 compass + 16 wedge + 8 bench), **88 JSONs present, every one `device
cuda` / `device_name NVIDIA B200`**, every room run with its four npz, every compass run with its four phase npz, every
`arm` field equal to the planned arm, every sense spec equal to the arm's (`'all+leg_cycle+haltere_sided'` on the four A
arms, `None` on the others), every `integrate.problems` empty (the LIFParams in force = the arm's; `cuda_kernels False`,
`cuda_graphs False` realised), every bench `backend 'eager torch'` with `full` in the B / BC configs -- `problems: 0`. Per
arm the room JSONs' `lif_in_force` read: shipped / A `receptor_model sign, w_syn_by_nt null`; B / AB `full / abs, gain
classes 1/1/1, additive 0.02, tau 200`; C / AC `sign, w_syn_by_nt {ACh 1.0, gaba 0.75, glutamate 0.75}`; BC / ABC both
(`analysis_console.txt` lines 2-25).

## 4. The room (40 runs: 8 arms x 5 brain seeds, 16 flies x 60 s, 5-60 s window)

Artefacts: `out/r3int/room_<arm>_r<seed>.json` (summary, per-fly rows, provenance, the `integrate` stamp), `_body.npz`
(per-frame heading / position / commands / commanded afferent Hz per channel, side and leg / cycle state / side-split
haltere MN readout), `_rec.npz` (the watch types' cells at 2-frame resolution), `_flies.npz` / `_max.npz` (per-fly window
mean and per-cell max of every cell: the decompose input). Tables: `out/r3int/analysis/room_table.csv` / `.txt` (every key,
per-run values, vs-shipped verdicts, the pairwise family; the CSV's verdict column holds the literal string `null`, which
`pandas.read_csv` turns into NaN unless `keep_default_na=False`), `pairwise.txt`, `dna02_decompose_summary.csv`,
`decompose_DNa02_{L,R}_per_type.csv`. Every number is a run mean +- SD over the five runs of an arm (runs are the
replicate unit); verdicts are `flyverse.interp.common.compare` (5 v 5, floor p 0.0079; `result` needs |z| >= 3 and p <=
0.05; **a zero-SD null with a non-zero diff reads `undetermined`, while a zero-SD null whose diff is exactly 0 reads
`null` -- the wedge survival rows of section 6 are the second case, "null with a zero-SD null (structural)", not
"undetermined"**; rows whose values are structurally 0 in both arms are not tests). The yaw
statistics are the body thread's CLEAN-frame ones (`probe_vnc_drive.robust_room`: on the table top before and after, no
airborne frame within 0.5 s, |yaw| <= 720 deg/s) because a table-edge crossing puts 9,000-18,000 deg/s into the plain SD
(`yaw_sd_deg_s` over all walking frames: 3.0 (shipped) / 78.7 (A) / 52.5 (C) / 65.3 (AC) deg/s; `yaw_max_abs_deg_s` 46 /
4,490 / 2,362 / 3,487). **Under the four C arms the clean frames are 12-17 % of the window** (every fly is off the table
top by 2-19 s; `left_table_s` per fly), so their clean statistics describe the first seconds of the rollout, on the table,
between hops; the A / AB arms keep 93-94 % of the window clean, shipped / B 99.8 %.

**Selection confounding, checked and quoted (skeptic pass).** Because the clean fraction differs by more than 5x between
the C arms (12-17 %) and A (94 %), every AC-vs-A row is selection-confounded and the window-matched statistic belongs
beside the full-window one: re-scored on a matched **5-16 s** window (A clean fraction 1.00, AC 0.51) the AC v A clean
yaw SD difference is **+4.55 (z +26.3, p 0.0079, `result`)** and on **5-10 s** it is **+3.78 (z +9.0, `result`)**, against
the +5.09 the full-window table reports. The ordering AC > C > A > shipped survives window matching and AC v C stays
`null` in every window; the magnitude is selection-inflated by 15-26 %.

**The animal's bracket (no ledger row exists for free-walking turning; quoted from the literature, `op report` in
spirit).** DeAngelis, Zavatone-Veth & Clark 2019 (eLife 8:e46409, Fig. 1C-D): forward speeds between -1.3 and 30.4 mm/s
(2.5th-97.5th percentile) with peaks at 0 and ~17.5 mm/s, and "flies turn at a broad range of yaw rates across many
forward speeds" -- no yaw-rate SD is stated. Katsov et al. 2017 (eLife 6:e26410): the rotational-velocity phase space
spans |v_R| < 450 deg/s and "local peaks in v_R arise frequently ..., with an inter-peak interval of 250 +- 110 ms"; forward
speed -0.6 to 3.2 cm/s. Geurten et al. 2014 (Front Behav Neurosci 8:365): body saccades above a 200 deg/s threshold turn the
fly by ~15 deg in 40-120 ms (median 90 ms), rotations occupy 9 % of the time against 29 % translation and 63 % rest, with
"some residual rotations during translatory phases ... much lower than during phases of turning". The bracket the room
rows are read against is therefore: a walking fly turns in discrete saccades of a few hundred deg/s every few hundred ms
with a low residual yaw between them, at 10-20 mm/s. The shipped fly's clean-frame yaw SD of ~2.7 deg/s with no frame
above 100 deg/s and straightness 0.995 is below that bracket by two orders of magnitude on the saccade axis; the question
of this round is whether any arm moves it towards the bracket by a turn signal rather than by a hop, an edge or a bias.

### 4.1 Behaviour and the motor readout (run mean +- SD over 5 runs; per-run values in `room_table.csv`)

| key | shipped | A | B | C | AB | AC | BC | ABC |
|---|---|---|---|---|---|---|---|---|
| yaw SD, clean frames (deg/s) | 2.75 +- 0.08 | **7.67 +- 0.11** | 2.72 +- 0.21 | 11.56 +- 0.51 | 7.82 +- 0.15 | 12.76 +- 0.68 | 11.73 +- 0.15 | 13.39 +- 0.47 |
| median \|yaw\|, clean (deg/s) | 0.74 +- 0.01 | 1.49 +- 0.07 | 0.75 +- 0.02 | 4.32 +- 0.55 | 1.51 +- 0.08 | 7.38 +- 0.77 | 5.04 +- 0.28 | 8.11 +- 1.14 |
| p99 \|yaw\|, clean (deg/s) | 15.8 +- 0.3 | 24.7 +- 0.8 | 15.5 +- 1.5 | 39.1 +- 1.2 | 26.0 +- 0.8 | 39.1 +- 2.8 | 37.3 +- 1.9 | 41.8 +- 1.4 |
| clean frames with \|yaw\| > 30 / > 100 deg/s | 0.000 / 0 | 0.005 / 0 | 0.000 / 0 | 0.036 / 0 | 0.006 / 0 | 0.047 / 0 | 0.036 / 0 | 0.058 / 0 |
| mean SIGNED yaw, clean (deg/s; + left) | +0.18 +- 0.05 | +1.01 +- 0.18 | +0.24 +- 0.06 | +2.02 +- 0.47 | +1.02 +- 0.16 | **+5.24 +- 0.75** | +2.41 +- 0.94 | **+6.18 +- 0.62** |
| flies (of 16) with mean yaw > 0 | 13.4 | 15.2 | 14.8 | 13.6 | 15.2 | 15.8 | 14.2 | 15.8 |
| straightness | 0.995 +- 0.001 | 0.867 +- 0.045 | 0.994 +- 0.001 | 0.859 +- 0.033 | 0.852 +- 0.039 | **0.385 +- 0.043** | 0.806 +- 0.040 | 0.392 +- 0.040 |
| net heading change (turns / fly) | 0.032 | 0.183 +- 0.042 | 0.038 | 0.156 | 0.201 | 0.190 | 0.216 | 0.183 |
| clean fraction of the window | 0.999 | 0.941 +- 0.030 | 0.998 | 0.162 +- 0.012 | 0.930 | 0.171 +- 0.034 | 0.125 +- 0.012 | 0.157 +- 0.020 |
| flies (of 16) that left the table top | 0.2 +- 0.4 | 6.8 +- 2.2 (4 7 7 6 10) | 0.2 | **16.0** | 8.2 +- 2.2 (9 10 5 7 10) | **16.0** | **16.0** | **16.0** |
| time to leave, mean per fly (s; never = 60) | 60.0 | 55.3 +- 2.9 | 59.5 | 7.7 +- 2.0 | 54.9 +- 1.8 | 11.1 +- 2.3 | 8.8 +- 1.0 | 10.7 +- 3.7 |
| hops per fly | 0.06 +- 0.08 | 0.36 +- 0.22 | 0.10 +- 0.06 | **43.2 +- 0.7** | 0.48 +- 0.25 | **22.3 +- 0.7** | 44.6 +- 0.7 | 23.3 +- 0.6 |
| airborne fraction | 0.000 | 0.004 +- 0.002 | 0.000 | 0.138 +- 0.004 | 0.005 | 0.081 +- 0.002 | 0.142 +- 0.002 | 0.085 +- 0.002 |
| commanded speed, window mean (mm/s; airborne frames included) | 8.83 +- 0.00 | 9.71 +- 0.02 | 8.81 +- 0.01 | 5.34 +- 0.02 | 9.71 +- 0.01 | 8.63 +- 0.06 | 5.22 +- 0.05 | 8.49 +- 0.03 |
| DNa02_L / R (Hz) | 0.016 / 0.080 | **0.505 / 0.385** | 0.018 / 0.077 | 0.948 / 0.915 | 0.523 / 0.395 | **2.014 / 0.940** | 0.986 / 0.835 | 2.112 / 1.013 |
| DNa02 L-R (Hz) | -0.064 +- 0.008 | +0.121 +- 0.037 | -0.059 +- 0.012 | +0.021 +- 0.031 | +0.128 +- 0.037 | **+1.124 +- 0.100** | +0.124 +- 0.024 | +1.149 +- 0.053 |
| fly-runs (of 80) with DNa02 L-R > 0 | 2 | 63 | 4 | 44 | 67 | **80** | 61 | **80** |
| frames with DNa02 > 5 Hz | 0.007 | 0.063 | 0.007 | 0.124 | 0.065 | 0.218 | 0.121 | 0.230 |
| leg MN L / R (Hz) | 2.55 / 2.39 | 4.75 / 4.60 | 2.59 / 2.43 | 7.75 / 7.56 | 4.76 / 4.61 | 9.50 / 9.41 | 7.65 / 7.47 | 9.57 / 9.47 |
| leg MN L-R (Hz) | +0.152 +- 0.004 | +0.147 +- 0.008 | +0.160 +- 0.002 | +0.182 +- 0.006 | +0.147 +- 0.009 | **+0.097 +- 0.008** | +0.180 +- 0.012 | +0.095 +- 0.006 |
| fly-runs (of 80) with leg L-R < 0 | 0 | 0 | 0 | 0 | 0 | **2** (r2) | 0 | **1** (r3) |
| haltere MN L / R (Hz) | 7.70 / 9.44 | 15.12 / 17.91 | 7.72 / 9.43 | 21.76 / 23.33 | 15.14 / 17.91 | 32.24 / 35.37 | 21.34 / 22.91 | 32.55 / 35.61 |
| haltere MN L-R (Hz) | -1.75 +- 0.01 | -2.78 +- 0.10 | -1.71 | -1.57 +- 0.06 | -2.77 | -3.12 +- 0.04 | -1.56 | -3.06 |
| wing power MN mean / 0.3 s-sustained max (Hz) | 20.4 / 67.3 +- 0.8 | 15.2 / 65.0 +- 1.3 | 20.8 / 66.2 | 58.0 / **136.5 +- 1.8** | 15.4 / 65.7 | 47.2 / 137.2 | 57.5 / 134.3 | 47.8 / 138.4 |
| GF per-frame max (Hz) | 25.2 +- 0.5 | 27.9 +- 4.8 | 25.7 +- 0.9 | 51.1 +- 4.0 | 30.3 +- 3.8 | 45.2 +- 1.1 | 48.6 +- 2.6 | 48.4 +- 6.9 |
| commanded chordotonal / hair plate / campaniform / haltere (Hz) | 0 | 86.4 / 46.5 / 24.9 / 16.5 | 0 | 0 | 86.1 / 46.4 / 24.9 / 16.5 | 82.1 / 44.7 / 23.0 / 33.6 | 0 | 81.2 / 44.3 / 22.9 / 33.9 |
| step frequency (Hz) / stance fraction | | 7.67 / 0.770 | | | 7.66 / 0.770 | 7.44 / 0.777 | | 7.34 / 0.780 |

**Caveat: the `DNa02_L / R` row and the `DNa02 L-R` row are computed on DIFFERENT frame masks and do not subtract**
(inherited from `probe_vnc_drive`'s reducer, not introduced here; skeptic pass). The L and R keys include airborne
frames; the L-R key is the non-airborne post-skip window. The gap by arm is shipped +0.0002, A +0.0010, C -0.0122, AC
+0.0500, BC -0.0278, ABC +0.0498 Hz -- which is why the AC column reads 2.014 / 0.940 (difference 1.074) next to a
reported L-R of +1.124. Recomputed on the L-R row's own mask the pair is AC 2.061 / 0.937 and ABC 2.150 / 1.001, which
DO subtract to the reported L-R. The two rows should not be read against each other without this line.

Per-seed values behind the calls (seed order r0..r4): yaw SD clean A 7.71 / 7.58 / 7.84 / 7.58 / 7.61, AB 7.80 / 7.67 /
8.06 / 7.82 / 7.74, C 11.12 / 12.18 / 11.19 / 12.04 / 11.28, AC 12.56 / 13.01 / 13.81 / 12.18 / 12.23, ABC 12.89 / 13.80 /
13.56 / 13.82 / 12.89, shipped 2.66 / 2.86 / 2.81 / 2.71 / 2.71, B 2.48 / 2.85 / 2.68 / 3.01 / 2.56; straightness A 0.905 /
0.899 / 0.866 / 0.876 / 0.792, AC 0.443 / 0.333 / 0.391 / 0.405 / 0.352, ABC 0.437 / 0.420 / 0.395 / 0.339 / 0.369; DNa02_L
A 0.493 / 0.485 / 0.522 / 0.499 / 0.525, AC 1.946 / 2.046 / 2.050 / 2.061 / 1.966, C 0.940 / 0.920 / 0.921 / 0.981 / 0.980;
DNa02 L-R AC 0.983 / 1.187 / 1.161 / 1.226 / 1.061, ABC 1.145 / 1.240 / 1.107 / 1.139 / 1.115; signed yaw AC 4.67 / 4.52 /
6.15 / 5.94 / 4.94, ABC 5.75 / 6.18 / 6.49 / 7.02 / 5.45, A 0.86 / 0.85 / 1.04 / 1.01 / 1.28; hops C 43.8 / 43.9 / 42.3 /
42.8 / 43.3, AC 22.7 / 22.1 / 22.0 / 23.3 / 21.4; leg L-R AC 0.109 / 0.089 / 0.092 / 0.099 / 0.096, ABC 0.093 / 0.088 /
0.102 / 0.099 / 0.092; leavers A 4 / 7 / 7 / 6 / 10, AB 9 / 10 / 5 / 7 / 10; speed (mm/s) A 9.68-9.74, C 5.31-5.37, AC
8.54-8.69.

### 4.2 The pairwise family (`pairwise.txt`; diff, z = diff / SD(null), exact-U p, verdict; 5 v 5)

**The z column of the table below is transcribed to the nearest INTEGER**, which in a table whose criterion is exactly
|z| >= 3 makes three rows read as a contradiction: `A v shipped` wing power sustained is printed here as `z-3 p0.032
null` at an actual z of **-2.88**, and `BC v C` DNa02_R as `z-3 p0.008 null` at **z -2.77** (skeptic's numbers). The
verdicts are correct; the printed z is not. The generators themselves do not round -- `pairwise.csv` stores z at full
precision and both reducers' consoles (`scripts/probe_vnc_drive.py pairs`, `probe_round3_integrate.py analyse` ->
`pairwise.txt`) already print `z{:+.1f}`, e.g. `-2.9` for the first row -- so the fix is in the transcription, and any
table whose criterion is |z| >= 3 gets one decimal from here on. No analysis was re-run for this correction and the
table below was not re-typed: read the two `z-3 ... null` rows as **-2.88** and **-2.77**.

| key | A v shipped | B v shipped | C v shipped | AB v A | AB v B | AC v A | AC v C | BC v B | BC v C | ABC v AB | ABC v AC | ABC v BC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yaw SD clean | +4.92 z+60 result | -0.03 null | +8.81 z+108 result | +0.15 z+1 p0.15 null | +5.10 result | +5.09 z+46 result | +1.20 z+2 p0.016 null | +9.01 result | +0.17 null | +5.57 result | +0.63 z+1 null | +1.66 z+11 result |
| median \|yaw\| clean | +0.75 result | +0.02 null | +3.58 result | +0.02 null | +0.76 result | +5.89 result | +3.06 z+6 result | +4.29 result | +0.72 p0.056 null | +6.61 result | +0.74 null | +3.07 result |
| signed yaw | +0.83 result | +0.05 null | +1.84 result | +0.01 null | +0.78 result | +4.23 z+24 result | +3.22 z+7 result | +2.17 result | +0.39 null | +5.16 result | +0.94 null | +3.77 result |
| straightness | -0.128 result | -0.001 null | -0.137 result | -0.015 null | -0.142 result | -0.483 z-11 result | -0.474 z-14 result | -0.188 result | -0.053 p0.056 null | -0.460 result | +0.007 null | -0.414 result |
| flies off the table | +6.6 result | 0 null | +15.8 result | +1.4 null | +8.0 result | +9.2 z+4 result | 0 (both 16) | +15.8 result | 0 | +7.8 result | 0 | 0 |
| hops | +0.30 z+4 result | +0.04 null | +43.1 result | +0.11 null | +0.38 result | +21.9 result | **-20.9 z-31 result** | +44.5 result | +1.4 z+2 p0.032 null | +22.8 result | +1.0 p0.056 null | -21.3 result |
| airborne fraction | +0.004 result | 0 null | +0.138 result | +0.001 null | +0.005 result | +0.077 result | **-0.057 result** | +0.141 result | +0.003 null | +0.080 result | +0.004 p0.032 null | -0.056 result |
| speed | +0.88 result | -0.02 z-5 p0.016 result | -3.49 result | 0.00 null | +0.90 result | -1.08 result | +3.30 result | -3.59 result | -0.12 result | -1.22 result | -0.14 z-2 null | +3.27 result |
| DNa02_L | +0.489 result | +0.002 null | +0.933 result | +0.018 null | +0.505 result | +1.509 result | +1.065 result | +0.969 result | +0.038 null | +1.589 result | +0.098 z+2 p0.016 null | +1.126 result |
| DNa02_R | +0.305 result | -0.003 null | +0.835 result | +0.010 null | +0.319 result | +0.555 result | +0.025 null | +0.758 result | -0.080 z-3 p0.008 null | +0.617 result | +0.072 null | +0.178 result |
| DNa02 L-R | +0.185 result | +0.006 null | +0.085 z+11 result | +0.007 null | +0.186 result | +1.003 result | +1.103 result | +0.182 result | +0.103 z+3 result | +1.022 result | +0.026 null | +1.026 result |
| DNa02 active fraction | +0.056 result | 0 null | +0.117 result | +0.002 null | +0.059 result | +0.155 result | +0.094 result | +0.114 result | -0.003 null | +0.165 result | +0.012 z+5 result | +0.110 result |
| leg L-R | -0.006 null | +0.008 z+2 p0.016 null | +0.029 z+8 result | +0.001 null | -0.013 result | -0.050 result | -0.084 result | +0.020 result | -0.001 null | -0.052 result | -0.002 null | -0.085 result |
| wing power sustained | -2.3 z-3 p0.032 null | -1.0 null | +69.2 result | +0.8 null | -0.5 null | +72.2 result | +0.7 null | +68.1 result | -2.1 null | +72.6 result | +1.2 null | +4.0 p0.056 null |
| GF max | +2.7 z+5 p0.31 null | +0.5 null | +25.9 result | +2.4 null | +4.6 p0.056 null | +17.3 z+4 result | -5.9 p0.056 null | +23.0 result | -2.5 null | +18.2 result | +3.2 null | -0.2 null |

Reading it, with the three overstatements an earlier version of this paragraph made corrected by the skeptic pass.
(i) **B is invisible on the behavioural and DNa02 keys -- not "invisible everywhere", and speed is not its only
`result`.** B vs shipped is `null` on yaw, straightness, hops, leavers and every DNa02 key, which is the behavioural
conclusion and it holds; but the full `room_table.csv` gives B v shipped **8 `result` rows of 86 tested** (far above
the ~0.7 expected by chance): leg_L_hz 2.5457 -> 2.5851 (+0.039, z +4.52), leg_abs_LR_hz +0.0070 (z +4.68),
power_mean_hz 20.413 -> 20.783 (+0.371, z +3.55), AN07B035_L_hz +0.087 (z +3.14), IN12B014_L_hz 15.181 -> 15.677
(+0.497, z +21.8), IN12B014_R_hz +0.403 (z +11.6), IN19A003_R_hz +0.482 (z +4.47), and speed.
**And the speed `result` needs a calibration**: the skeptic re-ran the IDENTICAL shipped configuration at fresh seeds
and its commanded speed moved by -0.0137 mm/s (8.8309 -> 8.8172) against a within-batch run SD of 0.0045, i.e. z -3.03,
p 0.056 -- 60 % of the -0.0227 mm/s "effect" at z -5.0 that B v shipped reports. The within-batch run SD understates
the true between-batch scatter on the seed-locked motor keys, and the paired-seed design protects the rank test, not
the z. Dismissing the speed row as a 0.3 % magnitude is right, and this is the quantitative reason.
(i-b) **AB vs A and ABC vs AC are NOT `null` on every key.** AB v A has 2 `result` rows -- GLNO_L_hz 4.2092 -> 4.8903
(+0.681, z +5.03, p 0.0079) and GLNO_R_hz 4.0961 -> 4.7848 (+0.689, z +5.71) -- and ABC v AC has 9, including GLNO_L_hz
11.900 -> 13.244 (+1.344, z +12.8), GLNO_R_hz 11.769 -> 13.080 (+1.311, z +18.3), yaw_cmd_sd_deg_s +0.881 (z +3.35),
DNa02_abs_LR_hz +0.136 (z +3.27), corr_yaw_hairLR -0.0244 (z -4.27), AN07B035_R +0.0735, AN07B037_a_L +0.127 and
IN19A003_L +0.744. The GLNO effect has perfect rank separation over the 5 paired seeds (A max 4.405 vs AB min 4.737;
AC max 12.022 vs ABC min 13.157), so it is not a multiplicity artefact -- see section 9 item 5(a), which this refutes.
(ii) A does the same thing on top
of shipped and on top of C: DNa02_L +0.49 / +1.07, DNa02 L-R +0.19 / +1.10, straightness -0.13 / -0.47, signed yaw +0.83 /
+3.22 -- and on top of C it halves the hops (-20.9), the airborne fraction (-0.057) and raises the commanded speed (+3.3
mm/s, fewer airborne frames), because the transducer's afferent drive lowers the wing-power MNs (power mean 58.0 -> 47.2
Hz; 20.4 -> 15.2 under A alone, the body thread's finding) without touching the sustained maximum (136.5 vs 137.2,
`null`). (iii) C does the same thing on top of shipped and on top of A: DNa02_L +0.93 / +1.51, hops +43 / +22, every fly
off the table, wing power +69 / +72, straightness -0.14 / -0.48. The two mechanisms are additive on DNa02's rate and on the
yaw amplitude, and C's take-off cost is not removed by A, only halved.

### 4.3 Sidedness per frame (`probe_vnc_drive.sided_frames`; clean walking frames, per fly, then the run mean; SD over 5 runs)

| statistic | A | AB | AC | ABC |
|---|---|---|---|---|
| corr(realised yaw, chordotonal cmd L-R) | -0.147 +- 0.002 | -0.150 +- 0.002 | -0.271 +- 0.010 | -0.300 +- 0.008 |
| corr(realised yaw, leg MN L-R) [shipped +0.261, B +0.261, C +0.147] | +0.034 | +0.036 | -0.012 | -0.043 |
| corr(realised yaw, haltere MN L-R) [shipped +0.148] | +0.078 | +0.078 | +0.066 | +0.068 |
| corr(AN04B003 L-R, chordotonal cmd L-R) | -0.297 +- 0.005 | -0.293 +- 0.002 | -0.120 +- 0.009 | -0.097 +- 0.020 |
| E[AN04B003 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) | -3.38 +- 0.03 | -3.35 +- 0.01 | -1.89 +- 0.19 | -1.85 +- 0.18 |
| corr(DNa02 L-R, chordotonal cmd L-R) | -0.140 +- 0.003 | -0.144 +- 0.002 | -0.217 +- 0.007 | -0.237 +- 0.012 |
| E[DNa02 L-R \| chord L-R > 0] - E[. \| < 0] (Hz) | -0.33 +- 0.02 | -0.35 +- 0.01 | **-0.93 +- 0.10** | **-1.02 +- 0.07** |
| corr(AN04B003 L-R, realised yaw) | -0.064 | -0.077 | -0.134 | -0.152 |
| corr(DNa02 L-R, haltere cmd L-R) | +0.137 +- 0.011 | +0.137 +- 0.016 | +0.167 +- 0.019 | +0.166 +- 0.044 |
| corr(PS196_b L-R, haltere cmd L-R) | +0.151 +- 0.013 | +0.150 +- 0.012 | +0.125 +- 0.019 | +0.144 +- 0.020 |
| corr(PS196_b L-R, realised yaw) | +0.079 | +0.073 | +0.054 | +0.023 |
| chordotonal cmd L-R mean / per-frame \|L-R\| (Hz) | -0.27 / 12.2 | -0.27 / 12.2 | -2.34 / 13.6 | -2.50 / 13.7 |
| \|amp L-R\| / corr(amp L-R, yaw) per fly | 0.016 / -0.59 | 0.016 / -0.50 | 0.063 / -0.11 | 0.067 / -0.12 |

The tripod-locked sided term of the body thread is reproduced under A / AB to the second decimal -- **against the body
thread's arm D, which is what A is** (D: AN04B003 swing -3.40 +- 0.04, DNa02 swing -0.33 +- 0.02, corr(yaw, chord L-R)
-0.144 +- 0.005; here -3.377 +- 0.034, -0.325 +- 0.019, -0.147 +- 0.002). An earlier version quoted -3.39 / -0.35 /
-0.148, which is the arm-C column of `docs/audits/body_sided_state.md` section 4.3, not arm D (skeptic pass); the
replication conclusion is unchanged and if anything closer. The term is three times larger on DNa02
under C (-0.93 / -1.02 Hz per tripod half-cycle) while it is HALVED on AN04B003 (-1.89): C's disinhibition lets the ascending
term through to DNa02 more and mixes more of the rest of the graph into AN04B003. The kinematic sign holds (a left turn
shortens the left steps; corr -0.15 to -0.30) and strengthens under C because the turn itself is larger (chordotonal DC
L-R -2.3 Hz vs -0.27); the leg-MN bias no longer correlates with the yaw at all under the transducer (+0.26 shipped ->
+0.03 / -0.01). None of it is a turn signal: corr(AN04B003 L-R, yaw) stays -0.06 to -0.15 and corr(PS196_b L-R, yaw)
+0.02 to +0.08, the same fraction-of-a-tenth as the body thread found. `|amp L-R|` 0.063 under AC / ABC (0.016 under A) is
the larger yaw's kinematic consequence, and its per-fly correlation with the yaw collapses from -0.59 to -0.11 because the
yaw those flies realise is, on most frames, a landing.

### 4.4 DNa02's input per arm (`decompose_DNa02_{L,R}_per_type.csv`, rate-weighted input in mV/s per post cell; each arm decomposed on ITS OWN shaped weights, the null on the shipped ones; `dna02_decompose_summary.csv`)

| DNa02_L input | shipped | A | B | C | AB | AC | BC | ABC |
|---|---|---|---|---|---|---|---|---|
| E total / I total / **net** | +743 / -1047 / -304 | +1341 / -1251 / **+90** | +754 / -1065 / -310 | +2566 / -2903 / -337 | +1354 / -1268 / +86 | +2833 / -2867 / **-34** | +2573 / -2901 / -328 | +2892 / -2929 / -37 |
| AN04B003/L (leg afferents, k1) | +9.3 | **+352.9** | +9.6 | +36.8 | +351.6 | +356.6 | +37.9 | +352.0 |
| SNpp45/? (unsided afferent, direct) | | +127.4 | | | +127.2 | +128.4 | | +126.7 |
| GNG100/R, PVLP141/R, DNae007/L, PLP012/L | +24, +35, +36, +34 | +22, +22, +23, +31 | +25, +34, +37, +34 | **+123, +121, +117, +107** | +23, +21, +24, +31 | +124, +75, +89, +87 | +125, +119, +118, +107 | +125, +76, +92, +88 |
| PS059/L (haltere afferents, k1) | -91.2 | **-202.5** | -92.3 | -333.7 | -203.9 | **-479.0** | -336.5 | -486.2 |
| IN12B014/R | -90.5 | -65.4 | -93.3 | -129.3 | -65.8 | -93.8 | -129.8 | -97.3 |
| GNG562/L / LT51/L / IN19A003/L | -68.4 / -57.9 / -43.2 | -45.4 / -45.1 / -21.2 | -70.1 / -58.1 / -44.3 | -203.4 / -192.2 / -108.3 | -46.2 / -46.3 / -21.9 | -146.8 / -138.1 / -66.5 | -205.2 / -193.7 / -107.9 | -150.5 / -142.7 / -69.2 |
| LAL046/L, LAL120_a/R, PLP029/L | --, -32.6, -42.3 | -44.7, -42.2, -30.0 | --, -33.2, -42.2 | -69.7, -103.2, -134.0 | -45.1, -42.9, -30.0 | -110.6, -100.3, -94.4 | -70.3, -102.9, -132.7 | -112.5, -102.6, -96.5 |

| DNa02_R input | shipped | A | B | C | AB | AC | BC | ABC |
|---|---|---|---|---|---|---|---|---|
| E total / I total / **net** | +808 / -1170 / -362 | +1332 / -1492 / -161 | +822 / -1194 / -372 | +2204 / -2998 / **-794** | +1344 / -1512 / -169 | +2497 / -3118 / **-621** | +2175 / -2994 / -819 | +2538 / -3182 / -644 |
| AN04B003/R | +4.7 | **+351.9** | +5.0 | +24.2 | +351.1 | +349.5 | +25.1 | +345.8 |
| PLP012/R, PVLP141/L, GNG100/L | +48, +31, +16 | +43, +28, -- | +48, +31, +16 | **+130, +109, +81** | +43, +28, -- | +107, +90, +89 | +129, +108, +81 | +108, +90, +90 |
| PS059/R | -49.6 | **-169.1** | -50.1 | -145.6 | -169.1 | **-373.6** | -142.5 | -375.2 |
| IN12B014/L | -125.9 | -98.6 | -129.2 | -168.1 | -99.8 | -138.3 | -165.7 | -142.6 |
| LT51/R / IN19A003/R / GNG562/R | -68.3 / -59.8 / -54.8 | -66.9 / -57.5 / -35.2 | -69.6 / -62.5 / -56.3 | -196.1 / -101.9 / -150.1 | -69.0 / -58.3 / -36.1 | -169.2 / -76.0 / -110.5 | -197.4 / -100.1 / -150.7 | -174.3 / -77.5 / -112.5 |
| LAL120_a/L, LAL053/R, GNG502/R, IN14B004/L | -47.3, -28.9, -37.4, -28.9 | -56.5, -45.3, --, -- | -47.8, --, -38.4, -29.0 | -151.8, -132.1, -120.5, -102.7 | -57.2, -45.0, --, -- | -135.0, -132.3, -84.2, -80.0 | -152.3, -131.1, -118.6, -102.5 | -137.8, -133.5, -85.6, -82.0 |

(`decompose` `check []` on both targets; the interneuron rates behind the rows, window means over the 5 runs: IN12B014
L / R 15.2 / 12.7 Hz (shipped), 11.1 / 8.5 (A), 22.2 / 19.2 (C), 17.6 / 13.2 (AC); IN19A003 12.1 / 12.2, 5.6 / 11.1, 33.9 /
26.0, 21.9 / 18.8; PS059 8.9 / 4.9, 19.7 / 16.8, 30.5 / 13.6, 43.8 / 35.0; PS196_b 0.2 / 0.5, 8.0 / 6.8, 4.8 / 7.6, 16.6 /
15.2; GLNO 0.06 / 0.06, 4.2 / 4.1, 4.0 / 4.3, 11.9 / 11.8; AN04B003 0.6 / 0.3, 22.9 / 23.3, 2.3 / 1.5, 21.7 / 21.8; run SD
0.01-0.5 Hz on every row.)

Answers. **B changes little of what DNa02 receives, and the E / I TOTALS are within 1.5-2.2 % -- but "every named row
within 3 %" was too strong** (skeptic pass). On DNa02_R, B vs shipped: AN04B003/R +6.56 % (4.695 -> 5.003 mV/s),
AN07B035/L +4.50 %, IN19A003/R -4.49 % (-59.77 -> -62.45, against a shipped run SD of 0.38, i.e. z -7.0, a `result` at
5 v 5), rate_hz +6.6 %. On DNa02_L, ABC vs AC: IN19A003/L -3.99 %, IN12B014/R -3.74 %, LT51/L -3.29 %, and the NET
-8.66 % (-34.17 -> -37.13). The reading stands on the totals -- the 0.02 tone does on DNa02's input what it did on the
behaviour -- but the per-row 3 % bound does not hold. **What A does is the
body thread's mechanism, unchanged by C**: AN04B003/L +353 vs PS059/L -203 is the cancelling pair under A and AB, and
AN04B003 stays at +350-357 mV/s under AC / ABC while PS059 rises to -479 / -486 -- the haltere afferents' inhibition through
PS059 grows under C (PS059 fires 44 / 35 Hz against 20 / 17) faster than the leg-afferent excitation, so the PS059
cancellation the body thread found "outgrown" under A is restored by C on the left side (net +90 -> -34) and deepened on
the right (-161 -> -621). **What C does is a whole-graph rate rise, not a route**: E and I on DNa02 both grow by a
factor **between x2.6 and x3.5** -- on DNa02_L E x3.45 and I x2.77, on DNa02_R E x2.73 and I x2.56 (C / shipped from
`dna02_decompose_summary.csv`; an independent recomputation from `brain._shaped_weights` gives the same pattern, E x3.25
/ I x2.61 un-normalised). An earlier version wrote "both go x3.5", which is wrong by up to 40 % and hides that E grows
faster than I only on the left (skeptic pass). The tonic inhibitors triple (GNG562/L -68 -> -203, LT51/L -58 -> -192, IN19A003/L -43 -> -108) and
the excitatory rows too (GNG100/R +24 -> +123, PVLP141/R +35 -> +121), the net goes MORE negative on both sides, and DNa02
fires at 0.95 Hz because the fluctuation of a 5.5 V/s input on a 7 mV gap is what fires it -- the same disinhibited
operating point the unitary thread read off its `mid` arm (a -11 Hz bias) at a smaller scale. **Why AC drifts left**: the
net input is -34 mV/s on DNa02_L and -621 on DNa02_R, a 587 mV/s asymmetry that the shipped model carries at 58 (-304 vs
-362) and A at 250 (+90 vs -161). **The asymmetry is majority EXCITATORY under AC, not inhibitory** (skeptic pass):
decomposing AC's L-R net gap of +586.8 mV/s gives E_L - E_R = **+335.5 (57 %)** and I_L - I_R = **+251.3 (43 %)**; under
ABC +354 (58 %) / +253; under C alone +362 (79 %) / +95; under BC +398 (81 %) / +93. **Only under A alone is the story
inhibitory** (+9 E / +241 I, 96 %) -- which is the body thread's reading, and it does not carry over to the C arms. One
"unsided" afferent group supplies a fifth of AC's excitatory gap on its own: SNpp45/? delivers **+128.4 mV/s to DNa02_L
against +60.7 to DNa02_R** (under A, +127.4 vs +58.7) -- a row the DNa02_R table above omits while listing it for
DNa02_L. The inhibitory side is still asymmetric in the same direction (PS059 -479 vs -374, with the RIGHT side carrying
more of the rest: LT51/R -169 vs LT51/L -138, LAL053/R -132 vs LAL046/L -111, GNG502/R -84, IN14B004/L -80 with no
left-side counterpart), and the AN04B003 rows are equalised by the cap (+357 / +350) -- but "the excitatory rows are
symmetric and the inhibitory ones are not" is not the mechanism here. What is: the connectome's sidedness expressed
through BOTH an unsided afferent's unequal delivery and the inhibitors' own rates, multiplied by C's disinhibition.
A bias of one sign in 80 / 80 fly-runs is not a turning signal; it is the class of thing the project rule forbids
adopting.

## 5. The efferent compass (24 runs: 8 arms x 3 seeds; gE 2 / gD 15; DNa02_L / _R at 20 Hz for 10 s; 3 s skipped; `compass_table.csv`, `compass_flip_<arm>.csv`, `compass_flip_chain.csv`, `compass_chain_<arm>.csv`)

The body thread's protocol exactly (its section 6) with the arm's LIFParams in force and, under the A arms, the scalar
body carrying `Locomotion.cycle` and the sense fed from `proprio_state` with `read_haltere_sides`; sparse matmul on the
eager path in every arm. All 24 runs `device cuda NVIDIA B200`, blocks `fam_c<seed>`. The pinned fly walks at 9-10 mm/s
against the pin under shipped / A / B / AB (`cyc_speed` 0.009-0.010) and at **5.5-9.1 mm/s** under the C arms (the
disinhibited brain commands less forward drive: `cyc_speed` 0.0055-0.0061 m/s under C and 0.0057-0.0062 under BC, i.e.
5.5-6.2 mm/s, with AC / ABC at 8.0-9.1 mm/s; an earlier version said "6-9 mm/s"); the cycle runs at 7.4-8.5 Hz.

| arm | drift rest / ccw / rest2 / cw (w/s; mean +- SD over 3 seeds) | heading rest / ccw / rest2 / cw (deg/s) | ccw vs rests, cw vs rests (diff, verdict) |
|---|---|---|---|
| shipped | -0.003 +- 0.004 / -0.000 +- 0.002 / +0.003 +- 0.003 / +0.003 +- 0.002 | +1.1 / +101.2 / +1.2 / -92.0 | -0.0001, +0.0029 underpowered |
| A | -0.002 / +0.001 / +0.003 / +0.004 | +1.8 / +97.1 / +2.7 / -85.8 | +0.0005, +0.0031 underpowered |
| B | -0.002 / +0.001 / +0.003 / +0.002 | +1.5 / +102.2 / +0.9 / -92.5 | +0.0001, +0.0014 underpowered |
| C | -0.003 / +0.002 +- 0.006 / +0.004 / +0.003 | +3.1 / +95.3 / +2.1 / -83.6 | +0.0016, +0.0026 underpowered |
| AB | -0.002 / +0.001 / +0.004 / +0.005 | +1.4 / +98.4 / +3.4 / -86.8 | +0.0003, +0.0034 underpowered |
| AC | -0.001 / +0.002 / +0.004 / +0.004 | **+8.8** / +91.7 / **+7.1** / **-73.4** | +0.0009, +0.0027 underpowered |
| BC | -0.002 / -0.001 / +0.001 / +0.003 | +3.8 / +96.4 / +1.0 / -88.1 | -0.0003, +0.0033 underpowered |
| ABC | -0.002 / -0.001 / +0.001 / +0.001 | **+6.6** / +94.3 / **+11.2** / **-72.2** | -0.0002, +0.0011 underpowered |

The bump does not drift under any arm (ideal +-4.0 w/s at the realised +-90 deg/s; the largest per-phase mean is 0.005;
vector strength 0.76-0.79, peak 250-286 Hz, the pinned bump of `cx_shift.md`; the C arms raise the peak by 20-30 Hz). The
fixed left bias of section 4 is on the pinned body too: under AC / ABC the fly turns left at +6.6 to +11.2 deg/s with no
stimulus and its right turn is 20 deg/s weaker than its left (-73 vs +92-94).

Flip table (L-R at ccw minus L-R at cw, per seed, against the rest2 - rest null; every row `underpowered` at 3 v 3 by the
rule, so the per-seed lists carry the evidence; Hz). **These rows are quoted with the reducer's convention** -- its
windowing and per-cell averaging -- and an independent recomputation from the phase `_rec.npz` agrees in substance but
not to the digit on the two downstream types: PS196_b ABC **-3.31 +- 0.20** recomputed against **-3.00 +- 0.29** here,
GLNO ABC **+0.12 +- 0.16** against **-0.02 +- 0.08** (skeptic pass). The qualitative reading is unchanged in both (ABC's
three PS196_b seeds all outside the null's +-1.6; GLNO at zero in every arm), but "reproduced to 0.01 Hz per seed"
belongs only to the wedge rows of section 6, where it really does hold, and not to this table:

| type (depth) | shipped | A | B | C | AB | AC | BC | ABC |
|---|---|---|---|---|---|---|---|---|
| AN04B003 (1) | -0.13 +- 0.15 | **-10.49 +- 0.46** (-10.38 / -11.00 / -10.10; null +1.27 +- 1.22) | -0.03 | -0.33 +- 0.05 | **-10.87 +- 0.21** (null -0.30 +- 0.76) | **-11.52 +- 1.10** (-11.62 / -12.57 / -10.38; null +0.41 +- 0.91) | -0.41 +- 0.11 | **-12.24 +- 1.50** (-10.71 / -13.71 / -12.29; null -0.32 +- 1.15) |
| AN06A026 (1) | +0.60 | -2.17 +- 0.15 | +0.60 | +0.95 | -2.26 +- 0.62 | -4.76 +- 1.02 | +0.67 | -5.05 +- 1.21 |
| PS047_b (2) | -1.57 | -2.71 +- 1.12 | -1.43 | -2.14 +- 2.30 | -3.52 +- 0.46 | -1.38 +- 1.86 | -2.81 +- 0.50 | -3.29 +- 0.25 |
| PS196_b (2) | +0.10 +- 0.33 | -1.14 +- 1.45 (+0.14 / -0.86 / -2.71; null +0.76 +- 1.40) | -0.19 | -0.19 +- 2.36 | -1.48 +- 0.33 (-1.29 / -1.86 / -1.29; null +0.10 +- 0.79) | -2.48 +- 2.22 (-4.57 / -2.71 / -0.14; null -0.43 +- 1.55) | -1.19 +- 1.39 | **-3.00 +- 0.29** (-2.71 / -3.29 / -3.00; null -0.57 +- 1.62) |
| GLNO (3) | -0.33 +- 0.54 | +0.14 +- 0.63 | -0.21 | -0.52 +- 0.15 | +0.10 +- 0.44 | +0.50 +- 0.57 (+0.93 / -0.14 / +0.71; null -0.12 +- 0.29) | -0.19 +- 0.15 | -0.02 +- 0.08 (null +0.12 +- 0.15) |
| PEN_a / PEN_b / EPG (4) | -0.09 / -0.18 / +0.62 | -0.05 / -0.07 / +0.69 | +0.01 / -0.18 / +0.70 | +0.19 / -0.08 / +0.42 | -0.01 / -0.23 / +0.53 | +0.30 / -0.12 / +0.44 | +0.23 / -0.04 / +0.50 | +0.36 / -0.11 / +0.44 |
| PS059 (1; DNa02-driven) | +3.38 | +5.24 | +1.83 | +7.07 | +6.21 | +3.38 | +3.29 | +3.57 |
| DNa02 (the stimulus) | +37.3 | +36.0 | +37.5 | +34.9 | +36.6 | +32.4 | +35.8 | +33.1 |

GLNO's L-R at rest: +27.7 (shipped), +27.1 (A), +29.5 (B), +31.5 (C), +29.1 (AB), +30.6 (AC), +33.7 (BC), +32.3 (ABC) Hz,
the same at every phase. The commanded chordotonal L-R in the turns: -23.5 / +21.3 (A), -23.9 / +21.9 (AB), -28.0 / +26.8
(AC), -29.1 / +24.6 (ABC); the haltere MN L-R -4.3 / +1.2 (A), -4.6 / +1.3 (AB), -4.2 / +3.0 (AC), -4.4 / +2.6 (ABC).

So the compass question has the same answer under every combination: **the ring receives no signed report of the
self-turn.** The report is sided at the ascending neurons in every A arm (AN04B003 -10.5 to -12.2 Hz of flip, 10 x its
null; AN06A026 -2.2 to -5.0), reaches PS047_b / PS196_b at 1-3 Hz two synapses on -- and C makes that second-step report
larger and cleaner (PS196_b -3.00 +- 0.29 under ABC against -1.14 +- 1.45 under A; its seeds -2.71 / -3.29 / -3.00 all
outside the null's +-1.6), because the disinhibited PS196_b fires 16 / 15 Hz instead of 8 / 7 -- and it is gone at GLNO
(|flip| <= 0.5 in every arm against nulls of 0.1-0.5) and at PEN / EPG in every arm. The ascending report grows with C; the
dilution at GLNO does not change. Nothing here is a heading-tracking compass.

## 6. The cx_wedge compass at the shipped gains (16 runs: shipped / B / C / BC x 4 seeds; `wedge_table.csv`)

`probe_unitary compass`: gE = gD = gR = 1, receptor rule as the arm's, compass adaptation off, 10 Hz Poisson background on
the 46 EPG, 1 s settle, one 4-wedge block driven +40 Hz for 2 s, 5 s free; scored with `probe_compass_room.bump_frames`.
Every run `NVIDIA B200`; unrounded values compared (the unitary thread's report rounded before testing; not here).

| arm | survival s | confined post | EPG in, pulse | EPG in / out, post | PEN post | Delta7 post | Ring post | rest of brain | GLNO post | ledger survival / rate / width |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped | 0.00 x4 | 0.00 x4 | 48.6 | 10.37 / 9.81 | 0.041 (0.036 / 0.020 / 0.062 / 0.047) | 10.42 (10.23 / 10.53 / 10.50 / 10.40) | 0.807 | 0.00154 | 0.15 | FAIL x4 / N/A x4 / N/A x4 |
| B | 0.00 x4 | 0.00 x4 | 48.6 | 10.38 / 9.81 | 0.052 | 10.43 | 0.807 | 0.00155 | 0.20 | FAIL x4 / N/A / N/A |
| C | 0.00 x4 | 0.00 x4 | 48.6 | 10.38 / 9.81 | 0.047 | **11.38** (11.15 / 11.56 / 11.51 / 11.30) | 0.864 | 0.00165 | 0.23 | FAIL x4 / N/A / N/A |
| BC | 0.00 x4 | 0.00 x4 | 48.6 | 10.38 / 9.81 | 0.049 | 11.37 | 0.865 | 0.00166 | 0.26 | FAIL x4 / N/A / N/A |

`compass.EPG.bump_survival_s` **FAIL in 16 / 16** (0.00 s against >= 5; no confined frame before, during or after the
pulse), `bump_rate_hz` / `bump_width_wedges` NOT_APPLICABLE x16. `common.compare` vs shipped (4 v 4, floor 0.029):
survival and confinement are **`null` with a zero-SD null (structural)** -- `compare` returns `null`, not
`undetermined`, because the diff is exactly 0 in both arms; `undetermined` would require a non-zero diff, and any
summary of this round that says "undetermined / structural" here should read "null with a zero-SD null (structural)"; EPG in / out, PEN, GLNO `null` in every arm (PEN +0.006
to +0.011 Hz); Delta7 +0.96 Hz (C, z +7.0, p 0.029) / +0.95 (BC) `result`, Ring +0.057 / +0.059 `result`, rest of brain
+0.0001 `result` -- the sub-hertz interneuron shifts the unitary thread reported for `high` (Delta7 11.38 [11.15-11.56],
Ring 0.86, PEN 0.05, default Delta7 10.42 [10.23-10.53]) reproduced here **to 0.01 Hz per seed** on a different device and
backend: the wedge protocol's rates are the seeded Poisson stimulus's, and the recurrent ring contributes nothing
measurable to them at the shipped gains under any of the four brain configurations. B changes nothing in the ring
(EL -> EPG's Octbeta1R row at 0.02 is 0.01 Hz on the EPG).

## 7. The benchmark sections rest / taste / smell / walk / loom_escape (8 runs: shipped / B / C / BC x 2 draws, `--eager`, seeds 0,1,2; `bench_checks.csv`, `bench_table.txt`, `bench_worse.csv`)

| check (criterion) | shipped d0 / d1 | B d0 / d1 | C d0 / d1 | BC d0 / d1 |
|---|---|---|---|---|
| rest.spikes_per_step (< 5) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| taste.MN9_hz (> 2) | 10.93 / 10.93 | **2.48** / 2.48 | **20.08** / 20.08 | 12.03 / 12.03 |
| smell.PN_hz (< 100) | 7.86 / 7.86 | 6.93 / 6.93 | 5.58 / 5.58 | 9.15 / 9.15 |
| smell.KC_active (> 0) | 816 / 816 | 790 / 790 | 867 / 867 | 1121 / 1121 |
| walk.GF_max_hz (< 38) | 4.63 / 4.63 | 9.52 / 9.52 | 8.69 / 8.71 | 13.36 / 13.36 |
| walk.power_max_hz (reported) | 48.48 / 48.48 | 59.05 / 59.05 | 133.7 / 92.4 | 136.9 / 136.9 |
| walk.power_sustained_hz (< 50) | 20.11 / 20.11 | 24.52 / 24.52 | **90.47 FAIL / 44.70 PASS** | **103.89 FAIL / 103.89 FAIL** |
| loom.GF_peak_hz (>= 20) | 43.6 / 47.4 | 50.4 / 51.9 | 40.8 / 62.4 | 55.5 / 54.1 |
| loom.escape_cm (not none) | 3.5 / 3.5 | 3.5 / 3.5 | 3.5 / 3.5 | 3.5 / 3.5 |
| rotate.DNp20_flip_hz (< -2) | -33.5 / -37.2 | -39.3 / -42.0 | -32.1 / -41.7 | -37.4 / -30.9 |
| loom_escape.GF_peak_hz (>= 33) | 44.1 / 56.3 | 48.6 / 49.8 | 66.4 / 60.1 | 59.5 / 73.2 |
| loom_escape.escapes (>= 1) | 3 / 3 | 3 / 3 | 3 / 2 | 3 / 3 |
| **pass / fail of 12** | **12 / 0, 12 / 0** | **12 / 0, 12 / 0** | 11 / 1, 12 / 0 | 11 / 1, 11 / 1 |

Two draws are magnitudes and statuses, not verdicts (the taste / smell / walk sections are seed-locked on the eager path:
identical to the digit between draws in shipped, B and BC; `checks worse in status than the shipped arm's worst draw`:
C d0 and BC d0 / d1 on `walk.power_sustained_hz`, nothing else). What is new against the threads' own suites: (i) B's
seed-locked values are the monoamine thread's H200 values to the digit (MN9 2.48, PN 6.93, KC 790, GF 9.52, power 59.1 /
24.5), a cross-device replication of a deterministic section; (ii) under C the walk section is NOT seed-locked on the eager
path -- `walk.power_sustained_hz` 90.47 in draw 0 and 44.70 in draw 1 (`power_max` 133.7 / 92.4), one side of the 50 Hz bound
each, where the unitary thread's three native draws read 89.3 / 102.9 / 96.4; so the "least-cost bracket" is a coin toss
against the bound on this path and a fail on the native one; (iii) BC fails the same check in both draws at 103.9 (both
identical) with `taste.MN9_hz` 12.0 -- B's downward tone (10.9 -> 2.5 alone) and C's disinhibition (10.9 -> 20.1 alone)
partly cancel on MN9 and add on the wing-power route. The `loom_escape` GF peaks are higher under every C arm --
**59.5-73.2 against the shipped arm's 44.1-56.3** (BC draw 0 is 59.45 and shipped draw 1 is 56.29, so the two ranges
nearly touch; B's 48.6 / 49.8 sits inside the shipped range) -- the room's take-off cost seen from the loom. The point
stands; the brackets an earlier version gave (60-73 vs 44-56) were off by a hertz at each end.

## 8. Answer: which combination gets closest, by how much, at what suite cost

**Closest to emergent turning: A (with or without B), and it is not close.** It is the only combination whose added
turning is made of walking frames on the table (94 % clean), whose take-offs stay near the shipped level (0.36 vs 0.06 per
fly) and whose wing-power / GF rows do not move (`null`), while DNa02 fires (0.5 / 0.4 Hz) from a mechanism the data imply
(the leg afferents at literature-typical rates through AN04B003, the cancelling pair against PS059). By how much: clean yaw
SD x2.8 (2.75 -> 7.67 deg/s, z +60), median |yaw| x2 (0.74 -> 1.49), p99 15.8 -> 24.7, straightness 0.995 -> 0.867, net
heading change 0.03 -> 0.18 turns per fly, speed +0.9 mm/s, every one `result` at 5 v 5 -- against an animal that saccades at
200-450 deg/s every ~250 ms (section 4): no clean frame above 100 deg/s in any run, 0.5 % above 30. It carries a fixed left
drift (+1.0 deg/s in 15 / 16 flies; DNa02_L > DNa02_R in 63 / 80 fly-runs) that is the connectome's own asymmetry, the leg
L-R stays a fixed positive bias in 80 / 80 fly-runs, and its suite cost is unmeasurable by the suite (no body) and nil in
the room. B on top of it changes nothing (`null` on every key).

**Largest yaw numbers: AC / ABC, and they are a bias on a hopping fly.** Clean yaw SD 12.8-13.4, median |yaw| 7.4-8.1, p99
39-42, straightness 0.39, signed yaw +5.2 / +6.2 deg/s in 15.8 / 16 flies, DNa02 2.0 / 0.9 Hz with a +1.1 Hz L-R of one sign
in 80 / 80 fly-runs -- on the 17 % of the window before the fly leaves the table (all 16 within 11 s), with 22-23 hops per
fly and 8 % of the time airborne, at C's suite cost (`walk.power_sustained_hz` 90 / 45 and 104 / 104 against < 50; room
power sustained 137 vs 67). The transducer halves C's hops (43 -> 22, z -31) and restores the commanded speed (5.3 -> 8.6
mm/s); it does not make C's disinhibition into steering. The one sign flip of the leg L-R in the round (2 + 1 of 160
fly-runs under AC / ABC) is a footnote, not a mechanism.

**A heading-tracking compass: none.** No bump at the shipped gains under any brain configuration (0 / 16 survival; PEN
0.04-0.05 Hz; C's Delta7 +1 Hz is the unitary thread's number again), no drift of the pinned bump under any of the eight
arms (|drift| <= 0.005 w/s vs 4.0 ideal), no signed report at GLNO / PEN / EPG under any arm; the ascending report (AN04B003
-10 to -12 Hz) and its second step (PS196_b -1.1 -> -3.0 Hz under ABC) grow with the disinhibition and are diluted at GLNO
exactly as before. On the compass axis the round's three mechanisms, combined, move nothing the ledger scores.

**Suite cost, by arm**: shipped 12 / 0 x2; B 12 / 0 x2 (MN9 2.48 seed-locked); A / AB unscored by the suite (no body),
nil in the room; C 11 / 1 and 12 / 0 (`walk.power_sustained_hz` 90.5 / 44.7); BC 11 / 1 x2 (103.9); AC / ABC = C / BC on the
suite, plus the room's 16 / 16 leavers and 22 hops per fly.

## 9. What is a mechanism the data imply, and what is still missing (nothing is adopted)

Mechanisms the data imply, as measured here:

1. **The leg afferents drive DNa02 (A).** The cancelling pair AN04B003/L +353 vs PS059/L -203 mV/s, DNa02 at 0.5 / 0.4
   Hz, the sided term per tripod half-cycle (-0.33 Hz on DNa02, -3.4 Hz on AN04B003) -- reproduced on a second device and
   backend to the second decimal. What is still unproven is the body thread's own caveat: whether the phase structure
   contributes anything beyond the afferent LEVEL (its level-matched control arm, `'all'` with `mn_ref_hz` ~3.5 Hz, 5 seeds
   in one block, has not been run by anyone). Nothing in this round separates them.
2. **The monoamine receptor rows are data; the 0.02 scale is not, and at that scale the class is inert** on the behaviour,
   on DNa02's input (every row within 3 %), on the ring (0.01 Hz) and on the compass -- with `taste.MN9_hz` 10.9 -> 2.5 as its
   only footprint. The three-way class split, the KC>MBON plasticity module and the VNC receptor rows the monoamine thread
   listed are what would make B a measurable mechanism; none exists yet.
3. **The per-transmitter scale is not a mechanism at any bracket on record.** I / E 0.75 is the chloride argument in a
   current-based model; what it does is multiply every input of DNa02 by 3.5, fire the wing-power motor at 137 Hz and the
   DNa02 pair from its own fluctuation, and express the connectome's sidedness as a 1.1 Hz bias once the afferent drive is on.
   A measured unitary IPSP (or reversal potentials) and an ACh-only family remain the prerequisites the unitary thread named.

What is still missing for emergent turning and a heading-tracking compass, in the order the data put it:

1. **A signed steering command.** The sided afferent report exists (AN04B003 flip -10 to -12 Hz; corr(DNa02 L-R, chord L-R)
   -0.14 to -0.24) and DNa02 follows it per frame, but the realised yaw never follows DNa02's sided term: corr(AN04B003 L-R,
   yaw) -0.06 to -0.15 with the KINEMATIC sign (the turn shapes the afferent, not the reverse) in every A arm. No route in the
   graph as wired turns a sided VNC report into a steering-side DNa02 rate that leads the yaw; the round's mechanisms change
   the amplitude of the report, not its direction of causation.
2. **The leg L-R fixed bias** (+0.15 Hz, positive in 397 / 400 fly-runs of this batch): a body-model fact of the leg-MN
   readout that no brain-side mechanism touched; the three exceptions are under AC / ABC where the readout is dominated by
   landings.
3. **A saccade generator.** The animal turns in discrete 200-450 deg/s events; nothing in any arm produces a clean frame
   above 100 deg/s, and the yaw the model adds is a slow meander (median 1.5-8 deg/s) plus a drift.
4. **The compass report beyond GLNO**: the dilution AN04B003 (-11 Hz) -> PS196_b (-3 Hz) -> GLNO (0 Hz) is the wiring's
   fan-in and the sign-0 GLNO -> PEN link, unchanged by any of the three mechanisms; the ring at the shipped gains has no
   attractor for any report to move (0 / 16 bumps), and under the experiment gains the bump is pinned.
5. **The three threads' own adoption lists stand unchanged** (body_sided_state.md 8, monoamine_slow_term.md 4,
   unitary_strength.md 7): the level-matched control, the walking-kinematics ledger rows, `MotorRates.haltere_L/_R` as
   fields (an owner decision), the per-transmitter slow split, a measured IPSP, an ACh-only family, the suite kept in >= 4
   draws. What this integration adds to them: (a) **B and A DO interact, and a future B arm must be scored WITH the
   transducer as well as without it.** An earlier version of this item said the opposite ("B and A do not interact at 0.02
   -- any future B arm can be scored without the transducer"); the batch's own runs refute it (skeptic pass). B's
   footprint on GLNO is `null` on the shipped background (0.0557 -> 0.0611 Hz, z +0.35, p 1.0) and a `result` on every
   background where the transducer has woken the VNC: **+0.681 / +0.689 Hz on A (z +5.0 / +5.7), +1.344 / +1.311 Hz on AC
   (z +12.8 / +18.3), +0.241 Hz on GLNO_R over C (z +6.33)** -- a 10-30x background-dependent effect on the exact relay
   type this audit's compass narrative turns on, and mechanistically expected: an additive slow term scales with the
   presynaptic rate, so it is invisible by construction in a model whose VNC is silent. (The compass conclusion itself is
   unaffected: GLNO's L-R is unchanged, A +0.113 -> AB +0.105, AC +0.131 -> ABC +0.164.) (b) any C candidate must be
   scored in the room with the transducer ON as well as off, because A halves its
   hops and raises its DNa02 bias, so a room verdict on C alone does not transfer; (c) a house (B200) rerun of the H200 room /
   compass rows of the other threads is a replication across devices AND backends, and the shipped arm's rows here are the
   reference for that (section 1).

## Owner decisions (2026-09-14)

Recorded here because the round's commit rests on them; nothing in this audit's numbers changes.

* **(a) `LIFParams.w_syn_by_nt` is KEPT**, as an opt-in instrument: default `None`, byte-identical shaped weights under
  `None` / `{}` / an all-ones dict, pinned by `tests/test_unitary.py` (and `tests/test_bit_identity.py`). `flyverse/brain.py`
  was on this round's never-edit list and thread unitary used the "new `LIFParams` field unavoidable" carve-out; both of
  the carve-out's conditions are met and the field stays.
* **(b) `MotorRates.haltere_L / haltere_R` as fields is DEFERRED.** `motor.read_haltere_sides` stays a free function
  beside `MotorRates`, and `tests/test_bit_identity.py`'s recorded golden (which hashes `asdict(fb.motor())` field by
  field) is untouched. Promoting the pair to fields remains a one-line `read_motor` change plus a golden regeneration,
  and remains an owner decision (`docs/audits/body_sided_state.md` 8 item 4).
* **(c) The adopt-alone rate-half is restated as a two-sample exact Poisson comparison (two-sided) of the candidate's
  and the baseline's pooled take-off counts**, replacing "the candidate's point estimate inside the baseline's exact
  Poisson 95 % CI" -- the containment form is not a 5 % test (14.9-15.6 % false failure at 3-6 batches per arm, not
  improving with n) and it was applied asymmetrically. The transducer arm and both retirement candidates are to be
  re-read under that form in the **next guard round**; no number changes now
  (`docs/audits/guard_suites_r3.md` 4, `anti_runaway.md` round 7).

## 10. Files, provenance and reproduction

* Code (this task): `scripts/probe_round3_integrate.py` only (`arms` / `room` / `compass` / `wedge` / `bench` / `plan` /
  `verify` / `analyse`; the `patched` context manager is where the arm reaches the Brain; `_stamp` is the read-back).
  Depended on, authored by other round-3 tasks, not at `origin/main` (section 3): `flyverse/body.py`, `batch_body.py`,
  `batch_sim.py`, `motor.py`, `senses.py`, `scripts/probe_vnc_drive.py` (body-state); `flyverse/brain.py` (`w_syn_by_nt`),
  `scripts/probe_unitary.py` (unitary); `scripts/probe_monoamines.py`, the ledger rows (monoamines).
* Data: `out/r3int/` (40 room x 5 files, 24 compass x 17 files, 16 wedge x 3, 8 bench x 2, `batch.sh`, `predeclared.json`,
  `submit_tree.txt`), console `out/r3int_cluster.log`, `out/r3int/verify/` (`verify.json`, `verify_runs.csv`),
  `out/r3int/analysis/` (`analysis_console.txt`, `room_table.csv` / `.txt`, `pairwise.txt`, `compass_table.csv`,
  `compass_flip_<arm>.csv`, `compass_flip_chain.csv`, `compass_chain_<arm>.csv`, `wedge_table.csv`, `bench_checks.csv`,
  `bench_table.txt`, `bench_worse.csv`, `decompose_DNa02_{L,R}.json` / `_per_type.csv`, `dna02_decompose_summary.csv`,
  `summary.json`). The analysis was emitted twice: `analysis_console_run1.txt` is the first pass; the reducer then gained two
  table rows (`speed_cmd_mean_m_s`, `yaw_sd_all_deg_s`) and was re-run on the same fetched batch in full (INTERP 10.4 item 11);
  `analysis_console.txt` and every table are the second pass, whose other numbers equal the first's (the reducer is
  deterministic on the CPU).
* Reproduce: `PYTHONIOENCODING=utf-8 python scripts/probe_round3_integrate.py arms`; `... plan --dir out/r3int --runs 5
  --compass-seeds 3 --wedge-seeds 4 --draws 2`; `bash out/r3int/batch.sh`; `... verify --dir out/r3int`; `... analyse --dir
  out/r3int --out out/r3int/analysis` (CPU, ~15 min; the `sided_frames` and `decompose` steps are the slow part). A house rerun
  is a new set of B200 draws (the GPU path is not run-to-run reproducible at a fixed seed); the CPU smokes in
  `out/r3int/smoke/` (`room_ABC_r0`, `compass_ABC_r0`, `wedge_BC_s0`, `bench_BC_d0`, `bench_C_d0`) are the only deterministic
  runs and they are smokes.
* The `null` verdict string in the CSVs: `pandas.read_csv(..., keep_default_na=False)` is needed to read it back as the
  verdict it is.
* Skeptic-pass artefacts, not this task's: `out/r3int_skep/` (10 room runs, shipped + A at fresh seeds 5-9,
  `r3iskep-9ac46f`, house <cluster-node>, 8 x B200, one block `fam_skep`, 10 jobs / 0 failed / 27.2 min), `out/r3int_skep/batch.sh`
  and `out/r3int_skep_cluster.log`. **It is a same-device, same-backend re-draw at fresh seeds -- a seed / batch check,
  not a device check.** Every key of that check is `null` for both arms against this batch's seeds 0-4, so no seed-set or
  batch effect stands behind the numbers above; and it is the source of the speed calibration in section 4.2 and the
  window-matched AC v A numbers in section 4. Nothing under `flyverse/`, `docs/` or `scripts/` was changed by that pass.
