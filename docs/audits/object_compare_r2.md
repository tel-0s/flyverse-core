# The fixed-anatomy model comparison, round 2 (`scripts/object_round2_compare.py`) -- Neurome intake, experiments 2 + 3

**Status: ANALYSED -- sections 3-8 are written from `out/interp/objr2c/compare.json`; sections 0-2 predate the batch
(the predeclaration stamp is `out/objr2c/predeclared.json` at 2026-09-13T22:04:04Z, not this document, whose mtime is
after submission -- see section 9). Section 6.1 is a CPU re-analysis of the stored recordings, added after the
batch.** Nothing is adopted; every hook stays off by default (the project rule); the arms are
diagnoses on fixed anatomy, not proposals.

Read first: `D:\Projects\neurome\reports\flyverse-size-tuning-intake.md` (the corrections and the recommended order),
`docs/NEUROME_INTERFACE.md` 3b items 2-3 (what was agreed), `docs/audits/deficit_object.md` 0 / 4 / 6 (where the
figure is lost, the held-carrier arms, why an output rectifier cannot recover cancelled terms),
`docs/audits/optic_stream_hooks.md` (the four opt-in fields, their equations, the liveness batch and its open
level-shift question), `docs/audits/object_matched_assay.md` (the matched sphere ladder) and
`docs/audits/object_synthetic_stimuli.md` (the radiance-path stimuli and the RF localizer), `docs/INTERP.md` 2.4 /
10.2 / 10.4 (replicates, verdict vocabulary, process rules).

Files: `scripts/object_round2_compare.py` (`plan` / `submit` / `run-job` / `check-job` / `bench` / `verify` /
`analyse` / `streams`), `out/objr2c/` (`batch.sh`, `jobs.json`, `predeclared.json`, `tree_state.json`,
`submit_console.txt`, then `sph/`, `spec/`, `bench/`, `verify.json`, `per_body_sphere.csv`),
`out/objr2c_cluster.log` (the cluster console), `out/interp/objr2c/compare.json` (the Result). The CPU smoke of the
analysis on copies of prior runs is `out/objr2c_smoke/` (`analyse_console.txt`, `verify.json`; nothing in it is a
number of this comparison).

## 0. Predeclared reading rules (written before the batch was submitted; `out/objr2c/predeclared.json`)

The rules are the `PREDECLARED` dict of `scripts/object_round2_compare.py`, dumped by `plan` and copied into the
Result unchanged by `analyse`. In prose:

* **The question.** Which mechanism class -- rectification, adaptation, spatial suppression, or none -- makes the
  small-field stage (T3, T2) carry a small-object figure on the MATCHED sphere ladder with specificity preserved, at
  what magnitude, and whether LC11 output follows. A diagnosis on fixed anatomy; the graph is untouched; nothing is
  adopted.
* **Replicate unit.** One process = one (A, B) pair under one brain seed = one run. A cluster job is a container of
  sequential processes. Runs, not cells, not seeds.
* **Counts.** Sphere: 5 object runs per rung x 5 rungs (4.5 / 8.8 / 11 / 20 / 30 deg) + 5 blank/blank nulls, per arm,
  every arm in ONE submission. Specificity: 4 runs per stimulus x 7 stimuli + 4 blank/blank nulls, per arm. Transfer:
  3 benchmark draws per arm.
* **Why five and not six.** The primary families have THREE members (the three small rungs), so Holm's smallest
  threshold is 0.05 / 3 = 0.0167, above the 5 v 5 exact-U floor 0.0079 (`common.p_floor`): a fully separated member
  can survive Holm at five runs per arm. (The baseline's 12-member families needed six.) The specificity battery at
  4 v 4 (floor 0.029) is decidable per stimulus unadjusted and is read as an exploratory screen per stimulus, not as
  one Holm family.
* **Window rule.** The baseline's (`object_round2_baseline.PREDECLARED['window_rule']`): a body's window is its
  fitted box in the shipped localizer map, else in the fb0 map, else the anatomical column of
  `trace.column_of_cells` with the type's median fitted width, else the whole window. The maps are those on disk at
  analysis time (`--rf-maps`; default `out/objr2/rfmap_ship.csv`, `out/objr2/rfmap_fb0.csv` if the baseline batch
  has landed, then `out/synth2/rfmap_150_fb0_p3.csv`). A stimulus-driven RF map is a property of the retinotopy,
  which no arm changes, so one map serves every arm; the map used is recorded in the Result.
* **P1, the carrier figure.** T3 `diff_signed_best_cell` (max over cells of |mean_t dr_A - mean_t dr_B|, whole
  window, `probe_object_matched`'s definition) per run. Per hook arm the family is the three small rungs {4.5, 8.8,
  11} = 3 members, Holm within, and two questions per member: (i) the arm's object runs vs the arm's own blank/blank
  runs (is there a figure), (ii) the arm's object runs vs the base arm's object runs at the same rung (does the
  mechanism change it). Tie-aware exact permutation U; verdict from `common.compare`. **An arm CARRIES the
  small-object figure at T3 if at least one small rung reads `result` with p_holm <= 0.05 on BOTH questions (with a
  positive diff on (ii)).** T2 is the same family as the second carrier; "the small-field stage carries" needs both.
* **P2, LC11.** Per run the population MEDIAN over the 143 LC11 bodies of the per-body RF-windowed time-mean received
  drive (A - B, mV). Family = {4.5, 8.8, 11} x drive_median = 3 members, Holm within, the same two questions.
  **LC11 FOLLOWS if at least one small rung reads `result` with p_holm <= 0.05 vs the arm's null AND vs base.**
* **fb0 (arm B).** Its blank/blank null has SD ~0 for optic quantities, so question (i) returns `undetermined` (read
  as a magnitude with p, counted as a figure when diff > 0); question (ii) vs base is a plain 5 v 5 call.
* **Secondary, exploratory.** The large rungs {20, 30} (LC10a's target range); LC11 / LC10a spikes_median,
  drive_mean, drive_max, the whole-window `diff_max_over_cells_mean_mv` and `diff_rate_hz_max_cell`; LC10a
  everything; the preference tests (Spearman over the 25 object runs, small-vs-large contrast) per arm; T2 / Tm5Y /
  TmY21 `diff_signed_best_cell`, `diff_signed_mean`, `diff_abs_best_cell_mean` kept distinct; the RF-windowed medians
  of T2 / T3 / Tm5Y over fitted bodies. **The level-shift question** the hooks audit raised: the blank-arm levels per
  arm (T3 / T2 `dev_mean_b`, `dev_abs_mean_b`; LC11 / LC10a `drive_mean_mv_b`, `rate_hz_mean_b`) and the flicker
  stimulus (spatially uniform: a pure level shift gives a figure there, a rectified transient of a local object does
  not).
* **Specificity.** Per arm x stimulus: (i) vs the arm's blank/blank null, (ii) vs base on the same stimulus, both 4
  v 4, unadjusted. **An arm RELEASES bar / grating / flicker responses at LC11 if, for that stimulus, (ii) reads
  `result` with diff > 0 on `diff_max_over_cells_mean_mv` or `diff_rate_hz_max_cell`** (Keles et al. 2020: the
  LC11-specific Rdl disruption reduced small-object responses without releasing bar / grating responses). **T2 and
  T3 KEEP both transitions if the arm's mean `diff_signed_best_cell` on the ON flashes and on the OFF flashes are
  each >= 0.5 x the base's** (a magnitude rule, stated as such, the 4-run scatter beside it). Bright vs dark 11-deg
  squares: the ratio of LC11 means, a magnitude, no call.
* **Transfer.** `scripts/benchmark.py --sections motion,loom_escape` under each arm, 3 draws: the benchmark's own
  PASS / FAIL / KNOWN GAP on `motion.min_dsi`, `motion.correct_directions`, `loom_escape.GF_peak_hz`,
  `loom_escape.escapes`, and the values against the base arm's draws (3 v 3: magnitudes, `underpowered` by
  construction). **An arm COSTS a section if any draw fails a check the base passes in every draw.**
* **Classification.** Each finding is classified under the project rule: a mechanism the physiology implies (Keles
  2020: T2 / T3 respond to both ON and OFF, T3 -> LC11 excitatory; Tanaka & Clark 2020: fast-adapting size-tuned
  inputs) vs a hand-set number (the modes, tau, gain, k, radius are hypotheses, not data). NOTHING IS ADOPTED.

## 1. The arms (`ARMS`; `streams` prints the flags every probe receives)

| arm | letter | `--optic` override | what it is |
|---|---|---|---|
| `base` | A | none | the shipped sum (OpticParams defaults) |
| `fb0` | B | `gain_fb=0.0` | the deterministic lobe: the exact-arithmetic control |
| `rectify` | C | `stream_rectify=[[^(Mi1\|Tm3\|Tm2)$, ^T3$, pos], [^(Tm1\|Tm4)$, ^T3$, neg], [^(Tm2\|L5\|Tm3\|Mi1)$, ^T2$, pos], [^C3$, ^T2$, neg]]` | per-stream half-wave rectification of the T3 and T2 carriers before their sums; weights and signs untouched |
| `adapt100` | D | `stream_adapt=[[^(Mi1\|Tm3\|Tm2\|Tm1\|Tm4)$, ^T3$, 100, 1], [^(Tm2\|L5\|Tm3\|Mi1\|C3)$, ^T2$, 100, 1]]` | the same streams minus a 100 ms fast-adaptation state (gain 1: high-passed) |
| `adapt300` | D' | the same at 300 ms | |
| `suppress` | E | `spatial_suppress=[[^(Mi1\|Tm1\|Tm2\|Tm3\|Tm4)$, 0.5, 10]]` | centre-surround (k 0.5 within 10 deg, same type) on the medulla carriers, every target |
| `rect_adapt` | F | C + D | |
| `rect_supp` | G | C + E | |

The streams are the carrier classes `deficit_object.md` 2 names, split by MEASURED figure sign (not by ON / OFF
pathway): at T3 the raising {Mi1, Tm3, Tm2} and the lowering {Tm1, Tm4} (all excitatory, exact tier -- the two hold
arms of `deficit_object.md` 4.3); at T2 the raising {Tm2, L5, Tm3, Mi1} (+2.0e-5) and the lowering {C3} (-1.9e-5,
cancellation 0.99). **C3 -> T2 is inhibitory**: the `neg` half-wave keeps the sign of W (`x >= 0`, `W x <= 0`), so the
hook's inhibitory guarantee -- the one the hooks skeptic found vacuous on the T3-only batch, where all 26,833 matched
entries were excitatory -- is exercised on a real stream here. `spatial_suppress` has no post regex (it transforms
the presynaptic cell's signal for every target), so E / G suppress the medulla carriers for every target, T2 / T3
included, and the arm's own `hook_info` (recorded per run) says how many entries onto how many cells.

What the `neg` mode assumes, stated up front (the hooks audit's caveat, repeated so it travels with any number
below): it preserves the sign of W but inverts the sign of the SIGNAL -- an excitatory carrier that drops below its
operating point produces excitation. That is the reading under which "T2 / T3 respond to both ON and OFF
transitions" (Keles 2020) could arise from opposite-figure carriers through same-sign synapses; it is a hypothesis
about the presynaptic transfer, not a fact of the connectome, and the arm exists so it can fail.

## 2. The batch (`objr2c`, 20 jobs = 520 processes, ONE submission)

`python scripts/object_round2_compare.py plan --out out/objr2c --name objr2c --minutes 240` (defaults: 5 runs, 4 spec
runs, 3 bench draws, 12 s after 3 s settle for the sphere, 6 s after 2 s for the spec battery) ->
`python scripts/object_round2_compare.py submit --out out/objr2c` (one `scripts/cluster_run.py` call, `--fetch
out/objr2c/`, console teed to `out/objr2c_cluster.log`).

* **8 sphere jobs** (`sph_<arm>`, 30 processes each): `probe_object_matched.py run` at 4.5 / 8.8 / 11 / 20 / 30 deg
  x seeds 0-4 + 5 nulls, elevation 0, 5 cm, +-50 deg at 40 deg/s, dark ball, eye 0.15 m, headlamp -- the baseline's
  protocol verbatim, with the arm's `--optic` flags.
* **4 spec jobs** (`spec_<arm>_<arm>`, 64 processes each): `probe_synthetic_stimuli.py record` for the 11-deg dark and
  bright squares on the sphere's arc, a 7-deg dark bar on the arc, a 30-deg grating (contrast 0.5, front->back), 2-Hz
  full-field flicker (0.5), isolated ON and isolated OFF 15-deg flashes at (-30, 0) deg (0.5 s on, 1.5 s period), and
  a blank/blank null, x seeds 0-3, per arm. The flash sits at (-30, 0) and not at (0, 0) because the shipped retina
  has no column within 4.4 deg of (0, 0) and two within 7.5 (the equatorial hole the synthetic skeptic measured),
  against 3 / 8 at (-30, 0) on the same arc (`FLASH['why_here']`, from `col_az_el`).
* **8 bench jobs** (`bench_<arm>`, 3 processes each): `object_round2_compare.py bench --arm <arm> --seed k`, which
  installs the arm through `interp_trace.install_overrides` (the path every probe uses) and runs
  `benchmark.py --sections motion,loom_escape --seeds k --json`, writing a provenance sidecar (`_prov.json`: resolved
  LIFParams / OpticParams, device, cache fingerprint, source fingerprint, the benchmark's checks).

Every job line carries `mkdir -p ... && source .venv/bin/activate && python -c 'import torch; assert
torch.cuda.is_available()' && ... run-job`, and `run-job` regenerates the job's process list ON THE BOX from the same
arguments, runs the processes in sequence (a failed process does not stop the next) and exits 1 unless every expected
output exists and every console says `device cuda` (`check-job`), so the cluster log's `<n> job(s), 0 failed` means
what it says. The tree the batch shipped from is `out/objr2c/tree_state.json` (commit, porcelain status, sha256 of
every simulation source): it carries other tasks' uncommitted edits (`flyverse/optic.py` = the hooks task's, the
proprioception task's `fly.py` / `senses.py`), so `provenance.source_fingerprint` -- not the commit -- identifies the
code of every run (`docs/INTERP.md` 10.4 rule 6).

`verify` (CPU, before any analysis): the cluster log's `0 failed` line, every expected output present,
`probe_object_matched.verify_runs` on the sphere runs (footprint from the captured radiance, the elevation band,
body coverage, console-vs-meta device), the resolved OpticParams of every run against its arm (every hook field, and
`gain_fb`), the Torch-substep warning in every hook-arm console and in no plain-arm console (the hook's liveness on
the sphere path, where `hook_info` is not recorded), `probe_synthetic_stimuli.verify_dir` on the spec runs plus
`hook_info.active` against the arm, and the bench sidecars' checks / device / resolved `stream_rectify`.

The CPU smoke (`out/objr2c_smoke/`: the skeptic's 5 + 5 runs at 11 deg copied under `base_` and, as a deliberate
fake, under `rectify_`; the batch-1 synthetic families copied likewise; no bench): `analyse --no-verify --skip-bench`
runs end to end (every P1 / P2 row `null` or `underpowered`, `ratio_to_base` 1.0 on the copies, `Result.check()`
none), and `verify` flags the fake arm on both checks it should -- `stream_rectify differs from arm rectify: None`
and `hook arm without the Torch-substep warning` on all 10 sphere copies, `optic overrides {} vs arm rectify` /
`hook_info.active False` on the 6 spec copies.

## 3. What ran, and how the data got here

Run id `objr2c-a9af1d`, **20 jobs = 520 processes, one submission** (8 sphere x 30, 4 spec x 64,
8 bench x 3), submitted 2026-09-13 at 22:06:37Z from the tree of `out/objr2c/tree_state.json`
(commit `c78cb93d...` plus other tasks' uncommitted edits; `provenance.source_fingerprint`, not
the commit, identifies the code of every run). The `cluster_run.py` client was killed before the
fetch, so **`out/objr2c_cluster.log` carries no `20 job(s), N failed` line and nothing here claims
one**; the batch was fetched by hand per box and the replacement check is
`object_round2_compare.py verify` -> `out/objr2c/verify.json`:

* `expected_missing`: **[]** -- every expected output present (240 sphere runs, 256 spec runs,
  24 bench sidecars).
* Sphere: 240 runs, `el_band_spread_deg` **0.988**, verdict `same band`, devices
  {NVIDIA B200, NVIDIA H200}, **console `device cuda` true in 240 of 240**.
* Spec: 256 runs, `n_bad` **0** (checksums, arms, console-vs-provenance device, `hook_info.active`
  against the arm).
* Bench: 24 sidecars, all `ok`, with the benchmark's own checks and the resolved `OpticParams`.
* **Hook liveness: the Torch-substep warning appears in 30 of 30 consoles of every hook arm
  (`rectify`, `adapt100`, `adapt300`, `suppress`, `rect_adapt`, `rect_supp`) and in 0 of 30 of
  `base` and `fb0`.**
* The only `problems` entry is `cluster log: None`, and `analyse` was run with `--force` for it.

Windows are the baseline's (`summary.windows`): maps `out/objr2/rfmap_ship.csv`,
`out/objr2/rfmap_fb0.csv`, `out/synth2/rfmap_150_fb0_p3.csv`. **LC11 143/143 `anat`** (no map
fits an LC11 body at `z_min` 5); LC10a 262 `anat` + 7 `rf:rfmap_ship` + 6 `rf:rfmap_fb0`;
T3 1,777 `anat` + 96 + 67; T2 1,409 + 114 + 107; Tm5Y 671 + 147 + 80; TmY21 361 + 8 + 3.
**No statement below should call an LC window a measured receptive field.**

## 4. P1 and P2: does an arm carry the small-object figure, and does LC11 follow

Predeclared: family = the three small rungs {4.5, 8.8, 11}, Holm within, two questions per member
-- (i) the arm's object runs vs the arm's own blank/blank runs, (ii) the arm's object runs vs the
**base** arm's object runs at the same rung. An arm CARRIES at T3 if at least one small rung reads
`result` with `p_holm <= 0.05` on BOTH; "the small-field stage carries" needs T3 and T2 both.

**T3 `diff_signed_best_cell` (max over cells), small rungs, mean +- SD over 5 runs:**

| arm | 4.5 deg | 8.8 deg | 11 deg | own null | (i) | (ii) vs base | ratio to base |
|---|---|---|---|---|---|---|---|
| base | 0.00349 +- 0.00127 | 0.00354 +- 0.00097 | 0.00500 +- 0.00183 | 0.00290 | null / null / `result` p_holm 0.095 | -- | 1.00 |
| fb0 | 0.00278 +- 0 | 0.00529 +- 0 | 0.00527 +- 0 | 0.0 (SD 0) | undetermined x3 | null x3 | 0.80 / 1.50 / 1.05 |
| **rectify** | **0.02097 +- 0.00110** | **0.03712 +- 0.00034** | **0.04498 +- 0.00067** | 0.00847 | **`result` x3** z 14.6 / 33.5 / 42.7 | **`result` x3** z 13.7 / 34.6 / 21.9 | **6.01 / 10.49 / 8.99** |
| adapt100 | 0.00314 | 0.00311 | 0.00329 | 0.00313 | null x3 | null x3 | 0.90 / 0.88 / 0.66 |
| adapt300 | 0.00327 | 0.00357 | 0.00353 | 0.00310 | null x3 | null x3 | 0.94 / 1.01 / 0.71 |
| suppress | 0.00375 | 0.00399 | 0.00403 | 0.00303 | null x3 | null x3 | 1.08 / 1.13 / 0.80 |
| rect_adapt | 0.01619 | 0.01972 | 0.02119 | **0.00837** | null x3 (z 1.62 / 2.35 / 2.66) | **`result` x3** z 10.0 / 16.7 / 8.9 | 4.64 / 5.58 / 4.23 |
| rect_supp | 0.02109 | 0.03653 | 0.04257 | **0.01759** | null x3 (z 0.35 / 1.87 / 2.46) | **`result` x3** z 13.8 / 34.0 / 20.5 | 6.05 / 10.33 / 8.51 |

**T2** is the same picture at a smaller gain: `rectify` `result` on both questions at all three
rungs (ratios 3.75 / 3.27 / 3.13), `rect_adapt` and `rect_supp` `result` on (ii) at all three and
on (i) at 8.8 and 11 only; every other arm `null` throughout.

**P2, LC11 RF-windowed drive median: `null` on both questions at every small rung in every arm.
`lc11_follows` is false 8/8.**

**Only `rectify` reads `carries_small_field = true`.** The two combination arms fail question (i)
for one reason: **rectification raises the arm's own blank/blank null** -- T3 null 0.00290 (base)
-> 0.00847 (rectify, 2.9x) -> 0.00837 (rect_adapt) -> **0.01759 (rect_supp, 6.1x)** -- so a
figure of the same absolute size no longer clears z >= 3 against it. The hooks skeptic predicted
exactly this on the liveness batch (rectify is 5.19x its OWN null, not 13x the shipped one).

**The statistic matters, and here is the row that decides the reading.** Beside P1 the design
carries `abs_drive_median`: the per-body RF-windowed median |drive| at T3 and T2. **Arm vs base,
all 8 arms x 2 types x 3 small rungs = 48 rows, every one `null`**, ratios 0.73-1.27 (rectify's
T3 0.97 / 1.27 / 1.19; rect_supp's 0.81 / 1.08 / 1.13; T2 0.84-1.00 across the three rectify arms,
0.77-1.00 over all eight). **Rectification multiplies the extremum over 1,940 T3 cells by 6-10x
and leaves the typical cell's windowed response where it was.** P1 is a within-run population
maximum and must be quoted as one. (The same statistic at Tm5Y and TmY21, outside the predeclared
P1 pair, is `null` in 44 of its 48 rows; the four exceptions are all Tm5Y at 4.5 deg and all
*below* base -- 0.57-0.86x -- so nothing there reverses the reading.)

**Shape.** No arm produces small-object selectivity. `rectify`'s T3 figure across the full ladder
is 0.0210 / 0.0371 / 0.0450 / 0.0554 / 0.0580 at 4.5 / 8.8 / 11 / 20 / 30 deg -- monotone
increasing; its gain over base peaks at 8.8-11 deg (10.5x, 9.0x) and falls to 5.1x at 30 deg,
which is the only size-dependence any arm introduces. On the old headline statistic LC11's excess
over null under `rectify` runs +0.013 / +0.054 / +0.099 / +0.307 / +0.531 mV with Spearman
**rho +0.937** (p_perm 5e-05), i.e. a large-object figure. Base's own is rho +0.635.
Large-rung rows (20 / 30 deg, exploratory) are `result` vs null for every arm at T3 and T2 and
`result` vs base only for the three rectify arms.

**LC10a** (exploratory) is `null` on both questions at 20 and 30 deg in every arm; the closest is
`suppress` at 30 deg (z 2.64, p_holm 0.111).

**Read the `fb0` rows as magnitudes.** Its vs-null verdict is `undetermined` against an SD-0 null
and is counted as a figure on the sign of `diff` alone, a strictly weaker bar than every other
arm's; its one `called: true` cell (T2 at 4.5 deg, vs-base z 11.65) rests on an arm with SD
exactly 0 and is not a mechanism claim.

## 5. Levels: the operating point the rectify arms move

`levels`, blank-arm means over the 30 sphere runs of each arm (the level-shift question the hooks
audit raised):

| arm | T3 dev_mean_b | T3 dev_abs_mean_b | T2 dev_mean_b | LC11 drive_mean_mv_b | LC11 rate_hz_mean_b | LC10a drive_mean_mv_b |
|---|---|---|---|---|---|---|
| base | +0.00002 | 0.0838 | +0.00436 | 0.0927 | 0.0027 | +0.0019 |
| fb0 | -0.00044 | 0.0826 | +0.00431 | 0.0636 | 0.0001 | -0.0013 |
| **rectify** | **+0.02838** | 0.0746 | **+0.03185** | **0.3477** | 0.0016 | **-0.1041** |
| adapt100 | -0.00055 | 0.0832 | +0.00482 | 0.0799 | 0.0010 | +0.0001 |
| adapt300 | -0.00061 | 0.0844 | +0.00476 | 0.0799 | 0.0019 | +0.0020 |
| suppress | -0.00007 | 0.0819 | +0.00511 | 0.1336 | 0.0017 | +0.0050 |
| **rect_adapt** | **+0.02596** | 0.0729 | **+0.02941** | **0.3063** | 0.0012 | **-0.1005** |
| **rect_supp** | **+0.02543** | 0.0754 | **+0.02974** | **0.3825** | 0.0039 | **-0.0796** |

The three rectify arms move T3's and T2's blank-arm operating point by **three orders of
magnitude** and LC11's blank-arm drive by **3.3-4.1x**, in the blank condition, where there is no
object. That is the level shift, and section 6's flicker row is its direct test. LC11
`rate_hz_mean_b` is at the counting floor throughout (143 cells x 12 s: base 4.7 spikes, rect_supp
6.6, adapt100 1.7 for the whole population) and carries no information -- it is quoted here only
so nobody reads it as one.

## 6. Specificity: the rescuing arm fails the Keles 2020 constraint

Predeclared: an arm RELEASES bar / grating / flicker at LC11 if, for that stimulus, the
arm-vs-base comparison reads `result` with diff > 0 on `diff_max_over_cells_mean_mv` or
`diff_rate_hz_max_cell` (4 v 4, unadjusted, an exploratory screen per stimulus).

**LC11 `diff_max_over_cells_mean_mv` (mV) / `diff_rate_hz_max_cell` (Hz), arm vs base:**

| arm | bar | grating | flicker | released |
|---|---|---|---|---|
| base | 0.182 / 0.083 | 0.611 / 0.042 | 0.145 / 4.708 | -- |
| fb0 | 0.166 / 0.167 | 0.669 / 0.167 | 0.274 / 4.833 | no |
| **rectify** | **0.705** (z 6.2) / 0.333 | **3.202** (z 98.8) / **2.917** (z 34.5) | **2.994** (z 66.1) / **10.667** (z 17.3) | **bar, grating, flicker** |
| adapt100 | 0.188 / 0.083 | 0.648 / 0.083 | 0.201 / 3.667 (z -3.0, lower) | no |
| adapt300 | 0.212 / 0.042 | 0.604 / 0.042 | 0.156 / 4.083 | no |
| **suppress** | 0.133 / 0.083 | 0.599 / 0.292 | **0.487** (z 7.9) / **5.750** (z 3.0) | **flicker** |
| **rect_adapt** | 0.352 (z 2.0, null) | **1.697** (z 41.4) / **0.667** (z 7.5) | **1.485** (z 31.1) / **5.833** (z 3.3) | **grating, flicker** |
| **rect_supp** | **0.634** (z 5.4) | **2.753** (z 81.7) / **3.250** (z 38.5) | **1.725** (z 36.7) / **8.458** (z 10.9) | **bar, grating, flicker** |

**The one arm that carries the carrier figure releases all three.** The flicker is spatially
uniform, so a figure there is the predeclared level-shift diagnostic -- and it is the largest
release of the three (20.7x base for `rectify`). `rect_adapt` and `rect_supp` ran on the **same
GPU model as base** in the spec battery (H200), so the release is not a device artifact.

**A finding about the base model, independent of any arm.** Against its own blank/blank null, base
LC11 reads: 11-deg dark square **0.081 vs 0.085, z -0.2**; 11-deg bright square 0.097, z +0.5;
bar 0.182, **z +4.0**; grating 0.611, **z +21.5**; flicker 0.145 (and 4.71 Hz at the best cell).
**The shipped model's LC11 responds to wide-field motion and to full-field flicker and does not
respond to a small dark square** -- the reverse of the animal's selectivity, before any hook.

### 6.1 ON and OFF transitions -- measured after the batch, on CPU, from the stored recordings

**The batch's own `flashon` / `flashoff` rows do not measure what their name says.** `flash()` is a
PERIODIC square (0.5 s on, 1.5 s period over 6 s = 4 cycles), so every run contains both
polarities, and the spec statistic is the whole-window time mean, which pools them. The
`flashon` / `flashoff` pair therefore separates a **bright** flash from a **dark** one; the
predeclared rule "T2 and T3 KEEP both transitions if the arm's mean `diff_signed_best_cell` on the
ON flashes and on the OFF flashes are each >= 0.5x the base's" reads `T2_T3_keep_on_and_off = true`
for all eight arms **on the bright-vs-dark split**, not on ON vs OFF. No `*_summary.json` under
`out/objr2c/spec/` carries a `per_type_transitions` key, and `spec_comparisons.stim` contains only
{bar, bright110, dark110, flashon, flashoff, flicker, grating}.

**The split was regenerated afterwards from the 2,048 stored spec recordings**, on CPU, without
touching the cluster: `probe_synthetic_stimuli.py analyse --dir out/objr2c/spec
--transition-window-s 0.3 --json out/interp/objr2c/spec_transitions.json` -> 46,144 `per_type`
rows, `transition` in {pooled, on, off}, a 0.3 s window after each ON edge and each OFF edge
truncated at the next edge, with the blank/blank arm scored on each family's own edge schedule as
the floor (`problems: none`). `diff_signed_best_cell`, mean over the 4 runs per arm:

| arm | bright flash T3 pooled / ON / OFF | bright flash T2 | dark flash T3 | dark flash T2 |
|---|---|---|---|---|
| base | 0.0045 / **0.0630 / 0.0568** | 0.0104 / 0.1069 / 0.0904 | 0.0065 / **0.0830 / 0.0639** | 0.0161 / 0.1217 / 0.1149 |
| fb0 | 0.0061 / 0.0534 / 0.0429 | 0.0111 / 0.0596 / 0.0715 | 0.0038 / 0.0784 / 0.0386 | 0.0124 / 0.0882 / 0.0978 |
| **rectify** | 0.0135 / 0.0601 / 0.0566 | 0.0143 / 0.0785 / 0.0778 | 0.0211 / **0.1418 / 0.0455** | 0.0274 / 0.0873 / **0.1329** |
| adapt100 | 0.0059 / 0.0499 / 0.0553 | 0.0114 / 0.0866 / 0.0888 | 0.0052 / 0.0491 / 0.0542 | 0.0130 / 0.1191 / 0.1221 |
| adapt300 | 0.0072 / 0.0535 / 0.0573 | 0.0109 / 0.0841 / 0.0942 | 0.0073 / 0.0610 / 0.0549 | 0.0173 / 0.1164 / 0.0977 |
| suppress | 0.0075 / 0.0536 / 0.0475 | 0.0114 / 0.0896 / 0.0625 | 0.0054 / 0.0711 / 0.0502 | 0.0120 / 0.0986 / 0.0866 |
| rect_adapt | 0.0078 / 0.0463 / 0.0454 | 0.0095 / 0.0759 / 0.0728 | 0.0147 / 0.0527 / 0.0417 | 0.0223 / 0.0810 / **0.1474** |
| **rect_supp** | 0.0249 / 0.0623 / 0.0636 | 0.0247 / 0.0856 / 0.1034 | 0.0219 / **0.1246 / 0.0582** | 0.0249 / 0.0718 / 0.1095 |

**What this says, as magnitudes** (4-run means of a max-over-cells statistic, no null-referenced
verdict, no Holm):

1. **The "keep both transitions" half of the Keles 2020 constraint holds for all eight arms.**
   Every ON window and every OFF window, in both flash families, at both T2 and T3, is at least
   **0.56x** the base's (minimum: fb0's bright-flash T2 ON, 0.56x; the next lowest are 0.59x). No
   arm abolishes a polarity.
2. **What rectification changes is the ON/OFF asymmetry, not the presence of either.** Base's two
   windows sit within 6-30 % of each other (T3 bright 0.063 / 0.057 = 1.11; T3 dark 0.083 / 0.064
   = 1.30; T2 dark 1.06). `rectify` tilts the **dark** flash's T3 toward ON by **3.1x**
   (0.142 / 0.045) and its T2 toward OFF by **1.5x** (0.087 / 0.133); `rect_supp` does the same at
   T3 (0.125 / 0.058 = 2.1x). Adaptation and spatial suppression leave the ratios within 0.9-1.4,
   i.e. near base's. `fb0`'s dark-flash T3 also tilts (2.0x), so part of that asymmetry is not
   specific to the hook.
3. **The honest caveat on the windows themselves.** Against the **matched blank/blank floor on the
   same edge schedule** -- the right comparator, and the reason the floor was computed -- almost
   every transition-window figure is at its own null: base reads 0.77-1.36x its floor, adaptation
   0.85-1.71x, suppression 0.68-1.14x, and rect_adapt 0.68-1.54x. **Only two cells clear their
   floor by more than 2x: `rectify` and `rect_supp` on the dark flash's ON window (2.36x and
   2.38x).** A 0.3 s window is 30 frames, and a max over 1,940 cells of an object-minus-blank
   difference over 30 frames has a high noise floor; the pooled whole-window numbers are an order
   of magnitude smaller than the windowed ones for exactly the complementary reason (the 6 s mean
   includes the between-flash frames, where there is nothing to see).

So the supported statement is: **both ON and OFF transitions survive in every arm; the split is
now measured rather than assumed; and the only transition-window figure that clearly exceeds its
own blank/blank floor is the rectify family's ON response to the dark flash.** The pooled
`flashon` / `flashoff` rows of the batch remain a bright-vs-dark comparison and are kept as such:

| arm | T3 bright flash / dark flash (ratio to base) | T2 bright / dark |
|---|---|---|
| base | 0.00450 / 0.00653 (1.00 / 1.00) | 0.01039 / 0.01614 (1.00 / 1.00) |
| fb0 | 0.00609 / 0.00382 (1.35 / 0.59) | 0.01106 / 0.01235 (1.07 / 0.77) |
| rectify | 0.01348 / 0.02109 (3.00 / 3.23) | 0.01425 / 0.02740 (1.37 / 1.70) |
| adapt100 | 0.00591 / 0.00518 (1.31 / 0.79) | 0.01142 / 0.01304 (1.10 / 0.81) |
| adapt300 | 0.00723 / 0.00729 (1.61 / 1.12) | 0.01094 / 0.01728 (1.05 / 1.07) |
| suppress | 0.00749 / 0.00536 (1.66 / 0.82) | 0.01136 / 0.01203 (1.09 / 0.75) |
| rect_adapt | 0.00778 / 0.01474 (1.73 / 2.26) | 0.00953 / 0.02225 (0.92 / 1.38) |
| rect_supp | 0.02485 / 0.02185 (5.52 / 3.35) | 0.02469 / 0.02493 (2.38 / 1.54) |

Bright vs dark 11-deg squares (a magnitude, no call): LC11 bright/dark 1.19 (base), 1.54 (fb0),
0.74 (rectify), 1.18 / 0.75 (adapt100 / 300), 1.65 (suppress), 1.14 (rect_adapt), 0.69
(rect_supp), against 4-run SDs of 0.012-0.090. Nothing separates.

## 7. Transfer: what each arm costs the benchmark

`benchmark.py --sections motion,loom_escape`, 3 draws per arm (every arm-vs-base comparison is
`underpowered` by construction, `p_floor(3,3) = 0.10`; the PASS / FAIL below are the benchmark's
own checks). An arm COSTS a section if any draw fails a check base passes in every draw.

| arm | motion.min_dsi | motion.correct_directions | loom_escape.GF_peak_hz | loom_escape.escapes | costs |
|---|---|---|---|---|---|
| base | 0.2427 +- 0.0028 PPP | 8 PPP | 50.03 +- 3.12 PPP | 1.00 PPP | -- |
| fb0 | 0.2591 PPP | 8 PPP | 54.11 +- 8.58 PPP | 1.00 PPP | none |
| rectify | 0.2241 PPP | 8 PPP | 49.64 +- 8.48 PPP | 1.00 PPP | none |
| adapt100 | 0.2270 PPP | 8 PPP | 45.20 +- 6.89 PPP | 1.00 PPP | none |
| adapt300 | 0.2383 PPP | 8 PPP | 52.81 +- 3.79 PPP | 1.00 PPP | none |
| **suppress** | 0.2810 PPP | 8 PPP | **31.24 +- 8.89 F F P** | **0.33 +- 0.58 F F P** | **GF_peak_hz, escapes** |
| rect_adapt | 0.2228 PPP | 8 PPP | 48.65 +- 1.10 PPP | 1.00 PPP | none |
| **rect_supp** | 0.2739 PPP | 8 PPP | **32.02 +- 2.14 P F F** | **0.33 +- 0.58 P F F** | **GF_peak_hz, escapes** |

**Spatial suppression costs the escape section**: GF peak -37 %, escapes 1.00 -> 0.33, in the two
arms that carry it and in no other. `rect_supp` ran on the same GPU model as base in the bench
section, so the cost is not a device artifact. No `errors` in any arm (the CUDA-graph / Torch-optic
combination the plan flagged as untested did not fail). `motion.min_dsi` moves -0.020 (rect_adapt)
to +0.038 (suppress) and passes everywhere.

## 8. Reading, classification, and what this batch cannot say

**The answer.** No mechanism class makes the small-field stage carry a small-object figure with
specificity preserved. Rectification carries a figure -- large (T3 6.0 / 10.5 / 9.0x base at the
small rungs, T2 3.1-3.7x), reproducible across three arms and two GPU models -- and it is the
wrong figure: it grows with object size, it lives in the extremum over cells rather than in the
typical cell's windowed response, it rides on a 3-4x operating-point shift that a uniform field
reproduces, and it releases bar, grating and flicker responses at LC11 that the Keles 2020
perturbation says must not appear. Adaptation at 100 and 300 ms changes nothing on this assay.
Spatial suppression changes nothing on this assay and costs the escape benchmark. **LC11 output
follows in no arm. NOTHING IS ADOPTED; every hook remains `None` by default.**

**Classification under the project rule.** *Physiology-implied:* that T2 and T3 respond to both ON
and OFF transitions and that T3 -> LC11 is functionally excitatory (Keles et al. 2020) -- this
motivates testing a per-stream split at all; that LC11 pools tightly size-tuned, fast-adapting
inputs (Tanaka & Clark 2020) -- this motivates testing adaptation and surround suppression as a
class. *Hand-set numbers, every one:* the stream memberships ({Mi1, Tm3, Tm2} vs {Tm1, Tm4} at T3;
{Tm2, L5, Tm3, Mi1} vs {C3} at T2), which were chosen by **measured figure sign** in
`deficit_object.md` 2 and not by ON/OFF pathway identity; the modes `pos` / `neg`; tau 100 and
300 ms; gain 1; k 0.5; radius 10 deg. And the `neg` mode's own premise -- it preserves the sign of
W but **inverts the sign of the signal**, so an excitatory carrier below its operating point
excites -- is a stand-in for an unmodelled OFF pathway, not a per-synapse mechanism. No number
here was fitted to any of these results and none should be.

**What this batch cannot say.**
1. **Arm was confounded with cluster box.** One job per arm, scheduled least-loaded-first:
   `base` and `fb0`, `adapt100`, `adapt300`, `rect_adapt`, `rect_supp` on B200s; **`rectify` and
   `suppress` on H200s**. Base itself sat on a B200 for the sphere and an H200 for the specificity
   and bench sections. Every `rectify` and `suppress` vs-base row on the sphere is device-crossed.
   The rectification magnitude survives because `rect_adapt` and `rect_supp` carry the same hook on
   base's own GPU model and give ratios 2.2-10.3x; the specificity release and the suppression
   bench cost each replicate on a same-device pair. **A 3-job re-run of `base` on the box that
   hosted `rectify` and `suppress` (5 object runs + 5 nulls, one submission) is owed before any
   vs-base row is quoted without the caveat.**
2. **The ON/OFF transition question was unanswerable from the batch's own outputs and was answered
   afterwards** by re-deriving the split on CPU from the stored recordings (section 6.1). What that
   re-analysis gives is a set of 4-run magnitudes with no null-referenced verdict; the batch itself
   still contains no ON/OFF comparison row.
3. **LC11 windows are anatomical boxes, all 143 of them.** No per-body LC11 statement in this file
   rests on a measured receptive field.
4. **The three members of each primary family are not independent**: each arm has five blank/blank
   runs and the same five are the reference at all three rungs. Holm is valid under arbitrary
   dependence, so the adjustment is sound, but these are not three independent tests.
5. **The specificity screen is 4 v 4** (`p_floor` 0.029), read per stimulus, unadjusted, and is
   exploratory by predeclaration; it cannot survive Holm across seven stimuli.
6. **Effective retinal contrast is not matched across the sphere ladder** (inherited from the
   baseline: extreme relative change -0.50 at 4.5 deg to -0.95 at 30 deg), so the size axis of
   every arm's ladder still carries a contrast gradient.

## 9. Verification

**The compare skeptic** (`verify:compare`, Opus, verdict **mostly sound**) read the design and the
queued batch *before* any data landed; its verdict is recorded verbatim in
`docs/audits/receptor_verification.md` ("Object round 2"). Its two refutations and its leading
correction all landed, and the results above are written around them:

* The **"isolated ON / OFF transitions"** label was refuted from the stimulus generator: `flash()`
  is a periodic square and the spec statistic is a whole-window mean, so the battery separated
  bright from dark and pooled both polarities. Section 6.1 is the consequence -- the split has now
  been re-derived on CPU, and the batch's own `T2_T3_keep_on_and_off = true` is reported as a
  bright-vs-dark statement.
* The claim that **this audit's sections 0-2 were the predeclaration** was refuted on file times:
  `out/objr2c/predeclared.json` 22:04:04Z and `submit_console.txt` 22:06:37Z, but this document's
  mtime was 22:07:27Z, *after* submission. The JSON is the stamp; the skeptic verified that the
  live `PREDECLARED` / `ARMS` dicts are byte-identical to it, so no rule was rewritten -- but this
  file is not the predeclaration and is not cited as one.
* **"Arm is fully confounded with cluster box"**, called before the data existed, is exactly what
  the realised `device_name` column shows (section 8, item 1). The skeptic's prescription -- read
  question (i) as unaffected, and either re-run base on each box or restrict vs-base calls to
  same-device pairs -- is what section 8 records as owed.
* The skeptic also predicted that **P1 is a max over cells** and asked for the per-body companion
  in the same table. Section 4's `abs_drive_median` row is that companion, and it is `null` in all
  48 arm x type x rung rows.

**The round-2 critic** (after the batch landed) recomputed every number in sections 3-8 from
`out/interp/objr2c/compare.json` and the raw consoles. Its findings, beyond confirming the tables:
the **`abs_drive_median` row is the finding that decides the reading** (6-10x on the extremum,
0.73-1.27x and `null` on the typical cell); the **combination arms rescue the device confound**
by carrying the same hook on base's own GPU model; `base`'s own LC11 **already fails the Keles
2020 specificity picture** before any arm (section 6); and the rectify family's figure is
**size-monotone increasing** (Spearman rho +0.937), i.e. the opposite shape from LC11's biological
target. Two numbers in the critic's draft were corrected against the files when this section was
written: the submission time is **22:06:37Z** (not 15:06:37Z -- the file times are local -0700),
and the `levels` rows are means over **30** sphere runs per arm (`levels.n_runs`), not 25.

**The `spec_transitions` re-analysis** (section 6.1) post-dates both. It is
`out/interp/objr2c/spec_transitions.json`, 46,144 `per_type` rows, generated on CPU from the
recordings already on disk; nothing in it required the cluster, and none of its numbers changes an
answer in sections 4, 5, 7 or 8.
