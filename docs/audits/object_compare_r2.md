# The fixed-anatomy model comparison, round 2 (`scripts/object_round2_compare.py`) -- Neurome intake, experiments 2 + 3

**Status: SUBMITTED, RESULTS PENDING -- sections 3-8 are written when the batch is analysed; sections 0-2 were written
and stamped before submission.** Nothing is adopted; every hook stays off by default (the project rule); the arms are
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

## 3-8. Results

_Pending the batch (see the status line at the top)._
