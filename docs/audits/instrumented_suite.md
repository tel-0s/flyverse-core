# The 29-check suite under `instrumented` with the three-instrument list (round 8, item 3; batches `suite-inst`, `suite-inst-room`)

Question (TODO.md section B, follow-up 5 of the Session-13 skeptics; PRESETS_SPEC section 2 item 5): the suite under
`instrumented` with round 7's list -- `sided_turn_afferent:k=0.5`, `ring_dc_hold`, `glno_sign` -- at three draws
beside `raw` at three, the way [compass_standin.md](compass_standin.md) ran it, and then the room rate-half at >= 6
runs per arm only if the suite half passes. The compass stand-in's own suite run was rejected on `taste.MN9_hz`, so
the expectation was another rejection; the point was to have the column on the record. Nothing is adopted either
way: this records whether the list is admissible.

## 1. What runs (code)

`scripts/benchmark.py` gained two opt-in flags, refused under `raw`: `--hold-edges PRE:POST` appends the hold at
factor 0 to `LIFParams.type_path_gain` in every brain the suite builds (the legacy `Brain` probes, the demo `Sim`s
through the patched factories, the shiu row) and attaches the `edges` record; `--nt-override TYPE=nt` compiles the
relabel into `cx_wedge`'s scratch cache and attaches the `relabel` record; `cx_wedge.build_instruments` supplies fresh
objects per brain, and the JSON `config` records `hold_edges`, `nt_override` and `instrument_records`. Without the
flags the instrument list is the string list it was. One pre-existing adapter gap was found by the first submission and
fixed: `InstrumentedBenchmarkBrain.drive` wrote `fb._extensions.base_drive`, and the extension scheduler exists only
when a module is attached -- a sense-side transducer plus two configuration records attach none (test on the small
graph). The afferent is live wherever a body feeds `yaw_rate` (the `room_demo.Sim` sections) and silent, body-less,
in the legacy probes, as `compass` was in compass_standin's run; the hold and the relabel act in every section.

The room rate-half runs `scripts/batch_sustain.py` as it is (its own argv) under `scripts/instrumented_room.py`,
which rebinds batch_sustain's `BatchSim` to a subclass supplying the relabel cache, `preset="instrumented"` and fresh
records, and `flyverse.brain.LIFParams` to a subclass carrying the hold (the `ring_dc_hold` record verifies it at
attach); the JSON gets a `room` block (resolved preset, records, hold, GLNO transmitter, cache, device; problems ->
exit 3) and `provenance`. Both arms smoked on CPU.

## 2. The suite half (`out/suite-inst`)

Protocol: `benchmark.py --sections all --draw-seed S --seeds S --preset raw` and `... --preset instrumented
--instruments sided_turn_afferent:k=0.5 --hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)' --nt-override GLNO=glutamate`,
S = 0, 1, 2, one job each, six jobs in one `cluster_run.py` call, each pinned to one GPU of the pool 4-7 of the second
node, `--ship flyverse,scripts` ([determinism_gate.md](determinism_gate.md) section 2). The shipped native backend,
as compass_standin ran it. Frozen rule (`out/suite-inst/predeclared.json`): a status change is an instrumented
draw's status on a row that is not the status of every raw draw (`PASS (gap closed)` and `PASS` one status); rows
whose raw draws disagree are `unstable`, reported, not counted; a status change on any row outside the declared gap
rows -- `compass.wedge_cells_persisting` -- is a rejection in either direction; the 3 v 3 value comparison is
descriptive only (underpowered by construction, and the native path does not repeat).

Two stamps: 2026-09-18T00:00:58Z, submitted as `suite-inst-a260b3` -- the raw draws valid, every instrumented draw
losing eight checks (walk x 3, loom x 2, rotate, motion x 2) to the adapter gap above, so 19/0/2/8, 18/1/2/8, 19/0/2/8
against raw 27/0/2/0, 26/1/2/0, 27/0/2/0 (kept, ignored, under `out/suite-inst_attempt1/`, unused); then
2026-09-18T00:26:46Z with the adapter fixed and the analysis rejecting any run whose console carries a traceback, the
plan and the rule byte-identical, submitted as `suite-inst-4b511a`.

Result (`out/suite-inst/analysis/`, 0 problems): six runs, `NVIDIA B200`, `device cuda` in 6/6 consoles, no traceback,
29 checks each; the raw runs on the shipped cache (`ef23cc27...`, 13 controller records each), the instrumented runs
on the GLNO=glutamate cache (`7a10d93b...`, 19 controller records each, every one carrying the three instruments,
the hold in `config.type_path_gain`).

| run | pass / fail / gap / missing |
|---|---|
| raw s0 / s1 / s2 | 27 / 0 / 2 / 0; 26 / 1 / 2 / 0; 27 / 0 / 2 / 0 |
| instrumented s0 / s1 / s2 | 27 / 0 / 2 / 0; 26 / 1 / 2 / 0; 27 / 0 / 2 / 0 |

**No row changes status in any draw.** The only row whose statuses are not uniform is `taste.MN9_hz` (FAIL in draw 1
under both presets, 1.69046 Hz against `> 2`): raw's own instability, identical under the instruments, reported and not
counted. The compass row is KNOWN GAP (0 persisting cells) in all six runs. The raw tallies and values reproduce
compass_standin's raw column exactly on eight rows (rest, taste, smell, dn, walk.GF_max, bitter.calibrated -- the B=1
legacy probes; these repeat like cx_wedge does) and differ on the room-derived rows (walk.power, loom, rotate,
walk_gf, loom_escape), as determinism_gate.md says they must.

Per row, values by draw seed [0, 1, 2] and statuses (`out/suite-inst/analysis/suite_rows.csv`, columns `value`,
`status`; pasted, INTERP 10.4 rule 28; P PASS, F FAIL, G KNOWN GAP):

| check (criterion) | raw [0,1,2] | instrumented [0,1,2] | statuses raw / inst |
|---|---|---|---|
| rest.spikes_per_step (< 5) | 0 P, 0 P, 0 P | 0 P, 0 P, 0 P | PPP / PPP |
| taste.MN9_hz (> 2) | 10.9342 P, 1.69046 F, 3.12959 P | 10.9342 P, 1.69046 F, 3.12959 P | PFP / PFP (raw unstable) |
| smell.PN_hz (< 100) | 7.86116, 11.1984, 11.6049 | 7.86116, 11.1984, 11.6049 | PPP / PPP |
| smell.KC_active (> 0) | 816, 1345, 1518 | 816, 1345, 1518 | PPP / PPP |
| dn.DNa02_L_leg_asym_hz (> 0.3) | 2.58063, 2.40017, 2.38502 | 2.58063, 2.40017, 2.38502 | PPP / PPP |
| dn.MDN_top_hz (< 250) | 153, 172, 182 | 153, 172, 182 | PPP / PPP |
| dn.DNp09_top_hz (< 250) | 152, 157, 122 | 152, 157, 122 | PPP / PPP |
| walk.GF_max_hz (< 38) | 4.62916, 4.96265, 4.60406 | 4.62916, 4.96265, 4.60406 | PPP / PPP |
| walk.power_max_hz (notnone) | 48.4805, 53.5587, 56.1464 | 48.4805, 53.5587, 55.9379 | PPP / PPP |
| walk.power_sustained_hz (< 50) | 20.1091, 27.893, 25.5384 | 20.1091, 27.893, 25.3919 | PPP / PPP |
| loom.GF_peak_hz (>= 20) | 47.2279, 51.3004, 52.8799 | 47.2162, 51.4427, 45.4985 | PPP / PPP |
| loom.escape_cm (notnone) | 3.5, 3.5, 3.5 | 3.5, 3.5, 3.5 | PPP / PPP |
| rotate.DNp20_flip_hz (< -2) | -39.5479, -36.2069, -25.1727 | -39.3318, -36.1501, -38.9102 | PPP / PPP |
| motion.min_dsi (>= 0.1) | 0.241096, 0.24595, 0.241096 | 0.240637, 0.248435, 0.240637 | PPP / PPP |
| motion.correct_directions (== 8) | 8, 8, 8 | 8, 8, 8 | PPP / PPP |
| loom_escape.GF_peak_hz (>= 33) | 48.0924, 55.2507, 48.8023 | 49.8767, 42.0818, 51.3857 | PPP / PPP |
| loom_escape.escapes (>= 1) | 1, 1, 1 | 1, 1, 1 | PPP / PPP |
| walk_gf.p99_hz (< 38) | 20.099, 22.5368, 13.546 | 20.2437, 16.3014, 25.8704 | PPP / PPP |
| rotation.group_flip_hz (<= -3) | -9.48312, -10.4283, -10.9066 | -9.25218, -9.80203, -9.68861 | PPP / PPP |
| object.LC10a_flip_hz (abs >= 1.0) | 0.000725196, -0.00979827, 0.0091777 | -0.00188607, 0.0209366, 0.00974457 | GGG / GGG |
| bitter.calibrated_sugar_MN9_hz (> 2) | 5.51844, 4.28767, 3.92663 | 5.51844, 4.28767, 3.92663 | PPP / PPP |
| bitter.calibrated_sugar_bitter_MN9_hz (< 1) | 0, 0, 0 | 0, 0, 0 | PPP / PPP |
| bitter.shiu_sugar_MN9_hz (> 50) | 139.898, 138.934, 131.523 | 140.26, 149.314, 132.249 | PPP / PPP |
| bitter.shiu_sugar_bitter_MN9_hz (< 10) | 0.81784, 0, 0 | 1.37552, 0, 0 | PPP / PPP |
| wind.DNp18_flip_hz (>= 15) | 44.7584, 46.9604, 46.862 | 45.839, 45.8612, 46.8457 | PPP / PPP |
| wind.DNp33_flip_hz (<= -15) | -50.2723, -50.6726, -50.3201 | -49.1691, -49.2368, -50.0192 | PPP / PPP |
| odour.apple_channel_8cm_hz (>= 10) | 17.4191, 17.3895, 17.4635 | 17.4431, 17.443, 17.4208 | PPP / PPP |
| odour.apple_channel_clean_hz (<= 6) | 4.40823, 4.39386, 4.68342 | 4.39271, 4.25439, 4.48907 | PPP / PPP |
| compass.wedge_cells_persisting (>= 6) | 0, 0, 0 | 0, 0, 0 | GGG / GGG (declared gap row) |

Suite verdict: **admissible by the suite half** (no status change outside the declared gap rows); the room rate-half
is owed and was run.

## 3. The room rate-half (`out/suite-inst-room`; frozen 2026-09-18T00:45:27Z, submitted as `suite-inst-room-dab10c`)

Protocol: guard_suites.sh's room take-off protocol, `batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx
--fruit apple --fence` (live escape route, native backend), brain seed k with env seeds 16k-16k+15, k = 0-5: six
seed-matched runs per arm, 16 rooms x 300 s = 4,800 fly-s per run, 12 jobs in one call. Frozen rule (INTERP 10.4 rule
17): the primary is total take-offs (escape + voluntary) per run at equal exposure, a two-sample exact Poisson
(conditional binomial on the pooled counts), ONE-SIDED, direction stated once -- the list is worse iff its take-off
rate is higher (H1 instrumented > raw), fail at p < 0.05 -- with the run-level `common.compare` (6 v 6) quoted beside
it; escape and voluntary separately are descriptive.

Result (`out/suite-inst-room/analysis/`, 0 problems): 12 runs, `device cuda` in 12/12 consoles, `NVIDIA B200`,
GPUs 4-7 three jobs each, the raw runs `preset raw` on the shipped cache with no record, the instrumented runs with
the three records, the hold in `type_path_gain` and GLNO glutamate on the relabel cache; 16 rows x 30,000 frames each;
wall 2,251-2,400 s.

| measure | role | K instrumented | K raw | per 1,000 fly-s inst / raw | one-sided exact p (inst higher) | two-sided | run-level compare (6 v 6) | instrumented runs | raw runs |
|---|---|---|---|---|---|---|---|---|---|
| take-offs (hops) | primary | 110 | 101 | 3.819 / 3.507 | **0.291** | 0.582 | null (diff +1.50, z +0.44, p 0.937) | 18, 16, 12, 17, 17, 30 | 13, 18, 17, 22, 18, 13 |
| escape | descriptive | 42 | 39 | 1.458 / 1.354 | 0.412 | 0.824 | null (diff +0.50, z +0.21, p 0.818) | 5, 7, 5, 7, 7, 11 | 4, 8, 6, 10, 7, 4 |
| voluntary | descriptive | 68 | 62 | 2.361 / 2.153 | 0.331 | 0.661 | null (diff +1.00, z +0.83, p 1.000) | 13, 9, 7, 10, 10, 19 | 9, 10, 11, 12, 11, 9 |

**Rate-half verdict: PASS -- the instrumented take-off rate is not higher (one-sided exact p 0.291).** Undeclared
descriptives from the same runs (`room_runs.csv`, run lists in seed order; a 6 v 6 `compare` each, none a verdict):
meals instrumented 1, 1, 2, 2, 4, 4 against raw 4, 7, 1, 6, 7, 0 (diff -1.83, z -0.60, p 0.394, null); path m 3.705,
3.695, 3.663, 3.708, 3.635, 3.696 against 3.602, 3.610, 3.703, 3.678, 3.617, 3.693 (null); walking-GF median Hz 30.15,
30.80, 31.07, 31.51, 30.80, 33.37 against 31.03, 32.64, 31.04, 32.12, 31.36, 31.48 (null); rows at the GF threshold 3,
6, 3, 5, 6, 9 against 4, 7, 5, 7, 7, 4 (null). The meals difference is the one a reader should look at: it is not part of
the rule, it is null at 6 v 6, and it is the kind of thing the rule's "rows the instrument is declared to touch" does
not cover (the hold and the relabel change the ring's operating state in a room the compass program `cx` steers).

## 4. Answer

The three-instrument list is admissible under PRESETS_SPEC section 2 item 5: the 29-check suite changes no row's
status in three instrumented draws beside three raw draws (27/0/2, 26/1/2, 27/0/2 under both presets; the one non-
uniform row, taste.MN9_hz, is raw's own instability and identical under the instruments; the compass row stays KNOWN
GAP), and the room rate-half at six seed-matched runs per arm passes: 110 take-offs against 101 over 28,800 fly-s each
(3.82 vs 3.51 per 1,000 fly-s; one-sided exact p 0.291; run-level null). The expected rejection did not come: this
list, unlike the compass stand-in, moves no suite row. Admissible is all it is -- nothing is adopted, `raw` stays the
default, the afferent's law is still `unverified` and the relabel still has no transmitter source; and a fewer-meals
descriptive (1-4 against 0-7 per run, null at 6 v 6) is on the record for the next reader. The first submission's
instrumented draws lost eight checks to a benchmark adapter gap (fixed, tested, re-frozen, re-run).

## 5. Reproduction

```sh
PYTHONIOENCODING=utf-8 python scripts/instrumented_suite.py analyse --runs out/suite-inst --out out/suite-inst/analysis
PYTHONIOENCODING=utf-8 python scripts/instrumented_suite.py analyse-room --runs out/suite-inst-room --out out/suite-inst-room/analysis
```

Committed: `out/suite-inst/{batch.sh,jobs.json,predeclared.json}`, `out/suite-inst/analysis/{runs,suite_rows,suite_table}.csv`,
`analysis.md`, `summary.json`; `out/suite-inst-room/{batch.sh,jobs.json,predeclared.json}`,
`out/suite-inst-room/analysis/{room_runs,room_tests}.csv`, `room_analysis.md`, `room_summary.json`. Run JSONs and
consoles stay ignored (host paths); attempt 1 under `out/suite-inst_attempt1/`, ignored.

## 6. Author self-review (the independent skeptic pass is not this section)

- The suite half's rule is the status rule the round inherited; it has no power against a value that moves inside a
  PASS band (loom.GF_peak_hz 52.9 -> 45.5 in draw 2; walk_gf.p99 22.5 -> 16.3 / 13.5 -> 25.9), and these rows do not
  repeat run to run under `raw` either (determinism_gate.md), so a 3 v 3 value test would be underpowered by
  construction. The suite is a status gate and is reported as one.
- The afferent gets no yaw in the legacy probes; the suite therefore tests the hold and the relabel everywhere and
  the afferent only in the room sections. compass_standin's `compass` had the same scope; a suite that drives the
  afferent in the legacy probes would need a body it does not have.
- The rate-half's direction is one-sided by the frozen rule (more take-offs = worse); a two-sided reading gives
  p 0.58 and changes nothing. Six seed-matched runs per arm is the rule-17 replication.
- The meals descriptive was not declared; it is reported with its run lists and its null and is not turned into a
  finding after the fact.
- Two submissions for the suite half: the first's instrumented draws were invalid for a reason that is now a test;
  the raw draws of both submissions carry the same B=1 legacy values (they repeat) and different room values (they do
  not); the second submission is the one analysed, whole.
- "Admissible" means the PRESETS_SPEC gate; it does not mean the afferent's gain, the hold or the relabel are right.

## Report

```yaml
summary: |-
  The three-instrument list is admissible under PRESETS_SPEC section 2 item 5: the 29-check suite changes no row's
  status in three instrumented draws beside three raw draws (27/0/2, 26/1/2, 27/0/2 under both presets; the one non-
  uniform row, taste.MN9_hz, is raw's own instability and identical under the instruments; the compass row stays KNOWN
  GAP), and the room rate-half at six seed-matched runs per arm passes: 110 take-offs against 101 over 28,800 fly-s each
  (3.82 vs 3.51 per 1,000 fly-s; one-sided exact p 0.291; run-level null). The expected rejection did not come: this
  list, unlike the compass stand-in, moves no suite row. Admissible is all it is -- nothing is adopted, `raw` stays the
  default, the afferent's law is still `unverified` and the relabel still has no transmitter source; and a fewer-meals
  descriptive (1-4 against 0-7 per run, null at 6 v 6) is on the record for the next reader. The first submission's
  instrumented draws lost eight checks to a benchmark adapter gap (fixed, tested, re-frozen, re-run).
key_claims:
- Suite half: no status change on any of 29 rows in 3 + 3 draws; tallies identical under both presets.
- Room rate-half: 110 vs 101 take-offs at equal exposure, one-sided exact p 0.291, run-level null; escape 42 vs 39, voluntary 68 vs 62.
- The list is admissible by the PRESETS_SPEC gate; nothing adopted.
- Meals per run 1-4 (instrumented) vs 0-7 (raw), undeclared, null at 6 v 6, on the record.
validation:
- suite-inst-4b511a: 6 runs, 6/6 consoles device cuda, no traceback, 29 checks each, caches ef23cc27 (raw) / 7a10d93b (instrumented), the three records in every instrumented controller, 0 problems.
- suite-inst-room-dab10c: 12 runs, 12/12 device cuda, room blocks with no problems, 0 analysis problems.
- Frozen before submission: out/suite-inst/predeclared.json (00:00:58Z, re-stamped 00:26:46Z after the adapter fix, plan and rule unchanged), out/suite-inst-room/predeclared.json (00:45:27Z).
- benchmark.py --hold-edges / --nt-override are refused under raw; without the flags nothing changes; InstrumentedBenchmarkBrain's setter guard is tested.
recommendations:
- Nothing adopted. If the list is ever proposed for adoption, the meals descriptive needs a declared arm pair first.
open_questions:
- Whether the fewer-meals descriptive is a real effect of the hold plus relabel on the cx program's steering.
```
