# The deterministic-kernel gate (round 8, item 1; batch `det1`)

Question (TODO.md section B, follow-up 4 of the Session-13 skeptics): which execution paths repeat exactly on one GPU?
The native event-driven CUDA path gave per-cell rate differences to 44.9 Hz under matched inputs
([navigation_instruments.md](navigation_instruments.md) section 4), while the cx_wedge protocol reproduced across two
submissions (cx8 == cx8r, [compass_velocity_route.md](compass_velocity_route.md) 5.6). Until one of an exact-workload
pass, a deterministic path or repeated draws with the spread exists, every room number is one draw. This audit runs
each protocol twice on one pinned GPU and compares every saved array and metric exactly. Nothing is adopted; no
default changes; the gate decides only which path later items run on and how their numbers may be quoted.

## 1. Protocols and paths

Every pair is one scheduler job that runs its command twice, sequentially, on one GPU of the pool 4-7 of the second
cluster node (`cluster_run.py --gpu-ids 4,5,6,7 --node <cluster-node-2>`: `gpus 1`, `gpu_ids [id]`, `vram_gb 24`).

| pair | protocol | path (FlyBrain flags) | B | world / optic |
|---|---|---|---|---|
| cxS | `scripts/cx_wedge.py` round-7 S arm, seed 0 (raw, no instrument, prescribed turn 90 deg/s over 0.5-3.5 s, `--ledger`) | torch: `cuda_kernels` off (constructor default, `FLYVERSE_CUDA_KERNELS` unset), `event_driven` False, `cuda_sparse torch`, `cuda_graphs` on | 1 | none |
| plume_native | the plume_steering_probe room: BatchSim, env seeds 0-5, `compass plume hunger flight`, headings 5/90/-90/185/5/5 deg, energy 0.1, all fruit, no fence, program none, brain seed 0, 10 s | native: `cuda_kernels`, `event_driven`, `cuda_graphs`; `cuda_sparse torch` (warp is B=1 only) | 6 | yes |
| plume_torch | the same room | torch: `cuda_kernels` False, `event_driven` False, `cuda_sparse torch`, `cuda_graphs` on | 6 | yes |
| raw_native | the plain room: BatchSim, preset raw, no instruments, program none, all fruit, default start and heading, brain seed 0, 10 s | native | 6 | yes |
| raw_torch | the same room | torch | 6 | yes |

Recorded per room run: the body trace per frame (x, y, z, heading, yaw rate, energy, feeding, tasting, nearest-fruit
distance, antenna L/R, motor turn_L/turn_R/power), the cumulative spike count per row per frame, the plume state per
frame (turn, steer input, goal, strength, integral, DNa02 L/R) where attached, and the final brain tensors
(`FlyBrain.BRAIN_TENSORS`); per cx_wedge run its ledger NPZ (31 arrays) and JSON row (546 numeric leaves).

## 2. Predeclaration

`scripts/determinism_gate.py plan / predeclare`; the frozen file is `out/det1/predeclared.json`. Three stamps, all
before the valid submission, and the history is the record:

| stamp (UTC) | what | outcome |
|---|---|---|
| 2026-09-17T23:14:57Z | first freeze | never submitted: the job line held an apostrophe inside `"${CLUSTER_NODE_2:?...}"`, which bash rejects inside a double-quoted parameter expansion; the plan was regenerated with the message reworded and `bash -n` checked |
| 2026-09-17T23:17:01Z | second freeze, submitted as run `det1-5b7a42` | invalid: every fetched run's `provenance.source_fingerprint` showed four files loaded from the box's own checkout that differ from `origin/main` (`flyverse/compass.py`, `fly.py`, `instruments.py`, `navigation.py`) (and, for the room runs, the pre-re-stamp `scripts/determinism_gate.py`); `cluster_run.py` ships only the files that differ from `origin/main` here, so a target checkout behind `origin/main` runs its stale copy of everything else. The stale `PlumeNavigation` has no DNa02 reads and both plume pairs crashed (`KeyError: 'dna_L'`) while the scheduler listed all five jobs as `completed`, `exit_code None`. The three landed pairs are kept, ignored, under `out/det1_attempt1/` and are not used |
| 2026-09-17T23:31:08Z | third freeze, submitted as run `det1-a4facc` at ~23:32Z with `--ship flyverse` (78 package files copied over the run copy) | valid: 5 jobs, 10 consoles `device cuda`, 0 analysis problems |

The plan and the comparison rule are byte-identical across the three stamps; the third re-stamp exists because the
analysis file's hash changed (a loaded-source check was added -- the check that would have caught the second
submission); `runs.csv` was added later, with the analysis, so the predeclared `scripts/determinism_gate.py` hash is
the hash of the code the RUNS loaded, not of the analysis that produced the CSVs. Disclosed: the author had seen
attempt 1's raw pairs (both one draw, first differing frames 24 and 125) before the third stamp; nothing in the plan
or rule was changed on that account.

**Withdrawn:** "the third re-stamp exists because the analysis file's hash changed (a loaded-source check was added
-- the check that would have caught the second submission -- and a per-run metrics file)" as to the metrics file --
`git diff 2d70edb^..2d70edb` adds only the loaded-source check; the per-run `runs.csv` came later, with the post-run
analysis commit.

The frozen comparison rule (`predeclared.json` -> `comparison_rule`): every array of the run's NPZ compared with
`numpy.array_equal(equal_nan=True)` after a shape check, max |a-b| per array; every numeric leaf of the run's JSON
compared with `==` (NaN == NaN), the bookkeeping keys `wall_s`, `host`, `platform`, timestamps and file paths
excluded; a pair `repeats exactly` iff every compared array and metric is equal, otherwise `one draw` -- no tolerance,
no partial credit. Backend check: every run's `provenance.execution.backend` must carry its path's flags and its device
must be CUDA; a mismatch invalidates the pair. Decision rule for items 3-4: torch repeats and native does not -> torch
path; both -> the shipped native path stays; neither -> >= 6 draws, the run is the replicate unit, nothing quoted beyond
its across-draw SD.

Two tool changes were needed and are in this branch: `cluster_run.py --gpu-ids` (the pool; step 0) and
`cluster_run.py --ship PATH[,PATH]` (copy named tracked paths whether or not they differ). The scheduler's job list
reports `exit_code None` for every job and classes a crashed job `completed`, so its "n completed, 0 failed" is not a
receipt of success for this round (INTERP 10.4 rule 4): the per-run validity is the console's `device cuda`, the
presence of the outputs and the analysis's zero-problem check. The scheduler's receipt was not fetched as a file
into `out/det1/` this round (INTERP 10.4 rule 21); only the client console records the submission.

## 3. Results (`out/det1/analysis/`, generated by `determinism_gate.py analyse`; 0 problems)

| pair | protocol | path | verdict | arrays equal / n | max abs diff (arrays) | metrics equal / n | max abs diff (metrics) | first differing frame (cumulative spikes) | GPU |
|---|---|---|---|---|---|---|---|---|---|
| cxS | cx_wedge S arm seed 0 | torch (cx_wedge default) | **repeats exactly** | 31 / 31 | 0 | 546 / 546 | 0 | -- | B200, id 4 (from the client log; not recorded in the run) |
| plume_native | plume room 10 s B=6 seed 0 | native | **one draw** | 1 / 15 | 37083 | 992 / 1026 | 32271 | 30 | B200, id 5 |
| plume_torch | plume room 10 s B=6 seed 0 | torch | **one draw** | 1 / 15 | 41225 | 992 / 1026 | 26408 | 60 | B200, id 6 |
| raw_native | raw room 10 s B=6 seed 0 | native | **one draw** | 1 / 14 | 49326 | 142 / 178 | 26811 | 4 | B200, id 7 |
| raw_torch | raw room 10 s B=6 seed 0 | torch | **one draw** | 1 / 14 | 37865 | 142 / 178 | 29177 | 86 | B200, id 4 |

The one equal array in every room pair is `final__g_slow` (all zero). The largest array differences are the cumulative
spike counts (tens of thousands of spikes over 10 s across a row); per cell, final rates differ by up to 146.2 Hz
(plume_native), 131.0 (plume_torch), 141.5 (raw_native) and 127.3 (raw_torch), final voltages by up to 66.1 / 110.7 /
56.1 / 100.1 mV, and the body traces by up to 91.6 / 94.0 / 86.9 / 78.6 (heading, rad-scale and metre-scale columns
mixed; `arrays.csv` has every array). The metrics that stay equal in the room pairs are the frame counts, the
configuration leaves and the six feeding times (0.0 s in every row of every run: no fly reached fruit in 10 s).

Per-run values (`out/det1/analysis/runs.csv`, columns `run, path_m, final_spike_counts, wall_s`; INTERP 10.4 rule 28):

| run | path m (rows 0-5) | final cumulative spikes (rows 0-5) | wall s |
|---|---|---|---|
| plume_native_r1 | 0.1415,0.1382,0.1406,0.1426,0.1439,0.1436 | 1075260,1071182,1104681,1037737,1105742,1098109 | 128.5 |
| plume_native_r2 | 0.1375,0.1405,0.1412,0.1415,0.1438,0.1455 | 1082577,1072301,1106051,1042875,1073471,1079602 | 115.6 |
| plume_torch_r1 | 0.1383,0.138,0.1382,0.1398,0.1414,0.1426 | 1092037,1077844,1090091,1054224,1092254,1100648 | 122.1 |
| plume_torch_r2 | 0.1424,0.1369,0.142,0.1382,0.1418,0.1448 | 1069919,1062898,1072813,1050029,1106798,1074240 | 126.8 |
| raw_native_r1 | 0.0891,0.0875,0.0861,0.0894,0.0871,0.0883 | 1063088,1041779,1052490,1042311,1063936,1037722 | 77.3 |
| raw_native_r2 | 0.0872,0.0885,0.0875,0.0881,0.0869,0.0873 | 1036277,1062962,1025802,1033554,1047146,1032131 | 116.1 |
| raw_torch_r1 | 0.0887,0.088,0.086,0.089,0.088,0.0879 | 1043694,1065256,1050876,1041395,1039686,1064241 | 42.5 |
| raw_torch_r2 | 0.0879,0.0877,0.0851,0.0888,0.0871,0.0879 | 1036581,1040864,1036088,1012218,1053326,1038856 | 113.3 |

The cx_wedge pair: `GLNO_LR_hz` -0.001827 in both runs, `PEN_LR_hz` 0.032198 in both, `survival_s` 0.0 in both
(`runs.csv`, rows `cxS_r1` / `cxS_r2`); the values are also cx8r's `S_s0` values to the digit
([compass_sign_control.md](compass_sign_control.md) section 4 reproduces all of cx8r's S and V on this node).

## 4. Reading

1. **The cx_wedge protocol repeats exactly** on this GPU: a B=1 FlyBrain on the torch path, no world, no optic lobe,
   a fixed Poisson programme. That is the third exact repeat of this protocol (cx8 == cx8r on node 1, twelve minutes
   apart; cx9 == cx8r across nodes). By INTERP 10.4 rule 16 this is "reproduced exactly in n draws on the torch path",
   never "bit-identical".
2. **Neither room path repeats.** The native path diverges within the first 4 (raw) / 30 (plume) frames; the torch
   path later, at 86 (raw) / 60 (plume) frames -- but it diverges, and once it does the runs are as different as two
   seeds. The difference between the two protocols is not the LIF kernel alone: the room adds B=6, the optic lobe with
   the ray-traced retina, the body and the world. Which of those is the non-repeating stage (batched cuSPARSE SpMM,
   the optic path, the ray tracer, or an atomic in the readout) is **not established here** and is the open question
   this gate leaves. What the saved arrays do show: the first difference in every room pair is exactly +-1 spike in
   exactly one of the six rows (plume_native row 0 at frame 30, plume_torch row 4 at 60, raw_native row 5 at 4,
   raw_torch row 0 at 86); that row's body diverges 1-51 frames later and its antenna / plume columns later still,
   and the other five rows stay bit-identical for a further 215-329 frames. The body integrator and the air / odour
   field are therefore not the source. On the native path the event-scatter CUDA kernel accumulates with a float
   `atomicAdd` (`flyverse/kernels/neural.cu:122`), which is order-nondeterministic by construction and is a
   sufficient mechanism for that path. Nor is the Python layer: the same B=6 raw room on the torch path repeats
   bit-exactly across two separate CPU processes for 120 frames (body, cumulative spikes and the full `v` and `g`
   tensors), with Python hash randomisation on, so an RNG-seeding or iteration-order bug in the Python layer is
   excluded. `scripts/benchmark.py --deterministic` already provides `torch.use_deterministic_algorithms(True)` +
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, so the torch half of the next arm has an existing, unused knob.
   The cx_wedge result shows the torch LIF at B=1 without optics is not the culprit.
3. **Decision (frozen rule): items 3 and 4 run as >= 6 draws with the run as the replicate unit, and no room number
   is quoted to more than its across-draw SD.** The plume_steering.md first-contact times (32.0 ... 21.9 s) and the
   0.841211 end energy stay one draw; the compass_standin room repeats stay observations.
4. The scheduler's "completed" is not evidence: it listed a crashed job as completed with no exit code.

## 5. Reproduction

```sh
PYTHONIOENCODING=utf-8 python scripts/determinism_gate.py analyse --runs out/det1 --out out/det1/analysis
```

Raw runs, consoles and the client log stay ignored under `out/det1/` (they carry host paths); committed are
`batch.sh`, `jobs.json`, `predeclared.json` and `analysis/{pairs,arrays,runs}.csv`, `analysis.md`, `summary.json`.
Every run: device `cuda`, `NVIDIA B200`, the path's backend flags in `provenance.execution.backend`, and every
loaded source file the predeclaration covers at the predeclared hash (LF-normalised; the shipped copies are CRLF and
`files_lf` / the raw hash of the same content are both accepted; three loaded paths -- `flyverse/data/manifest.json`,
`scripts/interp_export.py`, `scripts/probe_object_sweep.py` -- fall outside `source_hashes()` and are not checked);
the pinned id is in `CUDA_VISIBLE_DEVICES` (`runs.csv`) for the eight room runs -- `cx_wedge` does not record the
variable, so cxS's id is only in the client log's `[gpu 4]`.

**Withdrawn:** "the pinned id in `CUDA_VISIBLE_DEVICES` (`runs.csv`)" as a statement about every run, and "every
loaded source file at the predeclared hash" without its qualification -- `cx_wedge` records no
`CUDA_VISIBLE_DEVICES` (cxS's `runs.csv` rows carry the field empty and `summary.json` has
`"cuda_visible": [null, null]`), and three loaded paths fall outside `source_hashes()` and are silently skipped.

`out/det1/batch.sh` was regenerated after the runs, with `determinism_gate.py plan --ship flyverse`, so that the
committed plan carries the `--ship flyverse` of the valid submission. As first generated (stamped 23:17:01Z) it
carried no `--ship flyverse` and therefore reproduced the *invalid* second submission; the submitted command line is
also in the ignored `out/det1/client_stdout_0.txt`. The predeclared JSON is NOT re-stamped for the regeneration: it
happened after the results, and neither the plan's job lines nor the comparison rule changed -- which means
`predeclared.json`'s `batch_sha256` (`553ec9cc...`) is the hash of the plan as frozen at 23:17:01Z and does not
match the regenerated file; `jobs_sha256` and the job lines are untouched. `plan --ship` is new tool surface on this
branch (`tests/test_determinism_gate.py`): without the flag the generated plan is byte-identical to the frozen one
apart from its own generation stamp. It also moves the script's own hash, so re-running `analyse` on the merged
branch writes a different `analysis_sha256` into `summary.json` (`fbcd498f...` is the hash of the code that
produced the committed files); `pairs.csv`, `arrays.csv`, `runs.csv` and `analysis.md` all reproduce
byte-identically, 0 problems.

## 6. Author self-review (the independent skeptic pass is the section after the Report block below)

- The comparison is exact and predeclared; the verdicts cannot be moved by a tolerance choice. The `first differing
  frame` is the first frame at which any row's cumulative spike count differs, so a divergence that starts in a
  non-spiking quantity is reported one or more frames late; the array table carries the per-array first index.
- Three stamps is two more than it should be. The first was a shell-quoting error in my planner, the second the
  stale box checkout; both are recorded above rather than folded away, and the tool fixes (`--ship`, the loaded-source
  check, the receipt caveat) are on this branch so the next round does not repeat them.
- The rooms are 10 s, not 60 s: a pair that diverges by frame 86 at 10 s diverges at 60 s; a pair that repeated at
  10 s would have needed the 60 s check before the word "repeats" was used for the 60 s rooms. None repeated.
- The gate does not localise the non-repeating stage; it was not asked to. A pair on FlyBrain alone at B=6 without
  optics, and one with optics but a frozen body, would split it in two jobs.
- "Same GPU": each pair ran on one pinned id, sequentially, in one job; the four room pairs ran on four different ids
  concurrently, which the rule allows (the unit is the pair) and which a strict reader may prefer to see
  serialised, and cxS shared id 4 with raw_torch (5 jobs, round-robin over a pool of 4): the pair that repeats
  exactly ran concurrently with one that does not, which is evidence against co-tenancy as the cause.
- The `det1_attempt1` preview numbers (24 / 125) are quoted for the record only; they come from stale package files
  and support no claim.
- No number in this audit is a physiological claim; the only inference is about the software.

This is the author's review, not the independent skeptic pass. That pass ran on 2026-09-18 (Opus, verdict
mostly sound for this audit); its verdict line and all ten of its claim lines are quoted verbatim in
[Skeptic pass (independent, Opus, 2026-09-18)](#skeptic-pass-independent-opus-2026-09-18), after the Report
block, and the corrections it required are applied above.

## Report

```yaml
summary: |-
  The cx_wedge protocol (B=1 FlyBrain, torch path, no world) repeats exactly on one B200: two sequential runs of
  the round-7 S arm at seed 0 agree on 31/31 saved arrays and 546/546 numeric metrics. Neither room path repeats:
  a 10 s B=6 room diverges on the native path from frame 4 (raw) / 30 (plume) and on the torch path from frame 86
  (raw) / 60 (plume), after which the two runs differ as two seeds do (final per-cell rates to 127-146 Hz apart, 1
  of 14-15 arrays equal, the all-zero g_slow). By the frozen rule, items 3 and 4 run as >= 6 draws with the run as
  the replicate unit and no room number is quoted beyond its across-draw SD; every room number quoted before this
  gate stays one draw. The non-repeating stage in the room stack (batched SpMM, optic lobe, ray tracer or readout)
  is not localised here. Two submissions were invalid before the valid one (a shell-quoting error, then a box
  checkout behind origin/main whose stale PlumeNavigation crashed the plume pairs); both are recorded, and
  cluster_run.py gained --gpu-ids and --ship. The scheduler listed a crashed job as completed with no exit code, so
  its receipt is not evidence of success in this round. Nothing adopted; no default changed.
key_claims:
- cx_wedge S arm seed 0 repeats exactly on one B200 (31/31 arrays, 546/546 metrics; two sequential runs, one pinned GPU).
- Every B=6 room pair is one draw on both paths; first differing frames 4 / 30 (native) and 86 / 60 (torch).
- Frozen decision: items 3-4 are >= 6 draws, run = replicate unit, numbers quoted with their across-draw SD.
- The divergence is in the room stack, not the B=1 torch LIF; its stage is not localised.
validation:
- Valid run det1-a4facc: 5 jobs, 10/10 consoles device cuda, NVIDIA B200 x 10, pinned ids 4-7 (recorded in the eight room runs; cx_wedge records no CUDA_VISIBLE_DEVICES, so cxS's id 4 is from the client log), 0 analysis problems, every loaded source the predeclaration covers at the predeclared hash (three loaded paths fall outside source_hashes()).
- Comparison rule frozen in out/det1/predeclared.json (stamps 23:14:57Z / 23:17:01Z / 23:31:08Z, all before the valid submission) and applied without tolerance.
- The second submission (det1-5b7a42) is invalid and unused; its files are kept, ignored, under out/det1_attempt1/.
recommendations:
- Quote no room number to more than its across-draw SD; run rooms as >= 6 draws.
- Localise the non-repeating stage with two more pairs (FlyBrain B=6 without optics; optics with a frozen body) before any exact-workload claim.
- Run a deterministic-kernels arm next: the native event-scatter atomicAdd replaced by a deterministic reduction, the torch path under torch.use_deterministic_algorithms(True) (scripts/benchmark.py --deterministic already sets it with CUBLAS_WORKSPACE_CONFIG=:4096:8), then det1 repeated.
- Treat the scheduler's completed status as a status, never a receipt; read the console and the outputs.
open_questions:
- Which room stage does not repeat, and whether torch.use_deterministic_algorithms (already available as scripts/benchmark.py --deterministic) or a B=1 room would repeat; the Python layer is excluded by a two-process CPU control, and the native path's float atomicAdd (flyverse/kernels/neural.cu:122) is a sufficient mechanism for that path.
```


## Skeptic pass (independent, Opus, 2026-09-18)

An independent skeptic pass ran on 2026-09-18 (Opus, CPU only, no cluster job, nothing adopted). Its verdict line
and its claim lines are quoted verbatim below. The CORRECTIONS REQUIRED list is applied in place in the sections
above; where a correction replaced a sentence that stated a finding, the original sentence stays in the record
marked **Withdrawn:** (INTERP 10.4 rule 29 iii). The pass's NOT CHECKED list is recorded verbatim with this
round's entry in [receptor_verification.md](receptor_verification.md).

One pass covered round-8 items 1 and 2 -- this audit (batch `det1`) and
[compass_sign_control.md](compass_sign_control.md) (batch `cx9`) -- and carried one verdict line per audit; the line for this audit is
quoted below, and all ten claims are quoted in each of the two.

### Verdict

```text
VERDICT
- determinism_gate: mostly sound

Both audits' quantitative content reproduced exactly; every correction below is documentation-level or an under-statement, none overturns a verdict.
```

### Claims

```text
CLAIMS 1-10
1. det1 cxS "31/31 arrays, 546/546 metrics" -- REPRODUCED; GPU-id provenance REFUTED in part. Own comparator gives cxS 31/31 arrays and 532/532 numeric leaves equal (booleans dropped), max |d| 0. Re-running `determinism_gate.py analyse` reproduces pairs.csv / arrays.csv / runs.csv byte-identically, 0 problems; summary.json.analysis_sha256 = LF sha256 of the script at HEAD. Same GPU: the eight room runs record cuda_visible_devices 5,5/6,6/7,7/4,4 -- both runs of each pair on one id. cxS records no GPU id at all: runs.csv rows cxS_r1/r2 have the field empty, summary.json has "cuda_visible": [null, null]; "id 4" comes only from the client log. Shipped source: the loaded-file check recomputed for all 10 runs against predeclared.json.source_sha256_lf (73 paths) -- 0 stale, no CRLF fallback needed; but 3 loaded paths are outside the predeclared set and silently skipped (flyverse/data/manifest.json, scripts/interp_export.py, scripts/probe_object_sweep.py). The four stale box files, recomputed from out/det1_attempt1: flyverse/compass.py, fly.py, instruments.py, navigation.py -- as stated (the attempt-1 room runs also show scripts/determinism_gate.py, which changed at the third stamp).
2. "Every B=6 room pair is one draw; native 4/30, torch 86/60" -- REPRODUCED, and the divergence IS partly localisable. Frames 4/30/60/86 confirmed. The first difference in every pair is exactly +-1 spike in exactly one of six rows (plume_native row 0 @30, plume_torch row 4 @60, raw_native row 5 @4, raw_torch row 0 @86); that row's body follows 1-51 frames later, its antenna/plume columns later still, and the other five rows stay bit-identical for a further 215-329 frames. So the source is upstream of the body integrator and of the air/odour field. New CPU control: the same B=6 raw room (world + optics + body, torch flags) run 120 frames in two separate processes on the CPU is bit-equal in body, cumulative spikes and the full v and g tensors (max |d| 0), with Python hash randomisation on -- this excludes an RNG-seeding or dict/set-ordering bug in the Python layer; the non-repeat is CUDA-kernel-level. Two mechanisms visible in the repo: flyverse/kernels/neural.cu:122 uses a float atomicAdd in the event-scatter kernel (cuda.event_scatter), order-nondeterministic by construction -- a sufficient mechanism for the native path; and the torch path's Brain._matmul_add cuSPARSE CSR SpMM, deterministic at B=1 (cxS repeats) but B=6 in the room. Also: scripts/benchmark.py --deterministic already sets torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8 -- the gate's own open question has an existing, unused knob. "Not localised" is honest but understated.
3. Frozen-before-results -- CONFIRMED from git. Stamps 23:14:57Z / 23:17:01Z / 23:31:08Z = commits c4fbb9d / 091e268 / 2d70edb. The comparison_rule (which contains decision_for_items_3_and_4) hashes identically across all three (5fcecaa5...), as do pairs -- the decision was frozen at the first stamp, before attempt 1 and the valid submission. Runs loaded determinism_gate.py at the third stamp's hash; room() unchanged after the runs. The audit applies the decision to itself.
4. Three re-freezes / superseded data -- RECORDED, two defects. (a) The stated reason for the third re-stamp is partly wrong: git diff 2d70edb^..2d70edb adds only the loaded-source check; runs.csv was added later, in the post-run analysis commit 113f737. (b) out/det1_attempt1/ is untracked -- kept, unused, but not reproducible by a reader. The scheduler caveat IS stated in sections 2 and 4.4 and the audit never quotes "0 failed" as evidence -- but no scheduler receipt file was fetched into out/det1/ or out/cx9/ (INTERP 10.4 rule 21); only the gitignored client console.
5. cx9 three primaries -- REPRODUCED exactly, including against the MAIN checkout's out/cx8r. -2.0665 / Holm 0.0065 / Welch -7.83; +4.2848 / +6.63 / +10.61; -36.1679 / -54.66 / -47.58; all separated; 18 runs, 0 problems; compare.csv/descriptive.csv/per_seed.csv/replication.csv/runs.csv byte-identical to the committed ones. Family satisfiable (6v6 floor 0.0021645 x 3 = 0.0065). Bit-equality confirmed at full precision: V-'s GLNO_L_hz_turn = S's to the last bit in all six seeds; 96 of 138 numeric metrics are exactly equal V- vs S in all six seeds. Arm reversal survives all three (reversed z = +3.20, -5.72, +20.78, all still result) -- but primary 1 clears |z| >= 3 by only 0.20, where round 7's analogous primary 3 reversed gave 2.963 and flipped to null. The "mirror of V" is overstated: V's GLNO_R equals S's exactly in only 3 of 6 seeds (2,3,5); seeds 0/1/4 give 0.3316/1.3144/0.4936 vs S's 0.0018/0.0020/0.0000.
6. "76/76 rows equal across two nodes" -- TRUE, but a METRICS-ONLY check, and understated. replication.csv compares the run JSON's 38 KEYS x 2 arms = 76 rows; no array is compared. Reproduced against <workstation-checkout>\out\cx8r, max |diff| 0. Extended: the ledger NPZs are bit-equal too -- 384/384 arrays over the 12 S/V runs (31 per S run, 33 per V run), and the full numeric JSON record matches 6894/6906 leaves, the 12 differences being solely provenance.source_fingerprint.n_files (55 vs 53, a --ship bookkeeping artefact).
7. The two failed/infrastructure runs -- COMPLETE and not treated differently. V-_s1 and S_s2: device cuda, NVIDIA B200, frames 800, ledger NPZ present (33 and 31 arrays), cache md5 ef23cc27..., wall 75.7 / 40.5 s, consoles end with the full closing summary line, checks column empty. load_runs globs *_s*.json and applies identical checks. Caveat: cx_wedge records carry no timestamp at all. Dropping both seeds: all three still result (5v5 -2.1246; 6v5 +4.3432; 6v5 -36.3768), all separated, Holm 0.013. Job indices corroborated by the client log (cx9-2daa4d-5 = V-_s1, -6 = S_s2), but the receipt itself is in no file.
8. Scope of the inference -- CORRECT in section 4, OVERREACHED once in the Report. Three unheld, unrelabelled arms can only speak to the unheld ring, and section 4 says exactly that. Section 4 "Answer" == Report summary verbatim (925 chars each). But Report open_questions says "Why the hold plus relabel removes the sign specificity...", which presupposes a removal that is not established: recomputed from out/cx8r, HGV vs HGV- on GLNO_LR_hz is a null by the project's own rule (diff +3.1092, z +1.625, p 0.0260, not separated; HGV +3.7955 +- 1.8362 vs HGV- +0.6862 +- 1.9132, HG +2.5705 +- 1.2286).
9. Rule checks -- PASS. 14/14 spot-checked per-seed/per-run values re-derived from the run JSONs (10 cx9 + 4 det1) match at the stated precision; det1's cxS GLNO_LR -0.001827 / PEN_LR 0.032198 / survival 0.0 identical in both runs and bit-equal to cx9's S_s0. Stamps precede runs (det1 23:31:08Z vs runs 23:32-23:36Z; cx9 23:43:39Z vs 23:46-23:47Z). Holm families satisfiable. git diff a455bcf..1972568 scanned for hostnames / IPs / beegfs / /mnt / ssh / user@host: zero hits. No commit ids are quoted in either audit -- note INTERP rule 30 exists on main but not on this branch, so re-check on merge. No unfilled ALL_CAPS placeholders. Both self-review sections carry "(the independent skeptic pass is not this section)". The items 1-2 commits (9f93871..6111bbc) touch no file under flyverse/ -- "nothing adopted" is verifiable.
10. Tool changes -- SOUND, with two unrecorded consequences. tests/test_cluster_run.py: 52 passed. --gpu-ids: validated before anything ships (rejects "", "4,x", "-1", "4,4"); round-robin pool[i % len(pool)] in command order; gpus: 1 + gpu_ids: [id] set only with the flag, vram_gb untouched, and with no flag no gpu_ids key and no "gpu pool" log line. Empirically honoured (det1 requested 4,5,6,7,4 and the room runs recorded CUDA_VISIBLE_DEVICES 5,6,7,4; cx9's 18 jobs end at [gpu 5]). "The scheduler treats gpu_ids as a strict pin" is an assertion about the scheduler that neither the tests nor the tool can enforce; a pool smaller than the job count means jobs share a GPU (det1 5/4, cx9 18/4). --ship does loosen "checkout == provenance" -- the run tree is a hybrid (box checkout + local diff + all tracked files under the shipped paths), recorded as source_fingerprint.git.commit = "unknown", identity falling back to content (rule 6). What shipped is NOT recorded in any run's provenance: the only record is a truncated client-console line in a gitignored file. Second limitation: the overlay only copies -- a file deleted locally but present on the box is never removed, so --ship can only bring a stale box tree closer, never make it equal.
[the quoting agent replaced one workstation checkout path with a placeholder in this quotation]
```

Claim 9's re-check on the merge (INTERP 10.4 rule 30 is now on this branch): clean. The commit ids the
quotation above names -- `a455bcf`, `9f93871`, `6111bbc`, `1972568`, `c4fbb9d`, `091e268`, `2d70edb`,
`113f737` -- all resolve in the post-rewrite history, and none of them is an id the 2026-09-17 rewrite
changed (`docs/audits/commit_map_2026-09-17.json`). A sweep of every tracked text file for pre-rewrite ids
returns nothing, so no sentence in this audit needs remapping.
