# The lesion tool: check x lesion matrices, replicate scatter and double dissociations

Tool: `flyverse/interp/lesion.py` (`lesion(manifest, *, out_dir, mode, ...)`; `plan` / `run` / `analyse`), CLI
`scripts/interp_lesion.py`, tests `tests/test_interp.py::LesionTests` (8 CPU tests). Contract: `docs/INTERP.md`
section 4.4. Data: `out/les_holds/` and `out/les_holds_run1/` (the two GPU batches, 12 job JSONs each),
`out/les_holds_all/` (the two merged: 4 draws per arm), `out/les_cpu/` and `out/les_cpu2/` (two independent CPU runs,
24 job JSONs each), `out/les_orn/` (9 job JSONs, the weight-mask arm). Results:
`out/interp/lesion/holds_all.json` (the GPU matrix), `holds.json` (the second batch alone), `holds_cpu.json` /
`holds_cpu2.json` (the CPU dissociation), `orn.json`, `holds_plan.json` (the resolved manifest + batch). Console logs:
`out/les-holds_cluster_bd4380.log`, `out/les-holds_cluster_1df2ca.log`, `out/les-holds_cluster_80f423_failed.log`
(the first, failed submission), `out/les_cpu/run_console.txt`, `out/les_cpu2/run_console.txt`,
`out/les_orn/run_console.txt`, and the printed matrices in `out/les_holds_all/analyse_console.txt`,
`out/les_cpu/analyse_console.txt`.

**Result in one line.** The tool reproduces the round-4 / round-5 hold attribution: **59 of 60 published GPU numbers
and 32 of 32 published per-seed CPU numbers**, all six hold tables' changed-entry counts and md5s exactly, and it
extracts the `(holdBrainGlu, holdBrainHis) x (smell.KC_active, taste.MN9_hz)` double dissociation automatically. The
single miss is `loom.GF_peak_hz` under `holdOptic` -- the check `receptor_integration.md` E.2 itself flagged as
scattering.

---

## 1. What the tool is

```python
lesion(manifest, *, out_dir, mode="plan", replicates=3, checks="all", probes=(), seeds=None, cluster=False,
       minutes=60, baseline="baseline", ...) -> common.Result
```

| mode | where | what it does |
|---|---|---|
| `plan` | CPU, local | resolves every lesion on the connectome (bodies, entries, hold-table md5s, changed entries vs `sign(W.data)`) into `<out_dir>/manifest.resolved.json`, and writes `<out_dir>/batch.sh`: **one** `scripts/cluster_run.py` call, one job per lesion x replicate, `--fetch <out_dir>/` |
| `run` | GPU (cluster) or CPU | one arm in this process: `scripts/benchmark.py`'s sections through a `benchmark.Context` subclass (the `retire_measures.py` pattern, so a weight mask can be installed) plus the manifest's probes through their CLIs; writes `<out_dir>/<lesion>_r<k>.json` (`flyverse.interp.lesion.job/1`) |
| `analyse` | CPU, local | the finished job JSONs -> `matrix` (check x lesion), `sensitivity` (Neurome fields), `dissociations`, `lesions`, and `validation` filled from `EXPECTED` |

The manifest is a dict / JSON / YAML path / JSON string, or one of two builtins: **`holds`** (the round-4 + round-5
arms: `baseline`, `holdKC`, `holdDN1`, `holdBrain`, `holdOptic`, `off`; sections `rest,taste,smell,walk,bitter`) and
**`holds_cpu`** (the E.4 arms with `holdBrainGlu` / `holdBrainHis`, sections `taste,smell`, brain seeds 0/1/2).

## 2. How each lesion is applied -- nothing in `flyverse/` is edited

| kind | mechanism | recorded in the job JSON |
|---|---|---|
| `hold_table` | `LIFParams.receptor_table` = a `scripts/build_hold_tables.py` table, rebuilt by the job when missing (`out/` is never shipped to the cluster) | path, md5, group, `entries_changed_vs_sign_W` |
| `receptor_tier` | a temporary table with every row the lookup decided at that tier set back to `NT_SIGN[transmitter]`, `fast_net_abs='held'` (`tier_hold_table`) | path, md5, `rows_held` |
| `population` / `types` / `module` / `transmitter` | `brain._shaped_weights` is wrapped for the duration and the presynaptic columns zeroed (`weight_mask`); `scope='graph'` also zeroes them in `c.W` so the rate optic lobe sees the lesion | bodies (up to 5,000), types, `n_entries_removed`, `synapses_removed`, `applies_to`, `fan_in_note` |
| `lif` / `optic` / `pair_gain` | `LIFParams` / `OpticParams` overrides applied in `Context._apply_lif` / `_apply_optic` | the override dict |
| `none` | the baseline arm | -- |

Two mechanics worth stating because they change what a number means:

* **`scope='lif'` does not reach the optic lobe.** `optic.OpticLobe` reads `c.W` directly, so a population lesion of
  an optic type silences it for the LIF only unless `scope='graph'` is asked for. The job JSON says so in
  `applies_to`.
* **A weight mask also removes its entries from the postsynaptic fan-in totals**, because `Brain` normalises on the
  matrix it is handed. This differs from `screen.ablate` (which zeroes rates and keeps the fan-in) and is recorded as
  `fan_in_note`. Section 6 shows it mattering.

The wrapper copies `c.W` before shaping (`retire_measures.py`'s `_protected`); `brain._shaped_weights` copies
internally as well, so several arms in one process are independent, and `ctx.c.reference._norm_cache` is cleared per
arm.

---

## 3. Validation A -- the hold tables themselves (`receptor_integration.md` E.0), local CPU

`plan` recomputes each table's effect with `connectome.receptor_signs(c, table_path=..., net_rule='abs')` against
`sign(W.data)` on the shipped cache (25,578,600 stored entries, sum |W| 121,460,584, md5
`ef23cc27bea13be7f6a96f3c04fd3737`).

| table | E.0 entries changed | this tool | md5 (E.0) | md5 measured |
|---|---|---|---|---|
| shipped default | 48,295 | **48,295** | `0381a446107e6050e75cc87b16d7f830` | same |
| `out/receptors_holdBrain.csv` | 44,463 | **44,463** | `d902daf5c7efd94cfedaade7a2f135f3` | same |
| `out/receptors_holdOptic.csv` | 3,832 | **3,832** | `c3baf4293f508bb39d2d42f4b87163dc` | same |
| `out/receptors_holdBrainGlu.csv` | 44,586 | **44,586** | `2e1b53f2026fb077dab06b806aaf2d1f` | same |
| `out/receptors_holdBrainHis.csv` | 48,172 | **48,172** | `6fadd62df4d3e1a4121b9d08310ed565` | same |
| `out/receptors_holdKC.csv` | 45,461 | **45,461** | `95bf26f5c78ec4964fba0d1c1fbfccfd` | same |
| `out/receptors_holdDN1.csv` | 47,420 | **47,420** | `3ac8523efb800c6789d6d76c07233cfd` | same |
| `off` (`receptor_model=None`) | 0 | **0** | -- | -- |

8 of 8 reproduced (`out/interp/lesion/holds_all.json`, `validation.measured.entries_changed`; the six arms of the GPU
manifest there, all eight in `holds_cpu.json`). The KC and DN1 tables are byte-identical to the round-4 ones.

## 4. Validation B -- the GPU hold matrix (E.2 and the round-4 re-score)

**The batches.** `out/les_holds/batch.sh` -> one `cluster_run.py` call, 12 jobs (6 arms x 2 replicates),
`--fetch out/les_holds/`. Run twice, exactly as E.2's own (accidental) duplicate batch did, so every arm has **4
independent runs**:

| run dir | jobs | failed | wall | log |
|---|---|---|---|---|
| `les-holds-80f423` | 12 | **12** | 1.5 min | `out/les-holds_cluster_80f423_failed.log` (the bug of section 8.1) |
| `les-holds-bd4380` | 12 | **0** | 2.9 min | `out/les-holds_cluster_bd4380.log` |
| `les-holds-1df2ca` | 12 | **0** | 2.7 min | `out/les-holds_cluster_1df2ca.log` |

Every job: `device` = **cuda**, `device_name` **NVIDIA B200** (the realised device out of `brain_probe`, never the
request), `receptor_model 'sign'`, `receptor_net_rule 'abs'`, receptor table md5 `0381a446107e6050e75cc87b16d7f830`,
`w_syn 0.275`, `conn_cap 60`, `type_path_gain [('^(LC4|LPLC2)$', '^DNp01$', 3.0)]` (post-retirement), body thresholds
`gf_hz 33`, `takeoff_power_hz 50`. 1.5-2.0 min per job. The cluster's run-copy cache and this desktop's cache compile
to the **same graph**: md5 `ef23cc27bea13be7f6a96f3c04fd3737`, 25,578,600 stored entries, sum |W| 121,460,584,
167,106 cells in both (`provenance.compiled_connectome` of `holds_all.json` vs `holds_cpu.json`).

One provenance note the `variants` record turned up: with sections `rest,taste,smell,walk,bitter` **no Brain in a job
is built with the native backend flags** -- all 8 per job are plain torch CSR on cuda (`event_driven false`,
`cuda_kernels false`, `cuda_sparse 'torch'`). `benchmark.py`'s `native (cuda_kernels, cuda_graphs, event_driven,
warp)` label, which the job JSON copies, describes the flags `Context.sim()` hands the room demo, i.e. the demo
sections this manifest does not run.

**The matrix** (`out/interp/lesion/holds_all.json`, 24 jobs; `x4` = the four draws are bit-identical). Published =
E.2's table for `baseline` / `holdBrain` / `holdOptic` / `off`, the round-4 re-score for `holdKC` / `holdDN1`.

| check | arm | published | this tool (4 draws) |
|---|---|---|---|
| `rest.spikes_per_step` | all six | 0.0000 | 0.0000 x4 |
| `taste.MN9_hz` | baseline / holdKC / holdDN1 / holdOptic | 10.9342 / 10.93 / 10.93 / 10.9342 | **10.9342 x4** in all four |
| | holdBrain / off | 5.8455 / 5.8455 | **5.8455 x4** in both |
| `smell.KC_active` | baseline / holdKC / holdDN1 / holdOptic / holdBrain / off | 816 / 1155 / 1109 / 816 / 1426 / 1426 | **816 / 1155 / 1109 / 816 / 1426 / 1426**, x4 each |
| `smell.PN_hz` | same order | 7.8612 / 11.64 / 10.64 / 7.8612 / 11.1877 / 11.1877 | 7.8612 / **11.6413** / **10.6367** / 7.8612 / 11.1877 / 11.1877, x4 each |
| `bitter.calibrated_sugar_MN9_hz` | same order | 5.5184 / 5.52 / 5.52 / 5.5184 / 4.5659 / 4.5659 | 5.5184 / 5.5184 / 5.5184 / 5.5184 / 4.5659 / 4.5659, x4 |
| `bitter.calibrated_sugar_bitter_MN9_hz` | all six | 0.0000 | 0.0000 x4 |
| `bitter.shiu_sugar_MN9_hz` | same order | 139.8985 / 139.90 / 129.27 / 139.8985 / 123.5394 / 123.5394 | 139.8985 / 139.8985 / **129.2657** / 139.8985 / 123.5394 / 123.5394, x4 |
| `bitter.shiu_sugar_bitter_MN9_hz` | same order | 0.8178 / 0.00 / 0.82 / 0.8178 / 2.1243 / 2.1243 | 0.8178 / 0.0000 / 0.8178 / 0.8178 / 2.1243 / 2.1243, x4 |
| `walk.GF_max_hz` | baseline / holdOptic / holdBrain / off | 4.6292 / 13.3109 / 12.5174 / 4.9641 | **4.6292 / 13.3109 / 12.5174 / 4.9641**, x4 each |
| | holdKC / holdDN1 | (8.41 / 4.72, pre-retirement) | 8.4497 x4 / 4.7458 x4 -- **new** |
| `walk.power_max_hz` | baseline / holdOptic / holdBrain | 48.4805 / 64.9147 / 47.0012 | **48.4805 / 64.9147 / 47.0012**, x4 each |
| | off | 95.5416 / 97.1014 (E.2), 96.46 (skeptic) | 95.5416 x3 / **96.4573** |
| | holdKC / holdDN1 | (55.63 / 59.04, pre-retirement) | 42.2556 x4 / 59.0372 x4 -- **new** |
| `walk.power_sustained_hz` | baseline / holdOptic / holdBrain | 20.1091 / 33.5072 / 21.3426 | **20.1091 / 33.5072 / 21.3426**, x4 each |
| | off | 49.1802 / 50.5960 | 49.1802 x3 / **49.6141** |
| | holdKC / holdDN1 | (23.68 / 30.75, pre-retirement) | 17.0012 x4 / 30.7489 x4 -- **new** |
| `loom.GF_peak_hz` | baseline | 43.5929 / 46.4969 | 43.5929 / 44.1239 / 47.2162 / 47.2162 |
| | holdBrain | 50.0089 / 50.0089 / 51.2904 / 60.0354 | 50.0089 / 60.0354 x3 |
| | holdOptic | 31.7754 x3 / 32.0792 | 31.7754 / 32.1402 / 34.1576 x2 -- **the one miss** |
| | off | 27.9926 / 29.0034 | 26.4500 / 27.5783 / 27.9926 / 30.0596 |
| | holdKC / holdDN1 | (49.01 x2 / 48.72, 53.76, pre-retirement) | 42.2293 x2 / 43.1074 / 47.6674; 47.2542 / 55.3060 / 55.5591 x2 |
| `loom.escape_cm` | all six | 3.50 | 3.50 x4 |
| `rotate.DNp20_flip_hz` | baseline | -41.8507 / -39.8623 | -40.7614 / -34.6514 / -33.5969 / -30.5012 |
| | holdBrain / holdOptic / off | -26.20..-42.68 / -26.38..-31.47 / -30.05, -30.62 | -28.00..-32.08 / -12.76..-24.26 / -23.61..-30.42 |

**59 of 60 compared rows reproduced** (`validation.measured.n_reproduced`). **Ten** of the fourteen checks are
bit-identical across all four draws in **every** arm -- taste, both smell checks, all four bitter checks,
`walk.GF_max_hz`, `rest.spikes_per_step`, `loom.escape_cm` -- so those attributions carry no scatter at all, exactly
as E.2 reported for its nine bit-stable checks. `walk.power_max_hz` and `walk.power_sustained_hz` are bit-identical in
five of the six arms; only `off` varies, in 1 of 4 draws (96.4573 and 49.6141), and both of those values sit inside
the published set (95.54 / 96.46 / 97.10 and 49.18 / 49.61 / 50.60), so those two checks fall through to the scatter
rule, which at four draws still calls every arm moved.

**The miss.** `loom.GF_peak_hz` under `holdOptic`: two of our four draws are inside E.2's range (31.7754 -- bit-equal
to one of its draws -- and 32.1402), two are 2.1 Hz above it (34.1576), and the mean 33.0577 lies 1.0 Hz above the
top of the published range, so the row is scored `not reproduced` and the whole GPU arm's status is `not reproduced`.
This is the check E.2 itself refused to quote as a value (10.0 Hz of spread inside `holdBrain` there). The direction
is unchanged: `holdOptic` 33.06 sits between `off` 28.02 and `baseline` 45.54, i.e. the Brain side alone recovers
29 % of the default's loom GF, against E.2's 20-22 %.

**The dissociation the tool extracts here** (bit-identical criterion, 4 draws): `holdDN1` moves
`bitter.shiu_sugar_MN9_hz` (-10.6328) and not `bitter.shiu_sugar_bitter_MN9_hz`, while `holdKC` moves the bitter
check (0.8178 -> 0.0000) and not Shiu sugar -- so within the Brain side's glutamate group, **the DN1 flips carry
65 % of Shiu sugar and none of the bitter suppression, and the KC flips the reverse**, bit-exactly, in 4 of 4 draws.
Round 4 had both halves of this in its table (`holdDN1` 129.27, `holdKC` bitter 0.00) but did not pair them.

**`walk.power_max` is non-monotone in the number of applied flips**, as round 4 said, and now under the current
default: baseline 48.4805, `holdKC` 42.2556 (**below** the default), `holdDN1` 59.0372, `holdBrain` 47.0012,
`holdOptic` 64.9147, `off` 95.5416 -- the two complementary single-side holds move it by -1.48 and +16.43 while
removing both moves it by +47.29, so the sides are strongly non-additive. Per the project rule this check is reported,
never used as a verdict and never tuned to.

## 5. Validation C -- the CPU double dissociation (E.4)

`interp_lesion.py run --manifest holds_cpu --device cpu --sections taste,smell --replicates 3`: 8 arms x 3 brain
seeds = 24 jobs in one local process, ~13 s of simulation each. Run **twice** (`out/les_cpu/`, `out/les_cpu2/`); the
two runs agree in **24 of 24 matrix rows, every digit**, and each reproduces **32 of 32** published per-seed numbers.

| condition | check | E.4 per seed 0 / 1 / 2 | this tool |
|---|---|---|---|
| off | `taste.MN9_hz` | 1.554839 / 4.341760 / 2.309808 | 1.5548391 / 4.3417597 / 2.3098083 |
| baseline | | 5.090923 / 4.315772 / 2.360074 | 5.0909228 / 4.3157721 / 2.3600743 |
| `holdBrain` | | = off | = off, every digit |
| `holdOptic` / `holdBrainGlu` / `holdKC` / `holdDN1` | | = default | = default, every digit |
| `holdBrainHis` | | = off | = off, every digit |
| off | `smell.KC_active` | 1079 / 427 / 1178 | 1079 / 427 / 1178 |
| baseline | | 486 / 412 / 543 | 486 / 412 / 543 |
| `holdBrainGlu` | | = off | 1079 / 427 / 1178 |
| `holdBrainHis` | | = default | 486 / 412 / 543 |
| `holdKC` | | 1225 / 464 / 1006 | 1225 / 464 / 1006 |
| `holdDN1` | | 525 / 433 / 540 | 525 / 433 / 540 |
| `holdKC` | `smell.PN_hz` | 9.9661 / 2.9307 / 8.1874 | 9.9661236 / 2.9306905 / 8.1874275 |
| `holdDN1` | | 3.9050 / 2.4012 / 4.2898 | 3.9050236 / 2.4011796 / 4.2897840 |

`analyse` extracts **6 double dissociations** from these 24 jobs under the seed-paired criterion, the first of which
is the one the contract names:

| lesion A | lesion B | check A (moved by A, not B) | check B (moved by B, not A) |
|---|---|---|---|
| `holdBrainGlu` | `holdBrainHis` | `smell.KC_active` +414.33 | `taste.MN9_hz` -1.19 |
| `holdBrainGlu` | `holdBrainHis` | `smell.PN_hz` +2.66 | `taste.MN9_hz` -1.19 |
| `holdKC` | `holdBrainHis` | `smell.KC_active` +418.00 | `taste.MN9_hz` -1.19 |
| `holdKC` | `holdBrainHis` | `smell.PN_hz` +3.40 | `taste.MN9_hz` -1.19 |
| `holdDN1` | `holdBrainHis` | `smell.KC_active` +19.00 | `taste.MN9_hz` -1.19 |
| `holdDN1` | `holdBrainHis` | `smell.PN_hz` -0.09 | `taste.MN9_hz` -1.19 |

(The deltas are means over the three seeds; the criterion is per seed. The `holdDN1` rows are real but small -- 525 /
433 / 540 against the default's 486 / 412 / 543 -- which is E.4's "the DN1 flips carry little of it".)

## 6. A weight-mask lesion end to end (`out/les_orn/`, CPU, 3 seeds)

To exercise the other lesion kind on the real connectome, a manifest with `{"id": "no_ORN", "kind": "population",
"spec": "~^ORN"}` beside `holdBrainHis`. The resolved record: **2,635 bodies, 229,681 entries, 1,349,270 synapses
removed**, `scope='lif'`.

| check | baseline (seeds 0/1/2) | `no_ORN` | `holdBrainHis` |
|---|---|---|---|
| `smell.PN_hz` | 3.6941 / 2.9017 / 4.2780 | **0.0 / 0.0 / 0.0** | = baseline |
| `smell.KC_active` | 486 / 412 / 543 | **0 / 0 / 0** | = baseline |
| `taste.MN9_hz` | 5.0909 / 4.3158 / 2.3601 | 1.6043 / 4.3710 / 11.5924 | 1.5548 / 4.3418 / 2.3098 |

The positive control is exact (no ORN output -> no PN, no KC, in 3 of 3 seeds). The tool reports **0 dissociations**
here, correctly: `no_ORN` also moves `taste.MN9_hz` at every seed (and not in one direction -- down at seeds 0 and 1,
up 5x at seed 2), so it does not dissociate from `holdBrainHis`. Two mechanisms are confounded in that and were not
separated here: the lost ORN drive itself, and the fan-in effect of section 2 (the 229,681 removed entries also leave
the input totals of every postsynaptic partner, so their input scale changes). That is why a weight mask is not a
drop-in replacement for `screen.ablate`, and why the job record carries `fan_in_note`.

## 7. What the analysis calls, and what it refuses to call

`matrix` decides "did this lesion move this check" by three criteria, in order, and prints which one it used:

1. **bit-identity within every arm** (10 of our 14 GPU checks): any difference at all is a move. This is E.2's own
   criterion and needs no scatter.
2. **paired by brain seed** (the CPU protocol): moved iff every paired difference is non-zero, not moved iff every
   one is zero -- E.4's "= off, every digit".
3. **the scatter rule** otherwise: beyond twice the pooled replicate sd = moved, at or below one pooled sd = not
   moved, between = unclear -- **and `underpowered`, whatever the numbers, when an arm has fewer than three runs**
   (`common.MIN_REPLICATES`). An `underpowered` row is quoted with its draws and cannot enter a dissociation.

That third clause was added because of this validation. With the first batch alone (2 draws per arm) the matrix called
`rotate.DNp20_flip_hz` moved by `holdOptic` (+17.12 Hz against a pooled sd of 7.71) and produced **13** dissociations,
most of them resting on that one noisy check -- the very check E.2 declined to attribute. Under the scatter rule those
rows become `underpowered` and the count drops to **1**. With both batches (4 draws) the check is callable again and
the tool says: `holdOptic` moved it (-34.88 -> -20.44, pooled sd 4.81), `holdBrain` and `off` **unclear**, `holdKC`
not moved. E.2, with 12 draws, declined to attribute the check at all because its `holdBrain` draws spanned every
condition; ours are unclear there too. The `holdOptic` call is therefore **new and provisional** -- 4 draws on a check
with 5 Hz of scatter -- and is not a result of this round.

Summary of the merged GPU run: 38 moved rows, 0 underpowered, 8 dissociations, 10 bit-identical checks
(`out/interp/lesion/holds_all.json`, `summary`).

## 8. Defects this validation found (all fixed in `flyverse/interp/lesion.py`)

1. **The first submission lost all 12 jobs in 34 s.** Every job redirected its stdout into `out/<run>/`, but `out/` is
   git-ignored, so `cluster_run.py` never ships it and the cluster's run copy has no such directory:
   `/tmp/<scheduler>_run_2a50702c.sh: line 20: out/les_holds/off_r1.txt: No such file or directory`, twice per job (the
   manager retries once). `job_command` now emits `mkdir -p <out_dir> && python -c 'import torch; ...' && ...`, and
   `tests/test_interp.py::LesionTests::test_plan_writes_one_batch_line` asserts every job line starts with it.
2. **`plan` embedded the whole manifest in every command line** (2 kB x 12 = a 24 kB `batch.sh`) even for a builtin.
   `manifest_argument` now names a builtin (`--manifest holds`) when the loaded manifest still equals it, or a
   repo-relative file path when the manifest came from a tracked file, and falls back to embedded JSON only when the
   manifest cannot otherwise reach the job. `batch.sh` is 3.9 kB.
3. **`plan` crashed on Windows when `out_dir` was on another drive** (`ValueError: path is on mount 'C:', start on
   mount 'D:'` -- the temp directory the test uses). `_rel` falls back to the absolute path.
4. **The validation arm was chosen by device alone**, so a GPU matrix whose provenance said `cpu` was compared against
   E.4's per-seed table. The arm is now `cpu` only when every job ran on a cpu device **and** carries its own brain
   seed (E.4's table is per seed); otherwise `gpu`.
5. **Provenance gaps.** A cluster run copy is a file tree, not a git checkout, so every job recorded
   `flyverse_commit.commit = "unknown"`; `analyse` now substitutes the analysing checkout's state with a note and
   keeps the job's value in `job_commit`. `execution.device_name` was always None because the device shim carries a
   string, not a `torch.device`; `brain_probe` now reads `torch.cuda.get_device_name` at Brain construction (hence
   `NVIDIA B200` above) and also records `backend.variants`, every distinct backend configuration the sections used --
   the top-level flags are those of the **first** Brain a job builds (benchmark's taste Brain), not of all of them.
   `benchmark_backend` is now labelled as what it is -- benchmark.py's configuration string, not a realised fact; the
   JSONs of this round carry the string without that label, with `variants` beside it as the realised record.
6. **A second run of `batch.sh` overwrote the first run's console log.** The script now moves an existing
   `out/<name>_cluster.log` aside to `out/<name>_cluster.<timestamp>.log` before teeing.

## 9. Reproducing this

```bash
# A. the GPU matrix: plan (CPU, local) -> ONE batch -> analyse (CPU, local)
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py plan --manifest holds --out out/les_holds \
    --replicates 2 --minutes 30 --name les-holds --json out/interp/lesion/holds_plan.json
sh out/les_holds/batch.sh                       # 12 jobs, --fetch out/les_holds/, log out/les-holds_cluster.log
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py analyse --out out/les_holds --json out/interp/lesion/holds.json
# 4 draws per arm: run the batch twice into two directories and analyse the union
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py analyse --out out/les_holds_all --json out/interp/lesion/holds_all.json

# B. the CPU double dissociation (no optic lobe; ~12 min on this desktop)
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py run --manifest holds_cpu --out out/les_cpu \
    --device cpu --sections taste,smell --replicates 3
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py analyse --manifest holds_cpu --out out/les_cpu \
    --json out/interp/lesion/holds_cpu.json

# C. a weight-mask lesion, manifest given inline
PYTHONIOENCODING=utf-8 python scripts/interp_lesion.py run --manifest '{"name":"orn_cpu","sections":"taste,smell",
  "brain_seeds":[0,1,2],"baseline":{"id":"baseline","kind":"none","receptor_model":"sign","receptor_net_rule":"abs"},
  "lesions":[{"id":"no_ORN","kind":"population","spec":"~^ORN","receptor_model":"sign","receptor_net_rule":"abs"}]}' \
  --out out/les_orn --device cpu --replicates 3

PYTHONIOENCODING=utf-8 python -m pytest tests/test_interp.py::LesionTests -q     # 8 CPU tests, seconds
```

## 10. Limits and open questions

* **Two replicates are not enough for `loom.GF_peak_hz` or `rotate.DNp20_flip_hz`.** The default `--replicates 3`
  (six arms = 18 jobs, ~3 min of cluster time) should be used for any matrix that includes them; at 2 the tool now
  says `underpowered` instead of guessing.
* **`holdKC` / `holdDN1` walk and loom numbers are new** (round 4 measured them before the GF-damping retirement) and
  have no published counterpart to check against. `walk.power_max` `holdDN1` 59.0372 and `power_sustained` 30.7489
  happen to equal round 4's pre-retirement values to 2 dp while every other arm moved; that is not explained here and
  is not claimed as stability.
* **The bit-identity criterion is a property of the protocol, not of the tool.** It holds for 10 of these 14 checks on
  the GPU (12 of 14 at two draws, before `off`'s walk pair showed a second value); a manifest whose sections draw RNG (the room / hops sections) will fall through to the scatter rule, where
  three runs per arm is the floor.
* **A weight mask changes the fan-in normalisation** of the lesioned cells' targets (section 6). A `scope='graph'`
  variant that also reaches the optic lobe exists but was not exercised on the full connectome in this round.
* **The tool never applies a lesion to the model on disk**; it applies parameter overrides and an in-process mask, and
  `docs/INTERP.md`'s rule that the toolkit reads the model is unchanged by anything here.
