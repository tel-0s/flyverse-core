# The Neurome probe export (`flyverse/interp/export.py`) -- build and validation

Interpretability toolkit, build task **export** (2026-09-12). Owns `flyverse/interp/export.py`,
`scripts/interp_export.py`, the `ExportTests` class of `tests/test_interp.py` and this file. Contract:
`docs/INTERP.md` (the `Result` schema, the CLI convention) and `docs/NEUROME_INTERFACE.md` section 1 (the run
directory, the interchange key, the unit-handling table). The export computes no new number: every value it writes
was already in a tool's `common.Result`, so this module is field mapping, hashing and a refusal rule.

---

## 1. What the tool is

`export(result, out_root='out/export', run_id=None, retina=None, parquet_rows=1e6, control_ids=None) -> Path`
writes one run directory `out/export/<run_id>/`:

| file | role | what it is |
|---|---|---|
| `manifest.json` | -- | every field of NEUROME_INTERFACE section 1, taken from the Result's provenance block |
| `result.json` | -- | the source Result verbatim, its SHA-256 in the manifest |
| `readout_per_body.csv` | interchange | one row per (body, quantity); `bodyId` a decimal string; LC11 / LC10a never pooled |
| `contributions.csv` | interchange | `body_pre` / `body_post` edge contributions with the sign and gain rules that made them |
| `sensitivity.csv` | interchange | lesion / hold deltas with replicate scatter |
| `retina_columns.csv` / `retina_bodies.csv` / `retina_radiance.{csv,parquet}` | interchange | the retinal sampling actually presented, and the column -> photoreceptor-body map |
| the tool's own tables (`per_type`, `reference_per_type`, `delta_links`, ...) | tool | written and hashed too, so the directory does not depend on a reader parsing `result.json` |
| `checks.json` | -- | the round trip and `verify` report; written *after* the manifest, so it is not one of the hashed tables |

Every table carries `dataset` / `release` columns (interchange tables), its row count, column list, units and SHA-256
in the manifest, and `role` distinguishes the three contract tables + retina from a tool's own. A `Result` whose
`check()` is non-empty (no provenance, a contract table missing columns, `execution.device is None`) is **refused**
with `ValueError` -- the export never invents provenance it was not given.

**API** (`flyverse/interp/export.py`, CPU only, no torch):

```
export(result, *, out_root, run_id, retina, parquet_rows, control_ids) -> Path      # the serializer
verify(run_dir, *, neurons, expect_paired=('LC11','LC10a'), expect_counts) -> dict  # re-read and re-check
read_table(run_dir, name) -> DataFrame          round_trip_check(result, run_dir) -> dict
retina_tables(retina, out_dir, parquet_rows) -> (table metas, the manifest's retina block)
write_table(df, out_dir, name, parquet_rows, role) -> manifest entry   sha256_file(path) -> str
source_fingerprint(root, patterns, include_loaded, extra) -> dict      loaded_sources(root, extra) -> dict
match_sources(recorded, root) -> dict                                  # which commit a cluster run actually ran
raw_counts(c) -> (csr, sign0_available)         silent_flags(c, pre_idx, frozen_idx, rates) -> DataFrame
links_to_contributions(links, kind, normalisation, reference_graph, window) -> DataFrame
static_decompose_result(c, target, pre, ...) -> Result      # the structural contributions path, end to end on CPU
result_from_object_sweep(stim_jsons, null_jsons, *, cells, null_cells, provenance, reference, reference_null,
                         retina, control_ids, run_id, generator) -> Result
```

**CLI** (`scripts/interp_export.py`): `record` (the only GPU subcommand -- runs the object-sweep protocol and
captures what the probe JSON pools away: per-cell drive and rate, and the retinal sampling), `run` (recorded runs ->
Result -> export), `analyse` (any Result JSON -> export; the default when no subcommand is given), `verify`,
`static-decompose`. Common flags per `docs/INTERP.md` 2.6 / 5.

---

## 2. What ran

**GPU (cluster).** Four batches of 6 jobs, each `--receptor-model off --receptor-net-rule class`, seeds 0/1/2 for the
ball arm and the same three for the none-vs-none null, 12 s window after a 3 s settle, ball r 0.005 m at 0.05 m
(11.42 deg), half sweep 0.06 m, every job `device cuda`, NVIDIA B200, torch 2.11.0+cu128:

| batch | run dir | jobs | wall | why it was re-run |
|---|---|---|---|---|
| `exp-obj-ce800c` | `out/expobj/` | 6, 0 failed | 3.7 min | first record; provenance said `commit unknown` |
| `exp-obj2-48e70d` | `out/expobj2/` | 6, 0 failed | 2.9 min | + `source_fingerprint`; 17/43 matched -- CRLF vs LF |
| `exp-obj3-e2d551` | `out/expobj3/` | 6, 0 failed | 3.2 min | + content-based match and `files_loaded`; 26/28 |
| `exp-obj4-690a5f` | `out/expobj4/` | 6, 0 failed | 3.3 min | the shipped run, code frozen |

Logs `out/exp-obj{,2,3,4}_cluster.log`. In batch 3 the automatic `--fetch` returned 25 of 26 files
(`stim_s0_retina.npz` missing); it was fetched by `scp` per the process rule.

**CPU (local).** `scripts/interp_export.py run` (the object sweep -> Result -> export),
`static-decompose --target "LC11|LC10a"`, and `analyse` on every other tool's Result; consoles in
`out/interp/export/*_console.txt`, Results in `out/interp/export/{object_sweep,lc11_lc10a_static}.json`.

---

## 3. Validation 1 -- the round-3 object sweep, side by side

`scripts/interp_export.py run` reduces the reference arm (`out/r3obj/ball_off_s{0..4}.json` and
`null_off_s{0..4}.json`, the five round-3 seeds of `docs/audits/object_sweep.md` 8.4) with **exactly the same code**
that reduces the new runs (`_arm_table` -> `common.compare`), so the two tables are comparable row by row.
Statistic: `diff_max_over_cells_mean_mv` (mV).

**The reference arm reproduces `object_sweep.md` 8.4's `off` rows number for number** -- the **12 columns x 6 `off`
types** of that table, 72 numbers (per-seed ball values, mean, SD, per-seed null values, null mean, null SD, z,
Welch, U, p). The export's own `reference_per_type.csv` is a different shape and should not be described as "12
statistics": it holds **10 distinct statistics in 72 rows** (6 spiking types x 6 statistics + 9 rate types x 4), of
which `diff_max_over_cells_mean_mv` is the one compared here:

| type | cells | export mean | 8.4 mean | export SD | 8.4 SD | export null | 8.4 null | export z | 8.4 z | export Welch | 8.4 Welch | U | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC11 | 143 | +0.0658 | +0.066 | 0.0349 | 0.035 | +0.0681 | +0.068 | -0.09 | -0.1 | -0.12 | -0.1 | 12 | 1.00 |
| LC10a | 275 | +0.0813 | +0.081 | 0.0187 | 0.019 | +0.0763 | +0.076 | +0.27 | +0.3 | +0.43 | +0.4 | 14 | 0.84 |
| LC10b | 95 | +0.1286 | +0.129 | 0.0736 | 0.074 | +0.0854 | +0.085 | +0.69 | +0.7 | +1.00 | +1.0 | 17 | 0.42 |
| LC16 | 182 | +0.0962 | +0.096 | 0.0387 | 0.039 | +0.1016 | +0.102 | -0.14 | -0.1 | -0.22 | -0.2 | 11 | 0.84 |
| LPLC2 | 185 | +0.2885 | +0.289 | 0.0140 | 0.014 | +0.1904 | +0.190 | +2.03 | +2.0 | +4.35 | +4.4 | 25 | 0.0079 |
| LC4 | 126 | +0.0857 | +0.086 | 0.0506 | 0.051 | +0.0891 | +0.089 | -0.07 | -0.1 | -0.10 | -0.1 | 10 | 0.69 |

The `verdict` column reads `null` for every row including LPLC2 (p 0.0079 but |z| 2.03 < 3), which is what
`object_sweep.md` 8.5 concluded for `off` ("at z > 3 ... LPLC2 under `sign`/`abs` and nothing else").

**The new runs** (batch 4, `out/expobj4/`, 3 runs per arm, the same protocol and mode, run directory
`out/export/export-20260912T235822Z-ef267800/`):

| type | cells | runs | mean | SD | null mean | null SD | z | Welch | U | p | verdict | reference mean (5 runs) | reference z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC11 | 143 | 3 | +0.0561 | 0.0243 | +0.0452 | 0.0370 | +0.30 | +0.43 | 5 | 1.00 | null | +0.0658 | -0.09 |
| LC10a | 275 | 3 | +0.0602 | 0.0096 | +0.0573 | 0.0121 | +0.25 | +0.33 | 5 | 1.00 | null | +0.0813 | +0.27 |
| LC10b | 95 | 3 | +0.0842 | 0.0459 | +0.1353 | 0.0679 | -0.75 | -1.08 | 2 | 0.40 | null | +0.1286 | +0.69 |
| LC16 | 182 | 3 | +0.0861 | 0.0134 | +0.0809 | 0.0208 | +0.25 | +0.36 | 5 | 1.00 | null | +0.0962 | -0.14 |
| LPLC2 | 185 | 3 | +0.3032 | 0.0391 | +0.1775 | 0.0329 | +3.82 (unstable, see below) | +4.26 | 9 | 0.10 | null | +0.2885 | +2.03 |
| LC4 | 126 | 3 | +0.0740 | 0.0319 | +0.0705 | 0.0174 | +0.20 | +0.17 | 5 | 1.00 | null | +0.0857 | -0.07 |

**Reading.** Every new arm mean lands inside the reference arm's scatter (LC11 +0.056 vs +0.066 +- 0.035; LC10a
+0.060 vs +0.081 +- 0.019; LPLC2 +0.303 vs +0.289 +- 0.014), and every type except LPLC2 sits on its null, exactly
as round 3 found: the object localization of `object_sweep.md` is unchanged by this export. LPLC2 is the one type
above the null in every arm -- but **quote the difference, not the z**. Over three independent submissions of the
identical protocol the arm mean is stable to 5 % (**+0.2885** reference, `out/r3obj/` 5 seeds / **+0.3032** batch 4,
`out/expobj4/` / **+0.2974** a third three-seed submission, `out/skexp/`) and so is the null **mean** (+0.1904 /
+0.1775 / +0.1950), but the null **SD** swings by 2.6x (0.0484 / 0.0329 / 0.0862), so z reads **+2.03 / +3.82 /
+1.19** on the same data-generating process. The reproducible quantity is the excess over the null: **+0.098 /
+0.126 / +0.102 mV**, above the null in 3 of 3 submissions. A z against a 3-run null SD is not a stable statistic and
is not treated as one here. The verdict reads `null` in every case because
`common.compare` also requires `p <= 0.05`, which three runs per arm cannot reach (section 10.3). Run-to-run
scatter on the same protocol is again of order the effect (LC10b's arm moved from +0.158 in batch 1 to +0.084 in
batch 4, its null from +0.080 to +0.135), which is the reason `runs` and not `seeds` is the replicate unit.

---

## 4. Validation 2 -- the per-body readouts Neurome asked for

`readout_per_body.csv`, 26,482 rows over 13,241 bodies and 15 types, window 3.0-15.0 s, `control_ids` naming
the three matched null runs:

| unit_kind | quantity | unit | rows | bodies |
|---|---|---|---|---|
| spiking | `upstream_drive_mV` | mV | 1,006 | 1,006 |
| spiking | `output_Hz` | Hz | 1,006 | 1,006 |
| graded | `rate_deviation` | rate units [0-1] | 12,235 | 12,235 |
| graded | `abs_rate_deviation` | rate units [0-1] | 12,235 | 12,235 |

Per type: **LC11 143** and **LC10a 275** bodies, each with *both* `upstream_drive_mV` and `output_Hz` and never
pooled (the rule `verify` enforces), plus LC10b 95, LC16 182, LC4 126, LPLC2 185 spiking, and the graded medulla
types Mi1 1,773 / Mi4 1,772 / T3 1,940 / T2 1,630 / Tm3 2,054 / Tm5Y 898 / TmY5a 1,364 / TmY13 432 / TmY21 372.
Every `bodyId` is a decimal string present in `cache/neurons.parquet` (167,106 ids; `verify` checks all of
`readout_per_body`, `contributions` and `retina_bodies`).

Each row carries `stimulus_value`, `control_value`, `stimulus_minus_control`, `n_trials`, `trial_sd`,
`stimulus_sd` / `control_sd`, and -- because the null arm was recorded per body as well -- `null_mean`, `null_sd`,
`z_vs_null` and a `verdict`.

**Caveat on the per-body `verdict`.** It is `|z_vs_null| >= 3` with the null SD estimated from three runs and no
exact test (`common.compare`'s U test is a per-arm statistic, not a per-cell one). 1,568 of 26,482 rows read
`result` on that rule (5.9 %; it was 2,937 = 11.1 % in batch 2 on the same protocol). With 13,241 bodies x 2
quantities and a null SD from three runs, that count is what a heavy-tailed z on n = 3 produces, not evidence for
~1,500 responsive cells: **the per-type arms of section 3 are the verdict of record**, and the per-body columns are
data for Neurome to re-reduce, which is why `n_trials`, `trial_sd` and `null_sd` travel with every row.

**Defect in the interchange table, unfixed: 42 of those `result` rows are a division by zero.** In
`out/export/export-20260912T235822Z-ef267800/readout_per_body.csv`, **739 of the 26,482 rows have `null_sd` exactly
0** -- three null runs returning the identical value on sparse spike counts, so `trial_sd` is 0.
`flyverse/interp/export.py:753-755` computes `z = (stimulus_minus_control - null_mean) / null_sd` under
`np.errstate(divide="ignore")` and then `np.where(np.abs(z) >= common.Z_RESULT, "result", "null")`, so a non-zero
numerator over a zero denominator gives `+-inf`, and `inf >= 3` is True. Those 42 rows therefore ship with
`verdict = result` and an **empty** `z_vs_null` column (e.g. LC11 body 17476, `output_Hz`,
`stimulus_minus_control -0.027778`, `null_mean 0.0`, `null_sd 0.0`, `z_vs_null` blank, `verdict result`). The other
697 zero-SD rows have a zero numerator too, so `0/0` is NaN and they fall to `null`. A verdict must not be reachable
without a finite z; the fix belongs in `export.py` and is not applied in this round.

---

## 5. Validation 3 -- the retinal sampling actually presented

`--retina` replays the presented geometry frame by frame at the pinned pose (the ray tracer is deterministic, so the
replay is the radiance the optic lobe received) and writes three tables:

* `retina_columns.csv` -- **1,466** hex columns: side, hex coordinates, azimuth / elevation, unit direction, the
  number of photoreceptors and the `|`-joined list of their bodyIds. Every column has at least one; the lists cover
  all 5,895 photoreceptors.
* `retina_bodies.csv` -- **5,895** photoreceptors: bodyId, model index, type, `unit_kind = photoreceptor`, column,
  and the four spectral sensitivities. Types: R1-R6 3,344, R8y 481, R7y 469, R8_unclear 436, R7_unclear 350,
  R8p 330, R7p 296, R7d 80, R8d 75, R7R8_unclear 34.
* `retina_radiance.parquet` -- **1,759,200** rows = 1,200 frames x 1,466 columns, `[UV, B, G, R]` plus the ball
  offset of that frame (Parquet because it is above `parquet_rows = 1e6`).

**It is the presented stimulus, not a re-render.** `scripts/probe_object_sweep.py` writes its own `radiance_check`
at four frames (how many columns the ball changes by > 5 %, how many it darkens by > 50 %, the minimum relative
radiance). Recomputing those statistics from the exported table reproduces the probe's numbers exactly, and the
darkest column's azimuth tracks the ball:

| t | ball offset | probe: changed / darkened / min rel | export table: changed / darkened / min rel | darkest column azimuth |
|---|---|---|---|---|
| 0.00 s | +0.060 m | 5 / 2 / -0.9803 | 5 / 2 / -0.9803 | +51.4 deg |
| 0.75 s | +0.030 m | 9 / 4 / -0.9879 | 9 / 4 / -0.9879 | +31.5 deg |
| 1.50 s | +0.000 m | 2 / 2 / -0.9638 | 2 / 2 / -0.9638 | +2.4 deg |
| 2.25 s | -0.030 m | 8 / 2 / -0.9833 | 8 / 2 / -0.9833 | -33.4 deg |

Over the whole 12 s sweep 45 of 1,466 columns dim by more than 5 %, spanning azimuth -53.3 to +55.4 deg and
elevation -3.0 to +10.0 deg (23 right, 22 left) -- an 11.4 deg ball crossing the frontal field.

**The stimulus side is deterministic even though the spiking side is not.** The radiance array is byte-identical --
`sha256(radiance) = ed07d8eb525541c0...` for all seven `--retina` jobs of the four batches, across both seeds and
four independent submissions -- while the per-type z of the same protocol moves by more than 1 between runs
(section 3). The export therefore ships one retina record per run
directory and treats `runs`, not seeds, as the replicate unit -- the rule of `docs/audits/object_sweep.md` 8.
This also means the `retina_*` tables are a clean, reusable description of the stimulus for Neurome's size-tuning
ladder (`docs/NEUROME_INTERFACE.md` section 3).

---

## 6. Validation 4 -- the structural decomposition, and Neurome's edge counts

`scripts/interp_export.py static-decompose --target "LC11|LC10a"` builds the `contributions` path end to end on the
CPU: `common.effective_weights` (the shaped weights that reproduce `Brain._W_cpu`) -> `common.links` ->
`links_to_contributions`. Run `out/export/decompose-20260912T233843Z-c840196b/`: 418 post cells, 30,160 presynaptic cells, **149,696 contribution rows**
covering 595,389 raw synapses, 1,616 sign-0 entries (2,119 synapses), total +89,770 / -62,579 mV.

`synaptic_pair_count` is the raw, unsigned, uncapped count Neurome joins on, and it **agrees with Neurome's own
exported input distributions to the synapse** for all sixteen edges listed in `docs/NEUROME_INTERFACE.md`:

| post <- pre | export | Neurome | mV per post cell per presynaptic volley | share of raw input | entries |
|---|---|---|---|---|---|
| LC11 <- T3 | 69,205 | 69,205 | +133.09 | 0.2012 | 9,177 |
| LC11 <- T2 | 36,917 | 36,917 | +70.99 | 0.1073 | 10,641 |
| LC11 <- Tm6 | 24,736 | 24,736 | +47.57 | 0.0719 | 6,075 |
| LC11 <- T2a | 24,572 | 24,572 | +47.25 | 0.0715 | 6,662 |
| LC11 <- Tm12 | 19,721 | 19,721 | +37.93 | 0.0573 | 4,080 |
| LC11 <- TmY18 | 15,593 | 15,593 | +29.99 | 0.0453 | 4,558 |
| LC11 <- LC11 | 15,491 | 15,491 | +2.98 | 0.0450 | 6,967 |
| LC11 <- Li15 | 13,918 | 13,918 | -26.76 | 0.0405 | 1,249 |
| LC10a <- LC10a | 25,244 | 25,244 | +2.52 | 0.1004 | 6,189 |
| LC10a <- TuTuA_2 | 18,097 | 18,097 | -16.40 | 0.0720 | 463 |
| LC10a <- AOTU042 | 14,492 | 14,492 | -13.66 | 0.0576 | 373 |
| LC10a <- Tm5Y | 14,221 | 14,221 | +14.22 | 0.0565 | 2,706 |
| LC10a <- LC9 | 9,661 | 9,661 | +9.66 | 0.0384 | 1,173 |
| LC10a <- LC10c-1 | 8,823 | 8,823 | +8.82 | 0.0351 | 2,098 |
| LC10a <- TmY21 | 7,406 | 7,406 | +7.41 | 0.0294 | 1,641 |
| LC10a <- LC10c-2 | 6,981 | 6,981 | +6.98 | 0.0278 | 1,938 |

The two self-input rows show why the raw count and the effective weight must both travel: LC11 <- LC11 is 15,491
synapses but only +2.98 mV per post cell per volley, and LC10a <- LC10a 25,244 synapses for +2.52 mV -- the
same-type gain (0.1) and the 60-synapse cap, both named in every row's `gain_rule`.

---

## 7. Validation 5 -- every tool's Result goes through the same serializer

`docs/NEUROME_INTERFACE.md` requires that every tool's JSON be exportable. Running `export` over one Result from
each of the eight tools (`scripts/interp_export.py analyse --result ...`), all eight pass `Result.check()` and write
a manifest with matching hashes. **Seven come back clean from `verify`; the atlas does not** (bodyIds checked
against `cache/neurons.parquet`):

| tool | Result | tables written | verify |
|---|---|---|---|
| decompose | `out/interp/decompose/validate_taste.json` | contributions 4,651 + 9 tool tables | no problems |
| trace | `out/interp/trace/object_best_cell.json` | readout_per_body 13,396 + 10 tool tables | no problems |
| paths | `out/interp/paths/validate_rot_pen.json` | contributions 4,003 + paths 83, links 91, b_inputs 106 | no problems |
| lesion | `out/interp/lesion/holds_cpu.json` | sensitivity 21 + matrix 24, dissociations 6, lesions 24 | no problems |
| atlas | `out/interp/atlas/validation.json` | readout_per_body 2,128 + atlas 660, movers 396 | **EXIT=1** -- `readout_per_body: no rows for LC11`, `readout_per_body: no rows for LC10a` |
| health | `out/interp/health/validation.json` | 10 tool tables, 414 rows | no problems |
| ledger | `out/interp/ledger/validate.json` | ledger 111, sources 2, observations 66, validation 9 | no problems |
| export | `out/interp/export/object_sweep.json` | readout_per_body 26,482 + retina 1,766,561 + 144 | no problems |

**The atlas row is the tool's own default biting an unrelated tool.** `verify`'s signature is
`verify(run_dir, *, neurons=None, expect_paired=("LC11","LC10a"), expect_counts=None)`
(`flyverse/interp/export.py:415`), and both the CLI (`scripts/interp_export.py:417`, `--paired` default
`"LC11,LC10a"`) and `_report` (`scripts/interp_export.py:316`, the function `analyse` calls) use it
**unconditionally, for every tool's export**. The rule only fires on exports that carry a `readout_per_body` table
(`export.py:469`), which is why decompose / paths / lesion / health / ledger sail past it; trace carries LC11 143 /
LC10a 275 and passes on the merits. The atlas `readout_per_body` holds 2,128 rows over 98 types, **none of them LC11
or LC10a**, so `interp_export.py verify --run-dir out/export/atlas-20260912T233906Z-f8234031` exits 1 with the two
problems above (`export.py:475`, `cmd_verify` returns `1 if info["problems"]`). Today the only escape is
`verify --paired ''` (`scripts/interp_export.py:346` drops empty entries); `analyse` has no such flag. **The fix the
record asks for: a documented `--paired` option on `analyse`, or an `expect_paired` rule that fires only when the
readout actually contains those types.** The object-pathway expectation is not a property of every tool's Result.

The trace export independently satisfies the paired-quantity rule (LC11 143 / LC10a 275 bodies with both
quantities), i.e. two different tools produce the shape Neurome asked for.

**Provenance of this table.** The eight rows above are quoted from `out/interp/export/exportability.csv`, and **that
file's generator is not in the repo** -- `grep -rn exportability --include=*.py --include=*.sh --include=*.md .`
returns nothing, which breaks the round-1 process rule "ship every generator of every quoted number into `scripts/`
or `flyverse/interp/`." It was not produced off the shipped code path: six of the run directories it names --
`atlas-20260912T233906Z-f8234031`, `health-20260912T233347Z-527eabd9`, `ledger-20260912T233657Z-d1762311`,
`lesion-20260912T233535Z-10b172fd`, `paths-20260912T233911Z-4047cc00`, `decompose-20260912T233259Z-174a5418` --
contain **no `checks.json`**, so no round trip and no `verify` report was ever written into them, and its `export`
row names `export-20260912T233810Z-bf35208f`, which does not exist under `out/export/` at all. Its `problems: none`
for those rows is not reproducible, and the atlas one is wrong.

---

## 8. What the manifest alone lets a skeptic reconstruct

From `out/export/<run_id>/manifest.json` of the shipped object-sweep run, with no other file:

* **commit** -- `0d32fd6e74067569317d1a4bd604ce2f82f69dd5`, `dirty: true`, with `commit_verified` saying how
  it is known (see below).
* **dataset** -- `male-cns v1.0 flat-connectome`, the four MaleCNS files with SHA-256 (`body-annotations`
  `2177e246...`, `body-neurotransmitters` `95c92892...`, `connectome-weights` `e35da783...`, `tbar-neurotransmitters`
  `bade84c9...`).
* **compiled connectome** -- md5 of `W_post_pre.npz` data / indices / indptr (`e015d9d4...` / `d898bcfb...` /
  `718a6a97...`, combined `ef23cc27...`), `sum|W|` 121,460,584, nnz 25,578,600, 167,106 neurons, the NT counts, and
  the `TYPE_NT_OVERRIDE` (TmY14 glutamate, Mi19 serotonin, aMe8 ACh) and `UNKNOWN_NT_OVERRIDE_REGEX` in force.
* **model** -- every resolved `LIFParams` field (`w_syn` 0.275, `conn_cap` 60, `t_ref` 2.2 ms, `input_norm`
  (5000, alpha 1.0), `receptor_model = None` under `--receptor-model off`, `path_gain`
  `[descending_neuron -> vnc_ 3.0, visual_projection -> descending_neuron 2.0]`, `type_path_gain`
  `[(LC4|LPLC2) -> DNp01 3.0]`), the receptor table path and md5, every `OpticParams` field (`gain_out_mv` 100,
  `gain_fb` 0.5, `out_norm` l1, `drive_clip_mv` 35, the pair gains), and the body thresholds
  (`gf_hz` 33, `takeoff_power_hz` 50, `takeoff_hold_s` 0.3, `mdn_threshold_hz` 15, `k_opto` 0).
* **execution** -- the **realised** device `cuda` / NVIDIA B200 (not the request, which was `None`), backend flags
  (native kernels, event-driven, `cuda_sparse warp`, no CUDA graphs), dt (LIF 0.5 ms, optic 1 ms, frame 10 ms),
  seeds, batch 1, `replicate_unit = runs`, host, platform, torch 2.11.0+cu128.
* **stimulus** -- protocol `object_sweep`, the generator, every geometry and timing parameter (11.42 deg ball,
  0.005 m at 0.05 m, half sweep 0.06 m, sweep 3 s, 12 s window after 3 s settle, pose and heading, the fruit /
  fence / wind state, the contrast), the matched control, the six run files, and the ten reference files with their
  SHA-256.
* **retina**, **units** (the six unit-handling facts of NEUROME_INTERFACE section 2), **populations**,
  **replicates**, **conventions** (the NA rule, decimal ids, `|`-joined lists, the table roles), **tables** (row
  counts, columns, units, SHA-256 each) and the source Result's SHA-256.

### The one hole, and how far it is closed (not all the way)

`common.provenance` takes the commit from `common.git_state()`. A cluster job runs from
`/mnt/beegfs/neurome/runs/<run>/` -- `scripts/cluster_run.py` rsyncs the cluster's own checkout **without `.git`** and
overlays the locally-changed files -- so `git rev-parse` fails there and the block says `commit: unknown`. Four of
the eight tools' Results in section 7 (every GPU-recorded one: atlas, health, trace, export) carry
`commit: unknown`.

`record` therefore hashes the source itself: `source_fingerprint(include_loaded=True)` records the SHA-256 of every
module the process actually imported from the repo (28 files for this protocol: `flyverse/brain.py`,
`connectome.py`, `optic.py`, `retina.py`, `screen.py`, ..., `scripts/probe_object_sweep.py`,
`scripts/interp_export.py`, `flyverse/interp/{common,export}.py`) plus a glob of the tree, and the CPU analysis step
calls `match_sources`, which compares them against this checkout and -- when every one matches -- writes the local
commit into the manifest with `commit_verified` saying **how** it is known. Two details had to be right:

* the match is on **content**, not bytes: the cluster's checkout is LF and a Windows working tree is CRLF, and only
  files differing from `origin/main` are shipped, so identical source hashes differently on the two sides. Batch 2
  matched 17 of 43 files for this reason alone; with LF/CRLF-tolerant matching the same recording matched 40 of 43.
* the verdict is taken on the **imported** set, not a glob: six implementers were editing `flyverse/interp/*.py`
  while these jobs ran, and a module the job never imported cannot have changed its numbers. Batch 3 matched 26 of
  28 imported files, the two differing ones being `export.py` / `interp_export.py`, which this task edited after
  submitting; batch 4 ran with the code frozen and matched **28 of 28**, so the shipped manifest reads
  `commit: 0d32fd6e...` with `commit_verified: "by source hash (loaded scope): every one of the 28 source files the
  run loaded holds the content of this checkout"`. The glob scope of that same run was 40 of 43 -- the three
  differing files being `flyverse/interp/{atlas,ledger,trace}.py`, none of them imported by the job, which is
  exactly the noise the imported-set scope removes.

A skeptic who distrusts the inference still has the raw evidence: `flyverse_commit.source_fingerprint.files` and
`files_loaded` list the per-file SHA-256 the job saw, so any single file can be checked against any commit.

**The hole is closed for this tool's own recordings and nowhere else.** `_stamp_commit` can pin a commit only when
the recording carried a `source_fingerprint`, and only `export`'s own `record` writes one, so three of the four
GPU-recorded tools still ship an unknown commit: `out/export/atlas-20260912T233906Z-f8234031/manifest.json`,
`.../health-20260912T233347Z-527eabd9/manifest.json` and `.../trace-20260912T233909Z-94902d11/manifest.json` all read
`flyverse_commit.commit = "unknown"` with `commit_verified = null`. Worse, `verify()` returns **no problems** for the
health and trace directories anyway: `docs/NEUROME_INTERFACE.md` section 1 makes `flyverse_commit` mandatory, but
`Result.check()` and `verify()` test only that the **key exists**, never that it names a commit. So "the commit hole
is closed" would be false as a general claim -- it is closed for the object-sweep export in this record, and section
10.4 is the change that would close it for the other seven tools.

---

## 9. Defects found and fixed in this task (in the files this task owns)

1. **`--stim "out/expobj/stim_s*.json"` also matched the `_prov.json` sidecars** that `record` writes beside each
   probe JSON. The arm values were still right (a sidecar has no `ball` block, so `_arm_values` skipped it) but the
   manifest recorded `n_stim_runs: 6` for three runs and polluted `control_ids` with sidecar stems. `_glob` now
   drops `SIDECARS` and returns forward-slashed paths.
2. **`verdict = 'null'` was silently destroyed by the CSV round trip.** `null` is in pandas' default NA list, so
   `pd.read_csv` turns `common.compare`'s three-valued answer into a missing value -- 24,914 of the 26,482
   `readout_per_body` rows of the shipped export. `read_table` now reads with `keep_default_na=False, na_values=[""]`, the
   manifest declares the convention under `conventions`, and `round_trip_check` gained a **text-column** comparison
   (it previously compared numeric columns only, which is exactly why the corruption was invisible).
3. **`common.raw_counts` returns zero for every ordinary edge** (section 10); `export.raw_counts` overrides it, and
   the sixteen Neurome edge counts of section 6 are the check that the override is right.
4. **`common.silent_flags(..., rates=None)` marks every link `never_firing`** (section 10); `export.silent_flags`
   leaves the flag `False` when no rollout was given and the table says so in `silent_rule`.
5. **The commit hole of section 8**, closed with `source_fingerprint` / `match_sources` **for this tool's own
   `record` path only** -- the atlas, health and trace exports still ship `commit: unknown` with
   `commit_verified: null`, and `verify()` does not flag it (section 8, section 10.4).

## 10. Findings for other owners (not fixed here -- `flyverse/interp/common.py` is not this task's file)

1. **`common.raw_counts(c)` (the default `with_sign0=True`) replaces the whole count vector with
   `connectome.sign0_counts`,** which is "non-zero only where `W.data == 0`" (`flyverse/connectome.py:308`). On the
   shipped cache it returns 2,701,289 synapses over 916,626 non-zero entries instead of 124,161,872 over all
   25,578,600 -- **every ordinary edge reads 0**. LC11's input comes out as 963 synapses instead of 343,904.
   `common.links` calls it whenever `counts` is not passed, so any tool that takes the default writes
   `synaptic_pair_count = 0` for real edges. The fix is one line -- `C.data = np.maximum(np.abs(c.W.data), cnt)`,
   which is what `export.raw_counts` does.
2. **`common.silent_flags(..., rates=None)` returns `never_firing = NaN`, and `common.links` builds its `silent`
   string with `bool(flag) is True`** -- and `bool(float('nan'))` is `True`. Every entry of a structural table
   therefore reads `never_firing` when no rollout was supplied. (The paths tool hit the same thing: its
   `test_sign0_loop_and_never_firing` expected `sign0` and got `sign0|never_firing`.) Either return `False` /
   `None`, or have `links` test `flag is True`.
3. **`common.compare` cannot return `result` at three runs per arm.** The exact two-sided Mann-Whitney p at
   n = 3, 3 is 0.1 at best, and the verdict requires `p <= alpha = 0.05` as well as `|z| >= 3`; so an arm separated
   by |z| = 99 still reads `null`. `MIN_REPLICATES = 3` and the verdict rule are therefore inconsistent: three runs
   is the floor for *not* being `underpowered` and simultaneously a guaranteed `null`. Either `alpha` should be
   applied only when the exact test can reach it (n >= 4, 4 gives p_min 0.029), or the documented minimum for a
   `result` should be four runs per arm. This is why every row of section 3 reads `null`, including the reference
   arm's LPLC2 at p 0.0079 -- there the block is |z| 2.03 < 3, which is the intended behaviour.
4. **`common.provenance` records no source fingerprint,** so every GPU-recorded Result says `commit: unknown`
   (4 of the 8 tools in section 7). Suggested: call `export.source_fingerprint(include_loaded=True)` from
   `provenance()` whenever `git_state()['commit'] == 'unknown'`; `export.match_sources` then pins it at analysis
   time for every tool, not just this one.

## 11. Caveats and open questions

* **Graded units carry no received drive in mV.** `docs/NEUROME_INTERFACE.md` section 2 promises graded units
  "with rate [0-1] and received drive in mV". `scripts/probe_object_sweep.py` records only the optic rate deviation
  for its rate types, so the export ships `rate_deviation` / `abs_rate_deviation` in rate units and no mV column for
  the 12,235 graded bodies. Adding it needs a change in the probe or the optic lobe's per-cell input accessor, not
  in the export.
* **`verify`'s `expect_paired` default is object-pathway-specific and hard-wired for every tool.**
  `expect_paired=("LC11","LC10a")` (`flyverse/interp/export.py:415`) is the default of the function, of the CLI
  (`scripts/interp_export.py:417`) and of `_report` (`:316`), which does not pass one through -- so there is no CLI
  route to export another tool's Result through `analyse` without the two spurious problems, and the atlas export
  fails `verify` for that reason alone (section 7). Either the rule should fire only when `readout_per_body`
  actually contains those types, or `analyse` needs a `--paired` flag; `verify --paired ''` is the only escape
  today, and it is not documented anywhere but here.
* **The export never re-derives anything.** A Result with a wrong number exports cleanly; `check()` and `verify`
  test the *shape*, the provenance block and the hashes, never the physiology. The ledger tool is where a number is
  judged against an expectation.
* **`result.json` is copied into the run directory verbatim**, so a large Result doubles the directory (the static
  decomposition is 100 MB of JSON plus a 48 MB CSV). It is what makes the run self-contained and hashable; a future
  version could store only the tables plus a pointer when the Result is above a size.
* **`checks.json` is written after `manifest.json`** and is therefore not one of the hashed tables. It reports the
  round trip and `verify`; it is not part of the interchange.
* **Three runs per arm** is the floor of the scatter rule, and section 10.3 shows it cannot produce a `result`
  verdict. Neurome's size-tuning ladder (`NEUROME_INTERFACE` section 3, the `apply:object` task) should budget at
  least four, preferably five, runs per size and per matched null.

## 12. Reproducing this

```bash
# GPU: the object-sweep protocol with the per-body capture and the retinal sampling (one batch, 6 jobs)
python scripts/cluster_run.py --name exp-obj4 --minutes 30 \
  "mkdir -p out/expobj4 && python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_export.py record --receptor-model off --receptor-net-rule class --seed 0 --retina --out out/expobj4/stim_s0 > out/expobj4/stim_s0.txt; cat out/expobj4/stim_s0.txt" \
  ... (seeds 1, 2; and the same three with --null) ... --fetch out/expobj4/

# CPU: the recorded runs -> one Result -> the export, with the round-3 arm side by side
PYTHONIOENCODING=utf-8 python scripts/interp_export.py run --stim "out/expobj4/stim_s*.json" \
  --null-runs "out/expobj4/null_s*.json" --reference "out/r3obj/ball_off_s*.json" \
  --reference-null "out/r3obj/null_off_s*.json" --retina out/expobj4/stim_s0_retina.npz \
  --json out/interp/export/object_sweep.json --out out/export --expect-counts '{"LC11": 143, "LC10a": 275}'

# CPU: the structural decomposition of LC11 / LC10a through the contributions table
PYTHONIOENCODING=utf-8 python scripts/interp_export.py static-decompose --target "LC11|LC10a" \
  --json out/interp/export/lc11_lc10a_static.json --out out/export

# CPU: any other tool's Result, and a re-check of a finished directory
PYTHONIOENCODING=utf-8 python scripts/interp_export.py analyse --result out/interp/<tool>/<run>.json --out out/export
PYTHONIOENCODING=utf-8 python scripts/interp_export.py verify --run-dir out/export/<run_id>
```

## 13. Tests

`tests/test_interp.py::ExportTests`, 8 CPU tests on the 8-neuron synthetic graph of `graph()`, no dataset and no GPU
(`python -m pytest tests/test_interp.py -k "AtlasTests or ExportTests" -q` -- **14 passed**, of which
`ExportTests` is **8 passed in 3.3 s**. No whole-file total is quoted: `tests/test_interp.py` is shared and its count
moves under other owners. `tests/test_control.py` 18 passed; `import flyverse.interp` still pulls in no torch):

| test | what it pins |
|---|---|
| `test_manifest_tables_and_round_trip` | every mandatory manifest field, the interchange key, the realised device, the cache fingerprint, `role`, per-table SHA-256, the `result.json` hash, decimal ids, column order, units, the numeric **and text** round trip, `verify` clean with `expect_counts`, and a hash mismatch after an edit |
| `test_refusals_and_the_paired_quantity_rule` | a Result without provenance is refused; a body missing `output_Hz`, a missing type and an id outside the connectome are all reported |
| `test_retina_tables` | the column -> photoreceptor-body map, the long radiance form, the manifest's retina block, the Parquet switch |
| `test_raw_counts_keep_the_uncapped_synapse_count` | the raw count of an ordinary edge (120, above the cap) and of a sign-0 edge (200, from `sign0_counts`) |
| `test_static_decompose_result_and_its_export` | the `contributions` columns, the sign-0 row (value 0, 200 synapses, `silent = sign0`, no `never_firing`), the gain rule, `share_of_raw_input`, and the export of it |
| `test_source_fingerprint_names_the_commit_of_a_cluster_run` | the hashes, a changed and a missing file, LF/CRLF tolerance, and the imported-set scope beating a glob |
| `test_object_sweep_result_arms_and_readouts` | the arms, the p floor at 3 v 3, the reference arm side by side with its SHA-256, two rows per spiking body, one per graded unit, the units and the window |
| `test_cli_analyse_and_verify` | `analyse` / `verify` / the bare default form, `checks.json`, `--control-ids` |
