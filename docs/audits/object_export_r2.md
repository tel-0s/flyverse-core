# The round-2 matched-baseline export for Neurome

Round 2, task **export** (2026-09-13). Owns `scripts/object_round2_export.py`, this file, and the run directories it
writes under `out/export/`. It answers `D:\Projects\neurome\reports\flyverse-size-tuning-intake.md` and
`D:\Projects\neurome\docs\flyverse-size-tuning-reply.md` by shipping the MATCHED sphere ladder
(`scripts/probe_object_matched.py`, batch `out/objr2/`, predeclared analysis `scripts/object_round2_baseline.py`)
through the revised exporter `flyverse/interp/export.py` (`docs/audits/interp_export.md` section 14, schema
`flyverse.neurome.export/2`, revision 2).

**This module computes no comparison of its own.** Every arm, z, U, p, verdict and Holm value in the exported tables
came from the predeclared analysis (`out/interp/objr2/baseline.json`) and is copied through; what the export adds is
the interchange shape -- the per-body rows, the two references named apart, the retina tables with the in-loop
radiance of both arms, the statistic definitions, the hashes -- plus the two rank-test columns the export contract
asks for (`p_mannwhitney` / `p_method`) beside the predeclared test.

**The location to quote (for `docs/NEUROME_INTERFACE.md`, which this task does not edit):**

> Round-2 matched sphere ladder, `out/export/objr2-<lobe>-d<rung>-<stamp>/` (12 rung directories: `ship` and `fb0`
> x 4.5 / 8.8 / 11 / 15 / 20 / 30 deg) plus the ladder summary `out/export/objr2-ladder-20260913T232912Z-72041020/`.
> The index of all thirteen, with each one's tables, round trip and verify report, is `out/export/objr2_index.json`;
> the flat table list with every SHA-256 is `out/export/objr2_tables.csv`. Export schema
> `flyverse.neurome.export/2`, revision 2.

---

## 1. The run directories

**Thirteen run directories under `out/export/`, 1.2 GB, `verify` problems `none` in every one.** The
machine-readable index (with each directory's table list, round trip and verify report) is
`out/export/objr2_index.json`; the flat table list with every SHA-256 is `out/export/objr2_tables.csv` /
`.json`; the console is `out/export/objr2_export_console.txt`.

| rung | lobe `ship` (the shipped OpticParams) | lobe `fb0` (`gain_fb = 0`, the deterministic control) |
|---|---|---|
| 4.5 deg | `objr2-ship-d045-20260913T232754Z-b7230e82` (101 MB) | `objr2-fb0-d045-20260913T232636Z-9cdc7071` (95 MB) |
| 8.8 deg | `objr2-ship-d088-20260913T232807Z-39bdadb1` (97 MB) | `objr2-fb0-d088-20260913T232649Z-ee874c6b` (91 MB) |
| 11 deg | `objr2-ship-d110-20260913T232820Z-d93246de` (96 MB) | `objr2-fb0-d110-20260913T232702Z-4e396ef6` (89 MB) |
| 15 deg | `objr2-ship-d150-20260913T232833Z-63b3ddad` (95 MB) | `objr2-fb0-d150-20260913T232715Z-fb73b6c9` (89 MB) |
| 20 deg | `objr2-ship-d200-20260913T232847Z-835c32f6` (98 MB) | `objr2-fb0-d200-20260913T232728Z-04ec7c69` (92 MB) |
| 30 deg | `objr2-ship-d300-20260913T232900Z-80d7e5d1` (98 MB) | `objr2-fb0-d300-20260913T232741Z-a0815234` (92 MB) |
| **ladder summary** | `objr2-ladder-20260913T232912Z-72041020` (22 MB) -- both lobes, all six rungs, the predeclared family's Holm columns, the old ladder beside it | |

Each rung directory holds **six object runs against six independent blank/blank runs of the same lobe**, all from
the one submission `objr2-6915ce`; `paired_control_ids` names the six recordings whose arm b gave `control_value`
and `null_reference_ids` the six blank/blank runs, and the two sets are disjoint in every directory.

**Provenance, as delivered.** Every manifest reads `commit c78cb93d6d84283f7df248fcf4d97d44e3346f88` with
`commit_verified: "by source hash (loaded scope): every one of the 29 source files the run loaded holds the content
of this checkout"` -- and the glob scope agrees too (**43 of 43** files identical, no differences). A cluster job has
no `.git`, so its own `flyverse_commit.commit` is `unknown`; this is `export.match_sources` against the recorded
`source_fingerprint`. (The working tree is `dirty: true` -- other tasks' uncommitted edits -- which is why the
per-file evidence, not the commit name alone, is the thing to check.) Realised device: `cuda`, NVIDIA H200 (seeds 0,
1, 2, 5) and NVIDIA B200 (seeds 3, 4), torch 2.11.0+cu128; **each seed's 14 runs -- both lobes, every rung and the
null -- ran on one box**, so an object arm and the null it is compared with never straddle two GPUs
(`out/objr2/verify_sph.json`). Model: the shipped `OpticParams` (`gain_fb` 0.5 on the `ship` lobe, `gain_out_mv` 100,
`out_norm` l1, `drive_clip_mv` 35) with every round-2 stream hook **off** (`stream_rectify`, `stream_adapt`,
`spatial_suppress`, `fb_hold` all null).

## 2. Which of Neurome's points each field answers

| the intake's point | where it is in this export |
|---|---|
| "the old ladder confounded size with retinal position and with angular speed" | the ladder is the MATCHED arc: `retina_object_track` carries azimuth / centre elevation / angular diameter / distance and angular speed **per frame** as the probe measured them in the loop, and `manifest.stimulus.matched` states what is held constant. Section 5 quotes the realised bands: elevation 0.00 deg and diameter constant along the sweep to 1.4e-14 deg at every rung, angular speed 40.000 deg/s. The old ladder travels beside it in the summary's `old_ladder_size_tuning` / `old_ladder_footprint` / `old_ladder_geometry` tables |
| "the retina table was a geometry REPLAY ... do not silently relabel replayed input as recorded input" | `manifest.retina.mode = "in_loop_capture"` with the probe's own sentence in `sampling` / `sampling_recorded` (`sim.col_rad` after every `sim.step()` of BOTH arms) and `in_loop: true`. Nothing in this export is a replay |
| "the blank radiance was omitted" | `retina_radiance_blank` in the identical schema to `retina_radiance`, from the same in-loop capture, so object-minus-blank is a join on `(frame, column_id)` inside the run directory |
| "the exported `control_value` is arm b while `control_ids` names the blank/blank runs" | every `readout_per_body` row carries `paired_control_ids` (arm b of the same recording, `#arm_b` suffix), `null_reference_ids` (the independent blank/blank runs) and the deprecated `control_ids` alias equal to the second; `export.verify` enforces the rule and reports `problems: none` |
| "the headline drive values are maxima of time-mean differences, not membrane voltages" | `manifest.statistic_definitions` defines every `quantity` and `statistic` the tables use, each saying what it is; the per-type rows also carry the definition in a `statistic_definition` column. The primary statistic of this round is a population MEDIAN over bodies, not a maximum over cells -- the old headline statistic is still exported for continuity, flagged exploratory |
| "the LC cells are not literally silent" | nothing in the export says silent. The `verdict` column is `common.compare`'s vocabulary (`result` / `null` / `underpowered` / `undetermined`) and the answer this round reports is "no detected size preference at 6 v 6 on the predeclared statistic" (section 6) |
| "LC10a's biological target is 15-30 deg; only LC11 is the small-object concern" | the two populations are never pooled (`export.verify`'s paired-quantity rule, LC11 143 / LC10a 275 bodies with both quantities in every directory) and the per-type expectations are the predeclared ones, copied verbatim into `manifest.summary.predeclared` from `out/objr2/predeclared.json` (LC11: small-object preference, Keles & Frye 2017; LC10a: 15-30 deg, Schretter et al. 2024) |
| "the exact U encounters ties; reproducing it does not validate its use with tied data" | two rank tests travel per row: `p` (the PREDECLARED two-sided tie-aware exact **permutation** U, which enumerates the C(12,6) assignments with average ranks) and `p_mannwhitney` with `p_method` in {`exact`, `asymptotic_tie_corrected`, `none`} (the export's contract test, `export.mann_whitney`). Section 5 shows they agree exactly on every untied row |
| "keep results exploratory unless a predeclared family survives Holm" | two multiplicity corrections travel and neither overwrites the other: `family` / `p_holm` / `p_holm_mannwhitney` are the PREDECLARED primary family (declared before submission, 12 members per LC type), and `analysis_family` / `analysis_p_holm` / `analysis_survives_holm` are the analysis' own families (one per exploratory / secondary set). `role` says `primary` / `exploratory` / `secondary_exploratory` / `control_fb0` per row. No verdict rests on any Holm column |
| "a per-body receptive-field localizer" | `rf_map`, one row per body: the window the predeclared rule gave it (`window_source` = `rf_ship` / `rf_fb0` / `anat` / `whole`) and the localizer fit it came from in each lobe (centre, width, peak, z, `fitted`, the anatomical column) |
| "per-body time courses before population maxima" | `readout_per_body` (per body, per quantity, whole-window AND RF-windowed) and `time_course` (per body, per azimuth bin along the arc, object and null arms) |
| "distinguish `diff_signed_best_cell` from `diff_abs_best_cell_mean`" | both, plus `diff_signed_mean`, ship as separate rows of `size_tuning` / `upstream_per_run` with separate definitions; they are never collapsed |

## 3. What ran, and how the data got here

**The batch** is the one the baseline task submitted: run id `objr2-6915ce`, 16 jobs = 238 processes, on the rented
boxes (`docs/audits/object_baseline_r2.md`). At the forced close of that task its jobs were still queued and its
local `cluster_run.py` client was killed, so `out/objr2_cluster.log` is **empty (0 bytes)** and **the fetch was done
by hand** from the four boxes, per the process rule. There is therefore no "`<n> job(s), 0 failed`" line for this
batch, and nothing here claims one: the batch check that replaced it is
`python scripts/probe_object_matched.py verify out/objr2/sph` -> **`out/objr2/verify_sph.json`, problems none** over
all **84 sphere runs** (every run `device cuda`, the file name's lobe / null / diameter / seed against the JSON's own
record, `retina.mode = in_loop_capture`, LC11 143 / LC10a 275 per run, the dimmed-elevation band consistent to
1.76 deg across runs). Every run's realised `execution.device_name` is in the `runs` table of every export directory.

**Correction (found on review).** An earlier draft of this section said "the per-process console files
(`out/objr2/sph/*.txt`) came with the data". **Only 14 of 84 arrived**: `ls out/objr2/sph/*.txt | wc -l` = **14**
against 84 run JSONs, and `verify_sph.json` records `console_device_cuda: None` for **70 of the 84** runs. The
**device claim still holds, but it rests on the run JSONs** -- all 84 read `execution.device = cuda`, and that is
what `runs.device_name` is copied from -- not on the console cross-check. The consequence is that the
console-vs-JSON check `docs/INTERP.md` 10.4 item 4 makes mandatory, and which was supposed to *replace* the missing
`<n> job(s), 0 failed` line for this batch, exists for **14 runs, not 84**. (The compare batch's hand fetch pulled
240/240 and is the model to copy: pull `*.json` and `*.txt` before the `*.npz`.)

The fetch had one wrinkle worth recording: a finished 12 s matched-sphere run npz is **114 MB**, of which **106 MB**
is the two `(1200, 12235)` graded per-frame arrays `a__optic_dr` / `b__optic_dr` -- the captured radiance of both
arms compresses to 0.5 MB, because the scene is static and only the ball moves. Neither the predeclared analysis
(which reads the graded types from each run's JSON summary) nor this export (whose graded per-body rows are the
probe's own whole-window means from the run JSON, which is the predeclared treatment of the upstream types) reads
those two arrays. `scripts/object_round2_export.py slim` drops them **on the box** before the transfer, which turns a
~10 GB fetch into ~0.7 GB; every other array is copied through unchanged, the run JSONs are untouched, and the
`runs` table of every export records `npz_arrays` and an `npz_note` saying exactly what the fetched npz no longer
holds.

```sh
# per box (vast-a/b/c/d in .cluster.json), incremental: slim on the box, then pull only what is missing
ssh -p <port> root@<ip> "cd /root/runs/objr2-6915ce && source .venv/bin/activate && \
  python scripts/object_round2_export.py slim --dir out/objr2/sph --slim-out out/objr2/fetch/sph"
ssh -p <port> root@<ip> "cd /root/runs/objr2-6915ce/out/objr2/fetch/sph && tar c ." | tar x -C out/objr2/sph
ssh -p <port> root@<ip> "cd /root/runs/objr2-6915ce/out/objr2 && tar cz -C sph \$(cd sph && ls *.json *.txt)" | tar xz -C out/objr2/sph
```

Then, unchanged from the baseline task's own documented pipeline:

```sh
PYTHONIOENCODING=utf-8 python scripts/object_round2_baseline.py rfmap   --out out/objr2
PYTHONIOENCODING=utf-8 python scripts/object_round2_baseline.py analyse --out out/objr2 --json out/interp/objr2/baseline.json
PYTHONIOENCODING=utf-8 python scripts/object_round2_export.py  ladder   --out out/objr2 --baseline out/interp/objr2/baseline.json \
    --export-root out/export --index out/export/objr2_index.json --group ship
PYTHONIOENCODING=utf-8 python scripts/object_round2_export.py  tables   --run-dir out/export/objr2-* --csv out/export/objr2_tables.csv
PYTHONIOENCODING=utf-8 python scripts/object_round2_export.py  check    --run-dir out/export/<a rung> --out out/objr2 --summary-dir out/export/<summary>
PYTHONIOENCODING=utf-8 python scripts/object_round2_export.py  verify   --run-dir out/export/objr2-*
```

## 4. Every table, with row counts and SHA-256

**194 tables over the 13 directories, 42,750,310 rows, 490 MB of table files** (`out/export/objr2_tables.csv`, which
carries the per-directory SHA-256 of all 194). Each rung directory has the same 15 tables; the six `ship` rungs and
the six `fb0` rungs differ only in the numbers. The 11 deg `ship` rung, in full:

| table | role | format | rows | cols | SHA-256 |
|---|---|---|---|---|---|
| `readout_per_body` | interchange | csv | 26,118 | 45 | `a2d0e88e1b5adb58519a314a00d48eceac7223c8b70d477207cfe2a8df4bc9d2` |
| `retina_columns` | interchange | csv | 1,466 | 11 | `a7ca0890f3e957b90d2ea9de51003992f70ab8176e5b32f731c3922972749c31` |
| `retina_bodies` | interchange | csv | 5,895 | 9 | `8e22b3ed23cb9fb876013a754c48c1d5aa25ebbc41f91670f49abaf4b990b35a` |
| `retina_radiance` | interchange | parquet | 1,759,200 | 7 | `baedb04056e7bd8e08f2116d6f4b7c66a7ca4f13870f1dfe092701c9b7fb674a` |
| `retina_radiance_blank` | interchange | parquet | 1,759,200 | 7 | `007788ead1a852b19e8a1fccc7635aee0cffd9e844a71eabc58a1716b685dc65` |
| `retina_object_track` | interchange | csv | 1,200 | 16 | `8470af17f94069695aaa4f560e49657a95c4535dc19cea5dfe472b2112ce65e9` |
| `per_type` | tool | csv | 24 | 37 | `12a5b2b545d3e4d43c2fc1b4e79a267dd615f1c29b543ed8e6d04a8d0d5437b0` |
| `per_run` | tool | csv | 24 | 20 | `1cec9fdc1a719f007e514ea72146e0af4e49484f271fd913adb55cfb38690005` |
| `upstream_per_run` | tool | csv | 48 | 9 | `b70b160e9730cc7dad81b1695d2649bcbdd5186ffc20a3f2bc37267bdd81950e` |
| `time_course` | tool | csv | 800 | 11 | `00adf637f6b02466ea3b2cfff8340079daf29969e5560d00838fda7d32976c58` |
| `retina_footprint` | tool | csv | 1 | 18 | `f0204c71e7078fbc73b20dd16949574e2b600cc08de0663c8b22f56de5fb5020` |
| `retina_footprint_runs` | tool | csv | 6 | 22 | `d095218c2a4588c7b5820f106c7845d29ff4e67c8709d7309e2643fd53792b60` |
| `rf_map` | tool | csv | 7,031 | 42 | `3ea3a711271593161c642444d2749c0c8767717c92d578142c61946c0339bd34` |
| `column_definitions` | tool | csv | 13 | 3 | `7e67404dfdfef0ffe209ba6139b1a3486ddedd60974190b1ca1c0f53ce8b7931` |
| `runs` | tool | csv | 12 | 20 | `60f53812cae7022cab64099646166391755ec9649a89db39f63a43e0e5541476` |

`readout_per_body`: **26,118 rows over 13,059 bodies and 14 types**, two rows per body -- `upstream_drive_mV` and
`output_Hz` for the 824 spiking bodies (**LC11 143**, **LC10a 275**, LC10b 95, LPLC2 185, LC4 126) and
`rate_deviation` and `abs_rate_deviation` for the 12,235 graded optic units (Mi1 1,773, Mi4 1,772, T3 1,940,
T2 1,630, Tm3 2,054, Tm5Y 898, TmY5a 1,364, TmY13 432, TmY21 372). LC11 and LC10a are never pooled:
`export.verify` checks that every body of both carries both quantities, and `expect_counts` 143 / 275 holds in all
twelve rung directories.

The ladder summary (`objr2-ladder-20260913T232912Z-72041020`), 14 tables:

| table | rows | cols | SHA-256 |
|---|---|---|---|
| `size_tuning` | 288 | 37 | `007d6c69c489cd5295db318ac7307e7fcdb9d979b988f9f77eba06200e08b1cb` |
| `per_run` | 288 | 20 | `d78d2dbe9493257690eec9e72c8fcb18376dd94a5f2dadba586839d3046c5aee` |
| `preference` | 24 | 12 | `fbd2888e3d0509cd36bd78ee54bd4e357872165deb2f4ca4ea9991d81aa292c6` |
| `retina_footprint` | 12 | 18 | `32af84815435d27805d5dfe929b8382e7c8665bc9a1fbc335e914cb3f15cda92` |
| `retina_footprint_runs` | 84 | 22 | `aab5bdcf30f4cd4a670449e2b1f4ec5f67d85132c2f0fdbeb5a26c7712730721` |
| `upstream_per_run` | 336 | 9 | `37625f3e9afef10999a5f43f4d8ef302c288c08befb195e91d0fb337a5c4eb9f` |
| `time_course` | 9,600 | 11 | `512698e9629bce305861dac9640b31c91fae80240cdeeec88307e6aa23c46ad5` |
| `rf_map` | 7,031 | 42 | `3ea3a711271593161c642444d2749c0c8767717c92d578142c61946c0339bd34` |
| `runs` | 84 | 20 | `2fca42acd63175a773347ca0c3fa644315f26aa18ca30c58685af06e2d0e2840` |
| `column_definitions` | 23 | 3 | `689c192756e854253258031cf36b53c280b586fb77b1b3fc9d184c3e9af0cf1b` |
| `old_ladder_size_tuning` | 64 | 14 | `25d70ae0d8e31eb6629ae203553e7960b86d04484a2be46adcdca2b7d974b313` |
| `old_ladder_footprint` | 4 | 7 | `b5df12be46690f232811a39cd4cd81242d779b1885d17642138a1443beadf15a` |
| `old_ladder_geometry` | 4 | 9 | `2e798dad99d7a845f3063c1015e696ca69405591be06ed259984a340eac61d1c` |
| `export_directories` | 12 | 5 | `6de820b904e4e5cdd1158929ebb0860b03670bb91446853ebe23141a833c24ca` |

**Correction (found on review): the ladder summary's `preference.csv` shipped a refuted p, and has been
re-emitted.** In the 2026-09-13 delivery, four of `preference`'s 24 rows -- both lobes x both LC types, statistic
`spikes_median`, **all four `role = primary`** -- carried `spearman_p_perm = 4.999750012499375e-05` with an **empty**
`spearman_rho` (and `null_spearman_p_perm = 4.9975e-04` beside them). The Spearman is undefined there because every
spike median is exactly 0.0 in all 36 object runs, so the correct value is NaN; the 5e-05 is the
`spearman_perm` floor the baseline skeptic refuted (`cnt = 0` when rho is NaN, giving p = 1/20001). The export
copies the analysis through and computes no statistic of its own, so this is a defect of
`out/interp/objr2/baseline.json` as it stood, faithfully carried into the table.

**Closed 2026-09-14, on CPU, without touching the cluster.** `scripts/object_round2_baseline.py analyse` was re-run
on the already-fetched batch and `scripts/object_round2_export.py ladder --out out/objr2 --baseline
out/interp/objr2/baseline.json --family primary` re-emitted all **13 directories** (194 tables, 42,750,310 rows,
`export.verify()` problems none in every one; index `out/export/objr2_index.json`,
`baseline_sha256 393b7631cc37cfa9...`). In
`out/export/objr2-ladder-20260914T024906Z-035363c0/preference.csv` those four rows now carry empty `spearman_rho`
**and** empty `spearman_p_perm`; no other number in the delivery changed. The 2026-09-13 directories stay on disk
as the superseded delivery, and the row hashes in the two tables above are theirs.

One cost of the re-emit, recorded rather than hidden: the 2026-09-14 directories were written from a working tree
that had drifted since the batch ran, so their `flyverse_commit` reads `unknown` with `source_match.verified false`
(19 of 29 loaded identical, 30 of 43 glob), where the 2026-09-13 directories carry commit `c78cb93d...` **verified
29/29 loaded and 43/43 glob** (section 1). For content-pinned provenance, read the 2026-09-13 pair; for the
corrected `preference.csv`, read the 2026-09-14 ladder summary.

## 5. Validation

**5.1 The exporter's own checks.** `export.verify` -- hashes recomputed from the files, decimal bodyIds, every
bodyId present among `cache/neurons.parquet`'s 167,106, the paired-quantity rule with `expect_counts` LC11 143 /
LC10a 275, the `control_ids` alias rule, the retina-mode rule -- reports **`problems: none` for all 13
directories** (the summary carries no `readout_per_body`, so it is verified with `--paired ''`). The round trip
Result to files and back, in each `checks.json`, reproduces all 26,118 rows x 31 numeric `readout_per_body` columns
to **3.6e-15**, with every text column identical (including the three-valued `verdict`, whose value `null` a default
pandas reader would turn into a missing value) and the ids matching.

**5.2 The exported radiance IS the presented stimulus.** Recomputing `probe_object_matched.footprint`'s four summary
statistics from `retina_radiance` against `retina_radiance_blank` alone -- nothing but the two Parquet tables and
`retina_columns` -- reproduces the probe's own numbers at every rung:

| rung | dimmed > 5 % per frame | dimmed > 50 % per frame | min relative radiance | max abs diff, probe vs the exported tables |
|---|---|---|---|---|
| 4.5 deg | 1.842 | 0.363 | 0.2793 | 0.0 |
| 8.8 deg | 4.058 | 1.884 | 0.0480 | 5.6e-17 |
| 11 deg | 5.196 | 2.790 | 0.0510 | 5.6e-17 |
| 15 deg | 9.113 | 5.057 | 0.0383 | 2.2e-16 |
| 20 deg | 14.354 | 9.824 | 0.0307 | 1.4e-16 |
| 30 deg | 30.948 | 23.043 | 0.0186 | 0.0 |

The footprint is **identical between the `ship` and `fb0` lobes at every rung** (`retina_footprint`, 12 rows): the
stimulus does not depend on the lobe, which is what an in-loop capture with the fly pinned should give.

**5.3 The assay is matched -- per frame, not per rung.** From `retina_object_track` (1,200 frames per rung), beside
the old scene ladder in the same summary directory (`old_ladder_geometry`):

| | matched (this export) | old ladder (`objsize-d*`, revision 2) |
|---|---|---|
| centre elevation | **0.00 deg at every rung**, `realised_el_maxdev` 0.0 | 0.88 / 4.34 / 8.66 / 13.71 deg at azimuth 0, and moving along the sweep |
| angular diameter | constant along the sweep to **1.4e-14 deg** (4.500000 / 8.800000 / 11.000000 / 15.000000 / 20.000000 / 30.000000) | 4.50 to 2.88, 11.42 to 7.32, 20.08 to 12.90, 30.18 to 19.52 within each sweep |
| angular speed | **40.000 deg/s** (39.999999999998 to 40.000000000006 off the turns) | 45.84 deg/s at the centre, 18.79 at the ends (2.4x) |
| azimuth swept | -50.000 to +50.000 deg | -50.19 to +50.19 deg |
| dimmed-column elevation band | symmetric about 0 (-1.5 to +1.6 at 4.5 deg, -15.3 to +14.6 at 30 deg) | -2.95 to +3.95 at 4.5 deg, -2.95 to +29.25 at 30 deg |

**5.4 The RF window in the interchange table is the analysis' own.** The window columns of `readout_per_body`
(`window_stimulus_minus_control` and friends) are computed by this export from the per-frame arrays; joined against
`out/objr2/per_body_sphere.csv`, which the predeclared analysis computed with a different function
(`object_round2_baseline.windowed_body_stats`), over all **418 LC bodies** of each rung they agree to **1.2e-16 mV**
on the drive, **5.6e-17** on the windowed spike count and **0 frames** on the window itself.

**5.5 The two rank tests agree where they must.** Over the 288 `size_tuning` rows, **129 are untied**, and there the
predeclared tie-aware exact permutation U and the export's contract `export.mann_whitney` give
**max |p - p_mannwhitney| = 0.0**; the other **159 rows carry ties** (mostly arms in which every run returns the same
sparse-spike value) and there `p_method` reads `asymptotic_tie_corrected`, never a false `exact`. The export's Holm
within the predeclared family reproduces the analysis' own Holm exactly: **max |p_holm - analysis_p_holm| = 0.0**
over the 24 predeclared-family rows.

## 6. What the delivered numbers say

This is the analysis' answer, copied through; its record is `docs/audits/object_baseline_r2.md`.

**The predeclared primary families do not call a size preference.** Twelve members per LC type (6 rungs x
{`drive_median`, `spikes_median`}: the per-run population median over bodies of the RF-windowed object-minus-blank
drive and spike count), 6 runs per arm:

* **LC11: 12 of 12 members `null`**, smallest unadjusted p 0.180, smallest `p_holm` 1.000. The excess over the null
  is largest at the small rungs (+0.103 mV at 4.5 deg, then +0.042 / +0.051 / +0.009 / +0.015 / +0.005 mV at 8.8 /
  11 / 15 / 20 / 30) -- the direction Keles and Frye 2017 predicts -- but the Spearman of the per-run statistic
  against diameter is rho -0.216, permutation p 0.206, and the small-minus-large contrast +0.029 mV, p 0.277.
  **Not a result.**
* **LC10a: 11 of 12 `null`, one `result`** -- 30 deg, `drive_median`, +0.0155 +- 0.0132 mV against a null of
  -0.0046 +- 0.0060, z +3.37, p 0.00866 -- **which does not survive Holm within its own 12-member family
  (`p_holm` 0.1039)**. The preference tests are flat (rho +0.030, p 0.872). By the predeclared call rule (`result`
  AND `p_holm <= 0.05`) **no size preference is called for either type**; the one member that reads `result`
  unadjusted sits at 15-30 deg, the direction Schretter et al. 2024 predicts for LC10a, and is reported as
  exploratory.
* **`spikes_median` is exactly 0.000 +- 0.000 in every arm of every rung**, object and null alike: the LC
  populations emit essentially no spikes in this protocol, so the spike half of each family is uninformative rather
  than negative.

**Upstream, the object figure is large, monotone in size, and survives its own family's Holm**
(`secondary_exploratory`, whole-window, the probe's definitions; 19 `result` rows on the shipped lobe of which 18
survive the analysis' Holm): `diff_signed_best_cell` for **T2** rises +0.0141 (11 deg) to +0.0218 (15) to +0.0328
(20) to +0.0442 (30) against a null of +0.0081 +- 0.0016 (z +3.8 to +23.0); **Tm5Y** +0.0111 to +0.0354 (z +4.7 to
+28.2); **T3** +0.0072 at 20 deg and +0.0116 at 30 against +0.0036 (z +8.7, +19.4); **TmY21** at 20 and 30 deg. On
the OLD headline statistic `diff_max_over_cells_mean_mv` the LC types do move at 30 deg (**LC11 +0.159 vs a null of
+0.053, z +10.6**; LC10a +0.165 vs +0.074, z +3.6) -- both exploratory, both at the LARGE end, which is the opposite
of LC11's biological small-object preference and is exactly the kind of statistic Neurome's intake warned about: a
maximum over 143 cells taken within each run.

The one-line reading of this delivery: **on a matched assay with the confounds removed, no size preference is
detected in either LC population's predeclared statistic at 6 v 6, while the upstream small-field medulla types
carry a clear, size-monotone object figure.**

## 7. Three departures from the shipped exporter, and why

The export goes through `flyverse/interp/export.py` unchanged -- that module belongs to another task and this one did
not edit it. Three things it cannot do for this protocol were done in `scripts/object_round2_export.py` instead, each
recorded in the manifest it writes:

1. **`retina_object_track` is the probe's in-loop measurement, not `export.object_track()`.** That helper models the
   OLD probe's geometry -- a ball resting on the table sliding along a lateral LINE, so centre elevation is
   `r - eye_above_table` above the eye and both elevation and angular diameter move along the sweep. It cannot
   express a constant-elevation, constant-distance, constant-angular-speed ARC; feeding it a `ball_offset_m` would
   ship a wrong track. The table is therefore built from the run's own per-frame `seen_az/el/diam/dist` arrays
   (beside the intended arc and the realised angular speed), written with the exporter's own `export.write_table`
   under the contract name, and its manifest entry appended to `manifest.json` after `export()` returns.
   `export.verify()` re-reads and re-hashes it exactly like any other table (it is in every directory's `problems:
   none`). `manifest.retina.object_track.why_not_export_object_track` says this in the delivered file.
   **What the exporter should grow:** a `retina_track=` argument that accepts a ready-made frame, so a probe that
   measured its own geometry does not have to append to the manifest.
2. **The statistic definitions are registered at runtime.** `export.STATISTIC_DEFINITIONS` has no entry for this
   round's statistics (`drive_median`, `spikes_median`, `drive_mean`, `drive_max`, ...), and
   `export._statistic_definitions` would write "not defined in this revision of the export" for each of them.
   `scripts/object_round2_export.py` updates that module dict at import time (the module FILE is untouched) and also
   carries the definition per row in a `statistic_definition` column, so the definitions survive with the table if
   the registration ever goes away. **What the exporter should grow:** a `statistic_definitions=` argument to
   `export()`.
3. **The empirical columns of the track use the matched blank frame**, not the blank's frame 0 as
   `export.retina_tables` does. Both are reported: `manifest.retina.object_track.blank_is_static` and
   `max_abs_diff_vs_blank_frame0_convention`, which is **0.0** at every rung of this delivery (the blank scene is
   static with the fly pinned, so the two conventions coincide exactly).

## 8. Findings for other owners (not fixed here)

1. **`export.verify`'s `expect_paired` default still fires on any export that has a `readout_per_body`**
   (`docs/audits/interp_export.md` 11 records the same thing for the atlas). The ladder-summary directory here has no
   `readout_per_body` at all, so it passes; a summary that grew one would need `--paired ''`.
2. **`scripts/object_round2_baseline.py::load_sphere_runs` does not check that a run's `.npz` exists** beside its
   `.json`. During a hand fetch the two can arrive out of step -- the probe writes the npz first and the JSON second,
   so a run that finished between the two transfers ships a JSON with no npz -- and `analyse` then dies with a bare
   `FileNotFoundError` in the middle of the reduction. This export's own `load_runs` skips such a pair and says so.
   One `os.path.exists` in `load_sphere_runs` would make the analysis restartable on a partial fetch.
3. **`common.Result.add_table` silently drops duplicate columns** (`DataFrame.to_dict('records')` keeps the last),
   which is how a `p_method` produced by two different reducers can collide without an error. The export renames the
   incoming column instead; a `raise` on duplicate column names in `add_table` would have caught it at the source.
4. **A per-rung `result.json` is 58 MB** -- the 26,118 `readout_per_body` rows, written verbatim into the run
   directory beside the 18 MB CSV of the same rows. `docs/audits/interp_export.md` 11 already lists this ("a future
   version could store only the tables plus a pointer"); over the twelve rung directories it is ~700 MB of the
   1.2 GB delivery.

## 9. Every number in this file, and what generated it

| where | generator |
|---|---|
| the run directories, every table, `manifest.json`, `checks.json` | `scripts/object_round2_export.py ladder` |
| the table list with row counts, bytes and SHA-256 (section 4) | `scripts/object_round2_export.py tables --csv out/export/objr2_tables.csv` |
| the retina reproduction, the window cross-check, the arc bands, the rank-test agreement (section 5) | `scripts/object_round2_export.py check --json out/export/objr2_check_<rung>.json` |
| `problems: none` per directory | `scripts/object_round2_export.py verify` = `flyverse.interp.export.verify` (hashes, decimal ids, every bodyId in `cache/neurons.parquet`, LC11 143 / LC10a 275 with both quantities, the control-alias rule, the retina-mode rule) |
| the round trip Result -> files -> the same numbers | `flyverse.interp.export.round_trip_check`, written into each `checks.json` |
| every arm, z, U, p, verdict, Holm and preference test (section 6) | `scripts/object_round2_baseline.py analyse` (the predeclared analysis), copied through |
| the RF map and the window each body got | `scripts/object_round2_baseline.py rfmap` -> `scripts/probe_synthetic_stimuli.py rfmap`, then the predeclared window rule in `object_round2_baseline.load_windows` |
| the recordings themselves | `scripts/probe_object_matched.py run` on the cluster, batch `objr2-6915ce` |
| the npz slimming done before the hand fetch | `scripts/object_round2_export.py slim` |

## 10. Caveats

* **Every rung's arms are the same six brain seeds, and the null arm is shared across the rungs of a lobe.** The
  replicate unit is the run (`docs/INTERP.md` 2.4): six object runs per rung against the six blank/blank runs of the
  same lobe, windowed with that rung's angular size, all from the one submission `objr2-6915ce`. So the six rungs of
  a lobe are not independent of each other through their common null, which is why the predeclared family puts Holm
  across the rungs rather than treating each as a fresh test. Cells are not replicates and nothing here treats them
  as such.
* **The RF localizer does not localize LC11.** At the predeclared `z_min` 5 not one of the 143 LC11 bodies is fitted
  in either lobe's 15-deg 3-pass map (`rfmap` coverage at z_min 5: LC11 0.000, LC10a 0.047), so of the 418 LC bodies
  **405 get the anatomical fallback** (`window_source = anat`: the type's median fitted box around
  `trace.column_of_cells`'s column), 7 a shipped-lobe fit and 6 an fb0 fit. That is the predeclared rule working as written, and `rf_map` says per body
  which clause it took -- but a reader should not describe these windows as measured receptive fields.
* **The synthetic rectangle ladders of the same batch are not in this export.** They are recorded on the boxes
  (`out/objr2/syn/`, the height / width / square-dark / square-bright ladders on the same arc) but their recordings
  are ~64 MB per run before slimming, ~10 GB in all, and were not pulled in this round's hand fetch. The path is
  ready: `object_round2_baseline.py analyse` without `--skip-synth` puts them in the Result, and the same `ladder`
  command then writes `synthetic_size_tuning` / `synthetic_footprint` / `synthetic_preference` into the summary
  directory.
* **The model-comparison batch (`objr2c`) is not exported here.** It was still running on the same boxes while this
  export was written and has no analysis Result yet;
  `scripts/object_round2_export.py compare --out out/objr2c --baseline out/interp/objr2c/compare.json` is the entry
  when it does (one run directory per arm x rung, the arm taking the place of the lobe).
* **The export never re-derives physiology.** `verify` and the round trip check shape, provenance and hashes, never
  whether a number is right; the analysis that produced the numbers is `docs/audits/object_baseline_r2.md`'s.
