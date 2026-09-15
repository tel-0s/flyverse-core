# Review of `feat/connectome-backends` (d9f8cf2) against `docs/CONNECTOME_BACKENDS_SPEC.md`

Independent read-only review (Opus, 2026-09-14) of Astra's implementation branch (one commit on `eac71e0`, 32 files,
+4,252/-75), run in its worktree `flyverse-connectome`; nothing edited or committed there, and no cluster job
submitted. Every number below was recomputed by the reviewer from the repository, the two FlyWire releases in
`D:\Datasets\flywire\`, and the author's house-run JSONs; where a number could not be recomputed it is marked *not
verified*. **Verdict: merge with fixes** (B1-B4). The MaleCNS gate is met more strongly than the branch claims: a
full recompile with the branch code reproduces main's shipped cache **byte for byte, file for file**.

## 1. Coverage against the spec

| spec item | status | where |
|---|---|---|
| §1.1 `neurons` columns, one row per cell, row index = matrix index | implemented | `backends/common.py:71-85` (`finish` fills every missing `KEEP_COLS` column, sets `bodyId` int64, rejects duplicates, sets `status="Traced"`, synthesises `instance`); `backends/fafb.py:36-48`; `backends/banc.py:60-86` |
| §1.1 `bodyId` = `root_id` / `Root ID` as int64, unique | implemented, verified | `common.py:77-79`; measured: FAFB and BANC both `int64`, `is_unique` True |
| §1.1 `somaNeuromere` for BANC ("derive if cheap, else NaN") | implemented (the NaN branch taken) | `common.py:74-76` fills NaN |
| §1.1 BANC `hex*` all NaN, retina/optic unavailable | implemented, verified | `connectome.py:673-675`; measured `hex1`/`hex2` all NaN |
| §1.2 `W` CSR (N,N), rows=post, cols=pre, float32, `sign(pre)*count`, `min_weight` 1 after summing per-neuropil rows | implemented | `backends/common.py:63-68` (`groupby` sums neuropils **before** the filter); `connectome.py:660-662` |
| §1.2 explicit zeros for sign-0 presynaptic cells, no `eliminate_zeros` | implemented, verified | `connectome.py:660-672`; measured: FAFB 431,202 / BANC 212,379 stored zeros; **every** stored entry with a sign-0 presynaptic cell is an explicit zero, and no non-zero entry carries a sign-0 raw count |
| §1.2 pair threshold recorded in the cache manifest | implemented, verified | `fafb.py:51` (`pair_threshold=5`, `1` when unthresholded), `banc.py:91` (`3`); measured `min |W|` over non-zero entries = 5.0 (FAFB) and 3.0 (BANC) |
| §1.2 `--edges no_threshold` for FAFB, off by default | implemented, verified | `connectome.py:683-689` `_validate_options`; `fafb.py:49`; variant cache at `cache/fafb/no_threshold/`, manifest `edges="no_threshold"`, `pair_threshold=1`, nnz 19,773,733, raw 76,944,499 |
| §1.3 `cache/` stays the MaleCNS cache, same files, same md5 | implemented, verified | md5s unchanged (section 2) |
| §1.3 new datasets in `cache/<dataset>/` with the same file set + `manifest.json` | implemented, verified | `connectome.py:786-792`; manifests carry dataset, release, per-source sha256, `alias_sha256`, pair threshold, NT rule, nt_threshold, min_weight, N, nnz, raw synapses, NT counts, compile date, empty override tables |
| §1.3 `Connectome.dataset` / `.release` | implemented | `connectome.py:402-403`; survive `subset` (`:461-466`), `save`/`load`, `extend` (`:537-551`) |
| §1.3 `connectome_fingerprint` carries dataset **and** release | partial, deliberate | `interp/common.py:881-886`: female graphs only; the MaleCNS fingerprint dictionary is pinned unchanged to satisfy §4.1. Owner decision, documented in the audit |
| §1.3 `provenance()["model"]` carries both; every non-MaleCNS Result says so | implemented, verified | `interp/common.py:986`; measured on the house JSONs: `model.dataset="banc"/"fafb"`, `dataset_release.name` likewise |
| §1.3 `FLYVERSE_DATA_FAFB` / `FLYVERSE_DATA_BANC` with the Windows defaults | implemented | `connectome.py:604-608` |
| §1.3 `fetch_data.py --fafb / --banc`, public URLs + sha256 in `data/manifest.json` | implemented, verified | `scripts/fetch_data.py:72-74,85-90,108-110`; all **9** manifest sha256s match the local release files byte for byte (URLs not fetched -- *not verified*) |
| §1.4 `load(dataset=…)`, `edges=`, `c.dataset/c.release`, `load()` unchanged | implemented | `connectome.py:794-869` |
| §1.4 `compile_connectome` a thin dispatcher; MaleCNS code moved verbatim; shared tail stays in `connectome.py` | implemented | `connectome.py:616-681`; `backends/malecns.py:14-23` is the old body verbatim; NT sign / overrides / photoreceptor columns / `in_syn` / save stay in `connectome.py` |
| §1.4 MaleCNS override constants apply to MaleCNS only; a backend declares its own (empty) tables | implemented, verified | `connectome.py:635,644`; measured (section 4): the `^(lLN|v2LN|…)` regex matches 353 FAFB cells of which **232 stay `unknown`**, and `TmY14` keeps **196 `unknown`** -- both would have been rewritten by the MaleCNS tables |
| §2.1 superclass maps incl. sensory/motor by nerve, `ol_sensory` photoreceptors, glia/trachea/not_a_neuron dropped | implemented, verified | `fafb.py:7-10,40-41`; `banc.py:10-15,61,74-77`; measured: 458 glia + 7 not_a_neuron + 8 trachea dropped, none of their IDs in the graph |
| §2.2 transmitter vocabulary incl. `TYR`→`tyramine`, `NT_SIGN["tyramine"]=0.0`, no MaleCNS cell carries it | implemented, verified | `backends/common.py:9-12`; `connectome.py:49`; measured `NT_SIGN["tyramine"]==0.0`, MaleCNS tyramine cells **0**, BANC 173 all sign 0 |
| §2.2 FAFB `nt_type` when `nt_type_score >= 0.5`, else unknown; probabilities kept in `nt_scores.parquet`; threshold in manifest | implemented, verified | `fafb.py:39,53`; measured: 110,264 rows at/above threshold, 28,991 below/NaN, and **0** non-photoreceptor cells violate the rule in either direction; `nt_scores.parquet` 139,255 rows × 9 columns |
| §2.2 BANC verified-first, first classical else first monoamine, whole string in `nt_verified`, NO/peptides dropped | implemented, verified | `common.py:55-60`; `banc.py:72-73`; measured (section 4): Delta7, PFL2, PFL3, LAL074, PS059, DNa02 all behave as specified |
| §2.2 photoreceptors histamine in every backend | implemented, verified | `connectome.py:633`; `banc.py:78`; `fafb.py:42` |
| §2.2 no MaleCNS override on a female graph; conflicts into `NT_INTEGRATION.md` | implemented | `docs/NT_INTEGRATION.md` §8 (PFL3, PFL2, Delta7, LAL074, PS059, the sign-0 populations) |
| §2.3 BANC class/subclass table for `senses` / `motor`, explicit, in `backends/banc.py` | implemented | `banc.py:16-22,38-43,79-85` |
| §2.3 mapping cell counts part of the acceptance test | partial | counts are **recorded** (`acceptance.json` `proprioception`, `motor_subclasses`) but nothing asserts them: `scripts/check_connectome_backends.py:151-154` asserts only finiteness and the column checks; `tests/test_connectome_data.py` asserts none of them |
| §2.3 FAFB `senses` / `motor` raise `NotAvailable("dataset fafb has no VNC")` | implemented, verified | `connectome.py:416-420`; `motor.py:42-44,84-86,103`; `senses.py:167`; `fly.py:509`; measured: all three raise with that exact message |
| §2.3 …"rather than a silent empty group" | **violated for BANC wing motor neurons** | `motor.py:87,92` -- see **B1** |
| §2.4 nerve table, side stripped into `somaSide` | implemented | `banc.py:23-35,46-57`; `adapt` uses the nerve side as a `somaSide` fallback (`banc.py:68`) |
| §2.5 affine `(p,q)->(hex1,hex2)` per hemisphere | implemented | `fafb.py:15-27`: `hex1=q+18`, `hex2=p+20`, identical in both hemispheres (linear part is the p/q swap) |
| §2.5 (i) DRA on the dorsal rim | partial -- passes as orientation, fails strictly; branch says so | section 5 |
| §2.5 (ii) L/R mirror | **implemented, verified, and better than MaleCNS's own** | measured max 1.583° / mean 1.435° over 772 shared coordinates vs 4.6° spacing; MaleCNS's own retina: max 5.944° / mean 4.661° |
| §2.5 (iii) T4a-d subtype offsets reproduce the MaleCNS layout | implemented, verified | measured min cosine 0.9818 over all 8 side×subtype pairs; a p/q-swapped control drops to **−0.949** |
| §2.5 (iv) `Retina.n_columns ~ 1,580` | implemented, verified (but near-vacuous as a test) | measured 1,581 = 785 L + 796 R, exactly the release's distinct `column_id` count per hemisphere; a pure translation cannot change it |
| §2.5 R1-6 via `_assign_photoreceptor_columns`; the 31 annotated types get `hex_source="annotation"` | implemented, verified | `connectome.py:677,691-712`; measured `hex_source`: annotation 45,528 (= the column_assignment row count), partner 6,818 |
| §2.5 BANC: `build_retina` raises, `FlyBrain(optic=None)` runs | implemented, verified | `retina.py:93`; `optic.py:200`; `fly.py:44-46,68`; measured both raises and a full-graph CPU `FlyBrain(banc, optic=None)` that steps |
| §3 aliases through the CSV only, with evidence; generator preserves the manual rows | implemented | `backends/common.py:24-46`; 4 new rows + 2 flag edits in `data/type_aliases.csv`, each with an evidence string; `scripts/build_type_map_typing.py:653-665` |
| §3 coverage numbers reported | implemented, verified | measured FAFB 99,203 (59.37 %) → 132,239 (79.13 %); BANC 120,413 (72.06 %) → 148,340 (88.77 %); 257 / 202 ambiguous source names |
| §4.1 bit-identity of the MaleCNS path | implemented, verified (section 2) | -- |
| §4.2 compile: FAFB ~139 k, BANC, nnz and NT counts in the manifest, under ~5 min | implemented, verified | FAFB 139,255; BANC 157,789 (the spec's ~147 k estimate is wrong -- see section 11); BANC recompiled in **7.2 s** on this desktop |
| §4.3 CPU smoke, `receptor_signs()` tier histogram, `regions.labels` counts | implemented, verified | `check_connectome_backends.py:87-115`; tier histograms and region counts reproduce the audit exactly |
| §4.4 FAFB optic: retina, one GPU room frame, four validations printed | implemented, with one validation passing only in the weak sense | `gpu/fafb.json` (B200, 1,581 columns, 77,873 rate units, finite); section 5 |
| §4.5 BANC walking replicate, one submission, `--arm-block`, 5 runs, beside `out/vncd3/` | implemented, verified | section 7 |
| §4.6 `scripts/cross_connectome.py` through the `Connectome` API only, per-release scale | implemented, verified | the script reads no release file; measured edge counts and the 1 : 0.592145 : 0.281268 scale reproduce exactly |
| §4.7 data-dependent tests behind the `data` marker; vocabulary and `(p,q)->hex` CPU tests on synthetic tables | implemented | `tests/test_connectome_data.py:10` (`pytestmark = pytest.mark.data`); `tests/test_connectome_backends.py` is entirely synthetic |
| §4.8 `CONTROL_SURFACE.md` `dataset=`, `INSTALL.md` fetch lines, `docs/audits/connectome_backends.md` with the sex caveat | implemented | `docs/CONTROL_SURFACE.md:7-35`; `docs/INSTALL.md:70-72,90-104`; the audit and `connectome_sex_matches.csv` |
| §5 no default tuned for a female graph | implemented, verified | every `lif` and `optic` field recorded in the house Results equals the shipped default, including `path_gain`, `type_path_gain`, `std_u_by_type`, `tau_by_type`, `pair_gain`, `baseline_by_type` |
| §5 per-release scale recorded; BANC optic lobes not treated as complete; verified over predicted monoamines | implemented, verified | scale in `anatomy/report.json`; BANC has no optic module at all; PFL2 verified tyramine vs PFL3 predicted tyramine both recorded, neither overridden |
| §5 male-specific types listed, no `fru`/`dsx` sex-sharing assumed | implemented, verified | `connectome_sex_matches.csv`: 266 annotated male-specific types / 1,258 cells, 263 absent from both female graphs, 3 name collisions in BANC, 2,310 female-unmatched names -- all reproduced from the CSV |
| §5 the 13.9 GB skeleton archive not fetched | implemented | not in the manifest's fetch set |

## 2. MaleCNS bit-identity: verified with the reviewer's own harness

The branch claims "a fresh recompile equals the original neuron DataFrame and the complete legacy fingerprint". The
reviewer's harness (`connectome.load(cache_dir=<scratch>, rebuild=True)` under the branch code, compared against
`D:\Projects\flyverse\cache`) shows something stronger:

| check | result |
|---|---|
| `cache/` md5 (main) vs `cache/` md5 (worktree) | identical: `neurons.parquet c50c598a708b5b373cbaffca7d6a9d82`, `W_post_pre.npz ac131529cebf98decde58d0c227b7954`, `sign0_counts.npz bf01d724acf2a1fec8fdb60ef8a9e066` |
| recompiled `W.data / .indices / .indptr` md5 | `e015d9d4007c4d2e71b604036c5e72e9` / `d898bcfbe222a506f82e835bafc4df8a` / `718a6a97e25439ab1dc2285c39e064ad` -- **equal to main's, array-for-array**, same dtypes (float32/int32/int32), same shape |
| recompiled `W_post_pre.npz` **file** md5 | `ac131529cebf98decde58d0c227b7954` -- byte-identical to main's cache file |
| recompiled `neurons.parquet` **file** md5 | `c50c598a708b5b373cbaffca7d6a9d82` -- byte-identical |
| neurons table | 167,106 × 24; identical column **order**; every column `Series.equals` True; no dtype change |
| `connectome_fingerprint(load())` on the branch vs on main at eac71e0 | identical in every key and value except `cache_dir` (the worktree path); key set unchanged -- **no new keys on the MaleCNS path**, `md5 ef23cc27bea13be7f6a96f3c04fd3737` |
| `load().dataset / .release` | `malecns` / `v1.0` |
| `provenance()["model"]` on a MaleCNS graph | gains **two** keys, `dataset` and `release` (`interp/common.py:986`); everything else unchanged, `units` unchanged, `REQUIRED_PROVENANCE` unchanged |

`flyverse/data/type_aliases.csv` is read only by `flyverse/backends/common.py:8`; nothing on the MaleCNS path
consumes it, so the four new alias rows cannot move a MaleCNS number. `receptors_by_type.csv` and
`tests/test_bit_identity.py`'s golden are untouched by the diff.

Line-by-line reading of the diff to `optic.py`, `brain.py` (untouched), `senses.py`, `motor.py`, `retina.py`,
`fly.py`: everything that executes on the MaleCNS path is a constant-time guard -- `c.require(...)` (`optic.py:200`,
`retina.py:93`, `senses.py:167`, `motor.py:44,86,103`, `fly.py:509`), `self.c.has_optic_columns` (`fly.py:68`),
`motor_groups(..., allow_missing_vnc=True)` (`fly.py:93`), and `unit_kinds`'s `& c.has_optic_columns`
(`interp/common.py:465`). The one place that can change a **recorded value** is
`interp/common.py:987-988`, which forces `model["optic"] = None` when an `fb` without an optic is passed; no shipped
caller hits it (`scripts/interp_export.py:171` derives `optic` from the same `fb`; `pm_bound_report.py:351` and
`retire_measures.py:419` pass `optic=` without an `fb`).

## 3. Tests

| run | result |
|---|---|
| branch full CPU suite (`pytest tests -q --ignore=tests/test_cuda.py --ignore=tests/test_metal.py`) | **357 passed, 19 skipped, 215 subtests passed** in 127 s -- exactly the author's claim |
| `tests/test_bit_identity.py` + `test_connectome_backends.py` + `test_connectome_data.py` | 22 passed, 5 subtests; the two `data`-marked female cache tests **ran** (caches present), they did not skip |
| baseline at eac71e0 (author's log, `out/connectome_backends/baseline_tests.txt`) | 338 passed / 19 skipped -- *not re-run by the reviewer*; the +19 is the two new files |

`tests/test_unitary.py` is the only existing test changed: `_nt_factor({"tyramine": 1.0})` no longer raises because
tyramine is now a valid transmitter, so the invalid-NT case uses `"not-a-transmitter"`. Correct and minimal.

## 4. Female backends: reviewer's numbers

Loaded from the worktree's `cache/fafb` and `cache/banc`, and independently **recompiled from the release tables**
into a scratch directory.

| quantity | FAFB v783 | BANC v888 |
|---|---:|---:|
| N | 139,255 | 157,789 |
| `W.nnz` | 3,732,460 | 3,036,600 |
| raw synapses (recomputed as Σ\|W\| + Σ sign-0 counts) | 50,666,648 | 23,553,107 |
| explicit zero entries | 431,202 | 212,379 |
| sign-0 cells | 21,868 | 16,828 |
| min \|W\| over non-zero entries | 5.0 | 3.0 |
| NT counts | ACh 74,204 / GABA 14,650 / Glu 17,380 / His 11,153 / DA 467 / OA 41 / 5-HT 696 / TYR 0 / unknown 20,664 | ACh 89,248 / GABA 21,178 / Glu 24,443 / His 6,092 / DA 7,601 / OA 1,537 / 5-HT 1,496 / TYR 173 / unknown 6,021 |
| regions | optic 89,024, central 25,281, vis-proj 8,206, MB 5,608, mechano 3,580, AL 3,433, vnc 2,286, desc 1,305, gust 532 | optic 74,419, central 37,443, vnc 16,490, mechano 10,070, vis-proj 7,793, MB 4,661, AL 4,137, gust 1,460, desc 1,316 |
| receptor tiers (stored pairs) | fallback 2,222,685 / pre_unknown 383,458 / nt_class 0 / class 116,403 / fuzzy 145,426 / alias 32,596 / exact 831,892 | 2,459,054 / 943 / 0 / 74,290 / 84,448 / 17,780 / 400,085 |

Every one of these equals the branch's audit table. The survey's independent pandas reading (139,255 cells /
50.7 M synapses for FAFB; 158,262 rows / 23.6 M for BANC) agrees.

**No MaleCNS override leaks into a female graph.** The decisive measurements: the `UNKNOWN_NT_OVERRIDE_REGEX`
pattern `^(lLN|v2LN|v3LN|il3LN|l2LN|vLN|LN)` matches 353 FAFB cells, of which **232 remain `unknown`** (the MaleCNS
rule would make them GABA); `TmY14` keeps **196 `unknown`** cells in FAFB (`TYPE_NT_OVERRIDE` would force
glutamate). The photoreceptor histamine rule *is* applied, as the spec allows (FAFB: all 11,151 cells of the
MaleCNS photoreceptor type names are histamine).

**BANC exclusions and retained rows**: source 158,262 rows − 458 glia − 7 not_a_neuron − 8 trachea = **157,789**;
none of the 473 dropped IDs appears in the graph; **11,499** rows with no `Super Class` are retained, of which 97
carry a primary type. The spec's "~147 k" estimate is simply wrong about this release; the branch's correction is
right and documented.

**FAFB NT threshold**: exactly the `nt_type_score >= 0.5` rule, with the photoreceptor rule as the only exception
(0 violations in either direction among non-photoreceptors).

**BANC verified-first**, measured on named types:

| type | cells | `Verified NT type` | `Predicted NT type` | compiled `nt` | `sign` |
|---|---:|---|---|---|---|
| Delta7 | 40 | `glutamate,serotonin` (all) | GLUT | glutamate | −1 |
| PFL2 | 12 | `tyramine` (all) | TYR | tyramine | 0 |
| PFL3 | 25 | none | TYR ×24, ACH ×1 | tyramine ×24, acetylcholine ×1 | 0 / +1 |
| LAL074 | 4 | none | SER ×2, GLUT ×2 | serotonin ×2, glutamate ×2 | 0 / −1 |
| PS059 | 4 | none | GABA | gaba | −1 |
| DNa02 | 2 | `acetylcholine` | ACH | acetylcholine | +1 |

The whole verified string survives in a new `nt_verified` column. `tyramine` is `NT_SIGN` 0.0, carried by 173 BANC
cells and **0** MaleCNS cells.

**Capabilities.** FAFB: `Proprioception`, `motor_groups`, `wing_groups` all raise
`NotAvailable: dataset fafb has no VNC`; `build_retina` builds. BANC: `build_retina` and `OpticLobe` raise
`NotAvailable: dataset banc has no optic column map`; a full-graph CPU `FlyBrain(c, optic=None)` constructs and
steps. `senses.Taste(c).sweet` is empty on **both** female graphs, as the branch states.

**Selections (spec §2.3).** The spec's MaleCNS reference numbers describe *labelled* cells; the transducer
*selections* are smaller. Both readings, measured:

| | MaleCNS labelled | MaleCNS selected (`Proprioception.counts()`) | BANC labelled | BANC selected |
|---|---:|---:|---:|---:|
| chordotonal organ | 425 | 615 (= 425 chordotonal + 190 proprioceptive `leg`) | 911 | 985 |
| subclass `leg` (all classes) | 868 | -- | -- | -- |
| hair plate | 113 | 113 | 320 | 320 |
| campaniform sensilla | 426 | **12** (only these are in `LEG_NERVES`) | 236 | 61 |
| haltere | 205 (201 proprioceptive) | 201 | 392 | 392 |
| `vnc_motor` `wm` / `hm` | 67 / 16 | -- | 64 / 25 | -- |
| MN9 / DNa02 | 2 / 2 | -- | 2 / 2 (sides R, L) | -- |

The spec's `425 + 868 / 113 / 426 / 205 / 67 / 16` are all reproduced on MaleCNS, and the branch's audit is right
that the handoff table mixes the two definitions. BANC motor subclasses fl 139 / ml 126 / hl 128 / wm 64 / hm 25 and
`haltere_side_groups` 13 L / 12 R / 0 unsided, all as published.

**Reproducibility of the female caches.** Recompiled from the release tables with the committed code into a scratch
directory:

* BANC: `W_post_pre.npz` and `neurons.parquet` **byte-identical** to the shipped `cache/banc`; `sign0_counts` arrays
  equal; manifest equal apart from `compile_date`. Compile time 7.2 s (author: 7.0 s).
* FAFB: `W_post_pre.npz` byte-identical (`6dd642be0cd3fc8f48e68a98cda3b48c`), `sign0_counts` equal, manifest equal
  apart from the date -- but `neurons.parquet` differs (`7a0aa6ca…` shipped vs `b9674b13…` fresh) **in column order
  only**: the shipped cache puts `hex_side` after `hex2`, a fresh compile puts it at position 11. Every column is
  value- and dtype-identical after reordering. See **B3**.

## 5. Column map: the reviewer's own recomputation, and the judgement

The transform is `hex1 = q + 18`, `hex2 = p + 20`, **identical in both hemispheres** (`backends/fafb.py:15`); its
linear part is the p/q swap, so it is an affine map with no shear or reflection. Resulting ranges `hex1` 1..35,
`hex2` 1..38, both positive, matching MaleCNS's 1..36 / 1..39 convention.

| validation | reviewer's measurement | discriminating control | verdict |
|---|---|---|---|
| (iv) column count | 1,581 = 785 L + 796 R, exactly the 785/796 distinct `column_id`s per hemisphere in `column_assignment.csv.gz` (45,528 rows, 31 types) | a translation cannot change the count | passes, but tests almost nothing about the transform |
| (ii) L/R mirror | 772 shared coordinates, max reflection error **1.583°**, mean 1.435°, against 4.6° interommatidial spacing | swapping the two hex axes on the right eye only: max **15.12°** | passes, with teeth; MaleCNS's own retina scores max 5.944° / mean 4.661° on the same statistic, i.e. worse |
| (iii) T4a-d offsets | per-cell Mi4−Mi1 input centroids, 8 side×subtype pairs, min cosine vs MaleCNS **0.9818**, three pairs > 0.999 | same statistic with the FAFB axes swapped: min cosine **−0.949** (T4a and T4b invert) | passes, and is the strongest evidence for the transform |
| (i) DRA on the dorsal rim | 126 hex-assigned non-putative community DRA photoreceptors (62 L / 64 R, from 129 labelled bodies / 167 label rows in `labels.csv.gz`). Flat band (h1+h2 ≥ the grid's 80th percentile): **100/126** (50 L, 50 R). Dorsal envelope at equal `h1−h2`: median depth **1 row** on both sides, **118/126 within 2 rows**, **125/126 within 4**; the one outlier is body `720575940624946103`, 22 rows deep, whose column came from the strongest-partner vote, not the annotation | depth below the **ventral** envelope, same population: median **24 rows** on both sides | passes as orientation by a 24 : 1 margin; **fails** as a strict per-cell rim membership test |

**Judgement.** The curved-envelope reading is legitimate on the anatomy: a fly eye's dorsal boundary is not a level
set of `h1+h2`, so the flat-band test mis-scores cells in the eye's lateral wings, and the envelope test is the
better-posed question. It is also honestly reported -- the audit states 100/126 and 26 off-band cells, the JSON
lists every off-rim body ID and every cell deeper than two rows with its `hex_source`. Two caveats a reviewer should
keep: the envelope test is defined *in terms of the same transform*, so it mostly asks "is this cell near the
lattice boundary", which is much weaker than the orientation question; and the test was reformulated after the first
formulation failed. The directional claim is carried by the ventral/dorsal asymmetry (1 row vs 24) and by the T4
and mirror controls, all of which pass independently.

So: **§2.5 (ii), (iii), (iv) are met. (i) is met in the population-orientation sense the data can support, and is
not met in the strict sense; the branch says so.** The remaining problem is the gate, not the reading -- see **B4**.
Two further observations: the off-band cells are **not** only partner-voted R1-6 (L eye 6 annotation / 6 partner;
R eye 14 of 14 annotation), so this is not purely an `_assign_photoreceptor_columns` artefact; and the FAFB retina's
1,581 columns include **51 with no photoreceptor at all**, contributed by `retina.py:114-118` (`columns_with_photoreceptors`
= 1,530). Those 51 columns are permanently dark and they shift the per-eye centring that the mirror statistic uses.
Without that special case `n_columns` would be 1,530, still inside the acceptance band 1,500-1,650, so the special
case is **not** load-bearing for the check -- it is an owner decision (section 11).

## 6. Aliases

`type_aliases.csv` gains four rows and two flag edits; the reviewer diffed them and every one carries an evidence
string (204-224 characters, citing the release, the survey and the spec section):

| malecns_type | alias | system/tier/flag | MaleCNS cells behind the target |
|---|---|---|---:|
| `PS196_a` | `PS196a` | flywire / alias / — | 2 |
| `PS196_b` | `PS196b` | flywire / alias / — | 2 |
| `PEN_a(PEN1)` | `PEN_a/PEN1` | flywire / alias / — | 20 |
| `PEN_b(PEN2)` | `PEN_b/PEN2` | flywire / alias / — | 22 |
| `R7_unclear` | `R7` | flywire / alias / `backend_primary` (flag added) | 404 |
| `R8_unclear` | `R8` | flywire / alias / `backend_primary` (flag added) | 442 |

All four new targets exist as MaleCNS type names. `DNg02_a..e` are **not** collapsed, as decided. No normalisation
regex was added to code: `backends/common.py:24-46` is table-driven, prefers `exact` rows, then `backend_primary`,
and refuses to pick a last row when an alias is genuinely ambiguous (`test_aliases_never_choose_an_ambiguous_last_row`).

Recomputed coverage (MaleCNS cells whose type name is present in the female graph, before vs after normalisation):

| | source names | before | after | ambiguous source names |
|---|---:|---:|---:|---:|
| FAFB | 8,772 | 99,203 (**59.37 %**) | 132,239 (**79.13 %**) | 257 |
| BANC | 11,543 | 120,413 (**72.06 %**) | 148,340 (**88.77 %**) | 202 |

Exactly the published numbers, and the "before" figures reproduce the survey's 59 % / 72 %.

The generator change (`scripts/build_type_map_typing.py:653-665`) preserves manual rows by the `backend_primary`
flag or the evidence string, concatenates them last and dedupes on `(malecns_type, alias, system)` with
`keep="last"`, so the manual row wins. The logic is correct by reading; **the generator itself was not executed**
(it needs the Schlegel/Nern external tables) -- *not verified end to end*.

## 7. Walking replicate: independently recomputed

Read directly from `out/connectome_backends/walking/*.json` + `*_body.npz` and `D:\Projects\flyverse\out\vncd3`,
with the reviewer's own arithmetic (means of 16-fly run means, SD across runs).

* **All 25 BANC room runs**: `NVIDIA B200`, `dataset_release.name = banc`, `release v888`, `model.dataset = banc`,
  N = 157,789, batch 16, 60 s, blocks `fam_r0..fam_r4`, batch id `cbwalk-384afd` in every file. Arm specs are
  exactly `ARMS_BODY`: A none, B `all`, C `all+leg_cycle`, D `+haltere_sided`, E `+haltere_coriolis`.
* **MaleCNS comparison**: 20 runs on `NVIDIA H200`, `male-cns`, N = 167,106. The reviewer checked the submission
  attribution independently: every `room_*_r{0,1,3,4}.json` in `out/vncd3` names **`vncd3b-c2eeaf`** and every
  `room_*_r2.json` names `vncd3-f3bb50`. The default `--malecns-seeds 0,1,3,4` therefore does use one submission.
  (`summarize_connectome_walk.py` records the batch name as a *declared* label; it does not verify it -- the
  reviewer did.)

| arm | MaleCNS clean yaw SD (°/s) | BANC clean yaw SD | MaleCNS straightness | BANC straightness |
|---|---:|---:|---:|---:|
| A shipped | 2.641 ± 0.146 | 0.275 ± 0.004 | 0.9943 ± 0.0007 | 0.9984 ± 0.0001 |
| B all proprioception | 3.327 ± 0.091 | 0.381 ± 0.003 | 0.9786 ± 0.0013 | 0.9956 ± 0.0001 |
| C + leg cycle | 7.837 ± 0.127 | 2.651 ± 0.161 | 0.8249 ± 0.0371 | 0.8655 ± 0.0172 |
| D + sided haltere | 7.731 ± 0.205 | 2.833 ± 0.137 | 0.8693 ± 0.0405 | 0.8519 ± 0.0143 |
| E + Coriolis | 7.915 ± 0.202 | 2.709 ± 0.107 | 0.8696 ± 0.0143 | 0.8614 ± 0.0091 |

n = 4 (MaleCNS) / 5 (BANC). Every digit matches the audit. Within-dataset `common.compare`, recomputed: B−A and C−B
are `result` on both measures in both cohorts (BANC p = 0.007937, MaleCNS p = 0.028571); D−C and E−D are `null` on
both measures in both cohorts. BANC C−B yaw z = 655.9, as the audit warns.

Rates, recomputed: BANC DNa02 L/R = 0/0 in A and B; **C: L 0, R 0.0985 ± 0.0133 Hz**; D 0.1132, E 0.1035. MaleCNS C
L 0.5421 / R 0.3792. BANC **PS059 C: 0.00087 / 0 Hz**, MaleCNS C 20.058 / 16.772. AN04B003 B 0.021/0.116 → C
10.738/10.273; commanded chordotonal 14.37 → 84.78; leg L−R A −0.0552, C −0.3792. All BANC body arms: clean frame
fraction 1.000, hops 0, `n_left_table` 0. Every quoted figure in the reply and audit is confirmed.

**Plain-probe counter bug.** `world.make_room(...)["table_extent"] = (-0.6, 0.6, -0.4, 0.4)` is a bounds tuple.
The pre-existing `probe_walk_straightness.py` line tested `abs(x) <= ext[0] + 1e-3`, i.e. `abs(x) <= -0.6`, which is
never true -- so every fly is "off table" from frame 0. `scripts/probe_unitary.py:305-306` already carried a comment
naming this exact bug in `probe_walk_straightness.py`, and `probe_vnc_drive.py:259` already used the correct bounds,
which is why the body-arm counters are valid. The branch's fix (`probe_walk_straightness.py:37-41,106`) is
**reporter-only**: `on_top` feeds only `left_table_s`, `frac_on_table` and `n_left_table` in the summary; no model,
gain or body constant changes, and `yaw_sd`, `straightness`, `path_m`, `hops` are untouched. All five
`straight_r*.json` carry no `metrics_version` (they predate the fix) and all report `n_left_table = 16`;
`summarize_connectome_walk.py:94-100` excludes them and instead reports **9,600 / 9,600** retained track samples
on-table under the correct bounds. Plain-walk metrics recomputed: straightness **0.998427 ± 0.000063**, path
**0.494932 ± 0.000018 m**, yaw SD **0.27710 ± 0.00385 °/s**, hops 0. Exactly as published.

**Snapshot vs commit.** The cluster snapshots' `source_fingerprint` shows the runs did **not** use the committed
tree: `connectome.py`, `fly.py` and `interp/common.py` differ from both `eac71e0` and `d9f8cf2` (and differ between
the two submissions); `optic.py` in the snapshot is the *parent* version; `retina.py`, `senses.py`, `motor.py` are
the committed ones. This is disclosed in the reply. It is closed from the other side by evidence the reviewer could
check: every Result's `compiled_connectome` digest equals the **current** cache exactly (BANC
`27e330891b62641686f64fd7a1b66138`, FAFB `31cffad8895690b83a3b02437b37bbcc`, plus md5_data/indices/indptr, nnz,
`sum_abs_W`, NT counts and the whole manifest), so the intermediate compiler produced byte-identical graphs; and
every `lif`/`optic` field recorded equals the shipped default. The walking runs also used the exact committed
`backends/banc.py` and `backends/common.py` (`files_loaded` hashes match); the FAFB GPU run used an **earlier**
`backends/fafb.py` (see B3).

## 8. Merge-conflict map

`main` is at `eac71e0`, clean, and `eac71e0` is an ancestor of `d9f8cf2`: the merge is a fast-forward and **no
conflict is possible**. Nothing in `out/`, no cache, no infra file is committed. Grepping the whole diff for IPs,
hostnames, `datasci`, `athuser`, `/mnt/beegfs`, `vast`, `ssh`, `slurm`, `sbatch`, `node1`, `hostname`: **zero
hits**. `pyproject.toml` changes exactly one line (`packages` gains `flyverse.backends`).

## 9. Blocking issues

* **B1 -- BANC wing steering and power motor groups are silently empty.** `motor.py:87` selects the steering MNs by
  `type="~^(b[123]|i[12]|iii[13]|hg[1-4]|ps[12]|tp[12]|tpn) MN$"` and `motor.py:92` the power MNs by
  `type="~^(DLMn|DVMn)"`. BANC has all of those cells, correctly classed `vnc_motor`/`wm` -- it just names them
  without the MaleCNS suffix: `b1, b2, b3, hg1-hg4, i1, i2, iii1, iii3, iii4, ps1, ps2, tp1, tp2, tpn, PSI, DLM1-4,
  DLM5, DVM1a-c, DVM2a-b, DVM3a-b, MNxm01`. Measured: MaleCNS `steer_L`/`steer_R`/`power` = 16 / 16 / 24; BANC =
  **0 / 0 / 0**, with no error and no warning, while `haltere` (selected by subclass, not by name) = 25. `read_motor`
  therefore reports zero wing-steering and zero power rates on BANC. This is precisely the "silent empty group" that
  §2.3 forbids and precisely the "notational case decided one by one in the table with evidence" that §3 requires --
  the alias pass was applied to `PS196a` and `PEN_a/PEN1` but not to the wing MNs. It does not touch any reported
  result (the walking replicate reads legs and DNa02), but the capability is broken and the audit's
  "BANC motor subclasses fl 139, ml 126, hl 128, wm 64, hm 25" hides it by counting subclass rather than the selected
  group. Fix: add the `b1 MN` ↔ `b1` (etc.) rows to `type_aliases.csv` with evidence, or make the empty steering
  group raise.
* **B2 -- `connectome.load()` on the default MaleCNS path now follows `$FLYVERSE_CACHE`.** `connectome.py:786-792`
  resolves the default cache through `Path(os.environ.get("FLYVERSE_CACHE", CACHE_DIR))`; main's
  `load(cache_dir=CACHE_DIR)` never read the environment (on main only *scripts* did, as a `--cache-dir` default).
  Measured: with `FLYVERSE_CACHE` set to a scratch directory, the branch's `load()` returns `cache_dir =
  <scratch>` where main returns `D:\Projects\flyverse\cache`. `save()` with no `cache_dir` moves the same way. The
  house submissions relied on exactly this override to point at the female caches, so it is load-bearing for the
  branch -- but it makes the shipped MaleCNS default environment-dependent, which the project rule for this path did
  not previously allow, and it is documented only obliquely (`INSTALL.md` "FLYVERSE_CACHE overrides the cache
  parent"). It also leaves `tests/test_connectome_data.py:36-41` checking `cn.CACHE_DIR` while `cn.load()` may read
  somewhere else. Either keep MaleCNS pinned to `CACHE_DIR` and route only the female datasets through the env var,
  or call the change out explicitly in `CONTROL_SURFACE.md` and the changelog.
* **B3 -- the shipped FAFB cache, and therefore the FAFB acceptance and GPU evidence, does not come from the
  committed code.** A fresh compile with `d9f8cf2` writes `cache/fafb/neurons.parquet` with `hex_side` at column 11;
  the worktree's cache has it at column 20, i.e. it was built before `backends/fafb.py:48`
  (`n["hex_side"] = n.bodyId.map(col_side)`) existed. The GPU Result's `files_loaded` confirms it: its
  `flyverse/backends/fafb.py` hash matches **neither** the committed file nor the parent. The contents are
  harmless -- W, `sign0_counts` and the manifest are byte-identical, every neuron column is value- and
  dtype-identical after reordering, and on this release `hex_side == somaSide` for **all 52,346** hex-assigned cells,
  so the changed code path is a no-op here. Still, `acceptance.json`, `gpu/fafb.json` and the column-map numbers were
  produced from a cache the committed code does not reproduce. Recompile `cache/fafb` with the merged code and
  re-run `check_connectome_backends.py` before the numbers are quoted anywhere public.
* **B4 -- the acceptance gate silently omits the validation that fails.** `scripts/check_connectome_backends.py:78-84`
  builds `checks = {dra_orientation, mirror, t4_direction, column_count}` and `:154` asserts `all(...)`, while the
  strict result `all_labels_on_rim = False` is recorded one level up, in `dra`, where the assertion never looks. A
  future run of the acceptance script therefore reports a clean pass for a spec §2.5 (i) validation that does not
  strictly hold. The prose is honest; the machine gate is not. Either put `all_labels_on_rim` into `checks` and
  record the expected failure explicitly, or rename the item so the gate and the spec agree on what is being tested.

## 10. Nits

Author resolutions are recorded separately in [connectome_backends_followups.md](connectome_backends_followups.md).

1. `connectome.py:744-745` -- the new "saving would strip an extension" guard makes the older guard at `:758-762`
   dead code, and changes the message for that case. Delete one of them.
2. `connectome.py:409-414` -- `has_vnc` / `has_optic_columns` are hard-coded from the dataset *name*, and `dataset`
   is a plain mutable dataclass field (`tests/test_connectome_backends.py:171` sets `c.dataset = "banc"` to fake a
   capability). A MaleCNS subset with no VNC cells still reports `has_vnc = True`; any future dataset name reports
   both capabilities. Derive from the data, or validate the name.
3. `connectome.py:244` -- the tyramine carve-out in `receptor_signs` is keyed on `c.dataset != "malecns"` rather
   than on the value, so a synthetic extension of a MaleCNS graph carrying `nt="tyramine"` (which
   `tests/test_connectome_backends.py:61` does for a female graph) would be tiered `pre_unknown` on MaleCNS.
4. `backends/fafb.py:38` -- `entryNerve` is set from `d.nerve` for **every** FAFB cell, not only sensory ones
   (`exitNerve` is properly gated). Harmless today because FAFB raises on every VNC consumer.
5. `backends/common.py:31` -- `normalize_types` filters `system in ("flywire", "banc")` for **both** backends, so
   FAFB also consumes rows authored for BANC and vice versa. Probably intended; say so.
6. `backends/common.py:81` -- the synthesised `instance` means `Proprioception` resolves every female side from the
   instance string rather than from laterality (measured `side_source`: BANC `instance` 985/320/61/392,
   `laterality` 0; MaleCNS chordotonal `laterality` 593 / `instance` 22). A consequence is that `both` (bilateral) is
   structurally impossible on a female graph -- MaleCNS reports 85 bilateral chordotonal cells, BANC 0. Not wrong,
   but it is a different measurement, not the same one.
7. FAFB's whole R7/R8 population maps to the MaleCNS `R7_unclear`/`R8_unclear` aggregates (1,338 and 1,357 cells vs
   404/442 MaleCNS cells behind those names), so the per-subtype `R7p/R7y/R7d` entries of the receptor and tau
   tables never apply to a FAFB optic lobe. Documented in the alias evidence; worth repeating where the optic model
   is described.
8. `retina.py:114-118` adds 51 photoreceptor-free columns to the FAFB retina. They are permanently dark and they
   move the per-eye centring that the mirror statistic measures. Flag them in `Retina.summarize` so a reader of an
   optic Result knows 51 of 1,581 columns can never carry light.
9. `interp/export.py:244-245` -- `SOURCE_PATTERNS` covers `flyverse/*.py` and `flyverse/interp/*.py` but not
   `flyverse/backends/*.py`, `data/type_aliases.csv` or `data/manifest.json`, so the static half of a cluster
   Result's `source_fingerprint` names none of the code that built a female graph. Mitigated in practice:
   `files_loaded` does include the imported backend modules, and `compiled_connectome.manifest` carries
   `alias_sha256` and the per-source sha256s. Add the package to the pattern list anyway.
10. Every Result JSON now gains `model.dataset` and `model.release`, including on the MaleCNS path -- a schema
    change on the shipped path (harmless; consumers should know). The fingerprint key set is unchanged.
11. `subset()` now propagates `_cache_dir` (`connectome.py:462`), which fixes nit 2 of
    `docs/audits/extensibility_review.md`. In a clean environment nothing moves; with `FLYVERSE_CACHE` set the
    recorded `cache_dir` of a subset changes. Changelog it.
12. `save()` writes `neurons.parquet` and `W_post_pre.npz` before `manifest.json`; an interrupted female save leaves
    a cache that `load()` then refuses with "a female cache must have a manifest". Write the manifest first, or to a
    temp name.
13. `tests/conftest.py`'s docstring says marks are applied by file name there rather than by decorator;
    `tests/test_connectome_data.py:10` uses an in-file `pytestmark` instead. Works, but it breaks the stated
    convention -- add the file to `FILE_MARKS`.
14. `summarize_connectome_walk.py:70,139` takes the historical submission name from `--malecns-batch` as a label and
    never checks it against the JSONs. The reviewer confirmed the attribution by hand; the script should assert it.
15. `scripts/check_connectome_backends.py` records the §2.3 selection counts but asserts none of them, so the
    spec's "make the mapping's cell counts part of the acceptance test" is met only in the recording sense.

## 11. Owner decisions the spec left open

Accepted as decided on the branch, all of them defensible and all of them stated in `docs/audits/connectome_backends.md`:

1. **Spec 4.1 beats spec 1.3 for MaleCNS**: the legacy fingerprint dictionary keeps its exact key set, so `dataset`
   and `release` appear in the fingerprint only for female graphs (and in `provenance()["model"]` for all graphs).
2. **BANC keeps its 11,499 unclassified rows** and drops only the three named non-neuron classes, giving 157,789
   rather than the spec's estimated ~147 k. The spec's estimate is not reachable without an exclusion it never names.
3. **Generic FAFB/BANC `R7`/`R8` map to the MaleCNS unclear aggregates**, flagged `backend_primary`, explicitly not a
   p/y/d assignment. `DNg02_a..e` are not collapsed.
4. **The FAFB retina keeps every annotated column**, including 51 with no reconstructed R axon (`retina.py:114-118`),
   giving 1,581 rather than 1,530 columns.
5. **The DRA validation is reported against a curved dorsal envelope** rather than a flat height band, after the flat
   band failed (100/126). Legitimate anatomy; see section 5 and B4.
6. **A single translation serves both hemispheres** (`fafb.py:15`), i.e. the published p/q lattice is taken to be in
   retina.py's handedness already. Justified by the mirror and T4 controls, not asserted.
7. **MaleCNS defaults remain pinned to `CACHE_DIR`; `FLYVERSE_CACHE` selects only female cache roots.**
   Corrected by B2 in `34c2eb0`; see `default_cache_directory()` in
   [connectome.py](../../flyverse/connectome.py) and the [author follow-up record](connectome_backends_followups.md).
   An explicit `cache_dir` can relocate any dataset.
8. **BANC optic-intrinsic cells run as LIF cells**, with no optic module and no retina, rather than being pruned.
9. **Female sweet taste is unavailable** because the shipped taste table carries MaleCNS body IDs; no cross-animal ID
   transfer is attempted (verified: 0 sweet cells on both female graphs).
