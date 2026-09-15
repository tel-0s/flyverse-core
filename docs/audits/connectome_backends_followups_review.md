# Review of `fix/connectome-review-followups` @ 281a68b

Read-only review (Opus, 2026-09-14) of Astra's follow-up commit in the worktree `D:\Projects\flyverse-connectome`
(branch `fix/connectome-review-followups`, `281a68b` on `4db2e8c`). Nothing edited or committed in either repo;
the worktree was left clean (`git status` empty at 281a68b). CPU only, no cluster job. Every number below was
recomputed by the reviewer; script outputs live in this scratchpad (`acceptance.json`, `cache_comparison.json`,
`pytest_wt.txt`).

**Verdict: merge with fixes.** The code is correct and the MaleCNS gate is met exactly. The fixes are operational,
not algorithmic: main's female caches must be refreshed in the same operation or `tests/test_connectome_data.py`
goes red on main (verified below, F1), and `TODO.md:317` still advertises the nits as open (F2).

---

## 1. Diff map: 12 files, +268/-11

`git diff 4db2e8c..281a68b --stat`

| file | +/- | maps to |
|---|---|---|
| `docs/CONTROL_SURFACE.md` | +26 | nits 2, 5, 6, 7, 10, 11 + new (retina coverage) |
| `docs/audits/connectome_backends_review.md` | +46 | process record (see F3) |
| `flyverse/backends/__init__.py` | +13/-2 | nit 2 |
| `flyverse/backends/common.py` | +3 | nit 5 (docstring only) |
| `flyverse/backends/fafb.py` | +2/-1 | nit 4 |
| `flyverse/connectome.py` | +8/-3 | nit 2 |
| `flyverse/data/type_aliases.csv` | +4 | new (optional BANC haltere cleanup) |
| `flyverse/interp/common.py` | +7/-1 | new (retina coverage in provenance) |
| `flyverse/retina.py` | +23 | nit 8 + nit 7 (module docstring) |
| `scripts/check_connectome_backends.py` | +61/-3 | new (`--compare-cache`) + nit 8 |
| `tests/test_connectome_backends.py` | +59 | nits 2, 4, 8 regressions |
| `tests/test_connectome_data.py` | +16/-1 | nit 8 + haltere aliases |

`flyverse/optic.py`, `motor.py`, `senses.py`, `fly.py`, `backends/banc.py`, `backends/malecns.py`: **untouched**
(empty diff, verified explicitly).

### Nit-by-nit

**Nit 2 — `has_vnc` / `has_optic_columns` hard-coded from the name. RESOLVED** (the review offered "derive from the
data, *or* validate the name"; Astra took the second). `backends/__init__.py:4-8` adds a `CAPABILITIES` registry and
`capabilities(dataset)`; `connectome.py:411-421,427` consults it from both properties, from `require()` and from
`__post_init__`, so an unregistered dataset name now raises at construction. Behaviour for the three registered
names is identical to the old `!= "fafb"` / `!= "banc"` tests (checked all three). The other half of the nit
("a MaleCNS subset with no VNC cells still reports `has_vnc = True`") is *deliberately kept* and now documented as
source-release availability (`connectome.py:410`, `CONTROL_SURFACE.md:29-31`) — a defensible reading, and the new
test asserts the subset behaviour rather than leaving it accidental. `dataset` is still a mutable field; the new
test exploits that (`c.dataset = "unregistered"`), so the mutability itself is unchanged. Acceptable.
*No regression risk*: the only `Connectome(...)` constructions in either repo pass `malecns`/`fafb`/`banc` or
`base.dataset`; the `"synthetic"` occurrences are a **neurons column**, not the dataset field (`connectome.py:507-510`,
`tests/test_extend.py:53`), so `extend()` is unaffected.

**Nit 4 — FAFB `entryNerve` set for every cell. RESOLVED and verified.** `backends/fafb.py:38` now gates on
`super_class in ("sensory","sensory_ascending","sensory_descending")`. Recomputed against the two caches:
exactly **3,250** values cleared, **0** gained, and every cleared row is non-sensory
(ascending_neuron 1,750 / descending_neuron 1,303 / cb_motor 110 / cb_endocrine 79 / cb_intrinsic 8); retained
non-null `entryNerve` falls 9,647 → 6,397 = the 5,785 `sensory` + 612 `sensory_ascending` rows of the release;
`exitNerve` is bit-identical. New regression `test_fafb_nerve_roles_are_distinct` covers both roles. See N1 for the
third super_class string.

**Nit 5 — `normalize_types` consumes flywire+banc rows for both backends; "probably intended; say so". RESOLVED
as asked.** `backends/common.py:29-31` docstring plus `CONTROL_SURFACE.md:49-51`. Verified the shared vocabulary is
harmless in practice: the four new `system=banc` rows change **0** FAFB cells (the FAFB cache diff shows only
`entryNerve`).

**Nit 6 — synthesised `instance` makes `both` structurally impossible on female graphs. RESOLVED as asked**
(the review said "not wrong, but a different measurement"). `CONTROL_SURFACE.md:51-56` states the derivation,
that no BANC afferent uses the connectivity-laterality fallback, that MaleCNS reports 85 bilateral chordotonal
cells and BANC 0, and that this is not evidence of absent bilateral anatomy. No code change, correctly.

**Nit 7 — FAFB R7/R8 → the MaleCNS unclear aggregates; "worth repeating where the optic model is described".
RESOLVED as asked.** `retina.py:7-11` module docstring and `CONTROL_SURFACE.md:41-47` (with the 1,338/1,357 counts
and the explicit statement that `R7p/R7y/R7d` and `R8p/R8y/R8d` entries cannot match).

**Nit 8 — flag the 51 photoreceptor-free FAFB columns in `Retina.summarize`. RESOLVED, beyond the ask.**
`retina.py:74-82` adds `Retina.coverage()`; `summarize` gains the count line and an empty-side guard;
`interp/common.py:991-996` adds `retina.coverage` to provenance when a live retina has uncovered columns;
`check_connectome_backends.py:101` records it in the acceptance JSON. Recomputed on the real FAFB cache:
`n_columns 1581, with 1530, without 51, L 785/42, R 796/9` — matches Astra's claim exactly, and the acceptance JSON
carries the same. The nit's second half (the dark columns move the per-eye centring) is acknowledged rather than
fixed, consistent with owner decision 11.4; `CONTROL_SURFACE.md:44-48` says so.

**Nit 10 — `model.dataset`/`model.release` on the MaleCNS path is a schema change consumers should know about.
RESOLVED as asked** (`CONTROL_SURFACE.md:33-38`, including "the legacy MaleCNS fingerprint keeps its exact key set").

**Nit 11 — `subset()` now propagates `_cache_dir`; "changelog it". RESOLVED as asked** (`CONTROL_SURFACE.md:35-38`).
There is no `CHANGELOG*` file in the repo, so `CONTROL_SURFACE.md` is the right home.

**New work (not a nit): retina input coverage report** — `coverage()`, the two `summarize` lines, `retina.coverage`
in provenance, `columns.coverage` in the acceptance JSON, the `CONTROL_SURFACE` paragraph, and three regressions
(`test_retina_reports_missing_photoreceptors_in_summary_and_provenance`, incl. an empty eye).
**New work: `--compare-cache` / `--rebuild`** in `check_connectome_backends.py:175-217` — an audit harness that
loads a retained baseline cache beside the current one, asserts matrix/column-order/cache-file identity,
asserts the full MaleCNS neuron table and fingerprint (mod `cache_dir`), and reports receptor-sign and
default-shaped-weight deltas. `rebuild=rebuild and dataset != "malecns"` correctly never recompiles MaleCNS.
**New work: optional BANC haltere aliases** — 4 rows in `type_aliases.csv` (`hi1 MN,hi1,banc,alias` etc.).
Verified: the direction matches the existing DVMn convention (col 1 = MaleCNS canonical, col 2 = source name);
all four canonical targets are real MaleCNS types (hi1 MN 2, hi2 MN 4, hDVM MN 2, hiii2 MN 2 cells); no duplicate
alias rows; exactly 11 BANC cells rename (4/3/2/2), all `class=haltere_motor_neuron, subclass=hm`, `flywireType`
preserved; the haltere group stays 25 cells (acceptance JSON). The evidence strings say "notation only, counts are
not assumed equal", which is right — the MaleCNS and BANC counts differ.

---

## 2. THE GATE: MaleCNS bit-identity — **PASS**

| check | result |
|---|---|
| `cache/W_post_pre.npz` md5 | `ac131529cebf98decde58d0c227b7954` ✓ (identical to `D:\Projects\flyverse\cache`) |
| `cache/neurons.parquet` md5 | `c50c598a708b5b373cbaffca7d6a9d82` ✓ |
| `cache/sign0_counts.npz` md5 | `bf01d724acf2a1fec8fdb60ef8a9e066` ✓ |
| `connectome_fingerprint(load())["md5"]` | `ef23cc27bea13be7f6a96f3c04fd3737` ✓ |
| fingerprint key set | the legacy 12: `cache_dir, md5, md5_data, md5_indices, md5_indptr, n_neurons, nnz, nt_counts, subset, sum_abs_W, type_nt_override, unknown_nt_override_regex` — **no** `dataset`/`release` ✓ |
| `md5_data / md5_indices / md5_indptr` | `e015d9d4…` / `d898bcfb…` / `718a6a97…`, identical worktree vs main |
| `sum_abs_W`, `nnz`, `n`, `nt_counts` | `121460584.0`, `25578600`, `167106`, identical |
| `neurons` frame | all 24 columns hash-identical worktree vs main (`pandas.util.hash_pandas_object` per column and whole-frame `944bee8a2bb5081fe6108c99dad4bcc0`); column order identical |
| `has_vnc` / `has_optic_columns` | `True` / `True`, unchanged |

`python -m pytest tests -q --ignore=tests/test_cuda.py --ignore=tests/test_metal.py` in the worktree:
**364 passed, 19 skipped, 215 subtests passed** in 182 s, exit 0 — exactly Astra's reported counts.

`scripts/check_connectome_backends.py` (CPU) passes: `malecns_fingerprint.md5 = ef23cc27…`, `cache_md5` reproduces
the three MaleCNS md5s, the four column validations hold with the documented strict-DRA expected failure
(`i_dra_all_labels_on_rim: False`), MaleCNS §2.3 selections unchanged
(`chordotonal 615 / hair_plate 113 / campaniform 12 / haltere 201`; `steer 16/16, power 24, haltere 16, ttm 2, gf 2`).

### MaleCNS default path, line by line

Four touchpoints, all no-ops for `malecns`, each verified by execution rather than by reading:

1. `connectome.py:427` `capabilities(self.dataset)` in `__post_init__` — a frozenset lookup on every `Connectome`
   construction (subset/prune/extend included). No behaviour change, negligible cost.
2. `connectome.py:411-416` — `capabilities("malecns") == {"vnc","optic_columns"}`, so both properties return the
   same booleans as the old name comparisons.
3. `interp/common.py:991-996` — the `retina.coverage` key is added **only** when `coverage["without_photoreceptors"]`
   is non-zero. Measured on the shipped MaleCNS retina: **1,466 columns, 1,466 with photoreceptors, 0 without**
   (L 682/0, R 784/0) → the key is never added and the MaleCNS Result schema is unchanged.
   `to_jsonable(dict)` returns a fresh dict, so no caller record is mutated (the new test asserts this too).
4. `retina.py:169-177` — the coverage line and the empty-side guard both stay silent on MaleCNS (0 uncovered
   columns; both sides populated). `coverage()` is now computed on every `summarize` call: O(n_columns), negligible.

The 4 new `type_aliases.csv` rows **cannot** reach MaleCNS: `backends/malecns.py` never calls
`common.finish` / `normalize_types` (only `fafb.py:51` and `banc.py:91` do), and the top-level `cache/` has no
manifest carrying `alias_sha256`. `flyverse/data/type_aliases.csv` *is* in `export.SOURCE_PATTERNS:248`, but
`common.source_fingerprint` only hashes when git cannot resolve the commit, so no desktop MaleCNS Result changes.

---

## 3. Female caches vs a fresh compile — **PASS**

md5, worktree `cache/` vs main `cache/`:

| file | banc | fafb | fafb/no_threshold |
|---|---|---|---|
| `W_post_pre.npz` | SAME | SAME | SAME |
| `sign0_counts.npz` | SAME | SAME | SAME |
| `nt_scores.parquet` | — | SAME | SAME |
| `neurons.parquet` | DIFF | DIFF | DIFF |
| `manifest.json` | DIFF | DIFF | DIFF |

The `neurons.parquet` differences are exactly what the commit claims and nothing else (row order unchanged,
column set unchanged): **banc** `type` 11 + `instance` 11; **fafb** and **fafb/no_threshold** `entryNerve` 3,250 each.

**Fresh BANC recompile** (`FLYVERSE_CACHE=<scratch> python -c "cn.load(dataset='banc', rebuild=True)"`,
n=157,789, nnz=3,036,600): `W_post_pre.npz`, `sign0_counts.npz` **and** `neurons.parquet` are byte-identical to the
worktree's `cache/banc`; `manifest.json` differs in `compile_date` only (every other key equal, including
`alias_sha256` and the per-source sha256s). So Astra's "byte-identical" claim holds, with the correct caveat that
`neurons.parquet`/`manifest.json` are *not* identical to **main's** copies — main's are the pre-follow-up ones.

`check_connectome_backends.py --compare-cache D:\Projects\flyverse\cache` (read-only against main, no `--rebuild`)
reproduces the commit's headline result independently:

```
malecns  threshold    changed columns: {}
fafb     threshold    changed columns: {'entryNerve': 3250}  receptor: all 0  shaped weights: 0
banc     threshold    changed columns: {'type': 11, 'instance': 11}  receptor: all 0  shaped weights: 0
fafb     no_threshold changed columns: {'entryNerve': 3250}
```

i.e. `fast_sign / fast_gain / slow_sign / slow_gain / slow_class / tier` and the default shaped weights have
**zero** changed entries on both female graphs. Nothing downstream of the annotation moves.

---

## 4. Hygiene — clean

* No infrastructural strings anywhere in the diff: grepped the full patch for IPs, `beegfs`, `<cluster-domain>`, `<cluster-user>`,
  `vast`, `/mnt/`, `ssh`, `slurm`, `sbatch`, `.edu`, `hpc` — zero hits.
* `git ls-tree -r 281a68b` has nothing under `out/` or `cache/`; `.gitignore:1,4` covers both.
* The two commands the appendix documents write into `out/` (gitignored). No generated artefact is tracked.
* Worktree left clean; all reviewer output went to the scratchpad.

## 5. Merge-conflict map against `D:\Projects\flyverse` main @ 78d36c3 — **no conflicts**

`git merge-tree --write-tree 78d36c3 281a68b` → tree `f471195c8cd1252838e352a2c1d9b6b88223f50c`, exit 0, zero
conflict records. Main gained `27d265c` (`CITATION.cff`, `README.md`, `TODO.md`, `docs/INSTALL.md`,
`docs/OVERVIEW.md`, `docs/REPRODUCIBILITY.md`), `5838c20` (`TODO.md`, `docs/audits/column_ground_truth.md`,
`object_baseline_r2.md`, `object_rectangles_r3.md`, `receptor_verification.md`, `scripts/probe_column_ground_truth.py`,
`tests/test_column_ground_truth.py`) and `78d36c3` (`docs/NEUROME_INTERFACE.md`). **281a68b touches none of them.**
Semantic check of main's new code against this commit's API changes: `probe_column_ground_truth.py:67` builds a
`Connectome` with `dataset=c.dataset` (registered), `tests/test_column_ground_truth.py:61-64` uses a synthetic
`hex_graph()` on the default `malecns` and only reads `n_columns` — both safe under the new validation.

---

## 6. Findings

### Fixes required at merge (not code defects)

**F1. Main's female caches go stale the moment this merges, and a test goes red.** `cache/` is gitignored, so main
still carries the pre-follow-up `cache/banc/neurons.parquet`. Verified: running the worktree's suite with
`FLYVERSE_CACHE=D:\Projects\flyverse\cache` fails
`tests/test_connectome_data.py:70::test_banc_haltere_names_normalize_without_changing_selected_cells`
(1 failed, 4 passed) because the 11 haltere cells still carry the raw `hi1`/`hi2`/`hDVM`/`hiii2` types.
The appendix does say "copy those or recompile after merging", but it is buried in a review-doc appendix.
Copy `D:\Projects\flyverse-connectome\cache\{banc,fafb}` over main's, or recompile both, as part of the merge.
(`fafb` has no test that detects its staleness, but leave the two consistent.)

**F2. `TODO.md:317` on main still reads "open nits 2/4/5/6/7/8/10/11 listed there"** and 281a68b does not touch
`TODO.md`. Tick it in the merge commit, the way `4db2e8c` did for the B-fixes.

### Nits (non-blocking)

**N1. `flyverse/backends/fafb.py:38` names a super_class that does not exist in FAFB v783.** The release's
`classification.csv.gz` vocabulary is `optic, central, sensory, visual_projection, ascending, descending,
sensory_ascending, visual_centrifugal, motor, endocrine` — there is **no `sensory_descending`**, so the third
string in the `isin` is a dead branch. Worse, `SUPERCLASS` (`fafb.py:7-10`) has no `sensory_descending` key either,
so if a future release introduced one those cells would keep `entryNerve` while `superclass` went NaN. The
appendix and `CONTROL_SURFACE.md` both state all three as though they were release values. Drop the string, or
keep it as forward-compatibility with a matching `SUPERCLASS` row and a comment.

**N2. `flyverse/connectome.py:418`** re-hardcodes `("vnc", "optic_columns")` in `require()` instead of deriving
from `backends.CAPABILITIES` — the exact duplication the nit asked to remove, one level down. A fifth capability
added to the registry would be rejected here as "unknown connectome capability".
Fix: `if capability not in set().union(*CAPABILITIES.values()):`.

**N3. `flyverse/interp/common.py:991-996`** assumes `to_jsonable(retina)` is a `dict` before assigning
`retina_record["coverage"]`. Every in-repo caller passes a dict or `None` (checked all 20), so it is safe today,
but a `str`/`Path` `retina=` would raise `TypeError` inside `provenance()` — which the neighbouring
`source_fingerprint` docstring explicitly promises never to let provenance fail a run. Guard with
`isinstance(retina_record, dict)`.

**N4. `flyverse/retina.py:78`** puts the full `without_photoreceptors_indices` list into `coverage()`, which goes
verbatim into every Result's provenance. 51 ints is fine; a future release with thousands of uncovered columns
would bloat every JSON. Consider capping the list in the provenance record (the count and side counts are the
part a reader needs).

**N5. `tests/test_connectome_data.py:58-70`** skips only when the BANC cache is *absent*, and fails with a bare
`assert len(selected) == count` when it is merely *stale* (see F1). An assertion message naming the cause
("recompile the BANC cache: `type_aliases.csv` changed"), or a manifest `alias_sha256`-vs-CSV skip, would save the
next person the diagnosis.

**N6. `scripts/check_connectome_backends.py:230`** —
`if args.gpu and args.compare_cache or args.rebuild and not args.compare_cache:` is correct by precedence but
should be parenthesised.

**N7. `scripts/check_connectome_backends.py:196-199`** builds `changed_columns` by indexing `new.neurons[k]` for
every `k in old.neurons` *before* asserting `same_column_order`; a column-set change raises `KeyError` instead of
the intended assertion.

**N8. Process: the follow-up record was appended into the reviewer's audit file.** The diff is a clean append
(`@@ -419,3 +419,49 @@` — 3 context lines, 46 added, **nothing above the appended section altered**; I diffed the
whole file), and the new section 12 is explicitly labelled "Astra, after merge at 4db2e8c" and says "the
independent review above is unchanged". That is about as careful as an in-place append can be. Still, an
independent audit that accumulates author-written resolutions becomes ambiguous in authorship for a later reader,
and the file's top-line **Verdict: merge with fixes (B1-B4)** no longer describes the file's end state. A sibling
`docs/audits/connectome_backends_followups.md` linked from section 10 (or a TODO entry) would keep the audit an
audit. Related: section 12 supersedes owner decision **11.7** ("`FLYVERSE_CACHE` becomes the cache root for every
dataset including MaleCNS") while 11.7 itself still reads as current — the actual behaviour is
`connectome.py:796-806`, MaleCNS pinned to `CACHE_DIR`, female roots from `FLYVERSE_CACHE`, which is what the
appendix says. One of the two should point at the other in place.

**N9. Appendix wording.** "Every `W_post_pre.npz` and `sign0_counts.npz` is byte-identical to its merged baseline"
is true and I reproduced it; but it sits next to "The merged female caches were copied to main before these
follow-ups and all 14 copied files checked by SHA-256", which reads as if main's caches were current. They are
not (F1). Say plainly that main's `neurons.parquet`/`manifest.json` are superseded by this commit.

### What I could not verify

* The appendix's "Main's CPU suite then passed 360 tests with 19 skipped" — main has since gained
  `tests/test_column_ground_truth.py`, so that count is no longer reproducible as stated. Not material.
* No GPU path was exercised (CPU-only mandate); the commit claims none was needed and touches no GPU code.
