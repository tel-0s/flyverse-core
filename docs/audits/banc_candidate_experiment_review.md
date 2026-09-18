# Review: `docs/banc-lattice-review` (79ba161) and `feat/banc-candidate-experiment` (715d1ac)

Read-only review. Worktree `D:\Projects\flyverse-connectome`; main `D:\Projects\flyverse` at `0b3668f`, read
but never written. Both worktrees are byte-unchanged by this review (`git status` clean in the review
worktree; main still carries exactly the other agents' edits). All runs CPU-only
(`PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1`), no cluster job. Everything I ran went to the scratchpad
or to gitignored `out/`.

My interpreter is `<workstation-home>\miniconda3\python.exe` — Python 3.13.2, NumPy 2.3.5, SciPy 1.17.0,
pandas 3.0.1, torch 2.10.0+cu128 — i.e. exactly the stack the branches' `report.json` records, which
matters because the reconstruction is thread- and BLAS-sensitive (see B2 below). The repo `.venv`
(NumPy 2.5.3 / SciPy 1.18.1) is a *different* stack and would not reproduce these numbers.

## Verdicts

| branch | verdict |
|---|---|
| `docs/banc-lattice-review` @ 79ba161 | **Merge.** Every B1-B3 fix and every nit is present and every number I recomputed reproduces, including the root cause of B2 that the review could not find. |
| `feat/banc-candidate-experiment` @ 715d1ac | **Merge with fixes.** The isolation is real and complete, the acceptance runs reproduce bit-for-bit on my CPU, and the construction is honest. Three things must change first: a now-false statement in `docs/CONTROL_SURFACE.md`, two undisclosed asymmetries that inflate the one cross-dataset number the branch reports (the loom GF row), and an unsourced cartridge rule. |

79ba161 is an ancestor of 715d1ac, so merging branch 2 brings branch 1 with it.

---

# Branch 1 — `docs/banc-lattice-review` @ 79ba161

`git diff 0b3668f..79ba161 --stat`: 3 files, +202 / -27.
`docs/audits/banc_column_reconstruction.md` +91, `scripts/recover_banc_columns.py` +133,
`tests/test_banc_columns.py` +5. **`flyverse/` untouched.** Nothing under `out/` or `cache/`.

`scripts/recover_banc_columns.py` and `tests/test_banc_columns.py` are byte-identical at 79ba161 and
715d1ac, so everything below tests branch 1's script exactly even though I ran it from the branch-2
checkout (whose only `flyverse/` deltas are inert on this path — verified in branch 2, section 1).

## 1. Every B1-B3 fix is present and correct

**B1 (DRA gate unreadable as written).** `docs/audits/banc_column_reconstruction.md:9`, `:131-140`:
the summary now says "the strict DRA check is unpassable as written, including on FAFB's published map;
its BANC result primarily exposes missing annotations", the 80/90 and 60/64 ground-truth numbers are
stated, the FAFB *reconstruction*'s own 75/87 is reported, the proxy's precision and recall are given,
and the mapping failure (11/28) is explicitly separated from the rim failure (7/11) as "the strong
incompleteness signal". All of it is now machine-generated rather than prose: `review_controls()`
(`scripts/recover_banc_columns.py:548-628`) emits it under a new `--review-controls` flag.

My run of `python scripts/recover_banc_columns.py --out <scratch> --fafb-control --review-controls`
(**exit 2**, see section 3):

| audit claim | my `report.json` |
|---|---|
| FAFB published-map proxy 80/90 within two rows | `review_controls.fafb_published_proxy`: n 90, mapped 90, within_two_rows **80** |
| proxy precision 61/90 = 67.8 %, recall 61/64 = 95.3 % | `proxy_labels`: real 64, overlap 61, precision **0.67778**, recall **0.95313** |
| labels source hashed | `labels.csv.gz` sha256 `bdd4eafa…` recorded |

**B2 (BANC diagnostics not regenerable).** The audit's tables are regenerated and a fingerprint block
sits beside them (`:79-96`). Every figure reproduces on my machine to the digit:

| quantity | audit @ 79ba161 | my run |
|---|---:|---:|
| right: Mi1 seeds / sites / neighbour edges | 878 / 877 / 2,374 | 878 / 877 / 2,374 |
| right: collisions / displaced / step agreement | 64 / 136 / 82.39 % | **64 / 136 / 82.3926 %** |
| right: max displacement | 1.323 | 1.32321 |
| right T4 cosines a/b/c/d | .9999846 / .9996227 / .9996896 / **.9996263** | .99998461 / .99962275 / .99968958 / **.99962629** |
| left: collisions / displaced | 114 / 175 | 114 / 175 |
| left T4c/d cosines | -0.973 / -0.819 | **-0.9733785 / -0.8186576** |
| unassigned seed | `720575941689026011` | same |

Note the T4d cosine: the independent review's table said `.9996363`; the *audit* says `.9996263` and
the audit is right — my run gives 0.99962629. The reviewer's digit was the transposed one.

The fingerprint block also checks out: `generator SHA256 a0630eca…` is exactly
`git show 0b3668f:scripts/recover_banc_columns.py | sha256sum`, i.e. the block correctly identifies the
*0b3668f* generator for a table regenerated with the 0b3668f reconstruction code, and my own run's
generator hash is the different `c276340…` of the 79ba161 script (which changed only in ways that do
not touch the solve). The raw-data hashes match my disk:
`neurons.csv.gz 40a2201554a8c34d…`, `connections_princeton.csv.gz 8772298eb6945575…`.

**And the branch found the cause the review could not.** The audit claims (`:88-92`) that setting
`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to 4 reproduces the *original* ebd6f26 numbers. I reran the
committed script under `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` and got **right 65 collisions /
139 displaced / 82.013 % step agreement, left 116 / 179** — the ebd6f26 row exactly. So B2 was a
BLAS-thread-count difference at the `np.rint` boundary, not a different script or cache, and
`report.json` now records `thread_environment` so a future run is self-identifying. This is a
better resolution than the fix the review asked for.

**B3 (94.38 % is a tuned optimum).** `:177-181` states it in bold with the k = 5 / 6 / 7 sweep
(78.54 / 94.38 / 67.82) and "the narrow peak has no robustness margin", and `:193-198` adds the
BANC-specific 80-85 % exact / 97-98 % within-one estimate with its assumptions and the instruction
that it must travel with any use of the candidate. My `fafb_control.comparison` reproduces the whole
13-row table cell for cell (Mi1 739/783 = 94.3806 %, 777/783 = 99.2337 %; T4c 842/725/582/700;
T4d 777/718/629/703; …).

## 2. The nits are all done, and the new controls are real

- **True holdout.** `--review-controls` runs `recover_eye(..., partners=("Tm3",))` and `("T4a","T4b")`
  and records `cd_excluded_from_neighbour_graph: true`. My run: FAFB Mi1 **576/765 = 75.29 %** (Tm3) and
  **585/782 = 74.81 %** (T4a+b) — the audit's numbers exactly; BANC held-out c/d cosines Tm3
  **.99836 / .99659**, T4a+b **.99998 / .99876** — the audit's numbers exactly. The audit is careful
  that these "test orientation independently, at lower exact column accuracy".
- **Per-release-scaled DRA threshold.** `scale = 13536/28497 = 0.474997368` from raw DNa02 input
  budgets; my run's table is identical to the audit's (FAFB 5 → 90/90/80, 10 → 85/85/77; BANC
  2.375 → 28/11/7, 4.750 → 21/7/6), and it is labelled "a global release-yield assumption, not a
  calibrated local DRA model".
- **Exit 0 unreachable** is now stated in the module docstring (`:9-11`) and the audit (`:21-23`),
  as "a closed integration boundary, not a new anatomical finding".
- **Citations named**: Seung (2024) *Nature* and Matsliah et al. (2024) *Nature* replace the bare PMC URLs.
- **`sys.path` bootstrap** added at `tests/test_banc_columns.py:5-11`; the 10 column tests collect and
  pass when invoked by absolute path from outside the repo (verified).
- **Em dash** gone: `scripts/recover_banc_columns.py:541` is now `"DIAGNOSTIC ONLY -- anatomical gate
  remains closed"`, and the whole file is ASCII.
- **Bare `assert`** replaced by `raise RuntimeError("diagnostic changed a source connectome fingerprint")`.
- **`direct_dra_labels()` usecols** now wrapped in `try/except ValueError` returning `unavailable`.
- **`--out` mkdir ordering** fixed: the emptiness check precedes `mkdir`, so a rejected invocation no
  longer leaves a directory behind.

Two nits the review raised that the branch did *not* take, both defensibly: the DRA `counts > 0`
default stays (the scaled thresholds are added as a *sensitivity* diagnostic rather than a new default,
which is the right call for a gate), and the gate itself is unchanged.

## 3. The gate still behaves

My run printed
`Integration gate: {'i_dra_rim': 'fail', 'ii_lr_mirror': 'unavailable', 'iii_t4_direction': 'pass',
'iv_column_count': 'pass'}` and **exited 2** — as designed and as documented. Exit 2, not 0, and not
because of the new holdout: the holdout controls run in a separate `review_controls` block and never
feed `checks`, so the gate's inputs are unchanged from 0b3668f. `unchanged_fingerprints`
`{malecns: true, banc: true}`, `synthetic_nodes_added: 0`, `synthetic_edges_added: 0`,
`malecns_cache_md5` all three matching.

## 4. Branch 1 nits (none blocking)

- `docs/audits/banc_column_reconstruction.md:246-253` ("Subsequent owner scope") announces the branch-2
  experiment inside branch 1. It is accurate and it says the diagnostic gate remains closed, but it makes
  79ba161 non-self-contained: merged alone, it forward-references an audit file that does not exist.
- `review_controls()` re-runs `recover_eye` four times and re-reads FAFB, roughly doubling wall time
  (my full `--fafb-control --review-controls` run was ~4 min). Fine for a flag, worth a docstring note.
- `review_controls()` raises `ValueError` if the DNa02 budgets are unavailable, so a release without raw
  counts kills the whole `--review-controls` run rather than reporting `unavailable` — the same failure
  mode the branch just fixed in `direct_dra_labels()`.

---

# Branch 2 — `feat/banc-candidate-experiment` @ 715d1ac

`git diff 0b3668f..715d1ac --stat`: 16 files, +15,649 / -48. 10,600 of those lines are
`flyverse/data/banc_candidate_columns.csv` and 3,941 are `flyverse/data/banc_candidate_vision.json`.
Nothing under `out/` or `cache/` is committed (both are in `.gitignore`); `docs/HANDOFF_CONNECTOME_REPLY.md`
is correctly ignored by `docs/HANDOFF_*.md`.

## 1. What executes on the MaleCNS path, or on plain `load(dataset="banc")`

Under `flyverse/` the branch touches **only** `connectome.py` (+22/-6), `interp/common.py` (+8/-6) and
the new `banc_vision.py`. `retina.py`, `optic.py`, `fly.py`, `motor.py`, `brain.py`, `body.py` and all of
`backends/` are **untouched** — confirmed by `git diff --name-only`. Every hunk read line by line:

- `connectome.py:414-418` — `has_optic_columns` becomes
  `"optic_columns" in capabilities(self.dataset) or self.vision is not None`.
- `connectome.py:420-423` — new `vision` property, `(self._extension or {}).get("vision")`.
  For MaleCNS and for plain BANC `_extension is None` → `None` → the boolean is unchanged.
  `require("optic_columns")` (`:425-431`) consults `has_optic_columns`, which is the single switch that
  unlocks `build_retina` for the candidate. Clean and auditable.
- `connectome.py:816, 822-832` — `load()` gains `vision=` / `vision_cache_dir=`. Both default `None`;
  `vision=None` takes the original code path with no added statement executed before it except the two
  `if` tests. `vision` other than `"candidate"` raises; `vision_cache_dir` without `vision` raises.
  The candidate path calls plain `load()` first and then extends its result, so the biological load is
  literally the same call.
- `interp/common.py:987-988, 1005-1010` — `provenance` adds `model["vision"]` and swaps two `units`
  strings, all three guarded by `if c.vision is not None`. On MaleCNS nothing changes.
- `scripts/probe_motion.py`, `scripts/probe_loom.py` — `device=args.device` is now passed to
  `OpticLobe` and `Brain`; both already default `device: str | None = None`, so passing `None` is
  identical to omitting it. `probe_loom` sets `w.device` only `if args.device is not None`.
  `probe_motion` adds an `included` mask that is `np.ones(...)` when `--eye both` (the default), so the
  recorded quantity is unchanged on the default invocation.

I could not find anything that runs on the MaleCNS path.

## 2. The gate, with my own harness

I wrote an independent harness (`scratchpad/harness_gate.py`) that loads MaleCNS, BANC and FAFB, computes
the CSR md5 itself rather than trusting `connectome_fingerprint`, and ran it against **both** the branch-2
worktree and main at 0b3668f.

| check | result |
|---|---|
| MaleCNS fingerprint `md5` | **`ef23cc27bea13be7f6a96f3c04fd3737`** — my own `md5(data‖indices‖indptr)` agrees |
| fingerprint key set | exactly **12** keys: `cache_dir, md5, md5_data, md5_indices, md5_indptr, n_neurons, nnz, nt_counts, subset, sum_abs_W, type_nt_override, unknown_nt_override_regex` — the pinned legacy set, nothing added |
| MaleCNS cache md5s | `neurons.parquet c50c598a708b5b373cbaffca7d6a9d82`, `W_post_pre.npz ac131529cebf98decde58d0c227b7954`, `sign0_counts.npz bf01d724acf2a1fec8fdb60ef8a9e066` — all three match |
| MaleCNS `n` / `nnz` / `sum|W|` | 167,106 / 25,578,600 / 121,460,584 |
| `tests/test_bit_identity.py` + `test_extend.py` + both BANC test files | **22 passed, 5 subtests**, exit 0 |
| full CPU suite | **385 passed, 40 skipped, 215 subtests, exit 0** in 150 s |

The suite count matches the branch's claimed **385 passed / 215 subtests**. The skip count differs
(40 vs 19) for exactly the environment reason the previous review documented; passes and subtests are
identical, so this is not a regression. One caveat for whoever reruns it: on this machine
`tests/test_banc_candidate.py` **errors at collection** with
`PermissionError: [WinError 5] … \Temp\pytest-of-ethee` unless `--basetemp` is redirected — a sandbox
permission on the system temp root, not a code defect. With `--basetemp=<scratch>` all four pass.

**Plain `load(dataset="banc")` is unchanged.** Harness run on branch 2 vs on main 0b3668f, every field
identical except the new `vision` attribute existing and returning `None`:

| | value |
|---|---|
| `n` / `nnz` | 157,789 / 3,036,600 (both trees) |
| CSR md5 (my own) | `27e330891b62641686f64fd7a1b66138` — equals main's, and equals the branch's claimed "BANC biological MD5" |
| `has_optic_columns` | `False` |
| `build_retina(banc)` | raises `NotAvailable: dataset banc has no optic column map` |
| `neurons.hex1` non-null | 0 |
| negative body IDs | 0 |
| `cache/banc/*` bytes | md5-identical between `D:\Projects\flyverse\cache\banc` and the worktree's copy |

FAFB is also unchanged (`md5 31cffad8895690b83a3b02437b37bbcc`, `has_optic_columns True`).

## 3. The opt-in, verified against the graph rather than the docs

Invocation: `connectome.load(dataset="banc", vision="candidate", vision_cache_dir=<scratch>)`, or via
`scripts/probe_vision_common.py` as `--dataset banc --vision candidate --vision-cache <dir>`.
`_scratch_only()` refuses a path inside `cache/`, and `extend_candidate` additionally refuses a
`cache_dir` that is the biological cache or nested with it.

**The pinned lattice regenerates from source on my machine.** My own `--review-controls` run's
`banc_R_candidate.csv`, LF-normalised, hashes to
`60181451e52435ff31ca950d63d71833eb43740c470ab32d78721c2c74716e3d` — **byte-identical** to the committed
`flyverse/data/banc_candidate_columns.csv` and to `meta["columns_sha256"]`. That is the strongest
reproducibility result in either branch: the shipped asset is not a hand-carried artefact.

My harness (`scratchpad/harness_candidate.py`) then rebuilt the candidate and measured it:

| property | measured |
|---|---|
| synthetic nodes | **5,262** = 877 cartridges × **6** (every cartridge exactly 6, no exceptions) |
| body IDs | −1 … −5,262, `int64`, unique, **zero** overlap with biological IDs |
| node fields | `dataset='synthetic'` ×5,262, `type='R1-R6'`, `nt='histamine'`, `sign=-1`, `superclass='ol_sensory'`, `somaSide='R'`, `release='banc-v888-right-k6-v1'` |
| synthetic edges | **12,762**, all with negative (inhibitory, histaminergic) weight |
| edges *into* synthetic cells | **0** — the layer is pure input, nothing feeds back into it |
| postsynaptic types | L1 5,190 / L2 4,122 / L3 3,450; unique target cells **865 / 687 / 575** |
| edge weights | exactly two values: **−8.438026** (L1, L2) and **−1.406338** (L3) |
| **column locality** | **12,762 / 12,762** edges land on a cell with the *same* `(hex1,hex2)` as the presynaptic R cell; **0** violations; all targets `somaSide == 'R'` |
| cartridges with no lamina target | **6** |
| biological block | `c.W[:n,:n]` is **byte-identical** to plain BANC's `W` in `data`, `indices` *and* `indptr`; `(bio − base).nnz == 0` entry by entry; **212,379 stored zeros preserved on both sides** |
| biological cells given coordinates | 9,783, all `somaSide == 'R'`, all `hex_source == 'connectivity_candidate'`, across the 13 mapped types (Mi1 877, L1 865, L5 850, T4b 786, T4a 784, Mi4 767, T4d 762, T4c 743, R8_unclear 709, R7_unclear 708, L2 687, Mi9 670, L3 575) |
| retina | 877 columns, all side R; 6,679 photoreceptors = 5,262 synthetic R1-R6 + 709 native R8_unclear + 708 native R7_unclear (native R7/R8 retained, as claimed) |
| candidate CSR md5 | **`9281a7fb3586cce55bc90a906ab09416`** — the claimed value, and the value the house run recorded |
| prune | `c.prune(bodyId<0)` returns the original base object: n 157,789, md5 `27e330891b…`, `has_optic_columns False`, 0 hex |
| save → reload → prune | reload md5 `9281a7fb…`, `vision["mode"]=="candidate"`, `has_optic_columns True`; pruning the reloaded graph gives md5 `27e330891b…`, `has_optic_columns False`, 0 hex — the restoration survives a round trip |
| sign-zero counts | identical prefix arrays before and after extension |

**Provenance.** `provenance(candidate)["model"]["vision"]` carries `qualification` =
`"synthetic input layer on a candidate lattice"`, `accuracy_estimate.exact_fraction == [0.80, 0.85]`,
`within_one_column_fraction == [0.97, 0.98]` with method/source/assumptions/errors strings, the full
four-check block (`i_dra_rim: fail`, `ii_lr_mirror: unavailable`, `iii_t4_direction: pass`,
`iv_column_count: pass`), the whole `review_controls` blob, `anatomically_validated: false`, and the
generator/thread/version fingerprints. `units` becomes
`photoreceptor: "synthetic input layer on a candidate lattice; native R7/R8 retained"` and
`node_set: "banc v888; … ; negative-ID synthetic R1-R6 candidate extension"`. On plain BANC, `model`
has no `vision` key at all. This satisfies the requirement in full.

**The cartridge rule, stated precisely.** For each of the 877 Mi1-seeded columns: create 6 identical
histaminergic `R1-R6` cells carrying that column's `(hex1,hex2)`; connect *each* of the 6 to *every*
mapped native `L1`, `L2`, `L3` in that same column, with per-edge weight
`median(MaleCNS right-eye native R1-R6 → T) × (13,536 / 48,125)`. I recomputed the derivation from the
shipped caches and it is exact:

| target | male right native pairs | median raw count | × 0.28126753 | shipped weight |
|---|---:|---:|---:|---:|
| L1 | 2,180 | 30.0 (IQR 22-38) | 8.438025974 | 8.438025974 |
| L2 | 2,167 | 30.0 (IQR 23-40) | 8.438025974 | 8.438025974 |
| L3 | 2,004 | 5.0 (IQR 3-10) | 1.406337662 | 1.406337662 |

`DNa02` raw input budgets: BANC 13,536, MaleCNS 48,125, ratio 0.28126753. Every number checks.
So the *counts* do have a stated source and a stated scale — they are a measurement on MaleCNS's own
graph, not invented — and `input_layer.assumption` says so. The *rule* does not; see F3.

## 4. The acceptance runs, recomputed and controlled

Provenance from the five Result JSONs: `device_requested cuda`, `device cuda`,
`device_name NVIDIA B200`, `host <cluster-node>`, `platform Linux-6.8.0-63-generic`, `torch 2.11.0+cu128`;
run IDs `ledger-20260915T0520…` to `…T052149Z`, i.e. one contiguous 93-second block — **one submission,
four sequential probes plus the aggregate**. `replicates {n: 1, unit: "run", runs: [0], null: None}`;
loom seed 0. MaleCNS Results carry `compiled md5 ef23cc27…`, `model.vision` absent; BANC Results carry
`compiled md5 9281a7fb…`, `model.dataset "banc"`, `model.release "v888"`, `model.vision.qualification
"synthetic input layer on a candidate lattice"`. Both probes ran `--eye right` with identical stimulus
parameters across datasets (motion: 60 deg/s, 30 deg period, contrast 0.5, 1.5 s/direction, default
`OpticParams`; loom: side right, 1 m/s, radius 0.03, gain_out 80, out_norm l1, receptor off).
`Result.check()` returns `[]` (no problems) for all five; the five SHA-256s in the audit match the files
byte for byte; `export.match_sources` reports `verified True`, scope `loaded`, 19-21 files, 0 differ,
0 missing for all five — every claim in that paragraph of the audit is true.

### Direction selectivity, recomputed from the stored responses

`probe_motion.py`'s definitions: `DIRS = {front->back, back->front, up, down}`, subtype *a* expected
front→back, *b* back→front, *c* up, *d* down; `DSI = (max − min) / (|max| + |min| + 1e-6)` over the four
directions. I recomputed `argmax` and DSI myself from `tables.per_type[*].responses` and reproduce all
16 decisions and all 16 DSIs to six decimals:

| subtype | male responses (f→b, b→f, up, down) | male DSI | BANC responses | BANC DSI | best−2nd (BANC) |
|---|---|---:|---|---:|---:|
| T4a | .04689 .03155 .02942 .02997 | 0.228987 | **.10013** .08208 .07803 .07729 | 0.128753 | 18.0 % |
| T4b | .02754 **.05471** .04045 .04008 | 0.330266 | .07991 **.10002** .08203 .08501 | 0.111756 | 15.0 % |
| T4c | .03204 .04318 **.04991** .02634 | 0.309110 | .11329 .11821 **.14446** .11207 | 0.126262 | 18.2 % |
| T4d | .02865 .03850 .02356 **.04729** | 0.334832 | .09804 .10139 .09335 **.11976** | 0.123962 | 15.3 % |
| T5a | **.09707** .03868 .06873 .05359 | 0.430114 | **.16325** .14828 .13630 .13889 | 0.089971 | 9.2 % |
| T5b | .03678 **.09010** .05822 .06199 | 0.420194 | .15450 **.15984** .14265 .14858 | 0.056832 | **3.3 %** |
| T5c | .08017 .09299 **.12437** .06585 | 0.307647 | .18971 .18925 **.20586** .18906 | 0.042548 | 7.9 % |
| T5d | .09489 .09742 .07811 **.12861** | 0.244265 | .18066 .18263 .16839 **.18938** | 0.058681 | **3.6 %** |

So **"preferred direction right for all 8" is true**, in both graphs, at BANC DSI 0.043-0.129, against
the male's 0.229-0.430 under the same protocol. The T5 claim is precise: all four BANC T5 DSIs are below
0.1 (0.043-0.090), and the audit says exactly that, and says it fails the older ledger criterion.
Context the audit gives correctly: the repo's own recorded MaleCNS guard value is
`motion.min_dsi 0.241` with a `>= 0.1` gate (`README.md:70`, `docs/audits/guard_suites_r3.md:61`) and
the ledger reference is 0.16 (`docs/INTERP.md:465,547`). BANC's minimum is **0.0425**, so the candidate
passes the owner's preferred-direction gate and fails the repo's pre-existing DSI gate — which is
disclosed, and the Result's `validation.reference` is `{}` with `source: "owner functional experiment
gate"`, so nothing claims the ledger reference. The male comparator's own minimum here is 0.229, a
little under the 0.241 guard value, because `--eye right` is not the guard protocol.

The decisions are not comfortable: BANC T5b and T5d separate their winner from the runner-up by 3.3 %
and 3.6 % of the peak, with n = 1 run and no null. They are deterministic, not noisy — see below — but
"8/8" carries no margin at the bottom.

### The loom row, recomputed from the stored time series

Both loom Results store the full 80-step `(t_s, distance_m, gf_hz, ttm_hz)` trace. Recomputing peaks
and the threshold crossing myself:

| | MaleCNS | BANC candidate |
|---|---:|---:|
| walking GF max (150 pre-loom steps) | 0.000 | 0.000 |
| loom GF peak | **44.096458** | **53.298664** |
| GF threshold | 33.0 | 33.0 |
| first crossing | t = 0.53 s, d = 3.5 cm, GF 35.53 | t = 0.46 s, d = 5.0 cm, GF 33.86 |
| frames at or above threshold | 23 / 80 | 17 / 80 |
| loom TTM peak | 38.360828 | **0.000000** |

Every number the audit prints is exactly what the recordings contain. The gate as coded
(`probe_loom.py:110`, `max(walk_gf) < flight.gf_hz <= gf_peak`) is genuinely passed by both, and the
criterion string explicitly says a TTM-only escape would not pass.

**"TTM silent" is not a naming artefact, and it is not caused by the candidate.** `wing_groups`
(`flyverse/motor.py:122`) selects `type == "TTMn"`, and BANC *has* 2 cells typed `TTMn` and 2 typed
`DNp01`, exactly like MaleCNS — the group is populated and `_no_silent_empty_wing_group` would have
fired otherwise. The BANC TTMn simply never reaches threshold, and the connectome says why:

| | MaleCNS | BANC | ratio |
|---|---:|---:|---:|
| DNp01 → TTMn direct raw weights | 70 and 20 | **8 and 5** | 0.11 |
| DNp01 out-degree | 313, 342 | **34, 29** | 0.10 |
| TTMn total positive input | 1,181 / 818 | 304 / 444 | 0.26 / 0.54 |
| 2-hop positive DNp01→*→TTMn | 2,662-3,069 | **0, 0, 89, 40** | ~0.02 |

BANC's giant-fibre output is ~0.10 of the male's where the release's global yield ratio is 0.28, so the
GF→jump pathway is roughly 2.5× sparser than even BANC's overall incompleteness predicts. This is a
property of the BANC VNC reconstruction, not of the synthetic retina, the lattice, or the alias fix. The
audit's statement — "the GF result does not establish recovery of the downstream jump motor pathway" —
is correct; the *reason* is worth one sentence because it is a clean, checkable connectome fact.

### Two controls I ran that the branch does not

`probe_motion.py` runs in ~3.5 min on CPU for this graph, so I ran it three times.

**(a) Device independence — bit-exact.** Re-running the BANC candidate motion probe on **CPU** from the
committed scratch cache reproduces the B200 run's 32 response values and 8 DSIs to every printed digit
(T4a `.10013 .08208 .07803 .07729`, DSI 0.128753 — identical). The acceptance measurement is not a
GPU artefact and can be reproduced off-cluster.

**(b) The gate is sensitive to the lattice — a real result.** I rebuilt the candidate with the
**same 877 sites** but cell→column identity randomly permuted *within each type* (10,593 of 10,599 rows
moved), so the geometry, the cartridge count and the synthetic layer are unchanged and only the
reconstruction's content is destroyed:

| | correct | DSI range |
|---|---:|---|
| shipped candidate | **8 / 8** | 0.043 - 0.129 |
| identity-shuffled lattice | **1 / 8** (chance) | **0.010 - 0.036** |

So the 8/8 is not a property of the hex geometry alone. The reconstruction carries real column
structure and the probe detects it.

**(c) The direction *labels* are inherited, not measured.** I rebuilt the candidate with the lattice
rotated 180° (`hex → −hex`), which is one of the twelve symmetries `orient()` chooses among:

| | correct | T4a-d DSI |
|---|---:|---|
| shipped candidate | 8 / 8 | .1288 .1118 .1263 .1240 |
| 180°-rotated lattice | **0 / 8**, every direction exactly inverted | **.1355 .1150 .1244 .1226** |

The DSIs are unchanged; only the labels flip. That is the decomposition the branch does not make:
`orient()` (`scripts/recover_banc_columns.py:281-321`) *chooses* the lattice symmetry that maximises
cosine agreement of BANC's T4a/T4b one-column offsets with MaleCNS's, and the exported CSV stores the
transformed coordinates; `build_retina` then maps `+(h1+h2)` to dorsal and `(h1−h2)` to anterior with a
fixed geometry. So "which lattice axis is up" came from the male, before the probe ran, and check (iii)
already gated on it. **The 8/8 is one structure test plus one inherited 12-way convention, not eight
independent successes.** Both halves are worth having; the audit reports them as one number.

## 5. Blocking issues

### F1. `docs/CONTROL_SURFACE.md` now contains a false statement about the surface it documents

`docs/CONTROL_SURFACE.md:29`: "`has_vnc` and `has_optic_columns` describe the **source release**, and
survive subsetting…". After `flyverse/connectome.py:414-418` that is no longer true of
`has_optic_columns`: it returns `True` for a graph whose `_extension` carries a `vision` key, which is
not a property of the source release. `:40` ("BANC has no optic column map: `build_retina` raises
`NotAvailable`") is likewise now conditional, and
`docs/CONNECTOME_BACKENDS_SPEC.md:39` ("`hex*` all NaN, `Retina` / `OpticLobe` unavailable") reads the
same way. Neither `vision=` nor `vision_cache_dir=` appears in `CONTROL_SURFACE.md`'s "Connectome
datasets" section, in its "Hooks, modules, graph extension" table (which documents
`Connectome.extend` and would be the natural home), in `EXTENSIBILITY.md`, or in `INSTALL.md` — the only
user-facing description of the opt-in is inside two audit files and a gitignored handoff. The branch
changed a documented public property and did not update its documentation.

**Fix:** amend `CONTROL_SURFACE.md:29` and `:40`, and add the two-line opt-in with its caveats
(right eye only, unvalidated map, synthetic input layer, scratch cache required) to the datasets section
or the extension table.

### F2. The one cross-dataset number in the branch has two undisclosed asymmetries, both favouring BANC

The loom row ("BANC 53.30 Hz vs MaleCNS 44.10, escape 70 ms earlier at 5.0 cm vs 3.5 cm") is the single
most quotable result here. Two construction choices push it in that direction and neither is mentioned:

1. **The synthetic cartridge delivers ~2× the reference's own per-lamina-cell photoreceptor budget.**
   `scripts/build_banc_candidate.py:55-61` takes MaleCNS's *per-edge* median (30/30/5) and
   `flyverse/banc_vision.py:91-109` applies it once per each of **6** receptors. But MaleCNS's own right
   eye carries **2.85** R1-R6 per column (2,237 cells over 784 columns; only 526 of those columns have
   any R1-R6 at all), and each right L1 actually receives **75.0** raw counts of R input, not 6 × 30 = 180.
   Scaled to BANC's yield that is 21.1; the synthetic layer delivers **50.6**. Same arithmetic for L2
   (21.7 vs 50.6) and L3 (4.3 vs 8.4) — **2.40× / 2.33× / 1.97×**. Six per cartridge is the
   anatomically right number; 30 per edge is a number measured on an incomplete reconstruction; the
   branch multiplies them together.
2. **The candidate eye is complete and the comparator's is not.** The BANC candidate right eye has
   6 R1-R6 in every one of its 877 columns. The MaleCNS right eye that served as comparator has
   **zero** R1-R6 in **258 of its 784 columns** (distribution 0:258, 1:65, 2:64, 3:64, 4:63, 5:79, 6:164,
   7+:27), because "the retina is partly outside the EM volume" (`docs/NOTES.md:43`). The candidate's eye
   is also **11.9° wider in azimuth** (span 143.4° vs 131.5°, from the stored `retina.directions`).

Both are consistent with what the recordings show: BANC's optic-lobe responses run **2-3× the male's**
across every probed type (T4a .100 vs .047, T5a .163 vs .097, T5c .206 vs .124), which is also a
candidate explanation for the lower DSI (a more strongly driven, more saturated front end). The audit
does label the weights "an unmeasured synthetic assumption" and says the loom row is "not evidence of a
female/male behavioural difference" — that discipline is real and I credit it — but it never gives the
reader the 2× number or the 258-empty-columns number, and it tabulates MaleCNS beside BANC candidate as
though the two eyes were matched.

**Fix:** state both quantities beside the loom table (one sentence each), or demote the male/BANC GF
comparison to a same-graph statement ("the candidate's looming response crosses the escape threshold;
the MaleCNS value is for orientation only, under a differently populated retina").

### F3. The cartridge rule itself has no named source

The counts are sourced and scaled (section 3). The *rule* — six receptors per cartridge, targeting
L1/L2/L3, uniform weight per target type — is asserted as "neural-superposition cartridge" in exactly
three places (`docs/audits/banc_candidate_experiment.md:23`,
`scripts/build_banc_candidate.py:104`, `flyverse/data/banc_candidate_vision.json:3847`) with **no
citation anywhere in the branch** (`git grep -i "meinertzhagen|o'neil|braitenberg|kirschfeld"` over the
new files: nothing). One commit earlier, branch 1 was required to replace bare PMC URLs with named
authors for a far less invented claim. The same standard applies to the only fabricated part of the
graph.

**Fix:** name a source (Braitenberg 1967 / Kirschfeld 1967 for neural superposition;
Meinertzhagen & O'Neil 1991 for the lamina cartridge's R1-R6 → L1/L2/L3 connectivity), *or* state
plainly that the rule is copied from the MaleCNS graph's own R1-R6 → L1/L2/L3 topology rather than from
the literature — which is in fact what the code does, and is a perfectly good answer.

## 6. Nits

- **Persisted-candidate loading bypasses every identity guard.** `extend_candidate` verifies
  `columns_sha256`, `release`, `base_csr_md5` and `base_annotations_sha256`
  (`flyverse/banc_vision.py:64-77`). `probe_vision_common.load()` (`scripts/probe_vision_common.py:50-57`),
  when `--vision-cache` already contains `extension.json`, loads that cache directly and checks only
  `c.vision["mode"] == "candidate"`. A stale or hand-edited scratch cache is accepted without a single
  hash check. Harmless in the house run (which builds the candidate first) but worth one re-verification.
- **`has_optic_columns` trusts an unvalidated dict.** Any `_extension` containing a `vision` key flips
  it. Consider keying on something the loader validates.
- **`connectome_fingerprint` now carries the whole vision blob.** `fp["extension"]` is
  `to_jsonable(c._extension)`, which for the candidate includes the entire 102 KB metadata (all 28 DRA
  cell records, all review controls, the left-eye unassigned list). The CSR `md5` is unaffected, but
  every candidate Result's provenance is ~100 KB heavier for it (`banc_motion.json` 540 KB vs
  `malecns_motion.json` 99 KB).
- **Non-ASCII in the new audit.** `docs/audits/banc_candidate_experiment.md` contains 7 × `→`,
  6 × `–` and 1 × `≥`, while `docs/audits/banc_column_reconstruction.md` is pure ASCII and branch 1
  deliberately removed a single em dash from a plot title as a nit. Inconsistent with the style the same
  author had just accepted. (`TODO.md` already has non-ASCII on main, so the `→` there is fine.)
- **1.04 MB of experimental asset ships in the wheel.** `pyproject.toml:64` includes `data/*.csv` and
  `data/*.json`, so `banc_candidate_columns.csv` (936 KB) and `banc_candidate_vision.json` (102 KB) are
  installed for every user, adding ~6 % to `flyverse/data`. Defensible (the experiment must be
  reproducible from an install) but worth a deliberate decision.
- **`build_banc_candidate.py --acceptance` rewrites the shipped metadata after the run.** The committed
  `banc_candidate_vision.json` is therefore *not* the file the house run used: it gains
  `functional_validation` and a changed `functional_gate` string. The branch discloses this and the
  generator guards five keys. I verified the guard works: `columns_sha256`, `base_csr_md5`,
  `base_annotations_sha256`, `input_layer` and `checks` are all **MATCH** between the committed metadata
  and `acceptance.json`'s `provenance.model.vision`, and `functional_validation` is the only added key.
  The CSR md5 is unchanged at `9281a7fb…`, as claimed. Still: a future reader cannot hash-compare the
  shipped asset to the tested one, only these five fields.
- **No null, no replicate, no margin.** `replicates {n: 1, runs: [0]}` for a gate whose two weakest
  decisions separate at 3.3 % and 3.6 %. The runs are deterministic (my CPU rerun is bit-identical), so
  this is a design limit rather than instability, but "8/8" should be read with the margins attached.
- **`probe_loom.py` lost its `# noqa: E402`** when the imports were reordered; ruff's default rules do
  not flag it here, so this is cosmetic only. For the record: `ruff check` reports **zero** findings on
  all five new files and on the changed `probe_motion.py` / `probe_loom.py` / `recover_banc_columns.py` /
  `test_banc_columns.py`, and the findings it reports in `connectome.py` / `interp/common.py` are all on
  lines the branch did not touch. No trailing whitespace, no tabs, no CRLF in any new file.
- **Infra strings: clean in code, and consistent with precedent in docs.** Zero matches for
  `beegfs|<cluster-node>|<cluster-user>|<cluster-domain>|sbatch|slurm|ssh|hostname|IP` in any new or changed `.py`
  (the one hit, `interp/common.py:947`, is the pre-existing `socket.gethostname()`).
  `docs/audits/banc_candidate_experiment.md:38,86,126` do name `<cluster-node>`, `NVIDIA B200` and a
  `<cluster-fs>/neurome/runs/…` path — but main already does this in `docs/audits/interp_atlas.md`,
  `receptor_integration.md`, `round3_integration.md`, `unitary_strength.md` and `docs/media/README.md`,
  so it matches house practice. Not a finding; noting it because the previous review checked it.
- **Every claim in the candidate audit has a named file**, and I spot-checked the load-bearing ones:
  the five Result SHA-256s, `Result.check()`, `export.match_sources` (19-21 files, verified, 0 differ),
  the three MaleCNS cache md5s, the two CSR md5s, the candidate CSV hash, the suite counts, and the
  weights derivation all hold. The required qualifier **"synthetic input layer on a candidate lattice"**
  appears in `flyverse/banc_vision.py:11` as a constant, in the runtime provenance, in the aggregate
  Result's `summary.qualification`, in the audit, in `TODO.md:378` and in the handoff. I found **no**
  behavioural claim beyond it: the audit explicitly declines walking, food-finding, sex-difference and
  anatomical-validation claims, and says so in four separate places.

## 7. Merge map

- `git merge-tree` against main `0b3668f`: **clean for both branches**, zero conflict markers.
- 79ba161 is an ancestor of 715d1ac.
- **Neither branch touches any file with uncommitted work in `D:\Projects\flyverse`**:
  `flyverse/body.py`, `flyverse/senses.py`, `scripts/probe_vnc_drive.py`, `scripts/cx_wedge.py`,
  `tests/test_proprioception.py`, `tests/test_body_cycle.py` — 0 hits each. The untracked files there
  (`docs/audits/compass_ring_mechanism.md`, `glno_relabel.md`, `level_controls.md`,
  `scripts/cx_ring_structure.py`, `scripts/glno_relabel.py`, `scripts/probe_an04b003_single_cell.py`,
  `tests/test_cx_ring_structure.py`) are new paths neither branch creates.
- **`TODO.md` conflict map** (the only shared file at risk): `git diff -U0` shows two hunks —
  a one-line rewrite at **line 369**, and **old lines 376-377 → new 376-381**, all inside section F's
  "A female fly that sees and walks (BANC)" item. Anything that edits TODO.md outside lines
  **369 and 376-377** merges cleanly, including elsewhere in section F.
- `docs/audits/banc_column_reconstruction.md` is modified identically by both branches; main's copy is
  unmodified, so no collision.

---

## Judgement: what the experiment establishes, and what it does not

**What is model.** The retina geometry, the graded optic-lobe dynamics, the LIF brain and the body are
the shipped MaleCNS-derived model, unchanged; applying them to BANC is the point of the exercise and is
fair. The T4/T5 direction-selectivity mechanism — one-column offsets of Mi4/Mi9 and Tm inputs — is a
property of the *biological* BANC graph read through the candidate coordinates, and my identity-shuffle
control shows it is genuinely sensitive to those coordinates: destroy the cell→column assignment while
keeping the geometry and the DSIs collapse from 0.043-0.129 to 0.010-0.036 and 7 of 8 directions go
wrong. That is a real, non-trivial finding, and it is the strongest thing this branch has: **BANC's own
connectivity, arranged by its own reconstructed lattice, produces direction-selective T4 and T5 with the
correct relative geometry.**

**What is construction.** Four things. (1) The photoreceptors do not exist: 5,262 synthetic cells with
negative IDs, six identical copies per column that differ in nothing — same input, same targets, same
weight — so the layer is arithmetically one receptor at 6× gain wearing a cartridge's clothes. (2) Their
strength is a stated assumption that, measured against its own reference, runs about 2× hot. (3) Which
way the eye points is not measured from BANC at all: `orient()` picked the lattice symmetry that makes
BANC's T4a/T4b offsets agree with MaleCNS's, before any probe ran, and my 180°-rotation control shows
that flipping that one discrete choice inverts all eight preferred directions while leaving every DSI
unchanged. So "8/8 correct" is one structure test plus one convention inherited from the male — one bit
of evidence, not eight. (4) The lattice underneath is ~80-85 % exact per cell, and the probe averages
over ~800 cells per subtype, which is precisely the quantity that is blind to that error: a mean offset
direction survives one cell in six being in the wrong column. The gate therefore cannot distinguish an
85 %-correct lattice from a 95 %-correct one, and it never tested the checks that are actually closed
(DRA, mirror), which remain closed and are carried, honestly, inside every Result's provenance.

**So the female fly does not "see" in any sense that adds anatomical knowledge.** It responds to light
through an invented, uniformly complete input layer, on a lattice whose orientation was borrowed from the
male and whose per-cell assignment is right about five times in six, and the one number that invites a
cross-animal reading — 53.3 Hz vs 44.1 Hz, escaping 70 ms earlier — is exposed to a 2× input-gain
asymmetry and a comparator eye that is empty in a third of its columns. What the branch legitimately
establishes is narrower and still worth having: the opt-in is airtight (the biological CSR is
byte-identical including its 212,379 stored zeros, every one of 12,762 synthetic edges is column-local,
pruning restores the original graph through a save/load round trip, and `load(dataset="banc")` is
bit-unchanged); the whole pipeline reproduces from source on an unrelated machine (the pinned lattice CSV
regenerates to the same SHA-256, the B200 acceptance run reproduces bit-for-bit on my CPU); and the BANC
graph, given *a* coherent column map, supports direction-selective motion computation and a
threshold-crossing looming response while its giant-fibre-to-jump pathway does not fire — the last being
a genuine, checkable statement about BANC's VNC reconstruction (DNp01 out-degree 34/29 against MaleCNS's
313/342), not about female flies. Merge it, with F1-F3 fixed, as an engineering result about the
integration path and a negative result about BANC's descending completeness — and keep the qualifier
attached to every number that leaves this branch.
