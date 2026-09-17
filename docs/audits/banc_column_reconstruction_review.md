# Review: `feat/banc-vision` @ ebd6f26 (parent 67c76a4 = main)

Read-only review. Worktree `D:\Projects\flyverse-connectome`. All runs CPU-only
(`PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1`). No cluster job. Nothing edited or committed.

## Verdict

**Merge as a diagnostic, with three documentation fixes (B1-B3).**

The code is clean, the integration boundary is real, the suite is green, and nothing under `flyverse/`
changes. The branch's own numbers reproduce, with one exception (B2). What needs correcting is not the
code but three claims in the audit: the DRA gate is unpassable by *any* lattice and so its failure says
nothing about BANC (B1); the BANC diagnostics row is not regenerable from the committed script (B2); and
the 94.38 % FAFB control is a tuned maximum that must not be read as an accuracy estimate for BANC (B3).
My own estimate of the BANC lattice's accuracy is ~80-85 % exact / ~97-98 % within one column, derived
below -- a number the branch does not provide and the one a reader actually needs.

## 1. Diff, isolation, merge

`git diff 67c76a4..ebd6f26 --stat`: 4 files, +997 / -0.

| file | lines |
|---|---:|
| `TODO.md` | +3 |
| `docs/audits/banc_column_reconstruction.md` | +188 (new) |
| `scripts/recover_banc_columns.py` | +653 (new) |
| `tests/test_banc_columns.py` | +153 (new) |

Read every hunk. Confirmed:

- **`flyverse/` untouched.** No model, loader, capability or data-file change. Verified at runtime:
  `banc.has_optic_columns` is `False`; `build_retina(banc)` raises `NotAvailable: dataset banc has no
  optic column map`.
- **Nothing under `out/` or `cache/` committed.** The four files above are the whole commit.
- **No synthetic nodes or edges**; `report.json` carries `synthetic_nodes_added: 0`,
  `synthetic_edges_added: 0`, and `unchanged_fingerprints: {malecns: true, banc: true}`.
- **MaleCNS cache md5s unchanged**, recomputed by me directly off `c.cache_dir`:
  `neurons.parquet c50c598a708b5b373cbaffca7d6a9d82`, `W_post_pre.npz ac131529cebf98decde58d0c227b7954`,
  `sign0_counts.npz bf01d724acf2a1fec8fdb60ef8a9e066`. All three match.
- **No cross-animal seam in the BANC path.** `recover_eye(banc, male, side)` touches only BANC and MaleCNS;
  MaleCNS supplies the T4 direction convention and nothing else. FAFB is loaded only inside the
  `--fafb-control` block. The audit's claim here is accurate.
- **Merge is clean.** `git merge-tree $(git merge-base main feat/banc-vision) main feat/banc-vision` ->
  `merged`, no conflict markers. `TODO.md` in `D:\Projects\flyverse` is unmodified vs 67c76a4, so the
  other agents' uncommitted work cannot collide.

### TODO.md conflict map for the round-4b agent

`git diff -U0` shows a **pure insertion, no deletions**: 3 new lines after old line 371, landing at new
**372-374**, inside section F's "A female fly that sees and walks (BANC)" item. The lines added are the
`Astra's candidate reconstruction is in ...` / `T4 direction check passes, DRA fails ...` /
`blocked on validation (owner decision). scripts/recover_banc_columns.py emits diagnostics only.` block.
A future TODO edit collides only if it rewrites **lines 369-375** of that same F item; any edit elsewhere
in TODO.md (including elsewhere in section F, and the "Do not:" bullet at old 372+) merges cleanly.

## 2. Headline numbers, recomputed

Ran `python scripts/recover_banc_columns.py --fafb-control --plot --out <scratch>`; **exit code 2**, as
designed, in ~70 s. Recomputed every headline from the emitted `report.json` / CSVs with my own code.

**Reproduces exactly:**

| claim | audit | mine |
|---|---|---|
| Mi1 seeds, right eye | 878 | 878 |
| main component / candidate sites | 877 | 877 (`component_sizes [877, 1]`) |
| mutual neighbour edges | 2,374 | 2,374 |
| max assignment displacement | 1.323 | 1.3232 |
| unassigned seed | `720575941689026011` | same |
| left eye | 321/560, 114 components | 321/560, 114 components |
| DRA proxy | 28 candidates, 11 mapped, 7 within two rows, 5 in top height quintile | identical |
| T4 cosines all >= 0.9996 | a .9999939 b .9996495 c .9997653 d .9996363 | a .9999846 b .9996227 c .9996896 d .9996363 |
| FAFB control Mi1 | 739/783 = **94.38 %** exact, 777/783 = **99.23 %** within one | 94.3806 % / 99.2337 %, identical counts |
| FAFB control, all 13 types | table of 13 rows | every cell matches |
| BANC type-assignment table | 12 rows (L1 874/865 ... T4d 804/762) | every cell matches |
| focused tests | 13 passed | 13 passed |
| full CPU suite | 381 passed / 19 skipped / 215 subtests | 381 passed / **40** skipped / 215 subtests, exit 0 |

The whole-population FAFB Mi1 figure is 739/796 = 92.84 %, as the audit also says.

Skip-count delta: the repo has 25 `gpu`-marked and 9 `data`-marked tests; a CPU-only run skips more than
Astra reports. Passes and subtests match exactly, so this is an environment difference, not a regression.

## 3. Blocking issues

### B1. The DRA gate cannot be passed by any lattice, so its failure is not evidence about BANC

`docs/audits/banc_column_reconstruction.md:126` reports check (i) as **Fail**, 7 of 28 within two rows,
and at :139 says "FAFB's known strict-rim expected failure is not an exemption for a new BANC map". That
framing treats the criterion as a standard BANC failed to meet. It is not a meetable standard.

I took the branch's own proxy definition (`R7_unclear -> Dm-DRA1`, `R8_unclear -> Dm-DRA2`, any count > 0)
and scored it **against FAFB's published column map**, i.e. under a lattice that is correct by definition:

- proxy set under FAFB's **published** lattice: **80/90 within two rows (88.9 %)** -- the gate needs 90/90.
- FAFB's **real community-labelled** DRA R7/R8 under the published lattice: **60/64 (93.8 %)** -- still not 100 %.
- the branch's own FAFB **reconstruction** scores 75/87 (`report.json` ->
  `fafb_control.reconstruction.dra`), which the audit never mentions.

So check (i) as implemented returns "fail" for a perfect map. A gate with no passing configuration carries
no information about the thing it gates. This is consistent with the pre-existing finding in
`docs/audits/connectome_backends_review.md` section 5 (FAFB: 118/126 within two rows, recorded as an
expected failure), but the new audit does not connect the two, and it does not print its own control's
DRA numbers even though they are in the JSON it emits.

I also validated the **proxy itself** against FAFB's real labels (the branch never does): precision
**61/90 = 67.8 %**, recall **61/64 = 95.3 %**. About a third of the selected cells are not DRA cells.

And the BANC-vs-FAFB comparison the audit implies is statistically weak: 7/11 vs 75/87 gives Fisher
p = 0.078; 7/11 vs the 80/90 ground-truth rate gives p = 0.044. The *strong* signal is elsewhere and is
not about the lattice at all: only **11 of 28** BANC proxy cells received coordinates vs **87 of 90** in
FAFB (p = 1.3e-10), and BANC's right eye contains only **11 Dm-DRA1 + 3 Dm-DRA2** cells (FAFB: 23 + 12)
carrying **3-20** synapses each (FAFB: 5-111). Most of the DRA "failure" is BANC's incomplete optic-lobe
annotation, which TODO.md's own "do not treat BANC's optic lobes as complete" bullet already warns about.

**Fix:** state in the audit that (i) is unpassable as written (with the 80/90 ground-truth number), report
the FAFB control's own DRA row, report the proxy's 68 % precision / 95 % recall, and separate the
*mapping-rate* failure (11/28, a data-completeness fact) from the *rim* failure (7/11, n too small to
conclude anything). The gate can stay as it is; the reading of it must change.

### B2. The audit's BANC diagnostics row is not regenerable from the committed script

| quantity | audit | my run (3 identical repeats) |
|---|---:|---:|
| collisions before injectivity, right | 65 | **64** |
| seeds displaced, right | 139 | **136** |
| neighbour-edge step agreement, right | 82.01 % | **82.39 %** |
| collisions, left | 116 | **114** |
| seeds displaced, left | 179 | **175** |
| left T4c / T4d cosine | -0.144 / -0.981 | **-0.973 / -0.819** |

`reconstruct()` is deterministic in this environment -- three repeats in one process give 64/136/82.39 %
every time -- and **every FAFB control number reproduces to the digit**. So this is not run-to-run noise;
the BANC rows were produced by a different script version, a different BANC cache, or a different
numeric stack. The cause is that the continuous solve is knife-edge at the rounding step: 12 IRLS rounds
of `lsqr` at `atol=btol=1e-10` with no `iter_lim`, then `np.rint`, so a change in the last bits flips a
handful of seeds. `report.json` records package versions and the raw-file SHA-256, which is good practice,
but the audit's table predates them.

**Fix:** regenerate the audit's right- and left-eye tables from the committed script and record the
`report.json` fingerprint block (or the raw BANC `neurons.csv.gz` SHA-256) beside them.

### B3. The FAFB control's 94.38 % is a tuned maximum and must not be transferred to BANC

`docs/audits/banc_column_reconstruction.md:155` does disclose "a development control, not an untouched
test set: exploratory method choices were inspected against FAFB". The disclosure is correct but far too
quiet for how load-bearing it is. Sweeping the single most important parameter
(`scripts/recover_banc_columns.py:55`, `"neighbours": 6`) against FAFB's labels:

| `neighbours` | 3 | 4 | 5 | **6** | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| FAFB Mi1 exact | 2.43 % | 3.47 % | 78.54 % | **94.38 %** | 67.82 % | 20.82 % |
| within one column | 5.31 % | 11.71 % | 97.57 % | **99.23 %** | 97.32 % | 56.07 % |

The shipped value sits on a knife-edge peak, 16 points above its nearest neighbour and 27 below on the
other side. In fairness: 6 is also the a priori correct number for a hexagonal lattice, so the choice is
anatomically principled rather than purely fitted -- I do not think it was reverse-engineered. But there
is no margin, and nothing establishes that 6 is the right value for BANC's noisier graph. The branch is a
single squashed commit, so the diff contains no development history; the audit's sentence is the only
evidence either way.

**One concern I checked and cleared.** The scoring alignment in `compare_truth` picks the symmetry and
integer translation that maximise exact agreement with the very labels it then scores, which looked like
fitting on the test set. It is not, materially:

- the winning symmetry scores 739 modal-exact, the runner-up 30 -- a landslide, not a close call;
- fitting the alignment on a random 50 % of Mi1 and scoring the held-out 50 %, 20 repeats: held-out exact
  **94.45 %** (min 93.11, max 95.92), within one 99.20 %, and the identical alignment is recovered
  **20/20** times;
- the 12 non-Mi1 types never enter the alignment fit at all and score **92.00 % exact / 98.85 % within
  one** over 8,526 cells.

So the alignment method is fair and 94.38 % is a real measurement of the method **on FAFB, at its tuned
optimum**. That is the caveat that has to travel with the number.

**Fix:** put the `neighbours` sweep (or an explicit "this is a tuned maximum on the control") in the audit,
next to the 94.38 %, and add the BANC-specific accuracy estimate below.

## 4. How good is the BANC lattice? (a number the branch does not provide)

Nothing in the branch tells a reader how accurate the BANC candidate is likely to be, and the 94.38 %
control invites the wrong inference. I estimated it with a label-free statistic applied identically to
both animals: split the T4 partner population in half at random, reconstruct twice, and compare the two
lattices to each other up to symmetry and translation.

| dataset, full T4 budget | split-half exact | split-half within one |
|---|---:|---:|
| FAFB right | 48.8 - 52.5 % | 89.4 - 91.1 % |
| **BANC right** | **24.4 %** (sd 3.2) | **75.7 %** |

Calibrating that statistic against truth on FAFB, by shrinking its partner budget:

| FAFB budget | split-half exact / within one | accuracy vs published map |
|---|---|---|
| 100 % | 48.8 % / 91.1 % | 94.4 % / 99.2 % |
| 80 % | 23.6 % / 71.6 % | **83.3 % / 98.0 %** |
| 65 % | 8.1 % / 37.5 % | 76.6 % / 96.9 % |
| 50 % | 8.0 % / 36.1 % | 59.2 % / 93.7 % |

BANC's full-budget self-consistency (24.4 % / 75.7 %) sits at FAFB's **80 %** budget, where FAFB's true
accuracy is **83.3 % exact / 98.0 % within one column**. So a matched-statistic estimate for the BANC
candidate is roughly **80-85 % exact, ~97-98 % within one column** -- roughly one cell in six in the
wrong column, not one in eighteen.

Two independent indicators agree:

- neighbour-edge step agreement after assignment: BANC **82.4 %** vs FAFB control **88.6 %**;
- T4 offset **magnitude** against the same MaleCNS reference: BANC 0.86-0.99 of reference (T4a .985,
  T4b .873, T4c .860, T4d .886), the FAFB reconstruction 0.98-1.11. The gated statistic is a cosine, so
  this systematic shrinkage -- the signature of centroid regression under noisy column assignment -- is
  invisible to check (iii) and is not reported anywhere.

Caveat on my estimate: it assumes the difference between the animals is an effective-evidence-budget
difference. The nonzero connectivity structure is in fact comparably dense (12.5 vs 13.2 nonzero T4
partners per Mi1), so this is not a uniform synapse-count rescaling -- which would cancel exactly in
`w.T @ w` -- but genuinely less consistent partner sets. BANC could also be worse for structural reasons
the calibration cannot see.

## 5. Is the T4 holdout real?

**Yes, but narrow -- and it can be made complete for free.** The audit's own caveat
(`scripts/recover_banc_columns.py:313`, "all T4s contributed unordered shared partners") is honest. I
measured how much the c/d holdout actually decides by scoring all twelve symmetries:

| rank by a/b fit | T4a | T4b | T4c | T4d |
|---|---:|---:|---:|---:|
| chosen | +1.0000 | +0.9996 | **+0.9997** | **+0.9996** |
| runner-up | +0.8943 | +0.9187 | **-0.9484** | **-0.9966** |

The runner-up fits a/b at 0.89/0.92 -- respectable -- and inverts c/d. So the held-out subtypes are what
rules out the reflected lattice; the check is not vacuous. The left eye demonstrates the same from the
failure side (c -0.973 in my run). But the check only discriminates among 12 discrete orientations; it
cannot say anything about per-cell placement, which the audit states correctly at :151.

It can be made a **true** holdout at a known cost. On FAFB, building the neighbour graph from partner
populations that exclude the check's own cells:

| partner population | FAFB Mi1 exact | within one |
|---|---:|---:|
| T4a-d (as shipped) | 94.38 % | 99.23 % |
| **Tm3 only (no T4 at all)** | 75.29 % | 95.56 % |
| Tm1/2/3/4/9/20/21 | 75.29 % | 95.56 % |
| **T4a + T4b only** | 74.81 % | 96.80 % |
| C2 + C3 | 5.49 % | 22.15 % |
| Tm1 / Tm2 / Tm9 / T1 alone | fail or degenerate | -- |

Either of the two bolded options makes check (iii) fully independent for ~20 points of exact accuracy.
Worth doing if the reconstruction is ever revisited; not a reason to hold the merge.

## 6. Is the machine gate honest?

Mostly yes, with one framing problem.

- **No flag can silence it.** `argparse` defines exactly `--out`, `--fafb-control`, `--plot`
  (`scripts/recover_banc_columns.py:535-537`). There is no threshold override, no `--force`, no
  extension flag, no way to load the CSVs as a backend annotation.
- **Failing checks are printed and recorded.** One stdout line with all four statuses, plus
  `report.json["checks"]` with reasons, per-cell DRA evidence, output hashes and the generator SHA-256.
  `gate()` (line 433) requires all four to be literally `"pass"`; `tests/test_banc_columns.py` pins that
  `"fail"`, `"unavailable"` and `"measured"` all fail, and that a *missing* check fails.
- **`--out` must be an empty directory**, so a run cannot overwrite previous evidence or a cache path.
- **But exit 0 is unreachable.** `i_dra_rim` is built as `"fail" if ... else "unavailable"`
  (`scripts/recover_banc_columns.py:598-604`) and `ii_lr_mirror` is a literal `{"status": "unavailable"}`
  (`:605-608`). Neither can ever be `"pass"`, so `gate()` can never return `True` and the script always
  exits 2. That is fail-closed, which is the right direction, and I would not change it. But the module
  docstring at `:9-10` ("A zero exit is reserved for all four gates passing") and the audit at :20
  ("Exit code 2 is the failed anatomical gate") present it as a live measurement. The exit code is a
  constant. Say so, so nobody later reads a future exit 2 as new information.

## 7. Nits

- `tests/test_banc_columns.py:1-17` has no `sys.path.insert(0, ROOT)` bootstrap. It is the only test in
  the repo that does `from scripts.… import …`; every other one uses
  `sys.path.insert(0, parents[1]/'scripts')` (`tests/test_batch_sim.py:235`) or
  `importlib.util.spec_from_file_location` (`tests/test_cluster_run.py:27`), and
  `tests/test_bit_identity.py:42` inserts the root explicitly. Consequence: run pytest from outside the
  repo root and the module dies at collection with `ModuleNotFoundError: No module named 'flyverse'`,
  while `test_bit_identity.py` collects fine from the same cwd.
- `scripts/recover_banc_columns.py:527`: `fig.suptitle("DIAGNOSTIC ONLY — anatomical gate has not
  passed")` is the one non-ASCII character in the branch (em dash). The audit prose is correctly `--`
  throughout, and the other non-ASCII in `scripts/` is box-drawing or proper names (`Özel`), so this is
  the only stylistic deviation. No infra strings anywhere (the single grep hit, "no GPU or house batch"
  at audit :164, is a negative statement).
- `scripts/recover_banc_columns.py:639`: `assert all(report["unchanged_fingerprints"].values())` is a
  bare `assert` for a safety property -- stripped under `python -O`. Prefer an explicit raise.
- `direct_dra_labels()` (`:483`) guards `path.exists()` but not the four column names it passes to
  `usecols`; a release that renames any of them kills the whole run with a pandas error instead of
  reporting `unavailable`, which is the function's declared purpose.
- `main` calls `args.out.mkdir(parents=True, exist_ok=True)` at `:540` *before* the non-empty check, so a
  rejected invocation leaves an empty directory behind.
- `docs/audits/banc_column_reconstruction.md:61,63`: the two literature citations are bare PMC URLs with
  no author, year or title. The pairing they support (DRA R7 -> Dm-DRA1, DRA R8 -> Dm-DRA2) is the
  standard one and I believe it is right, but a reader cannot check it without fetching a URL. Name them.
- The DRA proxy selects on `counts > 0` (`:341`) -- a single synapse qualifies. The spec's own
  "do not mix counts across releases unscaled" rule bites here: the same threshold picks 90 cells in FAFB
  (5-111 synapses) and 28 in BANC (3-20). A per-release scaled threshold would make the two denominators
  comparable.

## 8. Is the candidate lattice usable for anything now?

It is more than a negative result and much less than a column map. At an estimated ~80-85 % exact and
~97-98 % within one column, it is usable wherever a *neighbourhood* is enough and an *identity* is not:
choosing which BANC cells a human should examine next, targeting the dorsal boundary where the audit
itself says the next evidence is needed, providing the initial guess for a registration against an
independently registered BANC scaffold, and -- concretely -- the emitted per-seed
`seed_displaced_for_injectivity` / `seed_assignment_distance` columns, the 28 DRA proxy body IDs and the
one orphan seed are exactly the right worklist for that examination. The orientation is the strongest
transferable piece: the c/d holdout discriminates the reflected lattice by cosine +0.9997 against -0.948,
so a future anchor set only has to confirm a chirality that is already picked out rather than search
twelve of them. It is *not* usable for anything that needs a cell's identity -- no retina, no per-column
photoreceptor mapping, no T4/T5 wiring, no column-by-column cross-connectome comparison -- because roughly
one cell in six would sit in the wrong column and the errors concentrate exactly where a retina cares
most: the rim (7/11 proxy cells within two rows), the boundary, and the T4 home columns, which score only
82-84 % exact even in the FAFB control where the seeds score 94 %. The branch's decision to ship it as
diagnostics behind a fail-closed gate, with `has_optic_columns` still false, is the right call and the
reason this should merge.
