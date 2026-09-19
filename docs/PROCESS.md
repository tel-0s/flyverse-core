# How a round is run

This is the operating procedure, not the contract. [`INTERP.md`](INTERP.md) section 10 is the contract:
rules 1-30 in 10.4, the reading rules in 10.2, the seven-step procedure in 10.1 and the command
sequence in 10.3. This file says how a round is assembled around them, in the order the work happens,
so that someone who has never run one can run **round 9**. Where a sentence restates a rule it names
the rule and stops; nothing here supersedes `INTERP.md`.

A **round** is: a numbered list of questions in [`../TODO.md`](../TODO.md) section B, one predeclared
batch per question, one audit per batch in [`audits/`](audits/), one independent skeptic pass over the
audits, the corrections applied in place, and one [`NOTES.md`](NOTES.md) entry that says what moved.
Round 8 (2026-09-18) is the worked example throughout: four items, four batches
(`det1`, `cx9`, `suite-inst` + `suite-inst-room`, `plume-go`), four audits, two skeptic passes,
nothing adopted.

---

## 1. Before anything runs: the predeclaration

**The predeclaration is a stamped JSON file, never the prose audit** (`INTERP.md` 10.4 rule 10). The
audit document is written around the stamp and is routinely edited afterwards -- in object round 2 the
JSON was stamped 15:04:04Z, the batch submitted 15:06:37Z and the audit document written 15:07:27Z, and
only the first of those three is the predeclaration. Cite the JSON path and its `stamped_utc`.

What the file must carry before a job is submitted:

| field | what it pins | worked example |
|---|---|---|
| `stamped_utc` | the freeze time, which must precede the submission | `out/det1/predeclared.json` at 2026-09-17T23:31:08Z against runs at 23:32-23:36Z ([`audits/determinism_gate.md`](audits/determinism_gate.md) 2, skeptic claim 9) |
| the arms, seeds, measures and durations | what will be run, so the analysis cannot be chosen afterwards | `out/cx9/predeclared.json`, stamped 2026-09-17T23:43:39Z ([`audits/compass_sign_control.md`](audits/compass_sign_control.md) 2) |
| the comparison / decision rule | the verdict rule, including what counts as a rejection | det1's rule: every array compared with `numpy.array_equal(equal_nan=True)`, every numeric JSON leaf with `==`, no tolerance; a pair `repeats exactly` iff everything is equal (`determinism_gate.md` 2) |
| `batch_sha256`, `jobs_sha256` | the exact plan and job lines | `determinism_gate.md` 5 |
| the frozen source set | the code identity of the runs | det1 froze 73 paths as `source_sha256_lf` (`determinism_gate.md` skeptic claim 1); the plume v5 declaration froze 72 ([`audits/plume_transduced.md`](audits/plume_transduced.md) 7.1) |
| a declaration digest | so the box can be asked to prove it ran the frozen plan | plume v5: declaration SHA-256 `79fba517...`, re-checked by a remote preflight together with 72 source hashes and three cache MD5s (`plume_transduced.md` 7.1) |

Four practices that came out of round 8's own failures, all recorded in `determinism_gate.md` 2 and 5:

* **`bash -n` the generated plan before stamping it.** det1's first freeze was never submitted: the job
  line held an apostrophe inside a double-quoted parameter expansion, which bash rejects.
* **A re-stamp is kept, not folded away.** det1 carries three stamps and says why each exists; the
  comparison rule and the job lines are byte-identical across all three, which is what makes the record
  readable.
* **Regenerating the plan after the runs breaks `batch_sha256`.** det1's `batch.sh` was regenerated with
  `--ship flyverse` so that the committed plan reproduces the *valid* submission; the predeclared
  `batch_sha256` (`553ec9cc...`) is therefore the hash of the plan as frozen and does not match the
  committed file. Say so in the audit rather than re-stamping after results.
* **The frozen set can silently miss a loaded file.** Three paths det1's runs loaded fall outside
  `source_hashes()` and are skipped without a warning (`determinism_gate.md` 5 and its **Withdrawn:**
  note). Check the skip list, and say what it contains.

And the standing dependency rule: **a shared dependency is committed before the batch is submitted**
(`INTERP.md` 10.4 rule 20, 10.1 step 7(b)), or the batch is declared a working-tree batch and the audit
header carries `submit_tree.txt` plus the md5 of every `flyverse/` file that differs from HEAD.
`cluster_run.py` ships the working tree, so an uncommitted file from another task is in the run and in
nobody's history.

---

## 2. Sizing the arms, and sizing the family

**Runs are the replicate unit** -- the cluster job, not the seed and not the batch row
(`INTERP.md` 10.4 rule 1). Round 8's determinism gate is why: no B=6 room repeats on either GPU path,
so two same-code runs at the same seed are two draws (`determinism_gate.md` 3-4; section 4 below).

* `min(n_a, n_b) >= 4`, five when the effect is small. `common.compare` enforces the run count itself
  and returns `underpowered` below four in either arm whatever the exact-U floor says (`INTERP.md`
  10.2, 10.4 rule 1).
* **No arm is planned below 4 runs unless the predeclaration says its rows are magnitudes only**
  (rule 24). Both of the round-4d batches ran 3 seeds per compass arm and were `underpowered` by rule
  before a job was submitted.
* The run floor is **not** the exact-U floor. `common.p_floor(n_a, n_b) = 2 / C(n_a + n_b, n_a)`:
  0.10 at 3 v 3, 0.029 at 4 v 4, **0.0079365 at 5 v 5**, **0.0021645 at 6 v 6** (`INTERP.md` 10.2 and
  its `p_floor` line; the 6 v 6 value is `NOTES.md`, "Session 12, the level-matched control").

**Size the Holm family so it can be satisfied.** With the floor above, the smallest attainable adjusted
p is `p_floor x m`, so:

* **m <= 6 at 5 v 5** (0.0079365 x 6 = 0.047619) and **m <= 23 at 6 v 6** (0.0021645 x 23 = 0.04978)
  (`NOTES.md`, "Session 12, the level-matched control").
* A family declared at **m = 7 at 5 v 5** is unsatisfiable before a single job runs -- 0.0079 x 7 =
  0.0556 -- which is exactly what the level-matched control predeclared, so no row of any family could
  be called (same section).
* **A member whose statistic is constant by construction is not a test.** An all-zero spike median or a
  saturated count returns p = 1.0 through ties and still inflates the denominator; declare such
  quantities as reported magnitudes **outside** the family and size the family from the members that can
  move (`INTERP.md` 10.2). Object round 2 raised its arms from 5 to 6 runs for exactly this reason.

Round 8's `cx9` is the shape to copy: three primaries at 6 v 6, family floor 0.0021645 x 3 = 0.0065,
satisfiable, and all three cleared it (`audits/compass_sign_control.md` 3;
`determinism_gate.md` skeptic claim 5).

Two more sizing rules that bite in practice:

* **Block the comparison family onto one box** (`INTERP.md` 10.4 rule 9): one job per arm makes arm
  collinear with box, and the fleet mixes GPU models. `cluster_run.py --arm-block KEY` keeps every job
  carrying the same `<KEY>_<value>` on one target. A one-job block is a bug (rule 15): the job commands
  carry a family token that is the **same** string on every arm (`fam_<family>`), and the audit prints
  the block-to-box table verbatim.
* **A rate comparison is a two-sample exact Poisson at equal exposure, stated in one direction**, quoted
  beside the run-level `compare` (rule 17). The instrumented room rate-half is the worked example: 110
  take-offs against 101 over 28,800 fly-s each, one-sided exact Poisson p 0.291
  ([`audits/instrumented_suite.md`](audits/instrumented_suite.md) 3).

---

## 3. The verdict vocabulary

Four words, and no others: **`result`**, **`null`**, **`underpowered`**, **`undetermined`**. None of them
is a tool's own invention -- all four come from `flyverse.interp.common.compare` and appear in every
JSON (`INTERP.md` 10.2).

| verdict | when `compare` returns it |
|---|---|
| `result` | abs(z) >= 3 **and** p <= 0.05 **and** `min(n_a, n_b) >= 4` (5 for a small effect) |
| `null` | the comparison ran and did not clear that bar |
| `underpowered` | an arm has fewer than `min_n` runs, **or** `p_floor > alpha` |
| `undetermined` | the reference arm has SD 0 (`null_sd_zero`), so z is NaN or astronomical; read the magnitude `diff` with `p` |

Never "silent" for a population that is firing; never "confirms" for a result that a different
measurement produced. Four traps, all from `INTERP.md` 10.2 unless noted:

1. **The z is orientation-dependent, because `compare` divides by the *reference* arm's SD.** Two round-7
   verdicts reverse when the arms are exchanged -- primary 3 to abs(z) 2.96 and a null, the cx8t transfer
   to z -2.70 and a null -- while the symmetric Welch statistics are +7.26 and +5.05 and U is unchanged
   (`NOTES.md`, "Session 13, the instrumented preset"). Declare the orientation in the predeclaration,
   and report Welch and U beside z.
2. **A zero-SD null's p is a structural constant, not evidence.** A deterministic arm produces
   p = 0.00216 at 6 v 6 and p_holm = 0.026 on every member regardless of sign or size.
3. **A max-over-cells statistic never travels alone**: report the per-body median or mean of the same
   quantity in the same table, and say in the sentence that the headline is a within-run population
   maximum.
4. **Compare on unrounded values; round only at print time**, and print z to one decimal in any table
   whose criterion is abs(z) >= 3 (rule 18).

---

## 4. Runs are the replicate unit, and rooms are quoted as mean +- across-run SD

The deterministic-kernel gate (round 8 item 1, batch `det1`) ran each protocol twice on one pinned GPU
and compared every saved array and metric exactly under a rule frozen before submission
(`audits/determinism_gate.md` 3):

* The `cx_wedge` protocol -- B=1 FlyBrain on the torch path, no world, no optic lobe -- **repeats
  exactly**: 31/31 arrays and 546/546 numeric metrics.
* **No B=6 room repeats on either path.** First differing frame 4 (raw native), 30 (plume native),
  86 (raw torch), 60 (plume torch); after that the two runs differ as two seeds do.

The frozen decision, and the rule for every room number since (`determinism_gate.md` 2 and 4.3):
**rooms are >= 6 draws, the run is the replicate unit, and no room number is quoted to more than its
across-draw SD.** Every room number quoted before the gate stays one draw.

Two consequences for how a round is planned. A protocol that repeats exactly (the B=1 `cx_wedge` shape)
is read as "reproduced the recorded value exactly in n draws on backend X" -- never "bit-identical",
which is a CPU claim (rule 16). A room protocol cannot be read that way, so it is budgeted at six runs
per arm from the start, and the audit's tables carry the across-run SD beside the mean.

---

## 5. Submitting: the `cluster_run.py` pattern

One batch, one call, one named fetch directory. The full worked sequence is `INTERP.md` 10.3; the shape:

```sh
#!/bin/bash
set -o pipefail                    # or set -euo pipefail; a bare `| tee` otherwise exits with tee's status
cmds=()
for a in stim ctrl null; do for s in 0 1 2 3 4 5; do
  cmds+=("mkdir -p out/r9 && python -c 'import torch; assert torch.cuda.is_available()' \
          && python scripts/<probe>.py --arm $a --seed $s --fam fam_<family> --out out/r9/${a}_r$s \
             > out/r9/${a}_r$s.txt 2>&1; st=\$?; tail -4 out/r9/${a}_r$s.txt; exit \$st")
done; done
python scripts/cluster_run.py --name r9 --minutes 40 \
  --node "<cluster-node>" --gpu-ids 4,5,6,7 --ship flyverse,scripts \
  --arm-block fam "${cmds[@]}" --fetch out/r9/ 2>&1 | tee out/r9/client_stdout.txt
```

Every element of that line is a rule:

* **`mkdir -p` in every job line**, and **one client per fetch directory**. Two `cluster_run.py` calls
  fetching into one directory interleave and silently mix arms (rule 3). `--fetch` takes a **named**
  subdirectory, never a bare `out/`.
* **The job line preserves python's exit code.** A line ending `> file 2>&1; tail -4 file` exits with
  *tail's* status, so the python can die mid-write and the job still reports `completed exit 0`. Write
  `&& tail` or `; st=$?; tail; exit $st` (rule 4). `cluster_run.py` warns on the `;` form and submits it
  unchanged.
* **`set -o pipefail` in the generated wrapper.** A `... | tee` tail without it exits with `tee`'s
  status; a generated wrapper that omits it is a recorded defect
  ([`audits/compass_local_recurrence.md`](audits/compass_local_recurrence.md), the submission-wrapper
  section). The round-8 and plume v5 wrappers carry `set -euo pipefail`
  (`audits/plume_transduced.md` skeptic claim 6).
* **`--gpu-ids ID[,ID...]`** takes a pool and gives each job `gpus 1` plus one id, round-robin in command
  order (`determinism_gate.md` 2). A pool smaller than the job count means jobs share a GPU -- det1 ran
  5 jobs over 4 ids -- so a pin is a request, not a guarantee, and the audit says which (skeptic claim
  10).
* **`--ship PATH[,PATH]`** copies the named tracked paths whether or not they differ from `origin/main`.
  It exists because a box checkout behind `origin/main` runs its own stale copy of everything
  `cluster_run.py` did not ship: det1's second submission ran four stale package files and crashed both
  plume pairs while the scheduler listed all five jobs `completed` (`determinism_gate.md` 2). Two
  limits, both recorded: `--ship` only copies, so a file deleted locally but present on the box is never
  removed, and *what* shipped is not recorded in any run's provenance (skeptic claim 10).
* **Infrastructure stays out of the repository.** Targets live in `.cluster.json` and the notes in
  `docs/CLUSTER.md`, both git-ignored; committed plans and audits carry `<cluster-node>`,
  `<cluster-node-2>`, `<cluster-host>`, `<cluster-user>`, `<cluster-fs>`, `<scheduler>` and
  `<rented-box-ip>` (`NOTES.md`, "Session 13, continued: the history rewrite";
  [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) section 3).
* **Fetch consoles before arrays** (rule 12): slim on the box, pull `*.json` and `*.txt` before `*.npz`.
  Only 14 of 84 consoles of one object-round-2 batch survived a hand fetch, which left the
  console-vs-JSON device cross-check unavailable for 70 runs.

---

## 6. Receipts: what counts as "it ran"

**`'<n> job(s), 0 failed'` is necessary and never sufficient** (`INTERP.md` 10.4 rule 4). Round 8 and
session 14 both hit the failure it cannot see: the scheduler listed a **crashed** job as `completed` with
`exit_code None` (`determinism_gate.md` 2 and 4.4), and the plume v5 receipt records 18 completed with
`exit_code null` for every job (`plume_transduced.md` 7.1). Neither round uses "0 failed" as evidence.

Before any analysis:

1. Count the expected artefacts against arms x runs.
2. Open **each** run's console block: its `device cuda` line, the device it really used, and the arm
   spec it echoed. A job that fell back to the CPU analyses fine and means nothing; a job that ran the
   wrong arm is the failure mode "0 failed" cannot see.
3. Run the tool's own batch check (`interp_apply_rotation.py verify-batch <dir>`, or the analyser's
   console-vs-metadata check) and require **0 problems**.
4. Check the loaded-source hashes of every run against the predeclared set, and name what the check
   skips.

Round 8's validity statement is the template (`determinism_gate.md` Report `validation`): *5 jobs, 10/10
consoles device cuda, NVIDIA B200 x 10, pinned ids recorded in the eight room runs, 0 analysis problems,
every loaded source the predeclaration covers at the predeclared hash (three loaded paths fall outside
`source_hashes()`)*.

Two further receipt rules:

* **Fetch the scheduler's own completion receipt as a file into `out/<dir>/`** -- the client console is
  not the record (rule 21). det1 did not, and says so (`determinism_gate.md` 2); plume v5 did, and keeps
  the private receipts beside the run (`plume_transduced.md` 7.1).
* **"md5/sha256-verified" needs a receipt path, or the word is "size-verified"** (rule 19). Plume v5
  re-checked 54 fetched files against remote SHA-256s after transfer and names the receipt
  (`plume_transduced.md` 7.1).

---

## 7. Writing it down

The audit is `audits/<name>.md`, in the shape of the shipped ones: the answer first, then the protocol,
the predeclaration, the results, the reading, the reproduction command, the author's self-review, and
the independent skeptic pass quoted verbatim. Section 0 IS the Report summary, verbatim -- it is never
retyped (rule 26; `determinism_gate.md` section 4 and its Report `summary` are 925 characters each).

Binding rules while writing:

* **Every per-run / per-seed list quoted in prose is emitted by the analysis script into a named file,
  the file and its column are named in the sentence, and the values are pasted from that file**
  (rule 28). A quantity no file carries is not quoted. det1 pastes `out/det1/analysis/runs.csv` columns
  `run, path_m, final_spike_counts, wall_s` (`determinism_gate.md` 3);
  `instrumented_suite.md` pastes `out/suite-inst/analysis/suite_rows.csv` columns `value`, `status`
  (section 2). The rule exists because four of six per-seed lists in the round-4b level-controls audit
  had been built from the published means instead of transcribed (`NOTES.md`, "Session 12, the
  level-matched control").
* **The analysis is regenerated on the complete fetch before any number is written**, and the audit
  pastes the analysis console's run-count line verbatim, per table, with the directory it came from
  (rule 7). An audit still containing an unfilled ALL-CAPS placeholder does not ship: grep for
  `[A-Z_]{6,}` before the commit.
* **The Result records the analysis code that produced it**, and changing that code after a Result is
  emitted obliges a re-run on the same fetched batch and a re-emit of everything derived from it
  (rule 11).
* **A verdict-agreement statement is produced by a script that diffs the two CSVs and prints the count
  and the flipped keys, pasted verbatim** (rule 13).
* **Identify the code by content, not by the commit** (rule 6): quote
  `provenance.compiled_connectome.md5`, `files.effective_weights_md5` and
  `provenance.source_fingerprint`, because every cluster JSON says `flyverse_commit.commit: unknown`.
* **A literature claim carries its source at page / figure level or the word `unverified`**, and a
  novelty claim is preceded by a named search that covers thesis repositories and preprints (rule 29).
  Every bracket edge names its source or is labelled "headroom" (rule 25).
* **A withdrawn claim stays in the record as withdrawn**, in the audit and in `NOTES.md`, never deleted
  (rule 29 iii). The **Withdrawn:** notes in `determinism_gate.md` 2 and 5 and in
  `instrumented_suite.md` 2 are the form.
* **After a history rewrite, quote commit ids through the map** (rule 30; section 5 of
  [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).

---

## 8. The independent skeptic pass

Every round closes with a pass by a **separate agent** (Opus), run **refute-first**: its job is to break
the audit's claims, not to confirm them. CPU only, no cluster job, nothing adopted.

The discipline, as round 8 and session 14 executed it:

* **The verdict line is quoted verbatim** in the audit it judges, in a fenced block, under its own
  heading with the date and the model. `determinism_gate.md`'s begins
  `VERDICT` / `- determinism_gate: mostly sound`; all ten of its claim lines are quoted verbatim beneath
  it.
* **Vocabulary**: `mostly sound`, `refuted`, and the per-claim words the pass itself uses
  (`REPRODUCED`, `CONFIRMED`, `RECORDED`, `REFUTED in part`). The pass's own words, not a paraphrase.
* **CORRECTIONS REQUIRED are applied in place**, in the sections they belong to; where a correction
  replaces a sentence that stated a finding, the original stays marked **Withdrawn:**
  (`determinism_gate.md`, the paragraph opening its skeptic section).
* **The NOT CHECKED list is recorded verbatim** with the round's entry in
  [`audits/receptor_verification.md`](audits/receptor_verification.md), which is **the** record of every
  pass since round 1. A claim outside that list is not covered by the pass.
* **A refutation is a result.** The transduced-plume CPU pass was `refuted (the section-1 mechanism; the
  SNR arithmetic itself is sound and reproduces exactly)`; the defect was fixed, the characterisation
  regenerated, the draft re-frozen and the verdict kept in the record
  (`receptor_verification.md`, the plume entry; `audits/plume_transduced.md` skeptic section).
* **The pass may localise more than the audit did, and the audit is corrected upward.** The det1 skeptic
  established the +-1-spike first difference and the two-process CPU control that excludes the Python
  layer, and the audit now carries both (`determinism_gate.md` 4.2, skeptic claim 2).
* **The reader is told which reading is not licensed.** The rooms skeptic states plainly that "the rooms
  confirm the CPU SNR finding" is **not** licensed, and supplies the wording that is
  (`audits/plume_transduced.md` skeptic claim 7, and section 5 of the same file): the rooms add that the
  measured in-room cue noise is 1.3-2.6x the analytical estimate and that the cue's sign was right in
  only 0.52 +/- 0.10 of samples.

One more standing rule: a round with two workflows names **one owner per file and one closing critic**,
and the critic covers both workflows or says plainly which one it does not (rule 27).

---

## 9. Admitting an instrument

An instrument is a labelled stand-in for physiology the connectome names and the model cannot supply.
The admission rules are [`PRESETS_SPEC.md`](PRESETS_SPEC.md) section 2, all six required: a named gap, a
source for its law or the word `unverified`, the neural boundary only, held edges counted as instruments
too, **its own suite run** (item 5), and a removal condition. Section 5 of the same file admits
program-shaped stand-ins under the owner's extension, each declaring `replaces` in `describe()`.

**Item 5 is the gate a round has to run**: the 29-check suite under `instrumented` is a second column
beside `raw`, at >= 3 draws, and an instrument that moves a row the declared gap does not cover is
rejected -- in either direction. The room rate-half runs at >= 6 runs per arm, and only if the suite
half passes.

Round 8 item 3 is the worked example (`audits/instrumented_suite.md`):

* Protocol: `benchmark.py --sections all` at draw seeds 0, 1, 2 under each preset, six jobs in one
  `cluster_run.py` call, each pinned to one GPU of a four-id pool, `--ship flyverse,scripts` (section 2).
* Frozen rule: a status change is an instrumented draw's status on a row that is not the status of every
  raw draw; rows whose raw draws disagree are `unstable`, reported, not counted; a change on any row
  outside the declared gap row is a rejection either way; the 3 v 3 value comparison is descriptive only
  (section 2).
* Result: **no row changes status in any draw** -- 27/0/2, 26/1/2, 27/0/2 under both presets; the one
  non-uniform row, `taste.MN9_hz`, is raw's own instability and identical under the instruments
  (section 2). The room rate-half: 110 take-offs against 101 over 28,800 fly-s each, one-sided exact
  Poisson p 0.291, run-level null (section 3).
* Verdict: the list is **admissible**, and admissible is all it is -- nothing adopted, `raw` stays the
  default, the afferent's law is still `unverified` (section 5).

Note the two stamps: the first submission lost eight checks per instrumented draw to an adapter gap and
is kept, ignored and unused; the plan and the rule are byte-identical across both stamps (section 2).
That is the pattern for any re-submission.

---

## 10. The checklist for round 9

1. Write the questions in `../TODO.md` section B, one per batch, each naming the audit it will produce.
2. Decide the arms and the family; check `p_floor x m <= 0.05` **before** stamping (section 2).
3. Commit every shared dependency (rule 20). Generate the plan; `bash -n` it; stamp
   `out/<name>/predeclared.json` with the rule, the source hashes and the plan digests (section 1).
4. Submit once, one named fetch directory, exit-code-preserving job lines, the family blocked onto one
   box (section 5).
5. Fetch the scheduler receipt as a file, then the consoles, then the arrays. Check every run's device,
   arm and loaded-source hashes; require 0 analysis problems (section 6).
6. Regenerate the analysis on the complete fetch; emit every per-run list into a named CSV; write the
   audit with section 0 generated from the tables (section 7).
7. Run the independent skeptic pass; quote its verdict verbatim; apply the corrections in place; keep
   every replaced finding as **Withdrawn:**; record the NOT CHECKED list in
   `audits/receptor_verification.md` (section 8).
8. Write the `NOTES.md` entry: what ran, what it found, what is not licensed, what is next as questions.
9. Grep the diff for infrastructure identifiers and for unfilled ALL-CAPS placeholders before the
   commit (section 7; [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) section 3).
