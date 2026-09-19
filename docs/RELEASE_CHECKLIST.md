# Release checklist: flipping the repository public

The concrete list, in the order it should be worked. `[x]` is done and says where the receipt is;
`[ ]` is owed and says who decides. The release blockers in [`../TODO.md`](../TODO.md) section A are
the history of this list; this file is the list as it stands.

Nothing here is a claim about the model. What the model does, and at which commit it was measured, is
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md); how a round that produces such a number is run is
[`PROCESS.md`](PROCESS.md).

**Receipts in this file marked "this pass" were produced by the release pass of 2026-09-19 at
`bffbb0a`, on the author's desktop, CPU only.** They are receipts of the checklist, not audit numbers;
every model number in this file is cited to its audit.

---

## 1. History, and the one decision still owed

- [x] **The infrastructure scrub was done over the whole history, not the tip** (2026-09-17):
      `git filter-repo` over all 161 commits with a replacement map -- hostnames, the cluster user, the
      shared-filesystem root, the scheduler's name, three rented-box IPs and the workstation home path
      replaced by `<cluster-host>`, `<cluster-node>`, `<cluster-node-2>`, `<cluster-user>`,
      `<cluster-fs>`, `<scheduler>`, `<rented-box-ip>`, `<workstation-home>` -- applied to blobs **and**
      commit messages, verified by a full-history grep returning nothing. Every commit id changed, the
      ten remote branches were force-pushed, and a pre-rewrite bundle is kept outside the repository
      ([`NOTES.md`](NOTES.md), "Session 13, continued: the history rewrite"; `../TODO.md` A).
- [x] **The rewrite's commit map is committed** as
      [`audits/commit_map_2026-09-17.json`](audits/commit_map_2026-09-17.json) (161 entries, 93 renamed,
      68 unchanged), and analysers resolve recorded ids through
      `flyverse.interp.common.resolve_commit` (`INTERP.md` 10.4 rule 30;
      `tests/test_commit_map.py`). See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) section 5.
- [ ] **Decide the GitHub object purge, or push fresh.** A force-push does not delete the pre-rewrite
      objects on GitHub: they stay reachable by SHA until GitHub's own garbage collection, and any
      cached view, fork or pull request built on them keeps them alive. `../TODO.md` A records the
      intent ("ask support to purge before the public flip"). Two ways to close it, and the owner picks
      one **before** the repository is made public:
      1. Ask GitHub Support to run a GC on the repository, then spot-check that a known pre-rewrite id
         from the commit map (any key whose value differs from it) 404s at
         `https://github.com/<owner>/<repo>/commit/<old id>`; or
      2. **push fresh**: delete the remote repository and create it again from the rewritten history, so
         no pre-rewrite object was ever pushed to the new one. This also drops stars, issues and
         existing forks -- which is the trade-off to weigh, not a detail.
      Whichever is chosen, record it in `NOTES.md` with the date, and keep the pre-rewrite bundle.
- [ ] **Decide the commit-author identity.** Every one of the 196 commits carries one author address,
      a personal e-mail account (this pass: `git log --all --format=%ae` gives a single distinct
      address). That is not an infrastructure identifier and the scrub deliberately left it, but it
      becomes public with the repository. Options: leave it, set a GitHub `noreply` address and rewrite
      the author fields once more (a second commit map, by rule 30), or add a `.mailmap`. Owner's call;
      a second rewrite has to be made before the public flip, not after, because it changes every id
      again.

---

## 2. Files that must stay ignored

These are ignored today and the check is that they **stay** ignored; none of them is edited.

| path | why |
|---|---|
| `docs/CLUSTER.md` | the cluster's own notes: targets, hosts, quotas |
| `.cluster.json` | `cluster_run.py`'s target list, with ssh destinations and ports |
| `docs/HANDOFF_*.md` | per-round hand-off files; they name boxes and directories (`INTERP.md` 10.4 rule 27) |
| `out/cx5`, `out/cx6`, `out/cx7`, `out/objr3sd*`, `out/vncd4-7` `predeclared.json` | five frozen declaration families carrying `host` / `ssh` fields. Frozen run records are **never edited** (rule 30 (ii)), so the only correct handling is that they remain ignored (`../TODO.md` A, the code skeptic's list) |
| `cache/`, `data/external/`, `out/` (except the tracked files below) | ~200 MB of compiled cache, other people's expression tables, and run output |

- [x] **The `.gitignore` entries exist** for all of the above.
- [x] **The tracked exceptions under `out/` are deliberate and clean**: 76 files -- the committed plans,
      declarations and analysis CSVs of the compass stand-in, `cx8` / `cx8r` / `cx8t` / `cx9`, `det1`,
      `plume-go`, `suite-inst` and `suite-inst-room`. This pass: the only `host` / `ssh` strings in them
      are the **excluded-key names** inside det1's comparison rule
      (`out/det1/predeclared.json`, `out/det1/analysis/summary.json`), not values.
- [ ] **Re-run the check on the day of the flip**, because a new round adds new `out/` families:
      `git status --ignored --short` and confirm every `predeclared.json` outside the tracked list is
      still ignored.

---

## 3. The identifier grep: over the tree, and over `git log -p`

Run **both**. The tree grep catches what is shipped; the history grep catches what a rewrite missed.

```sh
# 1. the working tree (tracked files only; -I skips binaries)
git grep -nIE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\b([0-9]{1,3}\.){3}[0-9]{1,3}\b|C:\\Users|/mnt/|/home/[a-z]|beegfs|slurm|sbatch|\bssh [a-z0-9._-]+@' -- . ':!uv.lock'

# 2. the whole history, blobs and commit messages
git log -p --all > hist.txt
grep -nE '<the same pattern>' hist.txt
rm hist.txt
```

Then **read every hit**. The pattern is deliberately wide, so a clean tree still returns matches. The
known-benign set, confirmed this pass at `bffbb0a`:

| hit | why it is benign |
|---|---|
| `root@1.2.3.4` in `scripts/cluster_run.py` | the documented example of a target's `ssh` field, not a real host |
| `127.0.0.1` in `cluster_run.py`, `box_status.py` | loopback, the local end of the ssh tunnel |
| `root@1.2.3.4` / `127.0.0.1` in `tests/test_cluster_run.py` | the same documented fixture target and loopback, in the mocked cluster tests |
| dotted quads in `uv.lock` | package version strings (excluded by the pathspec above) |
| `p@np.exp(...)`, `profile@np.exp(...)` | numpy matrix multiplication |
| `t@example.invalid` in `tests/test_commit_map.py` | the throwaway repository's fixture identity |
| `beegfs` / `slurm` / `sbatch` / `/mnt/` in `audits/*.md` and `tests/test_cx_velocity_route.py` | the words appear only **inside descriptions of this grep** and in an assertion that they are absent |

This pass's result: **no hit outside that set, in the tree or in the full history**, except the commit
author address of section 1.

**The expected count, so the re-run is pass / fail.** On the tree this checklist ships in -- this file
and its two companions present -- the tree grep returns **40 hits across 13 files**:
`tests/test_cluster_run.py` 14, `scripts/cluster_run.py` 7, this file's own description of the grep 7,
`tests/test_commit_map.py` 2, `docs/audits/connectome_backends_followups_review.md` 2, and one each in
`scripts/box_status.py`, `scripts/compass_driver_analyse.py`, `scripts/probe_compass_driver.py`,
`tests/test_cx_velocity_route.py` and the four remaining audits. A re-run that returns that count over
that file list closes the check; a different count, or a file not on that list, is read before the flip.

The generic ssh / scp machinery in `cluster_run.py`, `fetch_run.py`, `box_status.py` and
`object_round3_export.py` is code, not an identifier, and stays.

- [x] Tree grep clean (this pass).
- [x] History grep clean (this pass, over all 196 commits).
- [ ] Re-run both immediately before the flip, and once more after any further rewrite.
- [ ] Grep the tree for unfilled ALL-CAPS placeholders as well (`[A-Z_]{6,}`), which is the audit rule
      (`INTERP.md` 10.4 rule 7) and catches a half-written audit as effectively as a leaked host.

---

## 4. Tests, bit-identity and the cache fingerprints

- [x] **The CPU test subset is green.** This pass, at `bffbb0a`:
      `python -m pytest -m "not gpu and not data and not cluster" -q` ->
      **520 passed, 34 skipped, 30 deselected, 220 subtests passed in 203.55 s**
      (`PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1`, `SDL_VIDEODRIVER=dummy`). The previously
      recorded figure is 364 passed / 7 skipped / 29 deselected at `28e862f`
      ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) section 8); the suite has grown since.
- [x] **The raw bit-identity gate passes.** This pass:
      `CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_bit_identity.py -q` ->
      **6 passed, 10 subtests passed**. That file pins the default path against a recorded golden and
      also asserts that `preset="raw"` and `preset="instrumented", instruments=[]` reproduce it exactly
      ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) 4.4; [`PRESETS_SPEC.md`](PRESETS_SPEC.md) 1).
      **Regenerating that golden is an owner decision, not a test fix** -- a failure means the simulated
      model moved.
- [ ] **Re-check the three cache md5s on the release machine** and confirm they still read
      `neurons.parquet c50c598a708b5b373cbaffca7d6a9d82`,
      `W_post_pre.npz ac131529cebf98decde58d0c227b7954`,
      `sign0_counts.npz bf01d724acf2a1fec8fdb60ef8a9e066`, with compiled-CSR md5
      `ef23cc27bea13be7f6a96f3c04fd3737` and `sum|W|` 121,460,584
      ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) section 2; commands in its section 7.1). The cache is
      git-ignored, so this is a check that a fresh rebuild from the manifest reproduces the published
      fingerprint, which is the only thing a reader can check.
- [ ] **A clean-clone run-through on CUDA is still owed** (`../TODO.md` A, packaging): `pip install -e .`
      from a fresh clone, `python scripts/fetch_data.py --malecns`, rebuild, `python scripts/room_demo.py`
      on CPU and on CUDA. `uv.lock` exists; the CI job installs CPU torch explicitly.

---

## 5. Lint and CI

- [x] **CI runs the data-free CPU subset** on push and pull request to `main`
      ([`../.github/workflows/ci.yml`](../.github/workflows/ci.yml)): Python 3.12, CPU torch from the
      PyTorch CPU index, `FLYVERSE_DATA` pointed at a non-existent path so nothing can silently find a
      stale dataset, headless SDL, then `pytest -m "not gpu and not data and not cluster"`.
- [ ] **Decide what the ruff step is for.** It runs `ruff check flyverse scripts tests --select
      E9,F63,F7,F82` with `continue-on-error: true` -- it reports, it does not gate. This pass, with
      ruff 0.16.7, it finds **14 `F821` undefined names in three scripts**: `scripts/cx_wedge.py` (8),
      `scripts/cx_shift.py` (4), `scripts/compass_driver_trace.py` (2), all of them `fb` or `sim` in
      what look like nested closures. `../TODO.md` A already asks whether those paths are dead or rely
      on an enclosing scope. Close it one of two ways before the flip: fix or delete the paths and drop
      `continue-on-error`, or keep the step advisory and say in the CI file why 14 known hits are
      tolerated. A public repository whose first CI annotation is 14 undefined names invites the wrong
      first impression.

---

## 6. Data: `fetch_data.py` and the manifest

- [x] **The repository ships no connectome data and no third-party tables.**
      `flyverse/data/manifest.json` records the URL, SHA-256, size, citation and licence of every file
      `scripts/fetch_data.py` can fetch -- the four MaleCNS files, the FAFB and BANC releases, and 55
      external expression / typing files ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) 3.1-3.4).
- [x] **`fetch_data.py` works from the manifest.** This pass: `python scripts/fetch_data.py --list`
      enumerates every source with its licence line and reports which files are present locally.
      Downloads stream to a `.part` file and are renamed only after the recorded SHA-256 matches.
- [ ] **Run `python scripts/fetch_data.py --verify` on the release machine** and paste the result into
      the release note: it re-hashes everything present and reports mismatches, which is the check a
      reader will repeat.
- [ ] **Re-check the two licence statements against the download pages on the day of the flip**: the
      MaleCNS terms (recorded as "Janelia FlyEM data terms, CC BY 4.0 at the time of writing -- check")
      and the Codex releases' CC BY 4.0.

---

## 7. Citation, licence and version

- [x] **`LICENSE`**: MIT, for the code.
- [x] **[`../CITATION.cff`](../CITATION.cff)** exists and validates, cites the MaleCNS v1.0 release, the
      FAFB v783 and BANC v888 releases and Shiu et al. 2024, and names the entity author `tel0s`.
- [ ] **Close the three TODOs inside `CITATION.cff`** -- they are written into the file and will be read
      by anyone who opens it: confirm and add the DOI of the MaleCNS release paper (the manifest records
      the PII `S0092-8674(26)00942-6`, which has **not** been checked against a registered DOI), confirm
      the data licence, and confirm the canonical author list for the release. Either fill them or
      restate them as a note; do not ship a `TODO:` in the citation file.
- [ ] **Decide the version.** `pyproject.toml` and `CITATION.cff` both say `0.1.0`, and `CITATION.cff`
      says `date-released: 2026-09-13`, which is now stale. Pick the release version, set it in both
      files in one commit, update `date-released`, and tag it -- a citable release wants a tag, and the
      tag is what a DOI service would archive.

---

## 8. Media

- [x] **`docs/media/` holds the committed demo media** -- `loom.gif` / `loom.mp4`,
      `wind_apple.gif` / `.mp4`, `toolkit_loom_gf.png`, `compass_standin.png` -- with every command
      line, commit and device recorded in [`media/README.md`](media/README.md) and the generator in
      `scripts/make_demo_media.py` (`../TODO.md` A, demo media). The `.gitignore` carve-out
      (`!docs/media/*.gif`, `!docs/media/*.mp4`) keeps every other `.gif` / `.mp4` ignored.
- [x] **The captions are honest**: the wind/apple clip is captioned so that no wind orientation and no
      feeding is claimed from it (`../TODO.md` A).
- [ ] **Check that the README's images render from the public URL** once the repository is public
      (relative paths work, but a GIF over a certain size is throttled in the GitHub renderer), and that
      `media/README.md` carries no path from the machine they were rendered on.

---

## 9. The two documents the README must link

Two documents are being written for the release and are **not in the tree yet**. The README's "Where
things are" table and its reproducibility section must link them, and the links must resolve before the
flip:

- [ ] **`docs/PREDICTIONS.md`** -- the falsifiable predictions the model makes, each with the assay that
      would refute it. Must use the verdict vocabulary of [`PROCESS.md`](PROCESS.md) section 3
      (`result` / `null` / `underpowered` / `undetermined`) and cite its audit per claim.
- [ ] **`docs/RESULTS.md`** -- the results as a reader should quote them, drawn from the audits. Two
      constraints it inherits, both recorded:
      * room numbers are quoted as mean +- across-run SD with the run as the replicate unit, and every
        room number produced before the determinism gate stays **one draw**
        ([`audits/determinism_gate.md`](audits/determinism_gate.md) 4.3);
      * the transduced-plume rooms are a **descriptive room observation** -- not a significance test, not
        an SNR measurement, not a claim about flies -- and they do **not** confirm the CPU SNR finding
        ([`audits/plume_transduced.md`](audits/plume_transduced.md) 7.3 and skeptic claim 7).
- [ ] **Add both to the README's document table** in the same commit that adds the files, so the
      repository never has a dangling link; a link check over the tree is the last item of section 10.

---

## 10. The last pass, on the day of the flip

Run these in order, and keep the console:

1. `git status --ignored --short` -- nothing ignored has become tracked (section 2).
2. The two identifier greps, and the ALL-CAPS placeholder grep (section 3).
3. `python -m pytest -m "not gpu and not data and not cluster" -q` and
   `CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_bit_identity.py -q` (section 4).
4. `python -m flyverse.connectome` and `python scripts/hash_weights.py` -- the cache md5s and the
   compiled-W md5 against [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) section 2 (section 4).
5. `python scripts/fetch_data.py --verify` (section 6).
6. `ruff check flyverse scripts tests --select E9,F63,F7,F82` -- expected hits only (section 5).
7. A relative-link check over every Markdown file in `docs/` and `README.md`: every relative link
   resolves in the tree, including the two new documents of section 9.
8. Confirm the GitHub object decision of section 1 is made and recorded in `NOTES.md`, then flip.
