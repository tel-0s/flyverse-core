# Review of `feat/instruments` against `PRESETS_SPEC.md`

Astra, 2026-09-15. Reviewed `a41d0f2..6d1c501`, then merged the two authorized documentation closeouts
(`ebfabbf`) into the branch before fixing and testing. **Verdict: merge with fixes, applied in this review.**
The implementation preserves the raw model and puts the optional signed afferent in the existing body-to-sense
boundary. The original batch checker was too permissive for the promised experiment, and the public token path
could bypass the preset/provenance contract. Both are corrected before any cx8 submission.

This is Astra's implementation review and self-review of the fixes, not an independent Opus pass. No GPU jobs
were submitted during review. The scientific result and independent skeptic pass remain pending.
**Post-merge correction:** the first cx8 attempt exposed a reviewer error in the new frozen LIF record;
see section 5. That attempt is invalidated.

## 1. Coverage against the handoff and spec

| item | reproduction and disposition |
|---|---|
| 1. Raw bit identity | The original `tests/test_bit_identity.py` golden predates this branch and is unchanged. It passes for the default and explicit raw paths; an added case requires the same golden under `preset="instrumented", instruments=[]`. Instrument removal here means reconstruction on the original connectome and parameters, not undoing a separately supplied scratch graph or held-edge configuration. |
| 2. Defaults | CPU `cx_wedge.py --no-structure --sim 1:1 --device cpu --no-graphs` reproduces all 120 non-timing fields shared with `out/cx6/smoke/smoke_default_path.json`: 121 shared fields, only `wall_s` differs. Output: `out/instruments_review/smoke_raw.json`. Proprioception `all` excludes `turn_afferent`; default settle remains 1 s. BatchSim forwards the preset and preserves an explicitly installed afferent when it constructs its sense. |
| 3. Afferent law and anatomy | Tests check zero yaw, both signs on the same cells, all three k levels, rate ceiling, graph ordering, and actual Poisson writes with unrelated cells unchanged. Cache counts recomputed through raw counts, rows post / columns pre, `somaSide`; section 3. `law="unverified"`, gap, source pointers and removal condition survive provenance. Fractional sign and repeated options are now refused. |
| 4. Instrument inventory and holds | Synthetic graph tests exercise raw legacy diagnostic flags separately from the instrumented record and verify exact order of afferent / hold / relabel / edge gain descriptions. On the real cache, PEN-only selects precisely PEN_a(PEN1) and PEN_b(PEN2), 42 cells, 402 entries, 7,893 synapses; it selects no EPG. The round-7 checker independently pins both holds and the two compiled graph fingerprints. |
| 5. Turn and readout | A synthetic centre advancing four wedges/s across wraparound reproduces the expected signed slope. Side-group tests select by `somaSide`. The full protocol has 800 10-ms frames and 300 turn frames on [3.5, 6.5) s: 1 s settle, 2 s pulse, then turn at +0.5 to +3.5 s. A shortened CPU V command executes the actual CLI, observes left afferent firing for +90 deg/s, and records side metrics and descriptions. No body is present in this assay. |
| 6. Statistical contract | Six original contrasts, one Holm family of six, using `common.compare`; no missing contrast shrinks the family. The draft's confinement >=0.5 gate is now explicit before submission. No eligible follow measurements is `undetermined`; fewer than four per arm is `underpowered`. Test fixtures cover missing / duplicate runs and independently corrupted metadata. Per-seed CSV now includes filenames and run ids; comparisons expose eligible ids. The first-step Holm floor is disclosed without incorrectly treating it as an impossibility bound on later steps. |
| 7. Wrapper | Generator produces 48 jobs, eight arms by seeds 0-5, two sequential calls of 24 with `--target house --arm-block fam` and one fetch directory. Tests parse all generated arm flags against the manifest. Bash stubs execute the job's exit-status handling and verify that a failed first client stops before the second. They submit no jobs. |
| 8. Tests | Perturbation cases cover wrong sign, gain, nested preset, hold factor/count, turn window, resolved model override, duplicate and missing seeds. The end-to-end V smoke checks the actual transducer and JSON, not just a description. The unchanged golden and cache hashes provide independent raw-path controls. |
| 9. Repository hygiene | Changed public files scanned for infrastructure identifiers. Private cluster configuration and handoff replies remain ignored. No GPU work or cache regeneration occurred in review. |
| 10. Documentation | Inventory updated for PEN-only and the merged ExR6 evidence. The spec's module-only wording conflicted with its named edge/relabel candidates and the handoff's explicit transducer checklist; the narrow existing-boundary interpretation is documented in `PRESETS_SPEC.md` before submission. No new body-to-neural-module interface was introduced. |

## 2. Findings and fixes

**F1: token and replay bypass.** `BatchSim(proprioception="all+turn_afferent", preset="raw")` could attach an
unrecorded stand-in. Raw now refuses it before constructing the brain. Instrumented token use registers the
actual attached object; FlyBrain checks the same boundary when feeding or reporting a manually attached sense.
Checkpoint loading refuses a different preset or instrument description before altering state. The afferent
also refuses a different ordered bodyId array, preventing writes to stale indices after subsetting/reordering.

**F2: descriptions could claim an inactive instrument.** The protocol check accepted a describe-only object.
It now requires a callable installer and consistent name/kind plus nonempty law, gap, removal and audit fields.
`sign=1.5` no longer truncates to +1, and duplicate options no longer silently overwrite.

**F3: batch labels were insufficient proof.** The original reducer checked names and broad hold presence but
could accept a different k/sign, incomplete seeds, mismatched nested provenance or a changed protocol. It now
checks all 48 unique arm/seed identities, fixed timing/stimulus, resolved LIF, hold counts and factors, relabel,
compiled graph and source fingerprints, ledger availability and CUDA consoles. Any batch problem suppresses
scientific verdicts as `undetermined`. A frozen predeclaration supplies the complete per-arm resolved LIF and
LF-normalized source hashes; the generator refuses to replace a frozen declaration.

**F4: submission and inference gaps.** Both client calls now select house explicitly and abort on failure;
`pipefail` alone did not stop the second call. The eligibility gate, included/excluded run handling, and positive
follow plus opposite-sign control are now declared before submission. A sign-control failure does not prove
no afferent effect. A GLNO null in V does not mean PS196_b is unaffected or that every descriptive HGV k level
was null. The outcome wording now makes those limits explicit. Descriptive CSV includes SD and n, not means alone.

**F5: stale records.** Corrected five-arm / five-primary language to eight arms / six primaries, the erroneous
461-entry PEN-only test fixture to the independently measured 402, the withdrawn 6B inference, and the obsolete
ExR6 transmitter-unknown removal condition. No transmitter, receptor, gain or default is adopted.

## 3. Anatomy and MaleCNS gate

Reproduction file `out/instruments_review/anatomy.json`, from the shipped MaleCNS graph and raw counts:

| hop | L->L | L->R | R->L | R->R |
|---|---:|---:|---:|---:|
| AN07B037_a -> PS196_b | 0 | 202 | 214 | 3 |
| AN07B037_b -> PS196_b | 0 | 28 | 24 | 0 |
| PS196_b -> GLNO | 0 | 832 | 966 | 3 |
| GLNO -> PEN_a(PEN1) | 0 | 5,024 | 4,782 | 0 |
| GLNO -> PEN_b(PEN2) | 0 | 3,212 | 3,353 | 0 |
| CB0675 -> PS196_b | 51 | 1 | 4 | 51 |
| GNG580 -> PS196_b | 10 | 0 | 0 | 13 |
| PS047_b -> PS196_b | 185 | 16 | 27 | 217 |

The full hold selects 17 pre cells onto 88 PEN/EPG cells, 1,149 entries and 37,256 synapses. The PEN-only hold
selects the same 17 pre cells onto 42 PEN, 402 entries and 7,893 synapses. Sign-0 GLNO edges are included in the
raw-count reproduction; abs(W) alone would incorrectly omit them.

Pinned cache MD5s (unchanged in main and the review worktree):

| file | MD5 |
|---|---|
| neurons.parquet | c50c598a708b5b373cbaffca7d6a9d82 |
| W_post_pre.npz | ac131529cebf98decde58d0c227b7954 |
| sign0_counts.npz | bf01d724acf2a1fec8fdb60ef8a9e066 |

Compiled CSR MD5: `ef23cc27bea13be7f6a96f3c04fd3737`. The GLNO-glutamate scratch graph expected by the experiment
is `7a10d93ba2086f2c76bcdabdca79b4ec`; it does not replace the shipped cache.

## 4. Validation and remaining limits

Full CPU suite on the merged and fixed tree: **468 passed, 19 skipped, 220 subtests passed** (177.92 s),
including the unchanged bit-identity golden and the new empty-instrument case. Command:
`python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_cuda.py --ignore=tests/test_metal.py`,
with `CUDA_VISIBLE_DEVICES=-1`, `PYTHONIOENCODING=utf-8`, PATH Python. Log:
`out/instruments_review/full_cpu_final.log`. No local GPU was used. The generated wrapper is pinned to LF in
Git so a Windows checkout can execute the same reviewed script. Bash syntax and failure handling pass.

Operational corrections during validation: the first raw smoke omitted `--no-graphs` and was refused on CPU;
the corrected command above completed. The first full suite hit only the pre-existing cache-directory assertion
because this worktree's cache was a junction to main. Preserved the junction separately and copied the cache into
a real worktree-local directory; no connectome/test relaxation. A shell test initially selected Windows' WSL shim
and then exceeded reliable `bash -c` quoting with the complete wrapper. It now selects Git Bash on Windows and
executes a script file, as the real submission will. Final validation follows those fixes.

This review verifies the declared instrument paths, not arbitrary third-party Python installers. Held-edge and
relabel records verify explicit caller configuration; removing only their descriptions cannot restore a separately
modified configuration. Generic caller-supplied resolved-count fields are not a biological measurement; cx8 checks
its exact counts and model independently. Physiology and transfer-law claims remain unverified until sourced.
The preset remains raw. Round 7 is an experiment, and a functional result still requires the authorized suite and
room comparisons before any adoption. Independent scientific skeptic: pending, Fable when accounts reset.

## 5. Post-merge protocol correction before the valid experiment

At `a20d0ed`, the reviewed declaration and its synthetic fixtures incorrectly named `receptor_net_rule="class"`.
`cx_wedge.py --receptor-model shipped` actually resolves both receptor fields from `LIFParams()`: `sign` and
`abs`. Its separate CLI default of `class` is overridden by `shipped`. The simulation commands therefore ran
the intended shipped setting, but not the frozen model record. This is Astra's review error, not a GPU discrepancy.
The actual V CPU smoke passed, but the test had not compared its full resolved LIF to the new declaration.

The complete initial cx8 attempt is retained as invalid, with its declaration unchanged. No scientific decision
uses those measurements. Corrected the declaration builder to `abs` and added that comparison to the existing
actual-CLI CPU smoke. A fresh complete batch, cx8r, repeats the same arm commands, seeds and primary family;
its own declaration is frozen before submission. This is an operational replacement, not the one permitted
scientific follow-up. The raw defaults and physical experiment settings are unchanged by this correction.

Both initial client calls completed and fetched 24 jobs with zero scheduler failures. The original strict
reducer emits 144 issues, exactly the same mismatch in the top-level protocol, stimulus and full resolved LIF
for each of 48 runs. Separate `scripts/cx8_verify.py` reproduces 972 trace measurements within 5e-5 Hz absolute /
1e-6 relative tolerance and matches 52 source hashes (including `files_loaded` for the two probe scripts).
It retains the 48 model mismatches. These are validation diagnostics only; none of the invalid attempt's
scientific verdicts is used. The primary checker also now verifies the loaded probe hashes against the freeze.

Post-correction full CPU suite: **469 passed, 19 skipped, 220 subtests passed**, 176.98 s,
`out/compass7/cx8_protocol_cpu_final.log`, including the actual-CLI full-LIF comparison and unchanged golden.
All three main/worktree cache MD5s above remain unchanged. All 48 replacement commands compare equal to the
initial commands after replacing only the output directory (`out/compass7/cx8r_commands_check.json`).
