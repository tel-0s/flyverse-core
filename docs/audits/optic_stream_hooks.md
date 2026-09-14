# Per-stream hooks of the rate optic lobe (`OpticParams.stream_rectify` / `stream_adapt` / `spatial_suppress` / `fb_hold`)

**Status: opt-in mechanisms, all OFF by default; the shipped defaults are bit-identical ON CPU / a deterministic
backend (tests; on CUDA the claim is untestable -- section 3). Nothing is tuned; the one cluster batch here
(section 5: 15 jobs, 0 failed, 3 runs per arm, every verdict `underpowered` by construction) shows each hook is live
on the full lobe and what its magnitude is.**

**Skeptic-checked (2026-09-13): `verify:hooks`, verdict `mostly sound`, three refutations and nine corrections**
(verbatim in `receptor_verification.md`, section "Object round 2 (2026-09-13) -- build skeptic verdicts"). Applied in
place below. The three that change numbers: (i) this batch has **no hook-matched null** -- the rectify hook raises its
own none-vs-none null 2.6x, so T3's figure is **5.2x its own null, not 13x** (section 5); (ii) LC11's rectify figure
is **1.8x its own matched null with U 8 / p 0.2**, not 2.7x with U 9, i.e. the rank support disappears; (iii) the LC11
**rate** statistics are at the counting floor (5-6 spikes in a whole 143-cell population over 12 s) and are no longer
cited as liveness evidence.

Built for the round-2 model comparison that Neurome's intake asked for (`docs/NEUROME_INTERFACE.md` 3b, item 2;
`docs/audits/deficit_object.md` 6): on **fixed anatomy**, the existing linear sum vs (b) a **local presynaptic-stream
rectification** before the sum, synaptic signs preserved, vs (c) **adaptation + spatial suppression** (the Tanaka &
Clark 2020 'fast-adapting, size-tuned inputs' hypothesis, as our own arm, not copied parameters), with the
`gain_fb = 0` deterministic lobe and the feedback-hold as controls. Round 1 found the T3 figure to be the residual of
two opposing carrier sums (Mi1 / Tm3 / Tm2 raise, Tm1 / Tm4 lower; 96 % cancelled in the deterministic lobe) and
noted that the `rect` arm -- a ReLU on T3's *output* -- cannot recover terms that already cancelled inside the sum
(`deficit_object.md` 4.4 / 6). An `edges`-kind lesion cannot express a per-input nonlinearity; it needs a hook in
the substep. This file documents that hook: the fields, their defaults, the exact equations, the tests, the CLI, and
the liveness batch.

Files: `flyverse/optic.py` (the fields and the Torch-substep implementation; the CUDA / Metal kernels are untouched),
`flyverse/interp/common.py::parse_kv` (the CLI grammar for list-valued fields), `tests/test_optic_hooks.py`,
`scripts/probe_object_hooks.py` (the batch: `plan` / `record` / `analyse`), `out/hooks/` (15 run JSONs + consoles,
`batch.sh`, `manifest.json`, `analyse_console.txt`), `out/hooks_cluster.log` (the cancelled first submission in
`out/hooks_cluster.20260913T132956.log`), `out/interp/hooks/object_hooks.json` (the Result).

## 1. The fields (every one defaults to `None` = off)

| field | value | what it does | forces the Torch substep |
|---|---|---|---|
| `stream_rectify` | `[(pre_regex, post_regex, mode)]`, mode in `pos` / `neg` / `abs` | the matched block multiplies a rectified presynaptic deviation instead of the deviation | yes |
| `stream_adapt` | `[(pre_regex, post_regex, tau_ms, gain)]` | a separate fast-adaptation state per entry, subtracted from the block's presynaptic signal | yes |
| `spatial_suppress` | `[(pre_regex, k, radius_deg)]` | centre-surround on the presynaptic deviation of the matched types, for every target of the stream | yes |
| `fb_hold` | `[(spiking_pre_regex, rate_post_regex)]` | the spiking -> rate feedback entries of the matched blocks removed from `W_rs` at build time | no (a weight edit) |

Matching is `re.fullmatch` on the type string (pair_gain's rule is `re.match`; the hooks are stricter so that `Tm1`
cannot match `Tm12`). Where two entries of the same list match one (pre, post) entry, the **first wins**. A 'stream'
is the block of `W_rr` (rate <- rate) **and** `W_sr` (spiking <- rate) entries whose presynaptic type matches
`pre_regex` and whose postsynaptic type matches `post_regex`; `spatial_suppress` has no post regex (it transforms
the presynaptic cell's signal, so it applies to every target of that cell). The weights of a block are the shipped,
normalised, pair-gained `W_rr` / `W_sr` entries, **untouched** (the tests assert `abs(M_block - W[block]).max() == 0`).

## 2. The equations

The shipped substep (`optic.py` module docstring; `dr_j = r_j - b_j` the deviation from the operating point):

    inp_i = gain_rr * sum_j W_ij dr_j + gain_in * sum_p W_ip a_p + gain_fb * sum_s W_is s_s - adapt_gain * adapt_i
    v_i <- inp_i + (v_i - inp_i) exp(-dt / tau_i);   adapt_i <- dr_i + (adapt_i - dr_i) exp(-dt / adapt_tau)
    drive_s = clamp(gain_out * sum_i W_si dr_i, +-drive_clip)

With hooks, the recurrent sum and the output sum are split into the unmatched entries (the 'rest', on `dr`) and the
matched blocks, each on a transformed presynaptic signal `x` computed per entry (i <- j) in this order:

    u_j = dr_j - k * (1 / n_j) * sum_{j' in N(j)} dr_j'        spatial_suppress   (else u_j = dr_j)
    y_j = u_j - gain * A_j                                      stream_adapt       (else y_j = u_j)
    x_j = max(y_j, 0) | max(-y_j, 0) | |y_j|                    stream_rectify pos | neg | abs   (else x_j = y_j)

    inp_i = gain_rr * ( sum_{j in rest} W_ij dr_j + sum_{j in blocks} W_ij x_j ) + ...   (the other terms unchanged)
    drive_s = clamp(gain_out * ( sum_{i in rest} W_si dr_i + sum_{i in blocks} W_si x_i ), +-drive_clip)
    A_j <- u_j + (A_j - u_j) exp(-dt / tau_ms)                  after each substep, per stream_adapt entry

* `N(j)` is the set of cells **of j's own type** whose retinal column lies within `radius_deg` of j's column, **j
  itself included**; `n_j = |N(j)|`. Columns come from `flyverse.interp.trace.column_of_cells` (the hex annotation
  plus three propagation passes) and the angles from the retina's `col_dir`. A cell without a column is left as
  `u_j = dr_j`. In matrix form `u = dr - G dr` with `G[j, j'] = k / n_j` on `N(j)` (rows sum to `k`), `G` built once
  at construction (`OpticLobe.G_supp`; `hook_info['spatial']` records cells, cells with a column, and the min /
  mean / max of `n_j`). A spatially uniform stream therefore becomes `(1 - k) dr` (0 at `k = 1`); a single active
  cell keeps `(1 - k / n_j) dr_j` and each neighbour `j'` receives `-k dr_j / n_j'`.
* The adaptation state `A` (`OpticLobe.stream_adapt_state`, shape `(n_entries, B, n_rate)`, zeroed by `reset`) tracks
  the *suppressed* deviation `u` of the presynaptic cell with `tau_ms`; `gain = 1` subtracts it fully (a high-pass of
  the stream), `gain < 1` partially. It sits beside the lobe's own slow `adapt` (400 ms, on every unit's own output).
* Rectification acts on the presynaptic signal only; `x_j >= 0` under every mode, so every entry contributes
  `W_ij x_j` **with the sign of `W_ij`**: an inhibitory entry never excites (test 3c) -- structurally guaranteed by
  `x >= 0`, and therefore *vacuous on the configuration section 5 actually ran*: all **26,833** Mi1 / Tm3 / Tm2 /
  Tm1 / Tm4 -> T3 entries are EXCITATORY (`inhibitory_entries` 0 in the skeptic's full-lobe check, all three modes),
  so Neurome's constraint ("rectification must never turn inhibitory input excitatory") is tested on no inhibitory
  synapse in that batch -- only on the synthetic Pm1 (GABA) -> T3 subset of `tests/test_optic_hooks.py` (test 3c).
  The load-bearing half of the same test is the other one: **the LINEAR sum flips sign on 13,354 of those 26,833
  excitatory entries** (presynaptic cells below baseline disinhibiting), and the rectifier is exactly what removes
  those flips. Cite that number, not the inhibitory guarantee, when the mechanism is described. `pos` passes the positive
  half-wave, `neg` the negative half-wave *as a positive drive* (the OFF transition signalled as excitation through
  an excitatory synapse -- the reading under which Tm1 / Tm4's negative figure and Mi1 / Tm3 / Tm2's positive one
  ADD at T3 instead of cancelling), `abs` both. Note that the linear model, by contrast, lets an inhibitory entry
  contribute positively whenever its presynaptic cell sits below its operating point (disinhibition): the tests show
  that on the same entries.
* A block is one distinct (rectify entry, adapt entry) combination of matched entries, so a stream can carry any
  subset of the three transforms; the suppression is a property of the presynaptic cell and rides along.

`fb_hold` is not a substep hook: the matched entries of `W_rs` are removed before the matrix is uploaded (the other
entries keep their CSR order, so their products are unchanged bit for bit), and the native kernels stay in use.
`[('.*', '.*')]` removes every entry and is `gain_fb = 0` (bit-identical, test 3e). What the round-1 audits call the
'feedback-hold arms' are `interp_apply_object.py`'s `fb0` (`gain_fb = 0`, an OpticParams override) and its `edges`
holds (`t3_off_held` / `t3_on_held`: rate -> rate carrier blocks x 0 in `c.W`); there was no per-block hold of the
**spiking -> rate** feedback. `fb_hold` supplies that -- `[('^LoVC16$', '^T3$')]` holds the largest spiking entry
point of T3 (`deficit_object.md` 4.5, table `feedback_share`) and leaves the rest of the feedback stochastic.

## 3. Tests (`tests/test_optic_hooks.py`; CPU, `CUDA_VISIBLE_DEVICES=-1`; 14 tests, all pass)

```
CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m unittest discover -s tests -p test_optic_hooks.py
```

Two fixtures: a **synthetic hex patch** (7 columns = a centre and its six nearest neighbours; 7 photoreceptors, 7
Mi1, a GABA Pm1 at the centre, one T3 without an annotation, one spiking LC11) where every number is checkable by
hand, and the **cached subset** `Connectome.subset` of 18 types (photoreceptors, L1 / L2, Mi1 / Tm3 / Tm2 / Tm1 / Tm4,
T3 / T2 / Tm5Y, LC11 / LC10a, LoVC16; 22,483 cells, 1,208 columns) driven by random radiance and a 60 Hz LoVC16
feedback source over 12 frames.

| # | what | how it is proved |
|---|---|---|
| (a) | the shipped defaults are bit-identical | `git show HEAD:flyverse/optic.py` loaded as a sibling module, both lobes run on the subset: every frame's drive, `v`, `adapt`, `delta_rate` equal under `torch.equal`; a hook that matches no type (`stream_rectify [('^NoSuchType$', '^T3$', 'pos')]`, i.e. the split code path with everything in the rest block) is also bit-identical, as is `fb_hold` on no type; one frame of the shipped model recomputed from the docstring's equations equals the lobe's `v` / `adapt` bit for bit |
| (b) | each hook changes only its stream | rectify: the rest matrix equals the unmatched entries exactly and the blocks the matched ones (`abs(M - block).max() == 0`, 26,833 T3 entries in the subset); for a random `dr` the recurrent input of every non-T3 cell is `torch.equal` to the plain sum, the T3 rows equal `gain_rr (R dr + M_pos max(dr,0) + M_neg max(-dr,0))` to 1e-5, and the rest product is bit-identical to the plain product on every row without a matched entry. adapt: the first substep (state 0) equals the plain sum, the second changes T3 rows only, to the closed form. suppress: `G` lives within a type, rows sum to `k`, rows without a matched presynaptic entry are `torch.equal`, and a uniform Mi1 deviation is halved at `k = 0.5` |
| (c) | sign preservation | per entry of the (Mi1, Pm1) -> T3 block under `pos` / `neg` / `abs` with a two-signed random `dr`: `x >= 0`, `W_ij x_j <= 0` on every inhibitory entry, `>= 0` on every excitatory one; the block's weights equal the shipped `W_rr` entries; the linear sum on the same entries does flip |
| (d) | spatial suppression | hex patch, radius 5 deg (interommatidial 4.6): the centre's `n_j = 7` from the geometry alone; uniform Mi1 input -> `0.3 (1 - k)` (`< 1e-6` at `k = 1`); a single centre cell -> `0.5 (1 - k / 7)` at the centre and `-k 0.5 / n_j'` at each of the six neighbours, `n_j'` computed independently from `col_dir` |
| (e) | fb_hold | `[('.*', '.*')]` is `gain_fb = 0` bit for bit over 12 frames (and the feedback source does change the shipped lobe); `[('^LoVC16$', '^T3$')]` differs from `W_rs` exactly on LoVC16 -> T3 entries and the feedback product on every non-T3 row is `torch.equal` |
| (f) | the CLI grammar, the Torch path, errors | `parse_kv` round-trips the `;`/`,` grammar into the fields and leaves every old behaviour alone; `ol.cuda` / `ol.metal` are False with a hook on; a bad mode or `tau_ms <= 0` raise `ValueError`; `reset` zeroes the adaptation state; the state relaxes as `u (1 - a^n)` |

The regression suites `tests/test_receptor_model.py` and `tests/test_interp.py` (120 tests, the slow-term lobe and
the decompose tool's use of `W_rr` / `W_rs` / `W_sr` included) pass unchanged.

**Test added for the correction above (`test_c_batch_configuration_has_no_inhibitory_entry_and_the_linear_sum_flips`,
CPU, the cached 18-type subset).** On the configuration section 5 actually ran, it pins: the block is **26,833**
entries (`hook_info['entries_rr_matched']` equals the independently computed mask), **0** of them inhibitory -- so
the sign guarantee is exercised on no inhibitory synapse here -- while under a two-signed random `dr` the **linear**
sum makes **13,432 of the 26,833** excitatory entries contribute negatively (the skeptic's full-lobe count on its own
`dr` was 13,354; the number is `dr`-dependent, the fact is not) and the **rectified** stream makes **zero** entry of
either block contribute against its weight's sign (`x >= 0` in both streams). That is the mechanism claim, tested.

**Scope of "bit-identical", corrected (`verify:hooks`).** Row (a) is a CPU / deterministic-backend statement and
cannot be tested on CUDA at all: on a B200 the SAME default lobe object replayed twice differs by max **2.098e-05**
(`cuda_kernels` true / warp) and **2.480e-05** (`cuda_kernels` false / torch sparse); two separate builds of the same
code differ by 3.052e-05 / 5.341e-05, and worktree-vs-HEAD by 1.907e-05 / 1.144e-05 -- i.e. BELOW the same-object
non-determinism floor (`out/skeptic_hooks2/verify.json`). There is therefore no GPU evidence against the claim and
none obtainable; this is a pre-existing property of the shipped pipeline (the round-1 fact that the figure stage is
not fixed-seed reproducible), not something the hooks introduced. Read "bit-identical" as CPU / deterministic
backend, as this section's own tests do.

**The backend switch was an unmeasured confound; it is now measured and immaterial.** Every hook arm runs the Torch
substep while base and null run the native CUDA optic kernels, so base-vs-hook confounds the hook with a backend
switch. The skeptic quantified it: a no-match hook (`stream_rectify [('^NoSuchType$', '^T3$', 'pos')]` -- Torch
substep, split matrices, 0 matched entries) differs from the native CUDA default by max **1.335e-05**, below the
2.098e-05 same-object floor. At these magnitudes the backend switch is not a material confound.

## 4. CLI

Every probe that takes `--optic KEY=VALUE` through `common.params_from_args` / `common.parse_kv` selects a hook.
JSON or a Python literal works as before; list-valued fields (`TUPLE_LIST_KEYS` = the four hook fields and
`pair_gain`) also accept the shell-friendly grammar `entry;entry` with `,`-separated fields, numbers where a field
parses as one (a regex containing `,` or `;` needs the JSON form). **Precisely (`verify:hooks`): the grammar is
broader than "the five `TUPLE_LIST_KEYS`" -- a value containing `;` takes the tuple-list grammar under ANY key.**
That is in `parse_kv`'s docstring; it is stated here too because `flyverse/interp/common.py::parse_kv` is a file
shared with `probe_object_matched.py` and `probe_synthetic_stimuli.py` this round (the change is additive and
backward-compatible, covered by the 120-test regression):

```
--optic "stream_rectify=^(Mi1|Tm3|Tm2)$,^T3$,pos;^(Tm1|Tm4)$,^T3$,neg"
--optic "stream_adapt=^(Mi1|Tm3|Tm2|Tm1|Tm4)$,^T3$,100,1"
--optic "spatial_suppress=^(Mi1|Tm1|Tm3|Tm4)$,0.5,10"
--optic "fb_hold=^LoVC16$,^T3$"            --optic "fb_hold=.*,.*"   (= gain_fb=0)
```

`scripts/probe_object_hooks.py` (`plan` / `record` / `analyse`) names the arms of section 5 (`ARMS`), takes extra
`--optic` on top, and every run JSON carries the resolved `OpticParams` (the new fields included, through
`common.model_record`), the realised device, the cache fingerprint and the commit / source fingerprint
(`common.provenance`), plus `hook_info`: the entry counts of the rest and of every block (rate -> rate and
rate -> spiking), the spatial operator's neighbourhood statistics and whether the Torch substep ran.

Limits, stated: (1) any hook active runs the Torch substep (a `warp` `cuda_sparse` request is downgraded to `torch`
with a warning; the LIF's own CUDA kernels are unaffected); (2) `FlyBrain.state_dict` (`OPTIC_TENSORS`, `fly.py`,
not edited here) does not carry `stream_adapt_state`, so a checkpoint round-trip under `stream_adapt` restarts the
adaptation at 0 -- the same limitation as `g_slow_cls` before it was added there; (3) the column assignment is the
anatomical one of `trace.column_of_cells` (no stimulus-driven RF map exists yet; the `build:synthetic` task's RF map,
when it lands, is the better neighbourhood definition and the operator accepts any per-cell column vector).

## 5. The liveness batch (`hooks-0a5c6e`, 15 jobs = 5 arms x 3 runs, `out/hooks/`) -- run, verified, analysed

**How it ran.** `python scripts/probe_object_hooks.py plan --out out/hooks --runs 3 --name hooks --minutes 60` ->
`sh out/hooks/batch.sh` -> one `cluster_run.py` call, 15 jobs scheduled across the rented single-GPU boxes vast-a /
vast-b / vast-c / vast-d (run dir `/root/runs/hooks-0a5c6e` on each; B200 and H200), `--fetch out/hooks/`. The
console `out/hooks_cluster.log` ends `15 job(s), 0 failed (4.5 min)`; every job `completed exit 0`; every run console
(`out/hooks/<arm>_r<seed>.txt`) has `device cuda`, the GPU name and its `written` line; `analyse`'s `verify_batch`
(console vs JSON, realised device, arm / seed in the file name, hook liveness flags) reads `ok`
(`out/hooks/analyse_console.txt`, first line). A first submission of the same plan (`hooks-b56356`, 00:16, before the
job lines carried `source .venv/bin/activate`) went to the house cluster and was cancelled with that cluster's
reservation -- `out/hooks_cluster.20260913T132956.log`: 15 cancelled, no logs, `FETCH FAILED`; no number from it
exists. Wall time 75-106 s per hook-arm job and 94-181 s per plain job (each job = two 15 s conditions), so the
Torch substep costs nothing measurable at batch 1 on these GPUs.

Provenance (every run JSON, `provenance`): `flyverse_commit.commit` `unknown` (the run copy has no `.git`; the
identity is `source_fingerprint`, 43 files by content), compiled connectome md5 `ef23cc27bea13be7f6a96f3c04fd3737`,
sum|W| 121,460,584, 167,106 neurons, 25,578,600 entries; `execution.device` `cuda` (H200 / B200), the LIF backend's
own CUDA kernels and warp CSR in use; the resolved `OpticParams` carries the four hook fields (the arm's list, or
`None`). Stimulus: `probe_object_sweep`'s protocol verbatim -- ball radius 0.005 m at 0.05 m (11.4 deg), half sweep
0.06 m, 3 s settle, 12 s scored; the in-run `radiance_check` sees 2-9 columns changed > 5 % per sample.

**The protocol is the one Neurome asked be replaced, and these magnitudes inherit its confound** (`verify:hooks`):
the in-run `radiance_check` of these very runs shows the object's elevation varying from **-1.49 to +10.01 deg** and
its azimuth sweeping **51.4 -> 2.4 deg** within a run, i.e. the size / position / speed confound of Neurome's point 3
is fully present. Nothing below is a size-tuning result and the T3 = 0.0451 figure must never be quoted as one; the
matched geometry is `probe_object_matched.py`'s (`object_matched_assay.md`), and the model comparison runs there.

**Liveness (`hook_info`, full lobe; identical across the three seeds of an arm by construction).**

| arm | what the split found |
|---|---|
| rectify | `W_rr` (8,779,037 entries) split into 26,833 matched onto the 1,940 T3 cells -- 15,659 Mi1 / Tm3 / Tm2 -> T3 `pos` (5,474 presynaptic cells, syn-eq 2,847) and 11,174 Tm1 / Tm4 -> T3 `neg` (3,446 cells, syn-eq 1,981; = `deficit_object.md` 4.1's 11,174) -- plus 8,752,204 rest; no `W_sr` entry matched (T3 is a rate unit), 2,042,677 rest; 2 streams; `torch_substep` true, `optic_cuda_kernels` false (base: true) |
| adapt | one block of the same 26,833 entries (8,920 presynaptic cells), state `stream_adapt_state` (1, 1, 89,390), tau 100 ms, gain 1 |
| suppress | 7,274 Mi1 / Tm1 / Tm3 / Tm4 cells, all 7,274 with a column (19,817 of the lobe's rate cells are hex-annotated), 3-88 cells within 10 deg (mean 28.6), `G` 207,850 entries; applied to 967,564 `W_rr` entries onto 79,262 postsynaptic rate cells and 111,649 `W_sr` entries onto 5,779 spiking cells |

(Harness note, `verify:hooks`, not a discrepancy: a bare `OpticLobe` (`receptor=None`) gives `W_rr` nnz 8,780,774 =
26,833 matched + 8,753,941 rest against this batch's 8,779,037 = 26,833 + 8,752,204, because these runs build through
`Fly` with `receptor_model='sign'`, which rewrites the optic edge values. The split arithmetic is internally exact
and the skeptic's own cluster runs through this CLI reproduced 26,833 / 8,752,204 to the entry.)

**The numbers** (`out/interp/hooks/object_hooks.json`, table `comparisons`; mean +- SD over the 3 runs, the per-run
values in the JSON and in `analyse_console.txt`; rate types in rate units, LC types in mV / Hz; `null` = the shipped
model none-vs-none). Every `common.compare` verdict is **`underpowered`** (`p_floor(3, 3)` = 0.10): these are
magnitudes with their scatter, not results. **The `null` column is the SHIPPED model's none-vs-none null, which is
the wrong null for a hook arm** -- see the hook-matched null below, which is the reference every rectify number in
this table should be read against.

**The hook-matched null (`verify:hooks`, 3 runs, `out/skeptic_hooks/rectnull_r{0,1,2}.json`, B200, this file's own
CLI: `probe_object_hooks.py record --arm null --optic "stream_rectify=^(Mi1|Tm3|Tm2)$,^T3$,pos;^(Tm1|Tm4)$,^T3$,neg"`;
`hook_info` rr_matched 26,833 / rr_rest 8,752,204 and W md5 `ef23cc27bea13be7f6a96f3c04fd3737` identical to the runs
above).** The arm set of this batch scored every arm against the SHIPPED model's none-vs-none null, and the rectify
hook raises its own none-vs-none null by 2.6x, so two ratios in the reading below are corrected:

| statistic | rectify (3 runs) | shipped none-vs-none | **rectify-matched none-vs-none** | corrected reading |
|---|---|---|---|---|
| T3 `diff_signed_best_cell` | 0.0451 +- 0.0023 | 0.0034 +- 0.0013 | **0.0087 +- 0.0020** [0.0101, 0.0097, 0.0064] | **5.19x its own null** (U 9, p 0.10, z 17.9, `underpowered`), not 13.4x; the direction and the U 9/9 separation survive |
| LC11 `diff_max_over_cells_mean_mv` | 0.130 +- 0.035 | 0.048 +- 0.031 | **0.0726 +- 0.0434** [0.0716, 0.1165, 0.0298] | **1.80x, U 8.0, p 0.2, z 1.33** -- overlapping runs, NO rank separation (against 2.72x / U 9.0 / z 2.63 on the shipped null) |

Every hook arm of any future batch carries its own matched null (arm `null` + that arm's `--optic` override, 3 jobs
per hook); the difference here is the difference between "13x" and "5.2x", and between "U 9" and "no separation".

| type . statistic | base | rectify | adapt | suppress | null (none v none) |
|---|---|---|---|---|---|
| T3 diff_signed_best_cell | 0.0050 +- 0.0007 [0.0043, 0.0057, 0.0050] | **0.0451 +- 0.0023** [0.0435, 0.0477, 0.0442] | 0.0036 +- 0.0009 | 0.0043 +- 0.0002 | 0.0034 +- 0.0013 [0.0049, 0.0028, 0.0024] |
| T3 diff_abs_best_cell_mean | 0.0282 +- 0.0132 [0.043, 0.020, 0.021] | **0.0545 +- 0.0016** [0.056, 0.054, 0.053] | 0.0184 +- 0.0032 | 0.0304 +- 0.0091 | 0.0171 +- 0.0068 |
| T3 diff_signed_mean (population) | -9.8e-5 +- 1.1e-4 | +1.5e-4 +- 2.0e-4 | +1.6e-6 +- 2.0e-4 | +6.2e-5 +- 3.0e-5 | -1.9e-5 +- 7.0e-5 |
| T3 dev_mean, ball arm (level) | -5.5e-5 +- 5.7e-5 | **+0.0305 +- 0.0003** | -5.9e-4 +- 2.6e-5 | +1.1e-4 +- 1.2e-4 | -- |
| T3 dev_abs_mean, ball arm (level) | 0.0838 +- 0.0010 | 0.0731 +- 0.0005 | 0.0804 +- 0.0005 | 0.0788 +- 0.0013 | -- |
| T2 diff_signed_best_cell | 0.0144 +- 0.0006 | 0.0142 +- 0.0014 | 0.0144 +- 0.0008 | 0.0134 +- 0.0004 | 0.0080 +- 0.0017 |
| T2 diff_abs_best_cell_mean | 0.0427 +- 0.0162 | 0.0349 +- 0.0039 | 0.0349 +- 0.0020 | 0.0589 +- 0.0068 | 0.0256 +- 0.0078 |
| T2 dev_mean, ball arm (level) | +0.0043 +- 0.0005 | -0.0054 +- 0.0006 | +0.0053 +- 0.0007 | +0.0042 +- 0.0005 | -- |
| Tm5Y diff_signed_best_cell | 0.0102 +- 0.0004 | 0.0106 +- 0.0011 | 0.0094 +- 0.0017 | 0.0095 +- 0.0009 | 0.0068 +- 0.0012 |
| Tm5Y diff_abs_best_cell_mean | 0.0502 +- 0.0026 | 0.0423 +- 0.0017 | 0.0479 +- 0.0035 | 0.0519 +- 0.0056 | 0.0266 +- 0.0094 |
| Tm5Y dev_mean, ball arm (level) | +0.0018 +- 0.0001 | -0.0042 +- 0.0003 | +0.0022 +- 0.0001 | +0.0016 +- 0.0002 | -- |
| TmY21 diff_signed_best_cell | 0.0100 +- 0.0024 | 0.0116 +- 0.0015 | 0.0089 +- 0.0006 | 0.0085 +- 0.0012 | 0.0086 +- 0.0008 |
| TmY21 diff_abs_best_cell_mean | 0.0315 +- 0.0169 | 0.0300 +- 0.0017 | 0.0261 +- 0.0069 | 0.0399 +- 0.0106 | 0.0278 +- 0.0040 |
| LC11 diff_max_over_cells_mean_mv | 0.070 +- 0.038 [0.110, 0.035, 0.065] | 0.130 +- 0.035 [0.168, 0.124, 0.099] | 0.069 +- 0.033 | 0.061 +- 0.050 | 0.048 +- 0.031 [0.083, 0.024, 0.036] |
| LC11 diff_rate_hz_max_cell | 0.194 +- 0.048 | 0.083 +- 0.000 | 0.056 +- 0.048 | 0.056 +- 0.048 | 0.056 +- 0.048 |
| LC11 diff_rate_hz_mean | +0.0017 +- 0.0015 | +0.0004 +- 0.0003 | -0.0008 +- 0.0020 | -0.0006 +- 0.0010 | -0.0014 +- 0.0003 |
| LC11 drive_best_cell_mean_mv, ball arm (level) | 0.252 +- 0.019 | **0.703 +- 0.013** | 0.237 +- 0.005 | 0.275 +- 0.036 | -- |
| LC11 drive_mean_mv, ball arm (level) | 0.081 +- 0.032 | **0.479 +- 0.014** | 0.101 +- 0.009 | 0.133 +- 0.019 | -- |
| LC11 rate_hz_mean / rate_hz_max_cell, ball arm (level) | 0.0031 / 0.222 | 0.0035 / 0.139 | 0.0008 / 0.056 | 0.0006 / 0.056 | -- |
| LC10a diff_max_over_cells_mean_mv | 0.077 +- 0.008 | 0.077 +- 0.009 | 0.071 +- 0.021 | 0.050 +- 0.012 | 0.069 +- 0.019 |
| LC10a diff_rate_hz_max_cell | 0.25 +- 0.14 | 0.39 +- 0.19 | 0.33 +- 0.14 | 0.39 +- 0.10 | 0.33 +- 0.08 |
| LC10a drive_best_cell_mean_mv, ball arm (level) | 0.391 +- 0.016 | 0.309 +- 0.017 | 0.428 +- 0.020 | 0.375 +- 0.023 | -- |
| LC10a drive_mean_mv, ball arm (level) | +0.003 +- 0.005 | -0.035 +- 0.005 | +0.009 +- 0.005 | +0.004 +- 0.006 | -- |
| LC10a rate_hz_mean / rate_hz_max_cell, ball arm (level) | 0.018 / 0.83 | 0.028 / 1.14 | 0.016 / 0.61 | 0.019 / 0.89 | -- |

**Reading (magnitudes; nothing here is a result and nothing was tuned).**

* **Every hook is live on the full lobe** and changes the stage it addresses: the evidence is `hook_info` (the entry
  counts and `torch_substep` of the table above) and T3's `dev_abs_mean` (0.0838 base -> 0.0731 / 0.0804 / 0.0788),
  both real; the unmatched fields behave as the tests say (LC10a's diff statistics under rectify / adapt are the
  base's: 0.077 vs 0.077 mV). **LC11's firing is NOT liveness evidence** (`verify:hooks`): LC11 has 143 cells and a
  12-s window, so `rate_hz_mean` x 143 x 12 s is the whole population's spike count -- base **5.3 spikes** [5, 5, 6],
  rectify 6.0 [6, 7, 5], adapt 1.3 [2, 0, 2], suppress 1.0 [2, 0, 1], null 2.7 [3, 3, 2], and `cells_over_1hz` is 0
  in every arm. "LC11 firing unchanged" is 6 vs 5 spikes and the adapt / suppress "level move" is 5 spikes -> 1:
  those rows carry no information and are struck from the liveness argument (they remain in the table as what they
  are, counts at the floor).
* **rectify (the per-stream half-wave rectification of T3's carriers) is the one hook with a large effect on the
  quantity round 1 localised the loss to.** T3's (ball - none) best-cell signed figure is 0.0451 +- 0.0023 against
  the base's 0.0050 +- 0.0007 -- **9x the base and 5.2x its own hook-matched null (0.0087 +- 0.0020), not 13x the
  shipped null** (`verify:hooks`; the 0.0034 +- 0.0013 column is the shipped model's none-vs-none, which this arm
  should not be scored against) -- with every rectify run (0.0435 / 0.0477 / 0.0442) above every base and null run
  (U 9/9, p 0.10 = the 3 v 3 floor, so `underpowered`);
  the absolute-deviation figure is 0.0545 +- 0.0016 against 0.0282 +- 0.0132 / 0.0171 +- 0.0068,
  with a run-to-run SD that has fallen from the base's 0.013 to 0.002 (the same collapse of scatter the edge-hold
  arms showed, `deficit_object.md` 4.3: 0.062 / 0.072 +- 0.002 there, a different batch and 4 runs -- a scale
  reference, not a row-by-row comparison). This is what the mechanism was built to test: the two opposite-signed
  carrier sums no longer cancel when each is passed as a non-negative half-wave.
* **rectify also shifts T3's operating point, and that shift propagates.** With `x >= 0` on both half-waves through
  excitatory synapses, the carriers' zero-mean fluctuation becomes a positive-mean input: T3's mean deviation in
  the ball arm goes from -0.00005 to **+0.0305** (and the same in the none arm -- the (ball - none) subtraction
  removes it, the population `diff_signed_mean` stays at the null, +1.5e-4 vs -1.9e-5), T2's mean deviation flips
  from +0.0043 to -0.0054, Tm5Y's from +0.0018 to -0.0042, LC11's mean drive rises from 0.081 to **0.479 mV** and its
  best cell's from 0.252 to 0.703 mV (both arms), LC10a's mean drive from +0.003 to -0.035 mV and its mean firing
  from 0.018 to 0.028 Hz. So the rectify arm's T3 figure arrives together with a tonic change of the small-field
  stage's input in both conditions, and this batch cannot separate the two: **the model comparison needs a
  level-matched control** (the same tonic offset at T3 without the rectification -- an `OpticParams.baseline_by_type`
  arm, or a `stream_rectify` arm on one carrier class only -- and the stationary-flicker / uniform-field stimuli of
  the specificity battery, under which a pure level shift gives a figure and a rectified transient does not). This
  is the first open question the arm raises, and it is a question for the assay, not a parameter to adjust.
* **rectify at the LC output: a point estimate that the scatter covers -- and, against the right null, one with no
  rank support at all.** LC11's `diff_max_over_cells_mean_mv` is 0.130 +- 0.035 (0.168 / 0.124 / 0.099) against the
  base's 0.070 +- 0.038 (0.110 / 0.035 / 0.065) -- 1.9x the base with overlapping runs (z vs base +1.6, U 8/9 at
  3 v 3) -- and against its **hook-matched** null (0.0726 +- 0.0434) it is **1.80x, U 8.0, p 0.2, z 1.33**, not the
  2.72x / U 9.0 / z 2.63 the shipped-null column suggests (`verify:hooks`). LC11 firing does not follow either, but
  that row is spike counts at the floor (6 vs 5 spikes in the whole population; `diff_rate_hz_max_cell` 0.083 vs
  0.194 is one spike against three) and is not evidence in either direction.
  LC10a's diff statistics are untouched (0.077 vs 0.077 mV). This
  is the pooling fact of `deficit_object.md` 0.5 again: a 0.045 rate-unit figure in the few columns the ball crosses
  is a few hundredths of a mV at a cell pooling 94 columns under `out_norm l1`.
* **adapt (100 ms, gain 1 = a high-pass of the same carriers) is live but leaves T3's figure at the base / null**
  (0.0036 +- 0.0009 vs 0.0050 / 0.0034; abs 0.0184 vs 0.0282 / 0.0171), while it does change the level (T3
  `dev_mean` -0.0006, `dev_abs_mean` 0.0804 -- the LC11 firing row, 2 spikes of 143 cells in 12 s, says nothing).
  As expected of a *linear* operation
  applied to both carrier classes alike: it removes their common slow component and preserves their cancellation.
  A fast-adapting *input* in the Tanaka & Clark sense is not what un-cancels two opposite-signed sums; that
  hypothesis is about size tuning under matched geometry, which this sweep does not test.
* **suppress (k 0.5 within 10 deg on Mi1 / Tm1 / Tm3 / Tm4, every target) is live and broad** -- it touches 1.08 M
  entries onto 79,262 rate and 5,779 spiking cells and lowers the level of every small-field type (T3 0.0788, T2
  0.1364 vs 0.1438, Tm5Y 0.1585 vs 0.1616) and LC10a's diff statistic (0.050 vs 0.077 mV, the three runs 0.048 /
  0.064 / 0.039 against 0.085 / 0.069 / 0.076) -- but, again linear, it leaves T3's figure at the base (0.0043 vs
  0.0050) and moves T2's / TmY21's abs figures only within their scatter (0.0589 +- 0.0068 vs 0.0427 +- 0.0162;
  0.0399 +- 0.0106 vs 0.0315 +- 0.0169).
* T2 / Tm5Y under the base arm sit 1.4-1.8x above the none-vs-none null on `diff_signed_best_cell` (0.0144 vs
  0.0080; 0.0102 vs 0.0068; z +3.7 / +2.7) in every arm -- the batch-to-batch threshold call of `deficit_object.md`
  0, seen again and again `underpowered` at 3 v 3.
* `fb_hold` was not an arm of this batch (the task named defaults vs the three substep hooks); its full-lobe
  liveness rests on the subset test (e) and on its being a build-time edit of `W_rs` (the same code path on the
  full lobe).
* **The tests behind every verdict here are not tie-aware, and Neurome's request for a tie-aware U is still open**
  (`verify:hooks`): `common.compare` calls `scipy.stats.mannwhitneyu(..., method='exact')` for n <= 40, which is not
  tie-corrected, and **26 of the 234 comparisons** in `out/interp/hooks/object_hooks.json` contain ties (e.g. LC11
  `diff_rate_hz_max_cell` = 0.083 on all three rectify runs). Nothing rests on it in this batch -- every verdict is
  `underpowered` -- but the round's later tools (`object_round2_baseline.py`, `object_round2_compare.py`) implement
  the tie-aware exact U, and any re-scoring of these arms uses that, not `common.compare`'s p alone.

## 6. What this does and does not license

* The hooks are **swappable modules** in the project-rule sense: off by default, bit-identical off, selectable per
  run, documented here and in `OpticParams`. Nothing in the shipped defaults changed.
* The batch of section 5 is a **liveness and magnitude** measurement at 3 runs per arm -- every `common.compare`
  verdict is `underpowered` by construction (`p_floor(3, 3) = 0.10`) and is reported as such. It is not the model
  comparison: that runs after the matched visual assay (`scripts/probe_object_matched.py`, the `build:sphere` task)
  with >= 5 runs per arm in one submission, the specificity battery (`scripts/probe_synthetic_stimuli.py`, the
  `build:synthetic` task) and the predeclared LC11 / LC10a statistics, and its parameters (which streams, which
  modes, `tau`, `gain`, `k`, `radius`) are the hypotheses to declare, not knobs to fit -- the literature gives none
  of them (`deficit_object.md` 6; Keles et al. 2020 constrain outputs, Tanaka & Clark 2020 is a competing model).
* The rectification modes on the T3 carriers encode one specific reading of 'T2 / T3 respond to both ON and OFF
  transitions': that the OFF carriers' negative-going deviation reaches T3 as a positive drive. That is a hypothesis
  about the presynaptic transfer, not a fact of the connectome, and the arm exists so it can fail. Said plainly
  (`verify:hooks`): **the `neg` mode preserves the sign of `W` but INVERTS the sign of the signal**, so an excitatory
  synapse whose presynaptic cell drops below baseline now produces excitation -- not a per-synapse-plausible
  mechanism but a stand-in for an unmodelled OFF pathway. Any one-line summary that says only "synaptic signs
  preserved" understates it and must not go to Neurome without this sentence.

## 7. What `verify:hooks` says to do before this goes to Neurome

1. **Add a per-hook matched null to the arm set** (arm `null` + that arm's own `--optic` override): 3 jobs per hook,
   and it is the difference between "13x" and "5.2x" (section 5).
2. **Restate the LC11 rectify result as 1.8x its own null with U 8 / p 0.2**, not 2.7x with U 9.
3. **Drop LC11 `rate_hz_mean` / `diff_rate_hz_max_cell` from the liveness evidence** (5-6 spikes per population per
   12 s; `cells_over_1hz` 0 in every arm) and cite `hook_info` and T3 `dev_abs_mean` instead.
4. **The measured Torch-substep-vs-CUDA-kernel delta is recorded** (1.335e-05, below the 2.098e-05 same-object
   floor; section 3), so the backend switch is documented as checked rather than assumed.
5. **Qualify "bit-identical" as CPU / deterministic backend** wherever it is quoted (section 3), as the tests do.
6. Beyond the verdict, unchanged from section 6: the level-matched control for the rectify arm, >= 5 runs per arm in
   one submission, the matched geometry of `probe_object_matched.py`, and the specificity battery of
   `probe_synthetic_stimuli.py` (whose ON / OFF transition rows are the `transition = on | off` split of
   `object_synthetic_stimuli.md` 6.1, not the bright-vs-dark pooled row).
