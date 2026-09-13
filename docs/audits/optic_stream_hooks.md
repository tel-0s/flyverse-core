# Per-stream hooks of the rate optic lobe (`OpticParams.stream_rectify` / `stream_adapt` / `spatial_suppress` / `fb_hold`)

**Status: opt-in mechanisms, all OFF by default; the shipped defaults are bit-identical (tests). Nothing is tuned;
the one cluster batch here shows each hook is live on the full lobe and what its magnitude is.**

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
`batch.sh`, `manifest.json`), `out/hooks_cluster.log`, `out/interp/hooks/object_hooks.json` (the Result).

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
  `W_ij x_j` **with the sign of `W_ij`**: an inhibitory entry never excites (test 3c). `pos` passes the positive
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

## 3. Tests (`tests/test_optic_hooks.py`; CPU, `CUDA_VISIBLE_DEVICES=-1`; 13 tests, all pass)

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

## 4. CLI

Every probe that takes `--optic KEY=VALUE` through `common.params_from_args` / `common.parse_kv` selects a hook.
JSON or a Python literal works as before; list-valued fields (`TUPLE_LIST_KEYS` = the four hook fields and
`pair_gain`) also accept the shell-friendly grammar `entry;entry` with `,`-separated fields, numbers where a field
parses as one (a regex containing `,` or `;` needs the JSON form):

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

## 5. The liveness batch (`hooks-<id>`, 15 jobs = 5 arms x 3 runs, `out/hooks/`)

**Status at the time of writing: SUBMITTED, NOT FINISHED.** `sh out/hooks/batch.sh` (generated by
`python scripts/probe_object_hooks.py plan --out out/hooks --runs 3 --name hooks --minutes 45`) submitted one
`cluster_run.py` call with 15 jobs (`hooks-b56356-0 ... -14`, job ids in `out/hooks/job_ids.json`; 5 arms x 3
seeds 0 / 1 / 2, `--fetch out/hooks/`, console `out/hooks_cluster.log`). At 00:20 PDT all 15 were `queued` behind
two running `kv` jobs and 9 queued `proprio` jobs (cluster API `/api/v1/jobs`). No number from this batch is
quoted here; when it finishes, the reading procedure is: (1) read `'15 job(s), 0 failed'` in `out/hooks_cluster.log`
and every console's `device cuda`; resubmit any `device cpu` job; (2) `PYTHONIOENCODING=utf-8 python
scripts/probe_object_hooks.py analyse --dir out/hooks --json out/interp/hooks/object_hooks.json` (its `verify_batch`
checks console vs JSON, the realised device, the arm / seed in the file name and that every hook arm's `hook_info`
says `active`, `torch_substep` and `n_streams > 0` before any statistic is computed); (3) report, per type (T3 /
T2 / Tm5Y / TmY21 / LC11 / LC10a) and statistic, `common.compare` of each arm vs the none-vs-none null and of each
hook arm vs base on `diff_signed_best_cell` / `diff_abs_best_cell_mean` / `diff_signed_mean` (rate types, kept
distinct) and `diff_max_over_cells_mean_mv` / `diff_rate_hz_max_cell` (LC types), plus the ball-arm absolute levels;
every verdict at 3 v 3 is `underpowered` (`p_floor` 0.10) and is quoted as magnitude +- run SD with the three
per-run values, never as a result. Expected liveness evidence, from the same code on the CPU subset (section 3,
`tests/test_optic_hooks.py`, and the construction smoke): the rectify arm splits 26,833 `W_rr` entries onto T3
(15,659 Mi1 / Tm3 / Tm2 -> T3 'pos', 11,174 Tm1 / Tm4 -> T3 'neg', matching `deficit_object.md` 4.1's 11,174 for the
Tm1|Tm4 -> T3 block), the adapt arm one block of the same 26,833 entries with a 100 ms state, the suppress arm a
`G` over 7,274 Mi1 / Tm1 / Tm3 / Tm4 cells (7,271 with a column, 7-88 cells within 10 deg, mean 29.1 in the
subset's 1,208 columns) applied to 106,828 rate -> rate and 3,868 rate -> spiking entries; the full-lobe counts are
in each run JSON's `hook_info`.

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
  about the presynaptic transfer, not a fact of the connectome, and the arm exists so it can fail.
