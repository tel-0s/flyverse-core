# interp:decompose -- what drives a cell set, and the two validation targets

**Task** build:decompose (interpretability toolkit, docs/INTERP.md 4.1). **Files:** `flyverse/interp/decompose.py`,
`scripts/interp_decompose.py`, `scripts/interp_decompose_gf_batch.sh`, `tests/test_interp.py::DecomposeTests` (4 tests;
the file's 52 tests pass on the CPU, 9.6 s). The toolkit reads the model: nothing under `flyverse/` outside `interp/`
was edited. Every number below names its file; every generator is in `scripts/` or `flyverse/interp/`.

## 0. Verdict

1. **The tool.** `decompose(c, target, ...)` keeps every parameter of the contract and returns `common.Result`.
   Static: per (target type, presynaptic group) the effective input one volley of the group delivers (mV per post
   cell per volley = `EffectiveWeights.type_matrix`'s quantity), raw uncapped count, share, sign rule, silence flags.
   Dynamic: per run and arm the window-mean of `I_i(t) = sum_j A[i,j] r_j(t)` (mV/s per post cell) split by `by` ⊆
   {type, transmitter, tier, sign, module, side, cell}, plus the input **at the frame of the target's peak output**
   (the frame a `max over frames` check reads), `common.compare` of every arm against the null arm, E / I totals and
   the largest cancelling pair, `contributions` / `readout_per_body` in Neurome fields. Runs recorded under another
   receptor arm are weighted by **that arm's own effective weights** (`meta['arm']` -> `arm_params`), so a hold-table
   comparison sees the weight change and the rate change together. `contrast(c, target, params_a, params_b)` is the
   structural difference of two parameter sets over any target (the whole brain in 18 s), split into **sign / gain
   changes** of the shaped weights and **rescaled-only** entries (the fan-in route of receptor_integration.md E.3).
2. **Validation (a), walk.GF_max: reproduced to the digit at seed 0, NOT replicated over runs.** The four arms'
   seed-0 maxima are 4.964 / 4.629 / 12.517 / 13.311 Hz exactly (`out/dec_gf_cluster.log`, 12 jobs, 0 failed,
   device cuda). Seeds 1-2 scatter as widely as the effect: off 4.96 / 9.38 / 13.41, default 4.63 / 4.96 / 0.00,
   holdBrain 12.52 / 13.68 / 9.61, holdOptic 13.31 / 4.89 / 12.13; every arm vs off is verdict `null` (z -1.4 / +0.6
   / +0.2, exact U p 0.1 / 0.4 / 1.0). **DNp01's 1,455 input entries are identical in all four arms** (0 entries differ
   between the arms' effective weights), so whatever the arms do to DNp01 they do through presynaptic rates. The
   per-type input table shows **no group whose contribution is opposite-signed between the two halves**: the groups
   that move in one half only are LC4 (+34 mV/s under holdBrain, z 4.3, sd 38) and CB3513 / PVLP017 / LHAD1g1 /
   CL022_a / CB0115 / PVLP122 (holdBrain) vs LoVC5 / PVLP100 / LAL047 (holdOptic); 22 groups move the same way in
   every arm (IN00A062 inhibition halved, z +6.4 / +6.0 / +4.4). The "cancellation" the handover quoted is a
   seed-0 reading of a check whose run-to-run sd (2-4.6 Hz) equals its arm differences.
3. **Validation (b), the 282 histamine synapses: reproduced, and localized further.** The whole-brain contrast
   default -> holdBrainHis changes the sign of exactly **123 entries / 282 raw synapses, all histamine 0 -> -1**, onto
   OA-AL2i3 62 (131 syn), TmY14 25 (63), DNge138 5, s-LNv 4, VP5+Z_adPN 4, DNge150 4, DNge149 4, OA-VUMa2 3, ... from
   R8p / R8_unclear / R8y / R8d / R7y / R7p / R7_unclear / HBeyelet / T1 / AN27X004 / AN27X008 / GNG043 / IN27X004
   (E.4's list, completed), rescales 14,064 entries on exactly E.3's seven cells (DNge149 scale 0.847458 -> 0.844880
   ...), and touches **0 of MN9's 415 entries**. In `sec_taste` on the CPU the arms reproduce E.4 to the digit
   (default 5.0909 / 4.3158 / 2.3601; holdBrainHis = off 1.5548 / 4.3418 / 2.3098) and the dynamic decomposition
   says which of the 282 synapses carry anything: **the photoreceptor / eyelet entries carry 0 mV/s in every arm
   (R8 / HBeyelet are silent without an optic lobe) and the five named targets never fire (0.0 Hz, all arms)**; the
   current-carrying entries are GNG043's (a 56 Hz taste-driven GNG cell: -44 mV/s onto OA-VUMa2, -15 onto DNge150,
   -7.7 DNg34, -7.0 DNg104 / OA-VPM4 under holdBrainHis, 0 under the default) and AN27X004's (0.8 Hz, -0.4 to -0.8
   onto DNge138 / 149 / 150) -- about 25 of the 282 synapses. MN9's input differs between the arms through one group
   beyond scatter, DNge051 (-983 vs -1029 mV/s, z +5.8; DNge051 fires 51.4 vs 53.8 Hz).
4. **Two contract issues for the design task** (section 5): `common.raw_counts` reports 0 synapses for every signed
   entry on the real cache (it substitutes `sign0_counts`, which is non-zero only at explicit zeros) -- the tool uses
   its own `counts_matrix`; and `common.compare` cannot return `result` at exactly three runs per arm (the exact
   two-sided Mann-Whitney floor is p = 0.1 > 0.05), so every three-run comparison in this round is `null` by
   construction and the z column carries the information.

## 1. The tool

### 1.1 API (`flyverse/interp/decompose.py`)

```python
decompose(c, target, *, recording=None, params=None, optic_params=None, receptor=None, by=("type",), tiers=True,
          window=None, kind="current", null_recording=None, fb=None, top=40,
          counts=None, ew=None, frozen=None, arm_label=None, keep_links=True, arm_weights_override=None) -> Result
contrast(c, target, params_a, params_b, *, labels=("a", "b"), receptor_a=None, receptor_b=None, counts=None,
         by=("type",), optic_params=None, top=40) -> Result
arm_params(arm, base=None, out_dir=None) -> LIFParams      # off | default | hold<Group> | <table.csv> | as-given
arm_of(params) -> str;  counts_matrix(c) -> (csr, sign0_available);  edges(c, ew, target_idx, ...) -> (DataFrame, pre_idx)
static_table(df, n_post, by);  cancelling_pair(per_type);  print_per_type(res, top)
```

`recording` is a Recording, a list of runs, a `{arm: [runs]}` dict or a glob; `null_recording` the matched control
arm (one label). Batched recordings are refused (rows are not replicates). Units: static values mV per post cell per
presynaptic volley; dynamic values mV/s per post cell (A in mV per spike x Hz), `g_mv` = value x tau_syn / 1000 the
mean synaptic conductance in mV, directly comparable with the recorded optic drive (pseudo-group `optic_drive` when
the recording has `drive_mv`). Graded targets are decomposed through the OpticLobe's own `W_rr` / `W_rs` (kind
`optic_input`, an input decomposition, not an output attribution) and need `optic_params`.

Tables. Static: `per_type` (post_type, pre_group, value, value_E, value_I, n_entries, n_pre, raw_count, share, sign,
sign_rule, silent_entries, n_post, kind, unit), `per_tier`, `contributions`, `readout_per_body`, `links`. Dynamic:
`per_type` with `<arm>_mean / _sd / _n / _values / _rate_hz / _g_mv / _value_E / _value_I / _value_at_peak / _peak_t_ms
/ _output_peak_hz / _weight_mv_per_volley / _n_entries_missing` and, against the null arm, `<arm>_z / _welch / _U / _p
/ _verdict / _diff`, plus the static reference columns; `per_tier`, `timeseries` (first run, top 12 groups per
type), `contributions`, `readout_per_body` (input_current_mV_per_s, output_Hz, upstream_drive_mV per body and arm;
LC11 / LC10a never pooled), `links`. Summary: E / I totals and the largest cancelling pair per target type (static
and per arm), `arm_weights` (each arm's effective-weight md5 and the entries that differ from `params`'), coverage
(entries recorded, |weight| share, presynaptic cells missing), silent-entry counts, fan-in scale range.

`contrast` tables: `delta_links` (every sign / gain-changed entry: shaped a / b, effective a / b, tier a / b, change
`sign` | `gain`), `delta_per_type`, `delta_by_transmitter`, `delta_by_post_type`, `rescaled_cells` (bodyId, tot a /
b, scale a / b, sign-changed and rescaled entry counts), `rescaled_by_post_type`, `contributions` (sign-changed first,
then rescaled, capped at 5,000). Summary: `entries_total`, `entries_sign_changed`, `synapses_sign_changed`,
`entries_rescaled_only`, `target_cells_with_moved_fanin_scale`, `by_transmitter`, `by_post_type`, `pre_types_changed`.

Every JSON carries the provenance block of docs/INTERP.md 2.5 (a dynamic result inherits it from the first
recording's meta, written by `record` with the REALISED device, and adds every run's seed); `Result.check()` is empty
on all files of this round.

### 1.2 CLI (`scripts/interp_decompose.py`)

```
record   --target SPEC --protocol walk|taste|smell|rest --arm ARM --seed S --out PATH [--frames N] [--quantities rate_hz,drive_mv]
analyse  --target SPEC [--static --arm ARM] | --recordings LABEL=GLOB ... [--null-arm LABEL=GLOB] --by type,... [--window s0,s1] [--no-tiers] --json
contrast --target SPEC --arms A,B [--by ...] --json
validate --case gf --dir out/dec | --case taste --dir out/dec_taste --json
```

plus the common flags. `record` builds the Recorder over the target and every presynaptic cell with a stored entry
(`c.W[target]`'s columns), runs the named benchmark section verbatim (`sec_walk`'s walking phase with the ray tracer:
a GPU job; `sec_taste` / `sec_smell` / rest: CPU-able), captures every 10 ms frame (`t_ms` = the frame's start; the
state is the frame's end, so `--window 0.5,1.5` on `walk` is exactly the section's frames 50-149), and writes the
section's own scalars (`walk.GF_max_hz`, `taste.MN9_hz`, ...) into the recording's meta with the arm, seed,
`receptor_changed_entries`, the installed matrix's md5 and the provenance block. Hold tables are rebuilt in the run
directory by `scripts/build_hold_tables.py` when missing.

### 1.3 Tests (`tests/test_interp.py::DecomposeTests`, CPU, graph() + two edges)

`test_static_is_the_effective_weight_table` (values = `ew.A` entries; raw count 120 / share; the E / I cancelling
pair; GLNO sign-0 silent; tier grouping; Neurome columns; round trip), `test_dynamic_is_A_times_rate_per_frame`
(per-run values = A x rate window means; z against the null; `underpowered` at two runs; `null` at three with p 0.1;
the peak-frame value; E / I totals; timeseries; readout_per_body with control ids; a missing presynaptic cell is
counted; batched / graded / non-current kinds refused; an arm weighted by its own parameters), `test_contrast_
separates_sign_changes_from_rescaling` (a sign flip = 1 sign change, 0 rescaled; a cap change = 1 gain change + 1
rescaled entry with DNp01's tot 363 -> 723), `test_arms_and_cli_helpers` (arm round trips, a hold table built into a
temp dir by build_hold_tables and applied to the synthetic graph, the CLI parsers, `counts_matrix`).

## 2. Validation (a): the walk.GF_max cancellation

**Batch** `scripts/interp_decompose_gf_batch.sh` -> `cluster_run.py --name dec-gf`, run dir
`<cluster-fs>/neurome/runs/dec-gf-37cc6e`, console `out/dec_gf_cluster.log`: **12 job(s), 0 failed (1.5 min)**, every
job `device cuda` (NVIDIA B200, torch 2.11.0+cu128, cuda_sparse torch). The batch's `--fetch out/dec/` failed
(`FETCH FAILED out/dec/`); the files were copied by `scp` from the run directory (process rule) and are in `out/dec/`
(`<arm>_r<seed>.npz/.json/.txt`, 4 arms x seeds 0-2). Analysis: `scripts/interp_decompose.py validate --case gf --dir
out/dec --json out/interp/decompose/validate_gf.json` (CPU; log `validate_gf.log`).

**The check itself, side by side** (`walk.GF_max_hz`, max over frames 50-149 of the DNp01 mean rate; the recording's
own max reproduces the section's scalar in all 12 runs):

| arm | reference G.4 (seed 0) | seed 0 | seed 1 | seed 2 | mean +- sd | vs off: z / U / p / verdict |
|---|---|---|---|---|---|---|
| off | 4.964 | **4.964** | 9.381 | 13.414 | 9.25 +- 4.23 | -- |
| default | 4.629 | **4.629** | 4.963 | 0.000 | 3.20 +- 2.77 | -1.43 / 0 / 0.10 / null |
| holdBrain (optic side alone) | 12.517 | **12.517** | 13.685 | 9.615 | 11.94 +- 2.10 | +0.64 / 7 / 0.40 / null |
| holdOptic (Brain side alone) | 13.311 | **13.311** | 4.889 | 12.131 | 10.11 +- 4.56 | +0.20 / 4 / 1.00 / null |

`validation.status` = `reproduced` (the four seed-0 numbers to 3 decimals); `measured.replicated_over_runs` = false.
The installed matrices are the audited ones (md5 off `888fd350...`, default `ed1df661...`, holdBrain `6c5da91a...`,
holdOptic `874ea6eb...`; receptor-changed entries 0 / 48,295 / 44,463 / 3,832 = G.0's counts), and the arms'
effective weights onto DNp01 differ in **0 of 1,455 entries** (`summary.arm_weights`), i.e. G.0's "0 changed entries
on DNp01" seen from the tool.

**The decomposition** (`tables.cancellation` = per_type with the deltas; window means over the section's walking
window, 3 runs per arm, mV/s per DNp01 cell; 455 presynaptic types, 1,261 cells, 27 sign-0 entries, 979 of the
1,455 entries never firing in the window):

| pre group | weight mV/volley | raw syn | off | default | holdBrain | holdOptic | z vs off (default / holdBrain / holdOptic) |
|---|---|---|---|---|---|---|---|
| LPLC2 | +485.0 | 4,862 | +192.4 +- 22.1 | +171.0 +- 3.9 | +193.1 +- 27.8 | +192.0 +- 25.2 | -1.0 / 0.0 / 0.0 |
| SAD073 | -8.1 | 1,177 | -98.3 +- 19.2 | -75.1 +- 2.3 | -76.1 +- 14.5 | -69.7 +- 17.7 | +1.2 / +1.2 / +1.5 |
| IN12B015 | -1.8 | 124 | -49.9 +- 4.9 | -55.6 +- 3.5 | -52.9 +- 4.9 | -54.4 +- 1.8 | -1.2 / -0.6 / -0.9 |
| GNG300 | -2.1 | 474 | -41.7 +- 6.1 | -46.1 +- 2.9 | -43.4 +- 3.7 | -42.2 +- 0.7 | -0.7 / -0.3 / -0.1 |
| LC4 | +610.8 | 6,362 | +27.3 +- 8.0 | +27.0 +- 12.4 | +61.7 +- 38.5 | +41.9 +- 15.4 | 0.0 / **+4.3** / +1.8 |
| CB3513 | -3.2 | 190 | -27.1 +- 1.9 | -26.3 +- 4.0 | -19.8 +- 9.4 | -30.0 +- 2.7 | +0.4 / **+3.8** / -1.5 |
| IN00A062 | -4.3 | 257 | -26.0 +- 2.1 | -12.5 +- 2.1 | -13.4 +- 7.7 | -16.7 +- 4.9 | **+6.4 / +6.0 / +4.4** |
| PVLP026 | -2.3 | 149 | -14.8 +- 0.4 | -13.7 +- 2.7 | -10.5 +- 4.8 | -16.3 +- 0.5 | +3.1 / +12.1 / -4.4 |
| PVLP010 | -2.0 | 711 | -9.3 +- 0.7 | -7.1 +- 1.7 | -7.2 +- 2.8 | -6.5 +- 1.7 | +3.3 / +3.2 / +4.2 |

Totals (E / I, mV/s): off 304 / -502, default 247 / -414, holdBrain 315 / -406, holdOptic 295 / -424. At the frame of
DNp01's peak (`<arm>_value_at_peak`, the frame `walk.GF_max` reads): LPLC2 566 / 377 / 488 / 637, LC4 2.8 / 54 /
**288** / 5.3, SAD073 -24 / -32 / -59 / -36, CL367 -8 / -4 / -26 / -68 -- LPLC2 carries the peak in every arm and
LC4's peak-frame excitation is the one quantity that separates the optic-side arm from the others (holdBrain 288
mV/s, the rest <= 54), consistent with G.5's reading that the optic half is an upstream medulla state change that
reaches the giant fibre through the loom pathway, and with the same caveat: three runs, sd 38.

What the table does and does not show. **It does not show two opposite-signed halves that cancel.** Every group
that differs from off under one half alone differs in the same direction as the default or not at all; the
`nonadditivity` column (delta default - delta holdBrain - delta holdOptic) is large only for LC4 (-49) and SAD073
(-28), which is the arithmetic of the seed-0 GF maxima restated, not a mechanism. The one replicated reading (|z| >= 3
in all three arms, and the same sign) is that **every receptor arm halves the IN00A062 inhibition onto DNp01**
(-26 -> -12 / -13 / -17 mV/s) and trims PVLP010 / PVLP026 -- a common consequence of any receptor table, on neither
half. With the check's own run-to-run scatter (off 4.96-13.41 Hz at seeds 0-2), `walk.GF_max` is not a quantity a
cancellation can be read from at three runs; the tool's honest output is the seed-0 reproduction plus the scatter.

## 3. Validation (b): the 282 histamine synapses of taste

**Static, whole brain** (`validate --case taste`, CPU; `out/interp/decompose/validate_taste.json` tables
`whole_brain_delta_per_type` / `whole_brain_delta_by_post_type` / `whole_brain_rescaled_cells`; 18 s per contrast):

| | reference (E.4 / G.0 / E.3) | measured |
|---|---|---|
| entries whose shaped weight differs, default -> holdBrainHis | 123 silencings, 282 syn | **123 / 282.0**, all `histamine +0 -> -1` |
| onto | OA-AL2i3 62, TmY14 25, DNge138 5, s-LNv / DNge150 / VP5+Z_adPN / DNge149 4 each, OA-VUMa2 3, 12 singles | OA-AL2i3 62 (131 syn), TmY14 25 (63), DNge138 5 (14), s-LNv 4 (18), VP5+Z_adPN 4 (7), DNge150 4 (5), DNge149 4 (18), OA-VUMa2 3 (6), DNge152 2, DNg34 2, DNg104 / AVLP476 / OA-AL2i2 / OA-AL2i1 / OA-VUMa1 / OA-VPM4 / OA-AL2i4 / PPM1202 1 each |
| from | R8p / R8_unclear / R8y / HBeyelet | those plus R8d, R7y, R7p, R7_unclear, T1, AN27X004, AN27X008, GNG043, IN27X004 (tiers class / exact / fuzzy) |
| cells whose fan-in scale moves | 7 (E.3: DNge149 0.844880 -> 0.847458, ...) | **7**: DNge149 5,918 -> 5,900 / 0.844880 -> 0.847458 (1,598 entries rescaled), DNge138 5,819 -> 5,809 / 0.859254 -> 0.860733 (1,582), DNge138 5,017 -> 5,013 (1,371), AVLP476, OA-AL2i2, OA-VUMa1, OA-AL2i1 (14,064 rescaled entries, sum |delta| 8.95 mV) |
| MN9 | 0 changed input entries, scale 1.000000 | **0 sign-changed, 0 rescaled, 0 cells moved** (415 entries) |
| the five named targets alone | -- | 100 entries / 231 syn (OA-AL2i3 62, TmY14 25, DNge138 5, DNge149 4, DNge150 4); 4,551 rescaled on 3 cells |

`validation.status` = `reproduced`. (The design's `VALIDATION['decompose']` lists the 123 entries against the five
target types; the 123 are the whole-brain count and the five carry 100 of them -- both are in `measured.static`.)

**Dynamic, `sec_taste` on the CPU** (`record --protocol taste --device cpu`, 3 seeds x arms default / holdBrainHis /
off, target MN9 + the five types, `out/dec_taste/`; then the 20-type chain round below). The section's scalars
reproduce E.4 to every digit: `taste.MN9_hz` default 5.090923 / 4.315772 / 2.360074, holdBrainHis 1.554839 /
4.341760 / 2.309808 = off (MN9 and GNG175 bit-identical at all three seeds; `holdBrainHis_equals_off` true); default
vs holdBrainHis on the scalar itself: diff +1.19 Hz, z +0.8, U 7, p 0.4, `null` -- the CPU magnitudes are E.4's, and
what the tool adds is *where* the arms differ:

* **The five named targets never fire in the protocol** (`readout_per_body` output_Hz 0.0 for all 14 bodies of
  OA-AL2i3 / TmY14 / DNge138 / DNge149 / DNge150, every arm, every run), so they relay nothing to MN9 whatever their
  input does. The R8 / HBeyelet histamine entries onto OA-AL2i3 and TmY14 (100 of the 123, 194 of the 282 synapses)
  carry **0.0 mV/s in every arm**: without an optic lobe the photoreceptors and the eyelet are silent (rate 0).
* The current-carrying part of the 123 entries (`out/interp/decompose/taste_chain_by_transmitter.json`, histamine
  group per post type, holdBrainHis = the entries at -1, default = 0): GNG043 (56.4 Hz in the protocol) onto
  OA-VUMa2 **-44.3 mV/s** (6 syn, z 13.4 vs the default's 0), DNge150 -15.4 (1 syn, z 11.6), DNg34 -7.7 (3), DNg104
  -7.0 (1), OA-VPM4 -7.0 (1); AN27X004 (0.81 Hz) onto DNge138 -0.78, DNge149 -0.37, DNge150 -0.44. Everything else
  (AN27X008, IN27X004, s-LNv's HBeyelet, VP5+Z_adPN's GNG043 ...) is 0 because the pre cell is silent or the post cell
  is absent from the arm's differences. Of those post cells only OA-VUMa2 (1.16 vs 1.11 Hz) and DNg104 (0.25 vs
  0.00 Hz) fire at all in the protocol; DNge138 / 149 / 150, DNg34, OA-VPM4 are at 0 Hz in both arms.
* **MN9's input** (`validate_taste_MN9_dynamic.json`, 236 presynaptic types, all 415 entries recorded): the groups
  beyond scatter between default and holdBrainHis are DNge051 (-983 +- 21 vs -1029 +- 8 mV/s, z +5.8; DNge051 itself
  51.4 vs 53.8 Hz), MN7 (+3.0 vs +2.2, z +7.9), GNG473 (-3.8 vs -2.9, z -8.1), GNG568, GNG069, GNG199, DNg89 (all <
  1 mV/s). DNge051's own input differs through DNge059 (+75 +- 15 vs +68 +- 2, z +3.0) only. E / I totals of MN9:
  default 2,105 / -1,592, holdBrainHis 2,153 / -1,619 mV/s.

So the tool narrows E.4's "282 synapses" to the ~25 that carry current in `sec_taste` -- GNG043's and AN27X004's
histamine onto OA-VUMa2 / DNge150 / DNg34 / DNg104 / OA-VPM4 / DNge138 / DNge149 -- and shows that the dynamic route
from there to MN9 runs through DNge051 (the largest inhibitory input of MN9, -19 mV per volley over 231 synapses),
not through the five targets E.4 named (silent here) and not through the fan-in rescaling (the seven rescaled cells
are at 0 Hz except OA-VUMa1 / OA-AL2i1 / OA-AL2i2, also 0 Hz). The hop GNG043 -> OA-VUMa2 / DNg104 -> ... -> DNge051
is the next `trace` / `paths` question; every comparison above is three runs per arm (verdict `null` by the p floor;
the z values are 5-49 because the hold arm's scatter is 0.01-3 mV/s).

## 4. Files

| file | what |
|---|---|
| `flyverse/interp/decompose.py` | the tool (decompose, contrast, arm_params / arm_of, counts_matrix, edges, static_table, cancelling_pair, optic_matrices, print_per_type) |
| `scripts/interp_decompose.py` | CLI: record / analyse / contrast / validate |
| `scripts/interp_decompose_gf_batch.sh` | the 12-job cluster batch of validation (a) |
| `out/dec_gf_cluster.log` | its console (12 job(s), 0 failed; FETCH FAILED -> scp) |
| `out/dec/<arm>_r<seed>.{npz,json,txt}` | the walk recordings (cuda), 4 arms x 3 seeds |
| `out/dec_taste/<arm>_r<seed>.*`, `out/dec_taste_all/<arm>_r<seed>.*` | the taste recordings (cpu): MN9 + five targets, 3 arms x 3 seeds; the 20 chain types, 2 arms x 3 seeds |
| `out/interp/decompose/validate_gf.json` + `.log` | validation (a): cancellation table, peak view, arm comparison, provenance (device cuda) |
| `out/interp/decompose/validate_taste.json` + `.log`, `validate_taste_MN9_dynamic.json`, `validate_taste_targets_dynamic.json`, `validate_taste_targets_by_transmitter.json` | validation (b): whole-brain / five-target / MN9 contrasts, the CPU dynamics |
| `out/interp/decompose/taste_chain_by_type.json`, `taste_chain_by_transmitter.json` + `.log` | the 20-type chain round (`analyse --target ... --recordings default=out/dec_taste_all/default_r*.npz --null-arm holdBrainHis=...`) |
| `out/receptors_hold{Brain,Optic,BrainHis}.csv` | the hold tables (rebuilt by build_hold_tables.py through arm_params) |

## 5. Contract notes and open questions

1. **`common.raw_counts` is wrong on the real cache.** It replaces `|W.data|` by `connectome.sign0_counts(c)`, which
   is non-zero only at explicit-zero entries, so every signed entry reports 0 synapses (the first taste contrast of
   this task printed `synapses_changed 0.0`). `decompose.counts_matrix` merges the sign-0 counts into `|W|` instead;
   the fix for `common.py` is the same three lines. `links()` inherits the bug when called with `counts=None`.
2. **`common.compare` cannot say `result` at `MIN_REPLICATES = 3`.** With 3 vs 3 the exact two-sided U test's smallest
   p is 0.1, above `alpha 0.05`, so `verdict` is `null` for every three-run comparison however large z is (the test
   class pins this). Four runs per arm (p floor 0.029) or a verdict rule that does not require p at n = 3 is the
   design task's call; this round reports z and the values alongside and never calls a three-run difference a result.
3. **`VALIDATION['decompose'].taste_carrier`** lists 123 entries against five target types; the five carry 100
   (231 syn), the 123 / 282 are the whole brain. Both are reported.
4. Added keyword-only parameters (all defaulted, StubTests passes): `counts`, `ew`, `frozen`, `arm_label`,
   `keep_links`, `arm_weights_override`; `contrast` is a second public function of the module (the contract has none,
   and the taste target needs it). `kind` accepts only `current`; graded targets switch to `optic_input`.
5. The walk arms differ in DNp01's inputs by 0 entries, so a decomposition of DNp01 alone can only report rate
   changes; where the halves act (LC4's peak-frame excitation under holdBrain) is a `trace` question from the
   medulla, with `gain_fb = 0` as the deterministic null arm and >= 4 runs per arm.
6. The CPU taste protocol has no photoreceptor drive; the GPU suite's `taste.MN9` (5.85 -> 10.93, E.2) may use more
   of the 282 synapses than the ~25 found here. The same `record --protocol taste` line on the cluster (3 seeds x 2
   arms, one batch) would settle it; not run in this task (validation (b) is CPU-only by the contract).
