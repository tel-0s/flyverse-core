# The health tool (`flyverse/interp/health.py`): validation against the compass bump, the NT audit and the walk section

Generator of every number here: `scripts/interp_health.py` (`record` on the cluster, `analyse` / `structure` /
`validate` on the CPU) over `flyverse/interp/health.py`. Tests: **HealthTests 6/6** -- `tests/test_interp.py::HealthTests`
has SIX test methods on the 8-neuron `graph()` (`test_per_type_operating_points`, `test_full_counts_is_abs_w_plus_sign0`,
`test_structure_only_reproduces_group_shares`, `test_replicates_and_arms`,
`test_readout_source_through_the_observatory_hook`, `test_cli_module`); `pytest tests/test_interp.py -k HealthTests`
collects 6, and all 6 passed when this record was written (59 deselected) -- see section 5 for the one that is red
right now against in-flight shared code. The readout source is checked through `FlyBrain.neurotransmitters`, the
observatory's own getter, the same call `tests/test_nt_readout.py` uses. Contract: `docs/INTERP.md` 4.6; readout
interface: `docs/NT_READOUT.md`.

## 1. What the tool computes

`health(recording, *, c, params, window, by, ew, thresholds, fb, presyn, counts, ...) -> Result` (every parameter of
`stubs.health` kept; `presyn` / `counts` / `readout_rows` / `max_readout_cells` added as keyword-only defaults). Per group
(a type by default; `type_side`, `module`, `superclass`, `class`, `nt`, any `neurons` column, or `{label: spec}` in the
common selection grammar) over the window: `rate_mean` / `rate_max` / `spike_rate_hz` (from `spike_count`), `silent_frac`
(max rate < 0.5 Hz), `at_threshold_frac` (mean non-refractory `v` within 1 mV of `v_th`), `refractory_load` = rate x
`t_ref` (mean and max over cells; `refractory_limited_frac` above 0.4), `refrac_measured` (the recorded flag), `adapt_load`
= adapt / (`v_th` - `v_rest`), `e_in_mv_s` / `i_in_mv_s` / `ei_balance` / `input_coverage` (rate-weighted signed input
from the recorded presynaptic cells through `common.effective_weights`, decompose's `current`), `fanin_scale_min/med/max`
(`EffectiveWeights.scale`), `sign0_in_share` / `sign0_out_share` / `frozen_in_share` / `frozen_out_share` (raw synapses
onto / out of the group whose presynaptic cell is sign 0 or an optic rate unit), `state_flags`. With `recording=None` the
structural columns alone are computed for every cell (the NT audit's tables as a function). Tables `per_type` and the
Neurome `readout_per_body` (one row per body x quantity; `common.EXPORT_TABLES` columns); `replicate_table` /
`compare_arms` (mean / sd / n / values per group x statistic over independent runs, `common.compare` between arms).

`HealthReadout(fb, channels)` is the observatory readout source (`NTSource`): `rate_hz`, `v_margin_mv`, `adapt_mv`,
`refractory`, `fanin_scale`, `drive_mv`, `optic_rate`, sampled from `fb.brain` / `fb.optic` only when the map asks,
NaN on the optic rate units for the LIF channels, owned CPU copies; `fb.nt_source = HealthReadout(fb)`, no UI edit.

Raw counts. `health.full_counts(c)` = |W| on the signed entries (`c.W` holds the raw signed synapse counts before the
connection cap) plus `cache/sign0_counts.npz` on the explicit zeros, in float64. `common.raw_counts(c)` (the design's
helper) returns only the sign-0 counts when that cache is present (its `C.data = cnt` replaces every signed entry by the
sign-0 array, which is zero there) -- the first structural run through it reported every group's input as 100 % sign 0
(`out/health/structure_module.txt` before the fix); the tool now uses `full_counts` and the contract's helper should be
corrected by the common module's owner (reported, not edited: `common.py` is shared).

## 2. The sign-0 / silent populations of the NT audit (CPU, the cached connectome)

`PYTHONIOENCODING=utf-8 python scripts/interp_health.py structure --by module --json out/interp/health/structure_module.json`
(`out/health/structure_module.txt`), `--by superclass` (`structure_superclass.json` / `.txt`), and the same on the
pre-override cache `--cache-dir out/cache_pre_override` (`structure_module_pre_override.json`). The connectome loaded is
the adopted cache (`cache/`, 167,106 cells, 25,578,600 entries).

| quantity | `docs/audits/nt_audit.md` | tool (adopted cache) | tool (`out/cache_pre_override`) |
|---|---|---|---|
| neurons | 167,106 | 167,106 | 167,106 |
| synapses (raw, on the node set) | 124,161,873 | 124,161,873 (`structure_module.json`, `validation.json`) | 124,128,427 (fresh; the shipped file says 124,128,432) |
| sign-0 synapses | 2,701,289 (2.2 %) | 2,701,289 (2.176 %) | 2,701,289 (2.176 %) |
| sign-0 neurons | 3,312 | 3,312 | **3,407** |
| sign-0 presynaptic neurons (>= 1 output synapse) | 2,683 | 2,683 | 2,683 (2,778 with stored output entries) |
| mushroom_body: input silenced / output silenced | 9.8 % / 12.5 % | 0.0983 / 0.1251 | 0.0983 / 0.1251 |
| gustatory: in / out | 5.5 % / 8.4 % | 0.0548 / 0.0845 | 0.0548 / 0.0845 |
| central: in | 3.7 % | 0.0370 | 0.0370 |
| descending: out | 4.7 % | 0.0466 | 0.0466 |
| visual_centrifugal (superclass): out silenced, sign-0 presynaptic cells | 17.6 %, 73 | 0.1763, 73 | -- |
| cb_intrinsic: pre sign-0 / out / in | 1,083 / 4.3 % / 4.1 % | 1,083 / 0.0430 / 0.0405 | -- |
| vnc_motor: pre sign-0 / out | 363 / 34.8 % | 363 / 0.3481 | -- |
| ol_intrinsic: out / in | 0.1 % / 0.7 % | 0.0008 / 0.0070 | -- |

**Two of the shipped structure JSONs are stale float32 leftovers of the pre-fix pass; read the totals off the others.**
`out/interp/health/structure_superclass.json` carries `summary.sign0.synapses_total` 124,161,**872** and share
0.021756187081336975, and two of its rows are two synapses off -- `cb_intrinsic` `in_syn` 43,028,080 and `ol_intrinsic`
`out_syn` 43,984,240. The float64 values are 124,161,**873** / 0.02175618758586221 and 43,028,078 / 43,984,242, and
those are what `structure_module.json` and the `structure_module` / `structure_superclass` tables inside
`validation.json` carry (a re-run reproduces them at maxdiff 0.0; `out/skeptic_health/structure_superclass.json`).
Likewise `out/interp/health/structure_module_pre_override.json` carries the stale 124,128,432; a fresh
`structure --by module --cache-dir out/cache_pre_override` gives **124,128,427** (`out/skeptic_health/structure_pre.json`).
The pre-override headline figures are not affected and reproduce exactly: 3,407 sign-0 cells, 2,683 presynaptic, 2,778
with stored output entries, 2,701,289 sign-0 synapses.

Every group row of the audit's tables 1a / 1b that the tool reports agrees to the audit's printed precision (the per-module
`in_syn` / `out_syn` totals are the audit's to the synapse: mushroom_body 3,604,315 / 3,108,950; antennal_lobe 3,661,200 /
4,649,864; visual_centrifugal 1,333,472 / 2,139,737). The structural columns also give the frozen shares the audit does not
have: 91.3 % of the optic module's raw input and 98.6 % of its output are optic rate units (never spike in the LIF; their
effect arrives through the optic drive), 68.2 % of the visual-projection module's input and 27.5 % of the visual
centrifugal cells' input is frozen -- i.e. the LC / LPLC / centrifugal cells are driven almost entirely through the
`OpticLobe`, not `W`.

**The 3,407 of `common.VALIDATION['health']` is not this model's count.** On the adopted cache the tool gives 3,312 sign-0
cells (the NT audit's number); 3,407 is what the pre-override cache gives (`out/cache_pre_override`, the cache without
`TYPE_NT_OVERRIDE` = {TmY14, Mi19, aMe8}; `docs/audits/receptor_verification.md` line 165 quotes "sign-0 cells 3,407"
against sum|W| 121,427,136, the pre-override sum). The 95 cells are the override's rescued sign-0 cells -- checked on the
two `neurons.parquet` files: 91 TmY14 + 4 aMe8, `nt` unknown (94) / serotonin (1) before the override, glutamate / ACh
after; no cell is sign 0 in the adopted cache only. In the pre-override cache their output entries are explicit zeros
whose raw counts that cache's `sign0_counts.npz` does not carry (a fresh run gives total 124,128,427 = 124,161,873 -
33,446, `out/skeptic_health/structure_pre.json`; the shipped `structure_module_pre_override.json` carries the stale
float32 124,128,432 and the 33,441 that follows from it), so its presynaptic count 2,683 is a lower bound (2,778 cells
have stored zero entries).

**Scope of the verdict.** `validate` deliberately scores the sign-0 part against `docs/audits/nt_audit.md` (3,312 /
2,683 / 2,701,289 / 124,161,873 and the group shares), NOT against the literal `common.VALIDATION['health']` reference,
whose `sign0_presynaptic_bodies` is 3,407: the substituted `audit` dict is hard-coded at `scripts/interp_health.py`
line 434, and the contract's 3,407 is recorded alongside as `sign0_presynaptic_bodies_VALIDATION`. The substitution is
proven correct above (3,407 reproduces only on `out/cache_pre_override`; the 95 cells are identified), but `status:
reproduced` is therefore a verdict against a CORRECTED reference, not the shipped one. Owner items, uncorrected and
not this task's files: `common.VALIDATION['health'].sign0_presynaptic_bodies` and `docs/NEUROME_INTERFACE.md` line 73
both still carry 3,407 where the adopted release is 3,312.

## 3. The refractory-limited compass bump (cluster batch `health-val-5c6ad2`)

Protocol: `scripts/interp_health.py record --protocol compass --gE 2 --gD 15 --receptor-model off --seed {0,1,2}` =
`cx_wedge.simulate` as the round-4 grid ran it (`cx_glno.md` 5, config `base`): FlyBrain on the full connectome, compass
adaptation 0, 10 Hz Poisson background on all 46 EPG, 1 s settle, wedges 0-3 at +40 Hz for 2 s, 5 s free, Delta7 gain on
Delta7 -> EPG only, ER/ExR x1, CUDA graphs, torch sparse, receptor model off; the recording holds rate / v / adapt /
refrac / spike_count of the 464 EPG / EPGt / PEN / PEG / Delta7 / ER / ExR / GLNO cells every 10 ms plus rate_hz of their
4,390 non-frozen presynaptic cells (`compass_r{s}_presyn`). Scored window 5.5-8.0 s (the last 2.5 s free). All three jobs
`device cuda` (NVIDIA B200, torch 2.11.0+cu128; `out/health/compass_r{s}.txt`); cache md5 `ef23cc27bea13be7f6a96f3c04fd3737`,
sum|W| 121,460,584, `TYPE_NT_OVERRIDE` {TmY14, Mi19, aMe8}.

**Determinism check first.** The recorder's own `cx_wedge` sample at 5 s after release (`meta.cx_wedge_marks.t5.0`)
reproduces the round-4 seed-matched grid to the digit: in 201.0 / 201.4 / 204.3 Hz for seeds 0 / 1 / 2 (`cx_glno.md` 5:
"201.0/201.4/204.3"), out 10.2 / 9.5 / 8.9 with 1 / 3 / 2 outside cells above 22 Hz (the audit's per-seed constants
1 / 3 / 2), PEN 48.7 / 48.2 / 48.3 (audit 47.9 +- 0.8), Delta7 101.2 / 99.5 / 99.7 (100.1 +- 1.6), GLNO 132.8 / 137.2 /
132.0 (132.5 +- 3.7), vs 0.76 / 0.78 / 0.78 (0.763 +- 0.015). So the recording did not perturb the protocol and the
per-type statistics below are statistics of exactly the bump the round-4 audit reported.

`validate` (`out/interp/health/validation.json`, table `compass_validation`; `out/health/validate.txt`):

| quantity | reference (`cx_wedge.md` 7 / `cx_glno.md` 5 / VALIDATION) | tool, seeds 0 / 1 / 2 (window 5.5-8 s) | mean +- sd (n = 3) |
|---|---|---|---|
| EPG_in (11 driven-wedge cells) rate_mean | 180-260 Hz; 2/15 base 199.2 +- 4.5 | 201.1 / 198.0 / 200.2 | 199.8 +- 1.6 |
| EPG_in spike_rate_hz (from spike_count) | -- | 200.9 / 198.2 / 200.1 | 199.8 +- 1.4 |
| EPG_in refractory_load (rate x 2.2 ms) | 0.40-0.57 | 0.443 / 0.436 / 0.441 | 0.440 +- 0.004 |
| EPG_in refractory_load_max (the hottest cell) | -- | -- | 0.569 +- 0.003 (259 Hz) |
| EPG_in refractory_limited_frac (cells above 0.4) | -- | -- | 0.67 +- 0.05 |
| EPG_in refrac_measured (recorded flag) | -- | 0.509 / 0.489 / 0.499 | 0.499 +- 0.010 |
| EPG_in state_flags | "saturated" | refractory_limited, high_rate (3/3) | |
| EPG_out (35) rate_mean | out 10.2 | 9.8 / 10.1 / 9.5 | 9.8 +- 0.3 |
| PEN (42) rate_mean | 40-65; 47.9 +- 0.8 | 48.0 / 47.4 / 47.8 | 47.7 +- 0.3 |
| PEN refractory_load mean / max / frac > 0.4 | -- | 0.106 / 0.104 / 0.105 | 0.105 +- 0.001 / 0.568 +- 0.001 / 0.15 +- 0.01 |
| PEN silent_frac | -- | -- | 0.75 +- 0.01 |
| PEN state_flags | "saturated" | mostly_silent, refractory_limited, high_rate, sign0_input (3/3) | |
| Delta7 (42) rate_mean | 90-112; 100.1 +- 1.6 | 100.3 / 99.4 / 99.5 | 99.7 +- 0.5 |
| Delta7 refractory_load mean / max / frac > 0.4 | -- | 0.221 / 0.219 / 0.219 | 0.219 +- 0.001 / 0.451 +- 0.003 / 0.17 |
| Delta7 state_flags | "saturated" | refractory_limited, high_rate (3/3) | |
| GLNO (4) rate_mean | 132.5 +- 3.7 | 133.7 / 131.9 / 132.7 | 132.7 +- 0.9 |
| GLNO state_flags | sign 0 | high_rate, adapted (adapt_load 5.7), sign0_output (3/3) | |

`parts.bump.reproduced = true` (3/3 runs inside every band, `refractory_limited` on EPG_in in each); `status:
reproduced`; `Result.check()` empty.

What the tool adds to the audit's rates:

* **The refractory bound is 2.5 ms, not 2.2.** `refrac_measured` (the fraction of 10 ms frames whose `refrac` counter
  is > 0) is 0.4992 +- 0.0098 over the three runs at 199.8 Hz, i.e. 2.50 ms per spike, while rate x `t_ref` gives 0.44.
  It is a FRAME-SAMPLED estimate of the duty cycle, not the duty cycle itself: the quantity it estimates is 5 steps x
  0.5 ms x rate, and a direct recomputation on the same recorded frames gives 0.50924 for seed 0 against the tool's
  0.50909, the 1.5e-4 gap being the scoring window's exclusive upper edge (250 frames, not 251). `Brain.step` sets the counter
  to `t_ref` = 2.2 and tests `refrac > 0` before subtracting `dt` = 0.5 each step (`flyverse/brain.py` 701-712, and the
  CUDA path 791-801), so a cell is held for 5 steps = 2.5 ms and can fire again on the 6th: the minimum inter-spike
  interval is 3.0 ms (333 Hz) and the hottest EPG at 259 Hz spends 0.65 of its time refractory. `refractory_load`
  keeps the contract's definition (rate x `t_ref`) so its 0.4 bound stays comparable with `cx_wedge.md` 7; the measured
  column is the one to read for the actual duty cycle.
* **The mean rates hide which cells are saturated.** PEN at 47.7 Hz and Delta7 at 99.7 Hz carry the
  `refractory_limited` flag through `refractory_load_max` (their hottest cells run at 258 and 205 Hz), not through the
  mean; 75 % of the 42 PEN cells and 11.9 % of the 42 Delta7 cells never fire in the window at seed 0 (`silent_frac`;
  the Delta7 3-run mean is 0.103 +- 0.027, seeds 0.119 / 0.071 / 0.119), so the bump is carried by ~10 PEN and ~37
  Delta7 cells. `at_threshold_frac` is 0.000 for every compass group: no cell sits within 1
  mV of threshold -- the bump cells are either refractory or being reset, and the others are far below.
* **How far below: the undriven ring is hyperpolarised by hundreds of mV.** `v_margin_mv` (v_th - mean non-refractory
  v) is 252 +- 2 mV for EPG_out, 56 for PEN, 66 for PEG, 7.3 for the ring neurons (rest), 5.7 for EPG_in. Under the
  Delta7 -> EPG x15 gain the outside EPG sit ~250 mV below threshold (there is no lower clamp on `v` in the LIF), which is
  the mechanism of the "pinned" bump of `compass_room.md`: a cell 250 mV down cannot be recruited by a few Hz of
  neighbouring drive. `ei_balance` reads it directly: EPG_out -0.84 (inhibition-dominated), EPG_in +0.26, Delta7 +0.49,
  PEN -0.34, PEG -0.45, GLNO +0.85 (its 19 % sign-0 input excluded: `sign0_in_share` PEN_a 0.26 / PEN_b 0.17, GLNO
  0.08, all flagged `sign0_input` above the audit's 15 % bound).
* **The ring cells that fire are the ExR4 / ExR6 / ER6 loop of `cx_wedge.md` 3, and they are adaptation-saturated.**
  ExR6 233 Hz / ExR4 181 / ER6 183 / ER4m 66 Hz (seed 0, `compass_per_type`), `refractory_limited` + `high_rate` +
  `adapted` with `adapt_load` 7.8-10.0 (adapt / (v_th - v_rest) -- they are outside `adapt_by_type`'s zero set, so the
  default adaptation runs and is eight to ten times the threshold gap); **23 of the 34** ER / ExR types carry the
  `silent` flag at seed 0 (`validation.json` table `compass_per_type`: 21 at exactly 0.0 Hz, ER3w_b and ExR8 at
  `rate_max` ~1e-14 / `rate_mean` 2e-17 and 1e-15), ER3d_b / ER3p_a / ER3p_b silent with 21 / 26 / 30 % sign-0 input.
  EPGt (4) silent.

## 4. The walk-section operating points, receptor default vs off (cluster batch `health-val-5c6ad2`)

Protocol: `record --protocol walk --receptor-model {default, off} --seed {0,1,2}` = `scripts/benchmark.py sec_walk`'s
walking + loom phases step for step (Brain + OpticLobe eager, smell on, 1.5 s walking at 4 mm/s, 0.8 s loom from the
front), recording all LIF quantities of the 1,719 cells = every descending neuron (1,314) + DNp01 + the 24 wing-power MNs
+ the 381 leg MNs, plus rate_hz of their 47,854 non-frozen presynaptic cells; window 0.5-1.5 s (the section's scored
walking window). All six jobs `device cuda` (`out/health/walk_{default,off}_r{s}.txt`).

**The protocol is the benchmark's to the digit.** The recorder's own `walk.GF_max_hz` (`meta.walk`) is 4.629 (default,
seed 0) and 4.964 (off, seed 0) -- `receptor_integration.md` G.4's single runs are 4.629 default / 4.964 off. Seeds 1
and 2 give 4.963 / 4.604 (default) and 9.381 / 13.414 (off): the off arm's GF_max scatter (4.96-13.41 over three seeds)
is ten times the G.4 default-vs-off difference (0.34 Hz), so that difference was a seed, as round 1 already said of
`walk.power_max`.

**But the walk recorder is NOT reproducible at fixed seed over its full length.** A re-record of `walk`, receptor
default, seed 0 (`out/health_skep/walk_default_r0.npz`) agrees with `out/health/walk_default_r0.npz` bit-for-bit on
`rate_hz`, `spike_count`, `adapt_mv` and `refrac` for every frame up to t = 2.11 s (`v_mv` differs there only at
float32 ULP scale, max 1.5e-5 mV inside the scored window), then diverges over the last 19 frames, t >= 2.12 s -- the
loom tail: max |delta rate| 124.6 Hz, 487 of the 1,719 cells differ at the final frame, cumulative `spike_count`
9,247 vs 8,212, max |delta v| 81.4 mV. The scored 0.5-1.5 s walking window is identical and the two consoles agree to
the digit (GF max 4.629, wing power 48.5 / sustained 20.1, loom GF peak 47.2, escape 3.50 s), so **no walk number
reported here is affected** -- but any future round that scores the loom tail must not assume determinism at fixed seed.

Operating points (`validation.json` tables `walk_per_group`, `walk_per_group_null`, `walk_compare_group`; the
`compare` columns are `common.compare` of the default arm against the off arm as null, n = 3 each):

| group (n cells) | stat | default, seeds 0 / 1 / 2 | off, seeds 0 / 1 / 2 | z | U, p | `verdict` (tool) |
|---|---|---|---|---|---|---|
| DN_all (1,314) | rate_mean Hz | 1.21 / 1.42 / 1.35 (mean 1.33) | 1.96 / 1.51 / 1.41 (mean 1.63, +23 %) | -1.03 | 1, 0.20 | null |
| DN_all | silent_frac | 0.653 / 0.619 / 0.637 | 0.627 / 0.639 / 0.635 | +0.41 | 5, 1.0 | null |
| DN_all | at_threshold_frac | 0 / 0 / 0 | 0 / 0 / 0 | -- | 4.5, 1.0 | null |
| DN_all | v_margin_mv | 7.88 / 8.09 / 7.96 | 8.71 / 8.19 / 8.10 | | | |
| DN_all | ei_balance | -0.082 / -0.068 / -0.066 | -0.102 / -0.074 / -0.072 | | | |
| DN_all | refractory_load | 0.0027 / 0.0031 / 0.0030 | 0.0043 / 0.0033 / 0.0031 | -1.03 | 1, 0.20 | null |
| leg_MN (381) | rate_mean Hz | 2.05 / 2.71 / 2.07 (mean 2.28) | 3.20 / 2.31 / 2.18 (mean 2.56, +12 %) | -0.51 | 2, 0.40 | null |
| leg_MN | silent_frac | 0.622 / 0.591 / 0.656 | 0.635 / 0.635 / 0.622 | -1.04 | 3.5, 1.0 | null |
| wing_power (24) | rate_mean Hz | 11.8 / 17.1 / 16.2 (mean 15.1, sd 2.8) | 31.1 / 18.3 / 18.2 (mean 22.5, sd 7.4, +50 %) | -1.01 | 0, 0.10 (the 3-vs-3 floor) | null |
| wing_power | silent_frac | 0 / 0 / 0 | 0 / 0 / 0 | | | |

`null` is the verdict string `common.compare` wrote into the shipped `validation.json`, and it means "no effect
resolvable at this n", not "no difference": the wing_power arms are perfectly separated (every off seed above every
default seed) and still score `null`, because U = 0, p = 0.10 is the floor of a two-sided 3-vs-3 rank test. (The
common module's owner is renaming this string to `underpowered`, which is the better name for exactly this reason;
the quoted values above are the shipped file's.)

Per type (`out/interp/health/walk.json`, 524 DN / MN types, 2,096 group x statistic comparisons, all `null`): 327 of the
524 types are `silent` in the default arm's seed 0 (max rate < 0.5 Hz over the walking second), 0 are `at_threshold`,
0 `refractory_limited`, 0 `high_rate`; the top types by rate over three runs are DNb05 30.0 +- 0.8 Hz, DVMn 1a-c 20.4
+- 4.0, DVMn 3a,b 18.6 +- 4.1, DVMn 2a,b 17.7 +- 3.8, MNhl62 16.1 +- 2.2, DNp32 15.2 +- 1.0, DNpe040 15.1 +- 1.6, all
flagged `adapted`. The walk-section operating point of the descending population is therefore: 62-65 % of DNs silent,
none within 1 mV of threshold, mean `v` about 1 mV below rest (margin 7.9-8.7 mV against the 7 mV rest gap), weakly
inhibition-dominated input (ei_balance -0.07 to -0.10), 99.75 % of their |A| input recorded (`input_coverage`; the rest
is frozen optic units, which reach the DNs through the optic drive) -- **and NOT DISTINGUISHABLE between the two
receptor arms at n = 3, given the off arm's scatter**, which is a weaker statement than "identical". The arm means
differ substantially (`walk_compare_group`): DN_all `rate_mean` 1.33 vs 1.63 Hz (+23 %), leg_MN 2.28 vs 2.56,
wing_power 15.1 vs 22.5 (+50 %) -- and wing_power's U = 0, p = 0.10 is the SMALLEST p a two-sided 3-vs-3 rank test can
return, i.e. a perfect separation that n = 3 cannot call. What licenses "null" is the off arm's own spread
(GF_max 4.96-13.41, sd 7.4 Hz on wing_power against a 7.5 Hz difference), not an absence of movement. `parts.walk.reproduced`
is `null` because no prior per-type reference exists: these are the tool's first numbers, and the reference for the next
round. The G.4 "attribution" of the walk section's take-off cost to the optic half is not contradicted -- but neither is
it confirmed at this n; what the data say is that a 23-50 % arm difference in the DN operating point is inside the off
arm's noise, so "0 changed entries land on DNp01 / LC4 / LPLC2 / motor types" is consistent with, not demonstrated by, these runs.

## 5. Readout source, batch record and files

* `HealthReadout` is loaded by the interface's own hook: `tests/test_interp.py::HealthTests::test_readout_source_through_the_observatory_hook`
  attaches it with `fb.nt_source = HealthReadout(fb)` on `test_control.graph()` (`FlyBrain(batch=2)`) and reads it back
  through `fb.neurotransmitters(batch_index=1)` -- the getter `tests/test_nt_readout.py` exercises (batch bounds,
  snapshot type, immutability, alignment by bodyId, NaN on graded units, +-inf never published). `tests/test_nt_readout.py`
  passes unchanged (5 tests). Not run: the room UI (the contract says verify with the interface, not the UI).
* Batch: `python scripts/cluster_run.py --name health-val --minutes 25 "<9 record commands>" --fetch out/health/`;
  console `out/health_val_cluster.log`, terminal line `9 job(s), 0 failed  (3.8 min)  run dir
  /mnt/beegfs/neurome/runs/health-val-5c6ad2`; every job asserted CUDA and printed `device cuda`. Recording jsons carry
  the provenance block (resolved LIFParams / OpticParams, fingerprint, realised device, seeds, stimulus); the cluster's
  run dir is a file copy, not a checkout, so `flyverse_commit.commit` reads `unknown` there -- `analyse` adds
  `provenance.analysis.flyverse_commit` from the checkout that analysed. `Result.check()` is empty on every JSON.
* Files: recordings `out/health/{compass_r0..2, walk_default_r0..2, walk_off_r0..2}{.npz,.json,_presyn.npz,_presyn.json,.txt}`;
  Results `out/interp/health/{validation, compass, walk, structure_module, structure_superclass,
  structure_module_pre_override}.json` with their consoles `out/health/{validate, compass_analyse, walk_analyse,
  structure_*}.txt`; CPU smoke tests `out/health_smoke/` (0.09 s compass, 5-frame walk, `--allow-cpu`).
* Tests: **HealthTests 6/6** at the time of writing -- `PYTHONIOENCODING=utf-8 python -m pytest tests/test_interp.py
  -k HealthTests` collects 6 and passed 6 (59 deselected). `tests/test_interp.py` is shared across the seven
  implementers and its total drifts run to run (40 at the first writing, 62 at the skeptic's, 65 as of this edit), so
  the class count is the one to quote, never the file's. `tests/test_nt_readout.py` 5 passed.
* **Red right now, and not this tool's doing:** `HealthTests::test_replicates_and_arms` fails
  (`AssertionError: 'underpowered' != 'null'`) as of this edit, because the common module's owner is renaming
  `common.compare`'s verdict string from `null` to `underpowered` in the working tree while the shared test still
  asserts `null`. Neither file is this tool's to edit; the assertion belongs with the rename. `health.py` is unaffected
  -- it passes the string through -- and every verdict quoted in section 4 is read off the shipped `validation.json`,
  which carries `null`. The other 5 HealthTests and all 5 PathsTests pass.
* **Unification note (maintenance risk, not a numerical objection).** `record --protocol compass`
  (`scripts/interp_health.py::record_compass`) REIMPLEMENTS `cx_wedge.simulate`'s stimulation loop rather than calling
  it: it imports only `cx_wedge.compass_cells` for the wedge identities and re-derives the type-path gain through its
  own `compass_type_path_gain`. Equivalence is demonstrated numerically to the digit on all three seeds (section 3's
  determinism check, including the per-seed above-22-Hz counts 11 / 10 / 11 inside and 1 / 3 / 2 outside), so the
  numbers are not in question -- but the duplicate loop is the thing to unify under the contract's "unify, not duplicate".

## 6. What to carry forward

1. `common.raw_counts` returns the sign-0 counts alone when `cache/sign0_counts.npz` exists (section 1); every tool that
   reads raw counts through it (links' `synaptic_pair_count`, paths, decompose shares) is affected -- fix in `common.py`.
2. `common.VALIDATION['health'].sign0_presynaptic_bodies` = 3,407 is the pre-override count; the adopted model's is 3,312
   (2,683 presynaptic). `docs/NEUROME_INTERFACE.md` line 73 says 3,407 too.
3. The LIF's effective refractory period is 2.5 ms (5 x 0.5 ms steps) for `t_ref` 2.2: the bump's measured duty cycle
   is 0.4992 +- 0.0098, not 0.44; a 300+ Hz cell is at the bound. Whether to change `t_ref`'s meaning is the model
   owner's call (the toolkit reads); the health `refrac_measured` column is the number to quote, remembering it is a
   10 ms-frame-sampled estimate of 5 x 0.5 ms x rate (direct recompute 0.50924 vs the tool's 0.50909 at seed 0).
4. The undriven EPG sit ~250 mV below threshold under gD 15 with no lower clamp on `v`: the ring's pinning
   (`compass_room.md`) has a membrane-potential signature the tool can now track per cell, and `HealthReadout`'s
   `v_margin_mv` channel shows it live (display range 0-7 mV saturates; the statistics keep the value).
5. The walk-section DN operating point is not distinguishable between receptor default and off at n = 3, given the off
   arm's scatter -- not "the same": the arm means move 23-50 % (DN_all 1.33 vs 1.63 Hz, leg_MN 2.28 vs 2.56, wing_power
   15.1 vs 22.5) and wing_power separates perfectly at U = 0, p = 0.10, the floor of a 3-vs-3 rank test. More seeds, or
   a fourth arm (`holdOptic`, one `record --receptor-table out/receptors_holdOptic.csv` per seed), are what would settle
   whether the take-off cost has any DN-level signature at all.
6. The walk recorder is not reproducible at fixed seed past t ~ 2.11 s (section 4): the loom tail diverges (max
   |delta rate| 124.6 Hz, spike_count 9,247 vs 8,212). The 0.5-1.5 s scored window is bit-identical, so nothing here is
   affected -- but a round that scores the loom must re-establish determinism first.
7. `validate`'s `status: reproduced` for the sign-0 part is scored against the corrected reference
   (`nt_audit.md` 3,312), hard-coded at `scripts/interp_health.py` line 434, not against the shipped
   `common.VALIDATION['health']` 3,407 (section 2). Both that entry and `docs/NEUROME_INTERFACE.md` line 73 remain
   uncorrected and belong to their owners.
8. `record --protocol compass` duplicates `cx_wedge.simulate`'s stimulation loop instead of calling it (section 5);
   numerically equivalent on three seeds, but it is the one place in this tool that violates "unify, not duplicate".
