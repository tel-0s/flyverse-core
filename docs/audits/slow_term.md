# Slow (metabotropic / monoamine) term, round 2: design, coverage, sweep

Written 2026-09-12 (receptor integration round 2, task `score:slow`). Applies the critic's verdict on the round-1 slow
term (`docs/audits/receptor_verification.md`, "SLOW TERM"; `docs/NT_INTEGRATION.md` section 7, finding 5). Code:
`flyverse/brain.py` (`LIFParams.slow_*`, `SlowSpec`, `_slow_spec`, `_slow_weights`, the two Torch step loops),
`flyverse/optic.py` (`OpticLobe(slow=)`), `flyverse/connectome.py` (`SLOW_CLASSES`, `ReceptorSigns.slow_class`),
`flyverse/fly.py` (threading + state tensors), `scripts/benchmark.py` (flags), `tests/test_receptor_model.py`
(`SlowTermRound2Tests`, 7 new tests; 25 in the file, 55 with `test_nt_readout` / `test_world` / `test_control`, CPU 13.5 s;
one round-1 test line adapted to the per-class `W_slow` list).
Every number below comes from a named file: `out/slow_term_stats.json` (written by the statistics script of this task on
the adopted cache and the round-2 `flyverse/data/receptors_by_type.csv`, class rule, default gain classes), the cluster
batches `slow-sweep-f8a6a4` (`out/slow_<mode>_<gain>[_dop1r1].json` / `.txt`, 13 jobs) and `slow-unit-8336d4`
(`out/slow_u_<mode>_<gain>.json` / `.txt`, 7 jobs), and the Dop1R-only rebuild (`out/receptors_by_type_dop1r1.csv`).

Denominators (`out/slow_term_stats.json`): MaleCNS v1.0 node set 167,106 neurons, 25,578,600 stored entries of `W`,
121,460,584 |W| synapses; 124,161,872 raw synapses when the explicit-zero (monoamine / unknown presynaptic) entries are
counted from `cache/sign0_counts.npz`. Every fraction in this file says which of the two it is against.

## 1. What was wrong in round 1 (the critic's five points) and what changed

| # | round-1 finding | round-2 change |
|---|---|---|
| a | 98.4 % of the slow entries were classical-metabotropic (mAChR-A/B, GABA-B, mGluR); the monoamines got a slow sign on 12.2 % of their synapses | the entries are split into two classes by the presynaptic transmitter (`connectome.SLOW_CLASSES`: `metabotropic_classical` 5,065,951 entries / 21,761,304 syn-eq; `monoamine` 88,172 / 232,708). Each class has its own scale and time constant (`LIFParams.slow_gain_by_class`, `slow_tau_by_class`); the classical class defaults to scale 0. The monoamine share is unchanged by design (it is a table property: section 3). |
| b | the dopamine "+" rests on DopEcR in 560 of 607 types | `benchmark.py --dopamine-lead dop1r1` rebuilds the table with the dopamine slow + group = Dop1R1 / Dop1R2 (DopEcR ignored) through `build_receptor_table`'s own rule and scores it as one extra column (section 5) |
| c | 4x the integrated charge of a fast synapse per synapse (0.1 x 200 / 5); one tau for all classes | scale swept at 0.01 / 0.02 / 0.03 (0.4x / 0.8x / 1.2x the fast charge); per-class tau (monoamine 200 ms, classical 100 ms, both parameters) |
| d | an additive membrane current, where the animal's monoamines mostly scale gain; no term in the rate optic lobe | `slow_mode` = `additive` / `gain` / `threshold` (section 2); the same term with the same classes, scales and taus in `optic.OpticLobe` (section 2.3) |
| e | forces the Torch path; 13.3 vs 9.6 min per suite | still Torch-only (no native kernel carries the tone), but with every class scale 0 nothing is built and the step loop does no slow work (`test_zero_gains_cost_nothing_and_equal_sign_gain`: byte-identical state to `sign+gain` on the Torch path after 200 steps). `benchmark.py --receptor-gain low,mid,high` (added for section 5b) runs `full` on the `sign` fast weights with `1,1,1` |

## 2. Design

### 2.1 Classes

`connectome.receptor_signs` now returns `slow_class` per stored entry (int8 into `SLOW_CLASSES = ["none",
"metabotropic_classical", "monoamine"]`): 0 wherever the slow sign is 0, else 1 for a classical presynaptic transmitter
(the row's mAChR-A / mAChR-B, GABA-B, mGluR call) and 2 for dopamine / octopamine / serotonin. `ReceptorSigns.slow_factor
(gain, slow_class=name)` restricts the per-entry multiplier to one class. `brain._slow_weights` builds one capped
synapse-equivalent matrix per class with a non-zero scale; `Brain` keeps one tone state per class, `g_slow_cls` (K, B, N),
and their sum `g_slow` (B, N) (both in `FlyBrain.BRAIN_TENSORS`, so state save / restore and CUDA-graph capture carry them).

Per transmitted presynaptic spike the class tone jumps by `slow_gain[class] x w_syn x min(count, conn_cap) x slow sign x
gain-class factor` (fan-in scaled like the fast weights; path gains and same-type damping NOT applied, as in round 1) and
decays with `slow_tau[class]`. The steady tone of a presynaptic cell at rate R is therefore `slow_gain x tau_slow /
tau_syn` times the fast conductance the same synapses would carry: gain 0.01 / 0.02 / 0.03 at tau 200 ms = 0.4x / 0.8x /
1.2x (round 1: 0.1 -> 4x). Defaults: `slow_gain` (monoamine) 0.02, `slow_tau_ms` (monoamine) 200, classical scale 0.0 and
tau 100 ms (`DEFAULT_SLOW_GAIN_BY_CLASS`, `DEFAULT_SLOW_TAU_BY_CLASS`; the 100 ms is the GABA-B IPSP scale and has no
calibration behind it -- the class is off).

### 2.2 Modes (`LIFParams.slow_mode`)

With `gap = v_th - v_rest = 7 mV` (`SlowSpec.norm_mv`) and `g_slow` the summed tone in mV:

| mode | membrane target | spike condition | meaning of a +7 mV tone |
|---|---|---|---|
| `additive` | `v_rest + g + drive - adapt + g_slow` | `v >= v_th` | a 7 mV depolarising current (round 1's form) |
| `gain` | `v_rest + g x clamp(1 + g_slow / gap, 0, 4) + drive - adapt` | `v >= v_th` | the target's fast synaptic input is doubled; -7 mV silences it (factor 0, never inverted); injected `drive` is not scaled |
| `threshold` | `v_rest + g + drive - adapt` | `v >= v_th - min(g_slow, 0.9 gap)` | the threshold sits 0.7 mV above rest; a negative tone raises it without bound |

The normalisation by the rest-to-threshold gap makes the three modes comparable at the same `slow_gain`: the tone that
would carry an isolated cell to threshold under `additive` doubles its input under `gain` and puts its threshold at
rest (capped at 0.9 of the way) under `threshold`. `test_gain_mode_scales_the_fast_input` and
`test_threshold_mode_shifts_the_threshold` check the three targets on the two-neuron graph (dopamine -> TA, slow +, mid,
20 synapses, tau -> inf: tone 0.55 mV; a fast conductance of 2 mV is scaled by 1 + 0.55 / 7; a membrane at
v_th - 0.275 mV fires under `threshold` and not under `additive` with the tone removed; the clamps hold at +-100 mV).

### 2.3 The optic-lobe term (`optic.OpticLobe(slow=SlowSpec)`)

The rate units receive the same classes, scales and taus. Per active class k the unit i carries a tone that relaxes
with tau_k towards

    scale_k x (gain_fb x sum_s Wslow_is s_s + gain_rr x sum_j Wslow_ij dr_j),   scale_k = gain_k x tau_k / tau_syn,

with `Wslow` = count x slow sign x gain-class factor of the row, normalised by the same denominators as the fast
optic weights (`in_syn_l2` under `norm = "l2"`; no pair gains), `s` = spiking rate / 100 Hz held per frame and `dr` the
rate units' deviations (per substep). The steady tone of a presynaptic cell is thus `gain_k tau_k / tau_syn` times the
fast input the same synapses would give -- the LIF's ratio. `g_slow = sum_k` acts per mode: `additive` adds it to the
unit's input; `gain` multiplies the fast synaptic input (recurrent + photoreceptor + spiking) by `clamp(1 + g_slow /
(1 - baseline), 0, 4)` -- the normaliser is the distance from the operating point (0.5) to saturation, the counterpart
of the LIF's gap; `threshold` shifts the output nonlinearity, `r = clip(v + b + min(g_slow, 0.9 (1 - b)), 0, 1)`. State
`g_slow_cls`, `g_slow`, `_slow_in_s` are in `FlyBrain.OPTIC_TENSORS`. An active term runs the Torch substep (the native
optic kernels do not carry it; a warning if they were requested). `test_optic_slow_term` checks the steady tone
(`gain tau / tau_syn x gain_fb x 20 / l2-norm` with an octopamine cell at 100 Hz), that the round-1 optic model is
unchanged without the spec, and the three modes on a 4-cell graph.

What the LIF / optic split leaves out: the LIF prunes frozen (rate) presynaptic cells, so monoamine slow entries FROM
rate units ONTO spiking targets are in neither model: 97 entries / 394 syn-eq (Mi19 53 entries, l-LNv 39, Lat3 5; `out/
slow_term_stats.json` `monoamine_slow_landing.onto_spiking_from_rate_DROPPED`). Everything else lands: onto spiking
targets from spiking cells 51,971 entries / 152,217 syn-eq (LIF), onto rate units from spiking cells 27,641 / 68,602 and
from rate units 8,463 / 11,495 (optic; the rate -> rate part is the per-substep term).

### 2.4 Cost and path

`receptor_model == "full"` still forces the Torch LIF path (and, when the optic term is active, the Torch optic
substep). With every class scale 0 (`--slow-gain-monoamine 0 --slow-gain-classical 0`) `_slow_spec` returns None: no
slow matrices, `g_slow_cls` has shape (0, B, N), the step loop skips the slow block, and the dynamics are byte-identical
to `sign+gain` on the Torch path (test). With the term active the per-step cost is one sparse product per class (the
monoamine class: 88,172 entries, 0.34 % of the 25.6 M fast entries) plus a few elementwise ops; the eager suite ran in
20-24 min per job with 13 jobs sharing 8 B200s (section 5), against 11.4 min for the eager `off` control of round 1 alone.

## 3. Coverage of the slow term (adopted cache, round-2 table, class rule; `out/slow_term_stats.json`)

Per class, over all 25,578,600 entries / 124,161,872 raw synapses (sign-0 entries counted from `sign0_counts.npz`):

| class | entries | % of entries | syn-eq (count x abs(sign)) | % of raw syn | + syn-eq | - syn-eq | gain-weighted + / - |
|---|---|---|---|---|---|---|---|
| none | 20,424,477 | 79.85 | 102,167,856 | 82.29 | - | - | - |
| metabotropic_classical | 5,065,951 | 19.81 | 21,761,304 | 17.53 | 10,431,505 | 11,329,798 | 10,759,190 / 12,918,699 |
| monoamine | 88,172 | 0.34 | 232,708 | 0.19 | 111,300 | 121,408 | 136,302 / 167,099 |

Classical class by presynaptic transmitter: ACh 3,067,623 entries / 13,269,862 syn-eq (+ mAChR-A 2,250,674 entries,
- mAChR-B 816,949); GABA 1,434,290 / 6,418,661 (all -, GABA-B); glutamate 564,038 / 2,072,780 (all -, mGluR). By target
module: optic 3,084,669 entries / 13.1 M syn-eq, visual projection 803,782 / 3.41 M, mushroom body 720,269 / 2.40 M,
antennal lobe 230,870 / 1.34 M, central 203,553 / 1.41 M, descending 12,192 / 47,842. This is the class that is OFF by
default.

Monoamine class (the term that is on): by presynaptic transmitter, over the 1,877,078 raw monoamine synapses
(dopamine 590,338, octopamine 628,363, serotonin 658,377 = 1.51 % of the raw total):

| transmitter | raw syn | matched syn (profiled target) | slow + syn | slow - syn | mixed (equal classes) syn | matched, no receptor group | share with a slow sign |
|---|---|---|---|---|---|---|---|
| dopamine | 590,338 | 353,542 (59.9 %) | 41,160 | 42,038 | 270,344 (KC DopEcR-vs-Dop2R ties) | 0 | 14.1 % |
| octopamine | 628,363 | 157,368 (25.0 %) | 39,029 | 47,649 | 56,500 | 14,190 | 13.8 % |
| serotonin | 658,377 | 71,638 (10.9 %) | 31,111 | 31,721 | 5,595 | 3,211 | 9.5 % |
| all | 1,877,078 | 582,548 (31.0 %) | 111,300 | 121,408 | 332,439 | 17,401 | 12.4 % |

By target module: optic 36,128 entries / 80,137 syn-eq, mushroom body 25,041 / 64,797, central 13,428 / 59,390, visual
projection 8,841 / 17,213, antennal lobe 3,747 / 8,453, descending 736 / 2,297, mechanosensory 224 / 372, gustatory
27 / 49. The octopaminergic visual-centrifugal output (94,986 entries / 241,368 raw syn): 102,253 syn (42.4 %) onto
profiled targets, 73,424 (30.4 %) with a slow sign, of which 64,207 land on rate units -- this is the part the optic
term adds; the rest (69.6 %) carries no slow sign (unprofiled target, or Oamb/Octbeta vs Octalpha2R tie).

Per spiking target cell (capped, gain-weighted, fan-in scaled syn-eq; `top_spiking_targets_by_net_monoamine_slow_syn_eq`):
26,073 of 167,106 cells (15.6 %) receive any monoamine slow input, of which 17,973 are optic rate units (the optic term) and 8,076 of the 71,625 spiking non-photoreceptor cells (11.3 %); the 10.4 % in out/slow_term_stats.json is 8,100 / 77,716 non-rate cells. The largest positive loads are the octopaminergic cells
themselves (OA-VPM3 +525 per cell, OA-VUMa3 +274, OA-VUMa8 +239, OA-VPM4 +209, OA-VUMa6 +179: Oamb / Octbeta on
octopaminergic neurons -- 760 OA -> OA entries, 2,919 syn-eq, all +, 232 same-type; same-type damping is not applied
to the slow matrix, so this is a positive feedback loop of the term), DNge150 +428, EL +239 (18 cells), OA-AL2i3 +166,
KCa'b'-m +140.5 (205 cells; KCg-m +1.4, KCab-m +0.6 only, their DopEcR-vs-Dop2R rows are mixed), ER3d +110-122,
MBON11 +121. The largest negative: ExR3 -825, PRW068 -524, FB4M -358, SMP503 -314, PPL101 -136, PAM08 -21.7, LPLC2
-8.8; LC4 +2.9, MBON01 -1.8. The implied steady tone of the top cell with every + input at 50 Hz is 14.4 / 28.9 / 43.3 mV
at gain 0.01 / 0.02 / 0.03 (KCa'b'-m: 3.9 / 7.7 / 11.6 mV) -- i.e. the sweep spans "below the 7 mV gap for all but a
few dozen cells" to "several gaps for the octopaminergic cluster". The GF (DNp01), MN9, DNp18 and GNG175 have no
monoamine slow input (unprofiled, as in round 1).

## 4. The Dop1R-only variant (`--dopamine-lead dop1r1`)

`scripts/benchmark.py --dopamine-lead dop1r1` patches `build_receptor_table.RECEPTOR_GROUPS["dopamine"]["slow"]` to
`[(+1, "Dop1R", ["Dop1R1", "Dop1R2"]), (-1, "Dop2R", ["Dop2R"])]` and rebuilds the table in memory with the shipped
rule (11-13 s without the raw weights, which only feed the per-type synapse columns: the rebuild under the unpatched
groups reproduces every sign / class / tier / source column of the shipped table exactly, checked offline). A simpler
runtime mask ("a DopEcR-led row loses its + group") agrees with the exact rebuild on only 307 of 609 dopamine rows
(Dop1R1 / Dop1R2 are on below DopEcR in 450 of the 562 DopEcR-led rows), so the exact rebuild is used. Dopamine rows
(609 type rows; `out/receptors_by_type_dop1r1.csv`):

| | shipped (Dop1R1 / Dop1R2 / DopEcR) | dop1r1 (Dop1R1 / Dop1R2) |
|---|---|---|
| slow_net +1 / -1 / mixed / none | 400 / 131 / 78 / 0 | 86 / 188 / 251 / 84 |
| + lead Dop1R1 / Dop1R2 / DopEcR / none | 38 / 9 / 562 / 0 | 323 / 174 / - / 112 |
| transitions shipped -> dop1r1 | +1 -> +1 83, -> mixed 222, -> none 84, -> -1 11; -1 -> -1 122, -> mixed 8, -> +1 1; mixed -> -1 55, -> mixed 21, -> +1 2 | |
| KC rows | KC +1, KCa'b'-m +1, KCab-p -1, the other 12 mixed | KC +1, KCa'b'-m +1 (now Dop1R1-led), KCab-p mixed, the other 12 mixed |

The 38 Dop1R1-led and 9 Dop1R2-led shipped rows keep their sign in 34 / 9 cases. Monoamine slow entries on the graph:
82,418 / 251,794 syn-eq under dop1r1 (88,172 / 232,708 shipped): fewer entries (84 types lose the group), more
synapse-equivalents (the -1 rows gained by Dop2R now un-tied carry larger counts).

## 5. Cluster sweep (`slow-sweep-f8a6a4`; `benchmark.py --eager --receptor-model full --seeds 0,1,2`, classical 0)

Two batches, 20 jobs, all on the adopted (TYPE_NT_OVERRIDE) cache (`--cache-dir <cluster-fs>/neurome/runs/ntov-r2-e7706e/
cache_override`, the one the override-adoption run built on the cluster; `nt_counts.serotonin` 415 in every JSON), eager
Torch path, seeds 0,1,2 for the loom section, one run per setting (the suite is chaotic: round 1 measured a run-to-run
scatter of ~10 Hz on `walk_gf.p99` and 23-42 Hz on the loom GF peak, so a single PASS / FAIL within a few Hz of a
bound is not a verdict). `walk.power_max_hz` (wing-power MN under walking optic flow) FAILS in the reference off run
too (`out/rm_off.json`, 79.1 Hz vs < 50) and is not counted below. Files: `out/slow_<mode>_<gain>[_dop1r1].json/.txt`
(batch `slow-sweep-f8a6a4`, 13 jobs, 16-24 min each with 13 sharing 8 B200s), `out/slow_u_<mode>_<gain>.json/.txt`
(batch `slow-unit-8336d4`, 7 jobs, `--receptor-gain 1,1,1`). Columns: the scores the task named; "runaway 4" =
rest.spikes_per_step < 5, walk_gf.p99 < 38, loom_escape.GF_peak >= 33, wind.DNp18_flip >= 15; KC Hz is
`sections.smell.KC_hz` (not a suite check; the critic's < 5); loom per seed = the three seeds' GF peaks.

### 5a. `--receptor-model full` as specified: default gain classes (low 0.5 / mid 1 / high 1.5), classical 0

| run | mode | monoamine gain | rest spk/step | KC Hz (active) | walk_gf p99 (median) | loom GF peak (per seed) | escapes | wind DNp18 | taste MN9 | bitter cal sugar / +bitter | Shiu sugar / +bitter | rotation flip | pass/fail/gap | runaway 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_0 (term off = `sign+gain`) | - | 0 | 0 | 5.58 (2220) | 80.5 (43) | 93.2 (65.9, 58.3, 93.2) | 2 | 45.8 | 3.53 | 5.84 / 0.00 | 132 / 5.97 | -3.25 PASS | 25/2/2 | FAIL walk_gf |
| additive_01 | additive | 0.01 | 0 | 3.80 (2188) | 71.0 (40) | 82.8 (68.9, 65.2, 82.8) | 2 | 44.0 | 6.31 | 8.12 / 0.00 | 121 / 0.29 | -2.62 FAIL | 24/3/2 | FAIL walk_gf |
| additive_02 | additive | 0.02 | 0 | 5.94 (2228) | 79.4 (43) | 47.6 (45.1, 47.6, 47.0) | 3 | 43.9 | 2.06 | 5.28 / 0.00 | 67.3 / 5.09 | -2.15 FAIL | 24/3/2 | FAIL walk_gf |
| additive_03 | additive | 0.03 | 0 | 4.27 (2288) | 80.2 (49) | 68.8 (68.8, 27.7, 38.3) | 2 | 44.0 | 1.92 FAIL | 6.00 / 0.00 | 56.1 / 3.48 | -1.81 FAIL | 23/4/2 | FAIL walk_gf |
| gain_01 | gain | 0.01 | 0 | 1.81 (2520) | 69.2 (57) | 57.4 (57.4, 46.8, 55.4) | 2 | 43.1 | 3.92 | 6.04 / 0.00 | 117 / 4.15 | -1.90 FAIL | 24/3/2 | FAIL walk_gf |
| gain_02 | gain | 0.02 | 0 | 0.91 (783) | 70.2 (39) | 69.9 (54.5, 69.9, 66.4) | 2 | 44.0 | 11.68 | 5.90 / 0.00 | 119 / 1.01 | -1.84 FAIL | 24/3/2 | FAIL walk_gf |
| gain_03 | gain | 0.03 | 0 | 1.87 (1093) | 71.1 (46) | 58.1 (47.7, 58.1, 45.6) | 3 | 43.0 | 6.62 | 6.07 / 0.00 | 115 / 1.19 | -1.70 FAIL | 24/3/2 | FAIL walk_gf |
| threshold_01 | threshold | 0.01 | 0 | 1.87 (2396) | 66.6 (38) | 76.2 (49.9, 76.2, 73.6) | 3 | 45.6 | 4.54 | 4.84 / 0.00 | 113 / 0.03 | -1.55 FAIL | 24/3/2 | FAIL walk_gf |
| threshold_02 | threshold | 0.02 | 0 | 6.01 (2361) | 51.1 (41) | 61.4 (60.3, 61.4, 50.6) | 3 | 44.6 | 3.93 | 5.46 / 0.00 | 69.4 / 0.00 | -2.56 FAIL | 24/3/2 | FAIL walk_gf |
| threshold_03 | threshold | 0.03 | 0 | 1.88 (1985) | 74.4 (41) | 49.9 (49.9, 48.2, 46.0) | 1 | 44.5 | 4.59 | 8.35 / 0.00 | 139 / 0.03 | -2.69 FAIL | 24/3/2 | FAIL walk_gf |
| additive_02_dop1r1 | additive | 0.02 | 0 | 4.43 (2356) | 70.1 (38) | 77.5 (43.0, 77.5, 40.1) | 2 | 43.5 | 2.06 | 5.28 / 0.00 | 95.6 / 2.38 | -2.52 FAIL | 25/2/2 | FAIL walk_gf |
| gain_02_dop1r1 | gain | 0.02 | 0 | 1.65 (2326) | 63.9 (41) | 98.7 (98.7, 72.3, 44.0) | 2 | 45.0 | 11.68 | 5.90 / 0.00 | 125 / 2.91 | -2.48 FAIL | 24/3/2 | FAIL walk_gf |
| threshold_02_dop1r1 | threshold | 0.02 | 0 | 3.84 (2337) | 71.9 (48) | 71.4 (71.4, 34.3, 57.4) | 3 | 44.1 | 3.93 | 5.46 / 0.00 | 55.9 / 0.01 | -2.06 FAIL | 24/3/2 | FAIL walk_gf |
| round-1 full (0.1 every class, additive; `out/rm_full.json`) | additive | 0.1 (+ classical 0.1) | 0 | 37.3 (597) | 6.2 | 0.0 (0, 0, 0) | 2 (hops) | 4.6 FAIL | 5.11 | 5.76 / 0.00 | 74.2 / 1.81 | -2.67 FAIL | 22/5/1 | FAIL loom, wind |

Reading: (1) the round-1 runaway is gone -- rest 0 spikes/step, KC 0.9-6.0 Hz (round 1: 37.3), wind DNp18 43-46 Hz
(round 1: 4.6), loom GF peak 48-99 Hz (round 1: 0) in every setting, at every gain and in every mode. (2) Nothing passes
the four checks because `walk_gf.p99` fails in every row INCLUDING the term-off control (80.5 Hz; per-second GF median
38-57 Hz against the reference's 13; 3 voluntary hops in each 5 s walking window of the loom section): this is the
`sign+gain` fast-weight storm of round 1 (`out/rm_signgain.json`: p99 68.27, the same three failures), inherited by
`full`, not the slow term. (3) The slow term's own signature at these gains is the `rotation.group_flip_hz` check
(DNp20 + HSN + HSE (L-R) under yaw): -3.25 in the control (bound <= -3; -1.73 under round-1 `sign+gain`, i.e. marginal
already), -1.5 to -2.7 in all 12 active runs; and `taste.MN9` scatter (1.9-11.7 Hz, one FAIL at additive 0.03).
(4) Dop1R-only (one extra column per mode at 0.02): no consistent direction on any score -- KC 4.43 vs 5.94 (additive),
1.65 vs 0.91 (gain), 3.84 vs 6.01 (threshold); loom / walk_gf within scatter; taste and bitter identical to 2 decimals
in the additive and gain rows (the sugar -> MN9 path has no dopamine slow input, so those sections do not see the
table change). The variant cannot be distinguished from the DopEcR rule on this suite; it changes 82,418 entries /
251,794 syn-eq of monoamine slow input against 88,172 / 232,708 and is the biologically defensible rule (section 4).

### 5b. Same term on the `sign` fast weights (`--receptor-gain 1,1,1`: gain classes off), classical 0

| run | mode | monoamine gain | rest spk/step | KC Hz (active) | walk_gf p99 (median) | loom GF peak (per seed) | escapes | wind DNp18 | taste MN9 | bitter cal sugar / +bitter | Shiu sugar / +bitter | rotation flip | pass/fail/gap | runaway 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u_control_0 (term off = `sign`) | - | 0 | 0 | 1.68 (704) | 19.5 (11) | 78.1 (51.6, 78.1, 46.5) | 3 | 45.4 | 4.94 | 4.84 / 0.00 | 6.63 FAIL / 2.03 | -6.34 PASS | 25/2/2 | PASS |
| u_additive_01 | additive | 0.01 | 0 | 1.94 (771) | 38.5 (14) FAIL | 62.3 (42.4, 56.3, 62.3) | 3 | 45.5 | 4.21 | 7.79 / 0.00 | 129 / 2.67 | -8.86 PASS | 25/2/2 | FAIL walk_gf (38.5 vs < 38) |
| u_additive_02 | additive | 0.02 | 0 | 3.45 (1353) | 24.3 (13) | 61.3 (44.4, 61.3, 46.7) | 3 | 46.7 | 7.15 | 7.69 / 0.00 | 65.8 / 2.30 | -5.76 PASS | 26/1/2 | PASS |
| u_gain_01 | gain | 0.01 | 0 | 3.05 (1108) | 20.3 (14) | 55.7 (55.7, 50.8, 42.0) | 3 | 45.5 | 5.41 | 6.97 / 0.00 | 114 / 2.97 | -6.19 PASS | 26/1/2 | PASS |
| u_gain_02 | gain | 0.02 | 0 | 0.52 (485) | 31.3 (15) | 53.4 (37.9, 53.4, 53.1) | 3 | 46.4 | 4.92 | 6.80 / 0.00 | 130 / 2.93 | -6.77 PASS | 26/1/2 | PASS |
| u_threshold_01 | threshold | 0.01 | 0 | 3.57 (1419) | 19.0 (11) | 51.2 (51.2, 42.7, 41.6) | 3 | 46.4 | 6.75 | 6.55 / 0.00 | 128 / 0.00 | -5.76 PASS | 26/1/2 | PASS |
| u_threshold_02 | threshold | 0.02 | 0 | 3.36 (1234) | 23.4 (11) | 61.3 (38.9, 40.8, 61.3) | 3 | 46.5 | 1.79 FAIL | 4.99 / 0.00 | 9.66 FAIL / 2.77 | -6.98 PASS | 24/3/2 | PASS |
| reference off, native (`out/rm_ntov.json`, override cache) | - | - | 0 | 2.62 (1426) | 19.2 (9) | 37.4 | 1 | 45.2 | 5.85 | 4.57 / 0.00 | 124 / 2.12 | -7.38 PASS | 26/1/2 | PASS |

Reading: on the `sign` fast weights the four runaway-sensitive checks pass in every setting but additive 0.01 (walk_gf
p99 38.5 against < 38 with a per-second median of 14 -- one outlier second, inside the ~10 Hz scatter). Four settings
pass every check the reference passes (26 / 1 / 2 = the reference's own tally): additive 0.02, gain 0.01, gain 0.02,
threshold 0.01. threshold 0.02 keeps the four but loses taste.MN9 (1.79) and the Shiu sugar check (9.66). Two further
observations: (a) the Shiu-rules sugar -> MN9 check, which `sign` alone breaks (6.63 Hz here, 5.6 in round 1: the
uncapped / unadapted network is a knife-edge of total activity), is above 50 Hz again under additive 0.02, gain 0.01,
gain 0.02 and threshold 0.01 (66-130 Hz) -- consistent with a knife-edge, not evidence of a mechanism; (b) the modes
differ where expected: `gain` 0.02 lowers the Kenyon cells (0.52 Hz, 485 active vs 1.68 / 704 off; the KC dopamine
rows are mixed or Dop2R-led, so the KC tone is negative and scales their PN input down) while `additive` 0.02 raises
them (3.45 / 1353); `threshold` 0.01 leaves KC 3.57 / 1419. No setting changed `dn.MDN_top` (149-154), `loom.GF_peak`
(a) (27-41), `odour` (15-19 / 4.2-5.1) or the `object` gap (LC10a |flip| <= 0.04).

## 6. Verdict and what precedes assay 7 (hunger)

1. The redesign removes the round-1 runaway (a): with the classical class off and the monoamine class at 0.01-0.03 x
   w_syn, the CNS is silent at rest (0 spikes/step in 20 / 20 runs), the wind and loom checks pass in 20 / 20, KC
   0.5-6.0 Hz. The remaining failure under `--receptor-model full` as specified (default gain classes) is
   `walk_gf.p99` at 51-80 Hz in 13 / 13 runs including the term-off control: it belongs to the `sign+gain` gain
   classes (round 1's untested 0.5 / 1 / 1.5 tertile factors), not to the slow term. **With the default gain classes no
   setting passes the four checks; the precondition for assay 7 is not met by `full` as configured.**
2. **With the gain classes off (`--receptor-gain 1,1,1`, the `sign` fast weights + the monoamine slow term) four
   settings pass the four runaway-sensitive checks and every other check the reference passes** (26 / 1 / 2 with the
   standing `walk.power_max` failure): `--slow-mode additive --slow-gain-monoamine 0.02`, `gain 0.01`, `gain 0.02`,
   `threshold 0.01`. This is the precondition the plan set for the hunger experiment (assay 7: starvation -> DAN / OA
   tone in `body.Metabolism` -> KC / GNG targets). Each is a single run of a chaotic suite; before any of them is adopted
   as a default it needs 3 replicates of walk_gf / loom_escape (round-1 protocol, `out/lo_*`), and the choice between
   them is a modelling decision the data here do not make: `gain` is the form the critic argued for (monoamines scale
   gain in the animal) and is the only mode that moves the Kenyon cells in the direction a Dop2R-led tone should
   (down), `additive` is the round-1 form, `threshold` is the cheapest.
3. The Dop1R-only sign rule is indistinguishable from the DopEcR rule on the suite (section 5a, three columns);
   since the DopEcR-led "+" rests on a receptor with micromolar affinity (the critic's point b) and the exact rebuild
   costs 13 s per run, `--dopamine-lead dop1r1` should be the rule of any hunger experiment, and the table builder
   should ship it as a column set (`slow_sign_dop1r1`) rather than a runtime rebuild -- a `build_receptor_table.py`
   change outside this task's files.
4. What the sweep did not test and what stands in the way of adopting the term as a default: (i) the classical
   metabotropic class stayed at 0 (5.07 M entries, 21.8 M syn-eq); its presynaptic role (release, adaptation) is not a
   postsynaptic tone at all and should be recast, if ever, as a modulator of `std_u` / `adapt_jump`; (ii) the
   octopamine -> octopaminergic-cell autoreceptor loop (760 entries, 2,919 syn-eq, all +, OA-VPM3 +525 syn-eq per cell:
   28.9 mV of steady tone at gain 0.02 if its inputs ran at 50 Hz) is a positive-feedback structure that no check in the
   suite exercises (the OA cells are silent in every section) but a hunger tone would; same-type damping or a per-cell
   cap of the slow load (like `input_norm_ref` for the fast weights) is the obvious guard; (iii) the term is Torch-only
   (16-24 min per suite under 13-way sharing, 11.4 min for the eager off run alone in round 1); a native kernel needs
   the per-class tone in `cuda.lif_update` and the optic kernels; (iv) the 97 rate -> spiking monoamine entries (Mi19,
   l-LNv) are in neither model.
5. Reproducibility notes: every JSON records `config.receptor.slow` (mode, gains, taus, active flag, entries per class),
   `config.receptor.gain_classes`, `config.receptor.dopamine_lead` and `config.cache_dir`; `cluster_run.py --fetch out/`
   copies into `out/out/` (scp nesting), the files were copied up by hand; the cluster's shared `cache/` still lacks the
   TYPE_NT_OVERRIDE and every run here used the override cache of `ntov-r2-e7706e` explicitly (its `sign0_counts.npz`
   is the shared cache's, a superset keyed by (post, pre), which is correct for the override graph's zero entries).

## Corrections (round-2 verification, `verify:score:slow`)

* "'gain' is the only mode that lowers the Kenyon cells as a Dop2R-led tone should" is withdrawn: on the 1,1,1 weights the control is KC 1.68 Hz and gain 0.005 / 0.01 / 0.015 / 0.02 give 3.77 / 3.05 / 0.94 / 0.52 Hz -- not monotone -- and the KC populations' own net monoamine tone is positive (KCa'b'-m +93.6 syn-eq / cell, KCg-m +1.5; only KCab-p strongly negative, -112.4), so the drop at gain >= 0.015 is not the Dop2R story of section 5b. KC_hz spans 0.5-6 Hz across modes with no separable mode ordering.
* 'gain' mode multiplies the NET synaptic input (brain.py `_membrane_target`, optic.py `_substep`): a negative tone reduces the inhibition of a net-inhibited target (disinhibits) and silences only net-excited ones. The monoamine class is net-negative by synapse-equivalents (121,408 - vs 111,300 +), so over much of the graph 'gain' acts as disinhibition. The LIFParams comment and section 2.2 were corrected; a per-sign gain needs separate excitatory / inhibitory accumulators.
* `--dopamine-lead dop1r1` is not "indistinguishable": taste.MN9 and the calibrated bitter checks are bit-identical per pair, but the Shiu bitter checks differ (additive 0.02 sugar 67.3 -> 95.6 Hz; gain 0.02 118.8 -> 124.9; threshold 0.02 69.4 -> 55.9); the sentence "the sugar -> MN9 path has no dopamine slow input" is withdrawn.
* Isolation: with the classical class at 0 the monoamine class runs at the round-1 scale 0.1 without the runaway (KC 7.88, loom_escape 66.8 PASS, rotation -3.49 PASS) -- the runaway removal is the class split, not the 0.1 -> 0.02 scale.
* Scatter of the adopted setting over three identical reruns: walk_gf.p99 31.3 / 26.5 / 29.0 Hz, loom_escape 53.4 / 51.1 / 53.4, rotation -6.8 / -5.3 / -6.1; everything outside the room simulation is bit-reproducible. So u_additive_01's 38.5 FAIL is within one scatter width of a PASS and gain 0.005 / 0.015 also reach 26/1/2.
* walk.power_max over the four 26/1/2 runs is 70.4-82.3 Hz (65.48 belongs to u_additive_01). Runtime 15-24 min per job across the two batches.
* `benchmark.py --receptor-model full` without `--eager` used to abort every room section (warp requested while the slow term forces the Torch path) and still print a pass count with 9 checks MISSING; brain.py now downgrades warp to cuSPARSE with a warning when a model option turns the kernels off.
* The generator of out/slow_term_stats.json is now `scripts/slow_term_stats.py`.
