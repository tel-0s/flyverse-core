# The monoamine slow class at data-anchored magnitudes (behaviour round 3, thread:monoamines)

Written 2026-09-14 on HEAD d2abf3c (working tree; nothing committed). Files of this thread:
`scripts/build_monoamine_tables.py` (coverage + bracket, CPU), `scripts/probe_monoamines.py` (`arms` / `batch` / `record` /
`analyse` / `taste`, the last a CPU replay of the benchmark's taste section with the realised tone read off the Brain),
`tests/test_monoamines.py` (7 tests, CPU, 5 s), five `lit.*` op-report rows appended to
`flyverse/data/expected_responses.csv`, and the scratch tables under `out/monoamines/` (md5 in `out/monoamines/md5.txt` for
the coverage tables and `out/monoamines/md5_runs.txt` for the run-side files: `analysis.json` / `.txt`, `taste/*.json`;
the shipped `flyverse/data/receptors_by_type.csv`, md5 `0381a446107e6050e75cc87b16d7f830`, is untouched). The GPU batch is
`mono-fd76d2` (25 jobs, `out/mono_cluster.log`, `out/monoamines/runs/`, section 3); its client died with the predecessor
session, the finished jobs were pulled from both boxes by hand (`out/monoamines/fetch.log`: 155 files, 0 failed). No default
moved: `LIFParams()` is the reference arm and `tests/test_monoamines.py::test_off_arm_is_the_shipped_default_and_no_default_moved`
pins `receptor_model 'sign' / abs`, `slow_gain 0.02`, `slow_tau_ms 200`, `slow_mode additive`, `_slow_spec(LIFParams()) is None`.

Dependencies not at origin/main that the cluster batch shipped (working tree, other threads; `out/monoamines/shipped_diffstat.txt`):
`flyverse/brain.py` (+30/-1: `LIFParams.w_syn_by_nt = None`, thread unitary, opt-in), `flyverse/body.py` (+134/-0),
`flyverse/batch_body.py` (+26/-4), `flyverse/batch_sim.py` (+5/-1), `flyverse/motor.py` (+29/-0), `flyverse/senses.py` (+157/-12)
(thread body-state, all opt-in by their own tests; `git diff --numstat origin/main -- flyverse/` at submission). Every run's `provenance.source_fingerprint` is the code identity;
`flyverse_commit.commit` reads `unknown` on the boxes (no `.git` in a run copy).

Denominators: MaleCNS v1.0 cache `ef23cc27bea13be7f6a96f3c04fd3737` (167,106 neurons, 25,578,600 stored entries,
121,460,584 |W| synapses, 124,161,873 raw synapses with the explicit-zero monoamine / unknown entries counted from
`cache/sign0_counts.npz`). Net rule `abs` (the shipped default) throughout; the class-rule numbers of
`docs/audits/slow_term.md` section 3 differ (there the KC dopamine rows were ties).

## 0. Answer

With the monoamine class switched on at the shipped fast weights (`receptor_model 'full'`, gain classes 1/1/1 = the `sign`
weights entry for entry; `tests/test_monoamines.py`), one class scalar, 5 runs x 8 flies x 30 s per arm on the H200s
(batch `mono-fd76d2`, every run `device cuda / NVIDIA H200`, torch 2.11.0+cu128; section 3):

* **`add_low` = the shipped `slow_gain 0.02`, additive** -- nothing the fly does changes: yaw-rate SD 2.73 +- 0.20 vs
  2.60 +- 0.29 deg/s, straightness 0.9971 +- 0.0005 vs 0.9981 +- 0.0004, speed 8.81 +- 0.03 vs 8.83 +- 0.04 mm/s, DNa02 R-L
  0.049 +- 0.018 vs 0.048 +- 0.016 Hz, take-offs 0.075 +- 0.17 vs 0.025 +- 0.06 per fly (5 runs each; every key `null`).
  Baseline activity moves by +3.0 % (53.70 +- 0.25 vs 52.14 +- 0.13 spikes/step, `result` but tiny), the mushroom body
  from 1.70 to 1.95 Hz (KCg-m 2.20 -> 2.58), OA-VPM3 6.8 -> 11.3 Hz; no type leaves 0 Hz in more than a single row of a
  single run (29 candidates, all `null` / `undetermined`); **the suite is 12 / 0 / 0 in 5 / 5 runs ON CUDA ONLY**. One
  seed-locked shift to watch: `taste.MN9_hz` 10.93 -> 2.48 Hz (bound > 2, 5 / 5 identical on each side). The CPU replay
  (section 3f) puts it where it belongs: at 0.02 every realised tone is below 0.9 mV (264 types non-zero, GNG572 -0.65,
  GNG056 -0.35, DNg30 -0.37, DNge150 +0.87), and MN9's rate is the residual of two ~1.5 V/s inputs (E +1,786 / I -1,677
  mV/s in `off`, +1,475 / -1,473 under `add_low`), so a sub-mV 5-HT tone on the SEZ tips it -- **and on the CPU it tips
  it past the bound: `taste.MN9_hz` 1.9669914245605469 against "> 2" = FAIL** (off on the same CPU 5.0909 = PASS;
  skeptic pass, `scratchpad/skept/bench_addlow_cpu.json` / `bench_off_cpu.json`). So the shipped `slow_gain` under
  `full` already fails a scored suite check on the CPU reference path -- the path the project rule uses for
  bit-identity -- and the 12 / 0 / 0 above is a CUDA statement, not a suite pass. The CPU-vs-H200 gap on this check is
  **CPU vs CUDA, not device-to-device**: the skeptic's B200 rerun reproduces the H200's seed-locked suite values to the
  last digit (`taste.MN9_hz` 10.93417739868164 on all 5 H200 and all 4 B200 off runs, 2.4795215129852295 on all 9
  add_low runs; `smell.PN_hz`, `smell.KC_active`, `walk.GF_max_hz`, `walk.power_sustained_hz` likewise identical), so
  an earlier wording here -- "the off values themselves differ 2x between devices" -- was wrong as a generalisation.
  It is a knife-edge readout, and it is knife-edged between backends.
  **Classification: null on behaviour and baseline; a FAIL on the taste section on the CPU path, a regression watch on CUDA.**
* **`add_mid` (0.2) and `add_high` (1.0), additive -- RUNAWAY, 5 / 5 runs each**, aborted by the rollout guard at
  t = 0.51 s (0.5 s running mean 604-680 and 1,273-1,277 spikes/step = 12.8x and 24.5x the off arm's 52.1; the rest guard
  never trips because the unstimulated brain is silent in every arm: rest 0.00 spikes/step, section 3a). It is round 1's
  runaway reproduced with the monoamine class alone: the whole mushroom body sits at the refractory ceiling within 250 ms
  (KCg-m 223 / 318 Hz vs 2.20, PAM01 276 / 319 vs 3.0, PPL101 317 / 321 vs 17, APL and DPM 318-321 vs 18 / 16, MBON01 314
  vs 16; 24 / 76 types above 300 Hz). The loop, in the shipped weights (`flyverse.interp.decompose`, static): KCg-m gives
  PAM08 163 mV per volley per cell (58.8 % of PAM08's input), PAM01 168 mV (60.9 %), PPL101 560 mV (35.4 %); back, KCg-m
  carries 75.8 dopamine syn-eq per cell (slow +1, Dop1R2-led, gain class high), i.e. 0.83 mV of tone per Hz of DAN rate
  at 0.2 -- **the DAN POPULATION at 8.4 Hz** puts every KCg-m at threshold (75.8 is the SUMMED dopamine load on a KCg-m
  cell over every dopaminergic presynaptic cell, so 8.4 Hz is the rate the whole population would have to reach, not a
  single DAN) -- against 0.083 mV/Hz at 0.02. The CPU replay of the taste section (no vision, sweet GRNs only) also runs
  away at 1.0 (22.7 % of all cells above 1 Hz, KCg-s1 tone +4,861 mV).
  **Classification: result, structural -- the structural evidence is the MAGNITUDE scatter (665.6 +- 3.2 and 1,277.0 +-
  0.5 spikes/step on the H200, 666.3 +- 0.5 on the skeptic's B200), not the abort time: the guard is only checked from
  frame 50 (`probe_monoamines.py`: `if k >= 50 and run_mean > args.abort_rollout`), so frame 51 is the earliest abort
  possible and every run sits on that floor by construction. Nothing at or above 0.2 additive is usable with one class
  scalar and the shipped table.**
* **`gain_mid` (0.2, gain mode)** -- no runaway; 2.55x the spikes (132.9 +- 0.6 spikes/step, flat over 30 s). Baseline:
  mushroom body 18.5 Hz (KCg-m 9.5, KCab-s 18.4, APL 149, DPM 135), the DANs 24 Hz (PPL105 154), the OA cells 102 Hz
  (OA-VUMa2 329, OA-VPM4 137, OA-VPM3 96), CSD 134 (vs 7.3), the antennal lobe doubles (uniglomerular PNs 18.9 vs 7.7 Hz,
  LNs 78 vs 34, multiglomerular PNs 260-320 Hz), the descending neurons 4.4 vs 1.8 Hz (DNp32 174 vs 15, DNge150 120 vs
  8.8); the ring LEAVES SILENCE (EPG 3.19 vs 0.03 Hz, PEN_a 2.91, PEN_b 1.65, Delta7 6.87, EL 19.2, ExR3 29.3, GLNO 7.4;
  44 of 292 CX types above 0.5 Hz, 154 still at 0 vs 204); and the MBONs are ZEROED (MBON01 / 03 / 05 / 06 / 09 at 0.00 Hz
  vs 15.6 / 6.7 / 19.9 / 11.1 / 15.9): their Dop2R-led rows carry **-276 to -876** syn-eq per cell across these five
  (`target_load.csv`: MBON01 276.2, MBON09 306.3, MBON05 485.9, MBON06 490.2, MBON03 875.8 -- the -480 .. -876 range an
  earlier version quoted here is section 1d's separate top-7 list, which MBON01 and MBON09 are not in; skeptic pass), so
  the factor 1 + g_slow / 7 mV clips to 0 for any DAN rate above ~0.7 Hz at -876 syn-eq and above ~2.3 Hz at -276, and
  `gain_mid`'s DANs run at 24 Hz -- the mode's documented artefact (a negative tone on a net-excited cell zeroes all of
  its input), unaffected by which end of the range a given MBON sits at. The plain fly: straightness 0.974 +- 0.008 vs 0.998 (`result`), yaw-rate SD
  4.47 +- 0.43 vs 2.60 +- 0.29 deg/s (`result`), speed 9.10 +- 0.40 vs 8.83 (`null`), the leg L-R bias doubles (0.332 vs
  0.157 Hz, `result`), take-offs 1.55 +- 0.46 vs 0.025 per fly per 28 s, all voluntary (`result`), DNa02 |R-L| 0.225 vs
  0.091 Hz (`result`, but at 0.1 Hz: the turning deficit's silent DNa02 is untouched, its mean rate 0.11 vs 0.05 Hz).
  Suite 12 / 0 / 0 in 5 / 5; `loom_escape.GF_peak_hz` 63.9 +- 3.3 vs 48.4 +- 1.8 (`result`, z 8.6), `smell.PN_hz` 25.0
  vs 7.9 and `smell.KC_active` 1,550 vs 816, `walk.power_sustained_hz` 39.2 +- 5.7 vs 20.1 (seed-locked shifts).
  **Classification: result -- a large, reproducible change of baseline activity that is not a behavioural gain**: the fly
  hops rather than turns, the MBONs go dark, and the monoaminergic cells amplify their own inputs (OA-VUMa2 at 329 Hz,
  the ceiling). Whether the ring activity is heading-locked was not tested here (`compass_room.md`'s rows are the test).
* **Coverage** (section 1): 72.8 % of the 1.88 M monoamine synapses are silenced in every model (69.0 % onto unprofiled
  targets, 3.8 % ties), the VNC / ascending / OA-VUM system that the locomotor-state physiology is about has 0 signed
  synapses, and the dopamine class is 59 % one system (DAN -> KC, +1). **Magnitude** (section 2): the two quantitative
  anchors are ~50-100x apart on one scalar -- the run shows both ends: 1.0 (the octopamine anchor's low edge) detonates the
  KC loop in 0.5 s, 0.02 (the KC ceiling) leaves the OA rows invisible (HSS tone 0.07 mV at 10 Hz).
* **Adoption**: nothing in this round can be adopted; what an adoption would require is section 4.

## 1. Coverage: which sign-0 synapses the monoamine class can carry (`out/monoamines/coverage_by_nt.csv`, `coverage_by_lead.csv`, `coverage_by_superclass.csv`, `top_pairs.csv`)

**The sign rule the table uses** (`scripts/build_receptor_table.py` RECEPTOR_GROUPS lines 110-148, cited in
`docs/audits/receptor_rules.md`): a receptor GROUP's sign is the direction of its postsynaptic effect in *Drosophila*
neurons, i.e. its G-protein coupling -- Gs / Gq = +1, Gi = -1: dopamine `Dop1R1` (Gs; Sugamori 1995 / Gotzes 1994),
`Dop1R2` / DAMB (Gs / Gq; Han 1996, Feng 1996) and `DopEcR` (Gs; Srivastava 2005) form the +1 group, `Dop2R` (Gi; Hearn
2002) the -1 group; octopamine `Oamb` (Gq; Han 1998) and `Octbeta1R / 2R / 3R` (Gs; Maqueira 2005) +1, `Octalpha2R` (Gi;
Qi 2017) -1; serotonin `5-HT2A` (Gq; Colas 1995), `5-HT2B` (Gq; Gasque 2013), `5-HT7` (Gs; Witz 1990) +1, `5-HT1A / 1B`
(Gi; Saudou 1992) -1. `Oct-TyrR` is in no expression source and no group. A (type, transmitter) row's slow sign is the
sign of the group with the larger summed expression under a 2-fold margin (`abs` rule); a tie or an absent group gives
0 = no `W_slow` entry. The gene identity does not enter the model (only sign, gain class, and the two-way slow class:
`connectome.receptor_signs`); `tests/test_monoamines.py::test_group_sign_rule_matches_the_table_builder` checks the
table above against `RECEPTOR_GROUPS` gene for gene.

### 1a. Per transmitter (sign-0 presynaptic bodies; raw synapses)

| transmitter | bodies | presyn with output | stored entries | raw syn | onto a profiled target (row exists) | slow +1 | slow -1 | tie / no group (0) | fallback (no row) | carried by `W_slow` | silenced in every model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dopamine | 395 | 395 | 244,694 | 590,338 | 353,542 (59.9 %) onto 441 types | 282,753 | 49,173 | 21,616 | 236,796 (40.1 %) onto 6,583 types | 56.2 % | 43.8 % |
| octopamine | 141 | 141 | 205,938 | 628,363 | 157,368 (25.0 %) onto 584 types | 63,405 | 51,878 | 42,085 | 470,995 (75.0 %) onto 10,562 types | 18.3 % | 81.7 % |
| serotonin | 415 | 415 | 211,550 | 658,377 | 71,638 (10.9 %) onto 592 types | 32,618 | 30,558 | 8,462 | 586,739 (89.1 %) onto 9,736 types | 9.6 % | 90.4 % |
| unknown | 2,361 | 1,732 | 254,444 | 824,211 | 0 | 0 | 0 | 0 | 824,211 (100 %) onto 11,419 types | 0 | 100 % |
| monoamines together | 951 | 951 | 662,182 | 1,877,078 | 582,548 (31.0 %) | 378,776 | 131,609 | 72,163 | 1,294,530 (69.0 %) | 27.2 % (510,385 syn) | 72.8 % |

"Silenced in every model" = fallback + tie: zero on the fast path (`abs(0) x 0`), absent from `W_slow`. What `W_slow`
carries under `full` with the shipped table: 215,579 entries, 497,256 capped synapse-equivalents (conn_cap 60), landing on
32,254 cells of which 10,092 are spiking (the other 22,162 are optic rate units, reached only by the opt-in
`OpticLobe(slow=)` term that `FlyBrain` wires and `benchmark.py`'s walk section does not).

### 1b. Per lead receptor (the gene that wins the row; synapses onto rows it leads)

| transmitter | lead of the winning group | sign | raw syn | entries | post types |
|---|---|---|---|---|---|
| dopamine | Dop1R2 | +1 | 166,209 | 88,057 | 9 (the Kenyon-cell types) |
| dopamine | DopEcR | +1 | 114,434 | 54,801 | 212 |
| dopamine | Dop1R1 | +1 | 2,110 | 521 | 24 |
| dopamine | Dop2R | -1 | 49,173 | 5,539 | 102 |
| dopamine | tie (Dop1R/DopEcR vs Dop2R within 2-fold) | 0 | 21,616 | 6,789 | 94 |
| octopamine | Octbeta2R | +1 | 31,567 | 15,206 | 178 |
| octopamine | Octbeta1R | +1 | 12,696 | 4,890 | 25 |
| octopamine | Oamb | +1 | 12,271 | 4,519 | 14 |
| octopamine | Octbeta3R | +1 | 6,871 | 3,447 | 139 |
| octopamine | Octalpha2R | -1 | 51,878 | 13,700 | 30 |
| octopamine | tie or no group expressed | 0 | 42,085 | 17,704 | 198 |
| serotonin | 5-HT7 | +1 | 27,951 | 10,609 | 206 |
| serotonin | 5-HT2A | +1 | 3,807 | 3,261 | 21 |
| serotonin | 5-HT2B | +1 | 860 | 720 | 53 |
| serotonin | 5-HT1A | -1 | 28,279 | 8,582 | 223 |
| serotonin | 5-HT1B | -1 | 2,279 | 1,727 | 16 |
| serotonin | tie or no group expressed | 0 | 8,462 | 5,233 | 73 |

Under the `abs` rule the DAN -> KC rows are no longer ties (the class rule's `mixed`): every Kenyon-cell type is +1,
Dop1R2-led (KCg-m, KCg-d, KCa'b'-ap1/-ap2/-m, KCab-m/-s/-p/-c) at gain class `high` -- 166,209 of the 282,753 positive
dopamine synapses sit on nine KC types. The dopamine "+" of the other 212 types rests on DopEcR (the round-2 critic's
point b; `--dopamine-lead dop1r1` exists in `benchmark.py` and is not used here: the task is the shipped table).

### 1c. Per postsynaptic superclass (raw synapses >= 5,000; `coverage_by_superclass.csv`)

| transmitter | postsynaptic superclass | raw syn | profiled | signed (+ / -) |
|---|---|---|---|---|
| dopamine | cb_intrinsic | 538,945 | 63.9 % | 60.0 % (276,122 / 47,200) |
| dopamine | visual_projection | 19,897 | 26.3 % | 26.3 % (4,187 / 1,049) |
| dopamine | ol_intrinsic | 19,392 | 13.5 % | 13.0 % |
| dopamine | descending_neuron | 5,719 | 5.8 % | 5.8 % (131 / 199) |
| octopamine | cb_intrinsic | 305,512 | 17.3 % | 14.8 % (22,654 / 22,585) |
| octopamine | ol_intrinsic | 177,367 | 49.7 % | 33.4 % (29,879 / 29,293) |
| octopamine | visual_projection | 45,263 | 29.6 % | 18.6 % (8,434 / 0) |
| octopamine | vnc_intrinsic | 44,705 | 0 | 0 |
| octopamine | descending_neuron | 17,097 | 2.9 % | 2.9 % (502 / 0) |
| octopamine | ascending_neuron | 14,879 | 0 | 0 |
| octopamine | visual_centrifugal | 11,199 | 13.8 % | 8.6 % |
| serotonin | cb_intrinsic | 441,315 | 10.0 % | 9.2 % (16,336 / 24,275) |
| serotonin | ol_intrinsic | 63,284 | 31.2 % | 25.3 % (13,820 / 2,221) |
| serotonin | vnc_intrinsic | 44,629 | 0 | 0 |
| serotonin | ascending_neuron | 39,184 | 0 | 0 |
| serotonin | descending_neuron | 21,272 | 8.1 % | 8.1 % (612 / 1,111) |
| serotonin | visual_projection | 15,904 | 14.5 % | 8.4 % |
| serotonin | vnc_motor | 7,082 | 0 | 0 |
| serotonin | cb_endocrine | 6,937 | 12.3 % | 12.3 % (0 / 850) |
| unknown | cb_intrinsic / vnc_intrinsic / ascending / descending / ... | 458,960 / 153,372 / 59,273 / 36,058 | 0 | 0 |

The VNC (vnc_intrinsic, vnc_motor, ascending) receives 0 signed monoamine synapses: no VNC type has a receptor row
(coverage 0, as `docs/audits/receptor_integration.md` 1 found for the fast rows). The 23 descending cells of 13 types
that do carry a tone (`target_load.csv`): DNge150 +291 syn-eq (5-HT +267), DNp32 -127 (5-HT -134, DA -45, OA +52),
DNge151 +113, DNg30 -63, DNp29 -49, DNg26 -78, DNg104 +79, DNge138 +77, DNpe048 -64, DNg66 +58, DNg34 +56, DNge149 +42;
DNa02, DNp01, DNp09, MDN, MN9 have none.

### 1d. The 20 largest (presynaptic type, postsynaptic type) pairs per transmitter (`top_pairs.csv`; raw syn, row sign, tier, lead)

Dopamine -- 18 of 20 are the DAN / DPM -> Kenyon-cell system, all +1 / high, Dop1R2-led (fuzzy tier) or DopEcR-led
(alias tier): PAM08 -> KCg-m 26,725; PAM01 -> KCg-m 20,477; DPM -> KCg-m 12,754; PPL103 -> KCg-m 11,468; PAM07 -> KCg-m
10,336; PAM06 -> KCa'b'-ap2 9,973; PAM12 -> KCg-m 9,045; PAM05 -> KCa'b'-ap1 8,883; PAM06 -> KCa'b'-m 7,639; PAM10 ->
KCab-s 7,589; PPL101 -> KCg-m 7,224; PAM04 -> KCab-s 5,617; PAM09 -> KCab-p 5,072 (mid); PAM10 -> KCab-m 4,841; DPM ->
KCab-s 4,741; PAM08 -> KCg-d 4,733; PAM11 -> KCab-s 3,919; PAM14 -> KCa'b'-m 3,761. The two that fall through: DPM ->
APL 3,521 and FB4M -> hDeltaB 3,391 (no row). Dopamine onto the MBONs is the largest NEGATIVE load per cell in the graph
(Dop2R-led): MBON03 -876 syn-eq / cell, MBON04 -631, MBON21 -604, MBON26 -495, MBON06 -490, MBON05 -486, MBON07 -479
(`target_load.csv`).

Octopamine -- the optic-lobe-projecting OA cells and the CX ring, not the antennal lobe / mushroom body: OA-AL2i1 ->
Dm12 7,454 (-1, Octalpha2R, exact); OA-AL2i2 -> TmY5a 6,828 (-1); OA-ASM1 -> TmY5a 6,492 (-1); OA-ASM1 -> MeLo8 4,796
(no row); OA-AL2i2 -> Pm6 4,270 (+1, Oamb, low, class tier); OA-AL2i2 -> Pm4 4,265 (tie); EL -> ER4d 4,178 (-1,
Octalpha2R, class tier); OA-AL2i1 -> TmY5a 4,072 (-1); OA-AL2i3 -> Dm18 3,951 (no row); EL -> EPG 3,709 (+1, Octbeta1R,
alias); OA-AL2i2 -> Pm3 3,483 (tie); OA-AL2i4 -> Cm7 3,336 (no row); EL -> ER3p_a 3,253 (-1); OA-AL2i2 -> MeLo12 3,146
(no row); OA-AL2i1 -> Dm10 3,076 (tie); OA-AL2i2 -> Tm6 2,918 (no row); OA-AL2i3 -> Dm1 2,785 (+1); EL -> ER3d_b 2,733
(-1); OA-AL2i2 -> T3 2,415 (-1, low); OA-AL2i4 -> Cm3 2,232 (no row). The VUM / VPM system that the physiology is about
(section 2) is almost entirely silenced: OA-VUMa1 40,467 raw syn of which 994 signed, OA-VPM3 34,231 / 5,255, OA-VUMa6
33,703 / 2,034, OA-VUMa8 32,377 / 437, OA-VUMa4 27,948 / 784, OA-VPM4 23,669 / 2,313 (`presyn_types.csv`), because their
targets (VNC, ascending, most cb_intrinsic types) have no receptor row.

Serotonin -- mostly fallthrough: Mi19 -> Dm18 7,870 (no row; Mi19 is a rate unit anyway); aMe17a -> Cm3 6,901 (no row);
PFGs -> hDeltaK 4,790 (no row); SAxx02 -> AN09B018 4,758 (no row); PRW006 -> DH44 3,655 (no row); Mi19 -> Tm6 3,499 (no
row); SAxx02 -> AN05B004 2,981 (no row); l-LNv -> Mi1 2,700 (+1, 5-HT7, exact; a rate -> rate pair of the optic term);
PFR_a -> hDeltaA 2,666 (no row); ExR3 -> ER3d_b 2,245 (+1, 5-HT7, class tier); ExR3 -> hDeltaK 2,092 (no row); CSD ->
lLN2F_b 2,015 (no row -- the AL LNs have no 5-HT row at all); l-LNv -> Tm3 1,978 (+1); PFGs -> FB6A_c 1,904 (no row);
ExR3 -> PFGs 1,823 (-1, 5-HT1A); SNpp23 -> ANXXX136 1,812 (no row); aMe17a -> MeVP6 1,798 (no row); aMe17a -> TmY10
1,733 (no row); ExR3 -> ER3d_d 1,558 (+1); FB4Y -> hDeltaA 1,539 (no row). The 5-HT -> CX system therefore reaches the
model only through the ExR3 -> ring rows (ER3d_b +, ER3d_d +, PFGs -; ExR3 itself carries the largest negative 5-HT load
of any cell, -554 syn-eq, 5-HT1A) and 5-HT -> AL only through PN rows (net + 4.48 / - 2.49 syn-eq per PN; 0 on every LN
type); the CSD -> LN synapses are silenced.

## 2. Magnitude anchor: what the literature gives, and the bracket it implies (`out/monoamines/anchor_bracket.csv`; ledger rows `lit.HS.oa_response_gain`, `lit.Mi4.oa_state_baseline`, `lit.KC.dan_tone_mv`, `lit.MBON11.kc_mbon_depression`, `lit.PN.5ht_endogenous_direction`, all op `report`)

The term's steady tone on a target cell is `g_slow = slow_gain x w_syn x L x R x tau` with `L` the capped, fan-in-scaled
synapse-equivalents from the monoamine cells (what `W_slow` holds; `tests/test_monoamines.py::test_bracket_formula_is_what_the_brain_integrates`
checks the formula against `Brain` on the two-neuron graph: 0.2 x 0.275 mV x 20 x 100 Hz x 0.2 s = 22 mV, measured 22 +- 15 %),
`R` the presynaptic rate and `tau` 200 ms. The modes are normalised by the 7 mV rest-to-threshold gap
(`docs/audits/slow_term.md` 2.2), so one `slow_gain` gives the same tone in mV under `additive` and the factor
`1 + g_slow / 7 mV` under `gain`. No *Drosophila* octopaminergic / dopaminergic / serotonergic walking rate is published
in Hz (Babski et al. 2024, Heliyon, PMC11064449: OA-VPM1 / VPM2 fire tonic single spikes whose rate rises in locomotor
bouts, no value given; CSDn 1-2 Hz spontaneous, Zhang & Gaudry 2016), so `R` is bracketed at 5 / 10 / 20 Hz and the
model's own presynaptic rates are read off the off arm in section 3 (PAM08 3.0, PPL101 17, OA-VPM3 7.3, CSD 5.0 Hz in
the CPU smoke `out/monoamines/smoke/off_cpu_cells.npz`, one fly, 1.5 s).

| anchor | effect in the literature | mode it maps to | tone bracket | model load on the named targets (syn-eq / cell) | slow_gain that reaches the bracket at R = 10 Hz (5 / 20 Hz scale it x2 / x0.5) |
|---|---|---|---|---|---|
| octopamine -> visual motion pathway | flight doubles VS peak-to-peak responses (Maimon, Straw & Dickinson 2010, Nat Neurosci 13:393); OA neurons are necessary and sufficient and OA application reproduces it in quiescent flies (Suver, Mamiya & Dickinson 2012, Curr Biol 22:2294; qualitative only); CDM 2.5 uM on blowfly H2: spontaneous 7.9 -> 15.6 Hz, initial response gain +126 %, mean rate x1.49 (Longden & Krapp 2010, Front Syst Neurosci 4:153); walking / CDM 10 uM raise Mi1, Tm3, Mi9, T4 responses at high temporal frequency and OA-neuron photoactivation drives Mi4 (Strother et al. 2018, PNAS 115:E102 -- its qualitative claims are confirmed; its specific numbers were behind a 403 and are unverified in this session) | `gain`, x1.5-2.3 -- **the low edge 1.5 is Longden's x1.49 and 2.0 is Maimon's "doubles"; the 2.3 upper edge has NO number in any cited source and is HEADROOM, not a literature value** (the same convention as the proprioception thread's 250 Hz haltere ceiling; the row is `op report` and nothing is scored on it) | +3.5 .. +9.1 mV | HSN 0.33, HSE 2.07, HSS 6.05, VS 0.49 (14 of 34 cells with any), Mi4 0.43 (574 / 1,772), Mi1 0.26, Tm3 0.87 | HSS 1.05-2.7; HSE 3.1-8.0; Tm3 7.3-19; VS 13-33; Mi4 15-38; HSN 20-51; Mi1 25-64 |
| dopamine -> Kenyon cells | DAN activation alone does not change gamma-KC baseline voltage or evoked spiking (Cohn, Morantte & Ruta 2015, Cell 163:1742, Fig. S6B); the DAN effect is a KC>MBON synaptic-weight change -- pairing depresses MBON-gamma1pedc's odour response 118 -> 24 spikes (-80 +- 5.7 %), charge -90 +- 3.7 %, > 40 min (Hige et al. 2015, Neuron 88:985); DAN alone potentiates KC>MBON (Cohn 2015 Fig. 6); dopamine raises KC cAMP (Tomchik & Davis 2009, Neuron 64:510) | `additive` on KCs: ~0 mV; the real effect is a plasticity term the slow tone cannot express | 0 mV | KC 63.4 (KCg-m 75.8, KCab-s 44.5, KCa'b'-m 93.8, KCab-p 110.2), all +, 4,063 / 4,064 cells | 0 (any non-zero scale gives 0.35 / 0.7 / 1.4 mV per KC at slow_gain 0.02, 3.5 / 7 / 14 mV at 0.2, 17 / 35 / 70 mV at 1.0 for R = 5 / 10 / 20 Hz) |
| serotonin -> antennal lobe | endogenous 5-HT lowers PN odour responses (fluoxetine decreases, methysergide increases DA1 PN responses) by enhancing GABAergic presynaptic inhibition of ORN terminals; exogenous 100 uM raises PN responses; CSDn stimulation gives LNs a brief depolarisation then a delayed hyperpolarisation, no mV published (Zhang & Gaudry 2016, eLife 5:e16836; Dacks et al. 2009, J Neurogenet 23:366; Sizemore & Dacks 2016, Sci Rep 6:37119: receptor map only) | `additive`, negative on the PN response; presynaptic in the animal | -0.7 .. -3.5 mV (0.1-0.5 gap: the model's own scale, no measured mV -- labelled) | PN +4.48 / -2.49 (net +2.0, 5-HT7-led: the wrong sign for the endogenous effect), ORN 0.20, LN 0 (no rows) | PN 0.64-3.2 (with the sign wrong); LN unreachable |
| serotonin -> central complex | the 5-HT -> CX synapses (PFGs, PFR_a, FB4Y, ExR3: 65,000 raw) have no effect-size literature at the cell level; only the ExR3 -> ring rows carry a sign (section 1d) | -- | -- | ExR3 -554 (5-HT1A), ER3d_b +137 / -100 (mixed with OA), PFGs -, hDelta 0 | not anchorable |

**Reading of the bracket.** The two quantitative anchors pull one class scale in opposite directions by a factor of
~50-100: the octopamine gain on the visual pathway needs `slow_gain` >= 1 (HSS, the best-loaded HS cell, at 10 Hz; 3-60
on the typical target) because the OA synapse counts onto those cells are 0.3-6 per cell after the cap and the fan-in
scale, while the dopamine tone on KCs must be ~0 and any `slow_gain` puts 35 mV x (slow_gain) x (R / 10 Hz) on every
KC because the DAN -> KC counts are 45-110 per cell. **One per-class scalar cannot satisfy both**; the hand-set 0.02 is the
KC-side ceiling (0.7 mV per KC at 10 Hz, invisible on the HS cells: 0.07 mV) and the physiology's OA gain is 50x above it.
The arms therefore span the bracket: `add_low` 0.02 (the shipped value = the KC-anchored ceiling), `add_mid` 0.2 (the
geometric middle), `add_high` 1.0 (the LOW edge of the OA anchor), plus `gain_mid` 0.2 in the mode the OA anchor maps to.
The serotonin anchor cannot be reached at all: the endogenous effect is presynaptic on ORN terminals and the table gives
the PNs a net positive row and the LNs none.

## 3. The run (`out/monoamines/runs/`, batch `mono-fd76d2`, `out/mono_cluster.log`)

One submission (`scripts/probe_monoamines.py batch` -> `out/monoamines/cmds.txt`, 25 job lines; `cluster_run.py --name mono
--arm-block fam`), five blocks `fam_r0..r4` = one replicate each, every arm of a replicate on one box: r0 / r2 / r4 on
`r3-h200a`, r1 / r3 on `r3-h200b`; 25 completed, 0 failed (box queues polled after the restart: a 15 completed, b 10
completed, `out/monoamines/fetch.log`). Every run JSON: `device cuda`, `device_name NVIDIA H200`, torch 2.11.0+cu128, eager
Torch path (`cuda_kernels False`, `event_driven False`, `cuda_graphs False`) in every arm including `off`, B = 8 flies
(env seeds 1000 r .. 1000 r + 7), brain seed r, program `none`, fence on, fruit `all`, 30 s, window [2, 30] s; health
target 15,575 cells (DNs, ANs, CX, MB, AL, VNC motor, every monoaminergic cell, MN9) captured every 100 ms; the
benchmark sections `rest,taste,smell,walk,loom_escape` with seeds 0,1,2 after the rollout of every non-aborted run.
`scripts/probe_monoamines.py analyse` -> `out/monoamines/analysis.json` (tables `runs`, `behaviour`, `health_replicates_*`,
`health_compare_*`, `leave_zero_*`, `movers_*`, `bench_runs`, `bench_checks`; md5 of every input file in `files.inputs_md5`)
and `analysis.txt`. Runs are the replicates (n = 5 per arm); `common.compare` against `off`; a per-run value is the mean
over the run's 8 flies. `flyverse_commit.commit` reads `unknown` on the boxes; the code identity is
`provenance.source_fingerprint` (44 files; the desktop's CRLF tree hashes differently on identical source, the fingerprint
says so).

### 3a. The runaway record (`analysis.json` table `runs`; r0 trajectories from `r0_<arm>_cells.npz` `spikes_per_step`)

| arm | runs | status | rest 500 ms (spikes/step) | rollout spikes/step, mean +- SD over runs (range) | vs off | max single frame |
|---|---|---|---|---|---|---|
| off | 5 | ok | 0.00 in every row of every run | 52.14 +- 0.13 (51.99-52.34) | 1.00 | 305-344 |
| add_low 0.02 add | 5 | ok | 0.00 | 53.70 +- 0.25 (53.50-54.14) | 1.03 | 308-357 |
| gain_mid 0.2 gain | 5 | ok | 0.00 | 132.94 +- 0.59 (132.1-133.6) | 2.55 | 360-463 |
| add_mid 0.2 add | 5 | **aborted_rollout at frame 51 (t 0.51 s)** | 0.00 | 665.6 +- 3.2 over [0.255, 0.51] s (660.8-668.7) | 12.8 | 758-794 |
| add_high 1.0 add | 5 | **aborted_rollout at frame 51** | 0.00 | 1,277.0 +- 0.5 (1,276.4-1,277.4) | 24.5 | 1,334-1,351 |

The guard as specified ("abort if the rest bound is exceeded 10x") is implemented (`--abort-rest` 50 = 10 x the suite's
`rest.spikes_per_step < 5`) and structurally inert: the model has no spontaneous activity, so every arm rests at 0.00. The
guard that fired is the rollout one (`--abort-rollout` 500 spikes/step on a 0.5 s running mean, checked from frame 50 so the
vision-onset transient cannot trip it; 500 is 9.6x the off rollout's 52). Per-frame means of run 0 (B = 8): `off` 11, 93,
299, 219, 128, 89, 66, 58, 58, 62, 67, 72 over frames 1-12 (the onset transient) then 40 over frames 20-50 and 52 over the
30 s; `add_mid` is identical for two frames (11, 93) then 311, 244, 157, 119, 109, 130, 168, 222, 271, 329 -- the transient
turns over at ~70 ms and climbs to 604 (frames 20-30) and 680 (40-50); `add_high` 380, 598, 685, 727, 789, 876, 961, 981,
1,008, 1,052 by frame 12 and 1,274 from frame 20 on (the ceiling of the target cells). `gain_mid` 331, 274, 185, 138, 124,
135, 150, 150, 156, 167 then 131.5 (frames 20-30) and 131-134 in every 5 s bin of the 30 s: a new plateau, not a climb.
The CPU smoke of the predecessor (`out/monoamines/smoke/add_mid_abort.json`, B = 1, cpu) aborted at the same frame with
**516.1 spikes/step as the 0.5 s RUNNING MEAN printed at the abort** (`out/monoamines/smoke/add_mid_abort.txt`); the same
run's scored WINDOW mean in its run JSON is 656.8 (max frame 709.5), which is the quantity the H200 numbers beside it
(665.6) are. Both are in the artefacts; the sentence has to say which it quotes (skeptic pass). Either way the runaway
is device-independent.

Where the spikes are (health per type over the aborted arms' window [0.255, 0.51] s; `off` over [2, 30] s):

| type (cells) | off | add_low | gain_mid | add_mid | add_high |
|---|---|---|---|---|---|
| KCg-m (1,342) | 2.20 +- 0.01 | 2.58 +- 0.02 | 9.51 +- 0.07 | 223.2 +- 0.1 | 318.2 +- 0.1 |
| KCab-s (657) | 0.91 | 1.05 | 18.41 | 178.4 | 314.6 |
| KCab-p (129) | 0.16 | 0.22 | 0.72 | 255.8 | 318.2 |
| APL (2) | 18.42 | 21.14 | 148.7 | 318.1 | 320.7 |
| DPM (2) | 15.76 | 18.70 | 135.1 | 318.0 | 321.4 |
| PAM01 (44) | 2.99 | 3.49 | 51.3 | 276.4 | 318.7 |
| PAM08 (50) | 3.11 | 3.49 | 33.5 | 262.6 | 296.3 |
| PPL101 (2) | 17.16 | 19.07 | 80.4 | 316.9 | 321.0 |
| PPL103 (2) | 13.30 | 15.02 | 0.13 | 315.1 | 320.2 |
| MBON01 (2) | 15.59 | 17.24 | 0.00 | 314.4 | 28.3 |
| MBON03 (2) | 6.71 | 7.17 | 0.00 | 231.4 | 5.9 |
| MBON11 (2) | 16.65 | 21.93 | 275.6 | 319.2 | 321.1 |
| types > 300 Hz / > 100 Hz (of 2,041) | 0 / 2 | 0 / 2 | 4 / 61 | 24 / 114 | 76 / 321 |

The loop, named on the shipped weights (`flyverse.interp.decompose`, static, mV per post cell per presynaptic volley;
`receptor:fuzzy` / `alias` / `class` tiers): KCg-m -> PAM08 163.3 mV (19,326 entries, 29,697 raw syn, 58.8 % of PAM08's
input), KCg-m -> PAM01 168.3 mV (60.9 %), KCg-d -> PAM08 / PAM01 21.6 / 22.1 mV, KCg-m -> PPL101 560 mV (35.4 %; KCab-s /
-m / -c another 119 / 111 / 101 mV); the return edge is the largest monoamine pair in the graph (PAM08 -> KCg-m 26,725 raw
syn; section 1d) at 75.8 dopamine syn-eq per KCg-m cell after the cap and the fan-in scale (`target_load.csv`), slow +1.
Tone per Hz of DAN rate = slow_gain x 0.275 x 75.8 x 0.2 s = 4.17 x slow_gain mV/Hz: 0.083 at 0.02 (the DANs at 3-4 Hz
in the off room give 0.3 mV), 0.83 at 0.2 (the DAN POPULATION at 8.4 Hz = the 7 mV gap; 75.8 is the summed load over all
dopaminergic presynaptic cells, not one DAN's), 4.2 at 1.0. KCg-m's own recurrence adds
8.5 mV per volley (50 % of its input, same-type-damped) and APL's -15.5 mV per volley is the only brake.

### 3b. Baseline activity per system (health per type, cell-weighted means; `off` window [2, 30] s, `gain_mid` / `add_low` the same, the aborted arms [0.255, 0.51] s)

| system (types / cells in the target) | off | add_low | gain_mid | add_mid | add_high |
|---|---|---|---|---|---|
| descending neurons (480 / 1,310) | 1.84 Hz, 39 % cells silent, 82 types at 0 | 1.89, 39 %, 81 | 4.37, 41 %, 55 | 5.59, 63 %, 149 | 20.4, 54 %, 160 |
| ascending neurons (566 / 1,840) | 0.49, 74 %, 296 | 0.50, 73 %, 291 | 0.80, 73 %, 272 | 1.09, 86 %, 368 | 4.70, 81 %, 356 |
| CX (class; 292 / 2,950) | 0.06, 92 %, 204 | 0.06, 91 %, 201 | 0.57, 84 %, 154 | 3.25, 89 %, 177 | 25.6, 71 %, 150 |
| VNC motor (142 / 699) | 4.09, 31 %, 11 | 4.18, 31 %, 10 | 6.67, 32 %, 9 | 10.2, 49 %, 19 | 33.9, 45 %, 19 |
| Kenyon cells (15 / 4,064) | 1.49, 46 %, 0 | 1.73, 42 %, 0 | 17.9, 15 %, 0 | 198.6, 0 %, 0 | 312.6, 0 %, 0 |
| MBONs (37 / 97) | 6.55, 11 %, 1 | 6.88, 10 %, 1 | 12.7, 70 %, 4 | 212.3, 2 %, 0 | 90.6, 6 %, 2 |
| DANs PAM / PPL / PPM (31 / 352) | 2.62, 38 %, 2 | 2.90, 35 %, 2 | 23.9, 11 %, 2 | 225.9, 3 %, 3 | 261.7, 4 %, 3 |
| OA cells (15 / 35) | 12.3, 2 %, 0 | 14.0, 2 %, 0 | 101.8, 0 %, 0 | 126.4, 2 %, 0 | 297.0, 3 %, 0 |
| uniglomerular PNs (68 / 280) | 7.71, 28 %, 3 | 8.04, 26 %, 3 | 18.9, 25 %, 2 | 19.1, 26 %, 4 | 108.8, 10 %, 1 |
| AL LNs (64 / 317) | 34.0, 19 %, 4 | 35.3, 19 %, 4 | 78.0, 14 %, 4 | 78.3, 18 %, 4 | 153.9, 3 %, 1 |
| whole target, types at exactly 0 Hz (of 2,041) | 680 | 671 | 563 | 796 | 746 |

Module means (`analysis.json` `health_replicates_module`, `rate_mean`): antennal lobe 8.04 / 8.23 / 16.4 / 14.3 / 45.5 Hz;
central 0.23 / 0.24 / 1.53 / 4.66 / 28.1; descending 1.84 / 1.89 / 4.37 / 5.59 / 20.4; mushroom body 1.70 / 1.95 / 18.5 /
201.6 / 304.7; visual projection 6.97 / 7.45 / 60.2 / 67.3 / 218.6; vnc 1.54 / 1.57 / 2.53 / 3.85 / 13.2; gustatory 0 / 0 /
1.25 / 0 / 2.24 (the only module that leaves 0 Hz as a module: under `gain_mid` and `add_high`); optic and mechanosensory
0 everywhere (rate units / never driven).

The presynaptic rates the bracket needed (section 2 assumed 5 / 10 / 20 Hz), recomputed by the skeptic pass and
restated here (an earlier version said "63 types, median 4.0 Hz, quartiles 0.7-10.7, 5 silent" and then named eight
types, and no shipped file reproduces it): in the off room the monoaminergic types of the health target number **210**
(median 0.58 Hz, quartiles 0.0-6.76, 64 silent), of which 189 are presynaptic in `W_slow` (median 0.47, 61 silent); the
set the bracket's R is about -- `presyn_types.csv` intersected with the target -- is **45 types, median 4.29 Hz,
quartiles 0.69-10.49, 5 silent: SAxx02, Mi19, PFR_a, PFGs, SNpp23**. The per-type rates below are exact as quoted:
PAM08 3.1, PAM01 3.0, PPL101 17.2, PPL103 13.3, DPM 15.8, OA-VPM3 6.9, OA-VPM4 8.7, OA-VUMa1 9.6, OA-VUMa2 68.7, OA-AL2i1
7.8, CSD 7.3, ExR3 4.3, aMe17a 21.1, 5-HTPMPV03 1.9 Hz -- the 10 Hz assumption is fair for the anchors' presynaptic cells
but the spread is 20x across cells, so a per-class scale meets different tones on different targets.

### 3c. Which types leave 0 Hz (silent in every off run and row, active in the arm; `leave_zero_type`)

`add_low`: 29 candidates (15 vnc, 7 central, 6 descending, 1 antennal lobe), none below `silent_frac` 0.70 -- single rows
of single runs; all `null` / `undetermined`. `gain_mid`: 141 types (59 central, 34 vnc / ascending, 29 descending, 14
antennal lobe, 3 mushroom body, 2 gustatory), 17 of them active in most rows: FB2I_b, GNG002, M_vPNml78, M_lvPNm30 (0.00
silent), PAM14 0.01, ER3d_c 0.11, PAM03 0.16, ER3d_a 0.25, PFR_a 0.31, hDeltaB 0.31, M_vPNml87 0.33, PAM09 0.36, FB5T
0.36, GNG540 0.38, FB4J 0.39, FB4D_a / _c 0.40 / 0.46 -- the fan-shaped body, the ring's ER3d cells, the silent PAMs and
the multiglomerular PNs. The CX types that move most under `gain_mid` (Hz, off -> gain_mid): FB4Y 0.76 -> 46.6 (a 5-HT
cell), FB2H_b 8.3 -> 34.0, ExR3 4.3 -> 29.3, EL 0.48 -> 19.2 (OA), FB2H_a 4.9 -> 19.1, ExR6 0.09 -> 12.1, ExR4 0.10 -> 9.3,
ER6 0.09 -> 8.7, FB1G 1.2 -> 9.1, GLNO 0.05 -> 7.4, Delta7 0.03 -> 6.9, ExR5 0.03 -> 5.7, EPG 0.03 -> 3.2, PEN_a 0.02 ->
2.9, PEN_b 0.01 -> 1.65; PFL3 stays at 0.00 and ER4d at 0.00 (its EL -> ER4d 4,178 syn are Octalpha2R -1). The aborted
arms' lists (82 / 158 types) are 0.25 s windows of a runaway and are not read.

### 3d. The plain fly (per-run mean over 8 flies; 5 runs per arm; `behaviour`)

| key | off | add_low | verdict | gain_mid | verdict |
|---|---|---|---|---|---|
| yaw-rate SD, walking frames (deg/s) | 2.60 +- 0.29 | 2.73 +- 0.20 | null (z 0.45) | 4.47 +- 0.43 | result (z 6.4, p 0.008) |
| mean abs yaw rate (deg/s) | 1.24 +- 0.09 | 1.29 +- 0.06 | null | 2.37 +- 0.18 | result (z 13) |
| straightness net / path | 0.9981 +- 0.0004 | 0.9971 +- 0.0005 | null (z -2.3, p 0.03) | 0.974 +- 0.008 | result (z -57) |
| walking speed (mm/s) | 8.83 +- 0.04 | 8.81 +- 0.03 | null | 9.10 +- 0.40 | null (p 0.15) |
| DNa02 R-L (Hz, readout) | 0.048 +- 0.016 | 0.049 +- 0.018 | null | -0.080 +- 0.031 | result (z -8; at 0.1 Hz) |
| DNa02 abs R-L (Hz) | 0.091 +- 0.018 | 0.100 +- 0.018 | null | 0.225 +- 0.037 | result |
| DNa02 mean rate (Hz) | 0.033-0.054 | 0.042-0.062 | -- | 0.095-0.139 | -- |
| leg MN L-R (Hz) | 0.157 +- 0.005 | 0.156 +- 0.006 | null | 0.332 +- 0.006 | result (z 37) |
| take-offs per fly (28 s) | 0.025 +- 0.056 | 0.075 +- 0.17 | null | 1.55 +- 0.46 | result (z 27) |
| of which escape / voluntary | 0.025 / 0 | 0.05 / 0.025 | null / null | 0.05 / 1.50 | null / undetermined (zero-SD null) |
| airborne fraction | 0.0002 | 0.0006 | null | 0.0077 +- 0.0019 | result |
| spikes/step, window | 52.14 +- 0.13 | 53.70 +- 0.25 | result (+3.0 %) | 132.9 +- 0.6 | result (x2.55) |

The aborted arms' behaviour rows are 0.25 s windows and are not read (add_high: every fly airborne by the abort, 1.00
escape take-off each; add_mid: DNa02 0.00). In `gain_mid` the fly still walks straight between hops (straightness 0.974);
the yaw-rate SD and the leg L-R bias rise together with the voluntary take-offs (1.5 per fly), i.e. the wandering is
hopping, not steering: DNa02 stays at 0.1 Hz.

### 3e. The suite (`bench_runs` / `bench_checks`; 5 runs x 3 seeds per arm; every run 12 PASS / 0 FAIL / 0 KNOWN GAP; NVIDIA H200, eager torch)

| check (bound) | off | add_low | gain_mid |
|---|---|---|---|
| rest.spikes_per_step (< 5) | 0.00 | 0.00 | 0.00 |
| taste.MN9_hz (> 2) | 10.93 (5 / 5 identical) | **2.48** (5 / 5) | 5.04 (5 / 5) |
| smell.PN_hz | 7.86 | 6.93 | 25.02 |
| smell.KC_active | 816 | 790 | 1,550 |
| walk.GF_max_hz | 4.63 | 9.52 | 5.66 +- 1.57 |
| walk.power_max_hz (de-scored, round 2) | 48.5 | 59.1 | 65.5 +- 4.8 |
| walk.power_sustained_hz | 20.1 | 24.5 | 39.2 +- 5.7 |
| loom.GF_peak_hz | 46.5 +- 1.6 | 51.2 +- 5.8 (null, but z 2.93 at p 0.008 -- marginal; see below) | 49.7 +- 2.8 (null) |
| loom.escape_cm | 3.5 | 3.5 | 3.5 |
| rotate.DNp20_flip_hz (< -2) | -35.5 +- 4.4 | -40.5 +- 6.9 (null) | -31.7 +- 5.7 (null) |
| loom_escape.GF_peak_hz (>= 33) | 48.4 +- 1.8 | 50.2 +- 2.5 (null) | **63.9 +- 3.3 (result, z 8.6)** |
| loom_escape.escapes (>= 1) | 3 / 3 | 3 / 3 | 3 / 3 |
| hops before the loom (3 seeds) | 0 | 0 | 1 in 3 of 5 runs |

**This table is a CUDA table.** Every value in it is from the H200 batch, and the seed-locked sections are seed-locked
*on CUDA*: the skeptic's B200 rerun reproduces them to the last digit (H200 and B200 identical on `taste.MN9_hz`,
`smell.PN_hz`, `smell.KC_active`, `walk.GF_max_hz`, `walk.power_sustained_hz`), so the device story of this thread is
CPU vs CUDA. On the CPU reference path the `add_low` arm does NOT return 12 / 0 / 0: `taste.MN9_hz` measures
1.9669914245605469 against "> 2" and the check is a **FAIL** (`off` on the same CPU 5.0909, PASS). The one `null` worth
a number rather than a word: `loom.GF_peak_hz` under `add_low` is 51.241 +- 5.809 against 46.493 +- 1.621, i.e.
**z 2.93 at p 0.008** -- `null` only because |z| < 3, and it does not reproduce on the B200 (50.42 +- 1.89 vs 45.73 +-
1.92, z 2.44), which supports the null; but the section did move and the z belongs in the record (skeptic pass).

The taste / smell / walk sections are seed-locked (identical over the 5 runs on each side), so `common.compare` returns
`undetermined` with a structural p for every one of them: they are deterministic shifts, reported as such, not results.
Under `add_low` the fast weights are the off arm's (48,295 sign-changed entries in both, `bench_runs` `receptor`); the
shifts are the 0.02 tone alone. The benchmark reports the classical class's 4.57 M entries beside the monoamine class's
215,579 in its summary line; its gain is 0.0 and `Brain` builds no matrix for it (`_slow_spec` drops zero-gain classes; the
probe's `slow.entries` shows the monoamine class only).

### 3f. The taste section replayed on the CPU (`scripts/probe_monoamines.py taste`; `out/monoamines/taste/<arm>.json` / `.txt`)

`benchmark.sec_taste` exactly (144 sweet labellar-bristle / taste-peg GRNs at 100 Hz, 600 ms, Brain seed 0, `rate_np()` at
600 ms) on the desktop CPU under each arm, with the realised `g_slow` read off the Brain per type and MN9's input
attributed per presynaptic type (A x rate, `common.effective_weights`). One deterministic replay per arm; a different
device from the H200 benchmark (a replication across devices, not a bit check).

| arm | MN9 Hz (CPU) | MN9 Hz (H200 suite) | MN9 input E / I / net (mV/s per cell) | tone range over types (mV) | types with non-zero tone |
|---|---|---|---|---|---|
| off | 5.09 | 10.93 | +1,786 / -1,677 / +110 | 0 | 0 |
| add_low 0.02 | 1.97 | 2.48 | +1,475 / -1,473 / +2 | -0.65 (GNG572) .. +0.87 (DNge150) | 264 |
| gain_mid 0.2 | 3.48 | 5.04 | +1,441 / -1,499 / -59 | -5.0 (PRW006; GNG572 -4.6) .. +3.7 (DNge151) | 271 |
| add_mid 0.2 | 3.94 | (not run: aborted rollout) | +1,805 / -1,596 / +209 | -6.1 .. +7.6 | 267 |
| add_high 1.0 | 0.02 | (not run) | +500 / -2,086 / -1,586 | -12,219 .. +4,861 (KCg-s1); 22.7 % of all cells > 1 Hz | 577 |

MN9 is the residual of two ~1.5 V/s inputs (DNge051 -1,083 mV/s, DNge080 +477, GNG108 +311, GNG120 +206, GNG130 -199 in
`off`), which is why its rate differs 2x between devices at the same seed and why sub-mV tones move it. The section's
whole-brain activity is 16.9 / 16.7 / 17.0 / 17.5 spikes/step over the second 300 ms under `off` / `add_low` / `gain_mid` /
`add_mid` (the GRN-driven SEZ alone; the MB loop needs the visual drive to ignite at 0.2) and 1,690 under `add_high`. Under `add_low`
the monoaminergic cells that fire in the section are the serotonergic SEZ cells the GRNs drive -- GNG550 36 Hz, GNG190
23, GNG056 19, GNG002 13, DNg30 11, GNG572 10, GNG540 2 -- and the tones they lay on their 5-HT1A-led targets are GNG572
-0.65 mV (**79 5-HT syn-eq per cell**: `target_load.csv` `serotonin_neg` 79.0 for the type; the 136 an earlier version
quoted is the MAX over GNG572's three cells of their |slow load| -- 136.0 / 124.0 / 108.0, mean 122.7 -- and that 136
cell is -87 serotonin plus +42 octopamine, so it is neither a per-cell type value nor serotonin alone; skeptic pass),
GNG056 -0.35, DNg30 -0.37, GNG540 -0.33, GNG137 -0.30, GNG101 -0.24 (the OA cells:
DNge150 +0.87, OA-VPM4 +0.69, DNge151 +0.46, from OA-VUMa2 at 1.7 Hz); the SEZ interneurons that move are GNG099 46 -> 13
Hz, MNx01 33 -> 8.7, GNG334 35 -> 11, GNG479 42 -> 18, GNG019 30 -> 8.5, MN10 26 -> 6.2, MN11V 23 -> 5.9, and MN9's inputs
shift by DNge051 +164 (less inhibition), GNG120 -99, DNge059 -83, DNge080 -52, GNG108 -49 mV/s: the excitation and the
inhibition drop together and the residual goes from +110 to +2 mV/s. Under `gain_mid` the same GNG cells carry -1 to -5
mV (GNG572 -4.6, GNG056 -2.4, GNG540 -2.0, GNG137 -1.7, GNG101 -1.7, DNg30 -2.8) and the SEZ goes the other way (GNG479
42 -> 79, GNG099 46 -> 74, MN11D 30 -> 56: disinhibition through the mode's factor on net-inhibited cells). Under
`add_high` the section runs away without any visual input: 577 types with a tone, 370 of them above +3.5 mV, the MB and
the OA / 5-HT cells at 330 Hz (OA-AL2i3, 5-HTPMPV01, OA-VUMa1, DNge151, MBON11, KCg-s1 / -s2), MN9 silenced.

## 4. Classification, and what a default adoption would require

| arm | baseline activity | plain fly | suite | runaway | verdict |
|---|---|---|---|---|---|
| add_low 0.02 additive (the shipped value) | +3.0 % spikes (result, tiny); MB +0.25 Hz; nothing leaves 0 Hz | every key null (5 runs) | 12 / 0 / 0 x 5 **on CUDA**; taste MN9 10.9 -> 2.5 (seed-locked, knife-edge readout, section 3f); **on the CPU path `taste.MN9_hz` 1.967 < 2 = FAIL** | no | **null** on behaviour and baseline; **a scored-check FAIL on the CPU reference path**, a regression watch on CUDA |
| add_mid 0.2 additive | MB at the ceiling in 0.25 s | not measurable | not run | **5 / 5 at t 0.51 s, 12.8x** | **result (structural): the KC <-> DAN loop** |
| add_high 1.0 additive | as above, 24.5x; the taste section alone runs away | -- | not run | **5 / 5** | result (structural) |
| gain_mid 0.2 gain | x2.55 spikes, flat; MBONs zeroed; OA / 5-HT / DAN cells at 100-330 Hz; AL x2; ring leaves silence; DNs x2.4 | straightness 0.974 (result), yaw SD x1.7 (result), 1.5 voluntary take-offs per fly (result), speed null, DNa02 still ~0.1 Hz | 12 / 0 / 0 x 5; loom_escape GF peak +15 Hz (result); PN 25 Hz, KC_active 1,550 (seed-locked) | no | **result: a large baseline change that is not a behavioural gain** |

None of it is adoptable, and the reasons are structural rather than a matter of the scale:

1. **One class scalar cannot carry the three transmitters.** Section 2's anchors are 50-100x apart, and the run shows both
   ends: at 1.0 (the octopamine anchor's low edge on the best-loaded HS cell) the dopamine rows detonate the mushroom body
   in 0.5 s; at 0.02 (the KC-anchored ceiling: Cohn 2015, `lit.KC.dan_tone_mv` 0) the octopamine rows are invisible (HSS
   tone 0.07 mV at 10 Hz) and the plain fly does not move. An adoption needs the slow class split by transmitter
   (dopamine / octopamine / serotonin instead of `SLOW_CLASSES`' two-way classical / monoamine split), i.e. a
   `connectome.receptor_signs` slow_class assignment per presynaptic transmitter (thread B's table side) and
   `LIFParams.slow_gain_by_class` / `slow_tau_by_class` accepting the three keys (thread C's field), each class bracketed
   on its own anchor -- dopamine ~0 on the KCs (or expressed as what it is, item 2), octopamine as a gain, serotonin
   additive negative. Nothing in the shipped table has to change for that; the gene identity is already lost at
   `receptor_signs` (only sign / gain class / class survive), so a finer split (per lead receptor: Dop1R2 vs Dop2R,
   Octbeta vs Octalpha2R, 5-HT7 vs 5-HT1A) would need a table column carried through.
2. **The dopamine effect the data describe is not a tone.** DAN -> KC is 59 % of the signed dopamine synapses and the
   literature says its postsynaptic tone is ~0 (Cohn 2015) while the effect is a KC>MBON weight change (-80 % after
   pairing, Hige 2015; `lit.MBON11.kc_mbon_depression`). The slow term cannot express it in any mode: additive detonates
   the loop, gain zeroes the MBONs through their Dop2R rows. A plasticity module on the KC>MBON edges gated by the DAN
   rate is the mechanism the data imply -- a swappable module (docs/INTERP.md's module protocol), not a default change.
3. **`gain` mode needs separate excitatory / inhibitory accumulators** (`brain.py`'s own note at `slow_mode`): the
   MBON zeroing (-480 to -876 syn-eq per cell -> factor clipped to 0 for any DAN rate above 0.7 Hz), the SEZ
   disinhibition under `gain_mid` (GNG479 42 -> 79 Hz) and the self-amplification of the monoaminergic cells (OA-VUMa2
   329 Hz) are mode artefacts, not physiology. Until then no gain-mode result can be read as octopamine's gain.
4. **Coverage.** 72.8 % of the monoamine synapses are silenced in every model; the VNC, the ascending neurons and the
   OA-VUM / VPM system's targets have no receptor rows at all (0 signed synapses onto vnc_intrinsic / vnc_motor / ascending;
   OA-VUMa1 994 of 40,467 raw syn signed). The "arousal / locomotor-state system" the task names cannot be tested through
   this term until VNC types are profiled (no VNC expression source is in the builder's five; thread B's table side).
   The DNs that do carry a tone are the 13 types of `target_load.csv` (DNge150 +291, DNp32 -127, DNge151 +113 ...), not
   DNa02 / DNp09 / MDN / DNp01, which have none -- so the turning deficit of rounds 1-2 is out of this term's reach by
   construction, whatever the scale.
5. **The 0.02 default itself is hand-set.** It survives as a value because it is invisible on behaviour (this round's
   `add_low`), but the KC anchor says 0 and the taste section already tips at it. Keeping it at 0.02 under `full` would
   require, at least: **(a) a suite pass on the CPU path, which it does not currently have** -- at the shipped gain under
   `full` the CPU measures `taste.MN9_hz` 1.967 against "> 2" = FAIL (skeptic pass), and the project rule uses the CPU
   path as the bit-identity reference, so a candidate default has to be reported on it as well as on the GPU; **(b) the
   taste check itself re-anchored** on a readout that is not a residual of +-1.7 V/s (MN9's off value is 5.09 on the CPU
   against 10.93 on CUDA at one seed -- a 2x gap on a scored check is a defect of the check); **(c) nothing further on the
   GPU replication side -- it is done**: the skeptic's house-cluster B200 rerun (4 fresh seeds per arm, one submission,
   `out/monoamines_rerun/`, `monoskep-587c1b`) reproduces the H200 on every seed-locked suite section to the last digit
   and every behaviour key `null` at 4 v 4, so the open device question is CPU vs CUDA, not GPU vs GPU; (d) the CPU
   bit-identity test of the shipped path (`test_off_arm_is_the_shipped_default_and_no_default_moved` pins it;
   `receptor_model` stays `sign`, so no shipped weight moves); and (e) the ledger rows of section 2 scored rather
   than reported, which needs assays the model does not have (a flight / CDM octopamine assay on HS / VS with the
   opt-in `OpticLobe(slow=)` term; a DAN-pairing assay).
6. **What the next round could legitimately run** (every item opt-in, every arm labelled, 5 runs per arm, one submission
   per family on the house cluster): (a) the three-way class split with dopamine 0, octopamine `gain` at 1-3 on its own
   rows, serotonin additive at 0.02-0.2 -- the OA visual-gain anchor on HS / VS / Mi4 tested with the optic term, without
   the MB loop; (b) the same with a per-sign accumulator in gain mode; (c) a KC>MBON plasticity module driven by the DAN
   rate against `lit.MBON11.kc_mbon_depression`; (d) the compass rows (`compass.EPG.bump_*`, `circ_corr_heading`) under
   (a), since the ring leaves silence under octopamine (EL -> EPG +1, Octbeta1R) -- whether that activity is heading-locked
   is the question `compass_room.md` left open.

Scratch tables and their md5 (generators in `scripts/`): the coverage / bracket tables in `out/monoamines/md5.txt`
(`scripts/build_monoamine_tables.py`); the run-side files in `out/monoamines/md5_runs.txt` (`analysis.json` / `.txt` from
`scripts/probe_monoamines.py analyse`, `taste/<arm>.json` / `.txt` from `scripts/probe_monoamines.py taste`). Of the 155
raw job files under `out/monoamines/runs/`, **40 are listed with their md5 in `analysis.json` `files.inputs_md5` -- the
25 run JSONs and the 15 bench JSONs** (an earlier version of this sentence said all 155; the `_health_type.json`,
`_health_module.json`, `_cells.npz` and `.txt` files, 115 of the 155, carry no md5 anywhere). All 40 listed md5s verify
against the local files, as do `md5.txt` and `md5_runs.txt` (skeptic pass).

**Three things this audit does not claim to have verified**, recorded so nothing is scored on them by accident:
(i) Cohn, Morantte & Ruta 2015 Fig. S6B (the "no change of gamma-KC baseline voltage or evoked spiking" that
`lit.KC.dan_tone_mv = 0` rests on) and Strother et al. 2018's specific numbers (air-puff responses 10-30 % of the
maximum grating dF/F, CDM 10 uM) are behind 403s and are UNVERIFIED in this session -- neither is refuted, and Cohn's
companion claim (the DAN effect is a KC>MBON weight change) is independently verified through Hige 2015.
(ii) "off = the shipped fly" rests on the other threads' opt-in tests, not on a measurement against `origin/main`: the
batch shipped the working tree (`body.py`, `senses.py`, `motor.py`, `batch_body.py`, `batch_sim.py`, `brain.py`), their
suites do pass here (31 tests) and `brain.py`'s diff is default-`None`, but a same-seed off-arm A/B against
`origin/main` was not run.
(iii) `out/monoamines_rerun/` and `out/monoskep_cluster.log` are the SKEPTIC's artefacts, not this thread's: the B200
replication `monoskep-587c1b` (4 fresh seeds per arm, one submission, one family; an accidental double-submission
`monoskep-1c3938` was cancelled and its runs excluded). No file of this thread was edited by that pass.
