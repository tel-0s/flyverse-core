# The optic layer's hand-crafted measures: what each stands in for, what breaks without it, and where the object is lost

Scripts: `scripts/audit_optic.py` (the configurations = OpticParams overrides only, `flyverse/optic.py` untouched; `--check`
prints the structure of every entry on the CPU; `--one` runs one configuration on the GPU; `--batch --submit` is the
cluster batch; `--report` builds `out/optic_audit/report.md` / `report.json`) and `scripts/probe_figure_stages.py` (the
per-stage figure map, static apple + moving ball, with a none-vs-none null). Console log of the batch:
`out/optic_audit_cluster.log`; per job `out/optic_audit/<config>.txt`; results under `out/optic_audit/<config>/`
(`<config>.json` = the benchmark sections, `stages_s<k>.json`, `obj_ball_s<k>.json`, `obj_null_s<k>.json`); the CPU
structure in `out/optic_audit/check.json` and `out/optic_audit/connectivity_structure.json`. Smoke batch (2 jobs, the
probe end to end and the override path): `out/optic_smoke_cluster.log`, `out/optic_audit_smoke/`.

Read with docs/audits/anti_runaway.md (the LIF's measures, the same question), docs/audits/object_sweep.md (the
null-referenced object assay), docs/NOTES.md sessions 2, 3, 8, 9.

Skeptic (dynamics round 1, 2026-09-12): verdict **mostly sound**. Its refutations and corrections are integrated in
place below; its independent replication is `out/optic_verify_cluster.log` (cluster run `optic-verify-acf696`, 3 jobs,
0 failed, 14.5 min, NVIDIA B200 / torch 2.11.0+cu128) with results under `out/optic_verify/`. The verdict verbatim is
in docs/audits/receptor_verification.md, "Dynamics round 1 (2026-09-12) -- skeptic verdicts".

## 1. What the rate optic lobe adds on top of the connectome, and what each measure stands in for

The rate model (`flyverse/optic.py` module docstring): every `ol_intrinsic` neuron (89,390 cells) is a rate unit
`tau dv/dt = -v + g_rr sum_j W_ij (r_j - b) + g_in sum_p W_ip a_p + g_fb sum_s W_is s_s`, `r = clip(v + b, 0, 1)`,
weights `sign x |synapses|` normalised per target; the spiking targets get `drive = gain_out sum_i W_si (r_i - b)`,
clipped. The hand-set layer on top of the connectome is the following. "Introduced" cites the NOTES session and what
the value was set against; "data" says whether the receptor / transmitter tables in `flyverse/data/` (Nern 2025
type map, `receptors_by_type.csv`) speak to it. Structural numbers: `out/optic_audit/check.json` (CPU, the shipped
cache, `sum|W|` 121,460,584) and `out/optic_audit/connectivity_structure.json`.

| # | measure (`OpticParams` / module constant) | value | stands in for | introduced | what the data say |
|---|---|---|---|---|---|
| 1 | `DEFAULT_PAIR_GAIN[0]`: Mi4 / Mi9 / CT1 / C3 -> T4a-d x5 | 75,746 rate->rate edges, 370,849 \|W\| = 23.3 % of T4's L2-normalised rate input (Mi9 22.9 %, CT1 14.1 %, Mi4 13.6 %, C3 6.9 % of T4a's with the gain in force); all inhibitory (GABA 44,654 edges, glutamate 31,092) | the delayed inhibitory arms of the T4 motion detector (Mi4 / CT1 GABA, Mi9 glutamate trailing / leading the Mi1 / Tm3 centre): "null-direction suppression must be able to clip to zero" | session 3, against `probe_motion` (DSI 0.03-0.06 -> 0.16-0.26); together with #7 and #8 | sign: confirmed on every edge (T4a/b: Rdl mid, GluCl-alpha high; Nern / FlyWire / validated NT gaba / glutamate / gaba / gaba); strength: nothing. Kinetics: T4a/b express GABA-B-R1 `high` (a slow GABA arm the fast-sign model does not have) and no mGluR (the Mi9 arm is fast GluCl only) |
| 2 | `DEFAULT_PAIR_GAIN[1]`: Tm4 / Tm9 / CT1 / TmY15 -> T5a-d x5 | 101,619 edges, 586,543 \|W\| = 38.9 % of T5's rate input; **78,877 of the edges are cholinergic (Tm4 17.3 %, Tm9 28.5 % of T5a's input, excitatory) and 22,742 GABA (CT1 20.6 %, TmY15 6.3 %)** | the same "delayed inhibition" for T5 -- but as written the entry multiplies T5's main excitation (Tm9, the OFF centre delay line) by 5 as well | session 3, same assay | Tm9 and Tm4 are acetylcholine in all three sources (Nern validated TAPIN), nAChR-alpha5 `mid` on T5: the entry is not an inhibition gain; it is a x5 on the **cholinergic 45.8 %** of T5a's input (Tm9 28.5 + Tm4 17.3; the per-type shares live in `connectivity_structure.json`, not `check.json`) and, counting both signs, on 72.7 % of T5a's input and 38.9 % across T5a-d (`check.json` `share_of_post_l2_input_rr` 0.3887). T5a/b GABA-B-R1 `high` as for T4 |
| 3 | `DEFAULT_PAIR_GAIN[2]`: LPi34 / LPi43 -> LPLC2 x4 | 1,231 rate->spiking edges, 7,543 \|W\| (glutamate, all -1) | lobula-plate inhibitory interneurons making LPLC2 expansion-selective: under the uniform synapse LPi are a tenth of LPLC2's T4/T5 excitation and the fly's own turning drove the giant fibre | session 9 (x1 / x4 / x10 sweep on the walking GF and the loom peak); round-4 scan x1 / x2 / x3 / x4 under the damped gains (anti_runaway.md) | sign confirmed (GluCl-alpha `high` on LPLC2, plus mGluR `mid` = a slow negative arm); strength: nothing (100 % of LPLC2's input matched, 0 sign changes) |
| 4 | `DEFAULT_PAIR_GAIN[3]`: T4a-d / T5a-d -> everything x2 | 712,375 rate->rate edges / 2,808,140 \|W\| (4.7 % of the targets' rate input; LPi34 gets 89 % of its input from T4c / T5c through it) + 244,590 rate->spiking edges / 1,260,461 \|W\| onto LPLC / LC / LPT / HS / VS | "rectified, strongly inhibited DS units respond weakly to natural scenes; restores drive to LPi / HS / VS / LPLC and the DNs" | session 3 (x3 raised the loom margin but cost walking-GF margin; kept at x2, session 4) | sign confirmed (ACh, nAChR on every target checked); strength: nothing |
| 5 | `DEFAULT_PAIR_GAIN[4]`: * -> LC4 / LPLC2 x1 | 128,208 rate->spiking edges | a placeholder: "the loom detectors keep x1 optic-lobe drive; the margin comes from LC4/LPLC2 -> GF x3 in brain.py" | session 6 (`probe_loom --lc-gain` sweep) | a no-op (factor 1; `apply_pair_gain` multiplies, so the T4/T5 x2 onto LC4 / LPLC2 stays x2). Listed for completeness; not ablated |
| 6 | `DEFAULT_TAU_BY_TYPE` | Mi4 / Mi9 / CT1 / Tm9 150 ms; L3 40; L1 / L2 6; Mi1 / Tm3 / Tm1 / Tm2 / Tm4 8; T4 / T5 10; everything else `tau_ms` 10 | the temporal asymmetry of the T4 / T5 inputs (slow Mi4 / Mi9 / CT1 / Tm9 vs fast Mi1 / Tm3, Tm1 / Tm2 / Tm4) -- "flyvis learns the same" | session 2 (alone: DSI 0.03-0.06), session 3 (with #1, #7) | no time constants in the data. The one thing the receptor table offers is the slow-receptor column: GABA-B `high` on T4 / T5 and mAChR-B `high` on Mi4 (its ACh input), i.e. the animal has metabotropic slow arms at the places the 150 ms stands in for; Tm9 -> T5 is nAChR (fast) in the data |
| 7 | `DEFAULT_BASELINE_BY_TYPE` | T4a-d, T5a-d operating point 0 (ReLU); every other unit 0.5 | rectification: DS needs null-direction suppression to clip at zero | session 3 (with #1, #6) | nothing (an operating point is not a receptor) |
| 8 | `norm` = `l2` | optic-lobe weights `W / in_syn_l2` (coherent fan-in of N equal inputs amplified ~sqrt N) instead of `l1` fractions | dendritic summation of many small inputs | session 2 (the first model that propagated beyond the lamina) | nothing |
| 9 | `gain_in` = 3 | photoreceptor contrast -> lamina | "lamina cells saturate at ~20 % contrast" (Laughlin) | session 2 | the L1 / L2 histamine sign (ort / HisCl `high`) is confirmed; the gain is not in the data |
| 10 | `gain_fb` = 0.5 | spiking -> rate units, rate / 100 Hz | visual centrifugal feedback | session 2 | signs only |
| 11 | `adapt_gain` 1, `adapt_tau_ms` 400 | slow relaxation of every rate unit to its operating point | "removes after-images / persistent states from recurrent gain > 1" | session 2 | nothing |
| 12 | `gain_out_mv` = 100 | optic delta-rate -> injected current (mV) | the optic-lobe synapse onto spiking neurons | session 2: 80 "left LC4 / LPLC2 below threshold for the loom", 120 "makes them fire from walking flow" | nothing |
| 13 | `out_norm` = `l1` | optic -> spiking weights as fractions of total input (not L2) | dendritic normalisation at the projection neurons | session 2 ("L2-normalised output weights make them fire from walking flow") | nothing |
| 14 | `drive_clip_mv` = 35 | +-35 mV clip on the injected current | saturation of the drive | session 2 | nothing |
| 15 | photoreceptor stage `tau_lp_ms` 10, `tau_adapt_ms` 300, `contrast_clip` 2, `eps` 0.02 | low-pass + contrast adaptation | photoreceptor / lamina temporal filtering and light adaptation | session 2 | nothing; not ablated (a static object survives it as a 0.2-0.4 % offset, NOTES 9) |
| 16 | `gain_rr` = 1.0 (`optic.py:52`) | the recurrent optic-lobe gain on `W_rr` | the strength of the lobe's own recurrence | session 2 | nothing. A no-op at 1.0, like #5, and not ablated -- but it multiplies exactly the term #11's adaptation exists to tame ("recurrent gain > 1"), so it belongs in the inventory |
| 17 | `baseline` = 0.5 (`optic.py:51`) | the global operating point of every unit not in `DEFAULT_BASELINE_BY_TYPE` | where a rate unit rests inside `clip(v + b, 0, 1)` | session 2 | nothing (an operating point is not a receptor). Previously listed only inside #7, which overrides it for T4 / T5 |

So of seventeen hand-set entries, the data speak to the SIGN of #1-#5 and #9 (every one confirmed, none changed by
the receptor model: 0 sign changes on T4 / T5 / LPLC2 / LC inputs, docs/audits/object_sweep.md 8.3) and to the
EXISTENCE of slow arms where #6 puts a 150 ms delay (GABA-B on T4 / T5; mAChR-B on Mi4); they say nothing about any
strength, time constant, operating point or normalisation. Two entries are mis-described by their own comments:
#2 (78 % of the edges it multiplies are cholinergic excitation of T5) and #4, whose comment in `optic.py:89` reads
"T4/T5 outputs x4" while the shipped factor is 2.0 (`optic.py:91`).

## 2. Configurations (each measure removed alone; `python scripts/audit_optic.py --list`)

| config | measure | change |
|---|---|---|
| baseline | -- | shipped |
| no_t4_pair_gain | #1 | Mi4 / Mi9 / CT1 / C3 -> T4 x5 -> x1 |
| no_t5_pair_gain | #2 | Tm4 / Tm9 / CT1 / TmY15 -> T5 x5 -> x1 |
| pair_gain_lpi_x1, pair_gain_lpi_x2 | #3 | LPi -> LPLC2 x4 -> x1 / x2 (the round-4 scan repeated under the shipped gains) |
| no_t4t5_out_gain | #4 | T4 / T5 -> * x2 -> x1 |
| no_pair_gain | #1-#5 | `pair_gain = []` |
| no_tau_by_type | #6 | every unit at 10 ms |
| no_t4t5_rectify | #7 | T4 / T5 at 0.5 |
| norm_l1 | #8 | l1 input normalisation |
| gain_in_1 | #9 | contrast gain 1 |
| no_spk_feedback | #10 | `gain_fb` 0 |
| no_optic_adapt | #11 | `adapt_gain` 0 |
| gain_out_80, gain_out_120 | #12 | the two values NOTES 2 rejected |
| out_norm_l2 | #13 | L2 output normalisation |
| no_drive_clip | #14 | clip at 1e9 mV |

Every configuration is one cluster job running, in this order and in one process: `benchmark.py --sections
walk,a,b,walk_gf --seeds 0,1` through `retire_measures.run_one` (the `walk` legacy section is added to the three the
task named because `walk.power_max` / `walk.GF_max` / `loom.GF_peak` are where the LPi factor showed in round 4);
`probe_figure_stages.py --stimulus both` at seeds 0, 1 (baseline 0, 1, 2); `probe_object_sweep.py` ball and `--null` at
seeds 0, 1 (baseline 0, 1, 2). Receptor model: the shipped default (`sign` / `abs`) everywhere. Native backend
(cuda_kernels, event_driven, warp; the benchmark's demo sections with cuda_graphs), NVIDIA B200.

## 3. The batch

`python scripts/audit_optic.py --batch --submit --minutes 90`: cluster run `optic-audit-926faa`, 17 jobs, 0 failed,
27.3 min wall (`out/optic_audit_cluster.log`; every job `device cuda`, NVIDIA B200, torch 2.11.0+cu128; 20-30 min per job:
bench 1.5-2.8 min, stages 10-15 min, object 7-11 min). A first submission (`optic-audit-6c55e7`) failed in all 17 jobs at
`import retire_measures`: **`scripts/retire_measures.py:245` references `brain` at module import without importing it**
(the round-5 `gf_damped` entry), so the script cannot be imported or run at all as shipped -- a defect for that file's
owner; `audit_optic._import_retire_measures` loads it with `brain` pre-bound (no edit to the file). Smoke first:
`optic-smoke-700dac` (2 jobs, 5.6 min; `out/optic_smoke_cluster.log`, `out/optic_audit_smoke/`).

Scatter available for reading the tables: the benchmark half is ONE draw per configuration (seeds 0,1 for section b; the
baseline 0,1,2). The shipped default's own draws on record: `motion.min_dsi` 0.241 / 0.246 / 0.246 (r5 adopt x3; here
0.241), `walk.power_max` 48.4805 in 14/14 + here (bit-identical), `walk_gf.p99` 15.8-20.5 (r5) / 15.7-27.0 (r4; here 18.0),
`loom_escape.GF_peak` 45.4-59.7 (here 48.9-52.7 over three seeds), `loom.GF_peak` 46.8-54.3 (here 47.2). A shift is read as
a result only when it is outside those ranges. The probe halves have their own null (three sims per seed) and 2-3 seeds.

## 4. Ablation table: the benchmark sections

`out/optic_audit/report.md` section 1 (every value; `<cfg>/<cfg>.json`). The columns the task named plus the legacy
`walk` section; bounds in parentheses.

| config | motion.min_dsi (>= 0.1) / correct dirs (8) | loom_escape GF peak (>= 33) / escapes (of 2; baseline of 3) | walk_gf.p99 (< 38) | walk.GF_max (< 38) | walk.power_max (< 50) | loom.GF_peak (>= 20) | rotate.DNp20_flip (< -2) | tally |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.241 / 8 | 52.7 / 3 (52.7, 50.7, 48.9) | 18.0 | 4.63 | 48.48 | 47.2 | -34.1 | 11/0 |
| no_t4_pair_gain (#1) | **0.099 FAIL** / 8 (T4a-d 0.10-0.22, T5 unchanged 0.25-0.50) | 45.7 / 2 (40.8, 45.7) | 23.1 | 8.79 | **55.3 FAIL** | 44.3 | -26.1 | 9/2 |
| no_t5_pair_gain (#2) | 0.149 / 8 (T5a-d 0.15-0.17, T4 unchanged) | 36.1 / 1 (29.4 no escape, 36.1) | 11.3 | 0.00 | **60.7 FAIL** | 28.5 | -13.9 | 10/1 |
| pair_gain_lpi_x1 (#3) | 0.247 / 8 | 48.1 / 2 (43.8, 48.1) | 26.8 | 9.80 | **51.5 FAIL** | 59.5 | -34.7 | 10/1 |
| pair_gain_lpi_x2 (#3) | 0.244 / 8 | 54.6 / 2 (54.6, 50.9) | 32.2 | 9.03 | 48.1 | 51.4 | -29.3 | 11/0 |
| no_t4t5_out_gain (#4) | 0.171 / 8 | **26.7 FAIL / 0 FAIL** (26.7, 20.0) | 9.1 | 4.79 | **59.6 FAIL** | 21.5 | -15.3 | 8/3 |
| no_pair_gain (#1-5) | 0.108 / 8 | **17.8 FAIL / 0 FAIL** (11.9, 17.8) | 12.1 | 0.00 | **70.9 FAIL** | **19.1 FAIL** (no escape range) | **+3.9 FAIL** | 5/5/1 |
| no_t4t5_rectify (#7) | **0.026 FAIL / 6 FAIL** | 47.7 / 2 (47.7, 46.8) | **42.8 FAIL** (walking GF max 42.7 / 34.0 in the loom seeds) | **38.2 FAIL** | **60.4 FAIL** | 56.7 (escape range 9.0 cm) | -11.9 | 6/5 |
| no_tau_by_type (#6) | **0.024 FAIL / 0 FAIL** | 56.5 / 2 | 17.0 | 4.82 | **67.0 FAIL** | 59.4 | -18.7 | 8/3 |
| norm_l1 (#8) | 0.150 / 8 | **0.0 FAIL / 0 FAIL** | 0.0 | 0.00 | 46.7 | **0.0 FAIL** | **+1.2 FAIL** | 6/4/1 |
| out_norm_l2 (#13) | 0.225 / 8 | 240.0 / 2 | **234.8 FAIL** (26 voluntary take-offs in 15 s) | **182 FAIL** | **198.8 FAIL** | 207.9 (50 cm) | **-0.6 FAIL** | 6/5 |
| gain_out_80 (#12) | 0.250 / 8 | 34.6 / 1 (31.9 no escape, 34.6) | 13.6 | 4.84 | **63.8 FAIL** | 34.3 | -26.2 | 10/1 |
| gain_out_120 (#12) | 0.229 / 8 | 66.5 / 2 | 33.4 | 21.5 | **51.7 FAIL** | 53.8 | -45.3 | 10/1 |
| no_drive_clip (#14) | 0.237 / 8 | 45.2 / 2 (41.5, 45.2) | 18.2 | 13.3 | 49.2 | 42.7 | -36.9 | 11/0 |
| no_optic_adapt (#11) | 0.147 / 8 (T5b 0.15, T5d 0.15) | 62.7 / 2 | 28.5 (1 take-off) | 7.92 | **70.7 FAIL** | 56.1 | -26.6 | 10/1 |
| no_spk_feedback (#10) | 0.259 / 8 | 63.9 / 2 (50.3, 63.9) | 20.9 | 4.99 | **63.3 FAIL** | 51.4 | -32.0 | 10/1 |
| gain_in_1 (#9) | 0.200 / 8 | 40.3 / 2 | 12.1 | 4.96 | **77.8 FAIL** | 38.0 | -30.0 | 10/1 |

Two things about the tally column. It is pass / fail, plus a third figure where a check is neither: `loom.escape_cm`
is `MISSING` (no value) under `no_pair_gain` and `norm_l1`, and the generator counts only `status == 'KNOWN GAP'`
(`scripts/audit_optic.py:308`), so `report.md` prints 5/5 and 6/4 for 10 of those configurations' 11 checks; the
rows above are corrected to 5/5/1 and 6/4/1. And `loom.escape_cm`'s criterion is `notnone`, so it scores a PASS on
any value at all -- the runaway `out_norm_l2` passes it at 50.0 cm and `no_t4t5_rectify` at 9.0 cm against a 3.5 cm
reference.

**These tables are one draw per configuration, and the motion / loom / rotate sub-sections are NOT bit-stable across
runs.** An independent cluster rerun of `pair_gain_lpi_x1` (skeptic run `optic-verify-acf696`, 3 jobs, 0 failed,
B200, same cache `sum|W|` 121,460,584, same seeds; `out/optic_verify_cluster.log`,
`out/optic_verify/pair_gain_lpi_x1/pair_gain_lpi_x1.json`) reproduces `walk.GF_max` 9.79509, `walk.power_max`
51.5078 and `walk.power_sustained` 20.2129 bit for bit, but gives `loom.GF_peak` 52.7937 against 59.5391 (-6.75 Hz),
`rotate.DNp20_flip` -24.2460 against -34.7047 (+10.46 Hz), `loom_escape.GF_peak` 54.0282 against 48.1493 (+5.88 Hz),
`motion.min_dsi` 0.237748 against 0.246778 and `walk_gf.p99` 26.3640 against 26.7633. So only the `walk.*` columns are
bit-stable; the `loom` / `loom_escape` / `rotate` / `min_dsi` columns swing by 6-10 Hz (0.009 on DSI) inside one
configuration. Read them accordingly: `no_t4_pair_gain`'s **min_dsi 0.099 is on the bound, not a reproducible FAIL**
(one rerun moved min_dsi by 0.009 and T4a's subtype DSI from 0.3907 to ~0.41), and the `loom.GF_peak` and
`rotate.DNp20_flip` differences quoted below are inside a single configuration's own run-to-run swing.

Reading, per check:

* **motion.min_dsi (the motion detector).** Load-bearing: #7 the T4/T5 ReLU (0.026, 6 of 8 directions), #6 the per-type
  time constants (0.024, 0 of 8 directions: without the 150 ms delay line the preferred directions are wrong, not just
  weak), #1 the T4 inhibition x5 (T4 subtypes 0.10-0.22; the check's minimum 0.099 sits ON the 0.10 bound and is not a
  reproducible FAIL -- one rerun of another configuration moved min_dsi by 0.009). The T5 entry (#2)
  halves T5's DSI (0.44/0.34/0.29/0.24 -> 0.15-0.17) but the check still passes; all pair gains off gives 0.108 (passes)
  with every subtype at 0.11-0.16 -- the ReLU and the taus, not the gains, are what makes the detector direction
  selective; the x5 gains double its DSI. norm_l1 0.15, gain_in_1 0.20, no_optic_adapt 0.147 (T5b / T5d): moderate.
  Everything else is within +-0.02 of the baseline (the r5 scatter is 0.005).
* **loom_escape (section b) and the legacy loom.GF_peak.** Load-bearing: #4 the T4/T5 output x2 (26.7 / 20.0 Hz, 0
  escapes; legacy 21.5) and #8 the L2 input normalisation (with l1 the optic lobe drives nothing: loom 0 Hz, walking
  GF 0, DNp20 flip +1.2 -- see section 6 for why). #2 the T5 entry costs the loom half its peak (29.4 / 36.1 Hz, 1
  escape of 2; legacy 28.5) because T5's main excitation (Tm9 / Tm4) is inside the x5; #12 gain_out 80 likewise (31.9 /
  34.6, 1 escape; legacy 34.3) -- both are marginal, not dead. #3 LPi x1 / x2 leaves the loom inside the scatter
  (43.8-54.6); the legacy peak reads 59.5 / 51.4 against the baseline's 47.2, but the rerun above puts x1's own
  legacy peak at 52.8, so that rise is inside one configuration's swing and is not a measured difference. Removing
  #10 the spiking feedback, #11 the optic adaptation, or #14 the clip does not cost the loom (50-64 Hz). Note also
  that `loom_escape.GF_peak` is a MAX over seeds -- 3 draws for the baseline, 2 for every ablation -- so the baseline
  figure is biased upward relative to the ablations by the extra draw.
* **walk_gf.p99 / walk.GF_max (self-motion drive of the giant fibre).** Load-bearing: #7 the ReLU (p99 42.8, GF max
  38.2: a linear T4/T5 drives LPLC2 -> GF during walking), #13 the l1 output normalisation (L2: 235 Hz, 26 take-offs in
  15 s -- the runaway NOTES 2 recorded). Costs, all under the bound: #3 LPi x1 / x2 p99 26.8 / 32.2 and GF max 9.80 /
  9.03 (round 4 under the damped gains measured p99 26.98-31.14 / 23.63-31.09 and GF max 9.7681 / 9.0290: **not the
  same numbers** -- x1's p99 26.7633 falls below round 4's range and x2's 32.1787 above it, and GF max 9.795086 is
  0.027 Hz from round 4's 9.7681), #12
  gain_out 120 p99 33.4 / GF max 21.5, #14 no clip GF max 13.3 (p99 18.2: the clip binds on the walking GF's peak frames
  only), #11 no adaptation p99 28.5, #1 no T4 gain 23.1 / 8.8.
* **walk.power_max (the hand-set 50 Hz bound of handover item 4).** **13 of the 16 ablations fail it, over
  51.51-198.77 Hz** (51.51, 51.70, 55.26, 59.63, 60.44, 60.75, 63.33, 63.82, 66.97, 70.66, 70.90, 77.85, 198.77 =
  `out_norm_l2`); the three that pass are `pair_gain_lpi_x2` 48.12, `norm_l1` 46.65 and `no_drive_clip` 49.25, against
  the baseline's 48.48. The check is again two-regime and non-monotone (LPi x1 51.5 / x2 48.1 / x4 48.5 under the
  shipped gains vs 51.5 / 48.1 / 79.5 under the damped ones -- the x1 and x2 walk.power_max values are bit-for-bit
  round 4's 51.5078 / 48.1205, and that is the ONLY column of theirs that is, the x4 point moved by 31 Hz with the
  DNp01 damping; `gain_out` 80 -> 63.8, 100 -> 48.5, 120 -> 51.7). Nothing here should be tuned to it; it is reported
  because handover item 4 asks for every scan of it to be on file.
* **rotate.DNp20_flip.** Dead under norm_l1 (+1.2), no_pair_gain (+3.9), out_norm_l2 (-0.6); weakened by the T5 entry
  (-13.9), the T4/T5 output gain (-15.3), the ReLU (-11.9) and the taus (-18.7); the rest -26 to -45 (baseline -34).
  That last spread is not a measurement: the rerun of `pair_gain_lpi_x1` moved its own value from -34.7 to -24.2, so
  the whole -26 to -45 band is one configuration's swing.

**Which measures are load-bearing for which check** (a check fails without them, outside the baseline scatter):
#6 taus and #7 ReLU for the motion detector (and #7 for the walking-GF margin); #1 for the T4 half of the DSI; #4 the
T4/T5 output gain and #8 the L2 input normalisation for the loom; #13 the l1 output normalisation against runaway; #2
and #12 are marginal for the loom (half the peak, 1 escape of 2). Load-bearing **only through `walk.power_max`**: #3
LPi x4 (x1 scores 10 pass / 1 FAIL, the FAIL being walk.power_max 51.5078 against a baseline recorded at 48.4805 in
14/14 draws with zero scatter -- a real +3.0 Hz effect crossing a scored bound, which is the identical number round 4
used to conclude "cannot be retired"; what is disputed here is the bound's meaning, not the measurement), and the
same wording applies to #10 spiking feedback, #11 optic adaptation and #9 the contrast gain, whose only FAIL is also
that check. Not load-bearing for any scored check at all: #14 the drive clip (11/11 without it).

**Which the data contradict:** #2. The entry is described as delayed inhibition for T5 but 78 % of its edges are Tm4 /
Tm9 -> T5 acetylcholine (nAChR-alpha5 on T5; Tm9 validated ACh by TAPIN in Nern 2025), so it is a x5 on the OFF
detector's centre excitation too; removing it halves T5's DSI and the loom peak because T5's drive, not its
selectivity, is what it buys. No other measure is contradicted by the tables (they say nothing about strengths); the
T4 entry (#1) is at least sign-consistent with what it claims (100 % inhibitory edges, Rdl / GluCl-alpha on T4).

## 5. The figure-propagation map: where the object is lost

`scripts/probe_figure_stages.py`, baseline, seeds 0-2 (`out/optic_audit/baseline/stages_s{0,1,2}.json`; report.md
section 2). Statistic per type: (A - B) in the object columns minus (A - B) in the background columns, z against the
background scatter -- the signed and the |dev| version -- and the same for (C - B), the none-vs-none pair. A type
"carries" when |z(A-B)| >= 3 in every seed with one sign and above its own (C-B).

**The none-vs-none null is clean only in the aggregate.** No type reaches |z(C-B)| >= 3 in *every* seed, and the
~35 named types of the apple map have mean |z(C-B)| <= 0.9 -- but single-seed values go far higher:
from `baseline/stages_s{0,1,2}.json` the per-seed |z(C-B)| reaches 5.53 (MeVP11 signed), 5.25 (TmY17), 4.56 (LC25),
3.42 (Tm31), 3.22 (MeLo3b) in the ball map and 1.83 (Pm10 abs) in the apple map; the largest seed-mean |z(C-B)| is
1.95 (MeLo3b, ball) and 1.09 (LLPC3, apple). Nor is a single value stable: a rerun of baseline seed 0 gives MeVP11
ball (C-B) = +4.20 where the recorded run gives -5.53. Read the null as "no type carries in 3/3 by chance", not as
"|z| <= 0.9 everywhere".

**Stage assignment: the type-name family rule, in all 17 runs -- not the Nern 2025 table.** Every `stages_s*.json`
records `stage_source: "type-name family rule"`, because `data/external` is not shipped to the cluster. That is not
cosmetic: 14 `ol_intrinsic` types the Nern table stages as VPN (H1, LC14a-1/2, LC14b, LT54, MeVPLp2, MeVPMe5/7/10/12/13,
MeVPOL1, dCal1, l-LNv) get no stage under the family rule and were therefore **unscored in every map**.

### 5.1 Static apple (159 object / 1050 background columns; 3 seeds)

| stage | types scored | carry signed / abs | best (signed z, abs z; mean of 3 seeds) | small-field / loom types |
|---|---|---|---|---|
| 1 lamina | 10 | 5 / 2 | L1 +18.4 / +4.2, L2 +12.0, L3 +6 | -- |
| 2a medulla intrinsic | 40 | 5 / 7 | Mi1 -8.8 / +0.7, Dm15 abs +7.9, Mi9 +2.7 / +5.6, Mi4 -1.9 / +4.0 | -- |
| 2b medulla -> lobula | 33 | 5 / 5 | Tm3 -6.8 / +7.1, Tm9 +0.5 / +8.1, Tm20 +5.3, Tm4 +4.8, Tm1 +4.5 / -5.9, Tm2 +3.2 / -3.4 | **Tm5Y +0.2 / +1.3, TmY21 -0.6 / -0.2, TmY13 +2.2 / +0.1, TmY5a +1.8 / -0.2, TmY3 +0.8 / +1.2: none carries** |
| 3 T cells | 13 | 2 / 2 | T4c +4.2, T4d +4.0 (T4b +2.6, T4a -0.1; T5a-d +1.2 to +2.1) | **T2 +2.7 / -1.1, T2a -0.6, T3 -1.0 / -1.3: none carries** |
| 4 Li / LPi | 4 | 0 / 0 | Li19 +0.5 | -- |
| 5 VPN (drive) | 17 | 1 / 2 | **LPLC2 +3.6 / +3.5** (carries in 3/3) | LC10a +0.1 / +0.0, LC4 +0.1 / +1.1, LC17 -0.7, LC12 -0.2 |

The static figure is carried through the lamina, the medulla interneurons and the medulla -> lobula columnar types
(the ON channel lowered: Mi1 -8.8, Tm3 -6.8; the OFF channel raised: L2 +12, Tm1 +4.5, Tm2 +3.2, Tm4 +4.8, Tm20
+5.3; Tm9 / Mi9 / Mi4 / Dm15 in |dev|), then by T4c / T4d and LPLC2 (the loom chain sees the stationary-relative
object, as NOTES 9 found), and is **lost at the same stage as NOTES 9 named: the inputs of LC10 / LC11 -- Tm5Y,
TmY21, TmY13, TmY5a, TmY3 (stage 2b) and T2, T2a, T3 (stage 3)** -- with LC10a / LC11 / LC4 at zero. This reproduces
the session-9 `probe_figure_ground` map (Mi4 z 7.4 there vs +4.0 abs here, Tm3 4.0 / -6.8 signed, Tm5Y 0.5 / +0.2,
TmY21 -0.2 / -0.6, T2 -0.8 / +2.7, T3 -1.3 / -1.0) now with a null and three seeds.

### 5.2 Moving ball (59 object / 1117 background columns; 3 seeds)

The retinotopic time-mean is a weak statistic for a ball that sits in any one column ~1/8 of the window: best types Mi9
abs +3.5 (carries), T4b +2.9, Tm9 abs +2.4, L2 +1.9; the small-field types T2 +0.7 / +0.3, T3 -0.1, Tm5Y +0.3 / +0.1,
TmY21 -0.1 / +0.3 and every LC type |z| <= 0.6. The per-cell object-sweep statistic (section 6) is the sensitive one;
it puts the same loss at the same stage.

### 5.3 Connectivity between the last carrying stage and the first losing stage (baseline weights; report.md section 4)

Inputs as the share of the target's L2-normalised, pair-gain-scaled rate input; sign under the shipped receptor model;
every entry here is at pair gain x1 (no hand-set gain touches this stage: the T4/T5 x2 reaches TmY5a through T5d 4.5 %
and T2 through T5b 2.1 % only). The input types' apple figure (signed z) is from 5.1.

| target (own apple z) | inputs: share, sign, apple z of the input | linear estimate sum(signed share x input figure) vs own figure |
|---|---|---|
| T3 (-1.0) | Mi1 22.5 % + (-8.8); Tm1 16.0 % + (+4.5); Tm3 8.2 % + (-6.8); Tm4 6.7 % + (+4.8); Pm5 5.6 % - (+4.9); Pm1 5.3 % - (-5.6); Mi2 3.3 % -; Li26 3.2 % - | -0.00041 vs -0.00114: the ON (Mi1, Tm3: lowered) and OFF (Tm1, Tm4: raised) carriers arrive with opposite signs through excitatory synapses and cancel; the Pm inhibition cancels again |
| T2 (+2.7) | Tm2 11.3 % + (+3.2); L5 8.3 % + (-1.6); Tm3 7.0 % + (-6.8); T2 4.7 % +; Pm2a 4.2 % -; Mi1 3.4 % + (-8.8); Mi2 3.2 % -; TmY3 2.8 % + | +0.0031 vs +0.0054: the same ON / OFF cancellation, less complete (T2 keeps a +2.7 that does not clear 3 in every seed) |
| T2a (-0.6) | Mi1 16.3 % + (-8.8); Tm1 13.1 % + (+4.5); MeLo10 5.9 % -; Li25 4.8 % -; Mi2 4.5 % -; Pm4 / Pm3 / Pm10 4 % each - | +0.0004 vs -0.0009 |
| Tm5Y (+0.2) | Tm20 14.0 % + (+5.3); Li19 9.3 % - (-0.2); Y3 7.5 % + (-0.5); Li25 4.6 % -; Tm32 3.9 % - (+4.2); Tm5a 3.4 % +; T2a 3.2 % +; photoreceptors 0.1 % | +0.00084 vs +0.00081: the only carrying input is Tm20 at 14 %, and Tm32 (a carrier, inhibitory) takes a third of it back |
| TmY21 (-0.6) | TmY5a 9.7 % - (+1.8); TmY13 7.6 % + (+2.2); Tm20 5.8 % + (+5.3); Tm5a 5.1 % +; Tm5Y 3.8 % +; TmY17 3.5 % +; Dm3a 3.4 % - | +0.0014 vs -0.0021: its two largest inputs are non-carriers (TmY5a, TmY13), as are Tm5a and Tm5Y; its **third input Tm20 at 5.8 % IS a carrier** (z +5.3) and supplies its largest single positive contribution, +0.000615, which the non-carriers and Dm3a inhibition then swamp |
| TmY13 (+2.2) | Mi9 27.9 % - (+2.7); Mi4 22.8 % - (-1.9); TmY3 4.4 % +; Mi10 4.0 % +; Y3 3.5 % +; Tm4 3.2 % + (+4.8) | +0.0001 vs +0.0041: half its input is Mi9 + Mi4 inhibition with opposite figures |
| TmY5a (+1.8) | Tm4 11.1 % + (+4.8); Tm3 10.2 % + (-6.8); T5d 4.5 % + x2; T2 3.4 % +; Y3 3.2 % +; TmY9a 3.1 % + | +0.0065 vs +0.0116: Tm4 (+) and Tm3 (-) cancel |

So the loss is structural and sign-correct -- but it is **two mechanisms, not one**. At T3 / T2 / T2a it is
cancellation: the types sum ON- and OFF-channel carriers that respond to a dark object with opposite signs, through
excitatory synapses of equal weight (T3: Mi1 + Tm3 30.7 % vs Tm1 + Tm4 22.7 %), and the Pm / Li inhibition is not
tuned to a surround in this model (a rate unit per cell with no spatial structure beyond the connectome's own). At
Tm5Y it is dilution plus scatter: 14 % of a +0.0107 carrier gives an estimate of +0.00084 against its own background
sd of 3.9e-3. Apart from the two x2 edges named above (T5d -> TmY5a 4.5 %, T5b -> T2 2.1 %), no pair gain, time
constant or operating point sits on these edges, so no hand-crafted measure can be "what kills the object" here --
and section 7 finds only one ablation under which any of these types crosses the carry threshold, in a lobe that is
quiet rather than selective.

## 6. Where the null comes from, and the object sweep under the shipped gains

**The rate optic lobe is deterministic; the whole none-vs-none scatter is the spiking feedback.** With `gain_fb = 0`
(`no_spk_feedback`) the none-vs-none pair is **numerically zero, not bit-identical**: 16 of the 78 diff statistics in
`no_spk_feedback/obj_null_s0.json` are non-zero, at 1e-9 to 3e-7 mV (LC11 `diff_max_over_cells` 5.18e-09,
`diff_peak_100ms` 2.62e-07, `diff_tuning_peak` 5.14e-08, `diff_mean_over_cells` 1.28e-11; LC10b, LPLC2 and LC4
likewise), and those values differ between s0 and s1 and between the recorded run and a rerun (max delta 4.8e-09).
Nor are the two ball seeds identical beyond the headline statistic: `diff_max_over_cells_mean_mv` matches for all six
LC types, but LC10b `rate_hz_mean` is 1.65965 vs 1.68158, `rate_hz_max_cell` 7.667 vs 7.750, `cells_over_1hz` 49 vs 48,
and LPLC2 `cells_over_1hz` 7 vs 6. The rate lobe is deterministic; **the spiking network is not, even at
`gain_fb = 0`**. The residual is five to seven orders below the shipped null (0.06 mV), so it is still the right
control -- it just is not exact arithmetic. Under the baseline the null is +0.06 to +0.24 mV on the LC types and
0.008-0.04 rate units on the optic units, i.e. the LIF's run-to-run nondeterminism (round 4: the CSR x dense product)
enters the rate lobe through `W_rs` and is what every difference statistic has to be read against. This answers
object_sweep.md 8.8's open caveat.

**The handover's object null re-established under the shipped gains** (baseline, 3 seeds; `obj_{ball,null}_s{0,1,2}.json`;
mV, `diff_max_over_cells_mean_mv`; z on the null SD):

| type | ball per seed | mean | null per seed | null mean +- SD | z | round 3 (damped gains, 5 seeds): ball / null / z |
|---|---|---|---|---|---|---|
| LPLC2 | +0.344 +0.333 +0.311 | +0.329 | +0.269 +0.154 +0.291 | +0.238 +- 0.074 | +1.2 | +0.351 / +0.139 / +5.4 |
| LC11 | +0.079 +0.054 +0.043 | +0.058 | +0.053 +0.058 +0.071 | +0.061 +- 0.009 | -0.2 | +0.074 / +0.066 / +0.4 |
| LC10a | +0.067 +0.076 +0.046 | +0.063 | +0.063 +0.059 +0.069 | +0.063 +- 0.005 | -0.1 | +0.082 / +0.085 / -0.1 |
| LC10b | +0.084 +0.098 +0.105 | +0.096 | +0.131 +0.125 +0.217 | +0.158 +- 0.052 | -1.2 | +0.125 / +0.082 / +1.0 |
| LC16 | +0.084 +0.103 +0.111 | +0.099 | +0.071 +0.118 +0.069 | +0.086 +- 0.027 | +0.5 | +0.068 / +0.120 / -0.8 |
| LC4 | +0.058 +0.090 +0.078 | +0.076 | +0.064 +0.103 +0.097 | +0.088 +- 0.021 | -0.6 | +0.074 / +0.096 / -1.0 |
| Mi4 (rate units, best cell) | +0.048 +0.049 +0.043 | +0.047 | +0.010 +0.008 +0.008 | +0.0085 +- 0.001 | +35 | +0.046 / +0.008 / +22 |
| Tm3 | +0.108 +0.105 +0.111 | +0.108 | +0.030 +0.017 +0.020 | +0.022 +- 0.007 | +12.5 | +0.106 / +0.025 / +16 |
| Mi1 | +0.075 +0.073 +0.070 | +0.073 | +0.025 +0.011 +0.014 | +0.017 +- 0.007 | +7.6 | +0.072 / +0.015 / +28 |
| Tm5Y | +0.051 +0.044 +0.045 | +0.047 | +0.047 +0.020 +0.022 | +0.030 +- 0.015 | +1.1 | +0.050 / +0.029 / +2.6 |
| TmY21 / T2 / T3 / TmY13 / TmY5a | | +0.038 / +0.041 / +0.025 / +0.019 / +0.039 | | +0.040 / +0.038 / +0.023 / +0.024 / +0.038 | -0.1 / +0.2 / +0.4 / -0.3 / +0.1 | all abs z <= 1.2 |

The LC10 / LC11 / LC4 / LC16 entries sit at the null in 3/3 (as in round 3); the medulla carries the ball at the same
magnitudes as round 3 (Mi4 +0.047 vs +0.046, Tm3 +0.108 vs +0.106) and Tm5Y / TmY21 / T2 / T3 sit at the null.
**LPLC2's excess over the null is smaller than round 3's**: +0.09 mV (z +1.2, 3 ball values all above 2 of 3 null values)
against +0.21 mV (z +5.4, 5/5 above 5/5) under the damped gains; the ball arm is unchanged (+0.329 vs +0.351), the null
rose (+0.238 vs +0.139: the null seeds 0 and 2 are at +0.27 / +0.29). That is a gain-change effect (the retired DNp01
damping changes the spiking feedback's scatter, not the stimulus), as the handover predicted. Tested exactly rather
than by eye: a tie-aware rank-sum of the 3 shipped-gain null draws (0.154, 0.269, 0.291) against round 3's 5
damped-gain draws (0.081, 0.124, 0.137, 0.174, 0.178) gives U = 13 of 15, one-sided **p = 4/56 = 0.071**, while the
ball arm's apparent drop gives p = 0.29. So "the null rose while the ball arm did not move" is the right description
and it is not significant at n = 3 vs 5.

**The deterministic magnitudes** (`no_spk_feedback`, identical in both seeds, null exactly 0): rate units' best-cell
(ball - none) |dev| Mi4 0.047, Mi1 0.068, Tm3 0.104, **Tm5Y 0.072, TmY21 0.062, TmY5a 0.084, T2 0.066, T3 0.041,
TmY13 0.035**; spiking drive LPLC2 +0.54 mV, LC10b +0.30, LC10a +0.080, LC16 +0.062, LC4 +0.052, LC11 +0.046
(best-cell rate differences: LC10b +1.6 Hz, LC4 +1.3, LC16 +0.6, LC10a +0.4, LPLC2 +0.4, LC11 +0.08). So in the
deterministic lobe the small-field types DO pass a single-cell object signal of the medulla's size (T2 0.066 vs Mi4
0.047), which the baseline's feedback noise at those types (null 0.03-0.04, four times the medulla's 0.008-0.017) hides;
what does not follow is a retinotopic population figure (5.1-5.2: population-mean diffs at T2 +0.002, Tm5Y -0.001 rate
units) or a drive: LC11 / LC10a receive 0.05-0.08 mV against the 7 mV criterion, 7-11x less than LPLC2 / LC10b. The
pooling arithmetic at the `l1` output stage: an LC11 cell pools 94 columns (median; LC10a 32, LC10b 41, LPLC2 86) with
30 % of its optic input from its best three columns (LC10a 48 %, LPLC2 34 %), so a few cells at 0.07 in ~3 of 94
columns give ~0.1 mV at 100 mV per unit of mean input deviation. With `out_norm = l2` the same drives are +2.5 (LC11; 3.7 / 1.3 in the two seeds) /
+1.9 (LC10a) / +4.0 mV (LPLC2) over nulls of +1.1 / +0.6 / +1.6 (2-seed means) -- still under 7 mV -- while the whole brain
runs away (section 4). The object signal at LC10 / LC11 is therefore a per-type output-normalisation question (how a small-field
projection neuron weights a few columns) that neither `l1` nor `l2` answers and no data in the tree decides.

## 7. Does any measure kill the object at that stage?

One does, and only in a lobe that has stopped working. Across the sixteen ablations (report.md section 2, "Named
types"; `out/optic_audit/summary_compact.txt`), the small-field inputs of LC10 / LC11 carry the static apple in none
of them (Tm5Y |z| <= 0.7, TmY21 <= 1.1, T3 <= 2.4, T2 <= 3.3 in a single configuration [pair_gain_lpi_x1, not in
every seed]) **except `norm_l1`, where the report's own carry flags read T2 True / True (+5.4 signed, +6.3 abs),
T3 True / True (-4.6, +7.9) and TmY13 False / True (+5.1)** -- for the reason given in the first bullet below, a
~110x quieter linear lobe, not new selectivity. The object-sweep statistic at Tm5Y /
TmY21 / T2 / T3 stays at the null in every configuration with a null (the exceptions are z's over a 2-run null SD of
0.001-0.004: `no_t4t5_out_gain` Tm5Y "z 323" on SD 0.0003, `no_t4t5_rectify` T3 "22" on 0.001 -- not results). What the
ablations do to the object chain:

* `norm_l1` (#8): the figure z's explode (L1 +519, Tm4 +81, T4c +17, T2 +5.4, T3 -4.6 apple; T4b +101, T3 +62, T2 +28,
  LC12 +46 ball) -- crossing the carry threshold at T2, T3 and TmY13 -- because the lobe becomes quiet and linear
  while the magnitudes collapse **~110x, not 10x**: the apple signed figure at T2 is 4.8e-05 here against 5.5e-03 at
  the baseline, with the background sd down from 2.2e-03 to 9.1e-06 (~240x); T3 2.7e-05 vs 1.1e-03 (ball |dev| figure
  at T2 0.00008 vs 0.0028; object-sweep drives 0.000-0.02 mV). The L2 normalisation is what amplifies the columnar
  signal (NOTES 9's "3-6x") and what carries the loom; it is not what loses the object, and a z computed against a
  240x smaller null is not a demonstration that these types carry a figure.
* `no_optic_adapt` (#11): the static-apple figure collapses -- **signed** z at L1 +1.4 vs +18.4 (the abs pair is
  +3.6 vs +4.2; the pairing previously printed here, +3.6 vs +18.4, mixed the two measures), Mi1 -2.9 vs -8.8, Tm3
  -1.7 vs -6.8 -- and
  the object-sweep noise floor doubles (LC10a null +0.34 vs +0.06 mV, LC11 +0.29): without the 400 ms relaxation the
  units hold after-images and the background scatter swamps the retinotopic figure; the measure is what makes a static
  figure measurable, not what removes it.
* `gain_in_1` (#9) scales every figure down (L1 +6.6, Mi1 -4.7, Tm3 -4.6; LPLC2 +3.5 unchanged).
* `no_tau_by_type` (#6): Mi4 / Mi9 / Tm9 lose their |dev| figure (+4.0 / +5.6 / +8.1 -> -1.2 / -3.1 / -1.6): the slow
  taus are what lets the delay-line types integrate a static figure; T4/T5 lose theirs (+4.2 -> +1.5).
* `no_t4t5_rectify` (#7): T4c / T4d lose the apple (+4.2 -> -2.5), LPLC2's apple drive drops (+3.6 -> +2.0).
* The pair gains (#1-#5, `no_pair_gain`): every stage-1 to 2b figure unchanged (L1 +17 to +22, Mi1 -8 to -10, Tm9 +8
  to +12); T4 carries the apple MORE without its inhibition gain (T4d +6.4 / +6.9 with #1 / all off); LPLC2 +3.5 to
  +3.7 in all of them (the apple through LPLC2 does not need the T4/T5 x2: `no_t4t5_out_gain` +3.7). The moving ball at
  LPLC2 does: object-sweep LPLC2 +0.16 (`no_t4t5_out_gain`) / +0.14 (`no_pair_gain`) / +0.20 (`no_t5_pair_gain`) vs
  +0.33 baseline, with nulls of +0.06-0.09.
* `no_spk_feedback` (#10): the maps are the baseline's (L1 +14.7, Mi1 -7.7, Tm9 +9.0, T4d +3.9, LPLC2 +3.6; small-field
  |z| <= 2.4) -- the loss is not the feedback noise.
* `no_drive_clip`, `gain_out_80/120`, `out_norm_l2`, LPi x1 / x2: rate-unit maps unchanged (these act after the rate
  lobe); `out_norm_l2` weakens the apple's LPLC2 drive figure (+1.1) because the drive scatter grows.

## 8. Verdicts per measure

The rule of anti_runaway.md applies: retire nothing unless a replacement passes with a status margin; data-driven
replacements only.

* **#1 Mi4/Mi9/CT1/C3 -> T4 x5 -- keep (load-bearing for T4's DSI).** Sign-correct; the data offer the existence of a
  slow GABA arm on T4 (GABA-B-R1 `high`) that the fast model lacks, which is a documented mechanism, not a strength.
  Not tunable to the object (it is not on the small-field edges).
* **#2 Tm4/Tm9/CT1/TmY15 -> T5 x5 -- mis-described; keep provisionally as a T5 drive gain, re-label it.** The data
  contradict its comment (78 % of its edges are ACh excitation); its ablation halves T5's DSI and the loom peak. A
  data-consistent form would separate the inhibitory arm (CT1 / TmY15 GABA, 22 % of the edges) from the excitatory
  centre (Tm9 / Tm4), but any factor on either is a hand-set number; this is a re-labelling and a scan for another
  round, not a retirement.
* **#3 LPi34/43 -> LPLC2 x4 -- load-bearing only through `walk.power_max`; the round-4 verdict stands under the
  shipped gains** (x1 51.5 FAIL / x2 48.1 / x4 48.5 on walk.power_max; GF_max 9.80 / 9.03 / 4.63; loom inside one
  configuration's own swing). x1 scores 10 pass / 1 FAIL and that FAIL is a real +3.0 Hz move against a baseline
  recorded at 48.4805 in 14/14 draws with zero scatter -- it is the same number round 4 used to say the entry cannot
  be retired. What this audit disputes is the bound, not the measurement. It also suppresses the walking GF as
  designed, by amounts the suite does not score. Sign confirmed, plus mGluR `mid` on LPLC2 (a slow negative arm the
  data support). Handover item 4's decision on the walk.power_max bound comes first.
* **#4 T4/T5 -> * x2 -- keep (the loom dies without it: 20-27 Hz, 0 escapes).** No data on strength.
* **#5 * -> LC4/LPLC2 x1 -- a no-op entry; remove the line for clarity (no behaviour change).**
* **#6 per-type taus -- keep (0 of 8 preferred directions without them).** The receptor table's slow classes (GABA-B
  on T4/T5, mAChR-B on Mi4) are the data-backed place for slow kinetics if the delay line is ever re-derived: a slow
  GABA term on T4/T5 (the receptor model's `full` mode has the machinery) instead of slow membranes on Mi4/Mi9/CT1/Tm9.
* **#7 T4/T5 ReLU -- keep (DSI 0.026 and walking GF 38-43 Hz without it).**
* **#8 L2 input normalisation -- keep (l1 silences the lobe's output entirely).** Not data-driven either way.
* **#9 contrast gain 3 -- not load-bearing; the "20 % contrast saturation" referent is physiology (Laughlin), not in
  the tree's data.**
* **#10 spiking feedback 0.5 -- its only FAIL is walk.power_max (63.3); it IS the null.** For every probe that reads
  a difference against a none-vs-none null, `gain_fb = 0` is the near-exact control -- a deterministic rate lobe with a
  residual of 1e-9 to 3e-7 mV from the still-nondeterministic spiking network (section 6), five to seven orders below
  the shipped null; it should become a standard arm of the object assays (an experiment setting, not a default
  change).
* **#11 optic adaptation -- keep (the static figure is unmeasurable without it; walk.power_max 70.7).**
* **#12 gain_out 100 -- keep; 80 is loom-marginal (1 escape of 2), 120 walking-GF-marginal (p99 33.4).** No data.
* **#13 l1 output normalisation -- keep against runaway (l2: 182-235 Hz walking GF); it is also where the LC10 / LC11
  object drive is divided by the receptive field (section 6).** The data-driven alternative would be a per-type
  dendritic normalisation from a published source; none is in the tree.
* **#14 drive clip 35 mV -- not load-bearing (11/11 without it); costs walk.GF_max 4.6 -> 13.3.** A candidate for
  retirement under the anti_runaway rule once replicated (one draw here) -- it is the one measure whose removal
  passes every check with the loom, DSI and walking margins inside the scatter.
* **#15 photoreceptor stage -- not ablated.**

**On the object:** the stage is named (5.1, 5.3) and no hand-crafted measure is on its edges or removes the loss (7);
the mechanism at that stage is ON/OFF cancellation through sign-correct excitatory convergence plus untuned Pm / Li
inhibition, and at the next stage the l1 pooling of a few columns per LC cell (6). The data-driven routes left are
(a) the receptor model's slow classes on the T cells (the tables give GABA-B on T2 `mid` / T3 `high`, mGluR on T2), (b)
the flyvis per-type parameters (Lappalainen et al. 2024 -- published, not in `flyverse/data/`), and (c) a per-type
output normalisation from a published dendritic measurement. None is a tuning to the object.

## 9. Caveats and limits

* **LIMIT: the figure-stage map is not reproducible at fixed seed, so the replicate unit is "runs", not "seeds".**
  Rerunning `probe_figure_stages` for baseline seed 0 on the cluster (`out/optic_verify/baseline/stages_s0.json` vs
  `out/optic_audit/baseline/stages_s0.json`, same `stage_source`, same B200) and comparing the 234 apple and 376 ball
  per-type z(A-B) values: median |dz| 0.36 / 0.49, p90 1.34 / 1.70, max 5.24 (MeLo7 signed +0.52 -> +5.76) / 7.56, and
  the single-seed |z| >= 3 threshold flips for 15 of the 234 apple type-measures (Dm18, Dm3a, Dm3b, L1 abs 3.98 ->
  2.79, L4, MeLo7, Mi4 abs 3.72 -> 3.00, Mi9, Pm4, Pm5, Tm2 ...). The figures themselves move, not only the background
  sd (Tm3 apple -1.508e-02 -> -1.740e-02; MeLo7 1.28e-03 -> 6.34e-03). The object sweep behaves the same way at fixed
  seed 0: LC10b ball 0.0835 -> 0.2514 (3x), LC11 ball 0.0788 -> 0.1201, LPLC2 null 0.2693 -> 0.2286, while the rate
  types stay put (Mi4 0.0482 -> 0.0464, Tm3 0.1082 -> 0.1010). Consequences: **every per-type z in sections 5-7 should
  be read as +-1.5**, every per-stage carry COUNT is soft, and the marginal calls (T2 +2.7, T4b +2.6 / +2.9, T4c min
  3.59 -> 3.16 on rerun) are calls about noise. What survives across draws is the coarse statement: the carriers stay
  far above the criterion (L1 15.9-20.9, L2 10.5-13.4, Mi1 -8.2 to -10.1, Tm3 -5.4 to -8.6, Tm20 4.6-6.2, LPLC2
  3.45-3.63) and the LC10 / LC11 inputs stay far below it (Tm5Y 0.1-0.5, TmY21 -0.6, T2 1.7-2.2, T3 -0.6 to -2.5,
  LC10a +-0.2).
* **LIMIT: the static-apple protocol is not A/B matched on odour.** `run_apple` builds the stimulus arm with
  `fruit_set='apple'` -- so the apple's ODOUR is present 9 cm away -- and the none arms with `fruit_set='all'` plus
  every fruit teleported to (9,9,9) (`scripts/probe_figure_stages.py:209-217`). (A-B) therefore contains an olfactory
  difference as well as a visual one, and the (C-B) null does not control for it. The retinotopic subtraction removes
  its spatially uniform part, which is most of it, but the odour reaches the lobe through the same `gain_fb` path this
  document identifies as the noise source, so the arms should be matched in the next version.
* One benchmark draw per configuration (seeds 0,1 in section b). The `walk.*` columns draw no RNG and are bit-stable;
  the `loom` / `loom_escape` / `rotate` / `min_dsi` columns are NOT (section 4: 6-10 Hz within one configuration on an
  independent rerun). The probe halves have 2 seeds (3 baseline); a z on a 2-run null SD is reported by the report
  script but is not a result (section 7 lists the ones to ignore).
* The stage map's ball statistic is diluted (5.2); the object-sweep per-cell max is the sensitive one and is biased
  upward, comparable only against its own null (object_sweep.md 8.8).
* `carries` requires |z| >= 3 in every seed; T2 at +2.7 (apple, baseline) and T4b at +2.6 / +2.9 are below it by the
  rule, not by much -- and by the item above, by less than the run-to-run spread.
* The `.* -> LC4/LPLC2` entry and the T4/T5 x2 interact multiplicatively in `apply_pair_gain` (sequential products);
  `no_pair_gain` removes all five at once and is the only combination run.
* `walk.power_max` values are reported, not interpreted (handover item 4).

## 10. Files

Generators: `scripts/audit_optic.py`, `scripts/probe_figure_stages.py`. Batch logs: `out/optic_audit_cluster.log`
(17 jobs), `out/optic_smoke_cluster.log` (2 jobs), `out/optic_audit_submit.txt`. Results: `out/optic_audit/<config>/`
(`<config>.json`, `stages_s{0,1[,2]}.json`, `obj_{ball,null}_s{0,1[,2]}.json`), `out/optic_audit/<config>.txt` (console),
`out/optic_audit/report.md` / `report.json` (`--report`), `out/optic_audit/summary_compact.txt`, `out/optic_audit/check.json`
(`--check all`), `out/optic_audit/connectivity_structure.json`, `out/optic_audit_smoke/`. Skeptic replication (the
reruns quoted in sections 4-6 and 9, produced with this document's own `scripts/audit_optic.py --one`):
`out/optic_verify_cluster.log` and `out/optic_verify/{baseline/{stages_s0,obj_ball_s0,obj_null_s0}.json,
pair_gain_lpi_x1/pair_gain_lpi_x1.json, no_spk_feedback/obj_{ball,null}_s{0,1}.json}`. Reporting debt for other
owners: `scripts/retire_measures.py:245` (`brain` undefined at import; the script is unrunnable as shipped). Reporting
debt of this document: the LC pooling arithmetic in section 6 ("an LC11 cell pools 94 columns ... 30 % of its optic
input from its best three") has no shipped generator and is the one number here not traceable to a named file.
