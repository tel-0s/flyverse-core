# Receptor model: scoring against the current model (docs/NT_INTEGRATION.md step 6)

Generated 2026-09-12 from two cluster batches (`scripts/cluster_run.py`, run dirs `rm-score-80badf`, 20 jobs, and
`rm-rep-3e214a`, 9 replicate jobs; B200, several jobs per GPU). Every number below is from those runs unless a
NOTES / benchmark_suite.json reference is named. Inputs: `LIFParams.receptor_model` in {None (off), 'sign',
'sign+gain', 'full'}, net rule `class`, no NT-class fallback, default gain classes (low 0.5 / mid 1 / high 1.5),
slow term tau 200 ms, `slow_gain` 0.1 x w_syn. Scripts: `scripts/benchmark.py`, `probe_figure_ground.py`,
`probe_loom.py`, `screen_rotation.py`, `probe_bitter.py`, each with a `--receptor-model` flag added for this study
(`probe_loom.py` also got `--seed`).

## 0. Verdict

**No mode is better than the current model on the suite without breaking a sugar / bitter check; none is adopted
as a default.** Suite (29 checks): off 26 pass / 1 fail / 2 known gaps; **'sign' 25 / 2 / 2** (new fail: the
Shiu-rules sugar response, 124 -> 5.6 Hz, a whole-brain storm under uniform uncapped synapses; the calibrated
sugar / bitter replication is intact at 6.9 / 0.0 Hz); **'sign+gain' 24 / 3 / 2** (walking GF p99 68 Hz, optomotor
group flip -1.7 Hz); **'full' 22 / 5 / 1 + 1 missing** (a slow-term runaway at the default `slow_gain` 0.1: KC 37
Hz, MBONs 330 Hz, no GF response to the loom, 31 spontaneous hops in 15 s; its "closed" object gap is LC10a at 11.5
Hz on both sides). An eager `off` control confirms the 'full' backend change alone changes nothing.

'sign' is the only mode worth keeping as an option: it improves the loom / self-motion separation (demo loom
escapes 3 / 3 seeds at 40-46 Hz vs 2 / 3 at 24-42; walking GF p99 16 vs 20-21 Hz; pinned-loom GF peak 32 vs 21-24
Hz, deterministic) and gives the medulla a real small-object figure (Mi4 z +3.1 +- 0.9 over three replicates vs
+0.0 +- 0.4). It does not deliver on the three step-6 hypotheses, for a reason that is structural rather than
dynamical: the receptor table covers 25.7 % of synapses and none of the direct inputs of Tm5Y, TmY21, LC11, the
GF, DNp04, DNp20, LPT27 / LPT30, MN9 or the sweet interneurons; on the profiled loom / optomotor targets (LC10a,
LPLC2, LPi34, LC4, HSN / HSE) the class rule keeps every glutamatergic input GluCl-dominant, i.e. unchanged. The
'sign' model is a medulla change (Mi4, L3, Mi9, T2a, Tm9 receive 11-59 % of their input flipped Glu -> +) and
everything downstream moves only through those five types. (a) figure-ground at Tm5Y / TmY21 / LC10a: not
recovered (Tm5Y z 0.5-0.7, LC10a best cell +0.2-0.3 mV against 7 mV, rate 0.02 Hz with or without the apple);
(b) GF loom-rate coding: unchanged (final-size coded in both modes), peak +40 %; (c) DNp04 / LPT27 / LPT30: no
flip in any mode (|d'| <= 0.6 in 13 samples).

## 1. Coverage the model actually ran under

`connectome.receptor_signs(c, net_rule='class')` on the cached MaleCNS v1.0 graph (167,106 neurons, 25,578,600
stored entries incl. the 926,233 explicit zeros of sign-0 presynaptic cells), as printed by every probe at start-up:

| tier | edges | edges frac | \|W\| synapses | syn frac |
|---|---|---|---|---|
| exact | 4,266,655 | 16.7 % | 18,015,681 | 14.8 % |
| alias | 427,477 | 1.7 % | 1,446,047 | 1.2 % |
| fuzzy | 2,263,428 | 8.8 % | 8,539,678 | 7.0 % |
| class | 632,949 | 2.5 % | 3,262,654 | 2.7 % |
| nt_class (fallback off) | 0 | 0 | 0 | 0 |
| **matched (table sign used)** | **7,590,509** | **29.7 %** | **31,264,060** | **25.7 %** |
| fallback (NT_SIGN of the presynaptic cell) | 17,710,120 | 69.2 % | 90,163,078 | 74.3 % |
| pre_unknown (nothing to look up) | 277,971 | 1.1 % | 0 (880,886 raw) | 0 (0.7 % raw) |

Denominator: all 124.2 M raw synapses / 121.4 M |W| synapses of the whole CNS. Per postsynaptic module
(`docs/audits/receptor_rules.md` 4b): optic 49.2 % of edges / 52.2 % of |W| synapses matched (exact 34.3 %, fuzzy
14.5 %, alias 0.4 %); visual_projection 36.4 % / 37.6 %; mushroom body 99.3 % / 93.2 % (mostly fuzzy / alias);
antennal lobe 57.5 % / 47.8 % (class-tier pooled priors for 17 % of its edges); central 4.0 % / 4.8 %; gustatory
0.3 % / 0.1 %; descending 3.6 % / 1.7 %; VNC 0 %. Per type, the optic lobe (ol_intrinsic + visual projection +
photoreceptors) has 96 of 628 types matched = 71 % of cells and 50 % of input synapses (exact 46 % of cells);
532 types = 29 % of cells / 50 % of input synapses run under the old rule.

What the matched 25.7 % changes ('sign', whole CNS): 283,093 entries = 1.11 % of entries, 1,188,925 |W| synapses =
0.98 %. 242,949 entries flipped -1 -> +1 (902,325 synapses, all glutamate onto iGluR-dominant targets: 4.26 % of
glutamatergic synapses) and 40,144 silenced -> 0 (286,600 synapses: glutamate 149,489, histamine 116,366 =
photoreceptor / T1 synapses onto profiled targets with no ort / HisCl1 call, GABA 20,745). Nothing else changes: no
+1 -> -1 flips. 'sign+gain' additionally scales every matched entry by its gain class (4,417,302 entries differ
from off after the cap and fan-in normalisation). 'full' adds the slow matrix: 5,238,228 entries (20.5 % of
edges) with a non-zero slow sign, 23.6 M synapse-equivalents (+9.28 M / -14.32 M before gain classes; +9.55 M /
-14.80 M with them), 0.0275 mV of g_slow per synapse-equivalent per presynaptic spike before fan-in scaling.

## 2. Where the sign changes land (structural, CPU; the populations the hypotheses are about)

Per postsynaptic type: input synapses (|W|, uncapped), whether the type has a receptor profile (matched), the
excitatory / inhibitory synapse totals under the present rule (E0 / I0) and under 'sign' (E_sign / I_sign), the
synapses flipped (Glu -1 -> +1) and silenced, and the net (E - I) synapses per cell before and after.

| post | cells | in_syn | matched | E0 | I0 | E_sign | I_sign | flipped | flipped frac | silenced | silenced frac | net/cell off | net/cell sign |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tm5Y | 898 | 406,046 | 0 % | 198,039 | 208,007 | 198,039 | 208,007 | 0 | 0.0 % | 0 | 0.0 % | -11 | -11 |
| TmY21 | 372 | 201,182 | 0 % | 109,771 | 91,411 | 109,771 | 91,411 | 0 | 0.0 % | 0 | 0.0 % | +49 | +49 |
| TmY3 | 824 | 605,342 | 100 % | 313,230 | 292,112 | 313,230 | 292,013 | 0 | 0.0 % | 99 | 0.0 % | +26 | +26 |
| T2 | 1630 | 914,199 | 100 % | 448,059 | 466,140 | 448,059 | 465,428 | 0 | 0.0 % | 712 | 0.1 % | -11 | -11 |
| T2a | 1872 | 583,889 | 100 % | 259,628 | 324,261 | 374,221 | 209,642 | 114,593 | 19.6 % | 26 | 0.0 % | -35 | +88 |
| T3 | 1940 | 465,829 | 100 % | 267,629 | 198,200 | 267,629 | 198,200 | 0 | 0.0 % | 0 | 0.0 % | +36 | +36 |
| Mi4 | 1772 | 535,844 | 100 % | 269,627 | 266,217 | 463,904 | 51,916 | 194,277 | 36.3 % | 20,024 | 3.7 % | +2 | +232 |
| Tm3 | 2054 | 749,349 | 100 % | 364,684 | 384,665 | 364,684 | 384,189 | 0 | 0.0 % | 476 | 0.1 % | -10 | -9 |
| Mi1 | 1773 | 683,213 | 100 % | 206,201 | 477,012 | 206,201 | 463,999 | 0 | 0.0 % | 13,013 | 1.9 % | -153 | -145 |
| Mi9 | 1775 | 480,597 | 100 % | 248,767 | 231,830 | 352,673 | 121,978 | 103,906 | 21.6 % | 5,946 | 1.2 % | +10 | +130 |
| L3 | 1772 | 202,121 | 100 % | 18,572 | 183,549 | 137,118 | 65,003 | 118,546 | 58.7 % | 0 | 0.0 % | -93 | +41 |
| Tm9 | 1771 | 201,993 | 100 % | 108,207 | 93,786 | 131,003 | 70,821 | 22,796 | 11.3 % | 169 | 0.1 % | +8 | +34 |
| Tm1 | 1777 | 668,003 | 100 % | 244,450 | 423,553 | 244,450 | 423,281 | 0 | 0.0 % | 272 | 0.0 % | -101 | -101 |
| Tm2 | 1766 | 651,125 | 100 % | 301,590 | 349,535 | 301,590 | 349,535 | 0 | 0.0 % | 0 | 0.0 % | -27 | -27 |
| Tm4 | 1670 | 722,810 | 100 % | 336,999 | 385,811 | 336,999 | 385,558 | 0 | 0.0 % | 253 | 0.0 % | -29 | -29 |
| Tm20 | 1762 | 369,239 | 100 % | 194,389 | 174,850 | 194,389 | 174,850 | 0 | 0.0 % | 0 | 0.0 % | +11 | +11 |
| LC10a | 275 | 250,329 | 100 % | 138,484 | 111,845 | 138,484 | 111,845 | 0 | 0.0 % | 0 | 0.0 % | +97 | +97 |
| LC10b | 95 | 89,524 | 100 % | 59,656 | 29,868 | 59,656 | 29,868 | 0 | 0.0 % | 0 | 0.0 % | +314 | +314 |
| LC11 | 143 | 342,940 | 0 % | 224,615 | 118,325 | 224,615 | 118,325 | 0 | 0.0 % | 0 | 0.0 % | +743 | +743 |
| LC16 | 182 | 97,184 | 100 % | 65,117 | 32,067 | 65,117 | 32,067 | 0 | 0.0 % | 0 | 0.0 % | +182 | +182 |
| LC4 | 126 | 305,161 | 100 % | 215,307 | 89,854 | 215,307 | 89,854 | 0 | 0.0 % | 0 | 0.0 % | +996 | +996 |
| LPLC2 | 185 | 349,720 | 100 % | 269,820 | 79,900 | 269,820 | 79,900 | 0 | 0.0 % | 0 | 0.0 % | +1027 | +1027 |
| LPi34 | 119 | 213,185 | 100 % | 181,132 | 32,053 | 181,132 | 32,053 | 0 | 0.0 % | 0 | 0.0 % | +1253 | +1253 |
| LPi43 | 61 | 119,692 | 0 % | 93,499 | 26,193 | 93,499 | 26,193 | 0 | 0.0 % | 0 | 0.0 % | +1103 | +1103 |
| T4a | 1684 | 414,117 | 100 % | 221,359 | 192,758 | 221,359 | 192,758 | 0 | 0.0 % | 0 | 0.0 % | +17 | +17 |
| T5a | 1664 | 370,394 | 100 % | 233,270 | 137,124 | 233,270 | 137,124 | 0 | 0.0 % | 0 | 0.0 % | +58 | +58 |
| DNp01 (GF) | 2 | 36,589 | 0 % | 23,468 | 13,121 | 23,468 | 13,121 | 0 | 0.0 % | 0 | 0.0 % | +5174 | +5174 |
| DNp04 | 2 | 20,938 | 0 % | 16,654 | 4,284 | 16,654 | 4,284 | 0 | 0.0 % | 0 | 0.0 % | +6185 | +6185 |
| DNp20 | 2 | 9,513 | 0 % | 6,486 | 3,027 | 6,486 | 3,027 | 0 | 0.0 % | 0 | 0.0 % | +1730 | +1730 |
| HSN | 2 | 32,830 | 100 % | 29,508 | 3,322 | 29,508 | 3,322 | 0 | 0.0 % | 0 | 0.0 % | +13093 | +13093 |
| HSE | 2 | 36,454 | 100 % | 31,320 | 5,134 | 31,320 | 5,134 | 0 | 0.0 % | 0 | 0.0 % | +13093 | +13093 |
| LPT27 | 2 | 21,299 | 0 % | 16,795 | 4,504 | 16,795 | 4,504 | 0 | 0.0 % | 0 | 0.0 % | +6146 | +6146 |
| LPT30 | 2 | 10,181 | 0 % | 6,125 | 4,056 | 6,125 | 4,056 | 0 | 0.0 % | 0 | 0.0 % | +1034 | +1034 |
| DNa02 | 2 | 47,737 | 0 % | 31,673 | 16,064 | 31,673 | 16,064 | 0 | 0.0 % | 0 | 0.0 % | +7804 | +7804 |
| MN9 | 2 | 6,410 | 0 % | 3,225 | 3,185 | 3,225 | 3,185 | 0 | 0.0 % | 0 | 0.0 % | +20 | +20 |
| GNG175 (Usnea) | 2 | 2,560 | 0 % | 2,274 | 286 | 2,274 | 286 | 0 | 0.0 % | 0 | 0.0 % | +994 | +994 |
| GNG132 (Rattle) | 2 | 6,218 | 0 % | 4,255 | 1,963 | 4,255 | 1,963 | 0 | 0.0 % | 0 | 0.0 % | +1146 | +1146 |
| KC (all) | 4064 | 1,818,619 | 100 % | 1,572,611 | 246,008 | 1,581,086 | 237,533 | 8,475 | 0.5 % | 0 | 0.0 % | +326 | +331 |
| EPG | 46 | 194,231 | 100 % | 51,087 | 143,144 | 51,087 | 143,144 | 0 | 0.0 % | 0 | 0.0 % | -2001 | -2001 |

Top changed presynaptic types per target (|W| synapses): Mi4 <- TmY16 42,909, Dm4 39,619, Mi9 35,908, Dm6 22,347,
Dm9 19,766, Dm15 13,744 (all glutamate flips); L3 <- Dm12 109,232; Mi9 <- Dm12 27,967, TmY16 19,272, Dm15 14,082,
Dm20 9,900; T2a <- MeLo10 32,407, Mi2 26,188, TmY5a 10,342, Mi9 6,375; Tm9 <- Dm12 12,144, Mi13 5,037; Mi1 <- R8y /
R8_unclear / R8p 4,276 / 4,190 / 3,832 (histamine silenced: Mi1 has no ort / HisCl1 call); KC <- MBON05 2,755,
MBON06 2,250 (glutamate flips).

So: **none of the direct inputs of Tm5Y, TmY21, LC11, LPi43, the GF, DNp04, DNp20, LPT27 / LPT30, DNa02, MN9 or
the sweet second-order cells is touched** (unprofiled types keep NT_SIGN), and the profiled loom / optomotor targets
(LC10a, LPLC2, LPi34, LC4, T4 / T5, HSN / HSE) keep every sign: their glutamatergic inputs are GluCl-dominant in
the class rule. The receptor model, as built, is a **medulla change**: Mi4 (+2 -> +232 net synapses per cell), L3
(-93 -> +41), Mi9 (+10 -> +130), T2a (-35 -> +88), Tm9 (+8 -> +34). Everything downstream can only change through
those five types.

Slow term of 'full' per cell (synapse-equivalents = count x slow sign x gain class; x 0.0275 mV = g_slow jump per
presynaptic spike before fan-in scaling; the GF, DNs, MN9 and GNG cells are unprofiled and get no slow input):

| post | cells | slow + / cell | slow - / cell | net mV / cell / volley | fast net syn-eq / cell ('sign+gain') | by presynaptic NT |
|---|---|---|---|---|---|---|
| Mi4 | 1772 | +1 | -258 | -7.07 | +226 | ACh -228 (mAChR-B); GABA -29 (GABA-B) |
| L3 | 1772 | +16 | -7 | +0.24 | +32 | ACh +16; GABA -7 |
| Mi9 | 1775 | +1 | -243 | -6.67 | +125 | ACh -140; GABA -103 |
| T2a | 1872 | +0 | -168 | -4.62 | +188 | GABA -168 |
| Tm9 | 1771 | +61 | -20 | +1.13 | +17 | ACh +61; GABA -20 |
| Mi1 | 1773 | +177 | -393 | -5.92 | -144 | Glu -215 (mGluR); GABA -178; ACh +174 |
| Tm3 | 2054 | +91 | -60 | +0.87 | -68 | ACh +89; GABA -60 |
| T4a | 1684 | +1 | -261 | -7.16 | -8 | ACh -197; GABA -64 |
| T5a | 1664 | +0 | -264 | -7.27 | +44 | ACh -210; GABA -54 |
| LPi34 | 119 | +761 | -363 | +10.96 | +1879 | ACh +761; Glu -279; GABA -83 |
| LPLC2 | 185 | +2 | -442 | -12.12 | +1613 | Glu -285; GABA -147; OA -11 |
| LC4 | 126 | +3 | -2587 | -71.06 | +1493 | ACh -1709; GABA -782; Glu -96 |
| LC10a | 275 | +758 | -502 | +7.04 | -106 | ACh +755; GABA -283; Glu -218 |
| HSN / HSE | 2 / 2 | +22144 / +23506 | -1661 / -2567 | +563 / +576 | +13093 | ACh +22131 / +23490 (mAChR-A); GABA -1342 / -1555; DA +10 / +8 |
| DNp01, DNp04, DNp20, DNa02, MN9, GNG175/132/229/232 | 2 each | 0 | 0 | 0 | -- | unprofiled |
| KC (all) | 4064 | +121 | -217 | -2.64 | +452 | GABA -55; ACh -42; DA +2 |
| EPG | 46 | +82 | -5752 | -156 | -3002 | GABA -4045; ACh -1666; OA +44; 5-HT -42; DA +38 |
| PEN_a | 20 | +1001 | -934 | +1.85 | +470 | ACh +936; Glu -636; GABA -297; DA +50 |
| ORN | 2635 | +1 | -150 | -4.10 | +18 | GABA -94; ACh -56 |
| uniglomerular PN | 467 | +756 | -835 | -2.16 | +1513 | ACh +747; GABA -723; Glu -111 |

The slow term is dominated by metabotropic ACh (mAChR-A / -B) and GABA-B, not by the monoamines (DA / OA / 5-HT are
+2 to +50 per cell where present); in the optic lobe the rate model ignores it entirely (no slow term in
`optic.py`), and rate cells never spike, so W_slow entries from frozen optic cells are pruned with W. 'full' therefore
differs from 'sign+gain' only through spiking presynaptic cells.

## 3. Benchmark suite (`scripts/benchmark.py --seeds 0,1,2`, all 14 sections)

One run per mode (Brain seed 0 everywhere; the demo sections are chaotic, so single-run magnitudes carry the
suite's usual scatter -- compare the two `off` columns, native vs eager, for its size). 'full' cannot run on the
native kernels (the slow term is Torch-only and `FlyBrain` refuses CUDA graphs with Torch events), so it ran with
`--eager`; the `off (eager)` column is the control for that backend change. P = pass, F = fail, gap = known gap,
P! = pass (gap closed), miss = section crashed / value None.

| check | off (native) | sign | sign+gain | full (eager) | off (eager control) | criterion |
|---|---|---|---|---|---|---|
| rest.spikes_per_step | 0.00 P | 0.00 P | 0.00 P | 0.00 P | 0.00 P | < 5 |
| taste.MN9_hz | 5.85 P | 7.92 P | 2.34 P | 5.11 P | 5.85 P | > 2 |
| smell.PN_hz | 12.39 P | 5.66 P | 14.29 P | 1.63 P | 12.39 P | < 100 |
| smell.KC_active | 1599 P | 1502 P | 2292 P | 597 P | 1599 P | > 0 |
| dn.DNa02_L_leg_asym_hz | 2.58 P | 2.58 P | 2.58 P | 2.58 P | 2.58 P | > 0.3 |
| dn.MDN_top_hz | 152 P | 152 P | 152 P | 293 F | 152 P | < 250 |
| dn.DNp09_top_hz | 152 P | 152 P | 152 P | 152 P | 152 P | < 250 |
| walk.GF_max_hz | 8.51 P | 4.99 P | 4.63 P | 4.58 P | 8.51 P | < 38 |
| walk.power_max_hz | 79.06 F | 90.91 F | 57.64 F | 41.88 P | 79.06 F | < 50 |
| walk.power_sustained_hz | 37.41 P | 38.86 P | 26.55 P | 19.68 P | 37.41 P | < 50 |
| loom.GF_peak_hz | 30.80 P | 30.99 P | 32.36 P | 13.67 F | 30.80 P | >= 20 |
| loom.escape_cm | 3.50 P | 3.50 P | 3.50 P | -- miss | 3.50 P | notnone |
| rotate.DNp20_flip_hz | -20.41 P | -20.47 P | -30.02 P | -44.94 P | -16.17 P | < -2 |
| motion.min_dsi | 0.17 P | 0.17 P | 0.18 P | 0.14 P | 0.17 P | >= 0.1 |
| motion.correct_directions | 8 P | 8 P | 8 P | 8 P | 8 P | == 8 |
| loom_escape.GF_peak_hz | 41.66 P | 46.36 P | 65.14 P | 0.00 F | 36.54 P | >= 33 |
| loom_escape.escapes (of 3 seeds) | 2 P | 3 P | 2 P | 2 P | 1 P | >= 1 |
| walk_gf.p99_hz | 20.63 P | 16.32 P | 68.27 F | 6.22 P | 20.13 P | < 38 |
| rotation.group_flip_hz | -6.65 P | -7.62 P | -1.73 F | -2.67 F | -8.06 P | <= -3 |
| object.LC10a_flip_hz | -0.00 gap | -0.00 gap | -0.04 gap | 5.54 P! | 0.00 gap | abs >= 1.0 |
| bitter.calibrated_sugar_MN9_hz | 4.57 P | 6.91 P | 5.91 P | 5.76 P | 4.57 P | > 2 |
| bitter.calibrated_sugar_bitter_MN9_hz | 0.00 P | 0.00 P | 0.00 P | 0.00 P | 0.00 P | < 1 |
| bitter.shiu_sugar_MN9_hz | 124 P | 5.57 F | 123 P | 74.18 P | 124 P | > 50 |
| bitter.shiu_sugar_bitter_MN9_hz | 2.12 P | 0.88 P | 2.19 P | 1.81 P | 2.12 P | < 10 |
| wind.DNp18_flip_hz | 45.49 P | 46.61 P | 44.74 P | 4.62 F | 43.67 P | >= 15 |
| wind.DNp33_flip_hz | -50.68 P | -50.67 P | -49.89 P | -46.26 P | -49.48 P | <= -15 |
| odour.apple_channel_8cm_hz | 17.37 P | 17.14 P | 15.07 P | 17.25 P | 17.41 P | >= 10 |
| odour.apple_channel_clean_hz | 4.38 P | 4.59 P | 4.23 P | 4.28 P | 4.67 P | <= 6 |
| compass.wedge_cells_persisting | 0 gap | 0 gap | 0 gap | 0 gap | 0 gap | >= 6 |

| mode | pass | fail | known gap | missing | runtime (min, shared B200) | backend |
|---|---|---|---|---|---|---|
| off | 26 | 1 | 2 | 0 | 9.6 | native |
| sign | 25 | 2 | 2 | 0 | 10.6 | native |
| sign+gain | 24 | 3 | 2 | 0 | 10.8 | native |
| full | 22 | 5 | 1 | 1 | 13.3 | eager |
| off (eager control) | 26 | 1 | 2 | 0 | 11.4 | eager |

Status changes against `off` (native):

| mode | check | off | mode |
|---|---|---|---|
| sign | bitter.shiu_sugar_MN9_hz | 124 PASS | 5.57 FAIL |
| sign+gain | walk_gf.p99_hz | 20.63 PASS | 68.27 FAIL |
| sign+gain | rotation.group_flip_hz | -6.65 PASS | -1.73 FAIL |
| full | dn.MDN_top_hz | 152 PASS | 293 FAIL (MBON13 / PPL104 / MBON03 at ~290 Hz under MDN drive) |
| full | walk.power_max_hz | 79.06 FAIL | 41.88 PASS |
| full | loom.GF_peak_hz | 30.80 PASS | 13.67 FAIL |
| full | loom.escape_cm | 3.50 PASS | MISSING (GF never reached 20 Hz) |
| full | loom_escape.GF_peak_hz | 41.66 PASS | 0.00 FAIL |
| full | rotation.group_flip_hz | -6.65 PASS | -2.67 FAIL |
| full | object.LC10a_flip_hz | KNOWN GAP | 5.54 PASS (gap closed) -- spurious, see below |
| full | wind.DNp18_flip_hz | 45.49 PASS | 4.62 FAIL |

Section detail behind the table:

| mode | loom_escape: walking GF max / hops before loom / loom peak / escape, per seed | walk_gf median / p90 / p99 / max Hz (voluntary takeoffs in 15 s) | rotation per-type flip Hz | object flip Hz (LC10a rate Hz) | smell PN / KC Hz (KC active) | taste MN9 |
|---|---|---|---|---|---|---|
| off | 17 / 0 / 34 / yes; 18 / 0 / 42 / yes; 12 / 0 / 24 / no | 13 / 18 / 21 / 21 (0) | DNp20 -9.3 HSN -8.0 HSE -2.7 DNp04 +0.9 LPT27 +0.0 LPT30 +0.2 | LC10a -0.00 LC10b +0.01 LC16 +0.24 (0.02) | 12.4 / 2.9 (1599) | 5.8 |
| sign | 16 / 0 / 46 / yes; 15 / 0 / 45 / yes; 17 / 0 / 40 / yes | 9 / 15 / 16 / 16 (0) | DNp20 -13.7 HSN -4.9 HSE -4.3 DNp04 +0.5 LPT27 -0.9 LPT30 +0.2 | LC10a -0.00 LC10b +0.25 LC16 +0.43 (0.02) | 5.7 / 0.9 (1502) | 7.9 |
| sign+gain | 84 / 3 / 48 / yes; 96 / 3 / 49 / yes; 65 / 3 / 65 / no | 45 / 56 / 68 / 70 (0) | DNp20 -3.8 HSN -0.3 HSE -1.0 DNp04 +4.4 LPT27 +0.6 LPT30 -1.2 | LC10a -0.04 LC10b -0.26 LC16 +0.58 (0.08) | 14.3 / 5.8 (2292) | 2.3 |
| full | 7 / 5 / 0 / "yes"; 12 / 1 / 0 / no; 13 / 5 / 0 / "yes" | 0 / 2 / 6 / 7 (31) | DNp20 -5.0 HSN -2.0 HSE -1.0 DNp04 +0.0 LPT27 +1.5 LPT30 -0.4 | LC10a +5.54 LC10b +1.43 LC16 +0.88 DNa02 -1.19 (11.46) | 1.6 / 37.3 (597) | 5.1 |
| off (eager) | 17 / 0 / 37 / yes; 12 / 0 / 33 / no; 14 / 0 / 33 / no | 13 / 17 / 20 / 21 (0) | DNp20 -12.6 HSN -8.1 HSE -3.4 DNp04 +0.4 LPT27 +0.1 LPT30 -1.1 | LC10a +0.00 LC10b +0.04 LC16 +0.25 (0.02) | 12.4 / 2.9 (1599) | 5.8 |

Reading:

* **sign**: the only change of status is the Shiu-rules sugar response (uniform 0.275 mV synapses, no cap / fan-in
  / adaptation): 124 -> 5.6 Hz, reproduced by the standalone probe (5.6 Hz, 657 spikes/step in the sugar condition
  against 114 under off) -- under uncapped uniform synapses the 902 k flipped glutamate synapses tip the whole brain
  into a storm that shuts MN9. The calibrated replication is intact (6.9 / 0.0 Hz). Two magnitudes move the right way
  and beyond the suite's scatter: the demo loom escapes 3 / 3 seeds with peaks 40-46 Hz (off 2 / 3 at 24-42, eager
  control 1 / 3 at 33-37) while the walking GF falls (per-second p99 16 vs 20-21, max 16 vs 21). The antennal lobe
  runs cooler (PN 5.7 vs 12.4 Hz, LN 25 vs 62; 38,937 glutamatergic synapses onto PNs flip to +, mostly onto the
  l2PN / lv2PN types, which changes the LN-PN balance). Nothing about the optomotor readout changes (section 6).
* **sign+gain**: a self-motion storm -- walking GF per-second maxima 45 / 56 / 68 / 70 Hz (off 13 / 18 / 21 / 21),
  3 spontaneous hops per seed before the loom, and the DNp20 + HSN + HSE flip drops to -1.7 Hz. The gain classes
  (low 0.5 / mid 1 / high 1.5 by expression tertile) are the culprit: they are applied before the cap and fan-in
  normalisation, so a 'high' class on a few strong optic pairs is a x1.5 that the normalisation redistributes.
  Worse than off on every count that matters.
* **full** (default `slow_gain` 0.1 x w_syn, tau 200 ms): a slow runaway. The metabotropic ACh (mAChR-A, +) and
  GABA-B / mGluR (-) terms dominate the slow matrix (section 2); with a 200 ms time constant every recurrently
  connected cholinergic population integrates itself: smell -> KC 37 Hz (597 active) and MBON / DPM at 330 Hz, MDN
  drive -> MBON13 / PPL104 at 290 Hz, the walking fly hops 31 times in 15 s (voluntary takeoffs; the "escapes" in
  loom_escape are such hops, the loom GF peak is 0), the GF never crosses 20 Hz in the legacy loom, the wind DNp18
  flip collapses (45 -> 4.6 Hz), and the object "gap closed" is LC10a firing 11.5 Hz on both sides (+11.1 / +5.5 Hz
  L-R) -- a hot brain, not a small-object signal. The slow term needs a scale sweep (0.01-0.03 x w_syn) and probably
  per-class time constants before it is scoreable; at 0.1 it is not a candidate.
* The `off` native / eager control pair shows the suite's own scatter: loom_escape 2 / 3 vs 1 / 3, peaks 24-42 vs
  33-37, rotate.DNp20 -20 vs -16, wind DNp18 45 vs 44. Differences of that size between modes mean nothing.

## 4. Figure-ground (`probe_figure_ground.py`: apple 5 cm ahead-left, heading +-20 deg, apple minus none, apple columns minus background, z vs background scatter)

Three replicates for off and 'sign', two for 'sign+gain' (the probe has no seed: replicates sample the native
backend's run-to-run nondeterminism, which is the relevant scatter). 159 columns view the apple, 1,050 are
background; 89,380 of 89,390 rate cells have a column. z per replicate (mean); `figure` is in rate units.

| type | off z per replicate (mean) | sign | sign+gain | reads |
|---|---|---|---|---|
| Mi4 | +0.4 / +0.1 / -0.4 (+0.0) | +3.3 / +3.9 / +2.1 (+3.1) | +2.2 / +1.7 (+2.0) | **gained**: figure +0.0003 -> +0.0246 (apple columns 0.030 vs none 0.008: a 4x modulation); 36 % of Mi4's input flipped |
| Mi9 | +2.1 / +1.7 / +2.6 (+2.1) | +3.3 / +3.7 / +1.2 (+2.7) | +1.9 / +1.3 (+1.6) | figure +0.002 -> +0.020; within scatter on z |
| L3 | +1.0 / +0.9 / +0.9 (+0.9) | +2.7 / +2.5 / +1.2 (+2.1) | +2.8 / +2.2 (+2.5) | gained (figure +0.004 -> +0.052); 59 % of L3's input flipped (Dm12) |
| Tm9 | +0.1 / +0.1 / +0.1 (+0.1) | +2.4 / +3.0 / +1.1 (+2.2) | +1.7 / +1.2 (+1.5) | gained (+0.0002 -> +0.019) |
| T2a | -0.2 / -0.3 / -0.5 (-0.3) | +1.2 / +2.1 / +2.1 (+1.8) | +0.8 / +1.8 (+1.3) | gained, small (+0.014) |
| T2 | +0.8 / +0.5 / +0.8 (+0.7) | +2.2 / +1.9 / +1.7 (+1.9) | +0.2 / +1.2 (+0.7) | gained under sign (+0.0013 -> +0.014); T2's own inputs are unchanged, it inherits Mi4 / Mi9 |
| T3 | -1.3 / -1.0 / -0.9 (-1.0) | -0.3 / -1.0 / -1.2 (-0.9) | +0.0 / -0.0 (-0.0) | unchanged |
| **Tm5Y** | -0.2 / -0.2 / +0.0 (-0.1) | +0.5 / +0.7 / +0.7 (+0.6) | +0.2 / +0.3 (+0.3) | consistent but tiny: figure -0.0005 -> +0.0035 rate units; z < 1 in every run |
| **TmY21** | -0.6 / -0.6 / -0.6 (-0.6) | -0.0 / -0.5 / -0.6 (-0.4) | -0.2 / -0.4 (-0.3) | unchanged (no profile; its inputs Tm20 / TmY5a / TmY13 are unchanged) |
| TmY3 | +0.5 / +0.6 / -0.3 (+0.3) | -1.3 / -1.4 / -0.6 (-1.1) | -1.0 / -1.2 (-1.1) | sign reversed, small |
| Mi1 | -10.9 / -7.7 / -8.0 (-8.9) | -1.0 / -0.8 / -1.8 (-1.2) | -5.8 / -3.1 (-4.5) | **lost**: the ON-channel figure (Mi1 has no ort / HisCl1 call, so its 13 k R8 histamine synapses are silenced and its L1 drive is renormalised) |
| Tm3 | -6.4 / -6.2 / -5.5 (-6.0) | -6.1 / -5.8 / -5.0 (-5.6) | -7.2 / -4.9 (-6.1) | unchanged |
| Tm1 | +4.8 / +4.1 / +4.0 (+4.3) | -0.1 / -0.5 / +0.6 (-0.0) | +2.2 / +1.9 (+2.0) | **lost** under sign (its own inputs are unchanged; the L2 / Mi1 balance upstream moved) |
| Tm2 | +3.5 / +3.2 / +2.9 (+3.2) | +3.7 / +3.0 / +2.9 (+3.2) | +1.5 / +1.8 (+1.6) | unchanged |
| Tm4 | +4.5 / +5.1 / +3.7 (+4.4) | +3.2 / +2.7 / +3.0 (+3.0) | +4.3 / +3.2 (+3.7) | slightly weaker |
| L1 | +19.2 / +16.7 / +16.3 (+17.4) | +9.8 / +8.7 / +13.2 (+10.6) | +48.7 / +35.6 (+42.2) | the figure itself (+0.016) is unchanged; z moves because the background scatter changes |
| L2 | +11.5 / +11.6 / +10.2 (+11.1) | +8.0 / +7.1 / +7.7 (+7.6) | +19.4 / +14.1 (+16.8) | same |
| T1 | +4.8 / +4.1 / +4.2 (+4.4) | +4.1 / +3.3 / +4.5 (+3.9) | +11.4 / +7.7 (+9.6) | same |
| Tm20 | +4.1 / +3.3 / +3.4 (+3.6) | +3.2 / +2.8 / +1.6 (+2.6) | +4.0 / +2.9 (+3.5) | unchanged |

Spiking targets: population mean |drive| (mV, apple / none) and per-cell (apple - none) drive median / 90th
percentile / max, per replicate. LC10a's threshold is ~7 mV.

| mode | replicate | LC10a pop. | LC10a per cell | LC10b per cell | LC4 pop. | LPLC2 pop. | LC16 per cell |
|---|---|---|---|---|---|---|---|
| off | 1 | -0.00 / -0.01 | +0.012 / +0.04 / **+0.07** | +0.02 / +0.11 / +0.52 | -0.27 / -0.29 | 1.53 / 1.41 | -0.01 / +0.04 / +1.21 |
| off | 2 | +0.02 / -0.02 | +0.035 / +0.08 / +0.13 | -0.05 / +0.07 / +0.47 | -0.24 / -0.27 | 1.53 / 1.41 | +0.01 / +0.08 / +1.24 |
| off | 3 | -0.01 / -0.00 | -0.006 / +0.03 / +0.09 | +0.04 / +0.14 / +0.52 | -0.25 / -0.28 | 1.53 / 1.41 | -0.02 / +0.04 / +1.25 |
| sign | 1 | -0.06 / -0.01 | -0.045 / +0.07 / **+0.23** | +0.28 / +0.63 / +1.01 | -0.73 / -0.74 | 1.74 / 1.49 | -0.06 / +0.22 / +1.51 |
| sign | 2 | -0.01 / -0.03 | +0.008 / +0.13 / +0.29 | -0.01 / +0.63 / +1.02 | -0.79 / -0.74 | 1.72 / 1.45 | +0.03 / +0.38 / +1.48 |
| sign | 3 | -0.03 / -0.04 | +0.006 / +0.13 / +0.27 | +0.17 / +0.62 / +1.00 | -0.79 / -0.76 | 1.69 / 1.50 | +0.00 / +0.39 / +1.38 |
| sign+gain | 1 | -0.05 / +0.02 | -0.077 / +0.06 / +0.19 | +0.10 / +0.31 / +0.53 | -0.96 / -1.08 | 1.16 / 1.11 | -0.18 / +0.16 / +1.36 |
| sign+gain | 2 | -0.07 / +0.00 | -0.064 / +0.06 / +0.21 | +0.14 / +0.48 / +0.81 | -1.15 / -1.15 | 1.12 / 1.06 | -0.23 / +0.17 / +1.49 |

Under 'sign' the medulla's figure moves from the ON / OFF contrast channels (Mi1, Tm1 lose it) to the glutamate-fed
columnar types (Mi4, Mi9, Tm9, L3, T2a gain it), and the drive that reaches the object detectors grows 2-4x (LC10a
per-cell max 0.07-0.13 -> 0.23-0.29 mV, LC10b 90th percentile 0.07-0.14 -> 0.62 mV) -- but that is 25-30x short
of LC10a's threshold, LC10a's rate stays 0.02 Hz with or without the apple in the benchmark's object section, and
Tm5Y / TmY21 (LC10a's input types, both unprofiled) carry a figure of at most +0.0035 rate units (z 0.5-0.7). The
inference of NOTES session 9 stands: the object pathway is missing between the medulla and Tm5Y / TmY21, and the
receptor table does not reach that stage (Tm5Y, TmY21, LC11 have no expression profile in any of the four sources).

## 5. Loom (`probe_loom.py`: black 3 cm ball from the left at 1 m/s after 1.5 s of walking; Brain + OpticLobe, eager, LPi x4 default)

Three Brain seeds per mode. The probe has no Poisson input, so the seed changes nothing in the drive: the three
runs of a mode are near-identical (GF peak 32 / 32 / 32 Hz under 'sign', 21 / 24 / 24 under off; the small
differences are backend nondeterminism, not the model's noise). Peaks are the maxima over the 100 ms reports.

| mode | seed | GF walking max | GF loom peak (Hz) | LPLC2 pk | LC4 pk | LC16 pk | DNp04 pk | DNp06 pk | TTMn pk | OL \|LPi34\| pk | OL \|T4a\| pk | OL \|T5a\| pk | spikes/step pk | escape (gf_hz 33) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 0 | 0 | 21 | 3 | 0 | 1 | 5 | 13 | 16 | 0.25 | 0.04 | 0.06 | 30 | no |
| off | 1 | 0 | 24 | 3 | 0 | 1 | 6 | 10 | 15 | 0.24 | 0.03 | 0.07 | 31 | no |
| off | 2 | 0 | 24 | 3 | 0 | 1 | 6 | 10 | 15 | 0.24 | 0.03 | 0.07 | 28 | no |
| sign | 0 | 0 | 32 | 4 | 0 | 1 | 8 | 11 | 18 | 0.33 | 0.12 | 0.09 | 63 | no |
| sign | 1 | 0 | 32 | 4 | 0 | 1 | 8 | 9 | 18 | 0.33 | 0.12 | 0.09 | 63 | no |
| sign | 2 | 0 | 32 | 4 | 0 | 1 | 8 | 11 | 18 | 0.33 | 0.12 | 0.09 | 63 | no |

GF (DNp01) time course, seed 0, Hz at each report (t after loom start / distance):

| mode | 0.1 s / 41 cm | 0.1 s / 40 cm | 0.2 s / 31 cm | 0.3 s / 21 cm | 0.4 s / 11 cm | 0.5 s / 3 cm | 0.6 s / 3 cm | 0.7 s / 3 cm | 0.8 s / 3 cm |
|---|---|---|---|---|---|---|---|---|---|
| off | 0 | 0 | 0 | 0 | 0 | 20 | 21 | 8 | 6 |
| sign | 2 | 2 | 1 | 0 | 0 | 20 | 32 | 17 | 6 |

The GF still fires only when the ball has reached its final size (3 cm, t >= 0.5 s) in both modes; 'sign' raises
the peak by 8-11 Hz (21-24 -> 32; LPLC2 3 -> 4 Hz, DNp04 5-6 -> 8), with a 2 Hz GF blip at loom onset and a
brain-wide activity rise during walking (spikes/step 1-11 -> 19-70 in the first second; |T4a| optic modulation
0.03 -> 0.12). The ordering (LC4-side final-size coding, no expansion-rate response) is unchanged; the escape stays
below `Flight.gf_hz` 33 in this eager probe in both modes (32 vs 33 is inside the batch-1 benchmark's own scatter,
see section 3).

## 6. Optomotor rotation (`screen_rotation.py`: pinned, +-90 deg/s for 10 s each, L - R flip = (L-R)_ccw - (L-R)_cw)

Two GPU replicates per mode from `screen_rotation.py` (a third `off` run of batch 1 initialised without CUDA and
was cancelled), plus the benchmark's rotation section (same protocol, same numbers reported as `bench`) and the
session-8 reference. d' = flip / pooled within-condition s.d.; a course-stabilising readout needs |d'| >~ 1.

| type | off r2 | off r3 | off bench | off session 8 | sign r1 | sign r2 | sign bench | sign+gain bench | full bench |
|---|---|---|---|---|---|---|---|---|---|
| DNp20 | -7.3 (d' -1.8) | -11.0 (-2.8) | -9.3 | -12.3 (-3.1) | -11.1 (-1.4) | -13.5 (-1.8) | -13.7 | -3.8 | -5.0 |
| HSN | -7.3 (-3.8) | -8.1 (-4.2) | -8.0 | -7.1 (-4.1) | -4.3 (-1.8) | -5.4 (-2.2) | -4.9 | -0.3 | -2.0 |
| HSE | -2.9 (-1.8) | -3.3 (-2.4) | -2.7 | -3.3 (-2.2) | -4.2 (-1.9) | -4.5 (-1.9) | -4.3 | -1.0 | -1.0 |
| group DNp20+HSN+HSE | -17.5 | -22.5 | -6.7 (smoothed) | -- | -19.6 | -23.4 | -7.6 (smoothed) | -1.7 | -2.7 |
| **DNp04** | +0.4 (+0.2) | +0.2 (+0.1) | +0.9 | -0.7 (-0.3) | +0.5 (+0.3) | +1.1 (+0.6) | +0.5 | +4.4 | +0.0 |
| **LPT27** | +0.4 (+0.2) | -0.2 (-0.1) | +0.0 | +0.4 (+0.1) | -0.3 (-0.1) | -1.1 (-0.2) | -0.9 | +0.6 | +1.5 |
| **LPT30** | +0.7 (+0.4) | -0.8 (-0.4) | +0.2 | +0.9 (+0.5) | -0.5 (-0.2) | -0.5 (-0.1) | +0.2 | -1.2 | -0.4 |
| DNa02 | +0.0 | +0.1 | -- | 0.0 | +0.1 | +0.2 | -- | -0.4 | -1.2 |
| rest rate Hz (DNp20 / HSN / HSE / DNp04 / LPT27 / LPT30) | 16 / 2.2 / 0.9 / 0.9 / 1.7 / 2.2 | 15 / 2.4 / 1.0 / 0.8 / 1.5 / 2.1 | | 16 / 2.2 / 0.7 / 1.3 / 1.7 / 2.1 | 22 / 2.3 / 2.8 / 0.7 / 4.0 / 3.5 | 23 / 2.4 / 2.9 / 0.8 / 4.1 / 3.3 | | | |

(`group` from the CSV is the plain sum of the three flips; the benchmark's group value is the 1 s-smoothed
`opto_L - opto_R` readout, hence the smaller number.)

**DNp04, LPT27 and LPT30 do not flip in any mode** (|d'| <= 0.6 in all 13 samples; the +4.4 Hz DNp04 value under
'sign+gain' is one sample from a run whose walking GF was in a storm, and its d' was not computed there). This is
what the structural table predicts: none of the three has a receptor profile, so none of their inputs changed. The
existing readout DNp20 + HSN + HSE keeps its flip under 'sign' (DNp20 -11 to -14 Hz, HSE -4.2 to -4.5, HSN weaker
at -4.3 to -5.4 vs -7.1 to -8.1; LPT27 / LPT30 / HSE / DNp20 rest rates rise 1.5-3x) and loses it under 'sign+gain'
and 'full'. The top-ranked flippers under 'sign' are the same population as under off (HSN, HSE, DNp20, DNp15,
LPT26, Nod1 / Nod4, LPT50 with the opposite sign) with H2 (+1.9 / +1.6 d') joining under 'sign'.

## 7. Sugar / bitter (`probe_bitter.py`: labellar sweet 165 GRNs, bitter 47, 100 Hz, 1.5 s; MN9 mean Hz)

Brain seed 0 throughout (the probe is Poisson-driven, so a seed is a real replicate here; only one was run). The
`bitter_off.txt` of batch 1 initialised without CUDA and was cancelled; `off` is the batch-2 rerun, plus the
benchmark's own bitter section and the session-8 / 9 references. Spikes/step of the whole brain in brackets.

| mode | source | calibrated: sugar | sugar + bitter | bitter alone | Shiu rules: sugar | sugar + bitter | bitter alone |
|---|---|---|---|---|---|---|---|
| off | probe (batch 2) | 4.6 (14) | 0.0 (25) | 0.0 (8) | 123.5 (114) | 2.1 (113) | 0.0 (100) |
| off | benchmark f | 4.57 | 0.00 | -- | 123.5 | 2.12 | -- |
| off | references | 3.5-5.8 (sessions 8-9) | 0.0 | 0.0 | 97.6-123.5 | 1.1-2.1 | 0.0 |
| sign | probe | 6.9 (25) | 0.0 (20) | 0.0 (10) | **5.6 (657)** | 0.9 (122) | 0.0 (122) |
| sign | benchmark f | 6.91 | 0.00 | -- | **5.57** | 0.88 | -- |
| sign+gain | probe | 5.9 (20) | 0.0 (24) | 0.0 (32) | 122.5 (90) | 2.2 (189) | 0.0 (877) |
| sign+gain | benchmark f | 5.91 | 0.00 | -- | 123 | 2.19 | -- |
| full | probe | 5.8 (25) | 0.0 (25) | 0.0 (11) | 74.2 (440) | 1.8 (256) | 0.0 (292) |
| full | benchmark f | 5.76 | 0.00 | -- | 74.2 | 1.81 | -- |

The calibrated replication (sugar -> MN9 fires, bitter shuts it, bitter alone nothing) holds in every mode; the
sweet pathway itself is untouched by the table (9 flipped synapses in the whole gustatory module, none onto MN9 or
the Usnea / Rattle / Phantom / G2N-1 interneurons; MN9 and the GNG types are unprofiled), and the 4.6 -> 6.9 Hz
rise under 'sign' is the same network-level warming seen in `taste.MN9_hz` (5.85 -> 7.92). The **Shiu-rules
replication breaks under 'sign'** (124 -> 5.6 Hz, twice): with uniform uncapped synapses and no fan-in
normalisation the flipped glutamate turns the sugar response into a brain-wide storm (657 spikes/step against
114) and MN9 is shut by it; 'sign+gain' (same flips, gain classes 0.5 / 1 / 1.5) does not storm (90 spikes/step,
122.5 Hz) and 'full' sits between (440 spikes/step, 74 Hz), so the Shiu-rules outcome is a knife-edge of total
excitation in the uncalibrated model, not a gustatory-pathway effect. It is nevertheless one of the suite's
acceptance criteria, and 'sign' fails it.

## 8. Verdicts on the step-6 hypotheses

**(a) Glutamatergic optic types signed per target change the figure-ground signal for Tm5Y / TmY21 / LC10 and the
LPi -> LPLC2 balance.** *Partly, in the wrong place.* The per-target glutamate sign does move the medulla's static
figure: Mi4 z +0.0 -> +3.1 (3 / 3 replicates; a 4x modulation in its apple columns), Mi9, Tm9, L3, T2a and T2 gain,
Mi1 and Tm1 lose theirs. At the stage NOTES 9 identified as the loss -- Tm5Y / TmY21 -> LC10a -- nothing changes
that matters: Tm5Y's figure goes from -0.0005 to +0.0035 rate units (z 0.5-0.7), TmY21 stays at z -0.4, LC10a's
best cell gains 0.16-0.2 mV against a 7 mV threshold, its rate stays 0.02 Hz with and without the apple, and the
suite's object check stays a KNOWN GAP under 'sign' and 'sign+gain' (the 'full' "pass" is a hot-brain artefact).
The reason is coverage, not sign: Tm5Y, TmY21 and LC11 have no expression profile, LC10a's own inputs keep every
sign, and the table changes 0 of Tm5Y's / TmY21's input synapses. The LPi -> LPLC2 balance is untouched (LPi34 and
LPLC2 are profiled, their glutamatergic inputs stay GluCl-dominant -> -1; LPi43 is unprofiled); the x4 pair gain
remains a stop-gap the data neither reproduces nor contradicts.

**(b) Loom-rate coding of the GF.** *No change in what the GF encodes; a quantitative improvement in the escape /
self-motion separation under 'sign'.* In the pinned probe the GF still fires only when the ball reaches its final
size (t >= 0.5 s / 3 cm) in both modes -- LC4-side final-size coding, no expansion-rate response -- and the peak
rises 21-24 -> 32 Hz (3 / 3, deterministic probe) with LPLC2 3 -> 4 Hz and DNp04 5-6 -> 8. In the demo the
'sign' loom escapes 3 / 3 seeds at 40-46 Hz (off 2 / 3 at 24-42; eager control 1 / 3 at 33-37) while the walking
GF falls (p99 16 vs 20-21 Hz, no spontaneous hops), i.e. the same direction as the LPi x4 change of session 9,
achieved without a hand gain. Under 'sign+gain' the walking GF storms (p99 68) and under 'full' the GF never fires
to the loom. Single seeds per configuration; the bistability of `loom_escape` at the 33 Hz threshold (NOTES 9
addendum 2) applies, so 3 / 3 vs 2 / 3 is suggestive, not established.

**(c) The optomotor DNp04 / LPT27 / LPT30 readout.** *No.* None of the three flips in any mode (|d'| <= 0.6 in 13
samples: two replicates and the benchmark section per mode, plus session 8). All three are unprofiled, so the
receptor table cannot change their inputs; the profiled optomotor cells (HSN / HSE) keep every sign. The existing
DNp20 + HSN + HSE readout survives 'sign' (DNp20 -11 to -14 Hz, HSE -4.2 to -4.5, HSN weaker at -4.3 to -5.4) and is
lost under 'sign+gain' (-1.7 Hz group flip) and 'full' (-2.7).

**Sugar / bitter.** Calibrated replication intact in every mode (sugar 4.6-6.9 Hz, sugar + bitter 0.0, bitter
0.0). Shiu-rules replication intact under off / 'sign+gain' (123 Hz -> 2.2), degraded under 'full' (74 -> 1.8),
**broken under 'sign'** (5.6 -> 0.9: a whole-brain storm at 657 spikes/step under uniform uncapped synapses).

## 9. Caveats and infrastructure notes

* **Coverage is the limiting factor for every hypothesis.** 25.7 % of |W| synapses are under the table's rule, and
  the populations the hypotheses name are almost all in the other 74 %: Tm5Y, TmY21, LC11, LPi43, DNp01, DNp04,
  DNp20, LPT27, LPT30, DNa02, MN9 and the sweet interneurons have no profile in Özel 2021, Davis 2020, FCA 2022 or
  Davie 2018 at any tier. The `nt_class_fallback` (Davis ChAT / Gad1 / VGlut class baselines, 71.7 % of synapses)
  was not scored here; it would give those types the class-average receptor set, which for glutamate onto
  cholinergic targets is again GluCl-dominant (-1), so it is unlikely to change (a)-(c) either.
* **The class rule keeps glutamate inhibitory on every profiled loom / optomotor target**; the `abs` rule (GluCl
  must exceed iGluR 2-fold in summed level) and `nonmda` (iGluR without NMDA subunits) were not scored. Given that
  GluCl is ~10x iGluR in every Davis 2020 optic type (`receptor_rules.md` section 3), `abs` would flip fewer edges,
  not more.
* **Single runs.** The benchmark ran once per mode (3 loom seeds inside it); figure-ground and rotation have 2-3
  replicates; the loom probe is deterministic; bitter has one Poisson seed. The native / eager `off` pair bounds the
  suite's scatter (loom_escape 2 / 3 vs 1 / 3, DNp20 -20 vs -16 Hz). Claims above that rest on single runs are
  marked as such.
* **'sign+gain' and 'full' fail for parameter, not data, reasons.** The gain classes (0.5 / 1 / 1.5) and the slow
  scale (0.1 x w_syn, tau 200 ms) are the plan's "parameters to be swept, documented as such"; both were run at
  their first guess. 'sign+gain' storms the walking GF, 'full' runs the mushroom body and the LC / DN populations
  hot through the metabotropic ACh / GABA-B / mGluR terms (the monoamines contribute +2 to +50 synapse-equivalents
  per cell, the metabotropic classical transmitters hundreds to tens of thousands). A slow-term sweep should start
  at 0.01-0.03 x w_syn with the classical metabotropic classes separated from the monoamine ones, and be scored on
  `rest`, `smell.KC`, `walk_gf` and `loom_escape` first.
* **'full' forces the Torch path** (no native kernel for g_slow) and refuses CUDA graphs with Torch events, so its
  benchmark ran `--eager`; the eager `off` control shows the backend alone changes no status.
* **Infrastructure.** Two of the six jobs packed onto GPU 2 in batch 1 (`rot_off`, `bitter_off`) initialised with
  `torch.cuda.is_available() == False` although the scheduler had assigned the GPU, and ran on CPU; they were
  cancelled (cancel does not kill the process; killed by PID) and rerun in batch 2. Worth a check on the node agent's
  `CUDA_VISIBLE_DEVICES` handling when many budgeted jobs start within the same second. Everything else: 27 jobs,
  exit 0. The `out/bitter_off.txt` and `out/rot_off.txt` fetched from batch 1 are the truncated CPU logs; the
  numbers above use `bitter_off_r2`, `rot_off_r2`, `rot_off_r3`.
* Files: `out/rm_{off,sign,signgain,full,off_eager}.{json,txt}`, `out/fg_{off,sign,signgain}[_r2|_r3].{csv,txt}`,
  `out/loom_{off,sign}_s{0,1,2}.txt`, `out/rot_{off_r2,off_r3,sign,sign_r2}.{csv,txt}`,
  `out/bitter_{off_r2,sign,signgain,full}.txt` (git-ignored). Cluster run dirs `rm-score-80badf`, `rm-rep-3e214a`.
