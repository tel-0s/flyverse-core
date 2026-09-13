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
Hz on both sides). An eager `off` control confirms the 'full' backend change alone changes nothing in the 15 Brain-only checks (the room-simulation checks differ run to run on either backend).

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
0.98 %. 242,949 entries flipped -1 -> +1 (902,325 synapses, all glutamate onto targets whose iGluR tertile ranks above their GluCl tertile -- on Mi4 / Mi9 / L3 the absolute GluClalpha TPM exceeds the whole iGluR sum, see round 2: 4.26 % of
glutamatergic synapses) and 40,144 silenced -> 0 (286,600 synapses: glutamate 149,489, histamine 116,366 =
photoreceptor / T1 synapses onto profiled targets with no ort / HisCl1 call, GABA 20,745). Nothing else changes: no
+1 -> -1 flips. 'sign+gain' additionally scales every matched entry by its gain class (4,417,302 entries differ
from off after the cap and fan-in normalisation). 'full' adds the slow matrix: 5,238,228 entries (20.5 % of
edges) with a non-zero slow sign, 22.1 M synapse-equivalents uncapped (+9.34 M / -12.80 M; 24.3 M with gain classes; +9.55 M /
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
columnar types (Mi4, Mi9, Tm9, L3, T2a gain it), and the drive that reaches the object detectors grows 1.1-4x (LC10a best cell 0.07-0.13 -> 0.14-0.29 mV; LC10b 90th percentile 0.05-0.14 -> 0.44-0.63 mV) (LC10a
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
brain-wide activity rise during walking (spikes/step 1 / 11 / 51 -> 70 / 19 / 23 at the three walking reports, i.e. front-loaded; |T4a| optic modulation
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

## Round 2: the rebuilt table's three net rules and the NT-class fallback against the adopted model

Generated 2026-09-12 from one cluster batch (`scripts/cluster_run.py --name rm2-rules`, run dir `rm2-rules-088e7c`: 17 jobs
in parallel on node1 B200s, 0 failures, 5 benchmark jobs of 6.5-8.6 min each) plus a replicate batch (`rm2-rep`, 4 benchmark
jobs: off x2, abs x2). Every job asserted `torch.cuda.is_available()` and printed `cache ok: glutamate cells 29707 cuda
NVIDIA B200`. Inputs: the round-2 receptor table (`flyverse/data/receptors_by_type.csv`: `select_profile()` with
`group_on_override` / `none_contested` / `none_single_source`, KaiR1D = CG3822, Kurmangaliyev 2020 added; section 7 of
`docs/NT_INTEGRATION.md`), the adopted TYPE_NT_OVERRIDE cache (the run dir's `cache` symlink was renamed onto
`runs/ntov-r2-e7706e/cache_override`, whose W / neuron tables hash identically to the local `cache/`: W data+indices+indptr
md5 `ef23cc27...`, neurons `677a2863...`; nt counts acetylcholine 104,044 / glutamate 29,707 / GABA 22,135 / histamine 7,908 /
unknown 2,361 / serotonin 415 / dopamine 395 / octopamine 141), `LIFParams.receptor_model = 'sign'` with
`receptor_net_rule` in {class, abs, nonmda}, and a fifth mode `class` + `receptor_nt_class_fallback = True` ("classfb").
`off` is the adopted model itself (same cache, no receptor stage; the reference `out/rm_ntov.json` of the override adoption
is the same configuration). Scripts: `scripts/benchmark.py --seeds 0,1,2 --json out/rm2_<mode>.json --cache-dir <override
cache>` (native backend: cuda_kernels, cuda_graphs, event_driven, warp); `scripts/probe_bitter.py --seed {0,1,2}` (`--seed` and
`--receptor-nt-class-fallback` added this round; Brain seed = Poisson GRN drive, a real replicate); `scripts/probe_figure_ground.py
--seed {0,1}` (`--seed` added: `room_demo.Sim(seed)`); `scripts/probe_loom.py --seed {0,1,2}` (deterministic probe, no Poisson
input). The cluster's shared cache (`$CLUSTER_DATA/cache`, W md5 `289047c3...`) still lacks the override; it was
not touched.

Files: `out/rm2_{off,class,abs,nonmda,classfb}.{json,txt}`, `out/rm2_{off,abs}_r{2,3}.{json,txt}`, `out/bitter2_<mode>_s{0,1,2}.txt`,
`out/fg2_<mode>_s{0,1}.{csv,txt}`, `out/loom2_<mode>_s{0,1,2}.txt`; CPU tables from `connectome.receptor_signs` on the local cache:
`out/rm2_coverage_rules.txt`, `out/rm2_coverage_targets.csv`, `out/rm2_fg_type_flips.txt`; parsed tables `out/rm2_analysis.txt`.

### R2.0 Verdict

**`abs` is the first mode that is at least neutral on the suite: 27 PASS / 0 FAIL / 2 KNOWN GAP against off's 26 / 1 / 2, with
no check changing status for the worse** (the one change is `walk.power_max_hz` 73.2 FAIL -> 46.1 PASS; the Brain-only sections
are deterministic per cache, so this is a model change, not scatter -- see R2.2). It keeps the Shiu-rules sugar response
(137.4 / 131.1 / 133.7 Hz over three Poisson seeds vs off 123.5 / 122.0 / 114.7; the calibrated replication 3.9-5.5 / 0.0 / 0.0),
raises the pinned-loom GF peak 19 -> 37 Hz with an escape at t = 0.60 s in 3 / 3 deterministic seeds (off: 19 Hz, no escape; class:
24-26 Hz, no escape), and lifts the demo loom to 3 / 3 escapes at 47-57 Hz (off 2 / 3 at 31-35; over three off runs 3 / 9 seeds, over three abs runs 9 / 9 at 47-67 Hz) without a walking-GF storm
(walk_gf p99 23.3 vs 22.2). It does so with 127,462 flipped glutamate synapses (0.60 % of glutamate; T1 62 k, Dm9 25 k, Tm9 23 k,
KCg-m 4 k, L1 2.6 k) and none of the Nmdar2-led Mi4 / L3 / Mi9 / T2a / Dm10 flips. What it does not do: the small-object figure
(Mi4 z -1.8 / -0.1, Tm5Y +0.7 / +0.6, LC10a best cell 0.16-0.17 mV against a 7 mV threshold, `object.LC10a_flip` still a KNOWN GAP)
and the DNp04 / LPT27 / LPT30 optomotor readout (+1.2 / -0.1 / +0.2 Hz; these types have no profile in any rule) are unchanged.

`class` (879,459 flipped syn) still breaks `bitter.shiu_sugar_MN9_hz` (6.6 Hz, 847 spikes/step; 25 / 2 / 2) -- but only in Poisson
seed 0: seeds 1 and 2 give 129.2 and 121.8 Hz, so the round-1 "storm" is a knife-edge on one seed, not a systematic loss.
`nonmda` (1,705,996 flipped syn, incl. L1 -> Mi1 / L5 / C2 / C3 = the ON pathway inverted) storms the walking GF (p99 51.6 FAIL),
loses the optomotor group flip (-2.3 FAIL) and hops 1-3 times per seed before the loom; 24 / 3 / 2. The **NT-class fallback**
(classfb: 97.4 % of |W| synapses under the table, 71.4 % at tier nt_class) flips 7.59 M glutamate synapses (35.8 % of all
glutamate: every glutamate synapse onto an unprofiled GABAergic or glutamatergic cell, because the Davis 2020 Gad1 / VGlut class
baselines call glutamate +1 under the class rule, Nmdar2-led, although their GluClalpha TPM 1,800 / 1,919 exceeds the iGluR sum
753 / 738) and is the worst mode scored so far: 21 / 6 / 2 -- taste.MN9 1.8 FAIL, calibrated sugar 1.3 FAIL, Shiu sugar 11.8 FAIL,
loom_escape 21 Hz / 0 escapes FAIL, rotation group -0.4 FAIL, a cold brain (walk power max 7.8, smell PN 2.7, KC 353 active).

Hypotheses (a) (b) (c) of step 6: (a) no in every mode (Tm5Y / TmY21 / LC11 unprofiled; the medulla figure under class is Mi4 z
+1.7 / +1.8, weaker than round 1's +3.1 on the old table and cache); (b) yes for `abs`, in magnitude only (peak +18 Hz, an escape in the
pinned probe, 3 / 3 demo escapes; the GF still fires when the ball reaches its final size, no expansion-rate response); (c) no in
every mode. Details and every number below.

### R2.1 Coverage each mode ran under

`connectome.receptor_signs(c, net_rule=..., nt_class_fallback=...)` on the adopted cache (167,106 neurons, 25,578,600 stored entries,
sum |W| 121,460,584; `out/rm2_coverage_rules.txt`; the same tables are printed by every probe and benchmark at start-up and match).
The three net rules share one match: 7,654,565 entries = 29.9 % of edges, 31,627,236 |W| synapses = 26.0 % (exact 19.8 % / 17.1 %,
fuzzy 5.7 % / 4.7 %, class 2.7 % / 3.0 %, alias 1.7 % / 1.2 %); 69.1 % / 74.0 % fallback to NT_SIGN; 1.0 % of edges have an unknown
presynaptic transmitter (0 |W|). With the fallback: nt_class 17,317,086 entries / 86,702,413 syn = 67.7 % / 71.4 %, matched 97.6 % /
97.4 %, fallback 1.4 % / 2.6 %. Per postsynaptic module (|W| synapses matched): optic 53.1 %, visual_projection 37.6 %, mushroom
body 93.2 %, antennal lobe 47.8 %, central 4.8 %, descending 1.7 %, mechanosensory 3.2 %, gustatory 0.1 %, VNC 0 (classfb: 99.9 /
99.8 / 100 / 99.1 / 98.4 / 99.0 / 99.7 / 48.9 / 90.9 %).

What each rule changes (whole CNS; flips are all glutamate -1 -> +1, 0 flips +1 -> -1; silenced = histamine onto profiled targets
where every source, 2-5 of them, has HisCl1 / ort off):

| mode | changed entries | changed \|W\| syn | flipped entries / syn | share of glutamate syn (21,205,508) | silenced entries / syn | changed syn per module (optic / central / VP / AL / MB / DN / VNC) |
|---|---|---|---|---|---|---|
| class | 257,578 | 962,932 | 240,199 / 879,459 | 4.15 % | 17,379 / 83,473 | 858,490 / 27,903 / 22,326 / 38,944 / 12,074 / 3,186 / 0 |
| abs | 63,380 | 210,935 | 46,001 / 127,462 | 0.60 % | 17,379 / 83,473 | 196,648 / 9,000 / 215 / 7 / 5,018 / 47 / 0 |
| nonmda | 389,284 | 1,789,469 | 371,905 / 1,705,996 | 8.05 % | 17,379 / 83,473 | 1,669,963 / 45,010 / 21,082 / 38,944 / 11,275 / 3,186 / 0 |
| classfb | 1,672,197 | 7,677,245 | 1,654,818 / 7,593,772 | 35.81 % | 17,379 / 83,473 | 2,983,672 / 2,663,797 / 422,999 / 53,542 / 14,440 / 155,464 / 1,294,622 |

(The task's quoted 878,406 / 127,309 / 1,703,719 are the same rules on the pre-override cache, reproduced exactly from
`out/cache_pre_override/`; the adopted cache adds 1,053 / 153 / 2,277 flipped synapses through the 91 TmY14 cells relabelled glutamate.)

Top flipped pairs, class: Dm12 -> L3 109,232; TmY16 -> Mi4 42,909; Dm4 -> Mi4 39,619; Mi9 -> Mi4 35,908; MeLo10 -> T2a 32,407; Dm12 ->
Mi9 27,967; Mi2 -> T2a 26,188; Dm6 -> T1 25,476; aMe17c -> Dm10 25,317. abs: Dm6 -> T1 25,476; Dm19 -> T1 18,018; Dm12 -> Tm9 12,144;
Dm8b -> Dm9 10,322; Dm8a -> Dm9 8,129; Mi13 -> Tm9 5,037; MBON05 -> KCg-m 2,649 (targets T1 62,361, Dm9 25,467, Tm9 22,946, KCg-m
3,906, L1 2,591, DN1pB 1,535, ER1_a 1,183). nonmda adds L1 -> Mi1 142,185, L1 -> L5 136,202, L1 -> C3 94,518, L1 -> C2 53,476, Mi13 ->
Tm4 40,871, Mi13 -> Mi1 38,775 (targets Mi1 254,177 = 37 % of its input, Tm4 197,318, L5 188,826, C3 105,137, C2 57,453, Mi15 53,169).
classfb adds Tm5c -> Tm37 37,060, TmY5a -> MeLo10 29,293, TmY5a -> LC20b 22,108, MeLo13 -> Li25 20,940 and 3.5 M / 3.6 M syn onto
GABAergic / glutamatergic targets brain-wide (LPi43 12.6 % of its input: Tlp12 3,971, LPi34 3,124). Silenced pairs are the same in
every rule: R8y / R8_unclear / R8p -> Mi15 18,141, -> Mi4 19,659, -> Mi1 12,298, -> Dm2 2,891 (R8* -> Mi1 12,491 in total).

Per-target input under the receptor lookup (share of |W| input synapses matched / flipped; `out/rm2_coverage_targets.csv`):

| target | matched (class/abs/nonmda) | matched classfb | flipped class | flipped abs | flipped nonmda | flipped classfb | silenced syn (all rules) |
|---|---|---|---|---|---|---|---|
| Mi4 | 100 % | 100 % | 36.3 % | 0 | 36.3 % | 36.3 % | 20,024 |
| L3 | 100 % | 100 % | 58.7 % | 0 | 58.7 % | 58.7 % | 0 |
| Mi9 | 100 % | 100 % | 21.6 % | 0 | 0 | 21.6 % | 5,946 |
| T2a | 100 % | 100 % | 19.6 % | 0 | 19.6 % | 19.6 % | 26 |
| Tm9 | 100 % | 100 % | 11.4 % | 11.4 % | 11.4 % | 11.4 % | 169 |
| T1 | 100 % | 100 % | 17.9 % | 17.9 % | 17.9 % | 17.9 % | 0 |
| Mi1 | 100 % | 100 % | 0 | 0 | 37.2 % | 0 | 13,013 |
| Tm5Y, TmY21, LC11 | 0 | 100 % | 0 | 0 | 0 | 0 | 0 |
| LC10a, LC10b, LC4, LPLC2, LPi34, T4a, T5a, HSN, HSE, EPG, PEN_a | 100 % | 100 % | 0 | 0 | 0 | 0 | 0 |
| LPi43 | 0 | 100 % | 0 | 0 | 0 | 12.6 % | 0 |
| DNp01, DNp04, DNp20, LPT27, LPT30, MN9 | 0 | 100 % | 0 | 0 | 0 | 0 | 0 |
| KCg-m | 100 % | 100 % | 0.5 % | 0.5 % | 0.5 % | 0.5 % | 0 |

The nonmda Mi1 flip is the tertile artefact the round-1 skeptic described, now on the ON pathway: Davis 2020 Mi1 has GluClalpha
1,235 TPM against GluRIA 311 (alt sources FCA -1, Davie -1, Kurmangaliyev mixed, Ozel none), yet the nonmda class ranking puts
GluRIA's tertile above GluClalpha's; the same for L5 (547 vs 268), C2 (1,630 vs 462), C3 (703 vs 334), Tm4 (545 vs 104), Mi15,
Dm12. L1 -> Mi1 is the canonical GluClalpha sign-inverting synapse of the ON pathway, so `nonmda` is not a candidate on
structural grounds alone (`receptors_by_type.csv` rows Mi1 / L5 / C2 / C3 / Tm4, columns fast_net_nonmda, fast_pos_val, fast_neg_val).

### R2.2 Benchmark suite (`scripts/benchmark.py --seeds 0,1,2`, 29 checks, native backend, adopted cache)

P = pass, F = fail, gap = known gap. `rm_ntov` is the adoption run of the same off configuration (`out/rm_ntov.json`); `r1 sign`
is round 1's class rule on the old table and the pre-override cache (`out/rm_sign.json`), for reference.

| check | criterion | off | class | abs | nonmda | classfb | rm_ntov (off) | r1 sign (old table) |
|---|---|---|---|---|---|---|---|---|
| rest.spikes_per_step | < 5 | 0.00 P | 0.00 P | 0.00 P | 0.00 P | 0.00 P | 0.00 P | 0.00 P |
| taste.MN9_hz | > 2 | 5.85 P | 4.94 P | 10.93 P | 4.94 P | **1.76 F** | 5.85 P | 7.92 P |
| smell.PN_hz | < 100 | 11.19 P | 7.07 P | 7.86 P | 3.49 P | 2.69 P | 11.19 P | 5.66 P |
| smell.KC_active | > 0 | 1426 P | 704 P | 816 P | 559 P | 353 P | 1426 P | 1502 P |
| dn.DNa02_L_leg_asym_hz | > 0.3 | 2.58 P | 2.58 P | 2.58 P | 2.58 P | 2.58 P | 2.58 P | 2.58 P |
| dn.MDN_top_hz | < 250 | 153 P | 153 P | 153 P | 154 P | 148 P | 153 P | 152 P |
| dn.DNp09_top_hz | < 250 | 152 P | 152 P | 152 P | 152 P | 152 P | 152 P | 152 P |
| walk.GF_max_hz | < 38 | 4.63 P | 0.00 P | 8.52 P | 5.01 P | 5.05 P | 4.63 P | 4.99 P |
| walk.power_max_hz | < 50 | 73.18 F | 73.29 F | **46.10 P** | 58.93 F | 7.84 P | 73.18 F | 90.91 F |
| walk.power_sustained_hz | < 50 | 31.58 P | 36.44 P | 21.36 P | 26.10 P | 2.90 P | 31.58 P | 38.86 P |
| loom.GF_peak_hz | >= 20 | 43.88 P | 33.02 P | 28.04 P | 34.42 P | 29.36 P | 41.79 P | 30.99 P |
| loom.escape_cm | not none | 3.50 P | 3.50 P | 3.50 P | 3.50 P | 3.50 P | 3.50 P | 3.50 P |
| rotate.DNp20_flip_hz | < -2 | -13.27 P | -33.67 P | -36.02 P | -16.69 P | -27.39 P | -27.48 P | -20.47 P |
| motion.min_dsi | >= 0.1 | 0.17 P | 0.17 P | 0.17 P | 0.17 P | 0.17 P | 0.18 P | 0.17 P |
| motion.correct_directions | == 8 | 8 P | 8 P | 8 P | 8 P | 8 P | 8 P | 8 P |
| loom_escape.GF_peak_hz | >= 33 | 35.18 P | 57.51 P | 57.21 P | 40.29 P | **21.33 F** | 37.39 P | 46.36 P |
| loom_escape.escapes (of 3) | >= 1 | 2 P | 3 P | 3 P | 2 P | **0 F** | 2 P | 3 P |
| walk_gf.p99_hz | < 38 | 22.20 P | 19.45 P | 23.32 P | **51.63 F** | 14.88 P | 19.16 P | 16.32 P |
| rotation.group_flip_hz | <= -3 | -7.00 P | -8.16 P | -9.90 P | **-2.32 F** | **-0.38 F** | -7.38 P | -7.62 P |
| object.LC10a_flip_hz | abs >= 1 | 0.00 gap | 0.00 gap | -0.01 gap | -0.01 gap | -0.03 gap | -0.00 gap | -0.00 gap |
| bitter.calibrated_sugar_MN9_hz | > 2 | 4.57 P | 4.84 P | 5.52 P | 4.84 P | **1.27 F** | 4.57 P | 6.91 P |
| bitter.calibrated_sugar_bitter_MN9_hz | < 1 | 0.00 P | 0.00 P | 0.00 P | 0.00 P | 0.00 P | 0.00 P | 0.00 P |
| bitter.shiu_sugar_MN9_hz | > 50 | 123.5 P | **6.63 F** | 137.4 P | 136.5 P | **11.81 F** | 123.5 P | 5.57 F |
| bitter.shiu_sugar_bitter_MN9_hz | < 10 | 2.12 P | 2.03 P | 0.47 P | 2.07 P | 0.00 P | 2.12 P | 0.88 P |
| wind.DNp18_flip_hz | >= 15 | 46.50 P | 45.97 P | 46.44 P | 46.39 P | 41.67 P | 45.20 P | 46.61 P |
| wind.DNp33_flip_hz | <= -15 | -49.76 P | -50.07 P | -49.06 P | -49.68 P | -44.78 P | -50.48 P | -50.67 P |
| odour.apple_channel_8cm_hz | >= 10 | 17.50 P | 17.25 P | 17.53 P | 17.17 P | 18.11 P | 17.59 P | 17.14 P |
| odour.apple_channel_clean_hz | <= 6 | 4.44 P | 4.47 P | 4.34 P | 4.42 P | 4.13 P | 4.34 P | 4.59 P |
| compass.wedge_cells_persisting | >= 6 | 0 gap | 0 gap | 0 gap | 0 gap | 0 gap | 0 gap | 0 gap |

| mode | pass | fail | known gap | status changes vs off | changed entries | runtime (min) |
|---|---|---|---|---|---|---|
| off | 26 | 1 | 2 | -- | 0 | 6.5 |
| class | 25 | 2 | 2 | bitter.shiu_sugar 123.5 P -> 6.6 F | 257,578 | 8.3 |
| **abs** | **27** | **0** | 2 | walk.power_max 73.2 F -> 46.1 P | 63,380 | 7.4 |
| nonmda | 24 | 3 | 2 | walk_gf.p99 22.2 P -> 51.6 F; rotation.group -7.0 P -> -2.3 F | 389,284 | 8.3 |
| classfb | 21 | 6 | 2 | taste 5.85 P -> 1.76 F; walk.power_max 73.2 F -> 7.8 P; loom_escape 35.2 P -> 21.3 F; escapes 2 P -> 0 F; rotation -7.0 P -> -0.4 F; calibrated sugar 4.57 P -> 1.27 F; Shiu sugar 123.5 P -> 11.8 F | 1,672,197 | 8.6 |

Determinism: the Brain-only sections (rest, taste, smell, dn, walk incl. walk.power_max, bitter) are bit-identical across every
off run on the same cache (`rm_ntov`, `rm2_off`, `rm2_off_r2`, `rm2_off_r3`: taste 5.85, smell 11.19 / 1426, walk.power_max 73.18,
Shiu 123.54; the pre-override runs `rm_off`, `rm_base_r2`, `skeptic_a/b` all give 79.06 / 12.39 / 1599), so their per-mode
values are model effects, not scatter. The demo sections (loom, rotate, loom_escape, walk_gf, rotation, object, wind, odour) are
chaotic: over the six off runs on file, loom_escape peak 31.0-41.7 Hz with 0-2 escapes, walk_gf p99 17.7-22.3, rotate.DNp20
-13.3 to -28.6, rotation.group -6.6 to -7.6, wind.DNp18 44.5-46.5. Differences of that size between modes mean nothing.

Replicates (`rm2-rep-08ac4a`, two more suite runs each of off and abs, same cache, same flags; `out/rm2_analysis_replicates.txt`):

| run | pass / fail / gap | taste.MN9 | smell PN / KC active | walk.power_max | Shiu sugar | loom.GF_peak | rotate.DNp20 | loom_escape peak / escapes (per seed) | walk_gf p99 | rotation group (DNp20 / HSN / HSE / DNp04 / LPT27 / LPT30) | wind DNp18 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 26 / 1 / 2 | 5.85 | 11.19 / 1426 | 73.18 F | 123.54 | 43.9 | -13.3 | 35.2 / 2 (31 n, 33 y, 35 y) | 22.2 | -7.0 (-9.8 / -7.9 / -3.3 / +0.1 / -0.4 / +0.7) | 46.5 |
| off_r2 | 24 / 3 / 2 | 5.85 | 11.19 / 1426 | 73.18 F | 123.54 | 36.9 | -26.2 | **31.0 F / 0 F** (26 n, 27 n, 31 n) | 19.3 | -7.2 (-10.5 / -7.5 / -3.5 / -0.5 / +0.3 / +0.8) | 43.9 |
| off_r3 | 26 / 1 / 2 | 5.85 | 11.19 / 1426 | 73.18 F | 123.54 | 36.7 | -26.6 | 35.5 / 1 (36 y, 30 n, 28 n) | 20.0 | -7.5 (-11.1 / -8.5 / -3.0 / +0.1 / +0.1 / +0.1) | 47.2 |
| abs | 27 / 0 / 2 | 10.93 | 7.86 / 816 | 46.10 P | 137.42 | 28.0 | -36.0 | 57.2 / 3 (47 y, 57 y, 56 y) | 23.3 | -9.9 (-13.0 / -10.6 / -6.1 / +1.2 / -0.1 / +0.2) | 46.4 |
| abs_r2 | 27 / 0 / 2 | 10.93 | 7.86 / 816 | 46.10 P | 137.42 | 27.1 | -31.4 | 66.8 / 3 (58 y, 64 y, 67 y) | 29.3 | -11.9 (-18.9 / -10.8 / -6.0 / +0.2 / +0.4 / -0.3) | 45.7 |
| abs_r3 | 27 / 0 / 2 | 10.93 | 7.86 / 816 | 46.10 P | 137.42 | 27.1 | -31.9 | 61.4 / 3 (52 y, 59 y, 61 y) | 25.9 | -12.2 (-19.1 / -11.2 / -6.3 / +1.3 / -0.2 / +0.3) | 44.8 |

The Brain-only checks are identical across the three runs of each mode (walk.power_sustained 31.58 / 31.58 / 31.44 is the only
non-identical Brain-only value); abs's walk.power_max 46.10 PASS, taste 10.93, Shiu 137.42 are reproducible model effects.
The chaotic sections: abs escapes in 9 / 9 demo looms at 47-67 Hz against off's 3 / 9 at 26-36 Hz (off_r2 fails the loom_escape
check outright, 31.0 Hz / 0 escapes -- the round-1 bistability of this check at the 33 Hz threshold); abs's walking GF p99 23-29
Hz vs off 19-22 (PASS < 38 in all), rotation group -9.9 to -12.2 vs -7.0 to -7.5 (DNp20 -13 to -19, HSN -10.6 to -11.2, HSE -6.0
to -6.3: the existing optomotor readout is stronger under abs), loom.GF_peak (legacy loom) 27-28 vs 37-44, rotate.DNp20 -31 to -36
vs -13 to -27. Over three suite runs abs is 27 / 0 / 2 three times; off is 26 / 1 / 2, 24 / 3 / 2, 26 / 1 / 2.

Section detail (loom_escape per seed: walking GF max / hops before loom / loom peak / escape; walk_gf median / p90 / p99 / max
(voluntary takeoffs); rotation flips; object LC10a flip (rate); smell PN / KC (active) / LN; taste MN9; walk power max / sustained):

| mode | loom_escape | walk_gf | rotation | object LC10a (LC16) | smell | taste | walk power |
|---|---|---|---|---|---|---|---|
| off | 9 / 0 / 31 / no; 19 / 0 / 33 / yes; 16 / 0 / 35 / yes | 12 / 18 / 22 / 23 (0) | group -7.0: DNp20 -9.8 HSN -7.9 HSE -3.3 DNp04 +0.1 LPT27 -0.4 LPT30 +0.7 | +0.00 (0.02 Hz) (LC16 +0.22) | 11.2 / 2.62 (1426) / 63 | 5.85 | 73.2 / 31.6 |
| class | 19 / 0 / 41 / yes; 13 / 0 / 58 / yes; 20 / 0 / 47 / yes | 12 / 16 / 19 / 20 (0) | -8.2: -14.6 / -5.8 / -4.0 / +1.1 / -1.6 / -1.7 | +0.00 (0.01) (LC16 +0.33) | 7.1 / 1.68 (704) / 23 | 4.94 | 73.3 / 36.4 |
| abs | 19 / 0 / 47 / yes; 22 / 0 / 57 / yes; 12 / 0 / 56 / yes | 19 / 22 / 23 / 23 (0) | -9.9: -13.0 / -10.6 / -6.1 / +1.2 / -0.1 / +0.2 | -0.01 (0.02) (LC16 +0.40) | 7.9 / 2.17 (816) / 26 | 10.93 | 46.1 / 21.4 |
| nonmda | 38 / 2 / 28 / no; 35 / 1 / 39 / yes; 34 / 3 / 40 / yes | 27 / 42 / 52 / 53 (0) | -2.3: -3.2 / -3.0 / -0.8 / -1.5 / +0.9 / -0.3 | -0.01 (0.04) (LC16 +0.31) | 3.5 / 0.56 (559) / 12 | 4.94 | 58.9 / 26.1 |
| classfb | 12 / 0 / 16 / no; 8 / 0 / 21 / no; 13 / 0 / 21 / no | 8 / 14 / 15 / 15 (0) | -0.4: +0.6 / -1.0 / -0.7 / -0.3 / +1.8 / +0.2 | -0.03 (0.08) (LC16 -0.13) | 2.7 / 0.35 (353) / 10 | 1.76 | 7.8 / 2.9 |

Reading: `abs` is the only mode whose walking GF stays at the off level (p99 23.3) while every demo loom escapes (47 / 57 / 56 Hz)
-- the same direction round 1 saw under the old `class` rule, now without the Mi4 / L3 / Mi9 flips and without the Shiu-sugar
loss. Its taste.MN9 rises 5.85 -> 10.93 (deterministic) and the antennal lobe runs cooler (PN 7.9, LN 26; the AL's 38,944
flipped synapses of the class rule are not made by abs, the change comes through the 5,018 MB and the T1 / Tm9 / Dm9 optic flips'
downstream effect on the network's tone). `nonmda` hops before the loom (2 / 1 / 3 voluntary takeoffs, walking GF 34-38 Hz), a
walking storm through the inverted ON pathway. `classfb` is a cold brain: walk power 7.8, GF never reaches 33 Hz, no escapes,
MN9 1.3-1.8 Hz under sugar, the top walking types are FB6A / OA-VPM3 at ~250 Hz.

### R2.3 Sugar / bitter (`probe_bitter.py`, three Poisson seeds per mode; MN9 Hz, spikes/step of the whole brain in brackets)

| mode | seed | calibrated: sugar | sugar + bitter | bitter | Shiu rules: sugar | sugar + bitter | bitter |
|---|---|---|---|---|---|---|---|
| off | 0 / 1 / 2 | 4.6 (14) / 5.6 (21) / 4.5 (22) | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 123.5 (114) / 122.0 (124) / 114.7 (117) | 2.1 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| class | 0 / 1 / 2 | 4.8 (18) / 5.4 (12) / 2.5 (17) | 0.0 x3 | 0.0 x3 | **6.6 (847)** / 129.2 (124) / 121.8 (133) | 2.0 / 1.2 / 0.0 | 0.0 / 0.0 / 0.0 (765) |
| abs | 0 / 1 / 2 | 5.5 (19) / 4.3 (17) / 3.9 (25) | 0.0 x3 | 0.0 x3 | 137.4 (100) / 131.1 (115) / 133.7 (121) | 0.5 / 0.0 / 0.0 | 0.0 x3 |
| nonmda | 0 / 1 / 2 | 4.8 (18) / 5.4 (12) / 2.5 (17) | 0.0 x3 | 0.0 x3 | 136.4 (119) / 156.0 (159) / 132.2 (136) | 2.1 / 1.3 / 0.0 | 0.0 / 0.0 / 0.0 (819) |
| classfb | 0 / 1 / 2 | **1.3 (11) / 1.6 (15) / 0.0 (19)** | 0.0 x3 | 0.0 x3 | **11.8 (667) / 17.7 (644) / 15.4 (699)** | 0.0 x3 | 0.0 x3 (679-805) |

`bitter.shiu_sugar_MN9_hz` **survives under `abs`** (131-137 Hz, 3 / 3 seeds, at the off level of whole-brain activity 100-121
spikes/step) **and under `nonmda`** (132-156 Hz, 3 / 3). Under `class` it fails in seed 0 only (6.6 Hz at 847 spikes/step; seeds 1 / 2
at 129 / 122 Hz with 124 / 133 spikes/step): the round-1 storm is a single-seed knife-edge of the uniform uncapped model, not a
systematic property of the 879 k flips, though the benchmark (seed 0) will keep failing it. The fallback fails both replications:
calibrated sugar 0.0-1.6 Hz (criterion > 2), Shiu 12-18 Hz at 644-699 spikes/step in all three conditions -- the brain storms
without sugar (bitter alone 679-805 spikes/step), and the sweet -> MN9 path (MN9 cholinergic, 0 flipped inputs; GNG038 8.4 % and
GNG175 4.8 % of their inputs flipped) is drowned. The calibrated replication (sugar fires, bitter shuts it, bitter alone nothing)
holds in every mode but classfb.

### R2.4 Figure-ground (`probe_figure_ground.py --seed 0,1`; apple 5 cm ahead-left, heading +-20 deg; z per replicate, figure in rate units in brackets)

159 columns view the apple, 1,050 are background; 89,380 of 89,390 rate cells have a column; 112 types scored per run.

| type | off s0 / s1 | class s0 / s1 | abs s0 / s1 | nonmda s0 / s1 | reads |
|---|---|---|---|---|---|
| Mi4 | +1.2 (+0.0012) / +1.1 (+0.0012) | +1.7 (+0.0121) / +1.8 (+0.0126) | -1.8 (-0.0021) / -0.1 (-0.0002) | +2.2 (+0.0116) / +0.2 (+0.0017) | class: figure x10 (apple columns 0.026 / 0.031 vs none 0.013 / 0.021), z 1.7-1.8 -- weaker than round 1's +3.1; abs: nothing (0 Mi4 inputs flipped) |
| Mi9 | +1.0 / +2.2 | +1.6 (+0.012) / +2.3 (+0.017) | +3.5 (+0.004) / +2.0 (+0.003) | +0.4 / +0.1 | |
| L3 | +1.0 (+0.0045) / +1.0 (+0.0046) | +1.2 (+0.027) / +2.3 (+0.048) | +1.2 (+0.005) / +1.3 (+0.006) | +1.3 (+0.024) / +0.2 (+0.005) | class / nonmda: Dm12 -> L3 flipped (59 % of L3's input) |
| Tm9 | +0.1 / +0.3 | +1.1 (+0.011) / +1.9 (+0.019) | +1.1 (+0.006) / +1.2 (+0.006) | +1.0 / -0.2 | gains in all three (Dm12 -> Tm9 flipped in all three) |
| T2a | -0.5 / -0.4 | +0.6 / +0.8 | -1.6 / -1.2 | -0.0 / -0.4 | |
| T2 | +0.3 / +0.8 | +0.6 / +0.8 | +2.8 (+0.006) / +3.8 (+0.009) | -0.3 / -0.5 | abs: gained (T2's own inputs unchanged) |
| **Tm5Y** | -0.2 (-0.0008) / -0.0 | +0.5 (+0.0030) / +0.4 (+0.0030) | +0.7 (+0.0025) / +0.6 (+0.0025) | -0.0 / -0.0 | z < 1 in every run; figure at most +0.003 rate units |
| **TmY21** | -0.5 / -0.4 | -0.7 / -0.7 | -0.4 / -1.2 | +0.0 / -0.3 | unchanged (unprofiled) |
| TmY13 | -0.5 / -1.4 | -0.8 (-0.016) / -1.4 (-0.030) | +1.7 (+0.0035) / +1.7 (+0.0039) | -1.8 / +0.5 | |
| Mi1 | -8.6 (-0.0053) / -8.8 (-0.0057) | -1.8 (-0.0050) / -1.6 (-0.0037) | **-11.3 (-0.0058) / -9.4 (-0.0057)** | **+3.2 (+0.0070) / +1.1 (+0.0033)** | ON figure kept under abs although its 13,013 R8 histamine synapses are silenced in all three rules -- the round-1 attribution of the class-rule loss to that silencing was wrong; the loss comes from the class flips upstream (Mi4 / Mi9 / L3 feedback). nonmda inverts it (L1 -> Mi1 +) |
| Tm3 | -7.8 / -7.7 | -5.5 / -5.8 | -8.5 / -7.6 | -1.8 / -1.8 | |
| Tm1 | +3.7 / +3.6 | +0.7 / +0.4 | +4.9 / +4.4 | -1.0 / +0.3 | lost under class (as round 1), kept under abs |
| Tm2 | +3.5 / +4.1 | +2.2 / +2.2 | +4.4 / +4.3 | +0.3 / +0.7 | |
| Tm4 | +4.9 / +5.7 | +3.0 / +3.0 | +5.8 / +4.9 | +0.1 / +1.2 | |
| T1 | +4.1 (+0.0042) / +4.5 | +3.9 (+0.008) / +4.6 (+0.008) | +9.5 (+0.008) / +7.9 (+0.008) | +0.6 / +1.2 | abs: T1 figure x2 (Dm6 / Dm19 -> T1 flipped, 18 % of T1's input) |
| L1 / L2 | +17.1 / +16.4; +12.0 / +11.9 | +9.3 / +13.0; +6.5 / +7.6 | +18.9 / +18.7; +13.9 / +12.2 | +15.9 / +12.0; +5.7 / +4.5 | figure itself unchanged (+0.016 / +0.011) |
| C2 / C3 / L5 | -7.8 / -5.3; -5.5 / -4.5; -2.0 / -2.0 | -2.3 / -3.8; -1.3 / -2.0; -0.9 / -1.5 | -8.6 / -5.7; -7.0 / -6.3; -2.1 / -0.8 | **+9.1 / +5.8; +5.6 / +3.8; +5.7 / +5.0** | nonmda: the whole ON / lamina-feedback stage changes sign (L1 -> Mi1, L5, C2, C3 flipped) |
| T4a / T5a | +1.7 / +2.1; +0.8 / +0.7 | +2.8 / +0.9; +1.9 / +1.8 | -0.1 / -0.6; +1.9 (+0.053) / +2.0 (+0.054) | -0.9 / -0.7; +0.1 / -0.2 | abs: T5a figure x3 (Tm9 input) |

Spiking targets, population mean |drive| (mV, apple / none) and per-cell (apple - none) drive median / 90th percentile / max:

| mode | seed | LC10a pop. | LC10a per cell | LC10b per cell | LC4 pop. | LPLC2 pop. | LPLC2 per cell max | LC16 per cell |
|---|---|---|---|---|---|---|---|---|
| off | 0 | -0.01 / -0.02 | +0.008 / +0.04 / +0.13 | +0.06 / +0.16 / +0.50 | -0.27 / -0.29 | 1.53 / 1.41 | +3.79 | -0.02 / +0.05 / +1.24 |
| off | 1 | -0.01 / -0.00 | -0.004 / +0.03 / +0.09 | +0.00 / +0.13 / +0.63 | -0.25 / -0.30 | 1.53 / 1.42 | +3.88 | +0.01 / +0.08 / +1.26 |
| class | 0 | -0.05 / -0.02 | -0.029 / +0.04 / +0.12 | -0.09 / +0.25 / +0.88 | -0.75 / -0.75 | 1.83 / 1.53 | +4.35 | -0.08 / +0.19 / +1.09 |
| class | 1 | -0.05 / -0.01 | -0.024 / +0.04 / +0.14 | +0.09 / +0.23 / +0.59 | -0.75 / -0.73 | 1.84 / 1.54 | +4.12 | -0.06 / +0.28 / +1.36 |
| abs | 0 | 0.00 / -0.00 | +0.006 / +0.05 / +0.17 | +0.07 / +0.18 / +0.59 | -0.21 / -0.25 | 1.90 / 1.67 | +5.64 | -0.01 / +0.06 / +1.37 |
| abs | 1 | -0.02 / -0.02 | +0.008 / +0.04 / +0.16 | +0.05 / +0.21 / +0.72 | -0.21 / -0.27 | 1.90 / 1.66 | +5.84 | +0.01 / +0.06 / +1.42 |
| nonmda | 0 | -0.06 / -0.05 | -0.018 / +0.06 / +0.37 | +0.07 / +0.23 / +0.45 | -0.36 / -0.37 | 1.28 / 1.26 | +1.00 | -0.04 / +0.15 / +0.53 |
| nonmda | 1 | -0.04 / -0.05 | -0.002 / +0.09 / +0.25 | -0.11 / +0.03 / +0.59 | -0.34 / -0.37 | 1.29 / 1.27 | +1.07 | -0.00 / +0.36 / +0.67 |

LC10a's best cell gains at most 0.04-0.08 mV in any mode (0.09-0.13 -> 0.12-0.17 mV; nonmda 0.25-0.37) against a ~7 mV threshold;
its rate is 0.01-0.04 Hz with or without the apple in every benchmark's object section. Under abs the LPLC2 population drive to the
apple rises 1.53 -> 1.90 mV (best cell 3.8 -> 5.6-5.8 mV): the small dark object drives the loom detector harder, not the
small-object detector -- consistent with the loom results below and with the same failure mode NOTES 9 described.

### R2.5 Loom (`probe_loom.py --seed 0,1,2`; black 3 cm ball from the left at 1 m/s after 1.5 s of walking; eager Brain + OpticLobe, LPi x4 default)

Reproducible per seed across jobs and nodes (byte-identical reruns), but seeds 1 and 2 are byte-identical in every mode and off / class seed 0 differs from them in spikes/step and (class) in the GF trace itself -- so at most two distinct samples per mode, one for abs. Peaks over the 100 ms reports.

| mode | GF walking max | GF loom peak (Hz) | LPLC2 pk | LC4 | LC16 | DNp04 | DNp06 | DNp11 | TTMn | \|LPi34\| | \|T4a\| | \|T5a\| | spikes/step (walking reports / peak) | escape (gf_hz 33) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off (3 seeds) | 0 | 19 | 3 | 0 | 0-1 | 2 | 8-10 | 0 | 11 | 0.32 | 0.03 | 0.05 | 2 / 3 / 28; pk 83-172 | no |
| class (3) | 3 | 24-26 | 3 | 0 | 1-2 | 4-5 | 6-9 | 0 | 13-14 | 0.32 | 0.09 | 0.08 | 1 / 3 / 3; pk 26-33 | no |
| **abs (3)** | 1 | **37** | 5 | 0 | 1 | 12 | 14 | 7 | 20 | 0.43 | 0.03 | 0.13 | 3 / 1 / 4; pk 33 | **yes: t = 0.60 s, 3.5 cm, GF 35 Hz, TTMn 20 Hz** |

GF (DNp01) time course, Hz at each report (t after loom start / distance): off 0 0 0 0 0 16 19 7 6; class 0 0 0 5 5 14 26 15 6
(seed 0: 14 26 15 6; seeds 1-2: 13 24 12 4); abs 0 0 0 0 5 14 35 37 29 (reports at 0.1 s / 41 cm, 0.1 / 40, 0.2 / 31, 0.3 / 21,
0.4 / 11, 0.5 / 3, 0.6 / 3, 0.7 / 3, 0.8 / 3). Round 1 on the old table and pre-override cache: off 21 / 24 / 24, class 32 / 32 / 32.
On the adopted cache the off peak is 19 in all three seeds; the rebuilt class rule adds +5-7 Hz (24-26, with a 5 Hz early response at
0.3-0.4 s / 21-11 cm), the abs rule +18 Hz (37, sustained 35-37-29 over three reports) and crosses the escape threshold. The
mechanism under abs is the T5 / LPLC2 side: Tm9's flipped input (Dm12 12,144 + Mi13 5,037 syn) raises |T5a| 0.05 -> 0.13 and
LPLC2 3 -> 5 Hz, DNp04 2 -> 12, DNp06 8-10 -> 14, DNp11 0 -> 7; Mi4 / Mi9 / T4a are unchanged (|T4a| 0.03). The GF still fires only when
the ball has reached its final size (3 cm, t >= 0.5 s) in every mode -- final-size coding, no expansion-rate response.

### R2.6 Verdicts on the step-6 hypotheses, round 2

**(a) Glutamatergic optic types signed per target change the figure-ground signal for Tm5Y / TmY21 / LC10 and the LPi -> LPLC2
balance.** *No, in every rule.* Tm5Y's figure is at most +0.003 rate units (z 0.4-0.7 under class / abs, 0 under nonmda), TmY21
z -0.4 to -1.2, LC10a's best cell 0.12-0.17 mV (0.25-0.37 under nonmda) against 7 mV and 0.01-0.04 Hz in the object section, all
three unprofiled in every source (0 % matched input; under the fallback they take the ChAT class baseline, glutamate -1, 0 flips).
The medulla figure the class rule gave in round 1 (Mi4 z +3.1) is weaker on the rebuilt table and adopted cache (+1.7 / +1.8; figure
x10 in rate units) and absent under abs, which flips none of Mi4's inputs; under nonmda the ON pathway inverts (Mi1 / C2 / C3 / L5
change sign). LPi34 -> LPLC2 stays GluCl-dominant (-1) in every rule (LPi34 alias Davis 2020 with Kurmangaliyev alias -1; LPLC2 exact Davis, abs -1, Ozel -1),
LPi43 is unprofiled; the x4 pair gain remains a stop-gap the data neither reproduce nor contradict.

**(b) Loom-rate coding of the GF.** *No change in what the GF encodes; a real quantitative gain under `abs`.* Pinned probe: GF peak
19 -> 37 Hz, escape at 0.60 s in 3 / 3 deterministic seeds (class 24-26, no escape), LPLC2 3 -> 5, DNp04 2 -> 12, still final-size
coding (first GF spike at 0.5 s / 3 cm; abs has a 5 Hz report at 0.4 s / 11 cm). Demo: 3 / 3 escapes at 47-57 Hz (off 3 / 9 seeds over three runs; 24-42 Hz, 0-2 escapes per run over the eight off runs on file) with the walking GF unchanged (p99 23.3 vs 17.7-22.3 across off runs,
0 hops), i.e. the escape / self-motion separation improves without a hand gain -- the round-1 'sign' effect, now obtained from
127 k flips (T1, Dm9, Tm9) that no alternative source contradicts for T1 / Dm9 (T1: Davis KaiR1D 109 TPM, Kurmangaliyev +1, Ozel / FCA / Davie none; Dm9: Davis / Kurmangaliyev / FCA +1, Ozel / Davie none) but that Kurmangaliyev / FCA / Davie contradict for Tm9 (all -1; Davis KaiR1D 107 TPM vs GluCl 0) -- the Tm9 flip
(23 k syn, 11 % of Tm9's input) is the one abs flip on the loom path that rests on a single source.

**(c) The optomotor DNp04 / LPT27 / LPT30 readout.** *No, in every rule.* DNp04 +0.1 / +1.1 / +1.2 / -1.5 / -0.3, LPT27 -0.4 / -1.6
/ -0.1 / +0.9 / +1.8, LPT30 +0.7 / -1.7 / +0.2 / -0.3 / +0.2 Hz (off / class / abs / nonmda / classfb, benchmark rotation section;
5 more samples, 18 with round 1's 13; |flip| <= 1.8 Hz). All three are unprofiled in every rule (0 % matched; 100 % under the fallback with 0
flips, they are cholinergic). The existing DNp20 + HSN + HSE readout survives class (-8.2) and abs (-9.9, HSN -10.6, HSE -6.1, the
strongest on file) and is lost under nonmda (-2.3) and classfb (-0.4).

**Sugar / bitter.** Calibrated replication intact under off / class / abs / nonmda (3.9-5.6 / 0.0 / 0.0 Hz over three Poisson seeds),
lost under classfb (0.0-1.6 Hz). Shiu-rules replication intact under abs (131-137 Hz) and nonmda (132-156), broken in 1 of 3 seeds
under class (6.6 / 129 / 122) and in 3 of 3 under classfb (12-18 Hz at 644-699 spikes/step).

**The NT-class fallback** (71.4 % of |W| synapses at tier nt_class) is not usable as built: under the class rule the Davis 2020
Gad1 and VGlut class baselines call glutamate +1 (Nmdar2-led tertile; GluClalpha TPM 1,800 / 1,919 vs iGluR 753 / 738), so
7.59 M glutamate synapses onto every unprofiled GABAergic / glutamatergic cell in the CNS flip (3.52 M / 3.64 M syn; 2.66 M in
the central brain, 1.29 M in the VNC, 2.90 M in the optic lobe), and the suite drops to 21 / 6 / 2. The same baselines under the
abs column are -1 for all three classes (`<nt=...>` rows of `receptors_by_type.csv`), i.e. an `abs` + fallback run would change only
the histamine / ACh / GABA class calls, which equal NT_SIGN -- so that combination is a no-op on the fast sign and was not run.
The fallback cannot reach hypotheses (a)-(c) either way: Tm5Y / TmY21 / LC11 / DNp04 / LPT27 / LPT30 / MN9 are cholinergic, and the
ChAT baseline is NT_SIGN for every classical transmitter.

**Is any mode at least neutral on the suite?** Yes: `abs`. 27 / 0 / 2 against off 26 / 1 / 2; no check worsens in status; the
deterministic sections move taste.MN9 5.85 -> 10.93, walk.power_max 73.2 -> 46.1 (the off FAIL), Shiu sugar 123.5 -> 137.4, smell
PN 11.2 -> 7.9 / KC active 1426 -> 816; the chaotic sections stay inside the off scatter except loom_escape (3 / 3 at 47-57 Hz, above
any off run) and rotate.DNp20 (-36, the largest on file, same sign). Three suite runs of abs give 27 / 0 / 2 three times (off: 26 / 1 / 2, 24 / 3 / 2, 26 / 1 / 2), the Brain-only values bit-identical, the demo loom escaping in 9 / 9 seeds against off's 3 / 9. Step 8 (stop-gap retirement with the
receptor model on) can proceed with `--receptor-model sign --receptor-net-rule abs`; `class` and `nonmda` are not candidates
(class: Shiu sugar seed-0 storm, Mi9 / L3 / Tm9 flips contradicted by every alternative source; nonmda: ON pathway inverted,
walking storm), and the fallback needs its class-baseline rows recomputed under the abs criterion before it is scored again.

### R2.7 Caveats

* Coverage is unchanged at 26.0 % of |W| synapses (29.9 % of edges) for the three rules; the populations named by (a) and (c)
  are still in the other 74 % (Tm5Y, TmY21, LC11, DNp04, DNp20, LPT27, LPT30, MN9, the sweet interneurons, LPi43). abs changes
  0.17 % of |W| synapses (210,935): optic 196,648, central 9,000 (clock DN1pB / DN1pA / DN1a 3,533, ring ER1-4 5,454), MB 5,018 (MBON05 -> KCg-m
  2,649), visual projection 215, descending 47, antennal lobe 7, VNC 0.
* One benchmark run per mode (class, nonmda, classfb) and three each of off and abs; bitter three Poisson seeds; figure-ground two seeds; the
  loom probe is deterministic (3 identical seeds). The demo sections' scatter is documented above from six off runs; claims that
  rest on single chaotic values (rotate.DNp20 -36, smell.KC 816) are not load-bearing.
* The pinned loom's off peak on the adopted cache is 19 Hz (round 1: 21-24 on the pre-override cache), so the +18 Hz of abs is
  measured against a slightly lower baseline; the 37 Hz peak and the escape are absolute.
* The abs Tm9 flip rests on Davis 2020 alone (KaiR1D = CG3822 107 TPM, GluClalpha 0) against three sources at -1; T1's on
  Davis + Kurmangaliyev with Ozel / FCA / Davie silent. If the Tm9 row is contested by a stricter rule, the loom gain should be re-run.
* The cluster's shared cache lacks the TYPE_NT_OVERRIDE; every job here pointed at `runs/ntov-r2-e7706e/cache_override`
  (benchmark via `--cache-dir`, probes via the run dir's `cache` symlink). Rebuilding the shared cache is still pending.
* `scripts/probe_bitter.py` gained `--seed` and `--receptor-nt-class-fallback`, `scripts/probe_figure_ground.py` gained `--seed`
  (`room_demo.Sim(seed)`); scripts/benchmark.py also gained --receptor-nt-class-fallback and --cache-dir (used by every rm2_* run).

### Corrections (round-2 verification, `verify:score:rules`)

* `loom.GF_peak_hz` (the legacy in-suite loom) is a systematic **cost** of `abs`, not scatter: 28.0 / 27.1 / 27.1 / 28.0 Hz over four runs against off 43.9 / 36.9 / 36.7 / 38.0 (still PASS, criterion >= 20). The legacy loom and the demo `loom_escape` move in opposite directions under the same rule.
* Off-escape denominator with a fourth run: demo loom escapes in 3 of 12 seeds (peaks 21.5-36.0 Hz) under off, 12/12 at 47-74 Hz under abs; off scores 26/1/2 in 2 of 4 suite runs and 24/3/2 in the other 2; abs 27/0/2 in 4 of 4.
* The classfb flips decided by the class baseline (tier nt_class) are gaba 3,216,485 syn + glutamate 3,497,828 = 6,714,313; the remaining 439,935 of the 7.59 M are ordinary class-rule flips onto profiled targets, and 0.44 M land on cholinergic / histaminergic / octopaminergic targets.
* "LC10a per-cell max drive" is the maximum positive (apple - none) drive; the largest-magnitude LC10a cells under class are -0.26 / -0.29 mV. "TmY21 unchanged" means TmY21's input (0 % matched, 0 flipped); its z moves -0.45 / -0.39 (off) -> -0.71 / -0.75 (class) -> -0.44 / -1.20 (abs). Mi1's 13,013 silenced input synapses are R8* 12,491 + 522 from R7 types and T1.
* The mechanism proposed for abs's +18 Hz pinned-loom gain (the Tm9 flip) is inferred, not tested: abs also flips T1 (62,361 syn) and Dm9 (25,467); no run holds Tm9 at -1. Tm9's +1 rests on Davis alone (KaiR1D 107 TPM vs GluClalpha 0) against Kurmangaliyev / FCA / Davie at -1 -- the decisive control before step 8 adopts abs (round 3).
* The nine pinned-loom jobs print no device line; their cache and rule are proven by the printed sign-change counts (63,380 / 257,578 = override cache). "17 jobs" for 37 outputs means the per-mode probe seeds ran as sequential commands inside one job.

## Round 3: contested flips

Task `rule` of the round-3 workflow (docs/NT_INTEGRATION.md section 7 "Round 3"; round-2 critic follow-up 1 in
`receptor_verification.md`). Working tree at HEAD 743bda6 (nothing committed); the adopted TYPE_NT_OVERRIDE cache on both sides
(local `cache/` and the cluster's shared cache, sum|W| 121,460,584, glutamate cells 29,707 in every JSON's `nt_counts`; no
`--cache-dir`). Every number below comes from a file named next to it. Files: `scripts/build_receptor_table.py` (`contest_flip`,
`--flip-rule any|majority|off`), `flyverse/data/receptors_by_type.csv` (rebuilt; md5 0381a446107e6050e75cc87b16d7f830),
`docs/audits/receptor_rules.md` (rebuilt; section 3 documents the rule, section 6 the changes), `scripts/benchmark.py` /
`probe_loom.py` / `probe_bitter.py` (`--receptor-table PATH` -> `LIFParams.receptor_table`), `tests/test_receptor_model.py`
(32 passed; 62 passed over the four CPU files), `out/receptors_r2.csv` (the round-2 table, md5 d8557fdee6e4fad6d592870244c75432),
`out/receptors_r3_majority.csv`, `out/r3_build.log`, `out/r3_variants.txt`, `out/r3_abs_c{1,2,3}.json`, `out/r3_loom_*.txt`,
`out/r3_bitter_abs_s{0,1,2}.txt`, `out/r3_suite_table.md` (cluster batch `r3-abs-contested-e8980d`, 12 jobs, 0 failed, 8.7 min,
every job printing `cuda ok NVIDIA B200`). `nt_by_type_transcriptome.csv` and `receptor_nt_disagreements.md` are unchanged by
the rebuild (git diff empty).

### R3.1 The rule

`build_receptor_table.contest_flip` is the symmetric case of `none_contested`: a FLIP (the selected profile's fast net at the
opposite sign of the `NT_SIGN` prior -- a `+1` on glutamate; a `-1` on a +1 transmitter, which no fast group produces today)
contradicted by >= 1 other source profiling the type whose net under the SAME variant is the prior's sign falls back to
`NT_SIGN`: `fast_net*` = `flip_contested`, `fast_sign*` = the prior, `fast_gain_class*` = `none` (factor 1); the selected profile,
tier and source are kept. `mixed` and `none` sources neither contradict nor agree. `--flip-rule majority` falls back only when the
contradicting sources outnumber the other sources agreeing with the flip (`flip_contested_majority`); `--flip-rule off`
reproduces the round-2 table. The rule is evaluated per variant with each source's class / abs / nonmda net (the round-2
`alt_sources` column listed class nets only; it now carries `fast=` (class), `abs=`, `nonmda=` and `slow=`, which changes that
column's text in 1,785 rows), and the new column `flip_contested` names the contradicting sources per variant
(`class:...;abs:...;nonmda:...`). A side fix: `compare_tables` compared empty lead columns as NaN vs "" and reported
"fast_pos_lead 3170 changed" against the previous table (a CSV-to-CSV check gives 0); it now fills NaN before comparing.

### R3.2 What changed in the table (`--flip-rule any`, the shipped table)

Against the round-2 table (`docs/audits/receptor_rules.md` section 6; `out/r3_build.log`): 4,291 rows, 0 type / source / tier /
slow / lead changes; every sign change is a glutamate row going +1 -> the prior -1.

| variant | rows +1 -> `flip_contested` | which (`flip_contested` column) |
|---|---|---|
| abs | **26** | Tm9 (Davis exact KaiR1D 107.32 TPM vs GluClalpha 0.00; Kurmangaliyev / FCA / Davie exact at -1 under abs), L1 (Davis exact 45.62 vs 0.00; Kurmangaliyev exact -1 and Davie exact abs -1), and the 24 ER ring rows ER1_a/b/c, ER2_a-d, ER3a_a-d, ER3d_a-e, ER3m, ER3p_a/b, ER3w_a-c, ER4d, ER4m (FCA class Nmdar2-led 1.945 vs 0.643 against Davie class -1) -- exactly the 26 the round-2 critic counted |
| class | 36 | Tm9, L1 (Kurmangaliyev), L3 (Ozel + FCA + Davie), Mi9 (Ozel + Kurmangaliyev + FCA), Dm11 (FCA + Davie), Lawf2, Lat3 (fuzzy), MBON11 (Davie), PAM11 / PAM12 (FCA + Davie), EL, GNG629, LB2b, LoVCLo3, DNg104 / DNg34 / DNg66 / DNge138 / DNge149 / DNge150 / DNge151 / DNge152 and the 13 OA-* rows (FCA class profiles contradicted by Davie class -1) |
| nonmda | 36 | the class list minus Lat3 / MBON11 / Mi9 / PAM11, plus L4 (Ozel + Kurmangaliyev + FCA), Mi1 (FCA + Davie), Mi15 (FCA), Tm4 (Ozel + Kurmangaliyev + FCA) |

Net-call tallies over the 4,263 type rows (`out/r3_variants.txt`; all classical transmitters): `fast_net` +1 802 -> 766,
`fast_net_abs` +1 649 -> 623, `fast_net_nonmda` +1 804 -> 768; the silence labels are untouched (`none` 986, `none_single_source`
1,500, `none_contested` 19) and silenced classical edges stay 17,379 entries / 83,473 syn (all histamine) in every rule and variant.
The abs +1 rows that survive are the uncontested ones: T1 (Kurmangaliyev exact +1, Ozel / FCA / Davie `none`), Dm9, KCg-m,
KCa'b'-ap1/ap2, KCg-s1, DN1pB / DN1pA / DN1a, Lai and the rest of the critic's 14.

`--flip-rule majority` (`out/receptors_r3_majority.csv`, built in memory): abs identical to `any` (the same 26 rows -- every abs
contradiction is unopposed); class 34 (Lat3 and Lawf2 survive at 1 agreeing vs 1 contradicting); nonmda 35 (Mi15 survives, 1 vs 1).
`--flip-rule off` reproduces the round-2 sign / net / gain / source / tier columns in all 4,263 rows.

Edge-level effect on the adopted cache (`out/r3_variants.txt`, `connectome.receptor_signs` on each table; entries / |W| synapses):

| flip rule | variant | fast sign changed | glutamate flips -1 -> +1 | silenced |
|---|---|---|---|---|
| off (= round 2) | abs | 63,380 / 210,935 | 46,001 / 127,462 | 17,379 / 83,473 |
| **any (shipped)** | **abs** | **48,295 / 179,944** | **30,916 / 96,471** | 17,379 / 83,473 |
| majority | abs | 48,295 / 179,944 | 30,916 / 96,471 | 17,379 / 83,473 |
| off | class | 257,578 / 962,932 | 240,199 / 879,459 | 17,379 / 83,473 |
| any | class | 161,877 / 630,436 | 144,498 / 546,963 | 17,379 / 83,473 |
| majority | class | 170,130 / 660,155 | 152,751 / 576,682 | 17,379 / 83,473 |
| off | nonmda | 389,284 / 1,789,469 | 371,905 / 1,705,996 | 17,379 / 83,473 |
| any | nonmda | 207,023 / 1,030,946 | 189,644 / 947,473 | 17,379 / 83,473 |
| majority | nonmda | 218,407 / 1,084,115 | 201,028 / 1,000,642 | 17,379 / 83,473 |

abs on the new table changes 179,944 |W| synapses = 0.148 % (round 2: 210,935 = 0.174 %). The 30,991 synapses (15,085 entries, all
+1 -> -1) it no longer flips are Tm9 22,946 (Dm12 12,144, Mi13 5,037, Dm3a 551, Mi9 528, Mi14 479, TmY14 454, ...), L1 2,591 (Dm9 1,026,
Dm1 781, Lai 220, ...) and the ER ring 5,454 (ER1_a 1,183, ER4d 725, ER3a_b 554, ER2_c 543, ...; the compass task's 1,487 entries).
Top abs flip targets now: T1 62,361 [KaiR1D], Dm9 25,467 [Nmdar2], KCg-m 3,906 [GluRIB], DN1pB 1,535, DN1pA 1,100, DN1a 898
(`out/r3_build.log`); coverage by tier is unchanged (matched 7,654,565 edges / 31,627,236 |W| = 29.9 / 26.0 %).

### R3.3 Suite: three abs replicates on the new table (`benchmark.py --seeds 0,1,2 --receptor-model sign --receptor-net-rule abs`)

`out/r3_abs_c1.json`, `_c2`, `_c3`: **27 PASS / 0 FAIL / 2 KNOWN GAP in 3 of 3** (native backend, B200, `fast_sign_changed_entries`
48,295, runtimes 5.3 / 5.4 / 7.8 min). Against the four round-2 abs runs (27/0/2 x4; `out/rm2_abs*.json`, `out/skeptic2/rm_abs_r4.json`)
and the four off runs (26/1/2, 24/3/2, 26/1/2, 24/3/2; `out/rm2_off*.json`, `out/skeptic2/rm_off_r4.json`); full per-check table in
`out/r3_suite_table.md`:

| check | criterion | r3 abs c1 / c2 / c3 | round-2 abs x4 | off x4 |
|---|---|---|---|---|
| taste.MN9_hz | > 2 | 10.93 x3 | 10.93 x4 | 5.85 x4 |
| smell.PN_hz / KC_active | < 100 / > 0 | 7.86 / 816 x3 | 7.86 / 816 x4 | 11.19 / 1426 x4 |
| walk.GF_max_hz | < 38 | 8.52 x3 | 8.52 x4 | 4.63 x4 |
| walk.power_max_hz | < 50 | 46.10 P x3 | 46.10 P x4 | 73.18 **F** x4 |
| walk.power_sustained_hz | < 50 | 21.36 x3 | 21.36 x4 | 31.4-31.6 |
| loom.GF_peak_hz (legacy) | >= 20 | **28.10 / 28.10 / 28.04** | 28.04 / 27.05 / 27.05 / 28.04 | 43.88 / 36.94 / 36.74 / 38.00 |
| loom_escape.GF_peak_hz (demo, max of 3 seeds) | >= 33 | 48.60 / 52.21 / 48.79 | 57.21 / 66.78 / 61.35 / 73.68 | 35.18 / 31.01 F / 35.50 / 28.65 F |
| loom_escape.escapes (of 3 seeds) | >= 1 | 3 / 3 / 3 | 3 / 3 / 3 / 3 | 2 / 0 F / 1 / 0 F |
| walk_gf.p99_hz | < 38 | 24.24 / 20.78 / 17.18 | 23.32 / 29.34 / 25.91 / 26.50 | 22.20 / 19.34 / 19.97 / 22.64 |
| rotation.group_flip_hz | <= -3 | -9.37 / -9.42 / -9.85 | -9.90 / -11.89 / -12.19 / -10.93 | -7.00 / -7.17 / -7.52 / -7.98 |
| rotate.DNp20_flip_hz | < -2 | -28.87 / -31.40 / -37.56 | -36.02 / -31.40 / -31.88 / -29.85 | -13.27 / -26.23 / -26.63 / -31.72 |
| bitter.calibrated_sugar_MN9_hz | > 2 | 5.52 x3 | 5.52 x4 | 4.57 x4 |
| bitter.shiu_sugar_MN9_hz | > 50 | **139.90** x3 | 137.42 x4 | 123.54 x4 |
| bitter.shiu_sugar_bitter_MN9_hz | < 10 | 0.82 x3 | 0.47 x4 | 2.12 x4 |
| wind / odour / motion / dn / rest | -- | inside the off scatter | -- | -- |
| object.LC10a_flip_hz, compass.wedge_cells_persisting | known gaps | 0.01-0.02 G, 0 G | same | same |

Readings. (i) Every deterministic Brain-only value is bit-identical to round-2 abs except the two Shiu-rule numbers (137.42 -> 139.90,
0.47 -> 0.82): the 26 contested rows touch nothing on the taste / smell / walking / calibrated-bitter paths. (ii) The legacy
`loom.GF_peak_hz` cost is unchanged: 28.0-28.1 Hz with or without the Tm9 / L1 / ER rows, against off's 36.7-43.9 (seven abs runs
27.05-28.10, four off runs 36.74-43.88, no overlap) -- so that -10 Hz is NOT the Tm9 flip; it comes from the uncontested abs flips
(T1 / Dm9 / MB / clock) or the histamine silencings, and stays a measured cost of abs (still PASS, criterion >= 20). (iii) The demo
loom keeps escaping in 9 of 9 seeds (peaks 40.2-52.2 Hz, escape at 0.51-0.58 s after loom onset; per seed `out/r3_abs_c*.json`
sections.loom_escape) against off's 3 of 12 (21.5-35.5 Hz), but the peaks are lower than round-2 abs's 47.4-73.7 (9 of 12 round-2 abs
seeds above 52.2, the r3 maximum): the contested rows carried part of the demo loom gain, and the remaining margin over the 33 Hz
threshold is 7-19 Hz instead of 14-41. (iv) No check has a worse status in any r3 replicate than in the worst off run (`out/r3_compare.log`,
empty `worse` list); the only checks whose value is worse than every off run in every replicate are loom.GF_peak_hz (as in round 2)
and smell.KC_active (816 vs 1426, PASS either way, unchanged from round 2).

### R3.4 Pinned loom with vs without the Tm9 row (`scripts/probe_loom.py`; batch r3-abs-contested-e8980d; escape threshold `body.Flight.gf_hz` = 33 Hz on the smoothed GF rate)

| run | table (md5) | fast sign changed | DNp01 mean at t = 0.5 / 0.6 / 0.7 / 0.8 s (Hz) | escape |
|---|---|---|---|---|
| abs seed 0, new table (`out/r3_loom_abs_s0.txt`) | receptors_by_type.csv 0381a446 | 48,295 | 11 / **35** / 31 / 22 | **yes, t = 0.59 s, GF 33 Hz, TTMn 16** |
| abs seed 1, new table (`out/r3_loom_abs_s1.txt`) | same | 48,295 | 12 / **28** / 25 / 20 | no (TTMn 24 at 0.8 s) |
| abs seed 0, round-2 table (`out/r3_loom_abs_r2_s0.txt`) | out/receptors_r2_s0.csv d8557fde (Tm9 +1) | 63,380 | 14 / 35 / **37** / 29 | yes, t = 0.60 s, GF 35 Hz, TTMn 20 |
| abs seed 1, round-2 table (`out/r3_loom_abs_r2_s1.txt`) | same | 63,380 | 14 / 35 / **37** / 29 | yes, t = 0.60 s, GF 35, TTMn 20 |
| off seed 0 (`out/r3_loom_off_s0.txt`) | -- | -- | 16 / **19** / 7 / 6 | no |
| off seed 1 (`out/r3_loom_off_s1.txt`) | -- | -- | 16 / **19** / 7 / 6 | no |

The round-2 table (copied on the cluster from the origin/main checkout, md5 d8557fde = local `out/receptors_r2.csv`) reproduces round 2
line for line (`out/loom2_abs_s0.txt`: 14 / 35 / 37 / 29, escape 0.60 s at GF 35) in both seeds (byte-identical probe lines, as the
round-2 skeptic found: the seed barely decorrelates this probe); off reproduces round 2's 19 Hz in both seeds. With Tm9 / L1 / ER held
at -1 the DNp01 peak is 35 (seed 0) / 28 (seed 1) against 37 / 37 with them and 19 / 19 off, and the escape fires in one of two seeds,
at exactly the 33 Hz threshold. So the Tm9 flip is not the whole pinned-loom gain: removing it (with L1 and the ring) costs 2-9 Hz of
a +18 Hz effect, and the uncontested abs flips (T1 62,361 syn [KaiR1D], Dm9 25,467 [Nmdar2], ...) plus the histamine silencings carry
the peak from 19 to 28-35; but the pinned escape goes from certain (37 Hz, 4 of 4 runs over rounds 2-3) to marginal (1 of 2 seeds at
33 Hz). Two seeds only; the skeptic's reruns show the probe is NOT reproducible per seed on this table (two runs of seed 0: 28 and 30 Hz, no escape; six runs pooled: 28 / 28 / 30 / 30 / 35 / 35 Hz, escape 2 of 6), so the '35 vs 28' pairing is run-to-run scatter, not a seed effect; the seed-1 value is a real second sample.

### R3.5 Shiu sugar (`scripts/probe_bitter.py --receptor-model sign --receptor-net-rule abs`, 3 Poisson seeds, new table)

| seed (`out/r3_bitter_abs_s*.txt`) | Shiu sugar MN9 Hz (spikes/step) | sugar + bitter | bitter | calibrated sugar / +bitter / bitter |
|---|---|---|---|---|
| 0 | 139.9 (127) | 0.8 | 0.0 | 5.5 / 0.0 / 0.0 |
| 1 | 138.9 (116) | 0.0 | 0.0 | 4.3 / 0.0 / 0.0 |
| 2 | 131.5 (119) | 0.0 | 0.0 | 3.9 / 0.0 / 0.0 |

Round-2 abs on the round-2 table: 137.4 (100) / 131.1 (115) / 133.7 (121), calibrated 5.5 / 4.3 / 3.9 (`out/bitter2_abs_s*.txt`);
off 123.5 / 122.0 / 114.7. The calibrated values are identical to round 2's per seed; the Shiu-rule values move +2.5 / +7.8 / -2.2 Hz,
inside the 6-8 Hz seed spread of either round. Bitter suppression holds (0.0-0.8 Hz). No storm (spikes/step 116-127 vs class's 847).

### R3.6 Decision and caveats

**abs on the contested-flip table meets the adoption criterion as stated in the round-3 list: 27 / 0 / 2 in 3 of 3 replicates, and no
check's status is worse than in any off run in any replicate.** Quantitatively the same two values sit below off in every replicate as
in round 2 -- loom.GF_peak_hz 28.0-28.1 vs 36.7-43.9 (PASS, >= 20) and smell.KC_active 816 vs 1,426 (PASS) -- and neither moved
when the 26 contested rows were removed, so the Tm9 row explains neither. What the contested rows did carry is part of the loom gain:
the pinned peak 37 -> 35 / 28 Hz (escape 2/2 -> 1/2 at the threshold) and the demo peaks 47-74 -> 40-52 Hz (escape still 9/9 vs off
3/12). The table now rests on 14 uncontested abs +1 rows (T1 / Dm9 / KC / clock / Lai) and 17,379 two-source histamine silencings;
the flips it makes are contradicted by no profiled source.

Caveats: three suite replicates and two pinned-loom seeds; the pinned escape at 33.0 Hz is a threshold coincidence, not a margin;
`--flip-rule majority` would change nothing under abs (it matters only for Lat3 / Lawf2 under class and Mi15 under nonmda); the r3
JSONs' `config.cache_dir` is the run directory's `cache` symlink to the cluster's shared cache (glutamate 29,707 cells = the
override cache), not a `--cache-dir` path; `alt_sources` changed format (1,785 rows), which any parser of that column must follow;
the step-8 stop-gap retirement and the default change (`LIFParams.receptor_model = 'sign'`, `receptor_net_rule = 'abs'`) are the next
task, not this one.

## Round 3: adoption

Task `adopt` of the round-3 workflow (the round-2 critic's follow-up 2; docs/NT_INTEGRATION.md "Round 3"). Working tree at HEAD 743bda6
(nothing committed). Every number below comes from a file named next to it; the cluster batch is `r3-adopt-63e3de` (13 jobs, 0 failed,
25.2 min, every job printing `cuda ok NVIDIA B200`, the run directory's `cache` symlink to the cluster's shared override cache: every JSON
records `nt_counts.glutamate` 29,707 and 167,106 neurons; log `out/r3_adopt_cluster.log`). The comparison was generated by a scratch script
(`compare_r3_adopt.py`, CPU) into `out/r3_adopt_compare.log` and `out/r3_adopt_suite_table.md`.

### A.1 What changed in the code

* `flyverse/brain.py`: `LIFParams.receptor_model = "sign"` (was `None`) and `receptor_net_rule = "abs"` (was `"class"`); the comment block
  records the adoption. `None` remains selectable (`RECEPTOR_MODELS` unchanged) and every other receptor field keeps its default
  (`receptor_nt_class_fallback False`, `receptor_table None` = `flyverse/data/receptors_by_type.csv`, md5 0381a446107e6050e75cc87b16d7f830,
  the round-3 contested-flip table).
* `flyverse/connectome.py` `receptor_signs`: a graph whose `neurons` frame has no `nt` / `type` (or `sign`) column -- the synthetic graphs of
  `tests/test_control.py`, `test_world.py`, `test_nt_readout.py`, which now reach the lookup through `LIFParams()` -- gets no match: every
  entry keeps `sign(W.data)` at tier `fallback`. Without this 15 of the 30 tests in those three files failed (first failure
  `test_control.py::SubsetTests::test_weights_invariant_with_custom_parameters`: `'DataFrame' object has no attribute 'nt'`). On the cached connectome the function is unchanged (the pinned hashes below were computed before the edit and reproduce
  after it).
* `tests/test_receptor_model.py`: the three tests that used `LIFParams()` as the "off" reference now say `receptor_model=None` explicitly
  (`test_off_is_byte_identical`, `test_zero_gains_cost_nothing_and_equal_sign_gain`, `CachedConnectomeTests.test_off_identical...`, the last
  pinned to `receptor_net_rule="class"` which it was written for); two new tests in `CachedConnectomeTests`:
  `test_default_is_sign_abs_and_none_is_selectable` (defaults, `_receptor_key`, a Brain under `None` carries no lookup) and
  `test_none_reproduces_previous_weights_byte_for_byte`, which pins md5 hashes of `brain._shaped_weights` (sorted CSR data + indices + indptr)
  computed with EXPLICIT settings before the default changed (scratch `hash_weights.py`, local cache: W.data md5 e015d9d4007c4d2e71b604036c5e72e9,
  nnz 25,578,600, sum|W| 121,460,584, glutamate cells 29,707): `receptor_model=None` -> `2e276b30b6117c1f62688b01775eda6b` (= `LIFParams()`
  before the change, verified in the same run), `sign`/`abs` -> `f0d145d1bb81b446ebc51f89ded7bd4b` (= `LIFParams()` after the change);
  the two differ on exactly 48,295 entries = 30,916 sign flips + 17,379 zeroed, the rule task's edge counts; the hash assertions are skipped
  on a cache that is not the adopted override cache (nnz / sum|W| / glutamate-cell guard), the structural assertions always run.
  `python -m pytest tests/test_receptor_model.py -q`: 34 passed (32.3 s); with `test_nt_readout.py`, `test_world.py`, `test_control.py`
  (`SDL_VIDEODRIVER=dummy`): 64 passed.
* `scripts/batch_sustain.py`: `--receptor-model {default,off,sign}` (default `default` = leave `LIFParams` alone; `off` sets
  `receptor_model=None` on every `LIFParams` built afterwards, the `probe_object_sweep.patch_receptor` pattern) and `--receptor-net-rule`; the
  run prints `receptor model <m> (<rule>); fast sign changed on N of 25,578,600 entries` and writes `receptor` into the JSON. CPU smoke test
  (batch 1, 12 frames): `off` -> `None (None); 0 entries`, default -> `sign (abs); 48,295`.
* `scripts/audit_nt.py` / `docs/audits/nt_audit.md`: one paragraph after "Convention audited" stating the receptor model in force by default
  and what it changes on top of the presynaptic convention (48,295 entries: 30,916 flipped = 96,471 |W| synapses, all glutamate; 17,379 silenced
  = 83,473 synapses, all histamine; 0 sign-0 entries un-silenced, so every count of the audit is unchanged). Regenerated: `git diff` = 2 inserted
  lines, nothing else moved.
* `docs/audits/receptor_rules.md`: NOT regenerated -- its sections 4a-4h are functions of the table and the cache, neither of which this task
  changed (the rule task rebuilt both; a second build was byte-identical).
* Not changed (not this task's files): `scripts/benchmark.py`, `probe_loom.py`, `probe_bitter.py`, `probe_object_sweep.py`,
  `probe_figure_ground.py` keep `--receptor-model` default `"off"`. Consequence, measured below: `benchmark.py`, `probe_object_sweep.py` and
  `probe_figure_ground.py` treat `off` as "leave `LIFParams` alone" (their `_apply_receptor` / `patch_receptor` are no-ops), so a no-flag run
  NOW RUNS THE DEFAULT sign/abs but records `config.receptor.model: None` (benchmark) / `config.mode: "off"` (object sweep) in its JSON;
  `probe_loom.py` and `probe_bitter.py` set `receptor_model=None` explicitly on `off`, so a no-flag run of those two is a true off run and the
  default has to be spelled out (`--receptor-model sign --receptor-net-rule abs`, as done here). The owner of those scripts should give
  `--receptor-model` a `default` choice and record the LIFParams actually used (open question 1 below).

### A.2 Suite: three no-flag replicates on the default (`benchmark.py --seeds 0,1,2`, nothing else)

`out/r3_default_{1,2,3}.json` (runtimes 6.7 / 6.8 / 7.7 min) plus `out/r3_default_flagged.json` (the same command with
`--receptor-model sign --receptor-net-rule abs` spelled out, 6.8 min, JSON `fast_sign_changed_entries` 48,295): **27 PASS / 0 FAIL / 2 KNOWN GAP
in 3 of 3 (4 of 4 with the flagged run)**. Against the four off runs of round 2 (`out/rm2_off*.json`, `out/skeptic2/rm_off_r4.json`: 26/1/2,
24/3/2, 26/1/2, 24/3/2), no check has a worse status in any default replicate than in the BEST off run (strict) or the WORST off run (lenient):
both lists empty (`out/r3_adopt_compare.log`). Full per-check table in `out/r3_adopt_suite_table.md`; the checks that move:

| check | criterion | default 1 / 2 / 3 (no flags) | flagged | r3 abs c1-c3 | off x4 |
|---|---|---|---|---|---|
| taste.MN9_hz | > 2 | 10.93 x3 | 10.93 | 10.93 x3 | 5.85 x4 |
| smell.PN_hz / KC_active | < 100 / > 0 | 7.86 / 816 x3 | same | same | 11.19 / 1426 |
| walk.power_max_hz | < 50 | 46.10 P x3 | 46.10 | 46.10 x3 | 73.18 **F** x4 |
| walk.power_sustained_hz | < 50 | 21.36 x3 | 21.36 | 21.36 | 31.4-31.6 |
| loom.GF_peak_hz (legacy) | >= 20 | **28.10 / 31.90 / 28.04** | 27.05 | 28.10 / 28.10 / 28.04 | 43.88 / 36.94 / 36.74 / 38.00 |
| loom_escape.GF_peak_hz (max of 3 seeds) | >= 33 | 53.43 / 51.37 / 46.87 | 47.35 | 48.60 / 52.21 / 48.79 | 35.18 / 31.01 F / 35.50 / 28.65 F |
| loom_escape.escapes (of 3) | >= 1 | 3 / 3 / 3 | 3 | 3 / 3 / 3 | 2 / 0 F / 1 / 0 F |
| walk_gf.p99_hz | < 38 | 16.40 / 22.59 / 26.59 | 21.31 | 24.24 / 20.78 / 17.18 | 19.34-22.64 |
| rotation.group_flip_hz | <= -3 | -9.81 / -10.05 / -8.94 | -8.82 | -9.37 / -9.42 / -9.85 | -7.00 to -7.98 |
| rotate.DNp20_flip_hz | < -2 | -22.02 / -31.72 / -29.60 | -39.48 | -28.87 / -31.40 / -37.56 | -13.27 to -31.72 |
| bitter.calibrated_sugar_MN9_hz | > 2 | 5.52 x3 | 5.52 | 5.52 | 4.57 |
| bitter.shiu_sugar_MN9_hz / +bitter | > 50 / < 10 | 139.90 / 0.82 x3 | same | same | 123.54 / 2.12 |
| object.LC10a_flip_hz, compass.wedge_cells_persisting | known gaps | 0.00-0.01 G, 0 G | same | same | same |

Readings. (i) Every deterministic Brain-only value of the no-flag runs is bit-identical to the flagged run and to the rule task's three abs runs
(18-19 of 29 checks identical to `r3_abs_c1.json`; the 10-11 that differ are the chaotic room / native-backend sections: loom_escape,
walk_gf, rotation, rotate, wind, odour, motion.min_dsi, object.LC10a), and the taste / walk / Shiu values are the abs values (10.93 / 46.10 /
139.90), not the off values (5.85 / 73.18 / 123.54): the default IS in force in a no-flag run although the JSON header says `model: None`.
(ii) The demo loom escapes in 12 of 12 seeds (per seed 40.7-53.4 Hz over the twelve seeds of the three no-flag and the flagged run, escape 0.52-0.57 s after loom onset; `sections.loom_escape.seeds`)
against off's 3 of 12 (21.5-35.5 Hz), the round-3 abs 9 of 9 (40.2-52.2) and round-2 abs 12 of 12 (47.4-73.7). (iii) The legacy
`loom.GF_peak_hz` cost stays: 27.05-31.90 over the four runs here (the 31.90 of replicate 2 is the first abs-weights value above 28.1 in
11 runs over rounds 2-3: 27.05-28.10 in the other ten) against off 36.74-43.88 -- still no overlap, still PASS (>= 20), still the one
measured cost. (iv) smell.KC_active 816 vs 1426 (PASS either way) is the other value below every off run, unchanged since round 2.

### A.3 Probes on the default

* **Object sweep** (`scripts/probe_object_sweep.py --seed 0 / 1`, no receptor flag = the default; `out/r3_obj_default_s{0,1}.json/.txt`,
  JSON `config.mode` says `off`, see A.1): FAIL in both seeds, as in all 17 previous runs of the protocol (docs/audits/object_sweep.md):
  LC11 best-cell (ball / none) drive 0.25 / 0.28 and 0.23 / 0.24 mV (off s0/s1 0.26 / 0.26, 0.23 / 0.25; abs 0.25 / 0.24, 0.30 / 0.25),
  LC10a 0.42 / 0.48 and 0.42 / 0.38 (off 0.40 / 0.43, 0.40 / 0.51; abs 0.45 / 0.44, 0.54 / 0.49), no LC11 / LC10a cell > 1 Hz in any
  condition, LPLC2 cells > 1 Hz 5 / 6 and 5 / 7 (off 7 / 6, 6 / 4; abs 8 / 9, 10 / 8); population mean |drive| per sweep LC11 1.71-2.05 vs
  1.69-1.94 (none). Inside the off / abs spread on every field.
* **Figure-ground** (`scripts/probe_figure_ground.py --seed 0`, no receptor flag = the default; `out/r3_fg_default_s0.csv/.txt`, 112 types,
  159 apple columns): the whole figure_z column correlates with the two round-2 abs seeds at Spearman 0.948 / 0.973 (Pearson 0.972 / 0.985)
  and with the two off seeds at 0.882 / 0.897 (0.941 / 0.955) -- the default run is an abs run. Per type (z; figure in rate units):
  Mi4 -0.83 (abs -1.84 / -0.11; off +1.21 / +1.14), Tm5Y +0.66 (abs +0.65 / +0.64; off -0.25 / -0.03), Mi1 -8.94 (abs -11.30 / -9.45; off
  -8.65 / -8.77), T1 +6.66 (abs +9.47 / +7.94; off +4.15 / +4.48), T5a +1.92 fig +0.048 (abs +1.93 / +1.99, fig +0.053 / +0.054; off +0.85 /
  +0.74, fig +0.017 / +0.014), Tm9 +0.74 (abs +1.08 / +1.20; off +0.08 / +0.31), T4a -0.45 (abs -0.09 / -0.60; off +1.70 / +2.10).
  The largest departure from both abs seeds is L1: z +14.49 against abs +18.91 / +18.74 and off +17.12 / +16.41, with the figure value
  itself at +0.0152 vs +0.0154 / +0.0159 (abs) and +0.0163 / +0.0158 (off) -- the z moved with the background scatter, not the signal;
  C2 -4.86 sits inside abs's own seed spread (-8.63 / -5.69). One seed only; the abs seed-to-seed spread (Mi4 -1.84 vs -0.11) is the
  scatter to read it against, and the default sits inside it on the listed types.
* **Pinned loom** (`probe_loom.py --receptor-model sign --receptor-net-rule abs --seed 0`, spelled out because `off` is a true off there;
  `out/r3_loom_default_s0.txt`): same table (48,295 changed entries), escape at t = 0.59 s in both this run and the rule task's
  `out/r3_loom_abs_s0.txt`, at GF 34 Hz / TTMn 10 here vs 33 / 16 there; DNp01 at 0.5 / 0.6 / 0.7 / 0.8 s = 12 / 31 / 28 / 16 vs 11 / 35 / 31 / 22;
  spikes/step differ by 1-4 in seven of the 0.1-s lines. So the probe is not bit-reproducible across jobs on this table (the round-2
  skeptic found it reproducible per seed on the round-2 table); the escape at the 33 Hz threshold holds in 2 of 2 runs of seed 0
  (seed 1 gave 28 Hz and no escape in the rule task). Off: 19 Hz, no escape (`out/r3_loom_off_s{0,1}.txt`).
* **Bitter** (`probe_bitter.py --receptor-model sign --receptor-net-rule abs --seed 0,1,2`; `out/r3_bitter_default_s{0,1,2}.txt`): every
  value identical to the rule task's `out/r3_bitter_abs_s*.txt` (only the table path in the header differs): Shiu sugar 139.9 (127) /
  138.9 (116) / 131.5 (119) Hz, sugar + bitter 0.8 / 0.0 / 0.0, bitter 0.0; calibrated 5.5 / 4.3 / 3.9, 0.0, 0.0 (off 123.5 / 122.0 / 114.7
  and 4.6 / 5.6 / 4.5).

### A.4 Sustain sweep: 16 flies x 5 min, program cx, apple, fence, default vs off

`scripts/batch_sustain.py --batch 16 --seeds 0..15 --program cx --fruit apple --fence --minutes 5 --cuda-graphs --cuda-kernels --event-driven
--cuda-sparse torch` twice, `out/r3_sustain_default.json` (receptor `sign` / `abs`, 48,295 changed) and `out/r3_sustain_off.json`
(`--receptor-model off`: `None`, 0 changed); 30,000 frames = 300 s each, wall 1,409 / 1,414 s (3.41 / 3.39 fly-s per wall-s on a shared
B200), start energy 0.4, same 16 environment seeds, one batched brain RNG per run.

| per fly (n = 16) | default | off | Mann-Whitney p |
|---|---|---|---|
| meals | 0 in 16 / 16 flies | 0 in 16 / 16 | 1.00 |
| energy at 300 s (min_energy) | 0.000 in 16 / 16 | 0.000 in 16 / 16 | 1.00 |
| energy at 10 / 30 / 60 s (mean) | 0.350 / 0.249 / 0.095 | 0.350 / 0.250 / 0.097 | -- |
| first frame with every fly at energy 0 | t = 80 s | t = 80 s | -- |
| hops (take-offs) | mean 1.50, sd 0.97, median 1, range 0-3 (total 24) | mean 0.19, sd 0.54, median 0, range 0-2 (total 3; 14 flies 0) | 0.0003 |
| path (m) | 3.918 +- 0.065 | 3.825 +- 0.088 | 0.001 |
| final distance to fruit (cm) | 23.1 +- 8.0 (12.5-39.2) | 19.1 +- 7.4 (7.2-33.9) | 0.17 |
| mode fractions | exploring 0.652, surging 0.209, searching 0.071, casting 0.068 | 0.601, 0.240, 0.082, 0.077 | -- |

The meals / energy distribution is not worse under the default -- it is identical, and degenerate: with `--energy 0.4` and the metabolism's
drain the energy reaches 0 in every fly of both runs at t = 80 s (the same 0.005 / s slope at 10, 30 and 60 s) and no fly of either run
eats in 5 min, so this assay cannot rank the two models on feeding; it would need a longer horizon or a higher start energy (open
question 3). What it does show is a behavioural difference: the default takes off 24 times across 16 flies against 3 under off
(p = 0.0003), walks 2.4 % farther (p = 0.001) and ends slightly farther from the fruit (n.s.). The hops are the room-demo counterpart of the
demo-loom escapes (`loom_escape.escapes` 12/12 vs 3/12) -- the descending escape pathway is more excitable on the abs weights
(`walk.GF_max_hz` 8.52 vs 4.63, `walk_gf.p99_hz` inside the off range): spontaneous take-offs in a room with no looming object are a cost
to watch, not a criterion of this task (the criterion named meals / energy). One batch per condition, 16 flies each; no replicate of the
batch itself.

### A.5 Decision

**The default stays `receptor_model = "sign"`, `receptor_net_rule = "abs"`.** The keeping criterion holds on every clause: (a) 27 / 0 / 2 in
3 of 3 no-flag suite replicates with no check's status worse than in any of the four round-2 off runs (strict and lenient lists empty); (b)
the object sweep (2 seeds, FAIL as in all 17 prior runs, every LC11 / LC10a field inside the off / abs spread) and the figure-ground probe
(1 seed, Spearman 0.95-0.97 with the two abs seeds, every listed type inside abs's seed spread, L1 z lower with an unchanged figure
value) are within their scatter; (c) the sustain sweep's meals / energy distribution is identical to off (0 meals, energy 0 at 80 s in
32 / 32 flies), i.e. not worse. `LIFParams(receptor_model=None)` reproduces the pre-round-3 weights byte for byte (pinned hash).

Costs carried into the record: legacy `loom.GF_peak_hz` 27.05-31.90 vs off 36.74-43.88 (PASS); smell.KC_active 816 vs 1426 (PASS); more
spontaneous take-offs in the room (24 vs 3 in 16 x 5 min); the pinned-loom escape at the threshold (GF 33-34 Hz) rather than with margin.
Caveats: the two probe JSON headers misreport the model (A.1); the pinned loom is not bit-reproducible across jobs on this table; one seed
per probe and one batch per sustain condition; the sustain assay is degenerate for feeding at 5 min from energy 0.4.

Open questions for the next round: (1) the five scripts' `--receptor-model off` semantics (benchmark / object / figure-ground: "leave the
default"; loom / bitter: "true off") and the JSON headers that record the flag rather than the LIFParams used; (2) step 8 (LPi x4, GF x0.3,
AL LN override) under the new default, with the spontaneous take-off rate as a watch value; (3) a feeding-capable sustain protocol (longer
than 80 s of energy, or `--energy` above 0.4) before meals / energy can rank models; (4) the source of the legacy loom -10 Hz (T1 / Dm9 /
histamine silencings), never isolated.

### Corrections (round-3 verification)

* Provenance: `glutamate 29,707` is recorded in the four benchmark JSONs of batch r3-adopt-63e3de only; the object-sweep and sustain JSONs carry no nt_counts (they inherit the shared run-directory cache).
* A.2's list of checks that differ from r3_abs_c1 should include `loom.GF_peak_hz` (3 of 4 runs); the 4th round-3 replicate (out/sk3_abs_c4.json) widens the demo-loom range to 40.2-55.3 Hz, overlapping round-2 abs (min 47.4), so no maximum should be used as a bound.
* Pinned loom on the round-3 table: round-2 table 37 Hz with escape in 5 of 5 runs; new table 28-35 Hz with escape in 2 of 6 (GF 33-34 at the 33.0 Hz threshold); the spread is run-to-run.
* A.1's bullet on `probe_object_sweep.py` was stale within the round: that script now has `--receptor-model {default,off,sign}` (default 'default') and applies 'off' explicitly; the same fix is now in benchmark.py, probe_figure_ground.py and screen_rotation.py (below).
* **The headline caveat found by the round-3 critic:** `scripts/benchmark.py`'s legacy `walk` and `motion` sections built their optic lobe without the receptor lookup, so `walk.*` (including `walk.power_max_hz`, the one FAIL -> PASS), the legacy `loom.GF_peak_hz` (the one quantified cost) and `motion.*` were measured with the Brain under abs and the rate optic lobe under NT_SIGN -- 8,833 of the default's 179,944 changed synapses (KC, DN1 clock, OA silencings) in force and the 44,463 optic entries absent. Every room section, `room_demo`, `batch_sustain` and the probes ran the model as shipped. Fixed (the sections now pass `receptor=` / `receptor_gain=` like `fly.py`), and `--receptor-model off` -- which had become a silent no-op after the default change -- now sets `receptor_model=None` explicitly while `default` leaves LIFParams alone; the JSON header records the LIFParams used. The re-score (round 4, item 1) decides whether 27/0/2, walk.power_max 46.10 and the -10 Hz loom cost are properties of the shipped model; until then they are not to be quoted as such. Structurally the -10 Hz legacy-loom cost cannot come from T1 / Dm9 / the histamine silencings (absent in that section); it is a Brain-side effect of the KC / DN1 / OA entries.

## Round 4: take-offs and feeding

Round-3 item: "the one behavioural change no check scores -- spontaneous take-offs 24 vs 3 in 16 flies x 5 min -- was measured once and
should be replicated with a feeding-capable sustain before the default is relied on by the sustain / RL work"
(`receptor_verification.md` round-3 critic, assessment and follow-up 3). This section settles it with 48 flies per condition over three
independent brain RNGs, plus a fixed-seed rerun pair that measures the run-to-run scatter the round-3 single batch could not.

### S.0 Protocol and provenance

`scripts/batch_sustain.py --batch 16 --program cx --fruit apple --fence --minutes 5 --energy 0.9 --cuda-graphs --cuda-kernels
--event-driven --cuda-sparse torch --seed <k> --seeds <16 env seeds>`, six jobs in one cluster batch (run dir
`$CLUSTER_RUNS/r4-sustain-f2857e`, 6 jobs, 0 failed, 44.2 min; every job log prints `cuda ok NVIDIA B200` and
`device=cuda`), then a two-job fixed-seed replicate batch (`r4-sustain-rep-c6d643`, 2 jobs, 0 failed, 25.7 min). No `--cache-dir`:
every job used the cluster's shared `TYPE_NT_OVERRIDE` cache. 30,000 frames = 300 simulated s per job; start energy 0.9 (round 3 used
the 0.4 default, at which every fly of both conditions sat at energy 0 from t = 80 s and nothing ate).

| job | brain seed (`--seed`) | environment seeds | flag | header line printed | wall s | fly-s/wall-s |
|---|---|---|---|---|---|---|
| `out/r4_sustain_default_1.json` | 0 | 0-15 | (none) | `receptor model sign (abs); fast sign changed on 48,295 of 25,578,600 entries` | 1,792.7 | 2.68 |
| `out/r4_sustain_default_2.json` | 1 | 16-31 | (none) | same | 1,793.0 | 2.68 |
| `out/r4_sustain_default_3.json` | 2 | 32-47 | (none) | same | 2,553.2 | 1.88 |
| `out/r4_sustain_off_1.json` | 0 | 0-15 | `--receptor-model off` | `receptor model None (None); fast sign changed on 0 of 25,578,600 entries` | 1,018.0 | 4.71 |
| `out/r4_sustain_off_2.json` | 1 | 16-31 | `--receptor-model off` | same | 2,556.9 | 1.88 |
| `out/r4_sustain_off_3.json` | 2 | 32-47 | `--receptor-model off` | same | 2,565.9 | 1.87 |
| `out/r4_sustain_default_1b.json` | 0 | 0-15 | (none) -- rerun of batch 1 | same as default | 1,471.1 | 3.26 |
| `out/r4_sustain_off_1b.json` | 0 | 0-15 | `--receptor-model off` -- rerun of batch 1 | same as off | 1,471.3 | 3.26 |

**Flag semantics confirmed before the runs** (`batch_sustain.py` lines 33-50, 84-86; local CPU smoke test
`--batch 1 --seeds 0 --program cx --fruit apple --fence --minutes 0.002 --energy 0.9 --device cpu`,
`out/r4_smoke_{default,off}.json`): `--receptor-model default` (the default value) leaves `LIFParams` untouched and the run prints
`sign (abs); 48,295`; `--receptor-model off` sets `receptor_model=None` and prints `None (None); 0`. Both JSONs record
`receptor = {model, net_rule, fast_sign_changed_entries}` taken from the constructed `LIFParams`, not from the flag, so the condition
labels above are checkable in the files. No change to `batch_sustain.py` was needed. Each batch pair shares its environment seeds and
its brain seed, so `default_k` vs `off_k` is seed-matched; the three batches differ in both brain RNG and environment seeds and are
therefore three independent replications of the contrast, not three samples of one.
Generator: `scripts/compare_sustain_runs.py` (logs `out/r4_sustain_compare.log`, `out/r4_sustain_replicate.log`,
`out/r4_sustain_energy.log`, `out/r4_hopcheck_power.log`). It reproduces the round-3 batch exactly from
`out/r3_sustain_{default,off}.json` (hops 24 vs 3, U 225.5, two-sided asymptotic p 7.73822e-05, exact 1.06239e-04), i.e. the
round-3 numbers as the verification record recomputed them, not the `0.0003` the round-3 text printed.

### S.1 Hops (spontaneous take-offs): per batch

Per fly over 300 s; no looming object is ever presented in these rooms and `escape_gating` is off, so every airborne transition is
spontaneous. p = two-sided Mann-Whitney over the 16 vs 16 flies of that seed-matched pair (asymptotic / exact, scipy 1.17.0).

| batch | brain seed | env seeds | default total (mean +- sd) | off total (mean +- sd) | U | p asym | p exact |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0-15 | 21 (1.313 +- 1.078) | 1 (0.063 +- 0.250) | 233.5 | 9.40e-06 | 1.87e-05 |
| 2 | 1 | 16-31 | 26 (1.625 +- 1.258) | 7 (0.438 +- 0.727) | 198.5 | 4.92e-03 | 7.49e-03 |
| 3 | 2 | 32-47 | 27 (1.688 +- 1.401) | 3 (0.188 +- 0.403) | 213.5 | 4.62e-04 | 9.05e-04 |
| **pooled 48 v 48** | 0,1,2 | 0-47 | **74 (1.542 +- 1.237)** | **11 (0.229 +- 0.515)** | **1914.5** | **1.35e-09** | **3.94e-09** |

Per-fly vectors (default / off): batch 1 `[1,3,1,0,1,4,3,1,1,1,1,1,1,1,0,1]` / `[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]`; batch 2
`[0,2,2,3,0,1,1,0,2,3,1,3,4,2,2,0]` / `[0,0,2,1,0,0,1,0,1,2,0,0,0,0,0,0]`; batch 3 `[0,1,0,2,0,2,4,1,4,0,2,2,4,2,1,2]` /
`[0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0]`. Pooled histograms (flies with 0,1,2,3,4 hops): default `[10,17,11,5,5]`, off `[39,7,2,0,0]`;
fraction of flies with >= 1 hop 0.792 vs 0.188, with >= 2 hops 0.438 vs 0.042.

**The excess replicates across all three brain RNGs.** Default exceeds off in 3 of 3 seed-matched pairs at p < 0.01 each, and the
per-batch means do not overlap between conditions (default 1.313 / 1.625 / 1.688, range 0.375; off 0.063 / 0.438 / 0.188, range 0.375;
Kruskal-Wallis across the three batches within a condition H 0.994 p 0.608 for default, H 3.540 p 0.170 for off -- no detectable
brain-seed effect within either condition). Pooled rate 1.542 vs 0.229 hops per fly per 5 min = 5.14e-03 vs 7.64e-04 hops per fly-second,
a factor **6.73**. Round 3's single batch (24 vs 3 at energy 0.4) is inside both pooled distributions.

**Run-to-run scatter at fixed seeds** (`r4-sustain-rep-c6d643`, brain seed 0 and environment seeds 0-15, the identical command rerun):
default 21 -> 17 hops (1.313 -> 1.063 per fly), off 1 -> 0. The batched rollout is **not** deterministic (event-driven CUDA atomics;
`docs/BATCH_SIM.md` says so, and meals moved 2 -> 1 and mean final energy 0.058 -> 0.000 across the same default pair), but the
within-condition scatter (17-21 default, 0-1 off) is an order of magnitude smaller than the between-condition gap. Pooled over the two
seed-0 default batches vs the two seed-0 off batches (32 v 32): 1.188 +- 0.998 vs 0.031 +- 0.177, U 900.0, p asym 2.92e-09,
exact 2.07e-08.

### S.2 Feeding, energy, path, distance: pooled 48 v 48

| per fly (n = 48 each) | default | off | U | p asym | p exact |
|---|---|---|---|---|---|
| hops | 1.5417 +- 1.2370 (total 74) | 0.2292 +- 0.5153 (total 11) | 1914.5 | 1.35e-09 | 3.94e-09 |
| meals | 0.2292 +- 0.4722 (total 11; 2 / 5 / 4 per batch) | 0.1667 +- 0.3766 (total 8; 1 / 4 / 3) | 1204.0 | 0.577 | 0.707 |
| final energy (t = 300 s) | 0.0478 +- 0.1337 | 0.0055 +- 0.0353 | 1253.0 | 0.124 | 0.463 |
| minimum energy over the run | 0.0120 +- 0.0551 | 0.0038 +- 0.0238 | 1202.0 | 0.387 | 0.718 |
| path (m) | 3.6765 +- 0.1674 | 3.6003 +- 0.1203 | 1566.0 | 2.45e-03 | 2.21e-03 |
| final distance to nearest fruit (cm) | 20.96 +- 11.98 | 22.51 +- 11.44 | 941.0 | 0.123 | 0.123 |

Per-batch means, default then off: meals 0.125 / 0.313 / 0.250 vs 0.063 / 0.250 / 0.188; final energy 0.058 / 0.023 / 0.063 vs
0.000 / 0.015 / 0.001; path 3.683 / 3.687 / 3.659 vs 3.604 / 3.599 / 3.598 (default longer in 3 of 3, and in the fixed-seed rerun pair
3.666 vs 3.570; per-batch p 0.052 / 0.080 / 0.169, only the pooled test clears 0.05); distance 20.88 / 20.67 / 21.33 vs
23.10 / 23.85 / 20.58 (default nearer in 2 of 3; n.s. pooled). Mode fractions, mean over 48 flies: default exploring 0.501, surging
0.300, searching 0.115, casting 0.078, feeding 0.0071; off 0.498 / 0.308 / 0.112 / 0.079 / 0.0036.

**The feeding assay is no longer degenerate but still does not rank the two models.** At `--energy 0.9` the mean energy trace is
0.853 / 0.756 / ~0.62 / ~0.50 / ~0.35 at t = 10 / 30 / 60 / 90 / 120 s in both conditions (the drain is a body-model constant, identical
to three decimals at 10 and 30 s in all six jobs); the first flies reach 0 between t = 120 and 180 s, and at t = 300 s 13-15 of 16 flies
are at 0 under default and 15-16 of 16 under off (off batch 1 has every fly at 0 from t = 270 s). 19 meals were eaten in total
(11 default, 8 off) against 0 in round 3, so flies do find and eat the apple -- but a meal is a rare event (0.17-0.23 per fly per 5 min),
it fluctuates run-to-run at fixed seeds (2 -> 1), and the Mann-Whitney on meals, final energy and minimum energy returns p 0.12-0.72.
Ranking models on feeding needs either a longer horizon (the 300 s window ends after most flies have starved) or a metabolism whose
drain does not consume 0.9 of energy in ~150 s; `--energy 0.9` fixes only the round-3 "nothing ever eats" degeneracy.

Round-3 statements that do **not** replicate: "ends slightly farther from the fruit" (round 3 default 23.1 vs off 19.1 cm, n.s.; round 4
default 20.96 vs off 22.51 cm, n.s. -- the sign flips, so that was noise). Round-3 statements that do: the take-off excess, and
"walks farther" (+2.1 % here, p 2.4e-03, same direction in 3 of 3 batches and in the rerun pair; +2.4 % p 1.4e-03 in round 3).

### S.3 Proposed spontaneous-hop check for `benchmark.py` (proposal only -- nothing added)

The cost is real, replicated and seed-matched, so it should be scored rather than noted in prose. **The existing instrument cannot score
it.** `sec_walk_gf` (`scripts/benchmark.py` lines 516-535) already counts `voluntary_takeoffs` with `sim.flight.gf_hz = 1e9`, i.e. exactly
spontaneous hops, and it is **0 in all 13 round-2 / round-3 suite JSONs** -- `out/r3_default_{1,2,3}.json`, `r3_default_flagged.json`,
`r3_abs_c{1,2,3}.json`, `sk3_abs_c4.json`, `sk3_default_4.json`, `sk3_offflag.json`, `rm2_off{,_r2,_r3}.json`,
`skeptic2/rm_off_r4.json` -- under both models, because 15 fly-seconds at the default's 5.14e-03 hops/fly-s expects 0.077 hops. The
section needs fly-seconds, which means the batched rollout.

Proposal: a new opt-in section `hops` built on `BatchSim` (the same object `batch_sustain.py` drives), reporting
`hops.per_1000_fly_s` plus the raw total, with the reference taken from the off runs of S.1:

* **Reference (from the off condition, three brain RNGs, 14,400 fly-s):** `0.76 hops per 1,000 fly-s` (11 hops / 48 flies / 300 s;
  per batch 1, 7, 3, and 0 in the fixed-seed rerun). Shipped default for comparison: `5.14 per 1,000 fly-s` (74 hops; per batch 21, 26,
  27, and 17 in the rerun). Round-1 `receptor_model='full'` as the ceiling: 31 voluntary take-offs in 15 s of one fly =
  2,070 per 1,000 fly-s (`receptor_verification.md`, `verify:score` claim 9).
* **Two candidate entries, because one bound cannot do both jobs** (Poisson power table in `out/r4_hopcheck_power.log`):
  1. a storm guard both models pass -- `Ref(0.8, "<", 50, "10", note="spontaneous take-offs per 1,000 fly-s in a room with no looming
     object; off 0.76, sign/abs 5.14, receptor_model full 2,070")`. It separates every sane model from the round-1 storms by ~40x and
     would have caught `sign+gain` / `full` in round 1.
  2. a `gap=True` entry that records the 6.73x excess without turning the suite red --
     `Ref(0.8, "<", 2.0, "10", gap=True, note="the sign/abs default hops 6.7x more often than off (5.14 vs 0.76 per 1,000 fly-s,
     48 flies x 5 min x 3 brain RNGs, p 1.4e-09); KNOWN GAP until the DN-excitability cost is paid down")`. Under the existing `Ref`
     semantics the shipped default then prints KNOWN GAP, not FAIL, exactly like `object.LC10a_flip_hz`.
  A single strict bound (e.g. `< 2.0` without `gap`) would make the shipped default FAIL and move the suite to 27/1/2; that is a
  default-policy decision for the round-4 owner, not something this section takes.
* **Cost of the instrument** (Poisson power at the two measured rates, best total-count bound at each size):
  16 flies x 300 s = 4,800 fly-s -> E 24.7 vs 3.7, bound 11, P(default passes) 7e-04 / P(off fails) 1.5e-03, ~25-43 min wall;
  16 x 150 s = 2,400 fly-s -> E 12.3 vs 1.8, bound 6, 0.017 / 0.011, ~12-21 min;
  16 x 60 s = 960 fly-s -> E 4.9 vs 0.7, bound 3, 0.130 / 0.038 -- too weak.
  So the check needs >= ~2,400 fly-s, i.e. 12-21 min on a shared B200, more than the whole native suite (9.6 min). The recommendation is
  therefore a **separate opt-in section** (`--sections hops`, excluded from the default run and from `--fast`) rather than a 30th member
  of the scored set, with `scripts/batch_sustain.py --batch 16 --minutes 2.5` as the reference implementation and the numbers above as its
  reference; promote it into the default suite only if the suite's time budget grows.

### S.4 What this settles and what it leaves open

Settled: the round-3 take-off finding is not a one-batch artefact. The `sign` / `abs` default takes off spontaneously 6.73x as often as
the presynaptic-sign model (5.14 vs 0.76 hops per 1,000 fly-s; 74 vs 11 hops over 48 flies x 5 min per condition; p 1.35e-09 asymptotic /
3.94e-09 exact), the excess appears in 3 of 3 independent brain RNGs at p < 0.01 each with non-overlapping per-batch means, and the
within-condition run-to-run scatter at fixed seeds (17-21 vs 0-1 hops) is far smaller than the gap. The 2 % longer path replicates; the
round-3 distance-to-fruit difference does not. Feeding at `--energy 0.9` produces meals (19 across 96 fly-rollouts vs 0 at
`--energy 0.4`) but cannot rank the models (p 0.58 meals, 0.12 final energy) and still ends with almost every fly starved.

Open: (1) which of the default's 8,833 Brain-side synapses drives the excess -- the same KC / DN1 / OA entries the round-3 critic
narrowed the legacy-loom cost to; a hold-DN experiment would attribute it (round-4 item 2); (2) whether the excess is GF / DNp
excitability specifically (`walk.GF_max_hz` 8.52 vs 4.63 and 12/12 vs 3/12 demo-loom escapes point that way) or a general motor-drive
effect -- the mode fractions differ by less than 0.008, so it is not a change in the search program's structure; (3) whether a
feeding-capable protocol exists at all under the present metabolism (S.2); (4) the check in S.3 is proposed, not added.

## Round 4: re-score with the fixed benchmark (the evidence statement that supersedes A.5)

`scripts/benchmark.py`'s legacy walk / motion sections now build the optic lobe with the receptor lookup and
`--receptor-model off` sets `receptor_model=None` explicitly; the JSON header records the LIFParams used. Batch
`r4-rescore-680005` (default x3, off x3, holdKC x2, holdDN1 x2; fetched md5-verified) plus the skeptic's fourth
replicates (`out/sk4_default_4.json`, `out/sk4_off_4.json`); table by `scripts/compare_suite_runs.py --round4`
(`out/r4_compare.log`, `out/r4_suite_table.md`).

# Round-4 re-score: benchmark suite with the fixed walk / motion sections and an explicit off

| check | criterion | r4 default x3 (fixed benchmark) | r4 off x3 | r3 default x3 (half-applied) | r2 off x4 | holdKC x2 | holdDN1 x2 |
|---|---|---|---|---|---|---|---|
| rest.spikes_per_step | < 5 | 0.00 P / 0.00 P / 0.00 P | 0.00 P / 0.00 P / 0.00 P | 0.00 P / 0.00 P / 0.00 P | 0.00 P / 0.00 P / 0.00 P / 0.00 P | 0.00 P / 0.00 P | 0.00 P / 0.00 P |
| taste.MN9_hz | > 2 | 10.93 P / 10.93 P / 10.93 P | 5.85 P / 5.85 P / 5.85 P | 10.93 P / 10.93 P / 10.93 P | 5.85 P / 5.85 P / 5.85 P / 5.85 P | 10.93 P / 10.93 P | 10.93 P / 10.93 P |
| smell.PN_hz | < 100 | 7.86 P / 7.86 P / 7.86 P | 11.19 P / 11.19 P / 11.19 P | 7.86 P / 7.86 P / 7.86 P | 11.19 P / 11.19 P / 11.19 P / 11.19 P | 11.64 P / 11.64 P | 10.64 P / 10.64 P |
| smell.KC_active | > 0 | 816.00 P / 816.00 P / 816.00 P | 1426.00 P / 1426.00 P / 1426.00 P | 816.00 P / 816.00 P / 816.00 P | 1426.00 P / 1426.00 P / 1426.00 P / 1426.00 P | 1155.00 P / 1155.00 P | 1109.00 P / 1109.00 P |
| dn.DNa02_L_leg_asym_hz | > 0.3 | 2.58 P / 2.58 P / 2.58 P | 2.58 P / 2.58 P / 2.58 P | 2.58 P / 2.58 P / 2.58 P | 2.58 P / 2.58 P / 2.58 P / 2.58 P | absent / absent | absent / absent |
| dn.MDN_top_hz | < 250 | 153.00 P / 153.00 P / 153.00 P | 153.00 P / 153.00 P / 153.00 P | 153.00 P / 153.00 P / 153.00 P | 153.00 P / 153.00 P / 153.00 P / 153.00 P | absent / absent | absent / absent |
| dn.DNp09_top_hz | < 250 | 152.00 P / 152.00 P / 152.00 P | 152.00 P / 152.00 P / 152.00 P | 152.00 P / 152.00 P / 152.00 P | 152.00 P / 152.00 P / 152.00 P / 152.00 P | absent / absent | absent / absent |
| walk.GF_max_hz | < 38 | 4.61 P / 4.61 P / 4.61 P | 4.63 P / 4.63 P / 4.63 P | 8.52 P / 8.52 P / 8.52 P | 4.63 P / 4.63 P / 4.63 P / 4.63 P | 8.41 P / 8.41 P | 4.72 P / 4.72 P |
| walk.power_max_hz | < 50 | 79.47 F / 79.47 F / 79.47 F | 73.18 F / 73.18 F / 73.18 F | 46.10 P / 46.10 P / 46.10 P | 73.18 F / 73.18 F / 73.18 F / 73.18 F | 55.63 F / 55.63 F | 59.04 F / 59.04 F |
| walk.power_sustained_hz | < 50 | 37.89 P / 37.89 P / 37.89 P | 31.44 P / 31.58 P / 31.44 P | 21.36 P / 21.36 P / 21.36 P | 31.58 P / 31.58 P / 31.44 P / 31.58 P | 23.68 P / 23.68 P | 30.75 P / 30.75 P |
| loom.GF_peak_hz | >= 20 | 50.38 P / 50.38 P / 50.86 P | 40.13 P / 38.16 P / 40.83 P | 28.10 P / 31.90 P / 28.04 P | 43.88 P / 36.94 P / 36.74 P / 38.00 P | 49.01 P / 49.01 P | 48.72 P / 53.76 P |
| loom.escape_cm | notnone 0 | 3.50 P / 3.50 P / 3.50 P | 3.50 P / 3.50 P / 3.50 P | 3.50 P / 3.50 P / 3.50 P | 3.50 P / 3.50 P / 3.50 P / 3.50 P | 3.50 P / 3.50 P | 3.50 P / 3.50 P |
| rotate.DNp20_flip_hz | < -2 | -32.24 P / -34.04 P / -31.90 P | -16.89 P / -32.09 P / -27.55 P | -22.02 P / -31.72 P / -29.60 P | -13.27 P / -26.23 P / -26.63 P / -31.72 P | -36.29 P / -45.89 P | -27.79 P / -34.68 P |
| motion.min_dsi | >= 0.1 | 0.23 P / 0.23 P / 0.23 P | 0.17 P / 0.17 P / 0.17 P | 0.17 P / 0.17 P / 0.17 P | 0.17 P / 0.17 P / 0.17 P / 0.17 P | absent / absent | absent / absent |
| motion.correct_directions | == 8 | 8.00 P / 8.00 P / 8.00 P | 8.00 P / 8.00 P / 8.00 P | 8.00 P / 8.00 P / 8.00 P | 8.00 P / 8.00 P / 8.00 P / 8.00 P | absent / absent | absent / absent |
| loom_escape.GF_peak_hz | >= 33 | 49.39 P / 49.92 P / 52.95 P | 32.26 F / 34.47 P / 33.95 P | 53.43 P / 51.37 P / 46.87 P | 35.18 P / 31.01 F / 35.50 P / 28.65 F | absent / absent | absent / absent |
| loom_escape.escapes | >= 1 | 3.00 P / 3.00 P / 3.00 P | 0.00 F / 1.00 P / 1.00 P | 3.00 P / 3.00 P / 3.00 P | 2.00 P / 0.00 F / 1.00 P / 0.00 F | absent / absent | absent / absent |
| walk_gf.p99_hz | < 38 | 22.28 P / 18.31 P / 20.35 P | 22.52 P / 23.59 P / 17.81 P | 16.40 P / 22.59 P / 26.59 P | 22.20 P / 19.34 P / 19.97 P / 22.64 P | absent / absent | absent / absent |
| rotation.group_flip_hz | <= -3 | -9.61 P / -10.18 P / -10.39 P | -7.15 P / -7.08 P / -7.06 P | -9.81 P / -10.05 P / -8.94 P | -7.00 P / -7.17 P / -7.52 P / -7.98 P | absent / absent | absent / absent |
| object.LC10a_flip_hz | abs>= 1.0 | 0.00 G / 0.01 G / -0.00 G | -0.00 G / 0.00 G / 0.01 G | 0.00 G / 0.00 G / 0.00 G | 0.00 G / 0.00 G / 0.01 G / 0.00 G | absent / absent | absent / absent |
| bitter.calibrated_sugar_MN9_hz | > 2 | 5.52 P / 5.52 P / 5.52 P | 4.57 P / 4.57 P / 4.57 P | 5.52 P / 5.52 P / 5.52 P | 4.57 P / 4.57 P / 4.57 P / 4.57 P | 5.52 P / 5.52 P | 5.52 P / 5.52 P |
| bitter.calibrated_sugar_bitter_MN9_hz | < 1 | 0.00 P / 0.00 P / 0.00 P | 0.00 P / 0.00 P / 0.00 P | 0.00 P / 0.00 P / 0.00 P | 0.00 P / 0.00 P / 0.00 P / 0.00 P | 0.00 P / 0.00 P | 0.00 P / 0.00 P |
| bitter.shiu_sugar_MN9_hz | > 50 | 139.90 P / 139.90 P / 139.90 P | 123.54 P / 123.54 P / 123.54 P | 139.90 P / 139.90 P / 139.90 P | 123.54 P / 123.54 P / 123.54 P / 123.54 P | 139.90 P / 139.90 P | 129.27 P / 129.27 P |
| bitter.shiu_sugar_bitter_MN9_hz | < 10 | 0.82 P / 0.82 P / 0.82 P | 2.12 P / 2.12 P / 2.12 P | 0.82 P / 0.82 P / 0.82 P | 2.12 P / 2.12 P / 2.12 P / 2.12 P | 0.00 P / 0.00 P | 0.82 P / 0.82 P |
| wind.DNp18_flip_hz | >= 15 | 46.61 P / 45.18 P / 46.21 P | 45.22 P / 45.93 P / 44.40 P | 45.21 P / 45.15 P / 45.60 P | 46.50 P / 43.87 P / 47.21 P / 45.91 P | absent / absent | absent / absent |
| wind.DNp33_flip_hz | <= -15 | -49.68 P / -49.50 P / -49.79 P | -50.24 P / -50.26 P / -49.58 P | -49.71 P / -50.00 P / -50.20 P | -49.76 P / -49.89 P / -50.10 P / -49.99 P | absent / absent | absent / absent |
| odour.apple_channel_8cm_hz | >= 10 | 17.44 P / 17.44 P / 17.42 P | 17.54 P / 17.42 P / 17.47 P | 17.55 P / 17.35 P / 17.44 P | 17.50 P / 17.42 P / 17.50 P / 17.48 P | absent / absent | absent / absent |
| odour.apple_channel_clean_hz | <= 6 | 4.28 P / 4.63 P / 4.47 P | 4.42 P / 4.36 P / 4.36 P | 4.67 P / 4.50 P / 4.49 P | 4.44 P / 4.58 P / 4.40 P / 4.42 P | absent / absent | absent / absent |
| compass.wedge_cells_persisting | >= 6 | 0.00 G / 0.00 G / 0.00 G | 0.00 G / 0.00 G / 0.00 G | 0.00 G / 0.00 G / 0.00 G | 0.00 G / 0.00 G / 0.00 G / 0.00 G | absent / absent | absent / absent |

**Result.** Fully applied, the default scores **26 PASS / 1 FAIL / 2 KNOWN GAP in 11 of 11 draws** (r4_default x3,
sk4_default_4, r4_ntmaj x3 -- W byte-identical to the shipped cache -- retire_r4 baselines x3, sk4_retire baseline),
the FAIL always `walk.power_max_hz` (79.4650 in 15 draws, 80.0928 in 3; bound < 50). Off: 26/1/2 x3 and 24/3/2 x1 in
round 4 (walk.power_max 73.1827 in 7/7; r4_off_1 also fails loom_escape). The adoption criterion -- no check worse
in status than any off run -- **holds** (strict and lenient lists empty), but the tally equals off's best rather than
exceeding it. Off reproduces the round-2 off values bit for bit on the 15 bit-stable checks, so off is off again.

**The round-3 headline numbers were half-applied artefacts and are void:** 27/0/2; walk.power_max 46.10 PASS (fully
applied 79.47 FAIL, 6.3 Hz *worse* than off, both FAIL); the "-10 Hz legacy loom cost" (fully applied 50.4-54.3 vs
off 35.0-40.8 Hz: a +10-14 Hz *gain*); walk.GF_max 8.52 (4.61 vs off 4.63); motion.min_dsi 0.17 -> 0.23 (T4a 0.42
vs 0.17). Unchanged and bit-stable: taste.MN9 10.93 vs 5.85, KC_active 816 vs 1426, Shiu 139.90 vs 123.54,
calibrated 5.52 vs 4.57; demo loom escapes 3/3 at 47.6-52.9 Hz vs 0-1/3 at 32.3-38.2.

**Attribution (hold tables).** Neither the KC flips (2,834 entries / 5,018 syn) nor the DN1 clock flips (875 /
3,533) carry taste 5.85 -> 10.93 (10.9342 under both holds); DN1 carries 65 % of Shiu (129.27 of the 123.54 ->
139.90); walk.power_max is non-monotone in the number of applied flips (79.47 / holdKC 55.63 / holdDN1 59.04 /
off 73.18) and cannot be attributed additively; KC_active 816 / 1155 / 1109 / 1426 shows the groups interact.
After both holds 123 entries / 282 syn of Brain-side change remain, so the taste rise is most plausibly a
fan-in-normalisation effect of the 171,111 optic-side synapses inside the Brain (untested).

**Where the default stands.** Justified in kind (expression-derived sign, contested flips removed, no fitted
parameter, byte-for-byte reversible: `receptor_rules.md` section 3), and "not worse in status than off"; **not**
"better on the suite". Measured advantages: demo loom escapes, taste, Shiu, legacy loom GF, direction selectivity.
Measured costs: KC_active 816 vs 1426; walk.power_max +6.3 Hz (both FAIL); and the room take-off excess
(section below), ~40 % of which are GF escape jumps on room optic flow from a higher walking-GF tail -- the one
unscored cost, and the one open threat to the default. The sentence "27/0/2 in 10 of 10 runs" is withdrawn wherever
it appears (A.5 above, NOTES, NT_INTEGRATION).

### Corrections to "Round 4: take-offs and feeding" (verify:exp:sustain)

* "Spontaneous" is refuted: `batch_body.step()` launches by the GF escape route (GF >= 33 Hz) OR the voluntary route
  (wing power >= 50 Hz for 0.3 s) and `hops` counts both. At seed 0 the default's 24 hops are 10 escape + 14 voluntary
  vs off 2 + 0 (`scripts/probe_hop_route.py`, `out/sk4_route_default_1.json`); with the escape route disabled
  (gf_hz = 1e9) 10 voluntary vs 0. Per-row walking-GF max: default median 33.0 Hz (8/16 rows at the 33 Hz threshold)
  vs off 28.5 (2/16), U 214, p 1.3e-3.
* The proposed hop check's reference (off 0.76 / default 5.14 per 1,000 fly-s) was calibrated on both routes; on the
  instrument the check would use (voluntary only) it is default 2.08 vs off 0.00 (one seed) and must be re-derived over
  >= 3 batches. A second unscored counter exists: `sec_loom_escape`'s `hops_before_loom` (both routes live).
* p_exact values are invalid under the heavy ties (hops takes 5 values; final energy is 0 in 88/96): a tie-corrected
  permutation test gives hops p 1e-5 / 4.7e-3 / 3.4e-4 per batch (conclusion unchanged), final energy p 0.072 (not
  0.463); quote the energy comparisons as "not significant, p 0.07-0.12".
* The +2.1 % path length is collinear with hops (OLS: condition p 0.22 once hops is in the model; zero-hop flies p
  0.51) and is not a separate cost. Identical-seed reruns: default 17-24 hops, off 0-2 over three runs. The seed-0
  "pooled 32 v 32" is 16 flies measured twice. Off reference over 4 batches: 0.57 per 1,000 fly-s. Wall 17-43 min.
* The round-1 "full" ceiling is >= 2,330 hops per 1,000 fly-s (31-35 in 15 s). `out/r4_sustain_energy.log` and
  `out/r4_hopcheck_power.log` are reproduced by `scripts/skeptic_sustain_energy_power.py`.

## Round 5: the take-off check

Round-4 item 3 and the sustain skeptic's corrections above: the room take-off excess (74 vs 11 hops over 48 flies x 5 min)
was counted on an instrument that could not tell a GF escape jump on room optic flow (GF >= `Flight.gf_hz` 33) from a
voluntary take-off (wing power >= 50 Hz held 0.3 s), and the proposed hop check's reference (off 0.76 / default 5.14 per
1,000 fly-s) was calibrated on both routes while the instrument it named counts one. This section makes the split
first-class, re-measures both routes over three brain RNGs with the escape route live and disabled, derives a
voluntary-only reference, and ships the scored section. **Status at the time of writing: the instrument is built and
verified; the 14-job cluster batch (R5.1) is submitted and running; its numbers, the derived references and the
pass / gap / fail verdict are filled in by `scripts/r5_fetch_and_report.sh` once the batch lands (R5.2-R5.5 below carry the
protocol and the provisional entries, not results).**

### R5.0 Instrument (built, verified on the CPU)

* `flyverse/batch_body.py`: `BatchBody.step()` already computed the two launch masks (`escape`, `voluntary`) it hands to
  `body.Flight.launch`; it now also accumulates them per row (`hops_escape`, `hops_voluntary`, int64) and keeps the last
  step's masks (`launched_escape`, `launched_voluntary`). No dynamics change: the masks, the thresholds
  (`gf_threshold` from the wing command, `takeoff_power_hz` 50, `takeoff_hold_s` 0.3, `landing_refractory_s` 1) and the
  launch calls are the ones that were there; the diff is 8 added lines. `flyverse/batch_sim.py` exposes the counters
  (`BatchSim.hops_escape` / `hops_voluntary`, 12 added lines) and zeroes them for the rows a `reset` selects; they are
  measurement bookkeeping, not checkpoint state (a resumed run counts from the resume, as `batch_sustain`'s `hops`
  always did; `state_dict` version unchanged).
* Verification, CPU only: `scripts/check_hop_route_bookkeeping.py` drives a 9-row `BatchBody` as `tests/test_batch_sim.py`
  does (random motor samples, a forced GF burst of 80 Hz on row 6 at steps 0 and 200, forced wing power 60 Hz on row 7 whose
  hold starts at 0.29 s) next to a copy stepped with the scalar `Flight.maybe_takeoff` / `Flight.step` / `Locomotion.step`,
  400 steps, both fence settings: every fly state equal to the reference (tolerance 2e-12), airborne transitions
  `[0,0,0,0,0,0,2,1,0]` = escape `[0,0,0,0,0,0,2,0,0]` + voluntary `[0,0,0,0,0,0,0,1,0]` on both settings (the second row-6
  escape is the one after the 1 s landing refractory). `tests/test_batch_sim.py`: 8 passed, 4 GPU-opt-in skipped, 200
  subtests, on the CPU (`CUDA_VISIBLE_DEVICES=""`).
* `scripts/batch_sustain.py`: per-fly rows carry `hops` (airborne transitions, unchanged), `hops_escape`, `hops_voluntary`,
  `gf_max_walk_hz` (the row's maximum GF over frames it began on the ground -- the quantity the flight model compares with
  `gf_hz`), `gf_max_hz`, `power_max_hz`, `airborne_frac`; the header carries `flight` (gf_hz, takeoff_power_hz,
  takeoff_hold_s, landing_refractory_s), `fly_s`, the totals, the two rates per 1,000 fly-s, `gf_max_walk_median_hz` and
  `rows_gf_at_threshold`. New `--gf-hz` overrides every row's `Flight.gf_hz` before the loop (1e9 = voluntary route only,
  `benchmark.sec_walk_gf`'s trick; recorded in `options.gf_hz` and `flight.gf_hz`); it lives in `main()`, not
  `add_options`, so `scripts/probe_hop_route.py` (which defines its own `--gf-hz`) is untouched. The progress line prints
  the running escape / voluntary totals; the script warns if the split ever disagrees with the transition count. Local CPU
  smoke (`--batch 2 --seeds 0,1 --minutes 0.002 --energy 0.9 --device cpu`, `out/r5_smoke_default.json`,
  `out/r5_smoke_off_nogf.json` with `--receptor-model off --gf-hz 1e9`): headers `sign (abs); 48,295` / `None (None); 0`,
  `flight.gf_hz` 33 / 1e9, the new keys present.
* `scripts/benchmark.py`: opt-in section `hops` (letter j). `select_sections` excludes it from `all` and `new`
  (`OPTIONAL = ["hops"]`) and drops it under `--fast` with a printed note (no fast variant: 960 fly-s cannot separate the
  two rates, S.3). `sec_hops` builds `BatchSim(16, c=ctx.c, seed=0, seeds=0..15, start=(-0.15,0.15,0.75), program="cx",
  fruit_set="apple", fence=True)` under `ctx.patched_params()` (so `--receptor-model off` reaches it as
  `receptor_model=None`; the section records the LIFParams' `receptor` beside the JSON header), native flags
  `cuda_kernels + cuda_graphs + event_driven + cuda_sparse="torch"` (batches need the torch CSR path), batch_sustain's
  per-seed headings and `energy = 0.9`, `--hops-minutes` (2.5 = 2,400 fly-s) x `--hops-batch` (16). It reports
  `hops.voluntary_per_1000_fly_s`, `hops.escape_per_1000_fly_s` and `hops.walk_gf_max_median_hz`, and stores per-row
  counts, the walking-GF maxima, the launch list (row, t, route) and `route_split_consistent`. Section selection checked:
  `all` -> 14 sections without hops, `new` -> 9 without hops, `hops` / `j` -> hops, `j` + `--fast` -> skipped with the
  note, `walk_gf,hops` -> both. Smoke (`--sections hops --eager --hops-minutes 0.002 --hops-batch 2`,
  `out/r5_hops_smoke_{default,off}.json`): table printed with three checks, JSON `config.receptor.model` sign / None,
  `sections.hops.receptor` `{sign, abs}` / `{None, abs}`. **Process note:** that 6-second smoke ran on the desktop GPU
  because `--eager` lets the demo path pick CUDA when present -- a breach of the "no GPU work on this desktop" rule
  (two smokes, out/r5_hops_smoke_{default,off}.json, 2 flies x 0.12 s each); every later local run was forced to the CPU.
* `scripts/compare_sustain_runs.py`: the split metrics and rates per 1,000 fly-s per batch and pooled, the walking-GF
  tail (median, rows at the threshold, per-batch medians), Mann-Whitney asymptotic p at full precision and 3 significant
  digits; the exact p is printed only when the pooled data carry no ties (never, for counts) and reads `n/a (ties)`
  otherwise; `--ref-proposal` (references from condition B with a Poisson pass / fail table for a 2,400 fly-s section at
  candidate bounds 0.5-10 per 1,000 fly-s) and `--benchmark-json` (re-score a `hops` section against the `REFERENCES`
  now in `benchmark.py`; the measured values do not depend on the bounds a run was scored with). Round-4 JSONs (no split
  keys) still work: hops 74 vs 11, U 1914.5, p_asym 1.35387e-09 reproduce.

### R5.1 The batch (landed: 14 jobs, 0 failed, 92.0 min; run dir r5-hops-218d81; NOTE: submitted at 03:25 local, before the 03:33 GF-damping retirement, so every number in it describes the DAMPED-gains default)

`scripts/r5_cluster_batch.sh` -> `python scripts/cluster_run.py --name r5-hops --minutes 150 <14 commands> --fetch out/`,
run dir `$CLUSTER_RUNS/r5-hops-218d81`, 14 jobs submitted 2026-09-12 10:25:37-49 UTC, all `running` on node2
from 10:25:47-49 (ids 147e2d7a1adf, bfbdcbc8efa8, 0f34ad6c03f6, 0064f5540a4a, 8055a2d7b028, cdc99c887aed, d3a57084e189,
2fa47639c38b, 5c4160d8f842, c8fcdcb7506e, df24f1b39154, 88e6fa4ac4a1, 197bed82262d, a548f1c0e669 = jobs 0-13 in the order
below). Shipped local files: the six edited / new files only (out/ is git-ignored). Every job's stdout header was read
from the run directory 3 min in: all 12 sustain jobs print `BatchSim B=16 neurons=167,106 device=cuda`, the default jobs
`receptor model sign (abs); fast sign changed on 48,295 of 25,578,600 entries`, the off jobs `None (None); 0`, the live
jobs `escape at GF >= 33 Hz`, the `--gf-hz 1e9` jobs `escape at GF >= 1e+09 Hz`; the two benchmark jobs print
`receptor model sign (abs; flag --receptor-model default)` / `None (the presynaptic-sign rule; flag --receptor-model off)`.
Console of the submitting session: `out/r5_cluster.log` (block-buffered; complete only if that session outlived the batch).

| job | command (all: `python -c 'import torch; assert torch.cuda.is_available()' && ... > out/<name>.txt; cat out/<name>.txt`) | output |
|---|---|---|
| 0-2 | `batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch --seed k --seeds <16k..16k+15>` | `out/r5_sustain_default_live_{1,2,3}.json` |
| 3-5 | same `--gf-hz 1e9` | `out/r5_sustain_default_nogf_{1,2,3}.json` |
| 6-8 | same `--receptor-model off` | `out/r5_sustain_off_live_{1,2,3}.json` |
| 9-11 | same `--receptor-model off --gf-hz 1e9` | `out/r5_sustain_off_nogf_{1,2,3}.json` |
| 12 | `benchmark.py --sections hops --json out/r5_hops_default.json` | `out/r5_hops_default.json` |
| 13 | `benchmark.py --sections hops --receptor-model off --json out/r5_hops_off.json` | `out/r5_hops_off.json` |

Completion: `bash scripts/r5_fetch_and_report.sh > out/r5_hops_report.log` fetches the run directory (`scp` of
`<runs>/r5-hops-218d81/out/.`), echoes every job's device / receptor / final lines, and prints the six comparisons
(live default vs off with `--ref-proposal`; nogf default vs off with `--ref-proposal`; default live vs nogf; off live vs
nogf; the two benchmark JSONs re-scored against the current `REFERENCES`; the identical-seed check of batch 1 against the
round-4 seed-0 runs 21 / 17 / 24 default and 1 / 0 / 2 off).

### R5.2 Escape and voluntary rates per condition -- MEASURED (out/r5_hops_report.log, recounted in out/r5_close_recount.log): live route, 3 brain RNGs x 16 flies x 300 s = 14,400 fly-s per arm: pre-retirement default 75 hops = 31 escape + 44 voluntary (per batch 25 / 25 / 25) = 5.208 / 2.153 / 3.056 per 1,000 fly-s; off 9 = 9 + 0 (3 / 4 / 2) = 0.625 / 0.625 / 0.000; Mann-Whitney 48 v 48 voluntary U 1896 p 3.98e-11, escape U 1502.5 p 1.61e-03. Escape route disabled (--gf-hz 1e9): default 30 voluntary (10 / 15 / 5) vs off 0 (0 / 0 / 0), U 1728 p 2.42e-08 -- off makes 0 voluntary take-offs in 6 of 6 batches (28,800 fly-s). (Originally written as PENDING, from R5.1, `compare_sustain_runs.py` blocks A-D)

Per batch and pooled (48 v 48) counts, rates per 1,000 fly-s, Mann-Whitney U with the asymptotic p to 3 significant
digits (no exact p: counts are tied); the voluntary rate compared between the live and the `--gf-hz 1e9` arms of the same
condition (a voluntary take-off that the live arm records as an escape because the GF crossed 33 Hz during the power hold
would show as a live-arm deficit).

### R5.3 The walking-GF tail -- MEASURED: per-fly walking-GF maximum median 32.25 Hz (default; per batch 32.96 / 32.82 / 31.16) vs 27.20 (off; 27.22 / 26.83 / 27.99), rows at or above the 33 Hz escape threshold 22/48 vs 8/48, U 1991 p 8.04e-10. In the --gf-hz 1e9 arm the recorded 'rows >= threshold' degenerates (threshold 1e9); against a fixed 33 Hz it is 24/48 vs 6/48, and the GF maxima are shifted up because no fly leaves the ground by an escape. (Originally PENDING, block A)

Per-row `gf_max_walk_hz`: median, rows at or above 33 Hz, per-batch medians, Mann-Whitney default vs off (round 4, one
seed: 33.04 vs 28.51 Hz, 8/16 vs 2/16 rows, U 214, p 1.27e-03).

### R5.4 The shipped references -- RE-DERIVED from R5.2 / R5.3 (benchmark.py): voluntary reference 0.0 (bound < 1.0, gap-style: the shipped default is a KNOWN GAP in substance, P(pass per 2,400 fly-s draw) ~ 0.10 at 2.2 per 1,000 fly-s); escape reference 0.63 (bound < 10, storm guard); walking-GF median reference 27.2 (bound < 33, can FAIL; a 150 s section reads 2-3 Hz below the 300 s value). The note ranges first shipped ('2.2-2.9', '0.5 / 2.8', '26.7-28.5 / 30.6-33.2') were written before the batch and matched no file; replaced. (Originally PROVISIONAL until R5.2)

`benchmark.py` ships three `Ref` entries for the section (session "10"). The bounds below are the ones in the file at
submission time, set from the round-4 single-seed route runs (`out/sk4_route_*.json`: default voluntary 2.08, escape
2.08; off 0.00 / 0.42 per 1,000 fly-s; GF medians 33.0 / 28.5) and to be replaced by the pooled off values of R5.2 with
the Poisson table of `--ref-proposal` (the benchmark JSONs are then re-scored with block E; the run-time status column
in `out/r5_hops_*.json` reflects the provisional bounds):

* `hops.voluntary_per_1000_fly_s`: `Ref(0.0, "<", 1.0, gap=True)` -- reference voluntary-only (off), bound 1.0 (<= 2 hops
  in the 2,400 fly-s section); `gap=True` so the shipped default prints KNOWN GAP, not FAIL, while it exceeds it.
* `hops.escape_per_1000_fly_s`: `Ref(0.5, "<", 10.0)` -- reported separately; the bound is a storm guard (round-1 `full`
  >= 2,330), both models expected to pass.
* `hops.walk_gf_max_median_hz`: `Ref(28.0, "<", 33.0, gap=True)` -- the escape threshold; the default's median sits at
  it in 1 of 7 default batches on file (33.0 was the round-4 single-seed probe; the pooled live median is 32.25, the scored 150 s section 30.1-31.4), so the entry is a plain PASS/FAIL check at 33 Hz rather than gap-style.

### R5.5 What this settles and leaves open

Settled now: the split is first-class in the batched room (R5.0), the flight-model thresholds are untouched, and the
scored section exists and runs end to end (smoke) -- the round-3 critic's "add a check" and the round-4 skeptic's
"voluntary-only reference, escape rate separate" are both implemented. Pending the batch: the rates, the references, the
seed-0 reproduction of the round-4 totals, and the decision whether the shipped default passes, gaps or fails.

## Round 5: default after the GF-damping decision

Round-4 item 4 and the round-4 critic's follow-up 3 -- adopt or decline the GF x0.3 retirement, linked to the hop cost --
are settled in `docs/audits/anti_runaway.md` "Round 5: GF damping adoption" (structure, tests, the suite table, the room
batches, the caveats). This section records what the default is after that decision and the numbers it rests on, so that
the receptor thread's evidence statement (round 4 above, "Where the default stands") reads against the shipped model.

### D.1 The default after this task

* `LIFParams.receptor_model = 'sign'`, `receptor_net_rule = 'abs'`, contested-flip table: unchanged (48,295 entries, 30,916
  glutamate flips onto iGluR targets, 17,379 two-source histamine silencings).
* `brain.DEFAULT_TYPE_PATH_GAIN = [(r"^(LC4|LPLC2)$", r"^DNp01$", 3.0)]`: **the GF x0.3 input damping is retired.** The
  previous list is `brain.GF_DAMPED_TYPE_PATH_GAIN`; `LIFParams(type_path_gain=brain.GF_DAMPED_TYPE_PATH_GAIN)` reproduces
  the round-3 / round-4 default byte for byte (md5 `f0d145d1bb81b446ebc51f89ded7bd4b`, the round-3 pin), the shipped default
  hashes `0e30e4a80cb607d4a168d1b08ebd6a40`, and `receptor_model=None` on the new gains `fcb5bec2a6c492196a622e31cdb24fc6`
  (`scripts/r5_adopt_structure.py`; pinned in `tests/test_receptor_model.py`). The two defaults differ on 21 entries, all
  onto DNp01 (SAD073 8, CL367 4, DNp70 4, GNG300 3, PVLP010 2 edges; 842 shaped |W| restored, 674 of it inhibitory); the
  receptor lookup matches none of them, so the receptor model and the gain change are independent.
* Everything else (`DEFAULT_PATH_GAIN`, `optic.DEFAULT_PAIR_GAIN` with LPi x4, the AL LN override, `TYPE_NT_OVERRIDE`, the
  anti-runaway measures) is as in round 4.

### D.2 What the shipped default scores (every number from the files named; generators `scripts/r5_adopt_report.py`, `scripts/compare_sustain_runs.py`)

**Suite** (`benchmark.py --seeds 0,1,2`, no flags, batch `r5-adopt-fb3608`, `out/r5_adopt_default_{1,2,3}.json`):
**27 PASS / 0 FAIL / 2 KNOWN GAP in 4 of 4** (the skeptic's out/r5_skeptic_default_4.json added); `walk.power_max` 48.4805 (< 50) in 4/4 and in 14/14 draws of this
configuration across three batches (round 4's `no_gf_damping` x10 + these), the FAIL of the round-4 default (79.47) gone;
no check worse in status than the round-4 default (`out/r4_default_{1,2,3}.json`) or than off (`out/r4_off_{1,2,3}.json`,
26/1/2 x2, 24/3/2 x1). Bit-identical to round 4's `no_gf_damping` on every bit-stable check: taste 10.93, KC_active 816,
PN 7.86, Shiu 139.90 / 0.82, calibrated 5.52 / 0.00, `walk.GF_max` 4.63, `power_sustained` 20.11. Moved by the
retirement: `walk.power_max` 79.47 -> 48.48, `walk.power_sustained` 37.89 -> 20.11, `loom.GF_peak` 50.4-50.9 -> 47.22
(off 38-41), `motion.min_dsi` 0.233 -> 0.241-0.246 (off 0.17). Smallest PASS margins: `motion.min_dsi` 0.141,
`bitter.calibrated_sugar_bitter` 1.00 Hz, `walk.power_max` 1.52 Hz, `odour.apple_channel_clean` 1.57 Hz. The receptor
default's measured advantages over off (demo loom escapes 3/3 vs 0-1/3 at 45-60 vs 32-34 Hz, taste, Shiu, legacy loom GF,
direction selectivity) and its measured cost (KC_active 816 vs 1426) are unchanged by the retirement. The round-4
sentence "the default is not worse than off on the suite and equal to off's best tally" becomes **"27/0/2 in 3/3, one
tally better than off's best (26/1/2), with no check worse in status"** -- the improvement is the GF-damping retirement's,
not the receptor model's (off with the damping retired was not run; round 4's `walk.power_max` non-monotonicity says the
two changes need not add).

**Room take-offs** (`batch_sustain.py --batch 16 --minutes 5 --energy 0.9`, cx / apple / fence, live escape route, brain
seeds 0, 1, 2 x environment seeds 0-47; shipped default `out/r5_adopt_sustain_live_{1,2,3}.json`, pre-retirement default
`out/r5_sustain_default_live_{1,2,3}.json` and off `out/r5_sustain_off_live_{1,2,3}.json` from the instrument task's batch
`r5-hops-218d81`; `out/r5_adopt_report.log`):

| per 1,000 fly-s (14,400 fly-s each) | shipped default | pre-retirement default | off | shipped vs pre-retirement U / p2 / p(shipped <) / p(shipped >) | shipped vs off U / p2 |
|---|---|---|---|---|---|
| take-offs, both routes | **3.89** (56 = 19 / 21 / 16) | 5.21 (75 = 25 / 25 / 25) | 0.63 (9 = 3 / 4 / 2) | 926 / 0.087 / 0.044 / 0.957 | 1786.5 / 1.7e-07 |
| escape route (GF >= 33 Hz on room optic flow) | **1.67** (24 = 7 / 11 / 6) | 2.15 (31 = 12 / 11 / 8) | 0.63 (9) | 1064 / 0.47 / 0.23 / 0.77 | 1423 / 0.012 |
| voluntary route (wing power >= 50 Hz for 0.3 s) | **2.22** (32 = 12 / 10 / 10) | 3.06 (44 = 13 / 14 / 17) | 0.00 (0) | 942 / 0.099 / 0.049 / 0.951 | 1656 / 3.2e-07 |
| walking-GF max per row, median (rows >= 33 Hz) | 31.90 Hz (19/48) | 32.25 Hz (22/48) | 27.20 Hz (8/48) | 934 / 0.11 / 0.056 / 0.945 | 1846 / 3.7e-07 |

The retirement takes the take-off rates down in both routes (point estimates 0.73-0.77x, lower in 3 of 3 seed-matched
batches for the total and the voluntary route, 2 of 3 for escapes) and removes roughly a quarter to a half of the excess over (a point estimate inside identical-seed rerun scatter -- the shipped default's seed-0 batch gives 19 hops in one run and 11 in another, U 182.5, p2 0.029 -- against an off arm that was run under the DAMPED gains) off; it does
not close it. **The room take-off excess remains the receptor default's one cost the suite does not score:** 6.2x off in
take-offs (p 1.7e-07), 32 voluntary take-offs where off makes 0, a walking-GF tail at the escape threshold in 19 of 48
rows vs off's 8. The round-4 statement "~40 % of the excess are GF escape jumps from a higher walking-GF tail" now reads,
on the split instrument: escapes are 43 % of the shipped default's take-offs (24 of 56; pre-retirement 31 of 75, 41 %),
and the tail (median 31.9 vs 27.2 Hz) is the receptor model's, not the damping's (the retirement moved it by 0.35 Hz,
p 0.11).

**The `hops` section** (`benchmark.py --sections hops`, 2,400 fly-s, one draw per arm; `out/r5_adopt_hops.json`,
`r5_hops_default.json`, `r5_hops_off.json`; bounds as shipped by the instrument task, `Ref(0.0, "<", 1.0, gap)` /
`Ref(0.5, "<", 10.0)` / `Ref(28.0, "<", 33.0, gap)`): shipped default 3 take-offs = 2 escape + 1 voluntary -> voluntary
0.42 PASS (gap closed), escape 0.83 PASS, GF median 29.27 PASS (gap closed), 3/0/0; pre-retirement 7 = 4 + 3 -> 1.25
KNOWN GAP / 1.67 / 30.11, 2/0/1; off 1 = 1 + 0 -> 0.00 / 0.42 / 25.74, 3/0/0. **The shipped default's voluntary "PASS
(gap closed)" is a one-draw result that the 300-s room batches contradict** (2.22 per 1,000 fly-s over 14,400 fly-s, 2.2x
the bound; a section at that rate expects 5.3 voluntary take-offs, P(X <= 1) = 0.031), and the pre-retirement draw is low
by the same token (7 observed vs 12.5 expected from its room rate, P = 0.070) -- either two low draws (joint ~1e-3) or
a take-off rate that is not stationary over the 300 s (the section samples the first 150 s, energy 0.9 -> ~0.1; the flies
reach energy 0 at 150-180 s). The section's owner should settle that before its references are read as calibrated; the
status of the voluntary entry for the shipped default is KNOWN GAP in substance.

### D.3 What this settles and leaves open

Settled: the GF-damping retirement is adopted on its own evidence (suite 27/0/2 x3, take-offs not worse in either route),
the receptor default stays, and the two changes are structurally independent (0 DNp01 input edges in the receptor
lookup). The link the round-4 critic asked to test -- does restoring inhibition onto DNp01 lower the walking-GF tail that
fires the 33 Hz escape? -- is answered weakly: it lowers `walk.power_max` by 31 Hz and the take-off rates by about a
quarter, and leaves the GF tail where it was.

Open, carried to the dynamics questions: (1) the room take-off excess of the receptor default (3.89 vs 0.63 per 1,000
fly-s; voluntary 2.22 vs 0) -- neither the KC / DN1 holds (round 4) nor the DNp01 inhibition (here) account for it;
(2) the `hops` section's stationarity (D.2) and the reading of its references, which are the instrument task's; (3) off
with the damping retired was not run, so "27/0/2 vs 26/1/2" is a statement about the shipped default, not about the
receptor model's contribution; (4) `default + pair_gain_lpi_x2` remains the untested combination of round 4.

## Round 5: Brain-side vs optic-side attribution

Round-4 item 2 closed negatively -- neither the KC flips nor the DN1-clock flips carry `taste.MN9` 5.85 -> 10.93 or
`smell.KC_active` 1426 -> 816 -- and left one hypothesis standing: "the taste rise is most plausibly a
fan-in-normalisation effect of the 171,111 optic-side synapses inside the Brain (untested)" (round-4 re-score section;
`receptor_verification.md` round-4 critic, follow-up 7). This section tests it two ways: a **hold pair** that splits the
default's 48,295 changed entries by the side of the brain they land on, run on the GPU, and the **fan-in normalisation
itself**, computed on the CPU. The hypothesis is refuted on both. The by-product is an attribution table: every check
whose value differs between the default and off is assigned to a side.

### E.0 The two tables (`scripts/build_hold_tables.py`, extended; `--verify` on the local `TYPE_NT_OVERRIDE` cache)

Each table is `flyverse/data/receptors_by_type.csv` with one group of rows set back to the presynaptic prior --
`fast_sign_abs = NT_SIGN[transmitter]` (glutamate and histamine -1: no +1 flip, no silencing), `fast_gain_class_abs =
none`, `fast_net_abs = 'held'` -- so the held targets keep NT_SIGN while every other entry of the default stays. The
side is read off the table's own `superclass` column (optic = `ol_intrinsic` / `ol_sensory`); the 28 synthetic
`<nt=...>` / `<superclass=...>` rows are in neither side (`receptor_signs` reads them only under `nt_class_fallback`,
off everywhere here). Only rows that actually differ from the prior are touched, so the diff is minimal and the KC /
DN1 tables of round 4 are reproduced byte for byte by the extended builder (md5 `95bf26f5c78ec4964fba0d1c1fbfccfd`,
`3ac8523efb800c6789d6d76c07233cfd`).

Verification against `connectome.receptor_signs(c, table_path=..., net_rule='abs')` on the shipped cache
(25,578,600 stored entries, sum |W| 121,460,584; `out/r5_attr_verify.log`):

| table | rows held | md5 | entries changed vs `sign(W.data)` | vs the default | changed \|W\| |
|---|---|---|---|---|---|
| `flyverse/data/receptors_by_type.csv` (shipped) | -- | `0381a446107e6050e75cc87b16d7f830` | **48,295** | -- | 179,944 |
| `out/receptors_holdBrain.csv` | 187 (11 glutamate, 176 histamine) | `d902daf5c7efd94cfedaade7a2f135f3` | **44,463** (optic only) | -3,832, 0 new | 171,111 |
| `out/receptors_holdOptic.csv` | 48 (3 glutamate, 45 histamine) | `c3baf4293f508bb39d2d42f4b87163dc` | **3,832** (Brain only) | -44,463, 0 new | 8,833 |
| `out/receptors_holdBrainGlu.csv` | 11 (glutamate) | `2e1b53f2026fb077dab06b806aaf2d1f` | 44,586 | -3,709, 0 new | 171,393 |
| `out/receptors_holdBrainHis.csv` | 176 (histamine) | `6fadd62df4d3e1a4121b9d08310ed565` | 48,172 | -123, 0 new | 179,662 |

44,463 + 3,832 = 48,295 exactly, with no entry changed by both tables and none changed by neither, so the pair is a
partition of the default's effect. The default's changed entries by postsynaptic superclass reproduce the round-4
critic's split exactly: `ol_intrinsic` 44,463 entries / 171,111 syn, `cb_intrinsic` 3,720 / 8,571,
`visual_centrifugal` 65 / 134, `visual_projection` 29 / 81, `descending_neuron` 18 / 47, `ol_sensory` 0. The last two
tables split the Brain side by transmitter and are used on the CPU in E.4: the Brain side's glutamate rows are
**exactly** the KC and DN1 groups (3,709 entries / 8,551 syn = 2,834 + 875), so `holdBrainGlu` is round 4's `holdKC`
and `holdDN1` applied together -- a condition round 4 never ran -- and `holdBrainHis` holds exactly the 123 entries /
282 syn the round-4 critic was left with.

### E.1 The batch

One batch, six jobs, run dir `$CLUSTER_RUNS/r5-attr-6aa260`, **6 jobs, 0 failed, 3.4 min**
(`scripts/r5_attr_batch.sh`, console log `out/r5_attr_cluster.log`); every job rebuilds the two tables in its own run
copy (`out/` is not shipped; `build_hold_tables.write_atomic` keeps concurrent jobs of one run directory from tearing
the file -- the in-job md5s equal the local ones) and then runs

    python scripts/benchmark.py --sections rest,taste,smell,walk,bitter,motion --seeds 0,1,2 \
        --receptor-model sign --receptor-net-rule abs [--receptor-table out/receptors_hold{Brain,Optic}.csv]

`holdBrain` x2, `holdOptic` x2 **plus a `default` and an `off` anchor in the same batch**. The anchors are not
redundant: the shipped default changed in this round (`anti_runaway.md` "Round 5: GF damping adoption", section D.1
above), so the round-4 endpoints (`walk.power_max` default 79.47 / off 73.18) were measured under different gains and
cannot anchor the walk rows. `--receptor-model sign --receptor-net-rule abs` is passed explicitly because
`--receptor-model default` returns before `--receptor-table` is applied (round-4 verification, `verify:rescore`
KEY CLAIM 4).

Provenance. `config.device` is `NVIDIA B200` in all 12 JSONs (no CPU fallback). The run copy's `flyverse/brain.py`
md5 `9caf67b228a211434f8524d8685eed3b` equals the working tree's, i.e. the post-adoption default
`DEFAULT_TYPE_PATH_GAIN = [(r"^(LC4|LPLC2)$", r"^DNp01$", 3.0)]` with the GF x0.3 damping retired; `scripts/benchmark.py`
md5 `062680bfc24d6060082e05919878872c`; the receptor table `0381a446107e6050e75cc87b16d7f830`. Each arm's JSON header
records the condition independently of the flag: `fast_sign_changed_entries` 48,295 (default) / 44,463 (holdBrain) /
3,832 (holdOptic) / `model: null` (off), with the table path.

**A second, identical batch ran by accident** (the launcher was started twice; run dir `r5-attr-05bc30`, same six
commands, 6 jobs, all exitcode 0). Rather than discard it, it is used as an independent replicate: the fetched
`out/r5_attr_*.json` are md5-verified against `r5-attr-6aa260` and the second run's JSONs are in
`out/r5_attr_dup/` (md5-verified against `r5-attr-05bc30`), its per-job logs in `out/r5_attr_dup_cluster.log`. Every
number below is quoted over **both** run directories: 4 draws per hold arm, 2 per anchor. (The duplicate's
`cluster_run` console log was overwritten by the second launch and is not recoverable; the per-job logs are.)

### E.2 The attribution table (`scripts/r5_attr_report.py`; `out/r5_attr_table.md`, `out/r5_attr_report.log`)

All draws of both run directories. `holdBrain` applies the optic side only, `holdOptic` the Brain side only.

| check | criterion | default x2 | holdBrain x4 (optic 44,463 only) | holdOptic x4 (Brain 3,832 only) | off x2 | carried by |
|---|---|---|---|---|---|---|
| `rest.spikes_per_step` | < 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no difference |
| `taste.MN9_hz` | > 2 | **10.9342** | **5.8455** | **10.9342** | **5.8455** | **Brain, 100 %** |
| `smell.PN_hz` | < 100 | 7.8612 | 11.1877 | 7.8612 | 11.1877 | **Brain, 100 %** |
| `smell.KC_active` | > 0 | **816** | **1426** | **816** | **1426** | **Brain, 100 %** |
| `walk.GF_max_hz` | < 38 | 4.6292 | 12.5174 | 13.3109 | 4.9641 | neither: both holds 2.5-2.9x *above* both endpoints |
| `walk.power_max_hz` | < 50 | 48.4805 P | 47.0012 P | 64.9147 **F** | 97.1014 / 95.5416 **F** | optic 103 %, Brain 66 % (non-additive) |
| `walk.power_sustained_hz` | < 50 | 20.1091 P | 21.3426 P | 33.5072 P | 50.5960 / 49.1802 **F** | optic 96 %, Brain 55 % (non-additive) |
| `loom.GF_peak_hz` | >= 20 | 43.5929 / 46.4969 | 50.0089 / 50.0089 / 51.2904 / 60.0354 | 31.7754 x3 / 32.0792 | 29.0034 / 27.9926 | optic 130-191 %, Brain 20-22 % |
| `loom.escape_cm` | notnone | 3.50 | 3.50 | 3.50 | 3.50 | no difference |
| `rotate.DNp20_flip_hz` | < -2 | -41.8507 / -39.8623 | -26.1967 / -31.5105 / -41.6825 / -42.6848 | -26.3840 / -26.7702 / -29.0932 / -31.4699 | -30.0512 / -30.6196 | not attributable (the holdBrain draws span every condition) |
| `motion.min_dsi` | >= 0.1 | 0.2411 | 0.2411 x2 / 0.2460 x2 | **0.1863** | **0.1863** | **optic, 100-109 %; Brain exactly 0** |
| `motion.correct_directions` | == 8 | 8 | 8 | 8 | 8 | no difference |
| `bitter.calibrated_sugar_MN9_hz` | > 2 | 5.5184 | 4.5659 | 5.5184 | 4.5659 | **Brain, 100 %** |
| `bitter.calibrated_sugar_bitter_MN9_hz` | < 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no difference |
| `bitter.shiu_sugar_MN9_hz` | > 50 | **139.8985** | **123.5394** | **139.8985** | **123.5394** | **Brain, 100 %** |
| `bitter.shiu_sugar_bitter_MN9_hz` | < 10 | 0.8178 | 2.1243 | 0.8178 | 2.1243 | **Brain, 100 %** |

Percentages are `(hold - off) / (default - off)` on the condition means. Subset tallies over these 16 checks (**not**
the 29-check suite): default 16 PASS, `holdBrain` 16 PASS, `holdOptic` 15 PASS / 1 FAIL (`walk.power_max` 64.91),
off 14 PASS / 2 FAIL in the primary run dir; 15 / 1 in the duplicate run dir and in the skeptic's (walk.power_sustained 49.18 and 49.61 PASS there; only walk.power_max fails in 3 of 3: 97.10 / 95.54 / 96.46).

**Scatter.** Nine of the sixteen checks are bit-identical across all 12 draws within their condition, so the
100 %-Brain and 0 %-Brain attributions carry no scatter at all: `taste.MN9` 10.9342 in 4 of 4 `holdOptic` draws and
5.8455 in 4 of 4 `holdBrain` draws; likewise `smell.PN_hz`, `smell.KC_active`, both calibrated-sugar checks, both Shiu
checks, `motion.min_dsi` under `holdOptic` / off, and the three `walk.*` values (`sec_walk` and `sec_motion` draw no
RNG). The endpoints also reproduce the existing corpus exactly: taste 10.9342 / 5.8455, `KC_active` 816 / 1426, PN
7.8612 / 11.1877, Shiu 139.8985 / 123.5394 and calibrated 5.5184 / 4.5659 are the round-4 values (11 default / 7 off
draws there), and the default's `walk.power_max` 48.4805, `power_sustained` 20.1091 and `GF_max` 4.6292 are round 4's
`no_gf_damping` and round 5's `r5_adopt_default` values (13/13 draws). The two checks that scatter --
`loom.GF_peak_hz` (spread 10.0 Hz within `holdBrain`) and `rotate.DNp20_flip_hz` (16.5 Hz within `holdBrain`) -- are
the two that round 4's verification already flagged as scattering; `rotate.DNp20` is therefore not attributed here,
and `loom.GF_peak` is attributed only as a direction (the four `holdOptic` draws, 31.78-32.08, sit 3.4 Hz above off's
28.0-29.0 while the four `holdBrain` draws, 50.0-60.0, sit 5-15 Hz *above* the default's 43.6-46.5).

**The answers to the seven questions asked.**

1. `taste.MN9` 5.85 -> 10.93: **the Brain side, all of it.** The 3,832 Brain-side entries alone give 10.9342 in 4 of 4
   draws; the 44,463 optic-side entries alone give 5.8455 in 4 of 4. The fan-in hypothesis is dead (E.3).
2. `smell.KC_active` 1426 -> 816 and `smell.PN_hz` 11.19 -> 7.86: **the Brain side, all of it** (816 / 7.8612 under
   `holdOptic` x4, 1426 / 11.1877 under `holdBrain` x4). This also explains round 4's puzzle -- `holdKC` 1155 and
   `holdDN1` 1109 both sat between 816 and 1426 because each held only part of one side.
3. Shiu sugar 123.54 -> 139.90 (and sugar+bitter 2.12 -> 0.82, calibrated 4.57 -> 5.52): **the Brain side, all of it.**
   Round 4's "DN1 carries 65 % of Shiu" is a statement about one group inside that side.
4. `walk.power_max` and `walk.power_sustained`: **both sides, non-additively.** Under the shipped gains off now FAILS
   at 95.54-97.10 and 49.18-50.60; the optic side alone recovers 103 % / 96 % of the way to the default and the Brain
   side alone 66 % / 55 %, summing to 169 % / 151 %. `holdOptic` is the one arm here that fails a check the default
   passes.
5. `loom.GF_peak`: **mostly the optic side** (123-179 % of the gap over three run dirs, overshooting the default; Brain-only 22-41 %) with a small Brain-side
   contribution (20-22 %).
6. `motion.min_dsi` 0.1863 -> 0.2411: **the optic side, all of it.** The Brain side moves it by exactly 0.0000 (0.1863
   bit-identical in 4 of 4 `holdOptic` draws and 2 of 2 off draws) -- as it must, since T4/T5 direction selectivity is
   built from `ol_intrinsic` input.
7. `walk.GF_max`: **neither side, and not monotone.** Default 4.6292 and off 4.9641 are 0.33 Hz apart, but *either*
   half of the receptor signs on its own puts the walking giant-fibre maximum at 12.52 (optic only) or 13.31 Hz
   (Brain only) -- 2.5-2.9x both endpoints, 22-25x the default-off difference, bit-identical in 4 of 4 draws each. The
   two sides' effects on the GF drive cancel; any partial application of the receptor signs breaks that cancellation.

### E.3 The fan-in hypothesis, tested directly (`scripts/build_hold_tables.py --fanin`, CPU; `out/r5_fanin.log`, `out/r5_fanin.json`, `out/r5_fanin_types.csv`)

The hypothesis was that the optic-side entries reach the Brain-only sections through
`Brain.__init__`'s fan-in normalisation, `tot = abs(shaped W).sum(axis=1)`,
`input_scale = clip((input_norm_ref / max(tot, 1)) ** input_norm_alpha, 0.02, 1.0)` (shipped `LIFParams`:
`input_norm_alpha` 1.0, `input_norm_ref` 5,000, `conn_cap` 60). Two facts kill it.

**(a) A sign flip cannot move a fan-in sum at all.** `brain._shaped_weights` sets
`W.data = abs(W.data) * receptor.fast_factor(gain)` and `_receptor_gain` is `None` under `receptor_model='sign'`, so
`|W.data|` is `|W_raw|` times `|fast_sign| in {0, 1}`: only an entry silenced to `fast_sign = 0` changes a row sum. Of
the default's 48,295 changed entries the 30,916 glutamate flips are weight-preserving and only the 17,379 histamine
silencings can move anything. Measured (sum of `tot` over the 166,383 cells with non-zero fan-in): off
115,282,448 -> default 115,198,976, a drop of 83,473 (int64; the float32 row-sum arithmetic cannot resolve single synapses) = the silenced |W|, of which the optic side accounts for 83,192
(`holdBrain` 115,199,256) and the Brain side for 280 (`holdOptic` 115,282,168); the two add to the whole exactly.

**(b) The optic side moves the row sums of 10,411 cells and the `input_scale` of none of them.** Because the scale is
clipped at 1.0 and every cell whose row sum the optic side moves has `tot < input_norm_ref = 5,000`, its scale is
1.000000 before and after. Across the whole graph the scale moves on **7 of 166,383 cells** under the default -- and
all 7 are Brain-side: `holdOptic` (Brain entries only) reproduces all 7, `holdBrain` (optic entries only) reproduces
**0**.

| cell | superclass | silenced input entries | `tot` off -> default | `input_scale` off -> default |
|---|---|---|---|---|
| 10011 OA-AL2i1 | visual_centrifugal | 1 | 11,003 -> 11,002 | 0.454422 -> 0.454463 |
| 10110 OA-VUMa1 | cb_intrinsic | 1 | 8,551 -> 8,550 | 0.584727 -> 0.584795 |
| 10269 OA-AL2i2 | visual_centrifugal | 1 | 6,022 -> 6,021 | 0.830289 -> 0.830427 |
| 10447 DNge138 | descending_neuron | 3 | 5,819 -> 5,809 | 0.859254 -> 0.860733 |
| 10898 AVLP476 | cb_intrinsic | 1 | 6,889 -> 6,885 | 0.725795 -> 0.726216 |
| 11466 DNge149 | descending_neuron | 4 | 5,918 -> 5,900 | 0.844880 -> **0.847458** (+0.31 %, the largest move in the graph) |
| 12151 DNge138 | descending_neuron | 2 | 5,017 -> 5,013 | 0.996612 -> 0.997407 |

**And nothing in the taste pathway moves.** Taking `sec_taste`'s own populations (sweet labellar-bristle / taste-peg
GRNs, their postsynaptic partners, MN9, MN9's presynaptic partners):

| population | cells | input entries | changed entries (default) | of those, Brain-side | cells whose `input_scale` moves |
|---|---|---|---|---|---|
| driven sweet GRNs | 144 | 4,067 | 0 | 0 | 0 |
| sweet second order | 1,220 | 275,076 | 13 (20 syn) | 13 | 0 |
| MN9 | 2 | 415 | **0** | 0 | **0** (`tot` 1,727.55 and scale 1.000000 in all four conditions) |
| MN9's presynaptic partners | 350 | 121,587 | 4 (18 syn) | 4 | 1 (DNge149, +0.0026) |

So the hypothesis is refuted twice over: the side it named moves zero fan-in scales, and MN9's own fan-in is
bit-identical under all four tables while its rate doubles. **`taste.MN9` 5.85 -> 10.93 is not a fan-in-normalisation
effect; it is carried by the Brain side's 3,832 sign changes as a dynamical effect.**

### E.4 Which Brain-side entries, then? A double dissociation on the CPU (`scripts/r5_attr_taste_cpu.py`; `out/r5_attr_taste_cpu.log`, `.json`)

`sec_taste` and `sec_smell` build `brain.Brain` on the whole connectome with no optic lobe and no rate freeze, so they
run in 13 s on this desktop's CPU and more hold tables can be afforded than one cluster batch allows. The Brain side's
3,832 entries split cleanly by transmitter: the **3,709 glutamate flips are exactly the KC and DN1 groups** (2,834 +
875; 8,551 syn) and the other **123 are histamine silencings** (282 syn) -- so `holdBrainGlu` is round 4's `holdKC`
and `holdDN1` held *together*, which round 4 never ran, and `holdBrainHis` holds exactly the residual the round-4
critic was left with. Both protocols verbatim (sweet labellar-bristle / taste-peg GRNs at 100 Hz for 600 ms; the apple
plume at the benchmark's coordinates for 800 ms), three brain seeds, `device='cpu'`:

| condition | entries applied | `taste.MN9_hz` per seed 0 / 1 / 2 | `smell.KC_active` | `smell.PN_hz` |
|---|---|---|---|---|
| off | 0 | 1.554839 / 4.341760 / 2.309808 | 1079 / 427 / 1178 | 11.5126 / 2.4370 / 4.9003 |
| default | 48,295 | 5.090923 / 4.315772 / 2.360074 | 486 / 412 / 543 | 3.6941 / 2.9017 / 4.2780 |
| `holdBrain` (optic 44,463 only) | 44,463 | **= off, every digit** | **= off** | **= off** |
| `holdOptic` (Brain 3,832 only) | 3,832 | **= default, every digit** | **= default** | **= default** |
| `holdBrainGlu` (holds KC+DN1; leaves the 123 silencings + optic) | 44,586 | **= default** | **= off** | **= off** |
| `holdBrainHis` (holds the 123; leaves KC+DN1 + optic) | 48,172 | **= off** | **= default** | **= default** |
| `holdKC` | 45,461 | = default | 1225 / 464 / 1006 | 9.9661 / 2.9307 / 8.1874 |
| `holdDN1` | 47,420 | = default | 525 / 433 / 540 | 3.9050 / 2.4012 / 4.2898 |

The dissociation is exact, in both directions, at all three seeds (`MN9_hz`, `GNG175_hz` and `frac_active` compared
together for taste; `PN_hz`, `KC_hz` and `KC_active` for smell):

* **`taste.MN9` depends on the 123 Brain-side histamine silencings and on nothing else in the receptor model.** Holding
  the 44,463 optic entries, the 2,834 KC flips, the 875 DN1 flips or all 3,709 glutamate flips together leaves every
  taste number bit-identical to the *default*; holding the 123 silencings alone makes them bit-identical to *off*.
  Round 4's "neither KC nor DN1 carries taste" is confirmed and completed: the carrier is the group the round-4 critic
  dismissed -- "the taste doubling cannot be carried by those 282 synapses of visual histamine silencing under any
  drive" is **refuted**; those 282 synapses are the whole of it. Their targets are 62 entries onto OA-AL2i3, 25 onto
  TmY14, 5 DNge138, 4 each onto s-LNv / DNge150 / VP5+Z_adPN / DNge149, 3 OA-VUMa2 and 12 single entries (AVLP476,
  DNg104, OA-AL2i1 / 2 / 4, OA-VUMa1, OA-VPM4, DNge152 x2, DNg34 x2, PPM1202), presynaptically R8p / R8_unclear / R8y /
  HBeyelet: photoreceptor and eyelet histamine onto octopaminergic (OA-AL2i, OA-VUMa, OA-VPM4) and gnathal descending
  (DNge138 / DNge149 / DNge150) cells, six of whose types are in the sugar -> MN9 pathway.
* **`smell.KC_active` / `PN_hz` depend on the 3,709 KC + DN1 glutamate flips and on nothing else** -- the mirror image
  (`holdBrainGlu` = off, `holdBrainHis` = default, both bit-exact). Within that group the KC flips carry most of it
  (`holdKC` 1225 / 464 / 1006, near off's 1079 / 427 / 1178) and the DN1 flips little (`holdDN1` 525 / 433 / 540, near
  the default's 486 / 412 / 543), and the two are not additive -- which is why round 4's single holds both landed
  between 816 and 1426 on the GPU.

Caveat on magnitudes. CPU Poisson draws are not the CUDA ones, so these absolute rates are not the suite's: the CPU
default-minus-off difference in `MN9_hz` is +1.19 Hz on the mean of three seeds and -0.03 Hz at seed 1, against the
GPU's bit-stable +5.09 Hz (10.9342 vs 5.8455 in 4 of 4 `holdOptic` / `holdBrain` draws here and in 11 / 7 draws in
round 4). What the CPU measures is not the size of the effect but **which entries the section's dynamics depend on at
all**, and that it measures exactly: bit-identity in both directions at three seeds, reproduced by an independent
rerun of the four deciding conditions at seed 0 (`out/r5_attr_taste_cpu_rerun.log`: off 1.5548 / 1079, default
5.0909 / 486, `holdBrainGlu` 5.0909 / 1079, `holdBrainHis` 1.5548 / 486 -- every digit of the first run). The 2-job
GPU confirmation is owed and cheap -- `benchmark.py --sections taste,smell,bitter --seeds 0,1,2 --receptor-model sign --receptor-net-rule
abs --receptor-table out/receptors_hold{BrainGlu,BrainHis}.csv` -- and would also settle which subgroup carries Shiu
sugar, which this section did not run (round 4: `holdDN1` 129.27 against the 123.54 -> 139.90 range, i.e. DN1 carries
65 % of Shiu and KC none, so Shiu's carrier is not taste's).

### E.5 Two things the batch settles in passing

**Off with the GF damping retired** -- listed as open in D.3 item (3) -- for these six sections: `walk.power_max`
97.1014 / 95.5416 **FAIL** and `walk.power_sustained` 50.5960 / 49.1802 **FAIL**, against the shipped default's
48.4805 / 20.1091 PASS. The retirement and the receptor model therefore interact strongly and in opposite directions on
that check: with the damping in force the default was 6.3 Hz *worse* than off (79.47 vs 73.18, both FAIL, round 4);
with it retired the default is **47.8 Hz better** than off (48.48 PASS vs 96.32 FAIL). Round 4's "the receptor default
does not repair `walk.power_max`" is a statement about the pre-retirement gains only, and its "6.3 Hz worse than off"
should not be carried forward without that condition. Two draws, six sections, no full-suite tally.

**The Shiu arm needs no fan-in argument at all.** `sec_bitter`'s `shiu` setting is
`LIFParams(adapt_jump=0, conn_cap=0, same_type_gain=1, input_norm_alpha=0, path_gain=[], type_path_gain=[])` and
`Brain.__init__` applies the fan-in normalisation only `if p.input_norm_alpha > 0`, so that arm runs with **no fan-in
normalisation in the model**; `bitter.shiu_sugar_MN9_hz` nevertheless moves 123.5394 -> 139.8985 and is 100 %
Brain-side in 4 of 4 draws. That is a second, code-level refutation of the fan-in hypothesis, independent of E.3.

### E.6 What this settles and what it leaves open

Settled.

1. **The fan-in-normalisation hypothesis is dead.** A sign flip cannot move a fan-in sum (only the 17,379 histamine
   silencings can, by 83,473 (int64; the float32 row-sum arithmetic cannot resolve single synapses) |W| in total), the optic side moves 10,411 row sums and **zero** `input_scale` values
   (every affected cell sits below `input_norm_ref` = 5,000, where the scale is clipped at 1.0), the scale moves on 7
   of 166,383 cells in the whole graph and all 7 moves are Brain-side, the largest is +0.31 %, MN9's own fan-in is
   bit-identical under all four tables, and the Shiu arm has the normalisation switched off entirely.
2. **Every check that separates the default from off is attributed to a side** (E.2): taste, both smell checks and all
   three sugar checks to the Brain side (100 %, bit-exact); `motion.min_dsi` to the optic side (100 %, the Brain side
   moving it by exactly 0); `loom.GF_peak` mostly optic (123-179 % vs 22-41 % over three run dirs); `walk.power_max` / `power_sustained` to
   both sides non-additively (103 % + 66 %, 96 % + 55 %); `walk.GF_max` to neither -- the two sides cancel there and
   either half alone raises the walking GF maximum 2.5-2.9x.
3. **Round-4 item 2 is closed positively** (E.4, CPU): the taste rise is the 123 Brain-side histamine silencings (282
   synapses onto octopaminergic and gnathal descending cells), the smell change is the 3,709 KC / DN1 glutamate flips,
   and neither group touches the other's section -- a double dissociation, bit-exact in both directions at three
   seeds. Round 4 could not see it because both of its holds were inside the glutamate group.

Open.

1. The E.4 subgroup split is CPU-measured; the 2-job GPU confirmation (`holdBrainGlu` / `holdBrainHis` on
   `taste,smell,bitter`) is owed, and with it the subgroup that carries Shiu sugar.
2. `walk.GF_max`'s non-monotonicity (4.63 default / 12.52 optic-only / 13.31 Brain-only / 4.96 off) is the
   walking-GF tail the room take-off cost rides on (D.2). Nothing here explains why the two sides cancel; it is the
   same kind of regime boundary round 4 found in the LPi scan, and it says a *partial* receptor model is worse for the
   giant fibre than either endpoint.
3. `rotate.DNp20_flip_hz` cannot be attributed at this sample size (the four `holdBrain` draws span -26.20 to -42.68,
   wider than the default-to-off contrast).
4. Why photoreceptor / eyelet histamine onto OA-AL2i3, OA-VUMa and the DNge14x cells doubles the sugar-driven MN9 rate
   is a dynamics question (an octopaminergic gain on the gustatory motor pathway), not a receptor-data one -- the
   silencings are two-source calls and stand.

Files: `scripts/build_hold_tables.py` (the `Brain` / `Optic` / `BrainGlu` / `BrainHis` groups, `--verify`, `--fanin`),
`scripts/r5_attr_batch.sh`, `scripts/r5_attr_report.py`, `scripts/r5_attr_taste_cpu.py`;
`out/receptors_hold{Brain,Optic,BrainGlu,BrainHis}.csv`, `out/r5_attr_verify.log`,
`out/r5_attr_{default_1,holdBrain_1,holdBrain_2,holdOptic_1,holdOptic_2,off_1}.{json,txt}` (run dir
`r5-attr-6aa260`) and `out/r5_attr_dup/*.json` (run dir `r5-attr-05bc30`), `out/r5_attr_cluster.log`, `out/r5_attr_dup_cluster.log`,
`out/r5_attr_table.md`, `out/r5_attr_report.log`, `out/r5_fanin.{log,json}`, `out/r5_fanin_types.csv`,
`out/r5_attr_taste_cpu.{log,json}`, `out/r5_attr_taste_cpu_rerun.log`. Every table re-derives with the same md5 in two
runs (including round 4's `holdKC` / `holdDN1`), the `--fanin` log is identical between two runs, and the test suite is
80 passed / 29 skipped / 200 subtests after the builder change.

### Corrections (closing-round verification)

* Attribution (E.*): the CPU double dissociation establishes what taste DEPENDS on (the 123 Brain-side histamine
  silencings, 282 synapses onto OA-AL2i3, TmY14, DNge138/149/150, ...), not the SIZE of the GPU's +5.09 Hz rise: the
  CPU default-minus-off is +3.54 / -0.03 / +0.05 Hz at seeds 0 / 1 / 2, and -1.44 / +0.18 / -5.65 with
  input_norm_alpha = 0. The fan-in refutation's decisive test is the skeptic's alpha = 0 taste run
  (`scripts/skeptic_attr_fanin_test.py`, out/sk_attr_fanin.log: the dissociation survives bit-exactly with the
  normalisation off); the Shiu alpha = 0 argument does not transfer to taste because Shiu's carrier is DN1, not
  taste's. "Brain side" = postsynaptic superclass outside the optic lobe: 94 of the 123 silenced entries land on
  visual centrifugal / projection cells. The 2-job GPU confirmation of holdBrainGlu / holdBrainHis on taste, smell,
  bitter is still owed (tables exist).
* Instrument (R5.*): the pre-retirement seed-0 voluntary rate is 2.92 per 1,000 fly-s (14 / 4,800), the escape rate
  2.08; section-to-section scatter of the 150 s hops section at identical settings is a factor ~2.7 (7 vs 17 hops,
  3 vs 6), so any verdict on the voluntary entry needs >= 3 draws; the "non-stationary rate" worry is not supported
  (launches start at 20 s, 8 of 17 in the first 75 s).
* Adoption (D.*): the hops section's "PASS (gap closed)" for the shipped default was one low draw (3 hops); a rerun
  gives 6 = 1 + 5 -> 2.08 KNOWN GAP. `retire_measures.py` now has a `gf_damped` configuration that restores the
  damping, so its `no_gf_damping` ablation is no longer a no-op against the baseline.

## Dynamics round 1: the room hold pair

The receptor plan closed with one cost unaccounted for: the shipped default takes off in the room 3.889 times per
1,000 fly-s against off's 0.625 (voluntary 2.222 vs 0.000, escape 1.667 vs 0.625; walking-GF median 31.90 vs 27.20 Hz,
19/48 vs 8/48 rows at the 33 Hz escape threshold), and neither the KC / DN1-clock holds nor the DNp01 inhibition
accounts for it (`receptor_verification.md` round 5, Handover item 5). Two things were wrong with that statement as
evidence: the off arm on record was run under the **damped** gains (`brain.py` md5 `5f04ee4e...` in run dir
`r5-hops-218d81`), so the room contrast crossed a default change; and the one attribution result that bears on
take-offs -- `walk.GF_max` in the pinned walk section, where **either half of the receptor signs alone raises the
walking GF 2.5-2.9x while both together cancel** -- was never checked in the room. This section runs the hold pair
in the room and adds the missing denominator (off under the shipped gains). It is a dynamics measurement on the
shipped default: no gain override anywhere, `LIFParams` untouched except for the receptor table the hold arms select.

**Result: the optic side carries most of the room take-off cost and the walking-GF tail (75-100 % of the hop excess
over four matched batches); the Brain side carries none of it; the halves do not cancel in the room as they do in
`walk.GF_max`; and off with the damping retired is the same denominator as off with it (0.486 vs 0.625 per 1,000
fly-s, 0 voluntary take-offs in both).** The asymmetry is the honest headline: *the Brain side carries none of it* is
a tight, replicated bound, *the optic side carries most of it* the matching weak one (G.5).

### G.0 What the two tables change, and what they do not touch (CPU, read-only; `out/d1_hold_structure.log`, `out/d1_hold_weights.log`)

The tables are round 5's, rebuilt byte for byte by `scripts/build_hold_tables.py --groups Brain,Optic`
(`out/receptors_holdBrain.csv` md5 `d902daf5c7efd94cfedaade7a2f135f3`, 187 rows held; `out/receptors_holdOptic.csv`
`c3baf4293f508bb39d2d42f4b87163dc`, 48 rows held), both locally and inside the run directory. `holdBrain` holds the
Brain-side rows at the presynaptic prior, so the **optic-side signs act alone**; `holdOptic` holds the optic-side rows,
so the **Brain-side signs act alone**. Against `sign(W.data)` on the shipped cache (25,578,600 stored entries, sum |W|
121,460,584; `--verify` in job 0 and the same counts locally):

| arm | entries changed | changed \|W\| | flips -1 -> +1 | silencings -1 -> 0 | `_shaped_weights` md5 |
|---|---|---|---|---|---|
| shipped default (both sides) | 48,295 | 179,944 | 30,916 (96,471) | 17,379 (83,473) | `0e30e4a80cb607d4a168d1b08ebd6a40` |
| `holdBrain` = optic side alone | 44,463 | 171,111 | 27,207 (87,920) | 17,256 (83,191) | `022894e72a0a702ceb6e0c9ccad48df1` |
| `holdOptic` = Brain side alone | 3,832 | 8,833 | 3,709 (8,551) | 123 (282) | `40fd50c7aa887a62dca585ebaca3c0dd` |
| off (`receptor_model=None`) | 0 | 0 | -- | -- | `fcb5bec2a6c492196a622e31cdb24fc6` |

44,463 + 3,832 = 48,295 with **0 entries changed by both tables and 0 by neither** -- the pair is an exact partition of
the default's effect, verified on the shaped weights as well as on the signs (the two `None` / default md5s are the
values pinned in `tests/test_receptor_model.py`). Where the changes land: the optic side is the medulla, entirely
`ol_intrinsic` -- T1 21,157 entries / 62,361 \|W\|, Dm9 5,981 / 25,467, Mi15 2,507 / 20,666, Mi4 1,818 / 20,024, Mi1
1,601 / 13,013, Dm2 2,643 / 12,682, Mi9 1,841 / 5,946, then L4 / C3 / L5 / Tm29 / T2 (presynaptically Dm6, R8y,
R8_unclear, R8p, Dm19, Dm8a/b). The Brain side is the mushroom body and the clock -- KCg-m 2,137 / 3,906, DN1pB 277 /
1,535, DN1pA 395 / 1,100, DN1a 203 / 898, KCa'b'-ap2 371 / 498, KCg-s1 68 / 273, KCa'b'-ap1 216 / 272, OA-AL2i3 62 /
131, TmY14 25 / 63 (presynaptically MBON05 1,123 entries / 2,752 \|W\|, MBON01, MBON03, LHMB1, MBON30, SLP\*).

**Neither side changes a single entry on the take-off pathway.** Counting input entries of the cells the two routes run
through: DNp01 (the giant fibre) 1,455 input entries, **0 changed**; MN9 415, 0; LC4 63,315, 0; LPLC2 102,690, 0; and 0
on each of the five formerly damped inputs (DNp70 3,196, SAD073 2,258, GNG300 1,573, CL367 1,461, PVLP010 3,299). The
most motor-adjacent changes in the whole default are 18 descending-neuron silencings worth 47 \|W\| -- DNge149 (4
entries), DNge138 (5), DNge150 (4), DNge152 (2), DNg34 (2), DNg104 (1), all `-1 -> 0` from IN27X004 / AN27X004 /
AN27X008 / GNG043 -- and they sit on the **Brain** side, i.e. in the arm that turns out to carry nothing. So whatever
the room cost is, it is not a sign change on the take-off pathway; it is an upstream state change.

### G.1 The batch

One cluster batch, 9 jobs, 0 failed, 92.0 min wall (run dir `$CLUSTER_RUNS/d1-hold-8ed117`; launcher
`out/d1_hold_batch.sh`, console log `out/d1_hold_cluster.log`, per-job logs `out/d1_hold_joblogs/`). Each job:

    python scripts/build_hold_tables.py --groups Brain,Optic [--verify] &&
    python scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence \
        --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch \
        [--receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_hold{Brain,Optic}.csv | --receptor-model off] \
        --seed k --seeds <16 environment seeds> --json out/d1_<arm>_<k+1>.json

with brain seeds 0 / 1 / 2 against environment seeds 0-15 / 16-31 / 32-47 -- the protocol, seeds and flags of the
shipped-default arm `out/r5_adopt_sustain_live_{1,2,3}.json`, which is therefore the reference here and was not
re-run. `out/` is git-ignored and not shipped, so every hold job rebuilds its own tables (atomic writes; the in-job
md5s equal the local ones). Results: `out/d1_{holdBrain,holdOptic,off}_{1,2,3}.json` + `.txt`, md5-verified against
the run directory after the fetch (9/9 match), exit code 0 in 9/9, `device=cuda` (B200) in 9/9, `gf_hz` 33 in 9/9,
and the JSON header's `fast_sign_changed_entries` reads 44,463 / 3,832 / 0 -- the condition is recorded independently
of the flag. Wall 5,404-5,423 s per job (two off jobs 2,890 s), 0.89-1.66 aggregate fly-s per wall-s under heavy
contention (39 jobs from four tasks were on the cluster).

Provenance against the two run directories this section compares with: `d1-hold-8ed117` has `flyverse/brain.py`
`9caf67b228a211434f8524d8685eed3b` (the shipped gains: `DEFAULT_TYPE_PATH_GAIN` = LC4/LPLC2 -> DNp01 x3 only),
`body.py` `1dcb3a8137dac7cb0d77bde356404f08`, `batch_body.py` `1e8be4ee0ab955af5ffb88f144d9a7f8`, `batch_sim.py`
`bfae1359a34c2bc7939a37d3c5582f12`, `flyverse/data/receptors_by_type.csv` `0381a446107e6050e75cc87b16d7f830`.
`r5-adopt-fb3608` (the shipped-default arm) is **byte-identical on all four code files**; `r5-hops-218d81` (the off arm
on record) differs only in `brain.py` (`5f04ee4e...` = the damped gains). The room JSONs still do not record
`type_path_gain`, so those md5s are the arm labels (`anti_runaway.md` caveat (1)).

`scripts/batch_sustain.py` gained the `--receptor-table` pass-through this section needed (+17 / -4 lines):
`patch_receptor(model, net_rule, table=None)` sets `LIFParams.receptor_table` whenever the receptor stage is on, so
the flag also works with `--receptor-model default` (the gap round 4 found in `benchmark.py`, whose `_apply_receptor`
returns before the table under `default`); it is refused with `--receptor-model off` (where it would be a silent
no-op) and with a missing file; and it lives in `main()`, not `add_options`, following `--gf-hz`'s precedent (R5.0), so
`scripts/probe_hop_route.py` and `scripts/profile_batch.py`, which import `add_options` / `patch_receptor`, are
untouched. The JSON header's `receptor` block now carries `table` beside `model` / `net_rule` /
`fast_sign_changed_entries`. Smoke-tested on the CPU before submission (`--batch 2 --minutes 0.02 --device cpu`
with `out/receptors_holdBrain.csv`: header `sign (abs) table out/receptors_holdBrain.csv; fast sign changed on 44,463`).

### G.2 The four arms in the room (3 batches x 16 flies x 300 s = 14,400 fly-s each; `out/d1_hold_compare.log`)

| arm | hops | escape | voluntary | per 1,000 fly-s | escape | voluntary | walking-GF median | rows >= 33 Hz |
|---|---|---|---|---|---|---|---|---|
| shipped default (both sides) | 56 | 24 | 32 | 3.889 | 1.667 | 2.222 | 31.90 | 19/48 |
| `holdBrain` = **optic side alone** | 52 | 18 | 34 | 3.611 | 1.250 | 2.361 | 32.08 | 17/48 |
| `holdOptic` = **Brain side alone** | 5 | 5 | 0 | 0.347 | 0.347 | 0.000 | 27.04 | 5/48 |
| off, shipped gains (new) | 7 | 7 | 0 | 0.486 | 0.486 | 0.000 | 26.53 | 6/48 |
| off, damped gains (on record) | 9 | 9 | 0 | 0.625 | 0.625 | 0.000 | 27.20 | 8/48 |

Per batch (brain seed 0 / 1 / 2), hops = escape + voluntary, then the batch's walking-GF median and rows at 33 Hz:

* shipped: 19 (7+12), 21 (11+10), 16 (6+10) | 31.15 / 33.32 / 31.22 Hz | 6, 9, 4 of 16
* `holdBrain`: 14 (4+10), 22 (8+14), 16 (6+10) | 31.15 / 33.01 / 31.77 Hz | 4, 8, 5 of 16
* `holdOptic`: 1 (1+0), 4 (4+0), 0 (0+0) | 27.10 / 27.14 / 26.75 Hz | 1, 4, 0 of 16
* off (shipped gains): 3 (3+0), 3 (3+0), 1 (1+0) | 26.82 / 28.20 / 26.18 Hz | 2, 3, 1 of 16
* off (damped gains): 3 (3+0), 4 (4+0), 2 (2+0) | 27.22 / 26.83 / 27.99 Hz | 3, 3, 2 of 16

Kruskal-Wallis across the three batches of an arm is n.s. for every count metric (`holdBrain` hops H 2.998 p 0.223,
voluntary p 0.676; shipped hops p 0.686), so the three brain RNGs behave as replicates of one rate, and the
`hops`-count route split agreed with the airborne-transition count in 9/9 jobs (no `WARNING` line).

### G.3 Pooled Mann-Whitney (two-sided, asymptotic, 48 rows per arm; exact p is `n/a (ties)` for counts)

| pair | hops | escape | voluntary | walking-GF max |
|---|---|---|---|---|
| `holdBrain` vs shipped | U 1118.5, p **0.800** | U 1079.0, p **0.532** | U 1222.0, p **0.577** | U 1225.0, p **0.595** |
| `holdOptic` vs shipped | U 459.0, p 5.55e-09 | U 806.0, p 7.91e-04 | U 648.0, p 3.19e-07 | U 379.0, p 1.51e-08 |
| off (shipped gains) vs shipped | U 487.5, p 2.86e-08 | U 837.0, p 2.62e-03 | U 648.0, p 3.19e-07 | U 394.0, p 2.85e-08 |
| `holdBrain` vs off (shipped) | U 1871.5, p 2.98e-09 | U 1410.5, p 1.08e-02 | U 1776.0, p 4.28e-09 | U 1974.0, p 1.75e-09 |
| `holdOptic` vs off (shipped) | U 1125.5, p **0.730** | U 1125.5, p **0.730** | U 1152.0, p **1.000** | U 1179.0, p **0.846** |
| `holdBrain` vs `holdOptic` | U 1904.5, p 4.22e-10 | U 1442.5, p 3.56e-03 | U 1776.0, p 4.28e-09 | U 1991.0, p 8.04e-10 |
| off (shipped) vs off (damped) | U 1105.0, p **0.578** | U 1105.0, p **0.578** | U 1152.0, p **1.000** | U 991.0, p **0.240** |

The optic-side arm is statistically indistinguishable from the full default on all four measures and separated from
off on all four; the Brain-side arm is the mirror image. As rates with exact Poisson 95 % intervals and as the share
of the default's excess over off that an arm reproduces:

| measure | optic side alone | Brain side alone |
|---|---|---|
| hops | 3.611 [2.697, 4.735] vs shipped 3.889 [2.938, 5.050]; count ratio 0.93x [0.64, 1.35] | 0.347 [0.113, 0.810]; ratio 0.089x [0.033, 0.205] |
| escape | 1.250 [0.741, 1.976] vs 1.667 [1.068, 2.480]; 0.75x [0.40, 1.37] | 0.347; 0.21x [0.07, 0.51] |
| voluntary | 2.361 [1.635, 3.299] vs 2.222 [1.520, 3.137]; 1.06x [0.66, 1.72] | 0.000 [0.000, 0.256]; 0.00x [0.00, 0.08] |
| share of the excess over off (shipped gains) | hops 92 %, escape 65 %, voluntary 106 %, GF-median shift 103 % | hops -4 %, escape -12 %, voluntary 0 %, GF shift 10 % |

The one number below 90 % is the escape route's 65 % (18 vs 24 hops), and it is not a difference: p 0.53 pooled, and
the escape counts per batch (4, 8, 6 vs 7, 11, 6) sit inside the rerun scatter on record for this protocol -- two
identical-seed runs of the shipped default's seed-0 batch give 19 and 11 hops (7+12 and 3+8;
`out/r5_adopt_sustain_live_1.json` vs `out/r5_skeptic_sustain_live_4.json`), so a single batch measures itself to
about a factor 1.7 and the optic-side seed-0 batch (14 = 4+10) lies between them.

### G.4 The room does not behave like `walk.GF_max` (recount of the round-5 suite JSONs: `out/d1_walk_gfmax_recount.log`)

The pinned walk section, same weights, same shipped gains, over the three round-5 run directories (5 draws per hold
arm, 3 per anchor; every arm's value bit-identical across draws):

| arm | `walk.GF_max_hz` | `walk.power_max_hz` | `walk.power_sustained_hz` |
|---|---|---|---|
| off | 4.964 | 95.542 / 97.101 / 96.457 FAIL | 49.180 / 50.596 / 49.614 |
| shipped default | 4.629 | 48.481 PASS | 20.109 |
| optic side alone (`holdBrain`) | **12.517** | 47.001 PASS | 21.343 |
| Brain side alone (`holdOptic`) | **13.311** | 64.915 FAIL | 33.507 |

That is the cancellation the handover quoted: each half alone puts the pinned walking GF at 2.5-2.9x either endpoint,
and the two together return it to the endpoint. **The room shows no trace of it.** In the room the optic half alone
reproduces the default (32.08 vs 31.90 Hz median, 17/48 vs 19/48 rows at threshold) and the Brain half alone
reproduces off (27.04 vs 26.53 Hz, 5/48 vs 6/48); the ordering is monotone in the optic half and flat in the Brain
half. So `walk.GF_max` is not a proxy for the room's escape route, and the two quantities are not even on the same
scale (pinned 4.6-13.3 Hz over one fly's walking window against 26-32 Hz for the median over 48 flies of each fly's
300 s maximum while on the ground). Two further readings of that table are worth carrying:

* `walk.power_sustained_hz` is the pinned analogue of the voluntary route's own criterion (>= 50 Hz held 0.3 s), and
  **off sits at 49.2-50.6 Hz on it** -- straddling the threshold -- while the shipped default sits at 20.1. Read as a
  prediction for the room it is simply wrong: off makes 0 voluntary take-offs in 14,400 fly-s under the shipped
  gains (0 in 43,200 fly-s over all 9 off batches now on file), and the default, whose pinned value is 2.5x lower,
  makes 32. Whatever the voluntary route rides on in the room is not the pinned section's wing-power level.
* the Brain-side arm is the only arm that fails `walk.power_max` while being at off's take-off rate in the room
  (64.9 Hz, `holdOptic`), and off fails it hardest (95.5-97.1) while being the quietest room arm. The hand-set 50 Hz
  bound (Handover item 4) therefore anti-correlates with the room take-off rate across these four arms; nothing
  should be tuned to it on the grounds that it predicts spurious take-offs.

### G.5 Answer, and what it does not settle

1. **Which side carries the excess: the optic side, most of it.** 44,463 entries / 171,111 \|W\| of medulla sign
   changes (95 % of the default's changed weight; T1, Dm9, Mi15, Mi4, Mi1, Dm2, Mi9 ...) reproduce the shipped
   default's take-off rate in both routes and its walking-GF tail (3.611 vs 3.889 per 1,000 fly-s, voluntary 2.361 vs
   2.222, median 32.08 vs 31.90 Hz; p 0.53-0.80 on every measure). A **4th matched batch** puts a number on how well
   that is measured. The closing skeptic ran all four arms in ONE submission at a brain seed / env block no batch here
   used (brain seed 3, env 48-63; 5 jobs, 0 failed, 56.2 min, run dir `$CLUSTER_RUNS/sk-d1-hold-f691cf`,
   `scripts/sk_d1_hold_verify.sh`, console `out/sk_d1_hold_cluster.log`, results
   `out/sk_d1_{shipped,holdBrain,holdOptic,off}_4.json`): shipped 19 = 3 escape + 16 voluntary (GF med 31.56), optic
   side 7 = 1 + 6 (30.29), Brain side 3 = 3 + 0 (25.95), off 4 = 4 + 0 (27.76) -- there the optic side is 0.37x the
   default on hops (binomial p 0.029). Pooled over the four matched batches per arm (19,200 fly-s each;
   `out/sk_d1_hold_pool.log`): shipped 75 = 27 + 48, optic side 59 = 19 + 40, Brain side 8 = 8 + 0, off 11 = 11 + 0,
   and the shares of the default's excess over off move from 92 / 65 / 106 / 103 % to **hops 75 %, escape 50 %,
   voluntary 83 %, GF-median shift 95 %**. Three-batch shares are single-batch-noise-dominated point estimates that
   moved 17-20 points on one added replicate, and they move the other way too: including the 4th shipped-default batch
   already on record (`out/r5_skeptic_sustain_live_4.json`, 11 hops at batch 1's seed / env) gives 104 / 83 / 113 %.
   So the supported statement is **most of it, indistinguishable from all of it at this exposure** -- these shares are
   not measured to better than a factor ~1.5 by three or four batches -- not *all of it*.

   The Brain side's bound is the strong half. The 3,832 Brain-side entries (KCg-m, the DN1 clock, OA-AL2i3, TmY14,
   and the 18 descending-neuron silencings) reproduce off (0.347 vs 0.486 per 1,000 fly-s, 0 voluntary, 27.04 vs
   26.53 Hz; p 0.73-1.00 against off), at **<= 0.21x the default on hops** (4-batch count ratio 0.107x, Jeffreys 95 %
   CI [0.049, 0.210]) and 0.000x on voluntary ([0.000, 0.053]); and off itself now makes **0 voluntary take-offs in
   52,800 fly-s over 11 batches** (`out/d1_off_{1,2,3}.json`, `out/r5_sustain_off_live_{1,2,3}.json`,
   `out/r5_sustain_off_nogf_{1,2,3}.json`, `out/sk5_sustain_off_live_1r.json`, `out/sk_d1_off_4.json`). This also
   closes, negatively, the possibility that the take-off cost is the same thing as the taste / smell dependence found
   in round 5: those are 100 % Brain-side (E.6), the take-offs are 0 % Brain-side.
2. **The escape route is not attributable to either side at this exposure; the voluntary route is.** With the 4th
   matched batch the optic side's escape excess over off is no longer significant -- U 2300.0, **p 0.095** (three
   batches: U 1410.5, p 0.011), 19 vs 11 counts, ratio to the default 0.70x [0.39, 1.26] -- and the excess being
   attributed has itself shrunk to **2.45x** (shipped 27 vs off 11 over 19,200 fly-s each; binomial p 0.014, Jeffreys
   95 % CI 1.25-5.08x) from the 3.43x on record, with the default's escape count (3) *below* off's (4) in the
   skeptic's env block. Only the voluntary route (48 vs 0; optic side 40 vs off 0, U 3040, p 2.7e-10) and the GF
   median carry the attribution. The two are in any case **one channel, not two**: `flyverse/batch_body.py:85-88`
   gives escape priority (`voluntary = ~airborne & ~escape & hold >= takeoff_hold_s` = 0.3 s), so an arm with a
   higher GF pre-empts voluntary launches -- the optic side's lower escape (18 vs 24) with higher voluntary (34 vs
   32) at a near-equal total (52 vs 56) is that trade-off, not two independent deviations -- and `rows >= 33 Hz` is
   identical to "rows with at least one escape hop" (47 = 47, 0 disagreements over 192 rows). G.2's and G.3's four
   measures are therefore two channels, and the escape-count share (65 %) disagreeing with the GF-median share
   (103 %) is itself evidence that the 65 % is count noise.
3. **The halves do not cancel in the room.** Unlike `walk.GF_max` (4.6 both, 5.0 neither, 12.5 / 13.3 either alone),
   the room measure is ordered off ~ Brain-side << optic-side ~ default on hops, on each route separately and on the
   GF tail. The receptor model's room cost is therefore a single-sided, monotone effect, and the pinned walk
   section's non-monotonicity is a property of that section, not of the model's take-off behaviour.
4. **Off with the damping retired is the same denominator.** 0.486 [0.195, 1.002] vs 0.625 [0.286, 1.186] per 1,000
   fly-s (U 1105, p 0.58), walking-GF median 26.53 vs 27.20 Hz (p 0.24), 6/48 vs 8/48 rows at 33 Hz, and 0 voluntary
   take-offs in both -- so retiring the GF x0.3 input damping did not make off quieter or noisier, and every ratio
   the record quotes against off survives with the correct denominator: the shipped default is 8.0x off on hops
   (3.889 vs 0.486; count ratio 8.0x, 95 % CI 3.9-18.5x) and its 32 voluntary take-offs stand against 0 in 52,800
   fly-s of off (11 batches).
5. **Dose is not controlled, and the next hold must control it.** The optic side is 92 % of the changed entries and
   **95 % of the changed \|W\|** (44,463 / 48,295 entries; 171,111 / 179,944 raw-synapse \|W\|). Nothing in this
   section separates "the optic PATHWAY carries it" from "the larger perturbation carries it". Round 5's finding that
   the 3,832-entry Brain side carries 100 % of taste and smell (E.4) shows the small side is not globally inert,
   which mitigates but does not close it. So the `holdOpticHis` / `holdOpticGlu` split of item 6 is **required, not
   optional**, and it needs a dose control run beside it: a random 3,832-entry subset of the optic side, matched to
   the Brain side's entry count, as a fifth arm.
6. **What this does not settle.** (a) The mechanism. No changed entry touches DNp01, LC4, LPLC2, MN9 or the five
   formerly damped inputs, so the optic side must act through the state of the medulla it changes; which of T1 /
   Dm9 / Mi15 / Mi4 / Mi1 / Dm2 / Mi9 does it, and whether through the loom pathway (escape) or through the wing-power
   command (voluntary), is untested -- the natural next hold is a per-type or per-transmitter split of the optic side
   (the builder already supports `--groups`; a `holdOpticHis` / `holdOpticGlu` pair would separate the 17,256
   silencings from the 27,207 flips, 83,191 vs 87,920 \|W\|). (b) Whether the optic-side rate is *too high*: this
   section measures attribution, not correctness. There is no animal reference on file for spontaneous take-off rate
   in this arena, and the voluntary route's KNOWN-GAP status (P(pass per 2,400 fly-s draw) ~ 0.1) is unchanged.
   (c) Three batches per arm here (four with the skeptic's) and one protocol (cx program, apple, fenced, energy 0.9,
   300 s); the flies are at energy 0 for the last ~2 min of every rollout (median minimum energy 0.0 in all 9
   batches, 0-8 meals per batch), so the rates are over a mixed fed / starved regime, as every take-off number on
   record is. (d) Also unsettled by design: `holdOptic` is "postsynaptic superclass outside `ol_intrinsic` /
   `ol_sensory`", and 94 of its 3,832 entries land on visual centrifugal / visual projection cells (OA-AL2i3 65,
   TmY14 29), so "the Brain side carries nothing" is a statement about that partition, not about every visual neuron
   outside the medulla.

**Two bookkeeping notes.** (i) `out/d1_hold_weights.log`'s 175,706 / 167,029 / 8,678 are **shaped-weight** \|W\|
(167,029 + 8,678 = 175,707, the 1 a float-sum rounding); the G.0 table and every claim above quote **raw synapse**
\|W\| 179,944 / 171,111 / 8,833. Both are correct and they are different quantities -- the log does not say so.
(ii) `scripts/batch_sustain.py`'s JSON header should carry `type_path_gain` and the resolved `LIFParams`. This
section had to certify its arm labels by md5-ing `brain.py` in three run directories (G.1) because the room JSONs
record `receptor` but not the gains; it is a two-line header change and it makes arms self-describing, which is also
what `anti_runaway.md` caveat (1) and the round-5 critic asked for.

Files: `out/d1_hold_batch.sh` (launcher), `out/d1_hold_cluster.log` (console; the local `tee` was cut by the
launcher's `| head -30` after the first job log and is completed in the file from the run directory's per-job logs,
which are kept in full in `out/d1_hold_joblogs/`), `out/d1_{holdBrain,holdOptic,off}_{1,2,3}.{json,txt}` (results),
`out/d1_hold_compare.log` (`scripts/compare_sustain_runs.py`, 7 pairs, + the four-arm table),
`out/d1_hold_summary.py` -> `out/d1_hold_summary.log` (the four-arm table, the Poisson intervals and the excess
shares; the generator is in `out/` because this task's file list allows only `scripts/batch_sustain.py` under
`scripts/`), `out/d1_hold_compare.sh` (the
analysis runner), `out/d1_hold_structure.log` (G.0), `out/d1_hold_weights.log` (the shaped-weight md5s and the
partition check), `out/d1_walk_gfmax_recount.log` (G.4). The closing skeptic's 4th matched batch and its
re-derivations (G.5): `scripts/sk_d1_hold_verify.sh`, `scripts/sk_d1_hold_{structure,weights,stats,pool}.py`,
`out/sk_d1_hold_cluster.log`, `out/sk_d1_{shipped,holdBrain,holdOptic,off}_4.{json,txt}`,
`out/sk_d1_hold_{structure,weights,stats,pool}.log`.
