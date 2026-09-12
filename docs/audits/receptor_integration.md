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
input). The cluster's shared cache (`/mnt/beegfs/datasets/flyverse/cache`, W md5 `289047c3...`) still lacks the override; it was
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
