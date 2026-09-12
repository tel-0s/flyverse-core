# Compass shift: can the ring attractor be moved? (compass thread, first dynamics round)

Script: `scripts/cx_shift.py` (`--structure` CPU; `--shift` and `--rotation` GPU via `scripts/cx_shift_batch.sh`, one
cluster batch; `--report` CPU). Data: `out/cx_shift_structure.{json,md,txt}` (structure, done locally),
`out/cx_shift_{base,gaba}_{2_15,2.5_25}[_hz].{json,txt}` (experiment 1, 60 runs), `out/cx_shift_rot_{default,gaba}[_g2_15].{json,txt}`
(experiment 2, 12 runs), tables `out/cx_shift_shift.{md,csv}`, `out/cx_shift_rotation.{md,csv}`, batch console log
`out/cx_shift_cluster.log`; replication (seeds 3-5) `out/vcx_{base,gaba}_hz40_s345.{json,txt}`,
`out/vcx_rot_default_g2_15_s345.{json,txt}`, log `out/cxshift_verify_cluster.log`. Tree: the runs' tree differs from
ec281f2 only in `scripts/` (`scripts/batch_sustain.py`, another thread; `cluster_run.py` ships every modified file) --
`git diff ec281f2 -- flyverse/` is empty, so no model file changed. The brain.py / fly.py / connectome.py / body.py md5s
recorded in every JSON row are not a portable check (they differ under CRLF vs LF); use the git diff against the commit.
Ring protocol and gains as in `docs/audits/cx_wedge.md` section 6 /
`cx_glno.md` section 5: EPG <-> PEN and EPG <-> PEG x gE, Delta7 -> EPG x gD (Delta7 -> PEN x1), ER/ExR x1, compass
adaptation 0, `receptor_model=None` on the ring runs (the shipped sign/abs default changes 0 ring-core entries, `cx_glno.md`
section 2); the room runs use the shipped default (sign/abs). These are experiments that apply gains to ask a question; no
default moves here.

**Provenance.** Section 1 (structure) is from the local CPU run. The GPU batch `cx-shift-eab402` (10 jobs, 0 failed,
run dir `<cluster-fs>/neurome/runs/cx-shift-eab402/`, `out/cx_shift_cluster.log`) landed after the thread returned; the
replication batch `cxshift-verify-85eb2e` (3 jobs, 0 failed, B200, `out/cxshift_verify_cluster.log`) added seeds 3-5 for
the 40 Hz arm of experiment 1 and for the gains arm of experiment 2. Every run reports `device NVIDIA B200`. Sections 2-4
below are the protocols and the landed results.

## 1. Structure: which cells carry anything into PEN, and which are silent (`--structure`, local cache, sign-0 entries from `cache/sign0_counts.npz`)

PEN (42 cells) raw input 84,572 synapses; 21.9 % of it arrives from presynaptic cells with sign 0 (silent in the model).
Top inputs by raw synapses (full table `out/cx_shift_structure.md`):

| rank | pre type | raw syn | share | cells | nt | sign | silent |
|---|---|---|---|---|---|---|---|
| 1 | GLNO | 16,371 | 19.4 % | 4 | unknown | 0 | **yes** |
| 2 | EPG | 12,330 | 14.6 % | 46 | ACh | +1 | |
| 3 | PEN_b | 9,291 | 11.0 % | 22 | ACh | +1 | |
| 4 | PEN_a | 8,625 | 10.2 % | 20 | ACh | +1 | |
| 5 | Delta7 | 7,239 | 8.6 % | 42 | Glu | -1 | |
| 6 | ExR4 | 7,122 | 8.4 % | 2 | Glu | -1 | |
| 7 | ExR6 | 5,271 | 6.2 % | 2 | Glu | -1 | |
| 8 | LPsP | 2,725 | 3.2 % | 2 | ACh | +1 | |
| 9 | IbSpsP | 2,637 | 3.1 % | 31 | ACh | +1 | |
| 10 | PEG | 2,424 | 2.9 % | 18 | ACh | +1 | |
| 11-14 | ER6 / ER1_b / ER1_a / ER4d | 2,317 / 2,301 / 1,102 / 897 | 2.7-1.1 % | | GABA | -1 | |
| 15-19 | FB4Y / ExR2 / FB1C / EL | 615 / 522 / 507 / 266 | 0.7-0.3 % | | 5-HT / DA / DA / OA | 0 | yes |

The task's candidate list against this connectome (`-> PEN` = raw synapses onto the 42 PEN; side matrix = pre somaSide ->
post PEN glomerulus side, L->L / L->R / R->L / R->R):

| type | cells | MaleCNS nt (sign) | -> PEN | % of PEN input | side matrix | -> GLNO | own input | from optic superclasses | from rotation-flip types (screen_rotation.csv, d' >= 2) | main outputs |
|---|---|---|---|---|---|---|---|---|---|---|
| GLNO | 4 | unknown (0) | 16,371 | 19.4 % | 0 / 8,236 / 8,135 / 0 (fully contralateral) | 400 (self) | 9,371 | 8 | 3 (Nod1) | PEN 92 % |
| LNO1 | 4 | GABA (-1) | 1 | 0.0 % | | 72 | 10,705 | 10 | 0 | PFNv 32 %, PFNd 24 %, LNO2 14 % |
| LNO2 | 2 | Glu (-1) | 6 | 0.0 % | | 6 | 18,937 | 344 | 7 (Nod1) | PFNd 78 %, LNO1 13 % |
| LNOa | 2 | Glu (-1) | 0 | 0.0 % | | 6 | 5,023 | 3 | 0 | PFNa 98 % |
| PS196_b | 2 | ACh (+1) | 0 | 0.0 % | | **1,801 (19.2 % of GLNO's input)** | 6,032 | 69 | 1 (Nod1) | GLNO 14 %, OA-VUMa1 8 %, LPsP 8 %, ExR2 7 %, ExR4 6 % |
| SpsP | 4 | Glu (-1) | 13 | 0.0 % | 5 / 0 / 0 / 8 | 0 | 3,467 | 115 | 0 | PFNd 88 %, Delta7 6 % |
| IbSpsP | 31 | ACh (+1) | 2,637 | 3.1 % | 1,347 / 0 / 0 / 1,290 (ipsilateral) | 0 | 16,435 | 1,698 | 3 (LPT26) | PFNa 17 %, PFNd 12 %, PFNp_b 11 %, PEN_b 9 %, PEN_a 6 % |
| LPsP (added) | 2 | ACh (+1) | 2,725 | 3.2 % | 747 / 679 / 630 / 669 (bilateral) | 3 | 12,276 | 152 | 0 | EPG 27 %, PEG 20 %, PEN_a 18 % |

So in MaleCNS v1.0 the LNO types and SpsP are PFN inputs, not PEN inputs (0-13 synapses onto PEN); GLNO is PEN's only
nodulus input and the only large silent one. What feeds the two PEN-reaching candidates: GLNO's input is PEN_a 25.6 %,
PS196_b 19.2 %, PEN_b 11.7 %, EPG 8.2 %, WED040_a 4.9 %, GLNO 4.3 %, LAL139 3.7 %, LAL184 3.3 % (cb_intrinsic 99.8 %; 8 raw
synapses from optic superclasses); PS196_b's input is PS099_a 11.6 %, PS048_a 8.0 %, PS099_b 7.4 %, PS047_b 7.4 %,
AN07B037_a 6.9 % (ascending), PS262 6.3 % (posterior-slope / ascending premotor territory, 8.3 % ascending_neuron; PS196_b
itself takes 1.1 % from optic superclasses, and the optic share of its top inputs at the second step is 0.4-23.2 %
-- PS262 23.2 %, PS047_b 16.1 %, PS099_b 7.3 %, PS099_a 4.0 %, PS048_a 3.0 %, AN07B037_a 0.4 % -- while their share from
the populations that flip under imposed rotation is 0.0-0.4 %). IbSpsP's input is Delta7 31.7 %, GNG311 11.6 %, OLVC5
5.1 % (the largest single optic-side entry, a visual centrifugal cell), EPG 4.7 %; its total optic-superclass share is
10.3 % of 16,435 raw input synapses (visual_centrifugal 6.9 %, visual_projection 3.4 %) -- the largest of the eight
candidates, but it reaches PEN ipsilaterally, so it cannot signed-rotate the bump. LPsP's input is Delta7 45 %, EPG 17 %,
PS196_b 7.7 %. None
of the eight candidates receives more than 7 raw synapses from the rotation-flip populations (HSN, DNp20, Nod1, LPT26,
LPT50, HSE, DNp15) directly, and at the second step the share is <= 3.8 % (LAL133_c -> LNOa). Structurally, then, there is
no short optic-lobe -> PEN route: the candidates are fed from LAL / PS / ascending premotor cells (an efference / proprioceptive
territory), and the only one that reaches PEN in strength is silent.

Transmitter evidence per candidate (MaleCNS body call = the model's sign; MaleCNS per-T-bar prediction, argmax shares over the
type's T-bars; FlyWire Schlegel 2024 file 1 `top_nt` per cell with its confidence, and `known_nt` where annotated):

| type | MaleCNS call (model sign) | T-bars: argmax shares | FlyWire top_nt (conf) | FlyWire known_nt |
|---|---|---|---|---|
| GLNO | unknown (0) | 3,132: Glu 0.505, ACh 0.373, 5-HT 0.072 | gaba 3 / glutamate 1 (0.30-0.33) | none |
| LNO1 | GABA (-1) | 2,766: GABA 0.442, ACh 0.363, Glu 0.155 | gaba 3 / ACh 1 (0.47-0.60) | gaba (ACh-, Glu-) x4 |
| LNO2 | Glu (-1) | 2,272: Glu 0.834 | gaba 1 / glutamate 1 (0.33-0.43) | glutamate x2 |
| LNOa | Glu (-1) | 1,376: Glu 0.922 | gaba 2 (0.65-0.66) | glutamate (ACh-, GABA-) x2 |
| PS196_b | ACh (+1) | 4,681: ACh 0.982 | ACh 2 (0.86-0.89) | none |
| SpsP | Glu (-1) | 1,646: Glu 0.676, ACh 0.148, His 0.142 | glutamate 7 (0.65-0.93) | none |
| IbSpsP | ACh (+1) | 5,538: ACh 0.951 | ACh 30 (0.55-0.94) | ACh (GABA-, Glu-) x30 |
| LPsP | ACh (+1) | 5,137: ACh 0.905 | ACh 1 / glutamate 1 (0.45-0.48) | dopamine; glutamate x2 |
| PEN_a / PEN_b | ACh (+1) | 10,999 / 9,657: ACh 0.81 / 0.86 | ACh 20 / 21 + 5-HT 1 | ACh x20 / x22 |

GLNO stays the one PEN input whose transmitter no source settles: the two EM predictions disagree between glutamate (MaleCNS
T-bars, 51 %) and GABA (FlyWire, 3 of 4 cells at confidence 0.30-0.33); both map to sign -1 under `NT_SIGN`
(`flyverse/connectome.py`: `"gaba": -1.0, "glutamate": -1.0`), neither is above the 0.5 confidence the project accepts for
a type-level override, and GLNO has no expression profile in any of the five sources (`cx_glno.md` section 1). Because the
ring runs use `receptor_model=None`, the sign is the only thing the model consumes: **a GLNO=gaba arm and a GLNO=glutamate
arm are the same run**, so the two low-confidence predictions disagree about the label and agree about the physics, and
experiment 1's `gaba` arm covers both. LNOa is a second disagreement (MaleCNS Glu 92 % of T-bars vs FlyWire gaba at 0.65) that
does not matter here (0 synapses onto PEN). LPsP's FlyWire `known_nt` (dopamine; glutamate) against MaleCNS ACh is a
candidate for the receptor / NT threads, not this one.

## 2. Experiment 1 protocol: the PEN L / R asymmetry (`--shift`)

1 s settle on a 10 Hz Poisson background (all 46 EPG), wedges 0-3 (L1 R8 L2 R7, 11 cells) at +40 Hz for 2 s, 2 s free, then
`fb.stimulate({'type': '~^PEN_', 'somaSide': SIDE}, PEN_HZ, 1000 ms)` (21 cells per side; somaSide = PB glomerulus side for
every PEN, verified), then 2 s free. Every 100 ms: the EPG rate vector and the EPG spike counts; the bump centre is the circular
mean of the 16-wedge spike-count profile (ring order L1 R8 L2 R7 L3 R6 L4 R5 L5 R4 L6 R3 L7 R2 L8 R1; + = towards increasing
wedge index), unwrapped over the run. Statistics per run: shift over the stimulus window (mean of the last two bins before
offset minus before onset), its rate (wedges / s), the rate per Hz of PEN drive, a linear slope fit over the window, the
drift in the free window before and the 2 s after, PEN L / R and GLNO L / R rates per window, and whether the bump was alive
at onset and at the end (peak wedge > 50 Hz, vector strength > 0.4). Conditions: gE 2 / gD 15 and gE 2.5 / gD 25 x GLNO silent
/ GLNO = gaba (scratch cache) x seeds 0-2 x sides L / R / none at 20 Hz (36 runs), plus 10 and 40 Hz at gE 2 / gD 15 in both
GLNO conditions (24 runs) for the per-Hz slope. Runs are deterministic given (connectome, gains, seed), so the three sides of
one seed share their first 5 s and the L - R and side - none contrasts are seed-matched.

Seeds 3-5 were added at 40 Hz / gE 2 / gD 15 in both GLNO conditions by the replication batch (18 further runs,
`out/vcx_{base,gaba}_hz40_s345.json`), because three seeds cannot reject anything: with 3 seed-matched pairs the exact
sign-flip test has a floor of 2 / 2^3 = **p 0.25**, so a 3/3 result is not evidence. Six pairs give a floor of
2 / 2^6 = 0.031.

## 2b. Experiment 1 result: one side of PEN deflects the bump only when GLNO is made inhibitory, and the deflection is elastic

Read the **side minus none** contrast, not L - R. The no-stimulus arm is not a flat line: on seeds 3-5 the 0-Hz `none`
run's own "shift over the window" is +0.009 to +0.180 wedges (`vcx_*_hz40_s345.json`: GLNO silent +0.086 / +0.039 /
+0.164, GLNO=gaba +0.123 / +0.009 / +0.180, against -0.020 to +0.030 on seeds 0-2), so L - R subtracts two arms that are
both moving and flatters the effect. Every number below is seed-matched (the three sides of one seed share their first
5 s, `cx_shift.py` reseeds nothing at the PEN stimulus).

**40 Hz on 21 PEN of one side for 1 s, gE 2 / gD 15, 6 seeds** (wedges over the stimulus window, mean +- sd; per-seed
values from `out/cx_shift_shift.csv` seeds 0-2 and `out/vcx_*_hz40_s345.json` seeds 3-5):

| GLNO | L - none | R - none | L - R | sign of L - none |
|---|---|---|---|---|
| silent (default, sign 0) | **+0.038 +- 0.018** | -0.009 +- 0.018 | +0.046 +- 0.033 | 6 / 6 positive (exact p 0.031) |
| gaba (sign -1) | **+0.301 +- 0.036** | +0.015 +- 0.058 | +0.285 +- 0.055 | 6 / 6 positive (exact p 0.031) |
| gaba - silent (paired) | **+0.263 +- 0.041** | +0.024 +- 0.052 | | 6 / 6 positive (exact p 0.031) |

So: with GLNO silent, driving one side of PEN at twice its free rate moves the bump 0.038 wedges = 0.9 deg -- consistent
in sign across 6 seeds but a twentieth of a wedge, i.e. nothing. Relabelling GLNO inhibitory multiplies that by 8, to
+0.301 wedges = 6.8 deg, still under half a wedge. The right side does nothing in either condition (the R - none mean is
within its own sd in both, and the gaba R arm's sd is inflated by one seed, +0.116 on seed 3). The asymmetry is not a
bug in the stimulus: PEN L / R during the L stimulus is 70.2 / 44.7 Hz (gaba) and 79.8 / 55.2 Hz (silent) against a free
39.2 / 41.2 and 41.0 / 54.5, and the R stimulus is the mirror image (39.0 / 76.1 and 42.0 / 93.5). It is the ring's own
left / right asymmetry -- the bump sits between wedge 1.16 and 1.81 at stimulus onset in all 78 runs, so the two sides of
PEN are not symmetric with respect to it.

**The deflection does not integrate: it is elastic.** Over the same 6 seeds, gaba, L at 40 Hz (`total_shift_onset_to_end`
minus the same quantity in the seed's `none` run):

| quantity, gaba L 40 Hz, seed-matched to `none` | mean +- sd (6 seeds) |
|---|---|
| shift during the 1 s stimulus | +0.301 +- 0.036 |
| net shift onset -> end of run (2 s after offset) | **+0.002 +- 0.016** (0.4 % of the deflection retained) |
| drift in the 2 s after offset | **-0.149 +- 0.015** wedges / s, negative in 6 / 6 |

Seed 0 is the picture: centre 1.256 at onset -> 1.680 at offset -> 1.217 at the end, with the after-drift at
-0.231 wedges / s (`cx_shift_gaba_2_15_hz.json`). Raw (not seed-matched) `total_shift_onset_to_end` is negative on seeds
0-2 (-0.144 / -0.070 / -0.113) and positive on seeds 3-5 (+0.171 / +0.191 / +0.199) -- but the seed's own `none` arm
drifts by the same amount on those seeds, which is exactly why the contrast, not the raw number, is the statistic. The
raw after-drift is negative in 6 / 6 at -0.052 to -0.231 wedges / s against -0.084 to +0.089 in the `none` arms.

**10 and 20 Hz are inside the drift noise** (seeds 0-2 only, gE 2 / gD 15, L - none): silent +0.008 +- 0.021 at 10 Hz and
+0.032 +- 0.027 at 20 Hz; gaba +0.023 +- 0.071 at 10 Hz and +0.098 +- 0.107 at 20 Hz, the latter carried by one seed
(+0.218 on seed 2 against +0.014 and +0.062). The per-Hz deflection under gaba does rise with drive
(+0.0023 / +0.0049 / +0.0074 wedges per Hz at 10 / 20 / 40 Hz) but only the 40 Hz point is out of the noise.

**gE 2.5 / gD 25 at 20 Hz is null in both conditions** (3 seeds, `out/cx_shift_shift.md` summary): L - none
+0.002 +- 0.009 (silent) and +0.019 +- 0.011 (gaba), R - none -0.015 +- 0.022 and +0.003 +- 0.005. Across all 78 runs of
experiment 1 the bump is alive at onset and at the end in 78 / 78 (peak 238-286 Hz, vector strength 0.68-0.81); nothing
here destroys the bump, and nothing here integrates.

## 3. Experiment 2 protocol: imposed rotation in the room (`--rotation`)

`scripts/screen_rotation.py`'s protocol called through `room_demo.Sim` (pinned fly re-placed every 10 ms at (0, 0, 0.75),
wind off, shipped defaults): rest / +90 deg/s / rest / -90 deg/s for 10 s each, 1 s smoothing, the first 3 s of each phase
skipped; per type and side the L - R asymmetry, flip = (L - R at ccw) - (L - R at cw), d' = flip / (pooled sd + 1). Recorded:
PEN_a / PEN_b, GLNO, LNO1 / 2 / a, LPsP, PS196 a / b, SpsP, IbSpsP, EPG, Delta7, PEG, PFN a / d / v, the ring neurons, and
the screen's rotation populations (HSN, HSE, VS, DNp20, LPT26 / 50, Nod1 / 4) as positive controls. Conditions: default
(GLNO silent), GLNO = gaba, and each of those with the compass gains gE 2 / gD 15 applied (compass adaptation 0, 10 Hz EPG
background, a 2 s pulse on wedges 0-3 at +40 Hz in the first rest phase) so that the bump's centre drift per phase is
measured in the room with self-rotation. Seeds 0-2 per condition (12 runs).

## 3b. Experiment 2 result: the visual world rotates, the screen cells flip, the bump does not move

Only the two `+gains 2/15` conditions have a bump at all: without the compass gains the ring is silent in the room
(`out/cx_shift_rotation.md`, "EPG bump per phase": peak 0 Hz, centre pinned at the placeholder 11.00 in every phase of
`default` and `gaba`), which is `cx_wedge.md`'s LIF-alone result reproduced with senses on. With the gains the bump is
there in all 4 phases x 3 seeds x 2 GLNO conditions (vector strength 0.73-0.77, peak 234-255 Hz).

At 90 deg/s a heading-anchored bump would move at 90 / 22.5 = **4.000 wedges / s**. Measured drift per phase
(rest / ccw / rest2 / cw, 3 seeds = 12 values per condition, `cx_shift_rot_{default,gaba}_g2_15.json`):

| condition | bump drift, wedges / s | vs ideal |
|---|---|---|
| default + gains 2/15 (GLNO silent) | **-0.008 to +0.006** (12 / 12) | <= 0.2 % |
| gaba + gains 2/15 | **-0.010 to +0.010** in 11 / 12 | <= 0.25 % |
| gaba + gains 2/15, the 12th | -0.164 (seed 1, cw; peak drops to 234 Hz, vs 0.73) | 4 %, and the phase's net displacement is only -0.07 wedges |
| replication seeds 3-5, default + gains | **-0.007 to +0.007** | <= 0.2 % |

and the drift has no relation to the sign of the rotation: in `default + gains` the ccw phases give -0.002 / +0.002 /
+0.006 and the cw phases +0.002 / +0.006 / +0.005, i.e. the same numbers. Net displacement over a phase's analysed
window (10 s less the 3 s skipped and 1 s smoothing) is -0.15 to +0.19 wedges in both conditions, against the ~25 wedges
a heading-anchored bump owes over that window.

**The stimulus was delivered.** The same runs' positive controls flip hard (`out/cx_shift_rotation.md`, seed-mean d' in
`default + gains 2/15`, 3 seeds):

| type | flip d' | flip Hz | rate at rest Hz |
|---|---|---|---|
| HSN | -3.81 +- 0.02 | -9.52 | 1.02 |
| HSE | -3.29 | -5.40 | 0.57 |
| Nod1 | -3.07 | -7.92 | 1.22 |
| DNp20 | -2.72 | -13.47 | 8.60 |
| LPT26 | -2.47 | -6.85 | 0.70 |
| LPT50 | +2.25 | +5.48 | 1.60 |
| (Nod4 +1.84, VS -1.66 fall short of \|d'\| 2) | | | |

Across all 60 recorded types the \|d'\| >= 2 set is exactly {HSN, HSE, Nod1, DNp20, LPT26, LPT50} -- a strict subset of
the seven types `out/screen_rotation.csv` finds under the same imposed rotation ({HSN, DNp20, Nod1, LPT26, LPT50, HSE,
DNp15}); the seventh, DNp15, is simply not in this script's recorded panel, so nothing here contradicts it. The
replication seeds 3-5 reproduce the set (HSN -3.77 to -4.45, Nod1 -2.84 to -3.82, HSE -2.93 to -3.38, DNp20 -2.60 to
-3.33, LPT50 +2.04 to +2.53; LPT26 -1.93 to -2.52, missing the bar on one seed --
`out/vcx_rot_default_g2_15_s345.txt`).

**Nothing of that reaches the compass.** In the same runs GLNO's flip d' is +0.20 (per seed +0.17 / +0.69 / -0.28) with
GLNO silent -- i.e. the 132 Hz GLNO rate is driven by PEN, not by the screen -- and +0.13 (+0.07 / +0.57 / -0.25) with
GLNO=gaba; EPG +0.24 / +0.16, Delta7 +0.24 / +0.30, PEG -0.13 / -0.04, PS196_b -0.10 / -0.12, IbSpsP -0.02 / -0.06,
LPsP and SpsP +0.00. The largest per-seed magnitude anywhere in this set is 0.93 (LNOa, default + gains, seed 0, whose
other two seeds are +0.45 and -0.48) against the screen cells' 2.2-4.3, and the only types with a consistent sign across
seeds are PS196_b (-0.09 to -0.10) and IbSpsP (-0.01 to -0.03), carrying flips of 0.12 Hz and 0.02 Hz on types running
at 0.06 and 1.12 Hz. The largest seed-mean response outside the screen set is LNO2 at d' +0.41 -- a PFN input, not a PEN
input.

## 4. Answers

**(1) Is there a signed rotation input to PEN in MaleCNS v1.0? No.** PEN's only large nodulus input is GLNO
(19.4 % of PEN's 84,572 raw input synapses, fully contralateral, 0 / 8,236 / 8,135 / 0), and GLNO's transmitter is
`unknown` in every MaleCNS column, so it carries sign 0 -- its 17,698 output synapses are explicit zeros in W. The
candidates the task named are not PEN inputs at all in this connectome: LNO1 1 synapse, LNO2 6, LNOa 0, PS196_b 0,
SpsP 13, against PEN's 84,572. And GLNO is not a visual cell: 8 of its 9,371 raw input synapses come from optic
superclasses (99.8 % cb_intrinsic), and what feeds it is PEN itself (37.3 %: PEN_a 25.6 % + PEN_b 11.7 %), PS196_b
19.2 %, EPG 8.2 %, then LAL cells (WED040_a 4.9 %, LAL139 3.7 %, LAL184 3.3 %). That is efference-copy /
premotor territory, not an optic-flow line: a loop that reads what PEN and EPG are doing and what the LAL is
commanding. The one candidate with a real optic share (IbSpsP, 10.3 %) projects to PEN ipsilaterally
(1,347 / 0 / 0 / 1,290), which cannot produce a signed rotation. Structurally there is no route by which a visual
rotation could turn the bump, and the loop that exists is the wrong shape for one.

**(2) What does relabelling GLNO inhibitory buy? A one-sided elastic lever, not a heading update.** Making GLNO
sign -1 turns a 19.4 % silent input into the largest inhibitory input onto PEN and multiplies the bump's response to
unilateral PEN drive by 8 (+0.038 -> +0.301 wedges, section 2b). But the deflection (a) works on one side only
(R - none +0.015 +- 0.058, indistinguishable from zero), (b) is 6.8 deg at 40 Hz -- twice PEN's free rate, held for a
second -- where a walking fly turns hundreds of degrees per second, and (c) is **elastic**: the seed-matched net shift
from onset to the end of the run is +0.002 +- 0.016 wedges against +0.301 +- 0.036 during the stimulus (0.4 % of the
deflection on average), and the bump springs back at -0.149 +- 0.015 wedges / s in 6 / 6 seeds. An angular-velocity input
to a ring attractor has to *integrate*: the bump must stay where the velocity signal put it. This one is a spring. So
the answer to "would a signed GLNO give the ring a rotation input" is no even under the relabelling the two EM
transmitter predictions jointly support. (And because `NT_SIGN` maps gaba and glutamate alike to -1 and the ring runs
use `receptor_model=None`, this arm *is* the glutamate arm; the two predictions agree on the only quantity the model
consumes.)

**(3) Does the visual world rotating move the bump? No -- but only the visual route was tested.** At 90 deg/s the bump
drifts at -0.010 to +0.010 wedges / s where 4.000 is due (<= 0.25 %) in 35 of the 36 phase measurements -- both GLNO
conditions x 3 seeds x 4 phases, plus 3 replication seeds in the GLNO-silent condition -- the exception being one phase
at -0.164 w/s, still 4 % of the ideal and with a net displacement of 0.07 wedges. The stimulus was demonstrably
delivered (HSN d' -3.81, HSE -3.29, Nod1 -3.07, DNp20 -2.72, LPT26 -2.47, LPT50 +2.25 in the same runs) and none of it
reaches GLNO (d' +0.20 / +0.13, per-seed values straddling zero). That is a clean null for the *optic* route, which
section 1 predicts.

It is not a null for the *efferent* route, and the protocol is why: experiment 2 teleports the fly
(`scripts/cx_shift.py` line 526, `sim.fly.place(0.0, 0.0, 0.75, heading=state["h"])` every 10 ms), so the world turns
around a body that never issues a turn. The route section 1 actually points at -- PS196_b (19.2 % of GLNO's input) and
the LAL cells, i.e. premotor efference copy -- is only excited when the fly turns *itself*. **Next experiment:**
`--rotation-mode efferent` -- the fly steers with its own descending output (the `cx` program driven with a held
steering error, or direct DNa02 stimulation) instead of being re-placed, same 90 deg/s target, 3 seeds x
{GLNO silent, GLNO=gaba}, same bump-drift statistic and the same positive controls. If the efferent route is also null,
the model has no rotation input to the compass by any path and the question is closed; if it is not, the 6.8 deg
elastic lever of section 2b is where to look for why.

### Verification

The Opus skeptic's pass over this document found it **"mostly sound"** and returned the replication batch
`cxshift-verify-85eb2e` (3 jobs, 0 failed, B200, `out/cxshift_verify_cluster.log`): seeds 3-5 at 40 Hz in both GLNO
conditions (`out/vcx_{base,gaba}_hz40_s345.json`) and seeds 3-5 of the rotation gains arm
(`out/vcx_rot_default_g2_15_s345.json`). All headline numbers above are the 6-seed versions. `out/verify_cx_shift_shift.{md,csv}`
is byte-identical to `out/cx_shift_shift.{md,csv}` (the `--report` recompute reproduced the table exactly). The
corrections the pass produced are folded in above: the 3-seed permutation floor (p 0.25), the gaba == glutamate identity
under `NT_SIGN` with `receptor_model=None`, five expression sources not six, PS196_b's second-step optic share
0.4-23.2 %, IbSpsP's total optic share 10.3 %, and the provenance sentence (`git diff ec281f2 -- flyverse/` is empty;
md5s are not portable across line endings). One defect remains in the data: `out/vcx_rot_default_g2_15_s345.json` holds
2 rows (seeds 3, 4) although its `.txt` logs three seeds and ends "3 rows -> ..."; seed 5's numbers are therefore
available only from the console log, at 2 dp (all four phases \|drift\| <= 0.005 wedges / s), which does not change the
conclusion.

## 5. Files

* `scripts/cx_shift.py` (new), `scripts/cx_shift_batch.sh` (the batch generator), `out/cx_shift_structure.{json,md,txt}`,
  `out/cx_shift_cluster.log`, `out/cx_shift_{base,gaba}_{2_15,2.5_25}[_hz].{json,txt}`, `out/cx_shift_rot_*.{json,txt}`,
  `out/cx_shift_shift.{md,csv}`, `out/cx_shift_rotation.{md,csv}`; smokes (CPU, GPU hidden): `out/smoke/cx_shift_smoke.json`,
  `out/smoke/cx_shift_rot_smoke.json`.
* Replication (`cxshift-verify-85eb2e`): `out/cxshift_verify_cluster.log`, `out/vcx_{base,gaba}_hz40_s345.{json,txt}`,
  `out/vcx_rot_default_g2_15_s345.{json,txt}`, `out/verify_cx_shift_shift.{md,csv}` (identical to `cx_shift_shift.*`).
