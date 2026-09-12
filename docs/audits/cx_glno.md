# Compass, round 2: the GLNO -> PEN sign and the receptor model in the ring

Scripts: `scripts/cx_glno.py` (driver; `--run <config>` on the cluster, `--report` / `--check-receptor` on the CPU),
`scripts/cx_wedge.py` (round-2 flags `--nt-override TYPE=nt`, `--scratch-cache`, `--receptor-model`,
`--receptor-net-rule`; every simulation row now records GLNO's rate). Data: `out/cx_glno_{base,glu,ach,sign-class,
sign-abs}.json` + `.txt` (cluster logs), `out/cx_glno_table.{md,csv}` (the table below), `out/cx_glno_receptor_check.json`
(the receptor-entry count). Protocol = `docs/audits/cx_wedge.md` section 6: FlyBrain on the full connectome, compass
adaptation 0, 10 Hz Poisson background on all 46 EPG, wedges 0-3 (11 cells) at +40 Hz for 2 s, 5 s free; gains
EPG <-> PEN and EPG <-> PEG x gE, Delta7 -> EPG x gD, Delta7 -> PEN x1, ER/ExR x1; gE 2 / gD 15 and gE 1.75 / gD 15;
seeds 0, 1, 2. Cluster (B200, torch 2.11+cu128, cuda graphs, torch sparse backend), one batch of five jobs, each
compiling its connectome from the raw MaleCNS files into `out/cache_<hash>/` with the adopted `TYPE_NT_OVERRIDE`
(TmY14 / Mi19 / aMe8) so that the cluster's shared cache (which predates the override) is not used.

Coverage: the ring under test is 46 EPG + 42 PEN + 18 PEG + 42 Delta7 (+ 4 EPGt, 308 ER/ExR, 4 GLNO) of the 167,106
MaleCNS cells (0.28 %); their input entries 27,553 (EPG 16,848 / PEN 5,596 / Delta7 5,109) of 25,578,600 stored
entries (0.11 %). The receptor rows that cover them: EPG and Delta7 tier `alias` (Davis 2020 PB_2 / Delta7 drivers),
PEN_a / PEN_b tier `fuzzy` (Davis PB_3, the Mi1-contaminated driver); GLNO has no row (tier `fallback` on its 646
input entries). The simulation itself runs the whole connectome (every cell, every entry).

## 1. GLNO in the connectome (local cache, `cx_glno.py --check-receptor`; raw counts via `sign0_counts`)

| item | value |
|---|---|
| cells | 4: GLNO(LAL-NO1)_L x2, GLNO(LAL-NO1)_R x2 (bodyIds 12104, 14881, 23325, 25939) |
| transmitter | `unknown` in every MaleCNS column (consensus / cell-type / per-body all `unclear`, conf 0.48); T-bar prediction over 783 T-bars glutamate 51 % / acetylcholine 37 % / serotonin 7 % (`nt_audit.md` row GLNO); in none of the five expression sources |
| sign under NT_SIGN | 0 -> its 17,698 raw output synapses are explicit zeros in W |
| GLNO -> PEN | 84 entries, 16,371 raw synapses = 19.36 % of PEN's raw input (84,572); every edge 88-345 synapses (mean 195, median 182), i.e. every one of the 84 sits above the connection cap 60 |
| per PEN cell | exactly 2 GLNO inputs (both from the contralateral GLNO pair: GLNO_L -> R-glomerulus PENs 8,236 syn, GLNO_R -> L-glomerulus PENs 8,135); PEN_a 9,806 syn, PEN_b 6,565 |
| effective weight if signed | 0.275 mV x 60 = 16.5 mV per GLNO spike per pair, 33.0 mV per PEN per volley of its 2 GLNO (EPG -> PEN is +5.05 mV per pair, +79.9 mV per PEN per EPG volley; Delta7 -> PEN -4.70 per pair) ; PEN fan-in scale stays 1.00 (totals 867-1,598 -> +120) |
| PEN -> GLNO | 84 entries, 3,496 raw syn = 37.3 % of GLNO's 9,371 raw input (ACh 6,985 = 74.5 %, GABA 896, Glu 742, monoamine 338, unknown 410); capped, +222 mV per GLNO per PEN volley (fan-in totals 1,476-1,789, scale 1.00) |
| GLNO outputs elsewhere | ExR8 435, GLNO 400, FB4Y 218, FB1C 114, others < 25 each; 92.5 % of its output is PEN |

So the silent loop is PEN -> GLNO (+222 mV per GLNO per PEN volley, on) -> PEN (16.5 mV per pair, off). Signed
either way it is the largest single unitary input a PEN receives.

## 2. Does the receptor model touch the ring? (CPU, local adopted cache, `out/cx_glno_receptor_check.json`)

`connectome.receptor_signs(c, net_rule)` compared entry by entry with the presynaptic NT_SIGN, restricted to entries whose
POSTSYNAPTIC cell is EPG / PEN_a / PEN_b / Delta7 (27,553 entries; 27,409 matched by a table row, 144 `pre_unknown`):

| net rule | EPG (16,848) | PEN_a (2,821) | PEN_b (2,775) | Delta7 (5,109) | ring core total | PEG (1,796) | EPGt (360) | ER/ExR (49,483) | GLNO (646) | ring core as PRE (27,269 entries) |
|---|---|---|---|---|---|---|---|---|---|---|
| class | 0 | 0 | 0 | 0 | **0 / 27,553** | 0 | 0 | 0 | 0 | 63 |
| abs | 0 | 0 | 0 | 0 | **0 / 27,553** | 0 | 0 | 1,487 (pre ExR5 438, AOTU046 394, WED035 110, PLP046 76, ExR6 45 ...) | 0 | 0 |
| nonmda | 0 | 0 | 0 | 0 | **0 / 27,553** | 0 | 0 | 0 | 0 | 608 |

The expected 0 is verified for all three rules: the EPG / Delta7 (alias, Davis PB_2 / Delta7 drivers) and PEN_a / PEN_b
(fuzzy, Davis PB_3) rows carry acetylcholine +1, GABA -1, glutamate -1 under class / abs / nonmda alike (table rows
printed by `--check-receptor`), which is NT_SIGN. Under `sign` the ring's own weights are therefore bit-identical to the
base run; any difference between `sign-*` and `base` rows below comes from outside the ring (63 / 0 / 608 entries
whose PRE is a ring cell, and the 1,487 ER/ExR input entries under `abs`, i.e. the ring neurons' glutamate input
from ExR5 / AOTU046 flipped by the GluR-led `abs` call) and from the rest of the brain (the model-wide 879,459 / 127,462 flipped synapses on the adopted TYPE_NT_OVERRIDE cache that every run used).

## 3. Simulation runs (cluster batch `cx-glno-f2e987`, run dir `<cluster-fs>/neurome/runs/cx-glno-f2e987`)

Five jobs (`cx_glno.py --run base | glu | ach | sign-class | sign-abs`, 2 gains x 3 seeds each = 30 runs) submitted
2026-09-12 03:18 UTC, all completed 03:21-03:22 UTC (3.4-4.5 min each, B200, CUDA asserted). Results fetched to
`out/cx_glno_*.json` and tabulated by `scripts/cx_glno.py --report` (`out/cx_glno_table.md`, reproduced below). The
skeptic added seeds 3-5 for base / glu / ach (batch `sk-glno-55aaf4`, 18 runs); those numbers are quoted in the answers.

Table columns (config x gE x seed): EPG in / out mean Hz before the pulse (cells > 22 Hz in brackets), at 0.5 s, 2 s
and 5 s after release; cells > 22 Hz in / out at 5 s (persistence = >= 8 / 11 in and <= 3 / 35 out); circular vector
strength and centre wedge at 5 s; capture ("captured" = a bump elsewhere before the pulse, >= 4 outside cells > 22 Hz,
and the 5 s bump centre within the driven block +-0.5 wedge); PEN / Delta7 / GLNO / ER-ExR / rest-of-brain rates at
5 s and GLNO during the pulse.

## Per run

| config | gE | gD | seed | before pulse in/out (>22 Hz in, out) | 0.5 s in/out (>22) | 2 s | 5 s in/out Hz | in >22 Hz | out >22 Hz | persists | vs | centre wedge | capture | PEN | Delta7 | GLNO | GLNO during pulse | ER/ExR | rest | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 2.0 | 15.0 | 0 | 14/64 (0/11, 12/35) | 204/8 (11, 0) | 201/9 (10, 1) | 201/10 | 11/11 | 1/35 | yes | 0.76 | 1.5 | captured | 48.7 | 101.2 | 132.8 | 147.9 | 9.6 | 0.033 | 27.1 |
| base | 2.0 | 15.0 | 1 | 10/60 (1/11, 13/35) | 201/10 (10, 4) | 197/10 (10, 4) | 201/9 | 10/11 | 3/35 | yes | 0.78 | 1.5 | captured | 48.2 | 99.5 | 137.2 | 150.9 | 9.8 | 0.034 | 17.3 |
| base | 2.0 | 15.0 | 2 | 8/56 (0/11, 9/35) | 203/10 (10, 4) | 200/10 (11, 3) | 204/9 | 11/11 | 2/35 | yes | 0.78 | 1.4 | captured | 48.3 | 99.7 | 132.0 | 143.1 | 9.7 | 0.033 | 17.0 |
| glu | 2.0 | 15.0 | 0 | 21/11 (6/11, 2/35) | 190/8 (10, 0) | 184/9 (10, 1) | 180/10 | 10/11 | 1/35 | yes | 0.75 | 1.2 | spontaneous at driven tile | 40.3 | 95.6 | 114.8 | 126.9 | 8.7 | 0.029 | 18.6 |
| glu | 2.0 | 15.0 | 1 | 10/51 (1/11, 12/35) | 181/10 (11, 4) | 180/10 (10, 4) | 181/9 | 10/11 | 3/35 | yes | 0.77 | 1.2 | captured | 39.8 | 95.4 | 118.2 | 126.7 | 8.8 | 0.032 | 18.9 |
| glu | 2.0 | 15.0 | 2 | 8/17 (0/11, 12/35) | 180/10 (11, 4) | 184/10 (11, 3) | 184/9 | 10/11 | 2/35 | yes | 0.78 | 1.2 | captured | 40.6 | 94.7 | 116.3 | 122.4 | 8.6 | 0.03 | 28.0 |
| ach | 2.0 | 15.0 | 0 | 14/80 (0/11, 14/35) | 238/8 (11, 0) | 236/9 (11, 1) | 234/10 | 11/11 | 1/35 | yes | 0.77 | 1.5 | captured | 65.5 | 110.8 | 173.2 | 170.2 | 12.1 | 0.05 | 18.8 |
| ach | 2.0 | 15.0 | 1 | 10/88 (1/11, 16/35) | 235/10 (11, 4) | 234/10 (11, 4) | 233/9 | 11/11 | 3/35 | yes | 0.79 | 1.6 | captured | 65.1 | 109.8 | 168.8 | 174.3 | 12.0 | 0.046 | 20.2 |
| ach | 2.0 | 15.0 | 2 | 8/88 (0/11, 12/35) | 234/10 (11, 4) | 229/10 (11, 3) | 230/9 | 11/11 | 2/35 | yes | 0.79 | 1.5 | captured | 63.6 | 108.5 | 166.1 | 173.6 | 11.7 | 0.047 | 27.6 |
| sign-class | 2.0 | 15.0 | 0 | 14/64 (0/11, 12/35) | 204/8 (11, 0) | 201/9 (10, 1) | 201/10 | 11/11 | 1/35 | yes | 0.76 | 1.5 | captured | 48.7 | 101.2 | 132.8 | 147.9 | 9.6 | 0.039 | 32.2 |
| sign-class | 2.0 | 15.0 | 1 | 10/60 (1/11, 13/35) | 201/10 (10, 4) | 197/10 (10, 4) | 201/9 | 10/11 | 3/35 | yes | 0.78 | 1.5 | captured | 48.2 | 99.5 | 137.2 | 150.9 | 9.8 | 0.04 | 33.9 |
| sign-class | 2.0 | 15.0 | 2 | 8/56 (0/11, 9/35) | 203/10 (10, 4) | 200/10 (11, 3) | 204/9 | 11/11 | 2/35 | yes | 0.78 | 1.4 | captured | 48.3 | 99.7 | 132.0 | 143.1 | 9.7 | 0.04 | 22.2 |
| sign-abs | 2.0 | 15.0 | 0 | 14/63 (0/11, 12/35) | 205/8 (11, 0) | 204/9 (11, 1) | 201/10 | 11/11 | 1/35 | yes | 0.76 | 1.5 | captured | 49.6 | 101.0 | 134.5 | 145.1 | 10.1 | 0.034 | 19.0 |
| sign-abs | 2.0 | 15.0 | 1 | 10/60 (1/11, 13/35) | 196/10 (10, 4) | 193/10 (10, 4) | 200/9 | 10/11 | 3/35 | yes | 0.77 | 1.5 | captured | 47.6 | 99.0 | 136.9 | 147.1 | 10.0 | 0.035 | 18.1 |
| sign-abs | 2.0 | 15.0 | 2 | 8/53 (0/11, 9/35) | 194/10 (10, 4) | 200/10 (10, 3) | 203/9 | 11/11 | 2/35 | yes | 0.78 | 1.4 | captured | 48.3 | 98.3 | 135.7 | 141.4 | 9.8 | 0.035 | 31.5 |
| base | 1.75 | 15.0 | 0 | 14/10 (0/11, 2/35) | 167/8 (9, 0) | 169/9 (10, 1) | 164/10 | 10/11 | 1/35 | yes | 0.74 | 1.3 | no prior bump | 40.8 | 89.2 | 116.1 | 128.9 | 8.0 | 0.028 | 18.3 |
| base | 1.75 | 15.0 | 1 | 10/12 (1/11, 4/35) | 165/10 (10, 4) | 163/10 (9, 4) | 42/42 | 3/11 | 9/35 | no | 0.73 | 4.1 | not captured | 39.3 | 82.5 | 94.8 | 128.9 | 7.4 | 0.025 | 18.1 |
| base | 1.75 | 15.0 | 2 | 8/13 (0/11, 4/35) | 160/10 (11, 4) | 163/10 (11, 3) | 169/9 | 9/11 | 2/35 | yes | 0.77 | 1.2 | captured | 40.9 | 88.5 | 117.4 | 121.5 | 8.1 | 0.029 | 20.1 |
| glu | 1.75 | 15.0 | 0 | 14/10 (0/11, 2/35) | 146/8 (9, 0) | 18/17 (3, 8) | 10/13 | 0/11 | 7/35 | no | 0.22 | 11.9 | no prior bump | 3.4 | 14.1 | 11.4 | 101.1 | 1.4 | 0.003 | 19.8 |
| glu | 1.75 | 15.0 | 1 | 10/12 (1/11, 4/35) | 138/10 (9, 4) | 131/10 (8, 4) | 12/15 | 2/11 | 7/35 | no | 0.22 | 13.6 | not captured | 5.8 | 20.4 | 18.9 | 100.3 | 1.8 | 0.005 | 18.8 |
| glu | 1.75 | 15.0 | 2 | 8/10 (0/11, 0/35) | 34/12 (9, 4) | 10/12 (0, 5) | 8/11 | 0/11 | 4/35 | no | 0.31 | 13.2 | no prior bump | 2.7 | 11.5 | 9.2 | 82.3 | 1.1 | 0.003 | 18.8 |
| ach | 1.75 | 15.0 | 0 | 14/59 (0/11, 12/35) | 200/8 (11, 0) | 193/9 (11, 1) | 193/10 | 11/11 | 1/35 | yes | 0.75 | 1.4 | captured | 54.8 | 99.2 | 149.2 | 158.5 | 10.3 | 0.041 | 24.9 |
| ach | 1.75 | 15.0 | 1 | 10/59 (1/11, 12/35) | 193/10 (10, 4) | 192/10 (11, 4) | 194/9 | 11/11 | 3/35 | yes | 0.77 | 1.4 | captured | 53.8 | 98.4 | 148.6 | 158.6 | 10.0 | 0.037 | 28.7 |
| ach | 1.75 | 15.0 | 2 | 8/52 (0/11, 9/35) | 188/10 (11, 4) | 189/10 (11, 3) | 191/9 | 11/11 | 2/35 | yes | 0.78 | 1.4 | captured | 53.1 | 95.6 | 147.8 | 154.9 | 9.9 | 0.039 | 28.7 |
| sign-class | 1.75 | 15.0 | 0 | 14/10 (0/11, 2/35) | 167/8 (9, 0) | 169/9 (10, 1) | 164/10 | 10/11 | 1/35 | yes | 0.74 | 1.3 | no prior bump | 40.8 | 89.2 | 116.1 | 128.9 | 8.0 | 0.034 | 22.4 |
| sign-class | 1.75 | 15.0 | 1 | 10/12 (1/11, 4/35) | 165/10 (10, 4) | 163/10 (9, 4) | 42/42 | 3/11 | 9/35 | no | 0.73 | 4.1 | not captured | 39.3 | 82.5 | 94.8 | 128.9 | 7.4 | 0.031 | 18.1 |
| sign-class | 1.75 | 15.0 | 2 | 8/13 (0/11, 4/35) | 160/10 (11, 4) | 163/10 (11, 3) | 169/9 | 9/11 | 2/35 | yes | 0.77 | 1.2 | captured | 40.9 | 88.5 | 117.4 | 121.5 | 8.1 | 0.034 | 17.9 |
| sign-abs | 1.75 | 15.0 | 0 | 14/10 (0/11, 2/35) | 168/8 (10, 0) | 166/9 (10, 1) | 163/10 | 9/11 | 1/35 | yes | 0.74 | 1.2 | no prior bump | 40.4 | 87.5 | 117.2 | 123.7 | 8.3 | 0.028 | 34.4 |
| sign-abs | 1.75 | 15.0 | 1 | 10/12 (1/11, 4/35) | 165/10 (10, 4) | 158/10 (10, 4) | 161/9 | 9/11 | 3/35 | yes | 0.76 | 1.2 | captured | 40.2 | 87.0 | 117.8 | 129.2 | 8.5 | 0.03 | 24.7 |
| sign-abs | 1.75 | 15.0 | 2 | 8/10 (0/11, 0/35) | 158/10 (10, 4) | 160/10 (10, 3) | 165/9 | 9/11 | 2/35 | yes | 0.77 | 1.1 | no prior bump | 39.9 | 86.4 | 114.5 | 120.9 | 8.1 | 0.029 | 18.4 |

## Summary

| gE | gD 15, config | runs | persist (n) | bump Hz at 5 s | out Hz | vs | PEN | Delta7 | GLNO (5 s) | GLNO (pulse) | captured / with prior bump elsewhere |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 | base | 3 | 3 | 202 | 9.5 | 0.77 | 48.4 | 100.1 | 134.0 | 147.3 | 3/3 |
| 2.0 | glu | 3 | 3 | 182 | 9.5 | 0.77 | 40.2 | 95.2 | 116.4 | 125.3 | 2/2 |
| 2.0 | ach | 3 | 3 | 233 | 9.5 | 0.78 | 64.7 | 109.7 | 169.4 | 172.7 | 3/3 |
| 2.0 | sign-class | 3 | 3 | 202 | 9.5 | 0.77 | 48.4 | 100.1 | 134.0 | 147.3 | 3/3 |
| 2.0 | sign-abs | 3 | 3 | 201 | 9.5 | 0.77 | 48.5 | 99.4 | 135.7 | 144.5 | 3/3 |
| 1.75 | base | 3 | 2 | 125 | 20.3 | 0.75 | 40.3 | 86.7 | 109.4 | 126.4 | 1/2 |
| 1.75 | glu | 3 | 0 | 10 | 12.8 | 0.25 | 4.0 | 15.3 | 13.2 | 94.6 | 0/1 |
| 1.75 | ach | 3 | 3 | 193 | 9.5 | 0.77 | 53.9 | 97.7 | 148.5 | 157.3 | 3/3 |
| 1.75 | sign-class | 3 | 2 | 125 | 20.3 | 0.75 | 40.3 | 86.7 | 109.4 | 126.4 | 1/2 |
| 1.75 | sign-abs | 3 | 3 | 163 | 9.5 | 0.76 | 40.2 | 87.0 | 116.5 | 124.6 | 1/1 |

### Answers

**Does the receptor model change anything in the ring?** No. `sign-class` is bit-identical to `base` on every ring
metric at both gains and all three seeds (pre / 0.5 / 2 / 5 s in and out means, cells above threshold, vector
strength, centre, PEN, Delta7, GLNO, PEG, ER/ExR rates), including the seed-1 gE-1.75 collapse to 42 Hz; the only
field that moves is the rest-of-brain rate (0.033 -> 0.039 Hz), which shows the stage was active. `sign-abs` moves the
ring by a few Hz through the 1,487 ER/ExR input entries the `abs` rule flips (5 s bump 200.7 / 199.7 / 203.0 vs base
201.0 / 201.4 / 204.3 at gE 2; 162.9 / 161.4 / 165.0 vs 164.0 / 42.0 / 169.3 at gE 1.75) -- inside seed scatter. This
matches section 2: 0 of the 27,553 EPG / PEN / Delta7 input entries change sign under any net rule. The receptor data
also confirm the sign of the loop `cx_wedge.md` found: the glutamate onto EPG from ExR6 (6,236 raw syn), ExR5 (6,057),
Delta7 (4,294) and ExR4 (1,660) -- 18,247 of EPG's 201,371 raw input synapses (9.1 %; 19,110 = 9.5 % for all
glutamatergic inputs) -- is fast GluCl inhibition by the E-PG driver profile (Davis PB_2: GluClalpha 10,283 vs iGluR
911 TPM), so the "untuned global feedback" is inhibitory as modelled; PEN's only profile is the Mi1-contaminated Davis
PB_3 driver.

**Does the silent PEN <-> GLNO loop matter?** Yes, and its sign decides the persistence window. EPG in-rate at 5 s per
seed (seeds 0-2 | 3-5): gE 2.0 base 201 / 201 / 204 | 196 / 192 / 201, glu 180 / 181 / 184 | 179 / 176 / 182, ach 234
/ 233 / 230 | 229 / 83 / 235; gE 1.75 base 164 / 42 / 169 | 46 / 160 / 167, glu 10 / 12 / 8 | 11 / 10 / 15, ach 193 /
194 / 191 | 191 / 179 / 193. Persistence over six seeds: gE 2.0 base 5/6, glu 5/6, ach 4/6; gE 1.75 base 3/6, **glu
0/6**, ach 5/6. A glutamatergic GLNO (the MaleCNS T-bar majority call, 51 % at confidence 0.48) abolishes the bump at
gE 1.75 in every seed (vector strength 0.04-0.31, PEN 0.5-5.8 Hz, Delta7 10-20 Hz) and costs ~11 % of the bump rate at
gE 2; a cholinergic GLNO holds the bump in 6/6 at gE 1.75, adds ~15 % at gE 2 (PEN 53-66 Hz vs 40-48; GLNO 140-174 Hz)
and produces a spontaneous pre-pulse bump in 11 of 12 runs. Magnitudes: a signed GLNO delivers 33.0 mV per PEN per
volley (two contralateral GLNO per PEN, every edge above the cap), against the EPG volley of +159.8 mV per PEN at gE 2
(20.6 %) and +139.8 at gE 1.75 (23.6 %) -- the earlier "about 40 %" was the gain-x1 figure. PEN -> GLNO is
contralateral as well (GLNO_L reads only R-glomerulus PENs, 1,938 raw syn; GLNO_R only L, 1,558), so the loop closes
within one PB side over 21 PEN partners at each end with no wedge structure; signing GLNO also signs its 1,327 non-PEN
output synapses, including GLNO -> GLNO 400 syn (self-excitation under `ach`, self-inhibition under `glu`).

**What GLNO is.** No expression source profiles it. The two EM predictions both favour an inhibitory GLNO: MaleCNS
T-bars glutamate 51 % / ACh 37 % (all three consensus columns `unclear`), and FlyWire SD1 `top_nt` for the four cells
GABA x3 (confidence 0.30-0.33) / glutamate x1 (0.30) (`data/external/typing/schlegel2024_Supplemental_file1_neuron_annotations.tsv`).
That is the case that abolishes the bump at gE 1.75 -- i.e. the working compass gains of `cx_wedge.md` were tuned
against a silent GLNO, and a GLNO=gaba gain scan is the next compass experiment (`scripts/cx_wedge.py --sim
--no-structure --nt-override GLNO=gaba --out out/cx_glno_gaba`). Because the label rests on two low-confidence EM
predictions and not on expression data, GLNO is not added to `TYPE_NT_OVERRIDE`.

**Caveats (skeptic).** The persist criterion (>= 8/11 in, <= 3/35 out) is brittle at the boundary: four seed-3-5 runs
with a clean 191-235 Hz bump, 10-11/11 inside cells and vs 0.76-0.78 count as "no" because 4 outside cells sit above
22 Hz; ach gE 2 seed 4 is "no" because the pre-pulse ring was already saturated (in 81 / out 57 Hz). The pre-pulse
state is seed-dependent enough to change what the pulse is asked to do (from a quiet ring to a fully formed bump at the
driven tile), so the capture counts (denominators 1-3 runs) carry little weight; read the rate / vs columns.

## 4. Round 3: the GLNO = gaba gain scan (cluster batch `r3-glno-gaba-5f4869`)

Question: the round-2 gains (EPG <-> PEN, PEG x gE; Delta7 -> EPG x gD, Delta7 -> PEN x1; compass adaptation 0) were
tuned against a silent GLNO, and both EM predictions favour an inhibitory GLNO (section 3). Where does a confined,
persistent, capturable bump exist once the PEN <-> GLNO loop is closed inhibitory, is that window narrower or shifted
against the silent-GLNO window, and does the bump rate change?

**Protocol.** `scripts/cx_glno.py --run gaba` (config `gaba` = `TYPE_NT_OVERRIDE` + {GLNO: gaba}, compiled from the raw
MaleCNS files into the job's `out/cache_72164311/`; GLNO nt `gaba`, sign -1, GLNO -> PEN W sum -16,371, PEN -> GLNO
+3,496, sum|W| 121,478,280 vs 121,460,584 on the adopted cache), the section-3 protocol (FlyBrain on the full connectome,
compass adaptation 0, 10 Hz Poisson background on all 46 EPG, wedges 0-3 at +40 Hz for 2 s, 5 s free, cuda graphs,
torch sparse backend), grid gE in {1.75, 2, 2.25, 2.5} x gD in {8, 15, 25, 40} (Delta7 -> EPG only), seeds 0-2 = 48 runs
in four jobs (one per gE, 12 runs each, 6.5-7.9 min per job; the anchor job 2.2 min on a B200, torch 2.11.0+cu128, CUDA asserted). Comparison:
config `glu` on the same gD grid at gE 2 and 2.25 (24 runs, two jobs, cache `cache_c51b23e2`). Anchor: config `base` at
gE 2 / gD 15 from the cluster's shared cache (`--no-scratch`; the shared cache now carries `TYPE_NT_OVERRIDE`, sum|W|
121,460,584). Submitted 2026-09-12 06:44 UTC, all seven jobs exit 0 by 06:52 UTC. Data: `out/cx_glno_gaba_gE{1.75,2,
2.25,2.5}.json` + `.txt`, `out/cx_glno_glu_gE{2,2.25}.json` + `.txt`, `out/cx_glno_base_r3.json` + `.txt`; tables
`out/cx_glno_gaba_table.{md,csv}` and `out/cx_glno_gaba_table_summary.csv` from `python scripts/cx_glno.py --report --files
"out/cx_glno_gaba_*.json" "out/cx_glno_glu_gE*.json" out/cx_glno_base_r3.json --table cx_glno_gaba_table`.

**Two identity checks (every field of every row compared, `wall_s` / `cache_dir` excluded).** (i) The base anchor from the
shared cache is bit-identical to the round-2 base rows (3/3 seeds), so the shared cache and the round-2 scratch compile are
the same connectome. (ii) `gaba` and `glu` are bit-identical at all 24 shared grid points (gE 2 / 2.25 x gD 8 / 15 / 25 /
40 x 3 seeds, 0 differing fields): under `NT_SIGN` both labels are fast -1 and the receptor model is off, so the only
thing the label changes is the sign of GLNO's 17,698 output synapses. The round-2 `glu` rows at gE 1.75 / gD 15 (10 / 12 /
8 Hz) and gE 2 / gD 15 (180 / 181 / 184 Hz) are therefore the `gaba` rows at those points, and the six `glu` runs the task
asked for are a same-code replication rather than a second condition. The `glu` rows are omitted from the per-run table
below (they are in `out/cx_glno_gaba_table.md`).

**Magnitudes at the scanned gains** (`cx_wedge.effective_weights` on the local adopted cache, CPU): the EPG volley onto a
PEN is +139.8 / +159.8 / +179.8 / +199.7 mV at gE 1.75 / 2 / 2.25 / 2.5 (per-cell range 95.8-297.0; PEN fan-in scale 1.00
throughout), the Delta7 volley onto a PEN -45.2 mV (x1 at every gE), and the signed GLNO volley 33.0 mV per PEN (two
contralateral GLNO, every edge above the cap) is 23.6 / 20.7 / 18.4 / 16.5 % of the EPG volley. The Delta7 volley onto
an EPG at gD 15 is -341 / -335 / -329 / -322 mV (the EPG fan-in scale falls slightly with gE).

### Summary per grid point (3 seeds each; rates at 5 s after release; rho = gD / gE^2)

| config | gE | gD | rho = gD / gE^2 | persist (n/3) | bump Hz at 5 s (per seed) | out Hz (per seed) | in > 22 Hz (/11) | out > 22 Hz (/35) | vs (per seed) | PEN | Delta7 | GLNO 5 s | GLNO pulse | captured / prior bump elsewhere | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 2.0 | 15 | 3.75 | 3 | 201 / 201 / 204 | 10 / 10 / 9 | 11/10/11 | 1/3/2 | 0.76 / 0.78 / 0.78 | 48 | 100 | 134 | 147 | 3/3 | window (round-2 anchor, bit-identical to section 3) |
| gaba | 1.75 | 8 | 2.61 | 3 | 154 / 148 / 152 | 10 / 10 / 9 | 9/9/9 | 1/3/2 | 0.73 / 0.75 / 0.76 | 33 | 84 | 96 | 103 | 1/1 | window |
| gaba | 1.75 | 15 | 4.90 | 0 | 10 / 12 / 8 | 13 / 15 / 11 | 0/2/0 | 7/7/4 | 0.22 / 0.22 / 0.31 | 4 | 15 | 13 | 95 | 0/1 | dead (3/3 by 2 s) |
| gaba | 1.75 | 25 | 8.16 | 0 | 10 / 8 / 7 | 12 / 12 / 11 | 0/0/0 | 7/6/4 | 0.21 / 0.17 / 0.31 | 3 | 13 | 9 | 78 | 0/1 | dead (3/3 by 2 s) |
| gaba | 1.75 | 40 | 13.06 | 0 | 10 / 8 / 7 | 10 / 12 / 10 | 0/0/0 | 1/5/3 | 0.10 / 0.17 / 0.24 | 2 | 11 | 7 | 53 | 0/1 | dead (3/3 by 1 s) |
| gaba | 2.0 | 8 | 2.00 | 3 | 194 / 192 / 197 | 10 / 10 / 9 | 9/10/10 | 1/3/2 | 0.76 / 0.78 / 0.79 | 42 | 99 | 121 | 128 | 2/2 | window |
| gaba | 2.0 | 15 | 3.75 | 3 | 180 / 181 / 184 | 10 / 10 / 9 | 10/10/10 | 1/3/2 | 0.75 / 0.77 / 0.78 | 40 | 95 | 116 | 125 | 2/2 | window |
| gaba | 2.0 | 25 | 6.25 | 2 | 161 / 42 / 163 | 10 / 42 / 9 | 9/3/9 | 1/9/2 | 0.73 / 0.73 / 0.77 | 38 | 85 | 101 | 119 | 1/2 | 2/3; seed 1 jump at 3-5 s |
| gaba | 2.0 | 40 | 10.00 | 1 | 135 / 8 / 8 | 10 / 19 / 16 | 9/0/0 | 1/12/9 | 0.70 / 0.27 / 0.47 | 17 | 40 | 48 | 104 | 0/1 | 1/3; seeds 1-2 dead by 1-2 s |
| gaba | 2.25 | 8 | 1.58 | 2 | 10 / 235 / 237 | 75 / 10 / 9 | 0/11/11 | 10/3/2 | 0.79 / 0.79 / 0.79 | 49 | 109 | 131 | 139 | 2/3 | 2/3; seed 0 stiff (pre-pulse bump at wedges 12-14 not displaced) |
| gaba | 2.25 | 15 | 2.96 | 3 | 214 / 216 / 223 | 10 / 10 / 9 | 10/11/11 | 1/3/2 | 0.76 / 0.78 / 0.79 | 48 | 105 | 132 | 141 | 3/3 | window (seed 0's pre-pulse bump captured) |
| gaba | 2.25 | 25 | 4.94 | 3 | 196 / 199 / 192 | 10 / 10 / 9 | 10/10/10 | 1/3/2 | 0.75 / 0.77 / 0.77 | 45 | 97 | 126 | 135 | 3/3 | window |
| gaba | 2.25 | 40 | 7.90 | 1 | 171 / 45 / 7 | 10 / 47 / 52 | 10/3/0 | 1/9/11 | 0.74 / 0.75 / 0.79 | 42 | 84 | 108 | 126 | 0/2 | 1/3; seed 1 jump at 3-5 s, seed 2 jump to the opposite side at 1-2 s |
| gaba | 2.5 | 8 | 1.28 | 2 | 10 / 264 / 265 | 102 / 10 / 9 | 0/11/11 | 13/3/2 | 0.79 / 0.80 / 0.80 | 63 | 122 | 162 | 160 | 2/3 | 2/3; seed 0 stiff |
| gaba | 2.5 | 15 | 2.40 | 2 | 10 / 250 / 249 | 83 / 10 / 9 | 0/11/11 | 11/3/2 | 0.80 / 0.80 / 0.80 | 55 | 113 | 145 | 149 | 2/3 | 2/3; seed 0 stiff |
| gaba | 2.5 | 25 | 4.00 | 3 | 227 / 225 / 222 | 10 / 10 / 9 | 10/10/10 | 1/3/2 | 0.77 / 0.79 / 0.79 | 52 | 106 | 143 | 150 | 3/3 | window |
| gaba | 2.5 | 40 | 6.40 | 2 | 196 / 56 / 191 | 10 / 52 / 9 | 10/3/10 | 1/9/2 | 0.75 / 0.78 / 0.77 | 49 | 96 | 129 | 141 | 2/3 | 2/3; seed 1 jump at 3-5 s |

Verdicts: **window** = persists in 3/3 (>= 8/11 inside and <= 3/35 outside cells above 22 Hz at 5 s) and no run in which a
pulse failed to capture a prior bump; **jump** = a bump that survives at full rate (vs 0.73-0.79) but leaves the driven
block; **stiff** = a spontaneous pre-pulse bump elsewhere that the +40 Hz pulse does not displace; **dead** = no bump by 2 s.
The jumps are one event in every case where they occur: with seed 1 the bump sits at wedges 0-3 through 3 s and is at
wedges 3-5 (133 / 234 / 159 Hz, centre 4.1) at 5 s, the same profile at gaba gE 2 / gD 25, 2.25 / 40, 2.5 / 40 and at
the round-2 base gE 1.75 / gD 15 (133 / 231 / 160) -- the seed-1 background realisation moves the bump once gD is high
enough (rho >= 6.25 with GLNO signed, 4.9 silent); at gE 2.25 / gD 40 seed 2 the bump jumps to the opposite side (wedges
12-14, 173 / 197 / 176 Hz) between 1 and 2 s and is at wedges 10-12 at 5 s. The stiff cases are all seed 0 at gE >= 2.25
with gD <= 15: the 1 s settle produces a 3-4-wedge bump at wedges 12-14 (225-278 Hz per wedge) that the pulse captures at
gE 2.25 / gD 15 and 25 and at 2.5 / 25 but not at 2.25 / 8, 2.5 / 8 or 2.5 / 15 (in 9-10 Hz, out 75-102 Hz at 5 s).

### Per run

| config | gE | gD | seed | before pulse in/out (>22 Hz in, out) | 0.5 s in/out (>22) | 2 s | 5 s in/out Hz | in >22 Hz | out >22 Hz | persists | vs | centre wedge | capture | PEN | Delta7 | GLNO | GLNO during pulse | ER/ExR | rest | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 2.0 | 15.0 | 0 | 14/64 (0/11, 12/35) | 204/8 (11, 0) | 201/9 (10, 1) | 201/10 | 11/11 | 1/35 | yes | 0.76 | 1.5 | captured | 48.7 | 101.2 | 132.8 | 147.9 | 9.6 | 0.033 | 30.9 |
| base | 2.0 | 15.0 | 1 | 10/60 (1/11, 13/35) | 201/10 (10, 4) | 197/10 (10, 4) | 201/9 | 10/11 | 3/35 | yes | 0.78 | 1.5 | captured | 48.2 | 99.5 | 137.2 | 150.9 | 9.8 | 0.034 | 29.8 |
| base | 2.0 | 15.0 | 2 | 8/56 (0/11, 9/35) | 203/10 (10, 4) | 200/10 (11, 3) | 204/9 | 11/11 | 2/35 | yes | 0.78 | 1.4 | captured | 48.3 | 99.7 | 132.0 | 143.1 | 9.7 | 0.033 | 26.1 |
| gaba | 1.75 | 8.0 | 0 | 14/10 (0/11, 2/35) | 160/8 (9, 0) | 155/9 (9, 1) | 154/10 | 9/11 | 1/35 | yes | 0.73 | 1.2 | no prior bump | 32.9 | 85.5 | 96.5 | 105.2 | 7.3 | 0.023 | 26.9 |
| gaba | 1.75 | 8.0 | 1 | 10/13 (1/11, 4/35) | 152/10 (10, 4) | 149/10 (9, 4) | 148/9 | 9/11 | 3/35 | yes | 0.75 | 1.2 | captured | 32.4 | 83.7 | 96.0 | 107.2 | 7.1 | 0.025 | 23.7 |
| gaba | 1.75 | 8.0 | 2 | 8/10 (0/11, 0/35) | 150/10 (9, 4) | 149/10 (9, 3) | 153/9 | 9/11 | 2/35 | yes | 0.76 | 1.1 | no prior bump | 32.8 | 82.7 | 96.3 | 97.6 | 7.1 | 0.025 | 20.8 |
| gaba | 1.75 | 15.0 | 0 | 14/10 (0/11, 2/35) | 146/8 (9, 0) | 18/17 (3, 8) | 10/13 | 0/11 | 7/35 | no | 0.22 | 11.9 | no prior bump | 3.4 | 14.1 | 11.4 | 101.1 | 1.4 | 0.003 | 22.5 |
| gaba | 1.75 | 15.0 | 1 | 10/12 (1/11, 4/35) | 138/10 (9, 4) | 131/10 (8, 4) | 12/15 | 2/11 | 7/35 | no | 0.22 | 13.6 | not captured | 5.8 | 20.4 | 18.9 | 100.3 | 1.8 | 0.005 | 31.1 |
| gaba | 1.75 | 15.0 | 2 | 8/10 (0/11, 0/35) | 34/12 (9, 4) | 10/12 (0, 5) | 8/11 | 0/11 | 4/35 | no | 0.31 | 13.2 | no prior bump | 2.7 | 11.5 | 9.2 | 82.3 | 1.1 | 0.003 | 45.6 |
| gaba | 1.75 | 25.0 | 0 | 14/10 (0/11, 2/35) | 125/8 (9, 0) | 10/11 (0, 3) | 10/12 | 0/11 | 7/35 | no | 0.21 | 11.8 | no prior bump | 3.1 | 13.4 | 11.4 | 83.7 | 1.3 | 0.003 | 30.3 |
| gaba | 1.75 | 25.0 | 1 | 10/12 (1/11, 4/35) | 26/16 (7, 10) | 7/10 (0, 4) | 8/12 | 0/11 | 6/35 | no | 0.17 | 12.0 | not captured | 3.4 | 13.7 | 5.9 | 90.8 | 1.1 | 0.003 | 34.3 |
| gaba | 1.75 | 25.0 | 2 | 8/9 (0/11, 0/35) | 22/13 (5, 4) | 10/11 (0, 4) | 7/11 | 0/11 | 4/35 | no | 0.31 | 13.1 | no prior bump | 2.6 | 10.9 | 9.0 | 59.1 | 1.0 | 0.003 | 23.5 |
| gaba | 1.75 | 40.0 | 0 | 14/10 (0/11, 2/35) | 36/9 (8, 1) | 8/9 (0, 1) | 10/10 | 0/11 | 1/35 | no | 0.1 | 11.8 | no prior bump | 1.4 | 9.8 | 2.0 | 52.6 | 0.9 | 0.002 | 28.4 |
| gaba | 1.75 | 40.0 | 1 | 10/12 (1/11, 4/35) | 12/13 (1, 6) | 7/10 (0, 4) | 8/12 | 0/11 | 5/35 | no | 0.17 | 12.4 | not captured | 3.6 | 13.8 | 13.6 | 73.3 | 1.3 | 0.003 | 24.3 |
| gaba | 1.75 | 40.0 | 2 | 8/9 (0/11, 0/35) | 15/12 (2, 4) | 10/11 (0, 4) | 7/10 | 0/11 | 3/35 | no | 0.24 | 13.1 | no prior bump | 1.7 | 8.8 | 6.7 | 32.9 | 0.8 | 0.002 | 27.8 |
| gaba | 2.0 | 8.0 | 0 | 61/48 (4/11, 8/35) | 198/8 (10, 0) | 195/9 (10, 1) | 194/10 | 9/11 | 1/35 | yes | 0.76 | 1.2 | no prior bump | 41.5 | 98.8 | 120.9 | 128.6 | 9.0 | 0.03 | 19.3 |
| gaba | 2.0 | 8.0 | 1 | 10/58 (1/11, 12/35) | 195/10 (11, 4) | 194/10 (10, 4) | 192/9 | 10/11 | 3/35 | yes | 0.78 | 1.2 | captured | 42.2 | 100.1 | 121.4 | 128.8 | 9.2 | 0.031 | 17.4 |
| gaba | 2.0 | 8.0 | 2 | 8/49 (0/11, 9/35) | 190/10 (10, 4) | 192/10 (10, 3) | 197/9 | 10/11 | 2/35 | yes | 0.79 | 1.1 | captured | 42.2 | 99.0 | 119.6 | 125.8 | 8.8 | 0.03 | 19.6 |
| gaba | 2.0 | 15.0 | 0 | 21/11 (6/11, 2/35) | 190/8 (10, 0) | 184/9 (10, 1) | 180/10 | 10/11 | 1/35 | yes | 0.75 | 1.2 | spontaneous at driven tile | 40.3 | 95.6 | 114.8 | 126.9 | 8.7 | 0.029 | 22.2 |
| gaba | 2.0 | 15.0 | 1 | 10/51 (1/11, 12/35) | 181/10 (11, 4) | 180/10 (10, 4) | 181/9 | 10/11 | 3/35 | yes | 0.77 | 1.2 | captured | 39.8 | 95.4 | 118.2 | 126.7 | 8.8 | 0.032 | 34.5 |
| gaba | 2.0 | 15.0 | 2 | 8/17 (0/11, 12/35) | 180/10 (11, 4) | 184/10 (11, 3) | 184/9 | 10/11 | 2/35 | yes | 0.78 | 1.2 | captured | 40.6 | 94.7 | 116.3 | 122.4 | 8.6 | 0.03 | 21.7 |
| gaba | 2.0 | 25.0 | 0 | 15/10 (0/11, 2/35) | 167/8 (9, 0) | 165/9 (10, 1) | 161/10 | 9/11 | 1/35 | yes | 0.73 | 1.2 | no prior bump | 38.0 | 89.3 | 106.2 | 124.6 | 7.9 | 0.027 | 30.2 |
| gaba | 2.0 | 25.0 | 1 | 10/20 (1/11, 11/35) | 162/10 (10, 4) | 160/10 (10, 4) | 42/42 | 3/11 | 9/35 | no | 0.73 | 4.1 | not captured | 37.4 | 80.9 | 88.4 | 119.8 | 7.3 | 0.023 | 18.5 |
| gaba | 2.0 | 25.0 | 2 | 8/21 (0/11, 10/35) | 152/10 (10, 4) | 158/10 (11, 3) | 163/9 | 9/11 | 2/35 | yes | 0.77 | 1.2 | captured | 37.5 | 84.8 | 108.3 | 112.7 | 7.7 | 0.028 | 23.8 |
| gaba | 2.0 | 40.0 | 0 | 14/10 (0/11, 2/35) | 143/8 (9, 0) | 117/9 (9, 1) | 135/10 | 9/11 | 1/35 | yes | 0.7 | 1.2 | no prior bump | 34.1 | 78.2 | 95.7 | 110.7 | 6.6 | 0.022 | 27.2 |
| gaba | 2.0 | 40.0 | 1 | 10/12 (1/11, 4/35) | 135/10 (10, 4) | 9/12 (0, 5) | 8/19 | 0/11 | 12/35 | no | 0.27 | 12.1 | not captured | 9.7 | 20.6 | 25.5 | 109.7 | 2.1 | 0.006 | 20.7 |
| gaba | 2.0 | 40.0 | 2 | 8/9 (0/11, 0/35) | 30/12 (9, 4) | 11/12 (1, 6) | 8/16 | 0/11 | 9/35 | no | 0.47 | 13.6 | no prior bump | 7.8 | 21.0 | 21.9 | 92.6 | 2.1 | 0.006 | 25.7 |
| gaba | 2.25 | 8.0 | 0 | 14/76 (0/11, 12/35) | 9/76 (0, 10) | 8/74 (0, 11) | 10/75 | 0/11 | 10/35 | no | 0.79 | 13.2 | not captured | 45.8 | 103.6 | 117.3 | 115.9 | 10.4 | 0.034 | 25.2 |
| gaba | 2.25 | 8.0 | 1 | 10/69 (1/11, 14/35) | 240/10 (11, 4) | 241/10 (11, 4) | 235/9 | 11/11 | 3/35 | yes | 0.79 | 1.5 | captured | 50.3 | 111.8 | 137.8 | 150.3 | 10.9 | 0.037 | 22.4 |
| gaba | 2.25 | 8.0 | 2 | 8/68 (0/11, 10/35) | 231/10 (11, 4) | 241/10 (11, 3) | 237/9 | 11/11 | 2/35 | yes | 0.79 | 1.4 | captured | 50.9 | 112.3 | 137.7 | 151.5 | 10.8 | 0.037 | 33.8 |
| gaba | 2.25 | 15.0 | 0 | 14/71 (0/11, 12/35) | 226/8 (11, 0) | 220/9 (10, 1) | 214/10 | 10/11 | 1/35 | yes | 0.76 | 1.6 | captured | 47.1 | 104.6 | 130.6 | 139.3 | 10.0 | 0.033 | 21.8 |
| gaba | 2.25 | 15.0 | 1 | 10/65 (1/11, 13/35) | 218/10 (10, 4) | 213/10 (10, 4) | 216/9 | 11/11 | 3/35 | yes | 0.78 | 1.6 | captured | 47.5 | 104.7 | 132.1 | 142.7 | 10.2 | 0.035 | 32.0 |
| gaba | 2.25 | 15.0 | 2 | 8/63 (0/11, 9/35) | 216/10 (11, 4) | 219/10 (11, 3) | 223/9 | 11/11 | 2/35 | yes | 0.79 | 1.5 | captured | 49.7 | 105.5 | 134.0 | 141.9 | 10.3 | 0.037 | 31.2 |
| gaba | 2.25 | 25.0 | 0 | 14/64 (0/11, 11/35) | 198/8 (10, 0) | 197/9 (10, 1) | 196/10 | 10/11 | 1/35 | yes | 0.75 | 1.6 | captured | 45.1 | 97.6 | 122.9 | 141.2 | 9.3 | 0.031 | 36.7 |
| gaba | 2.25 | 25.0 | 1 | 10/59 (1/11, 13/35) | 194/10 (10, 4) | 199/10 (10, 4) | 199/9 | 10/11 | 3/35 | yes | 0.77 | 1.6 | captured | 45.0 | 98.3 | 130.7 | 128.5 | 9.5 | 0.033 | 30.4 |
| gaba | 2.25 | 25.0 | 2 | 8/57 (0/11, 9/35) | 193/10 (10, 4) | 195/10 (10, 3) | 192/9 | 10/11 | 2/35 | yes | 0.77 | 1.5 | captured | 45.3 | 94.6 | 123.0 | 134.3 | 9.2 | 0.032 | 26.8 |
| gaba | 2.25 | 40.0 | 0 | 163/10 (10/11, 2/35) | 177/8 (10, 0) | 171/9 (10, 1) | 171/10 | 10/11 | 1/35 | yes | 0.74 | 1.6 | spontaneous at driven tile | 42.0 | 89.4 | 114.8 | 127.0 | 8.3 | 0.027 | 29.5 |
| gaba | 2.25 | 40.0 | 1 | 10/51 (1/11, 12/35) | 162/10 (9, 4) | 164/10 (10, 4) | 45/47 | 3/11 | 9/35 | no | 0.75 | 4.1 | not captured | 44.7 | 88.2 | 106.4 | 130.7 | 8.4 | 0.029 | 31.1 |
| gaba | 2.25 | 40.0 | 2 | 8/49 (0/11, 8/35) | 153/10 (10, 4) | 11/53 (0, 11) | 7/52 | 0/11 | 11/35 | no | 0.79 | 11.2 | not captured | 39.3 | 73.8 | 102.2 | 120.7 | 7.6 | 0.025 | 26.0 |
| gaba | 2.5 | 8.0 | 0 | 14/89 (0/11, 14/35) | 9/89 (0, 12) | 8/88 (0, 12) | 10/102 | 0/11 | 13/35 | no | 0.79 | 12.7 | not captured | 71.9 | 127.5 | 170.2 | 157.7 | 14.0 | 0.047 | 17.9 |
| gaba | 2.5 | 8.0 | 1 | 10/107 (1/11, 17/35) | 267/10 (11, 4) | 266/10 (11, 4) | 264/9 | 11/11 | 3/35 | yes | 0.8 | 1.5 | captured | 59.0 | 120.4 | 158.6 | 161.1 | 12.3 | 0.043 | 18.9 |
| gaba | 2.5 | 8.0 | 2 | 8/105 (0/11, 13/35) | 265/10 (11, 4) | 270/10 (11, 3) | 265/9 | 11/11 | 2/35 | yes | 0.8 | 1.4 | captured | 58.8 | 119.4 | 156.1 | 160.3 | 12.2 | 0.042 | 19.9 |
| gaba | 2.5 | 15.0 | 0 | 14/81 (0/11, 13/35) | 9/79 (0, 11) | 8/79 (0, 11) | 10/83 | 0/11 | 11/35 | no | 0.8 | 13.2 | not captured | 53.9 | 109.7 | 135.5 | 135.7 | 11.6 | 0.039 | 19.1 |
| gaba | 2.5 | 15.0 | 1 | 10/99 (1/11, 17/35) | 254/10 (11, 4) | 249/10 (11, 4) | 250/9 | 11/11 | 3/35 | yes | 0.8 | 1.6 | captured | 56.1 | 115.4 | 152.1 | 156.6 | 11.5 | 0.04 | 17.1 |
| gaba | 2.5 | 15.0 | 2 | 8/100 (0/11, 13/35) | 251/10 (11, 4) | 249/10 (11, 3) | 249/9 | 11/11 | 2/35 | yes | 0.8 | 1.5 | captured | 55.9 | 114.0 | 146.5 | 155.4 | 11.5 | 0.039 | 19.3 |
| gaba | 2.5 | 25.0 | 0 | 14/73 (0/11, 12/35) | 230/8 (10, 0) | 234/9 (10, 1) | 227/10 | 10/11 | 1/35 | yes | 0.77 | 1.6 | captured | 52.9 | 108.4 | 141.6 | 153.5 | 10.7 | 0.037 | 47.8 |
| gaba | 2.5 | 25.0 | 1 | 10/89 (1/11, 17/35) | 226/10 (10, 4) | 228/10 (10, 4) | 225/9 | 10/11 | 3/35 | yes | 0.79 | 1.6 | captured | 52.7 | 107.3 | 145.2 | 152.3 | 10.8 | 0.037 | 30.2 |
| gaba | 2.5 | 25.0 | 2 | 8/90 (0/11, 13/35) | 229/10 (10, 4) | 227/10 (10, 3) | 223/9 | 10/11 | 2/35 | yes | 0.79 | 1.6 | captured | 51.5 | 103.7 | 142.0 | 144.2 | 10.4 | 0.037 | 33.3 |
| gaba | 2.5 | 40.0 | 0 | 14/67 (0/11, 11/35) | 197/8 (10, 0) | 201/9 (10, 1) | 196/10 | 10/11 | 1/35 | yes | 0.75 | 1.7 | captured | 47.1 | 97.8 | 127.9 | 141.5 | 9.5 | 0.032 | 26.9 |
| gaba | 2.5 | 40.0 | 1 | 10/62 (1/11, 12/35) | 192/10 (10, 4) | 190/10 (10, 4) | 56/52 | 3/11 | 9/35 | no | 0.78 | 4.1 | not captured | 54.0 | 97.8 | 133.4 | 142.3 | 9.9 | 0.036 | 21.8 |
| gaba | 2.5 | 40.0 | 2 | 8/61 (0/11, 9/35) | 188/10 (10, 4) | 182/10 (10, 3) | 191/9 | 10/11 | 2/35 | yes | 0.77 | 1.6 | captured | 46.7 | 93.9 | 126.7 | 140.6 | 9.2 | 0.031 | 18.2 |

### Answers

**Where does a confined, persistent, capturable bump exist with the loop closed inhibitory?** At six of the sixteen grid
points, all with rho = gD / gE^2 between 2.0 and 4.9: gE 1.75 / gD 8 (148-154 Hz, PEN 32-33, captured 1/1 with prior
bump), 2 / 8 (192-197, 2/2), 2 / 15 (180-184, 2/2), 2.25 / 15 (214-223, 3/3), 2.25 / 25 (192-199, 3/3) and 2.5 / 25
(222-227, 3/3). At each the outside mean is 9-10 Hz with 1-3 of 35 cells above 22 Hz, vs 0.73-0.79, and the rest of the
brain 0.02-0.05 Hz. Above that band (rho >= 6.25) the bump jumps in seed 1 (gE 2 / gD 25, 2.25 / 40, 2.5 / 40) or dies
(gE 2 / gD 40 seeds 1-2 by 1-2 s; every gE 1.75 point with gD >= 15 in 3/3 seeds -- in 0/11 to 2/11 cells, PEN 2-6 Hz,
Delta7 11-20, vs 0.10-0.31); below it (rho <= 2.4 at gE >= 2.25) the ring forms a spontaneous bump the pulse cannot move
in seed 0.

**Narrower or shifted vs the silent-GLNO window?** (Corrected by the skeptic's seed-matched silent control, out/sk_base_gE{1.75,2,2.5}.json: over the same 12 grid points x 3 seeds both conditions give 21/36 persisting runs -- NOT narrower; the 3/3 window moves from silent {1.75/8, 2/15, 2/25, 2.5/25, 2.5/40} to gaba {2/8, 2/15, 2.5/25, and 1.75/8 at 3/6 over six seeds}; gaba gains 2/8 and 2.5/8 and loses 1.75/15, 2/25, 2.5/40; the session-9 silent window 'gE 1.75-2, gD 15-40' was a seed-0 statement and fails at 3 seeds at 1.75/25, 1.75/40, 2/40.) The original reading follows: shifted down in gD by a factor of about 2, and narrower at the low-gE
end. Silent GLNO (`cx_wedge.md` section 6, seed 0; round-2 seeds 0-5 where available): gE 1.75-2 with gD 15-40, rho
3.75-13 -- gE 2 / gD 15 persists 5/6, 2 / 25 and 2 / 40 persist at seed 0, 1.75 / 15 persists 3/6. With GLNO inhibitory
the same gE 2 needs gD 8-15 (3/3 each), gD 25 is 2/3 (seed-1 jump) and gD 40 is 1/3 (two deaths); at gE 1.75 only gD 8
works (3/3) and gD 15-40 is dead in 9/9 runs, where the silent ring still held 3/6 at gD 15. The window regains two gD
points only at gE 2.25 (gD 15-25) and the round-2 operating point gE 2 / gD 15 is at its upper edge in rho. Read as a
gain budget: the signed GLNO volley (33 mV, 16-24 % of the EPG volley on a PEN) is PEN inhibition proportional to PEN
activity itself, so it substitutes for part of the Delta7 -> EPG inhibition; the gD that was needed silent becomes too
much.

**Does the bump rate change?** Within the window it is 148-227 Hz: at the round-2 point gE 2 / gD 15 an inhibitory GLNO
takes the bump from 201-204 (base; 192-204 over six round-2 seeds) to 180-184 Hz (-10 %), PEN 48 -> 40 Hz, Delta7 100 ->
95, and the lowest persistent bump in the scan is 148-154 Hz at gE 1.75 / gD 8 (PEN 32-33, Delta7 83-86, GLNO 96 Hz).
That is still an order of magnitude above the animal's E-PG rates (no numeric animal rate is cited in these audits), so closing the loop inhibitory does not fix the rate; it lowers it by ~10 %
at fixed gains and by ~25 % at the low-gain edge of its window. GLNO itself fires at 96-145 Hz at 5 s inside the window (above Delta7's 83-108 Hz at every window point; 162 was the mean at gE 2.5 / gD 8, outside the window)dow
(driven by the +222 mV PEN volley it receives), higher than any compass type but Delta7.

**Caveats.** Three seeds per point; persistence at a point is a 3/3, 2/3 or worse count and the 2/3 points differ from
the 3/3 points by one seed-1 event, so the window edges are +-1 grid step. The persist criterion's brittleness noted in
section 3 applies (four outside cells above 22 Hz would count as "no"); in this scan every "no" is a real jump, a stiff
pre-pulse bump or a death, none is the 4-cell boundary case. The gD grid is coarse (8 / 15 / 25 / 40); the gE 1.75 window
may extend below gD 8 and the gE 2 window between 15 and 25, neither was sampled. GLNO stays out of `TYPE_NT_OVERRIDE`
(section 3: two low-confidence EM predictions, no expression profile); if it is ever adopted, gE 2 / gD 15 still works
but with a 10 % slower bump and no margin in gD, and gE 2.25 / gD 15-25 is the better-centred setting.

### Corrections (round-3 verification, `verify:exp:compass`)

* Seed-matched silent control (out/sk_base_gE{1.75,2,2.5}.json, seeds 0-2) and seeds 3-5 (out/sk_{base,gaba}_s345.json): persisting runs 21/36 in each condition; 1.75/8 is 3/6 for gaba (seeds 3-4 die at 17 / 10 Hz) against 5/6 silent, so the "lowest persistent bump 148-154 Hz" and "-25 % at the low-gain edge" clauses are withdrawn; the rate answer is -10 % at fixed gains (2/15: 180-184 vs 201-204 Hz).
* gaba 1.75/15: dead 3/3 by 5 s; seeds 0 and 2 by 2 s / 1 s, seed 1 only between 3 and 5 s (131-134 Hz at 2-3 s).
* The three seed-1 jump rows share the destination (wedges 3-5, centre 4.1) and the background wedges, not the amplitudes (133/234/159, 144/247/197, 182/270/238). The seed-0 pre-pulse bump at gE >= 2.25 is 206-287 Hz per wedge and occurs at every gD, not only <= 15.
* The persist criterion's 4-outside-cells boundary case fires at seed 5 in BOTH conditions (bump confined at 154-244 Hz, out mean 11.2 Hz); 3/3-vs-2/3 distinctions are within the criterion's own noise -- read the rate / vs columns. Capture tests per window point are 1-2, not 3.
* Job durations 367-471 s (12-run jobs), anchor 114 s. Dead gE 1.75 points: Delta7 8.8-20.4 Hz; rest of brain 0.023-0.037 Hz at window points.
* Consequence for the compass thread: the session-9 conclusions (ring-attractor wiring; Delta7 gain on Delta7->EPG only; the ExR/ER loop; 200 Hz bump) stand; gE 2 / gD 15 lies inside both windows; the robust operating points from the present data are gE 2 / gD 15 and gE 2.5 / gD 25. GLNO stays out of TYPE_NT_OVERRIDE (EM predictions only).

## 5. Round 4: the seed-matched grid finished (6 seeds x both GLNO conditions; cluster batch `r4-glno-d04de8`)

Question (round-3 critic follow-up 6): round 3 compared a 3-seed `gaba` scan against a 1-seed silent table, and the
skeptic's controls (seeds 0-2 silent on the same grid, seeds 3-5 at four points in both conditions) showed the window
edges moving by one seed. Which operating points hold the bump in **both** GLNO conditions over six seeds, i.e. which
compass setting is robust to the sign the data cannot fix?

**What was run.** One batch, three jobs, 0 failed, 8.3 min wall (run dir `<cluster-fs>/neurome/runs/r4-glno-d04de8`,
B200, torch 2.11.0+cu128, CUDA asserted in every job):

| job | command | runs | s |
|---|---|---|---|
| 1 | `cx_glno.py --run base --no-scratch --gains 2.25:8,2.25:15,2.25:25,2.25:40 --seeds 0,1,2 --out out/r4_base_gE2.25.json` | 12 | 320 |
| 2 | `cx_glno.py --run gaba --gains 2:8,2:15,2.25:15,2.25:25,2.5:25 --seeds 3,4,5 --out out/r4_gaba_s345b.json` | 15 | 433 |
| 3 | `cx_glno.py --run base --no-scratch --gains 2:8,2.25:15,2.25:25 --seeds 3,4,5 --out out/r4_base_s345b.json` | 9 | 206 |

`base` reads the cluster's shared cache (`--no-scratch`; sum|W| 121,460,584, `TYPE_NT_OVERRIDE` = {TmY14, Mi19, aMe8},
GLNO nt `unknown` sign 0, GLNO -> PEN W sum +0 on 84 entries / 16,371 raw synapses); `gaba` compiles from the raw
MaleCNS files into the job's own `out/cache_72164311/` (79 s; sum|W| 121,478,280, GLNO nt `gaba` sign -1, GLNO -> PEN
W sum -16,371, PEN -> GLNO +3,496). Protocol as section 4 (FlyBrain, full connectome, compass adaptation 0, 10 Hz
Poisson background on all 46 EPG, wedges 0-3 at +40 Hz for 2 s, 5 s free, Delta7 gain on Delta7 -> EPG only, cuda
graphs, torch sparse). Table: `python scripts/cx_glno.py --seed-table --files "out/cx_glno_gaba_gE*.json"
"out/cx_glno_base*.json" "out/sk_base_*.json" "out/sk_gaba_*.json" "out/r4_base_*.json" "out/r4_gaba_*.json" --table
cx_glno_r4_seeds` -> `out/cx_glno_r4_seeds.{md,csv}` + `_summary.csv` (138 unique (config, gE, gD, seed) rows from 15
JSONs; all 32 grid-point x condition rows are in the .md, the 14 six-seed rows below).

**Determinism (27 repeated keys, 0 differing metric fields).** The 15 files overlap on 27 (config, gE, gD, seed) keys
and every one of them agrees field for field (`wall_s` excluded) -- including three pairs of independent from-raw
compiles of the same hash in three different run directories (`r3-glno-gaba-5f4869`, `sk-glno-grid-9bf2b1`,
`r4-glno-d04de8` all produce `out/cache_72164311`): `gaba` 2.25/{8,15,25,40} seeds 0-2 (round 3 vs skeptic) and
`gaba` {2/15, 2.5/25} seeds 3-5 (skeptic vs this batch), plus `base` 1.75/15 and 2/15 seeds 0-2 across four files.
Two identical reruns, so these runs are deterministic given (config, gains, seed).

**The 3/35 rule is a seed effect, not a gain effect.** Persistence is `in_above >= 8 of 11 AND out_above <= 3 of 35`
at 5 s. Over the 97 runs in this grid whose bump is confined (`in_above >= 8`), the number of outside EPG above 22 Hz
is **exactly constant within each seed and independent of gE, gD and the GLNO sign**: seed 0 -> 1, seed 1 -> 3,
seed 2 -> 2, seed 3 -> 2, seed 4 -> 2, seed 5 -> 4 (19/19/21/12/12/14 runs). The outside mean is the same per seed too
(10.2 / 9.5 / 8.9 / 10.0 / 11.4 / 11.2 Hz) -- it is the background realisation, not the compass. So the criterion
scores "no" at seed 5 for *every* confined run at *every* point in *both* conditions, and seed 1 sits exactly on the
bound. Both columns are reported below: `persist` (the shipped rule) and `boundary` (confined but out_above = 4);
`persist + boundary` = the number of seeds with a confined bump.

### Six seeds in both conditions (rates at 5 s after release; rho = gD / gE^2)

| gE | gD | rho | config | persist (3/35) | boundary | confined | in > 22 Hz (/11, seeds 0-5) | out > 22 Hz (/35, seeds 0-5) | bump Hz at 5 s (seeds 0-5) | bump mean (range) | out Hz | vs (mean; min) | PEN | Delta7 | GLNO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.75 | 8 | 2.61 | base | 5/6 | 1 | 6/6 | 9/10/9/11/11/11 | 1/3/2/2/2/4 | 177.4/178.0/178.8/172.3/170.5/176.6 | 176 (170-179) | 10.2 | 0.76; 0.73 | 42.5 (41.6-43.3) | 93.6 (91.4-95.8) | 123.0 (118.2-128.3) |
| 1.75 | 8 | 2.61 | gaba | 3/6 | 1 | 4/6 | 9/9/9/4/2/9 | 1/3/2/3/2/4 | 154.4/148.2/152.5/17.2/9.9/154.5 | 106 (10-154) | 10.4 | 0.52; 0.05 | 22.5 (0.8-33.5) | 61.1 (13.3-87.5) | 65.9 (0.7-96.5) |
| 2.0 | 8 | 2.00 | base | 3/6 | 1 | 4/6 | 0/11/4/11/11/11 | 10/3/8/2/2/4 | 9.9/216.6/70.4/213.5/210.2/219.3 | 157 (10-219) | 27.5 | 0.77; 0.75 | 49.7 (43.3-52.9) | 104.6 (97.8-109.5) | 134.3 (111.8-145.0) |
| 2.0 | 8 | 2.00 | gaba | 5/6 | 1 | 6/6 | 9/10/10/10/11/11 | 1/3/2/2/2/4 | 193.5/192.2/196.7/194.7/191.1/196.1 | 194 (191-197) | 10.2 | 0.77; 0.74 | 42.0 (41.5-42.8) | 100.3 (98.8-102.2) | 120.1 (116.9-122.8) |
| 2.0 | 15 | 3.75 | base | 5/6 | 1 | 6/6 | 11/10/11/10/10/10 | 1/3/2/2/2/4 | 201.0/201.4/204.3/196.5/191.6/200.5 | 199 (192-204) | 10.2 | 0.76; 0.74 | 47.9 (46.3-48.7) | 100.1 (98.4-102.7) | 132.5 (126.7-137.2) |
| 2.0 | 15 | 3.75 | gaba | 5/6 | 1 | 6/6 | 10/10/10/11/11/11 | 1/3/2/2/2/4 | 179.9/181.2/184.4/179.0/175.8/182.3 | 180 (176-184) | 10.2 | 0.76; 0.73 | 40.4 (39.8-41.0) | 94.9 (93.4-96.6) | 114.9 (111.4-118.2) |
| 2.0 | 25 | 6.25 | base | 5/6 | 1 | 6/6 | 10/10/10/10/10/10 | 1/3/2/2/2/4 | 186.8/187.0/184.2/173.3/167.3/181.1 | 180 (167-187) | 10.2 | 0.75; 0.72 | 44.9 (42.7-46.5) | 92.4 (88.6-95.1) | 125.1 (112.4-134.1) |
| 2.0 | 25 | 6.25 | gaba | 3/6 | 1 | 4/6 | 9/3/9/4/9/9 | 1/9/2/8/2/4 | 161.2/42.0/163.0/46.5/122.2/163.1 | 116 (42-163) | 21.1 | 0.73; 0.67 | 36.6 (30.0-38.3) | 83.6 (72.9-89.9) | 99.5 (88.4-109.4) |
| 2.25 | 15 | 2.96 | base | 3/6 | 1 | 4/6 | 0/11/11/11/5/11 | 11/3/2/2/7/4 | 9.9/243.5/240.9/237.6/83.2/245.6 | 177 (10-246) | 29.2 | 0.79; 0.78 | 58.5 (57.0-61.0) | 111.3 (104.8-115.9) | 152.6 (140.5-159.0) |
| 2.25 | 15 | 2.96 | gaba | 5/6 | 1 | 6/6 | 10/11/11/10/11/10 | 1/3/2/2/2/4 | 213.7/216.2/223.3/213.6/209.6/216.8 | 216 (210-223) | 10.2 | 0.77; 0.75 | 48.0 (47.1-49.7) | 105.2 (103.8-107.7) | 132.0 (129.6-134.9) |
| 2.25 | 25 | 4.94 | base | 5/6 | 1 | 6/6 | 10/10/10/10/10/10 | 1/3/2/2/2/4 | 210.7/214.7/210.4/206.8/206.4/219.3 | 211 (206-219) | 10.2 | 0.77; 0.75 | 53.2 (51.3-55.7) | 103.0 (100.7-108.5) | 143.4 (137.4-149.2) |
| 2.25 | 25 | 4.94 | gaba | 5/6 | 1 | 6/6 | 10/10/10/10/10/10 | 1/3/2/2/2/4 | 195.9/199.0/192.5/189.6/179.6/198.3 | 192 (180-199) | 10.2 | 0.76; 0.73 | 44.6 (43.2-45.3) | 96.4 (92.4-100.5) | 122.7 (112.0-130.7) |
| 2.5 | 25 | 4.00 | base | 5/6 | 1 | 6/6 | 11/11/11/10/11/11 | 1/3/2/2/2/4 | 243.4/241.0/241.2/235.0/236.6/243.8 | 240 (235-244) | 10.2 | 0.78; 0.77 | 61.1 (59.9-62.5) | 112.0 (110.0-114.6) | 159.8 (156.0-164.4) |
| 2.5 | 25 | 4.00 | gaba | 5/6 | 1 | 6/6 | 10/10/10/10/10/10 | 1/3/2/2/2/4 | 227.1/225.4/222.5/224.0/219.7/228.8 | 225 (220-229) | 10.2 | 0.78; 0.76 | 52.3 (51.1-53.6) | 107.2 (103.7-111.3) | 140.2 (131.3-145.2) |

Mean +- sd over the confined runs of each cell (same data, `out/cx_glno_r4_seeds.csv`): 2/15 base bump 199.2 +- 4.5 Hz,
PEN 47.9 +- 0.8, Delta7 100.1 +- 1.6, GLNO 132.5 +- 3.7, vs 0.763 +- 0.015; 2/15 gaba 180.4 +- 3.0 / 40.4 +- 0.4 /
94.9 +- 1.2 / 114.9 +- 2.3 / 0.757 +- 0.018. 2.25/25 base 211.4 +- 4.9 / 53.2 +- 1.4 / 103.0 +- 2.9 / 143.4 +- 4.7 /
0.767 +- 0.012; gaba 192.5 +- 7.2 / 44.6 +- 1.0 / 96.4 +- 2.9 / 122.7 +- 6.1 / 0.757 +- 0.015. 2.5/25 base 240.2 +- 3.6
/ 61.1 +- 0.9 / 112.0 +- 1.6 / 159.8 +- 3.2 / 0.783 +- 0.010; gaba 224.6 +- 3.3 / 52.3 +- 0.9 / 107.2 +- 2.5 /
140.2 +- 5.3 / 0.778 +- 0.012.

The other nine grid points have three seeds in one or both conditions and are unchanged from section 4; the new silent
column at gE 2.25 (jobs 1 and 3) is, by the 3/35 rule, 2/3 at gD 8 (seed 0 an undisplaceable pre-pulse bump), 3/6 at
gD 15, 5/6 at gD 25 and 1/3 at gD 40 (seed-1 and seed-2 jumps to centre 4.0-4.1).

### Answers

**Which points are robust to the GLNO sign?** Three, all with rho between 3.75 and 4.94: **gE 2 / gD 15**, **gE 2.25 /
gD 25** and **gE 2.5 / gD 25**. Each is 6/6 confined in both conditions (5/6 by the 3/35 rule, the sixth being the
seed-5 boundary case that fires at every point in both conditions), 10/11 or 11/11 EPG above 22 Hz at 5 s in all 36
runs (12 per point), outside mean 8.9-11.4 Hz, vs 0.730-0.797, rest of the brain 0.028-0.042 Hz. Their capture columns
are also full: not one `not captured` run among the 36, `captured` 6/6 at 2.5/25 in both conditions, 5/6 at 2.25/25
in both, 5/6 base and 4/6 gaba at 2/15, the remaining seeds being `spontaneous at driven tile` (no prior bump
elsewhere to capture) -- so the "capturable" claim now rests on 4-6 capture tests per point x condition, not 1-2.
The two candidates named by the round-3 critic are confirmed and gE 2.25 / gD 25 joins them.

**Which candidate points are not robust, and how they fail.** Four of the seven six-seed points fail in one condition
only, and the failures are seed-matched (the same seeds hold in the other condition):
* gE 1.75 / gD 8 (rho 2.61): silent 6/6; **gaba dies at seeds 3 and 4** (in 17.2 / 9.9 Hz, 4/11 and 2/11 cells, PEN
  2.4 / 0.8 Hz, GLNO 9.6 / 0.7 Hz) -- the loop is below threshold once 33 mV of the PEN drive is inhibitory.
* gE 2 / gD 8 (rho 2.00): gaba 6/6; **silent fails seeds 0 and 2** with a stiff pre-pulse bump the 40 Hz pulse cannot
  displace (seed 0: in 9.9 / out 70.1 Hz, 10/35 outside cells, vs 0.78 at centre 13.2, classified `not captured`;
  seed 2 a partial jump to centre 4.0 at in 70.4 / out 52.6).
* gE 2.25 / gD 15 (rho 2.96): gaba 6/6; **silent fails seeds 0 (undisplaceable bump, in 9.9 / out 77.5, 11/35) and 4
  (jump to centre 4.0, in 83.2 / out 58.0)**.
* gE 2 / gD 25 (rho 6.25): silent 6/6; **gaba jumps at seeds 1 and 3** (in 42.0 / out 42.0 and 46.5 / 43.0 Hz, centre
  4.1, 8-9 of 35 outside cells above 22 Hz).
Read as a gain budget, this is the section-4 reading with the seeds filled in: below rho ~3 the ring is over-driven and
the silent condition locks into a spontaneous bump the pulse cannot move, while the inhibitory loop (33 mV per PEN per
GLNO volley, 16-24 % of the EPG volley) removes just enough drive to keep the pulse in control; above rho ~6 the
inhibitory condition loses the bump to a jump first. The two conditions' failure modes are opposite, and the overlap
where neither fails is rho 3.75-4.94.

**What the GLNO sign costs at the robust points.** The inhibitory loop lowers every rate by a seed-matched, disjoint
margin: bump -9.4 % at 2/15 (199.2 -> 180.4 Hz; ranges 191.6-204.3 vs 175.8-184.4), -8.9 % at 2.25/25 (211.4 -> 192.5;
206.4-219.3 vs 179.6-199.0), -6.5 % at 2.5/25 (240.2 -> 224.6; 235.0-243.8 vs 219.7-228.8). PEN -14 to -16 % (47.9 ->
40.4, 53.2 -> 44.6, 61.1 -> 52.3), Delta7 -4 to -6 % (100.1 -> 94.9, 103.0 -> 96.4, 112.0 -> 107.2), GLNO itself -12 to
-14 % (132.5 -> 114.9, 143.4 -> 122.7, 159.8 -> 140.2, still above Delta7 at every point). Vector strength is unchanged
within scatter (0.763 -> 0.757, 0.767 -> 0.757, 0.783 -> 0.778, sd 0.010-0.018). The 150-250 Hz rate problem is
untouched by either sign: the lowest six-seed robust bump is 180 Hz (gaba at gE 2 / gD 15), still an order of magnitude
above the animal's E-PG rates.

**Caveats.** (i) The 3/35 half of the persist rule is a background-realisation effect at this background (10 Hz on 46
EPG): it removes exactly one seed in six from every point and condition, so a "5/6" in this table is a clean sweep and
a "3/6" is two real failures. Reporting `confined` (in_above >= 8) alongside it is the honest count; the criterion
itself is not changed here, and no compass default moves. (ii) The gD grid is still coarse (8 / 15 / 25 / 40) -- the
robust band rho 3.75-4.94 is bounded by untested cells at rho ~3.2 and ~5.5. (iii) Six seeds resolve a 1-in-6 failure
rate at best; a point that is 6/6 here could still fail at a rate below ~15 %. (iv) GLNO stays out of
`TYPE_NT_OVERRIDE` (section 3: two low-confidence EM predictions, no expression profile in any of the six sources);
this section says only which gains survive either answer. (v) The silent-ring six-seed result supersedes the seed-0
window quoted in `cx_wedge.md` section 6, which is corrected there.

### Corrections (round-4 verification, `verify:exp:compass`)

* At the three robust points the raw 5 s vector strength is 0.730-0.797 (0.72 belongs to the silent-ring no-failure
  set in cx_wedge.md, whose minimum is at gE 2 / gD 25 seed 4, a point not robust to the GLNO sign). Percentage
  ranges: PEN -14.4 to -16.2 %, Delta7 -4.3 to -6.4 %, GLNO -12.3 to -14.4 %; the lowest single robust-point bump is
  175.8 Hz (gaba 2/15 seed 4). Six section-5 cells differ by one unit in the last digit from raw-value aggregation
  (round-then-mean). gE 2 / gD 40 seed 1 is an undisplaceable pre-pulse bump (class a), not a jump.
* Every grid run used receptor_model=None; the shipped sign/abs default changes 0 of 27,553 ring-core entries and 27
  sign/abs runs are byte-identical to off on the ring (`out/sk4_signabs_{robust,fail}.json`,
  `out/sk4_glno_receptor_check.txt`), so the grid stands under the current default -- a future table change must
  re-check this. Working-tree diff 259 insertions; the batch log for r4-glno-d04de8 was not shipped.
