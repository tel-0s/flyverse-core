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

## 3. Simulation runs (cluster batch `cx-glno-f2e987`, run dir `/mnt/beegfs/neurome/runs/cx-glno-f2e987`)

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
