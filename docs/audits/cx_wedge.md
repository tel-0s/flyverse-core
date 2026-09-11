# Central-complex wiring audit: is the ring attractor in the connectome under this model's rules?

Script: `scripts/cx_wedge.py` (structure: `python scripts/cx_wedge.py`; rate model: `--rate-grid`; LIF: `--sim`).
Data: `docs/audits/cx_wedge.json` (all structural numbers), `cx_wedge_matrices.npz` (cell-level and wedge-level
matrices), `cx_wedge_*_{cells,16,8}.png` (heatmaps), `cx_wedge_rate_*.json` (rate-model grids),
`cx_wedge_sim.json` + `cx_wedge_sim_profiles.png` (LIF runs, 27 rows). Connectome cache as of 2026-09-11,
`LIFParams()` defaults (w_syn 0.275 mV, conn_cap 60, same_type_gain 0.1, fan-in ref 5000 / alpha 1).

## Short answer

1. The tuned structure the attractor needs IS in the connectome and survives the model's synaptic rules:
   PEN-relayed excitation is wedge-local (84 % of its mass within +-2 wedges = +-45 deg) and Delta7-relayed
   inhibition is cosine-shaped around the ring (own wedge / opposite wedge = 0.10; own / mean of the other 15 = 0.19).
2. What shuts the loop at gain x1 is NOT Delta7 (the session-8 hypothesis). It is an untuned global feedback
   EPG -> ExR6 / ExR4 / ER6 / ER4m -> EPG, PEN (glutamate / GABA, driven only by EPG in this model): its two-step
   weight onto EPG is flat around the ring at -2,760 to -3,210 mV^2 per presynaptic wedge, i.e. larger than the
   on-wedge PEN excitation (+2,465) and 7x the peak Delta7 inhibition (-433). In the LIF, with Delta7 cut entirely,
   a PEN next to a driven 4-wedge bump still receives net -7.7 mV (+9.7 from EPG, -17.8 from ExR6/ExR4/ER6).
3. With the Delta7 gain applied to Delta7 -> EPG only (Delta7 -> PEN left at x1) the LIF holds a confined,
   persistent bump: gE (EPG <-> PEN, PEG, both links) 1.75-2.0 with gD 15-40, e.g. gE 2 / gD 15: driven tile
   201 Hz (11 / 11 cells > 22 Hz), other 35 EPG 10 Hz (1 / 35), for the whole 5 s after the pulse, vector strength
   0.76, rest of the brain 0.03 Hz; the same without any background (216 Hz vs 0 Hz, 0 / 35, vector strength 0.93),
   at another wedge and seed, and from a one-tile drive (the bump settles at its own width, 4-5 wedges). The ring
   also breaks symmetry spontaneously under the 10 Hz background at these gains (a bump forms before the pulse
   and the pulse relocates it). Applying gD to Delta7 -> PEN as well (the session-8 grid's convention) kills it at
   every gain tried (PEN 0.2-5 Hz). Alternative route with the recurrence at x1: damp ER/ExR -> EPG/PEN/PEG to
   x0.3, then gE 1 / gD 4-15 holds the bump (200 Hz / 10 Hz, 11 / 11 vs 1-2 / 35).
4. Caveat: the bump fires at 150-250 Hz (the loop saturates on the refractory period; PEN 40-65 Hz, Delta7
   90-100 Hz), an order of magnitude above the animal's EPG. Confinement and persistence are real; the rate is not.

## 1. Cells and wedge identity

| type | cells | glomeruli (instance) | notes |
|---|---|---|---|
| EPG | 46 | L1-L8, R1-R8; 2-4 per glomerulus (L1 2, R8 4, L2 2, R7 3, L3 3, R6 3, L4 3, R5 3, L5 3, R4 3, L6 3, R3 3, L7 3, R2 2, L8 4, R1 2) | ACh, sign +1 |
| EPGt | 4 | L9, R9 | excluded from the ring analysis (Delta7 -> EPGt is -10.2 mV/pair; with them the session-8 "-2.9 mV over 506 pairs" is reproduced) |
| PEN_a / PEN_b | 20 / 22 | L2-L9, R2-R9 | ACh |
| PEG | 18 | L1-L9, R1-R9, one each | ACh |
| Delta7 | 42 | instance = OUTPUT glomeruli: L1L9R8 (5), L2R7 (4), L3R6 (4), L3R7 (2), L4R5 (5), L4R6 (1), L5R4 (5), L6R3 (3), L6R4 (2), L7R2 (4), L7R3 (2), L8R1R9 (5) | glutamate, sign -1 |
| ER* / ExR* | 308 | no wedge (ring neurons) | GABA / glutamate |

Fan-in scale (input_norm) is 1.00 for every compass cell except one EPG at 0.98 (totals: EPG 2.8-5.1k, PEN 0.9-1.6k,
PEG 0.8-1.1k, Delta7 0.5-0.7k). No path gain touches them. So A[post, pre] = 0.275 x min(count, 60) x sign x
(0.1 if same type).

Ring order of the 16 wedges, read off the direct EPG -> EPG matrix (L_i contacts R_(9-i) and R_(8-i) only):
**L1 R8 L2 R7 L3 R6 L4 R5 L5 R4 L6 R3 L7 R2 L8 R1** (L9 wraps onto L1, R9 onto R1). Delta7 output pairs are always
two adjacent wedges of this ring (a "tile"), e.g. L1L9R8 = wedges 0-1, L3R7 = wedges 3-4. Tiles below are
L1/R8, L2/R7, L3/R6, L4/R5, L5/R4, L6/R3, L7/R2, L8/R1 (45 deg each).

## 2. One-step effective weights (mV per presynaptic spike per pair; total = if the whole presynaptic type fired once)

| projection | pairs | mV / pair | total per post cell |
|---|---|---|---|
| EPG -> PEN | 664 | +5.05 | +79.9 |
| PEN -> EPG | 735 | +7.98 | +127.5 |
| EPG -> PEG | 379 | +3.36 | +70.8 |
| PEG -> EPG | 290 | +0.94 | +5.9 |
| EPG -> Delta7 | 1670 | +3.28 | +130.3 |
| Delta7 -> EPG | 479 | -2.46 | -25.7 |
| Delta7 -> PEN | 404 | -4.70 | -45.2 |
| Delta7 -> PEG | 219 | -4.91 | -59.7 |
| Delta7 -> Delta7 | 1717 | -0.43 | -17.4 |
| EPG -> EPG (same type x0.1) | 842 | +0.37 | +6.7 |
| PEN -> PEN (x0.1) | 1069 | +1.91 | +48.5 |
| EPG -> ER/ExR (308 cells) | 6299 | +1.40 | +28.7 |
| ER/ExR -> EPG | 12338 | -2.86 | -766.9 |
| ER/ExR -> PEN | 2008 | -2.01 | -96.3 |

Delta7 profile against ring distance from the Delta7's own output tile (mean mV per cell pair, distance in wedges
0..8): EPG -> Delta7 = 0.49, 0.48, 0.34, 0.69, 1.66, 3.35, 4.65, 5.68, 5.84 (a Delta7 is driven by the opposite
side of the ring; own / opposite = 0.084); Delta7 -> EPG = -4.01, -3.95, -0.68, -0.13, -0.01, ... (its output is
confined to its own tile). That is the textbook Delta7: read the far side, inhibit the near side.

The ER/ExR feedback, by type (EPG -> type -> EPG two-step total per EPG cell, mV^2; and onto PEN):

| type | cells | NT | EPG -> type / pair | type -> EPG / pair | type -> PEN / pair | two-step onto EPG | onto PEN |
|---|---|---|---|---|---|---|---|
| ExR6 | 2 | glutamate | +11.44 | -15.01 | -8.66 | -15,792 | -8,892 |
| ER4m | 11 | GABA | +2.45 | -11.39 | -0.46 | -14,097 | -222 |
| ExR4 | 2 | glutamate | +8.65 | -4.96 | -15.22 | -3,946 | -12,110 |
| ER6 | 4 | GABA | +7.63 | -3.55 | -4.62 | -3,608 | -4,744 |
| ExR5 | 4 | glutamate | +1.97 | -9.02 | -0.48 | -3,276 | -81 |
| ER2_c | 20 | GABA | +0.89 | -4.36 | -0.37 | -3,246 | -40 |
| ER4d | 26 | GABA | +0.72 | -2.84 | -0.59 | -2,032 | -172 |
| others (ER2_a, ER1_b, ER3w_*, ER2_d ...) | | GABA | | | | -554 to -1,526 each | small |

In the LIF at gain x1 during a 4-wedge drive (Delta7 cut): ExR6 fires 89 Hz, ER6 56 Hz, ExR4 41 Hz, ER4m 16 Hz,
all driven by EPG (+51, +31, +38, +10 mV mean input); every EPG receives -20 (in the bump) to -26 mV (opposite)
from them, PEN near the bump -17.8 mV.

## 3. EPG x EPG two-step matrices (mV^2 per spike; "input to a typical cell of wedge i if every cell of wedge k fires once, relayed through X")

Profiles against ring distance (mean over the 16-wedge matrix's diagonal bands; distance 0..8 wedges of 22.5 deg):

| path | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| PEN (EPG -> PEN -> EPG) | +2465 | +2044 | +1211 | +567 | +222 | +56 | +11 | +1 | +0 |
| PEG | +131 | +47 | +16 | +8 | +37 | +6 | +8 | +6 | +41 |
| Delta7 (EPG -> Delta7 -> EPG) | -44 | -38 | -49 | -99 | -190 | -299 | -374 | -425 | -433 |
| ER/ExR (EPG -> ER/ExR -> EPG) | -3212 | -2822 | -3177 | -2786 | -3166 | -2775 | -3152 | -2760 | -3165 |
| direct EPG -> EPG (mV, one step) | +2.4 | +1.8 | +0.2 | 0 | 0 | 0 | 0 | 0 | 0 |
| EPG -> Delta7 -> PEN -> EPG (mV^3) | -19.9k | -23.2k | -30.4k | -38.1k | -44.1k | -48.8k | -53.3k | -57.6k | -59.2k |

8-tile matrices (rows post, cols pre; from `cx_wedge.json["M8"]`):

PEN: diagonal 2,794-4,937, first neighbour 2,000-3,300, second neighbour 400-970, third 40-150, opposite 1-60.
Delta7: diagonal -55 to -131, neighbour -66 to -368, second -430 to -840, opposite -780 to -985.
ER/ExR: every entry -4,500 to -6,950 (flat).

* PEN locality: 61 % of the matrix mass within +-1 wedge, 84 % within +-2, 99 % within +-4, 1.3 % on the opposite
  half; diagonal / mean = 3.7. Wedge-local with a half-width of ~2 wedges (45 deg); not nearest-neighbour only.
  PEN_a and PEN_b give the same profile; the L / R shift of the animal's PEN (input at the glomerulus, output one
  tile over) is present in the EPG -> PEN input side (a left PEN's input is centred one tile down, a right PEN's one
  tile up) but the two-step product is symmetric.
* PEG: 20x weaker than PEN and less local (19 % on the opposite half). Negligible.
* Delta7 own-wedge inhibition: -43.5 (own), -37.8 (adjacent wedge), -225 (mean of the other 15), -376 (far half),
  -433 (opposite). **Ratios own / other 0.19, own / far-half 0.12, own / opposite 0.10.**

## 4. The linear ring-attractor condition

Bump = k contiguous bins at uniform rate; E = PEN + PEG, I = Delta7 two-step input from the bump, per post cell.
With gE on both EPG <-> PEN links and gD on Delta7 -> EPG, u = gE^2 E + gD I; a wedge inside must be net-excited
(gD < E_i / |I_i|), every wedge outside net-inhibited (gD > E_j / |I_j|): window in rho = gD / gE^2 =
(max_out E/|I|, min_in E/|I|), worst case over the bump's starting position (the ring has 2-4 cells per wedge).

| level | k | E_in | E_out | I_in | I_out | E_in/E_out | \|I_in\|/\|I_out\| | rho_low (worst / mean) | rho_high (worst / mean) | positions with a window |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 wedges | 2 | 4687 | 919 | -81 | -477 | 5.10 | 0.17 | 68.2 / 47.9 | 29.2 / 54.4 | 10 / 16 |
| 16 | 3 | 6202 | 1134 | -126 | -761 | 5.47 | 0.17 | 49.0 / 25.7 | 30.3 / 40.3 | 15 / 16 |
| 16 | 4 | 7246 | 1290 | -198 | -1075 | 5.62 | 0.18 | 21.7 / 12.9 | 20.0 / 25.6 | 16 / 16 |
| 16 | 5 | 7977 | 1427 | -317 | -1412 | 5.59 | 0.22 | 9.8 / 7.2 | 10.1 / 14.3 | 16 / 16 |
| 16 | 6 | 8485 | 1579 | -496 | -1756 | 5.37 | 0.28 | 6.6 / 4.6 | 6.2 / 8.7 | 16 / 16 |
| 8 tiles | 1 | 4335 | 961 | -84 | -466 | 4.51 | 0.18 | 50.3 / 30.2 | 33.1 / 54.4 | 8 / 8 |
| 8 | 2 | 6982 | 1361 | -193 | -1052 | 5.13 | 0.18 | 8.8 / 7.5 | 28.2 / 33.9 | 8 / 8 |
| 8 | 3 | 8282 | 1670 | -484 | -1718 | 4.96 | 0.28 | 4.1 / 3.2 | 8.6 / 12.2 | 8 / 8 |
| 8 | 4 | 8966 | 2099 | -990 | -2358 | 4.27 | 0.42 | 2.1 / 1.8 | 4.3 / 5.6 | 8 / 8 |

The locality ratio (E_in/E_out) / (|I_in|/|I_out|) is 25-30: the excitation is far more local than the inhibition,
which is the attractor's requirement. The binding constraint is always the tile next to the bump (it gets ~60 % of
the bump's PEN excitation and as little Delta7 inhibition as the bump itself), so confinement needs
**rho = gD / gE^2 of roughly 9-28 for a 2-tile (90 deg) bump, 4-9 for 3 tiles, 2-4 for half the ring**. The
session-8 grid (gE 1.5-3 x gD 0.5-8) never exceeded rho = 3.6, i.e. it only ever allowed half-ring bumps, which is
the "whole-ring" state it saw.

With the flat ER/ExR term added at gain 1 (u = gE^2 E + gD I + G), the feasible gD per gE (8-tile, 2-tile bump,
worst / mean over positions): gE <= 1.25 -> none (the bump cannot be net-excited: gD_high < 0); gE 1.5 -> gD < 7.4
(mean < 17); gE 2 -> gD 8.4-59 (mean 4.9-76); gE 3 -> gD 53-200. For 16:4 (4 wedges): gE 2 -> gD 32-40
(mean 16-54); 16:5: gE 2 -> gD 10-16 (mean 5-23). So the global feedback moves the loop's closing point to
gE ~1.5-2 and then leaves a gD window of about 10-50 -- which is where the LIF finds the bump.

Absolute scale at a nominal intermediate gain of 6 Hz/mV (tau_syn 5 ms): a 2-tile bump delivers +1.05 mV per Hz
of bump rate to its own EPG through PEN and -0.03 through Delta7 at x1; outside +0.20 / -0.16. PEN drive per EPG
wedge by distance: direct +11.6, +4.3, +14.1, +13.5, +4.0, +0.6, 0 ... mV/spike; via Delta7 (at 6 Hz/mV) -2.0,
-1.8, -2.3, -5.0, -10.3, -15.9, -20.5, -23.1, -23.8 mV; via ER/ExR -51 mV at every distance. A gain on Delta7 ->
PEN therefore hits the bump's own PEN (-2 x gD) as hard as the loop's excitation at gD ~ 5, which is the
"Delta7 clamps PEN" seen in session 8 and why gD has to be applied to Delta7 -> EPG alone.

## 5. Threshold-linear rate model (`--rate-grid`)

Mean-field fixed point r = f(tau_syn A r + forced) on the cell-level effective matrix (EPG, PEN, PEG, Delta7
+ 308 ER/ExR), LIF f-I with 2 mV input noise, same protocol as the LIF (10 Hz background on EPG, 4 wedges at +40 Hz,
then release). Without the ER/ExR cells it predicts bumps at gE 0.8-1.2 (e.g. gE 1 / gD 4-40: in 226-329 Hz, out
13-88); with them (gain 1, Delta7 -> PEN x1) only at gE >= 2.5 (gE 2.5, gD 1-60: in 187-333, out 10; gE 3, gD 4-60);
with gD also on Delta7 -> PEN only at (2.5, 1), (3, 4), (3, 8); with ER/ExR damped to 0.3 at gE 1 / gD 1-15 and
gE 1.25-1.5 / gD 4-40. The model is more pessimistic than the LIF by ~0.5 in gE (the LIF closes the loop at gE
1.75) but reproduces every qualitative feature: no bump at gE <= 1.5 with the feedback intact, a window in gD,
the Delta7 -> PEN clamp, and the damped-feedback route.

## 6. LIF cross-check (`--sim`, full connectome, FlyBrain, cuda graphs, compass adaptation off, seed 0 unless noted)

Protocol: 1 s on a 10 Hz Poisson background (all 46 EPG), wedges 0-3 (L1 R8 L2 R7 = tiles L1/R8 + L2/R7, 11 cells)
at +40 Hz for 2 s, 5 s free. "in / out" = mean EPG Hz inside / outside the driven wedges (cells > 22 Hz in
brackets); vs = circular vector strength of the EPG rate profile (0 uniform, 1 one wedge). Delta7 gain on
Delta7 -> EPG only unless "D7->PEN x gD". Full rows in `cx_wedge_sim.json`, profiles in `cx_wedge_sim_profiles.png`.

| gE | gD | variant | before pulse | 0.5 s after | 2 s | 5 s | PEN / D7 / ER-ExR at 5 s | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | 1, 8, 15, 25, 40 | | 14 / 9.5 (0, 2) | 9.4 / 7.9 (0, 0) | 8.2 / 8.9 | 9.9 / 10.2 (0, 1) vs 0.09 | 0.0 / 9.7 / 1 | dies in 0.5 s; PEN 0.9 Hz even during the pulse (session 8's result) |
| 0.8 | 25 | | same | same | same | same | | dies |
| 1.2 | 8 | | same | 9.4 / 7.9 | | 9.9 / 10.2 | 0.4 / 9.7 | dies |
| 1.5 | 8 | | 14 / 9.5 | 124 / 7.9 (8, 0) | 9.4 / 10.0 (0, 1) | 9.9 / 11.2 (0, 4) | 1.5 / 11.8 / 1.1 | metastable, ~1.5 s |
| 1.5 | 15 | | 14 / 9.5 | 115 / 7.9 (8, 0) | 8.2 / 8.9 | 9.9 / 11.3 | 1.5 / 11.6 | metastable, ~1 s |
| 1.75 | 15 | | 14 / 9.5 | 167 / 7.9 (9, 0) | 169 / 8.9 (10, 1) | 164 / 10.2 (10, 1) vs 0.74 | 41 / 89 / 8 | **stable, confined** (profile 116 233 225 63 then 5-22) |
| 2 | 8 | | bump elsewhere: 14 / 66.7 (0, 12) | 9.4 / 66.5 (0, 10) | 8.2 / 67.5 | 9.9 / 70.1 (0, 10) vs 0.78 | 43 / 98 / 10 | spontaneous 4-5-wedge bump that the 40 Hz pulse cannot displace |
| 2 | 15 | | bump elsewhere: 14 / 64.1 (0, 12) vs 0.75 | 204 / 7.9 (11, 0) | 201 / 8.9 (10, 1) | 201 / 10.2 (11, 1) vs 0.76 | 49 / 101 / 10 | **stable, confined, captured by the pulse** (profile 106 251 251 164 then 5-22) |
| 2 | 25 | | 60 / 48 (4, 8) vs 0.76 | 188 / 7.9 (10, 0) | 188 / 8.9 | 187 / 10.2 (10, 1) vs 0.75 | 47 / 95 / 9 | **stable, confined** |
| 2 | 40 | | 147 / 9.5 (10, 2) vs 0.73 (spontaneous bump at the driven tile) | 157 / 7.9 (10, 0) | 160 / 8.9 | 156 / 10.2 (10, 1) vs 0.72 | 42 / 83 / 8 | **stable, confined** |
| 2 | 60 | | 14 / 9.5 | 131 / 7.9 (10, 0) | 8.5 / 36.6 (0, 10) vs 0.67 | 11.7 / 21.3 (1, 11) vs 0.35 | 17 / 31 / 3 | drifts off the driven tile after 1 s, then decays |
| 2 | 100 | | 14 / 9.5 | 49 / 9.2 (8, 1) | 9.4 / 10.8 | 9.9 / 10.9 (0, 2) | 3.6 / 11 / 1 | dies by 2 s |
| 2.5 | 15 | | bump elsewhere: 14 / 96.6 (0, 15) vs 0.77 | 9.4 / 95.6 (0, 14) | 8.2 / 93.2 | 9.9 / 98.6 (0, 13) vs 0.79 | 83 / 123 / 14 | 5-6-wedge spontaneous bump, too stiff for the pulse |
| 2 | 15 | D7->PEN x gD | 14 / 9.5 | 9.6 / 8.8 (0, 0) | 8.2 / 9.3 | 9.9 / 10.2 | 0.4 / 9.7 | dies; PEN 5 Hz during the pulse (the clamp) |
| 2 | 25 | D7->PEN x gD | 14 / 9.5 | 9.4 / 7.9 | 8.2 / 8.9 | 9.9 / 10.2 | 0.2 / 9.7 | dies |
| 2 | 15 | background 0 Hz | 0 / 0 | 223 / 0.0 (11, 0) | 219 / 0.0 | 216 / 0.0 (11, 0) vs 0.93 | 51 / 97 / 9 | **stable without tonic drive** (profile 165 249 249 183, all else 0) |
| 2 | 15 | wedges 6-9, seed 1 | bump elsewhere: 96 / 31 (6, 8) | 183 / 22 (10, 6) | 189 / 19 | 188 / 21 (10, 5) vs 0.79 | 65 / 101 / 11 | **stable**, 4 wedges > half max (206 278 252 151), neighbours 3-17 Hz |
| 2 | 15 | 1 tile (wedges 10-11), seed 2 | 168 / 26 (5, 4) | 224 / 29 (6, 9) | 221 / 29 | 227 / 28 (6, 7) vs 0.81 | 54 / 94 / 10 | **stable**; settles at its own width (35 210 244 227 35 = 3 wedges > half, 5 > 22 Hz) |
| 1 | 4 | ER/ExR x0.3 | 204 / 9.6 (11, 2) (spontaneous bump at the driven tile) | 200 / 8.1 (11, 1) | 200 / 9.0 | 200 / 10.4 (11, 2) vs 0.75 | 49 / 103 / 10 | **stable, confined at recurrence x1** |
| 1 | 15 | ER/ExR x0.3 | 175 / 9.5 (11, 2) | 175 / 7.9 (11, 0) | 177 / 8.9 | 170 / 10.2 (11, 1) vs 0.73 | 46 / 89 / 8 | **stable, confined** |
| 1.25 | 8 | ER/ExR x0.3 | bump elsewhere: 22 / 96 (4, 16) | two bumps 136 / 62 (6, 9) | 135 / 61 | 9.9 / 101 (0, 14) | 83 / 128 / 15 | pulse cannot take over; merges back after 2 s |

Bump width at 5 s (wedges above half the peak / above 22 Hz): 3-4 / 4-5 in every stable run, i.e. 70-110 deg,
matching the linear estimate that the narrowest confinable bump is 2 tiles (90 deg). Rest-of-brain mean rate
0.00-0.05 Hz in every run: nothing spreads outside the compass. Wall time 15-27 s per 8 s run on the shared 4090.

## 7. What this says about the model

* The connectome's EPG / PEN / Delta7 wiring is a ring attractor's wiring under the cap, the same-type damping and
  the L1 fan-in rule: local excitation through PEN, cosine-shaped inhibition through Delta7 with the Delta7's own
  tile spared 10x. Session 8's conclusion "under uniform 0.275 mV synapses ... it does not produce winner-take-all"
  was about the gains it tried (rho <= 3.6, Delta7 gain also on Delta7 -> PEN), not the structure.
* The missing piece was an untuned inhibitory loop through the ring neurons ER4m / ER6 and the extrinsic ring
  neurons ExR4 / ExR6 (2 + 2 cells with 8-15 mV unitary weights onto PEN / EPG under the cap; ExR neurons in the
  animal are broadly acting and several are modulatory, their predicted "glutamate" makes them fast inhibitors
  here). It sets the loop's closing gain at gE ~1.75 and is itself the global inhibition a ring attractor uses; with
  it damped x0.3 the compass works at the default recurrence.
* Two workable settings for a compass in the model, both as `LIFParams(type_path_gain=...)` plus
  `adapt_by_type={"^(EPG|PEN|PEG|Delta7)": 0}`: (a) EPG <-> PEN / PEG x2 and Delta7 -> EPG x15-25 (Delta7 -> PEN
  x1); (b) ER/ExR -> EPG / PEN / PEG x0.3 and Delta7 -> EPG x4-15. Neither fixes the bump's 200 Hz rate, and neither
  makes the compass move with the fly: PEN L / R differences (angular-velocity input) and the ring neurons' visual
  pathway (TuBu / AOTU silent, session 8) are separate questions.
