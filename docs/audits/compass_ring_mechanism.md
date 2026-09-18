# Compass, thread 5A: a type-level ring mechanism, or none

Scripts: `scripts/cx_ring_structure.py` (new; CPU structure pass, `--plan-batch`, `--analyse`), `scripts/cx_wedge.py` (new flags
only: `--ledger`, `--lif KEY=VALUE`, `--arm`, `--block`, `--device`, `--pulse-s`, `--receptor-model shipped`; every default
unchanged -- the default `--sim` path re-run on the CPU after the edit, `out/cx5/smoke/smoke_default_path.json`),
`tests/test_cx_ring_structure.py` (the wedge-matrix / eigenmode / fixed-point code on a synthetic ring; 5 passed).
Data: `out/cx5/structure/{structure.json, structure.md, matrices.npz, evidence_glno.json}` (the structure pass),
`out/cx5/predeclared.json` (stamped **2026-09-15T05:39:53Z**, amended with the resubmission note at 05:42:48Z before the second
submission), `out/cx5/tree_state.json`, `out/cx5/batch.sh`, `out/cx5/<arm>_s<seed>.{json,txt,npz}` (48 runs),
`out/cx5/scheduler_receipt.json` (the scheduler's own job records), `out/cx5_cluster.log` (the client console),
`out/cx5/analysis/{analysis.md, runs.csv, compare.csv, decision.csv, analysis.json, scatter.png}`.
Tree: HEAD 0b3668f (= origin/main); nothing committed by this thread. The working-tree files shipped with the batch are listed
with their sha256 in `tree_state.json`: this thread's `scripts/cx_wedge.py`; another thread's concurrent `flyverse/body.py`,
`flyverse/senses.py`, `scripts/probe_vnc_drive.py`, `tests/*`, `docs/audits/level_controls.md`, and the untracked
`scripts/glno_relabel.py` / `docs/audits/glno_relabel.md` -- none on the simulated path (`cx_wedge.py --sim` builds a FlyBrain
without a world and imports none of them).

## 0. Answer

**No type-level ring mechanism reachable through the data-implied changes makes the shipped compass hold a bump, and none
makes a working compass.** At the shipped gains (gE 1 / gD 1) the shipped path, GLNO = glutamate, the receptor `sign+gain`
tier and their combination all score `bump_survival_s` 0.00 in 4/4 seeds (a structural zero-SD null against the shipped
arm, which is also 0.00 in 4/4). The structure pass says why. **The WEIGHT matrix has the k = 1 structure a ring attractor
needs** (lambda_1 = +10,019 mV^2 shipped; a uniform-gain linearisation would be unstable in that mode at gamma >
1.69-2.00 Hz/mV, against an LIF slope of 2.3-3.2 Hz/mV at the 140-200 Hz the surviving bumps run at) -- but that is a
necessary condition, not the attractor: the criterion is on diag(gamma_i) tau A with gamma_i = f'(u_i) at a realised state,
and at the rate model's own fixed point every compass cell has gamma = 0 except Delta7 (3.6 at background, 6.2 during the
pulse), so the true Jacobian's leading eigenvalue is +0.045 (background) / +0.079 (pulse) -- stable in every mode. The wiring
could be a ring attractor's; under the shipped rules it never is one, because the gain is never there (section 1.2; the
Jacobian is the independent skeptic pass's R2). **The relays stay far below the rate a bump needs**: the measured PEN
population mean is 0.29-0.66 Hz during the pulse in S (0.02-0.06 after), and PEN's mean input in the driven glomeruli is
-11 mV at background and -16 mV *during* the pulse, because the EPG drives 2 ExR6 + 4 ER6 (+ 11 ER4m onto EPG) harder than it
drives PEN (EPG -> ExR6 +11.4 mV per pair vs EPG -> PEN +5.05; ExR6 at 110-183 Hz delivers -9 to -15 mV to every PEN -- a
rate-model number, not measured; section 1.2, and the skeptic pass's correction 5 below). The ER/ExR loop carries no k = 1 content (-206, 2.5 % of
PEN's +8,327); it is untuned in the bump mode, not a flattened cosine, so it is a threshold problem -- and it is untouched by
the GLNO sign (GLNO fires 0.00 Hz before the pulse, 1.4-3.0 Hz during it and 0.06-0.39 Hz after; at 3 Hz its -33 mV per PEN
per volley is -0.5 mV on a PEN 7 mV below threshold, so G vs S at the shipped gains is a structural null by construction
rather than a test of the relabel), by the receptor tiers (which scale the tuned ring and the untuned feedback by the same
x1.5), by the monoamine slow class (< 0.2 mV), by the fan-in rule (PEN scale 1.00) and, in the wrong direction, by the cap
(lifting it doubles the ExR inhibition on PEN: -41 mV). The one data-implied change that *does* move the balance is the
receptor tier: C raises PEN to 4.4-6.4 Hz during the pulse and GLNO to 15-22 Hz, a ~15x change that still does not close the
loop. No data-implied per-type change moves it far enough.

The one thing that does carry a bump at the shipped gains is not a type-level mechanism but the removal of a global hand
rule: with the same-type damping off (`same_type_gain` 1, arm F -- a labelled INSTRUMENT), the connectome's own EPG -> EPG
synapses (842 pairs, +3.65 mV per pair, wedge-local: +24 mV per spike within the wedge) turn the pulsed wedges into a
self-sustaining block -- 151-156 Hz for the whole 5 s in 4/4 seeds, width 3.0 wedges, confined 82-87 % of the post-pulse
frames, PEN 30 Hz, Delta7 38 Hz, GLNO 87-91 Hz, rest of brain 0.04 Hz. (That attribution to the EPG's own synapses is
*inferred from the structure pass, not measured* -- `same_type_gain` is one global scalar, and in F the PEN loop is engaged
too; section 3.3.) It is **a bump, not a working compass** (rate 3x above
the 5-60 Hz ledger bound, like the experiment gains' 200 Hz bump), and **it does not survive the correct GLNO sign**: with
GLNO = glutamate the same configuration (FG) loses the bump within 0.15 s in 4/4 seeds (GLNO's -33 mV per PEN per volley,
now signed and firing at 90 Hz, opens the loop). Adding the receptor tier back (CFG: sign+gain + damping off + GLNO =
glutamate) restores it (156-158 Hz, 4/4, width 3.0), because sign+gain raises EPG -> PEN to +114 mV per volley; CF (without
the relabel) runs away instead (PEN 185-276 Hz, GLNO 227-289 Hz, the bump jumps to wedges 7-8 in 3/4 seeds). The labelled
references reproduce cx_glno's **`base`** rows (receptor model off, and its identical `sign-class` rows -- not its `sign-abs`
rows) to the printed digit on the same GPU class: R (gE 2 / gD 15) 201-203 Hz in 4/4, RG 181-184
in 4/4, R175 holds in 4/4 (two seeds after a jump), RG175 dies in 4/4. **The rate-model structure pass missed the F-family
bump because of two analysis choices -- an owned defect of this thread's own tool, not a fluctuation regime**: the
EPG-only reduction drops the
one-step EPG -> EPG term (its undamped k = 1 ring-Fourier coefficient is +59.76 mV, gamma_crit 3.35 Hz/mV, predicting
saturation at 145 Hz against the observed 151-156), and `rate_fixed_point` enters the forced background as a rate rather than
as a current, so the driven EPG sits at u -22 to -68 mV while "firing" at 10-50 Hz and gamma_EPG = 0 by construction (during
the pulse the driven EPGs run at 190 Hz in F against 57 Hz in S; section 3.3, and the independent skeptic pass's R3). The
linear mode analysis is right that the k = 1 structure is there and wrong about which change engages it.

Nothing is adopted. `same_type_gain` 1 is a global default whose x0.1 exists to stop runaway cliques elsewhere (FR1, lLN1_bc,
DLMn, DNg33; `brain.py`), so an adoption would need the full benchmark suite at >= 3 draws with no check changing status, on
the CPU path as well as the GPU, and a per-type mechanism would be a new code path (there is none: the damping is one scalar).
The GLNO relabel needs an entry in `TYPE_NT_OVERRIDE` with its three sources (MaleCNS T-bars 51 % glutamate at conf 0.48; the
hemibrain name; BANC v888 glutamate 4/4, read here from `cache/banc`) and the same suite test; at the shipped gains it changes
nothing that decides anything (**G = S exactly on the four primaries and on the t5.0 EPG in/out row in 4/4 seeds, but not on
every ring metric**: 12-17 of the recorded secondaries differ per seed, largest `GLNO_mean_post` s1 0.1115 -> 0.1470 (+32 %)
and s3 0.0881 -> 0.0618 (-30 %), `PEN_mean_during` s1 0.6614 -> 0.5003 (-24 %) and s2 0.3370 -> 0.2766 (-18 %),
`PEG_mean_post` s3 0.1273 -> 0.1710 (+34 %), `epg_in_mean_post` s2 10.7638 -> 10.7984 -- and the comparison is a **structural
null by construction** rather than a test of the relabel, because GLNO fires 0-3 Hz at these gains; the sign is tested only
where GLNO fires, F 87-91 Hz vs FG and R 130-137 Hz vs RG), and at the experiment gains it lowers the bump 10 % (RG
vs R) and kills the gE 1.75 point (RG175), exactly as `cx_glno.md` 3-5 found.

## 1. The effective ring circuit under the shipped rules (CPU; `out/cx5/structure/structure.md`)

Weights are `brain._shaped_weights(c, LIFParams())` x fan-in scale x w_syn 0.275 mV, exactly what `Brain` installs: receptor
stage `sign` / `abs` (0 of the 27,553 ring-core entries change sign, `cx_glno.md` 2 -- and stronger than that: over the
57,261 nonzero core x core entries, `max |A_receptor_off - A_shipped| = 0.000`, so **the receptor stage is a bit-exact no-op
on the ring core**, 0 sign changes *and* 0 weight changes), connection cap 60, path gains (none on
these cells), same-type damping x0.1, fan-in reference 5,000 (EPG scale 0.98-1.00, one cell below 1; PEN 1.00 on totals
867-1,598; PEG / Delta7 / EPGt / GLNO 1.00). Cells: 46 EPG, 42 PEN (20 PEN_a + 22 PEN_b), 18 PEG, 42 Delta7, 4 EPGt, 308 ER /
ExR, 4 GLNO. Ring order and wedge identities as `cx_wedge.md` 1 (L1 R8 L2 R7 ... L8 R1; Delta7 at its output tile). Shipped
cache compiled-W md5 ef23cc27..., GLNO = glutamate scratch cache 7a10d93b... (`out/cache_c51b23e2`, nnz identical, sum|W|
121,478,280 vs 121,460,584).

### 1.1 Offset matrices (post - pre in wedges of 22.5 deg; mean mV per spike per post cell from all pre cells at that offset)

| projection | -4 | -3 | -2 | -1 | 0 | +1 | +2 | +3 | +4 | centroid |
|---|---|---|---|---|---|---|---|---|---|---|
| EPG -> PEN (L PEN) | 0 | 0 | +0.3 | +0.2 | +10.6 | +8.2 | +27.1 | +24.8 | +7.5 | +2.16 |
| EPG -> PEN (R PEN) | +7.4 | +25.0 | +27.9 | +7.6 | +10.3 | +0.1 | +0.1 | 0 | 0 | -2.18 |
| PEN -> EPG (L PEN) | +10.4 | +17.5 | +19.2 | +10.7 | +2.9 | +0.6 | +0.1 | 0 | 0 | -2.45 |
| PEN -> EPG (R PEN) | 0 | 0 | 0 | +0.3 | +2.4 | +10.3 | +20.2 | +18.1 | +9.3 | +2.45 |
| EPG -> PEG | +4.1 | +0.1 | +2.0 | +4.6 | +43.3 | +6.0 | +2.0 | 0 | +3.9 | |
| PEG -> EPG | +0.5 | 0 | +0.2 | +0.8 | +2.5 | +0.7 | +0.1 | 0 | +0.5 | |
| Delta7 -> EPG | 0 | 0 | -1.6 | -1.7 | -17.8 | -2.1 | -2.0 | 0 | 0 | |
| Delta7 -> PEN | 0 | -0.1 | -2.6 | -2.5 | -34.5 | -2.1 | -3.3 | 0 | 0 | |
| EPG -> Delta7 (its output tile at 0) | +12.8 | +0.5 | +2.8 | +0.3 | +2.4 | +0.3 | +2.7 | +0.5 | +12.9 (+-6: +25.4 / +26.2; +-8: +14.6) | |
| EPG -> EPG (x0.1) | +0.1 | 0 | +0.2 | +1.7 | +2.6 | +1.7 | +0.2 | 0 | +0.1 | |

The +-1-tile shift lives in the labels: a PEN labelled at glomerulus g reads EPG 2-3 wedges to one side (L PEN +2.16, R PEN
-2.18) and writes 2.45 wedges to the other (L -2.45, R +2.45), so per side the loop returns to the EPG wedge it started from
(net -0.3 / +0.3) and the two-step EPG x EPG kernel is symmetric, peaked at distance 0 (`cx_wedge.md` 3: 2465 / 2044 / 1211 /
566 at 0-3 wedges, 84 % within +-2). Delta7 reads the far side of the ring (+-4..8) and writes its own tile (-17.8 mV at 0,
-2 at +-1..2). PEG is a same-wedge relay at 1/20 of PEN's weight. ER / ExR carry no wedge.

### 1.2 The linearised rate model and the ring modes

The threshold-linear model of `cx_wedge.md` 5 is r_i = f(u_i), u_i = tau_syn sum_j A_ij r_j + forced_i, with f the LIF f-I
(1000 / (t_ref + tau_m ln(u / (u - theta))), theta 7 mV, smoothed over 2 mV of input noise). Linearised at a uniform state with
one slope gamma = f'(u) (Hz/mV) for every cell: delta r = gamma tau A delta r, so a mode of A with eigenvalue mu (mV per spike)
is self-sustaining once gamma > 1 / (tau mu). With the relays taken adiabatic at the same gamma the EPG-only reduction is
delta r_E = gamma^2 tau^2 M delta r_E with M = sum over relays X of A_EX A_XE (mV^2 per spike; PEN, PEG, Delta7, ER/ExR), and
on the 16-wedge ring the loop gain of Fourier mode k is gamma^2 tau^2 lambda_k, lambda_k = Re(e_k^H M e_k) / 16; a bump (k =
1) can hold when lambda_1 > 0 at gamma_crit(1) = 1 / (tau sqrt(lambda_1)), and the uniform state is held down when
lambda_0 < 0. The LIF slope f'(u) = f^2 tau_m theta / (1000 u (u - theta)) falls monotonically across the firing range: 25.8
Hz/mV at u 7.1 (11 Hz), 9.1 at u 8 (23 Hz), 6.0 at u 12 (51 Hz), 4.6 at u 20 (93 Hz), 2.9 at u 40 (165 Hz), 2.4 at u 50 (192
Hz), and 0 below threshold. It is in the 5-8 band only for u 9-16 mV (31-70 Hz); at the 140-200 Hz the surviving bumps run at
it is 2.3-3.2 Hz/mV, still above gamma_crit(k1) 1.69-2.00 but by a factor of 1.4, not 3.
Tests: `tests/test_cx_ring_structure.py` (a circulant's Fourier quotients are its eigenvalues; a
local-excitation / global-inhibition profile puts k = 1 first with k = 0 negative; on a two-population synthetic ring the
full-circuit k = 1 eigenvalue obeys mu^2 = lambda_1 so both gamma_crit agree to 1e-6; a one-wedge shift is recovered by the
offset profile; the fixed-point iteration holds a synthetic bump and relaxes without local excitation).

Shipped (tau 5 ms; `structure.json` configs[0]):

| quantity | PEN | PEG | Delta7 | ER / ExR | direct EPG->EPG -- **not summed into net** (mV, one step) | net |
|---|---|---|---|---|---|---|
| lambda_0 (uniform) | +10,687 | +430 | -3,423 | **-47,650** | +6.77 | **-39,956** |
| lambda_1 (bump) | **+8,327** | +178 | **+1,719** | -206 | +6.00 | **+10,019** |
| lambda_2 | +4,033 | +152 | -190 | -73 | +5 | +3,923 |
| lambda_8 (L/R alternation) | +17 | +161 | +19 | -3,080 | 0 | -2,883 |

The `direct EPG->EPG` column is a **one-step mV coefficient and is not part of `net`** (+10,019 = 8,327 + 178 + 1,719 - 206,
mV^2): the two are different units, and the mixed row was a defect of this table. That term is the one section 3.3 shows
should have been carried separately rather than dropped -- its damped k = 1 coefficient is +6.00 mV (gamma_crit = 1 / (tau
d_1) = 33.4 Hz/mV, never reached) and its undamped one +59.76 mV (3.35 Hz/mV, reached); k = 0 is +6.77 damped, +67.50
undamped, and the undamped `cf` (sign+gain) pair is d_0 +76.05 / d_1 +66.51 at +4.164 mV per pair over 842 pairs (against
+0.366 shipped and +3.653 undamped).

* **The k = 1 mode's gain**: lambda_1 = +10,019 mV^2 -> gamma_crit(1) = 2.00 Hz/mV in the two-step reduction; the full
  sub-circuit (46 EPG + 42 PEN + 18 PEG + 42 Delta7 + 4 EPGt + 308 ER/ExR + 4 GLNO, all loop orders) has its leading k = 1
  eigenvalue at mu = +118.4 mV/spike -> gamma_crit 1.69 Hz/mV, and its leading k = 0 mode at +40 +- 238i (gamma_crit 5.0).
  At the nominal 6 Hz/mV the k = 1 loop gain is 9.0 and the k = 0 gain -36. The exact eigenvalues of the 16 x 16 net matrix
  are 10,415 and 9,561 (both k = 1), then 4,147 / 3,660 (k = 2). **The WEIGHT matrix has the k=1 structure a ring attractor
  needs** (lambda_1 = +10,019 mV^2; PEN +8,327, Delta7 +1,719, PEG +178, Ring -206; lambda_0 = -39,956), and a uniform-gain
  linearisation would be unstable in that mode at gamma > 1.69-2.00 Hz/mV. That is a necessary condition, not the attractor:
  the criterion is on diag(gamma_i) tau A with gamma_i = f'(u_i) at a realised state, and at the rate model's own fixed point
  every compass cell has gamma = 0 except Delta7 (3.6 at background, 6.2 during the pulse), so the true Jacobian's leading
  eigenvalue is +0.045 (background) / +0.079 (pulse), i.e. stable in every mode. The wiring could be a ring attractor's; under
  the shipped rules it never is one, because the gain is never there. (`cx_wedge.md` read the weight matrix the same way; the
  Jacobian statement is the independent skeptic pass's, R2 below.)
* **Delta7 inhibition : ring feedback**: k = 0 -3,423 : -47,650 = 0.072; per volley Delta7 -> EPG -25.7 mV against ER/ExR ->
  EPG -767 mV = 0.033; at the peak -433 (opposite wedge) against the flat -3,001 = 0.14. At k = 1 the ratio inverts: Delta7
  contributes +1,719 (its cosine is the bump-sharpening term) and the ring feedback -206 (2 % of PEN's +8,327): **the ER/ExR
  loop carries no k = 1 content (-206, 2.5 % of PEN's +8,327); it is untuned in the bump mode.** It is **not purely k = 0**:
  its spectrum is -47,650 (k = 0), -3,080 (k = 8, the L/R alternation, 6.5 % of k = 0 and 15x the k = 1 one), -206 (k = 1),
  -73 (k = 2), -39 (k = 3), -51 (k = 4).
* **Why there is no bump at gamma x1 anyway** -- the fixed point (10 Hz EPG background, wedges 0-3 at +40 Hz, release):

| state | u_PEN (driven glomeruli) | = EPG | + Delta7 | + ER/ExR | of which | u_EPG (driven) | = Ring | Delta7 | EPG | PEN rate | Delta7 | ExR6 | ER6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| background | **-11.1 mV** | +4.2 | -1.8 | -13.5 | ExR6 -8.9 @ 110 Hz, ER6 -4.5 @ 54 | -22.0 | -21.3 | -1.0 | +0.3 | 0.0 Hz | 7.2 | 110 | 54 |
| pulse (EPG in 50 Hz) | **-15.8 mV** | +9.7 | -0.6 | -24.9 | ExR6 -14.8 @ 183, ER6 -9.8 @ 116 | -45.0 | -45.9 | -0.5 | +1.4 | 0.0 | 34.9 | 183 | 116 |
| after | -11.1 | +4.2 | -1.8 | -13.5 | | -22.0 | | | | 0.0 | 7.2 | | |

Every per-type rate in that table is a **rate-model output; not measured** (the skeptic pass's correction 5, below). The same fixed point puts
2 ExR6 + 4 ER6 + 11 ER4m at a summed **484 Hz** at background and **1,075 Hz** during the pulse, while in the batch the LIF's
*entire* 308-cell ER/ExR population sums to 244 Hz after release, 630 Hz at the end of the pulse and peaks at 800 Hz (arm S
seed 0: ring population mean 0.807 / 2.044 / 2.599 / 0.792 Hz) -- so **the DC term is overstated by roughly 2x** and the
per-type ring rates are not quantities this batch measured. The conclusion rests on the measured PEN rate, not on these.

PEN sits 1.6-2.3 threshold gaps below threshold at every stage, and the pulse *lowers* u_PEN (the EPG drives ExR6 / ER6
harder than it drives PEN: EPG -> ExR6 +11.4 mV per pair, EPG -> PEN +5.05). gamma_PEN = 0 in this model -- and note that it
is 0 **by construction**, because `rate_fixed_point` adds the forced background as a rate rather than as a current (section
3.3): the loop is open and the k = 1 gain is never engaged. The measured statement, in the LIF, is that **the relays stay far
below the rate a bump needs**: the PEN population mean (42 cells) is 0.29-0.66 Hz during the pulse in S and 0.02-0.06 after,
against `INTERP.md` 10's definition of `silent` (max rate < 0.5 Hz per cell), which none of these arms meets. The mechanism
that is missing is not a ring mode; it is a DC balance on the relays, set by 2 ExR6 + 4 ER6 (+ 11 ER4m onto EPG) that the EPG
itself drives.

### 1.3 One change at a time (each labelled; `structure.md` "Configurations")

| # | change | evidence / status | lambda_1 net | gamma_crit k1 (two-step / circuit) | u_PEN during pulse | Delta7 -> EPG volley | Ring -> EPG / PEN volley | fixed point after release |
|---|---|---|---|---|---|---|---|---|
| 0 | shipped | reference | +10,019 | 2.00 / 1.69 | -15.8 | -25.7 | -767 / -96 | uniform 10 / 10 Hz, PEN 0.0 |
| a | GLNO = glutamate | MaleCNS T-bars 51 % glu / 37 % ACh (conf 0.48 'unclear'); the hemibrain name (GLutamatergic LAL-NOduli); BANC v888 `nt` glutamate on 4/4 (`evidence_glno.json`) -- a relabel, no code path | +10,019 | 2.00 / 1.69 | -15.8 (GLNO fires 0.03-0.23 Hz) | -25.7 | -767 / -96 | uniform, PEN 0.0 |
| b | cap lifted (conn_cap 0) | INSTRUMENT, global | +11,560 | 1.86 / 1.55 | **-41.0** | -25.6 | -773 / -130 | uniform, PEN 0.0 |
| c | receptor sign+gain (abs) | the table's expression tertiles (data) x {0.5, 1, 1.5} (a parameter map); an existing opt-in mode | **+14,977** | 1.63 / 1.37 | -17.6 | -34.5 | -995 / -124 | uniform, PEN 0.0 |
| d | monoamine slow class (`full`) on EPG / PEN | 724 / 514 entries; +0.09 / +0.06 mV per presynaptic Hz per cell (EL octopamine, ExR2 / FB1C dopamine); < 0.2 mV at their rest rates | = c (fast weights) | = c | = c | | | inert |
| e | fan-in on PEN | totals 867-1,598, scale 1.00 (0 of 42 below 1); EPG 2,759-5,099, scale 0.98-1.00 -- confirmed, nothing to change | | | | | | |
| f | same-type damping off (same_type_gain 1) | INSTRUMENT, global; the synapses are data, the x0.1 is the hand rule | +9,985 | 2.00 / 1.35 | -15.6 (background **-8.4**) | -25.6 | -764 / -96 | uniform, PEN 0.0, Delta7 2.4 Hz |
| a+c | | | +14,977 | 1.63 / 1.37 | -17.7 | | | uniform |
| a+b | GLNO -> PEN uncapped: -53.6 mV per pair, -107 per volley | | +11,560 | 1.86 / 1.55 | -41.1 | | | uniform |
| a+f | | | +9,985 | 2.00 / 1.35 | -15.7 | | | uniform |
| c+f | | | +14,292 | 1.67 / **1.15** | -15.8 (background -8.8) | -32.9 | -950 / -124 | uniform |
| b+f | | | +11,492 | 1.87 / 1.26 | -37.5 | | | uniform |

* **(a) GLNO = glutamate.** 84 GLNO -> PEN edges, 16,371 raw synapses, every edge 88-345 (mean 195) -- 84 of 84 above the cap:
  capped -16.5 mV per pair (-33.0 per PEN per volley of its two contralateral GLNO), uncapped -53.6 / -107. PEN -> GLNO +10.6
  per pair, +222 per GLNO per PEN volley (18 of 84 above the cap). At the shipped gains the loop PEN -> GLNO -> PEN is inert
  because the relays stay far below the rate a bump needs (PEN 0.29-0.66 Hz during the pulse in S; its other inputs --
  PS196_b 19 %, LAL139, LAL184, `cx_shift.md` 1 -- are at rest); the relabel changes the ring's numbers in the fourth
  decimal. In the LIF, **GLNO fires 0.00 Hz before the pulse, 1.4-3.0 Hz during it and 0.06-0.39 Hz after** (the 0.03-0.23 Hz
  figures in the rate-model column of the table above are the rate model's, 10-20x lower). At 3 Hz its -33 mV per PEN per
  volley is -0.5 mV on a PEN 7 mV below threshold, so G vs S at the shipped gains is a **structural null by construction
  rather than a test of the relabel**: this batch tests GLNO's sign only where GLNO fires (F 87-91 Hz vs FG; R 130-137 Hz vs
  RG). It matters only once PEN fires (the reference gains, `cx_glno.md` 3-5; and the F family, section 3).
* **(b) The cap** (capped vs uncapped per pair; `structure.md` "The connection cap"): EPG -> PEN +5.05 -> +5.11 (14 of 664 pairs
  above 60; mean 19 synapses), PEN -> EPG +7.98 -> +8.97 (133 of 735), EPG -> PEG +3.36 -> +4.47 (45 of 379), Delta7 -> EPG
  -2.46 -> -2.46 (**0 of 479 above the cap**), Delta7 -> PEN -4.70 -> -4.93 (27 of 404), ExR4 -> PEN -15.2 -> -23.3 (55 of 84),
  ExR6 -> PEN -8.66 -> -17.7 (40 of 82), ExR6 -> EPG -15.0 -> -18.6 (53 of 92), ER4m -> EPG -11.4 -> -11.5 (36 of 506).
  **The cap does not flatten the Delta7 cosine** (its profile is -44 ... -433 capped and -43 ... -432 uncapped; no Delta7 -> EPG
  edge reaches 60) and it does not clip the ring's excitation much (lambda_1 PEN +8,327 -> +9,778); what it clips are the giant
  ExR4 / ExR6 / GLNO edges, so lifting it doubles the DC inhibition on PEN (-25 -> -50 mV during the pulse). The edges it
  touches are the same transmitter class as the tuned ones (`unitary_strength.md`), so it is not adoptable per type without a
  mechanism; reported as an instrument only and not spent a GPU arm on.
* **(c) Receptor tiers** (`receptors_by_type.csv`; `structure.md` first tables): EPG rows tier `alias` (Davis 2020 PB_2),
  PEN_a / PEN_b `fuzzy` (Davis PB_3), Delta7 `alias` (PB_1); PEG, EPGt, GLNO, ExR1 / 4 / 5 / 6 / 7 / 8 have no row (tier
  fallback = NT_SIGN). ER / ExR -> EPG: GABA fast -1 **Rdl (GABA-A), class high**, slow -1 GABA-B-R3 high (the classical
  metabotropic slow class is OFF by default); the glutamatergic ExR4 / ExR5 / ExR6 -> EPG: fast -1 **GluClalpha high** (Nmdar2
  is the + lead, net -1 under class / abs / nonmda), mGluR none. Delta7 -> EPG: the same glutamate row (GluCl, no mGluR).
  EPG -> PEN: **nAChRbeta1** fast +1 high, mAChR-A slow +1 mid; EPG -> Delta7: nAChRbeta1 mid. Modes: `sign` = NT_SIGN on every
  ring-core entry (0 changed under class / abs / nonmda -- `cx_glno.md` 2); `sign+gain` multiplies the sub-cap magnitudes by
  the class factor: x1.5 on 15,632 of 16,848 EPG input entries (GABA 11,624, ACh 2,716, Glu 1,292) and 4,800 of 5,596 PEN
  entries, x1.5 on Delta7's 1,869 glutamate entries and x1 on its 2,983 ACh ones, x1 on PEG (no row); `full` = `sign+gain`
  plus the monoamine slow term (d). So sign+gain raises the tuned ring (lambda_1 +50 %, gamma_crit 1.63 / 1.37) **and** the
  untuned feedback (Ring -> EPG -767 -> -995 per volley; ExR6's capped edges stay) by the same factor; u_PEN goes from -15.8
  to -17.6.
* **(d) The monoamine slow class on the ring.** Under `full` (slow_gain 0.02, tau 200 ms: steady tone per presynaptic Hz =
  w_syn x 0.02 x count x sign x class x 0.2 s): EPG 724 entries, +0.090 mV per Hz per cell (EL octopamine +2.04 summed over
  post, ExR2 dopamine +1.58, FB4M / FB1C +0.15), PEN 514 entries, +0.057 (ExR2 +0.86, FB1C +0.84, EL +0.44). The EPG row's own
  entries are DopEcR +1 high, 5-HT7 / 5-HT1B mixed -1 mid, Octbeta1R +1 low; PEN's DopEcR / Octbeta1R +1 high, 5-HT mixed.
  At the presynaptic rest rates (0-2 Hz) that is < 0.2 mV per cell; restricting the class to the ring is not reachable
  through a flag (`slow_gain_by_class` is per class, not per type) and, at these magnitudes, would not need to be.
* **(e) Fan-in on PEN**: confirmed 1.00 (totals 867-1,598 against the 5,000 reference; also 1.00 uncapped, 932-1,874).
* **(f) Same-type damping**: EPG -> EPG 842 pairs, +0.37 per pair damped (+6.7 per volley) / +3.65 undamped (+66.9;
  wedge-local: +24.3 / +17.8 / +2.4 mV per spike at 0 / 1 / 2 wedges), PEN_a -> PEN_a 257 pairs (+7.5 -> +74.9), PEN_b ->
  PEN_b 274 (+7.8 -> +77.8), Delta7 -> Delta7 1,717 pairs (-17.4 -> -173.8), PEG 114 (+0.8 -> +8.3). PEN_a <-> PEN_b (269 +
  269 pairs, +39.7 / +42.2 per volley) are **not** damped -- different type strings -- so 84 % of the PEN -> PEN volley is
  already at face value in the shipped model. Undamped, Delta7's mutual inhibition drops Delta7 to 2.4 Hz at background (from
  7.2) and EPG -> EPG adds +3.2 mV to EPG; u_PEN improves by 2.7 mV at background (-8.4) and by 0.2 during the pulse (-15.6);
  no bump in the rate model.

**Ranking** (by the fixed point's bump -- none; then gamma_crit(k1), then the PEN margin during the pulse): c (1.63 / 1.37,
-17.6) ~ a+c > c+f (1.67 / 1.15, -15.8) > b (1.86 / 1.55, -41.0) > a = shipped (2.00 / 1.69, -15.8) ~ f (2.00 / 1.35, -15.6).
No single data-implied change, and no pair, moves the rate model's fixed point off the uniform state at the shipped gains;
the changes that raise the k = 1 gain (c, f) raise it by 30-50 % while the relays stay 2 gaps below threshold. The batch tests
the three candidates the ranking puts first (c, f, c+f), each with and without the GLNO relabel, against the shipped path,
with the experiment gains as the labelled reference.

**This ranking is computed at a zero-gain state and should not be trusted until the structure tool is fixed.** The fixed
point it ranks by has gamma = 0 on every compass cell (section 1.2), and the miss on the F family (section 3.3) is the
demonstration: `f` is ranked last-equal here and is the only shipped-gain arm that holds a bump in the batch. The two defects
are named in section 3.3 and in `scripts/cx_ring_structure.py`'s module docstring.

## 2. Predeclaration and the batch

`out/cx5/predeclared.json`, stamped **2026-09-15T05:39:53Z** (`written_before_submission` true). Batch `cx5` on the house
cluster: ONE `cluster_run.py --name cx5 --minutes 30 --arm-block fam` call, 8 jobs = 4 seeds x {GLNO silent, GLNO =
glutamate}, each job running its 6 arms sequentially (`exit $((s0 | ... | s5))` preserves every python exit code), blocks
`fam_s<seed>`, fetch `out/cx5/`. Twelve arms (all `cx_wedge.py --no-structure --sim <gE:gD> --ledger`; the cx_wedge protocol:
full connectome, compass adaptation 0, 10 Hz Poisson background on all 46 EPG, 1 s settle, wedges 0-3 (11 EPG) at +40 Hz for
2 s, 5 s free; EPG per 10 ms frame scored by `probe_compass_room.bump_frames` -- vs > 0.6, >= 8 of the block's 11 cells > 22
Hz, <= 3 of 35 outside; survival = end of the last confined post-pulse frame - pulse end, max 5.00 s):

| arm | gains | GLNO | flags | classification |
|---|---|---|---|---|
| S | 1 / 1 | silent | `--receptor-model shipped` (sign / abs) | the shipped path (reference) |
| G | 1 / 1 | glutamate | `--nt-override GLNO=glutamate` | data-implied relabel |
| C | 1 / 1 | silent | `--receptor-model sign+gain --receptor-net-rule abs` | data-implied tier, parameter factor map; existing opt-in mode |
| CG | 1 / 1 | glutamate | C + G | |
| F | 1 / 1 | silent | `--lif same_type_gain=1` | INSTRUMENT (global) |
| FG | 1 / 1 | glutamate | F + G | |
| CF | 1 / 1 | silent | C + F | the structure pass's lowest gamma_crit |
| CFG | 1 / 1 | glutamate | C + F + G | |
| R | 2 / 15 | silent | `--no-delta7-pen` (gD on Delta7 -> EPG only) | LABELLED reference: the bump that exists |
| RG | 2 / 15 | glutamate | | LABELLED reference: cx_glno's glu row (holds, -10 %) |
| R175 | 1.75 / 15 | silent | | LABELLED reference (3/6 in cx_glno) |
| RG175 | 1.75 / 15 | glutamate | | LABELLED reference: the point cx_glno's glutamatergic GLNO abolished 6/6 |

Primaries per arm vs S: `survival_s`, `bump_hz_post`, `width_half_post`, `frac_confined_post` (`common.compare`, runs = the
unit, 4 v 4; Holm within each arm's four; zero-SD nulls are structural). Decision rule, in the predeclared words: **'a working
compass' = survival >= 5 s AND width 2.5-5 wedges AND rate 5-60 Hz in >= 3 of 4 seeds at the shipped gains.** Predicted from
section 1: every gE 1 / gD 1 arm scores survival 0.00 in 4/4 (structural null vs S); R and RG hold a bump at 150-220 Hz (rate
FAIL, width PASS); RG175 dies 4/4. (The prediction is wrong for F, CF and CFG -- section 3.)

**Submissions.** The first submission (`cx5-fd8b02`, local `date -u` 2026-09-15T05:40:00Z) **was cancelled 8/8; the client ran
3.7 min and reported `8 job(s), 8 failed` with `FETCH FAILED`, so no artefact of it was retrieved. Its archived console
(`client_console_fd8b02_cancelled.txt`) does carry 21 of the 48 runs' result lines -- they appear only inside the client's
cancellation log dump, which was written at 05:43:46Z, after the resubmission was stamped at 05:42:48Z, so the resubmission
decision preceded them. Every one of those 21 lines is identical to the second batch's (F s3 151.1 / 5.00 s, CFG s3 156.4,
CF s3 144.4 / 4.92 s, R s3 200.8), as determinism on this protocol requires. No pre-amendment copy of `predeclared.json` was
archived, so 'arms and primaries unchanged' is verifiable only for the arms: the 8 job lines recovered from the cancelled
console are identical to `batch.sh`'s once the exit expression is normalised. Archive the pre-amendment predeclaration next
time.** (`out/cx5/predeclared.json`'s `submission_history` says "cancelled ... within two minutes and before any result was
read"; that stamped record is left as written, and **its wording is superseded by this paragraph**.) The defect itself: the job
lines carried `exit $(( |  |  |  |  | ))` -- the `$s0 ... $s5` inside the arithmetic had been expanded at submission time, so
the python exit codes would not have reached the scheduler (console archived as
`out/cx5/client_console_fd8b02_cancelled.txt`). The generator was fixed (bare names inside `$(( ))`), `batch.sh` regenerated
(sha256 04f8fdb8...9d3e, the value recorded in `predeclared.json`), the predeclaration amended with this note
(`submission_history`), and the
batch resubmitted as **`cx5-5cde5e`**: local `date -u` **2026-09-15T05:42:48Z**, scheduler `submitted_at` 05:42:48.84-05:42:50.17Z
(`out/cx5/scheduler_receipt.json`), all 8 jobs on <cluster-node> (GPU ids 1-3, NVIDIA B200), finished 05:45:28-05:45:59Z. Scheduler
receipt: **8 job(s), 0 failed ({'completed': 8})**; client console (`out/cx5_cluster.log`): `8 job(s), 0 failed  (6.0 min)  run
dir <cluster-fs>/neurome/runs/cx5-5cde5e`, `client exit 0`. Artefacts: 48 `<arm>_s<seed>.json`, 48 `.txt`, 48 `.npz`; 48 of
48 consoles say `device cuda`; every run's `provenance.execution` is `cuda` / `NVIDIA B200` / `<cluster-node>`; the analysis's per-run
checks (arm == file name, gains, `nt_override` / `glno_nt`, `same_type_gain`, `receptor_model`) raise **0 problems**;
compiled-W md5 ef23cc27... on every GLNO-silent run and 7a10d93b... on every GLNO = glutamate run (the same two hashes as on
this desktop). Wall 13.5-24.8 s per run.

## 3. Results (`n_runs 48 (out/cx5/<arm>_s<seed>.json); problems 0` -- `analysis.md`)

### 3.1 Per run (rates at 5 s after release unless named; `runs.csv`)

| arm | seed | survival s | bump Hz | width | confined post | EPG in / out | EPG in during pulse | PEN during / post | Delta7 post | Ring | GLNO | rest | profile after (wedges 0-4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | 0 / 1 / 2 / 3 | 0 / 0 / 0 / 0 | - | - | 0 / 0 / 0 / 0 | 10.2 / 9.6; 10.4 / 9.9; 10.8 / 9.8; 10.2 / 9.9 | 57 / 57 / 54 / 50 | 0.4-0.7 / 0.02-0.06 | 10.2-10.5 | 0.8 | 0.1-0.3 | 0.00 | 11 10 11 9 9 |
| G | 0 / 1 / 2 / 3 | 0 / 0 / 0 / 0 | - | - | 0 / 0 / 0 / 0 | = S to 0.1 Hz | | 0.3-0.5 / 0.02-0.08 | 10.2-10.5 | 0.8 | 0.06-0.4 | 0.00 | = S |
| C | 0 / 1 / 2 / 3 | 0 / 0 / 0 / 0 | - | - | 0 / 0 / 0 / 0 | 10.2-11.3 / 9.6-10.0 | | 4.4-6.4 / 0.3-0.7 | 8.9-9.7 | 1.0-1.1 | 0.7-2.5 | 0.00 | |
| CG | 0 / 1 / 2 / 3 | 0 / 0 / 0 / 0 | - | - | 0 / 0 / 0 / 0 | 10.2-11.0 / 9.6-9.9 | | 2.9-4.0 / 0.25-0.46 | 8.9-9.6 | 1.0-1.1 | 0.5-1.7 | 0.00 | |
| **F** | 0 / 1 / 2 / 3 | **5.00 x4** | **155.8 / 150.9 / 152.3 / 151.1** | 3.0 x4 | 0.87 / 0.85 / 0.82 / 0.83 | 155 / 9.6; 150 / 9.9; 151 / 9.8; 151 / 9.9 | **193 / 192 / 186 / 190** | 25-27 / 29.6-30.9 | 37.4-38.8 | 5.7-5.8 | 87-91 | 0.04 | 153 222 201 37 9 |
| FG | 0 / 1 / 2 / 3 | 0.08 / 0.15 / 0.14 / 0.15 | (95 / 124 / 91 / 104 over 1-3 frames) | 2-3 | 0.01-0.03 | 12.9-15.7 / 10.2-10.8 | 111 / 149 / 102 / 120 | 4.9-14.1 / 0.8-1.6 | 4.7-5.4 | 0.9-1.0 | 2.6-4.2 | 0.01-0.02 | 12 14 15 10 9 |
| CF | 0 / 1 / 2 / 3 | 5.00 / 4.96 / 5.00 / 4.92 | 143 / 149 / 142 / 144 | 2.2 / 2.9 / 2.0 / 2.1 | 0.29 / 0.71 / 0.36 / 0.34 | 36 / 49; 124 / 19; 23 / 55; 27 / 53 | 178 / 177 / 193 / 184 | 133-247 / **185-276** | 29.7-30.3 | 20-27 | **227-289** | 0.28-0.32 | bump at wedges 7-8 (s0, s2, s3; 178-209 Hz), at 0-2 until 3 s then 7-8 (s1) |
| **CFG** | 0 / 1 / 2 / 3 | **5.00 x4** | **158.2 / 157.3 / 156.3 / 156.4** | 3.0 x4 | 0.86 / 0.83 / 0.81 / 0.83 | 158 / 9.6; 156 / 10.0; 154 / 10.0; 156 / 9.9 | 194 / 198 / 188 / 194 | 34-37 / 39.2-39.8 | 30.0-30.9 | 7.6-7.7 | 98-100 | 0.04-0.05 | 143 228 205 42 9 |
| R | 0 / 1 / 2 / 3 | 5.00 x4 | 203.2 / 201.0 / 201.1 / 200.8 | 3.06 / 3.03 / 3.04 / 3.00 | 0.87 / 0.85 / 0.85 / 0.84 | 203 / 9.6; 200 / 9.9; 200 / 9.7; 200 / 9.9 | 227-236 | 46-50 / 47.8-48.4 | 99.8-100.7 | 9.5-9.6 | 133-134 | 0.03 | |
| RG | 0 / 1 / 2 / 3 | 5.00 x4 | 183.7 / 181.4 / 181.7 / 181.6 | 2.84 / 2.64 / 2.70 / 2.74 | 0.87 / 0.85 / 0.85 / 0.84 | 183 / 9.6; 181 / 9.9; 181 / 9.7; 181 / 9.9 | 199-210 | 37-42 / 40.2-40.8 | 94.7-95.6 | 8.5-8.6 | 115-117 | 0.03 | |
| R175 | 0 / 1 / 2 / 3 | 5.00 x4 | 167.2 / 155.4 / 163.6 / 138.5 | 2.27 / 2.31 / 2.12 / 2.84 | 0.87 / 0.84 / 0.85 / 0.75 | 167 / 9.6; 124 / 19.9; 163 / 9.7; 61 / 35.6 | 179-194 | 38-46 / 37.5-41.3 | 79-89 | 7.3-8.1 | 96-118 | 0.03 | seeds 1 and 3 jump (to centre 4, at 3-5 s and at 1 s) |
| RG175 | 0 / 1 / 2 / 3 | 1.55 / 3.29 / 0.25 / 1.33 | (145 / 134 / 134 / 134 while it lasts) | 2.0 | 0.28 / 0.58 / 0.05 / 0.17 | 54 / 12; 94 / 12; 21 / 11; 41 / 12 | 145-172 | 24-34 / 4.9-21.7 | 19-58 | 1.1-4.8 | 9-61 | 0.00-0.02 | dead by 2 s (s0, s2, s3), by 5 s (s1) |

Per-seed scatter of the four primaries per arm: `out/cx5/analysis/scatter.png` (copied to
`docs/audits/compass_ring_mechanism_scatter.png`).

**Cross-batch check (a different run directory, the same GPU class as cx_glno's B200 rows):** R seeds 0-2 at 5 s read
201 / 10 (11, 1), 201 / 9 (10, 3), 204 / 9 (11, 2) Hz -- `cx_glno.md` 3's base rows are 201 / 10 (11, 1), 201 / 9 (10, 3),
204 / 9 (11, 2); RG 180 / 10, 181 / 9, 184 / 9 against glu 180 / 10, 181 / 9, 184 / 9; RG175 10 / 13 (0, 7), 12 / 15 (2, 7),
8 / 11 (0, 4) against glu 1.75 / 15 10 / 13 (0, 7), 12 / 15 (2, 7), 8 / 11 (0, 4). R175 reproduces too, one row more than was
claimed here first: 164 / 10 (10, 1), **42 / 42 (3, 9)**, 169 / 9 (9, 2) against `cx_glno.md`'s base 1.75 / 15. Identical to
the printed digit: the runs are deterministic given (config, gains, seed) on this backend, as `cx_glno.md` 5 found (27
repeated keys, 0 differing fields). **Which config is matched: cx_glno's `base` (receptor model *off*) and its identical
`sign-class` rows -- not its `sign-abs` rows (201 / 10, 200 / 9, 203 / 9 at 2.0 / 15), although cx5's arms declare
`--receptor-model shipped` = sign / abs.** That is consistent with the receptor stage being a bit-exact no-op on the ring
core (section 1), but the config had to be named.

### 3.2 The decision rule and the verdicts (`decision.csv`, `compare.csv`)

| arm | gains | seeds with a surviving bump (>= 5 s) | seeds meeting the rule (survival AND width 2.5-5 AND rate 5-60) | working compass | ledger survival / rate / width |
|---|---|---|---|---|---|
| S | 1 / 1 | 0 / 4 | 0 / 4 | no | FAIL / n.a. / n.a. x4 |
| G | 1 / 1 | 0 / 4 | 0 / 4 | no | FAIL / n.a. / n.a. x4 |
| C | 1 / 1 | 0 / 4 | 0 / 4 | no | FAIL / n.a. / n.a. x4 |
| CG | 1 / 1 | 0 / 4 | 0 / 4 | no | FAIL / n.a. / n.a. x4 |
| F | 1 / 1 | **4 / 4** | 0 / 4 (rate 151-156) | no -- a bump, not a working compass | PASS / FAIL / PASS x4 |
| FG | 1 / 1 | 0 / 4 | 0 / 4 | no | FAIL / n.a. / n.a. x4 |
| CF | 1 / 1 | 2 / 4 (5.00, 5.00; 4.96, 4.92) | 0 / 4 (rate 142-149; width 2.0-2.9) | no | PASS / FAIL / FAIL-PASS |
| CFG | 1 / 1 | **4 / 4** | 0 / 4 (rate 156-158) | no -- a bump, not a working compass | PASS / FAIL / PASS x4 |
| R | 2 / 15 | 4 / 4 | 0 / 4 (rate 201-203) | no (reference) | PASS / FAIL / PASS x4 |
| RG | 2 / 15 | 4 / 4 | 0 / 4 (rate 181-184) | no (reference) | PASS / FAIL / PASS x4 |
| R175 | 1.75 / 15 | 4 / 4 | 0 / 4 (rate 139-167; width 2.1-2.8) | no (reference) | PASS / FAIL / FAIL-PASS |
| RG175 | 1.75 / 15 | 0 / 4 | 0 / 4 | no (reference) | FAIL / n.a. / n.a. x4 |

`common.compare` vs S (4 v 4, p_floor 0.029; Holm within each arm's four primaries): S's `survival_s` and
`frac_confined_post` are 0.00 in every run, so every comparison has a **zero-SD null**. G, C, CG: diff 0.00 on both, verdict
`null` (a structural zero-vs-zero; their `bump_hz_post` / `width_half_post` have no confined frame in either arm -- "no
data"). F, CF, CFG, R, RG, R175: survival diff +5.00 / +4.97 / +5.00 / +5.00 / +5.00 / +5.00 and confined-fraction diff +0.84 /
+0.42 / +0.83 / +0.85 / +0.85 / +0.82, exact-U p 0.029 (Holm 0.057), verdict **`undetermined`** by compare's rule (z undefined
on a deterministic null); read as magnitudes they are the whole effect: 0.00 -> 5.00 s in 4/4 seeds. FG and RG175: survival
+0.13 / +1.6 s (`undetermined`, p 0.029): bumps that die. `bump_hz_post` / `width_half_post` are one-sided for every arm (the
reference has no confined frame): F 151-156 Hz / 3.0, CFG 156-158 / 3.0, R 201-203 / 3.0-3.06, RG 181-184 / 2.6-2.8 -- all
above the 5-60 Hz row. **No arm meets the predeclared rule; every surviving bump fails on rate; the shipped-gain answer is
"no working compass", with a bump in the F family.** (A note on `compare.csv`: pandas reads the verdict string `null` back as
NaN; `analysis.json` and `analysis.md` carry the string.)

A caveat on the R175 row. `bump_survival_s` scores the last confined frame **wherever the bump is, not at the driven tile**:
R175 seeds 1 and 3 read 5.00 s with the bump at centre 4.1 and only 3-4 of 11 driven cells above 22 Hz at 5 s (42/42 and
46/43 in/out Hz), which is cx_glno's "not captured". The tile-level count is 2/4, the ledger count 4/4;
`frac_confined_post` (0.746-0.868) and the centre are what carry the difference.

### 3.3 Reading the F family

* **F (damping off, GLNO silent)** holds a 3-wedge bump for the whole 5 s in 4/4 seeds at 151-156 Hz (profile 153 / 222 /
  201 / 37 then 9-12 Hz), vs 0.72-0.76, 9-10 of 11 driven cells above 22 Hz and 1-4 of 35 outside, PEN 30 Hz, Delta7 38 Hz,
  ER/ExR 5.7 Hz, GLNO 87-91 Hz (sign 0: it fires but transmits nothing), rest of brain 0.04 Hz. During the pulse the driven
  EPGs run at 186-193 Hz against 50-57 in S: the undamped EPG -> EPG synapses (+24 mV per spike within the wedge, +18 at +-1)
  make the pulsed wedges self-amplifying, PEN then crosses threshold (25-27 Hz during the pulse vs 0.3-0.7 in S) and the loop
  closes. After release the wedge's own recurrence (+26 mV per volley at 150 Hz = 19 mV per 5 ms) is enough on its own; Delta7
  (38 Hz) and the ExR feedback keep the rest at 10 Hz. Width 3.0 = the +-1-wedge reach of EPG -> EPG. **The bump is carried by
  the EPG's within-type synapses, not by the PEN loop that `cx_wedge.md` gains up**; it is the "runaway clique" the x0.1 was
  introduced against, confined here by Delta7 and the ring neurons. **This attribution is inferred from the structure pass,
  not measured**: `same_type_gain` is one global scalar, so no arm damps EPG -> EPG while leaving PEN_a -> PEN_a,
  PEN_b -> PEN_b, Delta7 -> Delta7 and PEG -> PEG undamped, and in F the PEN loop *is* engaged (PEN 25-27 Hz during the pulse,
  30 Hz after). The independent support is structural: the undamped EPG -> EPG 16-wedge k = 1 coefficient +59.76 mV is
  supercritical on its own (gamma_crit 3.35 Hz/mV) and predicts a saturation at 145 Hz against the observed 151-156.
* **FG (the same with GLNO = glutamate)**: the bump collapses within 0.08-0.15 s of release in 4/4 seeds (GLNO 90 Hz in F now
  delivers its -33 mV per PEN volley: PEN 0.8-1.6 Hz after release, EPG in 13-16 Hz). The correct-sign ring does not hold this
  bump.
* **CFG (sign+gain + damping off + GLNO = glutamate)**: 156-158 Hz in 4/4, width 3.0, PEN 39 Hz, Delta7 30, GLNO 98-100, rest
  0.04: with EPG -> PEN at +114 mV per volley (x1.5 on its sub-cap edges) the GLNO volley no longer opens the loop. This is the
  only configuration at the shipped gains in which the correct-sign ring carries a bump -- and it is one existing opt-in mode
  plus one global instrument, at 3x the ledger's rate bound.
* **CF (sign+gain + damping off, GLNO silent)** is a different state: PEN 185-276 Hz and GLNO 227-289 Hz after release, ER/ExR
  20-27 Hz, rest of brain 0.28-0.32 Hz, the bump at wedges 7-8 (centre 7.4) in three seeds and, in seed 1, at 0-2 until 3 s
  and then at 7-8. Confined by the rule in 29-71 % of the post-pulse frames, survival 4.92-5.00 because the last frame is
  confined -- but not at the driven tile. With PEN_a -> PEN_a / PEN_b -> PEN_b x1.5 undamped (+110 / +113 mV per volley) and no
  signed GLNO to brake PEN, the PEN population saturates; the GLNO relabel is what tames it in CFG.
* **The structure pass's miss -- two defects of this thread's own tool.** The rate model's fixed point at f put the driven
  EPG at u = -21.6 mV during the pulse and PEN at -15.6, i.e. no bump, while the LIF runs the driven EPGs from 57 to 190 Hz.
  **The structure pass's miss is an analysis choice, not a fluctuation regime.** (i) The EPG-only reduction keeps only the
  two-step terms; the one-step EPG -> EPG matrix is computed and then dropped (`M16['net'] = PEN + PEG + Delta7 + Ring`;
  `direct` is printed in mV in the same table row as the mV^2 lambdas and never used). Its k = 1 ring-Fourier coefficient is
  +6.00 mV damped (gamma_crit = 1 / (tau d_1) = 33.4 Hz/mV, never reached) and +59.76 mV undamped (3.35 Hz/mV, reached by the
  LIF at any u below ~34 mV); the rate at which the gain falls back to 1 is 145 Hz for f and 161 Hz for cf, against the
  observed 151-156 (F) and 156-158 (CFG) -- 4-7 %. (ii) `rate_fixed_point` adds the forced background as a **rate** rather
  than as a **current** (`r = f(tau A r) + forced`), so the driven EPG sits at u -22 to -68 mV while "firing" at 10-50 Hz and
  gamma_EPG = 0 by construction -- no recurrent EPG term, direct or two-step, can engage. With the same deterministic model,
  the same sigma 2 mV, the drive entered as a current (10 Hz <-> 6.63 mV, 50 Hz <-> 11.99 mV) and the one-step term kept, the
  EPG-only 16-wedge model returns to 14.3 / 15.3 Hz under the shipped damping (60.1 / 16.3 during the pulse, no bump) and runs
  away to 264 / 272 Hz undamped. **The open question is not a diffusion approximation; it is to fix these two** -- and until
  it is fixed, section 1.3's ranking is computed at a zero-gain state (the defects are also named in the module docstring of
  `scripts/cx_ring_structure.py`). The eigenmode statement stands as a statement about the weight matrix (the k = 1 structure
  is there); the fixed-point prediction for large-unitary configurations does not.

## 4. What an adoption would require (nothing is adopted)

* **The GLNO relabel** (G): an entry `"GLNO": "glutamate"` in `connectome.TYPE_NT_OVERRIDE` with its evidence line (MaleCNS
  T-bars glutamate 51 % / ACh 37 % at conf 0.48, all consensus columns 'unclear'; the hemibrain name; BANC v888 `nt` glutamate
  4/4 -- three sources, none an expression profile: GLNO has no receptor-table row), a cache rebuild, and the round-2 rule:
  the full benchmark suite at >= 3 draws with no check changing status (the cluster shows 0 failed here, but a compass-only
  batch is not the suite). What it would change: nothing that decides anything at the shipped gains (**G = S exactly on the
  four primaries and the t5.0 EPG in/out row in 4/4 seeds, not on every ring metric** -- 12-17 secondaries differ per seed,
  up to +-30 % on `GLNO_mean_post`, `PEN_mean_during`, `PEG_mean_post`; and the comparison is a **structural null** because
  GLNO fires 0.00 / 1.4-3.0 / 0.06-0.39 Hz before / during / after the pulse, worth -0.5 mV on a PEN 7 mV below threshold),
  -10 % on the experiment bump (RG 181-184 vs R 201-203), the gE 1.75 point lost (RG175 0/4 vs R175 4/4), and
  the F-family bump lost (FG) unless the receptor gain tier is on (CFG). `docs/audits/glno_relabel.md` (another thread,
  untracked in this tree) is the place for that decision; this audit only supplies the shipped-gain result, and it should be
  handed over as exactly that: **a `null` at these gains -- a silent neuron's sign is untested, not confirmed harmless.**
* **The receptor `sign+gain` tier** (C): an existing opt-in mode; adopting it as the default is a global change to 25.6 M
  entries (every matched row's gain class), the suite x >= 3 draws on the CPU path as well as the GPU; on the ring alone it is
  inert (C = no bump; CF / CFG differ from F only through the PEN saturation / GLNO brake above).
* **The same-type damping** (F): `same_type_gain` is one scalar; "off for the compass only" is a code path that does not exist
  (a per-type table would be a new `LIFParams` field, default `None`, CPU bit-identity test), and off globally re-opens the
  runaway cliques the x0.1 was adopted against (`brain.py` comment: FR1, lLN1_bc, DLMn, DNg33) -- untested here because this
  batch has no world (rest of brain 0.04 Hz in F is the ring's own drive). Adoption would need: the per-type mechanism with a
  data argument, and **the data argument is the other way round from how this audit first put it.** The connectome's 842
  EPG -> EPG pairs *are* annotated chemical synapses -- that is what the EM reconstruction stores; gap junctions are not in it
  at all. So the thing without evidence is the x0.1, not the synapses: the `brain.py` comment's premise ("such populations are
  typically gap-junction coupled") is the claim that needs the literature. What would decide it: innexin (ShakB / Inx7)
  expression in E-PG and paired E-PG recordings for electrical coupling; and, in the model, a per-type `same_type_gain` field
  (default `None`, CPU bit-identity test) plus the suite at >= 3 draws -- which this batch does not justify, because the bump
  it buys fails the rate row 3x. Adoption would also need the compass room (`probe_compass_room.py`) to show the bump
  tracks anything -- the rate (151-158 Hz) already fails the
  `compass.EPG.bump_rate_hz` row, so it would be a known-gap swap, not a fix. And the F-family bump should not be brought back
  as a candidate until it is run in the room with a world: the rest-of-brain 0.04 Hz in this world-less protocol is no
  evidence at all about the runaway cliques the x0.1 was adopted against.
* **The cap and the monoamine slow class**: no candidate (section 1.3 b, d).
* **The ExR6 / ER6 / ER4m DC term**: the decomposition above makes the question decidable **in one arm, and that arm was not
  run.** With the ER/ExR term held at 0 onto PEN, u_PEN(driven) during the pulse is +9.7 - 0.6 = **+9.1 mV**, above the 7 mV
  gap (f ~ 30 Hz) -- so an `edges`-kind hold of ExR6 / ER6 / ER4m onto PEN and EPG (`INTERP.md` 10.1 step 5, the wiring row)
  is the arm that decides whether the DC balance is the whole story, and it turns "the DC inhibition keeps the relays below
  threshold" from a decomposition into a tested attribution. The data half of the question is: ExR6 is 2 cells, MaleCNS `nt`
  glutamate, sign -1, with no receptor-table row (tier fallback = NT_SIGN), and the -1 onto EPG comes from the E-PG row's
  GluClalpha (Davis 2020 PB_2, tier alias) -- so the sign is already from an expression profile, and what is open is ExR6's
  transmitter confidence and whether it acts ionotropically at all. **No published transmitter or function for ExR6 could be
  verified**: Hulse et al. 2021 (eLife 66039) defines ExR1-ExR8 morphologically and by connectivity, ExR2 is the dopaminergic
  PPM3 class, and the classical ring neurons (ER) are GABAergic inhibitors of E-PG (Omoto et al. 2017; Fisher et al. 2019;
  Kim et al. 2019) -- for ExR6 specifically, **unknown**; it is to be stated as unknown rather than as "the ring neurons
  inhibit EPG".

## Report

```yaml
summary: >
  Thread 5A asked whether a data-implied, type-level change to the shipped rules lets the correct-sign compass ring carry a
  bump at the shipped gains. Structure pass (CPU): the WEIGHT matrix has the k=1 structure a ring attractor needs (lambda_1
  +10,019 mV^2 shipped; PEN +8,327, Delta7 +1,719, PEG +178, Ring -206; lambda_0 -39,956), and a uniform-gain linearisation
  would be unstable in that mode at gamma > 1.69-2.00 Hz/mV -- but that is a NECESSARY CONDITION, NOT THE ATTRACTOR: the
  criterion is on diag(gamma_i) tau A at a realised state, and at the rate model's own fixed point every compass cell has
  gamma = 0 except Delta7 (3.6 background, 6.2 pulse), so the TRUE JACOBIAN IS STABLE IN EVERY MODE (leading eigenvalue
  +0.045 background, +0.079 pulse). The LIF slope is 2.3-3.2 Hz/mV at the 140-200 Hz the surviving bumps run at, not 5-8
  (5-8 holds only for u 9-16 mV, 31-70 Hz). The Delta7 cosine is intact and uncapped (0 of 479 Delta7->EPG edges reach 60);
  the ER/ExR feedback carries no k=1 content (-206, 2.5 % of PEN's +8,327) but is NOT purely k=0 (k=8 = -3,080). The relays
  stay far below the rate a bump needs -- measured PEN population mean 0.29-0.66 Hz during the pulse in S -- because the EPG
  drives 2 ExR6 + 4 ER6 harder than PEN (u_PEN -11 mV at background, -16 during the pulse); the per-type ExR6 / ER6 rates are
  rate-model outputs, overstated ~2x against this batch's own 308-cell ring population. GLNO=glutamate (MaleCNS T-bars,
  hemibrain name, BANC 4/4) is a structural null at these gains because GLNO fires only 0-3 Hz; sign+gain scales tuned and
  untuned terms alike; the monoamine slow class is < 0.2 mV; PEN fan-in scale is 1.00; lifting the cap doubles the ExR
  inhibition. Batch cx5-5cde5e (house B200, 8 jobs, 0 failed, 48/48 runs, 0 provenance problems): at gE 1 / gD 1 the shipped
  path, GLNO=glu, sign+gain and their pair score survival 0.00 in 4/4 (structural zero-SD null). Only the global instrument
  same_type_gain 1 carries a bump (F 151-156 Hz, width 3.0, 5.00 s in 4/4; the structural support is the undamped EPG->EPG
  k=1 coefficient +59.76 mV predicting 145 Hz, not a measured attribution), it dies under the correct GLNO sign (FG) and
  returns only with the receptor gain tier added (CFG 156-158 Hz). The structure pass missed the F bump through TWO ANALYSIS
  CHOICES -- AN OWNED DEFECT OF THIS THREAD'S OWN TOOL: the one-step EPG->EPG term dropped from the two-step reduction, and
  the forced drive entered as a rate rather than as a current so gamma_EPG = 0 by construction -- not through a
  fluctuation regime. NO ARM MEETS THE PREDECLARED RULE AND EVERY SURVIVING BUMP FAILS ON RATE; references reproduce
  cx_glno's `base` rows to the printed digit; the receptor stage is a bit-exact no-op on the ring core. Nothing adopted.
  Corrected throughout by an independent Opus skeptic pass (verdict MOSTLY SOUND, 14 corrections): the Answer stands, the
  argument around it is what changed.
skeptic:
  source: "independent Opus pass, 2026-09-15"
  verdict: "mostly sound"
key_claims:
  - "THE WEIGHT MATRIX HAS THE k=1 STRUCTURE, THE REALISED STATE HAS NO GAIN. Weight matrix: lambda_1 = +10,019 mV^2 (PEN +8,327, Delta7 +1,719, PEG +178, Ring -206), lambda_0 = -39,956; a uniform-gain linearisation is unstable in k=1 above gamma_crit(k1) 2.00 Hz/mV two-step / 1.69 full circuit. That is a necessary condition, not the attractor: the criterion is on diag(gamma_i) tau A with gamma_i = f'(u_i) at a state the system occupies, and at the structure pass's own fixed point gamma = 0 on EVERY compass cell except Delta7 (3.56 background, 6.21 pulse; c 7.78, f 0.18), so the true Jacobian's leading eigenvalue is +0.045 (background) / +0.079 (pulse) -- against +3.55 under the audit's uniform gamma 6 -- spectral radius 0.16-1.01, 7-40x below the instability threshold. The wiring could be a ring attractor's; under the shipped rules it never is one."
  - "LIF slope f'(u) = f^2 tau_m theta / (1000 u (u-theta)): 25.8 Hz/mV at u 7.1 (11 Hz), 9.13 at u 8 (22.8 Hz), 7.46 at u 9 (31.0), 6.76 at u 10 (38.1), 6.01 at u 12 (50.7), 5.35 at u 15 (67.7), 4.60 at u 20 (92.5), 3.59 at u 30 (133.1), 2.90 at u 40 (165.4), 2.39 at u 50 (191.7), 0 below threshold. The 5-8 band holds only for u 9-16 mV (31-70 Hz); at the 139-203 Hz the bumps in this batch run at it is 2.3-3.2 Hz/mV -- still above gamma_crit(k1) 1.69-2.00, but by a factor of ~1.4, not 3-5."
  - "Delta7 : ring-feedback ratio 0.072 at k=0 (-3,423 : -47,650), 0.033 per volley (-25.7 : -767 mV), 0.14 at the peak; at k=1 Delta7 +1,719 vs ring -206. The ER/ExR loop carries no k=1 content (2.5 % of PEN's +8,327) -- it is untuned in the bump mode -- but it is NOT a pure k=0 term: its spectrum is -47,650 (k=0), -3,080 (k=8, the L/R alternation, 6.5 % of k=0 and 15x the k=1 one), -206 (k=1), -73 (k=2), -39 (k=3), -51 (k=4)."
  - "rate-model fixed point at gE1/gD1: u_PEN(driven) -11.1 mV background, -15.8 during the pulse (EPG +9.7, Delta7 -0.6, ER/ExR -24.9: ExR6 -14.8 @ 183 Hz, ER6 -9.8 @ 116); no configuration reaches a bump. THE PER-TYPE RING RATES ARE RATE-MODEL OUTPUTS, NOT MEASURED: the same fixed point puts 2 ExR6 + 4 ER6 + 11 ER4m at a summed 484 Hz at background and 1,075 Hz during the pulse, while in the batch the LIF's entire 308-cell ER/ExR population sums to 244 Hz after release, 630 Hz at the end of the pulse and peaks at 800 Hz (arm S seed 0 means 0.807 / 2.044 / 2.599 / 0.792 Hz) -- the DC term is overstated by roughly 2x, and no arm of this batch reports a per-type ring rate. The conclusion rests on the measured PEN rate, not on these."
  - "the cap does not flatten the cosine: Delta7->EPG profile -44..-433 capped, -43..-432 uncapped, 0 of 479 pairs above the cap (synapse counts 1-44, mean 9.0); it clips ExR4/ExR6/GLNO edges (u_PEN -41 mV uncapped, ExR6 -14.8 -> -40.5)"
  - "THE RELAYS ARE NOT SILENT, THEY ARE TOO SLOW. Measured PEN population mean (42 cells) during the pulse: S 0.29-0.66 Hz (pre 0.00-0.024, post 0.020-0.062), G 0.25-0.50, C 4.43-6.36, CG 2.89-3.99. INTERP.md 10 defines silent as max rate < 0.5 Hz per cell and none of these arms meets it. The one data-implied change that MOVES the balance is the receptor tier: C raises PEN ~15x over S and GLNO from 1.4 to 15-22 Hz during the pulse, an order of magnitude, and still does not reach a bump. 'Nothing moves that balance' is 'nothing moves it far enough'."
  - "batch: S, G, C, CG survival 0.00 in 4/4; F 5.00 x4 at 151-156 Hz width 3.0; FG 0.08-0.15 s; CF 4.92-5.00 s but PEN 185-276 Hz and the bump at wedges 7-8; CFG 5.00 x4 at 156-158 Hz; R 201-203, RG 181-184, R175 139-167 (2 jumps), RG175 dead 4/4"
  - "NO ARM MEETS 'survival >= 5 AND width 2.5-5 AND rate 5-60 Hz in >= 3 of 4 seeds', AND EVERY SURVIVING BUMP FAILS ON RATE: no type-level ring mechanism; a bump only by removing a global hand rule, and then only with GLNO silent or with sign+gain added. Caveat on R175: bump_survival_s scores the last confined frame wherever the bump is, not at the driven tile -- seeds 1 and 3 read 5.00 s with the bump at centre 4.1 and only 3-4 of 11 driven cells above 22 Hz at 5 s (42/42 and 46/43 in/out Hz), cx_glno's 'not captured'; the tile-level count is 2/4 against the ledger's 4/4."
  - "G = S EXACTLY on the four primaries (0.00 / nan / nan / 0.00) and on the t5.0 EPG in/out row in 4/4 seeds -- NOT on every ring metric: 12-17 recorded secondaries differ per seed (GLNO_mean_post s1 0.1115 -> 0.1470 +32 %, s3 0.0881 -> 0.0618 -30 %; PEN_mean_during s1 0.6614 -> 0.5003 -24 %, s2 0.3370 -> 0.2766 -18 %; PEG_mean_post s3 0.1273 -> 0.1710 +34 %; epg_in_mean_post s2 10.7638 -> 10.7984). AND IT IS A STRUCTURAL NULL BY CONSTRUCTION, not a test of the relabel: measured GLNO fires 0.00 Hz before the pulse, 1.42-3.00 Hz during it and 0.081-0.335 Hz after (the 0.03-0.23 Hz figures are the rate model's, 10-20x lower), and at 3 Hz its -33 mV per PEN per volley is -0.5 mV on a PEN 7 mV below threshold. The sign is tested only where GLNO fires: F 87-91 Hz vs FG, R 115-137 Hz vs RG."
  - "THE STRUCTURE PASS'S MISS IS AN OWNED DEFECT OF THIS THREAD'S TOOL, NOT A FLUCTUATION REGIME. (i) The two-step reduction drops the one-step EPG->EPG term (M16['net'] = PEN + PEG + Delta7 + Ring; 'direct' is computed, printed and never used). Its k=1 ring-Fourier coefficient over 842 pairs: shipped (x0.1) +0.366 mV/pair, d_0 +6.77, d_1 +6.00, gamma_crit 33.4 Hz/mV (never reached, no bump); f undamped +3.653, d_0 +67.50, d_1 +59.76, gamma_crit 3.35 Hz/mV, predicted saturation 145 Hz against F's observed 151-156; cf undamped +4.164, d_0 +76.05, d_1 +66.51, 3.01 Hz/mV, predicted 161 Hz against CFG 156-158 and CF 142-150 -- 4-7 %. (ii) rate_fixed_point adds the forced background as a RATE, not a current (r = f(tau A r) + forced), so the driven EPG sits at u -22 to -68 mV while 'firing' at 10-50 Hz and gamma_EPG = 0 by construction. With the drive entered as a current (10 Hz <-> 6.63 mV, 50 Hz <-> 11.99 mV), the same sigma 2 mV and the one-step term kept, the EPG-only 16-wedge model gives 14.3 / 15.3 Hz background, 60.1 / 16.3 pulse, 14.3 / 15.3 after release (no bump) shipped, and 264 / 272 Hz throughout (runaway) undamped. Section 1.3's ranking is therefore computed at a zero-gain state and its miss on the F family shows it."
  - "F's attribution to the EPG's within-type synapses is INFERRED FROM THE STRUCTURE PASS, NOT MEASURED: same_type_gain is one global scalar, so no arm damps EPG->EPG while leaving PEN_a->PEN_a, PEN_b->PEN_b, Delta7->Delta7 and PEG->PEG undamped, and in F the PEN loop IS engaged (25-27 Hz during the pulse, 30 Hz after). The independent support is structural (the undamped k=1 coefficient +59.76 mV, gamma_crit 3.35, predicting 145 Hz)."
  - "THE FIRST SUBMISSION'S CANCELLATION: cx5-fd8b02 was cancelled 8/8; the client ran 3.7 min and reported '8 job(s), 8 failed' with FETCH FAILED, so no artefact was retrieved -- but its archived console client_console_fd8b02_cancelled.txt carries 21 of the 48 runs' result lines, 7 with a surviving bump, every one identical to the second batch's (F s3 5.00 s / 151.1 Hz / width 3.0; CFG s2 156.3, s3 156.4; CF s3 4.92 s / 144.4; R s3 5.00 s / 200.8 / 3.0), as determinism requires. They appear only inside the client's cancellation log dump, written at 05:43:46Z -- AFTER the resubmission stamp 05:42:48Z -- so the resubmission decision demonstrably preceded them; 'cancelled within two minutes and before any result was read' is wrong as written and predeclared.json submission_history's wording is superseded by section 2. No pre-amendment copy of predeclared.json was archived, so 'arms and primaries unchanged' is verifiable only for the ARMS: the 8 job lines recovered from the cancelled console are identical to batch.sh's once the exit expression is normalised."
  - "the receptor stage is a BIT-EXACT NO-OP on the ring core, stronger than 'no sign changes': over 57,261 nonzero core x core entries (EPG, PEN, PEG, Delta7, EPGt, ER/ExR, GLNO) max |A_receptor_off - A_shipped| = 0.000 -- 0 sign changes AND 0 weight changes"
  - "the references match cx_glno.md's `base` config (receptor model OFF) and its identical sign-class rows, NOT its sign-abs rows (201/10, 200/9, 203/9 at 2.0/15), although cx5's arms declare --receptor-model shipped = sign/abs -- consistent with the receptor stage being a bit-exact no-op on the ring core. R 201/10 (11,1), 201/9 (10,3), 204/9 (11,2); RG 180/10, 181/9, 184/9; RG175 10/13 (0,7), 12/15 (2,7), 8/11 (0,4); and R175 164/10 (10,1), 42/42 (3,9), 169/9 (9,2) reproduces cx_glno's base 1.75/15 too, one row more than the audit claimed."
  - "code identity per INTERP.md 10.2 is provenance.compiled_connectome.md5 + files.effective_weights_md5 + provenance.source_fingerprint: the fingerprint hashes 51 flyverse files per run, and after line-ending normalisation only flyverse/body.py and flyverse/senses.py differ from this tree (another thread's later edits) -- the cluster ran this working tree; tree_state HEAD = origin/main = 0b3668fc"
  - "cross-batch determinism: R / RG / RG175 seeds 0-2 reproduce cx_glno.md 3-4's B200 rows exactly"
files_written:
  - scripts/cx_ring_structure.py (new: structure pass, --plan-batch, --analyse)
  - scripts/cx_wedge.py (new flags only: --ledger, --lif, --arm, --block, --device, --pulse-s, --receptor-model shipped; simulate() gains keyword-only defaults)
  - tests/test_cx_ring_structure.py (5 tests, synthetic ring)
  - docs/audits/compass_ring_mechanism.md, docs/audits/compass_ring_mechanism_scatter.png
  - out/cx5/structure/{structure.json,structure.md,matrices.npz,evidence_glno.json}
  - out/cx5/{predeclared.json,tree_state.json,batch.sh,scheduler_receipt.json,scheduler_receipt_submitted.json,submit_stamp.txt,predeclare_stamp.txt,client_console.txt,client_console_fd8b02_cancelled.txt}
  - out/cx5/<arm>_s<seed>.{json,txt,npz} x 48; out/cx5/analysis/{analysis.md,runs.csv,compare.csv,decision.csv,analysis.json,scatter.png}; out/cx5/smoke/*; out/cx5_cluster.log
api:
  - "cx_wedge.simulate(..., lif_overrides=None, ledger=False, arm=None, block=None, device=None, ledger_npz=None): defaults reproduce the previous behaviour; ledger=True records EPG per 10 ms frame, scores probe_compass_room.bump_frames into row['metrics'] / row['ledger'] with common.provenance"
  - "cx_ring_structure: Circuit, fourier_modes, eig_modes, circuit_modes, offset_profile, rate_fixed_point, analyse, plan_batch, analyse_batch, BATCH_ARMS, PRIMARIES"
validation:
  - "tests/test_cx_ring_structure.py: 5 passed (CPU)"
  - "CPU smokes: cx_wedge --ledger on the shipped cache and with --nt-override GLNO=glutamate --lif same_type_gain=1 (out/cx5/smoke/smoke_{shipped,FG}.json); the default --sim path after the edit (smoke_default_path.json)"
  - "scheduler receipt 8 job(s), 0 failed; 48/48 json+txt+npz; 48/48 consoles 'device cuda'; analyse_batch problems 0; compiled-W md5 ef23cc27 (silent) / 7a10d93b (glutamate) on every run"
  - "code identity, all three parts of INTERP.md 10.2: provenance.compiled_connectome.md5, files.effective_weights_md5 and provenance.source_fingerprint -- 51 flyverse files hashed per run; after line-ending normalisation only flyverse/body.py and flyverse/senses.py differ from this tree (another thread's later edits)"
  - "first submission cx5-fd8b02 cancelled 8/8 (exit-expression expansion defect), client 3.7 min, '8 job(s), 8 failed' with FETCH FAILED, no artefact retrieved -- NOT 'before any result was read': the archived cancellation log dump carries 21 of the 48 result lines, written at 05:43:46Z, after the 05:42:48Z resubmission stamp; every one identical to the second batch's. predeclared.json submission_history is a stamped record and is left as written; its wording is superseded by section 2"
  - "independent Opus skeptic pass, 2026-09-15, verdict MOSTLY SOUND: its own re-implementation of bump_frames and of the survival / rate / width / confinement rules on all 48 .npz reproduces the shipped JSON metrics with max |diff| = 0 on all fourteen quantities; its own weight-shaping pipeline reproduces matrices.npz to max |diff| 7.5e-4 mV^2 on every M16 block of every configuration and every printed lambda_k, unitary weight and the Delta7-cap result exactly; the decision rule applied verbatim gives the same per-arm counts; the stamps order correctly and batch.sh's sha256 04f8fdb8...9d3e matches predeclared.json. 14 corrections applied to this document"
recommendations:
  - "DO NOT ADOPT ANYTHING FROM THIS THREAD. The shipped-gain question is closed as a NULL, not as a result: every gE 1 / gD 1 arm is a structural zero-SD comparison against a reference that is also zero, read as magnitudes, and the two arms that carry a bump are a labelled global INSTRUMENT (same_type_gain 1) and a labelled reference (gE 2 / gD 15) -- neither a mechanism the data imply, both failing compass.EPG.bump_rate_hz 3x, so compass.EPG.bump_survival_s stays FAIL and the rate and width rows stay NOT_APPLICABLE. What is established on the type level: the WEIGHT matrix has the k=1 structure, the realised state has no gain, and the DC inhibition through ExR6 / ER6 / ER4m is what keeps the relays below threshold -- no data-implied per-type change moves that balance FAR ENOUGH (the receptor tier moves it ~15x and still falls short)."
  - "WHAT THE COMPASS NOW NEEDS, in this order. (1) FIX THE STRUCTURE TOOL -- forced drive as a CURRENT, the one-step EPG->EPG term KEPT, per-cell gamma at the realised fixed point -- and re-derive the ranking, because the present ranking is computed at a state with zero gain everywhere and its miss on the F family shows it. (2) RUN THE ONE COUNTERFACTUAL ARM the decomposition names and that this batch did not run: an `edges`-kind hold of ExR6 / ER6 / ER4m at 0 onto PEN and EPG (INTERP.md 10.1 step 5, the wiring row), which the fixed point predicts puts u_PEN(driven) at +9.7 - 0.6 = +9.1 mV during the pulse, above the 7 mV gap, f ~ 30 Hz -- the only thing that turns 'the DC inhibition keeps the relays below threshold' from a decomposition into a tested attribution."
  - "The data question alongside it is a RELABEL WITH SOURCES, not a gain: ExR6's transmitter confidence and whether it has any fast receptor on E-PG at all. ExR6 is 2 cells, MaleCNS nt glutamate, sign -1, with NO receptor-table row (tier fallback = NT_SIGN), and the -1 onto EPG rides on the E-PG row's GluClalpha profile (Davis 2020 PB_2, tier alias). No published transmitter or function for ExR6 could be verified: Hulse et al. 2021 (eLife 66039) defines ExR1-ExR8 morphologically and by connectivity, ExR2 is the dopaminergic PPM3 class, and the classical ring neurons (ER) are GABAergic inhibitors of E-PG (Omoto et al. 2017; Fisher et al. 2019; Kim et al. 2019) -- for ExR6 specifically, UNKNOWN. Report it as `unknown`, not as 'the ring neurons inhibit EPG'."
  - "The GLNO relabel decision belongs with glno_relabel.md, and this thread's contribution should be handed over as exactly what it is: a NULL at these gains -- G = S on the primaries because GLNO fires 0-3 Hz, so A SILENT NEURON'S SIGN IS UNTESTED, NOT CONFIRMED HARMLESS -- plus the costs where GLNO does fire: the gE 1.75 point (RG175 0/4) and the F-family bump (FG), recovered only with sign+gain (CFG)."
  - "A per-type same-type damping would be a new LIFParams field (default None, CPU bit-identity test) and a suite run; it is not justified by these results because the bump it buys fails the rate row 3x. And the data argument runs the other way from how this audit first put it: the connectome's 842 EPG->EPG pairs ARE annotated chemical synapses (gap junctions are not in the EM reconstruction at all), so what lacks evidence is the x0.1, not the synapses."
  - "Do not bring the F-family bump back as a candidate until it is run in the room with a world: at 151-158 Hz it is a KNOWN GAP swap, and the rest-of-brain 0.04 Hz in this world-less protocol is no evidence at all about the runaway cliques the x0.1 was adopted against."
open_questions:
  - "FIX THE STRUCTURE PASS (the F-family miss is explained, not open): put the forced drive in as a CURRENT rather than a rate, and KEEP the one-step EPG->EPG term in the reduction. Both are defects of scripts/cx_ring_structure.py, not of the mean field -- the same deterministic model with the same sigma 2 mV separates F from S once they are undone (shipped returns to 14.3 / 15.3 Hz, undamped runs away to 264 / 272 Hz), and the undamped k=1 coefficient +59.76 mV predicts 145 Hz against F's 151-156 and 161 Hz against CFG's 156-158. A diffusion approximation is NOT what is needed."
  - "Does holding ExR6 / ER6 / ER4m at 0 onto PEN and EPG produce the bump the fixed point predicts (u_PEN +9.1 mV, f ~ 30 Hz during the pulse)? This is the untested attribution at the centre of the Answer, and it is one `edges`-kind arm."
  - "What is ExR6's transmitter and does it act ionotropically on E-PG at all? No published transmitter or function could be verified for ExR6 specifically -- UNKNOWN, to be reported as such. The parallel E-PG -> E-PG question is not 'are the synapses chemical' (they are, 842 annotated pairs) but whether E-PG are ALSO electrically coupled: innexin (ShakB / Inx7) expression in E-PG and paired E-PG recordings would decide it, and that is what the x0.1's premise needs."
  - "The F-family bump in the room (senses, walking body) and its effect on the rest of the brain with the damping off globally -- not run here (no world in this protocol; rest of brain 0.04 Hz, which is no evidence either way)."
```

## Skeptic pass (independent, 2026-09-15)

Own code, CPU only, nothing committed, nothing under `docs/` `scripts/` `flyverse/` `tests/` touched. Scratch scripts:
`redo_ledger.py` (own re-implementation of `probe_compass_room.bump_frames` + the survival / rate / width / confinement
rules, applied to the 48 `.npz`), `redo_structure.py` (own shaping pipeline from `c.W` -- receptor stage, cap 60, path
gains, same-type damping, fan-in, w_syn -- **not** `brain._shaped_weights` / `cx_wedge.effective_weights`),
`redo_gain.py` (per-cell LIF slope at the structure pass's own fixed point, the true Jacobian, the dropped one-step
term), `redo_current.py` (EPG-only 16-wedge model with the forced drive as a current), `redo_refs.py` (references,
G vs S, GLNO, ring-population rates). No house rerun was needed: every claim was checkable by recomputation.

---

### Refuted

**R1. "The LIF slope is 5-8 Hz/mV anywhere in its firing range" (section 1.2; Report `summary`, `key_claims`).**
Computed analytically from `brain.LIFParams()` (theta 7, tau_m 20, t_ref 2.2), f'(u) = f(u)^2 tau_m theta / (1000 u (u-theta)):

| u (mV) | 7.1 | 8 | 9 | 10 | 12 | 15 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| f (Hz) | 11.4 | 22.8 | 31.0 | 38.1 | 50.7 | 67.7 | 92.5 | 133.1 | 165.4 | 191.7 |
| f'(u) (Hz/mV) | **25.8** | 9.13 | 7.46 | 6.76 | 6.01 | 5.35 | 4.60 | 3.59 | **2.90** | **2.39** |

The 5-8 band holds only for u in roughly 9-16 mV (31-70 Hz). The audit's "5-8" is the set of secant slopes between its
own four sample points (f(8)->f(9) 8.2, f(9)->f(12) 6.6, f(12)->f(20) 5.2), not the derivative. Over the rates the
bumps in this batch actually run at (139-203 Hz) the slope is **2.3-3.2 Hz/mV**, not 5-8. `gamma_crit(k1)` 1.69-2.00
is still below 2.3, so the sign of the comparison survives -- but by a factor of ~1.4, not the factor of 3-5 the
wording implies. (This number appears in the Report and is in no named file.)

**R2. "The k=1 bump mode exists in every configuration ... the wiring is a ring attractor's under the shipped rules"
(section 1.2; Report `summary` and `key_claims` line 1).** A positive k=1 eigenvalue of the *weight* matrix is not the
attractor criterion; the criterion is on `diag(gamma_i) tau A` with `gamma_i = f'(u_i)` at a state the system occupies.
At the structure pass's **own** fixed point (`structure.json` `rate_model`) I get:

| config / state | u_EPG(driven) | gamma_EPG | u_PEN(driven) | gamma_PEN | gamma_Delta7 | leading eig of `diag(gamma) tau A` | same with the audit's uniform gamma 6 |
|---|---|---|---|---|---|---|---|
| shipped, background | -22.0 | **0.00** | -11.1 | **0.00** | 3.56 | **+0.045** | +3.55 |
| shipped, pulse | -45.0 | **0.00** | -15.8 | **0.00** | 6.21 | **+0.079** | +3.55 |
| c (sign+gain), pulse | -68.0 | 0.00 | -17.6 | 0.00 | 7.78 | +0.023 | +4.45 |
| f (damping off), pulse | -21.6 | 0.00 | -15.6 | 0.00 | 0.18 | +0.149 | +4.39 |

Every compass cell in the rate model has gamma = 0 except Delta7 (the inhibitory one, 3.6-7.8). The linearisation at
the model's actual operating point is **stable in every mode**, spectral radius 0.16-1.01, leading real part 0.023-0.15
-- 7-40x below the instability threshold. The +10,019 / gamma_crit 1.69-2.00 numbers describe a uniform-gain
linearisation at a state that never exists in this model, and the "vs an LIF slope of 5-8" comparison is against a
slope that **no cell in the loop has** (the only cell with a slope in that band is the inhibition). The audit does
qualify this in the body ("strongly unstable-from-uniform *as soon as the relays fire*"); the Answer, the summary and
`key_claims` state it flatly and are what a reader will carry away.

**R3. "The rate-model structure pass missed the F-family bump: its 2 mV noise term understates the fluctuations ...
a fluctuation-driven regime the fixed-2-mV noise term of `cx_wedge.lif_fi` does not model" (section 3.3; Report
`open_questions` 1).** Refuted as the explanation. The same **deterministic** mean-field with the **same** sigma = 2 mV
separates F from S, if two analysis choices are undone.

*(a) The two-step reduction drops the one-step EPG -> EPG term.* `M16['net'] = PEN + PEG + Delta7 + Ring`; `direct` is
computed, printed in mV in the same table row as the mV^2 lambdas, and then never used. Its ring-Fourier coefficients
(my own rebuild, matching `matrices.npz` to 2e-7):

| configuration | pairs | mV/pair | d_0 | d_1 | gamma_crit(k1) = 1/(tau d_1) | predicted saturation rate where f'(u) = gamma_crit | observed bump |
|---|---|---|---|---|---|---|---|
| shipped (x0.1) | 842 | +0.366 | +6.77 | **+6.00** | **33.4 Hz/mV** (never reached) | -- | none |
| f: undamped | 842 | +3.653 | +67.50 | **+59.76** | **3.35 Hz/mV** | **145 Hz** | F 151-156 Hz |
| cf: sign+gain, undamped | 842 | +4.164 | +76.05 | **+66.51** | **3.01 Hz/mV** | **161 Hz** | CFG 156-158, CF 142-150 |

The undamped EPG -> EPG ring is supercritical on its own at any operating point below ~34 mV, and the rate at which
its gain falls back to 1 predicts the observed bump to 4-7 %. Nothing about fluctuations is needed.

*(b) The forced background enters `rate_fixed_point` as an added **rate**, not as a current* (`r = f(tau A r) + forced`),
so the driven EPG sits at u = -22 to -68 mV while "firing" at 10-50 Hz, and gamma_EPG = 0 **by construction** -- no
recurrent EPG term, direct or two-step, can ever engage in that model. Put the same drive in as a current (10 Hz <->
u 6.63 mV, 50 Hz <-> u 11.99 mV) in an EPG-only 16-wedge model, same sigma, no noise change:

| | background | pulse | after release |
|---|---|---|---|
| shipped (x0.1) | 14.3 / 15.3 | 60.1 / 16.3 | **14.3 / 15.3 Hz** (no bump) |
| f: undamped | 264 / 272 | 273 / 272 | **264 / 272 Hz** (runaway) |

So: the structure pass missed F because of a reduction that dropped the term and a drive that zeroed the gain -- an
analysis choice, not a fluctuation regime. The correct `open_question` is not "a diffusion approximation would be the
next tool"; it is "put the forced drive in as a current and keep the one-step term".

**R4. "G = S to the printed digit on every ring metric" (section 0, 3.1, 4; Report `key_claims`).** Not exact.
Per seed, 12-17 of the recorded metrics differ. The largest relative differences: `GLNO_mean_post` s1 0.1115 -> 0.1470
(+32 %), s3 0.0881 -> 0.0618 (-30 %); `PEN_mean_during` s1 0.6614 -> 0.5003 (-24 %), s2 0.3370 -> 0.2766 (-18 %);
`PEG_mean_post` s3 0.1273 -> 0.1710 (+34 %); `epg_in_mean_post` s2 10.7638 -> 10.7984. What *is* exactly equal in 4/4
seeds: all four primaries (0.00 / nan / nan / 0.00) and the t5.0 EPG in/out row.
**More important: the batch does not test the GLNO sign at the shipped gains in any informative sense.** Measured GLNO
rate in S: 0.000 Hz before the pulse, **1.42-3.00 Hz during it**, 0.081-0.335 Hz after. At 3 Hz its -33 mV per PEN per
volley contributes tau x 3 x (-33) = -0.5 mV to a PEN sitting 7 mV below threshold. G vs S is a null by construction,
not a finding about the relabel. (The sign *is* tested where GLNO fires: F vs FG at 87-91 Hz and R vs RG at 115-137 Hz;
those are the informative comparisons and the audit reads them correctly.) Also, "GLNO fires 0.1 Hz while PEN is
silent" (section 0) understates the pulse value 10-20x -- the 0.03-0.23 Hz figures are the *rate model's*, the LIF's
are 1.2-3.0 Hz.

**R5. "The relays never fire" / "PEN is silent" / "no data-implied per-type change moves that balance"
(section 0, 1.2; Report `recommendations` 1).** Measured PEN population mean (42 cells), from the shipped `.npz`:

| arm | pre | during pulse | post |
|---|---|---|---|
| S | 0.00-0.024 | **0.29-0.66** | 0.020-0.062 |
| G | 0.00-0.024 | 0.25-0.50 | 0.025-0.076 |
| **C** (sign+gain) | 0.061-0.135 | **4.43-6.36** | 0.31-0.72 |
| CG | 0.061-0.135 | 2.89-3.99 | 0.25-0.46 |

The project's own definition of silent is `silent_frac` = max rate < 0.5 Hz per cell (`INTERP.md` 10 / health). A
population mean of 4.4-6.4 Hz does not meet it, and neither does S's 0.66 Hz mean over 42 cells. The **data-implied**
tier C raises the PEN relay ~15x over S (and GLNO from 1.4 to 15-22 Hz during the pulse) -- it moves the DC balance by
an order of magnitude and still does not reach a bump. "Nothing moves that balance" should be "nothing moves it far
enough".

**R6. "ExR6 -14.8 mV at 183 Hz, ER6 -9.8 at 116 Hz" presented unqualified in the Report `summary` / `key_claims`.**
These are rate-model outputs the batch's own data contradict. `structure.json`'s shipped fixed point puts 2 ExR6 +
4 ER6 + 11 ER4m at a summed **484 Hz** at background and **1,075 Hz** during the pulse. In the same protocol the LIF's
*entire* 308-cell ER/ExR population sums to:

| arm S, seed 0 | pre | end of pulse | peak during pulse | post |
|---|---|---|---|---|
| Ring mean (308 cells) | 0.807 Hz | 2.044 | 2.599 | 0.792 |
| **summed over 308 cells** | **249 Hz** | **630 Hz** | **800 Hz** | **244 Hz** |

So the rate model's 6-17 cells alone exceed what the whole ring population does in the simulation the audit ran: the
DC term is overstated by ~2x at background. (The *conclusion* survives independently -- PEN is measured at 0.29-0.66 Hz
-- but no arm of this batch reports a per-type ring rate, and "183 Hz" is not a measured quantity of this model, let
alone of the animal.)

**R7. "The first submission (cx5-fd8b02) was cancelled 8/8 within two minutes and before any result was read"
(section 2; `predeclared.json` `submission_history`; Report `validation` line 4).** False as written. The archived
console `out/cx5/client_console_fd8b02_cancelled.txt` carries **21 of 48 completed run results**, 7 of them with a
surviving bump, every one identical to the second batch's:

| from the cancelled console | final batch |
|---|---|
| F s3 `survival 5.00 s, bump 151.1 Hz, width 3.0` | 5.00 / 151.083 / 3.0 |
| CFG s2 `156.3` / CFG s3 `156.4` | 156.335 / 156.365 |
| CF s3 `survival 4.92 s, bump 144.4 Hz, width 2.1` | 4.92 / 144.436 / 2.093 |
| R s3 `5.00 s, 200.8 Hz, 3.0` | 5.00 / 200.838 / 3.0 |

The client ran **3.7 min**, not two, and reported `8 job(s), 8 failed`; `FETCH FAILED`, so no artefact was retrieved.
What *is* verifiable and should be the claim instead: the results appear only inside the client's cancellation log
dump, and that file was written at **05:43:46Z -- after the resubmission stamp 05:42:48Z**. So the resubmission
decision demonstrably preceded the results; "no result was read" does not, and 21 of them are in the repo.

**R8. "The ER/ExR feedback is a pure k = 0 term" (section 1.2; Report `summary`).** Ring two-step lambda_k =
**-47,650** (k=0), -206 (k=1), -73 (k=2), -39 (k=3), -51 (k=4), **-3,080 (k=8)**. The k=8 (L/R alternation) component
is 6.5 % of k=0 and 15x the k=1 one. "Carries no k=1 content (2.5 % of PEN's)" is right; "a pure k=0 term" is not.

---

### Confirmed (recomputed independently)

**Wedge matrices and modes.** My own shaping (own receptor stage via `receptor_signs`, own cap, own path gains, own
same-type damping, own fan-in, w_syn 0.275) reproduces `matrices.npz` to max |diff| 7.5e-4 mV^2 on every M16 block of
every configuration, and the printed lambdas exactly:

| | PEN | PEG | Delta7 | Ring | net | direct (mV) | gamma_crit(k1) |
|---|---|---|---|---|---|---|---|
| lambda_1 shipped | +8,327 | +178 | +1,719 | -206 | **+10,019** | +6.00 | **1.998** |
| lambda_0 shipped | +10,687 | +430 | -3,423 | -47,650 | **-39,956** | +6.77 | |
| lambda_2 / lambda_8 net | | | | | +3,923 / -2,883 | | |

Exact eigenvalues of the 16x16 net: 10,415 / 9,561 (k=1), 4,147 / 3,660 (k=2) -- exact match. c: +14,977 / 1.634;
b: +11,560 / 1.860; f: +9,985 / 2.002 -- all match. Full-circuit leading k=1 mu +118.4 (gamma_crit 1.69) and k=0
+40.0 (5.0) -- match. The two-step algebra gamma_crit = 1/(tau sqrt(lambda_1)) is correct. Ring k=1 / PEN k=1 =
206/8,327 = **2.5 %** -- confirmed.

**The Delta7 cosine and the cap.** Delta7 -> EPG: **479 pairs, synapse counts 1-44 (mean 9.0), 0 above the cap 60**.
Capped -2.464 vs uncapped -2.458 mV per pair; volley -25.66 vs -25.60; the 16-wedge distance profile is identical to
0.1 mV (-10.7 / -6.3 / -1.0 / -0.2 / ~0 at 0-4 wedges, both). **The cap does not flatten the Delta7 cosine.** Confirmed
exactly. Confirmed too: lifting the cap doubles the ring term on PEN (`PEN <- Ring` -24.85 -> -50.15 mV during the
pulse, ExR6 -14.8 -> -40.5).

**The receptor stage is a complete no-op on the ring core.** Over 57,261 nonzero core x core entries (EPG, PEN, PEG,
Delta7, EPGt, ER/ExR, GLNO), `max |A_receptor_off - A_shipped| = 0.000` -- not merely 0 sign changes, 0 weight changes.
Stronger than the audit's claim.

**Unitary weights.** EPG -> ExR6 +11.438 mV/pair (92 pairs, 2 cells), EPG -> PEN +5.054 (664 pairs), EPG -> ER6 +7.628
(164), EPG -> ER4m +2.448 (504); ExR6 -> PEN -8.659 (volley -16.91), ExR6 -> EPG -15.010 (-30.02), ER6 -> PEN -4.617
(-15.17), ER4m -> EPG -11.393 (-125.32), ExR4 -> PEN -15.217 (-30.43). Every audit digit reproduces. 2 ExR6 + 4 ER6 +
11 ER4m confirmed.

**The batch, exactly.** My own re-implementation of `bump_frames` and of the survival / rate / width / confinement
rules, run on all 48 `.npz`, reproduces the shipped JSON metrics with **max |diff| = 0** on all fourteen quantities
(`survival_s`, `bump_hz_post`, `width_half_post`, `frac_confined_post`, `frac_confined_pre`, EPG in/out post and
during, PEN/Delta7/Ring/GLNO/rest means). Per-arm decision rule applied verbatim:

| arm | survive >= 5 s | meets survival AND width 2.5-5 AND rate 5-60 | rate | width |
|---|---|---|---|---|
| S / G / C / CG | 0/4 each | 0/4 | -- | -- |
| **F** | **4/4** | 0/4 | 150.9-155.8 | 3.00 x4 |
| FG | 0/4 (0.08-0.15 s) | 0/4 | (90.6-124.0) | (2.0-3.0) |
| CF | 2/4 (5.00, 4.96, 5.00, 4.92) | 0/4 | 142.3-149.5 | 2.04-2.91 |
| **CFG** | **4/4** | 0/4 | 156.3-158.2 | 3.00 x4 |
| R / RG / R175 | 4/4 each | 0/4 | 200.8-203.2 / 181.4-183.7 / 138.5-167.2 | 3.00-3.06 / 2.64-2.84 / 2.12-2.84 |
| RG175 | 0/4 (0.25-3.29 s) | 0/4 | (134.1-145.2) | (2.0) |

**No arm meets the predeclared rule; every surviving arm fails on rate.** Confirmed. F's confinement 0.824-0.866,
PEN 29.6-30.9 post, Delta7 37.4-38.8, GLNO 87.2-91.4, Ring 5.67-5.82, rest 0.038-0.042 -- all confirmed.

**The references reproduce cx_glno.md exactly, and one more than claimed.** cx5 at 5 s vs `cx_glno.md` section 3:

| | cx5 (this batch) | cx_glno.md |
|---|---|---|
| R s0/s1/s2 | 201/10 (11,1), 201/9 (10,3), 204/9 (11,2) | base 2.0/15: identical |
| RG s0/s1/s2 | 180/10 (10,1), 181/9 (10,3), 184/9 (10,2) | glu 2.0/15: identical |
| RG175 s0/s1/s2 | 10/13 (0,7), 12/15 (2,7), 8/11 (0,4) | glu 1.75/15: identical |
| R175 s0/s1/s2 | 164/10 (10,1), **42/42 (3,9)**, 169/9 (9,2) | base 1.75/15: identical (not claimed by the audit) |

One clarification the audit owes: the rows it reproduces are cx_glno's **`base`** config (receptor model *off*) and
its identical `sign-class` rows -- **not** its `sign-abs` rows (201/10, 200/9, 203/9 at 2.0/15), although cx5's arms
declare `--receptor-model shipped` = sign/abs. That is consistent with my finding that the receptor stage is a bit-exact
no-op on the ring core, but the audit should name which config it matched.

**Stamps, submissions, reducer identity.** predeclare 05:39:53Z < submit#1 05:40:05Z < `batch.sh` regenerated 05:42:14Z
< `predeclared.json` amended 05:42:48Z == submit#2 05:42:48Z < scheduler `submitted_at` 05:42:50.17Z < analysis
05:49:10Z < audit 05:55:31Z. `batch.sh` sha256 `04f8fdb8...9d3e` = the value recorded in `predeclared.json`. The
**arms did not change between the two submissions**: the 8 job lines recovered from the cancelled console are identical
to `batch.sh`'s 8 (arms, flags, seeds, outputs) once the exit expression and `$?` are normalised; the defect itself is
visible verbatim (`exit $(( |  |  |  |  | ))`). Reducer identity: `analysis.json`'s `analysis_sha256` / `cx_wedge_sha256`
/ `common_sha256` = `tree_state.json`'s = the current files' (`b484fec7` / `e77a8daf` / `f61045a3`); `n_runs` 48,
`problems` [], `txt_missing` []. `source_fingerprint` (51 flyverse files): after CRLF->LF normalisation only
`flyverse/body.py` and `flyverse/senses.py` differ from the current tree (the other thread's later edits) -- the cluster
ran this working tree. `tree_state` HEAD = origin/main = 0b3668fc.

**Script hygiene.** `scripts/cx_wedge.py` diff vs HEAD is +136 / -22: new flags only (`--ledger`, `--ledger-npz`,
`--lif`, `--arm`, `--block`, `--device`, `--pulse-s`, and a `shipped` choice added to `--receptor-model`); every
default unchanged; the non-ledger branch is the previous code verbatim; `simulate()`'s new arguments all default to
the previous behaviour. The only structural change is that `main()` now calls `simulate` once per gain instead of
passing the list -- behaviourally equivalent (FlyBrain is rebuilt per gain inside `simulate` either way).
`tests/test_cx_ring_structure.py`: **5 passed** here on the CPU. Per-seed scatter present (`analysis/scatter.png`,
4 panels x 12 arms, per-seed points with a mean bar) and copied to `docs/`.

**The conclusion itself.** Independently: at gE 1 / gD 1 no arm in this batch produces a bump that meets the rule; the
only arms that hold a bump are the ones with the global same-type damping removed or the experiment gains applied; the
DC inhibition onto the relays is the binding constraint (the driven PEN's input decomposes as +4.2 EPG, -1.8 Delta7,
-13.5 Ring at background and +9.7 / -0.6 / -24.9 during the pulse, against a 7 mV gap), and the ring feedback carries
essentially no k=1. **The Answer of the audit stands.** What does not stand is a good deal of how it is argued.

---

### Corrections (exact replacement text)

1. **Section 1.2, the LIF slope sentence.** Replace
   "The LIF slope is 5-8 Hz/mV anywhere in its firing range (f(8) 22.8, f(9) 31.0, f(12) 50.7, f(20) 92.6 Hz) and 0 below threshold."
   with
   > "The LIF slope f'(u) = f^2 tau_m theta / (1000 u (u - theta)) falls monotonically across the firing range: 25.8 Hz/mV at u 7.1 (11 Hz), 9.1 at u 8 (23 Hz), 6.0 at u 12 (51 Hz), 4.6 at u 20 (93 Hz), 2.9 at u 40 (165 Hz), 2.4 at u 50 (192 Hz), and 0 below threshold. It is in the 5-8 band only for u 9-16 mV (31-70 Hz); at the 140-200 Hz the surviving bumps run at it is 2.3-3.2 Hz/mV, still above gamma_crit(k1) 1.69-2.00 but by a factor of 1.4, not 3."

2. **Section 0 and section 1.2, the ring-attractor sentence; Report `summary` and `key_claims` line 1.** Replace
   "The bump mode exists and is strongly unstable-from-uniform as soon as the relays fire; the wiring is a ring attractor's under the shipped rules"
   and the Report's "k=1 ring mode present at the shipped gains: lambda_1 = +10,019 mV^2 ... gamma_crit(k1) 2.00 Hz/mV two-step, 1.69 full circuit"
   with
   > "The WEIGHT matrix has the k=1 structure a ring attractor needs (lambda_1 = +10,019 mV^2; PEN +8,327, Delta7 +1,719, PEG +178, Ring -206; lambda_0 = -39,956), and a uniform-gain linearisation would be unstable in that mode at gamma > 1.69-2.00 Hz/mV. That is a necessary condition, not the attractor: the criterion is on diag(gamma_i) tau A with gamma_i = f'(u_i) at a realised state, and at the rate model's own fixed point every compass cell has gamma = 0 except Delta7 (3.6 at background, 6.2 during the pulse), so the true Jacobian's leading eigenvalue is +0.045 (background) / +0.079 (pulse), i.e. stable in every mode. The wiring could be a ring attractor's; under the shipped rules it never is one, because the gain is never there."

3. **Section 3.3 "The structure pass's miss"; Report `open_questions` 1.** Replace the fluctuation explanation with
   > "The structure pass's miss is an analysis choice, not a fluctuation regime. (i) The EPG-only reduction keeps only the two-step terms; the one-step EPG -> EPG matrix is computed and then dropped. Its k=1 ring-Fourier coefficient is +6.00 mV damped (gamma_crit = 1/(tau d_1) = 33.4 Hz/mV, never reached) and +59.76 mV undamped (3.35 Hz/mV, reached by the LIF at any u below ~34 mV); the rate at which the gain falls back to 1 is 145 Hz for f and 161 Hz for cf, against the observed 151-156 (F) and 156-158 (CFG). (ii) `rate_fixed_point` adds the forced background as a rate rather than as a current (r = f(tau A r) + forced), so the driven EPG sits at u -22 to -68 mV while 'firing' at 10-50 Hz and gamma_EPG = 0 by construction -- no recurrent EPG term can engage. With the same deterministic model, the same sigma 2 mV, the drive entered as a current (10 Hz <-> 6.63 mV, 50 Hz <-> 11.99 mV) and the one-step term kept, the EPG-only 16-wedge model returns to 14 Hz under the shipped damping and runs away to 264 Hz undamped. The open question is not a diffusion approximation; it is to fix these two."

4. **Section 0, 1.2, 4 and Report `recommendations` 1, the word "silent".** Replace "the relays never fire" / "PEN is
   silent" / "no data-implied per-type change moves that balance" with
   > "The relays stay far below the rate a bump needs: the measured PEN population mean is 0.29-0.66 Hz during the pulse in S (0.02-0.06 after). The one data-implied change that does move the balance is the receptor tier: C raises PEN to 4.4-6.4 Hz during the pulse and GLNO to 15-22 Hz, a ~15x change that still does not close the loop. No data-implied per-type change moves it far enough."
   (`INTERP.md` 10 defines `silent` as max rate < 0.5 Hz per cell; none of these arms meets it.)

5. **Section 1.2 table, Report `summary` and `key_claims` line 3, the ExR6 / ER6 rates.** Add the qualifier:
   > "(rate model; not measured. The same fixed point puts 2 ExR6 + 4 ER6 + 11 ER4m at a summed 484 Hz at background and 1,075 Hz during the pulse, while in the batch the LIF's entire 308-cell ER/ExR population sums to 244 Hz after release, 630 Hz at the end of the pulse and peaks at 800 Hz -- so the DC term is overstated by roughly 2x and the per-type ring rates are not quantities this batch measured. The conclusion rests on the measured PEN rate, not on these.)"

6. **Section 2 "Submissions"; `predeclared.json` `submission_history`; Report `validation` line 4.** Replace
   "was cancelled 8/8 within two minutes and before any result was read"
   with
   > "was cancelled 8/8; the client ran 3.7 min and reported `8 job(s), 8 failed` with `FETCH FAILED`, so no artefact of it was retrieved. Its archived console (`client_console_fd8b02_cancelled.txt`) does carry 21 of the 48 runs' result lines -- they appear only inside the client's cancellation log dump, which was written at 05:43:46Z, after the resubmission was stamped at 05:42:48Z, so the resubmission decision preceded them. Every one of those 21 lines is identical to the second batch's (F s3 151.1 / 5.00 s, CFG s3 156.4, CF s3 144.4 / 4.92 s, R s3 200.8), as determinism on this protocol requires. No pre-amendment copy of `predeclared.json` was archived, so 'arms and primaries unchanged' is verifiable only for the arms: the 8 job lines recovered from the cancelled console are identical to `batch.sh`'s once the exit expression is normalised. Archive the pre-amendment predeclaration next time."

7. **Section 1.2 and Report `summary`, "a pure k=0 term".** Replace with
   > "the ER/ExR loop carries no k=1 content (-206, 2.5 % of PEN's +8,327); it is untuned in the bump mode. It is not purely k=0: its spectrum is -47,650 (k=0), -3,080 (k=8, the L/R alternation), -206 (k=1)."

8. **Section 0 and 1.3(a), "GLNO fires 0.1 Hz while PEN is silent".** Replace with
   > "GLNO fires 0.00 Hz before the pulse, 1.4-3.0 Hz during it and 0.06-0.39 Hz after (the 0.03-0.23 Hz figures in the rate-model column are the rate model's). At 3 Hz its -33 mV per PEN per volley is -0.5 mV on a PEN 7 mV below threshold, so G vs S at the shipped gains is a structural null by construction rather than a test of the relabel: this batch tests GLNO's sign only where GLNO fires (F 87-91 Hz vs FG; R 130-137 Hz vs RG)."

9. **Section 3.3 F attribution.** Add after "The bump is carried by the EPG's within-type synapses, not by the PEN loop":
   > "This attribution is inferred from the structure pass, not measured: `same_type_gain` is one global scalar, so no arm damps EPG -> EPG while leaving PEN_a -> PEN_a, PEN_b -> PEN_b, Delta7 -> Delta7 and PEG -> PEG undamped, and in F the PEN loop is engaged (PEN 25-27 Hz during the pulse, 30 Hz after). The independent support is structural: the undamped EPG -> EPG 16-wedge k=1 coefficient +59.76 mV is supercritical on its own (gamma_crit 3.35 Hz/mV) and predicts a saturation at 145 Hz against the observed 151-156."

10. **Section 3.2 / the R175 row.** Add: > "`bump_survival_s` scores the last confined frame wherever the bump is, not
    at the driven tile: R175 seeds 1 and 3 read 5.00 s with the bump at centre 4.1 and only 3-4 of 11 driven cells
    above 22 Hz at 5 s (42/42 and 46/43 in/out Hz), which is cx_glno's 'not captured'. The tile-level count is 2/4,
    the ledger count 4/4; `frac_confined_post` (0.746-0.868) and the centre are what carry the difference."

11. **Section 1.1 table.** The `lambda_1` row mixes units: `direct EPG->EPG (mV)` +6 is a one-step mV coefficient and
    is not part of `net` +10,019 = 8,327 + 178 + 1,719 - 206 (mV^2). Mark the column as "not summed into net (mV, one
    step)" -- it is the term R3 shows should have been used separately, not added.

12. **Report `key_claims` / `validation`.** `INTERP.md` 10.2 asks for `provenance.compiled_connectome.md5` **+
    `files.effective_weights_md5` + `provenance.source_fingerprint`** as the code identity. The audit quotes only the
    first. Add: > "`provenance.source_fingerprint`: 51 flyverse files hashed per run; after line-ending normalisation
    only `flyverse/body.py` and `flyverse/senses.py` differ from this tree (another thread's later edits)."

13. **Section 4 / `recommendations` 2 (ExR6 / ER6).** The recommendation is supported but stops one arm short. Add:
    > "The audit's own decomposition makes the question decidable in one arm: with the ER/ExR term held at 0 onto PEN,
    u_PEN(driven) during the pulse is +9.7 - 0.6 = +9.1 mV, above the 7 mV gap (f ~ 30 Hz) -- so an `edges`-kind hold
    of ExR6 / ER6 / ER4m onto PEN and EPG (`INTERP.md` 10.1 step 5, the wiring row) is the arm that decides whether the
    DC balance is the whole story, and it was not run. The data half of the question is: ExR6 is 2 cells, MaleCNS `nt`
    glutamate, sign -1, with no receptor-table row (tier fallback = NT_SIGN), and the -1 onto EPG comes from the E-PG
    row's GluClalpha (Davis 2020 PB_2, tier alias) -- so the sign is already from an expression profile, and what is
    open is ExR6's transmitter confidence and whether it acts ionotropically at all. I could not verify a published
    transmitter or function for ExR6: Hulse et al. 2021 (eLife 66039) defines ExR1-ExR8 morphologically and by
    connectivity, ExR2 is the dopaminergic PPM3 class, and the classical ring neurons (ER) are GABAergic inhibitors of
    E-PG (Omoto et al. 2017; Fisher et al. 2019; Kim et al. 2019) -- for ExR6 specifically, **unknown**; state it as
    unknown rather than as 'the ring neurons inhibit EPG'."

14. **Section 4 / `recommendations` 4 and `open_questions` 2 (EPG -> EPG chemical vs gap junction).** Replace "whether
    they are chemical in the animal ... is not decided by MaleCNS" with
    > "The connectome's 842 EPG -> EPG pairs *are* annotated chemical synapses -- that is what the EM reconstruction
    stores; gap junctions are not in it at all. So the thing without evidence is the x0.1, not the synapses: the
    `brain.py` comment's premise ('such populations are typically gap-junction coupled') is the claim that needs the
    literature. What would decide it: innexin (ShakB / Inx7) expression in E-PG and paired E-PG recordings for
    electrical coupling; and, in the model, a per-type `same_type_gain` field (default None, CPU bit-identity test)
    plus the suite at >= 3 draws -- which this batch does not justify, because the bump it buys fails the rate row 3x."

---

### Verdict

**Mostly sound.** Every measurement in the batch is exactly reproducible: my own re-implementation of the bump rule
recovers all 48 runs' four primaries and ten secondaries with zero difference; my own weight-shaping pipeline recovers
every wedge matrix, every lambda_k, every unitary weight and the Delta7-cap result to the printed digit; the decision
rule is applied verbatim; the references reproduce `cx_glno.md` exactly (and one row more than claimed); the stamps
order correctly, the arms provably did not change between the two submissions, and the reducer's hashes match at
stamp, at analysis and now. The Answer -- no type-level, data-implied change makes the shipped compass hold a bump at
the shipped gains, and the binding constraint is a DC balance on the relays, not a missing ring mode -- survives the
pass intact. What fails is the argument around it: the headline "the wiring is a ring attractor's" is a property of the
weight matrix asserted at a state where every compass gain is zero (true Jacobian leading eigenvalue 0.045, not 3.55);
the "LIF slope of 5-8" it is compared against is a secant artefact and is 2.4-3.2 at the rates that matter; the stated
reason the rate model missed the F bump is wrong, and the right reason -- a reduction that dropped the one-step
EPG -> EPG term whose undamped k=1 coefficient predicts 145 Hz against the observed 151-156, and a forced drive entered
as a rate instead of a current -- is a defect in this thread's own tool that the audit should own; the ExR6 / ER6
rates that carry the mechanistic story are rate-model outputs the batch's own ring population contradicts by 2x; "the
relays never fire" is contradicted by arm C's measured 4.4-6.4 Hz; and "cancelled before any result was read" is
contradicted by 21 result lines sitting in the repository's own archived console. None of these changes a verdict in
`decision.csv`; all of them change what a reader would believe about why. With corrections 1-14 the audit is sound.

**What the compass now needs.** The shipped-gain question is closed as a *null*, not as a *result*: every gE 1 / gD 1
arm is a structural zero-SD comparison against a reference that is also zero, read as magnitudes, and the two arms that
carry a bump are a labelled global INSTRUMENT (`same_type_gain` 1) and a labelled reference (gE 2 / gD 15) -- neither is
a mechanism the data imply, and both fail `compass.EPG.bump_rate_hz` 3x, so `compass.EPG.bump_survival_s` stays FAIL and
the rate and width rows stay NOT_APPLICABLE. Nothing here licenses an adoption. What it licenses is one more
**measurement** and one more **counterfactual arm**, in that order: first fix the structure tool (forced drive as a
current, the one-step term kept, per-cell gamma at the realised fixed point) and re-derive the ranking, because the
present ranking is computed at a state with zero gain everywhere and its miss on the F family shows it; then run the
single `edges`-kind arm the audit's own decomposition names -- ExR6 / ER6 / ER4m held at 0 onto PEN and EPG -- which
the fixed point predicts puts the driven PEN at +9.1 mV and ~30 Hz during the pulse, and which is the only thing that
turns "the DC inhibition keeps the relays silent" from a decomposition into a tested attribution. Alongside it, the
data question is a **relabel with sources**, not a gain: ExR6's transmitter confidence and whether it has any fast
receptor on E-PG at all (it has no receptor-table row; the -1 rides on the E-PG GluClalpha profile), reported as
`unknown` where it is unknown. The GLNO relabel is a `null` at these gains and should be handed to `glno_relabel.md`
saying exactly that -- a silent neuron's sign is untested, not confirmed harmless. And the F-family bump should not be
brought back as a candidate until it is run in the room with a world: at 151-158 Hz it is a KNOWN GAP swap, and the
rest-of-brain 0.04 Hz in this world-less protocol is no evidence at all about the runaway cliques the x0.1 was adopted
against.
