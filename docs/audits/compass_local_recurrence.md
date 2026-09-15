# Compass, thread 6B: the two-instrument arm -- the hold plus a wedge-local recurrence

Scripts: `scripts/measure_lif_sigma.py` (new; the input-noise measurement), `scripts/cx_wedge.py` (ONE new flag
`--edge-gain PRE_REGEX:POST_REGEX:FACTOR`, default `None`, plus extra RECORDED keys: per-type ring rates and per-cell
maxima; shipped path bit-identical), `scripts/cx_ring_structure.py` (`--sigma` as an argument, `--local-recurrence`,
`--batch cx7`, `--predictions-csv`, `analyse_batch_cx7`; `--out` never defaulted here), `tests/test_cx_wedge_hold.py`
(+6 tests, 12 total), `tests/test_cx_ring_structure.py` (+9 tests, 22 total: 6 for the tool, 3 for the sigma
measurement).
Data: `out/cx7/sigma/{sigma_S_s0.json, sigma_S_s1.json, sigma_H3_s0.json, sigma_*.txt, sigma_summary.json,
sigma_table.csv, slope_vs_sigma.json}`, `out/cx7/structure/{structure.json, structure.md, matrices.npz,
evidence_glno.json, predictions.csv, validation_sigma2.0/, validation_sigma4.6/, validation_sigma11.8/}` (the pass at
the MEASURED sigma 4.6 mV), `out/cx7/structure_sigma2/` (the same at the assumed 2.0 mV), `out/cx7/predeclared.json`
(stamped **2026-09-15T18:29:58Z**, never amended), `out/cx7/{batch.sh, tree_state.json, predeclare_stamp.txt,
submit_stamp.txt, scheduler_receipt.json, scheduler_receipt.py}`, `out/cx7/smoke/*` (the CPU smokes, the job-line
check and the analysis path check), `out/cx7/<arm>_s<seed>.{json,txt,npz}` (**40 runs**), `out/cx7_cluster.log`,
`out/cx7/analysis/{analysis.md, runs.csv, compare.csv, decision.csv, call.csv, scatter.csv, state.csv,
ring_rates.csv, predictions_vs_measured.csv, cross_batch_cx6.csv, analysis.json, scatter.png}`.
Tree: HEAD a41d0f2 (this thread's offline half, committed by the owner before the batch ran); nothing committed by
this thread. `out/cx7/tree_state.json` (stamped 2026-09-15T18:30:29Z) lists every file that shipped with the batch
with its sha256 -- this thread's `scripts/cx_wedge.py`, `scripts/cx_ring_structure.py`,
`scripts/measure_lif_sigma.py`, the two test files, and another thread's concurrent `scripts/probe_vnc_drive.py`,
`tests/test_proprioception.py`, `scripts/derive_level_fixed_point.py` (none on the simulated path: `cx_wedge.py --sim`
builds a `FlyBrain` without a world).

## 0. Answer

**No. With the DC brake off and the wedge-local recurrence on, the ring still holds no bump at the driven tile -- and
the reason is the opposite of the one the DC story predicted.** Both two-instrument arms score `bump_survival_s` 0.00
and `frac_confined_post` 0.000 in **5 of 5 seeds**, so the predeclared working-compass rule is met in **0 of 5** in
every arm of the batch and `compass.EPG.bump_survival_s` stays FAIL. But H3E and H3F fail in opposite ways, and the
difference is the result:

* **H3E -- the hold plus a PER-TYPE undamping of the 842 EPG -> EPG pairs -- keeps the ring and builds the hump in
  roughly the right place.** In **4 of 5 seeds** the EPG profile at 5 s is a seven-wedge hump centred **1.58-1.73
  wedges** from the driven block, with **11 of 11 driven cells above 22 Hz**, in **242.4-244.0 Hz** against out
  **73.1-73.6 Hz**, vector strength **0.67-0.68**. It misses the confinement rule **only on the off-block count**
  (11-12 of 35 cells above 22 Hz against a budget of 3) and the driven-tile rule **only marginally** (1.58-1.73
  against 1.5).
* **H3F -- the same hold plus the GLOBAL `same_type_gain` 1 -- destroys the ring.** Every EPG cell runs at
  **267.6-268.9 Hz inside and 282.9-283.4 Hz outside**, PEN at **278.0-278.4 Hz**, **11 of 11 and 35 of 35** cells
  above 22 Hz, vector strength **0.044-0.050**: a flat saturated ring with no spatial structure at all.

So **what is still missing at the shipped gains is not excitation at the driven tile but inhibition of the rest of the
ring** -- removed by the very hold that let the tile fire. And the global instrument is not a stronger version of the
per-type one: turning on the PEN_a, PEN_b, PEG and Delta7 cliques as well is what takes the ring from "a hump in
roughly the right place that is too wide" to "no ring at all" (section 4.1). The predeclared H3E-vs-H3F contrast
phrase prints "the per-type EPG->EPG recurrence is sufficient", and **that sentence is not a finding**: the two arms
land in the same class because both failed, which the rule's wording did not anticipate (section 4.5).

The GLNO relabel changes the geometry without buying a bump: **H3EG puts the hump within 1.5 wedges of the driven
block in 2 of 5 seeds** (centre 0.29 and 0.53, 11 of 11 driven cells above 22 Hz, in 229-236 Hz) and on the opposite
side in the other three, still with `frac_confined_post` 0.000 in 5 of 5. H3G, H3, S and F reproduce cx6 **exactly**
(max |cx7 - cx6| = 0.0 over 760 metric-run pairs).

**Two measurements the rate model needed were made first, and both stand on their own:**

1. **The spiking LIF's effective input noise, measured for the first time.** The free membrane of the sub-threshold
   relays fluctuates with SD **4.6 mV** (median over 128 never-spiked PEN / PEG / EPGt cells), the EPG's with 5.5 mV,
   and the synaptic input itself with **11.8 mV**, against the assumed, never-measured `SIGMA_MV = 2.0`. The model
   should assume the bracket **4.6-11.8 mV**. Its consequence runs *toward* 5A / 6A rather than against them: a larger
   sigma lowers the maximum slope of the smoothed f-I (8.27 -> **6.06** -> 4.31 Hz/mV), so the shipped ring's
   `gamma_E_crit` of 33.3 Hz/mV is above every operating point by **5.5x** instead of 4.0x, while the F-family
   recurrence at 3.33 stays supercritical across the whole bracket and its predicted rate moves 1.5 % (section 2).
2. **The per-type ring rates, which close 6A's named weak link.** Arm S post-pulse: **ExR6 41.3-42.8 Hz, ER6
   23.1-23.9, ER4m 2.4-2.9**, against the fixed point's 107.7 / 51.9 / 10.3 -- **the rate model is 2.1-2.6x high on
   exactly the two types that carry the PEN DC**, the same factor by which 6A found it overstates H3's PEN.

**The tool, re-run at the measured sigma, called H3 and H3F right and H3E's geometry wrong.** It predicted for H3 a
saturated ring rather than 6A's confined bump (right: measured 17-19 of 35 off-block cells above 22 Hz, no bump in
5/5), for H3F a total runaway with 35 of 35 cells above 22 Hz (right, 1.26x high on PEN), and for H3E a bump with the
driven wedges pinned at the 10 Hz floor -- **the mirror image of what 4 of 5 seeds did** (section 4.2). Its saturation
rates remain good wherever a bump exists (H3G 165.0 predicted against 159.0-164.2 measured; H3EG 241.8 against
229-236; F 144.7 against 150.6-155.8).

**Nothing is adopted, and nothing here could be**: every arm that moved anything is two labelled instruments at once,
both new flags default to `None`, the shipped CPU path is bit-identical (120 of 121 recorded fields), and
`flyverse/brain.py` is untouched -- the per-type instrument turned out to need no new field at all (section 3.2).

## 1. What was measured first, and the bit-identity

### 1.1 The new recorded keys (measurement a)

`cx_wedge.simulate(..., --ledger)` now records, **in addition to** everything it recorded before and without touching
the simulation:

| what | how | why |
|---|---|---|
| per-type ring rates **ExR6 / ER6 / ER4m**, and **EPGt** | four more `g__*` groups in the per-frame `.npz` and `<g>_mean_{pre,during,post}` in `metrics` | 6A's weak link: the rate model's ring rates were the candidate explanation for its 2.1-4.6x error on the held arms and no arm measured them |
| the **per-cell** rates of PEN / Delta7 / PEG / GLNO / EPGt / ExR6 / ER6 / ER4m (127 cells) | `cells__*` in the `.npz`, and `<g>_cell_max_{pre,during,post}` in `metrics` = the max over cells of the window-mean rate | `INTERP.md` 10 defines `silent` by the per-cell maximum; the protocol recorded only group means, so 6A could not decide it (its skeptic pass R4) |

Adding recorded keys cannot change the simulation, and the check is the shipped path itself. **Bit-identity, CPU**
(`CUDA_VISIBLE_DEVICES=-1`, `--device cpu --no-graphs`): `out/cx7/smoke/smoke_default_path_noledger.json` against the
pre-6B `out/cx6/smoke/smoke_default_path.json` is equal on **120 of 121 shared recorded fields**; the single
difference is `wall_s` (a timing field) and the only additions are the two record-keeping keys `edge_gains` /
`edge_gains_resolved`. (6A's own smoke was counted as 117/117 and by its skeptic as 118/119 with `wall_s` excluded;
the field count grew by the 6A and 6B record-keeping keys.)

### 1.2 The instrument flag

`--edge-gain PRE_REGEX:POST_REGEX:FACTOR` (repeatable, default `None`) appends `(pre, post, factor)` to
`LIFParams.type_path_gain` -- the same stage the 6A hold and the gE / gD gains use. With the flag absent the installed
gain list is entry-for-entry the previous one (tested). It is a **LABELLED INSTRUMENT** (a per-type gain), never a
candidate default.

Tests (CPU, `tests/test_cx_wedge_hold.py` 12 passed, `tests/test_cx_ring_structure.py` 22 passed, with
`tests/test_bit_identity.py` 37 passed in one run): the
parser accepts `PRE:POST:FACTOR` and rejects a missing factor, a non-numeric factor, a negative factor, an empty
regex and a bad regex; the flag off is bit-identical; `^EPG$:^EPG$:10` under `same_type_gain` 0.1 reproduces
`same_type_gain=1` **bit for bit on the EPG -> EPG block and leaves every other entry bit-identical to the shipped
matrix**; `(n*10f)*0.1f == n` for every integer count the cap leaves (1-60); the resolved record reports 842 entries,
842 of them same-type, effective factor 1.0; a hold and an edge gain compose in either order; and the 6B structure
configurations carry the intended `LIFParams` (H3E: hold + gain at `same_type_gain` 0.1; H3F: hold at
`same_type_gain` 1).

## 2. The sigma measurement (measurement b) and what it does to the rate model

### 2.1 What was measured

`scripts/measure_lif_sigma.py` runs **the protocol itself** -- `FlyBrain` on the full connectome, no world, compass
adaptation 0, 10 Hz Poisson background on all 46 EPG, 1 s settle, then wedges 0-3 at +40 Hz for 2 s -- on the CPU and
reads the `Brain`'s own state tensors after every 0.5 ms LIF step (`v`, `g`, `refrac`, `spikes`; `INTERP.md` 2.3, no
new brain code, no recorder). Per cell it reports, with the first 200 ms of each window dropped:

* **sigma_g** -- the SD over time of the synaptic input `g` (mV). `Brain._membrane_target` is
  `v_rest + g + drive - adapt`, so `g` is exactly the quantity the rate model's `u` is the mean of. This is the
  **frozen-noise** reading: the width the f-I would be smoothed over if the input were quasi-static on the scale of an
  interspike interval.
* **sigma_v** -- the SD of the **free** membrane potential (samples outside the refractory period), i.e. the same
  noise after the membrane's `tau_m` = 20 ms low-pass. This is the **fast-noise** reading.
* both restricted to the cells that **never spiked** in the window -- the cleanest estimate the protocol offers,
  because a spiking cell's membrane SD is clipped by reset-to-threshold (every firing ring cell reads 2.1-2.4 mV
  whatever its rate, which is an artefact of the reset, not a measurement of its input).

Per-seed values, pasted from `out/cx7/sigma/sigma_table.csv` (columns
`label,seed,window,group,n,never_spiked,sigma_v_median,sigma_v_min,sigma_v_max,sigma_v_never_spiked_median,sigma_g_median,sigma_g_min,sigma_g_max,mean_g_median,rate_mean`):

```
S,0,settle,EPG,46,0,6.7918,5.3666,9.3765,nan,14.4483,11.1645,16.9286,-12.4555,9.9185
S,0,settle,PEN,42,42,4.9104,4.1753,7.6919,4.9104,12.5884,10.7036,17.0809,-7.4663,0.0000
S,0,settle,PEG,18,18,5.2312,4.6004,6.3073,5.2312,12.4854,11.2485,13.7479,-7.6523,0.0000
S,0,settle,EPGt,4,4,4.2282,3.8284,4.9894,4.2282,9.7695,9.4589,11.0178,-6.2078,0.0000
S,1,settle,EPG,46,0,4.5893,3.7215,5.6200,nan,10.8826,9.2966,12.6973,-10.5833,9.2663
S,1,settle,PEN,42,42,3.7795,2.7326,6.3958,3.7795,11.5310,8.8850,14.7765,-6.8518,0.0000
S,1,settle,PEG,18,18,4.4999,3.5678,5.7120,4.4999,11.3746,10.0986,12.2184,-6.7191,0.0000
S,1,settle,EPGt,4,4,3.5643,3.1918,3.9962,3.5643,9.0540,8.6328,9.9559,-6.0820,0.0000
H3,0,settle,EPG,46,0,6.4060,2.0825,16.8563,nan,10.2526,7.0018,18.0114,-28.1552,76.1141
H3,0,settle,PEN,42,29,2.4750,1.4579,4.7532,2.6793,12.3516,6.4570,20.8662,-25.5422,51.8155
H3,0,settle,PEG,18,16,3.2162,2.0291,5.5117,3.2162,14.6611,12.1484,18.4726,-88.3671,3.6806
H3,0,settle,EPGt,4,4,2.5057,1.4396,3.7520,2.5057,11.0162,9.7731,12.3411,-55.9493,0.0000
```

The whole file carries all 9 groups x 2 windows x 3 runs, plus the driven / undriven EPG split during the pulse.
The pooled headline (`out/cx7/sigma/sigma_summary.json`, `headline`): **sigma_v of the sub-threshold relays 4.638 mV**
(range 2.733-7.692 over 128 never-spiked PEN / PEG / EPGt cells of S seeds 0 and 1), **sigma_v of the EPG 5.521 mV**
(92 cells), **sigma_g of the relays 11.800 mV**, **sigma_g of the EPG 12.456 mV**.

**What the rate model should assume.** Not 2.0 mV. The defensible statement is a bracket: the smoothed f-I's sigma is
the width of the input distribution the spike generator sees over an interspike interval, which is bounded below by
the membrane-filtered SD (**4.6 mV**) and above by the frozen input SD (**11.8 mV**). This thread re-runs the tool at
**4.6 mV** (the lower, conservative end, and the one a diffusion picture of an integrator justifies) and validates at
**2.0, 4.6 and 11.8**. Three honest caveats: the measurement is a **CPU** one at **two seeds of one arm** (a third run
covers H3); it is the noise of the *shipped* state, and under the hold the same cells read 2.5-3.2 mV on the membrane
and 12.4-14.7 mV on the input (the hold changes the noise as well as the mean); and a cell's sigma varies 2.7-7.7 mV
across cells, so a single scalar sigma is a model of a distribution either way.

### 2.2 The consequence: the slope bound falls, and the conclusions do not move

`out/cx7/sigma/slope_vs_sigma.json` (generator `scripts/measure_lif_sigma.py` measures sigma;
`cx_ring_structure.max_slope / rate_at_gain / u_for_rate` at each sigma produce this table):

| sigma (mV) | 0.25 | 1.0 | **2.0 (assumed)** | **4.6 (measured, membrane)** | 6.0 | **11.8 (measured, input)** |
|---|---|---|---|---|---|---|
| max slope of the smoothed f-I (Hz/mV) | 26.09 | 11.35 | **8.27** | **6.06** | 5.54 | **4.31** |
| at u (mV) / f (Hz) | 7.10 / 9.1 | 7.80 / 18.3 | 8.60 / 25.1 | 12.53 / 51.6 | 14.22 / 60.8 | 21.19 / 92.6 |
| saturation rate at gamma_crit 3.345 (the F family) | 144.2 | 144.2 | **144.3** | **144.7** | 145.0 | **146.5** |
| saturation rate at gamma_crit 3.017 (CF / CFG) | 159.6 | 159.7 | 159.7 | 160.0 | 160.3 | 162.1 |
| u for a forced 10 Hz / 50 Hz (mV) | 7.18 / 11.88 | 7.00 / 11.90 | 6.63 / 11.99 | 3.96 / 12.23 | 2.59 / 12.22 | -2.43 / 10.66 |

These are the tool's own 61-node Gauss-Hermite values, which 6A's skeptic showed to be ~3.4 % high against adaptive
quadrature at sigma 2 (8.27 vs 8.00); the same bias is expected at the other sigmas and does not change a sign
anywhere below.

Two consequences, and they point the same way:

* **The shipped-ring conclusion is strengthened, not threatened.** `gamma_E_crit` is a property of the weights alone
  (33.34 Hz/mV shipped, 28.77 under the receptor tier) and does not move with sigma; the bound it is compared against
  falls. At the measured sigma the shipped ring's critical gain is above every operating point of the smoothed f-I by
  **5.5x** (33.34 / 6.06) instead of 4.0x, and by 7.7x at the frozen-noise end. **The measured sigma cannot rescue the
  shipped compass**; sigma would have to be **0.17 mV**, 27x smaller than measured, for 33.3 Hz/mV to be reachable.
* **The F-family / H3E-H3F recurrence stays supercritical and its predicted rate is sigma-insensitive.**
  `gamma_E_crit` 3.33-3.34 is below the maximum slope at every sigma in the bracket, and the predicted saturation rate
  moves by 1.5 % across the whole 2.0-11.8 mV range (144.3 -> 146.5 Hz). This is why re-running the ranking at the
  measured sigma changes almost nothing about **which** configurations can hold a bump.

### 2.3 The ranking and the validation re-run at the measured sigma -- what moved

The fixed tool re-scored against the cx5 batch it originally missed, at three sigmas
(`out/cx7/structure/validation_sigma{2.0,4.6,11.8}/validation.{md,json}`; the arms are cx5's eight shipped-gain arms,
4 runs each):

| sigma | max slope | bump calls correct | rate calls within 10 % | F predicted / measured | CFG | CF |
|---|---|---|---|---|---|---|
| 2.0 (assumed; reproduces 6A) | 8.27 | **7 of 8** | 2 of 3 | 144.3 / 152.5 (5.4 %) | 159.7 / 157.1 (1.7 %) | 159.7 / 142.8 (11.8 %) |
| **4.6 (measured)** | **6.06** | **7 of 8** | 2 of 3 | **144.7 / 152.5 (5.1 %)** | 160.0 / 157.1 (1.9 %) | 160.0 / 142.8 (12.0 %) |
| 11.8 (input-side) | 4.31 | **7 of 8** | 2 of 3 | 146.5 / 152.5 (3.9 %) | 162.1 / 157.1 (3.2 %) | 162.1 / 142.8 (13.5 %) |

**Nothing moved in the validation**: the same 7 of 8 arms (which 6A's skeptic correctly reduced to **3 of 4 distinct
predictions**, because `gamma_E_crit` takes four values over eight arms and the classifier is "does this arm carry
`same_type_gain` 1?"), the same single miss (FG), the same 2 of 3 rate calls. What *did* move is the H3 family's fixed
point, which is the subject of section 4: at sigma 4.6 the driven EPG's own gain at the realised state changes enough
to turn 6A's predicted confined bump into a predicted saturated ring -- the direction the measurement had already
gone. The full 40-row (configuration x sigma) comparison is `out/cx7/structure/predictions.csv`.

## 3. The two instruments, on the real connectome

### 3.1 The hold (carried from 6A, unchanged)

`--hold-edges '^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)'`: 17 presynaptic cells (2 ExR6 + 4 ER6 + 11 ER4m) onto 88
postsynaptic cells (46 EPG + 42 PEN), **1,149 entries, 37,256 synapses** at factor 0, EPGt excluded. Reconfirmed on
this desktop by the CPU code check below. A LABELLED COUNTERFACTUAL.

### 3.2 The per-type recurrence, and why no `brain.py` field was needed

The task allowed a new opt-in `LIFParams` field if unavoidable. **It was avoidable.** `brain._shaped_weights` applies
`type_path_gain` at line ~328 and `same_type_gain` at line ~332 -- the per-type stage comes **first** -- so

    --edge-gain '^EPG$:^EPG$:10'        (x10 before the shipped x0.1)  =  x1.0 on the 842 EPG -> EPG pairs only

and the float32 arithmetic is exact for every count the connection cap leaves: `(n * 10f) * 0.1f == n` for
n = 1..60, so the block is **bit-identical** to `same_type_gain=1`'s, not merely close (pinned by
`test_every_integer_count_survives_x10_then_x0_1_in_float32` and by a direct matrix comparison on a synthetic graph).
On the real connectome (`out/cx7/structure/structure.json`, `same_type` per configuration, mV per pair / per volley):

| configuration | EPG -> EPG (842 pairs) | PEN_a -> PEN_a (257) | PEN_b -> PEN_b (274) | PEG (114) | Delta7 -> Delta7 (1,717) |
|---|---|---|---|---|---|
| shipped | +0.366 / +6.7 | +0.583 / +7.5 | +0.624 / +7.8 | +0.131 / +0.8 | -0.425 / -17.4 |
| **H3E** (per-type) | **+3.666 / +67.1** | +0.583 / +7.5 | +0.624 / +7.8 | +0.131 / +0.8 | -0.425 / -17.4 |
| **H3F** (global) | **+3.666 / +67.1** | **+5.827 / +74.9** | **+6.245 / +77.8** | **+1.310 / +8.3** | **-4.251 / -173.8** |

That is the separation 5A and 6A both wanted and neither could express: H3E and H3F differ **only** in the four
non-EPG same-type cliques. The wedge-local one-step EPG -> EPG coefficient goes from +6.00 mV (shipped) to
**+60.01 mV** in both, and `gamma_E_crit` from 33.33 to **3.33 Hz/mV** -- below the 6.06 Hz/mV maximum slope at the
measured sigma, i.e. supercritical, predicting saturation at 145.2 Hz.

`--edge-gain` does not touch EPGt (`^EPG$` is anchored) and its resolved record proves what it multiplied: **842
entries, 842 of them same-type, effective factor 1.000 on the same-type entries** -- checked per run by the analysis.

### 3.3 The per-type ring rates: the rate model is ~2.2x high on exactly the DC types

The rate model's ring rates were 6A's named weak link. The fixed point's per-type rates (from
`structure.json`, `decomposition.<state>.PEN_in<-Ring_by_type`) against the LIF's, measured on the CPU by
`measure_lif_sigma.py` (population means over the same windows; per-seed values in
`out/cx7/sigma/sigma_table.csv`, column `rate_mean`):

| arm / state | type | rate model (sigma 2.0) | rate model (sigma 4.6) | **measured LIF (CPU)** | model / measured |
|---|---|---|---|---|---|
| S, background | ExR6 | 110.2 Hz | 107.7 Hz | **41.9-43.1 Hz** | **2.5-2.6x** |
| S, background | ER6 | 53.8 | 51.9 | **22.8-24.4** | **2.1-2.4x** |
| S, background | ER4m | 4.4 | 10.3 | **1.7-3.2** | 1.4-2.6x at sigma 2.0, 3.2-6.1x at 4.6 |
| S, pulse | ExR6 | 183.2 | 181.5 | **83.9-85.8** | **2.1-2.2x** |
| S, pulse | ER6 | 116.3 | 115.2 | **50.4-52.5** | **2.2-2.3x** |
| S, pulse | ER4m | 22.3 | 24.1 | **11.2-11.9** | 1.9-2.2x |

(The "measured LIF (CPU)" column is the two-seed CPU measurement of section 2; the GPU batch's five-seed post-pulse
means are quoted below and in `out/cx7/analysis/ring_rates.csv`.)

**The model overstates the two types that carry the PEN DC by a factor of 2.1-2.6, the same factor by which 6A found
it overstates H3's PEN rate (2.1-2.6x).** That is the quantitative answer to 6A's open question, and it says the error
is in the ring's operating point rather than in the criterion. One caveat on the table above: the comparison is of
population means at matched windows, with the model's "background" read against the LIF's settle window.

**The batch confirms it at five seeds per arm on the GPU** (`out/cx7/analysis/ring_rates.csv`, post-pulse means over
5 seeds): arm S **ExR6 41.3-42.8 Hz, ER6 23.1-23.9, ER4m 2.4-2.9, GLNO 0.01-0.33**; H3 **ExR6 255.2-264.6, ER6
179.8-182.9, ER4m 90.1-105.6, GLNO 135.9-137.6**; H3E **286.3-291.3 / 237.8-241.0 / 129.0-137.1 / 174.2-178.5**; H3F
**332.5-333.3 / 326.8-327.8 / 168.1-168.4 / 317.0-318.6** (ExR6 at its ceiling). The CPU figures above and the GPU
batch agree to within the seed spread, and every arm of the batch now carries its per-type ring rates -- the
recording gap 6A named is closed.

Under the hold the same cells are **not** silenced -- only their edges onto PEN and EPG are -- and they run much
harder, because the EPG that drives them is itself released: H3, settle, CPU seed 0 gives **ExR6 274.4 Hz, ER6
191.3 Hz, ER4m 101.4 Hz, GLNO 149.4 Hz** with EPG at 76.1 Hz (`sigma_table.csv`, `rate_mean`). This is the same
picture 6A read from the group means (GLNO 135.9-137.6 Hz under H3) now resolved per type.

### 3.4 `silent`, decided

`INTERP.md` 10 defines `silent` as a per-cell maximum below 0.5 Hz, which 6A's protocol could not measure. In arm S's
settle window on the CPU (seeds 0 and 1), the `never_spiked` column of `sigma_table.csv` reads **42 of 42 PEN, 18 of
18 PEG, 4 of 4 EPGt and 4 of 4 GLNO** -- no cell of those four groups emits a single spike in 0.8 s, so every per-cell
maximum is 0.0 Hz and all four groups **are silent** in the project's sense at rest under the shipped model. During
the pulse 6 of 42 PEN fire (group mean 0.34 Hz in seed 0, 0.28 in seed 1) and all 4 GLNO do (1.25-1.67 Hz), so
**neither is silent during the pulse**; ExR6, ER6, ER4m and Delta7 are never silent in any window (0 of 2, 0 of 4,
1-3 of 11 and 0 of 42 never-spiked at rest). Where this thread quotes a group mean
it says so; no cell with a non-zero recorded mean is called silent anywhere in this audit.

## 4. The decisive table: predictions against measurements

Predictions are `out/cx7/predeclared.json` `predictions_from_the_structure_pass.per_arm`, stamped
**2026-09-15T18:29:58Z** at the **measured** sigma 4.6 mV, hours before the batch ran; measurements are
`out/cx7/analysis/runs.csv` over 5 seeds (ranges), assembled by the analysis into
`out/cx7/analysis/predictions_vs_measured.csv`. "in>22 / out>22" is the count of driven (of 11) and off-block (of 35)
EPG cells above 22 Hz at 5 s -- the ledger's own confinement count, and the criterion 6A's H3 prediction lacked.

| arm | PEN during: pred / meas | EPG in / out post: pred / meas | in>22, out>22: pred / meas | fixed-point call | measured: bump 5s / at tile | verdict on the prediction |
|---|---|---|---|---|---|---|
| S | 0.0 / **0.3-0.7** | 10.0 / 10.0 vs **10.2-10.8 / 9.6-9.9** | 0, 0 / **0-2, 1-3** | no bump | 0/5, 0/5 | **right** |
| H3 | 99.3 / **40.2-48.3** | 34.1 / 95.3 vs **25.9-116.0 / 55.5-89.5** | 3, 14 / **0-4, 17-19** | RUNAWAY, no confined bump | 0/5, 0/5 | **right on the call and the profile**; 2.1-2.5x high on PEN |
| **H3F** | 351.5 / **278.0-278.4** | 304.3 / 326.7 vs **267.6-268.9 / 282.9-283.4** | 11, 35 / **11, 35** | RUNAWAY, no confined bump | 0/5, 0/5 | **right, including 35 of 35**; 1.26x high on PEN |
| **H3E** | 16.3 / **57.3-62.9** | 10.0 / 164.7 vs **10.7-244.0 / 73.1-151.4** | 0, 20 / **2-11, 11-23** | RUNAWAY, a bump OFF the driven tile | 0/5, 0/5 | **right on the call, WRONG on the geometry in 4 of 5 seeds** (section 4.2) |
| F | 0.0 / **22.5-27.4** | 10.0 / 10.0 vs **59.4-155.2 / 9.6-11.2** | 0, 0 / **2-9, 1-3** | no fixed-point bump; recurrence 144.7 Hz | 4/5, 4/5 | rate **right to 3.9-5.1 %** (meas 150.6-155.8); the fixed point misses the bump, as in 5A / 6A |
| H3G | 36.0 / **26.5-29.0** | 165.0 / 11.6 vs **17.7-160.7 / 14.3-54.8** | 11, 2 / **0-11, 5-16** | BUMP, CONFINED | 1/5, 1/5 | rate **right to 0.5-3.8 %** (meas 159.0-164.2); confinement right in 1 of 5 seeds only |
| H3FG | 0.0 / **89.9-119.8** | 10.0 / 149.5 vs **226.2-272.3 / 100.0-142.6** | 0, 18 / **11, 16-19** | RUNAWAY, bump off the tile | 0/5, 0/5 | **wrong on PEN** (predicted silent, measured 90-120 Hz); right that nothing is confined |
| H3EG | 46.3 / **27.7-37.1** | 241.8 / 39.4 vs **10.2-236.2 / 38.0-126.1** | 10, 6 / **0-11, 8-22** | BUMP, not confined | 0/5, **2/5** | rate **right to 2-5 %** in the 2 seeds that land on the tile; geometry 2 of 5 |

### 4.1 The answer to the thread's question: no, and not for the predicted reason

**Neither recurrence buys a bump at the driven tile.** `bump_survival_s` is **0.00 and `frac_confined_post` 0.000 in
5 of 5 seeds in both H3E and H3F** (and in H3FG and H3EG), so by the predeclared working-compass rule every
two-instrument arm scores **0 of 5**, exactly as the shipped arm and H3 do. The ledger rows stay `FAIL` /
`NOT_APPLICABLE` in all 40 runs.

But the two arms fail in **opposite** ways, and that is the finding:

* **H3F (the global hand-rule removal) destroys the ring.** Every EPG cell sits at **267.6-268.9 Hz inside and
  282.9-283.4 Hz outside** the driven block, PEN at **278.0-278.4 Hz**, GLNO at 317-319, ExR6 at 332-333 (its
  ceiling), with **11 of 11 and 35 of 35** cells above 22 Hz and a vector strength of **0.044-0.050** -- a flat,
  saturated ring with no spatial structure at all. Its `wedge_profile_end` is flat to within 10 % across all 16
  wedges in every seed (`state.csv`, `profile_end`; seed 0:
  `272;268;287;262;290;285;288;280;290;289;281;287;290;288;276;266`).
* **H3E (the per-type EPG -> EPG undamping) keeps the ring's structure and puts a tall hump on the driven block.**
  In **4 of 5 seeds** the profile at 5 s is a 7-wedge hump centred at wedge **3.08-3.23**, i.e. **1.58-1.73 wedges
  from the driven centre 1.5**, with **11 of 11 driven cells above 22 Hz**, EPG in **242.4-244.0 Hz** against out
  **73.1-73.6 Hz**, and vector strength **0.67-0.68** (seed 0:
  `197;243;271;257;284;252;200;29;8;15;6;14;12;22;5;16`). It fails the confinement rule **only on the off-block
  count** (11-12 of 35 above 22 Hz, against the rule's <= 3) and the driven-tile rule **only marginally** (1.58-1.73
  against the 1.5 bound). The fifth seed puts the same hump on the opposite side (centre 11.39).

So the honest statement of the mechanism result is: **the wedge-local recurrence at its connectome weight, with the DC
brake off, produces a wide high hump anchored on the driven block -- and what is still missing is not excitation at
the tile but inhibition everywhere else.** The ring cannot suppress the other 35 cells to the 3-cell budget the
confinement rule demands, because the same hold that let the driven wedges fire also removed the inhibition that would
have silenced the rest.

### 4.2 The one prediction that was wrong, and how

The fixed point called H3E **"a bump in the wrong place"** -- driven wedges pinned at the 10 Hz forced floor
(`EPG in 10.0`) while the rest of the ring ran at 164.7 Hz, 0 of 11 driven cells above 22 Hz. The measurement is the
**opposite geometry in 4 of 5 seeds**: in 242-244 Hz, out 73 Hz, 11 of 11 driven cells above 22 Hz. Only **seed 4**
matches the prediction (in 10.7 / out 151.4 Hz, 2 of 11 in, 23 of 35 out).

So the tool got H3E's **call** right (no confined bump) for the **wrong reason** in four seeds out of five. The
prediction and the measurement agree that the ring saturates and that nothing is confined; they disagree about
**where** the excess sits, which is precisely the quantity this round added `centre_dist_t5` to measure. That is this
round's headline tool miss and it is recorded as one: a rate-model fixed point that lands on a mirror-image branch of
a bistable ring is not a validated predictor of bump position, and no adoption or ranking should rest on its
placement.

### 4.3 The verdicts (`out/cx7/analysis/compare.csv`; `common.compare`, runs = the unit, 5 v 5, p_floor 0.0079)

The predeclared family is five primaries, Holm within each **(arm, reference)** family, references **S** and **H3**
scored separately; **m = 5 in every one of the 14 families**, so Holm at the floor is 5 x 0.0079 = **0.0397 <= 0.05**,
as predeclared. `bump_hz_post` and `width_half_post` returned no p in every family (neither reference has a confined
frame) and are read as magnitudes, exactly as the predeclaration says.

| arm vs S | `PEN_mean_during` | `PEN_mean_post` | `centre_dist_t5` | `survival_s` / `frac_confined_post` |
|---|---|---|---|---|
| H3 | **+43.53, z 294.0, Holm 0.0397, `result`** | **+52.38, z 2504, `result`** | -0.87, z -1.30, p 0.0079 | +0.00 / +0.000, zero-SD both arms, `null` |
| **H3F** | **+277.80, z 1876, `result`** | **+277.80, z 13280, `result`** | +1.06, z +1.58, Holm 0.0476 | +0.00 / +0.000, `null` |
| **H3E** | **+59.21, z 399.9, `result`** | **+56.50, z 2701, `result`** | -2.83, **z -4.21, p 0.095** (not a result) | +0.00 / +0.000, `null` |
| F | **+24.69, z 166.8, `result`** | **+26.41, z 1262, `result`** | -3.80, z -5.65, p 0.151 | +4.37 / +0.720, `undetermined` (zero-SD null) |
| H3G | **+26.94, z 182.0, `result`** | **+23.74, z 1135, `result`** | -0.72, z -1.07, p 0.690 | +4.88 / +0.324, `undetermined` |
| H3FG | **+100.48, z 678.7, `result`** | **+114.07, z 5453, `result`** | **-2.54, z -3.78, Holm 0.0397, `result`** | +0.00 / +0.000, `null` |
| H3EG | **+31.01, z 209.4, `result`** | **+35.57, z 1700, `result`** | -0.87, z -1.29, p 0.690 | +0.00 / +0.000, `null` |

Against the second reference **H3**, the two instruments separate cleanly: **H3F +234.27 Hz of PEN during the pulse
(z 67.7, `result`)** and **+1.93 wedges of centre distance (z 11.3, `result`)** -- it drives PEN four times harder
*and* pushes the profile further from the driven tile -- while **H3E is +15.68 Hz on PEN (z 4.53, `result`)** and
**-1.96 wedges** (z -11.5, p 0.151, not a result at 5 v 5). H3FG is -1.67 wedges vs H3 (z -9.8, `result`).

**One reading rule earned its place here.** H3E's `centre_dist_t5` moves by -2.83 wedges with **z -4.21** and is
**not** a result, because the exact-U p is 0.095: four seeds move together and the fifth (the mirror-image seed) sits
on the other side, so the rank test cannot separate the arms at 5 v 5 however large the mean shift. `|z| >= 3` alone
would have called it (`INTERP.md` 10.2 requires `|z| >= 3` **and** `p <= 0.05`); the per-seed scatter below is what a
reader needs, and it is why this audit quotes the four-seed cluster and its outlier rather than the mean.

### 4.4 Per-seed scatter (pasted from `out/cx7/analysis/scatter.csv`, columns `arm,key,seeds,values`)

```
S,PEN_mean_during,"0,1,2,3,4","0.3599,0.6614,0.3370,0.2865,0.4528"
H3,PEN_mean_during,"0,1,2,3,4","40.1718,43.5047,41.1931,48.3111,46.5668"
H3,centre_dist_t5,"0,1,2,3,4","4.5472,4.6128,4.6129,4.5462,4.2069"
H3F,survival_s,"0,1,2,3,4","0.0000,0.0000,0.0000,0.0000,0.0000"
H3F,PEN_mean_during,"0,1,2,3,4","278.1882,278.0305,278.4355,278.1423,278.2829"
H3F,centre_dist_t5,"0,1,2,3,4","6.3972,6.5695,6.6093,6.1958,6.3963"
H3E,survival_s,"0,1,2,3,4","0.0000,0.0000,0.0000,0.0000,0.0000"
H3E,PEN_mean_during,"0,1,2,3,4","61.4718,57.3392,57.5085,62.8778,58.9596"
H3E,centre_wedge_t5,"0,1,2,3,4","3.1511,3.1804,3.0846,3.2266,11.3917"
H3E,centre_dist_t5,"0,1,2,3,4","1.6511,1.6804,1.5846,1.7266,6.1083"
F,survival_s,"0,1,2,3,4","5.0000,5.0000,5.0000,5.0000,1.8400"
F,survival_at_tile_s,"0,1,2,3,4","5.0000,5.0000,5.0000,5.0000,1.8400"
F,bump_hz_post,"0,1,2,3,4","155.7587,150.9075,152.2624,151.0834,150.5561"
F,centre_dist_t5,"0,1,2,3,4","0.4175,0.3663,0.4645,0.2547,6.3934"
H3G,survival_s,"0,1,2,3,4","5.0000,4.7600,4.9100,4.9800,4.7700"
H3G,survival_at_tile_s,"0,1,2,3,4","0.8500,4.7600,2.7900,1.1800,0.0000"
H3G,bump_hz_post,"0,1,2,3,4","160.7983,163.5748,161.2387,164.1680,158.9786"
H3G,centre_dist_t5,"0,1,2,3,4","3.9343,0.5849,7.4853,7.4682,3.8036"
H3FG,PEN_mean_during,"0,1,2,3,4","119.8306,97.9420,99.6572,97.1451,89.9416"
H3FG,centre_dist_t5,"0,1,2,3,4","3.5758,1.8178,1.8486,3.5323,3.4030"
H3EG,PEN_mean_during,"0,1,2,3,4","27.7225,27.8863,36.0670,37.0846,28.3813"
H3EG,centre_wedge_t5,"0,1,2,3,4","10.7896,10.7663,0.2927,0.5278,10.5719"
H3EG,centre_dist_t5,"0,1,2,3,4","6.7104,6.7337,1.2073,0.9722,6.9281"
```

The whole file carries all 8 arms x 7 keys; the six-panel figure is `out/cx7/analysis/scatter.png`, and the per-seed
state table (profiles, in / out counts, vector strength, at-tile survival) is `out/cx7/analysis/state.csv`.

**The bistability is per seed, not per arm, in three arms.** H3E splits 4 + 1 (centre 3.08-3.23 vs 11.39), H3EG splits
**2 + 3** (centre 0.29 / 0.53 vs 10.57-10.79) and H3G splits 1 + 2 + 2 (0.92, then 5.30 / 5.43, then 10.01 / 10.03).
Each ring settles on one of two or three mirror-image positions depending on the background realisation -- the same
seed-dependence `cx_wedge.md` 6 found in the gain grid and 6A found in H3G. **No claim in this audit rests on a
single seed.**

### 4.5 The two rules and the decision phrases, applied verbatim

`out/cx7/analysis/call.csv`, produced by the analysis script from the predeclared words (`n_runs 40; problems 0`):

| arm | working-compass seeds (>= 3 of 5?) | driven-tile seeds | seeds with a 5 s bump | bump-at-tile seeds | call |
|---|---|---|---|---|---|
| S | 0/5 | 0/5 | 0/5 | 0/5 | no bump |
| H3 | 0/5 | 0/5 | 0/5 | 0/5 | no bump |
| **H3F** | **0/5** | **0/5** | **0/5** | **0/5** | **no bump** |
| **H3E** | **0/5** | **0/5** | **0/5** | **0/5** | **no bump** |
| F | 0/5 (rate 150.6-155.8) | 4/5 | 4/5 | 4/5 | a bump at the driven tile, not a compass (rate outside 5-60 Hz) |
| H3G | 0/5 | 1/5 | 1/5 | 0/5 | no bump |
| H3FG | 0/5 | 0/5 | 0/5 | 0/5 | no bump |
| H3EG | 0/5 | **2/5** | 0/5 | 0/5 | no bump |

**'A compass at the driven tile' is refused for both H3E and H3F** (0 of 5 seeds meet the working-compass rule, and
the driven-tile rule is met in 0 of 5 for both). **'A bump at the driven tile, not a compass' is refused** (no seed
survives 5 s). **The call for both is 'no bump'.**

**The predeclared H3E-vs-H3F contrast is uninformative here, and saying so is part of applying it.** The rule words it
as: "'the per-type EPG -> EPG recurrence is sufficient' when H3E lands in the same or a better class than H3F". H3E
does land in the same class, so the analysis prints **"the per-type EPG->EPG recurrence is sufficient (H3E in the same
class as H3F)"** -- but they are in the same class because **both failed**, not because the per-type instrument
achieved what the global one did. The rule was worded on the assumption that at least one arm would hold a bump, and
at 0/5 versus 0/5 it cannot discriminate. **That sentence is not a finding and is not quoted as one**; it is the 6A
defect (`call.csv` printing a phrase outside the case its rule covers) recurring in this thread's own predeclaration,
and it is recorded here rather than quietly dropped.

The informative comparison is the one the primaries make, and it points the **other** way: with the DC brake off,
**the per-type recurrence preserves the ring's spatial structure and the global one destroys it** -- H3E vector
strength **0.67-0.68** with a hump on the driven block, H3F **0.044-0.050** and flat at ~280 Hz; H3E's PEN at
57-63 Hz against H3F's 278 Hz; H3E's off-block EPG at 73 Hz against H3F's 283 Hz. Turning on the PEN_a, PEN_b, PEG
and Delta7 cliques as well is what takes the ring from "a hump in roughly the right place that is too wide" to "no
ring at all".

## 5. The batch

`out/cx7/predeclared.json`, stamped **2026-09-15T18:29:58Z** (`written_before_submission` true), **never amended**
(no archive file exists because none was needed), carrying the arms, the two references, the family and its drop rule,
both decision rules, the four-way phrases and the fixed tool's per-arm predictions at the **measured** sigma. Order,
from the files' own stamps and mtimes: the three sigma runs (18:0x-18:1xZ) < the structure pass at the measured sigma
(`structure.json` `generated_utc` **18:19:50Z**) < `batch.sh` **18:23:25Z** (sha256
`1552e602f0d81510e795ded8a9858dab0f2f41bb1b34189aa23fd22339a91bfc`, the value recorded in the predeclaration) < the
three validations (18:23-18:29Z) < **predeclaration 18:29:58Z** < `predeclare_stamp.txt` 18:29:59Z <
`tree_state.json` 18:30:29Z < **a first submission attempt at 18:30:36Z that reached no cluster** (section 5.1) <
the successful submission **19:11:54Z** < scheduler `submitted_at` 19:11:55Z < last job finished 19:14:48Z < the
analysis. **`batch.sh` was not regenerated between the two attempts**: the sha256 at submission is the one stamped in
the predeclaration.

**Submission.** ONE `cluster_run.py --name cx7 --minutes 30 --arm-block fam <10 job lines> --fetch out/cx7/` call,
10 jobs = 5 seeds x {5 GLNO-silent arms sequential, 3 relabel arms sequential}, blocks `fam_s<seed>` (every arm of one
seed on one target, so no experimental factor is collinear with a box).
**The failure line: `10 job(s), 0 failed  (4.3 min)  run dir /mnt/beegfs/neurome/runs/cx7-201788`, `client exit 0`**;
scheduler receipt (`out/cx7/scheduler_receipt.json`, queried from the job manager itself): **`10 job(s), 0 failed
({'completed': 10})`**, all on **<cluster-node>**, GPU ids 0 and 1, NVIDIA B200, first submitted 19:11:55.46Z, last finished
19:14:48.92Z. (The job manager records `exit_code: None` for completed jobs, so the exit-code half of `INTERP.md`
10.4 rule 21 rests on the status field and on the per-run checks below.)

**Artefacts: 40 `<arm>_s<seed>.json`, 40 `.txt`, 40 `.npz`**, 5 per arm x 8 arms; **40 of 40 consoles say
`device cuda`**; the analysis's per-run checks raise **0 problems** over device, arm label, gains, `nt_override` /
`glno_nt`, `same_type_gain`, `receptor_model`, **the hold** (spec, factor 0, exactly 1,149 entries) and **the edge
gain** (spec, exactly 842 entries, **all 842 same-type**, effective factor 1.000), plus the presence of every new
per-type ring rate. Compiled-W md5 `ef23cc27bea13be7f6a96f3c04fd3737` on all 25 GLNO-silent runs and
`7a10d93ba2086f2c76bcdabdca79b4ec` on all 15 relabel runs -- the same two hashes as cx5, cx6 and this desktop.
`same_type_gain` is 1.0 in exactly F / H3F / H3FG and 0.1 in the other five arms; `edge_gains` matches 842 entries in
exactly H3E / H3EG. Wall 13.7-27.5 s per run.

**Cross-batch reproduction.** The four arms cx6 and cx7 share (S, H3, H3G, F; 20 runs) reproduce cx6 seed for seed:
**max |cx7 - cx6| = 0.0 over 760 (metric, run) pairs** (`out/cx7/analysis/cross_batch_cx6.csv`, all 38 recorded
metrics x 20 runs, NaN-vs-NaN counted as equal). The protocol is deterministic given (config, gains, seed) on this
backend, the new recorded keys changed nothing, and the three references this thread inherited are the same runs 6A
read.

### 5.1 The first submission attempt, and the defect it exposed

The batch was first submitted at **18:30:36Z** and reached no cluster: the house VPN was down on this machine, so
`<cluster-node>` did not resolve. The client's own words:

```
[house] UNAVAILABLE (API silent for 20s): <urlopen error [Errno 11001] getaddrinfo failed>
no target answered; nothing submitted
client exit 0
```

Nothing was submitted, so nothing was fetched and nothing needed recovering; the batch was re-run **unchanged** at
19:11:54Z once the VPN was up, which is why `batch.sh`'s stamped hash still holds. **The `client exit 0` was the
wrapper's, not the client's** (owner's correction, 2026-09-15): `cluster_run.py` returns **2** on both
nothing-submitted branches, and `batch.sh` ends `... | tee out/cx7/client_stdout.txt`, so without `pipefail` the
pipeline exits with `tee`'s status. The second submission was run as `bash -o pipefail out/cx7/batch.sh`. **The defect
is in the generated wrapper**: `cx_ring_structure.plan_batch` emits a `| tee` line with no `set -o pipefail`, so every
`batch.sh` this project has generated -- cx5's, cx6's and cx7's -- reports the tee's exit code rather than the
client's. That is `INTERP.md` 10.4 rule 4 ("a job line must preserve the python exit code") applied one level up, to
the submission wrapper itself, and it is the one process defect this round found.

## 6. What this establishes, what it does not

**Established.** With the DC brake held off and a wedge-local recurrence on, at the shipped gains:

1. **No arm holds a bump at the driven tile.** `survival_s` 0.00 and `frac_confined_post` 0.000 in 5 of 5 seeds in
   H3E, H3F, H3FG and H3EG; the working-compass rule is met in 0 of 5 seeds in every arm of the batch; the ledger
   rows stay `FAIL` / `NOT_APPLICABLE` in all 40 runs.
2. **The per-type and the global instrument fail differently, and the per-type one is far closer.** H3E keeps a
   242-244 Hz hump on the driven block (11 of 11 driven cells above 22 Hz, vector strength 0.67-0.68) in 4 of 5
   seeds and fails the confinement rule on the off-block count (11-12 of 35 above 22 Hz, budget 3); H3F is a flat
   saturated ring (267-283 Hz everywhere, 35 of 35 above 22 Hz, vector strength 0.044-0.050). What is missing at the
   shipped gains is therefore **not local excitation at the driven tile but inhibition of the rest of the ring** --
   removed by the very hold that let the tile fire.
3. **The rate model's ring rates are 2.1-2.6x high, measured per type on the GPU batch** (arm S post-pulse: ExR6
   41.3-42.8 Hz, ER6 23.1-23.9, ER4m 2.4-2.9, against the fixed point's 107.7 / 51.9 / 10.3 at background), closing
   6A's named weak link with the batch rather than with a CPU pair of seeds.
4. **`silent` is measurable in this protocol now, and the answer is window-dependent.** In arm S the per-cell maximum
   over the **post-pulse** window is PEN 0.13-0.58 Hz, PEG 0.40-0.61, GLNO 0.02-0.39 and EPGt 0.00 -- so EPGt and
   GLNO meet `INTERP.md` 10's `silent` (max < 0.5 Hz per cell) in 5 of 5 seeds, while **PEN and PEG straddle the
   0.5 Hz line across seeds** and are silent in some and not others. (The CPU settle-window measurement of section
   3.4 is a different window: there no PEN, PEG, EPGt or GLNO cell spikes at all.) No population with a non-zero
   recorded mean is called silent anywhere in this audit.
5. **The tool re-run at the measured sigma called H3 and H3F right and H3E's geometry wrong** (section 4.2), and its
   saturation rates are right to 0.5-5 % wherever a bump exists (H3G 159-164 measured against 165.0 predicted, H3EG
   229-236 against 241.8, F 150.6-155.8 against 144.7).

**Not established.** That a wedge-local recurrence *cannot* produce a compass here -- only that these two, at the
connectome's own EPG -> EPG weight and under this hold, do not. The off-block budget, not the tile, is what fails, and
no arm of this batch varies the inhibition that would fix it (a Delta7 gain, a partial hold, or a hold restricted to
the driven wedges would each be a new instrument and a new question). Nothing here measures the animal.

**Nothing is adopted, and nothing here could be.** Every arm that moved anything is two labelled instruments at once:
a counterfactual `edges` hold composed with a global hand-rule removal or a per-type gain. `--hold-edges` and
`--edge-gain` both default to `None`, the shipped CPU path is bit-identical with them absent (120 of 121 recorded
fields), `flyverse/brain.py` is untouched, and `compass.EPG.bump_survival_s` stays **FAIL** with the rate and width
rows **NOT_APPLICABLE** in all 40 runs.

**What a data-driven route to a wedge-local recurrence would look like** is unchanged by the measurement, and this
batch sharpens one half of it. The 842 EPG -> EPG synapses are data (EM chemical synapses, T-bar / PSD pairs); the
x0.1 is the hand rule; and the postsynaptic side is supported by the project's own tiering -- EPG is cholinergic and
the `EPG, acetylcholine` row of `flyverse/data/receptors_by_type.csv` is `fast_sign` **+1**, lead **nAChRbeta1**,
class **high**, tier `alias`, source **davis2020 / PB_2**. What is still unknown is whether E-PG cells are
**additionally electrically coupled**, which is the justification the `same_type_gain` docstring offers for the
cliques it was introduced against and which a chemical-synapse EM volume cannot see (no innexin column exists in the
connectome or the receptor table). The new half is this: **undamping EPG -> EPG alone is now known not to collapse the
compass into a saturated ring** -- H3F does that, H3E does not -- so the per-type route is the one worth costing, and
its cost is no longer a new code path (section 3.2) but the full benchmark suite at >= 3 draws on the CPU as well as
the GPU, because the x0.1 exists to stop runaway cliques elsewhere.

**Defects recorded this round.** (1) The generated `batch.sh` wrapper loses the client's exit code through an
unguarded `| tee` (section 5.1) -- present in cx5's, cx6's and cx7's generators. (2) `plan_batch` raised
`NameError: name 'sha' is not defined` on its final print, after writing `batch.sh` and `tree_state.json`; fixed.
(3) `measure_lif_sigma.py` first passed `control=` to `common.provenance`, which does not accept it; fixed before any
measurement was recorded. (4) A spiking cell's free-membrane SD is not a measurement of its input noise (reset clips
every firing ring cell to 2.1-2.4 mV), so a sigma estimate must restrict to sub-threshold cells. (5) This thread's own
predeclared H3E-vs-H3F contrast phrase is uninformative when both arms fail, and printing it verbatim would assert
something the data do not support (section 4.5) -- the same class of defect 6A's skeptic found in `call.csv`.

## Report

```yaml
summary: >
  Thread 6B ran THE TWO-INSTRUMENT ARM -- the 6A hold (ExR6 + ER6 + ER4m -> PEN, EPG at 0) plus a wedge-local
  recurrence -- at the shipped gains, after making two measurements the rate model needed. Batch cx7-201788 (house
  <cluster-node>, B200, 10 jobs, 0 failed, 40/40 runs, 0 provenance problems), 8 arms x 5 seeds, ONE submission.
  THE ANSWER IS NO, AND NOT FOR THE PREDICTED REASON: bump_survival_s 0.00 and frac_confined_post 0.000 in 5 of 5
  seeds in H3E, H3F, H3FG and H3EG, so the predeclared working-compass rule is met in 0 of 5 seeds in every arm and
  compass.EPG.bump_survival_s stays FAIL. But the two instruments fail oppositely. H3E (the hold plus a PER-TYPE
  undamping of the 842 EPG->EPG pairs) keeps the ring and builds the hump roughly where it was driven: in 4 of 5 seeds
  a seven-wedge hump centred 1.58-1.73 wedges from the driven block, 11 of 11 driven cells above 22 Hz, EPG in
  242.4-244.0 Hz against out 73.1-73.6, vector strength 0.67-0.68 -- missing the confinement rule ONLY on the
  off-block count (11-12 of 35 above 22 Hz against a budget of 3) and the tile rule only marginally. H3F (the same
  hold plus the GLOBAL same_type_gain 1) destroys the ring: 267.6-268.9 Hz inside and 282.9-283.4 outside, PEN
  278.0-278.4, 11 of 11 AND 35 of 35 cells above 22 Hz, vector strength 0.044-0.050, a flat profile across all 16
  wedges. So WHAT IS MISSING AT THE SHIPPED GAINS IS NOT EXCITATION AT THE DRIVEN TILE BUT INHIBITION OF THE REST OF
  THE RING -- removed by the very hold that let the tile fire -- and the global instrument is not a stronger version
  of the per-type one: turning on the PEN_a, PEN_b, PEG and Delta7 cliques as well is what takes the ring from "a hump
  in roughly the right place that is too wide" to "no ring at all". The predeclared H3E-vs-H3F contrast phrase prints
  "the per-type EPG->EPG recurrence is sufficient" and IS NOT A FINDING: the arms land in the same class because both
  failed, a case the rule's wording did not anticipate, and that is recorded as this thread's own instance of the
  defect 6A's skeptic found in call.csv. The GLNO relabel moves the geometry without buying a bump (H3EG on the driven
  tile in 2 of 5 seeds, centre 0.29 / 0.53 with 11 of 11 driven cells above 22 Hz, frac_confined_post 0.000 in 5/5).
  MEASUREMENT (a): the protocol now records per-type ring rates and per-cell maxima; shipped CPU path equal to the
  pre-6B run on 120 of 121 fields (wall_s excepted). MEASUREMENT (b): the LIF's effective input noise, never measured
  before -- sub-threshold relay membrane 4.6 mV, EPG 5.5, synaptic input 11.8, against the assumed SIGMA_MV = 2.0; the
  model should assume the bracket 4.6-11.8. Its consequence runs TOWARD 5A/6A: the maximum smoothed-f-I slope falls
  8.27 -> 6.06 -> 4.31 Hz/mV, so the shipped ring's gamma_E_crit 33.3 is above every operating point by 5.5x instead
  of 4.0x (sigma would have to be 0.17 mV, 27x smaller than measured, to be reachable), while the F-family recurrence
  at 3.33 stays supercritical across the bracket and its predicted rate moves 1.5 %; re-validated against cx5 at
  sigma 2.0 / 4.6 / 11.8 the tool is UNCHANGED (7 of 8 arms = 3 of 4 distinct predictions, 2 of 3 rate calls, at every
  sigma). THIRD RESULT: the rate model is 2.1-2.6x high on exactly the two types that carry the PEN DC (arm S
  post-pulse ExR6 41.3-42.8 Hz, ER6 23.1-23.9 against the fixed point's 107.7 / 51.9), closing 6A's named weak link.
  FOURTH: the per-type instrument needed NO brain.py change -- type_path_gain is applied BEFORE same_type_gain, so
  --edge-gain '^EPG$:^EPG$:10' lands the 842 pairs at exactly x1.0, bit-identical to same_type_gain=1 on that block,
  while the other cliques stay x0.1. THE TOOL RE-RUN AT THE MEASURED SIGMA called H3 and H3F right (H3's saturated
  ring replacing 6A's mis-predicted confined bump; H3F's 35 of 35) and got H3E's GEOMETRY WRONG in 4 of 5 seeds
  (it predicted the driven wedges pinned at the 10 Hz floor with the excess outside; the measurement is the mirror
  image), which is this round's headline tool miss. Nothing is adopted and nothing here could be: two labelled
  instruments make a mechanism statement, both new flags default to None, flyverse/brain.py is untouched.
key_claims:
  - "NO ARM HOLDS A BUMP AT THE DRIVEN TILE. bump_survival_s 0.00 and frac_confined_post 0.000 in 5 of 5 seeds in H3E, H3F, H3FG and H3EG; the predeclared working-compass rule (survival >= 5 s AND width 2.5-5 AND rate 5-60 Hz, in >= 3 of 5 seeds) is met in 0 of 5 seeds in EVERY arm of the batch, S and H3 included; the ledger rows stay FAIL / NOT_APPLICABLE in all 40 runs. The predeclared call for both H3E and H3F is 'no bump'."
  - "THE TWO INSTRUMENTS FAIL OPPOSITELY, AND THE PER-TYPE ONE IS FAR CLOSER. H3E: in 4 of 5 seeds a seven-wedge hump centred at wedge 3.08-3.23 (1.58-1.73 wedges from the driven centre 1.5), 11 of 11 driven cells above 22 Hz, EPG in 242.4-244.0 Hz against out 73.1-73.6, vector strength 0.67-0.68, PEN 57.3-62.9 -- failing confinement ONLY on the off-block count (11-12 of 35 above 22 Hz, budget 3). H3F: EPG 267.6-268.9 in / 282.9-283.4 out, PEN 278.0-278.4, GLNO 317-319, ExR6 332-333 at its ceiling, 11 of 11 AND 35 of 35 above 22 Hz, vector strength 0.044-0.050, profile flat to within 10 % across all 16 wedges. What is missing is inhibition of the rest of the ring, not excitation at the tile."
  - "THE PREDECLARED H3E-vs-H3F CONTRAST PHRASE IS NOT A FINDING. The rule words 'the per-type EPG->EPG recurrence is sufficient' for H3E landing in the same or a better class than H3F, and the analysis prints exactly that -- but they share a class because BOTH scored 0/5, which the rule's wording did not anticipate. Quoting it as a result would assert something the data do not support. The informative comparison is the primaries', and it points the other way: the per-type recurrence PRESERVES the ring's spatial structure (vs 0.67-0.68) and the global one DESTROYS it (vs 0.044-0.050)."
  - "THE GLNO RELABEL MOVES THE GEOMETRY WITHOUT BUYING A BUMP: H3EG puts the hump within 1.5 wedges of the driven block in 2 of 5 seeds (centre 0.29 and 0.53, 11 of 11 driven cells above 22 Hz, EPG in 228.8-236.2 Hz against out 38.0-39.9) and on the opposite side in the other three (centre 10.57-10.79), with frac_confined_post 0.000 in 5 of 5. H3FG is 11 of 11 in and 16-19 of 35 out in every seed, PEN 89.9-119.8 Hz, nothing confined."
  - "THE BISTABILITY IS PER SEED, NOT PER ARM, IN THREE ARMS: H3E splits 4 + 1 (centre 3.08-3.23 vs 11.39), H3EG splits 2 + 3 (0.29 / 0.53 vs 10.57-10.79), H3G splits 1 + 2 + 2 (0.92, 5.30 / 5.43, 10.01 / 10.03). Each ring settles on one of two or three mirror-image positions depending on the background realisation, as cx_wedge.md 6 found in the gain grid. No claim in this audit rests on a single seed."
  - "THE INPUT NOISE IS MEASURED: sigma_v of the sub-threshold relays 4.638 mV (median over 128 never-spiked PEN / PEG / EPGt cells, arm S seeds 0-1, range 2.733-7.692), sigma_v of the EPG 5.521 mV, sigma_g of the relays 11.800 mV -- against the assumed, never-measured SIGMA_MV = 2.0. The model should assume the bracket 4.6-11.8 mV (membrane-filtered to frozen-noise). A spiking cell's membrane SD is unusable (reset clips every firing ring cell to 2.1-2.4 mV whatever its rate), so only never-spiked cells enter the estimate."
  - "THE MEASURED SIGMA STRENGTHENS 5A / 6A RATHER THAN THREATENING THEM: the maximum slope of the smoothed f-I is 8.27 Hz/mV at sigma 2.0, 6.06 at 4.6 and 4.31 at 11.8, while gamma_E_crit is a property of the weights and does not move -- so the shipped ring's 33.34 Hz/mV is above every operating point by 5.5x at the measured sigma instead of 4.0x, and sigma would have to be 0.17 mV to make it reachable. The F-family / H3E-H3F recurrence at gamma_E_crit 3.33 stays supercritical across the whole bracket and its predicted saturation rate moves 1.5 % (144.3 / 144.7 / 146.5 Hz at sigma 2.0 / 4.6 / 11.8). Re-validated against the cx5 batch at all three sigmas the tool is unchanged: 7 of 8 arms (= 3 of 4 distinct predictions) and 2 of 3 rate calls at every sigma."
  - "THE RATE MODEL IS 2.1-2.6x HIGH ON THE TYPES THAT CARRY THE PEN DC, measured per type on the GPU batch and on the CPU: arm S post-pulse ExR6 41.3-42.8 Hz, ER6 23.1-23.9, ER4m 2.4-2.9 (5 seeds, out/cx7/analysis/ring_rates.csv) against the fixed point's 107.7 / 51.9 / 10.3 at background -- the same factor by which 6A found it overstates H3's PEN rate. Under H3 the held cells are not silenced, only their edges: ExR6 255.2-264.6 Hz, ER6 179.8-182.9, ER4m 90.1-105.6, GLNO 135.9-137.6. 6A's recording gap is closed for every arm."
  - "THE TOOL'S HEADLINE MISS THIS ROUND IS H3E's GEOMETRY. At the measured sigma it predicted the driven wedges pinned at the 10 Hz forced floor (EPG in 10.0 / out 164.7, 0 of 11 driven cells above 22 Hz) -- the mirror image of the measurement in 4 of 5 seeds (in 242-244 / out 73, 11 of 11). Only seed 4 matches it. The call (no confined bump) was right for the wrong reason, so a rate-model fixed point is not a validated predictor of bump POSITION on a bistable ring. Where a bump exists its rates remain good: H3G 165.0 predicted against 159.0-164.2 measured (0.5-3.8 %), H3EG 241.8 against 228.8-236.2 (2-5 %), F 144.7 against 150.6-155.8 (3.9-5.1 %). It also got H3 right where 6A had it wrong (RUNAWAY, 14 of 35 off-block above 22 Hz predicted, 17-19 measured) and H3F right including the 35 of 35, and it was WRONG on H3FG's PEN (predicted 0.0 Hz, measured 89.9-119.8)."
  - "A PER-TYPE WEDGE-LOCAL RECURRENCE NEEDED NO brain.py CHANGE: brain._shaped_weights applies type_path_gain BEFORE same_type_gain, so --edge-gain '^EPG$:^EPG$:10' composes with the shipped x0.1 to exactly x1.0 on the 842 EPG->EPG pairs -- bit-identical to same_type_gain=1 on that block because (n*10f)*0.1f == n in float32 for every count the cap leaves (1-60) -- while PEN_a, PEN_b, PEG and Delta7 cliques stay damped. Verified on the real connectome (EPG +3.666 mV/pair under both H3E and H3F; PEN_a +0.583, PEN_b +0.624, PEG +0.131, Delta7 -0.425 under H3E against +5.827 / +6.245 / +1.310 / -4.251 under H3F) and per run by the analysis (842 entries, all 842 same-type, effective factor 1.000). The opt-in LIFParams field the task allowed was not needed and not added."
  - "THE FAMILY WAS SATISFIABLE AND EVERY PEN ROW IS A RESULT: five primaries Holm-corrected within each of the 14 (arm, reference) families against BOTH S and H3, m = 5 in every one, Holm at the floor 5 x 0.0079 = 0.0397 <= 0.05. PEN_mean_during and PEN_mean_post are `result` for all seven arms against both references (H3F +277.80 Hz / z 1876 vs S and +234.27 / z 67.7 vs H3; H3E +59.21 / z 399.9 vs S and +15.68 / z 4.53 vs H3). centre_dist_t5 is a `result` for H3FG vs S (-2.54, z -3.78) and for H3F vs H3 (+1.93, z 11.3). H3E's centre_dist_t5 moves -2.83 wedges at z -4.21 and is NOT a result because the exact-U p is 0.095: four seeds move together and the mirror-image fifth does not, so the rank test cannot separate the arms at 5 v 5 -- INTERP.md 10.2's '|z| >= 3 AND p <= 0.05' doing exactly what it exists for."
  - "silent IS MEASURABLE NOW AND THE ANSWER IS WINDOW-DEPENDENT: in arm S the per-cell maximum over the post-pulse window is PEN 0.13-0.58 Hz, PEG 0.40-0.61, GLNO 0.02-0.39, EPGt 0.00 -- so EPGt and GLNO meet INTERP.md 10's silent (max < 0.5 Hz per cell) in 5 of 5 seeds while PEN and PEG STRADDLE the 0.5 Hz line across seeds. In the CPU settle window of the sigma runs no PEN, PEG, EPGt or GLNO cell spikes at all. No population with a non-zero recorded mean is called silent anywhere in this audit."
  - "REPRODUCIBILITY AND PROVENANCE: the four arms cx6 and cx7 share (S, H3, H3G, F; 20 runs) reproduce cx6 seed for seed with max |cx7 - cx6| = 0.0 over 760 (metric, run) pairs, all 38 recorded metrics x 20 runs. 40/40 consoles say device cuda; compiled-W md5 ef23cc27 on all 25 GLNO-silent runs and 7a10d93b on all 15 relabel runs; same_type_gain 1.0 in exactly F / H3F / H3FG; edge_gains 842 entries in exactly H3E / H3EG; analysis problems 0. Scheduler receipt: 10 job(s), 0 failed ({'completed': 10}), <cluster-node>, GPU ids 0 and 1, submitted 19:11:55Z, last finished 19:14:48Z."
  - "PROCESS, AND ONE DEFECT IN THE GENERATED WRAPPER: the batch was first submitted at 18:30:36Z and reached no cluster (house VPN down; '[house] UNAVAILABLE ... getaddrinfo failed / no target answered; nothing submitted'), then re-run UNCHANGED at 19:11:54Z -- batch.sh was never regenerated, so its stamped sha256 1552e602...1bfc still holds. The 'client exit 0' of the failed attempt was the WRAPPER's, not the client's: cluster_run.py returns 2 on both nothing-submitted branches, and batch.sh ends '| tee' with no pipefail, so cx5's, cx6's and cx7's generated wrappers all report the tee's status. The second submission used bash -o pipefail. This is INTERP.md 10.4 rule 4 applied one level up, to the submission wrapper."
files_written:
  - scripts/measure_lif_sigma.py (new; the per-step v / g / refrac / spikes record on the cx_wedge protocol, per-cell sigma_g / sigma_v_free with the never-spiked subset, --summarise pooling into sigma_summary.json + sigma_table.csv)
  - scripts/cx_wedge.py (ONE new flag --edge-gain PRE_REGEX:POST_REGEX:FACTOR, default None; parse_edge_gains; hold_edge_counts(..., same_type_gain); new RECORDED keys only: per-type ring rates ExR6 / ER6 / ER4m + EPGt as g__ groups, per-cell rates as cells__ arrays and <g>_cell_max_{pre,during,post} in metrics)
  - scripts/cx_ring_structure.py (sigma as an argument throughout, --sigma, --local-recurrence, the CX7 arm table and its five primaries, ring_distance / DRIVEN_CENTRE / DRIVEN_TILE_TOL, summarise_state's confinement count on the fixed point, analyse_batch_cx7, write_predeclaration_cx7, --predictions-csv, --batch cx7; the plan_batch NameError fixed)
  - tests/test_cx_wedge_hold.py (+6 tests, 12 passed), tests/test_cx_ring_structure.py (+9 tests, 22 passed)
  - docs/audits/compass_local_recurrence.md
  - out/cx7/sigma/* (3 runs, sigma_summary.json, sigma_table.csv, slope_vs_sigma.json)
  - out/cx7/structure/* at the MEASURED sigma 4.6 (structure.json, structure.md, matrices.npz, predictions.csv, validation_sigma{2.0,4.6,11.8}/); out/cx7/structure_sigma2/* at the assumed 2.0
  - out/cx7/{predeclared.json,batch.sh,tree_state.json,predeclare_stamp.txt,submit_stamp.txt,scheduler_receipt.json,scheduler_receipt.py}; out/cx7/smoke/* (CPU smokes, jobline_check.txt, analysis_pathcheck_cx6/)
  - out/cx7/<arm>_s<seed>.{json,txt,npz} x 40; out/cx7/analysis/{analysis.md,runs.csv,compare.csv,decision.csv,call.csv,scatter.csv,state.csv,ring_rates.csv,predictions_vs_measured.csv,cross_batch_cx6.csv,analysis.json,scatter.png}; out/cx7_cluster.log
api:
  - "cx_wedge.parse_edge_gains(['PRE:POST:FACTOR', ...]) -> [(pre, post, factor)]; cx_wedge.simulate(..., edge_gains=None) appends them to LIFParams.type_path_gain after the holds; the row records edge_gains and edge_gains_resolved (cells, entries, same-type entries, effective same-type factor). Default None = the previous gain list, entry for entry."
  - "cx_ring_structure.analyse(..., sigma=SIGMA_MV) and validate_batch(..., sigma=...): every slope bound, gain, fixed point and saturation rate is computed at that sigma and labelled with it; summarise_state adds in_above_22 / out_above_22 / confined_by_ledger_rule and rate_model adds confined_bump_after"
  - "cx_ring_structure: recurrence_configs, EPG_EPG_GAIN, CX7_ARMS, PRIMARIES_CX7, MAGNITUDES_CX7, SECONDARIES_CX7, ring_distance, DRIVEN_CENTRE, DRIVEN_TILE_TOL, analyse_batch_cx7 (two references, both rules, the four-way call, per-type ring rates, survival_at_tile_s from the .npz), write_predeclaration_cx7; CLI --sigma, --local-recurrence, --predictions-csv, --batch cx7"
  - "measure_lif_sigma: window_stats, summarise, summarise_dir; CLI --out (required) --label --hold-edges --edge-gain --lif --nt-override --save-traces --summarise"
validation:
  - "tests/test_cx_wedge_hold.py 12 passed, tests/test_cx_ring_structure.py 22 passed, tests/test_bit_identity.py 3 passed + 5 subtests -- 37 passed in one CPU run (PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m pytest -p no:cacheprovider). flyverse/ is untouched by this thread."
  - "shipped-path bit-identity on CPU: out/cx7/smoke/smoke_default_path_noledger.json equals the pre-6B out/cx6/smoke/smoke_default_path.json on 120 of 121 shared recorded fields (wall_s the only difference; edge_gains / edge_gains_resolved the only additions)"
  - "the per-type instrument on the real connectome: --edge-gain '^EPG$:^EPG$:10' resolves to 842 entries, 842 same-type, effective factor 1.000, and reproduces H3F's EPG->EPG block (+3.666 mV/pair, +67.1 per volley) with PEN_a / PEN_b / PEG / Delta7 left at the shipped damped values; checked per run on all 10 H3E / H3EG runs"
  - "the fixed tool against out/cx5 at sigma 2.0 / 4.6 / 11.8: 7 of 8 arms and 2 of 3 rate calls at every sigma (out/cx7/structure/validation_sigma*/validation.md)"
  - "job line tested on CPU before submission: exit expression 0 / 7 / 7 / 7, both flag specs intact through both quoting levels (out/cx7/smoke/jobline_check.txt)"
  - "predeclaration stamped 2026-09-15T18:29:58Z, never amended; batch.sh sha256 1552e602...1bfc recorded in it, unchanged across both submission attempts"
  - "scheduler receipt 10 job(s), 0 failed ({'completed': 10}), <cluster-node> / B200 / GPU ids 0-1; client '10 job(s), 0 failed (4.3 min) run dir /mnt/beegfs/neurome/runs/cx7-201788'; 40/40 json+txt+npz; 40/40 consoles 'device cuda'; analysis problems 0"
  - "cross-batch: S, H3, H3G, F reproduce cx6 with max |cx7 - cx6| = 0.0 over 760 (metric, run) pairs (out/cx7/analysis/cross_batch_cx6.csv)"
recommendations:
  - "DO NOT ADOPT ANYTHING FROM THIS THREAD. Every arm that moved anything is two labelled instruments at once (a counterfactual edges hold composed with a global hand-rule removal or a per-type gain). Both new flags default to None, the shipped CPU path is bit-identical, flyverse/brain.py is untouched, and compass.EPG.bump_survival_s stays FAIL with the rate and width rows NOT_APPLICABLE in all 40 runs."
  - "THE NEXT ARM IS ABOUT INHIBITION, NOT EXCITATION. H3E fails the confinement rule only on the off-block count (11-12 of 35 above 22 Hz against a budget of 3) while holding 11 of 11 driven cells and a vector strength of 0.67. The question that follows is which inhibition the hold removed that the ring needed: a Delta7 gain under H3E, a hold restricted to the driven wedges, or a partial hold (ExR6 only, which 6A showed carries most of the PEN DC) would each separate 'the brake is off everywhere' from 'the brake is off where it mattered'. That is one more two-instrument arm and still not an adoption."
  - "DO NOT USE THE FIXED POINT TO PREDICT BUMP POSITION. It got H3E's geometry mirror-imaged in 4 of 5 seeds while calling the arm correctly, and three arms of this batch are per-seed bistable. Its rates where a bump exists are good to 0.5-5 % and its bump / no-bump calls are 7 of 8 on cx5 -- position is the quantity it has now been shown to miss."
  - "THE RATE MODEL SHOULD ASSUME sigma IN 4.6-11.8 mV, NOT 2.0, and every audit quoting a slope bound should quote it at the measured value (max slope 6.06 Hz/mV at 4.6, 4.31 at 11.8, against 8.27 at 2.0). Nothing in 5A or 6A reverses; the shipped ring's 33.3 Hz/mV goes from 4.0x to 5.5x above every operating point. SIGMA_MV = 2.0 remains the code default and is now labelled an assumption with a measurement beside it."
  - "FIX THE GENERATED SUBMISSION WRAPPER: cx_ring_structure.plan_batch emits 'cluster_run.py ... | tee <log>' with no 'set -o pipefail', so batch.sh reports tee's exit code rather than the client's -- cx5's, cx6's and cx7's wrappers all have it. cluster_run.py itself is correct (it returns 2 when nothing is submitted). This is INTERP.md 10.4 rule 4 one level up."
  - "MAKE --out REQUIRED ON scripts/cx_ring_structure.py (6A's open item, still open). This thread never invoked it without --out, but the default out/cx5/structure remains the trap that caused the 6A incident."
  - "THE DATA QUESTION IS UNCHANGED AND NOW BETTER POSED: the 842 EPG->EPG synapses are EM chemical synapses and the postsynaptic side is supported by the project's own tiering (EPG is cholinergic; the EPG / acetylcholine row is fast_sign +1, nAChRbeta1, class high, tier alias, davis2020 PB_2), so the x0.1 is the only thing between them and the model. What would license removing it FOR THIS TYPE is evidence about electrical coupling -- whether E-PG are gap-junction coupled (innexin expression), which this chemical-synapse dataset cannot see -- plus the full benchmark suite at >= 3 draws on the CPU as well as the GPU. What the batch adds: undamping EPG->EPG alone does NOT collapse the compass into a saturated ring (H3F does that, H3E does not), so the per-type route is the one worth costing."
open_questions:
  - "Which inhibition does the ring need that the hold removed? H3E holds 11 of 11 driven cells above 22 Hz and fails only on the 11-12 of 35 off-block cells. Nothing in this batch varies the inhibition, so the answer is open and is the obvious next arm."
  - "Why does the fixed point mirror-image H3E's bump? Three arms are per-seed bistable (H3E 4+1, H3EG 2+3, H3G 1+2+2) and the rate model picks one branch by its initial condition. A basin analysis, or seeding the fixed point from the driven state, would say whether the model can be made to track position at all."
  - "What is the right scalar sigma inside the measured 4.6-11.8 mV bracket? The membrane and input readings differ by 2.6x, the per-cell spread is 2.7-7.7 mV, and the hold changes both (relay membrane 2.5-3.2 mV under H3). A calibration against the LIF's own measured f-I on these cells would settle it."
  - "Why is the rate model 2.1-2.6x high on the ring types? Now measured on 40 runs rather than conjectured, but not explained -- the candidates are the adiabatic relay assumption, the missing adaptation of the ring cells in the fixed point, and the fan-in / cap treatment of their giant edges."
  - "What are ExR6's and ER6's transmitter, receptor and modulatory status in the animal? Unchanged from 6A: UNKNOWN at the type level, and still the largest single term in the balance all three rounds are about."
  - "Are E-PG cells electrically coupled? This is what the same_type_gain x0.1 rests on for the compass, and a chemical-synapse EM volume with no innexin column cannot answer it."
  - "Does survival_at_tile_s belong in the ledger? It separates H3G's 4.76-5.00 s of survival from its 0.00-4.76 s at the driven tile, which compass.EPG.bump_survival_s cannot -- a rule change that belongs to whoever owns expected_responses.csv."
```
