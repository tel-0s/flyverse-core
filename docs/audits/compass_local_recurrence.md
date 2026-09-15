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

**No. The hold plus either recurrence does not produce a working compass at the shipped gains.**
`survival_s` and `frac_confined_post` are zero in all five runs of H3E, H3F, H3EG and H3FG.
Every arm has zero runs meeting the joint survival, width and rate rule (`analysis/call.csv`, section 4.5).
This does not mean every ledger row fails: F seeds 0-3 and H3G seed 0 pass survival and width but fail rate;
the other 35 runs fail survival with width/rate NOT_APPLICABLE (`analysis/runs.csv`, `ledger_*`).

**H3E produces a broad, high-rate hump; H3F produces a nearly flat saturated profile.** In H3E seeds 0-3,
the final profile has seven wedges above half-maximum (the working-compass width range is 2.5-5),
with a systematic +1.58-1.73-wedge displacement from the driven centre (limit 1.5). All 11 driven cells
exceed 22 Hz, but so do 11-12 off-block cells (budget 3). Whole-post-window EPG means are
242.4-244.0 Hz inside and 73.1-73.6 outside, not the final-frame rates; they also exceed the compass's
5-60 Hz rate range. H3F's corresponding population means are 267.6-268.9 and 282.9-283.4 Hz, with final
vector strength 0.044-0.050 versus H3E's 0.67-0.68 (`analysis/runs.csv`, `epg_*_mean_post`,
`centre_dist_t5`, `vs_t5`; `analysis/state.csv`, `profile_end`; section 4.1).

**Withdrawn:** "what is missing is inhibition of the rest of the ring, not excitation at the tile."
The skeptic found the far half at 8.7-12.3 Hz in H3E seeds 0-3, near background; the off-block excess is
mainly adjacent to the driven block. Per-type recurrence makes that far half quieter than H3 while
Delta7 fires more, so this batch does not isolate a missing-inhibition mechanism. Both the hold and
the EPG gain act on driven and off-block cells. H3E already has a structured high-rate profile before
the pulse; the pulse relocates it in four runs rather than creating it from a quiet ring (section 4.1).

The predeclared H3E-vs-H3F sufficiency phrase is uninformative when both arms fail and is not a finding
(section 4.5). H3EG changes placement without producing confinement: two runs land within the tile
bound, three do not (`analysis/scatter.csv`, `centre_wedge_t5` and `centre_dist_t5`). The rate model
gets H3E's geometry wrong in four runs; the 0.5-7.1% saturation-rate error range does not validate its
position prediction (sections 4.2 and 6).

**The two missing measurements are now recorded.** First, never-spiked relay cells have membrane SD
4.638 mV in the two shipped CPU settle runs, versus the assumed 2 mV; synaptic-input SD is 11.800 mV.
The latter is a quasi-static sensitivity assumption, not an established upper bound for the f-I sigma.
The EPG 5.521 mV reading is from spiking cells and is not a clean sub-threshold noise estimate
(`sigma/sigma_summary.json`, section 2.1). Second, arm S's GPU post-window means are ExR6 41.3-42.8,
ER6 23.1-23.9 and ER4m 2.4-2.9 Hz (`analysis/ring_rates.csv`); the fixed point overestimates the two
major PEN-brake populations (section 3.3).

Nothing is adopted. The new flags default to `None`; the CPU smoke agrees on 120 shared non-timing
fields (the 121st is `wall_s`). Forty run JSONs and consoles were verified on CUDA; the shared cx6/cx7
measurements reproduce the recorded values exactly in these batches, without establishing general
GPU determinism (section 5). ExR6 glutamate and ER6 GABA now have direct expression support;
their EB/GA receptor placement and kinetics remain open (`exr6_evidence.md`).

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
Both compared files are non-ledger runs. No pre-6B CPU ledger counterpart was saved; the ledger-path argument
is that the additional `Brain.rates`/`mean_rate` calls only read the existing EMA state and draw no RNG.

### 1.2 The instrument flag

`--edge-gain PRE_REGEX:POST_REGEX:FACTOR` (repeatable, default `None`) appends `(pre, post, factor)` to
`LIFParams.type_path_gain` -- the same stage the 6A hold and the gE / gD gains use. With the flag absent the installed
gain list is entry-for-entry the previous one (tested). It is a **LABELLED INSTRUMENT** (a per-type gain), never a
candidate default.

Tests (CPU, `tests/test_cx_wedge_hold.py` 12 passed, `tests/test_cx_ring_structure.py` 22 passed, with
`tests/test_bit_identity.py` 37 passed in one run): the
parser accepts `PRE:POST:FACTOR` and rejects a missing factor, a non-numeric factor, a negative factor, an empty
regex and a bad regex; the flag off is bit-identical; `^EPG$:^EPG$:10` under `same_type_gain` 0.1 reproduces
`same_type_gain=1` **bit for bit on the shaped EPG -> EPG block**. On the real installed matrix,
H3E versus H3 changes exactly those 842 entries and no others. This does not generalize to the no-hold E
configuration: its fan-in scale changes 1,353 non-EPG->EPG entries (skeptic section 4).
`(n*10f)*0.1f == n` for every integer count the cap leaves (1-60); the resolved record reports 842 entries,
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
  `v_rest + g + drive - adapt`, so for undriven relays `g` supplies the rate model's input variable. Driven EPG
  instead receives forced Poisson spikes, not an added current; its `g` is not the rate model's full `u`.
  This is the
  **frozen-noise** reading: the width the f-I would be smoothed over if the input were quasi-static on the scale of an
  interspike interval.
* **sigma_v** -- the SD of the **free** membrane potential (samples outside the refractory period), i.e. the same
  noise after the membrane's `tau_m` = 20 ms low-pass. This is the **fast-noise** reading.
* both restricted to the cells that **never spiked** in the window -- the cleanest estimate the protocol offers,
  because a spiking cell's membrane SD is clipped by reset-to-threshold. Over 98 firing ring-cell/window
  readings the skeptic found 1.041-2.909 mV (median 2.108; 45% in 2.1-2.4), with rate dependence.
  **Withdrawn:** "every firing ring cell reads 2.1-2.4 mV whatever its rate."

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
(92 spiking-cell readings, zero never-spiked; **not usable as a sub-threshold noise estimate**),
**sigma_g of the relays 11.800 mV**, **sigma_g of the EPG 12.456 mV**.

**What the measurement supports.** The defensible membrane value is **4.6 mV** in this shipped CPU state.
**Withdrawn:** 4.6-11.8 mV as an established bracket for the smoothed f-I sigma. The 11.8 mV input reading
only supplies a quasi-static sensitivity scenario: sigma_v/sigma_g = 0.393 is near
sqrt(tau_syn/(tau_syn+tau_m)) = sqrt(5/25) = 0.447, and the input correlation time is four times shorter
than the membrane time, not frozen. The tool was evaluated at **2.0, 4.6 and 11.8** without changing its
default. Measurements cover two CPU seeds of S and one of H3. Under H3, relay group medians are
2.5-3.2 mV on the never-spiked membrane and 11.0-14.7 mV on the input over all cells (EPGt included;
12.0-14.4 restricted to never-spiked cells). Across S cells the membrane SD spans 2.7-7.7 mV.
The transfer from these state-dependent readings to a scalar f-I sigma remains a modelling assumption.

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
quadrature at sigma 2 (8.27 vs 8.00); the independent skeptic checked all three by adaptive quadrature: 7.9969 / 5.9297 / 4.2496 Hz/mV,
so the respective overestimates are 3.47 / 2.13 / 1.31 %. The bias shrinks with sigma; no sign changes.

Two consequences, and they point the same way:

* **The shipped-ring conclusion is strengthened, not threatened.** `gamma_E_crit` is a property of the weights alone
  (33.34 Hz/mV shipped, 28.77 under the receptor tier) and does not move with sigma; the bound it is compared against
  falls. At the measured sigma the shipped ring's critical gain is above every operating point of the smoothed f-I by
  **5.5x** (33.34 / 6.06) instead of 4.0x, and by 7.7x at the frozen-noise end. **The measured sigma cannot rescue the
  shipped compass**; sigma would have to be **0.17 mV**, 27x smaller than measured, for 33.3 Hz/mV to be reachable.
* **The F-family / H3E-H3F recurrence stays supercritical and its predicted rate is sigma-insensitive.**
  `gamma_E_crit` 3.33-3.34 is below the maximum slope at each tested sigma, and the predicted saturation rate
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

**The scored validation calls did not move**: the same 7 of 8 arms (which 6A's skeptic correctly reduced to **3 of 4 distinct
predictions**, because `gamma_E_crit` takes four values over eight arms and the classifier is "does this arm carry
`same_type_gain` 1?"), the same single miss (FG), the same 2 of 3 rate calls. What *did* move is the H3 family's fixed
point, which is the subject of section 4: at sigma 4.6 the driven EPG's own gain at the realised state changes enough
to turn 6A's predicted confined bump into a predicted saturated ring -- the direction the measurement had already
gone. The full 40-row (configuration x sigma) comparison is `out/cx7/structure/predictions.csv`. The unscored `predicted_bump_with_relays` does move: True -> False for all four
F-family arms from sigma 2.0 to 4.6, with `saturation_with_relays` 105.7 Hz -> NaN.

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

That is the separation 5A and 6A both wanted and neither could express: within this ring matrix, H3E and H3F differ in the four
non-EPG same-type cliques listed above. The global gain also applies to same-type cliques elsewhere in the brain. The wedge-local one-step EPG -> EPG coefficient goes from +6.00 mV (shipped) to
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
| S, background | ExR6 | 110.2 Hz | 107.7 Hz | **41.9-43.1 Hz** | **2.6x at sigma 2.0** |
| S, background | ER6 | 53.8 | 51.9 | **22.8-24.4** | **2.2-2.4x at sigma 2.0** |
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
it says so. The word `silent` uses the stated window and the project
threshold, and does not imply exactly zero spikes in other windows. PEG also has 4 firing cells during the pulse;
EPGt remains at zero. In the GPU post window, PEN and PEG meet the per-cell < 0.5 Hz criterion in 4/5 seeds each,
and GLNO and EPGt in 5/5 (`runs.csv`, `<TYPE>_cell_max_post`).

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
| F | 0.0 / **22.5-27.4** | 10.0 / 10.0 vs **59.4-155.2 / 9.6-11.2** | 0, 0 / **2-9, 1-3** | no fixed-point bump; recurrence 144.7 Hz | 4/5, 4/5 | rate **right to 3.9-7.1 %** (meas 150.6-155.8); the fixed point misses the bump, as in 5A / 6A |
| H3G | 36.0 / **26.5-29.0** | 165.0 / 11.6 vs **17.7-160.7 / 14.3-54.8** | 11, 2 / **0-11, 5-16** | BUMP, CONFINED | 1/5, 1/5 | rate **right to 0.5-3.8 %** (meas 159.0-164.2); confinement right in 1 of 5 seeds only |
| H3FG | 0.0 / **89.9-119.8** | 10.0 / 149.5 vs **226.2-272.3 / 100.0-142.6** | 0, 18 / **11, 16-19** | RUNAWAY, bump off the tile | 0/5, 0/5 | **wrong on PEN** (predicted silent, measured 90-120 Hz); right that nothing is confined |
| H3EG | 46.3 / **27.7-37.1** | 241.8 / 39.4 vs **10.2-236.2 / 38.0-126.1** | 10, 6 / **0-11, 8-22** | BUMP, not confined | 0/5, **2/5** | rate **right to 2.4-5.7 %** in the 2 seeds that land on the tile; geometry 2 of 5 |

### 4.1 The answer to the thread's question: no, and not for the predicted reason

H3E, H3F, H3EG and H3FG have `survival_s` 0.00 and `frac_confined_post` 0.000 in every seed.
All eight arms have **0/5 joint working-compass successes**. The individual ledger rows are different:
survival is 35 FAIL / 5 PASS, width 35 NOT_APPLICABLE / 5 PASS, and rate 35 NOT_APPLICABLE / 5 FAIL.
F seeds 0-3 and H3G seed 0 pass survival and width but fail rate (`analysis/runs.csv`).

* H3F has post-window population means **267.6-268.9 Hz in / 282.9-283.4 Hz out** and PEN during
  **278.0-278.4 Hz**. Its final profile has all 46 EPG above 22 Hz and vector strength 0.044-0.050.
  It is nearly flat: `(max-min)/mean` across 16 wedges is within **15 %**, not 10 % (maximum 14.7 %).
  Final-profile means are 262.9-270.4 in / 282.8-284.4 out. These are different windows.
* H3E seeds 0-3 have **seven wedges above half the final-profile maximum**, against a width target of 2.5-5.
  Their centres are 3.08-3.23, a **systematic +1.58-1.73 wedge offset** from the driven centre 1.5,
  outside the 1.5-wedge bound. Post-window population means are **242.4-244.0 in / 73.1-73.6 out**;
  final-profile means are 239.4-243.8 / 71.0-73.5. Rate also exceeds the 5-60 Hz target, and 11-12 off-block
  cells exceed 22 Hz against a budget of 3. Seed 4 instead centres at 11.39. This is a wide, high-rate hump,
  not a failure on only the off-block count and not a marginal compass success.

**Withdrawn:** "what is still missing is not excitation at the tile but inhibition everywhere else."
In H3E seeds 0-3 the far half (wedges 8-13, 17 cells, at least five wedges from the driven centre)
averages **8.7-12.3 Hz**, near the 10 Hz background, with only 0-2 cells above 22 Hz. Most excess lies in
adjacent wedges 4-7. Compared with H3, recurrence lowers far-half activity from 31.2-46.4 Hz and the
off-block count from 17-19 to 11-12 while Delta7's post-window mean rises from 120.9-124.0 to 165.7-170.4 Hz.
Thus recurrence recruits inhibition and quiets the off-block ring. Both the hold and the 842-pair gain
act on driven and off-block cells; no arm isolates inhibition of the rest of the ring.

H3E is already structured before the pulse in all five seeds (`pre_vector_strength` 0.65-0.69,
`pre_out_mean` 94.6-153.2 Hz): the pulse moves an existing high attractor. It does not build a hump
from a quiet ring. Sources: `analysis/state.csv`, `analysis/runs.csv`, raw `H3E_s<seed>.json` and its
`ledger_npz` (for example `H3E_s0_gE1_gD1_s0.npz`); the independent skeptic's sections 10-12 below.

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
as predeclared. This controls each family separately, **not the whole round** of 70 rows. Pooling both
references (m=10) would be unsatisfiable at 5 v 5. `bump_hz_post` and `width_half_post` returned no p in every family (neither reference has a confined
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

`PEN_mean_during` is `result` in 14/14 contrasts; `PEN_mean_post` in **13/14**: H3E vs H3 is `null`
(+4.13 Hz, z 2.16). `centre_dist_t5` has **four** results: H3FG vs S; S vs H3 (+0.870, z 5.12);
H3F vs H3; and H3FG vs H3. These are read from `analysis/compare.csv`, including the reverse-reference S row.

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
**2 + 3** (centre 0.29 / 0.53 vs 10.57-10.79) and H3G splits 1 + 2 + 2 (0.92, then 5.30 / 5.43, then 10.01 / 10.03; **sorted**, not seed order,
`analysis/runs.csv`, `centre_wedge_t5`). H3EG's two near centres above belong to seeds 2 and 3, respectively;
its far-centre range is sorted (`analysis/scatter.csv`, `centre_wedge_t5`).
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
strength **0.67-0.68** with a wide, systematically offset hump, H3F **0.044-0.050** and flat at ~280 Hz; H3E's PEN at
57-63 Hz against H3F's 278 Hz; H3E's off-block EPG at 73 Hz against H3F's 283 Hz. The global gain, which additionally changes PEN_a, PEN_b, PEG,
Delta7 and other same-type cliques, takes the ring from "a hump in roughly the right place that is too wide" to "no
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
analysis. Five of the 15 `tree_state.json` source hashes no longer matched by the successful submission:
`scripts/cx_ring_structure.py`, `scripts/measure_lif_sigma.py`, `tests/test_cx_ring_structure.py`,
`scripts/probe_vnc_drive.py`, and `scripts/derive_level_fixed_point.py`. The simulated path
`scripts/cx_wedge.py` matched. The sigma summary/table were regenerated at 18:42:30Z and
structure predictions at 18:42:35Z, after the stamp; the skeptic reproduced the slope values.
This limits the source stamp: it does not describe an immutable full analysis tree. **`batch.sh` was not regenerated between the two attempts**: the sha256 at submission is the one stamped in
the predeclaration.

**Submission.** ONE `cluster_run.py --name cx7 --minutes 30 --arm-block fam <10 job lines> --fetch out/cx7/` call,
10 jobs = 5 seeds x {5 GLNO-silent arms sequential, 3 relabel arms sequential}, blocks `fam_s<seed>` (every arm of one
seed on one target, so no experimental factor is collinear with a box).
**The failure line: `10 job(s), 0 failed  (4.3 min)  run dir <cluster-run-root>/cx7-201788`, `client exit 0`**;
scheduler receipt (`out/cx7/scheduler_receipt.json`, queried from the job manager itself): **`10 job(s), 0 failed
({'completed': 10})`**, all on **<cluster-node>**, GPU ids 0 and 1, NVIDIA B200, first submitted 19:11:55.46Z, last finished
19:14:48.92Z. (The job manager records `exit_code: None` for completed jobs, so the exit-code half of `INTERP.md`
10.4 rule 21 rests on the status field and on the per-run checks below.)

**Artefacts: 40 `<arm>_s<seed>.json`, 40 `.txt`, 40 ledger `.npz`**
(the NPZ names come from each JSON's `ledger_npz`, e.g. `H3E_s0_gE1_gD1_s0.npz`), 5 per arm x 8 arms; **40 of 40 consoles say
`device cuda`**; the analysis's per-run checks raise **0 problems** over device, arm label, gains, `nt_override` /
`glno_nt`, `same_type_gain`, `receptor_model`, **the hold** (spec, factor 0, exactly 1,149 entries) and **the edge
gain** (spec, exactly 842 entries, **all 842 same-type**, effective factor 1.000), plus the presence of every new
per-type ring rate. Compiled-W md5 `ef23cc27bea13be7f6a96f3c04fd3737` on all 25 GLNO-silent runs and
`7a10d93ba2086f2c76bcdabdca79b4ec` on all 15 relabel runs -- the same two hashes as cx5, cx6 and this desktop.
`same_type_gain` is 1.0 in exactly F / H3F / H3FG and 0.1 in the other five arms; `edge_gains` matches 842 entries in
exactly H3E / H3EG. Wall 13.7-27.5 s per run.

**Cross-batch reproduction.** The four arms cx6 and cx7 share (S, H3, H3G, F; 20 runs) reproduce cx6 seed for seed:
**max |cx7 - cx6| = 0.0 over 760 (metric, run) pairs** (`out/cx7/analysis/cross_batch_cx6.csv`, all 38 recorded
metrics x 20 runs, NaN-vs-NaN counted as equal). **Withdrawn:** a general determinism claim from this repetition. These recorded values reproduce exactly on this
backend; that does not establish GPU determinism for other configurations or machines. The skeptic's broader
comparison also found zero differences over 6,105 numeric pairs, with 50 NaN pairs matching.

### 5.1 The first submission attempt, and the defect it exposed

The batch was first submitted at **18:30:36Z** and reached no cluster: the house VPN was down on this machine, so
`<cluster-node>` did not resolve. **The following is the author's recollection, not a surviving console quotation.**
The successful attempt overwrote `client_stdout.txt`, `cx7_cluster.log` and `submit_stamp.txt`; the two
logs now have mtime 19:12:03Z and contain no `UNAVAILABLE` line:

```
[house] UNAVAILABLE (API silent for 20s): <urlopen error [Errno 11001] getaddrinfo failed>
no target answered; nothing submitted
client exit 0
```

Nothing was submitted, so nothing was fetched and nothing needed recovering; the batch was re-run **unchanged** at
19:11:54Z once the VPN was up, which is why `batch.sh`'s stamped hash still holds. **The `client exit 0` was the
wrapper's, not the client's** (owner's correction, 2026-09-15): `cluster_run.py` returns **2** on both
nothing-submitted branches, and `batch.sh` ends `... | tee out/cx7/client_stdout.txt`, so without `pipefail` the
pipeline exits with `tee`'s status. The author reports running the second submission with `bash -o pipefail out/cx7/batch.sh`;
that invocation also has no surviving command record. **The defect
is in the generated wrapper**: `cx_ring_structure.plan_batch` emits a `| tee` line with no `set -o pipefail`, so every
`batch.sh` this project has generated -- cx5's, cx6's and cx7's -- reports the tee's exit code rather than the
client's. That is `INTERP.md` 10.4 rule 4 ("a job line must preserve the python exit code") applied one level up, to
the submission wrapper itself, and it is the one process defect this round found.

## 6. What this establishes, what it does not

1. **The joint working-compass result is NO, 0/5 in every arm.** H3E, H3F, H3EG and H3FG have zero
   survival and confinement in all seeds. Five F/H3G runs pass the individual survival and width ledger
   rows and fail rate; the other 35 fail survival with rate and width not applicable.
2. **H3E and H3F fail differently.** H3E has a seven-wedge, high-rate, systematically offset hump in
   seeds 0-3; H3F is nearly flat at about 280 Hz. **Withdrawn:** "not local excitation at the driven tile
   but inhibition of the rest of the ring". The far half is already near background, excess is adjacent,
   and H3E recruits Delta7 and reduces off-block activity relative to H3. Width, rate and alignment all
   fail; no causal rest-of-ring inhibition experiment was done (section 4.1).
3. **Per-type ring rates are now measured.** S post-pulse ExR6 41.3-42.8 Hz and ER6 23.1-23.9 are below
   the sigma-4.6 fixed point's 107.7 and 51.9 Hz. The CPU settle comparison at sigma 2.0 gives 2.6x for
   ExR6 and 2.2-2.4x for ER6. The recording gap is closed, but the cause of the rate-model error is open.
4. **Silence is window- and threshold-dependent.** GPU post-window per-cell maxima are PEN 0.13-0.58 Hz,
   PEG 0.40-0.61, GLNO 0.02-0.39 and EPGt 0.00: threshold < 0.5 Hz holds in 4/5, 4/5, 5/5 and 5/5 seeds.
   CPU settle cells of all four groups never spike. These statements do not imply literal zero firing
   in the other windows (`runs.csv`, `<TYPE>_cell_max_post`; `sigma_table.csv`, `never_spiked`).
5. **The fixed point misses H3E's geometry in 4/5 seeds**, despite correctly predicting no confined bump.
   Saturation-rate errors where a bump is measured span **0.5-7.1 %**: H3G 0.5-3.8 %, H3EG 2.4-5.7 %,
   F 3.9-7.1 %. Position remains unvalidated in this multistable ring.

These interventions do not establish that local recurrence can never yield a compass. Nor do they identify
a missing inhibition to add. Future partial holds or inhibitory gains are separate, predeclared experiments.
The 11.8 mV input-noise scenario is a quasi-static assumption, not an established upper bound; the defensible
membrane measurement is 4.6 mV in shipped S, with strong state dependence and a calibration still needed.

**Nothing adopted.** These are labelled counterfactual arms and combinations (F and H3 are single interventions).
Both flags default to None; `flyverse/brain.py` is untouched. The CPU non-ledger comparison matches 120
non-timing fields, and the full closeout CPU suite passes **441 tests / 19 skipped, 215 subtests** including
the bit-identity gate. MaleCNS cache MD5s are unchanged. The new ledger reads do not advance the RNG;
there is no saved pre-6B CPU ledger counterpart. Exact cx6/cx7 recorded-value reproduction has the scope in section 5.

EPG's chemical self-connections are data; the x0.1 gain is a model rule. The existing EPG/acetylcholine receptor
row is fast_sign +1, nAChRbeta1, high, alias tier, Davis 2020 PB_2. Electrical coupling and the appropriate
self-gain are not measured by this experiment. H3E avoids H3F's near-uniform saturation but remains excessively
active; no adoption is supported without biological evidence and the required full suite and room checks.

**Superseded data question:** ExR6 and ER6 transmitter identity is no longer wholly unknown. The merged
`exr6_evidence.md` retains glutamate and GABA with type-linked EASI-FISH support. Receptor placement and
kinetics at EB/GA contacts, and co-transmission, remain unresolved; nothing from that audit is adopted here.

**Process defects retained.** The generated submission wrapper lacks pipefail; the `plan_batch` final-print
NameError and unsupported `control=` provenance argument were fixed before these outputs. `cx_ring_structure.py`
still has a hazardous default output directory; always pass `--out`. The false sigma range, generic ledger failure,
inhibition inference, and uninformative H3E-vs-H3F decision phrase are explicitly corrected above.

## Report

```yaml
summary: |-
  **No. The hold plus either recurrence does not produce a working compass at the shipped gains.**
  `survival_s` and `frac_confined_post` are zero in all five runs of H3E, H3F, H3EG and H3FG.
  Every arm has zero runs meeting the joint survival, width and rate rule (`analysis/call.csv`, section 4.5).
  This does not mean every ledger row fails: F seeds 0-3 and H3G seed 0 pass survival and width but fail rate;
  the other 35 runs fail survival with width/rate NOT_APPLICABLE (`analysis/runs.csv`, `ledger_*`).

  **H3E produces a broad, high-rate hump; H3F produces a nearly flat saturated profile.** In H3E seeds 0-3,
  the final profile has seven wedges above half-maximum (the working-compass width range is 2.5-5),
  with a systematic +1.58-1.73-wedge displacement from the driven centre (limit 1.5). All 11 driven cells
  exceed 22 Hz, but so do 11-12 off-block cells (budget 3). Whole-post-window EPG means are
  242.4-244.0 Hz inside and 73.1-73.6 outside, not the final-frame rates; they also exceed the compass's
  5-60 Hz rate range. H3F's corresponding population means are 267.6-268.9 and 282.9-283.4 Hz, with final
  vector strength 0.044-0.050 versus H3E's 0.67-0.68 (`analysis/runs.csv`, `epg_*_mean_post`,
  `centre_dist_t5`, `vs_t5`; `analysis/state.csv`, `profile_end`; section 4.1).

  **Withdrawn:** "what is missing is inhibition of the rest of the ring, not excitation at the tile."
  The skeptic found the far half at 8.7-12.3 Hz in H3E seeds 0-3, near background; the off-block excess is
  mainly adjacent to the driven block. Per-type recurrence makes that far half quieter than H3 while
  Delta7 fires more, so this batch does not isolate a missing-inhibition mechanism. Both the hold and
  the EPG gain act on driven and off-block cells. H3E already has a structured high-rate profile before
  the pulse; the pulse relocates it in four runs rather than creating it from a quiet ring (section 4.1).

  The predeclared H3E-vs-H3F sufficiency phrase is uninformative when both arms fail and is not a finding
  (section 4.5). H3EG changes placement without producing confinement: two runs land within the tile
  bound, three do not (`analysis/scatter.csv`, `centre_wedge_t5` and `centre_dist_t5`). The rate model
  gets H3E's geometry wrong in four runs; the 0.5-7.1% saturation-rate error range does not validate its
  position prediction (sections 4.2 and 6).

  **The two missing measurements are now recorded.** First, never-spiked relay cells have membrane SD
  4.638 mV in the two shipped CPU settle runs, versus the assumed 2 mV; synaptic-input SD is 11.800 mV.
  The latter is a quasi-static sensitivity assumption, not an established upper bound for the f-I sigma.
  The EPG 5.521 mV reading is from spiking cells and is not a clean sub-threshold noise estimate
  (`sigma/sigma_summary.json`, section 2.1). Second, arm S's GPU post-window means are ExR6 41.3-42.8,
  ER6 23.1-23.9 and ER4m 2.4-2.9 Hz (`analysis/ring_rates.csv`); the fixed point overestimates the two
  major PEN-brake populations (section 3.3).

  Nothing is adopted. The new flags default to `None`; the CPU smoke agrees on 120 shared non-timing
  fields (the 121st is `wall_s`). Forty run JSONs and consoles were verified on CUDA; the shared cx6/cx7
  measurements reproduce the recorded values exactly in these batches, without establishing general
  GPU determinism (section 5). ExR6 glutamate and ER6 GABA now have direct expression support;
  their EB/GA receptor placement and kinetics remain open (`exr6_evidence.md`).
key_claims:
- Every arm has 0/5 joint working-compass successes; individual ledger survival and width pass in five runs that
  fail rate.
- 'Withdrawn: missing inhibition of the rest of the ring. H3E has excessive width, rate and offset; its far half
  is near background and quieter than H3.'
- The membrane sigma measurement is 4.638 mV in 128 never-spiked relay cells; EPG 5.521 is a spiking-cell reading;
  11.800 input sigma is a quasi-static sensitivity scenario, not a validated bracket.
- PEN during is result in 14/14 contrasts, PEN post in 13/14. Centre distance has four results. Holm m=5 is per
  (arm, reference), not round-wide.
- The fixed point misses H3E placement in 4/5 seeds. Measured bump rate errors span 0.5-7.1%.
files_written:
- scripts/measure_lif_sigma.py (new; the per-step v / g / refrac / spikes record on the cx_wedge protocol, per-cell
  sigma_g / sigma_v_free with the never-spiked subset, --summarise pooling into sigma_summary.json + sigma_table.csv)
- '{''scripts/cx_wedge.py (ONE new flag --edge-gain PRE_REGEX:POST_REGEX:FACTOR, default None; parse_edge_gains;
  hold_edge_counts(..., same_type_gain); new RECORDED keys only'': ''per-type ring rates ExR6 / ER6 / ER4m + EPGt
  as g__ groups, per-cell rates as cells__ arrays and <g>_cell_max_{pre,during,post} in metrics)''}'
- scripts/cx_ring_structure.py (sigma as an argument throughout, --sigma, --local-recurrence, the CX7 arm table
  and its five primaries, ring_distance / DRIVEN_CENTRE / DRIVEN_TILE_TOL, summarise_state's confinement count on
  the fixed point, analyse_batch_cx7, write_predeclaration_cx7, --predictions-csv, --batch cx7; the plan_batch NameError
  fixed)
- tests/test_cx_wedge_hold.py (+6 tests, 12 passed), tests/test_cx_ring_structure.py (+9 tests, 22 passed)
- docs/audits/compass_local_recurrence.md
- out/cx7/sigma/* (3 runs, sigma_summary.json, sigma_table.csv, slope_vs_sigma.json)
- out/cx7/structure/* at the MEASURED sigma 4.6 (structure.json, structure.md, matrices.npz, predictions.csv, validation_sigma{2.0,4.6,11.8}/);
  out/cx7/structure_sigma2/* at the assumed 2.0
- out/cx7/{predeclared.json,batch.sh,tree_state.json,predeclare_stamp.txt,submit_stamp.txt,scheduler_receipt.json,scheduler_receipt.py};
  out/cx7/smoke/* (CPU smokes, jobline_check.txt, analysis_pathcheck_cx6/)
- out/cx7/<arm>_s<seed>.{json,txt} plus each JSON ledger_npz x 40; out/cx7/analysis/{analysis.md,runs.csv,compare.csv,decision.csv,call.csv,scatter.csv,state.csv,ring_rates.csv,predictions_vs_measured.csv,cross_batch_cx6.csv,analysis.json,scatter.png};
  out/cx7_cluster.log
api:
- cx_wedge.parse_edge_gains(['PRE:POST:FACTOR', ...]) -> [(pre, post, factor)]; cx_wedge.simulate(..., edge_gains=None)
  appends them to LIFParams.type_path_gain after the holds; the row records edge_gains and edge_gains_resolved (cells,
  entries, same-type entries, effective same-type factor). Default None = the previous gain list, entry for entry.
- 'cx_ring_structure.analyse(..., sigma=SIGMA_MV) and validate_batch(..., sigma=...): every slope bound, gain, fixed
  point and saturation rate is computed at that sigma and labelled with it; summarise_state adds in_above_22 / out_above_22
  / confined_by_ledger_rule and rate_model adds confined_bump_after'
- 'cx_ring_structure: recurrence_configs, EPG_EPG_GAIN, CX7_ARMS, PRIMARIES_CX7, MAGNITUDES_CX7, SECONDARIES_CX7,
  ring_distance, DRIVEN_CENTRE, DRIVEN_TILE_TOL, analyse_batch_cx7 (two references, both rules, the four-way call,
  per-type ring rates, survival_at_tile_s from the .npz), write_predeclaration_cx7; CLI --sigma, --local-recurrence,
  --predictions-csv, --batch cx7'
- 'measure_lif_sigma: window_stats, summarise, summarise_dir; CLI --out (required) --label --hold-edges --edge-gain
  --lif --nt-override --save-traces --summarise'
validation:
- 'Full closeout CPU suite: 441 passed, 19 skipped, 215 subtests; tests/test_bit_identity.py included. Cache MD5s
  unchanged.'
- 'Non-ledger CPU smoke: 120 equal non-timing fields, wall_s differs; no saved pre-6B CPU ledger comparison.'
- 40 CUDA runs; analysis problems 0; scheduler completed 10/10 jobs. cx6/cx7 recorded values reproduce exactly,
  not a general GPU determinism guarantee.
- Batch SHA-256 matches its predeclaration; five of 15 source-stamp entries had changed by successful submission.
  Failed-attempt console and second invocation survive only as author recollection.
recommendations:
- Adopt nothing; raw defaults and cache stay unchanged.
- 'Withdraw the rest-of-ring inhibition recommendation: the excess is adjacent and the far half already quiet. New
  width/rate/orientation interventions need separate predeclarations.'
- Calibrate state-dependent noise against LIF transfer; keep 11.8 mV as sensitivity, not a measured bound.
- Keep pipefail in future submission wrappers and require explicit structure output paths.
open_questions:
- Can a predeclared partial hold preserve physiological width and rate while allowing a turn-following bump?
- Can a rate model predict the selected branch and position, and explain the per-type DC-rate discrepancy?
- What are ExR6 contact-specific receptor placement/kinetics and co-transmission? Transmitter labels now have evidence;
  the earlier wholly UNKNOWN claim is superseded.
- What biological evidence supports EPG electrical coupling and the chemical recurrence gain?
```

## Skeptic pass

Independent skeptic's completed notes follow. Fable's handoff described the verdict as **"mostly sound"**,
16 claims with one refuted. **No final VERDICT, CLAIMS or CORRECTIONS REQUIRED block exists:** the owner
confirmed on 2026-09-15 that the reviewer hit the usage limit before writing it. We retain the completed
sections verbatim below rather than invent that missing summary. Infrastructure and local interpreter
paths alone are replaced by explicit placeholders. The notes are chronological: part 1 precedes the
successful batch; part 2 checks its completed outputs. Author corrections are in the audit above, not
inserted into the reviewer's text.

<details>
<summary>Completed independent review, sections 0-16 and carry-over corrections</summary>

> # Skeptic pass, compass 6B OFFLINE part (docs/audits/compass_local_recurrence.md, repo main a41d0f2)
>
> Reviewer: independent skeptic. Everything below was recomputed on this desktop (CPU, PATH python
> <PATH-python>, CUDA_VISIBLE_DEVICES=-1). No repo file edited.
>
> ## 0. Overtaking event found first (affects section 0, 5 and defect 1)
>
> `out/cx7_cluster.log` and `out/cx7/client_stdout.txt` (byte-identical, both mtime **2026-09-15 19:12:03Z**)
> no longer contain the text the audit quotes verbatim. `grep -c UNAVAILABLE` = 0. They now record a
> **successful submission**:
>
> ```
> [house] slots 112 (estimated at vram_gb 24, headroom 4 GB)
> [house] run dir <cluster-run-root>/cx7-201788: 3 local file(s) shipped (...)
> block s0..s4: 2 job(s) -> @house
> job 00e93fe544fd queued @house cx7-201788-0 [block s0] ...   (6 of the 10 job lines written so far)
> ```
>
> `out/cx7/submit_stamp.txt` now reads **2026-09-15T19:11:54Z**, not the 18:30:36Z the audit's stamp chain
> states. Audit committed 19:08:50Z; submission 19:11:54Z; this review ran 19:13-19:20Z. So the batch was
> resubmitted ~3 min after the audit was committed and is (at review time) queued at house with 0 of 40
> artefacts fetched. origin/main == a41d0f2, so the box gets the 6B code from the repo (only 3 other-thread
> files were shipped as a diff).
>
> ## 1. Sigma measurement
>
> Recomputed from out/cx7/sigma/sigma_{S_s0,S_s1,H3_s0}.json with summarise_dir's own pooling:
>
> | quantity | recomputed | audit |
> |---|---|---|
> | sigma_v relays sub-threshold, median over 128 cells | 4.638431549072266 | 4.638 / "4.6" OK |
> | its range | 2.7326462268829346 - 7.691947937011719 | 2.733-7.692 OK |
> | sigma_v EPG, median over 92 cells | 5.52144193649292 | 5.521 OK (but see (a)) |
> | sigma_g relays | 11.800286293029785 | 11.800 OK |
> | sigma_g EPG | 12.456063270568848 | 12.456 OK |
>
> Runs: three (S s0 18:14:39Z, S s1 18:17:16Z, H3 s0 18:15:58Z), all device cpu, dt 0.5 ms, settle 1.0 s /
> pulse 2.0 s, transient 0.2 s dropped => settle window 1600 steps = 0.8 s, pulse 3600 = 1.8 s, bg 10 Hz,
> pulse 40 Hz, receptor_model "sign" (the shipped default). Headline pools S seeds 0+1 only. Units mV.
> Sub-threshold: in arm S settle PEN 42/42, PEG 18/18, EPGt 4/4 never spiked -> the 4.64 estimate is clean.
>
> Arm dependence is large and only partly disclosed: pooling the same never-spiked relay cells under the
> hold gives median sigma_v **2.773** and sigma_g **13.22** (H3 s0) against 4.638 / 11.800 in S. Per group
> under H3: sigma_v_ns 2.679 / 3.216 / 2.506 (PEN/PEG/EPGt) -> the audit's "2.5-3.2 mV" OK;
> sigma_g medians 12.35 / 14.66 / **11.02** -> the audit's "12.4-14.7 mV" **excludes EPGt**; true range
> 11.0-14.7 (11.99-14.36 restricted to never-spiked cells).
>
> (a) **The EPG 5.52 mV is a spiking-cell reading.** All 46 EPG spiked in both S seeds (rate 9.9 / 9.3 Hz),
> so `pooled_epg_v` is 92 cells of which zero are sub-threshold. That is exactly the reading the audit's own
> defect 4 ("a spiking cell's membrane SD is not a measurement of its input noise") and its own method
> statement ("both restricted to the cells that never spiked in the window") declare unusable. It is quoted
> as a measurement in section 0 item 1, section 2.1, section 6 and key_claims 1.
>
> (b) **"every firing ring cell reads 2.1-2.4 mV whatever its rate" is false.** Over all 98 firing ring
> cells (ExR6/ER6/ER4m, 3 runs x 2 windows) the per-cell sigma_v_free runs **1.041-2.909 mV**, median 2.108,
> and only 45 % fall in [2.1, 2.4]. The lowest are ER4m cells at 1.2 Hz (1.04-1.13 mV); the highest an ER4m
> at 81.7 Hz (2.909). So it does vary with rate, in the direction opposite to the claim's "whatever".
>
> (c) **Is it the rate model's sigma?** Partly.
>   - For the relays yes: they carry no forced drive, so `g` is their whole input and
>     `_membrane_target = v_rest + g + drive - adapt` makes g the model's u.
>   - For EPG no: `fb.stimulate` writes `poisson_p` (a forced per-step spike probability), not `drive`, so
>     the 10 Hz background / 40 Hz pulse is an imposed spike, never a current. The rate model converts that
>     forced rate into a current through `u_for_rate` (6A fix 1); the LIF has no such term, so EPG's g and v
>     are not the model's u for the driven cells.
>   - The **upper edge of the bracket is not established**. The smoothed f-I is a *quasi-static* Gaussian
>     convolution, which would want sigma_g; the audit re-runs at sigma_v on a diffusion argument. The
>     measured ratio sigma_v/sigma_g = 4.638/11.800 = **0.393** is close to sqrt(tau_syn/(tau_syn+tau_m)) =
>     sqrt(5/25) = 0.447 (brain.py: tau_m 20 ms, tau_syn 5 ms), i.e. the input correlation time is 4x
>     shorter than the membrane's, so the frozen-noise limit that justifies 11.8 does not apply. The
>     defensible statement is the membrane value ~4.6 (which is what every headline uses anyway).
>
> ## 2. Consequence table
>
> `scripts/cx_ring_structure.max_slope / rate_at_gain / u_for_rate` re-run at every sigma in
> out/cx7/sigma/slope_vs_sigma.json: **all 36 cells of that file reproduce exactly** (max slope 8.2747 /
> 6.0557 / 4.3052; at_u, at_hz, sat 3.345, sat 3.017, u10, u50 all identical).
>
> - 33.34 / 6.0557 = **5.506** ("5.5x"); 33.34 / 8.2747 = 4.029 ("4.0x"); 33.34 / 4.3052 = 7.744 ("7.7x").
> - (146.5 - 144.3)/144.3 = **1.52 %** ("1.5 %").
> - gamma 3.33 < 4.3052 at sigma 11.8 -> supercritical across the bracket. OK
> - max_slope(0.17) = 34.154 >= 33.34, max_slope(0.18) = 32.751 < 33.34 -> "sigma would have to be 0.17 mV"
>   OK; 4.638/0.17 = 27.3 ("27x").
> - Adaptive-quadrature check of the audit's unverified aside: exact max slope 7.9969 / 5.9297 / 4.2496 at
>   sigma 2.0 / 4.6 / 11.8, so the 61-node value is **3.47 % / 2.13 % / 1.31 %** high. The bias shrinks with
>   sigma, and on exact values the ratio is 4.17x / **5.62x** / 7.85x -- the conclusion strengthens.
>
> Validation vs out/cx5 at the three sigmas (validation_sigma*/validation.json): **7/8 bump calls and 2/3
> rate calls at every sigma**, single miss FG, F predicted 144.302 / 144.670 / 146.515 vs measured 152.503
> (5.38 / 5.14 / 3.93 %), CFG 159.711 / 160.005 / 162.149 vs 157.061 (1.69 / 1.87 / 3.24 %), CF vs 142.823
> (11.82 / 12.03 / 13.53 %). gamma_crit_k1 takes exactly four values over the eight arms (33.340, 28.768,
> 3.3449, 3.0169) -> the audit's own "3 of 4 distinct predictions" caveat is right.
>
> NUANCE the audit's "Nothing moved in the validation" hides: the tool's `predicted_bump_with_relays` flips
> **True -> False** for all four F-family arms between sigma 2.0 and 4.6, and `saturation_with_relays` goes
> 105.7 Hz -> NaN. The scored columns (bump_call_ok, rate_call_ok) are unchanged.
>
> Side observation (not 6B's): out/cx5/F_s0..3 and out/cx6/F_s0..3 carry identical bump_hz_post to 3 dp
> (155.759 / 150.908 / 152.262 / 151.083) though the files differ; the cx5 batch the tool is validated
> against and the cx6 batch it is scored against share those four runs' numbers.
>
> ## 3. Per-type ring rates and the bit-identity
>
> Bit-identity: out/cx7/smoke/smoke_default_path_noledger.json vs out/cx6/smoke/smoke_default_path.json ->
> **121 shared keys, 120 equal, only `wall_s` (31.8 -> 33.7)**, extra keys exactly `edge_gains` and
> `edge_gains_resolved`. Reproduced exactly.
> SCOPE: both files are **non-ledger** rows (neither has `metrics`). The ledger path -- the one every batch
> run uses, and the one that gained 12 group-mean keys, 24 `*_cell_max_*` keys, 8 `*_n_cells` keys and the
> `cells__*` npz arrays -- has no pre-6B CPU counterpart and was never diffed. The purity argument does hold
> on inspection: `Brain.rates` / `mean_rate` are pure reads of the EMA `self.rate`, no RNG draw
> (flyverse/brain.py:972-984), so the extra recording cannot move the simulation.
>
> Ring rates: the "measured" column is real spiking-LIF CPU runs (measure_lif_sigma.py, arm S seeds 0/1,
> `rate_mean` in sigma_table.csv) -- **not** the rate model and **not** cx6. Model values reproduce from
> out/cx7/structure_sigma2/structure.json and out/cx7/structure/structure.json
> (`configs[shipped].decomposition.<state>.PEN_in<-Ring_by_type`):
>
> | type/state | model s2.0 | model s4.6 | measured | ratio (s2.0) | audit |
> |---|---|---|---|---|---|
> | ExR6 rest | 110.214 | 107.676 | 43.125 / 41.875 | 2.56-2.63 | "2.5-2.6x" (should read 2.6x) |
> | ER6 rest | 53.780 | 51.932 | 24.375 / 22.813 | 2.21-2.36 | table says "2.1-2.4x", should be 2.2-2.4 |
> | ER4m rest | 4.428 | 10.287 | 3.182 / 1.705 | 1.39-2.60 | OK |
> | ExR6 pulse | 183.174 | 181.478 | 85.833 / 83.889 | 2.13-2.18 | OK |
> | ER6 pulse | 116.320 | 115.190 | 52.500 / 50.417 | 2.21-2.31 | OK |
> | ER4m pulse | 22.348 | 24.071 | 11.869 / 11.162 | 1.88-2.00 | OK |
>
> Section 0's "2.2-2.4x on ER6" is right; the 3.3 table and key_claims 5's "2.1-2.4x" is not.
> Hold row verified: H3 s0 settle ExR6 274.375, ER6 191.250, ER4m 101.3636, GLNO 149.375, EPG 76.1141.
>
> ## 4. The per-type gain on the effective weights
>
> flyverse.interp.common.effective_weights on the real connectome, four configurations built exactly as
> cx_wedge.simulate builds them (shipped tpg + hold + edge gain):
>
> - EPG -> EPG block, H3E vs H3F: **bit-for-bit equal** (float32 bit patterns, 842 nnz both).
> - H3E vs H3 over the WHOLE matrix: **842 entries changed, all 842 of them EPG -> EPG, zero others.**
>   EPG's fan-in `scale` is exactly 1.0 in both (tot median 3324 / 3540 < input_norm_ref 5000), so nothing
>   propagates. "No other block moved" is literally true for the batch arm.
> - Clique table (mean mV per pair / mean per-post volley, nnz), all matching section 3.2 to 3 dp:
>
> | cfg | EPG (842) | PEN_a (257) | PEN_b (274) | PEG (114) | Delta7 (1717) |
> |---|---|---|---|---|---|
> | shipped | +0.3664/+6.71 | +0.5827/+7.49 | +0.6245/+7.78 | +0.1310/+0.83 | -0.4251/-17.38 |
> | H3E | +3.6658/+67.10 | +0.5827/+7.49 | +0.6245/+7.78 | +0.1310/+0.83 | -0.4251/-17.38 |
> | H3F | +3.6658/+67.10 | +5.8274/+74.88 | +6.2447/+77.78 | +1.3099/+8.30 | -4.2506/-173.77 |
>
> - `brain._shaped_weights` applies type_path_gain (line ~322) before same_type_gain (line ~332). OK
> - `--edge-gain` resolved record in the real H3E code-check run: 842 entries, 842 same-type,
>   effective_factor_same_type 1.0. OK
> - tests: 12 + 22 = **34 passed** (pytest, CPU).
> - CAVEAT: the *general* form of the section 1.2 sentence fails at the effective-weights level without the
>   hold. Configuration **E** (per-type gain, no hold) moves **1,353 non-EPG->EPG entries** relative to
>   shipped, because EPG's fan-in scale falls from 0.9806 to 0.9376 once the undamped EPG->EPG input pushes
>   `tot` past input_norm_ref = 5000. E's EPG->EPG reads +3.6529/+66.86, not +3.6658/+67.10. True of the
>   shaped matrix, false of the installed one. No batch arm is affected (all the per-type arms carry the
>   hold).
>
> ## 5. `silent`
>
> From out/cx7/sigma/sigma_S_s{0,1}.json (`windows.settle.<g>.summary.rate_hz.max` and `n_never_spiked`),
> tabulated in sigma_table.csv column `never_spiked`:
>
> - settle (0.8 s): PEN 42/42, PEG 18/18, EPGt 4/4, GLNO 4/4 never spiked; per-cell max **0.000 Hz** in both
>   seeds. Meets INTERP 10's `silent_frac` (max rate < 0.5 Hz). Reproduced.
> - pulse (1.8 s): PEN 6 of 42 fire, max 5.556 / 2.778 Hz; GLNO 4 of 4 fire, max 3.889 / 2.222 Hz -> neither
>   silent. PEG also fires (4 of 18, max 6.111 / 8.333) -- not mentioned, not contradicted. EPGt stays at
>   max 0.000 in the pulse too.
> - Vocabulary rule holds: `silent` is window-scoped in INTERP 10 and no population with a non-zero recorded
>   rate is called silent anywhere in the audit.
> - Attribution slip: section 0 item 5 ("The runs record the per-cell maximum ... 42 of 42 PEN ...") reads
>   as if the new cx_wedge ledger keys delivered the decision. They did not -- 0 of 40 ledger runs exist;
>   the numbers are from measure_lif_sigma.py. Section 3.4 attributes it correctly.
>
> ## 6. The predeclaration
>
> - 8 arms x 5 seeds = 40 runs, 10 jobs, one submission. OK (predeclared.json `batch`, batch.sh 10 job
>   strings, jobline_check.txt "job lines: 10 ... 5 3 5 3 5 3 5 3 5 3").
> - `common.p_floor(5,5)` = 0.007936507936507936 = 2/252, and scipy's exact 5v5 two-sided MWU on fully
>   separated samples returns the same. 5 x = **0.03968 <= 0.05**. OK
> - `cx_ring_structure.holm` is a correct step-down Holm (running max of (m-i) p, monotone, most significant
>   = p x m). The cx6 path-check has m_holm = 5 on all 30 primary rows across 6 (arm, ref) families. OK
> - Two references do **not** double the family as predeclared (Holm is applied within each (arm, reference)
>   family). But that cut is load-bearing: a per-arm family over both references would be m = 10 and
>   10 x 0.0079 = 0.0794 > 0.05, i.e. unsatisfiable. FWER is therefore controlled at 0.05 inside each of the
>   12 families, not across the round, and at 5 v 5 each family has exactly one attainable p (perfect
>   separation) -- no margin at all.
> - batch.sh sha256 = 1552e602f0d81510e795ded8a9858dab0f2f41bb1b34189aa23fd22339a91bfc, **identical** to
>   predeclared.json `batch.batch_sh_sha256`. OK
> - Stamp order from the files: structure.json generated_utc 18:19:50Z (mtime 18:28:30Z) < batch.sh 18:23:25Z
>   < validations 18:23:34 / 18:26:24 / 18:29:15Z < predeclared.json 18:29:58Z (stamped_utc 18:29:58Z) <
>   predeclare_stamp 18:29:59Z < tree_state 18:30:29Z. All as stated. **submit_stamp is now 19:11:54Z**, not
>   18:30:36Z, and the client log is the successful one (section 0 above).
> - Files rewritten after the tree_state stamp: **five of the thread's own**, not two (section 5's heading)
>   or three (key_claims): scripts/cx_ring_structure.py 19:08:50Z, scripts/measure_lif_sigma.py 18:42:07Z,
>   tests/test_cx_ring_structure.py 19:04:29Z, out/cx7/sigma/{sigma_summary.json, sigma_table.csv} 18:42:30Z,
>   out/cx7/structure/predictions.csv 18:42:35Z (this last is quoted in section 2.3 and is not in the
>   audit's list at all). Plus two other-thread files. tree_state.json's sha256 map: **5 of 15 entries no
>   longer match** (cx_ring_structure.py, measure_lif_sigma.py, test_cx_ring_structure.py,
>   probe_vnc_drive.py, derive_level_fixed_point.py).
> - Nothing in the predictions is affected: the whole slope table and all three validations were re-derived
>   here from the CURRENT cx_ring_structure.py and match the recorded files exactly.
> - The 8-arm prediction table in section 4 reproduces cell-for-cell from
>   predeclared.json `predictions_from_the_structure_pass.per_arm` (H3 99.34/71.63, 34.13/95.31, 3/11 14/35;
>   H3F 351.48/350.76, 304.29/326.75, 11/11 35/35, 145.23; H3E 16.34/16.66, 10.0/164.71, 0/11 20/35;
>   H3G 35.98/32.38, 164.99/11.63, 11/11 2/35, confined True; H3EG 46.25/45.99, 241.75/39.43, 10/11 6/35;
>   H3FG 10.0/149.45, 0/11 18/35; F 144.67; local_EPG_EPG_mV 6.0 -> 60.01).
>
> ## 7. Defects
>
> - `scripts/cluster_run.py` line 693 `print("no target answered; nothing submitted")` / 694 `return 2`,
>   and 720 / 721 `return 2`. **Both branches return 2.** Owner's correction reproduced.
> - `out/cx7/batch.sh` is 3 lines (shebang, comment, one command) ending
>   `... --fetch out/cx7/ 2>&1 | tee out/cx7/client_stdout.txt`; no `set -o pipefail`, no `set -e`.
>   `$?` is tee's. Owner's correction reproduced.
> - But the evidence line ("client exit 0") is no longer in out/cx7_cluster.log -- see section 0.
>
> ## 8. Prose per-seed / per-cell quotes (rule 28)
>
> Sourced and verified exactly: the 12-row sigma_table.csv paste (values exact; the audit reorders S before
> H3, the file lists H3 first); the sigma_summary headline; H3 s0 per-type rates (274.375 / 191.250 /
> 101.3636 / 149.375 / 76.1141, column `rate_mean`); the codecheck console line and "out 152.8 Hz, 25 of 35
> above 22 Hz" (pre_out_mean 152.847, pre_out_above 25); H3G survival 5.00/4.76/4.91/4.98/4.77 and
> survival_at_tile_s 0.85/4.76/2.79/1.18/0.00 and F 5.00/5.00/5.00/5.00/1.84 (runs.csv, seed order correct);
> H3G bump_hz_post 158.979-164.168 -> "159.0-164.2"; F 150.556-155.759 -> "150.6-155.8"; the pathcheck's 84
> problems and m_holm = 5; the jobline_check.txt block verbatim.
>
> Failures:
> 1. "H3G 0.58 / 3.80 / 3.93 / 7.47 / 7.49" is the **sorted** list; the per-seed order in runs.csv
>    (`centre_dist_t5`) is 3.934 / 0.585 / 7.485 / 7.468 / 3.804. The sentence also names no file or column.
> 2. "every firing ring cell reads 2.1-2.4 mV whatever its rate" -- no file carries that range (actual
>    1.041-2.909 over 98 firing ring cells). Quoted twice (2.1 and defect 4) plus key_claims 1.
> 3. Header line 7 says tests/test_cx_ring_structure.py "+6 tests, 19 total"; the file has 22 tests and
>    section 1.2 / the Report say 22. 34 passed in one CPU run here (12 + 22).
> 4. Section 5 says the analysis path "emits all ten files"; the pathcheck copy has 9 of the 10 named files
>    plus README.txt -- scatter.png is absent.
>
> # PART 2 -- the batch (cx7-201788, 40/40 runs). Claims 9-16.
>
> Everything below recomputed from `out/cx7/<arm>_s<seed>.json` / `.npz` directly, cross-checked against
> `out/cx7/analysis/*.csv`.
>
> ## 9. survival / frac_confined / the working-compass rule -- REPRODUCED, one sentence false
>
> From the 40 run JSONs' `metrics`: survival_s 0.00 and frac_confined_post 0.000 in 5/5 seeds for **H3E, H3F,
> H3FG, H3EG** (and S, H3). Working-compass rule (survival >= 5 AND width in [2.5,5] AND rate in [5,60], per
> seed) met in **0/5 in every one of the 8 arms**: F and H3G clear survival and width but their bump rates are
> 150.6-155.8 and 159.0-164.2 Hz, outside the 5-60 Hz bound. `call.csv` matches my recomputation row for row.
>
> FALSE as written: "the ledger rows stay FAIL / NOT_APPLICABLE in all 40 runs" (4.1) and
> "compass.EPG.bump_survival_s stays FAIL with the rate and width rows NOT_APPLICABLE in all 40 runs" (6,
> recommendations). Over the 40 runs the ledger reads: `bump_survival_s` 35 FAIL / 5 PASS,
> `bump_width_wedges` 35 NOT_APPLICABLE / 5 PASS, `bump_rate_hz` 35 NOT_APPLICABLE / 5 FAIL. The five PASS
> runs are F seeds 0-3 (survival 5.00, width 3.0) and H3G seed 0 (5.00, 4.0). The claim holds for the 30 runs
> of S, H3 and the four two-instrument arms, and the shipped-path row (arm S) is FAIL in 5/5.
>
> ## 10. H3E geometry -- REPRODUCED, but two of the quoted numbers are from the wrong window
>
> Seeds 0-3 (`t5.0_*` in the run JSONs): centre_wedge 3.151 / 3.180 / 3.085 / 3.227 -> 3.08-3.23; dist 1.651 /
> 1.680 / 1.585 / 1.727 -> 1.58-1.73; in_above 11/11 in all four; vector strength 0.6676 / 0.6750 / 0.6738 /
> 0.6757 -> 0.67-0.68; out_above 11 / 12 / 12 / 12 -> 11-12 of 35. Seed 4 mirror-images (centre 11.392, dist
> 6.108, 2/11, 23/35). `state.csv` `profile_end` seed 0 is quoted exactly:
> `197;243;271;257;284;252;200;29;8;15;6;14;12;22;5;16` (reproduced from the npz's last frame). "Seven-wedge
> hump" reproduces as the half-maximum width (7 wedges above half-max in all four seeds); the count above
> 22 Hz is 7-8 wedges. The criterion is not stated in the audit.
>
> WINDOW ERROR: "EPG in 242.4-244.0 Hz against out 73.1-73.6 Hz" is not the 5 s profile the sentence
> attributes it to. Those are `epg_in_mean_post` / `epg_out_mean_post` (means over the whole 5 s post-pulse
> window): 243.99 / 242.35 / 243.14 / 243.05 and 73.07 / 73.59 / 73.52 / 73.50 -- exactly the quoted ranges. At
> 5 s the values are in 239.4-243.8 / out 71.0-73.5. Same slip in section 0, the Report summary and
> key_claims 2. (The section 4 table is correct: it labels the column "post".)
>
> "fails ... ONLY on the off-block count" and "only marginally" on the tile rule overstate how close it is.
> The same hump is 7 wedges wide (the working-compass rule wants 2.5-5) and runs at ~243 Hz (the rule wants
> 5-60): had it been confined it would have failed the width row and the rate row by 4x. And the 1.58-1.73
> offset is in the same direction in all four seeds (always +1.6 from the driven centre, never -1.6) -- a
> systematic offset, not a marginal miss.
>
> NOT MENTIONED: the hump is not built by the pulse. H3E's pre-pulse state is already a structured, high ring
> in every seed (`pre_vector_strength` 0.65-0.69, `pre_out_mean` 94.6-153.2 Hz), with the hump at wedge 14.8
> (seeds 0, 3), 7.3 (seeds 1, 2) or 11.4 (seed 4). The pulse moves an existing attractor to ~3.1 in 4 of 5
> seeds; it does not light up a quiet ring.
>
> ## 11. H3F flat saturated ring -- REPRODUCED, same window slip, "flat to within 10 %" false
>
> `epg_in_mean_post` 267.56-268.89 -> 267.6-268.9; `epg_out_mean_post` 282.88-283.35 -> 282.9-283.4 (again
> post-window means, not the 5 s values, which are 262.9-270.4 / 282.8-284.4). PEN_mean_during 278.03-278.44 ->
> 278.0-278.4; 11/11 and 35/35 above 22 Hz in 5/5; `t5.0_vector_strength` 0.0442-0.0496 -> 0.044-0.050;
> GLNO_mean_post 317.0-318.6 -> "317-319"; ExR6_mean_post 332.5-333.3 -> "332-333". All OK.
>
> "profile flat to within 10 % across all 16 wedges in every seed" is FALSE. From `state.csv` `profile_end`:
> max/min per seed = 1.107 / 1.165 / 1.107 / 1.124 / 1.120, i.e. (max-min)/mean = 10.0 / 14.7 / 10.0 / 11.4 /
> 11.1 %. Seed 1 spans 249-290 Hz. "Flat to within 15 %" or "to +-8 % of the mean" would be true.
>
> ## 12. "What is missing is inhibition of the rest of the ring" -- NOT LICENSED, and contradicted by the data
>
> The description is right: H3E fails the ledger's confinement count on the off-block side (11-12 of 35 above
> 22 Hz, budget 3) while holding 11 of 11 driven cells. The inference is not.
>
> (a) The rest of the ring is not running. Splitting H3E's 5 s profile: the far half (wedges 8-13, 17 cells,
> >= 5 wedges from the driven centre) sits at mean 8.7-12.3 Hz with 0-2 cells above 22 Hz in seeds 0-3 -- at
> the 10 Hz background floor. The off-block excess lives in wedges 4-7, contiguous with the driven block, plus
> one or two isolated cells. The failure is that the hump is too wide, not that the ring is disinhibited.
>
> (b) The per-type recurrence made the off-block side quieter, not noisier. Against H3 (hold, no recurrence)
> H3E's far half falls from 31.2-46.4 Hz to 8.7-12.3 Hz and its off-block count above 22 Hz from 17-19 to
> 11-12, while Delta7 rises from 120.9-124.0 to 165.7-170.4 Hz. The excitation the arm adds recruits more
> inhibition and improves confinement -- the opposite of the sentence's causal picture.
>
> (c) No arm separates the two. The hold zeroes ExR6/ER6/ER4m -> EPG and -> PEN for all 46 EPG and 42 PEN,
> driven and off-block alike, so "inhibition of the rest of the ring" cannot be varied independently of
> "inhibition at the tile" anywhere in this batch. Equally, `--edge-gain ^EPG$:^EPG$:10` multiplies all 842
> EPG -> EPG pairs (every EPG cell's recurrent drive, +6.7 -> +67.1 mV per volley), so the arm is not
> "excitation at the tile" either. Section 6's "Not established" paragraph concedes exactly this; sections 0,
> 4.1, 6(2), the Report summary, key_claims 2 and the first recommendation state it as a finding.
>
> (d) 1 of 5 seeds puts the same instrument's hump 6.1 wedges away, so even "excitation at the tile is
> sufficient to place the hump" is a 4-of-5 statement.
>
> ## 13. H3EG -- REPRODUCED
>
> `t5.0_centre_wedge` 10.790 / 10.766 / 0.293 / 0.528 / 10.572; `centre_dist_t5` 6.710 / 6.734 / 1.207 / 0.972
> / 6.928 -> 2 of 5 seeds within the 1.5 bound, 3 on the opposite side at 10.57-10.79. `call.csv`
> `driven_tile_seeds` = 2. In the two tile seeds: 11/11 driven cells above 22 Hz, `epg_in_mean_post` 236.21 /
> 228.80 -> "228.8-236.2" (section 0 rounds to "229-236"), out 38.03 / 39.85 -> "38.0-39.9".
> frac_confined_post 0.000 in 5/5. All OK.
>
> ## 14. cx7 reproduces cx6 -- REPRODUCED, and stronger than claimed
>
> `cross_batch_cx6.csv`: 760 rows (4 arms x 5 seeds x 38 metrics), max `absdiff` 0.0. Independently, I compared
> every shared numeric field of the 20 shared run JSONs (metrics, ledger values, all seven t-mark 16-wedge
> profiles, vector strengths, counts): 6,105 numeric pairs, 50 NaN-vs-NaN, max |cx7 - cx6| = 0.0, zero
> differing fields. Provenance: 40/40 rows and 40/40 consoles `device cuda`; compiled-W md5
> `ef23cc27bea13be7f6a96f3c04fd3737` on the 25 GLNO-silent runs and `7a10d93ba2086f2c76bcdabdca79b4ec` on the
> 15 relabel runs; `same_type_gain` 1 in exactly F / H3F / H3FG; `edge_gains_resolved` = (842 entries, 842
> same-type, effective factor 1.0) in exactly H3E / H3EG; hold = (1149 entries, factor 0, 37256 syn) in the six
> hold arms and absent in S / F; `glno_nt` glutamate in exactly the three relabel arms; wall 13.7-27.5 s;
> analysis `problems` 0, `n_runs` 40. Scheduler receipt: 10 jobs, 0 failed, {'completed': 10}, <cluster-node>, submitted
> 19:11:55.46Z, last finished 19:14:48.92Z, `exit_code: null` (disclosed).
>
> ## 15. The Holm families -- RECOMPUTED, one claim false
>
> Rebuilt every (arm, reference) family from the run files with `common.compare` + `cx_ring_structure.holm`
> (centre_dist_t5 recomputed as `ring_distance(t5.0_centre_wedge, 1.5)`). Every number in the section 4.3 table
> matches: H3 +43.53 / z 294.0 / Holm 0.0397; H3F +277.80 / z 1876 and +277.80 / z 1.328e4; H3E +59.21 / z
> 399.9, +56.50 / z 2701, centre -2.825 / z -4.208 / p 0.09524 (null); F +24.69 / z 166.8, +26.41 / z 1262,
> centre -3.796 / z -5.654 / p 0.1508, survival +4.368 and frac +0.7204 `undetermined`; H3G +26.94 / z 182.0,
> +23.74 / z 1135, +4.884 / +0.324; H3FG +100.48 / z 678.7, +114.07 / z 5453, centre -2.540 / z -3.783 / Holm
> 0.0397 `result`; H3EG +31.01 / z 209.4, +35.57 / z 1700, centre -0.865 / z -1.288 / p 0.6905. Against H3:
> H3F +234.3 / z 67.67 and +1.928 / z 11.34 (both `result`); H3E +15.68 / z 4.53 (`result`) and -1.955 / z
> -11.5 / p 0.1508 (null); H3FG -1.670 / z -9.821 (`result`). m = 5 in all 14 families (70 primary rows). The
> H3E `centre_dist_t5` reading rule is exactly right: the exact-U p is 0.0952 because the mirror-image seed
> crosses over, so `|z| >= 3` alone would have called it.
>
> FALSE: key_claims -- "PEN_mean_during and PEN_mean_post are `result` for all seven arms against both
> references". H3E vs H3, `PEN_mean_post`: diff +4.13, z +2.16, verdict `null` -- in my recomputation and in the
> project's own `compare.csv`. It is 13 of 14 arm x reference pairs, not 14.
> Also incomplete: `centre_dist_t5` is a `result` in four places, not the two key_claims lists -- H3FG vs S,
> S vs H3 (+0.870, z +5.12), H3F vs H3, H3FG vs H3.
>
> ## 16. Rule 28 over sections 4-6
>
> Verified against the named files: all 23 lines of the section 4.4 scatter block match `analysis/scatter.csv`
> character for character (file and columns named); the `call.csv` table (8 rows) matches; the H3F and H3E
> seed-0 `profile_end` strings match `state.csv`; every "measured" range in the section 4 table reproduces from
> the run files (S PEN 0.3-0.7, H3 40.2-48.3, H3F 278.0-278.4, H3E 57.3-62.9, F 22.5-27.4, H3G 26.5-29.0,
> H3FG 89.9-119.8, H3EG 27.7-37.1; in/out post and the >22 counts all check out); the section 3.3 GPU ring
> rates match `ring_rates.csv` and the run files; the section 6 `silent` numbers match the `*_cell_max_post`
> columns (PEN 0.13-0.58, PEG 0.40-0.61, GLNO 0.02-0.39, EPGt 0.00; EPGt and GLNO silent 5/5, PEN and PEG
> silent 4/5 each -- "straddle" is right).
>
> Gaps:
> 1. "EPG in 242.4-244.0 Hz against out 73.1-73.6" (4.1, 0, Report, key_claims 2) names no file or column and
>    is attributed to the 5 s profile when it is `epg_in_mean_post` / `epg_out_mean_post`. Same for H3F's
>    "267.6-268.9 / 282.9-283.4".
> 2. "H3G splits 1 + 2 + 2 (0.92, then 5.30 / 5.43, then 10.01 / 10.03)" (4.4) names no file; the values are
>    `centre_wedge_t5` (0.915, 5.304, 5.434, 10.015, 10.032) and are sorted, not in seed order -- as is
>    "H3EG ... 0.29 / 0.53 vs 10.57-10.79".
> 3. Section 6 item 4's per-cell maxima name no file or column.
> 4. Section 5.1 quotes the first attempt's client output verbatim ("[house] UNAVAILABLE ... no target
>    answered; nothing submitted / client exit 0"). No file on disk carries that text any more:
>    `out/cx7_cluster.log` and `out/cx7/client_stdout.txt` were overwritten by the successful 19:11:54Z run
>    (they now end `10 job(s), 0 failed  (4.3 min) ... client exit 0`), `submit_stamp.txt` now reads
>    19:11:54Z, and a `grep -rl UNAVAILABLE out/` finds no copy. The 18:30:36Z first attempt is unsourceable
>    from the repo, and so is "the second submission was run as `bash -o pipefail out/cx7/batch.sh`".
>    (I can attest the overwrite happened at 19:12:03Z, between my two passes.)
> 5. Rate-error percentages: F "right to 3.9-5.1 %" mixes the cx5-mean error (5.1 % at sigma 4.6 against
>    152.503) with the cx7 per-seed range. Against 150.6-155.8 the error is 3.9-7.1 %. Section 6 item 5's
>    "right to 0.5-5 % wherever a bump exists" is therefore also wrong (max 7.1 %). H3EG's "2-5 %" is
>    2.4-5.7 %. H3G's "0.5-3.8 %" is exact.
>
> ## Carry-over from part 1 that the rewrite did NOT fix
>
> - The EPG sigma_v 5.52 mV is still quoted as a measurement (section 0 item 1, 2.1, key_claims 6) although all
>   46 EPG spiked in both seeds -- the reading the audit's own defect 4 calls unusable.
> - "reset clips every firing ring cell to 2.1-2.4 mV whatever its rate" (2.1, defect 4, key_claims 6): over the
>   98 firing ring cells the per-cell sigma_v is 1.04-2.91 mV, only 45 % inside [2.1, 2.4]; the low end is ER4m
>   at 1.2 Hz, so it does vary with rate.
> - "under the hold ... 12.4-14.7 mV on the input" excludes EPGt's 11.0 (11.99 never-spiked-only).
> - Section 1.2's test sentence "leaves every other entry bit-identical to the shipped matrix" is true of the
>   shaped matrix and of the H3E arm's effective matrix, but false for the no-hold E configuration (1,353
>   non-EPG->EPG entries move via the fan-in scale).
> - The bracket's upper edge (11.8 mV, the frozen-noise reading) is not established: sigma_v/sigma_g = 0.393 is
>   close to sqrt(tau_syn/(tau_syn+tau_m)) = 0.447, i.e. the input is 4x faster than the membrane, so the
>   quasi-static picture the upper edge rests on does not hold here.

</details>
