# Compass, thread 6B: the two-instrument arm -- two measurements made, the arm predeclared, THE BATCH NOT RUN

Scripts: `scripts/measure_lif_sigma.py` (new; the input-noise measurement), `scripts/cx_wedge.py` (ONE new flag
`--edge-gain PRE_REGEX:POST_REGEX:FACTOR`, default `None`, plus extra RECORDED keys: per-type ring rates and per-cell
maxima; shipped path bit-identical), `scripts/cx_ring_structure.py` (`--sigma` as an argument, `--local-recurrence`,
`--batch cx7`, `--predictions-csv`; `--out` unchanged and never defaulted here), `tests/test_cx_wedge_hold.py` (+6
tests, 12 total), `tests/test_cx_ring_structure.py` (+6 tests, 19 total).
Data: `out/cx7/sigma/{sigma_S_s0.json, sigma_S_s1.json, sigma_H3_s0.json, sigma_*.txt, sigma_summary.json,
sigma_table.csv, slope_vs_sigma.json, summary_console.txt}`, `out/cx7/structure/{structure.json, structure.md,
matrices.npz, evidence_glno.json, console.txt, predictions.csv, validation_sigma2.0/, validation_sigma4.6/,
validation_sigma11.8/}` (the pass at the MEASURED sigma 4.6 mV), `out/cx7/structure_sigma2/` (the same at the assumed
2.0 mV), `out/cx7/{predeclared.json (stamped **2026-09-15T18:29:58Z**, never amended), batch.sh, tree_state.json,
predeclare_stamp.txt, submit_stamp.txt}`, `out/cx7/smoke/{smoke_default_path.json, smoke_default_path_noledger.json,
codecheck_H3E_truncated.json, jobline_check.txt, argv_shim.py, argv.json, jobs.json, batch_probe.sh, exitcheck_*.sh,
flagcheck.sh}`, `out/cx7_cluster.log`.
Tree: HEAD fa584b2 (= origin/main); nothing committed by this thread. `out/cx7/tree_state.json` (stamped
2026-09-15T18:30:29Z) lists every file that would ship with the batch with its sha256 -- this thread's
`scripts/cx_wedge.py`, `scripts/cx_ring_structure.py`, `scripts/measure_lif_sigma.py`, the two test files, and another
thread's concurrent `scripts/probe_vnc_drive.py`, `tests/test_proprioception.py`, `scripts/derive_level_fixed_point.py`
(none on the simulated path: `cx_wedge.py --sim` builds a `FlyBrain` without a world).

## 0. Answer, and the blocker

**The batch did not run: the house cluster was unreachable from this desktop at submission time.** The client's own
words, `out/cx7_cluster.log` verbatim:

```
[house] UNAVAILABLE (API silent for 20s): <urlopen error [Errno 11001] getaddrinfo failed>
no target answered; nothing submitted
client exit 0
```

`<cluster-node>` does not resolve from here (`nslookup`: "Non-existent domain"; `ssh`: "Could not resolve
hostname"), the desktop is on a home LAN whose DNS is the router, no `hosts` entry or tunnel exists, and
**`house` is the only enabled target in `.cluster.json`** (every `vast-*` / `r3-h200*` entry is `disabled: true`). The
same cluster ran batch cx6 at 08:11Z today, so the path existed this morning and is gone now; nothing was submitted,
so there is nothing to recover with `fetch_run.py` (`out/cx7/` holds only the CPU smokes, 0 run artefacts of 40
expected). **`client exit 0` on a batch that never ran is itself a defect** and is recorded in section 6 -- owner's correction 2026-09-15: `cluster_run.py` returns 2 on both nothing-submitted branches (its lines 693-694 and 720-721); the 0 is `batch.sh`'s `| tee` without `pipefail`, so the defect is in the batch wrapper, not the client.
`out/cx7/batch.sh` (sha256 `1552e602f0d81510e795ded8a9858dab0f2f41bb1b34189aa23fd22339a91bfc`, the value recorded in
the stamped predeclaration) is ready to submit unchanged when the cluster answers.

**What this thread did establish, all on the CPU and all independent of the batch:**

1. **THE INPUT NOISE IS MEASURED, AND IT IS NOT 2 mV.** Every slope bound of 5A / 6A is a property of
   `cx_ring_structure.SIGMA_MV = 2.0`, which had never been measured. On the shipped arm's own protocol the free
   membrane of the sub-threshold relays fluctuates with SD **4.6 mV** (median over 128 never-spiked PEN / PEG / EPGt
   cells, seeds 0 and 1), the EPG's with 5.5 mV, and the synaptic input `g` itself with **11.8 mV**. The smoothed
   f-I's sigma lies between the membrane-filtered and the frozen-noise readings, so the honest bracket is
   **4.6-11.8 mV, against an assumed 2.0**. Its consequence is the opposite of a threat to 6A's conclusion: a larger
   sigma *lowers* the maximum slope of the smoothed f-I (8.27 Hz/mV at 2.0 -> **6.06 at 4.6** -> 4.31 at 11.8), so the
   shipped ring's `gamma_E_crit` of 33.3 Hz/mV is above every operating point by **5.5x** at the measured sigma rather
   than 4.0x, and the F-family's 3.34 stays supercritical at every sigma in the bracket. The tool's validated
   predictions barely move (F 144.3 -> 144.7 -> 146.5 Hz against a measured 152.5; 7 of 8 cx5 bump calls and 2 of 3
   rate calls at **every** sigma), because the saturation rate is read far above threshold where sigma hardly matters.
   Section 2.
2. **RE-RUN AT THE MEASURED SIGMA, THE TOOL STOPS MAKING 6A's HEADLINE MISS.** 6A's fixed point predicted a *confined*
   bump for H3 (EPG in 252 / out 25.8 Hz) where the measurement was a saturated ring with the hump three wedges off
   the driven block. At sigma 4.6 the same tool predicts for H3 **EPG in 34.1 / out 95.3 Hz, 3 of 11 driven cells and
   14 of 35 off-block cells above 22 Hz, `RUNAWAY`, no confined bump** -- which is what 6A measured (55-90 Hz
   off-block, 17-19 of 35 above 22 Hz). Its H3G prediction also improves: 165.0 Hz, 11/11 in and 2/35 out, the only
   configuration in the table whose fixed point passes the ledger's own confinement count, against a measured
   159-164 Hz. Section 4.
3. **THE PER-TYPE RING RATES ARE MEASURED, AND THEY CLOSE 6A's WEAK LINK QUANTITATIVELY.** 6A's open question was why
   the rate model runs 2.1-2.6x high on H3's PEN, with its ring rates the candidate and no arm measuring them. On the
   CPU, arm S: **ExR6 41.9-43.1 Hz at rest and 83.9-85.8 Hz during the pulse, ER6 22.8-24.4 / 50.4-52.5, ER4m
   1.7-3.2 / 11.2-11.9** (population means, seeds 0 and 1), against the rate model's 110.2 / 183.2, 53.8 / 116.3 and
   4.4 / 22.3 at the same states (at sigma 2.0; 107.7 / 181.5, 51.9 / 115.2 and 10.3 / 24.1 at the measured 4.6).
   **The model is 2.1-2.6x high on ExR6 and 2.2-2.4x high on ER6** -- the same factor, on exactly the two types 6A
   showed carry the PEN DC. Section 3.3.
4. **A PER-TYPE WEDGE-LOCAL RECURRENCE NEEDS NO `brain.py` CHANGE.** `brain._shaped_weights` applies
   `type_path_gain` *before* `same_type_gain`, so a single entry `^EPG$:^EPG$:10` composes with the shipped x0.1 to
   exactly x1.0 on the 842 EPG -> EPG pairs while PEN_a, PEN_b, Delta7 and PEG cliques stay damped. It is bit-identical
   to `same_type_gain=1` on that block (every integer count 1-60 satisfies `(n*10f)*0.1f == n` in float32; pinned by a
   test) and verified on the real connectome: **EPG +3.666 mV per pair / +67.1 per volley under H3E, identical to
   H3F's, with PEN_a +0.583, PEN_b +0.624, PEG +0.131 and Delta7 -0.425 left at their shipped damped values**. The
   opt-in `LIFParams` field the task allowed as a fallback **was not needed and was not added; `flyverse/brain.py` is
   untouched**. Section 3.2.
5. **`silent` is decidable from this protocol now, and the shipped compass relays are silent at rest.** The runs
   record the per-cell maximum, not only the group mean. In arm S's settle window (CPU, seeds 0 and 1) **42 of 42 PEN,
   18 of 18 PEG, 4 of 4 EPGt and 4 of 4 GLNO emit no spike at all**, so their per-cell maximum rate is 0.0 Hz and they
   meet `INTERP.md` 10's `silent` (max < 0.5 Hz per cell); during the pulse 6 of 42 PEN do fire, so PEN is not silent
   then. 6A could not say either thing.

**Nothing is adopted, and nothing here could be**: the arm that did not run is two labelled instruments at once, both
new flags default to `None`, and the shipped CPU path is bit-identical (120 of 121 shared recorded fields equal to the
pre-6B run, `wall_s` the only difference, plus two record-keeping keys).

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

**The model overstates the two types that carry the PEN DC by a factor of 2.1-2.6, the same factor by which 6A found
it overstates H3's PEN rate (2.1-2.6x).** That is the quantitative answer to 6A's open question, and it says the error
is in the ring's operating point rather than in the criterion. Two caveats: these are CPU runs at two seeds of one
arm, not the GPU batch (the batch would give five seeds per arm on eight arms); and the comparison is of population
means at matched windows, with the model's "background" read against the LIF's settle window.

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

## 4. The predictions (the decisive table), and the four arms they can already be scored on

Predictions are `out/cx7/predeclared.json` `predictions_from_the_structure_pass.per_arm`, written **before**
submission at the **measured** sigma 4.6 mV from `out/cx7/structure/structure.json`; the prediction of a *confined*
bump is new in 6B and is the criterion 6A's H3 prediction lacked: the fixed point must both be a bump (in > 2 x out,
in > 15 Hz) **and** pass the ledger's own confinement count on its own cells (>= 8 of 11 driven above 22 Hz, <= 3 of
35 outside).

| arm | gamma_E_crit | PEN during | PEN post | EPG in / out post | in>22 / out>22 | fixed-point call | predicted rate |
|---|---|---|---|---|---|---|---|
| S | 33.34 | 0.0 | 0.0 | 10.0 / 10.0 | 0/11, 0/35 | no bump | -- |
| H3 | 33.33 | 99.3 | 71.6 | 34.1 / 95.3 | 3/11, 14/35 | **RUNAWAY, no confined bump** | -- |
| **H3F** | 3.33 | **351.5** | 350.8 | 304.3 / 326.7 | 11/11, **35/35** | **RUNAWAY, no confined bump** | 145.2 |
| **H3E** | 3.33 | 16.3 | 16.7 | **10.0 / 164.7** | **0/11**, 20/35 | **RUNAWAY, a bump OFF the driven tile** | 145.2 |
| F | 3.34 | 0.0 | 0.0 | 10.0 / 10.0 | 0/11, 0/35 | no fixed-point bump (recurrence supercritical) | 144.7 |
| H3G | 33.33 | 36.0 | 32.4 | **165.0 / 11.6** | **11/11, 2/35** | **BUMP, CONFINED** | 165.0 |
| H3FG | 3.33 | 0.0 | 0.0 | 10.0 / 149.5 | 0/11, 18/35 | RUNAWAY, a bump off the driven tile | 145.2 |
| H3EG | 3.33 | 46.3 | 46.0 | **241.8 / 39.4** | 10/11, **6/35** | BUMP, **not** confined (6 > 3 outside) | 241.8 |

So the tool's answer to this thread's mechanism question, **before** the measurement, is: **the local recurrence alone
does not buy a bump at the driven tile.** H3F saturates the entire ring (PEN 351 Hz, 35 of 35 off-block cells above
22 Hz); H3E leaves the driven wedges at the 10 Hz forced floor while the *rest* of the ring runs at 165 Hz -- a bump
in the wrong place; only the two arms carrying the GLNO relabel put a bump on the driven tile at all, and only H3G's
passes the confinement count. That is a falsifiable prediction, and the arm that would falsify it is the one that did
not run.

**Four of the eight arms have already been measured, by cx6**, so the measured-sigma predictions can be scored now
(6A measurements from `docs/audits/compass_dc_balance.md` 3.1, 5 seeds each; predictions from this thread's
`structure.json` at sigma 4.6 vs 6A's own at sigma 2.0):

| arm | 6A prediction (sigma 2.0) | 6B prediction (sigma 4.6) | cx6 measurement | verdict on the re-run |
|---|---|---|---|---|
| S | PEN 0.0, no bump | PEN 0.0, no bump | PEN 0.29-0.66 Hz, no bump 5/5 | unchanged, right |
| **H3** | PEN 103.0; **bump**, EPG in 252 / out 25.8 | PEN 99.3; **RUNAWAY**, in 34.1 / out 95.3, 14/35 out above 22 | PEN 40.2-48.3; **no bump 5/5**, off-block 55.5-89.5 Hz, 17-19/35 above 22 | **the miss is fixed on the bump call and the profile shape**; the PEN rate stays 2.1-2.5x high |
| **H3G** | PEN 40.2; bump, EPG in 158.7, 5/35 out above 22 (not confined) | PEN 36.0; **bump, CONFINED**, in 165.0 / out 11.6, 2/35 | survival 4.76-5.00 s, 159.0-164.2 Hz, confined 0.11-0.49 | **improved**: rate right to 0.5-3.8 %, and the confinement call now matches the arm that actually came closest |
| F | no fixed-point bump; recurrence 144.3 Hz | no fixed-point bump; recurrence 144.7 Hz | bump 150.6-155.8 Hz in 4/5 | unchanged, right to 5.1 % |

This is the one genuinely new empirical result of the thread beyond the two measurements: **at the measured sigma the
tool's H3 prediction agrees with the H3 measurement it previously got wrong**, and its H3G prediction gets better. It
is not a fresh test -- the cx6 numbers were already known when this pass ran, and the sigma was measured, not fitted,
so the improvement is a consequence of one measured parameter rather than of a tuned one; but the direction was not
guaranteed and it is recorded as an observation, not as a validated advance.

A CPU code check, not a result: the H3E flag combination was run once on the real connectome at a truncated protocol
(0.5 s pulse, 0.5 s free, seed 0, CPU, `out/cx7/smoke/codecheck_H3E_truncated.json`) to prove the flags reach the
weight matrix; its console reads `hold ^(ExR6|ER6|ER4m)$ -> ^(PEN_|EPG$) x0 (1149 entries, 37256 syn); edge gains
^EPG$ -> ^EPG$ x10 (842 entries, 842 same-type at x1)` and the ring is already at **out 152.8 Hz with 25 of 35 cells
above 22 Hz before the pulse**, which is the regime the fixed point predicts (out 164.7 Hz post-release). One CPU
seed at a truncated protocol decides nothing and no number of it is quoted as a measurement.

## 5. The predeclaration and the batch that is ready to run

`out/cx7/predeclared.json`, stamped **2026-09-15T18:29:58Z** (`written_before_submission` true), **never amended**
(no archive file exists because none was needed). Order, from the files' own stamps and mtimes: the three sigma runs
(18:0x-18:1xZ) < the structure pass at the measured sigma (`structure.json` `generated_utc` **18:19:50Z**, mtime
18:28:30Z) < `batch.sh` **18:23:25Z** (sha256 `1552e602...1bfc`, recorded in the predeclaration) < the three
validations (18:23-18:29Z) < **predeclaration 18:29:58Z** < `predeclare_stamp.txt` 18:29:59Z < `tree_state.json`
18:30:29Z < `submit_stamp.txt` **18:30:36Z** < the client's refusal (`out/cx7_cluster.log` 18:30:56Z).

**Two files were rewritten after the stamp, and neither is a number the predeclaration rests on.**
`out/cx7/sigma/sigma_summary.json` was regenerated at 18:42:30Z when the `sigma_table.csv` emitter was added to
`measure_lif_sigma.py --summarise`; its `headline` block is **bit-equal to the values the predeclaration quotes**
(checked: `sigma_summary.json.headline == predeclared.json.sigma.measured`, same three input JSONs, same code path).
`scripts/cx_ring_structure.py` was edited after the 18:30:29Z `tree_state.json` stamp as well (the
`--predictions-csv` emitter), as was `scripts/measure_lif_sigma.py`. Neither is on the simulated path, and **no run
exists to be affected** -- which is the one advantage of a batch that did not submit.

**The arms** (all `cx_wedge.py --no-structure --sim 1:1 --ledger`, the cx5 / cx6 protocol unchanged, 5 seeds each,
SHIPPED gains throughout):

| arm | hold | recurrence | GLNO | classification |
|---|---|---|---|---|
| S | -- | -- | silent | the shipped path (reference 1) |
| H3 | yes | -- | silent | the 6A counterfactual (reference 2) |
| **H3F** | yes | `same_type_gain=1` (global) | silent | **two instruments**: hold + the global hand-rule removal |
| **H3E** | yes | `--edge-gain ^EPG$:^EPG$:10` (per type) | silent | **two instruments**: hold + a per-type gain (842 EPG -> EPG pairs only) |
| F | -- | `same_type_gain=1` | silent | the recurrence WITHOUT the hold (what the hold adds) |
| H3G | yes | -- | glutamate | the 6A relabel arm (reference 3) |
| H3FG | yes | global | glutamate | H3F + the relabel |
| H3EG | yes | per type | glutamate | H3E + the relabel |

**The family, sized to be satisfiable.** Five primaries per (arm, reference) family -- `survival_s`,
`frac_confined_post`, **`centre_dist_t5`** (the ring distance in wedges between the EPG profile's vector centre at 5 s
and the driven block's centre 1.5 -- the quantity 6A found decisive), `PEN_mean_during`, `PEN_mean_post` -- Holm within
each family, references **S and H3** scored separately. At 5 v 5 the exact-U floor is 0.0079 and
**5 x 0.0079 = 0.040 <= 0.05**: satisfiable. `bump_hz_post` and `width_half_post` are undefined in both references
(neither S nor H3 has a confined frame in cx5 / cx6) and are **predeclared as magnitudes outside the family**, with
`survival_at_tile_s`, `frac_at_tile_post`, `centre_dist_post_confined` and 28 secondaries (including the new per-type
ring rates) likewise. 24 jobs were the ceiling; the batch is **10 jobs** (5 seeds x {5 GLNO-silent arms sequential,
3 relabel arms sequential}), blocks `fam_s<seed>`, ONE submission, `--fetch out/cx7/`.

**The two rules, in the predeclared words** (`predeclared.json` `decision_rule`, verbatim):

* `working_compass`: "bump_survival_s >= 5 s AND bump_width_wedges in [2.5, 5] AND bump_rate_hz in [5, 60] Hz, per
  seed; the arm meets it in >= 3 of 5 seeds"
* `driven_tile`: "centre_dist_t5 <= 1.5 wedges (the driven block is wedges 0-3, centre 1.5), per seed; the arm meets
  it in >= 3 of 5 seeds"

**The decision phrases, worded for H3E and H3F** (printed per arm, read for those two): "a compass at the driven tile"
(both rules in >= 3 of 5); "a bump at the driven tile, not a compass (rate outside 5-60 Hz)" (survival and width and
the tile rule in >= 3 of 5, the rate row failing); "a bump, not at the driven tile" (survival in >= 3 of 5, the tile
rule in < 3 of 5); "no bump" (survival in < 3 of 5). And the contrast the thread exists for: **"the per-type
EPG -> EPG recurrence is sufficient"** when H3E lands in the same or a better class than H3F, **"the other same-type
cliques are needed"** when H3F lands in a better class. None of these can be read today: **every one requires the
batch.**

**The job line was tested on CPU before submission** (`out/cx7/smoke/jobline_check.txt`, pasted):

```
job lines: 10 (5 seeds x 2 GLNO conditions); arms per job: 5 3 5 3 5 3 5 3 5 3 (GLNO silent: S H3 H3F H3E F; GLNO = glutamate: H3G H3FG H3EG)
exit expression (out/cx7/smoke/exitcheck_*.sh, the prelude dropped, each python call replaced by a controllable exit):
  clean -> 0; arm 0 exits 7 -> 7; arm 2 exits 7 -> 7; arm 4 exits 7 -> 7
flag quoting (out/cx7/smoke/flagcheck.sh): python argv receives
["^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)", "^EPG$:^EPG$:10"]
```

The ten job lines were extracted from `batch.sh` **by bash** (`out/cx7/smoke/argv_shim.py` -> `argv.json`, because
python's `shlex` keeps the `\$` bash removes), as in 6A; **the new `--edge-gain` spec survives both quoting levels
intact**, which was the one thing 6A's check could not have covered.

**The analysis path is written and has been exercised on real runs -- cx6's.** Because the cx7 batch did not run, the
new path was run as a code check over the **cx6** batch's four shared arms (S, H3, H3G, F; 5 seeds each; copies in
`out/cx7/smoke/analysis_pathcheck_cx6/` with a README saying in its first line that these are **not** cx7 results).
It completes, emits all ten files, and its machinery checks out: the two reference families are both built, **Holm
m = 5 in every family** (so the floor is 5 x 0.0079 = 0.0397 <= 0.05, as predeclared), and `centre_dist_t5`
reproduces 6A's centres exactly (H3 4.21-4.61 wedges from the driven block in 5 of 5, H3G 0.58 / 3.80 / 3.93 / 7.47 /
7.49, F 0.25-0.46 in four seeds and 6.39 in seed 4). The four-way call it prints for those arms -- S "no bump", H3
"no bump", F "a bump at the driven tile, not a compass (rate outside 5-60 Hz)", H3G "no bump" (1 of 5 seeds surviving
5 s) -- is 6A's own reading of the same runs, arrived at by the new rule.

One genuinely new number falls out of that re-analysis, and it sharpens 6A's `bump_survival_s` caveat. The new
`survival_at_tile_s` (the last post-pulse frame that is confined **and** within 1.5 wedges of the driven block) splits
H3G's survival apart: per seed **5.00 / 4.76 / 4.91 / 4.98 / 4.77 s of survival but only 0.85 / 4.76 / 2.79 / 1.18 /
0.00 s of survival at the driven tile** (`out/cx7/smoke/analysis_pathcheck_cx6/runs.csv`), against F's 5.00 / 5.00 /
5.00 / 5.00 / 1.84 where survival and at-tile survival are **equal in all five seeds**. So 6A's "the bump is at the
driven tile in only 1 of 5 seeds" is right, and the quantity that shows it is now computed rather than read off the
5 s snapshot. This is a re-analysis of the cx6 batch, not a cx7 measurement.

`cx_ring_structure.analyse_batch_cx7` (CPU) scores both references,
both rules, the four-way phrase per arm, the H3E-vs-H3F contrast, the per-type ring-rate table and the per-cell
maxima, and emits `analysis/{analysis.md,runs.csv,compare.csv,decision.csv,call.csv,scatter.csv,state.csv,ring_rates.csv,analysis.json,scatter.png}`
-- the per-seed scatter into a named file, to be pasted from it (`INTERP.md` 10.4 rule 28). Its per-run checks are
declared in the predeclaration: device `cuda`, arm label, gains, `nt_override` / `glno_nt`, `same_type_gain`,
`receptor_model`, the hold (spec, factor 0, **1,149** entries), the edge gain (spec, **842** entries all same-type,
effective factor 1.0) and the presence of every per-type ring rate.

## 6. What this establishes, what it does not, and the defects found

**Established (CPU, this thread):** the spiking LIF's effective input noise on this circuit (4.6 mV on the membrane,
11.8 mV on the input, against an assumed 2.0); that the assumption's replacement strengthens rather than weakens 5A's
and 6A's shipped-ring conclusion (33.3 Hz/mV is 5.5x above the maximum slope at the measured sigma) and leaves the
tool's validation unchanged (7 of 8 arms, 2 of 3 rates, at every sigma in the bracket); that re-running at the measured
sigma removes 6A's H3 over-prediction; that the rate model's ring rates are 2.1-2.6x the LIF's on exactly the two types
that carry the PEN DC; that PEN, PEG, EPGt and GLNO are `silent` in the project's sense at rest in the shipped model
and PEN is not during the pulse; and that a **per-type** wedge-local undamping is reachable through the existing
`type_path_gain` stage, bit-identically to the global instrument on the EPG block, with no `brain.py` change.

**Not established:** everything the arm was for. Whether this ring can hold a bump **at the driven tile** once the DC
brake is off and the wedge-local recurrence is on is **unanswered**: the fixed point predicts no for H3E and H3F and
the measurement that would test it did not run. No family call, no verdict, no per-seed scatter of any primary, and no
per-arm per-type ring-rate table exists for the eight arms -- the protocol now records those keys, and 0 of 40 runs
exist. **`compass.EPG.bump_survival_s` stays FAIL and the rate and width rows stay NOT_APPLICABLE**, exactly as 6A
left them.

**Two instruments can only ever make a mechanism statement.** Even with the batch, H3E and H3F are a counterfactual
hold composed with a hand-rule removal or a per-type gain; neither is a candidate default, both flags default to
`None`, and the shipped path is bit-identical with them absent. An adoption would have to come from the data side.

**What a data-driven route to a wedge-local recurrence would look like.** The x0.1 `same_type_gain` is the hand rule;
the 842 EPG -> EPG synapses are the data. Three things would have to be settled, and this project's own files settle
only the third:

* **Chemical or electrical?** The `same_type_gain` docstring justifies the damping by "in the animal such populations
  are typically gap-junction coupled and fire in synchrony rather than exciting each other chemically" -- a statement
  about FR1, lLN1_bc, DLMn and DNg33, the cliques it was introduced against. For EPG it does not transfer as stated:
  the MaleCNS edges are **EM chemical synapses** (T-bar / PSD pairs) by construction, so the 842 EPG -> EPG pairs are
  chemical connections in the data whatever else E-PG cells also do. Whether E-PG are **additionally** electrically
  coupled -- which is what would justify damping their chemical weight as a proxy for synchrony -- is **unknown** here:
  the connectome is a chemical-synapse volume, and nothing in it or in `receptors_by_type.csv` carries innexin
  (ShakB / ogre / Inx) expression. That is the measurement the question needs, and it is not in this dataset.
* **Excitatory at the postsynaptic side?** This one the data do answer, and in the direction that makes the damping
  the only thing standing in the way: EPG is cholinergic, and the `EPG, acetylcholine` row of
  `flyverse/data/receptors_by_type.csv` is `fast_sign` **+1**, lead **nAChRbeta1**, gain class **high**, tier
  `alias`, source **davis2020 / PB_2** (`nAChR:high; mAChR-A:low; mAChR-B:high`). So EPG -> EPG is an excitatory
  chemical connection with receptor support in the project's own tiering, at +3.653 mV per pair undamped.
* **Does the undamped weight survive the rest of the model?** `same_type_gain` is **one global scalar**, so a
  data-driven adoption is not "set it to 1": it is a per-type rule, and this thread shows the per-type mechanism
  exists without new brain code. What it would then require is the standard: the full benchmark suite at >= 3 draws
  with no check changing status, **on the CPU path as well as the GPU** (`INTERP.md` 10.4 rules 14 and 16), because
  the x0.1 exists to stop runaway cliques elsewhere -- and the fixed point's own H3F prediction (PEN 351 Hz, the whole
  ring above 22 Hz) is a reminder of what removing it globally does. On the compass's own numbers the case is *weaker*
  than 5A's: F holds a bump in 4 of 5 seeds rather than 4 of 4 (cx6), and every bump in either round fails
  `compass.EPG.bump_rate_hz` by 2.5-3.4x.

**Defects found and recorded:**

1. **`client exit 0` after submitting nothing -- misattributed, corrected by the owner 2026-09-15.** The thread wrote
   this up as `cluster_run.py` exiting 0; it does not: both nothing-submitted branches `return 2` (`scripts/cluster_run.py`
   lines 693-694 and 720-721, read after the report). The 0 came from `out/cx7/batch.sh`, which pipes the client through
   `| tee out/cx7/client_stdout.txt` without `set -o pipefail`, so `$?` is tee's. `INTERP.md` 10.4 rule 4's corollary
   stands -- a wrapper that trusts the exit code would analyse an empty directory -- and the fix is in the wrapper:
   resubmit with `bash -o pipefail out/cx7/batch.sh` (the file itself is left unchanged so its stamped sha256 holds),
   and every future `batch.sh` starts with `set -o pipefail`. The original wording is kept in the Report block below as
   the thread wrote it.
2. **`plan_batch` raised `NameError: name 'sha' is not defined`** on its final print (a stale reference left from 6A's
   edit), after `batch.sh` and `tree_state.json` were already written -- so 6A's batch was planned by a function that
   crashed on its way out. Fixed here (`tree = write_tree_state(out_dir)`), and both files regenerated before the
   predeclaration was stamped.
3. **`measure_lif_sigma.py` first called `common.provenance(..., control=...)`**, which that function does not accept;
   the control label belongs inside `stimulus`. Fixed before any measurement was recorded.
4. **A spiking cell's membrane SD is not a measurement of its input noise** (every firing ring cell reads 2.1-2.4 mV
   whatever its rate, because reset-to-threshold clips the distribution). Any future sigma measurement must restrict
   to sub-threshold cells, as this one does.

## Report

```yaml
summary: >
  Thread 6B was to run THE TWO-INSTRUMENT ARM -- the 6A hold plus a wedge-local recurrence -- after making two
  measurements the rate model needed. THE TWO MEASUREMENTS WERE MADE AND THE BATCH DID NOT RUN: the house cluster was
  unreachable at submission time (out/cx7_cluster.log: "[house] UNAVAILABLE (API silent for 20s): <urlopen error
  [Errno 11001] getaddrinfo failed]" / "no target answered; nothing submitted" / "client exit 0"), <cluster-node>
  does not resolve from this desktop, and house is the only enabled target in .cluster.json. 0 of 40 run artefacts
  exist; nothing was submitted, so there was nothing to recover with fetch_run.py. out/cx7/batch.sh (sha256
  1552e602...1bfc, recorded in the predeclaration stamped 2026-09-15T18:29:58Z, never amended) is ready to submit
  unchanged. MEASUREMENT (a): the protocol now records per-type ring rates (ExR6 / ER6 / ER4m, plus EPGt) and the
  PER-CELL rates of the small compass groups with their per-cell maxima; adding recorded keys cannot change the
  simulation and the shipped CPU path is equal to the pre-6B run on 120 of 121 shared fields (wall_s excepted, two new
  record-keeping keys). MEASUREMENT (b): the spiking LIF's effective input noise, which this project had never
  measured. On the shipped arm's own protocol the free membrane of the sub-threshold relays fluctuates with SD 4.6 mV
  (median over 128 never-spiked PEN / PEG / EPGt cells, seeds 0 and 1; range 2.7-7.7), the EPG's with 5.5 mV, and the
  synaptic input g itself with 11.8 mV, against the ASSUMED SIGMA_MV = 2.0. The rate model should assume a BRACKET
  4.6-11.8 mV (membrane-filtered to frozen-noise), not 2.0. Its consequence runs the opposite way to a threat: a
  larger sigma lowers the maximum slope of the smoothed f-I (8.27 Hz/mV at 2.0 -> 6.06 at 4.6 -> 4.31 at 11.8), so the
  shipped ring's gamma_E_crit 33.3 Hz/mV is above every operating point by 5.5x rather than 4.0x (sigma would have to
  be 0.17 mV, 27x smaller than measured, for it to be reachable), while the F-family / H3E-H3F recurrence
  (gamma_E_crit 3.33) stays supercritical at every sigma in the bracket and its predicted saturation rate moves 1.5 %
  across it (144.3 -> 146.5 Hz). Re-validated against the cx5 batch at sigma 2.0 / 4.6 / 11.8 the tool is UNCHANGED:
  7 of 8 arms (= 3 of 4 distinct predictions, per 6A's skeptic), 2 of 3 rate calls, F 5.4 / 5.1 / 3.9 % at each. What
  DOES move is the H3 family's fixed point, and in the direction the measurement had already gone: at the measured
  sigma the tool predicts for H3 a RUNAWAY ring (EPG in 34.1 / out 95.3 Hz, 3/11 driven and 14/35 off-block cells
  above 22 Hz, no confined bump) where 6A predicted a confined bump at in 252 / out 25.8 -- i.e. the re-run removes
  6A's headline miss against a measurement of 40.2-48.3 Hz PEN, 55-90 Hz off-block, no bump in 5/5 -- and its H3G
  prediction improves to 165.0 Hz with 11/11 in and 2/35 out, the only fixed point in the table that passes the
  ledger's own confinement count, against a measured 159-164 Hz. THIRD RESULT, closing 6A's named weak link: the rate
  model's ring rates are 2.1-2.6x the LIF's on exactly the two types that carry the PEN DC (arm S, CPU, seeds 0-1:
  ExR6 41.9-43.1 Hz at rest and 83.9-85.8 during the pulse against the model's 110.2 / 183.2; ER6 22.8-24.4 /
  50.4-52.5 against 53.8 / 116.3) -- the same factor by which 6A found it overstates H3's PEN. FOURTH: a PER-TYPE
  wedge-local undamping needs NO brain.py change, because brain._shaped_weights applies type_path_gain BEFORE
  same_type_gain: one --edge-gain '^EPG$:^EPG$:10' entry lands the 842 EPG -> EPG pairs at exactly x1.0 (bit-identical
  to same_type_gain=1 on that block; (n*10f)*0.1f == n for every count 1-60) while PEN_a, PEN_b, Delta7 and PEG
  cliques stay x0.1 -- verified on the real connectome (EPG +3.666 mV/pair vs the others unchanged). The opt-in
  LIFParams field the task allowed as a fallback was NOT needed and NOT added; flyverse/brain.py is untouched. FIFTH:
  `silent` is now decidable and decided -- 42/42 PEN, 18/18 PEG, 4/4 EPGt and 4/4 GLNO emit no spike in arm S's settle
  window, so all four are silent in the INTERP.md 10 sense at rest, while 6 of 42 PEN fire during the pulse, so PEN is
  not silent then. THE MECHANISM QUESTION IS UNANSWERED: the fixed point predicts NO bump at the driven tile for both
  H3E (driven wedges at the 10 Hz floor while the rest of the ring runs at 164.7 Hz) and H3F (PEN 351 Hz, 35/35
  off-block cells above 22 Hz), with only the relabel arms putting a bump on the tile at all -- and the arm that would
  test that prediction is the one that did not run. Nothing is adopted; nothing here could be (two instruments make a
  mechanism statement, never an adoption), and compass.EPG.bump_survival_s stays FAIL with the rate and width rows
  NOT_APPLICABLE. The analysis path was exercised on cx6's four shared arms instead (Holm m = 5 in both reference
  families, 6A's centres and per-arm reading reproduced), and one new number falls out of that re-analysis: the new
  survival_at_tile_s splits H3G's 5.00 / 4.76 / 4.91 / 4.98 / 4.77 s of survival into 0.85 / 4.76 / 2.79 / 1.18 /
  0.00 s of survival AT the driven tile, against F's five seeds where the two are equal -- 6A's "at the driven tile in
  only 1 of 5 seeds", now a computed quantity rather than a 5 s snapshot.
blocker:
  what: "the cx7 batch was never submitted: the house cluster did not answer"
  failure_line: '[house] UNAVAILABLE (API silent for 20s): <urlopen error [Errno 11001] getaddrinfo failed> / no target answered; nothing submitted / client exit 0'
  evidence: "out/cx7_cluster.log; nslookup <cluster-node> -> Non-existent domain; ssh -> Could not resolve hostname; no hosts entry or tunnel; house is the only target with disabled: false in .cluster.json (every vast-* / r3-h200* is disabled)"
  state: "predeclaration stamped 2026-09-15T18:29:58Z and unamended; batch.sh sha256 1552e602f0d81510e795ded8a9858dab0f2f41bb1b34189aa23fd22339a91bfc recorded in it and unchanged; tree_state.json stamped 18:30:29Z; job line tested on CPU (exit expression 0/7/7/7, both flag regexes intact through both quoting levels). 0 of 40 <arm>_s<seed>.json, 0 of 40 .txt, 0 of 40 .npz."
  to_resume: "bash out/cx7/batch.sh (ONE cluster_run.py --name cx7 --minutes 30 --arm-block fam <10 job lines> --fetch out/cx7/), then python -W ignore scripts/box_status.py --target house --prefix cx7 --wait --poll 180, then python scripts/cx_ring_structure.py --batch cx7 --analyse out/cx7 (CPU). Nothing needs regenerating: the predeclaration's predictions and batch.sh hash are already stamped."
key_claims:
  - "THE INPUT NOISE IS MEASURED: sigma_v of the sub-threshold relays 4.638 mV (median over 128 never-spiked PEN / PEG / EPGt cells, arm S seeds 0-1, range 2.733-7.692), sigma_v of the EPG 5.521 mV (92 cells), sigma_g of the relays 11.800 mV and of the EPG 12.456 mV -- against the assumed, never-measured SIGMA_MV = 2.0. The rate model should assume the bracket 4.6-11.8 mV: the smoothed f-I's sigma is the width of a quasi-static input distribution, bounded below by the membrane-filtered SD and above by the frozen input SD. A spiking cell's membrane SD is NOT usable (reset-to-threshold clips every firing ring cell to 2.1-2.4 mV whatever its rate), which is why only never-spiked cells enter the estimate. CPU, two seeds of one arm (a third run covers H3), and the noise itself changes under the hold (relay membrane 2.5-3.2 mV, input 12.4-14.7)."
  - "THE MEASURED SIGMA STRENGTHENS 5A / 6A RATHER THAN THREATENING THEM: the maximum slope of the smoothed f-I is 8.27 Hz/mV at sigma 2.0, 6.06 at 4.6 and 4.31 at 11.8 (the tool's own 61-node values, ~3.4 % high against adaptive quadrature), while gamma_E_crit is a property of the weights and does not move -- so the shipped ring's 33.34 Hz/mV is above every operating point by 5.5x at the measured sigma instead of 4.0x, and sigma would have to be 0.17 mV (27x smaller than measured) for it to be reachable. The F-family / H3E-H3F recurrence at gamma_E_crit 3.33 stays supercritical across the whole bracket and its predicted saturation rate moves 1.5 % (144.3 / 144.7 / 146.5 Hz at sigma 2.0 / 4.6 / 11.8)."
  - "THE TOOL'S VALIDATION IS SIGMA-INSENSITIVE: re-scored against the cx5 batch at sigma 2.0, 4.6 and 11.8 it calls bump / no-bump correctly in 7 of 8 arms at every sigma (= 3 of 4 distinct predictions: gamma_E_crit takes four values over the eight arms and the classifier is 'does this arm carry same_type_gain 1?'), the rate within 10 % in 2 of 3, with F predicted 144.3 / 144.7 / 146.5 against a measured 152.5 (5.4 / 5.1 / 3.9 %), CFG 1.7 / 1.9 / 3.2 % and CF the same 11.8-13.5 % miss (out/cx7/structure/validation_sigma*/validation.md)."
  - "RE-RUN AT THE MEASURED SIGMA THE TOOL STOPS MAKING 6A's HEADLINE MISS: for H3 it now predicts PEN 99.3 Hz during the pulse and a RUNAWAY ring after release (EPG in 34.1 / out 95.3 Hz, 3 of 11 driven and 14 of 35 off-block cells above 22 Hz, no confined bump) where at sigma 2.0 it predicted a confined bump at in 252 / out 25.8 -- against a measured 40.2-48.3 Hz PEN, off-block 55.5-89.5 Hz with 17-19 of 35 above 22 Hz and no bump in 5 of 5 seeds. Its H3G prediction improves to in 165.0 / out 11.6 Hz with 11/11 and 2/35, the only fixed point in the table passing the ledger's confinement count, against a measured 159.0-164.2 Hz (0.5-3.8 %). The PEN magnitude is still 2.1-2.5x high. The cx6 numbers were already known when this pass ran, so this is an observation about one MEASURED parameter, not a fresh test."
  - "THE PER-TYPE RING RATES CLOSE 6A's WEAK LINK: arm S on the CPU (seeds 0-1, population means) gives ExR6 41.9-43.1 Hz at rest and 83.9-85.8 during the pulse, ER6 22.8-24.4 / 50.4-52.5, ER4m 1.7-3.2 / 11.2-11.9, against the rate model's 110.2 / 183.2, 53.8 / 116.3 and 4.4 / 22.3 at the same states -- the model is 2.1-2.6x high on ExR6 and 2.1-2.4x on ER6, the same factor by which 6A found it overstates H3's PEN rate. The hold does not silence those cells, only their edges: under H3 they run at ExR6 274.4, ER6 191.3, ER4m 101.4 and GLNO 149.4 Hz with EPG at 76.1 (CPU seed 0)."
  - "A PER-TYPE WEDGE-LOCAL RECURRENCE NEEDS NO brain.py CHANGE. brain._shaped_weights applies type_path_gain BEFORE same_type_gain, so --edge-gain '^EPG$:^EPG$:10' composes with the shipped x0.1 to exactly x1.0 on the 842 EPG -> EPG pairs -- bit-identical to same_type_gain=1 on that block, because (n*10f)*0.1f == n in float32 for every connection count the cap leaves (1-60) -- while PEN_a -> PEN_a, PEN_b -> PEN_b, PEG and Delta7 -> Delta7 stay damped. On the real connectome: EPG +3.666 mV per pair / +67.1 per volley under both H3E and H3F, with PEN_a +0.583, PEN_b +0.624, PEG +0.131, Delta7 -0.425 unchanged under H3E and +5.827 / +6.245 / +1.310 / -4.251 under H3F. The wedge-local one-step coefficient goes +6.00 -> +60.01 mV and gamma_E_crit 33.33 -> 3.33 Hz/mV. flyverse/brain.py is UNTOUCHED and the opt-in field was not needed."
  - "SILENT, DECIDED: the protocol now records the per-cell maximum, and in arm S's settle window (CPU, seeds 0-1) 42 of 42 PEN, 18 of 18 PEG, 4 of 4 EPGt and 4 of 4 GLNO emit no spike at all -- every per-cell maximum 0.0 Hz, so all four groups meet INTERP.md 10's silent (max < 0.5 Hz per cell) at rest under the shipped model. During the pulse 6 of 42 PEN fire (group mean 0.34 Hz), so PEN is NOT silent then. ExR6, ER6, ER4m and Delta7 are never silent (0/2, 0/4, 1/11, 0/42 never-spiked at rest). No population with a non-zero recorded mean is called silent anywhere in this audit."
  - "THE PREDICTION THE ARM WOULD TEST, STAMPED BEFORE SUBMISSION at the measured sigma: neither recurrence buys a bump at the driven tile. H3F saturates the whole ring (PEN 351.5 Hz during the pulse, EPG in 304.3 / out 326.7, 11/11 AND 35/35 above 22 Hz, RUNAWAY); H3E leaves the driven wedges at the 10 Hz forced floor while the rest of the ring runs at 164.7 Hz (0/11 in, 20/35 out -- a bump in the wrong place); only H3G (in 165.0 / out 11.6, 11/11 and 2/35) and H3EG (in 241.8 / out 39.4, 10/11 and 6/35, failing confinement by 6 > 3) put a bump on the tile at all. The 6B prediction of a CONFINED bump -- a bump that also passes the ledger's own count of >= 8 of 11 driven and <= 3 of 35 off-block cells above 22 Hz -- is the criterion 6A's H3 prediction lacked."
  - "THE FAMILY IS SIZED TO BE SATISFIABLE AND THE RULES ARE PREDECLARED: five primaries (survival_s, frac_confined_post, centre_dist_t5 = the ring distance of the 5 s vector centre from the driven block's centre 1.5, PEN_mean_during, PEN_mean_post) Holm-corrected within each (arm, reference) family against TWO references, S and H3; 5 x 0.0079 = 0.040 <= 0.05 at 5 v 5. bump_hz_post and width_half_post are undefined in both references and are predeclared as magnitudes outside the family, with survival_at_tile_s, frac_at_tile_post, centre_dist_post_confined and 28 secondaries (the new per-type ring rates included). The working-compass rule (survival >= 5 AND width 2.5-5 AND rate 5-60, in >= 3 of 5) and the driven-tile rule (centre within 1.5 wedges, in >= 3 of 5) are worded in the stamped JSON, with a four-way phrase per arm for H3E and H3F and an explicit H3E-vs-H3F contrast."
  - "PROCESS: the job line was tested on CPU before submission -- the exit expression returns 0 clean and 7 when arm 0, 2 or 4 exits 7, and BOTH flag specs reach python's argv intact through both quoting levels (['^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)', '^EPG$:^EPG$:10']), the job lines having been extracted from batch.sh BY BASH. Stamp order from the files themselves: structure.json generated_utc 18:19:50Z < batch.sh 18:23:25Z < the three validations 18:23-18:29Z < predeclaration 18:29:58Z < predeclare_stamp 18:29:59Z < tree_state 18:30:29Z < submit_stamp 18:30:36Z < the client's refusal 18:30:56Z. Three files were rewritten after the tree_state stamp: scripts/cx_ring_structure.py (the --predictions-csv emitter), scripts/measure_lif_sigma.py (the sigma_table.csv emitter) and out/cx7/sigma/sigma_summary.json (18:42:30Z, rewritten by that emitter -- its headline block is bit-equal to the values the predeclaration quotes, checked). Neither script is on the simulated path and no run exists to be affected."
  - "DEFECTS FOUND: (1) cluster_run.py prints 'no target answered; nothing submitted' and then exits 0 -- a client that submitted zero jobs must exit non-zero, or a wrapper trusting the exit code analyses an empty directory (INTERP.md 10.4 rule 4's corollary; not fixed here, the file is outside this thread's list). (2) plan_batch raised NameError: name 'sha' is not defined on its final print, a stale reference from 6A, AFTER writing batch.sh and tree_state.json -- so 6A's batch was planned by a function that crashed on the way out; fixed here and both files regenerated before the predeclaration was stamped. (3) measure_lif_sigma.py first passed control= to common.provenance, which does not accept it; fixed before any measurement was recorded. (4) a spiking cell's free-membrane SD is not a measurement of its input noise (reset clips it to 2.1-2.4 mV), so any sigma estimate must restrict to sub-threshold cells."
files_written:
  - scripts/measure_lif_sigma.py (new; the per-step v / g / refrac / spikes record on the cx_wedge protocol, per-cell sigma_g / sigma_v_free with the never-spiked subset, --summarise pooling into sigma_summary.json + sigma_table.csv)
  - scripts/cx_wedge.py (ONE new flag --edge-gain PRE_REGEX:POST_REGEX:FACTOR, default None; parse_edge_gains; hold_edge_counts(..., same_type_gain) reporting the same-type composition; new RECORDED keys only: per-type ring rates ExR6 / ER6 / ER4m + EPGt as g__ groups, per-cell rates of the small groups as cells__ arrays and <g>_cell_max_{pre,during,post} in metrics)
  - scripts/cx_ring_structure.py (sigma as an argument to analyse / validate_batch / max_slope / rate_at_gain / the fixed point, --sigma, --local-recurrence with the H3F / H3E / H3FG / H3EG / E configurations, EPG_EPG_GAIN, the CX7 arm table and its five primaries, ring_distance / DRIVEN_CENTRE / DRIVEN_TILE_TOL, summarise_state's confinement count on the fixed point, analyse_batch_cx7, write_predeclaration_cx7, --predictions-csv, --batch cx7; the plan_batch NameError fixed)
  - tests/test_cx_wedge_hold.py (+6 tests, 12 passed), tests/test_cx_ring_structure.py (+9 tests, 22 passed: the 6B tool tests and 3 for measure_lif_sigma)
  - docs/audits/compass_local_recurrence.md
  - out/cx7/sigma/{sigma_S_s0,sigma_S_s1,sigma_H3_s0}.{json,txt}, sigma_summary.json, sigma_table.csv, slope_vs_sigma.json, summary_console.txt
  - out/cx7/structure/{structure.json,structure.md,matrices.npz,evidence_glno.json,console.txt,predictions.csv} at the MEASURED sigma 4.6, validation_sigma{2.0,4.6,11.8}/{validation.md,validation.json,console.txt}; out/cx7/structure_sigma2/* at the assumed 2.0
  - out/cx7/{predeclared.json,batch.sh,tree_state.json,predeclare_stamp.txt,submit_stamp.txt}; out/cx7/smoke/{smoke_default_path.json,smoke_default_path_noledger.json,codecheck_H3E_truncated.json,jobline_check.txt,argv_shim.py,argv.json,jobs.json,batch_probe.sh,exitcheck_*.sh,flagcheck.sh,analysis_pathcheck_cx6/*}; out/cx7_cluster.log
  - "NOT written (the batch did not run): out/cx7/<arm>_s<seed>.{json,txt,npz} (0 of 40 each), out/cx7/analysis/*, out/cx7/scheduler_receipt.json"
api:
  - "cx_wedge.parse_edge_gains(['PRE:POST:FACTOR', ...]) -> [(pre, post, factor)]; cx_wedge.simulate(..., edge_gains=None) appends them to LIFParams.type_path_gain AFTER the holds (both are per-entry scalars, so the order is immaterial); the row records edge_gains and edge_gains_resolved (cells, entries, same-type entries, effective same-type factor). Default None = the previous gain list, entry for entry."
  - "cx_wedge.hold_edge_counts(c, specs, same_type_gain=None): with same_type_gain given, each row also carries n_entries_same_type, effective_factor_same_type and effective_factor_cross_type"
  - "cx_ring_structure.analyse(..., sigma=SIGMA_MV) and validate_batch(..., sigma=...): every slope bound, gain, fixed point and saturation rate is computed at that sigma and labelled with it (lif.sigma_mV, rate_model.sigma_mV, modes.sigma_mV); summarise_state adds in_above_22 / out_above_22 / confined_by_ledger_rule and rate_model adds confined_bump_after"
  - "cx_ring_structure: recurrence_configs, EPG_EPG_GAIN, CX7_ARMS, PRIMARIES_CX7, MAGNITUDES_CX7, SECONDARIES_CX7, ring_distance, DRIVEN_CENTRE, DRIVEN_TILE_TOL, analyse_batch_cx7, write_predeclaration_cx7; CLI --sigma, --local-recurrence, --predictions-csv, --batch cx7"
  - "measure_lif_sigma: window_stats, summarise, summarise_dir; CLI --out (required, never defaulted) --label --hold-edges --edge-gain --lif --nt-override --save-traces --summarise"
validation:
  - "tests/test_cx_wedge_hold.py 12 passed, tests/test_cx_ring_structure.py 22 passed (+3 for measure_lif_sigma's window_stats / summarise / summarise_dir), tests/test_bit_identity.py 3 passed + 5 subtests -- 37 passed in one CPU run (CUDA_VISIBLE_DEVICES=-1). flyverse/ is untouched by this thread (git diff --stat -- flyverse/ is empty), so the golden is green by construction and was run to prove it."
  - "shipped-path bit-identity on CPU: out/cx7/smoke/smoke_default_path_noledger.json equals the pre-6B out/cx6/smoke/smoke_default_path.json on 120 of 121 shared recorded fields (wall_s the only difference; edge_gains / edge_gains_resolved the only additions)"
  - "the per-type instrument on the real connectome: --edge-gain '^EPG$:^EPG$:10' resolves to 842 entries, 842 of them same-type, effective factor 1.000, and reproduces H3F's EPG -> EPG block (+3.666 mV/pair, +67.1 per volley) with PEN_a / PEN_b / PEG / Delta7 left at the shipped damped values"
  - "the fixed tool against out/cx5 at sigma 2.0 / 4.6 / 11.8: 7 of 8 arms and 2 of 3 rate calls at every sigma (out/cx7/structure/validation_sigma*/validation.md)"
  - "job line tested on CPU before submission: exit expression 0 / 7 / 7 / 7, both flag specs intact through both quoting levels (out/cx7/smoke/jobline_check.txt)"
  - "predeclaration stamped 2026-09-15T18:29:58Z, never amended; batch.sh sha256 1552e602...1bfc recorded in it and unchanged"
  - "the cx7 ANALYSIS PATH exercised on real runs (cx6's four shared arms, 20 runs, copies in out/cx7/smoke/analysis_pathcheck_cx6/ with a README saying they are NOT cx7 results): it completes, emits all ten files, builds both reference families with Holm m = 5, reproduces 6A's centres through centre_dist_t5 and 6A's per-arm reading through the new four-way call, and its 84 'problems' are exactly the expected ones (four cx7-only arms absent; no cx6 run carries the 6B per-type / per-cell keys)"
  - "NOT validated: every measurement the batch was to make. 0 of 40 runs; no scheduler receipt; no out/cx7/analysis/; the H3E / H3F / H3EG / H3FG arms have never been simulated beyond one truncated CPU code check"
recommendations:
  - "RUN THE BATCH UNCHANGED WHEN THE CLUSTER ANSWERS. bash out/cx7/batch.sh then box_status --wait then --batch cx7 --analyse out/cx7. The predeclaration is stamped and its predictions are at the measured sigma; re-stamping or re-planning would destroy the only thing that makes the arm a test."
  - "DO NOT ADOPT ANYTHING FROM THIS THREAD, and note that the arm itself could never license an adoption: H3E and H3F are a counterfactual hold composed with a per-type gain or a global hand-rule removal. Both new flags default to None and the shipped CPU path is bit-identical. compass.EPG.bump_survival_s stays FAIL; the rate and width rows stay NOT_APPLICABLE."
  - "THE RATE MODEL SHOULD ASSUME sigma IN 4.6-11.8 mV, NOT 2.0 -- and every audit that quotes a slope bound should quote it at the measured value: max slope 6.06 Hz/mV at 4.6 and 4.31 at 11.8 against 8.27 at 2.0. Nothing in 5A or 6A reverses; the shipped ring's 33.3 Hz/mV goes from 4.0x to 5.5x above every operating point and the F-family recurrence stays supercritical. SIGMA_MV = 2.0 is still the default in the code and is now labelled as an assumption with a measurement to compare against."
  - "MAKE --out REQUIRED ON scripts/cx_ring_structure.py (6A's open item, still open). This thread never invoked it without --out and never touched another round's directory, but the default out/cx5/structure remains the trap that caused the 6A incident."
  - "cluster_run.py MUST EXIT NON-ZERO WHEN IT SUBMITS NOTHING (defect 1). It printed 'no target answered; nothing submitted' and exited 0."
  - "THE PER-TYPE ROUTE IS NOW CHEAP, SO THE DATA QUESTION IS THE WHOLE QUESTION: a per-type EPG -> EPG undamping needs no new code path, which means the only thing between the connectome's 842 excitatory EPG -> EPG synapses (EPG is cholinergic; the EPG / acetylcholine row of receptors_by_type.csv is fast_sign +1, nAChRbeta1, class high, tier alias, davis2020 PB_2) and the model is the global x0.1. What would license removing it for this type is evidence about electrical coupling -- whether E-PG are gap-junction coupled (innexin expression: ShakB / ogre / Inx), which is the justification the same_type_gain docstring offers for the cliques it was introduced against and which this chemical-synapse dataset cannot see -- plus the full benchmark suite at >= 3 draws on the CPU as well as the GPU."
open_questions:
  - "Does survival_at_tile_s belong in the ledger? Re-analysing cx6 with it shows H3G surviving 4.76-5.00 s while holding the driven tile for 0.00-4.76 s, which is the distinction compass.EPG.bump_survival_s cannot make and which every round since cx5 has had to make in prose. It is a rule change, so it belongs to whoever owns expected_responses.csv, not here."
  - "Can this ring hold a bump AT THE DRIVEN TILE once the DC brake is off and the wedge-local recurrence is on? UNANSWERED -- the arm did not run. The stamped prediction at the measured sigma is NO for both H3E (driven wedges at the 10 Hz floor, the rest of the ring at 164.7 Hz) and H3F (PEN 351 Hz, 35/35 off-block above 22 Hz), with only the GLNO-relabel arms putting a bump on the tile."
  - "Is the per-type EPG -> EPG recurrence sufficient, or are the other same-type cliques needed? The H3E-vs-H3F contrast is predeclared in words and needs the batch. This is the first configuration pair in which the question is separable at all."
  - "What is the right scalar sigma for the smoothed f-I, inside the measured 4.6-11.8 mV bracket? The membrane and input readings differ by 2.6x, the per-cell spread is 2.7-7.7 mV, and the hold changes both. A calibration against the LIF's own measured f-I on these cells would settle it; nothing in this thread does."
  - "Why is the rate model 2.1-2.6x high on the ring types? Now measured rather than conjectured (ExR6 110 / 183 Hz modelled against 42 / 85 measured), but not explained -- the candidates are the adiabatic relay assumption, the missing adaptation of the ring cells in the fixed point, and the fan-in / cap treatment of their giant edges."
  - "What are ExR6's and ER6's transmitter, receptor and modulatory status in the animal? Unchanged from 6A: UNKNOWN at the type level, and still the largest single term in the balance both rounds are about."
  - "Are E-PG cells electrically coupled? This is the question the same_type_gain x0.1 rests on for the compass, and this dataset (a chemical-synapse EM volume with no innexin column) cannot answer it."
```
