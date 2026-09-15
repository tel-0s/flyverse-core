# Compass, thread 6A: the DC-balance test

Scripts: `scripts/cx_ring_structure.py` (the 5A structure tool, with its three named defects fixed and the 5A behaviour
kept behind `--legacy`; new modes `--holds`, `--predeclare`, `--tree-state`, `--validate-batch`, `--batch cx6`),
`scripts/cx_wedge.py` (ONE new flag: `--hold-edges PRE_REGEX:POST_REGEX`, default `None`, shipped path bit-identical),
`tests/test_cx_ring_structure.py` (+8 tests for the fixes), `tests/test_cx_wedge_hold.py` (new, 6 tests).
Data: `out/cx6/structure/{structure.json, structure.md, matrices.npz, validation.md, validation.json}`,
`out/cx6/predeclared.json` (stamped **2026-09-15T08:09:25Z**, never amended), `out/cx6/{batch.sh, tree_state.json,
submit_stamp.txt, predeclare_stamp.txt, scheduler_receipt.json, scheduler_receipt.py}`, `out/cx6/smoke/*` (the CPU
smokes and the job-line check), `out/cx6/<arm>_s<seed>.{json,txt,npz}` (40 runs), `out/cx6_cluster.log`,
`out/cx6/analysis/{analysis.md, runs.csv, compare.csv, decision.csv, call.csv, scatter.csv, state.csv,
cross_batch_cx5.csv, analysis.json, scatter.png}`.
Tree: HEAD a9946cd (= origin/main); nothing committed by this thread. `tree_state.json` lists every file that shipped
with the batch with its sha256 -- this thread's `scripts/cx_ring_structure.py`, `scripts/cx_wedge.py`,
`tests/test_cx_ring_structure.py`, `tests/test_cx_wedge_hold.py`, and another thread's concurrent
`flyverse/body.py`, `scripts/probe_vnc_drive.py`, `tests/test_body_cycle.py`, `uv.lock` (none on the simulated path:
`cx_wedge.py --sim` builds a `FlyBrain` without a world).

## 0. Answer

**The DC balance is necessary and not sufficient.** Holding ExR6 + ER6 + ER4m at 0 onto PEN and EPG -- an `edges`-kind
LABELLED COUNTERFACTUAL, 17 presynaptic cells, 1,149 weight entries, 37,256 synapses -- lifts the driven PEN population
from **0.29-0.66 Hz to 40.2-48.3 Hz during the pulse** (5 v 5, diff +43.53 Hz, z 294.0, p 0.0079, Holm 0.0317,
`result`) and to 49.3-54.0 Hz after release. That is the 5A attribution, tested: the relays are held below threshold by
2 ExR6 + 4 ER6 + 11 ER4m, and removing exactly those edges makes them fire ~100x harder. **It buys no bump.**
`bump_survival_s` is 0.00 and `frac_confined_post` 0.000 in **5 of 5 seeds** (a structural zero-vs-zero null against S,
which is also 0.00), because the ring does not settle into a confined bump: it saturates. Under H3 the 10 Hz background
*alone* already drives EPG to 19.7-62.9 Hz, PEN to 8.8-42.7 Hz, Delta7 to 28.3-103.5 Hz and GLNO to 32.5-131.4 Hz
during the 1 s settle, before any pulse; at 5 s the EPG profile has a wide hump centred at wedge **5.7-6.1** -- not the
driven block, whose centre is 1.5 -- with **17-19 of the 35 off-block cells above 22 Hz**. The same DC term that keeps
the relays below threshold is what keeps the resting compass at rest.

Per type, the DC on PEN is **ExR6 first**: holding ExR6 alone gives PEN 8.9-14.9 Hz (+10.37, z 70.0, `result`) and a
bump that is confined in 8-59 % of the *pulse* frames but 0.0 % after release; ER6 alone gives 2.4-3.1 Hz (+2.25,
z 15.2, `result`); **ER4m alone changes nothing on PEN** (0.29-0.70 Hz, diff +0.03, z 0.22, p 0.69, `null`) -- ER4m is
a 125 mV-per-volley term onto EPG and only 2 mV onto PEN. The one configuration that comes close to a bump is the hold
*with* the correct GLNO sign: H3G survives 4.76-5.00 s (1 of 5 seeds >= 5 s), runs at 159-164 Hz, width 3.85-4.00,
confined in 11-49 % of post-pulse frames -- and its bump sits at the driven tile in only 1 of 5 seeds (centre 0.92;
the others 5.30, 5.43, 10.01, 10.03). **No arm meets the predeclared working-compass rule in any seed**: every
surviving bump in this batch fails `compass.EPG.bump_rate_hz` 2.5-3.4x (F 150.6-155.8, H3G 159.0-164.2, R 200.6-203.2
against the 5-60 Hz row).

The structure tool was fixed first and validated against the batch it had missed. With the forced drive entered as a
current, the one-step EPG -> EPG term kept, and per-cell gains at the realised fixed point, it calls bump / no-bump
correctly in **3 of the 4 distinct predictions** the eight shipped-gain cx5 arms carry (7 of 8 arms, but `gamma_E_crit`
takes only four values over them, so each prediction is made twice -- 1.2) and predicts the saturation rate of the arms
that held one to **5.4 %** (F: 144 vs 152.5 Hz) and **1.7 %** (CFG: 160 vs 157.1) -- and it predicts *no* bump for S, G,
C and CG, because their critical gain is 28.8-33.3 Hz/mV against a maximum smoothed-f-I slope of **8.00 Hz/mV at the
sigma = 2 mV this rate model assumes** (a comparison 5A never had: the Gaussian smoothing bounds the slope, and the
"25.8 Hz/mV at u 7.1" of the 5A skeptic pass is a point on the *deterministic* f-I's divergence at threshold, not a
maximum; the bound is a property of the assumed sigma rather than a measurement, and the 15-node `lif_fi` the fixed
point actually iterates is not smooth near threshold -- 1.4). The fixed tool's ranking puts the hold arms first, where
5A's put them nowhere
-- but it also over-predicts: it said H3 would hold a bump (PEN 103 Hz, EPG in 252 / out 25.8 Hz after release), and
the measurement is PEN 40-48 Hz with no confinement at all. **Nothing is adopted; a hold is a counterfactual and the
flag that implements it defaults to off.**

## 1. The structure tool (`scripts/cx_ring_structure.py`)

### 1.1 The three defects and what replaced them

Each fix is the default; `--legacy` (= `--drive rate --reduction two-step --gain uniform`) reproduces 5A exactly and is
what the comparison column below is computed with.

| # | 5A | 6A (default) | effect |
|---|---|---|---|
| 1 | `rate_fixed_point` added the forced background as a **rate** (`r = f(tau A r) + forced`), so a driven EPG sat at u -22 to -68 mV while "firing" and `gamma_EPG = 0` **by construction** | the drive enters as a **current**: `r = max(f(tau A r + u_forced), r_forced)` with `u_forced = f^-1(rate)` (`u_for_rate`, derived: 10 Hz -> **6.6276 mV**, 50 Hz -> **11.9933 mV** at sigma 2 mV) | the driven EPG has a real gain wherever it is above threshold; the forced rate stays a **floor** because `FlyBrain.stimulate` forces spikes through `poisson_p` (ORed with the threshold crossing in `brain.py`), and without that floor the same fixed point collapses the whole ring to 2.7 Hz against a measured 8.7-10.0 Hz (`analysis/state.csv`, `epg_mean_pre` for S; `--drive current-nofloor` keeps the literal reading) |
| 2 | the EPG-only reduction dropped the one-step EPG -> EPG term (`M16['net'] = PEN + PEG + Delta7 + Ring`; `direct` printed in mV in the same row and never used) | the term is kept: the loop gain of ring mode k is `gamma tau d_k + gamma^2 tau^2 lambda_k` (`gamma_crit_combined`), and the **wedge-local** one-sided criterion (below) is what the bump call uses | the F family stops being invisible: its one-step coefficient is +59.79 mV wedge-local against the shipped +6.00 |
| 3 | one uniform gamma (the audit read the modes at gamma 6; its skeptic recomputed the true Jacobian by hand) | `jacobian_modes`: `J = diag(f'(u_i)) tau A` at each realised state, with `f'` computed by **integration by parts** -- `f_sigma'(u) = (1/sigma) E[x f_det(u + sigma x)]`, which removes the integrable singularity of `f_det'` at threshold; a cell whose rate is the forced floor gets gamma 0 | the gains are reportable per group and per state, and the maximum slope of the smoothed f-I becomes a computed quantity: **8.00 Hz/mV at u 8.61 (25.5 Hz)** at the assumed sigma = 2 mV (`structure.json` records the tool's own 61-node estimate, `max_lif_slope` 8.2747 at u 8.6018 / 25.13 Hz, which is 3.4 % high; 1.4) |

**The bump criterion the tool is validated on.** A bump is a one-sided local excess, so the return path that matters is
the **wedge-local** band (|post - pre| <= 1) of the kernels, not the k = 1 Fourier component of the whole ring: the
k = 1 component of the Delta7 path is positive (+1,719 mV^2) only because its far-side lobe is negative, and cashing
that in requires the off-bump EPGs to *drop*, which they cannot -- they are already at the forced background floor.
So

    gamma_E_crit = 1 / (tau d_local + tau^2 sum_X gamma_X L_local^X),   X in {PEN, PEG, Delta7, Ring}

with `gamma_X` read off the realised fixed point (0 while a relay is below threshold, which is the shipped case), and
the loop, once closed, grows until `f'(u)` has fallen back to `gamma_E_crit` -- `rate_at_gain`, the predicted bump
rate. A `gamma_E_crit` above 8.00 Hz/mV is above every operating point of the exact smoothed f-I at sigma 2 mV (1.4
states what that does and does not establish). Both readings are reported per configuration
(`bump_criterion.epg_recurrent` and `.with_relays`).

Tests (`tests/test_cx_ring_structure.py`, 13 passed; `tests/test_cx_wedge_hold.py`, 6 passed; CPU): `u_for_rate`
inverts `cx_wedge.lif_fi` to 1e-6 and returns 6.6276 / 11.9933; `lif_fi_prime` matches a central difference of
`lif_fi` to 2-3 % above u 12 and is bounded; `rate_at_gain(3.35) = 144`, `(3.01) = 160`, `(33.4) = NaN`;
`gamma_crit_combined` reduces to 5A's `1/(tau sqrt(lambda))` at d = 0 and to `1/(tau d)` at lambda = 0 and solves its
quadratic to 1e-9; the current drive puts the driven cells at `u = f^-1(rate)` while the 5A drive leaves them at u = 0
with slope < 0.01 Hz/mV; the forced floor holds a cell at its driven rate under -50 mV of inhibition and the literal
reading loses it; `jacobian_modes` reproduces `eig(diag(f'(u)) tau A)` to 1e-9 and zeroes a floored cell's gain.

### 1.2 Validation against the 5A batch it missed (`out/cx6/structure/validation.md`)

Eight shipped-gain cx5 arms, each configuration built from the arm's own command flags, compared with the 4 runs of
that arm in `out/cx5/`:

| arm | gamma_E crit (Hz/mV) | local EPG->EPG (mV) | loop closes (max smoothed-f-I slope 8.00) | predicted Hz | measured Hz | rel err | seeds with a bump |
|---|---|---|---|---|---|---|---|
| S | 33.34 | +6.00 | no | -- | (none) | -- | 0/4 |
| G | 33.34 | +6.00 | no | -- | (none) | -- | 0/4 |
| C | 28.77 | +6.95 | no | -- | (none) | -- | 0/4 |
| CG | 28.77 | +6.95 | no | -- | (none) | -- | 0/4 |
| **F** | 3.34 | +59.79 | yes | **144** | **152.5** | **5.4 %** | 4/4 |
| FG | 3.34 | +59.79 | yes | 144 | (none) | -- | 0/4 |
| CF | 3.02 | +66.29 | yes | 160 | 142.8 | 11.8 % | 2/4 |
| **CFG** | 3.02 | +66.29 | yes | **160** | **157.1** | **1.7 %** | 4/4 |

**7 of 8 arms called correctly -- which is 3 of 4 distinct predictions (below); the rate within 10 % in 2 of the 3 arms
that held a bump** (CF, the arm whose bump jumps to wedges 7-8, is 11.8 % low). The one miss is **FG**, where the
criterion predicts a bump and the LIF has none: the EPG-recurrence criterion cannot see the glutamatergic GLNO's -33 mV
per PEN volley, which is what opens that loop (`cx_wedge` arm FG, 5A section 3.3). The requirement this thread was set
-- predict F to ~10 % and no bump for S / G / C / CG -- is met.

The eight calls are four predictions made twice. `gamma_E_crit` depends only on the wedge-local one-step
EPG -> EPG coefficient, which takes four values over these arms (5.9987 mV for S and G, 6.9523 for C and CG,
59.7926 for F and FG, 66.2921 for CF and CFG) -- the GLNO relabel cannot enter an EPG-recurrence criterion at all.
So the classifier is 'does this arm carry `same_type_gain` 1?', the arms split 4/4 on exactly that flag, and the
one pair the measurement splits (F 4/4, FG 0/4) is the pair the criterion gets wrong. Read as **3 of 4 distinct
predictions**, with the rate half 2 of 3. The 144 / 160 Hz saturation rates are also not new here: 5A's skeptic
pass R3 already derived 145 Hz for `f` and 161 for `cf` from the same coefficients, against the same measurements;
what 6A adds is the tool that produces them, the per-cell gains and the hold configurations.

### 1.3 The re-derived ranking, and what moved

Ranking key: the fixed point's post-release bump, then the PEN rate during the pulse, then the EPG margin, then
`gamma_E_crit`. 15 configurations (`out/cx6/structure/structure.md`); the top of the table and 5A's, side by side:

| rank | 6A (fixed) | PEN during pulse (Hz) | u_PEN (mV) | 5A (`--legacy`) |
|---|---|---|---|---|
| 1 | **H3 (hold all three)** | 103.0 | +27.04 | b+f: cap lifted + damping off |
| 2 | **H3 + GLNO = glutamate** | 40.2 | -5.86 | b: cap lifted |
| 3 | **H_ExR6** | 14.7 | +1.26 | a+b |
| 4 | **H_ER6** | 10.8 | -4.72 | f: damping off |
| 5 | b+f (cap lifted + damping off) | 0.0 | -37.51 | a+f |
| 8 | f (damping off) | 0.0 | -15.61 | shipped |
| 10 | H_ER4m | 0.0 | -15.53 | a: GLNO = glutamate |
| 11 | shipped | 0.0 | -15.78 | c+f |
| 14 | c (receptor sign+gain) | 0.0 | -17.63 | c: receptor sign+gain (**1st in 5A's published ranking; 9th of 10 in the `--legacy` rerun, which is keyed on the PEN rate**) |

**What moved.** (i) The data-implied receptor tier `c` is no longer the best non-instrument configuration on the
criterion that matters: its `gamma_E_crit` falls only 13.7 % (33.340 -> 28.768 Hz/mV), stays 3.6x above the largest
slope the smoothed LIF has anywhere, and its PEN margin *worsens* (-15.78 -> -17.63 mV), where 5A's ranking put it first
for having the lowest two-step gamma_crit. The table's *positions* below the two hold arms should not be read: ranks
5-15 all have no bump and `in_minus_out` = -1.8e-15, and are then ordered by PEN rates between 1e-4 and 3e-179 Hz,
i.e. by arithmetic underflow -- the same float-noise tie-breaking section 6 notes for the legacy run. (5A's published
ranking was keyed on gamma_crit and the PEN margin; the `--legacy` rerun of this tool, keyed on the PEN rate, puts `c`
9th of 10, not 1st.) (ii) The global instrument `f` (same-type damping
off), which 5A ranked last-equal and which is the only shipped-gain cx5 arm that held a bump, is now the highest-ranked
*non-hold* configuration on the bump criterion (`gamma_E_crit` 3.34, predicted 144 Hz, supercritical) even though its
PEN margin is unchanged -- the fix separates "the relay loop closes" from "the EPG's own recurrence closes", and F is
the second. (iii) The four hold configurations, which 5A could not express at all, take the top four places on the PEN
rate. (iv) `b` (cap lifted) keeps its 5A place in the DC ordering and stays the worst margin (-41.00 mV). The ranking
is still a rate-model ordering, and section 3.2 shows where it is wrong.

### 1.4 Two 5A numbers the fix corrects

* **The bounded comparison, and exactly what it is worth.** 5A's skeptic pass R1 tabulated `f'(u)` from the
  deterministic f-I (25.8 at u 7.1, 9.13 at 8, 6.01 at 12, 2.90 at 40). That function *diverges* as u -> theta+, so
  "25.8 at 7.1" is a sample on a divergence, and a node-wise derivative of the smoothed f-I inherits it. The Gaussian
  smoothing the model actually uses caps the slope, and the corrected statement is this:

  The maximum slope of the smoothed f-I is **8.00 Hz/mV at u 8.61 (25.5 Hz)** at the sigma = 2 mV this rate model
  assumes (adaptive quadrature; `lif_fi_prime`'s Gauss-Hermite estimate is not converged -- 8.66 / 8.31 / 8.27 / 8.19
  / 8.13 at 15 / 31 / 61 / 101 / 201 nodes, still falling, and the tool's default 61-node value 8.27 is 3.4 % high;
  the implementation returns NaN above ~201 nodes). Two limits on what that number means. It is a property of the
  **assumed** sigma, not a measurement: the maximum slope is 25.3 Hz/mV at sigma 0.25 mV and 11.0 at 1 mV, and the
  shipped ring's 33.3 Hz/mV would be reached at sigma 0.17 mV; `SIGMA_MV` is hard-coded and this project has never
  measured the spiking LIF's effective input noise. And the f-I the fixed point actually iterates,
  `cx_wedge.lif_fi` at 15 nodes, is not smooth near threshold: its own numerical slope reaches 7,528 Hz/mV at
  u 7.0001 and exceeds 33.3 Hz/mV at 346 of 600,001 grid points on u in [0, 60]. So the right statement is that
  **in this rate model, at sigma 2 mV, a gamma_crit of 28.8-33.3 Hz/mV is above every operating point of the exact
  smoothed f-I by a factor of 3.6-4.2** -- not that it is "unreachable" as a property of the LIF. 5A's skeptic's
  25.8 Hz/mV at u 7.1 is exactly right for the *deterministic* f-I and is indeed a sample on its divergence
  (622 Hz/mV at u 7.001), so it was never a maximum; the bounded comparison is the correct one to make, and it is
  the measurement -- no bump in S / G / C / CG in 4 of 4 seeds -- that carries the conclusion.

  Sourcing: `structure.json` and `validation.json` carry the tool's own single number (`max_lif_slope` 8.274678 at
  `_at_u` 8.60177, `_at_hz` 25.1297). The 8.00 and the quadrature-order sequence above are the independent skeptic
  pass's own recomputation (appended at the end of this file); no file under `out/cx6/` records either, and the
  earlier draft's figure for the discarded node-wise implementation, which was in no file either, is withdrawn.

* **The per-cell gains in 5A's skeptic table are the node-wise ones.** With the corrected derivative the same 5A fixed
  point gives Delta7 4.68 (background) / 5.22 (pulse) against the 3.56 / 6.21 printed there; every compass cell except
  Delta7 is still zero to within 1e-7 (EPG 9e-41, PEN 1e-08, PEG 6e-18, EPGt 1e-09), and the two non-compass groups in
  the sub-circuit are not: Ring 0.242 (background) / 0.359 (pulse) and GLNO 0.049 / 0.320. The leading Jacobian
  eigenvalue is +0.139 / +0.088 against +0.045 / +0.079. The statement "the true Jacobian is stable in every mode at
  the shipped gains" is unchanged.

## 2. The predeclaration and the batch

`out/cx6/predeclared.json`, stamped **2026-09-15T08:09:25Z** (`written_before_submission` true), **never amended** (no
archive file exists because none was needed). It carries the arms, the family, the decision rule in words, and the
structure pass's per-arm predictions, all written before the first arm was run on the cluster; `batch.sh` sha256
`01d3c38fe698935302043b5d14693f3152ee6b89daaeeb3f7ffac58a1bb10f3e` is recorded in it and is the file that was
submitted. Order of stamps: validation 07:50:51Z < structure pass 07:57:20Z (finished 07:59:41Z) < `batch.sh`
08:07:12Z < predeclaration **08:09:25Z** < `predeclare_stamp.txt` 08:09:33Z < tree_state 08:11:05Z < submit `date -u`
**08:11:21Z** < scheduler `submitted_at` 08:11:22Z < last job finished 08:14:42Z < analysis 08:18:32Z. The CPU smoke
of the H3 arm (`out/cx6/smoke/smoke_H3_cpu.json`) was run **after** the predeclaration was stamped and before
submission, as a code check on the real connectome; it is one CPU seed, no number of it is quoted as a result, and it
showed the regime the batch then measured (PEN 45.6 Hz and EPG 222 Hz during the pulse, survival 0.00, and the
pre-pulse ring already at 97.7 Hz outside the block).

**The tree state, qualified.** `scripts/cx_ring_structure.py` was edited again at 08:18:29Z, after the
08:11:05Z `tree_state.json` stamp and after the batch finished, immediately before the analysis; `validation.md`
(07:50:51Z) was written by the earlier version. Nothing on the simulated path changed -- `scripts/cx_wedge.py` is
byte-identical to its stamped hash -- so no run is affected.

**The arms** (all `cx_wedge.py --no-structure --sim <gains> --ledger`, the cx5 protocol unchanged: full connectome, no
world, compass adaptation 0, 10 Hz Poisson background on all 46 EPG, 1 s settle, wedges 0-3 (11 EPG) at +40 Hz for 2 s,
5 s free, EPG scored per 10 ms frame by `probe_compass_room.bump_frames`), 5 seeds each:

| arm | gains | GLNO | hold | classification |
|---|---|---|---|---|
| S | 1 / 1 | silent | -- | the shipped path (reference) |
| H3 | 1 / 1 | silent | `^(ExR6\|ER6\|ER4m)$:^(PEN_\|EPG$)` | LABELLED COUNTERFACTUAL: the DC term 5A's fixed point names |
| H_ExR6 | 1 / 1 | silent | `^ExR6$:^(PEN_\|EPG$)` | the same, ExR6 alone (2 cells) |
| H_ER6 | 1 / 1 | silent | `^ER6$:^(PEN_\|EPG$)` | ER6 alone (4 cells) |
| H_ER4m | 1 / 1 | silent | `^ER4m$:^(PEN_\|EPG$)` | ER4m alone (11 cells) |
| H3G | 1 / 1 | **glutamate** | as H3 | the correct-sign ring under the hold |
| F | 1 / 1 | silent | -- | LABELLED INSTRUMENT (`same_type_gain` 1), carried from cx5 |
| R | 2 / 15 | silent | -- | LABELLED reference (`--no-delta7-pen`), carried from cx5 |

**The hold mechanism.** `cx_wedge.py --hold-edges PRE:POST` (new, default `None`) appends `(pre, post, 0.0)` to
`LIFParams.type_path_gain` -- the same stage of `brain._shaped_weights` that carries gE / gD, i.e. the `edges` kind of
`INTERP.md` 2 / 10.1 step 5 expressed through machinery that already exists (`flyverse/interp/lesion.py` still cannot
address one presynaptic class onto one postsynaptic class; that is `INTERP.md` 11.2 defect 12, untouched here). Every
run records the hold and what it resolved to: **17 pre cells (2 ExR6 + 4 ER6 + 11 ER4m) onto 88 post cells (46 EPG +
42 PEN), 1,149 entries, 37,256 synapses**, and the single-type arms resolve to 174 + 287 + 688 = 1,149 entries exactly.
EPGt is not held (`^(PEN_|EPG$)` excludes it, tested).

One arithmetic consequence of installing the hold at the `type_path_gain` stage: `input_norm` takes its total on
the shaped matrix, so removing these edges also lifts the fan-in down-scaling of any postsynaptic cell that was
above the 5,000 reference. Checked: one of 46 EPG is (total 5,099, scale 0.9806) and no PEN is, and under H3 every
EPG is below (max 4,362, scale 1.000) -- so the hold up-scales one EPG's surviving inputs by 2 %, and the effect
is not part of the result.

**Bit-identity with the flag off**: the default `--sim` path re-run on this desktop's CPU after the edit
(`out/cx6/smoke/smoke_default_path.json`) is equal to the pre-6A CPU run
(`out/cx5/smoke/smoke_default_path.json`) on **117 of 117** recorded fields, the two new record-keeping keys aside;
and the batch's own S, F and R arms reproduce cx5's seeds 0-3 with **max |cx5 - cx6| = 0.0** over 144 (metric, run)
pairs (`out/cx6/analysis/cross_batch_cx5.csv`).

**The job line was tested on CPU before submission** (`out/cx6/smoke/jobline_check.txt`, `exitcheck_*.sh`,
`holdcheck.sh`): the ten job lines were extracted from `batch.sh` **by bash** (`out/cx6/smoke/argv_shim.py` ->
`argv.json`, because Python's `shlex` keeps the `\$` that bash removes), and then (i) the exit expression returns 0
when every arm succeeds and 7 when arm 0, 2 or 6 exits 7 -- cx5's first submission died here -- and (ii) the four hold
regexes reach python's `argv` intact through both quoting levels.

**Submission.** ONE `cluster_run.py --name cx6 --minutes 30 --arm-block fam <10 job lines> --fetch out/cx6/` call,
10 jobs = 5 seeds x {GLNO silent (7 arms, sequential), GLNO = glutamate (1 arm)}, blocks `fam_s<seed>`.
**The failure line: `10 job(s), 0 failed  (5.0 min)  run dir /mnt/beegfs/neurome/runs/cx6-995cd5`, `client exit 0`**;
scheduler receipt (`out/cx6/scheduler_receipt.json`, queried from the job manager itself): `10 job(s), 0 failed
({'completed': 10})`, all on **node1**, GPU ids 0 and 1, NVIDIA B200, submitted 08:11:22Z, finished 08:14:42Z.
Artefacts: **40 `<arm>_s<seed>.json`, 40 `.txt`, 40 `.npz`**, 5 per arm x 8 arms; 40 of 40 consoles say `device cuda`;
the analysis's per-run checks (device, arm label, gains, `nt_override` / `glno_nt`, `same_type_gain`,
`receptor_model`, **and the hold: spec, factor 0, non-zero matched entries**) raise **0 problems**; compiled-W md5
`ef23cc27bea13be7f6a96f3c04fd3737` on every GLNO-silent run and `7a10d93ba2086f2c76bcdabdca79b4ec` on the five H3G
runs (the same two hashes as cx5 and as this desktop). Wall 13.6-24.9 s per run.

## 3. Results (`n_runs 40 (out/cx6/<arm>_s<seed>.json); problems 0`)

### 3.1 Per arm (ranges over the 5 seeds; per-seed values in `analysis/scatter.csv` and `analysis/state.csv`)

| arm | survival s | bump Hz | width | confined post | PEN during | PEN post | EPG in / out post | Delta7 post | GLNO post | Ring post | rest post | bump centre at 5 s (driven = 1.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | 0.00 x5 | -- | -- | 0.000 x5 | **0.29-0.66** | 0.009-0.062 | 10.2-10.8 / 9.6-9.9 | 10.2-10.9 | 0.01-0.34 | 0.79-0.83 | 0.002 | -- (vs 0.04-0.19) |
| **H3** | **0.00 x5** | -- | -- | **0.000 x5** | **40.2-48.3** | **49.3-54.0** | 25.9-116.0 / 55.5-89.5 | 120.9-124.0 | 135.9-137.6 | 10.7-11.2 | 0.035 | **5.71-6.11**, vs 0.73, 17-19/35 out above 22 Hz |
| H_ExR6 | 0.00-0.02 | (73, 90 in 2 seeds) | (3.0, 4.0) | 0.000-0.004 | **8.9-14.9** | 0.55-1.01 | 10.7-12.4 / 9.9-10.0 | 11.0-11.9 | 1.43-3.71 | 0.85-0.93 | 0.002 | -- (confined in 8-59 % of the PULSE frames) |
| H_ER6 | 0.00 x5 | -- | -- | 0.000 x5 | **2.4-3.1** | 0.10-0.19 | 10.2-10.8 / 9.6-9.9 | 10.2-10.9 | 0.08-0.44 | 0.80-0.83 | 0.002 | -- |
| H_ER4m | 0.00 x5 | -- | -- | 0.000 x5 | **0.29-0.70** | 0.016-0.062 | 10.2-10.8 / 9.6-9.9 | 10.2-10.9 | 0.04-0.26 | 0.79-0.83 | 0.002 | -- |
| F | 5.00 x4, 1.84 | 150.6-155.8 | 3.00 x5 | 0.230-0.866 | 22.5-27.4 | 11.9-30.9 | 59.4-155.2 / 9.6-11.2 | 16.2-38.8 | 33.7-91.4 | 2.7-5.8 | 0.028-0.042 | 1.04-1.25 (4 seeds), 7.89 (seed 4) |
| R | 5.00 x5 | 200.6-203.2 | 3.00-3.06 | 0.812-0.868 | 45.5-50.4 | 47.7-48.4 | 199.5-202.5 / 9.6-9.9 | 99.8-100.7 | 132.6-134.4 | 9.5-9.6 | 0.033-0.034 | 1.45-1.62 |
| **H3G** | 4.76-5.00 (1 x >= 5) | 159.0-164.2 | 3.85-4.00 | 0.108-0.488 | **26.5-29.0** | 21.0-26.5 | 17.7-160.7 / 14.3-54.8 | 86.1-94.6 | 64.0-83.2 | 6.5-7.2 | 0.020-0.022 | 0.92 (s1), 5.30, 5.43, 10.01, 10.03 |

This protocol records population **means** (`fb.brain.mean_rate` per group), not the per-cell maximum that
`INTERP.md` 10 defines `silent` by (max rate < 0.5 Hz per cell), so no arm here is shown to be silent or not
silent. What the means do say is that the relays are not off: S's PEN is 0.29-0.66 Hz during the pulse and
0.009-0.062 Hz after, over 42 cells. Two populations quoted in this section are at or below the silence
threshold on the mean alone and would meet it: the `rest` group (0.0015 Hz in S) and S's GLNO during the settle
(exactly 0.000 Hz in 5 of 5 seeds).

**The pre-pulse state matters and is where H3 differs most** (`analysis/state.csv`, the 1 s settle on 10 Hz background
alone): S 8.7-10.0 Hz EPG, PEN 0.00-0.02, Delta7 8.4-11.8, GLNO 0.00; **H3 19.7-62.9 Hz EPG, PEN 8.8-42.7, Delta7
28.3-103.5, GLNO 32.5-131.4**; H3G 10.3-40.5 / 1.0-16.5 / 12.4-68.8 / 5.1-58.8; H_ExR6 9.0-10.0 / 0.28-0.55; F
8.7-10.1 / 0.00-0.16; R 16.7-42.8 / 10.5-34.0 (`analysis/state.csv`, `epg_mean_pre`: 8.7337-10.0317, 8.9619-10.0433,
8.7337-10.0532). The hold does not "release the compass"; it removes the brake on the whole ring, background included.

### 3.2 Predictions vs measurements -- the decisive table

Predictions are `out/cx6/predeclared.json` `predictions_from_the_structure_pass`, written 08:09:25Z; measurements are
`analysis/scatter.csv` / `state.csv`.

| arm | predicted PEN during pulse | measured (5 seeds) | predicted bump after release | measured | verdict on the prediction |
|---|---|---|---|---|---|
| S | 0.0 Hz (u -15.78 mV) | 0.29-0.66 Hz | no | none in 5/5 | **right** (rate-model 0 vs a measured sub-Hz mean) |
| **H3** | **103.0 Hz** (u +27.04 mV) | **40.2-48.3 Hz** | **yes**: EPG in 252 / out 25.8 Hz, PEN 103 | **PEN yes, bump no**: EPG in 25.9-116.0 / out 55.5-89.5, confined 0.000 in 5/5 | **half right**: the sign and the threshold crossing, not the magnitude (2.1-2.6x high) and not the confinement |
| H_ExR6 | 14.7 Hz (u +1.26) | 8.9-14.9 Hz | no | no (0/5) | **right**, and the rate is inside the measured range |
| H_ER6 | 10.8 Hz (u -4.72) | 2.4-3.1 Hz | no | no | right on the call, 3.5-4.6x high on the rate |
| H_ER4m | 0.0 Hz (u -15.53) | 0.29-0.70 Hz | no | no | **right**: indistinguishable from S (p 0.69) |
| H3G | 40.2 Hz (u -5.86) | 26.5-29.0 Hz | yes: EPG in 158.7 Hz | bump 159.0-164.2 Hz, survival 4.76-5.00, confined 0.11-0.49 | **right**, including the rate to 1-4 % |
| F | (no fixed-point bump; recurrence criterion 144 Hz) | 22.5-27.4 Hz | yes at 144 Hz | 150.6-155.8 Hz in 4/5 | **right** to 4.7 % on the rate |
| R | not predicted (outside the shipped-gain rate model); cx5's measurement was the expectation: 201-203 Hz in 4/4 | 45.5-50.4 Hz | 200-203 Hz | 200.6-203.2 Hz in 5/5 | **right** |

The two systematic errors are informative. The tool is **2.1-2.6x high on H3's PEN rate and 3.5-4.6x high on H_ER6's**
(but 1.0-1.7x on H_ExR6's and 1.4-1.5x on H3G's), in the direction 5A's skeptic already flagged: the rate model's ring neurons run at 110-185 Hz where the LIF's whole
308-cell ER/ExR population averages 0.79-0.83 Hz at rest, so the DC term it removes is overstated. And it predicted a
**confined** bump under H3 where the measurement gives a saturated ring: the fixed point has no way to represent
"every off-block cell is also above threshold", because `bump_after` only asks whether the driven wedges exceed the
rest.

### 3.3 Verdicts (`analysis/compare.csv`; `common.compare`, runs = the unit, 5 v 5, p_floor 0.0079)

The predeclared family is the six primaries, Holm within each arm; members that return no p (`bump_hz_post` and
`width_half_post` have no confined frame in S, so they are "no data" or one-sided) are dropped from the denominator
and read as magnitudes, exactly as predeclared. The surviving family is m = 4, so Holm at the floor is
4 x 0.0079 = **0.0317 <= 0.05**: satisfiable, as required.

| arm vs S | `PEN_mean_during` | `PEN_mean_post` | `survival_s` | `frac_confined_post` |
|---|---|---|---|---|
| **H3** | **+43.53 Hz, z 294.0, p 0.0079, Holm 0.0317, `result`** | **+52.38, z 2504, Holm 0.0317, `result`** | +0.00, zero-SD both arms, `null` | +0.000, zero-SD both arms, `null` |
| H_ExR6 | **+10.37, z 70.0, Holm 0.0317, `result`** | **+0.69, z 32.9, `result`** | +0.006, Holm 0.84, `null` | +0.001, `null` |
| H_ER6 | **+2.25, z 15.2, Holm 0.0317, `result`** | **+0.11, z 5.5, `result`** | 0.00, `null` | 0.000, `null` |
| H_ER4m | +0.03, z 0.22, p 0.69, **`null`** | +0.004, z 0.18, p 0.84, **`null`** | 0.00, `null` | 0.000, `null` |
| F | +24.69, z 166.8, `result` | +26.41, z 1262, `result` | +4.37, `undetermined` (zero-SD null) | +0.720, `undetermined` |
| R | +48.72, z 329.0, `result` | +47.91, z 2290, `result` | +5.00, `undetermined` | +0.843, `undetermined` |
| H3G | +26.94, z 182.0, `result` | +23.74, z 1135, `result` | +4.88, `undetermined` | +0.324, `undetermined` |

Every `undetermined` above is a **structural zero-SD null**: S's `survival_s` and `frac_confined_post` are 0.00 in all
five runs, so z is undefined and the reading is the magnitude with p = 0.0079 (`INTERP.md` 10.2). Read as magnitudes:
F +4.37 s, R +5.00 s, H3G +4.88 s of bump against a reference that has none, and H3 **+0.00 s**.

### 3.4 Per-seed scatter (pasted from `out/cx6/analysis/scatter.csv`, columns `arm,key,seeds,values`)

```
S,PEN_mean_during,"0,1,2,3,4","0.3599,0.6614,0.3370,0.2865,0.4528"
H3,PEN_mean_during,"0,1,2,3,4","40.1718,43.5047,41.1931,48.3111,46.5668"
H3,PEN_mean_post,"0,1,2,3,4","49.3423,53.9561,53.3244,51.7953,53.6536"
H3,survival_s,"0,1,2,3,4","0.0000,0.0000,0.0000,0.0000,0.0000"
H3,frac_confined_post,"0,1,2,3,4","0.0000,0.0000,0.0000,0.0000,0.0000"
H_ExR6,PEN_mean_during,"0,1,2,3,4","10.4628,14.8751,8.8878,9.5299,10.1689"
H_ER6,PEN_mean_during,"0,1,2,3,4","2.7357,3.1168,2.3525,2.4249,2.7001"
H_ER4m,PEN_mean_during,"0,1,2,3,4","0.4451,0.6958,0.3478,0.2919,0.4814"
F,survival_s,"0,1,2,3,4","5.0000,5.0000,5.0000,5.0000,1.8400"
F,bump_hz_post,"0,1,2,3,4","155.7587,150.9075,152.2624,151.0834,150.5561"
R,bump_hz_post,"0,1,2,3,4","203.2319,201.0141,201.1460,200.8380,200.5677"
H3G,survival_s,"0,1,2,3,4","5.0000,4.7600,4.9100,4.9800,4.7700"
H3G,bump_hz_post,"0,1,2,3,4","160.7983,163.5748,161.2387,164.1680,158.9786"
H3G,width_half_post,"0,1,2,3,4","4.0000,3.9600,3.9945,3.8519,4.0000"
H3G,frac_confined_post,"0,1,2,3,4","0.4880,0.2000,0.3640,0.1080,0.4600"
```

The whole file carries all 8 arms x 6 primaries; the four-panel figure is `out/cx6/analysis/scatter.png`.
**F seed 4 is new**: cx5 had F at 5.00 s in 4/4, and the fifth seed dies at 1.84 s (its bump is at centre 7.89 at 5 s
with 2 of 11 driven cells above threshold), so the F family's own instrument is 4/5 rather than 4/4.

### 3.5 Reading H3 and H3G

* **H3 is a saturated ring, not a compass.** During the pulse the driven EPGs reach 81-221 Hz and PEN 40-48 Hz; after
  release the profile does not decay back and does not stay where it was put. At 5 s seed 0 reads
  `11 9 13 14 183 210 231 218 152 46 6 14 12 22 5 15` Hz across the 16 wedges (`analysis/state.csv`, `profile_end`) --
  a five-wedge hump at wedges 4-8, three wedges away from the driven block, with 19 of 35 off-block cells above 22 Hz.
  All five seeds land at centre 5.71-6.11 with vector strength 0.73. The bump rule scores it 0.00 because of the
  off-block count, and the ledger rows read FAIL / NOT_APPLICABLE / NOT_APPLICABLE.
* **The hold also removes the resting state** (section 3.1): the background alone takes the ring to 20-63 Hz EPG,
  9-43 Hz PEN and 32-131 Hz GLNO before any stimulus. Whatever else the 2 ExR6 + 4 ER6 + 11 ER4m do in this model,
  they are the term that keeps the unstimulated compass near 10 Hz.
* **H3G (the hold plus the GLNO relabel) is the closest thing to a bump at the shipped gains** that this project has
  produced without a global instrument: 159-164 Hz, width 3.85-4.00, confined in 11-49 % of post-pulse frames,
  surviving 4.76-5.00 s. The glutamatergic GLNO is doing exactly what it did in cx5's CFG arm -- braking PEN (its
  -26.8 mV per PEN volley in the fixed point) so the ring does not saturate. But **the bump is at the driven tile in
  only 1 of 5 seeds** (centre 0.92 in seed 1, with 11/11 driven cells above 22 Hz and 5/35 outside; the others at
  5.30, 5.43, 10.01, 10.03), which is cx5's R175 caveat again: `bump_survival_s` scores the last confined frame
  **wherever the bump is**, not at the driven tile.
* **This is the first shipped-gain configuration in which the GLNO sign is testable.** In cx5 the relabel was a
  structural null at these gains because GLNO fired 0.00-3.0 Hz. Here GLNO fires **135.9-137.6 Hz in H3 and
  64.0-83.2 Hz in H3G**, and the sign is worth 4.88 s of bump and 0.324 of confinement (H3G vs H3, read as
  magnitudes; both against S's zero). It does not make the relabel adoptable -- H3G is a counterfactual arm -- but it
  is the first evidence at the shipped gains that the sign does anything at all.

## 4. The decision, in the predeclared words

`out/cx6/analysis/call.csv`, produced by the analysis script from the predeclared rule:

| arm | working-compass seeds | seeds with a surviving bump | PEN during (min-max) | call |
|---|---|---|---|---|
| S | 0/5 | 0/5 | 0.29-0.66 | not the story (PEN stays below 1 Hz) |
| **H3** | **0/5** | **0/5** | **40.17-48.31** | **necessary but not sufficient (PEN fires, no working bump)** |
| H_ExR6 | 0/5 | 0/5 | 8.89-14.88 | necessary but not sufficient |
| H_ER6 | 0/5 | 0/5 | 2.35-3.12 | necessary but not sufficient |
| H_ER4m | 0/5 | 0/5 | 0.29-0.70 | not the story (PEN stays below 1 Hz) |
| F | 0/5 | 4/5 | 22.55-27.36 | necessary but not sufficient |
| R | 0/5 | 5/5 | 45.55-50.44 | necessary but not sufficient |
| H3G | 0/5 | 1/5 | 26.49-29.00 | necessary but not sufficient |

**'The DC balance is the whole story' is refused** (H3 meets the working-compass rule in 0 of 5 seeds, not >= 3).
**'Not the story' is refused** (PEN fires at 40.2-48.3 Hz, not below 1 Hz). The call is the middle one:
**necessary but not sufficient**. Two things are now attributed rather than decomposed: the DC inhibition through
ExR6 / ER6 / ER4m is what holds the compass relays below threshold (H3, H_ExR6, H_ER6 `result` on PEN; H_ER4m `null`,
so it is ExR6 and ER6 that do it on PEN), and it is also what holds the unstimulated ring at rest. What is still
missing for a bump is the thing 5A's F family showed and this batch confirms: **a local, wedge-tied recurrence** --
which at the shipped gains exists only as the x0.1-damped EPG -> EPG synapses (wedge-local +6.00 mV, gamma_crit
33.3 Hz/mV against a maximum smoothed-f-I slope of 8.00 at the assumed sigma = 2 mV).

The predeclared rule defines 'the whole story' / 'necessary but not sufficient' / 'not the story' for **H3**.
`call.csv` prints the same three phrases per arm; for the labelled instrument F and the labelled reference R they
are outside the rule and should be read as the PEN and bump counts only.

## 5. What an adoption would require (nothing is adopted)

**A hold is not a mechanism and can never be a default.** `--hold-edges` defaults to `None`, the shipped path is
bit-identical with it absent (117/117 fields on CPU; 144/144 metric-run pairs against cx5 on the GPU), and no default
changed. The only thing this thread could hand to a default is a **data** claim about the three types, and that is the
open half:

* **ExR6** -- 2 cells. MaleCNS `nt` **glutamate**, `sign` -1; 82 pairs onto PEN (1-177 synapses, mean 64, **40 above
  the connection cap 60**) and 92 onto EPG (30-113, mean 68, 53 above the cap), i.e. -8.66 and -15.01 mV per pair
  capped. It has **no receptor-table row of its own**, and the inhibitory sign of ExR6 -> EPG comes from the *E-PG*
  row's GluClalpha (Davis 2020 PB_2, tier `alias`) -- which reproduces the presynaptic NT_SIGN there rather than
  overriding it: `structure.json`'s `receptor_stage` records `sign_changed` 0 on EPG, PEN, PEG and Delta7, so
  ExR6 -> EPG would be -1 with the receptor table off. What is open is the transmitter call itself, not which stage
  of the pipeline signs it. Published transmitter or function for ExR6 specifically:
  **unknown** -- 5A could not verify one; Hulse et al. 2021 (eLife 66039) defines ExR1-ExR8 morphologically and by
  connectivity and ExR2 is the dopaminergic PPM3 class. Whether ExR6 acts ionotropically on E-PG at all is
  **unknown**. Its modulatory (slow-class) status here is nil by construction: the monoamine slow class is off by
  default and, on these cells, is < 0.2 mV at their rest rates (5A section 1.3 d).
* **ER6** -- 4 cells. MaleCNS `nt` **gaba**; 138 pairs onto PEN and 149 onto EPG, **none above the cap** (1-45
  synapses, mean 13-17), -4.62 / -3.55 mV per pair. **No receptor-table row of its own** (the sign rides on the E-PG
  and PEN rows). The general statement that ring neurons are GABAergic inhibitors of E-PG is in the literature 5A
  cites (Omoto et al. 2017; Fisher et al. 2019; Kim et al. 2019), but those papers are about the R2 / R4 classes;
  **for ER6 specifically, unknown**.
* **ER4m** -- 11 cells. MaleCNS `nt` **gaba**; 506 pairs onto EPG (9-75 synapses, mean 42, 36 above the cap), -11.39
  mV per pair, -125.3 mV per volley -- the largest single ring term onto EPG -- and almost nothing onto PEN (-2.0 mV
  per volley), which is exactly why holding it alone changes PEN by 0.03 Hz (`null`). **ER4m does have a
  receptor-table row** (unlike the other two), so its postsynaptic side is tiered from an expression profile; its
  presynaptic transmitter is the MaleCNS call. Its modulatory status: **unknown**.

So the data question the next thread inherits is narrow and answerable: **is ExR6 glutamatergic and does it open a
chloride conductance on E-PG and PEN?** -- because ExR6 alone carries most of the PEN DC (8.9-14.9 Hz when held,
against 2.4-3.1 for ER6 and 0.3-0.7 for ER4m). A relabel of ExR6 would need the same standard as any other entry in
`TYPE_NT_OVERRIDE`: sources that are not one EM classifier, a cache rebuild and the full benchmark suite at >= 3 draws
with no check changing status. **Nothing here licenses that**, because nothing here shows the animal's ExR6 is
anything other than what the connectome says.

Two smaller things this batch does not license: the F-family bump is now 4/5 rather than 4/4 (its fifth seed dies at
1.84 s), which if anything weakens the case for `same_type_gain` 1 that 5A already refused; and H3G's near-bump is a
counterfactual arm, so the GLNO relabel's status is unchanged -- `docs/audits/glno_relabel.md` owns that decision, and
what 6A adds to it is that **the sign is no longer untestable at the shipped gains**: in a configuration where GLNO
fires 64-137 Hz it is worth the difference between a saturated ring (H3) and a 159-164 Hz bump (H3G).

## 6. An incident, recorded

**Between 08:03 and 08:05Z per this thread's log, not recorded in any file**, the fixed tool ran once **with no
arguments** -- the file it overwrote recorded its own provenance as `generator: python .\scripts\cx_ring_structure.py`
with a `generated_utc` in that window, carrying the 6A `modes` block, and that file has since been overwritten in turn
by the regeneration, so no surviving file and no file mtime carries the overwrite time (the audit gave it as 08:03:28Z
here and as 08:05Z in the Report and in `REGENERATED_BY_6A.txt`; neither is supported, and both are now stated as the
window). Its default `--out` is `out/cx5/structure`, so that run **overwrote 5A's structure artefacts**
(`structure.json`, `structure.md`, `matrices.npz`, `evidence_glno.json`) with the fixed tool's output. The fixed tool
existed only in this working tree, so this thread owns the incident; I could not identify which invocation issued it
(no command in this thread's log runs the script without `--out`), and the default `--out` is the trap -- it should be
required, which is an open item for whoever next touches the script. `out/` is git-ignored, so the originals could not
be restored from the repository. They were regenerated at **2026-09-15T08:20:39Z** -- the regenerated
`structure.json`'s own `generated_utc`, with the four files' mtimes at 08:22:11Z and `REGENERATED_BY_6A.txt` written
alongside them at 08:22:41Z; the "08:31Z" printed in an earlier draft of this section and of the .txt is in no file and
is withdrawn -- with `python scripts/cx_ring_structure.py --out out/cx5/structure --legacy --no-banc`, which runs the
5A algorithm and
reproduces every number `docs/audits/compass_ring_mechanism.md` quotes from that file to the printed digit (shipped
u_PEN -15.785 mV during the pulse and -11.057 at background; lambda_1 +10,019 with PEN +8,327 / Delta7 +1,719 / PEG
+178 / Ring -206; ExR6 -14.78 mV @ 183.2 Hz, ER6 -9.77 @ 116.3, ER4m -0.25 @ 22.3; b 1.860 / -41.00, b+f 1.866 /
-37.51, c 1.634 / -17.63, f 2.002 / -15.61). The regenerated files additionally carry the 6A keys and break
`ranking` ties differently, and `out/cx5/structure/REGENERATED_BY_6A.txt` says so alongside them. The authoritative 5A
numbers are the ones printed in 5A's own audit, and none of them changed.

## Report

```yaml
summary: >
  Thread 6A fixed the 5A structure tool and then ran the one counterfactual arm 5A's skeptic named. An independent
  skeptic pass (Opus, 2026-09-15, verdict MOSTLY SOUND, corrections 1-13, all applied and appended below) recomputed
  every measurement in the batch from the raw .npz with max |diff| 0.0 and corrected the argument around them; what
  follows is the corrected reading. THE FIX: the forced drive now enters the rate model as a CURRENT (u = f^-1(rate),
  derived: 10 Hz -> 6.6276 mV, 50 Hz -> 11.9933 mV) with the forced rate as a FLOOR (FlyBrain.stimulate forces spikes;
  without the floor the same fixed point collapses the ring to 2.7 Hz against a measured 9-10 Hz); the one-step
  EPG->EPG term is KEPT, and the bump criterion is the wedge-local one-sided return
  gamma_E_crit = 1 / (tau d_local + tau^2 sum_X gamma_X L_local^X) with per-cell gammas from the realised fixed point;
  the linearisation is the true Jacobian diag(f'(u_i)) tau A with f' computed by INTEGRATION BY PARTS. --legacy
  reproduces 5A exactly. THE SLOPE BOUND, AS CORRECTED: the maximum slope of the smoothed f-I is 8.00 Hz/mV at u 8.61
  (25.5 Hz) at the sigma = 2 mV this rate model ASSUMES. That sigma is an assumption and has never been measured
  (SIGMA_MV is hard-coded; the maximum slope is 25.3 Hz/mV at sigma 0.25 mV and 11.0 at 1 mV, and the shipped ring's
  33.3 Hz/mV would be reached at sigma 0.17 mV), the tool's own Gauss-Hermite estimate is NOT converged (8.66 / 8.31 /
  8.27 / 8.19 / 8.13 at 15 / 31 / 61 / 101 / 201 nodes, the default 61-node 8.27 being 3.4 % high), and the 15-node
  cx_wedge.lif_fi that the fixed point actually iterates is not smooth near threshold (its own numerical slope reaches
  7,528 Hz/mV at u 7.0001 and exceeds 33.3 Hz/mV at 346 of 600,001 grid points). So the SIGN of the comparison is
  earned and the word 'unreachable' is NOT: at sigma 2 mV a gamma_crit of 28.8-33.3 Hz/mV is above every operating
  point of the exact smoothed f-I by 3.6-4.2x, and it is the measurement -- no bump in S / G / C / CG in 4 of 4 seeds
  -- that carries the conclusion. 5A's '25.8 Hz/mV at u 7.1' is exactly right for the DETERMINISTIC f-I and is a sample
  on its divergence, so it was never a maximum. VALIDATION on the cx5 batch the tool had missed: 7 of 8 arms called
  correctly, which is 3 OF 4 DISTINCT PREDICTIONS -- gamma_E_crit depends only on the wedge-local one-step EPG->EPG
  coefficient, which takes four values over the eight arms, so the classifier is 'does this arm carry same_type_gain
  1?', the arms split 4/4 on that flag, and the one pair the measurement splits (F 4/4, FG 0/4) is the pair the
  criterion gets wrong -- with the rate half 2 of 3: F predicted 144 Hz against a measured 152.5 (5.4 %), CFG 160
  against 157.1 (1.7 %), CF 160 against 142.8 (11.8 %). In the re-derived ranking the receptor tier c is no longer the
  best non-instrument configuration (its gamma_E_crit falls only 13.7 %, 33.340 -> 28.768, stays 3.6x above the largest
  slope the smoothed LIF has anywhere, and its PEN margin WORSENS, -15.78 -> -17.63 mV), and the table's positions
  below the hold arms are not to be read: ranks 5-15 all have no bump and are then ordered by PEN rates between 1e-4
  and 3e-179 Hz, i.e. by arithmetic underflow. THE TEST: batch cx6-995cd5 (house node1, B200, 10 jobs, 0 failed, 40/40
  runs, 0 provenance problems), 8 arms x 5 seeds at the SHIPPED gains, holding ExR6 + ER6 + ER4m at 0 onto PEN and EPG
  through a new cx_wedge.py --hold-edges flag (default None; shipped path bit-identical). RESULT: the hold lifts the
  driven PEN from 0.29-0.66 Hz to 40.2-48.3 Hz during the pulse (+43.53, z 294, Holm 0.0317, result) and to 49.3-54.0
  after release, and buys NO bump -- survival 0.00 and frac_confined_post 0.000 in 5/5, because the ring saturates
  (off-block EPGs 55-90 Hz, 17-19 of 35 above 22 Hz, the hump at wedge 5.7-6.1 rather than the driven 1.5) and because
  the same DC term also holds the UNSTIMULATED ring at rest (background alone takes EPG to 19.7-62.9 Hz and PEN to
  8.8-42.7 under the hold, against S's 8.7-10.0 / 0.00-0.02). Per type it is ExR6 (PEN 8.9-14.9 Hz held alone) and ER6
  (2.4-3.1) that carry the PEN DC; ER4m alone is a null on PEN (0.29-0.70, p 0.69) and is a 125 mV-per-volley term onto
  EPG instead. THE PREDECLARED CALL IS 'NECESSARY BUT NOT SUFFICIENT' -- a rule the predeclaration words for H3, whose
  three phrases call.csv then prints per arm beyond it. Nothing is adopted: a hold is a counterfactual, the flag
  defaults to off, and the data half -- ExR6's transmitter, receptor and modulatory status in the animal -- is UNKNOWN.
skeptic: {source: "independent Opus pass, 2026-09-15", verdict: "mostly sound"}
key_claims:
  - "THE DC BALANCE IS NECESSARY AND NOT SUFFICIENT. H3 (ExR6+ER6+ER4m -> PEN,EPG held at 0; 17 pre cells, 88 post cells, 1,149 entries, 37,256 synapses) raises PEN_mean_during from S's 0.29-0.66 Hz to 40.17-48.31 Hz (diff +43.53, z 294.0, p 0.0079, Holm 0.0317, result) and PEN_mean_post from 0.009-0.062 to 49.34-53.96 (+52.38, z 2504, result), while bump_survival_s and frac_confined_post stay 0.00 / 0.000 in 5 of 5 seeds -- a structural zero-vs-zero null against S."
  - "THE HOLD ALSO REMOVES THE RESTING STATE: on the 10 Hz background alone, before any pulse, H3 sits at EPG 19.7-62.9 Hz, PEN 8.8-42.7, Delta7 28.3-103.5, GLNO 32.5-131.4 (S: 8.7-10.0 / 0.00-0.02 / 8.4-11.8 / 0.00; the EPG ranges are analysis/state.csv epg_mean_pre, S 8.7337-10.0317, H_ExR6 8.9619-10.0433, F 8.7337-10.0532). Whatever else these 17 cells do, they are the term that keeps the unstimulated compass near its drive. At the instant t = 1.00 s H3 already sits at PEN 44.3-53.1 Hz, GLNO 135.6-151.1, Delta7 110.9-125.5 and EPG out-of-block 21.3-99.2 Hz."
  - "PER TYPE, ExR6 CARRIES THE PEN DC: held alone PEN reaches 8.9-14.9 Hz (+10.37, z 70.0, result) and the ring is confined in 8-59 % of the PULSE frames though 0 % after release; ER6 alone 2.4-3.1 Hz (+2.25, z 15.2, result); ER4m alone 0.29-0.70 Hz, indistinguishable from S (+0.03, z 0.22, p 0.69, null) -- ER4m is -125.3 mV per volley onto EPG and only -2.0 onto PEN. The three single-type holds partition the 1,149 entries exactly: 174 + 287 + 688."
  - "H3 IS A SATURATED RING, NOT A COMPASS: at 5 s the profile is a five-wedge hump at wedges 4-8 (seed 0: 11 9 13 14 183 210 231 218 152 46 6 14 12 22 5 15 Hz), centre 5.71-6.11 in 5/5 seeds against a driven-block centre of 1.5, vector strength 0.73, with 17-19 of 35 off-block cells above 22 Hz."
  - "H3G (the hold with GLNO = glutamate) is the closest a shipped-gain configuration has come to a bump without a global instrument: survival 4.76-5.00 s (1/5 >= 5 s), 159.0-164.2 Hz, width 3.85-4.00, confined 0.108-0.488 -- and the bump is AT THE DRIVEN TILE IN ONLY 1 OF 5 SEEDS (centre 0.92; the others 5.30, 5.43, 10.01, 10.03), cx5's R175 caveat again. It is also the first shipped-gain configuration in which the GLNO sign is testable at all: GLNO fires 135.9-137.6 Hz in H3 and 64.0-83.2 in H3G, against 0.01-0.34 Hz in S."
  - "THE TOOL FIX IS VALIDATED ON THE BATCH IT MISSED, AND IS WORTH 3 OF 4 DISTINCT PREDICTIONS, NOT 7 OF 8 ARMS: gamma_E_crit depends only on the wedge-local one-step EPG->EPG coefficient, which takes four values over the eight arms (5.9987 mV for S and G, 6.9523 for C and CG, 59.7926 for F and FG, 66.2921 for CF and CFG), so the eight calls are four predictions made twice, the classifier is 'does this arm carry same_type_gain 1?', the arms split 4/4 on exactly that flag, and the one pair the measurement splits (F 4/4, FG 0/4) is the pair the criterion gets wrong. The rate half is 2 of 3 arms: F predicted 144 Hz vs measured 152.5 (5.4 %), CFG 160 vs 157.1 (1.7 %), CF 160 vs 142.8 (11.8 %). The 144 / 160 Hz rates are not new here either -- 5A's skeptic pass R3 derived 145 and 161 Hz from the same coefficients against the same measurements; what 6A adds is the tool, the per-cell gains and the hold configurations."
  - "THE MAXIMUM SLOPE OF THE SMOOTHED f-I IS 8.00 Hz/mV (u 8.61, 25.5 Hz) AT THE ASSUMED sigma = 2 mV -- and that is what the comparison is against, not a property of the LIF. Three qualifications: sigma is hard-coded (SIGMA_MV = 2.0) and this project has never measured the spiking LIF's effective input noise (max slope 25.3 Hz/mV at sigma 0.25, 11.0 at 1.0; the shipped 33.3 would be reached at sigma 0.17); lif_fi_prime's Gauss-Hermite estimate is NOT converged (8.66 / 8.31 / 8.27 / 8.19 / 8.13 at 15 / 31 / 61 / 101 / 201 nodes, NaN above ~201, the tool's default 61-node 8.27 being 3.4 % high); and the 15-node cx_wedge.lif_fi the fixed point actually iterates is not smooth near threshold (7,528 Hz/mV at u 7.0001, above 33.3 at 346 of 600,001 grid points). So gamma_crit 28.8-33.3 Hz/mV is above every operating point of the exact smoothed f-I by 3.6-4.2x at sigma 2 mV -- the SIGN of the comparison is earned, 'unreachable' is not, and the conclusion rests on the measurement (no bump in S / G / C / CG in 4 of 4 seeds). 5A's '25.8 Hz/mV at u 7.1' is right for the DETERMINISTIC f-I and is a sample on its divergence (622 Hz/mV at u 7.001), so it was never a maximum. The 8.00 and the node sequence are the independent skeptic pass's recomputation; no file under out/cx6/ carries them, and structure.json / validation.json carry only the tool's own max_lif_slope 8.274678 at u 8.60177 / 25.1297 Hz."
  - "THE RE-DERIVED RANKING: the data-implied receptor tier c is no longer the best non-instrument configuration on the criterion that matters -- its gamma_E_crit falls only 13.7 % (33.340 -> 28.768 Hz/mV), stays 3.6x above the largest slope the smoothed LIF has anywhere, and its PEN margin WORSENS (-15.78 -> -17.63 mV), where 5A's ranking put it first for having the lowest two-step gamma_crit. The table's POSITIONS below the two hold arms are not to be read: ranks 5-15 all have no bump and in_minus_out = -1.8e-15 and are then ordered by PEN rates between 1e-4 and 3e-179 Hz, i.e. by arithmetic underflow -- the same float noise section 6 notes for the legacy run, whose PEN-rate-keyed rerun puts c 9th of 10, not 1st. What does move: the global instrument f rises from last-equal to the best non-hold configuration on the bump criterion (gamma_E_crit 3.34, predicted 144 Hz), and the four hold configurations take the top four places on the PEN rate. 5A's ranking was computed at a state with zero gain everywhere; this one is not."
  - "THE RATE MODEL IS SYSTEMATICALLY HIGH ON THE HELD ARMS, BUT NOT UNIFORMLY: predicted PEN 103.0 Hz for H3 against a measured 40.17-48.31 (2.1-2.6x) and 10.8 for H_ER6 against 2.35-3.12 (3.5-4.6x), but 14.7 for H_ExR6 against 8.89-14.88 (1.0-1.7x) and 40.2 for H3G against 26.49-29.00 (1.4-1.5x). The direction is the one 5A's skeptic flagged: the rate model's ring neurons run at 110-185 Hz where the LIF's whole 308-cell ER/ExR population averages 0.79-0.83 Hz at rest."
  - "NO ARM MEETS THE PREDECLARED WORKING-COMPASS RULE IN ANY SEED, and every surviving bump fails compass.EPG.bump_rate_hz by 2.5-3.4x: F 150.6-155.8 Hz (4/5 seeds, one dies at 1.84 s -- cx5 had 4/4), H3G 159.0-164.2, R 200.6-203.2, against the 5-60 Hz row."
  - "REPRODUCIBILITY: the S, F and R arms reproduce cx5's seeds 0-3 with max |cx5 - cx6| = 0.0 over 144 (metric, run) pairs (out/cx6/analysis/cross_batch_cx5.csv), and the CPU default path is equal to the pre-6A CPU run on 117 of 117 recorded fields. Compiled-W md5 ef23cc27 on every GLNO-silent run, 7a10d93b on the five H3G runs."
  - "PROCESS: the job lines were extracted from batch.sh BY BASH before submission and tested -- the exit expression returns 0 when every arm succeeds and 7 when arm 0, 2 or 6 fails, and the four hold regexes reach python's argv intact through both quoting levels. cx5's first submission died exactly there. One qualification on tree_state.json (08:11:05Z): scripts/cx_ring_structure.py was edited AGAIN at 08:18:29Z, after the stamp, after the last job finished (08:14:42Z) and three seconds before the analysis ran, and validation.md (07:50:51Z) was written by the earlier version. Nothing on the simulated path changed -- scripts/cx_wedge.py is byte-identical to its stamped hash -- so no run is affected."
  - "AN UNNAMED SIDE EFFECT OF THE HOLD, CHECKED AND NEGLIGIBLE: installing it at the type_path_gain stage means input_norm takes its total on the SHAPED matrix, so removing the 1,149 entries also lifts the fan-in down-scaling of any postsynaptic cell above the 5,000 reference. Exactly one of 46 EPG is (total 5,099, scale 0.9806) and no PEN is; under H3 every EPG is below (max 4,362, scale 1.000). So the hold up-scales one EPG cell's surviving inputs by 2 %, and that is not part of the result."
  - "THE CALL'S WORDS ARE DEFINED FOR H3: the predeclared rule words 'the whole story' / 'necessary but not sufficient' / 'not the story' for H3 only. analysis/call.csv prints the same three phrases per arm; for the labelled instrument F and the labelled reference R they are outside the rule and are to be read as the PEN and bump counts only. The section 4 headline is about H3."
  - "NOTHING IN THIS BATCH IS SHOWN SILENT OR NOT SILENT: the protocol records population MEANS (fb.brain.mean_rate per group), not the per-cell maximum INTERP.md 10 defines silent by (max rate < 0.5 Hz per cell). What the means say is that the relays are not off (S's PEN 0.29-0.66 Hz during the pulse, 0.009-0.062 after, over 42 cells); two quoted populations are at or below the threshold on the mean alone -- the rest group (0.0015 Hz in S) and S's GLNO during the settle (exactly 0.000 Hz in 5 of 5 seeds)."
  - "THE ExR6 SIGN IS NOT A PIPELINE ARTEFACT: the -1 on ExR6 -> EPG rides on the E-PG row's GluClalpha (Davis 2020 PB_2, tier alias), but that row REPRODUCES the presynaptic NT_SIGN rather than overriding it -- structure.json's receptor_stage records sign_changed 0 on EPG, PEN, PEG and Delta7, so ExR6 -> EPG would be -1 with the receptor table off. What is open is the transmitter call itself, not which stage of the pipeline signs it."
  - "AN INCIDENT: an argument-less run of scripts/cx_ring_structure.py overwrote 5A's out/cx5/structure/* (its default --out). The overwrite happened between 08:03 and 08:05Z per this thread's log and is recorded in no surviving file and in no file mtime (the earlier 08:03:28Z / 08:05Z are both withdrawn as sourced times). out/ is git-ignored, so the files were regenerated with --legacy at 2026-09-15T08:20:39Z -- the regenerated structure.json's own generated_utc, with the four files' mtimes at 08:22:11Z and REGENERATED_BY_6A.txt's at 08:22:41Z; the earlier '08:31Z' is in no file and is withdrawn. The regeneration reproduces every number 5A's audit quotes from these files to the printed digit, and out/cx5/structure/REGENERATED_BY_6A.txt records it."
files_written:
  - scripts/cx_ring_structure.py (the three fixes as defaults, --legacy / --drive / --reduction / --gain, --holds, --validate-batch, --predeclare, --tree-state, --batch cx6, the CX6 arm table, state.csv / scatter.csv / call.csv in the analysis)
  - scripts/cx_wedge.py (ONE new flag: --hold-edges PRE_REGEX:POST_REGEX, default None; parse_hold_edges, hold_edge_counts; every row records the hold and what it resolved to)
  - tests/test_cx_ring_structure.py (+8 tests for the fixes; 13 passed), tests/test_cx_wedge_hold.py (new; 6 passed)
  - docs/audits/compass_dc_balance.md
  - out/cx6/structure/{structure.json,structure.md,matrices.npz,evidence_glno.json,validation.md,validation.json}
  - out/cx6/{predeclared.json,batch.sh,tree_state.json,submit_stamp.txt,predeclare_stamp.txt,scheduler_receipt.json,scheduler_receipt.py}
  - out/cx6/smoke/{smoke_default_path.json,smoke_H3_cpu.json,jobline_check.txt,argv_shim.py,argv.json,batch_probe.sh,exitcheck_*.sh,holdcheck.sh}
  - out/cx6/<arm>_s<seed>.{json,txt,npz} x 40; out/cx6/analysis/{analysis.md,runs.csv,compare.csv,decision.csv,call.csv,scatter.csv,state.csv,cross_batch_cx5.csv,analysis.json,scatter.png}; out/cx6_cluster.log
  - out/cx5/structure/* REGENERATED with --legacy after an accidental overwrite, with out/cx5/structure/REGENERATED_BY_6A.txt alongside (section 6)
api:
  - "cx_wedge.simulate(..., hold_edges=None): [(pre_re, post_re, factor)] appended to LIFParams.type_path_gain; the row records `hold_edges` and `hold_edges_resolved` (cells, entries, synapses). Default None = the previous gain list, entry for entry."
  - "cx_wedge.parse_hold_edges(['PRE:POST', ...]) -> [(pre, post, 0.0)]; cx_wedge.hold_edge_counts(c, holds) -> what a hold silences"
  - "cx_ring_structure: lif_fi_det, lif_fi_prime (integration by parts), u_for_rate, max_slope, rate_at_gain, gamma_crit_combined, rate_fixed_point(drive='current'|'current-nofloor'|'rate'), jacobian_modes(..., floor=None), hold_params, hold_configs, config_for_arm, validate_batch, write_predeclaration, write_tree_state, CX6_ARMS, PRIMARIES_CX6"
validation:
  - "tests/test_cx_ring_structure.py 13 passed, tests/test_cx_wedge_hold.py 6 passed (CPU, CUDA_VISIBLE_DEVICES=-1)"
  - "the fixed tool against out/cx5: 7/8 arms = 3/4 DISTINCT bump calls correct (four values of gamma_E_crit over eight arms), F 144 vs 152.5 Hz (5.4 %), CFG 160 vs 157.1 (1.7 %), no bump predicted for S/G/C/CG (out/cx6/structure/validation.md)"
  - "shipped-path bit-identity on CPU: 117 of 117 recorded fields equal to the pre-6A run (out/cx6/smoke/smoke_default_path.json vs out/cx5/smoke/smoke_default_path.json)"
  - "cross-batch: S, F, R seeds 0-3 reproduce cx5 with max |diff| = 0.0 over 144 (metric, run) pairs (out/cx6/analysis/cross_batch_cx5.csv)"
  - "scheduler receipt 10 job(s), 0 failed ({'completed': 10}), all node1 / B200 / GPU ids 0-1; client '10 job(s), 0 failed (5.0 min) run dir /mnt/beegfs/neurome/runs/cx6-995cd5', client exit 0; 40/40 json+txt+npz; 40/40 consoles 'device cuda'; analyse problems 0; the hold check (spec, factor 0, entries) passes on all 40"
  - "job line tested on CPU before submission: exit expression 0 / 7 / 7 / 7, hold regexes intact through both quoting levels (out/cx6/smoke/jobline_check.txt)"
  - "predeclaration stamped 2026-09-15T08:09:25Z, never amended; batch.sh sha256 01d3c38f...0f3e recorded in it and unchanged at submission"
recommendations:
  - "DO NOT ADOPT ANYTHING FROM THIS THREAD. The arm is a hold and holds are counterfactuals; --hold-edges defaults to None and the shipped path is bit-identical with it absent. compass.EPG.bump_survival_s stays FAIL and the rate / width rows stay NOT_APPLICABLE."
  - "WHAT IS NOW ATTRIBUTED, NOT DECOMPOSED: the DC inhibition through ExR6 (2 cells) and ER6 (4 cells) is what holds the compass relays below threshold -- PEN 0.29-0.66 Hz shipped, 40.2-48.3 with all three held, 8.9-14.9 with ExR6 alone, 2.4-3.1 with ER6 alone, unchanged with ER4m alone. It is ALSO what holds the unstimulated ring at rest. What is still missing for a bump is a local, wedge-tied recurrence, which at the shipped gains exists only as the x0.1-damped EPG->EPG synapses (wedge-local +6.00 mV, gamma_crit 33.3 Hz/mV against a maximum smoothed-f-I slope of 8.00 at the assumed sigma = 2 mV -- a comparison whose sign is earned in the rate model, not a statement that the gain is unreachable in the spiking LIF)."
  - "THE DATA QUESTION IS NOW NARROW: is ExR6 glutamatergic, and does it open a chloride conductance on E-PG and PEN? ExR6 alone carries most of the PEN DC. It is 2 cells, MaleCNS nt glutamate / sign -1, 40 of 82 PEN pairs and 53 of 92 EPG pairs above the connection cap, with NO receptor-table row of its own -- the -1 rides on the E-PG row's GluClalpha (Davis 2020 PB_2, tier alias). Published transmitter or function for ExR6 specifically: UNKNOWN. ER6 (4 cells, MaleCNS gaba, no row of its own): the R2 / R4 GABAergic literature (Omoto 2017, Fisher 2019, Kim 2019) does not cover it -- UNKNOWN. ER4m (11 cells, MaleCNS gaba) DOES have a receptor-table row. Modulatory status of all three: UNKNOWN."
  - "THE NEXT ARM, if the compass thread continues, is the one this batch makes obvious: the hold PLUS a wedge-local recurrence -- H3 with same_type_gain 1 (or a per-type EPG->EPG field) -- to ask whether the ring can hold a bump AT THE DRIVEN TILE once both the DC brake is off and the local recurrence is on. H3G already gets within 1 of 5 seeds of it. That arm is still two labelled instruments, so it decides a mechanism question, not an adoption."
  - "THE RATE MODEL'S RING RATES REMAIN THE WEAK LINK (2.1-2.6x high on H3's PEN and 3.5-4.6x on H_ER6's, but 1.0-1.7x on H_ExR6 and 1.4-1.5x on H3G). The protocol records no per-type ring rate: adding ExR6 / ER6 / ER4m to the recorded groups in cx_wedge.simulate would cost nothing and would close the loop on every number in section 3.2."
  - "MEASURE THE SPIKING LIF'S EFFECTIVE INPUT NOISE. Every slope bound in this audit is a property of SIGMA_MV = 2.0, which is hard-coded, never measured, and varied by no arm of this batch; the maximum slope is 25.3 Hz/mV at sigma 0.25 mV and the shipped gamma_crit 33.3 would be reached at sigma 0.17 mV. Until it is measured, the bound is a statement about the rate model at an assumed sigma, not about the LIF."
  - "MAKE --out REQUIRED ON scripts/cx_ring_structure.py. Its default --out of out/cx5/structure is what caused section 6's overwrite, and the overwrite's own time is now unrecoverable because the file that recorded it was itself overwritten by the regeneration."
open_questions:
  - "Can any shipped-gain configuration hold a bump AT THE DRIVEN TILE? Every bump in this batch and in cx5 either sits at the driven tile and fails the rate row 2.5-3.4x (F, R) or drifts (H3G 4/5 seeds, H3 5/5, cx5's R175 and CF). bump_survival_s does not distinguish them -- it scores the last confined frame wherever the bump is."
  - "What are ExR6's and ER6's transmitter, receptor and modulatory status in the animal? Both are UNKNOWN at the type level; both are load-bearing for the model's compass. ExR6 first: it carries 3-5x more of the PEN DC than ER6. The question is the transmitter call itself, not the pipeline stage that signs it (receptor_stage sign_changed 0 on EPG, PEN, PEG and Delta7)."
  - "Why is the rate model 2.1-2.6x high on H3's PEN? The candidate is its ring rates (110-185 Hz for 17 cells against a 308-cell population mean of 0.79-0.83 Hz in the LIF at rest). Recording per-type ring rates in the protocol would settle it in one batch."
  - "What is the spiking LIF's effective input noise? SIGMA_MV = 2.0 is an assumption that sets every slope bound quoted here, and nothing in this project has measured it."
  - "Is any arm here SILENT in the project's sense? Unanswerable from this batch: the protocol records group means, not the per-cell maximum INTERP.md 10 defines silent by. Recording a per-group max rate would close it."
  - "The GLNO sign now has a shipped-gain configuration in which it does something (H3 vs H3G: a saturated ring against a 159-164 Hz bump, GLNO at 136 vs 71 Hz). Whether that changes the relabel's status belongs to docs/audits/glno_relabel.md."
```

## Skeptic pass (independent, 2026-09-15)


Independent, CPU only (`PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1`), nothing committed, nothing under `docs/`
`scripts/` `flyverse/` `tests/` `out/` touched. Own scratch scripts (scratchpad only): `a_lif.py` / `a2.py` (the LIF
f-I, its derivative by adaptive quadrature rather than Gauss-Hermite, and the sigma sweep), `b_struct.py` (my own
weight-shaping pipeline from `c.W` -- own receptor-factor application, own cap, own superclass/type path gains, own
same-type damping, own fan-in, own `w_syn` -- **not** `brain._shaped_weights` / `cx_wedge.effective_weights`),
`d_ledger.py` / `d2.py` (own re-implementation of `probe_compass_room.bump_frames` and of the survival / rate / width /
confinement / group-rate rules on all 40 `.npz`, own `common.compare` and own Holm), `e_f.py` (cross-batch, stamps,
prose-vs-CSV diff). No cluster job was needed: every claim was checkable by recomputation.

---

### Refuted

**R1. "THE MAXIMUM GAIN THIS LIF HAS AT ANY OPERATING POINT IS 8.27 Hz/mV (u 8.60, 25.1 Hz), converged over quadrature
order" (section 0, 1.1 defect 3, 1.2, 1.4, 4; Report `summary` and `key_claims` line 7).** Three separate problems.

*(a) 8.27 is the quadrature, not the function.* `lif_fi_prime` estimates `f_sigma'(u) = (1/sigma) E[x f_det(u + sigma x)]`
by Gauss-Hermite. I computed the same integral by adaptive quadrature (`scipy.integrate.quad`, `epsabs` 1e-13, split at
threshold):

| | max f'(u) (Hz/mV) | at u (mV) | f there (Hz) |
|---|---|---|---|
| **exact (adaptive quadrature), sigma 2 mV** | **7.9969** | **8.6069** | **25.54** |
| `lif_fi_prime`, 15 nodes | 8.6571 | 8.598 | |
| `lif_fi_prime`, 31 nodes | 8.3128 | 9.241 | |
| `lif_fi_prime`, 61 nodes (**the tool's default; the audit's 8.27**) | 8.2747 | 8.602 | |
| `lif_fi_prime`, 101 nodes | 8.1864 | 8.871 | |
| `lif_fi_prime`, 201 nodes | 8.1270 | 8.771 | |
| `lif_fi_prime`, >= 401 nodes | NaN (numpy `hermegauss` overflows) | | |

The sequence is monotone decreasing after the first term and is still moving at 201 nodes; it is *not* converged, it is
walking down to 8.00. 8.27 is 3.4 % high, and the implementation cannot be pushed past ~201 nodes to show it.

*(b) The f-I the fixed point actually iterates is not bounded by 8.27 at all.* `rate_fixed_point`, `u_for_rate` and
`rate_at_gain` all evaluate `cx_wedge.lif_fi`, which is the **15-node** quadrature. That function has step-like
structure at the node images `u = theta - sigma x_j` (one of which is exactly u = 7.0). On a 1e-4 mV grid over
u in [0, 60] its own numerical slope reaches **7,528 Hz/mV at u 7.0001**, 2,099 Hz/mV somewhere in u in [9, 12] and
41.4 Hz/mV in [12, 20]; it exceeds 33.34 Hz/mV at 346 of 600,001 grid points. At the audit's quoted peak u = 8.60 the
central difference of `lif_fi` is **107.8 Hz/mV** against `lif_fi_prime`'s 8.27. The test said to pin this
(`test_lif_fi_prime_matches_a_finite_difference_and_is_bounded`) samples u = 9, 12, 20, 40, 80 at 10 / 3 / 2 / 2 / 2 %
-- i.e. everywhere except the peak the headline quotes.

*(c) The bound is a property of the assumed sigma, which is hard-coded and never measured.* `SIGMA_MV = 2.0`. Exact max
slope against sigma:

| sigma (mV) | 0.25 | 0.50 | 1.00 | 1.50 | **2.00** | 3.00 | 4.00 | 6.00 |
|---|---|---|---|---|---|---|---|---|
| max f' (Hz/mV) | 25.26 | 16.15 | 10.97 | 9.04 | **8.00** | 6.86 | 6.21 | 5.43 |

gamma_crit 33.34 is reached at sigma **0.168 mV**, 28.77 at 0.207 mV, 25.8 at 0.242 mV. Nothing in this project has
measured the effective input noise of the spiking LIF, and no arm of this batch varies sigma.

*What survives, and which number is right.* 5A's skeptic's "25.8 Hz/mV at u 7.1" is exactly reproduced from the
deterministic f-I (I get 25.782), and `f_det'` does diverge (622 Hz/mV at u 7.001, 3,939 at 7.0001), so the audit is
right that 25.8 is a sample on a divergence and not a maximum. **The right number to compare `gamma_E_crit` against is
the maximum of the same f-I whose slope defines gamma, i.e. 8.00 Hz/mV at sigma 2 mV -- not 8.27 and not 25.8.** With
33.3 vs 8.00 the sign of the comparison is unchanged, and S / G / C / CG measurably hold no bump, so the *conclusion*
stands. The word **"unreachable" is not justified as an absolute**: it is "unreachable in the rate model at the
sigma = 2 mV that model assumes", and the model's own 15-node numerics do not respect the bound.

**R2. The quadrature-convergence sequence is in no named file.** "8.66 at 15 nodes, 8.31 at 31, 8.27 at 61, 8.19 at 101"
(section 1.1, 1.4 and Report `key_claims` 7) and "the first implementation here returned 724-3,135 Hz/mV" (section 1.4)
appear only in the docstring of `scripts/cx_ring_structure.py`; no file under `out/cx6/` records them. `max_lif_slope`
8.274678082726133 / `_at_u` 8.60177 / `_at_hz` 25.129694787680478 *are* in `structure.json` and `validation.json`, so
the single number 8.27 is sourced and the convergence claim is not (the same complaint 5A's skeptic R1 made). The
script's own docstring also contradicts the audit: it says the peak is "at u ~ 9 mV, **~33 Hz**" where the audit and
`structure.json` say u 8.60, **25.1 Hz**.

**R3. The pre-pulse EPG ranges (section 3.1 "The pre-pulse state matters"; Report `key_claims` 2).** The sentence names
its file (`analysis/state.csv`, the 1 s settle) and then misquotes three of its columns:

| arm | audit | `state.csv` `epg_mean_pre` (my recomputation from the `.npz` is identical) |
|---|---|---|
| S | "9.1-10.0" | **8.7337 - 10.0317** (per seed 9.068, 9.339, **8.734**, 10.032, 9.780) |
| H_ExR6 | "9.1-10.0" | **8.9619 - 10.0433** |
| F | "9.1-10.1" | **8.7337 - 10.0532** |

"9.1" is seed 0's value (9.0678), not the minimum -- a number reconstructed rather than read from the named file, which
is exactly the failure `INTERP.md` 10.4 rule 28 exists against. Everything else in the same sentence is exact (S PEN
0.00-0.02, Delta7 8.42-11.78, GLNO 0.00; H3 19.73-62.91 / 8.79-42.66 / 28.27-103.51 / 32.49-131.39; H3G
10.33-40.52 / 1.03-16.46 / 12.38-68.83 / 5.06-58.78; R 16.67-42.79 / 10.47-34.00; H_ExR6 PEN 0.28-0.55; F PEN
0.00-0.16).

**R4. "Nothing in this batch is `silent` in the project's sense (`INTERP.md` 10: max rate < 0.5 Hz per cell): the
quietest population mean quoted here is S's PEN at 0.29-0.66 Hz during the pulse and 0.009-0.062 Hz after"
(section 3.1).** Wrong twice.
(i) The quietest population means the audit quotes are far lower: the `rest post` column of the same table is
**0.00151-0.00156 Hz** for S / H_ER6 / H_ER4m (printed there as 0.002), and S's `GLNO_mean_pre` is **exactly 0.000000 Hz
in 5 of 5 seeds** -- quoted two paragraphs later as "GLNO 0.00".
(ii) More seriously, `silent` is a per-cell max, and this protocol records only group means (`fb.brain.mean_rate` into
`g__*`); no quantity in the batch can decide it either way for PEN. Where a mean is exactly 0 the max is 0, so S's GLNO
during the settle **is** silent by the project's definition. The audit's own note that a quantity no file carries is not
quoted applies here.
*Checked and clean:* the audit never says PEN is silent. The only other uses of the word are the `GLNO` column of the
arm table and "GLNO-silent run", both the transmitter label, not the rate rule.

**R5. "The data-implied receptor tier `c`, which 5A's ranking put first, falls to 14th of 15" (section 1.3(i); Report
`key_claims` 8).** The rank move is not a finding. Ranks 5-15 of the 6A table and 1-10 of the `--legacy` table all have
`bump` False and `in_minus_out` = -1.78e-15 and are then ordered by `pen_pulse_hz`, whose values are

    1.0e-04 (b+f), 2.2e-09 (b), 2.1e-09 (a+b), 1.8e-09 (f), 1.7e-09 (a+f), 1.5e-09 (H_ER4m),
    1.4e-09 (shipped), 1.4e-09 (a), 5.2e-98 (c+f), 3.5e-179 (c), 3.5e-179 (a+c)

-- arithmetic underflow, not rates. Worse, the two rankings are keyed differently: 5A's published ranking (its section
1.3) is "by the fixed point's bump -- none; then gamma_crit(k1), then the PEN margin", which is what put `c` first; the
regenerated `--legacy` run of the same tool puts `c` **9th of 10**, not 1st. So "1st -> 14th of 15" compares a
gamma_crit-ordered list with a PEN-rate-ordered list whose tail is float noise -- and the audit's own section 6 already
says the legacy tie order "is float noise" without saying the same of its own. The substantive half is verified and
should carry the claim alone: `c`'s `gamma_E_crit` falls 33.340 -> 28.768 (-13.7 %), stays 3.6x above the LIF's maximum
slope, and its PEN margin *worsens*, -15.78 -> -17.63 mV.

**R6. "every compass cell except Delta7 is still exactly 0" (section 1.4, second bullet).** In the regenerated 5A fixed
point (`out/cx5/structure/structure.json`, `jacobian`) the per-group mean gains are

| state | EPG | PEN | PEG | EPGt | Delta7 | **Ring** | **GLNO** | leading eig |
|---|---|---|---|---|---|---|---|---|
| background | 8.8e-41 | 1.1e-08 | 5.6e-18 | 1.4e-09 | **4.676** | **0.242** | **0.049** | +0.1387 |
| pulse | 0 | 2.1e-10 | 1.7e-26 | 1.5e-07 | **5.224** | **0.359** | **0.320** | +0.0878 |

Delta7 4.68 / 5.22 and the leading eigenvalues +0.139 / +0.088 reproduce exactly, but Ring (the ER/ExR population, 308
cells) and GLNO are not zero, and the compass cells are effectively but not "exactly" zero. (The audit inherits the
phrasing from 5A's skeptic table; it is inaccurate in both.)

**R7. "7 of 8 shipped-gain cx5 arms called correctly" overstates the evidence: it is 3 of 4 distinct predictions.** The
call `predicted_bump` depends on one number, `local_kernels['direct']`, which takes only four values over the eight
arms, because the GLNO relabel never enters an EPG-recurrence criterion:

| | S, G | C, CG | F, FG | CF, CFG |
|---|---|---|---|---|
| local EPG->EPG (mV), my own shaping | 5.9987 | 6.9523 | 59.7926 | 66.2921 |
| gamma_E_crit (Hz/mV) | 33.340 | 28.768 | 3.345 | 3.017 |
| call | no bump | no bump | bump | bump |

So the eight calls are four predictions each made twice, and the single pair that *splits* in the measurement (F 4/4,
FG 0/4) is the pair the tool gets wrong. The classifier is "does the arm carry `same_type_gain` 1?", and the arms split
4/4 on exactly that flag. The rate half is 2 of 3 arms, as stated, and is sound.
Related attribution: the 144 / 160 Hz predictions are not new to 6A -- 5A's skeptic pass R3 already published 145 Hz for
`f` and 161 Hz for `cf` against the same measurements, from the same one-step coefficients (+59.76 / +66.51 mV). The
audit credits the skeptic for the Jacobian fix (1.1 defect 3) and for the 25.8 (1.4) but presents the F / CFG rate
validation as the fixed tool's own.

**R8. The incident is timestamped three different ways and none matches the file (section 6; Report `key_claims` 13).**
Section 6 says the argument-less run was at **08:03:28Z**; the Report `key_claims` and
`out/cx5/structure/REGENERATED_BY_6A.txt` both say **08:05Z**. Section 6 and the .txt say the artefacts were regenerated
"at **08:31Z**"; the regenerated file itself records `generated_utc` **2026-09-15T08:20:39Z** (mtime 08:22:11Z, and
`REGENERATED_BY_6A.txt` mtime 08:22:41Z). Nothing in `out/cx5/structure/` carries 08:31Z or 08:03:28Z.

**R9. "THE RATE MODEL'S RING RATES REMAIN THE WEAK LINK (2.2-4.6x high on the held arms)" (Report `recommendations` 5).**
Only two of the four held arms are in that band, and the ratios are:

| arm | predicted PEN during pulse | measured | ratio |
|---|---|---|---|
| H3 | 102.96 | 40.17-48.31 | **2.13-2.56x** (audit: "2.2x") |
| H_ER6 | 10.79 | 2.35-3.12 | 3.46-4.59x |
| H_ExR6 | 14.68 | 8.89-14.88 | **0.99-1.65x** |
| H3G | 40.23 | 26.49-29.00 | **1.39-1.52x** |

Section 3.2 and `key_claims` 9 state this correctly with the exceptions named; the recommendation line generalises past
its own data.

---

### Confirmed (recomputed independently)

**The batch, exactly.** My own `bump_frames` and my own survival / rate / width / confinement / group-rate code, run on
all 40 `.npz`, reproduce the shipped JSON `metrics` with **max |diff| = 0.0** on 30 quantities x 40 runs
(`survival_s`, `bump_hz_post`, `width_half_post`, `frac_confined_pre/during/post`, `vs_post_all`, EPG in/out during and
post, `epg_mean_pre`, `in_above_end`, `out_above_end`, and `PEN / Delta7 / PEG / Ring / GLNO / rest` x pre/during/post),
and `wedge_profile_end` exactly.

**Every verdict, z, p and Holm value of section 3.3.** My own `compare` (Welch-free: z = diff / SD(null), exact
Mann-Whitney, the deterministic-null rule) and my own Holm reproduce the whole table: H3 `PEN_mean_during` +43.5300,
z 294.0, p 0.0079365, Holm 0.03175, `result`; `PEN_mean_post` +52.3793, z 2504, `result`; H_ExR6 +10.3654 / z 70.01 and
+0.6888 / z 32.93; H_ER6 +2.2465 / z 15.17 and +0.1148 / z 5.489; **H_ER4m +0.0329, z 0.2221, p 0.6905, `null`** and
+0.0037, z 0.1772, p 0.8413, `null`; F +24.6890 / +26.4052; R +48.7165 / +47.9143; H3G +26.9410 / +23.7380; every
`survival_s` / `frac_confined_post` against S either a p = 1.0 `null` (H3, H_ER6, H_ER4m), a p = 0.42 `null` (H_ExR6) or
a zero-SD `undetermined` with p = 0.0079 (F +4.368, R +5.000, H3G +4.884).

**m = 4 is the declared rule applied, not a post-hoc choice.** `predeclared.json` `primaries.family` (stamped
08:09:25Z) says verbatim: "A member that returns no p ... is DROPPED from the Holm denominator and reported as a
magnitude". In every one of the seven arms exactly `bump_hz_post` and `width_half_post` return no p, because S has zero
finite values for both, leaving m = 4. Holm at the floor = 4 x 0.0079365 = **0.03175 <= 0.05**, as the audit says.
`m_max` 6 and "6 x 0.0079 = 0.047" are also in the predeclaration.

**The hold resolves to what the audit says, and the regexes bite exactly there.** All 10 H3/H3G runs record
`^(ExR6|ER6|ER4m)$` -> `^(PEN_|EPG$)` x 0, 17 pre cells, 88 post cells, **1,149 entries, 37,256 synapses**, pre types
{ER4m, ER6, ExR6}, post types {EPG, PEN_a(PEN1), PEN_b(PEN2)}; the single-type arms give 174 + 287 + 688 = **1,149** and
11,507 + 4,242 + 21,507 = **37,256**. Matched against the full 167,106-row type column: `^(ExR6|ER6|ER4m)$` matches
exactly those three types (2 + 4 + 11 = 17 cells) and nothing else; `^(PEN_|EPG$)` matches exactly EPG (46) and the two
PEN types (42) -- **EPGt excluded** (its `^EPG$` branch requires end-of-string). Both `cx_wedge.hold_edge_counts` and
`brain._shaped_weights` use `re.match` on the same column, so what is counted is what is zeroed.

**The hold is a factor 0 in `type_path_gain`, and its one unnamed side effect is negligible.** `parse_hold_edges`
appends `(pre, post, 0.0)` after the gE/gD/gR entries and before `same_type_gain` and the fan-in; with the flag absent
the list is entry-for-entry the previous one. The side effect the audit does not mention: `input_norm` computes its
total on the **shaped** W, so removing 1,149 edges raises the fan-in scale of any postsynaptic cell that was above
`input_norm_ref` = 5,000. I checked it: in the shipped arm exactly **one of 46 EPG** is above (total 5,099, scale
0.98058) and **no PEN** is (867-1,598); under H3 every EPG is below (max 4,362) and every scale is 1.000. So the hold
up-scales one EPG cell's surviving inputs by 2 %. (This also corrects a stale docstring in `cx_wedge.gained_blocks`,
"the fan-in scale ... stays 1.00 for these cells": it is 0.9806-1.00 shipped, 0.71-1.00 under `c`, 0.68-1.00 under `cf`.)

**The wedge matrices, the local kernels and gamma_E_crit.** My own shaping pipeline reproduces `matrices.npz` for every
stored configuration to **max |diff| 7.5e-4 mV^2** on the M16 PEN / PEG / Delta7 / Ring / net blocks and 1.5e-6 mV on
`direct` (float32 vs float64), and reproduces the criterion to four decimals:

| arm | local EPG->EPG (mV) mine / tool | gamma_E_crit mine / tool | predicted Hz | measured Hz | rel err |
|---|---|---|---|---|---|
| S, G | 5.9987 / 5.9987 | 33.340 / 33.340 | -- | (none) | -- |
| C, CG | 6.9523 / 6.9523 | 28.768 / 28.768 | -- | (none) | -- |
| F, FG | 59.7926 / 59.7926 | 3.345 / 3.345 | 144.30 | 152.50 (F) | 5.4 % |
| CF, CFG | 66.2921 / 66.2921 | 3.017 / 3.017 | 159.71 | 142.82 / 157.06 | 11.8 % / 1.7 % |

In every shipped arm the relay term contributes nothing (`gamma_relay` = 0 while the relays are below threshold), so
`gamma_E_crit` reduces exactly to `1 / (tau d_local)`. `rate_at_gain` is accurate where it is used: my exact quadrature
gives 144.07 Hz for gamma 3.35 and 160.05 Hz for gamma 3.01, matching the tool's 144.07 / 160.05. `u_for_rate` returns
6.627584 and 11.993303 mV; against the *exact* sigma = 2 convolution the 50 Hz value is right to 0.02 % and the 10 Hz
value is 2.4 % high (the exact inverse is 6.4745 mV) -- a consequence of the same 15-node quadrature, self-consistent
inside the tool and immaterial to any conclusion.

**The primaries and the readings built on them.** H3 hump centre **5.71-6.11** in 5/5 with vector strength 0.73-0.74 and
**17-19 of 35** off-block cells above 22 Hz; seed 0's `profile_end` `11 9 13 14 183 210 231 218 152 46 6 14 12 22 5 15`
exact; H3's driven EPGs 81.5-220.6 Hz during the pulse; H3 PEN 40.17-48.31 during / 49.34-53.96 after; H3G centres
**0.92 (seed 1), 5.30, 5.43, 10.01, 10.03** exact, with 11/11 driven cells above 22 Hz and 5/35 outside in seed 1;
H3G survival 4.76-5.00 (1 seed at 5.00), 158.98-164.17 Hz, width 3.85-4.00, confined 0.108-0.488; H_ExR6 confined in
**8.0-59.0 %** of the pulse frames and 0.0-0.4 % after; F seed 4 dies at 1.84 s with centre **7.89** and **2 of 11**
driven cells above threshold; every other range in the section 3.1 table verified. The claim "the hold also removes the
resting state" is if anything understated: at the *instant* t = 1.00 s (the `pre` sample, before any pulse) H3 sits at
PEN 44.3-53.1 Hz, GLNO 135.6-151.1, Delta7 110.9-125.5 and EPG out-of-block 21.3-99.2 Hz.

**The predeclared call, verbatim (g).** Applying the predeclaration's own words per seed to my own numbers: no arm
satisfies `survival >= 5 s AND width in [2.5, 5] AND rate in [5, 60] Hz` in any seed (F 5.00 s / 3.00 / 150.6-155.8;
R 5.00 / 3.00-3.06 / 200.6-203.2; H3G seed 0 5.00 / 4.00 / 160.8 -- all fail the rate row by 2.5-3.4x), so
"the DC balance is the whole story" (needs H3 >= 3/5) fails; H3's `PEN_mean_during` is 40.17-48.31 Hz, >= 1 Hz in 5/5,
so "not the story" fails. **The call is "necessary but not sufficient", exactly as the rule words it.** One
over-extension: the predeclared rule defines those three phrases **for H3 only**; `call.csv` applies the same words per
arm to H_ExR6, H_ER6, F, R and H3G, which the rule never covers (and for the labelled instrument F and the labelled
reference R the phrase means nothing). The audit's section 4 headline is about H3 and is correct.

**Cross-batch reproduction (e), stronger than claimed.** Comparing **all 38 recorded numeric metrics x 12 runs = 456
(metric, run) pairs** for S / F / R seeds 0-3, cx5 vs cx6: **max |cx5 - cx6| = 0.0**. The audit's "144 pairs" is the
12-metric subset written to `cross_batch_cx5.csv` (whose own max `absdiff` is 0.0).

**Shipped-path bit identity on CPU.** `out/cx5/smoke/smoke_default_path.json` vs `out/cx6/smoke/smoke_default_path.json`:
119 shared top-level fields, **118 equal**, the single difference `wall_s` (a timing field), plus the two new
record-keeping keys `hold_edges` / `hold_edges_resolved`. (The audit's "117 of 117" is a different count of the same
fact; my number is 118 of 119 with `wall_s` excluded from "recorded fields".)

**Stamps and the batch file (f).** `predeclared.json` mtime == its own `stamped_utc` **08:09:25Z**; no archive or
pre-amendment copy exists and none is needed. Order on disk: validation 07:50:51 < structure 07:59:41 < `batch.sh`
08:07:12 < `jobline_check.txt` 08:09:23 < **predeclaration 08:09:25** < `predeclare_stamp.txt` 08:09:33 <
`smoke_H3_cpu.json` 08:10:26 (after the predeclaration, before submission -- as the audit says) < `tree_state.json`
08:11:05 < **submit 08:11:21** < scheduler `submitted_at` 08:11:22.30 < last job finished 08:14:42.21 < analysis
08:18:32. `sha256(batch.sh)` = `01d3c38fe698935302043b5d14693f3152ee6b89daaeeb3f7ffac58a1bb10f3e` = the value recorded
in the predeclaration. `batch.sh` is ONE `cluster_run.py --name cx6 --minutes 30 --arm-block fam` call with 10 quoted
job strings (7 sequential arms per GLNO-silent job, 1 per H3G job), each ending `exit $((s0 | ... | s6))`;
`jobline_check.txt` records 0 / 7 / 7 / 7 for clean / arm 0 / arm 2 / arm 6 and the four hold regexes arriving intact in
python's `argv`.

**Run provenance.** 40 runs, `device` cuda in 40/40 JSONs and "device cuda" in 40/40 consoles; compiled-W md5
`ef23cc27bea13be7f6a96f3c04fd3737` on all 35 GLNO-silent runs and `7a10d93ba2086f2c76bcdabdca79b4ec` on all 5 H3G runs;
wall 13.6-24.9 s; `analysis.json` `n_runs` 40, `problems` []; `out/cx6_cluster.log` ends
`10 job(s), 0 failed  (5.0 min)  run dir /mnt/beegfs/neurome/runs/cx6-995cd5` / `client exit 0`; the scheduler receipt
lists 10 completed jobs on node1, GPU ids 0 and 1. `tests/test_cx_ring_structure.py` + `tests/test_cx_wedge_hold.py`:
**19 passed** on CPU (13 + 6).

**Section 3.4 per-seed scatter (rule 28).** All 15 pasted lines are byte-identical to the `arm,key,seeds,values` fields
of `out/cx6/analysis/scatter.csv`. Section 3.5's 16-wedge profile names its file and column (`state.csv`,
`profile_end`) and is exact. The per-seed lists that are *not* in section 3.4 -- H3G's five centres, F seed 4's centre
and driven-cell count, H3's five centres -- are all in `state.csv` (`centre_wedge_t5`, `in_above_t5`) and all reproduce.
The only rule-28 failure is R3 above.

**The ExR6 / ER6 / ER4m data (section 5), every digit, from the raw connectome.**

| | cells | MaleCNS `nt` | onto PEN | onto EPG | mV/pair (capped) | volley |
|---|---|---|---|---|---|---|
| ExR6 | 2 | glutamate (NT_SIGN -1) | 82 pairs, 1-177 syn, mean 64.3, **40 above cap 60** | 92 pairs, 30-113, mean 67.8, **53 above** | -8.659 / -15.010 | -16.91 / -30.02 |
| ER6 | 4 | gaba | 138 pairs, 1-45, mean 16.8, **0 above** | 149 pairs, 1-41, mean 12.9, **0 above** | -4.617 / -3.551 | -15.17 / -11.50 |
| ER4m | 11 | gaba | 182 pairs, 1-6, mean 1.7 | 506 pairs, 9-75, mean 41.9, **36 above** | -0.461 / **-11.393** | **-2.00 / -125.32** |

`receptor_rows.missing` lists EPGt, PEG, GLNO, ExR1, ExR4, ExR5, **ExR6**, ExR7, ExR8; **ER6 likewise has no row**
(confirmed directly against `flyverse/data/receptors_by_type.csv`: 0 rows for ExR6 and ER6, 7 rows each for ER4m, EPG,
PEN_a, PEN_b). The EPG glutamate row is `fast_sign` -1, `fast_neg_lead` **GluClalpha**, `fast_pos_lead` Nmdar2, tier
**alias**, source **davis2020 / PB_2** -- exactly as quoted.

**The literature statements (h) are not overstated in either direction.** For **ER6** the audit cites nothing beyond the
MaleCNS `nt` label; it explicitly says the Omoto 2017 / Fisher 2019 / Kim 2019 GABAergic-ring literature is about
R2 / R4 and that "for ER6 specifically, unknown". For **ExR6** it says published transmitter and function are unknown
and characterises Hulse 2021 (eLife 66039) as morphology + connectivity with ExR2 as the dopaminergic PPM3 class. Both
match 5A's skeptic correction 13 and I found nothing that overstates or understates them. One qualification (see
Corrections 7): "the inhibitory sign of ExR6 -> EPG comes from the E-PG row's GluClalpha" is true of the code path but
is not load-bearing -- `structure.json`'s own `receptor_stage` records `sign_changed` **0** on EPG, PEN, PEG and Delta7,
so the table reproduces the presynaptic NT_SIGN there and the sign would be -1 without it.

**Section 6, the regeneration itself.** Sixteen numbers, all reproduced from the regenerated
`out/cx5/structure/structure.json` to the printed digit: u_PEN(driven) **-15.7849** pulse / **-11.0575** background;
lambda_1 net **+10,019.0** with PEN +8,327.5, Delta7 +1,719.3, PEG +178.2, Ring -206.0; lambda_0 **-39,956.3**;
gamma_crit(k1) **1.9981**; ExR6 **-14.78 mV @ 183.2 Hz**, ER6 **-9.77 @ 116.3**, ER4m **-0.25 @ 22.3** onto the driven
PEN; b **1.860 / -41.00**, b+f **1.866 / -37.51**, c **1.634 / -17.63**, f **2.002 / -15.61**. The file's own
`generator` is `python scripts/cx_ring_structure.py --out out/cx5/structure --legacy --no-banc` and its `rate_model`
records `drive: rate`, `reduction: two-step` -- `--legacy` did what it claims. Only the *time* is wrong (R8).

**The predeclared predictions are the ones section 3.2 reports.** From `predeclared.json`
`predictions_from_the_structure_pass.per_arm`: S 0.00 Hz / u -15.78; H3 **102.96** / +27.04, after EPG in **251.95** out
**25.83**, PEN 103.44, Delta7 142.54, GLNO 246.99, `fixed_point_bump_after_release` true; H_ExR6 **14.68** / +1.26;
H_ER6 **10.79** / -4.72; H_ER4m 0.00 / -15.53; H3G **40.23** / -5.86, after EPG in **158.74**; F no fixed-point bump
with `epg_recurrence_saturation_hz` **144.302**. Nothing was written after the fact.

**Process note (not a defect in the science).** `tree_state.json` (08:11:05Z) hashes 14 files; three now differ.
`scripts/probe_vnc_drive.py` and `uv.lock` are the concurrent thread's. The third is this thread's own
**`scripts/cx_ring_structure.py`**, whose mtime is **08:18:29Z** -- after the stamp, after the last job finished
(08:14:42Z), three seconds before the analysis ran (08:18:32Z), and 28 minutes after `validation.md` was generated
(07:50:51Z) by an earlier version of the same file. `scripts/cx_wedge.py` (07:30:54Z) is unchanged and is the only one
of the three on the simulated path, so no run is affected; but "tree_state lists every file that shipped with the batch"
should say that the structure / analysis tool was edited after the stamp.

---

### Corrections (exact replacement text)

1. **Section 0, 1.1 (defect 3), 1.2, 1.4 (first bullet) and Report `key_claims` 7 -- the maximum LIF slope.** Replace
   "the maximum gain the LIF has anywhere is now a computed quantity: **8.27 Hz/mV at u 8.60 (25.1 Hz)**, converged
   (8.66 at 15 quadrature nodes, 8.31 at 31, 8.27 at 61, 8.19 at 101)" with
   > "The maximum slope of the smoothed f-I is **8.00 Hz/mV at u 8.61 (25.5 Hz)** at the sigma = 2 mV this rate model
   > assumes (adaptive quadrature; `lif_fi_prime`'s Gauss-Hermite estimate is not converged -- 8.66 / 8.31 / 8.27 / 8.19
   > / 8.13 at 15 / 31 / 61 / 101 / 201 nodes, still falling, and the tool's default 61-node value 8.27 is 3.4 % high;
   > the implementation returns NaN above ~201 nodes). Two limits on what that number means. It is a property of the
   > **assumed** sigma, not a measurement: the maximum slope is 25.3 Hz/mV at sigma 0.25 mV and 11.0 at 1 mV, and the
   > shipped ring's 33.3 Hz/mV would be reached at sigma 0.17 mV; `SIGMA_MV` is hard-coded and this project has never
   > measured the spiking LIF's effective input noise. And the f-I the fixed point actually iterates,
   > `cx_wedge.lif_fi` at 15 nodes, is not smooth near threshold: its own numerical slope reaches 7,528 Hz/mV at
   > u 7.0001 and exceeds 33.3 Hz/mV at 346 of 600,001 grid points on u in [0, 60]. So the right statement is that
   > **in this rate model, at sigma 2 mV, a gamma_crit of 28.8-33.3 Hz/mV is above every operating point of the exact
   > smoothed f-I by a factor of 3.6-4.2** -- not that it is "unreachable" as a property of the LIF. 5A's skeptic's
   > 25.8 Hz/mV at u 7.1 is exactly right for the *deterministic* f-I and is indeed a sample on its divergence
   > (622 Hz/mV at u 7.001), so it was never a maximum; the bounded comparison is the correct one to make, and it is
   > the measurement -- no bump in S / G / C / CG in 4 of 4 seeds -- that carries the conclusion."

2. **Section 1.1 defect 3 and 1.4 -- the convergence sequence and the discarded implementation.** Either record
   `max_slope` at several quadrature orders in `structure.json` and cite it, or delete the sequence and the
   "724-3,135 Hz/mV" from the audit and the Report: no file under `out/cx6/` carries either. Also fix the
   `lif_fi_prime` docstring, which says the peak is "at u ~ 9 mV, ~33 Hz" against `structure.json`'s u 8.60, 25.1 Hz.

3. **Section 3.1, the pre-pulse paragraph, and Report `key_claims` 2.** Replace "S 9.1-10.0 Hz EPG" with
   **"S 8.7-10.0 Hz EPG"**, "H_ExR6 9.1-10.0 / 0.28-0.55" with **"H_ExR6 9.0-10.0 / 0.28-0.55"**, and
   "F 9.1-10.1 / 0.00-0.16" with **"F 8.7-10.1 / 0.00-0.16"** (`analysis/state.csv`, `epg_mean_pre`: 8.7337-10.0317,
   8.9619-10.0433, 8.7337-10.0532).

4. **Section 3.1, the `silent` sentence.** Replace
   "Nothing in this batch is `silent` in the project's sense (`INTERP.md` 10: max rate < 0.5 Hz per cell): the quietest
   population mean quoted here is S's PEN at 0.29-0.66 Hz during the pulse and 0.009-0.062 Hz after."
   with
   > "This protocol records population **means** (`fb.brain.mean_rate` per group), not the per-cell maximum that
   > `INTERP.md` 10 defines `silent` by (max rate < 0.5 Hz per cell), so no arm here is shown to be silent or not
   > silent. What the means do say is that the relays are not off: S's PEN is 0.29-0.66 Hz during the pulse and
   > 0.009-0.062 Hz after, over 42 cells. Two populations quoted in this section are at or below the silence
   > threshold on the mean alone and would meet it: the `rest` group (0.0015 Hz in S) and S's GLNO during the settle
   > (exactly 0.000 Hz in 5 of 5 seeds)."

5. **Section 1.3(i) and Report `key_claims` 8 -- the ranking move.** Replace "The data-implied receptor tier `c`, which
   5A's ranking put **first**, falls to 14th of 15" with
   > "The data-implied receptor tier `c` is no longer the best non-instrument configuration on the criterion that
   > matters: its `gamma_E_crit` falls only 13.7 % (33.340 -> 28.768 Hz/mV), stays 3.6x above the largest slope the
   > smoothed LIF has anywhere, and its PEN margin *worsens* (-15.78 -> -17.63 mV), where 5A's ranking put it first for
   > having the lowest two-step gamma_crit. The table's *positions* below the two hold arms should not be read: ranks
   > 5-15 all have no bump and `in_minus_out` = -1.8e-15, and are then ordered by PEN rates between 1e-4 and 3e-179 Hz,
   > i.e. by arithmetic underflow -- the same float-noise tie-breaking section 6 notes for the legacy run. (5A's
   > published ranking was keyed on gamma_crit and the PEN margin; the `--legacy` rerun of this tool, keyed on the PEN
   > rate, puts `c` 9th of 10, not 1st.)"

6. **Section 1.4, second bullet.** Replace "every compass cell except Delta7 is still exactly 0" with
   > "every compass cell except Delta7 is still zero to within 1e-7 (EPG 9e-41, PEN 1e-08, PEG 6e-18, EPGt 1e-09), and
   > the two non-compass groups in the sub-circuit are not: Ring 0.242 (background) / 0.359 (pulse) and GLNO 0.049 /
   > 0.320."

7. **Section 1.2 / 1.3 and `validation.md` -- what "7 of 8" is worth.** Add after the validation table:
   > "The eight calls are four predictions made twice. `gamma_E_crit` depends only on the wedge-local one-step
   > EPG -> EPG coefficient, which takes four values over these arms (5.9987 mV for S and G, 6.9523 for C and CG,
   > 59.7926 for F and FG, 66.2921 for CF and CFG) -- the GLNO relabel cannot enter an EPG-recurrence criterion at all.
   > So the classifier is 'does this arm carry `same_type_gain` 1?', the arms split 4/4 on exactly that flag, and the
   > one pair the measurement splits (F 4/4, FG 0/4) is the pair the criterion gets wrong. Read as **3 of 4 distinct
   > predictions**, with the rate half 2 of 3. The 144 / 160 Hz saturation rates are also not new here: 5A's skeptic
   > pass R3 already derived 145 Hz for `f` and 161 for `cf` from the same coefficients, against the same measurements;
   > what 6A adds is the tool that produces them, the per-cell gains and the hold configurations."

8. **Section 6 and `out/cx5/structure/REGENERATED_BY_6A.txt`, the times.** The overwriting run's own provenance is the
   only record of when it happened; the audit gives 08:03:28Z in section 6 and 08:05Z in the Report and in the .txt.
   The regeneration is recorded as `generated_utc` **2026-09-15T08:20:39Z** (file mtimes 08:22:11Z / 08:22:41Z), not
   "08:31Z". Use the file's times in both places, and say which of 08:03:28Z / 08:05Z is the overwrite.

9. **Report `recommendations` 5.** Replace "(2.2-4.6x high on the held arms)" with
   > "(2.1-2.6x high on H3's PEN and 3.5-4.6x on H_ER6's, but 1.0-1.7x on H_ExR6 and 1.4-1.5x on H3G)".

10. **Section 2, the hold mechanism.** Add:
    > "One arithmetic consequence of installing the hold at the `type_path_gain` stage: `input_norm` takes its total on
    > the shaped matrix, so removing these edges also lifts the fan-in down-scaling of any postsynaptic cell that was
    > above the 5,000 reference. Checked: one of 46 EPG is (total 5,099, scale 0.9806) and no PEN is, and under H3 every
    > EPG is below (max 4,362, scale 1.000) -- so the hold up-scales one EPG's surviving inputs by 2 %, and the effect
    > is not part of the result."

11. **Section 2, the tree state.** Add: > "`scripts/cx_ring_structure.py` was edited again at 08:18:29Z, after the
    08:11:05Z `tree_state.json` stamp and after the batch finished, immediately before the analysis; `validation.md`
    (07:50:51Z) was written by the earlier version. Nothing on the simulated path changed -- `scripts/cx_wedge.py` is
    byte-identical to its stamped hash -- so no run is affected."

12. **Section 4 / `analysis/call.csv`.** Add: > "The predeclared rule defines 'the whole story' / 'necessary but not
    sufficient' / 'not the story' for **H3**. `call.csv` prints the same three phrases per arm; for the labelled
    instrument F and the labelled reference R they are outside the rule and should be read as the PEN and bump counts
    only."

13. **Section 5, the ExR6 sign.** Add after "comes from the *E-PG* row's GluClalpha (Davis 2020 PB_2, tier `alias`)":
    > "-- which reproduces the presynaptic NT_SIGN there rather than overriding it: `structure.json`'s `receptor_stage`
    > records `sign_changed` 0 on EPG, PEN, PEG and Delta7, so ExR6 -> EPG would be -1 with the receptor table off. What
    > is open is the transmitter call itself, not which stage of the pipeline signs it."

---

### Verdict

**Mostly sound.** Every measurement in this batch is exactly reproducible: my own `bump_frames` recovers all 40 runs'
six primaries and two dozen secondaries with zero difference; my own weight-shaping pipeline recovers every M16 block,
every local kernel and every `gamma_E_crit` to four decimals; every `compare` verdict, z, p and Holm value reproduces,
and m = 4 is the predeclaration's own drop rule applied, not a post-hoc denominator; the hold resolves to exactly the
17 cells, 88 targets, 1,149 entries and 37,256 synapses claimed, its regexes touch nothing else and exclude EPGt, and
its one unnamed side effect (the fan-in) is 2 % on one cell; S / F / R reproduce cx5 with max |diff| 0.0 not over the
144 pairs claimed but over all 456; the stamps order correctly, the predeclaration is unamended and its `batch.sh`
hash is the file that was submitted, the job-line exit expression was tested before submission, and the accidentally
overwritten 5A artefacts regenerate every number 5A's audit quotes from them to the printed digit. The predeclared call
-- **necessary but not sufficient** -- is what the rule's own words give for these numbers, and the two headline
findings (the hold lifts PEN ~100x and buys no bump; the same DC term holds the unstimulated ring at rest) survive the
pass intact and are, if anything, understated. What fails is again the argument around the result. The headline
"8.27 Hz/mV is the maximum gain this LIF has anywhere" is a 61-node quadrature estimate of a number that is 8.00, of a
function that is not the one the fixed point iterates (whose own numerical slope reaches 7,528 Hz/mV near threshold),
at a noise level that is assumed rather than measured and at which the answer is entirely determined -- so
"unreachable" is not earned even though the comparison's sign is right; "7 of 8 arms called correctly" is 3 of 4
distinct predictions from a one-bit classifier that tracks `same_type_gain`; the receptor tier's fall "from 1st to 14th
of 15" compares two different ranking keys with a tail ordered by float underflow; three pre-pulse EPG ranges quote a
seed-0 value where the named file gives a lower minimum; the `silent` sentence asserts something the protocol cannot
measure and picks the wrong quietest number; and the incident carries three mutually inconsistent timestamps, none of
which is the one in the file. None of these changes a row of `decision.csv` or the call; all of them change what a
reader would believe about how firmly it is established. With corrections 1-13 the audit is sound.

**What compass rounds 5 and 6 jointly say.** At the shipped gains the compass question is **closed as a null and opened
as an attribution**: 5A showed that no data-implied type-level change -- the GLNO relabel, the receptor tier, the
monoamine slow class, the fan-in, the cap -- moves the shipped ring off its uniform state, and that the binding
constraint is a DC balance on the relays rather than a missing k = 1 mode; 6A fixed the three tool defects 5A's skeptic
named and then ran the one `edges`-kind counterfactual that decision implied, and the answer is that the DC term is
**real and insufficient**: holding 2 ExR6 + 4 ER6 + 11 ER4m off PEN and EPG lifts the driven PEN from 0.29-0.66 Hz to
40.2-48.3 Hz (`result`, Holm 0.0317) -- ExR6 and ER6 carrying it, ER4m a `null` on PEN -- and buys zero seconds of
confined bump in 5 of 5 seeds, because the same term is what keeps the unstimulated ring near its drive, so removing it
saturates the ring instead of releasing it. Everything that holds a bump across both rounds is a **labelled instrument
or a labelled reference** (`same_type_gain` 1, gE 2 / gD 15, or the hold with the GLNO relabel), every one of them
fails `compass.EPG.bump_rate_hz` by 2.5-3.4x, and three of the four drift off the driven tile -- so
`compass.EPG.bump_survival_s` stays FAIL and the rate and width rows stay NOT_APPLICABLE, and **nothing in either round
licenses an adoption**. What the two rounds jointly license is one measurement and one data question: record per-type
ring rates in the protocol (the rate model's weakest link, 2.1-2.6x high on H3 and 3.5-4.6x on H_ER6, is its ring
rates, and no arm in either round measures them), and settle ExR6's transmitter, receptor and modulatory status in the
animal, which is `UNKNOWN` at the type level and is the single largest term in the balance both rounds are about.
