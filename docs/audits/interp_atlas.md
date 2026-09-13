# interp atlas -- what every motor readout does when population X is stimulated

**Tool.** `flyverse/interp/atlas.py` (`atlas`, `run_once`, `analyse_runs`, `make_populations`, `preset_populations`,
`readout_groups`, `validate`), CLI `scripts/interp_atlas.py {run|record|analyse}`, tests `AtlasTests` in
`tests/test_interp.py` (6 tests, CPU, 4.0 s). Contract: `docs/INTERP.md` section 4.5 and the stub in
`flyverse/interp/__init__.py`. Every parameter of the stub is kept; the implementation adds keyword-only parameters
with defaults (`split`, `n_null`, `min_cells`, `modules`, `include_pn`, `per_body`, `top`, `summary`, `out_dir`,
`quiet`, `sd_floor`, `per_body_for`).

**Status (2026-09-12).** Built and validated. The three reference results of `common.VALIDATION['atlas']` reproduce
(the wind DN flips, PFL3 -> DNa02, DNa02 -> leg motor neurons), and the atlas was run over all 480 descending-neuron
types by side, every sensory class and superclass and the model's five named sensory channels -- 981 populations x 27
motor readouts x 3 independent runs, one cluster batch, `0 failed`, every job on `device cuda`.

Files written by this round (every number below comes from one of them):

| file | what |
|---|---|
| `out/atlas/val_r{0,1,2}.npz` / `.json` / `.txt` | the validation preset, three independent runs (seeds 0/1/2), GPU |
| `out/atlas/dn_r{0,1,2}.npz` / `.json` / `.txt` | every DN type + every sensory class, three runs, GPU |
| `out/atlas_cluster.log` | the console output of the one `cluster_run.py` batch (`6 job(s), 0 failed  (3.0 min)`) |
| `out/interp/atlas/validation.json` | the Result of the validation arms (pulse-window mean) |
| `out/interp/atlas/validation_half.json` | the same, second half of the pulse window (`--summary half`) |
| `out/interp/atlas/dn_sensory.json` | the Result of the 981-population atlas (23.2 MB) |
| `out/interp/atlas/*_analyse.txt` | the printed tables of each `analyse` call (quoted verbatim in section 5) |

`Result.check()` is empty on all three JSONs.

---

## 1. What the tool measures

`scripts/screen_dns.py` generalised into the toolkit's contract. One `FlyBrain(batch=B)`; inside a batch, row `r`
carries the pulse for population `r` (`FlyBrain.stimulate`, Poisson, `hz` for `ms`, after `settle_ms` of quiet) and
the remaining rows carry **no pulse at all** -- those are the null rows, the matched control of every number here.
The populations are walked in chunks of `batch - n_null`. `replicates` independent runs (separate jobs, separate
seeds -- the project's replicate unit) give the scatter, and every (population, readout) difference goes through
`common.compare`, never as a bare delta.

Per (population, readout) the tool records four window statistics -- `mean` (the whole pulse), `half` (its second
half, the steady state the 30 s screens report), `final` (the last frame, what `scripts/benchmark.py`'s `dn` section
reads after 400 ms) and `pre` (the pre-pulse baseline) -- and reports the chosen one (`--summary`, default `mean`)
against the null.

Readouts are the `MotorRates` cell sets themselves (`flyverse.motor.motor_groups` / `wing_groups`, pooled exactly as
`read_motor` pools them; 21 groups, 532 distinct cells), their six left-right differences, and -- when `--pattern`
is given -- every type it matches through `screen.TypeRecorder`, by soma side.

The tool reads the model and changes nothing. A pulse is `FlyBrain.stimulate`; a context is the FlyBrain's own
sensory entry points (`fb.wind` / `fb.smell` / `fb.taste`); returning to rest between chunks is `FlyBrain.reset()`,
the public entry point; the connectome, `LIFParams` and `OpticParams` are whatever the caller resolved (here: the
shipped defaults, `receptor_model 'sign'`, `w_syn 0.275`, `conn_cap 60`, `same_type_gain 0.1`, `adapt_jump 1.5`,
`input_norm_alpha 1 / ref 5000`, `t_ref 2.2`, `prune_frozen True`).

### 1.1 Three columns that keep a row honest

* **`self_drive` / `readout_shared_cells`.** `turn_L` *is* DNa02_L; `gf` *is* DNp01; `wind_ipsi_L` contains DNp18_L,
  DNge016_L, DNg05_a_L, DNp19_L, DNge175_L; `fwd_dn` contains DNa01/DNa02/DNa03/DNa04/DNb01/DNp09. Stimulating those
  cells moves those readouts **by construction**, not through a circuit. `analyse_runs` rebuilds the readout cell
  sets on the CPU (`readout_membership`) and marks every row where the stimulated population intersects the readout,
  and the summary reports both the top mover and the top mover that is *not* inside the readout
  (`top_population_per_readout[...]['top_upstream']`). 50 of the 3,212 `result` rows are self-drive.
* **`out_to_pruned_share`.** With `prune_frozen` (the shipped default) `FlyBrain` freezes and prunes the optic rate
  units (`OpticLobe.rate_idx` = superclass `ol_intrinsic` minus photoreceptors) out of the LIF matrix, so a pulse into
  a population whose output goes there reaches nothing. `pruned_out_shares` computes, per population, the share of its
  outgoing |W| that lands on pruned cells. This is why the photoreceptors read null (section 6.2), and the tool says so
  instead of leaving it unexplained.
* **`z_floor`.** `common.compare`'s z is `diff / SD(null)`, which is NaN when the null arm is bit-identical -- and a
  silent motor group's null rows *are* bit-identical (SD 0). `z_floor` repeats it with the null SD floored at
  `SD_FLOOR_HZ = 0.05` Hz, a declared constant well below the toolkit's 0.5 Hz 'never firing' bound, not a fitted one.
  The verdict uses `z_floor` and keeps `common.compare`'s full dict (`verdict_compare`, `welch`, `U`, `p`) beside it,
  and `underpowered` still wins whenever an arm has fewer than three runs.

### 1.2 The arms and what counts as a replicate

The stimulus arm of a (population, readout) is that population's value **in each independent run** -- 3 draws, the
replicate unit is the run (`docs/BATCH_SIM.md`). The null arm is every unstimulated row of every run for the same
readout. The **realised** layout, not the parameter: `n_null = 4` and `per_chunk = batch - n_null = 60`, so the 981
populations fall into 16 full chunks of 60 populations contributing 4 null rows each plus a last chunk of 21
populations contributing 43 -- **107 null rows per run** (`out/atlas_cluster.log`: `981 populations x 27 readouts
107 null rows`) x 3 runs = **321 draws** (`null_n` is 321 on every row of `out/interp/atlas/dn_sensory.json`); the
validation batch is one chunk of 10, so 54 per run and 162 draws. `replicates.null.rows_per_run` in the Result
records the `n_null` **parameter** (4), not the realised 107, and `replicates.null.ids` is truncated to the first 64
ids (`flyverse/interp/atlas.py:762`) -- read the log or `null_n` for the realised count. Batch rows are independent
Poisson draws of the same compiled model, so they are legitimate null draws; they are not extra *stimulus*
replicates, and the tool never treats them as such. One caveat the whole tool rests on: the null rows are always the
**same batch slots** (rows 60-63 of every full chunk, 21-63 of the last), so a hypothetical per-row bias would not
average out -- the argument is that batch rows are exchangeable with the stimulated ones, not that they were
randomised.

---

## 2. How this round was run (the exact lines)

One cluster batch, six commands, run concurrently:

```bash
python scripts/cluster_run.py --name atlas --minutes 90 \
 "mkdir -p out/atlas && python -c 'import torch; assert torch.cuda.is_available()' && python scripts/interp_atlas.py run --preset validation  --seed 0 --out out/atlas/val_r0 > out/atlas/val_r0.txt 2>&1; cat out/atlas/val_r0.txt" \
 "... --preset validation  --seed 1 --out out/atlas/val_r1 ..." \
 "... --preset validation  --seed 2 --out out/atlas/val_r2 ..." \
 "... --preset dn+sensory  --seed 0 --out out/atlas/dn_r0  ..." \
 "... --preset dn+sensory  --seed 1 --out out/atlas/dn_r1  ..." \
 "... --preset dn+sensory  --seed 2 --out out/atlas/dn_r2  ..." \
 --fetch out/atlas/ 2>&1 | tee out/atlas_cluster.log
```

then, on the desktop (CPU):

```bash
PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/val_r*" --top 8  --json out/interp/atlas/validation.json
PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/val_r*" --top 8  --summary half --json out/interp/atlas/validation_half.json
PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/dn_r*"  --top 20 --json out/interp/atlas/dn_sensory.json
```

`out/atlas_cluster.log` ends `6 job(s), 0 failed  (3.0 min)  run dir /mnt/beegfs/neurome/runs/atlas-20c1b7`, and every
run's own log line reads `device cuda` (`NVIDIA B200`, torch 2.11.0+cu128, host node1). Wall time per run: 114.8 /
114.2 / 112.4 s for the 981-population batch, 48.8 s for the validation batch.

---

## 3. Validation -- reproduced, side by side

`common.VALIDATION['atlas']` asks for three things. All three reproduce; `validation.status` in
`out/interp/atlas/validation.json` reads `reproduced`.

### 3.1 The wind flips

The reference protocol (`scripts/benchmark.py::sec_wind`, `scripts/screen_steering.py`) pins a fly in the room at the
plume-free spot (0.55, 0.35, 0.75) facing -90 deg and +90 deg in the default 0.3 m/s wind, records 10 s per heading,
and reports `flip = (L-R)_wind-left - (L-R)_wind-right`. The atlas arm does not build a room: it takes the per-cell
Johnston's-organ rates `senses.Wind.rates` produces for those two poses **at the nominal wind direction**
(`jo_wind_rates`; `wind_deflections(-90) = (+0.4243, -0.4243)`, the mirror at +90) and presents them as a stimulated
population of 335 JO-C/JO-E cells, 3 s per arm, plus a **head-on** arm (`wind_deflections(0) = (+0.4243, +0.4243)`,
both antennae deflected equally) as an internal control the reference protocol does not have.

"Nominal" is the one real protocol difference, and it is not cosmetic: the room's `Air` meanders the wind direction
by **+-20 deg with an 8 s period** (`flyverse/air.py` `WindParams.meander_deg 20`, `meander_period_s 8`;
`Air.direction` adds it and `Air.vector` uses it), and `benchmark.py::sec_wind` averages 10 s of that. Over that
window the reference's deflection on the stimulated antenna wanders **+0.254 .. +0.544** (mean **+0.393**, JO drive
`2 + 50 * d` = **21.65 Hz**) against the atlas arm's fixed **+0.4243** (**23.21 Hz**) -- about **7 % more drive in
the atlas**. `atlas.wind_deflections` reproduces `Air.deflections` only at `t = 0`, where the meander term is zero.

| type | reference (L-R flip, Hz) | source | atlas flip, Hz (mean of 3 runs) | per-run values | run sd | head-on (L-R) |
|---|---|---|---|---|---|---|
| **DNp18** | **+45.2** | `out/benchmark_suite.json` `wind.DNp18_flip_hz` 45.237 (NOTES session 8: +45) | **+50.98** | +51.85 / +49.45 / +51.63 | 1.33 | +13.45 |
| **DNp33** | **-49.6** | `out/benchmark_suite.json` `wind.DNp33_flip_hz` -49.554 (NOTES: -49) | **-49.31** | -49.14 / -49.97 / -48.80 | 0.60 | -10.68 |
| WED080 | -41 | NOTES session 8 | -44.87 | -45.72 / -44.43 / -44.47 | 0.73 | -17.07 |
| DNge016 | +30 | NOTES session 8 | +33.26 | +33.77 / +32.71 / +33.31 | 0.53 | +9.89 |
| DNg99 | -19 | NOTES session 8 | -24.94 | -23.96 / -25.25 / -25.60 | 0.86 | -5.11 |
| DNp73 | +18 | NOTES session 8 | +16.62 | +16.71 / +18.02 / +15.13 | 1.45 | +19.51 |
| DNp19 | +17 | NOTES session 8 | +18.95 | +18.57 / +20.13 / +18.14 | 1.05 | +9.75 |
| DNg05_a | (listed, no value) | NOTES session 8 | +14.78 | +13.58 / +16.21 / +14.55 | 1.33 | +1.66 |

Benchmark criteria `wind.DNp18_flip_hz >= 15` and `wind.DNp33_flip_hz <= -15` both pass. Seven of the eight wind types
reproduce the reference sign and magnitude within a few Hz on a protocol that shares no code with the reference beyond
`senses.Wind.rates` itself: no room, no ray tracer, no `Air`, no 10 s rollout, 3 s of direct JO drive instead.
With `--summary half` (the second half of the 3 s window, `out/interp/atlas/validation_half.json`) DNp18 is +48.87
(sd 1.88) and DNp33 -48.85 (sd 1.62), so the numbers are a steady state and not an onset transient.

**The DNp18 overshoot is the meander.** DNp18 is the one type that runs high: +50.98 (`out/interp/atlas/validation.json`)
and +50.59 in an independent three-seed repeat (`out/skeptic_atlas/val_analyse.txt` `wind.DNp18_flip_hz` 50.5937)
against the reference +45.2, i.e. **+12 %** -- the sign of the extra 7 % of JO drive above. Presenting the same JO
drive at eight phases of one meander cycle and averaging the flips closes it: DNp18 **+47.49** (phases +42.6 .. +52.2)
against the fixed-direction **+51.27** in the same batch, DNp33 **-48.59** (fixed -49.24), WED080 **-42.78** (fixed
-45.98), DNge016 **+31.64** (fixed +33.83) -- every one within ~2 Hz of its reference
(`out/skeptic_atlas/meander_s10.txt`; generator `scripts/skeptic_atlas_wind_meander.py`). The flips need no room, but
they do need the room's wind statistics.

The per-arm table (`out/interp/atlas/validation_analyse.txt`) shows the mechanism, which the flip statistic hides:

```
=== type.DNp18_LR ===  (null +0.000 Hz over 3 runs)
 rank      population  n_cells  stim_mean  null_mean    diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1    JO_wind_left      335    +32.727     +0.000 +32.727 +654.542 +0.000  result       False   +1.025       3
    2   JO_wind_right      335    -18.249     +0.000 -18.249 -364.979 +0.000  result       False   +0.467       3
    3 JO_wind_head_on      335    +13.450     +0.000 +13.450 +269.004 +0.000  result       False   +0.769       3

=== type.DNp33_LR ===  (null +0.000 Hz over 3 runs)
    1   JO_wind_right      335    +30.984     +0.000 +30.984 +619.673 +0.000  result       False   +0.296       3
    2    JO_wind_left      335    -18.322     +0.000 -18.322 -366.431 +0.000  result       False   +0.318       3
    3 JO_wind_head_on      335    -10.677     +0.000 -10.677 -213.533 +0.000  result       False   +0.490       3
```

**New: the wind DNs carry a fixed left-right offset under symmetric drive.** Head-on wind deflects both antennae by
the same +0.424 and therefore drives the two JO fields symmetrically, yet DNp18 sits at L-R **+13.45 Hz**, DNp73 at
**+19.51**, WED080 at **-17.07**, DNge016 at **+9.89**, DNp19 at **+9.75**, DNp33 at **-10.68**, DNg05_a at **+1.66**
(all `result`, run sd 0.4-2.0 Hz). **None** of DNp73's flip is this offset. The flip is a difference of differences,
so a fixed head-on offset enters both arms and cancels exactly: `out/interp/atlas/validation.json` puts DNp73's L-R
at **+26.51** under wind-left and **+9.89** under wind-right (head-on +19.51), and 26.51 - 9.89 = **+16.62** whatever
the offset is. The defensible statement is the opposite one -- DNp73's L-R **never changes sign**, so a
*single-condition* L-R readout would be dominated by the +19.5 Hz fixed gradient while the flip statistic is immune
to it. This is the same kind of fixed anatomical gradient that dynamics round 1 found in PFL3 (a 2-3 Hz L-R offset of
bump position, `docs/audits/compass_room.md`) -- a per-type asymmetry of the wiring, not a signal, and a measurement
rather than an artefact. The **difference of differences** that `benchmark.py` and `screen_steering.py` use cancels
any term common to the two arms exactly; the head-on value is **not** that common term for half the set --
`((L-R)_left + (L-R)_right)/2` minus head-on is +17.01 (DNp33), +8.64 (DNg99), +7.32 (WED080), -6.21 (DNp18) Hz,
against -4.57 / -1.89 / -1.31 / -0.17 for DNp19 / DNge016 / DNp73 / DNg05_a
(`out/interp/atlas/validation.json`), so the head-on arm bounds the offset rather than measuring it. No change to
`body.py` follows from this: it is a measurement of what is already there.

### 3.2 DNa02 -> leg motor neurons

| quantity | reference | source | atlas (3 runs) | per-run | sd |
|---|---|---|---|---|---|
| DNa02_L 150 Hz, left leg MNs | 3.1 Hz | `benchmark.py` `dn.DNa02_L` note / NOTES session 4 | 2.45 Hz | -- | -- |
| DNa02_L 150 Hz, right leg MNs | 0.1 Hz | same | 0.32 Hz | -- | -- |
| **DNa02_L 150 Hz, leg L-R asymmetry** | **2.581 Hz** | `out/benchmark_suite.json` `dn.DNa02_L_leg_asym_hz` (criterion `> 0.3`) | **2.133 Hz** | 2.176 / 2.387 / 1.838 | **0.277** |
| the same, measured in the independent 981-population batch | 2.581 Hz | as above | **2.235 Hz** | 2.107 / 2.228 / 2.370 | 0.132 |
| the same, last frame of the pulse | -- | -- | 2.071 Hz (validation batch) / 2.122 (dn batch) | -- | -- |

The criterion passes (`> 0.3`) and the sign and the ipsilateral bias are exactly the reference's. The measured value
sits 1.6 run-sd below the suite's 2.581: the protocols differ (the suite runs a bare `brain.Brain` with
`set_poisson` for 400 ms from rest; the atlas runs a `FlyBrain` with 200 ms of settle first, and averages the whole
pulse window rather than reading the final rate), and the two independent atlas batches agree with each other
(2.133 +- 0.277 and 2.235 +- 0.132) better than either agrees with the suite. Quoting it as "reproduced to within the
protocol difference" is the honest reading; it is not bit-identity and is not claimed as such.

`DNa02_R` gives the mirror: leg L-R **-1.666 Hz** (sd 0.091) in the validation batch, -1.415 in the dn batch.

### 3.3 PFL3 -> DNa02

| quantity | reference | source | atlas (3 runs) | per-run | sd |
|---|---|---|---|---|---|
| PFL3_L at 80 Hz -> DNa02_**R** | 22.6 Hz | NOTES session 8 | **24.11 Hz** | 25.25 / 22.55 / 24.52 | 1.40 |
| PFL3_L at 80 Hz -> DNa02_**L** | 0.0 Hz | NOTES session 8 | **0.00 Hz** | 0 / 0 / 0 | 0 |
| PFL3_R at 80 Hz -> DNa02_L | (mirror) | -- | 22.33 Hz | -- | 2.01 |
| PFL3_R at 80 Hz -> DNa02_R | (mirror) | -- | 0.00 Hz | -- | 0 |

Exactly the reference, including the zero on the ipsilateral side: PFL3 -> DNa02 is a clean contralateral projection
in this model, and it works when PFL3 is driven. (Dynamics round 1's finding stands beside it unchanged: in the room
the CX never drives PFL3 hard enough for this to fire -- `docs/audits/compass_room.md`. The atlas measures the
*output* pathway, not whether anything upstream uses it.)

With `--summary half` PFL3_L -> DNa02_R is 27.22 Hz (26.21 / 26.56 / 28.90), i.e. still rising over 400 ms.

---

## 4. The atlas: 981 populations x 27 motor readouts

`--preset dn+sensory` builds, from `cn.load()`:

* **953 descending-neuron populations** -- all 480 distinct types with `superclass == descending_neuron` *plus* the
  four untyped DN bodies, **1,314 cells in all** (1,310 typed + 4 untyped; counted from `populations` in
  `out/interp/atlas/dn_sensory.json`), split per type and, where both sides exist, per soma side (`DNa02_L`,
  `DNa02_R`, ...; a type present on one side only keeps its bare name; the four untyped DN cells become
  `(untyped)_L`, 3 cells, `body:11851|13539|55579`, and `(untyped)_R`, 1 cell, `body:13964`). So both "all 480 DN
  types" and "every DN cell" hold here -- the untyped bodies are populations 952 and 953, not a gap.
* **13 sensory-class populations** -- every value of the `class` column among cells whose superclass contains
  `sensory` (`chemosensory` 58, `gustatory` 1428, `hygrosensory` 66, `mechanosensory` 1733,
  `mechanosensory_proprioceptive` 1454, `mechanosensory_tactile` 2558, `mechanosensory_tbc` 11, `olfactory` 2639,
  `thermosensory` 25, `unknown_sensory` 1707, `visual` 6091 split L/R/unsided).
* **10 sensory-superclass populations** -- `cb_sensory` 4868, `vnc_sensory` 6365, `ol_sensory` 6098,
  `sensory_ascending` 537, `sensory_descending` 12 and the `_tbc` remnants; this is the split the `class` column
  loses (VNC mechanosensors vs brain mechanosensors).
* **5 named channels** the model's own front ends read: `channel.photoreceptors` (6091),
  `channel.ORN_all` (`senses.Smell.orn_idx`, 2639), `channel.sweet_GRN` (`senses.Taste.sweet`, 165),
  `channel.JO_wind_left` / `channel.JO_wind_right` (335 JO cells each at the per-cell rates `senses.Wind` produces).

Every recorded `spec` round-trips: `common.resolve(c, spec)` returns exactly the population's cells for all 976
grammar-expressible populations (the five channel populations record the constructor that made them instead).

Protocol: 150 Hz for 400 ms after 200 ms of settle, batch 64 with 4 null rows per chunk (107 realised per run,
section 1.2), no sensory context, three runs. 3,212 (population, readout) pairs out of 26,487 reach verdict `result`
-- **with its floor attached**: in this round every null arm is exactly 0.000 Hz with SD exactly 0, so the verdict is
set entirely by the declared `SD_FLOOR_HZ = 0.05` Hz. The smallest `result` is `|diff| = 0.15000086` Hz = 3 x 0.05
exactly, 846 of the 3,212 (26 %) sit below 0.5 Hz and 1,461 (45 %) below 1 Hz
(`out/interp/atlas/dn_sensory.json`). "Reaches verdict `result`" therefore means "moved a silent readout by
>= 0.15 Hz", not "moved it meaningfully"; the floor is declared-not-fitted (section 7.3), and any downstream use --
the ledger, the export -- should carry the `diff`, not the verdict count.

578 of the 981 populations move at least one motor readout to `result`, **347 move nothing at all** (`max |diff|`
exactly 0 across all 27 readouts) -- and the remaining **56** move something by a non-zero amount that never reaches
`result`. The two categories are not exhaustive.

---

## 5. Top 20 movers of each motor readout

Verbatim from `out/interp/atlas/dn_sensory_analyse.txt` (`python scripts/interp_atlas.py analyse --runs "out/atlas/dn_r*"
--top 20 --json out/interp/atlas/dn_sensory.json`). `diff` is Hz above the null rows; `z_floor` is `diff / max(SD(null),
0.05 Hz)`; `stim_sd` is the scatter over the three runs; `self_drive` True means the stimulated cells are themselves
part of the readout (section 1.1). A readout with fewer than 20 rows had fewer than 20 populations that moved it at
all -- the table is not padded with zeros.

```

=== back_dn ===  (null +0.000 Hz over 3 runs)
 rank                   population  n_cells  stim_mean  null_mean    diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1                        MDN_R        2    +61.276     +0.000 +61.276 +1225.516 +0.000  result        True   +7.219       3
    2                        MDN_L        2    +60.762     +0.000 +60.762 +1215.248 +0.000  result        True   +3.976       3
    3        superclass.cb_sensory     4868    +55.837     +0.000 +55.837 +1116.732 +0.000  result       False   +4.879       3
    4              class.gustatory     1428    +55.097     +0.000 +55.097 +1101.947 +0.000  result       False   +4.354       3
    5         class.mechanosensory     1733    +50.457     +0.000 +50.457 +1009.135 +0.000  result       False   +1.100       3
    6 class.mechanosensory_tactile     2558    +36.102     +0.000 +36.102  +722.040 +0.000  result       False   +0.902       3
    7           class.chemosensory       58    +35.796     +0.000 +35.796  +715.918 +0.000  result       False   +4.433       3
    8       superclass.vnc_sensory     6365    +31.645     +0.000 +31.645  +632.894 +0.000  result       False   +2.216       3
    9        class.unknown_sensory     1707    +27.900     +0.000 +27.900  +558.006 +0.000  result       False   +2.443       3
   10 superclass.sensory_ascending      537    +25.552     +0.000 +25.552  +511.042 +0.000  result       False   +3.113       3
   11                    DNpe023_L        1    +24.429     +0.000 +24.429  +488.571 +0.000  result       False   +5.556       3
   12                    DNpe023_R        1    +23.291     +0.000 +23.291  +465.827 +0.000  result       False   +4.090       3
   13                      DNp43_L        1    +19.978     +0.000 +19.978  +399.561 +0.000  result       False   +2.586       3
   14                    DNge130_R        1    +17.805     +0.000 +17.805  +356.108 +0.000  result       False   +0.840       3
   15                    DNpe053_R        1    +17.320     +0.000 +17.320  +346.397 +0.000  result       False   +2.249       3
   16                    DNge124_R        1    +15.804     +0.000 +15.804  +316.074 +0.000  result       False   +4.617       3
   17                      DNp43_R        1    +14.817     +0.000 +14.817  +296.346 +0.000  result       False   +2.452       3
   18                    DNpe053_L        1    +14.547     +0.000 +14.547  +290.940 +0.000  result       False   +0.481       3
   19                    DNpe022_L        1    +14.533     +0.000 +14.533  +290.665 +0.000  result       False   +2.071       3
   20                    DNge050_R        1    +14.264     +0.000 +14.264  +285.273 +0.000  result       False   +5.842       3

=== fwd_dn ===  (null +0.000 Hz over 3 runs)
 rank                   population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1                      DNa04_L        1    +13.201     +0.000 +13.201 +264.016 +0.000  result        True   +2.175       3
    2                      DNa03_L        1    +13.111     +0.000 +13.111 +262.229 +0.000  result        True   +3.197       3
    3                      DNb01_L        1    +12.561     +0.000 +12.561 +251.218 +0.000  result        True   +0.768       3
    4                      DNa03_R        1    +12.305     +0.000 +12.305 +246.105 +0.000  result        True   +4.392       3
    5                      DNp09_R        1    +11.906     +0.000 +11.906 +238.126 +0.000  result        True   +0.960       3
    6                      DNp09_L        1    +11.530     +0.000 +11.530 +230.608 +0.000  result        True   +1.726       3
    7                      DNa04_R        1    +11.220     +0.000 +11.220 +224.401 +0.000  result        True   +2.253       3
    8                      DNa01_L        1    +11.217     +0.000 +11.217 +224.342 +0.000  result        True   +1.988       3
    9                      DNa01_R        1    +10.720     +0.000 +10.720 +214.408 +0.000  result        True   +0.219       3
   10                      DNb01_R        1    +10.651     +0.000 +10.651 +213.015 +0.000  result        True   +0.272       3
   11              class.gustatory     1428    +10.610     +0.000 +10.610 +212.193 +0.000  result       False   +0.891       3
   12       superclass.vnc_sensory     6365     +8.797     +0.000  +8.797 +175.949 +0.000  result       False   +0.467       3
   13        class.unknown_sensory     1707     +8.141     +0.000  +8.141 +162.816 +0.000  result       False   +0.614       3
   14           class.chemosensory       58     +4.263     +0.000  +4.263  +85.251 +0.000  result       False   +0.649       3
   15 class.mechanosensory_tactile     2558     +3.239     +0.000  +3.239  +64.787 +0.000  result       False   +0.084       3
   16                    DNae007_L        1     +3.089     +0.000  +3.089  +61.789 +0.000  result       False   +0.262       3
   17                    DNge130_R        1     +2.458     +0.000  +2.458  +49.162 +0.000  result       False   +0.889       3
   18                    DNae007_R        1     +2.436     +0.000  +2.436  +48.719 +0.000  result       False   +0.746       3
   19                    DNde003_L        2     +2.279     +0.000  +2.279  +45.589 +0.000  result       False   +1.078       3
   20                      DNp03_L        1     +2.236     +0.000  +2.236  +44.716 +0.000  result       False   +0.582       3

=== gf ===  (null +0.000 Hz over 3 runs)
 rank population  n_cells  stim_mean  null_mean    diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1    DNp01_R        1    +58.754     +0.000 +58.754 +1175.086 +0.000  result        True   +4.121       3
    2    DNp01_L        1    +49.742     +0.000 +49.742  +994.841 +0.000  result        True   +6.369       3
    3    DNp70_R        1     +1.626     +0.000  +1.626   +32.524 +0.000  result       False   +0.427       3

=== haltere ===  (null +0.000 Hz over 3 runs)
 rank                          population  n_cells  stim_mean  null_mean     diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1              superclass.vnc_sensory     6365   +121.056     +0.000 +121.056 +2421.120 +0.000  result       False   +0.759       3
    2 class.mechanosensory_proprioceptive     1454    +88.941     +0.000  +88.941 +1778.814 +0.000  result       False   +1.404       3
    3        class.mechanosensory_tactile     2558    +67.936     +0.000  +67.936 +1358.719 +0.000  result       False   +2.529       3
    4               superclass.cb_sensory     4868    +55.360     +0.000  +55.360 +1107.205 +0.000  result       False   +0.966       3
    5                class.mechanosensory     1733    +49.984     +0.000  +49.984  +999.679 +0.000  result       False   +1.060       3
    6        superclass.sensory_ascending      537    +44.117     +0.000  +44.117  +882.331 +0.000  result       False   +0.855       3
    7               class.unknown_sensory     1707    +23.695     +0.000  +23.695  +473.909 +0.000  result       False   +0.243       3
    8                     class.gustatory     1428    +23.579     +0.000  +23.579  +471.577 +0.000  result       False   +0.547       3
    9                     channel.ORN_all     2639    +12.328     +0.000  +12.328  +246.557 +0.000  result       False   +1.024       3
   10                     class.olfactory     2639     +9.791     +0.000   +9.791  +195.819 +0.000  result       False   +1.425       3
   11                  class.chemosensory       58     +9.574     +0.000   +9.574  +191.486 +0.000  result       False   +0.459       3
   12                             DNg51_L        2     +9.368     +0.000   +9.368  +187.369 +0.000  result       False   +1.358       3
   13                             DNa13_L        2     +9.154     +0.000   +9.154  +183.077 +0.000  result       False   +0.474       3
   14                           DNbe001_R        1     +9.096     +0.000   +9.096  +181.916 +0.000  result       False   +0.684       3
   15                           DNbe001_L        1     +8.625     +0.000   +8.625  +172.496 +0.000  result       False   +1.178       3
   16                             DNg51_R        2     +8.154     +0.000   +8.154  +163.088 +0.000  result       False   +0.928       3
   17                           DNge114_L        4     +7.982     +0.000   +7.982  +159.639 +0.000  result       False   +0.528       3
   18                           DNg02_a_L        5     +7.958     +0.000   +7.958  +159.151 +0.000  result       False   +0.637       3
   19                           DNae007_L        1     +7.704     +0.000   +7.704  +154.089 +0.000  result       False   +0.810       3
   20                             DNp43_L        1     +7.576     +0.000   +7.576  +151.514 +0.000  result       False   +0.663       3

=== leg_L ===  (null +0.000 Hz over 3 runs)
 rank                          population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1              superclass.vnc_sensory     6365    +25.942     +0.000 +25.942 +518.834 +0.000  result       False   +0.247       3
    2        class.mechanosensory_tactile     2558    +18.530     +0.000 +18.530 +370.605 +0.000  result       False   +0.141       3
    3                class.mechanosensory     1733    +15.796     +0.000 +15.796 +315.928 +0.000  result       False   +0.183       3
    4               superclass.cb_sensory     4868    +15.200     +0.000 +15.200 +304.008 +0.000  result       False   +0.144       3
    5               class.unknown_sensory     1707    +10.590     +0.000 +10.590 +211.792 +0.000  result       False   +0.324       3
    6 class.mechanosensory_proprioceptive     1454    +10.572     +0.000 +10.572 +211.440 +0.000  result       False   +0.150       3
    7                     class.gustatory     1428     +9.308     +0.000  +9.308 +186.151 +0.000  result       False   +0.156       3
    8                  class.chemosensory       58     +7.784     +0.000  +7.784 +155.678 +0.000  result       False   +0.313       3
    9                           DNge035_R        1     +7.570     +0.000  +7.570 +151.399 +0.000  result       False   +1.719       3
   10                             DNp43_L        1     +5.831     +0.000  +5.831 +116.614 +0.000  result       False   +0.365       3
   11                     channel.ORN_all     2639     +5.051     +0.000  +5.051 +101.025 +0.000  result       False   +0.574       3
   12                     class.olfactory     2639     +4.690     +0.000  +4.690  +93.796 +0.000  result       False   +0.497       3
   13                           DNge050_R        1     +4.601     +0.000  +4.601  +92.022 +0.000  result       False   +0.762       3
   14                             DNa13_L        2     +4.228     +0.000  +4.228  +84.559 +0.000  result       False   +0.125       3
   15                 class.thermosensory       25     +3.936     +0.000  +3.936  +78.727 +0.000  result       False   +0.158       3
   16                  class.hygrosensory       66     +3.818     +0.000  +3.818  +76.354 +0.000  result       False   +0.188       3
   17                             DNb08_L        2     +3.764     +0.000  +3.764  +75.283 +0.000  result       False   +0.505       3
   18                            DNg100_R        1     +3.663     +0.000  +3.663  +73.270 +0.000  result       False   +0.518       3
   19        superclass.sensory_ascending      537     +3.410     +0.000  +3.410  +68.203 +0.000  result       False   +0.361       3
   20                             DNg16_L        1     +3.341     +0.000  +3.341  +66.816 +0.000  result       False   +0.625       3

=== leg_LR ===  (null +0.000 Hz over 3 runs)
 rank                   population  n_cells  stim_mean  null_mean   diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1                    DNge035_R        1     +7.500     +0.000 +7.500 +150.005 +0.000  result       False   +1.677       3
    2                    DNge035_L        1     -6.193     +0.000 -6.193 -123.863 +0.000  result       False   +1.119       3
    3                      DNa13_L        2     +2.557     +0.000 +2.557  +51.144 +0.000  result       False   +0.152       3
    4                    DNge037_R        1     +2.535     +0.000 +2.535  +50.709 +0.000  result       False   +0.358       3
    5                    DNge049_R        1     +2.400     +0.000 +2.400  +47.998 +0.000  result       False   +0.741       3
    6                    DNge073_R        1     +2.357     +0.000 +2.357  +47.140 +0.000  result       False   +0.309       3
    7                      DNb08_L        2     +2.287     +0.000 +2.287  +45.747 +0.000  result       False   +0.833       3
    8 class.mechanosensory_tactile     2558     +2.248     +0.000 +2.248  +44.952 +0.000  result       False   +0.249       3
    9                      DNa02_L        1     +2.235     +0.000 +2.235  +44.699 +0.000  result       False   +0.132       3
   10                      DNg16_L        1     +1.999     +0.000 +1.999  +39.973 +0.000  result       False   +0.330       3
   11                      DNg37_R        1     +1.983     +0.000 +1.983  +39.661 +0.000  result       False   +0.204       3
   12                    DNge037_L        1     -1.974     +0.000 -1.974  -39.485 +0.000  result       False   +0.240       3
   13                    DNge144_R        1     -1.972     +0.000 -1.972  -39.449 +0.000  result       False   +0.407       3
   14                      DNg16_R        1     -1.967     +0.000 -1.967  -39.335 +0.000  result       False   +0.201       3
   15                      DNp43_R        1     -1.935     +0.000 -1.935  -38.703 +0.000  result       False   +0.086       3
   16                      DNp43_L        1     +1.930     +0.000 +1.930  +38.603 +0.000  result       False   +0.241       3
   17         class.mechanosensory     1733     +1.889     +0.000 +1.889  +37.790 +0.000  result       False   +0.124       3
   18                    DNge082_R        1     +1.854     +0.000 +1.854  +37.080 +0.000  result       False   +0.126       3
   19                      DNg95_L        1     +1.852     +0.000 +1.852  +37.045 +0.000  result       False   +0.085       3
   20                    DNge050_L        1     -1.852     +0.000 -1.852  -37.034 +0.000  result       False   +0.330       3

=== leg_R ===  (null +0.000 Hz over 3 runs)
 rank                          population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1              superclass.vnc_sensory     6365    +24.385     +0.000 +24.385 +487.701 +0.000  result       False   +0.294       3
    2        class.mechanosensory_tactile     2558    +16.283     +0.000 +16.283 +325.652 +0.000  result       False   +0.108       3
    3                class.mechanosensory     1733    +13.907     +0.000 +13.907 +278.139 +0.000  result       False   +0.152       3
    4               superclass.cb_sensory     4868    +13.465     +0.000 +13.465 +269.302 +0.000  result       False   +0.083       3
    5                     class.gustatory     1428    +10.079     +0.000 +10.079 +201.586 +0.000  result       False   +0.271       3
    6               class.unknown_sensory     1707     +9.606     +0.000  +9.606 +192.124 +0.000  result       False   +0.107       3
    7 class.mechanosensory_proprioceptive     1454     +9.082     +0.000  +9.082 +181.633 +0.000  result       False   +0.173       3
    8                  class.chemosensory       58     +7.089     +0.000  +7.089 +141.785 +0.000  result       False   +0.199       3
    9                           DNge035_L        1     +6.249     +0.000  +6.249 +124.975 +0.000  result       False   +1.140       3
   10                     channel.ORN_all     2639     +5.108     +0.000  +5.108 +102.157 +0.000  result       False   +0.754       3
   11                             DNp43_R        1     +4.544     +0.000  +4.544  +90.882 +0.000  result       False   +0.302       3
   12                     class.olfactory     2639     +4.185     +0.000  +4.185  +83.698 +0.000  result       False   +0.592       3
   13                             DNp43_L        1     +3.901     +0.000  +3.901  +78.011 +0.000  result       False   +0.183       3
   14                 class.thermosensory       25     +3.408     +0.000  +3.408  +68.153 +0.000  result       False   +0.249       3
   15                            DNg100_L        1     +3.315     +0.000  +3.315  +66.300 +0.000  result       False   +0.499       3
   16                  class.hygrosensory       66     +3.296     +0.000  +3.296  +65.923 +0.000  result       False   +0.092       3
   17        superclass.sensory_ascending      537     +3.147     +0.000  +3.147  +62.949 +0.000  result       False   +0.400       3
   18                           DNge050_R        1     +3.087     +0.000  +3.087  +61.747 +0.000  result       False   +0.572       3
   19                             DNg75_L        1     +3.053     +0.000  +3.053  +61.053 +0.000  result       False   +0.667       3
   20                           DNge050_L        1     +2.947     +0.000  +2.947  +58.947 +0.000  result       False   +0.833       3

=== lh_odour.apple ===  (null +0.000 Hz over 3 runs)
 rank                   population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1              class.olfactory     2639    +28.024     +0.000 +28.024 +560.486 +0.000  result       False   +0.122       3
    2              channel.ORN_all     2639    +27.859     +0.000 +27.859 +557.176 +0.000  result       False   +0.148       3
    3        superclass.cb_sensory     4868    +25.475     +0.000 +25.475 +509.502 +0.000  result       False   +0.108       3
    4          class.thermosensory       25     +8.848     +0.000  +8.848 +176.954 +0.000  result       False   +0.340       3
    5           class.hygrosensory       66     +6.893     +0.000  +6.893 +137.853 +0.000  result       False   +0.224       3
    6         class.mechanosensory     1733     +2.482     +0.000  +2.482  +49.630 +0.000  result       False   +2.435       3
    7              class.gustatory     1428     +1.229     +0.000  +1.229  +24.588 +0.000  result       False   +0.069       3
    8 superclass.sensory_ascending      537     +0.420     +0.000  +0.420   +8.397 +0.000  result       False   +0.060       3

=== lh_odour.berry ===  (null +0.000 Hz over 3 runs)
 rank                   population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1              channel.ORN_all     2639    +49.648     +0.000 +49.648 +992.962 +0.000  result       False   +0.429       3
    2              class.olfactory     2639    +49.393     +0.000 +49.393 +987.854 +0.000  result       False   +0.460       3
    3        superclass.cb_sensory     4868    +39.142     +0.000 +39.142 +782.834 +0.000  result       False   +0.396       3
    4          class.thermosensory       25     +6.054     +0.000  +6.054 +121.078 +0.000  result       False   +0.243       3
    5           class.hygrosensory       66     +4.521     +0.000  +4.521  +90.422 +0.000  result       False   +0.142       3
    6         class.mechanosensory     1733     +2.140     +0.000  +2.140  +42.795 +0.000  result       False   +1.990       3
    7 class.mechanosensory_tactile     2558     +0.877     +0.000  +0.877  +17.538 +0.000  result       False   +0.089       3
    8       superclass.vnc_sensory     6365     +0.639     +0.000  +0.639  +12.782 +0.000  result       False   +0.125       3
    9              class.gustatory     1428     +0.168     +0.000  +0.168   +3.355 +0.000  result       False   +0.146       3
   10 superclass.sensory_ascending      537     +0.158     +0.000  +0.158   +3.166 +0.000  result       False   +0.016       3

=== opto_L ===  (null +0.000 Hz over 3 runs)
 rank           population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1              DNp20_L        1    +39.610     +0.000 +39.610 +792.192 +0.000  result        True   +7.232       3
    2            DNpe005_L        1     +0.507     +0.000  +0.507  +10.144 +0.000  result       False   +0.449       3
    3 channel.JO_wind_left      335     +0.465     +0.000  +0.465   +9.295 +0.000  result       False   +0.407       3

=== opto_LR ===  (null +0.000 Hz over 3 runs)
 rank            population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1               DNp20_L        1    +39.610     +0.000 +39.610 +792.192 +0.000  result        True   +7.232       3
    2               DNp20_R        1    -37.500     +0.000 -37.500 -749.997 +0.000  result        True   +1.829       3
    3             DNpe005_L        1     +0.507     +0.000  +0.507  +10.144 +0.000  result       False   +0.449       3
    4 channel.JO_wind_right      335     -0.469     +0.000  -0.469   -9.381 +0.000  result       False   +0.412       3
    5  channel.JO_wind_left      335     +0.465     +0.000  +0.465   +9.295 +0.000  result       False   +0.407       3
    6             DNpe005_R        1     -0.280     +0.000  -0.280   -5.591 +0.000  result       False   +0.484       3
    7               DNp15_R        1     -0.254     +0.000  -0.254   -5.074 +0.000  result       False   +0.439       3

=== opto_R ===  (null +0.000 Hz over 3 runs)
 rank            population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1               DNp20_R        1    +37.500     +0.000 +37.500 +749.997 +0.000  result        True   +1.829       3
    2 channel.JO_wind_right      335     +0.469     +0.000  +0.469   +9.381 +0.000  result       False   +0.412       3
    3             DNpe005_R        1     +0.280     +0.000  +0.280   +5.591 +0.000  result       False   +0.484       3
    4               DNp15_R        1     +0.254     +0.000  +0.254   +5.074 +0.000  result       False   +0.439       3

=== power ===  (null +0.000 Hz over 3 runs)
 rank                          population  n_cells  stim_mean  null_mean     diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1              superclass.vnc_sensory     6365   +174.766     +0.000 +174.766 +3495.315 +0.000  result       False   +0.425       3
    2        class.mechanosensory_tactile     2558   +125.326     +0.000 +125.326 +2506.521 +0.000  result       False   +2.271       3
    3                     class.gustatory     1428    +95.290     +0.000  +95.290 +1905.807 +0.000  result       False   +0.668       3
    4                             DNa08_L        1    +82.238     +0.000  +82.238 +1644.758 +0.000  result       False   +6.155       3
    5               class.unknown_sensory     1707    +72.441     +0.000  +72.441 +1448.816 +0.000  result       False   +3.418       3
    6 class.mechanosensory_proprioceptive     1454    +70.769     +0.000  +70.769 +1415.376 +0.000  result       False   +2.263       3
    7                           DNg02_a_R        5    +70.285     +0.000  +70.285 +1405.701 +0.000  result       False   +4.175       3
    8                             DNa08_R        1    +70.251     +0.000  +70.251 +1405.029 +0.000  result       False  +10.639       3
    9                           DNg02_a_L        5    +64.261     +0.000  +64.261 +1285.229 +0.000  result       False   +4.293       3
   10                  class.chemosensory       58    +47.921     +0.000  +47.921  +958.424 +0.000  result       False   +2.985       3
   11                             DNp31_R        1    +45.524     +0.000  +45.524  +910.487 +0.000  result       False   +2.550       3
   12                     channel.ORN_all     2639    +42.825     +0.000  +42.825  +856.497 +0.000  result       False   +3.900       3
   13                             DNp31_L        1    +42.236     +0.000  +42.236  +844.712 +0.000  result       False   +7.712       3
   14        superclass.sensory_ascending      537    +41.223     +0.000  +41.223  +824.465 +0.000  result       False   +3.557       3
   15                     class.olfactory     2639    +36.407     +0.000  +36.407  +728.145 +0.000  result       False   +6.615       3
   16                           DNpe036_L        1    +33.939     +0.000  +33.939  +678.771 +0.000  result       False   +5.156       3
   17                             DNa13_L        2    +28.641     +0.000  +28.641  +572.811 +0.000  result       False   +3.498       3
   18                           DNpe036_R        1    +28.626     +0.000  +28.626  +572.512 +0.000  result       False   +4.889       3
   19                             DNp43_L        1    +27.018     +0.000  +27.018  +540.366 +0.000  result       False   +3.665       3
   20                           DNpe002_L        1    +23.115     +0.000  +23.115  +462.302 +0.000  result       False   +4.379       3

=== proboscis ===  (null +0.000 Hz over 3 runs)
 rank                    population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1         class.unknown_sensory     1707    +12.435     +0.000 +12.435 +248.696 +0.000  result       False   +0.125       3
    2                     DNge062_L        1    +11.981     +0.000 +11.981 +239.625 +0.000  result       False   +2.987       3
    3                     DNge080_L        1    +11.728     +0.000 +11.728 +234.563 +0.000  result       False   +0.638       3
    4                     DNge080_R        1    +11.165     +0.000 +11.165 +223.292 +0.000  result       False   +1.243       3
    5             channel.sweet_GRN      165     +7.913     +0.000  +7.913 +158.267 +0.000  result       False   +2.738       3
    6                     DNge062_R        1     +7.223     +0.000  +7.223 +144.465 +0.000  result       False   +0.484       3
    7                     DNge059_L        1     +7.140     +0.000  +7.140 +142.792 +0.000  result       False   +2.019       3
    8                     DNge077_L        1     +2.946     +0.000  +2.946  +58.929 +0.000  result       False   +1.045       3
    9         superclass.cb_sensory     4868     +2.896     +0.000  +2.896  +57.914 +0.000  result       False   +2.151       3
   10                     DNge059_R        1     +1.622     +0.000  +1.622  +32.435 +0.000  result       False   +0.437       3
   11            class.chemosensory       58     +0.934     +0.000  +0.934  +18.679 +0.000  result       False   +0.731       3
   12                     DNge042_R        1     +0.897     +0.000  +0.897  +17.936 +0.000  result       False   +0.804       3
   13                     DNge105_L        1     +0.724     +0.000  +0.724  +14.475 +0.000  result       False   +0.649       3
   14                     DNge050_L        1     +0.405     +0.000  +0.405   +8.107 +0.000  result       False   +0.702       3
   15                     DNge032_L        1     +0.393     +0.000  +0.393   +7.856 +0.000  result       False   +0.680       3
   16                       aSP22_R        1     +0.359     +0.000  +0.359   +7.173 +0.000  result       False   +0.621       3
   17 superclass.sensory_descending       12     +0.350     +0.000  +0.350   +7.005 +0.000  result       False   +0.607       3

=== steer_L ===  (null +0.000 Hz over 3 runs)
 rank                          population  n_cells  stim_mean  null_mean    diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1              superclass.vnc_sensory     6365    +82.165     +0.000 +82.165 +1643.297 +0.000  result       False   +0.242       3
    2 class.mechanosensory_proprioceptive     1454    +65.620     +0.000 +65.620 +1312.399 +0.000  result       False   +0.658       3
    3        class.mechanosensory_tactile     2558    +59.400     +0.000 +59.400 +1187.993 +0.000  result       False   +0.464       3
    4               superclass.cb_sensory     4868    +47.148     +0.000 +47.148  +942.951 +0.000  result       False   +1.049       3
    5                class.mechanosensory     1733    +44.591     +0.000 +44.591  +891.824 +0.000  result       False   +0.558       3
    6        superclass.sensory_ascending      537    +35.182     +0.000 +35.182  +703.646 +0.000  result       False   +0.263       3
    7                     class.gustatory     1428    +24.949     +0.000 +24.949  +498.977 +0.000  result       False   +0.862       3
    8               class.unknown_sensory     1707    +22.574     +0.000 +22.574  +451.479 +0.000  result       False   +0.356       3
    9                     channel.ORN_all     2639    +14.556     +0.000 +14.556  +291.118 +0.000  result       False   +0.731       3
   10                              pMP2_L        1    +13.981     +0.000 +13.981  +279.624 +0.000  result       False   +2.482       3
   11                  class.chemosensory       58    +13.518     +0.000 +13.518  +270.362 +0.000  result       False   +0.798       3
   12                              pMP2_R        1    +13.516     +0.000 +13.516  +270.324 +0.000  result       False   +2.453       3
   13                     class.olfactory     2639    +11.791     +0.000 +11.791  +235.817 +0.000  result       False   +0.847       3
   14                 class.thermosensory       25    +10.039     +0.000 +10.039  +200.774 +0.000  result       False   +0.711       3
   15                             pIP10_L        1     +9.794     +0.000  +9.794  +195.889 +0.000  result       False   +1.720       3
   16                  class.hygrosensory       66     +9.475     +0.000  +9.475  +189.509 +0.000  result       False   +0.613       3
   17                             DNp45_L        1     +9.036     +0.000  +9.036  +180.711 +0.000  result       False   +1.169       3
   18                             pIP10_R        1     +8.954     +0.000  +8.954  +179.077 +0.000  result       False   +1.356       3
   19                           DNg02_a_L        5     +8.728     +0.000  +8.728  +174.568 +0.000  result       False   +0.562       3
   20                           DNpe021_L        1     +8.532     +0.000  +8.532  +170.644 +0.000  result       False   +0.798       3

=== steer_LR ===  (null +0.000 Hz over 3 runs)
 rank                          population  n_cells  stim_mean  null_mean   diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1 class.mechanosensory_proprioceptive     1454     +7.588     +0.000 +7.588 +151.769 +0.000  result       False   +1.361       3
    2                             DNg04_R        2     -6.944     +0.000 -6.944 -138.876 +0.000  result       False   +0.430       3
    3                             DNg04_L        2     +5.726     +0.000 +5.726 +114.528 +0.000  result       False   +0.104       3
    4                             DNa15_R        1     -5.157     +0.000 -5.157 -103.131 +0.000  result       False   +0.356       3
    5                     DNp51,DNpe019_L        2     +5.084     +0.000 +5.084 +101.671 +0.000  result       False   +0.769       3
    6                             DNg15_R        1     +4.900     +0.000 +4.900  +97.995 +0.000  result       False   +0.257       3
    7                             DNa15_L        1     +4.766     +0.000 +4.766  +95.324 +0.000  result       False   +0.631       3
    8                     DNp51,DNpe019_R        2     -4.707     +0.000 -4.707  -94.145 +0.000  result       False   +0.264       3
    9                             DNa04_L        1     +4.452     +0.000 +4.452  +89.032 +0.000  result       False   +0.695       3
   10                           DNae004_R        1     -4.353     +0.000 -4.353  -87.053 +0.000  result       False   +0.431       3
   11                             DNa05_R        1     -4.328     +0.000 -4.328  -86.565 +0.000  result       False   +0.707       3
   12                             DNg32_R        1     +4.278     +0.000 +4.278  +85.565 +0.000  result       False   +0.170       3
   13                             DNa04_R        1     -4.196     +0.000 -4.196  -83.919 +0.000  result       False   +0.839       3
   14                             DNg32_L        1     -3.852     +0.000 -3.852  -77.036 +0.000  result       False   +0.208       3
   15                             DNp03_L        1     -3.840     +0.000 -3.840  -76.801 +0.000  result       False   +0.652       3
   16                             DNa03_R        1     -3.800     +0.000 -3.800  -76.000 +0.000  result       False   +1.370       3
   17                             DNp18_L        1     +3.766     +0.000 +3.766  +75.318 +0.000  result       False   +0.358       3
   18              superclass.vnc_sensory     6365     +3.743     +0.000 +3.743  +74.868 +0.000  result       False   +0.290       3
   19                           DNge140_R        1     +3.734     +0.000 +3.734  +74.683 +0.000  result       False   +0.650       3
   20                             DNp26_R        1     +3.242     +0.000 +3.242  +64.837 +0.000  result       False   +0.472       3

=== steer_R ===  (null +0.000 Hz over 3 runs)
 rank                          population  n_cells  stim_mean  null_mean    diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1              superclass.vnc_sensory     6365    +78.421     +0.000 +78.421 +1568.429 +0.000  result       False   +0.060       3
    2        class.mechanosensory_tactile     2558    +60.762     +0.000 +60.762 +1215.236 +0.000  result       False   +0.380       3
    3 class.mechanosensory_proprioceptive     1454    +58.032     +0.000 +58.032 +1160.630 +0.000  result       False   +0.780       3
    4               superclass.cb_sensory     4868    +47.831     +0.000 +47.831  +956.628 +0.000  result       False   +1.123       3
    5                class.mechanosensory     1733    +42.580     +0.000 +42.580  +851.607 +0.000  result       False   +0.580       3
    6        superclass.sensory_ascending      537    +33.608     +0.000 +33.608  +672.168 +0.000  result       False   +0.792       3
    7                     class.gustatory     1428    +25.710     +0.000 +25.710  +514.207 +0.000  result       False   +0.893       3
    8               class.unknown_sensory     1707    +21.388     +0.000 +21.388  +427.755 +0.000  result       False   +0.481       3
    9                     channel.ORN_all     2639    +15.560     +0.000 +15.560  +311.193 +0.000  result       False   +1.048       3
   10                              pMP2_L        1    +14.085     +0.000 +14.085  +281.694 +0.000  result       False   +2.638       3
   11                              pMP2_R        1    +13.246     +0.000 +13.246  +264.925 +0.000  result       False   +2.401       3
   12                     class.olfactory     2639    +12.060     +0.000 +12.060  +241.201 +0.000  result       False   +0.551       3
   13                  class.chemosensory       58    +11.760     +0.000 +11.760  +235.201 +0.000  result       False   +1.013       3
   14                             pIP10_L        1     +9.704     +0.000  +9.704  +194.076 +0.000  result       False   +2.058       3
   15                             pIP10_R        1     +9.504     +0.000  +9.504  +190.079 +0.000  result       False   +1.248       3
   16                 class.thermosensory       25     +9.295     +0.000  +9.295  +185.905 +0.000  result       False   +0.987       3
   17                           DNbe001_L        1     +9.176     +0.000  +9.176  +183.510 +0.000  result       False   +1.265       3
   18                             DNp45_R        1     +9.055     +0.000  +9.055  +181.093 +0.000  result       False   +0.730       3
   19                             DNp42_R        1     +8.837     +0.000  +8.837  +176.740 +0.000  result       False   +2.807       3
   20                  class.hygrosensory       66     +8.407     +0.000  +8.407  +168.147 +0.000  result       False   +0.294       3

=== ttm ===  (null +0.000 Hz over 3 runs)
 rank      population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1         DNp01_R        1    +34.438     +0.000 +34.438 +688.769 +0.000  result       False   +2.439       3
    2 class.gustatory     1428    +28.283     +0.000 +28.283 +565.664 +0.000  result       False   +2.325       3
    3         DNp06_R        1    +25.044     +0.000 +25.044 +500.882 +0.000  result       False   +3.454       3
    4         DNp06_L        1    +12.319     +0.000 +12.319 +246.372 +0.000  result       False   +1.809       3
    5         DNp01_L        1    +11.915     +0.000 +11.915 +238.305 +0.000  result       False   +2.713       3
    6         DNp02_R        1    +11.240     +0.000 +11.240 +224.802 +0.000  result       False   +1.179       3
    7       DNge130_R        1    +11.033     +0.000 +11.033 +220.670 +0.000  result       False   +3.731       3
    8         DNp70_R        1    +10.679     +0.000 +10.679 +213.573 +0.000  result       False   +1.378       3
    9         DNa11_L        1    +10.406     +0.000 +10.406 +208.125 +0.000  result       False   +3.485       3
   10         DNp14_R        1     +9.979     +0.000  +9.979 +199.576 +0.000  result       False   +2.190       3
   11       DNde003_L        2     +9.787     +0.000  +9.787 +195.744 +0.000  result       False   +4.753       3
   12         DNp43_R        1     +8.387     +0.000  +8.387 +167.744 +0.000  result       False   +1.312       3
   13         DNp43_L        1     +7.960     +0.000  +7.960 +159.203 +0.000  result       False   +2.141       3
   14       DNge124_R        1     +7.342     +0.000  +7.342 +146.842 +0.000  result       False   +3.769       3
   15         DNa03_L        1     +7.271     +0.000  +7.271 +145.414 +0.000  result       False   +3.223       3
   16       DNde003_R        2     +7.079     +0.000  +7.079 +141.585 +0.000  result       False   +0.966       3
   17         DNg75_L        1     +7.027     +0.000  +7.027 +140.542 +0.000  result       False   +0.441       3
   18       DNae007_L        1     +6.635     +0.000  +6.635 +132.707 +0.000  result       False   +3.300       3
   19         DNp14_L        1     +6.466     +0.000  +6.466 +129.317 +0.000  result       False   +1.438       3
   20       DNge037_L        1     +6.439     +0.000  +6.439 +128.781 +0.000  result       False   +1.578       3

=== turn_L ===  (null +0.000 Hz over 3 runs)
 rank      population  n_cells  stim_mean  null_mean     diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1         DNa02_L        1   +117.188     +0.000 +117.188 +2343.752 +0.000  result        True   +6.168       3
    2       DNae007_L        1     +5.473     +0.000   +5.473  +109.466 +0.000  result       False   +0.782       3
    3       DNde003_L        2     +3.864     +0.000   +3.864   +77.281 +0.000  result       False   +1.643       3
    4       DNpe024_L        1     +2.935     +0.000   +2.935   +58.703 +0.000  result       False   +0.798       3
    5       DNg09_a_R        3     +2.434     +0.000   +2.434   +48.671 +0.000  result       False   +1.001       3
    6         DNa11_L        1     +1.632     +0.000   +1.632   +32.632 +0.000  result       False   +0.319       3
    7         DNp09_L        1     +1.183     +0.000   +1.183   +23.655 +0.000  result       False   +1.201       3
    8         DNg75_L        1     +0.751     +0.000   +0.751   +15.011 +0.000  result       False   +0.940       3
    9         DNp52_R        1     +0.746     +0.000   +0.746   +14.929 +0.000  result       False   +1.293       3
   10         DNp52_L        1     +0.739     +0.000   +0.739   +14.781 +0.000  result       False   +1.280       3
   11 class.olfactory     2639     +0.360     +0.000   +0.360    +7.204 +0.000  result       False   +0.624       3
   12         DNa03_L        1     +0.079     +0.000   +0.079    +1.589 +0.000    null       False   +0.138       3

=== turn_LR ===  (null +0.000 Hz over 3 runs)
 rank      population  n_cells  stim_mean  null_mean     diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1         DNa02_L        1   +117.188     +0.000 +117.188 +2343.752 +0.000  result        True   +6.168       3
    2         DNa02_R        1   -106.843     +0.000 -106.843 -2136.852 +0.000  result        True  +17.598       3
    3       DNae007_L        1     +5.473     +0.000   +5.473  +109.466 +0.000  result       False   +0.782       3
    4       DNpe024_R        1     -4.861     +0.000   -4.861   -97.219 +0.000  result       False   +1.331       3
    5       DNae007_R        1     -4.105     +0.000   -4.105   -82.091 +0.000  result       False   +0.678       3
    6       DNde003_R        2     -3.995     +0.000   -3.995   -79.908 +0.000  result       False   +0.814       3
    7       DNde003_L        2     +3.864     +0.000   +3.864   +77.281 +0.000  result       False   +1.643       3
    8       DNpe024_L        1     +2.935     +0.000   +2.935   +58.703 +0.000  result       False   +0.798       3
    9       DNg09_a_R        3     +2.434     +0.000   +2.434   +48.671 +0.000  result       False   +1.001       3
   10         DNa11_L        1     +1.632     +0.000   +1.632   +32.632 +0.000  result       False   +0.319       3
   11         DNp09_L        1     +1.183     +0.000   +1.183   +23.655 +0.000  result       False   +1.201       3
   12         DNg75_L        1     +0.751     +0.000   +0.751   +15.011 +0.000  result       False   +0.940       3
   13         DNp52_R        1     +0.746     +0.000   +0.746   +14.929 +0.000  result       False   +1.293       3
   14         DNp52_L        1     +0.739     +0.000   +0.739   +14.781 +0.000  result       False   +1.280       3
   15         DNa03_R        1     -0.481     +0.000   -0.481    -9.618 +0.000  result       False   +0.833       3
   16 class.olfactory     2639     +0.360     +0.000   +0.360    +7.204 +0.000  result       False   +0.624       3
   17         DNa03_L        1     +0.079     +0.000   +0.079    +1.589 +0.000    null       False   +0.138       3

=== turn_R ===  (null +0.000 Hz over 3 runs)
 rank population  n_cells  stim_mean  null_mean     diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1    DNa02_R        1   +106.843     +0.000 +106.843 +2136.852 +0.000  result        True  +17.598       3
    2  DNpe024_R        1     +4.861     +0.000   +4.861   +97.219 +0.000  result       False   +1.331       3
    3  DNae007_R        1     +4.105     +0.000   +4.105   +82.091 +0.000  result       False   +0.678       3
    4  DNde003_R        2     +3.995     +0.000   +3.995   +79.908 +0.000  result       False   +0.814       3
    5    DNa03_R        1     +0.481     +0.000   +0.481    +9.618 +0.000  result       False   +0.833       3

=== wind_contra_L ===  (null +0.000 Hz over 3 runs)
 rank            population  n_cells  stim_mean  null_mean     diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1  class.mechanosensory     1733   +141.898     +0.000 +141.898 +2837.969 +0.000  result       False   +3.049       3
    2 superclass.cb_sensory     4868   +137.765     +0.000 +137.765 +2755.296 +0.000  result       False   +4.390       3
    3               DNp33_L        1    +56.774     +0.000  +56.774 +1135.472 +0.000  result        True   +0.370       3
    4               DNg99_L        1    +52.772     +0.000  +52.772 +1055.437 +0.000  result        True  +12.558       3
    5 channel.JO_wind_right      335    +23.586     +0.000  +23.586  +471.726 +0.000  result       False   +3.539       3
    6               DNg07_R        8    +23.187     +0.000  +23.187  +463.747 +0.000  result       False   +0.749       3
    7              DNg110_L        3    +16.414     +0.000  +16.414  +328.273 +0.000  result       False   +0.996       3
    8       class.olfactory     2639    +12.368     +0.000  +12.368  +247.353 +0.000  result       False   +1.377       3
    9       channel.ORN_all     2639    +11.963     +0.000  +11.963  +239.266 +0.000  result       False   +1.777       3
   10   class.thermosensory       25     +8.946     +0.000   +8.946  +178.921 +0.000  result       False   +0.308       3
   11           (untyped)_L        3     +8.477     +0.000   +8.477  +169.542 +0.000  result       False   +2.292       3
   12             DNge145_L        2     +7.856     +0.000   +7.856  +157.110 +0.000  result       False   +1.397       3
   13               DNp47_L        1     +7.486     +0.000   +7.486  +149.712 +0.000  result       False   +1.672       3
   14               DNp47_R        1     +7.370     +0.000   +7.370  +147.402 +0.000  result       False   +2.112       3
   15    class.hygrosensory       66     +6.532     +0.000   +6.532  +130.649 +0.000  result       False   +1.325       3
   16               aSP22_L        1     +6.521     +0.000   +6.521  +130.416 +0.000  result       False   +1.810       3
   17               DNa13_L        2     +5.788     +0.000   +5.788  +115.752 +0.000  result       False   +0.989       3
   18               DNa07_L        1     +5.228     +0.000   +5.228  +104.557 +0.000  result       False   +1.642       3
   19               DNb05_L        1     +4.751     +0.000   +4.751   +95.016 +0.000  result       False   +1.367       3
   20       class.gustatory     1428     +4.680     +0.000   +4.680   +93.604 +0.000  result       False   +0.585       3

=== wind_contra_LR ===  (null +0.000 Hz over 3 runs)
 rank            population  n_cells  stim_mean  null_mean     diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1  class.mechanosensory     1733   +125.648     +0.000 +125.648 +2512.953 +0.000  result       False   +4.807       3
    2 superclass.cb_sensory     4868   +122.860     +0.000 +122.860 +2457.192 +0.000  result       False   +3.578       3
    3               DNg99_R        1    -58.984     +0.000  -58.984 -1179.680 +0.000  result        True   +5.172       3
    4               DNp33_R        1    -57.219     +0.000  -57.219 -1144.385 +0.000  result        True   +6.367       3
    5               DNp33_L        1    +56.774     +0.000  +56.774 +1135.472 +0.000  result        True   +0.370       3
    6               DNg99_L        1    +52.772     +0.000  +52.772 +1055.437 +0.000  result        True  +12.558       3
    7               DNg07_L        8    -24.357     +0.000  -24.357  -487.149 +0.000  result       False   +0.340       3
    8 channel.JO_wind_right      335    +23.586     +0.000  +23.586  +471.726 +0.000  result       False   +3.539       3
    9              DNg110_L        3    +16.414     +0.000  +16.414  +328.273 +0.000  result       False   +0.996       3
   10              DNg110_R        3    -16.208     +0.000  -16.208  -324.155 +0.000  result       False   +2.962       3
   11  channel.JO_wind_left      335    -16.167     +0.000  -16.167  -323.344 +0.000  result       False   +1.316       3
   12               DNg07_R        8    +15.116     +0.000  +15.116  +302.330 +0.000  result       False   +0.559       3
   13             DNge111_R        3    -11.698     +0.000  -11.698  -233.955 +0.000  result       False   +1.593       3
   14               DNb05_R        1    -10.579     +0.000  -10.579  -211.582 +0.000  result       False   +1.775       3
   15           (untyped)_L        3     +8.477     +0.000   +8.477  +169.542 +0.000  result       False   +2.292       3
   16    class.hygrosensory       66     -8.374     +0.000   -8.374  -167.484 +0.000  result       False   +0.977       3
   17             DNge145_L        2     +7.856     +0.000   +7.856  +157.110 +0.000  result       False   +1.397       3
   18             DNge145_R        2     -7.087     +0.000   -7.087  -141.746 +0.000  result       False   +1.092       3
   19               aSP22_L        1     +6.521     +0.000   +6.521  +130.416 +0.000  result       False   +1.810       3
   20       channel.ORN_all     2639     -5.622     +0.000   -5.622  -112.431 +0.000  result       False   +0.784       3

=== wind_contra_R ===  (null +0.000 Hz over 3 runs)
 rank            population  n_cells  stim_mean  null_mean    diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1               DNg99_R        1    +58.984     +0.000 +58.984 +1179.680 +0.000  result        True   +5.172       3
    2               DNp33_R        1    +57.219     +0.000 +57.219 +1144.385 +0.000  result        True   +6.367       3
    3               DNg07_L        8    +24.357     +0.000 +24.357  +487.149 +0.000  result       False   +0.340       3
    4       class.olfactory     2639    +17.714     +0.000 +17.714  +354.282 +0.000  result       False   +1.641       3
    5       channel.ORN_all     2639    +17.585     +0.000 +17.585  +351.697 +0.000  result       False   +2.304       3
    6  class.mechanosensory     1733    +16.251     +0.000 +16.251  +325.016 +0.000  result       False   +2.250       3
    7              DNg110_R        3    +16.208     +0.000 +16.208  +324.155 +0.000  result       False   +2.962       3
    8  channel.JO_wind_left      335    +16.167     +0.000 +16.167  +323.344 +0.000  result       False   +1.316       3
    9    class.hygrosensory       66    +14.907     +0.000 +14.907  +298.132 +0.000  result       False   +0.351       3
   10 superclass.cb_sensory     4868    +14.905     +0.000 +14.905  +298.104 +0.000  result       False   +0.844       3
   11             DNge111_R        3    +11.698     +0.000 +11.698  +233.955 +0.000  result       False   +1.593       3
   12   class.thermosensory       25    +11.647     +0.000 +11.647  +232.934 +0.000  result       False   +0.459       3
   13               DNb05_R        1    +10.579     +0.000 +10.579  +211.582 +0.000  result       False   +1.775       3
   14               DNp47_R        1     +8.921     +0.000  +8.921  +178.418 +0.000  result       False   +2.764       3
   15               DNg07_R        8     +8.071     +0.000  +8.071  +161.417 +0.000  result       False   +0.221       3
   16             DNge145_R        2     +7.087     +0.000  +7.087  +141.746 +0.000  result       False   +1.092       3
   17             DNpe002_L        1     +5.713     +0.000  +5.713  +114.258 +0.000  result       False   +2.708       3
   18               aSP22_R        1     +5.406     +0.000  +5.406  +108.126 +0.000  result       False   +0.189       3
   19               DNa13_L        2     +5.369     +0.000  +5.369  +107.379 +0.000  result       False   +0.663       3
   20             DNae005_L        1     +4.012     +0.000  +4.012   +80.245 +0.000  result       False   +0.684       3

=== wind_ipsi_L ===  (null +0.000 Hz over 3 runs)
 rank                   population  n_cells  stim_mean  null_mean     diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1        superclass.cb_sensory     4868   +101.051     +0.000 +101.051 +2021.023 +0.000  result       False   +1.361       3
    2         class.mechanosensory     1733    +90.073     +0.000  +90.073 +1801.466 +0.000  result       False   +4.498       3
    3                    DNg05_a_L        1    +27.003     +0.000  +27.003  +540.052 +0.000  result        True   +2.157       3
    4                      DNp19_L        1    +24.462     +0.000  +24.462  +489.232 +0.000  result        True   +3.280       3
    5                    DNge175_L        1    +23.901     +0.000  +23.901  +478.028 +0.000  result        True   +2.810       3
    6                    DNge016_L        1    +23.025     +0.000  +23.025  +460.507 +0.000  result        True   +3.570       3
    7                      DNp18_L        1    +20.354     +0.000  +20.354  +407.079 +0.000  result        True   +1.538       3
    8         channel.JO_wind_left      335    +18.085     +0.000  +18.085  +361.696 +0.000  result       False   +0.771       3
    9              class.olfactory     2639     +9.938     +0.000   +9.938  +198.751 +0.000  result       False   +1.243       3
   10              channel.ORN_all     2639     +9.606     +0.000   +9.606  +192.125 +0.000  result       False   +1.074       3
   11          class.thermosensory       25     +8.088     +0.000   +8.088  +161.752 +0.000  result       False   +0.606       3
   12       superclass.vnc_sensory     6365     +7.242     +0.000   +7.242  +144.836 +0.000  result       False   +0.482       3
   13 class.mechanosensory_tactile     2558     +7.197     +0.000   +7.197  +143.945 +0.000  result       False   +0.063       3
   14           class.hygrosensory       66     +6.743     +0.000   +6.743  +134.866 +0.000  result       False   +0.154       3
   15        class.unknown_sensory     1707     +6.334     +0.000   +6.334  +126.677 +0.000  result       False   +0.921       3
   16              DNp51,DNpe019_L        2     +5.992     +0.000   +5.992  +119.831 +0.000  result       False   +1.077       3
   17                    DNae010_L        1     +5.089     +0.000   +5.089  +101.785 +0.000  result       False   +0.472       3
   18                      DNp63_R        1     +4.298     +0.000   +4.298   +85.968 +0.000  result       False   +0.678       3
   19                    DNpe055_L        1     +4.144     +0.000   +4.144   +82.879 +0.000  result       False   +0.869       3
   20                    DNge030_L        1     +4.043     +0.000   +4.043   +80.855 +0.000  result       False   +1.107       3

=== wind_ipsi_LR ===  (null +0.000 Hz over 3 runs)
 rank            population  n_cells  stim_mean  null_mean    diff  z_floor      p verdict  self_drive  stim_sd  n_runs
    1 superclass.cb_sensory     4868    +27.807     +0.000 +27.807 +556.141 +0.000  result       False   +1.577       3
    2             DNg05_a_L        1    +27.003     +0.000 +27.003 +540.052 +0.000  result        True   +2.157       3
    3               DNp19_L        1    +24.462     +0.000 +24.462 +489.232 +0.000  result        True   +3.280       3
    4             DNg05_a_R        1    -24.005     +0.000 -24.005 -480.097 +0.000  result        True   +2.715       3
    5             DNge175_L        1    +23.901     +0.000 +23.901 +478.028 +0.000  result        True   +2.810       3
    6             DNge175_R        1    -23.161     +0.000 -23.161 -463.212 +0.000  result        True   +2.374       3
    7             DNge016_L        1    +23.025     +0.000 +23.025 +460.507 +0.000  result        True   +3.570       3
    8             DNge016_R        1    -22.638     +0.000 -22.638 -452.756 +0.000  result        True   +3.841       3
    9               DNp19_R        1    -22.351     +0.000 -22.351 -447.018 +0.000  result        True   +6.169       3
   10  class.mechanosensory     1733    +20.937     +0.000 +20.937 +418.747 +0.000  result       False   +5.555       3
   11               DNp18_R        1    -20.623     +0.000 -20.623 -412.455 +0.000  result        True   +1.624       3
   12               DNp18_L        1    +20.354     +0.000 +20.354 +407.079 +0.000  result        True   +1.538       3
   13  channel.JO_wind_left      335    +16.615     +0.000 +16.615 +332.293 +0.000  result       False   +0.183       3
   14 channel.JO_wind_right      335     -7.443     +0.000  -7.443 -148.866 +0.000  result       False   +0.824       3
   15       DNp51,DNpe019_L        2     +5.992     +0.000  +5.992 +119.831 +0.000  result       False   +1.077       3
   16             DNae010_L        1     +5.089     +0.000  +5.089 +101.785 +0.000  result       False   +0.472       3
   17             DNae010_R        1     -5.080     +0.000  -5.080 -101.593 +0.000  result       False   +0.208       3
   18               DNp26_L        1     -4.518     +0.000  -4.518  -90.352 +0.000  result       False   +0.244       3
   19             DNpe055_R        1     -4.306     +0.000  -4.306  -86.118 +0.000  result       False   +0.991       3
   20       DNp51,DNpe019_R        2     -4.178     +0.000  -4.178  -83.566 +0.000  result       False   +0.321       3

=== wind_ipsi_R ===  (null +0.000 Hz over 3 runs)
 rank                   population  n_cells  stim_mean  null_mean    diff   z_floor      p verdict  self_drive  stim_sd  n_runs
    1        superclass.cb_sensory     4868    +73.244     +0.000 +73.244 +1464.882 +0.000  result       False   +0.645       3
    2         class.mechanosensory     1733    +69.136     +0.000 +69.136 +1382.718 +0.000  result       False   +1.137       3
    3                    DNg05_a_R        1    +24.005     +0.000 +24.005  +480.097 +0.000  result        True   +2.715       3
    4                    DNge175_R        1    +23.161     +0.000 +23.161  +463.212 +0.000  result        True   +2.374       3
    5                    DNge016_R        1    +22.638     +0.000 +22.638  +452.756 +0.000  result        True   +3.841       3
    6                      DNp19_R        1    +22.351     +0.000 +22.351  +447.018 +0.000  result        True   +6.169       3
    7                      DNp18_R        1    +20.623     +0.000 +20.623  +412.455 +0.000  result        True   +1.624       3
    8              channel.ORN_all     2639    +11.399     +0.000 +11.399  +227.984 +0.000  result       False   +0.400       3
    9              class.olfactory     2639    +11.238     +0.000 +11.238  +224.761 +0.000  result       False   +0.480       3
   10           class.hygrosensory       66     +8.326     +0.000  +8.326  +166.529 +0.000  result       False   +0.446       3
   11        channel.JO_wind_right      335     +7.779     +0.000  +7.779  +155.571 +0.000  result       False   +0.429       3
   12          class.thermosensory       25     +7.552     +0.000  +7.552  +151.044 +0.000  result       False   +0.018       3
   13 class.mechanosensory_tactile     2558     +5.381     +0.000  +5.381  +107.629 +0.000  result       False   +0.221       3
   14       superclass.vnc_sensory     6365     +5.311     +0.000  +5.311  +106.229 +0.000  result       False   +0.131       3
   15                    DNae010_R        1     +5.080     +0.000  +5.080  +101.593 +0.000  result       False   +0.208       3
   16                      DNp26_L        1     +4.518     +0.000  +4.518   +90.352 +0.000  result       False   +0.244       3
   17                      DNp63_L        1     +4.385     +0.000  +4.385   +87.697 +0.000  result       False   +0.478       3
   18                    DNpe055_R        1     +4.306     +0.000  +4.306   +86.118 +0.000  result       False   +0.991       3
   19              DNp51,DNpe019_R        2     +4.178     +0.000  +4.178   +83.566 +0.000  result       False   +0.321       3
   20                      DNg07_L        8     +4.023     +0.000  +4.023   +80.458 +0.000  result       False   +0.672       3
```


### 5.1 What each readout is made of

The `self_drive` column only makes sense with the cell sets in front of it (`flyverse/motor.py`, unchanged by this
round):

| readout | cells | types |
|---|---|---|
| `fwd_dn` | 10 | DNa01, DNa03, DNa04, DNb01, DNp09 |
| `back_dn` | 4 | MDN |
| `turn_L` / `turn_R` | 1 / 1 | DNa02 (one cell per side) |
| `opto_L` / `opto_R` | 3 / 3 | DNp20, HSE, HSN |
| `wind_ipsi_L` / `_R` | 5 / 5 | DNg05_a, DNge016, DNge175, DNp18, DNp19 |
| `wind_contra_L` / `_R` | 2 / 2 | DNg99, DNp33 |
| `gf` | 2 | DNp01 |
| `ttm` | 2 | TTMn |
| `proboscis` | 2 | MN9 |
| `power` | 24 | DLMn, DVMn |
| `steer_L` / `_R` | 16 / 16 | b1/b2/b3, hg1-4, i1/i2, iii1/iii3, ps1/ps2, tp1/tp2 MNs |
| `haltere` | 16 | MNhm03/42/43, hDVM, hi1, hi2, hiii2 |
| `leg_L` / `leg_R` | 192 / 189 | front / mid / hind leg motor neurons |
| `lh_odour.apple` / `.berry` | 23 / 14 | the lateral-horn channels of `body.LH_ODOUR_TYPES` |

So six of the 21 readouts are *made of descending neurons* (`fwd_dn`, `back_dn`, `turn_*`, `opto_*`, `wind_*`, `gf`),
and every DN inside them appears at rank 1 of its own readout for a trivial reason. The `top_upstream` entry of
`summary.top_population_per_readout` is the non-trivial answer for those rows.

---

## 6. What the atlas says

### 6.1 The measured descending-neuron motor map (self-drive removed)

| readout | strongest population that is **not** part of the readout | diff, Hz |
|---|---|---|
| `gf` | DNp70_R | **+1.63** |
| `ttm` (TTMn) | DNp01_R (+34.44), DNp06_R (+25.04), DNp06_L (+12.32), DNp01_L (+11.92), DNp02_R (+11.24) | +34.44 |
| `turn_L` (DNa02_L) | DNae007_L | +5.47 |
| `turn_R` (DNa02_R) | DNpe024_R | +4.86 |
| `opto_L` (DNp20/HSE/HSN) | DNpe005_L | +0.51 |
| `opto_R` | `channel.JO_wind_right` | +0.47 |
| `back_dn` (MDN) | `superclass.cb_sensory` (+55.8); strongest DN: DNpe023_L **+24.43**, DNpe023_R +23.29, DNp43_L +19.98 | +55.84 |
| `fwd_dn` | `class.gustatory` (+10.61); strongest DN: DNae007_L +3.09, DNge130_R +2.46 | +10.61 |
| `leg_LR` | DNge035_R **+7.50** / DNge035_L **-6.19**, then DNa13_L +2.56, DNge037_R +2.54, DNge049_R +2.40, DNa02_L +2.24 | +7.50 |
| `power` | `superclass.vnc_sensory` (+174.8); strongest DN: DNa08_L **+82.24**, DNg02_a_R +70.29, DNa08_R +70.25, DNp31_R +45.52 | +174.77 |
| `proboscis` (MN9) | `class.unknown_sensory` +12.43, DNge062_L +11.98, DNge080_L +11.73, DNge080_R +11.17, `channel.sweet_GRN` **+7.91** | +12.43 |
| `steer_LR` | `class.mechanosensory_proprioceptive` +7.59, DNg04_R -6.94 / DNg04_L +5.73, DNa15_R -5.16 | +7.59 |
| `lh_odour.apple` / `.berry` | `class.olfactory` = `channel.ORN_all` (+28.02 / +49.65) | +49.65 |

Four readings worth keeping:

1. **Nothing reaches the giant fibre by stimulation.** Only three populations of 981 move `gf` at all: DNp01 itself
   (both sides, self-drive) and **DNp70_R at +1.63 Hz**, and that +1.63 is at DNp70's **full** weight onto DNp01.
   The round-3 GF damping (`^(SAD073|GNG300|DNp70|CL367|PVLP010)$ -> ^DNp01$`, x0.3) was **retired in round 5**
   (`docs/audits/anti_runaway.md`) and is not in force here: this round's own provenance records
   `type_path_gain = [["^(LC4|LPLC2)$", "^DNp01$", 3.0]]` and nothing else
   (`provenance.model.lif.type_path_gain`, identical in all three of `out/interp/atlas/*.json`). So the one
   population that can reach the GF from outside is a type the *previous* defaults singled out, undamped in this
   model. Neither LC4 nor LPLC2 appears -- they are
   visual projection neurons, not in this population list; the atlas over DNs and sensory classes says only that no
   *descending* or *sensory* population drives the GF. DNp01 does drive the escape motor neurons it should:
   `ttm` rank 1 is DNp01_R +34.44 with DNp06 beside it.
2. **`DNge035` is the strongest lateralised leg driver in this model**, at +7.50 / -6.19 Hz of leg L-R, three times
   DNa02's +2.23 / -1.45, with run sd 1.68 / 1.12 over three runs -- and it is *contralateral* where DNa02 is
   ipsilateral: DNge035_R puts the **left** legs at 7.57 Hz and leaves the right at 0.07 (verdict `null`), DNge035_L
   the mirror (right 6.25, left 0.06), while DNa02_L gives left 2.73 / right 0.50. It is a measurement, not a
   recommendation: the
   project rule stands, and nothing in `body.py` changes on the strength of a stimulation screen. It is recorded here
   because `body.py`'s turning term currently reads DNa02 and the leg MNs, and any future data-driven revision of that
   readout would have to account for DNge035, DNa13, DNge037, DNge049, DNge073 (all above +2.3 Hz).
3. **The wing-power readout is dominated by sensory volleys, not by DNs.** `superclass.vnc_sensory` at 150 Hz puts the
   DLMn/DVMn set at +174.8 Hz; the strongest single DN type is DNa08 at +82.2. Section 6.3 is the dose caveat that
   goes with this: 6,365 cells against 1.
4. **The `opto_*` readout has exactly one driver.** DNp20 moves it (39.6 / 37.5 Hz, self-drive: DNp20 is in the group),
   and the next entry is 0.51 Hz. HSE/HSN, the other two members, are visual and not in this population list. The
   optomotor readout of `body.py` is therefore a DNp20 readout plus two cells nothing in this sweep can drive.

### 6.2 The photoreceptors read null because their output is pruned out of the LIF

`channel.photoreceptors` (6,091 cells at 150 Hz), `class.visual_*` and `superclass.ol_sensory_*` move **no motor
readout at all** -- every `diff` exactly 0.000. The tool says why: `summary.populations_whose_output_is_pruned`

```
{"class.visual_L": 0.813, "class.visual_R": 0.939, "class.visual_?": 0.869,
 "superclass.ol_sensory_L": 0.694, "superclass.ol_sensory_R": 0.690, "superclass.ol_sensory_?": 0.869,
 "channel.photoreceptors": 0.868}
```

87 % of the photoreceptors' outgoing |W| lands on the optic rate units, which `FlyBrain` freezes and prunes out of the
LIF matrix (`prune_frozen`, the shipped default). Photoreceptor spikes therefore have almost nowhere to go: the
photoreceptor -> lamina -> medulla path exists only inside the rate-model `OpticLobe`, driven by the ray tracer, not by
Poisson spikes. This is the same structural fact `docs/audits/object_sweep.md` works with from the other end, stated as
a property of the stimulation experiment: **the atlas cannot reach the visual system through `stimulate`, and a null
row for an optic population is an artefact of the compiled model, not a claim about the circuit.** A visual atlas needs
either `--modules` without `prune_frozen`, or a vision context (`fb.vision`), not a pulse.

### 6.3 The dose caveat -- sensory classes are not comparable with single DNs

A sensory-class population is 58-6,365 cells; a DN population is 1-12. Both get 150 Hz. The comparison *within* the
sensory rows (vnc_sensory +174.8 > tactile +125.3 > gustatory +95.3 on `power`) and *within* the DN rows is fair; the
comparison across the two is a dose comparison, exactly the uncontrolled-dose problem
`docs/audits/receptor_integration.md` records for the optic/Brain receptor halves. Every row carries `n_cells`, and
`--populations` with an explicit list is how a dose-matched comparison would be run. No conclusion in this document
ranks a sensory class against a DN.

### 6.4 347 of 981 populations move nothing

`summary.n_populations_that_moved_nothing = 347` (the list is in `summary.populations_that_moved_nothing`, first 200):
DNb02, DNb04, DNb07, DNb09, DNbe004, DNbe005, DNc01, DNc02, DNd02, DNd03, DNd04, DNde001, DNde006, DNde007, DNg01_a,
... -- and the visual populations of 6.2. "Moves nothing" is the strict reading, `max |diff|` **exactly 0** across all
27 readouts; recomputed from `out/interp/atlas/dn_sensory.json`, exactly 347 populations qualify. It is not the
complement of the 578 that reach a `result`: a further **56** populations move a readout by a non-zero amount that
never reaches verdict `result` (578 + 347 + 56 = 981). A DN that fires at 150 Hz for 400 ms and leaves all 21 motor groups exactly
at their null value either projects to VNC interneurons the readout does not pool, or its targets do not reach
threshold at this dose. The atlas' answer here is a *fact about the readouts*, not about the neurons: `flyverse/motor.py`
pools 532 cells out of 167,106, and a DN that drives its own premotor set without moving those 532 reads null. The
follow-up this points at is `--pattern` (record every VNC type through `screen.TypeRecorder`), which the tool supports
and this round did not run.

### 6.5 Relation to `scripts/screen_dns.py`

The ad hoc screen this tool generalises (`out/dn_screen.csv`, written 2026-09-10, i.e. before the session-10 weights)
stimulates both sides of each type at once, so a lateralised effect cancels: it gives DNa02 legL 3.31 / legR 2.74
(asymmetry +0.57) where the by-side atlas gives DNa02_L 2.73 / 0.50 (+2.23) and DNa02_R 0.25 / 1.70 (-1.45). The old
CSV is not a comparison arm -- different weights, different protocol -- and no number from it is quoted as a reference
anywhere above. What the atlas adds over it: the by-side split, a matched null in the same batch, three independent
runs with the scatter, `common.compare` verdicts, the self-drive and pruned-output columns, the sensory half, the
provenance block and the Neurome `readout_per_body` table.

---

## 7. Contract notes, and one bug in a shared file

1. **`flyverse/interp/__init__.py`'s tool hand-off is shadowed by its own import** (not my file; reported, not
   changed). `__getattr__` calls `importlib.import_module(".atlas", ...)`, which sets `flyverse.interp.atlas` to the
   *module*; every later `getattr(interp, "atlas")` finds that attribute and never reaches `__getattr__`, so it
   returns the module instead of the function. Any test or caller that does `from flyverse.interp import atlas as A`
   (as every implementer's test class does) breaks `StubTests.test_stubs_import_and_signatures` for that tool if
   `StubTests` runs afterwards. Under `pytest` the classes run in definition order, `StubTests` comes first and the
   whole file passes (a whole-file count is not quoted here: `tests/test_interp.py` is shared and moves under other
   owners; `AtlasTests` alone is `6 passed` in 4.0 s); under `python -m unittest` the classes run alphabetically and `StubTests` fails --
   with `HealthTests` alone, before this task's class existed. One line in `_implementation` fixes it:
   `globals()[name] = obj` before the `return`, so the package attribute holds the function, not the module. I did not
   apply it: `__init__.py` is the design task's file (`docs/INTERP.md` section 8).
2. **`provenance.flyverse_commit.commit` is `unknown` for every cluster job.** `cluster_run.py` ships a file copy of
   the checkout without `.git`, so `common.git_state()` on the node has nothing to read. `analyse_runs` now records
   `provenance.flyverse_commit_analysis` -- the desktop checkout that produced the code and ran the analysis
   (`0d32fd6e74067569317d1a4bd604ce2f82f69dd5`, dirty, with the interp files listed) -- *beside* the run's own
   `flyverse_commit`, rather than overwriting what the run actually reported. Worth fixing centrally (shipping
   `git rev-parse HEAD` into the run environment), since every tool's GPU half has the same hole.
3. **`SD_FLOOR_HZ = 0.05` is a declared constant, not a fit.** It only affects rows whose null arm is bit-identical
   (an all-silent motor group), where `common.compare`'s z is NaN by construction. `verdict_compare` (the unfloored
   verdict) is kept in every row, so the effect of the floor is auditable per row.
4. **`walk.power_max` is not used anywhere in this tool**, and no default was tuned toward any number here. The
   `power` readout is reported as a rate in Hz against its null; the 50 Hz take-off bound never appears.
5. Contract parameters kept exactly: `atlas(c, populations, *, hz, ms, settle_ms, batch, readouts, pattern, by_side,
   null, replicates, params, optic_params, device, context, seed)`. Added keyword-only with defaults: `split`,
   `n_null`, `min_cells`, `modules`, `include_pn`, `per_body`, `top`, `summary`, `out_dir`, `quiet`, `sd_floor`,
   `per_body_for`. `StubTests` enforces this and passes.
6. The `context` argument is implemented (`CONTEXTS`: `wind_left`, `wind_right`, `wind_head_on`, `wind_off`, `sugar`;
   a `{sense: args}` dict; or a callable driving the FlyBrain each frame) but **this round ran every population with
   `context=None`**, the `screen_dns.py` protocol. `docs/INTERP.md` section 9 asks "whether `stimulate` of JO-C/E
   alone (no room) gives the same DNp18 / DNp33 flips": section 3.1 answers **yes, once the wind's meander is
   carried across** -- +50.98 / -49.31 with the direction pinned at its nominal value, +47.49 / -48.59 when the JO
   drive is swept over the room's own +-20 deg / 8 s meander, against the room's +45.2 / -49.6. The JO drive is the
   whole of that signal; the residual DNp18 overshoot is the meander (7 % more drive), not the room.
7. **`docs/INTERP.md` section 7's atlas example passes `--by-side`.** When this record was first written the CLI
   implemented only `--no-by-side` and that documented line died with `unrecognized arguments: --by-side`. It now
   parses: `scripts/interp_atlas.py:142` adds `--by-side` as a **no-op alias** -- by-side is already the default and
   only `--no-by-side` changes anything (`by_side=not args.no_by_side`, lines 44 / 49 / 63). The example runs today
   and does what it says; the flag is decorative, not a switch.
8. **The null rows' averaging window is not matched to the stimulus row's.** In `run_once` the stimulus row uses that
   population's own pulse length (`trace[:w, row]`, `w = round(ps.ms / frame_ms)`, `flyverse/interp/atlas.py:562-566`)
   but the null rows use `w = n_pulse`, the **longest** pulse in the chunk (`:571-576`). In the validation batch that
   is 300 frames of null (the 3 s JO arms) against 40 frames of DNa02 / PFL3 stimulus. It changes nothing in this
   round -- every null is identically zero, and the dn+sensory batch pulses every population for the same 400 ms --
   but it would bite the moment the tool is run with a sensory `--context` (`wind_left`, `sugar`), where the null rows
   are not silent and the two windows would no longer be comparable.

---

## 8. Provenance (identical in all three Results)

```
dataset          male-cns, v1.0 flat-connectome, 4 files
                 body-annotations-male-cns-v1.0-minconf-0.5.feather   sha256 2177e246113e4cfb...
                 body-neurotransmitters-male-cns-v1.0.feather         sha256 95c9289220663abe...
                 connectome-weights-male-cns-v1.0-minconf-0.5.feather sha256 e35da783d1c686b2...
                 tbar-neurotransmitters-male-cns-v1.0.feather         sha256 bade84c9eab431dd...
compiled W       md5 ef23cc27bea13be7f6a96f3c04fd3737   nnz 25,578,600   n 167,106   sum|W| 121,460,584
model.lif        w_syn 0.275, conn_cap 60, same_type_gain 0.1, adapt_jump 1.5, input_norm alpha 1 / ref 5000,
                 t_ref 2.2 ms, v_th -45 / v_rest -52 mV, receptor_model 'sign', receptor_net_rule 'abs',
                 prune_frozen True, path_gain [(descending_neuron -> vnc_, 3.0), (visual_projection -> descending_neuron, 2.0)],
                 type_path_gain [((LC4|LPLC2) -> DNp01, 3.0)]   <- the only entry; the round-5 default, GF damping retired
model.body       gf_hz 33, takeoff_power_hz 50, takeoff_hold_s 0.3, mdn_threshold_hz 15, k_opto 0
execution        device cuda (NVIDIA B200), devices ['cuda','cuda','cuda'], host node1, torch 2.11.0+cu128,
                 dt lif 0.5 / optic 1 / frame 10 ms, batch 64, seeds brain [0,1,2], replicate_unit 'runs'
stimulus         protocol 'atlas', hz 150, ms 400, settle_ms 200, n_null_rows 4, context None,
                 control '4 unstimulated rows of the same FlyBrain batch, same seed, same context'
analysis         flyverse_commit_analysis 0d32fd6e74067569317d1a4bd604ce2f82f69dd5 (dirty)
```

`Result.check()` returns `[]` for `validation.json`, `validation_half.json` and `dn_sensory.json`; the
`readout_per_body` tables carry every column of `common.EXPORT_TABLES` with bodyIds as decimal strings (2,660 rows in
the dn batch, 2,128 in each validation Result), so `interp_export` can serialise them unchanged.

---

## 9. Reproducing and extending

```bash
# the whole round again (one cluster batch, ~3 minutes of wall time)
python scripts/cluster_run.py --name atlas --minutes 90 <the six commands of section 2> --fetch out/atlas/
PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/val_r*" --top 8  --json out/interp/atlas/validation.json
PYTHONIOENCODING=utf-8 python scripts/interp_atlas.py analyse --runs "out/atlas/dn_r*"  --top 20 --json out/interp/atlas/dn_sensory.json

# an arbitrary population list, with the VNC recorded per type as well as the pooled motor groups
python scripts/interp_atlas.py run --populations "~^LC1[01]" --populations "superclass=visual_projection" \
    --pattern "^(DN|IN0|MN)" --hz 150 --ms 400 --seed 0 --out out/atlas/vpn_r0

# with a sensory context held on for the whole run (the wind arms as a context rather than a pulse)
python scripts/interp_atlas.py run --preset validation --context wind_left --seed 0 --out out/atlas/ctx_r0

# the CPU test
PYTHONIOENCODING=utf-8 python -m pytest tests/test_interp.py::AtlasTests -q      # 6 passed, 4.0 s
```

`AtlasTests` is six methods -- `test_population_list_and_specs`, `test_readout_groups_and_derived`,
`test_wind_deflections_and_contexts`, `test_run_roundtrip_and_null_comparison`, `test_atlas_end_to_end`,
`test_per_body_rows_and_validation` -- exercising the whole path on a 6-neuron subset of `graph()` (the photoreceptor and the optic rate unit are
dropped so `FlyBrain` builds no retina -- the synthetic graph has no hex coordinates): the stimulation grammar and its
spec round trip, the readout groups and their left-right derivations, `wind_deflections` / `CONTEXTS` against
`flyverse/air.py`'s convention, a real three-run `FlyBrain` sweep with its null rows, the `AtlasRun` npz/json round
trip, the arm comparison (including "two runs are never a result"), the `self_drive` marking, the Neurome
`readout_per_body` rows, and `validate`'s extraction of the reference numbers from an atlas table.
