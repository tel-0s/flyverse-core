# The expectation ledger (`flyverse/interp/ledger.py`)

**Status: built and validated, 2026-09-12.** Interpretability toolkit, tool 7 of 8 (`docs/INTERP.md` 4.7).
Generators: `flyverse/interp/ledger.py`, `scripts/interp_ledger.py`, the table `flyverse/data/expected_responses.csv`,
tests `tests/test_interp.py::LedgerTests` (10 CPU tests). Artefacts quoted here:
`out/interp/ledger/all.json` / `all.csv` (run `ledger-20260912T234849Z-1df46537`, 98 sources, 301 scored rows) and
`out/interp/ledger/validate.json` / `validate.csv` (the validation target).

---

## 1. What it is, and the three things it is not

One question -- *does the model's measured response to a named stimulus match what the literature and this project's
own audits say it should be?* -- asked of every file the project has already produced, with the citation and the
replicate scatter attached to each answer. The tool simulates nothing: it reads finished outputs and the shipped
table, and writes PASS / FAIL / MISSING per (row, arm).

* **It is not a fitting target.** A row is an expectation with a citation, not a knob. The project rule stands: the
  plain model is not hand-tuned toward behaviour, so a FAIL is a localization task for `trace` / `lesion` / `paths`.
  The table is the *input* to those tools, never a loss.
* **It is not a second opinion on the battery.** Where a row carries `check_key`, the ledger's status must equal the
  status `scripts/benchmark.py` wrote for the same key on the same JSON. 69 of 70 such rows agree; the one that does
  not is explained by the stored file's own criterion (section 6), not by a scoring difference.
* **It is not a place to hide a bad bound.** `walk.power_MN.rate_max_hz` carries op `report`: dynamics round 1 showed
  the 50 Hz bound is unfit (it fails under 13 of 16 optic ablations and anti-correlates with room take-offs), so the
  ledger records 79.06 Hz from `out/benchmark_suite.json` as `RECORDED` and never turns it into a verdict.

---

## 2. The table: `flyverse/data/expected_responses.csv`

110 rows, one per expectation. Columns beyond the `docs/INTERP.md` 4.7 contract (`population`, `stimulus`,
`quantity`, `expected`, `op`, `unit`, `source`, `model_reference`, `notes`) are this implementation's:

| column | meaning |
|---|---|
| `row_id` | stable id `<family>.<population>.<quantity>`; the ledger's key |
| `label` | the short name a probe file uses for the population -- the **match** key. `population` stays the resolvable `flyverse.interp.common` grammar spec and is used for `n_cells` / bodyIds and the Neurome export (58 populations resolved in `all.json`) |
| `bound` | the criterion value the `op` compares against. `expected` stays the *reference value*, exactly as `scripts/benchmark.py`'s `Ref.value` does -- a reference is not a criterion |
| `gap` | 1 = a documented model deficit: failing prints `KNOWN GAP`, meeting it `PASS (gap closed)` (`Ref.gap`) |
| `check_key` | the `benchmark.REFERENCES` key this row mirrors, so the two statuses can be compared |
| `requires` | another `row_id` this row is only meaningful when. Unless that row PASSes **in the same arm**, the status is `NOT_APPLICABLE` (section 5c) |

Ops: `> < >= <= == abs>= abs<= sign range notnone is report`. `sign` / `range` / `abs>=` are the contract; `== abs<=
notnone is` exist to reproduce `benchmark.py`'s own criteria exactly; `report` records a number that the project has
decided is not a verdict. Statuses: `PASS`, `PASS (gap closed)`, `FAIL`, `KNOWN GAP`, `MISSING`, `RECORDED`,
`NOT_APPLICABLE`.

**Coverage of the mandated validation table** (enforced by `LedgerTests.test_shipped_table_covers_the_mandated_literature`):

| result | rows | citation |
|---|---|---|
| T4/T5 direction selectivity | `motion.T4a..T5d.dsi` (8), `.direction` (8), `motion.T4_T5.min_dsi`, `.correct_directions` | Maisak et al. 2013, Nature 500:212 |
| LC4 / LPLC2 loom | `loom.LC4.rate_peak_hz`, `loom.LC6.rate_peak_hz`, `loom.LPLC2.rate_peak_hz`, `objsweep.LPLC2.figure_z`, `objsweep.LC4.figure_z` | Ache et al. 2019, Curr Biol 29:1073; Klapoetke et al. 2017, Nature 551:237 |
| HS / VS optomotor | `rotation.HSN.dprime`, `.HSE.dprime`, `.VS.dprime`, `.HSN.flip_sign`, `.DNp20.dprime`, `.group.flip_hz`, `.DNa02.rate_hz` | Schnell et al. 2010, J Neurophysiol 103:1646 |
| ORN odour tuning | `orn.ORN_DM2/DM4/DC1.rate_hz` | Hallem & Carlson 2006, Cell 125:143 |
| sugar -> MN9 and bitter suppression | `taste.MN9.rate_hz`, `.rate_hz_calibrated`, `.rate_hz_calibrated_bitter`, `.rate_hz_shiu`, `.rate_hz_shiu_bitter`, `.rate_hz_bitter_only` | Shiu et al. 2024, Nature 634:210 |
| EPG bump (rate and width) | `compass.EPG.bump_rate_hz`, `.bump_width_wedges`, `.bump_survival_s`, `.circ_corr_heading`, `.cells_persisting` | Seelig & Jayaraman 2015, Nature 521:186 |
| DNp09 / MDN | `dn.MDN.top_rate_hz`, `.leg_L_hz`, `.leg_R_hz`, `dn.DNp09.top_rate_hz`, `.leg_L_hz` | Bidaye et al. 2014, Science 344:97; Bidaye et al. 2020, Neuron 108:469 |
| GF escape latency | `loom.DNp01.rate_peak_hz`, `.rate_peak_hz_demo`, `.escape_range_cm`, `.escape_latency_ms`, `loom.body.escapes`, `loom.TTMn.rate_peak_hz` | von Reyn et al. 2014, Nat Neurosci 17:962 |
| LC10 courtship-object | `object.LC10a.flip_hz`, `.rate_hz`, `object.LC10b.rate_hz`, `objsweep.LC10a.figure_z` | Ribeiro et al. 2018, Cell 174:607 |

Plus the audits' own per-type numbers: `objsweep.*` (object_sweep.md 8.4 / 8.7), `fg.*` (optic_measures.md 5.1),
`compass.*` (compass_room.md 4), `hops.*` + `walk.*` (receptor_integration.md round 5), `struct.*` (cx_glno.md 1),
`smell.*` / `odour.*` / `rest.*` / `wind.*` (benchmark.REFERENCES). Every `check_key` in the table is a real
`benchmark.REFERENCES` key (also a test).

Rows whose `source` says *"no literature number"* are honest about it: `taste.GNG175.rate_hz` (a storm guard),
`rest.brain.spikes_per_step`, `hops.*`, `odour.LH_apple.*` (a model readout), `struct.*` (structural facts of the
compiled graph), `compass.PEN_L` / `Delta7_L` (no in-vivo rate to compare).

---

## 3. Sources: what the ledger can read

`read_source` sniffs the kind; nothing is declared. An unreadable or unrecognised file becomes a row of the `sources`
table with an `error`, never an exception.

| kind | file | what it yields |
|---|---|---|
| `benchmark` | `scripts/benchmark.py` JSON (`out/benchmark_suite.json`, `out/rm2_*.json`, `out/r5_*_report_suite.json`) | 40 section scalars + per-subtype motion / rotation / wind / object / odour / dn / top-type rates, **and** every `checks` entry with the status and criterion the battery wrote |
| `object_sweep` | `probe_object_sweep.py` JSON (`out/r3obj/{ball,null}_*.json`) | the 9 per-type difference statistics; `config.null` marks the none-vs-none arm |
| `figure_stages` | `probe_figure_stages.py` / `audit_optic.py` stage JSON (`out/optic_audit/<config>/stages_s*.json`) | per type and stage the signed and \|dev\| figure z of the static apple and moving ball; **the file's own `CB` block is emitted as the null arm** |
| `figure_ground` | `probe_figure_ground.py` CSV (`out/fg*.csv`) | the signed retinotopic figure z |
| `rotation` | `screen_rotation.py` CSV (`out/screen_rotation.csv`, `out/rot_*.csv`) | per-type d', flip Hz, cw / ccw / rest L-R |
| `compass_room` | `probe_compass_room.py` JSON (`out/cxroom/*.json`) | 20 post-pulse population / bump summaries (mean over the run's 16 flies) |
| `batch_sustain` | `batch_sustain.py` JSON (`out/r5_sustain_*.json`, `out/d1_*.json`) | voluntary / escape take-offs per 1,000 fly-s and the walking GF maximum |
| `taste_cpu` | `r5_attr_taste_cpu.py` JSON | the arm x seed CPU taste / smell table (one run per seed) |
| `loom`, `bitter` | `probe_loom.py` / `probe_bitter.py` console logs | per-type peak rates through the approach, the escape range and flag; MN9 under the calibrated and Shiu rule sets |
| `result` | any `flyverse.interp.result/1` JSON from any of the eight tools | every table row that names a population, with `readout_per_body` pooled per (type, quantity) |

**Arms.** Every observation carries the arm it came from, taken from the file's own resolved config, not its name:
the receptor model + net rule (`off`, `sign-abs`, `sign-class`, `sign-nonmda`, `full-class`), the hold table when one
was passed (`holdBrain`, `holdOptic`, `holdBrainGlu`, `holdBrainHis`, `holdKC`, `holdDN1`), the ring gains of a
compass run (`control`, `gE2/gD15`, `gE2.5/gD25`, `+cx` when a walking program drove the fly) and the optic
configuration of a stage file (`sign-abs+no_spk_feedback`). `all.json` scores 20 arms.

---

## 4. Validation: the ledger reproduces the battery's own statuses

`VALIDATION['ledger']` (docs/INTERP.md 6) scored from `out/benchmark_suite.json` and `out/screen_rotation.csv`.
`PYTHONIOENCODING=utf-8 python scripts/interp_ledger.py validate --json out/interp/ledger/validate.json --csv out/interp/ledger/validate.csv`
-> `out/interp/ledger/validate.json`, **status `reproduced`**.

| item | reference (docs/INTERP.md 6) | ledger row | measured | ledger status | `benchmark.py` status | agree |
|---|---|---|---|---|---|---|
| `motion.min_dsi` | 0.16 | `motion.T4_T5.min_dsi` | **0.1665** | PASS | PASS | yes |
| loom GF peak | 34-56 Hz | `loom.DNp01.rate_peak_hz_demo` | **37.117** | PASS | FAIL | stale criterion (section 6) |
| optomotor d' HSN | 4.2 | `rotation.HSN.dprime` | **-4.148** (\|4.148\|) | PASS | -- (no check key) | -- |
| optomotor d' DNp20 | 3.1 | `rotation.DNp20.dprime` | **-3.109** | PASS | -- | -- |
| optomotor d' HSE | 2.2 | `rotation.HSE.dprime` | **-2.211** | PASS | -- | -- |
| calibrated sugar MN9 | 4.6 Hz | `taste.MN9.rate_hz_calibrated` | **4.566** | PASS | PASS | yes |
| calibrated sugar+bitter MN9 | 0.0 Hz | `taste.MN9.rate_hz_calibrated_bitter` | **0.000** | PASS | PASS | yes |
| Shiu sugar MN9 | 123.5 Hz | `taste.MN9.rate_hz_shiu` | **123.539** | PASS | PASS | yes |
| Shiu sugar+bitter MN9 | 2.1 Hz | `taste.MN9.rate_hz_shiu_bitter` | **2.124** | PASS | PASS | yes |

Every reference is matched. Two notes on the reading, both of them corrections to the design document rather than to
the model:

* The **34-56 Hz loom band** is session 9's loom **demo** (`benchmark.REFERENCES['loom_escape.GF_peak_hz']`, the
  comment above that `Ref`), not the per-100 ms approach probe `loom.GF_peak_hz`, which reads **30.798 Hz** in
  `out/benchmark_suite.json` and passes its own `>= 20` criterion. The ledger scores both rows; the validation maps
  the band to the demo row, which is where it comes from.
* `screen_rotation.py` stores the **signed** d' (HSN -4.148, DNp20 -3.109, HSE -2.211): the sign is the tuning
  direction, so the table scores `abs>=` and the row `rotation.HSN.flip_sign` (op `sign`, bound -1) carries the
  direction separately. That row passes in all three arms measured (off -8.02 +- 0.70 over 3 runs, sign-class
  -4.86, the shipped screen CSV -7.06).

Over the whole corpus (`all.json`): **70 rows mirror a battery check, 69 agree, 0 disagree**, 1 stale criterion.

---

## 5. What the ledger reads off the finished corpus

`bash`-form of the run that produced `out/interp/ledger/all.json` is in `scripts/interp_ledger.py`'s docstring;
98 sources, 0 unreadable, devices of the scored runs: NVIDIA B200 / RTX 4090 / `cuda`.

```
status counts: PASS 183, PASS (gap closed) 2, FAIL 23, KNOWN GAP 73, MISSING 7, RECORDED 1, NOT_APPLICABLE 12
```

### 5a. The object sweep -- `docs/audits/object_sweep.md` 8.4 and 8.7, reproduced digit for digit

The ledger's `@z` quantities score the z against the arm-matched none-vs-none null and report `common.compare`'s
whole dict. Ball arms `out/r3obj/ball_{off,abs}_s0..4.json`, null arms `out/r3obj/null_{off,abs}_s0..4.json`
(5 runs per arm).

| row | arm | ledger mean / null mean | ledger z / Welch / U / p | audit 8.4-8.7 z |
|---|---|---|---|---|
| `objsweep.LPLC2.figure_z` | sign-abs | +0.3511 / +0.1388 | **+5.353** / +8.648 / 25 / 0.0079 | **+5.4** / +8.6 / 25 / 0.0079 |
| `objsweep.LPLC2.figure_z` | off | +0.2885 / +0.1904 | **+2.027** / +4.354 / 25 / 0.0079 | **+2.0** / +4.4 / 25 / 0.0079 |
| `objsweep.LC11.figure_z` | off | +0.0658 / +0.0681 | **-0.090** / -0.121 / 12 / 1.00 | **-0.1** / -0.1 / 12 / 1.00 |
| `objsweep.LC11.figure_z` | sign-abs | +0.0743 / +0.0663 | **+0.426** / +0.540 / 14 / 0.84 | **+0.4** / +0.5 / 14 / 0.84 |
| `objsweep.LC10a.figure_z` | off / sign-abs | | **+0.274** / **-0.098** (U 14 / 11) | **+0.3** / **-0.1** |
| `objsweep.LC4.figure_z` | off / sign-abs | | **-0.066** / **-0.978** (U 10 / 7) | **-0.1** / **-1.0** |
| `objsweep.Mi4.figure_z` | off / sign-abs | | **+28.557** / **+22.254** | **+28.6** / **+22.3** |
| `objsweep.Mi1.figure_z` | off / sign-abs | | **+7.841** / **+27.931** | **+7.8** / **+27.9** |
| `objsweep.Tm3.figure_z` | off / sign-abs | | **+7.820** / **+15.576** | **+7.8** / **+15.6** |
| `objsweep.T2.figure_z` | off / sign-abs | | +0.452 / +1.166 | +0.5 / +1.2 |
| `objsweep.T3.figure_z` | off / sign-abs | | -0.007 / +0.704 | -0.0 / +0.7 |
| `objsweep.Tm5Y.figure_z` | off / sign-abs | | +0.405 / +2.593 | +0.4 / +2.6 |
| `objsweep.TmY21.figure_z` | off / sign-abs | | +1.128 / +0.563 | +1.1 / +0.6 |
| `objsweep.TmY13.figure_z` | off / sign-abs | | -0.630 / +0.505 | -0.6 / +0.5 |
| `objsweep.TmY5a.figure_z` | off / sign-abs | | -0.999 / +1.194 | -1.0 / +1.2 |

Every number reproduces. The scoring then says what the audit says: the medulla carries (`PASS`), the LC10 / LC11
inputs sit at the null (`KNOWN GAP`), LPLC2 is the one spiking type above it -- **in the `sign-abs` arm only**
(`FAIL` under `off` at z +2.0, which is the audit's own reading at 8.5: "at z > 3 on the primary statistic: LPLC2
under sign/abs and nothing else"). `Tm5Y sign-abs` is the marginal case: z +2.593, p 0.0079 but below the |z| >= 3
criterion, so `compare`'s verdict is not `result` and the row stays `KNOWN GAP`.

### 5b. The static-apple figure map -- `docs/audits/optic_measures.md` 5.1

Pooled over 5 runs of the same arm: `out/optic_audit/baseline/stages_s{0,1,2}.json` (whose `CB` block is the null)
and `out/fg2_abs_s{0,1}.csv`.

| row (arm `sign-abs`) | ledger mean +- sd (5 runs) | audit 5.1 (3-seed mean) | status |
|---|---|---|---|
| `fg.L1.figure_z` | **+18.567 +- 1.779** | +18.4 | PASS |
| `fg.L2.figure_z` | **+12.453 +- 1.337** | +12.0 | PASS |
| `fg.Mi1.figure_z` | **-9.457 +- 1.158** | -8.8 | PASS |
| `fg.Tm3.figure_z` | **-7.302 +- 1.364** | -6.8 | PASS |
| `fg.Tm1.figure_z` | **+4.563 +- 0.252** | +4.5 | PASS |
| `fg.Tm4.figure_z` | **+5.037 +- 0.578** | +4.8 | PASS |
| `fg.Tm20.figure_z` | **+5.210 +- 0.611** | +5.3 | PASS |
| `fg.T4c.figure_z` | **+3.695 +- 0.872** | +4.2 | PASS |
| `fg.T4d.figure_z` | **+3.692 +- 0.685** | +4.0 | PASS |
| `fg.T2.figure_z` | **+2.911 +- 0.866** | +2.7 | KNOWN GAP |
| `fg.T3.figure_z` | **-1.711 +- 0.991** | -1.0 | KNOWN GAP |
| `fg.Tm5Y.figure_z` | **+0.387 +- 0.248** | +0.2 | KNOWN GAP |
| `fg.TmY21.figure_z` | **-0.706 +- 0.352** | -0.6 | KNOWN GAP |
| `fg.TmY13.figure_z` | **+2.009 +- 0.610** | +2.2 | KNOWN GAP |

Every type within the +-1.5 the audit itself declares for a per-type figure z. **A new reading the ledger makes:**
the same rows scored per receptor arm show the carriers are arm-dependent, and the class net rule loses them.
`fg.Tm1` is +4.56 under `sign-abs`, +4.04 under `off`, **+0.21 +- 0.50 under `sign-class`** and -0.35 under
`sign-nonmda`; `fg.Mi1` -9.46 / -8.81 / **-1.40** / +2.11; `fg.Tm4` +5.04 / +4.77 / **+2.99** / +0.67; `fg.T4c`
+3.69 / +2.56 / +2.03 / -0.07. **17 of the 23 FAIL rows in `all.json` are exactly this**: a figure carrier below
|z| = 3 in a receptor arm other than `sign-abs` (13 of them under `sign-class` / `sign-nonmda`, 4 under `sign` /
`off` on the two marginal T4 rows). That is a localization question for `trace` / `decompose` -- which edges the
class net rule re-signs on the ON/OFF columnar types -- not a bound to relax. The other six FAILs are
`loom.body.escapes` (0 of 1 demo seed escapes in `out/benchmark_suite.json`), `rotation.HSN.dprime` /
`rotation.DNp20.dprime` under `sign-class` (-1.998, -1.635 over 2 runs each, against \|d'\| >= 2.0),
`objsweep.LPLC2.figure_z` under `off` (5a) and `compass.EPG.bump_survival_s` in the two control arms (5c).

### 5c. The compass -- `docs/audits/compass_room.md` 4, and why `requires` exists

12 `out/cxroom/*.json`, 2 runs per (gain, program) arm.

| row | control (+cx) | gE2/gD15 (+cx) | gE2.5/gD25 (+cx) | expectation |
|---|---|---|---|---|
| `compass.EPG.bump_survival_s` | **0.033 / 0.033** FAIL | **38.0 / 38.0** PASS | **38.0 / 38.0** PASS | >= 5 s (Seelig 2015) |
| `compass.EPG.bump_rate_hz` | 34.90 / 34.92 `NOT_APPLICABLE` | **219.49 / 219.08** KNOWN GAP | **259.55 / 260.55** KNOWN GAP | 5-60 Hz |
| `compass.EPG.bump_width_wedges` | 3.875 `NOT_APPLICABLE` | **3.624 / 3.593** PASS | **3.806 / 3.830** PASS | 2.5-5.0 wedges |
| `compass.EPG.circ_corr_heading` | NaN MISSING | **+0.002 / -0.026** KNOWN GAP | **-0.045 / +0.018** KNOWN GAP | \|r\| >= 0.5 |
| `compass.PEN_L.rate_hz` | 0.016 `NOT_APPLICABLE` | **56.65 / 56.62** PASS | **75.47 / 76.18** PASS | 5-120 Hz |
| `compass.Delta7_L.rate_hz` | 0.050 `NOT_APPLICABLE` | **104.08 / 103.44** PASS | **112.52 / 110.83** PASS | 5-200 Hz |
| `compass.PFN.rate_hz` | 5e-05 `NOT_APPLICABLE` | **0.600 / 0.611** KNOWN GAP | **0.776 / 0.783** KNOWN GAP | >= 1 Hz |
| `compass.hDelta.rate_hz` | 3e-05 `NOT_APPLICABLE` | **1.490 / 1.499** KNOWN GAP | **1.826 / 1.815** KNOWN GAP | >= 2 Hz |

This reproduces dynamics round 1's compass result exactly (219-260 Hz, 38 s persistence, \|circ corr\| <= 0.045 here
against the round's <= 0.11 over 12 runs, PFN 0.60-0.78, hDelta 1.49-1.83), and it is the case that motivated the
`requires` column. The shipped default has **no bump** (it dies in 0.033 s), yet its `bump_hz_post` reads 34.9 Hz --
the peak of a wedge that is not an attractor -- which is inside the 5-60 Hz literature window and would have scored a
**false PASS (gap closed)**. With `requires = compass.EPG.bump_survival_s`, every row that presupposes a bump reads
`NOT_APPLICABLE` in the arms where the bump row failed, and the one row that failed is named in the precondition
string. 12 rows in `all.json` are labelled this way; none of them is a verdict.

### 5d. The round-5 double dissociation and the take-off hold split

From `out/r5_attr_taste_cpu.json` (arms x seeds 0-2, CPU) and 15 `batch_sustain.py` room rollouts
(`out/r5_sustain_*_live_*.json`, `out/d1_*.json`; 4,800 fly-s each):

| row | off | default (`sign-abs`) | holdBrain | holdOptic | holdBrainGlu | holdBrainHis | holdKC | holdDN1 |
|---|---|---|---|---|---|---|---|---|
| `taste.MN9.rate_hz` | 2.735 | **3.922** | **2.735** | **3.922** | **3.922** | **2.735** | 3.922 | 3.922 |
| `smell.KC.n_active` | 894.7 | **480.3** | **894.7** | **480.3** | **894.7** | **480.3** | 898.3 | 499.3 |

`holdBrain` = `off` and `holdOptic` = `default` to every digit on both checks; `holdBrainGlu` mirrors `off` on smell
and `default` on taste while `holdBrainHis` does the opposite -- the double dissociation of
`docs/audits/receptor_integration.md` E.4, read straight off the ledger. (`off` in the pooled row above is 3.513 /
1070.75 because the arm also pools the GPU benchmark draw; the per-source numbers are in the `observations` table.
The E.4 reference means -- off 2.735 / 894.7, default 3.922 / 480.3, holdKC 898.3, holdDN1 499.3 -- are the CPU
protocol's three seeds and match `common.VALIDATION['lesion']['reference']` exactly.)

| row | off (6 runs) | default `sign-abs` (3) | holdBrain (3) | holdOptic (3) |
|---|---|---|---|---|
| `hops.body.voluntary_per_1000_fly_s` | **0.000 +- 0.000** PASS (gap closed) | **3.056 +- 0.434** KNOWN GAP | **2.361 +- 0.481** KNOWN GAP | **0.000 +- 0.000** PASS (gap closed) |
| `hops.body.escape_per_1000_fly_s` | 0.556 +- 0.215 | 2.153 +- 0.434 | 1.250 +- 0.417 | 0.347 +- 0.434 |
| `hops.DNp01.rate_max_median_hz` | 27.205 +- 0.767 | 32.316 +- 1.001 | 31.976 +- 0.949 | 26.996 +- 0.216 |

Holding the **Brain** half of the receptor signs at NT sign (`holdBrain`, i.e. only the optic half's changes live)
keeps the whole take-off cost -- 2.36 of the default's 3.06 per 1,000 fly-s, GF median 31.98 of 32.32 -- while
holding the **optic** half (`holdOptic`) removes all of it and returns the `off` numbers to three decimals. That is
dynamics round 1 result (iii) with three replicates per arm and the scatter quoted.

### 5e. What is MISSING, and why

7 rows, all of them by design rather than by accident:

* `loom.DNp01.escape_latency_ms` -- **no shipped probe measures it.** `probe_loom.py` prints 100 ms frames and no
  JSON, so the von Reyn short-mode latency has no number in this project. `scripts/interp_atlas.py` (DNp01 pulse ->
  TTMn / DLMn onset) or a `--json` output from `probe_loom.py` would resolve it.
* `compass.EPG.circ_corr_heading` in the two control arms -- the correlation is NaN when there is no bump centre.
* `struct.GLNO_PEN.synaptic_pair_count`, `struct.GLNO_PEN.sign`, `struct.EPG_PEN.effective_mv`,
  `struct.Delta7_PEN.effective_mv` -- these score a `flyverse.interp.result/1` `per_type` row from
  `scripts/interp_paths.py`, which had not been run into `out/interp/paths/` when this document was written. The
  rows exist so that the structural claims of `cx_glno.md` 1 (GLNO -> PEN 16,371 synapses, sign 0; EPG -> PEN +5.05
  mV per pair; Delta7 -> PEN -4.70) are scored the moment `paths` writes its JSON, through the same `result` reader
  that serves every other tool.

---

## 6. The one battery disagreement, and what it means

`loom.DNp01.rate_peak_hz_demo` (check `loom_escape.GF_peak_hz`): the ledger says PASS, `out/benchmark_suite.json`
says FAIL, on the same measured 37.117 Hz. Neither is a scoring bug -- **the stored file kept an older criterion**.
`out/benchmark_suite.json` records `reference 46, criterion ">= 38"`; the live `scripts/benchmark.py` has
`Ref(41, ">=", 33)` since session 9 (LPi x4, `gf_hz` 33). The table follows the live battery (bound 33), so the
ledger flags the row `battery_criterion_same = False` and reports it under `summary.battery_criterion_mismatch`
rather than `summary.battery_disagree`.

That distinction is worth keeping: a checker that silently agreed with whatever a stored file said would have hidden
the retune, and one that reported it as a disagreement would have cried wolf. `summary.battery_disagree` is
therefore *only* the rows where both sides used the same criterion and still differ -- in this corpus, none. Rows
scored with op `report` are excluded from both lists by construction (their divergence from the battery is the
point). Re-running `scripts/benchmark.py` would clear the flag.

---

## 7. Reporting debt and open questions

1. **A contract bug in `flyverse/interp/__init__.py` (not this task's file).** `_implementation` does
   `importlib.import_module(f".{name}", __name__)`, which binds the *submodule* as `flyverse.interp.<tool>`; the
   function it returns is not cached, so the **second** `getattr(interp, "<tool>")` returns a module, not a
   callable. Any code that imports a tool submodule (`from flyverse.interp import ledger`) triggers it too. In the
   full suite this makes `StubTests.test_stubs_import_and_signatures` fail on `lesion`
   (`tests/test_interp.py::LesionTests` imports that module first). `LedgerTests.tearDownClass` drops the binding so
   this tool does not add to it, but the fix belongs in `__init__.py`: cache the function
   (`globals()[name] = obj`) after resolving it. `tests/test_control.py` is unaffected (18 tests, pass).
2. **Pooling across protocols.** An arm pools every run that measured the same (population, stimulus, quantity),
   even from different protocols -- `taste.MN9.rate_hz` under `off` pools the CPU `r5_attr_taste_cpu` seeds (2.735)
   with the GPU benchmark draw (5.845), giving 3.513 +- 1.950 over 4 runs. The `source_kinds` column names the kinds
   that went in and the sd shows the spread, but the ledger has no notion of "the same protocol". If that becomes a
   problem, the fix is a `protocol` column in the table (or `group_by='arm,kind'`), not a second stimulus id.
3. **`--null` is the common flag with values.** `common.add_common_args` declares `--null` as `store_true`; the
   ledger needs paths, so `scripts/interp_ledger.py` builds its parser with `conflict_handler="resolve"` and re-adds
   `--null` with `nargs="*"`. A bare `--null` is then a no-op rather than an error. This is the only place the CLI
   convention of docs/INTERP.md 2.6 was bent, and the flag name is unchanged.
4. **Two rows have no replicate at all in this corpus** (`n = 1`): everything read from `out/benchmark_suite.json`
   alone. 78 of the 301 rows are under three runs; they are listed in `summary.underpowered_rows` and no difference
   is called a result there. The fix is more draws, not a lower bar.
5. **`rotation.*` from `out/screen_rotation.csv` has arm `unknown`** -- that CSV has no sibling console log, so
   neither the header nor the file name names a receptor model. Re-running `screen_rotation.py` with its stdout kept
   beside the CSV would resolve it.
6. **`walk.power_MN.rate_max_hz` stays op `report`.** If the project ever establishes a fit bound for the wing-power
   maximum, that row (and only that row) changes -- the ledger is where such a decision should be recorded, with the
   file that justified it in `model_reference`.

---

## 8. Files

| file | what |
|---|---|
| `flyverse/interp/ledger.py` | `load_table`, `evaluate`, the ten source readers, `read_sources`, `ledger()`, `validate()` |
| `flyverse/data/expected_responses.csv` | 110 expectations, 15 columns, the citations |
| `scripts/interp_ledger.py` | `analyse` (default) / `validate` / `table`; `record` / `run` say there is no GPU part |
| `tests/test_interp.py::LedgerTests` | 10 CPU tests (table validity + literature coverage, every op, the benchmark reader and battery agreement, the object-sweep null reproducing 8.4, the stage reader's own null, the text / CSV readers, arms + preconditions + Result schema, the validation helpers, the CLI) |
| `out/interp/ledger/all.json` / `all.csv` | the whole corpus: 98 sources, 301 rows, 20 arms, `Result.check()` empty |
| `out/interp/ledger/validate.json` / `validate.csv` | the validation target, status `reproduced` |

Provenance of `all.json`: commit `0d32fd6e7406` (dirty), MaleCNS v1.0 flat-connectome (4 files with SHA-256),
compiled `W` md5 `ef23cc27bea13be7f6a96f3c04fd3737`, nnz 25,578,600, 167,106 neurons, sum\|W\| 121,460,584,
`LIFParams` and `OpticParams` fully resolved, body thresholds `gf_hz 33 / takeoff_power_hz 50 / takeoff_hold_s 0.3 /
mdn_threshold_hz 15`, `execution.device = cpu` (the ledger simulates nothing) with
`execution.scored_devices = [NVIDIA B200, NVIDIA GeForce RTX 4090, cuda]` and every scored run's own device in the
`sources` table, `replicate_unit = runs`, the units table -- so the Result passes `check()` and
`scripts/interp_export.py` can serialize it unchanged.
