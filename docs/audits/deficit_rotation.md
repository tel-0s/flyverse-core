# apply:rotation -- where the fly's rotation dies on its way to PEN (the toolkit applied to one deficit)

**The deficit.** The ring bump does not follow the fly's rotation: at 90 deg/s an imposed visual rotation moves the
bump 0.00 +- 0.01 wedges/s against 4.0 ideal (`docs/audits/cx_shift.md` 3b), a walking fly's yaw is ignored the same way
(`compass_room.md` 4b), and the structural reading was that PEN has no signed rotation input because GLNO -- 19.4 % of
PEN's raw input -- carries sign 0 (`cx_glno.md` 1, `cx_shift.md` 1). The open question left by dynamics round 1 was the
*efferent* route: the fly was teleported in every rotation experiment, so PS196_b / LAL -> GLNO (premotor efference
copy) was never excited. This audit answers, with the validated tools and no change to the model: (1) which link or
population blocks the rotation signal, (2) what the connectome data say about it, (3) what a data-driven resolution
would be.

**Tool.** `scripts/interp_apply_rotation.py` (`paths` / `record` / `analyse` / `report` / `chain` / `verify-batch` /
`selftest`), composing `flyverse.interp.paths`, `.trace` (its `ArmAccumulator` recorder and `trace()`), `.decompose`
and `common.compare`; the rotation protocol is `scripts/screen_rotation.py` / `cx_shift.py --rotation` (pinned fly at
(0, 0, 0.75), wind off, rest / +90 / rest / -90 deg/s, 10 s per phase, the first 3 s of each phase skipped), run under
the compass gains gE 2 / gD 15 (a bump in the ring; the shipped ring has none in the room) in two conditions (GLNO
silent = the shipped cache; GLNO = gaba = `cx_wedge`'s `--nt-override` scratch cache `out/cache_72164311`,
TYPE_NT_OVERRIDE + {GLNO: gaba}, the same compile as `cx_glno.md` 4-5) and two modes: **visual** (the fly is re-placed
every 10 ms, the world turns around a body that issues no turn -- `cx_shift.md`'s experiment 2) and **efferent** (new:
the fly turns *itself* -- DNa02 of one side is pulsed at 20 Hz for the phase and `body.Locomotion` integrates the turn
from the DNa02 L - R and leg-MN asymmetry, position pinned, heading free; the visual world then rotates with the body
as it would in a walking fly, and PS / LAL / DNa02 are active). Every stochastic number is over **10 independent runs
per (condition, mode)** -- seeds 0-4 in each of two same-code cluster batches (`rot-cf0c43`, `rot-7de91e`; section
2.1) -- and every difference is referenced to a null of the same shape (the second rest phase against the first);
the two batches are also analysed separately as a replication (`out/interp/apply_rotation/batch_<id>/`). `selftest`
checks the flip and bump statistics on synthetic data.

Files: `out/interp/apply_rotation/paths/` (8 paths Results + `paths_summary.json`, CPU), `paths_nf/` (4 paths Results
with the never_firing flag from a batch recording), `out/rot_cf0c43/` and `out/rot_7de91e/` (20 runs x 4 phases x
{per-cell means + pooled series, PEN / GLNO presynaptic per-frame recording} + console `.txt` per job, each batch
directory checked by `verify-batch`: `out/interp/apply_rotation/verify/verify_rot_<id>.csv`), `out/rot_cluster.log`,
`out/interp/apply_rotation/<condition>_<mode>/` (pooled 10 runs: 4 trace Results, 3 decompose Results, `flip.csv`,
`bump.csv`, `summary.json` + `summary.md`), `batch_cf0c43/<condition>_<mode>/` and `batch_7de91e/<condition>_<mode>/`
(the same per batch, 5 runs), `report_{bump,flip}.csv` (pooled; per batch under `batch_<id>/`), `chain_rates.csv`
(pooled; per batch under `batch_<id>/`), `mixed/` (the first analysis pass on the interleaved fetch, kept for the
process note in 2.1, not used for any number below). Every JSON carries the resolved LIFParams / OpticParams, the
type_path_gain in force, the realised device and the cache fingerprint (`Result.check()` empty on all 12 summaries and
their tool files; a summary passes `export.export` + `verify(expect_paired=())` with no problems).

---

## 1. Structure: every yaw carrier -> PEN and -> GLNO (`paths`, CPU, shared cache, `LIFParams()`)

Sources: the populations that flip under imposed rotation in the model (`out/screen_rotation.csv` |d'| >= 2:
HSN, DNp20, Nod1, LPT26, LPT50, HSE, DNp15) with the other HS / VS / H2 / Nod cells (`optic_yaw`, 36 cells;
`descending_yaw` = DNp20 | DNp15, 4 cells), Johnston's organ (`jo`, `~^JO-`, 672 cells), and the efference candidates
that feed GLNO (`efference` = DNa02 | DNa01 | DNa03 | PS196_b | PS196_a | LAL139 | LAL184 | WED040_a, 29 cells).
Targets: PEN (42 cells) and GLNO (4). k <= 3, type level, gains in mV^k per volley; the source node is one type of the
source group (`interp_paths.md` 3). Files `out/interp/apply_rotation/paths/paths_<source>_to_<target>.json`
(`paths_summary.json` for the table).

| source -> target | direct raw syn | top signed walk k = 2 (mV^2) | top signed walk k = 3 (mV^3) | strongest silent link on a listed walk |
|---|---|---|---|---|
| optic_yaw -> PEN | **0** | Nod4 -> LPsP -> PEN +2.1 (1 syn onto LPsP) | Nod1 -> PLP078 -> ExR4 -> PEN +4,438 (430 / 37 / 7,122 syn); Nod1 -> LAL184 -> EPG -> PEN +3,658 | GLNO -> PEN [sign0] +33.0 mV per PEN volley if signed (k 2 and 3: Nod1 -> GLNO -> PEN, 3 syn; Nod1 -> LAL184 -> GLNO -> PEN +1,965 if signed) |
| optic_yaw -> GLNO | 3 (Nod1) | Nod1 -> LAL184 -> GLNO +59.6 (28 / 309); Nod1 -> LAL139 -> GLNO -49.7 (22 / 345) | Nod1 -> WED153 -> LAL139 -> GLNO -5,390 (182 / 296 / 345); **H2 -> PS047_b -> PS196_b -> GLNO +5,060** (778 / 445 / 1,801); HSS -> PS047_b -> PS196_b -> GLNO +3,293 | ExR2 -> GLNO [sign0, dopamine] +6.1 (k 2); OA-VUMa4 -> LAL139 [sign0] +13.2 (k 3) |
| descending_yaw -> PEN | 0 | none | DNp15 -> OLVC5 -> IbSpsP -> PEN +13.9 (1 / 831 / 2,637) | GLNO -> PEN +33.0 (k 3: DNp15 -> PS047_b -> GLNO -> PEN, 1 / 3 syn) |
| descending_yaw -> GLNO | 0 | DNp15 -> PS047_b -> GLNO +0.02 | DNp15 -> AN19B039 -> PS196_b -> GLNO +258 | FB1C -> GLNO [sign0] +4.5 (k 3) |
| jo -> PEN | 0 | JO-ED2_b -> SAD047 -> PEN -0.07 | JO-EV1 -> CB2585 -> ER1_b -> PEN -513 | GLNO -> PEN +33.0 (k 3: JO-FV -> GNG284 -> GLNO -> PEN); DNc02 -> PEN [sign0] +0.1 (k 2) |
| jo -> GLNO | 0 | JO-FV -> GNG284 -> GLNO -2.0 | JO-EV3 -> PS326 -> WED011 -> GLNO -4,635 | OA-VUMa1 -> GLNO [sign0] +4.0 (k 2); OA-VUMa4 -> LAL139 +13.2 (k 3) |
| efference -> PEN | 0 | LAL184 -> EPG -> PEN +950 (1,997 / 12,330); PS196_b -> LPsP -> PEN +503 | LAL184 -> EPG -> ExR4 -> PEN -144,016; PS196_b -> LPsP -> EPG -> PEN +60,212 | GLNO -> PEN +33.0 (k 2 and 3: WED040_a -> GLNO -> PEN -1,046 if signed; PS196_b -> GLNO -> PEN +551; LAL139 -> GLNO -> PEN -542) |
| efference -> GLNO | **2,919** (PS196_b 1,801, WED040_a 461, LAL139 345, LAL184 309, PS196_a 3) | LAL184 -> EPG -> GLNO +626 | LAL184 -> EPG -> PEN_a -> GLNO +146,201; PS196_b -> LPsP -> PEN_a -> GLNO +145,284 | ExR2 -> GLNO [sign0] +6.1 (k 2); OA-VUMa1 -> LAL104 [sign0] +29.1 (k 3) |

GLNO's input (`b_inputs` of `paths_efference_to_GLNO.json`, 9,371 raw synapses, sign-0 share 8.0 %): PEN_a 2,397
(25.6 %, +146.6 mV per GLNO per PEN_a volley), PS196_b 1,801 (19.2 %, +16.7), PEN_b 1,099 (11.7 %, +75.6), EPG 766
(8.2 %, +52.7), WED040_a 461 (4.9 %, -31.7), GLNO 400 (4.3 %, sign 0), LAL139 345 (3.7 %, -16.4), LAL184 309 (3.3 %,
+15.5), LAL104 135, PEG 130, CB2037 102, ExR2 89 (sign 0) -- `cx_shift.md` section 1 reproduced by the tool. The link
weights of the two signed optic-yaw routes into GLNO (`links` table of `paths_optic_yaw_to_GLNO.json`): H2 -> PS047_b
+13.5 mV per PS047_b per H2 volley (778 syn, 3.5 % of PS047_b's input), HSS -> PS047_b +8.8 (79 syn), PS047_b ->
PS196_b +22.4 (445 syn, 7.4 % of PS196_b's input), PS196_b -> GLNO +16.7 (1,801 syn, 19.2 %); Nod1 -> WED153 +8.3
(182 syn), WED153 -> LAL139 +39.3 (296 syn, 7.5 %), LAL139 -> GLNO -16.4 (345 syn); Nod1 -> LAL184 +3.9 (28 syn),
LAL184 -> GLNO +15.5. The visual relay of the strongest signed k = 3 walk into PEN, PLP078 (2 cells), is fed by Nod1
430 syn (+33.3 mV per volley, 3.6 % of its input), LPT26 230 and Nod4 219, and reaches the ring territory through
ExR4 (37 syn, -4.4 mV per ExR4 volley, 0.18 % of ExR4's input), CB2037 (210 syn) and GLNO (3 syn, -0.2 mV).

**Structural reading.** No yaw carrier of any kind -- optic (HS / VS / H2 / Nod / LPT), descending (DNp20 / DNp15),
Johnston's organ, or the efference candidates -- makes a single direct synapse onto PEN; the signed routes into PEN
of any weight all pass through EPG, LPsP or the ring neurons (ExR4 / ExR6 / ER), i.e. through the ring itself, not
through a nodulus input. The one nodulus input of PEN, GLNO, is reached by the yaw carriers only at k = 3 and only
through the LAL / PS premotor cells (H2 / HSS -> PS047_b -> PS196_b -> GLNO; Nod1 -> WED153 / LAL184 / LAL139 ->
GLNO) -- the efference-copy territory `cx_shift.md` named -- and GLNO -> PEN is sign 0 at the end of every one of
those walks: the strongest silent link at k = 2 and k = 3 for every source group that reaches PEN is GLNO -> PEN
(+33.0 mV per PEN per GLNO volley if signed, 16,371 raw synapses). So structurally the candidate block is one link,
GLNO -> PEN, and the question the dynamics must answer is whether anything ever arrives at GLNO from the fly's
rotation (visual or efferent) that the link could carry if it were signed.

## 2. Dynamics: does anything from the fly's rotation ever reach GLNO? (two cluster batches, 40 runs)

### 2.1 The batches, and a fetch defect that the first analysis pass inherited

`out/rot_cf0c43/batch.sh` -> `python scripts/cluster_run.py --name rot --minutes 25 <20 commands> --fetch out/rot/`:
20 jobs (`record --condition {default, gaba} --mode {visual, efferent} --seed {0..4}`, shipped sign/abs receptor
default, gains 2/15, pooled series every 5 frames). The batch was submitted twice by the first pass of this task (a
`nohup` client was taken for dead when its buffered log stayed empty; its jobs ran on as `rot-7de91e`, the same code
and the same commands): **`20 job(s), 0 failed (57.5 min)`, run dir `/mnt/beegfs/neurome/runs/rot-cf0c43`** and
**`20 job(s), 0 failed (59.3 min)`, run dir `rot-7de91e`** (`out/rot_cluster.log` lines 2467 and 2470). Every run of
both batches realised `device cuda (NVIDIA B200)` (40/40 `_run.json`; 440-577 s wall each, the gaba jobs compiling
their own `out/cache_72164311/` from the raw files), 0 airborne frames in 160 phases.

Both clients fetched into the same `out/rot/` at the same time (file mtimes 18:32-18:51, one interleaved stream), so
`out/rot/` held files of both batches, per file whichever copy landed last: `verify-batch out/rot` (the `_run.json`,
the recording meta, the `_pen` meta and the console must agree on each phase's bump drift and heading, since one job
writes all four) finds **35 of 80 (run, phase) files inconsistent**, every console naming `rot-7de91e`
(`out/interp/apply_rotation/verify/verify_rot.csv`). The draft of this audit analysed that directory while the second
fetch was still writing into it: re-analysing the final mixed set reproduces its `default_visual`, `default_efferent`
and `gaba_efferent` statistics bit-for-bit but not `gaba_visual` (max |dz| 4.9 over the flip table), i.e. the draft's
gaba-visual numbers came from files that changed under it. Resolution: both run directories were copied again by
`scp` into named subdirectories, `out/rot_cf0c43/` and `out/rot_7de91e/` (520 files each), and `verify-batch` finds
**0 of 80 inconsistent in each**, consoles naming their own run dir (`verify_rot_cf0c43.csv`, `verify_rot_7de91e.csv`).
Every number below is from those two directories; the mixed set's analysis is kept under
`out/interp/apply_rotation/mixed/` and is not quoted. The lesson for the process rules: two clients must never
`--fetch` the same directory, and a fetched batch directory is verified (`verify-batch`) before it is analysed.

Two same-code batches with the same seeds are not the same draws: the GPU rollout is not reproducible at fixed seed
(`docs/INTERP.md` 2.4), and here that shows as qualitatively different ring histories -- gaba visual seed 4 has a bump
drifting -2.657 wedges / s through its first rest phase in `rot-cf0c43` and +0.005 in `rot-7de91e`; gaba efferent seed
2's second rest phase drifts -0.596 in `rot-7de91e` and +0.008 in `rot-cf0c43` (`report_bump.csv`, per-run columns).
So the replicate unit is the run, the pooled arms below have n = 10 (5 seeds x 2 batches), and the per-batch
analyses (n = 5 each) are the replication of every verdict.

Analyses: `analyse --runs "out/rot_cf0c43/<c>_<m>_r*" "out/rot_7de91e/<c>_<m>_r*" --out out/interp/apply_rotation/<c>_<m>`
x 4 (pooled) and `analyse --runs "out/rot_<id>/<c>_<m>_r*" --out .../batch_<id>/<c>_<m>` x 8 (CPU, 4-7 min each;
consoles `<c>_<m>_console.txt` beside each output directory), `report` (pooled and per batch),
`chain` (the per-phase L / R rates of the chain cells; pooled and per batch), and `scripts/interp_paths.py --recording
out/rot_cf0c43/default_efferent_r0_ccw.npz` x 4 (`paths_nf/`; the generator line is in each JSON's `files.generator`).

### 2.2 The stimulus is delivered in both modes; the bump does not move in either

Bump drift per phase, wedges / s (mean +- sd over 10 runs; ideal +-4.000 at 90 deg/s; `report_bump.csv` holds the ten
per-run values per cell), and the realised body heading rate:

| condition, mode | rest | ccw | rest2 | cw | heading ccw / cw (deg/s) |
|---|---|---|---|---|---|
| default, visual | -0.000 +- 0.005 | +0.002 +- 0.005 | -0.000 +- 0.005 | +0.004 +- 0.002 | +90.0 / -90.0 (imposed) |
| default, efferent | +0.000 +- 0.005 | +0.002 +- 0.005 | +0.000 +- 0.005 | +0.004 +- 0.001 | +101.8 +- 6.3 / -98.8 +- 9.6 |
| gaba, visual | -0.264 +- 0.841 (cf0c43 seed 4: -2.657) | -0.058 +- 0.196 (the same run: -0.615) | -0.001 +- 0.007 | -0.012 +- 0.055 (cf0c43 seed 1: -0.170) | +90.0 / -90.0 |
| gaba, efferent | +0.001 +- 0.009 | +0.005 +- 0.006 | -0.061 +- 0.188 (7de91e seed 2: -0.596) | -0.007 +- 0.059 (7de91e seed 1: -0.170; seed 2: +0.049) | +101.6 +- 6.2 / -98.7 +- 9.6 |

`bump_drift_vs_rest` (ccw and cw drifts against the twenty rest-phase drifts of the same condition, `summary.json`):
default visual z +0.50 / +0.96, p 0.25 / 0.015, `null` both; default efferent z +0.33 / +0.78, p 0.35 / 0.010,
`null`; gaba visual z +0.13 / +0.20, `null`; gaba efferent z +0.26 / +0.17, `null`. In the shipped-cache condition
all 80 phases (both modes, rest and rotating) lie within -0.006 and +0.010 wedges / s, and the small positive mean of
the cw phases (+0.004) is the same sign as the ccw phases' (+0.002): a drift that does not follow the rotation's
sign. In the GLNO = gaba condition 74 of 80 phases lie within +-0.017 and the six others are single-run jump events of
the signed-GLNO ring (`cx_glno.md` 4: the bump leaves its block once) -- two of them in rest phases, and among the
four rotating ones -0.615 (ccw) and +0.049 (cw) have the wrong sign for a heading-anchored bump, -0.170 (cw, twice)
the right one, so they are not sign-locked either. The realised self-turn in efferent mode is 86-113 deg/s per run
from DNa02 at 19.8 / 19.3 Hz on the stimulated side (<= 0.1 on the other; `chain_rates.csv`), i.e. the body module
turns on a DNa02 asymmetry as designed.

The sided flip statistic, (L - R at ccw) - (L - R at cw) against its null (L - R at rest2) - (L - R at rest), 10
runs, exact Mann-Whitney floor 1.1e-5 (`report_flip.csv`; flip in Hz, z on the null SD, verdict = |z| >= 3 and
p <= 0.05; in brackets the two per-batch verdicts, `batch_<id>/report_flip.csv`):

| type | default visual | default efferent | gaba visual | gaba efferent |
|---|---|---|---|---|
| HSN | -9.77 +- 0.29, z -15.9, result [r, r] | -9.89, z -14.4, result [r, r] | -10.03, z -17.1, result [r, r] | -9.63, z -13.2, result [r, r] |
| HSE | -4.91, z -10.4, result [r, r] | -4.96, z -13.2, result [r, r] | -4.89, z -12.4, result [r, r] | -5.19, z -16.9, result [r, r] |
| Nod1 | -8.71, z -16.0, result [r, r] | -9.96, z -7.8, result [r, r] | -8.04, z -12.0, result [r, r] | -9.89, z -7.0, result [r, r] |
| LPT26 | -7.11, z -17.5, result [r, r] | -6.27, z -3.8, result [r, r] | -6.77, z -17.9, result [r, r] | -6.54, z -7.7, result [r, r] |
| LPT50 | +5.63, z +22.0, result [r, r] | +5.37, z +8.4, result [r, r] | +6.10, z +11.3, result [r, r] | +4.96, z +6.7, result [r, r] |
| DNp20 | -13.63, z -3.9, result [r, r] | -14.66, z -4.0, result [r, r] | -12.36, z -8.5, result [r, r] | -14.67, z -4.5, result [r, r] |
| DNp15 | -4.47, z -8.5, result [r, r] | -4.53, z -9.2, result [r, r] | -4.39, z -9.5, result [r, r] | -4.36, z -9.3, result [r, r] |
| VS | -3.25, z -7.0, result [r, r] | -3.45, z -3.4, result [r, r] | -3.00, z -8.6, result [r, r] | -3.48, z -5.4, result [r, r] |
| DNa02 | -0.09, z -0.5, null | **+39.00**, z +332, result (the stimulus) | -0.04, z -1.1, null | +38.97, z +288, result |
| PLP078 (Nod1 -> PLP078 -> ExR4 / CB2037 / GLNO) | -2.49, z -3.1, result [n, r] | -4.51, z -3.3, result [r, n] | -3.64, z -4.3, result [r, r] | -4.76, z -2.7, null [n, n] |
| WED153 (Nod1 -> WED153 -> LAL139 -> GLNO) | +0.54, z +3.0, result [n, r] | +0.68, z +1.8, null [n, n] | +0.47, z +5.0, result [r, n] | +0.47, z +2.2, null [n, n] |
| LAL158 (Nod1 -> LAL158 -> GLNO) | +1.53, z +2.4, null | +1.43, z +0.5, null | +1.04, z +1.7, null | +1.40, z +0.9, null |
| LAL139 (-> GLNO 345 syn) | +0.56, z +1.1, null [n, n] | +0.16, z +0.3, null | +0.46, z +1.8, null [r, n] | -0.09, z -0.5, null |
| PS047_b (H2 / HSS -> PS047_b -> PS196_b) | -0.71, z -1.6, null | -1.09, z -2.2, null | +0.07, z +0.0, null | -1.14, z -1.5, null |
| PS196_b (-> GLNO 1,801 syn) | -0.03, z -0.1, null [n, n] | -0.03, z +0.3, null [n, n] | +0.13, z +1.0, null [n, n] | +0.01, z +1.0, null [n, n] |
| LAL184 / WED040_a / CB2037 / LPsP / SpsP / IbSpsP | 0.00-0.07, null | 0.00-0.01, null | 0.00-0.07, null | 0.00-0.13, null |
| ER1_a (-> PEN_a, the rotation-locked ring-neuron term of 2.4) | +0.51, z +1.9, null [n, n] | +0.42, z +0.5, null | +0.26, z +0.2, null [n, r] | +0.67, z +1.6, null |
| ExR4 / ExR6 | -0.09 / -0.06, null | -0.44 / -0.19, null | -2.37 / +0.51, null | +3.54 / -1.33, null |
| GLNO | -0.17, z -0.6, null [n, n] | -0.26, z -1.2, null [n, n] | +1.25, z +0.6, null [n, n] | -1.55, z -2.4, null [n, n] (7de91e seed 2 jump) |
| PEN_a / PEN_b | -0.04 / -0.03, null | -0.03 / -0.04, null | +0.08 / +0.30, null | -0.67 / -1.02, null |
| EPG / Delta7 / PEG | +0.30 / -0.05 / +0.09, null | +0.45 / -0.14 / +0.10, null | +0.89 / +1.85 / -0.71, null | -0.93 / -4.20 / +1.21 (PEG z +5.0, p 0.023 "result" [n, n]: flip sd 4.0 Hz, a jump-run artefact) |

The per-batch verdicts agree with the pooled ones on 73 of 76 named (arm, type) cells; the three disagreements are the
sub-Hz WED153 and LAL139 cells (z 2.1 vs 4.4, 7.1 vs 3.9, 4.7 vs 1.1). The `flip_results` sets (198-256 types per
pooled arm, `summary.json`) are the optic lobe's motion pathway (T4 / T5 / TmY / LLPC / LPC / LPi / Am1 / LC10 ...),
the HS / VS / Nod / LPT cells, DNp15 / DNp20, PLP078, and in efferent mode the VNC interneurons and leg motor
neurons downstream of DNa02. **No central-complex or nodulus type is in any of the four pooled sets** except PEG in
gaba efferent (above; absent from both per-batch sets, whose only central-complex entries are LNO2 in cf0c43 default
efferent and IbSpsP in 7de91e gaba efferent, each on one batch only).

Per-phase rates of the chain cells (`chain_rates.csv`, pooled 10 runs, L / R Hz; max over all cells and runs in
brackets): **PS196_b 0.03-0.16 / 0.00-0.16 in every phase of every condition [0.57]**; LAL184 0.00-0.01 / 0.09-0.40
[0.71]; WED040_a 0.00 / 0.00-0.01 [0.14]; CB2037 0.00-0.02 / 0.00 [0.43]; LPsP 0.00 / 0.00 [0.00]; AN07B037_a
0.00-0.01 / 0.00 [0.29]; PS099_a 0.01-0.07 / 0.01-0.16 [0.43]; PS262 1.3-1.7 / 1.1-1.6 unchanged by rotation [2.3];
PS047_b 0.3-0.6 / 0.6-1.9 -- its R cell rises from 0.6-0.8 at rest to 1.4 (visual) and 1.7-1.9 (efferent) in the ccw
phase [2.6]; LAL139 0.1-0.5 / 1.0-2.4, the R cell 1.0-1.8 at rest and 1.9-2.4 under rotation of either sign [3.0];
WED153 0.1-0.5 / 0.4-1.0, the L cells 0.1-0.5 at rest and 0.7-0.9 at ccw [3.1]; GLNO 146.3-146.9 / 118.6-119.1
(default) and 115.6-117.7 / 110.4-113.9 (gaba) in every phase; DNa02 19.8 / 0.1 and 0.0 / 19.3 in the efferent ccw /
cw phases, <= 0.1 otherwise; HSN 0.7-2.6 / 1.3-3.3 at rest, 1.5-1.7 / 4.7-5.3 at ccw, 6.3-7.1 / 0.2-0.4 at cw in both
modes. The efferent rest phases carry more optic-flow activity than the visual ones (DNp20 21-24 vs 6-11 Hz, LPT26
5.6-8.5 vs 0.1-1.2, Nod1 4.8-5.8 vs 1.1-1.5): the free heading jitters under the leg-MN noise (rest heading rate
+0.6 to +0.9 deg/s, sd 0.3-0.7), which is what the rest2-minus-rest null absorbs.

### 2.3 Trace: the rotation is carried everywhere in the eye and dies before the nodulus at PS196_b

`trace_{photoreceptors,yaw_cells}_{ccw,cw}.json` per pooled condition x mode (stat `best_cell` on `rate_hz` /
`optic_dr`, stimulus = ccw or cw, control = rest, null = rest2, 10 pairs; 11,387 types scored at >= 2 cells, depth
graph at 2 % input share). A full-field rotation is carried by 397-894 types (default visual ccw 869: depth 1 22,
depth 2 119, 3 200, 4 324, 5 191, 6 10, 7 2, 8 1; from the yaw cells: depth 0 10, 1 44, 2 73, 3 105, 4 259, 5 332,
6 45, 7 1), so the contract's "first lost depth" is 7-8 or None -- a whole-brain stimulus has a carrier at every depth
(as odour did, `interp_trace.md` 3) -- and the reading is the named chain. From the yaw cells (depth 0: Nod1 z +56.6
/ HSN +11.4 / DNp20 +12.0 result in default visual ccw): depth 1 **WED153 result in 6 of the 8 pooled traces** (z +8.1
/ +5.5 / +4.2 / +1.1 / +6.3 / +4.0 / +4.7 / +1.4; 12 of 16 per-batch traces), depth 2 LAL139 z -0.2 to +5.0 (result in
default visual ccw +3.5 and cw +5.0, null elsewhere), **PS047_b z +0.0 to +4.4 (result only in default efferent ccw,
+4.4; per batch +8.1 and +3.0 there, 2 of 16 traces overall), PS196_b z -0.4 to +0.6 null in 8 / 8 pooled and 16 / 16
per-batch traces, WED040_a null in 8 / 8**, depth 3 **GLNO z -0.4 to +0.5 null in 8 / 8 pooled and 16 / 16
per-batch**, PEN_a -0.3 to +2.5 null in 8 / 8, PEN_b -0.5 to +2.6 null, EPG -0.9 to +4.0 (the +4.0 is gaba efferent
cw, the jump batch's phase; null in the other 7), Delta7 -1.1 to +3.2 null, LAL184 null. So on the one signed route
from the eye into GLNO (H2 / HSS -> PS047_b -> PS196_b -> GLNO) the signal reaches PS047_b at ~1 Hz in the self-turn
and is lost at **PS196_b** (0.03-0.16 Hz, never a carrier); on the other (Nod1 -> WED153 -> LAL139 -> GLNO) it reaches
WED153 (+0.5 Hz, result) and LAL139 (+0.5 Hz at ccw, z 2-5) and is lost at **GLNO**.

The trace's `lost_inputs` table (now with the real raw counts, section 4) names what the two lost stages receive.
GLNO (share of its shaped input; mV per GLNO per presynaptic volley; raw synapses per GLNO cell; the presynaptic
type's own z / verdict in the default efferent ccw trace): PEN_a 0.32 / +146.6 / 599 / +2.0 null, PEN_b 0.17 / +75.6
/ 275 / +1.1 null, EPG 0.12 / +52.7 / 192 / +0.7 null, WED040_a 0.07 / -31.7 / 115 / 0.0 null, PS196_b 0.04 / +16.7 /
450 / +0.4 null, LAL139 0.04 / -16.4 / 86 / +0.9 null (+3.5 result in default visual), LAL184 0.03 / +15.5 / 77 / +0.8
null, LAL104, PEG, CB2037 (-0.3 null), WED011, PS060 (+4.2 result, -5.4 mV). PS196_b: PS239 0.08 / +38.9 / 156 / 0.0
null, AN07B037_a 0.07 / +33.4 / 210 / -0.2 null, PS292 0.06 / +29.2 / 121 / +1.7 null, ExR8 0.06 / +27.6 / 101 / +0.1
null, **PS047_b 0.04 / +22.4 / 223 / +4.4 result**, PS262 0.04 / +22.3 / 190 / -0.8 null, GNG411, LAL085, PS099_a
(350 syn, -0.3 to +0.9 null), PS048_a (240, +0.4 null), PS099_b (224, -0.2 null), CB3220 -- of PS196_b's twelve
largest inputs exactly one carries the turn, at +1 Hz on a 2-cell type, and PS196_b itself stays at 0.0-0.6 Hz.

### 2.4 Decompose: GLNO's and PEN's input during the turn is the ring's own activity, and what arrives from the eye is direction-blind

`decompose_{PEN,PEN_by_side,GLNO}.json` per pooled condition x mode (window 3-10 s of each phase, arms ccw / cw / rest
vs the null rest2, mV/s per post cell, `common.compare` per presynaptic type over 10 runs). GLNO, default (both modes
within 0.3 %): PEN_a +7,359 to +7,397, PEN_b +4,064 to +4,076, EPG +3,362 to +3,368, ER6 -879 to -880, PEG +417, ExR4
-199, ExR6 -159; E total +15,215 to +15,264, I -1,293 to -1,314; every PEN / EPG / PEG ccw and cw z between -0.4 and
+0.5; PS196_b, LAL184, WED040_a, CB2037 absent (0 mV/s). GLNO = gaba: PEN_a +6,277 to +6,381, EPG +3,149 to +3,289,
PEN_b +3,192 to +3,236, ER6 -820 to -829, PEG +406 to +419, GLNO self -187 to -191 (the signed self-loop), ExR4 -179
to -183; E +13,059 to +13,336, I -1,388 to -1,424; the only ring |z| >= 3 is EPG -3.7 in gaba visual ccw and PEG -3.0
in gaba efferent cw, the jump runs' phases. PEN_a, default: ExR6 -7,628 to -7,642, ExR4 -4,954 to -4,985, EPG +4,376
to +4,388, PEN_b +2,349 to +2,355, Delta7 -1,551 to -1,554, ER1_b -428 to -439, PEN_a +393 to +395; E +7,214 to
+7,225, I -15,239 to -15,278; GLNO 0 (sign 0). PEN_a, gaba: the same list plus **GLNO -3,739 to -3,821 mV/s** (the
33 mV per volley at 115 Hz, the largest inhibitory input after ExR6 / ExR4), z +1.5 / +0.3 (visual ccw / cw) and -0.3
/ +0.6 (efferent); E +6,398 to +6,469, I -17,781 to -18,038. The largest cancelling pair is EPG vs ExR6 (PEN_a) and
PEN_a vs ER6 (GLNO) in every arm of every condition.

What *does* change with rotation in these tables is small, and -- this is the finding the pooled-by-type decompose
adds to the flip table -- **direction-blind**. GLNO receives from LAL139 -10.4 mV/s at rest and -19.6 / -20.7 in the
ccw / cw phase (default visual; z -4.1 / -4.6 pooled, -3.5 / -4.1 and -4.8 / -5.1 in the two batches), from PLP078 -0.6
-> -3.4 / -3.0 (z -40 / -34; -49 / -42 and -38 / -32 per batch), from LAL158 +1.0 -> +2.4 / +2.3 (z +16 / +14), from
WED153 +0.02 -> +0.18 / +0.08, and in efferent mode the same terms at half the z (PLP078 -1.9 -> -3.7 / -3.0, z -9 /
-5; LAL139 -17 -> -22 / -17, z -1.5 / +0.3). PEN_a receives from ER1_a -197 mV/s at rest and -213 / -213 in ccw / cw
(z -10.0 / -9.8 pooled; -9.0 / -8.7 and -10.8 / -10.7 per batch; -204 -> -216 / -211 efferent, z -5.2 / -3.4;
`PEN_by_side`: the R-side ER1_a term -126 -> -138 / -140, the L-side -71 -> -75 / -73, both sides in both directions).
Each of these is the same for ccw and cw: a "the world is moving" signal, not an angular velocity. Their sum on GLNO
(-12 mV/s) is 0.09 % of its +13,900 mV/s net drive, ER1_a's rise on PEN_a (-16 mV/s) 0.1 % of its input; the one term
with a direction is Nod1's 3 synapses onto GLNO (+0.2 -> +0.4 ccw / +1.1 cw mV/s, z +5 / +20; 1e-4 of the net), and
the flip table confirms the sided reading -- PLP078's own L - R flips by -2.5 to -4.8 Hz (result in 3 of 4 arms)
while GLNO's L - R stays at +27.7 to +28.0 Hz in every phase (flip -0.17 / -0.26, null) and ExR4's, ExR6's and ER1_a's
flips are null in all four arms. So the visual yaw signal reaches the edge of the ring territory (PLP078 -> ExR4 37 syn /
CB2037 / GLNO 3 syn; Nod1 -> WED153 -> LAL139 -> GLNO; ER1_a) with its sign, and the cells that receive it there are
driven 99.9 % by the ring itself (GLNO: PEN_a + PEN_b + EPG = +14,800 of +15,200 mV/s E) or pool both sides
(ER1_a's L and R rise together), so nothing sided comes out. Nothing in PEN's or GLNO's input differs between rotating
and resting beyond the run scatter, in either GLNO condition, in either mode, except these direction-blind 0.1 % terms.

A tool note: on presynaptic groups whose null arm is numerically zero (ExR1, EPGt, LAL050, ... at 1e-3 mV/s) the
decompose z blows up (1e18-1e41 in `per_type`); those rows are the near-zero groups the `|z|>3` filter picks up and are
not read here. decompose needs a floor on the null SD (reported to the tool's owner; not this task's file).

### 2.5 Paths with the batch's never_firing flag

`paths_nf/{efference,optic_yaw}_to_GLNO.json`, `efference_to_PEN.json`, `ascending_to_GLNO.json` (`--recording
out/rot_cf0c43/default_efferent_r0_ccw.npz`: a link is `never_firing` when the presynaptic type's max rate over the
self-turn window is < 0.5 Hz; partial flags carry the share of cells). Efference -> GLNO: strongest silent link per k
= WED040_a -> GLNO (k 1, never_firing, -31.7 mV per GLNO volley, 461 syn), PS196_b -> ExR2 (k 2, +31.1, 899), PS196_b
-> LPsP (k 3, +32.8, 942); 30 of the 78 listed links never fire; GLNO's `b_inputs` flags PS196_b, WED040_a, LAL184,
CB2037, WED011, LNO1, PS292 `never_firing` and PEN_a / PEN_b / PEG / LAL139 partially (0.60-0.80 of their cells: the
PEN cells outside the bump), leaving EPG, LAL104, PS060 and the in-bump PEN as the signed-and-firing inputs. Optic yaw
-> GLNO: 14 of 69 links never fire; the strongest silent link at k 2 and 3 is WED040_a -> GLNO; the top silent
if-signed walks are Nod1 -> LAL184 -> GLNO (+59.6, LAL184 never firing), Nod4 / Nod1 -> CB2037 -> GLNO, H2 -> PS196_b
-> GLNO; the walks that fire end to end are Nod1 -> GLNO (3 syn, +0.2), Nod1 -> LAL139 -> GLNO (-49.7, LAL139 firing
in 0.4 of its cells), Nod1 -> LAL158 / PLP078 / LAL157 -> GLNO. Efference -> PEN: GLNO -> PEN (sign0, +33.0 if
signed) at k 2 and 3, then PS196_b -> LPsP (never_firing) and LAL184 -> EPG (never_firing); 18 of 61 links never fire.
Ascending neurons (1,846 cells) -> GLNO: strongest silent AN07B037_a -> PS196_b (k 2, never_firing on both links,
+33.4 mV per PS196_b volley, 419 syn) and PS239 -> PS196_b (k 3, +38.9, 312); 24 of 80 links never fire; the top
if-signed walks AN07B037_a / AN07B037_b -> PS196_b -> GLNO and AN08B026 -> LAL104 -> GLNO name the VNC-side inputs of
the chain: the ascending neurons that would carry a turn into PS196_b are themselves silent in the model.

## 3. Answers

**(1) The link whose state blocks the rotation signal, and the population behind it.** The link is **GLNO -> PEN**
(84 entries, 16,371 raw synapses, 19.4 % of PEN's input, fully contralateral, sign 0 because GLNO's transmitter is
`unknown`): the strongest silent link at k = 2 and k = 3 for every yaw source that reaches PEN at all (section 1),
while no yaw carrier -- HS / VS / H2 / Nod / LPT, DNp20 / DNp15, Johnston's organ, DNa02 / DNa01 / DNa03, PS196_b,
LAL139 / LAL184, WED040_a -- makes one synapse onto PEN directly. But the block is two layers deep, and the second
layer is what the round-1 audits could not see because the fly was teleported: **the population that would bring the
rotation into GLNO, PS196_b, never fires** (0.03-0.16 Hz, max cell 0.57 Hz, in all 160 phases of the two batches;
z -0.4 to +0.6, null in 8 / 8 pooled and 16 / 16 per-batch traces), and so do LAL184, WED040_a, CB2037, LPsP and the
ascending neurons that feed them (AN07B037_a, PS239 ...). The visual consequence of a self-turn gets exactly as far
as under the teleport: HSN / HSE / Nod1 / LPT26 / LPT50 / DNp20 / DNp15 / VS flip at +-3 to 15 Hz (|z| 3.4-22, `result`
in 32 of 32 pooled condition x mode x type cells and 64 of 64 per-batch ones), PLP078 flips -2.5 to -4.8 Hz, WED153
carries +0.5 Hz, PS047_b +1 Hz, LAL139 +0.5 Hz, and GLNO 0 (z -0.4 to +0.5, null in 24 / 24 traces; L - R fixed at
+27.7 to +28.0 Hz). GLNO's input is 97 % PEN / EPG (+14,800 of +15,200 mV/s E in the default, +12,600 of +13,200 with
GLNO = gaba): in this model GLNO is an efference copy of the *bump*, not of the *body*, which is why `cx_glno.md`
found its sign sets the bump's persistence window and `cx_shift.md` found a signed GLNO gives a one-sided elastic
lever and no integration -- here, with GLNO = gaba, its -3,740 to -3,820 mV/s onto PEN_a is the largest inhibitory term
after ExR6 / ExR4 and does not change by more than scatter (z -0.3 to +1.5) when the fly turns. What the eye does
deliver to the ring's doorstep -- LAL139 / PLP078 / LAL158 onto GLNO, ER1_a onto PEN -- is direction-blind and 0.1 %
of those cells' drive (2.4). Signing GLNO cannot supply a rotation input on its own; the signed signal the link would
carry is absent one synapse upstream (PS196_b) on the efferent route and arrives unsigned on the visual one. The bump
moved -0.000 +- 0.005 to +0.005 +- 0.006 wedges / s under a 90 deg/s visual turn and an 86-113 deg/s self-turn in 74 of
80 rotating phases (six single-run gaba jumps, not sign-locked; every shipped-cache phase within +-0.010) against
+-4.000 ideal.

**(2) What the connectome data say.** GLNO: MaleCNS `unknown` in every column (consensus / cell-type / per-body
`unclear`, conf 0.48); T-bar prediction over 3,132 T-bars glutamate 0.505 / ACh 0.373 / 5-HT 0.072; FlyWire Schlegel
2024 `top_nt` gaba x3 / glutamate x1 at confidence 0.30-0.33; no expression profile in any of the five sources
(`cx_glno.md` 1, `cx_shift.md` 1). Both EM calls map to sign -1 under `NT_SIGN`, neither clears the 0.5 confidence the
project accepts for a type-level override, so GLNO stays out of `TYPE_NT_OVERRIDE` -- and section 2 shows that the -1
both calls imply carries nothing rotation-locked (the `gaba` arms of both batches are the test: GLNO -> PEN_a -3,752
/ -3,797 vs -3,749 mV/s ccw / cw vs rest in visual mode, -3,809 / -3,739 vs -3,821 efferent). The transmitters of
the silent upstream cells are, by contrast, settled (PS196_b ACh 0.98 of 4,681 T-bars, FlyWire ACh 0.86-0.89; LAL184
ACh; LAL139 GABA; WED040_a glutamate, `cx_shift.md` 1): their problem is not a sign but their drive. PS196_b's inputs
are PS099_a 11.6 % / PS048_a 8.0 % / PS099_b 7.4 % / PS047_b 7.4 % / AN07B037_a 6.9 % / PS262 6.3 % (`cx_shift.md` 1;
2.3 above: 350 / 240 / 224 / 223 / 210 / 190 raw synapses per PS196_b cell) -- posterior-slope premotor and ascending
cells at 0-2 Hz here, of which only PS047_b carries the turn, at +1 Hz -- and the ascending neurons that reach it
(AN07B037_a 419 syn, PS239 312) are at 0 Hz: the model's VNC sends the brain no report of the turn it is making. The
visual relay that does carry a signed flip, PLP078 (Nod1 430 / LPT26 230 / Nod4 219 syn in; ExR4 37, CB2037 210, GLNO
3 syn out), is 0.18 % of ExR4's input and 0.03 % of GLNO's in MaleCNS: the connectome offers no visual route into the
ring of a weight that could move a 200 Hz bump without a gain. In the animal the LAL / PS / WED efference-copy
territory (Rayshubskiy 2020) and the Nod / LNO angular-velocity inputs (PFN inputs in MaleCNS, `cx_shift.md` 1) carry
self-motion from the VNC and the halteres / legs; in this model the body turns by a hand-written readout
(`body.Locomotion`) that sends nothing back, so the only self-motion signal the brain sees is the visual one, and
sections 2.3-2.4 show exactly where that one dies: PS196_b on the efferent-territory route, and the direction-blind
pooling at GLNO / ER1_a on the visual one.

**(3) A data-driven resolution.** Not a GLNO override (two low-confidence EM calls, and the `gaba` arms show it would
be moot), not a gain on PS196_b or PLP078 (tuning toward behaviour). The mechanism the connectome implies is the
missing **afferent / efference report of the turn**: (a) the ascending inputs of the PS196_b chain are now named
(`paths_nf/ascending_to_GLNO.json`: AN07B037_a / AN07B037_b -> PS196_b 419 / 52 syn, PS239 -> PS196_b 312, AN08B026
-> LAL104 269, plus AN04B003 -> PS047_a/b, AN06B009 -> LPsP, AN19B017 -> LAL139 / LAL104 from the draft's run of the
same tool); (b) what those ascending neurons receive in the VNC in MaleCNS (leg sensory, haltere, VNC intrinsic -- the
model's `vnc_sensory` superclass is never driven) is a `paths` question on the same cache; (c) drive them from the
body as a *sensor*, the way wind is fed to JO and sugar to the GRNs (`FlyBrain.wind` / `.taste`: a documented sensor
path with a physical variable, not a behavioural gain): leg-proprioceptive / haltere afferents fed from
`body.Locomotion`'s realised yaw and speed. Then repeat this audit's efferent arm (`record --mode efferent`, two
batches, `verify-batch`, `analyse`): if PS196_b / LAL / WED fire and GLNO's input gains a body-locked *sided* term
(the `PEN_by_side` / flip tables are the test, not the pooled decompose), the GLNO sign becomes testable on a signal,
and if the bump still does not move the ring's shape (`cx_wedge.md`: PEN one-step + ring loop; `cx_shift.md`: the
elastic lever) is the next suspect. Until (a)-(c) are done the model statement is: the compass has no rotation input
because the brain receives no report of the body's rotation other than through the eye, the eye's report dies at
PS196_b on the efference-copy route and arrives direction-blind on the LAL139 / PLP078 / ER1_a route, and the one
nodulus link into PEN, GLNO -> PEN, is silent.

## 4. Process notes

* The toolkit was used as built: `paths` (8 + 4 Results), `trace` (4 Results per condition x mode x {pooled, 2
  batches} = 48), `decompose` (3 per = 36; arms ccw / cw / rest vs rest2), `common.compare` for every difference
  (`flip_table`, `bump_drift_vs_rest`), `Recording` / `ArmAccumulator` for every recording. `Result.check()` is empty
  on every JSON (execution.device realised: cuda B200 for the batches, cpu for the structural runs).
* **Trace defect 1 fixed** (`flyverse/interp/trace.py`, `full_raw_counts`): `TypeGraph` and `trace()` called
  `common.raw_counts(c)` with `with_sign0=True`, which installs `connectome.sign0_counts` -- non-zero only on the
  explicit-zero entries -- as the whole data vector, so every `lost_inputs.raw_synapses_per_post` was 0.0. The
  helper now merges |W| with the sign-0 counts (as `paths.raw_counts` and `decompose.counts_matrix` do); on the
  real cache the GLNO row reads PEN_a 599.25 raw synapses per GLNO cell (= 2,397 / 4). The fix touches no statistic
  (the depth graph uses shaped weights): the three unaffected arms of the mixed set reproduce the draft's flip z and
  trace z bit-for-bit under the fixed tool. `TraceTests` (7) pass.
* The fetch defect of 2.1 (two clients fetching one directory) cost one analysis pass; the `verify-batch`
  subcommand is the check that would have caught it before analysis, and the two batches now serve as a replication.
* `selftest` (CPU, no connectome): a planted +5.9 Hz L - R flip is `result` at z +13 over 5 synthetic runs and its
  unflipped neighbour `null`; a graded unit's flip is read from `optic_dr`; a bump moving at +4.000 wedges / s across
  the 16-wedge wrap reads back +4.000 and a 90 deg/s heading +90.0.
* Validation block: default visual `reproduced` in the pooled set and both batches (drift within +-0.05 in every
  phase and the six reference types flip with `cx_shift.md` 3b's signs); gaba visual `not reproduced` in the pooled
  set and in `rot-cf0c43` (the seed-4 and seed-1 jumps), `reproduced` in `rot-7de91e` -- jump events of the
  signed-GLNO ring (`cx_glno.md` 4), reported as such, not a rotation response; the efferent arms have no reference
  (they are the open question) and are marked `not applicable`.
* `chain_rates.csv` is now generated by `chain` (the draft's table came from an inline script; the shipped
  generator agrees with it on every row that was not in the mixed `gaba efferent cw` phase).
* Defects of the first pass, kept for the record: (i) its first smoke of `record` ran on this desktop's GPU for 27 s
  against the cluster rule (the script set `CUDA_VISIBLE_DEVICES=""` after torch was imported and Windows ignores
  the empty value; fixed to `-1` before the import; no number in this document comes from that smoke); (ii) the
  double submission (2.1); (iii) `tests/test_interp.py` is not this task's file, so the tool's CPU coverage is
  `selftest` plus the smoke analyses rather than a test class (a `RotationApplyTests` class with `selftest`'s
  assertions and a `verify_batch` / `chain_rates` check on synthetic recordings is the addition for whoever owns the
  test file).
* `trace` needs `min_cells=2` for the yaw cells (HSN / HSE / H2 / LPT / DNp20 are 2-cell types); sub-Hz "results" on
  near-silent cells (DNa02 -0.17 Hz in the draft, IbSpsP -0.29 Hz in one batch here) are what the |z| rule admits
  when the null SD is 0.05-0.1 Hz and are not read as anything; the same goes for decompose's z on numerically-zero
  groups (2.4).

## 5. Files

`scripts/interp_apply_rotation.py`; `out/interp/apply_rotation/paths/` (8 Results, `paths_summary.json`,
`paths_console.txt`), `paths_nf/` (4 Results + consoles, never_firing from `out/rot_cf0c43/default_efferent_r0_ccw.npz`);
`out/rot_cf0c43/batch.sh`, `out/rot_cluster.log`, `out/rot_cf0c43/` and `out/rot_7de91e/` (20 runs x 4 phases x 3
recordings, 20 `_run.json`, 20 console `.txt` each), `out/interp/apply_rotation/verify/` (the three `verify-batch`
tables); `out/interp/apply_rotation/<cond>_<mode>/` x 4 pooled and `batch_<id>/<cond>_<mode>/` x 8 (`trace_*` x 4,
`decompose_*` x 3, `flip.csv`, `bump.csv`, `summary.json`, `summary.md`) + `<cond>_<mode>_console.txt` beside each;
`report_bump.csv`, `report_flip.csv`, `chain_rates.csv` (pooled and per batch); `mixed/` (the first pass on the
interleaved `out/rot/`, not used); `out/rot/` (the interleaved directory itself, kept as the evidence for 2.1);
`out/rot_smoke/` and `out/rot_smoke_cpu/` (the smokes; not used for any number above).
